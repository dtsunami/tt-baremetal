"""
bhtop.l2cpu — bring up and live-load code onto the Blackhole L2CPU (SiFive x280) harts.

Each Blackhole has four L2CPU tiles, each a cluster of four x280 RV64GC harts that
boot held in reset and can be released only ONCE (silicon bug; re-reset needs a full
ASIC reset via `tt-smi -r 0`). This module:

  * brings a tile's harts out of reset (ARC PLL glide + L2CPU_RESET flip, the
    tt-bh-linux sequence) with every hart parked in a spin and a self-re-arming
    RNMI "redirect" trampoline installed,
  * compiles asm / C / Rust to a flat binary (see toolchain.py) and loads it into
    the tile's DRAM, then redirects a chosen hart to it LIVE via an RNMI — no reset,
    repeatable, so you can iterate on hart code on the fly,
  * reads a simple telemetry block the hart code writes.

TRANSPORT (learned the hard way): L2CPU NoC access goes through tt-exalens (the proven
path); ARC (PLL + reset) goes through pyluwen `axi_*` (a separate transport that avoids
the NoC0-hang hazard of NoC-to-ARC). tt-exalens is initialized first. Every device op is
address-guarded, and a canary read precedes any write.
"""
import ctypes
import time

# All addresses/tiles live in ONE place: regmap.py (the canonical chip model the
# harness, docs and `map`/`regs` CLI also read). Import them so there's no second copy.
from .regmap import (                                                  # noqa: F401
    TILES, HARTS, PASS_HI, NIU_DANGER,
    PLL4_BASE, PLL_CNTL_1, PLL_CNTL_5, L2CPU_RESET, ARC_ALLOW,
    RESET_VEC, RNMI_TRAP, RNMI_EXC, TRIGGER, HART_STATUS, SPIN_ADDR,
    TRAMP_ADDR, CODE_ADDR, TELE_ADDR, TELE_SLOTS, TELE_STRIDE,
    ARCH_ADDR, ARCH_STRIDE, ARCH_MAGIC, ARCH_CSR_OFF,
    CMD_ADDR, CMD_STRIDE, CMD_SLOTS,
    VARCH_ADDR, VARCH_STRIDE, VARCH_VREGS, VARCH_MAGIC, VARCH_CSR_OFF,
    PLL_SOL as SOL,                                                    # clock.py verbatim
)
from . import regmap as _regmap  # GPR ABI names for arch-state decode

# assembler-verified blobs (sfpi riscv-tt-elf-as; see toolchain/README)
#   trampoline: clear trigger; csrsi mnstatus,8 (NMIE=1); fence.i; ld reset-vec; jr  -> re-armable
TRAMPOLINE = [0x200102B7, 0x4002AA23, 0x35346073, 0x0000100F, 0x0002B303, 0x00030067]
SPIN = [0x0000006F]              # j .


class Hang(RuntimeError):
    """Raised on any device error so callers stop instead of hammering a wedged NoC."""


def _decode_vtype(vtype):
    """Human-readable RVV vtype: SEW/LMUL/tail/mask (RVV 1.0 encoding)."""
    if vtype == 0:
        return "unset"
    lmul = {0: "m1", 1: "m2", 2: "m4", 3: "m8", 5: "mf8", 6: "mf4", 7: "mf2"}.get(vtype & 7, "?")
    sew = {0: "e8", 1: "e16", 2: "e32", 3: "e64"}.get((vtype >> 3) & 7, "?")
    ta = "ta" if (vtype >> 6) & 1 else "tu"
    ma = "ma" if (vtype >> 7) & 1 else "mu"
    vill = " VILL" if vtype >> 63 else ""
    return f"{sew},{lmul},{ta},{ma}{vill}"


# ---- PLL glide (ported verbatim from tt-bh-linux clock.py) ------------------
class _PLL5(ctypes.LittleEndianStructure):
    _fields_ = [("postdiv", 4 * ctypes.c_uint8)]

    def step(self, chip, target, field):
        one = max(min(target - self.postdiv[field], 1), -1)
        while self.postdiv[field] != target:
            self.postdiv[field] += one
            chip.axi_write(PLL4_BASE + PLL_CNTL_5, bytearray(self))
            time.sleep(1e-9)


class _PLL1(ctypes.LittleEndianStructure):
    _fields_ = [("refdiv", ctypes.c_uint8), ("postdiv", ctypes.c_uint8), ("fbdiv", ctypes.c_uint16)]

    def step_fbdiv(self, chip, target):
        one = max(min(target - self.fbdiv, 1), -1)
        while self.fbdiv != target:
            self.fbdiv += one
            chip.axi_write(PLL4_BASE + PLL_CNTL_1, bytearray(self))
            time.sleep(1e-9)


class L2cpu:
    def __init__(self, device=0, ctx=None):
        from ttexalens import init_ttexalens
        self.device = device
        # Reuse an existing tt-exalens context when given one (the web backend shares
        # its single DeviceManager ctx so the chip never sees a second device owner);
        # otherwise open our own (the CLI path).
        self.ctx = ctx or init_ttexalens()          # FIRST: sets up NoC access
        self.dev = self.ctx.devices[device]
        self._chip = None
        self.loc = {}
        for idx, (xy, _bit) in TILES.items():
            l = next((c for c in self.dev.get_block_locations("l2cpu") if tuple(c.to("noc0")) == xy), None)
            if l is not None:
                self.loc[idx] = l

    @property
    def chip(self):
        if self._chip is None:
            from pyluwen import PciChip
            self._chip = PciChip(pci_interface=self.device)
        return self._chip

    def hold_power(self, aiclk=False, mrisc=True, tensix=True, l2cpu=True, pcie=False):
        """Assert the KMD per-domain power-enable flags on THIS pyluwen fd so the named domains stay OUT of
        clock-gate for as long as this L2cpu (its PciChip fd) is alive. l2cpu=True -> TT_POWER_FLAG_L2CPU_ENABLE
        (1<<3, docstring '0=Clock Gate L2CPU'): the candidate fix for the idle-gate NoC/NIU wedge on the het
        pipeline, which drives the chip over exalens and otherwise never calls set_power, leaving the L2CPU
        domain free to gate during idle windows (render-wait, step boundary) -> racy gate-exit wedges the tile
        NIU (host read/DMA timeout). Default = the exact UMD 'busy' set MRISC_PHY_WAKEUP|L2CPU_ENABLE|
        TENSIX_ENABLE (pci_device.cpp:1066) — asserted TOGETHER because L2CPU needs the DRAM PHY (MRISC) awake;
        the lone l2cpu bit risks an invalid combination (-> EINVAL). The KMD OR-aggregates power_flags across
        all open fds, so this one fd holding the bits un-gates the whole chip; contribution drops when the fd
        closes -> fully reversible. Needs KMD >=2.6.0 (host is 2.8.0); best-effort — missing binding/old KMD
        just no-ops."""
        try:
            self.chip.set_power(aiclk=aiclk, mrisc=mrisc, tensix=tensix, l2cpu=l2cpu, pcie=pcie)
            return True
        except Exception as e:                                             # noqa: BLE001
            print(f"[l2cpu] hold_power failed (clock-gating NOT disabled): {type(e).__name__}: {e}", flush=True)
            return False

    # ---- L2CPU NoC (tt-exalens) ----
    def rd(self, tile, addr):
        from ttexalens.tt_exalens_lib import read_words_from_device
        if not (0 <= addr <= PASS_HI):
            raise ValueError(f"addr 0x{addr:X} outside safe passthrough window")
        try:
            return read_words_from_device(self.loc[tile], addr, word_count=1,
                                          context=self.ctx, noc_id=0, safe_mode=False)[0]
        except Exception as e:                       # noqa: BLE001
            raise Hang(f"read tile{tile} 0x{addr:X}: {type(e).__name__}: {e}") from e

    def rdn(self, tile, addr, n):
        from ttexalens.tt_exalens_lib import read_words_from_device
        if not (0 <= addr <= PASS_HI):
            raise ValueError(f"addr 0x{addr:X} outside safe passthrough window")
        try:
            return read_words_from_device(self.loc[tile], addr, word_count=n,
                                          context=self.ctx, noc_id=0, safe_mode=False)
        except Exception as e:                       # noqa: BLE001
            raise Hang(f"read tile{tile} 0x{addr:X} x{n}: {type(e).__name__}: {e}") from e

    def wr(self, tile, addr, words):
        # Fast bulk write: marshal to bytes with numpy (no per-element Python) + the DMA-capable
        # write_to_device(bytes) path. write_words_to_device does b"".join(x.to_bytes(4)...) over the int
        # list — ~150x slower on big buffers (20 MB gt: 462ms -> 3ms, measured + bit-identical on silicon).
        # Accepts a list of u32 words OR a numpy uint32 array; small writes (doorbells) auto-fall to the
        # register path inside write_to_device (below dma_threshold).
        from ttexalens.tt_exalens_lib import write_to_device
        import numpy as np
        if not (0 <= addr <= PASS_HI):
            raise ValueError(f"addr 0x{addr:X} outside safe passthrough window")
        buf = np.ascontiguousarray(words, dtype=np.uint32).tobytes()   # little-endian words (== x.to_bytes(4,"little"))
        try:
            write_to_device(self.loc[tile], addr, buf, context=self.ctx, noc_id=0, safe_mode=False)
        except Exception as e:                       # noqa: BLE001
            raise Hang(f"write tile{tile} 0x{addr:X}: {type(e).__name__}: {e}") from e

    # ---- ARC (pyluwen axi, allowlisted) ----
    def axi_rd(self, addr):
        if addr not in ARC_ALLOW:
            raise ValueError(f"ARC addr 0x{addr:08X} not allowlisted")
        try:
            return self.chip.axi_read32(addr)
        except Exception as e:                       # noqa: BLE001
            raise Hang(f"axi_read 0x{addr:08X}: {type(e).__name__}: {e}") from e

    def axi_wr(self, addr, val):
        if addr not in ARC_ALLOW:
            raise ValueError(f"ARC addr 0x{addr:08X} not allowlisted")
        try:
            self.chip.axi_write32(addr, val)
        except Exception as e:                       # noqa: BLE001
            raise Hang(f"axi_write 0x{addr:08X}: {type(e).__name__}: {e}") from e

    def clocks(self):
        """Read core vs uncore vs Tensix clocks (ARC telemetry; safe, read-only). The x280
        CORE clock is l2cpuclk0..3; the UNCORE/fabric clock is axiclk (this is the transport
        we talk to the chip over — see set_core_freq for why it's not settable here)."""
        t = self.chip.get_telemetry()
        def g(n):
            try:
                return getattr(t, n)
            except Exception:
                return None
        return {"core_l2cpu_mhz": [g(f"l2cpuclk{i}") for i in range(4)],
                "uncore_axi_mhz": g("axiclk"), "arc_mhz": g("arcclk"),
                "tensix_ai_mhz": g("aiclk"), "ddr_speed": g("ddr_speed"),
                "vcore_mv": g("vcore")}

    def set_core_freq(self, mhz):
        """Set the L2CPU CORE PLL — VERIFIED points only (200, 1750). Arbitrary PLL solutions
        are unverified on this card and a bad one can wedge the clock (hang), so we refuse them.
        The UNCORE/NoC clock is deliberately NOT settable here: it is the very transport we use
        to reach the tile, so changing it can cut us off. Glides via the proven bringup path."""
        if mhz not in SOL:
            raise ValueError(f"only verified L2CPU core PLL points {sorted(SOL)} are allowed "
                             "(arbitrary solutions are a hang risk); uncore/NoC clock is the "
                             "transport and is not settable here")
        self.set_pll(mhz)
        return {"ok": True, "mhz": mhz, "clocks": self.clocks()}

    def set_pll(self, mhz):
        fb, pds = SOL[mhz]
        b5 = bytearray(4); self.chip.axi_read(PLL4_BASE + PLL_CNTL_5, b5); p5 = _PLL5.from_buffer(b5)
        b1 = bytearray(4); self.chip.axi_read(PLL4_BASE + PLL_CNTL_1, b1); p1 = _PLL1.from_buffer(b1)
        for i, t in [(i, t) for i, t in enumerate(pds) if t > p5.postdiv[i]]:
            p5.step(self.chip, t, i)
        p1.step_fbdiv(self.chip, fb)
        for i, t in [(i, t) for i, t in enumerate(pds) if t < p5.postdiv[i]]:
            p5.step(self.chip, t, i)

    # ---- state ----
    def reset_state(self, tile):
        bit = TILES[tile][1]
        raw = self.axi_rd(L2CPU_RESET)
        return {"raw": raw, "released": bool(raw != 0xFFFFFFFF and raw & (1 << bit)),
                "wedged": raw == 0xFFFFFFFF, "bit": bit}

    def status(self, tile):
        return {"reset": self.reset_state(tile),
                "hart_status": self.rd(tile, HART_STATUS) & 0xFFFF,
                "reset_vec": [self.rd(tile, RESET_VEC + h * 8) for h in range(HARTS)]}

    # ---- bringup: release a tile with all harts parked + trampoline installed ----
    def bringup(self, tile):
        st = self.reset_state(tile)
        if st["wedged"]:
            raise Hang("ARC L2CPU_RESET reads 0xFFFFFFFF (wedge) — run tt-smi -r 0")
        if st["released"]:
            raise Hang(f"tile {tile} already released — run tt-smi -r 0 to start over")
        canary = self.rd(tile, RESET_VEC)
        if canary == 0xFFFFFFFF:
            raise Hang("canary read all-ones — NoC not reaching tile; run tt-smi -r 0")
        # stage trampoline + spin + handlers + parked reset vectors (harmless in reset)
        self._install_trampoline(tile)
        self.wr(tile, TELE_ADDR, [0] * (TELE_SLOTS * HARTS))     # all 4 per-hart windows
        for h in range(HARTS):
            self.wr(tile, RESET_VEC + h * 8, [SPIN_ADDR, 0])     # park
        # release: clock low, flip reset bit, clock high
        bit = TILES[tile][1]
        self.set_pll(200)
        v = self.axi_rd(L2CPU_RESET)
        if v == 0xFFFFFFFF or v & (1 << bit):
            raise Hang(f"L2CPU_RESET unexpected 0x{v:08X} — aborting before flip")
        self.axi_wr(L2CPU_RESET, v | (1 << bit))
        rb = self.axi_rd(L2CPU_RESET)
        self.set_pll(1750)
        return {"ok": bool(rb & (1 << bit)), "l2cpu_reset": rb}

    # ---- live load + redirect ----
    def _install_trampoline(self, tile):
        """(Re)stamp OUR self-re-arming trampoline + per-hart RNMI handlers. Idempotent,
        harmless DRAM writes (the same ones `bringup` does). `redirect` calls this every
        time so a live load never depends on which trampoline an earlier bringup left at
        TRAMP_ADDR: ours clears the trigger and re-enables mnstatus.NMIE, so the seize is
        repeatable. A foreign trampoline that skips the NMIE restore would mask all later
        RNMIs after the first redirect — the bug this guards against."""
        self.wr(tile, TRAMP_ADDR, TRAMPOLINE)
        self.wr(tile, SPIN_ADDR, SPIN)
        for h in range(HARTS):
            self.wr(tile, RNMI_TRAP + h * 16, [TRAMP_ADDR, 0])
            self.wr(tile, RNMI_EXC + h * 16, [SPIN_ADDR, 0])
        if self.rdn(tile, TRAMP_ADDR, len(TRAMPOLINE)) != list(TRAMPOLINE):
            raise Hang("trampoline did not read back — NoC not reaching tile; run tt-smi -r 0")

    def redirect(self, tile, hart, addr, verify=True, timeout=0.5):
        """Point hart's reset-vector at addr and pull its RNMI trigger -> trampoline
        re-arms (clears trigger, NMIE=1, fence.i) and jumps the hart to addr. Live.

        With verify=True (default) this confirms the hart actually took the RNMI: the
        trampoline clears the trigger bit on seize, so a bit that stays set means the
        seize never landed (the running code masked RNMIs, usually because an earlier
        non-package trampoline left NMIE=0) — we raise rather than report a false success."""
        self._install_trampoline(tile)
        self.wr(tile, RESET_VEC + hart * 8, [addr & 0xFFFFFFFF, (addr >> 32) & 0xFFFFFFFF])
        self.wr(tile, TRIGGER, [1 << hart])               # the seize (matches proven path)
        if not verify:
            return {"ok": True, "seized": None, "addr": addr}
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not (self.rd(tile, TRIGGER) & (1 << hart)):
                return {"ok": True, "seized": True, "addr": addr}
            time.sleep(0.005)
        raise Hang(
            f"hart {hart} did not take the RNMI (trigger bit still set after {timeout:g}s). "
            "It is likely running code left with mnstatus.NMIE=0 by a non-package trampoline "
            "(e.g. a standalone-script bringup), which masks further RNMIs. Recover with "
            "`tt-smi -r 0`, then `bringup` and `load` through this package.")

    def load(self, tile, hart, words, addr=CODE_ADDR, redirect=True):
        """Write a flat code image (list of u32 words) to DRAM and (optionally)
        redirect the hart to it. Requires the tile brought up."""
        self.wr(tile, addr, list(words))
        if self.rdn(tile, addr, len(words)) != list(words):
            raise Hang("code did not read back in DRAM (harvested? wrong addr)")
        seized = None
        if redirect:
            seized = self.redirect(tile, hart, addr).get("seized")
        return {"ok": True, "addr": addr, "words": len(words), "seized": seized}

    # ---- telemetry (per-hart windows: hart N at TELE_ADDR + N*TELE_STRIDE) ----
    def telemetry(self, tile, slots=16, hart=0):
        return self.rdn(tile, TELE_ADDR + hart * TELE_STRIDE, min(slots, TELE_SLOTS))

    def telemetry_all(self, tile):
        """All 4 harts' windows in one read -> {hart: [64 slots]}."""
        words = self.rdn(tile, TELE_ADDR, TELE_SLOTS * HARTS)
        return {h: words[h * TELE_SLOTS:(h + 1) * TELE_SLOTS] for h in range(HARTS)}

    # ---- arch state (bh_dump_state() snapshots the hart's registers to DRAM) ----
    def arch_state(self, tile, hart):
        """Read + decode a hart's register-file dump. Returns GPRs (x0..x31, hex strings)
        and CSRs; `valid` is True only if the magic marker is present (i.e. the kernel
        actually called bh_dump_state())."""
        base = ARCH_ADDR + hart * ARCH_STRIDE
        words = self.rdn(tile, base, ARCH_STRIDE // 4)        # 128 u32 = 512 bytes
        def u64(byte_off):
            i = byte_off // 4
            return (words[i + 1] << 32) | words[i]
        gpr = [u64(n * 8) for n in range(32)]
        csr = {name: u64(off) for name, off in ARCH_CSR_OFF.items()}
        valid = csr.get("magic", 0) == ARCH_MAGIC
        hx = lambda v: f"0x{v:016X}"
        return {"tile": tile, "hart": hart, "valid": valid,
                "gpr": [{"x": n, "abi": _regmap.GPR[n]["abi"], "val": hx(gpr[n])} for n in range(32)],
                "csr": {name: hx(val) for name, val in csr.items()}}

    # ---- command mailbox (host->hart doorbell; the hart polls + applies to a register) ----
    def command(self, tile, hart, op, arg0=0, arg1=0):
        """Ring hart N's DRAM doorbell so it updates a register live (cooperative, no RNMI).

        Writes the payload (op, args) FIRST, then bumps the seq word LAST, so the hart never
        sees a half-written command — it polls seq, and on a change reads a complete op/args.
        Requires a kernel that polls bh_cmd() (e.g. examples/mailbox.c). Returns the new seq;
        the kernel echoes the consumed seq into its telemetry so you can confirm it landed."""
        base = CMD_ADDR + hart * CMD_STRIDE
        self.wr(tile, base + 4, [op & 0xFFFFFFFF, arg0 & 0xFFFFFFFF, arg1 & 0xFFFFFFFF])
        seq = (self.rd(tile, base) + 1) & 0xFFFFFFFF
        self.wr(tile, base, [seq])
        return seq

    def mailbox(self, tile, hart):
        """Read back a hart's command window (debug): {seq, op, arg0, arg1}."""
        w = self.rdn(tile, CMD_ADDR + hart * CMD_STRIDE, 4)
        return {"seq": w[0], "op": w[1], "arg0": w[2], "arg1": w[3]}

    # ---- vector-register state (bh_dump_vec() snapshots the whole vector file to DRAM) ----
    def vec_state(self, tile, hart, ew=32):
        """Read + decode a hart's vector-register dump: 32 vregs (v0..v31) + vector CSRs.
        `valid` is True only if the magic marker is present (kernel called bh_dump_vec()).
        Each vreg is returned as a list of `ew`-bit element hex strings (ew in {8,16,32,64})
        plus the raw little-endian bytes, so the UI can show whatever SEW view it wants."""
        base = VARCH_ADDR + hart * VARCH_STRIDE
        words = self.rdn(tile, base, VARCH_STRIDE // 4)          # 0x900/4 = 576 u32
        vlenb = 64                                              # VLEN/8 on x280
        per = vlenb // 4                                        # u32 per vreg = 16
        def u64(byte_off):
            i = byte_off // 4
            return (words[i + 1] << 32) | words[i]
        csr = {name: u64(off) for name, off in VARCH_CSR_OFF.items()}
        valid = csr.get("magic", 0) == VARCH_MAGIC
        step = ew // 32 if ew >= 32 else 1
        vregs = []
        for n in range(VARCH_VREGS):
            w = words[n * per:(n + 1) * per]                    # 16 u32 = 64 bytes
            raw = b"".join(int(x).to_bytes(4, "little") for x in w)
            if ew == 64:
                elems = [f"0x{int.from_bytes(raw[i:i+8],'little'):016X}" for i in range(0, 64, 8)]
            elif ew == 16:
                elems = [f"0x{int.from_bytes(raw[i:i+2],'little'):04X}" for i in range(0, 64, 2)]
            elif ew == 8:
                elems = [f"0x{b:02X}" for b in raw]
            else:  # ew == 32
                elems = [f"0x{x:08X}" for x in w]
            vregs.append({"v": n, "e%d" % ew: elems})
        return {"tile": tile, "hart": hart, "valid": valid, "vlen": vlenb * 8, "ew": ew,
                "vregs": vregs, "csr": {k: f"0x{v:X}" for k, v in csr.items()},
                "vtype": _decode_vtype(csr.get("vtype", 0))}

    # ---- board power / clocks / temperature (ARC telemetry via pyluwen — safe transport) ----
    def power(self):
        """Live board power draw + clocks + temperature, read over the safe ARC path (same
        transport as PLL/reset, NOT NoC-to-ARC). Use it to correlate the vector virus with real
        watts: park vs run, and which instruction class draws the most. Fields that the FW does
        not populate come back as None."""
        t = self.chip.get_telemetry()
        def g(n, d=None):
            try:
                return getattr(t, n)
            except Exception:
                return d
        raw = g("asic_temperature")
        temp = round(raw / 65536.0, 1) if isinstance(raw, (int, float)) and raw > 4096 else raw
        return {
            "power_w": g("tdp"), "current_a": g("tdc"), "input_power_w": g("input_power"),
            "vcore_mv": g("vcore"), "aiclk_mhz": g("aiclk"),
            "l2cpuclk_mhz": [g(f"l2cpuclk{i}") for i in range(4)],
            "asic_temp_c": temp, "fan_rpm": g("fan_rpm"),
            "throttler": g("throttler"), "power_limit_w": g("board_power_limit"),
        }

    # ---- ARC DVFS / voltage margining (pyluwen arc_msg — the SAFE ARC transport, same as PLL/telemetry) ----
    # BH ARC message codes mirror tt-umd device/api/.../blackhole_arc.hpp (ArcMessageType). pyluwen/luwen wrap the
    # code with the 0xAA00 "valid message" prefix (matches tt-llk python_tests/helpers/device.py). vcore is NOT a
    # writable register on BH — it's CMFW/AVS-driven, auto-scaled to the AICLK perf-state. GO_BUSY is ttnn's lever
    # (=> ~810 mV @ 1350 MHz busy point); FORCE_VDD/FORCE_AICLK pin rail/clock past the aiclk_ppm governor. The
    # x280 CORE clock is a SEPARATE PLL (set_core_freq) that merely RIDES this shared rail. Voltage MUST lead
    # clock. Every send is code-allowlisted + value-clamped; run get_voltage() (a pure read) as a canary first.
    # Full RE + ladder + numbers: memory bh-arc-dvfs-voltage.
    ARC_PREFIX      = 0xAA00
    MSG_GET_VOLTAGE = 0x13
    MSG_GET_AICLK   = 0x34
    MSG_GO_BUSY     = 0x52          # AICLK_GO_BUSY
    MSG_GO_IDLE     = 0x54          # AICLK_GO_LONG_IDLE
    MSG_FORCE_AICLK = 0x33
    MSG_FORCE_VDD   = 0x39
    _ARC_MSG_ALLOW  = {0x13, 0x34, 0x52, 0x54, 0x33, 0x39}   # DVFS only — NEVER reset/flash/i2c/switch-vout codes
    SAFE_VCORE_MV   = 950          # hard ceiling regardless of vdd_max (convex top-end budget for x280@1750)
    SAFE_VCORE_STEP = 40           # max mV change per force_vdd call — force small steps
    EXPLORE_VCORE_MV = 1050        # allow_over ceiling: modest over-volt for freq exploration (<< ~1.3V process max)
    EXPLORE_FBDIV_MAX = 210        # set_fbdiv_explore cap: fbdiv 140=1750 MHz -> 210 ~ 2.6 GHz (well past fail)

    def arc_msg(self, code, arg0=0, arg1=0, wait=True, timeout=1.0, _raw=False):
        """Send ONE allowlisted ARC DVFS message over the safe pyluwen transport (NOT NoC-to-ARC). `code` = the
        raw BH ArcMessageType (e.g. 0x52); the 0xAA00 prefix is applied unless _raw. Returns the pyluwen reply."""
        if code not in self._ARC_MSG_ALLOW:
            raise ValueError(f"ARC msg 0x{code:02X} not in DVFS allowlist {sorted(hex(c) for c in self._ARC_MSG_ALLOW)}")
        msg = code if _raw else (self.ARC_PREFIX | code)
        try:
            return self.chip.arc_msg(msg, wait_for_done=wait, arg0=arg0, arg1=arg1, timeout=timeout)
        except Exception as e:                       # noqa: BLE001
            raise Hang(f"arc_msg 0x{msg:04X}(arg0={arg0},arg1={arg1}): {type(e).__name__}: {e}") from e

    def limits(self):
        """Firmware safety ceilings (live telemetry, read-only): the vcore window + AICLK/current/power/thermal
        clamps. Read BEFORE margining — never exceed vdd_max / aiclk_fmax (CMFW clamps FORCE_VDD to vdd_max)."""
        t = self.chip.get_telemetry()
        def g(n):
            try: return getattr(t, n)
            except Exception: return None
        vl = g("vdd_limits") or 0
        return {"vdd_min_mv": vl & 0xFFFF, "vdd_max_mv": (vl >> 16) & 0xFFFF,
                "aiclk_fmax_mhz": g("aiclk_limit_max"), "tdc_max_a": g("tdc_limit_max"),
                "tdp_max_w": g("tdp_limit_max"), "thm_throttle_c": g("thm_limit_throttle"),
                "board_power_limit_w": g("board_power_limit")}

    def monitor(self):
        """One-shot safety snapshot + a `safe` verdict vs abort thresholds. Poll BETWEEN margining steps; if not
        safe, back off / perf_idle() / reset. `heartbeat` must ADVANCE across calls (frozen => wedged)."""
        p = self.power(); lim = self.limits(); t = self.chip.get_telemetry()
        def g(n):
            try: return getattr(t, n)
            except Exception: return None
        temp, thr = p.get("asic_temp_c"), p.get("throttler")
        vmax, vc = lim.get("vdd_max_mv"), p.get("vcore_mv")
        alarms = []
        if isinstance(temp, (int, float)) and temp > 85: alarms.append(f"temp {temp}C>85")
        if thr not in (0, None): alarms.append(f"throttler={thr}")
        if vmax and vc and vc > vmax: alarms.append(f"vcore {vc}>vdd_max {vmax}")
        return {**p, **lim, "vreg_temp_c": g("vreg_temperature"),
                "heartbeat": g("timer_heartbeat"), "safe": not alarms, "alarms": alarms}

    def get_voltage(self):
        """CANARY + read: send GET_VOLTAGE (a pure read, no state change) and cross-check the reply against
        telemetry vcore. Run this FIRST to confirm the arc_msg path/prefix work on THIS card without risk."""
        reply = self.arc_msg(self.MSG_GET_VOLTAGE)
        return {"reply": reply, "telemetry_vcore_mv": self.chip.get_telemetry().vcore}

    def perf_busy(self):
        """Rung 1 (ttnn-safe): AICLK_GO_BUSY — request the busy perf-state so ARC/AVS raises the SHARED vcore to
        the ~810 mV / 1350 MHz point. Gives the x280 PLL headroom; then set_core_freq(1750) rides it. LATCHED (no
        auto-revert), but the aiclk_ppm governor may still droop it mid-run if it reads the load as idle."""
        before = self.power().get("vcore_mv")
        r = self.arc_msg(self.MSG_GO_BUSY, 0, 0)
        return {"ok": True, "msg": "GO_BUSY", "reply": r, "vcore_before_mv": before}

    def perf_idle(self):
        """Restore the idle perf-state (AICLK_GO_LONG_IDLE): drops the shared rail back to ~711 mV. Call to unwind
        a margining experiment or before leaving the card."""
        r = self.arc_msg(self.MSG_GO_IDLE, 0, 0)
        return {"ok": True, "msg": "GO_LONG_IDLE", "reply": r}

    def force_aiclk(self, mhz):
        """Rung 2 (governor-defeating): FORCE_AICLK — pin AICLK at `mhz`, CMFW applies its own matched vcore.
        Clamped to the live aiclk_fmax. Arg-unit (MHz) is the documented intent but CLOSED CMFW — verify the
        actual effect via monitor() before trusting it."""
        fmax = self.limits().get("aiclk_fmax_mhz")
        if fmax and mhz > fmax:
            raise ValueError(f"aiclk {mhz} > firmware fmax {fmax}; refuse (would be clamped/hang)")
        r = self.arc_msg(self.MSG_FORCE_AICLK, arg0=int(mhz))
        return {"ok": True, "msg": "FORCE_AICLK", "mhz": mhz, "reply": r}

    def force_vdd(self, mv, allow_step=False, allow_over=False):
        """Rung 2: FORCE_VDD — pin the shared vcore at `mv`. VERIFIED on silicon: arg unit = mV, ~10 mV regulator
        granularity, CMFW clamps to live vdd_max (900). Guards: refuses mv > SAFE_VCORE_MV (or EXPLORE_VCORE_MV
        when allow_over), mv > live vdd_max (unless allow_over — to probe/exceed the firmware clamp), or a jump >
        SAFE_VCORE_STEP (unless allow_step). Voltage must lead clock; monitor() after every step."""
        lim = self.limits(); vmax = lim.get("vdd_max_mv") or self.SAFE_VCORE_MV
        cur = self.power().get("vcore_mv") or 0
        cap = self.EXPLORE_VCORE_MV if allow_over else self.SAFE_VCORE_MV
        if mv > cap: raise ValueError(f"{mv} mV > ceiling {cap} (allow_over raises to {self.EXPLORE_VCORE_MV})")
        if not allow_over and vmax and mv > vmax:
            raise ValueError(f"{mv} mV > live vdd_max {vmax} (CMFW would clamp; allow_over to probe past it)")
        if not allow_step and cur and abs(mv - cur) > self.SAFE_VCORE_STEP:
            raise ValueError(f"jump {cur}->{mv} mV exceeds {self.SAFE_VCORE_STEP} mV/step; step up gradually "
                             "(allow_step=True overrides)")
        r = self.arc_msg(self.MSG_FORCE_VDD, arg0=int(mv))
        return {"ok": True, "msg": "FORCE_VDD", "mv": mv, "vcore_before_mv": cur, "vdd_max_mv": vmax, "reply": r}

    def set_fbdiv_explore(self, fb):
        """EXPERIMENTAL frequency overclock: glide the L2CPU core-PLL fbdiv to `fb` (per-lane postdivs left as-is)
        from the CURRENT state. Call set_core_freq(1750) first so postdiv=[1,1,1,1] (fbdiv 140 = 1750 MHz, so
        each +1 fbdiv ~= +12.5 MHz). Bounded to EXPLORE_FBDIV_MAX; a too-high fbdiv can unlock the PLL / wedge the
        clock -> recover with tt-smi -r 0. Returns the telemetry-MEASURED core MHz (empirical, no PLL model)."""
        if not (120 <= fb <= self.EXPLORE_FBDIV_MAX):
            raise ValueError(f"fbdiv {fb} outside [120,{self.EXPLORE_FBDIV_MAX}] — refuse (unlock/hang risk)")
        b1 = bytearray(4); self.chip.axi_read(PLL4_BASE + PLL_CNTL_1, b1); p1 = _PLL1.from_buffer(b1)
        p1.step_fbdiv(self.chip, fb)                       # glide +-1/step (same as set_pll)
        time.sleep(0.05)
        return {"fbdiv": fb, "core_mhz": self.clocks()["core_l2cpu_mhz"]}

    def peek(self, tile, addr, n=1):
        return self.rdn(tile, addr, n) if n > 1 else self.rd(tile, addr)

    def poke(self, tile, addr, val):
        self.wr(tile, addr, [val])
