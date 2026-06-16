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
    PLL_SOL as SOL,                                                    # clock.py verbatim
)
from . import regmap as _regmap  # GPR ABI names for arch-state decode

# assembler-verified blobs (sfpi riscv-tt-elf-as; see toolchain/README)
#   trampoline: clear trigger; csrsi mnstatus,8 (NMIE=1); fence.i; ld reset-vec; jr  -> re-armable
TRAMPOLINE = [0x200102B7, 0x4002AA23, 0x35346073, 0x0000100F, 0x0002B303, 0x00030067]
SPIN = [0x0000006F]              # j .


class Hang(RuntimeError):
    """Raised on any device error so callers stop instead of hammering a wedged NoC."""


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
        from ttexalens.tt_exalens_lib import write_words_to_device
        if not (0 <= addr <= PASS_HI):
            raise ValueError(f"addr 0x{addr:X} outside safe passthrough window")
        try:
            write_words_to_device(self.loc[tile], addr, list(words),
                                  context=self.ctx, noc_id=0, safe_mode=False)
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

    def peek(self, tile, addr, n=1):
        return self.rdn(tile, addr, n) if n > 1 else self.rd(tile, addr)

    def poke(self, tile, addr, val):
        self.wr(tile, addr, [val])
