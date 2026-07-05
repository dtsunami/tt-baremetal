"""
tensix.bootloader — drive the RESIDENT bootloader kernel over the NoC (tt-exalens): read its
control mailbox, poke live params, STAGE a code overlay into an L1 slot, and ring the doorbell to
hot-swap code — no metal relaunch, no reset. The live console for the resident-program path
([[tensix-bootloader]]); sibling of the x280 cmd-mailbox ([[l2cpu-cmd-mailbox]]).

Python twin of `bootloader_abi.h` (keep in sync). The contract lives at:
  ~/tt-metal/tt_metal/programming_examples/contributed/bootloader/  (host launcher + resident kernel)

MODEL: tt-metal multicasts the bootloader to every worker + parks (no Finish/close → no reset). The
kernel never returns: it loops polling a FIXED L1 control mailbox at CTRL_BASE. From then on these
calls own the grid live — params are pokeable each loop iteration (no re-go), and new machine code is
staged into a slot + invoked by ringing EXEC (the kernel invalidates i$ then calls slot base).

All I/O rides TensixLauncher.rd/wr, so it stays on the SAFE Tensix-L1-over-NoC surface (not the
ARC/PCIe/L2CPU hang hazard — [[bh-noc-hang-hazard]]).
"""
import struct
import time

# ---- per-RISC L1 carve-out (MIRROR bootloader_abi.h — re-check after editing the header) ----
# A Tensix tile has 5 baby RISCs sharing ONE L1; each owns a disjoint 64 KiB region. From the
# NoC side all 5 live at the SAME core coord — per-RISC is purely an L1 OFFSET (the region base).
NUM_RISCS = 5
REGION_BASE = 0x00100000     # region 0 (BRISC) base
REGION_STRIDE = 0x00010000   # 64 KiB per RISC
CTRL_OFF = 0x0000            # mailbox header
TELEM_OFF = 0x1000           # 4 KiB scratch the kernel/overlay publishes
SLOT_A_OFF = 0x2000          # 24 KiB overlay slot A
SLOT_B_OFF = 0x8000          # 32 KiB overlay slot B (double-buffer)
SLOT_A_SIZE = 0x6000
SLOT_B_SIZE = 0x8000

# RISC index -> name (matches BlRisc in the ABI; index is what each kernel stamps in RISC_ID).
RISC_NAMES = {0: "BRISC", 1: "NCRISC", 2: "TRISC0", 3: "TRISC1", 4: "TRISC2"}
RISC_INDEX = {v: k for k, v in RISC_NAMES.items()}


def risc_idx(risc):
    """Normalize a RISC selector to an int index. Accepts an int, a numeric string ('0'..'4'),
    or a RISC name like 'TRISC0'/'brisc'."""
    if isinstance(risc, str):
        s = risc.strip()
        i = int(s, 0) if s.lstrip("+-").isdigit() else RISC_INDEX[s.upper()]
    else:
        i = int(risc)
    if not 0 <= i < NUM_RISCS:
        raise ValueError(f"risc index {i} out of range 0..{NUM_RISCS - 1}")
    return i


def region_base(risc):
    """L1 byte base of RISC `risc`'s region."""
    return REGION_BASE + risc_idx(risc) * REGION_STRIDE


# Back-compat single-region constants (BRISC = region 0).
CTRL_BASE = REGION_BASE + CTRL_OFF        # 0x100000
TELEM_BASE = REGION_BASE + TELEM_OFF      # 0x101000
CODE_SLOT_A = REGION_BASE + SLOT_A_OFF    # 0x102000
CODE_SLOT_B = REGION_BASE + SLOT_B_OFF    # 0x108000
CODE_SLOT_SIZE = SLOT_A_SIZE
SLOTS = {"A": CODE_SLOT_A, "B": CODE_SLOT_B}

# word indices off a region's CTRL base
DOORBELL, ARG0, ARG1 = 0, 1, 2
PARAM0 = 4                   # PARAM0..PARAM3 = words 4..7
HEARTBEAT, LAST_CMD, STATUS, OVL_RET, WALLCLK, RISC_ID = 16, 17, 18, 19, 20, 21
CTRL_WORDS = 22             # how many words `status()` reads

# commands (host -> kernel via DOORBELL)
CMD_NONE = 0x00000000
CMD_SETPARAM = 0x00000001
CMD_EXEC = 0x00000002
CMD_HALT = 0x0000DEAD
CMD_NAME = {CMD_NONE: "NONE", CMD_SETPARAM: "SETPARAM", CMD_EXEC: "EXEC", CMD_HALT: "HALT"}

STATUS_NAME = {0: "BOOT", 1: "IDLE", 2: "OVERLAY", 3: "HALTED"}


def ctrl_addr(word):
    """L1 byte address of control word `word`."""
    return CTRL_BASE + word * 4


class Bootloader:
    """Wrap a TensixLauncher to drive the resident bootloader on ONE of a core's 5 RISCs.

    `launcher` is a tensix.loader.TensixLauncher (shares the DeviceManager ctx). `risc` selects
    which of the 5 baby RISCs (0..4 or a name like 'TRISC0'); all addresses are computed off that
    RISC's region base, so the same NoC core hosts 5 independent mailboxes. Compose, don't
    subclass — the loader rides metal's launch protocol; this rides our own L1 mailbox on top."""

    def __init__(self, launcher, risc=0):
        self.L = launcher
        self.risc = risc_idx(risc)
        self.risc_name = RISC_NAMES[self.risc]
        self.region = region_base(self.risc)
        self.slots = {"A": self.region + SLOT_A_OFF, "B": self.region + SLOT_B_OFF}
        self.telem = self.region + TELEM_OFF

    # ---- address helpers (off THIS RISC's region) ------------------------------------
    def ctrl_addr(self, word):
        """L1 byte address of control word `word` in this RISC's region."""
        return self.region + CTRL_OFF + word * 4

    # ---- read state ------------------------------------------------------------------
    def status(self):
        """One block read of the control mailbox -> decoded dict."""
        w = self.L.rd(self.region + CTRL_OFF, CTRL_WORDS)
        return {
            "risc": self.risc, "risc_name": self.risc_name, "region": self.region,
            "doorbell": w[DOORBELL], "doorbell_name": CMD_NAME.get(w[DOORBELL], hex(w[DOORBELL])),
            "arg0": w[ARG0], "arg1": w[ARG1],
            "params": list(w[PARAM0:PARAM0 + 4]),
            "heartbeat": w[HEARTBEAT],
            "last_cmd": w[LAST_CMD], "last_cmd_name": CMD_NAME.get(w[LAST_CMD], hex(w[LAST_CMD])),
            "status": w[STATUS], "status_name": STATUS_NAME.get(w[STATUS], hex(w[STATUS])),
            "ovl_ret": w[OVL_RET],
            "wallclk": w[WALLCLK],          # on-core wall-clock stamp at last heartbeat publish
            "risc_id": w[RISC_ID],          # RISC index the kernel stamped (should == self.risc)
        }

    def alive(self, dt=0.05):
        """Liveness: is the heartbeat advancing? (two reads dt apart)."""
        a = self.L.rd(self.ctrl_addr(HEARTBEAT), 1)[0]
        time.sleep(dt)
        b = self.L.rd(self.ctrl_addr(HEARTBEAT), 1)[0]
        return {"heartbeat": b, "advancing": b != a, "delta": (b - a) & 0xFFFFFFFF}

    # ---- live params (no compile, no re-go) ------------------------------------------
    def set_param(self, i, val):
        """Poke PARAM[i]. The resident loop / overlay reads it on its next iteration."""
        addr = self.ctrl_addr(PARAM0 + i)
        self.L.wr(addr, [val & 0xFFFFFFFF])
        return addr

    def set_params(self, values, start=0):
        self.L.wr(self.ctrl_addr(PARAM0 + start), [v & 0xFFFFFFFF for v in values])

    # ---- code overlay (the "load new kernel" path) -----------------------------------
    def stage(self, data, slot="A", chunk_words=1024):
        """Write an overlay .bin (raw bytes from objcopy) into a code slot. NoC block write,
        chunked. Guards against slot overflow. Stage a slot that ISN'T currently executing.
        NOTE: overlays must be LINKED at this RISC's slot address (per-region slots differ)."""
        base = self.slots[slot.upper()]
        cap = SLOT_A_SIZE if slot.upper() == "A" else SLOT_B_SIZE
        if len(data) > cap:
            raise ValueError(f"overlay {len(data)}B exceeds slot {slot.upper()} ({cap}B)")
        words = _bytes_to_words(data)
        for i in range(0, len(words), chunk_words):
            self.L.wr(base + i * 4, words[i:i + chunk_words])
        return {"slot": slot.upper(), "addr": base, "bytes": len(data), "words": len(words)}

    # Overlay EXEC = jump the RISC's PC into the L1 code slot. VALIDATED on BRISC only. NCRISC and
    # the TRISCs fetch instructions from their own local instruction memory, NOT arbitrary L1 — so
    # jumping their PC to an L1 slot address HANGS them (confirmed on device: NCRISC froze on EXEC).
    # The resident loop + params + halt are fine on all 5; only the code-jump is BRISC-only.
    #
    # COROLLARY (2026-06): compute-engine kernels (matrix/FPU, SFPU/vector) are categorically NOT
    # BRISC-overlay material. Even though only BRISC can EXEC an L1 overlay, BRISC drives the T0
    # (UNPACK) instruction FIFO; MVMUL/SFPMAD belong on T1 (MATH) and stall on cross-thread
    # SrcA/B-dvalid / DEST-sync that a single BRISC stream can't produce. Such kernels must boot as
    # real TRISC LLK-lane programs (LLK_BOOT_MODE_TRISC via llk_run.py), not hot-swapped L1 overlays.
    # See kernels/tensix/overlays/{matrix,sfpu}.c for the full root-cause writeup.
    EXEC_SAFE_RISCS = {0}   # BRISC

    def exec(self, slot="A", force=False):
        """Point ARG0 at the slot and ring EXEC -> kernel invalidates i$ and calls slot base.
        Refuses on NCRISC/TRISC (they wedge — see EXEC_SAFE_RISCS) unless force=True."""
        if self.risc not in self.EXEC_SAFE_RISCS and not force:
            raise ValueError(
                f"overlay EXEC on {self.risc_name} hangs the RISC (can't fetch code from L1 slot); "
                f"only BRISC is exec-safe. Pass force=True to override (will likely wedge the core).")
        base = self.slots[slot.upper()]
        self.L.wr(self.ctrl_addr(ARG0), [base])
        self.L.wr(self.ctrl_addr(DOORBELL), [CMD_EXEC])
        return {"slot": slot.upper(), "addr": base}

    def halt(self):
        """Break the resident loop -> kernel returns -> host may close() cleanly."""
        self.L.wr(self.ctrl_addr(DOORBELL), [CMD_HALT])

    def ring(self, cmd, arg0=None):
        """Generic doorbell ring (escape hatch)."""
        if arg0 is not None:
            self.L.wr(self.ctrl_addr(ARG0), [arg0 & 0xFFFFFFFF])
        self.L.wr(self.ctrl_addr(DOORBELL), [cmd & 0xFFFFFFFF])

    # ---- self-reset escape hatch (recover a wedged baby RISC WITHOUT tt-smi / metal re-init) -----
    # A wedged baby RISC (e.g. NCRISC/TRISC after a bad EXEC) normally forces a full ASIC reset
    # (tt-smi -r 0) + metal re-launch, AND blocks the next metal launch ("Timeout waiting for
    # cores"). The Tensix per-RISC SOFT_RESET (debug reg 0xFFB121B0) lets us soft-reset JUST that
    # RISC over the NoC — the Tensix analog of the x280 RNMI seize. Driven through tt-exalens'
    # BabyRiscDebug (the documented, NoC-safe debug-bus path; metal/exalens use the same regs).
    EXALENS_RISC_NAME = {0: "brisc", 1: "ncrisc", 2: "trisc0", 3: "trisc1", 4: "trisc2"}

    def _risc_debug(self):
        """tt-exalens BabyRiscDebug for THIS RISC (soft-reset / reset-PC / halt over NoC)."""
        dev = self.L.ctx.devices[self.L.device_id]
        block = dev.get_block(self.L.coord)
        return block.get_risc_debug(self.EXALENS_RISC_NAME[self.risc])

    def in_reset(self):
        """Is this RISC currently held in soft reset?"""
        return bool(self._risc_debug().is_in_reset())

    def soft_reset(self, revive=True, entry=None, settle=0.05):
        """Soft-reset this baby RISC over the NoC — recover a wedge with no ASIC reset / relaunch.
          revive=False : assert reset and LEAVE it held (clean halt) -> unblocks the next metal
                         launch without tt-smi -r 0.
          revive=True  : assert -> (optionally re-vector reset-PC to `entry`) -> deassert, so the
                         RISC restarts. With entry=None it restarts from its existing reset vector
                         (metal firmware -> re-runs the resident kernel). BRISC has no reset-PC
                         override, so `entry` is honored only for NCRISC/TRISC.
        Returns a dict with before/after reset state. Tensix debug regs are the SAFE NoC surface
        (not the ARC/PCIe/L2CPU hang hazard)."""
        rd = self._risc_debug()
        was = bool(rd.is_in_reset())
        rd.set_reset_signal(True)                       # assert -> halt the wedged core
        if revive:
            if entry is not None and self.risc != 0:    # BRISC boots from hardwired L1 0x0
                rd.set_code_start_address(int(entry))
            time.sleep(settle)
            rd.set_reset_signal(False)                  # deassert -> restart
        now = bool(rd.is_in_reset())
        return {"risc": self.risc, "risc_name": self.risc_name, "was_in_reset": was,
                "revived": revive, "in_reset": now,
                "entry": (hex(entry) if entry is not None else None)}

    def wait_ack(self, timeout=2.0, poll=0.01):
        """Poll until the kernel acks (writes DOORBELL back to NONE). True on ack, False on timeout."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.L.rd(self.ctrl_addr(DOORBELL), 1)[0] == CMD_NONE:
                return True
            time.sleep(poll)
        return False


def _bytes_to_words(b):
    if len(b) % 4:
        b = b + b"\x00" * (4 - len(b) % 4)
    return list(struct.unpack(f"<{len(b) // 4}I", b))
