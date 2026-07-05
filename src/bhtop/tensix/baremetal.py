"""
tensix.baremetal — COLD-BOOT a Tensix baby-RISC over tt-exalens, with NO tt-metal.

The third Tensix launch path in bhtop, and the only one that owes nothing to tt-metal:
  * loader.TensixLauncher  — rides tt-metal's resident firmware launch (poke RTAs, re-go)
  * bootloader.Bootloader  — drives a tt-metal-parked resident kernel (stage/exec L1 overlays)
  * baremetal.BareMetal    — writes a freestanding kernel to L1 0x0 and deasserts the RISC reset

A bare-metal kernel is `crt0.s + bm_main()` (see kernels/tensix/baremetal/): crt0 sets sp, loads up
to 4 u32 params from the L1 arg block (0x1000), calls bm_main(a0..a3), and parks. The host writes the
image to L1 0x0 (the BRISC hardwired reset vector), pokes params, and toggles the RISC reset over
exalens' BabyRiscDebug — the Tensix analog of the x280 bringup. tt-metal's JIT/device-init is out of
the loop entirely (it hangs on this board's firmware bundle anyway; see tt-het-noc-poc).

VALIDATED ON SILICON (2026-07-04): BRISC executes our machine code from a cold reset (hello), issues
a bare-metal NOC0 read (nocread, worker->worker bit-exact), and reads the x280's GDDR window — the
x280->Tensix cooperative handoff, no tt-metal. Coexists with a live x280 on one shared exalens ctx.

    bm = BareMetal(1, 2)                       # worker noc0 (x,y), default BRISC
    bm.run(BareMetal.build("nocread"), params=[bm_coord(8, 3), 0x30002000, 32])
    print(bm.result())                         # the payload the kernel published
"""
import os
import struct
import subprocess
import time

from .loader import TensixLauncher

# L1 windows the crt0/bm_main ABI fixes (mirror kernels/tensix/baremetal/baremetal.h).
BM_ARGS = 0x1000        # host pokes up to 4 u32 params here before deasserting reset
BM_RESULT = 0x2000      # kernels publish their payload here
BM_DBG = 0x2100         # per-kernel debug scratch
_CANON = os.path.join(os.path.dirname(__file__), "..", "kernels", "tensix", "baremetal")
_BUILD = os.path.expanduser("~/bhtop/kernels/tensix/baremetal/_build")

# Per-RISC reset PC (Blackhole, L1 — silicon-verified cold-boot). BRISC is hardwired to 0x0; the
# compute threads' PCs are PROGRAMMABLE via BabyRiscDebug.set_code_start_address, so a kernel just
# links at its RISC's PC. ("TRISC fetches from IRAM" is a Wormhole-ism — false on Blackhole.)
_RESET_PC = {"brisc": 0x0, "trisc0": 0x6000, "trisc1": 0xA000, "trisc2": 0xE000}


def bm_coord(x, y):
    """Encode a NoC0 (x, y) into the target-coordinate word (y<<6)|x — matches baremetal.h."""
    return ((y & 0x3F) << 6) | (x & 0x3F)


class BareMetal:
    """Own one Tensix baby-RISC bare-metal over exalens: load a freestanding kernel, run it by
    toggling the RISC reset, read results — no tt-metal. `x, y` is the worker's noc0 coordinate."""

    def __init__(self, x, y, ctx=None, risc="brisc", device_id=0):
        self.L = TensixLauncher.at(x, y, ctx=ctx, device_id=device_id)
        self.ctx = self.L.ctx
        self.coord = self.L.coord
        self.risc = risc
        self.pc = _RESET_PC[risc]                 # where this RISC boots (BRISC 0x0, TRISCs 0x6000/A000/E000)
        self.noc = tuple(self.coord.to("noc0"))
        self._rd = self.ctx.devices[device_id].get_block(self.coord).get_risc_debug(risc)

    # ---- reset control (the "bringup" knob) ------------------------------------------
    def in_reset(self):
        return bool(self._rd.is_in_reset())

    def halt(self):
        """Assert the RISC reset — park the core (clean stop; re-run with run())."""
        self._rd.set_reset_signal(True)

    # ---- load + run ------------------------------------------------------------------
    def load(self, binpath):
        """Write a freestanding kernel image (raw .bin, linked at this RISC's reset PC) to L1; verify."""
        data = open(binpath, "rb").read()
        if len(data) % 4:
            data += b"\x00" * (4 - len(data) % 4)
        words = list(struct.unpack(f"<{len(data) // 4}I", data))
        self.L.wr(self.pc, words)
        if self.L.rd(self.pc, len(words)) != words:
            raise RuntimeError("L1 load verify mismatch")
        return words

    def set_params(self, params):
        self.L.wr(BM_ARGS, [(p & 0xFFFFFFFF) for p in list(params)[:4]])

    def run(self, binpath, params=None, poison=True, settle=0.2):
        """Halt, load the kernel + params, program the reset PC (compute RISCs), deassert -> it runs.
        Poisons BM_RESULT first so a stale/failed run is distinguishable from a real one."""
        self._rd.set_reset_signal(True)                  # assert (halt) before touching L1
        if poison:
            self.L.wr(BM_RESULT, [0xEEEEEEEE] * 8)
        if params:
            self.set_params(params)
        self.load(binpath)
        if self.risc != "brisc":                         # BRISC boots hardwired 0x0; TRISCs are programmable
            self._rd.set_code_start_address(self.pc)
        self._rd.set_reset_signal(False)                 # deassert -> fetch from self.pc and run
        if settle:
            time.sleep(settle)
        return self

    def run_canon(self, name, params=None, **kw):
        """Build canon kernel `name` linked at THIS RISC's reset PC, then run it."""
        return self.run(self.build(name, self.pc), params=params, **kw)

    # ---- read back -------------------------------------------------------------------
    def result(self, n=8):
        return self.L.rd(BM_RESULT, n)

    def dbg(self, n=4):
        return self.L.rd(BM_DBG, n)

    def rd(self, addr, n=1):
        return self.L.rd(addr, n)

    # ---- canon kernels ---------------------------------------------------------------
    @staticmethod
    def build(name, base=0x0):
        """Compile a canon kernel (crt0 + {name}/{name}.c) linked at `base` — the target RISC's reset
        PC (brisc 0x0, trisc0 0x6000, trisc1 0xA000, trisc2 0xE000) — and return the .bin path."""
        sfpi = os.path.expanduser("~/tt-metal/runtime/sfpi/compiler/bin")
        os.makedirs(_BUILD, exist_ok=True)
        ld = os.path.join(_BUILD, f"link_{base:#x}.ld")
        with open(ld, "w") as f:
            f.write(f"SECTIONS {{ . = {base:#x}; .text : {{ *(.text.start) *(.text*) }} "
                    f".rodata : {{ *(.rodata*) }} }}\n")
        elf = os.path.join(_BUILD, f"{name}_{base:#x}.elf")
        binf = os.path.join(_BUILD, f"{name}_{base:#x}.bin")
        subprocess.run([f"{sfpi}/riscv-tt-elf-gcc", "-march=rv32im", "-mabi=ilp32", "-Os", "-nostdlib",
                        "-ffreestanding", f"-I{_CANON}", "-T", ld, os.path.join(_CANON, "crt0.s"),
                        os.path.join(_CANON, name, f"{name}.c"), "-o", elf],
                       check=True, capture_output=True, text=True)
        subprocess.run([f"{sfpi}/riscv-tt-elf-objcopy", "-O", "binary", "-j", ".text", "-j", ".rodata",
                        elf, binf], check=True)
        return binf
