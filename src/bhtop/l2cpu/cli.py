"""
bhtop-l2cpu — interactive loader for the Blackhole L2CPU x280 harts.

Run with no args for an interactive shell, or pass a command for one-shot use:

    bhtop-l2cpu tiles                      # list tiles + reset state
    bhtop-l2cpu bringup 0                  # release tile 0's harts (one-shot!)
    bhtop-l2cpu load 0 0 counter.c         # compile + load + run on tile0 hart0
    bhtop-l2cpu tele 0                      # read telemetry slots
    bhtop-l2cpu examples                    # bundled example sources

Inside the shell the same words work (e.g. `load 0 0 counter.c`, `tele 0`, `quit`).
Device access is created lazily, so `examples`/`help`/`disasm` work with no chip.
"""
import os
import shlex
import sys
import time

from . import L2cpu, TILES, Hang, CODE_ADDR, TELE_ADDR
from . import (HARTS, HART_STATUS, TRIGGER, RESET_VEC, RNMI_TRAP, RNMI_EXC,
               TELE_SLOTS, TELE_STRIDE)
from . import toolchain
from . import regmap

EXAMPLES = os.path.join(os.path.dirname(__file__), "examples")


def _i(s):
    return int(s, 0)


class App:
    def __init__(self):
        self.dev = None

    def need(self):
        if self.dev is None:
            print("(opening device…)")
            self.dev = L2cpu()
        return self.dev

    # ---- commands ----------------------------------------------------------
    def c_help(self, a):
        print(__doc__.strip())
        print("\ncommands: tiles | status <t> | bringup <t> [--yes] | load <t> <hart> <file> "
              "[--addr 0x..] [--lang asm|c|rust] | tele <t> [hart] | peek <t> <addr> [n] | "
              "poke <t> <addr> <val> | disasm <file> [--addr 0x..] | map | regs <t> [hart] | "
              "examples | reset | quit")

    def c_examples(self, a):
        print(f"bundled examples ({EXAMPLES}):")
        for f in sorted(os.listdir(EXAMPLES)):
            print(f"  {f:16s} {os.path.join(EXAMPLES, f)}")
        print("load one with:  load <tile> <hart> " + os.path.join(EXAMPLES, "counter.c"))

    def c_tiles(self, a):
        dev = self.need()
        for idx, (xy, bit) in TILES.items():
            if idx not in dev.loc:
                print(f"  tile {idx} {xy}: not present"); continue
            st = dev.reset_state(idx)
            s = "WEDGED (tt-smi -r 0)" if st["wedged"] else ("released/running" if st["released"] else "in reset")
            print(f"  tile {idx} noc0={xy} reset-bit{bit}: {s}")

    def c_status(self, a):
        dev = self.need(); t = _i(a[0])
        s = dev.status(t)
        print(f"tile {t}: reset={'released' if s['reset']['released'] else 'in-reset'} "
              f"(L2CPU_RESET=0x{s['reset']['raw']:08X})  hart_status=0x{s['hart_status']:04X}")
        print("  reset vectors: " + "  ".join(f"h{h}=0x{v:08X}" for h, v in enumerate(s['reset_vec'])))

    def c_bringup(self, a):
        dev = self.need(); t = _i(a[0])
        if "--yes" not in a:
            print(f"bringup releases tile {t}'s harts OUT OF RESET — IRREVERSIBLE until tt-smi -r 0.")
            if input("type 'yes' to proceed: ").strip().lower() != "yes":
                print("aborted."); return
        r = dev.bringup(t)
        print(f"bringup tile {t}: {'OK' if r['ok'] else 'FAILED'} (L2CPU_RESET=0x{r['l2cpu_reset']:08X}). "
              f"harts parked + trampoline installed. Use `load {t} <hart> <file>`.")

    def c_load(self, a):
        dev = self.need()
        pos = [x for x in a if not x.startswith("--")]
        t, hart, path = _i(pos[0]), _i(pos[1]), pos[2]
        addr = _i(_opt(a, "--addr", hex(CODE_ADDR)))
        lang = _opt(a, "--lang", None)
        if not os.path.exists(path):
            alt = os.path.join(EXAMPLES, path)
            path = alt if os.path.exists(alt) else path
        if not dev.reset_state(t)["released"]:
            print(f"tile {t} is still in reset — run `bringup {t}` first."); return
        print(f"compiling {os.path.basename(path)} ({lang or toolchain.detect_lang(path)}) @0x{addr:08X} …")
        words = toolchain.compile_source(path, base=addr, lang=lang)
        print(f"  {len(words)} words ({len(words)*4} bytes); loading to tile {t} hart {hart} + redirecting…")
        res = dev.load(t, hart, words, addr=addr, redirect=True)
        time.sleep(0.15)
        hb = dev.telemetry(t, 1)[0]
        if res.get("seized"):
            print(f"  loaded + seized hart {hart} ✓  telemetry[0]={hb:#x}  (`tele {t}` to watch)")
        else:
            print(f"  loaded. telemetry[0]={hb:#x}")

    def c_tele(self, a):
        dev = self.need(); t = _i(a[0]); hart = _i(a[1]) if len(a) > 1 else 0
        vals = dev.telemetry(t, TELE_SLOTS, hart)
        print(f"tile {t} hart {hart} telemetry @0x{TELE_ADDR + hart * TELE_STRIDE:08X}:")
        for i, v in enumerate(vals):
            tag = "  (heartbeat)" if i == 0 else ""
            if v:
                print(f"  [{i:2d}] 0x{v:08X} = {v}{tag}")

    def c_peek(self, a):
        dev = self.need(); t = _i(a[0]); addr = _i(a[1]); n = _i(a[2]) if len(a) > 2 else 1
        ws = dev.rdn(t, addr, n)
        for i, w in enumerate(ws):
            print(f"  0x{addr + i*4:08X}: 0x{w:08X}")

    def c_poke(self, a):
        dev = self.need(); dev.poke(_i(a[0]), _i(a[1]), _i(a[2])); print("ok")

    def c_disasm(self, a):
        pos = [x for x in a if not x.startswith("--")]
        addr = _i(_opt(a, "--addr", hex(CODE_ADDR)))
        print(toolchain.disasm(pos[0], base=addr))

    def c_map(self, a):
        """Print the static register/memory map — no chip needed (a quick reference)."""
        print(regmap.render_map())

    def c_regs(self, a):
        """Read + decode a tile's live hart-control registers (the bhtop 'see it live' bit)."""
        dev = self.need(); t = _i(a[0])
        only = _i(a[1]) if len(a) > 1 else None
        st = dev.reset_state(t)
        state = "WEDGED (tt-smi -r 0)" if st["wedged"] else ("released" if st["released"] else "in reset")
        print(f"tile {t} noc0={regmap.tile_coord(t)}  reset={state}  (L2CPU_RESET=0x{st['raw']:08X})")
        status = dev.rd(t, HART_STATUS) & 0xFFFF
        busy = regmap.decode_bits(status, [(i, f"hart{i}") for i in range(HARTS)])
        print(f"  HART_STATUS @0x{HART_STATUS:08X} = 0x{status:04X}   "
              f"({', '.join(busy) + ' active' if busy else 'all parked / in reset'})")
        trig = dev.rd(t, TRIGGER)
        pend = regmap.decode_bits(trig, [(i, f"h{i}") for i in range(HARTS)])
        print(f"  TRIGGER     @0x{TRIGGER:08X} = 0x{trig:08X}   "
              f"({'seize pending: ' + ','.join(pend) if pend else 'idle — no seize in flight'})")
        print("  per-hart vectors (reset_vec = where the hart runs):")
        for h in range(HARTS) if only is None else [only]:
            rv = dev.rdn(t, RESET_VEC + h * 8, 2)
            trap = dev.rdn(t, RNMI_TRAP + h * 16, 2)
            exc = dev.rdn(t, RNMI_EXC + h * 16, 2)
            j = lambda w: (w[1] << 32) | w[0]
            print(f"    hart{h}  reset_vec=0x{j(rv):010X}  rnmi_trap=0x{j(trap):08X}  "
                  f"rnmi_exc=0x{j(exc):08X}")

    def c_reset(self, a):
        print("This tool never resets the chip. To put harts back in reset, run:  tt-smi -r 0")

    CMDS = {"help": c_help, "?": c_help, "examples": c_examples, "tiles": c_tiles, "status": c_status,
            "bringup": c_bringup, "load": c_load, "run": c_load, "tele": c_tele, "telemetry": c_tele,
            "peek": c_peek, "poke": c_poke, "disasm": c_disasm, "map": c_map, "regs": c_regs,
            "reset": c_reset}

    def run(self, argv):
        cmd, rest = argv[0], argv[1:]
        fn = self.CMDS.get(cmd)
        if not fn:
            print(f"unknown command '{cmd}'. try `help`."); return 1
        try:
            fn(self, rest)
            return 0
        except (Hang, toolchain.ToolError, ValueError, IndexError, KeyError) as e:
            print(f"error: {e}")
            return 2

    def repl(self):
        print("bhtop-l2cpu interactive — `help` for commands, `quit` to exit.")
        while True:
            try:
                line = input("l2cpu> ").strip()
            except (EOFError, KeyboardInterrupt):
                print(); break
            if not line:
                continue
            if line in ("quit", "exit", "q"):
                break
            self.run(shlex.split(line))
        return 0


def _opt(args, name, default):
    if name in args:
        i = args.index(name)
        if i + 1 < len(args):
            return args[i + 1]
    return default


def main():
    argv = sys.argv[1:]
    app = App()
    return app.repl() if not argv else app.run(argv)


if __name__ == "__main__":
    sys.exit(main())
