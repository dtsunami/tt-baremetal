#!/usr/bin/env python3
"""vec_virus_run.py — load the RVV power-virus and print per-instruction max-IPC.

Loads l2cpu_kernels/vec_virus.c onto an x280 hart (live RNMI redirect, repeatable),
lets it run a couple of suite passes, then reads the per-class telemetry and prints
max IPC / cyc-per-op / element + FLOP throughput for each vector instruction class.

  python scripts/vec_virus_run.py [--tile 0] [--hart 0] [--all-harts]

Needs the tile already released (it is, normally). NEVER run while the bhtop web
server owns the device — that's a second device owner = PCIe hang (see l2cpu-cockpit).
"""
import argparse, time, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bhtop.l2cpu import L2cpu, toolchain, CODE_ADDR  # noqa: E402

KERNEL = os.path.join(os.path.dirname(__file__), "..", "l2cpu_kernels", "vec_virus.c")
NAMES = ["vadd.vv","vsub.vv","vxor.vv","vsll.vv","vmul.vv","vmacc.vv",
         "vfadd.vv","vfmul.vv","vfmacc.vv","vredsum.vs","vrgather.vv","vfmacc.vv@m4"]
UNIT  = ["int ALU","int ALU","int logic","int shift","int MUL","int MAC",
         "FP add","FP mul","FP FMA","reduce(serial)","permute net","FP FMA LMUL=4"]
NOPS = 256 * 32   # vector instructions issued per class (must match the kernel)


def table(s):
    misa = s[3]; v = (misa >> 21) & 1
    print(f"VLEN={s[1]} bits  vlenb={s[2]}  misa=0x{misa:08X} (V={'yes' if v else 'NO'})  "
          f"vl@m1={s[6]}  vl@m4={s[7]}  passes={s[0]}")
    hdr = (f"  {'#':>2} {'instruction':<13}{'unit':<16}{'dcycles':>9}{'cyc/op':>8}"
           f"{'max-IPC':>9}{'tot-IPC':>9}{'elem/cyc':>9}")
    print("\n" + hdr); print("  " + "-" * (len(hdr) - 2))
    for i in range(12):
        dcyc, dret, nops = s[8 + i], s[24 + i], s[40 + i] or NOPS
        vl = s[7] if i == 11 else s[6]
        if not dcyc:
            print(f"  {i:>2} {NAMES[i]:<13}{UNIT[i]:<16}{'--- did not run ---':>35}"); continue
        print(f"  {i:>2} {NAMES[i]:<13}{UNIT[i]:<16}{dcyc:>9}{dcyc/nops:>8.2f}"
              f"{nops/dcyc:>9.3f}{dret/dcyc:>9.3f}{vl*nops/dcyc:>9.1f}")
    fl = lambda i, e: 2 * e * (s[40 + i] or NOPS) / s[8 + i]
    print(f"\n  FP32 FMA throughput:  m1 = {fl(8,16):.2f} FLOP/cyc   m4 = {fl(11,64):.2f} FLOP/cyc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--hart", type=int, default=0)
    ap.add_argument("--all-harts", action="store_true", help="run the virus on all 4 harts")
    a = ap.parse_args()
    dev = L2cpu()
    if not dev.reset_state(a.tile)["released"]:
        sys.exit(f"tile {a.tile} is in reset — `bhtop-l2cpu bringup {a.tile}` first (one-shot).")
    words = toolchain.compile_source(KERNEL, base=CODE_ADDR)
    harts = range(4) if a.all_harts else [a.hart]
    for h in harts:
        r = dev.load(a.tile, h, words, redirect=True)
        print(f"loaded vec_virus -> tile {a.tile} hart {h}  seized={r['seized']}  ({len(words)} words)")
    time.sleep(1.5)
    for h in harts:
        print(f"\n=== tile {a.tile} hart {h} ===")
        table(dev.telemetry(a.tile, 64, h))


if __name__ == "__main__":
    main()
