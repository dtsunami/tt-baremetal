#!/usr/bin/env python3
"""vec_power_sweep.py — steer the vector virus across instruction classes and TRACK POWER.

Loads vec_virus on all 4 harts of a tile, then for each RVV instruction class it steers
every hart onto that one class (mailbox doorbell), lets the board power settle, and reads
real watts/current/temp via ARC telemetry. Prints which instruction draws the most power.

  python scripts/vec_power_sweep.py [--tile 0] [--mutate]

NEVER run while the bhtop web server owns the device (second owner = PCIe hang).
"""
import argparse, time, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bhtop.l2cpu import L2cpu, toolchain, CODE_ADDR  # noqa: E402

KERNEL = os.path.join(os.path.dirname(__file__), "..", "l2cpu_kernels", "vec_virus.c")
NAMES = ["vadd","vsub","vxor","vsll","vmul","vmacc","vfadd","vfmul","vfmacc",
         "vredsum","vrgather","vfmacc@m4"]
OP_SELECT, OP_SEED, OP_MUTATE, OP_PARK, OP_RUN = 10, 11, 12, 4, 5


def read_power(dev, n=3, gap=0.15):
    """Average a few telemetry reads to smooth the power number."""
    acc = {"power_w": 0, "current_a": 0, "temp": 0.0, "l2clk": 0}
    got = 0
    for _ in range(n):
        p = dev.power()
        if p["power_w"] is None:
            continue
        acc["power_w"] += p["power_w"]; acc["current_a"] += (p["current_a"] or 0)
        acc["temp"] += (p["asic_temp_c"] or 0); acc["l2clk"] += (p["l2cpuclk_mhz"][0] or 0)
        got += 1; time.sleep(gap)
    if not got:
        return None
    return {k: round(v / got, 1) for k, v in acc.items()}


def bcast(dev, tile, op, arg0=0):
    for h in range(4):
        dev.command(tile, h, op, arg0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--mutate", action="store_true", help="auto-randomize the seed each pass")
    ap.add_argument("--settle", type=float, default=1.5, help="seconds to let power settle")
    a = ap.parse_args()
    dev = L2cpu()
    words = toolchain.compile_source(KERNEL, base=CODE_ADDR)
    print(f"loading vec_virus -> tile {a.tile} harts 0-3 ({len(words)} words @0x{CODE_ADDR:08X})")
    for h in range(4):
        dev.load(a.tile, h, words, redirect=True)
    if a.mutate:
        bcast(dev, a.tile, OP_MUTATE, 1)

    # baseline: park all harts (scalar poll loop only — vector unit idle)
    bcast(dev, a.tile, OP_PARK)
    time.sleep(a.settle)
    base = read_power(dev)
    print(f"\nbaseline (4 harts parked, vector idle): {base['power_w']}W  {base['current_a']}A  "
          f"{base['temp']}C  l2cpuclk={base['l2clk']}MHz")
    bcast(dev, a.tile, OP_RUN)

    rows = []
    print(f"\n  {'#':>2} {'instruction':<12}{'power_W':>9}{'ΔW':>7}{'current_A':>11}{'Tasic_C':>9}")
    print("  " + "-" * 50)
    for i, nm in enumerate(NAMES):
        bcast(dev, a.tile, OP_SELECT, i)
        time.sleep(a.settle)
        p = read_power(dev)
        d = round(p["power_w"] - base["power_w"], 1)
        rows.append((i, nm, p["power_w"], d, p["current_a"], p["temp"]))
        print(f"  {i:>2} {nm:<12}{p['power_w']:>9}{d:>+7}{p['current_a']:>11}{p['temp']:>9}")

    bcast(dev, a.tile, OP_SELECT, 0xFFFFFFFF)   # back to sweep-all
    top = max(rows, key=lambda r: r[2])
    print(f"\n  hottest instruction: {top[1]} at {top[2]}W (+{top[3]}W over parked baseline)")
    print("  (one tile / 4 harts; scale to all 16 harts for a bigger swing)")


if __name__ == "__main__":
    main()
