#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
run_x280.py — compile + hot-load the CRT/quarter-square matmul kernels onto a live x280 hart and
read back the on-silicon result + cycle cost, then check bit-exact against the Python golden model.

Loads via the RNMI redirect path (no bringup if the tile is already running). Runs the SCALAR
baseline and the RVV kernel on the same hart and prints the perf comparison.

  python crt/run_x280.py [tile] [hart]
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from bhtop.l2cpu import L2cpu, CODE_ADDR
from bhtop.l2cpu import toolchain as tc
import crt_matmul as golden

# Each kernel runs as a fresh FIRST-load on its OWN hart (the reliable redirect path) — swapping a
# second kernel onto a still-running hart is the fragile case (it can wedge the seize), so we don't.
# Sources are the cockpit canon (single source of truth) — the same crt_scalar/crt_rvv kernels the
# x280 lab ships; this runner just compiles + on-silicon-validates them against the golden model.
KERNELS = [
    ("scalar", "src/bhtop/kernels/x280/crt_scalar/crt_scalar.c", "rv64gc", 0),  # hart0 = reliable seize
    ("rvv",    "src/bhtop/kernels/x280/crt_rvv/crt_rvv.c", "rv64gcv", 0),
]


def main():
    tile = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    exp_cs, exp_samples = golden.kernel_checksum()
    print(f"golden model: checksum={exp_cs}  samples={exp_samples}\n")

    dev = L2cpu()
    here = os.path.join(os.path.dirname(__file__), "..")
    results = {}
    for name, src, march, hart in KERNELS:
        path = os.path.join(here, src)
        words = tc.compile_source(path, base=CODE_ADDR, march=march)
        exp_tag = 1 if name == "scalar" else 2
        t = None
        for _ in range(6):                              # redirect seize is occasionally flaky — retry
            dev.load(tile, hart, words)
            time.sleep(0.4)
            t = dev.telemetry(tile, slots=16, hart=hart)
            if t[8] == exp_tag:
                break
        cs, cyc, c00, c12, c1515, c3131, instret, tag = t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8]
        ok = (cs == exp_cs and (c00, c12, c1515, c3131) == exp_samples and tag == exp_tag)
        results[name] = {"cyc": cyc, "instret": instret, "ok": ok, "cs": cs}
        print(f"[{name:6s} hart{hart}] checksum={cs} samples=({c00},{c12},{c1515},{c3131}) "
              f"-> {'MATCH' if ok else 'MISMATCH(tag=%d)' % tag}   cycles={cyc:,}  instret={instret:,}")

    if "scalar" in results and "rvv" in results and results["scalar"]["cyc"] and results["rvv"]["cyc"]:
        sp = results["scalar"]["cyc"] / results["rvv"]["cyc"]
        print(f"\n32x32 int8 CRT matmul (1024 MACs/output x 1024 outputs):")
        print(f"  scalar {results['scalar']['cyc']:>9,} cyc   |   rvv {results['rvv']['cyc']:>9,} cyc"
              f"   ->  {sp:.2f}x speedup")
    all_ok = all(r["ok"] for r in results.values())
    print("\nVERDICT:", "all kernels bit-exact vs golden model ✓" if all_ok else "MISMATCH — investigate")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
