#!/usr/bin/env python3
"""vec_freq_sweep.py — DVFS sweep: set the L2CPU CORE freq, measure throughput + IPC + power.

Demonstrates the key point: IPC (instructions per CYCLE) is frequency-INVARIANT; what scales
with core frequency is throughput (passes/sec = work/sec) and power. The UNCORE/NoC clock is
NOT swept (it is the transport to the tile). Only verified core-PLL points (200, 1750) are used.

  python scripts/vec_freq_sweep.py            # needs vec_virus already running on tile 0

NEVER run while the web server owns the device (second owner = hang).
"""
import time, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from bhtop.l2cpu import L2cpu, toolchain, CODE_ADDR  # noqa: E402

TILE, HART = 0, 0
NOPS = 256 * 32
KERNEL = os.path.join(os.path.dirname(__file__), "..", "l2cpu_kernels", "vec_virus.c")


def main():
    dev = L2cpu()
    rs = dev.reset_state(TILE)
    if rs["wedged"] or not rs["released"]:
        sys.exit("tile 0 not released/healthy — bringup first")
    # make sure the virus is running (heartbeat advancing); load it if not
    h = dev.telemetry(TILE, 1, HART)[0]; time.sleep(0.3)
    if dev.telemetry(TILE, 1, HART)[0] == h:
        print("loading vec_virus on tile 0 harts 0-3 …")
        w = toolchain.compile_source(KERNEL, base=CODE_ADDR)
        for hh in range(4):
            dev.load(TILE, hh, w, redirect=True)
        time.sleep(0.5)

    print(f"\n  {'core_MHz':>8}{'passes/s':>10}{'rel':>6}{'power_W':>9}{'IPC_FMA':>9}{'tele_l2clk':>11}")
    print("  " + "-" * 53)
    base = None
    try:
        for mhz in (1750, 200, 1750):
            dev.set_core_freq(mhz)
            time.sleep(0.6)
            if dev.reset_state(TILE)["wedged"]:
                print("  WEDGED after freq change — recover with tt-smi -r 0"); return
            h0 = dev.telemetry(TILE, 1, HART)[0]; t0 = time.time()
            time.sleep(1.0)
            s = dev.telemetry(TILE, 64, HART)
            rate = (s[0] - h0) / (time.time() - t0)
            base = base or rate
            pw = dev.power()["power_w"]
            ipc = NOPS / s[16] if s[16] else 0          # class 8 = vfmacc dcycles
            clk = dev.clocks()["core_l2cpu_mhz"][0]
            print(f"  {mhz:>8}{rate:>10.0f}{rate/base:>6.2f}{pw:>9}{ipc:>9.3f}{str(clk):>11}")
    finally:
        dev.set_core_freq(1750)
        print("\n  restored core PLL -> 1750 MHz")
    print("  IPC is flat across freq (per-cycle); passes/s and power scale with core MHz.")


if __name__ == "__main__":
    main()
