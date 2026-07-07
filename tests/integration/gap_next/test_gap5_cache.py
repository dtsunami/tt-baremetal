"""Gap 5 throughput de-risk: time a GDDR read-modify-write sweep through the UNCACHED window vs the
CACHED alias of the same GDDR (bench_cache.c). If cached is much faster, resident param/Adam state for
millions of Gaussians should live in the cached alias (with coherency managed for host reads) to make
training-speed steps feasible."""
import sys, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
SRC = "/home/starboy/bhtop/src/bhtop/kernels/x280/het/bench_cache.c"


def main():
    ctx = init_ttexalens(); dev = L2cpu(ctx=ctx)
    try: dev.bringup(0); print("[setup] x280 bringup ok")
    except Exception as e: print("[setup] bringup:", type(e).__name__, "(already up)")
    dev.load(0, 0, tc.compile_source(SRC, base=CODE_ADDR, march="rv64gc"))
    for _ in range(200):
        if dev.telemetry(0, slots=1, hart=0)[0] == 0x42454E43: break
        time.sleep(0.05)
    else:
        print("FAIL: bench never signalled BENC (cached alias may fault -> tt-smi -r 0)"); return
    t = dev.telemetry(0, slots=9, hart=0)
    nw, niter = t[1], t[2]
    unc = t[3] | (t[4] << 32); cac = t[5] | (t[6] << 32)
    accesses = nw * (niter + 2)      # seed + niter RMW + checksum read
    print(f"working set = {nw*4/1024:.0f} KiB, {niter} RMW passes, {accesses:,} word-accesses each")
    print(f"  UNCACHED (0x30100000):     {unc:>15,} cyc  = {unc/accesses:8.1f} cyc/word")
    print(f"  CACHED   (0x400030100000): {cac:>15,} cyc  = {cac/accesses:8.1f} cyc/word")
    print(f"  checksums: unc=0x{t[7]:08x} cac=0x{t[8]:08x}  {'MATCH' if t[7]==t[8] else 'DIFFER'}")
    if cac > 0:
        print(f"  cached speedup = {unc/cac:.1f}x")
    print("GAP5_CACHE_BENCH_DONE" if t[7] == t[8] else "GAP5_CACHE_BENCH_CHECKSUM_MISMATCH")


if __name__ == "__main__":
    main()
