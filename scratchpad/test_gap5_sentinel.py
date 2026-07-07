"""Gap 5 DRAM sentinel: prove the x280 can directly use a large span of off-chip GDDR as distinct DRAM.
Loads dram_sentinel.c on the x280 (which writes a unique value per probe across [base,top), reads all back,
counts mismatches), then the HOST independently reads several probe offsets over NoC and cross-checks they
hold the x280-written values (proves host<->x280 coherency across the span + that it's real distinct DRAM).

  usage: test_gap5_sentinel.py [top_hex] [n]     (default 0x40000000 240)  # 256MB window
         test_gap5_sentinel.py 0x100000000 512                              # probe beyond, to 4GB
"""
import sys, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR

SRC = "/home/starboy/bhtop/src/bhtop/kernels/x280/het/dram_sentinel.c"
BASE = 0x30010000
TOP = int(sys.argv[1], 0) if len(sys.argv) > 1 else 0x40000000
N = int(sys.argv[2]) if len(sys.argv) > 2 else 240
KEY = lambda i: (0xC0DE0000 ^ ((i * 0x9E3779B1) & 0xFFFFFFFF)) & 0xFFFFFFFF


def main():
    stride = ((TOP - BASE) // N) & ~0xF
    if stride == 0: stride = 0x10
    print(f"[sentinel] base=0x{BASE:x} top=0x{TOP:x} n={N} stride=0x{stride:x} "
          f"span={(TOP-BASE)/2**20:.0f} MB")
    ctx = init_ttexalens(); dev = L2cpu(ctx=ctx)
    try: dev.bringup(0); print("[setup] x280 bringup ok")
    except Exception as e: print("[setup] bringup:", type(e).__name__, "(already up)")

    dev.load(0, 0, tc.compile_source(SRC, base=CODE_ADDR, march="rv64gc",
                                     defines={"PROBE_BASE": BASE, "PROBE_TOP": TOP, "PROBE_N": N}))
    for _ in range(120):
        if dev.telemetry(0, slots=1, hart=0)[0] == 0x53454E54: break
        time.sleep(0.03)
    else:
        print("FAIL: sentinel never signalled SENT (possible NoC wedge -> tt-smi -r 0)"); return
    t = dev.telemetry(0, slots=6, hart=0)
    n_dev, bad, first_bad, stride_dev, rb0 = t[1], t[2], t[3] << 4, t[4], t[5]
    print(f"[x280 self-check] probes={n_dev} bad={bad} first_bad=0x{first_bad:x} "
          f"stride=0x{stride_dev:x} readback@base=0x{rb0:08x} (want 0x{KEY(0):08x})")

    # host independently reads a spread of probe offsets over NoC and cross-checks
    host_bad = 0; sample = list(range(0, N, max(1, N // 12)))
    for i in sample:
        a = BASE + i * stride
        got = dev.rd(0, a) & 0xFFFFFFFF
        ok = (got == KEY(i))
        if not ok: host_bad += 1
        if i in sample[:6] or not ok:
            print(f"  host rd 0x{a:09x} = 0x{got:08x}  want 0x{KEY(i):08x}  {'ok' if ok else 'MISMATCH'}")
    span_mb = (TOP - BASE) / 2**20
    ok = (bad == 0 and host_bad == 0 and rb0 == KEY(0))
    print(f"\n[result] x280 bad={bad}/{n_dev}, host bad={host_bad}/{len(sample)}, span={span_mb:.0f} MB")
    print("GAP5_DRAM_SENTINEL_OK" if ok else "GAP5_DRAM_SENTINEL_FAIL")


if __name__ == "__main__":
    main()
