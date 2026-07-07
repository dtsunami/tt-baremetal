"""Gap 5b: prove the x280 holds + optimizes N (up to ~1.3M) 3D Gaussians RESIDENT in big GDDR.
opt_proj_big.c inits N params + Adam state in the 256MB window (0x30010000+) from a deterministic hash,
then does one full projection->whiten-bwd->proj-bwd->Adam step over ALL N, timed. Host spot-checks sampled
Gaussians (init and post-step) against a bit-exact Python golden, and reports throughput (cycles/Gaussian).

  usage: test_gap5_big.py [N]     (default 100000)
"""
import sys, struct, time, math
sys.path.insert(0, "/home/starboy/bhtop/src"); sys.path.insert(0, "/home/starboy/bhtop/scratchpad")
import numpy as np
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
import gap1_proj_golden as G

fb = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]
bf = lambda u: struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]
SRC = "/home/starboy/bhtop/src/bhtop/kernels/x280/het/opt_proj_big.c"
PBASE = 0x30010000
N = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
Rv = np.eye(3); tv = np.array([0.0, 0.0, 6.0]); fx = fy = 70.0; cx = cy = 32.0
LR = [0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05]
B1, B2, EPS = 0.9, 0.999, 1e-8
M32 = 0xFFFFFFFF


def h32(x):
    x = (x * 0x9E3779B1) & M32; x ^= x >> 16
    x = (x * 0x85EBCA6B) & M32; x ^= x >> 13
    return x & M32
def u01(i, salt): return (h32(i ^ ((salt * 0x2545F491) & M32)) >> 8) * (1.0 / 16777216.0)


def init_gauss(i):
    return np.array([
        (u01(i, 1) - 0.5) * 3, (u01(i, 2) - 0.5) * 3, (u01(i, 3) - 0.5) * 3,
        -1.8 + u01(i, 4) * 0.4, -1.8 + u01(i, 5) * 0.4, -1.8 + u01(i, 6) * 0.4,
        1.0 + (u01(i, 7) - 0.5) * 0.4, u01(i, 8) - 0.5, u01(i, 9) - 0.5, u01(i, 10) - 0.5,
        0.4 + u01(i, 11) * 0.5, u01(i, 12), u01(i, 13), u01(i, 14)], np.float64)
def syn_grad(i):
    dpsi = [(u01(i, 20 + k) - 0.5) * 0.2 for k in range(5)]
    dLdop = (u01(i, 25) - 0.5) * 0.2
    dcol = [(u01(i, 26 + k) - 0.5) * 0.2 for k in range(3)]
    return dpsi, dLdop, dcol


def whiten_bwd(a, b, c, gx, gy, dpsi):
    d_sa, d_m12, d_tx, d_m22, d_ty = dpsi
    sa = math.sqrt(max(a, 1e-8)); m12 = b / sa
    t = max(c - b * b / a, 1e-8); m22 = math.sqrt(t)
    Dsa = d_sa + d_tx * (-gx); Dm12 = d_m12 + d_tx * (-gy); Dm22 = d_m22 + d_ty * (-gy)
    g_gx = d_tx * (-sa); g_gy = d_tx * (-m12) + d_ty * (-m22)
    g_a = Dsa * (0.5 / sa) + Dm12 * (-0.5 * b / (a * sa)) + Dm22 * ((b * b / (a * a)) / (2 * m22))
    g_b = Dm12 * (1.0 / sa) + Dm22 * (-b / (a * m22))
    g_c = Dm22 * (1.0 / (2 * m22))
    return g_a, g_b, g_c, g_gx, g_gy


def golden_step(i):
    p = init_gauss(i)
    gx, gy, dep, a, b, c, cache = G.project_forward(p[:3], p[3:6], p[6:10], Rv, tv, fx, fy, cx, cy)
    dpsi, dLdop, dcol = syn_grad(i)
    g_a, g_b, g_c, g_gx, g_gy = whiten_bwd(a, b, c, gx, gy, dpsi)
    dmean, dsl, dq = G.project_backward(g_a, g_b, g_c, g_gx, g_gy, cache)
    g = np.array([*dmean, *dsl, *dq, dLdop, *dcol])
    bc1 = 1 / (1 - B1); bc2 = 1 / (1 - B2)
    m = (1 - B1) * g; v = (1 - B2) * g ** 2
    np_ = p - np.array(LR) * (m * bc1) / (np.sqrt(v * bc2) + EPS)
    np_[10] = min(max(np_[10], 0.05), 0.99); np_[11:14] = np.clip(np_[11:14], 0, 1)
    return p, np_


def main():
    bytes_per = 14 * 4 * 3
    print(f"[gap5b] N={N}  resident footprint = {N*bytes_per/2**20:.1f} MB (params+m+v) in the 256MB window")
    ctx = init_ttexalens(); dev = L2cpu(ctx=ctx)
    try: dev.bringup(0); print("[setup] x280 bringup ok")
    except Exception as e: print("[setup] bringup:", type(e).__name__, "(already up)")

    dev.wr(0, 0x30005000, [N, 0] + [fb(v) for v in (1/(1-B1), 1/(1-B2), B1, B2, EPS)] + [fb(x) for x in LR])
    dev.wr(0, 0x30005060, [fb(x) for x in (list(Rv.flatten()) + list(tv) + [fx, fy, cx, cy])])
    dev.wr(0, 0x30004000, [0]); dev.wr(0, 0x30004010, [0]); dev.wr(0, 0x30004020, [3])
    print("[run] loading opt_proj_big.c")
    dev.load(0, 0, tc.compile_source(SRC, base=CODE_ADDR, march="rv64gc"))
    for _ in range(80):
        if dev.telemetry(0, slots=1, hart=0)[0] == 0x42494721: break
        time.sleep(0.03)
    else:
        print("FAIL: opt_proj_big not resident"); return

    sample = [0, 1, N // 7, N // 3, N // 2, (2 * N) // 3, N - 2, N - 1]

    def ring_and_wait(cmd, label, tmo):
        dev.wr(0, 0x30004020, [cmd]); ring = dev.rd(0, 0x30004000) + 1; dev.wr(0, 0x30004000, [ring])
        t0 = time.time()
        while time.time() - t0 < tmo and dev.rd(0, 0x30004010) != ring:
            time.sleep(0.02)
        cyc = dev.telemetry(0, slots=4, hart=0); cycles = cyc[2] | (cyc[3] << 32)
        print(f"[{label}] N={cyc[1]} cycles={cycles:,} ({cycles/max(N,1):.0f} cyc/Gaussian, {time.time()-t0:.2f}s wall)")
        return cycles

    ring_and_wait(3, "init", 120)
    # verify init
    bad_i = 0
    for i in sample:
        dev_p = np.array([bf(u) for u in dev.rdn(0, PBASE + i * 14 * 4, 14)])
        gold_p = init_gauss(i)
        if np.max(np.abs(dev_p - gold_p)) > 3e-3: bad_i += 1
    print(f"[init check] {len(sample)-bad_i}/{len(sample)} sampled Gaussians match golden init")

    ring_and_wait(1, "adam-step", 120)
    worst = 0.0; bad_s = 0
    for i in sample:
        dev_p = np.array([bf(u) for u in dev.rdn(0, PBASE + i * 14 * 4, 14)])
        _, gold_p = golden_step(i)
        rel = np.max(np.abs(dev_p - gold_p) / (np.abs(gold_p) + 1e-6)); worst = max(worst, rel)
        if rel > 5e-3: bad_s += 1
    print(f"[step check] {len(sample)-bad_s}/{len(sample)} sampled Gaussians match golden Adam step (worst rel {worst:.2e})")
    ok = (bad_i == 0 and bad_s == 0)
    print("GAP5_BIG_SILICON_OK" if ok else "GAP5_BIG_SILICON_FAIL")


if __name__ == "__main__":
    main()
