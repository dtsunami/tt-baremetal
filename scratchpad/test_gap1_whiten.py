"""Gap 1.1 silicon de-risk: run proj_fwd + Cholesky whiten ON the x280 (cb_whiten.c) and check the
published psi (sa,m12,m22,c1,c2) + depth against the numpy golden (gap1_proj_golden). Isolated: does
not touch the resident Tensix trainer. Recover any wedge with tt-smi -r 0."""
import sys, struct, math, time
sys.path.insert(0, "/home/starboy/bhtop/src"); sys.path.insert(0, "/home/starboy/bhtop/scratchpad")
import numpy as np
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
import gap1_proj_golden as G

fb = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]
bf = lambda u: struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]
SRC = "/home/starboy/bhtop/src/bhtop/kernels/x280/het/cb_whiten.c"
K = 12
Rv = np.eye(3); tv = np.array([0.0, 0.0, 6.0]); fx = fy = 70.0; cx = cy = 32.0


def host_whiten(a, b, c, gx, gy):
    sa = math.sqrt(max(a, 1e-8)); m12 = b / sa
    t = c - b * b / a; t = t if t > 0 else 0.0; m22 = math.sqrt(t)
    return sa, m12, m22, -(sa * gx + m12 * gy), -(m22 * gy)


def main():
    rng = np.random.default_rng(3)
    mean = rng.normal(0, 1.2, (K, 3)); sl = np.log(0.25 + rng.random((K, 3)) * 0.5)
    q = rng.normal(0, 1, (K, 4))

    # golden psi + depth
    gold = []
    for i in range(K):
        gx, gy, dep, a, b, c, _ = G.project_forward(mean[i], sl[i], q[i], Rv, tv, fx, fy, cx, cy)
        gold.append((*host_whiten(a, b, c, gx, gy), dep))
    gold = np.array(gold)

    ctx = init_ttexalens(); dev = L2cpu(ctx=ctx)
    try: dev.bringup(0); print("[setup] x280 bringup ok")
    except Exception as e: print("[setup] bringup:", type(e).__name__, "(already up)")

    dev.wr(0, 0x30005000, [K])
    cam = list(Rv.flatten()) + list(tv) + [fx, fy, cx, cy]
    dev.wr(0, 0x30005004, [fb(x) for x in cam])
    params = [v for i in range(K) for v in (*mean[i], *sl[i], *q[i])]
    dev.wr(0, 0x30005044, [fb(x) for x in params])
    dev.wr(0, 0x30006000, [0] * (K * 5)); dev.wr(0, 0x30006400, [0] * K)

    print("[run] loading cb_whiten.c on x280")
    dev.load(0, 0, tc.compile_source(SRC, base=CODE_ADDR, march="rv64gc"))
    for _ in range(80):
        if dev.telemetry(0, slots=1, hart=0)[0] == 0x57484954: break
        time.sleep(0.03)
    else:
        print("FAIL: cb_whiten never signalled WHIT"); return

    psi = np.array([bf(u) for u in dev.rdn(0, 0x30006000, K * 5)]).reshape(K, 5)
    dep = np.array([bf(u) for u in dev.rdn(0, 0x30006400, K)])
    dev_out = np.concatenate([psi, dep[:, None]], axis=1)

    absd = np.abs(dev_out - gold); reld = absd / (np.abs(gold) + 1e-6)
    labels = ["sa", "m12", "m22", "c1", "c2", "depth"]
    print(f"{'field':>6} {'max|abs|':>11} {'max|rel|':>11}")
    worst = 0.0
    for j, lb in enumerate(labels):
        print(f"{lb:>6} {absd[:, j].max():11.2e} {reld[:, j].max():11.2e}"); worst = max(worst, reld[:, j].max())
    print(f"\nworst rel (x280 fp32 vs fp64 golden) = {worst:.2e}")
    print("GAP1_WHITEN_SILICON_OK" if worst < 5e-3 else "GAP1_WHITEN_SILICON_FAIL")


if __name__ == "__main__":
    main()
