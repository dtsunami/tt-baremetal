"""Gap 1.2 silicon de-risk: opt_proj_step.c on the x280 does whiten-bwd -> proj_bwd -> Adam over the
14-param 3D Gaussian, and publishes proj_fwd. Validate (a) the boot publish vs golden proj_fwd, and
(b) one full Adam step: feed known params + upstream dpsi grads, ring once, read back the 14 updated
params, compare to a numpy golden single step. Recover any wedge with tt-smi -r 0."""
import sys, struct, math, time
sys.path.insert(0, "/home/starboy/bhtop/src"); sys.path.insert(0, "/home/starboy/bhtop/scratchpad")
import numpy as np
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
import gap1_proj_golden as G

fb = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]
bf = lambda u: struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]
SRC = "/home/starboy/bhtop/src/bhtop/kernels/x280/het/opt_proj_step.c"
K = 12
Rv = np.eye(3); tv = np.array([0.0, 0.0, 6.0]); fx = fy = 70.0; cx = cy = 32.0
LR = [0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05]
B1, B2, EPS = 0.9, 0.999, 1e-8


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


def main():
    rng = np.random.default_rng(9)
    mean = rng.normal(0, 1.0, (K, 3)); sl = np.log(0.3 + rng.random((K, 3)) * 0.4); q = rng.normal(0, 1, (K, 4))
    param0 = np.concatenate([mean, sl, q, rng.uniform(0.4, 0.9, (K, 1)), rng.uniform(0, 1, (K, 3))], axis=1)  # [K,14]
    order = list(range(K))                                  # identity order (depth-sort tested elsewhere)
    dpsi = rng.normal(0, 0.5, (K, 5)); dLdop = rng.normal(0, 0.3, K); dcol = rng.normal(0, 0.3, (K, 3))

    # ---- golden: boot publish + one Adam step ----
    pub_gold = []
    grad = np.zeros((K, 14))
    for i in range(K):
        o = order[i]
        gx, gy, dep, a, b, c, cache = G.project_forward(param0[o, :3], param0[o, 3:6], param0[o, 6:10], Rv, tv, fx, fy, cx, cy)
        pub_gold.append([gx, gy, a, b, c, dep])
        g_a, g_b, g_c, g_gx, g_gy = whiten_bwd(a, b, c, gx, gy, dpsi[i])
        dmean, dsl, dq = G.project_backward(g_a, g_b, g_c, g_gx, g_gy, cache)
        grad[o, :3] = dmean; grad[o, 3:6] = dsl; grad[o, 6:10] = dq
        grad[o, 10] = dLdop[i]; grad[o, 11:14] = dcol[i]
    pub_gold = np.array(pub_gold)
    step = 1; bc1 = 1 / (1 - B1 ** step); bc2 = 1 / (1 - B2 ** step)
    m = np.zeros((K, 14)); v = np.zeros((K, 14))
    m = B1 * m + (1 - B1) * grad; v = B2 * v + (1 - B2) * grad ** 2
    p_gold = param0 - np.array(LR) * (m * bc1) / (np.sqrt(v * bc2) + EPS)
    p_gold[:, 10] = np.clip(p_gold[:, 10], 0.05, 0.99); p_gold[:, 11:14] = np.clip(p_gold[:, 11:14], 0, 1)

    # ---- device ----
    ctx = init_ttexalens(); dev = L2cpu(ctx=ctx)
    try: dev.bringup(0); print("[setup] x280 bringup ok")
    except Exception as e: print("[setup] bringup:", type(e).__name__, "(already up)")

    dev.wr(0, 0x30005000, [K, 0])
    cam = list(Rv.flatten()) + list(tv) + [fx, fy, cx, cy]
    dev.wr(0, 0x30005060, [fb(x) for x in cam])
    dev.wr(0, 0x300050A0, [o & 0xFFFFFFFF for o in order])
    dev.wr(0, 0x30005800, [fb(param0[o, j]) for o in range(K) for j in range(14)])
    dev.wr(0, 0x30006000, [0] * (K * 14)); dev.wr(0, 0x30006400, [0] * (K * 14))
    dev.wr(0, 0x30007000, [0] * (K * 6))                 # zero publish buffer
    dev.wr(0, 0x30004000, [0]); dev.wr(0, 0x30004010, [0])

    print("[run] loading opt_proj_step.c")
    dev.load(0, 0, tc.compile_source(SRC, base=CODE_ADDR, march="rv64gc"))
    for _ in range(80):
        if dev.telemetry(0, slots=1, hart=0)[0] == 0x4F505421: break
        time.sleep(0.03)
    else:
        print("FAIL: opt_proj_step never signalled OPT!"); return

    pub_dev = np.array([bf(u) for u in dev.rdn(0, 0x30007000, K * 6)]).reshape(K, 6)
    r = np.abs(pub_dev - pub_gold) / (np.abs(pub_gold) + 1e-6)
    print(f"[boot publish] max rel vs golden proj_fwd = {r.max():.2e}")

    # one step
    hdr = [K, step] + [fb(bc1), fb(bc2), fb(B1), fb(B2), fb(EPS)] + [fb(x) for x in LR]
    dev.wr(0, 0x30005000, hdr)
    dev.wr(0, 0x30005100, [fb(v) for i in range(K) for v in (*dpsi[i], dLdop[i], *dcol[i])])
    dev.wr(0, 0x30004000, [step])
    for _ in range(80):
        if dev.rd(0, 0x30004010) == step: break
        time.sleep(0.03)
    else:
        print("FAIL: step never completed"); return

    p_dev = np.array([bf(u) for u in dev.rdn(0, 0x30005800, K * 14)]).reshape(K, 14)
    absd = np.abs(p_dev - p_gold); reld = absd / (np.abs(p_gold) + 1e-6)
    names = ["mx", "my", "mz", "sl0", "sl1", "sl2", "q0", "q1", "q2", "q3", "op", "c0", "c1", "c2"]
    print(f"{'param':>5} {'max|abs|':>11} {'max|rel|':>11}")
    worst = 0.0
    for j, nm in enumerate(names):
        print(f"{nm:>5} {absd[:, j].max():11.2e} {reld[:, j].max():11.2e}"); worst = max(worst, reld[:, j].max())
    print(f"\nworst rel (x280 fp32 vs fp64 golden, 1 Adam step) = {worst:.2e}")
    print("GAP1_OPTPROJ_SILICON_OK" if worst < 5e-3 and r.max() < 5e-3 else "GAP1_OPTPROJ_SILICON_FAIL")


if __name__ == "__main__":
    main()
