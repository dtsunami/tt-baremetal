"""het_x280 PRODUCER chain — the x280 projects+whitens its RESIDENT params (cmd=2) then PRODUCES a tile's
tilized operands from the coeff buffer by id list (cmd=5), fully on-device. Compare the 6 produced operand
tiles to the host tilization of the same 3D scene. Proves operand production needs NO host coeffs."""
import sys, struct, time
sys.path.insert(0, "/home/starboy/bhtop/src"); sys.path.insert(0, "/home/starboy/bhtop/scratchpad")
import numpy as np
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
from bhtop.tensix import splat as SP, matmul as MM
import gap1_proj_golden as PG

K = 12
SRC = "/home/starboy/bhtop/src/bhtop/kernels/x280/het/het_x280.c"
X_HDR, X_CAM, X_IDL, X_DB, X_DONE, X_CMD = 0x30005000, 0x30005060, 0x300050A0, 0x30004000, 0x30004010, 0x30004020
PARAM = 0x30100000; OPBASE = 0x30080000
Rv = np.eye(3); tv = np.array([0.0, 0.0, 6.0]); fx = fy = 70.0; cx = cy = 32.0
fb = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]
def enc(flat): return MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
def pad(rc): return SP._pad32(rc)
def bf16dec(words): return MM.untilize32(MM.unpack_bf16_words(words))


def rand_scene3d(seed, n):
    rng = np.random.default_rng(seed)
    mean = rng.normal(0, 1.0, (n, 3)); sl = rng.normal(-1.8, 0.2, (n, 3))
    q = rng.normal(0, 1, (n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    op = rng.uniform(0.4, 0.9, (n, 1)); col = rng.uniform(0.1, 0.9, (n, 3))
    return np.concatenate([mean, sl, q, op, col], axis=1)


def host_operands(param):
    gs = []
    for o in range(K):
        gx, gy, dep, a, b, c, _ = PG.project_forward(param[o, :3], param[o, 3:6], param[o, 6:10], Rv, tv, fx, fy, cx, cy)
        gs.append((gx, gy, a, b, c, float(param[o, 10]), *[float(x) for x in param[o, 11:14]], dep))
    order = sorted(range(K), key=lambda i: gs[i][9]); gso = [gs[i] for i in order]
    psi, Ppair, Dop, Dnop, Mcomb, color = SP._consts(gso, K)
    colorT = [[color[k][r] for k in range(K)] for r in range(3)]
    op = [Dop[k][k] for k in range(K)]
    opB = [[(op[k] if k < K else 0.5) for k in range(32)] for _ in range(32)]
    psi_rows = [[psi[r][c] for c in range(2 * K)] for r in range(3)]
    return {"psi": pad(psi_rows), "Dop": pad(Dop), "Dnop": pad(Dnop), "color": pad(color),
            "colorT": pad(colorT), "opB": pad(opB)}


def main():
    pr = lambda *a: print(*a, flush=True)
    ctx = init_ttexalens(); dev = L2cpu(ctx=ctx)
    try: dev.bringup(0)
    except Exception: pass
    param = rand_scene3d(7, K)
    ho = host_operands(param)

    dev.wr(0, X_HDR, [K])
    dev.wr(0, X_CAM, [fb(x) for x in (list(Rv.flatten()) + list(tv) + [fx, fy, cx, cy])])
    dev.wr(0, PARAM, [fb(param[o, j]) for o in range(K) for j in range(14)])
    dev.wr(0, X_DB, [0]); dev.wr(0, X_DONE, [0])
    dev.load(0, 0, tc.compile_source(SRC, base=CODE_ADDR, march="rv64gc"))
    for _ in range(80):
        if dev.telemetry(0, slots=1, hart=0)[0] == 0x48455421: break
        time.sleep(0.03)
    else: pr("FAIL: het_x280 no HET!"); return

    def ring(cmd):
        dev.wr(0, X_CMD, [cmd]); r = dev.rd(0, X_DB) + 1; dev.wr(0, X_DB, [r])
        while dev.rd(0, X_DONE) != r: time.sleep(0.002)
        return dev.telemetry(0, slots=4, hart=0)

    t2 = ring(2)                                    # project + whiten all resident params
    dev.wr(0, X_IDL, [K] + list(range(K)))          # tile id list = all K Gaussians
    t5 = ring(5)                                    # produce this tile's operands
    pr(f"[het] project+whiten {t2[2]:,} cyc ; produce {t5[2]:,} cyc")

    names = ["psi", "Dop", "Dnop", "color", "colorT", "opB"]
    print(f"{'operand':>7} {'max|abs Δ|':>11} {'max|rel Δ|':>11}")
    worst = 0.0
    for i, nm in enumerate(names):
        dev_t = np.array(bf16dec(dev.rdn(0, OPBASE + i * 0x800, 512)))
        host_t = np.array(bf16dec(enc(ho[nm])))
        d = np.abs(dev_t - host_t); rel = (d / (np.abs(host_t) + 1e-3)).max()
        print(f"{nm:>7} {d.max():11.2e} {rel:11.2e}"); worst = max(worst, rel)
    print(f"\nworst rel (x280-produced vs host, fp32/bf16) = {worst:.2e}")
    print("HET_PRODUCE_OK — x280 produces operands from RESIDENT PARAMS, no host coeffs" if worst < 5e-2
          else "HET_PRODUCE_FAIL")


if __name__ == "__main__":
    main()
