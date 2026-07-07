"""Lever 3 / grad-consume — the x280 reads the worker's tilized grad tiles from its GDDR inbox (written by
cb_writer), DETILIZES + extracts per-Gaussian grads (5 psi coeffs + dLdop + dLdcolor=w^T@dLdC) + the scalar
SSE loss, entirely on-device (opt_grad_step). Verify vs the host extraction. This closes the fully-on-device
data path: host reads NO grads and NO image — only a scalar loss (telemetry)."""
import sys, struct, time, math
sys.path.insert(0, "/home/starboy/bhtop/src"); sys.path.insert(0, "/home/starboy/bhtop/scratchpad")
import numpy as np
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device as rd, read_words_from_device as rds, write_words_to_device as wr
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import splat as SP, matmul as MM, llk_run
from bhtop.tensix.resident import boot_resident
from bhtop.tensix.baremetal import BareMetal, bm_coord

K, SIZE, P = 12, 16, 32
H = dict(phi=0x21000, psi=0x22000, Ppair=0x23000, Dop=0x24000, Dnop=0x25000, Stri=0x26000, Iden=0x27000,
         color=0x28000, gt=0x29000, opB=0x2A000, colorT=0x2B000, PpairT=0x2C000, U=0x2D000, phi2T=0x2E000,
         ones=0x2F000, ones1P=0x30000)
S_w, S_dLdC, O_dLdop, O_dLdpsi, DB, DONE = 0x42000, 0x43000, 0x51000, 0x52000, 0x16000, 0x16010
GRAD_SRC = "/home/starboy/bhtop/src/bhtop/kernels/x280/het/opt_grad_step.c"
GOUT, LOUT = 0x30042000, 0x30042800
bf = lambda u: struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]
def enc(flat): return MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
def pad(rc): return SP._pad32(rc)
def dec(a, ctx, c): return MM.untilize32(MM.unpack_bf16_words(rds(c, a, word_count=512, context=ctx)))
grp = [[(x, y) for y in range(SIZE) for x in range(SIZE)][i:i + 32] for i in range(0, SIZE * SIZE, 32)]


def all_consts(gs):
    order = sorted(range(K), key=lambda i: gs[i][9]); gso = [gs[i] for i in order]
    psi, Ppair, Dop, Dnop, Mcomb, color = SP._consts(gso, K)
    Stri = [Mcomb[r] for r in range(K)]; op = [Dop[k][k] for k in range(K)]
    colorT = [[color[k][r] for k in range(K)] for r in range(3)]
    PpairT = [[Ppair[r][c] for r in range(2 * K)] for c in range(K)]
    opB = [[(op[k] if k < K else 0.5) for k in range(32)] for _ in range(P)]
    psi_rows = [[psi[r][c] for c in range(2 * K)] for r in range(3)]
    U = [[1.0 if j > i else 0.0 for i in range(K)] for j in range(K)]
    return dict(psi=psi_rows, Ppair=Ppair, Dop=Dop, Dnop=Dnop, Stri=Stri, color=color, colorT=colorT,
                PpairT=PpairT, opB=opB, Iden=[[1.0 if r == c else 0.0 for c in range(32)] for r in range(32)],
                U=U, ones=[[1.0] * 32 for _ in range(32)], ones1P=[[1.0] * P])


def main():
    pr = lambda *a: print(*a, flush=True)
    ctx = init_ttexalens(); coord = TensixLauncher.at(1, 2, ctx=ctx).coord
    dev = L2cpu(ctx=ctx)
    try: dev.bringup(0)
    except Exception: pass
    gs = SP.scene_rgb(k=K, seed=7, span=float(SIZE)); C = all_consts(gs)

    llk_run.build("resident_train_perf", run_type="L1_TO_L1", fidelity="HiFi4", fp32_acc=False,
                  formats=None, overrides={"ITERATIONS": "constexpr int ITERATIONS = 32;"})
    boot_resident("resident_train_perf", coord, ctx=ctx, runtime_words=[1, 1, 1, 1, 1, 128, 128, 0, 4, 4], clear_words=56)
    time.sleep(0.3)
    for nm in C: wr(coord, H[nm], enc(pad(C[nm])), context=ctx)

    g = grp[3]
    phi = pad([[float(x), float(y), 1.0] for (x, y) in g])
    phi2T = pad([[[2.0 * x, 2.0 * y, 2.0][r] for (x, y) in g] for r in range(3)])
    gt = pad([[0.3, 0.5, 0.2] for _ in range(len(g))])
    wr(coord, H["phi"], enc(phi), context=ctx); wr(coord, H["phi2T"], enc(phi2T), context=ctx)
    wr(coord, H["gt"], enc(gt), context=ctx)
    ring = rd(coord, DONE, context=ctx) + 1; wr(coord, DB, [ring], context=ctx)
    t0 = time.time()
    while time.time() - t0 < 4.0 and rd(coord, DONE, context=ctx) != ring: time.sleep(0.002)

    # host extraction (baseline)
    dp = dec(O_dLdpsi, ctx, coord); do = dec(O_dLdop, ctx, coord)
    wv = dec(S_w, ctx, coord); dC = dec(S_dLdC, ctx, coord)
    host_g = np.zeros((K, 9))
    for i in range(K):
        host_g[i, 0] = dp[0 * 32 + 2 * i]; host_g[i, 1] = dp[1 * 32 + 2 * i]; host_g[i, 2] = dp[2 * 32 + 2 * i]
        host_g[i, 3] = dp[1 * 32 + 2 * i + 1]; host_g[i, 4] = dp[2 * 32 + 2 * i + 1]; host_g[i, 5] = do[i]
        for ch in range(3): host_g[i, 6 + ch] = sum(wv[p * 32 + i] * dC[p * 32 + ch] for p in range(32))
    host_sse = sum(dC[p * 32 + ch] ** 2 for p in range(32) for ch in range(3))

    # worker cb_writer -> x280 GDDR inbox
    bm = BareMetal(1, 2, ctx=ctx, risc="brisc")
    bm.run(BareMetal.build("cb_writer"), params=[bm_coord(8, 3), 0, 0, 0]); time.sleep(0.1)

    # x280 opt_grad_step: detilize + extract + SSE
    dev.wr(0, 0x30005000, [K]); dev.wr(0, 0x30004000, [0]); dev.wr(0, 0x30004010, [0])
    dev.load(0, 0, tc.compile_source(GRAD_SRC, base=CODE_ADDR, march="rv64gc"))
    for _ in range(80):
        if dev.telemetry(0, slots=1, hart=0)[0] == 0x47524144: break
        time.sleep(0.03)
    else: pr("FAIL: opt_grad_step no GRAD"); return
    xr = dev.rd(0, 0x30004000) + 1; dev.wr(0, 0x30004000, [xr])
    while dev.rd(0, 0x30004010) != xr: time.sleep(0.002)
    xt = dev.telemetry(0, slots=4, hart=0)

    dev_g = np.array([bf(u) for u in dev.rdn(0, GOUT, K * 9)]).reshape(K, 9)
    dev_sse = bf(dev.rdn(0, LOUT, 1)[0])
    d = np.abs(dev_g - host_g); rel = d / (np.abs(host_g) + 1e-4)
    pr(f"[x280 grad-consume] extracted {K} Gaussians in {xt[2]:,} cyc")
    pr(f"  per-Gaussian grad: max|abs Δ|={d.max():.2e}  max|rel Δ|={rel.max():.2e}")
    pr(f"  scalar SSE loss: x280={dev_sse:.5f} host={host_sse:.5f} relΔ={abs(dev_sse-host_sse)/max(host_sse,1e-6):.2e}")
    ok = rel.max() < 1e-2 and abs(dev_sse - host_sse) / max(host_sse, 1e-6) < 1e-2
    print("LEVER3_OK — x280 consumes grads + loss on-device; host reads only the scalar loss" if ok else "LEVER3_FAIL")


if __name__ == "__main__":
    main()
