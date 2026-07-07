"""FULL FUSED RESIDENT TRAINING STEP: forward render + dLdC=C-gt + backward, all in ONE ring on one
worker (resident_train_perf). Verify the RGB (forward) AND the leaf grads (backward) against an exact-float
golden of the whole step. One pixel-group (P=32, K=16).

gt = 0.7*golden_C so dLdC is nonzero. opB col-padding = 0.5 keeps the on-device alpha=ar*opB finite in the
padding (else 1/alpha -> inf -> nan through the contractions). P=32 => no row padding.

Run: cd ~/bhtop && .venv/bin/python scratchpad/test_resident_train.py
"""
import sys, math, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device as rd, read_words_from_device as rds, write_words_to_device as wr
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import matmul as MM, llk_run, splat as SP
from bhtop.tensix.resident import boot_resident

DB, DONE, HB, DBG_U, DBG_M, DBG_P = 0x16000, 0x16010, 0x16020, 0x16030, 0x16040, 0x16050
H = dict(phi=0x21000, psi=0x22000, Ppair=0x23000, Dop=0x24000, Dnop=0x25000, Stri=0x26000, Iden=0x27000,
         color=0x28000, gt=0x29000, opB=0x2A000, colorT=0x2B000, PpairT=0x2C000, U=0x2D000, phi2T=0x2E000,
         ones=0x2F000, ones1P=0x30000)
S_C, O_dLdop, O_dLdpsi, O_dLdcol = 0x42800, 0x51000, 0x52000, 0x53000
K, SIZE, P = 16, 16, 32


def enc(flat): return MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
def pad(rc): return SP._pad32(rc)
def dec(a, ctx, c): return MM.untilize32(MM.unpack_bf16_words(rds(c, a, word_count=512, context=ctx)))
def mmg(A, B): return MM.matmul_golden(pad(A), pad(B))
def sub(flat, rows, cols): return [[flat[r * 32 + c] for c in range(cols)] for r in range(rows)]
def relerr(dev, gr, rows, cols):
    n = d = 0.0
    for r in range(rows):
        for c in range(cols):
            n += abs(dev[r * 32 + c] - gr[r][c]); d += abs(gr[r][c])
    return n / (d + 1e-12)


def main():
    ctx = init_ttexalens()
    coord = TensixLauncher.at(1, 2, ctx=ctx).coord

    gs = SP.scene_rgb(k=K, seed=5, span=float(SIZE))
    order = sorted(range(K), key=lambda i: gs[i][9])
    gso = [gs[i] for i in order]
    psi, Ppair, Dop, Dnop, Mcomb, color = SP._consts(gso, K)
    Stri = [Mcomb[r] for r in range(K)]; Iden = [Mcomb[K + r] for r in range(K)]
    op = [Dop[k][k] for k in range(K)]
    psi_rows = [[psi[r][c] for c in range(2 * K)] for r in range(3)]
    pixels = [(x, y) for y in range(SIZE) for x in range(SIZE)][:P]
    phi_g = [[float(x), float(y), 1.0] for (x, y) in pixels]

    # ---- exact-float forward (mirrors the kernel's 6 fused stages) ----
    V = mmg(phi_g, psi_rows); Vsq = [v * v for v in V]
    E = mmg(sub(Vsq, 32, 32), Ppair); ar = [math.exp(min(e, 80.0)) for e in E]
    alpha = mmg(sub(ar, 32, 32), Dop); nalpha = mmg(sub(ar, 32, 32), Dnop)
    lpa = [math.log(a) if a > 1e-30 else -80.0 for a in alpha]
    la = [math.log1p(na) if na > -0.999999 else -80.0 for na in nalpha]
    logw = [a + b for a, b in zip(mmg(sub(la, 32, K), Stri), mmg(sub(lpa, 32, K), Iden))]
    w = [math.exp(min(x, 80.0)) for x in logw]
    C = mmg(sub(w, 32, 32), color)
    Cg = sub(C, P, 3)                                  # golden RGB [P][3]
    gt = [[0.7 * Cg[p][c] for c in range(3)] for p in range(P)]

    # ---- exact-float backward (uses the float forward intermediates) ----
    wg = sub(w, P, K); ag = sub(alpha, P, K); arg = sub(ar, P, K); vg = sub(V, P, 2 * K)
    dLdCg = [[Cg[p][c] - gt[p][c] for c in range(3)] for p in range(P)]
    colorT = [[color[k][r] for k in range(K)] for r in range(3)]
    PpairT = [[Ppair[r][c] for r in range(2 * K)] for c in range(K)]
    U = [[1.0 if j > i else 0.0 for i in range(K)] for j in range(K)]
    phi2 = [[2.0 * x, 2.0 * y, 2.0] for (x, y) in pixels]; phi2T = [[phi2[p][r] for p in range(P)] for r in range(3)]
    dw = sub(mmg(dLdCg, colorT), P, K)
    dwW = [[dw[p][k] * wg[p][k] for k in range(K)] for p in range(P)]
    suf = sub(mmg(dwW, U), P, K)
    recA = [[1.0 / ag[p][k] for k in range(K)] for p in range(P)]
    Tv = [[wg[p][k] * recA[p][k] for k in range(K)] for p in range(P)]
    oneMa = [[1.0 - ag[p][k] for k in range(K)] for p in range(P)]
    recOM = [[1.0 / oneMa[p][k] for k in range(K)] for p in range(P)]
    t1 = [[dw[p][k] * Tv[p][k] for k in range(K)] for p in range(P)]
    t2 = [[suf[p][k] * recOM[p][k] for k in range(K)] for p in range(P)]
    dLda = [[t1[p][k] - t2[p][k] for k in range(K)] for p in range(P)]
    dae = [[dLda[p][k] * arg[p][k] for k in range(K)] for p in range(P)]
    dLdop_g = [[sum(dae[p][k] for p in range(P)) for k in range(K)]]
    dLdE = [[dae[p][k] * op[k] for k in range(K)] for p in range(P)]
    dLdVsq = sub(mmg(dLdE, PpairT), P, 2 * K)
    dLdV = [[dLdVsq[p][c] * vg[p][c] for c in range(2 * K)] for p in range(P)]
    dLdpsi_g = sub(mmg(phi2T, dLdV), 3, 2 * K)
    wT = [[wg[p][k] for p in range(P)] for k in range(K)]
    dLdcol_g = sub(mmg(wT, dLdCg), K, 3)

    # ---- build + boot ----
    b = llk_run.build("resident_train_perf", run_type="L1_TO_L1", fidelity="HiFi4", fp32_acc=False,
                      formats=None, overrides={"ITERATIONS": "constexpr int ITERATIONS = 32;"})
    assert b["ok"], b["log"][-2000:]
    boot_resident("resident_train_perf", coord, ctx=ctx,
                  runtime_words=[1, 1, 1, 1, 1, 128, 128, 0, 4, 4], clear_words=56)
    time.sleep(0.3)

    # ---- stage inputs (opB col-padding = 0.5; ones = all 1) ----
    def st(name, rc): wr(coord, H[name], enc(pad(rc)), context=ctx)
    st("phi", phi_g); st("psi", psi_rows); st("Ppair", Ppair); st("Dop", Dop); st("Dnop", Dnop)
    st("Stri", Stri); st("color", color); st("gt", gt)
    wr(coord, H["Iden"], enc([1.0 if r == c else 0.0 for r in range(32) for c in range(32)]), context=ctx)
    opB_pad = [[(op[k] if k < K else 0.5) for k in range(32)] for _ in range(P)]
    wr(coord, H["opB"], enc(pad(opB_pad)), context=ctx)
    st("colorT", colorT); st("PpairT", PpairT); st("U", U); st("phi2T", phi2T)
    wr(coord, H["ones"], enc([1.0] * 1024), context=ctx); st("ones1P", [[1.0] * P])

    # ---- ring (whole training step) ----
    wr(coord, DB, [1], context=ctx)
    t0 = time.time()
    while time.time() - t0 < 10.0 and rd(coord, DONE, context=ctx) != 1:
        time.sleep(0.005)
    print(f"[train] done={rd(coord,DONE,context=ctx)} U={rd(coord,DBG_U,context=ctx)} "
          f"M={rd(coord,DBG_M,context=ctx)} P={rd(coord,DBG_P,context=ctx)}")

    # ---- verify RGB (fwd) + leaf grads (bwd) ----
    Cdev = dec(S_C, ctx, coord)
    mse = sum((Cdev[p * 32 + c] - Cg[p][c]) ** 2 for p in range(P) for c in range(3)) / (P * 3)
    psnr = 99.0 if mse < 1e-12 else 10 * math.log10(1.0 / mse)
    dop, dpsi = dec(O_dLdop, ctx, coord), dec(O_dLdpsi, ctx, coord)
    e_op, e_psi = relerr(dop, dLdop_g, 1, K), relerr(dpsi, dLdpsi_g, 3, 2 * K)
    print(f"[train] forward RGB (S_C) vs golden render: PSNR={psnr:.1f} dB")
    print(f"[train] leaf-grad rel-err vs exact-float golden:")
    print(f"    dLdpsi (geometry) {e_psi:.3e}   dLdop (opacity) {e_op:.3e}   "
          f"[dLdcolor delegated to x280 = wᵀ@dLdC]")
    ok = (rd(coord, DONE, context=ctx) == 1) and psnr >= 35.0 and e_op < 0.2 and e_psi < 0.2
    print(f"[train] FULL FUSED RESIDENT TRAINING STEP (fwd+dLdC+bwd, 1 ring): {'PASS' if ok else 'CHECK'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
