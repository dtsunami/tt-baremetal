"""Fused resident BACKWARD: run splat.backward_ondevice's whole 17-stage chain as ONE resident kernel,
verify the leaf grads (dLdpsi / dLdop / dLdcolor) against an exact-float golden. Standalone: the forward
intermediates (w, alpha, ar, v) are host-staged (from render_ondevice). One pixel-group (P=32, K=16).

Padding: the reciprocal stages would make inf->nan in zero col-padding, which propagates through the
matmul contractions — so alpha/oneMa col-padding (cols>=K) is a finite 0.5 while all data inputs pad 0.
P=32 (=1 group of a 16x16 tile) means no row padding.

Run: cd ~/bhtop && .venv/bin/python scratchpad/test_resident_backward.py
"""
import sys, os, math, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device as rd, read_words_from_device as rds, write_words_to_device as wr
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import matmul as MM, llk_run, splat as SP
from bhtop.tensix.resident import boot_resident

DB, DONE, HB, DBG_U, DBG_M, DBG_P = 0x16000, 0x16010, 0x16020, 0x16030, 0x16040, 0x16050
H = dict(dLdC=0x21000, w=0x22000, alpha=0x23000, ar=0x24000, v=0x25000, colorT=0x26000, PpairT=0x27000,
         U=0x28000, phi2T=0x29000, opB=0x2A000, ones=0x2B000, wT=0x2C000, ones1P=0x2D000, Iden=0x2E000)
O_dLdop, O_dLdpsi, O_dLdcol = 0x51000, 0x52000, 0x53000
K, SIZE, P = 16, 16, 32


def enc(flat):
    return MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])


def pad(rows_cols):
    return SP._pad32(rows_cols)


def dec(addr, ctx, coord):
    return MM.untilize32(MM.unpack_bf16_words(rds(coord, addr, word_count=512, context=ctx)))


def relerr(dev_flat, gold_rows, rows, cols):
    num = den = 0.0
    for r in range(rows):
        for c in range(cols):
            g = gold_rows[r][c]; d = dev_flat[r * 32 + c]
            num += abs(d - g); den += abs(g)
    return num / (den + 1e-12)


def main():
    ctx = init_ttexalens()
    coord = TensixLauncher.at(1, 2, ctx=ctx).coord

    # ---- forward intermediates (group 0) via render_ondevice ----
    fwd = SP.render_ondevice(coord, ctx=ctx, k=K, size=SIZE, seed=5, verbose=False)
    w, alpha, ar, v = fwd["w"], fwd["alpha"], fwd["ar"], fwd["v"]
    color, order, gs = fwd["color"], fwd["order"], fwd["gs"]
    gso = [gs[i] for i in order]
    _, Ppair, Dop, _, _, _ = SP._consts(gso, K)
    op = [Dop[k][k] for k in range(K)]
    pixels = [(x, y) for y in range(SIZE) for x in range(SIZE)][:P]

    # group-0 slices [P][*]
    dLdCg = [[0.3 * math.sin(p + 1) + 0.1 * c for c in range(3)] for p in range(P)]   # arbitrary loss grad
    wg = [[w[p][k] for k in range(K)] for p in range(P)]
    ag = [[alpha[p][k] for k in range(K)] for p in range(P)]
    arg = [[ar[p][k] for k in range(K)] for p in range(P)]
    vg = [[v[p][c] for c in range(2 * K)] for p in range(P)]

    colorT = [[color[k][r] for k in range(K)] for r in range(3)]
    PpairT = [[Ppair[r][c] for r in range(2 * K)] for c in range(K)]
    U = [[1.0 if j > i else 0.0 for i in range(K)] for j in range(K)]
    phi2 = [[2.0 * x, 2.0 * y, 2.0] for (x, y) in pixels]
    phi2T = [[phi2[p][r] for p in range(P)] for r in range(3)]
    wT = [[wg[p][k] for p in range(P)] for k in range(K)]

    # ---- exact-float golden (backward_ondevice math, group 0) ----
    def mm(A, B): return MM.matmul_golden(pad(A), pad(B))
    def sub2(flat, cols): return [[flat[p * 32 + c] for c in range(cols)] for p in range(P)]
    dw = sub2(mm(dLdCg, colorT), K)
    dwW = [[dw[p][k] * wg[p][k] for k in range(K)] for p in range(P)]
    suf = sub2(mm(dwW, U), K)
    recA = [[1.0 / ag[p][k] for k in range(K)] for p in range(P)]
    Tv = [[wg[p][k] * recA[p][k] for k in range(K)] for p in range(P)]
    oneMa = [[1.0 - ag[p][k] for k in range(K)] for p in range(P)]
    recOM = [[1.0 / oneMa[p][k] for k in range(K)] for p in range(P)]
    t1 = [[dw[p][k] * Tv[p][k] for k in range(K)] for p in range(P)]
    t2 = [[suf[p][k] * recOM[p][k] for k in range(K)] for p in range(P)]
    dLda = [[t1[p][k] - t2[p][k] for k in range(K)] for p in range(P)]
    dae = [[dLda[p][k] * arg[p][k] for k in range(K)] for p in range(P)]
    dLdop_g = [[sum(dae[p][k] for p in range(P)) for k in range(K)]]          # [1][K]
    dLdE = [[dae[p][k] * op[k] for k in range(K)] for p in range(P)]
    dLdVsq = sub2(mm(dLdE, PpairT), 2 * K)
    dLdV = [[dLdVsq[p][c] * vg[p][c] for c in range(2 * K)] for p in range(P)]
    dLdVf = mm(phi2T, dLdV); dLdpsi_g = [[dLdVf[r * 32 + c] for c in range(2 * K)] for r in range(3)]
    dLdcf = mm(wT, dLdCg); dLdcol_g = [[dLdcf[k * 32 + c] for c in range(3)] for k in range(K)]

    # ---- build + boot ----
    b = llk_run.build("resident_backward_perf", run_type="L1_TO_L1", fidelity="HiFi4", fp32_acc=False,
                      formats=None, overrides={"ITERATIONS": "constexpr int ITERATIONS = 32;"})
    assert b["ok"], b["log"][-2000:]
    boot_resident("resident_backward_perf", coord, ctx=ctx,
                  runtime_words=[1, 1, 1, 1, 1, 128, 128, 0, 4, 4], clear_words=48)
    time.sleep(0.3)

    # ---- stage inputs (alpha col-padding = 0.5 so 1/alpha stays finite; ones = all 1) ----
    alpha_pad = [[(ag[p][k] if k < K else 0.5) for k in range(32)] for p in range(P)]
    def stage(name, rows):
        wr(coord, H[name], enc(pad(rows)), context=ctx)
    stage("dLdC", dLdCg); stage("w", wg); stage("ar", arg); stage("v", vg)
    wr(coord, H["alpha"], enc(pad(alpha_pad)), context=ctx)
    stage("colorT", colorT); stage("PpairT", PpairT); stage("U", U); stage("phi2T", phi2T)
    stage("opB", [[op[k] for k in range(K)] for _ in range(P)])
    wr(coord, H["ones"], enc([1.0] * 1024), context=ctx)
    stage("wT", wT); stage("ones1P", [[1.0] * P])
    wr(coord, H["Iden"], enc([1.0 if r == c else 0.0 for r in range(32) for c in range(32)]), context=ctx)

    # ---- ring ----
    wr(coord, DB, [1], context=ctx)
    t0 = time.time()
    while time.time() - t0 < 8.0 and rd(coord, DONE, context=ctx) != 1:
        time.sleep(0.005)
    print(f"[bwd] done={rd(coord,DONE,context=ctx)} U={rd(coord,DBG_U,context=ctx)} "
          f"M={rd(coord,DBG_M,context=ctx)} P={rd(coord,DBG_P,context=ctx)}")

    # ---- compare leaf grads ----
    dop = dec(O_dLdop, ctx, coord); dpsi = dec(O_dLdpsi, ctx, coord); dcol = dec(O_dLdcol, ctx, coord)
    e_op = relerr(dop, dLdop_g, 1, K)
    e_psi = relerr(dpsi, dLdpsi_g, 3, 2 * K)
    e_col = relerr(dcol, dLdcol_g, K, 3)
    print(f"[bwd] leaf-grad rel-err vs exact-float golden:")
    print(f"    dLdop    {e_op:.3e}   dev[0,0]={dop[0]:+.4f} gold={dLdop_g[0][0]:+.4f}")
    print(f"    dLdpsi   {e_psi:.3e}   dev[0,0]={dpsi[0]:+.4f} gold={dLdpsi_g[0][0]:+.4f}")
    print(f"    dLdcolor {e_col:.3e}   dev[0,0]={dcol[0]:+.4f} gold={dLdcol_g[0][0]:+.4f}")
    ok = (rd(coord, DONE, context=ctx) == 1) and e_col < 0.05 and e_psi < 0.15 and e_op < 0.15
    print(f"[bwd] FUSED RESIDENT BACKWARD (17 stages, 1 ring): {'PASS' if ok else 'CHECK'} "
          f"(dLdcolor tight; dLdpsi/dLdop carry the known dLda cancellation, avg out in training)")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
