"""GAP-0 ISOLATION #1: drive resident_backward_perf MULTI-RING on ONE boot.

Backward has matmul + eltwise-binary (EMUL/ESUB) + reciprocal (MREC) but NO forward SFPU (square/exp/
log/log1p). render (matmul+SFPU, no eltwise) already proved 3 doorbell rings @51dB this session. So:
  - if backward STALLS multi-ring  -> the eltwise/recip mix is the cross-ring accumulator (isolated).
  - if backward SURVIVES multi-ring -> the forward SFPU is required for the wedge (interaction), and the
    split point for hetero orchestration is fwd(SFPU) | bwd(eltwise/recip).

Boots once, stages inputs once, then rings 1..N re-poisoning outputs each ring; checks DONE tracks + grads
stay correct. Bounded polls + llk_triage PC snapshot on stall; recover with tt-smi -r 0.

Run: /home/starboy/bhtop/.venv/bin/python scratchpad/test_gap0_bwd_multiring.py [NRINGS]
"""
import sys, math, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device as rd, read_words_from_device as rds, write_words_to_device as wr
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import matmul as MM, llk_run, splat as SP
from bhtop.tensix.resident import boot_resident

NRINGS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
DB, DONE, HB, DBG_U, DBG_M, DBG_P = 0x16000, 0x16010, 0x16020, 0x16030, 0x16040, 0x16050
H = dict(dLdC=0x21000, w=0x22000, alpha=0x23000, ar=0x24000, v=0x25000, colorT=0x26000, PpairT=0x27000,
         U=0x28000, phi2T=0x29000, opB=0x2A000, ones=0x2B000, wT=0x2C000, ones1P=0x2D000, Iden=0x2E000)
O_dLdop, O_dLdpsi, O_dLdcol = 0x51000, 0x52000, 0x53000
POISON = 0xBADF00D5
K, SIZE, P = 16, 16, 32
enc = lambda flat: MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
pad = lambda rc: SP._pad32(rc)
dec = lambda a, ctx, c: MM.untilize32(MM.unpack_bf16_words(rds(c, a, word_count=512, context=ctx)))


def relerr(dev, gr, rows, cols):
    n = d = 0.0
    for r in range(rows):
        for c in range(cols):
            n += abs(dev[r * 32 + c] - gr[r][c]); d += abs(gr[r][c])
    return n / (d + 1e-12)


def snap(rdbg):
    def one(h):
        try:
            if h.is_in_reset(): return "reset"
        except Exception as e: return f"err:{e}"
        try:
            pc = h.get_pc(); time.sleep(0.03); pc2 = h.get_pc()
            return f"pc=0x{pc:x} {'STUCK' if pc == pc2 else 'advancing'} halted={h.is_halted()}"
        except Exception as e: return f"err:{e}"
    return {c: one(rdbg[c]) for c in ("UNPACK", "MATH", "PACK")}


def main():
    pr = lambda *a: print(*a, flush=True)
    ctx = init_ttexalens()
    coord = TensixLauncher.at(1, 2, ctx=ctx).coord
    pr(f"[bwd-mr] driving {NRINGS} rings on ONE boot (no reboot)")

    fwd = SP.render_ondevice(coord, ctx=ctx, k=K, size=SIZE, seed=5, verbose=False)
    w, alpha, ar, v = fwd["w"], fwd["alpha"], fwd["ar"], fwd["v"]
    color, order, gs = fwd["color"], fwd["order"], fwd["gs"]
    gso = [gs[i] for i in order]
    _, Ppair, Dop, _, _, _ = SP._consts(gso, K)
    op = [Dop[k][k] for k in range(K)]
    pixels = [(x, y) for y in range(SIZE) for x in range(SIZE)][:P]
    dLdCg = [[0.3 * math.sin(p + 1) + 0.1 * c for c in range(3)] for p in range(P)]
    wg = [[w[p][k] for k in range(K)] for p in range(P)]; ag = [[alpha[p][k] for k in range(K)] for p in range(P)]
    arg = [[ar[p][k] for k in range(K)] for p in range(P)]; vg = [[v[p][c] for c in range(2 * K)] for p in range(P)]
    colorT = [[color[k][r] for k in range(K)] for r in range(3)]
    PpairT = [[Ppair[r][c] for r in range(2 * K)] for c in range(K)]
    U = [[1.0 if j > i else 0.0 for i in range(K)] for j in range(K)]
    phi2 = [[2.0 * x, 2.0 * y, 2.0] for (x, y) in pixels]; phi2T = [[phi2[p][r] for p in range(P)] for r in range(3)]
    wT = [[wg[p][k] for p in range(P)] for k in range(K)]
    # golden
    def mm(A, B): return MM.matmul_golden(pad(A), pad(B))
    def s2(flat, cols): return [[flat[p * 32 + c] for c in range(cols)] for p in range(P)]
    dw = s2(mm(dLdCg, colorT), K); dwW = [[dw[p][k] * wg[p][k] for k in range(K)] for p in range(P)]
    suf = s2(mm(dwW, U), K); recA = [[1.0 / ag[p][k] for k in range(K)] for p in range(P)]
    Tv = [[wg[p][k] * recA[p][k] for k in range(K)] for p in range(P)]
    oneMa = [[1.0 - ag[p][k] for k in range(K)] for p in range(P)]
    recOM = [[1.0 / oneMa[p][k] for k in range(K)] for p in range(P)]
    t1 = [[dw[p][k] * Tv[p][k] for k in range(K)] for p in range(P)]
    t2 = [[suf[p][k] * recOM[p][k] for k in range(K)] for p in range(P)]
    dLda = [[t1[p][k] - t2[p][k] for k in range(K)] for p in range(P)]
    dae = [[dLda[p][k] * arg[p][k] for k in range(K)] for p in range(P)]
    dLdop_g = [[sum(dae[p][k] for p in range(P)) for k in range(K)]]
    dLdE = [[dae[p][k] * op[k] for k in range(K)] for p in range(P)]
    dLdVsq = s2(mm(dLdE, PpairT), 2 * K); dLdV = [[dLdVsq[p][c] * vg[p][c] for c in range(2 * K)] for p in range(P)]
    dLdVf = mm(phi2T, dLdV); dLdpsi_g = [[dLdVf[r * 32 + c] for c in range(2 * K)] for r in range(3)]

    b = llk_run.build("resident_backward_perf", run_type="L1_TO_L1", fidelity="HiFi4", fp32_acc=False,
                      formats=None, overrides={"ITERATIONS": "constexpr int ITERATIONS = 32;"})
    assert b["ok"], b["log"][-2000:]
    rdbg = boot_resident("resident_backward_perf", coord, ctx=ctx,
                         runtime_words=[1, 1, 1, 1, 1, 128, 128, 0, 4, 4], clear_words=48)
    time.sleep(0.3)

    alpha_pad = [[(ag[p][k] if k < K else 0.5) for k in range(32)] for p in range(P)]
    def stage(name, rows): wr(coord, H[name], enc(pad(rows)), context=ctx)
    stage("dLdC", dLdCg); stage("w", wg); stage("ar", arg); stage("v", vg)
    wr(coord, H["alpha"], enc(pad(alpha_pad)), context=ctx)
    stage("colorT", colorT); stage("PpairT", PpairT); stage("U", U); stage("phi2T", phi2T)
    stage("opB", [[op[k] for k in range(K)] for _ in range(P)])
    wr(coord, H["ones"], enc([1.0] * 1024), context=ctx); stage("wT", wT); stage("ones1P", [[1.0] * P])
    wr(coord, H["Iden"], enc([1.0 if r == c else 0.0 for r in range(32) for c in range(32)]), context=ctx)
    pr("[bwd-mr] booted once, staged once\n")

    npass = 0
    for r in range(1, NRINGS + 1):
        for a in (O_dLdop, O_dLdpsi, O_dLdcol): wr(coord, a, [POISON] * 512, context=ctx)
        wr(coord, DB, [r], context=ctx)
        t0 = time.time()
        while time.time() - t0 < 4.0 and rd(coord, DONE, context=ctx) != r: time.sleep(0.004)
        if rd(coord, DONE, context=ctx) != r:
            pr(f"[ring {r}] *** STALL *** DONE={rd(coord,DONE,context=ctx)} DBG_U=0x{rd(coord,DBG_U,context=ctx):x} "
               f"DBG_M=0x{rd(coord,DBG_M,context=ctx):x}  {snap(rdbg)}")
            pr("[bwd-mr] stop. recover: tt-smi -r 0"); break
        e_op = relerr(dec(O_dLdop, ctx, coord), dLdop_g, 1, K)
        e_psi = relerr(dec(O_dLdpsi, ctx, coord), dLdpsi_g, 3, 2 * K)
        ok = e_op < 0.15 and e_psi < 0.15; npass += ok
        pr(f"[ring {r}] DONE in {time.time()-t0:.2f}s  dLdop={e_op:.2e} dLdpsi={e_psi:.2e} -> {'PASS' if ok else 'CHECK'}")

    pr(f"\n[bwd-mr] {npass}/{NRINGS} rings PASS on ONE boot => "
       f"{'backward IS multi-ring resident (SFPU-fwd is the wedge; split fwd|bwd)' if npass==NRINGS else 'backward STALLS multi-ring (eltwise/recip IS the accumulator)'}")
    return npass == NRINGS


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
