"""Lever 1b — the x280 PRODUCES all 6 dynamic Gaussian operands (cb_operands) in its GDDR, and the worker's
BRISC (cb_reader) streams them into L1 in ONE run. Static operands (Ppair/Stri/PpairT/Iden/U/ones) are staged
once at boot. The render matches the host-staged baseline, and the host stages ZERO per-tile Gaussian-operand
tiles — only the tiny per-Gaussian coeffs go to the x280 (which in the full flow come from resident params).
This is the -render_stage_consts lever on silicon.  exalens = orchestration + telemetry only for the operands."""
import sys, struct, math, time
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
S_C, DB, DONE = 0x42800, 0x16000, 0x16010
ZIN, PIN = 0x30002300, 0x30002400
STATIC = ["Ppair", "Stri", "PpairT", "Iden", "U", "ones", "ones1P"]     # staged once
DYNAMIC = ["psi", "Dop", "Dnop", "color", "colorT", "opB"]              # streamed by cb_reader
OPR_SRC = "/home/starboy/bhtop/src/bhtop/kernels/x280/het/cb_operands.c"
fbits = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]
def bf16(x): b = fbits(x); b += 0x7FFF + ((b >> 16) & 1); return (b >> 16) & 0xFFFF
def enc(flat): return MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
def pad(rc): return SP._pad32(rc)
def dec(a, ctx, c): return MM.untilize32(MM.unpack_bf16_words(rds(c, a, word_count=512, context=ctx)))
grp = [[(x, y) for y in range(SIZE) for x in range(SIZE)][i:i + 32] for i in range(0, SIZE * SIZE, 32)]


def whiten(g):
    gx, gy, a, b, c = g[0], g[1], g[2], g[3], g[4]
    sa = math.sqrt(max(a, 1e-8)); m12 = b / sa; m22 = math.sqrt(max(c - b * b / a, 0.0))
    return sa, m12, m22, -(sa * gx + m12 * gy), -(m22 * gy)


def all_consts(gs):
    order = sorted(range(K), key=lambda i: gs[i][9]); gso = [gs[i] for i in order]
    psi, Ppair, Dop, Dnop, Mcomb, color = SP._consts(gso, K)
    Stri = [Mcomb[r] for r in range(K)]
    op = [Dop[k][k] for k in range(K)]
    colorT = [[color[k][r] for k in range(K)] for r in range(3)]
    PpairT = [[Ppair[r][c] for r in range(2 * K)] for c in range(K)]
    opB = [[(op[k] if k < K else 0.5) for k in range(32)] for _ in range(P)]
    psi_rows = [[psi[r][c] for c in range(2 * K)] for r in range(3)]
    U = [[1.0 if j > i else 0.0 for i in range(K)] for j in range(K)]
    return dict(psi=psi_rows, Ppair=Ppair, Dop=Dop, Dnop=Dnop, Stri=Stri, color=color, colorT=colorT,
                PpairT=PpairT, opB=opB, Iden=[[1.0 if r == c else 0.0 for c in range(32)] for r in range(32)],
                U=U, ones=[[1.0] * 32 for _ in range(32)], ones1P=[[1.0] * P])


def render_once(coord, ctx):
    rgb = [[0.0, 0.0, 0.0] for _ in range(SIZE * SIZE)]
    for gi, g in enumerate(grp):
        phi = pad([[float(x), float(y), 1.0] for (x, y) in g])
        phi2T = pad([[[2.0 * x, 2.0 * y, 2.0][r] for (x, y) in g] for r in range(3)])
        wr(coord, H["phi"], enc(phi), context=ctx); wr(coord, H["phi2T"], enc(phi2T), context=ctx)
        wr(coord, H["gt"], enc(pad([[0.0, 0.0, 0.0] for _ in range(len(g))])), context=ctx)
        ring = rd(coord, DONE, context=ctx) + 1; wr(coord, DB, [ring], context=ctx)
        t0 = time.time()
        while time.time() - t0 < 4.0 and rd(coord, DONE, context=ctx) != ring: time.sleep(0.002)
        Cc = dec(S_C, ctx, coord)
        for p in range(len(g)): rgb[g[p][1] * SIZE + g[p][0]] = [Cc[p * 32 + ch] for ch in range(3)]
    return np.array(rgb)


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
    for nm in STATIC: wr(coord, H[nm], enc(pad(C[nm])), context=ctx)      # static staged ONCE
    pr("[setup] booted; static operands staged once")

    # baseline: also host-stage the 6 dynamic, render
    for nm in DYNAMIC: wr(coord, H[nm], enc(pad(C[nm])), context=ctx)
    base = render_once(coord, ctx)
    pr(f"[baseline] host-staged dynamic: rgb[min={base.min():.3f} max={base.max():.3f}]")

    # test: x280 produces the 6 dynamic (cb_operands), BRISC cb_reader streams them, host stages NONE
    dev.wr(0, ZIN, [K] + [fbits(gs[i][9]) for i in range(K)])
    pin = []
    for i in range(K):
        sa, m12, m22, c1, c2 = whiten(gs[i])
        pin += [bf16(v) for v in (sa, m12, m22, c1, c2, gs[i][5], gs[i][6], gs[i][7], gs[i][8])]
    dev.wr(0, PIN, pin)
    coeff_words = 1 + K + len(pin)                       # exalens data to x280 this tile
    dev.load(0, 0, tc.compile_source(OPR_SRC, base=CODE_ADDR, march="rv64gc"))
    for _ in range(80):
        if dev.telemetry(0, slots=1, hart=0)[0] == 0x4F505253: break
        time.sleep(0.03)
    else: pr("FAIL: cb_operands no OPRS"); return
    for nm in DYNAMIC: wr(coord, H[nm], [0xBADF00D5] * 512, context=ctx)   # poison L1 -> must be refilled by reader
    bm = BareMetal(1, 2, ctx=ctx, risc="brisc")
    bm.run(BareMetal.build("cb_reader"), params=[bm_coord(8, 3), 0, 0, 0])
    time.sleep(0.1)
    dbg = rds(coord, 0x2100, word_count=3, context=ctx)
    pr(f"[cb_reader] BRISC streamed 6 operands x280->worker: dbg(coord,n,resp)={[hex(x) for x in dbg]}")
    test = render_once(coord, ctx)

    d = np.abs(test - base)
    pr(f"[test] streamed-operand render vs baseline: max|Δ|={d.max():.5f}  mean|Δ|={d.mean():.6f}")
    pr(f"[traffic] per-tile operand DATA over exalens: baseline stages 6 tiles = {6*512} words; "
       f"streamed path = {coeff_words} coeff words to x280 ({6*512/max(coeff_words,1):.0f}x less), "
       f"operands move x280->worker on the NoC.")
    ok = d.max() < 1e-3
    print("LEVER1B_OK — all 6 Gaussian operands produced on x280 + streamed to worker; host stages 0 operand tiles"
          if ok else "LEVER1B_FAIL")


if __name__ == "__main__":
    main()
