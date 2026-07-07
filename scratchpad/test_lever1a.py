"""Lever 1a — prove an operand can reach the render kernel via ON-DEVICE NoC (x280 GDDR -> worker L1)
instead of a host stage. Boot resident_train_perf on a worker (TRISC0/1/2); the worker's BRISC runs the
proven `nocread` kernel to pull the tilized `psi` tile from the x280's local GDDR into H_psi. Compare the
render output to the host-staged baseline. If they match, the host is out of the psi operand path.

  exalens role here: orchestration (doorbells, load nocread) + telemetry only — the psi DATA moves x280->worker
  entirely on the NoC. This is the seam that generalizes to all 9 operands (lever 1b)."""
import sys, struct, math, time
sys.path.insert(0, "/home/starboy/bhtop/src"); sys.path.insert(0, "/home/starboy/bhtop/scratchpad")
import numpy as np
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device as rd, read_words_from_device as rds, write_words_to_device as wr
from bhtop.l2cpu import L2cpu
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import splat as SP, matmul as MM, llk_run
from bhtop.tensix.resident import boot_resident
from bhtop.tensix.baremetal import BareMetal, bm_coord

K, SIZE, P = 12, 16, 32
H = dict(phi=0x21000, psi=0x22000, Ppair=0x23000, Dop=0x24000, Dnop=0x25000, Stri=0x26000, Iden=0x27000,
         color=0x28000, gt=0x29000, opB=0x2A000, colorT=0x2B000, PpairT=0x2C000, U=0x2D000, phi2T=0x2E000,
         ones=0x2F000, ones1P=0x30000)
S_C, DB, DONE = 0x42800, 0x16000, 0x16010
GDDR_PSI = 0x30003000          # where the x280 GDDR holds the tilized psi (cb_operands' PSI slot)
fb = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]


def enc(flat): return MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
def pad(rc): return SP._pad32(rc)
def dec(a, ctx, c): return MM.untilize32(MM.unpack_bf16_words(rds(c, a, word_count=512, context=ctx)))
grp = [[(x, y) for y in range(SIZE) for x in range(SIZE)][i:i + 32] for i in range(0, SIZE * SIZE, 32)]


def build_consts(gs):
    order = sorted(range(K), key=lambda i: gs[i][9]); gso = [gs[i] for i in order]
    psi, Ppair, Dop, Dnop, Mcomb, color = SP._consts(gso, K)
    Stri = [Mcomb[r] for r in range(K)]
    op = [Dop[k][k] for k in range(K)]
    colorT = [[color[k][r] for k in range(K)] for r in range(3)]
    PpairT = [[Ppair[r][c] for r in range(2 * K)] for c in range(K)]
    opB = [[(op[k] if k < K else 0.5) for k in range(32)] for _ in range(P)]
    psi_rows = [[psi[r][c] for c in range(2 * K)] for r in range(3)]
    return dict(psi=psi_rows, Ppair=Ppair, Dop=Dop, Dnop=Dnop, Stri=Stri, color=color,
                colorT=colorT, PpairT=PpairT, opB=opB)


def stage_all(coord, ctx, C):
    for name, m in C.items():
        wr(coord, H[name], enc(pad(m)), context=ctx)


def render_once(coord, ctx):
    """ring all 8 groups over the resident train kernel, return the RGB tile."""
    rgb = [[0.0, 0.0, 0.0] for _ in range(SIZE * SIZE)]
    for gi, g in enumerate(grp):
        phi = pad([[float(x), float(y), 1.0] for (x, y) in g])
        phi2T = pad([[[2.0 * x, 2.0 * y, 2.0][r] for (x, y) in g] for r in range(3)])
        gt_g = pad([[0.0, 0.0, 0.0] for _ in range(len(g))])
        wr(coord, H["phi"], enc(phi), context=ctx); wr(coord, H["phi2T"], enc(phi2T), context=ctx)
        wr(coord, H["gt"], enc(gt_g), context=ctx)
        ring = rd(coord, DONE, context=ctx) + 1; wr(coord, DB, [ring], context=ctx)
        t0 = time.time()
        while time.time() - t0 < 4.0 and rd(coord, DONE, context=ctx) != ring: time.sleep(0.002)
        Cc = dec(S_C, ctx, coord)
        for p in range(len(g)):
            rgb[g[p][1] * SIZE + g[p][0]] = [Cc[p * 32 + ch] for ch in range(3)]
    return np.array(rgb)


def main():
    pr = lambda *a: print(*a, flush=True)
    ctx = init_ttexalens(); pr("[setup] exalens up")
    coord = TensixLauncher.at(1, 2, ctx=ctx).coord
    dev = L2cpu(ctx=ctx)
    try: dev.bringup(0)
    except Exception: pass

    gs = SP.scene_rgb(k=K, seed=7, span=float(SIZE))
    C = build_consts(gs)

    llk_run.build("resident_train_perf", run_type="L1_TO_L1", fidelity="HiFi4", fp32_acc=False,
                  formats=None, overrides={"ITERATIONS": "constexpr int ITERATIONS = 32;"})
    boot_resident("resident_train_perf", coord, ctx=ctx, runtime_words=[1, 1, 1, 1, 1, 128, 128, 0, 4, 4], clear_words=56)
    time.sleep(0.3)
    U = [[1.0 if j > i else 0.0 for i in range(K)] for j in range(K)]
    wr(coord, H["Iden"], enc([1.0 if r == c else 0.0 for r in range(32) for c in range(32)]), context=ctx)
    wr(coord, H["U"], enc(pad(U)), context=ctx); wr(coord, H["ones"], enc([1.0] * 1024), context=ctx)
    wr(coord, H["ones1P"], enc(pad([[1.0] * P])), context=ctx)
    pr("[setup] resident_train_perf booted on worker (1,2)")

    # ---- BASELINE: host stages ALL operands incl psi ----
    stage_all(coord, ctx, C)
    base = render_once(coord, ctx)
    pr(f"[baseline] host-staged render: rgb[min={base.min():.3f} max={base.max():.3f}]")

    # ---- TEST: psi comes from x280 GDDR via nocread on BRISC ----
    psi_words = enc(pad(C["psi"]))                     # the SAME tilized psi, 512 words = 2048 bytes
    dev.wr(0, GDDR_PSI, [w & 0xFFFFFFFF for w in psi_words])   # put psi in x280 local GDDR
    wr(coord, H["psi"], [0xBADF00D5] * 512, context=ctx)       # poison worker's H_psi so a stale read shows
    # BRISC pulls psi from (L2CPU 8,3) GDDR 0x30003000 -> worker H_psi 0x22000, 2048 bytes
    bm = BareMetal(1, 2, ctx=ctx, risc="brisc")
    bm.run(BareMetal.build("nocread"), params=[bm_coord(8, 3), GDDR_PSI, 2048, H["psi"]])
    time.sleep(0.1)
    # verify the operand landed
    got = rds(coord, H["psi"], word_count=512, context=ctx)
    match_l1 = all((got[i] & 0xFFFFFFFF) == (psi_words[i] & 0xFFFFFFFF) for i in range(512))
    dbg = rds(coord, 0x2100, word_count=3, context=ctx)   # nocread BM_DBG: [my_coord, target, responses]
    pr(f"[nocread] BRISC read psi x280->worker: L1 match={match_l1}  dbg(mycoord,target,resp)={[hex(x) for x in dbg]}")

    test = render_once(coord, ctx)
    d = np.abs(test - base)
    pr(f"[test] nocread-psi render vs baseline: max|Δ|={d.max():.5f}  mean|Δ|={d.mean():.6f}")
    ok = match_l1 and d.max() < 1e-3
    print("LEVER1A_OK — psi reached the render via on-device NoC, host out of the psi path" if ok
          else "LEVER1A_FAIL")


if __name__ == "__main__":
    main()
