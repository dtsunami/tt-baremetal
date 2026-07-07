"""Lever 2 mechanism — the worker NoC-WRITES its render grads to the x280 GDDR (cb_writer on BRISC), the
reverse of lever 1. Verify the grads land in x280 GDDR bit-identical to what the host reads from worker L1.
This is the path that lets the x280 scatter-add + Adam WITHOUT the host reading grads back (kills the 57%
render_readback). Together with lever 1, the host is out of the per-tile data path entirely."""
import sys, struct, time
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
DB, DONE = 0x16000, 0x16010
# worker-L1 grad tiles and their x280-GDDR inbox slots (must match cb_writer.c)
WSRC = {"dLdpsi": 0x52000, "dLdop": 0x51000, "w": 0x42000, "dLdC": 0x43000}
XDST = {"dLdpsi": 0x30040000, "dLdop": 0x30040800, "w": 0x30041000, "dLdC": 0x30041800}
def enc(flat): return MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
def pad(rc): return SP._pad32(rc)
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

    # render one group with a nonzero gt so grads are nonzero
    g = grp[0]
    phi = pad([[float(x), float(y), 1.0] for (x, y) in g])
    phi2T = pad([[[2.0 * x, 2.0 * y, 2.0][r] for (x, y) in g] for r in range(3)])
    gt = pad([[0.3, 0.5, 0.2] for _ in range(len(g))])
    wr(coord, H["phi"], enc(phi), context=ctx); wr(coord, H["phi2T"], enc(phi2T), context=ctx)
    wr(coord, H["gt"], enc(gt), context=ctx)
    ring = rd(coord, DONE, context=ctx) + 1; wr(coord, DB, [ring], context=ctx)
    t0 = time.time()
    while time.time() - t0 < 4.0 and rd(coord, DONE, context=ctx) != ring: time.sleep(0.002)
    pr("[render] one group done; grads live in worker L1")

    # baseline: host reads grads straight from worker L1
    base = {nm: [w & 0xFFFFFFFF for w in rds(coord, a, word_count=512, context=ctx)] for nm, a in WSRC.items()}

    # poison the x280 GDDR inbox, then have the worker BRISC NoC-write grads there
    for a in XDST.values(): dev.wr(0, a, [0xBADF00D5] * 512)
    bm = BareMetal(1, 2, ctx=ctx, risc="brisc")
    bm.run(BareMetal.build("cb_writer"), params=[bm_coord(8, 3), 0, 0, 0])
    time.sleep(0.1)
    dbg = rds(coord, 0x2100, word_count=3, context=ctx)
    pr(f"[cb_writer] BRISC wrote 4 grad tiles worker->x280 GDDR: dbg(coord,n,resp)={[hex(x) for x in dbg]}")

    # verify: what the x280 GDDR now holds == the worker's grads
    allok = True
    print(f"\n{'grad':>7} {'words match':>12}")
    for nm in WSRC:
        got = [w & 0xFFFFFFFF for w in dev.rdn(0, XDST[nm], 512)]
        mism = sum(1 for k in range(512) if got[k] != base[nm][k]); ok = mism == 0; allok &= ok
        print(f"{nm:>7} {512-mism:>8}/512")
    print("\nLEVER2_OK — worker NoC-wrote grads into x280 GDDR bit-identical; x280 can scatter-add+Adam, host reads 0 grads"
          if allok else "LEVER2_FAIL")


if __name__ == "__main__":
    main()
