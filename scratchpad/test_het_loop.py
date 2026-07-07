"""THE INTEGRATED FULLY-ON-DEVICE LOOP (single tile). Per step, the host issues ONLY doorbells and reads ONE
scalar loss; every byte of Gaussian data stays on the NoC:
  het_x280 cmd2 project+whiten resident params -> coeff buffer
  per tile: cmd5 produce operands -> worker cb_reader -> render (8 groups) -> worker cb_writer -> cmd6 consume
  cmd1 Adam over resident params
Runs the render on a RIGHT-block worker (NUMA-local to the x280's x=9 bank). Convergence = the on-device loss
falling. Only remaining host data = phi/phi2T/gt (pixel/target; lever 1b-ii moves those on-device too).

  usage: test_het_loop.py [STEPS]   (default 15)
"""
import sys, struct, math, time
sys.path.insert(0, "/home/starboy/bhtop/src"); sys.path.insert(0, "/home/starboy/bhtop/scratchpad")
import numpy as np
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device as rd, read_words_from_device as rds, write_words_to_device as wr
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
from bhtop.tensix.loader import worker_coord
from bhtop.tensix import splat as SP, matmul as MM, llk_run
from bhtop.tensix.resident import boot_resident
from bhtop.tensix.baremetal import BareMetal, bm_coord
import gap1_proj_golden as PG

K, SIZE, P, STEPS = 12, 16, 32, (int(sys.argv[1]) if len(sys.argv) > 1 else 15)
WX, WY = 11, 2                                        # RIGHT-block worker (NUMA-local to x=9 / the x280)
SRC = "/home/starboy/bhtop/src/bhtop/kernels/x280/het/het_x280.c"
H = dict(phi=0x21000, psi=0x22000, Ppair=0x23000, Dop=0x24000, Dnop=0x25000, Stri=0x26000, Iden=0x27000,
         color=0x28000, gt=0x29000, opB=0x2A000, colorT=0x2B000, PpairT=0x2C000, U=0x2D000, phi2T=0x2E000,
         ones=0x2F000, ones1P=0x30000)
S_C, DB, DONE = 0x42800, 0x16000, 0x16010
X_HDR, X_CAM, X_IDL, X_LOSS, X_DB, X_DONE, X_CMD = 0x30005000, 0x30005060, 0x300050A0, 0x30005B00, 0x30004000, 0x30004010, 0x30004020
PARAM, OPBASE, GINBOX = 0x30100000, 0x30080000, 0x300C0000
Rv = np.eye(3); tv = np.array([0.0, 0.0, 6.0]); fx = fy = 26.0; cx = cy = 8.0
fb = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]
bff = lambda u: struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]
def enc(flat): return MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
def pad(rc): return SP._pad32(rc)
def dec(a, ctx, c): return MM.untilize32(MM.unpack_bf16_words(rds(c, a, word_count=512, context=ctx)))
grp = [[(x, y) for y in range(SIZE) for x in range(SIZE)][i:i + 32] for i in range(0, SIZE * SIZE, 32)]


def rand_scene3d(seed):
    rng = np.random.default_rng(seed)
    mean = rng.normal(0, 0.7, (K, 3)); sl = rng.normal(-1.9, 0.2, (K, 3))
    q = rng.normal(0, 1, (K, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    op = rng.uniform(0.4, 0.9, (K, 1)); col = rng.uniform(0.1, 0.9, (K, 3))
    return np.concatenate([mean, sl, q, op, col], axis=1)


def gs_of(param):
    gs = []
    for o in range(K):
        gx, gy, dep, a, b, c, _ = PG.project_forward(param[o, :3], param[o, 3:6], param[o, 6:10], Rv, tv, fx, fy, cx, cy)
        gs.append((gx, gy, a, b, c, float(param[o, 10]), *[float(x) for x in param[o, 11:14]], dep))
    return gs


def main():
    pr = lambda *a: print(*a, flush=True)
    ctx = init_ttexalens(); coord = worker_coord(ctx, WX, WY)
    dev = L2cpu(ctx=ctx)
    try: dev.bringup(0)
    except Exception: pass
    pr(f"[setup] render worker = noc0 ({WX},{WY}) [right block]; x280 hub (8,3)")

    tgt = rand_scene3d(11); init = rand_scene3d(22)
    target = np.array(SP._golden_render(gs_of(tgt), SIZE)).reshape(SIZE, SIZE, 3)

    # boot render kernel + STATIC operands (staged once) on the worker
    llk_run.build("resident_train_perf", run_type="L1_TO_L1", fidelity="HiFi4", fp32_acc=False,
                  formats=None, overrides={"ITERATIONS": "constexpr int ITERATIONS = 32;"})
    boot_resident("resident_train_perf", coord, ctx=ctx, runtime_words=[1, 1, 1, 1, 1, 128, 128, 0, 4, 4], clear_words=56)
    time.sleep(0.3)
    Ppair = [[0.0] * K for _ in range(2 * K)]
    for i in range(K): Ppair[2 * i][i] = -0.5; Ppair[2 * i + 1][i] = -0.5
    Mcomb = [[(1.0 if r < c else 0.0) for c in range(K)] for r in range(2 * K)]
    for i in range(K): Mcomb[K + i][i] = 1.0
    Stri = [Mcomb[r] for r in range(K)]; PpairT = [[Ppair[r][c] for r in range(2 * K)] for c in range(K)]
    U = [[1.0 if j > i else 0.0 for i in range(K)] for j in range(K)]
    for nm, m in [("Ppair", Ppair), ("Stri", Stri), ("PpairT", PpairT), ("U", U)]:
        wr(coord, H[nm], enc(pad(m)), context=ctx)
    wr(coord, H["Iden"], enc([1.0 if r == c else 0.0 for r in range(32) for c in range(32)]), context=ctx)
    wr(coord, H["ones"], enc([1.0] * 1024), context=ctx); wr(coord, H["ones1P"], enc(pad([[1.0] * P])), context=ctx)

    # boot het_x280 with resident params + ZEROED Adam state (m/v/gacc) — het reads these from GDDR
    dev.wr(0, X_HDR, [K, 0])
    dev.wr(0, X_CAM, [fb(x) for x in (list(Rv.flatten()) + list(tv) + [fx, fy, cx, cy])])
    dev.wr(0, PARAM, [fb(init[o, j]) for o in range(K) for j in range(14)])
    M = PARAM + K * 14 * 4; V = M + K * 14 * 4; GACC = V + K * 14 * 4
    dev.wr(0, M, [0] * (K * 14)); dev.wr(0, V, [0] * (K * 14)); dev.wr(0, GACC, [0] * (K * 9))
    dev.wr(0, X_HDR + 21*4, [0])  # slot=0 (single tile)
    dev.wr(0, X_DB, [0]); dev.wr(0, X_DONE, [0])
    dev.load(0, 0, tc.compile_source(SRC, base=CODE_ADDR, march="rv64gc"))
    for _ in range(80):
        if dev.telemetry(0, slots=1, hart=0)[0] == 0x48455421: break
        time.sleep(0.03)
    else: pr("FAIL: het_x280 not resident"); return
    pr("[setup] het_x280 hub + render worker booted")

    rdr = BareMetal(WX, WY, ctx=ctx, risc="brisc")
    read_bin, write_bin = BareMetal.build("cb_reader"), BareMetal.build("cb_writer")

    def het(cmd, extra_hdr=None):
        if extra_hdr: dev.wr(0, X_HDR, extra_hdr)
        dev.wr(0, X_CMD, [cmd]); r = dev.rd(0, X_DB) + 1; dev.wr(0, X_DB, [r])
        while dev.rd(0, X_DONE) != r: time.sleep(0.001)

    for step in range(1, STEPS + 1):
        t0 = time.time()
        het(2, [K, step])                                       # project+whiten (zeros gacc+loss)
        dev.wr(0, X_IDL, [K] + list(range(K)))                  # one tile = all K Gaussians
        het(5)                                                  # produce operands (writes ORDER)
        rdr.run(read_bin, params=[bm_coord(8, 3), OPBASE, 0, 0])  # worker BRISC streams operands in
        for gi, g in enumerate(grp):
            phi = pad([[float(x), float(y), 1.0] for (x, y) in g])
            phi2T = pad([[[2.0 * x, 2.0 * y, 2.0][r] for (x, y) in g] for r in range(3)])
            gt_g = pad([[target[g[p][1], g[p][0]][ch] for ch in range(3)] for p in range(len(g))])
            wr(coord, H["phi"], enc(phi), context=ctx); wr(coord, H["phi2T"], enc(phi2T), context=ctx)
            wr(coord, H["gt"], enc(gt_g), context=ctx)
            ring = rd(coord, DONE, context=ctx) + 1; wr(coord, DB, [ring], context=ctx)
            tt = time.time()
            while time.time() - tt < 4.0 and rd(coord, DONE, context=ctx) != ring: time.sleep(0.001)
            rdr.run(write_bin, params=[bm_coord(8, 3), GINBOX, 0, 0])   # grads -> het inbox
            het(6)                                              # het consume this group -> scatter-add gacc
        bc1 = 1.0 / (1 - 0.9 ** step); bc2 = 1.0 / (1 - 0.999 ** step)
        LR = [0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05]
        gacc0 = [round(bff(u), 4) for u in dev.rdn(0, PARAM + (K * 14 + K * 14 + K * 14) * 4, 4)]  # gacc[0][0:4]
        het(1, [K, step, fb(bc1), fb(bc2), fb(0.9), fb(0.999), fb(1e-8)] + [fb(x) for x in LR])  # Adam
        p0 = [round(bff(u), 4) for u in dev.rdn(0, PARAM, 4)]   # param[0][0:4]
        loss = bff(dev.rdn(0, X_LOSS, 1)[0])                    # host reads ONE float
        psnr = 99.0 if loss < 1e-9 else 10 * math.log10((SIZE * SIZE * 3) / loss)
        pr(f"  step {step:2d}: loss={loss:.5f} ~PSNR={psnr:5.2f}  gacc0={gacc0}  p0={p0}  ({time.time()-t0:.1f}s)")

    pr("done — FULLY ON-DEVICE loop: host issued only doorbells + read 1 scalar/step; all Gaussian data on NoC")


if __name__ == "__main__":
    main()
