"""GAP-3 full-image multi-tile 3D Gaussian-splat trainer. The single-16x16-tile resident trainer
(train_resident3d) generalised to a W x H image of 16x16 tiles. Each step:
  - x280 projects all N 3D Gaussians (opt_proj_step publish) and Adam-updates them (all on-chip);
  - host bins the projected Gaussians into per-tile depth-sorted subsets (Gap-2 rule, proven on silicon
    in bin_tiles.c; used host-side here purely for dispatch orchestration);
  - each tile renders its top-Ktile subset through the resident Tensix render+backward over the tile's
    screen region; per-Gaussian gradients SCATTER-ADD across every tile that touched the Gaussian;
  - the accumulated per-Gaussian grad drives the x280 Adam step.
Turns 1 tile into a full image. N<=16 here (fits the L2CPU window); Gap 5 moves params to big GDDR for
millions. Ktile fixed = the render kernel K; tiles with fewer Gaussians are opacity-0 padded, with more
are depth-culled to top-Ktile (Gap 4).

  usage: train_multitile3d.py [STEPS] [IMG] [N]     (default 20 48 16)
"""
import sys, struct, math, time
sys.path.insert(0, "/home/starboy/bhtop/src"); sys.path.insert(0, "/home/starboy/bhtop/scratchpad")
import numpy as np
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device as rd, read_words_from_device as rds, write_words_to_device as wr
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import splat as SP, matmul as MM
from bhtop.tensix import llk_run
from bhtop.tensix.resident import boot_resident
import gap1_proj_golden as PG
import gap2_bin_golden as BIN

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 20
IMG = int(sys.argv[2]) if len(sys.argv) > 2 else 48
N = int(sys.argv[3]) if len(sys.argv) > 3 else 16
TILE, KT, P = 16, 12, 32                       # KT = render-kernel K (per-tile capacity)
NTX = IMG // TILE
OPT_SRC = "/home/starboy/bhtop/src/bhtop/kernels/x280/het/opt_proj_step.c"
H = dict(phi=0x21000, psi=0x22000, Ppair=0x23000, Dop=0x24000, Dnop=0x25000, Stri=0x26000, Iden=0x27000,
         color=0x28000, gt=0x29000, opB=0x2A000, colorT=0x2B000, PpairT=0x2C000, U=0x2D000, phi2T=0x2E000,
         ones=0x2F000, ones1P=0x30000)
S_C, S_w, S_dLdC, O_dLdop, O_dLdpsi = 0x42800, 0x42000, 0x43000, 0x51000, 0x52000
DB, DONE = 0x16000, 0x16010
X_HDR, X_CAM, X_ORDER, X_GIN, X_PARAM, X_M, X_V, X_PUB = 0x30005000, 0x30005060, 0x300050A0, 0x30005100, 0x30005800, 0x30006000, 0x30006800, 0x30007000
X_DB, X_DONE, X_CMD = 0x30004000, 0x30004010, 0x30004020
fb = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]
bf = lambda u: struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]
DUMMY = (-100.0, -100.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1e9)     # opacity-0, off-tile pad

Rv = np.eye(3); tv = np.array([0.0, 0.0, 4.0]); fx = fy = 60.0
cx = cy = IMG / 2.0
CAM = (Rv, tv, fx, fy, cx, cy)


def enc(flat): return MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
def pad(rc): return SP._pad32(rc)
def dec(a, ctx, c): return MM.untilize32(MM.unpack_bf16_words(rds(c, a, word_count=512, context=ctx)))


def rand_scene3d(seed, n):
    rng = np.random.default_rng(seed)
    mean = rng.normal(0, 1.3, (n, 3)); sl = rng.normal(-1.9, 0.25, (n, 3))
    q = rng.normal(0, 1, (n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    op = rng.uniform(0.4, 0.9, (n, 1)); col = rng.uniform(0.1, 0.9, (n, 3))
    return np.concatenate([mean, sl, q, op, col], axis=1)


def project_all(param):
    Rvm, tvm, fxm, fym, cxm, cym = CAM
    gx = np.zeros(N); gy = np.zeros(N); conic = np.zeros((N, 3)); depth = np.zeros(N); gs = []
    for o in range(N):
        u, v, dep, a, b, c, _ = PG.project_forward(param[o, :3], param[o, 3:6], param[o, 6:10], Rvm, tvm, fxm, fym, cxm, cym)
        gx[o], gy[o], conic[o], depth[o] = u, v, (a, b, c), dep
        gs.append((u, v, a, b, c, float(param[o, 10]), *[float(x) for x in param[o, 11:14]], dep))
    return gx, gy, conic, depth, gs


def golden_image(param):
    """full IMG x IMG host render of all N projected Gaussians (target)."""
    _, _, _, _, gs = project_all(param)
    return np.array(SP._golden_render(gs, IMG)).reshape(IMG, IMG, 3)


def tile_subset(gs, tile_ids):
    """pad/cull a tile's depth-sorted id list to exactly KT gs-tuples; return (gs_tile, real_ids)."""
    ids = list(tile_ids[:KT])
    gs_tile = [gs[i] for i in ids]
    real = list(ids)
    while len(gs_tile) < KT:
        gs_tile.append(DUMMY); real.append(-1)
    return gs_tile, real


def stage_consts(coord, ctx, gs_tile):
    order = list(range(KT))                    # gs_tile already depth-sorted
    gso = [gs_tile[i] for i in order]
    psi, Ppair, Dop, Dnop, Mcomb, color = SP._consts(gso, KT)
    Stri = [Mcomb[r] for r in range(KT)]
    op = [Dop[k][k] for k in range(KT)]
    colorT = [[color[k][r] for k in range(KT)] for r in range(3)]
    PpairT = [[Ppair[r][c] for r in range(2 * KT)] for c in range(KT)]
    opB = [[(op[k] if k < KT else 0.5) for k in range(32)] for _ in range(P)]
    psi_rows = [[psi[r][c] for c in range(2 * KT)] for r in range(3)]
    for name, m in [("psi", psi_rows), ("Ppair", Ppair), ("Dop", Dop), ("Dnop", Dnop), ("Stri", Stri),
                    ("color", color), ("colorT", colorT), ("PpairT", PpairT), ("opB", opB)]:
        wr(coord, H[name], enc(pad(m)), context=ctx)


def main():
    pr = lambda *a: print(*a, flush=True)
    ctx = init_ttexalens(); pr("[setup] exalens up")
    coord = TensixLauncher.at(1, 2, ctx=ctx).coord
    dev = L2cpu(ctx=ctx)
    try: dev.bringup(0); pr("[setup] x280 bringup ok")
    except Exception as e: pr("[setup] x280 bringup:", type(e).__name__, "(already up)")

    # 32-pixel groups of a 16x16 tile, in tile-local coords
    grp = [[(x, y) for y in range(TILE) for x in range(TILE)][i:i + 32] for i in range(0, TILE * TILE, 32)]
    U = [[1.0 if j > i else 0.0 for i in range(KT)] for j in range(KT)]

    tgt = rand_scene3d(11, N); init = rand_scene3d(22, N)
    target = golden_image(tgt)                                    # [IMG,IMG,3]

    llk_run.build("resident_train_perf", run_type="L1_TO_L1", fidelity="HiFi4", fp32_acc=False,
                  formats=None, overrides={"ITERATIONS": "constexpr int ITERATIONS = 32;"}); pr("[setup] render kernel built")
    boot_resident("resident_train_perf", coord, ctx=ctx,
                  runtime_words=[1, 1, 1, 1, 1, 128, 128, 0, 4, 4], clear_words=56)
    time.sleep(0.3); pr("[setup] render kernel booted")
    wr(coord, H["Iden"], enc([1.0 if r == c else 0.0 for r in range(32) for c in range(32)]), context=ctx)
    wr(coord, H["U"], enc(pad(U)), context=ctx)
    wr(coord, H["ones"], enc([1.0] * 1024), context=ctx)
    wr(coord, H["ones1P"], enc(pad([[1.0] * P])), context=ctx)

    dev.wr(0, X_HDR, [N, 0])
    dev.wr(0, X_CAM, [fb(x) for x in (list(Rv.flatten()) + list(tv) + [fx, fy, cx, cy])])
    dev.wr(0, X_ORDER, list(range(N)))
    dev.wr(0, X_PARAM, [fb(init[o, j]) for o in range(N) for j in range(14)])
    dev.wr(0, X_M, [0] * (N * 14)); dev.wr(0, X_V, [0] * (N * 14)); dev.wr(0, X_PUB, [0] * (N * 6))
    dev.wr(0, X_DB, [0]); dev.wr(0, X_DONE, [0]); dev.wr(0, X_CMD, [1])
    pr("[setup] loading x280 opt_proj_step")
    dev.load(0, 0, tc.compile_source(OPT_SRC, base=CODE_ADDR, march="rv64gc"))
    for _ in range(80):
        if dev.telemetry(0, slots=1, hart=0)[0] == 0x4F505421: break
        time.sleep(0.03)
    else:
        pr("FAIL: opt_proj_step not resident"); return
    pr("[setup] x280 opt_proj_step resident")

    ring = 0; xstep = 0
    pr(f"GAP-3 multi-tile 3D trainer: N={N} Gaussians, {IMG}x{IMG} image = {NTX}x{NTX} tiles, "
       f"Ktile={KT}, {STEPS} steps")
    for step in range(1, STEPS + 1):
        t0 = time.time()
        param = np.array([[bf(u) for u in dev.rdn(0, X_PARAM + o * 14 * 4, 14)] for o in range(N)])
        # x280 project-only publish for the (single) camera
        xstep += 1; dev.wr(0, X_CMD, [2]); dev.wr(0, X_DB, [xstep])
        for _ in range(200):
            if dev.rd(0, X_DONE) == xstep: break
            time.sleep(0.002)
        pub = np.array([[bf(u) for u in dev.rdn(0, X_PUB + o * 6 * 4, 6)] for o in range(N)])
        gs = [(pub[o, 0], pub[o, 1], pub[o, 2], pub[o, 3], pub[o, 4],
               float(param[o, 10]), *[float(x) for x in param[o, 11:14]], pub[o, 5]) for o in range(N)]
        tiles, ntx, nty = BIN.bin_tiles(pub[:, 0], pub[:, 1], pub[:, 2:5], pub[:, 5], IMG, IMG, tile=TILE, cap=64)

        gacc = np.zeros((N, 9))
        rgb_full = np.zeros((IMG, IMG, 3))
        for t in range(ntx * nty):
            gs_tile, real = tile_subset(gs, tiles[t])
            if all(r < 0 for r in real):                         # empty tile -> stays background
                continue
            ox, oy = (t % ntx) * TILE, (t // ntx) * TILE
            stage_consts(coord, ctx, gs_tile)
            for gi, g in enumerate(grp):
                phi = pad([[float(ox + x), float(oy + y), 1.0] for (x, y) in g])
                phi2T = pad([[[2.0 * (ox + x), 2.0 * (oy + y), 2.0][r] for (x, y) in g] for r in range(3)])
                gt_g = pad([[target[oy + g[p][1], ox + g[p][0]][ch] for ch in range(3)] for p in range(len(g))])
                wr(coord, H["phi"], enc(phi), context=ctx)
                wr(coord, H["phi2T"], enc(phi2T), context=ctx)
                wr(coord, H["gt"], enc(gt_g), context=ctx)
                ring += 1; wr(coord, DB, [ring], context=ctx)
                tt = time.time()
                while time.time() - tt < 4.0 and rd(coord, DONE, context=ctx) != ring:
                    time.sleep(0.003)
                C = dec(S_C, ctx, coord); wv = dec(S_w, ctx, coord); dC = dec(S_dLdC, ctx, coord)
                dp = dec(O_dLdpsi, ctx, coord); do = dec(O_dLdop, ctx, coord)
                for p in range(len(g)):
                    rgb_full[oy + g[p][1], ox + g[p][0]] = [C[p * 32 + ch] for ch in range(3)]
                # scatter-add grads to the real global Gaussians (skip dummies)
                for j in range(KT):
                    gid = real[j]
                    if gid < 0: continue
                    dcol = [sum(wv[p * 32 + j] * dC[p * 32 + ch] for p in range(len(g))) for ch in range(3)]
                    gacc[gid, 0] += dp[0 * 32 + 2 * j]; gacc[gid, 1] += dp[1 * 32 + 2 * j]
                    gacc[gid, 2] += dp[2 * 32 + 2 * j]; gacc[gid, 3] += dp[1 * 32 + 2 * j + 1]
                    gacc[gid, 4] += dp[2 * 32 + 2 * j + 1]; gacc[gid, 5] += do[j]
                    gacc[gid, 6] += dcol[0]; gacc[gid, 7] += dcol[1]; gacc[gid, 8] += dcol[2]

        mse = float(((rgb_full - target) ** 2).mean())
        psnr = 99.0 if mse < 1e-12 else 10 * math.log10(1.0 / mse)
        # x280 Adam over all N (identity order)
        bc1 = 1.0 / (1 - 0.9 ** step); bc2 = 1.0 / (1 - 0.999 ** step)
        LR = [0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05]
        hdr = [N, step, fb(bc1), fb(bc2), fb(0.9), fb(0.999), fb(1e-8)] + [fb(x) for x in LR]
        dev.wr(0, X_ORDER, list(range(N)))
        dev.wr(0, X_GIN, [fb(gacc[i][j]) for i in range(N) for j in range(9)])
        dev.wr(0, X_HDR, hdr)
        xstep += 1; dev.wr(0, X_CMD, [1]); dev.wr(0, X_DB, [xstep])
        for _ in range(200):
            if dev.rd(0, X_DONE) == xstep: break
            time.sleep(0.003)
        touch = sum(min(len(tiles[t]), KT) for t in range(ntx * nty))
        pr(f"  step {step:2d}: PSNR={psnr:5.2f} dB  loss={mse:.5f}  tile-touches={touch}  finite={np.isfinite(rgb_full).all()}  ({time.time()-t0:.1f}s)")

    pr("done (GAP-3 full-image multi-tile 3D: x280 proj+Adam, per-tile Tensix render/backward, grad scatter-add)")


if __name__ == "__main__":
    main()
