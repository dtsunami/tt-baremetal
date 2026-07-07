"""GAP-1 fully-on-device 3D Gaussian-splat trainer. REAL 3D params [mean3, scale_log3, quat4, op, col3]
live resident on the x280; the x280 runs the full camera PROJECTION (proj.h proj_fwd), the projection
BACKWARD (proj_bwd), and Adam — all on-chip (opt_proj_step.c). Each step the x280 publishes the projected
(gx,gy,a,b,c,depth); the host reads it to stage the resident Tensix render+backward (resident_train_perf),
gathers dLdpsi/dLdop, computes dLdcolor, and rings the x280 optimizer. Projection is NO LONGER host-side.

This is train_resident.py (2D) lifted to real 3D. Single view by default; pass a views count>1 to
accumulate grads across cameras before each Adam step (the 3D-structure test).

  usage: train_resident3d.py [STEPS] [VIEWS]     (default 30 1)
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

K, SIZE, P = 12, 16, 32
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
VIEWS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
OPT_SRC = "/home/starboy/bhtop/src/bhtop/kernels/x280/het/opt_proj_step.c"
H = dict(phi=0x21000, psi=0x22000, Ppair=0x23000, Dop=0x24000, Dnop=0x25000, Stri=0x26000, Iden=0x27000,
         color=0x28000, gt=0x29000, opB=0x2A000, colorT=0x2B000, PpairT=0x2C000, U=0x2D000, phi2T=0x2E000,
         ones=0x2F000, ones1P=0x30000)
S_C, S_w, S_dLdC, O_dLdop, O_dLdpsi = 0x42800, 0x42000, 0x43000, 0x51000, 0x52000
DB, DONE = 0x16000, 0x16010
# x280 GDDR (opt_proj_step.c)
X_HDR, X_CAM, X_ORDER, X_GIN, X_PARAM, X_M, X_V, X_PUB = 0x30005000, 0x30005060, 0x300050A0, 0x30005100, 0x30005800, 0x30006000, 0x30006800, 0x30007000
X_DB, X_DONE, X_CMD = 0x30004000, 0x30004010, 0x30004020
fb = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]
bf = lambda u: struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]

# camera: a pinhole looking down +z; intrinsics land an N(0,0.7) scene inside the 16x16 tile
CAMS = None
def make_cams(nv):
    cams = []
    for i in range(max(nv, 1)):
        ang = 0.5 * (i - (nv - 1) / 2.0) if nv > 1 else 0.0    # small orbit for multi-view
        c, s = math.cos(ang), math.sin(ang)
        Rv = np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])    # yaw about y
        tv = Rv @ np.array([0.0, 0.0, 4.0])                    # keep origin ~4 in front
        cams.append((Rv, tv, 26.0, 26.0, 8.0, 8.0))
    return cams


def enc(flat): return MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
def pad(rc): return SP._pad32(rc)
def dec(a, ctx, c): return MM.untilize32(MM.unpack_bf16_words(rds(c, a, word_count=512, context=ctx)))
def groups_pixels():
    pix = [(x, y) for y in range(SIZE) for x in range(SIZE)]
    return [pix[i:i + 32] for i in range(0, len(pix), 32)]


def rand_scene3d(seed):
    rng = np.random.default_rng(seed)
    mean = rng.normal(0, 0.7, (K, 3))
    sl = rng.normal(-1.7, 0.25, (K, 3))
    q = rng.normal(0, 1, (K, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    op = rng.uniform(0.4, 0.9, (K, 1)); col = rng.uniform(0.1, 0.9, (K, 3))
    return np.concatenate([mean, sl, q, op, col], axis=1)      # [K,14]


def project_host(param, cam):
    """3D params -> 2D gs tuples (gx,gy,a,b,c,op,c0,c1,c2,depth) via golden proj_fwd (for targets)."""
    Rv, tv, fx, fy, cx, cy = cam
    gs = []
    for o in range(K):
        gx, gy, dep, a, b, c, _ = PG.project_forward(param[o, :3], param[o, 3:6], param[o, 6:10], Rv, tv, fx, fy, cx, cy)
        gs.append((gx, gy, a, b, c, float(param[o, 10]), *[float(x) for x in param[o, 11:14]], dep))
    return gs


def gs_from_pub(pub, param):
    """build 2D gs tuples from the x280-published projection + resident op/color."""
    gs = []
    for o in range(K):
        gx, gy, a, b, c, dep = pub[o]
        gs.append((gx, gy, a, b, c, float(param[o, 10]), *[float(x) for x in param[o, 11:14]], dep))
    return gs


def stage_consts(coord, ctx, gs, order):
    gso = [gs[i] for i in order]
    psi, Ppair, Dop, Dnop, Mcomb, color = SP._consts(gso, K)
    Stri = [Mcomb[r] for r in range(K)]
    op = [Dop[k][k] for k in range(K)]
    colorT = [[color[k][r] for k in range(K)] for r in range(3)]
    PpairT = [[Ppair[r][c] for r in range(2 * K)] for c in range(K)]
    opB = [[(op[k] if k < K else 0.5) for k in range(32)] for _ in range(P)]
    psi_rows = [[psi[r][c] for c in range(2 * K)] for r in range(3)]
    for name, m in [("psi", psi_rows), ("Ppair", Ppair), ("Dop", Dop), ("Dnop", Dnop), ("Stri", Stri),
                    ("color", color), ("colorT", colorT), ("PpairT", PpairT), ("opB", opB)]:
        wr(coord, H[name], enc(pad(m)), context=ctx)


def main():
    pr = lambda *a: print(*a, flush=True)
    cams = make_cams(VIEWS)
    ctx = init_ttexalens(); pr("[setup] exalens up")
    coord = TensixLauncher.at(1, 2, ctx=ctx).coord
    dev = L2cpu(ctx=ctx)
    try: dev.bringup(0); pr("[setup] x280 bringup ok")
    except Exception as e: pr("[setup] x280 bringup:", type(e).__name__, "(already up)")

    grp = groups_pixels()
    U = [[1.0 if j > i else 0.0 for i in range(K)] for j in range(K)]
    phi2T = [pad([[2.0 * x, 2.0 * y, 2.0][r] for (x, y) in g] for r in range(3)) for g in grp]
    phi_g = [pad([[float(x), float(y), 1.0] for (x, y) in g]) for g in grp]

    # targets: fixed 3D scene projected+rendered per view (host golden)
    tgt = rand_scene3d(11)
    init = rand_scene3d(22)
    targets = []
    for cam in cams:
        gsT = project_host(tgt, cam)
        targets.append(np.array(SP._golden_render(gsT, SIZE)))     # [256][3]

    # boot resident render kernel + static tiles
    llk_run.build("resident_train_perf", run_type="L1_TO_L1", fidelity="HiFi4", fp32_acc=False,
                  formats=None, overrides={"ITERATIONS": "constexpr int ITERATIONS = 32;"}); pr("[setup] render kernel built")
    boot_resident("resident_train_perf", coord, ctx=ctx,
                  runtime_words=[1, 1, 1, 1, 1, 128, 128, 0, 4, 4], clear_words=56)
    time.sleep(0.3); pr("[setup] render kernel booted")
    wr(coord, H["Iden"], enc([1.0 if r == c else 0.0 for r in range(32) for c in range(32)]), context=ctx)
    wr(coord, H["U"], enc(pad(U)), context=ctx)
    wr(coord, H["ones"], enc([1.0] * 1024), context=ctx)
    wr(coord, H["ones1P"], enc(pad([[1.0] * P])), context=ctx)

    # x280 resident optimizer: 14 params (orig order), zero Adam, camera[0], load opt_proj_step ONCE
    dev.wr(0, X_HDR, [K, 0])
    dev.wr(0, X_CAM, [fb(x) for x in (list(cams[0][0].flatten()) + list(cams[0][1]) + list(cams[0][2:]))])
    dev.wr(0, X_ORDER, list(range(K)))
    dev.wr(0, X_PARAM, [fb(init[o, j]) for o in range(K) for j in range(14)])
    dev.wr(0, X_M, [0] * (K * 14)); dev.wr(0, X_V, [0] * (K * 14)); dev.wr(0, X_PUB, [0] * (K * 6))
    dev.wr(0, X_DB, [0]); dev.wr(0, X_DONE, [0]); dev.wr(0, X_CMD, [1])
    pr("[setup] loading x280 opt_proj_step")
    dev.load(0, 0, tc.compile_source(OPT_SRC, base=CODE_ADDR, march="rv64gc"))
    for _ in range(80):
        if dev.telemetry(0, slots=1, hart=0)[0] == 0x4F505421: break
        time.sleep(0.03)
    else:
        pr("FAIL: opt_proj_step not resident"); return
    pr("[setup] x280 opt_proj_step resident")

    def psnr_of(rgb_tile, target):
        mse = sum((rgb_tile[p][ch] - target[p][ch]) ** 2 for p in range(SIZE * SIZE) for ch in range(3)) / (SIZE * SIZE * 3)
        if not math.isfinite(mse):
            return float("nan"), mse
        return (99.0 if mse < 1e-12 else 10 * math.log10(1.0 / mse)), mse

    ring = 0
    xstep = 0
    pr(f"GAP-1 3D resident trainer: K={K} {SIZE}x{SIZE}, {STEPS} steps, {VIEWS} view(s) "
       f"(x280 projection+backward+Adam; Tensix render/backward)")
    for step in range(1, STEPS + 1):
        t0 = time.time()
        param = np.array([[bf(u) for u in dev.rdn(0, X_PARAM + o * 14 * 4, 14)] for o in range(K)])
        # accumulate the 9-vec SORTED gradin across views; do 1 Adam step / iteration
        acc_gin = None; last_psnr = None; last_mse = None
        for vi, cam in enumerate(cams):
            # tell x280 this view's camera, ring a projection-only publish (cmd=2, no Adam)
            dev.wr(0, X_CAM, [fb(x) for x in (list(cam[0].flatten()) + list(cam[1]) + list(cam[2:]))])
            xstep += 1
            dev.wr(0, X_CMD, [2]); dev.wr(0, X_DB, [xstep])
            for _ in range(200):
                if dev.rd(0, X_DONE) == xstep: break
                time.sleep(0.002)
            pub = np.array([[bf(u) for u in dev.rdn(0, X_PUB + o * 6 * 4, 6)] for o in range(K)])
            if not np.isfinite(pub).all():
                bad = np.where(~np.isfinite(pub).all(axis=1))[0]
                pr(f"  [diag step {step} view {vi}] non-finite pub rows {bad.tolist()}")
                for o in bad[:3]:
                    pr(f"    G{o} param={np.round(param[o],3).tolist()} pub={pub[o].tolist()}")
            gs = gs_from_pub(pub, param)
            order = sorted(range(K), key=lambda i: gs[i][9])
            dev.wr(0, X_ORDER, [order[i] & 0xFFFFFFFF for i in range(K)])
            target = targets[vi]
            gt_g = [[[target[g[p][1] * SIZE + g[p][0]][ch] for ch in range(3)] for p in range(len(g))] for g in grp]
            stage_consts(coord, ctx, gs, order)

            dLdpsi = [[0.0] * (2 * K) for _ in range(3)]; dLdop = [0.0] * K
            dLdcol = [[0.0] * 3 for _ in range(K)]; rgb_tile = [[0.0, 0.0, 0.0] for _ in range(SIZE * SIZE)]
            for gi, g in enumerate(grp):
                wr(coord, H["phi"], enc(phi_g[gi]), context=ctx)
                wr(coord, H["phi2T"], enc(phi2T[gi]), context=ctx)
                wr(coord, H["gt"], enc(pad(gt_g[gi])), context=ctx)
                ring += 1
                wr(coord, DB, [ring], context=ctx)
                tt = time.time()
                while time.time() - tt < 4.0 and rd(coord, DONE, context=ctx) != ring:
                    time.sleep(0.003)
                C = dec(S_C, ctx, coord); wv = dec(S_w, ctx, coord); dC = dec(S_dLdC, ctx, coord)
                dp = dec(O_dLdpsi, ctx, coord); do = dec(O_dLdop, ctx, coord)
                for p in range(len(g)):
                    rgb_tile[g[p][1] * SIZE + g[p][0]] = [C[p * 32 + ch] for ch in range(3)]
                for r in range(3):
                    for mm in range(2 * K): dLdpsi[r][mm] += dp[r * 32 + mm]
                for k in range(K): dLdop[k] += do[k]
                for k in range(K):
                    for ch in range(3):
                        dLdcol[k][ch] += sum(wv[p * 32 + k] * dC[p * 32 + ch] for p in range(len(g)))
            last_psnr, last_mse = psnr_of(rgb_tile, target)
            # build the SORTED 9-vec gradin for this view
            gin = []
            for i in range(K):
                o = order[i]
                gin.append([dLdpsi[0][2 * i], dLdpsi[1][2 * i], dLdpsi[2][2 * i], dLdpsi[1][2 * i + 1],
                            dLdpsi[2][2 * i + 1], dLdop[i], dLdcol[i][0], dLdcol[i][1], dLdcol[i][2]])
            gin = np.array(gin)
            acc_gin = gin if acc_gin is None else acc_gin + gin
            acc_order = order

        # one Adam step on the accumulated grad (x280): write gradin+order+hdr, ring, poll done
        bc1 = 1.0 / (1 - 0.9 ** step); bc2 = 1.0 / (1 - 0.999 ** step)
        LR = [0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05]
        hdr = [K, step, fb(bc1), fb(bc2), fb(0.9), fb(0.999), fb(1e-8)] + [fb(x) for x in LR]
        dev.wr(0, X_ORDER, [acc_order[i] & 0xFFFFFFFF for i in range(K)])
        dev.wr(0, X_GIN, [fb(acc_gin[i][j]) for i in range(K) for j in range(9)])
        dev.wr(0, X_HDR, hdr)
        xstep += 1; dev.wr(0, X_CMD, [1]); dev.wr(0, X_DB, [xstep])
        for _ in range(200):
            if dev.rd(0, X_DONE) == xstep: break
            time.sleep(0.003)
        rgbarr = np.array(rgb_tile)
        pr(f"  step {step:2d}: PSNR={last_psnr:5.2f} dB  loss={last_mse:.5f}  rgb[min={rgbarr.min():.2f},max={rgbarr.max():.2f} finite={np.isfinite(rgbarr).all()}]  ({time.time()-t0:.1f}s)")

    pr("done (GAP-1 3D resident: x280 projection+backward+Adam, Tensix render/backward)")


if __name__ == "__main__":
    main()
