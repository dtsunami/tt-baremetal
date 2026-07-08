"""INSTRUMENTED multi-tile fused 3DGS training flow, scalable to 1600x1600, built to START TUNING PERF.
Composes the silicon-proven pieces — x280 real-camera projection + backward + Adam over GDDR-resident params
(Gap 1/5, opt_proj_gddr.c), host tile-binning (Gap 2 rule), per-tile Tensix fused render+backward
(resident_train_perf) across 1..120 workers, per-Gaussian grad scatter-add — and TIMES EVERY STAGE so the
bottleneck is visible: x280 projection/Adam device-cycles, per-ring render device-cycles (kernel T_END),
and host<->device NoC wall-time per phase (the serial exalens relay that the grid does NOT parallelize).

  usage: train_fused_instrumented.py [IMG] [N] [WORKERS] [STEPS]   (default 512 1024 8 2)
Prints a per-stage breakdown each step. Emits a machine-readable JSON telemetry log to
scratchpad/telemetry_<IMG>.json for the dashboard.
"""
import sys, struct, math, time, json
sys.path.insert(0, "/home/starboy/bhtop/src"); sys.path.insert(0, "/home/starboy/bhtop/scratchpad")
import numpy as np
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device as rd, read_words_from_device as rds, write_words_to_device as wr
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
from bhtop.tensix.loader import TensixLauncher, worker_coords
from bhtop.tensix import splat as SP, matmul as MM, llk_run
from bhtop.tensix.resident import boot_resident
import gap1_proj_golden as PG
import gap2_bin_golden as BIN

IMG = int(sys.argv[1]) if len(sys.argv) > 1 else 512
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
NW = int(sys.argv[3]) if len(sys.argv) > 3 else 8
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 2
TILE, KT, P = 16, 12, 32
NTX = IMG // TILE
OPT_SRC = "/home/starboy/bhtop/src/bhtop/kernels/x280/het/opt_proj_gddr.c"
# Tensix render L1 map (resident_train_perf, via train_multitile3d)
H = dict(phi=0x21000, psi=0x22000, Ppair=0x23000, Dop=0x24000, Dnop=0x25000, Stri=0x26000, Iden=0x27000,
         color=0x28000, gt=0x29000, opB=0x2A000, colorT=0x2B000, PpairT=0x2C000, U=0x2D000, phi2T=0x2E000,
         ones=0x2F000, ones1P=0x30000)
S_C, S_w, S_dLdC, O_dLdop, O_dLdpsi = 0x42800, 0x42000, 0x43000, 0x51000, 0x52000
DB, DONE, T_END = 0x16000, 0x16010, 0x16114
# x280 control (opt_proj_gddr small-window control; state in big GDDR 0x30010000+)
X_HDR, X_CAM, X_DB, X_DONE, X_CMD = 0x30005000, 0x30005060, 0x30004000, 0x30004010, 0x30004020
fb = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]
bf = lambda u: struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]
DUMMY = (-1000.0, -1000.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1e9)

Rv = np.eye(3); tv = np.array([0.0, 0.0, 4.0]); fx = fy = IMG * 1.2; cx = cy = IMG / 2.0
CAM = (Rv, tv, fx, fy, cx, cy)


def gbases(n):
    PARAM = 0x30010000; M = PARAM + n * 14 * 4; V = M + n * 14 * 4
    GRADIN = V + n * 14 * 4; PUB = GRADIN + n * 9 * 4; ORDER = PUB + n * 6 * 4
    return PARAM, M, V, GRADIN, PUB, ORDER


def enc(flat): return MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
def pad(rc): return SP._pad32(rc)


class Tele:
    """per-stage accumulator: wall seconds, device cycles, count (rings/tiles/gaussians), host<->device bytes."""
    def __init__(self): self.d = {}
    def add(self, stage, wall=0.0, cyc=0, cnt=0, byt=0):
        s = self.d.setdefault(stage, [0.0, 0, 0, 0]); s[0] += wall; s[1] += cyc; s[2] += cnt; s[3] += byt
    def report(self, label):
        tot = sum(v[0] for v in self.d.values())
        print(f"  --- {label}: total {tot*1e3:.0f} ms ---")
        for st, (w, c, n, b) in sorted(self.d.items(), key=lambda kv: -kv[1][0]):
            cyc = f" | {c:,} dev-cyc" if c else ""
            cnt = f" | {n} x" if n else ""
            bw = f" | {b/max(w,1e-9)/1e6:8.1f} MB/s ({b/1e6:.2f} MB)" if b else ""
            print(f"    {st:22s} {w*1e3:8.1f} ms ({100*w/max(tot,1e-9):4.1f}%){cyc}{cnt}{bw}")
        return tot


def rand_scene3d(seed, n):
    rng = np.random.default_rng(seed)
    mean = rng.normal(0, 1.4, (n, 3)); sl = rng.normal(-2.2, 0.25, (n, 3))
    q = rng.normal(0, 1, (n, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    op = rng.uniform(0.4, 0.9, (n, 1)); col = rng.uniform(0.1, 0.9, (n, 3))
    return np.concatenate([mean, sl, q, op, col], axis=1)


def fast_target(gs, img, sigma=3.0):
    """bbox-local alpha-composite of the projected 2D Gaussians (near->far). O(N * bbox_area), so it
    stays fast at 1600px where the full golden render (O(pixels*N)) does not."""
    C = np.zeros((img, img, 3)); Tr = np.ones((img, img))
    order = sorted(range(len(gs)), key=lambda i: gs[i][9])
    for i in order:
        gx, gy, a, b, c, op = gs[i][:6]; col = np.array(gs[i][6:9])
        det = a * c - b * b
        if det <= 0: continue
        ex = sigma * math.sqrt(max(c / det, 1e-9)); ey = sigma * math.sqrt(max(a / det, 1e-9))
        x0 = max(0, int(gx - ex)); x1 = min(img, int(gx + ex) + 1)
        y0 = max(0, int(gy - ey)); y1 = min(img, int(gy + ey) + 1)
        if x1 <= x0 or y1 <= y0: continue
        dx = np.arange(x0, x1) - gx; dy = np.arange(y0, y1) - gy
        E = -0.5 * (a * dx[None, :] ** 2 + 2 * b * dx[None, :] * dy[:, None] + c * dy[:, None] ** 2)
        al = np.clip(op * np.exp(np.clip(E, -60, 0)), 0, 0.99)
        w = Tr[y0:y1, x0:x1] * al
        C[y0:y1, x0:x1] += w[..., None] * col
        Tr[y0:y1, x0:x1] *= (1 - al)
    return C


def stage_consts(coord, ctx, gs_tile):
    gso = gs_tile
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


def stage_static(coord, ctx):
    U = [[1.0 if j > i else 0.0 for i in range(KT)] for j in range(KT)]
    wr(coord, H["Iden"], enc([1.0 if r == c else 0.0 for r in range(32) for c in range(32)]), context=ctx)
    wr(coord, H["U"], enc(pad(U)), context=ctx)
    wr(coord, H["ones"], enc([1.0] * 1024), context=ctx)
    wr(coord, H["ones1P"], enc(pad([[1.0] * P])), context=ctx)


def main():
    pr = lambda *a: print(*a, flush=True)
    ctx = init_ttexalens()
    # Enable bulk DMA transfers on Blackhole: 4B mode chops every read/write into 4-byte register accesses
    # (each below the DMA threshold), pinning us to the ~2.6 MB/s register path. Disabling it sends each buffer
    # as one transfer, which crosses the DMA threshold -> ~9.5 GB/s copy-path DMA for the bulk param/render moves.
    ctx.use_4B_mode = False
    pr(f"[setup] exalens up | IMG={IMG} ({NTX}x{NTX}={NTX*NTX} tiles) N={N} workers={NW} steps={STEPS} | 4B_mode=off (DMA)")
    all_workers = worker_coords(ctx)
    workers = all_workers[:NW]
    pr(f"[setup] {len(all_workers)} Tensix workers available; using {len(workers)}: {[str(c) for c in workers]}")
    dev = L2cpu(ctx=ctx)
    try: dev.bringup(0); pr("[setup] x280 bringup ok")
    except Exception as e: pr("[setup] x280 bringup:", type(e).__name__, "(already up)")

    PARAM, M, V, GRADIN, PUB, ORDER = gbases(N)
    grp = [[(x, y) for y in range(TILE) for x in range(TILE)][i:i + 32] for i in range(0, TILE * TILE, 32)]
    tgt = rand_scene3d(11, N); init = rand_scene3d(22, N)

    # target image: fast bbox-local numpy alpha-composite (SP._golden_render is a Python triple-loop,
    # impractical at 1600px; this restricts each Gaussian to its 3-sigma bbox so it stays fast at any res).
    Rvm, tvm, fxm, fym, cxm, cym = CAM
    gsT = []
    for o in range(N):
        u, vv, dep, a, b, c, _ = PG.project_forward(tgt[o, :3], tgt[o, 3:6], tgt[o, 6:10], Rvm, tvm, fxm, fym, cxm, cym)
        gsT.append((u, vv, a, b, c, float(tgt[o, 10]), *[float(x) for x in tgt[o, 11:14]], dep))
    target = fast_target(gsT, IMG)

    # boot render kernel on every worker + stage static tiles
    t = time.time()
    llk_run.build("resident_train_perf", run_type="L1_TO_L1", fidelity="HiFi4", fp32_acc=False,
                  formats=None, overrides={"ITERATIONS": "constexpr int ITERATIONS = 32;"})
    for i, c in enumerate(workers):
        boot_resident("resident_train_perf", c, ctx=ctx,
                      runtime_words=[1, 1, 1, 1, 1, 128, 128, 0, 4, 4], clear_words=56)
    time.sleep(0.3)
    for c in workers:
        stage_static(c, ctx)
    pr(f"[setup] booted+staged {len(workers)} render workers in {time.time()-t:.1f}s")

    # boot x280 optimizer (opt_proj_gddr) with params in big GDDR
    dev.wr(0, X_HDR, [N, 0])
    dev.wr(0, X_CAM, [fb(x) for x in (list(Rv.flatten()) + list(tv) + [fx, fy, cx, cy])])
    dev.wr(0, PARAM, [fb(init[o, j]) for o in range(N) for j in range(14)])
    dev.wr(0, M, [0] * (N * 14)); dev.wr(0, V, [0] * (N * 14))
    dev.wr(0, ORDER, list(range(N)))
    dev.wr(0, X_DB, [0]); dev.wr(0, X_DONE, [0]); dev.wr(0, X_CMD, [1])
    dev.load(0, 0, tc.compile_source(OPT_SRC, base=CODE_ADDR, march="rv64gc"))
    for _ in range(80):
        if dev.telemetry(0, slots=1, hart=0)[0] == 0x47444452: break
        time.sleep(0.03)
    else:
        pr("FAIL: opt_proj_gddr not resident"); return
    pr("[setup] x280 opt_proj_gddr resident (params in 4GiB GDDR)")

    logs = []
    for step in range(1, STEPS + 1):
        T = Tele(); t_step = time.time()

        # 1) x280 project all N -> publish
        t0 = time.time(); xr = dev.rd(0, X_DB) + 1; dev.wr(0, X_CMD, [2]); dev.wr(0, X_DB, [xr])
        while dev.rd(0, X_DONE) != xr: time.sleep(0.001)
        xt = dev.telemetry(0, slots=4, hart=0)
        T.add("x280_project", wall=time.time() - t0, cyc=xt[2] | (xt[3] << 32), cnt=N)

        # 2) host readback pub + params (the x280->host relay cost)
        t0 = time.time()
        pub = np.array([bf(u) for u in dev.rdn(0, PUB, N * 6)]).reshape(N, 6)
        param = np.array([bf(u) for u in dev.rdn(0, PARAM, N * 14)]).reshape(N, 14)
        T.add("host_readback_pub", wall=time.time() - t0, byt=(N * 6 + N * 14) * 4)

        # 3) bin (host, Gap-2 rule)
        t0 = time.time()
        tiles, ntx, nty = BIN.bin_tiles(pub[:, 0], pub[:, 1], pub[:, 2:5], pub[:, 5], IMG, IMG, tile=TILE, cap=64)
        occ = [t for t in range(ntx * nty) if tiles[t]]
        T.add("host_bin", wall=time.time() - t0, cnt=len(occ))

        # 4) render+backward occupied tiles, round-robin across workers; scatter-add grads
        gacc = np.zeros((N, 9)); rgb_full = np.zeros((IMG, IMG, 3))
        gs = [(pub[o, 0], pub[o, 1], pub[o, 2], pub[o, 3], pub[o, 4],
               float(param[o, 10]), *[float(x) for x in param[o, 11:14]], pub[o, 5]) for o in range(N)]
        rings = 0; render_cyc = 0
        for ti, tl in enumerate(occ):
            coord = workers[ti % len(workers)]
            ids = list(tiles[tl][:KT]); real = list(ids)
            gs_tile = [gs[i] for i in ids]
            while len(gs_tile) < KT: gs_tile.append(DUMMY); real.append(-1)
            ox, oy = (tl % ntx) * TILE, (tl // ntx) * TILE
            t0 = time.time(); stage_consts(coord, ctx, gs_tile); T.add("render_stage_consts", wall=time.time() - t0, byt=9 * 512 * 4)
            for gi, g in enumerate(grp):
                phi = pad([[float(ox + x), float(oy + y), 1.0] for (x, y) in g])
                phi2T = pad([[[2.0 * (ox + x), 2.0 * (oy + y), 2.0][r] for (x, y) in g] for r in range(3)])
                gt_g = pad([[target[oy + g[p][1], ox + g[p][0]][ch] for ch in range(3)] for p in range(len(g))])
                t0 = time.time()
                wr(coord, H["phi"], enc(phi), context=ctx); wr(coord, H["phi2T"], enc(phi2T), context=ctx)
                wr(coord, H["gt"], enc(gt_g), context=ctx)
                ring = rd(coord, DONE, context=ctx) + 1; wr(coord, DB, [ring], context=ctx)
                T.add("render_stage_pergroup", wall=time.time() - t0, byt=3 * 512 * 4)
                t0 = time.time()
                while time.time() - t0 < 4.0 and rd(coord, DONE, context=ctx) != ring: time.sleep(0.0002)
                T.add("render_ring_wait", wall=time.time() - t0, cyc=rd(coord, T_END, context=ctx), cnt=1); rings += 1
                render_cyc += rd(coord, T_END, context=ctx)
                t0 = time.time()
                nz = lambda a: [0.0 if (x != x or x > 1e30 or x < -1e30) else x for x in a]
                C = MM.untilize32(MM.unpack_bf16_words(rds(coord, S_C, word_count=512, context=ctx)))
                wv = nz(MM.untilize32(MM.unpack_bf16_words(rds(coord, S_w, word_count=512, context=ctx))))
                dC = nz(MM.untilize32(MM.unpack_bf16_words(rds(coord, S_dLdC, word_count=512, context=ctx))))
                dp = nz(MM.untilize32(MM.unpack_bf16_words(rds(coord, O_dLdpsi, word_count=512, context=ctx))))
                do = nz(MM.untilize32(MM.unpack_bf16_words(rds(coord, O_dLdop, word_count=512, context=ctx))))
                T.add("render_readback", wall=time.time() - t0, byt=5 * 512 * 4)
                t0 = time.time()
                for pi in range(len(g)):
                    rgb_full[oy + g[pi][1], ox + g[pi][0]] = [C[pi * 32 + ch] for ch in range(3)]
                for j in range(KT):
                    gid = real[j]
                    if gid < 0: continue
                    dcol = [sum(wv[pi * 32 + j] * dC[pi * 32 + ch] for pi in range(len(g))) for ch in range(3)]
                    gacc[gid, 0] += dp[0 * 32 + 2 * j]; gacc[gid, 1] += dp[1 * 32 + 2 * j]
                    gacc[gid, 2] += dp[2 * 32 + 2 * j]; gacc[gid, 3] += dp[1 * 32 + 2 * j + 1]
                    gacc[gid, 4] += dp[2 * 32 + 2 * j + 1]; gacc[gid, 5] += do[j]
                    gacc[gid, 6] += dcol[0]; gacc[gid, 7] += dcol[1]; gacc[gid, 8] += dcol[2]
                T.add("scatter_add", wall=time.time() - t0)

        mse = float(((rgb_full - target) ** 2).mean()); psnr = 99.0 if mse < 1e-12 else 10 * math.log10(1.0 / mse)

        # 5) x280 Adam over all N
        t0 = time.time()
        dev.wr(0, GRADIN, [fb(gacc[i][j]) for i in range(N) for j in range(9)])
        T.add("host_write_gradin", wall=time.time() - t0, byt=N * 9 * 4)
        bc1 = 1.0 / (1 - 0.9 ** step); bc2 = 1.0 / (1 - 0.999 ** step)
        LR = [0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05]
        dev.wr(0, X_HDR, [N, step, fb(bc1), fb(bc2), fb(0.9), fb(0.999), fb(1e-8)] + [fb(x) for x in LR])
        t0 = time.time(); xr = dev.rd(0, X_DB) + 1; dev.wr(0, X_CMD, [1]); dev.wr(0, X_DB, [xr])
        while dev.rd(0, X_DONE) != xr: time.sleep(0.001)
        xt = dev.telemetry(0, slots=4, hart=0)
        T.add("x280_adam", wall=time.time() - t0, cyc=xt[2] | (xt[3] << 32), cnt=N)

        step_wall = time.time() - t_step
        pr(f"\nstep {step}: PSNR={psnr:5.2f} dB  {len(occ)} occ tiles  {rings} rings  render {render_cyc:,} dev-cyc  ({step_wall:.1f}s)")
        T.report(f"step {step} stage breakdown")
        logs.append({"step": step, "img": IMG, "n": N, "workers": len(workers), "occ_tiles": len(occ),
                     "rings": rings, "psnr": psnr, "step_wall_s": step_wall, "render_dev_cyc": render_cyc,
                     "stages": {k: {"ms": v[0] * 1e3, "cyc": v[1], "cnt": v[2]} for k, v in T.d.items()}})

    out = f"/home/starboy/bhtop/scratchpad/telemetry_{IMG}.json"
    json.dump(logs, open(out, "w"), indent=2)
    pr(f"\n[done] telemetry -> {out}")


if __name__ == "__main__":
    main()
