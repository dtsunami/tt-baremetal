"""GRID-SCALE fully-on-device fused 3DGS trainer. het_x280 is the hub (project+whiten / produce / consume /
Adam); tiles are dispatched across W workers of the RIGHT NUMA domain (x=11..16, local to the x280's x=9
bank). Per batch of W tiles: het produces each tile's operands into its per-worker slot, workers cb_reader
them, RENDER CONCURRENTLY (ring all, then collect), cb_writer grads back, het consumes. Host issues doorbells
+ reads a scalar loss. Emits per-stage / per-worker telemetry (JSON) for the live cockpit — NoC heatmap,
Amdahl-Pareto, DRAM-latency.  Placement (worker set, DRAM base) is FLEXIBLE for NoC0/1 routing experiments.

  usage: train_het_grid.py [STEPS] [IMG] [N] [W]   (default 8 48 24 6)
"""
import sys, struct, math, time, json
sys.path.insert(0, "/home/starboy/bhtop/src"); sys.path.insert(0, "/home/starboy/bhtop/scratchpad")
import numpy as np
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device as rd, read_words_from_device as rds, write_words_to_device as wr
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
from bhtop.tensix.loader import worker_coords, worker_coord
from bhtop.tensix import splat as SP, matmul as MM, llk_run
from bhtop.tensix.resident import boot_resident
from bhtop.tensix.baremetal import BareMetal, bm_coord
import gap1_proj_golden as PG
import gap2_bin_golden as BIN

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 8
IMG = int(sys.argv[2]) if len(sys.argv) > 2 else 48
N = int(sys.argv[3]) if len(sys.argv) > 3 else 24
W = int(sys.argv[4]) if len(sys.argv) > 4 else 6
K, TILE, P = 12, 16, 32
NTX = IMG // TILE
SRC = "/home/starboy/bhtop/src/bhtop/kernels/x280/het/het_x280.c"
H = dict(phi=0x21000, psi=0x22000, Ppair=0x23000, Dop=0x24000, Dnop=0x25000, Stri=0x26000, Iden=0x27000,
         color=0x28000, gt=0x29000, opB=0x2A000, colorT=0x2B000, PpairT=0x2C000, U=0x2D000, phi2T=0x2E000,
         ones=0x2F000, ones1P=0x30000)
T_END, DB, DONE = 0x16114, 0x16000, 0x16010
X_HDR, X_CAM, X_IDL, X_LOSS, X_DB, X_DONE, X_CMD = 0x30005000, 0x30005060, 0x300050A0, 0x30005B00, 0x30004000, 0x30004010, 0x30004020
PARAM, OPBASE, GINBOX = 0x30100000, 0x30080000, 0x300C0000
OPSTRIDE, GISTRIDE = 0x3000, 0x2000
HUB = (8, 3)                                          # x280 hub tile
Rv = np.eye(3); tv = np.array([0.0, 0.0, 4.0]); fx = fy = IMG * 1.1; cx = cy = IMG / 2.0
fb = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]
bff = lambda u: struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]
def enc(flat): return MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
def pad(rc): return SP._pad32(rc)
grp = [[(x, y) for y in range(TILE) for x in range(TILE)][i:i + 32] for i in range(0, TILE * TILE, 32)]
DUMMY_ID = -1


def rand_scene3d(seed):
    rng = np.random.default_rng(seed)
    mean = rng.normal(0, 1.3, (N, 3)); sl = rng.normal(-2.1, 0.2, (N, 3))
    q = rng.normal(0, 1, (N, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    op = rng.uniform(0.4, 0.9, (N, 1)); col = rng.uniform(0.1, 0.9, (N, 3))
    return np.concatenate([mean, sl, q, op, col], axis=1)


def gs_of(param):
    gs = []
    for o in range(N):
        u, v, dep, a, b, c, _ = PG.project_forward(param[o, :3], param[o, 3:6], param[o, 6:10], Rv, tv, fx, fy, cx, cy)
        gs.append((u, v, a, b, c, float(param[o, 10]), *[float(x) for x in param[o, 11:14]], dep))
    return gs


def fast_target(gs, img, sigma=3.0):
    C = np.zeros((img, img, 3)); Tr = np.ones((img, img))
    for i in sorted(range(len(gs)), key=lambda i: gs[i][9]):
        gx, gy, a, b, c, op = gs[i][:6]; col = np.array(gs[i][6:9]); det = a * c - b * b
        if det <= 0: continue
        ex = sigma * math.sqrt(max(c / det, 1e-9)); ey = sigma * math.sqrt(max(a / det, 1e-9))
        x0, x1 = max(0, int(gx - ex)), min(img, int(gx + ex) + 1); y0, y1 = max(0, int(gy - ey)), min(img, int(gy + ey) + 1)
        if x1 <= x0 or y1 <= y0: continue
        dx = np.arange(x0, x1) - gx; dy = np.arange(y0, y1) - gy
        E = -0.5 * (a * dx[None, :] ** 2 + 2 * b * dx[None, :] * dy[:, None] + c * dy[:, None] ** 2)
        al = np.clip(op * np.exp(np.clip(E, -60, 0)), 0, 0.99); wgt = Tr[y0:y1, x0:x1] * al
        C[y0:y1, x0:x1] += wgt[..., None] * col; Tr[y0:y1, x0:x1] *= (1 - al)
    return C


def stage_static(coord, ctx):
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


def main():
    pr = lambda *a: print(*a, flush=True)
    ctx = init_ttexalens()
    allw = [c for c in worker_coords(ctx) if tuple(c.to("noc0"))[0] > 8]   # RIGHT NUMA block (x>8)
    workers = allw[:W]; wxy = [tuple(c.to("noc0")) for c in workers]
    pr(f"[setup] IMG={IMG} ({NTX}x{NTX}={NTX*NTX} tiles) N={N} W={len(workers)} right-block workers {wxy}")
    dev = L2cpu(ctx=ctx)
    try: dev.bringup(0)
    except Exception: pass

    tgt = rand_scene3d(11); init = rand_scene3d(22)
    target = fast_target(gs_of(tgt), IMG)

    t = time.time()
    llk_run.build("resident_train_perf", run_type="L1_TO_L1", fidelity="HiFi4", fp32_acc=False,
                  formats=None, overrides={"ITERATIONS": "constexpr int ITERATIONS = 32;"})
    for c in workers:
        boot_resident("resident_train_perf", c, ctx=ctx, runtime_words=[1, 1, 1, 1, 1, 128, 128, 0, 4, 4], clear_words=56)
    time.sleep(0.3)
    for c in workers: stage_static(c, ctx)
    pr(f"[setup] booted+staged {len(workers)} render workers in {time.time()-t:.1f}s")

    dev.wr(0, X_HDR, [N, 0]); dev.wr(0, X_HDR + 21 * 4, [0])
    dev.wr(0, X_CAM, [fb(x) for x in (list(Rv.flatten()) + list(tv) + [fx, fy, cx, cy])])
    dev.wr(0, PARAM, [fb(init[o, j]) for o in range(N) for j in range(14)])
    M = PARAM + N * 14 * 4; V = M + N * 14 * 4; GACC = V + N * 14 * 4
    dev.wr(0, M, [0] * (N * 14)); dev.wr(0, V, [0] * (N * 14)); dev.wr(0, GACC, [0] * (N * 9))
    dev.wr(0, X_DB, [0]); dev.wr(0, X_DONE, [0])
    dev.load(0, 0, tc.compile_source(SRC, base=CODE_ADDR, march="rv64gc"))
    for _ in range(80):
        if dev.telemetry(0, slots=1, hart=0)[0] == 0x48455421: break
        time.sleep(0.03)
    else: pr("FAIL: het not resident"); return
    pr("[setup] het_x280 hub resident (right NUMA domain)")

    # RESIDENT cb_io on each worker's BRISC (boot once, then doorbell — no per-call reload)
    IO_DB, IO_DONE, IO_CFG = 0x3000, 0x3010, 0x3020
    io_bin = BareMetal.build("cb_io")
    for s, c in enumerate(workers):
        wr(c, IO_DB, [0] * 8, context=ctx); BareMetal(*wxy[s], ctx=ctx, risc="brisc").run(io_bin)
    time.sleep(0.2)
    pr(f"[setup] cb_io resident on {len(workers)} BRISCs")

    def cbio(s, base, mode):                              # ring the resident I/O engine (read=0 / write=1)
        wr(workers[s], IO_CFG, [bm_coord(*HUB), base, mode], context=ctx)
        r = rd(workers[s], IO_DONE, context=ctx) + 1; wr(workers[s], IO_DB, [r], context=ctx)
        t = time.time()
        while time.time() - t < 2.0 and rd(workers[s], IO_DONE, context=ctx) != r: time.sleep(0.0002)

    def het(cmd, slot=0, extra=None):
        if extra: dev.wr(0, X_HDR, extra)
        dev.wr(0, X_HDR + 21 * 4, [slot]); dev.wr(0, X_CMD, [cmd])
        r = dev.rd(0, X_DB) + 1; dev.wr(0, X_DB, [r])
        while dev.rd(0, X_DONE) != r: time.sleep(0.0005)
        return dev.telemetry(0, slots=4, hart=0)

    logs = []
    for step in range(1, STEPS + 1):
        T = {"produce": 0.0, "reader": 0.0, "render": 0.0, "writer": 0.0, "consume": 0.0, "adam": 0.0, "host_pxl": 0.0}
        cyc = {"produce": 0, "consume": 0, "adam": 0, "render": 0}
        t0 = time.time()
        het(2, extra=[N, step])                                    # project+whiten all params
        pub = np.array([[bff(u) for u in dev.rdn(0, (PARAM + (N * 14 * 3 + N * 9) * 4) + o * 6 * 4, 6)] for o in range(N)])  # PUB after gacc
        tiles, ntx, nty = BIN.bin_tiles(pub[:, 0], pub[:, 1], pub[:, 2:5], pub[:, 5], IMG, IMG, tile=TILE, cap=64)
        occ = [t for t in range(ntx * nty) if tiles[t]]
        worker_active = [0] * len(workers)
        for b0 in range(0, len(occ), len(workers)):                # batch of W tiles
            batch = occ[b0:b0 + len(workers)]
            slots = list(range(len(batch)))
            # produce operands (serial het) + cb_reader (per worker) for the batch
            for s, tl in zip(slots, batch):
                ids = list(tiles[tl][:K]); ids += [ids[-1]] * (K - len(ids)) if ids else [0] * K
                t = time.time(); dev.wr(0, X_IDL, [K] + ids); r = het(5, slot=s); T["produce"] += time.time() - t
                cyc["produce"] += r[2]; worker_active[s] += 1
                t = time.time(); cbio(s, OPBASE + s * OPSTRIDE, 0); T["reader"] += time.time() - t
            # render 8 groups, CONCURRENTLY across the batch (ring all, then collect all)
            for gi, g in enumerate(grp):
                th = time.time()
                for s, tl in zip(slots, batch):
                    ox, oy = (tl % ntx) * TILE, (tl // ntx) * TILE
                    phi = pad([[float(ox + x), float(oy + y), 1.0] for (x, y) in g])
                    phi2T = pad([[[2.0 * (ox + x), 2.0 * (oy + y), 2.0][r] for (x, y) in g] for r in range(3)])
                    gt = pad([[target[oy + g[p][1], ox + g[p][0]][ch] for ch in range(3)] for p in range(len(g))])
                    wr(workers[s], H["phi"], enc(phi), context=ctx); wr(workers[s], H["phi2T"], enc(phi2T), context=ctx)
                    wr(workers[s], H["gt"], enc(gt), context=ctx)
                T["host_pxl"] += time.time() - th
                t = time.time()
                rings = [rd(workers[s], DONE, context=ctx) + 1 for s in slots]
                for s in slots: wr(workers[s], DB, [rings[s]], context=ctx)     # ring ALL (concurrent)
                for s in slots:                                                # collect ALL
                    tt = time.time()
                    while time.time() - tt < 4.0 and rd(workers[s], DONE, context=ctx) != rings[s]: time.sleep(0.0005)
                cyc["render"] += sum(rd(workers[s], T_END, context=ctx) for s in slots)
                T["render"] += time.time() - t
            # grads back (cb_writer) + het consume, per tile
            for s, tl in zip(slots, batch):
                t = time.time(); cbio(s, GINBOX + s * GISTRIDE, 1); T["writer"] += time.time() - t
                ids = list(tiles[tl][:K]); ids += [ids[-1]] * (K - len(ids)) if ids else [0] * K
                t = time.time(); dev.wr(0, X_IDL, [K] + ids); r = het(6, slot=s); T["consume"] += time.time() - t; cyc["consume"] += r[2]
        bc1 = 1.0 / (1 - 0.9 ** step); bc2 = 1.0 / (1 - 0.999 ** step)
        LR = [0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05]
        t = time.time(); r = het(1, extra=[N, step, fb(bc1), fb(bc2), fb(0.9), fb(0.999), fb(1e-8)] + [fb(x) for x in LR])
        T["adam"] += time.time() - t; cyc["adam"] = r[2]
        loss = bff(dev.rdn(0, X_LOSS, 1)[0]); psnr = 99.0 if loss < 1e-9 else 10 * math.log10((IMG * IMG * 3) / loss)
        wall = time.time() - t0
        top = sorted(T.items(), key=lambda kv: -kv[1])[:3]
        pr(f"  step {step:2d}: PSNR~{psnr:5.2f}  loss={loss:.4f}  occ={len(occ)}tiles  render_dev={cyc['render']:,}cyc  "
           f"({wall:.1f}s)  top: {', '.join(f'{k}={v*1e3:.0f}ms' for k,v in top)}")
        logs.append({"step": step, "img": IMG, "n": N, "workers": len(workers), "wxy": wxy, "hub": HUB,
                     "occ": len(occ), "psnr": psnr, "loss": loss, "wall_s": wall,
                     "stage_ms": {k: v * 1e3 for k, v in T.items()}, "dev_cyc": cyc, "worker_active": worker_active})

    out = f"/home/starboy/bhtop/scratchpad/grid_telemetry_{IMG}.json"
    json.dump(logs, open(out, "w"), indent=2); pr(f"[done] fully-on-device grid; telemetry -> {out}")


if __name__ == "__main__":
    main()
