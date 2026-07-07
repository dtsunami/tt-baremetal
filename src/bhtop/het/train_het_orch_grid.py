"""GRID-SCALE x280-ORCHESTRATED fully-on-device 3DGS trainer. The host is OUT of the per-tile control AND data
path: per BATCH of W tiles it writes W id-lists + W tile-origins and rings ONE het doorbell (cmd9). het
produces every slot's operands (from resident params) + phi/phi2T/gt (from a resident target image), signals
ALL W conductors at once (workers render CONCURRENTLY), waits all acks, and consumes all grads. Each worker
runs a resident render (TRISC) + a resident conductor (BRISC) that NoC-reads its slot's operands/pixels,
drives the render 8 groups, and NoC-writes grads back. Host per step = cmd2 + ceil(occ/W) x cmd9 + cmd1
doorbells + 1 scalar loss. NO per-group host staging, NO per-tile cb_reader/writer reload.

  usage: train_het_orch_grid.py [STEPS] [IMG] [N] [W]   (default 8 48 24 6)
"""
import sys, struct, math, time, json
sys.path.insert(0, "/home/starboy/bhtop/src"); sys.path.insert(0, "/home/starboy/bhtop/scratchpad")
import numpy as np
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device as rd, read_words_from_device as rds, write_words_to_device as wr
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
from bhtop.tensix.loader import worker_coords
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
X_HDR, X_CAM, X_IDL, X_LOSS, X_DB, X_DONE, X_CMD = 0x30005000, 0x30005060, 0x300050A0, 0x30005B00, 0x30004000, 0x30004010, 0x30004020
FLAG, ACK, ASTRIDE = 0x30006400, 0x30006800, 0x40   # per-slot own 64B line (avoid concurrent-write granule race)
NSLOT, IMGW_A, IDLG, ORIG = 0x30005DF0, 0x30005DF4, 0x30005E00, 0x30006200   # cmd9 batch control
PARAM = 0x30100000
TGT_IMG, OPB_O, PXBASE, GINO = 0x30200000, 0x31000000, 0x32000000, 0x33000000
OPS_O = PXS_O = GIS_O = 0x10000
NHARTS_A, HGO, HDONE, LOSS_H, GACC_X = 0x300027F0, 0x30002800, 0x30002A00, 0x30002C00, 0x30280000
NH = int(sys.argv[5]) if len(sys.argv) > 5 else 4   # x280 harts (leader hart0 + workers) parallelizing produce/consume
CFG = 0x3200                                          # conductor cfg block (worker L1)
HUB = (8, 3)
Rv = np.eye(3); tv = np.array([0.0, 0.0, 4.0]); fx = fy = IMG * 1.1; cx = cy = IMG / 2.0
fb = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]
bff = lambda u: struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]
def enc(flat): return MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
def pad(rc): return SP._pad32(rc)


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
    pr(f"[setup] IMG={IMG} ({NTX}x{NTX}={NTX*NTX} tiles) N={N} W={len(workers)} right workers {wxy}")
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

    dev.wr(0, X_HDR, [N, 0]); dev.wr(0, X_CAM, [fb(x) for x in (list(Rv.flatten()) + list(tv) + [fx, fy, cx, cy])])
    dev.wr(0, PARAM, [fb(init[o, j]) for o in range(N) for j in range(14)])
    M = PARAM + N * 14 * 4; V = M + N * 14 * 4; GACC = V + N * 14 * 4
    dev.wr(0, M, [0] * (N * 14)); dev.wr(0, V, [0] * (N * 14)); dev.wr(0, GACC, [0] * (N * 9))
    dev.wr(0, TGT_IMG, [fb(v) for v in target.reshape(-1)])         # resident FULL target image (IMG*IMG*3 f32)
    dev.wr(0, IMGW_A, [IMG])
    for s in range(W): dev.wr(0, FLAG + s * ASTRIDE, [0]); dev.wr(0, ACK + s * ASTRIDE, [0])
    dev.wr(0, X_DB, [0]); dev.wr(0, X_DONE, [0])
    dev.wr(0, NHARTS_A, [NH])                                            # parallelize produce/consume across NH harts
    for h in range(1, 4): dev.wr(0, HGO + h * 0x40, [0]); dev.wr(0, HDONE + h * 0x40, [0])  # workers idle at boot
    dev.wr(0, GACC_X, [0] * (N * 9 * 3)); dev.wr(0, LOSS_H, [0] * 64)    # per-hart gacc/loss (cmd2 re-zeros each step)
    words = tc.compile_source(SRC, base=CODE_ADDR, march="rv64gc")
    dev.load(0, 0, words)                                                # hart 0 = LEADER (command loop)
    for _ in range(80):
        if dev.telemetry(0, slots=1, hart=0)[0] == 0x48455421: break
        time.sleep(0.03)
    else: pr("FAIL: het leader not resident"); return
    for h in range(1, NH): dev.redirect(0, h, CODE_ADDR)                 # boot WORKER harts on the same image
    time.sleep(0.2)
    hw = [dev.telemetry(0, slots=1, hart=h)[0] for h in range(NH)]
    pr(f"[setup] het_x280 resident on {NH} harts; tele[0]/hart={[hex(x) for x in hw]} (0x48455421=leader, 0x485700xx=worker)")

    # RESIDENT conductor on each worker's BRISC — worker s owns slot s (its own op/px/grad region + flag/ack)
    cbin = BareMetal.build("conductor")
    for s, c in enumerate(workers):
        wr(c, CFG, [bm_coord(*HUB), s, OPB_O + s * OPS_O, GINO + s * GIS_O, FLAG + s * ASTRIDE, ACK + s * ASTRIDE,
                    PXBASE + s * PXS_O, 0, 0, 8], context=ctx)
        BareMetal(*wxy[s], ctx=ctx, risc="brisc").run(cbin)
    time.sleep(0.2)
    pr(f"[setup] resident conductor on {len(workers)} BRISCs — x280 orchestrates the grid, host rings per batch")

    def het(cmd, extra=None):
        if extra: dev.wr(0, X_HDR, extra)
        dev.wr(0, X_CMD, [cmd]); r = dev.rd(0, X_DB) + 1; dev.wr(0, X_DB, [r])
        t = time.time()
        while dev.rd(0, X_DONE) != r and time.time() - t < 12.0: time.sleep(0.0005)
        return dev.telemetry(0, slots=8, hart=0)

    logs = []
    for step in range(1, STEPS + 1):
        t0 = time.time(); T = {"project": 0.0, "orch": 0.0, "adam": 0.0}
        t = time.time(); het(2, extra=[N, step]); T["project"] = time.time() - t
        # het GDDR layout: param, m(N*14), v(N*14), gacc(N*14), coeff(N*9), depth(N), pub(N*6). PUB at N*61.
        pub = np.array([[bff(u) for u in dev.rdn(0, PARAM + (N * 61 + o * 6) * 4, 6)] for o in range(N)])
        if step == 1: pr(f"  [diag] pub gx range=[{pub[:,0].min():.1f},{pub[:,0].max():.1f}] gy=[{pub[:,1].min():.1f},{pub[:,1].max():.1f}] (img={IMG})")
        tiles, ntx, nty = BIN.bin_tiles(pub[:, 0], pub[:, 1], pub[:, 2:5], pub[:, 5], IMG, IMG, tile=TILE, cap=64)
        occ = [tl for tl in range(ntx * nty) if tiles[tl]]
        nbatch = 0
        for b0 in range(0, len(occ), W):                          # batch of up to W tiles
            batch = occ[b0:b0 + W]; ns = len(batch)
            for s, tl in enumerate(batch):
                ids = list(tiles[tl][:K]); ids += [ids[-1]] * (K - len(ids)) if ids else [0] * K
                ox, oy = (tl % ntx) * TILE, (tl // ntx) * TILE
                dev.wr(0, IDLG + s * 0x40, [K] + ids)             # per-slot id list
                dev.wr(0, ORIG + s * 8, [ox, oy])                # per-slot tile origin
            dev.wr(0, NSLOT, [ns])
            t = time.time(); tel = het(9); T["orch"] += time.time() - t; nbatch += 1
            if step == 1 and nbatch == 1:
                acks = [dev.rd(0, ACK + s * ASTRIDE) for s in range(ns)]
                hbs = [rd(workers[s], 0x16020, context=ctx) for s in range(ns)]
                pr(f"  [diag] cmd9 ns={ns} cyc={tel[2]:,} TELE[ring,ndone,NH]={list(tel[5:8])} per-worker ACK={acks} render_HB={hbs}")
                for s in range(ns):
                    cd = list(rds(workers[s], 0x2100, word_count=8, context=ctx))
                    pr(f"    conductor[{s}] {wxy[s]} dbg(flag,last,nflag,g,rdone,ring,ack,DBwr)={cd}")
        bc1 = 1.0 / (1 - 0.9 ** step); bc2 = 1.0 / (1 - 0.999 ** step)
        LR = [0.03, 0.03, 0.03, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05]
        t = time.time(); het(1, extra=[N, step, fb(bc1), fb(bc2), fb(0.9), fb(0.999), fb(1e-8)] + [fb(x) for x in LR])
        T["adam"] = time.time() - t
        loss = bff(dev.rdn(0, X_LOSS, 1)[0]); psnr = 99.0 if loss < 1e-9 else 10 * math.log10((IMG * IMG * 3) / loss)
        wall = time.time() - t0
        pr(f"  step {step:2d}: PSNR~{psnr:5.2f}  loss={loss:.4f}  occ={len(occ)}t/{nbatch}batch  "
           f"({wall*1e3:.0f}ms host: {2+nbatch}+1 doorbells)  proj={T['project']*1e3:.0f} orch={T['orch']*1e3:.0f} adam={T['adam']*1e3:.0f}ms")
        logs.append({"step": step, "img": IMG, "n": N, "workers": len(workers), "wxy": wxy, "hub": HUB,
                     "occ": len(occ), "nbatch": nbatch, "psnr": psnr, "loss": loss, "wall_ms": wall * 1e3,
                     "stage_ms": {k: v * 1e3 for k, v in T.items()}})

    out = f"/home/starboy/bhtop/scratchpad/orch_grid_telemetry_{IMG}.json"
    json.dump(logs, open(out, "w"), indent=2); pr(f"[done] x280-orchestrated grid; telemetry -> {out}")


if __name__ == "__main__":
    main()
