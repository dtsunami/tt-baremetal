"""HetGridEngine — the x280-orchestrated, multi-hart, fully-on-device 3DGS fused-training engine, packaged as a
reusable class (the standalone train_het_orch_grid.py is the reference script it was lifted from). Boots once
(render workers + resident conductors + het multi-hart hub), then per step: upload camera + target image ->
project(cmd2) -> host bins tiles -> orchestrate batches(cmd9, all workers concurrent, NH harts) -> Adam(cmd1)
-> read scalar loss. Params live resident on the x280; host issues doorbells + reads a loss. Supports N>16,
rectangular IMGW x IMGH, and a real pinhole camera per step.

Contract: boot in __init__, set_params([N,14]) / read_params()->[N,14], step(cam16, tgt_flat)->loss."""
import struct, time, math, os
import numpy as np
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device as rd, read_words_from_device as rds, write_words_to_device as wr
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
from bhtop.tensix.loader import worker_coords
from bhtop.tensix import splat as SP, matmul as MM, llk_run
from bhtop.tensix.resident import boot_resident
from bhtop.tensix.baremetal import BareMetal, bm_coord
from bhtop.goldens import gap2_bin_golden as BIN

_SRC = "/home/starboy/bhtop/src/bhtop/kernels/x280/het/het_x280.c"
# x280 GDDR map (matches het_x280.c / conductor)
X_HDR, X_CAM, X_IDL, X_LOSS = 0x30005000, 0x30005060, 0x300050A0, 0x30005B00
X_DB, X_DONE, X_CMD = 0x30004000, 0x30004010, 0x30004020
FLAG, ACK, ASTRIDE = 0x30006400, 0x30006800, 0x40
NSLOT, IMGW_A, IDLG, ORIG = 0x30005DF0, 0x30005DF4, 0x30005E00, 0x30006200
# on-device bin (het cmd11) plumbed but DEFERRED — proj_sqrt precision on A=c/det (O(100+)) makes the device
# bbox/front-12 selection differ from the host golden (all tiles got the same 12). Host bin below is correct.
NHARTS_A, WCMD_A, IMG_BASE_A = 0x300027F0, 0x300027F4, 0x300027F8
HGO, HDONE, LOSS_H, GACC_X = 0x30002800, 0x30002A00, 0x30002C00, 0x30280000
TGT_BANK = 0x34000000                                 # resident bank: all view images (upload once)
PARAM, TGT_IMG, OPB_O, PXBASE, GINO = 0x30100000, 0x30200000, 0x31000000, 0x32000000, 0x33000000
OPS_O = PXS_O = GIS_O = 0x10000
CFG, HUB = 0x3200, (8, 3)
TILE, K, P = 16, 12, 32
_fb = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]
_bf = lambda u: struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]
_enc = lambda flat: MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
_pad = lambda rc: SP._pad32(rc)


class HetGridEngine:
    def __init__(self, N, imgw, imgh, W=6, NH=4):
        assert imgw % TILE == 0 and imgh % TILE == 0, "IMGW/IMGH must be multiples of 16"
        self.N, self.imgw, self.imgh, self.NH = N, imgw, imgh, NH
        self.on_device_orch = os.environ.get("TT_ONDEV_ORCH", "1") == "1"   # cmd10: bin+dispatch on x280 (host out of loop)
        self.ntx, self.nty = imgw // TILE, imgh // TILE
        self.grp = [[(x, y) for y in range(TILE) for x in range(TILE)][i:i + 32] for i in range(0, TILE * TILE, 32)]
        ctx = init_ttexalens(); self.ctx = ctx
        # W1: DMA readback. 4B mode chops every transfer into 4-byte register accesses (~2.6 MB/s), pinning
        # read_params() (per-step preview + PLY) to the slow path. Disabling it sends each bulk buffer as one
        # DMA transfer (~9.5 GB/s D2H) — the fork tt_umd (0.9.8, BlackholeDmaTransfer::d2h_transfer, verified
        # bit-exact on p150a) must be installed in the env. Gated so a missing DMA .so falls back safely.
        if os.environ.get("TT_DMA_READBACK", "1") == "1":
            ctx.use_4B_mode = False
        allw = [c for c in worker_coords(ctx) if tuple(c.to("noc0"))[0] > 8]   # RIGHT NUMA block
        self.workers = allw[:W]; self.wxy = [tuple(c.to("noc0")) for c in self.workers]
        self.W = len(self.workers)
        self.dev = L2cpu(ctx=ctx)
        # ---- Tensix render workers + resident conductors (must boot BEFORE x280 tiles) ----
        llk_run.build("resident_train_perf", run_type="L1_TO_L1", fidelity="HiFi4", fp32_acc=False,
                      formats=None, overrides={"ITERATIONS": "constexpr int ITERATIONS = 32;"})
        for c in self.workers:
            boot_resident("resident_train_perf", c, ctx=ctx, runtime_words=[1, 1, 1, 1, 1, 128, 128, 0, 4, 4], clear_words=56)
        time.sleep(0.3)
        for c in self.workers: self._stage_static(c)
        # ---- x280 tiles (het multi-hart hub on each tile) ----
        self.tiles = list(self.dev.loc.keys()) if hasattr(self.dev, "loc") else [0]
        words = tc.compile_source(_SRC, base=CODE_ADDR, march="rv64gc")
        for t in self.tiles:
            try: self.dev.bringup(t)
            except Exception: pass
            self.dev.wr(t, X_HDR, [N, 0] + [0]*25); self.dev.wr(t, IMGW_A, [imgw]); self.dev.wr(t, 0x30005DFC, [imgh])
            self.dev.wr(t, NHARTS_A, [NH]); self.dev.wr(t, 0x300027FC, [self.W])   # WORKERS_A for on-device cmd10 dispatch
            for h in range(1, 4): self.dev.wr(t, HGO + h * 0x40, [0]); self.dev.wr(t, HDONE + h * 0x40, [0])
            self.dev.wr(t, LOSS_H, [0] * 64)
            self.dev.wr(t, X_DB, [0]); self.dev.wr(t, X_DONE, [0])
            if t == self.tiles[0]:
                self.dev.wr(t, GACC_X, [0] * (N * 9 * 3))
                for s in range(self.W): self.dev.wr(t, FLAG + s * ASTRIDE, [0]); self.dev.wr(t, ACK + s * ASTRIDE, [0])
            self.dev.load(t, 0, words)
            for _ in range(80):
                if self.dev.telemetry(t, slots=1, hart=0)[0] == 0x48455421: break
                time.sleep(0.03)
            else: raise RuntimeError(f"het leader not resident on tile {t}")
            for h in range(1, NH): self.dev.redirect(t, h, CODE_ADDR)
        time.sleep(0.2)
        cbin = BareMetal.build("conductor")
        for s, c in enumerate(self.workers):
            wr(c, CFG, [bm_coord(*HUB), s, OPB_O + s * OPS_O, GINO + s * GIS_O, FLAG + s * ASTRIDE,
                        ACK + s * ASTRIDE, PXBASE + s * PXS_O, 0, 0, 8], context=ctx)
            BareMetal(*self.wxy[s], ctx=ctx, risc="brisc").run(cbin)
        time.sleep(0.2)
        M = PARAM + N * 14 * 4; V = M + N * 14 * 4; GACC = V + N * 14 * 4
        self.dev.wr(0, M, [0] * (N * 14)); self.dev.wr(0, V, [0] * (N * 14)); self.dev.wr(0, GACC, [0] * (N * 9))

    def _stage_static(self, coord):
        Ppair = [[0.0] * K for _ in range(2 * K)]
        for i in range(K): Ppair[2 * i][i] = -0.5; Ppair[2 * i + 1][i] = -0.5
        Mcomb = [[(1.0 if r < c else 0.0) for c in range(K)] for r in range(2 * K)]
        for i in range(K): Mcomb[K + i][i] = 1.0
        Stri = [Mcomb[r] for r in range(K)]; PpairT = [[Ppair[r][c] for r in range(2 * K)] for c in range(K)]
        U = [[1.0 if j > i else 0.0 for i in range(K)] for j in range(K)]
        H = dict(Ppair=0x23000, Stri=0x26000, PpairT=0x2C000, U=0x2D000)
        for nm, m in [("Ppair", Ppair), ("Stri", Stri), ("PpairT", PpairT), ("U", U)]:
            wr(coord, H[nm], _enc(_pad(m)), context=self.ctx)
        wr(coord, 0x27000, _enc([1.0 if r == c else 0.0 for r in range(32) for c in range(32)]), context=self.ctx)  # Iden
        wr(coord, 0x2F000, _enc([1.0] * 1024), context=self.ctx)                # ones
        wr(coord, 0x30000, _enc(_pad([[1.0] * P])), context=self.ctx)           # ones1P

    def set_params(self, params14):
        p = np.asarray(params14, np.float64).reshape(self.N, 14)
        self.dev.wr(0, PARAM, [_fb(p[o, j]) for o in range(self.N) for j in range(14)])

    def set_views(self, imgs):
        """Upload ALL view target images to GDDR ONCE (kills the per-step gt upload). imgs=[V,H,W,3] in
        (y,x,ch). step(view_idx=v) then just points IMG_BASE at TGT_BANK+v*stride (a 1-word write)."""
        imgs = np.asarray(imgs, np.float32)
        self._nviews = imgs.shape[0]; self._img_words = imgs.shape[1] * imgs.shape[2] * 3
        self._img_stride = self._img_words * 4                       # bytes per view
        for v in range(self._nviews):
            self.dev.wr(0, TGT_BANK + v * self._img_stride, [_fb(x) for x in imgs[v].reshape(-1)])
        self._views_resident = True

    def read_params(self):
        w = self.dev.rdn(0, PARAM, self.N * 14)
        return np.array([[_bf(w[o * 14 + j]) for j in range(14)] for o in range(self.N)], np.float64)

    def _het(self, cmd, extra=None, tile=0, timeout=12.0):
        if extra: self.dev.wr(tile, X_HDR, extra)
        self.dev.wr(tile, X_CMD, [cmd]); r = self.dev.rd(tile, X_DB) + 1; self.dev.wr(tile, X_DB, [r])
        t = time.time()
        while self.dev.rd(tile, X_DONE) != r and time.time() - t < timeout: time.sleep(0.0003)
        return self.dev.telemetry(tile, slots=8, hart=0)
        
    def _het_multi(self, cmd, extras, tiles):
        rs = []
        for i, t in enumerate(tiles):
            if extras[i]: self.dev.wr(t, X_HDR, extras[i])
            self.dev.wr(t, X_CMD, [cmd])
            r = self.dev.rd(t, X_DB) + 1
            self.dev.wr(t, X_DB, [r])
            rs.append(r)
        t0 = time.time()
        done = [False] * len(tiles)
        while not all(done) and time.time() - t0 < 12.0:
            for i, t in enumerate(tiles):
                if not done[i] and self.dev.rd(t, X_DONE) == rs[i]:
                    done[i] = True
            time.sleep(0.0003)
        return [self.dev.telemetry(t, slots=8, hart=0) for t in tiles]

    def step(self, cam16, tgt_flat, lr14, step, view_idx=None):
        """cam16 = [Rv(9), tv(3), fx, fy, cx, cy]; tgt_flat = imgh*imgw*3 f32 (y,x,ch); lr14 = per-param LR.
        Returns scalar loss. Bins on host from device projection; orchestrates all tiles across W workers/NH harts."""
        T = {}; _c = time.time
        t = _c(); self.dev.wr(0, X_CAM, [_fb(x) for x in cam16])
        if view_idx is not None and getattr(self, "_views_resident", False):
            self.dev.wr(0, IMG_BASE_A, [TGT_BANK + view_idx * self._img_stride])   # resident: 1-word write
        else:
            self.dev.wr(0, TGT_IMG, [_fb(v) for v in tgt_flat]); self.dev.wr(0, IMG_BASE_A, [TGT_IMG])
        T["gt_up"] = _c() - t
        t = _c(); self._het(2, extra=[self.N, step]); T["proj"] = _c() - t     # project+whiten all params (multi-hart)
        if self.on_device_orch:
            # FULLY ON-DEVICE: x280 bins (cmd11 body) + autonomously loops over occupied tiles dispatching batches
            # to the W workers. Host issues ONE doorbell and reads a scalar occ count. No PUB readback, no host bin,
            # no per-batch cmd9 relay.
            t = _c(); tele = self._het(10, timeout=300.0); occ_count = int(tele[4]); T["orch10"] = _c() - t
        else:
            off = self.N * 61                                                 # PUB float offset (host-bin reference path)
            t = _c(); pub = np.array([[_bf(u) for u in self.dev.rdn(0, PARAM + (off + o * 6) * 4, 6)] for o in range(self.N)])
            tiles, ntx, nty = BIN.bin_tiles(pub[:, 0], pub[:, 1], pub[:, 2:5], pub[:, 5],
                                            self.imgw, self.imgh, tile=TILE, cap=64)
            occ = [tl for tl in range(ntx * nty) if tiles[tl]]; T["pub_bin"] = _c() - t
            t = _c(); twr = 0.0
            for b0 in range(0, len(occ), self.W):
                batch = occ[b0:b0 + self.W]; ns = len(batch)
                tw = _c()
                for s, tl in enumerate(batch):
                    ids = list(tiles[tl][:K]); ids += [ids[-1]] * (K - len(ids)) if ids else [0] * K
                    ox, oy = (tl % ntx) * TILE, (tl // ntx) * TILE
                    self.dev.wr(0, IDLG + s * 0x40, [K] + ids); self.dev.wr(0, ORIG + s * 8, [ox, oy])
                self.dev.wr(0, NSLOT, [ns]); twr += _c() - tw
                self._het(9)
            T["batch"] = _c() - t; T["idlg_wr"] = twr; occ_count = len(occ)
        t = _c(); bc1 = 1.0 / (1 - 0.9 ** step); bc2 = 1.0 / (1 - 0.999 ** step)
        
        base_ext = [self.N, step, _fb(bc1), _fb(bc2), _fb(0.9), _fb(0.999), _fb(1e-8)] + [_fb(x) for x in lr14]
        base_ext += [0, 0, 0]  # pad to hdr[24]
        
        chunk = math.ceil(self.N / len(self.tiles))
        extras = []
        for i in range(len(self.tiles)):
            start_g = i * chunk
            end_g = min((i + 1) * chunk, self.N)
            extras.append(base_ext + [start_g, end_g])
            
        self._het_multi(1, extras, self.tiles)
        
        # Loss is computed across all tiles, but wait, loss is stored in LOSS_H per tile.
        # Let's just read the loss from Tile 0 for now (or sum them). Tile 0 computes its chunk's loss.
        # Actually, each tile computes loss for its slice. We should sum them.
        loss = sum(_bf(self.dev.rdn(t, X_LOSS, 1)[0]) for t in self.tiles)
        T["adam"] = _c() - t
        self.last_timing = T
        return loss, occ_count
