"""HetGridEngine — the x280-orchestrated, multi-hart, fully-on-device 3DGS fused-training engine, packaged as a
reusable class (the standalone train_het_orch_grid.py is the reference script it was lifted from). Boots once
(render workers + resident conductors + het multi-hart hub), then per step: upload camera + target image ->
project(cmd2) -> host bins tiles -> orchestrate batches(cmd9, all workers concurrent, NH harts) -> Adam(cmd1)
-> read scalar loss. Params live resident on the x280; host issues doorbells + reads a loss. Supports N>16,
rectangular IMGW x IMGH, and a real pinhole camera per step.

Contract: boot in __init__, set_params([N,14]) / read_params()->[N,14], step(cam16, tgt_flat)->loss."""
import struct, time, math
import numpy as np
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device as rd, read_words_from_device as rds, write_words_to_device as wr
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
from bhtop.tensix.loader import worker_coords
from bhtop.tensix import splat as SP, matmul as MM, llk_run
from bhtop.tensix.resident import boot_resident
from bhtop.tensix.baremetal import BareMetal, bm_coord
import gap2_bin_golden as BIN

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
        self.ntx, self.nty = imgw // TILE, imgh // TILE
        self.grp = [[(x, y) for y in range(TILE) for x in range(TILE)][i:i + 32] for i in range(0, TILE * TILE, 32)]
        ctx = init_ttexalens(); self.ctx = ctx
        allw = [c for c in worker_coords(ctx) if tuple(c.to("noc0"))[0] > 8]   # RIGHT NUMA block
        self.workers = allw[:W]; self.wxy = [tuple(c.to("noc0")) for c in self.workers]
        self.W = len(self.workers)
        self.dev = L2cpu(ctx=ctx)
        try: self.dev.bringup(0)
        except Exception: pass
        # render kernel + resident conductor on each worker; het multi-hart hub
        llk_run.build("resident_train_perf", run_type="L1_TO_L1", fidelity="HiFi4", fp32_acc=False,
                      formats=None, overrides={"ITERATIONS": "constexpr int ITERATIONS = 32;"})
        for c in self.workers:
            boot_resident("resident_train_perf", c, ctx=ctx, runtime_words=[1, 1, 1, 1, 1, 128, 128, 0, 4, 4], clear_words=56)
        time.sleep(0.3)
        for c in self.workers: self._stage_static(c)
        self.dev.wr(0, X_HDR, [N, 0]); self.dev.wr(0, IMGW_A, [imgw]); self.dev.wr(0, 0x30005DFC, [imgh])
        self.dev.wr(0, NHARTS_A, [NH])
        for h in range(1, 4): self.dev.wr(0, HGO + h * 0x40, [0]); self.dev.wr(0, HDONE + h * 0x40, [0])
        self.dev.wr(0, GACC_X, [0] * (N * 9 * 3)); self.dev.wr(0, LOSS_H, [0] * 64)
        for s in range(self.W): self.dev.wr(0, FLAG + s * ASTRIDE, [0]); self.dev.wr(0, ACK + s * ASTRIDE, [0])
        self.dev.wr(0, X_DB, [0]); self.dev.wr(0, X_DONE, [0])
        words = tc.compile_source(_SRC, base=CODE_ADDR, march="rv64gc")
        self.dev.load(0, 0, words)
        for _ in range(80):
            if self.dev.telemetry(0, slots=1, hart=0)[0] == 0x48455421: break
            time.sleep(0.03)
        else: raise RuntimeError("het leader not resident (tt-smi -r 0)")
        for h in range(1, NH): self.dev.redirect(0, h, CODE_ADDR)
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

    def _het(self, cmd, extra=None):
        if extra: self.dev.wr(0, X_HDR, extra)
        self.dev.wr(0, X_CMD, [cmd]); r = self.dev.rd(0, X_DB) + 1; self.dev.wr(0, X_DB, [r])
        t = time.time()
        while self.dev.rd(0, X_DONE) != r and time.time() - t < 12.0: time.sleep(0.0003)
        return self.dev.telemetry(0, slots=8, hart=0)

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
        off = self.N * 61                                                     # PUB float offset
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
        T["batch"] = _c() - t; T["idlg_wr"] = twr
        t = _c(); bc1 = 1.0 / (1 - 0.9 ** step); bc2 = 1.0 / (1 - 0.999 ** step)
        self._het(1, extra=[self.N, step, _fb(bc1), _fb(bc2), _fb(0.9), _fb(0.999), _fb(1e-8)] + [_fb(x) for x in lr14])
        loss = _bf(self.dev.rdn(0, X_LOSS, 1)[0]); T["adam"] = _c() - t
        self.last_timing = T
        return loss, len(occ)
