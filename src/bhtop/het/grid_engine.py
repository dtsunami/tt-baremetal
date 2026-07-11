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
X_PROG = 0x30004030                                         # live breadcrumb (phase<<24|index) the leader writes; host polls it
_PHASE = {1: "bin", 2: "dispatch", 3: "consume", 4: "adam", 5: "proj"}
def _decode_prog(p):
    return f"phase={_PHASE.get((p >> 24) & 0xFF, '?')}({(p>>24)&0xFF}) idx={p & 0xFFFFFF} (0x{p:08x})"
FLAG, ACK, ASTRIDE = 0x30006400, 0x30006800, 0x40
NSLOT, IMGW_A, IDLG, ORIG = 0x30005DF0, 0x30005DF4, 0x30005E00, 0x30006200
# on-device bin (het cmd11) plumbed but DEFERRED — proj_sqrt precision on A=c/det (O(100+)) makes the device
# bbox/front-12 selection differ from the host golden (all tiles got the same 12). Host bin below is correct.
NHARTS_A, WCMD_A, IMG_BASE_A = 0x300027F0, 0x300027F4, 0x300027F8
HGO, HDONE, LOSS_H = 0x30002800, 0x30002A00, 0x30002C00
# het-barrier hardening (must match het_x280.c): per-hart heartbeat + breadcrumb, watchdog gate, error flag
WHB, WDIAG, WDOG_EN, WERR = 0x30002400, 0x30002500, 0x30002600, 0x30002604
_WS = {0: "idle", 1: "enter", 2: "produce", 3: "signal", 4: "WAIT-ACK", 5: "consume", 6: "done", 10: "proj", 11: "adam"}
# DYNAMIC GDDR map — must match het_x280.c::lay() EXACTLY. PARAM chain + gacc_x + tgt_img are N/image-derived
# (see _layout); the worker/bin/view banks below are FIXED HIGH (independent of N so densify-resize can't move
# them and the conductor CFG stays valid). GACC_X/TGT_IMG are computed per-N in _layout(), not constants.
PARAM = 0x30100000
OPB_O, PXBASE, GINO = 0x60000000, 0x61000000, 0x62000000    # per-slot operand / pixel / grad-inbox banks
TGT_BANK = 0x66000000                                        # resident bank: all view images (upload once)
DESC = 0x68000000                                            # W4 worker-produce: per-slot compact coeff descriptor
DESC_STRIDE = 0x800                                          # must match het_x280.c DESC_STRIDE
GCOMPACT = 0x6A000000                                        # W4 STAGE-2 worker-consume: per-slot compact grad bank
GC_STRIDE = 0x800                                            # must match het_x280.c GC_STRIDE
WPROD_A = 0x300027E0                                         # x280 worker-produce flag (must match het_x280.c)
WCONS_A = 0x300027D0                                         # x280 worker-consume flag (must match het_x280.c)
BINCAP_A = 0x300027D4                                        # max per-Gaussian bin half-span in tiles (must match het_x280.c)
BINCAP_N_A = 0x300027D8                                      # kernel-written count of clamped (degenerate) splats
NAN_N_A = 0x300027DC                                         # kernel-written count of non-finite splats skipped (NaN/inf guard)
NOCPACE_A = 0x300027E4                                       # drain NoC store buffer every N Gaussians/hart (0=off) — caps 4-hart NIU flood
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
        self.wprod = os.environ.get("TT_BM_WPRODUCE", "0") == "1"           # W4: conductor tilizes locally (produce off the hub)
        self.wcons = os.environ.get("TT_BM_WCONSUME", "0") == "1"           # W4 STAGE-2: conductor detilizes+MACs (consume off the hub)
        self.wdog = os.environ.get("TT_BM_WDOG", "1") == "1"                # het-barrier: watchdog (fix) vs legacy 40M spin
        self.bincap = int(os.environ.get("TT_BM_BINCAP", "8"))              # bin: max per-splat half-span (tiles); big => off
        self.nocpace = int(os.environ.get("TT_BM_NOCPACE", "0"))            # proj/adam: drain NoC store buffer every N Gaussians/hart
                                                                            # (0=off) — fixes the 4-hart concurrent-GDDR NIU wedge
        self.diag = os.environ.get("TT_BM_DIAG", "0") == "1"                # capture per-phase barrier breadcrumbs
        self.diag_log = []; self.last_werr = 0
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
        # NoC-NIU clock-gate CANDIDATE FIX (KMD SET_POWER_STATE, MRISC|L2CPU|TENSIX busy set): hold the power-
        # domain flags on our pyluwen fd for this run's lifetime so the L2CPU domain can't idle-gate. Suspected
        # root of the reproducible het tile hang: the pipeline drives the chip over exalens and never calls
        # set_power, so cmfwcfg cg_en=1 lets the L2CPU NIU clock-gate in idle windows (render-wait, step
        # boundary) and the racy gate-exit wedges the tile (X_DONE read / X_CAM DMA timeout). Runtime,
        # reversible, safe BAR/ARC transport (no NoC poke into the L2CPU's fatal NIU window). OPT-IN
        # (TT_BM_HOLD_POWER=1) — default off after a host-freeze incident; enable only for a deliberate, watched
        # test with tt-smi -r 0 / power-cycle on standby.
        if os.environ.get("TT_BM_HOLD_POWER", "0") == "1":
            ok = self.dev.hold_power()
            print(f"[grid_engine] hold_power(MRISC|L2CPU|TENSIX): {'HELD — L2CPU clock-gate disabled' if ok else 'FAILED'}", flush=True)
        # OVERCLOCK (opt-in): TT_BM_X280_MHZ (+ TT_BM_VCORE_MV) — applied in _apply_oc() AFTER bringup (which leaves
        # the PLL at 1750/postdiv=1, the base for the fbdiv overclock). Measured max within the firmware vdd_max=900
        # clamp = 2400 MHz @ 900 mV (Vmin 882 + guardband; hart3-limited, memory bh-arc-dvfs-voltage) → +37% over
        # 1750, 3x the 800 default, on the x280-bound bin/proj/adam. No over-volt (<=900), no SPIROM edit needed.
        self._oc_mhz = int(os.environ.get("TT_BM_X280_MHZ", "0"))
        self._oc_mv  = int(os.environ.get("TT_BM_VCORE_MV", "0"))
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
            self.dev.wr(t, WPROD_A, [1 if self.wprod else 0])                       # W4 worker-produce flag
            self.dev.wr(t, WCONS_A, [1 if self.wcons else 0])                       # W4 STAGE-2 worker-consume flag
            self.dev.wr(t, WDOG_EN, [1 if self.wdog else 0]); self.dev.wr(t, WERR, [0])   # het-barrier watchdog gate + error
            self.dev.wr(t, BINCAP_A, [self.bincap & 0xFFFFFFFF]); self.dev.wr(t, BINCAP_N_A, [0])   # bin degenerate-splat cap
            self.dev.wr(t, NOCPACE_A, [self.nocpace & 0xFFFFFFFF])                                 # NoC-pacing (4-hart NIU wedge fix)
            for h in range(4): self.dev.wr(t, WHB + h * 0x40, [0] * 4); self.dev.wr(t, WDIAG + h * 0x40, [0] * 8)
            for h in range(1, 4): self.dev.wr(t, HGO + h * 0x40, [0]); self.dev.wr(t, HDONE + h * 0x40, [0])
            self.dev.wr(t, LOSS_H, [0] * 64)
            self.dev.wr(t, X_DB, [0]); self.dev.wr(t, X_DONE, [0])
            if t == self.tiles[0]:
                if NH > 1: self.dev.wr(t, self._layout()["gacc_x"], [0] * (N * 9 * (NH - 1)))   # gacc_x unused at NH=1 (empty write)
                for s in range(self.W): self.dev.wr(t, FLAG + s * ASTRIDE, [0]); self.dev.wr(t, ACK + s * ASTRIDE, [0])
            self.dev.load(t, 0, words)
            for _ in range(80):
                if self.dev.telemetry(t, slots=1, hart=0)[0] == 0x48455421: break
                time.sleep(0.03)
            else: raise RuntimeError(f"het leader not resident on tile {t}")
            _dead = int(os.environ.get("TT_BM_DEAD_HART", "-1"))   # DIAG: skip a hart's boot -> its heartbeat stays
            for h in range(1, NH):                                 # frozen (genuinely dead) to test the watchdog abort
                if h == _dead: print(f"[grid_engine] DIAG: NOT booting hart {h} (frozen-heartbeat test)", flush=True); continue
                self.dev.redirect(t, h, CODE_ADDR)
        time.sleep(0.2)
        cbin = BareMetal.build("conductor")
        _LO = self._layout()   # cfg[7]=this slot's descriptor, cfg[8]=target-image base, cfg[10]=wprod, cfg[11]=imgw
        stall = set(int(x) for x in os.environ.get("TT_BM_STALL_SLOT", "").split(",") if x.strip())  # DIAG: slots whose
        for s, c in enumerate(self.workers):                        # conductor we DON'T boot -> their ACK never comes
            wr(c, CFG, [bm_coord(*HUB), s, OPB_O + s * OPS_O, GINO + s * GIS_O, FLAG + s * ASTRIDE,
                        ACK + s * ASTRIDE, PXBASE + s * PXS_O, DESC + s * DESC_STRIDE, _LO["tgt_img"], 8,
                        1 if self.wprod else 0, self.imgw,
                        1 if self.wcons else 0, GCOMPACT + s * GC_STRIDE], context=ctx)   # cfg[12]=wcons cfg[13]=compact
            if s in stall: print(f"[grid_engine] DIAG: NOT booting conductor slot {s} (induced ack stall)", flush=True); continue
            BareMetal(*self.wxy[s], ctx=ctx, risc="brisc").run(cbin)
        time.sleep(0.2)
        M = PARAM + N * 14 * 4; V = M + N * 14 * 4; GACC = V + N * 14 * 4
        self.dev.wr(0, M, [0] * (N * 14)); self.dev.wr(0, V, [0] * (N * 14)); self.dev.wr(0, GACC, [0] * (N * 9))
        self._apply_oc()

    def _apply_oc(self):
        """Overclock the x280 core to TT_BM_X280_MHZ @ TT_BM_VCORE_MV — VOLTAGE LEADS FREQUENCY. Called after
        bringup (PLL at 1750/postdiv=1 = the fbdiv-overclock base). Ramps vcore up in <=40 mV steps, then glides
        the PLL fbdiv. Best-effort: any failure reverts the PLL to the 1750 baseline so training still runs.
        2400 MHz @ 900 mV is the measured max inside the firmware vdd_max=900 clamp — no over-volt, no flash edit."""
        if not (self._oc_mhz or self._oc_mv or os.environ.get("TT_BM_PERF_BUSY") == "1"): return
        _pl = lambda m: print(f"[grid_engine] oc: {m}", flush=True)
        try:
            _pl(f"limits {self.dev.limits()}  start vcore {self.dev.power().get('vcore_mv')} mV")
            if os.environ.get("TT_BM_PERF_BUSY") == "1":               # AICLK OC: GO_BUSY -> Tensix render 800->1350 (1.69x)
                self.dev.perf_busy(); time.sleep(0.3)                  # opt-in (engages aiclk_ppm governor); FORCE_AICLK
                _pl(f"GO_BUSY -> aiclk {self.dev.power().get('aiclk_mhz')} MHz")   # alone doesn't take at idle. 1350=asic_fmax ceiling.
            if self._oc_mv:                                             # VOLTAGE (x280) — force AFTER GO_BUSY so it pins past AVS
                cur = self.dev.power().get("vcore_mv") or 720
                while cur < self._oc_mv:
                    cur = min(cur + 40, self._oc_mv); self.dev.force_vdd(cur, allow_step=True); time.sleep(0.15)
                _pl(f"vcore -> {self.dev.power().get('vcore_mv')} mV")
            if self._oc_mhz and self._oc_mhz != 1750:                 # THEN frequency (fbdiv overclock) — only if requested
                fb = max(140, min(round(140 + (self._oc_mhz - 1750) / 12.5), self.dev.EXPLORE_FBDIV_MAX))
                r = self.dev.set_fbdiv_explore(fb); _pl(f"PLL fbdiv {fb} -> {r['core_mhz'][0]} MHz")
            mon = self.dev.monitor()
            _pl(f"APPLIED: x280 {self.dev.clocks()['core_l2cpu_mhz'][0]} MHz / aiclk {mon.get('aiclk_mhz')} MHz "
                f"@ {mon.get('vcore_mv')} mV  {mon.get('asic_temp_c')}C  alarms={mon['alarms']}")
            # only revert on THERMAL/THROTTLE danger — a few-mV vcore>vdd_max overshoot at the clamp is expected
            # (regulator granularity) and harmless, so don't let it trip a needless revert.
            _t, _thr = mon.get("asic_temp_c"), mon.get("throttler")
            if (isinstance(_t, (int, float)) and _t > 87) or (_thr not in (0, None)):
                raise RuntimeError(f"thermal/throttle unsafe: temp={_t}C throttler={_thr}")
        except Exception as e:                                         # noqa: BLE001
            _pl(f"OC FAILED ({type(e).__name__}: {e}) — reverting PLL to 1750 baseline")
            try: self.dev.set_fbdiv_explore(140)
            except Exception: pass

    def restore_clocks(self):
        """Best-effort return to idle (call on teardown): PLL 200 + drop the forced rail + hand back to the governor."""
        try:
            self.dev.set_core_freq(200)
            cur = self.dev.power().get("vcore_mv") or 900
            while cur > 720: cur = max(cur - 40, 710); self.dev.force_vdd(cur, allow_step=True); time.sleep(0.1)
            self.dev.perf_idle()
        except Exception as e:                                         # noqa: BLE001
            print(f"[grid_engine] restore_clocks failed: {e}", flush=True)

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

    def _layout(self, N=None):
        """DYNAMIC GDDR layout — MUST match het_x280.c::lay() EXACTLY. The param chain + gacc_x + current
        target image are N/image-derived (grow from PARAM); worker/bin/view banks are fixed-high. Recomputed
        from the CURRENT N so it shifts freely on densify-resize. Raises if the dynamic region would overrun
        the fixed worker banks (loud over-capacity guard vs silent corruption)."""
        N = self.N if N is None else int(N)
        algn = lambda x: (x + 0xFFF) & ~0xFFF
        m = PARAM + N * 14 * 4; v = m + N * 14 * 4; gacc = v + N * 14 * 4
        coeff = gacc + N * 9 * 4; depth = coeff + N * 9 * 4; pub = depth + N * 4
        gacc_x = algn(pub + N * 6 * 4)
        tgt_img = algn(gacc_x + (self.NH - 1) * N * 9 * 4)
        top = algn(tgt_img + self.imgw * self.imgh * 3 * 4)
        if top > OPB_O:
            raise RuntimeError(
                f"dynamic GDDR layout overflow: N={N}, {self.imgw}x{self.imgh} -> dynamic top 0x{top:x} exceeds "
                f"the worker-bank base 0x{OPB_O:x}. Lower TT_MAX_POINTS or TT_SIZE (ceiling ~2M Gaussians @ 1600px).")
        return dict(param=PARAM, m=m, v=v, gacc=gacc, coeff=coeff, depth=depth, pub=pub,
                    gacc_x=gacc_x, tgt_img=tgt_img, top=top, opb_o=OPB_O, pxbase=PXBASE, gino=GINO,
                    tgt_bank=TGT_BANK, desc=DESC)

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

    def set_view(self, v, img):
        """Upload ONE view to the resident bank (lazy alternative to set_views — for trainers that receive gt one
        view at a time). Sets the stride + resident flag on first call; thereafter step(view_idx=v) references it
        with a 1-word IMG_BASE write instead of re-uploading the whole gt each step. img = [H,W,3] f32."""
        img = np.asarray(img, np.float32)
        if not getattr(self, "_views_resident", False):
            self._img_words = img.shape[0] * img.shape[1] * 3
            self._img_stride = self._img_words * 4
            self._views_resident = True
        self.dev.wr(0, TGT_BANK + int(v) * self._img_stride, [_fb(x) for x in img.reshape(-1)])

    def read_params(self):
        w = self.dev.rdn(0, PARAM, self.N * 14)
        return np.array([[_bf(w[o * 14 + j]) for j in range(14)] for o in range(self.N)], np.float64)

    def read_screen(self):
        """W5 pose-opt: per-Gaussian screen state + mean-projection grads, reconstructed on host from GDDR (NO
        kernel change). Returns dict(u,v,zc,du,dv,valid) — the same contract as device_resident.last_screen —
        where du=dL/du, dv=dL/dv are exactly the whitening-backward the x280 Adam computes (het_x280.c:239-242).
        Sources already resident after engine.step(): PUB[N*6]=[u,v,a,b,c,zc] at PARAM+N*61w, per-hart grad
        accumulators gacc0 at PARAM+N*42w (hart 0) + GACC_X partitions (harts 1..NH-1). MUST be called AFTER
        step() and BEFORE the next step's cmd2 (which zeroes gacc). Single-hub (tile 0) assumption."""
        N = self.N; LO = self._layout()
        pub = np.array([_bf(x) for x in self.dev.rdn(0, LO["pub"], N * 6)], np.float64).reshape(N, 6)
        u, v, a, b, c, zc = (pub[:, j] for j in range(6))
        ga = np.array([_bf(x) for x in self.dev.rdn(0, LO["gacc"], N * 9)], np.float64).reshape(N, 9)
        for h in range(1, self.NH):                                  # merge the extra harts (else ~1/NH of grad)
            xw = self.dev.rdn(0, LO["gacc_x"] + (h - 1) * N * 9 * 4, N * 9)
            ga = ga + np.array([_bf(x) for x in xw], np.float64).reshape(N, 9)
        d_tx, d_ty = ga[:, 2], ga[:, 4]
        asafe = np.maximum(a, 1e-8)
        sa = np.sqrt(asafe); m12 = b / sa
        t = np.maximum(c - b * b / asafe, 1e-8); m22 = np.sqrt(t)   # mirror het_x280.c:240
        du = d_tx * (-sa)                                            # = g_gx (dL/du)
        dv = d_tx * (-m12) + d_ty * (-m22)                          # = g_gy (dL/dv)
        valid = np.isfinite(u) & np.isfinite(v) & np.isfinite(du) & np.isfinite(dv) & (zc > 0.2)
        return dict(u=u, v=v, zc=zc, du=du, dv=dv, valid=valid)

    # ---- W5 densify-resize: N is runtime on the x280 (read from each command header) → resize in place ----
    def cap(self):
        """Max N in the DYNAMIC GDDR map: the param chain (67w) + gacc_x ((NH-1)*9w) + current target image must
        stay below the fixed worker-bank base OPB_O. Depends on image size — ~2M Gaussians @ 1600px."""
        per = (67 + (self.NH - 1) * 9) * 4                     # dynamic-region bytes per Gaussian
        img = self.imgw * self.imgh * 3 * 4
        return max(1, (OPB_O - PARAM - img - 0x10000) // per)  # -64 KiB alignment slack

    def read_moments(self):
        """Adam m/v [N,14] resident at PARAM+N*14w / PARAM+N*28w (het_x280.c:225). For momentum-preserving edits."""
        m = np.array([_bf(x) for x in self.dev.rdn(0, PARAM + self.N * 14 * 4, self.N * 14)], np.float64).reshape(self.N, 14)
        v = np.array([_bf(x) for x in self.dev.rdn(0, PARAM + self.N * 28 * 4, self.N * 14)], np.float64).reshape(self.N, 14)
        return m, v

    def set_moments(self, m14, v14):
        m = np.asarray(m14, np.float64).reshape(self.N, 14); v = np.asarray(v14, np.float64).reshape(self.N, 14)
        self.dev.wr(0, PARAM + self.N * 14 * 4, [_fb(m[o, j]) for o in range(self.N) for j in range(14)])
        self.dev.wr(0, PARAM + self.N * 28 * 4, [_fb(v[o, j]) for o in range(self.N) for j in range(14)])

    def resize(self, newP14):
        """Change N in place — NO reboot/recompile (Tensix workers are N-independent; the x280 derives all
        m/v/gacc offsets from hdr[0]=N each command). Re-lays PARAM and ZEROES m/v/gacc at the new N-derived
        offsets (momentum reset; caller re-sets sliced momentum via set_moments for keep-mask edits). Single-hub."""
        p = np.asarray(newP14, np.float64).reshape(-1, 14); newN = p.shape[0]
        assert 0 < newN <= self.cap(), f"resize N={newN} exceeds dynamic-map ceiling {self.cap()}"
        self.N = newN
        self.set_params(p)
        LO = self._layout()                                   # recomputed at the NEW N (offsets shift)
        self.dev.wr(0, LO["m"], [0] * (newN * 14)); self.dev.wr(0, LO["v"], [0] * (newN * 14)); self.dev.wr(0, LO["gacc"], [0] * (newN * 9))
        for h in range(1, self.NH):
            self.dev.wr(0, LO["gacc_x"] + (h - 1) * newN * 9 * 4, [0] * (newN * 9))

    def hart_diag(self, tile=0):
        """Read the het-barrier breadcrumbs — the mid-hang state the leader + workers publish. Returns a dict:
        werr, aborted, break counts, per-hart {hb, state, ring, slot, ackspin, ns, hgo, hdone}, per-slot
        {flag, ack}. `hb` (heartbeat) advancing between calls => that hart is alive; a WAIT-ACK state with a
        FROZEN hb + a live slot's ACK != its FLAG pins a missing-ack stall on that slot's Tensix conductor."""
        tel = self.dev.telemetry(tile, slots=32, hart=0)
        harts = []
        for h in range(self.NH):
            hb = self.dev.rd(tile, WHB + h * 0x40)
            d = self.dev.rdn(tile, WDIAG + h * 0x40, 5)
            harts.append(dict(h=h, hb=hb, state=_WS.get(d[0], d[0]), ring=hex(d[1]), slot=d[2], ackspin=d[3],
                              ns=d[4], hgo=hex(self.dev.rd(tile, HGO + h * 0x40)),
                              hdone=hex(self.dev.rd(tile, HDONE + h * 0x40))))
        slots = [dict(s=s, flag=hex(self.dev.rd(tile, FLAG + s * ASTRIDE)),
                      ack=hex(self.dev.rd(tile, ACK + s * ASTRIDE))) for s in range(self.W)]
        return dict(werr=hex(self.dev.rd(tile, WERR)), aborted=tel[13], wdog_breaks=tel[31], legacy_breaks=tel[30],
                    last_break=dict(cmd=tel[24], h=tel[25], ring=hex(tel[26]), hdone=hex(tel[27]),
                                    wstate=_WS.get(tel[28], tel[28]), wslot=tel[29]),
                    harts=harts, slots=slots)

    def _het(self, cmd, extra=None, tile=0, timeout=12.0):
        if extra: self.dev.wr(tile, X_HDR, extra)
        self.dev.wr(tile, X_CMD, [cmd]); r = self.dev.rd(tile, X_DB) + 1; self.dev.wr(tile, X_DB, [r])
        t = time.time()
        if os.environ.get("TT_BM_BREADCRUMB") == "1":       # poll the device breadcrumb; report it on a NoC hang
            last = 0
            while time.time() - t < timeout:
                try:
                    last = self.dev.rd(tile, X_PROG)         # last-good progress read BEFORE the NoC dies
                    if self.dev.rd(tile, X_DONE) == r: break
                except Exception as e:                       # NoC0 hung -> the wedging op is the last breadcrumb
                    raise RuntimeError(f"het NoC-WEDGE cmd{cmd} tile{tile}: last breadcrumb {_decode_prog(last)}. "
                                       f"Recover: tt-smi -r 0") from e
                time.sleep(0.0003)
        else:
            while self.dev.rd(tile, X_DONE) != r and time.time() - t < timeout: time.sleep(0.0003)
        self.last_werr = self.dev.rd(tile, WERR)
        if self.wdog and self.last_werr:                    # dead hart aborted the barrier — fail loud, not silent
            raise RuntimeError(f"het barrier aborted (WERR=0x{self.last_werr:08x}): a worker hart's heartbeat froze "
                               f"mid-cmd{cmd} (dead/wedged). Recover with `tt-smi -r 0`. diag={self.hart_diag(tile)}")
        return self.dev.telemetry(tile, slots=16, hart=0)   # slots 8-12 carry the cmd10 W4 profile breakdown
        
    def _het_multi(self, cmd, extras, tiles):
        rs = []
        for i, t in enumerate(tiles):
            if extras[i]: self.dev.wr(t, X_HDR, extras[i])
            self.dev.wr(t, X_CMD, [cmd])
            r = self.dev.rd(t, X_DB) + 1
            self.dev.wr(t, X_DB, [r])
            rs.append(r)
        t0 = time.time()
        done = [False] * len(tiles); bc = os.environ.get("TT_BM_BREADCRUMB") == "1"; last = {t: 0 for t in tiles}
        while not all(done) and time.time() - t0 < 12.0:
            for i, t in enumerate(tiles):
                if done[i]: continue
                try:
                    if bc: last[t] = self.dev.rd(t, X_PROG)
                    if self.dev.rd(t, X_DONE) == rs[i]: done[i] = True
                except Exception as e:                       # which tile wedged, and on which op
                    if bc: raise RuntimeError(f"het NoC-WEDGE cmd{cmd} tile{t}: last breadcrumb {_decode_prog(last[t])}. "
                                              f"Recover: tt-smi -r 0") from e
                    raise
            time.sleep(0.0003)
        for t in tiles:
            self.last_werr = self.dev.rd(t, WERR)
            if self.wdog and self.last_werr:
                raise RuntimeError(f"het barrier aborted (WERR=0x{self.last_werr:08x}) on tile {t} in cmd{cmd} "
                                   f"(dead/wedged hart). Recover with `tt-smi -r 0`. diag={self.hart_diag(t)}")
        return [self.dev.telemetry(t, slots=8, hart=0) for t in tiles]

    def step(self, cam16, tgt_flat, lr14, step, view_idx=None):
        """cam16 = [Rv(9), tv(3), fx, fy, cx, cy]; tgt_flat = imgh*imgw*3 f32 (y,x,ch); lr14 = per-param LR.
        Returns scalar loss. Bins on host from device projection; orchestrates all tiles across W workers/NH harts."""
        T = {}; _c = time.time
        self.dev.wr(0, WERR, [0])                                                   # clear the barrier error each step
        tgt_img = self._layout()["tgt_img"]                                        # dynamic per-N target-image base
        t = _c(); self.dev.wr(0, X_CAM, [_fb(x) for x in cam16])
        if view_idx is not None and getattr(self, "_views_resident", False):
            self.dev.wr(0, IMG_BASE_A, [TGT_BANK + view_idx * self._img_stride])   # resident: 1-word write
        else:
            self.dev.wr(0, tgt_img, np.asarray(tgt_flat, np.float32).view(np.uint32)); self.dev.wr(0, IMG_BASE_A, [tgt_img])
        T["gt_up"] = _c() - t
        t = _c(); self._het(2, extra=[self.N, step]); T["proj"] = _c() - t     # project+whiten all params (multi-hart)
        if self.on_device_orch:
            # FULLY ON-DEVICE: x280 bins (cmd11 body) + autonomously loops over occupied tiles dispatching batches
            # to the W workers. Host issues ONE doorbell and reads a scalar occ count. No PUB readback, no host bin,
            # no per-batch cmd9 relay.
            t = _c(); tele = self._het(10, timeout=300.0); occ_count = int(tele[4]); ow = _c() - t; T["orch10"] = ow
            if self.diag: self.diag_log.append(("post-cmd10", step, round(ow * 1e3), self.hart_diag()))
            # W4 profile: split orch10 into the x280's device-cycle phases (bin / produce / Tensix render-wait /
            # consume, hart-0 kilocycles). render-wait dominant -> widen the worker grid; produce+consume+bin
            # dominant -> the x280 hub orchestration is the wall. Apportion the wall by the cycle ratio (-> ms in
            # step_log); raw kcyc + batch/worker counts in self.last_profile.
            bin_kc, prod_kc, wait_kc, cons_kc = (int(tele[i]) & 0xFFFFFFFF for i in (8, 9, 10, 11))
            nbatch = int(tele[12]) & 0xFFFFFFFF
            tot = bin_kc + prod_kc + wait_kc + cons_kc
            if tot > 0:
                T["o_bin"] = ow * bin_kc / tot; T["o_prod"] = ow * prod_kc / tot
                T["o_rend"] = ow * wait_kc / tot; T["o_cons"] = ow * cons_kc / tot
            ncap = int(self.dev.rd(0, BINCAP_N_A)) & 0xFFFFFFFF   # # degenerate splats the bin clamped this step
            nnan = int(self.dev.rd(0, NAN_N_A)) & 0xFFFFFFFF      # # NON-FINITE splats the bin skipped (NaN/inf guard)
            self.last_profile = dict(bin_kc=bin_kc, prod_kc=prod_kc, rend_kc=wait_kc, cons_kc=cons_kc,
                                     batches=nbatch, workers=self.W, occ=occ_count, ncap=ncap, nnan=nnan)
            if os.environ.get("TT_BM_PROFILE") == "1":
                print(f"[prof] orch10 {ow*1e3:5.0f}ms = bin {T.get('o_bin',0)*1e3:4.0f} + produce "
                      f"{T.get('o_prod',0)*1e3:4.0f} + render-wait {T.get('o_rend',0)*1e3:5.0f} + consume "
                      f"{T.get('o_cons',0)*1e3:4.0f}  |  {nbatch} batches x {self.W} workers, {occ_count} occ tiles, "
                      f"{ncap} capped, {nnan} non-finite (cap={self.bincap})", flush=True)
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
        if self.diag: self.diag_log.append(("post-adam", step, round(T["adam"] * 1e3), self.hart_diag()))
        self.last_timing = T
        return loss, occ_count
