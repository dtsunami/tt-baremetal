"""
DeviceManager — the single owner of the tt-exalens device for the web backend.

The chip must never see concurrent PCIe access, and tt-exalens calls are blocking.
So ALL device work runs on one dedicated worker thread (a 1-slot executor); the
async FastAPI endpoints dispatch onto it and therefore serialize automatically.
This also preserves the hang-hazard gating: the Poller only ever touches SAFE_KINDS.

State machine (self.mode):
  polling   - default; sample() every 1/hz, broadcast a Frame to WS clients
  injecting - host injection in flight (still polling between fires)
  busy      - a tt-metal subprocess owns the device; polling PAUSED until it exits
  error     - a HangError tripped; reset_needed=True until `tt-smi -r 0`
"""
import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor

from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_words_from_device

from .. import noc_counters as nc
from .. import geometry as G
from ..floorplan import build, KIND_RGB
from ..poller import Poller, SAFE_KINDS
from ..patterns import BUILDERS
from ..l2cpu import (L2cpu, regmap, Hang as L2Hang, HARTS, SPIN_ADDR,
                     HART_STATUS, TRIGGER, RESET_VEC, RNMI_TRAP, RNMI_EXC,
                     TELE_ADDR, TELE_SLOTS, TELE_STRIDE)
from . import l2lab
from . import labkit

CARD_PATH = os.path.expanduser("~/blackhole/uarch/" + G.CARD_IMAGE)


def _key(xy):
    return f"{xy[0]},{xy[1]}"


class DeviceManager:
    def __init__(self, hz=2.0):
        self.hz = hz
        self._exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tt-device")
        self._loop = None
        self.ctx = self.fp = self.poller = self.injector = None
        self._by_noc0 = {}
        self.mode = "init"
        self.reset_needed = False
        self.last_error = None
        self._paused = False
        self._last_frame = None
        self.noc_chan = labkit.Broadcaster(seed=lambda: self._last_frame)  # /ws/telemetry
        self._cells = {}             # noc0 (x,y) -> Tile, for pattern building
        self._stream = None          # active streaming inject spec, or None
        self._stream_src = None
        self._last_inject = None
        self._cum_bytes = 0
        self._kernel_running = None  # gtest name while a kernel job is in flight
        self._last_kernel = None     # last kernel job result (persisted for re-fetch)
        self._llk_test = None        # LLK "test all kernels on all cores" job state (progress + summary)
        self._build_running = None   # ninja target while a build job is in flight
        self._last_build = None      # last build job result (persisted for re-fetch)
        self._compute_running = None # compute example name while a tlab run is in flight
        self._last_compute = None    # last tlab compute run result (per-engine zones)
        self._tlab_deployed = {}     # Tensix core "x,y" -> {kernel, math_occ} last run there
        # ---- L2CPU cockpit (shares this single device owner) ----
        self._l2 = None              # lazy L2cpu controller bound to OUR ctx
        self.l2_chan = labkit.Broadcaster()  # /ws/l2cpu (no seed — client steers tile/hz)
        self._l2_active = 0          # tile the telemetry stream is focused on
        self._l2_hz = 5.0            # L2 telemetry sample rate (independent of NoC poll)
        self._l2_released = set()    # tiles known released (refreshed on probe/bringup)
        self._l2_busy = None         # label while a bringup job is in flight
        self._last_bringup = None    # last bringup result (persisted for re-fetch)
        self._l2_deployed = {}       # (tile,hart) -> {name,lang,addr,seized,words} last deploy
        self._tloop = None           # active Tensix re-go loop {x,y,hz,n,task} or None
        # ---- resident bootloader cockpit (hot-swap code overlays over exalens) ----
        self.bl_chan = labkit.Broadcaster()   # /ws/bootloader (client steers focused core/hz)
        self._bl_active = None       # focused (x,y) the telemetry stream reads, or None
        self._bl_hz = 4.0            # bootloader telemetry sample rate
        self._bl_loaded = {}         # (x,y) -> {slot: {"overlay","hash","bytes"}} last staged
        self._bl_launcher = None     # Popen of the metal bootloader launcher (deploys + parks)

    # ---- lifecycle ---------------------------------------------------------
    async def start(self):
        self._loop = asyncio.get_running_loop()
        await self._run(self._init_device)
        self.mode = "polling"
        asyncio.create_task(self._poll_loop())
        asyncio.create_task(self._l2_tele_loop())
        asyncio.create_task(self._bl_tele_loop())

    async def _run(self, fn, *a):
        """Run a blocking device call on the single worker thread."""
        return await self._loop.run_in_executor(self._exec, lambda: fn(*a))

    def _init_device(self):
        from ..inject import Injector
        self.ctx = init_ttexalens()
        self.fp = build(self.ctx)
        self._by_noc0 = {t.noc0: t for t in self.fp.placed}
        self._cells = {t.noc0: t for t in self.fp.placed}
        self.poller = Poller(self.fp, self.ctx)
        self.injector = Injector(self.fp, self.ctx)
        self.poller.sample()           # prime counters

    # ---- poll loop + broadcast --------------------------------------------
    async def _poll_loop(self):
        period = 1.0 / self.hz
        while True:
            if not self._paused and self.mode in ("polling", "injecting"):
                if self._stream:                              # sustain traffic between samples
                    try:
                        await self._run(self._stream_fire)
                    except Exception as e:                    # pragma: no cover
                        self.last_error = str(e)
                try:
                    frame = await self._run(self._sample_and_frame)
                    self._broadcast(frame)
                except Exception as e:                        # pragma: no cover
                    self.last_error = str(e)
            await asyncio.sleep(period)

    def _stream_fire(self):
        s = self._stream
        if s:
            self._build_and_fire(s["src"], s["pattern"], s["length"], s["fires"])

    def _sample_and_frame(self):
        self.poller.sample()
        return self._frame()

    def _frame(self):
        tiles = {}
        for t in self.fp.placed:
            if t.kind not in SAFE_KINDS:
                continue
            b0 = self.poller.bw.get((t.key, 0), {})
            b1 = self.poller.bw.get((t.key, 1), {})
            if not b0 and not b1:
                continue
            tiles[_key(t.noc0)] = {
                "noc0": nc.metric_scalar(b0, "total"),
                "noc1": nc.metric_scalar(b1, "total"),
                "b0": b0, "b1": b1,
            }
        dram = {}
        for c, ts in self.fp.dram_ctrl.items():
            r = w = 0.0
            for t in ts:
                for noc in (0, 1):
                    b = self.poller.bw.get((t.key, noc), {})
                    r += b.get("tx_slave", 0.0)   # SLV reads served out of DRAM
                    w += b.get("rx_slave", 0.0)   # SLV writes landed into DRAM
            dram[str(c)] = {"r": r, "w": w}
        return {"ts": round(time.monotonic(), 3), "mode": self.mode,
                "reset_needed": self.reset_needed, "tiles": tiles, "dram": dram,
                "inject": {"streaming": bool(self._stream),
                           "src": list(self._stream_src) if self._stream_src else None}}

    def _broadcast(self, frame):
        self._last_frame = frame
        self.noc_chan.broadcast(frame)

    def last_frame(self):
        """The most recent sampled telemetry frame — for HTTP polling clients (the poll loop
        keeps sampling the device at `hz` regardless of how clients read it)."""
        return self._last_frame

    async def subscribe(self):
        return await self.noc_chan.subscribe()

    def unsubscribe(self, q):
        self.noc_chan.unsubscribe(q)

    # ---- injection (host-driven traffic; runs on the same single owner) ----
    async def inject(self, src, pattern, length, fires, stream):
        return await self._run(self._inject, src, pattern, length, fires, stream)

    def _inject(self, src, pattern, length, fires, stream):
        st = self._build_and_fire(src, pattern, length, fires)
        if st.get("ok"):
            self._stream = {"src": list(src), "pattern": pattern, "length": length, "fires": fires} if stream else None
            self._stream_src = tuple(src)
            if stream:
                self.mode = "injecting"
        return st

    def _build_and_fire(self, src, pattern, length, fires):
        from ..inject import HangError
        cell = self._cells.get(tuple(src))
        if cell is None or cell.kind != "tensix":
            return {"ok": False, "error": f"source {list(src)} is not a tensix tile"}
        if pattern == "gddr6_write":
            pairs = [(cell, t) for t in self.injector.dram_access_tiles().values()]
        else:
            builder = BUILDERS.get(pattern)
            if builder is None:
                return {"ok": False, "error": f"unknown pattern '{pattern}'"}
            pairs = builder(cell, self._cells)
        if not pairs:
            return {"ok": False, "error": "pattern produced no valid (src,dst) pairs from this source"}
        try:
            db0 = self.injector.read_dram_counters()
            foot, total, secs = self.injector.run_dual(pairs, length=length, fires=fires)
            db1 = self.injector.read_dram_counters()
        except HangError as e:
            self.reset_needed = True
            self.mode = "error"
            self._stream = None
            self.last_error = f"NoC hang during inject: {e}"
            return {"ok": False, "error": self.last_error, "reset_needed": True}
        M = 0xFFFFFFFF
        dram = {str(c): {"0": (db1[c][0] - db0[c][0]) & M, "1": (db1[c][1] - db0[c][1]) & M} for c in db0}
        self._cum_bytes += total
        self._last_inject = {
            "src": list(src), "pattern": pattern,
            "pairs": [[list(s.noc0), list(d.noc0)] for s, d in pairs],
            "foot": {str(n): {_key(k): v for k, v in foot[n].items()} for n in (0, 1)},
            "moved_bytes": total, "secs": round(secs, 4),
            "bw": (total / secs) if secs else 0, "dram": dram,
        }
        return {"ok": True, **self._last_inject}

    async def inject_stop(self):
        return await self._run(self._inject_stop)

    def _inject_stop(self):
        self._stream = None
        self._stream_src = None
        if self.mode == "injecting":
            self.mode = "polling"
        return {"ok": True}

    # ---- tt-metal kernels (subprocess owns the device; polling pauses) -----
    async def kernels(self):
        from .. import metal
        return await self._run(lambda: {"available": metal.available(),
                                        "tests": metal.list_tests()})

    async def run_kernel(self, name, timeout=900, dprint_cores=None):
        """Start a kernel run as an async job (first-run JIT compiles can take many
        minutes — far longer than any sane HTTP timeout). Returns immediately;
        poll kernel_last() for the result."""
        if self.mode == "busy":
            return {"ok": False, "error": "a kernel is already running"}
        self._stream = None                      # stop any injection stream
        self.mode = "busy"
        self._paused = True                      # nothing touches PCIe while tt-metal owns it
        self._kernel_running = name
        self._broadcast_mode()
        asyncio.create_task(self._kernel_job(name, timeout, dprint_cores))
        return {"ok": True, "started": name}

    async def _kernel_job(self, name, timeout, dprint_cores=None):
        t0 = time.monotonic()
        try:
            result = await self._run(self._run_kernel_blocking, name, timeout, dprint_cores)
            result["secs"] = round(time.monotonic() - t0, 1)
            self._last_kernel = result
        except Exception as e:
            self.last_error = f"kernel run failed: {e}"
            self._last_kernel = {"ok": False, "name": name, "error": self.last_error}
        finally:
            self._kernel_running = None
            self._paused = False
            if self.mode == "busy":
                self.mode = "polling"
            self._broadcast_mode()

    def kernel_last(self):
        return {"running": self._kernel_running, "result": self._last_kernel}

    def _run_kernel_blocking(self, name, timeout, dprint_cores=None):
        from .. import metal
        passed, out = metal.run_test(name, timeout=timeout, dprint_cores=dprint_cores)
        # tt-metal reset the device on init: revalidate our context (reinit once if
        # stale) and read the footprint — counters were zeroed, so absolutes = kernel.
        try:
            foot = metal.read_footprint_per_noc(self.ctx, self.fp)
        except Exception:
            self._init_device()
            foot = metal.read_footprint_per_noc(self.ctx, self.fp)
        agg = metal.aggregate_bw()
        # re-baseline the poller (counters were zeroed; first delta would be garbage)
        self.poller.prev, self.poller.bw, self.poller.last_t = {}, {}, None
        self.poller.sample()
        tail = "\n".join(out.splitlines()[-12:])
        dprint = metal.extract_dprint(out)
        return {"ok": True, "passed": passed, "name": name,
                "foot": {str(n): {_key(k): v[n] for k, v in foot.items() if v[n]} for n in (0, 1)},
                "agg": agg and {k: v for k, v in agg.items() if k != "footprint"},
                "dprint": dprint[-400:], "log_tail": tail}

    # ---- tlab: Tensix Compute Lab (run a compute example, read per-engine zones) -----
    # Reuses run_kernel's busy-mode + the tt-metal device-reset recovery. tt-metal owns +
    # RESETS the device for the run (so the L2CPU/x280 harts go back to reset).
    async def tlab_examples(self):
        from .. import metal
        return await self._run(lambda: {"available": metal.available(),
                                        "examples": metal.example_names()})

    async def tlab_run(self, name, timeout=900):
        if self.mode == "busy":
            return {"ok": False, "error": "a kernel is already running"}
        self._stream = None
        self.mode = "busy"
        self._paused = True
        self._compute_running = name
        self._broadcast_mode()
        asyncio.create_task(self._tlab_job(name, timeout))
        return {"ok": True, "started": name}

    async def _tlab_job(self, name, timeout):
        t0 = time.monotonic()
        try:
            result = await self._run(self._tlab_run_blocking, name, timeout)
            result["secs"] = round(time.monotonic() - t0, 1)
            self._last_compute = result
        except Exception as e:
            self.last_error = f"compute run failed: {e}"
            self._last_compute = {"ok": False, "name": name, "error": self.last_error}
        finally:
            self._compute_running = None
            self._paused = False
            if self.mode == "busy":
                self.mode = "polling"
            self._broadcast_mode()

    def _tlab_run_blocking(self, name, timeout):
        from .. import metal
        from . import tlab_build
        passed, out = metal.run_example(name, timeout=timeout)
        short = name.replace("metal_example_", "")
        try:                                         # capture the JIT compile recipe for standalone builds
            tlab_build.save_recipe(short, out)
        except Exception:
            pass
        compute = metal.aggregate_compute()          # per-Tensix-core per-engine cycles
        if compute:                                  # remember which kernel ran on which Tensix core
            for core, c in compute["cores"].items():
                self._tlab_deployed[core] = {"kernel": short, "math_occ": c["math_occ"]}
        # tt-metal reset the device on init: revalidate ctx (reinit if stale) + rebaseline poller
        try:
            metal.read_footprint_per_noc(self.ctx, self.fp)
        except Exception:
            self._init_device()
        self.poller.prev, self.poller.bw, self.poller.last_t = {}, {}, None
        self.poller.sample()
        return {"ok": True, "passed": passed, "name": name, "compute": compute,
                "dprint": metal.extract_dprint(out)[-40:],
                "log_tail": "\n".join(out.splitlines()[-12:])}

    def tlab_last(self):
        return {"running": self._compute_running, "result": self._last_compute}

    def tlab_status(self):
        """Which compute kernel last ran on which Tensix core (+ MATH occupancy)."""
        return {"deployed": self._tlab_deployed, "running": self._compute_running}

    async def tlab_files(self, example):
        from . import tlab
        return await asyncio.to_thread(tlab.files, example)

    async def tlab_tree(self):
        from . import tlab
        return await asyncio.to_thread(tlab.tree)

    async def tlab_params(self, key):
        from . import tlab
        return await asyncio.to_thread(tlab.params, key)

    async def tlab_config_get(self, key):
        from . import tlab
        return await asyncio.to_thread(tlab.config_get, key)

    async def tlab_config_put(self, key, text):
        from . import tlab
        return await asyncio.to_thread(tlab.config_put, key, text)

    async def tlab_merge_params(self, key):
        from . import tlab
        return await asyncio.to_thread(tlab.merge_params, key)

    async def tlab_restore(self):
        from . import tlab
        return await asyncio.to_thread(tlab.restore)

    async def tlab_read(self, path):
        from . import tlab
        return await asyncio.to_thread(tlab.read_file, path)

    async def tlab_write(self, path, content):
        from . import tlab
        return await asyncio.to_thread(tlab.write_file, path, content)

    async def tlab_revert(self, path):
        from . import tlab
        return await asyncio.to_thread(tlab.revert_file, path)

    async def tlab_copy(self, src, name):
        from . import tlab
        return await asyncio.to_thread(tlab.copy_file, src, name)

    async def tlab_disasm(self):
        from . import tlab_disasm
        return await asyncio.to_thread(tlab_disasm.fetch_last)

    async def tlab_build_log(self):
        from . import tlab_disasm
        return await asyncio.to_thread(tlab_disasm.fetch_build_log)

    # ---- standalone (no-device) build of a kernel from a bhtop-extracted copy ----
    async def tlab_extract(self, example):
        from . import tlab_build
        return await asyncio.to_thread(tlab_build.extract, example)

    async def tlab_build_standalone(self, example):
        from . import tlab_build
        return await asyncio.to_thread(tlab_build.build, example)

    async def tlab_recipe(self, example):
        from . import tlab_build
        rec = await asyncio.to_thread(tlab_build.load_recipe, example)
        return {"have": bool(rec), "units": len(rec["units"]) if rec else 0}

    async def tlab_rebuild(self):
        from . import tlab_disasm
        return await asyncio.to_thread(tlab_disasm.force_rebuild)

    async def running(self):
        """Which tt-metal kernels are live, keyed by JIT build hash (web/inspector.py): the
        device tree badges a source 'running' (basename in by_source) / 'stale' (content no
        longer matches the running build). Host-side read of the Inspector dump."""
        from . import inspector
        return await asyncio.to_thread(inspector.read)

    async def tlab_docs_index(self):
        from . import tlab_docs
        return await asyncio.to_thread(tlab_docs.docs_index)

    async def tlab_doc(self, doc_id):
        from . import tlab_docs
        return await asyncio.to_thread(tlab_docs.doc, doc_id)

    def _broadcast_mode(self):
        f = dict(self._last_frame or {"ts": 0, "tiles": {}, "dram": {},
                                      "inject": {"streaming": False, "src": None}})
        f["mode"] = self.mode
        f["reset_needed"] = self.reset_needed
        self._broadcast(f)

    # ---- kernel lab: edit / build / docs (no device access) ---------------
    # File + doc ops are pure filesystem; builds are CPU subprocesses. None of
    # these touch PCIe, so they run on the default thread pool (NOT the single
    # device executor) and live polling keeps running straight through a build.
    async def lab_projects(self):
        from . import lab
        return await asyncio.to_thread(lab.projects)

    async def lab_files(self, project):
        from . import lab
        return await asyncio.to_thread(lab.files, project)

    async def lab_tree(self):
        from . import lab
        return await asyncio.to_thread(lab.tree)

    async def lab_params(self, key):
        from . import lab
        return await asyncio.to_thread(lab.params, key)

    async def lab_config_get(self, key):
        from . import lab
        return await asyncio.to_thread(lab.config_get, key)

    async def lab_config_put(self, key, text):
        from . import lab
        return await asyncio.to_thread(lab.config_put, key, text)

    async def lab_merge_params(self, key):
        from . import lab
        return await asyncio.to_thread(lab.merge_params, key)

    async def lab_restore(self):
        from . import lab
        return await asyncio.to_thread(lab.restore)

    async def lab_read(self, path):
        from . import lab
        return await asyncio.to_thread(lab.read_file, path)

    async def lab_write(self, path, content):
        from . import lab
        return await asyncio.to_thread(lab.write_file, path, content)

    async def lab_revert(self, path):
        from . import lab
        return await asyncio.to_thread(lab.revert_file, path)

    async def lab_copy(self, src, name):
        from . import lab
        return await asyncio.to_thread(lab.copy_file, src, name)

    async def lab_docs_index(self):
        from . import lab
        return await asyncio.to_thread(lab.docs_index)

    async def lab_doc(self, doc_id):
        from . import lab
        return await asyncio.to_thread(lab.doc, doc_id)

    async def lab_build(self, target="unit_tests_data_movement"):
        if self._build_running:
            return {"ok": False, "error": "a build is already running"}
        self._build_running = target
        asyncio.create_task(self._build_job(target))
        return {"ok": True, "started": target}

    async def _build_job(self, target):
        from . import lab
        t0 = time.monotonic()
        try:
            res = await asyncio.to_thread(lab.build, target)
            res["secs"] = round(time.monotonic() - t0, 1)
            self._last_build = res
        except Exception as e:                       # pragma: no cover
            self._last_build = {"ok": False, "target": target, "error": str(e)}
        finally:
            self._build_running = None

    def build_last(self):
        return {"running": self._build_running, "result": self._last_build}

    # ---- L2CPU cockpit: develop / deploy / observe ------------------------
    # Device ops run on the SAME single worker thread (via _run) as the NoC poll,
    # so they serialize automatically — the chip never sees concurrent access.
    # Compile is pure CPU and stays in l2lab on the default pool.
    def _l2_get(self):
        if self._l2 is None:
            self._l2 = L2cpu(ctx=self.ctx)        # reuse our ctx — no 2nd device owner
        return self._l2

    async def l2_tiles(self):
        return await self._run(self._l2_tiles)

    def _l2_tiles(self):
        l2 = self._l2_get()
        tiles = []
        for i in regmap.TILES:
            rs = l2.reset_state(i)
            tiles.append({"tile": i, "coord": list(regmap.TILES[i][0]),
                          "bit": regmap.TILES[i][1], "released": rs["released"],
                          "wedged": rs["wedged"]})
        self._l2_released = {t["tile"] for t in tiles if t["released"]}
        deployed = {f"{t},{h}": v for (t, h), v in self._l2_deployed.items()}
        return {"tiles": tiles, "harts": regmap.HARTS, "busy": self._l2_busy,
                "have_rust": l2lab.have_rust(), "deployed": deployed}

    async def l2_regs(self, tile):
        return await self._run(self._l2_regs, tile)

    def _l2_regs(self, tile):
        l2 = self._l2_get()
        rs = l2.reset_state(tile)
        out = {"tile": tile, "coord": list(regmap.TILES[tile][0]),
               "released": rs["released"], "wedged": rs["wedged"], "l2cpu_reset": rs["raw"]}
        if rs["released"]:
            out["hart_status"] = l2.rd(tile, HART_STATUS) & 0xFFFF
            out["trigger"] = l2.rd(tile, TRIGGER)
            harts = []
            for h in range(regmap.HARTS):
                rv = l2.rdn(tile, RESET_VEC + h * 8, 2)
                trap = l2.rdn(tile, RNMI_TRAP + h * 16, 2)
                exc = l2.rdn(tile, RNMI_EXC + h * 16, 2)
                j = lambda w: (w[1] << 32) | w[0]
                harts.append({"hart": h, "reset_vec": j(rv),
                              "rnmi_trap": j(trap), "rnmi_exc": j(exc)})
            out["harts"] = harts
        return out

    async def l2_arch(self, tile, hart):
        return await self._run(lambda: self._l2_get().arch_state(tile, hart))

    async def l2_bringup(self, tile):
        if self.reset_needed:
            return {"ok": False, "error": "NoC hang pending — run tt-smi -r 0 and restart"}
        if self.mode == "busy":
            return {"ok": False, "error": "a tt-metal kernel owns the device"}
        if self._l2_busy:
            return {"ok": False, "error": f"{self._l2_busy} already in progress"}
        self._l2_busy = f"bringup tile {tile}"
        asyncio.create_task(self._l2_bringup_job(tile))
        return {"ok": True, "started": tile}

    async def _l2_bringup_job(self, tile):
        try:
            res = await self._run(lambda: self._l2_get().bringup(tile))
            self._last_bringup = {"ok": True, "tile": tile, **res}
        except Exception as e:
            self._last_bringup = {"ok": False, "tile": tile, "error": str(e)}
        finally:
            self._l2_busy = None
            try:
                await self._run(self._l2_tiles)        # refresh released set
            except Exception:
                pass

    def l2_bringup_last(self):
        return {"running": self._l2_busy, "result": self._last_bringup}

    async def l2_deploy(self, tile, hart, content, lang, addr, name="", defines=None):
        """Compile (off the device thread) then load+redirect (on it). One call =
        the whole develop→deploy step. Returns the verified seize result and records
        which kernel now runs on this hart (so the cockpit can show it)."""
        if self.reset_needed:
            return {"ok": False, "error": "NoC hang pending — run tt-smi -r 0 and restart"}
        if self.mode == "busy":
            return {"ok": False, "error": "a tt-metal kernel owns the device"}
        if self._l2_busy:
            return {"ok": False, "error": f"{self._l2_busy} in progress"}
        comp = await asyncio.to_thread(l2lab.compile_kernel, content, lang, addr, defines)
        if not comp.get("ok"):
            return {"ok": False, "stage": "compile", **comp}

        def _deploy():
            l2 = self._l2_get()
            l2.wr(tile, TELE_ADDR + hart * TELE_STRIDE, [0] * TELE_SLOTS)  # fresh window
            return l2.load(tile, hart, list(comp["words"]), addr=addr, redirect=True)
        try:
            res = await self._run(_deploy)
        except Exception as e:
            return {"ok": False, "stage": "load", "error": str(e)}
        self._l2_active = tile                          # focus the stream on it
        self._l2_deployed[(tile, hart)] = {
            "name": name or f"{lang} kernel", "lang": lang, "addr": addr,
            "seized": bool(res.get("seized")), "words": len(comp["words"])}
        return {"ok": True, "stage": "load", "tile": tile, "hart": hart, "addr": addr,
                "words": len(comp["words"]), "bytes": comp["bytes"],
                "seized": res.get("seized"), "disasm": comp.get("disasm", "")}

    def _deployed_for(self, tile):
        """{hart: {name,lang,seized}} for the harts we've deployed to on this tile."""
        return {h: v for (t, h), v in self._l2_deployed.items() if t == tile}

    async def l2_deploy_all(self, tile, content, lang, addr, name="", defines=None, harts=None):
        """Compile once, load the SAME kernel onto a GROUPING of harts on a tile (each gets its
        own telemetry window). `harts` is any subset (None = all 4)."""
        if self.reset_needed or self.mode == "busy" or self._l2_busy:
            return {"ok": False, "error": "device not ready (reset/busy/bringup)"}
        sel = [h for h in (harts if harts else range(HARTS)) if 0 <= h < HARTS] or list(range(HARTS))
        comp = await asyncio.to_thread(l2lab.compile_kernel, content, lang, addr, defines)
        if not comp.get("ok"):
            return {"ok": False, "stage": "compile", **comp}

        def _all():
            l2 = self._l2_get()
            out = []
            for h in sel:
                l2.wr(tile, TELE_ADDR + h * TELE_STRIDE, [0] * TELE_SLOTS)   # fresh window
                try:
                    r = l2.load(tile, h, list(comp["words"]), addr=addr, redirect=True)
                    out.append({"hart": h, "seized": r.get("seized")})
                except Exception as e:
                    out.append({"hart": h, "error": str(e)})
            return out
        res = await self._run(_all)
        for o in res:
            if "error" not in o:
                self._l2_deployed[(tile, o["hart"])] = {
                    "name": name or f"{lang} kernel", "lang": lang, "addr": addr,
                    "seized": bool(o.get("seized")), "words": len(comp["words"])}
        self._l2_active = tile
        return {"ok": True, "deployed_all": res, "words": len(comp["words"]),
                "bytes": comp["bytes"], "disasm": comp.get("disasm", "")}

    async def l2_park_all(self, tile):
        """Park all 4 harts of a tile: redirect each to the bringup spin (no code needed —
        SPIN_ADDR already holds `j .`). Stops whatever they were running."""
        if self.reset_needed or self.mode == "busy" or self._l2_busy:
            return {"ok": False, "error": "device not ready (reset/busy/bringup)"}

        def _park():
            l2 = self._l2_get()
            out = []
            for h in range(HARTS):
                try:
                    r = l2.redirect(tile, h, SPIN_ADDR)
                    out.append({"hart": h, "seized": r.get("seized")})
                except Exception as e:
                    out.append({"hart": h, "error": str(e)})
            return out
        res = await self._run(_park)
        for h in range(HARTS):
            self._l2_deployed.pop((tile, h), None)        # no longer running user code
        return {"ok": True, "parked": res}

    async def l2_zero_tele(self, tile, hart=None):
        """Zero one hart's telemetry window (or all 4 if hart is None)."""
        def _z():
            l2 = self._l2_get()
            if hart is None:
                l2.wr(tile, TELE_ADDR, [0] * (TELE_SLOTS * HARTS))
            else:
                l2.wr(tile, TELE_ADDR + hart * TELE_STRIDE, [0] * TELE_SLOTS)
            return {"ok": True}
        return await self._run(_z)

    async def l2_poke(self, tile, addr, val):
        return await self._run(lambda: (self._l2_get().poke(tile, addr, val), {"ok": True})[1])

    async def l2_cmd(self, tile, hart, op, arg0=0, arg1=0):
        """Ring hart N's mailbox doorbell (cooperative register/virus update). Returns the seq."""
        return await self._run(
            lambda: {"ok": True, "seq": self._l2_get().command(tile, hart, op, arg0, arg1)})

    async def l2_vec(self, tile, hart, ew=32):
        """Decode v0..v31 + vector CSRs (kernel must have called bh_dump_vec())."""
        return await self._run(lambda: self._l2_get().vec_state(tile, hart, ew=ew))

    # ---- Tensix launch (exalens RTA poke + re-go) — device thread, shared ctx ----
    # Reads/pokes a Tensix worker's launch mailbox in L1 (see bhtop.tensix). Tensix L1 over NoC is
    # the safe surface (not the ARC/PCIe hang hazard). Rides a tt-metal-loaded program: runtime
    # args are pokeable, then go() re-runs without a rebuild.
    RTA_PREVIEW = 8                  # runtime-arg words read per enabled processor for display

    def _tensix_core(self, x, y):
        t = self._by_noc0.get((x, y))
        if t is None:
            raise ValueError(f"no tile at noc0 ({x},{y})")
        if t.kind != "tensix":
            raise ValueError(f"({x},{y}) is a {t.kind} tile, not a Tensix worker")
        return t

    async def tensix_launch(self, x, y, index=None):
        return await self._run(self._tensix_launch, x, y, index)

    def _tensix_launch(self, x, y, index):
        from ..tensix import TensixLauncher, abi
        from . import inspector
        L = TensixLauncher(self._tensix_core(x, y).coord, ctx=self.ctx)
        snap = L.snapshot(index, kernels=inspector.by_watcher_id())
        kc = L.read_launch(snap["active_index"])
        kbyproc = {pr["proc"]: pr.get("kernel") for pr in snap["procs"]}   # kernel identity per proc
        rta = []
        for p in kc["enabled_procs"]:
            try:
                vals = L.read_rta(p, self.RTA_PREVIEW, index=snap["active_index"])
            except Exception:
                vals = []
            name = abi.PROC_NAME.get(p, p)
            rta.append({"proc": name, "proc_id": p, "addr": hex(abi.rta_l1_addr(kc, p)),
                        "values": vals, "kernel": kbyproc.get(name)})
        snap["rta"] = rta
        lp = self.tensix_loop_status()               # is a re-go loop running on THIS core?
        snap["loop"] = lp if (lp["running"] and lp["x"] == x and lp["y"] == y) else {"running": False}
        return snap

    async def tensix_write_rta(self, x, y, proc, values, arg_offset=0, index=None):
        return await self._run(self._tensix_write_rta, x, y, proc, values, arg_offset, index)

    def _tensix_write_rta(self, x, y, proc, values, arg_offset, index):
        from ..tensix import TensixLauncher
        L = TensixLauncher(self._tensix_core(x, y).coord, ctx=self.ctx)
        addr = L.write_rta(int(proc), [int(v) & 0xFFFFFFFF for v in values],
                           index=index, arg_offset=int(arg_offset))
        return {"ok": True, "addr": hex(addr), "wrote": len(values)}

    async def tensix_go(self, x, y, signal=None):
        return await self._run(self._tensix_go, x, y, signal)

    def _tensix_go(self, x, y, signal):
        from ..tensix import TensixLauncher, abi
        L = TensixLauncher(self._tensix_core(x, y).coord, ctx=self.ctx)
        return L.go(abi.RUN_MSG_GO if signal is None else int(signal))

    async def tensix_scan(self):
        return await self._run(self._tensix_scan)

    def _tensix_scan(self):
        """Brief every Tensix worker so the cockpit can show WHICH cores have a resident program
        AND which kernel runs there (joins watcher_kernel_ids with the Inspector). ~5-6 NoC reads
        per core; runs on the device thread."""
        from ..tensix import TensixLauncher
        from . import inspector
        kmap = inspector.by_watcher_id()
        cores = []
        for t in self.fp.placed:
            if t.kind != "tensix":
                continue
            x, y = t.noc0
            try:
                b = TensixLauncher(t.coord, ctx=self.ctx).brief(kernels=kmap)
                cores.append({"x": x, "y": y, **b})
            except Exception as e:               # a core that won't read shouldn't sink the scan
                cores.append({"x": x, "y": y, "resident": False, "error": str(e)})
        return {"cores": cores, "n": len(cores), "inspector": bool(kmap),
                "n_resident": sum(1 for c in cores if c.get("resident"))}

    # ---- Tensix run-loop: re-issue go continuously so the one-shot kernel runs forever ----
    async def tensix_loop(self, x, y, on=True, hz=10, force=False):
        """Start/stop an infinite re-go loop on a core. A tt-metal kernel runs once per go (→DONE);
        this re-issues go at `hz` so it re-runs continuously (watch L1/heat change live). One loop
        at a time; starting a new one replaces it. Refuses dispatch-infra cores unless force."""
        if self._tloop and self._tloop.get("task"):
            self._tloop["task"].cancel()
            self._tloop = None
        if not on:
            return {"running": False}
        self._tensix_core(x, y)                       # validate it's a Tensix worker (raises -> 400)
        if not force:                                 # don't stomp the command-queue dispatch cores
            from . import inspector
            from ..tensix import TensixLauncher
            b = await self._run(lambda: TensixLauncher(self._tensix_core(x, y).coord, ctx=self.ctx)
                                .brief(kernels=inspector.by_watcher_id()))
            if b.get("resident") and not b.get("user_kernel"):
                raise ValueError(f"({x},{y}) runs dispatch infra ({', '.join(b.get('kernel_names') or [])}) "
                                 "— pass force=true to loop it anyway")
        hz = max(1, min(int(hz), 50))
        st = {"x": x, "y": y, "hz": hz, "n": 0, "running": True}
        st["task"] = asyncio.create_task(self._tensix_loop_run(st))
        self._tloop = st
        return {"running": True, "x": x, "y": y, "hz": hz}

    async def _tensix_loop_run(self, st):
        period = 1.0 / st["hz"]
        try:
            while True:
                try:
                    await self._run(self._tensix_go, st["x"], st["y"], None)
                    st["n"] += 1
                except Exception as e:                # keep looping across a transient read/write blip
                    st["error"] = str(e)
                await asyncio.sleep(period)
        except asyncio.CancelledError:
            st["running"] = False

    def tensix_loop_status(self):
        t = self._tloop
        if not t or not t.get("running"):
            return {"running": False}
        return {"running": True, "x": t["x"], "y": t["y"], "hz": t["hz"], "n": t["n"],
                "error": t.get("error")}

    async def tensix_peek(self, x, y, addr, n=8):
        return await self._run(self._tensix_peek, x, y, addr, n)

    def _tensix_peek(self, x, y, addr, n):
        """Raw L1 read window — the liveness primitive: poll an address the kernel writes to."""
        from ..tensix import TensixLauncher
        L = TensixLauncher(self._tensix_core(x, y).coord, ctx=self.ctx)
        words = L.rd(int(addr), max(1, min(int(n), 256)))
        return {"addr": int(addr), "n": len(words), "words": words}

    async def l2_power(self):
        """Board power/current/temperature via ARC telemetry."""
        return await self._run(lambda: self._l2_get().power())

    async def l2_clocks(self):
        """Core (l2cpuclk) vs uncore (axiclk/arcclk) vs Tensix (aiclk) clocks."""
        return await self._run(lambda: self._l2_get().clocks())

    async def l2_freq(self, mhz):
        """Set the L2CPU core PLL to a verified point (raises ValueError otherwise)."""
        return await self._run(lambda: self._l2_get().set_core_freq(mhz))

    # ---- L2 workspace + docs + compile (filesystem/CPU — off the device thread) ----
    async def l2_tree(self):
        return await asyncio.to_thread(l2lab.tree)

    async def l2_params(self, key):
        return await asyncio.to_thread(l2lab.kernel_meta, key)

    async def l2_save_params(self, key, values):
        return await asyncio.to_thread(l2lab.save_params, key, values)

    async def l2_config_get(self, key):
        return await asyncio.to_thread(l2lab.config_get, key)

    async def l2_config_put(self, key, text):
        return await asyncio.to_thread(l2lab.config_put, key, text)

    async def l2_merge_params(self, key):
        return await asyncio.to_thread(l2lab.merge_params, key)

    async def l2_folder_new(self, path):
        return await asyncio.to_thread(l2lab.folder_new, path)

    async def l2_folder_dup(self, src, name):
        return await asyncio.to_thread(l2lab.folder_dup, src, name)

    async def l2_folder_rename(self, src, name):
        return await asyncio.to_thread(l2lab.folder_rename, src, name)

    async def l2_folder_delete(self, path):
        return await asyncio.to_thread(l2lab.folder_delete, path)

    async def l2_regenerate(self):
        return await asyncio.to_thread(l2lab.regenerate)

    async def l2_files(self):
        return await asyncio.to_thread(l2lab.files)

    async def l2_examples(self):
        return await asyncio.to_thread(lambda: {"examples": l2lab.examples(),
                                                "have_rust": l2lab.have_rust()})

    async def l2_read(self, name):
        return await asyncio.to_thread(l2lab.read_file, name)

    async def l2_write(self, name, content):
        return await asyncio.to_thread(l2lab.write_file, name, content)

    async def l2_new(self, name, lang):
        return await asyncio.to_thread(l2lab.new_file, name, lang)

    async def l2_rename(self, src, name):
        from . import l2lab
        return await asyncio.to_thread(l2lab.rename_file, src, name)

    async def l2_copy(self, src, name):
        return await asyncio.to_thread(l2lab.copy_file, src, name)

    async def l2_delete(self, name):
        return await asyncio.to_thread(l2lab.delete_file, name)

    async def l2_compile(self, content, lang, addr, defines=None):
        return await asyncio.to_thread(l2lab.compile_kernel, content, lang, addr, defines)

    async def l2_docs_index(self):
        return await asyncio.to_thread(l2lab.docs_index)

    async def l2_doc(self, doc_id):
        return await asyncio.to_thread(l2lab.doc, doc_id)

    # ---- L2 telemetry stream (its own rate, decoupled from the NoC poll) ----
    async def l2_subscribe(self):
        return await self.l2_chan.subscribe()

    def l2_unsubscribe(self, q):
        self.l2_chan.unsubscribe(q)

    def l2_select(self, tile, hz=None):
        if tile is not None and tile in regmap.TILES:
            self._l2_active = tile
        if hz:
            self._l2_hz = max(1.0, min(10.0, float(hz)))

    async def _l2_tele_loop(self):
        while True:
            await asyncio.sleep(1.0 / max(self._l2_hz, 1.0))
            if len(self.l2_chan) == 0:
                continue
            tile = self._l2_active
            if self.reset_needed or self._paused or self.mode not in ("polling", "injecting"):
                self._l2_broadcast({"tile": tile, "paused": True, "mode": self.mode,
                                    "reset_needed": self.reset_needed})
                continue
            try:
                frame = await self._run(self._l2_tele_frame, tile)
            except Exception as e:                        # pragma: no cover
                frame = {"tile": tile, "error": str(e)}
            self._l2_broadcast(frame)

    def _l2_tele_frame(self, tile):
        l2 = self._l2_get()
        rs = l2.reset_state(tile)
        out = {"tile": tile, "released": rs["released"], "wedged": rs["wedged"],
               "mode": self.mode, "ts": round(time.monotonic(), 3), "hz": self._l2_hz,
               "busy": self._l2_busy, "deployed": self._deployed_for(tile)}
        if rs["released"]:
            # every hart's window in one read -> the UI can show any hart, no collision
            out["tele_by_hart"] = {str(h): v for h, v in l2.telemetry_all(tile).items()}
            out["hart_status"] = l2.rd(tile, HART_STATUS) & 0xFFFF
            out["trigger"] = l2.rd(tile, TRIGGER)
        return out

    def _l2_broadcast(self, frame):
        self.l2_chan.broadcast(frame)

    # ---- resident bootloader cockpit --------------------------------------
    # Rides the bootloader resident on each worker (deployed by the metal launcher); all I/O is
    # L1-over-NoC via Bootloader/TensixLauncher on the single device thread. Overlays + their
    # telemetry/param schemas + hashes come from tensix.overlays (the registry).
    def _bl(self, x, y):
        from ..tensix import bootloader as blmod
        from ..tensix import TensixLauncher
        return blmod.Bootloader(TensixLauncher(self._tensix_core(x, y).coord, ctx=self.ctx))

    def tensix_bl_overlays(self):
        """The overlay registry (metadata + telemetry/param schemas + hashes). No device touch."""
        from ..tensix import overlays
        return {"overlays": overlays.manifest(), "template": overlays.TEMPLATE}

    async def tensix_bl_compile(self, name, source):
        """Compile a user overlay source -> registered .bin (CPU work, off the device thread)."""
        from ..tensix import overlays
        return await asyncio.to_thread(overlays.compile, name, source)

    def tensix_bl_source(self, name):
        """Overlay .c source for the editor (file read; no device touch)."""
        from ..tensix import overlays
        return {"name": name, "source": overlays.source(name), "lang": "c"}

    # ---- deploy the resident bootloader (run the metal launcher subprocess) ----
    def _bl_launcher_path(self):
        import os
        home = os.path.expanduser(os.environ.get("TT_METAL_HOME") or "~/tt-metal")
        return home, os.path.join(home, "build_Release/programming_examples/contributed/bootloader")

    def _bl_kill_launchers(self):
        """Stop our launcher + any stray one. SIGINT first so it close()s cleanly (resets cores,
        frees the device) before a re-launch; then force."""
        import signal
        import subprocess
        import time
        if self._bl_launcher and self._bl_launcher.poll() is None:
            try:
                self._bl_launcher.send_signal(signal.SIGINT); self._bl_launcher.wait(timeout=8)
            except Exception:
                try: self._bl_launcher.kill()
                except Exception: pass
        self._bl_launcher = None
        subprocess.run(["pkill", "-INT", "-f", "programming_examples/contributed/bootloader"],
                       capture_output=True)
        time.sleep(3)
        subprocess.run(["pkill", "-9", "-f", "programming_examples/contributed/bootloader"],
                       capture_output=True)

    def bl_launch_status(self):
        ours = self._bl_launcher is not None and self._bl_launcher.poll() is None
        import subprocess
        stray = subprocess.run(["pgrep", "-f", "programming_examples/contributed/bootloader$"],
                               capture_output=True).returncode == 0
        return {"running": ours or stray, "owned": ours}

    async def tensix_bl_launch(self, grid="2x2"):
        return await asyncio.to_thread(self._bl_launch, grid)

    def _bl_launch(self, grid):
        """Kill any existing launcher, then spawn a fresh one that JIT-builds the bootloader and
        multicasts it to a `grid` (WxH or 'all') block of workers, then parks holding the device."""
        import os
        import re
        import subprocess
        if not re.fullmatch(r"all|\d+x\d+", grid or ""):
            return {"ok": False, "error": "grid must be WxH (e.g. 2x2) or 'all'"}
        home, binpath = self._bl_launcher_path()
        if not os.path.exists(binpath):
            return {"ok": False, "error": f"launcher not built: {binpath} — build the 'bootloader' target"}
        self._bl_kill_launchers()
        # metal determines its root from CWD or TT_METAL_RUNTIME_ROOT (TT_METAL_HOME alone isn't
        # enough in this build) — so run FROM the repo root and set the runtime root explicitly.
        env = dict(os.environ, TT_METAL_HOME=home, TT_METAL_RUNTIME_ROOT=home,
                   TT_METAL_SLOW_DISPATCH_MODE="1", BL_GRID=grid)
        env.pop("TT_METAL_WATCHER", None)
        logp = "/tmp/bl_web_launcher.log"
        log = open(logp, "w")
        self._bl_launcher = subprocess.Popen(
            [binpath], env=env, cwd=home, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return {"ok": True, "pid": self._bl_launcher.pid, "grid": grid, "log": logp,
                "note": "deploying — JIT build + multicast takes a few seconds; then Scan"}

    async def tensix_bl_launch_stop(self):
        return await asyncio.to_thread(self._bl_launch_stop)

    def _bl_launch_stop(self):
        self._bl_kill_launchers()
        return {"ok": True, "stopped": True}

    async def tensix_bl_save_source(self, name, source):
        from ..tensix import overlays
        return await asyncio.to_thread(overlays.save_source, name, source)

    async def tensix_bl_status(self, x, y):
        return await self._run(self._tensix_bl_status, x, y)

    def _tensix_bl_status(self, x, y):
        from ..tensix import overlays, bootloader as blmod
        b = self._bl(x, y)
        s = b.status()
        loaded = self._bl_loaded.get((x, y), {})
        # decode telemetry against whatever overlay is loaded in slot A (the exec target)
        ov = (loaded.get("A") or {}).get("overlay")
        telem = b.L.rd(blmod.TELEM_BASE, 8)
        s["loaded"] = loaded
        s["telemetry"] = overlays.decode_telemetry(ov, telem)
        s["telem_raw"] = list(telem)
        return s

    async def tensix_bl_scan(self):
        return await self._run(self._tensix_bl_scan)

    def _tensix_bl_scan(self):
        """Classify every Tensix worker for the unified cockpit grid. Residency is AUTHORITATIVE
        from the tt-metal launch mailbox (`enables`, set by metal at launch) + Inspector kernel
        identity — NOT the bootloader STATUS byte (stale L1 survives a soft-reset and lies). Each
        core gets kind ∈ {bootloader, ttmetal, idle, err}: 'bootloader' = a resident kernel named
        'bootloader' (our overlay host, drives the bl panel); 'ttmetal' = any other resident metal
        kernel (dispatch infra or a user program → the tt-metal panel); 'idle' = nothing resident."""
        from ..tensix import TensixLauncher, bootloader as blmod
        from . import inspector
        kmap = inspector.by_watcher_id()
        cores = []
        for t in self.fp.placed:
            if t.kind != "tensix":
                continue
            x, y = t.noc0
            try:
                L = TensixLauncher(t.coord, ctx=self.ctx)
                b = L.brief(kernels=kmap)
            except Exception as e:
                cores.append({"x": x, "y": y, "kind": "err", "resident": False, "error": str(e)})
                continue
            names = b.get("kernel_names") or []
            is_bl = bool(b.get("resident")) and ("bootloader" in names)
            kind = "bootloader" if is_bl else ("ttmetal" if b.get("resident") else "idle")
            e = {"x": x, "y": y, "kind": kind, "resident": bool(b.get("resident")),
                 "kernel_names": names, "user_kernel": b.get("user_kernel"),
                 "host_id": b.get("host_id"), "signal": b.get("signal")}
            if is_bl:
                try:
                    s = blmod.Bootloader(L).status()
                    loaded = self._bl_loaded.get((x, y), {})
                    e.update(status=s["status_name"], heartbeat=s["heartbeat"],
                             loaded=(loaded.get("A") or {}).get("overlay"),
                             hash=(loaded.get("A") or {}).get("hash"))
                except Exception:
                    pass
            cores.append(e)
        return {"cores": cores, "n": len(cores), "inspector": bool(kmap),
                "n_bootloader": sum(1 for c in cores if c["kind"] == "bootloader"),
                "n_ttmetal": sum(1 for c in cores if c["kind"] == "ttmetal")}

    async def tensix_bl_param(self, x, y, index, value):
        return await self._run(self._tensix_bl_param, x, y, index, value)

    def _tensix_bl_param(self, x, y, index, value):
        addr = self._bl(x, y).set_param(int(index), int(value) & 0xFFFFFFFF)
        return {"ok": True, "addr": hex(addr), "index": int(index), "value": int(value) & 0xFFFFFFFF}

    async def tensix_bl_stage(self, x, y, overlay, slot="A"):
        return await self._run(self._tensix_bl_stage, x, y, overlay, slot)

    def _tensix_bl_stage(self, x, y, overlay, slot):
        from ..tensix import overlays
        data = overlays.bin_bytes(overlay)               # raises if unknown/unbuilt
        r = self._bl(x, y).stage(data, slot=slot)
        self._bl_loaded.setdefault((x, y), {})[slot.upper()] = {
            "overlay": overlay, "hash": overlays.bin_hash(overlay), "bytes": len(data)}
        return {**r, "overlay": overlay, "hash": overlays.bin_hash(overlay)}

    async def tensix_bl_exec(self, x, y, slot="A", wait=True, timeout=5.0, force=False):
        return await self._run(self._tensix_bl_exec, x, y, slot, wait, timeout, force)

    def _tensix_bl_exec(self, x, y, slot, wait, timeout, force):
        from ..tensix import overlays, bootloader as blmod
        loaded = (self._bl_loaded.get((x, y), {}).get(slot.upper()) or {})
        ov = loaded.get("overlay")
        meta = overlays._reg().get(ov, {})
        if meta.get("verified") in ("wedges", "untested") and not force:
            raise ValueError(f"overlay {ov!r} is gated ({meta.get('verified')}): it can wedge the "
                             f"core. Pass force=true to run it anyway.")
        b = self._bl(x, y)
        r = b.exec(slot=slot)
        out = {**r, "overlay": ov}
        if wait:
            ok = b.wait_ack(timeout=float(timeout))
            s = b.status()
            telem = b.L.rd(blmod.TELEM_BASE, 8)
            out.update(ok=ok, status=s["status_name"], ovl_ret=s["ovl_ret"],
                       telemetry=overlays.decode_telemetry(ov, telem))
        return out

    async def tensix_bl_halt(self, x, y):
        return await self._run(self._tensix_bl_halt, x, y)

    def _tensix_bl_halt(self, x, y):
        self._bl(x, y).halt()
        return {"ok": True}

    # ---- LLK perf kernels: build on llk_lib (host) + load/run on a Tensix core (TRISC boot) ----
    async def llk_build(self, name, run_type=None):
        from ..tensix import llk_run
        return await asyncio.to_thread(llk_run.build, name, run_type)   # pure host compile, off device thread

    async def llk_disasm(self, name):
        from ..tensix import llk_run
        return await asyncio.to_thread(llk_run.disasm, name)

    async def overlay_disasm(self, name):
        from ..tensix import overlays
        return await asyncio.to_thread(overlays.disasm, name)

    async def llk_run(self, name, x, y, tile_cnt=16, timeout=5.0, run_type=None):
        return await self._run(self._llk_run, name, x, y, tile_cnt, timeout, run_type)

    def _llk_run(self, name, x, y, tile_cnt, timeout, run_type):
        from ..tensix import llk_run
        if self.reset_needed:
            raise ValueError("NoC hang pending — run `tt-smi -r 0` and restart the server")
        b = llk_run.build(name, run_type)                          # (re)build for the chosen run type
        if not b["ok"]:
            return {"ok": False, "stage": "build", "log": b["log"], "run_type": b.get("run_type")}
        coord = self._tensix_core(x, y).coord
        r = llk_run.run(name, coord, ctx=self.ctx, tile_cnt=int(tile_cnt), timeout=float(timeout))
        r["build_log"] = b["log"]
        r["run_type"] = b.get("run_type")
        return r

    # ---- LLK "test all": build every kernel once, run each on every core, summarize pass/fail ----
    def _all_tensix_cores(self):
        return [{"x": t.noc0[0], "y": t.noc0[1]} for t in self.fp.placed if t.kind == "tensix"]

    def _llk_run_built(self, name, x, y, tile_cnt, timeout):
        """Run an ALREADY-BUILT LLK kernel on one core (no rebuild). Device-thread (via _run)."""
        from ..tensix import llk_run
        if self.reset_needed:
            raise ValueError("NoC hang pending — run tt-smi -r 0 and restart the server")
        coord = self._tensix_core(x, y).coord
        return llk_run.run(name, coord, ctx=self.ctx, tile_cnt=int(tile_cnt), timeout=float(timeout))

    async def llk_test_all(self, cores=None, run_type=None, tile_cnt=16, timeout=5.0):
        """Smoke-test the whole LLK lane: build every kernel once, then run each on every core in
        `cores` (default = all Tensix cores). Background job; poll llk_test_last() for live progress
        + the final overview. Each (kernel,core) run records pass/fail + the error/status."""
        if (self._llk_test or {}).get("running"):
            return {"ok": False, "error": "an LLK test run is already in progress"}
        from ..tensix import llk
        cores = cores or self._all_tensix_cores()
        kernels = [k["name"] for k in (llk.load().get("kernels") or [])]
        self._llk_test = {"running": True, "done": 0, "total": len(kernels) * len(cores),
                          "run_type": run_type, "tile_cnt": int(tile_cnt), "timeout": float(timeout),
                          "cores": cores, "kernels": [], "summary": None, "aborted": None, "secs": 0.0}
        asyncio.create_task(self._llk_test_all_job(kernels, cores, run_type, int(tile_cnt), float(timeout)))
        return {"ok": True, "started": True, "kernels": len(kernels), "cores": len(cores),
                "total": self._llk_test["total"]}

    async def _llk_test_all_job(self, kernels, cores, run_type, tile_cnt, timeout):
        from ..tensix import llk_run
        t0 = time.monotonic()
        st = self._llk_test
        try:
            for name in kernels:
                krec = {"name": name, "build_ok": False, "build_err": None, "runs": [], "pass": 0, "fail": 0}
                st["kernels"].append(krec)
                try:
                    b = await asyncio.to_thread(llk_run.build, name, run_type)   # build once, off device thread
                except Exception as e:
                    b = {"ok": False, "log": str(e)}
                krec["build_ok"] = bool(b.get("ok"))
                if not krec["build_ok"]:
                    krec["build_err"] = (b.get("log") or "build failed").strip()[-400:]
                    st["done"] += len(cores)               # this kernel's per-core runs are skipped
                    st["secs"] = round(time.monotonic() - t0, 1)
                    continue
                for c in cores:
                    if self.reset_needed:
                        st["aborted"] = "NoC hang pending (recover: tt-smi -r 0) — stopped early"
                        break
                    try:
                        r = await self._run(self._llk_run_built, name, c["x"], c["y"], tile_cnt, timeout)
                        ok = bool(r.get("ok"))
                        rec = {"core": f'{c["x"]},{c["y"]}', "ok": ok,
                               "status": r.get("status", "?"), "error": None if ok else r.get("status", "error")}
                    except Exception as e:
                        ok = False
                        rec = {"core": f'{c["x"]},{c["y"]}', "ok": False, "status": "error", "error": str(e)[:200]}
                    krec["runs"].append(rec)
                    krec["pass" if ok else "fail"] += 1
                    st["done"] += 1
                    st["secs"] = round(time.monotonic() - t0, 1)
                if st["aborted"]:
                    break
        except Exception as e:
            st["aborted"] = f"test job crashed: {e}"
        finally:
            st["running"] = False
            st["secs"] = round(time.monotonic() - t0, 1)
            runs_pass = sum(k["pass"] for k in st["kernels"])
            runs_fail = sum(k["fail"] for k in st["kernels"])
            st["summary"] = {
                "kernels_total": len(kernels),
                "kernels_built": sum(1 for k in st["kernels"] if k["build_ok"]),
                "kernels_clean": sum(1 for k in st["kernels"] if k["build_ok"] and k["pass"] > 0 and k["fail"] == 0),
                "runs_total": runs_pass + runs_fail, "runs_pass": runs_pass, "runs_fail": runs_fail,
            }

    def llk_test_last(self):
        return self._llk_test or {"running": False, "kernels": [], "done": 0, "total": 0, "summary": None}

    # ---- bootloader telemetry stream (focused core, own rate) ----
    async def bl_subscribe(self):
        return await self.bl_chan.subscribe()

    def bl_unsubscribe(self, q):
        self.bl_chan.unsubscribe(q)

    def bl_select(self, x, y, hz=None):
        if x is not None and y is not None:
            self._bl_active = (int(x), int(y))
        if hz:
            self._bl_hz = max(1.0, min(10.0, float(hz)))

    async def _bl_tele_loop(self):
        while True:
            await asyncio.sleep(1.0 / max(self._bl_hz, 1.0))
            if len(self.bl_chan) == 0 or self._bl_active is None:
                continue
            x, y = self._bl_active
            if self.reset_needed or self._paused or self.mode not in ("polling", "injecting"):
                self._bl_broadcast({"x": x, "y": y, "paused": True, "mode": self.mode})
                continue
            try:
                frame = await self._run(self._tensix_bl_status, x, y)
                frame.update(x=x, y=y, ts=round(time.monotonic(), 3), hz=self._bl_hz)
            except Exception as e:                        # pragma: no cover
                frame = {"x": x, "y": y, "error": str(e)}
            self._bl_broadcast(frame)

    def _bl_broadcast(self, frame):
        self.bl_chan.broadcast(frame)

    # ---- static model + status (no device read) ---------------------------
    def floorplan_model(self):
        cols, rows = G.grid_dims(self.fp, "noc0")
        # The 2D torus is COMPLETE: coordinates without a functional block still
        # contain a router + NIU ("empty" tiles, per tt-isa-doc NoC/README — fused
        # tensix columns, top-row gaps, spine gaps). The rings pass through them
        # unbroken, so include them; die coords derive from the column/row-uniform
        # interleave. Never polled (not SAFE_KINDS), drawn dim by the UI.
        xmap, ymap = {}, {}
        for t in self.fp.placed:
            xmap.setdefault(t.noc0[0], t.die[0])
            ymap.setdefault(t.noc0[1], t.die[1])
        placed_keys = {t.noc0 for t in self.fp.placed}
        empties = [((x, y), (xmap[x], ymap[y]))
                   for x in range(cols) for y in range(rows)
                   if (x, y) not in placed_keys]
        overlay = G.card_overlay(self.fp, extra=empties)
        tiles = [{"noc0": list(t.noc0), "die": list(t.die), "kind": t.kind,
                  "label": t.label, "dram_ctrl": t.dram_ctrl,
                  "rect": overlay.get(t.noc0)}
                 for t in self.fp.placed]
        tiles += [{"noc0": list(k), "die": list(d), "kind": "empty",
                   "label": f"{k[0]},{k[1]}", "dram_ctrl": None,
                   "rect": overlay.get(k)}
                  for k, d in empties]
        ctrls = sorted(self.fp.dram_ctrl.keys())
        return {
            "image": {"src": "/api/card.png", "w": G.CARD_IMAGE_PX[0],
                      "h": G.CARD_IMAGE_PX[1], "package": G.CARD_PACKAGE_PX},
            "noc0_dims": [cols, rows],
            "kind_rgb": {**KIND_RGB, "empty": (95, 100, 112)},
            "safe_kinds": sorted(SAFE_KINDS), "tiles": tiles,
            "dram": {"ctrls": ctrls, "per_ctrl_gib": 4, "total_gib": len(ctrls) * 4},
            "pcie": {"link": "Gen5 x16", "gbps_per_dir": 63},
        }

    def status(self):
        return {"mode": self.mode, "reset_needed": self.reset_needed,
                "last_error": self.last_error, "hz": self.hz,
                "clients": len(self.noc_chan),
                "streaming": bool(self._stream), "cum_bytes": self._cum_bytes,
                "last_inject": self._last_inject}

    # ---- per-tile drilldown (device read on the worker) -------------------
    async def tile_detail(self, x, y):
        return await self._run(self._tile_detail, x, y)

    def _tile_detail(self, x, y):
        t = self._by_noc0.get((x, y))
        if t is None:
            return None
        out = {"noc0": list(t.noc0), "die": list(t.die), "kind": t.kind,
               "label": t.label, "dram_ctrl": t.dram_ctrl}
        if t.kind in SAFE_KINDS:
            nius = {}
            for noc in (0, 1):
                try:
                    words = read_words_from_device(
                        t.coord, nc.counter_base(noc), word_count=nc.COUNTER_ARRAY_LEN,
                        noc_id=noc, context=self.ctx)
                except Exception:
                    continue
                nius[str(noc)] = {
                    "counters": {nc.COUNTERS[i]: words[i] for i in nc.COUNTERS},
                    "bw": self.poller.bw.get((t.key, noc), {}),
                }
            out["nius"] = nius
            out["dram_affinity"] = G.dram_affinity(self.fp, t)
            out["fold_seam"] = [{"dir": d, "neighbor": list(nb.noc0), "hops": h}
                                for d, nb, h in G.physical_neighbor_hops(self.fp, t)]
        return out
