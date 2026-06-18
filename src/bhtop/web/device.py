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

    # ---- lifecycle ---------------------------------------------------------
    async def start(self):
        self._loop = asyncio.get_running_loop()
        await self._run(self._init_device)
        self.mode = "polling"
        asyncio.create_task(self._poll_loop())
        asyncio.create_task(self._l2_tele_loop())

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
                                        "examples": metal.compute_examples()})

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
        passed, out = metal.run_example(name, timeout=timeout)
        compute = metal.aggregate_compute()          # per-Tensix-core per-engine cycles
        if compute:                                  # remember which kernel ran on which Tensix core
            short = name.replace("metal_example_", "")
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
