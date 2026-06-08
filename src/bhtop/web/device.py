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
        self._clients = set()        # set[asyncio.Queue]
        self._last_frame = None

    # ---- lifecycle ---------------------------------------------------------
    async def start(self):
        self._loop = asyncio.get_running_loop()
        await self._run(self._init_device)
        self.mode = "polling"
        asyncio.create_task(self._poll_loop())

    async def _run(self, fn, *a):
        """Run a blocking device call on the single worker thread."""
        return await self._loop.run_in_executor(self._exec, lambda: fn(*a))

    def _init_device(self):
        from ..inject import Injector
        self.ctx = init_ttexalens()
        self.fp = build(self.ctx)
        self._by_noc0 = {t.noc0: t for t in self.fp.placed}
        self.poller = Poller(self.fp, self.ctx)
        self.injector = Injector(self.fp, self.ctx)
        self.poller.sample()           # prime counters

    # ---- poll loop + broadcast --------------------------------------------
    async def _poll_loop(self):
        period = 1.0 / self.hz
        while True:
            if not self._paused and self.mode in ("polling", "injecting"):
                try:
                    frame = await self._run(self._sample_and_frame)
                    self._broadcast(frame)
                except Exception as e:                       # pragma: no cover
                    self.last_error = str(e)
            await asyncio.sleep(period)

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
        dram = {str(c): sum(self.poller.scalar(t, 0, "total") + self.poller.scalar(t, 1, "total")
                            for t in ts)
                for c, ts in self.fp.dram_ctrl.items()}
        return {"ts": round(time.monotonic(), 3), "mode": self.mode,
                "reset_needed": self.reset_needed, "tiles": tiles, "dram": dram}

    def _broadcast(self, frame):
        self._last_frame = frame
        for q in list(self._clients):
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                pass

    async def subscribe(self):
        q = asyncio.Queue(maxsize=4)
        if self._last_frame:
            q.put_nowait(self._last_frame)
        self._clients.add(q)
        return q

    def unsubscribe(self, q):
        self._clients.discard(q)

    # ---- static model + status (no device read) ---------------------------
    def floorplan_model(self):
        overlay = G.card_overlay(self.fp)
        cols, rows = G.grid_dims(self.fp, "noc0")
        tiles = [{"noc0": list(t.noc0), "die": list(t.die), "kind": t.kind,
                  "label": t.label, "dram_ctrl": t.dram_ctrl,
                  "rect": overlay.get(t.noc0)}
                 for t in self.fp.placed]
        return {
            "image": {"src": "/api/card.png", "w": G.CARD_IMAGE_PX[0],
                      "h": G.CARD_IMAGE_PX[1], "package": G.CARD_PACKAGE_PX},
            "noc0_dims": [cols, rows], "kind_rgb": KIND_RGB,
            "safe_kinds": sorted(SAFE_KINDS), "tiles": tiles,
        }

    def status(self):
        return {"mode": self.mode, "reset_needed": self.reset_needed,
                "last_error": self.last_error, "hz": self.hz,
                "clients": len(self._clients)}

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
