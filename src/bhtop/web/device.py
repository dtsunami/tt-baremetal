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
        self._cells = {}             # noc0 (x,y) -> Tile, for pattern building
        self._stream = None          # active streaming inject spec, or None
        self._stream_src = None
        self._last_inject = None
        self._cum_bytes = 0

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
                "clients": len(self._clients),
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
