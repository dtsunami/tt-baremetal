"""
Live NIU-counter poller — the single source of truth for per-tile NoC bandwidth.

Shared by the Textual TUI (`textual_app`) and the web backend (`web.device`). Reads
the 62-entry NIU counter array on both NoCs for every data-movement tile, diffs
successive samples over wall-clock dt, and exposes per-tile/per-NoC bandwidths
keyed by the tile's noc0 identity.

SAFETY: only ever polls SAFE_KINDS (tensix / dram / eth). Reading NIU/router
registers on the management tiles (ARC / Security / PCIe / L2CPU) can wedge NoC0,
recoverable only with `tt-smi -r 0` (see the bh-noc-hang-hazard note).
"""
import time

from ttexalens.tt_exalens_lib import read_words_from_device

from . import noc_counters as nc


# Management tiles still render in the floorplan; they just never carry a heat value.
SAFE_KINDS = {"tensix", "dram", "eth"}


class Poller:
    """Reads NIU counters for the data-movement tiles on both NoCs; keys bandwidth by tile.key (noc0)."""
    def __init__(self, fp, ctx, calib=True):
        self.fp, self.ctx, self.calib = fp, ctx, calib
        self.prev, self.bw, self.last_t = {}, {}, None

    def sample(self):
        now = time.monotonic()
        dt = (now - self.last_t) if self.last_t else None
        cal = (nc.COUNTER_ARRAY_LEN + nc.FLIT_BYTES - 1) // nc.FLIT_BYTES
        for t in self.fp.addressable():
            if t.kind not in SAFE_KINDS:        # never touch management-tile NIUs (hang hazard)
                continue
            for noc in (0, 1):
                try:
                    words = read_words_from_device(
                        t.coord, nc.counter_base(noc), word_count=nc.COUNTER_ARRAY_LEN,
                        noc_id=noc, context=self.ctx)
                except Exception:
                    continue
                prev = self.prev.get((t.key, noc))
                if prev is not None and dt:
                    bw = nc.tile_bandwidths(words, prev, dt)
                    if self.calib:
                        bw["tx_slave"] = max(0.0, bw["tx_slave"] - cal * nc.FLIT_BYTES / dt)
                    self.bw[(t.key, noc)] = bw
                self.prev[(t.key, noc)] = words
        self.last_t = now

    def scalar(self, tile, noc, metric):
        if tile is None:
            return 0.0
        return nc.metric_scalar(self.bw.get((tile.key, noc), {}), metric)
