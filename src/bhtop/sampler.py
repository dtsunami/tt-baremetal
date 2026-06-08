"""
Live NoC sampler for Blackhole.

Sweeps every Tensix / DRAM / Ethernet tile, reads the 62-word NIU counter
block over NoC0 and NoC1, and derives per-tile directional bandwidth from
the delta against the previous sweep.

Notes on the observer effect: each counter read is itself a NoC transaction
routed via the PCIe tile, so the sweep perturbs the network slightly. Reads
target register space (not L1) and the perturbation is small but nonzero --
keep the refresh modest (10-30 Hz) for a faithful picture.
"""
import time
from dataclasses import dataclass, field

from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_words_from_device

from . import noc_counters as nc


@dataclass
class Tile:
    kind: str                 # 'tensix' | 'dram' | 'eth'
    coord: object             # OnChipCoordinate (passed straight to the reader)
    label: str                # human label, e.g. '1,1' or 'd0,0'
    col: int                  # grid column (NoC0 X)
    row: int                  # grid row    (NoC0 Y)
    prev: dict = field(default_factory=dict)   # noc_id -> last counter words
    bw: dict = field(default_factory=dict)     # noc_id -> directional bytes/s


class NocSampler:
    def __init__(self, context=None):
        self.ctx = context or init_ttexalens()
        self.dev = self.ctx.devices[0]
        self.tiles = self._enumerate()
        self._last_t = None

    def _enumerate(self):
        tiles = []
        groups = [
            ("tensix", "functional_workers"),
            ("dram",   "dram"),
            ("eth",    "eth"),
        ]
        for kind, bt in groups:
            for loc in self.dev.get_block_locations(bt):
                x, y = loc.to("noc0")          # (col,row) in NoC0 physical space for layout
                tiles.append(Tile(kind=kind, coord=loc, label=str(loc), col=x, row=y))
        return tiles

    def _read_block(self, tile: Tile, noc_id: int):
        return read_words_from_device(
            tile.coord, nc.counter_base(noc_id),
            word_count=nc.COUNTER_ARRAY_LEN, noc_id=noc_id, context=self.ctx,
        )

    def sample(self):
        """One full sweep. Populates tile.bw[noc_id] = {tx_master,rx_master,tx_slave,rx_slave}."""
        now = time.monotonic()
        dt = (now - self._last_t) if self._last_t else None
        for tile in self.tiles:
            for noc_id in (0, 1):
                try:
                    words = self._read_block(tile, noc_id)
                except Exception:
                    continue
                prev = tile.prev.get(noc_id)
                if prev is not None and dt:
                    tile.bw[noc_id] = nc.tile_bandwidths(words, prev, dt)
                tile.prev[noc_id] = words
        self._last_t = now
        return dt


def _fmt_bw(b):
    for unit, scale in (("GB/s", 1e9), ("MB/s", 1e6), ("kB/s", 1e3)):
        if b >= scale:
            return f"{b/scale:6.2f} {unit}"
    return f"{b:6.0f}  B/s"


if __name__ == "__main__":
    print("Initializing sampler...")
    s = NocSampler()
    n_t = sum(t.kind == "tensix" for t in s.tiles)
    n_d = sum(t.kind == "dram" for t in s.tiles)
    n_e = sum(t.kind == "eth" for t in s.tiles)
    print(f"Tiles: {n_t} tensix, {n_d} dram, {n_e} eth  ->  {len(s.tiles)} tiles x 2 NoCs "
          f"= {len(s.tiles)*2} reads/frame\n")

    # Benchmark a full sweep to size the refresh rate.
    s.sample()                       # prime
    t0 = time.monotonic()
    s.sample()
    sweep = time.monotonic() - t0
    print(f"Full sweep: {sweep*1e3:6.1f} ms  ->  ~{1/sweep:4.1f} Hz max refresh "
          f"({sweep*1e6/(len(s.tiles)*2):.0f} us/read)\n")

    # Show the busiest tiles over a 1s window.
    s.sample()
    time.sleep(1.0)
    s.sample()
    rows = []
    for t in s.tiles:
        for noc_id, bw in t.bw.items():
            total = sum(bw.values())
            if total > 0:
                rows.append((total, t.label, noc_id, bw))
    rows.sort(reverse=True)
    print(f"Busiest tiles (1s window):  {'tile':>6}  noc   tx_mst    rx_mst    tx_slv    rx_slv")
    for total, label, noc_id, bw in rows[:12]:
        print(f"  {label:>6}  noc{noc_id}  "
              f"{_fmt_bw(bw['tx_master'])}  {_fmt_bw(bw['rx_master'])}  "
              f"{_fmt_bw(bw['tx_slave'])}  {_fmt_bw(bw['rx_slave'])}")
    if not rows:
        print("  (idle - no NoC traffic; run a workload to see activity)")
