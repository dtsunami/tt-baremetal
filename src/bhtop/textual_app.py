"""
bhtop — live Blackhole NoC TUI that distinguishes PHYSICAL layout from LOGICAL topology.

Every tile carries two silicon coordinates: `die` (physical placement) and `noc0`
(logical 2D-torus index). The NoC is a folded/interleaved torus — the logical
numbering is permuted so the torus wraparound wires stay physically short. That
fold is exactly why "1 logical hop" can mean very different physical distances,
which is what drives real routing / sharding decisions.

Two views, toggled with `l`:
  * PHYSICAL — the real die floorplan (silicon `die` coords), rotatable with `r`.
    Default is card orientation (GDDR6 top/bottom, Ethernet left, power/PCIe right);
    `r` rotates, `r`→0° gives the die-geographic view (DRAM left/right) that matches
    the official Tenstorrent floorplan. Only "DRAM on two opposite die edges" is a
    pure silicon fact; which physical card edge that is comes from the board photo.
  * TOPOLOGY — the noc0 folded torus the router actually addresses.

Every link draws BOTH NoC wires (NoC0 ▸▾ purple / NoC1 ◂▴ cyan). Overlays:
  * `d` distance — tint each link by physical die-distance (the fold seam lights up).
  * `f` fold — colour tiles by noc0 index, so the logical ordering is visible snaking
    across physical space (up-ramp then fold-back).
"""
import time

from rich.table import Table
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Static

from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_words_from_device

from . import noc_counters as nc
from . import geometry as G
from .floorplan import build, KIND_RGB
from .render import (
    NOC0_COL, NOC1_COL, render_mesh, render_physical, _blend, _rgb, _fmt,
)


# Only ever poll the data-movement fabric. Reading NIU/router registers on the
# management tiles (ARC / Security / PCIe / L2CPU) can wedge NoC0 — recoverable
# only with `tt-smi -r 0`. See the bh-noc-hang-hazard note. Management tiles still
# render in the grid (from the floorplan); they just never carry a heat value.
SAFE_KINDS = {"tensix", "dram", "eth"}

OVERLAY_NAME = {None: "bandwidth", "dist": "phys-distance", "index": "fold/logical-order"}


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


class RoutingView(Static):
    def draw(self, app):
        arr = "links" if app.arrows else "heat"
        ov = OVERLAY_NAME[app.overlay]
        if app.layout == "physical":
            orient = G.ORIENTS.get(app.orient, f"{app.orient}°").split("(")[0].strip()
            self.border_title = f"Blackhole p150a · PHYSICAL · {orient} · {ov} · {arr}"
        else:
            nocs = {0: "NoC0", 1: "NoC1", 2: "NoC0+1"}[app.noc_mode]
            self.border_title = f"Blackhole · TOPOLOGY noc0 (folded torus) · {nocs} · {ov} · {arr}"
        self.update(app.draw_mesh())


class BhtopApp(App):
    CSS = """
    Horizontal { height: 1fr; }
    RoutingView { width: auto; height: auto; border: round $accent; padding: 0 1; margin: 0 1; }
    #side { width: 1fr; min-width: 38; }
    #detail { height: auto; border: round white; padding: 0 1; }
    #dram   { height: auto; border: round cyan; padding: 0 1; }
    #help   { height: auto; border: round grey; padding: 0 1; }
    """
    BINDINGS = [
        ("l", "layout", "layout"),
        ("r", "rotate", "rotate"),
        ("d", "dist", "dist"),
        ("f", "fold", "fold"),
        ("a", "arrows", "arrows"),
        ("m", "metric", "metric"),
        ("n", "noc", "NoC"),
        ("c", "calib", "calib"),
        ("up", "move(0,-1)", ""), ("down", "move(0,1)", ""),
        ("left", "move(-1,0)", ""), ("right", "move(1,0)", ""),
        ("q", "quit", "quit"),
    ]

    def __init__(self, hz=2.0, calib=True, layout="physical", orient=90, arrows=True):
        super().__init__()
        self.hz, self.metric, self.noc_mode, self.calib = hz, "total", 2, calib
        self.layout, self.orient, self.arrows = layout, orient, arrows
        self.overlay = None
        self.scale = 1e7
        self.sel = (0, 1)        # cursor (in the active layout's coords)
        self.sel_key = None

    def compose(self) -> ComposeResult:
        yield Horizontal(
            RoutingView(id="mesh"),
            Vertical(Static(id="detail"), Static(id="dram"), Static(id="help"), id="side"),
        )
        yield Footer()

    def on_mount(self):
        self.ctx = init_ttexalens()
        self.fp = build(self.ctx)
        self.die_dims = G.grid_dims(self.fp, "die")
        self.noc_dims = G.grid_dims(self.fp, "noc0")
        self.topo_grid = self.fp.grid("noc0")
        # static physical-vs-logical geometry (computed once from silicon coords)
        self.topo_link_dist = G.link_die_distance(self.fp)          # logical link -> physical length
        self.topo_link_max = max(self.topo_link_dist.values(), default=1)
        self.phys_link_hops = G.physical_link_hops(self.fp)        # physical link -> logical length
        self.phys_link_max = max(self.phys_link_hops.values(), default=1)
        self.link_stats = G.link_distance_stats(self.fp)
        self._use_grid()
        self._sync_sel()
        self.poller = Poller(self.fp, self.ctx, self.calib)
        self.poller.sample()
        self.set_interval(1.0 / self.hz, self.tick)

    # ---- grid / layout ----
    def _physical_grid(self):
        dcols, drows = self.die_dims
        cells, w, h = {}, dcols, drows
        for t in self.fp.placed:
            x, y, w, h = G.rotate(t.die, self.orient, dcols, drows)
            cells[(x, y)] = t
        return cells, w, h

    def _use_grid(self):
        if self.layout == "physical":
            self.cells, self.cols, self.rows = self._physical_grid()
        else:
            self.cells, self.cols, self.rows = self.topo_grid

    def _key_pos(self, key):
        if key is None:
            return None
        for (x, y), t in self.cells.items():
            if t.key == key:
                return (x, y)
        return None

    def _relayout(self):
        """Rebuild the active grid and keep the cursor on the same physical tile."""
        self._sync_sel()
        self._use_grid()
        pos = self._key_pos(self.sel_key)
        self.sel = pos or (min(self.sel[0], self.cols - 1), min(self.sel[1], self.rows - 1))
        self._sync_sel()

    def _index_rgb(self, tile):
        """Fold view: colour a tile by its noc0 column index (blue=low → yellow=high)."""
        frac = tile.noc0[0] / max(1, self.noc_dims[0] - 1)
        return _blend((45, 95, 235), (255, 215, 45), frac)

    def draw_mesh(self) -> Text:
        cells = self.cells
        def load(x, y, noc):
            return self.poller.scalar(cells.get((x, y)), noc, self.metric)
        physical = self.layout == "physical"
        # distance overlay shows the CROSS metric: in the physical view, tint each
        # physical link by its noc0-hop length (fold seam glows); in topology, tint
        # each logical link by its physical die-length.
        ld = self.phys_link_hops if physical else self.topo_link_dist
        lm = self.phys_link_max if physical else self.topo_link_max
        kw = dict(load=load, scale=self.scale, sel_key=self.sel_key, arrows=self.arrows,
                  dual=True, overlay=self.overlay, link_dist=ld, link_dist_max=lm)
        if self.overlay == "index":
            kw["tile_rgb"] = self._index_rgb
        if physical:
            top, bot = self._phys_labels()
            return render_physical(cells, self.cols, self.rows,
                                   top_label=top, bot_label=bot, **kw)
        return render_mesh(cells, self.cols, self.rows, noc_mode=self.noc_mode, **kw)

    def _phys_labels(self):
        # captions name where the real tiles land per rotation (verified from die coords)
        lr = "◀ GDDR6 (DRAM)    (DRAM) GDDR6 ▶"
        o = self.orient
        if o == 0:        # DRAM left/right · Eth+ARC+PCIe band on top
            return "▲ Ethernet · ARC · PCIe ▲", lr
        if o == 180:      # DRAM left/right · Eth+ARC+PCIe band on bottom
            return lr, "▼ Ethernet · ARC · PCIe ▼"
        if o == 90:       # DRAM top/bottom · Eth+ARC+PCIe on the left (card default)
            return "▲ GDDR6 (DRAM) ▲", "▼ GDDR6 (DRAM) ▼   ◀ Eth·ARC·PCIe"
        return "▲ GDDR6 (DRAM) ▲", "▼ GDDR6 (DRAM) ▼   Eth·ARC·PCIe ▶"   # 270: on the right

    def tick(self):
        self.poller.sample()
        peak = 0.0
        for t in self.fp.addressable():
            v = (self.poller.scalar(t, 0, self.metric) + self.poller.scalar(t, 1, self.metric)
                 if self.noc_mode == 2 else self.poller.scalar(t, self.noc_mode, self.metric))
            peak = max(peak, v)
        self.scale = max(peak, self.scale * 0.9, 1e7)
        self.query_one("#mesh", RoutingView).draw(self)
        self._detail()
        self._dram()
        self._help()

    # ---- selection ----
    def _sync_sel(self):
        t = self.cells.get(self.sel)
        self.sel_key = t.key if t else None

    def action_move(self, dx, dy):
        self.sel = (min(max(self.sel[0] + dx, 0), self.cols - 1),
                    min(max(self.sel[1] + dy, 0), self.rows - 1))
        self._sync_sel()

    # ---- panels ----
    def _detail(self):
        t = self.cells.get(self.sel)
        d = Text()
        d.append(f"cursor {self.layout}={self.sel}\n", style="dim")
        if not t:
            d.append("· router-only / empty node\n")
            self.query_one("#detail", Static).update(d)
            return
        rgb = KIND_RGB.get(t.kind, (200, 200, 200))
        d.append(f"{t.glyph} {t.label}", style=f"bold {_rgb(rgb)}")
        d.append(f"  {t.kind}\n")
        d.append(f"  noc0 {t.noc0}   die {t.die}\n", style="dim")
        if t.dram_ctrl is not None:
            d.append(f"  GDDR6 ctrl d{t.dram_ctrl} (3 tiles share 4GiB)\n", style="cyan")
        # physical-vs-logical readouts (static geometry)
        if t.kind == "tensix":
            aff = G.dram_affinity(self.fp, t)
            if aff:
                tag = "AGREE" if aff["agree"] else "DIVERGE"
                d.append(f"  nearest DRAM: phys d{aff['phys_ctrl']}={aff['phys_cells']}cells "
                         f"· logi d{aff['logi_ctrl']}={aff['logi_hops']}hops [{tag}]\n",
                         style="green" if aff["agree"] else "yellow")
            far = [r for r in G.physical_neighbor_hops(self.fp, t) if r[2] >= 4]
            if far:
                nm, _, h = max(far, key=lambda r: r[2])
                d.append(f"  ⚠ physical {nm}-neighbour is {h} noc0 hops away (FOLD SEAM)\n",
                         style="bold red")
            else:
                ns = G.neighbor_stretch(self.fp, t)
                avg = sum(r[2] for r in ns) / len(ns) if ns else 0
                d.append(f"  1 logical hop ≈ {avg:.0f} physical cells\n", style="dim")
        d.append("  2 NIUs (one router each):\n", style="dim")
        for noc, col, flow in ((0, _rgb(NOC0_COL), "▸▾ E+S"), (1, _rgb(NOC1_COL), "◂▴ W+N")):
            bw = self.poller.bw.get((t.key, noc), {})
            tot = nc.metric_scalar(bw, "total")
            d.append(f" NIU{noc}→NoC{noc} {flow} Σ {_fmt(tot)}\n", style=col)
            if bw:
                d.append(f"   tx m/s {_fmt(bw['tx_master'])}/{_fmt(bw['tx_slave'])}\n", style="dim")
                d.append(f"   rx m/s {_fmt(bw['rx_master'])}/{_fmt(bw['rx_slave'])}\n", style="dim")
        self.query_one("#detail", Static).update(d)

    def _dram(self):
        tbl = Table(expand=True, padding=0, title=f"GDDR6 ctrl · {self.metric}")
        tbl.add_column("d"); tbl.add_column("bar"); tbl.add_column("BW", justify="right")
        vals = {}
        for c, ts in self.fp.dram_ctrl.items():
            s = sum(self.poller.scalar(t, 0, self.metric) + self.poller.scalar(t, 1, self.metric)
                    for t in ts)
            vals[c] = s
        mx = max(vals.values(), default=1) or 1
        agg = 0.0
        for c in sorted(vals):
            v = vals[c]; agg += v
            tbl.add_row(f"d{c}", f"[green]{'█'*int(8*v/mx):<8}[/]", _fmt(v))
        tbl.add_row("Σ", "", _fmt(agg))
        self.query_one("#dram", Static).update(tbl)

    def _help(self):
        t = Text()
        for kind, rgb in KIND_RGB.items():
            t.append("█", style=_rgb(rgb)); t.append(f"{kind[:4]} ")
        t.append("\n")
        t.append("NoC0 ▸▾", style=_rgb(NOC0_COL)); t.append(" E+S   ", style="dim")
        t.append("NoC1 ◂▴", style=_rgb(NOC1_COL)); t.append(" W+N\n", style="dim")
        # live, this-card link-distance histogram (the fold made visible in numbers)
        s = self.link_stats
        ih = " ".join(f"{n}@{d}" for d, n in s["interior"].items())
        wh = " ".join(f"{n}@{d}" for d, n in s["wrap"].items())
        t.append(f"links (this card): interior {ih} · wrap {wh}\n", style="dim")
        t.append("→ torus wrap is physically SHORT (fold); '1 hop' ≈ 2 die cells\n", style="dim")
        if self.layout == "physical":
            t.append(f"orientation: {G.ORIENTS.get(self.orient, str(self.orient))}\n", style="dim")
            t.append("certain: DRAM on opposite die edges · card-edge = board photo\n", style="dim italic")
            dhint = "[d]ist = physical link's noc0-hop length (seam→red)"
        else:
            dhint = "[d]ist = logical link's physical die-length"
        t.append(f"overlay: {OVERLAY_NAME[self.overlay]}   {dhint if self.overlay=='dist' else ''}\n",
                 style="bold yellow" if self.overlay else "dim")
        t.append("[l]ayout [r]otate [d]ist [f]old [a]rrows [m]etric [n]oc [c]alib ↑↓←→ [q]",
                 style="dim")
        self.query_one("#help", Static).update(t)

    # ---- actions ----
    def action_layout(self):
        self.layout = "topology" if self.layout == "physical" else "physical"
        self._relayout()

    def action_rotate(self):
        self.orient = (self.orient + 90) % 360
        if self.layout != "physical":
            self.layout = "physical"
        self._relayout()

    def action_dist(self):
        self.overlay = None if self.overlay == "dist" else "dist"

    def action_fold(self):
        self.overlay = None if self.overlay == "index" else "index"

    def action_arrows(self):
        self.arrows = not self.arrows

    def action_metric(self):
        self.metric = nc.METRICS[(nc.METRICS.index(self.metric) + 1) % len(nc.METRICS)]

    def action_noc(self):
        self.noc_mode = (self.noc_mode + 1) % 3

    def action_calib(self):
        self.calib = not self.calib
        self.poller.calib = self.calib


def main():
    import argparse
    ap = argparse.ArgumentParser(description="bhtop — live Blackhole NoC TUI (physical vs logical)")
    ap.add_argument("--hz", type=float, default=2.0, help="refresh rate (default 2)")
    ap.add_argument("--topology", action="store_true", help="open in the noc0 folded-torus view")
    ap.add_argument("--orient", type=int, default=90, choices=(0, 90, 180, 270),
                    help="physical rotation (90=card default, 0=die-geographic)")
    ap.add_argument("--no-arrows", action="store_true", help="heat-only (no flow arrows)")
    ap.add_argument("--no-calib", action="store_true")
    args = ap.parse_args()
    BhtopApp(hz=args.hz, calib=not args.no_calib,
             layout="topology" if args.topology else "physical",
             orient=args.orient, arrows=not args.no_arrows).run()


if __name__ == "__main__":
    main()
