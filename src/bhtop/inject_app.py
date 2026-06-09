"""
bhtop-inject — interactive NoC traffic-injection explorer.

Drive real traffic from a source Tensix to chosen destinations (other Tensix, or
the GDDR6 controllers), and read back, from silicon, the route the NoC chose plus
bandwidth/flit numbers:
  * the MESH paints the measured route (decoded 0x500 transit counters) on the die;
  * the DATA panel gives moved bytes / flits / host bandwidth and the busiest nodes;
  * the GDDR6 panel gives per-controller writes-landed flits + bandwidth.

By default it STREAMS: it re-fires every tick so traffic is sustained and the
numbers update live (a one-shot burst is microseconds — you'd see nothing). Toggle
streaming with `x`, move the source with WASD, `r` to float it across the torus,
patterns 1–5, `f` for a single shot. Patterns 1–4 are Tensix↔Tensix; pattern 5
writes to every GDDR6 controller so their numbers light up. Because the traffic is
real silicon traffic, a separate `bhtop` (live view) running alongside sees it too.

LIMITATION: host injection currently drives NoC0 only (east+south). Destinations
west/north of the source are still reached, but NoC0 wraps the long way round the
torus to get there — visible as a route that lights the far columns. Dual-NoC
injection (NoC1 for west/north) is the next step; until then the route to such
nodes is the long way, not the short way a real kernel would take on NoC1.
"""
import random
import time

from rich.text import Text
from rich.table import Table
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Footer, Static

from ttexalens import init_ttexalens

from .floorplan import build
from .inject import Injector
from .render import render_mesh, _rgb, _fmt, NOC0_COL, NOC1_COL

DIRS = [(0, -1), (0, 1), (-1, 0), (1, 0)]
DRAM_RGB = (150, 205, 170)
NOC0_C, NOC1_C = NOC0_COL, NOC1_COL     # NoC0 purple (E+S) · NoC1 cyan (W+N)


from .patterns import ring as _ring, point as _pt   # shared with web.device


# (name, fn(center_tile, cells)->[(src,dst)], toggle-key, colour)
# "→GDDR6·write" is handled specially in fire() (uses probed write-access tiles).
PATTERNS = [
    ("gather·3hop",  lambda c, cl: [(n, c) for n in _ring(cl, c.noc0, 1, 3)], "1", (120, 230, 140)),
    ("scatter·3hop", lambda c, cl: [(c, n) for n in _ring(cl, c.noc0, 1, 3)], "2", (255, 120, 60)),
    ("neighbor·halo", lambda c, cl: [(c, n) for n in _ring(cl, c.noc0, 1, 1)], "3", (120, 160, 255)),
    ("scatter·far",  lambda c, cl: _pt(cl, c, (10, 0)), "4", (240, 90, 90)),
    ("→GDDR6·write", lambda c, cl: [], "5", DRAM_RGB),
]


def _fmtB(b):
    for u, s in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if b >= s:
            return f"{b/s:.1f} {u}"
    return f"{b:.0f} B"


class MeshView(Static):
    def draw(self, app):
        # Render the MEASURED route (0x500 transit footprint) as NoC0 flow arrows on
        # the physical die. Patterns are computed in noc0 space; the route is drawn on
        # the real floorplan where the hops are physically real.
        rcells, rcols, rrows = app.render_grid()
        f0, f1 = app.foot[0], app.foot[1]            # per-NoC route footprints (noc0 keys)
        mx = max([1] + list(f0.values()) + list(f1.values()))
        def load(x, y, noc):
            tile = rcells.get((x, y))
            if not tile:
                return 0.0
            return (f0 if noc == 0 else f1).get(tile.noc0, 0) / mx
        marks = {}
        for s, dt in app.pairs:
            marks.setdefault(s.key, "○")             # ○ source
            marks[dt.key] = "●"                      # ● destination (Tensix or GDDR6)
        ct = app.cells.get(app.center)
        if ct:
            marks[ct.key] = "◉"                      # ◉ source centre (wins ties)
        # both NoCs drawn together: NoC0 east+south (purple), NoC1 west+north (cyan)
        txt = render_mesh(rcells, rcols, rrows, load=load, noc_mode=2, scale=1.0,
                          sel_key=(ct.key if ct else None), arrows=True, marks=marks, dual=True)
        lay = "physical·die" if app.render_layout == "die" else "topology·noc0"
        fl = "[bold magenta]FLOAT[/]" if app.floating else "WASD"
        self.border_title = f"route · NoC0▸▾ + NoC1◂▴ · {lay} · src◉ {app.center} · {fl}"
        self.update(txt)


class InjectApp(App):
    CSS = """
    Horizontal { height: 1fr; }
    MeshView { width: auto; height: auto; border: round $accent; padding: 0 1; margin: 0 1; }
    #side { width: 1fr; min-width: 38; }
    #pat  { height: auto; border: round green; padding: 0 1; }
    #data { height: auto; border: round cyan; padding: 0 1; }
    #dram { height: auto; border: round magenta; padding: 0 1; }
    #help { height: auto; border: round grey; padding: 0 1; }
    """
    BINDINGS = [
        ("w", "move(0,-1)", "up"), ("s", "move(0,1)", "down"),
        ("a", "move(-1,0)", "left"), ("d", "move(1,0)", "right"),
        ("1", "toggle(0)", ""), ("2", "toggle(1)", ""), ("3", "toggle(2)", ""),
        ("4", "toggle(3)", ""), ("5", "toggle(4)", ""),
        ("x", "stream", "stream"),
        ("l", "layout", "layout"),
        ("r", "float", "float"), ("f", "fire", "fire"), ("q", "quit", "quit"),
    ]

    def __init__(self, length=0x80000, fires=1, float_hz=2.0):
        super().__init__()
        self.length, self.fires, self.float_hz = length, fires, float_hz
        self.center = (3, 4)
        self.enabled = {"→GDDR6·write"}
        self.streaming = True     # re-fire each tick so traffic is sustained + visible
        self.floating = False
        self.render_layout = "die"
        self.fire_count = 0
        self.foot = {0: {}, 1: {}}     # per-NoC computed route footprints
        self.total, self.secs, self.pairs = 0, 0.0, []
        self.dram = {}            # {ctrl -> {0: noc0 flits landed, 1: noc1 flits landed}}
        self.t0 = None; self.cum_bytes = 0    # for sustained (time-averaged) BW

    def compose(self) -> ComposeResult:
        yield Horizontal(
            MeshView(id="mesh"),
            Vertical(Static(id="pat"), Static(id="data"), Static(id="dram"),
                     Static(id="help"), id="side"),
        )
        yield Footer()

    def on_mount(self):
        self.ctx = init_ttexalens()
        self.fp = build(self.ctx)
        self.cells, self.cols, self.rows = self.fp.grid("noc0")   # patterns live in noc0 space
        self.dgrid = self.fp.grid("die")                          # physical render grid
        self.inj = Injector(self.fp, self.ctx)
        self._help()
        self.fire()
        self.set_interval(1.0 / self.float_hz, self._tick)

    def render_grid(self):
        return self.dgrid if self.render_layout == "die" else (self.cells, self.cols, self.rows)

    def action_layout(self):
        self.render_layout = "noc0" if self.render_layout == "die" else "die"
        self._redraw()

    # ---- source motion ----
    def action_move(self, dx, dy):
        self.floating = False
        self.center = ((self.center[0] + dx) % self.cols, (self.center[1] + dy) % self.rows)
        self.fire()

    def action_float(self):
        self.floating = not self.floating
        self._redraw()

    def action_stream(self):
        self.streaming = not self.streaming
        self._redraw()

    def _tick(self):
        # sustained traffic so the numbers actually move: float = walk+fire,
        # stream = re-fire in place, otherwise idle (single-shot via `f`).
        if self.floating:
            cx, cy = self.center
            for _ in range(24):                   # random-walk, only COMMIT Tensix landings
                dx, dy = random.choice(DIRS)
                cx, cy = (cx + dx) % self.cols, (cy + dy) % self.rows   # torus wrap
                t = self.cells.get((cx, cy))
                if t and t.kind == "tensix":
                    self.center = (cx, cy)
                    self.fire()
                    return
        elif self.streaming:
            self.fire()

    def action_fire(self):
        self.fire()

    # ---- drive + measure ----
    def fire(self):
        c = self.cells.get(self.center)
        self.pairs = []
        if c and c.kind == "tensix":
            for name, fn, key, col in PATTERNS:
                if name not in self.enabled:
                    continue
                if name == "→GDDR6·write":          # probed write-access tile per ctrl
                    self.pairs += [(c, t) for t in self.inj.dram_access_tiles().values()]
                else:
                    self.pairs += fn(c, self.cells)
        if self.pairs:
            db0 = self.inj.read_dram_counters()          # GDDR6 baseline (per-NoC writes landed)
            self.foot, self.total, self.secs = self.inj.run_dual(
                self.pairs, length=self.length, fires=self.fires)
            db1 = self.inj.read_dram_counters()
            M = 0xFFFFFFFF
            self.dram = {c: {n: (db1[c][n] - db0[c][n]) & M for n in (0, 1)} for c in db0}
            self.fire_count += 1
            if self.t0 is None:
                self.t0 = time.monotonic()
            self.cum_bytes += self.total
        else:
            self.foot, self.total, self.secs, self.dram = {0: {}, 1: {}}, 0, 0.0, {}
        self._redraw()

    def action_toggle(self, i):
        self.enabled ^= {PATTERNS[i][0]}
        self.fire()

    # ---- panels ----
    def _redraw(self):
        self.query_one("#mesh", MeshView).draw(self)
        self._pat(); self._data(); self._dram()

    def _pat(self):
        t = Text("patterns (toggle 1-5)\n", style="bold")
        for name, fn, key, col in PATTERNS:
            on = name in self.enabled
            t.append(f" [{key}] {'▣' if on else '☐'} ", style=_rgb(col) if on else "grey50")
            t.append(f"{name}\n", style="" if on else "grey50")
        t.append("\nfloat: ")
        t.append("ON" if self.floating else "off",
                 style="bold magenta" if self.floating else "grey50")
        t.append("  (r)\n")
        self.query_one("#pat", Static).update(t)

    def _data(self):
        secs = self.secs or 1.0
        mode = "[bold green]● STREAM[/]" if self.streaming else \
               ("[bold magenta]FLOAT[/]" if self.floating else "[grey50]paused (x)[/]")
        burst = self.total / secs
        sustained = self.cum_bytes / max(1e-6, time.monotonic() - self.t0) if self.t0 else 0.0
        n0 = sum(1 for v in self.foot[0].values() if v)
        n1 = sum(1 for v in self.foot[1].values() if v)
        tbl = Table(expand=True, padding=0, title=f"injection · {mode} · #{self.fire_count}")
        tbl.add_column("metric"); tbl.add_column("value", justify="right")
        tbl.add_row("source", f"{self.center}")
        tbl.add_row("paths", f"{len(self.pairs)}")
        tbl.add_row("moved/fire", f"{_fmtB(self.total)} · {self.total // 64}f")
        tbl.add_row("burst BW", _fmt(burst))
        tbl.add_row("[dim]sustained[/]", f"[dim]{_fmt(sustained)} ≈ live[/]")
        tbl.add_row("route nodes", f"[{_rgb(NOC0_C)}]{n0}[/] / [{_rgb(NOC1_C)}]{n1}[/]")
        self.query_one("#data", Static).update(tbl)

    def _dram(self):
        secs = self.secs or 1.0
        tbl = Table(expand=True, padding=0, title="GDDR6 ctrl · landed (NoC0|NoC1)")
        tbl.add_column("d")
        tbl.add_column("NoC0", justify="right"); tbl.add_column("NoC1", justify="right")
        tbl.add_column("BW", justify="right")
        agg = 0
        for c in sorted(self.dram):
            n0 = self.dram[c].get(0, 0); n1 = self.dram[c].get(1, 0)
            agg += n0 + n1
            c0 = f"[{_rgb(NOC0_C)}]{n0}[/]" if n0 else "[grey42]·[/]"
            c1 = f"[{_rgb(NOC1_C)}]{n1}[/]" if n1 else "[grey42]·[/]"
            tbl.add_row(f"d{c}", c0, c1, _fmt((n0 + n1) * 64 / secs))
        tbl.add_row("Σ", "", "", _fmt(agg * 64 / secs))
        self.query_one("#dram", Static).update(tbl)

    def _help(self):
        t = Text()
        t.append("WASD", style="bold cyan"); t.append(" move source  ")
        t.append("r", style="bold magenta"); t.append(" float\n")
        t.append("1-5", style="bold cyan"); t.append(" patterns  ")
        t.append("x", style="bold green"); t.append(" stream  ")
        t.append("f", style="bold cyan"); t.append(" 1-shot  ")
        t.append("l", style="bold cyan"); t.append(" layout  ")
        t.append("q", style="bold cyan"); t.append(" quit\n")
        t.append("◉ src  ● dst   ", style="dim")
        t.append("NoC0▸▾", style=_rgb(NOC0_C)); t.append(" E+S  ", style="dim")
        t.append("NoC1◂▴", style=_rgb(NOC1_C)); t.append(" W+N\n", style="dim")
        t.append("each dst routed on its shorter NoC · burst=peak,\n", style="dim")
        t.append("sustained≈ what a live `bhtop` alongside sees", style="dim italic")
        self.query_one("#help", Static).update(t)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="bhtop-inject — NoC injection + routing/bandwidth explorer")
    ap.add_argument("--mb", type=float, default=0.5, help="MB per path (default 0.5)")
    args = ap.parse_args()
    InjectApp(length=int(args.mb * 1024 * 1024)).run()


if __name__ == "__main__":
    main()
