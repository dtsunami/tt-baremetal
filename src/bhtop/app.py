#!/usr/bin/env python3
"""
tt-noc-top : live Blackhole NoC bandwidth heatmap.

Renders the physical tile grid for NoC0 and NoC1 side by side, colouring each
tile by live bandwidth derived from its NIU hardware counters.

Keys (live):
  m / space  cycle metric (total / tx / rx / master / slave)
  0 / 1 / b  show NoC0 only / NoC1 only / both
  + / -      faster / slower refresh
  c          toggle self-read (observer) calibration
  q          quit
"""
import argparse
import sys
import termios
import threading
import time
import tty

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import noc_counters as nc
from .sampler import NocSampler, _fmt_bw

# heat ramp: cool -> hot (256-colour codes)
HEAT = [17, 18, 19, 20, 26, 31, 37, 43, 49, 83, 118, 154, 190, 226, 220, 214, 208, 202, 196]
KIND_GLYPH = {"tensix": "T", "dram": "D", "eth": "E"}

METRICS = ["total", "tx", "rx", "master", "slave"]


def metric_value(bw: dict, metric: str) -> float:
    if not bw:
        return 0.0
    if metric == "total":
        return sum(bw.values())
    if metric == "tx":
        return bw["tx_master"] + bw["tx_slave"]
    if metric == "rx":
        return bw["rx_master"] + bw["rx_slave"]
    if metric == "master":
        return bw["tx_master"] + bw["rx_master"]
    if metric == "slave":
        return bw["tx_slave"] + bw["rx_slave"]
    return 0.0


def heat_color(frac: float) -> int:
    frac = max(0.0, min(1.0, frac))
    return HEAT[int(frac * (len(HEAT) - 1))]


class KeyReader(threading.Thread):
    """Non-blocking single-key reader so rich.Live stays interactive."""
    def __init__(self):
        super().__init__(daemon=True)
        self.key = None
        self._lock = threading.Lock()
        self._stop = False

    def run(self):
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while not self._stop:
                ch = sys.stdin.read(1)
                with self._lock:
                    self.key = ch
        except Exception:
            pass
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def pop(self):
        with self._lock:
            k, self.key = self.key, None
        return k

    def stop(self):
        self._stop = True


class App:
    def __init__(self, refresh_hz: float, calib: bool):
        self.s = NocSampler()
        self.metric = "total"
        self.show = "b"            # 'b' both, '0' noc0, '1' noc1
        self.refresh = refresh_hz
        self.calib = calib
        self.scale = 1e6           # sticky auto-scale ceiling (bytes/s)
        cols = max(t.col for t in self.s.tiles) + 1
        rows = max(t.row for t in self.s.tiles) + 1
        self.cols, self.rows = cols, rows

    # self-read observer baseline: each polled tile serves COUNTER_ARRAY_LEN words
    # per read on each NoC; subtract that from the slave-read direction.
    def _calib_flits(self, dt):
        if not (self.calib and dt):
            return 0.0
        flits = (nc.COUNTER_ARRAY_LEN + nc.FLIT_BYTES - 1) // nc.FLIT_BYTES  # ceil words->flits
        return flits * nc.FLIT_BYTES / dt

    def grid_panel(self, noc_id: int, dt) -> Panel:
        cells = {}
        peak = 0.0
        calib = self._calib_flits(dt)
        for t in self.s.tiles:
            bw = dict(t.bw.get(noc_id, {}))
            if bw and self.calib:
                bw["tx_slave"] = max(0.0, bw.get("tx_slave", 0.0) - calib)
            val = metric_value(bw, self.metric)
            cells[(t.row, t.col)] = (t.kind, val)
            peak = max(peak, val)

        # sticky auto-scale: ratchet up fast, decay slowly
        self.scale = max(peak, self.scale * 0.92, 1e5)

        grid = Table.grid(padding=0)
        for _ in range(self.cols):
            grid.add_column()
        for r in range(self.rows):
            row_cells = []
            for c in range(self.cols):
                cell = cells.get((r, c))
                if cell is None:
                    row_cells.append(Text("  ", style="on grey7"))
                    continue
                kind, val = cell
                color = heat_color(val / self.scale)
                glyph = KIND_GLYPH[kind]
                style = f"black on color({color})" if val > 0 else f"grey50 on color({HEAT[0]})"
                row_cells.append(Text(f"{glyph} ", style=style))
            grid.add_row(*row_cells)
        title = f"NoC{noc_id}  peak {_fmt_bw(peak)}  scale {_fmt_bw(self.scale)}"
        return Panel(grid, title=title, border_style="cyan", padding=0)

    def top_panel(self, dt) -> Panel:
        calib = self._calib_flits(dt)
        rows = []
        for t in self.s.tiles:
            for noc_id, bw in t.bw.items():
                if self.show == "0" and noc_id != 0:
                    continue
                if self.show == "1" and noc_id != 1:
                    continue
                bw = dict(bw)
                if self.calib:
                    bw["tx_slave"] = max(0.0, bw.get("tx_slave", 0.0) - calib)
                val = metric_value(bw, self.metric)
                if val > 0:
                    rows.append((val, t.label, t.kind, noc_id, bw))
        rows.sort(reverse=True)
        tbl = Table(expand=True, padding=0)
        tbl.add_column("tile"); tbl.add_column("noc"); tbl.add_column(self.metric, justify="right")
        tbl.add_column("tx_mst", justify="right"); tbl.add_column("rx_mst", justify="right")
        tbl.add_column("tx_slv", justify="right"); tbl.add_column("rx_slv", justify="right")
        for val, label, kind, noc_id, bw in rows[:14]:
            tbl.add_row(f"{label}", f"{noc_id}", _fmt_bw(val),
                        _fmt_bw(bw["tx_master"]), _fmt_bw(bw["rx_master"]),
                        _fmt_bw(bw["tx_slave"]), _fmt_bw(bw["rx_slave"]))
        if not rows:
            tbl.add_row("idle", "-", "-", "-", "-", "-", "-")
        return Panel(tbl, title=f"busiest tiles  (metric={self.metric})", border_style="green")

    def header(self, dt, fps) -> Panel:
        agg = 0.0
        for t in self.s.tiles:
            for bw in t.bw.values():
                agg += sum(bw.values())
        txt = Text()
        txt.append(" bhtop ", style="bold black on cyan")
        txt.append(f"  Blackhole p150a   aggregate NoC: {_fmt_bw(agg)}   "
                   f"refresh {fps:.1f} Hz   metric={self.metric}   show={self.show}   "
                   f"calib={'on' if self.calib else 'off'}")
        txt.append("\n [m]etric  [0/1/b] noc  [+/-] rate  [c]alib  [q]uit", style="dim")
        return Panel(txt, border_style="white")

    def render(self, dt, fps):
        grids = []
        if self.show in ("b", "0"):
            grids.append(self.grid_panel(0, dt))
        if self.show in ("b", "1"):
            grids.append(self.grid_panel(1, dt))
        gtab = Table.grid()
        for g in grids:
            gtab.add_column()
        gtab.add_row(*grids)
        return Group(self.header(dt, fps), gtab, self.top_panel(dt))

    def handle_key(self, k):
        if k in ("m", " "):
            self.metric = METRICS[(METRICS.index(self.metric) + 1) % len(METRICS)]
        elif k in ("0", "1", "b"):
            self.show = k
        elif k == "+":
            self.refresh = min(30.0, self.refresh + 2)
        elif k == "-":
            self.refresh = max(1.0, self.refresh - 2)
        elif k == "c":
            self.calib = not self.calib
        elif k == "q":
            return False
        return True

    def run(self):
        keys = KeyReader(); keys.start()
        console = Console()
        # Terminal "favicon": set the window/tab title via OSC. Many terminals
        # (iTerm2, Kitty, WezTerm, GNOME, Windows Terminal) show this in the tab.
        console.set_window_title("bhtop — Blackhole NoC")
        self.s.sample()                       # prime counters
        try:
            with Live(console=console, screen=True, auto_refresh=False) as live:
                while True:
                    t0 = time.monotonic()
                    dt = self.s.sample()
                    fps = 1.0 / max(1e-3, time.monotonic() - t0)
                    live.update(self.render(dt, fps), refresh=True)
                    k = keys.pop()
                    if k and not self.handle_key(k):
                        break
                    time.sleep(max(0.0, 1.0 / self.refresh - (time.monotonic() - t0)))
        finally:
            keys.stop()


def main():
    ap = argparse.ArgumentParser(description="Live Blackhole NoC bandwidth heatmap")
    ap.add_argument("--hz", type=float, default=10.0, help="refresh rate (default 10)")
    ap.add_argument("--no-calib", action="store_true", help="disable self-read calibration")
    args = ap.parse_args()
    App(refresh_hz=args.hz, calib=not args.no_calib).run()


if __name__ == "__main__":
    main()
