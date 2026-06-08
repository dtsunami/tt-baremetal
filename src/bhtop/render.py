"""
Shared NoC mesh renderer — one monolithic chip grid with both NoCs drawn as
opposite-direction flow arrows (the chip-scale version of the Tensix
data-movement / NoC-bandwidth slide). Reused by the live TUI, the injection
explorer, and the benchmark footprints so they all paint the chip the same way.

Routing convention (matches the slide and the decoded 0x500 transit counters):
  NoC0 routes east+south (+x,+y)  -> purple  ▸ ▾
  NoC1 routes west+north (-x,-y)  -> cyan    ◂ ▴

Each tile is heat-coloured by its bandwidth. A link between two adjacent tiles
is tinted by the activity of the nodes it connects. NOTE: the live per-tile NIU
counters measure inject/eject, not per-link transit, so live arrow brightness is
an endpoint proxy; the *true* per-link route is only measured during injection
benchmarks (inject.py's 0x500 decode), which feed this same renderer.

The grid is drawn on whichever coordinate system the caller hands in:
  * die  coords -> physical floorplan (I/O on the perimeter, neighbours real)
  * noc0 coords -> the interleaved 2D-torus topology (routing lattice)
In both, grid neighbours are genuine link neighbours (physical vs topological),
so the flow arrows are meaningful either way.
"""
from rich.style import Style
from rich.text import Text

from .floorplan import KIND_RGB

EMPTY = Style(bgcolor="grey11", color="grey35")
GRID_BG = Style(bgcolor="grey11")
HOT = (255, 60, 40)
NOC0_COL = (124, 124, 238)     # purple  (NoC0, east+south)
NOC1_COL = (78, 202, 212)      # cyan    (NoC1, west+north)
NOC_COL = {0: NOC0_COL, 1: NOC1_COL}

# (east-link glyph, south-link glyph) per NoC — NoC0 points +x/+y, NoC1 -x/-y
ARROW = {0: ("▸", "▾"), 1: ("◂", "▴")}   # ▸ ▾ ; ◂ ▴


def _blend(a, b, f):
    f = 0.0 if f < 0 else 1.0 if f > 1 else f
    return tuple(int(a[i] + (b[i] - a[i]) * f) for i in range(3))


def _rgb(c):
    return f"rgb({c[0]},{c[1]},{c[2]})"


def _fmt(b):
    for u, s in (("GB/s", 1e9), ("MB/s", 1e6), ("kB/s", 1e3)):
        if b >= s:
            return f"{b/s:.2f} {u}"
    return f"{b:.0f} B/s"


def cell_style(kind, frac, selected=False):
    base = KIND_RGB.get(kind)
    if base is None:
        return EMPTY
    bg = _blend(_blend((28, 30, 38), base, 0.55), HOT, frac)
    fg = (10, 10, 10) if frac > 0.4 else (240, 240, 240)
    st = Style(bgcolor=_rgb(bg), color=_rgb(fg), bold=frac > 0.45)
    return st + Style(reverse=True) if selected else st


def arrow_style(noc, frac):
    # idle links recede into the background; live flow brightens to NoC colour
    col = NOC_COL[noc]
    idle = _blend((20, 21, 28), col, 0.14)
    c = _blend(idle, col, max(0.0, frac) ** 0.6)   # clamp before the power (neg frac -> complex)
    return Style(color=_rgb(c), bgcolor="grey11")


def _link_noc(load, a, b, noc_mode, scale):
    """For the link joining tiles a=(x,y) and b=(x,y), pick (noc, frac).

    Link load is an endpoint proxy: the busier of the two nodes it connects on
    that NoC. In 'both' mode the dominant NoC on the link wins (so you see which
    network is carrying each region of the chip)."""
    def lk(noc):
        return max(load(a[0], a[1], noc), load(b[0], b[1], noc))
    if noc_mode in (0, 1):
        return noc_mode, lk(noc_mode) / scale
    l0, l1 = lk(0), lk(1)
    return (0 if l0 >= l1 else 1), max(l0, l1) / scale


def dist_ramp(d, dmax):
    """Physical-distance colour for the fold/distance overlay: near=blue, far=red.
    A noc0 1-hop link that is physically far (across the fold seam) lights up red."""
    span = max(1, dmax - 1)
    f = 0.0 if d <= 1 else min(1.0, (d - 1) / span)
    return _blend((70, 105, 205), (255, 70, 45), f)


def _emit_link(txt, axis, ta, tb, ac, bc, load, noc_mode, scale, dual,
               overlay, link_dist, link_dist_max):
    """Emit link cell(s) between tiles ta,tb. axis: 0=east(horiz) 1=south(vert).
    ac,bc are display (x,y) for the bandwidth lookup; ta,tb give noc0 identity."""
    if overlay == "dist" and link_dist is not None:
        d = link_dist.get(frozenset((ta.noc0, tb.noc0)), 1)
        st = Style(color=_rgb(dist_ramp(d, link_dist_max)), bgcolor="grey11",
                   bold=(d >= link_dist_max and d > 1))
        txt.append(ARROW[0][axis], style=st)
        if dual:
            txt.append(ARROW[1][axis], style=st)
        return
    if overlay == "index":                       # fold view: links recede, tiles carry it
        st = arrow_style(0, 0.0)
        txt.append(ARROW[0][axis], style=st)
        if dual:
            txt.append(ARROW[1][axis], style=arrow_style(1, 0.0))
        return
    if dual:
        f0 = max(load(ac[0], ac[1], 0), load(bc[0], bc[1], 0)) / scale
        f1 = max(load(ac[0], ac[1], 1), load(bc[0], bc[1], 1)) / scale
        txt.append(ARROW[0][axis], style=arrow_style(0, f0))
        txt.append(ARROW[1][axis], style=arrow_style(1, f1))
    else:
        noc, frac = _link_noc(load, ac, bc, noc_mode, scale)
        txt.append(ARROW[noc][axis], style=arrow_style(noc, frac))


def render_mesh(cells, cols, rows, *, load, noc_mode=2, scale=1.0,
                sel_key=None, arrows=True, marks=None, dual=False,
                overlay=None, link_dist=None, link_dist_max=1, tile_rgb=None):
    """Render the chip grid as a Rich Text lattice.

    cells   : {(x,y) -> Tile} in the chosen coord system
    load    : callable(x, y, noc) -> bandwidth (bytes/s); heat + arrow tint
    noc_mode: 0 NoC0 only · 1 NoC1 only · 2 both (heat=sum)
    scale   : normaliser for heat/arrow fraction (peak bandwidth)
    sel_key : Tile.key to highlight (cursor)
    arrows  : draw inter-tile flow arrows (False -> compact heat-only grid)
    marks   : {Tile.key -> glyph} overrides (e.g. src/dst/center markers)
    dual    : draw BOTH NoC wires per link (NoC0 ▸▾ + NoC1 ◂▴ side by side)
    overlay : None=live bandwidth; "dist"=tint links by physical die-distance
              (link_dist/link_dist_max); "index"=colour tiles by tile_rgb(tile)
              (the fold/logical-order view), links recede.
    """
    scale = scale or 1.0
    marks = marks or {}
    link_w = 2 if (arrows and dual) else 1
    txt = Text()

    def overlay_tile_style(t, sel):
        if overlay == "index" and tile_rgb is not None:
            base = Style(bgcolor=_rgb(tile_rgb(t)), color="rgb(15,15,18)", bold=True)
            return base + Style(reverse=True) if sel else base
        return cell_style(t.kind, 0.0, sel)              # "dist": structure only

    for y in range(rows):
        for x in range(cols):
            t = cells.get((x, y))
            if t:
                sel = (sel_key is not None and t.key == sel_key)
                if overlay in ("index", "dist"):
                    st = overlay_tile_style(t, sel)
                else:
                    if noc_mode == 2:
                        frac = (load(x, y, 0) + load(x, y, 1)) / scale
                    else:
                        frac = load(x, y, noc_mode) / scale
                    st = cell_style(t.kind, frac, sel)
                txt.append(marks.get(t.key, t.glyph), style=st)
            else:
                txt.append("·", style=EMPTY)            # · empty / router-only node
            east = cells.get((x + 1, y))
            if arrows and t and east:
                _emit_link(txt, 0, t, east, (x, y), (x + 1, y), load, noc_mode,
                           scale, dual, overlay, link_dist, link_dist_max)
            else:
                txt.append(" " * link_w, style=GRID_BG)
        txt.append("\n")
        if arrows and y < rows - 1:
            for x in range(cols):
                t = cells.get((x, y))
                south = cells.get((x, y + 1))
                if t and south:
                    _emit_link(txt, 1, t, south, (x, y), (x, y + 1), load, noc_mode,
                               scale, dual, overlay, link_dist, link_dist_max)
                else:
                    txt.append(" " * link_w, style=GRID_BG)
                txt.append(" ", style=GRID_BG)
            txt.append("\n")
    return txt


# ---- physical "card" view ---------------------------------------------------
# The PHYSICAL floorplan: the real die tiles (from silicon coords), framed, with
# honest edge captions. NOTHING is fabricated — the DRAM tiles that land on the
# top/bottom edges ARE the GDDR6 banks; the Eth tiles on the left ARE the Ethernet.
# The caller supplies an already-oriented cell dict (die coords rotated to taste);
# only "DRAM on two opposite die edges" is a pure silicon fact — which physical card
# edge that is comes from the board photo, so the caption says so.
def render_physical(cells, cols, rows, *, load, scale=1.0, sel_key=None,
                    arrows=True, dual=True, marks=None, overlay=None,
                    link_dist=None, link_dist_max=1, tile_rgb=None,
                    top_label="", bot_label="", side_note=""):
    mesh = render_mesh(cells, cols, rows, load=load, noc_mode=2, scale=scale,
                       sel_key=sel_key, arrows=arrows, marks=marks, dual=dual,
                       overlay=overlay, link_dist=link_dist,
                       link_dist_max=link_dist_max, tile_rgb=tile_rgb)
    lines = mesh.split("\n")
    while lines and lines[-1].plain == "":
        lines.pop()
    mw = max((len(ln.plain) for ln in lines), default=0)

    out = Text()
    if top_label:
        out.append(" " + top_label.center(mw) + "\n", style="dim")
    out.append("╔" + "═" * mw + "╗\n", style="grey50")
    for ln in lines:
        out.append("║", style="grey50")
        out.append_text(ln)
        if len(ln.plain) < mw:
            out.append(" " * (mw - len(ln.plain)), style=GRID_BG)
        out.append("║\n", style="grey50")
    out.append("╚" + "═" * mw + "╝\n", style="grey50")
    if bot_label:
        out.append(" " + bot_label.center(mw) + "\n", style="dim")
    if side_note:
        out.append(side_note + "\n", style="dim italic")
    return out


def legend(noc_mode, layout, arrows):
    """One-line legend describing the current routing/layout state."""
    t = Text()
    t.append("NoC0 ", style=_rgb(NOC0_COL)); t.append("▸▾ ", style=_rgb(NOC0_COL))
    t.append("east+south   ", style="dim")
    t.append("NoC1 ", style=_rgb(NOC1_COL)); t.append("◂▴ ", style=_rgb(NOC1_COL))
    t.append("west+north\n", style="dim")
    view = ("cartoon card · GDDR6+Eth heat, CPU static" if layout in ("cartoon", "die")
            else "noc0 torus topology (folded)")
    t.append(f"view: {view}\n", style="dim")
    if arrows:
        t.append("each link = both NoC wires · brightness = live per-NoC flow (endpoint proxy)\n",
                 style="dim italic")
    return t
