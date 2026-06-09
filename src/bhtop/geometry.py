"""
Physical vs logical geometry for the Blackhole NoC.

The NoC is a 2D *torus* addressed in `noc0` coordinates, but the tiles are placed
on the die in a FOLDED/interleaved order so that the torus wraparound wires stay
physically short. Every tile therefore carries two coordinates (from silicon, via
tt-exalens): `die` = physical placement, `noc0` = logical torus index.

Verified on this p150a silicon (firmware tables DIE_X_TO_NOC_0_X /
DIE_Y_TO_NOC_0_Y; reproduced by floorplan.build()):

  * The fold is a column- and row-uniform odd/even interleave:
      die col 0 -> noc0 0 ; odd die cols ascend into noc0 1..8 ;
      even die cols fold into noc0 16..9 (descending).  Same for rows.
  * NoC0 "+x" is therefore TWO monotone ramps (step ~2) with ONE sign flip at the
    fold apex (noc0 col 8/9) — a "fold ramp", not a per-step zig-zag.
  * Torus wraparound links are physically SHORT (die-distance 2) — that is what the
    fold buys. The physically-long jumps are across the fold *seam*: die-adjacent
    tiles can be up to 8 noc0 hops apart (a gradient that peaks at the apex — NOT a
    uniform seam). At the four ramp-end bands logical and physical coincide, so a few
    touching tiles are only 1 hop; the seam-crossing range is ~2..8.
  * Interior noc0 1-hop links are physically die-distance 2 (dist-1 occurs only at
    the four ramp-end col/row bands). So "1 logical hop" under-reports physical
    distance by up to ~2x.
  * Nearest-DRAM *controller* is invariant physical-vs-logical (hop count picks the
    right bank); only the distance *scale* differs.

Integer counts here are computed live from whatever tiles `build()` enumerates —
they are specific to this harvested card, not grid constants.
"""
from collections import Counter


# ---- coordinate transforms -------------------------------------------------
# The card mounts the die rotated 90°: die DRAM (cols 0 & 16, the two vertical die
# edges) ends up on the card's TOP & BOTTOM, with Ethernet on the left. That is a
# display rotation of the silicon `die` coords; only "DRAM on opposite die edges"
# is a pure silicon fact — which physical card edge is which comes from the board
# photo (GDDR6 top/bottom, QSFP/Eth left, power/PCIe right).
def rotate(xy, rot, w, h):
    """Rotate (x,y) in a w×h grid by rot∈{0,90,180,270} (CCW). Returns (x',y',w',h')."""
    x, y = xy
    if rot == 0:
        return x, y, w, h
    if rot == 90:            # CCW: die col 0 -> bottom, col max -> top; row 1 -> left
        return y, (w - 1 - x), h, w
    if rot == 180:
        return (w - 1 - x), (h - 1 - y), w, h
    if rot == 270:           # CW
        return (h - 1 - y), x, h, w
    raise ValueError(rot)


ORIENTS = {
    0:   "die-geographic (DRAM left/right · matches TT floorplan)",
    90:  "card (DRAM top/bottom · Eth left · power/PCIe right)",
    180: "die 180°",
    270: "card 90°cw",
}


# ---- distances -------------------------------------------------------------
def die_manhattan(a, b):
    """Physical distance proxy: Manhattan distance in die cells (~ wire length)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def torus_hops(a, b, cols, rows):
    """Logical NoC hop count: toroidal Manhattan in noc0 coords (wraparound)."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return min(dx, cols - dx) + min(dy, rows - dy)


def grid_dims(fp, system="noc0"):
    cells = {getattr(t, system): t for t in fp.placed}
    cols = max(x for x, _ in cells) + 1
    rows = max(y for _, y in cells) + 1
    return cols, rows


# ---- DRAM affinity: physical vs logical ------------------------------------
def dram_affinity(fp, tile):
    """For a tile, the nearest GDDR6 controller physically (die cells) vs logically
    (noc0 hops). Returns dict; `agree` is whether they pick the same controller.

    The teaching point: `agree` is almost always True (hop count picks the right
    bank) but `phys_cells` >> `logi_hops` — distance scale differs ~2x."""
    cols, rows = grid_dims(fp, "noc0")
    drams = [t for t in fp.placed if t.kind == "dram"]
    if not drams:
        return None
    pd, pc = min((die_manhattan(tile.die, d.die), d.dram_ctrl) for d in drams)
    ld, lc = min((torus_hops(tile.noc0, d.noc0, cols, rows), d.dram_ctrl) for d in drams)
    return {"phys_ctrl": pc, "phys_cells": pd,
            "logi_ctrl": lc, "logi_hops": ld,
            "agree": pc == lc}


def neighbor_stretch(fp, tile):
    """For each of a tile's 4 noc0-grid neighbours (1 logical hop): the physical
    die-distance it actually spans (≈2 in the interior — the '1 hop ≈ 2 cells' scale)."""
    cols, rows = grid_dims(fp, "noc0")
    by_noc0 = {t.noc0: t for t in fp.placed}
    x, y = tile.noc0
    out = []
    for dx, dy, name in ((1, 0, "E"), (-1, 0, "W"), (0, 1, "S"), (0, -1, "N")):
        nb = by_noc0.get(((x + dx) % cols, (y + dy) % rows))
        if nb:
            out.append((name, nb, die_manhattan(tile.die, nb.die)))
    return out


def physical_neighbor_hops(fp, tile):
    """The fold seam, per tile: for each PHYSICALLY-touching die neighbour, the noc0
    hop distance. A die neighbour that is many noc0 hops away is a fold-seam crossing
    (the genuine physical≠logical conflict — touching tiles routed the long way)."""
    cols, rows = grid_dims(fp, "noc0")
    by_die = {t.die: t for t in fp.placed}
    x, y = tile.die
    out = []
    for dx, dy, name in ((1, 0, "E"), (-1, 0, "W"), (0, 1, "S"), (0, -1, "N")):
        nb = by_die.get((x + dx, y + dy))
        if nb:
            out.append((name, nb, torus_hops(tile.noc0, nb.noc0, cols, rows)))
    return out


# ---- link distance histogram (live, this-card-specific) --------------------
def link_distance_stats(fp):
    """Histogram of physical die-distance over all noc0 grid-adjacent links, split
    into interior vs torus-wraparound. Computed live — counts are specific to the
    tiles this harvested card actually exposes, not universal constants."""
    cols, rows = grid_dims(fp, "noc0")
    by_noc0 = {t.noc0: t for t in fp.placed}
    interior, wrap = Counter(), Counter()
    seen = set()
    for t in fp.placed:
        x, y = t.noc0
        for dx, dy in ((1, 0), (0, 1)):
            nx, ny = (x + dx) % cols, (y + dy) % rows
            nb = by_noc0.get((nx, ny))
            if not nb:
                continue
            key = (t.noc0, nb.noc0)
            if key in seen:
                continue
            seen.add(key)
            is_wrap = (x + dx >= cols) or (y + dy >= rows)
            d = die_manhattan(t.die, nb.die)
            (wrap if is_wrap else interior)[d] += 1
    return {"interior": dict(sorted(interior.items())),
            "wrap": dict(sorted(wrap.items())),
            "max_interior": max(interior or [0])}


def link_die_distance(fp):
    """{frozenset(noc0_a, noc0_b) -> die_distance} for every noc0 adjacency — the
    PHYSICAL length of each LOGICAL link (for tinting links in the topology view)."""
    cols, rows = grid_dims(fp, "noc0")
    by_noc0 = {t.noc0: t for t in fp.placed}
    out = {}
    for t in fp.placed:
        x, y = t.noc0
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = by_noc0.get(((x + dx) % cols, (y + dy) % rows))
            if nb:
                out[frozenset((t.noc0, nb.noc0))] = die_manhattan(t.die, nb.die)
    return out


def physical_link_hops(fp):
    """{frozenset(noc0_a, noc0_b) -> noc0_hops} for every die-adjacent pair — the
    LOGICAL length of each PHYSICAL link (for tinting links in the physical view).
    High values are fold-seam crossings: physically touching, routed the long way."""
    cols, rows = grid_dims(fp, "noc0")
    by_die = {t.die: t for t in fp.placed}
    out = {}
    for t in fp.placed:
        x, y = t.die
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = by_die.get((x + dx, y + dy))
            if nb:
                out[frozenset((t.noc0, nb.noc0))] = torus_hops(t.noc0, nb.noc0, cols, rows)
    return out


# ---- fold-ramp trace -------------------------------------------------------
# ---- card-photo overlay registration ---------------------------------------
# Pixel bounding box of the BLACKHOLE die-package lid in blackhole_card.png
# (image is 2263x961). Measured by eye from the board photo — re-tune HERE if the
# live overlay visibly drifts off the package. (x0, y0, x1, y1).
CARD_IMAGE = "blackhole_card.png"
CARD_IMAGE_PX = (2263, 961)
#CARD_PACKAGE_PX = (1034, 281, 1399, 646)
CARD_PACKAGE_PX = (1046, 271, 1411, 636)
CARD_ORIENT = 90   # die mounted rotated 90° on the card (DRAM top/bottom, Eth left)


def card_overlay(fp, package_px=CARD_PACKAGE_PX, orient=CARD_ORIENT):
    """Map each tile's die coord -> a pixel rect inside the package lid, in card
    orientation. Returns {noc0_key: {x, y, w, h}} (pixels in the card image).

    The tile grid is NOT visible under the metal lid, so this registers the grid
    onto the package *footprint* via an affine from die cells to the lid box — not a
    per-tile photo feature. Honest approximation; the value is physical context
    (cooler / GDDR6 / PCIe around a live, glowing die)."""
    x0, y0, x1, y1 = package_px
    cols, rows = grid_dims(fp, "die")
    _, _, w, h = rotate((0, 0), orient, cols, rows)   # rotated grid dims
    cw, ch = (x1 - x0) / w, (y1 - y0) / h
    out = {}
    for t in fp.placed:
        rx, ry, _, _ = rotate(t.die, orient, cols, rows)
        out[t.noc0] = {"x": round(x0 + rx * cw, 1), "y": round(y0 + ry * ch, 1),
                       "w": round(cw, 1), "h": round(ch, 1)}
    return out


def fold_ramp(fp, noc_row):
    """Walk noc0 cols 0..max along one noc0 row; return the die-x sequence and the
    per-step die displacement. Shows NoC0 +x as two monotone ramps meeting at the
    fold apex (the 'fold ramp')."""
    cols, _ = grid_dims(fp, "noc0")
    by_noc0 = {t.noc0: t for t in fp.placed}
    path = []
    for x in range(cols):
        t = by_noc0.get((x, noc_row))
        if t:
            path.append((x, t.die))
    steps = [path[i + 1][1][0] - path[i][1][0] for i in range(len(path) - 1)]
    return path, steps
