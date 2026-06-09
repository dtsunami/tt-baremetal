"""
Data-movement traffic patterns for injection — pure (src, dst) pair builders.

Shared by the host-injection TUI (`inject_app`) and the web backend (`web.device`).
No tt-exalens, no Textual: just floorplan tiles + noc0-grid math. Each builder takes
the anchor (center) tile and the noc0 cell map and returns a list of (src, dst) tile
pairs. `gddr6_write` is special — it's filled at fire time from the injector's probed
DRAM write-access tiles (needs the device), so its builder returns [].
"""


def ring(cells, center, lo, hi):
    """Tensix tiles whose Manhattan distance from `center` (noc0 xy) is in [lo, hi]."""
    cx, cy = center
    out = []
    for dx in range(-hi, hi + 1):
        for dy in range(-hi, hi + 1):
            d = abs(dx) + abs(dy)
            if lo <= d <= hi:
                t = cells.get((cx + dx, cy + dy))
                if t and t.kind == "tensix":
                    out.append(t)
    return out


def point(cells, center_tile, off):
    """Single pair: center -> the tensix `off` away in noc0 space (or [] if none)."""
    d = cells.get((center_tile.noc0[0] + off[0], center_tile.noc0[1] + off[1]))
    return [(center_tile, d)] if (d and d.kind == "tensix") else []


# id -> builder(center_tile, cells) -> [(src, dst)]
BUILDERS = {
    "gather_3hop":   lambda c, cl: [(n, c) for n in ring(cl, c.noc0, 1, 3)],
    "scatter_3hop":  lambda c, cl: [(c, n) for n in ring(cl, c.noc0, 1, 3)],
    "neighbor_halo": lambda c, cl: [(c, n) for n in ring(cl, c.noc0, 1, 1)],
    "scatter_far":   lambda c, cl: point(cl, c, (10, 0)),
    "gddr6_write":   lambda c, cl: [],   # special — filled from injector.dram_access_tiles()
}

# display metadata for the UI (id, human label, rgb)
PATTERN_INFO = [
    {"id": "gddr6_write",   "label": "→ GDDR6 write", "rgb": [150, 205, 170]},
    {"id": "gather_3hop",   "label": "gather · 3hop",  "rgb": [120, 230, 140]},
    {"id": "scatter_3hop",  "label": "scatter · 3hop", "rgb": [255, 120, 60]},
    {"id": "neighbor_halo", "label": "neighbor · halo", "rgb": [120, 160, 255]},
    {"id": "scatter_far",   "label": "scatter · far (10)", "rgb": [240, 90, 90]},
]
