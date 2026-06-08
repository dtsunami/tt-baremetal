"""
Static model of the Blackhole tile grid in BOTH coordinate systems.

Each tile carries two positions:
  * noc0  -> NoC #0 coordinates: the interleaved 2D-torus topology (routing view)
  * die   -> die coordinates: physical-ish placement, I/O on the perimeter

Tile identity (and the live-bandwidth key) is the noc0 tuple, which is unique.
The live sampler colours these; nothing here changes per frame.
"""
from dataclasses import dataclass, field

from ttexalens import init_ttexalens

# device block-type name -> (glyph, kind)
KINDS = {
    "functional_workers": ("T", "tensix"),
    "dram":               ("D", "dram"),
    "eth":                ("E", "eth"),
    "arc":                ("A", "arc"),
    "pcie":               ("P", "pcie"),
    "l2cpu":              ("C", "l2cpu"),
    "security":           ("S", "security"),
}

KIND_RGB = {
    "tensix":   (235, 110,  70),
    "dram":     (150, 205, 170),
    "eth":      (170, 175, 230),
    "l2cpu":    (190, 130, 200),
    "arc":      (200, 100,  90),
    "pcie":     (140, 140, 180),
    "security": (200, 200, 110),
}


@dataclass
class Tile:
    noc0: tuple                   # (x, y) NoC0 topology coords  -- identity key
    die: tuple                    # (x, y) physical-ish die coords
    kind: str
    glyph: str
    label: str
    coord: object                 # OnChipCoordinate for live reads
    dram_ctrl: int | None = None

    @property
    def key(self):
        return self.noc0


@dataclass
class Floorplan:
    placed: list                  # list[Tile] (all are addressable)
    dram_ctrl: dict = field(default_factory=dict)   # ctrl_id -> [Tile, ...]

    def grid(self, system: str):
        """Return (cells:{(x,y)->Tile}, cols, rows) in 'noc0' or 'die' space."""
        cells = {getattr(t, system): t for t in self.placed}
        cols = max(x for x, _ in cells) + 1
        rows = max(y for _, y in cells) + 1
        return cells, cols, rows

    def addressable(self):
        return self.placed


def build(ctx=None) -> Floorplan:
    ctx = ctx or init_ttexalens()
    dev = ctx.devices[0]
    placed = []
    dram_ctrl = {}

    for bt_name, (glyph, kind) in KINDS.items():
        for loc in dev.get_block_locations(bt_name):
            try:
                n0 = loc.to("noc0")
                die = loc.to("die")
            except Exception:
                continue
            label = str(loc)
            if label == "N/A":
                label = f"{glyph}{n0[0]},{n0[1]}"
            ctrl = None
            if kind == "dram":
                ctrl = int(str(loc)[1:].split(",")[0])     # 'd3,1' -> 3
            t = Tile(noc0=n0, die=die, kind=kind, glyph=glyph,
                     label=label, coord=loc, dram_ctrl=ctrl)
            placed.append(t)
            if ctrl is not None:
                dram_ctrl.setdefault(ctrl, []).append(t)

    return Floorplan(placed=placed, dram_ctrl=dram_ctrl)


if __name__ == "__main__":
    fp = build()
    print(f"placed tiles: {len(fp.placed)}")
    for system in ("die", "noc0"):
        cells, cols, rows = fp.grid(system)
        print(f"\n=== {system} grid  ({cols}x{rows}) ===")
        for y in range(rows):
            print("  " + "".join(cells[(x, y)].glyph if (x, y) in cells else "·"
                                  for x in range(cols)))
    print("\nDRAM controllers:", {f"d{c}": [t.label for t in ts]
                                   for c, ts in sorted(fp.dram_ctrl.items())})
