"""
NoC data-movement pattern benchmark + footprint visualization.

Reproduces (as far as host-orchestrated injection allows) the patterns from the
"Blackhole built for AI data movement patterns" slide, and renders the *NoC
footprint* of each: which Tensix routers the pattern's traffic actually flowed
through, measured live from the decoded 0x500 directional transit counters.

This is "seeing the NoC data with a benchmark": pick a pattern -> drive it ->
the mesh lights up the route the hardware chose. Aggregate BW is host-limited
(the 16 TB/s spec needs on-chip concurrency / tt-metal); the value here is the
route footprint + per-path bandwidth.
"""
from PIL import Image, ImageDraw, ImageFont

from .floorplan import KIND_RGB

# spec aggregate BW per pattern (from the slide), for context
SPEC = {"neighbor (1 hop)": "47 TB/s", "gather/scatter (3 hop)": "16 TB/s",
        "gather/scatter (10 hop)": "5 TB/s"}


def _tensix(cells, x, y):
    t = cells.get((x, y))
    return t if (t and t.kind == "tensix") else None


def pattern_pairs(name, cells, cols, rows):
    """Return [(src_tile, dst_tile)] for a named pattern (Tensix-only, safe)."""
    pairs = []
    for (x, y), s in cells.items():
        if s.kind != "tensix":
            continue
        if name == "neighbor (1 hop)":
            d = _tensix(cells, x + 1, y)
        elif name == "gather/scatter (3 hop)":
            d = _tensix(cells, x, y + 3)
        elif name == "gather/scatter (10 hop)":
            d = _tensix(cells, x + 10, y)        # crosses spine/DRAM; transit only
        else:
            d = None
        if d:
            pairs.append((s, d))
    return pairs


# ---- Pillow footprint renderer ----
CELL, GAP, MARGIN = 46, 8, 30
PITCH = CELL + GAP
HOT = (255, 70, 40)


def _font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(p, sz)
        except OSError:
            pass
    return ImageFont.load_default()


def _blend(a, b, f):
    f = max(0.0, min(1.0, f))
    return tuple(int(a[i] + (b[i] - a[i]) * f) for i in range(3))


def render_footprint(cells, cols, rows, foot, pairs, title, bw_line):
    W = MARGIN * 2 + cols * PITCH
    H = MARGIN * 2 + rows * PITCH + 70
    img = Image.new("RGB", (W, H), (16, 18, 24))
    d = ImageDraw.Draw(img)
    fg = _font(22); fs = _font(13); ft = _font(26)
    srcs = {s.noc0 for s, _ in pairs}
    dsts = {dt.noc0 for _, dt in pairs}
    mx = max(foot.values(), default=1) or 1
    d.text((MARGIN, 10), title, font=ft, fill=(235, 235, 245))
    oy = 46
    for y in range(rows):
        for x in range(cols):
            t = cells.get((x, y))
            cx, cy = MARGIN + x * PITCH, oy + y * PITCH
            if t is None:
                d.rounded_rectangle([cx, cy, cx + CELL, cy + CELL], radius=6,
                                    outline=(40, 42, 50), width=1)
                continue
            if t.kind == "tensix":
                frac = (foot.get((x, y), 0) / mx)
                fill = _blend((34, 36, 46), HOT, frac)
            else:
                fill = _blend((28, 30, 38), KIND_RGB.get(t.kind, (90, 90, 90)), 0.5)
            d.rounded_rectangle([cx, cy, cx + CELL, cy + CELL], radius=8, fill=fill,
                                outline=(70, 72, 84), width=1)
            d.text((cx + CELL/2, cy + CELL/2 - 4), t.glyph, font=fg, anchor="mm",
                   fill=(245, 245, 245) if (t.kind != "tensix" or frac > .4) else (180, 184, 196))
            if (x, y) in srcs:
                d.rounded_rectangle([cx, cy, cx + CELL, cy + CELL], radius=8,
                                    outline=(90, 170, 255), width=3)       # source = blue
            if (x, y) in dsts:
                d.ellipse([cx + CELL - 14, cy + 4, cx + CELL - 4, cy + 14],
                          fill=(120, 230, 140))                            # dest = green dot
    d.text((MARGIN, H - 20), bw_line, font=fs, fill=(170, 200, 180))
    return img


if __name__ == "__main__":
    import sys
    from ttexalens import init_ttexalens
    from .floorplan import build
    from .inject import Injector

    ctx = init_ttexalens(); fp = build(ctx)
    cells, cols, rows = fp.grid("noc0")
    inj = Injector(fp, ctx)
    names = sys.argv[1:] or list(SPEC.keys())
    print(f"{'pattern':26s} {'pairs':>6} {'moved':>9} {'time':>7} {'host BW':>10} {'spec':>8}")
    for name in names:
        pairs = pattern_pairs(name, cells, cols, rows)
        foot, total, secs = inj.run_pattern(pairs, length=0x40000, fires=2)
        bw = total / secs if secs else 0
        bwline = f"{name}: moved {total/1e6:.0f} MB, {len(pairs)} paths, host-orchestrated {bw/1e9:.2f} GB/s  (spec {SPEC.get(name,'?')})"
        img = render_footprint(cells, cols, rows, foot, pairs, name, bwline)
        out = f"/tmp/bench_{name.split()[0].replace('/','_')}_{name.split('(')[1][0] if '(' in name else ''}.png"
        img.save(out)
        print(f"{name:26s} {len(pairs):>6} {total/1e6:>7.0f}MB {secs:>6.2f}s {bw/1e9:>8.2f}GB/s {SPEC.get(name,'?'):>8}   -> {out}")
