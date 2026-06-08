"""
Optional tt-metal benchmark adapter — ZERO hard dependency on tt-metal.

bhtop and tt-metal share exactly one thing: the silicon NIU counters. This module
runs a tt-metal data_movement benchmark as a *subprocess* (tt-metal owns the device
for its run), then — after it exits — reads the per-tile NIU footprint over
tt-exalens and parses the profiler CSV for aggregate bandwidth. Pure subprocess +
file parsing; if tt-metal isn't built, `available()` is False and bhtop falls back
to its built-in injection patterns.
"""
import csv
import os
import subprocess

from . import noc_counters as nc

FREQ = 1.35e9


def metal_home():
    h = os.environ.get("TT_METAL_HOME")
    if h and os.path.isdir(h):
        return h
    d = os.path.expanduser("~/tt-metal")
    return d if os.path.isdir(d) else None


def binary():
    h = metal_home()
    if not h:
        return None
    for sub in ("build_Release", "build"):
        p = os.path.join(h, sub, "test/tt_metal/unit_tests_data_movement")
        if os.path.exists(p):
            return p
    return None


def available():
    return binary() is not None


def _env():
    h = metal_home()
    e = dict(os.environ)
    e["TT_METAL_HOME"] = h
    e["TT_METAL_RUNTIME_ROOT"] = h     # newer tt-metal requires this (else "Root Directory not set")
    e["ARCH_NAME"] = "blackhole"
    e["PYTHONPATH"] = h
    e["LD_LIBRARY_PATH"] = f"{h}/build_Release/lib:" + e.get("LD_LIBRARY_PATH", "")
    e["TT_METAL_DEVICE_PROFILER"] = "1"
    return e


def list_tests(substr="DirectedIdeal"):
    b = binary()
    if not b:
        return []
    out = subprocess.run([b, "--gtest_list_tests"], env=_env(), cwd=metal_home(),
                         capture_output=True, text=True, timeout=120).stdout
    names = []
    for line in out.splitlines():
        s = line.strip()
        if s and not s.endswith(".") and "Running" not in s and (substr in s or not substr):
            names.append(s)
    return names


def run_test(name, timeout=900):
    """Run one gtest by name. Returns (passed, stdout). Device is owned by tt-metal here."""
    b = binary()
    if not b:
        raise RuntimeError("tt-metal not found (set TT_METAL_HOME)")
    r = subprocess.run([b, f"--gtest_filter=*{name}"], env=_env(),
                       capture_output=True, text=True, timeout=timeout)
    return ("[  PASSED  ]" in r.stdout), r.stdout


def profiler_csv():
    h = metal_home()
    return os.path.join(h, "generated/profiler/.logs/profile_log_device.csv") if h else None


def aggregate_bw(csv_path=None):
    """Parse the device-profiler CSV → aggregate NoC bandwidth (bytes/s) + details."""
    csv_path = csv_path or profiler_csv()
    if not csv_path or not os.path.exists(csv_path):
        return None
    rows = [r for r in csv.reader(open(csv_path)) if r and r[0].isdigit()]
    zones, stamp = {}, {}
    for r in rows:
        cx, cy, risc = r[1], r[2], r[3]
        cyc, val, zname, typ = int(r[5]), int(r[6]), r[10], r[11]
        if typ in ("ZONE_START", "ZONE_END") and zname.startswith("RISCV"):
            zones.setdefault((cx, cy, risc, zname), {})["start" if typ == "ZONE_START" else "end"] = cyc
        elif typ == "TS_DATA":
            stamp.setdefault((cx, cy, risc), {})[zname] = val
    total = 0
    footprint = {}        # (x,y) -> bytes that core moved (the exact kernel footprint)
    for (cx, cy, risc), s in stamp.items():
        b = s.get("Per-core bytes",
                  s.get("Number of transactions", 0) * s.get("Transaction size in bytes", 0))
        total += b
        footprint[(int(cx), int(cy))] = footprint.get((int(cx), int(cy)), 0) + b
    durs = sorted(z["end"] - z["start"] for z in zones.values() if "start" in z and "end" in z)
    if not durs:
        return None
    wall = durs[-1]
    cores = len({(cx, cy) for (cx, cy, _) in stamp})
    return {"total_bytes": total, "wall_cycles": wall, "cores": cores,
            "bw": total / (wall / FREQ), "bw_median": total / (durs[len(durs)//2] / FREQ),
            "footprint": footprint}


def read_footprint_per_noc(ctx, fp):
    """After a run, read each tile's cumulative NIU traffic, split per NoC →
    {tile.noc0: {0: bytes_noc0, 1: bytes_noc1}}.

    Counters are cumulative since tt-metal's device init, so the absolute values ARE
    the kernel's footprint. Tensix/DRAM/Eth only (never the management tiles)."""
    from ttexalens.tt_exalens_lib import read_words_from_device
    SAFE = {"tensix", "dram", "eth"}        # never touch ARC/Security/PCIe/L2CPU (hangs/timeouts)
    idx = nc.TX_MASTER_OUT + nc.RX_MASTER_IN + nc.TX_SLAVE_OUT + nc.RX_SLAVE_IN
    foot = {}
    for t in fp.addressable():
        if t.kind not in SAFE:
            continue
        per = {0: 0, 1: 0}
        for noc in (0, 1):
            try:
                w = read_words_from_device(t.coord, nc.counter_base(noc), word_count=62,
                                           noc_id=noc, context=ctx, safe_mode=False)
            except Exception:
                continue
            per[noc] = sum(w[i] for i in idx) * nc.FLIT_BYTES
        foot[t.noc0] = per
    return foot


def read_footprint(ctx, fp):
    """Back-compat: summed (NoC0+NoC1) bytes per tile → {tile.noc0: bytes}."""
    return {k: v[0] + v[1] for k, v in read_footprint_per_noc(ctx, fp).items()}


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Run a tt-metal NoC benchmark and visualize its footprint in bhtop")
    ap.add_argument("test", nargs="?", help="gtest name substring, e.g. AllToAllDirectedIdeal")
    ap.add_argument("--list", action="store_true", help="list available benchmarks")
    args = ap.parse_args()

    if not available():
        print("tt-metal not found (set TT_METAL_HOME, or build it at ~/tt-metal).")
        print("bhtop works fully without it — use `bhtop` (live) or `bhtop-inject` (patterns).")
        return
    if args.list or not args.test:
        print("available data_movement benchmarks (substring-match to run one):")
        for n in list_tests(""):
            print("  ", n)
        return

    print(f"running tt-metal '{args.test}' (tt-metal owns the device for its run)...")
    passed, _ = run_test(args.test)
    print("  result:", "PASSED" if passed else "FAILED")
    bw = aggregate_bw()
    if bw:
        print(f"  aggregate: {bw['bw']/1e12:.2f} TB/s ({bw['bw']/1e9:.0f} GB/s)  "
              f"[median {bw['bw_median']/1e12:.2f}]  {bw['cores']} cores  {bw['total_bytes']/1e6:.0f} MB")

    # Per-NoC footprint from the silicon NIU counters (cumulative since tt-metal's
    # device init = this kernel's footprint). Safe-kinds only (no hang hazard).
    # Drawn on the physical die with NoC0 ▸▾ / NoC1 ◂▴ flow arrows: which network
    # carried each region of the chip, and the route it took.
    from ttexalens import init_ttexalens
    from rich.console import Console
    from .floorplan import build
    from .render import render_mesh, legend
    ctx = init_ttexalens(); fp = build(ctx)
    cells, cols, rows = fp.grid("die")
    pernoc = read_footprint_per_noc(ctx, fp)          # {tile.noc0: {0:b, 1:b}}
    tot0 = sum(v[0] for v in pernoc.values())
    tot1 = sum(v[1] for v in pernoc.values())
    mx = max((v[0] + v[1] for v in pernoc.values()), default=1) or 1

    def load(x, y, noc):
        tile = cells.get((x, y))
        return (pernoc.get(tile.noc0, {}).get(noc, 0) / mx) if tile else 0.0

    def mb(b):
        return (f"{b/1e9:.1f} GB" if b >= 1e9 else
                f"{b/1e6:.1f} MB" if b >= 1e6 else f"{b/1e3:.1f} kB")

    con = Console()
    con.print("\n[bold]NoC footprint (physical die · per-NoC bytes moved):[/]")
    con.print(render_mesh(cells, cols, rows, load=load, noc_mode=2, scale=1.0, arrows=True, dual=True))
    con.print(legend(2, "die", True))
    lit = sum(1 for v in pernoc.values() if v[0] or v[1])
    print(f"  per-NoC bytes:  NoC0 {mb(tot0)}  ·  NoC1 {mb(tot1)}  "
          f"(total {mb(tot0 + tot1)} over {lit} tiles, silicon counters)")


if __name__ == "__main__":
    main()
