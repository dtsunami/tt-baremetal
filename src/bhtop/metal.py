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
import re
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


def _env(dprint_cores=None):
    h = metal_home()
    e = dict(os.environ)
    e["TT_METAL_HOME"] = h
    e["TT_METAL_RUNTIME_ROOT"] = h     # newer tt-metal requires this (else "Root Directory not set")
    e["ARCH_NAME"] = "blackhole"
    e["PYTHONPATH"] = h
    e["LD_LIBRARY_PATH"] = f"{h}/build_Release/lib:" + e.get("LD_LIBRARY_PATH", "")
    e["TT_METAL_DEVICE_PROFILER"] = "1"
    e["TT_METAL_INSPECTOR"] = "1"       # dump generated/inspector/*.yaml: kernel hash <-> source <-> program
    e.setdefault("TT_METAL_INSPECTOR_LOG_PATH", os.path.join(h, "generated", "inspector"))
    if dprint_cores:
        e["TT_METAL_DPRINT_CORES"] = dprint_cores   # on-device printf from kernels -> stdout
    return e


# DPRINT lines look like "0:(x=1,y=2):BR: my message" (device:core:risc: text).
_DPRINT_RE = re.compile(r"^\d+:\(x=\d+,y=\d+\):[A-Z0-9]+:")


def extract_dprint(stdout):
    """Pull the on-device DPRINT lines out of a run's stdout."""
    return [ln for ln in stdout.splitlines() if _DPRINT_RE.match(ln.strip())]


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


def run_test(name, timeout=900, dprint_cores=None):
    """Run one gtest by name. Returns (passed, stdout). Device is owned by tt-metal here.
    Pass dprint_cores (e.g. "0,0") to capture on-device DPRINT output in stdout."""
    b = binary()
    if not b:
        raise RuntimeError("tt-metal not found (set TT_METAL_HOME)")
    r = subprocess.run([b, f"--gtest_filter=*{name}"], env=_env(dprint_cores),
                       capture_output=True, text=True, timeout=timeout)
    return ("[  PASSED  ]" in r.stdout), r.stdout


def profiler_csv():
    h = metal_home()
    return os.path.join(h, "generated/profiler/.logs/profile_log_device.csv") if h else None


# ---- Tensix COMPUTE lab (tlab): run a standalone programming_example + read its per-engine
# compute zones. These are SEPARATE binaries (not the data_movement gtest), and the compute
# zones are named "<PROC>-KERNEL" on processors TRISC_0/1/2 (UNPACK/MATH/PACK) — which
# aggregate_bw drops (it filters zone-name startswith "RISCV"). ----
def examples_dir():
    h = metal_home()
    if not h:
        return None
    for sub in ("build_Release", "build"):
        d = os.path.join(h, sub, "programming_examples")
        if os.path.isdir(d):
            return d
    return None


def compute_examples():
    """Prebuilt standalone compute programming_examples (the Tensix UNPACK/MATH/PACK ones)."""
    d = examples_dir()
    if not d:
        return []
    keep = ("compute", "matmul", "eltwise", "sfpu", "add_2_integers_in_compute")
    return sorted(n for n in os.listdir(d)
                  if os.path.isfile(os.path.join(d, n)) and os.access(os.path.join(d, n), os.X_OK)
                  and any(k in n for k in keep))


def run_example(name, timeout=900):
    """Run a standalone compute programming_example by name (e.g.
    'metal_example_add_2_integers_in_compute'). tt-metal owns + resets the device for the run
    (so the L2CPU/x280 harts go back to reset). Returns (passed, combined stdout+stderr)."""
    d = examples_dir()
    p = os.path.join(d, os.path.basename(name)) if d else None
    if not p or not os.path.exists(p):
        raise RuntimeError(f"compute example not found: {name}")
    # fresh CSV so we only see this run's zones
    c = profiler_csv()
    if c and os.path.exists(c):
        try:
            os.remove(c)
        except OSError:
            pass
    r = subprocess.run([p], env=_env(), cwd=metal_home(), capture_output=True, text=True, timeout=timeout)
    return (r.returncode == 0), (r.stdout or "") + (r.stderr or "")


# Tensix RISC processor (profiler col 3) -> role. TRISC_0/1/2 = the compute triad.
_TRISC_ROLE = {"NCRISC": "reader", "BRISC": "writer",
               "TRISC_0": "unpack", "TRISC_1": "math", "TRISC_2": "pack"}


def aggregate_compute(csv_path=None):
    """Per-Tensix-core busy cycles for each engine (reader/writer DM + UNPACK/MATH/PACK) from
    the `<PROC>-KERNEL` profiler zones, plus a MATH-vs-wall occupancy. The honest Tier-1
    compute metric (coarse per-engine spans); true math-utilization needs instrumented kernels."""
    csv_path = csv_path or profiler_csv()
    if not csv_path or not os.path.exists(csv_path):
        return None
    zones = {}                                          # (cx,cy,proc) -> {start,end}
    for r in csv.reader(open(csv_path)):
        if not r or not r[0].isdigit() or len(r) < 12:
            continue
        cx, cy, proc, cyc, zname, typ = r[1], r[2], r[3], int(r[5]), r[10], r[11]
        if zname.endswith("-KERNEL") and typ in ("ZONE_START", "ZONE_END"):
            zones.setdefault((cx, cy, proc), {})["start" if typ == "ZONE_START" else "end"] = cyc
    cores = {}                                          # "x,y" -> {role: cycles}
    for (cx, cy, proc), s in zones.items():
        if "start" in s and "end" in s:
            cores.setdefault(f"{cx},{cy}", {})[_TRISC_ROLE.get(proc, proc.lower())] = s["end"] - s["start"]
    if not cores:
        return None
    out = {}
    for core, eng in cores.items():
        wall = max(eng.values()) if eng else 0          # per-core wall = slowest engine
        out[core] = {"engines": eng, "wall": wall,
                     "math_occ": round(eng.get("math", 0) / wall, 4) if wall else 0.0}
    n = len(out)
    return {"cores": out, "n_cores": n, "freq": FREQ,
            "avg_math_occ": round(sum(c["math_occ"] for c in out.values()) / n, 4) if n else 0.0}


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
