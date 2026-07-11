"""vmin_search.py [ceil_mv] — production Vmin search, all 4 x280 harts, multiple patterns.
Per pattern (self-check virus) and frequency (LOW->HIGH), find the chip Vmin: warm-start at a PASSING voltage
and step DOWN to the first hart failure. That first-failing hart is the weakest micro (the limiter); its Vmin is
the chip Vmin. Approaching from above never deep-undervolts past the weak hart, so it doesn't wedge it (a shared
rail means undervolting one undervolts all). Robustness:
  * every probe uses MEASURED vcore (readback) — the vdd_max clamp can't hose the result;
  * pass = zero NEW checksum errors AND heartbeat advancing over a dwell (soft-fail and wedge both read as fail);
  * if even the ceiling fails, Vmin = >ceil (search not corrupted), and higher freqs are skipped.
Warm-start across frequency (Vmin moves ~monotonically) is the main test-time saver. Default ceiling 900 mV
(firmware vdd_max); pass a higher ceil once the clamp is lifted. Card must be idle."""
import sys, time, json
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR

TILE, NH = 0, 4
ALL_PATTERNS = {"int": "src/bhtop/kernels/x280/vf_margin/vf_margin.c",
                "fp":  "src/bhtop/kernels/x280/vf_margin_fp/vf_margin_fp.c"}
PAT = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ALL_PATTERNS else "fp"
PATTERNS = {PAT: ALL_PATTERNS[PAT]}                # ONE pattern per reset (clean-reset protocol)
FREQS  = [1900, 2100, 2300, 2450]                  # low -> high
VLO, CEIL_FW = 705, 900
CEIL   = int(sys.argv[2]) if len(sys.argv) > 2 else CEIL_FW
COARSE, STEP_DN, DWELL = 15, 8, 0.35

ctx = init_ttexalens(); ctx.use_4B_mode = False
d = L2cpu(ctx=ctx)
out = {"ceil_mv": CEIL, "patterns": {}}

def measV(): return d.power().get("vcore_mv") or 0
def set_freq(mhz):
    fb = max(140, min(round(140 + (mhz - 1750) / 12.5), d.EXPLORE_FBDIV_MAX))
    return d.set_fbdiv_explore(fb)["core_mhz"][0]
def set_volt(v):
    v = min(v, CEIL)
    d.force_vdd(v, allow_step=True, allow_over=(v > CEIL_FW)); time.sleep(0.26)
    return measV()
def probe():                                        # {h:{pass,alive}} or None if tile wedged
    try: a0 = d.telemetry_all(TILE)
    except Exception: return None
    time.sleep(DWELL)
    try: a1 = d.telemetry_all(TILE)
    except Exception: return None
    return {h: {"pass": a1[h][3] == a0[h][3] and a1[h][0] > a0[h][0], "alive": a1[h][0] > a0[h][0]} for h in range(NH)}
def allpass(r): return r is not None and all(r[h]["pass"] for h in range(NH))

def load_pattern(kpath):
    d.set_core_freq(200); time.sleep(0.3)           # golden capture at 200 MHz — reliable, all-harts-equal
    w = tc.compile_source(kpath, base=CODE_ADDR)
    for h in range(NH): d.load(TILE, h, w, redirect=True)
    time.sleep(0.9); a = d.telemetry_all(TILE)
    d.set_core_freq(1750)                           # restore postdiv=1 so fbdiv-overclock is valid
    return a[0][1], len({a[h][1] for h in range(NH)}) == 1

def vmin_chip(start_V):
    """Warm-start at a passing V, step DOWN to first fail. Returns (chip_Vmin, limiter_hart, wedged)."""
    v = min(start_V, CEIL); mv = set_volt(v); r = probe()
    while not allpass(r) and v < CEIL:
        v = min(v + COARSE, CEIL); mv = set_volt(v); r = probe()
    if not allpass(r): return None, None, False     # can't pass even at ceil -> Vmin > ceil
    prev = mv
    while prev > VLO:
        mv = set_volt(prev - STEP_DN); r = probe()
        if r is None: return prev, "tile", True
        fails = [h for h in range(NH) if not r[h]["pass"]]
        if fails:
            return prev, fails[0], (not r[fails[0]]["alive"])   # chip Vmin = last all-pass voltage
        prev = mv
    return prev, None, False                         # all-pass down to VLO

try:
    rs = d.reset_state(TILE)
    # This test WEDGES harts at the fail edge, and a wedged hart can't be re-loaded -> demand a CLEAN reset
    # (harts in reset) so every run starts uncontaminated. force_vdd works without perf_busy, so we load at idle.
    if rs["wedged"] or rs["released"]:
        sys.exit("run `tt-smi -r 0` first — harts must be in reset for a clean, uncontaminated run")
    print("bringup:", d.bringup(TILE))
    for pname, kpath in PATTERNS.items():
        gold, eq = load_pattern(kpath)
        print(f"\n#### pattern={pname}  golden {hex(gold)} all-equal={eq}")
        out["patterns"][pname] = {}
        start_V = 820
        for F in FREQS:
            mhz = set_freq(F); time.sleep(0.4)
            vmin, lim, wedge = vmin_chip(start_V)
            tag = f"{vmin}mV  limiter=hart{lim}" if vmin else f">{CEIL}mV"
            print(f"  {mhz:>5} MHz  chip Vmin = {tag}" + ("  [wedge]" if wedge else ""))
            out["patterns"][pname][mhz] = {"vmin": vmin, "limiter": lim}
            if wedge:
                print("  wedge — a hart is stuck; this campaign is done (needs tt-smi -r 0 for more)"); break
            if vmin is None: print(f"  (needs >{CEIL} mV — stop this pattern)"); break
            start_V = min(vmin + 26, CEIL)
        d.set_fbdiv_explore(140)
finally:
    with open("scratchpad/vmin.json", "w") as f: json.dump(out, f, indent=1)
    print("\nwrote scratchpad/vmin.json")
    try:
        d.set_core_freq(200); cur = measV()
        while cur > 715: cur = max(cur - 40, 710); d.force_vdd(cur, allow_step=True); time.sleep(0.2)
        print("restored:", d.perf_idle(), "vcore", measV(), "mV @", d.clocks()["core_l2cpu_mhz"], "MHz")
    except Exception as e: print("restore issue:", e)
