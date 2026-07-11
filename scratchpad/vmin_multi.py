"""vmin_multi.py [int|fp] [ceil] — per-hart Vmin vs frequency, built on the PROVEN linear step-down (vmin_min2).
Self-resets + settles. Per freq (LOW->HIGH): voltage-lead to a frontier-estimated anchor, glide to the freq,
then step DOWN to the first hart failure = chip Vmin, that hart = the weak-die limiter. No anchor-raise loop
(that hung). Measurable where Vmin > the ~717 mV floor. Streams every probe (a hard-wedge shows as a stall)."""
import sys, time, json, os, subprocess
os.chdir("/home/starboy/bhtop"); sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
TILE, NH = 0, 4
PATS = {"int": "src/bhtop/kernels/x280/vf_margin/vf_margin.c",
        "fp":  "src/bhtop/kernels/x280/vf_margin_fp/vf_margin_fp.c"}
PAT = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in PATS else "fp"
CEIL_FW = 900
CEIL = int(sys.argv[2]) if len(sys.argv) > 2 else CEIL_FW
FREQS = [2100, 2200, 2300, 2400, 2500, 2600]
FLOOR, STEP = 712, 8
def log(*a): print(*a, flush=True)

log("reset..."); subprocess.run(["/home/starboy/.local/bin/tt-smi", "-r", "0"], capture_output=True, timeout=150)
time.sleep(18); log("settled; init")
ctx = init_ttexalens(); ctx.use_4B_mode = False
d = L2cpu(ctx=ctx)

def measV(): return d.power().get("vcore_mv") or 0
def set_freq(mhz):
    fb = max(140, min(round(140 + (mhz - 1750) / 12.5), d.EXPLORE_FBDIV_MAX)); d.set_fbdiv_explore(fb)
    return d.clocks()["core_l2cpu_mhz"][0]
def set_volt(v):
    v = min(v, CEIL); d.force_vdd(v, allow_step=True, allow_over=(v > CEIL_FW)); time.sleep(0.27); return measV()
def probe():
    a0 = d.telemetry_all(TILE); time.sleep(0.33); a1 = d.telemetry_all(TILE)
    p = [(a1[h][3] == a0[h][3] and a1[h][0] > a0[h][0]) for h in range(NH)]
    return p, [a1[h][3] - a0[h][3] for h in range(NH)]

def vmin_freq(est):
    """Step DOWN from est to first fail. Returns (chip_Vmin, limiter) or (None, None) if est itself fails (>est)."""
    prev = None; first = True
    v = est
    while v >= FLOOR:
        mv = set_volt(v)
        p, de = probe()
        fails = [h for h in range(NH) if not p[h]]
        log(f"    {mv}mV {['P' if x else 'F' for x in p]}" + (f" de={de}" if fails else ""))
        if fails:
            return (None, fails[0]) if first else (prev, fails[0])
        first = False; prev = mv; v -= STEP
    return prev, None

out = {"pattern": PAT, "ceil": CEIL, "freqs": {}}
try:
    if d.reset_state(TILE)["released"]: log("released — reset failed?"); sys.exit(1)
    log("bringup", d.bringup(TILE))
    w = tc.compile_source(PATS[PAT], base=CODE_ADDR)
    for h in range(NH): d.load(TILE, h, w, redirect=True)
    time.sleep(1.0); d.set_core_freq(1750)
    a = d.telemetry_all(TILE)
    log(f"pattern={PAT} golden {hex(a[0][1])} all-eq {len({a[h][1] for h in range(NH)})==1}\n")
    for F in FREQS:
        est = min(CEIL, round(765 + (F - 2100) * 0.41) + 38)   # frontier-estimated safe anchor
        set_volt(est)                                          # VOLTAGE LEADS FREQUENCY
        mhz = set_freq(F); time.sleep(0.4)
        log(f"  {mhz} MHz (anchor {est}mV):")
        vmin, lim = vmin_freq(est)
        tag = f"{vmin}mV limiter=hart{lim}" if vmin else f">{est}mV (>= ceil {CEIL})"
        log(f"  => {mhz} MHz chip Vmin = {tag}\n")
        out["freqs"][mhz] = {"vmin": vmin, "limiter": lim, "anchor": est}
        if vmin is None: break                                 # needs > this anchor/ceil; higher freqs worse
except Exception as e:
    import traceback; log("EXC", traceback.format_exc())

log("restoring ...")
try:
    d.set_core_freq(200)
    cur = measV()
    while cur > 720: cur = max(cur - 40, 710); d.force_vdd(cur, allow_step=True); time.sleep(0.15)
    log("idle", d.perf_idle(), measV(), "mV @", d.clocks()["core_l2cpu_mhz"][0], "MHz")
except Exception as e: log("restore issue", e)
with open("scratchpad/vmin.json", "w") as f: json.dump(out, f, indent=1)
log("wrote scratchpad/vmin.json")
