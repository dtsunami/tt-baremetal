"""vmin_run.py [int|fp] [ceil] — per-hart Vmin search built on the PROVEN vmin_min structure (explicit prints,
no hang-prone finally). ONE pattern per fresh reset. For each freq (LOW->HIGH), approach the fail edge FROM
ABOVE: anchor at a passing voltage (warm-started from the previous freq's Vmin, raised toward ceil if needed),
then step DOWN to the first hart failure = chip Vmin, and that hart = the weak-die limiter. Measurable only
where Vmin > the ~717 mV vdd_min floor (i.e. >~2000 MHz). Requires a fresh tt-smi -r 0 (harts in reset)."""
import sys, time, json, os
os.chdir("/home/starboy/bhtop")                        # relative kernel/output paths resolve here
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
TILE, NH = 0, 4
PATS = {"int": "src/bhtop/kernels/x280/vf_margin/vf_margin.c",
        "fp":  "src/bhtop/kernels/x280/vf_margin_fp/vf_margin_fp.c"}
PAT = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in PATS else "fp"
CEIL_FW = 900
CEIL = int(sys.argv[2]) if len(sys.argv) > 2 else CEIL_FW
FREQS = [2100, 2200, 2300, 2400, 2500]
VLO, COARSE, STEP_DN, DWELL = 712, 12, 8, 0.35
def log(*a): print(*a, flush=True)

import subprocess                                       # self-reset + SETTLE before init (avoids post-reset race)
log("resetting card ...")
subprocess.run(["/home/starboy/.local/bin/tt-smi", "-r", "0"], capture_output=True, timeout=150)
time.sleep(18)                                          # let PCIe/NoC fully re-enumerate before init_ttexalens
log("reset settled; init ...")

ctx = init_ttexalens(); ctx.use_4B_mode = False
d = L2cpu(ctx=ctx)

def measV(): return d.power().get("vcore_mv") or 0
def set_freq(mhz):
    fb = max(140, min(round(140 + (mhz - 1750) / 12.5), d.EXPLORE_FBDIV_MAX)); d.set_fbdiv_explore(fb)
    return d.clocks()["core_l2cpu_mhz"][0]
def set_volt(v):
    v = min(v, CEIL); d.force_vdd(v, allow_step=True, allow_over=(v > CEIL_FW)); time.sleep(0.27); return measV()
def probe():
    a0 = d.telemetry_all(TILE); time.sleep(DWELL); a1 = d.telemetry_all(TILE)
    return {h: {"p": a1[h][3] == a0[h][3] and a1[h][0] > a0[h][0], "alive": a1[h][0] > a0[h][0]} for h in range(NH)}
def allpass(r): return all(r[h]["p"] for h in range(NH))

def vmin_chip(start_V):
    v = min(start_V, CEIL); mv = set_volt(v); r = probe()
    while not allpass(r) and v < CEIL:                 # raise (from above) to find a passing anchor
        v = min(v + COARSE, CEIL); mv = set_volt(v); r = probe()
    if not allpass(r): return None, None, False        # can't pass at ceil -> Vmin > ceil
    prev = mv
    while prev > VLO:                                  # step DOWN to first fail
        mv = set_volt(prev - STEP_DN); r = probe()
        fails = [h for h in range(NH) if not r[h]["p"]]
        if fails: return prev, fails[0], (not r[fails[0]]["alive"])
        prev = mv
    return prev, None, False                           # all-pass down to the floor

out = {"pattern": PAT, "ceil": CEIL, "freqs": {}}
err = None
try:
    rs = d.reset_state(TILE); log("reset_state", rs)
    if rs["released"]: log("harts released — run tt-smi -r 0 first"); sys.exit(0)
    log("bringup", d.bringup(TILE))
    w = tc.compile_source(PATS[PAT], base=CODE_ADDR)
    for h in range(NH): d.load(TILE, h, w, redirect=True)
    time.sleep(1.0); d.set_core_freq(1750)
    a = d.telemetry_all(TILE)
    log(f"pattern={PAT} golden {hex(a[0][1])} all-eq {len({a[h][1] for h in range(NH)})==1}\n")
    for F in FREQS:
        est = min(CEIL, round(765 + (F - 2100) * 0.42) + 45)   # frontier-estimated safe anchor for THIS freq
        set_volt(est)                                          # VOLTAGE LEADS FREQUENCY (glide undervolts otherwise)
        mhz = set_freq(F); time.sleep(0.4)
        vmin, lim, wedge = vmin_chip(est)
        log(f"  {mhz:>5} MHz  chip Vmin = " + (f"{vmin} mV  limiter=hart{lim}" if vmin else f">{CEIL} mV")
            + ("  [WEDGE]" if wedge else ""))
        out["freqs"][mhz] = {"vmin": vmin, "limiter": lim}
        if wedge: log("  wedge -> stop (needs reset)"); break
        if vmin is None: break
except Exception as e:
    import traceback; err = traceback.format_exc()
    log("EXC:", err)

log("\nrestoring ...")
try:
    d.set_core_freq(200)
    cur = measV()
    while cur > 720: cur = max(cur - 40, 710); d.force_vdd(cur, allow_step=True); time.sleep(0.15)
    log("idle", d.perf_idle(), measV(), "mV @", d.clocks()["core_l2cpu_mhz"][0], "MHz")
except Exception as e:
    log("restore issue:", e)
with open("scratchpad/vmin.json", "w") as f: json.dump(out, f, indent=1)
log("wrote scratchpad/vmin.json")
