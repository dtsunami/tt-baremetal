"""vmin_1v.py — extend the Vmin curve into the unlocked 1.0 V band + find the true frequency ceiling at 1000 mV.
Part A: Vmin at 2500/2600/2700 MHz (voltage-lead, step-down, hart3 limiter). Part B: pin 1000 mV and push the
PLL up (fbdiv) until the datapath fails or the PLL unlocks -> the real max freq at 1 V (answers 'how close to 3G').
Safety: per-probe temp/current/throttler abort; firmware tdc/tdp/thm clamps still active. Self-resets + settles."""
import sys, time, json, os, subprocess
os.chdir("/home/starboy/bhtop"); sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
TILE, NH = 0, 4
KP = "src/bhtop/kernels/x280/vf_margin_fp/vf_margin_fp.c"
CEIL = 1000
FREQS = [2500, 2600, 2700]
FLOOR, STEP, TMAX = 900, 8, 87
def log(*a): print(*a, flush=True)

log("reset..."); subprocess.run(["/home/starboy/.local/bin/tt-smi", "-r", "0"], capture_output=True, timeout=150)
time.sleep(18); log("settled; init")
ctx = init_ttexalens(); ctx.use_4B_mode = False
d = L2cpu(ctx=ctx)
d.EXPLORE_FBDIV_MAX = 248                                    # allow the ceiling probe to reach ~3050 MHz

def measV(): return d.power().get("vcore_mv") or 0
def fbfor(mhz): return max(140, min(round(140 + (mhz - 1750) / 12.5), d.EXPLORE_FBDIV_MAX))
def set_freq(mhz): d.set_fbdiv_explore(fbfor(mhz)); return d.clocks()["core_l2cpu_mhz"][0]
def set_volt(v):
    v = min(v, CEIL); d.force_vdd(v, allow_step=True, allow_over=(v > 900)); time.sleep(0.27); return measV()
def safety():
    p = d.power(); return p.get("asic_temp_c"), p.get("current_a"), p.get("power_w"), p.get("throttler")
def unsafe(t, thr): return (isinstance(t, (int, float)) and t > TMAX) or (thr not in (0, None))
def probe():
    a0 = d.telemetry_all(TILE); time.sleep(0.33); a1 = d.telemetry_all(TILE)
    return [(a1[h][3] == a0[h][3] and a1[h][0] > a0[h][0]) for h in range(NH)], [a1[h][3]-a0[h][3] for h in range(NH)]

def vmin_freq(est):
    prev = None; first = True; v = est
    while v >= FLOOR:
        mv = set_volt(v); t, c, w, thr = safety()
        if unsafe(t, thr): log(f"    SAFETY abort @{mv}mV {t}C thr={thr}"); return prev, None, True
        p, de = probe(); fails = [h for h in range(NH) if not p[h]]
        log(f"    {mv}mV {['P' if x else 'F' for x in p]} {t}C {c}A" + (f" de={de}" if fails else ""))
        if fails: return (None, fails[0], False) if first else (prev, fails[0], False)
        first = False; prev = mv; v -= STEP
    return prev, None, False

out = {"ceil": CEIL, "vmin": {}, "ceiling_1v": None}
try:
    if d.reset_state(TILE)["released"]: log("released — abort"); sys.exit(1)
    log("bringup", d.bringup(TILE))
    w = tc.compile_source(KP, base=CODE_ADDR)
    for h in range(NH): d.load(TILE, h, w, redirect=True)
    time.sleep(1.0); d.set_core_freq(1750)
    a = d.telemetry_all(TILE); log(f"golden {hex(a[0][1])} all-eq {len({a[h][1] for h in range(NH)})==1}\n")

    # ---- Part A: Vmin at 2500/2600/2700 ----
    for F in FREQS:
        est = min(CEIL, round(765 + (F - 2100) * 0.37) + 40)
        set_volt(est); mhz = set_freq(F); time.sleep(0.4)
        log(f"  {mhz} MHz (anchor {est}mV):")
        vmin, lim, ab = vmin_freq(est)
        log(f"  => {mhz} MHz Vmin = " + (f"{vmin}mV hart{lim}" if vmin else f">{est}mV") + (" [SAFETY]" if ab else "") + "\n")
        out["vmin"][F] = {"vmin": vmin, "limiter": lim}

    # ---- Part B: max frequency at a pinned 1000 mV ----
    log("=== ceiling probe @ 1000 mV (pushing PLL toward 3 GHz) ===")
    set_volt(1000); set_freq(2700); time.sleep(0.4)
    last_ok = None; prevmhz = 0
    fb = fbfor(2700)
    while fb <= d.EXPLORE_FBDIV_MAX:
        d.set_fbdiv_explore(fb); mhz = d.clocks()["core_l2cpu_mhz"][0]
        t, c, w2, thr = safety()
        if unsafe(t, thr): log(f"  SAFETY abort fb{fb} {mhz}MHz {t}C thr={thr}"); break
        if mhz <= prevmhz - 50: log(f"  PLL not tracking (fb{fb} -> {mhz}MHz) — VCO ceiling"); break
        try:
            p, de = probe()
        except Exception as e:
            log(f"  fb{fb} {mhz}MHz read-fail ({type(e).__name__}) — WEDGE/unlock"); break
        ok = all(p)
        log(f"  fb{fb} {mhz}MHz {['P' if x else 'F' for x in p]} {t}C {c}A" + ("" if ok else f" de={de}"))
        if ok: last_ok = mhz
        else: break
        prevmhz = mhz; fb += 2
    out["ceiling_1v"] = last_ok
    log(f"\n>>> MAX STABLE @ 1000 mV = {last_ok} MHz  (3 GHz needs ~1104 mV by extrapolation)")
except Exception as e:
    import traceback; log("EXC", traceback.format_exc())

log("\nrestoring ...")
try:
    d.set_core_freq(200)
    cur = measV()
    while cur > 720: cur = max(cur - 40, 710); d.force_vdd(cur, allow_step=True); time.sleep(0.15)
    log("idle", d.perf_idle(), measV(), "mV @", d.clocks()["core_l2cpu_mhz"][0], "MHz")
except Exception as e: log("restore issue", e)
with open("scratchpad/vmin_1v.json", "w") as f: json.dump(out, f, indent=1)
log("wrote scratchpad/vmin_1v.json")
