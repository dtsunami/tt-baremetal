"""vf_freq_explore.py — empirical V/F FRONTIER, all 4 x280 harts. For each voltage level, step the PLL fbdiv UP
from 1750 MHz (measuring the ACTUAL core MHz each step) until the FP self-check virus's checksum breaks = the
max stable freq at that voltage. Soft checksum errors precede a hard PLL-unlock, so we find the fail edge and
keep the card alive across voltage levels. Writes the whole frontier to JSON for plotting. Card must be idle."""
import sys, time, json
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR

TILE, NH = 0, 4
KERNEL = "src/bhtop/kernels/x280/vf_margin_fp/vf_margin_fp.c"
OUT = "scratchpad/vf_frontier.json"
FB_START, FB_STEP, FB_CAP = 140, 2, 210      # 140=1750 MHz; +2 ~ +25 MHz/step
INTO_FAIL = 2                                 # log this many steps past first fail, then back off

ctx = init_ttexalens(); ctx.use_4B_mode = False
d = L2cpu(ctx=ctx)
frontier = {"kernel": KERNEL, "levels": [], "vdd_max": None}

def harts():
    a = d.telemetry_all(TILE)
    return {h: dict(passes=a[h][0], errs=a[h][3]) for h in range(NH)}
def vc(): return d.power().get("vcore_mv")
def ramp_to(target, over=False):
    cur = vc() or 0
    while abs(cur - target) > 8:
        nxt = cur + max(min(target - cur, 40), -40)
        d.force_vdd(nxt, allow_step=True, allow_over=over); time.sleep(0.25); got = vc()
        if got is None: raise RuntimeError("vcore read None during ramp")
        if abs(got - cur) < 2 and abs(got - target) > 12:       # not moving = clamped (over-volt refused)
            return got
        cur = got
    return cur

try:
    rs = d.reset_state(TILE)
    if rs["wedged"]: sys.exit("tile 0 wedged — tt-smi -r 0 first")
    if not rs["released"]: print("bringup:", d.bringup(TILE))
    d.set_core_freq(200); time.sleep(0.3)
    w = tc.compile_source(KERNEL, base=CODE_ADDR)
    for h in range(NH): d.load(TILE, h, w, redirect=True)
    time.sleep(0.9)
    frontier["vdd_max"] = d.limits()["vdd_max_mv"]
    _a = d.telemetry_all(TILE)
    print("golden@200:", {h: hex(_a[h][1]) for h in range(NH)}, "all-equal:", len({_a[h][1] for h in range(NH)}) == 1)

    d.perf_busy(); time.sleep(0.3)
    # probe whether CMFW honors over-volt past vdd_max (needed for the high-freq levels)
    ramp_to(900); d.force_vdd(920, allow_step=True, allow_over=True); time.sleep(0.4)
    over_ok = (vc() or 0) > 905
    print(f"over-volt probe: requested 920 -> vcore {vc()} mV -> over-volt {'HONORED' if over_ok else 'CLAMPED at vdd_max'}")
    levels = [760, 820, 880, 900] + ([960, 1010] if over_ok else [])

    for V in levels:
        d.set_core_freq(1750); time.sleep(0.3)                  # back to the safe anchor
        gotV = ramp_to(V, over=(V > 900)); time.sleep(0.3)
        base = harts()
        print(f"\n=== V={gotV} mV (target {V}) — stepping freq up ===")
        pts = []; fb = FB_START; last_ok_mhz = None; into = 0; prev = base
        while fb <= FB_CAP:
            fb += FB_STEP
            try:
                r = d.set_fbdiv_explore(fb); mhz = r["core_mhz"][0]
                time.sleep(0.6); cur = harts(); mon = d.monitor()
            except Exception as e:
                print(f"  fb{fb}: WEDGE/read-fail ({type(e).__name__})"); pts.append({"fbdiv": fb, "wedged": True}); break
            fail = False; derr = {}
            for h in range(NH):
                de = cur[h]["errs"] - prev[h]["errs"]; dp = cur[h]["passes"] - prev[h]["passes"]
                derr[h] = de
                if de > 0 or dp == 0: fail = True
            pts.append({"fbdiv": fb, "mhz": mhz, "vcore": vc(), "temp": mon.get("asic_temp_c"),
                        "derrs": derr, "fail": fail})
            print(f"  fb{fb:>3} {mhz:>5} MHz  {vc()}mV {mon.get('asic_temp_c')}C  errsΔ={list(derr.values())}"
                  + ("  <== FAIL" if fail else "  ok") + ("" if mon["safe"] else f"  ALARM{mon['alarms']}"))
            prev = cur
            if not fail: last_ok_mhz = mhz
            else:
                into += 1
                if into > INTO_FAIL: break
            if not mon["safe"]: break
        d.set_fbdiv_explore(FB_START)                            # restore 1750 before next level
        frontier["levels"].append({"vcore_mv": gotV, "max_stable_mhz": last_ok_mhz, "points": pts})
        print(f"  -> max stable @ {gotV} mV = {last_ok_mhz} MHz")
finally:
    with open(OUT, "w") as f: json.dump(frontier, f, indent=1)
    print("\nwrote", OUT)
    print("restoring idle ...")
    try:
        d.set_core_freq(200)
        cur = vc() or 900
        while cur > 715:
            cur = max(cur - 40, 710); d.force_vdd(cur, allow_step=True); time.sleep(0.2)
        print("  ", d.perf_idle(), "-> vcore", vc(), "mV @", d.clocks()["core_l2cpu_mhz"], "MHz")
    except Exception as e: print("  restore issue:", e)
