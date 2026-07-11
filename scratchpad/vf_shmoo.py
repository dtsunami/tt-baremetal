"""vf_shmoo.py [int|fp] [vstart] [vfloor] [vstep] — V/F Shmoo at 1750 MHz, all 4 x280 harts.
Sweeps vcore DOWN from vstart to vfloor, self-checking each hart's deterministic virus checksum vs a golden
captured at safe 200 MHz. Logs every (vcore, temp, per-hart errs/dpass, wedged) point to JSON for plotting.
Approaches the fail edge from the safe side; keeps sweeping a few steps INTO the error region to show the
fail point, then stops before a hard wedge (or catches the wedge + recovers). Card must be idle."""
import sys, time, json
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR

KIND    = sys.argv[1] if len(sys.argv) > 1 else "fp"
VSTART  = int(sys.argv[2]) if len(sys.argv) > 2 else 820
VFLOOR  = int(sys.argv[3]) if len(sys.argv) > 3 else 700
VSTEP   = int(sys.argv[4]) if len(sys.argv) > 4 else 10
TILE, NH = 0, 4
KERNEL  = f"src/bhtop/kernels/x280/vf_margin{'_fp' if KIND=='fp' else ''}/vf_margin{'_fp' if KIND=='fp' else ''}.c"
OUT     = f"scratchpad/shmoo_{KIND}.json"
EXTRA_INTO_FAIL = 3                        # keep logging this many steps past the first fail, then stop

ctx = init_ttexalens(); ctx.use_4B_mode = False
d = L2cpu(ctx=ctx)
log = {"kind": KIND, "kernel": KERNEL, "freq_mhz": 1750, "vdd_max": None, "vdd_min": None, "points": []}

def harts():
    a = d.telemetry_all(TILE)
    return {h: dict(passes=a[h][0], golden=a[h][1], chk=a[h][2], errs=a[h][3]) for h in range(NH)}
def vc(): return d.power().get("vcore_mv")
def ramp_to(target):                       # step force_vdd in <=40 mV chunks, verify tracking
    cur = vc() or 0
    while cur < target:
        nxt = min(cur + 40, target); d.force_vdd(nxt); time.sleep(0.3); got = vc()
        if got is None or abs(got - nxt) > 20: raise RuntimeError(f"force_vdd {nxt} -> {got}: not tracking")
        cur = got

try:
    rs = d.reset_state(TILE)
    if rs["wedged"]: sys.exit("tile 0 wedged — tt-smi -r 0 first")
    if not rs["released"]: print("bringup:", d.bringup(TILE))
    d.set_core_freq(200); time.sleep(0.3)
    w = tc.compile_source(KERNEL, base=CODE_ADDR)
    for h in range(NH): d.load(TILE, h, w, redirect=True)
    time.sleep(0.9)
    lim = d.limits(); log["vdd_max"], log["vdd_min"] = lim["vdd_max_mv"], lim["vdd_min_mv"]
    hs = harts(); g = {h: hs[h]["golden"] for h in range(NH)}
    print(f"[{KIND}] golden@200MHz:", {h: hex(g[h]) for h in range(NH)}, "all-equal:", len(set(g.values())) == 1)

    print("GO_BUSY:", d.perf_busy()); time.sleep(0.3); ramp_to(VSTART)
    d.set_core_freq(1750); time.sleep(0.6)
    print(f"\n1750 MHz Shmoo, sweeping {vc()} -> {VFLOOR} mV (step {VSTEP}):")
    print(f"{'vcore':>6}{'temp':>6}   per-hart errs(Δ)/dpass")
    prev = harts(); first_fail = None; into = 0
    v = min(VSTART, vc())
    while v >= VFLOOR:
        d.force_vdd(v); time.sleep(0.6)
        wedged = False
        try: cur = harts(); mon = d.monitor()
        except Exception as e:
            print(f"{v:>6}   READ FAIL ({type(e).__name__}) -> WEDGE"); wedged = True
            log["points"].append({"vcore": v, "wedged": True}); break
        row, any_fail = [], False
        pt = {"vcore": vc(), "temp": mon.get("asic_temp_c"), "wedged": False, "harts": {}}
        for h in range(NH):
            de = cur[h]["errs"] - prev[h]["errs"]; dp = cur[h]["passes"] - prev[h]["passes"]
            pt["harts"][h] = {"errs": cur[h]["errs"], "derrs": de, "dpass": dp}
            row.append(f"h{h}:{de}/{dp}")
            if de > 0 or dp == 0: any_fail = True
        log["points"].append(pt)
        mark = "  <== FAIL" if any_fail else ""
        print(f"{v:>6}{mon.get('asic_temp_c'):>6}   " + "  ".join(row) + mark +
              ("" if mon["safe"] else f"  ALARM {mon['alarms']}"))
        prev = cur
        if any_fail:
            if first_fail is None: first_fail = v
            into += 1
            if into > EXTRA_INTO_FAIL: print("  (logged into the fail region; stopping before hard wedge)"); break
        if not mon["safe"]: break
        v -= VSTEP

    log["first_fail_mv"] = first_fail
    log["vmin_mv"] = (first_fail + VSTEP) if first_fail else VFLOOR
    print(f"\n=== {KIND} Shmoo @1750: first-fail {first_fail} mV, Vmin {log['vmin_mv']} mV ===")
finally:
    with open(OUT, "w") as f: json.dump(log, f, indent=1)
    print("wrote", OUT, f"({len(log['points'])} points)")
    print("restoring idle ...")
    try:
        cur = vc()
        for t in range(cur - 40 if cur else 710, 700, -40): d.force_vdd(max(t, 710), allow_step=True); time.sleep(0.2)
    except Exception: pass
    try: d.set_core_freq(200); print("  ", d.perf_idle(), "vcore", vc(), "mV")
    except Exception as e: print("  restore failed:", e)
