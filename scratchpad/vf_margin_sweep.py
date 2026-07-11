"""vf_margin_sweep.py — V/F margining, all 4 x280 harts. Finds each hart's Vmin at 1750 MHz via the
self-checking vf_margin virus (deterministic checksum vs a golden captured at safe 200 MHz).

Sequence (approach the fail edge from the SAFE side so we catch a SOFT bit-error before a hard wedge):
  bringup -> 200 MHz (capture golden, cross-hart sanity) -> GO_BUSY(811) -> ramp force_vdd up to 900
  (readback DECODES force_vdd's arg units) -> PLL 1750 -> step vcore DOWN 10 mV until any hart's checksum
  errs climb (= fail) or heartbeat freezes (= wedge). Reports per-hart Vmin + a guard-banded operating V.

SAFE to abort: on wedge -> perf_idle + advises tt-smi -r 0. Card must be idle (no training / no web owner)."""
import sys, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR

TILE, NH = 0, 4
KERNEL = "src/bhtop/kernels/x280/vf_margin/vf_margin.c"
VMAX, VFLOOR, VSTEP = 900, 800, 10        # sweep 900 -> 800 in 10 mV steps (vdd_max=900 measured)
GUARDBAND = 40                            # add to Vmin for the real workload's worse timing paths

ctx = init_ttexalens(); ctx.use_4B_mode = False
d = L2cpu(ctx=ctx)

def harts():
    a = d.telemetry_all(TILE)
    return {h: dict(passes=a[h][0], golden=a[h][1], chk=a[h][2], errs=a[h][3], gset=a[h][5]) for h in range(NH)}

def vcore(): return d.power().get("vcore_mv")

def ramp_to(target):
    """Step force_vdd up in <=40 mV chunks; VERIFY vcore tracks (decodes the mV arg unit). Abort if it doesn't."""
    cur = vcore() or 0
    while cur < target:
        nxt = min(cur + 40, target)
        d.force_vdd(nxt); time.sleep(0.3)
        got = vcore()
        print(f"    force_vdd {nxt} -> vcore {got} mV")
        if got is None or abs(got - nxt) > 20:
            raise RuntimeError(f"force_vdd {nxt} but vcore={got} — arg unit NOT mV / not honored; ABORT")
        cur = got

try:
    rs = d.reset_state(TILE); print("reset_state:", rs)
    if rs["wedged"]: sys.exit("tile 0 wedged — recover with tt-smi -r 0 first")
    if not rs["released"]:
        print("bringup ..."); print("  ", d.bringup(TILE))
    d.set_core_freq(200); time.sleep(0.3)          # ultra-safe point to capture the golden
    w = tc.compile_source(KERNEL, base=CODE_ADDR)
    for h in range(NH): d.load(TILE, h, w, redirect=True)
    time.sleep(0.8)

    # --- golden capture + cross-hart sanity at 200 MHz ---
    hs = harts()
    if not all(hs[h]["gset"] for h in range(NH)):
        print("waiting for golden capture ..."); time.sleep(1.0); hs = harts()
    gset = {h: hs[h]["golden"] for h in range(NH)}
    print("\ngolden@200MHz:", {h: hex(gset[h]) for h in range(NH)},
          "| all-equal:", len(set(gset.values())) == 1, "| errs:", {h: hs[h]["errs"] for h in range(NH)})
    if len(set(gset.values())) != 1:
        print("WARNING: harts disagree at 200 MHz — golden not trustworthy");
    base_pass = {h: hs[h]["passes"] for h in range(NH)}
    time.sleep(0.6); hs = harts()
    print("200MHz advancing:", {h: hs[h]["passes"] - base_pass[h] for h in range(NH)},
          "errs:", {h: hs[h]["errs"] for h in range(NH)})

    # --- raise the rail, then go to 1750 ---
    print("\nGO_BUSY:", d.perf_busy()); time.sleep(0.4)
    print("ramp rail to", VMAX, "mV:"); ramp_to(VMAX)
    d.set_core_freq(1750); time.sleep(0.6)
    print(f"\n1750 MHz @ {vcore()} mV — anchor check:")
    a0 = harts(); time.sleep(0.8); a1 = harts()
    anchor_errs = {h: a1[h]["errs"] - a0[h]["errs"] for h in range(NH)}
    adv = {h: a1[h]["passes"] - a0[h]["passes"] for h in range(NH)}
    print("  advancing:", adv, "| new errs:", anchor_errs)
    if any(anchor_errs.values()):
        print(f"  !! 1750 MHz ERRORS even at {VMAX} mV (the vdd_max clamp) -> 1750 needs > vdd_max; not reachable "
              "within firmware limits (would need rung-3 flash edit).")
    else:
        print(f"  1750 MHz STABLE at {VMAX} mV. Sweeping voltage DOWN to find Vmin ...")
        # --- the sweep: lower vcore until a hart's errs climb (soft fail) or freezes (wedge) ---
        print(f"\n{'vcore':>6}{'temp':>6}  per-hart (Δpass / errs)")
        prev = harts(); vmin = {h: None for h in range(NH)}  # first-fail vcore per hart
        v = VMAX - VSTEP
        while v >= VFLOOR:
            d.force_vdd(v); time.sleep(0.6)
            try:
                cur = harts()
            except Exception as e:
                print(f"{v:>6}   read FAILED ({type(e).__name__}) -> WEDGE; stopping");
                for h in range(NH):
                    if vmin[h] is None: vmin[h] = v      # wedged here
                break
            mon = d.monitor()
            row = []; failed = False
            for h in range(NH):
                dp = cur[h]["passes"] - prev[h]["passes"]; de = cur[h]["errs"] - prev[h]["errs"]
                row.append(f"h{h}:{dp}/{cur[h]['errs']}")
                if (de > 0 or dp == 0) and vmin[h] is None:
                    vmin[h] = v; failed = True           # first fail (soft error or frozen heartbeat)
            print(f"{v:>6}{mon.get('asic_temp_c'):>6}  " + "  ".join(row) +
                  ("  <== FAIL" if failed else "") + ("" if mon["safe"] else f"  ALARM {mon['alarms']}"))
            prev = cur
            if all(vmin[h] is not None for h in range(NH)) or not mon["safe"]:
                break
            v -= VSTEP

        # --- verdict ---
        print("\n=== Shmoo @1750 MHz ===")
        for h in range(NH):
            fv = vmin[h]
            print(f"  hart{h}: first-fail {fv} mV -> Vmin {fv+VSTEP if fv else '<'+str(VFLOOR)} mV")
        clean = [ (vmin[h]+VSTEP) if vmin[h] else VFLOOR for h in range(NH) ]
        card_vmin = max(clean)
        op = min(card_vmin + GUARDBAND, VMAX)
        print(f"\n  card Vmin@1750 = {card_vmin} mV (worst hart) ; +{GUARDBAND} guardband -> operating {op} mV "
              f"(cap {VMAX}). Set: TT_BM_PERF_BUSY=1 TT_BM_VCORE_MV={op} TT_BM_X280_MHZ=1750")
finally:
    print("\nrestoring safe idle (PLL 200 + perf_idle) ...")
    try: d.set_core_freq(200)
    except Exception as e: print("  set 200 failed:", e)
    try: print("  ", d.perf_idle(), "vcore", vcore(), "mV")
    except Exception as e: print("  perf_idle failed:", e)
