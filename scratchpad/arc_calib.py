"""ARC DVFS calibration — SAFE first pass. Validates the arc_msg path + reads the firmware ceilings + confirms
GO_BUSY raises the shared vcore, then restores idle. NO force_vdd (its arg units are still unverified). If the
canary wedges, recover with `tt-smi -r 0`. Card must be idle (no training / no web server owning the device)."""
import sys, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu

ctx = init_ttexalens(); ctx.use_4B_mode = False
d = L2cpu(ctx=ctx)

print("baseline power:", {k: d.power().get(k) for k in ("vcore_mv","aiclk_mhz","asic_temp_c","throttler")})

# 1) CANARY — GET_VOLTAGE is a pure read; confirms arc_msg + the 0xAA00 prefix work without changing state.
print("\n[1] canary GET_VOLTAGE ...")
try:
    can = d.get_voltage()
    print("    reply:", can["reply"], "| telemetry vcore:", can["telemetry_vcore_mv"], "mV")
except Exception as e:
    print("    CANARY FAILED:", type(e).__name__, str(e)[:160]); print("    -> arc_msg path unusable; STOP."); sys.exit(1)

# 2) LIMITS — the firmware safety ceilings (never exceed these).
print("\n[2] limits (firmware clamps):")
lim = d.limits()
for k, v in lim.items(): print(f"    {k:22s} {v}")

# 3) GO_BUSY — request the busy perf-state; ARC/AVS should lift the shared rail toward ~810 mV.
print("\n[3] GO_BUSY (perf_busy) ...")
try:
    print("    ", d.perf_busy())
    time.sleep(0.4)
    mon = d.monitor()
    print("    post-busy: vcore=%s mV  aiclk=%s  temp=%s C  throttler=%s  safe=%s  alarms=%s" % (
        mon.get("vcore_mv"), mon.get("aiclk_mhz"), mon.get("asic_temp_c"),
        mon.get("throttler"), mon.get("safe"), mon.get("alarms")))
except Exception as e:
    print("    GO_BUSY FAILED:", type(e).__name__, str(e)[:160])

# 4) restore idle so we leave the card as we found it.
print("\n[4] restore idle (perf_idle) ...")
try:
    print("    ", d.perf_idle()); time.sleep(0.4)
    print("    post-idle vcore:", d.power().get("vcore_mv"), "mV")
except Exception as e:
    print("    perf_idle failed:", type(e).__name__, str(e)[:160])

print("\nDONE. If vcore moved 716 -> ~810 under GO_BUSY and back, the ARC DVFS path is live and calibrated.")
