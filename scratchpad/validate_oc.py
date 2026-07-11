"""Validate the 2.4 GHz @ 900 mV operating point on the recovered card (mirrors grid_engine._apply_oc):
bringup -> load FP virus -> voltage-lead to 900 mV -> PLL to 2400 -> run 3s, confirm all harts error-free.
Harts are in clean reset from the power-cycle, so bringup directly (no tt-smi reset)."""
import sys, time, os
os.chdir("/home/starboy/bhtop")
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
TILE, NH = 0, 4
def log(*a): print(*a, flush=True)
ctx = init_ttexalens(); ctx.use_4B_mode = False
d = L2cpu(ctx=ctx)
rs = d.reset_state(TILE); log("reset_state", rs)
if rs["released"]:
    log("harts already released (prior bringup) -> load over them, skip bringup")   # avoids a tt-smi reset
else:
    log("bringup", d.bringup(TILE))
w = tc.compile_source("src/bhtop/kernels/x280/vf_margin_fp/vf_margin_fp.c", base=CODE_ADDR)
for h in range(NH): d.load(TILE, h, w, redirect=True)
time.sleep(1.0)
a = d.telemetry_all(TILE); log("golden", [hex(a[h][1]) for h in range(NH)], "all-eq", len({a[h][1] for h in range(NH)})==1)
# --- apply OC: VOLTAGE LEADS FREQUENCY ---
cur = d.power().get("vcore_mv") or 720
while cur < 900: cur = min(cur + 40, 900); d.force_vdd(cur, allow_step=True); time.sleep(0.15)
log("vcore ->", d.power().get("vcore_mv"), "mV")
fb = round(140 + (2400 - 1750) / 12.5); r = d.set_fbdiv_explore(fb)
log(f"PLL fbdiv {fb} -> {r['core_mhz'][0]} MHz")
# --- 3 s stability run ---
a0 = d.telemetry_all(TILE); time.sleep(3.0); a1 = d.telemetry_all(TILE)
mon = d.monitor()
log(f"@ {d.clocks()['core_l2cpu_mhz'][0]} MHz {mon.get('vcore_mv')} mV {mon.get('asic_temp_c')}C safe={mon['safe']}:")
allok = True
for h in range(NH):
    dp = a1[h][0]-a0[h][0]; de = a1[h][3]-a0[h][3]
    log(f"  hart{h}: dpass={dp} derr={de} {'OK' if (dp>0 and de==0) else 'FAIL'}")
    allok = allok and dp>0 and de==0
log("=== 2.4 GHz @ 900 mV:", "STABLE (error-free, all harts)" if allok else "NOT STABLE", "===")
# --- restore idle ---
d.set_core_freq(200); cur = d.power().get("vcore_mv") or 900
while cur > 720: cur = max(cur - 40, 710); d.force_vdd(cur, allow_step=True); time.sleep(0.12)
log("restored:", d.perf_idle(), d.power().get("vcore_mv"), "mV @", d.clocks()["core_l2cpu_mhz"][0], "MHz")
