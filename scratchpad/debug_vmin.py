"""Pin down why the Vmin anchor fails. Walk the exact load_pattern -> set_freq -> set_volt -> probe path,
printing freq/golden/passes/errs at each step for all 4 harts."""
import sys, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
TILE, NH = 0, 4
K = "src/bhtop/kernels/x280/vf_margin_fp/vf_margin_fp.c"
ctx = init_ttexalens(); ctx.use_4B_mode = False
d = L2cpu(ctx=ctx)

def snap(tag):
    a = d.telemetry_all(TILE)
    print(f"  {tag:30s} clk={d.clocks()['core_l2cpu_mhz'][0]:>4} vcore={d.power().get('vcore_mv')} "
          f"pass={[a[h][0] for h in range(NH)]} err={[a[h][3] for h in range(NH)]} gold={[hex(a[h][1]) for h in range(NH)]}")
    return a

rs = d.reset_state(TILE)
if not rs["released"]: print("bringup", d.bringup(TILE))
print("perf_busy", d.perf_busy())

d.set_core_freq(200); time.sleep(0.3)
w = tc.compile_source(K, base=CODE_ADDR)
for h in range(NH): d.load(TILE, h, w, redirect=True)
time.sleep(1.0)
snap("after load@200")
time.sleep(0.6); snap("dwell@200 (advancing?)")
d.set_core_freq(1750); time.sleep(0.5)
snap("after set 1750")
time.sleep(0.6); snap("dwell@1750")
# emulate set_freq(1900)
fb = round(140 + (1900 - 1750) / 12.5)
print("set_fbdiv", fb, "->", d.set_fbdiv_explore(fb))
time.sleep(0.5); snap("after fbdiv@1900target")
# emulate set_volt(850)
d.force_vdd(850, allow_step=True); time.sleep(0.4)
a0 = snap("after force_vdd 850")
time.sleep(0.5); a1 = snap("dwell@1900/850")
print("  per-hart pass:", {h: (a1[h][3] == a0[h][3] and a1[h][0] > a0[h][0]) for h in range(NH)})
# restore
d.set_core_freq(200)
cur = d.power().get('vcore_mv')
while cur > 715: cur = max(cur - 40, 710); d.force_vdd(cur, allow_step=True); time.sleep(0.2)
print("restore", d.perf_idle(), d.power().get('vcore_mv'), "mV")
