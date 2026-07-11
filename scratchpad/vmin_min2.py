"""Isolate the 2100 MHz step-down with per-probe visibility (voltage-leads, self-reset). Streams every probe so
a hard-wedge shows as the last printed line before a stall; a soft-fail shows as F with heartbeat still moving."""
import sys, time, os, subprocess
os.chdir("/home/starboy/bhtop"); sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
TILE, NH = 0, 4
def log(*a): print(*a, flush=True)

log("reset..."); subprocess.run(["/home/starboy/.local/bin/tt-smi", "-r", "0"], capture_output=True, timeout=150)
time.sleep(18); log("settled")
ctx = init_ttexalens(); ctx.use_4B_mode = False
d = L2cpu(ctx=ctx)
if d.reset_state(TILE)["released"]: log("released, abort"); sys.exit(0)
log("bringup", d.bringup(TILE))
w = tc.compile_source("src/bhtop/kernels/x280/vf_margin_fp/vf_margin_fp.c", base=CODE_ADDR)
for h in range(NH): d.load(TILE, h, w, redirect=True)
time.sleep(1.0)
a = d.telemetry_all(TILE); log("golden", [hex(a[h][1]) for h in range(NH)])
d.set_core_freq(1750)
d.force_vdd(820, allow_step=True); time.sleep(0.3)          # VOLTAGE FIRST
d.set_fbdiv_explore(168); time.sleep(0.4)                    # then ~2100 MHz
log("freq", d.clocks()["core_l2cpu_mhz"][0], "vcore", d.power().get("vcore_mv"))

v = 820
while v >= 712:
    d.force_vdd(v, allow_step=True); time.sleep(0.28); mv = d.power().get("vcore_mv")
    log(f"  probing {mv} mV ...")                            # printed BEFORE the read -> a hang shows here
    a0 = d.telemetry_all(TILE); time.sleep(0.35); a1 = d.telemetry_all(TILE)
    r = [(a1[h][3] == a0[h][3] and a1[h][0] > a0[h][0]) for h in range(NH)]
    dp = [a1[h][0] - a0[h][0] for h in range(NH)]; de = [a1[h][3] - a0[h][3] for h in range(NH)]
    log(f"   {mv}  pass={['P' if x else 'F' for x in r]} dp={dp} de={de}")
    if not all(r): log(f"  FIRST FAIL {mv} mV  hart{r.index(False)}  chip Vmin ~ {mv+8}"); break
    v -= 8

d.set_core_freq(200)
cur = d.power().get("vcore_mv") or 900
while cur > 720: cur = max(cur - 40, 710); d.force_vdd(cur, allow_step=True); time.sleep(0.15)
log("idle", d.perf_idle(), d.power().get("vcore_mv"), "mV")
