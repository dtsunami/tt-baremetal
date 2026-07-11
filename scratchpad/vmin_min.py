"""Minimal, heavily-printed single-frequency Vmin probe. Assumes a FRESH tt-smi -r 0 (harts in reset).
Bringup -> load fp -> at 1900 MHz, approach the fail edge FROM ABOVE (820 mV down), print every probe.
No hang-prone finally; restore is best-effort with a print before it."""
import sys, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
TILE, NH = 0, 4
K = "src/bhtop/kernels/x280/vf_margin_fp/vf_margin_fp.c"
def log(*a): print(*a, flush=True)

ctx = init_ttexalens(); ctx.use_4B_mode = False
d = L2cpu(ctx=ctx)
rs = d.reset_state(TILE); log("reset_state", rs)
if rs["released"]:
    log("harts already released — need a fresh tt-smi -r 0; aborting cleanly"); sys.exit(0)
log("bringup", d.bringup(TILE))
w = tc.compile_source(K, base=CODE_ADDR)
for h in range(NH): d.load(TILE, h, w, redirect=True)
time.sleep(1.0)
a = d.telemetry_all(TILE)
log("golden", [hex(a[h][1]) for h in range(NH)], "all-eq", len({a[h][1] for h in range(NH)}) == 1)

d.set_core_freq(1750); fb = 152; d.set_fbdiv_explore(fb)
log("freq", d.clocks()["core_l2cpu_mhz"][0], "MHz (fbdiv", fb, ")")

def probe():
    a0 = d.telemetry_all(TILE); time.sleep(0.35); a1 = d.telemetry_all(TILE)
    return {h: {"p": a1[h][3] == a0[h][3] and a1[h][0] > a0[h][0], "dp": a1[h][0] - a0[h][0],
                "de": a1[h][3] - a0[h][3]} for h in range(NH)}

log("\n vcore  per-hart pass/dpass/derr")
v = 820
while v >= 705:
    d.force_vdd(v, allow_step=True); time.sleep(0.28); mv = d.power().get("vcore_mv")
    r = probe()
    row = "  ".join(f"h{h}:{'P' if r[h]['p'] else 'F'}/{r[h]['dp']}/{r[h]['de']}" for h in range(NH))
    fails = [h for h in range(NH) if not r[h]["p"]]
    log(f" {mv:>4}  {row}" + (f"   <== hart{fails[0]} fails (chip Vmin ~ {mv+8} mV)" if fails else ""))
    if fails: break
    v -= 8

log("\nrestoring ...")
try:
    d.set_core_freq(200)
    cur = d.power().get("vcore_mv") or 900
    while cur > 720: cur = max(cur - 40, 710); d.force_vdd(cur, allow_step=True); time.sleep(0.15)
    log("idle", d.perf_idle(), d.power().get("vcore_mv"), "mV @", d.clocks()["core_l2cpu_mhz"][0], "MHz")
except Exception as e:
    log("restore issue:", e)
