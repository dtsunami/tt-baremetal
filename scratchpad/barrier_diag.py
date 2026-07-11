"""het-barrier diagnostic / repro — drives HetGridEngine over a SYNTHETIC scene that spreads N Gaussians across
an IMGW x IMGH image (many occupied tiles -> many cmd10 batches at W workers), to reproduce + pin the
cmd10->cmd1 barrier stall and to validate the watchdog fix.

  TT_BM_WDOG=0  -> legacy 40M spin-and-break (reproduce the bug); breadcrumbs show WHERE a worker was stuck.
  TT_BM_WDOG=1  -> robust progress-watchdog barrier (the fix); should complete + converge, no stall.

  usage: barrier_diag.py [N] [IMG] [W] [NH] [STEPS]     (defaults 4000 256 8 4 3 = the reported hang config)
Env: TT_BM_WDOG (default 1), TT_BM_DIAG=1 (per-phase snapshots), TT_BM_WPRODUCE (0/1)."""
import sys, os, time, math
sys.path.insert(0, "/home/starboy/bhtop/src")
import numpy as np

N     = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
IMG   = int(sys.argv[2]) if len(sys.argv) > 2 else 256
W     = int(sys.argv[3]) if len(sys.argv) > 3 else 8
NH    = int(sys.argv[4]) if len(sys.argv) > 4 else 4
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 3
os.environ.setdefault("TT_BM_DIAG", "1")
IMGW = IMGH = IMG
WDOG = os.environ.get("TT_BM_WDOG", "1") == "1"

rng = np.random.default_rng(1234)
# identity camera at origin looking down +z: gx = fx*X/Z + cx.  fx=cx=IMG/2, Z=1 -> X in (-1,1) spans (0,IMG).
P = np.zeros((N, 14), np.float64)
P[:, 0] = rng.uniform(-0.92, 0.92, N)          # mean X
P[:, 1] = rng.uniform(-0.92, 0.92, N)          # mean Y
P[:, 2] = 1.0                                   # mean Z (all in front)
P[:, 3:6] = math.log(0.02)                      # small isotropic world scale -> compact ~2.5px screen footprint
P[:, 6] = 1.0                                   # quat w (identity)
P[:, 10] = 0.5                                  # opacity
P[:, 11:14] = rng.uniform(0.1, 0.9, (N, 3))    # color

cam16 = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, IMG / 2, IMG / 2, IMG / 2, IMG / 2]
yy, xx = np.mgrid[0:IMGH, 0:IMGW].astype(np.float32) / IMG
tgt = np.stack([xx, yy, 0.5 * (xx + yy)], -1).astype(np.float32)   # smooth gradient target
if os.environ.get("TT_TGT") == "rand":                            # high-entropy target: stress produce rounding
    tgt = rng.uniform(0, 1, (IMGH, IMGW, 3)).astype(np.float32)
LR = [0.005, 0.005, 0.005, 0.003, 0.003, 0.003, 0.001, 0.001, 0.001, 0.001, 0.02, 0.005, 0.005, 0.005]
LR = [x * float(os.environ.get("LR_SCALE", "1")) for x in LR]

print(f"[diag] N={N} {IMGW}x{IMGH} ({IMGW//16}x{IMGH//16}={IMGW//16*IMGH//16} tiles) W={W} NH={NH} "
      f"WDOG={'ON(fix)' if WDOG else 'OFF(legacy-40M)'} WPRODUCE={os.environ.get('TT_BM_WPRODUCE','0')}", flush=True)

from bhtop.het.grid_engine import HetGridEngine

t0 = time.time()
eng = HetGridEngine(N, IMGW, IMGH, W=W, NH=NH)
eng.set_params(P)
eng.set_views(tgt[None])
print(f"[diag] boot {time.time()-t0:.1f}s", flush=True)


def show(snap_list):
    for tag, st, ms, d in snap_list:
        lb = d["last_break"]
        print(f"    [{tag} step{st} {ms}ms] werr={d['werr']} aborted={d['aborted']} "
              f"wdog_breaks={d['wdog_breaks']} legacy_breaks={d['legacy_breaks']} "
              f"last_break=cmd{lb['cmd']}/h{lb['h']} ring={lb['ring']} hdone={lb['hdone']} "
              f"wstate={lb['wstate']} wslot={lb['wslot']}", flush=True)
        for h in d["harts"]:
            print(f"        hart{h['h']}: hb={h['hb']:>10} state={h['state']:<8} ring={h['ring']} "
                  f"slot={h['slot']} ackspin={h['ackspin']:>9} ns={h['ns']} hgo={h['hgo']} hdone={h['hdone']}", flush=True)


ok = True
for step in range(1, STEPS + 1):
    try:
        t = time.time(); loss, occ = eng.step(cam16, tgt.reshape(-1).astype(np.float64), LR, step, view_idx=0)
        dt = time.time() - t
        px = IMGW * IMGH * 3; psnr = 99.0 if loss < 1e-9 else 10 * np.log10(px / loss)
        tm = eng.last_timing
        brk = " ".join(f"{k}={tm[k]*1e3:.0f}" for k in ("gt_up", "proj", "orch10", "adam") if k in tm)
        print(f"  step {step}: loss={loss:.2f} ~PSNR={psnr:.2f} occ={occ}t/{-(-occ//W)}b ({dt*1e3:.0f}ms) "
              f"[{brk}]ms werr=0x{eng.last_werr:08x}", flush=True)
    except Exception as e:
        ok = False
        print(f"  step {step}: RAISED {type(e).__name__}: {str(e)[:200]}", flush=True)
        break

print("[diag] per-phase breadcrumbs:", flush=True)
show(eng.diag_log[-6:])
print("[diag] final snapshot:", flush=True)
d = eng.hart_diag()
show([("final", STEPS, 0, d)])
print(f"[diag] {'DONE-OK' if ok else 'STOPPED'}  slots(flag/ack sample):", flush=True)
for sl in d["slots"][:W]:
    print(f"    slot{sl['s']}: flag={sl['flag']} ack={sl['ack']}", flush=True)
