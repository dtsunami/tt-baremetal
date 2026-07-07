"""Scale profile: drive HetGridEngine over a REAL prepped scene (multi-view) and time the step breakdown to
find the next bottleneck. Cycles views, per step: upload cam+gt -> project -> host-bin -> orchestrate batches
(NH harts) -> Adam -> loss. Prints loss trend (should fall) + coarse timing.
  usage: profile_real.py <scene.npz> [NCAP] [STEPS] [W] [NH]"""
import sys, time, numpy as np
sys.path.insert(0, "/home/starboy/bhtop/src"); sys.path.insert(0, "/home/starboy/bhtop/scratchpad")
from bhtop.het.grid_engine import HetGridEngine

d = np.load(sys.argv[1])
NCAP = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
STEPS = int(sys.argv[3]) if len(sys.argv) > 3 else 16
W = int(sys.argv[4]) if len(sys.argv) > 4 else 6
NH = int(sys.argv[5]) if len(sys.argv) > 5 else 4
poses, imgs, params = d["poses"], d["imgs"], d["params"]
IMGW, IMGH = int(d["imgw"]), int(d["imgh"])
N = min(NCAP, len(params)); params = params[:N]
V = len(poses)
LR = [0.005, 0.005, 0.005, 0.003, 0.003, 0.003, 0.001, 0.001, 0.001, 0.001, 0.02, 0.005, 0.005, 0.005]
print(f"[profile] {sys.argv[1]}: N={N} {IMGW}x{IMGH} ({IMGW//16}x{IMGH//16}={IMGW//16*IMGH//16} tiles) "
      f"V={V} views W={W} NH={NH}", flush=True)
t0 = time.time(); eng = HetGridEngine(N, IMGW, IMGH, W=W, NH=NH); eng.set_params(params)
eng.set_views(imgs)                                    # upload all V images ONCE (resident); step by index
print(f"[profile] engine boot {time.time()-t0:.1f}s (views resident)", flush=True)
FIXV = int(sys.argv[6]) if len(sys.argv) > 6 else -1   # >=0: overfit ONE view (convergence check)
for step in range(1, STEPS + 1):
    v = FIXV if FIXV >= 0 else (step - 1) % V
    cam16 = [float(x) for x in poses[v]]
    tgt = imgs[v].reshape(-1).astype(np.float64)
    t = time.time(); loss, occ = eng.step(cam16, tgt, LR, step, view_idx=v); dt = time.time() - t
    px = IMGW * IMGH * 3; psnr = 99.0 if loss < 1e-9 else 10 * np.log10(px / loss)
    tm = eng.last_timing; brk = "  ".join(f"{k}={tm[k]*1e3:.0f}" for k in ("gt_up", "proj", "pub_bin", "batch", "idlg_wr", "adam"))
    print(f"  step {step:2d} v{v:2d}: loss={loss:.1f} ~PSNR={psnr:5.2f} occ={occ}t/{-(-occ//W)}b ({dt*1e3:.0f}ms) [{brk}]ms", flush=True)
print("[profile] done", flush=True)
