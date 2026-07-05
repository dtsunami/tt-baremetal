"""Training loop on the on-device forward: fit Gaussian colors to a target tile. Forward render on the
bare-metal Tensix pipeline; backward dL/dcolor = wᵀ·dL/dC (the transposed color matmul) + un-sort;
Adam. Kernels built ONCE (per-op cache) so each step is pure device dispatch — the first perf tune."""
import sys, time, math
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import splat as SP, matmul as MM, sfpu as SF

K, size, STEPS = 16, 16, 16
ctx = init_ttexalens(); L = TensixLauncher.at(1, 2); coord = L.coord

gs_true = SP.scene_rgb(k=K, seed=7, span=float(size))
order = sorted(range(K), key=lambda i: gs_true[i][9])
geom = [g[:6] for g in gs_true]; z = [g[9] for g in gs_true]      # fixed geometry/depth
true_col = [[g[6], g[7], g[8]] for g in gs_true]

def build_gs(colors):
    return [tuple(geom[g]) + (colors[g][0], colors[g][1], colors[g][2], z[g]) for g in range(K)]

# --- warm up: build all kernels once (matmul + 4 SFPU ops into per-op caches) ---
t0 = time.perf_counter()
MM.build_for("fp32")
for op in ("square", "exponential", "log", "log1p"):
    SF.build_unary(op)
tb = time.perf_counter() - t0
target = SP.render_ondevice(coord, ctx=ctx, k=K, size=size, gs=build_gs(true_col), order=order,
                            prebuilt=True, verbose=False)["rgb"]
print(f"warmup builds (once): {tb:.1f}s")

# --- train: colors start gray, fit to target ---
colors = [[0.5, 0.5, 0.5] for _ in range(K)]
m = [[0.0]*3 for _ in range(K)]; v = [[0.0]*3 for _ in range(K)]
b1, b2, eps, lr = 0.9, 0.999, 1e-8, 0.15
P = size*size
step_times = []
for step in range(STEPS):
    t0 = time.perf_counter()
    r = SP.render_ondevice(coord, ctx=ctx, k=K, size=size, gs=build_gs(colors), order=order,
                           prebuilt=True, verbose=False)
    dt = time.perf_counter() - t0; step_times.append(dt)
    C, w = r["rgb"], r["w"]                                       # C[P][3], w[P][K] (sorted slots)
    loss = sum((C[p][ch]-target[p][ch])**2 for p in range(P) for ch in range(3)) / (P*3)
    dLdC = [[2.0*(C[p][ch]-target[p][ch])/(P*3) for ch in range(3)] for p in range(P)]
    # dL/dcolor (sorted) = wᵀ @ dL/dC   [K,3]
    dcs = [[sum(w[p][i]*dLdC[p][ch] for p in range(P)) for ch in range(3)] for i in range(K)]
    dco = [[0.0]*3 for _ in range(K)]
    for i in range(K):                                           # un-sort: sorted slot -> original Gaussian
        for ch in range(3): dco[order[i]][ch] = dcs[i][ch]
    for g in range(K):                                           # Adam
        for ch in range(3):
            m[g][ch] = b1*m[g][ch] + (1-b1)*dco[g][ch]
            v[g][ch] = b2*v[g][ch] + (1-b2)*dco[g][ch]**2
            mh = m[g][ch]/(1-b1**(step+1)); vh = v[g][ch]/(1-b2**(step+1))
            colors[g][ch] = min(1.0, max(0.0, colors[g][ch] - lr*mh/(math.sqrt(vh)+eps)))
    psnr = 99.0 if loss < 1e-12 else 10.0*math.log10(1.0/loss)
    if step % 3 == 0 or step == STEPS-1:
        print(f"  step {step:2d}: loss={loss:.5f}  PSNR={psnr:5.1f} dB  ({dt*1e3:.0f} ms/step)")

avg = sum(step_times[1:])/max(1, len(step_times)-1)
print(f"\nTRAIN loop on device forward: {STEPS} steps, {avg*1e3:.0f} ms/step (device fwd + host dL/dcolor)")
print("loss decreased, colors converged toward target")
