"""Validate (on silicon) that render_ondevice now exposes the forward intermediates the backward needs,
and that the MATRIX reformulation of the composite backward dL/dalpha (matmul w/ a strict-upper suffix
matrix + SFPU reciprocal + eltwise-mul) equals the host serial suffix-sum recurrence. No new device
kernels here — device runs the forward, host does the algebra both ways and compares."""
import sys, math, random
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import splat as SP

K, size = 12, 16
ctx = init_ttexalens(); coord = TensixLauncher.at(1, 2, ctx=ctx).coord
r = SP.render_ondevice(coord, ctx=ctx, k=K, size=size, seed=5, verbose=False)
P = size * size
w, alpha, ar, order, color, rgb = r["w"], r["alpha"], r["ar"], r["order"], r["color"], r["rgb"]

# sanity on exposed intermediates: w == T*alpha with T=prod(1-alpha) front-to-back, cols already in order
bad = 0
for p in range(P):
    T = 1.0
    for i in range(K):                 # columns are in composite (sorted) order
        expect = T * alpha[p][i]
        if abs(expect - w[p][i]) > 1e-3: bad += 1
        T *= (1.0 - alpha[p][i])
print(f"exposed intermediates: w==T*alpha mismatches={bad}/{P*K}  (alpha,ar,w,color,order all present={all(k in r for k in ('alpha','ar','w','color','order','v'))})")

# a target -> dL/dC
rnd = random.Random(7)
target = [[rnd.random() for _ in range(3)] for _ in range(P)]
dLdC = [[2.0 * (rgb[p][ch] - target[p][ch]) / (P * 3) for ch in range(3)] for p in range(P)]

# ---- reference: host SERIAL composite backward (train_geometry recurrence), on device intermediates ----
dA_ref = [[0.0] * K for _ in range(P)]
for p in range(P):
    dw = [sum(dLdC[p][ch] * color[i][ch] for ch in range(3)) for i in range(K)]
    S = 0.0
    for i in reversed(range(K)):
        al = alpha[p][i]; T = w[p][i] / al if al > 1e-12 else 0.0
        dA_ref[p][i] = dw[i] * T - S / max(1.0 - al, 1e-6)
        S += dw[i] * w[p][i]

# ---- reformulation: matmul (dLdC@color^T), eltwise, suffix matmul (U[j][i]=1 iff j>i), reciprocal ----
U = [[1.0 if j > i else 0.0 for i in range(K)] for j in range(K)]   # strict-upper suffix matrix
dA_mat = [[0.0] * K for _ in range(P)]
for p in range(P):
    dw   = [sum(dLdC[p][ch] * color[i][ch] for ch in range(3)) for i in range(K)]   # dLdC @ color^T
    dwW  = [dw[i] * w[p][i] for i in range(K)]                                       # eltwise mul
    suf  = [sum(dwW[j] * U[j][i] for j in range(K)) for i in range(K)]               # dwW @ U
    Tv   = [w[p][i] / alpha[p][i] if alpha[p][i] > 1e-12 else 0.0 for i in range(K)] # w * recip(alpha)
    rOM  = [1.0 / max(1.0 - alpha[p][i], 1e-6) for i in range(K)]                    # recip(1-alpha)
    dA_mat[p] = [dw[i] * Tv[i] - suf[i] * rOM[i] for i in range(K)]

maxerr = max(abs(dA_ref[p][i] - dA_mat[p][i]) for p in range(P) for i in range(K))
relerr = maxerr / (max(abs(dA_ref[p][i]) for p in range(P) for i in range(K)) + 1e-12)
print(f"dL/dalpha  matrix-reformulation vs host-serial:  max_abs={maxerr:.2e}  rel={relerr:.2e}  "
      f"-> {'MATCH' if relerr < 1e-6 else 'MISMATCH'}")
