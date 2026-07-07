"""Device composite-backward dL/dalpha, on bare-metal Tensix, from proven primitives only:
   matmul (dLdC@color^T, dwW@U_suffix), SFPU reciprocal, FPU eltwise mul/sub.
Compares to the exact host serial suffix-sum recurrence (computed on the SAME device forward
intermediates), so the delta is pure bf16 arithmetic error through the device chain."""
import sys, random
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import splat as SP, matmul as MM, sfpu as SF

TILE = SP.TILE
pad, take = SP._pad32, SP._take

def _mm(coord, ctx, A, B, cols):
    return take(MM.run_matmul(coord, ctx=ctx, a=pad(A), b=pad(B), out_format="fp32",
                              prebuilt=True, verbose=False)["c_dev"], len(A), cols)
def _un(coord, ctx, M, op, cols):
    r, _ = SF.run_unary(coord, pad(M), ctx=ctx, op=op, prebuilt=True, timeout=3.0); return take(r, len(M), cols)
def _bin(coord, ctx, A, Bt, op, cols):
    r, _ = SF.run_binary(coord, pad(A), pad(Bt), ctx=ctx, op=op, prebuilt=True, timeout=3.0); return take(r, len(A), cols)

def dLda_device(coord, ctx, dLdC, w, alpha, color, K):
    """All per-pixel-group. Returns dL/dalpha [P][K]. color is [K][3]."""
    P = len(w); colorT = [[color[i][ch] for i in range(K)] for ch in range(3)]     # 3xK
    groups = [range(g, min(g+TILE, P)) for g in range(0, P, TILE)]
    out = [None]*P
    for grp in groups:
        gi = list(grp)
        dLdCg = [dLdC[p] for p in gi]; wg = [w[p] for p in gi]; ag = [alpha[p] for p in gi]
        ones  = [[1.0]*K for _ in gi]
        U     = [[1.0 if j > i else 0.0 for i in range(K)] for j in range(K)]       # KxK strict-upper
        dw    = _mm(coord, ctx, dLdCg, colorT, K)          # dLdC @ color^T   -> [Pg][K]
        dwW   = _bin(coord, ctx, dw, wg, "mul", K)         # dw . w
        suf   = _mm(coord, ctx, dwW, U, K)                 # dwW @ U          -> suffix_{j>i}
        recA  = _un(coord, ctx, ag, "reciprocal", K)       # 1/alpha
        Tv    = _bin(coord, ctx, wg, recA, "mul", K)       # T = w/alpha
        oma   = _bin(coord, ctx, ones, ag, "sub", K)       # 1 - alpha
        recOM = _un(coord, ctx, oma, "reciprocal", K)      # 1/(1-alpha)
        t1    = _bin(coord, ctx, dw, Tv, "mul", K)         # dw . T
        t2    = _bin(coord, ctx, suf, recOM, "mul", K)     # suffix . 1/(1-alpha)
        dA    = _bin(coord, ctx, t1, t2, "sub", K)         # dw.T - suffix.recip
        for r, p in enumerate(gi): out[p] = dA[r]
    return out

def main():
    K, size = 12, 16
    ctx = init_ttexalens(); coord = TensixLauncher.at(1, 2, ctx=ctx).coord
    # build all kernels once
    MM.build_for("fp32")
    SF.build_unary("reciprocal"); SF.build_binary("mul"); SF.build_binary("sub")
    r = SP.render_ondevice(coord, ctx=ctx, k=K, size=size, seed=5, verbose=False)
    P = size*size; w, alpha, color, rgb = r["w"], r["alpha"], r["color"], r["rgb"]
    rnd = random.Random(7); target = [[rnd.random() for _ in range(3)] for _ in range(P)]
    dLdC = [[2.0*(rgb[p][ch]-target[p][ch])/(P*3) for ch in range(3)] for p in range(P)]

    dA_dev = dLda_device(coord, ctx, dLdC, w, alpha, color, K)
    # exact host reference on the SAME device intermediates
    dA_ref = [[0.0]*K for _ in range(P)]
    for p in range(P):
        dwp = [sum(dLdC[p][ch]*color[i][ch] for ch in range(3)) for i in range(K)]
        S = 0.0
        for i in reversed(range(K)):
            al = alpha[p][i]; T = w[p][i]/al if al > 1e-12 else 0.0
            dA_ref[p][i] = dwp[i]*T - S/max(1.0-al, 1e-6); S += dwp[i]*w[p][i]
    scale = max(abs(dA_ref[p][i]) for p in range(P) for i in range(K)) + 1e-12
    maxabs = max(abs(dA_dev[p][i]-dA_ref[p][i]) for p in range(P) for i in range(K))
    import math
    l2 = math.sqrt(sum((dA_dev[p][i]-dA_ref[p][i])**2 for p in range(P) for i in range(K))
                   / sum(dA_ref[p][i]**2 for p in range(P) for i in range(K)))
    print(f"device dL/dalpha vs exact host: max_abs={maxabs:.2e} (scale {scale:.2e}, rel {maxabs/scale:.1%}), "
          f"L2-rel={l2:.1%}  -> {'OK' if l2 < 0.05 else 'CHECK bf16 drift'}")

if __name__ == "__main__":
    main()
