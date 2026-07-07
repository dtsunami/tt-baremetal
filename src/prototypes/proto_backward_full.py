"""Full on-device bf16 backward chain (Tensix): from dL/dC down to dL/dpsi, dL/dop, dL/dcolor, using only
the proven primitives (matmul fp32-out + eltwise-mul + reciprocal/sub for the composite stage). Verifies
the ASSEMBLY (transposes, Ppair^T, reductions) against a host-EXACT reference computed on the SAME device
forward intermediates — so an assembly bug shows up as a LARGE error, distinct from the ~bf16 noise floor.

Chain (all in depth-sorted gaussian space; columns = sorted gaussians):
  dLdw   = dLdC @ color^T
  dLda   = composite/transmittance backward (proven stage)
  dLdop  = sum_p (dLda . ar)                      [reduce over pixels]
  dLdE   = dLda . op . ar                          [op broadcast per column]
  dLdVsq = dLdE @ Ppair^T
  dLdV   = dLdVsq . V                              [x2 folded into 2*phi below]
  dLdpsi = (2*phi)^T @ dLdV                        [reduce over pixels]
  dLdcol = w^T @ dLdC                              [reduce over pixels]
"""
import sys, math, random
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import splat as SP, matmul as MM, sfpu as SF

TILE = SP.TILE; pad, take = SP._pad32, SP._take
def L2(a, b):
    n = sum((a[i]-b[i])**2 for i in range(len(a))); d = sum(x*x for x in b)
    return math.sqrt(n/d) if d > 0 else (0.0 if n == 0 else 9.9)
def flat(m): return [v for r in m for v in r]
def T_(m):   return [list(c) for c in zip(*m)]

def mm(coord, ctx, A, B, rows, cols):
    return take(MM.run_matmul(coord, ctx=ctx, a=pad(A), b=pad(B), out_format="fp32",
                              prebuilt=True, verbose=False)["c_dev"], rows, cols)
def un(coord, ctx, M, op, cols):
    r, _ = SF.run_unary(coord, pad(M), ctx=ctx, op=op, prebuilt=True, timeout=3.0); return take(r, len(M), cols)
def bn(coord, ctx, A, B, op, cols):
    r, _ = SF.run_binary(coord, pad(A), pad(B), ctx=ctx, op=op, prebuilt=True, timeout=3.0); return take(r, len(A), cols)

def dLda_group(coord, ctx, dLdCg, wg, ag, colorT, U, K):
    ones = [[1.0]*K for _ in wg]
    dw   = mm(coord, ctx, dLdCg, colorT, len(wg), K)
    dwW  = bn(coord, ctx, dw, wg, "mul", K)
    suf  = mm(coord, ctx, dwW, U, len(wg), K)
    recA = un(coord, ctx, ag, "reciprocal", K)
    Tv   = bn(coord, ctx, wg, recA, "mul", K)
    oma  = bn(coord, ctx, ones, ag, "sub", K)
    recOM= un(coord, ctx, oma, "reciprocal", K)
    t1   = bn(coord, ctx, dw, Tv, "mul", K)
    t2   = bn(coord, ctx, suf, recOM, "mul", K)
    return bn(coord, ctx, t1, t2, "sub", K), dw

def main():
    K, size = 12, 16
    ctx = init_ttexalens(); coord = TensixLauncher.at(1, 2, ctx=ctx).coord
    MM.build_for("fp32")
    for op in ("reciprocal",): SF.build_unary(op)
    for op in ("mul", "sub"): SF.build_binary(op)

    r = SP.render_ondevice(coord, ctx=ctx, k=K, size=size, seed=5, verbose=False)
    P = size*size
    w, alpha, ar, v, color, order, gs = r["w"], r["alpha"], r["ar"], r["v"], r["color"], r["order"], r["gs"]
    gso = [gs[i] for i in order]
    psi, Ppair, Dop, Dnop, Mcomb, color2 = SP._consts(gso, K)
    op = [Dop[i][i] for i in range(K)]
    colorT = T_(color); PpairT = T_(Ppair)                      # PpairT: K x 2K
    U = [[1.0 if j > i else 0.0 for i in range(K)] for j in range(K)]
    pixels = [(x, y) for y in range(size) for x in range(size)]
    phi2 = [[2.0*x, 2.0*y, 2.0] for (x, y) in pixels]           # 2*phi (x2 folded here)

    rnd = random.Random(7); target = [[rnd.random() for _ in range(3)] for _ in range(P)]
    dLdC = [[2.0*(r["rgb"][p][ch]-target[p][ch])/(P*3) for ch in range(3)] for p in range(P)]

    groups = [list(range(g, min(g+TILE, P))) for g in range(0, P, TILE)]
    dLdpsi = [[0.0]*(2*K) for _ in range(3)]
    dLdcol = [[0.0]*3 for _ in range(K)]
    dLdop  = [0.0]*K
    for gi in groups:
        dLdCg = [dLdC[p] for p in gi]; wg = [w[p] for p in gi]; ag = [alpha[p] for p in gi]
        arg = [ar[p] for p in gi]; opB = [op[:] for _ in gi]; phig = [phi2[p] for p in gi]
        dLda, dw = dLda_group(coord, ctx, dLdCg, wg, ag, colorT, U, K)
        # dLdop += sum_p dLda . ar
        dae   = bn(coord, ctx, dLda, arg, "mul", K)
        red   = mm(coord, ctx, [[1.0]*len(gi)], dae, 1, K)      # ones[1xP] @ dae[PxK] -> [1xK]
        for i in range(K): dLdop[i] += red[0][i]
        # dLdE = dLda . op . ar = dae . opB
        dLdE  = bn(coord, ctx, dae, opB, "mul", K)
        # dLdVsq = dLdE @ Ppair^T  (PxK @ Kx2K -> Px2K)
        dLdVsq = mm(coord, ctx, dLdE, PpairT, len(gi), 2*K)
        # dLdV = dLdVsq . V
        Vg    = [v[p] for p in gi]
        dLdV  = bn(coord, ctx, dLdVsq, Vg, "mul", 2*K)
        # dLdpsi += (2phi)^T @ dLdV   (3xP @ Px2K -> 3x2K)
        part  = mm(coord, ctx, T_(phig), dLdV, 3, 2*K)
        for c in range(3):
            for m in range(2*K): dLdpsi[c][m] += part[c][m]
        # dLdcol += w^T @ dLdC   (KxP @ Px3 -> Kx3)
        cpart = mm(coord, ctx, T_(wg), dLdCg, K, 3)
        for i in range(K):
            for ch in range(3): dLdcol[i][ch] += cpart[i][ch]

    # ---- host-EXACT reference on the SAME device intermediates ----
    Rw = [[sum(dLdC[p][ch]*color[i][ch] for ch in range(3)) for i in range(K)] for p in range(P)]
    Rda = [[0.0]*K for _ in range(P)]
    for p in range(P):
        S = 0.0
        for i in reversed(range(K)):
            al = alpha[p][i]; Tt = w[p][i]/al if al > 1e-12 else 0.0
            Rda[p][i] = Rw[p][i]*Tt - S/max(1.0-al, 1e-6); S += Rw[p][i]*w[p][i]
    Rop = [sum(Rda[p][i]*ar[p][i] for p in range(P)) for i in range(K)]
    RE  = [[Rda[p][i]*op[i]*ar[p][i] for i in range(K)] for p in range(P)]
    RVsq= [[sum(RE[p][i]*PpairT[i][m] for i in range(K)) for m in range(2*K)] for p in range(P)]
    RV  = [[RVsq[p][m]*v[p][m] for m in range(2*K)] for p in range(P)]
    Rpsi= [[sum(phi2[p][c]*RV[p][m] for p in range(P)) for m in range(2*K)] for c in range(3)]
    Rcol= [[sum(w[p][i]*dLdC[p][ch] for p in range(P)) for ch in range(3)] for i in range(K)]

    print(f"FULL on-device bf16 backward vs host-exact (same fwd intermediates), K={K} {size}x{size}:")
    print(f"  dL/dpsi   L2 = {L2(flat(dLdpsi), flat(Rpsi)):.1%}   (3x{2*K})")
    print(f"  dL/dop    L2 = {L2(dLdop, Rop):.1%}   ({K})")
    print(f"  dL/dcolor L2 = {L2(flat(dLdcol), flat(Rcol)):.1%}   ({K}x3)")
    print("  (assembly correct if these are bf16-noise-level ~1-30%, NOT >100%)")

if __name__ == "__main__":
    main()
