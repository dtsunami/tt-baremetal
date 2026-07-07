"""Option 0: attribute the device dL/dalpha error. (a) sweep SFPU reciprocal accuracy over the actual
alpha range, and via the exp(-log(1-a)) alternative; (b) per-stage device-vs-exact error, where each
stage's 'exact' is computed on the SAME device inputs that fed it — so each number is that op's own error.
One representative 32-pixel group. Short per-op timeouts so a bad kernel can't run away."""
import sys, math, random
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import splat as SP, matmul as MM, sfpu as SF

TILE = SP.TILE; pad, take = SP._pad32, SP._take
def L2(a, b):
    n = sum((a[i]-b[i])**2 for i in range(len(a))); d = sum(x*x for x in b)
    return math.sqrt(n/d) if d > 0 else 0.0
def flat(rows): return [v for r in rows for v in r]

def mm(coord, ctx, A, B, cols):
    return take(MM.run_matmul(coord, ctx=ctx, a=pad(A), b=pad(B), out_format="fp32",
                              prebuilt=True, verbose=False)["c_dev"], len(A), cols)
def un(coord, ctx, M, op, cols):
    r, _ = SF.run_unary(coord, pad(M), ctx=ctx, op=op, prebuilt=True, timeout=3.0); return take(r, len(M), cols)
def bn(coord, ctx, A, B, op, cols):
    r, _ = SF.run_binary(coord, pad(A), pad(B), ctx=ctx, op=op, prebuilt=True, timeout=3.0); return take(r, len(A), cols)

def main():
    K, size = 16, 16
    ctx = init_ttexalens(); coord = TensixLauncher.at(1, 2, ctx=ctx).coord
    MM.build_for("fp32")
    for op in ("reciprocal", "log", "exponential"): SF.build_unary(op)
    for op in ("mul", "sub"): SF.build_binary(op)

    # (a) reciprocal accuracy over the real alpha range ---------------------------------------------
    P = 32
    alphas = [0.02 + 0.96*i/(P*K-1) for i in range(P*K)]              # (0.02 .. 0.98)
    ag = [alphas[g*K:(g+1)*K] for g in range(P)]
    rec_d = un(coord, ctx, ag, "reciprocal", K)                       # device 1/alpha
    rel = [abs(rec_d[p][i]-1.0/ag[p][i])/(1.0/ag[p][i]) for p in range(P) for i in range(K)]
    worst = max(range(len(rel)), key=lambda k: rel[k]); wa = alphas[worst]
    print(f"(a) SFPU reciprocal 1/alpha over (0.02,0.98): mean_rel={sum(rel)/len(rel):.2%} "
          f"max_rel={max(rel):.2%} @alpha={wa:.3f}")
    # 1/(1-alpha): SFPU reciprocal vs exp(-log(1-alpha))
    oma = [[1.0-ag[p][i] for i in range(K)] for p in range(P)]        # host 1-alpha (exact input)
    recOM_d = un(coord, ctx, oma, "reciprocal", K)
    la = un(coord, ctx, oma, "log", K)                               # log(1-alpha)
    nla = [[-la[p][i] for i in range(K)] for p in range(P)]
    recOM_exp = un(coord, ctx, nla, "exponential", K)               # exp(-log(1-alpha))
    ref = [[1.0/oma[p][i] for i in range(K)] for p in range(P)]
    e_rec = L2(flat(recOM_d), flat(ref)); e_exp = L2(flat(recOM_exp), flat(ref))
    print(f"(a) 1/(1-alpha):  SFPU-reciprocal L2={e_rec:.2%}   vs   exp(-log(1-a)) L2={e_exp:.2%}")

    # (b) per-stage attribution on one real 32-pixel group -----------------------------------------
    r = SP.render_ondevice(coord, ctx=ctx, k=K, size=size, seed=5, verbose=False)
    w, alpha, color, rgb = r["w"], r["alpha"], r["color"], r["rgb"]
    rnd = random.Random(7); target = [[rnd.random() for _ in range(3)] for _ in range(size*size)]
    gi = list(range(32))
    dLdCg = [[2.0*(rgb[p][ch]-target[p][ch])/(size*size*3) for ch in range(3)] for p in gi]
    wg = [w[p] for p in gi]; agp = [alpha[p] for p in gi]; ones = [[1.0]*K for _ in gi]
    colorT = [[color[i][ch] for i in range(K)] for ch in range(3)]
    U = [[1.0 if j > i else 0.0 for i in range(K)] for j in range(K)]

    dw   = mm(coord, ctx, dLdCg, colorT, K)
    dwW  = bn(coord, ctx, dw, wg, "mul", K)
    suf  = mm(coord, ctx, dwW, U, K)
    recA = un(coord, ctx, agp, "reciprocal", K)
    Tv   = bn(coord, ctx, wg, recA, "mul", K)
    omag = bn(coord, ctx, ones, agp, "sub", K)
    recOM= un(coord, ctx, omag, "reciprocal", K)
    t1   = bn(coord, ctx, dw, Tv, "mul", K)
    t2   = bn(coord, ctx, suf, recOM, "mul", K)
    dA   = bn(coord, ctx, t1, t2, "sub", K)

    def he(dev, fn):     # this op's own error: device out vs exact(fn) on the device inputs
        ref = fn(); return L2(flat(dev), flat(ref))
    print("(b) per-stage own-error (device op vs exact on same device inputs):")
    print(f"    dw = dLdC@colorT        : {he(dw,   lambda: [[sum(dLdCg[p][ch]*color[i][ch] for ch in range(3)) for i in range(K)] for p in range(32)]):.2%}")
    print(f"    dwW = dw . w   (mul)     : {he(dwW,  lambda: [[dw[p][i]*wg[p][i] for i in range(K)] for p in range(32)]):.2%}")
    print(f"    suf = dwW @ U            : {he(suf,  lambda: [[sum(dwW[p][j] for j in range(i+1,K)) for i in range(K)] for p in range(32)]):.2%}")
    print(f"    recA = 1/alpha  (SFPU)   : {he(recA, lambda: [[1.0/agp[p][i] for i in range(K)] for p in range(32)]):.2%}  <==")
    print(f"    Tv = w . recA  (mul)     : {he(Tv,   lambda: [[wg[p][i]*recA[p][i] for i in range(K)] for p in range(32)]):.2%}")
    print(f"    oma = 1 - alpha (sub)    : {he(omag, lambda: [[1.0-agp[p][i] for i in range(K)] for p in range(32)]):.2%}")
    print(f"    recOM = 1/oma  (SFPU)    : {he(recOM,lambda: [[1.0/omag[p][i] for i in range(K)] for p in range(32)]):.2%}  <==")
    print(f"    t1 = dw . Tv   (mul)     : {he(t1,   lambda: [[dw[p][i]*Tv[p][i] for i in range(K)] for p in range(32)]):.2%}")
    print(f"    t2 = suf . recOM (mul)   : {he(t2,   lambda: [[suf[p][i]*recOM[p][i] for i in range(K)] for p in range(32)]):.2%}")
    print(f"    dA = t1 - t2   (sub)     : {he(dA,   lambda: [[t1[p][i]-t2[p][i] for i in range(K)] for p in range(32)]):.2%}")
    # overall vs fully-exact + bias
    exact = [[0.0]*K for _ in range(32)]
    for p in range(32):
        dwp = [sum(dLdCg[p][ch]*color[i][ch] for ch in range(3)) for i in range(K)]; S = 0.0
        for i in reversed(range(K)):
            al = agp[p][i]; T = wg[p][i]/al if al > 1e-12 else 0.0
            exact[p][i] = dwp[i]*T - S/max(1.0-al, 1e-6); S += dwp[i]*wg[p][i]
    diffs = [dA[p][i]-exact[p][i] for p in range(32) for i in range(K)]
    sc = max(abs(exact[p][i]) for p in range(32) for i in range(K)) + 1e-12
    print(f"(b) OVERALL dA device vs exact: L2={L2(flat(dA),flat(exact)):.1%}  "
          f"mean_signed={sum(diffs)/len(diffs)/sc:+.1%} of scale (bias check)")

if __name__ == "__main__":
    main()
