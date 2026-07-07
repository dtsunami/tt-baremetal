"""Verify the promoted splat.backward_ondevice reproduces the leaf grads vs host-exact (same fwd)."""
import sys, math, random
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import splat as SP

def L2(a, b):
    n = sum((a[i]-b[i])**2 for i in range(len(a))); d = sum(x*x for x in b); return math.sqrt(n/d) if d else 0.0
def flat(m): return [x for r in m for x in r]

K, size = 12, 16
ctx = init_ttexalens(); coord = TensixLauncher.at(1, 2, ctx=ctx).coord
r = SP.render_ondevice(coord, ctx=ctx, k=K, size=size, seed=5, verbose=False)
P = size*size
rnd = random.Random(7); target = [[rnd.random() for _ in range(3)] for _ in range(P)]
dLdC = [[2.0*(r["rgb"][p][ch]-target[p][ch])/(P*3) for ch in range(3)] for p in range(P)]
b = SP.backward_ondevice(coord, r, dLdC, ctx=ctx, verbose=True)

# host-exact reference on same fwd intermediates
w, alpha, ar, v, color, order, gs = r["w"], r["alpha"], r["ar"], r["v"], r["color"], r["order"], r["gs"]
gso = [gs[i] for i in order]; _, Ppair, Dop, _, _, _ = SP._consts(gso, K); op = [Dop[i][i] for i in range(K)]
PpairT = [list(c) for c in zip(*Ppair)]
pixels = [(x, y) for y in range(size) for x in range(size)]; phi2 = [[2.0*x, 2.0*y, 2.0] for (x, y) in pixels]
Rw = [[sum(dLdC[p][ch]*color[i][ch] for ch in range(3)) for i in range(K)] for p in range(P)]
Rda = [[0.0]*K for _ in range(P)]
for p in range(P):
    S = 0.0
    for i in reversed(range(K)):
        al = alpha[p][i]; Tt = w[p][i]/al if al > 1e-12 else 0.0
        Rda[p][i] = Rw[p][i]*Tt - S/max(1.0-al, 1e-6); S += Rw[p][i]*w[p][i]
Rop = [sum(Rda[p][i]*ar[p][i] for p in range(P)) for i in range(K)]
RE = [[Rda[p][i]*op[i]*ar[p][i] for i in range(K)] for p in range(P)]
RVsq = [[sum(RE[p][i]*PpairT[i][m] for i in range(K)) for m in range(2*K)] for p in range(P)]
RV = [[RVsq[p][m]*v[p][m] for m in range(2*K)] for p in range(P)]
Rpsi = [[sum(phi2[p][c]*RV[p][m] for p in range(P)) for m in range(2*K)] for c in range(3)]
Rcol = [[sum(w[p][i]*dLdC[p][ch] for p in range(P)) for ch in range(3)] for i in range(K)]
print(f"  dL/dpsi   L2 = {L2(flat(b['dLdpsi']), flat(Rpsi)):.1%}")
print(f"  dL/dop    L2 = {L2(b['dLdop'], Rop):.1%}")
print(f"  dL/dcolor L2 = {L2(flat(b['dLdcolor']), flat(Rcol)):.1%}")
