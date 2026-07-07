"""Option 5: does a ~27% L2, near-zero-mean error in dL/dalpha (the measured device character) break
training? Host-only: rerun train_geometry's full-Gaussian fit with a controllable relative noise injected
on dL/dalpha, compare PSNR trajectory to the clean baseline. If noisy still converges ~= clean, the
on-device bf16 dL/dalpha is good enough and we proceed; else precision (fp32/x280/reformulate) is required."""
import sys, math, random
sys.path.insert(0, "/home/starboy/bhtop/src")
from bhtop.het import train_geometry as TG   # forward/scene/whiten/loss_of, K, size, PIX, P

K, size, PIX, P = TG.K, TG.size, TG.PIX, TG.P

def backward_noisy(prm, order, interm, dLdC, noise, bias, rng):
    """train_geometry.backward, but each per-pixel dL/dalpha (dLdal) is scaled by (1+bias+noise*N(0,1))
    to mimic the measured device error (L2~0.27, mean~-0.008)."""
    gx, gy, a, b, c, op, col = prm
    W = [TG.whiten(a[i], b[i], c[i]) for i in range(K)]
    g = {k: [0.0]*K for k in ("gx","gy","a","b","c","op")}; gcol = [[0.0]*3 for _ in range(K)]
    for p, (px, py) in enumerate(PIX):
        dLdw = {}
        for i in order:
            dLdw[i] = sum(dLdC[p][ch]*col[i][ch] for ch in range(3))
            for ch in range(3): gcol[i][ch] += interm["w"][p][i]*dLdC[p][ch]
        S = 0.0
        for i in reversed(order):
            al = interm["al"][p][i]; T = interm["T"][p][i]; w = interm["w"][p][i]
            dLdal = dLdw[i]*T - S/max(1.0-al, 1e-6)
            dLdal *= (1.0 + bias + noise*rng.gauss(0, 1))          # inject device-like error
            S += dLdw[i]*w
            ar = interm["ar"][p][i]
            g["op"][i] += dLdal*ar
            dLdE = (dLdal*op[i])*ar
            v1 = interm["v1"][p][i]; v2 = interm["v2"][p][i]
            dLdv1 = dLdE*(-v1); dLdv2 = dLdE*(-v2)
            sa, m12, m22 = W[i]
            dsa = dLdv1*(px-gx[i]); dm12 = dLdv1*(py-gy[i]); dm22 = dLdv2*(py-gy[i])
            g["gx"][i] += dLdv1*(-sa); g["gy"][i] += dLdv1*(-m12) + dLdv2*(-m22)
            ai, bi = a[i], b[i]
            g["a"][i] += dsa*(0.5/sa) + dm12*(-0.5*bi/(ai*sa)) + dm22*((bi*bi/(ai*ai))/(2*m22))
            g["b"][i] += dm12*(1.0/sa) + dm22*(-bi/(ai*m22)); g["c"][i] += dm22*(1.0/(2*m22))
    return g, gcol

def train(noise, bias, seed=22):
    rng = random.Random(1234)
    tprm, z = TG.scene(11); order = sorted(range(K), key=lambda i: z[i]); target = TG.forward(tprm, order)
    prm, _ = TG.scene(seed)
    m = {k:[0.0]*K for k in ("gx","gy","a","b","c","op")}; v = {k:[0.0]*K for k in ("gx","gy","a","b","c","op")}
    mc = [[0.0]*3 for _ in range(K)]; vc = [[0.0]*3 for _ in range(K)]
    b1, b2, eps = 0.9, 0.999, 1e-8; lr = {"gx":0.15,"gy":0.15,"a":2e-3,"b":2e-3,"c":2e-3,"op":0.02}
    traj = []
    for step in range(60):
        C, im = TG.forward(prm, order, need_interm=True); L = TG.loss_of(C, target)
        dLdC = [[2.0*(C[p][ch]-target[p][ch])/(P*3) for ch in range(3)] for p in range(P)]
        g, gcol = backward_noisy(prm, order, im, dLdC, noise, bias, rng)
        for name, idx in [("gx",0),("gy",1),("a",2),("b",3),("c",4),("op",5)]:
            for i in range(K):
                m[name][i] = b1*m[name][i]+(1-b1)*g[name][i]; v[name][i] = b2*v[name][i]+(1-b2)*g[name][i]**2
                mh = m[name][i]/(1-b1**(step+1)); vh = v[name][i]/(1-b2**(step+1))
                prm[idx][i] -= lr[name]*mh/(math.sqrt(vh)+eps)
            if name in ("a","c"):
                for i in range(K): prm[idx][i] = max(prm[idx][i], 1e-3)
            if name == "op":
                for i in range(K): prm[idx][i] = min(0.99, max(0.05, prm[idx][i]))
        for i in range(K):
            for ch in range(3):
                mc[i][ch] = b1*mc[i][ch]+(1-b1)*gcol[i][ch]; vc[i][ch] = b2*vc[i][ch]+(1-b2)*gcol[i][ch]**2
                mh = mc[i][ch]/(1-b1**(step+1)); vh = vc[i][ch]/(1-b2**(step+1))
                prm[6][i][ch] = min(1.0, max(0.0, prm[6][i][ch]-0.1*mh/(math.sqrt(vh)+eps)))
        if step in (0, 19, 39, 59):
            traj.append((step, 99 if L < 1e-12 else 10*math.log10(1/L)))
    return traj

for label, noise, bias in [("clean (exact dL/dalpha)", 0.0, 0.0),
                           ("device-like: 27% L2, -0.8% bias", 0.27, -0.008),
                           ("pessimistic: 40% L2, -3% bias", 0.40, -0.03)]:
    tr = train(noise, bias)
    print(f"{label:34s} : " + "  ".join(f"step{s:2d}={p:4.1f}dB" for s, p in tr))
