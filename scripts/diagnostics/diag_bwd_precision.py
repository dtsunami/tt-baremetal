"""Host-only attribution of the bf16 error in the on-device dL/dalpha chain: replay the exact chain with
bf16 rounding applied per-stage, toggling individual stages to fp32, to see which stage dominates the
~12% device error. No device — this tells us which on-device-backward option is worth the silicon effort."""
import sys, math, random
sys.path.insert(0, "/home/starboy/bhtop/src")
from bhtop.tensix import matmul as MM

def bf16(x):                       # round one scalar to bf16 (same as device staging)
    return MM.unpack_bf16_words(MM.pack_bf16_words([float(x), 0.0]))[0]
def rnd(v, on): return [bf16(x) for x in v] if on else list(v)

K = 16
# realistic per-pixel column vectors: alpha in (0,1) incl near-1, dw arbitrary sign, w=T*alpha
def gen(seed):
    r = random.Random(seed)
    alpha = [r.uniform(0.02, 0.98) for _ in range(K)]
    color = [[r.random() for _ in range(3)] for _ in range(K)]
    dLdC = [r.uniform(-1, 1) for _ in range(3)]
    T = 1.0; w = []
    for i in range(K): w.append(T*alpha[i]); T *= (1-alpha[i])
    return alpha, w, color, dLdC

def chain(alpha, w, color, dLdC, *, r_dw, r_elt, r_recip, r_sub):
    # dw = dLdC @ color^T (matmul: fp32 acc, bf16 inputs) -> optionally round output
    dw = rnd([sum(dLdC[ch]*color[i][ch] for ch in range(3)) for i in range(K)], r_dw)
    dwW = rnd([dw[i]*w[i] for i in range(K)], r_elt)
    suf = rnd([sum(dwW[j] for j in range(i+1, K)) for i in range(K)], r_dw)   # dwW@U (matmul)
    recA = rnd([1.0/alpha[i] for i in range(K)], r_recip)
    Tv  = rnd([w[i]*recA[i] for i in range(K)], r_elt)
    oma = rnd([1.0-alpha[i] for i in range(K)], r_elt)
    recOM = rnd([1.0/oma[i] for i in range(K)], r_recip)
    t1 = rnd([dw[i]*Tv[i] for i in range(K)], r_elt)
    t2 = rnd([suf[i]*recOM[i] for i in range(K)], r_elt)
    return rnd([t1[i]-t2[i] for i in range(K)], r_sub)

def L2(a, b):
    num = sum((a[i]-b[i])**2 for i in range(len(a))); den = sum(b[i]**2 for i in range(len(b)))
    return math.sqrt(num/den) if den > 0 else 0.0

scenes = [gen(s) for s in range(400)]
def avg(cfg):
    tot = 0.0
    for al, w, co, d in scenes:
        ref = chain(al, w, co, d, r_dw=0, r_elt=0, r_recip=0, r_sub=0)
        got = chain(al, w, co, d, **cfg)
        tot += L2(got, ref)
    return tot/len(scenes)

print("stage-attribution of on-device dL/dalpha error (mean L2 over 400 column-scenes):")
print(f"  all bf16 (device model)          : {avg(dict(r_dw=1,r_elt=1,r_recip=1,r_sub=1)):.1%}")
print(f"  bf16, but final SUBTRACT in fp32  : {avg(dict(r_dw=1,r_elt=1,r_recip=1,r_sub=0)):.1%}")
print(f"  bf16, but RECIPROCALS in fp32     : {avg(dict(r_dw=1,r_elt=1,r_recip=0,r_sub=1)):.1%}")
print(f"  bf16, but recip + subtract fp32   : {avg(dict(r_dw=1,r_elt=1,r_recip=0,r_sub=0)):.1%}")
print(f"  only matmul outputs bf16, rest fp32: {avg(dict(r_dw=1,r_elt=0,r_recip=0,r_sub=0)):.1%}")
print(f"  only eltwise bf16, rest fp32       : {avg(dict(r_dw=0,r_elt=1,r_recip=0,r_sub=0)):.1%}")
