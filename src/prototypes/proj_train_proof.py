"""Gap-1 proof: the grad-checked camera projection makes a REAL 3D scene TRAIN through the REAL device
render/backward. Chain per step: project (golden fwd) -> (u,v,a,b,c) -> Tensix render_ondevice -> loss ->
Tensix backward_ondevice -> dL/dpsi -> whiten-backward (host, from opt_step.c) -> projection backward
(golden) -> Adam on 3D (mean,quat,logscale,op,color). If PSNR climbs, projection is correct end-to-end
and Gap 1 is proven; the fully-on-device port (x280 3D params + proj kernels) is the follow-on."""
import sys, math
sys.path.insert(0, "/home/starboy/bhtop/src"); sys.path.insert(0, "/home/starboy/bhtop/scratchpad")
import numpy as np
from ttexalens import init_ttexalens
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import splat as SP, matmul as MM, sfpu as SF
import proj_golden as PG

N, SIZE = 8, 16
# a camera looking down -z toward the origin; intrinsics chosen so the scene lands in the 16x16 tile
R = np.eye(3); t = np.array([0.0, 0.0, 4.0]); K = (26.0, 26.0, 8.0, 8.0)

def rand_scene(seed):
    rng = np.random.default_rng(seed)
    mean = rng.normal(0, 0.7, (N, 3)); mean[:, 2] += 0.0            # around origin, in front of cam
    quat = rng.normal(0, 1, (N, 4)); quat /= np.linalg.norm(quat, axis=1, keepdims=True)
    logscale = rng.normal(-1.6, 0.25, (N, 3))
    op = rng.uniform(0.4, 0.9, N)
    col = rng.uniform(0, 1, (N, 3))
    return dict(mean=mean, quat=quat, logscale=logscale, op=op, col=col)

def project(P):
    """3D scene -> bhtop 2D scene tuples (gx,gy,a,b,c,op,c0,c1,c2,z), + per-Gaussian fwd intermediates."""
    gs, ims = [], []
    for i in range(N):
        (u, v, a, b, c, z), im = PG.fwd(P["mean"][i], P["quat"][i], P["logscale"][i], R, t, K, need=True)
        gs.append((u, v, a, b, c, float(P["op"][i]), *[float(x) for x in P["col"][i]], z))
        ims.append(im)
    return gs, ims

def whiten_bwd(a, b, c, gx, gy, dpsi):
    """dL/dpsi (d_sa,d_m12,d_tx,d_m22,d_ty) -> dL/d(a,b,c,u,v). Mirrors opt_step.c's whiten-backward."""
    d_sa, d_m12, d_tx, d_m22, d_ty = dpsi
    sa = math.sqrt(max(a, 1e-8)); m12 = b / sa
    tt = max(c - b * b / a, 1e-8); m22 = math.sqrt(tt)
    Dsa = d_sa + d_tx * (-gx); Dm12 = d_m12 + d_tx * (-gy); Dm22 = d_m22 + d_ty * (-gy)
    g_u = d_tx * (-sa); g_v = d_tx * (-m12) + d_ty * (-m22)
    g_a = Dsa * (0.5 / sa) + Dm12 * (-0.5 * b / (a * sa)) + Dm22 * ((b * b / (a * a)) / (2 * m22))
    g_b = Dm12 * (1.0 / sa) + Dm22 * (-b / (a * m22))
    g_c = Dm22 * (1.0 / (2 * m22))
    return g_u, g_v, g_a, g_b, g_c

def main():
    ctx = init_ttexalens(); coord = TensixLauncher.at(1, 2, ctx=ctx).coord
    MM.build_for("fp32")
    for op in ("square", "exponential", "log", "log1p", "reciprocal"): SF.build_unary(op)
    for op in ("mul", "sub"): SF.build_binary(op)

    tgt = rand_scene(1); gsT, _ = project(tgt)
    orderT = sorted(range(N), key=lambda i: gsT[i][9])
    target = np.array(SP.render_ondevice(coord, ctx=ctx, k=N, size=SIZE, gs=gsT, order=orderT,
                                         prebuilt=True, verbose=False)["rgb"], np.float64).reshape(SIZE, SIZE, 3)

    P = rand_scene(2)                                                # perturbed init to train
    keys = ("mean", "quat", "logscale", "op", "col")
    m = {k: np.zeros_like(P[k], np.float64) for k in keys}; v = {k: np.zeros_like(P[k], np.float64) for k in keys}
    lr = {"mean": 0.03, "quat": 0.02, "logscale": 0.02, "op": 0.02, "col": 0.05}
    b1, b2, eps = 0.9, 0.999, 1e-8; Pp = SIZE * SIZE
    print(f"Gap-1 proof: train a real 3D scene (N={N}) through camera projection + device render/backward")
    for step in range(1, 41):
        gs, ims = project(P)
        order = sorted(range(N), key=lambda i: gs[i][9])
        fwd = SP.render_ondevice(coord, ctx=ctx, k=N, size=SIZE, gs=gs, order=order, prebuilt=True, verbose=False)
        rgb = np.array(fwd["rgb"], np.float64).reshape(SIZE, SIZE, 3)
        mse = float(((rgb - target) ** 2).mean())
        dLdC = [[2.0 * (rgb.reshape(-1, 3)[p][ch] - target.reshape(-1, 3)[p][ch]) / (Pp * 3) for ch in range(3)] for p in range(Pp)]
        bw = SP.backward_ondevice(coord, fwd, dLdC, ctx=ctx, prebuilt=True, verbose=False)
        dLdpsi, dLdop, dLdcol = bw["dLdpsi"], bw["dLdop"], bw["dLdcolor"]     # SORTED order
        g = {k: np.zeros_like(P[k], np.float64) for k in keys}
        for si in range(N):                                          # si = sorted slot -> original id
            oi = order[si]
            a, b_, c = gs[oi][2], gs[oi][3], gs[oi][4]; gx, gy = gs[oi][0], gs[oi][1]
            dpsi = (dLdpsi[0][2*si], dLdpsi[1][2*si], dLdpsi[2][2*si], dLdpsi[1][2*si+1], dLdpsi[2][2*si+1])
            g_u, g_v, g_a, g_b, g_c = whiten_bwd(a, b_, c, gx, gy, dpsi)
            dmean, dquat, dls = PG.bwd(P["mean"][oi], P["quat"][oi], P["logscale"][oi], R, t, K,
                                       (g_u, g_v, g_a, g_b, g_c), ims[oi])
            g["mean"][oi] += dmean; g["quat"][oi] += dquat; g["logscale"][oi] += dls
            g["op"][oi] += dLdop[si]; g["col"][oi] += np.array(dLdcol[si])
        for k in keys:                                              # Adam
            m[k] = b1*m[k] + (1-b1)*g[k]; v[k] = b2*v[k] + (1-b2)*g[k]**2
            mh = m[k]/(1-b1**step); vh = v[k]/(1-b2**step)
            P[k] -= lr[k]*mh/(np.sqrt(vh)+eps)
        P["op"] = np.clip(P["op"], 0.05, 0.99); P["col"] = np.clip(P["col"], 0, 1)
        if step in (1, 5, 10, 20, 30, 40):
            psnr = 99 if mse < 1e-12 else 10*math.log10(1/mse)
            print(f"  step {step:2d}: mse={mse:.5f} PSNR={psnr:5.2f} dB")

if __name__ == "__main__":
    main()
