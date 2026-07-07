"""Gap 2 (multi-tile correctness): project a 3D scene into a SIZE x SIZE screen made of 16x16 tiles, BIN
Gaussians per tile, render each tile at its screen origin on a (different) Tensix worker, assemble the
full image, and compare to a host golden full render. Proves the tiled rasterizer is correct (the
120-worker parallel/resident execution is the perf follow-on)."""
import sys, math
sys.path.insert(0, "/home/starboy/bhtop/src"); sys.path.insert(0, "/home/starboy/bhtop/scratchpad")
import numpy as np
from ttexalens import init_ttexalens
from bhtop.tensix.loader import TensixLauncher, worker_coords
from bhtop.tensix import splat as SP, matmul as MM, sfpu as SF
import proj_golden as PG

N = 18; T = 16; SIZE = 32                     # 2x2 tiles of 16
R = np.eye(3); t = np.array([0.0, 0.0, 4.0]); K = (52.0, 52.0, 16.0, 16.0)   # scene fills 32x32

def scene(seed):
    rng = np.random.default_rng(seed)         # spread wide + small scales so <=16 Gaussians/tile
    return dict(mean=rng.normal(0, 1.35, (N, 3)), op=rng.uniform(0.5, 0.9, N), col=rng.uniform(0, 1, (N, 3)),
                quat=(lambda q: q/np.linalg.norm(q, axis=1, keepdims=True))(rng.normal(0, 1, (N, 4))),
                logscale=rng.normal(-1.9, 0.2, (N, 3)))

def project(P):
    g = []
    for i in range(N):
        (u, v, a, b, c, z), _ = PG.fwd(P["mean"][i], P["quat"][i], P["logscale"][i], R, t, K, need=True)
        g.append(dict(u=u, v=v, a=a, b=b, c=c, op=float(P["op"][i]), col=P["col"][i], z=z))
    return g

def gauss_eval(gp, px, py):
    a, b, c = gp["a"], gp["b"], gp["c"]
    sa = math.sqrt(max(a, 1e-8)); m12 = b/sa; m22 = math.sqrt(max(c - b*b/a, 1e-8))
    v1 = sa*(px-gp["u"]) + m12*(py-gp["v"]); v2 = m22*(py-gp["v"])
    return gp["op"] * math.exp(max(-0.5*(v1*v1+v2*v2), -60.0))

def golden(gp):
    img = np.zeros((SIZE, SIZE, 3))
    order = sorted(range(N), key=lambda i: gp[i]["z"])
    for py in range(SIZE):
        for px in range(SIZE):
            Tt = 1.0
            for i in order:
                al = gauss_eval(gp[i], px+0.0, py+0.0)
                img[py, px] += Tt*al*gp[i]["col"]; Tt *= (1.0-al)
    return img

def radius(gp):                              # 3-sigma screen radius from the 2D covariance (=inv conic)
    det = gp["a"]*gp["c"] - gp["b"]**2
    s00 = gp["c"]/det; s11 = gp["a"]/det     # Sigma2 diagonal
    return 3.0*math.sqrt(max(s00, s11, 1e-6))

def main():
    ctx = init_ttexalens()
    MM.build_for("fp32")
    for op in ("square", "exponential", "log", "log1p", "reciprocal"): SF.build_unary(op)
    for op in ("mul", "sub"): SF.build_binary(op)
    workers = [TensixLauncher.at(x, y, ctx=ctx).coord for (x, y) in [(1, 2), (2, 2), (3, 2), (4, 2)]]

    gp = project(scene(1))
    gold = golden(gp)
    out = np.zeros((SIZE, SIZE, 3))
    ntiles = SIZE // T
    wi = 0; maxk = 0
    for ty in range(ntiles):
        for tx in range(ntiles):
            ox, oy = tx*T, ty*T
            # bin: Gaussians whose 3-sigma box overlaps this tile
            members = [i for i in range(N)
                       if gp[i]["u"]+radius(gp[i]) >= ox and gp[i]["u"]-radius(gp[i]) < ox+T
                       and gp[i]["v"]+radius(gp[i]) >= oy and gp[i]["v"]-radius(gp[i]) < oy+T]
            maxk = max(maxk, len(members))
            if not members:
                continue
            assert len(members) <= 16, f"tile ({tx},{ty}) has {len(members)}>16 Gaussians (need K>16 strategy)"
            gs = [(gp[i]["u"], gp[i]["v"], gp[i]["a"], gp[i]["b"], gp[i]["c"], gp[i]["op"],
                   *[float(x) for x in gp[i]["col"]], gp[i]["z"]) for i in members]
            order = sorted(range(len(members)), key=lambda j: gs[j][9])
            coord = workers[wi % len(workers)]; wi += 1
            r = SP.render_ondevice(coord, ctx=ctx, k=len(members), size=T, gs=gs, order=order,
                                   origin=(ox, oy), prebuilt=True, verbose=False)
            tile = np.array(r["rgb"], np.float64).reshape(T, T, 3)
            out[oy:oy+T, ox:ox+T] = tile
    mse = float(((out - gold)**2).mean())
    psnr = 99 if mse < 1e-12 else 10*math.log10(1/mse)
    print(f"multi-tile render: {ntiles}x{ntiles} tiles, N={N} (max {maxk}/tile), across {len(workers)} workers")
    print(f"  assembled {SIZE}x{SIZE} vs golden full render: PSNR = {psnr:.1f} dB  -> {'PASS' if psnr>40 else 'CHECK'}")

if __name__ == "__main__":
    main()
