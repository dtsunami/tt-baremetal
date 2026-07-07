"""Golden model: gsplat-style camera projection of a 3D Gaussian to a 2D screen conic, FORWARD +
ANALYTIC BACKWARD, grad-checked vs finite differences. This is the reference the x280 projection kernel
(Gap 1, extending cb_whiten) must match. Pure numpy, host-only.

Forward: (mean[3] world, quat[4] wxyz, logscale[3], camera R[3x3],t[3], intrinsics fx,fy,cx,cy)
  -> screen (u,v), 2D inverse-covariance conic (a,b,c) [same (a,b,c,gx,gy) cb_whiten already consumes], depth z.
Backward: dL/d(u,v,a,b,c) -> dL/d(mean, quat, logscale).
"""
import numpy as np

BLUR = 0.3   # 2D covariance dilation (low-pass / AA); matches a small isotropic add

def quat_to_R(q):
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]])

def fwd(mean, quat, logscale, R, t, K, need=False):
    fx, fy, cx, cy = K
    mu = R @ mean + t                       # camera-space mean
    x, y, z = mu
    u = fx * x / z + cx; v = fy * y / z + cy
    Rq = quat_to_R(quat); S = np.diag(np.exp(logscale))
    M = Rq @ S; Sig3 = M @ M.T              # world 3D covariance
    Sc = R @ Sig3 @ R.T                      # camera 3D covariance
    J = np.array([[fx/z, 0, -fx*x/z**2], [0, fy/z, -fy*y/z**2]])
    Sig2 = J @ Sc @ J.T + BLUR*np.eye(2)     # 2D covariance (+dilation)
    det = Sig2[0,0]*Sig2[1,1] - Sig2[0,1]**2
    a = Sig2[1,1]/det; b = -Sig2[0,1]/det; c = Sig2[0,0]/det   # conic = inv(Sig2)
    out = (u, v, a, b, c, z)
    if need:
        return out, dict(mu=mu, x=x, y=y, z=z, Rq=Rq, S=S, M=M, Sig3=Sig3, Sc=Sc, J=J, Sig2=Sig2, det=det)
    return out

def bwd(mean, quat, logscale, R, t, K, g, im):
    """g = (du,dv,da,db,dc). Returns dL/dmean[3], dL/dquat[4], dL/dlogscale[3]."""
    fx, fy, cx, cy = K
    du, dv, da, db, dc = g
    x, y, z, J, Sc, Sig2, det = im['x'], im['y'], im['z'], im['J'], im['Sc'], im['Sig2'], im['det']
    Rq, S, M = im['Rq'], im['S'], im['M']
    # a=r/det, b=-q/det, c=p/det with p=Sig2[0,0], q=Sig2[0,1], r=Sig2[1,1], det=pr-q^2. Differentiate
    # a,b,c wrt (p,q,r) explicitly; build dSig2 with the shared off-diagonal q HALVED so the symmetric
    # matrix chain (dSc=J^T dSig2 J) counts it once. (The naive -conic dConic conic double-counts q.)
    p, q, r = Sig2[0,0], Sig2[0,1], Sig2[1,1]; d2 = det*det
    dp = da*(-r*r/d2)        + db*(q*r/d2)          + dc*(1.0/det - p*r/d2)
    dq = da*(2*r*q/d2)       + db*(-1.0/det - 2*q*q/d2) + dc*(2*p*q/d2)
    dr = da*(1.0/det - r*p/d2) + db*(q*p/d2)        + dc*(-p*p/d2)
    dSig2 = np.array([[dp, 0.5*dq], [0.5*dq, dr]])             # symmetric, off-diag halved
    # Sig2 = J Sc J^T  ->  dSc = J^T dSig2 J ; dJ = dSig2 J Sc^T + dSig2^T J Sc = 2 dSig2 J Sc (sym)
    dSc = J.T @ dSig2 @ J
    dJ = 2.0 * dSig2 @ J @ Sc
    # Sc = R Sig3 R^T -> dSig3 = R^T dSc R
    dSig3 = R.T @ dSc @ R
    # Sig3 = M M^T -> dM = (dSig3 + dSig3^T) M = 2 dSig3 M (sym)
    dM = 2.0 * dSig3 @ M
    # M = Rq S -> dRq = dM S^T ; dS = Rq^T dM
    dRq = dM @ S.T
    dS = Rq.T @ dM
    dlogscale = np.array([dS[i, i] * np.exp(logscale[i]) for i in range(3)])   # S=diag(exp(logscale))
    dquat = _dquat(quat, dRq)
    # J depends on (x,y,z): J = [[fx/z,0,-fx*x/z^2],[0,fy/z,-fy*y/z^2]]
    dxyz_fromJ = np.array([
        dJ[0,2]*(-fx/z**2) ,                                     # dJ02/dx
        dJ[1,2]*(-fy/z**2) ,                                     # dJ12/dy
        dJ[0,0]*(-fx/z**2) + dJ[0,2]*(2*fx*x/z**3) + dJ[1,1]*(-fy/z**2) + dJ[1,2]*(2*fy*y/z**3)])
    # u,v depend on (x,y,z): u=fx x/z + cx, v=fy y/z + cy
    dxyz_fromUV = np.array([du*fx/z, dv*fy/z, du*(-fx*x/z**2) + dv*(-fy*y/z**2)])
    dmu = dxyz_fromJ + dxyz_fromUV
    dmean = R.T @ dmu                                            # mu = R mean + t
    return dmean, dquat, dlogscale

def _dquat(q, dR):
    """d(loss)/d(quat) given d(loss)/d(R) (3x3), via finite-diff of quat_to_R (robust, small)."""
    eps = 1e-6; out = np.zeros(4)
    for i in range(4):
        qp = q.copy(); qp[i] += eps; qm = q.copy(); qm[i] -= eps
        out[i] = np.sum(dR * (quat_to_R(qp) - quat_to_R(qm)) / (2*eps))
    return out

def _grad_check():
    rng = np.random.default_rng(0)
    mean = rng.normal(0, 0.4, 3) + np.array([0, 0, 3.0])
    quat = rng.normal(0, 1, 4); quat /= np.linalg.norm(quat)
    logscale = rng.normal(-1.5, 0.3, 3)
    ang = 0.2; R = quat_to_R(np.array([np.cos(ang/2), 0, np.sin(ang/2), 0])); t = np.array([0.1, -0.1, 0.2])
    K = (12.0, 12.0, 8.0, 8.0)                                  # ~16px tile
    out, im = fwd(mean, quat, logscale, R, t, K, need=True)
    g = tuple(rng.normal(0, 1, 5))                              # random upstream dL/d(u,v,a,b,c)
    def L(m, q, ls):
        o = fwd(m, q, ls, R, t, K)
        return sum(g[i] * o[i] for i in range(5))
    dmean, dquat, dlogscale = bwd(mean, quat, logscale, R, t, K, g, im)
    eps = 1e-6; ok = True
    for name, an, base, setter in (
        ("mean", dmean, mean, lambda v, i: L(_bump(mean, i, v), quat, logscale)),
        ("quat", dquat, quat, lambda v, i: L(mean, _bump(quat, i, v), logscale)),
        ("logscale", dlogscale, logscale, lambda v, i: L(mean, quat, _bump(logscale, i, v)))):
        for i in range(len(base)):
            num = (setter(eps, i) - setter(-eps, i)) / (2*eps)
            rel = abs(an[i]-num) / (abs(num)+1e-7)
            ok = ok and rel < 1e-3
            print(f"  {name}[{i}] analytic={an[i]:+.5e} numeric={num:+.5e} rel={rel:.1e} {'OK' if rel<1e-3 else 'BAD'}")
    print("projection fwd+bwd grad-check:", "CORRECT" if ok else "MISMATCH")

def _bump(a, i, e):
    b = a.copy(); b[i] += e; return b

if __name__ == "__main__":
    _grad_check()
