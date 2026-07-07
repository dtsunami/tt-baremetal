"""Gap 1 golden model — real 3D->2D camera projection + its ANALYTIC backward, host-verified.

The fused resident trainer today stores each Gaussian as 2D screen-space params [gx,gy,a,b,c,...]
where (a,b,c) is the 2D Sigma^-1 (conic) and (gx,gy) the screen mean. The render + whiten-backward
(opt_step.c) already map dL/dpsi -> dL/d(gx,gy,a,b,c). Gap 1 makes params REAL 3D
[mean(3), scale_log(3), quat(4), ...] and inserts, at the (gx,gy,a,b,c) seam:

    FORWARD  : (mean3, scale_log3, quat4, camera) --project--> (gx, gy, a, b, c) + depth
    BACKWARD : dL/d(gx,gy,a,b,c) --project_backward--> dL/d(mean3, scale_log3, quat4)

Everything downstream (Cholesky whitening psi, the eval matmul, the render, the existing
whiten-backward that produces dL/d(gx,gy,a,b,c)) is UNCHANGED. This module is the source of truth
the x280 port (cb_whiten.c forward, opt_step.c backward) is validated against.

Convention = the ttnn reference `project` (tt-splat/docs/pathclear/train3d.py), which itself matches
the (a,b,c) convention in bhtop/src/bhtop/tensix/splat.py:
    Sigma3 = R diag(exp(scale_log)^2) R^T          (R from a w-first quaternion)
    mc     = Rv @ mean + tv                          (world -> camera; Rv rows = cam basis)
    u,v    = fx*mc0/z + cx, fy*mc1/z + cy            (z = mc2, clamped >0)
    J      = [[fx/z,0,-fx*mc0/z^2],[0,fy/z,-fy*mc1/z^2]]     (affine perspective Jacobian)
    Sig2   = J (Rv Sigma3 Rv^T) J^T + 0.3*I          (EWA low-pass)
    a,b,c  = C/det, -B/det, A/det   with Sig2=[[A,B],[B,C]], det = A*C - B*B + 1e-9

Run: python3 gap1_proj_golden.py   -> validates forward vs torch and backward vs torch autograd.
"""
import numpy as np

EPS_DET = 1e-9
EPS_Z = 1e-4
BLUR = 0.3


# ---- quaternion (w,x,y,z, w-first) -> rotation, and its Jacobian ---------------------------------
def quat_to_rot(q):
    """q: (4,) raw (unnormalized). Returns (R 3x3, qn normalized, qnorm)."""
    qnorm = float(np.sqrt(q @ q))
    qn = q / qnorm
    w, x, y, z = qn
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y)],
        [2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y)],
    ])
    return R, qn, qnorm


def _dR_dqn(qn):
    """dR[j][k]/dqn[i] as a (4,3,3) array. qn=(w,x,y,z)."""
    w, x, y, z = qn
    dR = np.zeros((4, 3, 3))
    # d/dw
    dR[0] = [[0, -2 * z, 2 * y], [2 * z, 0, -2 * x], [-2 * y, 2 * x, 0]]
    # d/dx
    dR[1] = [[0, 2 * y, 2 * z], [2 * y, -4 * x, -2 * w], [2 * z, 2 * w, -4 * x]]
    # d/dy
    dR[2] = [[-4 * y, 2 * x, 2 * w], [2 * x, 0, 2 * z], [-2 * w, 2 * z, -4 * y]]
    # d/dz
    dR[3] = [[-4 * z, -2 * w, 2 * x], [2 * w, -4 * z, 2 * y], [2 * x, 2 * y, 0]]
    return dR


# ---- FORWARD ------------------------------------------------------------------------------------
def project_forward(mean, scale_log, quat, Rv, tv, fx, fy, cx, cy):
    """mean(3), scale_log(3), quat(4) raw; camera Rv(3x3) tv(3) fx fy cx cy.
    Returns (gx, gy, depth, a, b, c) and a cache for the backward pass."""
    mean = np.asarray(mean, float); scale_log = np.asarray(scale_log, float)
    quat = np.asarray(quat, float); Rv = np.asarray(Rv, float); tv = np.asarray(tv, float)
    R, qn, qnorm = quat_to_rot(quat)
    S2 = np.exp(2.0 * scale_log)                       # (exp(scale_log))^2
    Sig3 = R @ np.diag(S2) @ R.T
    mc = Rv @ mean + tv
    z_raw = mc[2]
    z = z_raw if z_raw > EPS_Z else EPS_Z
    gx = fx * mc[0] / z + cx
    gy = fy * mc[1] / z + cy
    J = np.array([[fx / z, 0.0, -fx * mc[0] / (z * z)],
                  [0.0, fy / z, -fy * mc[1] / (z * z)]])
    Sig_cam = Rv @ Sig3 @ Rv.T
    Sig2 = J @ Sig_cam @ J.T + BLUR * np.eye(2)
    A, B, C = Sig2[0, 0], Sig2[0, 1], Sig2[1, 1]
    det = A * C - B * B + EPS_DET
    a, b, c = C / det, -B / det, A / det
    cache = dict(mean=mean, scale_log=scale_log, quat=quat, Rv=Rv, fx=fx, fy=fy,
                 R=R, qn=qn, qnorm=qnorm, S2=S2, Sig3=Sig3, mc=mc, z=z, z_raw=z_raw,
                 J=J, Sig_cam=Sig_cam, A=A, B=B, C=C, det=det)
    return gx, gy, z_raw, a, b, c, cache


# ---- BACKWARD (analytic; this is what ports to opt_step.c) ---------------------------------------
def project_backward(da, db, dc, dgx, dgy, cache):
    """dL/d(a,b,c,gx,gy) -> dL/d(mean3, scale_log3, quat4)."""
    Rv, fx, fy = cache["Rv"], cache["fx"], cache["fy"]
    R, qn, qnorm, S2 = cache["R"], cache["qn"], cache["qnorm"], cache["S2"]
    Sig_cam, mc, z, z_raw = cache["Sig_cam"], cache["mc"], cache["z"], cache["z_raw"]
    J, A, B, C, det = cache["J"], cache["A"], cache["B"], cache["C"], cache["det"]
    mc0, mc1 = mc[0], mc[1]

    # (a,b,c) <- (A,B,C): exact scalar partials of a=C/det, b=-B/det, c=A/det, det=AC-B^2+eps.
    D2 = det * det
    dA = da * (-C * C / D2)     + db * (B * C / D2)                + dc * (1.0 / det - A * C / D2)
    dB = da * (2 * B * C / D2)  + db * (-1.0 / det - 2 * B * B / D2) + dc * (2 * A * B / D2)
    dC = da * (1.0 / det - A * C / D2) + db * (A * B / D2)          + dc * (-A * A / D2)
    GSig2 = np.array([[dA, dB / 2.0], [dB / 2.0, dC]])            # full symmetric 2x2

    # Sig2 = J Sig_cam J^T (+const):  GSig_cam = J^T GSig2 J ; GJ = 2 GSig2 J Sig_cam (Sig_cam sym).
    GSig_cam = J.T @ GSig2 @ J
    GJ = 2.0 * GSig2 @ J @ Sig_cam                               # (2,3)

    # accumulate dL/dmc from screen mean + from J's dependence on mc (z=mc2 when unclamped).
    dmc = np.zeros(3)
    z2, z3 = z * z, z * z * z
    dmc[0] += dgx * fx / z
    dmc[1] += dgy * fy / z
    dmc[2] += dgx * (-fx * mc0 / z2) + dgy * (-fy * mc1 / z2)
    # J00=fx/z, J02=-fx*mc0/z^2, J11=fy/z, J12=-fy*mc1/z^2
    dmc[2] += GJ[0, 0] * (-fx / z2)
    dmc[0] += GJ[0, 2] * (-fx / z2);  dmc[2] += GJ[0, 2] * (2.0 * fx * mc0 / z3)
    dmc[2] += GJ[1, 1] * (-fy / z2)
    dmc[1] += GJ[1, 2] * (-fy / z2);  dmc[2] += GJ[1, 2] * (2.0 * fy * mc1 / z3)
    if z_raw <= EPS_Z:                                           # clamp region: dz/dmc2 = 0
        # remove every mc2 contribution that flowed through z (all of them did)
        # simplest correct handling: recompute dmc[2] with z treated constant -> zero the z-path.
        # Here every dmc[2] term used z as a function of mc2, so zero it out.
        dmc[2] = 0.0

    # Sig_cam = Rv Sig3 Rv^T -> GSig3 ; Sig3 = R diag(S2) R^T.
    GSig3 = Rv.T @ GSig_cam @ Rv
    GSig3s = 0.5 * (GSig3 + GSig3.T)                            # symmetrize (defensive)
    GR = 2.0 * GSig3s @ R @ np.diag(S2)                        # Y=R D R^T, D sym
    dS2 = np.diag(R.T @ GSig3s @ R).copy()                    # dL/dS2_k
    dscale_log = dS2 * 2.0 * S2                                # S2 = exp(2*sl)

    # quat: GR -> dqn -> dq_raw (through normalization).
    dRdqn = _dR_dqn(qn)
    dqn = np.array([np.sum(GR * dRdqn[i]) for i in range(4)])
    dq_raw = (np.eye(4) - np.outer(qn, qn)) @ dqn / qnorm

    # mean: mc = Rv mean + tv -> dmean = Rv^T dmc
    dmean = Rv.T @ dmc
    return dmean, dscale_log, dq_raw


# ---- self test: forward vs torch, backward vs torch autograd -------------------------------------
def _selftest():
    import torch
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(7)

    def torch_project(mean, sl, q, Rv, tv, fx, fy, cx, cy):
        R_ = _torch_quat_to_rot(q)
        S2 = torch.exp(2.0 * sl)
        Sig3 = R_ @ torch.diag(S2) @ R_.T
        mc = Rv @ mean + tv
        z = mc[2].clamp(min=EPS_Z)
        gx = fx * mc[0] / z + cx; gy = fy * mc[1] / z + cy
        J = torch.zeros(2, 3)
        J[0, 0] = fx / z; J[0, 2] = -fx * mc[0] / z**2
        J[1, 1] = fy / z; J[1, 2] = -fy * mc[1] / z**2
        Sig2 = J @ (Rv @ Sig3 @ Rv.T) @ J.T + BLUR * torch.eye(2)
        A, B, C = Sig2[0, 0], Sig2[0, 1], Sig2[1, 1]
        det = A * C - B * B + EPS_DET
        return gx, gy, mc[2], C / det, -B / det, A / det

    def _torch_quat_to_rot(q):
        qn = q / q.norm()
        w, x, y, z = qn
        return torch.stack([
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)]),
            torch.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)]),
            torch.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])])

    # a fixed look-at-ish camera (world->cam), Gaussian in front of it
    Rv = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    tv = np.array([0.0, 0.0, 6.0]); fx, fy, cx, cy = 70.0, 70.0, 32.0, 32.0

    max_f, max_b = 0.0, 0.0
    for trial in range(200):
        mean = rng.normal(0, 1.5, 3)
        sl = np.log(0.25 + rng.random(3) * 0.5)
        q = rng.normal(0, 1, 4)
        # forward compare
        gx, gy, depth, a, b, c, cache = project_forward(mean, sl, q, Rv, tv, fx, fy, cx, cy)
        tm = torch.tensor(mean, requires_grad=True); tsl = torch.tensor(sl, requires_grad=True)
        tq = torch.tensor(q, requires_grad=True)
        tRv = torch.tensor(Rv); ttv = torch.tensor(tv)
        tgx, tgy, tdep, ta, tb, tc = torch_project(tm, tsl, tq, tRv, ttv, fx, fy, cx, cy)
        fdiff = max(abs(gx - float(tgx)), abs(gy - float(tgy)), abs(a - float(ta)),
                    abs(b - float(tb)), abs(c - float(tc)), abs(depth - float(tdep)))
        max_f = max(max_f, fdiff)
        # backward compare: random upstream grads on (a,b,c,gx,gy)
        wa, wb, wc, wgx, wgy = rng.normal(0, 1, 5)
        loss = wa * ta + wb * tb + wc * tc + wgx * tgx + wgy * tgy
        loss.backward()
        dmean, dsl, dq = project_backward(wa, wb, wc, wgx, wgy, cache)
        bdiff = max(np.max(np.abs(dmean - tm.grad.numpy())),
                    np.max(np.abs(dsl - tsl.grad.numpy())),
                    np.max(np.abs(dq - tq.grad.numpy())))
        max_b = max(max_b, bdiff)

    print(f"forward  max|golden - torch|         = {max_f:.2e}")
    print(f"backward max|analytic - autograd|    = {max_b:.2e}")
    ok = max_f < 1e-9 and max_b < 1e-7
    print("GAP1_PROJ_GOLDEN_OK" if ok else "GAP1_PROJ_GOLDEN_FAIL")
    return ok


if __name__ == "__main__":
    _selftest()
