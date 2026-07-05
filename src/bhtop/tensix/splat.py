"""
tensix.splat — Gaussian-splatting FORWARD on the BARE-METAL Tensix substrate (no ttnn, no tt-metal).

The int-matmul-eval raster (tt-splat scratchpad/proto_intmm_raster.py) reformulated onto bhtop's own
bare-metal MVMUL substrate ([[tensix-frame-as-ints]]). The Gaussian exponent field for a tile is
    E[pixel, gauss] = -0.5 * ( v1^2 + v2^2 ),   v = phi @ psi   (whitened surrogate, Sig^-1 = M^T M)
where phi[pixel, :] = [px, py, 1] (small exact ints) and psi[:, gauss] are the whitening coeffs. psi is
split into two int8 limbs (hi, lo, base 128) so the two contraction matmuls are exact int8->int32 on the
FPU integer datapath — the datapath proved bit-exact in tensix.matmul. We run those two matmuls
BARE-METAL over exalens (cold-booted TRISCs, matmul_perf L1_TO_L1, int8/HiFi4), reconstruct v in fp32 on
the host, and get E. exp/opacity -> alpha and the alpha-composite are the next rungs.

Whitening (matches proto): Sig^-1 = [[a,b],[b,c]]; sa=sqrt(a), m12=b/sa, m22=sqrt(c-b^2/a);
    v1 = sa*(px-gx) + m12*(py-gy),   v2 = m22*(py-gy)   =>   v1^2+v2^2 = a dx^2 + 2b dx dy + c dy^2.

Pure-Python host math (no numpy/torch — bhtop venv has neither): a tile is <=32 pixels x <=16 Gaussians
(one 32x32 int8 MVMUL: 32 pixel rows, 2*K<=32 interleaved v1/v2 columns).
"""
import math

from . import matmul as MM
from . import sfpu as SF

TILE = 32
NLIMB = 128            # limb base: psi/s = hi*128 + lo, hi,lo in int8 range


# ---- scene (pure python) -------------------------------------------------------------------------
def scene(k=16, seed=5, span=16.0):
    """k Gaussians with random pose/scale/opacity in a `span`x`span` region. Deterministic LCG."""
    st = [seed & 0x7FFFFFFF]

    def rnd():
        st[0] = (st[0] * 1103515245 + 12345) & 0x7FFFFFFF
        return st[0] / 0x7FFFFFFF

    g = []
    for _ in range(k):
        gx = rnd() * span
        gy = rnd() * span
        s1 = 1.5 + rnd() * 3.0
        s2 = 1.5 + rnd() * 3.0
        th = rnd() * math.pi
        op = 0.3 + rnd() * 0.6
        ct, sn = math.cos(th), math.sin(th)
        # Sigma = R diag(s1^2,s2^2) R^T  ; Sigma^-1 entries a,b,c
        s1s, s2s = s1 * s1, s2 * s2
        # Sigma = [[ct^2 s1s + sn^2 s2s, ct sn (s1s-s2s)], [.., sn^2 s1s + ct^2 s2s]]
        S00 = ct * ct * s1s + sn * sn * s2s
        S01 = ct * sn * (s1s - s2s)
        S11 = sn * sn * s1s + ct * ct * s2s
        det = S00 * S11 - S01 * S01
        a, b, c = S11 / det, -S01 / det, S00 / det
        g.append((gx, gy, a, b, c, op))
    return g


def tile_pixels(w=16, h=2):
    """w*h (<=32) pixel coords (px,py), row-major — the pixels this MVMUL tile evaluates."""
    return [(px, py) for py in range(h) for px in range(w)]


# ---- build phi / psi limbs (host) ----------------------------------------------------------------
def build_psi_limbs(g):
    """Gaussian-side inputs (pixel-independent): hi[32][32], lo[32][32] int8 limbs + per-column scale[32].
    Column 2i = v1 coeffs (sa, m12, const), col 2i+1 = v2 coeffs (0, m22, const). Reused across all rows."""
    k = len(g)
    assert 2 * k <= TILE
    psi = [[0.0] * (2 * k) for _ in range(3)]
    for i, (gx, gy, a, b, c, op) in enumerate(g):
        sa = math.sqrt(max(a, 1e-8))
        m12 = b / sa
        m22 = math.sqrt(max(c - b * b / a, 0.0))
        psi[0][2 * i] = sa
        psi[1][2 * i] = m12
        psi[2][2 * i] = -(sa * gx + m12 * gy)
        psi[1][2 * i + 1] = m22
        psi[2][2 * i + 1] = -(m22 * gy)
    scale = [0.0] * TILE
    hi = [[0] * TILE for _ in range(TILE)]
    lo = [[0] * TILE for _ in range(TILE)]
    for col in range(2 * k):
        amax = max(abs(psi[r][col]) for r in range(3))
        s = max(amax, 1e-12) / (127 * NLIMB)
        scale[col] = s
        for r in range(3):
            q = round(psi[r][col] / s)
            h = round(q / NLIMB)
            hi[r][col] = h
            lo[r][col] = q - h * NLIMB
    return hi, lo, scale, k


def build_phi(px_list):
    """Pixel-side input: phi[32][32], row p = pixel (px,py) -> cols [px, py, 1]."""
    phi = [[0] * TILE for _ in range(TILE)]
    for p, (px, py) in enumerate(px_list):
        phi[p][0] = int(px)
        phi[p][1] = int(py)
        phi[p][2] = 1
    return phi


def build_inputs(g, px_list):
    """(phi, hi, lo, scale, K) — convenience wrapper used by eval_tile."""
    hi, lo, scale, k = build_psi_limbs(g)
    return build_phi(px_list), hi, lo, scale, k


def _flat(m):
    return [m[r][c] for r in range(TILE) for c in range(TILE)]


def golden_E(g, px_list):
    """Reference exponent E[pixel][gauss] = -0.5(a dx^2 + 2b dx dy + c dy^2)."""
    E = [[0.0] * len(g) for _ in range(len(px_list))]
    for p, (px, py) in enumerate(px_list):
        for i, (gx, gy, a, b, c, op) in enumerate(g):
            dx, dy = px - gx, py - gy
            E[p][i] = -0.5 * (a * dx * dx + 2 * b * dx * dy + c * dy * dy)
    return E


# ---- the RUN: bare-metal int-matmul Gaussian eval ------------------------------------------------
def eval_tile(coord, *, ctx, device_id=0, k=16, w=16, h=2, seed=5, verbose=True):
    """Evaluate the Gaussian exponent field E for one tile on the BARE-METAL Tensix int8 MVMUL, and
    check it against the pure-Python golden. Returns a dict with the eval error + limb-matmul exactness."""
    g = scene(k=k, seed=seed)
    px_list = tile_pixels(w, h)
    P = len(px_list)
    phi, hi, lo, scale, K = build_inputs(g, px_list)

    # two exact int8->int32 limb matmuls on the FPU integer datapath, bare-metal over exalens
    r_hi = MM.run_matmul(coord, ctx=ctx, device_id=device_id, a=_flat(phi), b=_flat(hi),
                         out_format="int32", verbose=False)
    r_lo = MM.run_matmul(coord, ctx=ctx, device_id=device_id, a=_flat(phi), b=_flat(lo),
                         out_format="int32", verbose=False)
    Vhi, Vlo = r_hi["c_dev"], r_lo["c_dev"]        # each 1024 = [32 pixels][32 cols] row-major

    # host reference for the integer matmuls (proves the device limb-matmul is bit-exact)
    Ghi = MM.matmul_golden(_flat(phi), _flat(hi))
    Glo = MM.matmul_golden(_flat(phi), _flat(lo))
    limb_exact = (Vhi == Ghi) and (Vlo == Glo)

    # reconstruct v (fp32) and E on host; compare to golden
    Eg = golden_E(g, px_list)
    max_relE, num, den = 0.0, 0.0, 0.0
    wl1_num, wl1_den = 0.0, 0.0
    for p in range(P):
        for i in range(K):
            c1, c2 = 2 * i, 2 * i + 1
            v1 = (NLIMB * Vhi[p * TILE + c1] + Vlo[p * TILE + c1]) * scale[c1]
            v2 = (NLIMB * Vhi[p * TILE + c2] + Vlo[p * TILE + c2]) * scale[c2]
            E = -0.5 * (v1 * v1 + v2 * v2)
            Er = Eg[p][i]
            # contribution weight = alpha_ref (exp(E)*op) so the metric is image-relevant
            op = g[i][5]
            wref = math.exp(max(Er, -60.0)) * op
            wl1_num += abs(math.exp(max(E, -60.0)) * op - wref)
            wl1_den += wref
            if abs(Er) > 1e-3:
                max_relE = max(max_relE, abs(E - Er) / (abs(Er) + 1e-6))
    wl1 = wl1_num / (wl1_den + 1e-9)
    ok = limb_exact and r_hi["kernel_complete"] and r_lo["kernel_complete"]
    res = {"ok": ok, "limb_matmul_exact": limb_exact, "pixels": P, "gaussians": K,
           "eval_rel_max": max_relE, "alpha_weighted_L1": wl1,
           "kernel_complete": r_hi["kernel_complete"] and r_lo["kernel_complete"], "coord": str(coord)}
    if verbose:
        print(f"[splat.eval] {P}px x {K}gauss  bare-metal int8 MVMUL (phi@hi, phi@lo)")
        print(f"[splat.eval] limb-matmul bit-exact vs host int matmul: {limb_exact}")
        print(f"[splat.eval] eval rel-E max={max_relE:.3e}  alpha-weighted-L1={wl1:.3e}"
              f"  (proto CPU ref ~3.3e-4)  -> {'PASS' if ok and wl1 < 5e-3 else 'CHECK'}")
    return res


# ---- full-tile forward RENDER (bare-metal eval + front-to-back composite) -------------------------
def scene_rgb(k=16, seed=5, span=32.0):
    """Scene with color + depth: [(gx,gy,a,b,c,op, r,g,b, z)]. Gaussians spread over a `span` tile."""
    base = scene(k=k, seed=seed, span=span)
    st = [(seed * 2 + 1) & 0x7FFFFFFF]

    def rnd():
        st[0] = (st[0] * 1103515245 + 12345) & 0x7FFFFFFF
        return st[0] / 0x7FFFFFFF

    out = []
    for (gx, gy, a, b, c, op) in base:
        out.append((gx, gy, a, b, c, op, rnd(), rnd(), rnd(), rnd()))
    return out


def _composite(E_by_gauss, gs, order):
    """Front-to-back over `order`: E_by_gauss[i][pixel] -> RGB[pixel][3]. Returns (rgb, T_final)."""
    P = len(E_by_gauss[0])
    rgb = [[0.0, 0.0, 0.0] for _ in range(P)]
    T = [1.0] * P
    for i in order:
        op = gs[i][5]
        col = gs[i][6:9]
        Erow = E_by_gauss[i]
        for p in range(P):
            a = op * math.exp(max(Erow[p], -60.0))
            if a <= 1e-5:
                continue
            w = T[p] * a
            rgb[p][0] += w * col[0]
            rgb[p][1] += w * col[1]
            rgb[p][2] += w * col[2]
            T[p] *= (1.0 - a)
    return rgb, T


def _golden_render(gs, size):
    """Exact pure-Python forward render of a size x size tile -> RGB[P][3] (P = size*size)."""
    order = sorted(range(len(gs)), key=lambda i: gs[i][9])       # front-to-back by z
    px_list = [(px, py) for py in range(size) for px in range(size)]
    E = []
    for (gx, gy, a, b, c, op, *_rest) in gs:
        row = []
        for (px, py) in px_list:
            dx, dy = px - gx, py - gy
            row.append(-0.5 * (a * dx * dx + 2 * b * dx * dy + c * dy * dy))
        E.append(row)
    return _composite(E, gs, order)[0]


def render_tile(coord, *, ctx, device_id=0, k=16, size=32, seed=5, verbose=True):
    """Forward-render a size x size RGB tile: Gaussian EVAL on the bare-metal int8 MVMUL (per pixel-row
    tile, one prebuilt kernel), exp/composite on host. Compare to the exact golden render (PSNR).
    size <= 32, k <= 16 (one 32x32 int8 MVMUL per pixel-row: 32 pixels x 2K columns)."""
    assert size <= TILE and 2 * k <= TILE
    gs = scene_rgb(k=k, seed=seed, span=float(size))
    hi, lo, scale, K = build_psi_limbs([g[:6] for g in gs])
    hi_f, lo_f = _flat(hi), _flat(lo)

    MM.build_for("int32")                          # compile the int8 kernel ONCE
    order = sorted(range(K), key=lambda i: gs[i][9])

    # E_by_gauss[i][pixel], pixel = py*size + px
    E = [[0.0] * (size * size) for _ in range(K)]
    limb_exact = True
    for py in range(size):
        phi = build_phi([(px, py) for px in range(size)])     # size (<=32) pixels this row
        phi_f = _flat(phi)
        Vhi = MM.run_matmul(coord, ctx=ctx, device_id=device_id, a=phi_f, b=hi_f,
                            out_format="int32", prebuilt=True, verbose=False)["c_dev"]
        Vlo = MM.run_matmul(coord, ctx=ctx, device_id=device_id, a=phi_f, b=lo_f,
                            out_format="int32", prebuilt=True, verbose=False)["c_dev"]
        if limb_exact:
            limb_exact = (Vhi == MM.matmul_golden(phi_f, hi_f)) and (Vlo == MM.matmul_golden(phi_f, lo_f))
        for px in range(size):
            pix = py * size + px
            for i in range(K):
                c1, c2 = 2 * i, 2 * i + 1
                v1 = (NLIMB * Vhi[px * TILE + c1] + Vlo[px * TILE + c1]) * scale[c1]
                v2 = (NLIMB * Vhi[px * TILE + c2] + Vlo[px * TILE + c2]) * scale[c2]
                E[i][pix] = -0.5 * (v1 * v1 + v2 * v2)

    rgb = _composite(E, gs, order)[0]
    gold = _golden_render(gs, size)
    # PSNR over the RGB tile
    mse = sum((rgb[p][ch] - gold[p][ch]) ** 2 for p in range(size * size) for ch in range(3))
    mse /= (size * size * 3)
    psnr = 99.0 if mse < 1e-12 else 10.0 * math.log10(1.0 / mse)
    res = {"ok": limb_exact and psnr >= 40.0, "size": size, "gaussians": K,
           "limb_matmul_exact": limb_exact, "psnr_db": psnr, "mse": mse, "rgb": rgb,
           "golden": gold, "coord": str(coord)}
    if verbose:
        print(f"[splat.render] {size}x{size} tile, {K} Gaussians — bare-metal int8-MVMUL eval + "
              f"host exp/composite")
        print(f"[splat.render] limb-matmul bit-exact: {limb_exact} | vs golden render PSNR = "
              f"{psnr:.1f} dB (mse={mse:.2e}) -> {'PASS' if res['ok'] else 'CHECK'}")
    return res


# ---- FULLY on-device forward (every op on MVMUL + SFPU; host only builds constants + moves L1 tiles) --
# The serial front-to-back composite is reformulated as matmuls-with-constant-matrices so it maps onto
# the two proven vector engines only (no eltwise-binary kernel needed):
#   V=phi@psi -> Vsq=square(V) -> E=Vsq@Ppair(-0.5 pair-sum) -> ar=exp(E)
#   alpha=ar@diag(op), -alpha=ar@diag(-op)  (per-Gaussian opacity as a diagonal matmul)
#   la=log1p(-alpha), lpa=log(alpha) -> logw=[la|lpa]@[Stri;I]  (prefix-sum AND +log(alpha) in one mm)
#   w=exp(logw) -> C=w@color .  Host sim (bf16) ~56 dB; on silicon ~55 dB.
def _pad32(rows_cols):
    m = [[0.0] * TILE for _ in range(TILE)]
    for r, row in enumerate(rows_cols):
        for c, v in enumerate(row):
            m[r][c] = float(v)
    return [m[r][c] for r in range(TILE) for c in range(TILE)]


def _take(flat, rows, cols):
    return [[flat[r * TILE + c] for c in range(cols)] for r in range(rows)]


def _consts(gso, K):
    """Constant matrices (depend only on the depth-sorted Gaussians)."""
    def whiten(g):
        gx, gy, a, b, c, op = g[:6]; sa = math.sqrt(max(a, 1e-8)); m12 = b / sa
        m22 = math.sqrt(max(c - b * b / a, 0.0)); return [sa, m12, -(sa*gx+m12*gy)], [0.0, m22, -(m22*gy)]
    psi = [[0.0]*(2*K) for _ in range(3)]
    for i in range(K):
        w1, w2 = whiten(gso[i])
        for r in range(3): psi[r][2*i] = w1[r]; psi[r][2*i+1] = w2[r]
    Ppair = [[0.0]*K for _ in range(2*K)]
    for i in range(K): Ppair[2*i][i] = -0.5; Ppair[2*i+1][i] = -0.5
    Dop  = [[(gso[i][5] if i == j else 0.0) for j in range(K)] for i in range(K)]
    Dnop = [[(-gso[i][5] if i == j else 0.0) for j in range(K)] for i in range(K)]
    Mcomb = [[(1.0 if r < c else 0.0) for c in range(K)] for r in range(2*K)]
    for i in range(K): Mcomb[K+i][i] = 1.0                 # rows 0..K-1 = strict-upper, rows K..2K-1 = I
    color = [[gso[i][6], gso[i][7], gso[i][8]] for i in range(K)]
    return psi, Ppair, Dop, Dnop, Mcomb, color


def render_ondevice(coord, *, ctx, device_id=0, k=16, size=16, seed=5, order=None, gs=None,
                    ring=None, prebuilt=False, verbose=True):
    """FULLY on-device forward render of a size x size tile. Every arithmetic op runs on the bare-metal
    Tensix MVMUL + SFPU; the host only builds constant matrices and shuttles L1 tiles between stages.
    Stage-major (build each kernel once, loop all 32-pixel groups). Returns RGB + PSNR vs golden.

    order: depth-sort order (front-to-back) to composite in. Default None -> host sort. Pass an order
    computed elsewhere (e.g. the x280 depth-sort in the het-compute split) to drive the composite from
    another engine. gs: optional pre-built scene (else scene_rgb(k,seed,span=size)).
    ring: (worker_x, worker_y, psi_gddr, dop_gddr, dnop_gddr, color_gddr) — when set, the four
    order-dependent dense operands are NOT staged by the host; instead each is NoC-read from the
    x280-produced tilized tile in shared GDDR straight into PERF_INPUT_B (zero host Gaussian-data relay).
    The static operands (Ppair, Mcomb) and pixel coords stay host-side."""
    assert 2 * k <= TILE
    if gs is None:
        gs = scene_rgb(k=k, seed=seed, span=float(size))
    if order is None:
        order = sorted(range(k), key=lambda i: gs[i][9])
    gso = [gs[i] for i in order]
    psi, Ppair, Dop, Dnop, Mcomb, color = _consts(gso, k)
    K = k

    pixels = [(x, y) for y in range(size) for x in range(size)]
    groups = [pixels[i:i + TILE] for i in range(0, len(pixels), TILE)]     # <=32 px each
    phis = [[[float(x), float(y), 1.0] for (x, y) in g] for g in groups]

    def mm_all(As, B, cols):     # matmul each group's A by shared host-staged B; prebuilt matmul kernel
        return [_take(MM.run_matmul(coord, ctx=ctx, device_id=device_id, a=_pad32(A), b=_pad32(B),
                                    out_format="fp32", prebuilt=True, verbose=False)["c_dev"],
                      len(A), cols) for A in As]

    # DENSE OPERANDS VIA THE RING: NoC-read the x280-produced tilized operand straight into PERF_INPUT_B,
    # then matmul all groups against it (b_prestaged) — the host never stages this operand.
    _bm = None
    if ring is not None:
        from .baremetal import BareMetal, bm_coord
        _bm = BareMetal(ring[0], ring[1], ctx=ctx, device_id=device_id)
        _bm.build("nocread")

    def mm_ring(As, gddr, host_b, cols):
        _bm.run(_bm.build("nocread"), params=[bm_coord(8, 3), gddr, 2048, MM.PERF_INPUT_B])
        return [_take(MM.run_matmul(coord, ctx=ctx, device_id=device_id, a=_pad32(A), b=_pad32(host_b),
                                    out_format="fp32", prebuilt=True, b_prestaged=True, verbose=False)["c_dev"],
                      len(A), cols) for A in As]

    def sfpu_all(tiles, op, cols):   # build op once, apply to each group's tile
        SF.build_unary(op)
        out = []
        for t in tiles:
            r, _ = SF.run_unary(coord, _pad32(t), ctx=ctx, device_id=device_id, op=op, prebuilt=True)
            out.append(_take(r, len(t), cols))
        return out

    if not prebuilt:
        MM.build_for("fp32")                                 # matmul kernel (once; skip when prebuilt)
    V   = (mm_ring(phis, ring[2], psi, 2*K) if ring else mm_all(phis, psi, 2*K))   # 1  V = phi @ psi
    Vsq = sfpu_all(V, "square", 2*K)                         # 2  square
    E   = mm_all(Vsq, Ppair, K)                              # 3  sum-of-squares * -0.5 (static Ppair)
    ar  = sfpu_all(E, "exponential", K)                      # 4  exp -> alpha_raw
    alpha  = (mm_ring(ar, ring[3], Dop, K) if ring else mm_all(ar, Dop, K))        # 5  * opacity (diag)
    nalpha = (mm_ring(ar, ring[4], Dnop, K) if ring else mm_all(ar, Dnop, K))      # 6  * -opacity
    lpa = sfpu_all(alpha, "log", K)                          # 7  log(alpha)
    la  = sfpu_all(nalpha, "log1p", K)                       # 8  log1p(-alpha) = log(1-alpha)
    G   = [[la[g][p][c] for c in range(K)] + [lpa[g][p][c] for c in range(K)]
           for g in range(len(groups)) for p in range(len(groups[g]))]
    # regroup G back into per-group tiles [P,2K]
    Gg, idx = [], 0
    for g in groups:
        Gg.append(G[idx:idx + len(g)]); idx += len(g)
    logw = mm_all(Gg, Mcomb, K)                              # 9  prefix-sum + log(alpha) (static Mcomb)
    w    = sfpu_all(logw, "exponential", K)                  # 10 exp -> transmittance-weighted alpha
    Cg   = (mm_ring(w, ring[5], color, 3) if ring else mm_all(w, color, 3))        # 11 @ color -> RGB

    rgb = [row for g in Cg for row in g]                     # [size*size][3]
    _flat = lambda gg: [row for grp in gg for row in grp]
    w_flat     = _flat(w)                                    # [P][K]  transmittance·alpha
    alpha_flat = _flat(alpha)                                # [P][K]  alpha = op·ar (composite backward)
    ar_flat    = _flat(ar)                                   # [P][K]  exp(E)  (dL/dE = dL/dalpha·op·ar)
    v_flat     = _flat(V)                                    # [P][2K] whitened field v1,v2 interleaved
    gold = _golden_render(gs, size)
    mse = sum((rgb[p][ch] - gold[p][ch]) ** 2 for p in range(size*size) for ch in range(3)) / (size*size*3)
    psnr = 99.0 if mse < 1e-12 else 10.0 * math.log10(1.0 / mse)
    res = {"ok": psnr >= 40.0, "size": size, "gaussians": K, "groups": len(groups),
           "psnr_db": psnr, "mse": mse, "rgb": rgb, "golden": gold, "w": w_flat,
           "alpha": alpha_flat, "ar": ar_flat, "v": v_flat, "order": list(order),
           "gs": gs, "color": color, "coord": str(coord)}
    if verbose:
        print(f"[splat.ondevice] {size}x{size}, {K} Gaussians, {len(groups)} pixel-groups — "
              f"FULLY on-device (6 MVMUL + 5 SFPU stages, no host arithmetic)")
        print(f"[splat.ondevice] vs golden PSNR = {psnr:.1f} dB  -> {'PASS' if res['ok'] else 'CHECK'}")
    return res
