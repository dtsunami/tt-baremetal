# SPDX-License-Identifier: Apache-2.0
"""
crt_matmul.py — bit-exact GOLDEN REFERENCE for the CRT / quarter-square integer matrix multiply.

Pure Python, zero deps. This is the software truth model that any hardware kernel (x280 RVV, Tensix,
or RTL) must match bit-for-bit. Explicit loops on purpose — they mirror what the silicon does, not
what a BLAS does. The whole pipeline, modeled faithfully (not just `A @ B`):

  1. CRT forward   : each int -> residues mod a pairwise-coprime set (default {5,7,8}).
  2. residue matmul: per modulus m, C_m = (Aw_m @ Ad_m) mod m, where every scalar product is done
                     by the QUARTER-SQUARE lookup  a*b = q[a+b] - q[|a-b|],  q[n] = floor(n^2/4)
                     (the exact identity the hardware LUT implements), and the K-deep dot product
                     accumulates *mod m* (small accumulators — no big intermediate ever exists).
  3. CRT inverse   : reconstruct C mod N (N = prod of moduli) via  C = (sum_i coeff_i * r_i) mod N.

KEY PROPERTY (and the design-space knob): the result is EXACT iff the true result < N = prod(m_i).
With {5,7,8}, N=280 — exact for an 8-bit / mod-280 output; a full non-wrapping K=32 int8 matmul peaks
near 2.08M and needs prod(m_i) > 2^21 (a few more small coprime moduli). The reconstruction
coefficients are derived generally, so swapping the modulus set is a one-line change.

Run directly for the validation report:  python crt/crt_matmul.py
"""
from __future__ import annotations
import random
from math import gcd

DEFAULT_MODULI = (5, 7, 8)


# --------------------------------------------------------------------------- quarter-square multiply
def quarter_square_lut(m: int) -> list[int]:
    """q[n] = floor(n^2 / 4) for n in [0, 2m).  Domain covers a+b (max 2m-2) and |a-b| (max m-1)."""
    return [(n * n) // 4 for n in range(2 * m)]


def qs_mul_mod(a: int, b: int, m: int, q: list[int] | None = None) -> int:
    """(a*b) mod m for residues a,b in [0,m), via the quarter-square LUT.

    a*b = q[a+b] - q[|a-b|] is EXACT for non-negative integers: when a+b is odd, a-b is odd too and
    the +1/4 floor terms cancel in the difference; when even, both squares are multiples of 4."""
    if q is None:
        q = quarter_square_lut(m)
    return (q[a + b] - q[abs(a - b)]) % m


# --------------------------------------------------------------------------------------------- CRT
def crt_coeffs(moduli=DEFAULT_MODULI):
    """General CRT reconstruction: returns (N, coeffs) s.t. value = (sum_i coeffs[i]*r_i) mod N.
    coeff_i = M_i * (M_i^-1 mod m_i),  M_i = N / m_i.  For {5,7,8} this yields [56,120,105], N=280."""
    moduli = tuple(int(m) for m in moduli)
    for i in range(len(moduli)):
        for j in range(i + 1, len(moduli)):
            if gcd(moduli[i], moduli[j]) != 1:
                raise ValueError(f"moduli {moduli[i]} and {moduli[j]} are not coprime")
    N = 1
    for m in moduli:
        N *= m
    coeffs = []
    for m in moduli:
        M = N // m
        y = pow(M % m, -1, m)          # modular inverse of M mod m
        coeffs.append(M * y)
    return N, coeffs


def csd(c: int):
    """Canonical signed-digit (non-adjacent form) of a non-negative constant c.
    Returns a list of (sign, shift) with the MINIMUM number of nonzero terms, so a constant
    multiply x*c becomes a shift-add/sub network: x*c = sum(sign * (x << shift)).
    This is the 'park the constant / wire it with bitshifts' core — no general multiplier."""
    terms, sh = [], 0
    while c:
        if c & 1:
            d = 2 - (c & 3)        # +1 or -1 (NAF: never two adjacent nonzero digits)
            terms.append((d, sh))
            c -= d
        c >>= 1
        sh += 1
    return terms


def shift_add_mul(x: int, c: int) -> int:
    """x*c via shifts and adds/subs only (CSD network). Bit-identical to x*c."""
    return sum(d * (x << sh) for d, sh in csd(c))


def crt_reconstruct_shiftadd(residues, moduli=DEFAULT_MODULI) -> int:
    """CRT inverse using ONLY shift-add constant multiplies (the hardware-faithful path)."""
    N, coeffs = crt_coeffs(moduli)
    return sum(shift_add_mul(r, c) for c, r in zip(coeffs, residues)) % N


def shiftadd_cost(moduli=DEFAULT_MODULI):
    """Per-coefficient shift-add term counts (adds = terms-1) — the design-space cost metric."""
    _, coeffs = crt_coeffs(moduli)
    return [(c, len(csd(c)), len(csd(c)) - 1) for c in coeffs]


def pattern_matrix(rows, cols, kind, hi=256):
    """Deterministic test matrix shared bit-for-bit with the x280 C kernel (no PRNG to sync).
    kind 0 = weights Aw, kind 1 = data Ad. Cheap affine pattern, full 8-bit range."""
    if kind == 0:
        return [[(i * 7 + k * 13 + 1) % hi for k in range(cols)] for i in range(rows)]
    return [[(k * 5 + j * 11 + 3) % hi for j in range(cols)] for k in range(rows)]


def crt_forward(x: int, moduli=DEFAULT_MODULI) -> list[int]:
    """Pack one integer into its residues."""
    return [x % m for m in moduli]


def crt_reconstruct(residues, moduli=DEFAULT_MODULI) -> int:
    """Inverse transform residues -> value mod N."""
    N, coeffs = crt_coeffs(moduli)
    return sum(c * r for c, r in zip(coeffs, residues)) % N


# ----------------------------------------------------------------------------------- the matmuls
def rns_matmul(Aw, Ad, moduli=DEFAULT_MODULI):
    """CRT/quarter-square matmul C = Aw @ Ad, computed entirely in the residue number system.
    Returns C mod N (N = prod(moduli)) — exact when the true result < N.  Aw is IxK, Ad is KxJ."""
    I, K, J = len(Aw), len(Aw[0]), len(Ad[0])
    luts = {m: quarter_square_lut(m) for m in moduli}
    N, coeffs = crt_coeffs(moduli)
    C = [[0] * J for _ in range(I)]
    for i in range(I):
        for j in range(J):
            value = 0
            for mi, m in enumerate(moduli):       # one residue lane at a time
                q = luts[m]
                acc = 0
                for k in range(K):                # K-deep dot product, mod-m accumulation
                    a = Aw[i][k] % m
                    b = Ad[k][j] % m
                    acc = (acc + qs_mul_mod(a, b, m, q)) % m
                value += coeffs[mi] * acc          # fold this lane into the reconstruction
            C[i][j] = value % N
    return C


def reference_matmul(Aw, Ad):
    """Plain exact integer matmul (the ground truth)."""
    I, K, J = len(Aw), len(Aw[0]), len(Ad[0])
    return [[sum(Aw[i][k] * Ad[k][j] for k in range(K)) for j in range(J)] for i in range(I)]


# --------------------------------------------------------------------------------------- helpers
def _rand_matrix(rng, rows, cols, hi):
    return [[rng.randrange(hi) for _ in range(cols)] for _ in range(rows)]


def _mod_matrix(C, N):
    return [[v % N for v in row] for row in C]


def _validate(moduli, trials=200, n=32, seed=1234, hi=256):
    rng = random.Random(seed)
    N, coeffs = crt_coeffs(moduli)
    # 1) quarter-square LUT == direct multiply, exhaustively per modulus
    for m in moduli:
        q = quarter_square_lut(m)
        for a in range(m):
            for b in range(m):
                assert qs_mul_mod(a, b, m, q) == (a * b) % m, f"QS mul wrong mod {m}"
    # 2) full pipeline bit-exact vs (true matmul mod N)
    wrap_elems = total_elems = max_result = 0
    for _ in range(trials):
        Aw = _rand_matrix(rng, n, n, hi)
        Ad = _rand_matrix(rng, n, n, hi)
        got = rns_matmul(Aw, Ad, moduli)
        ref = reference_matmul(Aw, Ad)
        assert got == _mod_matrix(ref, N), "RNS matmul != reference mod N"
        for row in ref:
            for v in row:
                total_elems += 1
                if v >= N:
                    wrap_elems += 1
                if v > max_result:
                    max_result = v
    return {"N": N, "coeffs": coeffs, "max_result": max_result,
            "wrap_frac": wrap_elems / total_elems, "trials": trials, "n": n}


def kernel_checksum(n=32, moduli=DEFAULT_MODULI):
    """Reproduce the x280/host C kernel's compact validators (crt_kernel.h): the 32x32 pattern
    matmul's checksum (sum of all outputs, mod 2^32) + four spot values. Used to confirm the C
    kernel is bit-exact against this golden model."""
    Aw = pattern_matrix(n, n, 0)
    Ad = pattern_matrix(n, n, 1)
    C = rns_matmul(Aw, Ad, moduli)
    checksum = sum(C[i][j] for i in range(n) for j in range(n)) & 0xFFFFFFFF
    samples = (C[0][0], C[1][2], C[15][15], C[31][31])
    return checksum, samples


def main():
    print("=" * 72)
    print("CRT / quarter-square matmul — golden reference validation")
    print("=" * 72)

    N, coeffs = crt_coeffs(DEFAULT_MODULI)
    print(f"\nmoduli {DEFAULT_MODULI}:  N = {N}   reconstruction coeffs = {coeffs}")
    print(f"  published formula (56*r5 + 120*r7 + 105*r8) mod 280  ->  match = "
          f"{coeffs == [56, 120, 105] and N == 280}")
    for m in DEFAULT_MODULI:
        print(f"  quarter-square LUT mod {m}: q[0..{2*m-1}] = {quarter_square_lut(m)}")

    # constants-as-shift-add (the key efficiency feature): no general multiplier
    print("\nconstant multiplies as shift-add networks (CSD):")
    for c, terms, adds in shiftadd_cost(DEFAULT_MODULI):
        net = " ".join(f"{'+' if d > 0 else '-'}(x<<{s})" for d, s in csd(c))
        assert all(shift_add_mul(x, c) == x * c for x in range(64)), f"shift-add wrong for {c}"
        print(f"  x*{c:<3} = {net}   ({adds} add/sub)")
    # whole reconstruction with ONLY shifts/adds must match the multiply version
    rng = random.Random(7)
    ok = all(crt_reconstruct_shiftadd(crt_forward(v, DEFAULT_MODULI), DEFAULT_MODULI)
             == crt_reconstruct(crt_forward(v, DEFAULT_MODULI), DEFAULT_MODULI)
             for v in (rng.randrange(280) for _ in range(2000)))
    print(f"  shift-add reconstruction == multiply reconstruction (2000 vals): {ok}")

    print("\n--- published set {5,7,8} (8-bit / mod-280 output) ---")
    r = _validate(DEFAULT_MODULI, trials=50)
    print(f"  bit-exact vs (true matmul mod {r['N']}) over {r['trials']} random 32x32 int8 trials: PASS")
    print(f"  true result peaks at {r['max_result']:,}  ->  {r['wrap_frac']*100:.1f}% of outputs "
          f"exceed N={r['N']} (wrap = info loss vs full integer result)")

    full = (5, 7, 8, 9, 11, 13, 17)     # pairwise coprime; product > 2.08M
    Nf, _ = crt_coeffs(full)
    print(f"\n--- full-range set {full} (non-wrapping K=32 int8) ---")
    r2 = _validate(full, trials=50)
    print(f"  N = {Nf:,}  (> peak K=32 int8 result {r2['max_result']:,})")
    print(f"  bit-exact vs true matmul over {r2['trials']} trials: PASS")
    print(f"  outputs that wrap: {r2['wrap_frac']*100:.1f}%   (0% = exact full integer matmul)")
    print(f"  largest LUT domain: 2*max(m) = {2*max(full)} entries/modulus  ({len(full)} tiny LUTs)")
    print("\nGolden model ready. Any x280/Tensix/RTL kernel must reproduce rns_matmul() bit-for-bit.")


if __name__ == "__main__":
    main()
