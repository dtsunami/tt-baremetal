#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
sweep.py — interactive design-space explorer for the CRT / quarter-square matmul modulus set.

Picks pairwise-coprime modulus sets and scores each on the things that actually drive area / energy
of the quarter-square RNS unit (and the x280 RVV kernel that prototypes it):

  coverage     N = prod(m_i) must exceed the workload's max accumulated result (else outputs wrap)
  lut          quarter-square table size = sum(2*m_i) entries  (area / vrgather table)
  forward      cost of x mod m_i: FREE if m=2^k (mask), CHEAP if m=2^k±1 (shift-add digit sum), else HARD
  reconstruct  CRT recombine sum(coeff_i * r_i) mod N as a SHIFT-ADD network: total add/sub count (CSD)

The whole point ("park constants / wire with bitshifts"): favor moduli whose mod is a shift/mask and
whose CRT coefficients have low canonical-signed-digit weight — no general multiplier or divider anywhere.

  python crt/sweep.py                      # top sets for the default workload (K=32 int8, full range)
  python crt/sweep.py --bits 8 --out 8     # 8-bit inputs, 8-bit (mod-2^8) output  -> small N ok
  python crt/sweep.py --set 5,7,8          # score one specific set (validates bit-exact vs golden model)
  python crt/sweep.py --k 64 --max-m 31 --top 15
"""
from __future__ import annotations
import argparse
from itertools import combinations
from math import gcd, prod

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from crt_matmul import csd, crt_coeffs, rns_matmul, reference_matmul, pattern_matrix

# Candidate small moduli with CHEAP modular arithmetic (the ones worth using).
POOL = [2, 3, 4, 5, 7, 8, 9, 11, 13, 16, 17, 31, 32]


def mod_class(m: int):
    """Forward-transform cost class of x mod m. 0=free (power of two, just a mask),
    1=cheap (2^k +/- 1, mod via shift-add digit folding), 2=hard (general)."""
    k = m.bit_length() - 1
    if m == (1 << k):
        return 0, "mask"                      # 2,4,8,16,32
    if m in (2**i - 1 for i in range(2, 7)):
        return 1, "2^k-1"                     # 3,7,15,31
    if m in (2**i + 1 for i in range(1, 7)):
        return 1, "2^k+1"                     # 3,5,9,17,33
    return 2, "general"                       # 11,13,...


def pairwise_coprime(s):
    return all(gcd(a, b) == 1 for a, b in combinations(s, 2))


def score(moduli):
    """Cost metrics for one modulus set (assumed pairwise coprime)."""
    moduli = tuple(sorted(moduli))
    N, coeffs = crt_coeffs(moduli)
    lut = sum(2 * m for m in moduli)                       # quarter-square table entries
    fwd = sum(mod_class(m)[0] for m in moduli)             # forward residue gate proxy
    recon = sum(len(csd(c)) - 1 for c in coeffs)           # reconstruction add/sub count (shift-add)
    hard = [m for m in moduli if mod_class(m)[0] == 2]     # moduli needing a real divider/LUT for mod
    # combined proxy: area-ish (lut) + per-element gate-ish (fwd + recon), hard moduli penalized
    cost = lut + 2 * fwd + recon + 6 * len(hard)
    return {"moduli": moduli, "N": N, "n": len(moduli), "lut": lut, "fwd": fwd,
            "recon": recon, "hard": hard, "coeffs": coeffs, "cost": cost}


def max_result(k, bits, signed):
    """Max magnitude of a k-deep dot product of `bits`-bit values."""
    hi = (1 << (bits - 1)) - 1 if signed else (1 << bits) - 1
    return k * hi * hi


def enumerate_sets(need_N, max_m, max_moduli):
    """Pairwise-coprime subsets of POOL (m<=max_m) whose product >= need_N, minimal (no redundant m)."""
    pool = [m for m in POOL if m <= max_m]
    out = []
    for n in range(1, max_moduli + 1):
        for combo in combinations(pool, n):
            if prod(combo) < need_N or not pairwise_coprime(combo):
                continue
            # minimal: dropping any single modulus must break coverage
            if all(prod(c for c in combo if c != m) >= need_N for m in combo):
                continue
            out.append(combo)
    return out


def fmt_set(s):
    return "{" + ",".join(map(str, s)) + "}"


def report_set(s):
    sc = score(s)
    cls = " ".join(f"{m}:{mod_class(m)[1]}" for m in sc["moduli"])
    print(f"  {fmt_set(sc['moduli']):22s} N={sc['N']:>12,}  luts={sc['lut']:>3}e  "
          f"fwd={sc['fwd']}  recon={sc['recon']}adds  cost={sc['cost']:>3}")
    print(f"     mod-class: {cls}")
    print(f"     coeffs (CRT recombine, shift-add): "
          + " | ".join(f"{c}={'+'.join(str(t) for t in [len(csd(c))])}t" for c in sc["coeffs"]))
    # bit-exact spot check vs the golden model on the 32x32 pattern (mod N)
    Aw, Ad = pattern_matrix(32, 32, 0), pattern_matrix(32, 32, 1)
    got = rns_matmul(Aw, Ad, sc["moduli"])
    ref = reference_matmul(Aw, Ad)
    ok = all(got[i][j] == ref[i][j] % sc["N"] for i in range(32) for j in range(32))
    wrap = sum(ref[i][j] >= sc["N"] for i in range(32) for j in range(32))
    print(f"     golden-model check: {'bit-exact ✓' if ok else 'MISMATCH'}   "
          f"wrap(this pattern)={wrap}/1024")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=32, help="dot-product depth (default 32)")
    ap.add_argument("--bits", type=int, default=8, help="input element bit width (default 8)")
    ap.add_argument("--signed", action="store_true", help="signed inputs")
    ap.add_argument("--out", type=int, default=None,
                    help="output bit width: cover only 2^out (modular output); default = full range")
    ap.add_argument("--max-m", type=int, default=17, help="largest modulus to consider")
    ap.add_argument("--max-moduli", type=int, default=8, help="max set size")
    ap.add_argument("--top", type=int, default=12, help="how many best sets to show")
    ap.add_argument("--set", default=None, help="score one explicit set, e.g. 5,7,8")
    args = ap.parse_args()

    if args.set:
        s = tuple(int(x) for x in args.set.split(","))
        if not pairwise_coprime(s):
            print(f"  {fmt_set(s)} is NOT pairwise coprime — CRT not a bijection."); return 1
        print(f"Scoring {fmt_set(s)}:"); report_set(s); return 0

    need = (1 << args.out) if args.out else max_result(args.k, args.bits, args.signed) + 1
    kind = f"mod-2^{args.out} output" if args.out else f"FULL range (K={args.k} {'i' if args.signed else 'u'}{args.bits})"
    print(f"workload: {kind}  ->  need N >= {need:,}\n")
    sets = enumerate_sets(need, args.max_m, args.max_moduli)
    if not sets:
        print("no covering set in the pool — raise --max-m / --max-moduli."); return 1
    ranked = sorted((score(s) for s in sets), key=lambda d: (d["cost"], d["n"], -d["N"]))
    print(f"{len(sets)} covering pairwise-coprime sets; best by cost (lut + 2*fwd + recon + 6*hard):\n")
    print(f"  {'set':22s} {'N':>13}  luts fwd recon cost   classes")
    for sc in ranked[:args.top]:
        cls = ",".join(mod_class(m)[1][0] for m in sc["moduli"])   # m=mask k=2^k-1/+1 g=general
        flag = "" if not sc["hard"] else f"  <hard mod: {sc['hard']}>"
        print(f"  {fmt_set(sc['moduli']):22s} {sc['N']:>13,}  {sc['lut']:>4} {sc['fwd']:>3} "
              f"{sc['recon']:>5} {sc['cost']:>4}   {cls}{flag}")
    print(f"\nbest set in detail:")
    report_set(ranked[0]["moduli"])
    print("\n('mask'=free mod, '2^k±1'=shift-add mod, 'general'=needs a divider/LUT. "
          "lower cost = cheaper unit. Validate ops/watt on the x280 next.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
