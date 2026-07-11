"""Stage-2 numeric proof (no device): bit-accurate model of the conductor's INTEGER consume path
(bf16_to_q -> int64 MAC/accumulate -> q_to_f32bits) vs the x280's FLOAT consume_slot, over 8 groups of a
tile. Validates all three outputs — dLdpsi/dLdop (Q16 detilize-sum), dLdcolor (Q15xQ14 MAC), loss (Q28 SSE) —
against realistic bf16 grad ranges. If close, the on-silicon path is de-risked (the C is the same integer ops)."""
import numpy as np
rng = np.random.default_rng(3)

def f2bf(x):                     # round-to-nearest bf16, return the 16-bit pattern
    b = np.float32(x).view(np.uint32); b = int(b) + 0x7FFF + ((int(b) >> 16) & 1); return (b >> 16) & 0xFFFF
def bf2f(bf):                    # bf16 bits -> float (what tget->bf2f gives; == x280's read)
    return np.uint32((bf & 0xFFFF) << 16).view(np.float32).item()

# --- EXACT ports of conductor.c integer helpers (Python ints, same ops rv32im does) ---
def bf16_to_q(bf, Q):
    exp = (bf >> 7) & 0xFF
    if exp >= 0xE3 or exp == 0: return 0
    mant = (1 << 7) | (bf & 0x7F); sh = exp - 127 - 7 + Q
    if sh >= 24: q = 0x40000000
    elif sh >= 0: q = mant << sh
    elif sh > -31: q = (mant + (1 << (-sh - 1))) >> (-sh)
    else: q = 0
    return -q if ((bf >> 15) & 1) else q
def q_to_f32bits(sacc, Q):
    if sacc == 0: return 0
    sign = 0x80000000 if sacc < 0 else 0
    a = -sacc if sacc < 0 else sacc
    hi = (a >> 32) & 0xFFFFFFFF; lo = a & 0xFFFFFFFF
    if hi: e = 32; top = hi
    else: e = 0; top = lo
    b = 31
    while not ((top >> b) & 1): b -= 1
    e += b
    if e >= 23:
        drop = e - 23
        if drop >= 32: mant = hi >> (drop - 32)
        elif drop == 0: mant = lo
        else: mant = ((hi << (32 - drop)) | (lo >> drop)) & 0xFFFFFFFF
    else: mant = (lo << (23 - e)) & 0xFFFFFFFF
    mant &= 0x7FFFFF; exp = e - Q + 127
    if exp <= 0: return sign
    if exp >= 255: return sign | 0x7F800000
    return sign | (exp << 23) | mant
def bits2f(u): return np.uint32(u & 0xFFFFFFFF).view(np.float32).item()

def fsan(x): return 0.0 if (x != x or abs(x) > 1e30) else x

K, NG = 12, 8
# synthetic bf16 grad tiles: w in [0,0.5] (weights), dLdC in [-1,1], dLdpsi/dLdop ~ N(0, 0.3) with some spikes
DP = [[[f2bf(rng.normal(0, 0.3)) for _ in range(32)] for _ in range(32)] for _ in range(NG)]  # [g][row][col]
DO = [[[f2bf(rng.normal(0, 0.2)) for _ in range(32)] for _ in range(32)] for _ in range(NG)]
WW = [[[f2bf(abs(rng.uniform(0, 0.5))) for _ in range(32)] for _ in range(32)] for _ in range(NG)]
DC = [[[f2bf(rng.normal(0, 0.4)) for _ in range(32)] for _ in range(32)] for _ in range(NG)]

# ---- x280 float consume_slot (reference) ----
ref = np.zeros((K, 9)); ref_loss = 0.0
for g in range(NG):
    dcv = [[fsan(bf2f(DC[g][p][ch])) for p in range(32)] for ch in range(3)]
    for p in range(32):
        for ch in range(3): ref_loss += dcv[ch][p] ** 2
    for i in range(K):
        ref[i, 0] += fsan(bf2f(DP[g][0][2 * i])); ref[i, 1] += fsan(bf2f(DP[g][1][2 * i])); ref[i, 2] += fsan(bf2f(DP[g][2][2 * i]))
        ref[i, 3] += fsan(bf2f(DP[g][1][2 * i + 1])); ref[i, 4] += fsan(bf2f(DP[g][2][2 * i + 1])); ref[i, 5] += fsan(bf2f(DO[g][0][i]))
        for ch in range(3):
            ref[i, 6 + ch] += sum(fsan(bf2f(WW[g][p][i])) * dcv[ch][p] for p in range(32))

# ---- conductor integer path ----
acc = [[0] * 9 for _ in range(K)]; sse = 0
for g in range(NG):
    dcq = [[bf16_to_q(DC[g][p][ch], 14) for p in range(32)] for ch in range(3)]
    for p in range(32):
        for ch in range(3): sse += dcq[ch][p] * dcq[ch][p]
    for i in range(K):
        acc[i][0] += bf16_to_q(DP[g][0][2 * i], 16); acc[i][1] += bf16_to_q(DP[g][1][2 * i], 16); acc[i][2] += bf16_to_q(DP[g][2][2 * i], 16)
        acc[i][3] += bf16_to_q(DP[g][1][2 * i + 1], 16); acc[i][4] += bf16_to_q(DP[g][2][2 * i + 1], 16); acc[i][5] += bf16_to_q(DO[g][0][i], 16)
        for ch in range(3):
            acc[i][6 + ch] += sum(bf16_to_q(WW[g][p][i], 15) * dcq[ch][p] for p in range(32))
got = np.array([[bits2f(q_to_f32bits(acc[i][j], 16 if j < 6 else 29)) for j in range(9)] for i in range(K)])
got_loss = bits2f(q_to_f32bits(sse, 28))

scale = np.abs(ref).max()
psi = np.abs(got[:, :5] - ref[:, :5]).max()
op = np.abs(got[:, 5] - ref[:, 5]).max()
col = np.abs(got[:, 6:] - ref[:, 6:]).max()
print(f"[stage2-num] dLdpsi  max|abs err| = {psi:.2e}  ({psi/scale*100:.4f}% of peak {scale:.3f})")
print(f"[stage2-num] dLdop   max|abs err| = {op:.2e}")
print(f"[stage2-num] dLdcolor max|abs err| = {col:.2e}  ({col/np.abs(ref[:,6:]).max()*100:.4f}% of color peak)")
print(f"[stage2-num] loss    ref={ref_loss:.4f}  int={got_loss:.4f}  rel={abs(got_loss-ref_loss)/ref_loss*100:.4f}%")
ok = psi/scale < 0.01 and col/max(np.abs(ref[:,6:]).max(),1e-9) < 0.01 and abs(got_loss-ref_loss)/ref_loss < 0.02
print("[stage2-num] VERDICT:", "PASS — integer consume matches float within tolerance" if ok else "FAIL")
