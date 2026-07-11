"""Stage-2 de-risk: can the worker BRISC (rv32im, NO FPU) compute dLdcolor = wT@dLdC accurately enough to move
consume off the x280 hub? Read the REAL grad-inbox tiles (w, dLdC) the x280 consume_slot processes, extract
per (gaussian,channel) the length-32 dot product, and compare three MACs:
  (fp32)  the x280 reference: float accumulate of bf16*bf16   (what consume_slot does today)
  (int)   a BRISC-friendly INTEGER fixed-point MAC (bf16 -> Qn int, int64 accumulate, no FP)
  (f64)   host golden
Reports max/rel error of the integer MAC vs the fp32 reference. Bar (memory): backward is zero-mean-error
tolerant, ~1% is fine. Proves the wall is (or isn't) crossable with integer-only arithmetic."""
import sys, os, math, struct
sys.path.insert(0, "/home/starboy/bhtop/src")
import numpy as np
from bhtop.het.grid_engine import HetGridEngine, GINO
GIS_O = 0x10000
_bf = lambda u: struct.unpack("<f", struct.pack("<I", (u & 0xFFFF) << 16))[0]

def tget_bits(tile_words, row, col):   # inverse of place(): 32x32 4-face bf16 tile -> the bf16 half-word at (row,col)
    face = (2 if row >= 16 else 0) + (1 if col >= 16 else 0)
    e = face * 256 + (row % 16) * 16 + (col % 16)
    w = tile_words[e >> 1]
    return (w >> 16) if (e & 1) else (w & 0xFFFF)

# ---- integer fixed-point bf16 MAC (models exactly what a BRISC kernel would do; no float ops) ----
def bf16_to_q(bits, Q):
    """bf16 half-word -> signed Q-format fixed-point int (round-to-nearest). Pure integer (BRISC-doable)."""
    s = (bits >> 15) & 1; e = (bits >> 7) & 0xFF; m = bits & 0x7F
    if e == 0: return 0                     # zero/denormal -> 0 (denormals negligible)
    mant = (1 << 7) | m                     # 1.mmmmmmm as 8-bit int (implicit 1)
    # value = mant * 2^(e-127-7); Q-fixed = round(value * 2^Q) = mant << (e-127-7+Q), or >> if negative shift
    sh = e - 127 - 7 + Q
    if sh >= 0:
        q = mant << sh
    else:
        q = (mant + (1 << (-sh - 1))) >> (-sh)    # round-to-nearest on the right shift
    return -q if s else q

def int_mac(w_bits, dc_bits, Qw=15, Qc=14):
    """dot(w, dc) via int64 fixed-point. Returns float (only for comparison; BRISC would emit f32 bits)."""
    acc = 0
    for wb, db in zip(w_bits, dc_bits):
        acc += bf16_to_q(wb, Qw) * bf16_to_q(db, Qc)      # int32*int32 -> int64
    return acc / float(1 << (Qw + Qc))

N, IMG, W, NH = 2048, 128, 8, 4
rng = np.random.default_rng(7)
P = np.zeros((N, 14), np.float64)
P[:, 0] = rng.uniform(-0.92, 0.92, N); P[:, 1] = rng.uniform(-0.92, 0.92, N); P[:, 2] = 1.0
P[:, 3:6] = math.log(0.02); P[:, 6] = 1.0; P[:, 10] = 0.5; P[:, 11:14] = rng.uniform(0.1, 0.9, (N, 3))
cam16 = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, IMG / 2, IMG / 2, IMG / 2, IMG / 2]
yy, xx = np.mgrid[0:IMG, 0:IMG].astype(np.float32) / IMG
tgt = np.stack([xx, yy, 0.5 * (xx + yy)], -1).astype(np.float32)
LR = [0.005]*3 + [0.003]*3 + [0.001]*4 + [0.02] + [0.005]*3

eng = HetGridEngine(N, IMG, IMG, W=W, NH=NH); eng.set_params(P); eng.set_views(tgt[None])
eng.step(cam16, tgt.reshape(-1).astype(np.float64), LR, 1, view_idx=0)   # populate grad inbox
print("[stage2] booted + stepped; reading grad inbox tiles", flush=True)

d = eng.dev
w_all, dc_all, fp_ref, i_res = [], [], [], []
for slot in range(W):
    for g in range(8):
        gib = GINO + slot * GIS_O + g * 0x2000
        WW = d.rdn(0, gib + 2 * 0x800, 512); DC = d.rdn(0, gib + 3 * 0x800, 512)
        dcv = [[_bf(tget_bits(DC, p, ch)) for p in range(32)] for ch in range(3)]
        dcb = [[tget_bits(DC, p, ch) for p in range(32)] for ch in range(3)]
        for i in range(12):
            wv = [_bf(tget_bits(WW, p, i)) for p in range(32)]
            wb = [tget_bits(WW, p, i) for p in range(32)]
            for ch in range(3):
                fp = float(np.float32(sum(np.float32(wv[p]) * np.float32(dcv[ch][p]) for p in range(32))))
                it = int_mac(wb, dcb[ch])
                fp_ref.append(fp); i_res.append(it)
                w_all += wv; dc_all += dcv[ch]

fp_ref = np.array(fp_ref); i_res = np.array(i_res)
w_all = np.array(w_all); dc_all = np.array(dc_all)
mx = np.abs(fp_ref).max()
sig = np.abs(fp_ref) > 0.01 * mx                    # "meaningful" dot products (>1% of peak) — rel error is only
rel = np.abs(i_res[sig] - fp_ref[sig]) / np.abs(fp_ref[sig])   # informative where the true value isn't ~0
abserr = np.abs(i_res - fp_ref)
print(f"[stage2] samples: {len(fp_ref)} dot-products; {sig.sum()} above 1% of peak", flush=True)
print(f"[stage2] w range   [{w_all.min():+.4f}, {w_all.max():+.4f}]  |dc| max {np.abs(dc_all).max():.4f}", flush=True)
print(f"[stage2] dLdcolor  |fp_ref| max {mx:.4f}  rms {np.sqrt((fp_ref**2).mean()):.4f}", flush=True)
print(f"[stage2] INT fixed-point MAC vs x280 FP MAC:", flush=True)
print(f"           max|abs err|   = {abserr.max():.2e}   ({abserr.max()/mx*100:.4f}% of peak signal)", flush=True)
print(f"           rel err (>1% peak): mean {rel.mean()*100:.4f}%   max {rel.max()*100:.4f}%   (bar ~1%)", flush=True)
print(f"[stage2] VERDICT: integer-only MAC {'CROSSES' if abserr.max()/mx < 0.01 else 'MISSES'} the dLdcolor wall "
      f"(no soft-float needed — rv32im mul/add/shift suffices)", flush=True)
