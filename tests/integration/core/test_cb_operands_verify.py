"""Lever 1b crux: verify the x280 cb_operands PRODUCER emits psi/Dop/Dnop/color in the exact tilized bf16
layout the render kernel reads (so a worker can nocread them = drop-in for the host stage). Feed a tile's
per-Gaussian coeffs to the x280, run cb_operands, compare its GDDR output tiles to the host tilization."""
import sys, struct, math, time
sys.path.insert(0, "/home/starboy/bhtop/src"); sys.path.insert(0, "/home/starboy/bhtop/scratchpad")
import numpy as np
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
from bhtop.tensix import splat as SP, matmul as MM

K = 12
OPR_SRC = "/home/starboy/bhtop/src/bhtop/kernels/x280/het/cb_operands.c"
ZIN, PIN, PSI, DOP, DNOP, COLOR = 0x30002300, 0x30002400, 0x30003000, 0x30003800, 0x30004000, 0x30004800
def bf16(x):                       # round-to-nearest-even, matching MM.pack_bf16_words (not truncation)
    b = struct.unpack("<I", struct.pack("<f", float(x)))[0]
    b += 0x7FFF + ((b >> 16) & 1)
    return (b >> 16) & 0xFFFF
fbits = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]
def enc(flat): return MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
def pad(rc): return SP._pad32(rc)


def whiten(g):
    gx, gy, a, b, c, op = g[:6]
    sa = math.sqrt(max(a, 1e-8)); m12 = b / sa
    m22 = math.sqrt(max(c - b * b / a, 0.0))
    return sa, m12, m22, -(sa * gx + m12 * gy), -(m22 * gy)


def main():
    ctx = init_ttexalens(); dev = L2cpu(ctx=ctx)
    try: dev.bringup(0)
    except Exception: pass

    gs = SP.scene_rgb(k=K, seed=7, span=16.0)
    order = sorted(range(K), key=lambda i: gs[i][9]); gso = [gs[i] for i in order]
    # host reference operands (depth-sorted), tilized exactly as the render stages them
    psi_h, Ppair, Dop_h, Dnop_h, Mcomb, color_h = SP._consts(gso, K)
    psi_rows = [[psi_h[r][c] for c in range(2 * K)] for r in range(3)]
    ref = {PSI: enc(pad(psi_rows)), DOP: enc(pad(Dop_h)), DNOP: enc(pad(Dnop_h)), COLOR: enc(pad(color_h))}

    # x280 inputs: per-Gaussian coeffs (ORIGINAL order) + depths; cb_operands sorts by z internally
    dev.wr(0, ZIN, [K] + [fbits(gs[i][9]) for i in range(K)])
    pin = []
    for i in range(K):
        sa, m12, m22, c1, c2 = whiten(gs[i])
        coeffs = [sa, m12, m22, c1, c2, gs[i][5], gs[i][6], gs[i][7], gs[i][8]]
        pin += [bf16(x) for x in coeffs]
    dev.wr(0, PIN, pin)
    for a in ref: dev.wr(0, a, [0] * 512)

    dev.load(0, 0, tc.compile_source(OPR_SRC, base=CODE_ADDR, march="rv64gc"))
    for _ in range(80):
        if dev.telemetry(0, slots=1, hart=0)[0] == 0x4F505253: break
        time.sleep(0.03)
    else:
        print("FAIL: cb_operands never signalled OPRS"); return

    print(f"{'operand':>7} {'words match':>12} {'first mismatch':>16}")
    allok = True
    for name, a in [("psi", PSI), ("Dop", DOP), ("Dnop", DNOP), ("color", COLOR)]:
        dev_w = dev.rdn(0, a, 512); r = ref[a]
        mism = [k for k in range(512) if (dev_w[k] & 0xFFFFFFFF) != (r[k] & 0xFFFFFFFF)]
        ok = len(mism) == 0; allok &= ok
        print(f"{name:>7} {512-len(mism):>8}/512 {('-' if ok else hex(mism[0])):>16}")
    print("CB_OPERANDS_DROP_IN_OK" if allok else "CB_OPERANDS_MISMATCH (needs layout reconcile)")


if __name__ == "__main__":
    main()
