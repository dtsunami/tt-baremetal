"""Fused resident render: run splat.render_ondevice's whole 11-stage pipeline as ONE resident kernel
(6 fused super-stages) in one doorbell ring, and verify EVERY intermediate scratch against a host golden
(localizes any bug to a stage), then the final RGB. One pixel-group (32 px) first.

Run: cd ~/bhtop && .venv/bin/python scratchpad/test_resident_render.py
"""
import sys, os, math, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import load_elf, read_word_from_device as rd, read_words_from_device as rds, write_words_to_device as wr
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import matmul as MM, llk_run, splat as SP
from bhtop.tensix.resident import boot_resident

# addresses (must match resident_render_perf.cpp)
DB, DONE, HB, DBG_U, DBG_M, DBG_P = 0x16000, 0x16010, 0x16020, 0x16030, 0x16040, 0x16050
A_PHI, A_PSI = 0x21000, 0x31000
A_PPAIR, A_DOP, A_DNOP, A_STRI, A_IDEN, A_COLOR = 0x60000, 0x60800, 0x61000, 0x61800, 0x62000, 0x62800
S_VSQ, S_AR, S_LPA, S_LA, S_W = 0x40000, 0x40800, 0x41000, 0x41800, 0x42000
OUT_C = 0x51000
POISON = 0xBADF00D5
K, SIZE = 16, 16


def enc(flat):   # tilize + bf16 pack  (flat = 1024 row-major)
    return MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])


def dec(addr, ctx, coord):  # read + bf16 unpack + untilize -> 1024 row-major
    return MM.untilize32(MM.unpack_bf16_words(rds(coord, addr, word_count=512, context=ctx)))


def sub(flat, rows, cols):  # useful [rows,cols] region of a 32x32 flat
    return [[flat[r * 32 + c] for c in range(cols)] for r in range(rows)]


def relerr(dev, gold, rows, cols):
    num = den = 0.0
    for r in range(rows):
        for c in range(cols):
            g = gold[r][c]; d = dev[r * 32 + c]
            num += abs(d - g); den += abs(g)
    return num / (den + 1e-9)


def main():
    ctx = init_ttexalens()
    coord = TensixLauncher.at(1, 2, ctx=ctx).coord

    # ---- operands (match render_ondevice) ----
    gs = SP.scene_rgb(k=K, seed=5, span=float(SIZE))
    order = sorted(range(K), key=lambda i: gs[i][9])
    gso = [gs[i] for i in order]
    psi, Ppair, Dop, Dnop, Mcomb, color = SP._consts(gso, K)
    Stri = [Mcomb[r] for r in range(K)]          # [K][K] strict-upper
    Iden = [Mcomb[K + r] for r in range(K)]      # [K][K] identity
    psi_rows = [[psi[r][c] for c in range(2 * K)] for r in range(3)]
    consts = {A_PSI: SP._pad32(psi_rows), A_PPAIR: SP._pad32(Ppair), A_DOP: SP._pad32(Dop),
              A_DNOP: SP._pad32(Dnop), A_STRI: SP._pad32(Stri), A_IDEN: SP._pad32(Iden),
              A_COLOR: SP._pad32(color)}

    pixels = [(x, y) for y in range(SIZE) for x in range(SIZE)]
    g0 = pixels[0:32]
    phi_f = SP._pad32([[float(x), float(y), 1.0] for (x, y) in g0])

    # ---- host golden (per stage, float) ----
    V = MM.matmul_golden(phi_f, consts[A_PSI])
    Vsq = [v * v for v in V]
    E = MM.matmul_golden(Vsq, consts[A_PPAIR])
    ar = [math.exp(min(e, 80.0)) for e in E]
    alpha = MM.matmul_golden(ar, consts[A_DOP])
    nalpha = MM.matmul_golden(ar, consts[A_DNOP])
    lpa = [math.log(a) if a > 1e-30 else -80.0 for a in alpha]
    la = [math.log1p(na) if na > -0.999999 else -80.0 for na in nalpha]
    logw = [la_s + lp_s for la_s, lp_s in zip(MM.matmul_golden(la, consts[A_STRI]),
                                              MM.matmul_golden(lpa, consts[A_IDEN]))]
    w = [math.exp(min(x, 80.0)) for x in logw]
    C = MM.matmul_golden(w, consts[A_COLOR])
    G = {"Vsq": Vsq, "ar": ar, "lpa": lpa, "la": la, "w": w, "C": C}
    Gsub = {"Vsq": sub(Vsq, 32, 32), "ar": sub(ar, 32, K), "lpa": sub(lpa, 32, K),
            "la": sub(la, 32, K), "w": sub(w, 32, K), "C": sub(C, 32, 3)}

    # ---- build + boot ----
    b = llk_run.build("resident_render_perf", run_type="L1_TO_L1", fidelity="HiFi4", fp32_acc=False,
                      formats=None, overrides={"ITERATIONS": "constexpr int ITERATIONS = 32;"})
    assert b["ok"], b["log"][-2000:]
    rdbg = boot_resident("resident_render_perf", coord, ctx=ctx,
                         runtime_words=[1, 1, 1, 1, 1, 128, 128, 0, 4, 4], clear_words=48)
    time.sleep(0.3)

    # ---- stage operands, poison scratch, ring ----
    for addr, flat in consts.items():
        wr(coord, addr, enc(flat), context=ctx)
    wr(coord, A_PHI, enc(phi_f), context=ctx)
    for s in (S_VSQ, S_AR, S_LPA, S_LA, S_W, OUT_C):
        wr(coord, s, [POISON] * 512, context=ctx)
    wr(coord, DB, [1], context=ctx)
    t0 = time.time(); done = None
    while time.time() - t0 < 6.0:
        done = rd(coord, DONE, context=ctx)
        if done == 1: break
        time.sleep(0.005)
    print(f"[render] done={done} hb={rd(coord,HB,context=ctx)} "
          f"U={rd(coord,DBG_U,context=ctx)} M={rd(coord,DBG_M,context=ctx)} P={rd(coord,DBG_P,context=ctx)}")

    # ---- per-stage verify ----
    dev = {"Vsq": dec(S_VSQ, ctx, coord), "ar": dec(S_AR, ctx, coord), "lpa": dec(S_LPA, ctx, coord),
           "la": dec(S_LA, ctx, coord), "w": dec(S_W, ctx, coord), "C": dec(OUT_C, ctx, coord)}
    cols = {"Vsq": 32, "ar": K, "lpa": K, "la": K, "w": K, "C": 3}
    print("[render] per-stage rel-err (useful region) vs host golden:")
    for st in ("Vsq", "ar", "lpa", "la", "w", "C"):
        e = relerr(dev[st], Gsub[st], 32, cols[st])
        print(f"    {st:4s}: rel-err={e:.3e}  dev[0,0]={dev[st][0]:+.4f} gold[0,0]={Gsub[st][0][0]:+.4f}")

    # ---- RGB PSNR of the group vs golden C ----
    mse = sum((dev["C"][r * 32 + c] - Gsub["C"][r][c]) ** 2 for r in range(32) for c in range(3)) / (32 * 3)
    psnr = 99.0 if mse < 1e-12 else 10 * math.log10((max(max(row) for row in Gsub["C"]) ** 2) / mse)
    print(f"[render] group RGB vs golden: mse={mse:.3e} PSNR≈{psnr:.1f} dB")
    for c in rdbg.values():
        c.set_reset_signal(True)
    ok = (done == 1)
    print(f"[render] FUSED RESIDENT RENDER (one group): {'ran' if ok else 'DID NOT COMPLETE'}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
