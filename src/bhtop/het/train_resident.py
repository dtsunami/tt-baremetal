"""FUSED-KERNEL on-device Gaussian-splat trainer — the whole training step (forward render + on-device
dLdC=C-gt + backward) is ONE resident Tensix kernel (resident_train_perf), per 32-pixel group; grads
accumulate over the 8 groups of a 16x16 tile; the x280 does whiten-backward + Adam (opt_step, resident).
dLdcolor = wᵀ@dLdC is computed host-side (the transpose_dest primitive corrupts the eltwise pipeline;
in production this rides on the x280). This is `train_ondevice.py` with fwd+bwd collapsed to one ring.

  usage: train_resident.py [STEPS]     (default 12)
"""
import sys, struct, math, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from ttexalens.tt_exalens_lib import read_word_from_device as rd, read_words_from_device as rds, write_words_to_device as wr
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import splat as SP, matmul as MM, llk_run
from bhtop.tensix.resident import boot_resident

K, SIZE, P = 12, 16, 32
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
OPT_SRC = "/home/starboy/bhtop/src/bhtop/kernels/x280/het/opt_step.c"
H = dict(phi=0x21000, psi=0x22000, Ppair=0x23000, Dop=0x24000, Dnop=0x25000, Stri=0x26000, Iden=0x27000,
         color=0x28000, gt=0x29000, opB=0x2A000, colorT=0x2B000, PpairT=0x2C000, U=0x2D000, phi2T=0x2E000,
         ones=0x2F000, ones1P=0x30000)
S_C, S_w, S_dLdC, O_dLdop, O_dLdpsi = 0x42800, 0x42000, 0x43000, 0x51000, 0x52000
DB, DONE = 0x16000, 0x16010
fb = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]
bf = lambda u: struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def enc(flat): return MM.pack_bf16_words([float(x) for x in MM.tilize32(flat)])
def pad(rc): return SP._pad32(rc)
def dec(a, ctx, c): return MM.untilize32(MM.unpack_bf16_words(rds(c, a, word_count=512, context=ctx)))


def groups_pixels():
    pix = [(x, y) for y in range(SIZE) for x in range(SIZE)]
    return [pix[i:i + 32] for i in range(0, len(pix), 32)]


def stage_consts(coord, ctx, gs, order):
    """Stage the per-step gaussian-dependent const tiles for resident_train_perf."""
    gso = [gs[i] for i in order]
    psi, Ppair, Dop, Dnop, Mcomb, color = SP._consts(gso, K)
    Stri = [Mcomb[r] for r in range(K)]
    op = [Dop[k][k] for k in range(K)]
    colorT = [[color[k][r] for k in range(K)] for r in range(3)]
    PpairT = [[Ppair[r][c] for r in range(2 * K)] for c in range(K)]
    opB = [[(op[k] if k < K else 0.5) for k in range(32)] for _ in range(P)]   # finite col-pad for 1/alpha
    psi_rows = [[psi[r][c] for c in range(2 * K)] for r in range(3)]
    for name, m in [("psi", psi_rows), ("Ppair", Ppair), ("Dop", Dop), ("Dnop", Dnop), ("Stri", Stri),
                    ("color", color), ("colorT", colorT), ("PpairT", PpairT), ("opB", opB)]:
        wr(coord, H[name], enc(pad(m)), context=ctx)


def main():
    pr = lambda *a: print(*a, flush=True)
    ctx = init_ttexalens(); pr("[setup] exalens up")
    coord = TensixLauncher.at(1, 2, ctx=ctx).coord
    dev = L2cpu(ctx=ctx)
    try: dev.bringup(0); pr("[setup] x280 bringup ok")
    except Exception as e: pr("[setup] x280 bringup:", type(e).__name__, "(already up)")

    grp = groups_pixels()
    U = [[1.0 if j > i else 0.0 for i in range(K)] for j in range(K)]
    phi2T = [pad([[2.0 * x, 2.0 * y, 2.0][r] for (x, y) in g] for r in range(3)) for g in grp]
    phi_g = [pad([[float(x), float(y), 1.0] for (x, y) in g]) for g in grp]

    # 1) TARGET = exact host golden render (instant; a device render here would be the slow 88-dispatch path)
    tgt = SP.scene_rgb(k=K, seed=11, span=float(SIZE))
    init = SP.scene_rgb(k=K, seed=22, span=float(SIZE))
    order = sorted(range(K), key=lambda i: init[i][9])
    target = SP._golden_render(tgt, SIZE)                         # [256][3]
    gt_g = [[[target[g[p][1] * SIZE + g[p][0]][ch] for ch in range(3)] for p in range(len(g))] for g in grp]

    # 2) build + boot the resident train kernel + stage its static tiles.
    llk_run.build("resident_train_perf", run_type="L1_TO_L1", fidelity="HiFi4", fp32_acc=False,
                  formats=None, overrides={"ITERATIONS": "constexpr int ITERATIONS = 32;"}); pr("[setup] kernel built")
    boot_resident("resident_train_perf", coord, ctx=ctx,
                  runtime_words=[1, 1, 1, 1, 1, 128, 128, 0, 4, 4], clear_words=56)
    time.sleep(0.3); pr("[setup] kernel booted")
    wr(coord, H["Iden"], enc([1.0 if r == c else 0.0 for r in range(32) for c in range(32)]), context=ctx)
    wr(coord, H["U"], enc(pad(U)), context=ctx)
    wr(coord, H["ones"], enc([1.0] * 1024), context=ctx)
    wr(coord, H["ones1P"], enc(pad([[1.0] * P])), context=ctx)

    # x280 resident optimizer: params (orig order) + zero Adam + order; load opt_step ONCE
    param0 = [[init[o][j] for j in range(9)] for o in range(K)]
    dev.wr(0, 0x30005040, [o & 0xFFFFFFFF for o in order])
    dev.wr(0, 0x30005800, [fb(param0[o][j]) for o in range(K) for j in range(9)])
    dev.wr(0, 0x30006000, [0] * (K * 9)); dev.wr(0, 0x30006400, [0] * (K * 9))
    dev.wr(0, 0x30004000, [0]); dev.wr(0, 0x30004010, [0])
    pr("[setup] compiling+loading x280 opt_step")
    dev.load(0, 0, tc.compile_source(OPT_SRC, base=CODE_ADDR, march="rv64gc")); time.sleep(0.3)
    assert dev.telemetry(0, slots=1, hart=0)[0] == 0x4F505421, "opt_step not resident"
    pr("[setup] x280 opt_step resident")

    def full_psnr(rgb_tile):
        mse = sum((rgb_tile[p][ch] - target[p][ch]) ** 2 for p in range(SIZE * SIZE) for ch in range(3)) / (SIZE * SIZE * 3)
        return (99.0 if mse < 1e-12 else 10 * math.log10(1.0 / mse)), mse

    ring = 0
    pr(f"FUSED-kernel on-device trainer: K={K} {SIZE}x{SIZE}, {STEPS} steps "
          f"(resident_train_perf per group, x280 whiten-bwd+Adam)")
    for step in range(1, STEPS + 1):
        t0 = time.time()
        pw = dev.rdn(0, 0x30005800, K * 9)
        gs = [tuple(bf(pw[o * 9 + j]) for j in range(9)) + (init[o][9],) for o in range(K)]
        stage_consts(coord, ctx, gs, order)

        dLdpsi = [[0.0] * (2 * K) for _ in range(3)]; dLdop = [0.0] * K
        dLdcol = [[0.0] * 3 for _ in range(K)]; rgb_tile = [[0.0, 0.0, 0.0] for _ in range(SIZE * SIZE)]
        for gi, g in enumerate(grp):
            wr(coord, H["phi"], enc(phi_g[gi]), context=ctx)
            wr(coord, H["phi2T"], enc(phi2T[gi]), context=ctx)
            wr(coord, H["gt"], enc(pad(gt_g[gi])), context=ctx)
            # TRULY RESIDENT: boot once (setup), ring per group with an incrementing doorbell — no reboot.
            # The 27-stage kernel is multi-ring resident now that eltwise is matmul-load + SFPU-binary
            # (the unpack never switches mode -> avoids the BH mode-switch errata). 30/30 rings bit-exact.
            ring += 1
            wr(coord, DB, [ring], context=ctx)
            tt = time.time()
            while time.time() - tt < 4.0 and rd(coord, DONE, context=ctx) != ring:
                time.sleep(0.003)
            C = dec(S_C, ctx, coord); wv = dec(S_w, ctx, coord); dC = dec(S_dLdC, ctx, coord)
            dp = dec(O_dLdpsi, ctx, coord); do = dec(O_dLdop, ctx, coord)
            for p in range(len(g)):
                rgb_tile[g[p][1] * SIZE + g[p][0]] = [C[p * 32 + ch] for ch in range(3)]
            for r in range(3):
                for m in range(2 * K): dLdpsi[r][m] += dp[r * 32 + m]
            for k in range(K): dLdop[k] += do[k]
            for k in range(K):                        # dLdcolor += wᵀ@dLdC (host; delegated off the kernel)
                for ch in range(3):
                    dLdcol[k][ch] += sum(wv[p * 32 + k] * dC[p * 32 + ch] for p in range(len(g)))

        pval, mse = full_psnr(rgb_tile)
        gradin = [[dLdpsi[0][2 * i], dLdpsi[1][2 * i], dLdpsi[2][2 * i], dLdpsi[1][2 * i + 1],
                   dLdpsi[2][2 * i + 1], dLdop[i], dLdcol[i][0], dLdcol[i][1], dLdcol[i][2]] for i in range(K)]
        bc1 = 1.0 / (1 - 0.9 ** step); bc2 = 1.0 / (1 - 0.999 ** step)
        LR = [0.15, 0.15, 2e-3, 2e-3, 2e-3, 0.02, 0.1, 0.1, 0.1]
        hdr = [K, step, fb(bc1), fb(bc2), fb(0.9), fb(0.999), fb(1e-8)] + [fb(x) for x in LR]
        dev.wr(0, 0x30005100, [fb(gradin[i][j]) for i in range(K) for j in range(9)])
        dev.wr(0, 0x30005000, hdr); dev.wr(0, 0x30004000, [step])
        for _ in range(60):
            if dev.rd(0, 0x30004010) == step: break
            time.sleep(0.03)
        gnorm = sum(abs(gradin[i][j]) for i in range(K) for j in range(9))
        pw2 = dev.rdn(0, 0x30005800, K * 9)
        pdelta = sum(abs(bf(pw2[i]) - bf(pw[i])) for i in range(K * 9))
        pr(f"  step {step:2d}: PSNR={pval:5.2f} dB  loss={mse:.5f}  |grad|={gnorm:.4f}  |Δparam|={pdelta:.4f}  ({time.time()-t0:.1f}s)")
        pr(f"          pw[0:3]={[round(bf(pw[i]),4) for i in range(3)]}  post-opt pw2[0:3]={[round(bf(pw2[i]),4) for i in range(3)]}")

    pr("done (fused resident training step: fwd+dLdC+bwd in one ring/group, x280 optimizer)")


if __name__ == "__main__":
    main()
