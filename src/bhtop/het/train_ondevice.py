"""FULLY ON-DEVICE Gaussian-splat trainer — closing the loop, bare-metal, no ttnn/tt-metal:
  Tensix forward (render_ondevice) -> host loss grad dLdC -> Tensix dense backward (backward_ondevice)
  -> x280 whiten-backward + un-sort + Adam (resident opt_step kernel, params+m/v live on x280) -> repeat.
The host only orchestrates + computes the loss gradient; ALL model arithmetic (fwd, dense bwd,
whiten-bwd, optimizer) is on device. Prints the PSNR trajectory vs a device-rendered target.

  usage: train_ondevice.py [STEPS]   (default 6; each step ~ forward+backward+opt)
"""
import sys, struct, math, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import splat as SP, matmul as MM, sfpu as SF

K, size = 12, 16
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
P = size * size
OPT_SRC = "/home/starboy/bhtop/src/bhtop/kernels/x280/het/opt_step.c"
fb = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]
bf = lambda u: struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]

def main():
    ctx = init_ttexalens()
    coord = TensixLauncher.at(1, 2, ctx=ctx).coord
    dev = L2cpu(ctx=ctx)
    try: dev.bringup(0)
    except Exception as e: print("bringup:", type(e).__name__, "(already up)")

    # build every kernel once (fwd + bwd share the matmul/sfpu/eltwise ELFs)
    MM.build_for("fp32")
    for op in ("square", "exponential", "log", "log1p", "reciprocal"): SF.build_unary(op)
    for op in ("mul", "sub"): SF.build_binary(op)

    # scenes: target (rendered on device) + init (the one we train)
    tgt = SP.scene_rgb(k=K, seed=11, span=float(size))
    init = SP.scene_rgb(k=K, seed=22, span=float(size))
    tgt_order = sorted(range(K), key=lambda i: tgt[i][9])
    order = sorted(range(K), key=lambda i: init[i][9])            # fixed (z not trained)
    target = SP.render_ondevice(coord, ctx=ctx, k=K, size=size, gs=tgt, order=tgt_order,
                                prebuilt=True, verbose=False)["rgb"]

    # x280 resident state: params (orig order) + zero adam + order, load opt_step ONCE
    param0 = [[init[o][j] for j in range(9)] for o in range(K)]
    dev.wr(0, 0x30005040, [o & 0xFFFFFFFF for o in order])
    dev.wr(0, 0x30005800, [fb(param0[o][j]) for o in range(K) for j in range(9)])
    dev.wr(0, 0x30006000, [0]*(K*9)); dev.wr(0, 0x30006400, [0]*(K*9))
    dev.wr(0, 0x30004000, [0]); dev.wr(0, 0x30004010, [0])
    dev.load(0, 0, tc.compile_source(OPT_SRC, base=CODE_ADDR, march="rv64gc")); time.sleep(0.3)
    assert dev.telemetry(0, slots=1, hart=0)[0] == 0x4F505421, "opt_step not resident"

    def psnr(rgb):
        mse = sum((rgb[p][ch]-target[p][ch])**2 for p in range(P) for ch in range(3))/(P*3)
        return 99.0 if mse < 1e-12 else 10*math.log10(1.0/mse), mse

    print(f"fully-on-device trainer: K={K} {size}x{size}, {STEPS} steps "
          f"(Tensix fwd+bwd, x280 whiten-bwd+Adam)")
    for step in range(1, STEPS+1):
        t0 = time.time()
        # params from x280 -> gs (orig order); z carried from init (fixed order)
        pw = dev.rdn(0, 0x30005800, K*9)
        gs = [tuple(bf(pw[o*9+j]) for j in range(9)) + (init[o][9],) for o in range(K)]
        fwd = SP.render_ondevice(coord, ctx=ctx, k=K, size=size, gs=gs, order=order,
                                 prebuilt=True, verbose=False)
        pval, mse = psnr(fwd["rgb"])
        dLdC = [[2.0*(fwd["rgb"][p][ch]-target[p][ch])/(P*3) for ch in range(3)] for p in range(P)]
        bw = SP.backward_ondevice(coord, fwd, dLdC, ctx=ctx, prebuilt=True, verbose=False)
        dLdpsi, dLdop, dLdcol = bw["dLdpsi"], bw["dLdop"], bw["dLdcolor"]
        # assemble sorted-slot grads [d_sa,d_m12,d_tx,d_m22,d_ty,dLdop,dc0,dc1,dc2]
        gradin = []
        for i in range(K):
            gradin.append([dLdpsi[0][2*i], dLdpsi[1][2*i], dLdpsi[2][2*i],
                           dLdpsi[1][2*i+1], dLdpsi[2][2*i+1],
                           dLdop[i], dLdcol[i][0], dLdcol[i][1], dLdcol[i][2]])
        bc1 = 1.0/(1-0.9**step); bc2 = 1.0/(1-0.999**step)
        LR = [0.15,0.15,2e-3,2e-3,2e-3,0.02,0.1,0.1,0.1]              # per-param Adam LR (now host-supplied)
        hdr = [K, step, fb(bc1), fb(bc2), fb(0.9), fb(0.999), fb(1e-8)] + [fb(x) for x in LR]
        dev.wr(0, 0x30005100, [fb(gradin[i][j]) for i in range(K) for j in range(9)])
        dev.wr(0, 0x30005000, hdr)                                   # K,step,bc1,bc2,b1,b2,eps,lr[9]
        dev.wr(0, 0x30004000, [step])                                # ring x280
        for _ in range(60):
            if dev.rd(0, 0x30004010) == step: break
            time.sleep(0.03)
        print(f"  step {step:2d}: PSNR={pval:5.2f} dB  loss={mse:.5f}  ({time.time()-t0:.0f}s)")

    # final render with the trained params
    pw = dev.rdn(0, 0x30005800, K*9)
    gs = [tuple(bf(pw[o*9+j]) for j in range(9)) + (init[o][9],) for o in range(K)]
    fwd = SP.render_ondevice(coord, ctx=ctx, k=K, size=size, gs=gs, order=order, prebuilt=True, verbose=False)
    pval, _ = psnr(fwd["rgb"])
    print(f"  final : PSNR={pval:5.2f} dB  (trained fully on-device)")

    # side-by-side: trained render | target  -> renders/splat_trained_ondevice.png
    import zlib
    UP, GAP = 14, 16; TW = size*UP; W = TW*2+GAP; H = TW
    cv = [[0.12, 0.12, 0.15] for _ in range(W*H)]
    for src, x0 in [(fwd["rgb"], 0), (target, TW+GAP)]:
        for y in range(H):
            for x in range(TW):
                cv[y*W+x0+x] = [min(1., max(0., src[(y//UP)*size+(x//UP)][ch])) for ch in range(3)]
    raw = bytearray()
    for y in range(H):
        raw.append(0)
        for x in range(W):
            for ch in range(3): raw.append(max(0, min(255, int(cv[y*W+x][ch]*255+.5))))
    def ch_(t, d): return struct.pack(">I", len(d))+t+d+struct.pack(">I", zlib.crc32(t+d) & 0xffffffff)
    png = (b"\x89PNG\r\n\x1a\n" + ch_(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
           + ch_(b"IDAT", zlib.compress(bytes(raw), 9)) + ch_(b"IEND", b""))
    open("/home/starboy/bhtop/src/bhtop/het/poc/renders/splat_trained_ondevice.png", "wb").write(png)
    print("  wrote het/poc/renders/splat_trained_ondevice.png (left = trained on-device, right = target)")

if __name__ == "__main__":
    main()
