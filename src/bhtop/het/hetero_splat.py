"""Heterogeneous Gaussian-splat render: x280 owns the irregular tier (depth sort), the bare-metal
Tensix grid owns the dense tier (eval + exp + composite). Both cooperate on ONE shared exalens ctx."""
import sys, struct, time
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import splat as SP

SORT_SRC = "/home/starboy/bhtop/src/bhtop/kernels/x280/het/depth_sort.c"
IN_ADDR = 0x30002300      # uncached hart-3 tele window (host-written input)

def x280_sort(dev, tile, z, sort_words):
    """Run the x280 depth-sort on `z` (list of non-negative floats), return the sorted index order."""
    K = len(z)
    fb = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]
    dev.wr(tile, IN_ADDR, [K] + [fb(v) for v in z])
    for _ in range(6):
        dev.load(tile, 0, sort_words); time.sleep(0.33)
        t = dev.telemetry(tile, slots=2 + K, hart=0)
        if t[0] == 0x50575254 and t[1] == K:
            return [t[2 + i] for i in range(K)]
    raise RuntimeError(f"x280 sort did not complete (magic={t[0]:#x} K={t[1]})")

def main():
    ctx = init_ttexalens()                       # ONE shared context for both engines
    dev = L2cpu(ctx=ctx)                          # x280 (already brought up this session; persists)
    tile = 0
    sort_words = tc.compile_source(SORT_SRC, base=CODE_ADDR, march="rv64gc")

    size, K = 16, 16
    gs = SP.scene_rgb(k=K, seed=5, span=float(size))
    z = [g[9] for g in gs]

    order = x280_sort(dev, tile, z, sort_words)   # <-- x280 does the irregular work
    host_order = sorted(range(K), key=lambda i: z[i])
    print(f"x280 depth-sort order == host: {order == host_order}")

    L = TensixLauncher.at(1, 2, ctx=ctx)          # <-- Tensix does the dense work, same ctx
    r = SP.render_ondevice(L.coord, ctx=ctx, k=K, size=size, seed=5, order=order, gs=gs)
    print(f"HETERO render (x280 sort + Tensix eval/exp/composite): PSNR = {r['psnr_db']:.1f} dB  "
          f"-> {'PASS' if r['ok'] else 'CHECK'}")
    return r

if __name__ == "__main__":
    r = main()
    # save image
    import zlib
    rgb, gold, N = r["rgb"], r["golden"], 16
    def png(pix, W, H, path):
        raw = bytearray()
        for y in range(H):
            raw.append(0)
            for x in range(W):
                for ch in range(3): raw.append(max(0, min(255, int(pix[y*W+x][ch]*255+0.5))))
        def ch(t, d): c = struct.pack(">I", len(d))+t+d; return c+struct.pack(">I", zlib.crc32(t+d)&0xffffffff)
        open(path, "wb").write(b"\x89PNG\r\n\x1a\n"+ch(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))+ch(b"IDAT", zlib.compress(bytes(raw), 9))+ch(b"IEND", b""))
    UP, GAP = 14, 16; TW = N*UP; W = TW*2+GAP; H = TW
    canvas = [[0.12, 0.12, 0.15] for _ in range(W*H)]
    for src, x0 in [(rgb, 0), (gold, TW+GAP)]:
        for y in range(H):
            for x in range(TW): canvas[y*W+x0+x] = [min(1.0, max(0.0, src[(y//UP)*N+(x//UP)][c])) for c in range(3)]
    png(canvas, W, H, "/home/starboy/bhtop/src/bhtop/het/poc/renders/splat_hetero.png")
    print("wrote het/poc/renders/splat_hetero.png (left = x280+Tensix hetero, right = golden)")
