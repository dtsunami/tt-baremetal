"""FUSED streaming splat pipeline: x280 (sort, producer) ⇄ Tensix (forward render, consumer) through a
GDDR ring with backpressure. The x280 argsorts each tile front-to-back and streams the order into a
bounded ring; the Tensix render drains each tile as it arrives and acks to unblock the producer."""
import sys, struct, time, zlib
sys.path.insert(0, "/home/starboy/bhtop/src")
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
from bhtop.tensix.loader import TensixLauncher
from bhtop.tensix import splat as SP

T, N, K, size = 4, 2, 16, 16
ctx = init_ttexalens(); dev = L2cpu(ctx=ctx)
fb = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]

dev.wr(0, 0x30002000, [0]); dev.wr(0, 0x30002010, [0]); dev.wr(0, 0x30002020, [0])
scenes = [SP.scene_rgb(k=K, seed=10 + t, span=float(size)) for t in range(T)]
host_orders = [sorted(range(K), key=lambda i: scenes[t][i][9]) for t in range(T)]
depths = []
for sc in scenes:
    depths += [fb(g[9]) for g in sc]
dev.wr(0, 0x30002040, depths)                         # stage T×K depths (uncached)

pw = tc.compile_source("/home/starboy/bhtop/src/bhtop/kernels/x280/het/cb_render_producer.c", base=CODE_ADDR, march="rv64gc")
for _ in range(6):
    dev.load(0, 0, pw); time.sleep(0.25)
    if dev.rd(0, 0x30002000) > 0 or dev.rd(0, 0x30002020) == 0xD09E: break
print("x280 producer streaming; produced =", dev.rd(0, 0x30002000), f"(fills ring N={N} then waits on ack)")

L = TensixLauncher.at(1, 2, ctx=ctx)
tiles = []; sort_ok = True
for t in range(T):
    while dev.rd(0, 0x30002000) <= t:                 # backpressure/handshake: wait for tile t
        time.sleep(0.02)
    order = dev.peek(0, 0x30002200 + (t % N) * 64, K)  # x280's streamed sort order for tile t
    sort_ok = sort_ok and (order == host_orders[t])
    r = SP.render_ondevice(L.coord, ctx=ctx, k=K, size=size, seed=10 + t,
                           order=order, gs=scenes[t], verbose=False)      # Tensix renders it
    tiles.append(r)
    dev.wr(0, 0x30002010, [t + 1])                     # ack -> unblock producer
    print(f"  tile{t}: streamed-order==host {order == host_orders[t]} | render {r['psnr_db']:.1f} dB")

pdone = dev.rd(0, 0x30002020)
allok = sort_ok and all(x["ok"] for x in tiles) and pdone == 0xD09E
print(f"\nFUSED PIPELINE {'PASS' if allok else 'CHECK'}: {T} tiles streamed "
      f"x280(sort) -> ring(N={N}) -> Tensix(render); backpressure held (wrapped {T-N}×); "
      f"producer done={hex(pdone)}")

# visual: device (top) over golden (bottom), T tiles wide
UP, GAP = 12, 10; TW = size * UP; W = T * TW + (T + 1) * GAP; H = 2 * TW + 3 * GAP
cv = [[0.1, 0.1, 0.13] for _ in range(W * H)]
def blit(src, x0, y0):
    for y in range(TW):
        for x in range(TW):
            cv[(y0 + y) * W + x0 + x] = [min(1., max(0., src[(y // UP) * size + (x // UP)][c])) for c in range(3)]
for t in range(T):
    blit(tiles[t]["rgb"], GAP + t * (TW + GAP), GAP)
    blit(tiles[t]["golden"], GAP + t * (TW + GAP), 2 * GAP + TW)
def ch(tp, d): c = struct.pack(">I", len(d)) + tp + d; return c + struct.pack(">I", zlib.crc32(tp + d) & 0xffffffff)
raw = bytearray()
for y in range(H):
    raw.append(0)
    for x in range(W):
        for c in range(3): raw.append(max(0, min(255, int(cv[y * W + x][c] * 255 + .5))))
open("/home/starboy/bhtop/src/bhtop/het/poc/renders/splat_streaming.png", "wb").write(
    b"\x89PNG\r\n\x1a\n" + ch(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    + ch(b"IDAT", zlib.compress(bytes(raw), 9)) + ch(b"IEND", b""))
print("wrote het/poc/renders/splat_streaming.png (top = streamed x280→Tensix renders, bottom = golden)")
