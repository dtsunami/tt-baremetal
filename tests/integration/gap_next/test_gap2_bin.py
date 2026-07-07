"""Gap 2 silicon de-risk: bin_tiles.c on the x280 buckets projected Gaussians into per-16x16-tile
depth-sorted id lists. Feed a random projected set (using the golden proj_fwd so covariances are real),
compare the x280 per-tile counts + sorted ids to gap2_bin_golden. Recover wedges with tt-smi -r 0."""
import sys, struct, math, time
sys.path.insert(0, "/home/starboy/bhtop/src"); sys.path.insert(0, "/home/starboy/bhtop/scratchpad")
import numpy as np
from ttexalens import init_ttexalens
from bhtop.l2cpu import L2cpu, toolchain as tc, CODE_ADDR
import gap1_proj_golden as G
import gap2_bin_golden as B

fb = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]
bf = lambda u: struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]
SRC = "/home/starboy/bhtop/src/bhtop/kernels/x280/het/bin_tiles.c"
CAP = int(sys.argv[1]) if len(sys.argv) > 1 else 32     # small CAP forces the depth-cull eviction (Gap 4)
W = Hh = 48; TILE = 16          # 3x3 = 9 tiles
N = 16
# camera lands Gaussians across the 48x48 frame
Rv = np.eye(3); tv = np.array([0.0, 0.0, 4.0]); fx = fy = 60.0; cx = cy = 24.0


def main():
    rng = np.random.default_rng(4)
    mean = rng.normal(0, 1.3, (N, 3)); sl = np.log(0.15 + rng.random((N, 3)) * 0.4); q = rng.normal(0, 1, (N, 4))
    gx = np.zeros(N); gy = np.zeros(N); conic = np.zeros((N, 3)); depth = np.zeros(N)
    pub = []
    for i in range(N):
        u, vv, dep, a, b, c, _ = G.project_forward(mean[i], sl[i], q[i], Rv, tv, fx, fy, cx, cy)
        gx[i], gy[i], conic[i], depth[i] = u, vv, (a, b, c), dep
        pub.append([u, vv, a, b, c, dep])

    tiles_gold, ntx, nty = B.bin_tiles(gx, gy, conic, depth, W, Hh, tile=TILE, cap=CAP)
    ntiles = ntx * nty

    ctx = init_ttexalens(); dev = L2cpu(ctx=ctx)
    try: dev.bringup(0); print("[setup] x280 bringup ok")
    except Exception as e: print("[setup] bringup:", type(e).__name__, "(already up)")

    dev.wr(0, 0x30005000, [N, W, Hh, TILE])
    dev.wr(0, 0x30005010, [fb(v) for row in pub for v in row])
    dev.wr(0, 0x30006000, [0] * ntiles)
    dev.wr(0, 0x30006400, [0] * (ntiles * CAP))
    print("[run] loading bin_tiles.c")
    dev.load(0, 0, tc.compile_source(SRC, base=CODE_ADDR, march="rv64gc", defines={"CAP": CAP}))
    for _ in range(80):
        if dev.telemetry(0, slots=1, hart=0)[0] == 0x42494E21: break
        time.sleep(0.03)
    else:
        print("FAIL: bin_tiles never signalled BIN!"); return

    cnt = list(dev.rdn(0, 0x30006000, ntiles))
    ids_flat = list(dev.rdn(0, 0x30006400, ntiles * CAP))
    ok = True; mism = 0
    print(f"image {W}x{Hh} -> {ntx}x{nty}={ntiles} tiles, N={N}")
    for t in range(ntiles):
        dev_ids = [ids_flat[t * CAP + k] & 0xFFFFFFFF for k in range(cnt[t])]
        gold_ids = tiles_gold[t]
        match = (cnt[t] == len(gold_ids)) and (dev_ids == gold_ids)
        if not match:
            ok = False; mism += 1
            if mism <= 6:
                print(f"  tile {t:2d} MISMATCH  dev(n={cnt[t]})={dev_ids}  gold(n={len(gold_ids)})={gold_ids}")
        else:
            if t < 3 or cnt[t]:
                print(f"  tile {t:2d} ok  n={cnt[t]} ids={dev_ids}")
    total = sum(cnt)
    print(f"\ntotal tile-touches dev={total} gold={sum(len(g) for g in tiles_gold)}")
    print("GAP2_BIN_SILICON_OK" if ok else "GAP2_BIN_SILICON_FAIL")


if __name__ == "__main__":
    main()
