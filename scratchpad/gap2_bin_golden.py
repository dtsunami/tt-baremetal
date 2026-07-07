"""Gap 2 golden: multi-tile binning. Each projected Gaussian (screen mean gx,gy, conic a,b,c = Sigma2^-1,
depth) touches every 16x16 tile its 3-sigma screen bbox overlaps; per tile we keep the touching Gaussian
ids DEPTH-SORTED (near->far). This is the reference the x280 bin_tiles.c kernel is validated against."""
import math

TILE = 16
SIGMA = 3.0
CAP = 64                     # max Gaussians kept per tile (depth-cull beyond)


def bin_tiles(gx, gy, conic, depth, W, H, tile=TILE, sigma=SIGMA, cap=CAP):
    ntx, nty = W // tile, H // tile
    tiles = [[] for _ in range(ntx * nty)]
    for i in range(len(gx)):
        a, b, c = conic[i]
        det = a * c - b * b
        if det <= 0:
            continue
        A = c / det; C = a / det                       # Sigma2 diagonal (screen variance x,y)
        ext_x = sigma * math.sqrt(max(A, 0.0)); ext_y = sigma * math.sqrt(max(C, 0.0))
        tx0 = max(0, int(math.floor((gx[i] - ext_x) / tile))); tx1 = min(ntx - 1, int(math.floor((gx[i] + ext_x) / tile)))
        ty0 = max(0, int(math.floor((gy[i] - ext_y) / tile))); ty1 = min(nty - 1, int(math.floor((gy[i] + ext_y) / tile)))
        if tx1 < tx0 or ty1 < ty0:
            continue                                    # bbox entirely off-image
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                tiles[ty * ntx + tx].append(i)
    for t in range(len(tiles)):
        tiles[t] = sorted(tiles[t], key=lambda i: depth[i])[:cap]
    return tiles, ntx, nty
