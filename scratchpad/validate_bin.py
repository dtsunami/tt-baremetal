"""Compare the ON-DEVICE bin (het cmd11 -> IDLGB) against the host golden BIN.bin_tiles for the SAME pub,
to find why the device-bin loss differs. Prints per-tile occupancy + front-12 id-set match."""
import sys, numpy as np
sys.path.insert(0, "/home/starboy/bhtop/src"); sys.path.insert(0, "/home/starboy/bhtop/scratchpad")
from bhtop.het import grid_engine as GE
from bhtop.het.grid_engine import HetGridEngine, _bf, PARAM, OCC
import gap2_bin_golden as BIN

d = np.load(sys.argv[1]); N = 2000; params = d["params"][:N]
eng = HetGridEngine(N, 128, 128, W=6, NH=4); eng.set_params(params)
cam = [float(x) for x in d["poses"][0]]
eng.dev.wr(0, GE.X_CAM, [GE._fb(x) for x in cam])
eng._het(2, extra=[N, 1])
off = N * 61
pub = np.array([[_bf(u) for u in eng.dev.rdn(0, PARAM + (off + o * 6) * 4, 6)] for o in range(N)])
for i in range(4):                                     # host ext for the first Gaussians (bbox sanity)
    gx, gy, a, b, c, dep = pub[i]; det = a * c - b * b
    if det > 0:
        A = c / det; C = a / det; ex = 3 * np.sqrt(max(A, 0)); ey = 3 * np.sqrt(max(C, 0))
        print(f"G{i}: a={a:.4f} b={b:.4f} c={c:.4f} det={det:.5f} ex={ex:.1f}px ey={ey:.1f}px "
              f"gx={gx:.1f} gy={gy:.1f} dep={dep:.2f}")
    else:
        print(f"G{i}: det={det:.5f} (culled)")
eng._het(11)
occc = eng.dev.rdn(0, OCC, 64)
tiles, ntx, nty = BIN.bin_tiles(pub[:, 0], pub[:, 1], pub[:, 2:5], pub[:, 5], 128, 128, tile=16, cap=64)
IDLGB = 0x35000000
nmiss = 0
for t in range(64):
    dl = eng.dev.rdn(0, IDLGB + t * 0x40, 13); dcnt = dl[0]
    dev_ids = set(dl[1:1 + min(dcnt, 12)])
    host_ids = set(tiles[t][:12])
    if dev_ids != host_ids: nmiss += 1
    if t < 8:
        print(f"tile {t}: occ dev={occc[t]} host={len(tiles[t])} | ids match={dev_ids==host_ids} "
              f"dev={sorted(dev_ids)[:6]} host={sorted(host_ids)[:6]}")
print(f"tiles with mismatched front-12 set: {nmiss}/64")
