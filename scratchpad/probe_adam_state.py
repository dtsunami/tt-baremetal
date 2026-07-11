"""Pin WHAT the worker is doing during the slow cmd1: manually drive Adam (cmd1) and sample each hart's
breadcrumb (WDIAG state + WHB heartbeat) while polling DONE — single-threaded, no concurrent exalens. If harts
sit in WS_ADAM with an advancing heartbeat, the 12s is slow adam_slice COMPUTE (not the barrier, not an
ack-wait) — resolving the "worker stuck ~12s before adam" mystery."""
import sys, os, time, math
sys.path.insert(0, "/home/starboy/bhtop/src")
import numpy as np
from bhtop.het.grid_engine import HetGridEngine, WDIAG, WHB, X_CMD, X_DB, X_DONE, X_HDR, X_CAM, IMG_BASE_A, TGT_BANK, _WS
import struct
_fb = lambda x: struct.unpack("<I", struct.pack("<f", float(x)))[0]

N, IMG, W, NH = 4000, 256, 8, 4
rng = np.random.default_rng(1234)
P = np.zeros((N, 14), np.float64)
P[:, 0] = rng.uniform(-0.92, 0.92, N); P[:, 1] = rng.uniform(-0.92, 0.92, N); P[:, 2] = 1.0
P[:, 3:6] = math.log(0.02); P[:, 6] = 1.0; P[:, 10] = 0.5; P[:, 11:14] = rng.uniform(0.1, 0.9, (N, 3))
cam16 = [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, IMG / 2, IMG / 2, IMG / 2, IMG / 2]
yy, xx = np.mgrid[0:IMG, 0:IMG].astype(np.float32) / IMG
tgt = np.stack([xx, yy, 0.5 * (xx + yy)], -1).astype(np.float32)
LR = [0.005, 0.005, 0.005, 0.003, 0.003, 0.003, 0.001, 0.001, 0.001, 0.001, 0.02, 0.005, 0.005, 0.005]

eng = HetGridEngine(N, IMG, IMG, W=W, NH=NH); eng.set_params(P); eng.set_views(tgt[None])
print("[probe] booted", flush=True)
# warm one full step
eng.step(cam16, tgt.reshape(-1).astype(np.float64), LR, 1, view_idx=0)

# now drive step 2's proj+orch normally, then cmd1 by hand with sampling
d = eng.dev
d.wr(0, X_CAM, [_fb(x) for x in cam16])
d.wr(0, IMG_BASE_A, [TGT_BANK])  # IMG_BASE -> resident view 0
eng._het(2, extra=[N, 2])
eng._het(10, timeout=60.0)

# manual cmd1 (mirror _het_multi for tile 0) with live sampling
bc1 = 1.0 / (1 - 0.9 ** 2); bc2 = 1.0 / (1 - 0.999 ** 2)
ext = [N, 2, _fb(bc1), _fb(bc2), _fb(0.9), _fb(0.999), _fb(1e-8)] + [_fb(x) for x in LR] + [0, 0, 0, 0, N]
d.wr(0, X_HDR, ext); d.wr(0, X_CMD, [1]); r = d.rd(0, X_DB) + 1; d.wr(0, X_DB, [r])
t0 = time.time(); samples = 0
while d.rd(0, X_DONE) != r and time.time() - t0 < 20.0:
    st = []
    for h in range(NH):
        hb = d.rd(0, WHB + h * 0x40); s = d.rd(0, WDIAG + h * 0x40)
        st.append(f"h{h}:{_WS.get(s, s)}/hb{hb}")
    print(f"  t={time.time()-t0:5.1f}s cmd1 {'  '.join(st)}", flush=True)
    samples += 1; time.sleep(1.0)
print(f"[probe] cmd1 done in {time.time()-t0:.1f}s ({samples} samples), werr=0x{d.rd(0,0x30002604):08x}", flush=True)
