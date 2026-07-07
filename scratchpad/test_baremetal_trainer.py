"""Standalone silicon test of tt-splat's BareMetalResidentTrainer: construct it with a tt-splat-shaped
P dict + per-param LR dict, drive .step(cam,gt,mask) to convergence. Proves the DeviceResidentTrainer
contract + the config->x280-header meta-param plumbing on real hardware, no ttnn/tt-metal."""
import sys, math
sys.path.insert(0, "/home/starboy/tt-splat/server")
sys.path.insert(0, "/home/starboy/bhtop/src")
import numpy as np
from baremetal_resident import BareMetalResidentTrainer, SIZE
from bhtop.tensix import splat as SP

K = 12
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 40

def scene_to_P(gs):
    """bhtop 2D scene tuples -> tt-splat P dict (numpy), inverting the trainer's Stage-0 projection."""
    N = len(gs)
    mean = np.array([[g[0], g[1], g[9]] for g in gs], np.float64)
    scale = np.array([[math.log(1.0 / math.sqrt(max(g[2], 1e-6))),
                       math.log(1.0 / math.sqrt(max(g[4], 1e-6))), 0.0] for g in gs], np.float64)
    op = np.array([math.log(g[5] / max(1.0 - g[5], 1e-6)) for g in gs], np.float64)
    sh = np.zeros((N, 1, 3), np.float64)
    for i, g in enumerate(gs): sh[i, 0] = [g[6], g[7], g[8]]
    quat = np.tile(np.array([1.0, 0, 0, 0]), (N, 1))
    return {"mean": mean, "scale": scale, "quat": quat, "op": op, "sh": sh, "deg": 0}

tgt = SP.scene_rgb(k=K, seed=11, span=float(SIZE))
init = SP.scene_rgb(k=K, seed=22, span=float(SIZE))
# tt-splat-style per-param LR dict (proves config LRs flow through the x280 header)
lr = {"mean": 0.15, "scale": 2e-3, "quat": 0.01, "op": 0.02, "sh": 0.1}

print(f"BareMetalResidentTrainer: constructing (K={K}) — brings up x280 + Tensix, loads opt_step ...")
tr = BareMetalResidentTrainer(dev=None, P=scene_to_P(init), lr=lr, deg=0, lambda_dssim=0.0)
print(f"  contract surface: N={tr.N} deg={tr.deg} adam.lr={tr.adam.lr}")

# device-render the target scene once (through the trainer's substrate)
tgt_order = sorted(range(K), key=lambda i: tgt[i][9])
tfwd = SP.render_ondevice(tr._coord, ctx=tr._ctx, k=K, size=SIZE, gs=tgt, order=tgt_order, prebuilt=True, verbose=False)
target = np.array(tfwd["rgb"], np.float64).reshape(SIZE, SIZE, 3)

# emulate the train_tt host loop: LR-decay mean via adam.lr mutation, call .step()
LR_DECAY, lrm0 = 0.05, tr.adam.lr["mean"]
print(f"fully-on-device training via the trainer contract, {STEPS} steps:")
for step in range(1, STEPS + 1):
    tr.adam.lr["mean"] = lrm0 * (LR_DECAY ** (step / STEPS))      # exactly what train_tt.py:443 does
    loss, img = tr.step(cam=None, gt=target, mask=None)
    if step in (1, 5, 10, 20, 30, STEPS):
        psnr = 99 if loss < 1e-12 else 10 * math.log10(1.0 / loss)
        print(f"  step {step:2d}: loss={loss:.5f} PSNR={psnr:5.2f} dB  (adam.lr[mean]={tr.adam.lr['mean']:.4f})")
print("  trainer.densify() ->", tr.densify())
ph = tr.params_host(); print(f"  params_host(): mean{ph['mean'].shape} scale{ph['scale'].shape} op{ph['op'].shape}")
