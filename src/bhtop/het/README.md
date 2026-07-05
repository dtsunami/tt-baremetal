# `bhtop.het` — heterogeneous bare-metal compute on Blackhole

The x280 (L2CPU RISC-V + RVV) and the Tensix grid cooperating as **co-equal dataflow peers over
tt-exalens, with zero tt-metal**, sharing GDDR — demonstrated on a fully-on-device Gaussian-splatting
trainer. The x280 owns the irregular tier (sort / gather / scatter-add); the Tensix grid owns the
dense tier (matmul-eval, raster, SFPU). Full design + silicon log: [`../../../baremetal_plan.md`](../../../baremetal_plan.md).

## Why this lives in bhtop
The whole POC is built on bhtop's own primitives — `init_ttexalens` attach, `bhtop.l2cpu.L2cpu`
bringup, `BabyRiscDebug` reset control, the extracted NoC0 read/write, the LLK build/boot lane, and the
`bhtop.tensix.baremetal.BareMetal` launcher. bhtop owns the machine; this package is the machine being
driven. (The splatting *algorithm* was prototyped in the separate `tt-splat` repo and reimplemented
here on the bare-metal substrate.)

## Layout
| path | what |
|---|---|
| `bhtop.tensix.baremetal` | `BareMetal` — the tt-metal-free Tensix launcher (3rd launch path) |
| `bhtop.tensix.{matmul,sfpu,splat}` | MVMUL / SFPU / on-device splat render harness (pure-Python host) |
| `bhtop.tensix.{llk,llk_run}` | TRISC boot-over-exalens + parametric LLK build (fidelity/formats/fp32) |
| `bhtop.l2cpu` | x280 bringup / compile / load / telemetry |
| `bhtop/kernels/tensix/baremetal/` | Tensix cold-boot C canon (crt0, NoC read/write, cb_consumer, …) |
| `bhtop/kernels/x280/het/` | x280 POC C kernels (depth sort/gather, CB producer, whiten, scatter) |
| `het/*.py` | orchestration drivers (below) |
| `het/poc/` | the interview artifact: `gen_poc.py` → `hetero_poc.html`, `renders/` silicon proof |

## Drivers
- `hetero_splat.py` — x280 depth sort → Tensix `render_ondevice(order=…)`; one frame, two engines.
- `render_streaming.py` — **fused streaming**: x280 sorts + streams each tile through a GDDR ring;
  the Tensix render is the consumer, acks to unblock. Backpressure held on silicon.
- `cb_loop.py` — the M2 DRAM circular buffer in isolation (produced/acked backpressure).
- `ondevice_fwd.py` — fully-on-device forward render (6 MVMUL + 5 SFPU, no host arithmetic).
- `train_ondevice.py` — **the fully-on-device trainer (loop closed).** Tensix forward → host loss-grad →
  Tensix dense backward → x280 whiten-backward + un-sort + Adam (resident, doorbell-driven); params + Adam
  m/v live on the x280. Every model op on device; host only orchestrates + the loss gradient. `train_ondevice.py N`.
- `train_color.py` / `train_geometry.py` — earlier host-backward training references (color; full geometry, grad-checked).

## Run
```bash
tt-smi -r 0                      # clean baseline (also resets the x280 → re-bringup)
~/bhtop/.venv/bin/python -m bhtop.het.render_streaming     # from repo root, or:
cd ~/bhtop && ~/bhtop/.venv/bin/python src/bhtop/het/render_streaming.py
```
Tensix worker is noc0 `(1,2)`; x280 is L2CPU tile-0 `(8,3)`. Recovery from any wedge: `tt-smi -r 0`.
Gotchas (uncached GDDR window `0x30002000–0x30007000`, 64B-aligned NoC DRAM reads, x280 scalar-float
`csrs mstatus,0x6000`) are documented in the plan.

## Silicon status (2026-07-05)
**Fully-on-device training loop CLOSED.** `train_ondevice.py`: Tensix forward + Tensix dense backward
(`splat.backward_ondevice`, all bf16 matmul/eltwise/reciprocal) + x280 whiten-backward/un-sort/Adam
(`kernels/x280/het/opt_step.c`, resident, fp32); params + optimizer state live on the x280, host only
orchestrates + the loss gradient. Converges **13.4 → 33.9 dB over 100 steps** (~1 s/step), monotonic.
Image: `poc/renders/splat_trained_ondevice.png`.
- Forward: fully on-device + streaming, ~52.9 dB vs golden, zero host Gaussian-data relay.
- The composite backward's bf16 cancellation (~27% on `dL/dalpha`) is near-zero-mean → averages out of
  the pixel reductions; leaf grads dLdpsi/dLdop/dLdcolor are 0.4–0.7% vs host-exact. (Triage in the plan.)
- **Open (perf, not correctness):** cut the ~190 dispatches/step (multi-tile matmul `RT_DIM>1`, fused SFPU);
  move `dLdC` (loss grad) onto the x280 too for a zero-host-arithmetic claim; scale K/resolution.
