# Scaling the bare-metal resident trainer — from Stage 0 to 1600px + millions

Companion to `RESIDENT_GRID.md` (the kernels) and `baremetal_plan.md` (the POC). This is the roadmap from
what runs today to full-resolution, many-Gaussian training on the bare-metal Tensix + x280 substrate.

## Where we are (Stage 0 — proven on silicon)

A **fully-fused resident training step** works: one Tensix worker does forward render + on-device loss
gradient (`dLdC = C − gt`) + backward in a doorbell-driven kernel (`resident_train_perf`, 27 stages), the
x280 does whiten-backward + Adam, and the loop **converges** (13.6 → 19.7 dB on a fit-one-tile smoke test).
The 120-worker resident grid, the fused render, the fused backward, the matmul↔eltwise mode-switch, the
on-device transpose, and cycle telemetry are all proven. **But it is a Stage-0 primitive:**

- **One 16×16 tile**, **K ≤ 16 Gaussians**, flat RGB.
- **Orthographic-identity projection** — `mean.xy` used directly; no real camera.
- Gaussians assumed to cover the whole tile (no visibility/binning).
- Params host- or x280-resident for ≤16 Gaussians (fits L1 easily).

Everything below is what stands between that and 1600px + millions. None of it is a feasibility unknown —
the compute substrate is proven; this is the surrounding *dataflow* plumbing.

## The gaps, in dependency order

### 0. Fix multi-ring residency (prerequisite; small)
`resident_train_perf` completes ring 1 but stalls on ring > 1 (the 27-stage SyncHalf dest handshake doesn't
return to a clean state across the ring boundary — see RESIDENT_GRID.md and the note below). The trainer
re-boots per ring today (~0.3 s) which works but isn't truly resident. Needed for the grid to hold the
kernel resident and be driven at one ring/tile. **Effort: small** — instrument the dest/`MATH_PACK`
semaphore at the ring boundary, then fully reset the *hardware* dest-sync (not just the software offset)
at ring start, or balance the per-ring dest-section accounting.

### 1. Real camera projection (3D → 2D + covariance) — on the x280
Today's Stage 0 skips projection. Full-res needs, per Gaussian per view: `mean → clip → NDC → screen`,
the 2×2 screen covariance from the 3×3 world covariance via the perspective Jacobian, then the whitening
coefficients ψ the render consumes. The x280 already computes ψ from `(Σ⁻¹, mean)` in fp32 (`cb_whiten.c`,
4.2e-7 vs host) — extend it to the full 3D→2D projection (mean, R·diag(exp scale)²·Rᵀ, J, invert). This is
the "Gap 1" the earlier proj_golden proof grad-checked; port it onto the x280. **Effort: medium.** Its
analytic backward (2D grads → 3D mean/scale/quat grads) also lives on the x280 next to the optimizer.

### 2. Multi-tile binning — which Gaussians touch which tile (x280 counting-sort)
At 1600px a Gaussian touches a small box of tiles, not the whole image. Need: for each projected Gaussian,
its 3σ screen bounding box → the set of tiles it covers → a per-tile Gaussian list. This is the classic
tile-binning / counting-sort, and the x280 is its home (the irregular tier — the depth-sort + gather +
scatter-add are all proven x280 primitives). Output: per-tile `(gaussian ids, depth order)` in GDDR.
**Effort: medium** — reuse the x280 sort/gather; add the bbox→tile expansion and the counting-sort bucket.

### 3. Tile → worker grid orchestration
1600×1600 = **10 000 tiles of 16×16**; 120 workers ⇒ ~83 tiles/worker. The host (or a resident scheduler)
assigns tiles to workers, streams each tile's operands (its binned Gaussians' consts + pixels + gt), rings
`resident_train_perf` once/tile, and accumulates per-Gaussian grads across all tiles that touched each
Gaussian (the x280 scatter-add — proven). The 120-worker resident grid (proven, 240/240 bit-exact) is
exactly this substrate; this step is the driver loop + the grad scatter-add wiring. **Effort: medium**,
gated on gaps 0–2.

### 4. K > 16 Gaussians per tile
A dense tile can have >16 Gaussians. Options: **depth-cull top-16** (keep the 16 nearest, alpha-composite
error is small — the earlier multitile proof at 48.7 dB), or **RT_DIM > 1** multi-pass (composite in
chunks of 16). The render/backward kernels already loop; extend the tile Gaussian dimension. **Effort:
small–medium.**

### 5. Millions of Gaussians — GDDR-resident params + streaming
L1 is 128 KB/core; millions of Gaussians' params (9 floats each) don't fit. Params live in **GDDR**, the
x280 streams each tile's Gaussians into the worker's L1 per tile (operand streaming — item 2 in the grid
doc; the x280→GDDR→Tensix operand ring is proven for the render). Adam m/v also GDDR-resident. **Effort:
medium** — the streaming path is proven; scale it + the GDDR param layout + densify realloc.

### 6. Densification / pruning (grow toward millions)
Clone/split high-gradient Gaussians, prune transparent/degenerate ones — changes N. Realloc the GDDR param
+ Adam m/v buffers preserving survivors' momentum + the step counter (mirror `DeviceAdam.prune`). Periodic,
host-orchestrated. **Effort: medium.**

## What's already proven (the substrate you build on)

| Piece | Status |
|---|---|
| 120-worker resident grid, parallel drive | ✅ 240/240 bit-exact |
| Fused resident render (11→6 stages) | ✅ 51 dB, 1 ring/tile, group-loop |
| Fused resident backward (17 stages) | ✅ leaf grads <1.2% |
| Fused fwd+dLdC+bwd training step | ✅ RGB 52.6 dB + grads; converges 13.6→19.7 dB |
| matmul↔eltwise mode-switch, on-device transpose, SFPU recip | ✅ |
| x280: whiten-bwd + Adam, depth-sort, gather, scatter-add, ψ-project, operand ring | ✅ (POC-proven) |
| Cycle telemetry (per-stage device cycles) | ✅ |

## Suggested order

1. **Fix multi-ring residency** (gap 0) — unblocks true residency; small.
2. **x280 projection + its backward** (gap 1) — the biggest correctness piece; makes real 3D scenes train.
3. **Binning + tile→worker grid + grad scatter-add** (gaps 2–3) — unlocks multi-tile → arbitrary resolution.
4. **K>16, GDDR params + streaming, densify** (gaps 4–6) — unlocks density → millions.

Each is dataflow plumbing on a proven compute substrate. The ttnn `TT_DEVICE_RESIDENT` path (which already
trains 1600px/50k) is the reference for correctness at scale; the bare-metal path here is the efficiency
play — the value is that the dense inner loop never leaves the chip and the x280 owns the irregular tier.
