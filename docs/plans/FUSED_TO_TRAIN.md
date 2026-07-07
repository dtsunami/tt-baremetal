# Fused → Train — the plan from the resident fused kernel to 1600px training

**Status (2026-07-06): Gaps 0–5 essentially CLOSED ON SILICON.** The full fused on-device 3DGS step is
multi-ring resident (Gap 0), and on top of it **real 3D camera projection (Gap 1), multi-tile binning
(Gap 2), the full-image multi-tile trainer with grad scatter-add (Gap 3), per-tile depth-cull (Gap 4), and
GDDR residency for MILLIONS of Gaussians (Gap 5 — 1,000,000 held + optimized bit-exact on silicon; 25 M
addressable in the 4 GiB local bank) all run on the Blackhole.** The x280↔big-GDDR "handoff" turned out to
be direct-mapped local GDDR (no handoff). Remaining to 1600px: the fast cached-alias blocked optimizer
(Gap-5 build), Gap 6 (densification), and the 1600px/50k integration + ttnn validation.
Supersedes the "Gap 0" sections of `RESIDENT_SCALE.md` / `RESIDENT_GRID.md` (their odd-count / dest-sync
theories were wrong — see below).

**New this session (all silicon-validated, files):** `kernels/x280/het/proj.h` (projection math, one truth),
`cb_whiten.c` (proj_fwd+whiten), `opt_proj_step.c` (proj_bwd+Adam/14, publish, cmd modes, grad sanitize,
MAXK=24), `bin_tiles.c` (Gap-2 binning); `het/train_resident3d.py` (single+multi-view 3D), `het/train_multitile3d.py`
(full-image); de-risk harnesses `scratchpad/{gap1_proj_golden.py,gap1_proj.c,gap1_check_c.py,gap2_bin_golden.py,
test_gap1_whiten.py,test_gap1_optproj.py,test_gap2_bin.py}`. Device gotcha: `tt-smi -r 0` between resident runs.

---

## What is DONE and proven (the fused kernel)

- **`resident_train_perf`** — the whole training step in ONE doorbell ring: forward render + on-device
  `dLdC = C−gt` + backward, 27 stages, one worker. **Multi-ring resident: 30/30 rings on one boot,
  RGB 52.59 dB + leaf grads** (`scratchpad/test_gap0_ab.py`).
- **`het/train_resident.py`** — resident trainer (no reboot): boot once, ring per group, x280 whiten-bwd +
  Adam. **Converges 13.6 → 20.8 dB over 6 steps, ~0.1 s/step** (~24× faster than the old per-ring reboot).
- **Substrate (all silicon-proven earlier):** 120-worker resident grid (240/240 bit-exact), fused render
  (51 dB), fused backward (<1.2%), x280 tier (whiten-bwd + Adam, depth-sort, gather, scatter-add, ψ-project,
  operand ring).

### The Gap-0 fix (do NOT re-litigate — it's a Blackhole errata)
The multi-ring stall was a **silicon errata triggered by the UNPACK MODE SWITCH** (`_llk_unpack_AB_matmul_`
↔ `_llk_unpack_AB_` / `_llk_unpack_A_`), invisible above the RTL (all config / full 16384-word CFG / 1276
debug-bus signals / C-globals bit-identical entering the wedge ring; immune to every reset/drain/chicken-bit
incl. clock gating). `render` (always matmul-unpack) runs infinite; anything that switches wedges in 3–5 rings.
**FIX:** never switch unpack mode. Do every elementwise op as **matmul-load + SFPU-binary**: load both
operands into DEST via matmul (`a@Iden → DEST0`, `b@Iden → DEST1`, identity `H_Iden @ 0x27000`), then
`test_utils::call_binary_sfpu_operation<dest_sync,is_fp32_dest_acc_en,APPROX_MODE,ckernel::BinaryOp::MUL|SUB,ITERATIONS>(0,1,0)`.
Reciprocal = SFPU unary (already in-DEST). `BinaryOp::SUB` = in0−in1. Cost ~1 extra cheap matmul/eltwise,
parallel across the grid, fully resident. Proof: `scratchpad/test_gap0_sfpubin.py` (50/50 rings). The x280
is **freed** — it owns the irregular tier (Gaps 1–3), not residency.

### The fused kernel today is a Stage-0 primitive
16×16 tile · K ≤ 16 Gaussians · **orthographic-identity projection** (`mean.xy` used directly) · one worker ·
flat RGB · params fit L1. Everything below turns that into 1600px + millions.

---

## The plan: Gaps 1–6 to 1600px (plumbing on a proven substrate; no feasibility unknowns)

Ordered by critical path. Reference for correctness at scale: the ttnn `TT_DEVICE_RESIDENT` path already
trains 1600px / 50k Gaussians — validate the bare-metal path against it.

### Gap 1 — Real camera projection (x280) — ✅ CLOSED ON SILICON (2026-07-05)
**Done end-to-end on the Blackhole.** Real 3D Gaussians `[mean3, scale_log3, quat4, op, color3]` (14
params) train fully on-device: the x280 runs projection FORWARD + BACKWARD + Adam, the Tensix does
render/backward, projection is no longer host-side.
- **1.1** `cb_whiten.c` (proj_fwd + Cholesky whiten) — ψ+depth vs golden **3.9e-6** (`test_gap1_whiten.py`).
- **1.2** `opt_proj_step.c` (whiten-bwd → proj_bwd → Adam/14) — boot-publish + 1 Adam step vs golden
  **3.05e-5** (`test_gap1_optproj.py`).
- **1.3** `het/train_resident3d.py` — single-view **14.3→21.7 dB / 20 steps @ ~0.1 s**, multi-view
  (3 cams, grad-accum + cmd=2 project-only publish) **21.9→24.0 dB / 24 steps @ ~0.3 s**, all finite.
- **Shared math** = `kernels/x280/het/proj.h` (portable; self-contained expf, guarded sqrt), the ONE
  source of truth for both kernels + the host cross-check. Golden `scratchpad/gap1_proj_golden.py`
  (vs torch autograd 1.4e-14), scalar-C cross-check `scratchpad/gap1_check_c.py` (fp32 floor 3.95e-4).
- **Gotchas learned:** (a) `tt-smi -r 0` BETWEEN resident-kernel runs — a spinning kernel + reload
  degrades NoC state (bringup reports "Hang", results garbage). (b) The Tensix backward emits `inf` on
  degenerate/off-tile Gaussians ("recip finite col-pad") → **grad sanitization is mandatory** (finite-guard
  + clip in the Adam loop, standard 3DGS practice). (c) Boot publish must set its TELE 'OPT!' marker
  AFTER the publish completes, else the host reads a stale buffer.

<details><summary>original Gap-1 plan (host de-risk + seam) — for reference</summary>
Per Gaussian per view: `mean → clip → NDC → screen`; 2×2 screen covariance from the 3×3 world covariance
(`R·diag(exp scale)²·Rᵀ`) via the perspective Jacobian; invert → the whitening ψ the render consumes. The
x280 already computes ψ from `(Σ⁻¹, mean)` in fp32 (`cb_whiten.c`, 4.2e-7 vs host) — **extend it to the full
3D→2D projection + its analytic backward** (2D grads → 3D mean/scale/quat grads), which live next to the
optimizer on the x280. Turns the orthographic toy into real 3D scenes. **Medium.**

**The seam (why this is a clean prepend/append, not a rewrite):** params today are 2D
`[gx,gy,a,b,c,op,c0,c1,c2]`; Gap 1 makes them REAL 3D `[mean(3), scale_log(3), quat(4), op, color(3)]` (14)
and inserts, at the existing `(gx,gy,a,b,c)` seam:
- **forward** `(mean3, scale_log3, quat4, camera) → (gx,gy,a,b,c) + depth`, prepended in front of the
  Cholesky whitening `cb_whiten.c` already does. `cb_whiten.c` = **proj_fwd + existing whiten**.
- **backward** `dL/d(gx,gy,a,b,c) → dL/d(mean3,scale_log3,quat4)`, appended after the whiten-backward
  `opt_step.c` already does (its lines 52–64 produce exactly `g_gx,g_gy,g_a,g_b,g_c`). `opt_step.c` =
  **existing whiten-bwd + proj_bwd + Adam over the 14 params**.
The render, whitening, eval matmul, and whiten-backward are **UNCHANGED**. Only the front (projection),
the back (projection-backward), and the param vector + camera consts grow.

**DONE (2026-07-05, host, zero device ops) — the numerical core is proven:**
- `scratchpad/gap1_proj_golden.py` — numpy golden model. Forward matches torch to **5.6e-17**; the
  hand-derived **analytic backward matches torch autograd to 1.4e-14** over 200 random Gaussians.
  Convention = the ttnn ref `project` (`tt-splat/docs/pathclear/train3d.py`), which matches splat.py's
  `(a,b,c)`. Covariance `Σ3=R·diag(exp(2·scale_log))·Rᵀ` (w-first quat), `mc=Rv·mean+tv`,
  `J=[[fx/z,0,-fx·mc0/z²],[0,fy/z,-fy·mc1/z²]]`, `Σ2=J(RvΣ3Rvᵀ)Jᵀ+0.3I`, invert → `(a,b,c)`.
- `scratchpad/gap1_proj.c` — **PASTE-READY scalar C** `proj_fwd`/`proj_bwd` (pure float, no matrix libs;
  `proj_bwd` recomputes forward internals from resident params + camera, exactly like `opt_step.c`
  recomputes sa,m12,m22). Cross-checked vs the golden (`scratchpad/gap1_check_c.py`): worst **3.95e-4
  rel** over 300 cases — the fp32/fp64 rounding floor, all abs ≤1e-4. No symmetric-halving / transpose slips.

**Silicon port (the remaining Gap 1 work, ordered):**
1. **cb_whiten.c on-device de-risk (isolated, lowest risk):** replace its input ABI (was `[K, a,b,c,gx,gy]`)
   with `[K, camera(Rv9,tv3,fx,fy,cx,cy), then mean3,scale_log3,quat4 per G]`; body = `proj_fwd` →
   existing Cholesky whiten → ψ. Host harness writes 3D params + camera, rings it, reads back ψ (or a,b,c),
   compares to `gap1_proj_golden`. Doesn't touch the resident Tensix trainer or opt_step — recover via `tt-smi -r 0`.
2. **opt_step.c:** after the existing whiten-bwd (g_gx..g_c), call `proj_bwd` → (dmean3,dscale_log3,dquat4);
   Adam over the 14-param vector (new lr/clamp groups: mean, scale_log, quat, op, color). Depth for the
   sort = `mc[2]` (compute in projection, publish for the x280 depth-sort).
3. **train_resident.py:** 3D scene + camera generation (mirror `train3d.py` scene()/cameras()); feed 14
   params; stage camera consts; train over MULTIPLE views (the real 3D-vs-per-view-overfit test); target =
   host golden 3D render per view. Validate convergence + a held-out NOVEL view (train3d hits >35 train,
   >30 novel dB).
Apply the x280 architecture constraints (below) to the multi-view / per-camera orchestration.
</details>

### Gap 2 — Multi-tile binning (x280) — ✅ CLOSED ON SILICON (2026-07-05)
`bin_tiles.c` reads the projection publish `[gx,gy,a,b,c,depth]`, computes each Gaussian's 3σ screen bbox
(from Σ2=conic⁻¹ diagonal), expands to covered 16×16 tiles, and inserts into per-tile buckets kept
DEPTH-SORTED (near→far) — a bucket sort that also orders within the bucket and depth-culls the farthest
when full. **48×48/9-tile de-risk: 9/9 tiles match the golden exactly, 94/94 touches** (`test_gap2_bin.py`).
Golden `scratchpad/gap2_bin_golden.py`. The bucket cull is the Gap-4 primitive (see below).

### Gap 3 — Tile → grid + grad scatter-add — ✅ CLOSED ON SILICON (2026-07-05)
`het/train_multitile3d.py`: full W×H image of 16×16 tiles. Each step the x280 projects all N Gaussians
(publish) and Adam-updates them; the host bins (Gap-2 rule), and **each tile renders its own binned top-Ktile
subset** through the resident Tensix render+backward over the tile's screen region; **per-Gaussian gradients
SCATTER-ADD across every tile that touched the Gaussian**; the accumulated grad drives the x280 Adam.
**48×48/3×3 tiles, N=16: converges 17.5→21.0 dB / 15 steps @ ~0.8 s, all finite.** Tiles under-full are
opacity-0 padded; over-full are depth-culled (Gap 4). NOTE: binning is run host-side here for dispatch
orchestration (the on-device bin is proven separately in Gap 2); a resident scheduler + the 120-worker grid
(240/240 proven) replace the host loop at scale. The 1600²=10,000-tile / 120-worker (~83 tiles/worker) map
is the same pattern.

### Gap 4 — K > 16 Gaussians per tile — ✅ depth-cull PROVEN ON SILICON
Depth-cull to top-Ktile (near Gaussians dominate the front-to-back composite). The cull is the `bin_tiles`
full-bucket eviction: **CAP=8 de-risk culls each over-subscribed tile to exactly its 8 nearest, matching the
golden's top-8-by-depth exactly** (`test_gap2_bin.py 8`), and `train_multitile3d.tile_subset` culls to Ktile.
Also demonstrated **N=24 > the old K=16 window cap** (respaced adam_v→0x30006800; MAXK=24) — the multi-tile
path scales past 16 in-loop. `RT_DIM>1` multi-pass (composite in chunks) remains an option for very dense tiles.

### Gap 5 — Millions of Gaussians (GDDR-resident params) — ✅ MILLIONS PROVEN ON SILICON (2026-07-06)
**The feared "x280↔big-GDDR handoff" was a non-issue: it is direct-mapped LOCAL GDDR.** The authoritative
ISA map (`.isa_cache/BlackholeA0__L2CPUTile__MemoryMap.md:26`) shows **`0x30000000..0x130000000` = 4 GiB of
the L2CPU tile's OWN local GDDR6, uncached, x280-physical == NoC 1:1** — not the 256 MB bhtop annotated
(`regmap.py:149` is conservative/descriptive). So the x280 addresses its whole 4 GiB bank with plain
load/store. My earlier "12 KB window / N≈24" cap was self-imposed (packing under the code at 0x30008000).
- **5a — full bank addressable:** `dram_sentinel.c` writes 300 unique values across the full 4 GiB
  (0x30010000..0x130000000), reads all back **0 bad**, host cross-checks a spread — distinct DRAM, no
  aliasing, host↔x280 coherent (`test_gap5_sentinel.py 0x130000000`). **4 GiB / ~168 B ≈ 25 M Gaussians.**
- **5b — millions optimized:** `opt_proj_big.c` holds **N=1,000,000** Gaussians (160 MB: params+Adam m/v) in
  the bank and runs a full projection→whiten-bwd→proj-bwd→Adam step over **all 1 M** — every sampled
  Gaussian bit-matches the golden init AND Adam step (worst rel **6.4e-6**) (`test_gap5_big.py 1000000`).
- **5c — throughput lever:** the step is **135 K cyc/Gaussian (77 s/1 M), UNCACHED-GDDR-bound** (~80 NoC
  round-trips/Gaussian). The ISA map exposes a **CACHED alias of the same GDDR at `0x4000_3000_0000`
  (MemoryMap.md:34)**; `bench_cache.c` measures **13.8× speedup** (276→20 cyc/word) for a cache-resident
  block, checksums matching. ⇒ the fast-training path = resident param/Adam in the cached alias, processed
  in **cache-resident blocks** (blocked streaming), coherency managed for host reads.
- **Beyond 25 M:** shard across the **4 L2CPU tiles' local banks (16 GiB, ~95 M, no TLB)** — the NUMA-natural
  design (each tile optimizes its shard in its local bank), or the **large TLB windows** (`0x0804_3000_0000`,
  32×128 GiB, config @ `0x2000_0E00`) to the DRAM-only tiles for the full 32 GB (~190 M). TLB needs the
  `TLBWindows.md` register format (not in the cached ISA docs) — its own small de-risk. Not needed for real
  scenes (SOTA 3DGS is 1–10 M).
Artifacts: `kernels/x280/het/{dram_sentinel.c,opt_proj_big.c,bench_cache.c}`,
`scratchpad/{test_gap5_sentinel.py,test_gap5_big.py,test_gap5_cache.py}`.
**Remaining Gap-5 BUILD (not de-risk):** wire the cached-alias blocked optimizer into the multi-tile trainer
so real training steps at 100 K–1 M are fast; then Gap 6.

### Gap 6 — Densification / pruning — REMAINING
Clone/split high-grad Gaussians, prune transparent/degenerate; realloc GDDR params + Adam m/v preserving
survivors' momentum + step counter (mirror `DeviceAdam.prune`). Periodic, host-orchestrated. **Needs the
dynamic-N capacity from Gap 5** (fixed-N GDDR layout here bakes N into addresses/loops). **Medium.**

### Then: integrate + validate at 1600px/50k against the ttnn reference.

---

## x280 architecture constraints (Dan, for Gaps 1–3 orchestration)
- **NoC latency** — locality-aware worker↔x280 assignment; x280 is at noc0 (8,3), workers far. Minimize hops.
- **DRAM NUMA** — place each Gaussian's params/Adam state in the GDDR bank nearest its consuming worker.
- **Real-time telemetry** — reserve x280 hart(s) as the on-chip tracer (the CFGREG/debug-bus rig we built),
  separate from the ~14 compute harts (x280 has **16 harts**).
- **Flexible, locality-driven assignment** ("random networks") — dynamic, load-balanced tile→worker/hart
  mapping that follows the data, not a fixed grid.

---

## 1600px instrumented flow — BUILT + PROFILED (2026-07-06) → perf-tuning starts here
`het/train_fused_instrumented.py` composes the whole pipeline end-to-end and TIMES EVERY STAGE, parameterized
to 1600×1600: x280 projection+backward+Adam over GDDR-resident params (`opt_proj_gddr.c`, real grads, big
window, per-phase `rdcycle` telem), host Gap-2 binning, per-tile Tensix `resident_train_perf` render/backward
across 1..120 workers (per-ring `T_END` cycle telem), per-Gaussian grad scatter-add. Boots the render kernel
on N `worker_coords(ctx)` + the x280 optimizer simultaneously (different tiles, no conflict).

**Measured breakdown (256×256, 256 tiles, 2048 rings, 22.5 s/step) — scale-invariant, cost-model-confirmed:**
| stage | ms | % | nature |
|---|---|---|---|
| render_readback | 10276 | 46.4 | host↔dev NoC (5 tiles/ring readback) |
| render_stage_pergroup | 5090 | 23.0 | host↔dev NoC (phi/gt write) |
| render_ring_wait | 4387 | 19.8 | host poll (device work inside = **11 872 cyc/ring ≈ 8 µs**) |
| render_stage_consts | 1983 | 8.9 | host↔dev NoC (9 const tiles/tile) |
| scatter_add / bin | 256 | 1.2 | host CPU (numpy) |
| x280 project + Adam | 116 | 0.5 | **device** (31 M + 168 M cyc) |

**HEADLINE: device compute = 0.59 % of the step; host↔device exalens relay = 99.4 %.** The fused silicon is
nearly free; the wall-clock is the serial exalens NoC relay, which the 120-worker grid does NOT parallelize.
**MEASURED at true 1600×1600** (10,000 tiles, 80,000 rings, N=1500, 12 workers): **716.8 s/step**, breakdown
readback 56.9 % · stage 38.7 % · ring-poll 3.0 % · device 0.11 %. Render cycles **949,682,985 = the cost
model's 9.497×10⁸ to 4 sig figs**; the silicon did the whole step in **~800 ms** (render 633 ms + x280 167 ms).
So today host-bound ≈ 12 min/step vs **~seconds device-resident** (Tensix render ~5 ms/120 workers, N-independent;
x280 optimizer 0.56 s/100K cached). Dashboard artifact + JSON: `scratchpad/fused_profiler.html`,
`scratchpad/telemetry_{256,1600}.json`.

**Tuning roadmap (ordered by measured impact):** (1) on-device grad accumulation — x280 NoC-reads worker-L1
grads + scatter-adds (`cb_scatter.c`), kills render_readback+scatter ⇒ −47 %; (2) on-device operand production
— x280 tilizes + NoC-writes consts into worker L1 (`cb_operands.c`), kills render_stage_* ⇒ −32 %; (3) grid
dispatch on a DONE-counter (no per-ring poll) ⇒ −20 %; (4) cached-alias x280 optimizer (13.8×). Result:
device-bound, seconds/step at 1600px on the grid. Files: `train_fused_instrumented.py`, `opt_proj_gddr.c`,
`fast_target()` (bbox-local numpy target so 1600px is host-feasible). Gotcha: `SP._golden_render` is a Python
triple-loop — impractical ≥512px, use `fast_target`.

---

## Files / harnesses / gotchas
- Kernel: `src/bhtop/kernels/tensix/llk/resident_train_perf/resident_train_perf.cpp` (eltwise = matmul-load
  + SFPU-binary; forward SFPU + MREC-recip already matmul+SFPU).
- Trainer: `src/bhtop/het/train_resident.py` (resident, no reboot).
- SFPU-binary proof: `resident_mm_elw_perf` (holds the 50/50 proof; FPU version backed up
  `scratchpad/resident_mm_elw_perf.cpp.orig`). Harness `scratchpad/test_gap0_sfpubin.py`.
- Multi-ring trainer test: `scratchpad/test_gap0_ab.py` (rings N on one boot, checks PSNR + grads).
- On-chip tracer (reusable, = x280 read paths): `scratchpad/test_gap0_{cfgdiff,dbusdiff,fullcfg,globals2,wedge}.py`
  — full 16384-word CFG dump via CFGREG port (`RD_CNTL 0xFFB12058 / RDDATA 0xFFB12078`) in 0.2 s.
- Recover any wedge: `tt-smi -r 0`. NoC0-hang hazard classes: ARC/Security/PCIe/L2CPU regs (worker Tensix
  config/L1 reads are safe).
- Gotchas: matmul-load eltwise needs the identity operand staged (`H_Iden`); `BinaryOp::SUB` = in0−in1;
  grads shifted slightly with SFPU vs FPU eltwise (dLdpsi 1.6%→2.5%, bf16 fidelity) but pass; ring cost
  11871 cyc (~11% over FPU, for full residency).
