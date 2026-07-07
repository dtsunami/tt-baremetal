# The Resident Render Grid — bare-metal Tensix, doorbell-driven, no tt-metal

**Goal.** Render Gaussian-splat tiles on the full 120-worker Blackhole Tensix grid *in parallel*, each
worker running one **resident** fused kernel driven by a host **doorbell** — so the host stages a tile's
operands and rings once, instead of dispatching ~190 LLK ops/tile (build → load → boot → poll each). The
per-op host round-trip over exalens is what makes the current `splat.render_ondevice` serial and slow
(raster 588 ms + backward 863 ms/tile were host-dispatch-bound; the on-device compute is ~µs). Residency
collapses that round-trip; the grid gives 120× parallelism.

This is the last big perf lever for the bare-metal splat trainer (see `baremetal_plan.md`,
`src/bhtop/het/README.md`). As of 2026-07-05 **every mechanism it needs is proven on silicon** — what
remains is assembling the specific 11-stage render into one resident kernel.

## Mechanisms — all PROVEN on silicon (worker noc0 (1,2), fp32/bf16)

| # | Mechanism | Kernel / test | Result |
|---|-----------|---------------|--------|
| R1 | **3-thread Tensix doorbell residency** — all of unpack/math/pack run INIT once then spin in a `for(;;)` doorbell loop; host rings, they re-run, publish DONE, no reload | `kernels/tensix/llk/resident_mm_perf` · `scratchpad/test_resident_mm.py` | 3 rings, distinct operands, **bit-exact, no reload** (heartbeats 1→2→3) |
| R3 | **120-worker resident grid** — build once, boot all 120, ring in parallel, collect | `tensix/resident.py::ResidentMatmul` · `scratchpad/test_resident_grid.py` | boot 120 in 1.1 s; **240/240 (worker,round) bit-exact**, ring+collect ≈0.75 s/round |
| — | **MVMUL→SFPU reconfig in one math thread** (mid-kernel FPU→vector) | `kernels/tensix/llk/fused_mm_sq_perf` | `square(A@B)`, 0 error (prior session) |
| R2 | **Inter-stage on-device dataflow** — pack stage-N to L1 scratch, unpack it as stage-(N+1) input, pack→unpack synced on-device | `kernels/tensix/llk/resident_mm2_perf` · `scratchpad/test_resident_mm2.py` | `C2=(A@B)@D` chained, **3 rings bit-exact** |
| ✅ | **FUSED RESIDENT RENDER** — the whole 11-stage `render_ondevice` as ONE resident kernel (6 fused super-stages), full tile + grid, one ring/tile, cycle telemetry | `kernels/tensix/llk/resident_render_perf` · `het/render_resident.py` · `scratchpad/test_resident_render{,_full}.py` | full 16×16 tile **51 dB**; 16-worker grid all 51 dB; ~31.7k dev cyc/tile |
| ✅ | **Matmul ↔ eltwise mode-switch** (the backward's new op) — MVMUL then FPU ELWMUL in one resident kernel | `kernels/tensix/llk/resident_mm_elw_perf` | `E=(A@B)⊙D`, **3 rings bit-exact** |
| ✅ | **FUSED RESIDENT BACKWARD** — `backward_ondevice`'s whole 17-stage chain as ONE resident kernel (matmul / matmul+recip / eltwise-mul / eltwise-sub), stage-table driven | `kernels/tensix/llk/resident_backward_perf` · `scratchpad/test_resident_backward.py` | leaf grads **dLdcolor 0.3% · dLdop 0.9% · dLdpsi 1.1%** vs exact golden |
| ✅ | **FULL FUSED RESIDENT TRAINING STEP** — forward render + on-device `dLdC=C−gt` + backward, ALL in ONE ring (27 stages, one stage table) | `kernels/tensix/llk/resident_train_perf` · `scratchpad/test_resident_train.py` | RGB **52.6 dB** + **dLdpsi 1.6% · dLdop 1.1%**; `dLdcolor`→x280 |

Together these are *every* primitive the fused render uses: a resident doorbell loop (R1), replicated
across the grid (R3), whose body alternates matmul and SFPU in the math thread (fused_mm_sq) with
intermediates staged through L1 between stages (R2).

### The three non-obvious silicon lessons (baked into the kernels)

1. **RuntimeParams ABI is positional and alphabetical.** `llk.gen_build_h` builds the struct from the
   `params.X` fields the source references, ordered `TILE_CNT` first then alphabetical. Drop a field the
   host still writes (we dropped `LOOP_FACTOR`) and every later field shifts — here `num_faces_A` read 0
   and `_llk_unpack_hw_configure_`'s `LLK_ASSERT` **spun forever**. The resident kernels reference
   `LOOP_FACTOR` explicitly to keep the stock 10-field ABK `[TILE_CNT,CT,KT,LOOP,RT,TSA,TSB,TRANSPOSE,NFA,NFB]`.
2. **Doorbell reads need a fence.** The baby-RISC L0 DCACHE makes a bare `while(db[0]==last)` read a stale
   line — the boot thread T0 caches `DB=0` and never sees the host bump it (whole pipeline then stalls on
   unproduced src). `invalidate_data_cache()` is a `fence` (Blackhole invalidates DCACHE as a side effect);
   fencing before each poll and after each publish fixes it. (`load_blocking` alone is *not* enough — it
   adds a data dependency, not a fence.)
3. **Publish DONE only after the pack TDMA lands.** `_llk_pack_dest_section_done_` stalls the Tensix
   backend but not the RISC front-end, so a bare `done=1` races ahead of the pack → host reads a
   poisoned/partial tile and races into the next ring → wedge. `tensix_sync()` (pc_buf) *deadlocks*
   mid-loop. Instead the pack thread spins until the last output word changes off the host's poison — a
   RISC-visible proof the tile is in L1 — then publishes. Same trick gates the R2 pack→unpack handoff.

## The fused render — BUILT + verified on silicon (`resident_render_perf`)

`splat.render_ondevice`'s 11 stages **collapse to 6 fused super-stages** by doing each SFPU op in DEST
right after its matmul (the fused_mm_sq trick) and folding the `[la|lpa]@Mcomb` concatenation into a
KT=2 accumulating matmul (`la@Stri + lpa@I` = two matmuls into the same DEST, no clear between):

```
F1: Vsq = square(phi @ psi)            F2: ar  = exp(Vsq @ Ppair)
F3: lpa = log(ar @ Dop)                F4: la  = log1p(ar @ Dnop)
F5: w   = exp(la@Stri + lpa@I)  (KT=2) F6: C   = w @ color
```

Each super-stage is the R2 pattern: `unpack → math(matmul[,matmul] + SFPU) → pack to an L1 scratch →
publish a stage flag`; the next stage's unpack waits on the flags of the scratch tiles it reads. All
four SFPU ops (square/exp/log/log1p) are compiled in and dispatched by stage. All tiles are 32×32,
host zero-padded (`[P pixels] × [cols]`), so each is a full 32×32 matmul. Verified per-stage against a
host golden (rel-err 0.5–1.1%, bf16), then full 16×16 tile = **51 dB** vs the exact golden, then the
same on an 8-worker grid (all 51 dB) — one boot per worker, one ring per pixel-group, **no reload**.

**Drive it:** `het.render_resident.render(coord, ctx=...)` (build operands → boot → stage consts → ring
per group → assemble RGB). `build_operands` / `boot` / `render_tile` are separable so the grid boots all
workers, stages consts once each, then renders in parallel.

### Telemetry (for tuning)
The pack thread stamps `read_wall_clock()` at the tile start, after each group-0 stage, and at the tile
end, into an L1 telemetry region (`0x16080`); `het.render_resident.render(..., telem=True)` surfaces
per-stage device cycles + whole-tile cycles + host ms. Measured (16×16, bf16): whole tile ≈ **31.7k dev
cyc** (8 groups, ~4.1k/group); per stage F1 364 · F2 781 · F3 819 · **F4 1092** · F5 861 · F6 220 cyc —
the SFPU stages (esp. log1p) dominate, the matmuls are cheap. This is the knob for precision/throughput
trades.

### Done since the render landed
- **Item 1 (precision): resolved — bf16 dest wins.** Telemetry-measured: fp32 dest (`build(fp32_acc=True)`,
  the `(…,0,5,0,5)` bf16-pack FormatConfig) gives **49.2 dB @ 6.3k cyc/group** vs bf16's **51.0 dB @ 4.1k
  cyc**. Intermediates pack to bf16 regardless, so fp32 dest only adds SFPU cost with no precision gain →
  bf16 is the default. (fp32 path kept behind the flag.)
- **Item 3 (one ring per tile): DONE.** The kernel loops all NG pixel-groups internally (NG at `0x160A0`,
  phi[g]/OUT[g] strided by `STRIDE`, monotone per-(ring,group) flags `ring*32+g`); phi is staged once
  (it's identical for every tile of a size) and only the gaussian consts restage per tile. Host does 1
  ring/tile instead of 8. 51 dB unchanged; 16-worker grid renders 16 tiles in 128 ms.
- **Cleaner landing barrier:** `tensix_sync()` after `_llk_pack_dest_section_done_` (the earlier
  "deadlock" was the LOOP_FACTOR hang) replaces the poison-poll — no host poison, works with reused scratch.

### Item 4 (backward) — DONE as a standalone resident kernel
Both halves of a training step now run resident and correct: the **forward** (`resident_render_perf`, 51 dB)
and the **backward** (`resident_backward_perf`, 17 stages, leaf grads <1.2% vs exact). The backward is a
compile-time stage table (`ST_A/ST_B/ST_O/ST_OP/ST_W0/ST_W1`) the three threads share; ops are matmul (M),
matmul+SFPU-reciprocal (R, via matmul-by-identity then SFPU), FPU eltwise-mul (X, HiFi4), FPU eltwise-sub
(S, LoFi); reciprocal inputs get a finite col-padding so `inf→nan` can't propagate through the contractions.
It host-stages the forward intermediates today (standalone verify); the trainer will feed them from L1.

### The single-kernel merge — DONE (`resident_train_perf`)
One worker does a whole training-step tile in ONE ring, nothing crossing back to the host between forward
and backward: forward 6 stages → seam (`dLdC = C − gt` on-device against a host-staged gt; recompute
`α = ar⊙opB` and `V = phi@psi` from the forward's live tiles) → backward 16 stages → `dLdpsi` + `dLdop`.
27 stages, one compile-time stage table shared by the three threads. Verified on silicon: RGB 52.6 dB,
`dLdpsi` 1.6%, `dLdop` 1.1% vs an exact-float golden of the whole step.

Two hard-won lessons baked in: (1) reciprocal inputs (`α`, `1−α`) need a **finite col-padding** (0.5)
because `recip(0)=Inf→NaN` propagates through the matmul contractions and poisons the reductions;
(2) **`transpose_dest` corrupts the eltwise pipeline** — after it, the next FPU eltwise deadlocks (isolated
by bisection: the matmul-copy version completes, the transpose version stalls at the first following
eltwise, regardless of srcB balancing). So **`dLdcolor = wᵀ@dLdC` is delegated to the x280** param-server,
which already holds `w` and `dLdC` — a 16×3×32 scalar reduction, trivial there. Clean split: Tensix does
the dense chained grads (geometry + opacity), x280 does the color grad + the optimizer.

### Trainer — plumbed + converging (`het/train_resident.py`)
The fused kernel drives a real on-device training loop: per 32-px group `resident_train_perf` (Tensix
fwd + dLdC + bwd) → grads accumulated over the tile → x280 `opt_step` (whiten-backward + Adam, resident;
`dLdcolor = wᵀ@dLdC` computed host-side / on the x280). Smoke test (fit one 16×16 tile, K=12) **converges
13.6 → 19.7 dB over 4 steps**, loss 0.044 → 0.011 monotonic.

**One caveat / the one remaining kernel bug:** `resident_train_perf` completes ring 1 but **stalls on ring
> 1** — the 27-stage dest-sync (SyncHalf ping-pong) doesn't reset cleanly across rings (a `reset_dest_offset_id`
+ pack-sync re-init at the ring boundary didn't fix it; the render/R1/R2 kernels don't hit this because they
have fewer or even-count stages). The trainer **re-boots the kernel per ring** (each is a fresh ring 1;
operands at 0x21000+ persist across boot — ~0.3 s, faster than a 6 s stall) so the loop converges today.
The real fix — find the unbalanced dest-sync state so the 27-stage kernel is truly multi-ring resident —
is the last item before the trainer is fully resident.

### Remaining — scale + integration
- **Fix multi-ring residency** of `resident_train_perf` (above), then drive it truly resident (one ring/tile).
- **Item 2 (operand streaming)** — x280 computes/tilizes the whitened consts + NoC-writes them into each
  worker's L1 (extend `cb_operands.c`) for zero host Gaussian-data relay.
- **Grid + full-res (1600px)** — this is the *scale campaign*, not a wiring step. `resident_train_perf` is a
  16×16-tile / K≤16 / orthographic-identity-projection primitive (Stage 0). 1600px needs: real camera
  projection (3D→2D+cov, on the x280 — extend `cb_whiten.c`), multi-tile **binning** (which Gaussians touch
  which tile — the x280 counting-sort), tile→worker assignment across the proven 120-worker grid, and
  densification to many Gaussians. The resident grid is the compute substrate; the projection/binning/
  densification plumbing is the remaining scale work. (The **ttnn `TT_DEVICE_RESIDENT` path already trains
  1600px / 50k Gaussians** — the bare-metal path targets efficiency, and is at Stage 0.)

## Files

- `src/bhtop/kernels/tensix/llk/resident_render_perf/` — **the fused resident render** (6 super-stages).
- `src/bhtop/kernels/tensix/llk/resident_mm_perf/` — R1 keystone (resident doorbell matmul).
- `src/bhtop/kernels/tensix/llk/resident_mm2_perf/` — R2 chained matmul (inter-stage L1 dataflow).
- `src/bhtop/kernels/tensix/llk/resident_train_perf/` — **the full fused training step** (fwd+dLdC+bwd, 27-stage table) · `scratchpad/test_resident_train.py`.
- `src/bhtop/kernels/tensix/llk/resident_backward_perf/` — **the fused resident backward** (17-stage table).
- `src/bhtop/kernels/tensix/llk/resident_mm_elw_perf/` — matmul↔eltwise-binary mode-switch (backward's new op).
- `src/bhtop/kernels/tensix/llk/resident_probe/` — minimal 3-thread doorbell-residency isolation probe.
- `src/bhtop/het/render_resident.py` — render driver: `build_operands` / `boot` / `render_tile` / `render`.
- `src/bhtop/het/train_resident.py` — **the fused-kernel trainer** (converges 13.6→19.7 dB; per-ring reboot).
- `src/bhtop/tensix/resident.py` — host driver: `boot_resident()` (generic boot-without-KERNEL_COMPLETE),
  `ResidentMatmul` (boot / ring_async / collect), the grid-parallel path.
- `scratchpad/test_resident_render{,_full}.py` — render proofs (per-stage vs golden; full tile + grid).
- `scratchpad/test_resident_mm.py` · `test_resident_grid.py` · `test_resident_mm2.py` — the silicon proofs.
- Doorbell L1 map (free gap 0x15000–0x20000): `DB 0x16000 · DONE 0x16010 · HB 0x16020 ·
  DBG_U/M/P 0x16030/40/50 · stage flags 0x16060+`. Operands reuse the perf region
  (`A 0x21000 · B 0x31000 · scratch 0x41000 · OUTPUT 0x51000 · D 0x61000`).
