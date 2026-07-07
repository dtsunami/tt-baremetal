# Bootloader / Tensix area → x280 parity plan

Goal (Dan, 2026-06-21): bring the bare-metal Tensix (bootloader overlay + LLK) cockpit lane up to
the level of the x280 (L2CPU Hart Lab / HartObserve) lane. Three threads + four UI-parity items.
Host-doable work is marked [HOST]; work needing a live chip is [DEVICE].

## STATUS 2026-06-22 — all three threads landed (host-verified), device pass pending
Implemented + adversarially reviewed (2 workflows: understand + verify). Frontend builds clean; LLK
generator at 14/15; the verify pass's MUST-FIX findings are all fixed. No chip this session, so the
[DEVICE] items below are written-but-unvalidated.

## Thread 1 — make the bare-metal kernels all work — DONE (10→14/15)
- **1a-ClassA [HOST] DONE** dest_sync / DST_SYNC_MODE (=DstSync::SyncHalf) added to gen_build_h →
  matmul_perf (11/15).
- **1a-ClassB [HOST] DONE** — and the "Operand-typed FormatConfig" premise was a MISDIAGNOSIS (caught
  by the understand workflow). Real fix mirrors tt-llk's SPEED_OF_LIGHT/compile_time_formats path:
  `formats` is now a `static constexpr FormatConfig` (so `formats.math` works as a NON-TYPE TEMPLATE
  ARG — the actual blocker) with the const-member/constexpr-ctor struct; stimuli buffers are global
  `Operand` members; FormatConfig members stay std::uint32_t. → eltwise_binary_sfpu_perf,
  math_matmul_perf, sfpu_reduce_row_max_perf all build → **14/15**.
- **Loader ABI [HOST] DONE** — formats are compile-time now (not a runtime member). gen_build_h ALWAYS
  emits TILE_CNT as the first runtime field, and llk_run._runtime_params writes tile_cnt at offset 0.
  (Fix from the verify pass: without TILE_CNT-first, the 3 no-TILE_CNT kernels had tile_cnt corrupt
  CT_DIM/LOOP_FACTOR.)
- **build.example.h [HOST] DONE** — regenerated all 15 (import_kernels) so the CLI `build.sh <name>`
  default path matches the live generator (was stale → failed on dest_sync).
- **STILL FAILING (1/15): unpack_a_bcast_eltwise_perf** — NOT a generator bug. It calls
  `_llk_unpack_bcastA_B_init_` / `_llk_unpack_bcastA_B_`, which exist only in the WORMHOLE llk_lib,
  never ported to Blackhole (+ a `_llk_math_eltwise_binary_init_` signature drift). Needs an upstream
  BH port or a kernel rewrite — out of scope for the build.h generator.
- **1b [DEVICE]** Run each buildable kernel on a TRISC (llk_run) → KERNEL_COMPLETE. Also: the matmul
  family takes runtime dims (CT_DIM/KT_DIM/RT_DIM/LOOP_FACTOR) the loader does NOT feed (reads 0) —
  wiring those is a separate device-config step.

## Thread 2 — merge tt-llk learnings into the overlays — DONE (verdict, not a patch)
- **VERDICT:** matrix/sfpu CANNOT be fixed as BRISC overlays. Root cause = instruction-FIFO topology:
  the overlay writes MVMUL/SFPMAD to 0xFFE40000 (the T0/UNPACK per-thread FIFO; stride 0x10000, math
  = 0xFFE50000), but those ops belong on T1/MATH and stall on cross-thread SrcA/B-dvalid + DEST-sync
  with no producer/consumer → wedge. CFG/SFPU bring-up issued from BRISC binds to the wrong thread.
- **2a [HOST] DONE** — encoded the root cause in matrix.c + sfpu.c headers, added `superseded_by` +
  rewritten desc in both kernel.json, and a corollary note in bootloader.py. The "merged learning"
  is: compute kernels belong in the TRISC LLK lane — llk/matmul_perf + math_matmul_perf (FPU),
  eltwise_unary_sfpu_perf (SFPU), all of which build.
- **2b [DEVICE]** If a real FPU/SFPU power-probe is wanted, add a MATH_ISOLATE LLK-lane kernel
  (matmul_perf already does this); validate on silicon (recover wedge: tt-smi -r 0).

## Thread 3 — UI parity with HartObserve — DONE (host-verified; needs eyeball + a chip for live data)
- **3a Telemetry in the cartoon / 3e core clarity [HOST] DONE** — BootloaderPanel reworked: the live
  CorePicker cartoon (now with single-select `mode`, backward-compatible) is the selector; clicking a
  green (bootloader) core streams its /ws/bootloader telemetry inline. Kind-coloring + hash hover make
  residency unmistakable.
- **3b Per-slot labels [HOST] DONE** — labels are in-frame (telemetry.fields) + the raw-slot grid /
  plot axes are labelled from the loaded overlay's kernel.json telemetry schema.
- **3c Live plot [HOST] DONE** — ported HartObserve's history-ring plot (client-side, single core);
  slot picker from the overlay's telemetry labels, rate toggle.
- **3d Multi-tab [HOST] DONE** — overlay observe is now Deploy / Telemetry / Plot. The
  device-validated deploy flow (pickCore→openWs→compileBuffer→stage→setParams→exec) is verbatim.
- Verify-pass fix: CorePicker single mode is dispatch-only (parent owns `picked`), so the highlight
  can't desync from the focused core on a rejected non-bootloader click.

## Device pass (all that remains — needs a chip)
Run the 14 buildable LLK kernels on TRISCs; eyeball the reworked BootloaderPanel against live
telemetry; (optional) feed matmul runtime dims; (optional) build a TRISC FPU/SFPU power-probe.
Web server not running; bootloader was torn down by tt-smi resets — relaunch when a chip is available.
