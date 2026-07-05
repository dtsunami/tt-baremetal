# Bare-metal heterogeneous compute on Blackhole — plan & learnings

**Thesis:** the x280 (L2CPU RISC-V+RVV) and the Tensix grid can cooperate as co-equal dataflow peers,
**bare-metal over tt-exalens with zero tt-metal**, sharing GDDR — and that heterogeneous substrate is
the foundation for a fully-on-device Gaussian-splatting trainer (x280 owns the irregular tier:
gather/sort/scatter; Tensix owns the dense compute: matmul-eval, raster, SFPU). Two payoffs: (1) an
interview POC showing what TT's own stack doesn't do; (2) the real efficient splatting architecture.

**Why bare-metal, not tt-metal:** on *this* board tt-metal can't initialize — the bootloader example
hangs forever in device init (`Slow dispatch mode: Using full logical grid (12,10)`, clean device, no
x280; firmware bundle 19.11.0 > tested 19.5.0). Confirmed on silicon 2026-07-04. So we route around it
entirely. The bare-metal path is also *more robust* — nothing left to hang, JIT demoted to a plain
cross-compiler.

---

## ⏯ RESUME — START HERE (handoff, 2026-07-05)

**Where we are:** a Gaussian-splatting **trainer** on the heterogeneous machine, bare-metal, no
ttnn/tt-metal. On silicon:
- **Forward: fully on-device + streaming.** x280 sorts + gathers + tilizes ALL dense operands (ψ, opacity,
  color) → shared GDDR; Tensix renders (6 MVMUL + 5 SFPU); a DRAM circular buffer with backpressure feeds
  tiles x280→Tensix. Zero host Gaussian-data relay. ~52.9 dB vs golden.
- **Training works.** Full **geometry backward grad-checked** (every param) and trains pos+shape+opacity+
  color 17→37 dB; the device pipeline renders the trained scene (37.0 dB). x280 **projection/whitening**
  (fp32, `fsqrt.s`) and **scatter-add** both proven. Per-op build cache → 571 ms/step.
- **On-device backward de-risked.** All primitives verified bare-metal (matmul, eltwise-binary multiply,
  SFPU reciprocal; transpose side-stepped by host-transposing the small static operands). First gradient
  `dL/dcolor` on silicon (1.86e-3).

**Next session, in priority order:**
1. **Assemble the full on-device backward chain** (de-risked; the ONE new bit is caching the forward
   intermediates w/α/ar/v1/v2 in L1/GDDR so the backward reads them) → a **fully-on-device training loop**.
   See the "ON-DEVICE BACKWARD" section for the exact stage list.
2. **Wire x280 whitening + scatter-add into one live loop** (both proven, not yet in the loop).
3. **Perf lever:** cut the ~88 dispatches/render — multi-tile matmul (`RT_DIM>1`) + a fused SFPU kernel;
   then CB backpressure over many tiles; multiple x280 tiles.

**Env / how to run:** `~/bhtop/.venv/bin/python`, `sys.path.insert(0,'src')`. Tensix worker noc0 `(1,2)`;
x280 = `bhtop.l2cpu.L2cpu` (bringup one-shot after `tt-smi -r 0`). Shared exalens ctx = `init_ttexalens()`
passed to both `L2cpu(ctx=)` and `TensixLauncher.at(x,y,ctx=)`. Recovery: `tt-smi -r 0`.
**Gotchas:** keep multi-word host→x280 buffers in the OPEN uncached window `0x30002000–0x30007000` (~20 KB,
x280↔Tensix coherent), clear of the tele-window 0x100 boundaries; NoC DRAM reads want 64B-aligned length;
x280 scalar float needs `csrs mstatus,0x6000` (+ inline `fsqrt.s`, not `__builtin_sqrtf`). Renders/kernels
in `~/bhtop/scratchpad/` (survives reboots; `/tmp` does not). POC page: the published Artifact.
Session log with exact numbers: `~/.claude/.../memory/tt-het-noc-poc.md`.

---

## Status dashboard

| capability | status | how |
|---|---|---|
| device recover / baseline | ✅ proven | `tt-smi -r 0` (clean recovery, twice) |
| x280 bringup + compute + GDDR write | ✅ proven | `bhtop.l2cpu.L2cpu`: bringup(one-shot) → compile → load → telemetry |
| exalens owns a cold Tensix worker | ✅ proven | attach cold chip, 120 workers, L1 R/W bit-exact, `BabyRiscDebug` reset control |
| BRISC cold-boot (data movement) | ✅ proven | write kernel → L1 `0x0` → deassert reset → runs |
| bare-metal NOC0 read primitive | ✅ proven | extracted from tt-metal (~11 reg pokes); worker→worker bit-exact |
| **M1: x280 → Tensix handoff** | ✅ **proven** | x280 writes GDDR `(8,3):0x30002000`, Tensix BRISC NoC-reads it bit-exact |
| **TRISC compute-thread cold-boot** | ✅ **proven** | T0/T1/T2 run our code from L1 `0x6000/0xA000/0xE000` via `set_code_start_address` |
| BareMetal harness (all RISCs) | ✅ proven | `bhtop.tensix.baremetal.BareMetal`, in tree |
| MVMUL kernel builds + boots bare-metal | ✅ proven | `matmul_perf` L1_TO_L1, 3 ELFs, `llk_run.py` boots over exalens |
| **MVMUL bit-exact `A@B` (the RUN)** | ✅ **proven** | full unpack→math→pack on silicon, 0/1024 mismatch, reproducible; `tensix.matmul` |
| bf16 FPU precision wall (measured) | ✅ mapped | bit-exact to input 63 (out ~38k); 7-bit inputs (95,127) drift few ULP |
| **native Int8→Int32 matmul** | ✅ **proven** | stock `matmul_perf`, signed ±127, 0/1024 mismatch, 1 matmul (HiFi4 + sign-magnitude codec) |
| **exact integer matmul** (limb decomp) | ✅ proven | `run_matmul_int` base-64: inputs 0..4092, out 191M, 0/1024 (4 bf16 sub-matmuls; for values > int8) |
| SFPU exp/blend · GDDR stream · multi-core grid · DRAM CB (M2) · x280 gather (M3) | ▷ mapped | reachable — engineering, not unknowns |

---

## The substrate (proven on silicon, 2026-07-04)

Everything below runs **over tt-exalens (+ pyluwen/ARC for the x280), no tt-metal**, on a freshly
`tt-smi -r 0`'d chip. Env: `/home/starboy/bhtop/.venv/bin/python`, `sys.path += bhtop/src`.

1. **x280 side** (`bhtop.l2cpu.L2cpu`): `dev.bringup(0)` releases tile-0 harts (ONE-SHOT until
   `tt-smi -r 0`; `L2CPU_RESET=0x1f`). `tc.compile_source(src, base=CODE_ADDR=0x30008000, march)` →
   `dev.load(0,0,words)` → `dev.telemetry(0,16,0)`. The x280 writes its tile-local GDDR window
   `(8,3):0x30002000`; host reads it back over the NoC. RVV + CRT integer matmul proven (37.7×, bit-exact).

2. **Tensix substrate**: `from ttexalens import init_ttexalens; ctx = init_ttexalens()` attaches a cold
   chip. 120 workers via `bhtop.tensix.loader.worker_coords`. `TensixLauncher(coord, ctx)` does L1
   read/write. `ctx.devices[0].get_block(coord).get_risc_debug(risc)` → `BabyRiscDebug`:
   `set_reset_signal(bool)`, `set_code_start_address(addr)`, `is_in_reset()`.

3. **Cold-boot a baby-RISC**: assert reset → write kernel image to its reset PC in L1 → (for compute
   RISCs) `set_code_start_address(pc)` → deassert reset → it fetches from `pc` and runs. Read results
   from L1 over exalens. **BRISC** boots from hardwired L1 `0x0` (no override). **TRISC0/1/2** reset PCs
   are programmable L1 `0x6000/0xA000/0xE000`.

4. **M1 handoff**: x280 writes a ramp to `(8,3):0x30002000`; a bare-metal Tensix BRISC NoC-reads
   `(8,3):0x30002000` bit-exact (`responses=1`, no tearing). Both engines coexist on one shared exalens
   ctx. Validated raw and through the harness (fresh sentinels `0xC0FFEE00`, `0xDEC0DE00`).

5. **TRISC compute threads run our code**: all three (unpack/math/pack) cold-booted bare-metal and wrote
   the expected ramp, `in_reset=False`. This unblocks the **matrix engine (MVMUL, T1-math)** and **SFPU
   (T1 vector)** — where all dense splatting math lives.

---

## The harness (in tree)

**`src/bhtop/tensix/baremetal.py`** — `BareMetal`, the tt-metal-free launcher (3rd Tensix launch path,
sibling of `loader.py`=metal-hybrid, `bootloader.py`=metal-park):

```python
from bhtop.tensix.baremetal import BareMetal, bm_coord
bm = BareMetal(1, 2, ctx=ctx, risc="trisc1")     # worker noc0 (x,y); risc ∈ brisc/trisc0/1/2
bm.run(BareMetal.build("nocread", bm.pc), params=[bm_coord(8,3), 0x30002000, 32])
print(bm.result(), bm.dbg())                      # payload @0x2000, debug @0x2100
```
- `_RESET_PC = {brisc:0x0, trisc0:0x6000, trisc1:0xA000, trisc2:0xE000}`
- `run()` halts → poisons `BM_RESULT` → loads at `self.pc` → `set_code_start_address` (compute RISCs) →
  deasserts. `run_canon(name)` builds at `self.pc` and runs. `build(name, base)` compiles at `base`.

**`src/bhtop/kernels/tensix/baremetal/`** — cold-boot kernel canon (folder-per-kernel):
- `baremetal.h` — crt0/bm_main ABI (`BM_ARGS 0x1000`, `BM_RESULT 0x2000`, `BM_DBG 0x2100`) + the
  reusable inline **`bm_noc0_read(coord, src, dst, len)`** (the extracted NOC0 read).
- `crt0.s` — `_start`: set sp, load 4 params @`0x1000`, `call bm_main`, park. `link.ld` — `.=0x0`.
- `hello/` (proof-of-exec), `nocread/` (NoC read), `build.sh`.

---

## Extracted hardware knowledge (the reusable bricks)

### Bare-metal Blackhole NOC0 read (silicon-verified)
BRISC read cmd buffer (index 1) base `0xFFB20800`. Sequence: spin `NOC_CMD_CTRL(0xFFB20840)==0` →
snapshot `NOC_RD_RESP_CNT(0xFFB20208)` → write `NOC_CTRL(0x…1C)=0x2090`, RET `LO/MID/COORD`, TARG
`LO/MID/COORD`, `AT_LEN_BE(0x…20)` → `NOC_CMD_CTRL=1` (fire) → spin until resp-cnt +1.
- Coord encoding: `(y<<6)|x` — e.g. L2CPU tile0 = NoC0 `(8,3)` = `0xC8`. Full NoC addr = `(y<<42)|(x<<36)|local`; split LO=local, MID=`(local>>32)&0xF`, COORD=`(y<<6)|x`. Own coord from `NOC_NODE_ID(0xFFB20844)&0xFFF`.
- `NOC_CTRL=0x2090` = CPY|RD|RESP_MARKED(0x10)|VC_STATIC(0x80)|STATIC_VC(1)=0x2000. RESP_MARKED required or the counter never moves.
- **Cold worker: NoC works with NO init** (NIU live at reset; `noc_init` is only prefill). Clear
  `NIU_CFG_0(0xFFB20100)` bit14 (`NOC_ID_TRANSLATE_EN`) → physical coords. Align: L1 16B, DRAM 64B.
  NOC1 = +`0x10000` and mirrored coords.

### Address map
- x280 phys == NoC addr 1:1 in low window `0..0x7FFF_FFFFFFFF`. **DANGER:** NIU cfg window / ARC-over-NoC hang NoC0 → `tt-smi -r 0`.
- L2CPU tiles: idx→noc0/reset-bit = `0:(8,3)/4, 1:(8,9)/5, 2:(8,5)/6, 3:(8,7)/7`.
- x280 GDDR window (tile-local): TRAMP `0x30000000`, TELE `0x30002000` (host-readable), CODE `0x30008000`. Peripheral (uncached): RESET_VEC `0x20010000`, TRIGGER `0x20010414`, cmd mailbox `0x20010100`.
- Off-chip GDDR `dram_bank` @ noc0 `0x0` (4 GiB, real); Tensix L1 @ `0x2000000000`. **These are different endpoints** from the x280's tile-local window (the address-identity nuance — M1 read the L2CPU tile directly, which any NoC master can).

### TRISC cold-boot
Reset PCs are **L1** on Blackhole (`0x6000/0xA000/0xE000`) and programmable via
`set_code_start_address`. The "TRISC/NCRISC fetch from IRAM not L1" note in `bootloader.py:157-165` is a
**Wormhole-ism, FALSE on Blackhole** (`dev_mem_map.h:74` "No IRAM constraints"). Link the kernel at the
thread's reset PC; that's the only difference from BRISC. Issuing actual Tensix instrs (MVMUL/SFPU) needs
the gathering chicken-bit CSR `0x7c0` bit18 (`firmware_common.h:256-307`) — not needed for plain
load/store kernels.

### Coherence
x280 D-cache does NOT snoop inbound NoC writes, and `cbo.inval` **traps illegal** on this x280. So all
cross-engine control buffers (counters, work handoff) MUST live in **uncached** memory (peripheral
scratch `0x2001_xxxx`, or the uncached GDDR window) — coherence by region choice, never a cache op.

---

## Pathclear roadmap — critical hardware paths for splatting

Dependency-ordered. The compute-engine substrate (TRISC cold-boot → MVMUL) was the longest pole; it's
cleared/building. Everything after is reachable engineering.

1. **TRISC cold-boot** — ✅ CLEARED. Gates all dense compute.
2. **MVMUL correctness (matrix engine)** — ✅ **CLEARED (RUN done on silicon 2026-07-04).** The full
   `matmul_perf` L1_TO_L1 pipeline computes `A@B` bit-exact (0/1024 mismatch, reproducible ×3) over the
   bare-metal TRISC boot, no tt-metal. Driven by `tensix.matmul.run_matmul`. Precision characterized:
   the **bf16 FPU multiply is exact only to 6-bit inputs** (see *Immediate next* → now the SFPU/int lane).
3. **SFPU exp/blend** — ▷ reachable (T1 vector, same substrate as MVMUL). Raster exp/alpha, grad arith,
   D-SSIM. Minimal: bare-metal SFPU `exp` on a tile.
4. **GDDR stream** — ▷ bare-metal Tensix NoC-read of `dram_bank`@noc0 `0x0` + depth-≥2 CB prefetch
   (issue read for block i+1 while computing i). For DRAM-resident params at millions scale.
5. **Multi-core grid** — ▷ cold-boot the ~120-worker grid (loop BabyRiscDebug per worker, or a NoC
   multicast write of the kernel) with per-core tile assignment.
6. **DRAM circular buffer (M2)** — ▷ het producer/consumer with `tiles_received`/`tiles_acked`
   backpressure in **uncached** GDDR; x280 fills block i+1 while a bare-metal Tensix core drains i.
7. **x280 irregular tier (M3)** — ▷ bin/sort (scalar/counting), sorted-gid→param gather (`vrgather`/
   `vlseg3`, f2u-format), ordered-FP32 scatter-add. x280 side is mature (RVV catalog + CRT proven).

Note: 3 of 7 pathclear-map readers (sfpu/gddr/x280) hit the workflow's structured-output cap — re-run
`splat-hw-pathclear` if their detailed maps are wanted; they're downstream/parallel, not blocking.

---

## ✅ DONE: MVMUL bit-exact `A@B` (the RUN, 2026-07-04)

`matmul_perf` L1_TO_L1 runs the full unpack→math→pack pipeline bare-metal (all 3 mailboxes `0xFF`) and
computes `A@B` **bit-exact** on silicon (worker noc0 (1,2), 0/1024 mismatch, reproducible ×3), NO
tt-metal. Host harness = **`src/bhtop/tensix/matmul.py`** (`run_matmul` — tilize/pack/golden/verify in
pure Python, no numpy/torch; integers encoded to bf16 by bit-fiddling). The three deltas, resolved:

1. **Runtime dims** — `llk_run.run(runtime_words=[...])` now writes all 10 u32 @`0x20000` in the bhtop
   RuntimeParams order `TILE_CNT,CT,KT,LOOP,RT,TSA,TSB,TRANSPOSE,NFA,NFB`. Single 32×32 tile =
   `[1,1,1,1,1,128,128,0,4,4]`. **The tile size is 128** = 2048 B / 16-B words (tt-llk `TILE_SIZES`:
   bf16→default 128, Float32→256, Bfp8_b→68 — i.e. `format_tile_sizes / 16`; unit is 16-B words, the
   value stored to the TILE_SIZE_A/B GPR). Was the wedge risk; now nailed.
2. **Precision knobs** — `llk_run.build(..., fidelity='HiFi4', fp32_acc=..., formats=...)` threads into
   `llk.gen_build_h` (parametric `is_fp32_dest_acc_en`, `MATH_FIDELITY`, and the static-constexpr
   FormatConfig codes). Blackhole DataFormat codes: **bf16=5, Float32=0**. fp32 output = FormatConfig
   `(5×8, 0,0,0,0)` (bf16 in, fp32 pack_src+dst) + `fp32_acc=True` — verified correct on silicon.
3. **Operands + verify** — A,B tilized to 4×16×16 faces, written to `PERF_INPUT_A=0x21000` /
   `PERF_INPUT_B=0x31000` (raw byte addrs; the kernel-side `PERF_ADDRESS=buf/16-1` is the Tensix word
   encoding, not a host addr); output read from `PERF_OUTPUT=0x51000`, un-tiled, compared to an integer
   golden. **PASS = 3 mailboxes `0xFF` AND de-tiled == `A@B` bit-exact.**

### bf16 FPU precision wall (measured — the reason CRT/int-matmul-eval exists)
bf16 in / fp32 dest-acc / fp32 out, HiFi4, non-negative integer A,B, K=32:

| max input | golden max | bit-exact? |
|---|---|---|
| 7 | 448 | ✅ |
| 31 | 8 256 | ✅ |
| 63 | 38 128 | ✅ |
| 95 | 91 216 | ❌ (995/1024 off by a few ULP) |
| 127 | 161 216 | ❌ (981/1024) |

So the fp32 accumulator is not the limiter (outputs to 2^24 are fp32-exact) — the **bf16×bf16 FPU
multiply** is: exact for ≤6-bit operands, drifts at 7-bit (95, 127). Corroborates the earlier
`[[tt-splat-matmul-raster]]` "HiFi2==HiFi4, bf16 fringe" wall. **LoFi is fully inexact even at small
inputs** → HiFi4 is load-bearing.

### ✅ EXACT integer matmul on the proven datapath — limb decomposition (2026-07-04)
"Frame it as ints." The FPU multiplies **integer mantissas** under the hood; bf16 just wraps them in
float packaging that truncates at 7-bit. Two ways to get the ints out clean:

1. **Native Int8→Int32 datapath** (`run_matmul(out_format='int32')`) — **DONE, bit-exact on silicon,
   signed ±127, 1 matmul, STOCK `matmul_perf` (no kernel patch).** Setting the math format to Int8
   auto-flips the FPU into integer MAC (`_llk_math_hw_configure_` → `ALU_ACC_CTRL_INT8_math_enabled`;
   int8 requires FP32 dest mode). Three **host-side** facts made it exact (all from tt-isa
   MVMUL.md/SrcASrcB.md/Dst.md — same `MVMUL 0x26` opcode, no separate int instruction):
   - **HiFi4 is REQUIRED** (not LoFi). Integer fidelity phases are *inverted* vs float — phase 0 reads
     the HIGH magnitude bits, the int8 LSBs live in **phase 3**; the LLK matmul runs phases
     `0..fidelity-1`, so LoFi (phase 0 only) drops the LSBs → all-ones input gave an ALL-ZERO tile. The
     early guess "int MAC needs no phases → LoFi" was exactly backwards.
   - Src int8 is **sign-magnitude** on FP16 (SrcA ignores top 2 mag bits → usable ±255), not two's
     complement — the operand encoder writes sign-magnitude bytes.
   - The int32 **DEST is sign-magnitude** too — the readout decodes bit31=sign, bits0..30=magnitude.
   The zero-flag was a red herring (a correctly-unpacked int8 has exp=16, survives the denormal flush).
2. **Limb decomposition over the bf16 kernel** (`run_matmul_int`, base ≤ 64) — DONE, bit-exact. Split
   each int `x = base·xh + xl`, digits ≤ 63 (inside the bf16 exact window); four fp32-exact sub-matmuls
   recombine on host: `A@B = base²·Ah@Bh + base·(Ah@Bl + Al@Bh) + Al@Bl`. Measured inputs **0..4092**,
   out **191,067,664** (≫ 2²⁴), **0/1024**, 4 matmuls. The path for values **beyond int8** (splatting
   eval with wide accumulators / `[[project-crt-matmul]]` RNS); int8-native is the 1-matmul fast path.

Next: wire `proto_intmm` `phi@hi`/`phi@lo` (or a straight int8 GEMM) as TRISC LLK kernels on this
proven substrate → `proto_refuse` ones-reduce; SFPU exp/blend.

## ✅ Gaussian-splatting FORWARD on bare metal (2026-07-04) — `tensix.splat`

The int-matmul-eval raster (`tt-splat/scratchpad/proto_intmm_raster.py`) ported off ttnn onto our own
bare-metal MVMUL substrate. `src/bhtop/tensix/splat.py`, pure-Python host (no numpy/torch/ttnn):
- **Rung 1 — Gaussian EVAL on silicon.** The whitened field `v = φ·ψ` with ψ split into two int8 limbs
  (base 128) → two exact int8→int32 MVMULs (`φ@hi`, `φ@lo`) on the bare-metal FPU integer datapath;
  reconstruct `v`, `E = -½|v|²` on host. `eval_tile`: 32px × 16 Gaussians, **limb matmuls bit-exact vs
  host int matmul**, alpha-weighted-L1 = **2.66e-4** (matches proto CPU ref ~3.3e-4).
- **Rung 4 — full 32×32 tile RGB render.** `render_tile`: one prebuilt int8 kernel, loop the 32 pixel-row
  tiles (`build_for('int32')` once → `run_matmul(prebuilt=True)`), host exp/opacity→α + front-to-back
  composite. **67.7 dB vs the exact golden render** (16×16 = 78.9 dB), all limb matmuls bit-exact.
  Image: `scratchpad/splat_bare_metal.png` (device render == golden by eye).
- **Rung 2 — SFPU exp/log on bare metal: DONE (2026-07-04).** `src/bhtop/tensix/sfpu.py` runs
  `eltwise_unary_sfpu_perf` L1_TO_L1 with a `SFPU_UNARY_OPERATION` override (via new `gen_build_h`/
  `build(overrides=...)` hook). exp: 1024/1024 within 3%, exp(0)=1.0, ~1.9e-2 bf16 rel err. log: OK.
  **GOTCHA: the perf kernel's `ITERATIONS` bounds SFPU coverage — 8 = ONE face (256 datums); a full
  32×32 tile (4 faces) needs `ITERATIONS=32`** or faces 1-3 pass through as raw input.
- **Rung 3 — FULLY on-device forward: DONE on silicon (2026-07-04), 52.9 dB.** `render_ondevice`
  renders a tile with **every arithmetic op on MVMUL + SFPU** (host only builds constant matrices and
  shuttles L1 tiles — no host arithmetic). The serial composite is reformulated so it needs only the two
  proven engines (NO eltwise-binary): 6 MVMUL + 5 SFPU stages —
  `V=φ@ψ → Vsq=square(V) → E=Vsq@Ppair(−0.5 pair-sum) → α_raw=exp(E) → α=α_raw@diag(op),
  −α=α_raw@diag(−op) → lpa=log(α), la=log1p(−α) → logw=[la|lpa]@[Stri;I] → w=exp(logw) → C=w@color`.
  Two matmuls fold pointwise work into constant matrices (the −½·Σv² pair-sum; the prefix-sum **and**
  +log(α) in one triangular+identity matmul). Silicon: 16×16, 8 pixel-groups, stage-major (each kernel
  built once), **52.9 dB** vs golden (host sim 56.3; device bf16 exp/log the small gap). Per-stage
  verified vs host sim (V 8.8e-3, Vsq 1.2e-2, E 1.4e-2, exp 6.8e-2). Image:
  `scratchpad/splat_ondevice.png` (fully-on-device == golden by eye).
- **The forward Gaussian-splat render is now 100% on the bare-metal Tensix substrate** (MVMUL bf16+int8,
  SFPU exp/log/square/log1p) — no ttnn, no tt-metal, no host math. The 67.7 dB int-limb hybrid also
  stands (higher-quality eval when needed).

## ✅ HETEROGENEOUS splat: x280 (irregular) + Tensix (dense) — DONE on silicon (2026-07-04)

The het-compute POC thesis realized on the splatting workload: the **x280 owns the irregular tier
(depth sort), the bare-metal Tensix grid owns the dense tier (eval+exp+composite)**, cooperating on ONE
shared exalens context — no ttnn, no tt-metal.
- **x280 depth sort** (`scratchpad/depth_sort.c`): reads Gaussian depths from GDDR, argsorts front-to-
  back, publishes the order to its telemetry window. Matches the host sort exactly on silicon. Two x280
  gotchas learned: (1) **scalar float TRAPS** — crt0 enables the vector unit (VS) not `mstatus.FS`, so
  sort by the **u32 IEEE-754 bit pattern** (identical order for non-negative depths), no float; (2) the
  host→hart input must be in **uncached** GDDR — the telemetry region (e.g. hart-3 window `0x30002300`)
  works; plain `0x3000_xxxx` is cached → the hart reads stale.
- **Single hetero render** (`scratchpad/hetero_splat.py`): x280 sort → `render_ondevice(order=...)` →
  **52.9 dB**. Both engines contributed to one frame.
- **MULTI-hetero** (`scratchpad/depth_sort_multi.c` + `splat_multihetero.png`): **3 x280 harts each sort
  a different scene IN PARALLEL** (hart-id picks each hart's slice of a packed input; each writes its own
  tele window) → Tensix renders all 3 tiles → 52.9/52.0/54.3 dB. N x280 engines + Tensix cooperating.
- **FULL FLEET**: all 4 L2CPU tiles bring up = **16 x280 cores**; ~12/16 run the parallel sort first-try.
  Stragglers = the x280 seize/reset-once flakiness ([[l2cpu-bootstrap]]) — re-loading a parked hart WEDGES
  it, so use one reliable first-load per hart; a clean 16/16 needs the reset-once bug worked around.
- **Synthesis POC page**: `scratchpad/hetero_poc.html` (published Artifact) — the whole arc, thesis→fleet,
  every metric on silicon, for the interview.
## ✅ DIRECT HANDOFF + DRAM CIRCULAR BUFFER (M2) — DONE on silicon (2026-07-04)

The host is now out of the x280→Tensix DATA path, and the two engines stream cooperatively.
- **Direct handoff** (`scratchpad/depth_gather.c` + `hetero_splat.py`-style driver): the x280 does
  sort **+ gather** (reorders the per-Gaussian rgb into depth order, integer index copy) and writes the
  result to its uncached GDDR; a bare-metal Tensix worker **NoC-reads it straight from (8,3)** (the M1
  primitive) — **0/48 u32 mismatch**, host never relayed it. Gotchas: NoC DRAM reads want **64B-aligned
  length** (a 200B read truncated the tail); keep the host→hart input inside one uncached tele window
  (a 65-word packed input spilled across the hart-1/hart-2 boundary → one garbage word).
- **Bare-metal Tensix NoC-WRITE** — new primitive `bm_noc0_write` in
  `kernels/tensix/baremetal/baremetal.h` (mirror of the read: TARG=local src, RET=remote dst,
  CTRL=`0x2092` = CPY|WR|RESP_MARKED|VC_STATIC|STATIC_VC1) + `nocwrite/` kernel. Verified: Tensix wrote
  `0xBEEF` into x280 GDDR. Note: the write-ack counter (`0xFFB20204`) didn't increment — poll
  **CMD_CTRL-accepted** instead; the write lands and is NoC-ordered.
- **DRAM circular buffer (M2)** (`scratchpad/cb_producer.c` [x280] + `kernels/.../baremetal/cb_consumer/`
  [Tensix] + `cb_loop.py`): a bounded ring + `produced`/`acked` counters in uncached GDDR. Producer fills
  slot i%N, publishes `produced`, and **blocks when the ring is full** waiting on the consumer's `acked`
  (NoC-written back). Consumer NoC-reads `produced`, drains each slot, checksums it, NoC-writes `acked`.
  **12 items through a 4-slot ring — ring wrapped 8×, all checksums exact, backpressure held** (producer
  never overwrote an un-acked slot). Both engines run concurrently on one exalens ctx; host only launches.

### ✅ FUSED STREAMING PIPELINE — the CB drives the render (2026-07-05)
`scratchpad/cb_render_producer.c` (x280) + `render_streaming.py` + `splat_streaming.png`. The complete
loop: the x280 argsorts each of T image tiles front-to-back and **streams the depth order into the
bounded ring** (blocking on `acked` when full); the **Tensix forward render is the consumer** —
`render_ondevice(order=<streamed>)` drains each tile as it arrives, then acks to unblock the producer.
Silicon: **4 tiles, N=2 ring, wrapped 2×, every streamed order == host sort, renders 53–54 dB**,
backpressure held. `render_ondevice` now takes `order=`/`gs=` so the composite order comes from the
x280 via the ring. This is the het-compute POC end to end: **irregular tier (x280) → GDDR CB → dense
tier (Tensix) → image**, no ttnn/tt-metal.

### ✅ DENSE OPERANDS THROUGH THE RING — zero host Gaussian-data relay (2026-07-05)
`scratchpad/cb_operands.c` (x280) + `render_ondevice(ring=...)`. The x280 now produces **all four
order-dependent dense operands** — ψ (eval), diag(op) & diag(−op) (opacity), color (composite) — by
gathering the sorted Gaussians and laying each into the exact **tilized bf16 32×32 layout** the matmul
reads, into shared **uncached GDDR** (the whole `0x30002000–0x30007000` window is x280→Tensix coherent,
~20 KB — measured). The Tensix render NoC-reads each operand straight into `PERF_INPUT_B` (`nocread`
with an L1-dest arg) and the matmul consumes it (`run_matmul(b_prestaged=True)`). Result: the render is
**pixel-identical to the host-staged version (52.9 dB)** — the host stages only the *static* operands
(Ppair, Mcomb) and pixel coords, and relays **zero Gaussian data**. New bare-metal `bm_noc0_write` was
added earlier for the CB ack; the operand path reuses the proven `nocread`. This nails the forward
algorithm's data flow: **x280 (sort + gather + tilize all Gaussian operands) → GDDR → Tensix (dense
matmul/SFPU render), no ttnn/tt-metal, no host Gaussian data.**

## ◑ BACKWARD PASS — started: x280 scalar FPU + fp32 scatter-add (2026-07-05)
Toward training (not just rendering). `scratchpad/cb_scatter.c`.
- **x280 scalar float UNLOCKED.** `__asm__("csrs mstatus, %0" :: "r"(0x6000))` (FS=Dirty) enables the
  x280 FPU — the earlier "scalar float TRAPS" was simply the un-enabled FS field, not a HW limit. This
  also unblocks on-x280 projection/whitening.
- **fp32 gradient SCATTER-ADD** — the backward's irregular tier (mirror of the forward gather): per-tile
  per-Gaussian gradients arrive in sorted-slot order; the x280 scatters each back to its original
  Gaussian id and accumulates across tiles into a per-Gaussian fp32 buffer. Silicon: T=4 tiles × K=16,
  **max abs err 1.3e-7** vs host (bit-exact fp32). Gotcha reconfirmed: keep multi-word host→x280 buffers
  in the OPEN uncached window (0x30005000+), clear of the tele-window 0x100 boundaries that glitch
  contiguous writes (one order word at 0x30002400 read out-of-range before the move).

**Backward ladder (remaining):** the dense backward is matmuls — the transposes of the forward:
`dL/dcolor = wᵀ @ dL/dC`; `dL/dα` via the TRANSPOSED triangular matmul (forward's prefix-sum ↔ backward's
suffix-sum) + the log/exp chain; `dL/dE = dL/dα·(op·ar)`; `dL/dVsq = dL/dE @ Ppairᵀ`, `dL/dV = dL/dVsq·2V`,
`dL/dψ = φᵀ @ dL/dV`; then chain ψ→(mean/cov/op) and the x280 scatter-add (done) accumulates per-Gaussian.
All on the proven MVMUL+SFPU substrate. Then wire into an Adam step for an on-device training loop.

### ◑ TRAINING LOOP + PERF + x280 PROJECTION (2026-07-05)
- **(perf) per-op build cache** — `llk_run.build/run(variant=, cache=)` compile each kernel config into
  its own ELF dir and skip recompiles. The SFPU ops (square/exp/log/log1p) no longer clobber each other,
  so after a **one-time 3.8 s warmup** each render is pure device dispatch. This is what makes a training
  loop feasible. `render_ondevice(prebuilt=True)` skips the matmul build.
- **(a) TRAINING LOOP CONVERGES** — `scratchpad/train_color.py`: fit Gaussian colors to a target tile on
  the on-device forward. Backward `dL/dcolor = wᵀ·dL/dC` (transposed color matmul) + un-sort → Adam.
  Silicon: loss 0.012→0.0005, **PSNR 19.2 → 32.7 dB over 16 steps, 571 ms/step** (device forward + host
  dL/dcolor). First real training on the het pipeline. (render now returns `w` for the backward.)
- **(b) x280 PROJECTION (whitening) on-chip** — `scratchpad/cb_whiten.c`: the x280 computes ψ coeffs
  (sa, m12, m22, c1, c2) from Σ⁻¹ + mean in **fp32** via hardware `fsqrt.s` + `fdiv` (FPU now enabled).
  Silicon: **4.2e-7 vs host** — the last host-side per-Gaussian compute, moved to the x280. (Gotcha:
  `__builtin_sqrtf` emits a libm call under `-nostdlib`; use inline `fsqrt.s`.)

### ✅ GEOMETRY BACKWARD — complete & grad-checked (2026-07-05)
`scratchpad/train_geometry.py`. Full analytic reverse chain for EVERY parameter:
`dL/dC → dL/dw → dL/dα` (the transmittance chain, a suffix-sum over the sorted Gaussians:
`dL/dα_i = dL/dw_i·T_i − (Σ_{j>i} dL/dw_j·w_j)/(1−α_i)`) `→ dL/dE = dL/dα·op·ar → dL/dv1,v2 = −dL/dE·v`
`→ dL/d(sa,m12,m22,gx,gy) → whitening backward → dL/d(a,b,c)`, plus `dL/dop`, `dL/dcolor`.
- **Grad-check vs finite differences: every param OK** (rel 4e-8 … 5e-3 for gx,gy,a,b,c,op,color).
- **Trains the full Gaussian** (position + shape + opacity + color) to a target: **PSNR 17.0 → 36.9 dB**
  over 60 Adam steps (image-space fit, so positions settle to an equivalent config, not the exact target
  set — expected for an under-determined image loss).
- **The device pipeline renders the trained scene at 37.0 dB** (matches the host-trained fit) — the het
  forward handles the optimized geometry. The backward runs host-analytic (the correct algorithm); the
  on-device port is the transposed matmuls (dL/dcolor=wᵀ@dL/dC, dL/dα via the transposed triangular
  matmul, dL/dψ=φᵀ@dL/dV) — same substrate, needs eltwise-binary + reciprocal on bare metal.

### ◑ ON-DEVICE BACKWARD — all primitives verified, first gradient on silicon (2026-07-05)
The three gates for porting the backward to the substrate are cleared:
- **SFPU reciprocal** (for 1/(1−α), 1/α) — bare-metal via `sfpu.run_unary("reciprocal")`, 5.6e-3 (bf16).
- **eltwise-binary multiply** (tile⊙tile, for dL/dw·w etc.) — `eltwise_binary_fpu_perf` L1_TO_L1 with an
  `ELTWISE_BINARY_OP=ELWMUL` override; bit-exact on silicon.
- **transpose SOLVED without a transpose kernel** — the operands that need transposing (`dL/dC`, `color`,
  `φ`) are host-originated or static, so transpose those SMALL operands host-side and every backward
  matmul is a standard `A@B`. `UNPACK_TRANSPOSE_FACES=1` alone only swaps faces (not within-face) → not a
  clean transpose, so this side-steps it.
- **First backward gradient on silicon:** `dL/dcolorᵀ = dL/dCᵀ @ w` (standard matmuls summed over the
  32-pixel groups) matches the host backward at **1.86e-3** (bf16).

Remaining = ASSEMBLE the full chain from these proven primitives (dL/dw = dL/dC@colorᵀ; dL/dα via the
strict-lower-triangular suffix-sum matmul + reciprocal + eltwise-mul; dL/dE; dL/dv; dL/dψ = φᵀ@dL/dV) and
CACHE the forward intermediates (w, α, ar, v1, v2) in L1/GDDR so the backward can read them. De-risked
engineering, not unknowns. Then wire x280 whitening (done) + scatter-add (done) into one live loop.
- Perf lever still open: cut the ~88 dispatches/render (multi-tile matmul RT_DIM>1 + fused SFPU).

---

## Milestone ladder (the POC)

- **M1 — sentinel handoff** ✅ DONE. x280 → Tensix, bit-exact, no tt-metal.
- **M2 — DRAM circular buffer.** x280 producer ↔ Tensix consumer, `received`/`acked` backpressure in
  uncached GDDR, overlapped fill/drain. The cooperative-streaming centerpiece.
- **M3 — real workload + the number.** x280 runs the splatting gather/sort → CB → Tensix eval; measured
  head-to-head vs the host-orchestrated path. The "perf on the table" thesis, quantified.

---

## Gotchas & recovery (learned on silicon)

- **Worst case is `tt-smi -r 0`** — clears NoC hang / x280 / wedged state; re-init clean (proven twice).
  It resets the x280 too → re-`bringup`. A wedged baby-RISC: `bootloader.soft_reset` over NoC first,
  else `tt-smi -r 0`.
- **The one thing worse than `tt-smi -r`** = the PSU dI/dt host-reboot — ONLY from slamming the full
  grid with a power-virus compute load. The cooperative-buffer work is inherently low-power; just don't
  launch a full-grid matmul storm without a soft-start ramp.
- **Sequencing:** never run the x280 and tt-metal init concurrently (metal's grid init collides + hangs)
  — moot now (tt-metal is out), but the general rule: one owner of a given resource at a time.
- **`pkill -f "pattern"` matches its OWN command line** → self-kill (exit 144). Use the `[b]racket`
  trick: `pkill -f "contributed/[b]ootloader"`.
- **Foreground `sleep` is blocked** by the harness — use a Monitor until-loop (`until <cond>; do sleep
  N; done`) or `run_in_background`.
- Stay on the **safe register surfaces**: Tensix L1 over NoC, the `0xFFB2xxxx` NoC cmd registers, the
  x280 GDDR/peripheral windows. Never the NIU config window or ARC-over-NoC (hang NoC0).
- Keep every bare-metal spin **bounded** (the canon caps at 5M and breaks) so a bad kernel can't hard-hang.

---

## Key file references

- Harness: `src/bhtop/tensix/baremetal.py`; canon `src/bhtop/kernels/tensix/baremetal/`.
- LLK lane: `src/bhtop/tensix/llk_run.py` (TRISC boot over exalens; `build(fidelity/fp32_acc/formats)`,
  `run(runtime_words=[...])`), `src/bhtop/tensix/llk.py` (`gen_build_h` — parametric precision/formats),
  `src/bhtop/tensix/matmul.py` (**the MVMUL RUN**: stage/tilize/golden/verify, pure-Python bf16),
  `src/bhtop/kernels/tensix/llk/matmul_perf/` (+ `build.sh`, `build.example.h`, `perf.h`).
- x280: `src/bhtop/l2cpu/` (`__init__.py`=L2cpu API, `regmap.py`=address map, `toolchain.py`), `crt/`.
- Reset control: `ttexalens/hardware/baby_risc_debug.py` (`set_reset_signal`, `set_code_start_address`),
  `.../blackhole/functional_worker_block.py` (TRISC reset PCs / debug info).
- Session memory: `~/.claude/.../memory/tt-het-noc-poc.md` (the running log with exact numbers).
