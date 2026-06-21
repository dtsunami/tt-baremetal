# LLK perf kernels — built on llk_lib

These are tt-llk's `tests/sources/*_perf.cpp` compute micro-benchmarks, imported into the cockpit as
folder-per-kernel canon. **Yes — they are built on top of `llk_lib`**: each source `#include`s the
LLK headers (`llk_unpack_*`, `llk_math_*`, `llk_pack_*`) and calls the `_llk_*` primitives.

## The execution model (why each kernel has three parts)

A Tensix tile flows across the three compute threads, so every perf kernel defines `run_kernel()`
three times, guarded by the thread it runs on:

| Thread | Guard | Role | Typical llk_lib headers |
|---|---|---|---|
| T0 | `#ifdef LLK_TRISC_UNPACK` | unpack L1 → SrcA/B regs | `llk_unpack_AB.h`, `llk_unpack_tilize.h`, … |
| T1 | `#ifdef LLK_TRISC_MATH` | FPU / SFPU compute | `llk_math_eltwise_binary.h`, `llk_math_matmul.h`, … |
| T2 | `#ifdef LLK_TRISC_PACK` | pack Dest regs → L1 | `llk_pack.h`, `llk_pack_untilize.h`, … |

Each `kernel.json` records, per kernel, which threads it implements and the exact llk_lib headers
each thread pulls in (the `trisc` field), plus the compile-time knobs it keys on (`defines`) and the
isolation modes it supports (`perf_run_types`: `MATH_ISOLATE`, `UNPACK_ISOLATE`, `PACK_ISOLATE`,
`L1_TO_L1`, `L1_CONGESTION`).

## Re-importing from tt-llk

`kernel.json` + the `.cpp` copies are generated from `$TT_METAL_HOME/tt_metal/tt-llk/tests/sources`:

```bash
python -m bhtop.tensix.llk      # re-scan + regenerate this canon after a tt-llk pull
```

## Building on llk_lib

`build.sh` is the distilled, readable form of tt-llk's own build recipe
(`tests/python_tests/helpers/test_config.py`). It compiles a kernel's present threads on llk_lib:

```bash
./build.sh eltwise_binary_fpu_perf            # -> _build/<name>/{UNPACK,MATH,PACK}.o
```

Output lands in the gitignored working tree `~/bhtop/kernels/tensix/llk/_build/<name>/`.

### The `build.h` variant header

Each variant (op, data format, math fidelity, tile count, run type) is configured by a generated
`build.h`. tt-llk's harness generates it per test variant; `build.sh` uses a kernel's shipped
`build.example.h` (or one you pass as the 2nd arg). See
`eltwise_binary_fpu_perf/build.example.h` for the shape — swap `ELTWISE_BINARY_OP` / `MATH_FIDELITY`
/ `PERF_RUN_TYPE` / `FormatConfig` to build a different variant.

### Verified

`build.sh eltwise_binary_fpu_perf` compiles **and links** all three threads (UNPACK/MATH/PACK) clean
on llk_lib into per-thread ELFs, with the existing `runtime/sfpi` toolchain — no device, no pytest.

## Do we need `tt_metal/llrt` or `tt_metal/jit_build`? — No.

- **Linking: no.** The link uses only tt-llk's own startup + linker scripts
  (`tests/helpers/src/trisc.cpp` + `helpers/ld/{<thread>,sections,memory.blackhole}.ld` +
  `tmu-crt0.S`) and sfpi. `llrt`/`jit_build` appear nowhere in the tt-llk harness. Verified by
  linking a real TRISC ELF (entry `0xb000`) from those files alone.
- **`jit_build`** is metal's *runtime* kernel build (the "ride metal's JIT" alternative). We build
  standalone on llk_lib instead, so it's not used.
- **`llrt`** (`tt_elffile`, `llrt::load`, `hal`, `tt_cluster`) is metal's *C++* ELF loader — only
  relevant if we loaded via metal's runtime. We don't.

## Running on device (exalens loader)

The loader is [`tensix/llk_run.py`](../../../tensix/llk_run.py) — it ports tt-llk's TRISC-boot
sequence (`run_elf_files`, `BootMode.TRISC`) onto bhtop's exalens context (**not** llrt). build.sh
compiles with `-DLLK_BOOT_MODE_TRISC`, so T0 self-boots (`device_setup()` +
`clear_trisc_soft_reset()` → releases T1/T2); the cockpit only deasserts TRISC0. Sequence: assert all
TRISC resets → `load_elf` trisc0/1/2 → write `RuntimeParams` to L1 `0x20000` → reset mailboxes →
deassert TRISC0 → poll mailboxes (`0x1FFB8`/+4/+8) for `KERNEL_COMPLETE` (`0xFF`).

From the cockpit: pick an 🧮 LLK kernel in the TENSIX tree → the **LLK Run** tab → set core x,y +
tiles → *Build + Load + Run*. Or by API:
```
POST /api/tensix/llk/build  {name}
POST /api/tensix/llk/run    {name, x, y, tile_cnt, timeout}
```
`MATH_ISOLATE` (the shipped `build.example.h`) is the best first test — math-only, least dependent on
unpack/pack data. v1 reports per-thread `KERNEL_COMPLETE`; if a thread shows `no-ack`, recover with
`tt-smi -r 0`.

### Still a later layer: perf-counter decode

v1 surfaces completion + a raw peek at the perf-counter L1 region (`0x169000`). Full decode — config
the 5 banks (`-DPERF_COUNTERS_COMPILED` + the shared-config write) and read zoned cycles (`INIT`,
`TILE_LOOP`) per the `telemetry` block — is the next telemetry layer (port tt-llk
`helpers/{counters,metrics}.py`). The `RuntimeParams` formats are zeroed in v1 (activity/cycle run);
wiring real `FormatConfig` values + a `build.h` generator gives correct multi-variant runs.
