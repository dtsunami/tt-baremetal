"""
tensix.sfpu — bare-metal SFPU (vector unit) tile ops over exalens, no ttnn/tt-metal.

Runs tt-llk's `eltwise_unary_sfpu_perf` (L1_TO_L1) on a cold-booted Tensix: T0 datacopies PERF_INPUT_A
into DEST, T1 applies one SFPU unary op (exp/log/...), T2 packs to PERF_OUTPUT. The op is a compile-time
knob (`SFPU_UNARY_OPERATION = SfpuType::<name>`), selected via llk.gen_build_h overrides.

Silicon-verified gotcha (2026-07-04): the perf kernel's `ITERATIONS` bounds SFPU coverage — 8 covers ONE
face (256 datums), a full 32x32 tile (4 faces = 1024) needs **ITERATIONS=32** (else faces 1-3 pass through
as raw input). exp on bf16: ~1.9e-2 max rel err (SFPU bf16 precision).

    from bhtop.tensix import sfpu
    sfpu.build_unary("exponential")                 # compile once
    out = sfpu.run_unary(coord, tile1024, ctx=ctx, prebuilt=True)   # list[1024] -> exp(tile)
"""
from . import llk_run, matmul as MM

_KERNEL = "eltwise_unary_sfpu_perf"
# SfpuType enumerator names (tt-llk llk_params MathOperation cpp_enum_value): exp="exponential",
# log="log", reciprocal="reciprocal", sqrt="sqrt", square="square", sigmoid="sigmoid", ...
_RUNTIME = [1, 1, 0, 0, 4]     # [TILE_CNT, LOOP_FACTOR, UNPACK_TRANSPOSE_FACES, WITHIN_FACE, num_faces]


def _overrides(op):
    return {
        "SFPU_UNARY_OPERATION": f"constexpr SfpuType SFPU_UNARY_OPERATION = SfpuType::{op};",
        "ITERATIONS": "constexpr int ITERATIONS = 32;",   # full 32x32 tile (4 faces), not 1
    }


def build_unary(op, *, fp32_acc=False, formats=None, cache=True):
    """Compile eltwise_unary_sfpu_perf for SFPU op `op` into a per-op ELF cache (variant=op), so distinct
    ops don't clobber each other and a training loop compiles each op only once. Reuse via
    run_unary(prebuilt=True)."""
    return llk_run.build(_KERNEL, run_type="L1_TO_L1", fp32_acc=fp32_acc, formats=formats,
                         overrides=_overrides(op), variant=op, cache=cache)


def run_unary(coord, tile, *, ctx, device_id=0, op="exponential", fp32_acc=False,
              prebuilt=False, timeout=6.0):
    """Apply SFPU unary `op` to a 1024-element (row-major 32x32) tile on bare-metal Tensix; return the
    1024-element result (row-major). bf16 in/out (default)."""
    from ttexalens.tt_exalens_lib import write_words_to_device, read_words_from_device
    if not prebuilt:
        b = build_unary(op, fp32_acc=fp32_acc)
        if not b["ok"]:
            raise RuntimeError("sfpu build failed:\n" + b["log"][-1500:])
    words = MM.pack_bf16_words([float(x) for x in MM.tilize32(tile)])
    write_words_to_device(coord, MM.PERF_INPUT_A, words, device_id=device_id, context=ctx)
    write_words_to_device(coord, MM.PERF_OUTPUT, [0] * MM.BF16_TILE_WORDS, device_id=device_id, context=ctx)
    r = llk_run.run(_KERNEL, coord, ctx=ctx, device_id=device_id, runtime_words=_RUNTIME,
                    timeout=timeout, variant=op)
    out = read_words_from_device(coord, MM.PERF_OUTPUT, device_id=device_id,
                                 word_count=MM.BF16_TILE_WORDS, context=ctx)
    return MM.untilize32(MM.unpack_bf16_words(out)), r["ok"]
