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


# I/O precision: bf16 (default, fast, silicon-verified) or full fp32.
# ⚠️ fp32=True is EXPERIMENTAL and currently HANGS on silicon: with all-Float32 formats the eltwise/SFPU
# perf kernel's mailbox never reaches KERNEL_COMPLETE (every dispatch times out). The single-tile
# PERF_ADDRESS is tile-size-independent, so the FormatConfig swap *should* suffice, but something else
# (likely the fp32 tile-size GPR / TILE_SIZE runtime param, or fp32 dest-acc halving MAX_TILES_DEST) is
# needed. Left plumbed for a future fix; DO NOT use in a loop (it wedges the card via repeated timeouts).
# The backward's cancellation problem is real (bf16 dL/dalpha ≈ 12% L2) — fp32 or a better-conditioned
# reformulation is the fix, but not this code path yet.
_FP32_FORMATS = (MM.DF_FLOAT32,) * 12


def _io(fp32):
    if fp32:
        return (MM.pack_fp32_words, MM.unpack_fp32_words, MM.FP32_TILE_WORDS, _FP32_FORMATS, True, "_fp32")
    return (MM.pack_bf16_words, MM.unpack_bf16_words, MM.BF16_TILE_WORDS, None, False, "")


def build_unary(op, *, fp32=False, fp32_acc=None, formats=None, cache=True):
    """Compile eltwise_unary_sfpu_perf for SFPU op `op` into a per-op ELF cache (variant=op[+_fp32]), so
    distinct ops/precisions don't clobber each other. Reuse via run_unary(prebuilt=True)."""
    _, _, _, fmt, facc, suf = _io(fp32)
    return llk_run.build(_KERNEL, run_type="L1_TO_L1",
                         fp32_acc=fp32_acc if fp32_acc is not None else facc,
                         formats=formats if formats is not None else fmt,
                         overrides=_overrides(op), variant=op + suf, cache=cache)


def run_unary(coord, tile, *, ctx, device_id=0, op="exponential", fp32=False, fp32_acc=None,
              prebuilt=False, timeout=6.0):
    """Apply SFPU unary `op` to a 1024-element (row-major 32x32) tile on bare-metal Tensix; return the
    1024-element result (row-major). bf16 in/out by default; fp32=True for full fp32 I/O."""
    from ttexalens.tt_exalens_lib import write_words_to_device, read_words_from_device
    pack, unpack, nwords, _, _, suf = _io(fp32)
    if not prebuilt:
        b = build_unary(op, fp32=fp32, fp32_acc=fp32_acc)
        if not b["ok"]:
            raise RuntimeError("sfpu build failed:\n" + b["log"][-1500:])
    words = pack([float(x) for x in MM.tilize32(tile)])
    write_words_to_device(coord, MM.PERF_INPUT_A, words, device_id=device_id, context=ctx)
    write_words_to_device(coord, MM.PERF_OUTPUT, [0] * nwords, device_id=device_id, context=ctx)
    r = llk_run.run(_KERNEL, coord, ctx=ctx, device_id=device_id, runtime_words=_RUNTIME,
                    timeout=timeout, variant=op + suf)
    out = read_words_from_device(coord, MM.PERF_OUTPUT, device_id=device_id, word_count=nwords, context=ctx)
    return MM.untilize32(unpack(out)), r["ok"]


# --- eltwise BINARY (FPU) -------------------------------------------------------------------------
# Same perf-kernel substrate, different engine: eltwise_binary_fpu_perf reads PERF_INPUT_A + PERF_INPUT_B
# and applies one FPU binary op (add/sub/mul, tile ⊙ tile) selected by the ELTWISE_BINARY_OP override
# (ckernel::EltwiseBinaryType). Needed by the on-device backward (dL/dw·w, dL/dalpha subtract, ...).
_BIN_KERNEL = "eltwise_binary_fpu_perf"
_BIN_OP = {"add": "ELWADD", "sub": "ELWSUB", "mul": "ELWMUL"}


def build_binary(op, *, fp32=False, fp32_acc=None, formats=None, fidelity=None, cache=True):
    """Compile eltwise_binary_fpu_perf for FPU binary `op` ∈ {add,sub,mul} into a per-op ELF cache.
    Fidelity is op-dependent: MUL needs HiFi4 (the bf16 FPU multiply truncates mantissa bits under LoFi,
    ~3e-2 err → HiFi4 runs all mantissa phases); ADD/SUB don't multiply mantissas so LoFi is exact for
    them (and the LLK add/sub path doesn't compile under HiFi4 on this toolchain). fp32=True: full fp32 I/O."""
    assert op in _BIN_OP, f"binary op must be one of {list(_BIN_OP)}"
    if fidelity is None:
        fidelity = "HiFi4" if op == "mul" else None
    _, _, _, fmt, facc, suf = _io(fp32)
    ov = {"ELTWISE_BINARY_OP": f"constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::{_BIN_OP[op]};"}
    return llk_run.build(_BIN_KERNEL, run_type="L1_TO_L1", fidelity=fidelity,
                         fp32_acc=fp32_acc if fp32_acc is not None else facc,
                         formats=formats if formats is not None else fmt,
                         overrides=ov, variant=f"bin_{op}{suf}", cache=cache)


def run_binary(coord, a, b, *, ctx, device_id=0, op="mul", fp32=False, fp32_acc=None, fidelity=None,
               prebuilt=False, timeout=6.0):
    """Apply FPU binary `op` elementwise to two 1024-element (row-major 32x32) tiles on bare-metal
    Tensix; return the 1024-element result. bf16 in/out by default; fp32=True for full fp32 I/O.
    op ∈ {add,sub,mul} (a op b)."""
    from ttexalens.tt_exalens_lib import write_words_to_device, read_words_from_device
    pack, unpack, nwords, _, _, suf = _io(fp32)
    if not prebuilt:
        r = build_binary(op, fp32=fp32, fp32_acc=fp32_acc, fidelity=fidelity)
        if not r["ok"]:
            raise RuntimeError("binary build failed:\n" + r["log"][-1500:])
    write_words_to_device(coord, MM.PERF_INPUT_A, pack([float(x) for x in MM.tilize32(a)]),
                          device_id=device_id, context=ctx)
    write_words_to_device(coord, MM.PERF_INPUT_B, pack([float(x) for x in MM.tilize32(b)]),
                          device_id=device_id, context=ctx)
    write_words_to_device(coord, MM.PERF_OUTPUT, [0] * nwords, device_id=device_id, context=ctx)
    res = llk_run.run(_BIN_KERNEL, coord, ctx=ctx, device_id=device_id, runtime_words=_RUNTIME,
                      timeout=timeout, variant=f"bin_{op}{suf}")
    out = read_words_from_device(coord, MM.PERF_OUTPUT, device_id=device_id, word_count=nwords, context=ctx)
    return MM.untilize32(unpack(out)), res["ok"]
