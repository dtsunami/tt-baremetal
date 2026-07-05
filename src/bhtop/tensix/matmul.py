"""
tensix.matmul — bare-metal Tensix MVMUL RUN: stage operands, run matmul_perf, verify bit-exact A@B.

This is the host half of the "MVMUL bit-exact A@B" milestone (the dominant int-matmul-eval lever for
splatting). It drives the tt-llk `matmul_perf` kernel over the bare-metal TRISC boot path
([[tensix-llk]] llk_run) with PERF_RUN_TYPE=L1_TO_L1 — the FULL unpack(T0)->math(T1 FPU MVMUL)->pack(T2)
pipeline — and checks the de-tiled output against a pure-Python integer golden.

No numpy/torch: operands are integer matrices, encoded to bf16 by bit manipulation (exact for integers
that fit bf16's 8-bit mantissa) and tiled into the 4x16x16 face layout the unpacker expects.

Two proven-safe verification recipes:
  * run_matmul(..., out_format="bf16")  — all-bf16, dest accumulates bf16. Bit-exact iff every element
    of A@B stays <= 256 (bf16 represents integers 0..256 exactly, and non-negative partial sums are
    monotonic so no intermediate overflows the exact range). The low-risk pipeline proof.
  * run_matmul(..., out_format="fp32", fp32_acc=True) — bf16 in, fp32 dest-acc, fp32 out. The fp32
    accumulator removes the output-side rounding, but the operands are still bf16 so the FPU MULTIPLY
    is the limiter. MEASURED ON SILICON (2026-07-04, worker (1,2)): bit-exact for non-negative integer
    inputs up to 63 (products <= ~4k, outputs to ~38k); at 7-bit inputs (95, 127) it drifts a few ULP
    (~981/1024 elements off by 3-12). This is the bf16 FPU multiply wall — the reason exact large-int
    matmul needs the CRT/RNS residue decomposition ([[project-crt-matmul]]) or a true Int8->Int32
    datapath, not a naive bf16 GEMM. LoFi is fully inexact even at small inputs (HiFi4 is load-bearing).

Fixed L1 layout (tt-llk perf.h): PERF_INPUT_A=0x21000, PERF_INPUT_B=0x31000, PERF_OUTPUT=0x51000.
RuntimeParams struct order (bhtop gen_build_h): TILE_CNT, CT_DIM, KT_DIM, LOOP_FACTOR, RT_DIM,
TILE_SIZE_UNPACK_A, TILE_SIZE_UNPACK_B, UNPACK_TRANSPOSE_FACES, num_faces_A, num_faces_B. For a single
32x32@32x32 bf16 tile: [1, 1, 1, 1, 1, 128, 128, 0, 4, 4] — the 128 is 2048 bytes / 16-byte words
(tt-llk TILE_SIZES; Float32 would be 256, Bfp8_b 68).
"""
import struct

from . import llk_run

# L1 stimuli buffers (tt-llk tests/helpers/include/perf.h) — raw byte addresses; the host uses these
# directly (the kernel-side PERF_ADDRESS(buf, tile) = buf/16 - 1 is the Tensix 16B-word encoding the
# LLK primitives want, not an address the host pokes).
PERF_INPUT_A = 0x21000
PERF_INPUT_B = 0x31000
PERF_OUTPUT  = 0x51000

TILE_R = TILE_C = 32
TILE_ELEMS = TILE_R * TILE_C          # 1024
FACE = 16
BF16_TILE_WORDS = TILE_ELEMS // 2     # 512 u32 (two bf16 per word)
FP32_TILE_WORDS = TILE_ELEMS          # 1024 u32

DF_FLOAT32  = 0                      # Blackhole DataFormat codes (format_config.py)
DF_FLOAT16B = 5                       # bf16
DF_INT32    = 8
DF_INT8     = 14

# TILE_SIZE (16-byte words) per format — tt-llk test_config TILE_SIZES (= format tile bytes / 16).
# bf16/fp16 2048/16=128, fp32/int32 4096/16=256, int8 1024/16=64.
TILE_SIZE_WORDS = {DF_FLOAT16B: 128, DF_FLOAT32: 256, DF_INT32: 256, DF_INT8: 64}


# ---- pure-Python float codecs (exact for the integers we feed) ------------------------------------
def _f32_bits(x):
    return struct.unpack("<I", struct.pack("<f", float(x)))[0]


def _bits_f32(b):
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def f32_to_bf16(x):
    """fp32 -> bf16 bits, round-to-nearest-even. Exact for integers with <=8 significant bits."""
    b = _f32_bits(x)
    if (b & 0x7FFFFFFF) > 0x7F800000:     # NaN -> keep it a NaN
        return (b >> 16) | 0x40
    b += 0x7FFF + ((b >> 16) & 1)         # RNE bias
    return (b >> 16) & 0xFFFF


def bf16_to_f32(h):
    return _bits_f32((h & 0xFFFF) << 16)


# ---- tilize / untilize for a 32x32 tile (4 faces of 16x16, face order f0,f1,f2,f3) ---------------
_FACE_ORIGINS = [(0, 0), (0, 16), (16, 0), (16, 16)]   # f0 TL, f1 TR, f2 BL, f3 BR


def tilize32(m):
    """Row-major 1024 (32x32) -> face-ordered 1024. Matches tt-llk tilize(num_faces=4)."""
    out = []
    for r0, c0 in _FACE_ORIGINS:
        for r in range(FACE):
            base = (r0 + r) * TILE_C + c0
            out.extend(m[base:base + FACE])
    return out


def untilize32(t):
    """Face-ordered 1024 -> row-major 1024 (32x32). Inverse of tilize32."""
    m = [0.0] * TILE_ELEMS
    i = 0
    for r0, c0 in _FACE_ORIGINS:
        for r in range(FACE):
            base = (r0 + r) * TILE_C + c0
            m[base:base + FACE] = t[i:i + FACE]
            i += FACE
    return m


# ---- L1 word packing -----------------------------------------------------------------------------
def pack_bf16_words(vals):
    """1024 numbers -> 512 u32 words (bf16 pairs, low half = even index)."""
    h = [f32_to_bf16(v) for v in vals]
    return [(h[2 * i] | (h[2 * i + 1] << 16)) for i in range(len(h) // 2)]


def unpack_bf16_words(words):
    out = []
    for w in words:
        out.append(bf16_to_f32(w & 0xFFFF))
        out.append(bf16_to_f32((w >> 16) & 0xFFFF))
    return out


def pack_fp32_words(vals):
    return [_f32_bits(v) for v in vals]


def unpack_fp32_words(words):
    return [_bits_f32(w) for w in words]


def _int8_sm(v):
    """Encode an int (-127..127) as Tensix SIGN-MAGNITUDE int8 (bit7 = sign, bits0..6 = magnitude) —
    NOT two's complement (tt-isa SrcASrcB.md: the FPU int8 datum is sign/magnitude on FP16)."""
    v = int(v)
    return (abs(v) & 0x7F) | (0x80 if v < 0 else 0)


def pack_int8_words(vals):
    """1024 ints (-127..127) -> 256 u32 words, 4 sign-magnitude int8 per word."""
    b = [_int8_sm(v) for v in vals]
    return [(b[4 * i] | (b[4 * i + 1] << 8) | (b[4 * i + 2] << 16) | (b[4 * i + 3] << 24))
            for i in range(len(b) // 4)]


def unpack_int32_words(words):
    """Blackhole DEST 'integer 32' is SIGN-MAGNITUDE (bit31 = sign, bits0..30 = magnitude), NOT
    two's complement (tt-isa Dst.md). Decode accordingly."""
    return [-(w & 0x7FFFFFFF) if (w >> 31) else (w & 0x7FFFFFFF) for w in words]


# ---- golden ---------------------------------------------------------------------------------------
def matmul_golden(a, b):
    """Exact integer A@B for row-major 32x32 a, b -> row-major 32x32 list."""
    c = [0] * TILE_ELEMS
    for i in range(TILE_R):
        arow = a[i * TILE_C:(i + 1) * TILE_C]
        for j in range(TILE_C):
            s = 0
            for k in range(TILE_C):
                s += arow[k] * b[k * TILE_C + j]
            c[i * TILE_C + j] = s
    return c


def default_operands(kind="small"):
    """Deterministic integer A,B (row-major 32x32).
       "small"  -> values in {0,1,2}: A@B <= 128, bf16-exact end to end (out_format="bf16").
       "int8"   -> values 0..15 (4-bit): well inside the measured bf16-exact window (inputs <= 63),
                   so A@B is bit-exact with out_format="fp32". (Larger inputs hit the bf16 FPU wall
                   — see run_matmul; use CRT/RNS or an Int8->Int32 datapath for full int range.)"""
    if kind == "small":
        a = [((i + k) % 3) for i in range(TILE_R) for k in range(TILE_C)]
        b = [((k * j + 1) % 3) for k in range(TILE_R) for j in range(TILE_C)]
    else:
        a = [((i * 7 + k) % 16) for i in range(TILE_R) for k in range(TILE_C)]
        b = [((k * 5 + j * 3) % 16) for k in range(TILE_R) for j in range(TILE_C)]
    return a, b


# out_format -> (FormatConfig, input DataFormat, output word count, decoder, default fp32_acc, default
# fidelity).
#   "bf16"/"fp32" — VALIDATED bit-exact on silicon (see run_matmul; fp32 exact for inputs <= 63).
#   "int32" — the native "frame as ints" datapath, VALIDATED bit-exact on silicon (stock matmul_perf,
#     no kernel patch). Math format Int8 flips the FPU to integer MAC (_llk_math_hw_configure_ auto-sets
#     ALU_ACC_CTRL_INT8_math_enabled; int8 requires FP32 dest mode). Two non-obvious HW facts (tt-isa
#     MVMUL.md/SrcASrcB.md/Dst.md), both handled here:
#       * FIDELITY: integer fidelity phases are INVERTED vs float — phase 0 reads the HIGH magnitude
#         bits, the int8 LSBs live in PHASE 3. The LLK matmul runs phases 0..fidelity-1, so LoFi (phase
#         0 only) drops the LSBs → e.g. all-ones input gives an ALL-ZERO tile. **HiFi4 (phases 0-3) is
#         REQUIRED** for correct int8 matmul (opposite of the float intuition that int needs no phases).
#       * Src int8 is sign-magnitude on FP16 (SrcA ignores the top 2 magnitude bits → usable ±255); the
#         int32 DEST is sign-magnitude too (see unpack_int32_words).
_MODES = {
    "bf16": dict(formats=None,                                 in_df=DF_FLOAT16B,
                 out_words=BF16_TILE_WORDS, fp32_acc=False, fidelity="HiFi4"),
    "fp32": dict(formats=(DF_FLOAT16B,) * 8 + (DF_FLOAT32,) * 4, in_df=DF_FLOAT16B,
                 out_words=FP32_TILE_WORDS, fp32_acc=True,  fidelity="HiFi4"),
    "int32": dict(formats=(DF_INT8,) * 8 + (DF_INT32,) * 4,    in_df=DF_INT8,
                  out_words=FP32_TILE_WORDS, fp32_acc=True,  fidelity="HiFi4"),
}


# ---- the RUN -------------------------------------------------------------------------------------
def build_for(out_format, *, fp32_acc=None, fidelity=None, name="matmul_perf"):
    """Compile the matmul_perf ELFs for one out_format ("bf16"/"fp32"/"int32") once, so a render loop can
    reuse them via run_matmul(..., prebuilt=True) instead of recompiling every tile."""
    m = _MODES[out_format]
    return llk_run.build(name, run_type="L1_TO_L1",
                         fidelity=fidelity if fidelity is not None else m["fidelity"],
                         fp32_acc=fp32_acc if fp32_acc is not None else m["fp32_acc"],
                         formats=m["formats"])


def run_matmul(coord, *, ctx, device_id=0, a=None, b=None, kind="small",
               out_format="bf16", fp32_acc=None, fidelity=None,
               name="matmul_perf", timeout=8.0, verbose=True, prebuilt=False, b_prestaged=False):
    """Build matmul_perf (L1_TO_L1 + fidelity + fp32_acc + formats), stage A,B at PERF_INPUT_A/B, run
    the full unpack->math->pack pipeline on the Tensix core at `coord`, read PERF_OUTPUT, untile, and
    compare to the integer golden. Returns a dict with per-thread completion + exact-match verdict.

    coord: exalens core coord ('x,y' or OnChipCoordinate). ctx: the shared exalens context.
    out_format: "bf16" | "fp32" | "int32" (see _MODES). "int32" = int8 in / int32 out, the exact
    integer datapath. fp32_acc / fidelity default per-mode (override to force)."""
    from ttexalens.tt_exalens_lib import write_words_to_device, read_words_from_device

    if a is None or b is None:
        a, b = default_operands(kind)
    m = _MODES[out_format]
    formats = m["formats"]
    in_df = m["in_df"]
    out_words_n = m["out_words"]
    if fp32_acc is None:
        fp32_acc = m["fp32_acc"]
    if fidelity is None:
        fidelity = m["fidelity"]

    encode = ((lambda mtx: pack_int8_words(tilize32(mtx))) if in_df == DF_INT8
              else (lambda mtx: pack_bf16_words([float(x) for x in tilize32(mtx)])))
    decode = (unpack_int32_words if out_format == "int32"
              else unpack_fp32_words if out_format == "fp32"
              else unpack_bf16_words)

    tsize = TILE_SIZE_WORDS[in_df]            # 128 (bf16) or 64 (int8)
    # RuntimeParams: TILE_CNT, CT, KT, LOOP, RT, TSA, TSB, TRANSPOSE, NFA, NFB  (single 32x32 tile)
    runtime_words = [1, 1, 1, 1, 1, tsize, tsize, 0, 4, 4]

    if prebuilt:
        build = {"ok": True, "run_type": "L1_TO_L1"}      # reuse ELFs from a prior build_for()
    else:
        build = llk_run.build(name, run_type="L1_TO_L1", fidelity=fidelity,
                              fp32_acc=fp32_acc, formats=formats)
        if not build["ok"]:
            return {"ok": False, "stage": "build", "log": build["log"]}

    # stage operands (tilized, encoded to the input format). b_prestaged: B already sits at
    # PERF_INPUT_B (e.g. NoC-read from the x280's ring) — don't overwrite it; `b` is kept only for the
    # host golden. This is how a dense operand flows x280 -> GDDR ring -> Tensix matmul with no host relay.
    write_words_to_device(coord, PERF_INPUT_A, encode(a), device_id=device_id, context=ctx)
    if not b_prestaged:
        write_words_to_device(coord, PERF_INPUT_B, encode(b), device_id=device_id, context=ctx)
    # poison the output region so a no-op kernel can't accidentally "pass"
    write_words_to_device(coord, PERF_OUTPUT, [0xBADF00D5] * out_words_n,
                          device_id=device_id, context=ctx)

    r = llk_run.run(name, coord, ctx=ctx, device_id=device_id,
                    runtime_words=runtime_words, timeout=timeout)

    out_words = read_words_from_device(coord, PERF_OUTPUT, device_id=device_id,
                                       word_count=out_words_n, context=ctx)
    c_dev = untilize32(decode(out_words))
    c_gold = matmul_golden(a, b)

    mism = [(i // TILE_C, i % TILE_C, c_gold[i], c_dev[i])
            for i in range(TILE_ELEMS) if float(c_gold[i]) != c_dev[i]]
    gold_max = max(c_gold)
    bit_exact = (len(mism) == 0)
    res = {
        "ok": r["ok"] and bit_exact,
        "kernel_complete": r["ok"],
        "bit_exact": bit_exact,
        "threads": r["threads"],
        "run_type": build["run_type"],
        "out_format": out_format,
        "fp32_acc": fp32_acc,
        "fidelity": fidelity,
        "golden_max": gold_max,
        "mismatches": len(mism),
        "sample_mismatches": mism[:8],
        "corner_gold": c_gold[0], "corner_dev": c_dev[0],
        "c_dev": c_dev,
        "coord": str(coord),
    }
    if verbose:
        print(f"[matmul] {coord} run_type={build['run_type']} out={out_format} fp32_acc={fp32_acc} "
              f"fidelity={fidelity}")
        print(f"[matmul] threads={r['threads']} kernel_complete={r['ok']}")
        print(f"[matmul] golden_max={gold_max} mismatches={len(mism)} "
              f"C[0,0] gold={c_gold[0]} dev={c_dev[0]}  -> "
              f"{'BIT-EXACT PASS' if res['ok'] else 'FAIL'}")
        if mism:
            print("[matmul] first mismatches (row,col,gold,dev):", mism[:8])
    return res


def run_matmul_int(coord, *, ctx, device_id=0, a=None, b=None, kind="int8",
                   base=64, name="matmul_perf", timeout=8.0, verbose=True):
    """EXACT integer A@B on the PROVEN bf16 datapath via limb (radix-`base`) decomposition — the
    int-matmul-eval / RNS lever ([[project-crt-matmul]]) realized on the validated kernel, no int8
    kernel config needed.

    Each integer is split x = base*xh + xl with digits in [0, base). With base <= 64 every digit sits
    inside the MEASURED bf16-exact window (inputs <= 63), so the four digit sub-matmuls
    Ah@Bh, Ah@Bl, Al@Bh, Al@Bl are each bit-exact (fp32 out, results < 2^24). They recombine on the
    host in exact Python integers:  A@B = base^2*(Ah@Bh) + base*(Ah@Bl + Al@Bh) + (Al@Bl).
    Supports non-negative inputs up to base^2 - 1 (base=64 -> 0..4095) at 4 device matmuls."""
    assert 2 <= base <= 64, "digits must stay in the measured bf16-exact window (<=63)"
    if a is None or b is None:
        a, b = default_operands(kind)

    def digits(m):
        return [x // base for x in m], [x % base for x in m]
    ah, al = digits(a)
    bh, bl = digits(b)

    def sub(x, y):
        r = run_matmul(coord, ctx=ctx, device_id=device_id, a=x, b=y,
                       out_format="fp32", timeout=timeout, verbose=False)
        return r
    r_hh, r_hl, r_lh, r_ll = sub(ah, bh), sub(ah, bl), sub(al, bh), sub(al, bl)
    subs_ok = all(r["kernel_complete"] and r["bit_exact"] for r in (r_hh, r_hl, r_lh, r_ll))
    hh, hl, lh, ll = r_hh["c_dev"], r_hl["c_dev"], r_lh["c_dev"], r_ll["c_dev"]

    c_dev = [int(round(base * base * hh[i] + base * (hl[i] + lh[i]) + ll[i]))
             for i in range(TILE_ELEMS)]
    c_gold = matmul_golden(a, b)
    mism = [(i // TILE_C, i % TILE_C, c_gold[i], c_dev[i])
            for i in range(TILE_ELEMS) if c_gold[i] != c_dev[i]]
    ok = subs_ok and not mism
    res = {"ok": ok, "bit_exact": not mism, "sub_matmuls_exact": subs_ok, "base": base,
           "golden_max": max(c_gold), "input_max": max(max(a), max(b)),
           "mismatches": len(mism), "sample_mismatches": mism[:8],
           "corner_gold": c_gold[0], "corner_dev": c_dev[0], "coord": str(coord)}
    if verbose:
        print(f"[matmul-int] {coord} base={base} 4×bf16 sub-matmuls (each exact={subs_ok})")
        print(f"[matmul-int] input_max={res['input_max']} golden_max={res['golden_max']} "
              f"mismatches={len(mism)} C[0,0] gold={c_gold[0]} dev={c_dev[0]} -> "
              f"{'BIT-EXACT PASS' if ok else 'FAIL'}")
        if mism:
            print("[matmul-int] first mismatches:", mism[:8])
    return res
