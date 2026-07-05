"""
tensix.llk — the bare-metal LLK perf kernels lane of the cockpit.

These are tt-llk's `tests/sources/*_perf.cpp` micro-benchmarks: real compute kernels built ON TOP
OF llk_lib (each #includes llk_unpack_*/llk_math_*/llk_pack_* and calls the _llk_* primitives).
Unlike the bootloader overlays (one freestanding BRISC blob), an LLK kernel is split across the
three Tensix compute threads — one `run_kernel()` per `#ifdef LLK_TRISC_{UNPACK,MATH,PACK}` — so a
tile flows unpack(T0) -> math(T1, FPU/SFPU) -> pack(T2), exactly the LLK execution model.

This module (a) IMPORTS those sources into folder-per-kernel canon
(src/bhtop/kernels/tensix/llk/<name>/{<name>.cpp,kernel.json}) by parsing each one for its TRISC
roles + the llk_lib headers each thread pulls in + its compile-time knobs, and (b) LOADS that canon
back as a registry for the cockpit — mirroring tensix.overlays. The build recipe (the exact
includes/defines/linker that compile these on llk_lib) lives in kernels/tensix/llk/build.sh, lifted
from tt-llk's own tests/python_tests/helpers/test_config.py.
"""
import json
import os
import re

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # .../bhtop
CANON_DIR = os.path.join(PKG, "kernels", "tensix", "llk")              # tracked, shipped
# gitignored per-user working tree — holds the build-status sidecar (a compile-test result is
# toolchain/tt-llk-version specific, so it must NOT be committed into the declarative canon jsons).
WORKDIR = os.path.expanduser("~/bhtop/kernels/tensix/llk")
STATUS_PATH = os.path.join(WORKDIR, "_status.json")

# tt-llk tests/sources — the upstream the canon is imported from.
_LLK_REL = os.path.join("tt_metal", "tt-llk", "tests", "sources")
_BH_REL = os.path.join("tt_metal", "tt-llk", "tt_llk_blackhole")

# TRISC thread <-> the #ifdef guard the source uses for that thread's run_kernel().
_TRISC_GUARDS = {"unpack": "LLK_TRISC_UNPACK", "math": "LLK_TRISC_MATH", "pack": "LLK_TRISC_PACK"}

# Compile-time knobs an LLK perf source may key on (surfaced in kernel.json so the cockpit can show
# what a variant needs — these come from the per-variant build.h the harness generates).
_KNOWN_DEFINES = [
    "ELTWISE_BINARY_OP", "MATH_FIDELITY", "MATH_OP", "PERF_RUN_TYPE", "SPEED_OF_LIGHT",
    "BROADCAST_TYPE", "REDUCE_DIM", "POOL_TYPE", "SFPU_OP", "DATA_COPY_TYPE", "THROTTLE_LEVEL",
    "is_fp32_dest_acc_en", "RUNTIME_FORMATS", "MATH_TRANSPOSE_FACES", "ACC_TO_DEST",
]


def metal_home():
    return os.environ.get("TT_METAL_HOME") or os.path.expanduser("~/tt-metal")


def llk_src_dir():
    d = os.path.join(metal_home(), _LLK_REL)
    return d if os.path.isdir(d) else None


# Family grouping for the tree (browsability — 15 kernels is a lot flat).
def _family(name):
    n = name
    if n.startswith("eltwise"):
        return "eltwise"
    if "matmul" in n:
        return "matmul"
    if "reduce" in n:
        return "reduce"
    if n.startswith("pack"):
        return "pack"
    if n.startswith("unpack"):
        return "unpack"
    if "transpose" in n:
        return "transpose"
    return "other"


# ---- build.h generation: a stand-in for tt-llk's per-variant generate_build_header -----------
# Compile-time symbol -> a default C++ declaration (the value tt-llk's harness would template in).
# Defaults are the simplest valid choice; swap them (or the build.h) for a specific variant.
_DECLS = {
    "ELTWISE_BINARY_OP":     "constexpr auto ELTWISE_BINARY_OP = ckernel::EltwiseBinaryType::ELWADD;",
    "MATH_FIDELITY":         "constexpr ckernel::MathFidelity MATH_FIDELITY = ckernel::MathFidelity::LoFi;",
    "BROADCAST_TYPE":        "constexpr auto BROADCAST_TYPE = ckernel::BroadcastType::NONE;",
    "REDUCE_DIM":            "constexpr auto REDUCE_DIM = ckernel::ReduceDim::REDUCE_ROW;",
    "POOL_TYPE":             "constexpr auto POOL_TYPE = ckernel::PoolType::SUM;",
    "MATH_TRANSPOSE_FACES":  "constexpr bool MATH_TRANSPOSE_FACES = false;",
    "ACC_TO_DEST":           "constexpr bool ACC_TO_DEST = false;",
    "APPROX_MODE":           "constexpr bool APPROX_MODE = false;",
    "ITERATIONS":            "constexpr int ITERATIONS = 8;",
    "THROTTLE_LEVEL":        "constexpr int THROTTLE_LEVEL = 0;",
    "UNPACK_TRANSPOSE_FACES":      "constexpr bool UNPACK_TRANSPOSE_FACES = false;",
    "UNPACK_TRANSPOSE_WITHIN_FACE":"constexpr bool UNPACK_TRANSPOSE_WITHIN_FACE = false;",
    "SFPU_BINARY_OPERATION": "constexpr ckernel::BinaryOp SFPU_BINARY_OPERATION = ckernel::BinaryOp::ADD;",
    "SFPU_UNARY_OPERATION":  "constexpr SfpuType SFPU_UNARY_OPERATION = SfpuType::gelu;",
    # dest-register sync mode: default Half (harness DestSync.Half; matmul_perf uses SyncHalf literally).
    "dest_sync":             "constexpr auto dest_sync = ckernel::DstSync::SyncHalf;",
    "DST_SYNC_MODE":         "constexpr auto DST_SYNC_MODE = ckernel::DstSync::SyncHalf;",
}
# Per-variant integer/bool compile-time constants the harness templates in (defaults = simplest run).
_DIM_DEFAULTS = {
    "LOOP_FACTOR": 1, "CT_DIM": 1, "KT_DIM": 1, "RT_DIM": 1, "DST_INDEX": 0,
    "num_faces": 4, "num_faces_A": 4, "num_faces_B": 4, "NUM_BLOCKS": 1,
    "NUM_TILES_IN_BLOCK": 1, "NUM_TILES_IN_BANK": 1, "SRCA_REUSE_COUNT": 1, "L": 1,
    "BLOCK_CT_DIM": 1, "BLOCK_RT_DIM": 1, "FULL_CT_DIM": 1, "FULL_RT_DIM": 1, "NUM_GUARD_TILES": 0,
}
_DIM_BOOLS = {"PARTIAL_FACE_A", "PARTIAL_FACE_B", "PARTIAL_FACE_MATH", "PARTIAL_FACE_PACK",
              "FAST_MODE", "CLAMP_NEGATIVE", "NARROW_TILE", "ADD_TOP_ROW", "TO_FROM_INT8",
              "IS_MAX_OP", "STABLE_SORT", "tilize_en", "disable_src_zero_flag"}
# Symbols handled specially or that are enum *members* (defined by llk headers), not decls we emit.
_SKIP_SYMS = {"PERF_RUN_TYPE", "REDUCE_ROW", "REDUCE_COL", "REDUCE_SCALAR", "SPEED_OF_LIGHT",
              "RUNTIME_FORMATS", "TILE_SIZE_PACK", "TILE_SIZE_UNPACK_A", "TILE_SIZE_UNPACK_B"}
_SYM_RE = (r"\b(ELTWISE_BINARY_OP|MATH_FIDELITY|BROADCAST_TYPE|REDUCE_DIM|POOL_TYPE|"
           r"MATH_TRANSPOSE_FACES|ACC_TO_DEST|APPROX_MODE|ITERATIONS|THROTTLE_LEVEL|"
           r"UNPACK_TRANSPOSE_FACES|UNPACK_TRANSPOSE_WITHIN_FACE|"
           r"SFPU_BINARY_OPERATION|SFPU_UNARY_OPERATION|dest_sync|DST_SYNC_MODE)\b")
# Preferred default run type, family-aware — a kernel static_asserts against modes that don't apply
# to its engine (e.g. an unpack kernel rejects MATH_ISOLATE), so lead with the matching ISOLATE.
_RUN_TYPE_PREF = {
    "unpack": ["UNPACK_ISOLATE", "L1_TO_L1", "PACK_ISOLATE", "MATH_ISOLATE", "L1_CONGESTION"],
    "pack":   ["PACK_ISOLATE", "L1_TO_L1", "UNPACK_ISOLATE", "MATH_ISOLATE", "L1_CONGESTION"],
    "_":      ["MATH_ISOLATE", "L1_TO_L1", "UNPACK_ISOLATE", "PACK_ISOLATE", "L1_CONGESTION"],
}

_BUILD_H_PREAMBLE = '''// SPDX-License-Identifier: Apache-2.0
// AUTO-GENERATED default build config (bhtop tensix.llk.gen_build_h) — a stand-in for the per-variant
// header tt-llk's harness generates (test_config.generate_build_header). Defaults to the simplest
// valid variant; for a specific op/format/fidelity, edit the decls or pass your own build.h.
#pragma once
#include <array>
#include <type_traits>

#include "operand.h"
#include "llk_defs.h"
#include "llk_sfpu_types.h"
#include "perf.h"
#include "tensix_types.h"

#define RUNTIME_PARAMETERS [[maybe_unused]] const struct RuntimeParams&

constexpr bool l1_acc_en      = false;
constexpr bool unpack_to_dest = false;

// Compile-time form (const members + constexpr ctor) so a `constexpr FormatConfig` can be built and
// its members (e.g. formats.math) used as NON-TYPE TEMPLATE ARGS — which the perf kernels require
// (call_binary_sfpu_operation<..., formats.math>, init_reduce<..., formats.math, ...>). Mirrors
// tt-llk format_config.py FORMATS_CONFIG_STRUCT_COMPILETIME.
struct FormatConfig
{
    const std::uint32_t unpack_A_src;
    const std::uint32_t unpack_B_src;
    const std::uint32_t unpack_S_src;
    const std::uint32_t unpack_A_dst;
    const std::uint32_t unpack_B_dst;
    const std::uint32_t unpack_S_dst;
    const std::uint32_t math;
    const std::uint32_t sfpu_math;
    const std::uint32_t pack_src;
    const std::uint32_t pack_dst;
    const std::uint32_t pack_S_src;
    const std::uint32_t pack_S_dst;

    constexpr FormatConfig(
        std::uint32_t unpack_A_src_, std::uint32_t unpack_B_src_, std::uint32_t unpack_S_src_,
        std::uint32_t unpack_A_dst_, std::uint32_t unpack_B_dst_, std::uint32_t unpack_S_dst_,
        std::uint32_t math_, std::uint32_t sfpu_math_,
        std::uint32_t pack_src_, std::uint32_t pack_dst_, std::uint32_t pack_S_src_, std::uint32_t pack_S_dst_) :
        unpack_A_src(unpack_A_src_), unpack_B_src(unpack_B_src_), unpack_S_src(unpack_S_src_),
        unpack_A_dst(unpack_A_dst_), unpack_B_dst(unpack_B_dst_), unpack_S_dst(unpack_S_dst_),
        math(math_), sfpu_math(sfpu_math_),
        pack_src(pack_src_), pack_dst(pack_dst_), pack_S_src(pack_S_src_), pack_S_dst(pack_S_dst_)
    {
    }
};
'''


def gen_build_h(name, run_type=None, fidelity=None, fp32_acc=None, formats=None, overrides=None):
    """Generate a default build.h for an LLK kernel: scan its source for the compile-time symbols it
    uses (emit a default decl for each) + the runtime params.<field>s it reads (build a RuntimeParams
    struct), and set PERF_RUN_TYPE to `run_type` (or the kernel's first supported mode).

    Precision overrides (for the matmul RUN and any bit-exact variant):
      fidelity  — ckernel::MathFidelity name string ("LoFi"/"HiFi2"/"HiFi3"/"HiFi4"); overrides the
                  default LoFi MATH_FIDELITY decl. Integer inputs need HiFi4 for full-mantissa exactness.
      fp32_acc  — is_fp32_dest_acc_en (bool). Default None -> false (dest accumulates in bf16). True =
                  fp32 dest accumulation (needed to keep large integer sums exact before the packer).
      formats   — an explicit 12-tuple of DataFormat integer codes for the static constexpr FormatConfig
                  (unpack_A/B/S_src, unpack_A/B/S_dst, math, sfpu_math, pack_src/dst, pack_S_src/dst).
                  Default None -> all Float16_b (code 5). Pass a tuple to change I/O formats (e.g. an
                  fp32 pack_dst for a wide-integer-exact output)."""
    import re
    src_path = os.path.join(CANON_DIR, name, name + ".cpp")
    with open(src_path, encoding="utf-8", errors="replace") as f:
        text = f.read()

    run_types = set(re.findall(r"PerfRunType::([A-Z0-9_]+)", text))
    pref = _RUN_TYPE_PREF.get(_family(name), _RUN_TYPE_PREF["_"])
    rt = run_type or next((r for r in pref if r in run_types), "MATH_ISOLATE")

    # a symbol the source already DEFINES (`SYM = ...`, not a `==` compare) must not be re-declared
    def _self_defined(sym):
        return re.search(rf"\b{re.escape(sym)}\b\s*=(?!=)", text) is not None

    overrides = overrides or {}
    syms = sorted(set(re.findall(_SYM_RE, text)) - _SKIP_SYMS)
    decls = []
    for s in syms:
        if _self_defined(s):
            continue
        if s in overrides:
            decls.append(overrides[s])
        elif s == "MATH_FIDELITY" and fidelity:
            decls.append(f"constexpr ckernel::MathFidelity MATH_FIDELITY = "
                         f"ckernel::MathFidelity::{fidelity};")
        elif s in _DECLS:
            decls.append(_DECLS[s])

    # runtime fields: params.<field> excluding the formats struct + the `#include "params.h"` match
    fields, params_used = [], set()
    for ln in text.splitlines():
        if "#include" in ln:
            continue
        for m in re.findall(r"params\.([A-Za-z_][A-Za-z0-9_]*)", ln):
            params_used.add(m)
            if m != "formats" and m not in fields:
                fields.append(m)
    fields.sort(key=lambda f: (f != "TILE_CNT", f))      # TILE_CNT first (loader writes it)

    # per-variant compile-time constants used BARE (not params.X, not already defined): safe defaults
    for nm, dv in _DIM_DEFAULTS.items():
        if nm not in params_used and re.search(rf"\b{re.escape(nm)}\b", text) and not _self_defined(nm):
            decls.append(f"constexpr std::uint32_t {nm} = {dv};")
    for nm in sorted(_DIM_BOOLS):
        if nm not in params_used and re.search(rf"\b{re.escape(nm)}\b", text) and not _self_defined(nm):
            decls.append(f"constexpr bool {nm} = false;")

    lines = [_BUILD_H_PREAMBLE.rstrip(), "",
             f"constexpr bool is_fp32_dest_acc_en = {'true' if fp32_acc else 'false'};", ""]
    lines += decls
    lines.append(f"constexpr auto PERF_RUN_TYPE = PerfRunType::{rt};")
    lines.append("")
    # `formats` is COMPILE-TIME (static constexpr) so formats.math etc. are usable as non-type
    # template args (the SPEED_OF_LIGHT/compile_time_formats path). Every field defaults to Float16_b
    # (Blackhole DataFormat enum value 5 — the harness fallback). Because formats is no longer a
    # runtime instance member, the loader writes only the scalar runtime fields (see llk_run.py).
    # Stimuli buffers the kernel binds to `const Operand&` (params.buffer_*) must be Operand-typed,
    # not std::uint32_t.
    # TILE_CNT is ALWAYS the first runtime field. `formats` is static (zero instance storage), so the
    # first scalar field sits at offset 0 — and the loader writes tile_cnt at offset 0 (see
    # llk_run._runtime_params). Emitting TILE_CNT first unconditionally keeps that write correct even
    # for kernels that never read params.TILE_CNT (harmless unused member) instead of corrupting
    # whatever field happened to sort first (e.g. CT_DIM on the matmul kernels).
    lines.append("struct RuntimeParams")
    lines.append("{")
    fmt_lit = ",".join(str(int(c)) for c in formats) if formats else "5,5,5,5,5,5, 5,5, 5,5,5,5"
    lines.append(f"    static constexpr FormatConfig formats = FormatConfig({fmt_lit});")
    lines.append("    std::uint32_t TILE_CNT;")
    for fld in fields:
        if fld == "TILE_CNT":
            continue
        ctype = "Operand" if fld.startswith("buffer_") else "std::uint32_t"
        lines.append(f"    {ctype} {fld};")
    lines.append("};")
    return "\n".join(lines) + "\n", rt, fields


def _title(name):
    """eltwise_binary_fpu_perf -> 'Eltwise Binary · FPU'."""
    base = re.sub(r"_perf$", "", name)
    parts = base.split("_")
    pretty = {"fpu": "FPU", "sfpu": "SFPU", "sdpa": "SDPA", "bcast": "bcast", "wh": "WH"}
    return " ".join(pretty.get(p, p.capitalize()) for p in parts)


def _parse_source(path):
    """Scan one *_perf.cpp: which TRISC threads it implements, the llk_lib headers each pulls in,
    and the compile-time knobs + PerfRunType modes it references."""
    with open(path, encoding="utf-8", errors="replace") as f:
        text = f.read()
    name = os.path.splitext(os.path.basename(path))[0]

    trisc = {}
    for thread, guard in _TRISC_GUARDS.items():
        # the region between this thread's #ifdef GUARD and the matching #endif
        m = re.search(rf"#ifdef\s+{guard}\b(.*?)#endif", text, re.S)
        if not m:
            continue
        body = m.group(1)
        headers = re.findall(r'#include\s+"(llk_[^"]+)"', body)
        trisc[thread] = {"llk_headers": sorted(set(headers))}

    defines = sorted({d for d in _KNOWN_DEFINES if re.search(rf"\b{re.escape(d)}\b", text)})
    run_types = sorted(set(re.findall(r"PerfRunType::([A-Z0-9_]+)", text)))
    pref = _RUN_TYPE_PREF.get(_family(name), _RUN_TYPE_PREF["_"])
    default_rt = next((r for r in pref if r in run_types), "MATH_ISOLATE")

    return {
        "kind": "llk_perf",
        "name": name,
        "title": _title(name),
        "family": _family(name),                  # tree grouping (eltwise/matmul/pack/unpack/reduce/…)
        "engine": "tensix",
        "build": "llk_lib",                       # built on top of llk_lib (see build.sh)
        "source": os.path.basename(path),
        "desc": f"LLK perf micro-benchmark ({_title(name)}) — built on llk_lib, split across the "
                "Tensix compute threads (unpack→math→pack). Imported from tt-llk tests/sources.",
        "trisc": trisc,                           # per-thread llk_lib headers (the dependency)
        "defines": defines,                       # compile-time knobs (from the variant build.h)
        "perf_run_types": run_types,              # isolation modes the kernel supports
        "default_run_type": default_rt,           # deterministic (family-aware) default PERF_RUN_TYPE
        # perf telemetry is uniform: zoned counters (INIT, TILE_LOOP) over the 5 Tensix perf banks,
        # published by the kernel to the L1 software perf region at 0x169000 (tt-llk perf.h
        # PERF_COUNTERS_BASE_ADDR; runs to the 0x16AFF4 profiler boundary). NOTE: this is NOT the
        # 0xFFB12000 HW debug perf MMIO (RISCV_DEBUG_REG_PERF_CNT_*) — that's a separate source the
        # plan conflated with this L1 region. Full bank/zone decode of 0x169000 is not wired yet;
        # llk_run.py only peeks the base word (see [[tensix-llk]] handoff).
        "telemetry": {
            "kind": "perf_counters",
            "zones": ["INIT", "TILE_LOOP"],
            "banks": ["INSTRN_THREAD", "FPU", "TDMA_UNPACK", "TDMA_PACK", "L1"],
            "note": "elapsed cycles + event counts per zone; published to the L1 perf region "
                    "at 0x169000. Decode pending — runner currently peeks the base word only.",
        },
        "upstream": os.path.join(_LLK_REL, os.path.basename(path)),
    }


# ---- import: tt-llk sources -> tracked canon folders ----------------------------------------
def import_kernels(dry_run=False):
    """(Re)generate the canon folder-per-kernel tree from tt-llk's *_perf.cpp. Copies each source
    into kernels/tensix/llk/<name>/<name>.cpp and writes its kernel.json. Returns a summary."""
    src = llk_src_dir()
    if not src:
        return {"ok": False, "error": f"tt-llk sources not found at {metal_home()}/{_LLK_REL}"}
    import shutil
    out = []
    for fn in sorted(os.listdir(src)):
        if not fn.endswith("_perf.cpp"):
            continue
        name = os.path.splitext(fn)[0]
        meta = _parse_source(os.path.join(src, fn))
        kdir = os.path.join(CANON_DIR, name)
        if not dry_run:
            os.makedirs(kdir, exist_ok=True)
            shutil.copy2(os.path.join(src, fn), os.path.join(kdir, fn))
            with open(os.path.join(kdir, "kernel.json"), "w") as f:
                json.dump(meta, f, indent=2)
                f.write("\n")
            text, rt, _ = gen_build_h(name)          # default-variant build.h for CLI build.sh
            with open(os.path.join(kdir, "build.example.h"), "w") as f:
                f.write(text)
        out.append({"name": name, "family": meta["family"], "threads": list(meta["trisc"]),
                    "defines": meta["defines"]})
    return {"ok": True, "count": len(out), "canon": CANON_DIR, "kernels": out}


# ---- load: canon -> registry for the cockpit ------------------------------------------------
def _status():
    """Per-user build-status sidecar {name: buildable} (gitignored working tree), or {}."""
    try:
        with open(STATUS_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def load():
    """The LLK perf-kernel registry, read from the tracked canon kernel.json files. Returns
    {available, count, kernels:[...]} — available=False if the canon hasn't been imported yet. The
    `buildable` flag is merged in from the gitignored status sidecar (a compile-test result), so the
    committed canon jsons stay purely declarative; absent until `mark_buildable()` has been run."""
    if not os.path.isdir(CANON_DIR):
        return {"available": False, "count": 0, "kernels": [],
                "error": "LLK canon not imported yet (run tensix.llk.import_kernels)"}
    status = _status()
    kernels = []
    for name in sorted(os.listdir(CANON_DIR)):
        kj = os.path.join(CANON_DIR, name, "kernel.json")
        if not os.path.isfile(kj):
            continue
        try:
            with open(kj) as f:
                m = json.load(f)
        except (OSError, ValueError):
            continue
        if m.get("kind") == "llk_perf":
            if name in status:
                m["buildable"] = status[name]     # runtime hint (not committed)
            kernels.append(m)
    return {"available": True, "count": len(kernels), "kernels": kernels, "canon": CANON_DIR}


def mark_buildable():
    """Compile-test each kernel with its auto-generated default build.h and cache `buildable` in the
    gitignored status sidecar (NOT the committed kernel.json), so the cockpit can honestly show which
    build out-of-the-box vs which need a hand-tuned / harness build.h (matmul Operand types, special
    sfpu signatures …). `default_run_type` is deterministic and already lives in the canon json."""
    from . import llk_run
    status, out = {}, []
    for name in sorted(os.listdir(CANON_DIR)):
        kj = os.path.join(CANON_DIR, name, "kernel.json")
        if not os.path.isfile(kj):
            continue
        with open(kj) as f:
            m = json.load(f)
        if m.get("kind") != "llk_perf":
            continue
        ok = bool(llk_run.build(name)["ok"])
        status[name] = ok
        out.append((name, ok))
    os.makedirs(WORKDIR, exist_ok=True)
    with open(STATUS_PATH, "w") as f:
        json.dump(status, f, indent=2)
    return out


def source(name):
    """The .cpp source of an imported LLK kernel (for the editor)."""
    p = os.path.join(CANON_DIR, name, name + ".cpp")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"no LLK kernel {name!r} in canon")
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read()


if __name__ == "__main__":   # python -m bhtop.tensix.llk  -> (re)import from tt-llk
    import pprint
    pprint.pprint(import_kernels())
