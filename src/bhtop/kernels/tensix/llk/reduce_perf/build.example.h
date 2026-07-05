// SPDX-License-Identifier: Apache-2.0
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

constexpr bool is_fp32_dest_acc_en = false;

constexpr auto POOL_TYPE = ckernel::PoolType::SUM;
constexpr auto REDUCE_DIM = ckernel::ReduceDim::REDUCE_ROW;
constexpr std::uint32_t num_faces = 4;
constexpr auto PERF_RUN_TYPE = PerfRunType::MATH_ISOLATE;

struct RuntimeParams
{
    static constexpr FormatConfig formats = FormatConfig(5,5,5,5,5,5, 5,5, 5,5,5,5);
    std::uint32_t TILE_CNT;
};
