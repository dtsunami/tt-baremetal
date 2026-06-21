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

struct FormatConfig
{
    std::uint32_t unpack_A_src = 0, unpack_B_src = 0, unpack_S_src = 0;
    std::uint32_t unpack_A_dst = 0, unpack_B_dst = 0, unpack_S_dst = 0;
    std::uint32_t math = 0, sfpu_math = 0;
    std::uint32_t pack_src = 0, pack_dst = 0, pack_S_src = 0, pack_S_dst = 0;
};

constexpr bool is_fp32_dest_acc_en = false;

constexpr auto BROADCAST_TYPE = ckernel::BroadcastType::NONE;
constexpr std::uint32_t num_faces = 4;
constexpr auto PERF_RUN_TYPE = PerfRunType::UNPACK_ISOLATE;

struct RuntimeParams
{
    FormatConfig formats;
    std::uint32_t TILE_CNT;
    std::uint32_t BLOCK_CT_DIM;
    std::uint32_t BLOCK_RT_DIM;
    std::uint32_t FULL_CT_DIM;
    std::uint32_t FULL_RT_DIM;
    std::uint32_t LOOP_FACTOR;
};
