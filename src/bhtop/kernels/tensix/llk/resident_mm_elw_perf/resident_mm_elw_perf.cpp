// SPDX-License-Identifier: Apache-2.0
//
// resident_mm_elw_perf — GAP0 MATMUL-LOAD + SFPU-BINARY: E = (A@B) ⊙ D with NO unpack mode switch.
// The Blackhole errata that wedges a resident matmul<->eltwise loop is triggered by the UNPACK MODE SWITCH
// (matmul-unpack <-> AB/A-unpack); render never switches (always matmul-unpack) and runs infinite. So here
// BOTH operands reach DEST via matmul: matmul(A@B)->DEST0, matmul(D@I)->DEST1 (I=identity, D@I=D), then
// SFPU binary MUL(0,1)->DEST0. Unpack is ALWAYS matmul-unpack; math is matmul+SFPU (render-clean). If this
// is multi-ring resident, Gap 0 closes on Tensix — the fused splat kernel does every eltwise this way.
//
// L1: A 0x21000 | B 0x31000 | D 0x61000 | I(identity) 0x71000 | OUT(E) 0x51000

#include <algorithm>
#include <cstdint>
#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"
#include "tensor_shape.h"

using namespace ckernel;

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr std::uint32_t MAX_TILES_DEST = is_fp32_dest_acc_en ? 4 : 8;
static constexpr std::uint32_t RESIDENT_DB   = 0x16000;
static constexpr std::uint32_t RESIDENT_DONE = 0x16010;
static constexpr std::uint32_t RESIDENT_HB   = 0x16020;
static constexpr std::uint32_t DBG_U = 0x16030, DBG_M = 0x16040, DBG_P = 0x16050;
static constexpr std::uint32_t D_ADDR = 0x61000, I_ADDR = 0x71000;

static inline void dbg(std::uint32_t a, std::uint32_t v)
{ *reinterpret_cast<volatile std::uint32_t*>(a) = v; ckernel::invalidate_data_cache(); }
static inline void publish(std::uint32_t a, std::uint32_t v)
{ *reinterpret_cast<volatile std::uint32_t*>(a) = v; ckernel::invalidate_data_cache(); }
static inline std::uint32_t wait_ring(std::uint32_t last)
{ volatile std::uint32_t* db = reinterpret_cast<volatile std::uint32_t*>(RESIDENT_DB);
  std::uint32_t r; do { ckernel::invalidate_data_cache(); r = db[0]; } while (r == last); return r; }

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;
    const std::uint32_t TSA = params.TILE_SIZE_UNPACK_A, TSB = params.TILE_SIZE_UNPACK_B;
    const std::uint32_t nfa = params.num_faces_A, nfb = params.num_faces_B;
    std::uint32_t CT = params.CT_DIM, RT = params.RT_DIM, KT = params.KT_DIM;
    const bool TR = params.UNPACK_TRANSPOSE_FACES;
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR; (void)LOOP_FACTOR;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst,
        FACE_R_DIM, FACE_R_DIM, nfa, nfb, TSA, TSB);
    _llk_unpack_AB_matmul_init_<>(TR, CT, RT, KT, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES, false, false);

    std::uint32_t last = 0;
    for (;;)
    {
        last = wait_ring(last);
        // BOTH operands via matmul-unpack (never switch mode): A@B, then D@I.
        _llk_unpack_AB_matmul_<>(PERF_ADDRESS(PERF_INPUT_A, 0), PERF_ADDRESS(PERF_INPUT_B, 0), 0, 0, TSA, TSB, false, false, CT, RT, KT);
        dbg(DBG_U, (last << 4) | 0x1);
        _llk_unpack_AB_matmul_<>(PERF_ADDRESS(D_ADDR, 0), PERF_ADDRESS(I_ADDR, 0), 0, 0, TSA, TSB, false, false, CT, RT, KT);
        dbg(DBG_U, (last << 4) | 0x2);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"
#include "llk_math_eltwise_binary_sfpu.h"
#include "sfpu_operations.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;
    std::uint32_t CT = params.CT_DIM, RT = params.RT_DIM, KT = params.KT_DIM; (void)KT;

    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>(TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, false, CT, RT);
    test_utils::call_binary_sfpu_operation_init<APPROX_MODE, ckernel::BinaryOp::MUL, ITERATIONS>();

    std::uint32_t last = 0;
    for (;;)
    {
        last = wait_ring(last);
        _llk_math_wait_for_dest_available_<dest_sync>();
        _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(0, CT, RT);   // A@B -> DEST[0]
        dbg(DBG_M, (last << 4) | 0x1);
        _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(1, CT, RT);   // D@I -> DEST[1] (= D)
        test_utils::call_binary_sfpu_operation<dest_sync, is_fp32_dest_acc_en, APPROX_MODE, ckernel::BinaryOp::MUL, ITERATIONS>(0, 1, 0);  // DEST0 = C1 ⊙ D
        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        dbg(DBG_M, (last << 4) | 0x2);
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;
    std::uint32_t CT = params.CT_DIM, RT = params.RT_DIM;

    _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(formats.pack_src, formats.pack_dst, TILE_C_DIM * TILE_R_DIM);
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    std::uint32_t last = 0, beats = 0;
    for (;;)
    {
        last = wait_ring(last);
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t t = 0; t < CT * RT; t++)
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(t % MAX_TILES_DEST, PERF_ADDRESS(PERF_OUTPUT, t % MAX_TILES_DEST));
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        ckernel::tensix_sync();
        dbg(DBG_P, (last << 4) | 0x2);
        publish(RESIDENT_HB, ++beats);
        publish(RESIDENT_DONE, last);
    }
}

#endif
