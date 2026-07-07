// SPDX-License-Identifier: Apache-2.0
//
// resident_mm2_perf — a RESIDENT TWO-STAGE chained matmul: C2 = (A@B) @ D, both matmuls on one
// doorbell ring, with the stage-1 result STAGED IN L1 and re-consumed by stage 2. This proves the last
// mechanism the fused resident render needs: INTER-STAGE ON-DEVICE DATAFLOW — pack stage-N's output to
// an L1 scratch, then unpack it as stage-(N+1)'s input, synchronized on-device (no host round-trip
// between stages). The render's 11 stages are exactly this pattern (matmul/SFPU alternating), so once
// (A@B)@D works resident + bit-exact, the fused render is composition, not an unknown.
//
// The new sync: the dest-sync semaphores order math<->pack, but NOTHING orders pack->unpack across
// stages. So stage-2's unpack of C1 could read L1 before stage-1's pack TDMA has landed C1. We close
// that with an L1 flag S1: pack, after packing C1 AND confirming it landed (poll the last word off the
// host poison), publishes S1=ring; stage-2 unpack spins on S1==ring (with a fence each poll) before it
// reads C1. Same fence/doorbell discipline proven in resident_mm_perf.
//
// L1 map: A 0x21000 | B 0x31000 | C1 scratch 0x41000 | D 0x61000 | OUTPUT(C2) 0x51000
//   DB 0x16000 | DONE 0x16010 | HB 0x16020 | DBG_U/M/P 0x16030/40/50 | S1 0x16060

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "counters.h"
#include "llk_defs.h"
#include "params.h"
#include "perf.h"
#include "profiler.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr std::uint32_t MAX_TILES_DEST = is_fp32_dest_acc_en ? 4 : 8;

static constexpr std::uint32_t RESIDENT_DB   = 0x16000;
static constexpr std::uint32_t RESIDENT_DONE = 0x16010;
static constexpr std::uint32_t RESIDENT_HB   = 0x16020;
static constexpr std::uint32_t DBG_U         = 0x16030;
static constexpr std::uint32_t DBG_M         = 0x16040;
static constexpr std::uint32_t DBG_P         = 0x16050;
static constexpr std::uint32_t STAGE1_FLAG   = 0x16060;

static constexpr std::uint32_t C1_ADDR     = 0x41000;   // stage-1 output scratch = stage-2 A input
static constexpr std::uint32_t D_ADDR      = 0x61000;   // stage-2 B input
static constexpr std::uint32_t POISON      = 0xBADF00D5u;
static constexpr std::uint32_t OUT_WORDS   = is_fp32_dest_acc_en ? 1024u : 512u;

static inline void dbg(std::uint32_t addr, std::uint32_t v)
{
    *reinterpret_cast<volatile std::uint32_t*>(addr) = v;
    ckernel::invalidate_data_cache();
}
static inline void publish(std::uint32_t addr, std::uint32_t v)
{
    *reinterpret_cast<volatile std::uint32_t*>(addr) = v;
    ckernel::invalidate_data_cache();
}
static inline std::uint32_t wait_ring(std::uint32_t last)
{
    volatile std::uint32_t* db = reinterpret_cast<volatile std::uint32_t*>(RESIDENT_DB);
    std::uint32_t r;
    do { ckernel::invalidate_data_cache(); r = db[0]; } while (r == last);
    return r;
}
static inline void wait_flag(std::uint32_t addr, std::uint32_t val)
{
    volatile std::uint32_t* f = reinterpret_cast<volatile std::uint32_t*>(addr);
    do { ckernel::invalidate_data_cache(); } while (f[0] != val);
}
static inline void wait_landed(std::uint32_t last_word_addr)
{
    volatile std::uint32_t* p = reinterpret_cast<volatile std::uint32_t*>(last_word_addr);
    do { ckernel::invalidate_data_cache(); } while (p[0] == POISON);
}

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;
    const std::uint32_t TILE_SIZE_UNPACK_A = params.TILE_SIZE_UNPACK_A;
    const std::uint32_t TILE_SIZE_UNPACK_B = params.TILE_SIZE_UNPACK_B;
    const std::uint32_t num_faces_A        = params.num_faces_A;
    const std::uint32_t num_faces_B        = params.num_faces_B;
    std::uint32_t CT_DIM              = params.CT_DIM;
    std::uint32_t RT_DIM              = params.RT_DIM;
    std::uint32_t KT_DIM              = params.KT_DIM;
    const bool UNPACK_TRANSPOSE_FACES = params.UNPACK_TRANSPOSE_FACES;
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR; (void)LOOP_FACTOR;  // keep 10-field ABI

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst,
        FACE_R_DIM, FACE_R_DIM, num_faces_A, num_faces_B, TILE_SIZE_UNPACK_A, TILE_SIZE_UNPACK_B);
    _llk_unpack_AB_matmul_init_<>(UNPACK_TRANSPOSE_FACES, CT_DIM, RT_DIM, KT_DIM, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES, false, false);

    std::uint32_t last = 0;
    for (;;)
    {
        last = wait_ring(last);
        // stage 1: unpack A @ B
        for (std::uint32_t j = 0; j < KT_DIM; j++)
            _llk_unpack_AB_matmul_<>(PERF_ADDRESS(PERF_INPUT_A, j), PERF_ADDRESS(PERF_INPUT_B, j), j, j * CT_DIM,
                                     TILE_SIZE_UNPACK_A, TILE_SIZE_UNPACK_B, false, false, CT_DIM, RT_DIM, KT_DIM);
        dbg(DBG_U, (last << 4) | 0x1);
        // stage 2: wait until pack has landed C1 in L1, then unpack C1 @ D
        wait_flag(STAGE1_FLAG, last);
        for (std::uint32_t j = 0; j < KT_DIM; j++)
            _llk_unpack_AB_matmul_<>(PERF_ADDRESS(C1_ADDR, j), PERF_ADDRESS(D_ADDR, j), j, j * CT_DIM,
                                     TILE_SIZE_UNPACK_A, TILE_SIZE_UNPACK_B, false, false, CT_DIM, RT_DIM, KT_DIM);
        dbg(DBG_U, (last << 4) | 0x2);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;
    std::uint32_t CT_DIM = params.CT_DIM;
    std::uint32_t RT_DIM = params.RT_DIM;
    std::uint32_t KT_DIM = params.KT_DIM;

    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>(
        TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, false, CT_DIM, RT_DIM);

    std::uint32_t last = 0;
    for (;;)
    {
        last = wait_ring(last);
        // stage 1: C1 = A @ B
        _llk_math_wait_for_dest_available_<dest_sync>();
        for (std::uint32_t j = 0; j < KT_DIM; j++)
            _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(0, CT_DIM, RT_DIM);
        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        dbg(DBG_M, (last << 4) | 0x1);
        // stage 2: C2 = C1 @ D (matmul stalls in the backend until unpack produces C1/D src)
        _llk_math_wait_for_dest_available_<dest_sync>();
        for (std::uint32_t j = 0; j < KT_DIM; j++)
            _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(0, CT_DIM, RT_DIM);
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
    std::uint32_t CT_DIM = params.CT_DIM;
    std::uint32_t RT_DIM = params.RT_DIM;

    _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(formats.pack_src, formats.pack_dst, TILE_C_DIM * TILE_R_DIM);
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    std::uint32_t last = 0, beats = 0;
    for (;;)
    {
        last = wait_ring(last);
        // stage 1: pack C1 = A@B to the L1 scratch, confirm it landed, signal stage-2 unpack
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile = 0; tile < CT_DIM * RT_DIM; tile++)
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(tile % MAX_TILES_DEST, PERF_ADDRESS(C1_ADDR, tile % MAX_TILES_DEST));
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        wait_landed(C1_ADDR + (OUT_WORDS - 1) * 4);
        dbg(DBG_P, (last << 4) | 0x1);
        publish(STAGE1_FLAG, last);
        // stage 2: pack C2 = C1@D to OUTPUT, confirm landed, publish done
        _llk_packer_wait_for_math_done_();
        for (std::uint32_t tile = 0; tile < CT_DIM * RT_DIM; tile++)
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(tile % MAX_TILES_DEST, PERF_ADDRESS(PERF_OUTPUT, tile % MAX_TILES_DEST));
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        wait_landed(PERF_OUTPUT + (OUT_WORDS - 1) * 4);
        dbg(DBG_P, (last << 4) | 0x2);
        publish(RESIDENT_HB, ++beats);
        publish(RESIDENT_DONE, last);
    }
}

#endif
