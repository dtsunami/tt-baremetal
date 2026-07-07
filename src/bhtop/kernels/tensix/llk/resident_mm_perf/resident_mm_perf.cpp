// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// resident_mm_perf — matmul_perf made RESIDENT + DOORBELL-DRIVEN, the keystone for the 120-worker
// resident render grid. All three Tensix compute threads run INIT once, then spin in a `for(;;)`
// doorbell loop: on each host ring they re-run the unpack->math->pack matmul over whatever operands
// currently sit at PERF_INPUT_A/B and publish DONE. Host loads ONCE, drives N tiles by (re)staging
// operands + ringing — collapsing the per-op host round-trip (the x280 opt_step.c doorbell trick,
// now across the 3 Tensix threads).
//
// Doorbell/done/debug in the free L1 gap (loader-init ends 0x15000, runtime args start 0x20000):
//   DB 0x16000 host->kernel ring | DONE 0x16010 kernel->host done | HB 0x16020 heartbeat
//   DBG_U 0x16030 / DBG_M 0x16040 / DBG_P 0x16050 : per-thread phase (ring<<4 | step) for hang triage.

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

// Globals
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

static inline void dbg(std::uint32_t addr, std::uint32_t v)
{
    *reinterpret_cast<volatile std::uint32_t*>(addr) = v;
    ckernel::invalidate_data_cache();   // fence so the host reliably observes the phase (else DCACHE-stale)
}

// spin until the host ring counter advances past `last`; returns the new ring value.
// invalidate_data_cache() is a FENCE (Blackhole invalidates the L0 DCACHE as a fence side effect), so
// each poll re-fetches the doorbell from L1 rather than reading a stale cached line. Without it the
// doorbell read is a coherence race — the boot thread (T0) caches DB=0 at boot and never sees the host
// bump it (the whole pipeline then stalls on unproduced src); the x280's classic host->core hazard.
static inline std::uint32_t wait_ring(std::uint32_t last)
{
    volatile std::uint32_t* db = reinterpret_cast<volatile std::uint32_t*>(RESIDENT_DB);
    std::uint32_t r;
    do
    {
        ckernel::invalidate_data_cache();
        r = db[0];
    } while (r == last);
    return r;
}

// publish a word to L1 so the host reliably observes it (store, then fence to flush the line).
static inline void publish(std::uint32_t addr, std::uint32_t v)
{
    *reinterpret_cast<volatile std::uint32_t*>(addr) = v;
    ckernel::invalidate_data_cache();
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
    // Reference LOOP_FACTOR so gen_build_h keeps it in RuntimeParams — the struct field order MUST match
    // stock matmul_perf's 10-field ABI (TILE_CNT,CT,KT,LOOP,RT,TSA,TSB,TRANSPOSE,NFA,NFB) that the host
    // writes. Drop it and every field past RT_DIM shifts by one → num_faces_A reads 0 → the hw_configure
    // LLK_ASSERT spins forever. (Unused in the resident loop; one matmul per ring.)
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR;
    (void)LOOP_FACTOR;

    dbg(DBG_U, 0xE1);   // entered unpack run_kernel
    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst,
        FACE_R_DIM, FACE_R_DIM, num_faces_A, num_faces_B, TILE_SIZE_UNPACK_A, TILE_SIZE_UNPACK_B);
    dbg(DBG_U, 0xE2);   // hw_configure done
    _llk_unpack_AB_matmul_init_<>(UNPACK_TRANSPOSE_FACES, CT_DIM, RT_DIM, KT_DIM, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES, false, false);

    dbg(DBG_U, 0xE0);   // reached loop (INIT done)
    std::uint32_t last = 0;
    for (;;)
    {
        last = wait_ring(last);
        dbg(DBG_U, (last << 4) | 0x1);
        for (std::uint32_t j = 0; j < KT_DIM; j++)
        {
            _llk_unpack_AB_matmul_<>(
                PERF_ADDRESS(PERF_INPUT_A, j), PERF_ADDRESS(PERF_INPUT_B, j), j, j * CT_DIM,
                TILE_SIZE_UNPACK_A, TILE_SIZE_UNPACK_B, false, false, CT_DIM, RT_DIM, KT_DIM);
        }
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

    dbg(DBG_M, 0xE0);   // reached loop (INIT done)
    std::uint32_t last = 0;
    for (;;)
    {
        last = wait_ring(last);
        dbg(DBG_M, (last << 4) | 0x1);
        _llk_math_wait_for_dest_available_<dest_sync>();
        dbg(DBG_M, (last << 4) | 0x2);
        for (std::uint32_t j = 0; j < KT_DIM; j++)
        {
            _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(0, CT_DIM, RT_DIM);
        }
        dbg(DBG_M, (last << 4) | 0x3);
        _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        dbg(DBG_M, (last << 4) | 0x4);
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
    _llk_pack_init_wrapper_<PackMode::Default, false /* zero_output */>(formats.pack_dst);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // Output-landing sentinel: the packer streams the tile to L1 via TDMA, which is NOT ordered w.r.t.
    // this RISC's stores — so a bare `done=1` races ahead of the pack (host then reads a poisoned/partial
    // tile and races into the next ring => wedge). tensix_sync (pc_buf) deadlocks mid-loop here. Instead,
    // since a TRISC L1 read observes external writes (the doorbell proves it), spin until the LAST packed
    // word actually changes from the host's poison — a RISC-visible proof the tile has landed. Only then
    // publish done, so the host never (re)stages operands + rings until ring N is fully committed.
    static constexpr std::uint32_t OUT_WORDS = is_fp32_dest_acc_en ? 1024u : 512u;
    static constexpr std::uint32_t POISON    = 0xBADF00D5u;
    volatile std::uint32_t* out_last = reinterpret_cast<volatile std::uint32_t*>(PERF_OUTPUT + (OUT_WORDS - 1) * 4);
    dbg(DBG_P, 0xE0);   // reached loop (INIT done)
    std::uint32_t last = 0;
    std::uint32_t beats = 0;
    for (;;)
    {
        last = wait_ring(last);
        dbg(DBG_P, (last << 4) | 0x1);
        _llk_packer_wait_for_math_done_();
        dbg(DBG_P, (last << 4) | 0x2);
        for (std::uint32_t tile = 0; tile < CT_DIM * RT_DIM; tile++)
        {
            const std::uint32_t tile_index = tile % MAX_TILES_DEST;
            _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(tile_index, PERF_ADDRESS(PERF_OUTPUT, tile_index));
        }
        dbg(DBG_P, (last << 4) | 0x3);
        _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
        dbg(DBG_P, (last << 4) | 0x4);
        do
        {
            ckernel::invalidate_data_cache();
        } while (out_last[0] == POISON); // wait for the pack TDMA to land the tile in L1
        dbg(DBG_P, (last << 4) | 0x5);
        publish(RESIDENT_HB, ++beats);
        publish(RESIDENT_DONE, last);
    }
}

#endif
