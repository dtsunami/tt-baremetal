// SPDX-License-Identifier: Apache-2.0
//
// resident_backward_perf — the FUSED RESIDENT backward of the Gaussian-splat render: splat.backward_ondevice's
// whole 17-stage chain run resident in ONE doorbell ring, staging every intermediate through L1. Built on
// the proven mechanisms: matmul (R1/forward), matmul->SFPU reconfig (forward), matmul<->eltwise-binary
// mode-switch (resident_mm_elw_perf), inter-stage L1 dataflow + tensix_sync landing (R2/forward). Data-
// driven by a compile-time STAGE TABLE so the three threads share one dispatch; each stage is a matmul (M),
// matmul+SFPU-reciprocal (R), FPU eltwise-multiply (X, HiFi4), or FPU eltwise-subtract (S, LoFi).
//
// Chain (per group; leaf grads dLdpsi/dLdop/dLdcolor in depth-sorted space), all 32x32 zero-padded tiles:
//   dw=dLdC@colorᵀ · dwW=dw·w · suf=dwW@U · recA=1/α · Tv=w·recA · oneMα=1−α · recOM=1/oneMα ·
//   t1=dw·Tv · t2=suf·recOM · dLda=t1−t2 · dae=dLda·ar · dLdop=Σₚdae · dLdE=dae·op · dLdVsq=dLdE@Ppairᵀ ·
//   dLdV=dLdVsq·V · dLdpsi=(2phi)ᵀ@dLdV · dLdcolor=wᵀ@dLdC.
// Reciprocal loads its L1 operand into DEST via a matmul-by-identity, then SFPU (the forward's fused idiom).
// α and V are host-staged here (standalone verify vs backward_ondevice); the fused fwd+bwd kernel recomputes
// them from the forward's live ar/psi so nothing extra crosses the seam.

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
static constexpr std::uint32_t FLAG_BASE = 0x16060;      // FLAG(i) = FLAG_BASE + i*4, i = 0..16

// ---- host-staged inputs (fixed L1 addrs; 0x1000 apart) ----
static constexpr std::uint32_t H_dLdC=0x21000, H_w=0x22000, H_alpha=0x23000, H_ar=0x24000, H_v=0x25000,
    H_colorT=0x26000, H_PpairT=0x27000, H_U=0x28000, H_phi2T=0x29000, H_opB=0x2A000, H_ones=0x2B000,
    H_wT=0x2C000, H_ones1P=0x2D000, H_Iden=0x2E000;
// ---- scratch (produced) ----
static constexpr std::uint32_t S_dw=0x40000, S_dwW=0x41000, S_suf=0x42000, S_recA=0x43000, S_Tv=0x44000,
    S_oneMa=0x45000, S_recOM=0x46000, S_t1=0x47000, S_t2=0x48000, S_dLda=0x49000, S_dae=0x4A000,
    S_dLdE=0x4B000, S_dLdVsq=0x4C000, S_dLdV=0x4D000;
// ---- outputs ----
static constexpr std::uint32_t O_dLdop=0x51000, O_dLdpsi=0x52000, O_dLdcol=0x53000;

// op codes
static constexpr std::uint8_t M=0, R=1, X=2, S=3;
static constexpr int NST = 17;
static constexpr std::uint32_t ST_A[NST]  = {H_dLdC,S_dw,S_dwW,H_alpha,H_w,H_ones,S_oneMa,S_dw,S_suf,S_t1,S_dLda,H_ones1P,S_dae,S_dLdE,S_dLdVsq,H_phi2T,H_wT};
static constexpr std::uint32_t ST_B[NST]  = {H_colorT,H_w,H_U,H_Iden,S_recA,H_alpha,H_Iden,S_Tv,S_recOM,S_t2,H_ar,S_dae,H_opB,H_PpairT,H_v,S_dLdV,H_dLdC};
static constexpr std::uint32_t ST_O[NST]  = {S_dw,S_dwW,S_suf,S_recA,S_Tv,S_oneMa,S_recOM,S_t1,S_t2,S_dLda,S_dae,O_dLdop,S_dLdE,S_dLdVsq,S_dLdV,O_dLdpsi,O_dLdcol};
static constexpr std::uint8_t  ST_OP[NST] = {M, X, M, R, X, S, R, X, X, S, X, M, X, M, X, M, M};
static constexpr int ST_W0[NST] = {-1,0,1,-1,3,-1,5,0,2,7,9,10,10,12,13,14,-1};
static constexpr int ST_W1[NST] = {-1,-1,-1,-1,-1,-1,-1,4,6,8,-1,-1,-1,-1,-1,-1,-1};

static inline void dbg(std::uint32_t a, std::uint32_t v)
{ *reinterpret_cast<volatile std::uint32_t*>(a) = v; ckernel::invalidate_data_cache(); }
static inline void publish(std::uint32_t a, std::uint32_t v)
{ *reinterpret_cast<volatile std::uint32_t*>(a) = v; ckernel::invalidate_data_cache(); }
static inline std::uint32_t wait_ring(std::uint32_t last)
{ volatile std::uint32_t* db = reinterpret_cast<volatile std::uint32_t*>(RESIDENT_DB);
  std::uint32_t r; do { ckernel::invalidate_data_cache(); r = db[0]; } while (r == last); return r; }
static inline void wait_flag(int idx, std::uint32_t v)
{ if (idx < 0) return; volatile std::uint32_t* f = reinterpret_cast<volatile std::uint32_t*>(FLAG_BASE + idx * 4);
  do { ckernel::invalidate_data_cache(); } while (f[0] != v); }

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_AB.h"
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

    std::uint32_t last = 0;
    for (;;)
    {
        last = wait_ring(last);
        for (int i = 0; i < NST; i++)
        {
            wait_flag(ST_W0[i], last);
            wait_flag(ST_W1[i], last);
            if (ST_OP[i] == M || ST_OP[i] == R)
            {
                _llk_unpack_AB_matmul_init_<>(TR, CT, RT, KT, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES, false, false);
                _llk_unpack_AB_matmul_<>(PERF_ADDRESS(ST_A[i], 0), PERF_ADDRESS(ST_B[i], 0), 0, 0, TSA, TSB, false, false, CT, RT, KT);
            }
            else
            {
                _llk_unpack_AB_init_<>(DEFAULT_TENSOR_SHAPE);
                _llk_unpack_AB_<>(PERF_ADDRESS(ST_A[i], 0), PERF_ADDRESS(ST_B[i], 0));
            }
        }
        dbg(DBG_U, last);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"
#include "llk_math_eltwise_binary.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu_operations.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;
    std::uint32_t CT = params.CT_DIM, RT = params.RT_DIM, KT = params.KT_DIM;

    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();

    std::uint32_t last = 0;
    for (;;)
    {
        last = wait_ring(last);
        for (int i = 0; i < NST; i++)
        {
            const std::uint8_t op = ST_OP[i];
            if (op == M || op == R)
            {
                _llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>(TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, false, CT, RT);
                _llk_math_wait_for_dest_available_<dest_sync>();
                for (std::uint32_t j = 0; j < KT; j++)
                    _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(0, CT, RT);
                if (op == R)
                {
                    test_utils::call_unary_sfpu_operation_init<SfpuType::reciprocal, APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS, FAST_MODE, STABLE_SORT>();
                    test_utils::call_unary_sfpu_operation<dest_sync, is_fp32_dest_acc_en, SfpuType::reciprocal, APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS, FAST_MODE, STABLE_SORT>(0, formats.math);
                }
                _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
            else if (op == X)   // eltwise multiply (HiFi4)
            {
                _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, MathFidelity::HiFi4>(DEFAULT_TENSOR_SHAPE, 0);
                _llk_math_wait_for_dest_available_<dest_sync>();
                _llk_math_eltwise_binary_<EltwiseBinaryType::ELWMUL, BroadcastType::NONE, dest_sync, is_fp32_dest_acc_en, MathFidelity::HiFi4>(DEFAULT_TENSOR_SHAPE, 0, false);
                _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
            else                // eltwise subtract (LoFi)
            {
                _llk_math_eltwise_binary_init_<EltwiseBinaryType::ELWSUB, BroadcastType::NONE, MathFidelity::LoFi>(DEFAULT_TENSOR_SHAPE, 0);
                _llk_math_wait_for_dest_available_<dest_sync>();
                _llk_math_eltwise_binary_<EltwiseBinaryType::ELWSUB, BroadcastType::NONE, dest_sync, is_fp32_dest_acc_en, MathFidelity::LoFi>(DEFAULT_TENSOR_SHAPE, 0, false);
                _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        dbg(DBG_M, last);
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
        for (int i = 0; i < NST; i++)
        {
            _llk_packer_wait_for_math_done_();
            for (std::uint32_t t = 0; t < CT * RT; t++)
                _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(t % MAX_TILES_DEST, PERF_ADDRESS(ST_O[i], t % MAX_TILES_DEST));
            _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            ckernel::tensix_sync();          // landing barrier
            publish(FLAG_BASE + i * 4, last);
        }
        dbg(DBG_P, last);
        publish(RESIDENT_HB, ++beats);
        publish(RESIDENT_DONE, last);
    }
}

#endif
