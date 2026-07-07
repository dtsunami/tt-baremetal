// SPDX-License-Identifier: Apache-2.0
//
// resident_train_perf — the FULL FUSED RESIDENT training step: forward render + loss gradient + backward,
// the whole thing in ONE doorbell ring per tile, every intermediate living in L1 (nothing crosses back to
// the host between forward and backward). One worker = one training-step tile; the 120-worker grid does a
// batch in parallel. This unifies the two proven halves (resident_render_perf + resident_backward_perf)
// plus a 3-stage seam, so the forward's w/ar feed the backward directly on-device.
//
// 27 stages, one shared compile-time STAGE TABLE dispatched by op:
//   FORWARD (6, matmul+SFPU-in-DEST): Vsq=sq(phi@psi) · ar=exp(Vsq@Ppair) · lpa=log(ar@Dop) ·
//     la=log1p(ar@Dnop) · w=exp(la@Stri+lpa@I)[KT=2] · C=w@color
//   SEAM (4): dLdC=C−gt (on-device loss grad) · alpha=ar⊙opB (recompute) · V=phi@psi (recompute) ·
//     wT=transpose(w) (for dLdcolor; datacopy+transpose_dest = a true 32×32 transpose)
//   BACKWARD (17): dw=dLdC@colorᵀ · dwW=dw·w · suf=dwW@U · recA=1/α · Tv=w·recA · oneMα=1−α ·
//     recOM=1/oneMα · t1=dw·Tv · t2=suf·recOM · dLda=t1−t2 · dae=dLda·ar · dLdop=Σₚdae · dLdE=dae·op ·
//     dLdVsq=dLdE@Ppairᵀ · dLdV=dLdVsq·V · dLdpsi=(2phi)ᵀ@dLdV · dLdcolor=wᵀ@dLdC
// Outputs: RGB C + leaf grads dLdop/dLdpsi/dLdcolor. Every mechanism proven on silicon (R1/R2/render/
// backward/mode-switch/transpose); this is their assembly.

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
static constexpr std::uint32_t FLAG_BASE = 0x16060;      // FLAG(i) = FLAG_BASE + i*4, i = 0..26
// --- telemetry (past the flag scoreboard @ 0x160C8): liveness + per-ring cycles ---
// MK/PK/UK_PH = (ring<<16)|(stage<<8)|sub  per-thread sub-phase.  SEM_M = live MATH_PACK value.
// T_RING/T_END = math wall-clock (lo32) at ring start/end -> ring cycles.  T_CFG = ring-boundary re-init cycles.
static constexpr std::uint32_t MK_PH=0x16100, SEM_M=0x16104, PK_PH=0x16108, UK_PH=0x1610C,
    T_RING=0x16110, T_END=0x16114, T_CFG=0x16118;

// host-staged inputs
static constexpr std::uint32_t H_phi=0x21000, H_psi=0x22000, H_Ppair=0x23000, H_Dop=0x24000, H_Dnop=0x25000,
    H_Stri=0x26000, H_Iden=0x27000, H_color=0x28000, H_gt=0x29000, H_opB=0x2A000, H_colorT=0x2B000,
    H_PpairT=0x2C000, H_U=0x2D000, H_phi2T=0x2E000, H_ones=0x2F000, H_ones1P=0x30000;
// forward scratch
static constexpr std::uint32_t S_Vsq=0x40000, S_ar=0x40800, S_lpa=0x41000, S_la=0x41800, S_w=0x42000, S_C=0x42800;
// seam scratch
static constexpr std::uint32_t S_dLdC=0x43000, S_alpha=0x43800, S_v=0x44000, S_wT=0x44800;
// backward scratch
static constexpr std::uint32_t S_dw=0x45000, S_dwW=0x45800, S_suf=0x46000, S_recA=0x46800, S_Tv=0x47000,
    S_oneMa=0x47800, S_recOM=0x48000, S_t1=0x48800, S_t2=0x49000, S_dLda=0x49800, S_dae=0x4A000,
    S_dLdE=0x4A800, S_dLdVsq=0x4B000, S_dLdV=0x4B800;
// outputs
static constexpr std::uint32_t O_C=0x50000, O_dLdop=0x51000, O_dLdpsi=0x52000, O_dLdcol=0x53000;

// op codes
static constexpr std::uint8_t MM_=0, MSQ=1, MEXP=2, MLOG=3, ML1P=4, MREC=5, EMUL=6, ESUB=7, TRN=8;
static constexpr int NST = 27;
// Stage order: fwd(0-5) · seam(6-9: dLdC, alpha, v, wT) · bwd(10-26). The wT transpose (stage 9) is done
// as matmul(w@Iden) THEN transpose_dest — the matmul consumes srcB (Iden) so the srcB semaphore stays
// balanced (a datacopy+dummy_valid transpose leaves srcB +1 and deadlocks a later eltwise).
//                                          0    1     2    3    4(KT2) 5    6    7    8    9(T)  bwd 10..26
static constexpr std::uint32_t A_[NST]  = {H_phi,S_Vsq,S_ar,S_ar,S_la,S_w,S_C,S_ar,H_phi,S_w,
    S_dLdC,S_dw,S_dwW,S_alpha,S_w,H_ones,S_oneMa,S_dw,S_suf,S_t1,S_dLda,H_ones1P,S_dae,S_dLdE,S_dLdVsq,H_phi2T,S_wT};
static constexpr std::uint32_t B_[NST]  = {H_psi,H_Ppair,H_Dop,H_Dnop,H_Stri,H_color,H_gt,H_opB,H_psi,H_Iden,
    H_colorT,S_w,H_U,H_Iden,S_recA,S_alpha,H_Iden,S_Tv,S_recOM,S_t2,S_ar,S_dae,H_opB,H_PpairT,S_v,S_dLdV,S_dLdC};
static constexpr std::uint32_t A2_[NST] = {0,0,0,0,S_lpa,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
static constexpr std::uint32_t B2_[NST] = {0,0,0,0,H_Iden,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
static constexpr std::uint32_t O_[NST]  = {S_Vsq,S_ar,S_lpa,S_la,S_w,S_C,S_dLdC,S_alpha,S_v,S_wT,
    S_dw,S_dwW,S_suf,S_recA,S_Tv,S_oneMa,S_recOM,S_t1,S_t2,S_dLda,S_dae,O_dLdop,S_dLdE,S_dLdVsq,S_dLdV,O_dLdpsi,O_dLdcol};
// stage 9 is a plain matmul now: the transpose_dest primitive leaves dest-addressing state that
// deadlocks the next eltwise, so dLdcolor (= wᵀ@dLdC, stage 26) is delegated to the x280 param-server
// (it already holds w + dLdC); this kernel outputs the two hard grads dLdpsi (geometry) + dLdop (opacity).
static constexpr std::uint8_t OP_[NST]  = {MSQ,MEXP,MLOG,ML1P,MEXP,MM_,ESUB,EMUL,MM_,MM_,
    MM_,EMUL,MM_,MREC,EMUL,ESUB,MREC,EMUL,EMUL,ESUB,EMUL,MM_,EMUL,MM_,EMUL,MM_,MM_};
static constexpr std::uint8_t KT_[NST]  = {1,1,1,1,2,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
static constexpr int W0_[NST] = {-1,0,1,1,3,4,5,1,-1,4, 6,10,11,7,4,7,15,10,12,17,19,20,20,22,23,24,9};
static constexpr int W1_[NST] = {-1,-1,-1,-1,2,-1,-1,-1,-1,-1, -1,4,-1,-1,13,-1,-1,14,16,18,1,-1,-1,-1,8,-1,6};

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
static inline bool is_mm(std::uint8_t op) { return op <= MREC; }

#ifdef LLK_TRISC_UNPACK

#include "llk_unpack_AB_matmul.h"
#include "llk_unpack_AB.h"
#include "llk_unpack_A.h"
#include "llk_unpack_common.h"

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;
    const std::uint32_t TSA = params.TILE_SIZE_UNPACK_A, TSB = params.TILE_SIZE_UNPACK_B;
    const std::uint32_t nfa = params.num_faces_A, nfb = params.num_faces_B;
    std::uint32_t CT = params.CT_DIM, RT = params.RT_DIM, KT = params.KT_DIM;
    const bool TR = params.UNPACK_TRANSPOSE_FACES;
    const std::uint32_t LOOP_FACTOR = params.LOOP_FACTOR; (void)LOOP_FACTOR; (void)KT;

    _llk_unpack_hw_configure_<is_fp32_dest_acc_en>(
        formats.unpack_A_src, formats.unpack_B_src, formats.unpack_A_dst, formats.unpack_B_dst,
        FACE_R_DIM, FACE_R_DIM, nfa, nfb, TSA, TSB);
    // GAP0 FIX: init the AB-matmul unpack ONCE (render pattern); re-init only on eltwise->matmul transition.
    _llk_unpack_AB_matmul_init_<>(TR, CT, RT, 1, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES, false, false);
    bool prev_mm = true;

    std::uint32_t last = 0;
    for (;;)
    {
        last = wait_ring(last);
        for (int i = 0; i < NST; i++)
        {
            dbg(DBG_U, (last << 8) | (std::uint32_t)i);
            dbg(UK_PH, (last << 16) | ((std::uint32_t)i << 8) | 1);
            wait_flag(W0_[i], last);
            wait_flag(W1_[i], last);
            dbg(UK_PH, (last << 16) | ((std::uint32_t)i << 8) | 2);
            const std::uint8_t op = OP_[i];
            if (is_mm(op) || op == TRN)   // TRN = matmul(w@Iden) then transpose_dest; same balanced AB unpack
            {
                if (!prev_mm)   // re-init only on eltwise->matmul transition (render pattern)
                    _llk_unpack_AB_matmul_init_<>(TR, CT, RT, 1, FACE_R_DIM, FACE_R_DIM, TILE_NUM_FACES, TILE_NUM_FACES, false, false);
                _llk_unpack_AB_matmul_<>(PERF_ADDRESS(A_[i], 0), PERF_ADDRESS(B_[i], 0), 0, 0, TSA, TSB, false, false, CT, RT, 1);
                if (KT_[i] == 2)
                    _llk_unpack_AB_matmul_<>(PERF_ADDRESS(A2_[i], 0), PERF_ADDRESS(B2_[i], 0), 0, 0, TSA, TSB, false, false, CT, RT, 1);
                prev_mm = true;
            }
            else   // EMUL / ESUB: matmul-load both operands (A@Iden, B@Iden) — NO unpack mode switch
            {
                _llk_unpack_AB_matmul_<>(PERF_ADDRESS(A_[i], 0), PERF_ADDRESS(H_Iden, 0), 0, 0, TSA, TSB, false, false, CT, RT, 1);
                _llk_unpack_AB_matmul_<>(PERF_ADDRESS(B_[i], 0), PERF_ADDRESS(H_Iden, 0), 0, 0, TSA, TSB, false, false, CT, RT, 1);
                prev_mm = true;
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
#include "llk_math_eltwise_binary_sfpu.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_lib_math_wrappers.h"
#include "sfpu_operations.h"

#define SFPU(OP) do { \
    test_utils::call_unary_sfpu_operation_init<SfpuType::OP, APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS, FAST_MODE, STABLE_SORT>(); \
    test_utils::call_unary_sfpu_operation<dest_sync, is_fp32_dest_acc_en, SfpuType::OP, APPROX_MODE, is_fp32_dest_acc_en, ITERATIONS, FAST_MODE, STABLE_SORT>(0, formats.math); \
} while (0)

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;
    std::uint32_t CT = params.CT_DIM, RT = params.RT_DIM;

    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    // GAP0 FIX: init matmul ONCE like resident_render_perf (which is SFPU-multiring-proven). Re-initing
    // matmul_init every stage — the old code — is what wedged the SFPU dest-read on ring>1; render inits
    // once and re-inits only on a matmul<->eltwise mode switch. prev_mm persists ACROSS rings.
    _llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>(TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, false, CT, RT);
    bool prev_mm = true;

    std::uint32_t last = 0;
    for (;;)
    {
        last = wait_ring(last);
        const std::uint32_t t_ring = (std::uint32_t)read_wall_clock();
        dbg(T_RING, t_ring);
        dbg(T_CFG, (std::uint32_t)read_wall_clock() - t_ring);
        for (int i = 0; i < NST; i++)
        {
            dbg(DBG_M, (last << 8) | (std::uint32_t)i);
            dbg(SEM_M, semaphore_read(semaphore::MATH_PACK));
            dbg(MK_PH, (last << 16) | ((std::uint32_t)i << 8) | 1);
            const std::uint8_t op = OP_[i];
            if (is_mm(op))
            {
                if (!prev_mm)   // re-init only on eltwise->matmul transition (render pattern)
                    _llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>(TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, false, CT, RT);
                _llk_math_wait_for_dest_available_<dest_sync>();
                dbg(MK_PH, (last << 16) | ((std::uint32_t)i << 8) | 2);
                _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(0, CT, RT);
                if (KT_[i] == 2)
                    _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(0, CT, RT);
                if (op == MSQ)  SFPU(square);
                else if (op == MEXP) SFPU(exponential);
                else if (op == MLOG) SFPU(log);
                else if (op == ML1P) SFPU(log1p);
                else if (op == MREC) SFPU(reciprocal);
                _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                prev_mm = true;
            }
            else if (op == EMUL || op == ESUB)
            {
                // GAP0 FIX: eltwise via matmul-load + SFPU-binary (NO FPU eltwise, NO unpack mode switch).
                // A@Iden -> DEST0, B@Iden -> DEST1, SFPU binary(MUL/SUB) -> DEST0. Stays matmul+SFPU (resident).
                _llk_math_wait_for_dest_available_<dest_sync>();
                dbg(MK_PH, (last << 16) | ((std::uint32_t)i << 8) | 2);
                _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(0, CT, RT);   // A -> DEST0
                _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(1, CT, RT);   // B -> DEST1
                if (op == EMUL)
                {
                    test_utils::call_binary_sfpu_operation_init<APPROX_MODE, ckernel::BinaryOp::MUL, ITERATIONS>();
                    test_utils::call_binary_sfpu_operation<dest_sync, is_fp32_dest_acc_en, APPROX_MODE, ckernel::BinaryOp::MUL, ITERATIONS>(0, 1, 0);
                }
                else
                {
                    test_utils::call_binary_sfpu_operation_init<APPROX_MODE, ckernel::BinaryOp::SUB, ITERATIONS>();
                    test_utils::call_binary_sfpu_operation<dest_sync, is_fp32_dest_acc_en, APPROX_MODE, ckernel::BinaryOp::SUB, ITERATIONS>(0, 1, 0);
                }
                _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
                prev_mm = true;   // stayed in matmul-unpack mode
            }
            else   // TRN: matmul(w@Iden) -> DEST=w, then transpose_dest -> DEST=wT (full 32x32 transpose)
            {
                _llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>(TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, false, CT, RT);
                _llk_math_wait_for_dest_available_<dest_sync>();
                _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(0, CT, RT);
                _llk_math_transpose_dest_init_<false, is_fp32_dest_acc_en>();
                _llk_math_transpose_dest_wrapper_<is_fp32_dest_acc_en, false, is_fp32_dest_acc_en>(0);
                _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            }
        }
        dbg(T_END, (std::uint32_t)read_wall_clock() - t_ring);   // whole-ring math cycles
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
            dbg(PK_PH, (last << 16) | ((std::uint32_t)i << 8) | 1);
            _llk_packer_wait_for_math_done_();
            dbg(PK_PH, (last << 16) | ((std::uint32_t)i << 8) | 2);
            for (std::uint32_t t = 0; t < CT * RT; t++)
                _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(t % MAX_TILES_DEST, PERF_ADDRESS(O_[i], t % MAX_TILES_DEST));
            _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
            dbg(PK_PH, (last << 16) | ((std::uint32_t)i << 8) | 4);
            ckernel::tensix_sync();
            publish(FLAG_BASE + i * 4, last);
        }
        dbg(DBG_P, last);
        publish(RESIDENT_HB, ++beats);
        publish(RESIDENT_DONE, last);
    }
}

#endif
