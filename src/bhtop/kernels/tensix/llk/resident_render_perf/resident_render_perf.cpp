// SPDX-License-Identifier: Apache-2.0
//
// resident_render_perf — the FUSED, RESIDENT Gaussian-splat forward render: one Tensix worker runs the
// whole `splat.render_ondevice` pipeline (6 MVMUL + 5 SFPU) in ONE doorbell ring, staging every
// intermediate through L1, no host round-trip between ops. Built on the proven residency mechanisms
// (RESIDENT_GRID.md): R1 3-thread doorbell loop + R2 inter-stage pack->L1->unpack dataflow + the
// fused_mm_sq MVMUL->SFPU-in-DEST reconfig. Host stages the per-tile operands once, rings once per
// pixel-group; drive it across all 120 workers (R3) for the parallel render grid.
//
// The 11 render stages collapse to 6 FUSED super-stages by doing each SFPU op in DEST right after its
// matmul, and folding the [la|lpa]@Mcomb concatenation into a KT=2 accumulating matmul (la@Stri+lpa@I):
//   F1: Vsq = square(phi @ psi)           F2: ar  = exp(Vsq @ Ppair)
//   F3: lpa = log(ar @ Dop)               F4: la  = log1p(ar @ Dnop)
//   F5: w   = exp(la@Stri + lpa@I)  (KT=2) F6: C   = w @ color
// Each super-stage = unpack -> math(matmul[,matmul] + SFPU) -> pack to an L1 scratch -> publish a stage
// flag; the next stage's unpack waits on the flags of the scratch tiles it reads (pack->unpack sync).
// All tiles are 32x32, host zero-padded ([P pixels] x [cols]); the padding makes each a full 32x32 mm.
//
// L1 map (bf16 tiles, 2 KB): inputs phi 0x21000 (per-group) psi 0x31000 | consts Ppair 0x60000
//   Dop 0x60800 Dnop 0x61000 Stri 0x61800 Iden 0x62000 color 0x62800 | scratch Vsq 0x40000 ar 0x40800
//   lpa 0x41000 la 0x41800 w 0x42000 | out C 0x51000. Doorbell DB 0x16000 DONE 0x16010 HB 0x16020,
//   per-stage flags FLAG(i)=0x16060+(i-1)*4 (i=1..5; stage 6 publishes DONE), DBG_U/M/P 0x16030/40/50.

#include <algorithm>
#include <cstdint>

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
static constexpr std::uint32_t FLAG_BASE     = 0x16060;   // FLAG(i) = FLAG_BASE + (i-1)*4, i=1..5

// operand + scratch tiles (bf16, 0x800 bytes each)
static constexpr std::uint32_t A_PHI   = PERF_INPUT_A;    // 0x21000 (per pixel-group)
static constexpr std::uint32_t A_PSI   = PERF_INPUT_B;    // 0x31000
static constexpr std::uint32_t A_PPAIR = 0x60000;
static constexpr std::uint32_t A_DOP   = 0x60800;
static constexpr std::uint32_t A_DNOP  = 0x61000;
static constexpr std::uint32_t A_STRI  = 0x61800;
static constexpr std::uint32_t A_IDEN  = 0x62000;
static constexpr std::uint32_t A_COLOR = 0x62800;
static constexpr std::uint32_t S_VSQ   = 0x40000;
static constexpr std::uint32_t S_AR    = 0x40800;
static constexpr std::uint32_t S_LPA   = 0x41000;
static constexpr std::uint32_t S_LA    = 0x41800;
static constexpr std::uint32_t S_W     = 0x42000;
static constexpr std::uint32_t OUT_C   = PERF_OUTPUT;     // 0x51000 (group 0; group g at OUT_C + g*STRIDE)
static constexpr std::uint32_t POISON  = 0xBADF00D5u;
static constexpr std::uint32_t OUT_WORDS = 512u;          // packer always emits bf16 (2 B/elem)
static constexpr std::uint32_t TELEM    = 0x16080;        // pack-thread cycle stamps: [ring-start, F1..F6]
static constexpr std::uint32_t NG_ADDR  = 0x160A0;        // host-set pixel-group count (one ring = whole tile)
static constexpr std::uint32_t STRIDE   = 0x800;          // per-group stride for phi/out (bf16 32x32 = 2 KB)
// flag value for (ring, group g): monotone across both since g < 32 (a 16x16 tile = 8 groups).
static inline std::uint32_t fval(std::uint32_t ring, std::uint32_t g) { return ring * 32u + g; }
static inline std::uint32_t read_ng()
{
    ckernel::invalidate_data_cache();
    std::uint32_t n = *reinterpret_cast<volatile std::uint32_t*>(NG_ADDR);
    return (n == 0u || n > 32u) ? 1u : n;
}

static inline void dbg(std::uint32_t addr, std::uint32_t v)
{ *reinterpret_cast<volatile std::uint32_t*>(addr) = v; ckernel::invalidate_data_cache(); }
static inline void publish(std::uint32_t addr, std::uint32_t v)
{ *reinterpret_cast<volatile std::uint32_t*>(addr) = v; ckernel::invalidate_data_cache(); }
static inline std::uint32_t wait_ring(std::uint32_t last)
{
    volatile std::uint32_t* db = reinterpret_cast<volatile std::uint32_t*>(RESIDENT_DB);
    std::uint32_t r; do { ckernel::invalidate_data_cache(); r = db[0]; } while (r == last); return r;
}
static inline void wait_flag(std::uint32_t i, std::uint32_t val)
{
    volatile std::uint32_t* f = reinterpret_cast<volatile std::uint32_t*>(FLAG_BASE + (i - 1) * 4);
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

// unpack one A@B pair (single 32x32 tiles at explicit addresses).
static inline void unp(std::uint32_t a, std::uint32_t b, std::uint32_t tsa, std::uint32_t tsb)
{
    _llk_unpack_AB_matmul_<>(PERF_ADDRESS(a, 0), PERF_ADDRESS(b, 0), 0, 0, tsa, tsb, false, false, 1, 1, 1);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;
    const std::uint32_t TSA = params.TILE_SIZE_UNPACK_A;
    const std::uint32_t TSB = params.TILE_SIZE_UNPACK_B;
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
        const std::uint32_t NG = read_ng();
        for (std::uint32_t g = 0; g < NG; g++)               // one ring = a whole tile (NG pixel-groups)
        {
            const std::uint32_t v = fval(last, g);
            const std::uint32_t phig = A_PHI + g * STRIDE;
            unp(phig, A_PSI, TSA, TSB);                       // F1: Vsq = square(phi[g] @ psi)
            wait_flag(1, v); unp(S_VSQ, A_PPAIR, TSA, TSB);   // F2: ar = exp(Vsq @ Ppair)
            wait_flag(2, v); unp(S_AR, A_DOP, TSA, TSB);      // F3: lpa = log(ar @ Dop)
            unp(S_AR, A_DNOP, TSA, TSB);                      // F4: la = log1p(ar @ Dnop) (ar still live)
            wait_flag(3, v); wait_flag(4, v);
            unp(S_LA, A_STRI, TSA, TSB); unp(S_LPA, A_IDEN, TSA, TSB);  // F5: w=exp(la@Stri+lpa@I) (KT=2)
            wait_flag(5, v); unp(S_W, A_COLOR, TSA, TSB);     // F6: C = w @ color
        }
        dbg(DBG_U, last);
    }
}

#endif

#ifdef LLK_TRISC_MATH

#include "llk_math_common.h"
#include "llk_math_matmul.h"
#include "llk_math_eltwise_unary_sfpu.h"
#include "sfpu_operations.h"

#define SFPU_DO(OP)                                                                                       \
    do {                                                                                                  \
        test_utils::call_unary_sfpu_operation_init<SfpuType::OP, APPROX_MODE, is_fp32_dest_acc_en,        \
            ITERATIONS, FAST_MODE, STABLE_SORT>();                                                        \
        test_utils::call_unary_sfpu_operation<dest_sync, is_fp32_dest_acc_en, SfpuType::OP, APPROX_MODE,  \
            is_fp32_dest_acc_en, ITERATIONS, FAST_MODE, STABLE_SORT>(0, formats.math);                    \
    } while (0)

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;
    std::uint32_t CT = params.CT_DIM, RT = params.RT_DIM;

    _llk_math_hw_configure_<is_fp32_dest_acc_en>(formats.math, formats.math);
    _llk_math_pack_sync_init_<dest_sync, is_fp32_dest_acc_en>();
    _llk_math_matmul_init_<MATH_FIDELITY, THROTTLE_LEVEL>(TILE_R_DIM, TILE_C_DIM, TILE_R_DIM, TILE_C_DIM, false, false, CT, RT);

    std::uint32_t last = 0;
    for (;;)
    {
        last = wait_ring(last);
        const std::uint32_t NG = read_ng();
        for (std::uint32_t g = 0; g < NG; g++)
        {
            // F1: matmul(phi,psi) -> square
            _llk_math_wait_for_dest_available_<dest_sync>();
            _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(0, CT, RT);
            SFPU_DO(square);
            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            // F2: matmul(Vsq,Ppair) -> exp
            _llk_math_wait_for_dest_available_<dest_sync>();
            _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(0, CT, RT);
            SFPU_DO(exponential);
            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            // F3: matmul(ar,Dop) -> log
            _llk_math_wait_for_dest_available_<dest_sync>();
            _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(0, CT, RT);
            SFPU_DO(log);
            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            // F4: matmul(ar,Dnop) -> log1p
            _llk_math_wait_for_dest_available_<dest_sync>();
            _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(0, CT, RT);
            SFPU_DO(log1p);
            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            // F5: matmul(la,Stri) + matmul(lpa,Iden) accumulate -> exp
            _llk_math_wait_for_dest_available_<dest_sync>();
            _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(0, CT, RT);
            _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(0, CT, RT);
            SFPU_DO(exponential);
            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
            // F6: matmul(w,color)  (no SFPU)
            _llk_math_wait_for_dest_available_<dest_sync>();
            _llk_math_matmul_<MATH_FIDELITY, THROTTLE_LEVEL>(0, CT, RT);
            _llk_math_dest_section_done_<dest_sync, is_fp32_dest_acc_en>();
        }
        dbg(DBG_M, last);
    }
}

#endif

#ifdef LLK_TRISC_PACK

#include "llk_lib_pack_wrappers.h"
#include "llk_pack.h"
#include "llk_pack_common.h"

// pack the current dest section to `out`, wait for it to land, then publish a flag (a stage flag i>0,
// or DONE when i==0).
static inline void pack_stage(std::uint32_t out, std::uint32_t flag_i, std::uint32_t fv,
                              std::uint32_t ct, std::uint32_t rt)
{
    _llk_packer_wait_for_math_done_();
    for (std::uint32_t tile = 0; tile < ct * rt; tile++)
        _llk_pack_<DstSync::SyncHalf, is_fp32_dest_acc_en>(tile % MAX_TILES_DEST, PERF_ADDRESS(out, tile % MAX_TILES_DEST));
    _llk_pack_dest_section_done_<DstSync::SyncHalf, is_fp32_dest_acc_en>();
    // Landing barrier: _llk_pack_dest_section_done_ issues a TTI_STALLWAIT(STALL_MATH,PACK) that blocks
    // the Tensix backend until the pack TDMA completes; tensix_sync drains the RISC's instruction pipe
    // so that STALLWAIT has retired => the tile is in L1 before we publish the flag. (Earlier this looked
    // like a deadlock, but that was the LOOP_FACTOR/num_faces hang stalling the whole pipe upstream.)
    ckernel::tensix_sync();
    if (flag_i)
        publish(FLAG_BASE + (flag_i - 1) * 4, fv);
}

void run_kernel(RUNTIME_PARAMETERS params)
{
    const FormatConfig& formats = params.formats;
    std::uint32_t CT = params.CT_DIM, RT = params.RT_DIM;

    _llk_pack_hw_configure_<is_fp32_dest_acc_en, ckernel::PackMode::Default>(formats.pack_src, formats.pack_dst, TILE_C_DIM * TILE_R_DIM);
    _llk_pack_init_wrapper_<PackMode::Default, false>(formats.pack_dst);
    _llk_pack_dest_init_<DstSync::SyncHalf, is_fp32_dest_acc_en>();

    // per-stage cycle telemetry (wall clock lo): [ring-start, F1..F6]. Plain stores (no per-stamp
    // fence, so the measurement isn't perturbed); one fence at the end flushes them for the host.
    volatile std::uint32_t* tel = reinterpret_cast<volatile std::uint32_t*>(TELEM);
    std::uint32_t last = 0, beats = 0;
    for (;;)
    {
        last = wait_ring(last);
        const std::uint32_t NG = read_ng();
        // telem: tel[0]=tile start, tel[1..6]=group-0 stage ends (per-stage breakdown), tel[7]=tile end.
        tel[0] = static_cast<std::uint32_t>(ckernel::read_wall_clock());
        for (std::uint32_t g = 0; g < NG; g++)
        {
            const std::uint32_t v = fval(last, g);
            const std::uint32_t outg = OUT_C + g * STRIDE;
            pack_stage(S_VSQ, 1, v, CT, RT); if (g == 0) tel[1] = static_cast<std::uint32_t>(ckernel::read_wall_clock());
            pack_stage(S_AR,  2, v, CT, RT); if (g == 0) tel[2] = static_cast<std::uint32_t>(ckernel::read_wall_clock());
            pack_stage(S_LPA, 3, v, CT, RT); if (g == 0) tel[3] = static_cast<std::uint32_t>(ckernel::read_wall_clock());
            pack_stage(S_LA,  4, v, CT, RT); if (g == 0) tel[4] = static_cast<std::uint32_t>(ckernel::read_wall_clock());
            pack_stage(S_W,   5, v, CT, RT); if (g == 0) tel[5] = static_cast<std::uint32_t>(ckernel::read_wall_clock());
            pack_stage(outg,  0, v, CT, RT); if (g == 0) tel[6] = static_cast<std::uint32_t>(ckernel::read_wall_clock());
        }
        tel[7] = static_cast<std::uint32_t>(ckernel::read_wall_clock());   // whole-tile end
        ckernel::invalidate_data_cache();       // flush telemetry to L1 for the host
        dbg(DBG_P, last);
        publish(RESIDENT_HB, ++beats);
        publish(RESIDENT_DONE, last);
    }
}

#endif
