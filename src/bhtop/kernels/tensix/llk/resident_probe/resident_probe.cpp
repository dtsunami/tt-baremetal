// SPDX-License-Identifier: Apache-2.0
//
// resident_probe — the MINIMAL residency isolation test: all three Tensix compute threads run ONLY a
// doorbell-heartbeat loop (no matmul, no LLK compute, no dest-sync). Each thread spins on the host
// ring word and, when it advances, publishes its own per-thread heartbeat. This answers one question:
// can each of the three TRISCs — especially the boot thread T0 (unpack) — observe host writes to an L1
// doorbell in a resident `for(;;)` loop? If all three heartbeats track the ring, the doorbell mechanism
// is sound and any hang in resident_mm_perf lives in the matmul/dest-sync integration, not the doorbell.

#include <cstdint>
#include "ckernel.h"
#include "params.h"

std::uint32_t unp_cfg_context          = 0;
std::uint32_t pack_sync_tile_dst_ptr   = 0;
std::uint32_t math_sync_tile_dst_index = 0;

static constexpr std::uint32_t RESIDENT_DB   = 0x16000;
static constexpr std::uint32_t RESIDENT_DONE = 0x16010;
static constexpr std::uint32_t DBG_U         = 0x16030;
static constexpr std::uint32_t DBG_M         = 0x16040;
static constexpr std::uint32_t DBG_P         = 0x16050;

static inline void publish(std::uint32_t addr, std::uint32_t v)
{
    *reinterpret_cast<volatile std::uint32_t*>(addr) = v;
    ckernel::invalidate_data_cache();
}

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

static inline void heartbeat_loop(std::uint32_t dbg_addr, bool is_pack)
{
    std::uint32_t last = 0, c = 0;
    for (;;)
    {
        last = wait_ring(last);
        publish(dbg_addr, (last << 8) | (++c & 0xFF));
        if (is_pack)
        {
            publish(RESIDENT_DONE, last);
        }
    }
}

#ifdef LLK_TRISC_UNPACK
void run_kernel(RUNTIME_PARAMETERS params) { (void)params; heartbeat_loop(DBG_U, false); }
#endif
#ifdef LLK_TRISC_MATH
void run_kernel(RUNTIME_PARAMETERS params) { (void)params; heartbeat_loop(DBG_M, false); }
#endif
#ifdef LLK_TRISC_PACK
void run_kernel(RUNTIME_PARAMETERS params) { (void)params; heartbeat_loop(DBG_P, true); }
#endif
