// SPDX-License-Identifier: Apache-2.0
//
// bootloader_compute.cpp (compute kernel) — the RESIDENT loader for the THREE COMPUTE RISCs
// (TRISC0/1/2 = UNPACK/MATH/PACK). One CreateKernel(ComputeConfig) compiles this source three
// times (once per TRISC, with COMPILE_FOR_TRISC = 0/1/2) and runs it on all three. Each
// instance derives its OWN disjoint L1 region from COMPILE_FOR_TRISC, so the three TRISCs
// poll three independent control mailboxes — siblings of the BRISC/NCRISC DM bootloader
// (kernels/bootloader.cpp). Region index = 2 + COMPILE_FOR_TRISC (BRISC=0, NCRISC=1).
//
// We only need raw L1 pointer access + a couple of MMIO regs; the compute API is included so
// the JIT treats this as a proper compute kernel (the firmware inits the TRISC backend before
// calling kernel_main, exactly as in the hello_world_compute example). ABI constants are
// duplicated (kept in sync with ../bootloader_abi.h) — JIT'd kernels can't rely on the
// example dir being on the include path.

#include "api/compute/compute_kernel_api.h"
#include <cstdint>

#ifndef COMPILE_FOR_TRISC
#define COMPILE_FOR_TRISC 0
#endif

// ---- per-RISC region geometry (mirror bootloader_abi.h) ----
constexpr uint32_t BL_REGION_BASE   = 0x00100000u;
constexpr uint32_t BL_REGION_STRIDE = 0x00010000u;
// word indices off the region's CTRL base
enum { DOORBELL=0, ARG0=1, ARG1=2, PARAM0=4, HEARTBEAT=16, LAST_CMD=17, STATUS=18,
       OVL_RET=19, WALLCLK=20, RISC_ID=21 };
constexpr uint32_t WALL_CLOCK_L = 0xFFB121F0u;
constexpr uint32_t CMD_NONE=0x0u, CMD_SETPARAM=0x1u, CMD_EXEC=0x2u, CMD_HALT=0xDEADu;
constexpr uint32_t ST_IDLE=1u, ST_OVERLAY=2u, ST_HALTED=3u;
constexpr uint32_t BL_CFG_BASE = 0xFFEF0000u;   // == TENSIX_CFG_BASE (uniquely named: see DM kernel)

using overlay_fn = uint32_t (*)(volatile uint32_t*);

static inline __attribute__((always_inline)) void flush_icache() {
    volatile uint32_t* cfg = (volatile uint32_t*)BL_CFG_BASE;
    cfg[185] = 0x1f;   // RISCV_IC_INVALIDATE.InvalidateAll, all 5 RISCs
    __asm__ volatile("" ::: "memory");
#pragma GCC unroll 16
    for (int i = 0; i < 16; i++) {
        __asm__ volatile("nop");
    }
}

void kernel_main() {
    // This TRISC's region: BRISC=0, NCRISC=1, then TRISC0/1/2 = 2/3/4.
    const uint32_t risc_id   = 2u + (uint32_t)COMPILE_FOR_TRISC;
    const uint32_t ctrl_base = BL_REGION_BASE + risc_id * BL_REGION_STRIDE;
    volatile uint32_t* ctrl  = (volatile uint32_t*)ctrl_base;

    ctrl[DOORBELL]  = CMD_NONE;
    ctrl[STATUS]    = ST_IDLE;
    ctrl[HEARTBEAT] = 0;
    ctrl[RISC_ID]   = risc_id;

    uint32_t iter = 0;
    for (;;) {
        uint32_t cmd = ctrl[DOORBELL];

        if (cmd == CMD_HALT) {
            ctrl[LAST_CMD] = cmd;
            ctrl[STATUS]   = ST_HALTED;
            ctrl[DOORBELL] = CMD_NONE;     // ack
            return;
        }
        if (cmd == CMD_EXEC) {
            uint32_t slot = ctrl[ARG0];
            flush_icache();
            ctrl[STATUS] = ST_OVERLAY;
            overlay_fn fn = (overlay_fn)slot;
            uint32_t ret = fn((volatile uint32_t*)ctrl);
            ctrl[OVL_RET]  = ret;
            ctrl[STATUS]   = ST_IDLE;
            ctrl[LAST_CMD] = cmd;
            ctrl[DOORBELL] = CMD_NONE;
        } else if (cmd == CMD_SETPARAM) {
            ctrl[LAST_CMD] = cmd;
            ctrl[DOORBELL] = CMD_NONE;
        }

        if ((++iter & 0xFFFFFu) == 0) {
            ctrl[HEARTBEAT] = iter;
            ctrl[WALLCLK]   = *(volatile uint32_t*)WALL_CLOCK_L;
        }
    }
}
