// SPDX-License-Identifier: Apache-2.0
//
// bootloader.cpp (DM kernel) — the RESIDENT loader for the DATA-MOVEMENT RISCs (BRISC and
// NCRISC). The launcher creates this kernel once per DM processor and passes each its own
// L1 region base as a runtime arg, so the SAME source is resident on both BRISC and NCRISC,
// each polling a DISJOINT control mailbox. It never returns: it polls the mailbox and, on
// command, stages params or jumps into host-staged machine code in a code slot. This is
// metal's own firmware move (cast an L1 address to a fn ptr and call it, brisc.cc:517)
// generalized into a loop you drive live from bhtop. The TRISC variant is bootloader_compute.cpp.
//
// Compiled by metal's JIT, so get_arg_val<>, tt_l1_ptr, and the dataflow API are injected —
// we only need <cstdint>. ABI constants are duplicated here (kept in sync with
// ../bootloader_abi.h) because JIT'd kernels don't reliably get the example dir on the
// include path.

#include <cstdint>

// word indices off the region's CTRL base (mirror bootloader_abi.h BlCtrl)
enum { DOORBELL=0, ARG0=1, ARG1=2, PARAM0=4, HEARTBEAT=16, LAST_CMD=17, STATUS=18,
       OVL_RET=19, WALLCLK=20, RISC_ID=21 };
constexpr uint32_t WALL_CLOCK_L = 0xFFB121F0u;   // RISCV_DEBUG_REG_WALL_CLOCK_0 (free-running)
// commands
constexpr uint32_t CMD_NONE=0x0u, CMD_SETPARAM=0x1u, CMD_EXEC=0x2u, CMD_HALT=0xDEADu;
// status
constexpr uint32_t ST_IDLE=1u, ST_OVERLAY=2u, ST_HALTED=3u;
// Tensix CFG base — UNIQUELY named to avoid colliding with the TENSIX_CFG_BASE macro that
// the firmware headers already define at JIT time (a same-named constexpr is a hard error).
constexpr uint32_t BL_CFG_BASE = 0xFFEF0000u;

using overlay_fn = uint32_t (*)(volatile uint32_t*);

// Real i-cache invalidate (Blackhole): write CFG register word 185
// (RISCV_IC_INVALIDATE.InvalidateAll), mask 0x1f = {BRISC,TRISC0-2,NCRISC} — the same MMIO store
// firmware does at brisc.cc:239. Writing it from any RISC invalidates all 5 i-caches (a harmless
// refetch for the others). Fallback if insufficient on device: the 3072-NOP flood.
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
    // RTA0 = this RISC's region CTRL base; RTA1 = its RISC index (BlRisc, for display).
    uint32_t ctrl_base = get_arg_val<uint32_t>(0);
    uint32_t risc_id   = get_arg_val<uint32_t>(1);
    volatile tt_l1_ptr uint32_t* ctrl = (volatile tt_l1_ptr uint32_t*)ctrl_base;

    // First-boot init of our own control region (sole owner of this L1 window).
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
            return;                         // resident loop ends -> host may close() cleanly
        }
        if (cmd == CMD_EXEC) {
            uint32_t slot = ctrl[ARG0];     // host put the code-slot L1 addr here
            flush_icache();                 // MUST precede the jump: code was just written
            ctrl[STATUS] = ST_OVERLAY;
            overlay_fn fn = (overlay_fn)slot;
            uint32_t ret = fn((volatile uint32_t*)ctrl);   // <-- the bootloader call
            ctrl[OVL_RET]  = ret;
            ctrl[STATUS]   = ST_IDLE;
            ctrl[LAST_CMD] = cmd;
            ctrl[DOORBELL] = CMD_NONE;      // ack
        } else if (cmd == CMD_SETPARAM) {
            ctrl[LAST_CMD] = cmd;
            ctrl[DOORBELL] = CMD_NONE;
        }

        // Throttle the heartbeat PUBLISH (every ~1M spins) so external NoC reads of the hot
        // word land cleanly; doorbell polling stays every iteration so responsiveness holds.
        if ((++iter & 0xFFFFFu) == 0) {
            ctrl[HEARTBEAT] = iter;
            ctrl[WALLCLK]   = *(volatile uint32_t*)WALL_CLOCK_L;
        }
    }
}
