// SPDX-License-Identifier: Apache-2.0
// overlay.h — common ABI for bhtop bootloader code overlays (freestanding, runs on any RISC).
//
// An overlay is entered at run(ctrl) where ctrl points at ITS RISC's control-mailbox base
// (the region CTRL base — one of 5 per tile). Live params are ctrl[BL_PARAM0+i]. Telemetry and
// scratch are RELATIVE to ctrl (region base + fixed offsets), so the SAME overlay binary is
// correct on BRISC/NCRISC/TRISC0-2 without per-RISC edits. Overlays return a u32 (stored in
// OVL_RET). The bootloader invalidates i$ and calls run(); when it returns, the resident loop
// resumes — so write telemetry AFTER the hot loop (reads are clean once the core stops hammering).
//
// Region layout (mirror bootloader_abi.h): CTRL @ +0x0000, TELEM @ +0x1000 (4 KiB),
// SLOT_A @ +0x2000 (24 KiB), SLOT_B @ +0x8000 (32 KiB). When only slot A is in use, slot B's
// 32 KiB doubles as overlay scratch (ovl_scratch).
//
// Standard telemetry layout (every overlay fills these; the manifest labels them):
//   telem[0] = work units done      telem[1] = wall-clock cycles elapsed
//   telem[2] = result/checksum      telem[3] = 0xC0FFEE done marker
//   telem[4..] = overlay-specific
// The cockpit derives throughput = work / cycles.
#pragma once
#include <cstdint>

#define WALL_CLOCK_L 0xFFB121F0u              // RISCV_DEBUG_REG_WALL_CLOCK_0 (free-running, low 32b)

enum {
    BL_PARAM0       = 4,        // ctrl word index of PARAM0
    BL_TELEM_WOFF   = 0x1000u / 4,   // TELEM   region offset in WORDS (0x1000 bytes)
    BL_SCRATCH_WOFF = 0x8000u / 4,   // SLOT_B  region offset in WORDS — scratch when slot A runs
};

static inline uint32_t ovl_param(volatile uint32_t* ctrl, int i) { return ctrl[BL_PARAM0 + i]; }
// Telemetry / scratch are relative to THIS RISC's region (ctrl is the region CTRL base).
static inline volatile uint32_t* ovl_telem(volatile uint32_t* ctrl) { return ctrl + BL_TELEM_WOFF; }
static inline volatile uint32_t* ovl_scratch(volatile uint32_t* ctrl) { return ctrl + BL_SCRATCH_WOFF; }
static inline uint32_t ovl_cycles(void) { return *(volatile uint32_t*)WALL_CLOCK_L; }

// Fill the standard telemetry header. Call once, after the work loop.
static inline void ovl_publish(volatile uint32_t* ctrl, uint32_t work, uint32_t cycles, uint32_t result) {
    volatile uint32_t* t = ovl_telem(ctrl);
    t[0] = work; t[1] = cycles; t[2] = result; t[3] = 0xC0FFEEu;
}
