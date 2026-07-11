// SPDX-License-Identifier: Apache-2.0
//
// bootloader_abi.h — the shared contract between (1) the resident bootloader kernels,
// (2) the host launcher, and (3) bhtop's live console over tt-exalens.
//
// Concept: tt-metal brings the grid up (device open, NoC + Tensix backend init) and
// multicasts a resident "bootloader" kernel to every worker core. That kernel never
// returns — it loops polling a control mailbox in L1. The host then PARKS (no Finish,
// no close, so the cores are never reset). From there, bhtop owns all live interaction:
// poke params, stage new machine code into a code slot, ring the doorbell -> the
// bootloader invalidates the i-cache and *calls* the staged code (exactly what
// firmware does at brisc.cc:517). No re-JIT-launch, no soft-reset, no teardown.
//
// FIVE RISCS PER TILE. A Tensix tile has 5 baby RISC processors:
//   index 0 = BRISC  (data-movement RISCV_0)
//   index 1 = NCRISC (data-movement RISCV_1)
//   index 2 = TRISC0 (compute / UNPACK)
//   index 3 = TRISC1 (compute / MATH)
//   index 4 = TRISC2 (compute / PACK)
// All 5 share the SAME L1, so each owns a DISJOINT control+telem+code region. We carve
// one uniform 64 KiB region per RISC out of high L1, parameterized by the RISC index.
// The launcher creates a bootloader kernel on each of the 5 (one DataMovement kernel on
// BRISC, one on NCRISC, one Compute kernel that runs on all 3 TRISCs) and passes each its
// region base as a runtime arg. From the host/NoC side all 5 regions live at the SAME
// core coordinate — per-RISC is purely an L1 OFFSET (the region base), not a new endpoint.
//
// EVERYTHING here is FIXED L1 addresses (not metal-allocated), so bhtop and the overlay
// linker script can compute them. Metal's firmware/mailboxes live in LOW L1 (< 0x40000)
// and kernel stacks live in LOCAL mem (0xFFB00000), so our high-L1 block is collision-free
// (the launcher also allocates no CBs/buffers). L1 is 1536 KiB = 0x180000 on Blackhole.

#pragma once
#include <cstdint>

// ---- Per-RISC region carve-out (5 uniform regions in high L1) -------------------------
// Region(i) = BL_REGION_BASE + i * BL_REGION_STRIDE, for i in [0, BL_NUM_RISCS).
//   BRISC  region @ 0x100000   NCRISC region @ 0x110000
//   TRISC0 region @ 0x120000   TRISC1 region @ 0x130000   TRISC2 region @ 0x140000
// Top of the last region = 0x150000, well under L1 end (0x180000).
constexpr uint32_t BL_NUM_RISCS     = 5u;
constexpr uint32_t BL_REGION_BASE   = 0x00100000u;   // region 0 (BRISC) base
constexpr uint32_t BL_REGION_STRIDE = 0x00010000u;   // 64 KiB per RISC

// Offsets WITHIN a region (add to the region base):
constexpr uint32_t BL_CTRL_OFF   = 0x00000000u;      // mailbox header (control words)
constexpr uint32_t BL_TELEM_OFF  = 0x00001000u;      // 4 KiB scratch the kernel/overlay publishes
constexpr uint32_t BL_SLOT_A_OFF = 0x00002000u;      // 24 KiB code overlay slot A
constexpr uint32_t BL_SLOT_B_OFF = 0x00008000u;      // 32 KiB code overlay slot B (double-buffer)
constexpr uint32_t BL_SLOT_A_SIZE = 0x00006000u;     // 24 KiB
constexpr uint32_t BL_SLOT_B_SIZE = 0x00008000u;     // 32 KiB

// Helpers — region/sub-region byte address for RISC `i`.
constexpr uint32_t bl_region(uint32_t i) { return BL_REGION_BASE + i * BL_REGION_STRIDE; }
constexpr uint32_t bl_ctrl(uint32_t i)   { return bl_region(i) + BL_CTRL_OFF; }
constexpr uint32_t bl_telem(uint32_t i)  { return bl_region(i) + BL_TELEM_OFF; }
constexpr uint32_t bl_slot_a(uint32_t i) { return bl_region(i) + BL_SLOT_A_OFF; }
constexpr uint32_t bl_slot_b(uint32_t i) { return bl_region(i) + BL_SLOT_B_OFF; }

// RISC indices (also the value each kernel publishes in BL_RISC_ID).
enum BlRisc : uint32_t {
    BL_RISC_BRISC  = 0,
    BL_RISC_NCRISC = 1,
    BL_RISC_TRISC0 = 2,
    BL_RISC_TRISC1 = 3,
    BL_RISC_TRISC2 = 4,
};

// Back-compat single-region aliases (BRISC = region 0). Older overlays/tools that assumed
// one fixed region map onto BRISC's region.
constexpr uint32_t BL_CTRL_BASE   = BL_REGION_BASE + BL_CTRL_OFF;    // 0x100000
constexpr uint32_t BL_TELEM_BASE  = BL_REGION_BASE + BL_TELEM_OFF;   // 0x101000
constexpr uint32_t BL_CODE_SLOT_A = BL_REGION_BASE + BL_SLOT_A_OFF;  // 0x102000
constexpr uint32_t BL_CODE_SLOT_B = BL_REGION_BASE + BL_SLOT_B_OFF;  // 0x108000

// ---- Control mailbox layout (word indices off a region's CTRL base) -------------------
// Host writes DOORBELL (+ args); kernel acts then acks by writing DOORBELL = BL_CMD_NONE.
enum BlCtrl : uint32_t {
    BL_DOORBELL  = 0,   // command in  (host -> kernel)
    BL_ARG0      = 1,   // e.g. code-slot address for EXEC
    BL_ARG1      = 2,
    BL_PARAM0    = 4,   // live params the resident loop / overlay reads each iteration
    BL_PARAM1    = 5,
    BL_PARAM2    = 6,
    BL_PARAM3    = 7,
    BL_HEARTBEAT = 16,  // kernel increments every loop iteration (liveness)
    BL_LAST_CMD  = 17,  // echo of the last command consumed (host can confirm ack)
    BL_STATUS    = 18,  // BlStatus
    BL_OVL_RET   = 19,  // last overlay return value
    BL_WALLCLK   = 20,  // on-core wall-clock stamp at last heartbeat publish
    BL_RISC_ID   = 21,  // which RISC owns this region (BlRisc) — kernel stamps it at boot
};
constexpr uint32_t BL_CTRL_WORDS = 22;  // words the host reads for a full status

enum BlCmd : uint32_t {
    BL_CMD_NONE     = 0x00000000u,
    BL_CMD_SETPARAM = 0x00000001u,  // params already poked; just bump an epoch (optional)
    BL_CMD_EXEC     = 0x00000002u,  // ARG0 = code-slot L1 addr; invalidate i$ then call it
    BL_CMD_HALT     = 0x0000DEADu,  // break the resident loop -> kernel returns -> safe to close()
};

enum BlStatus : uint32_t {
    BL_ST_BOOT     = 0,
    BL_ST_IDLE     = 1,  // looping, no overlay active
    BL_ST_OVERLAY  = 2,  // inside a staged overlay call
    BL_ST_HALTED   = 3,
};

// ---- Overlay ABI ----------------------------------------------------------------------
// A staged code blob is entered at the slot base. It receives the control-mailbox base
// pointer (its OWN region's CTRL base) so it can read params (ctrl[BL_PARAM0+i]) and
// publish telemetry RELATIVE to ctrl (telem = ctrl + BL_TELEM_OFF), and returns a u32
// (stored in BL_OVL_RET). Return to hand control back to the resident loop.
//   extern "C" uint32_t run(volatile uint32_t* ctrl);
using bl_overlay_fn = uint32_t (*)(volatile uint32_t*);

// ---- I-cache invalidate (Blackhole) ---------------------------------------------------
// CFG register word 185, mask 0x1f selects {BRISC, TRISC0/1/2, NCRISC}. Same write
// firmware does at brisc.cc:239 before running freshly-loaded code. Writing it from ANY
// RISC invalidates all 5 i-caches (a harmless refetch for the others). We name the cfg
// base uniquely (BL_CFG_BASE) to dodge the TENSIX_CFG_BASE macro-redefinition collision
// in the JIT include env (it's a #define, so a same-named constexpr is a hard error).
constexpr uint32_t BL_CFG_BASE                = 0xFFEF0000u;  // == TENSIX_CFG_BASE (tensix.h)
constexpr uint32_t BL_RISCV_IC_INVALIDATE_ADDR32 = 185u;      // cfg_defines.h (blackhole)
constexpr uint32_t BL_RISCV_IC_INVALIDATE_ALL    = 0x1fu;
constexpr uint32_t BL_WALL_CLOCK_L            = 0xFFB121F0u;  // RISCV_DEBUG_REG_WALL_CLOCK_0
