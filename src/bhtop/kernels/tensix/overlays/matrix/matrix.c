// SPDX-License-Identifier: Apache-2.0
// matrix — TILE / MATRIX (FPU) exerciser.  *** EXPERIMENTAL ***
// Pushes MVMUL instructions straight into the Tensix instruction buffer from BRISC to drive the
// matrix engine (FPU) — a throughput/power probe for the tile unit. There's no Src/Dest tile
// bookkeeping, so results are meaningless; the point is matrix-engine ACTIVITY + cycle cost.
// HANG RISK: if the FPU stalls (e.g. waiting on src-valid) the instruction FIFO backs up and the
// BRISC write blocks → the core wedges and won't return to the bootloader loop. Recover with
// `tt-smi -r 0`. Try PARAM1 (clear_dvalid) = 3 to reduce src-valid stalls.
//   PARAM0 = MVMULs to issue   PARAM1 = clear_dvalid (0..3, default 3)
//
// *** WHY THIS CANNOT BE FIXED IN PLACE (root cause, 2026-06) ***
// This is a deliberate negative-control power probe, NOT a path to a working matmul. Two structural
// reasons, both from the Tensix instruction-FIFO topology:
//   1. INSTRN_BUF_BASE 0xFFE40000 is the per-thread instruction FIFO, and 0xFFE40000 is the T0
//      (UNPACK) port — MVMUL belongs on T1 (MATH). INSTRN_BUF_STRIDE is 0x10000 (math = 0xFFE50000).
//   2. Even retargeted to the math port, MVMUL stalls on SrcA/SrcB dvalid (produced by the unpack
//      thread) and DEST-bank sync (produced by the pack thread). A single isolated instruction
//      stream has no producer/consumer → the FPU stalls, the FIFO backs up, the RISC store wedges.
// Adding device_setup()/addr-mod/WRCFG from BRISC does NOT help: those TTI_* ops bind to the
// ISSUING thread's CFG state, so from the BRISC overlay (T0 port) they program the wrong thread.
// The CORRECT path for real FPU/matmul work is the TRISC-boot LLK lane (LLK_BOOT_MODE_TRISC) —
// see kernels/tensix/llk/{matmul_perf,math_matmul_perf} (MATH_ISOLATE drives the FPU without a live
// unpacker). Keep this overlay only as a "does the backend wedge?" probe.
#include "overlay.h"

#define INSTRN_BUF_BASE 0xFFE40000u
#define TT_OP(opc, p) ((((uint32_t)(opc)) << 24) | ((p) & 0x00FFFFFFu))
#define TT_OP_MVMUL(cd, im, am, d) TT_OP(0x26u, (((cd) << 22) | ((im) << 19) | ((am) << 14) | ((d) << 0)))
#define TT_OP_NOP TT_OP(0x02u, 0)

extern "C" __attribute__((section(".text.entry"))) uint32_t run(volatile uint32_t* ctrl) {
    uint32_t n = ovl_param(ctrl, 0);
    if (n == 0) n = 100000;
    uint32_t clr = ovl_param(ctrl, 1);
    if (clr == 0) clr = 3;                       // default: clear src-valid each op to avoid stalls
    clr &= 0x3;

    volatile uint32_t* ib = (volatile uint32_t*)INSTRN_BUF_BASE;
    uint32_t mv = TT_OP_MVMUL(clr, 0, 0, 0);

    uint32_t c0 = ovl_cycles();
    for (uint32_t i = 0; i < n; i++) ib[0] = mv;
    ib[0] = TT_OP_NOP;
    uint32_t c1 = ovl_cycles();

    ovl_publish(ctrl, n, c1 - c0, mv);
    return n;
}
