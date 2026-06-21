// SPDX-License-Identifier: Apache-2.0
// matrix — TILE / MATRIX (FPU) exerciser.  *** EXPERIMENTAL ***
// Pushes MVMUL instructions straight into the Tensix instruction buffer from BRISC to drive the
// matrix engine (FPU) — a throughput/power probe for the tile unit. There's no Src/Dest tile
// bookkeeping, so results are meaningless; the point is matrix-engine ACTIVITY + cycle cost.
// HANG RISK: if the FPU stalls (e.g. waiting on src-valid) the instruction FIFO backs up and the
// BRISC write blocks → the core wedges and won't return to the bootloader loop. Recover with
// `tt-smi -r 0`. Try PARAM1 (clear_dvalid) = 3 to reduce src-valid stalls.
//   PARAM0 = MVMULs to issue   PARAM1 = clear_dvalid (0..3, default 3)
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
