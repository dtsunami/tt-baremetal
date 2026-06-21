// SPDX-License-Identifier: Apache-2.0
// sfpu — VECTOR (SFPU) exerciser.  *** EXPERIMENTAL ***
// Seeds three SFPU vector registers (SFPLOADI) then loops a fused multiply-add (SFPMAD) across
// the 32-lane SFPU to drive the vector pipe — a throughput/power probe for the vector unit,
// pushed from BRISC via the Tensix instruction buffer. (SFPU ops are normally TRISC-only; this
// exercises whether the backend accepts them from the instruction FIFO.)
// HANG RISK: same as matrix — a stalled SFPU backs up the FIFO and wedges the core; `tt-smi -r 0`.
//   PARAM0 = SFPMADs to issue
#include "overlay.h"

#define INSTRN_BUF_BASE 0xFFE40000u
#define TT_OP(opc, p) ((((uint32_t)(opc)) << 24) | ((p) & 0x00FFFFFFu))
#define TT_OP_SFPLOADI(lreg, mod0, imm16) TT_OP(0x71u, (((lreg) << 20) | ((mod0) << 16) | ((imm16) << 0)))
#define TT_OP_SFPMAD(a, b, c, d, m)       TT_OP(0x84u, (((a) << 16) | ((b) << 12) | ((c) << 8) | ((d) << 4) | ((m) << 0)))
#define TT_OP_SFPNOP TT_OP(0x8fu, 0)

extern "C" __attribute__((section(".text.entry"))) uint32_t run(volatile uint32_t* ctrl) {
    uint32_t n = ovl_param(ctrl, 0);
    if (n == 0) n = 100000;

    volatile uint32_t* ib = (volatile uint32_t*)INSTRN_BUF_BASE;
    // seed lreg0=1.0-ish, lreg1, lreg2 with immediates (mod0=0 loads imm16 into low half)
    ib[0] = TT_OP_SFPLOADI(0, 0, 0x3f80);   // ~1.0 bf16 top half
    ib[0] = TT_OP_SFPLOADI(1, 0, 0x4000);
    ib[0] = TT_OP_SFPLOADI(2, 0, 0x3f00);
    uint32_t mad = TT_OP_SFPMAD(0, 1, 2, 3, 0);   // lreg3 = lreg0*lreg1 + lreg2

    uint32_t c0 = ovl_cycles();
    for (uint32_t i = 0; i < n; i++) ib[0] = mad;
    ib[0] = TT_OP_SFPNOP;
    uint32_t c1 = ovl_cycles();

    ovl_publish(ctrl, n, c1 - c0, mad);
    return n;
}
