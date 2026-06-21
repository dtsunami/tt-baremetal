// SPDX-License-Identifier: Apache-2.0
// rvv_gather — table lookup as ONE instruction: vrgather.vv reads q[idx] for all 32 lanes, no loop,
// no branch. q[n]=floor(n^2/4) is the quarter-square LUT at the heart of the CRT matmul. The
// hardware gather IS the LUT. TELE[3]=q[3].
#include <rvv.h>
int main(void) {
    RVV_INIT();
    size_t vl = rvv_vl32();
    static unsigned short q[32];
    for (int n = 0; n < 32; n++) q[n] = (unsigned short)((n * n) >> 2);   // quarter-square table
    vuint16m1_t qt  = __riscv_vle16_v_u16m1(q, vl);
    vuint16m1_t idx = __riscv_vand_vx_u16m1(__riscv_vadd_vx_u16m1(__riscv_vid_v_u16m1(vl), 3, vl), 31, vl);
    vuint16m1_t g   = __riscv_vmv_v_x_u16m1(0, vl);
    RVV_RUN(g = __riscv_vrgather_vv_u16m1(qt, idx, vl),
            __riscv_vmv_x_s_u16m1_u16(g));               // lane-0 = q[3] = 2
}
