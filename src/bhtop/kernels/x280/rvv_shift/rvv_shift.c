// SPDX-License-Identifier: Apache-2.0
// rvv_shift — the multiplier-free path. vsll/vsrl/vsra; a constant multiply becomes a SHIFT-ADD network
// (x*105 = x + (x<<3) - (x<<5) + (x<<7)) — the CRT reconstruction's "park the constant / wire with
// bitshifts" trick, no vmul. Branchless. TELE[3]=lane-0 = 1*105 = 105.
#include <rvv.h>
int main(void) {
    RVV_INIT();
    size_t vl = rvv_vl32();
    vuint16m1_t x = __riscv_vadd_vx_u16m1(__riscv_vid_v_u16m1(vl), 1, vl);     // 1..32
    vuint16m1_t y = __riscv_vmv_v_x_u16m1(0, vl);
    RVV_RUN(y = __riscv_vsub_vv_u16m1(
                __riscv_vadd_vv_u16m1(__riscv_vadd_vv_u16m1(x, __riscv_vsll_vx_u16m1(x, 3, vl), vl),
                                      __riscv_vsll_vx_u16m1(x, 7, vl), vl),
                __riscv_vsll_vx_u16m1(x, 5, vl), vl),
            __riscv_vmv_x_s_u16m1_u16(y));
}
