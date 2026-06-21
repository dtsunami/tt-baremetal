// SPDX-License-Identifier: Apache-2.0
// rvv_fixed — fixed-point arithmetic: vsaddu SATURATES instead of wrapping (clamp, no overflow) — the
// DSP / quantized-splat idiom. TELE[3]=lane-31: (32<<10)+0x8000 clamps to 0xFFFF instead of wrapping.
#include <rvv.h>
int main(void) {
    RVV_INIT();
    size_t vl = rvv_vl32();
    vuint16m1_t x = __riscv_vsll_vx_u16m1(__riscv_vadd_vx_u16m1(__riscv_vid_v_u16m1(vl), 1, vl), 10, vl);
    vuint16m1_t y = __riscv_vmv_v_x_u16m1(0, vl);
    RVV_RUN(y = __riscv_vsaddu_vx_u16m1(x, 0x8000, vl),
            __riscv_vmv_x_s_u16m1_u16(__riscv_vslidedown_vx_u16m1(y, 31, vl)));
}
