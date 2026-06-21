// SPDX-License-Identifier: Apache-2.0
// rvv_mask — predication = branchless control flow. Build a mask (vmsgtu: which lanes > 15), then
// vmerge SELECTS per lane: y[i] = mask[i] ? 0xFF : x[i]. This is "if without a branch" — the RVV way
// to do data-dependent work. TELE[3]=lane-20 (20>15 -> selected 0xFF).
#include <rvv.h>
int main(void) {
    RVV_INIT();
    size_t vl = rvv_vl32();
    vuint16m1_t x = __riscv_vid_v_u16m1(vl);                    // 0..31
    vbool16_t m = __riscv_vmsgtu_vx_u16m1_b16(x, 15, vl);       // lanes where x > 15
    vuint16m1_t y = __riscv_vmv_v_x_u16m1(0, vl);
    RVV_RUN(y = __riscv_vmerge_vxm_u16m1(x, 0xFF, m, vl),
            __riscv_vmv_x_s_u16m1_u16(__riscv_vslidedown_vx_u16m1(y, 20, vl)));
}
