// SPDX-License-Identifier: Apache-2.0
// rvv_reduce — horizontal reduction: vredsum collapses all 32 lanes to one scalar (the dot-product
// final step). Cross-lane, so slower than elementwise — watch cyc/op vs rvv_vadd. TELE[3]=sum(0..31)=496.
#include <rvv.h>
int main(void) {
    RVV_INIT();
    size_t vl = rvv_vl32();
    vuint16m1_t x = __riscv_vid_v_u16m1(vl);
    vuint16m1_t z = __riscv_vmv_v_x_u16m1(0, vl);
    vuint16m1_t s = __riscv_vmv_v_x_u16m1(0, vl);
    RVV_RUN(s = __riscv_vredsum_vs_u16m1_u16m1(x, z, vl),
            __riscv_vmv_x_s_u16m1_u16(s));
}
