// SPDX-License-Identifier: Apache-2.0
// rvv_perm — moving data ACROSS lanes: vslidedown shifts every lane down by N (rotate/window),
// the permute-network class (gather/scatter, sorting, splat compaction). Cross-lane. TELE[3]=lane-0
// after slidedown-by-4 = the original lane 4 = 4.
#include <rvv.h>
int main(void) {
    RVV_INIT();
    size_t vl = rvv_vl32();
    vuint16m1_t x = __riscv_vid_v_u16m1(vl);                    // 0..31
    vuint16m1_t y = __riscv_vmv_v_x_u16m1(0, vl);
    RVV_RUN(y = __riscv_vslidedown_vx_u16m1(x, 4, vl),
            __riscv_vmv_x_s_u16m1_u16(y));
}
