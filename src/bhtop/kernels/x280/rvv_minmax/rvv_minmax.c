// SPDX-License-Identifier: Apache-2.0
// rvv_minmax — BRANCHLESS absolute difference: |a-b| = max(a,b) - min(a,b) (vmaxu/vminu/vsub), no
// per-lane if. This is the quarter-square multiply's |a-b| step, and the canonical "masks not
// branches" RVV idiom. TELE[3]=lane-0 |a-b|.
#include <rvv.h>
int main(void) {
    RVV_INIT();
    size_t vl = rvv_vl32();
    vuint16m1_t a = __riscv_vid_v_u16m1(vl);             // 0..31
    vuint16m1_t b = __riscv_vrsub_vx_u16m1(a, 17, vl);   // 17-i (unsigned wrap -> a,b unordered)
    vuint16m1_t d = __riscv_vmv_v_x_u16m1(0, vl);
    RVV_RUN(d = __riscv_vsub_vv_u16m1(__riscv_vmaxu_vv_u16m1(a, b, vl),
                                      __riscv_vminu_vv_u16m1(a, b, vl), vl),
            __riscv_vmv_x_s_u16m1_u16(d));               // lane-0 = |0-17| = 17
}
