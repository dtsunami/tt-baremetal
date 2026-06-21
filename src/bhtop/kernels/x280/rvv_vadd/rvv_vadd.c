// SPDX-License-Identifier: Apache-2.0
// rvv_vadd — SIMD "hello world": one vadd.vv adds 32 int16 lanes at once. Branchless by nature.
// Cockpit: Plot TELE[2] (cyc/op) or TELE[0] in rate mode. TELE[3]=live lane-0 running sum.
#include <rvv.h>
int main(void) {
    RVV_INIT();                                         // MUST be first (vector ctx + chicken bits)
    size_t vl = rvv_vl32();
    vuint16m1_t a = __riscv_vid_v_u16m1(vl);            // [0,1,...,31]
    vuint16m1_t b = __riscv_vrsub_vx_u16m1(a, 31, vl);  // [31,...,0]  (a+b == 31 every lane)
    vuint16m1_t acc = __riscv_vmv_v_x_u16m1(0, vl);
    RVV_RUN(acc = __riscv_vadd_vv_u16m1(acc, __riscv_vadd_vv_u16m1(a, b, vl), vl),
            __riscv_vmv_x_s_u16m1_u16(acc));            // lane-0 = 31*iter (live, plottable)
}
