// SPDX-License-Identifier: Apache-2.0
// rvv_mac — fused multiply-accumulate: acc += a*b per lane (vmacc.vv) = 32 MACs per instruction,
// the dot-product workhorse (and the matmul inner step before we switched to quarter-square LUTs).
// Branchless. TELE[3]=lane-0 running MAC.
#include <rvv.h>
int main(void) {
    RVV_INIT();
    size_t vl = rvv_vl32();
    vuint16m1_t a = __riscv_vadd_vx_u16m1(__riscv_vid_v_u16m1(vl), 1, vl);    // 1..32
    vuint16m1_t b = __riscv_vrsub_vx_u16m1(__riscv_vid_v_u16m1(vl), 33, vl);  // 33..2
    vuint16m1_t acc = __riscv_vmv_v_x_u16m1(0, vl);
    RVV_RUN(acc = __riscv_vmacc_vv_u16m1(acc, a, b, vl),
            __riscv_vmv_x_s_u16m1_u16(acc));   // lane-0 accumulator (aliases to 0 at the 2^16 sample
                                               // boundary; the live signal is TELE[2] cyc/op = 7)
}
