// SPDX-License-Identifier: Apache-2.0
// rvv_widen — widening multiply: vwmulu does u16*u16 -> u32 (output LMUL doubles to u32m2), the
// overflow-safe way to build wide accumulators (matmul/conv without wrap). TELE[3]=lane-0 = 1000*1000
// = 1,000,000 (would wrap to a garbage u16 without widening).
#include <rvv.h>
int main(void) {
    RVV_INIT();
    size_t vl = rvv_vl32();
    vuint16m1_t a = __riscv_vmv_v_x_u16m1(1000, vl);
    vuint16m1_t b = __riscv_vmv_v_x_u16m1(1000, vl);
    vuint32m2_t w = __riscv_vmv_v_x_u32m2(0, vl);
    RVV_RUN(w = __riscv_vwmulu_vv_u32m2(a, b, vl),
            __riscv_vmv_x_s_u32m2_u32(w));
}
