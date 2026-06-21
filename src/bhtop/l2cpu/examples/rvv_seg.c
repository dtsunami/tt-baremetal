// SPDX-License-Identifier: Apache-2.0
// rvv_seg — segment (structured) load: vlseg3 de-interleaves [x0,y0,z0,x1,y1,z1,...] into three
// vector regs (all x's / all y's / all z's) in ONE op — exactly the access pattern for array-of-structs
// splat data (positions, colors). TELE[3]=second x = buf[3] = 3.
#include <rvv.h>
int main(void) {
    RVV_INIT();
    size_t vl = __riscv_vsetvl_e16m1(10);
    static unsigned short buf[30];
    for (int i = 0; i < 30; i++) buf[i] = (unsigned short)i;        // x,y,z interleaved
    vuint16m1_t vx = __riscv_vmv_v_x_u16m1(0, vl);
    RVV_RUN(({ vuint16m1x3_t s = __riscv_vlseg3e16_v_u16m1x3(buf, vl);
               vx = __riscv_vget_v_u16m1x3_u16m1(s, 0); }),           // vx = [0,3,6,9,...]
            __riscv_vmv_x_s_u16m1_u16(__riscv_vslidedown_vx_u16m1(vx, 1, vl)));
}
