// SPDX-License-Identifier: Apache-2.0
// rvv_diag.c — pinpoint which vector op faults on the x280. TELE[8] is bumped after each step;
// whatever value it's stuck at = the last op that SURVIVED (the next one trapped/parked the hart).
#include <bh.h>
#include <riscv_vector.h>

int main(void) {
    TELE[8] = 1;                                   // alive, scalar
    __asm__ volatile("csrw 0x7c1, zero");          // chicken bits
    TELE[8] = 2;                                   // csrw 0x7c1 survived
    bh_vec_enable();
    TELE[8] = 3;
    size_t vl = __riscv_vsetvl_e16m1(32);
    TELE[8] = 4; TELE[1] = (u32)vl;

    static unsigned short tab[16] = {0,0,1,2,4,6,9,12,16,20,25,30,36,42,49,56};
    vuint16m1_t t = __riscv_vle16_v_u16m1(tab, vl);
    TELE[8] = 5;                                   // vle16 ok
    vuint16m1_t idx = __riscv_vid_v_u16m1(vl);
    TELE[8] = 6;                                   // vid ok
    vuint16m1_t am = __riscv_vand_vx_u16m1(idx, 15, vl);
    vuint16m1_t g = __riscv_vrgather_vv_u16m1(t, am, vl);
    TELE[8] = 7; TELE[2] = __riscv_vmv_x_s_u16m1_u16(g);    // vrgather ok
    vuint16m1_t mx = __riscv_vmaxu_vx_u16m1(idx, 3, vl);
    vuint16m1_t mn = __riscv_vminu_vx_u16m1(idx, 3, vl);
    (void)__riscv_vsub_vv_u16m1(mx, mn, vl);
    TELE[8] = 8;                                   // vmaxu/vminu/vsub ok
    vuint16m1_t mu = __riscv_vmul_vx_u16m1(idx, 105, vl);
    TELE[8] = 9; TELE[3] = __riscv_vmv_x_s_u16m1_u16(mu);   // vmul ok
    vuint16m1_t r = __riscv_vremu_vx_u16m1(idx, 7, vl);
    TELE[8] = 10; TELE[4] = __riscv_vmv_x_s_u16m1_u16(r);   // vremu ok
    __riscv_vse16_v_u16m1(tab, r, vl);
    TELE[8] = 11;                                  // vse16 ok — all ops good

    unsigned hb = 0;
    for (;;) { hb++; TELE[0] = hb; }
}
