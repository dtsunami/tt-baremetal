// SPDX-License-Identifier: Apache-2.0
// rvv_remu — vector modulo, the CRT residue step. vremu.vx is the general divider; for 2^k moduli
// a vand mask is FREE (why crt/sweep.py favors power-of-2 / 2^k±1 moduli). Demos mod 7 here.
// TELE[3]=lane-1 (x mod 7).
#include <rvv.h>
int main(void) {
    RVV_INIT();
    size_t vl = rvv_vl32();
    vuint16m1_t x = __riscv_vmul_vx_u16m1(__riscv_vadd_vx_u16m1(__riscv_vid_v_u16m1(vl), 1, vl), 2654, vl);
    vuint16m1_t r = __riscv_vmv_v_x_u16m1(0, vl);
    RVV_RUN(r = __riscv_vremu_vx_u16m1(x, 7, vl),
            __riscv_vmv_x_s_u16m1_u16(r));               // lane-0 = 2654 % 7 = 1
}
