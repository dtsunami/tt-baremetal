// SPDX-License-Identifier: Apache-2.0
// rvv_fp — floating-point vector FMA: vfmacc (acc += a*b) on f32, 16 lanes/op. The x280 has full FP
// RVV — this is the projection/splatting workhorse (and what vec_virus heats the core with). The live
// signal is TELE[2] cyc/op; TELE[3]=raw f32 bits of lane-0 (grows then saturates — that's expected).
#include <rvv.h>
int main(void) {
    RVV_INIT();
    size_t vl = __riscv_vsetvl_e32m1(16);                       // f32: 16 lanes
    vfloat32m1_t a = __riscv_vfmv_v_f_f32m1(1.5f, vl);
    vfloat32m1_t b = __riscv_vfmv_v_f_f32m1(2.0f, vl);
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, vl);
    RVV_RUN(acc = __riscv_vfmacc_vv_f32m1(acc, a, b, vl),
            __riscv_vmv_x_s_u32m1_u32(__riscv_vreinterpret_v_f32m1_u32m1(acc)));
}
