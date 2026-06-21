// SPDX-License-Identifier: Apache-2.0
// rvv_vdump — vector-register INTROSPECTION. The host can't read the x280's vector registers over the
// NoC, so bh_dump_vec() snapshots all 32 vregs + the 7 vector CSRs into this hart's VARCH block; decode
// host-side with L2cpu.vec_state(tile,hart) or the cockpit Vectors tab. The earlier "crash" was the
// missing vector-enable / chicken-bit — RVV_INIT() first fixes it. Seeds recognizable patterns so the
// decoded dump is obviously real.  TELE[3]=0xCAFE witness.
#include <rvv.h>
int main(void) {
    RVV_INIT();                                    // chicken bit + vector context FIRST
    size_t vl = rvv_vl32();
    vuint16m1_t a = __riscv_vid_v_u16m1(vl);                          // 0,1,...,31
    vuint16m1_t b = __riscv_vmv_v_x_u16m1(0xCAFE, vl);               // splat 0xCAFE
    vuint16m1_t c = __riscv_vsll_vx_u16m1(a, 8, vl);                 // i << 8
    RVV_RUN(({ asm volatile("" :: "r"(0)); bh_dump_vec();            // snapshot all vregs + vcsrs
               a = __riscv_vadd_vx_u16m1(a, 1, vl);                  // keep a/b/c live across the dump
               b = __riscv_vxor_vv_u16m1(b, c, vl); }),
            __riscv_vmv_x_s_u16m1_u16(__riscv_vmv_v_x_u16m1(0xCAFE, vl)));
}
