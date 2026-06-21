// SPDX-License-Identifier: Apache-2.0
// crt_x280_rvv.c — RVV version of the CRT/quarter-square 32x32 int8 matmul on the SiFive x280.
//
// Vectorizes over the 32 OUTPUT COLUMNS: VLEN=512, e16, LMUL=1 -> exactly 32 lanes, so one vector
// register is a whole output row and there are NO reductions. Residues are computed ON THE FLY
// (vle8 the Ad row -> zero-extend -> vremu m) so the kernel's .bss stays small (~4 KB, like the
// scalar build) — large static residue tables overran the writable GDDR window (L2/L3 cache config
// is skipped by the minimal bringup) and faulted crt0's bss-zero before main. The quarter-square LUT
// is one `vrgather.vv`. Must match crt_x280.c / the golden model bit-exact.
//
//   TELE[1]=checksum  TELE[2]=cycles  TELE[3..6]=spot values  TELE[7]=instret  TELE[8]=2 (rvv tag)
#include <bh.h>
#include <riscv_vector.h>
#include <crt_kernel.h>

int main(void) {
    static crt_u8  Aw[CRT_N][CRT_N], Ad[CRT_N][CRT_N];   // raw pattern (1 KB each)
    static crt_u16 Ad16[CRT_N][CRT_N];                   // u16 copy of Ad (so we vle16, not vle8+vzext)
    static crt_u16 Qt[CRT_NMOD][40];                     // quarter-square LUTs
    static crt_u16 C[CRT_N][CRT_N];                      // output (2 KB)

    // Vector context MUST be enabled before ANY vector op. With -march=rv64gcv -O2 the compiler
    // AUTO-VECTORIZES the scalar setup loops below (crt_pattern, the copy, the LUT fill), so the
    // enable + chicken-bit clear have to be the very first thing in main — otherwise those
    // auto-emitted RVV instructions trap (VS=off) before we ever reach the matmul.
    __asm__ volatile("csrw 0x7c1, zero");                // clear SiFive feature-disable chicken bits
    bh_vec_enable();                                     // mstatus.VS on
    TELE[12] = 0xE0;                                     // entry marker

    crt_pattern(Aw, Ad);
    for (int mi = 0; mi < CRT_NMOD; mi++) {
        int m = CRT_MODULI[mi];
        for (int n = 0; n < 2 * m; n++) Qt[mi][n] = (crt_u16)crt_qsq(n);
    }
    for (int k = 0; k < CRT_N; k++)
        for (int j = 0; j < CRT_N; j++) Ad16[k][j] = Ad[k][j];
    TELE[9] = 0xAA;                                      // precompute done

    unsigned long c0 = bh_cycles(), ir0 = bh_instret();
    size_t vl = __riscv_vsetvl_e16m1(CRT_N);             // 32 lanes
    for (int i = 0; i < CRT_N; i++) {
        if (i == 0) TELE[10] = 0xB0;                     // entered row loop
        vuint16m1_t value = __riscv_vmv_v_x_u16m1(0, vl);
        for (int mi = 0; mi < CRT_NMOD; mi++) {
            int m = CRT_MODULI[mi];
            vuint16m1_t qtab = __riscv_vle16_v_u16m1(Qt[mi], vl);
            vuint16m1_t acc = __riscv_vmv_v_x_u16m1(0, vl);
            for (int k = 0; k < CRT_N; k++) {
                crt_u16 a = (crt_u16)(Aw[i][k] % m);                         // scalar residue
                vuint16m1_t bd = __riscv_vremu_vx_u16m1(
                    __riscv_vle16_v_u16m1(Ad16[k], vl), (crt_u16)m, vl);     // Ad[k][j] % m
                vuint16m1_t s  = __riscv_vadd_vx_u16m1(bd, a, vl);           // a + b
                vuint16m1_t mx = __riscv_vmaxu_vx_u16m1(bd, a, vl);
                vuint16m1_t mn = __riscv_vminu_vx_u16m1(bd, a, vl);
                vuint16m1_t d  = __riscv_vsub_vv_u16m1(mx, mn, vl);          // |a - b|
                vuint16m1_t qs = __riscv_vrgather_vv_u16m1(qtab, s, vl);     // q[a+b]
                vuint16m1_t qd = __riscv_vrgather_vv_u16m1(qtab, d, vl);     // q[|a-b|]
                acc = __riscv_vadd_vv_u16m1(acc, __riscv_vsub_vv_u16m1(qs, qd, vl), vl);
            }
            vuint16m1_t r = __riscv_vremu_vx_u16m1(acc, (crt_u16)m, vl);
            value = __riscv_vadd_vv_u16m1(
                value, __riscv_vmul_vx_u16m1(r, (crt_u16)CRT_COEFF[mi], vl), vl);
        }
        vuint16m1_t crow = __riscv_vremu_vx_u16m1(value, (crt_u16)CRT_BIGN, vl);
        __riscv_vse16_v_u16m1(C[i], crow, vl);
    }
    unsigned long c1 = bh_cycles(), ir1 = bh_instret();

    TELE[1] = crt_checksum(C);
    TELE[2] = (u32)(c1 - c0);
    TELE[3] = C[0][0];  TELE[4] = C[1][2];  TELE[5] = C[15][15];  TELE[6] = C[31][31];
    TELE[7] = (u32)(ir1 - ir0);
    TELE[8] = 2;                          // 2 = rvv build
    unsigned hb = 0;
    for (;;) { hb++; TELE[0] = hb; }
}
