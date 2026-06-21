// SPDX-License-Identifier: Apache-2.0
// crt_x280.c — SCALAR baseline of the CRT/quarter-square 32x32 int8 matmul on the SiFive x280.
// Generates the shared test pattern, runs the matmul (mod 280), and surfaces a checksum + four
// spot values + the mcycle/minstret cost into telemetry. The RVV kernel (crt_x280_rvv.c) must
// match the checksum bit-for-bit; the host harness compares both to crt/crt_matmul.py.
//
// Build/run:  bhtop-l2cpu bringup <tile>   then the crt/run_x280.py harness (compiles + loads + reads).
//   TELE[0]=heartbeat  TELE[1]=checksum  TELE[2]=cycles  TELE[3..6]=C[0][0],C[1][2],C[15][15],C[31][31]
//   TELE[7]=instret  TELE[8]=1 (scalar tag)
#include <bh.h>
#include <crt_kernel.h>

int main(void) {
    static crt_u8 Aw[CRT_N][CRT_N], Ad[CRT_N][CRT_N];
    static crt_u16 C[CRT_N][CRT_N];

    crt_pattern(Aw, Ad);

    unsigned long c0 = bh_cycles(), r0 = bh_instret();
    crt_matmul(Aw, Ad, C, /*shiftadd=*/0);
    unsigned long c1 = bh_cycles(), r1 = bh_instret();

    TELE[1] = crt_checksum(C);
    TELE[2] = (u32)(c1 - c0);
    TELE[3] = C[0][0];  TELE[4] = C[1][2];  TELE[5] = C[15][15];  TELE[6] = C[31][31];
    TELE[7] = (u32)(r1 - r0);
    TELE[8] = 1;                         // 1 = scalar build
    unsigned hb = 0;
    for (;;) { hb++; TELE[0] = hb; }
}
