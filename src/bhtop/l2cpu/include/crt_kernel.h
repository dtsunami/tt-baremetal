// SPDX-License-Identifier: Apache-2.0
// crt_kernel.h — pure-C scalar core of the CRT / quarter-square integer matmul.
//
// No hardware deps: this compiles on the host (validate bit-exact vs crt/crt_matmul.py) AND on the
// x280 (the bare-metal kernel includes it and wraps it with telemetry/cycles). The x280 RVV kernel
// vectorizes this same arithmetic, so this is also the scalar reference the vector path must match.
//
// Pipeline (mirrors the golden model): pack -> per-modulus quarter-square multiply + mod-m
// accumulate -> CRT reconstruct mod N.  Default modulus set {5,7,8}, N=280 (8-bit / mod-280 output).
#pragma once

#ifndef CRT_N
#define CRT_N 32                 // matrix dimension (square, K = N)
#endif
#define CRT_NMOD 3
#define CRT_BIGN 280             // N = 5*7*8

static const int CRT_MODULI[CRT_NMOD] = {5, 7, 8};
static const int CRT_COEFF[CRT_NMOD]  = {56, 120, 105};   // CRT reconstruction coefficients

typedef unsigned int   crt_u32;
typedef unsigned char  crt_u8;
typedef unsigned short crt_u16;

static inline int crt_qsq(int n) { return (n * n) >> 2; }   // floor(n^2/4), the quarter-square LUT

// Constant reconstruction multiply as a SHIFT-ADD network (no multiplier) — the efficiency lever.
// 56x=(x<<6)-(x<<3)  120x=(x<<7)-(x<<3)  105x=x+(x<<3)-(x<<5)+(x<<7)  (CSD, from the golden model).
static inline int crt_recon_shiftadd(int r5, int r7, int r8) {
    return ((r5 << 6) - (r5 << 3))
         + ((r7 << 7) - (r7 << 3))
         + (r8 + (r8 << 3) - (r8 << 5) + (r8 << 7));
}

// Deterministic test matrices, shared bit-for-bit with golden-model pattern_matrix() (no PRNG sync).
static void crt_pattern(crt_u8 Aw[CRT_N][CRT_N], crt_u8 Ad[CRT_N][CRT_N]) {
    for (int r = 0; r < CRT_N; r++)
        for (int c = 0; c < CRT_N; c++) {
            Aw[r][c] = (crt_u8)((r * 7 + c * 13 + 1) & 0xFF);   // weights  Aw[i][k]
            Ad[r][c] = (crt_u8)((r * 5 + c * 11 + 3) & 0xFF);   // data     Ad[k][j]
        }
}

// C = Aw @ Ad (mod N) via CRT + quarter-square. `shiftadd`: use the shift-add reconstruction.
static void crt_matmul(crt_u8 Aw[CRT_N][CRT_N], crt_u8 Ad[CRT_N][CRT_N],
                       crt_u16 C[CRT_N][CRT_N], int shiftadd) {
    int Q[CRT_NMOD][40];                                  // quarter-square LUTs (max 2*max_m)
    for (int mi = 0; mi < CRT_NMOD; mi++) {
        int m = CRT_MODULI[mi];
        for (int n = 0; n < 2 * m; n++) Q[mi][n] = crt_qsq(n);
    }
    for (int i = 0; i < CRT_N; i++)
        for (int j = 0; j < CRT_N; j++) {
            int r[CRT_NMOD];
            for (int mi = 0; mi < CRT_NMOD; mi++) {
                int m = CRT_MODULI[mi];
                const int *q = Q[mi];
                int acc = 0;
                for (int k = 0; k < CRT_N; k++) {         // K-deep dot, raw quarter-square products
                    int a = Aw[i][k] % m, b = Ad[k][j] % m;
                    int d = a - b; if (d < 0) d = -d;
                    acc += q[a + b] - q[d];               // == a*b, exactly
                }
                r[mi] = acc % m;                          // reduce once at the end
            }
            int value = shiftadd ? crt_recon_shiftadd(r[0], r[1], r[2])
                                 : CRT_COEFF[0] * r[0] + CRT_COEFF[1] * r[1] + CRT_COEFF[2] * r[2];
            C[i][j] = (crt_u16)(value % CRT_BIGN);
        }
}

// Compact validators: a 32-bit checksum + four spot values let the host confirm 1024 outputs cheaply.
static crt_u32 crt_checksum(crt_u16 C[CRT_N][CRT_N]) {
    crt_u32 s = 0;
    for (int i = 0; i < CRT_N; i++)
        for (int j = 0; j < CRT_N; j++) s += C[i][j];
    return s;
}
