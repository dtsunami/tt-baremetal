/* cb_operands.c — x280 produces ALL depth-ordered, tilized dense render operands in shared GDDR:
 * psi (eval), diag(op) & diag(-op) (opacity), color (composite). Host provides UNORDERED per-Gaussian
 * bf16 coeffs (sa,m12,m22,c1,c2,op,r,g,b) + depths; the x280 sorts, gathers, and lays each into the
 * exact tilized bf16 32x32 layout the Tensix matmuls read. Tensix NoC-reads each into PERF_INPUT_B —
 * the host relays ZERO Gaussian data.
 *   ZIN 0x30002300 [K,z(K)] · PIN 0x30002400 (9 bf16/Gaussian) · PSI 0x30003000 · DOP 0x30003800
 *   DNOP 0x30004000 · COLOR 0x30004800 (each a 512-word tile) */
#include <tele.h>
#include <stdint.h>
#define ZIN   ((volatile uint32_t *)0x30002300u)
#define PIN   ((volatile uint32_t *)0x30002400u)
#define PSI   ((volatile uint32_t *)0x30003000u)
#define DOP   ((volatile uint32_t *)0x30003800u)
#define DNOP  ((volatile uint32_t *)0x30004000u)
#define COLOR ((volatile uint32_t *)0x30004800u)

static inline void place(volatile uint32_t *t, int row, int col, uint32_t bf) {
    int face = ((row >= 16) ? 2 : 0) + ((col >= 16) ? 1 : 0);
    int e = face * 256 + (row % 16) * 16 + (col % 16), wd = e >> 1, h = e & 1;
    uint32_t cur = t[wd];
    t[wd] = h ? ((cur & 0x0000FFFFu) | (bf << 16)) : ((cur & 0xFFFF0000u) | bf);
}

int main(void) {
    int K = (int)ZIN[0];
    if (K < 0 || K > 16) K = 0;
    uint32_t z[16]; int idx[16];
    for (int i = 0; i < K; i++) { z[i] = ZIN[1 + i]; idx[i] = i; }
    for (int i = 1; i < K; i++) {
        int key = idx[i]; uint32_t kz = z[key]; int j = i - 1;
        while (j >= 0 && z[idx[j]] > kz) { idx[j + 1] = idx[j]; j--; }
        idx[j + 1] = key;
    }
    for (int w = 0; w < 512; w++) { PSI[w] = 0; DOP[w] = 0; DNOP[w] = 0; COLOR[w] = 0; }
    for (int i = 0; i < K; i++) {
        int g = idx[i];
        uint32_t sa = PIN[g*9+0]&0xFFFFu, m12 = PIN[g*9+1]&0xFFFFu, m22 = PIN[g*9+2]&0xFFFFu,
                 c1 = PIN[g*9+3]&0xFFFFu, c2 = PIN[g*9+4]&0xFFFFu, op = PIN[g*9+5]&0xFFFFu,
                 rr = PIN[g*9+6]&0xFFFFu, gg = PIN[g*9+7]&0xFFFFu, bb = PIN[g*9+8]&0xFFFFu;
        place(PSI, 0, 2*i,   sa);  place(PSI, 1, 2*i,   m12); place(PSI, 2, 2*i,   c1);   /* v1 coeffs */
        place(PSI, 1, 2*i+1, m22); place(PSI, 2, 2*i+1, c2);                              /* v2 coeffs */
        place(DOP, i, i, op);  place(DNOP, i, i, op ^ 0x8000u);                           /* diag(±op) */
        place(COLOR, i, 0, rr); place(COLOR, i, 1, gg); place(COLOR, i, 2, bb);           /* rgb row */
    }
    TELE[0] = 0x4F505253u;   /* 'OPRS' done */
    for (;;) { }
    return 0;
}
