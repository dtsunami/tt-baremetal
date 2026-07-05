/* cb_color_tilize.c — x280 produces a RENDER-READY dense operand: gathers the depth-ordered color and
 * lays it out in the exact tilized bf16 32x32 layout the Tensix matmul reads (color[i][c] at face-0
 * element i*16+c, 2 bf16/word). Host provides UNORDERED per-Gaussian color (bf16) + depths; the x280
 * sorts, gathers, tilizes, and writes the tile to shared uncached GDDR. Tensix NoC-reads it straight
 * into PERF_INPUT_B — no host relay of the operand.
 *   ZIN 0x30002300 [K, z(K)] · CIN 0x30002400 color bf16 (3K words) · OUT 0x30003000 (512-word tile) */
#include <tele.h>
#include <stdint.h>
#define ZIN ((volatile uint32_t *)0x30002300u)
#define CIN ((volatile uint32_t *)0x30002400u)
#define OUT ((volatile uint32_t *)0x30003000u)

int main(void) {
    int K = (int)ZIN[0];
    if (K < 0 || K > 16) K = 0;
    uint32_t z[16]; int idx[16];
    for (int i = 0; i < K; i++) { z[i] = ZIN[1 + i]; idx[i] = i; }
    for (int i = 1; i < K; i++) {                       /* sort front-to-back */
        int key = idx[i]; uint32_t kz = z[key]; int j = i - 1;
        while (j >= 0 && z[idx[j]] > kz) { idx[j + 1] = idx[j]; j--; }
        idx[j + 1] = key;
    }
    for (int w = 0; w < 512; w++) OUT[w] = 0u;          /* zero the tile (padding) */
    for (int i = 0; i < K; i++) {                       /* gather + tilize into face-0 rows */
        int g = idx[i];
        for (int c = 0; c < 3; c++) {
            uint32_t bf = CIN[g * 3 + c] & 0xFFFFu;      /* bf16 value (low 16) */
            int e = i * 16 + c;                          /* tilized element: row i, col c, face 0 */
            int wd = e >> 1, half = e & 1;
            uint32_t cur = OUT[wd];
            OUT[wd] = half ? ((cur & 0x0000FFFFu) | (bf << 16)) : ((cur & 0xFFFF0000u) | bf);
        }
    }
    TELE[0] = 0x434F4C52u;                              /* 'COLR' done */
    for (;;) { }
    return 0;
}
