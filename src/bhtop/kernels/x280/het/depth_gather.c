/* depth_gather.c — x280 irregular tier: SORT + GATHER. Reads [K, z(K), rgb(3K)] from 0x30002100
 * (uncached), argsorts front-to-back by depth, and writes the DEPTH-ORDERED rgb records to its hart-0
 * telemetry window (0x30002000, uncached GDDR) as [magic, K, ordered_rgb(3K)]. The Tensix grid then
 * NoC-reads that buffer directly from GDDR (coord (8,3)) — the host never relays the gathered data. */
#include <tele.h>
#include <stdint.h>

#define ZIN  ((volatile uint32_t *)0x30002300u)  /* uncached hart-3 window: [K, z(K)]  */
#define RGB  ((volatile uint32_t *)0x30002100u)  /* uncached hart-1 window: rgb(3K), 48 words fits */

int main(void) {
    int K = (int)ZIN[0];
    if (K < 0 || K > 16) K = 0;
    uint32_t z[16]; int idx[16];
    volatile uint32_t *rgb = RGB;
    for (int i = 0; i < K; i++) { z[i] = ZIN[1 + i]; idx[i] = i; }
    for (int i = 1; i < K; i++) {                /* sort by ascending u32 (== ascending non-neg depth) */
        int key = idx[i]; uint32_t kz = z[key]; int j = i - 1;
        while (j >= 0 && z[idx[j]] > kz) { idx[j + 1] = idx[j]; j--; }
        idx[j + 1] = key;
    }
    TELE[1] = (uint32_t)K;
    for (int i = 0; i < K; i++) {                /* GATHER: copy each gaussian's rgb into depth order */
        int g = idx[i];
        TELE[2 + i * 3 + 0] = rgb[g * 3 + 0];
        TELE[2 + i * 3 + 1] = rgb[g * 3 + 1];
        TELE[2 + i * 3 + 2] = rgb[g * 3 + 2];
    }
    TELE[0] = 0x47415448u;                        /* 'GATH' done magic */
    for (;;) { }
    return 0;
}
