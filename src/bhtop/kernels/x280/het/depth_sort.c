/* depth_sort.c — x280 owns the IRREGULAR tier: argsort Gaussians front-to-back by depth.
 * Host writes [K, z0..z_{K-1}] (float bits) into hart-3's telemetry window (0x30002300, UNCACHED so
 * the hart sees host writes). For NON-NEGATIVE floats the IEEE-754 u32 bit pattern sorts in the same
 * order as the value, so we sort the u32s as unsigned ints — NO scalar float (avoids the FPU-not-enabled
 * trap; crt0 enables the vector unit, not mstatus.FS). Sorted index list -> hart-0 telemetry window. */
#include <tele.h>
#include <stdint.h>

#define IN ((volatile uint32_t *)0x30002300u)   /* uncached: hart-3 tele window, host-written */

int main(void) {
    TELE[0] = 0xAAAA;                            /* early marker */
    int K = (int)IN[0];
    if (K < 0 || K > 60) K = 0;
    uint32_t z[64];
    int idx[64];
    for (int i = 0; i < K; i++) { z[i] = IN[1 + i]; idx[i] = i; }
    for (int i = 1; i < K; i++) {                /* insertion sort by ascending u32 (== ascending z>=0) */
        int key = idx[i]; uint32_t kz = z[key]; int j = i - 1;
        while (j >= 0 && z[idx[j]] > kz) { idx[j + 1] = idx[j]; j--; }
        idx[j + 1] = key;
    }
    TELE[1] = (uint32_t)K;
    for (int i = 0; i < K; i++) TELE[2 + i] = (uint32_t)idx[i];
    TELE[0] = 0x50575254u;                       /* done magic */
    uint32_t hb = 0;
    for (;;) { TELE[63] = ++hb; }                /* liveness */
    return 0;
}
