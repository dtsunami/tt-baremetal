/* depth_sort_multi.c — MULTI-hart x280 irregular tier. Each hart sorts its OWN scene's Gaussians in
 * parallel. Host packs [nH, K, tile0_z(K), tile1_z(K), ...] at 0x30002300 (uncached). Hart h reads its
 * K depths at offset 2 + h*K, argsorts by ascending u32 (== ascending non-negative float), and writes
 * the sorted index list to ITS OWN telemetry window (TELE auto-targets the running hart). */
#include <tele.h>
#include <stdint.h>

#define IN ((volatile uint32_t *)0x30002300u)

int main(void) {
    unsigned h = bh__hartid();
    int K = (int)IN[1];
    if (K < 0 || K > 20) K = 0;
    volatile uint32_t *zin = IN + 2 + h * K;
    uint32_t z[24];
    int idx[24];
    for (int i = 0; i < K; i++) { z[i] = zin[i]; idx[i] = i; }
    for (int i = 1; i < K; i++) {
        int key = idx[i]; uint32_t kz = z[key]; int j = i - 1;
        while (j >= 0 && z[idx[j]] > kz) { idx[j + 1] = idx[j]; j--; }
        idx[j + 1] = key;
    }
    TELE[1] = (uint32_t)K;
    for (int i = 0; i < K; i++) TELE[2 + i] = (uint32_t)idx[i];
    TELE[0] = 0x50575254u;
    uint32_t hb = 0;
    for (;;) { TELE[63] = ++hb; }
    return 0;
}
