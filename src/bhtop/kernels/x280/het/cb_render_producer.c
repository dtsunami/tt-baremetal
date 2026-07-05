/* cb_render_producer.c — x280 PRODUCER for the fused streaming splat pipeline. Host stages T tiles'
 * per-Gaussian depths at INPUT (uncached). For each tile the x280 argsorts front-to-back and publishes
 * the depth order into a bounded GDDR ring; it BLOCKS when the ring is full (backpressure) waiting on
 * the acked counter the Tensix render bumps after consuming a tile. The Tensix forward is the consumer.
 *   P 0x30002000 · A 0x30002010 · PDONE 0x30002020 · INPUT 0x30002040 (T×K depths) · RING 0x30002200 (N×64B) */
#include <stdint.h>
int main(void) {
    volatile uint32_t *P = (volatile uint32_t *)0x30002000u;
    volatile uint32_t *A = (volatile uint32_t *)0x30002010u;
    volatile uint32_t *PDONE = (volatile uint32_t *)0x30002020u;
    volatile uint32_t *IN   = (volatile uint32_t *)0x30002040u;   /* T*K depths (host-written) */
    volatile uint32_t *RING = (volatile uint32_t *)0x30002200u;   /* N slots of K order words */
    const uint32_t N = 2u, T = 4u, K = 16u;
    for (uint32_t t = 0; t < T; t++) {
        while ((*P - *A) >= N) { }                                /* backpressure: ring full */
        volatile uint32_t *din = IN + t * K;
        uint32_t z[16]; int idx[16];
        for (uint32_t i = 0; i < K; i++) { z[i] = din[i]; idx[i] = i; }
        for (uint32_t i = 1; i < K; i++) {                        /* argsort by ascending u32 depth */
            int key = idx[i]; uint32_t kz = z[key]; int j = i - 1;
            while (j >= 0 && z[idx[j]] > kz) { idx[j + 1] = idx[j]; j--; }
            idx[j + 1] = key;
        }
        volatile uint32_t *slot = RING + (t % N) * 16u;           /* 16 words = 64B/slot */
        for (uint32_t i = 0; i < K; i++) slot[i] = (uint32_t)idx[i];
        *P = t + 1u;                                              /* publish tile t's order */
    }
    *PDONE = 0xD09Eu;
    for (;;) { }
    return 0;
}
