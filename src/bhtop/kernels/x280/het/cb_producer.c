/* cb_producer.c — x280 PRODUCER of a het DRAM circular buffer (M2). Fills a bounded ring in uncached
 * GDDR and publishes via a monotonic `produced` counter; blocks (backpressure) when the ring is full,
 * waiting on the `acked` counter the Tensix consumer writes over the NoC. All state in the uncached
 * telemetry region so both engines see each other's writes coherently. Host inits P=A=0 before launch.
 *   P (produced) 0x30002000 · A (acked) 0x30002010 · PDONE 0x30002020 · RING 0x30002080 (N×64B slots) */
#include <stdint.h>
int main(void) {
    volatile uint32_t *P = (volatile uint32_t *)0x30002000u;
    volatile uint32_t *A = (volatile uint32_t *)0x30002010u;
    volatile uint32_t *PDONE = (volatile uint32_t *)0x30002020u;
    volatile uint32_t *RING = (volatile uint32_t *)0x30002080u;
    const uint32_t N = 4u, T = 12u, SLOTW = 16u;      /* 4 slots, 12 items, 16 words/slot */
    for (uint32_t i = 0; i < T; i++) {
        while ((*P - *A) >= N) { }                     /* backpressure: ring full -> wait for consumer */
        volatile uint32_t *slot = RING + (i % N) * SLOTW;
        for (uint32_t w = 0; w < SLOTW; w++) slot[w] = i * 100u + w;   /* item i payload (verifiable) */
        *P = i + 1u;                                    /* publish (uncached writes are ordered) */
    }
    *PDONE = 0xD09Eu;
    for (;;) { }
    return 0;
}
