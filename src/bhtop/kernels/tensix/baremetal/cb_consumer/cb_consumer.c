// SPDX-License-Identifier: Apache-2.0
// cb_consumer — Tensix CONSUMER of the het DRAM circular buffer (M2). Polls the producer's `produced`
// counter in x280 GDDR (NoC-read), drains each ring slot as it fills, processes it (checksum), and acks
// the producer by NoC-writing the `acked` counter. Backpressure is closed by the producer waiting on
// that ack. a0=coord (x280 (8,3)), a1=T (items), a2=N (ring slots).
//   P 0x30002000 · A 0x30002010 · RING 0x30002080 (64B slots).  Per-item checksums -> BM_DBG+0x40.
#include "baremetal.h"
void bm_main(uint32_t coord, uint32_t T, uint32_t N, uint32_t a3){
    (void)a3;
    volatile uint32_t* pbuf =(volatile uint32_t*)(uintptr_t)(BM_RESULT+0x200u);
    volatile uint32_t* slot =(volatile uint32_t*)(uintptr_t)(BM_RESULT+0x240u);
    volatile uint32_t* ackb =(volatile uint32_t*)(uintptr_t)(BM_RESULT+0x300u);
    volatile uint32_t* sums =(volatile uint32_t*)(uintptr_t)(BM_DBG+0x40u);
    uint32_t consumed=0u, polls=0u;
    while(consumed < T){
        uint32_t p;                                          // wait until produced > consumed
        do { bm_noc0_read(coord, 0x30002000u, (uint32_t)(BM_RESULT+0x200u), 64u); p=pbuf[0]; polls++; }
        while(p <= consumed);
        uint32_t slotaddr = 0x30002080u + (consumed % N) * 64u;
        bm_noc0_read(coord, slotaddr, (uint32_t)(BM_RESULT+0x240u), 64u);   // drain slot (16 words)
        uint32_t cs=0u; for(uint32_t w=0; w<16u; w++) cs += slot[w];        // process (checksum)
        sums[consumed]=cs;
        consumed++;
        ackb[0]=consumed;                                                   // ack -> free the slot
        bm_noc0_write(coord, 0x30002010u, (uint32_t)(BM_RESULT+0x300u), 4u);
    }
    ((volatile uint32_t*)(uintptr_t)BM_DBG)[0]=polls;                       // total P-polls (backpressure evidence)
    ((volatile uint32_t*)(uintptr_t)BM_RESULT)[0]=0xC0DE0000u | consumed;   // done marker
}
