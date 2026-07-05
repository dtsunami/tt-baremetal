// SPDX-License-Identifier: Apache-2.0
// nocwrite — bare-metal NOC0 write: a0=coord (y<<6|x), a1=dst local addr, a2=value. Writes one word
// (value) from local L1 to (coord):dst. BM_DBG gets [my_coord, coord, write-acks]. Enables the het
// CB consumer to ack the producer over the NoC.
#include "baremetal.h"
void bm_main(uint32_t coord, uint32_t dst, uint32_t value, uint32_t a3){
    (void)a3;
    volatile uint32_t* s=(volatile uint32_t*)(uintptr_t)(BM_RESULT+0x80u);
    s[0]=value;
    uint32_t ack = bm_noc0_write(coord, dst, (uint32_t)(BM_RESULT+0x80u), 4u);
    volatile uint32_t* d=(volatile uint32_t*)BM_DBG;
    d[0]=R(NOC_NODE_ID)&0xFFFu; d[1]=coord; d[2]=ack;
}
