// SPDX-License-Identifier: Apache-2.0
// hello — bare-metal proof-of-execution: write a magic ramp to BM_RESULT, then park.
#include "baremetal.h"
void bm_main(uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3){
    volatile uint32_t* p=(volatile uint32_t*)BM_RESULT;
    for(uint32_t i=0;i<8u;i++) p[i]=0xB16B00B5u+i;
    (void)a0;(void)a1;(void)a2;(void)a3;
}
