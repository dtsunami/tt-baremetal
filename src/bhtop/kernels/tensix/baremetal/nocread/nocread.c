// SPDX-License-Identifier: Apache-2.0
// nocread — bare-metal NOC0 read: a0=coord (y<<6|x), a1=src local addr, a2=len bytes, a3=dst L1 addr
// (0 -> BM_RESULT). BM_DBG gets [my_coord, target_coord, responses]. Proven silicon: x280->Tensix
// handoff. With a3=dst the reader lands the payload directly in a matmul input (e.g. PERF_INPUT_B).
#include "baremetal.h"
void bm_main(uint32_t coord, uint32_t src, uint32_t len, uint32_t dst){
    if(!len) len=32u;
    uint32_t d = dst ? dst : BM_RESULT;
    if(d==BM_RESULT){ volatile uint32_t* r=(volatile uint32_t*)BM_RESULT; for(uint32_t i=0;i<8u;i++) r[i]=0xEEEEEEEEu; }
    uint32_t resp = bm_noc0_read(coord, src, d, len);
    volatile uint32_t* dbg=(volatile uint32_t*)BM_DBG;
    dbg[0]=R(NOC_NODE_ID)&0xFFFu; dbg[1]=coord; dbg[2]=resp;
}
