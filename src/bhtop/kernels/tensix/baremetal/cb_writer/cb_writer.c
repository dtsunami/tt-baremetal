// cb_writer — bare-metal BRISC writer: NoC-write this worker's render grad tiles from L1 to the x280/DRAM
// (a0=coord) GDDR inbox, so the x280 consumes them WITHOUT the host reading grads back. Reverse of cb_reader.
//   a0 = dest coord (bm_coord of the DRAM/x280 tile — PLACEMENT FLEXIBLE, any bank)
//   a1 = grad inbox base in that tile's GDDR (0 -> default 0x30088000); grad slot i = base + i*0x800
//   a2 = noc select (0 -> noc0; noc1 hook reserved)
// Grad tiles (worker L1 -> GDDR): dLdpsi, dLdop, w, dLdC (2048 B each, bf16 32x32).
#include "baremetal.h"
void bm_main(uint32_t coord, uint32_t base, uint32_t noc, uint32_t a3){
    (void)noc; (void)a3;
    if(!base) base = 0x30088000u;
    static const uint32_t SRC[4] = {0x00052000u,0x00051000u,0x00042000u,0x00043000u}; // dLdpsi,dLdop,w,dLdC
    uint32_t resp = 0;
    for(int i=0;i<4;i++) resp += bm_noc0_write(coord, base + (uint32_t)i*0x800u, SRC[i], 2048u);
    volatile uint32_t* dbg=(volatile uint32_t*)BM_DBG;
    dbg[0]=coord; dbg[1]=base; dbg[2]=resp;
}
