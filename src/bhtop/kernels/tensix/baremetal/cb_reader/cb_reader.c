// cb_reader — bare-metal BRISC reader: NoC-read the 6 DYNAMIC render operands the x280 (het_x280) produced
// in GDDR, straight into this worker's L1 operand slots. One BRISC run streams a whole tile's operands; the
// resident TRISC render kernel consumes them. DRAM->CB reader of the fused flow — host stages ZERO operands.
//   a0 = source coord (bm_coord of the DRAM/x280 tile — PLACEMENT FLEXIBLE, any bank)
//   a1 = operand base addr in that tile's GDDR (0 -> default 0x30080000); operand slot i = base + i*0x800
//   a2 = noc select (0 -> noc0; noc1 hook reserved for congestion/latency routing)
#include "baremetal.h"
void bm_main(uint32_t coord, uint32_t base, uint32_t noc, uint32_t a3){
    (void)noc; (void)a3;
    if(!base) base = 0x30080000u;
    // worker-L1 dst for each operand (fixed = resident_train_perf's H_psi/H_Dop/H_Dnop/H_color/H_colorT/H_opB)
    static const uint32_t DST[6] = {0x00022000u,0x00024000u,0x00025000u,0x00028000u,0x0002B000u,0x0002A000u};
    uint32_t resp = 0;
    for(int i=0;i<6;i++) resp += bm_noc0_read(coord, base + (uint32_t)i*0x800u, DST[i], 2048u);
    volatile uint32_t* dbg=(volatile uint32_t*)BM_DBG;
    dbg[0]=coord; dbg[1]=base; dbg[2]=resp;
}
