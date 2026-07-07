// conductor_probe — de-risk the x280<->worker orchestration handshake. A resident BRISC loop POLLS a flag
// in the x280's GDDR (NoC read — the proven direction; x280->worker writes would need a TLB), and when it
// changes, writes an ack back into the x280's GDDR (NoC write). This is how the x280 will drive the tile
// loop with the host out of the control path: x280 sets flag -> worker acts -> worker acks -> x280 sees ack.
//   boot params: a0=hub_coord (bm_coord of the x280), a1=flag addr (x280 GDDR), a2=ack addr (x280 GDDR)
#include "baremetal.h"
#define SC 0x00003100u
void bm_main(uint32_t hub, uint32_t flag, uint32_t ack, uint32_t a3){
    (void)a3;
    volatile uint32_t* sc=(volatile uint32_t*)SC;
    volatile uint32_t* dbg=(volatile uint32_t*)BM_DBG;
    uint32_t last=0u, n=0u;
    for(;;){
        bm_noc0_read(hub, flag, SC, 4u);            // fresh NoC fetch of the remote flag (no cache staleness)
        uint32_t f=sc[0];
        if(f!=last){
            sc[1]=f; bm_noc0_write(hub, ack, SC+4u, 4u);   // ack = flag, back into x280 GDDR
            last=f; dbg[0]=++n; dbg[1]=f;
        }
    }
}
