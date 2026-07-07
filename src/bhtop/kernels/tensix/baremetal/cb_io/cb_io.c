// cb_io — RESIDENT, SWAPPABLE bare-metal BRISC I/O engine for the fused splat grid. Boots ONCE, then loops
// on a doorbell: reads its config from L1 (source coord, GDDR base, mode) and does the NoC transfer — no
// per-call ELF reload (which was the 1.8s grid bottleneck). Config is host/x280-pokeable so PLACEMENT is
// live-flexible (migrate bank / NoC route by poking cfg, not reloading). BRISC is free while the render
// kernel runs on TRISC0/1/2 — reader on BRISC, compute on TRISC, the tt-metal reader/compute split.
//   L1 doorbell: IO_DB 0x3000 (ring) · IO_DONE 0x3010 (ack) · IO_CFG 0x3020 [coord, base, mode] (mode 0=read
//   6 operands x280->L1, 1=write 4 grads L1->x280). Operand/grad L1 slots = resident_train_perf's.
#include "baremetal.h"
#define IO_DB   0x00003000u
#define IO_DONE 0x00003010u
#define IO_CFG  0x00003020u
static const uint32_t OP_DST[6]={0x00022000u,0x00024000u,0x00025000u,0x00028000u,0x0002B000u,0x0002A000u};
static const uint32_t GR_SRC[4]={0x00052000u,0x00051000u,0x00042000u,0x00043000u};

void bm_main(uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3){
    (void)a0;(void)a1;(void)a2;(void)a3;
    volatile uint32_t* db  =(volatile uint32_t*)IO_DB;
    volatile uint32_t* done=(volatile uint32_t*)IO_DONE;
    volatile uint32_t* cfg =(volatile uint32_t*)IO_CFG;
    done[0]=0u;
    uint32_t last=0u;
    for(;;){
        __asm__ volatile("fence");                 // see host/x280 NoC writes to the doorbell (no stale line)
        uint32_t r=db[0];
        if(r==last) continue;
        uint32_t coord=cfg[0], base=cfg[1], mode=cfg[2];
        if(mode==0u){ for(int i=0;i<6;i++) bm_noc0_read (coord, base+(uint32_t)i*0x800u, OP_DST[i], 2048u); }
        else        { for(int i=0;i<4;i++) bm_noc0_write(coord, base+(uint32_t)i*0x800u, GR_SRC[i], 2048u); }
        last=r; done[0]=r;
    }
}
