// conductor — RESIDENT BRISC tile conductor for the x280-ORCHESTRATED path. Host is OUT of the per-tile
// control loop. The conductor polls a flag in the x280's GDDR (NoC read); when the x280 raises it, the
// conductor runs the WHOLE tile locally: NoC-read operands (x280->L1), drive the resident TRISC render for
// NG pixel-groups, NoC-write each group's grads to the x280's GDDR inbox, then ack the tile. The x280 only
// sets the flag and reads the ack — no x280->worker writes (which would need a TLB).
//
// KEY (2026-07-06): the conductor(BRISC) and render(TRISC) are on the SAME tile sharing ONE L1. On-tile
// signalling goes through L1 DIRECTLY — a local store+fence to ring, a local load+fence to poll DONE, a
// local memcpy+fence to stage operands. NO NoC loopback (self-tile NoC is wrong AND slow; that was the
// bug that stalled rdone at 0). NoC is used ONLY for the genuinely-remote x280 GDDR hub. Coherency:
// baby-RISC store reaches L1 (proven — the render's DONE store is seen by the host); the reader's fence
// (=invalidate_data_cache on Blackhole) then sees fresh L1. Same contract both directions.
//   L1 cfg block @0x3200: [hub, slot, opbase, ginbase, flag, ack, pxbase, -, -, NG]
//   pxbase = x280 GDDR region holding this tile's phi/phi2T/gt (produced by het cmd8); the conductor
//   NoC-reads them per group like the 6 operands, so NO host pre-stage of pixel/target data.
#include "baremetal.h"
#define CFG 0x00003200u
#define SC  0x00003100u
#define R_DB   0x00016000u
#define R_DONE 0x00016010u
#define H_PHI  0x00021000u
#define H_PHI2T 0x0002E000u
#define H_GT   0x00029000u
static inline void fence(void){ __asm__ __volatile__("fence" ::: "memory"); }

void bm_main(uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3){
    (void)a0;(void)a1;(void)a2;(void)a3;
    volatile uint32_t* cfg=(volatile uint32_t*)CFG;
    volatile uint32_t* sc =(volatile uint32_t*)SC;
    volatile uint32_t* rdb =(volatile uint32_t*)R_DB;      // render doorbell  (LOCAL L1)
    volatile uint32_t* rdn =(volatile uint32_t*)R_DONE;    // render done flag (LOCAL L1)
    uint32_t hub=cfg[0], opbase=cfg[2], ginbase=cfg[3], flag=cfg[4], ack=cfg[5];
    uint32_t pxbase=cfg[6], ng=cfg[9];                     // pxbase = x280 GDDR phi/phi2T/gt (het cmd8)
    static const uint32_t OP_DST[6]={0x00022000u,0x00024000u,0x00025000u,0x00028000u,0x0002B000u,0x0002A000u};
    static const uint32_t GR_SRC[4]={0x00052000u,0x00051000u,0x00042000u,0x00043000u};
    volatile uint32_t* dbg=(volatile uint32_t*)BM_DBG;
    uint32_t last=0u, nflag=0u;
    for(;;){
        bm_noc0_read(hub, flag, SC, 4u);                  // poll x280 flag (REMOTE, NoC)
        uint32_t f=sc[0];
        dbg[0]=f; dbg[1]=last;
        if(f==last) continue;
        dbg[2]=++nflag;
        for(int i=0;i<6;i++) bm_noc0_read(hub, opbase+(uint32_t)i*0x800u, OP_DST[i], 2048u);   // operands x280->L1 (NoC)
        for(uint32_t g=0; g<ng; g++){
            // group g's phi/phi2T/gt live in x280 GDDR (het cmd8) — NoC-read them into render L1 inputs
            uint32_t pxg = pxbase + g*3u*0x800u;
            bm_noc0_read(hub, pxg+0u*0x800u, H_PHI,   2048u);
            bm_noc0_read(hub, pxg+1u*0x800u, H_PHI2T, 2048u);
            bm_noc0_read(hub, pxg+2u*0x800u, H_GT,    2048u);
            // ring the render: ring = current DONE + 1, LOCAL store, then poll DONE == ring (LOCAL load)
            fence(); uint32_t ring = rdn[0] + 1u;
            rdb[0] = ring; fence();                        // <-- LOCAL doorbell store (no NoC loopback)
            dbg[7]=rdb[0];                                 // readback the ring we just wrote
            uint32_t to=0u, d;
            do { fence(); d = rdn[0]; } while(d != ring && ++to < 2000000u);
            dbg[3]=g+1; dbg[4]=d; dbg[5]=ring;
            for(int i=0;i<4;i++) bm_noc0_write(hub, ginbase+g*0x2000u+(uint32_t)i*0x800u, GR_SRC[i], 2048u); // grads->x280 (NoC)
        }
        sc[1]=f; bm_noc0_write(hub, ack, SC+4u, 4u);      // ack the whole tile (REMOTE, NoC)
        dbg[6]=f; last=f;
    }
}
