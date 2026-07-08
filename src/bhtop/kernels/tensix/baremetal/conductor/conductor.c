// conductor — RESIDENT BRISC tile conductor for the x280-ORCHESTRATED path. Host is OUT of the per-tile
// control loop. The conductor polls a flag in the x280's GDDR (NoC read); when the x280 raises it, the
// conductor runs the WHOLE tile locally: (produce the render inputs), drive the resident TRISC render for
// NG pixel-groups, NoC-write each group's grads to the x280's GDDR inbox, then ack the tile. The x280 only
// sets the flag and reads the ack — no x280->worker writes (which would need a TLB).
//
// KEY (2026-07-06): the conductor(BRISC) and render(TRISC) are on the SAME tile sharing ONE L1. On-tile
// signalling goes through L1 DIRECTLY. NoC is used ONLY for the genuinely-remote x280 GDDR hub.
//
// W4 WORKER-PRODUCE (cfg[10]=wprod): instead of NoC-reading x280-PRE-TILIZED operands+pixels, the conductor
// tilizes its OWN tile locally on the BRISC (pure integer — no FPU): NoC-read a compact coeff DESCRIPTOR +
// the raw gt patch, scatter the 6 operand tiles, GENERATE phi/phi2T analytically from (ox,oy), and f32->bf16
// tilize gt. Moves the 30% "produce" bucket off the 4-hart x280 hub onto the 120 formerly-idle workers.
// Must be BIT-EXACT vs het_x280.c produce_ops/produce_pix (tput/f2bf ported verbatim, integer-only).
//   L1 cfg block @0x3200: [hub, slot, opbase, ginbase, flag, ack, pxbase, desc, img_base, NG, wprod, imgw]
#include "baremetal.h"
#define CFG 0x00003200u
#define SC  0x00003100u
#define R_DB   0x00016000u
#define R_DONE 0x00016010u
#define H_PHI  0x00021000u
#define H_PHI2T 0x0002E000u
#define H_GT   0x00029000u
#define DESC_L 0x00008000u     // local scratch: NoC-read descriptor [K,ox,oy,coeff[12*9]]
#define GT_L   0x00009000u     // local scratch: raw gt patch (16*16*3 f32 = 3072B)
#define DESC_STRIDE 0x800u     // must match het_x280.c DESC_STRIDE
static inline void fence(void){ __asm__ __volatile__("fence" ::: "memory"); }

// --- bit-exact tile put (ported verbatim from het_x280.c:tput — pure integer, 4-face 32x32 bf16) ---
static inline void tput(volatile uint32_t* t,int row,int col,uint32_t bf){
    int face=((row>=16)?2:0)+((col>=16)?1:0); int e=face*256+(row%16)*16+(col%16); int wd=e>>1;
    uint32_t cur=t[wd]; t[wd]=(e&1)?((cur&0x0000FFFFu)|(bf<<16)):((cur&0xFFFF0000u)|bf);
}
// f2bf on raw f32 bits (ported from het_x280.c:f2bf; no float locals — BRISC is rv32im, no F ext)
static inline uint32_t f2bf_bits(uint32_t b){ b += 0x7FFFu + ((b>>16)&1u); return (b>>16)&0xFFFFu; }
// unsigned int -> IEEE754 f32 bits (exact for n < 2^24; pixel coords qualify)
static inline uint32_t u32_to_f32bits(uint32_t n){
    if(n==0u) return 0u;
    int e=31; while(!((n>>e)&1u)) e--;
    uint32_t exp=(uint32_t)(e+127);
    uint32_t mant = (e<=23) ? ((n<<(23-e))&0x7FFFFFu) : ((n>>(e-23))&0x7FFFFFu);
    return (exp<<23)|mant;
}
static inline uint32_t int_to_bf16(uint32_t n){ return f2bf_bits(u32_to_f32bits(n)); }
static inline void zero_tile(volatile uint32_t* t){ for(int w=0;w<512;w++) t[w]=0u; }

void bm_main(uint32_t a0,uint32_t a1,uint32_t a2,uint32_t a3){
    (void)a0;(void)a1;(void)a2;(void)a3;
    volatile uint32_t* cfg=(volatile uint32_t*)CFG;
    volatile uint32_t* sc =(volatile uint32_t*)SC;
    volatile uint32_t* rdb =(volatile uint32_t*)R_DB;      // render doorbell  (LOCAL L1)
    volatile uint32_t* rdn =(volatile uint32_t*)R_DONE;    // render done flag (LOCAL L1)
    uint32_t hub=cfg[0], opbase=cfg[2], ginbase=cfg[3], flag=cfg[4], ack=cfg[5];
    uint32_t pxbase=cfg[6], desc=cfg[7], img_base=cfg[8], ng=cfg[9], wprod=cfg[10], imgw=cfg[11];
    static const uint32_t OP_DST[6]={0x00022000u,0x00024000u,0x00025000u,0x00028000u,0x0002B000u,0x0002A000u};
    static const uint32_t GR_SRC[4]={0x00052000u,0x00051000u,0x00042000u,0x00043000u};
    volatile uint32_t* dbg=(volatile uint32_t*)BM_DBG;
    volatile uint32_t* PSI =(volatile uint32_t*)OP_DST[0], *DOP=(volatile uint32_t*)OP_DST[1];
    volatile uint32_t* DNOP=(volatile uint32_t*)OP_DST[2], *COL=(volatile uint32_t*)OP_DST[3];
    volatile uint32_t* COLT=(volatile uint32_t*)OP_DST[4], *OPB=(volatile uint32_t*)OP_DST[5];
    volatile uint32_t* ph =(volatile uint32_t*)H_PHI, *p2=(volatile uint32_t*)H_PHI2T, *gt=(volatile uint32_t*)H_GT;
    volatile uint32_t* dl =(volatile uint32_t*)DESC_L;
    volatile uint32_t* gl =(volatile uint32_t*)GT_L;
    uint32_t last=0u, nflag=0u;
    for(;;){
        bm_noc0_read(hub, flag, SC, 4u);                  // poll x280 flag (REMOTE, NoC)
        uint32_t f=sc[0];
        dbg[0]=f; dbg[1]=last;
        if(f==last) continue;
        dbg[2]=++nflag;
        int ox=0, oy=0;
        if(!wprod){
            for(int i=0;i<6;i++) bm_noc0_read(hub, opbase+(uint32_t)i*0x800u, OP_DST[i], 2048u);   // pre-tilized operands
        } else {
            // ---- WORKER PRODUCE: read compact descriptor, scatter the 6 operand tiles locally (BIT-EXACT) ----
            bm_noc0_read(hub, desc, DESC_L, DESC_STRIDE);          // [K,ox,oy,coeff[12*9]] (depth-sorted)
            int K=(int)dl[0]; if(K>16)K=16; ox=(int)dl[1]; oy=(int)dl[2];
            zero_tile(PSI); zero_tile(DOP); zero_tile(DNOP); zero_tile(COL); zero_tile(COLT); zero_tile(OPB);
            for(int i=0;i<K;i++){ volatile uint32_t* co=dl+3+(uint32_t)i*9u;
                uint32_t sa=co[0],m12=co[1],m22=co[2],c1=co[3],c2=co[4],op=co[5],rr=co[6],gg=co[7],bb=co[8];
                tput(PSI,0,2*i,sa); tput(PSI,1,2*i,m12); tput(PSI,2,2*i,c1); tput(PSI,1,2*i+1,m22); tput(PSI,2,2*i+1,c2);
                tput(DOP,i,i,op); tput(DNOP,i,i,op^0x8000u); tput(COL,i,0,rr); tput(COL,i,1,gg); tput(COL,i,2,bb);
                tput(COLT,0,i,rr); tput(COLT,1,i,gg); tput(COLT,2,i,bb); for(int p=0;p<32;p++) tput(OPB,p,i,op); }
            for(int k=K;k<32;k++) for(int p=0;p<32;p++) tput(OPB,p,k,0x3F00u);
            // read the whole 16x16x3 gt patch once (16 strided rows) into local scratch
            for(int r=0;r<16;r++) bm_noc0_read(hub, img_base + (uint32_t)(((oy+r)*(int)imgw+ox)*3)*4u, GT_L + (uint32_t)r*16u*3u*4u, 16u*3u*4u);
        }
        for(uint32_t g=0; g<ng; g++){
            if(!wprod){
                uint32_t pxg = pxbase + g*3u*0x800u;
                bm_noc0_read(hub, pxg+0u*0x800u, H_PHI,   2048u);
                bm_noc0_read(hub, pxg+1u*0x800u, H_PHI2T, 2048u);
                bm_noc0_read(hub, pxg+2u*0x800u, H_GT,    2048u);
            } else {
                // ---- generate phi/phi2T analytically + tilize gt from the local patch (BIT-EXACT vs produce_pix) ----
                zero_tile(ph); zero_tile(p2); zero_tile(gt);
                for(int p=0;p<32;p++){
                    int idx=(int)g*32+p, lx=idx%16, ly=idx/16; uint32_t px=(uint32_t)(ox+lx), py=(uint32_t)(oy+ly);
                    tput(ph,p,0,int_to_bf16(px)); tput(ph,p,1,int_to_bf16(py)); tput(ph,p,2,0x3F80u);        // 1.0
                    tput(p2,0,p,int_to_bf16(2u*px)); tput(p2,1,p,int_to_bf16(2u*py)); tput(p2,2,p,0x4000u);  // 2.0
                    uint32_t gi=(uint32_t)(ly*16+lx)*3u;                                                     // into local gt patch
                    tput(gt,p,0,f2bf_bits(gl[gi+0])); tput(gt,p,1,f2bf_bits(gl[gi+1])); tput(gt,p,2,f2bf_bits(gl[gi+2]));
                }
            }
            fence(); uint32_t ring = rdn[0] + 1u;
            rdb[0] = ring; fence();                        // <-- LOCAL doorbell store (no NoC loopback)
            dbg[7]=rdb[0];
            uint32_t to=0u, d;
            do { fence(); d = rdn[0]; } while(d != ring && ++to < 2000000u);
            dbg[3]=g+1; dbg[4]=d; dbg[5]=ring;
            for(int i=0;i<4;i++) bm_noc0_write(hub, ginbase+g*0x2000u+(uint32_t)i*0x800u, GR_SRC[i], 2048u); // grads->x280 (NoC)
        }
        sc[1]=f; bm_noc0_write(hub, ack, SC+4u, 4u);      // ack the whole tile (REMOTE, NoC)
        dbg[6]=f; last=f;
    }
}
