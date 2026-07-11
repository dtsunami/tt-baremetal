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
#define COMPACT_L 0x00004000u  // W4 STAGE-2 worker-consume: local compact grad buffer (160 f32 words = 640B, 64B-aligned)
#define DESC_STRIDE 0x800u     // must match het_x280.c DESC_STRIDE
#define DESC_IMG    120u       // descriptor word carrying the CURRENT view's target-image base (must match het_x280.c)
#define GC_LOSS_OFF 144u       // compact[144] = tile SSE loss (fixed offset, independent of K); must match het_x280.c
static inline void fence(void){ __asm__ __volatile__("fence" ::: "memory"); }

// --- integer detilize (inverse of tput): bf16 half-word at (row,col) of a 32x32 4-face tile ---
static inline uint32_t tget(volatile uint32_t* t,int row,int col){
    int face=((row>=16)?2:0)+((col>=16)?1:0); int e=face*256+(row%16)*16+(col%16);
    uint32_t w=t[e>>1]; return (e&1)?(w>>16):(w&0xFFFFu);
}
// --- W4 STAGE-2: pure-integer fixed-point MAC (BRISC is rv32im, no FPU). bf16->Qn int32 (native 32-bit
// shifts), accumulate in int64 (add + widening 32x32->64 mul are inline — no libgcc; only variable 64-bit
// shifts would need it, which q_to_f32bits avoids via a union split). Matches het_x280.c consume_slot's FP
// MAC to ~0.003% (validated on real grads) — no soft-float. ---
static inline int32_t bf16_to_q(uint32_t bf, int Q){
    uint32_t exp=(bf>>7)&0xFFu;
    if(exp>=0xE3u || exp==0u) return 0;              // fsan (inf/nan or |x|>~1e30) or zero/denormal -> 0
    int32_t mant=(int32_t)((1u<<7)|(bf&0x7Fu));      // 1.mmmmmmm (8-bit)
    int sh=(int)exp-127-7+Q;
    int32_t q;
    if(sh>=24) q=0x40000000;                          // overflow guard -> saturate (large grad, clamped downstream)
    else if(sh>=0) q=mant<<sh;                        // fits int32 (mant<2^8, sh<24)
    else if(sh>-31) q=(mant+(1<<(-sh-1)))>>(-sh);     // round-to-nearest on the right shift
    else q=0;
    return ((bf>>15)&1u) ? -q : q;
}
static inline uint32_t q_to_f32bits(int64_t sacc, int Q){
    if(sacc==0) return 0u;
    uint32_t sign=0u; union { uint64_t u; struct { uint32_t lo, hi; } w; } v;
    v.u=(sacc<0)?(uint64_t)(-sacc):(uint64_t)sacc; if(sacc<0) sign=0x80000000u;
    uint32_t hi=v.w.hi, lo=v.w.lo; int e; uint32_t top;
    if(hi){ e=32; top=hi; } else { e=0; top=lo; }
    int b=31; while(!((top>>b)&1u)) b--; e+=b;         // MSB position (no 64-bit shift)
    uint32_t mant;
    if(e>=23){ int drop=e-23;
        if(drop>=32) mant=hi>>(drop-32); else if(drop==0) mant=lo; else mant=(hi<<(32-drop))|(lo>>drop);
    } else mant=lo<<(23-e);
    mant&=0x7FFFFFu; int exp=e-Q+127;
    if(exp<=0) return sign; if(exp>=255) return sign|0x7F800000u;
    return sign|((uint32_t)exp<<23)|mant;
}

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
    uint32_t wcons=cfg[12], gcompact=cfg[13];   // W4 STAGE-2: worker-consume flag + this slot's compact grad inbox
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
        int ox=0, oy=0, K=16;
        int64_t acc[16][9]; int64_t sse_acc=0;               // W4 STAGE-2 compact-grad accumulators (per tile)
        if(wcons){ for(int i=0;i<16;i++) for(int j=0;j<9;j++) acc[i][j]=0; }
        if(!wprod){
            for(int i=0;i<6;i++) bm_noc0_read(hub, opbase+(uint32_t)i*0x800u, OP_DST[i], 2048u);   // pre-tilized operands
        } else {
            // ---- WORKER PRODUCE: read compact descriptor, scatter the 6 operand tiles locally (BIT-EXACT) ----
            bm_noc0_read(hub, desc, DESC_L, DESC_STRIDE);          // [K,ox,oy,coeff[12*9],..,img@DESC_IMG] (depth-sorted)
            K=(int)dl[0]; if(K>16)K=16; ox=(int)dl[1]; oy=(int)dl[2];
            if(dl[DESC_IMG]) img_base=dl[DESC_IMG];                 // CURRENT view base (resident views) — not stale cfg[8]
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
            if(!wcons){
                for(int i=0;i<4;i++) bm_noc0_write(hub, ginbase+g*0x2000u+(uint32_t)i*0x800u, GR_SRC[i], 2048u); // grads->x280 (NoC)
            } else {
                // ---- WORKER CONSUME: detilize this group's grads from LOCAL L1 (no uncached x280 round-trip)
                // + integer dLdcolor MAC + SSE, accumulated across the NG groups (mirrors het_x280 consume_slot) ----
                volatile uint32_t *DP=(volatile uint32_t*)GR_SRC[0], *DO=(volatile uint32_t*)GR_SRC[1],
                                   *WW=(volatile uint32_t*)GR_SRC[2], *DC=(volatile uint32_t*)GR_SRC[3];
                int32_t dcq[3][32];
                for(int p=0;p<32;p++) for(int ch=0;ch<3;ch++){ int32_t q=bf16_to_q(tget(DC,p,ch),14);
                    dcq[ch][p]=q; sse_acc += (int64_t)q*(int64_t)q; }                            // loss = Sum dLdC^2
                for(int i=0;i<K;i++){
                    acc[i][0]+=bf16_to_q(tget(DP,0,2*i),16);   acc[i][1]+=bf16_to_q(tget(DP,1,2*i),16);
                    acc[i][2]+=bf16_to_q(tget(DP,2,2*i),16);   acc[i][3]+=bf16_to_q(tget(DP,1,2*i+1),16);
                    acc[i][4]+=bf16_to_q(tget(DP,2,2*i+1),16); acc[i][5]+=bf16_to_q(tget(DO,0,i),16);   // dLdpsi+dLdop
                    for(int ch=0;ch<3;ch++){ int64_t s=0;
                        for(int p=0;p<32;p++) s += (int64_t)bf16_to_q(tget(WW,p,i),15)*(int64_t)dcq[ch][p]; // dLdcolor MAC
                        acc[i][6+ch]+=s; }
                }
            }
        }
        if(wcons){                                            // W4 STAGE-2: pack + NoC-write ONE compact grad buffer
            volatile uint32_t* comp=(volatile uint32_t*)COMPACT_L;   // [K*9 grads (f32 bits) + loss@144] vs 4 tiles/grp
            for(int i=0;i<16;i++) for(int j=0;j<9;j++)
                comp[i*9+j]=(i<K)?q_to_f32bits(acc[i][j],(j<6)?16:29):0u;
            comp[GC_LOSS_OFF]=q_to_f32bits(sse_acc,28);
            bm_noc0_write(hub, gcompact, COMPACT_L, 160u*4u);       // 640B (64B-aligned) -> x280 pure scatter-add
        }
        sc[1]=f; bm_noc0_write(hub, ack, SC+4u, 4u);      // ack the whole tile (REMOTE, NoC)
        dbg[6]=f; last=f;
    }
}
