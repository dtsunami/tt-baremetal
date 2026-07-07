/* opt_grad_step.c — x280 CONSUMES the worker's raw grad tiles on-device: detilize the bf16 32x32 tiles the
 * render kernel produced (dLdpsi, dLdop, w, dLdC), extract per-Gaussian gradients (5 psi coeffs + dLdop +
 * dLdcolor = w^T @ dLdC), and the scalar SSE loss (Σ dLdC^2) — so the HOST never reads grads or the image.
 * This is the last piece of the fully-on-device data path (operands stream in via cb_reader, grads stream
 * back via cb_writer to the inbox below, x280 consumes them here). One invocation = one rendered GROUP.
 *
 *   in  0x30005000 hdr : [K:int]                (+ order/params reused by the Adam kernel)
 *   grad inbox (cb_writer target, tilized bf16 32x32, 2048 B each):
 *     0x30040000 dLdpsi · 0x30040800 dLdop · 0x30041000 w · 0x30041800 dLdC
 *   out 0x30042000 : per-Gaussian extracted grads [K][9] f32 = [d_sa,d_m12,d_tx,d_m22,d_ty,dLdop,dc0,dc1,dc2]
 *       0x30042800 : scalar SSE loss (f32)
 *   doorbell 0x30004000 -> done 0x30004010 ; TELE[1]=K TELE[2]=cyc_lo TELE[3]=cyc_hi */
#include <tele.h>
#include <stdint.h>
#define DP  ((volatile uint32_t*)0x30040000u)
#define DO  ((volatile uint32_t*)0x30040800u)
#define WW  ((volatile uint32_t*)0x30041000u)
#define DC  ((volatile uint32_t*)0x30041800u)
#define GOUT ((volatile float*)0x30042000u)
#define LOUT ((volatile float*)0x30042800u)
static inline uint64_t rdcycle(void){ uint64_t c; __asm__ volatile("rdcycle %0":"=r"(c)); return c; }
/* bf16 (top 16 bits) -> float */
static inline float bf16f(uint32_t h){ union{uint32_t u; float f;} v; v.u=(h&0xFFFFu)<<16; return v.f; }
/* read element (row,col) of a tilized bf16 32x32 tile (inverse of cb_operands place()) */
static inline float tget(volatile uint32_t* t, int row, int col){
    int face=((row>=16)?2:0)+((col>=16)?1:0);
    int e=face*256+(row%16)*16+(col%16);
    uint32_t w=t[e>>1];
    return bf16f((e&1)?(w>>16):(w&0xFFFFu));
}

int main(void){
    __asm__ volatile("csrs mstatus, %0" :: "r"(0x6000u));
    volatile int   *hdr =(volatile int  *)0x30005000u;
    volatile uint32_t *db  =(volatile uint32_t*)0x30004000u;
    volatile uint32_t *done=(volatile uint32_t*)0x30004010u;
    done[0]=0; TELE[0]=0x47524144u;    /* 'GRAD' */
    uint32_t last=0;
    for(;;){
        uint32_t ring=db[0];
        if(ring==last) continue;
        int K=hdr[0]; if(K>16)K=16;
        uint64_t c0=rdcycle();
        for(int i=0;i<K;i++){
            float d_sa =tget(DP,0,2*i),  d_m12=tget(DP,1,2*i),  d_tx=tget(DP,2,2*i);
            float d_m22=tget(DP,1,2*i+1),d_ty =tget(DP,2,2*i+1);
            float dLdop=tget(DO,0,i);
            float dc0=0.0f,dc1=0.0f,dc2=0.0f;
            for(int p=0;p<32;p++){                          /* dLdcolor = w^T @ dLdC (P-pixel reduction) */
                float w=tget(WW,p,i);
                dc0+=w*tget(DC,p,0); dc1+=w*tget(DC,p,1); dc2+=w*tget(DC,p,2);
            }
            volatile float *o=GOUT+(uint64_t)i*9;
            o[0]=d_sa; o[1]=d_m12; o[2]=d_tx; o[3]=d_m22; o[4]=d_ty; o[5]=dLdop; o[6]=dc0; o[7]=dc1; o[8]=dc2;
        }
        float sse=0.0f;                                     /* scalar loss = Σ dLdC^2 over the group's pixels */
        for(int p=0;p<32;p++) for(int ch=0;ch<3;ch++){ float e=tget(DC,p,ch); sse+=e*e; }
        LOUT[0]=sse;
        uint64_t c1=rdcycle();
        TELE[1]=K; TELE[2]=(uint32_t)(c1-c0); TELE[3]=(uint32_t)((c1-c0)>>32);
        last=ring; done[0]=ring;
    }
    return 0;
}
