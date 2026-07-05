/* cb_whiten.c — x280 PROJECTION (whitening) in fp32, now that the FPU is enabled. Reads Sigma^-1
 * entries (a,b,c) + mean (gx,gy) per Gaussian; computes the Cholesky-style whitening coeffs
 * sa=sqrt(a), m12=b/sa, m22=sqrt(c-b^2/a), c1=-(sa*gx+m12*gy), c2=-(m22*gy) — the ψ the eval matmul
 * needs. This is the last host-side per-Gaussian compute, moved on-chip.
 *   IN 0x30005000 [K, then a,b,c,gx,gy fp32 per Gaussian] · OUT 0x30006000 [sa,m12,m22,c1,c2 per G] */
#include <tele.h>
#include <stdint.h>
static inline float fsqrtf(float x){ float r; __asm__("fsqrt.s %0,%1":"=f"(r):"f"(x)); return r; }
int main(void){
    __asm__ volatile("csrs mstatus, %0" :: "r"(0x6000u));   /* FPU on */
    volatile uint32_t* H=(volatile uint32_t*)0x30005000u;
    int K=(int)H[0]; if(K>16)K=16;
    volatile float* in =(volatile float*)0x30005004u;
    volatile float* out=(volatile float*)0x30006000u;
    for(int g=0; g<K; g++){
        float a=in[g*5+0], b=in[g*5+1], c=in[g*5+2], gx=in[g*5+3], gy=in[g*5+4];
        float sa=fsqrtf(a>1e-8f? a:1e-8f);
        float m12=b/sa;
        float t=c-b*b/a; if(t<0.0f) t=0.0f;
        float m22=fsqrtf(t);
        out[g*5+0]=sa; out[g*5+1]=m12; out[g*5+2]=m22;
        out[g*5+3]=-(sa*gx+m12*gy); out[g*5+4]=-(m22*gy);
    }
    TELE[0]=0x57484954u;   /* 'WHIT' */
    for(;;){}
}
