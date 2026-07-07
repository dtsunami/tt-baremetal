/* cb_whiten.c — x280 PROJECTION (Gap 1): full 3D->2D camera projection + Cholesky whitening in fp32.
 * Was: orthographic toy reading 2D (a,b,c,gx,gy). Now: reads REAL 3D params [mean3,scale_log3,quat4]
 * + a camera, runs proj_fwd (proj.h) -> (gx,gy,a,b,c) + depth, then the same Cholesky whitening
 * (Sigma^-1 = M^T M): sa=sqrt(a), m12=b/sa, m22=sqrt(c-b^2/a), c1=-(sa*gx+m12*gy), c2=-(m22*gy).
 *   IN  0x30005000 : [K:int]
 *       0x30005004 : camera = Rv[9], tv[3], fx,fy,cx,cy   (16 f32)
 *       0x30005044 : per Gaussian mean[3], scale_log[3], quat[4]   (10 f32 each)
 *   OUT 0x30006000 : per Gaussian sa,m12,m22,c1,c2   (5 f32)
 *       0x30006400 : per Gaussian depth (camera z)   (1 f32) */
#include <tele.h>
#include <stdint.h>
#include "proj.h"

int main(void){
    __asm__ volatile("csrs mstatus, %0" :: "r"(0x6000u));   /* FPU on */
    volatile int   *H  = (volatile int   *)0x30005000u;
    volatile float *cam= (volatile float *)0x30005004u;      /* Rv9,tv3,fx,fy,cx,cy */
    volatile float *pin= (volatile float *)0x30005044u;      /* K * [mean3,sl3,quat4] */
    volatile float *out= (volatile float *)0x30006000u;
    volatile float *dep= (volatile float *)0x30006400u;
    int K=H[0]; if(K>16)K=16;
    float Rv[9],tv[3]; for(int i=0;i<9;i++)Rv[i]=cam[i]; for(int i=0;i<3;i++)tv[i]=cam[9+i];
    float fx=cam[12],fy=cam[13],cx=cam[14],cy=cam[15];
    for(int g=0; g<K; g++){
        float mean[3],sl[3],q[4];
        for(int i=0;i<3;i++) mean[i]=pin[g*10+i];
        for(int i=0;i<3;i++) sl[i]  =pin[g*10+3+i];
        for(int i=0;i<4;i++) q[i]   =pin[g*10+6+i];
        float gx,gy,depth,a,b,c;
        proj_fwd(mean,sl,q,Rv,tv,fx,fy,cx,cy,&gx,&gy,&depth,&a,&b,&c);
        float sa=proj_sqrt(a>1e-8f? a:1e-8f);
        float m12=b/sa;
        float t=c-b*b/a; if(t<0.0f) t=0.0f;
        float m22=proj_sqrt(t);
        out[g*5+0]=sa; out[g*5+1]=m12; out[g*5+2]=m22;
        out[g*5+3]=-(sa*gx+m12*gy); out[g*5+4]=-(m22*gy);
        dep[g]=depth;
    }
    TELE[0]=0x57484954u;   /* 'WHIT' */
    for(;;){}
}
