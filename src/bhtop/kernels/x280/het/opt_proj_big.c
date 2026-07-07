/* opt_proj_big.c — GAP-5b scale de-risk: hold N (up to ~1.3M) 3D Gaussians RESIDENT in the big off-chip
 * GDDR window and run the full projection + backward + Adam over ALL of them on the x280, at scale.
 * Params + Adam m/v live at 0x30010000+ (above user code) in the 256 MB uncached-GDDR window proven
 * addressable in dram_sentinel.c. To avoid multi-MB host uploads, init params AND the per-step gradients
 * are generated on-chip from a deterministic hash (mirrored bit-exact in the Python golden), so this is a
 * pure residency+throughput+correctness test: the host spot-checks sampled Gaussians against the golden.
 *
 *   hdr 0x30005000 : [N:int, step:int, bc1,bc2,b1,b2,eps:f32, lr[14]:f32]
 *   cam 0x30005060 : Rv[9],tv[3],fx,fy,cx,cy (16 f32)
 *   cmd 0x30004020 : 3 = init (deterministic params, zero Adam) ; 1 = one Adam step over all N (synthetic grads)
 *   doorbell 0x30004000 -> done 0x30004010 ; TELE[1]=N TELE[2]=cycles_lo TELE[3]=cycles_hi
 *   params: 0x30010000 [N*14] · adam_m: +N*14*4 · adam_v: +2*N*14*4   (all in the big GDDR window) */
#include <tele.h>
#include <stdint.h>
#include "proj.h"

#define PBASE 0x30010000ULL

static inline uint32_t h32(uint32_t x){ x*=0x9E3779B1u; x^=x>>16; x*=0x85EBCA6Bu; x^=x>>13; return x; }
static inline float u01(uint32_t i, uint32_t salt){
    uint32_t h=h32(i ^ (salt*0x2545F491u));
    return (float)(h>>8) * (1.0f/16777216.0f);
}
static void init_gauss(uint32_t i, float p[14]){
    p[0]=(u01(i,1)-0.5f)*3.0f; p[1]=(u01(i,2)-0.5f)*3.0f; p[2]=(u01(i,3)-0.5f)*3.0f;   /* mean */
    p[3]=-1.8f+u01(i,4)*0.4f;  p[4]=-1.8f+u01(i,5)*0.4f;  p[5]=-1.8f+u01(i,6)*0.4f;     /* scale_log */
    p[6]=1.0f+(u01(i,7)-0.5f)*0.4f; p[7]=u01(i,8)-0.5f; p[8]=u01(i,9)-0.5f; p[9]=u01(i,10)-0.5f; /* quat, w~1 */
    p[10]=0.4f+u01(i,11)*0.5f;                                                          /* opacity */
    p[11]=u01(i,12); p[12]=u01(i,13); p[13]=u01(i,14);                                  /* color */
}
static void syn_grad(uint32_t i, float dpsi[5], float *dLdop, float dcol[3]){
    for(int k=0;k<5;k++) dpsi[k]=(u01(i,20+k)-0.5f)*0.2f;
    *dLdop=(u01(i,25)-0.5f)*0.2f;
    for(int k=0;k<3;k++) dcol[k]=(u01(i,26+k)-0.5f)*0.2f;
}
static inline uint64_t rdcycle(void){ uint64_t c; __asm__ volatile("rdcycle %0":"=r"(c)); return c; }

int main(void){
    __asm__ volatile("csrs mstatus, %0" :: "r"(0x6000u));
    volatile int   *hdr =(volatile int  *)0x30005000u;
    volatile float *hdrf=(volatile float*)0x30005000u;
    volatile float *cam =(volatile float*)0x30005060u;
    volatile uint32_t *db  =(volatile uint32_t*)0x30004000u;
    volatile uint32_t *done=(volatile uint32_t*)0x30004010u;
    volatile uint32_t *cmd =(volatile uint32_t*)0x30004020u;
    done[0]=0; TELE[0]=0x42494721u;    /* 'BIG!' */

    uint32_t last=0;
    for(;;){
        uint32_t ring=db[0];
        if(ring==last) continue;
        int N=hdr[0]; if(N<0)N=0;
        volatile float *param=(volatile float*)PBASE;
        volatile float *m=param+(uint64_t)N*14;
        volatile float *v=m+(uint64_t)N*14;
        float Rv[9],tv[3],fx,fy,cx,cy;
        for(int i=0;i<9;i++)Rv[i]=cam[i]; for(int i=0;i<3;i++)tv[i]=cam[9+i];
        fx=cam[12];fy=cam[13];cx=cam[14];cy=cam[15];
        uint64_t c0=rdcycle();

        if(cmd[0]==3u){                                /* init */
            for(uint32_t i=0;i<(uint32_t)N;i++){
                float p[14]; init_gauss(i,p);
                volatile float *pp=param+(uint64_t)i*14, *mm=m+(uint64_t)i*14, *vv=v+(uint64_t)i*14;
                for(int j=0;j<14;j++){ pp[j]=p[j]; mm[j]=0.0f; vv[j]=0.0f; }
            }
        } else {                                       /* cmd==1: one Adam step over all N */
            float bc1=hdrf[2],bc2=hdrf[3],b1=hdrf[4],b2=hdrf[5],eps=hdrf[6];
            const volatile float *lr=&hdrf[7];
            for(uint32_t i=0;i<(uint32_t)N;i++){
                volatile float *p=param+(uint64_t)i*14;
                float mean[3]={p[0],p[1],p[2]}, sl[3]={p[3],p[4],p[5]}, q[4]={p[6],p[7],p[8],p[9]};
                float gx,gy,depth,a,b,c;
                proj_fwd(mean,sl,q,Rv,tv,fx,fy,cx,cy,&gx,&gy,&depth,&a,&b,&c);
                float dpsi[5],dLdop,dcol[3]; syn_grad(i,dpsi,&dLdop,dcol);
                float d_sa=dpsi[0],d_m12=dpsi[1],d_tx=dpsi[2],d_m22=dpsi[3],d_ty=dpsi[4];
                float sa=proj_sqrt(a>1e-8f?a:1e-8f), m12=b/sa;
                float t=c-b*b/a; if(t<1e-8f)t=1e-8f; float m22=proj_sqrt(t);
                float Dsa=d_sa+d_tx*(-gx),Dm12=d_m12+d_tx*(-gy),Dm22=d_m22+d_ty*(-gy);
                float g_gx=d_tx*(-sa), g_gy=d_tx*(-m12)+d_ty*(-m22);
                float g_a=Dsa*(0.5f/sa)+Dm12*(-0.5f*b/(a*sa))+Dm22*((b*b/(a*a))/(2.0f*m22));
                float g_b=Dm12*(1.0f/sa)+Dm22*(-b/(a*m22));
                float g_c=Dm22*(1.0f/(2.0f*m22));
                float dmean[3],dsl[3],dquat[4];
                proj_bwd(mean,sl,q,Rv,tv,fx,fy,g_a,g_b,g_c,g_gx,g_gy,dmean,dsl,dquat);
                float g[14]={dmean[0],dmean[1],dmean[2],dsl[0],dsl[1],dsl[2],
                             dquat[0],dquat[1],dquat[2],dquat[3],dLdop,dcol[0],dcol[1],dcol[2]};
                volatile float *mm=m+(uint64_t)i*14, *vv=v+(uint64_t)i*14;
                for(int j=0;j<14;j++){
                    float gj=g[j];
                    if(gj!=gj||gj>1e30f||gj<-1e30f)gj=0.0f;
                    if(gj>1e4f)gj=1e4f; if(gj<-1e4f)gj=-1e4f;
                    mm[j]=b1*mm[j]+(1.0f-b1)*gj;
                    vv[j]=b2*vv[j]+(1.0f-b2)*gj*gj;
                    float mh=mm[j]*bc1, vh=vv[j]*bc2;
                    float np=p[j]-lr[j]*mh/(proj_sqrt(vh)+eps);
                    if(j==10){ if(np<0.05f)np=0.05f; if(np>0.99f)np=0.99f; }
                    else if(j>=11){ if(np<0.0f)np=0.0f; if(np>1.0f)np=1.0f; }
                    p[j]=np;
                }
            }
        }
        uint64_t c1=rdcycle();
        TELE[1]=(uint32_t)N; TELE[2]=(uint32_t)(c1-c0); TELE[3]=(uint32_t)((c1-c0)>>32);
        last=ring; done[0]=ring;
    }
    return 0;
}
