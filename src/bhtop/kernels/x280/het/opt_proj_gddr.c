/* opt_proj_gddr.c — x280 REAL-gradient 3D optimizer with ALL state in the big 4 GiB local GDDR window
 * (Gap 5), so N scales to millions. Same math as opt_proj_step.c (projection fwd/bwd via proj.h + whiten-bwd
 * + Adam over 14 params) but param/Adam/gradin/pub/order bases are computed from N at 0x30010000+, and it
 * emits per-phase CYCLE telemetry for the perf-tuning breakdown.
 *
 *   hdr 0x30005000 : [N:int, step:int, bc1,bc2,b1,b2,eps:f32, lr[14]:f32]
 *   cam 0x30005060 : Rv[9],tv[3],fx,fy,cx,cy (16 f32)
 *   cmd 0x30004020 : 2 = project-only publish (no Adam) ; 1 = whiten-bwd+proj-bwd+Adam over all N
 *   doorbell 0x30004000 -> done 0x30004010
 *   TELE[1]=N  TELE[2]=cycles_lo  TELE[3]=cycles_hi   (per-ring x280 cycles: projection or Adam)
 *   GDDR (0x30010000 base, computed from N):
 *     PARAM[N*14] · M[N*14] · V[N*14] · GRADIN[N*9] (SORTED by order) · PUB[N*6] (gx,gy,a,b,c,depth) · ORDER[N] int
 */
#include <tele.h>
#include <stdint.h>
#include "proj.h"
#define GBASE 0x30010000ULL
static inline uint64_t rdcycle(void){ uint64_t c; __asm__ volatile("rdcycle %0":"=r"(c)); return c; }

int main(void){
    __asm__ volatile("csrs mstatus, %0" :: "r"(0x6000u));
    volatile int   *hdr =(volatile int  *)0x30005000u;
    volatile float *hdrf=(volatile float*)0x30005000u;
    volatile float *cam =(volatile float*)0x30005060u;
    volatile uint32_t *db  =(volatile uint32_t*)0x30004000u;
    volatile uint32_t *done=(volatile uint32_t*)0x30004010u;
    volatile uint32_t *cmd =(volatile uint32_t*)0x30004020u;
    done[0]=0; TELE[0]=0x47444452u;    /* 'GDDR' */

    uint32_t last=0;
    for(;;){
        uint32_t ring=db[0];
        if(ring==last) continue;
        uint32_t N=(uint32_t)hdr[0];
        volatile float *param=(volatile float*)GBASE;
        volatile float *m     =param  + (uint64_t)N*14;
        volatile float *v     =m      + (uint64_t)N*14;
        volatile float *gin   =v      + (uint64_t)N*14;   /* [N*9] SORTED */
        volatile float *pub   =gin    + (uint64_t)N*9;    /* [N*6] */
        volatile int   *order =(volatile int*)(pub + (uint64_t)N*6);
        float Rv[9],tv[3],fx,fy,cx,cy;
        for(int i=0;i<9;i++)Rv[i]=cam[i]; for(int i=0;i<3;i++)tv[i]=cam[9+i];
        fx=cam[12];fy=cam[13];cx=cam[14];cy=cam[15];
        uint64_t c0=rdcycle();

        if(cmd[0]==2u){                                   /* project all -> publish */
            for(uint32_t g=0; g<N; g++){
                volatile float *p=param+(uint64_t)g*14;
                float mean[3]={p[0],p[1],p[2]}, sl[3]={p[3],p[4],p[5]}, q[4]={p[6],p[7],p[8],p[9]};
                float gx,gy,depth,a,b,c;
                proj_fwd(mean,sl,q,Rv,tv,fx,fy,cx,cy,&gx,&gy,&depth,&a,&b,&c);
                volatile float *pu=pub+(uint64_t)g*6;
                pu[0]=gx;pu[1]=gy;pu[2]=a;pu[3]=b;pu[4]=c;pu[5]=depth;
            }
        } else {                                          /* whiten-bwd + proj-bwd + Adam over all N */
            float bc1=hdrf[2],bc2=hdrf[3],b1=hdrf[4],b2=hdrf[5],eps=hdrf[6];
            const volatile float *lr=&hdrf[7];
            for(uint32_t i=0;i<N;i++){
                int o=order[i]; if(o<0||(uint32_t)o>=N) continue;
                const volatile float *gs=gin+(uint64_t)i*9;
                float d_sa=gs[0],d_m12=gs[1],d_tx=gs[2],d_m22=gs[3],d_ty=gs[4];
                volatile float *p=param+(uint64_t)o*14;
                float mean[3]={p[0],p[1],p[2]}, sl[3]={p[3],p[4],p[5]}, q[4]={p[6],p[7],p[8],p[9]};
                float gx,gy,depth,a,b,c;
                proj_fwd(mean,sl,q,Rv,tv,fx,fy,cx,cy,&gx,&gy,&depth,&a,&b,&c);
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
                             dquat[0],dquat[1],dquat[2],dquat[3],gs[5],gs[6],gs[7],gs[8]};
                volatile float *mm=m+(uint64_t)o*14, *vv=v+(uint64_t)o*14;
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
            /* re-project so pub is fresh for the next step's render */
            for(uint32_t g=0; g<N; g++){
                volatile float *p=param+(uint64_t)g*14;
                float mean[3]={p[0],p[1],p[2]}, sl[3]={p[3],p[4],p[5]}, q[4]={p[6],p[7],p[8],p[9]};
                float gx,gy,depth,a,b,c;
                proj_fwd(mean,sl,q,Rv,tv,fx,fy,cx,cy,&gx,&gy,&depth,&a,&b,&c);
                volatile float *pu=pub+(uint64_t)g*6;
                pu[0]=gx;pu[1]=gy;pu[2]=a;pu[3]=b;pu[4]=c;pu[5]=depth;
            }
        }
        uint64_t c1=rdcycle();
        TELE[1]=N; TELE[2]=(uint32_t)(c1-c0); TELE[3]=(uint32_t)((c1-c0)>>32);
        last=ring; done[0]=ring;
    }
    return 0;
}
