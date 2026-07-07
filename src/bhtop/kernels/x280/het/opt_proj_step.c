/* opt_proj_step.c — x280 GAP-1 BACKWARD TAIL + OPTIMIZER for REAL 3D Gaussians (fp32), resident.
 * Supersedes opt_step.c (2D). Params are now 14 per Gaussian: [mean3, scale_log3, quat4, op, color3],
 * resident in GDDR across steps. One doorbell = one Adam step. Each step:
 *   whiten-backward (dLdpsi -> g_gx,g_gy,g_a,g_b,g_c)  [same as the 2D kernel]
 *   projection-backward (proj.h proj_bwd: g_a,g_b,g_c,g_gx,g_gy + params + camera -> dmean,dsl,dquat)
 *   Adam over all 14 params.
 * Also PUBLISHES the forward projection (proj.h proj_fwd) of the current params to 0x30007000 each step
 * (and once at boot) so the host reads (gx,gy,a,b,c,depth) to build the render tiles -- projection is
 * fully on-device.
 *
 * GDDR layout (uncached window):
 *   0x30005000 hdr : [K:int, step:int, bc1,bc2,b1,b2,eps:f32, lr[14]:f32]     (21 words)
 *   0x30005060 camera : Rv[9], tv[3], fx,fy,cx,cy  (16 f32)   -- may change per step (multi-view)
 *   0x300050A0 order[K] : int (sorted slot -> original id)
 *   0x30005100 gradin[K*9] : f32 SORTED, per slot [d_sa,d_m12,d_tx,d_m22,d_ty, dLdop, dc0,dc1,dc2]
 *   0x30005800 param[K*14] : f32 ORIGINAL order [mean3,sl3,quat4,op,color3]   (RESIDENT)
 *   0x30006000 adam_m[K*14] : f32 (RESIDENT)     0x30006800 adam_v[K*14] : f32 (RESIDENT)
 *   0x30007000 pub[K*6] : f32 ORIGINAL order [gx,gy,a,b,c,depth]   (host reads to stage the render)
 *   (m/v/pub respaced so K up to MAXK=24 fits the 0x30005000..0x30008000 window; bigger N -> big GDDR, Gap 5)
 *   0x30004000 doorbell (host->hart)   0x30004010 done (hart->host)
 *   0x30004020 cmd : 2 = project-only publish (uses X_CAM+params, NO Adam) ; else backward+Adam+publish.
 *              Multi-view: ring cmd=2 per camera to publish that view's projection, accumulate grads
 *              host-side, then ring cmd=1 once to apply Adam on the accumulated gradin.
 */
#include <tele.h>
#include <stdint.h>
#include "proj.h"

static void load_cam(volatile float *cam, float Rv[9], float tv[3], float *fx, float *fy, float *cx, float *cy){
    for(int i=0;i<9;i++) Rv[i]=cam[i];
    for(int i=0;i<3;i++) tv[i]=cam[9+i];
    *fx=cam[12]; *fy=cam[13]; *cx=cam[14]; *cy=cam[15];
}

static void publish(int K, volatile float *param, volatile float *pub,
                    float Rv[9], float tv[3], float fx, float fy, float cx, float cy){
    for(int g=0; g<K; g++){
        volatile float *p = param + g*14;
        float mean[3]={p[0],p[1],p[2]}, sl[3]={p[3],p[4],p[5]}, q[4]={p[6],p[7],p[8],p[9]};
        float gx,gy,depth,a,b,c;
        proj_fwd(mean,sl,q,Rv,tv,fx,fy,cx,cy,&gx,&gy,&depth,&a,&b,&c);
        pub[g*6+0]=gx; pub[g*6+1]=gy; pub[g*6+2]=a; pub[g*6+3]=b; pub[g*6+4]=c; pub[g*6+5]=depth;
    }
}

int main(void){
    __asm__ volatile("csrs mstatus, %0" :: "r"(0x6000u));            /* scalar FPU on */
    volatile int   *hdr  = (volatile int   *)0x30005000u;
    volatile float *hdrf = (volatile float *)0x30005000u;
    volatile float *cam  = (volatile float *)0x30005060u;
    volatile int   *order= (volatile int   *)0x300050A0u;
    volatile float *gin  = (volatile float *)0x30005100u;
    volatile float *param= (volatile float *)0x30005800u;
    volatile float *m    = (volatile float *)0x30006000u;
    volatile float *v    = (volatile float *)0x30006800u;
    volatile float *pub  = (volatile float *)0x30007000u;
    volatile uint32_t *db  = (volatile uint32_t *)0x30004000u;
    volatile uint32_t *done= (volatile uint32_t *)0x30004010u;
    volatile uint32_t *cmd = (volatile uint32_t *)0x30004020u;
    done[0]=0; TELE[0]=0x424F4F54u;    /* 'BOOT' — loaded, boot publish in progress */

    float Rv[9],tv[3],fx,fy,cx,cy;
    int K0 = hdr[0]; if(K0>24)K0=24; if(K0<0)K0=0;
    load_cam(cam,Rv,tv,&fx,&fy,&cx,&cy);
    publish(K0, param, pub, Rv,tv,fx,fy,cx,cy);        /* initial projection publish */
    TELE[0]=0x4F505421u;               /* 'OPT!' — publish done, resident (host may read pub now) */

    uint32_t last=0;
    for(;;){
        uint32_t ring=db[0];
        if(ring==last) continue;
        int K=hdr[0]; if(K>24)K=24;
        load_cam(cam,Rv,tv,&fx,&fy,&cx,&cy);
        if(cmd[0]==2u){                                 /* project-only publish (no Adam) */
            publish(K, param, pub, Rv,tv,fx,fy,cx,cy);
            last=ring; done[0]=ring; TELE[1]=ring; continue;
        }
        float bc1=hdrf[2], bc2=hdrf[3], b1=hdrf[4], b2=hdrf[5], eps=hdrf[6];
        const volatile float *lr=&hdrf[7];             /* lr[14] */
        for(int i=0;i<K;i++){
            int o=order[i]; if(o<0||o>=K) continue;
            const volatile float *gs = gin + i*9;      /* sorted grads */
            float d_sa=gs[0], d_m12=gs[1], d_tx=gs[2], d_m22=gs[3], d_ty=gs[4];
            volatile float *p = param + o*14;
            float mean[3]={p[0],p[1],p[2]}, sl[3]={p[3],p[4],p[5]}, q[4]={p[6],p[7],p[8],p[9]};
            /* forward proj to get (gx,gy,a,b,c) for the whiten-backward */
            float gx,gy,depth,a,b,c;
            proj_fwd(mean,sl,q,Rv,tv,fx,fy,cx,cy,&gx,&gy,&depth,&a,&b,&c);
            /* whiten-backward: psi coeffs -> (g_gx,g_gy,g_a,g_b,g_c) */
            float sa=proj_sqrt(a>1e-8f?a:1e-8f);
            float m12=b/sa;
            float t=c-b*b/a; if(t<1e-8f)t=1e-8f;
            float m22=proj_sqrt(t);
            float Dsa=d_sa+d_tx*(-gx), Dm12=d_m12+d_tx*(-gy), Dm22=d_m22+d_ty*(-gy);
            float g_gx=d_tx*(-sa);
            float g_gy=d_tx*(-m12)+d_ty*(-m22);
            float g_a=Dsa*(0.5f/sa)+Dm12*(-0.5f*b/(a*sa))+Dm22*((b*b/(a*a))/(2.0f*m22));
            float g_b=Dm12*(1.0f/sa)+Dm22*(-b/(a*m22));
            float g_c=Dm22*(1.0f/(2.0f*m22));
            /* projection-backward -> dmean,dsl,dquat */
            float dmean[3],dsl[3],dquat[4];
            proj_bwd(mean,sl,q,Rv,tv,fx,fy, g_a,g_b,g_c,g_gx,g_gy, dmean,dsl,dquat);
            /* assemble the 14-param gradient */
            float g[14];
            g[0]=dmean[0]; g[1]=dmean[1]; g[2]=dmean[2];
            g[3]=dsl[0];   g[4]=dsl[1];   g[5]=dsl[2];
            g[6]=dquat[0]; g[7]=dquat[1]; g[8]=dquat[2]; g[9]=dquat[3];
            g[10]=gs[5];                                 /* dLdop  */
            g[11]=gs[6]; g[12]=gs[7]; g[13]=gs[8];        /* dcolor */
            /* Adam over all 14 params of Gaussian o */
            for(int j=0;j<14;j++){
                volatile float *mm=m+o*14+j, *vv=v+o*14+j;
                float gj=g[j];
                /* sanitize: non-finite or blown-up grad -> skip this param's update (standard 3DGS
                 * guard; the Tensix backward can emit inf on degenerate/off-tile Gaussians). */
                if(gj!=gj || gj>1e30f || gj<-1e30f) gj=0.0f;
                if(gj> 1e4f) gj= 1e4f; if(gj<-1e4f) gj=-1e4f;   /* grad clip */
                *mm=b1*(*mm)+(1.0f-b1)*gj;
                *vv=b2*(*vv)+(1.0f-b2)*gj*gj;
                float mh=(*mm)*bc1, vh=(*vv)*bc2;
                float np=p[j]-lr[j]*mh/(proj_sqrt(vh)+eps);
                if(j==10){ if(np<0.05f)np=0.05f; if(np>0.99f)np=0.99f; }      /* opacity */
                else if(j>=11){ if(np<0.0f)np=0.0f; if(np>1.0f)np=1.0f; }     /* color */
                p[j]=np;
            }
        }
        publish(K, param, pub, Rv,tv,fx,fy,cx,cy);       /* re-project updated params */
        last=ring; done[0]=ring; TELE[1]=ring;
    }
    return 0;
}
