/* opt_step.c — x280 BACKWARD TAIL + OPTIMIZER (fp32), the irregular half of the fully-on-device splat
 * trainer. One invocation = one Adam step. The x280 is the parameter server: params + Adam m/v state
 * live RESIDENT in its GDDR across steps; the host only writes the fresh per-step gradients and reads
 * back the updated params to stage the next Tensix forward.
 *
 * Per depth-sorted slot i: map to original id via order[i]; whiten-backward turns the Tensix dLdpsi
 * coeffs (d_sa,d_m12,d_tx,d_m22,d_ty) into dL/d(gx,gy,a,b,c) using the resident (a,b,c,gx,gy); opacity
 * and color grads are already leaf. Then Adam-update all 9 params of that Gaussian in place.
 *
 * GDDR layout (uncached open window):
 *   0x30005000 hdr : [K:int, step:int, bc1:f32, bc2:f32, b1:f32, b2:f32, eps:f32, lr[9]:f32]
 *                    bc = 1/(1-beta^t) bias corrections; lr[9] = per-param Adam LR (gx,gy,a,b,c,op,c0,c1,c2).
 *                    All hyperparameters are host-supplied per step so schedules/decay need no rebuild.
 *   0x30005040 order[K] : int (sorted slot -> original id)
 *   0x30005100 gradin[K*9] : f32, SORTED order, per slot [d_sa,d_m12,d_tx,d_m22,d_ty,dLdop,dc0,dc1,dc2]
 *   0x30005800 param[K*9]  : f32, ORIGINAL order, per Gaussian [gx,gy,a,b,c,op,c0,c1,c2]  (RESIDENT)
 *   0x30006000 adam_m[K*9] : f32 (RESIDENT)   0x30006400 adam_v[K*9] : f32 (RESIDENT)
 *   0x30004000 doorbell : host writes step-id here to trigger a step (load ONCE, drive N steps)
 *   0x30004010 done     : kernel writes the step-id it just finished (host polls == doorbell)
 */
#include <tele.h>
#include <stdint.h>
static inline float fsqrtf(float x){ float r; __asm__("fsqrt.s %0,%1":"=f"(r):"f"(x)); return r; }

int main(void){
    __asm__ volatile("csrs mstatus, %0" :: "r"(0x6000u));            /* scalar FPU on */
    volatile int   *hdr  = (volatile int   *)0x30005000u;
    volatile float *hdrf = (volatile float *)0x30005000u;
    volatile int   *order  = (volatile int   *)0x30005040u;
    volatile float *gin    = (volatile float *)0x30005100u;
    volatile float *param  = (volatile float *)0x30005800u;
    volatile float *m      = (volatile float *)0x30006000u;
    volatile float *v      = (volatile float *)0x30006400u;
    volatile uint32_t *db  = (volatile uint32_t *)0x30004000u;       /* doorbell (host -> hart) */
    volatile uint32_t *done= (volatile uint32_t *)0x30004010u;       /* done   (hart -> host)  */
    done[0] = 0; TELE[0] = 0x4F505421u;    /* 'OPT!' resident, waiting */

    uint32_t last = 0;
    for(;;){
      uint32_t ring = db[0];
      if(ring == last) continue;                                     /* wait for the next step */
      int   K   = hdr[0]; if(K>16) K=16;
      float bc1 = hdrf[2], bc2 = hdrf[3];
      float b1  = hdrf[4], b2  = hdrf[5], eps = hdrf[6];             /* host-supplied hyperparams */
      const volatile float *lr = &hdrf[7];                          /* lr[9]: gx,gy,a,b,c,op,c0,c1,c2 */
      for(int i=0;i<K;i++){
        int o = order[i]; if(o<0||o>=K) continue;
        volatile float *gs = gin + i*9;                              /* sorted grads for this slot */
        float d_sa=gs[0], d_m12=gs[1], d_tx=gs[2], d_m22=gs[3], d_ty=gs[4];
        volatile float *p = param + o*9;                             /* resident params (orig order) */
        float gx=p[0], gy=p[1], a=p[2], b=p[3], c=p[4];
        /* whiten-backward: psi coeffs -> (gx,gy,a,b,c) */
        float sa = fsqrtf(a>1e-8f? a:1e-8f);
        float m12 = b/sa;
        float t  = c - b*b/a; if(t<1e-8f) t=1e-8f;
        float m22 = fsqrtf(t);
        float Dsa  = d_sa  + d_tx*(-gx);
        float Dm12 = d_m12 + d_tx*(-gy);
        float Dm22 = d_m22 + d_ty*(-gy);
        float g_gx = d_tx*(-sa);
        float g_gy = d_tx*(-m12) + d_ty*(-m22);
        float g_a = Dsa*(0.5f/sa) + Dm12*(-0.5f*b/(a*sa)) + Dm22*((b*b/(a*a))/(2.0f*m22));
        float g_b = Dm12*(1.0f/sa) + Dm22*(-b/(a*m22));
        float g_c = Dm22*(1.0f/(2.0f*m22));
        float g[9] = { g_gx, g_gy, g_a, g_b, g_c, gs[5], gs[6], gs[7], gs[8] };
        /* Adam update all 9 params of Gaussian o */
        for(int j=0;j<9;j++){
            volatile float *mm = m + o*9 + j, *vv = v + o*9 + j;
            float gj = g[j];
            *mm = b1*(*mm) + (1.0f-b1)*gj;
            *vv = b2*(*vv) + (1.0f-b2)*gj*gj;
            float mh = (*mm)*bc1, vh = (*vv)*bc2;
            float np = p[j] - lr[j]*mh/(fsqrtf(vh)+eps);
            if(j==2||j==4){ if(np<1e-3f) np=1e-3f; }                 /* a,c > 0 */
            else if(j==5){ if(np<0.05f)np=0.05f; if(np>0.99f)np=0.99f; }   /* opacity */
            else if(j>=6){ if(np<0.0f)np=0.0f; if(np>1.0f)np=1.0f; }       /* color */
            p[j]=np;
        }
      }
      last = ring;
      done[0] = ring;                                                /* publish: step `ring` complete */
      TELE[1] = ring;
    }
    return 0;
}
