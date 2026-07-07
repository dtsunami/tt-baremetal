/* bin_tiles.c — x280 GAP-2 multi-tile binning. Reads N projected Gaussians (the opt_proj_step.c publish
 * format [gx,gy,a,b,c,depth]) and buckets each into every 16x16 tile its 3-sigma screen bbox overlaps,
 * keeping each tile's touching-Gaussian ids DEPTH-SORTED (near->far) via insertion (a bucket sort that
 * also orders within the bucket, capped at CAP = depth-cull of the farthest). One-shot: computes on load,
 * signals 'BIN!', spins.
 *   IN  0x30005000 : [N:int, W:int, H:int, TILE:int]
 *       0x30005010 : per Gaussian [gx,gy,a,b,c,depth] (6 f32)   (== publish format)
 *   OUT 0x30006000 : count[ntiles] (int)
 *       0x30006400 : ids[ntiles*CAP] (int), tile t at t*CAP, depth-sorted (near first)
 * The L2CPU data window is 0x30005000..0x30008000 (code at 0x30008000): this de-risk kernel bounds
 * ntiles*CAP so ids + the depth scratch stay below code. Production tile lists live in the big GDDR.
 * Validated vs gap2_bin_golden.py. */
#include <tele.h>
#include <stdint.h>

#ifndef CAP
#define CAP 32
#endif
#define MAXTILES 16           /* de-risk: up to 4x4 tiles = 64x64 px; keeps ids+dep below code */

static inline float bsqrt(float x){ float r; __asm__("fsqrt.s %0,%1":"=f"(r):"f"(x)); return r; }
static inline int ffloor(float x){ int i=(int)x; return (x<0.0f && (float)i!=x)? i-1 : i; }

int main(void){
    __asm__ volatile("csrs mstatus, %0" :: "r"(0x6000u));
    volatile int   *H  = (volatile int   *)0x30005000u;
    volatile float *in = (volatile float *)0x30005010u;   /* [gx,gy,a,b,c,depth] per G */
    volatile int   *cnt= (volatile int   *)0x30006000u;
    volatile int   *ids= (volatile int   *)0x30006400u;   /* MAXTILES*CAP ints = 16*32*4 = 2KB */
    volatile float *dep= (volatile float *)0x30006C00u;   /* scratch depths, 2KB -> ends 0x30007400 */
    int N=H[0], W=H[1], Hh=H[2], T=H[3];
    int ntx=W/T, nty=Hh/T, ntiles=ntx*nty;
    if(ntiles>MAXTILES) ntiles=MAXTILES;
    for(int t=0;t<ntiles;t++) cnt[t]=0;

    for(int i=0;i<N;i++){
        float gx=in[i*6+0], gy=in[i*6+1], a=in[i*6+2], b=in[i*6+3], c=in[i*6+4], depth=in[i*6+5];
        float det=a*c-b*b; if(det<=0.0f) continue;
        float A=c/det, C=a/det;                          /* Sigma2 diagonal (screen var x,y) */
        float ex=3.0f*bsqrt(A>0.0f?A:0.0f), ey=3.0f*bsqrt(C>0.0f?C:0.0f);
        int tx0=ffloor((gx-ex)/T), tx1=ffloor((gx+ex)/T);
        int ty0=ffloor((gy-ey)/T), ty1=ffloor((gy+ey)/T);
        if(tx0<0)tx0=0; if(tx1>ntx-1)tx1=ntx-1;
        if(ty0<0)ty0=0; if(ty1>nty-1)ty1=nty-1;
        if(tx1<tx0||ty1<ty0) continue;
        for(int ty=ty0;ty<=ty1;ty++) for(int tx=tx0;tx<=tx1;tx++){
            int t=ty*ntx+tx; if(t>=ntiles) continue;
            int n=cnt[t];
            volatile int   *tid=ids+t*CAP;
            volatile float *td =dep+t*CAP;
            if(n<CAP){                                    /* room: insert, keep ascending by depth */
                int pos=n;
                while(pos>0 && td[pos-1]>depth){ tid[pos]=tid[pos-1]; td[pos]=td[pos-1]; pos--; }
                tid[pos]=i; td[pos]=depth; cnt[t]=n+1;
            } else if(depth < td[CAP-1]){                 /* full: replace the farthest if nearer */
                int pos=CAP-1;
                while(pos>0 && td[pos-1]>depth){ tid[pos]=tid[pos-1]; td[pos]=td[pos-1]; pos--; }
                tid[pos]=i; td[pos]=depth;
            }
        }
    }
    TELE[0]=0x42494E21u;   /* 'BIN!' */
    for(;;){}
}
