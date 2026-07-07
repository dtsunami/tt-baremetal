/* het_x280.c — the x280 HUB for the fully-on-device fused 3DGS step. One resident kernel, doorbell-driven
 * command modes, no host in the per-tile DATA path. Params live resident in GDDR; the x280 projects+whitens
 * them into a coeff buffer, PRODUCES each tile's tilized operands (workers NoC-read them), CONSUMES the
 * grads the workers NoC-write back, and runs Adam. Host writes only per-tile id lists + doorbells, reads a
 * scalar loss.  cmd: 2=project+whiten  5=produce(tile)  6=consume(tile grads)  1=adam.
 *
 * FIXED low scratch:  doorbell 0x30004000 / done 0x30004010 / cmd 0x30004020
 *   hdr 0x30005000 [N,step,bc1,bc2,b1,b2,eps, lr[14]]   cam 0x30005060 [Rv9,tv3,fx,fy,cx,cy]
 *   idlist 0x300050A0 [K, id0..idK-1]   order 0x30005200 [K sorted-slot->global-id]   loss 0x30005300 [f32]
 * OPERAND scratch (workers cb_reader from here): 0x30080000 + slot*0x800  (psi,Dop,Dnop,color,colorT,opB)
 * GRAD inbox (workers cb_writer to here):        0x30088000 + slot*0x800  (dLdpsi,dLdop,w,dLdC)
 * BIG buffers (from N): PARAM 0x30100000 · M · V · GACC[N*9] · COEFF[N*9 bf16] · DEPTH[N] · PUB[N*6] */
#include <tele.h>
#include <stdint.h>
#include "proj.h"
#define DB   ((volatile uint32_t*)0x30004000u)
#define DONE ((volatile uint32_t*)0x30004010u)
#define CMD  ((volatile uint32_t*)0x30004020u)
#define OPBASE 0x30080000u    /* per-worker slot: OPBASE + slot*0x3000 (6 operand tiles) — cmd5/6 legacy */
#define GINBOX 0x300C0000u    /* per-worker slot: GINBOX + slot*0x2000 (4 grad tiles)   — cmd5/6 legacy */
#define OPSTRIDE 0x3000u
#define GISTRIDE 0x2000u
/* ORCHESTRATOR path (cmd7/8/9) — respaced to well-separated high regions, stride 0x10000/slot (no collisions
 * up to 16 workers). NOTE: for millions of Gaussians these must move above the param buffers (compute from N). */
#define OPB_O   0x31000000u   /* per-slot 6 operand tiles (workers NoC-read) */
#define OPS_O   0x00010000u
#define PXBASE  0x32000000u   /* per-slot phi/phi2T/gt: 8grp*3 tiles (workers NoC-read) */
#define PXS_O   0x00010000u
#define GINO    0x33000000u   /* per-slot grad inbox: 8grp*4 tiles (workers NoC-write) */
#define GIS_O   0x00010000u
#define TGT_IMG 0x30200000u   /* resident target image (IMGW*IMGH*3 f32, host-loaded once) */
/* Per-slot FLAG/ACK on their OWN 64B line — concurrent 4B NoC writes to adjacent words in one granule race
 * and get clobbered (silicon-observed: 2 of 6 workers' acks lost at 4B stride). 0x40 stride = one line/slot. */
#define FLAG_B  0x30006400u
#define ACK_B   0x30006800u
#define ASTRIDE 0x40u
/* MULTI-HART (4 harts on the hub tile share its local GDDR): hart 0 = leader (command loop), harts 1..NH-1 =
 * workers. cmd9 partitions slots s%NH==hid across harts, each consuming into its OWN gacc/loss (no scatter-add
 * race), cmd1 merges. HGO/HDONE/LOSS_H each on own 64B line (concurrent-write granule rule). */
#define NHARTS_A 0x300027F0u  /* host-set hart count (1..4) */
#define WCMD_A   0x300027F4u  /* which slice the workers run this dispatch: 2=project 1=adam 9=orchestrate */
#define IMG_BASE_A 0x300027F8u /* current view's resident target-image base (0 => fall back to TGT_IMG) */
#define IMGH_A   0x30005DFCu   /* image height (px) for on-device binning */
#define IDLGB    0x35000000u   /* on-device bin: per-TILE id-list [count,id0..id11], stride 0x40 */
#define DEPB     0x35100000u   /* per-tile front-K depths (insertion sort), stride 0x30 (12 f32) */
#define OCC      0x35200000u   /* per-tile occupancy count (contiguous ints) — host reads this, not pub */
#define HGO      0x30002800u  /* leader->worker wake, +h*0x40 */
#define HDONE    0x30002A00u  /* worker->leader done, +h*0x40 */
#define LOSS_H   0x30002C00u  /* per-hart partial loss, +h*0x40 */
#define GACC_X   0x30280000u  /* per-hart extra gacc (hart h>0 at GACC_X + (h-1)*N*9); hart 0 uses main gacc */
static inline uint64_t rdcycle(void){ uint64_t c; __asm__ volatile("rdcycle %0":"=r"(c)); return c; }
static inline uint32_t f2bf(float x){ union{float f; uint32_t u;} v; v.f=x; uint32_t b=v.u; b+=0x7FFFu+((b>>16)&1u); return (b>>16)&0xFFFFu; }
static inline float bf2f(uint32_t h){ union{uint32_t u; float f;} v; v.u=(h&0xFFFFu)<<16; return v.f; }
/* sanitize: non-finite/blown-up -> 0 (render emits inf on degenerate/off-tile Gaussians) */
static inline float fsan(float x){ return (x!=x || x>1e30f || x<-1e30f) ? 0.0f : x; }
/* tilize: write bf16 into element (row,col) of a 32x32 tile (cb_operands place()) */
static inline void tput(volatile uint32_t* t,int row,int col,uint32_t bf){
    int face=((row>=16)?2:0)+((col>=16)?1:0); int e=face*256+(row%16)*16+(col%16); int wd=e>>1;
    uint32_t cur=t[wd]; t[wd]=(e&1)?((cur&0x0000FFFFu)|(bf<<16)):((cur&0xFFFF0000u)|bf);
}
static inline float tget(volatile uint32_t* t,int row,int col){
    int face=((row>=16)?2:0)+((col>=16)?1:0); int e=face*256+(row%16)*16+(col%16);
    uint32_t w=t[e>>1]; return bf2f((e&1)?(w>>16):(w&0xFFFFu));
}

/* ---- ORCHESTRATOR helpers (shared by cmd7 single-tile + cmd9 grid) ---------------------------------- */
/* tilize slot's 6 operands from coeff[], depth-sorted by the tile's id list -> OPB_O[slot], ordr[slot] */
static void produce_ops(int slot, volatile int* il, volatile uint32_t* coeff, volatile uint32_t* depth, volatile int* ordr){
    int K=il[0]; if(K>16)K=16; int ids[16]; float dz[16];
    for(int i=0;i<K;i++){ ids[i]=il[1+i]; union{uint32_t u; float f;} d; d.u=depth[ids[i]]; dz[i]=d.f; }
    int sl[16]; for(int i=0;i<K;i++) sl[i]=i;
    for(int i=1;i<K;i++){ int k=sl[i]; float kz=dz[k]; int j=i-1; while(j>=0 && dz[sl[j]]>kz){ sl[j+1]=sl[j]; j--; } sl[j+1]=k; }
    uint32_t opb=OPB_O+(uint32_t)slot*OPS_O; volatile int *ors=ordr+slot*16;
    volatile uint32_t *PSI=(volatile uint32_t*)(opb+0*0x800u),*DOP=(volatile uint32_t*)(opb+1*0x800u),
        *DNOP=(volatile uint32_t*)(opb+2*0x800u),*COL=(volatile uint32_t*)(opb+3*0x800u),
        *COLT=(volatile uint32_t*)(opb+4*0x800u),*OPB=(volatile uint32_t*)(opb+5*0x800u);
    for(int w=0;w<512;w++){ PSI[w]=0;DOP[w]=0;DNOP[w]=0;COL[w]=0;COLT[w]=0;OPB[w]=0; }
    for(int i=0;i<K;i++){ int gid=ids[sl[i]]; ors[i]=gid; volatile uint32_t *co=coeff+(uint64_t)gid*9;
        uint32_t sa=co[0],m12=co[1],m22=co[2],c1=co[3],c2=co[4],op=co[5],rr=co[6],gg=co[7],bb=co[8];
        tput(PSI,0,2*i,sa); tput(PSI,1,2*i,m12); tput(PSI,2,2*i,c1); tput(PSI,1,2*i+1,m22); tput(PSI,2,2*i+1,c2);
        tput(DOP,i,i,op); tput(DNOP,i,i,op^0x8000u); tput(COL,i,0,rr); tput(COL,i,1,gg); tput(COL,i,2,bb);
        tput(COLT,0,i,rr); tput(COLT,1,i,gg); tput(COLT,2,i,bb); for(int p=0;p<32;p++) tput(OPB,p,i,op); }
    for(int k=K;k<32;k++) for(int p=0;p<32;p++) tput(OPB,p,k,0x3F00u);
}
/* tilize slot's phi/phi2T/gt from tile origin (ox,oy) + resident image (IMGW wide) -> PXBASE[slot] */
static void produce_pix(int slot, int ox, int oy, int IMGW, volatile float* img){
    uint32_t pxb=PXBASE+(uint32_t)slot*PXS_O;
    for(int g=0;g<8;g++){
        volatile uint32_t *ph=(volatile uint32_t*)(pxb+(uint32_t)g*3u*0x800u+0u*0x800u);
        volatile uint32_t *p2=(volatile uint32_t*)(pxb+(uint32_t)g*3u*0x800u+1u*0x800u);
        volatile uint32_t *gt=(volatile uint32_t*)(pxb+(uint32_t)g*3u*0x800u+2u*0x800u);
        for(int w=0;w<512;w++){ ph[w]=0;p2[w]=0;gt[w]=0; }
        for(int p=0;p<32;p++){
            int idx=g*32+p; int lx=idx%16, ly=idx/16; float px=(float)(ox+lx), py=(float)(oy+ly);
            tput(ph,p,0,f2bf(px)); tput(ph,p,1,f2bf(py)); tput(ph,p,2,f2bf(1.0f));
            tput(p2,0,p,f2bf(2.0f*px)); tput(p2,1,p,f2bf(2.0f*py)); tput(p2,2,p,f2bf(2.0f));
            int ii=((oy+ly)*IMGW+(ox+lx))*3;
            tput(gt,p,0,f2bf(img[ii+0])); tput(gt,p,1,f2bf(img[ii+1])); tput(gt,p,2,f2bf(img[ii+2]));
        }
    }
}
/* detilize slot's 8 grad groups from GINO -> scatter-add into gacc[global], accumulate loss.
 * PERF: DC (dLdC) is identical for every Gaussian in a group — detilize it ONCE into a cached stack array
 * (was re-read K times = the dominant uncached-GDDR cost) and fold the SSE loss into the same pass. The
 * dLdcolor MAC then reads cached dcv[]. Bit-identical to the naive form (same fp ops, same order). */
static void consume_slot(int slot, int K, volatile int* ordr, volatile float* gacc, volatile float* loss){
    if(K>16)K=16; volatile int *ors=ordr+slot*16;
    for(int g=0;g<8;g++){
        uint32_t gib=GINO+(uint32_t)slot*GIS_O+(uint32_t)g*0x2000u;
        volatile uint32_t *DP=(volatile uint32_t*)(gib+0*0x800u),*DO=(volatile uint32_t*)(gib+1*0x800u),
            *WW=(volatile uint32_t*)(gib+2*0x800u),*DC=(volatile uint32_t*)(gib+3*0x800u);
        float dc0v[32],dc1v[32],dc2v[32]; float sse=0.0f;                    /* DC hoisted once (+ SSE folded in) */
        for(int p=0;p<32;p++){ float e0=fsan(tget(DC,p,0)),e1=fsan(tget(DC,p,1)),e2=fsan(tget(DC,p,2));
            dc0v[p]=e0; dc1v[p]=e1; dc2v[p]=e2; sse+=e0*e0+e1*e1+e2*e2; }
        loss[0]+=sse;
        for(int i=0;i<K;i++){ int gid=ors[i]; volatile float *ga=gacc+(uint64_t)gid*9;
            ga[0]+=fsan(tget(DP,0,2*i)); ga[1]+=fsan(tget(DP,1,2*i)); ga[2]+=fsan(tget(DP,2,2*i));
            ga[3]+=fsan(tget(DP,1,2*i+1)); ga[4]+=fsan(tget(DP,2,2*i+1)); ga[5]+=fsan(tget(DO,0,i));
            float dc0=0,dc1=0,dc2=0;
            for(int p=0;p<32;p++){ float w=fsan(tget(WW,p,i)); dc0+=w*dc0v[p]; dc1+=w*dc1v[p]; dc2+=w*dc2v[p]; }
            ga[6]+=fsan(dc0); ga[7]+=fsan(dc1); ga[8]+=fsan(dc2); }
    }
}

static inline int flr(float x){ int i=(int)x; return (x<(float)i)?i-1:i; }   /* floor (host uses math.floor) */
/* insert Gaussian gid (depth dep) into tile t's front-12-by-depth id-list (IDLGB[t]=[count,id..]) — the
 * on-device equivalent of the host bin_tiles' depth-sort + cap. Keeps the 12 nearest (smallest depth). */
static void bin_insert(int t, int gid, float dep){
    volatile int* L=(volatile int*)(IDLGB + (uint32_t)t*0x40u);
    volatile float* D=(volatile float*)(DEPB + (uint32_t)t*0x30u);
    int cnt=L[0];
    if(cnt<12){
        int p=cnt; while(p>0 && D[p-1]>dep){ D[p]=D[p-1]; L[1+p]=L[p]; p--; }
        D[p]=dep; L[1+p]=gid; L[0]=cnt+1;
    } else if(dep < D[11]){
        int p=11; while(p>0 && D[p-1]>dep){ D[p]=D[p-1]; L[1+p]=L[p]; p--; }
        D[p]=dep; L[1+p]=gid;
    }
}

/* one hart's slice of a cmd9 batch: produce+signal, then wait+consume, its slots (s%NH==hid), into its OWN
 * gacc/loss copy. Runs concurrently on all NH harts of the hub tile. `ring` distinguishes this batch. */
static void cmd9_slice(int hid, int NH, uint32_t ring){
    volatile int *hdr=(volatile int*)0x30005000u, *ordr=(volatile int*)0x30005200u;
    volatile int *tidx=(volatile int*)0x30005E00u;                      /* per-slot TILE INDEX (host, post on-device bin) */
    uint32_t N=(uint32_t)hdr[0];
    volatile float *param=(volatile float*)0x30100000u;
    volatile float *gacc0=param + (uint64_t)N*42;                       /* main gacc (hart 0) */
    volatile uint32_t *coeff=(volatile uint32_t*)(gacc0 + (uint64_t)N*9);
    volatile uint32_t *depth=(volatile uint32_t*)(coeff + (uint64_t)N*9);
    volatile float *mygacc = (hid==0) ? gacc0 : ((volatile float*)GACC_X + (uint64_t)(hid-1)*N*9);
    volatile float *myloss = (volatile float*)(LOSS_H + (uint32_t)hid*0x40u);
    uint32_t imb=((volatile uint32_t*)IMG_BASE_A)[0]; if(imb==0u) imb=TGT_IMG;   /* resident view image base */
    volatile float *img=(volatile float*)(uint64_t)imb;
    int ns=((volatile int*)0x30005DF0u)[0]; if(ns<1)ns=1; if(ns>16)ns=16;
    int IMGW=((volatile int*)0x30005DF4u)[0]; if(IMGW<=0)IMGW=16; int ntx=IMGW/16; if(ntx<1)ntx=1;
    for(int s=hid; s<ns; s+=NH){                                        /* produce + signal my slots */
        int tl=tidx[s]; volatile int* il=(volatile int*)(IDLGB+(uint32_t)tl*0x40u);   /* on-device tile id-list */
        produce_ops(s, il, coeff, depth, ordr);
        produce_pix(s, (tl%ntx)*16, (tl/ntx)*16, IMGW, img);
        ((volatile uint32_t*)(FLAG_B+(uint32_t)s*ASTRIDE))[0]=ring;
    }
    for(int s=hid; s<ns; s+=NH){                                        /* wait + consume my slots */
        volatile uint32_t* ack=(volatile uint32_t*)(ACK_B+(uint32_t)s*ASTRIDE);
        uint32_t to=0u; while(ack[0]!=ring){ if(++to>20000000u) break; }
        int tl=tidx[s]; consume_slot(s, ((volatile int*)(IDLGB+(uint32_t)tl*0x40u))[0], ordr, mygacc, myloss);
    }
}

/* one hart's stripe of PROJECT+WHITEN (Gaussians g%NH==hid) -> coeff/depth/pub; zeros its OWN gacc+loss */
static void proj_slice(int hid, int NH){
    volatile int *hdr=(volatile int*)0x30005000u; volatile float *cam=(volatile float*)0x30005060u;
    uint32_t N=(uint32_t)hdr[0];
    volatile float *param=(volatile float*)0x30100000u, *gacc0=param+(uint64_t)N*42;
    volatile uint32_t *coeff=(volatile uint32_t*)(gacc0+(uint64_t)N*9), *depth=(volatile uint32_t*)(coeff+(uint64_t)N*9);
    volatile float *pub=(volatile float*)(depth+(uint64_t)N);
    float Rv[9],tv[3],fx,fy,cx,cy; for(int i=0;i<9;i++)Rv[i]=cam[i]; for(int i=0;i<3;i++)tv[i]=cam[9+i];
    fx=cam[12];fy=cam[13];cx=cam[14];cy=cam[15];
    volatile float *mygacc=(hid==0)?gacc0:((volatile float*)GACC_X+(uint64_t)(hid-1)*N*9);
    for(uint64_t j=0;j<(uint64_t)N*9;j++) mygacc[j]=0.0f;                      /* zero my accumulator */
    ((volatile float*)(LOSS_H+(uint32_t)hid*0x40u))[0]=0.0f;
    for(uint32_t g=(uint32_t)hid; g<N; g+=(uint32_t)NH){
        volatile float *p=param+(uint64_t)g*14;
        float mean[3]={p[0],p[1],p[2]}, sl[3]={p[3],p[4],p[5]}, q[4]={p[6],p[7],p[8],p[9]};
        float gx,gy,dep,a,b,c; proj_fwd(mean,sl,q,Rv,tv,fx,fy,cx,cy,&gx,&gy,&dep,&a,&b,&c);
        float sa=proj_sqrt(a>1e-8f?a:1e-8f), m12=b/sa; float t=c-b*b/a; if(t<0.0f)t=0.0f; float m22=proj_sqrt(t);
        float c1=-(sa*gx+m12*gy), c2=-(m22*gy);
        volatile uint32_t *co=coeff+(uint64_t)g*9;
        co[0]=f2bf(sa); co[1]=f2bf(m12); co[2]=f2bf(m22); co[3]=f2bf(c1); co[4]=f2bf(c2);
        co[5]=f2bf(p[10]); co[6]=f2bf(p[11]); co[7]=f2bf(p[12]); co[8]=f2bf(p[13]);
        union{float f; uint32_t u;} dz; dz.f=dep; depth[g]=dz.u;
        volatile float *pu=pub+(uint64_t)g*6; pu[0]=gx;pu[1]=gy;pu[2]=a;pu[3]=b;pu[4]=c;pu[5]=dep;
    }
}
/* one hart's stripe of ADAM (Gaussians g%NH==hid): merge gacc across harts -> whiten-bwd + proj-bwd -> update */
static void adam_slice(int hid, int NH){
    volatile int *hdr=(volatile int*)0x30005000u; volatile float *hdrf=(volatile float*)0x30005000u;
    volatile float *cam=(volatile float*)0x30005060u; uint32_t N=(uint32_t)hdr[0];
    volatile float *param=(volatile float*)0x30100000u, *m=param+(uint64_t)N*14, *v=m+(uint64_t)N*14, *gacc0=v+(uint64_t)N*14;
    float Rv[9],tv[3],fx,fy,cx,cy; for(int i=0;i<9;i++)Rv[i]=cam[i]; for(int i=0;i<3;i++)tv[i]=cam[9+i];
    fx=cam[12];fy=cam[13];cx=cam[14];cy=cam[15];
    float bc1=hdrf[2],bc2=hdrf[3],b1=hdrf[4],b2=hdrf[5],eps=hdrf[6]; const volatile float *lr=&hdrf[7];
    for(uint32_t g=(uint32_t)hid; g<N; g+=(uint32_t)NH){
        volatile float *p=param+(uint64_t)g*14;
        float ga[9]; { volatile float* g0=gacc0+(uint64_t)g*9; for(int j=0;j<9;j++) ga[j]=g0[j];
            for(int h=1;h<NH;h++){ volatile float* gh=(volatile float*)GACC_X+(uint64_t)(h-1)*N*9+(uint64_t)g*9;
                for(int j=0;j<9;j++) ga[j]+=gh[j]; } }
        float mean[3]={p[0],p[1],p[2]}, s[3]={p[3],p[4],p[5]}, q[4]={p[6],p[7],p[8],p[9]};
        float gx,gy,dep,a,b,c; proj_fwd(mean,s,q,Rv,tv,fx,fy,cx,cy,&gx,&gy,&dep,&a,&b,&c);
        float d_sa=ga[0],d_m12=ga[1],d_tx=ga[2],d_m22=ga[3],d_ty=ga[4];
        float sa=proj_sqrt(a>1e-8f?a:1e-8f), m12=b/sa; float t=c-b*b/a; if(t<1e-8f)t=1e-8f; float m22=proj_sqrt(t);
        float Dsa=d_sa+d_tx*(-gx),Dm12=d_m12+d_tx*(-gy),Dm22=d_m22+d_ty*(-gy);
        float g_gx=d_tx*(-sa), g_gy=d_tx*(-m12)+d_ty*(-m22);
        float g_a=Dsa*(0.5f/sa)+Dm12*(-0.5f*b/(a*sa))+Dm22*((b*b/(a*a))/(2.0f*m22));
        float g_b=Dm12*(1.0f/sa)+Dm22*(-b/(a*m22)); float g_c=Dm22*(1.0f/(2.0f*m22));
        float dmean[3],dsl[3],dq[4];
        proj_bwd(mean,s,q,Rv,tv,fx,fy,g_a,g_b,g_c,g_gx,g_gy,dmean,dsl,dq);
        float gg[14]={dmean[0],dmean[1],dmean[2],dsl[0],dsl[1],dsl[2],dq[0],dq[1],dq[2],dq[3],ga[5],ga[6],ga[7],ga[8]};
        volatile float *mm=m+(uint64_t)g*14, *vv=v+(uint64_t)g*14;
        for(int j=0;j<14;j++){
            float gj=gg[j]; if(gj!=gj||gj>1e30f||gj<-1e30f)gj=0.0f; if(gj>1e4f)gj=1e4f; if(gj<-1e4f)gj=-1e4f;
            mm[j]=b1*mm[j]+(1.0f-b1)*gj; vv[j]=b2*vv[j]+(1.0f-b2)*gj*gj;
            float mh=mm[j]*bc1, vh=vv[j]*bc2; float np=p[j]-lr[j]*mh/(proj_sqrt(vh)+eps);
            if(j==10){ if(np<0.05f)np=0.05f; if(np>0.99f)np=0.99f; } else if(j>=11){ if(np<0.0f)np=0.0f; if(np>1.0f)np=1.0f; }
            p[j]=np;
        }
    }
}

int main(void){
    __asm__ volatile("csrs mstatus, %0" :: "r"(0x6000u));
    uint32_t hid=0u; __asm__ volatile("csrr %0, mhartid":"=r"(hid)); hid&=0xFu;
    if(hid!=0u){                          /* WORKER hart: poll HGO -> run its cmd9 slice -> signal HDONE */
        volatile uint32_t* hgo  =(volatile uint32_t*)(HGO  +hid*0x40u);
        volatile uint32_t* hdone=(volatile uint32_t*)(HDONE+hid*0x40u);
        TELE[0]=0x48570000u|hid;          /* 'HW'|hid — worker alive marker (own tele window) */
        uint32_t wlast=0u;
        for(;;){ uint32_t g=hgo[0]; if(g==wlast) continue;
            int NH=((volatile int*)NHARTS_A)[0]; if(NH<1)NH=1; if(NH>4)NH=4;
            int wc=((volatile int*)WCMD_A)[0];
            if(wc==2) proj_slice((int)hid, NH);            /* project stripe */
            else if(wc==1) adam_slice((int)hid, NH);       /* adam stripe */
            else cmd9_slice((int)hid, NH, g);              /* orchestrate stripe */
            wlast=g; hdone[0]=g; }
    }
    volatile int   *hdr =(volatile int  *)0x30005000u;
    volatile float *hdrf=(volatile float*)0x30005000u;
    volatile float *cam =(volatile float*)0x30005060u;
    volatile int   *idl =(volatile int  *)0x300050A0u;   /* [K, id0..] */
    volatile int   *ordr=(volatile int  *)0x30005200u;   /* per-slot sorted->global id: ordr + slot*16 (32 slots) */
    volatile float *loss=(volatile float*)0x30005B00u;   /* moved clear of the 2KB per-slot ORDER region */
    DONE[0]=0; TELE[0]=0x48455421u;    /* 'HET!' */
    uint32_t last=0;
    for(;;){
        uint32_t ring=DB[0]; if(ring==last) continue;
        uint32_t N=(uint32_t)hdr[0];
        volatile float *param=(volatile float*)0x30100000u;
        volatile float *m    =param + (uint64_t)N*14;
        volatile float *v    =m     + (uint64_t)N*14;
        volatile float *gacc =v     + (uint64_t)N*14;     /* [N*9] */
        volatile uint32_t *coeff=(volatile uint32_t*)(gacc + (uint64_t)N*9); /* [N*9] bf16 */
        volatile uint32_t *depth=(volatile uint32_t*)(coeff+ (uint64_t)N*9); /* [N] f32 bits */
        volatile float *pub  =(volatile float*)(depth + (uint64_t)N);        /* [N*6] */
        float Rv[9],tv[3],fx,fy,cx,cy;
        for(int i=0;i<9;i++)Rv[i]=cam[i]; for(int i=0;i<3;i++)tv[i]=cam[9+i];
        fx=cam[12];fy=cam[13];cx=cam[14];cy=cam[15];
        uint32_t cmd=CMD[0];
        int slot=hdr[21]; if(slot<0||slot>=32)slot=0;   /* per-worker slot for produce/consume (grid) */
        uint64_t c0=rdcycle();

        if(cmd==2u){                         /* PROJECT+WHITEN across NH harts (Gaussians g%NH==hid), zero gacc/loss */
            int NH=((volatile int*)NHARTS_A)[0]; if(NH<1)NH=1; if(NH>4)NH=4;
            ((volatile int*)WCMD_A)[0]=2;
            for(int h=1;h<NH;h++) ((volatile uint32_t*)(HGO+(uint32_t)h*0x40u))[0]=ring;   /* wake workers */
            proj_slice(0, NH);                                                             /* hart 0's stripe */
            for(int h=1;h<NH;h++){ volatile uint32_t* hd=(volatile uint32_t*)(HDONE+(uint32_t)h*0x40u);
                uint32_t to=0u; while(hd[0]!=ring){ if(++to>40000000u) break; } }          /* barrier */
        }
        else if(cmd==5u){                    /* PRODUCE tile operands from coeff[] by the tile's id list */
            int K=idl[0]; if(K>16)K=16;
            int ids[16]; float dz[16];
            for(int i=0;i<K;i++){ ids[i]=idl[1+i]; union{uint32_t u; float f;} d; d.u=depth[ids[i]]; dz[i]=d.f; }
            /* depth sort (ascending) -> slot order */
            int sl[16]; for(int i=0;i<K;i++) sl[i]=i;
            for(int i=1;i<K;i++){ int k=sl[i]; float kz=dz[k]; int j=i-1;
                while(j>=0 && dz[sl[j]]>kz){ sl[j+1]=sl[j]; j--; } sl[j+1]=k; }
            uint32_t opb=OPBASE+(uint32_t)slot*OPSTRIDE; volatile int *ors=ordr+slot*16;
            volatile uint32_t *PSI=(volatile uint32_t*)(opb+0*0x800u), *DOP=(volatile uint32_t*)(opb+1*0x800u),
                *DNOP=(volatile uint32_t*)(opb+2*0x800u), *COL=(volatile uint32_t*)(opb+3*0x800u),
                *COLT=(volatile uint32_t*)(opb+4*0x800u), *OPB=(volatile uint32_t*)(opb+5*0x800u);
            for(int w=0;w<512;w++){ PSI[w]=0;DOP[w]=0;DNOP[w]=0;COL[w]=0;COLT[w]=0;OPB[w]=0; }
            for(int i=0;i<K;i++){
                int gid=ids[sl[i]]; ors[i]=gid;
                volatile uint32_t *co=coeff+(uint64_t)gid*9;
                uint32_t sa=co[0],m12=co[1],m22=co[2],c1=co[3],c2=co[4],op=co[5],rr=co[6],gg=co[7],bb=co[8];
                tput(PSI,0,2*i,sa); tput(PSI,1,2*i,m12); tput(PSI,2,2*i,c1);
                tput(PSI,1,2*i+1,m22); tput(PSI,2,2*i+1,c2);
                tput(DOP,i,i,op); tput(DNOP,i,i,op^0x8000u);
                tput(COL,i,0,rr); tput(COL,i,1,gg); tput(COL,i,2,bb);
                tput(COLT,0,i,rr); tput(COLT,1,i,gg); tput(COLT,2,i,bb);
                for(int p=0;p<32;p++) tput(OPB,p,i,op);
            }
            for(int k=K;k<32;k++) for(int p=0;p<32;p++) tput(OPB,p,k,0x3F00u);
        }
        else if(cmd==6u){                    /* CONSUME tile grads: detilize inbox -> scatter-add into gacc[global] */
            int K=idl[0]; if(K>16)K=16;
            uint32_t gib=GINBOX+(uint32_t)slot*GISTRIDE; volatile int *ors=ordr+slot*16;
            volatile uint32_t *DP=(volatile uint32_t*)(gib+0*0x800u),*DO=(volatile uint32_t*)(gib+1*0x800u),
                *WW=(volatile uint32_t*)(gib+2*0x800u),*DC=(volatile uint32_t*)(gib+3*0x800u);
            for(int i=0;i<K;i++){
                int gid=ors[i]; volatile float *ga=gacc+(uint64_t)gid*9;
                ga[0]+=fsan(tget(DP,0,2*i)); ga[1]+=fsan(tget(DP,1,2*i)); ga[2]+=fsan(tget(DP,2,2*i));
                ga[3]+=fsan(tget(DP,1,2*i+1)); ga[4]+=fsan(tget(DP,2,2*i+1)); ga[5]+=fsan(tget(DO,0,i));
                float dc0=0,dc1=0,dc2=0;
                for(int p=0;p<32;p++){ float w=fsan(tget(WW,p,i)); dc0+=w*fsan(tget(DC,p,0)); dc1+=w*fsan(tget(DC,p,1)); dc2+=w*fsan(tget(DC,p,2)); }
                ga[6]+=fsan(dc0); ga[7]+=fsan(dc1); ga[8]+=fsan(dc2);
            }
            float sse=0; for(int p=0;p<32;p++) for(int ch=0;ch<3;ch++){ float e=fsan(tget(DC,p,ch)); sse+=e*e; }
            loss[0]+=sse;
        }
        else if(cmd==1u){                    /* ADAM across NH harts (Gaussians g%NH==hid); merge per-hart gacc/loss */
            int NH=((volatile int*)NHARTS_A)[0]; if(NH<1)NH=1; if(NH>4)NH=4;
            ((volatile int*)WCMD_A)[0]=1;
            for(int h=1;h<NH;h++) ((volatile uint32_t*)(HGO+(uint32_t)h*0x40u))[0]=ring;   /* wake workers */
            adam_slice(0, NH);                                                             /* hart 0's stripe */
            for(int h=1;h<NH;h++){ volatile uint32_t* hd=(volatile uint32_t*)(HDONE+(uint32_t)h*0x40u);
                uint32_t to=0u; while(hd[0]!=ring){ if(++to>40000000u) break; } }          /* barrier */
            { float ls=0.0f; for(int h=0;h<NH;h++) ls+=((volatile float*)(LOSS_H+(uint32_t)h*0x40u))[0]; loss[0]=ls; }
        }
        else if(cmd==7u){                    /* ORCHESTRATE one tile: produce -> signal -> wait ack -> consume */
            int K=idl[0]; if(K>16)K=16;
            produce_ops(slot, idl, coeff, depth, ordr);
            { volatile uint32_t* FLAG=(volatile uint32_t*)(FLAG_B+(uint32_t)slot*ASTRIDE);
              volatile uint32_t* ACK =(volatile uint32_t*)(ACK_B +(uint32_t)slot*ASTRIDE);
              FLAG[0]=ring; uint32_t to=0u; while(ACK[0]!=ring){ if(++to>20000000u) break; }
              TELE[4]=ACK[0]; TELE[5]=ring; }
            consume_slot(slot, K, ordr, gacc, (volatile float*)LOSS_H);   /* hart-0 partial; cmd1 sums LOSS_H->loss */
        }
        else if(cmd==8u){                    /* PRODUCE this slot's phi/phi2T/gt -> PXBASE (once/tile) */
            produce_pix(slot, hdr[22], hdr[23], 16, (volatile float*)TGT_IMG);   /* single-tile: local 16x16 image */
        }
        else if(cmd==9u){                    /* ORCHESTRATE a BATCH across NH harts (slots partitioned s%NH==hid) */
            int NH=((volatile int*)NHARTS_A)[0]; if(NH<1)NH=1; if(NH>4)NH=4;
            ((volatile int*)WCMD_A)[0]=9;                                                  /* workers run cmd9_slice */
            for(int h=1;h<NH;h++) ((volatile uint32_t*)(HGO+(uint32_t)h*0x40u))[0]=ring;   /* wake workers */
            cmd9_slice(0, NH, ring);                                                       /* hart 0's own slots */
            { uint32_t ndone=1u;                                                           /* barrier: workers done */
              for(int h=1;h<NH;h++){ volatile uint32_t* hd=(volatile uint32_t*)(HDONE+(uint32_t)h*0x40u);
                uint32_t to=0u; while(hd[0]!=ring){ if(++to>40000000u) break; } if(hd[0]==ring) ndone++; }
              TELE[5]=ring; TELE[6]=ndone; TELE[7]=(uint32_t)NH; }
        }
        else if(cmd==11u){                   /* ON-DEVICE BIN: pub -> per-tile front-12 id-lists (IDLGB) + occupancy (OCC) */
            int IMGW=((volatile int*)0x30005DF4u)[0]; if(IMGW<=0)IMGW=16;
            int IMGH=((volatile int*)IMGH_A)[0]; if(IMGH<=0)IMGH=16;
            int ntx=IMGW/16, nty=IMGH/16, ntile=ntx*nty;
            for(int t=0;t<ntile;t++){ ((volatile int*)(IDLGB+(uint32_t)t*0x40u))[0]=0; ((volatile int*)OCC)[t]=0; }
            for(uint32_t i=0;i<N;i++){
                volatile float* pu=pub+(uint64_t)i*6;
                float gx=pu[0],gy=pu[1],a=pu[2],b=pu[3],c=pu[4],dep=pu[5];
                float det=a*c-b*b; if(det<=0.0f) continue;
                float A=c/det, Cc=a/det;                                            /* Sigma2 diag (screen var x,y) */
                float ex=3.0f*proj_sqrt(A>0.0f?A:0.0f), ey=3.0f*proj_sqrt(Cc>0.0f?Cc:0.0f);
                int tx0=flr((gx-ex)*0.0625f); if(tx0<0)tx0=0; int tx1=flr((gx+ex)*0.0625f); if(tx1>ntx-1)tx1=ntx-1;
                int ty0=flr((gy-ey)*0.0625f); if(ty0<0)ty0=0; int ty1=flr((gy+ey)*0.0625f); if(ty1>nty-1)ty1=nty-1;
                if(tx1<tx0||ty1<ty0) continue;
                for(int ty=ty0;ty<=ty1;ty++) for(int tx=tx0;tx<=tx1;tx++) bin_insert(ty*ntx+tx, (int)i, dep);
            }
            for(int t=0;t<ntile;t++){                          /* OCC = real count; pad id-list to K=12 (match host) */
                volatile int* L=(volatile int*)(IDLGB+(uint32_t)t*0x40u); int cnt=L[0]; ((volatile int*)OCC)[t]=cnt;
                if(cnt>0){ int last=L[cnt]; for(int p=cnt;p<12;p++) L[1+p]=last; L[0]=12; }
            }
        }
        uint64_t c1=rdcycle();
        TELE[1]=cmd; TELE[2]=(uint32_t)(c1-c0); TELE[3]=(uint32_t)((c1-c0)>>32);
        last=ring; DONE[0]=ring;
    }
    return 0;
}
