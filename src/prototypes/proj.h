/* proj.h — Gap-1 3D->2D camera projection FORWARD + ANALYTIC BACKWARD, shared by the x280 kernels
 * (cb_whiten.c forward, opt_step.c backward) AND the host cross-check (scratchpad/gap1_check_c.py).
 * ONE validated source of truth. Portable freestanding C: no libm (self-contained expf, guarded sqrt).
 *
 * Validated vs gap1_proj_golden.py (itself vs torch autograd to 1.4e-14): fp32 rel error ~3.95e-4.
 * Convention = ttnn ref project (tt-splat/docs/pathclear/train3d.py):
 *   Sig3 = R diag(exp(2*scale_log)) R^T (w-first quat) ; mc = Rv mean + tv ;
 *   J = [[fx/z,0,-fx mc0/z^2],[0,fy/z,-fy mc1/z^2]] ; Sig2 = J (Rv Sig3 Rv^T) J^T + 0.3 I ; invert -> (a,b,c).
 */
#ifndef GAP1_PROJ_H
#define GAP1_PROJ_H

#define PROJ_EPS_DET 1e-9f
#define PROJ_EPS_Z   1e-4f
#define PROJ_BLUR    0.3f

#ifdef __riscv
static inline float proj_sqrt(float x){ float r; __asm__("fsqrt.s %0,%1":"=f"(r):"f"(x)); return r; }
#else
#include <math.h>
static inline float proj_sqrt(float x){ return sqrtf(x); }
#endif

/* self-contained expf: 2^(x*log2e), n=round(y) via 2^n float-bits, 2^f via degree-5 Taylor of 2^f. */
static inline float proj_expf(float x){
    if(x >  80.0f) x =  80.0f;
    if(x < -80.0f) x = -80.0f;
    float y = x * 1.44269504088896341f;
    int n = (int)(y >= 0.0f ? y + 0.5f : y - 0.5f);
    float f = y - (float)n;
    float p = 0.00133335f;
    p = p*f + 0.00961813f; p = p*f + 0.0555041f; p = p*f + 0.2402265f; p = p*f + 0.6931472f; p = p*f + 1.0f;
    union { unsigned u; float f; } v; v.u = (unsigned)((n + 127) << 23);
    return v.f * p;
}

static inline void proj_quat_to_rot(const float q[4], float R[9], float qn[4], float *qnorm_out){
    float qnorm = proj_sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    float w=q[0]/qnorm, x=q[1]/qnorm, y=q[2]/qnorm, z=q[3]/qnorm;
    qn[0]=w; qn[1]=x; qn[2]=y; qn[3]=z; *qnorm_out=qnorm;
    R[0]=1-2*(y*y+z*z); R[1]=2*(x*y-w*z);   R[2]=2*(x*z+w*y);
    R[3]=2*(x*y+w*z);   R[4]=1-2*(x*x+z*z); R[5]=2*(y*z-w*x);
    R[6]=2*(x*z-w*y);   R[7]=2*(y*z+w*x);   R[8]=1-2*(x*x+y*y);
}

typedef struct {
    float R[9], qn[4], qnorm, S2[3];
    float mc[3], z, z_raw;
    float j0,j1,j2,j3;                 /* J: [[j0,0,j2],[0,j1,j3]] */
    float Scam[9];                     /* Sig_cam (symmetric 3x3) */
    float A,B,C,det;
} proj_core_t;

static inline void proj_core(const float mean[3], const float sl[3], const float q[4],
                             const float Rv[9], const float tv[3], float fx, float fy, proj_core_t *o){
    proj_quat_to_rot(q, o->R, o->qn, &o->qnorm);
    for(int k=0;k<3;k++) o->S2[k]=proj_expf(2.0f*sl[k]);
    float Sig3[9];
    for(int i=0;i<3;i++) for(int j=0;j<3;j++)
        Sig3[i*3+j]=o->R[i*3+0]*o->S2[0]*o->R[j*3+0]
                   +o->R[i*3+1]*o->S2[1]*o->R[j*3+1]
                   +o->R[i*3+2]*o->S2[2]*o->R[j*3+2];
    for(int i=0;i<3;i++) o->mc[i]=Rv[i*3+0]*mean[0]+Rv[i*3+1]*mean[1]+Rv[i*3+2]*mean[2]+tv[i];
    o->z_raw=o->mc[2]; o->z = o->z_raw>PROJ_EPS_Z? o->z_raw:PROJ_EPS_Z;
    float z=o->z, z2=z*z;
    o->j0=fx/z; o->j2=-fx*o->mc[0]/z2; o->j1=fy/z; o->j3=-fy*o->mc[1]/z2;
    for(int i=0;i<3;i++) for(int j=0;j<3;j++){
        float s=0;
        for(int p=0;p<3;p++) for(int qq=0;qq<3;qq++) s+=Rv[i*3+p]*Sig3[p*3+qq]*Rv[j*3+qq];
        o->Scam[i*3+j]=s;
    }
    float j0=o->j0,j1=o->j1,j2=o->j2,j3=o->j3; float *M=o->Scam;
    o->A = j0*j0*M[0] + 2*j0*j2*M[2] + j2*j2*M[8] + PROJ_BLUR;
    o->C = j1*j1*M[4] + 2*j1*j3*M[5] + j3*j3*M[8] + PROJ_BLUR;
    o->B = j0*j1*M[1] + j0*j3*M[2] + j2*j1*M[5] + j2*j3*M[8];
    o->det = o->A*o->C - o->B*o->B + PROJ_EPS_DET;
}

/* FORWARD: 3D params + camera -> (gx,gy, depth, a,b,c). Used by cb_whiten.c. */
static inline void proj_fwd(const float mean[3], const float sl[3], const float q[4],
                            const float Rv[9], const float tv[3], float fx, float fy, float cx, float cy,
                            float *gx, float *gy, float *depth, float *a, float *b, float *c){
    proj_core_t o; proj_core(mean,sl,q,Rv,tv,fx,fy,&o);
    *gx = fx*o.mc[0]/o.z + cx;
    *gy = fy*o.mc[1]/o.z + cy;
    *depth = o.z_raw;
    *a = o.C/o.det; *b = -o.B/o.det; *c = o.A/o.det;
}

static inline void proj_dR_dqn(const float qn[4], float dR[4][9]){
    float w=qn[0],x=qn[1],y=qn[2],z=qn[3];
    float d0[9]={0,-2*z,2*y, 2*z,0,-2*x, -2*y,2*x,0};
    float d1[9]={0,2*y,2*z, 2*y,-4*x,-2*w, 2*z,2*w,-4*x};
    float d2[9]={-4*y,2*x,2*w, 2*x,0,2*z, -2*w,2*z,-4*y};
    float d3[9]={-4*z,-2*w,2*x, 2*w,-4*z,2*y, 2*x,2*y,0};
    for(int k=0;k<9;k++){dR[0][k]=d0[k];dR[1][k]=d1[k];dR[2][k]=d2[k];dR[3][k]=d3[k];}
}

/* BACKWARD: dL/d(a,b,c,gx,gy) -> dL/d(mean3, scale_log3, quat4). Recomputes fwd internals from params.
 * Used by opt_step.c (after the existing whiten-backward that produces dgx,dgy,da,db,dc). */
static inline void proj_bwd(const float mean[3], const float sl[3], const float q[4],
                            const float Rv[9], const float tv[3], float fx, float fy,
                            float da, float db, float dc, float dgx, float dgy,
                            float dmean[3], float dsl[3], float dq[4]){
    proj_core_t o; proj_core(mean,sl,q,Rv,tv,fx,fy,&o);
    float A=o.A,B=o.B,C=o.C,det=o.det, D2=det*det;
    float z=o.z, z2=z*z, z3=z2*z, mc0=o.mc[0], mc1=o.mc[1];
    float j0=o.j0,j1=o.j1,j2=o.j2,j3=o.j3;

    float dA = da*(-C*C/D2)      + db*(B*C/D2)               + dc*(1.0f/det - A*C/D2);
    float dB = da*(2*B*C/D2)     + db*(-1.0f/det - 2*B*B/D2) + dc*(2*A*B/D2);
    float dC = da*(1.0f/det - A*C/D2) + db*(A*B/D2)          + dc*(-A*A/D2);
    float g00=dA, g01=dB*0.5f, g11=dC;

    float *M=o.Scam;
    float GJ[2][3];
    {   float t00=g00*j0, t01=g01*j1, t02=g00*j2+g01*j3;
        float t10=g01*j0, t11=g11*j1, t12=g01*j2+g11*j3;
        for(int c2=0;c2<3;c2++){
            GJ[0][c2]=2.0f*(t00*M[0*3+c2]+t01*M[1*3+c2]+t02*M[2*3+c2]);
            GJ[1][c2]=2.0f*(t10*M[0*3+c2]+t11*M[1*3+c2]+t12*M[2*3+c2]);
        }
    }
    float GScam[9];
    {   float Jt[3][2]={{j0,0},{0,j1},{j2,j3}};
        float u[3][2];
        for(int i=0;i<3;i++){ u[i][0]=Jt[i][0]*g00+Jt[i][1]*g01; u[i][1]=Jt[i][0]*g01+Jt[i][1]*g11; }
        for(int i=0;i<3;i++) for(int c2=0;c2<3;c2++){
            float Jr0=(c2==0? j0 : (c2==2? j2 : 0.0f));
            float Jr1=(c2==1? j1 : (c2==2? j3 : 0.0f));
            GScam[i*3+c2]=u[i][0]*Jr0 + u[i][1]*Jr1;
        }
    }

    float dmc[3]={0,0,0};
    dmc[0]+=dgx*fx/z;               dmc[1]+=dgy*fy/z;
    dmc[2]+=dgx*(-fx*mc0/z2)+dgy*(-fy*mc1/z2);
    dmc[2]+=GJ[0][0]*(-fx/z2);
    dmc[0]+=GJ[0][2]*(-fx/z2); dmc[2]+=GJ[0][2]*(2.0f*fx*mc0/z3);
    dmc[2]+=GJ[1][1]*(-fy/z2);
    dmc[1]+=GJ[1][2]*(-fy/z2); dmc[2]+=GJ[1][2]*(2.0f*fy*mc1/z3);
    if(o.z_raw<=PROJ_EPS_Z) dmc[2]=0.0f;

    float GSig3[9];
    for(int i=0;i<3;i++) for(int j=0;j<3;j++){
        float s=0; for(int p=0;p<3;p++) for(int qq=0;qq<3;qq++) s+=Rv[p*3+i]*GScam[p*3+qq]*Rv[qq*3+j];
        GSig3[i*3+j]=s;
    }
    float GS[9];
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) GS[i*3+j]=0.5f*(GSig3[i*3+j]+GSig3[j*3+i]);

    float *R=o.R; float S2[3]={o.S2[0],o.S2[1],o.S2[2]};
    float GR[9];
    for(int i=0;i<3;i++) for(int k=0;k<3;k++){
        float s=0; for(int p=0;p<3;p++) s+=GS[i*3+p]*R[p*3+k];
        GR[i*3+k]=2.0f*s*S2[k];
    }
    for(int k=0;k<3;k++){
        float s=0; for(int p=0;p<3;p++) for(int r=0;r<3;r++) s+=R[p*3+k]*GS[p*3+r]*R[r*3+k];
        dsl[k]=s*2.0f*S2[k];
    }

    float dRq[4][9]; proj_dR_dqn(o.qn, dRq);
    float dqn[4];
    for(int i=0;i<4;i++){ float s=0; for(int k=0;k<9;k++) s+=GR[k]*dRq[i][k]; dqn[i]=s; }
    float dot=0; for(int i=0;i<4;i++) dot+=o.qn[i]*dqn[i];
    for(int i=0;i<4;i++) dq[i]=(dqn[i]-o.qn[i]*dot)/o.qnorm;

    for(int i=0;i<3;i++) dmean[i]=Rv[0*3+i]*dmc[0]+Rv[1*3+i]*dmc[1]+Rv[2*3+i]*dmc[2];
}

#endif /* GAP1_PROJ_H */
