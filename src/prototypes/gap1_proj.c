/* gap1_proj.c — PASTE-READY scalar C for the x280 Gap-1 projection front/back end, host-validated
 * against gap1_proj_golden.py (which is validated vs torch autograd). Pure scalar float math, no
 * matrix libs — exactly what cb_whiten.c (forward) and opt_step.c (backward) run.
 *
 *   proj_fwd: (mean3, scale_log3, quat4, camera) -> (gx,gy,depth, a,b,c)   [cb_whiten.c]
 *   proj_bwd: (mean3, scale_log3, quat4, camera, dL/d[a,b,c,gx,gy])        [opt_step.c]
 *             -> (dmean3, dscale_log3, dquat4)   -- RECOMPUTES forward internals from params,
 *                exactly like opt_step.c already recomputes sa,m12,m22 from resident (a,b,c).
 *
 * Camera: Rv[9] row-major (world->cam), tv[3], fx,fy,cx,cy. Conventions per gap1_proj_golden.py.
 * Build (host cross-check):  gcc -O2 -o gap1_proj gap1_proj.c -lm
 * The x280 build is the same scalar code; on device 'float' + fsqrt.s, no libm (sqrtf inlined). */
#include <math.h>
#include <stdio.h>

#define EPS_DET 1e-9f
#define EPS_Z   1e-4f
#define BLUR    0.3f

/* quat (w,x,y,z, raw) -> normalized qn[4] + R[9] row-major; returns 1/qnorm-independent qn stashed. */
static void quat_to_rot(const float q[4], float R[9], float qn[4], float *qnorm_out){
    float qnorm = sqrtf(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    float w=q[0]/qnorm, x=q[1]/qnorm, y=q[2]/qnorm, z=q[3]/qnorm;
    qn[0]=w; qn[1]=x; qn[2]=y; qn[3]=z; *qnorm_out=qnorm;
    R[0]=1-2*(y*y+z*z); R[1]=2*(x*y-w*z);   R[2]=2*(x*z+w*y);
    R[3]=2*(x*y+w*z);   R[4]=1-2*(x*x+z*z); R[5]=2*(y*z-w*x);
    R[6]=2*(x*z-w*y);   R[7]=2*(y*z+w*x);   R[8]=1-2*(x*x+y*y);
}

/* Shared forward core: fills camera-space + covariance internals used by both fwd output and bwd. */
typedef struct {
    float R[9], qn[4], qnorm, S2[3];
    float mc[3], z, z_raw;
    float j0,j1,j2,j3;                 /* J: [[j0,0,j2],[0,j1,j3]] */
    float Scam[9];                     /* Sig_cam (symmetric, full 3x3 stored) */
    float A,B,C,det;
} pcore;

static void proj_core(const float mean[3], const float sl[3], const float q[4],
                      const float Rv[9], const float tv[3], float fx, float fy,
                      pcore *o){
    quat_to_rot(q, o->R, o->qn, &o->qnorm);
    for(int k=0;k<3;k++) o->S2[k]=expf(2.0f*sl[k]);
    /* Sig3 = R diag(S2) R^T */
    float Sig3[9];
    for(int i=0;i<3;i++) for(int j=0;j<3;j++)
        Sig3[i*3+j]=o->R[i*3+0]*o->S2[0]*o->R[j*3+0]
                   +o->R[i*3+1]*o->S2[1]*o->R[j*3+1]
                   +o->R[i*3+2]*o->S2[2]*o->R[j*3+2];
    /* mc = Rv mean + tv */
    for(int i=0;i<3;i++) o->mc[i]=Rv[i*3+0]*mean[0]+Rv[i*3+1]*mean[1]+Rv[i*3+2]*mean[2]+tv[i];
    o->z_raw=o->mc[2]; o->z = o->z_raw>EPS_Z? o->z_raw:EPS_Z;
    float z=o->z, z2=z*z;
    o->j0=fx/z; o->j2=-fx*o->mc[0]/z2; o->j1=fy/z; o->j3=-fy*o->mc[1]/z2;
    /* Sig_cam = Rv Sig3 Rv^T */
    for(int i=0;i<3;i++) for(int j=0;j<3;j++){
        float s=0;
        for(int p=0;p<3;p++) for(int qq=0;qq<3;qq++) s+=Rv[i*3+p]*Sig3[p*3+qq]*Rv[j*3+qq];
        o->Scam[i*3+j]=s;
    }
    /* Sig2 = J Scam J^T + BLUR I ; J rows: r0=(j0,0,j2), r1=(0,j1,j3) */
    float j0=o->j0,j1=o->j1,j2=o->j2,j3=o->j3;
    float *M=o->Scam;
    o->A = j0*j0*M[0] + 2*j0*j2*M[2] + j2*j2*M[8] + BLUR;
    o->C = j1*j1*M[4] + 2*j1*j3*M[5] + j3*j3*M[8] + BLUR;
    o->B = j0*j1*M[1] + j0*j3*M[2] + j2*j1*M[5] + j2*j3*M[8];
    o->det = o->A*o->C - o->B*o->B + EPS_DET;
}

void proj_fwd(const float mean[3], const float sl[3], const float q[4],
              const float Rv[9], const float tv[3], float fx, float fy, float cx, float cy,
              float *gx, float *gy, float *depth, float *a, float *b, float *c){
    pcore o; proj_core(mean,sl,q,Rv,tv,fx,fy,&o);
    *gx = fx*o.mc[0]/o.z + cx;
    *gy = fy*o.mc[1]/o.z + cy;
    *depth = o.z_raw;
    *a = o.C/o.det; *b = -o.B/o.det; *c = o.A/o.det;
}

/* dR[i]/dqn as flat: returns 4x9. Matches _dR_dqn in the golden. */
static void dR_dqn(const float qn[4], float dR[4][9]){
    float w=qn[0],x=qn[1],y=qn[2],z=qn[3];
    float d0[9]={0,-2*z,2*y, 2*z,0,-2*x, -2*y,2*x,0};
    float d1[9]={0,2*y,2*z, 2*y,-4*x,-2*w, 2*z,2*w,-4*x};
    float d2[9]={-4*y,2*x,2*w, 2*x,0,2*z, -2*w,2*z,-4*y};
    float d3[9]={-4*z,-2*w,2*x, 2*w,-4*z,2*y, 2*x,2*y,0};
    for(int k=0;k<9;k++){dR[0][k]=d0[k];dR[1][k]=d1[k];dR[2][k]=d2[k];dR[3][k]=d3[k];}
}

void proj_bwd(const float mean[3], const float sl[3], const float q[4],
              const float Rv[9], const float tv[3], float fx, float fy, float cx, float cy,
              float da, float db, float dc, float dgx, float dgy,
              float dmean[3], float dsl[3], float dq[4]){
    pcore o; proj_core(mean,sl,q,Rv,tv,fx,fy,&o);
    float A=o.A,B=o.B,C=o.C,det=o.det, D2=det*det;
    float z=o.z, z2=z*z, z3=z2*z, mc0=o.mc[0], mc1=o.mc[1];
    float j0=o.j0,j1=o.j1,j2=o.j2,j3=o.j3;

    /* (a,b,c) <- (A,B,C): exact scalar partials */
    float dA = da*(-C*C/D2)      + db*(B*C/D2)              + dc*(1.0f/det - A*C/D2);
    float dB = da*(2*B*C/D2)     + db*(-1.0f/det - 2*B*B/D2)+ dc*(2*A*B/D2);
    float dC = da*(1.0f/det - A*C/D2) + db*(A*B/D2)         + dc*(-A*A/D2);
    /* GSig2 full symmetric: [[dA, dB/2],[dB/2, dC]] */
    float g00=dA, g01=dB*0.5f, g11=dC;

    /* GJ = 2 GSig2 J Scam   (2x3).  Also GScam = J^T GSig2 J (3x3). */
    float *M=o.Scam;
    /* (GSig2 J) is 2x3: row-combine of J rows by GSig2 */
    /* GSig2*J : rows i of GSig2 (g00,g01 / g01,g11) times J(2x3) */
    float GJ[2][3];
    {   /* T = GSig2 (2x2) @ J (2x3) */
        float t00=g00*j0+g01*0, t01=g00*0+g01*j1, t02=g00*j2+g01*j3;
        float t10=g01*j0+g11*0, t11=g01*0+g11*j1, t12=g01*j2+g11*j3;
        /* GJ = 2 * T @ Scam (3x3)  -> (2x3) */
        for(int c2=0;c2<3;c2++){
            GJ[0][c2]=2.0f*(t00*M[0*3+c2]+t01*M[1*3+c2]+t02*M[2*3+c2]);
            GJ[1][c2]=2.0f*(t10*M[0*3+c2]+t11*M[1*3+c2]+t12*M[2*3+c2]);
        }
    }
    /* GScam = J^T GSig2 J  (3x3, symmetric). J^T is 3x2. */
    float Jt[3][2]={{j0,0},{0,j1},{j2,j3}};
    float GScam[9];
    {   /* U = J^T (3x2) @ GSig2 (2x2) = 3x2 */
        float u[3][2];
        for(int i=0;i<3;i++){ u[i][0]=Jt[i][0]*g00+Jt[i][1]*g01; u[i][1]=Jt[i][0]*g01+Jt[i][1]*g11; }
        /* GScam = U (3x2) @ J (2x3) */
        for(int i=0;i<3;i++) for(int c2=0;c2<3;c2++){
            float Jr0 = (c2==0? j0 : (c2==2? j2 : 0.0f));    /* J row0 */
            float Jr1 = (c2==1? j1 : (c2==2? j3 : 0.0f));    /* J row1 */
            GScam[i*3+c2]=u[i][0]*Jr0 + u[i][1]*Jr1;
        }
    }

    /* dmc from screen mean + J's dependence on mc */
    float dmc[3]={0,0,0};
    dmc[0]+=dgx*fx/z;               dmc[1]+=dgy*fy/z;
    dmc[2]+=dgx*(-fx*mc0/z2)+dgy*(-fy*mc1/z2);
    dmc[2]+=GJ[0][0]*(-fx/z2);
    dmc[0]+=GJ[0][2]*(-fx/z2); dmc[2]+=GJ[0][2]*(2.0f*fx*mc0/z3);
    dmc[2]+=GJ[1][1]*(-fy/z2);
    dmc[1]+=GJ[1][2]*(-fy/z2); dmc[2]+=GJ[1][2]*(2.0f*fy*mc1/z3);
    if(o.z_raw<=EPS_Z) dmc[2]=0.0f;

    /* GSig3 = Rv^T GScam Rv ; symmetrize */
    float GSig3[9];
    for(int i=0;i<3;i++) for(int j=0;j<3;j++){
        float s=0; for(int p=0;p<3;p++) for(int qq=0;qq<3;qq++) s+=Rv[p*3+i]*GScam[p*3+qq]*Rv[qq*3+j];
        GSig3[i*3+j]=s;
    }
    float GS[9];
    for(int i=0;i<3;i++) for(int j=0;j<3;j++) GS[i*3+j]=0.5f*(GSig3[i*3+j]+GSig3[j*3+i]);

    /* GR = 2 GS R diag(S2) ; dS2 = diag(R^T GS R) */
    float *R=o.R; float S2[3]={o.S2[0],o.S2[1],o.S2[2]};
    float GR[9];
    for(int i=0;i<3;i++) for(int k=0;k<3;k++){
        float s=0; for(int p=0;p<3;p++) s+=GS[i*3+p]*R[p*3+k];
        GR[i*3+k]=2.0f*s*S2[k];
    }
    float dS2[3];
    for(int k=0;k<3;k++){
        float s=0; for(int p=0;p<3;p++) for(int r=0;r<3;r++) s+=R[p*3+k]*GS[p*3+r]*R[r*3+k];
        dS2[k]=s;
    }
    for(int k=0;k<3;k++) dsl[k]=dS2[k]*2.0f*S2[k];

    /* quat: GR -> dqn -> dq (normalization) */
    float dRq[4][9]; dR_dqn(o.qn, dRq);
    float dqn[4];
    for(int i=0;i<4;i++){ float s=0; for(int k=0;k<9;k++) s+=GR[k]*dRq[i][k]; dqn[i]=s; }
    float dot=0; for(int i=0;i<4;i++) dot+=o.qn[i]*dqn[i];
    for(int i=0;i<4;i++) dq[i]=(dqn[i]-o.qn[i]*dot)/o.qnorm;

    /* dmean = Rv^T dmc */
    for(int i=0;i<3;i++) dmean[i]=Rv[0*3+i]*dmc[0]+Rv[1*3+i]*dmc[1]+Rv[2*3+i]*dmc[2];
}

/* test harness: read cases from stdin, print fwd+bwd. Line: mean(3) sl(3) q(4) da db dc dgx dgy */
int main(void){
    /* fixed camera matching the golden self-test */
    float Rv[9]={1,0,0, 0,1,0, 0,0,1}, tv[3]={0,0,6}; float fx=70,fy=70,cx=32,cy=32;
    float mean[3],sl[3],q[4],da,db,dc,dgx,dgy;
    while(scanf("%f %f %f %f %f %f %f %f %f %f %f %f %f",
        &mean[0],&mean[1],&mean[2], &sl[0],&sl[1],&sl[2], &q[0],&q[1],&q[2],&q[3],
        &da,&db,&dc)==13 && scanf("%f %f",&dgx,&dgy)==2){
        float gx,gy,dep,a,b,c; proj_fwd(mean,sl,q,Rv,tv,fx,fy,cx,cy,&gx,&gy,&dep,&a,&b,&c);
        float dm[3],ds[3],dq[4]; proj_bwd(mean,sl,q,Rv,tv,fx,fy,cx,cy,da,db,dc,dgx,dgy,dm,ds,dq);
        printf("%.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",
            gx,gy,a,b,c, dm[0],dm[1],dm[2], ds[0],ds[1],ds[2], dq[0],dq[1],dq[2],dq[3]);
    }
    return 0;
}
