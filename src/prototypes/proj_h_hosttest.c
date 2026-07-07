#include <stdio.h>
#include "proj.h"
int main(void){
    float Rv[9]={1,0,0, 0,1,0, 0,0,1}, tv[3]={0,0,6}; float fx=70,fy=70,cx=32,cy=32;
    float mean[3],sl[3],q[4],da,db,dc,dgx,dgy;
    while(scanf("%f %f %f %f %f %f %f %f %f %f %f %f %f",
        &mean[0],&mean[1],&mean[2], &sl[0],&sl[1],&sl[2], &q[0],&q[1],&q[2],&q[3],
        &da,&db,&dc)==13 && scanf("%f %f",&dgx,&dgy)==2){
        float gx,gy,dep,a,b,c; proj_fwd(mean,sl,q,Rv,tv,fx,fy,cx,cy,&gx,&gy,&dep,&a,&b,&c);
        float dm[3],ds[3],dq[4]; proj_bwd(mean,sl,q,Rv,tv,fx,fy,da,db,dc,dgx,dgy,dm,ds,dq);
        printf("%.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",
            gx,gy,a,b,c, dm[0],dm[1],dm[2], ds[0],ds[1],ds[2], dq[0],dq[1],dq[2],dq[3]);
    }
    return 0;
}
