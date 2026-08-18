#include <cstdio>
#include "gtslinear.h"
__device__ static inline gts_real gabs(gts_real a){ gts_real z=a-a; return (a.x<0.0)?(z-a):a; }
__global__ void k(gts_real*a,gts_real*b,int*gt,gts_real*aa){
  *gt = (gabs(*b) > gabs(*a)) ? 1 : 0;   /* is |sqrt3| > |sqrt2| ? expect 1 */
  *aa = gabs(*a);
}
static double s3(float*v){return (double)v[0]+v[1]+v[2];}
int main(void){
  gts_real *da,*db,*daa; int *dgt; cudaMalloc(&da,12);cudaMalloc(&db,12);cudaMalloc(&daa,12);cudaMalloc(&dgt,4);
  float a[TSSIZE],b[TSSIZE],aa[TSSIZE]; int gt;
  rts_set_ui(a,2UL);rts_sqrt(a,a); rts_set_d(a,-s3(a));  /* a = -sqrt2 (approx via set_d; use neg) */
  rts_set_ui(a,2UL);rts_sqrt(a,a); a[0]=-a[0];a[1]=-a[1];a[2]=-a[2]; /* a=-sqrt2 exactly */
  rts_set_ui(b,3UL);rts_sqrt(b,b);
  cudaMemcpy(da,a,12,cudaMemcpyHostToDevice);cudaMemcpy(db,b,12,cudaMemcpyHostToDevice);
  k<<<1,1>>>(da,db,dgt,daa); cudaDeviceSynchronize();
  cudaMemcpy(&gt,dgt,4,cudaMemcpyDeviceToHost); cudaMemcpy(aa,daa,12,cudaMemcpyDeviceToHost);
  printf("gts: |sqrt3|>|-sqrt2| = %d (expect 1);  abs(-sqrt2)=%.10e (expect 1.4142...)\n", gt, s3(aa));
  return 0;
}
