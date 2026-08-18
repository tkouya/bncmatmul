#include <cstdio>
#include <cmath>
#include "gtslinear.h"
__global__ void k(gts_real*a,gts_real*b,gts_real*q){ *q = (*a)/(*b); }   /* device divide */
static double s3(float*v){return (double)v[0]+v[1]+v[2];}
int main(void){
  /* (sqrt2 / sqrt3) on device, compare to host */
  gts_real *da,*db,*dq; cudaMalloc(&da,sizeof(gts_real));cudaMalloc(&db,sizeof(gts_real));cudaMalloc(&dq,sizeof(gts_real));
  float a[TSSIZE],b[TSSIZE],q[TSSIZE];
  rts_set_ui(a,2UL);rts_sqrt(a,a); rts_set_ui(b,3UL);rts_sqrt(b,b);
  cudaMemcpy(da,a,sizeof(gts_real),cudaMemcpyHostToDevice); cudaMemcpy(db,b,sizeof(gts_real),cudaMemcpyHostToDevice);
  k<<<1,1>>>(da,db,dq); cudaDeviceSynchronize();
  cudaMemcpy(q,dq,sizeof(gts_real),cudaMemcpyDeviceToHost);
  float qh[TSSIZE]; rts_div(qh,a,b);
  printf("gts device div: GPU=%.10e CPU=%.10e absdiff=%.3e\n", s3(q), s3(qh), fabs(s3(q)-s3(qh)));
  /* (a/b)*b == a ? */
  float chk[TSSIZE]; rts_mul(chk,qh,b); printf("CPU (sqrt2/sqrt3)*sqrt3 - sqrt2 = %.3e\n", s3(chk)-s3(a));
  return 0;
}
