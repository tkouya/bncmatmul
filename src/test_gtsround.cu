#include <cstdio>
#include <cmath>
#include "gtslinear.h"
static double s3(float*v){return (double)v[0]+v[1]+v[2];}
int main(void){
  long int i,j,dim=20; int bl=64,th=128;
  TSVector v=init_tsvector(dim),v2=init_tsvector(dim);
  float t[TSSIZE];
  for(i=0;i<dim;i++){ rts_set_ui(t,(unsigned long)(i+2)); rts_sqrt(t,t); set_tsvector_i(v,i,t); }
  GTSVector gv=init_gtsvector_dev(dim);
  subst_gtsvector_dev_tsvec(gv,v); subst_tsvector_gtsvec_dev(v2,gv);
  double vr=0; for(i=0;i<dim;i++){ float *a=get_tsvector_i(v,i),*b=get_tsvector_i(v2,i); double e=fabs(s3(a)-s3(b)); if(e>vr)vr=e; }
  printf("vector subst roundtrip max abs diff = %.3e\n", vr);

  TSMatrix A=init_tsmatrix(dim,dim),A2=init_tsmatrix(dim,dim);
  for(i=0;i<dim;i++)for(j=0;j<dim;j++){ rts_set_ui(t,(unsigned long)((i+j)%5+1)); rts_sqrt(t,t); set_tsmatrix_ij(A,i,j,t);}
  GTSMatrix gA=init_gtsmatrix_dev(dim,dim);
  subst_gtsmatrix_dev_tsmat(gA,A); subst_tsmatrix_gtsmat_dev(A2,gA);
  double mr=0; for(i=0;i<dim;i++)for(j=0;j<dim;j++){ float *a=get_tsmatrix_ij(A,i,j),*b=get_tsmatrix_ij(A2,i,j); double e=fabs(s3(a)-s3(b)); if(e>mr)mr=e; }
  printf("matrix subst roundtrip max abs diff = %.3e\n", mr);

  TSVector x=init_tsvector(dim),yc=init_tsvector(dim),yg=init_tsvector(dim);
  for(i=0;i<dim;i++){ rts_set_ui(t,(unsigned long)(i%4+1)); rts_sqrt(t,t); set_tsvector_i(x,i,t);}
  mul_tsmatrix_tsvec(yc,A,x);
  GTSVector gx=init_gtsvector_dev(dim),gy=init_gtsvector_dev(dim);
  subst_gtsvector_dev_tsvec(gx,x); mul_gtsmatrix_gtsvec(gy,gA,gx,bl,th); cudaDeviceSynchronize();
  subst_tsvector_gtsvec_dev(yg,gy);
  double yr=0; for(i=0;i<dim;i++){ float *a=get_tsvector_i(yc,i),*b=get_tsvector_i(yg,i); double m=fabs(s3(a)); double e=m!=0?fabs(s3(a)-s3(b))/m:fabs(s3(a)-s3(b)); if(e>yr)yr=e;}
  printf("gts matvec GPU-vs-CPU max rel = %.3e\n", yr);
  return 0;
}
