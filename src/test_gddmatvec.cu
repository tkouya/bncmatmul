/* test_gddmatvec.cu : verify GPU gdd matvec (mul_gddmatrix_gddvec) vs CPU dd */
#include <cstdio>
#include <cmath>
#include "gddlinear.h"
int main(void){
  long int i,j,dim=127; int blocks=64,threads=128;
  fpu_fix_start(NULL);
  DDMatrix a=init_ddmatrix(dim,dim); DDVector x=init_ddvector(dim),yc=init_ddvector(dim),yg=init_ddvector(dim);
  double t[DDSIZE];
  for(i=0;i<dim;i++){ for(j=0;j<dim;j++){ rdd_set_ui(t,(unsigned long)((i+j)%7+1)); rdd_sqrt(t,t); set_ddmatrix_ij(a,i,j,t);} rdd_set_ui(t,(unsigned long)(i%5+1)); rdd_sqrt(t,t); set_ddvector_i(x,i,t); }
  mul_ddmatrix_ddvec(yc,a,x);  /* CPU ref */
  GDDMatrix ga=init_gddmatrix_dev(dim,dim); GDDVector gx=init_gddvector_dev(dim),gy=init_gddvector_dev(dim);
  subst_gddmatrix_dev_ddmat(ga,a); subst_gddvector_dev_ddvec(gx,x);
  mul_gddmatrix_gddvec(gy,ga,gx,blocks,threads); cudaDeviceSynchronize();
  subst_ddvector_gddvec_dev(yg,gy);
  double mr=0,d[DDSIZE],q[DDSIZE],cc[DDSIZE];
  for(i=0;i<dim;i++){ double *g=get_ddvector_i(yg,i),*c=get_ddvector_i(yc,i); rdd_sub(d,g,c); rdd_abs(d,d); rdd_abs(cc,c); if(cc[0]!=0.0){rdd_div(q,d,cc); if(q[0]>mr)mr=q[0];} }
  printf("GPU gdd matvec vs CPU dd (dim=%ld): max relative error = %10.3e -> %s\n", dim, mr, mr<1e-28?"PASS":"FAIL");
  return mr<1e-28?0:1;
}
