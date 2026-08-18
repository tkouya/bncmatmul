#include <cstdio>
#include <cstdlib>
#define USE_DDLINEAR
#include "matmul_strassen.h"
#include "bncsparse.h"
#include "get_secv.h"
#include <qd/qd_real.h>
int main(int argc,char**argv){
  unsigned int cw; fpu_fix_start(&cw);
  long n = argc>=2?atol(argv[1]):8192;
  int hbw = argc>=3?atoi(argv[2]):3;
  DDMatrix Ad=init_ddmatrix(n,n); double tmp[DDSIZE];
  for(long i=0;i<n;i++) for(long j=(i-hbw<0?0:i-hbw); j<=(i+hbw>=n?n-1:i+hbw); j++){
    for(int k=1;k<DDSIZE;k++)tmp[k]=0; tmp[0]=((i+j)%7+1)*0.5; set_ddmatrix_ij(Ad,i,j,tmp);}
  DDRSMatrix A=init_set_ddrsmatrix_ddmatrix(Ad);
  DDVector x=init_ddvector(n),y=init_ddvector(n);
  for(long i=0;i<n;i++){for(int k=1;k<DDSIZE;k++)tmp[k]=0;tmp[0]=(i%9+1)*0.5;set_ddvector_i(x,i,tmp);}
  mul_ddrsmatrix_ddvec(y,A,x); // warm
  long it=0; double t0=get_real_secv(),el=0;
  do{ for(int r=0;r<5;r++) mul_ddrsmatrix_ddvec(y,A,x); it+=5; el=get_real_secv()-t0; }while(el<0.5);
  printf("hbw=%d nnz=%ld  %.4e s/spmv\n", hbw, A->nzero_total_num, el/it);
  return 0;
}
