/* mm_double.c - native double matmul algorithm benchmark (library algorithms).
 * Modes/output identical to mm_multi.c.  Build: -DBACKEND_NAME=\"serial\".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpfr.h>
#include "dlinear.h"
#include "matmul_strassen.h"

#ifndef PREC_NAME
#define PREC_NAME "double"
#endif
#ifndef BACKEND_NAME
#define BACKEND_NAME "scalar"
#endif
#ifndef MIN_DIM
#define MIN_DIM 32
#endif
#ifndef REF_BITS
#define REF_BITS 512
#endif

static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec*1e-9; }
#define TIME_OP(tvar, CALL) do { long it=1; double e; for(;;){ \
    double s=now_s(); for(long _k=0;_k<it;_k++){ CALL; } e=now_s()-s; \
    if(e>=0.20 || it>=(1L<<22)) break; it*=2; } tvar=e/(double)it; } while(0)

int main(int argc, char**argv){
    long dim = (argc>=2)? atol(argv[1]) : 256;
    const char *mode = (argc>=3)? argv[2] : "time";
    long md = MIN_DIM;

    DMatrix A = init_dmatrix(dim,dim), B = init_dmatrix(dim,dim), Cm = init_dmatrix(dim,dim);
    for(long i=0;i<dim;i++) for(long j=0;j<dim;j++){
        set_dmatrix_ij(A,i,j, ((i*131+j*17)%97)*0.01 + 0.1);
        set_dmatrix_ij(B,i,j, ((i*13+j*131)%89)*0.01 + 0.1);
    }

    if(strcmp(mode,"time")==0){
        double t;
        TIME_OP(t, mul_dmatrix(Cm,A,B));                     printf("TIME,%s,%s,triple,%ld,%.9e\n",   PREC_NAME,BACKEND_NAME,dim,t);
        TIME_OP(t, mul_dmatrix_block(Cm,A,B,md));            printf("TIME,%s,%s,block,%ld,%.9e\n",    PREC_NAME,BACKEND_NAME,dim,t);
        TIME_OP(t, mul_dmatrix_strassen(Cm,A,B,md));         printf("TIME,%s,%s,strassen,%ld,%.9e\n", PREC_NAME,BACKEND_NAME,dim,t);
        TIME_OP(t, mul_dmatrix_winograd_even(Cm,A,B,md));    printf("TIME,%s,%s,winograd,%ld,%.9e\n", PREC_NAME,BACKEND_NAME,dim,t);
    } else {
        mpfr_set_default_prec(REF_BITS);
        long n = dim, N = n*n;
        mpfr_t *Am=malloc(N*sizeof(mpfr_t)), *Bm=malloc(N*sizeof(mpfr_t)), *Ct=malloc(N*sizeof(mpfr_t));
        mpfr_t acc,prod,num,den,rel,maxrel,gen;
        mpfr_inits2(REF_BITS,acc,prod,num,den,rel,maxrel,gen,(mpfr_ptr)0);
        for(long i=0;i<n;i++) for(long j=0;j<n;j++){
            mpfr_set_d(gen,2.0+((i*131+j*17)%97)*0.01,MPFR_RNDN); mpfr_sqrt(gen,gen,MPFR_RNDN);
            set_dmatrix_ij(A,i,j,mpfr_get_d(gen,MPFR_RNDN));
            mpfr_set_d(gen,3.0+((i*13+j*131)%89)*0.01,MPFR_RNDN); mpfr_sqrt(gen,gen,MPFR_RNDN);
            set_dmatrix_ij(B,i,j,mpfr_get_d(gen,MPFR_RNDN));
        }
        for(long i=0;i<n;i++) for(long j=0;j<n;j++){
            mpfr_init2(Am[i*n+j],REF_BITS); mpfr_set_d(Am[i*n+j],get_dmatrix_ij(A,i,j),MPFR_RNDN);
            mpfr_init2(Bm[i*n+j],REF_BITS); mpfr_set_d(Bm[i*n+j],get_dmatrix_ij(B,i,j),MPFR_RNDN);
            mpfr_init2(Ct[i*n+j],REF_BITS);
        }
        for(long i=0;i<n;i++) for(long j=0;j<n;j++){
            mpfr_set_zero(acc,1);
            for(long k=0;k<n;k++){ mpfr_mul(prod,Am[i*n+k],Bm[k*n+j],MPFR_RNDN); mpfr_add(acc,acc,prod,MPFR_RNDN); }
            mpfr_set(Ct[i*n+j],acc,MPFR_RNDN);
        }
        const char *names[4]={"triple","block","strassen","winograd"};
        for(int a=0;a<4;a++){
            switch(a){ case 0: mul_dmatrix(Cm,A,B); break; case 1: mul_dmatrix_block(Cm,A,B,md); break;
                       case 2: mul_dmatrix_strassen(Cm,A,B,md); break; case 3: mul_dmatrix_winograd_even(Cm,A,B,md); break; }
            mpfr_set_zero(maxrel,1);
            for(long i=0;i<n;i++) for(long j=0;j<n;j++){
                mpfr_set_d(acc,get_dmatrix_ij(Cm,i,j),MPFR_RNDN);
                mpfr_sub(num,acc,Ct[i*n+j],MPFR_RNDN); mpfr_abs(num,num,MPFR_RNDN);
                mpfr_abs(den,Ct[i*n+j],MPFR_RNDN);
                if(mpfr_zero_p(den)) continue;
                mpfr_div(rel,num,den,MPFR_RNDN);
                if(mpfr_cmp(rel,maxrel)>0) mpfr_set(maxrel,rel,MPFR_RNDN);
            }
            printf("ACC,%s,%s,%s,%ld,%.6e\n", PREC_NAME,BACKEND_NAME,names[a],dim,mpfr_get_d(maxrel,MPFR_RNDN));
        }
        for(long i=0;i<N;i++){ mpfr_clear(Am[i]); mpfr_clear(Bm[i]); mpfr_clear(Ct[i]); }
        free(Am); free(Bm); free(Ct);
    }
    free_dmatrix(A); free_dmatrix(B); free_dmatrix(Cm);
    return 0;
}
