/* mm_multi.c - multicomponent (dd,td,qd,ds,ts,qs) matmul algorithm benchmark.
 *
 * Measures 4 algorithms (triple loop / blocking / Strassen / Winograd) for one
 * precision, one SIMD backend (chosen at link time via the scalar/neon/sve2
 * library).  Two modes:
 *   <bin> <dim> time   -> TIME,<prec>,<backend>,<algo>,<dim>,<sec>
 *   <bin> <dim> acc    -> ACC,<prec>,<backend>,<algo>,<dim>,<max_rel_err>
 *
 * Accuracy = per-element maximum relative error of C=A*B against an MPFR
 * reference (REF_BITS bits) built from THIS precision's actually-stored inputs.
 *
 * Build with: -DP=<dd|td|qd|ds|ts|qs> -DMT=<DDMatrix..> -DBASE=<double|float>
 *             -DPSIZE=<DDSIZE..> -DPREC_NAME=\"dd\" -DBACKEND_NAME=\"serial\"
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpfr.h>
#include "ddlinear.h"
#include "tdlinear.h"
#include "qdlinear.h"
#include "dslinear.h"
#include "tslinear.h"
#include "qslinear.h"
#include "matmul_strassen.h"

#define _C(a,b) a##b
#define C(a,b) _C(a,b)
#define FN(pre,post) C(C(pre,P),post)

#ifndef PREC_NAME
#define PREC_NAME "multi"
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

/* time one call, auto-repeating until >=0.20s (or single iter for slow ops) */
#define TIME_OP(tvar, CALL) do { long it=1; double e; for(;;){ \
    double s=now_s(); for(long _k=0;_k<it;_k++){ CALL; } e=now_s()-s; \
    if(e>=0.20 || it>=(1L<<22)) break; it*=2; } tvar=e/(double)it; } while(0)

static void mkval(BASE *v, double x){ v[0]=(BASE)x; for(int k=1;k<PSIZE;k++) v[k]=(BASE)0; }

/* decompose a high-precision mpfr value into PSIZE normalized BASE components
 * (nearest-and-remainder), so every component of the type carries information. */
static void decompose(mpfr_t src, BASE *c){
    mpfr_t r; mpfr_init2(r, mpfr_get_prec(src)); mpfr_set(r, src, MPFR_RNDN);
    for(int k=0;k<PSIZE;k++){ c[k]=(BASE)mpfr_get_d(r,MPFR_RNDN); mpfr_sub_d(r,r,(double)c[k],MPFR_RNDN); }
    mpfr_clear(r);
}

/* accumulate the (unevaluated-sum) element (i,j) of a matrix into mpfr acc */
static void elem_to_mpfr(mpfr_t acc, MT M, long i, long j){
    long idx = M->real_col_dim * i + j;
    mpfr_set_d(acc, (double)M->element[0][idx], MPFR_RNDN);
    for(int k=1;k<PSIZE;k++)
        mpfr_add_d(acc, acc, (double)M->element[k][idx], MPFR_RNDN);
}

int main(int argc, char**argv){
    long dim = (argc>=2)? atol(argv[1]) : 256;
    const char *mode = (argc>=3)? argv[2] : "time";
    long md = MIN_DIM;
    BASE tmp[PSIZE];

    MT A = FN(init_,matrix)(dim,dim), B = FN(init_,matrix)(dim,dim), Cm = FN(init_,matrix)(dim,dim);
    for(long i=0;i<dim;i++)
        for(long j=0;j<dim;j++){
            mkval(tmp, ((i*131+j*17)%97)*0.01 + 0.1); FN(set_,matrix_ij)(A,i,j,tmp);
            mkval(tmp, ((i*13+j*131)%89)*0.01 + 0.1); FN(set_,matrix_ij)(B,i,j,tmp);
        }

    if(strcmp(mode,"time")==0){
        double t;
        TIME_OP(t, FN(mul_,matrix)(Cm,A,B));
        printf("TIME,%s,%s,triple,%ld,%.9e\n", PREC_NAME, BACKEND_NAME, dim, t);
        TIME_OP(t, FN(mul_,matrix_block)(Cm,A,B,md));
        printf("TIME,%s,%s,block,%ld,%.9e\n", PREC_NAME, BACKEND_NAME, dim, t);
        TIME_OP(t, FN(mul_,matrix_strassen)(Cm,A,B,md));
        printf("TIME,%s,%s,strassen,%ld,%.9e\n", PREC_NAME, BACKEND_NAME, dim, t);
        TIME_OP(t, FN(mul_,matrix_winograd_even)(Cm,A,B,md));
        printf("TIME,%s,%s,winograd,%ld,%.9e\n", PREC_NAME, BACKEND_NAME, dim, t);
    } else { /* acc */
        mpfr_set_default_prec(REF_BITS);
        long n = dim, N = n*n;
        mpfr_t *Am = malloc(N*sizeof(mpfr_t)), *Bm = malloc(N*sizeof(mpfr_t)), *Ct = malloc(N*sizeof(mpfr_t));
        mpfr_t acc, prod, num, den, rel, maxrel, gen;
        mpfr_inits2(REF_BITS, acc, prod, num, den, rel, maxrel, gen, (mpfr_ptr)0);
        /* refill A,B with full-precision (rich) irrational values for this type */
        for(long i=0;i<n;i++) for(long j=0;j<n;j++){
            mpfr_set_d(gen, 2.0 + ((i*131+j*17)%97)*0.01, MPFR_RNDN); mpfr_sqrt(gen,gen,MPFR_RNDN);
            decompose(gen, tmp); FN(set_,matrix_ij)(A,i,j,tmp);
            mpfr_set_d(gen, 3.0 + ((i*13+j*131)%89)*0.01, MPFR_RNDN); mpfr_sqrt(gen,gen,MPFR_RNDN);
            decompose(gen, tmp); FN(set_,matrix_ij)(B,i,j,tmp);
        }
        for(long i=0;i<n;i++) for(long j=0;j<n;j++){
            mpfr_init2(Am[i*n+j],REF_BITS); elem_to_mpfr(Am[i*n+j],A,i,j);
            mpfr_init2(Bm[i*n+j],REF_BITS); elem_to_mpfr(Bm[i*n+j],B,i,j);
            mpfr_init2(Ct[i*n+j],REF_BITS);
        }
        /* reference product Ct = Am * Bm */
        for(long i=0;i<n;i++) for(long j=0;j<n;j++){
            mpfr_set_zero(acc,1);
            for(long k=0;k<n;k++){ mpfr_mul(prod,Am[i*n+k],Bm[k*n+j],MPFR_RNDN); mpfr_add(acc,acc,prod,MPFR_RNDN); }
            mpfr_set(Ct[i*n+j],acc,MPFR_RNDN);
        }
        const char *names[4]={"triple","block","strassen","winograd"};
        for(int a=0;a<4;a++){
            switch(a){
              case 0: FN(mul_,matrix)(Cm,A,B); break;
              case 1: FN(mul_,matrix_block)(Cm,A,B,md); break;
              case 2: FN(mul_,matrix_strassen)(Cm,A,B,md); break;
              case 3: FN(mul_,matrix_winograd_even)(Cm,A,B,md); break;
            }
            mpfr_set_zero(maxrel,1);
            for(long i=0;i<n;i++) for(long j=0;j<n;j++){
                elem_to_mpfr(acc,Cm,i,j);            /* computed value */
                mpfr_sub(num,acc,Ct[i*n+j],MPFR_RNDN); mpfr_abs(num,num,MPFR_RNDN);
                mpfr_abs(den,Ct[i*n+j],MPFR_RNDN);
                if(mpfr_zero_p(den)) continue;
                mpfr_div(rel,num,den,MPFR_RNDN);
                if(mpfr_cmp(rel,maxrel)>0) mpfr_set(maxrel,rel,MPFR_RNDN);
            }
            printf("ACC,%s,%s,%s,%ld,%.6e\n", PREC_NAME, BACKEND_NAME, names[a], dim, mpfr_get_d(maxrel,MPFR_RNDN));
        }
        for(long i=0;i<N;i++){ mpfr_clear(Am[i]); mpfr_clear(Bm[i]); mpfr_clear(Ct[i]); }
        free(Am); free(Bm); free(Ct);
    }
    FN(free_,matrix)(A); FN(free_,matrix)(B); FN(free_,matrix)(Cm);
    return 0;
}
