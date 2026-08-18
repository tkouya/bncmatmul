/* dense_mr.c - multicomponent real (dd,td,qd,ds,ts,qs) dense benchmark.
 * Build with: -DP=<dd|td|qd|ds|ts|qs> -DMT=<DDMatrix..> -DVT=<DDVector..>
 *             -DBASE=<double|float> -DPSIZE=<DDSIZE..> -DPREC_NAME=\"dd\" -DBACKEND_NAME=\"serial\"
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "ddlinear.h"
#include "tdlinear.h"
#include "qdlinear.h"
#include "dslinear.h"
#include "tslinear.h"
#include "qslinear.h"

#define _C(a,b) a##b
#define C(a,b) _C(a,b)
#define FN(pre,post) C(C(pre,P),post)
#define MATVEC C(C(C(C(mul_,P),matrix_),P),vec)

#ifndef PREC_NAME
#define PREC_NAME "mr"
#endif
#ifndef BACKEND_NAME
#define BACKEND_NAME "scalar"
#endif

static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec*1e-9; }

#define TIME_OP(tvar, CALL) do { long it=1; double e; for(;;){ \
    double s=now_s(); for(long _k=0;_k<it;_k++){ CALL; } e=now_s()-s; \
    if(e>=0.20 || it>=(1L<<22)) break; it*=2; } tvar=e/(double)it; } while(0)

static void mkval(BASE *v, double x){ v[0]=(BASE)x; for(int k=1;k<PSIZE;k++) v[k]=(BASE)0; }

int main(int argc, char**argv){
    long dim = (argc>=2)? atol(argv[1]) : 512;
    long L = dim*dim;
    double t_ax=0, t_mv=0, t_mm=0;
    BASE tmp[PSIZE], alpha[PSIZE];

    /* ---- axpy ---- */
    VT xa = FN(init_,vector)(L), ya = FN(init_,vector)(L), za = FN(init_,vector)(L);
    for(long i=0;i<L;i++){
        mkval(tmp, ((i*131+7)%97)*0.01 + 0.1); FN(set_,vector_i)(xa,i,tmp);
        mkval(tmp, ((i*17+3)%89)*0.01 + 0.1);  FN(set_,vector_i)(ya,i,tmp);
    }
    mkval(alpha, 2.0);
    FN(add_cmul_,vector)(za,ya,alpha,xa);                 /* warmup */
    TIME_OP(t_ax, FN(add_cmul_,vector)(za,ya,alpha,xa));
    FN(free_,vector)(xa); FN(free_,vector)(ya); FN(free_,vector)(za);

    /* ---- matvec / matmul ---- */
    MT A = FN(init_,matrix)(dim,dim), B = FN(init_,matrix)(dim,dim), Cm = FN(init_,matrix)(dim,dim);
    VT x = FN(init_,vector)(dim), y = FN(init_,vector)(dim);
    for(long i=0;i<dim;i++){
        for(long j=0;j<dim;j++){
            mkval(tmp, ((i*131+j*17)%97)*0.01 + 0.1); FN(set_,matrix_ij)(A,i,j,tmp);
            mkval(tmp, ((i*13+j*131)%89)*0.01 + 0.1); FN(set_,matrix_ij)(B,i,j,tmp);
        }
        mkval(tmp, ((i*29+5)%83)*0.01 + 0.1); FN(set_,vector_i)(x,i,tmp);
    }
    MATVEC(y,A,x);                                        /* warmup */
    TIME_OP(t_mv, MATVEC(y,A,x));
    TIME_OP(t_mm, FN(mul_,matrix)(Cm,A,B));

    printf("RESULT,%s,%s,%ld,%.9e,%.9e,%.9e\n", PREC_NAME, BACKEND_NAME, dim, t_ax, t_mv, t_mm);
    FN(free_,matrix)(A); FN(free_,matrix)(B); FN(free_,matrix)(Cm);
    FN(free_,vector)(x); FN(free_,vector)(y);
    return 0;
}
