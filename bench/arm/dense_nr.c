/* dense_nr.c - native real (double / float) dense benchmark.
 * axpy (add_cmul: c = a + alpha*b), matvec (y=A*x), matmul (C=A*B).
 * Build with: -DP=<d|f> -DMT=<DMatrix|FMatrix> -DVT=<DVector|FVector>
 *             -DBASE=<double|float> -DPREC_NAME=\"double\" -DBACKEND_NAME=\"serial\"
 * Output: RESULT,<prec>,<backend>,<dim>,<axpy_s>,<matvec_s>,<matmul_s>
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "dlinear.h"
#include "flinear.h"

#define _C(a,b) a##b
#define C(a,b) _C(a,b)
#define FN(pre,post) C(C(pre,P),post)
#ifndef MVFN
#define MVFN MATVEC_default
#endif

#ifndef PREC_NAME
#define PREC_NAME "nr"
#endif
#ifndef BACKEND_NAME
#define BACKEND_NAME "scalar"
#endif

static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec*1e-9; }

#define TIME_OP(tvar, CALL) do { long it=1; double e; for(;;){ \
    double s=now_s(); for(long _k=0;_k<it;_k++){ CALL; } e=now_s()-s; \
    if(e>=0.20 || it>=(1L<<22)) break; it*=2; } tvar=e/(double)it; } while(0)

int main(int argc, char**argv){
    long dim = (argc>=2)? atol(argv[1]) : 512;
    long L = dim*dim;            /* axpy vector length = N*N */
    double t_ax=0, t_mv=0, t_mm=0;

    /* ---- axpy on length-L vectors ---- */
    VT xa = FN(init_,vector)(L), ya = FN(init_,vector)(L), za = FN(init_,vector)(L);
    for(long i=0;i<L;i++){
        BASE v = (BASE)(((i*131+7)%97)*0.01 + 0.1);
        FN(set_,vector_i)(xa,i,v);
        FN(set_,vector_i)(ya,i,(BASE)(((i*17+3)%89)*0.01 + 0.1));
    }
    BASE alpha = (BASE)2.0;
    FN(add_cmul_,vector)(za,ya,alpha,xa);                 /* warmup */
    TIME_OP(t_ax, FN(add_cmul_,vector)(za,ya,alpha,xa));
    FN(free_,vector)(xa); FN(free_,vector)(ya); FN(free_,vector)(za);

    /* ---- matvec / matmul on dim x dim ---- */
    MT A = FN(init_,matrix)(dim,dim), B = FN(init_,matrix)(dim,dim), Cm = FN(init_,matrix)(dim,dim);
    VT x = FN(init_,vector)(dim), y = FN(init_,vector)(dim);
    for(long i=0;i<dim;i++){
        for(long j=0;j<dim;j++){
            FN(set_,matrix_ij)(A,i,j,(BASE)(((i*131+j*17)%97)*0.01 + 0.1));
            FN(set_,matrix_ij)(B,i,j,(BASE)(((i*13+j*131)%89)*0.01 + 0.1));
        }
        FN(set_,vector_i)(x,i,(BASE)(((i*29+5)%83)*0.01 + 0.1));
    }
    MVFN(y,A,x);                                        /* warmup */
    TIME_OP(t_mv, MVFN(y,A,x));
    TIME_OP(t_mm, FN(mul_,matrix)(Cm,A,B));

    printf("RESULT,%s,%s,%ld,%.9e,%.9e,%.9e\n", PREC_NAME, BACKEND_NAME, dim, t_ax, t_mv, t_mm);
    FN(free_,matrix)(A); FN(free_,matrix)(B); FN(free_,matrix)(Cm);
    FN(free_,vector)(x); FN(free_,vector)(y);
    return 0;
}
