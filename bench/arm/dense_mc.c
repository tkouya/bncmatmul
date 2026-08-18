/* dense_mc.c - multicomponent complex (cdd,ctd,cqd,cds,cts,cqs) dense benchmark.
 * Build with: -DP=<cdd..> -DMT=<CDDMatrix..> -DVT=<CDDVector..>
 *             -DRCSET=<rcdd_set_d_d..> -DPREC_NAME=\"cdd\" -DBACKEND_NAME=\"serial\"
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#ifdef SINGLE_BASED
#include "cdslinear.h"
#include "ctslinear.h"
#include "cqslinear.h"
#else
#include "cddlinear.h"
#include "ctdlinear.h"
#include "cqdlinear.h"
#endif

#define _C(a,b) a##b
#define C(a,b) _C(a,b)
#define FN(pre,post) C(C(pre,P),post)
#define MATVEC C(C(C(C(mul_,P),matrix_),P),vec)
#define ELEMT  C(P,float)

#ifndef PREC_NAME
#define PREC_NAME "mc"
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
    long L = dim*dim;
    double t_ax=0, t_mv=0, t_mm=0;

    VT xa = FN(init_,vector)(L), ya = FN(init_,vector)(L), za = FN(init_,vector)(L);
    for(long i=0;i<L;i++){
        FN(set_,vector_i_cd)(xa,i, (((i*131+7)%97)*0.01+0.1) + (((i*5+1)%53)*0.01+0.05)*I);
        FN(set_,vector_i_cd)(ya,i, (((i*17+3)%89)*0.01+0.1) + (((i*7+2)%47)*0.01+0.05)*I);
    }
    ELEMT alpha; RCSET(&alpha, 2.0, 0.5);
    FN(add_cmul_,vector)(za,ya,&alpha,xa);                /* warmup */
    TIME_OP(t_ax, FN(add_cmul_,vector)(za,ya,&alpha,xa));
    FN(free_,vector)(xa); FN(free_,vector)(ya); FN(free_,vector)(za);

    MT A = FN(init_,matrix)(dim,dim), B = FN(init_,matrix)(dim,dim), Cm = FN(init_,matrix)(dim,dim);
    VT x = FN(init_,vector)(dim), y = FN(init_,vector)(dim);
    for(long i=0;i<dim;i++){
        for(long j=0;j<dim;j++){
            FN(set_,matrix_ij_cd)(A,i,j,(((i*131+j*17)%97)*0.01+0.1)+(((i*3+j*5)%41)*0.01+0.05)*I);
            FN(set_,matrix_ij_cd)(B,i,j,(((i*13+j*131)%89)*0.01+0.1)+(((i*7+j*11)%37)*0.01+0.05)*I);
        }
        FN(set_,vector_i_cd)(x,i,(((i*29+5)%83)*0.01+0.1)+(((i*2+1)%31)*0.01+0.05)*I);
    }
    MATVEC(y,A,x);                                        /* warmup */
    TIME_OP(t_mv, MATVEC(y,A,x));
    TIME_OP(t_mm, FN(mul_,matrix)(Cm,A,B));

    printf("RESULT,%s,%s,%ld,%.9e,%.9e,%.9e\n", PREC_NAME, BACKEND_NAME, dim, t_ax, t_mv, t_mm);
    FN(free_,matrix)(A); FN(free_,matrix)(B); FN(free_,matrix)(Cm);
    FN(free_,vector)(x); FN(free_,vector)(y);
    return 0;
}
