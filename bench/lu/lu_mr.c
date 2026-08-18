/* lu_mr.c - multicomponent real (dd,td,qd,ds,ts,qs) LU decomposition benchmark.
 * Verifies LUdecompPM + SolveLSPM against a known solution and times them.
 *
 * Build with: -DP=<dd|td|qd|ds|ts|qs> -DPU=<DD|TD|QD|DS|TS|QS>
 *             -DMT=<DDMatrix..> -DVT=<DDVector..>
 *             -DBASE=<double|float> -DPSIZE=<DDSIZE..>
 *             -DPREC_NAME=\"dd\" -DBACKEND_NAME=\"serial\"
 *             [-DBNC_USE_NEW_FMA]
 * Link the matching src/<prec>lu.c (or ts/qslinear.c) object BEFORE the library
 * so both sides of the FMA comparison run freshly compiled kernels.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
#define UFN(post) C(PU,post)
#define MATVEC C(C(C(C(mul_,P),matrix_),P),vec)

#ifndef PREC_NAME
#define PREC_NAME "mr"
#endif
#ifndef BACKEND_NAME
#define BACKEND_NAME "scalar"
#endif
#ifdef BNC_USE_NEW_FMA
#define FMA_NAME "fma"
#else
#define FMA_NAME "nofma"
#endif

static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec*1e-9; }

#define TIME_OP(tvar, CALL) do { long it=1; double e; for(;;){ \
    double s=now_s(); for(long _k=0;_k<it;_k++){ CALL; } e=now_s()-s; \
    if(e>=0.20 || it>=(1L<<22)) break; it*=2; } tvar=e/(double)it; } while(0)

static void mkval(BASE *v, double x){ v[0]=(BASE)x; for(int k=1;k<PSIZE;k++) v[k]=(BASE)0; }

/* raw component copy: dst := src (same shape) */
static void copy_mat(MT dst, MT src){
    long n = src->real_row_dim * src->real_col_dim;
    for(int c = 0; c < PSIZE; c++)
        memcpy(dst->element[c], src->element[c], (size_t)n * sizeof(BASE));
}

int main(int argc, char**argv){
    long dim = (argc>=2)? atol(argv[1]) : 512;
    double t_copy=0, t_lu=0, t_solve=0;
    BASE tmp[PSIZE];
    long i, j;

    MT A  = FN(init_,matrix)(dim,dim);
    MT LU = FN(init_,matrix)(dim,dim);
    VT xt = FN(init_,vector)(dim), b = FN(init_,vector)(dim), y = FN(init_,vector)(dim);
    long *ch = (long *)calloc((size_t)dim, sizeof(long));

    /* diagonally dominant test matrix; entries exactly representable in float */
    for(i=0;i<dim;i++){
        for(j=0;j<dim;j++){
            mkval(tmp, (i==j) ? (double)dim : ((i*131+j*17)%97)*0.01 + 0.1);
            FN(set_,matrix_ij)(A,i,j,tmp);
        }
        mkval(tmp, (double)((i%16)+1));   /* true solution */
        FN(set_,vector_i)(xt,i,tmp);
    }
    MATVEC(b, A, xt);                     /* rhs in working precision */

    /* verify once */
    copy_mat(LU, A);
    if(UFN(LUdecompPM)(LU, ch) != 0){ fprintf(stderr, "decomp failed\n"); return 1; }
    if(C(Solve,C(PU,LSPM))(y, LU, b, ch) != 0){ fprintf(stderr, "solve failed\n"); return 1; }

    double maxrelerr = 0.0;
    for(i=0;i<dim;i++){
        double xv = 0.0, tv = (double)((i%16)+1);
        memcpy(tmp, FN(get_,vector_i)(y, i), sizeof(BASE) * PSIZE);
        for(int c = PSIZE-1; c >= 0; c--) xv += (double)tmp[c];
        double e = fabs(xv - tv) / fabs(tv);
        if(e > maxrelerr) maxrelerr = e;
    }

    /* timings */
    TIME_OP(t_copy, copy_mat(LU, A));
    TIME_OP(t_lu, { copy_mat(LU, A); UFN(LUdecompPM)(LU, ch); });
    t_lu -= t_copy; if(t_lu < 0) t_lu = 0;
    TIME_OP(t_solve, C(Solve,C(PU,LSPM))(y, LU, b, ch));

    printf("RESULT,%s,%s,%s,%ld,%.9e,%.9e,%.6e\n",
           PREC_NAME, BACKEND_NAME, FMA_NAME, dim, t_lu, t_solve, maxrelerr);

    FN(free_,matrix)(A); FN(free_,matrix)(LU);
    FN(free_,vector)(xt); FN(free_,vector)(b); FN(free_,vector)(y);
    free(ch);
    return 0;
}
