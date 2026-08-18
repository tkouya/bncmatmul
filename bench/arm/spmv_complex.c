/* spmv_complex.c - complex sparse matrix-vector product benchmark.
 * Builds a complex sparse matrix (re = A, im = 0.3*A) from a real .mtx file
 * and times y := A_complex * x_complex.
 * Build: -DP=<cd|cdd|ctd|cqd> -DVT=<CDVector..> [-DNATIVE_BASE]
 *        -DSETVEC=<set_cdvector_i|...> -DPSIZE=<1|2|3|4>
 *        -DPREC_NAME=\"cd\" -DBACKEND_NAME=\"serial\"
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include "bncsparse.h"
#include "bncmm.h"
#include "cdlinear.h"
#include "cddlinear.h"
#include "ctdlinear.h"
#include "cqdlinear.h"

#define _C(a,b) a##b
#define C(a,b) _C(a,b)
#define FN(pre,post) C(C(pre,P),post)
#define SPMV C(C(C(C(mul_,P),rsmatrix_),P),vec)

#ifndef PREC_NAME
#define PREC_NAME "cx"
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
    if(argc<2){ fprintf(stderr,"usage: %s <matrix.mtx> [reps]\n",argv[0]); return 1; }
    DRSMatrix A = init_drsmatrix_readMMcoordinate(argv[1]);
    if(!A){ fprintf(stderr,"failed to load %s\n",argv[1]); return 1; }
    long nrow = A->row_dim, nnz = A->nzero_total_num;

    long *dup = (long*)malloc(sizeof(long)*nrow);
    memcpy(dup, A->nzero_col_dim, sizeof(long)*nrow);
    CT Ac = FN(init_,rsmatrix)(nrow, dup, nnz);
    free(dup);

    /* fill sparsity pattern + values: re = A, im = 0.3*A */
    for(long i=0;i<nrow;i++)
        for(long j=0;j<A->nzero_col_dim[i];j++){
            Ac->re->nzero_index[i][j] = A->nzero_index[i][j];
            Ac->im->nzero_index[i][j] = A->nzero_index[i][j];
        }
    for(long k=0;k<A->real_nzero_total_num;k++){
        double v = A->element[k];
#ifdef NATIVE_BASE
        Ac->re->element[k] = v;
        Ac->im->element[k] = 0.3*v;
#else
        for(int c=0;c<PSIZE;c++){ Ac->re->element[c][k]=0.0; Ac->im->element[c][k]=0.0; }
        Ac->re->element[0][k] = v;
        Ac->im->element[0][k] = 0.3*v;
#endif
    }

    VT x = FN(init_,vector)(nrow), y = FN(init_,vector)(nrow);
    for(long i=0;i<nrow;i++){
        double v = sin((double)i*0.001)+1.5;
        SETVEC(x,i, v + 0.5*v*I);
    }

    SPMV(y,Ac,x);                                  /* warmup */
    double t_spmv=0; TIME_OP(t_spmv, SPMV(y,Ac,x));

    double mflop = 8.0*(double)nnz / t_spmv / 1e6;  /* complex spmv ~ 8 flops/nz */
    const char *mname = strrchr(argv[1],'/'); mname = mname? mname+1: argv[1];
    printf("RESULT_SPMV,%s,%s,%s,%ld,%.9e,%.3f\n", PREC_NAME, BACKEND_NAME, mname, nnz, t_spmv, mflop);

    FN(free_,rsmatrix)(Ac); FN(free_,vector)(x); FN(free_,vector)(y); free_drsmatrix(A);
    return 0;
}
