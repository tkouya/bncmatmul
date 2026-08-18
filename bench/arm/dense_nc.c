/* dense_nc.c - native complex (cd) dense benchmark. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include "cdlinear.h"

#ifndef PREC_NAME
#define PREC_NAME "cd"
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

    CDVector xa = init_cdvector(L), ya = init_cdvector(L), za = init_cdvector(L);
    for(long i=0;i<L;i++){
        set_cdvector_i(xa,i, (((i*131+7)%97)*0.01+0.1) + (((i*5+1)%53)*0.01+0.05)*I);
        set_cdvector_i(ya,i, (((i*17+3)%89)*0.01+0.1) + (((i*7+2)%47)*0.01+0.05)*I);
    }
    double _Complex alpha = 2.0 + 0.5*I;
    add_cmul_cdvector(za,ya,alpha,xa);                    /* warmup */
    TIME_OP(t_ax, add_cmul_cdvector(za,ya,alpha,xa));
    free_cdvector(xa); free_cdvector(ya); free_cdvector(za);

    CDMatrix A = init_cdmatrix(dim,dim), B = init_cdmatrix(dim,dim), Cm = init_cdmatrix(dim,dim);
    CDVector x = init_cdvector(dim), y = init_cdvector(dim);
    for(long i=0;i<dim;i++){
        for(long j=0;j<dim;j++){
            set_cdmatrix_ij(A,i,j,(((i*131+j*17)%97)*0.01+0.1)+(((i*3+j*5)%41)*0.01+0.05)*I);
            set_cdmatrix_ij(B,i,j,(((i*13+j*131)%89)*0.01+0.1)+(((i*7+j*11)%37)*0.01+0.05)*I);
        }
        set_cdvector_i(x,i,(((i*29+5)%83)*0.01+0.1)+(((i*2+1)%31)*0.01+0.05)*I);
    }
    mul_cdmatrix_cdvec(y,A,x);                            /* warmup */
    TIME_OP(t_mv, mul_cdmatrix_cdvec(y,A,x));
    TIME_OP(t_mm, mul_cdmatrix(Cm,A,B));

    printf("RESULT,%s,%s,%ld,%.9e,%.9e,%.9e\n", PREC_NAME, BACKEND_NAME, dim, t_ax, t_mv, t_mm);
    free_cdmatrix(A); free_cdmatrix(B); free_cdmatrix(Cm);
    free_cdvector(x); free_cdvector(y);
    return 0;
}
