/* dense_ncf.c - native complex-float (cf) dense benchmark, hand-SIMD.
 * axpy/matvec/matmul on 'float _Complex' via the cf_simd.h primitives
 * (scalar / NEON / SVE2 / AVX2 / AVX-512, selected by -DBNC_ENABLE_*).
 * `--check` validates the active SIMD path against the scalar reference.
 * Build: -DPREC_NAME='"cf"' -DBACKEND_NAME='"serial"' [-mcpu=... -DBNC_ENABLE_*].
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <complex.h>
#include "cf_simd.h"

#ifndef PREC_NAME
#define PREC_NAME "cf"
#endif
#ifndef BACKEND_NAME
#define BACKEND_NAME "scalar"
#endif

static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec*1e-9; }

#define TIME_OP(tvar, CALL) do { long it=1; double e; for(;;){ \
    double s=now_s(); for(long _k=0;_k<it;_k++){ CALL; } e=now_s()-s; \
    if(e>=0.20 || it>=(1L<<22)) break; it*=2; } tvar=e/(double)it; } while(0)

/* y = A*x ; A row-major dim x dim (each output = complex dot of a row with x) */
static void cf_matvec(cf_t *y, const cf_t *A, const cf_t *x, long dim){
    for(long i=0;i<dim;i++){ float re,im; cf_dot(A+i*dim, x, dim, &re, &im);
        y[i]=re+im*(cf_t)I; }
}
/* C = A*B ; row-major (i-k-j: C[i,:] += A[i,k]*B[k,:] via complex axpy) */
static void cf_matmul(cf_t *C, const cf_t *A, const cf_t *B, long dim){
    for(long i=0;i<dim;i++){
        cf_t *Ci=C+i*dim;
        for(long j=0;j<dim;j++) Ci[j]=0.0f;
        for(long k=0;k<dim;k++){
            cf_t a=A[i*dim+k];
            cf_axpy(Ci, Ci, crealf(a), cimagf(a), B+k*dim, dim);   /* Ci += a*B[k,:] */
        }
    }
}

static int self_check(long dim){
    long L=dim*dim; int bad=0;
    cf_t *x=malloc(sizeof(cf_t)*L), *y=malloc(sizeof(cf_t)*L);
    cf_t *z=malloc(sizeof(cf_t)*L), *zr=malloc(sizeof(cf_t)*L);
    for(long i=0;i<L;i++){ x[i]=(float)((i%97)*0.01+0.1)+(float)((i%53)*0.02+0.05)*I;
                           y[i]=(float)((i%89)*0.013+0.2)+(float)((i%47)*0.011+0.03)*I; }
    /* axpy */
    cf_axpy(z, y, 1.7f, -0.9f, x, L);
    cf_axpy_scalar(zr, y, 1.7f, -0.9f, x, L);
    double em=0; for(long i=0;i<L;i++){ double d=cabsf(z[i]-zr[i]); if(d>em)em=d; }
    printf("  check axpy   maxabs=%.3e\n", em); if(em>1e-3) bad=1;
    /* dot */
    float re,im,rr,ir; cf_dot(x,y,L,&re,&im); cf_dot_scalar(x,y,L,&rr,&ir);
    double ed=fabs(re-rr)+fabs(im-ir);
    double rel=ed/(fabs(rr)+fabs(ir)+1e-30);
    printf("  check dot    abs=%.3e rel=%.3e\n", ed, rel); if(rel>1e-4) bad=1;
    free(x);free(y);free(z);free(zr);
    return bad;
}

int main(int argc, char**argv){
    if(argc>=2 && strcmp(argv[1],"--check")==0){
        printf("cf backend compiled = %s\n", CF_BACKEND);
        int bad = self_check(257) | self_check(1024);
        printf(bad?"CHECK FAILED\n":"CHECK OK\n");
        return bad;
    }
    long dim = (argc>=2)? atol(argv[1]) : 512;
    long L = dim*dim;
    double t_ax=0, t_mv=0, t_mm=0;

    cf_t *xa=malloc(sizeof(cf_t)*L), *ya=malloc(sizeof(cf_t)*L), *za=malloc(sizeof(cf_t)*L);
    for(long i=0;i<L;i++){
        xa[i]=(float)(((i*131+7)%97)*0.01+0.1)+(float)(((i*5+1)%53)*0.01+0.05)*I;
        ya[i]=(float)(((i*17+3)%89)*0.01+0.1)+(float)(((i*7+2)%47)*0.01+0.05)*I;
    }
    cf_axpy(za,ya,2.0f,0.5f,xa,L);                        /* warmup */
    TIME_OP(t_ax, cf_axpy(za,ya,2.0f,0.5f,xa,L));
    free(xa); free(ya); free(za);

    cf_t *A=malloc(sizeof(cf_t)*L), *B=malloc(sizeof(cf_t)*L), *Cm=malloc(sizeof(cf_t)*L);
    cf_t *x=malloc(sizeof(cf_t)*dim), *y=malloc(sizeof(cf_t)*dim);
    for(long i=0;i<dim;i++){
        for(long j=0;j<dim;j++){
            A[i*dim+j]=(float)(((i*131+j*17)%97)*0.01+0.1)+(float)(((i*3+j*5)%41)*0.01+0.05)*I;
            B[i*dim+j]=(float)(((i*13+j*131)%89)*0.01+0.1)+(float)(((i*7+j*11)%37)*0.01+0.05)*I;
        }
        x[i]=(float)(((i*29+5)%83)*0.01+0.1)+(float)(((i*2+1)%31)*0.01+0.05)*I;
    }
    cf_matvec(y,A,x,dim);                                 /* warmup */
    TIME_OP(t_mv, cf_matvec(y,A,x,dim));
    TIME_OP(t_mm, cf_matmul(Cm,A,B,dim));

    printf("RESULT,%s,%s,%ld,%.9e,%.9e,%.9e\n", PREC_NAME, BACKEND_NAME, dim, t_ax, t_mv, t_mm);
    free(A); free(B); free(Cm); free(x); free(y);
    return 0;
}
