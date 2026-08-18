/* spmv_cf.c - native complex-float (cf) sparse matrix-vector product benchmark.
 * Self-contained: parses a MatrixMarket "coordinate real general" .mtx directly
 * into a clean float-complex CSR (re = A, im = 0.3*A) and times y := A*x.
 * Matches the complex-SpMV convention used by the other complex benches.
 * Build: -DPREC_NAME='"cf"' -DBACKEND_NAME='"serial"' [-mcpu=...].
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

/* y = A*x in CSR. Per row: gather x[col[k]] into a contiguous temp buffer
 * (no SIMD gather on NEON), then the complex multiply-accumulate is the
 * vectorized cf_dot. tmp must hold at least max-row-nnz complex values. */
static void cf_spmv(cf_t *y, const long *row_ptr, const long *col_idx,
                    const cf_t *val, const cf_t *x, long n, cf_t *tmp){
    for(long i=0;i<n;i++){
        long b=row_ptr[i], e=row_ptr[i+1], len=e-b;
        for(long k=0;k<len;k++) tmp[k]=x[col_idx[b+k]];      /* gather */
        float re,im; cf_dot(val+b, tmp, len, &re, &im);      /* SIMD MAC */
        y[i]=re+im*(cf_t)I;
    }
}
/* scalar reference for --check */
static void cf_spmv_ref(cf_t *y, const long *row_ptr, const long *col_idx,
                        const cf_t *val, const cf_t *x, long n){
    for(long i=0;i<n;i++){ cf_t s=0.0f;
        for(long k=row_ptr[i];k<row_ptr[i+1];k++) s+=val[k]*x[col_idx[k]];
        y[i]=s; }
}

int main(int argc, char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s <matrix.mtx> [reps|--check]\n",argv[0]); return 1; }
    int check = (argc>=3 && strcmp(argv[2],"--check")==0);
    FILE *fp = fopen(argv[1],"r");
    if(!fp){ fprintf(stderr,"failed to open %s\n",argv[1]); return 1; }

    char line[512];
    /* skip comment/banner lines */
    do { if(!fgets(line,sizeof(line),fp)){ fprintf(stderr,"bad mtx\n"); return 1; } }
    while(line[0]=='%');
    long M,N,nnz;
    if(sscanf(line,"%ld %ld %ld",&M,&N,&nnz)!=3){ fprintf(stderr,"bad size line\n"); return 1; }

    /* read coordinate triples (1-based), build per-row counts */
    long *I_=malloc(sizeof(long)*nnz), *J_=malloc(sizeof(long)*nnz);
    double *V_=malloc(sizeof(double)*nnz);
    long *cnt=calloc(M+1,sizeof(long));
    for(long e=0;e<nnz;e++){
        if(!fgets(line,sizeof(line),fp)){ fprintf(stderr,"short mtx (%ld/%ld)\n",e,nnz); return 1; }
        long r,c; double v=1.0;                       /* pattern matrices: default val=1 */
        int got=sscanf(line,"%ld %ld %lf",&r,&c,&v);
        if(got<2){ fprintf(stderr,"bad entry\n"); return 1; }
        r--; c--;                                     /* MatrixMarket is 1-based */
        I_[e]=r; J_[e]=c; V_[e]=v; cnt[r]++;
    }
    fclose(fp);

    /* CSR row_ptr via prefix sum */
    long *row_ptr=malloc(sizeof(long)*(M+1));
    row_ptr[0]=0;
    for(long i=0;i<M;i++) row_ptr[i+1]=row_ptr[i]+cnt[i];
    long *col_idx=malloc(sizeof(long)*nnz);
    cf_t *val=malloc(sizeof(cf_t)*nnz);
    long *cur=malloc(sizeof(long)*M);
    for(long i=0;i<M;i++) cur[i]=row_ptr[i];
    for(long e=0;e<nnz;e++){
        long r=I_[e]; long p=cur[r]++;
        col_idx[p]=J_[e];
        val[p]=(float)V_[e] + (float)(0.3*V_[e])*I;   /* re = A, im = 0.3*A */
    }
    free(I_); free(J_); free(V_); free(cnt); free(cur);

    cf_t *x=malloc(sizeof(cf_t)*N), *y=malloc(sizeof(cf_t)*M);
    for(long i=0;i<N;i++){ double v=sin((double)i*0.001)+1.5; x[i]=(float)v+(float)(0.5*v)*I; }

    /* temp gather buffer sized to the longest row */
    long maxlen=0; for(long i=0;i<M;i++){ long l=row_ptr[i+1]-row_ptr[i]; if(l>maxlen)maxlen=l; }
    cf_t *tmp=malloc(sizeof(cf_t)*(maxlen>0?maxlen:1));

    if(check){
        cf_t *yr=malloc(sizeof(cf_t)*M);
        cf_spmv(y,row_ptr,col_idx,val,x,M,tmp);
        cf_spmv_ref(yr,row_ptr,col_idx,val,x,M);
        double em=0; for(long i=0;i<M;i++){ double d=cabsf(y[i]-yr[i]); if(d>em)em=d; }
        printf("cf backend compiled = %s\n", CF_BACKEND);
        printf("  check spmv   maxabs=%.3e (nrow=%ld nnz=%ld)\n", em, M, nnz);
        printf(em>1e-2?"CHECK FAILED\n":"CHECK OK\n");
        free(yr); free(tmp); free(row_ptr); free(col_idx); free(val); free(x); free(y);
        return em>1e-2;
    }

    cf_spmv(y,row_ptr,col_idx,val,x,M,tmp);           /* warmup */
    double t_spmv=0; TIME_OP(t_spmv, cf_spmv(y,row_ptr,col_idx,val,x,M,tmp));

    double mflop = 8.0*(double)nnz / t_spmv / 1e6;    /* complex spmv ~ 8 flops/nz */
    const char *mname = strrchr(argv[1],'/'); mname = mname? mname+1: argv[1];
    printf("RESULT_SPMV,%s,%s,%s,%ld,%.9e,%.3f\n", PREC_NAME, BACKEND_NAME, mname, nnz, t_spmv, mflop);

    free(tmp); free(row_ptr); free(col_idx); free(val); free(x); free(y);
    return 0;
}
