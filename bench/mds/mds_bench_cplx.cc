/****************************************************************************/
/* mds_bench_cplx.cc : complex-arithmetic micro-benchmark for BNCmatmul       */
/*   types: cd (complex double), cdd, ctd, cqd                                */
/*   ops  : axpy(add_cmul) / matvec / matmul / spmv                           */
/*   Usage: mds_bench_cplx <N> <backend-label> [type]                          */
/*   Output (CSV): backend,type,op,N,seconds,iters,mflops                      */
/****************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define USE_DDLINEAR
#define USE_TDLINEAR
#define USE_QDLINEAR
#define USE_FLINEAR
#define USE_DSLINEAR
#define USE_TSLINEAR
#define USE_QSLINEAR

#include "matmul_strassen.h"
#include "bncsparse.h"
#include "get_secv.h"
#include <qd/qd_real.h>

/* float matvec is compiled in the library under this (inconsistent) name
   (FVector args despite the "dvec" suffix), same as in mds_bench.cc. */
extern "C" void mul_fmatrix_dvec(FVector v, FMatrix a, FVector vb);

/* init_set converters for cdd/ctd/cqd sparse are not declared in bncsparse.h
   but are present in the library (C linkage). */
extern "C" CDDRSMatrix init_set_cddrsmatrix_cddmatrix(CDDMatrix);
extern "C" CTDRSMatrix init_set_ctdrsmatrix_ctdmatrix(CTDMatrix);
extern "C" CQDRSMatrix init_set_cqdrsmatrix_cqdmatrix(CQDMatrix);

volatile double g_sink = 0.0;
static const char *g_backend = "serial";
static long   g_N    = 128;
static double g_tmin = 0.20;
static double g_cap  = 120.0;

static void emit(const char *type, const char *op, double sec, long iters, double flops)
{
    double mflops = (sec > 0.0) ? (flops / sec / 1.0e6) : 0.0;
    printf("%s,%s,%s,%ld,%.6e,%ld,%.4f\n", g_backend, type, op, g_N, sec, iters, mflops);
    fflush(stdout);
}
template <class F> static double timeit(F f, long *it)
{
    double t0 = get_real_secv(); f(); double one = get_real_secv() - t0;
    if (one >= g_tmin || one >= g_cap) { *it = 1; return one; }
    long k = 2; double el = 0.0;
    for (;;) { double s = get_real_secv(); for (long i=0;i<k;i++) f(); el = get_real_secv()-s;
        if (el >= g_tmin) break; if (k > (1L<<24)) break; k *= 2; }
    *it = k; return el / (double)k;
}
#define HBW 3
static double banded_nnz(long n){ double s=0; for(long i=0;i<n;i++){ long lo=(i-HBW<0?0:i-HBW), hi=(i+HBW>=n?n-1:i+HBW); s += (double)(hi-lo+1);} return s; }

/* ---- cd : plain complex double ---- */
static void bench_cd(void)
{
    long n = g_N;
    CDMatrix A = init_cdmatrix(n,n), B = init_cdmatrix(n,n), C = init_cdmatrix(n,n);
    CDVector x = init_cdvector(n), y = init_cdvector(n), z = init_cdvector(n);
    for (long i=0;i<n;i++){
        for (long j=0;j<n;j++){
            double _Complex va, vb; __real__ va=((i+2*j)%7+1)*0.5; __imag__ va=((i+j)%3+1)*0.25;
            __real__ vb=((3*i+j)%5+1)*0.25; __imag__ vb=((i+2*j)%4+1)*0.1;
            set_cdmatrix_ij(A,i,j,va); set_cdmatrix_ij(B,i,j,vb);
        }
        double _Complex vx; __real__ vx=(i%9+1)*0.5; __imag__ vx=(i%5+1)*0.2; set_cdvector_i(x,i,vx);
        double _Complex vy; __real__ vy=(i%4+1)*0.5; __imag__ vy=(i%6+1)*0.1; set_cdvector_i(y,i,vy);
    }
    long it; double s; double _Complex alpha; __real__ alpha=1.5; __imag__ alpha=0.5;
    s=timeit([&]{ add_cmul_cdvector(z,y,alpha,x); },&it); g_sink+=__real__ z->element[0];
    emit("cd","axpy",s,it,2.0*n);
    s=timeit([&]{ mul_cdmatrix_cdvec(y,A,x); },&it); g_sink+=__real__ y->element[0];
    emit("cd","matvec",s,it,2.0*n*n);
    s=timeit([&]{ mul_cdmatrix(C,A,B); },&it); g_sink+=__real__ C->element[0];
    emit("cd","matmul",s,it,2.0*n*n*n);
    /* sparse */
    CDMatrix Ad = init_cdmatrix(n,n);
    for (long i=0;i<n;i++) for (long j=(i-HBW<0?0:i-HBW); j<=(i+HBW>=n?n-1:i+HBW); j++){
        double _Complex v; __real__ v=((i+j)%7+1)*0.5; __imag__ v=((i+j)%3+1)*0.25; set_cdmatrix_ij(Ad,i,j,v); }
    CDRSMatrix As = init_set_cdrsmatrix_cdmatrix(Ad);
    s=timeit([&]{ mul_cdrsmatrix_cdvec(y,As,x); },&it); g_sink+=__real__ y->element[0];
    emit("cd","spmv",s,it,2.0*banded_nnz(n));
    free_cdmatrix(A);free_cdmatrix(B);free_cdmatrix(C);free_cdmatrix(Ad);
    free_cdrsmatrix(As); free_cdvector(x);free_cdvector(y);free_cdvector(z);
}

/* ---- cdd/ctd/cqd : multi-precision complex ---- */
#define DEFINE_CPLX_MP(TOK, NAME, FT, SZ)                                       \
static void bench_##TOK(void)                                                   \
{                                                                               \
    long n = g_N;                                                               \
    TOK##matrix *A=init_##TOK##matrix(n,n),*B=init_##TOK##matrix(n,n),*C=init_##TOK##matrix(n,n); \
    TOK##vector *x=init_##TOK##vector(n),*y=init_##TOK##vector(n),*z=init_##TOK##vector(n); \
    for (long i=0;i<n;i++){                                                      \
        for (long j=0;j<n;j++){                                                  \
            double _Complex va,vb; __real__ va=((i+2*j)%7+1)*0.5; __imag__ va=((i+j)%3+1)*0.25; \
            __real__ vb=((3*i+j)%5+1)*0.25; __imag__ vb=((i+2*j)%4+1)*0.1;        \
            set_##TOK##matrix_ij_cd(A,i,j,va); set_##TOK##matrix_ij_cd(B,i,j,vb); \
        }                                                                       \
        double _Complex vx; __real__ vx=(i%9+1)*0.5; __imag__ vx=(i%5+1)*0.2; set_##TOK##vector_i_cd(x,i,vx); \
        double _Complex vy; __real__ vy=(i%4+1)*0.5; __imag__ vy=(i%6+1)*0.1; set_##TOK##vector_i_cd(y,i,vy); \
    }                                                                           \
    long it; double s; FT alpha; memset(&alpha,0,sizeof(alpha)); alpha.val_re[0]=1.5; alpha.val_im[0]=0.5; \
    s=timeit([&]{ add_cmul_##TOK##vector(z,y,&alpha,x); },&it); g_sink+=z->re->element[0][0]; \
    emit(NAME,"axpy",s,it,2.0*n);                                               \
    s=timeit([&]{ mul_##TOK##matrix_##TOK##vec(y,A,x); },&it); g_sink+=y->re->element[0][0]; \
    emit(NAME,"matvec",s,it,2.0*n*n);                                           \
    s=timeit([&]{ mul_##TOK##matrix(C,A,B); },&it); g_sink+=C->re->element[0][0]; \
    emit(NAME,"matmul",s,it,2.0*n*n*n);                                         \
    TOK##matrix *Ad=init_##TOK##matrix(n,n);                                     \
    for (long i=0;i<n;i++) for (long j=(i-HBW<0?0:i-HBW); j<=(i+HBW>=n?n-1:i+HBW); j++){ \
        double _Complex v; __real__ v=((i+j)%7+1)*0.5; __imag__ v=((i+j)%3+1)*0.25; set_##TOK##matrix_ij_cd(Ad,i,j,v); } \
    TOK##rsmatrix *As=init_set_##TOK##rsmatrix_##TOK##matrix(Ad);               \
    s=timeit([&]{ mul_##TOK##rsmatrix_##TOK##vec(y,As,x); },&it); g_sink+=y->re->element[0][0]; \
    emit(NAME,"spmv",s,it,2.0*banded_nnz(n));                            \
    free_##TOK##matrix(A);free_##TOK##matrix(B);free_##TOK##matrix(C);free_##TOK##matrix(Ad); \
    free_##TOK##rsmatrix(As); free_##TOK##vector(x);free_##TOK##vector(y);free_##TOK##vector(z); \
}
DEFINE_CPLX_MP(cdd, "cdd", cddfloat, DDSIZE)
DEFINE_CPLX_MP(ctd, "ctd", ctdfloat, TDSIZE)
DEFINE_CPLX_MP(cqd, "cqd", cqdfloat, QDSIZE)

/* ---- cf : plain complex float ----
   The float-based complex types (cf/cds/cts/cqs) are NOT implemented as native
   library types (only cd/cdd/ctd/cqd exist).  We therefore time the complex
   operations decomposed over the EXISTING real float / float-EFT routines on the
   real and imaginary parts -- exactly the re/im split the native cdd/ctd/cqd
   types use internally (cddvector = {DDVector re; DDVector im;}).
     axpy : z = y + alpha*x   (alpha,x,y,z complex)
     matvec/matmul/spmv : (re,im) via the 4-multiply rule
       (a+bi)(c+di) = (ac - bd) + (ad + bc) i                                  */
static void bench_cf(void)
{
    long n = g_N;
    FMatrix Are=init_fmatrix(n,n),Aim=init_fmatrix(n,n);
    FMatrix Bre=init_fmatrix(n,n),Bim=init_fmatrix(n,n);
    FMatrix Cre=init_fmatrix(n,n),Cim=init_fmatrix(n,n);
    FMatrix T1=init_fmatrix(n,n),T2=init_fmatrix(n,n);
    FVector xre=init_fvector(n),xim=init_fvector(n);
    FVector yre=init_fvector(n),yim=init_fvector(n);
    FVector zre=init_fvector(n),zim=init_fvector(n);
    FVector t1=init_fvector(n),t2=init_fvector(n);
    for (long i=0;i<n;i++){
        for (long j=0;j<n;j++){
            set_fmatrix_ij(Are,i,j,(float)(((i+2*j)%7+1)*0.5));
            set_fmatrix_ij(Aim,i,j,(float)(((i+j)%3+1)*0.25));
            set_fmatrix_ij(Bre,i,j,(float)(((3*i+j)%5+1)*0.25));
            set_fmatrix_ij(Bim,i,j,(float)(((i+2*j)%4+1)*0.1));
        }
        set_fvector_i(xre,i,(float)((i%9+1)*0.5)); set_fvector_i(xim,i,(float)((i%5+1)*0.2));
        set_fvector_i(yre,i,(float)((i%4+1)*0.5)); set_fvector_i(yim,i,(float)((i%6+1)*0.1));
    }
    long it; double s; float are=1.5f, aim=0.5f;
    /* complex axpy: z = y + alpha*x */
    s=timeit([&]{
        add_cmul_fvector(t1,yre,are,xre); sub_cmul_fvector(zre,t1,aim,xim);
        add_cmul_fvector(t1,yim,are,xim); add_cmul_fvector(zim,t1,aim,xre);
    },&it); g_sink+=(double)zre->element[0];
    emit("cf","axpy",s,it,2.0*n);
    /* complex matvec: y = A*x */
    s=timeit([&]{
        mul_fmatrix_dvec(t1,Are,xre); mul_fmatrix_dvec(t2,Aim,xim); sub_fvector(yre,t1,t2);
        mul_fmatrix_dvec(t1,Are,xim); mul_fmatrix_dvec(t2,Aim,xre); add_fvector(yim,t1,t2);
    },&it); g_sink+=(double)yre->element[0];
    emit("cf","matvec",s,it,2.0*n*n);
    /* complex matmul: C = A*B */
    s=timeit([&]{
        mul_fmatrix(T1,Are,Bre); mul_fmatrix(T2,Aim,Bim); sub_fmatrix(Cre,T1,T2);
        mul_fmatrix(T1,Are,Bim); mul_fmatrix(T2,Aim,Bre); add_fmatrix(Cim,T1,T2);
    },&it); g_sink+=(double)Cre->element[0];
    emit("cf","matmul",s,it,2.0*n*n*n);
    /* complex spmv */
    FMatrix Adre=init_fmatrix(n,n),Adim=init_fmatrix(n,n);
    for (long i=0;i<n;i++) for (long j=(i-HBW<0?0:i-HBW); j<=(i+HBW>=n?n-1:i+HBW); j++){
        set_fmatrix_ij(Adre,i,j,(float)(((i+j)%7+1)*0.5));
        set_fmatrix_ij(Adim,i,j,(float)(((i+j)%3+1)*0.25));
    }
    FRSMatrix Asre=init_set_frsmatrix_fmatrix(Adre), Asim=init_set_frsmatrix_fmatrix(Adim);
    s=timeit([&]{
        mul_frsmatrix_fvec(t1,Asre,xre); mul_frsmatrix_fvec(t2,Asim,xim); sub_fvector(yre,t1,t2);
        mul_frsmatrix_fvec(t1,Asre,xim); mul_frsmatrix_fvec(t2,Asim,xre); add_fvector(yim,t1,t2);
    },&it); g_sink+=(double)yre->element[0];
    emit("cf","spmv",s,it,2.0*banded_nnz(n));
    free_fmatrix(Are);free_fmatrix(Aim);free_fmatrix(Bre);free_fmatrix(Bim);
    free_fmatrix(Cre);free_fmatrix(Cim);free_fmatrix(T1);free_fmatrix(T2);
    free_fmatrix(Adre);free_fmatrix(Adim);
    free_frsmatrix(Asre);free_frsmatrix(Asim);
    free_fvector(xre);free_fvector(xim);free_fvector(yre);free_fvector(yim);
    free_fvector(zre);free_fvector(zim);free_fvector(t1);free_fvector(t2);
}

/* ---- cds/cts/cqs : multi-precision complex float (re/im over ds/ts/qs) ---- */
#define DEFINE_CPLX_FMP(TOK, NAME, SZ)                                          \
static void bench_c##TOK(void)                                                  \
{                                                                               \
    long n = g_N;                                                               \
    TOK##matrix *Are=init_##TOK##matrix(n,n),*Aim=init_##TOK##matrix(n,n);       \
    TOK##matrix *Bre=init_##TOK##matrix(n,n),*Bim=init_##TOK##matrix(n,n);       \
    TOK##matrix *Cre=init_##TOK##matrix(n,n),*Cim=init_##TOK##matrix(n,n);       \
    TOK##matrix *T1=init_##TOK##matrix(n,n),*T2=init_##TOK##matrix(n,n);         \
    TOK##vector *xre=init_##TOK##vector(n),*xim=init_##TOK##vector(n);           \
    TOK##vector *yre=init_##TOK##vector(n),*yim=init_##TOK##vector(n);           \
    TOK##vector *zre=init_##TOK##vector(n),*zim=init_##TOK##vector(n);           \
    TOK##vector *t1=init_##TOK##vector(n),*t2=init_##TOK##vector(n);             \
    float tmp[SZ]; for (int k=1;k<SZ;k++) tmp[k]=0.0f;                          \
    for (long i=0;i<n;i++){                                                      \
        for (long j=0;j<n;j++){                                                  \
            tmp[0]=(float)(((i+2*j)%7+1)*0.5); set_##TOK##matrix_ij(Are,i,j,tmp);\
            tmp[0]=(float)(((i+j)%3+1)*0.25);  set_##TOK##matrix_ij(Aim,i,j,tmp);\
            tmp[0]=(float)(((3*i+j)%5+1)*0.25);set_##TOK##matrix_ij(Bre,i,j,tmp);\
            tmp[0]=(float)(((i+2*j)%4+1)*0.1); set_##TOK##matrix_ij(Bim,i,j,tmp);\
        }                                                                       \
        tmp[0]=(float)((i%9+1)*0.5); set_##TOK##vector_i(xre,i,tmp);             \
        tmp[0]=(float)((i%5+1)*0.2); set_##TOK##vector_i(xim,i,tmp);             \
        tmp[0]=(float)((i%4+1)*0.5); set_##TOK##vector_i(yre,i,tmp);             \
        tmp[0]=(float)((i%6+1)*0.1); set_##TOK##vector_i(yim,i,tmp);             \
    }                                                                           \
    long it; double s; float are[SZ],aim[SZ];                                   \
    for (int k=1;k<SZ;k++){are[k]=0.0f;aim[k]=0.0f;} are[0]=1.5f; aim[0]=0.5f;   \
    /* complex axpy: z = y + alpha*x */                                         \
    s=timeit([&]{                                                                \
        add_cmul_##TOK##vector(t1,yre,are,xre); sub_cmul_##TOK##vector(zre,t1,aim,xim); \
        add_cmul_##TOK##vector(t1,yim,are,xim); add_cmul_##TOK##vector(zim,t1,aim,xre); \
    },&it); g_sink+=(double)zre->element[0][0];                                  \
    emit(NAME,"axpy",s,it,2.0*n);                                               \
    /* complex matvec: y = A*x */                                               \
    s=timeit([&]{                                                                \
        mul_##TOK##matrix_##TOK##vec(t1,Are,xre); mul_##TOK##matrix_##TOK##vec(t2,Aim,xim); sub_##TOK##vector(yre,t1,t2); \
        mul_##TOK##matrix_##TOK##vec(t1,Are,xim); mul_##TOK##matrix_##TOK##vec(t2,Aim,xre); add_##TOK##vector(yim,t1,t2); \
    },&it); g_sink+=(double)yre->element[0][0];                                  \
    emit(NAME,"matvec",s,it,2.0*n*n);                                           \
    /* complex matmul: C = A*B */                                               \
    s=timeit([&]{                                                                \
        mul_##TOK##matrix(T1,Are,Bre); mul_##TOK##matrix(T2,Aim,Bim); sub_##TOK##matrix(Cre,T1,T2); \
        mul_##TOK##matrix(T1,Are,Bim); mul_##TOK##matrix(T2,Aim,Bre); add_##TOK##matrix(Cim,T1,T2); \
    },&it); g_sink+=(double)Cre->element[0][0];                                  \
    emit(NAME,"matmul",s,it,2.0*n*n*n);                                         \
    /* complex spmv */                                                          \
    TOK##matrix *Adre=init_##TOK##matrix(n,n),*Adim=init_##TOK##matrix(n,n);     \
    for (int k=1;k<SZ;k++) tmp[k]=0.0f;                                         \
    for (long i=0;i<n;i++) for (long j=(i-HBW<0?0:i-HBW); j<=(i+HBW>=n?n-1:i+HBW); j++){ \
        tmp[0]=(float)(((i+j)%7+1)*0.5); set_##TOK##matrix_ij(Adre,i,j,tmp);    \
        tmp[0]=(float)(((i+j)%3+1)*0.25);set_##TOK##matrix_ij(Adim,i,j,tmp); }  \
    TOK##rsmatrix *Asre=init_set_##TOK##rsmatrix_##TOK##matrix(Adre);            \
    TOK##rsmatrix *Asim=init_set_##TOK##rsmatrix_##TOK##matrix(Adim);            \
    s=timeit([&]{                                                                \
        mul_##TOK##rsmatrix_##TOK##vec(t1,Asre,xre); mul_##TOK##rsmatrix_##TOK##vec(t2,Asim,xim); sub_##TOK##vector(yre,t1,t2); \
        mul_##TOK##rsmatrix_##TOK##vec(t1,Asre,xim); mul_##TOK##rsmatrix_##TOK##vec(t2,Asim,xre); add_##TOK##vector(yim,t1,t2); \
    },&it); g_sink+=(double)yre->element[0][0];                                  \
    emit(NAME,"spmv",s,it,2.0*banded_nnz(n));                                   \
    free_##TOK##matrix(Are);free_##TOK##matrix(Aim);free_##TOK##matrix(Bre);free_##TOK##matrix(Bim); \
    free_##TOK##matrix(Cre);free_##TOK##matrix(Cim);free_##TOK##matrix(T1);free_##TOK##matrix(T2); \
    free_##TOK##matrix(Adre);free_##TOK##matrix(Adim);                          \
    free_##TOK##rsmatrix(Asre);free_##TOK##rsmatrix(Asim);                      \
    free_##TOK##vector(xre);free_##TOK##vector(xim);free_##TOK##vector(yre);free_##TOK##vector(yim); \
    free_##TOK##vector(zre);free_##TOK##vector(zim);free_##TOK##vector(t1);free_##TOK##vector(t2); \
}
DEFINE_CPLX_FMP(ds, "cds", DSSIZE)
DEFINE_CPLX_FMP(ts, "cts", TSSIZE)
DEFINE_CPLX_FMP(qs, "cqs", QSSIZE)

int main(int argc, char **argv)
{
    unsigned int oldcw; fpu_fix_start(&oldcw);
    if (argc >= 2) g_N = atol(argv[1]);
    if (argc >= 3) g_backend = argv[2];
    const char *only = (argc >= 4) ? argv[3] : NULL;
    if (g_N < 2) return 1;
    if (!only || !strcmp(only,"cd"))  bench_cd();
    if (!only || !strcmp(only,"cdd")) bench_cdd();
    if (!only || !strcmp(only,"ctd")) bench_ctd();
    if (!only || !strcmp(only,"cqd")) bench_cqd();
    if (!only || !strcmp(only,"cf"))  bench_cf();
    if (!only || !strcmp(only,"cds")) bench_cds();
    if (!only || !strcmp(only,"cts")) bench_cts();
    if (!only || !strcmp(only,"cqs")) bench_cqs();
    if (g_sink == 1.2345e-300) fprintf(stderr, "%g", g_sink);
    return 0;
}
