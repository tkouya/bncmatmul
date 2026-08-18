/* mds_verify.cc : correctness check (checksums) for ds/ts/qs (+dd control).
   Build serial and avx512 and diff the output; values must match. */
#include <cstdio>
#include <cstdlib>
#define USE_DDLINEAR
#define USE_TDLINEAR
#define USE_QDLINEAR
#define USE_DSLINEAR
#define USE_TSLINEAR
#define USE_QSLINEAR
#include "matmul_strassen.h"
#include <qd/qd_real.h>

#define DEFINE_VERIFY(TOK, NAME, BASE, SZ, MATVEC)                            \
static void verify_##TOK(long n)                                             \
{                                                                            \
    TOK##matrix *A=init_##TOK##matrix(n,n),*B=init_##TOK##matrix(n,n),*C=init_##TOK##matrix(n,n); \
    TOK##vector *x=init_##TOK##vector(n),*y=init_##TOK##vector(n),*z=init_##TOK##vector(n); \
    BASE tmp[SZ];                                                            \
    for(long i=0;i<n;i++){                                                   \
        for(long j=0;j<n;j++){                                               \
            for(int k=1;k<SZ;k++) tmp[k]=(BASE)0;                            \
            tmp[0]=(BASE)(((i+2*j)%7+1)*0.5); set_##TOK##matrix_ij(A,i,j,tmp); \
            tmp[0]=(BASE)(((3*i+j)%5+1)*0.25); set_##TOK##matrix_ij(B,i,j,tmp); \
        }                                                                    \
        for(int k=1;k<SZ;k++) tmp[k]=(BASE)0;                               \
        tmp[0]=(BASE)((i%9+1)*0.5); set_##TOK##vector_i(x,i,tmp);            \
        tmp[0]=(BASE)((i%4+1)*0.5); set_##TOK##vector_i(y,i,tmp);            \
    }                                                                        \
    BASE alpha[SZ]; for(int k=1;k<SZ;k++) alpha[k]=(BASE)0; alpha[0]=(BASE)1.5; \
    add_cmul_##TOK##vector(z,y,alpha,x);                                     \
    MATVEC(y,A,x);                                                           \
    mul_##TOK##matrix(C,A,B);                                                \
    double sa=0,sv=0,sm=0;                                                   \
    for(long i=0;i<n;i++){ for(int q=0;q<SZ;q++){sa+=(double)get_##TOK##vector_i(z,i)[q]; sv+=(double)get_##TOK##vector_i(y,i)[q];} } \
    for(long i=0;i<n;i++) for(long j=0;j<n;j++) for(int q=0;q<SZ;q++) sm+=(double)get_##TOK##matrix_ij(C,i,j)[q]; \
    printf("%-4s axpy=%.6e matvec=%.6e matmul=%.6e\n", NAME, sa, sv, sm);    \
}
#define MV(TOK) mul_##TOK##matrix_##TOK##vec
DEFINE_VERIFY(dd,"dd",double,DDSIZE, MV(dd))
DEFINE_VERIFY(td,"td",double,TDSIZE, MV(td))
DEFINE_VERIFY(qd,"qd",double,QDSIZE, MV(qd))
DEFINE_VERIFY(ds,"ds",float, DSSIZE, MV(ds))
DEFINE_VERIFY(ts,"ts",float, TSSIZE, MV(ts))
DEFINE_VERIFY(qs,"qs",float, QSSIZE, MV(qs))

#include "bncsparse.h"
#define HBW 3
#define DEFINE_SPVERIFY(TOK, NAME, SZ)                                          \
static void spverify_##TOK(long n){                                            \
    TOK##matrix *Ad=init_##TOK##matrix(n,n); double tmp[SZ];                    \
    for(long i=0;i<n;i++) for(long j=(i-HBW<0?0:i-HBW); j<=(i+HBW>=n?n-1:i+HBW); j++){ \
        for(int k=1;k<SZ;k++) tmp[k]=0.0; tmp[0]=(double)(((i+j)%7+1)*0.5); set_##TOK##matrix_ij(Ad,i,j,tmp);} \
    TOK##rsmatrix *A=init_set_##TOK##rsmatrix_##TOK##matrix(Ad);                \
    TOK##vector *x=init_##TOK##vector(n),*y=init_##TOK##vector(n);              \
    for(long i=0;i<n;i++){ for(int k=1;k<SZ;k++) tmp[k]=0.0; tmp[0]=(double)((i%9+1)*0.5); set_##TOK##vector_i(x,i,tmp);} \
    mul_##TOK##rsmatrix_##TOK##vec(y,A,x);                                     \
    double s=0; for(long i=0;i<n;i++) for(int q=0;q<SZ;q++) s+=(double)get_##TOK##vector_i(y,i)[q]; \
    TOK##vector *yt=init_##TOK##vector(n);                                      \
    for(long i=0;i<n;i++){ for(int k=1;k<SZ;k++) tmp[k]=0.0; tmp[0]=0.0; set_##TOK##vector_i(yt,i,tmp);} \
    mul_##TOK##rsmatrixt_##TOK##vec(yt,A,x);                                    \
    double st=0; for(long i=0;i<n;i++) for(int q=0;q<SZ;q++) st+=(double)get_##TOK##vector_i(yt,i)[q]; \
    printf("%-4s spmv=%.6e spmvt=%.6e\n", NAME, s, st);                         \
}
DEFINE_SPVERIFY(dd,"dd",DDSIZE)
DEFINE_SPVERIFY(td,"td",TDSIZE)
DEFINE_SPVERIFY(qd,"qd",QDSIZE)

int main(int argc,char**argv){
    unsigned int cw; fpu_fix_start(&cw);
    long n = (argc>=2)?atol(argv[1]):96;
    verify_dd(n); verify_td(n); verify_qd(n);
    verify_ds(n); verify_ts(n); verify_qs(n);
    spverify_dd(n); spverify_td(n); spverify_qd(n);
    return 0;
}
