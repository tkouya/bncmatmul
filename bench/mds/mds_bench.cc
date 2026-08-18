/****************************************************************************/
/* mds_bench.cc : unified micro-benchmark for BNCmatmul                       */
/*   ops : axpy(add_cmul) / matvec / matmul / spmv                            */
/*   types: double,dd,td,qd, float,ds,ts,qs  (+ sparse: double,dd,td,qd)      */
/*   The SIMD backend (serial / avx2 / avx512) is selected at LINK time by    */
/*   the library this object is linked against, exactly as the other benches. */
/*                                                                            */
/*   Usage:  mds_bench <N> <backend-label>                                     */
/*   Output (CSV, one line per (type,op)):                                     */
/*     backend,type,op,N,seconds,iters,mflops                                  */
/****************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define USE_FLINEAR
#define USE_DDLINEAR
#define USE_TDLINEAR
#define USE_QDLINEAR
#define USE_DSLINEAR
#define USE_TSLINEAR
#define USE_QSLINEAR

#include "matmul_strassen.h"
#include "bncsparse.h"
#include "get_secv.h"

#include <qd/qd_real.h>   // fpu_fix_start

/* float matvec is compiled in the library under this (inconsistent) name */
extern "C" void mul_fmatrix_dvec(FVector v, FMatrix a, FVector vb);

volatile double g_sink = 0.0;

static const char *g_backend = "serial";
static long   g_N    = 128;
static double g_tmin = 0.20;   // accumulate repeats until this many seconds elapsed
static double g_cap  = 120.0;  // never repeat once a single call exceeds this

static void emit(const char *type, const char *op, double sec, long iters, double flops)
{
    double mflops = (sec > 0.0) ? (flops / sec / 1.0e6) : 0.0;
    printf("%s,%s,%s,%ld,%.6e,%ld,%.4f\n", g_backend, type, op, g_N, sec, iters, mflops);
    fflush(stdout);
}

/* run callable f() repeatedly, return per-call seconds (and #iters via *it) */
template <class F>
static double timeit(F f, long *it)
{
    double t0 = get_real_secv();
    f();
    double one = get_real_secv() - t0;
    if (one >= g_tmin || one >= g_cap) { *it = 1; return one; }

    long k = 2; double el = 0.0;
    for (;;) {
        double s = get_real_secv();
        for (long i = 0; i < k; i++) f();
        el = get_real_secv() - s;
        if (el >= g_tmin) break;
        if (k > (1L<<24)) break;
        k *= 2;
    }
    *it = k;
    return el / (double)k;
}

/* ============================ DENSE : plain (f,d) ======================== */
#define DEFINE_DENSE_PLAIN(TOK, NAME, BASE, MATVEC)                           \
static void bench_dense_##TOK(void)                                           \
{                                                                             \
    long n = g_N;                                                             \
    TOK##matrix *A = init_##TOK##matrix(n, n);                                \
    TOK##matrix *B = init_##TOK##matrix(n, n);                                \
    TOK##matrix *C = init_##TOK##matrix(n, n);                                \
    TOK##vector *x = init_##TOK##vector(n);                                   \
    TOK##vector *y = init_##TOK##vector(n);                                   \
    TOK##vector *z = init_##TOK##vector(n);                                   \
    for (long i = 0; i < n; i++) {                                            \
        for (long j = 0; j < n; j++) {                                        \
            BASE va = (BASE)(((i + 2*j) % 7 + 1) * 0.5);                       \
            BASE vb = (BASE)(((3*i + j) % 5 + 1) * 0.25);                      \
            set_##TOK##matrix_ij(A, i, j, va);                                \
            set_##TOK##matrix_ij(B, i, j, vb);                                \
        }                                                                     \
        set_##TOK##vector_i(x, i, (BASE)((i % 9 + 1) * 0.5));                  \
        set_##TOK##vector_i(y, i, (BASE)((i % 4 + 1) * 0.5));                  \
    }                                                                         \
    long it; double s; BASE alpha = (BASE)1.5;                                \
    s = timeit([&]{ add_cmul_##TOK##vector(z, y, alpha, x); }, &it);          \
    g_sink += (double)z->element[0];                                          \
    emit(NAME, "axpy", s, it, 2.0*(double)n);                                 \
    s = timeit([&]{ MATVEC(y, A, x); }, &it);                                 \
    g_sink += (double)y->element[0];                                          \
    emit(NAME, "matvec", s, it, 2.0*(double)n*(double)n);                     \
    s = timeit([&]{ mul_##TOK##matrix(C, A, B); }, &it);                      \
    g_sink += (double)C->element[0];                                          \
    emit(NAME, "matmul", s, it, 2.0*(double)n*(double)n*(double)n);           \
    free_##TOK##matrix(A); free_##TOK##matrix(B); free_##TOK##matrix(C);      \
    free_##TOK##vector(x); free_##TOK##vector(y); free_##TOK##vector(z);      \
}

/* ============================ DENSE : multi (dd,td,qd,ds,ts,qs) ========== */
#define DEFINE_DENSE_MP(TOK, NAME, BASE, SZ)                                  \
static void bench_dense_##TOK(void)                                           \
{                                                                             \
    long n = g_N;                                                             \
    TOK##matrix *A = init_##TOK##matrix(n, n);                                \
    TOK##matrix *B = init_##TOK##matrix(n, n);                                \
    TOK##matrix *C = init_##TOK##matrix(n, n);                                \
    TOK##vector *x = init_##TOK##vector(n);                                   \
    TOK##vector *y = init_##TOK##vector(n);                                   \
    TOK##vector *z = init_##TOK##vector(n);                                   \
    BASE tmp[SZ];                                                             \
    for (long i = 0; i < n; i++) {                                            \
        for (long j = 0; j < n; j++) {                                        \
            for (int k = 1; k < SZ; k++) tmp[k] = (BASE)0;                     \
            tmp[0] = (BASE)(((i + 2*j) % 7 + 1) * 0.5);                        \
            set_##TOK##matrix_ij(A, i, j, tmp);                              \
            tmp[0] = (BASE)(((3*i + j) % 5 + 1) * 0.25);                       \
            set_##TOK##matrix_ij(B, i, j, tmp);                              \
        }                                                                     \
        for (int k = 1; k < SZ; k++) tmp[k] = (BASE)0;                         \
        tmp[0] = (BASE)((i % 9 + 1) * 0.5); set_##TOK##vector_i(x, i, tmp);    \
        tmp[0] = (BASE)((i % 4 + 1) * 0.5); set_##TOK##vector_i(y, i, tmp);    \
    }                                                                         \
    long it; double s; BASE alpha[SZ];                                        \
    for (int k = 1; k < SZ; k++) alpha[k] = (BASE)0; alpha[0] = (BASE)1.5;     \
    s = timeit([&]{ add_cmul_##TOK##vector(z, y, alpha, x); }, &it);          \
    g_sink += (double)z->element[0][0];                                       \
    emit(NAME, "axpy", s, it, 2.0*(double)n);                                 \
    s = timeit([&]{ mul_##TOK##matrix_##TOK##vec(y, A, x); }, &it);           \
    g_sink += (double)y->element[0][0];                                       \
    emit(NAME, "matvec", s, it, 2.0*(double)n*(double)n);                     \
    s = timeit([&]{ mul_##TOK##matrix(C, A, B); }, &it);                      \
    g_sink += (double)C->element[0][0];                                       \
    emit(NAME, "matmul", s, it, 2.0*(double)n*(double)n*(double)n);           \
    free_##TOK##matrix(A); free_##TOK##matrix(B); free_##TOK##matrix(C);      \
    free_##TOK##vector(x); free_##TOK##vector(y); free_##TOK##vector(z);      \
}

DEFINE_DENSE_PLAIN(f,  "float",  float,  mul_fmatrix_dvec)
DEFINE_DENSE_PLAIN(d,  "double", double, mul_dmatrix_dvec)
DEFINE_DENSE_MP(dd, "dd", double, DDSIZE)
DEFINE_DENSE_MP(td, "td", double, TDSIZE)
DEFINE_DENSE_MP(qd, "qd", double, QDSIZE)
#ifndef MDS_NO_FLOATMP
DEFINE_DENSE_MP(ds, "ds", float,  DSSIZE)
DEFINE_DENSE_MP(ts, "ts", float,  TSSIZE)
DEFINE_DENSE_MP(qs, "qs", float,  QSSIZE)
#endif

/* ============================ SPARSE (banded) =========================== */
/* half-bandwidth; nnz/row ~ 2*HBW+1 */
#define HBW 3

/* sparse: double (plain) */
static void bench_sparse_d(void)
{
    long n = g_N;
    DMatrix Ad = init_dmatrix(n, n);
    for (long i = 0; i < n; i++)
        for (long j = (i-HBW<0?0:i-HBW); j <= (i+HBW>=n?n-1:i+HBW); j++)
            set_dmatrix_ij(Ad, i, j, (double)(((i + j) % 7 + 1) * 0.5));
    DRSMatrix A = init_set_drsmatrix_dmatrix(Ad);
    DVector x = init_dvector(n), y = init_dvector(n);
    for (long i = 0; i < n; i++) set_dvector_i(x, i, (double)((i % 9 + 1) * 0.5));
    long it; double s;
    s = timeit([&]{ mul_drsmatrix_dvec(y, A, x); }, &it);
    g_sink += (double)y->element[0];
    emit("double", "spmv", s, it, 2.0*(double)A->nzero_total_num);
    free_dmatrix(Ad); free_drsmatrix(A); free_dvector(x); free_dvector(y);
}

/* sparse: multi (dd,td,qd) -- build dense banded <TOK>matrix then convert */
#define DEFINE_SPARSE_MP(TOK, NAME, SZ)                                       \
static void bench_sparse_##TOK(void)                                          \
{                                                                             \
    long n = g_N;                                                             \
    TOK##matrix *Ad = init_##TOK##matrix(n, n);                               \
    double tmp[SZ];                                                           \
    for (long i = 0; i < n; i++)                                              \
        for (long j = (i-HBW<0?0:i-HBW); j <= (i+HBW>=n?n-1:i+HBW); j++) {     \
            for (int k = 1; k < SZ; k++) tmp[k] = 0.0;                         \
            tmp[0] = (double)(((i + j) % 7 + 1) * 0.5);                        \
            set_##TOK##matrix_ij(Ad, i, j, tmp);                            \
        }                                                                     \
    TOK##rsmatrix *A = init_set_##TOK##rsmatrix_##TOK##matrix(Ad);            \
    TOK##vector *x = init_##TOK##vector(n), *y = init_##TOK##vector(n);       \
    for (long i = 0; i < n; i++) {                                            \
        for (int k = 1; k < SZ; k++) tmp[k] = 0.0;                            \
        tmp[0] = (double)((i % 9 + 1) * 0.5); set_##TOK##vector_i(x, i, tmp);  \
    }                                                                         \
    long it; double s;                                                        \
    s = timeit([&]{ mul_##TOK##rsmatrix_##TOK##vec(y, A, x); }, &it);         \
    g_sink += (double)y->element[0][0];                                       \
    emit(NAME, "spmv", s, it, 2.0*(double)A->nzero_total_num);                \
    free_##TOK##matrix(Ad); free_##TOK##rsmatrix(A);                          \
    free_##TOK##vector(x); free_##TOK##vector(y);                             \
}
DEFINE_SPARSE_MP(dd, "dd", DDSIZE)
DEFINE_SPARSE_MP(td, "td", TDSIZE)
DEFINE_SPARSE_MP(qd, "qd", QDSIZE)

/* sparse: float (plain) -- newly added float-based real SpMV */
static void bench_sparse_f(void)
{
    long n = g_N;
    FMatrix Ad = init_fmatrix(n, n);
    for (long i = 0; i < n; i++)
        for (long j = (i-HBW<0?0:i-HBW); j <= (i+HBW>=n?n-1:i+HBW); j++)
            set_fmatrix_ij(Ad, i, j, (float)(((i + j) % 7 + 1) * 0.5));
    FRSMatrix A = init_set_frsmatrix_fmatrix(Ad);
    FVector x = init_fvector(n), y = init_fvector(n);
    for (long i = 0; i < n; i++) set_fvector_i(x, i, (float)((i % 9 + 1) * 0.5));
    long it; double s;
    s = timeit([&]{ mul_frsmatrix_fvec(y, A, x); }, &it);
    g_sink += (double)y->element[0];
    emit("float", "spmv", s, it, 2.0*(double)A->nzero_total_num);
    free_fmatrix(Ad); free_frsmatrix(A); free_fvector(x); free_fvector(y);
}

/* sparse: float-multi (ds,ts,qs) -- newly added float-extended SpMV.
 * set_<TOK>matrix_ij / set_<TOK>vector_i take float[SZ] (cf. double[SZ] for dd/td/qd). */
#define DEFINE_SPARSE_FMP(TOK, NAME, SZ)                                      \
static void bench_sparse_##TOK(void)                                          \
{                                                                             \
    long n = g_N;                                                             \
    TOK##matrix *Ad = init_##TOK##matrix(n, n);                               \
    float tmp[SZ];                                                            \
    for (long i = 0; i < n; i++)                                              \
        for (long j = (i-HBW<0?0:i-HBW); j <= (i+HBW>=n?n-1:i+HBW); j++) {     \
            for (int k = 1; k < SZ; k++) tmp[k] = 0.0f;                        \
            tmp[0] = (float)(((i + j) % 7 + 1) * 0.5);                         \
            set_##TOK##matrix_ij(Ad, i, j, tmp);                            \
        }                                                                     \
    TOK##rsmatrix *A = init_set_##TOK##rsmatrix_##TOK##matrix(Ad);            \
    TOK##vector *x = init_##TOK##vector(n), *y = init_##TOK##vector(n);       \
    for (long i = 0; i < n; i++) {                                            \
        for (int k = 1; k < SZ; k++) tmp[k] = 0.0f;                           \
        tmp[0] = (float)((i % 9 + 1) * 0.5); set_##TOK##vector_i(x, i, tmp);   \
    }                                                                         \
    long it; double s;                                                        \
    s = timeit([&]{ mul_##TOK##rsmatrix_##TOK##vec(y, A, x); }, &it);         \
    g_sink += (double)y->element[0][0];                                       \
    emit(NAME, "spmv", s, it, 2.0*(double)A->nzero_total_num);                \
    free_##TOK##matrix(Ad); free_##TOK##rsmatrix(A);                          \
    free_##TOK##vector(x); free_##TOK##vector(y);                             \
}
#ifndef MDS_NO_FLOATMP
DEFINE_SPARSE_FMP(ds, "ds", DSSIZE)
DEFINE_SPARSE_FMP(ts, "ts", TSSIZE)
DEFINE_SPARSE_FMP(qs, "qs", QSSIZE)
#endif

int main(int argc, char **argv)
{
    unsigned int oldcw; fpu_fix_start(&oldcw);
    if (argc >= 2) g_N = atol(argv[1]);
    if (argc >= 3) g_backend = argv[2];
    if (g_N < 2) { fprintf(stderr, "N too small\n"); return 1; }

    bench_dense_f();
#ifndef MDS_FLOAT_ONLY
    bench_dense_d();
    bench_dense_dd(); bench_dense_td(); bench_dense_qd();
#endif
#ifndef MDS_NO_FLOATMP
    bench_dense_ds(); bench_dense_ts(); bench_dense_qs();
#endif

#ifndef MDS_FLOAT_ONLY
    bench_sparse_d();
    bench_sparse_dd(); bench_sparse_td(); bench_sparse_qd();
#endif
    bench_sparse_f();
#ifndef MDS_NO_FLOATMP
    bench_sparse_ds(); bench_sparse_ts(); bench_sparse_qs();
#endif

    if (g_sink == 1.2345e-300) fprintf(stderr, "%g", g_sink); /* keep sink */
    return 0;
}
