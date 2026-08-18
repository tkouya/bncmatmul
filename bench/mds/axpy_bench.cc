/****************************************************************************/
/* axpy_bench.cc : AXPY (c = a + alpha*b, add_cmul) timing for BNCmatmul     */
/*   types  : double,dd,td,qd, float,ds,ts,qs                                */
/*            + MPFR at the matching mantissa bit length                     */
/*   config : serial (1 thread) or CPU-parallel (N threads), selected by the */
/*            <nthreads> argument.  Parallelism reuses the exact SIMD kernel  */
/*            by slicing each (struct-of-arrays) vector into per-thread       */
/*            chunks aligned to a multiple of 16 elements, so serial and      */
/*            OpenMP numbers are apples-to-apples (same kernel).              */
/*                                                                            */
/*   SIMD backend (serial/avx2/avx512) is selected at LINK time.             */
/*                                                                            */
/*   Usage : axpy_bench <N> <nthreads> <backend-label>                        */
/*   Output: CSV, one line per (type):                                        */
/*     backend,nthreads,family,type,bits,N,seconds,iters,mflops               */
/****************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <omp.h>

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
#include "mpflinear.h"

#include <qd/qd_real.h>   // fpu_fix_start

volatile double g_sink = 0.0;

static const char *g_backend = "serial";
static long   g_N    = 1048576;
static int    g_nt   = 1;
static double g_tmin = 0.30;   // accumulate repeats until this many seconds elapsed
static double g_cap  = 60.0;   // never repeat once a single call exceeds this
#define ALIGN16 16             // chunk alignment (covers AVX-512 float width 16)

static void emit(const char *family, const char *type, long bits,
                 double sec, long iters, double flops)
{
    double mflops = (sec > 0.0) ? (flops / sec / 1.0e6) : 0.0;
    printf("%s,%d,%s,%s,%ld,%ld,%.6e,%ld,%.4f\n",
           g_backend, g_nt, family, type, bits, g_N, sec, iters, mflops);
    fflush(stdout);
}

template <class F>
static double timeit_once(F f, long *it)
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

/* best (minimum) per-call time over a few outer measurements -> less jitter */
template <class F>
static double timeit(F f, long *it)
{
    double best = 1e30; long bit = 1;
    for (int r = 0; r < 3; r++) {
        long k; double s = timeit_once(f, &k);
        if (s < best) { best = s; bit = k; }
        if (best >= g_cap) break;
    }
    *it = bit;
    return best;
}

/* per-thread [off,off+chunk) split of real_dim, each chunk a multiple of 16 */
static inline void slice(long real_dim, int tid, int nt, long *off, long *chunk)
{
    long base = real_dim / nt;
    base -= base % ALIGN16;
    if (base < ALIGN16) base = ALIGN16;
    long o = (long)tid * base;
    if (o >= real_dim) { *off = real_dim; *chunk = 0; return; }
    *off = o;
    *chunk = (tid == nt - 1) ? (real_dim - o) : base;
}

/* ============================ PLAIN (f,d) =============================== */
#define DEFINE_AXPY_PLAIN(TOK, NAME, BASE, BITS, FAM)                         \
static void axpy_##TOK(TOK##vector *z, TOK##vector *y, BASE alpha,            \
                       TOK##vector *x, int nt)                               \
{                                                                            \
    _Pragma("omp parallel num_threads(nt)")                                  \
    {                                                                        \
        int tid = omp_get_thread_num(), n = omp_get_num_threads();           \
        long off, chunk; slice(z->real_dim, tid, n, &off, &chunk);           \
        if (chunk > 0) {                                                     \
            TOK##vector sz = *z, sy = *y, sx = *x;                           \
            sz.dim = sz.real_dim = chunk;                                    \
            sy.dim = sy.real_dim = chunk;                                    \
            sx.dim = sx.real_dim = chunk;                                    \
            sz.element += off; sy.element += off; sx.element += off;         \
            add_cmul_##TOK##vector(&sz, &sy, alpha, &sx);                    \
        }                                                                    \
    }                                                                        \
}                                                                            \
static void bench_##TOK(void)                                                \
{                                                                            \
    long n = g_N;                                                            \
    TOK##vector *x = init_##TOK##vector(n);                                  \
    TOK##vector *y = init_##TOK##vector(n);                                  \
    TOK##vector *z = init_##TOK##vector(n);                                  \
    for (long i = 0; i < n; i++) {                                           \
        set_##TOK##vector_i(x, i, (BASE)((i % 9 + 1) * 0.5));                 \
        set_##TOK##vector_i(y, i, (BASE)((i % 4 + 1) * 0.5));                 \
    }                                                                        \
    BASE alpha = (BASE)1.5;                                                  \
    long it;                                                                 \
    double s = timeit([&]{ axpy_##TOK(z, y, alpha, x, g_nt); }, &it);        \
    g_sink += (double)z->element[0];                                         \
    emit(FAM, NAME, BITS, s, it, 2.0*(double)n);                             \
    free_##TOK##vector(x); free_##TOK##vector(y); free_##TOK##vector(z);     \
}

/* ============================ MULTI (dd,td,qd,ds,ts,qs) ================= */
#define DEFINE_AXPY_MP(TOK, NAME, BASE, SZ, BITS, FAM)                       \
static void axpy_##TOK(TOK##vector *z, TOK##vector *y, BASE alpha[SZ],        \
                       TOK##vector *x, int nt)                               \
{                                                                            \
    _Pragma("omp parallel num_threads(nt)")                                  \
    {                                                                        \
        int tid = omp_get_thread_num(), n = omp_get_num_threads();           \
        long off, chunk; slice(z->real_dim, tid, n, &off, &chunk);           \
        if (chunk > 0) {                                                     \
            TOK##vector sz = *z, sy = *y, sx = *x;                           \
            sz.dim = sz.real_dim = chunk;                                    \
            sy.dim = sy.real_dim = chunk;                                    \
            sx.dim = sx.real_dim = chunk;                                    \
            for (int k = 0; k < SZ; k++) {                                   \
                sz.element[k] += off; sy.element[k] += off; sx.element[k] += off; \
            }                                                                \
            add_cmul_##TOK##vector(&sz, &sy, alpha, &sx);                    \
        }                                                                    \
    }                                                                        \
}                                                                            \
static void bench_##TOK(void)                                                \
{                                                                            \
    long n = g_N;                                                            \
    TOK##vector *x = init_##TOK##vector(n);                                  \
    TOK##vector *y = init_##TOK##vector(n);                                  \
    TOK##vector *z = init_##TOK##vector(n);                                  \
    BASE tmp[SZ];                                                            \
    for (long i = 0; i < n; i++) {                                           \
        for (int k = 1; k < SZ; k++) tmp[k] = (BASE)0;                       \
        tmp[0] = (BASE)((i % 9 + 1) * 0.5); set_##TOK##vector_i(x, i, tmp);   \
        tmp[0] = (BASE)((i % 4 + 1) * 0.5); set_##TOK##vector_i(y, i, tmp);   \
    }                                                                        \
    BASE alpha[SZ];                                                          \
    for (int k = 1; k < SZ; k++) alpha[k] = (BASE)0; alpha[0] = (BASE)1.5;    \
    long it;                                                                 \
    double s = timeit([&]{ axpy_##TOK(z, y, alpha, x, g_nt); }, &it);        \
    g_sink += (double)z->element[0][0];                                      \
    emit(FAM, NAME, BITS, s, it, 2.0*(double)n);                             \
    free_##TOK##vector(x); free_##TOK##vector(y); free_##TOK##vector(z);     \
}

DEFINE_AXPY_PLAIN(f, "float",  float,  24, "float-based")
DEFINE_AXPY_PLAIN(d, "double", double, 53, "double-based")
DEFINE_AXPY_MP(dd, "dd", double, DDSIZE, 106, "double-based")
DEFINE_AXPY_MP(td, "td", double, TDSIZE, 159, "double-based")
DEFINE_AXPY_MP(qd, "qd", double, QDSIZE, 212, "double-based")
DEFINE_AXPY_MP(ds, "ds", float,  DSSIZE,  48, "float-based")
DEFINE_AXPY_MP(ts, "ts", float,  TSSIZE,  72, "float-based")
DEFINE_AXPY_MP(qs, "qs", float,  QSSIZE,  96, "float-based")

/* ============================ MPFR (matching mantissa) ================= */
static void axpy_mpf(MPFVector z, MPFVector y, mpf_t alpha, MPFVector x, int nt)
{
    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num(), n = omp_get_num_threads();
        long base = z->dim / n;
        long off  = (long)tid * base;
        long chunk = (tid == n - 1) ? (z->dim - off) : base;
        if (off < z->dim && chunk > 0) {
            mpfvector sz = *z, sy = *y, sx = *x;
            sz.dim = sz.real_dim = chunk;
            sy.dim = sy.real_dim = chunk;
            sx.dim = sx.real_dim = chunk;
            sz.element += off; sy.element += off; sx.element += off;
            add_cmul_mpfvector(&sz, &sy, alpha, &sx);
        }
    }
}
static void bench_mpf(const char *name, long bits)
{
    long n = g_N;
    unsigned long prec = (unsigned long)bits;
    set_bnc_default_prec(prec);
    MPFVector x = init2_mpfvector(n, prec);
    MPFVector y = init2_mpfvector(n, prec);
    MPFVector z = init2_mpfvector(n, prec);
    for (long i = 0; i < n; i++) {
        set_mpfvector_i_d(x, i, (double)((i % 9 + 1) * 0.5));
        set_mpfvector_i_d(y, i, (double)((i % 4 + 1) * 0.5));
    }
    mpf_t alpha; mpf_init2(alpha, prec); mpf_set_d(alpha, 1.5);
    long it;
    double s = timeit([&]{ axpy_mpf(z, y, alpha, x, g_nt); }, &it);
    g_sink += mpf_get_d(z->element[0]);
    emit("mpfr", name, bits, s, it, 2.0*(double)n);
    mpf_clear(alpha);
    free_mpfvector(x); free_mpfvector(y); free_mpfvector(z);
}

int main(int argc, char **argv)
{
    unsigned int oldcw; fpu_fix_start(&oldcw);
    if (argc >= 2) g_N = atol(argv[1]);
    if (argc >= 3) g_nt = atoi(argv[2]);
    if (argc >= 4) g_backend = argv[3];
    if (g_N < 16) { fprintf(stderr, "N too small\n"); return 1; }
    if (g_nt < 1) g_nt = 1;
    /* round N up to a multiple of 16*nt so per-thread chunks stay 16-aligned */
    long unit = (long)ALIGN16 * g_nt;
    if (g_N % unit) g_N += unit - (g_N % unit);

    bench_f();
    bench_d();
    bench_dd(); bench_td(); bench_qd();
    bench_ds(); bench_ts(); bench_qs();

    /* MPFR at each matching mantissa length */
    bench_mpf("mpfr24",  24);   /* == float  */
    bench_mpf("mpfr48",  48);   /* == ds     */
    bench_mpf("mpfr53",  53);   /* == double */
    bench_mpf("mpfr72",  72);   /* == ts     */
    bench_mpf("mpfr96",  96);   /* == qs     */
    bench_mpf("mpfr106", 106);  /* == dd     */
    bench_mpf("mpfr159", 159);  /* == td     */
    bench_mpf("mpfr212", 212);  /* == qd     */

    if (g_sink == 1.2345e-300) fprintf(stderr, "%g", g_sink);
    return 0;
}
