/*****************************************************************
 * bench_gdtqs_cmul_vector.cu
 * ----------------------------------------------------------------
 * Performance benchmark for the GPU constant-times-vector kernels
 *   cmul_g{dd,td,qd,ds,ts,qs}vector_dev   (c = val * a)
 * across all 6 multi-precision layers (double-based: gdd/gtd/gqd,
 * float-based: gds/gts/gqs).
 *
 * For each precision and vector length N it times
 *     - CPU serial : cmul_<prec>vector(c, val[], a)
 *     - GPU kernel : cmul_g<prec>vector_dev(c_d, gprec_real_val, a_d, ...)
 *   (minimum wall-clock over `reps` repetitions, after one warm-up),
 *   compares the GPU output against the CPU output (Euclidean norm),
 *   and prints the speedup + relative error.
 *
 * Build:  make -C src/cuda bench_gdtqs_cmul_vector
 *
 * Usage:  ./bench_gdtqs_cmul_vector [options]
 *   --reps R          timed reps per measurement                 (default 5)
 *   --blocks B        CUDA blocks per grid                       (default 128)
 *   --threads T       CUDA threads per block                     (default 128)
 *   --sizes N1,N2,... override default size list
 *                     (default 4096,16384,65536,262144,1048576,4194304)
 *
 *   The CPU reference uses the bncmatmul scalar cmul; linking therefore
 *   requires libbncmatmul-X.Y.{a,so} in addition to libbncmm_cuda.a.
 *****************************************************************/
/* __NV_NO_VECTOR_DEPRECATION_DIAG is set on the nvcc command line. */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

#include <qd/qd_real.h>
#include <qd/fpu.h>

#include "rds.h"          /* dsfloat / tsfloat / qsfloat + c_{ds,ts,qs}_copy_d */

#include <cuda_runtime.h>
#include <vector_types.h>

#include "gqd_type.h"         /* gdtq-0.0.2: single header declares all gdd/gtd/gqd + gds/gts/gqs types */
#include "gddlinear.h"        /* GDDVector, GQDVector and bridges */
#include "gtdlinear.h"        /* GTDVector and bridges (triple-double) */
#include "gdslinear.h"        /* GDSVector, GQSVector and bridges */
#include "gtslinear.h"        /* GTSVector and bridges (triple-single) */
#include "ddlinear.h"         /* DDVector, init_ddvector, cmul_ddvector */
#include "tdlinear.h"
#include "qdlinear.h"
#include "dslinear.h"
#include "tslinear.h"
#include "qslinear.h"
/* (bncmatmul.h intentionally NOT included: it pulls bncsparse.h -> single/double
 * complex header read_test_linear_eq_cdd "C"-linkage clash.) */
#include <omp.h>

extern "C" {
void _bncomp_cmul_ddvector(DDVector, double[], DDVector);
void _bncomp_cmul_tdvector(TDVector, double[], TDVector);
void _bncomp_cmul_qdvector(QDVector, double[], QDVector);
void _bncomp_cmul_dsvector(DSVector, float[],  DSVector);
void _bncomp_cmul_tsvector(TSVector, float[],  TSVector);
void _bncomp_cmul_qsvector(QSVector, float[],  QSVector);
}

static FILE *g_csv = nullptr;
static void emit_csv(const char *op, const char *prec, long dim, long nnz,
                     double tcpu, double tgpu, const char *cpu_kind, double relerr){
    if(!g_csv) return;
    fprintf(g_csv,"RESULT,%s,%s,%ld,%ld,%.9g,%.9g,%s,%.6e\n",op,prec,dim,nnz,tcpu,tgpu,cpu_kind,relerr);
    fflush(g_csv);
}

/*--------------------------------------------------------------*/
/* timing helpers                                               */
/*--------------------------------------------------------------*/
using clk = std::chrono::high_resolution_clock;

static double secs_since(clk::time_point t0)
{
    auto t1 = clk::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

template<typename Fn>
static double measure_min_gpu(int reps, Fn fn)
{
    double best = 1e30;
    for (int i = 0; i < reps; ++i) {
        auto t0 = clk::now();
        fn();
        cudaDeviceSynchronize();
        double s = secs_since(t0);
        if (s < best) best = s;
    }
    return best;
}

template<typename Fn>
static double measure_min_cpu(int reps, Fn fn)
{
    double best = 1e30;
    for (int i = 0; i < reps; ++i) {
        auto t0 = clk::now();
        fn();
        double s = secs_since(t0);
        if (s < best) best = s;
    }
    return best;
}

/*--------------------------------------------------------------*/
/* options                                                      */
/*--------------------------------------------------------------*/
struct Opts {
    int  reps          = 5;
    int  blocks        = 128;
    int  threads       = 128;
    std::vector<long> sizes;
};

/*============================================================================
 *  Per-precision benchmark.
 *
 *  Token-pasting macro generates one function per precision tag.  The
 *  parameters are:
 *    P    = lower-case precision tag (dd / td / qd / ds / ts / qs)
 *    GP   = upper-case GPU type prefix  (GDD / GTD / GQD / GDS / GTS / GQS)
 *    V    = CPU vector type      (DDVector / ... / QSVector)
 *    GV   = GPU vector type      (GDDVector / ... / GQSVector)
 *    GR   = GPU scalar real type (gdd_real = double2, gtd_real = double3, ...)
 *    S    = limb scalar type     (double for dd/td/qd, float for ds/ts/qs)
 *    SZ   = limb count           (DDSIZE / ... / QSSIZE)
 *    VSFX = vector setter suffix
 *             d  : dd / td / qd / ds (historical ds setter is _d)
 *             f  : ts / qs
 *==========================================================================*/
#define DEFINE_GPU_CMUL_VECTOR_BENCH(P, GP, V, GV, GR, S, SZ, VSFX)                  \
static void bench_g##P(long dim, const Opts &opts,                                   \
                       double *t_cpu, double *t_gpu, double *rel_err)                \
{                                                                                    \
    /* ---- CPU side ---- */                                                         \
    V a_cpu  = init_##P##vector(dim);                                                \
    V cs_cpu = init_##P##vector(dim);    /* CPU reference (serial)        */         \
    V cp_cpu = init_##P##vector(dim);    /* GPU output, copied back here  */         \
                                                                                     \
    /* high-limb-only scalar; lower limbs left at 0 */                               \
    S val[SZ];                                                                       \
    for (int k = 0; k < (int)(SZ); ++k) val[k] = (S)0;                               \
    val[0] = (S)0.7283105;                                                           \
                                                                                     \
    srand(20260513u);                                                                \
    for (long i = 0; i < dim; ++i) {                                                 \
        set_##P##vector_i_##VSFX(a_cpu, i,                                           \
            (double)rand() / (double)RAND_MAX * 2.0 - 1.0);                          \
    }                                                                                \
                                                                                     \
    /* ---- CPU baseline (best of opts.reps) ---- */                                 \
    _bncomp_cmul_##P##vector(cs_cpu, val, a_cpu);   /* warm-up */                            \
    *t_cpu = measure_min_cpu(opts.reps, [&]() {                                      \
        _bncomp_cmul_##P##vector(cs_cpu, val, a_cpu);                                        \
    });                                                                              \
                                                                                     \
    /* ---- GPU side: allocate, copy, build gprec_real scalar ---- */                \
    GV a_dev = init_g##P##vector_dev(dim);                                           \
    GV c_dev = init_g##P##vector_dev(dim);                                           \
    subst_g##P##vector_dev_##P##vec(a_dev, a_cpu);                                   \
                                                                                     \
    GR gval;                                                                         \
    memset(&gval, 0, sizeof(gval));                                                  \
    /* gprec_real has 2..4 scalar limbs in .x .y .z .w; copy as many as exist */     \
    {                                                                                \
        S *limbs = reinterpret_cast<S *>(&gval);                                     \
        for (int k = 0; k < (int)(SZ); ++k) limbs[k] = val[k];                       \
    }                                                                                \
                                                                                     \
    /* ---- GPU warm-up + timed loop ---- */                                         \
    cmul_g##P##vector_dev(c_dev, gval, a_dev, opts.blocks, opts.threads);            \
    cudaDeviceSynchronize();                                                         \
    *t_gpu = measure_min_gpu(opts.reps, [&]() {                                      \
        cmul_g##P##vector_dev(c_dev, gval, a_dev, opts.blocks, opts.threads);        \
    });                                                                              \
                                                                                     \
    /* ---- copy GPU result back and compare to CPU reference ---- */                \
    subst_##P##vector_g##P##vec_dev(cp_cpu, c_dev);                                  \
    {                                                                                \
        S ns[SZ], nd[SZ];                                                            \
        norm2_##P##vector(ns, cs_cpu);                                               \
        sub_##P##vector(cp_cpu, cs_cpu, cp_cpu);    /* cp := cs - cp */              \
        norm2_##P##vector(nd, cp_cpu);                                               \
        double n_s = (double)ns[0];                                                  \
        double n_d = (double)nd[0];                                                  \
        *rel_err = (n_s != 0.0) ? (n_d / n_s) : n_d;                                 \
    }                                                                                \
                                                                                     \
    free_g##P##vector_dev(a_dev);                                                    \
    free_g##P##vector_dev(c_dev);                                                    \
    free_##P##vector(a_cpu);                                                         \
    free_##P##vector(cs_cpu);                                                        \
    free_##P##vector(cp_cpu);                                                        \
}

DEFINE_GPU_CMUL_VECTOR_BENCH(dd, GDD, DDVector, GDDVector, gdd_real, double, DDSIZE, d)
DEFINE_GPU_CMUL_VECTOR_BENCH(td, GTD, TDVector, GTDVector, gtd_real, double, TDSIZE, d)
DEFINE_GPU_CMUL_VECTOR_BENCH(qd, GQD, QDVector, GQDVector, gqd_real, double, QDSIZE, d)
DEFINE_GPU_CMUL_VECTOR_BENCH(ds, GDS, DSVector, GDSVector, gds_real, float,  DSSIZE, d) /* ds setter _d (takes float) */
DEFINE_GPU_CMUL_VECTOR_BENCH(ts, GTS, TSVector, GTSVector, gts_real, float,  TSSIZE, f)
DEFINE_GPU_CMUL_VECTOR_BENCH(qs, GQS, QSVector, GQSVector, gqs_real, float,  QSSIZE, f)

/*--------------------------------------------------------------*/
/* arg parsing                                                  */
/*--------------------------------------------------------------*/
static void parse_sizes(const char *s, std::vector<long> &out)
{
    out.clear();
    const char *p = s;
    while (*p) {
        char *end = nullptr;
        long v = strtol(p, &end, 10);
        if (end == p || v <= 0) break;
        out.push_back(v);
        p = end;
        while (*p == ',' || *p == ' ') ++p;
    }
}

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s [options]\n"
        "  --reps R          timed reps per measurement (default 5)\n"
        "  --blocks B        CUDA blocks per grid       (default 128)\n"
        "  --threads T       CUDA threads per block     (default 128)\n"
        "  --sizes N1,N2,... override default size list\n"
        "                    (default 4096,16384,65536,262144,1048576,4194304)\n",
        prog);
}

/*--------------------------------------------------------------*/
/* main                                                         */
/*--------------------------------------------------------------*/
int main(int argc, char **argv)
{
    Opts opts;
    opts.sizes = {4096L, 16384L, 65536L, 262144L, 1048576L, 4194304L};

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--reps") && i + 1 < argc)
            opts.reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--blocks") && i + 1 < argc)
            opts.blocks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc)
            opts.threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sizes") && i + 1 < argc)
            parse_sizes(argv[++i], opts.sizes);
        else { usage(argv[0]); return 1; }
    }

    /* qd/dd CPU paths need the x87 control word fixed up for round-to-double. */
    unsigned int cw;
    fpu_fix_start(&cw);

    const char *env_th = getenv("OMP_NUM_THREADS");
    int nthreads = env_th ? atoi(env_th) : omp_get_max_threads();
    if (nthreads < 1) nthreads = omp_get_max_threads();
    omp_set_num_threads(nthreads);
    g_csv = stdout;
    printf(" CPU baseline: OpenMP _bncomp_cmul (%d threads)\n", nthreads);

    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    printf("=========================================================\n");
    printf(" gdtqs + bncmatmul GPU cmul_<P>vector benchmark\n");
    printf("=========================================================\n");
    printf(" device     : %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf(" reps/meas  : %d (min wall-clock)\n", opts.reps);
    printf(" CUDA grid  : %d blocks x %d threads\n", opts.blocks, opts.threads);
    printf(" sizes      :");
    for (long N : opts.sizes) printf(" %ld", N);
    printf("\n");
    printf("=========================================================\n");

    struct prec_entry {
        const char *name;
        const char *desc;
        void (*fn)(long, const Opts &, double *, double *, double *);
        void (*start)(int);
        void (*end)(void);
    };
    const prec_entry PRECS[] = {
        { "gdd", "double-double  ~31 digits", bench_gdd, GDDStart, GDDEnd },
        { "gtd", "triple-double  ~47 digits", bench_gtd, GTDStart, GTDEnd },
        { "gqd", "quad-double    ~62 digits", bench_gqd, GQDStart, GQDEnd },
        { "gds", "double-single  ~14 digits", bench_gds, GDSStart, GDSEnd },
        { "gts", "triple-single  ~21 digits", bench_gts, GTSStart, GTSEnd },
        { "gqs", "quad-single    ~28 digits", bench_gqs, GQSStart, GQSEnd },
    };
    const int N_PRECS = (int)(sizeof(PRECS) / sizeof(PRECS[0]));

    for (int p = 0; p < N_PRECS; ++p) {
        printf("\n##### %s (%s) #####\n", PRECS[p].name, PRECS[p].desc);
        printf("  %-4s %-9s   %14s   %14s   %9s   %s\n",
               "prec", "N", "CPU [s]", "GPU [s]", "speedup", "||c_cpu - c_gpu||_2 / ||c_cpu||_2");
        PRECS[p].start(dev);
        for (long N : opts.sizes) {
            double tcpu = 0, tgpu = 0, rel_err = 0;
            PRECS[p].fn(N, opts, &tcpu, &tgpu, &rel_err);
            double sp = (tgpu > 0.0) ? tcpu / tgpu : 0.0;
            printf("  %-4s %-9ld   %14.6f   %14.6f   x%-8.2f  %.3e\n",
                   PRECS[p].name, N, tcpu, tgpu, sp, rel_err);
            emit_csv("axpy", PRECS[p].name + 1, N, 0, tcpu, tgpu, "omp", rel_err);
        }
        PRECS[p].end();
    }

    fpu_fix_end(&cw);

    printf("\n=========================================================\n");
    printf(" done.\n");
    printf("=========================================================\n");
    return 0;
}
