/*****************************************************************
 * bench_gdtqs_matvec.cu
 * ----------------------------------------------------------------
 * Performance benchmark for the GPU matrix-times-vector kernels
 *   mul_g{dd,td,qd,ds,ts,qs}matrix_g{dd,td,qd,ds,ts,qs}vec
 * across all 6 multi-precision layers (double-based: gdd/gtd/gqd,
 * float-based: gds/gts/gqs).
 *
 * For each precision and matrix size N (matrix is N x N, vector N x 1)
 *   - CPU serial : mul_<prec>matrix_<prec>vec(v, A, b)
 *   - GPU kernel : mul_g<prec>matrix_g<prec>vec(v_d, A_d, b_d, ...)
 * are timed (minimum wall-clock over `reps` reps after one warm-up),
 * the GPU output is compared to the CPU output (Euclidean norm), and
 * the speedup + relative error are printed.
 *
 * Build:  make -C src/cuda bench_gdtqs_matvec
 *
 * Usage:  ./bench_gdtqs_matvec [options]
 *   --reps R          timed reps per measurement                 (default 3)
 *   --blocks B        CUDA blocks per grid                       (default 128)
 *   --threads T       CUDA threads per block                     (default 128)
 *   --max-cpu N       skip CPU baseline for sizes > N            (default: no limit)
 *   --no-cpu          skip CPU baseline entirely
 *   --sizes N1,N2,... override default size list
 *                     (default 128,256,512,1024,2048,4096)
 *
 * Linking requires libbncmatmul-X.Y.{a,so} in addition to libbncmm_cuda.a.
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

#include "gqd_type.h"         /* gdtq-0.0.2: declares all gdd/gtd/gqd + gds/gts/gqs types */
#include "gddlinear.h"
#include "gtdlinear.h"
#include "gdslinear.h"
#include "gtslinear.h"
#include "ddlinear.h"
#include "tdlinear.h"
#include "qdlinear.h"
#include "dslinear.h"
#include "tslinear.h"
#include "qslinear.h"
#include "bncmatmul.h"

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
    int  reps     = 3;
    int  blocks   = 128;
    int  threads  = 128;
    bool no_cpu   = false;
    long max_cpu  = -1;
    std::vector<long> sizes;
};

/*============================================================================
 *  Per-precision benchmark.
 *
 *  Token-pasting macro generates one bench function per precision tag.
 *  Parameters:
 *    P    = lower-case precision tag (dd / td / qd / ds / ts / qs)
 *    M    = CPU matrix type  (DDMatrix / ... / QSMatrix)
 *    V    = CPU vector type  (DDVector / ... / QSVector)
 *    GM   = GPU matrix type  (GDDMatrix / ... / GQSMatrix)
 *    GV   = GPU vector type  (GDDVector / ... / GQSVector)
 *    S    = limb scalar type (double / float)
 *    SZ   = limb count       (DDSIZE / ... / QSSIZE)
 *    MSFX = matrix setter suffix ('d' for dd/td/qd, 'f' for ds/ts/qs)
 *    VSFX = vector setter suffix ('d' for dd/td/qd/ds, 'f' for ts/qs)
 *==========================================================================*/
#define DEFINE_GPU_MATVEC_BENCH(P, M, V, GM, GV, S, SZ, MSFX, VSFX)                  \
static void bench_g##P(long dim, const Opts &opts,                                   \
                       double *t_cpu, double *t_gpu, double *rel_err,                \
                       bool *cpu_ran)                                                \
{                                                                                    \
    M A_cpu  = init_##P##matrix(dim, dim);                                           \
    V b_cpu  = init_##P##vector(dim);                                                \
    V cs_cpu = init_##P##vector(dim);                                                \
    V cp_cpu = init_##P##vector(dim);                                                \
                                                                                     \
    srand(20260513u);                                                                \
    for (long i = 0; i < dim; ++i) {                                                 \
        set_##P##vector_i_##VSFX(b_cpu, i,                                           \
            (double)rand() / (double)RAND_MAX * 2.0 - 1.0);                          \
        for (long j = 0; j < dim; ++j) {                                             \
            set_##P##matrix_ij_##MSFX(A_cpu, i, j,                                   \
                (double)rand() / (double)RAND_MAX * 2.0 - 1.0);                      \
        }                                                                            \
    }                                                                                \
                                                                                     \
    /* ---- CPU baseline (skipped if --no-cpu or N > max-cpu) ---- */                \
    bool run_cpu = !opts.no_cpu && (opts.max_cpu < 0 || dim <= opts.max_cpu);        \
    *cpu_ran = run_cpu;                                                              \
    *t_cpu = 0.0;                                                                    \
    if (run_cpu) {                                                                   \
        mul_##P##matrix_##P##vec(cs_cpu, A_cpu, b_cpu);   /* warm-up */              \
        *t_cpu = measure_min_cpu(opts.reps, [&]() {                                  \
            mul_##P##matrix_##P##vec(cs_cpu, A_cpu, b_cpu);                          \
        });                                                                          \
    }                                                                                \
                                                                                     \
    /* ---- GPU side ---- */                                                         \
    GM A_dev = init_g##P##matrix_dev(dim, dim);                                      \
    GV b_dev = init_g##P##vector_dev(dim);                                           \
    GV c_dev = init_g##P##vector_dev(dim);                                           \
    subst_g##P##matrix_dev_##P##mat(A_dev, A_cpu);                                   \
    subst_g##P##vector_dev_##P##vec(b_dev, b_cpu);                                   \
                                                                                     \
    mul_g##P##matrix_g##P##vec(c_dev, A_dev, b_dev, opts.blocks, opts.threads); /* warm-up */ \
    cudaDeviceSynchronize();                                                         \
    *t_gpu = measure_min_gpu(opts.reps, [&]() {                                      \
        mul_g##P##matrix_g##P##vec(c_dev, A_dev, b_dev, opts.blocks, opts.threads);  \
    });                                                                              \
                                                                                     \
    /* ---- compare GPU result against CPU reference (only if CPU ran) ---- */       \
    *rel_err = 0.0;                                                                  \
    if (run_cpu) {                                                                   \
        subst_##P##vector_g##P##vec_dev(cp_cpu, c_dev);                              \
        S ns[SZ], nd[SZ];                                                            \
        norm2_##P##vector(ns, cs_cpu);                                               \
        sub_##P##vector(cp_cpu, cs_cpu, cp_cpu);                                     \
        norm2_##P##vector(nd, cp_cpu);                                               \
        double n_s = (double)ns[0];                                                  \
        double n_d = (double)nd[0];                                                  \
        *rel_err = (n_s != 0.0) ? (n_d / n_s) : n_d;                                 \
    }                                                                                \
                                                                                     \
    free_g##P##matrix_dev(A_dev);                                                    \
    free_g##P##vector_dev(b_dev);                                                    \
    free_g##P##vector_dev(c_dev);                                                    \
    free_##P##matrix(A_cpu);                                                         \
    free_##P##vector(b_cpu);                                                         \
    free_##P##vector(cs_cpu);                                                        \
    free_##P##vector(cp_cpu);                                                        \
}

DEFINE_GPU_MATVEC_BENCH(dd, DDMatrix, DDVector, GDDMatrix, GDDVector, double, DDSIZE, d, d)
DEFINE_GPU_MATVEC_BENCH(td, TDMatrix, TDVector, GTDMatrix, GTDVector, double, TDSIZE, d, d)
DEFINE_GPU_MATVEC_BENCH(qd, QDMatrix, QDVector, GQDMatrix, GQDVector, double, QDSIZE, d, d)
DEFINE_GPU_MATVEC_BENCH(ds, DSMatrix, DSVector, GDSMatrix, GDSVector, float,  DSSIZE, f, d) /* ds vector setter _d */
DEFINE_GPU_MATVEC_BENCH(ts, TSMatrix, TSVector, GTSMatrix, GTSVector, float,  TSSIZE, f, f)
DEFINE_GPU_MATVEC_BENCH(qs, QSMatrix, QSVector, GQSMatrix, GQSVector, float,  QSSIZE, f, f)

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
        "  --reps R          timed reps per measurement       (default 3)\n"
        "  --blocks B        CUDA blocks per grid             (default 128)\n"
        "  --threads T       CUDA threads per block           (default 128)\n"
        "  --max-cpu N       skip CPU baseline for sizes > N  (default: no limit)\n"
        "  --no-cpu          skip CPU baseline entirely\n"
        "  --sizes N1,N2,... override default size list\n"
        "                    (default 128,256,512,1024,2048,4096)\n",
        prog);
}

/*--------------------------------------------------------------*/
/* main                                                         */
/*--------------------------------------------------------------*/
int main(int argc, char **argv)
{
    Opts opts;
    opts.sizes = {128L, 256L, 512L, 1024L, 2048L, 4096L};

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--reps") && i + 1 < argc)
            opts.reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--blocks") && i + 1 < argc)
            opts.blocks = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc)
            opts.threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-cpu") && i + 1 < argc)
            opts.max_cpu = atol(argv[++i]);
        else if (!strcmp(argv[i], "--no-cpu"))
            opts.no_cpu = true;
        else if (!strcmp(argv[i], "--sizes") && i + 1 < argc)
            parse_sizes(argv[++i], opts.sizes);
        else { usage(argv[0]); return 1; }
    }

    /* qd/dd CPU paths need the x87 control word fixed up for round-to-double. */
    unsigned int cw;
    fpu_fix_start(&cw);

    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    printf("=========================================================\n");
    printf(" gdtqs + bncmatmul GPU mul_<P>matrix_<P>vec benchmark\n");
    printf("=========================================================\n");
    printf(" device     : %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf(" reps/meas  : %d (min wall-clock)\n", opts.reps);
    printf(" CUDA grid  : %d blocks x %d threads\n", opts.blocks, opts.threads);
    printf(" CPU ref    : ");
    if (opts.no_cpu)              printf("disabled (--no-cpu)\n");
    else if (opts.max_cpu < 0)    printf("enabled for ALL sizes\n");
    else                          printf("enabled for N <= %ld\n", opts.max_cpu);
    printf(" sizes      :");
    for (long N : opts.sizes) printf(" %ld", N);
    printf("\n");
    printf("=========================================================\n");

    struct prec_entry {
        const char *name;
        const char *desc;
        void (*fn)(long, const Opts &, double *, double *, double *, bool *);
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
               "prec", "N", "CPU [s]", "GPU [s]", "speedup", "||v_cpu - v_gpu||_2 / ||v_cpu||_2");
        PRECS[p].start(dev);
        for (long N : opts.sizes) {
            double tcpu = 0, tgpu = 0, rel_err = 0;
            bool cpu_ran = false;
            PRECS[p].fn(N, opts, &tcpu, &tgpu, &rel_err, &cpu_ran);
            if (cpu_ran) {
                double sp = (tgpu > 0.0) ? tcpu / tgpu : 0.0;
                printf("  %-4s %-9ld   %14.6f   %14.6f   x%-8.2f  %.3e\n",
                       PRECS[p].name, N, tcpu, tgpu, sp, rel_err);
            } else {
                printf("  %-4s %-9ld   %14s   %14.6f   %9s   %s\n",
                       PRECS[p].name, N, "(skipped)", tgpu, "-", "(no CPU ref)");
            }
        }
        PRECS[p].end();
    }

    fpu_fix_end(&cw);

    printf("\n=========================================================\n");
    printf(" done.\n");
    printf("=========================================================\n");
    return 0;
}
