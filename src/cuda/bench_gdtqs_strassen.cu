/*****************************************************************
 * bench_gdtqs_strassen.cu
 * ----------------------------------------------------------------
 * Performance benchmark for the GPU GDS / GTS / GQS matmul paths
 * (float-based double-single / triple-single / quad-single).
 *
 * For each problem size N and each precision (GDS, GTS, GQS):
 *   - GPU naive GEMM    : mul_g{ds,ts,qs}matrix_dev             (always)
 *   - GPU block GEMM    : _bncuda_mul_g{ds,ts,qs}matrix_block   (always)
 *   - GPU Strassen      : _bncuda_mul_g{ds,ts,qs}matrix_strassen(default on)
 *   - CPU Strassen      : mul_{ds,ts,qs}matrix_strassen         (default on)
 *
 * Each measurement is repeated `--reps` times after one warmup; the
 * minimum wall-clock time is reported (less sensitive to OS noise).
 *
 * Build:  make bench_gdtqs_strassen
 *
 * Usage:  ./bench_gdtqs_strassen [options]
 *   --no-strassen     skip GPU Strassen (on by default; slow)
 *   --no-cpu          skip CPU Strassen entirely (on by default; slow)
 *   --max-cpu N       skip CPU for sizes > N (default: no limit)
 *   --reps R          timed reps per measurement (default 3)
 *   --block-min D     min_dim for GPU block GEMM / GPU Strassen / CPU Strassen cutoff (default 16)
 *   --sizes N1,N2,... override default size list (default 64,128,256,512)
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

#include "rds.h"          /* dsfloat / tsfloat / qsfloat + c_{ds,ts,qs}_copy_d */

#include <cuda_runtime.h>
#include <vector_types.h>

#include "gqd_type.h"         /* gdtq-0.0.2: single header declares all gdd/gtd/gqd + gds/gts/gqs types */
#include "gdslinear.h"        /* GDSMatrix, GQSMatrix and bridges */
#include "gtslinear.h"        /* GTSMatrix and bridges (triple-double) */
#include "dslinear.h"         /* DSMatrix, init_dsmatrix, mul_dsmatrix(_strassen) */
#include "tslinear.h"         /* TSMatrix, init_tsmatrix, mul_tsmatrix(_strassen) */
#include "qslinear.h"         /* QSMatrix, init_qsmatrix, mul_qsmatrix(_strassen) */
#include "matmul_strassen.h"  /* CPU + GPU Strassen prototypes */

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
static double measure_min(int reps, Fn fn)
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
static double measure_cpu_min(int reps, Fn fn)
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
/* random fill into the SoA DSMatrix / QSMatrix used by         */
/* bncmatmul's CPU paths.                                       */
/*--------------------------------------------------------------*/
static void fill_random_ds(DSMatrix A, DSMatrix B, long N, unsigned seed)
{
    srand(seed);
    for (long i = 0; i < N; ++i) {
        for (long j = 0; j < N; ++j) {
            double a = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
            double b = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
            dsfloat va, vb;
            c_ds_copy_d(a, va.val);
            c_ds_copy_d(b, vb.val);
            A->element[0][i * N + j] = va.val[0];
            A->element[1][i * N + j] = va.val[1];
            B->element[0][i * N + j] = vb.val[0];
            B->element[1][i * N + j] = vb.val[1];
        }
    }
}

static void fill_random_ts(TSMatrix A, TSMatrix B, long N, unsigned seed)
{
    srand(seed);
    for (long i = 0; i < N; ++i) {
        for (long j = 0; j < N; ++j) {
            double a = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
            double b = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
            tsfloat va, vb;
            c_ts_copy_d(a, va.val);
            c_ts_copy_d(b, vb.val);
            A->element[0][i * N + j] = va.val[0];
            A->element[1][i * N + j] = va.val[1];
            A->element[2][i * N + j] = va.val[2];
            B->element[0][i * N + j] = vb.val[0];
            B->element[1][i * N + j] = vb.val[1];
            B->element[2][i * N + j] = vb.val[2];
        }
    }
}

static void fill_random_qs(QSMatrix A, QSMatrix B, long N, unsigned seed)
{
    srand(seed);
    for (long i = 0; i < N; ++i) {
        for (long j = 0; j < N; ++j) {
            double a = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
            double b = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
            qsfloat va, vb;
            c_qs_copy_d(a, va.val);
            c_qs_copy_d(b, vb.val);
            A->element[0][i * N + j] = va.val[0];
            A->element[1][i * N + j] = va.val[1];
            A->element[2][i * N + j] = va.val[2];
            A->element[3][i * N + j] = va.val[3];
            B->element[0][i * N + j] = vb.val[0];
            B->element[1][i * N + j] = vb.val[1];
            B->element[2][i * N + j] = vb.val[2];
            B->element[3][i * N + j] = vb.val[3];
        }
    }
}

/*--------------------------------------------------------------*/
/* run options                                                  */
/*--------------------------------------------------------------*/
struct Opts {
    bool run_strassen = true;   /* on by default; pass --no-strassen to skip */
    bool run_block    = true;   /* on by default; pass --no-block to skip    */
    bool no_cpu       = false;  /* on by default; pass --no-cpu to skip      */
    long max_cpu      = -1;     /* -1 means no upper limit                   */
    int  reps         = 3;
    long block_min    = 16;     /* used for both GPU block min and CPU Strassen cutoff */
    std::vector<long> sizes;
};

/*--------------------------------------------------------------*/
/* GDS benchmark                                                */
/*--------------------------------------------------------------*/
static void bench_gds(long N, const Opts &opts)
{
    /* Host-side bncmatmul matrices (SoA layout). */
    DSMatrix A_cpu = init_dsmatrix(N, N);
    DSMatrix B_cpu = init_dsmatrix(N, N);
    DSMatrix C_cpu = init_dsmatrix(N, N);
    fill_random_ds(A_cpu, B_cpu, N, /*seed=*/12345);

    /* GPU side: copy via the bncmatmul SoA-aware bridges. */
    GDSMatrix A_d = init_gdsmatrix_dev(N, N);
    GDSMatrix B_d = init_gdsmatrix_dev(N, N);
    GDSMatrix C_d = init_gdsmatrix_dev(N, N);
    subst_gdsmatrix_dev_dsmat(A_d, A_cpu);
    subst_gdsmatrix_dev_dsmat(B_d, B_cpu);

    /* --- GPU naive GEMM --- */
    mul_gdsmatrix_dev(C_d, A_d, B_d, 128, 128);
    cudaDeviceSynchronize();
    double t_gpu_naive = measure_min(opts.reps, [&]() {
        mul_gdsmatrix_dev(C_d, A_d, B_d, 128, 128);
    });
    printf("  N=%-5ld  GPU naive    : %10.3f ms\n",
           N, t_gpu_naive * 1e3);

    /* --- GPU block GEMM (default on; --no-block to skip) --- */
    if (opts.run_block) {
        _bncuda_mul_gdsmatrix_block(C_d, A_d, B_d, opts.block_min, 128, 128);
        cudaDeviceSynchronize();
        double t_gpu_block = measure_min(opts.reps, [&]() {
            _bncuda_mul_gdsmatrix_block(C_d, A_d, B_d, opts.block_min, 128, 128);
        });
        printf("  N=%-5ld  GPU block (min=%-3ld): %10.3f ms  (%.2fx vs naive)\n",
               N, opts.block_min, t_gpu_block * 1e3,
               t_gpu_naive / t_gpu_block);
    }

    /* --- GPU Strassen, cutoff sweep (default on) --- */
    if (opts.run_strassen) {
        long cutoffs[] = {16, 32, 64, 128};
        for (long cutoff : cutoffs) {
            if (cutoff > N / 2) continue;
            _bncuda_mul_gdsmatrix_strassen(C_d, A_d, B_d,
                                            cutoff, 128, 128);
            cudaDeviceSynchronize();
            double t = measure_min(opts.reps, [&]() {
                _bncuda_mul_gdsmatrix_strassen(C_d, A_d, B_d,
                                                cutoff, 128, 128);
            });
            printf("  N=%-5ld  GPU Strassen (cutoff=%-3ld): %10.3f ms"
                   "  (%.2fx vs naive)\n",
                   N, cutoff, t * 1e3, t_gpu_naive / t);
        }
    }

    /* --- CPU Strassen (bncmatmul) --- */
    bool run_cpu = !opts.no_cpu &&
                   (opts.max_cpu < 0 || N <= opts.max_cpu);
    if (run_cpu) {
        double t_cpu = measure_cpu_min(1, [&]() {
            mul_dsmatrix_strassen(C_cpu, A_cpu, B_cpu, opts.block_min);
        });
        printf("  N=%-5ld  CPU Strassen (cutoff=%-3ld): %10.3f ms"
               "  (%.2fx vs GPU naive)\n",
               N, opts.block_min, t_cpu * 1e3, t_cpu / t_gpu_naive);
    } else {
        printf("  N=%-5ld  CPU Strassen : (skipped)\n", N);
    }

    free_gdsmatrix_dev(A_d);
    free_gdsmatrix_dev(B_d);
    free_gdsmatrix_dev(C_d);
    free_dsmatrix(A_cpu);
    free_dsmatrix(B_cpu);
    free_dsmatrix(C_cpu);
}

/*--------------------------------------------------------------*/
/* GTS benchmark                                                */
/*--------------------------------------------------------------*/
static void bench_gts(long N, const Opts &opts)
{
    TSMatrix A_cpu = init_tsmatrix(N, N);
    TSMatrix B_cpu = init_tsmatrix(N, N);
    TSMatrix C_cpu = init_tsmatrix(N, N);
    fill_random_ts(A_cpu, B_cpu, N, /*seed=*/24680);

    GTSMatrix A_d = init_gtsmatrix_dev(N, N);
    GTSMatrix B_d = init_gtsmatrix_dev(N, N);
    GTSMatrix C_d = init_gtsmatrix_dev(N, N);
    subst_gtsmatrix_dev_tsmat(A_d, A_cpu);
    subst_gtsmatrix_dev_tsmat(B_d, B_cpu);

    /* --- GPU naive GEMM --- */
    mul_gtsmatrix_dev(C_d, A_d, B_d, 128, 128);
    cudaDeviceSynchronize();
    double t_gpu_naive = measure_min(opts.reps, [&]() {
        mul_gtsmatrix_dev(C_d, A_d, B_d, 128, 128);
    });
    printf("  N=%-5ld  GPU naive    : %10.3f ms\n",
           N, t_gpu_naive * 1e3);

    /* --- GPU block GEMM (default on; --no-block to skip) --- */
    if (opts.run_block) {
        _bncuda_mul_gtsmatrix_block(C_d, A_d, B_d, opts.block_min, 128, 128);
        cudaDeviceSynchronize();
        double t_gpu_block = measure_min(opts.reps, [&]() {
            _bncuda_mul_gtsmatrix_block(C_d, A_d, B_d, opts.block_min, 128, 128);
        });
        printf("  N=%-5ld  GPU block (min=%-3ld): %10.3f ms  (%.2fx vs naive)\n",
               N, opts.block_min, t_gpu_block * 1e3,
               t_gpu_naive / t_gpu_block);
    }

    /* --- GPU Strassen, cutoff sweep (default on) --- */
    if (opts.run_strassen) {
        long cutoffs[] = {16, 32, 64, 128};
        for (long cutoff : cutoffs) {
            if (cutoff > N / 2) continue;
            _bncuda_mul_gtsmatrix_strassen(C_d, A_d, B_d,
                                            cutoff, 128, 128);
            cudaDeviceSynchronize();
            double t = measure_min(opts.reps, [&]() {
                _bncuda_mul_gtsmatrix_strassen(C_d, A_d, B_d,
                                                cutoff, 128, 128);
            });
            printf("  N=%-5ld  GPU Strassen (cutoff=%-3ld): %10.3f ms"
                   "  (%.2fx vs naive)\n",
                   N, cutoff, t * 1e3, t_gpu_naive / t);
        }
    }

    /* --- CPU Strassen (bncmatmul) --- */
    bool run_cpu = !opts.no_cpu &&
                   (opts.max_cpu < 0 || N <= opts.max_cpu);
    if (run_cpu) {
        double t_cpu = measure_cpu_min(1, [&]() {
            mul_tsmatrix_strassen(C_cpu, A_cpu, B_cpu, opts.block_min);
        });
        printf("  N=%-5ld  CPU Strassen (cutoff=%-3ld): %10.3f ms"
               "  (%.2fx vs GPU naive)\n",
               N, opts.block_min, t_cpu * 1e3, t_cpu / t_gpu_naive);
    } else {
        printf("  N=%-5ld  CPU Strassen : (skipped)\n", N);
    }

    free_gtsmatrix_dev(A_d);
    free_gtsmatrix_dev(B_d);
    free_gtsmatrix_dev(C_d);
    free_tsmatrix(A_cpu);
    free_tsmatrix(B_cpu);
    free_tsmatrix(C_cpu);
}

/*--------------------------------------------------------------*/
/* GQS benchmark                                                */
/*--------------------------------------------------------------*/
static void bench_gqs(long N, const Opts &opts)
{
    QSMatrix A_cpu = init_qsmatrix(N, N);
    QSMatrix B_cpu = init_qsmatrix(N, N);
    QSMatrix C_cpu = init_qsmatrix(N, N);
    fill_random_qs(A_cpu, B_cpu, N, /*seed=*/67890);

    GQSMatrix A_d = init_gqsmatrix_dev(N, N);
    GQSMatrix B_d = init_gqsmatrix_dev(N, N);
    GQSMatrix C_d = init_gqsmatrix_dev(N, N);
    subst_gqsmatrix_dev_qsmat(A_d, A_cpu);
    subst_gqsmatrix_dev_qsmat(B_d, B_cpu);

    /* --- GPU naive GEMM --- */
    mul_gqsmatrix_dev(C_d, A_d, B_d, 128, 128);
    cudaDeviceSynchronize();
    double t_gpu_naive = measure_min(opts.reps, [&]() {
        mul_gqsmatrix_dev(C_d, A_d, B_d, 128, 128);
    });
    printf("  N=%-5ld  GPU naive    : %10.3f ms\n",
           N, t_gpu_naive * 1e3);

    /* --- GPU block GEMM (default on; --no-block to skip) --- */
    if (opts.run_block) {
        _bncuda_mul_gqsmatrix_block(C_d, A_d, B_d, opts.block_min, 128, 128);
        cudaDeviceSynchronize();
        double t_gpu_block = measure_min(opts.reps, [&]() {
            _bncuda_mul_gqsmatrix_block(C_d, A_d, B_d, opts.block_min, 128, 128);
        });
        printf("  N=%-5ld  GPU block (min=%-3ld): %10.3f ms  (%.2fx vs naive)\n",
               N, opts.block_min, t_gpu_block * 1e3,
               t_gpu_naive / t_gpu_block);
    }

    /* --- GPU Strassen, cutoff sweep (default on) --- */
    if (opts.run_strassen) {
        long cutoffs[] = {16, 32, 64, 128};
        for (long cutoff : cutoffs) {
            if (cutoff > N / 2) continue;
            _bncuda_mul_gqsmatrix_strassen(C_d, A_d, B_d,
                                            cutoff, 128, 128);
            cudaDeviceSynchronize();
            double t = measure_min(opts.reps, [&]() {
                _bncuda_mul_gqsmatrix_strassen(C_d, A_d, B_d,
                                                cutoff, 128, 128);
            });
            printf("  N=%-5ld  GPU Strassen (cutoff=%-3ld): %10.3f ms"
                   "  (%.2fx vs naive)\n",
                   N, cutoff, t * 1e3, t_gpu_naive / t);
        }
    }

    /* --- CPU Strassen (bncmatmul) --- */
    bool run_cpu = !opts.no_cpu &&
                   (opts.max_cpu < 0 || N <= opts.max_cpu);
    if (run_cpu) {
        double t_cpu = measure_cpu_min(1, [&]() {
            mul_qsmatrix_strassen(C_cpu, A_cpu, B_cpu, opts.block_min);
        });
        printf("  N=%-5ld  CPU Strassen (cutoff=%-3ld): %10.3f ms"
               "  (%.2fx vs GPU naive)\n",
               N, opts.block_min, t_cpu * 1e3, t_cpu / t_gpu_naive);
    } else {
        printf("  N=%-5ld  CPU Strassen : (skipped)\n", N);
    }

    free_gqsmatrix_dev(A_d);
    free_gqsmatrix_dev(B_d);
    free_gqsmatrix_dev(C_d);
    free_qsmatrix(A_cpu);
    free_qsmatrix(B_cpu);
    free_qsmatrix(C_cpu);
}

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
        "  --no-strassen     skip GPU Strassen (on by default; slow)\n"
        "  --no-block        skip GPU block GEMM (on by default)\n"
        "  --no-cpu          skip CPU Strassen entirely (on by default)\n"
        "  --max-cpu N       skip CPU for sizes > N (default: no limit)\n"
        "  --reps R          timed reps per measurement (default 3)\n"
        "  --block-min D     min_dim for GPU block GEMM / GPU Strassen / CPU Strassen cutoff (default 16)\n"
        "  --sizes N1,N2,... override default size list (default 64,128,256,512)\n"
        "\n"
        "Examples:\n"
        "  %s --no-block --no-strassen      # GPU naive + CPU Strassen only\n"
        "  %s --no-cpu --no-strassen        # GPU naive + GPU block only (fast)\n",
        prog, prog, prog);
}

/*--------------------------------------------------------------*/
/* main                                                         */
/*--------------------------------------------------------------*/
int main(int argc, char **argv)
{
    Opts opts;
    opts.sizes = {64, 128, 256, 512};

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--no-strassen"))
            opts.run_strassen = false;
        else if (!strcmp(argv[i], "--strassen"))
            opts.run_strassen = true;   /* alias; default already on */
        else if (!strcmp(argv[i], "--no-block"))
            opts.run_block = false;
        else if (!strcmp(argv[i], "--block"))
            opts.run_block = true;      /* alias; default already on */
        else if (!strcmp(argv[i], "--no-cpu"))
            opts.no_cpu = true;
        else if (!strcmp(argv[i], "--max-cpu") && i + 1 < argc)
            opts.max_cpu = atol(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc)
            opts.reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--block-min") && i + 1 < argc)
            opts.block_min = atol(argv[++i]);
        else if (!strcmp(argv[i], "--sizes") && i + 1 < argc)
            parse_sizes(argv[++i], opts.sizes);
        else {
            usage(argv[0]);
            return 1;
        }
    }

    /* float-element multi-float types need no x87 control-word fix-up. */

    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    printf("=========================================================\n");
    printf(" gdtqs + bncmatmul GPU matmul benchmark (float-based multi-float)\n");
    printf("=========================================================\n");
    printf(" device     : %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf(" reps/meas  : %d (min wall-clock)\n", opts.reps);
    printf(" block min  : %ld  (also used as CPU Strassen cutoff)\n",
           opts.block_min);
    printf(" GPU block  : %s\n",
           opts.run_block    ? "ENABLED (default; pass --no-block to skip)"
                             : "disabled (--no-block)");
    printf(" Strassen   : %s\n",
           opts.run_strassen ? "ENABLED (default; pass --no-strassen to skip)"
                             : "disabled (--no-strassen)");
    printf(" CPU ref    : ");
    if (opts.no_cpu)              printf("disabled (--no-cpu)\n");
    else if (opts.max_cpu < 0)    printf("ENABLED via mul_{dd,qd}matrix_strassen for ALL sizes\n");
    else                          printf("for N <= %ld (mul_{dd,qd}matrix_strassen)\n", opts.max_cpu);
    printf(" sizes      :");
    for (long N : opts.sizes) printf(" %ld", N);
    printf("\n");
    printf("=========================================================\n");

    /* ---------- GDS ---------- */
    printf("\n##### GDS (double-single, 2 floats, ~13 digits) #####\n");
    GDSStart(dev);
    for (long N : opts.sizes) {
        printf("\n[GDS N=%ld]\n", N);
        bench_gds(N, opts);
    }
    GDSEnd();

    /* ---------- GTS ---------- */
    printf("\n##### GTS (triple-single, 3 floats, ~20 digits) #####\n");
    GTSStart(dev);
    for (long N : opts.sizes) {
        printf("\n[GTS N=%ld]\n", N);
        bench_gts(N, opts);
    }
    GTSEnd();

    /* ---------- GQS ---------- */
    printf("\n##### GQS (quad-single, 4 floats, ~26 digits) #####\n");
    GQSStart(dev);
    for (long N : opts.sizes) {
        printf("\n[GQS N=%ld]\n", N);
        bench_gqs(N, opts);
    }
    GQSEnd();

    printf("\n=========================================================\n");
    printf(" done.\n");
    printf("=========================================================\n");
    return 0;
}
