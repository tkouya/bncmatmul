/*****************************************************************
 * bench_gdtq_strassen.cu
 * ----------------------------------------------------------------
 * Performance benchmark for the GPU GDD / GQD matmul paths.
 *
 * For each problem size N and each precision (GDD, GQD):
 *   - GPU naive GEMM    : mul_g{dd,qd}matrix_dev               (always)
 *   - GPU block GEMM    : _bncuda_mul_g{dd,qd}matrix_block     (always)
 *   - GPU Strassen      : _bncuda_mul_g{dd,qd}matrix_strassen  (default on)
 *   - CPU Strassen      : mul_{dd,qd}matrix_strassen           (default on)
 *
 * Each measurement is repeated `--reps` times after one warmup; the
 * minimum wall-clock time is reported (less sensitive to OS noise).
 *
 * Build:  make bench_gdtq_strassen
 *
 * Usage:  ./bench_gdtq_strassen [options]
 *   --no-strassen     skip GPU Strassen (on by default; slow)
 *   --no-cpu          skip CPU Strassen entirely (on by default; slow)
 *   --max-cpu N       skip CPU for sizes > N (default: no limit)
 *   --reps R          timed reps per measurement (default 3)
 *   --block-min D     min_dim for GPU block GEMM and CPU Strassen (default 16)
 *   --sizes N1,N2,... override default size list (default 64,128,256,512)
 *
 * The CPU side uses bncmatmul's mul_ddmatrix_strassen /
 * mul_qdmatrix_strassen (NOT a naive O(N^3) triple loop), so even at
 * large N the CPU run completes in reasonable time.  Linking therefore
 * requires libbncmatmul-X.Y.so in addition to libbncmm_cuda.a.
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

#include <cuda_runtime.h>
#include <vector_types.h>

#include "gqd_type.h"
#include "gddlinear.h"        /* GDDMatrix, GQDMatrix and bridges */
#include "gtdlinear.h"        /* GTDMatrix and bridges (triple-double) */
#include "ddlinear.h"         /* DDMatrix, init_ddmatrix, mul_ddmatrix(_strassen) */
#include "tdlinear.h"         /* TDMatrix, init_tdmatrix, mul_tdmatrix(_strassen) */
#include "qdlinear.h"         /* QDMatrix, init_qdmatrix, mul_qdmatrix(_strassen) */
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
/* random fill into the SoA DDMatrix / QDMatrix used by         */
/* bncmatmul's CPU paths.                                       */
/*--------------------------------------------------------------*/
static void fill_random_dd(DDMatrix A, DDMatrix B, long N, unsigned seed)
{
    srand(seed);
    for (long i = 0; i < N; ++i) {
        for (long j = 0; j < N; ++j) {
            double a = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
            double b = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
            dd_real va(a);
            dd_real vb(b);
            A->element[0][i * N + j] = va.x[0];
            A->element[1][i * N + j] = va.x[1];
            B->element[0][i * N + j] = vb.x[0];
            B->element[1][i * N + j] = vb.x[1];
        }
    }
}

static void fill_random_td(TDMatrix A, TDMatrix B, long N, unsigned seed)
{
    srand(seed);
    for (long i = 0; i < N; ++i) {
        for (long j = 0; j < N; ++j) {
            double a = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
            double b = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
            /* Use qd_real intermediate to get a 3-limb representation,
             * then drop the 4th limb when copying into TDMatrix. */
            qd_real va(a);
            qd_real vb(b);
            A->element[0][i * N + j] = va.x[0];
            A->element[1][i * N + j] = va.x[1];
            A->element[2][i * N + j] = va.x[2];
            B->element[0][i * N + j] = vb.x[0];
            B->element[1][i * N + j] = vb.x[1];
            B->element[2][i * N + j] = vb.x[2];
        }
    }
}

static void fill_random_qd(QDMatrix A, QDMatrix B, long N, unsigned seed)
{
    srand(seed);
    for (long i = 0; i < N; ++i) {
        for (long j = 0; j < N; ++j) {
            double a = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
            double b = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
            qd_real va(a);
            qd_real vb(b);
            A->element[0][i * N + j] = va.x[0];
            A->element[1][i * N + j] = va.x[1];
            A->element[2][i * N + j] = va.x[2];
            A->element[3][i * N + j] = va.x[3];
            B->element[0][i * N + j] = vb.x[0];
            B->element[1][i * N + j] = vb.x[1];
            B->element[2][i * N + j] = vb.x[2];
            B->element[3][i * N + j] = vb.x[3];
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
/* GDD benchmark                                                */
/*--------------------------------------------------------------*/
static void bench_gdd(long N, const Opts &opts)
{
    /* Host-side bncmatmul matrices (SoA layout). */
    DDMatrix A_cpu = init_ddmatrix(N, N);
    DDMatrix B_cpu = init_ddmatrix(N, N);
    DDMatrix C_cpu = init_ddmatrix(N, N);
    fill_random_dd(A_cpu, B_cpu, N, /*seed=*/12345);

    /* GPU side: copy via the bncmatmul SoA-aware bridges. */
    GDDMatrix A_d = init_gddmatrix_dev(N, N);
    GDDMatrix B_d = init_gddmatrix_dev(N, N);
    GDDMatrix C_d = init_gddmatrix_dev(N, N);
    subst_gddmatrix_dev_ddmat(A_d, A_cpu);
    subst_gddmatrix_dev_ddmat(B_d, B_cpu);

    /* --- GPU naive GEMM --- */
    mul_gddmatrix_dev(C_d, A_d, B_d, 128, 128);
    cudaDeviceSynchronize();
    double t_gpu_naive = measure_min(opts.reps, [&]() {
        mul_gddmatrix_dev(C_d, A_d, B_d, 128, 128);
    });
    printf("  N=%-5ld  GPU naive    : %10.3f ms\n",
           N, t_gpu_naive * 1e3);

    /* --- GPU block GEMM (default on; --no-block to skip) --- */
    if (opts.run_block) {
        _bncuda_mul_gddmatrix_block(C_d, A_d, B_d, opts.block_min, 128, 128);
        cudaDeviceSynchronize();
        double t_gpu_block = measure_min(opts.reps, [&]() {
            _bncuda_mul_gddmatrix_block(C_d, A_d, B_d, opts.block_min, 128, 128);
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
            _bncuda_mul_gddmatrix_strassen(C_d, A_d, B_d,
                                            cutoff, 128, 128);
            cudaDeviceSynchronize();
            double t = measure_min(opts.reps, [&]() {
                _bncuda_mul_gddmatrix_strassen(C_d, A_d, B_d,
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
            mul_ddmatrix_strassen(C_cpu, A_cpu, B_cpu, opts.block_min);
        });
        printf("  N=%-5ld  CPU Strassen (cutoff=%-3ld): %10.3f ms"
               "  (%.2fx vs GPU naive)\n",
               N, opts.block_min, t_cpu * 1e3, t_cpu / t_gpu_naive);
    } else {
        printf("  N=%-5ld  CPU Strassen : (skipped)\n", N);
    }

    free_gddmatrix_dev(A_d);
    free_gddmatrix_dev(B_d);
    free_gddmatrix_dev(C_d);
    free_ddmatrix(A_cpu);
    free_ddmatrix(B_cpu);
    free_ddmatrix(C_cpu);
}

/*--------------------------------------------------------------*/
/* GTD benchmark                                                */
/*--------------------------------------------------------------*/
static void bench_gtd(long N, const Opts &opts)
{
    TDMatrix A_cpu = init_tdmatrix(N, N);
    TDMatrix B_cpu = init_tdmatrix(N, N);
    TDMatrix C_cpu = init_tdmatrix(N, N);
    fill_random_td(A_cpu, B_cpu, N, /*seed=*/24680);

    GTDMatrix A_d = init_gtdmatrix_dev(N, N);
    GTDMatrix B_d = init_gtdmatrix_dev(N, N);
    GTDMatrix C_d = init_gtdmatrix_dev(N, N);
    subst_gtdmatrix_dev_tdmat(A_d, A_cpu);
    subst_gtdmatrix_dev_tdmat(B_d, B_cpu);

    /* --- GPU naive GEMM --- */
    mul_gtdmatrix_dev(C_d, A_d, B_d, 128, 128);
    cudaDeviceSynchronize();
    double t_gpu_naive = measure_min(opts.reps, [&]() {
        mul_gtdmatrix_dev(C_d, A_d, B_d, 128, 128);
    });
    printf("  N=%-5ld  GPU naive    : %10.3f ms\n",
           N, t_gpu_naive * 1e3);

    /* --- GPU block GEMM (default on; --no-block to skip) --- */
    if (opts.run_block) {
        _bncuda_mul_gtdmatrix_block(C_d, A_d, B_d, opts.block_min, 128, 128);
        cudaDeviceSynchronize();
        double t_gpu_block = measure_min(opts.reps, [&]() {
            _bncuda_mul_gtdmatrix_block(C_d, A_d, B_d, opts.block_min, 128, 128);
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
            _bncuda_mul_gtdmatrix_strassen(C_d, A_d, B_d,
                                            cutoff, 128, 128);
            cudaDeviceSynchronize();
            double t = measure_min(opts.reps, [&]() {
                _bncuda_mul_gtdmatrix_strassen(C_d, A_d, B_d,
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
            mul_tdmatrix_strassen(C_cpu, A_cpu, B_cpu, opts.block_min);
        });
        printf("  N=%-5ld  CPU Strassen (cutoff=%-3ld): %10.3f ms"
               "  (%.2fx vs GPU naive)\n",
               N, opts.block_min, t_cpu * 1e3, t_cpu / t_gpu_naive);
    } else {
        printf("  N=%-5ld  CPU Strassen : (skipped)\n", N);
    }

    free_gtdmatrix_dev(A_d);
    free_gtdmatrix_dev(B_d);
    free_gtdmatrix_dev(C_d);
    free_tdmatrix(A_cpu);
    free_tdmatrix(B_cpu);
    free_tdmatrix(C_cpu);
}

/*--------------------------------------------------------------*/
/* GQD benchmark                                                */
/*--------------------------------------------------------------*/
static void bench_gqd(long N, const Opts &opts)
{
    QDMatrix A_cpu = init_qdmatrix(N, N);
    QDMatrix B_cpu = init_qdmatrix(N, N);
    QDMatrix C_cpu = init_qdmatrix(N, N);
    fill_random_qd(A_cpu, B_cpu, N, /*seed=*/67890);

    GQDMatrix A_d = init_gqdmatrix_dev(N, N);
    GQDMatrix B_d = init_gqdmatrix_dev(N, N);
    GQDMatrix C_d = init_gqdmatrix_dev(N, N);
    subst_gqdmatrix_dev_qdmat(A_d, A_cpu);
    subst_gqdmatrix_dev_qdmat(B_d, B_cpu);

    /* --- GPU naive GEMM --- */
    mul_gqdmatrix_dev(C_d, A_d, B_d, 128, 128);
    cudaDeviceSynchronize();
    double t_gpu_naive = measure_min(opts.reps, [&]() {
        mul_gqdmatrix_dev(C_d, A_d, B_d, 128, 128);
    });
    printf("  N=%-5ld  GPU naive    : %10.3f ms\n",
           N, t_gpu_naive * 1e3);

    /* --- GPU block GEMM (default on; --no-block to skip) --- */
    if (opts.run_block) {
        _bncuda_mul_gqdmatrix_block(C_d, A_d, B_d, opts.block_min, 128, 128);
        cudaDeviceSynchronize();
        double t_gpu_block = measure_min(opts.reps, [&]() {
            _bncuda_mul_gqdmatrix_block(C_d, A_d, B_d, opts.block_min, 128, 128);
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
            _bncuda_mul_gqdmatrix_strassen(C_d, A_d, B_d,
                                            cutoff, 128, 128);
            cudaDeviceSynchronize();
            double t = measure_min(opts.reps, [&]() {
                _bncuda_mul_gqdmatrix_strassen(C_d, A_d, B_d,
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
            mul_qdmatrix_strassen(C_cpu, A_cpu, B_cpu, opts.block_min);
        });
        printf("  N=%-5ld  CPU Strassen (cutoff=%-3ld): %10.3f ms"
               "  (%.2fx vs GPU naive)\n",
               N, opts.block_min, t_cpu * 1e3, t_cpu / t_gpu_naive);
    } else {
        printf("  N=%-5ld  CPU Strassen : (skipped)\n", N);
    }

    free_gqdmatrix_dev(A_d);
    free_gqdmatrix_dev(B_d);
    free_gqdmatrix_dev(C_d);
    free_qdmatrix(A_cpu);
    free_qdmatrix(B_cpu);
    free_qdmatrix(C_cpu);
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
        "  --block-min D     min_dim for GPU block and CPU Strassen (default 16)\n"
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

    unsigned int cw;
    fpu_fix_start(&cw);

    int dev = 0;
    cudaSetDevice(dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);

    printf("=========================================================\n");
    printf(" gdtq + bncmatmul GPU matmul benchmark\n");
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

    /* ---------- GDD ---------- */
    printf("\n##### GDD (double-double, ~32 digits) #####\n");
    GDDStart(dev);
    for (long N : opts.sizes) {
        printf("\n[GDD N=%ld]\n", N);
        bench_gdd(N, opts);
    }
    GDDEnd();

    /* ---------- GTD ---------- */
    printf("\n##### GTD (triple-double, ~47 digits) #####\n");
    GTDStart(dev);
    for (long N : opts.sizes) {
        printf("\n[GTD N=%ld]\n", N);
        bench_gtd(N, opts);
    }
    GTDEnd();

    /* ---------- GQD ---------- */
    printf("\n##### GQD (quad-double, ~63 digits) #####\n");
    GQDStart(dev);
    for (long N : opts.sizes) {
        printf("\n[GQD N=%ld]\n", N);
        bench_gqd(N, opts);
    }
    GQDEnd();

    fpu_fix_end(&cw);
    printf("\n=========================================================\n");
    printf(" done.\n");
    printf("=========================================================\n");
    return 0;
}
