/*****************************************************************
 * test_gdd_strassen.cu
 * ----------------------------------------------------------------
 * Self-contained smoke test for the revived GPU GDD Strassen path.
 *
 * 4 mini-tests (each prints PASS/FAIL):
 *   1. Library init           — GDDStart / GDDEnd
 *   2. Naive GEMM identity    — mul_gddmatrix_dev: I * ones = ones
 *   3. Naive GEMM random      — mul_gddmatrix_dev vs CPU dd_real triple loop
 *   4. Strassen random        — _bncuda_mul_gddmatrix_strassen vs same CPU
 *
 * No dependency on libbncmatmul.so itself (only on libbncmm_cuda.a +
 * libqd + cudart + cublas).  CPU reference is computed inline using
 * QD's dd_real triple loop, so failures here mean the GPU side is wrong
 * (not the CPU side).
 *****************************************************************/
/* __NV_NO_VECTOR_DEPRECATION_DIAG is set on the nvcc command line by
 * the Makefile, so no #define here. */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#include <qd/qd_real.h>
#include <qd/fpu.h>

#include <cuda_runtime.h>
#include <vector_types.h>

#include "gqd_type.h"     // gdd_real (typedef of double2), GDDStart/GDDEnd
#include "gddlinear.h"    // GDDMatrix, init_gddmatrix_dev, mul_gddmatrix_dev
#include "matmul_strassen.h"  // _bncuda_mul_gddmatrix_strassen


/*----------------------------------------------------------------
 * Helpers
 *----------------------------------------------------------------*/
static dd_real to_dd(gdd_real g)
{
    dd_real r;
    r.x[0] = g.x;
    r.x[1] = g.y;
    return r;
}

static gdd_real to_g(const dd_real &d)
{
    return make_double2(d.x[0], d.x[1]);
}

/* Maximum relative error between two N×N gdd_real arrays interpreted
 * as dd_real.  Compares  |gpu - cpu| / max(|cpu|, 1).               */
static double max_rel_err(const gdd_real *gpu, const dd_real *cpu, long N)
{
    double max_err = 0.0;
    for (long i = 0; i < N * N; ++i) {
        dd_real g = to_dd(gpu[i]);
        dd_real diff = abs(g - cpu[i]);
        double abs_cpu = std::fabs(to_double(cpu[i]));
        double scale = (abs_cpu > 1.0) ? abs_cpu : 1.0;
        double rel = to_double(diff) / scale;
        if (rel > max_err) max_err = rel;
    }
    return max_err;
}

/* Naive O(N^3) CPU matmul over dd_real, used as the gold reference. */
static void cpu_dd_matmul(dd_real *C, const dd_real *A, const dd_real *B, long N)
{
    for (long i = 0; i < N; ++i) {
        for (long j = 0; j < N; ++j) {
            dd_real s = 0.0;
            for (long k = 0; k < N; ++k)
                s = s + A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
    }
}

/* Fill A, B with deterministic pseudo-random dd_real values in [-1, 1]. */
static void fill_random(dd_real *A, dd_real *B, long N, unsigned seed)
{
    srand(seed);
    for (long i = 0; i < N * N; ++i) {
        double a = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
        double b = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
        A[i] = dd_real(a) + dd_real(1e-17) * dd_real((double)i);
        B[i] = dd_real(b) - dd_real(1e-17) * dd_real((double)i);
    }
}

/*----------------------------------------------------------------
 * Test 1: library init only
 *----------------------------------------------------------------*/
static int test_init(void)
{
    printf("\n[Test 1] GDDStart / GDDEnd round-trip\n");
    /* GDDStart was already called by main(); just check it didn't
     * crash by getting some info. */
    int dev = -1;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) {
        printf("  cudaGetDevice failed: %s\n", cudaGetErrorString(e));
        return 1;
    }
    printf("  current CUDA device   = %d\n", dev);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("  device name           = %s\n", prop.name);
    printf("  compute capability    = %d.%d\n", prop.major, prop.minor);
    printf("  Result: PASS\n");
    return 0;
}

/*----------------------------------------------------------------
 * Test 2: I * ones = ones (catches gross transposition / indexing bugs)
 *----------------------------------------------------------------*/
static int test_identity_naive(long N)
{
    printf("\n[Test 2] mul_gddmatrix_dev:  I_%ld * ones = ones\n", N);

    gdd_real *A_h = new gdd_real[N * N];
    gdd_real *B_h = new gdd_real[N * N];
    gdd_real *C_h = new gdd_real[N * N];
    for (long i = 0; i < N; ++i)
        for (long j = 0; j < N; ++j) {
            A_h[i * N + j] = make_double2((i == j) ? 1.0 : 0.0, 0.0);
            B_h[i * N + j] = make_double2(1.0, 0.0);
        }

    GDDMatrix A_d = init_gddmatrix_dev(N, N);
    GDDMatrix B_d = init_gddmatrix_dev(N, N);
    GDDMatrix C_d = init_gddmatrix_dev(N, N);

    cudaMemcpy(A_d->element, A_h, N * N * sizeof(gdd_real),
               cudaMemcpyHostToDevice);
    cudaMemcpy(B_d->element, B_h, N * N * sizeof(gdd_real),
               cudaMemcpyHostToDevice);

    mul_gddmatrix_dev(C_d, A_d, B_d, 128, 128);
    cudaDeviceSynchronize();

    cudaMemcpy(C_h, C_d->element, N * N * sizeof(gdd_real),
               cudaMemcpyDeviceToHost);

    double max_err = 0.0;
    for (long i = 0; i < N * N; ++i) {
        double err = std::fabs(C_h[i].x - 1.0) + std::fabs(C_h[i].y);
        if (err > max_err) max_err = err;
    }
    printf("  max abs error         = %.3e\n", max_err);
    int ok = (max_err < 1e-28);
    printf("  Result: %s\n", ok ? "PASS" : "FAIL");

    free_gddmatrix_dev(A_d);
    free_gddmatrix_dev(B_d);
    free_gddmatrix_dev(C_d);
    delete[] A_h; delete[] B_h; delete[] C_h;
    return ok ? 0 : 1;
}

/*----------------------------------------------------------------
 * Test 3: random N×N, naive GPU GEMM vs CPU dd_real triple loop
 *----------------------------------------------------------------*/
static int test_random_naive(long N, unsigned seed)
{
    printf("\n[Test 3] mul_gddmatrix_dev vs CPU dd_real, N = %ld\n", N);

    dd_real  *A_cpu = new dd_real[N * N];
    dd_real  *B_cpu = new dd_real[N * N];
    dd_real  *C_cpu = new dd_real[N * N];
    gdd_real *A_h   = new gdd_real[N * N];
    gdd_real *B_h   = new gdd_real[N * N];
    gdd_real *C_h   = new gdd_real[N * N];

    fill_random(A_cpu, B_cpu, N, seed);
    cpu_dd_matmul(C_cpu, A_cpu, B_cpu, N);
    for (long i = 0; i < N * N; ++i) {
        A_h[i] = to_g(A_cpu[i]);
        B_h[i] = to_g(B_cpu[i]);
    }

    GDDMatrix A_d = init_gddmatrix_dev(N, N);
    GDDMatrix B_d = init_gddmatrix_dev(N, N);
    GDDMatrix C_d = init_gddmatrix_dev(N, N);

    cudaMemcpy(A_d->element, A_h, N * N * sizeof(gdd_real),
               cudaMemcpyHostToDevice);
    cudaMemcpy(B_d->element, B_h, N * N * sizeof(gdd_real),
               cudaMemcpyHostToDevice);

    mul_gddmatrix_dev(C_d, A_d, B_d, 128, 128);
    cudaDeviceSynchronize();

    cudaMemcpy(C_h, C_d->element, N * N * sizeof(gdd_real),
               cudaMemcpyDeviceToHost);

    double err = max_rel_err(C_h, C_cpu, N);
    printf("  max relative error    = %.3e\n", err);
    printf("  dd_real eps           = %.3e\n", to_double(dd_real::_eps));
    /* Threshold: ~ N * eps allowed (accumulation of rounding errors). */
    double thresh = (double)N * to_double(dd_real::_eps) * 16.0;
    int ok = (err < thresh);
    printf("  Result: %s (threshold %.3e)\n", ok ? "PASS" : "FAIL", thresh);

    free_gddmatrix_dev(A_d);
    free_gddmatrix_dev(B_d);
    free_gddmatrix_dev(C_d);
    delete[] A_cpu; delete[] B_cpu; delete[] C_cpu;
    delete[] A_h;   delete[] B_h;   delete[] C_h;
    return ok ? 0 : 1;
}

/*----------------------------------------------------------------
 * Test 4: random N×N, GPU Strassen vs CPU dd_real triple loop
 *----------------------------------------------------------------*/
static int test_random_strassen(long N, long min_dim, unsigned seed)
{
    printf("\n[Test 4] _bncuda_mul_gddmatrix_strassen vs CPU dd_real, "
           "N = %ld, cutoff = %ld\n", N, min_dim);

    dd_real  *A_cpu = new dd_real[N * N];
    dd_real  *B_cpu = new dd_real[N * N];
    dd_real  *C_cpu = new dd_real[N * N];
    gdd_real *A_h   = new gdd_real[N * N];
    gdd_real *B_h   = new gdd_real[N * N];
    gdd_real *C_h   = new gdd_real[N * N];

    fill_random(A_cpu, B_cpu, N, seed);
    cpu_dd_matmul(C_cpu, A_cpu, B_cpu, N);
    for (long i = 0; i < N * N; ++i) {
        A_h[i] = to_g(A_cpu[i]);
        B_h[i] = to_g(B_cpu[i]);
    }

    GDDMatrix A_d = init_gddmatrix_dev(N, N);
    GDDMatrix B_d = init_gddmatrix_dev(N, N);
    GDDMatrix C_d = init_gddmatrix_dev(N, N);

    cudaMemcpy(A_d->element, A_h, N * N * sizeof(gdd_real),
               cudaMemcpyHostToDevice);
    cudaMemcpy(B_d->element, B_h, N * N * sizeof(gdd_real),
               cudaMemcpyHostToDevice);

    _bncuda_reset_num_mul_gddmatrix_strassen();
    _bncuda_mul_gddmatrix_strassen(C_d, A_d, B_d, min_dim, 128, 128);
    cudaDeviceSynchronize();

    cudaMemcpy(C_h, C_d->element, N * N * sizeof(gdd_real),
               cudaMemcpyDeviceToHost);

    long num_addsub = 0, num_mul = 0;
    _bncuda_get_num_mul_gddmatrix_strassen(&num_addsub, &num_mul);
    printf("  Strassen counters     : add/sub = %ld, mul = %ld\n",
           num_addsub, num_mul);

    double err = max_rel_err(C_h, C_cpu, N);
    printf("  max relative error    = %.3e\n", err);
    /* Strassen accumulates more rounding than naive GEMM but with dd_real
     * we still expect ~ N * eps precision. */
    double thresh = (double)N * to_double(dd_real::_eps) * 64.0;
    int ok = (err < thresh);
    printf("  Result: %s (threshold %.3e)\n", ok ? "PASS" : "FAIL", thresh);

    free_gddmatrix_dev(A_d);
    free_gddmatrix_dev(B_d);
    free_gddmatrix_dev(C_d);
    delete[] A_cpu; delete[] B_cpu; delete[] C_cpu;
    delete[] A_h;   delete[] B_h;   delete[] C_h;
    return ok ? 0 : 1;
}


/*----------------------------------------------------------------
 * main
 *----------------------------------------------------------------*/
int main(int argc, char **argv)
{
    /* x86 FPU fix; no-op on AArch64 but cheap to call. */
    unsigned int cw;
    fpu_fix_start(&cw);

    /* Default sizes — overridable via argv. */
    long N_naive    = (argc > 1) ? atol(argv[1]) : 32;
    long N_strassen = (argc > 2) ? atol(argv[2]) : 64;
    long min_dim    = (argc > 3) ? atol(argv[3]) : 16;

    printf("=============================================\n");
    printf(" gdtq + bncmatmul GPU GDD Strassen smoke test\n");
    printf("=============================================\n");
    printf("naive matmul N    = %ld\n", N_naive);
    printf("Strassen matmul N = %ld (cutoff %ld)\n", N_strassen, min_dim);

    GDDStart(0);

    int fails = 0;
    fails += test_init();
    fails += test_identity_naive(N_naive);
    fails += test_random_naive(N_naive, /*seed=*/12345);
    fails += test_random_strassen(N_strassen, min_dim, /*seed=*/67890);

    GDDEnd();
    fpu_fix_end(&cw);

    printf("\n=============================================\n");
    printf(" Total failures: %d\n", fails);
    printf("=============================================\n");
    return fails == 0 ? 0 : 1;
}
