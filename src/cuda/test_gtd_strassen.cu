/*****************************************************************
 * test_gtd_strassen.cu
 * ----------------------------------------------------------------
 * GTD (triple-double, ~47 decimal digits) counterpart of
 * test_gdd_strassen.cu.  Exercises the GPU GTD path of bncmatmul:
 *
 *   1. Library init           — GTDStart / GTDEnd
 *   2. Naive GEMM identity    — mul_gtdmatrix_dev: I * ones = ones
 *   3. Naive GEMM random      — mul_gtdmatrix_dev vs CPU td_real triple loop
 *   4. Strassen random        — _bncuda_mul_gtdmatrix_strassen vs same CPU
 *
 * Uses dtq-0.0.2's td_real (~47 decimal digits, ~156-bit significand)
 * as the CPU reference, so comparisons are apples-to-apples with the
 * GPU gtd_real (also 3 doubles).
 *
 * Self-contained.  Depends only on libbncmm_cuda.a + libqd (with
 * dtq-0.0.2 td_real extension) + cudart + cublas.
 *****************************************************************/
/* __NV_NO_VECTOR_DEPRECATION_DIAG is set on the nvcc command line. */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#include <qd/qd_real.h>
#include <qd/td_real.h>
#include <qd/fpu.h>

#include <cuda_runtime.h>
#include <vector_types.h>

#include "gqd_type.h"     // gtd_real (typedef of double3), GTDStart/GTDEnd
#include "gtdlinear.h"    // GTDMatrix, init_gtdmatrix_dev, mul_gtdmatrix_dev
#include "matmul_strassen.h"  // _bncuda_mul_gtdmatrix_strassen


/*----------------------------------------------------------------
 * Helpers
 *----------------------------------------------------------------*/
static td_real to_td(gtd_real g)
{
    return td_real(g.x, g.y, g.z);
}

static gtd_real to_g(const td_real &t)
{
    return make_double3(t.x[0], t.x[1], t.x[2]);
}

/* Maximum relative error between two N×N gtd_real arrays vs td_real. */
static double max_rel_err(const gtd_real *gpu, const td_real *cpu, long N)
{
    double max_err = 0.0;
    for (long i = 0; i < N * N; ++i) {
        td_real g = to_td(gpu[i]);
        td_real diff = abs(g - cpu[i]);
        double abs_cpu = std::fabs(to_double(cpu[i]));
        double scale = (abs_cpu > 1.0) ? abs_cpu : 1.0;
        double rel = to_double(diff) / scale;
        if (rel > max_err) max_err = rel;
    }
    return max_err;
}

/* Naive O(N^3) CPU matmul over td_real, used as the gold reference. */
static void cpu_td_matmul(td_real *C, const td_real *A, const td_real *B, long N)
{
    for (long i = 0; i < N; ++i) {
        for (long j = 0; j < N; ++j) {
            td_real s = 0.0;
            for (long k = 0; k < N; ++k)
                s = s + A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
    }
}

static void fill_random(td_real *A, td_real *B, long N, unsigned seed)
{
    srand(seed);
    for (long i = 0; i < N * N; ++i) {
        double a = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
        double b = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
        /* Add tiny perturbation to lower limbs so the extra precision
         * is actually exercised. */
        A[i] = td_real(a) + td_real(1e-17) * td_real((double)i)
             + td_real(1e-34) * td_real((double)(i * i));
        B[i] = td_real(b) - td_real(1e-17) * td_real((double)i)
             + td_real(1e-34) * td_real((double)(i + 1));
    }
}

/*----------------------------------------------------------------
 * Test 1: library init only
 *----------------------------------------------------------------*/
static int test_init(void)
{
    printf("\n[Test 1] GTDStart / GTDEnd round-trip\n");
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
 * Test 2: I * ones = ones
 *----------------------------------------------------------------*/
static int test_identity_naive(long N)
{
    printf("\n[Test 2] mul_gtdmatrix_dev:  I_%ld * ones = ones\n", N);

    gtd_real *A_h = new gtd_real[N * N];
    gtd_real *B_h = new gtd_real[N * N];
    gtd_real *C_h = new gtd_real[N * N];
    for (long i = 0; i < N; ++i)
        for (long j = 0; j < N; ++j) {
            A_h[i * N + j] = make_double3((i == j) ? 1.0 : 0.0, 0.0, 0.0);
            B_h[i * N + j] = make_double3(1.0, 0.0, 0.0);
        }

    GTDMatrix A_d = init_gtdmatrix_dev(N, N);
    GTDMatrix B_d = init_gtdmatrix_dev(N, N);
    GTDMatrix C_d = init_gtdmatrix_dev(N, N);

    cudaMemcpy(A_d->element, A_h, N * N * sizeof(gtd_real),
               cudaMemcpyHostToDevice);
    cudaMemcpy(B_d->element, B_h, N * N * sizeof(gtd_real),
               cudaMemcpyHostToDevice);

    mul_gtdmatrix_dev(C_d, A_d, B_d, 128, 128);
    cudaDeviceSynchronize();

    cudaMemcpy(C_h, C_d->element, N * N * sizeof(gtd_real),
               cudaMemcpyDeviceToHost);

    double max_err = 0.0;
    for (long i = 0; i < N * N; ++i) {
        /* Sum the 3 limbs and compare against expected 1.0. */
        double total = C_h[i].x + C_h[i].y + C_h[i].z;
        double err = std::fabs(total - 1.0);
        if (err > max_err) max_err = err;
    }
    printf("  max abs error         = %.3e\n", max_err);
    int ok = (max_err < 1e-44);
    printf("  Result: %s\n", ok ? "PASS" : "FAIL");

    free_gtdmatrix_dev(A_d);
    free_gtdmatrix_dev(B_d);
    free_gtdmatrix_dev(C_d);
    delete[] A_h; delete[] B_h; delete[] C_h;
    return ok ? 0 : 1;
}

/*----------------------------------------------------------------
 * Test 3: random N×N, naive GPU GEMM vs CPU qd_real triple loop
 *----------------------------------------------------------------*/
static int test_random_naive(long N, unsigned seed)
{
    printf("\n[Test 3] mul_gtdmatrix_dev vs CPU qd_real, N = %ld\n", N);

    td_real  *A_cpu = new td_real[N * N];
    td_real  *B_cpu = new td_real[N * N];
    td_real  *C_cpu = new td_real[N * N];
    gtd_real *A_h   = new gtd_real[N * N];
    gtd_real *B_h   = new gtd_real[N * N];
    gtd_real *C_h   = new gtd_real[N * N];

    fill_random(A_cpu, B_cpu, N, seed);
    cpu_td_matmul(C_cpu, A_cpu, B_cpu, N);
    for (long i = 0; i < N * N; ++i) {
        A_h[i] = to_g(A_cpu[i]);
        B_h[i] = to_g(B_cpu[i]);
    }

    GTDMatrix A_d = init_gtdmatrix_dev(N, N);
    GTDMatrix B_d = init_gtdmatrix_dev(N, N);
    GTDMatrix C_d = init_gtdmatrix_dev(N, N);

    cudaMemcpy(A_d->element, A_h, N * N * sizeof(gtd_real),
               cudaMemcpyHostToDevice);
    cudaMemcpy(B_d->element, B_h, N * N * sizeof(gtd_real),
               cudaMemcpyHostToDevice);

    mul_gtdmatrix_dev(C_d, A_d, B_d, 128, 128);
    cudaDeviceSynchronize();

    cudaMemcpy(C_h, C_d->element, N * N * sizeof(gtd_real),
               cudaMemcpyDeviceToHost);

    double err = max_rel_err(C_h, C_cpu, N);
    printf("  max relative error    = %.3e\n", err);
    printf("  td_real eps           = %.3e\n", to_double(td_real::_eps));
    double thresh = (double)N * to_double(td_real::_eps) * 16.0;
    int ok = (err < thresh);
    printf("  Result: %s (threshold %.3e)\n", ok ? "PASS" : "FAIL", thresh);

    free_gtdmatrix_dev(A_d);
    free_gtdmatrix_dev(B_d);
    free_gtdmatrix_dev(C_d);
    delete[] A_cpu; delete[] B_cpu; delete[] C_cpu;
    delete[] A_h;   delete[] B_h;   delete[] C_h;
    return ok ? 0 : 1;
}

/*----------------------------------------------------------------
 * Test 4: random N×N, GPU Strassen vs CPU qd_real triple loop
 *----------------------------------------------------------------*/
static int test_random_strassen(long N, long min_dim, unsigned seed)
{
    printf("\n[Test 4] _bncuda_mul_gtdmatrix_strassen vs CPU qd_real, "
           "N = %ld, cutoff = %ld\n", N, min_dim);

    td_real  *A_cpu = new td_real[N * N];
    td_real  *B_cpu = new td_real[N * N];
    td_real  *C_cpu = new td_real[N * N];
    gtd_real *A_h   = new gtd_real[N * N];
    gtd_real *B_h   = new gtd_real[N * N];
    gtd_real *C_h   = new gtd_real[N * N];

    fill_random(A_cpu, B_cpu, N, seed);
    cpu_td_matmul(C_cpu, A_cpu, B_cpu, N);
    for (long i = 0; i < N * N; ++i) {
        A_h[i] = to_g(A_cpu[i]);
        B_h[i] = to_g(B_cpu[i]);
    }

    GTDMatrix A_d = init_gtdmatrix_dev(N, N);
    GTDMatrix B_d = init_gtdmatrix_dev(N, N);
    GTDMatrix C_d = init_gtdmatrix_dev(N, N);

    cudaMemcpy(A_d->element, A_h, N * N * sizeof(gtd_real),
               cudaMemcpyHostToDevice);
    cudaMemcpy(B_d->element, B_h, N * N * sizeof(gtd_real),
               cudaMemcpyHostToDevice);

    _bncuda_reset_num_mul_gtdmatrix_strassen();
    _bncuda_mul_gtdmatrix_strassen(C_d, A_d, B_d, min_dim, 128, 128);
    cudaDeviceSynchronize();

    cudaMemcpy(C_h, C_d->element, N * N * sizeof(gtd_real),
               cudaMemcpyDeviceToHost);

    long num_addsub = 0, num_mul = 0;
    _bncuda_get_num_mul_gtdmatrix_strassen(&num_addsub, &num_mul);
    printf("  Strassen counters     : add/sub = %ld, mul = %ld\n",
           num_addsub, num_mul);

    double err = max_rel_err(C_h, C_cpu, N);
    printf("  max relative error    = %.3e\n", err);
    double thresh = (double)N * to_double(td_real::_eps) * 64.0;
    int ok = (err < thresh);
    printf("  Result: %s (threshold %.3e)\n", ok ? "PASS" : "FAIL", thresh);

    free_gtdmatrix_dev(A_d);
    free_gtdmatrix_dev(B_d);
    free_gtdmatrix_dev(C_d);
    delete[] A_cpu; delete[] B_cpu; delete[] C_cpu;
    delete[] A_h;   delete[] B_h;   delete[] C_h;
    return ok ? 0 : 1;
}


/*----------------------------------------------------------------
 * main
 *----------------------------------------------------------------*/
int main(int argc, char **argv)
{
    unsigned int cw;
    fpu_fix_start(&cw);

    long N_naive    = (argc > 1) ? atol(argv[1]) : 32;
    long N_strassen = (argc > 2) ? atol(argv[2]) : 64;
    long min_dim    = (argc > 3) ? atol(argv[3]) : 16;

    printf("=============================================\n");
    printf(" gdtq + bncmatmul GPU GTD Strassen smoke test\n");
    printf("=============================================\n");
    printf("naive matmul N    = %ld\n", N_naive);
    printf("Strassen matmul N = %ld (cutoff %ld)\n", N_strassen, min_dim);

    GTDStart(0);

    int fails = 0;
    fails += test_init();
    fails += test_identity_naive(N_naive);
    fails += test_random_naive(N_naive, /*seed=*/12345);
    fails += test_random_strassen(N_strassen, min_dim, /*seed=*/67890);

    GTDEnd();
    fpu_fix_end(&cw);

    printf("\n=============================================\n");
    printf(" Total failures: %d\n", fails);
    printf("=============================================\n");
    return fails == 0 ? 0 : 1;
}
