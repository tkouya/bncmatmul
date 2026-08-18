/*****************************************************************
 * test_gqd_strassen.cu
 * ----------------------------------------------------------------
 * GQD (quad-double, ~63 decimal digits) counterpart of
 * test_gdd_strassen.cu.  Exercises the GPU GQD path of bncmatmul:
 *
 *   1. Library init           — GQDStart / GQDEnd
 *   2. Naive GEMM identity    — mul_gqdmatrix_dev: I * ones = ones
 *   3. Naive GEMM random      — mul_gqdmatrix_dev vs CPU qd_real triple loop
 *   4. Strassen random        — _bncuda_mul_gqdmatrix_strassen vs same CPU
 *
 * Self-contained.  Depends only on libbncmm_cuda.a + libqd + cudart +
 * cublas.
 *****************************************************************/
/* __NV_NO_VECTOR_DEPRECATION_DIAG is set on the nvcc command line. */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#include <qd/qd_real.h>
#include <qd/fpu.h>

#include <cuda_runtime.h>
#include <vector_types.h>

#include "gqd_type.h"     // gqd_real (typedef of double4), GQDStart/GQDEnd
#include "gddlinear.h"    // GQDMatrix, init_gqdmatrix_dev, mul_gqdmatrix_dev
#include "matmul_strassen.h"  // _bncuda_mul_gqdmatrix_strassen


/*----------------------------------------------------------------
 * Helpers
 *----------------------------------------------------------------*/
static qd_real to_qd(gqd_real g)
{
    qd_real r;
    r.x[0] = g.x;
    r.x[1] = g.y;
    r.x[2] = g.z;
    r.x[3] = g.w;
    return r;
}

static gqd_real to_g(const qd_real &q)
{
    return make_double4(q.x[0], q.x[1], q.x[2], q.x[3]);
}

static double max_rel_err(const gqd_real *gpu, const qd_real *cpu, long N)
{
    double max_err = 0.0;
    for (long i = 0; i < N * N; ++i) {
        qd_real g = to_qd(gpu[i]);
        qd_real diff = abs(g - cpu[i]);
        double abs_cpu = std::fabs(to_double(cpu[i]));
        double scale = (abs_cpu > 1.0) ? abs_cpu : 1.0;
        double rel = to_double(diff) / scale;
        if (rel > max_err) max_err = rel;
    }
    return max_err;
}

/* Naive O(N^3) CPU matmul over qd_real, used as the gold reference. */
static void cpu_qd_matmul(qd_real *C, const qd_real *A, const qd_real *B, long N)
{
    for (long i = 0; i < N; ++i) {
        for (long j = 0; j < N; ++j) {
            qd_real s = 0.0;
            for (long k = 0; k < N; ++k)
                s = s + A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
    }
}

static void fill_random(qd_real *A, qd_real *B, long N, unsigned seed)
{
    srand(seed);
    for (long i = 0; i < N * N; ++i) {
        double a = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
        double b = (rand() / (double)RAND_MAX) * 2.0 - 1.0;
        /* Add tiny perturbation to lower limbs so qd's extra precision
         * is actually exercised (not just identical to double). */
        A[i] = qd_real(a) + qd_real(1e-17) * qd_real((double)i)
             + qd_real(1e-34) * qd_real((double)(i * i));
        B[i] = qd_real(b) - qd_real(1e-17) * qd_real((double)i)
             + qd_real(1e-50) * qd_real((double)(i + 1));
    }
}

/*----------------------------------------------------------------
 * Test 1: library init only
 *----------------------------------------------------------------*/
static int test_init(void)
{
    printf("\n[Test 1] GQDStart / GQDEnd round-trip\n");
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
    printf("\n[Test 2] mul_gqdmatrix_dev:  I_%ld * ones = ones\n", N);

    gqd_real *A_h = new gqd_real[N * N];
    gqd_real *B_h = new gqd_real[N * N];
    gqd_real *C_h = new gqd_real[N * N];
    for (long i = 0; i < N; ++i)
        for (long j = 0; j < N; ++j) {
            A_h[i * N + j] = make_double4((i == j) ? 1.0 : 0.0, 0.0, 0.0, 0.0);
            B_h[i * N + j] = make_double4(1.0, 0.0, 0.0, 0.0);
        }

    GQDMatrix A_d = init_gqdmatrix_dev(N, N);
    GQDMatrix B_d = init_gqdmatrix_dev(N, N);
    GQDMatrix C_d = init_gqdmatrix_dev(N, N);

    cudaMemcpy(A_d->element, A_h, N * N * sizeof(gqd_real),
               cudaMemcpyHostToDevice);
    cudaMemcpy(B_d->element, B_h, N * N * sizeof(gqd_real),
               cudaMemcpyHostToDevice);

    mul_gqdmatrix_dev(C_d, A_d, B_d, 128, 128);
    cudaDeviceSynchronize();

    cudaMemcpy(C_h, C_d->element, N * N * sizeof(gqd_real),
               cudaMemcpyDeviceToHost);

    double max_err = 0.0;
    for (long i = 0; i < N * N; ++i) {
        /* Sum all 4 limbs to compare against expected 1.0. */
        double total = C_h[i].x + C_h[i].y + C_h[i].z + C_h[i].w;
        double err = std::fabs(total - 1.0);
        if (err > max_err) max_err = err;
    }
    printf("  max abs error         = %.3e\n", max_err);
    int ok = (max_err < 1e-60);
    printf("  Result: %s\n", ok ? "PASS" : "FAIL");

    free_gqdmatrix_dev(A_d);
    free_gqdmatrix_dev(B_d);
    free_gqdmatrix_dev(C_d);
    delete[] A_h; delete[] B_h; delete[] C_h;
    return ok ? 0 : 1;
}

/*----------------------------------------------------------------
 * Test 3: random N×N, naive GPU GEMM vs CPU qd_real triple loop
 *----------------------------------------------------------------*/
static int test_random_naive(long N, unsigned seed)
{
    printf("\n[Test 3] mul_gqdmatrix_dev vs CPU qd_real, N = %ld\n", N);

    qd_real  *A_cpu = new qd_real[N * N];
    qd_real  *B_cpu = new qd_real[N * N];
    qd_real  *C_cpu = new qd_real[N * N];
    gqd_real *A_h   = new gqd_real[N * N];
    gqd_real *B_h   = new gqd_real[N * N];
    gqd_real *C_h   = new gqd_real[N * N];

    fill_random(A_cpu, B_cpu, N, seed);
    cpu_qd_matmul(C_cpu, A_cpu, B_cpu, N);
    for (long i = 0; i < N * N; ++i) {
        A_h[i] = to_g(A_cpu[i]);
        B_h[i] = to_g(B_cpu[i]);
    }

    GQDMatrix A_d = init_gqdmatrix_dev(N, N);
    GQDMatrix B_d = init_gqdmatrix_dev(N, N);
    GQDMatrix C_d = init_gqdmatrix_dev(N, N);

    cudaMemcpy(A_d->element, A_h, N * N * sizeof(gqd_real),
               cudaMemcpyHostToDevice);
    cudaMemcpy(B_d->element, B_h, N * N * sizeof(gqd_real),
               cudaMemcpyHostToDevice);

    mul_gqdmatrix_dev(C_d, A_d, B_d, 128, 128);
    cudaDeviceSynchronize();

    cudaMemcpy(C_h, C_d->element, N * N * sizeof(gqd_real),
               cudaMemcpyDeviceToHost);

    double err = max_rel_err(C_h, C_cpu, N);
    printf("  max relative error    = %.3e\n", err);
    printf("  qd_real eps           = %.3e\n", to_double(qd_real::_eps));
    /* qd_real has ~63 digits.  Allow N * eps * 16 propagation slack. */
    double thresh = (double)N * to_double(qd_real::_eps) * 16.0;
    int ok = (err < thresh);
    printf("  Result: %s (threshold %.3e)\n", ok ? "PASS" : "FAIL", thresh);

    free_gqdmatrix_dev(A_d);
    free_gqdmatrix_dev(B_d);
    free_gqdmatrix_dev(C_d);
    delete[] A_cpu; delete[] B_cpu; delete[] C_cpu;
    delete[] A_h;   delete[] B_h;   delete[] C_h;
    return ok ? 0 : 1;
}

/*----------------------------------------------------------------
 * Test 4: random N×N, GPU Strassen vs CPU qd_real triple loop
 *----------------------------------------------------------------*/
static int test_random_strassen(long N, long min_dim, unsigned seed)
{
    printf("\n[Test 4] _bncuda_mul_gqdmatrix_strassen vs CPU qd_real, "
           "N = %ld, cutoff = %ld\n", N, min_dim);

    qd_real  *A_cpu = new qd_real[N * N];
    qd_real  *B_cpu = new qd_real[N * N];
    qd_real  *C_cpu = new qd_real[N * N];
    gqd_real *A_h   = new gqd_real[N * N];
    gqd_real *B_h   = new gqd_real[N * N];
    gqd_real *C_h   = new gqd_real[N * N];

    fill_random(A_cpu, B_cpu, N, seed);
    cpu_qd_matmul(C_cpu, A_cpu, B_cpu, N);
    for (long i = 0; i < N * N; ++i) {
        A_h[i] = to_g(A_cpu[i]);
        B_h[i] = to_g(B_cpu[i]);
    }

    GQDMatrix A_d = init_gqdmatrix_dev(N, N);
    GQDMatrix B_d = init_gqdmatrix_dev(N, N);
    GQDMatrix C_d = init_gqdmatrix_dev(N, N);

    cudaMemcpy(A_d->element, A_h, N * N * sizeof(gqd_real),
               cudaMemcpyHostToDevice);
    cudaMemcpy(B_d->element, B_h, N * N * sizeof(gqd_real),
               cudaMemcpyHostToDevice);

    _bncuda_reset_num_mul_gqdmatrix_strassen();
    _bncuda_mul_gqdmatrix_strassen(C_d, A_d, B_d, min_dim, 128, 128);
    cudaDeviceSynchronize();

    cudaMemcpy(C_h, C_d->element, N * N * sizeof(gqd_real),
               cudaMemcpyDeviceToHost);

    long num_addsub = 0, num_mul = 0;
    _bncuda_get_num_mul_gqdmatrix_strassen(&num_addsub, &num_mul);
    printf("  Strassen counters     : add/sub = %ld, mul = %ld\n",
           num_addsub, num_mul);

    double err = max_rel_err(C_h, C_cpu, N);
    printf("  max relative error    = %.3e\n", err);
    /* Strassen's extra additions amplify rounding more than naive GEMM. */
    double thresh = (double)N * to_double(qd_real::_eps) * 64.0;
    int ok = (err < thresh);
    printf("  Result: %s (threshold %.3e)\n", ok ? "PASS" : "FAIL", thresh);

    free_gqdmatrix_dev(A_d);
    free_gqdmatrix_dev(B_d);
    free_gqdmatrix_dev(C_d);
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
    printf(" gdtq + bncmatmul GPU GQD Strassen smoke test\n");
    printf("=============================================\n");
    printf("naive matmul N    = %ld\n", N_naive);
    printf("Strassen matmul N = %ld (cutoff %ld)\n", N_strassen, min_dim);

    GQDStart(0);

    int fails = 0;
    fails += test_init();
    fails += test_identity_naive(N_naive);
    fails += test_random_naive(N_naive, /*seed=*/12345);
    fails += test_random_strassen(N_strassen, min_dim, /*seed=*/67890);

    GQDEnd();
    fpu_fix_end(&cw);

    printf("\n=============================================\n");
    printf(" Total failures: %d\n", fails);
    printf("=============================================\n");
    return fails == 0 ? 0 : 1;
}
