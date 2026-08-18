/* mpflinear_cu.cu -- GPU (CUDA) multiple-precision real linear computation for
 * BNCmatmul.  Matrix multiply, matrix-vector multiply and LU-based linear-system
 * solve, all keeping every element at a fixed compile-time precision PREC bits
 * on the device.  Mirrors the CPU mpflinear.c (mul_mpfmatrix etc.).
 *
 * Numerical core: the register-resident cu_freal<PREC> type from mpc_cuda
 * (include/mpc_cuda/cu_freal.cuh) -- a header-only, value-semantic multiple-
 * precision real whose significand lives in registers.  Because it carries no
 * pointers/limb allocations, an n x n matrix can be stored on the device at full
 * precision (cu_freal array) and Gaussian elimination updates it in place WITHOUT
 * the double round-off that the double-in/out matmul representation would incur.
 * This is what makes a full-precision GPU LU feasible.
 *
 * cu_freal provides +,-,* (correctly rounded at PREC) but NO division.  Division
 * (needed only by LU) is built here via Newton-Raphson reciprocal using mul/sub
 * (cu_frecip): x_{n+1} = x_n*(2 - b*x_n), seeded from the double reciprocal.
 *
 * This translation unit is header-only with respect to mpc_cuda: it needs
 * -I<mpc_cuda>/include at compile time and links WITHOUT libmpc_cuda.a, so it
 * device-links cleanly alongside the g[dtq][ds]linear_cu objects.
 *
 * Build (standalone self-test):
 *   nvcc -arch=sm_121 -O3 -fmad=false -I<mpc_cuda>/include -DMPFLINEAR_CU_TEST \
 *        src/mpflinear_cu.cu -lmpfr -lgmp -o mpflinear_cu_test
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "mpc_cuda/cu_freal.cuh"
using namespace cu_fp;

#ifndef PREC
#define PREC 1024            /* GPU working precision in bits (multiple of 32) */
#endif

typedef cu_freal<PREC> F;

/* ----------------------------------------------------------------------------
 * Device arithmetic helpers
 * ------------------------------------------------------------------------- */

/* Number of Newton steps to reach PREC bits starting from a ~53-bit double seed
 * (each step roughly doubles the count of correct bits). */
__host__ __device__ static inline int cu_recip_iters()
{
  int iters = 1;
  for (int p = 53; p < PREC; p <<= 1) iters++;
  return iters + 1;          /* +1 guard step */
}

/* r = 1 / b  (b != 0), accurate to ~PREC bits. */
__host__ __device__ static F cu_frecip(const F &b)
{
  F two = F::from_double(2.0);
  F x   = F::from_double(1.0 / b.to_double());      /* ~53-bit seed */
  int iters = cu_recip_iters();
  for (int it = 0; it < iters; ++it)
    {
      F bx = cu_fmul<PREC>(b, x);
      F t  = cu_fsub<PREC>(two, bx);
      x    = cu_fmul<PREC>(x, t);
    }
  return x;
}

/* a / b */
__host__ __device__ static inline F cu_fdiv(const F &a, const F &b)
{
  return cu_fmul<PREC>(a, cu_frecip(b));
}

/* ----------------------------------------------------------------------------
 * double <-> cu_freal conversion kernels
 * ------------------------------------------------------------------------- */
__global__ static void mpf_d2f_kernel(long total, const double *d, F *f)
{
  for (long e = blockIdx.x * blockDim.x + threadIdx.x; e < total;
       e += (long) gridDim.x * blockDim.x)
    f[e] = F::from_double(d[e]);
}
__global__ static void mpf_f2d_kernel(long total, const F *f, double *d)
{
  for (long e = blockIdx.x * blockDim.x + threadIdx.x; e < total;
       e += (long) gridDim.x * blockDim.x)
    d[e] = f[e].to_double();
}

/* ----------------------------------------------------------------------------
 * Matrix multiply: C[i*n+j] = sum_k A[i*n+k] * B[k*n+j]
 * ------------------------------------------------------------------------- */
__global__ static void mpf_matmul_kernel(int n, const double *A, const double *B,
                                         double *C)
{
  long stride = (long) gridDim.x * blockDim.x;
  long total  = (long) n * n;
  for (long e = blockIdx.x * blockDim.x + threadIdx.x; e < total; e += stride)
    {
      int i = (int) (e / n), j = (int) (e % n);
      F acc; acc.set_zero();
      for (int k = 0; k < n; ++k)
        acc = cu_fadd<PREC>(acc,
                cu_fmul<PREC>(F::from_double(A[(size_t) i * n + k]),
                              F::from_double(B[(size_t) k * n + j])));
      C[e] = acc.to_double();
    }
}

/* ----------------------------------------------------------------------------
 * Matrix-vector multiply: y[i] = sum_k A[i*n+k] * x[k]
 * ------------------------------------------------------------------------- */
__global__ static void mpf_matvec_kernel(int n, const double *A, const double *x,
                                         double *y)
{
  long stride = (long) gridDim.x * blockDim.x;
  for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
      F acc; acc.set_zero();
      for (int k = 0; k < n; ++k)
        acc = cu_fadd<PREC>(acc,
                cu_fmul<PREC>(F::from_double(A[(size_t) i * n + k]),
                              F::from_double(x[k])));
      y[i] = acc.to_double();
    }
}

/* ----------------------------------------------------------------------------
 * LU solve (single cooperative block).  Right-looking elimination with partial
 * pivoting, then forward/back substitution.  A,b stored as full-precision F.
 * status[0] != 0 signals a (near-)singular pivot.
 * ------------------------------------------------------------------------- */
__global__ static void mpf_lu_solve_kernel(int n, F *A, F *b, F *x, int *status)
{
  int t = threadIdx.x, nt = blockDim.x;
  __shared__ int piv;
  if (t == 0) status[0] = 0;
  __syncthreads();

  for (int k = 0; k < n; ++k)
    {
      if (t == 0)
        {                                       /* partial pivot search */
          int p = k; F mx = cu_abs<PREC>(A[(size_t) k * n + k]);
          for (int i = k + 1; i < n; ++i)
            {
              F v = cu_abs<PREC>(A[(size_t) i * n + k]);
              if (cu_cmp<PREC>(v, mx) > 0) { mx = v; p = i; }
            }
          piv = p;
          if (A[(size_t) p * n + k].is_zero()) status[0] = k + 1;
        }
      __syncthreads();
      int p = piv;
      if (p != k)                               /* swap rows k,p (and b) */
        {
          for (int j = t; j < n; j += nt)
            { F tmp = A[(size_t) k * n + j];
              A[(size_t) k * n + j] = A[(size_t) p * n + j];
              A[(size_t) p * n + j] = tmp; }
          if (t == 0) { F tmp = b[k]; b[k] = b[p]; b[p] = tmp; }
        }
      __syncthreads();

      F rpiv = cu_frecip(A[(size_t) k * n + k]);
      for (int i = k + 1 + t; i < n; i += nt)   /* multipliers + trailing update */
        {
          F mik = cu_fmul<PREC>(A[(size_t) i * n + k], rpiv);
          A[(size_t) i * n + k] = mik;
          for (int j = k + 1; j < n; ++j)
            A[(size_t) i * n + j] =
              cu_fsub<PREC>(A[(size_t) i * n + j],
                            cu_fmul<PREC>(mik, A[(size_t) k * n + j]));
          b[i] = cu_fsub<PREC>(b[i], cu_fmul<PREC>(mik, b[k]));
        }
      __syncthreads();
    }

  if (t == 0)                                   /* back substitution (U x = b') */
    for (int i = n - 1; i >= 0; --i)
      {
        F s = b[i];
        for (int j = i + 1; j < n; ++j)
          s = cu_fsub<PREC>(s, cu_fmul<PREC>(A[(size_t) i * n + j], x[j]));
        x[i] = cu_fdiv(s, A[(size_t) i * n + i]);
      }
}

/* ============================================================================
 * Host entry points (C ABI)
 * ========================================================================= */
extern "C" int
cuda_mul_mpfmatrix(double *C_host, const double *A_host, const double *B_host,
                   int n, int lblocks, int lthreads)
{
  if (lblocks  <= 0) lblocks  = 256;
  if (lthreads <= 0) lthreads = 32;
  size_t M = (size_t) n * n * sizeof(double);
  double *dA, *dB, *dC;
  cudaMalloc(&dA, M); cudaMalloc(&dB, M); cudaMalloc(&dC, M);
  cudaMemcpy(dA, A_host, M, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B_host, M, cudaMemcpyHostToDevice);
  mpf_matmul_kernel<<<lblocks, lthreads>>>(n, dA, dB, dC);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  cudaMemcpy(C_host, dC, M, cudaMemcpyDeviceToHost);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  return err ? (int) err : 0;
}

extern "C" int
cuda_mul_mpfmatrix_vec(double *y_host, const double *A_host, const double *x_host,
                       int n, int lblocks, int lthreads)
{
  if (lblocks  <= 0) lblocks  = 256;
  if (lthreads <= 0) lthreads = 32;
  size_t MA = (size_t) n * n * sizeof(double), MV = (size_t) n * sizeof(double);
  double *dA, *dx, *dy;
  cudaMalloc(&dA, MA); cudaMalloc(&dx, MV); cudaMalloc(&dy, MV);
  cudaMemcpy(dA, A_host, MA, cudaMemcpyHostToDevice);
  cudaMemcpy(dx, x_host, MV, cudaMemcpyHostToDevice);
  mpf_matvec_kernel<<<lblocks, lthreads>>>(n, dA, dx, dy);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  cudaMemcpy(y_host, dy, MV, cudaMemcpyDeviceToHost);
  cudaFree(dA); cudaFree(dx); cudaFree(dy);
  return err ? (int) err : 0;
}

extern "C" int
cuda_mpf_lu_solve(double *x_host, const double *A_host, const double *b_host,
                  int n, int lblocks, int lthreads)
{
  (void) lblocks;                               /* factorization is single-block */
  if (lthreads <= 0) lthreads = 128;
  /* cu_frecip's loop makes the kernel stack non-static; give it headroom. */
  cudaDeviceSetLimit(cudaLimitStackSize, (size_t) 32 * 1024);

  size_t MA = (size_t) n * n * sizeof(double), MV = (size_t) n * sizeof(double);
  double *dAd, *dbd; F *dA, *db, *dx; int *dstat;
  cudaMalloc(&dAd, MA); cudaMalloc(&dbd, MV);
  cudaMalloc(&dA, (size_t) n * n * sizeof(F));
  cudaMalloc(&db, (size_t) n * sizeof(F));
  cudaMalloc(&dx, (size_t) n * sizeof(F));
  cudaMalloc(&dstat, sizeof(int));
  cudaMemcpy(dAd, A_host, MA, cudaMemcpyHostToDevice);
  cudaMemcpy(dbd, b_host, MV, cudaMemcpyHostToDevice);

  int blk = (int) (((long) n * n + lthreads - 1) / lthreads);
  if (blk < 1) blk = 1;
  mpf_d2f_kernel<<<blk, lthreads>>>((long) n * n, dAd, dA);
  mpf_d2f_kernel<<<1, lthreads>>>((long) n, dbd, db);
  mpf_lu_solve_kernel<<<1, lthreads>>>(n, dA, db, dx, dstat);
  mpf_f2d_kernel<<<1, lthreads>>>((long) n, dx, dbd);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  int stat = 0;
  cudaMemcpy(&stat, dstat, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(x_host, dbd, MV, cudaMemcpyDeviceToHost);

  cudaFree(dAd); cudaFree(dbd); cudaFree(dA); cudaFree(db); cudaFree(dx);
  cudaFree(dstat);
  return err ? (int) err : stat;
}

/* ============================================================================
 * Optional self-test: GPU vs CPU MPFR.  Build with -DMPFLINEAR_CU_TEST.
 * ========================================================================= */
#ifdef MPFLINEAR_CU_TEST
#include <gmp.h>
#include <mpfr.h>

static void cpu_lu_solve(int n, const double *A0, const double *b0, mpfr_t *xout)
{
  mpfr_t *A = (mpfr_t *) malloc((size_t) n * n * sizeof(mpfr_t));
  mpfr_t *b = (mpfr_t *) malloc((size_t) n * sizeof(mpfr_t));
  for (int i = 0; i < n * n; i++) { mpfr_init2(A[i], PREC); mpfr_set_d(A[i], A0[i], MPFR_RNDN); }
  for (int i = 0; i < n; i++)     { mpfr_init2(b[i], PREC); mpfr_set_d(b[i], b0[i], MPFR_RNDN); }
  mpfr_t mik, t; mpfr_init2(mik, PREC); mpfr_init2(t, PREC);
  for (int k = 0; k < n; k++) {
    int p = k;
    for (int i = k + 1; i < n; i++) if (mpfr_cmpabs(A[i*n+k], A[p*n+k]) > 0) p = i;
    if (p != k) { for (int j = 0; j < n; j++) mpfr_swap(A[k*n+j], A[p*n+j]); mpfr_swap(b[k], b[p]); }
    for (int i = k + 1; i < n; i++) {
      mpfr_div(mik, A[i*n+k], A[k*n+k], MPFR_RNDN); mpfr_set(A[i*n+k], mik, MPFR_RNDN);
      for (int j = k + 1; j < n; j++) { mpfr_mul(t, mik, A[k*n+j], MPFR_RNDN); mpfr_sub(A[i*n+j], A[i*n+j], t, MPFR_RNDN); }
      mpfr_mul(t, mik, b[k], MPFR_RNDN); mpfr_sub(b[i], b[i], t, MPFR_RNDN);
    }
  }
  for (int i = n - 1; i >= 0; i--) {
    mpfr_set(xout[i], b[i], MPFR_RNDN);
    for (int j = i + 1; j < n; j++) { mpfr_mul(t, A[i*n+j], xout[j], MPFR_RNDN); mpfr_sub(xout[i], xout[i], t, MPFR_RNDN); }
    mpfr_div(xout[i], xout[i], A[i*n+i], MPFR_RNDN);
  }
  for (int i = 0; i < n*n; i++) mpfr_clear(A[i]);
  for (int i = 0; i < n; i++)   mpfr_clear(b[i]);
  mpfr_clear(mik); mpfr_clear(t); free(A); free(b);
}

int main(int argc, char **argv)
{
  int n = argc > 1 ? atoi(argv[1]) : 64;
  srand(20240618);
  double *A = (double *) malloc((size_t) n*n*sizeof(double));
  double *B = (double *) malloc((size_t) n*n*sizeof(double));
  double *b = (double *) malloc((size_t) n*sizeof(double));
  for (int i = 0; i < n; i++) {
    double rs = 0;
    for (int j = 0; j < n; j++) { A[i*n+j] = (rand()%2000-1000)/100.0; B[i*n+j]=(rand()%2000-1000)/100.0; rs += fabs(A[i*n+j]); }
    A[i*n+i] = (A[i*n+i] >= 0 ? 1 : -1) * (rs + 1.0);   /* diag dominant */
    b[i] = (rand()%2000-1000)/100.0;
  }

  /* ---- matmul ---- */
  double *Cg = (double *) malloc((size_t) n*n*sizeof(double));
  cuda_mul_mpfmatrix(Cg, A, B, n, 256, 32);
  mpfr_t acc, ta, tb; mpfr_init2(acc, PREC); mpfr_init2(ta, PREC); mpfr_init2(tb, PREC);
  double mmrel = 0;
  for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
    mpfr_set_zero(acc, 1);
    for (int k = 0; k < n; k++) { mpfr_set_d(ta, A[i*n+k], MPFR_RNDN); mpfr_set_d(tb, B[k*n+j], MPFR_RNDN);
      mpfr_mul(ta, ta, tb, MPFR_RNDN); mpfr_add(acc, acc, ta, MPFR_RNDN); }
    double c = mpfr_get_d(acc, MPFR_RNDN), g = Cg[i*n+j];
    double r = fabs(c) > 1e-300 ? fabs(g-c)/fabs(c) : fabs(g-c);
    if (r > mmrel) mmrel = r;
  }
  printf("matmul  n=%d PREC=%d  max rel err = %.3e\n", n, PREC, mmrel);

  /* ---- matvec ---- */
  double *yg = (double *) malloc((size_t) n*sizeof(double));
  cuda_mul_mpfmatrix_vec(yg, A, b, n, 256, 32);
  double mvrel = 0;
  for (int i = 0; i < n; i++) {
    mpfr_set_zero(acc, 1);
    for (int k = 0; k < n; k++) { mpfr_set_d(ta, A[i*n+k], MPFR_RNDN); mpfr_set_d(tb, b[k], MPFR_RNDN);
      mpfr_mul(ta, ta, tb, MPFR_RNDN); mpfr_add(acc, acc, ta, MPFR_RNDN); }
    double c = mpfr_get_d(acc, MPFR_RNDN), g = yg[i];
    double r = fabs(c) > 1e-300 ? fabs(g-c)/fabs(c) : fabs(g-c);
    if (r > mvrel) mvrel = r;
  }
  printf("matvec  n=%d PREC=%d  max rel err = %.3e\n", n, PREC, mvrel);

  /* ---- LU solve ---- */
  double *xg = (double *) malloc((size_t) n*sizeof(double));
  int st = cuda_mpf_lu_solve(xg, A, b, n, 0, 128);
  mpfr_t *xc = (mpfr_t *) malloc((size_t) n*sizeof(mpfr_t));
  for (int i = 0; i < n; i++) mpfr_init2(xc[i], PREC);
  cpu_lu_solve(n, A, b, xc);
  double lurel = 0;
  for (int i = 0; i < n; i++) {
    double c = mpfr_get_d(xc[i], MPFR_RNDN), g = xg[i];
    double r = fabs(c) > 1e-300 ? fabs(g-c)/fabs(c) : fabs(g-c);
    if (r > lurel) lurel = r;
  }
  printf("LUsolve n=%d PREC=%d  max rel err = %.3e (status=%d)\n", n, PREC, lurel, st);

  return (mmrel < 1e-290 && mvrel < 1e-290 && lurel < 1e-12) ? 0 : 1;
}
#endif /* MPFLINEAR_CU_TEST */
