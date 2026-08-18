/* gmpflinear.cu -- CUDA multiple-precision (MPFR) real linear computation for
 * BNCmatmul.  Matrix multiply C = A*B with A,B,C real N x N (row-major double
 * I/O), each dot product accumulated at MPFR precision PREC on the GPU using the
 * mpc_cuda device port of MPFR.  Mirrors the CPU mul_mpfmatrix.
 *
 * Build (from repo root), linking the prebuilt mpc_cuda static library:
 *   MPC=/home/tkouya/na/cuda/mpc_cuda
 *   DEFS="$(grep '^-D' $MPC/tools/mpfr_cuda_defs.txt|tr '\n' ' ')"
 *   nvcc -x cu -dc -arch=sm_121 $DEFS -I$MPC/src/mpfr_cuda -I$MPC/include \
 *        -diag-suppress 20011 -fmad=false src/gmpflinear.cu -o gmpflinear.o
 *   nvcc -arch=sm_121 -rdc=true -diag-suppress 20011 -fmad=false \
 *        gmpflinear.o $MPC/build/libmpc_cuda.a -o gmpflinear   # -DGMPFLINEAR_TEST for the self-test
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
typedef long int gmp_randstate_t[1];
#include "mpfr.h"
#include "mpc_cuda/cuda_minigmp.h"
#include "mpc_cuda/cu_compat.h"

#ifndef PREC
#define PREC 1024
#endif
#ifndef GMPF_SLAB
#define GMPF_SLAB (16 * 1024)
#endif

/* C[i][j] = sum_k A[i*n+k] * B[k*n+j], accumulated in MPFR at PREC bits */
__host__ __device__ static double
gmpf_dot_ij (const double *A, const double *B, int n, int i, int j)
{
  MPFR_DECL_INIT (acc, PREC);
  MPFR_DECL_INIT (a, PREC);
  MPFR_DECL_INIT (b, PREC);
  MPFR_DECL_INIT (t, PREC);
  mpfr_set_zero (acc, 1);
  for (int k = 0; k < n; ++k)
    {
#ifdef __CUDA_ARCH__
      mpc_cuda_arena_reset ();
#endif
      mpfr_set_d (a, A[(size_t) i * n + k], MPFR_RNDN);
      mpfr_set_d (b, B[(size_t) k * n + j], MPFR_RNDN);
      mpfr_mul (t, a, b, MPFR_RNDN);
      mpfr_add (acc, acc, t, MPFR_RNDN);
    }
  return mpfr_get_d (acc, MPFR_RNDN);
}

__global__ static void
gmpf_matmul_kernel (int n, const double *A, const double *B, double *C)
{
  int stride = gridDim.x * blockDim.x;
  long total = (long) n * n;
  for (long e = blockIdx.x * blockDim.x + threadIdx.x; e < total; e += stride)
    C[e] = gmpf_dot_ij (A, B, n, (int) (e / n), (int) (e % n));
}

/* Host entry: C_host = A_host * B_host (all n x n row-major doubles), MPFR PREC
 * accumulation on the GPU.  Returns 0 on success. */
extern "C" int
cuda_mul_mpfmatrix (double *C_host, const double *A_host, const double *B_host,
                    int n, int lblocks, int lthreads)
{
  if (lblocks  <= 0) lblocks  = 256;
  if (lthreads <= 0) lthreads = 32;
  cudaDeviceSetLimit (cudaLimitMallocHeapSize, (size_t) 64 * 1024 * 1024);
  cudaDeviceSetLimit (cudaLimitStackSize,      (size_t) 64 * 1024);

  size_t M = (size_t) n * n * sizeof (double);
  double *dA, *dB, *dC;
  cudaMalloc (&dA, M); cudaMalloc (&dB, M); cudaMalloc (&dC, M);
  cudaMemcpy (dA, A_host, M, cudaMemcpyHostToDevice);
  cudaMemcpy (dB, B_host, M, cudaMemcpyHostToDevice);

  size_t ntot = (size_t) lblocks * lthreads;
  char *arena; size_t *top;
  cudaMalloc (&arena, ntot * (size_t) GMPF_SLAB);
  cudaMalloc (&top,   ntot * sizeof (size_t));
  cudaMemset (top, 0, ntot * sizeof (size_t));
  mpc_cuda_arena_base = arena; mpc_cuda_arena_slab = GMPF_SLAB; mpc_cuda_arena_top = top;
  cudaDeviceSynchronize ();

  gmpf_matmul_kernel<<<lblocks, lthreads>>> (n, dA, dB, dC);
  cudaError_t err = cudaDeviceSynchronize ();
  if (err == cudaSuccess)
    cudaMemcpy (C_host, dC, M, cudaMemcpyDeviceToHost);

  cudaFree (dA); cudaFree (dB); cudaFree (dC); cudaFree (arena); cudaFree (top);
  return err == cudaSuccess ? 0 : (int) err;
}

/* y_i = sum_j A[i*n+j] * x[j], MPFR PREC accumulation */
__host__ __device__ static double
gmpf_dot_row (const double *Arow, const double *x, int n)
{
  MPFR_DECL_INIT (acc, PREC);
  MPFR_DECL_INIT (a, PREC);
  MPFR_DECL_INIT (b, PREC);
  MPFR_DECL_INIT (t, PREC);
  mpfr_set_zero (acc, 1);
  for (int j = 0; j < n; ++j)
    {
#ifdef __CUDA_ARCH__
      mpc_cuda_arena_reset ();
#endif
      mpfr_set_d (a, Arow[j], MPFR_RNDN);
      mpfr_set_d (b, x[j], MPFR_RNDN);
      mpfr_mul (t, a, b, MPFR_RNDN);
      mpfr_add (acc, acc, t, MPFR_RNDN);
    }
  return mpfr_get_d (acc, MPFR_RNDN);
}

__global__ static void
gmpf_matvec_kernel (int n, const double *A, const double *x, double *y)
{
  int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    y[i] = gmpf_dot_row (A + (size_t) i * n, x, n);
}

/* Host entry: y_host = A_host * x_host  (A n x n row-major, x,y length n) */
extern "C" int
cuda_mul_mpfmatrix_vec (double *y_host, const double *A_host, const double *x_host,
                        int n, int lblocks, int lthreads)
{
  if (lblocks  <= 0) lblocks  = 256;
  if (lthreads <= 0) lthreads = 32;
  cudaDeviceSetLimit (cudaLimitMallocHeapSize, (size_t) 64 * 1024 * 1024);
  cudaDeviceSetLimit (cudaLimitStackSize,      (size_t) 64 * 1024);

  size_t M = (size_t) n * n * sizeof (double), V = (size_t) n * sizeof (double);
  double *dA, *dx, *dy;
  cudaMalloc (&dA, M); cudaMalloc (&dx, V); cudaMalloc (&dy, V);
  cudaMemcpy (dA, A_host, M, cudaMemcpyHostToDevice);
  cudaMemcpy (dx, x_host, V, cudaMemcpyHostToDevice);

  size_t ntot = (size_t) lblocks * lthreads;
  char *arena; size_t *top;
  cudaMalloc (&arena, ntot * (size_t) GMPF_SLAB);
  cudaMalloc (&top,   ntot * sizeof (size_t));
  cudaMemset (top, 0, ntot * sizeof (size_t));
  mpc_cuda_arena_base = arena; mpc_cuda_arena_slab = GMPF_SLAB; mpc_cuda_arena_top = top;
  cudaDeviceSynchronize ();

  gmpf_matvec_kernel<<<lblocks, lthreads>>> (n, dA, dx, dy);
  cudaError_t err = cudaDeviceSynchronize ();
  if (err == cudaSuccess)
    cudaMemcpy (y_host, dy, V, cudaMemcpyDeviceToHost);

  cudaFree (dA); cudaFree (dx); cudaFree (dy); cudaFree (arena); cudaFree (top);
  return err == cudaSuccess ? 0 : (int) err;
}

#ifdef GMPFLINEAR_TEST
int main (void)
{
  int N = 96;
  size_t M = (size_t) N * N * sizeof (double);
  double *A = (double *) malloc (M), *B = (double *) malloc (M);
  double *Cg = (double *) malloc (M), *Cc = (double *) malloc (M);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      { A[(size_t)i*N+j] = 1.0 + (i + 2*j)*1e-4; B[(size_t)i*N+j] = 0.5 + (2*i + j)*1e-4; }

  if (cuda_mul_mpfmatrix (Cg, A, B, N, 256, 32)) { fprintf (stderr, "GPU matmul failed\n"); return 1; }
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      Cc[(size_t)i*N+j] = gmpf_dot_ij (A, B, N, i, j);   /* CPU reference */

  double maxrel = 0.0; long tot = (long) N * N;
  for (long e = 0; e < tot; ++e)
    { double r = Cc[e] != 0.0 ? fabs ((Cg[e]-Cc[e])/Cc[e]) : fabs (Cg[e]); if (r > maxrel) maxrel = r; }
  printf ("CUDA MPFR matmul (N=%d, prec=%d): max relative |GPU-CPU| = %.3e  -> %s\n",
          N, PREC, maxrel, maxrel < 1e-13 ? "PASS" : "FAIL");

  /* matvec y = A*x */
  double *x = (double *) malloc ((size_t) N * sizeof (double));
  double *yg = (double *) malloc ((size_t) N * sizeof (double));
  double *yc = (double *) malloc ((size_t) N * sizeof (double));
  for (int i = 0; i < N; ++i) x[i] = 1.0 + i * 1e-3;
  cuda_mul_mpfmatrix_vec (yg, A, x, N, 256, 32);
  for (int i = 0; i < N; ++i) yc[i] = gmpf_dot_row (A + (size_t) i * N, x, N);
  double mv = 0.0;
  for (int i = 0; i < N; ++i) { double r = yc[i]!=0.0 ? fabs((yg[i]-yc[i])/yc[i]) : fabs(yg[i]); if (r>mv) mv=r; }
  printf ("CUDA MPFR matvec (N=%d, prec=%d): max relative |GPU-CPU| = %.3e  -> %s\n",
          N, PREC, mv, mv < 1e-13 ? "PASS" : "FAIL");

  free (A); free (B); free (Cg); free (Cc); free (x); free (yg); free (yc);
  return (maxrel < 1e-13 && mv < 1e-13) ? 0 : 1;
}
#endif
