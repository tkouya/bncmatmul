/* gqslu.cu -- GPU double-double LU decomposition (partial pivoting) + triangular
 * solve for BNCmatmul, using the gqd/gdtq device qs arithmetic.  Right-looking:
 * per pivot column k a 1-thread pivot search, a parallel row swap, and a parallel
 * trailing rank-1 update; then forward/back substitution.  Mirrors the CPU
 * QSLUdecompPM / SolveQSLSPM.  Operates on GQSMatrix/GQSVector device objects.
 *
 * Build (from repo root):
 *   nvcc -O3 -arch=sm_121 -std=c++17 -DCUDA_FMA -DUSE_QSLINEAR -DUSE_GMP -DUSE_MPFR \
 *        -Iinclude -I/usr/local/include/gdtq -I/usr/local/cuda/include \
 *        -diag-suppress 20011 -diag-suppress 177 -dc src/gqslu.cu -o gqslu.o
 *   nvcc -dlink gqslu.o cuda_obj/gqslinear.o -o dl.o
 *   nvcc ... gqslu.o cuda_obj/gqslinear.o dl.o -L. -lbncmatmul-0.24 <cuda+qd+mpfr libs>
 *   (aqs -DGQSLU_TEST for the self-test main)
 */
#include <cstdio>
#include <cmath>
#ifndef LU_THRESH
#define LU_THRESH 1e-28
#endif
#include "gdslinear.h"

__device__ static inline gqs_real gqs_abs (gqs_real a)
{ gqs_real z = a - a; return (a.x < 0.0) ? (z - a) : a; }

/* one-thread partial-pivot search over rows i = k..n-1 of column k */
__global__ static void gqslu_find_pivot (const gqs_real *A, int n, int k, int *imax)
{
  if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
  int p = k;
  gqs_real best = gqs_abs (A[(size_t) k * n + k]);
  for (int i = k + 1; i < n; ++i)
    {
      gqs_real v = gqs_abs (A[(size_t) i * n + k]);
      if (v > best) { best = v; p = i; }
    }
  *imax = p;
}

/* swap rows k and m of A (parallel over columns) and of b (thread 0) */
__global__ static void gqslu_swap_row (gqs_real *A, gqs_real *b, int n, int k, int m)
{
  int stride = gridDim.x * blockDim.x;
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += stride)
    { gqs_real t = A[(size_t) k * n + j]; A[(size_t) k * n + j] = A[(size_t) m * n + j]; A[(size_t) m * n + j] = t; }
  if (blockIdx.x * blockDim.x + threadIdx.x == 0)
    { gqs_real t = b[k]; b[k] = b[m]; b[m] = t; }
}

/* trailing update: for i>k, multiplier m=A[i][k]/A[k][k]; A[i][j]-=m*A[k][j] (j>k) */
__global__ static void gqslu_eliminate (gqs_real *A, int n, int k)
{
  int stride = gridDim.x * blockDim.x;
  for (int i = k + 1 + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
      gqs_real mik = A[(size_t) i * n + k] / A[(size_t) k * n + k];
      A[(size_t) i * n + k] = mik;
      for (int j = k + 1; j < n; ++j)
        A[(size_t) i * n + j] = A[(size_t) i * n + j] - mik * A[(size_t) k * n + j];
    }
}

/* forward (unit lower L) then back (upper U) substitution: solve in x (1 thread) */
__global__ static void gqslu_fbsub (const gqs_real *A, const gqs_real *b, gqs_real *x, int n)
{
  if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
  for (int i = 0; i < n; ++i)
    {
      gqs_real s = b[i];
      for (int j = 0; j < i; ++j) s = s - A[(size_t) i * n + j] * x[j];
      x[i] = s;                       /* unit diagonal of L */
    }
  for (int i = n - 1; i >= 0; --i)
    {
      gqs_real s = x[i];
      for (int j = i + 1; j < n; ++j) s = s - A[(size_t) i * n + j] * x[j];
      x[i] = s / A[(size_t) i * n + i];
    }
}

/* Host driver: A_dev is overwritten with its LU factors; solves A x = b_dev into
 * x_dev (all device GQS objects, n x n).  ch[] (host, length n) receives the pivot
 * row chosen at each step.  Returns 0 on success. */
extern "C" int
gqs_lu_solve_dev (GQSMatrix A_dev, GQSVector b_dev, GQSVector x_dev, long int *ch,
                  int blocks, int threads)
{
  int n = (int) A_dev->col_dim;
  if (blocks <= 0) blocks = 64;
  if (threads <= 0) threads = 128;
  int *d_imax, h_imax;
  cudaMalloc (&d_imax, sizeof (int));
  gqs_real *A = A_dev->element, *b = b_dev->element, *x = x_dev->element;

  for (int k = 0; k < n; ++k)
    {
      gqslu_find_pivot<<<1, 1>>> (A, n, k, d_imax);
      cudaMemcpy (&h_imax, d_imax, sizeof (int), cudaMemcpyDeviceToHost);
      if (ch) ch[k] = h_imax;
      if (h_imax != k) gqslu_swap_row<<<blocks, threads>>> (A, b, n, k, h_imax);
      gqslu_eliminate<<<blocks, threads>>> (A, n, k);
    }
  gqslu_fbsub<<<1, 1>>> (A, b, x, n);
  cudaError_t err = cudaDeviceSynchronize ();
  cudaFree (d_imax);
  return err == cudaSuccess ? 0 : (int) err;
}

#ifdef GQSLU_TEST
int main (void)
{
  long int i, j, dim = 100;

  QSMatrix a = init_qsmatrix (dim, dim);
  QSVector xtrue = init_qsvector (dim), b = init_qsvector (dim), xg = init_qsvector (dim);
  float t[QSSIZE];

  /* diagonally dominant A, known solution xtrue=[0,1,...] */
  for (i = 0; i < dim; i++) {
    for (j = 0; j < dim; j++) {
      if (i == j) rqs_set_ui (t, (unsigned long) (dim * dim));
      else { rqs_set_ui (t, (unsigned long) ((i + j) % 5 + 1)); rqs_sqrt (t, t); }
      set_qsmatrix_ij (a, i, j, t);
    }
    rqs_set_ui (t, (unsigned long) i); set_qsvector_i (xtrue, i, t);
  }
  mul_qsmatrix_qsvec (b, a, xtrue);          /* b = A * xtrue (CPU) */

  GQSMatrix ga = init_gqsmatrix_dev (dim, dim);
  GQSVector gb = init_gqsvector_dev (dim), gx = init_gqsvector_dev (dim);
  subst_gqsmatrix_dev_qsmat (ga, a);
  subst_gqsvector_dev_qsvec (gb, b);

  if (gqs_lu_solve_dev (ga, gb, gx, NULL, 64, 128)) { fprintf (stderr, "GPU LU failed\n"); return 1; }
  subst_qsvector_gqsvec_dev (xg, gx);        /* device -> host */

  double mr = 0.0; float d[QSSIZE], q[QSSIZE], cc[QSSIZE];
  for (i = 0; i < dim; i++)
    {
      qsfloat zg = get_qsvector_i_qsfloat (xg, i), zc = get_qsvector_i_qsfloat (xtrue, i);
      float *g = zg.val, *c = zc.val;
      rqs_sub (d, g, c); rqs_abs (d, d); rqs_abs (cc, c);
      if (cc[0] != 0.0) { rqs_div (q, d, cc); if (q[0] > mr) mr = q[0]; }
      else if (d[0] > mr) mr = d[0];
    }
  printf ("GPU gqs LU solve vs known x (dim=%ld): max relative error = %10.3e -> %s\n",
          dim, mr, mr < LU_THRESH ? "PASS (full quad-single precision)" : "FAIL");
  return mr < LU_THRESH ? 0 : 1;
}
#endif
