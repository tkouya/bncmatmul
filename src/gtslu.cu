/* gtslu.cu -- GPU double-double LU decomposition (partial pivoting) + triangular
 * solve for BNCmatmul, using the gqd/gdtq device ts arithmetic.  Right-looking:
 * per pivot column k a 1-thread pivot search, a parallel row swap, and a parallel
 * trailing rank-1 update; then forward/back substitution.  Mirrors the CPU
 * TSLUdecompPM / SolveTSLSPM.  Operates on GTSMatrix/GTSVector device objects.
 *
 * Build (from repo root):
 *   nvcc -O3 -arch=sm_121 -std=c++17 -DCUDA_FMA -DUSE_TSLINEAR -DUSE_GMP -DUSE_MPFR \
 *        -Iinclude -I/usr/local/include/gdtq -I/usr/local/cuda/include \
 *        -diag-suppress 20011 -diag-suppress 177 -dc src/gtslu.cu -o gtslu.o
 *   nvcc -dlink gtslu.o cuda_obj/gtslinear.o -o dl.o
 *   nvcc ... gtslu.o cuda_obj/gtslinear.o dl.o -L. -lbncmatmul-0.24 <cuda+qd+mpfr libs>
 *   (ats -DGTSLU_TEST for the self-test main)
 */
#include <cstdio>
#include <cmath>
#ifndef LU_THRESH
#define LU_THRESH 1e-28
#endif
#include "gtslinear.h"

__device__ static inline gts_real gts_abs (gts_real a)
{ gts_real z = a - a; return (a.x < 0.0) ? (z - a) : a; }

/* one-thread partial-pivot search over rows i = k..n-1 of column k */
__global__ static void gtslu_find_pivot (const gts_real *A, int n, int k, int *imax)
{
  if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
  int p = k;
  gts_real best = gts_abs (A[(size_t) k * n + k]);
  for (int i = k + 1; i < n; ++i)
    {
      gts_real v = gts_abs (A[(size_t) i * n + k]);
      if (v > best) { best = v; p = i; }
    }
  *imax = p;
}

/* swap rows k and m of A (parallel over columns) and of b (thread 0) */
__global__ static void gtslu_swap_row (gts_real *A, gts_real *b, int n, int k, int m)
{
  int stride = gridDim.x * blockDim.x;
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += stride)
    { gts_real t = A[(size_t) k * n + j]; A[(size_t) k * n + j] = A[(size_t) m * n + j]; A[(size_t) m * n + j] = t; }
  if (blockIdx.x * blockDim.x + threadIdx.x == 0)
    { gts_real t = b[k]; b[k] = b[m]; b[m] = t; }
}

/* trailing update: for i>k, multiplier m=A[i][k]/A[k][k]; A[i][j]-=m*A[k][j] (j>k) */
__global__ static void gtslu_eliminate (gts_real *A, int n, int k)
{
  int stride = gridDim.x * blockDim.x;
  for (int i = k + 1 + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
      gts_real mik = A[(size_t) i * n + k] / A[(size_t) k * n + k];
      A[(size_t) i * n + k] = mik;
      for (int j = k + 1; j < n; ++j)
        A[(size_t) i * n + j] = A[(size_t) i * n + j] - mik * A[(size_t) k * n + j];
    }
}

/* forward (unit lower L) then back (upper U) substitution: solve in x (1 thread) */
__global__ static void gtslu_fbsub (const gts_real *A, const gts_real *b, gts_real *x, int n)
{
  if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
  for (int i = 0; i < n; ++i)
    {
      gts_real s = b[i];
      for (int j = 0; j < i; ++j) s = s - A[(size_t) i * n + j] * x[j];
      x[i] = s;                       /* unit diagonal of L */
    }
  for (int i = n - 1; i >= 0; --i)
    {
      gts_real s = x[i];
      for (int j = i + 1; j < n; ++j) s = s - A[(size_t) i * n + j] * x[j];
      x[i] = s / A[(size_t) i * n + i];
    }
}

/* Host driver: A_dev is overwritten with its LU factors; solves A x = b_dev into
 * x_dev (all device GTS objects, n x n).  ch[] (host, length n) receives the pivot
 * row chosen at each step.  Returns 0 on success. */
extern "C" int
gts_lu_solve_dev (GTSMatrix A_dev, GTSVector b_dev, GTSVector x_dev, long int *ch,
                  int blocks, int threads)
{
  int n = (int) A_dev->col_dim;
  if (blocks <= 0) blocks = 64;
  if (threads <= 0) threads = 128;
  int *d_imax, h_imax;
  cudaMalloc (&d_imax, sizeof (int));
  gts_real *A = A_dev->element, *b = b_dev->element, *x = x_dev->element;

  for (int k = 0; k < n; ++k)
    {
      gtslu_find_pivot<<<1, 1>>> (A, n, k, d_imax);
      cudaMemcpy (&h_imax, d_imax, sizeof (int), cudaMemcpyDeviceToHost);
      if (ch) ch[k] = h_imax;
      if (h_imax != k) gtslu_swap_row<<<blocks, threads>>> (A, b, n, k, h_imax);
      gtslu_eliminate<<<blocks, threads>>> (A, n, k);
    }
  gtslu_fbsub<<<1, 1>>> (A, b, x, n);
  cudaError_t err = cudaDeviceSynchronize ();
  cudaFree (d_imax);
  return err == cudaSuccess ? 0 : (int) err;
}

#ifdef GTSLU_TEST
int main (void)
{
  long int i, j, dim = 100;

  TSMatrix a = init_tsmatrix (dim, dim);
  TSVector xtrue = init_tsvector (dim), b = init_tsvector (dim), xg = init_tsvector (dim);
  float t[TSSIZE];

  /* diagonally dominant A, known solution xtrue=[0,1,...] */
  for (i = 0; i < dim; i++) {
    for (j = 0; j < dim; j++) {
      if (i == j) rts_set_ui (t, (unsigned long) (dim * dim));
      else { rts_set_ui (t, (unsigned long) ((i + j) % 5 + 1)); rts_sqrt (t, t); }
      set_tsmatrix_ij (a, i, j, t);
    }
    rts_set_ui (t, (unsigned long) i); set_tsvector_i (xtrue, i, t);
  }
  mul_tsmatrix_tsvec (b, a, xtrue);          /* b = A * xtrue (CPU) */

  GTSMatrix ga = init_gtsmatrix_dev (dim, dim);
  GTSVector gb = init_gtsvector_dev (dim), gx = init_gtsvector_dev (dim);
  subst_gtsmatrix_dev_tsmat (ga, a);
  subst_gtsvector_dev_tsvec (gb, b);

  if (gts_lu_solve_dev (ga, gb, gx, NULL, 64, 128)) { fprintf (stderr, "GPU LU failed\n"); return 1; }
  subst_tsvector_gtsvec_dev (xg, gx);        /* device -> host */

  double mr = 0.0; float d[TSSIZE], q[TSSIZE], cc[TSSIZE];
  for (i = 0; i < dim; i++)
    {
      /* get_tsvector_i is a macro returning a pointer to a TEMPORARY's array
         member (dangling); read into locals via the typed accessor instead. */
      tsfloat zg = get_tsvector_i_tsfloat (xg, i), zc = get_tsvector_i_tsfloat (xtrue, i);
      float *g = zg.val, *c = zc.val;
      rts_sub (d, g, c); rts_abs (d, d); rts_abs (cc, c);
      if (cc[0] != 0.0) { rts_div (q, d, cc); if (q[0] > mr) mr = q[0]; }
      else if (d[0] > mr) mr = d[0];
    }
  printf ("GPU gts LU solve vs known x (dim=%ld): max relative error = %10.3e -> %s\n",
          dim, mr, mr < LU_THRESH ? "PASS (full triple-single precision)" : "FAIL");
  return mr < LU_THRESH ? 0 : 1;
}
#endif
