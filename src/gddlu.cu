/* gddlu.cu -- GPU double-double LU decomposition (partial pivoting) + triangular
 * solve for BNCmatmul, using the gqd/gdtq device dd arithmetic.  Right-looking:
 * per pivot column k a 1-thread pivot search, a parallel row swap, and a parallel
 * trailing rank-1 update; then forward/back substitution.  Mirrors the CPU
 * DDLUdecompPM / SolveDDLSPM.  Operates on GDDMatrix/GDDVector device objects.
 *
 * Build (from repo root):
 *   nvcc -O3 -arch=sm_121 -std=c++17 -DCUDA_FMA -DUSE_DDLINEAR -DUSE_GMP -DUSE_MPFR \
 *        -Iinclude -I/usr/local/include/gdtq -I/usr/local/cuda/include \
 *        -diag-suppress 20011 -diag-suppress 177 -dc src/gddlu.cu -o gddlu.o
 *   nvcc -dlink gddlu.o cuda_obj/gddlinear.o -o dl.o
 *   nvcc ... gddlu.o cuda_obj/gddlinear.o dl.o -L. -lbncmatmul-0.24 <cuda+qd+mpfr libs>
 *   (add -DGDDLU_TEST for the self-test main)
 */
#include <cstdio>
#include <cmath>
#ifndef LU_THRESH
#define LU_THRESH 1e-28
#endif
#include "gddlinear.h"

__device__ static inline gdd_real gdd_abs (gdd_real a)
{ gdd_real z = a - a; return (a.x < 0.0) ? (z - a) : a; }

/* one-thread partial-pivot search over rows i = k..n-1 of column k */
__global__ static void gddlu_find_pivot (const gdd_real *A, int n, int k, int *imax)
{
  if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
  int p = k;
  gdd_real best = gdd_abs (A[(size_t) k * n + k]);
  for (int i = k + 1; i < n; ++i)
    {
      gdd_real v = gdd_abs (A[(size_t) i * n + k]);
      if (v > best) { best = v; p = i; }
    }
  *imax = p;
}

/* swap rows k and m of A (parallel over columns) and of b (thread 0) */
__global__ static void gddlu_swap_row (gdd_real *A, gdd_real *b, int n, int k, int m)
{
  int stride = gridDim.x * blockDim.x;
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += stride)
    { gdd_real t = A[(size_t) k * n + j]; A[(size_t) k * n + j] = A[(size_t) m * n + j]; A[(size_t) m * n + j] = t; }
  if (blockIdx.x * blockDim.x + threadIdx.x == 0)
    { gdd_real t = b[k]; b[k] = b[m]; b[m] = t; }
}

/* trailing update: for i>k, multiplier m=A[i][k]/A[k][k]; A[i][j]-=m*A[k][j] (j>k) */
__global__ static void gddlu_eliminate (gdd_real *A, int n, int k)
{
  int stride = gridDim.x * blockDim.x;
  for (int i = k + 1 + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
      gdd_real mik = A[(size_t) i * n + k] / A[(size_t) k * n + k];
      A[(size_t) i * n + k] = mik;
      for (int j = k + 1; j < n; ++j)
        A[(size_t) i * n + j] = A[(size_t) i * n + j] - mik * A[(size_t) k * n + j];
    }
}

/* forward (unit lower L) then back (upper U) substitution: solve in x (1 thread) */
__global__ static void gddlu_fbsub (const gdd_real *A, const gdd_real *b, gdd_real *x, int n)
{
  if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
  for (int i = 0; i < n; ++i)
    {
      gdd_real s = b[i];
      for (int j = 0; j < i; ++j) s = s - A[(size_t) i * n + j] * x[j];
      x[i] = s;                       /* unit diagonal of L */
    }
  for (int i = n - 1; i >= 0; --i)
    {
      gdd_real s = x[i];
      for (int j = i + 1; j < n; ++j) s = s - A[(size_t) i * n + j] * x[j];
      x[i] = s / A[(size_t) i * n + i];
    }
}

/* Host driver: A_dev is overwritten with its LU factors; solves A x = b_dev into
 * x_dev (all device GDD objects, n x n).  ch[] (host, length n) receives the pivot
 * row chosen at each step.  Returns 0 on success. */
extern "C" int
gdd_lu_solve_dev (GDDMatrix A_dev, GDDVector b_dev, GDDVector x_dev, long int *ch,
                  int blocks, int threads)
{
  int n = (int) A_dev->col_dim;
  if (blocks <= 0) blocks = 64;
  if (threads <= 0) threads = 128;
  int *d_imax, h_imax;
  cudaMalloc (&d_imax, sizeof (int));
  gdd_real *A = A_dev->element, *b = b_dev->element, *x = x_dev->element;

  for (int k = 0; k < n; ++k)
    {
      gddlu_find_pivot<<<1, 1>>> (A, n, k, d_imax);
      cudaMemcpy (&h_imax, d_imax, sizeof (int), cudaMemcpyDeviceToHost);
      if (ch) ch[k] = h_imax;
      if (h_imax != k) gddlu_swap_row<<<blocks, threads>>> (A, b, n, k, h_imax);
      gddlu_eliminate<<<blocks, threads>>> (A, n, k);
    }
  gddlu_fbsub<<<1, 1>>> (A, b, x, n);
  cudaError_t err = cudaDeviceSynchronize ();
  cudaFree (d_imax);
  return err == cudaSuccess ? 0 : (int) err;
}

#ifdef GDDLU_TEST
int main (void)
{
  long int i, j, dim = 100;
  fpu_fix_start (NULL);

  DDMatrix a = init_ddmatrix (dim, dim);
  DDVector xtrue = init_ddvector (dim), b = init_ddvector (dim), xg = init_ddvector (dim);
  double t[DDSIZE];

  /* diagonally dominant A, known solution xtrue=[0,1,...] */
  frank_ddmatrix (a, dim);                    /* Frank matrix: needs partial pivoting */
  for (i = 0; i < dim; i++) { rdd_set_ui (t, (unsigned long) i); set_ddvector_i (xtrue, i, t); }
  mul_ddmatrix_ddvec (b, a, xtrue);          /* b = A * xtrue (CPU) */

  GDDMatrix ga = init_gddmatrix_dev (dim, dim);
  GDDVector gb = init_gddvector_dev (dim), gx = init_gddvector_dev (dim);
  subst_gddmatrix_dev_ddmat (ga, a);
  subst_gddvector_dev_ddvec (gb, b);

  if (gdd_lu_solve_dev (ga, gb, gx, NULL, 64, 128)) { fprintf (stderr, "GPU LU failed\n"); return 1; }
  subst_ddvector_gddvec_dev (xg, gx);        /* device -> host */

  double mr = 0.0, d[DDSIZE], q[DDSIZE], cc[DDSIZE];
  for (i = 0; i < dim; i++)
    {
      double *g = get_ddvector_i (xg, i), *c = get_ddvector_i (xtrue, i);
      rdd_sub (d, g, c); rdd_abs (d, d); rdd_abs (cc, c);
      if (cc[0] != 0.0) { rdd_div (q, d, cc); if (q[0] > mr) mr = q[0]; }
      else if (d[0] > mr) mr = d[0];
    }
  printf ("GPU gdd LU solve vs known x (dim=%ld): max relative error = %10.3e -> %s\n",
          dim, mr, mr < LU_THRESH ? "PASS (full double-double precision)" : "FAIL");
  return mr < LU_THRESH ? 0 : 1;
}
#endif
