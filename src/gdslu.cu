/* gdslu.cu -- GPU double-double LU decomposition (partial pivoting) + triangular
 * solve for BNCmatmul, using the gqd/gdtq device ds arithmetic.  Right-looking:
 * per pivot column k a 1-thread pivot search, a parallel row swap, and a parallel
 * trailing rank-1 update; then forward/back substitution.  Mirrors the CPU
 * DSLUdecompPM / SolveDSLSPM.  Operates on GDSMatrix/GDSVector device objects.
 *
 * Build (from repo root):
 *   nvcc -O3 -arch=sm_121 -std=c++17 -DCUDA_FMA -DUSE_DSLINEAR -DUSE_GMP -DUSE_MPFR \
 *        -Iinclude -I/usr/local/include/gdtq -I/usr/local/cuda/include \
 *        -diag-suppress 20011 -diag-suppress 177 -dc src/gdslu.cu -o gdslu.o
 *   nvcc -dlink gdslu.o cuda_obj/gdslinear.o -o dl.o
 *   nvcc ... gdslu.o cuda_obj/gdslinear.o dl.o -L. -lbncmatmul-0.24 <cuda+qd+mpfr libs>
 *   (ads -DGDSLU_TEST for the self-test main)
 */
#include <cstdio>
#include <cmath>
#ifndef LU_THRESH
#define LU_THRESH 1e-28
#endif
#include "gdslinear.h"

__device__ static inline gds_real gds_abs (gds_real a)
{ gds_real z = a - a; return (a.x < 0.0) ? (z - a) : a; }

/* one-thread partial-pivot search over rows i = k..n-1 of column k */
__global__ static void gdslu_find_pivot (const gds_real *A, int n, int k, int *imax)
{
  if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
  int p = k;
  gds_real best = gds_abs (A[(size_t) k * n + k]);
  for (int i = k + 1; i < n; ++i)
    {
      gds_real v = gds_abs (A[(size_t) i * n + k]);
      if (v > best) { best = v; p = i; }
    }
  *imax = p;
}

/* swap rows k and m of A (parallel over columns) and of b (thread 0) */
__global__ static void gdslu_swap_row (gds_real *A, gds_real *b, int n, int k, int m)
{
  int stride = gridDim.x * blockDim.x;
  for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += stride)
    { gds_real t = A[(size_t) k * n + j]; A[(size_t) k * n + j] = A[(size_t) m * n + j]; A[(size_t) m * n + j] = t; }
  if (blockIdx.x * blockDim.x + threadIdx.x == 0)
    { gds_real t = b[k]; b[k] = b[m]; b[m] = t; }
}

/* trailing update: for i>k, multiplier m=A[i][k]/A[k][k]; A[i][j]-=m*A[k][j] (j>k) */
__global__ static void gdslu_eliminate (gds_real *A, int n, int k)
{
  int stride = gridDim.x * blockDim.x;
  for (int i = k + 1 + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
      gds_real mik = A[(size_t) i * n + k] / A[(size_t) k * n + k];
      A[(size_t) i * n + k] = mik;
      for (int j = k + 1; j < n; ++j)
        A[(size_t) i * n + j] = A[(size_t) i * n + j] - mik * A[(size_t) k * n + j];
    }
}

/* forward (unit lower L) then back (upper U) substitution: solve in x (1 thread) */
__global__ static void gdslu_fbsub (const gds_real *A, const gds_real *b, gds_real *x, int n)
{
  if (blockIdx.x * blockDim.x + threadIdx.x != 0) return;
  for (int i = 0; i < n; ++i)
    {
      gds_real s = b[i];
      for (int j = 0; j < i; ++j) s = s - A[(size_t) i * n + j] * x[j];
      x[i] = s;                       /* unit diagonal of L */
    }
  for (int i = n - 1; i >= 0; --i)
    {
      gds_real s = x[i];
      for (int j = i + 1; j < n; ++j) s = s - A[(size_t) i * n + j] * x[j];
      x[i] = s / A[(size_t) i * n + i];
    }
}

/* Host driver: A_dev is overwritten with its LU factors; solves A x = b_dev into
 * x_dev (all device GDS objects, n x n).  ch[] (host, length n) receives the pivot
 * row chosen at each step.  Returns 0 on success. */
extern "C" int
gds_lu_solve_dev (GDSMatrix A_dev, GDSVector b_dev, GDSVector x_dev, long int *ch,
                  int blocks, int threads)
{
  int n = (int) A_dev->col_dim;
  if (blocks <= 0) blocks = 64;
  if (threads <= 0) threads = 128;
  int *d_imax, h_imax;
  cudaMalloc (&d_imax, sizeof (int));
  gds_real *A = A_dev->element, *b = b_dev->element, *x = x_dev->element;

  for (int k = 0; k < n; ++k)
    {
      gdslu_find_pivot<<<1, 1>>> (A, n, k, d_imax);
      cudaMemcpy (&h_imax, d_imax, sizeof (int), cudaMemcpyDeviceToHost);
      if (ch) ch[k] = h_imax;
      if (h_imax != k) gdslu_swap_row<<<blocks, threads>>> (A, b, n, k, h_imax);
      gdslu_eliminate<<<blocks, threads>>> (A, n, k);
    }
  gdslu_fbsub<<<1, 1>>> (A, b, x, n);
  cudaError_t err = cudaDeviceSynchronize ();
  cudaFree (d_imax);
  return err == cudaSuccess ? 0 : (int) err;
}

#ifdef GDSLU_TEST
int main (void)
{
  long int i, j, dim = 100;

  DSMatrix a = init_dsmatrix (dim, dim);
  DSVector xtrue = init_dsvector (dim), b = init_dsvector (dim), xg = init_dsvector (dim);
  float t[DSSIZE];

  /* diagonally dominant A, known solution xtrue=[0,1,...] */
  for (i = 0; i < dim; i++) {
    for (j = 0; j < dim; j++) {
      if (i == j) rds_set_ui (t, (unsigned long) (dim * dim));
      else { rds_set_ui (t, (unsigned long) ((i + j) % 5 + 1)); rds_sqrt (t, t); }
      set_dsmatrix_ij (a, i, j, t);
    }
    rds_set_ui (t, (unsigned long) i); set_dsvector_i (xtrue, i, t);
  }
  mul_dsmatrix_dsvec (b, a, xtrue);          /* b = A * xtrue (CPU) */

  GDSMatrix ga = init_gdsmatrix_dev (dim, dim);
  GDSVector gb = init_gdsvector_dev (dim), gx = init_gdsvector_dev (dim);
  subst_gdsmatrix_dev_dsmat (ga, a);
  subst_gdsvector_dev_dsvec (gb, b);

  if (gds_lu_solve_dev (ga, gb, gx, NULL, 64, 128)) { fprintf (stderr, "GPU LU failed\n"); return 1; }
  subst_dsvector_gdsvec_dev (xg, gx);        /* device -> host */

  double mr = 0.0; float d[DSSIZE], q[DSSIZE], cc[DSSIZE];
  for (i = 0; i < dim; i++)
    {
      float *g = get_dsvector_i (xg, i), *c = get_dsvector_i (xtrue, i);
      rds_sub (d, g, c); rds_abs (d, d); rds_abs (cc, c);
      if (cc[0] != 0.0) { rds_div (q, d, cc); if (q[0] > mr) mr = q[0]; }
      else if (d[0] > mr) mr = d[0];
    }
  printf ("GPU gds LU solve vs known x (dim=%ld): max relative error = %10.3e -> %s\n",
          dim, mr, mr < LU_THRESH ? "PASS (full double-double precision)" : "FAIL");
  return mr < LU_THRESH ? 0 : 1;
}
#endif
