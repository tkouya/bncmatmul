/* gdlu.cu -- GPU native double LU decomposition (partial pivoting) + triangular
 * solve for BNCmatmul.  Right-looking: per pivot column k a 1-thread pivot
 * search, a parallel row swap, and a parallel trailing rank-1 update; then
 * forward/back substitution.  Mirrors the CPU LU solve.  Operates on
 * GDMatrix/GDVector device objects.  (add -DGDLU_TEST for the self-test main)
 */
#include <cstdio>
#include <cmath>
#ifndef LU_THRESH
#define LU_THRESH 1e-8
#endif
#include "gdlinear.h"

/* one-thread partial-pivot search over rows i = k..n-1 of column k */
__global__ static void gdlu_find_pivot(const double *A, int n, int k, int *imax)
{
	if(blockIdx.x * blockDim.x + threadIdx.x != 0) return;
	int p = k;
	double best = fabs(A[(size_t)k * n + k]);
	for(int i = k + 1; i < n; ++i)
	{
		double v = fabs(A[(size_t)i * n + k]);
		if(v > best) { best = v; p = i; }
	}
	*imax = p;
}

/* swap rows k and m of A (parallel over columns) and of b (thread 0) */
__global__ static void gdlu_swap_row(double *A, double *b, int n, int k, int m)
{
	int stride = gridDim.x * blockDim.x;
	for(int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += stride)
	{ double t = A[(size_t)k * n + j]; A[(size_t)k * n + j] = A[(size_t)m * n + j]; A[(size_t)m * n + j] = t; }
	if(blockIdx.x * blockDim.x + threadIdx.x == 0)
	{ double t = b[k]; b[k] = b[m]; b[m] = t; }
}

/* trailing update: for i>k, multiplier m=A[i][k]/A[k][k]; A[i][j]-=m*A[k][j] (j>k) */
__global__ static void gdlu_eliminate(double *A, int n, int k)
{
	int stride = gridDim.x * blockDim.x;
	for(int i = k + 1 + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
	{
		double mik = A[(size_t)i * n + k] / A[(size_t)k * n + k];
		A[(size_t)i * n + k] = mik;
		for(int j = k + 1; j < n; ++j)
			A[(size_t)i * n + j] = A[(size_t)i * n + j] - mik * A[(size_t)k * n + j];
	}
}

/* forward (unit lower L) then back (upper U) substitution: solve in x (1 thread) */
__global__ static void gdlu_fbsub(const double *A, const double *b, double *x, int n)
{
	if(blockIdx.x * blockDim.x + threadIdx.x != 0) return;
	for(int i = 0; i < n; ++i)
	{
		double s = b[i];
		for(int j = 0; j < i; ++j) s = s - A[(size_t)i * n + j] * x[j];
		x[i] = s;                       /* unit diagonal of L */
	}
	for(int i = n - 1; i >= 0; --i)
	{
		double s = x[i];
		for(int j = i + 1; j < n; ++j) s = s - A[(size_t)i * n + j] * x[j];
		x[i] = s / A[(size_t)i * n + i];
	}
}

/* Host driver: A_dev is overwritten with its LU factors; solves A x = b_dev into
 * x_dev (all device GD objects, n x n).  ch[] (host, length n) receives the pivot
 * row chosen at each step.  Returns 0 on success. */
extern "C" int
gd_lu_solve_dev(GDMatrix A_dev, GDVector b_dev, GDVector x_dev, long int *ch, int blocks, int threads)
{
	int n = (int)A_dev->col_dim;
	if(blocks <= 0) blocks = 64;
	if(threads <= 0) threads = 128;
	int *d_imax, h_imax;
	cudaMalloc(&d_imax, sizeof(int));
	double *A = A_dev->element, *b = b_dev->element, *x = x_dev->element;

	for(int k = 0; k < n; ++k)
	{
		gdlu_find_pivot<<<1, 1>>>(A, n, k, d_imax);
		cudaMemcpy(&h_imax, d_imax, sizeof(int), cudaMemcpyDeviceToHost);
		if(ch) ch[k] = h_imax;
		if(h_imax != k) gdlu_swap_row<<<blocks, threads>>>(A, b, n, k, h_imax);
		gdlu_eliminate<<<blocks, threads>>>(A, n, k);
	}
	gdlu_fbsub<<<1, 1>>>(A, b, x, n);
	cudaError_t err = cudaDeviceSynchronize();
	cudaFree(d_imax);
	return err == cudaSuccess ? 0 : (int)err;
}

#ifdef GDLU_TEST
#include "dlinear.h"
int main(void)
{
	long int i, dim = 100;

	DMatrix a = init_dmatrix(dim, dim);
	DVector xtrue = init_dvector(dim), b = init_dvector(dim);

	frank_dmatrix(a, dim);                       /* Frank matrix: needs partial pivoting */
	for(i = 0; i < dim; i++) set_dvector_i(xtrue, i, (double)i);
	mul_dmatrix_dvec(b, a, xtrue);               /* b = A * xtrue (CPU) */

	GDMatrix ga = init_gdmatrix_dev(dim, dim);
	GDVector gb = init_gdvector_dev(dim), gx = init_gdvector_dev(dim);
	subst_gdmatrix_dev_dmat(ga, a);
	subst_gdvector_dev_dvec(gb, b);

	if(gd_lu_solve_dev(ga, gb, gx, NULL, 64, 128)) { fprintf(stderr, "GPU LU failed\n"); return 1; }

	DVector xg = init_dvector(dim);
	subst_dvector_gdvec_dev(xg, gx);             /* device -> host */

	double mr = 0.0;
	for(i = 0; i < dim; i++)
	{
		double g = get_dvector_i(xg, i), c = get_dvector_i(xtrue, i);
		double d = fabs(g - c);
		if(c != 0.0) d /= fabs(c);
		if(d > mr) mr = d;
	}
	printf("GPU gd LU solve vs known x (dim=%ld): max relative error = %10.3e -> %s\n",
		dim, mr, mr < LU_THRESH ? "PASS (native double precision)" : "FAIL");
	return mr < LU_THRESH ? 0 : 1;
}
#endif
