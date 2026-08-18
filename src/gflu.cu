/* gflu.cu -- GPU native float LU decomposition (partial pivoting) + triangular
 * solve for BNCmatmul.  Right-looking: per pivot column k a 1-thread pivot
 * search, a parallel row swap, and a parallel trailing rank-1 update; then
 * forward/back substitution.  Mirrors the CPU LU solve.  Operates on
 * GFMatrix/GFVector device objects.  (add -DGFLU_TEST for the self-test main)
 */
#include <cstdio>
#include <cmath>
#ifndef LU_THRESH
#define LU_THRESH 1e-3
#endif
#include "gflinear.h"

/* one-thread partial-pivot search over rows i = k..n-1 of column k */
__global__ static void gflu_find_pivot(const float *A, int n, int k, int *imax)
{
	if(blockIdx.x * blockDim.x + threadIdx.x != 0) return;
	int p = k;
	float best = fabs(A[(size_t)k * n + k]);
	for(int i = k + 1; i < n; ++i)
	{
		float v = fabs(A[(size_t)i * n + k]);
		if(v > best) { best = v; p = i; }
	}
	*imax = p;
}

/* swap rows k and m of A (parallel over columns) and of b (thread 0) */
__global__ static void gflu_swap_row(float *A, float *b, int n, int k, int m)
{
	int stride = gridDim.x * blockDim.x;
	for(int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += stride)
	{ float t = A[(size_t)k * n + j]; A[(size_t)k * n + j] = A[(size_t)m * n + j]; A[(size_t)m * n + j] = t; }
	if(blockIdx.x * blockDim.x + threadIdx.x == 0)
	{ float t = b[k]; b[k] = b[m]; b[m] = t; }
}

/* trailing update: for i>k, multiplier m=A[i][k]/A[k][k]; A[i][j]-=m*A[k][j] (j>k) */
__global__ static void gflu_eliminate(float *A, int n, int k)
{
	int stride = gridDim.x * blockDim.x;
	for(int i = k + 1 + blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
	{
		float mik = A[(size_t)i * n + k] / A[(size_t)k * n + k];
		A[(size_t)i * n + k] = mik;
		for(int j = k + 1; j < n; ++j)
			A[(size_t)i * n + j] = A[(size_t)i * n + j] - mik * A[(size_t)k * n + j];
	}
}

/* forward (unit lower L) then back (upper U) substitution: solve in x (1 thread) */
__global__ static void gflu_fbsub(const float *A, const float *b, float *x, int n)
{
	if(blockIdx.x * blockDim.x + threadIdx.x != 0) return;
	for(int i = 0; i < n; ++i)
	{
		float s = b[i];
		for(int j = 0; j < i; ++j) s = s - A[(size_t)i * n + j] * x[j];
		x[i] = s;                       /* unit diagonal of L */
	}
	for(int i = n - 1; i >= 0; --i)
	{
		float s = x[i];
		for(int j = i + 1; j < n; ++j) s = s - A[(size_t)i * n + j] * x[j];
		x[i] = s / A[(size_t)i * n + i];
	}
}

/* Host driver: A_dev is overwritten with its LU factors; solves A x = b_dev into
 * x_dev (all device GD objects, n x n).  ch[] (host, length n) receives the pivot
 * row chosen at each step.  Returns 0 on success. */
extern "C" int
gf_lu_solve_dev(GFMatrix A_dev, GFVector b_dev, GFVector x_dev, long int *ch, int blocks, int threads)
{
	int n = (int)A_dev->col_dim;
	if(blocks <= 0) blocks = 64;
	if(threads <= 0) threads = 128;
	int *d_imax, h_imax;
	cudaMalloc(&d_imax, sizeof(int));
	float *A = A_dev->element, *b = b_dev->element, *x = x_dev->element;

	for(int k = 0; k < n; ++k)
	{
		gflu_find_pivot<<<1, 1>>>(A, n, k, d_imax);
		cudaMemcpy(&h_imax, d_imax, sizeof(int), cudaMemcpyDeviceToHost);
		if(ch) ch[k] = h_imax;
		if(h_imax != k) gflu_swap_row<<<blocks, threads>>>(A, b, n, k, h_imax);
		gflu_eliminate<<<blocks, threads>>>(A, n, k);
	}
	gflu_fbsub<<<1, 1>>>(A, b, x, n);
	cudaError_t err = cudaDeviceSynchronize();
	cudaFree(d_imax);
	return err == cudaSuccess ? 0 : (int)err;
}

#ifdef GFLU_TEST
#include "flinear.h"
extern "C" void mul_fmatrix_dvec(FVector v, FMatrix a, FVector vb);
int main(void)
{
	long int i, j, dim = 100;

	FMatrix a = init_fmatrix(dim, dim);
	FVector xtrue = init_fvector(dim), b = init_fvector(dim);

	/* diagonally dominant A (Frank matrix is too ill-conditioned for single) */
	for(i = 0; i < dim; i++)
		for(j = 0; j < dim; j++)
			set_fmatrix_ij(a, i, j, (i == j) ? (float)(2 * dim) : 1.0f / (float)(i + j + 2));
	for(i = 0; i < dim; i++) set_fvector_i(xtrue, i, (float)i);
	mul_fmatrix_dvec(b, a, xtrue);               /* b = A * xtrue (CPU) */

	GFMatrix ga = init_gfmatrix_dev(dim, dim);
	GFVector gb = init_gfvector_dev(dim), gx = init_gfvector_dev(dim);
	subst_gfmatrix_dev_fmat(ga, a);
	subst_gfvector_dev_fvec(gb, b);

	if(gf_lu_solve_dev(ga, gb, gx, NULL, 64, 128)) { fprintf(stderr, "GPU LU failed\n"); return 1; }

	FVector xg = init_fvector(dim);
	subst_fvector_gfvec_dev(xg, gx);             /* device -> host */

	float mr = 0.0;
	for(i = 0; i < dim; i++)
	{
		float g = get_fvector_i(xg, i), c = get_fvector_i(xtrue, i);
		float d = fabs(g - c);
		if(c != 0.0) d /= fabs(c);
		if(d > mr) mr = d;
	}
	printf("GPU gf LU solve vs known x (dim=%ld): max relative error = %10.3e -> %s\n",
		dim, mr, mr < LU_THRESH ? "PASS (native float precision)" : "FAIL");
	return mr < LU_THRESH ? 0 : 1;
}
#endif
