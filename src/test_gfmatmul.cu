/********************************************************************************/
/* test_gfmatmul.cu: verify & benchmark native float GPU linear (gflinear)      */
/*   GPU gd matmul / matvec  vs  CPU float (mul_fmatrix / mul_fmatrix_fvec)      */
/********************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

#include "flinear.h"
#include "gflinear.h"

// flinear.h declares mul_fmatrix_fvec/_t, but the library exports the _dvec names.
extern "C" void mul_fmatrix_dvec(FVector v, FMatrix a, FVector vb);
extern "C" void mul_fmatrixt_dvec(FVector v, FMatrix a, FVector vb);

static double wtime(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}

static float max_relerr_mat(FMatrix ref, GFMatrix gpu_dev)
{
	long int i, j, n = ref->row_dim, m = ref->col_dim;
	GFMatrix h = init_gfmatrix(n, m);
	float maxe = 0.0;
	cudaMemcpy((void *)h->element, (void *)gpu_dev->element, (size_t)(sizeof(float) * n * m), cudaMemcpyDeviceToHost);
	for(i = 0; i < n; i++)
		for(j = 0; j < m; j++)
		{
			float a = get_fmatrix_ij(ref, i, j);
			float b = h->element[i * m + j];
			float e = fabs(a - b);
			if(a != 0.0) e /= fabs(a);
			if(e > maxe) maxe = e;
		}
	free_gfmatrix(h);
	return maxe;
}

static float max_relerr_vec(FVector ref, GFVector gpu_dev)
{
	long int i, n = ref->dim;
	GFVector h = init_gfvector(n);
	float maxe = 0.0;
	cudaMemcpy((void *)h->element, (void *)gpu_dev->element, (size_t)(sizeof(float) * n), cudaMemcpyDeviceToHost);
	for(i = 0; i < n; i++)
	{
		float a = ref->element[i];
		float b = h->element[i];
		float e = fabs(a - b);
		if(a != 0.0) e /= fabs(a);
		if(e > maxe) maxe = e;
	}
	free_gfvector(h);
	return maxe;
}

int main(int argc, char *argv[])
{
	long int dim = 256, i, j;
	int nbg = 32, ntb = 64;
	if(argc >= 2) dim = atol(argv[1]);

	cudaSetDevice(0);

	// CPU operands
	FMatrix A = init_fmatrix(dim, dim);
	FMatrix B = init_fmatrix(dim, dim);
	FMatrix C = init_fmatrix(dim, dim);
	FVector x = init_fvector(dim);
	FVector y = init_fvector(dim);

	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			set_fmatrix_ij(A, i, j, sqrt((float)(i + j + 1)));
			set_fmatrix_ij(B, i, j, 1.0 / (float)(i + j + 1)); // Hilbert-like
		}
		set_fvector_i(x, i, sqrt((float)(i + 1)));
	}

	// GPU operands
	GFMatrix gA = init_gfmatrix_dev(dim, dim);
	GFMatrix gB = init_gfmatrix_dev(dim, dim);
	GFMatrix gC = init_gfmatrix_dev(dim, dim);
	GFVector gx = init_gfvector_dev(dim);
	GFVector gy = init_gfvector_dev(dim);
	subst_gfmatrix_dev_fmat(gA, A);
	subst_gfmatrix_dev_fmat(gB, B);
	subst_gfvector_dev_fvec(gx, x);

	printf("===== native float GPU linear (gflinear), dim = %ld =====\n", dim);

	// ---- matmul correctness ----
	mul_fmatrix(C, A, B);
	mul_gfmatrix_dev(gC, gA, gB, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matmul ] max rel err (GPU vs CPU): %.3e\n", max_relerr_mat(C, gC));

	// ---- matvec correctness ----
	mul_fmatrix_dvec(y, A, x);
	mul_gfmatrix_gfvec(gy, gA, gx, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matvec ] max rel err (GPU vs CPU): %.3e\n", max_relerr_vec(y, gy));

	// ---- matmul^T*vec correctness ----
	mul_fmatrixt_dvec(y, A, x);
	mul_gfmatrixt_gfvec(gy, gA, gx, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matvecT] max rel err (GPU vs CPU): %.3e\n", max_relerr_vec(y, gy));

	// ---- performance: matmul ----
	double t0, t1, gpu_s, cpu_s;
	double flop = 2.0 * (float)dim * (float)dim * (float)dim;

	t0 = wtime();
	mul_gfmatrix_dev(gC, gA, gB, nbg, ntb);
	cudaDeviceSynchronize();
	t1 = wtime(); gpu_s = t1 - t0;

	t0 = wtime();
	mul_fmatrix(C, A, B);
	t1 = wtime(); cpu_s = t1 - t0;

	printf("[perf   ] matmul GPU: %.4e s (%.2f GFLOPS) | CPU: %.4e s (%.2f GFLOPS) | speedup x%.1f\n",
		gpu_s, flop / gpu_s * 1.0e-9, cpu_s, flop / cpu_s * 1.0e-9, cpu_s / gpu_s);

	free_fmatrix(A); free_fmatrix(B); free_fmatrix(C);
	free_fvector(x); free_fvector(y);
	free_gfmatrix_dev(gA); free_gfmatrix_dev(gB); free_gfmatrix_dev(gC);
	free_gfvector_dev(gx); free_gfvector_dev(gy);
	cudaDeviceReset();
	return 0;
}
