/********************************************************************************/
/* test_gdmatmul.cu: verify & benchmark native double GPU linear (gdlinear)      */
/*   GPU gd matmul / matvec  vs  CPU double (mul_dmatrix / mul_dmatrix_dvec)      */
/********************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

#include "dlinear.h"
#include "gdlinear.h"

static double wtime(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6;
}

static double max_relerr_mat(DMatrix ref, GDMatrix gpu_dev)
{
	long int i, j, n = ref->row_dim, m = ref->col_dim;
	GDMatrix h = init_gdmatrix(n, m);
	double maxe = 0.0;
	cudaMemcpy((void *)h->element, (void *)gpu_dev->element, (size_t)(sizeof(double) * n * m), cudaMemcpyDeviceToHost);
	for(i = 0; i < n; i++)
		for(j = 0; j < m; j++)
		{
			double a = get_dmatrix_ij(ref, i, j);
			double b = h->element[i * m + j];
			double e = fabs(a - b);
			if(a != 0.0) e /= fabs(a);
			if(e > maxe) maxe = e;
		}
	free_gdmatrix(h);
	return maxe;
}

static double max_relerr_vec(DVector ref, GDVector gpu_dev)
{
	long int i, n = ref->dim;
	GDVector h = init_gdvector(n);
	double maxe = 0.0;
	cudaMemcpy((void *)h->element, (void *)gpu_dev->element, (size_t)(sizeof(double) * n), cudaMemcpyDeviceToHost);
	for(i = 0; i < n; i++)
	{
		double a = ref->element[i];
		double b = h->element[i];
		double e = fabs(a - b);
		if(a != 0.0) e /= fabs(a);
		if(e > maxe) maxe = e;
	}
	free_gdvector(h);
	return maxe;
}

int main(int argc, char *argv[])
{
	long int dim = 256, i, j;
	int nbg = 32, ntb = 64;
	if(argc >= 2) dim = atol(argv[1]);

	cudaSetDevice(0);

	// CPU operands
	DMatrix A = init_dmatrix(dim, dim);
	DMatrix B = init_dmatrix(dim, dim);
	DMatrix C = init_dmatrix(dim, dim);
	DVector x = init_dvector(dim);
	DVector y = init_dvector(dim);

	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			set_dmatrix_ij(A, i, j, sqrt((double)(i + j + 1)));
			set_dmatrix_ij(B, i, j, 1.0 / (double)(i + j + 1)); // Hilbert-like
		}
		set_dvector_i(x, i, sqrt((double)(i + 1)));
	}

	// GPU operands
	GDMatrix gA = init_gdmatrix_dev(dim, dim);
	GDMatrix gB = init_gdmatrix_dev(dim, dim);
	GDMatrix gC = init_gdmatrix_dev(dim, dim);
	GDVector gx = init_gdvector_dev(dim);
	GDVector gy = init_gdvector_dev(dim);
	subst_gdmatrix_dev_dmat(gA, A);
	subst_gdmatrix_dev_dmat(gB, B);
	subst_gdvector_dev_dvec(gx, x);

	printf("===== native double GPU linear (gdlinear), dim = %ld =====\n", dim);

	// ---- matmul correctness ----
	mul_dmatrix(C, A, B);
	mul_gdmatrix_dev(gC, gA, gB, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matmul ] max rel err (GPU vs CPU): %.3e\n", max_relerr_mat(C, gC));

	// ---- matvec correctness ----
	mul_dmatrix_dvec(y, A, x);
	mul_gdmatrix_gdvec(gy, gA, gx, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matvec ] max rel err (GPU vs CPU): %.3e\n", max_relerr_vec(y, gy));

	// ---- matmul^T*vec correctness ----
	mul_dmatrixt_dvec(y, A, x);
	mul_gdmatrixt_gdvec(gy, gA, gx, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matvecT] max rel err (GPU vs CPU): %.3e\n", max_relerr_vec(y, gy));

	// ---- performance: matmul ----
	double t0, t1, gpu_s, cpu_s;
	double flop = 2.0 * (double)dim * (double)dim * (double)dim;

	t0 = wtime();
	mul_gdmatrix_dev(gC, gA, gB, nbg, ntb);
	cudaDeviceSynchronize();
	t1 = wtime(); gpu_s = t1 - t0;

	t0 = wtime();
	mul_dmatrix(C, A, B);
	t1 = wtime(); cpu_s = t1 - t0;

	printf("[perf   ] matmul GPU: %.4e s (%.2f GFLOPS) | CPU: %.4e s (%.2f GFLOPS) | speedup x%.1f\n",
		gpu_s, flop / gpu_s * 1.0e-9, cpu_s, flop / cpu_s * 1.0e-9, cpu_s / gpu_s);

	free_dmatrix(A); free_dmatrix(B); free_dmatrix(C);
	free_dvector(x); free_dvector(y);
	free_gdmatrix_dev(gA); free_gdmatrix_dev(gB); free_gdmatrix_dev(gC);
	free_gdvector_dev(gx); free_gdvector_dev(gy);
	cudaDeviceReset();
	return 0;
}
