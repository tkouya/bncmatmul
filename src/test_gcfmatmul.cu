/********************************************************************************/
/* test_gcfmatmul.cu: verify & benchmark native complex float GPU (gcflinear)    */
/*   GPU gcd matmul / matvec  vs  CPU complex float (mul_cdmatrix / _cdvec)       */
/********************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <complex.h>
#include <sys/time.h>

#include "cdlinear.h"
#include "gcflinear.h"

// GCC complex extension helpers (C99 complex.h macros unavailable in nvcc C++ mode)
static inline double _Complex _mkc(float re, float im) { double _Complex z; __real__ z = re; __imag__ z = im; return z; }
static inline float _cabs(double _Complex z) { float r = __real__ z, i = __imag__ z; return sqrt(r * r + i * i); }

static double wtime(void) { struct timeval tv; gettimeofday(&tv, NULL); return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6; }

static float max_relerr_cdmat(CDMatrix ref, GCFMatrix gpu)
{
	long int i, j, n = ref->row_dim, m = ref->col_dim;
	GCFMatrix h = init_gcfmatrix(n, m);
	float maxe = 0.0;
	cudaMemcpy((void *)h->re, (void *)gpu->re, (size_t)(sizeof(float) * n * m), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)gpu->im, (size_t)(sizeof(float) * n * m), cudaMemcpyDeviceToHost);
	for(i = 0; i < n; i++)
		for(j = 0; j < m; j++)
		{
			double _Complex a = get_cdmatrix_ij(ref, i, j);
			double _Complex b = _mkc(h->re[i * m + j], h->im[i * m + j]);
			float e = _cabs(a - b);
			if(_cabs(a) != 0.0) e /= _cabs(a);
			if(e > maxe) maxe = e;
		}
	free_gcfmatrix(h);
	return maxe;
}

static float max_relerr_cdvec(CDVector ref, GCFVector gpu)
{
	long int i, n = ref->dim;
	GCFVector h = init_gcfvector(n);
	float maxe = 0.0;
	cudaMemcpy((void *)h->re, (void *)gpu->re, (size_t)(sizeof(float) * n), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)gpu->im, (size_t)(sizeof(float) * n), cudaMemcpyDeviceToHost);
	for(i = 0; i < n; i++)
	{
		double _Complex a = get_cdvector_i(ref, i);
		double _Complex b = _mkc(h->re[i], h->im[i]);
		float e = _cabs(a - b);
		if(_cabs(a) != 0.0) e /= _cabs(a);
		if(e > maxe) maxe = e;
	}
	free_gcfvector(h);
	return maxe;
}

int main(int argc, char *argv[])
{
	long int dim = 256, i, j;
	int nbg = 32, ntb = 64;
	if(argc >= 2) dim = atol(argv[1]);
	cudaSetDevice(0);

	CDMatrix A = init_cdmatrix(dim, dim), B = init_cdmatrix(dim, dim), C = init_cdmatrix(dim, dim);
	CDVector x = init_cdvector(dim), y = init_cdvector(dim);

	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			double _Complex va = _mkc(sqrt((float)(i + j + 1)), 1.0 / (float)(i + j + 1));
			double _Complex vb = _mkc(1.0 / (float)(i + j + 2), sqrt((float)(i + 2)));
			set_cdmatrix_ij(A, i, j, va);
			set_cdmatrix_ij(B, i, j, vb);
		}
		set_cdvector_i(x, i, _mkc(sqrt((float)(i + 1)), (float)i));
	}

	GCFMatrix gA = init_gcfmatrix_dev(dim, dim), gB = init_gcfmatrix_dev(dim, dim), gC = init_gcfmatrix_dev(dim, dim);
	GCFVector gx = init_gcfvector_dev(dim), gy = init_gcfvector_dev(dim);
	subst_gcfmatrix_dev_cdmat(gA, A);
	subst_gcfmatrix_dev_cdmat(gB, B);
	subst_gcfvector_dev_cdvec(gx, x);

	printf("===== native complex float GPU linear (gcflinear), dim = %ld =====\n", dim);

	mul_cdmatrix(C, A, B);
	mul_gcfmatrix_dev(gC, gA, gB, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matmul ] max rel err (GPU vs CPU): %.3e\n", max_relerr_cdmat(C, gC));

	mul_cdmatrix_cdvec(y, A, x);
	mul_gcfmatrix_gcfvec(gy, gA, gx, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matvec ] max rel err (GPU vs CPU): %.3e\n", max_relerr_cdvec(y, gy));

	mul_cdmatrixt_cdvec(y, A, x);
	mul_gcfmatrixt_gcfvec(gy, gA, gx, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matvecT] max rel err (GPU vs CPU): %.3e\n", max_relerr_cdvec(y, gy));

	double t0, t1, gpu_s, cpu_s;
	double flop = 8.0 * (double)dim * (double)dim * (double)dim; // complex matmul ~ 4 mul + 4 add per term
	t0 = wtime(); mul_gcfmatrix_dev(gC, gA, gB, nbg, ntb); cudaDeviceSynchronize(); t1 = wtime(); gpu_s = t1 - t0;
	t0 = wtime(); mul_cdmatrix(C, A, B); t1 = wtime(); cpu_s = t1 - t0;
	printf("[perf   ] matmul GPU: %.4e s (%.2f GFLOPS) | CPU: %.4e s (%.2f GFLOPS) | speedup x%.1f\n",
		gpu_s, flop / gpu_s * 1.0e-9, cpu_s, flop / cpu_s * 1.0e-9, cpu_s / gpu_s);

	free_cdmatrix(A); free_cdmatrix(B); free_cdmatrix(C);
	free_cdvector(x); free_cdvector(y);
	free_gcfmatrix_dev(gA); free_gcfmatrix_dev(gB); free_gcfmatrix_dev(gC);
	free_gcfvector_dev(gx); free_gcfvector_dev(gy);
	cudaDeviceReset();
	return 0;
}
