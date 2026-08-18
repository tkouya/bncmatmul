/********************************************************************************/
/* test_cgqsmatmul.cu: verify & benchmark complex-qs GPU linear (cgqslinear)      */
/*   GPU cgqs matmul/matvec  vs  CPU complex-qs (mul_cqsmatrix_4m / _cqsvec_4m)     */
/********************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

#include "qd/qd_real.h"
#include "cgqslinear.h"   // pulls gddlinear.h (gqs_real, dd_real) + cqslinear.h

extern "C" void mul_cqsmatrix_4m(CQSMatrix, CQSMatrix, CQSMatrix);
extern "C" void mul_cqsmatrix_cqsvec_4m(CQSVector, CQSMatrix, CQSVector);
extern "C" void mul_cqsmatrixt_cqsvec(CQSVector, CQSMatrix, CQSVector);

static double wtime(void) { struct timeval tv; gettimeofday(&tv, NULL); return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6; }

#define CGSCAL float   /* element scalar type (float for single-based families) */
static cqsfloat mkcqs(double re, double im) { cqsfloat z; for(int l = 0; l < CGQS_SIZE; l++) { z.val_re[l] = 0.0; z.val_im[l] = 0.0; } z.val_re[0] = re; z.val_im[0] = im; return z; }

// (cpu - gpu)/cpu measured in qd_real (>= any d/s family precision); returned as double
static double cqs_relerr(cqsfloat cpu, gqs_real gre, gqs_real gim)
{
	double cr[4] = {0,0,0,0}, ci[4] = {0,0,0,0}, gr[4] = {0,0,0,0}, gi[4] = {0,0,0,0};
	CGSCAL *pgre = (CGSCAL *)&gre, *pgim = (CGSCAL *)&gim;
	for(int l = 0; l < CGQS_SIZE; l++) { cr[l] = cpu.val_re[l]; ci[l] = cpu.val_im[l]; gr[l] = pgre[l]; gi[l] = pgim[l]; }
	qd_real cre(cr[0], cr[1], cr[2], cr[3]), cim(ci[0], ci[1], ci[2], ci[3]);
	qd_real ggre(gr[0], gr[1], gr[2], gr[3]), ggim(gi[0], gi[1], gi[2], gi[3]);
	qd_real dre = cre - ggre, dim = cim - ggim;
	qd_real num = sqrt(dre * dre + dim * dim), den = sqrt(cre * cre + cim * cim);
	if(den == qd_real(0.0)) return to_double(num);
	return to_double(num / den);
}

static double max_relerr_mat(CQSMatrix ref, CGQSMatrix gpu)
{
	long int i, j, n = ref->re->row_dim, m = ref->re->col_dim;
	CGQSMatrix h = init_cgqsmatrix(n, m);
	double maxe = 0.0;
	cudaMemcpy((void *)h->re, (void *)gpu->re, (size_t)(sizeof(gqs_real) * n * m), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)gpu->im, (size_t)(sizeof(gqs_real) * n * m), cudaMemcpyDeviceToHost);
	for(i = 0; i < n; i++)
		for(j = 0; j < m; j++)
		{
			double e = cqs_relerr(get_cqsmatrix_ij_cqsfloat(ref, i, j), h->re[i * m + j], h->im[i * m + j]);
			if(e > maxe) maxe = e;
		}
	free_cgqsmatrix(h);
	return maxe;
}
static double max_relerr_vec(CQSVector ref, CGQSVector gpu)
{
	long int i, n = ref->re->dim;
	CGQSVector h = init_cgqsvector(n);
	double maxe = 0.0;
	cudaMemcpy((void *)h->re, (void *)gpu->re, (size_t)(sizeof(gqs_real) * n), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)gpu->im, (size_t)(sizeof(gqs_real) * n), cudaMemcpyDeviceToHost);
	for(i = 0; i < n; i++)
	{
		double e = cqs_relerr(get_cqsvector_i_cqsfloat(ref, i), h->re[i], h->im[i]);
		if(e > maxe) maxe = e;
	}
	free_cgqsvector(h);
	return maxe;
}

int main(int argc, char *argv[])
{
	long int dim = 192, i, j;
	int nbg = 32, ntb = 64;
	if(argc >= 2) dim = atol(argv[1]);
	fpu_fix_start(NULL);
	cudaSetDevice(0);

	CQSMatrix A = init_cqsmatrix(dim, dim), B = init_cqsmatrix(dim, dim), C = init_cqsmatrix(dim, dim);
	CQSVector x = init_cqsvector(dim), y = init_cqsvector(dim);
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			cqsfloat va = mkcqs(sqrt((double)(i + j + 1)), 1.0 / (double)(i + j + 1));
			cqsfloat vb = mkcqs(1.0 / (double)(i + j + 2), sqrt((double)(i + 2)));
			set_cqsmatrix_ij(A, i, j, &va);
			set_cqsmatrix_ij(B, i, j, &vb);
		}
		cqsfloat vx = mkcqs(sqrt((double)(i + 1)), (double)i);
		set_cqsvector_i(x, i, &vx);
	}

	CGQSMatrix gA = init_cgqsmatrix_dev(dim, dim), gB = init_cgqsmatrix_dev(dim, dim), gC = init_cgqsmatrix_dev(dim, dim);
	CGQSVector gx = init_cgqsvector_dev(dim), gy = init_cgqsvector_dev(dim);
	subst_cgqsmatrix_dev_cqsmat(gA, A);
	subst_cgqsmatrix_dev_cqsmat(gB, B);
	subst_cgqsvector_dev_cqsvec(gx, x);

	printf("===== complex-qs GPU linear (cgqslinear), dim = %ld =====\n", dim);

	mul_cqsmatrix_4m(C, A, B);
	mul_cgqsmatrix_dev(gC, gA, gB, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matmul ] max rel err (GPU vs CPU, dd): %.3e\n", max_relerr_mat(C, gC));

	mul_cqsmatrix_cqsvec_4m(y, A, x);
	mul_cgqsmatrix_cgqsvec(gy, gA, gx, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matvec ] max rel err (GPU vs CPU, dd): %.3e\n", max_relerr_vec(y, gy));

	mul_cqsmatrixt_cqsvec(y, A, x);
	mul_cgqsmatrixt_cgqsvec(gy, gA, gx, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matvecT] max rel err (GPU vs CPU, dd): %.3e\n", max_relerr_vec(y, gy));

	double t0, t1, gpu_s, cpu_s, flop = 8.0 * (double)dim * (double)dim * (double)dim;
	t0 = wtime(); mul_cgqsmatrix_dev(gC, gA, gB, nbg, ntb); cudaDeviceSynchronize(); t1 = wtime(); gpu_s = t1 - t0;
	t0 = wtime(); mul_cqsmatrix_4m(C, A, B); t1 = wtime(); cpu_s = t1 - t0;
	printf("[perf   ] matmul GPU: %.4e s | CPU: %.4e s | speedup x%.1f\n", gpu_s, cpu_s, cpu_s / gpu_s);

	cudaDeviceReset();
	return 0;
}
