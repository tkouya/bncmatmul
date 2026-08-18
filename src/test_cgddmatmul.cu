/********************************************************************************/
/* test_cgddmatmul.cu: verify & benchmark complex-dd GPU linear (cgddlinear)      */
/*   GPU cgdd matmul/matvec  vs  CPU complex-dd (mul_cddmatrix_4m / _cddvec_4m)     */
/********************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

#include "cgddlinear.h"   // pulls gddlinear.h (gdd_real, dd_real) + cddlinear.h

extern "C" void mul_cddmatrix_4m(CDDMatrix, CDDMatrix, CDDMatrix);
extern "C" void mul_cddmatrix_cddvec_4m(CDDVector, CDDMatrix, CDDVector);
extern "C" void mul_cddmatrixt_cddvec(CDDVector, CDDMatrix, CDDVector);

static double wtime(void) { struct timeval tv; gettimeofday(&tv, NULL); return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6; }

#define CGSCAL double   /* element scalar type (float for single-based families) */
static cddfloat mkcdd(double re, double im) { cddfloat z; for(int l = 0; l < CGDD_SIZE; l++) { z.val_re[l] = 0.0; z.val_im[l] = 0.0; } z.val_re[0] = re; z.val_im[0] = im; return z; }

// (cpu - gpu)/cpu measured in qd_real (>= any d/s family precision); returned as double
static double cdd_relerr(cddfloat cpu, gdd_real gre, gdd_real gim)
{
	double cr[4] = {0,0,0,0}, ci[4] = {0,0,0,0}, gr[4] = {0,0,0,0}, gi[4] = {0,0,0,0};
	CGSCAL *pgre = (CGSCAL *)&gre, *pgim = (CGSCAL *)&gim;
	for(int l = 0; l < CGDD_SIZE; l++) { cr[l] = cpu.val_re[l]; ci[l] = cpu.val_im[l]; gr[l] = pgre[l]; gi[l] = pgim[l]; }
	qd_real cre(cr[0], cr[1], cr[2], cr[3]), cim(ci[0], ci[1], ci[2], ci[3]);
	qd_real ggre(gr[0], gr[1], gr[2], gr[3]), ggim(gi[0], gi[1], gi[2], gi[3]);
	qd_real dre = cre - ggre, dim = cim - ggim;
	qd_real num = sqrt(dre * dre + dim * dim), den = sqrt(cre * cre + cim * cim);
	if(den == qd_real(0.0)) return to_double(num);
	return to_double(num / den);
}

static double max_relerr_mat(CDDMatrix ref, CGDDMatrix gpu)
{
	long int i, j, n = ref->re->row_dim, m = ref->re->col_dim;
	CGDDMatrix h = init_cgddmatrix(n, m);
	double maxe = 0.0;
	cudaMemcpy((void *)h->re, (void *)gpu->re, (size_t)(sizeof(gdd_real) * n * m), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)gpu->im, (size_t)(sizeof(gdd_real) * n * m), cudaMemcpyDeviceToHost);
	for(i = 0; i < n; i++)
		for(j = 0; j < m; j++)
		{
			double e = cdd_relerr(get_cddmatrix_ij_cddfloat(ref, i, j), h->re[i * m + j], h->im[i * m + j]);
			if(e > maxe) maxe = e;
		}
	free_cgddmatrix(h);
	return maxe;
}
static double max_relerr_vec(CDDVector ref, CGDDVector gpu)
{
	long int i, n = ref->re->dim;
	CGDDVector h = init_cgddvector(n);
	double maxe = 0.0;
	cudaMemcpy((void *)h->re, (void *)gpu->re, (size_t)(sizeof(gdd_real) * n), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)gpu->im, (size_t)(sizeof(gdd_real) * n), cudaMemcpyDeviceToHost);
	for(i = 0; i < n; i++)
	{
		double e = cdd_relerr(get_cddvector_i_cddfloat(ref, i), h->re[i], h->im[i]);
		if(e > maxe) maxe = e;
	}
	free_cgddvector(h);
	return maxe;
}

int main(int argc, char *argv[])
{
	long int dim = 192, i, j;
	int nbg = 32, ntb = 64;
	if(argc >= 2) dim = atol(argv[1]);
	fpu_fix_start(NULL);
	cudaSetDevice(0);

	CDDMatrix A = init_cddmatrix(dim, dim), B = init_cddmatrix(dim, dim), C = init_cddmatrix(dim, dim);
	CDDVector x = init_cddvector(dim), y = init_cddvector(dim);
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			cddfloat va = mkcdd(sqrt((double)(i + j + 1)), 1.0 / (double)(i + j + 1));
			cddfloat vb = mkcdd(1.0 / (double)(i + j + 2), sqrt((double)(i + 2)));
			set_cddmatrix_ij(A, i, j, &va);
			set_cddmatrix_ij(B, i, j, &vb);
		}
		cddfloat vx = mkcdd(sqrt((double)(i + 1)), (double)i);
		set_cddvector_i(x, i, &vx);
	}

	CGDDMatrix gA = init_cgddmatrix_dev(dim, dim), gB = init_cgddmatrix_dev(dim, dim), gC = init_cgddmatrix_dev(dim, dim);
	CGDDVector gx = init_cgddvector_dev(dim), gy = init_cgddvector_dev(dim);
	subst_cgddmatrix_dev_cddmat(gA, A);
	subst_cgddmatrix_dev_cddmat(gB, B);
	subst_cgddvector_dev_cddvec(gx, x);

	printf("===== complex-dd GPU linear (cgddlinear), dim = %ld =====\n", dim);

	mul_cddmatrix_4m(C, A, B);
	mul_cgddmatrix_dev(gC, gA, gB, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matmul ] max rel err (GPU vs CPU, dd): %.3e\n", max_relerr_mat(C, gC));

	mul_cddmatrix_cddvec_4m(y, A, x);
	mul_cgddmatrix_cgddvec(gy, gA, gx, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matvec ] max rel err (GPU vs CPU, dd): %.3e\n", max_relerr_vec(y, gy));

	mul_cddmatrixt_cddvec(y, A, x);
	mul_cgddmatrixt_cgddvec(gy, gA, gx, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matvecT] max rel err (GPU vs CPU, dd): %.3e\n", max_relerr_vec(y, gy));

	double t0, t1, gpu_s, cpu_s, flop = 8.0 * (double)dim * (double)dim * (double)dim;
	t0 = wtime(); mul_cgddmatrix_dev(gC, gA, gB, nbg, ntb); cudaDeviceSynchronize(); t1 = wtime(); gpu_s = t1 - t0;
	t0 = wtime(); mul_cddmatrix_4m(C, A, B); t1 = wtime(); cpu_s = t1 - t0;
	printf("[perf   ] matmul GPU: %.4e s | CPU: %.4e s | speedup x%.1f\n", gpu_s, cpu_s, cpu_s / gpu_s);

	cudaDeviceReset();
	return 0;
}
