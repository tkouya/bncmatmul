/********************************************************************************/
/* test_cgdsmatmul.cu: verify & benchmark complex-ds GPU linear (cgdslinear)      */
/*   GPU cgds matmul/matvec  vs  CPU complex-ds (mul_cdsmatrix_4m / _cdsvec_4m)     */
/********************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

#include "qd/qd_real.h"
#include "cgdslinear.h"   // pulls gddlinear.h (gds_real, dd_real) + cdslinear.h

extern "C" void mul_cdsmatrix_4m(CDSMatrix, CDSMatrix, CDSMatrix);
extern "C" void mul_cdsmatrix_cdsvec_4m(CDSVector, CDSMatrix, CDSVector);
extern "C" void mul_cdsmatrixt_cdsvec(CDSVector, CDSMatrix, CDSVector);

static double wtime(void) { struct timeval tv; gettimeofday(&tv, NULL); return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6; }

#define CGSCAL float   /* element scalar type (float for single-based families) */
static cdsfloat mkcds(double re, double im) { cdsfloat z; for(int l = 0; l < CGDS_SIZE; l++) { z.val_re[l] = 0.0; z.val_im[l] = 0.0; } z.val_re[0] = re; z.val_im[0] = im; return z; }

// (cpu - gpu)/cpu measured in qd_real (>= any d/s family precision); returned as double
static double cds_relerr(cdsfloat cpu, gds_real gre, gds_real gim)
{
	double cr[4] = {0,0,0,0}, ci[4] = {0,0,0,0}, gr[4] = {0,0,0,0}, gi[4] = {0,0,0,0};
	CGSCAL *pgre = (CGSCAL *)&gre, *pgim = (CGSCAL *)&gim;
	for(int l = 0; l < CGDS_SIZE; l++) { cr[l] = cpu.val_re[l]; ci[l] = cpu.val_im[l]; gr[l] = pgre[l]; gi[l] = pgim[l]; }
	qd_real cre(cr[0], cr[1], cr[2], cr[3]), cim(ci[0], ci[1], ci[2], ci[3]);
	qd_real ggre(gr[0], gr[1], gr[2], gr[3]), ggim(gi[0], gi[1], gi[2], gi[3]);
	qd_real dre = cre - ggre, dim = cim - ggim;
	qd_real num = sqrt(dre * dre + dim * dim), den = sqrt(cre * cre + cim * cim);
	if(den == qd_real(0.0)) return to_double(num);
	return to_double(num / den);
}

static double max_relerr_mat(CDSMatrix ref, CGDSMatrix gpu)
{
	long int i, j, n = ref->re->row_dim, m = ref->re->col_dim;
	CGDSMatrix h = init_cgdsmatrix(n, m);
	double maxe = 0.0;
	cudaMemcpy((void *)h->re, (void *)gpu->re, (size_t)(sizeof(gds_real) * n * m), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)gpu->im, (size_t)(sizeof(gds_real) * n * m), cudaMemcpyDeviceToHost);
	for(i = 0; i < n; i++)
		for(j = 0; j < m; j++)
		{
			double e = cds_relerr(get_cdsmatrix_ij_cdsfloat(ref, i, j), h->re[i * m + j], h->im[i * m + j]);
			if(e > maxe) maxe = e;
		}
	free_cgdsmatrix(h);
	return maxe;
}
static double max_relerr_vec(CDSVector ref, CGDSVector gpu)
{
	long int i, n = ref->re->dim;
	CGDSVector h = init_cgdsvector(n);
	double maxe = 0.0;
	cudaMemcpy((void *)h->re, (void *)gpu->re, (size_t)(sizeof(gds_real) * n), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)gpu->im, (size_t)(sizeof(gds_real) * n), cudaMemcpyDeviceToHost);
	for(i = 0; i < n; i++)
	{
		double e = cds_relerr(get_cdsvector_i_cdsfloat(ref, i), h->re[i], h->im[i]);
		if(e > maxe) maxe = e;
	}
	free_cgdsvector(h);
	return maxe;
}

int main(int argc, char *argv[])
{
	long int dim = 192, i, j;
	int nbg = 32, ntb = 64;
	if(argc >= 2) dim = atol(argv[1]);
	fpu_fix_start(NULL);
	cudaSetDevice(0);

	CDSMatrix A = init_cdsmatrix(dim, dim), B = init_cdsmatrix(dim, dim), C = init_cdsmatrix(dim, dim);
	CDSVector x = init_cdsvector(dim), y = init_cdsvector(dim);
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			cdsfloat va = mkcds(sqrt((double)(i + j + 1)), 1.0 / (double)(i + j + 1));
			cdsfloat vb = mkcds(1.0 / (double)(i + j + 2), sqrt((double)(i + 2)));
			set_cdsmatrix_ij(A, i, j, &va);
			set_cdsmatrix_ij(B, i, j, &vb);
		}
		cdsfloat vx = mkcds(sqrt((double)(i + 1)), (double)i);
		set_cdsvector_i(x, i, &vx);
	}

	CGDSMatrix gA = init_cgdsmatrix_dev(dim, dim), gB = init_cgdsmatrix_dev(dim, dim), gC = init_cgdsmatrix_dev(dim, dim);
	CGDSVector gx = init_cgdsvector_dev(dim), gy = init_cgdsvector_dev(dim);
	subst_cgdsmatrix_dev_cdsmat(gA, A);
	subst_cgdsmatrix_dev_cdsmat(gB, B);
	subst_cgdsvector_dev_cdsvec(gx, x);

	printf("===== complex-ds GPU linear (cgdslinear), dim = %ld =====\n", dim);

	mul_cdsmatrix_4m(C, A, B);
	mul_cgdsmatrix_dev(gC, gA, gB, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matmul ] max rel err (GPU vs CPU, dd): %.3e\n", max_relerr_mat(C, gC));

	mul_cdsmatrix_cdsvec_4m(y, A, x);
	mul_cgdsmatrix_cgdsvec(gy, gA, gx, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matvec ] max rel err (GPU vs CPU, dd): %.3e\n", max_relerr_vec(y, gy));

	mul_cdsmatrixt_cdsvec(y, A, x);
	mul_cgdsmatrixt_cgdsvec(gy, gA, gx, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matvecT] max rel err (GPU vs CPU, dd): %.3e\n", max_relerr_vec(y, gy));

	double t0, t1, gpu_s, cpu_s, flop = 8.0 * (double)dim * (double)dim * (double)dim;
	t0 = wtime(); mul_cgdsmatrix_dev(gC, gA, gB, nbg, ntb); cudaDeviceSynchronize(); t1 = wtime(); gpu_s = t1 - t0;
	t0 = wtime(); mul_cdsmatrix_4m(C, A, B); t1 = wtime(); cpu_s = t1 - t0;
	printf("[perf   ] matmul GPU: %.4e s | CPU: %.4e s | speedup x%.1f\n", gpu_s, cpu_s, cpu_s / gpu_s);

	cudaDeviceReset();
	return 0;
}
