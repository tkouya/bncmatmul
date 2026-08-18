/********************************************************************************/
/* test_cgtdmatmul.cu: verify & benchmark complex-td GPU linear (cgtdlinear)      */
/*   GPU cgtd matmul/matvec  vs  CPU complex-td (mul_ctdmatrix_4m / _ctdvec_4m)     */
/********************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

#include "cgtdlinear.h"   // pulls gddlinear.h (gtd_real, dd_real) + ctdlinear.h

extern "C" void mul_ctdmatrix_4m(CTDMatrix, CTDMatrix, CTDMatrix);
extern "C" void mul_ctdmatrix_ctdvec_4m(CTDVector, CTDMatrix, CTDVector);
extern "C" void mul_ctdmatrixt_ctdvec(CTDVector, CTDMatrix, CTDVector);

static double wtime(void) { struct timeval tv; gettimeofday(&tv, NULL); return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6; }

#define CGSCAL double   /* element scalar type (float for single-based families) */
static ctdfloat mkctd(double re, double im) { ctdfloat z; for(int l = 0; l < CGTD_SIZE; l++) { z.val_re[l] = 0.0; z.val_im[l] = 0.0; } z.val_re[0] = re; z.val_im[0] = im; return z; }

// (cpu - gpu)/cpu measured in qd_real (>= any d/s family precision); returned as double
static double ctd_relerr(ctdfloat cpu, gtd_real gre, gtd_real gim)
{
	double cr[4] = {0,0,0,0}, ci[4] = {0,0,0,0}, gr[4] = {0,0,0,0}, gi[4] = {0,0,0,0};
	CGSCAL *pgre = (CGSCAL *)&gre, *pgim = (CGSCAL *)&gim;
	for(int l = 0; l < CGTD_SIZE; l++) { cr[l] = cpu.val_re[l]; ci[l] = cpu.val_im[l]; gr[l] = pgre[l]; gi[l] = pgim[l]; }
	qd_real cre(cr[0], cr[1], cr[2], cr[3]), cim(ci[0], ci[1], ci[2], ci[3]);
	qd_real ggre(gr[0], gr[1], gr[2], gr[3]), ggim(gi[0], gi[1], gi[2], gi[3]);
	qd_real dre = cre - ggre, dim = cim - ggim;
	qd_real num = sqrt(dre * dre + dim * dim), den = sqrt(cre * cre + cim * cim);
	if(den == qd_real(0.0)) return to_double(num);
	return to_double(num / den);
}

static double max_relerr_mat(CTDMatrix ref, CGTDMatrix gpu)
{
	long int i, j, n = ref->re->row_dim, m = ref->re->col_dim;
	CGTDMatrix h = init_cgtdmatrix(n, m);
	double maxe = 0.0;
	cudaMemcpy((void *)h->re, (void *)gpu->re, (size_t)(sizeof(gtd_real) * n * m), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)gpu->im, (size_t)(sizeof(gtd_real) * n * m), cudaMemcpyDeviceToHost);
	for(i = 0; i < n; i++)
		for(j = 0; j < m; j++)
		{
			double e = ctd_relerr(get_ctdmatrix_ij_ctdfloat(ref, i, j), h->re[i * m + j], h->im[i * m + j]);
			if(e > maxe) maxe = e;
		}
	free_cgtdmatrix(h);
	return maxe;
}
static double max_relerr_vec(CTDVector ref, CGTDVector gpu)
{
	long int i, n = ref->re->dim;
	CGTDVector h = init_cgtdvector(n);
	double maxe = 0.0;
	cudaMemcpy((void *)h->re, (void *)gpu->re, (size_t)(sizeof(gtd_real) * n), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)gpu->im, (size_t)(sizeof(gtd_real) * n), cudaMemcpyDeviceToHost);
	for(i = 0; i < n; i++)
	{
		double e = ctd_relerr(get_ctdvector_i_ctdfloat(ref, i), h->re[i], h->im[i]);
		if(e > maxe) maxe = e;
	}
	free_cgtdvector(h);
	return maxe;
}

int main(int argc, char *argv[])
{
	long int dim = 192, i, j;
	int nbg = 32, ntb = 64;
	if(argc >= 2) dim = atol(argv[1]);
	fpu_fix_start(NULL);
	cudaSetDevice(0);

	CTDMatrix A = init_ctdmatrix(dim, dim), B = init_ctdmatrix(dim, dim), C = init_ctdmatrix(dim, dim);
	CTDVector x = init_ctdvector(dim), y = init_ctdvector(dim);
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			ctdfloat va = mkctd(sqrt((double)(i + j + 1)), 1.0 / (double)(i + j + 1));
			ctdfloat vb = mkctd(1.0 / (double)(i + j + 2), sqrt((double)(i + 2)));
			set_ctdmatrix_ij(A, i, j, &va);
			set_ctdmatrix_ij(B, i, j, &vb);
		}
		ctdfloat vx = mkctd(sqrt((double)(i + 1)), (double)i);
		set_ctdvector_i(x, i, &vx);
	}

	CGTDMatrix gA = init_cgtdmatrix_dev(dim, dim), gB = init_cgtdmatrix_dev(dim, dim), gC = init_cgtdmatrix_dev(dim, dim);
	CGTDVector gx = init_cgtdvector_dev(dim), gy = init_cgtdvector_dev(dim);
	subst_cgtdmatrix_dev_ctdmat(gA, A);
	subst_cgtdmatrix_dev_ctdmat(gB, B);
	subst_cgtdvector_dev_ctdvec(gx, x);

	printf("===== complex-td GPU linear (cgtdlinear), dim = %ld =====\n", dim);

	mul_ctdmatrix_4m(C, A, B);
	mul_cgtdmatrix_dev(gC, gA, gB, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matmul ] max rel err (GPU vs CPU, dd): %.3e\n", max_relerr_mat(C, gC));

	mul_ctdmatrix_ctdvec_4m(y, A, x);
	mul_cgtdmatrix_cgtdvec(gy, gA, gx, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matvec ] max rel err (GPU vs CPU, dd): %.3e\n", max_relerr_vec(y, gy));

	mul_ctdmatrixt_ctdvec(y, A, x);
	mul_cgtdmatrixt_cgtdvec(gy, gA, gx, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matvecT] max rel err (GPU vs CPU, dd): %.3e\n", max_relerr_vec(y, gy));

	double t0, t1, gpu_s, cpu_s, flop = 8.0 * (double)dim * (double)dim * (double)dim;
	t0 = wtime(); mul_cgtdmatrix_dev(gC, gA, gB, nbg, ntb); cudaDeviceSynchronize(); t1 = wtime(); gpu_s = t1 - t0;
	t0 = wtime(); mul_ctdmatrix_4m(C, A, B); t1 = wtime(); cpu_s = t1 - t0;
	printf("[perf   ] matmul GPU: %.4e s | CPU: %.4e s | speedup x%.1f\n", gpu_s, cpu_s, cpu_s / gpu_s);

	cudaDeviceReset();
	return 0;
}
