/********************************************************************************/
/* test_cgtsmatmul.cu: verify & benchmark complex-ts GPU linear (cgtslinear)      */
/*   GPU cgts matmul/matvec  vs  CPU complex-ts (mul_ctsmatrix_4m / _ctsvec_4m)     */
/********************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

#include "qd/qd_real.h"
#include "cgtslinear.h"   // pulls gddlinear.h (gts_real, dd_real) + ctslinear.h

extern "C" void mul_ctsmatrix_4m(CTSMatrix, CTSMatrix, CTSMatrix);
extern "C" void mul_ctsmatrix_ctsvec_4m(CTSVector, CTSMatrix, CTSVector);
extern "C" void mul_ctsmatrixt_ctsvec(CTSVector, CTSMatrix, CTSVector);

static double wtime(void) { struct timeval tv; gettimeofday(&tv, NULL); return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6; }

#define CGSCAL float   /* element scalar type (float for single-based families) */
static ctsfloat mkcts(double re, double im) { ctsfloat z; for(int l = 0; l < CGTS_SIZE; l++) { z.val_re[l] = 0.0; z.val_im[l] = 0.0; } z.val_re[0] = re; z.val_im[0] = im; return z; }

// (cpu - gpu)/cpu measured in qd_real (>= any d/s family precision); returned as double
static double cts_relerr(ctsfloat cpu, gts_real gre, gts_real gim)
{
	double cr[4] = {0,0,0,0}, ci[4] = {0,0,0,0}, gr[4] = {0,0,0,0}, gi[4] = {0,0,0,0};
	CGSCAL *pgre = (CGSCAL *)&gre, *pgim = (CGSCAL *)&gim;
	for(int l = 0; l < CGTS_SIZE; l++) { cr[l] = cpu.val_re[l]; ci[l] = cpu.val_im[l]; gr[l] = pgre[l]; gi[l] = pgim[l]; }
	qd_real cre(cr[0], cr[1], cr[2], cr[3]), cim(ci[0], ci[1], ci[2], ci[3]);
	qd_real ggre(gr[0], gr[1], gr[2], gr[3]), ggim(gi[0], gi[1], gi[2], gi[3]);
	qd_real dre = cre - ggre, dim = cim - ggim;
	qd_real num = sqrt(dre * dre + dim * dim), den = sqrt(cre * cre + cim * cim);
	if(den == qd_real(0.0)) return to_double(num);
	return to_double(num / den);
}

static double max_relerr_mat(CTSMatrix ref, CGTSMatrix gpu)
{
	long int i, j, n = ref->re->row_dim, m = ref->re->col_dim;
	CGTSMatrix h = init_cgtsmatrix(n, m);
	double maxe = 0.0;
	cudaMemcpy((void *)h->re, (void *)gpu->re, (size_t)(sizeof(gts_real) * n * m), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)gpu->im, (size_t)(sizeof(gts_real) * n * m), cudaMemcpyDeviceToHost);
	for(i = 0; i < n; i++)
		for(j = 0; j < m; j++)
		{
			double e = cts_relerr(get_ctsmatrix_ij_ctsfloat(ref, i, j), h->re[i * m + j], h->im[i * m + j]);
			if(e > maxe) maxe = e;
		}
	free_cgtsmatrix(h);
	return maxe;
}
static double max_relerr_vec(CTSVector ref, CGTSVector gpu)
{
	long int i, n = ref->re->dim;
	CGTSVector h = init_cgtsvector(n);
	double maxe = 0.0;
	cudaMemcpy((void *)h->re, (void *)gpu->re, (size_t)(sizeof(gts_real) * n), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)gpu->im, (size_t)(sizeof(gts_real) * n), cudaMemcpyDeviceToHost);
	for(i = 0; i < n; i++)
	{
		double e = cts_relerr(get_ctsvector_i_ctsfloat(ref, i), h->re[i], h->im[i]);
		if(e > maxe) maxe = e;
	}
	free_cgtsvector(h);
	return maxe;
}

int main(int argc, char *argv[])
{
	long int dim = 192, i, j;
	int nbg = 32, ntb = 64;
	if(argc >= 2) dim = atol(argv[1]);
	fpu_fix_start(NULL);
	cudaSetDevice(0);

	CTSMatrix A = init_ctsmatrix(dim, dim), B = init_ctsmatrix(dim, dim), C = init_ctsmatrix(dim, dim);
	CTSVector x = init_ctsvector(dim), y = init_ctsvector(dim);
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			ctsfloat va = mkcts(sqrt((double)(i + j + 1)), 1.0 / (double)(i + j + 1));
			ctsfloat vb = mkcts(1.0 / (double)(i + j + 2), sqrt((double)(i + 2)));
			set_ctsmatrix_ij(A, i, j, &va);
			set_ctsmatrix_ij(B, i, j, &vb);
		}
		ctsfloat vx = mkcts(sqrt((double)(i + 1)), (double)i);
		set_ctsvector_i(x, i, &vx);
	}

	CGTSMatrix gA = init_cgtsmatrix_dev(dim, dim), gB = init_cgtsmatrix_dev(dim, dim), gC = init_cgtsmatrix_dev(dim, dim);
	CGTSVector gx = init_cgtsvector_dev(dim), gy = init_cgtsvector_dev(dim);
	subst_cgtsmatrix_dev_ctsmat(gA, A);
	subst_cgtsmatrix_dev_ctsmat(gB, B);
	subst_cgtsvector_dev_ctsvec(gx, x);

	printf("===== complex-ts GPU linear (cgtslinear), dim = %ld =====\n", dim);

	mul_ctsmatrix_4m(C, A, B);
	mul_cgtsmatrix_dev(gC, gA, gB, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matmul ] max rel err (GPU vs CPU, dd): %.3e\n", max_relerr_mat(C, gC));

	mul_ctsmatrix_ctsvec_4m(y, A, x);
	mul_cgtsmatrix_cgtsvec(gy, gA, gx, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matvec ] max rel err (GPU vs CPU, dd): %.3e\n", max_relerr_vec(y, gy));

	mul_ctsmatrixt_ctsvec(y, A, x);
	mul_cgtsmatrixt_cgtsvec(gy, gA, gx, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matvecT] max rel err (GPU vs CPU, dd): %.3e\n", max_relerr_vec(y, gy));

	double t0, t1, gpu_s, cpu_s, flop = 8.0 * (double)dim * (double)dim * (double)dim;
	t0 = wtime(); mul_cgtsmatrix_dev(gC, gA, gB, nbg, ntb); cudaDeviceSynchronize(); t1 = wtime(); gpu_s = t1 - t0;
	t0 = wtime(); mul_ctsmatrix_4m(C, A, B); t1 = wtime(); cpu_s = t1 - t0;
	printf("[perf   ] matmul GPU: %.4e s | CPU: %.4e s | speedup x%.1f\n", gpu_s, cpu_s, cpu_s / gpu_s);

	cudaDeviceReset();
	return 0;
}
