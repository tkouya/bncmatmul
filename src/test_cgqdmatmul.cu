/********************************************************************************/
/* test_cgqdmatmul.cu: verify & benchmark complex-qd GPU linear (cgqdlinear)      */
/*   GPU cgqd matmul/matvec  vs  CPU complex-qd (mul_cqdmatrix_4m / _cqdvec_4m)     */
/********************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

#include "cgqdlinear.h"   // pulls gddlinear.h (gqd_real, dd_real) + cqdlinear.h

extern "C" void mul_cqdmatrix_4m(CQDMatrix, CQDMatrix, CQDMatrix);
extern "C" void mul_cqdmatrix_cqdvec_4m(CQDVector, CQDMatrix, CQDVector);
extern "C" void mul_cqdmatrixt_cqdvec(CQDVector, CQDMatrix, CQDVector);

static double wtime(void) { struct timeval tv; gettimeofday(&tv, NULL); return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6; }

#define CGSCAL double   /* element scalar type (float for single-based families) */
static cqdfloat mkcqd(double re, double im) { cqdfloat z; for(int l = 0; l < CGQD_SIZE; l++) { z.val_re[l] = 0.0; z.val_im[l] = 0.0; } z.val_re[0] = re; z.val_im[0] = im; return z; }

// (cpu - gpu)/cpu measured in qd_real (>= any d/s family precision); returned as double
static double cqd_relerr(cqdfloat cpu, gqd_real gre, gqd_real gim)
{
	double cr[4] = {0,0,0,0}, ci[4] = {0,0,0,0}, gr[4] = {0,0,0,0}, gi[4] = {0,0,0,0};
	CGSCAL *pgre = (CGSCAL *)&gre, *pgim = (CGSCAL *)&gim;
	for(int l = 0; l < CGQD_SIZE; l++) { cr[l] = cpu.val_re[l]; ci[l] = cpu.val_im[l]; gr[l] = pgre[l]; gi[l] = pgim[l]; }
	qd_real cre(cr[0], cr[1], cr[2], cr[3]), cim(ci[0], ci[1], ci[2], ci[3]);
	qd_real ggre(gr[0], gr[1], gr[2], gr[3]), ggim(gi[0], gi[1], gi[2], gi[3]);
	qd_real dre = cre - ggre, dim = cim - ggim;
	qd_real num = sqrt(dre * dre + dim * dim), den = sqrt(cre * cre + cim * cim);
	if(den == qd_real(0.0)) return to_double(num);
	return to_double(num / den);
}

static double max_relerr_mat(CQDMatrix ref, CGQDMatrix gpu)
{
	long int i, j, n = ref->re->row_dim, m = ref->re->col_dim;
	CGQDMatrix h = init_cgqdmatrix(n, m);
	double maxe = 0.0;
	cudaMemcpy((void *)h->re, (void *)gpu->re, (size_t)(sizeof(gqd_real) * n * m), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)gpu->im, (size_t)(sizeof(gqd_real) * n * m), cudaMemcpyDeviceToHost);
	for(i = 0; i < n; i++)
		for(j = 0; j < m; j++)
		{
			double e = cqd_relerr(get_cqdmatrix_ij_cqdfloat(ref, i, j), h->re[i * m + j], h->im[i * m + j]);
			if(e > maxe) maxe = e;
		}
	free_cgqdmatrix(h);
	return maxe;
}
static double max_relerr_vec(CQDVector ref, CGQDVector gpu)
{
	long int i, n = ref->re->dim;
	CGQDVector h = init_cgqdvector(n);
	double maxe = 0.0;
	cudaMemcpy((void *)h->re, (void *)gpu->re, (size_t)(sizeof(gqd_real) * n), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)gpu->im, (size_t)(sizeof(gqd_real) * n), cudaMemcpyDeviceToHost);
	for(i = 0; i < n; i++)
	{
		double e = cqd_relerr(get_cqdvector_i_cqdfloat(ref, i), h->re[i], h->im[i]);
		if(e > maxe) maxe = e;
	}
	free_cgqdvector(h);
	return maxe;
}

int main(int argc, char *argv[])
{
	long int dim = 192, i, j;
	int nbg = 32, ntb = 64;
	if(argc >= 2) dim = atol(argv[1]);
	fpu_fix_start(NULL);
	cudaSetDevice(0);

	CQDMatrix A = init_cqdmatrix(dim, dim), B = init_cqdmatrix(dim, dim), C = init_cqdmatrix(dim, dim);
	CQDVector x = init_cqdvector(dim), y = init_cqdvector(dim);
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			cqdfloat va = mkcqd(sqrt((double)(i + j + 1)), 1.0 / (double)(i + j + 1));
			cqdfloat vb = mkcqd(1.0 / (double)(i + j + 2), sqrt((double)(i + 2)));
			set_cqdmatrix_ij(A, i, j, &va);
			set_cqdmatrix_ij(B, i, j, &vb);
		}
		cqdfloat vx = mkcqd(sqrt((double)(i + 1)), (double)i);
		set_cqdvector_i(x, i, &vx);
	}

	CGQDMatrix gA = init_cgqdmatrix_dev(dim, dim), gB = init_cgqdmatrix_dev(dim, dim), gC = init_cgqdmatrix_dev(dim, dim);
	CGQDVector gx = init_cgqdvector_dev(dim), gy = init_cgqdvector_dev(dim);
	subst_cgqdmatrix_dev_cqdmat(gA, A);
	subst_cgqdmatrix_dev_cqdmat(gB, B);
	subst_cgqdvector_dev_cqdvec(gx, x);

	printf("===== complex-qd GPU linear (cgqdlinear), dim = %ld =====\n", dim);

	mul_cqdmatrix_4m(C, A, B);
	mul_cgqdmatrix_dev(gC, gA, gB, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matmul ] max rel err (GPU vs CPU, dd): %.3e\n", max_relerr_mat(C, gC));

	mul_cqdmatrix_cqdvec_4m(y, A, x);
	mul_cgqdmatrix_cgqdvec(gy, gA, gx, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matvec ] max rel err (GPU vs CPU, dd): %.3e\n", max_relerr_vec(y, gy));

	mul_cqdmatrixt_cqdvec(y, A, x);
	mul_cgqdmatrixt_cgqdvec(gy, gA, gx, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[matvecT] max rel err (GPU vs CPU, dd): %.3e\n", max_relerr_vec(y, gy));

	double t0, t1, gpu_s, cpu_s, flop = 8.0 * (double)dim * (double)dim * (double)dim;
	t0 = wtime(); mul_cgqdmatrix_dev(gC, gA, gB, nbg, ntb); cudaDeviceSynchronize(); t1 = wtime(); gpu_s = t1 - t0;
	t0 = wtime(); mul_cqdmatrix_4m(C, A, B); t1 = wtime(); cpu_s = t1 - t0;
	printf("[perf   ] matmul GPU: %.4e s | CPU: %.4e s | speedup x%.1f\n", gpu_s, cpu_s, cpu_s / gpu_s);

	cudaDeviceReset();
	return 0;
}
