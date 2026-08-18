/********************************************************************************/
/* test_cgddsparse.cu: complex-dd GPU sparse SpMV (cgddsparse)                     */
/*   GPU CSR SpMV (A*x, A^T*x) vs CPU dense complex-dd matvec (mul_cddmatrix_*).    */
/********************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

#include "cgddsparse.h"
#include "cddlinear.h"

extern "C" void mul_cddmatrix_cddvec_4m(CDDVector, CDDMatrix, CDDVector);
extern "C" void mul_cddmatrixt_cddvec(CDDVector, CDDMatrix, CDDVector);

static double wtime(void) { struct timeval tv; gettimeofday(&tv, NULL); return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6; }

#define SPSIZE DDSIZE
typedef double sp_scalar;

static cddfloat mkcdd(double re, double im) { cddfloat z; for(int l = 0; l < SPSIZE; l++) { z.val_re[l] = 0; z.val_im[l] = 0; } z.val_re[0] = re; z.val_im[0] = im; return z; }

static void dense_to_csr(CDDMatrix a, long int n, long int **rp_o, long int **ci_o, gdd_real **vr_o, gdd_real **vi_o, long int *nnz_o)
{
	long int i, j, k = 0, nnz = 0;
	for(i = 0; i < n; i++) for(j = 0; j < n; j++) { cddfloat z = get_cddmatrix_ij_cddfloat(a, i, j); int nz = 0; for(int l = 0; l < SPSIZE; l++) if(z.val_re[l] != 0 || z.val_im[l] != 0) nz = 1; if(nz) nnz++; }
	long int *rp = (long int *)malloc((n + 1) * sizeof(long int));
	long int *ci = (long int *)malloc(nnz * sizeof(long int));
	gdd_real *vr = (gdd_real *)malloc(nnz * sizeof(gdd_real)), *vi = (gdd_real *)malloc(nnz * sizeof(gdd_real));
	rp[0] = 0;
	for(i = 0; i < n; i++)
	{
		for(j = 0; j < n; j++)
		{
			cddfloat z = get_cddmatrix_ij_cddfloat(a, i, j);
			int nz = 0; for(int l = 0; l < SPSIZE; l++) if(z.val_re[l] != 0 || z.val_im[l] != 0) nz = 1;
			if(nz) { ci[k] = j; sp_scalar *pr = (sp_scalar *)&vr[k], *pii = (sp_scalar *)&vi[k]; for(int l = 0; l < SPSIZE; l++) { pr[l] = z.val_re[l]; pii[l] = z.val_im[l]; } k++; }
		}
		rp[i + 1] = k;
	}
	*rp_o = rp; *ci_o = ci; *vr_o = vr; *vi_o = vi; *nnz_o = nnz;
}

static void transpose_csr(long int n, long int nnz, const long int *rp, const long int *ci, const gdd_real *vr, const gdd_real *vi,
                          long int **trp_o, long int **tci_o, gdd_real **tvr_o, gdd_real **tvi_o)
{
	long int i, k;
	long int *trp = (long int *)calloc(n + 1, sizeof(long int));
	long int *tci = (long int *)malloc(nnz * sizeof(long int));
	gdd_real *tvr = (gdd_real *)malloc(nnz * sizeof(gdd_real)), *tvi = (gdd_real *)malloc(nnz * sizeof(gdd_real));
	for(k = 0; k < nnz; k++) trp[ci[k] + 1]++;
	for(i = 0; i < n; i++) trp[i + 1] += trp[i];
	long int *off = (long int *)malloc(n * sizeof(long int));
	for(i = 0; i < n; i++) off[i] = trp[i];
	for(i = 0; i < n; i++) for(k = rp[i]; k < rp[i + 1]; k++) { long int c = ci[k], p = off[c]++; tci[p] = i; tvr[p] = vr[k]; tvi[p] = vi[k]; }
	free(off);
	*trp_o = trp; *tci_o = tci; *tvr_o = tvr; *tvi_o = tvi;
}

static double max_relerr(CDDVector ref, gdd_real *gyre, gdd_real *gyim, long int n)
{
	gdd_real *hr = (gdd_real *)malloc(n * sizeof(gdd_real)), *hi = (gdd_real *)malloc(n * sizeof(gdd_real));
	double maxe = 0.0;
	cudaMemcpy((void *)hr, (void *)gyre, n * sizeof(gdd_real), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)hi, (void *)gyim, n * sizeof(gdd_real), cudaMemcpyDeviceToHost);
	for(long int i = 0; i < n; i++)
	{
		cddfloat zc = get_cddvector_i_cddfloat(ref, i);
		double cr[4] = {0,0,0,0}, cii[4] = {0,0,0,0}, gr[4] = {0,0,0,0}, gi[4] = {0,0,0,0};
		sp_scalar *pgr = (sp_scalar *)&hr[i], *pgi = (sp_scalar *)&hi[i];
		for(int l = 0; l < SPSIZE; l++) { cr[l] = zc.val_re[l]; cii[l] = zc.val_im[l]; gr[l] = pgr[l]; gi[l] = pgi[l]; }
		qd_real cre(cr[0],cr[1],cr[2],cr[3]), cim(cii[0],cii[1],cii[2],cii[3]), gre(gr[0],gr[1],gr[2],gr[3]), gim(gi[0],gi[1],gi[2],gi[3]);
		qd_real dre = cre - gre, dim = cim - gim;
		qd_real num = sqrt(dre * dre + dim * dim), den = sqrt(cre * cre + cim * cim);
		double e = (den == qd_real(0.0)) ? to_double(num) : to_double(num / den);
		if(e > maxe) maxe = e;
	}
	free(hr); free(hi);
	return maxe;
}

int main(int argc, char *argv[])
{
	long int dim = 600, i;
	int nbg = 64, ntb = 128;
	if(argc >= 2) dim = atol(argv[1]);
	fpu_fix_start(NULL);
	cudaSetDevice(0);

	CDDMatrix dense = init_cddmatrix(dim, dim);
	CDDVector x = init_cddvector(dim), y = init_cddvector(dim);
	for(i = 0; i < dim; i++)
	{
		cddfloat d2 = mkcdd(2.0, 0.5), dm = mkcdd(-1.0, 0.25), dp = mkcdd(-1.0, -0.25), dh = mkcdd(0.5, 0.1);
		set_cddmatrix_ij(dense, i, i, &d2);
		if(i > 0)       set_cddmatrix_ij(dense, i, i - 1, &dm);
		if(i < dim - 1) set_cddmatrix_ij(dense, i, i + 1, &dp);
		long int jj = (i * 7 + 3) % dim;
		if(jj != i && jj != i - 1 && jj != i + 1) set_cddmatrix_ij(dense, i, jj, &dh);
		cddfloat vx = mkcdd(sqrt((double)(i + 1)), (double)i);
		set_cddvector_i(x, i, &vx);
	}

	long int *rp, *ci, nnz; gdd_real *vr, *vi;
	dense_to_csr(dense, dim, &rp, &ci, &vr, &vi, &nnz);
	CGDDSPMatrix gA = init_cgddspmatrix_dev(dim, dim, nnz, rp, ci, vr, vi);

	gdd_real *hxr = (gdd_real *)malloc(dim * sizeof(gdd_real)), *hxi = (gdd_real *)malloc(dim * sizeof(gdd_real));
	for(i = 0; i < dim; i++) { cddfloat z = get_cddvector_i_cddfloat(x, i); sp_scalar *pr = (sp_scalar *)&hxr[i], *pii = (sp_scalar *)&hxi[i]; for(int l = 0; l < SPSIZE; l++) { pr[l] = z.val_re[l]; pii[l] = z.val_im[l]; } }
	gdd_real *gxr, *gxi, *gyr, *gyi;
	cudaMalloc((void **)&gxr, dim * sizeof(gdd_real)); cudaMalloc((void **)&gxi, dim * sizeof(gdd_real));
	cudaMalloc((void **)&gyr, dim * sizeof(gdd_real)); cudaMalloc((void **)&gyi, dim * sizeof(gdd_real));
	cudaMemcpy((void *)gxr, (void *)hxr, dim * sizeof(gdd_real), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)gxi, (void *)hxi, dim * sizeof(gdd_real), cudaMemcpyHostToDevice);

	printf("===== complex-dd GPU sparse SpMV (cgddsparse), dim = %ld, nnz = %ld =====\n", dim, nnz);

	mul_cddmatrix_cddvec_4m(y, dense, x);
	mul_cgddspmatrix(gyr, gyi, gA, gxr, gxi, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[SpMV  A*x ] max rel err (GPU sparse vs CPU dense, dd): %.3e\n", max_relerr(y, gyr, gyi, dim));

	long int *trp, *tci; gdd_real *tvr, *tvi;
	transpose_csr(dim, nnz, rp, ci, vr, vi, &trp, &tci, &tvr, &tvi);
	CGDDSPMatrix gAt = init_cgddspmatrix_dev(dim, dim, nnz, trp, tci, tvr, tvi);
	mul_cddmatrixt_cddvec(y, dense, x);
	mul_cgddspmatrix(gyr, gyi, gAt, gxr, gxi, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[SpMV A^T*x] max rel err (GPU sparse vs CPU dense, dd): %.3e\n", max_relerr(y, gyr, gyi, dim));

	int reps = 200; double t0, t1, gpu_s;
	t0 = wtime(); for(int r = 0; r < reps; r++) mul_cgddspmatrix(gyr, gyi, gA, gxr, gxi, nbg, ntb); cudaDeviceSynchronize(); t1 = wtime(); gpu_s = (t1 - t0) / reps;
	printf("[perf      ] SpMV GPU: %.4e s (%.2f Gnz/s)\n", gpu_s, (double)nnz / gpu_s * 1.0e-9);

	cudaDeviceReset();
	return 0;
}
