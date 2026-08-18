/********************************************************************************/
/* test_cgdssparse.cu: complex-ds GPU sparse SpMV (cgdssparse)                     */
/*   GPU CSR SpMV (A*x, A^T*x) vs CPU dense complex-ds matvec (mul_cdsmatrix_*).    */
/********************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

#include "qd/qd_real.h"
#include "cgdssparse.h"
#include "cdslinear.h"

extern "C" void mul_cdsmatrix_cdsvec_4m(CDSVector, CDSMatrix, CDSVector);
extern "C" void mul_cdsmatrixt_cdsvec(CDSVector, CDSMatrix, CDSVector);

static double wtime(void) { struct timeval tv; gettimeofday(&tv, NULL); return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6; }

#define SPSIZE DSSIZE
typedef float sp_scalar;

static cdsfloat mkcds(double re, double im) { cdsfloat z; for(int l = 0; l < SPSIZE; l++) { z.val_re[l] = 0; z.val_im[l] = 0; } z.val_re[0] = re; z.val_im[0] = im; return z; }

static void dense_to_csr(CDSMatrix a, long int n, long int **rp_o, long int **ci_o, gds_real **vr_o, gds_real **vi_o, long int *nnz_o)
{
	long int i, j, k = 0, nnz = 0;
	for(i = 0; i < n; i++) for(j = 0; j < n; j++) { cdsfloat z = get_cdsmatrix_ij_cdsfloat(a, i, j); int nz = 0; for(int l = 0; l < SPSIZE; l++) if(z.val_re[l] != 0 || z.val_im[l] != 0) nz = 1; if(nz) nnz++; }
	long int *rp = (long int *)malloc((n + 1) * sizeof(long int));
	long int *ci = (long int *)malloc(nnz * sizeof(long int));
	gds_real *vr = (gds_real *)malloc(nnz * sizeof(gds_real)), *vi = (gds_real *)malloc(nnz * sizeof(gds_real));
	rp[0] = 0;
	for(i = 0; i < n; i++)
	{
		for(j = 0; j < n; j++)
		{
			cdsfloat z = get_cdsmatrix_ij_cdsfloat(a, i, j);
			int nz = 0; for(int l = 0; l < SPSIZE; l++) if(z.val_re[l] != 0 || z.val_im[l] != 0) nz = 1;
			if(nz) { ci[k] = j; sp_scalar *pr = (sp_scalar *)&vr[k], *pii = (sp_scalar *)&vi[k]; for(int l = 0; l < SPSIZE; l++) { pr[l] = z.val_re[l]; pii[l] = z.val_im[l]; } k++; }
		}
		rp[i + 1] = k;
	}
	*rp_o = rp; *ci_o = ci; *vr_o = vr; *vi_o = vi; *nnz_o = nnz;
}

static void transpose_csr(long int n, long int nnz, const long int *rp, const long int *ci, const gds_real *vr, const gds_real *vi,
                          long int **trp_o, long int **tci_o, gds_real **tvr_o, gds_real **tvi_o)
{
	long int i, k;
	long int *trp = (long int *)calloc(n + 1, sizeof(long int));
	long int *tci = (long int *)malloc(nnz * sizeof(long int));
	gds_real *tvr = (gds_real *)malloc(nnz * sizeof(gds_real)), *tvi = (gds_real *)malloc(nnz * sizeof(gds_real));
	for(k = 0; k < nnz; k++) trp[ci[k] + 1]++;
	for(i = 0; i < n; i++) trp[i + 1] += trp[i];
	long int *off = (long int *)malloc(n * sizeof(long int));
	for(i = 0; i < n; i++) off[i] = trp[i];
	for(i = 0; i < n; i++) for(k = rp[i]; k < rp[i + 1]; k++) { long int c = ci[k], p = off[c]++; tci[p] = i; tvr[p] = vr[k]; tvi[p] = vi[k]; }
	free(off);
	*trp_o = trp; *tci_o = tci; *tvr_o = tvr; *tvi_o = tvi;
}

static double max_relerr(CDSVector ref, gds_real *gyre, gds_real *gyim, long int n)
{
	gds_real *hr = (gds_real *)malloc(n * sizeof(gds_real)), *hi = (gds_real *)malloc(n * sizeof(gds_real));
	double maxe = 0.0;
	cudaMemcpy((void *)hr, (void *)gyre, n * sizeof(gds_real), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)hi, (void *)gyim, n * sizeof(gds_real), cudaMemcpyDeviceToHost);
	for(long int i = 0; i < n; i++)
	{
		cdsfloat zc = get_cdsvector_i_cdsfloat(ref, i);
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

	CDSMatrix dense = init_cdsmatrix(dim, dim);
	CDSVector x = init_cdsvector(dim), y = init_cdsvector(dim);
	for(i = 0; i < dim; i++)
	{
		cdsfloat d2 = mkcds(2.0, 0.5), dm = mkcds(-1.0, 0.25), dp = mkcds(-1.0, -0.25), dh = mkcds(0.5, 0.1);
		set_cdsmatrix_ij(dense, i, i, &d2);
		if(i > 0)       set_cdsmatrix_ij(dense, i, i - 1, &dm);
		if(i < dim - 1) set_cdsmatrix_ij(dense, i, i + 1, &dp);
		long int jj = (i * 7 + 3) % dim;
		if(jj != i && jj != i - 1 && jj != i + 1) set_cdsmatrix_ij(dense, i, jj, &dh);
		cdsfloat vx = mkcds(sqrt((double)(i + 1)), (double)i);
		set_cdsvector_i(x, i, &vx);
	}

	long int *rp, *ci, nnz; gds_real *vr, *vi;
	dense_to_csr(dense, dim, &rp, &ci, &vr, &vi, &nnz);
	CGDSSPMatrix gA = init_cgdsspmatrix_dev(dim, dim, nnz, rp, ci, vr, vi);

	gds_real *hxr = (gds_real *)malloc(dim * sizeof(gds_real)), *hxi = (gds_real *)malloc(dim * sizeof(gds_real));
	for(i = 0; i < dim; i++) { cdsfloat z = get_cdsvector_i_cdsfloat(x, i); sp_scalar *pr = (sp_scalar *)&hxr[i], *pii = (sp_scalar *)&hxi[i]; for(int l = 0; l < SPSIZE; l++) { pr[l] = z.val_re[l]; pii[l] = z.val_im[l]; } }
	gds_real *gxr, *gxi, *gyr, *gyi;
	cudaMalloc((void **)&gxr, dim * sizeof(gds_real)); cudaMalloc((void **)&gxi, dim * sizeof(gds_real));
	cudaMalloc((void **)&gyr, dim * sizeof(gds_real)); cudaMalloc((void **)&gyi, dim * sizeof(gds_real));
	cudaMemcpy((void *)gxr, (void *)hxr, dim * sizeof(gds_real), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)gxi, (void *)hxi, dim * sizeof(gds_real), cudaMemcpyHostToDevice);

	printf("===== complex-ds GPU sparse SpMV (cgdssparse), dim = %ld, nnz = %ld =====\n", dim, nnz);

	mul_cdsmatrix_cdsvec_4m(y, dense, x);
	mul_cgdsspmatrix(gyr, gyi, gA, gxr, gxi, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[SpMV  A*x ] max rel err (GPU sparse vs CPU dense, ds): %.3e\n", max_relerr(y, gyr, gyi, dim));

	long int *trp, *tci; gds_real *tvr, *tvi;
	transpose_csr(dim, nnz, rp, ci, vr, vi, &trp, &tci, &tvr, &tvi);
	CGDSSPMatrix gAt = init_cgdsspmatrix_dev(dim, dim, nnz, trp, tci, tvr, tvi);
	mul_cdsmatrixt_cdsvec(y, dense, x);
	mul_cgdsspmatrix(gyr, gyi, gAt, gxr, gxi, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[SpMV A^T*x] max rel err (GPU sparse vs CPU dense, ds): %.3e\n", max_relerr(y, gyr, gyi, dim));

	int reps = 200; double t0, t1, gpu_s;
	t0 = wtime(); for(int r = 0; r < reps; r++) mul_cgdsspmatrix(gyr, gyi, gA, gxr, gxi, nbg, ntb); cudaDeviceSynchronize(); t1 = wtime(); gpu_s = (t1 - t0) / reps;
	printf("[perf      ] SpMV GPU: %.4e s (%.2f Gnz/s)\n", gpu_s, (double)nnz / gpu_s * 1.0e-9);

	cudaDeviceReset();
	return 0;
}
