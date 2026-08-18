/********************************************************************************/
/* test_cgtdsparse.cu: complex-td GPU sparse SpMV (cgtdsparse)                     */
/*   GPU CSR SpMV (A*x, A^T*x) vs CPU dense complex-td matvec (mul_ctdmatrix_*).    */
/********************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

#include "cgtdsparse.h"
#include "ctdlinear.h"

extern "C" void mul_ctdmatrix_ctdvec_4m(CTDVector, CTDMatrix, CTDVector);
extern "C" void mul_ctdmatrixt_ctdvec(CTDVector, CTDMatrix, CTDVector);

static double wtime(void) { struct timeval tv; gettimeofday(&tv, NULL); return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6; }

#define SPSIZE TDSIZE
typedef double sp_scalar;

static ctdfloat mkctd(double re, double im) { ctdfloat z; for(int l = 0; l < SPSIZE; l++) { z.val_re[l] = 0; z.val_im[l] = 0; } z.val_re[0] = re; z.val_im[0] = im; return z; }

static void dense_to_csr(CTDMatrix a, long int n, long int **rp_o, long int **ci_o, gtd_real **vr_o, gtd_real **vi_o, long int *nnz_o)
{
	long int i, j, k = 0, nnz = 0;
	for(i = 0; i < n; i++) for(j = 0; j < n; j++) { ctdfloat z = get_ctdmatrix_ij_ctdfloat(a, i, j); int nz = 0; for(int l = 0; l < SPSIZE; l++) if(z.val_re[l] != 0 || z.val_im[l] != 0) nz = 1; if(nz) nnz++; }
	long int *rp = (long int *)malloc((n + 1) * sizeof(long int));
	long int *ci = (long int *)malloc(nnz * sizeof(long int));
	gtd_real *vr = (gtd_real *)malloc(nnz * sizeof(gtd_real)), *vi = (gtd_real *)malloc(nnz * sizeof(gtd_real));
	rp[0] = 0;
	for(i = 0; i < n; i++)
	{
		for(j = 0; j < n; j++)
		{
			ctdfloat z = get_ctdmatrix_ij_ctdfloat(a, i, j);
			int nz = 0; for(int l = 0; l < SPSIZE; l++) if(z.val_re[l] != 0 || z.val_im[l] != 0) nz = 1;
			if(nz) { ci[k] = j; sp_scalar *pr = (sp_scalar *)&vr[k], *pii = (sp_scalar *)&vi[k]; for(int l = 0; l < SPSIZE; l++) { pr[l] = z.val_re[l]; pii[l] = z.val_im[l]; } k++; }
		}
		rp[i + 1] = k;
	}
	*rp_o = rp; *ci_o = ci; *vr_o = vr; *vi_o = vi; *nnz_o = nnz;
}

static void transpose_csr(long int n, long int nnz, const long int *rp, const long int *ci, const gtd_real *vr, const gtd_real *vi,
                          long int **trp_o, long int **tci_o, gtd_real **tvr_o, gtd_real **tvi_o)
{
	long int i, k;
	long int *trp = (long int *)calloc(n + 1, sizeof(long int));
	long int *tci = (long int *)malloc(nnz * sizeof(long int));
	gtd_real *tvr = (gtd_real *)malloc(nnz * sizeof(gtd_real)), *tvi = (gtd_real *)malloc(nnz * sizeof(gtd_real));
	for(k = 0; k < nnz; k++) trp[ci[k] + 1]++;
	for(i = 0; i < n; i++) trp[i + 1] += trp[i];
	long int *off = (long int *)malloc(n * sizeof(long int));
	for(i = 0; i < n; i++) off[i] = trp[i];
	for(i = 0; i < n; i++) for(k = rp[i]; k < rp[i + 1]; k++) { long int c = ci[k], p = off[c]++; tci[p] = i; tvr[p] = vr[k]; tvi[p] = vi[k]; }
	free(off);
	*trp_o = trp; *tci_o = tci; *tvr_o = tvr; *tvi_o = tvi;
}

static double max_relerr(CTDVector ref, gtd_real *gyre, gtd_real *gyim, long int n)
{
	gtd_real *hr = (gtd_real *)malloc(n * sizeof(gtd_real)), *hi = (gtd_real *)malloc(n * sizeof(gtd_real));
	double maxe = 0.0;
	cudaMemcpy((void *)hr, (void *)gyre, n * sizeof(gtd_real), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)hi, (void *)gyim, n * sizeof(gtd_real), cudaMemcpyDeviceToHost);
	for(long int i = 0; i < n; i++)
	{
		ctdfloat zc = get_ctdvector_i_ctdfloat(ref, i);
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

	CTDMatrix dense = init_ctdmatrix(dim, dim);
	CTDVector x = init_ctdvector(dim), y = init_ctdvector(dim);
	for(i = 0; i < dim; i++)
	{
		ctdfloat d2 = mkctd(2.0, 0.5), dm = mkctd(-1.0, 0.25), dp = mkctd(-1.0, -0.25), dh = mkctd(0.5, 0.1);
		set_ctdmatrix_ij(dense, i, i, &d2);
		if(i > 0)       set_ctdmatrix_ij(dense, i, i - 1, &dm);
		if(i < dim - 1) set_ctdmatrix_ij(dense, i, i + 1, &dp);
		long int jj = (i * 7 + 3) % dim;
		if(jj != i && jj != i - 1 && jj != i + 1) set_ctdmatrix_ij(dense, i, jj, &dh);
		ctdfloat vx = mkctd(sqrt((double)(i + 1)), (double)i);
		set_ctdvector_i(x, i, &vx);
	}

	long int *rp, *ci, nnz; gtd_real *vr, *vi;
	dense_to_csr(dense, dim, &rp, &ci, &vr, &vi, &nnz);
	CGTDSPMatrix gA = init_cgtdspmatrix_dev(dim, dim, nnz, rp, ci, vr, vi);

	gtd_real *hxr = (gtd_real *)malloc(dim * sizeof(gtd_real)), *hxi = (gtd_real *)malloc(dim * sizeof(gtd_real));
	for(i = 0; i < dim; i++) { ctdfloat z = get_ctdvector_i_ctdfloat(x, i); sp_scalar *pr = (sp_scalar *)&hxr[i], *pii = (sp_scalar *)&hxi[i]; for(int l = 0; l < SPSIZE; l++) { pr[l] = z.val_re[l]; pii[l] = z.val_im[l]; } }
	gtd_real *gxr, *gxi, *gyr, *gyi;
	cudaMalloc((void **)&gxr, dim * sizeof(gtd_real)); cudaMalloc((void **)&gxi, dim * sizeof(gtd_real));
	cudaMalloc((void **)&gyr, dim * sizeof(gtd_real)); cudaMalloc((void **)&gyi, dim * sizeof(gtd_real));
	cudaMemcpy((void *)gxr, (void *)hxr, dim * sizeof(gtd_real), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)gxi, (void *)hxi, dim * sizeof(gtd_real), cudaMemcpyHostToDevice);

	printf("===== complex-td GPU sparse SpMV (cgtdsparse), dim = %ld, nnz = %ld =====\n", dim, nnz);

	mul_ctdmatrix_ctdvec_4m(y, dense, x);
	mul_cgtdspmatrix(gyr, gyi, gA, gxr, gxi, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[SpMV  A*x ] max rel err (GPU sparse vs CPU dense, td): %.3e\n", max_relerr(y, gyr, gyi, dim));

	long int *trp, *tci; gtd_real *tvr, *tvi;
	transpose_csr(dim, nnz, rp, ci, vr, vi, &trp, &tci, &tvr, &tvi);
	CGTDSPMatrix gAt = init_cgtdspmatrix_dev(dim, dim, nnz, trp, tci, tvr, tvi);
	mul_ctdmatrixt_ctdvec(y, dense, x);
	mul_cgtdspmatrix(gyr, gyi, gAt, gxr, gxi, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[SpMV A^T*x] max rel err (GPU sparse vs CPU dense, td): %.3e\n", max_relerr(y, gyr, gyi, dim));

	int reps = 200; double t0, t1, gpu_s;
	t0 = wtime(); for(int r = 0; r < reps; r++) mul_cgtdspmatrix(gyr, gyi, gA, gxr, gxi, nbg, ntb); cudaDeviceSynchronize(); t1 = wtime(); gpu_s = (t1 - t0) / reps;
	printf("[perf      ] SpMV GPU: %.4e s (%.2f Gnz/s)\n", gpu_s, (double)nnz / gpu_s * 1.0e-9);

	cudaDeviceReset();
	return 0;
}
