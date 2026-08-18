/********************************************************************************/
/* test_gcdsparse.cu: native complex double GPU sparse SpMV (gcdsparse)            */
/*   GPU CSR SpMV (A*x, A^T*x) vs CPU dense complex matvec (mul_cdmatrix_cdvec / _t)*/
/********************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

#include "cdlinear.h"
#include "gcdsparse.h"

static inline double _cre(double _Complex z) { return __real__ z; }
static inline double _cim(double _Complex z) { return __imag__ z; }
static inline double _Complex _mkc(double re, double im) { double _Complex z; __real__ z = re; __imag__ z = im; return z; }
static inline double _cabs2(double _Complex z) { double r = __real__ z, i = __imag__ z; return sqrt(r * r + i * i); }
static double wtime(void) { struct timeval tv; gettimeofday(&tv, NULL); return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6; }

static void dense_to_csr(CDMatrix a, long int n, long int **rp_o, long int **ci_o, double **vr_o, double **vi_o, long int *nnz_o)
{
	long int i, j, k = 0, nnz = 0;
	for(i = 0; i < n; i++) for(j = 0; j < n; j++) { double _Complex z = get_cdmatrix_ij(a, i, j); if(_cre(z) != 0.0 || _cim(z) != 0.0) nnz++; }
	long int *rp = (long int *)malloc((n + 1) * sizeof(long int));
	long int *ci = (long int *)malloc(nnz * sizeof(long int));
	double *vr = (double *)malloc(nnz * sizeof(double)), *vi = (double *)malloc(nnz * sizeof(double));
	rp[0] = 0;
	for(i = 0; i < n; i++)
	{
		for(j = 0; j < n; j++)
		{
			double _Complex z = get_cdmatrix_ij(a, i, j);
			if(_cre(z) != 0.0 || _cim(z) != 0.0) { ci[k] = j; vr[k] = _cre(z); vi[k] = _cim(z); k++; }
		}
		rp[i + 1] = k;
	}
	*rp_o = rp; *ci_o = ci; *vr_o = vr; *vi_o = vi; *nnz_o = nnz;
}

static void transpose_csr(long int n, long int nnz, const long int *rp, const long int *ci, const double *vr, const double *vi,
                          long int **trp_o, long int **tci_o, double **tvr_o, double **tvi_o)
{
	long int i, k;
	long int *trp = (long int *)calloc(n + 1, sizeof(long int));
	long int *tci = (long int *)malloc(nnz * sizeof(long int));
	double *tvr = (double *)malloc(nnz * sizeof(double)), *tvi = (double *)malloc(nnz * sizeof(double));
	for(k = 0; k < nnz; k++) trp[ci[k] + 1]++;
	for(i = 0; i < n; i++) trp[i + 1] += trp[i];
	long int *off = (long int *)malloc(n * sizeof(long int));
	for(i = 0; i < n; i++) off[i] = trp[i];
	for(i = 0; i < n; i++) for(k = rp[i]; k < rp[i + 1]; k++) { long int c = ci[k], p = off[c]++; tci[p] = i; tvr[p] = vr[k]; tvi[p] = vi[k]; }
	free(off);
	*trp_o = trp; *tci_o = tci; *tvr_o = tvr; *tvi_o = tvi;
}

static double max_relerr(CDVector ref, double *gyre, double *gyim, long int n)
{
	double *hr = (double *)malloc(n * sizeof(double)), *hi = (double *)malloc(n * sizeof(double)), maxe = 0.0;
	cudaMemcpy((void *)hr, (void *)gyre, n * sizeof(double), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)hi, (void *)gyim, n * sizeof(double), cudaMemcpyDeviceToHost);
	for(long int i = 0; i < n; i++)
	{
		double _Complex a = get_cdvector_i(ref, i), b = _mkc(hr[i], hi[i]);
		double e = _cabs2(a - b);
		if(_cabs2(a) != 0.0) e /= _cabs2(a);
		if(e > maxe) maxe = e;
	}
	free(hr); free(hi);
	return maxe;
}

int main(int argc, char *argv[])
{
	long int dim = 1000, i;
	int nbg = 64, ntb = 128;
	if(argc >= 2) dim = atol(argv[1]);
	cudaSetDevice(0);

	CDMatrix dense = init_cdmatrix(dim, dim);
	CDVector x = init_cdvector(dim), y = init_cdvector(dim);
	for(i = 0; i < dim; i++)
	{
		set_cdmatrix_ij(dense, i, i, _mkc(2.0, 0.5));
		if(i > 0)       set_cdmatrix_ij(dense, i, i - 1, _mkc(-1.0, 0.25));
		if(i < dim - 1) set_cdmatrix_ij(dense, i, i + 1, _mkc(-1.0, -0.25));
		long int jj = (i * 7 + 3) % dim;
		if(jj != i && jj != i - 1 && jj != i + 1) set_cdmatrix_ij(dense, i, jj, _mkc(0.5, 0.1));
		set_cdvector_i(x, i, _mkc(sqrt((double)(i + 1)), (double)i));
	}

	long int *rp, *ci, nnz; double *vr, *vi;
	dense_to_csr(dense, dim, &rp, &ci, &vr, &vi, &nnz);
	GCDSPMatrix gA = init_gcdspmatrix_dev(dim, dim, nnz, rp, ci, vr, vi);

	double *hxr = (double *)malloc(dim * sizeof(double)), *hxi = (double *)malloc(dim * sizeof(double));
	for(i = 0; i < dim; i++) { double _Complex z = get_cdvector_i(x, i); hxr[i] = _cre(z); hxi[i] = _cim(z); }
	double *gxr, *gxi, *gyr, *gyi;
	cudaMalloc((void **)&gxr, dim * sizeof(double)); cudaMalloc((void **)&gxi, dim * sizeof(double));
	cudaMalloc((void **)&gyr, dim * sizeof(double)); cudaMalloc((void **)&gyi, dim * sizeof(double));
	cudaMemcpy((void *)gxr, (void *)hxr, dim * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)gxi, (void *)hxi, dim * sizeof(double), cudaMemcpyHostToDevice);

	printf("===== native complex double GPU sparse SpMV (gcdsparse), dim = %ld, nnz = %ld =====\n", dim, nnz);

	mul_cdmatrix_cdvec(y, dense, x);
	mul_gcdspmatrix(gyr, gyi, gA, gxr, gxi, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[SpMV  A*x ] max rel err (GPU sparse vs CPU dense): %.3e\n", max_relerr(y, gyr, gyi, dim));

	long int *trp, *tci; double *tvr, *tvi;
	transpose_csr(dim, nnz, rp, ci, vr, vi, &trp, &tci, &tvr, &tvi);
	GCDSPMatrix gAt = init_gcdspmatrix_dev(dim, dim, nnz, trp, tci, tvr, tvi);
	mul_cdmatrixt_cdvec(y, dense, x);
	mul_gcdspmatrix(gyr, gyi, gAt, gxr, gxi, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[SpMV A^T*x] max rel err (GPU sparse vs CPU dense): %.3e\n", max_relerr(y, gyr, gyi, dim));

	int reps = 200; double t0, t1, gpu_s;
	t0 = wtime(); for(int r = 0; r < reps; r++) mul_gcdspmatrix(gyr, gyi, gA, gxr, gxi, nbg, ntb); cudaDeviceSynchronize(); t1 = wtime(); gpu_s = (t1 - t0) / reps;
	printf("[perf      ] SpMV GPU: %.4e s (%.2f Gnz/s)\n", gpu_s, (double)nnz / gpu_s * 1.0e-9);

	cudaDeviceReset();
	return 0;
}
