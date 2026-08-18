/********************************************************************************/
/* test_gfsparse.cu: verify & benchmark native float GPU sparse SpMV (gfsparse)   */
/*   GPU CSR SpMV (A*x, A^T*x)  vs  CPU double DRS sparse reference (single tol)    */
/********************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

extern "C" {
#include "bncsparse.h"   // DRSMatrix, init_set_drsmatrix_dmatrix, mul_drsmatrix_dvec
}
#include "flinear.h"
#include "gflinear.h"
#include "gfsparse.h"

static double wtime(void) { struct timeval tv; gettimeofday(&tv, NULL); return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6; }

static void drs_to_csr_f(DRSMatrix m, long int **row_ptr, long int **col_idx, float **val, long int *nnz_out)
{
	long int i, j, k = 0, base = 0, nnz = m->nzero_total_num;
	long int *rp = (long int *)malloc((m->row_dim + 1) * sizeof(long int));
	long int *ci = (long int *)malloc(nnz * sizeof(long int));
	float    *vv = (float    *)malloc(nnz * sizeof(float));
	rp[0] = 0;
	for(i = 0; i < m->row_dim; i++)
	{
		for(j = 0; j < m->nzero_col_dim[i]; j++)
		{
			ci[k] = m->nzero_index[i][j];
			vv[k] = (float)m->element[base + j];
			k++;
		}
		base += m->real_nzero_col_dim[i];
		rp[i + 1] = k;
	}
	*row_ptr = rp; *col_idx = ci; *val = vv; *nnz_out = nnz;
}

static double max_relerr_vec(DVector ref, GFVector gpu_dev)
{
	long int i, n = ref->dim;
	GFVector h = init_gfvector(n);
	double maxe = 0.0;
	cudaMemcpy((void *)h->element, (void *)gpu_dev->element, (size_t)(sizeof(float) * n), cudaMemcpyDeviceToHost);
	for(i = 0; i < n; i++)
	{
		double a = ref->element[i], b = (double)h->element[i], e = fabs(a - b);
		if(a != 0.0) e /= fabs(a);
		if(e > maxe) maxe = e;
	}
	free_gfvector(h);
	return maxe;
}

int main(int argc, char *argv[])
{
	long int dim = 4096, i;
	int nbg = 64, ntb = 128;
	if(argc >= 2) dim = atol(argv[1]);
	cudaSetDevice(0);

	DMatrix dense = init_dmatrix(dim, dim);
	DVector x = init_dvector(dim), y = init_dvector(dim);
	FVector xf = init_fvector(dim);
	for(i = 0; i < dim; i++)
	{
		set_dmatrix_ij(dense, i, i, 2.0);
		if(i > 0)       set_dmatrix_ij(dense, i, i - 1, -1.0);
		if(i < dim - 1) set_dmatrix_ij(dense, i, i + 1, -1.0);
		set_dmatrix_ij(dense, i, (i * 7 + 3) % dim, 0.5);
		set_dvector_i(x, i, sqrt((double)(i + 1)));
		set_fvector_i(xf, i, (float)sqrt((double)(i + 1)));
	}

	DRSMatrix drs = init_set_drsmatrix_dmatrix(dense);

	long int *row_ptr, *col_idx, nnz;
	float *val;
	drs_to_csr_f(drs, &row_ptr, &col_idx, &val, &nnz);

	GFSPMatrix gA = init_gfspmatrix_dev(dim, dim, nnz, row_ptr, col_idx, val);
	GFVector gx = init_gfvector_dev(dim), gy = init_gfvector_dev(dim);
	subst_gfvector_dev_fvec(gx, xf);

	printf("===== native float GPU sparse SpMV (gfsparse), dim = %ld, nnz = %ld =====\n", dim, nnz);

	mul_drsmatrix_dvec(y, drs, x);
	mul_gfspmatrix_gfvec(gy, gA, gx, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[SpMV  A*x ] max rel err (GPU float vs CPU double): %.3e\n", max_relerr_vec(y, gy));

	mul_drsmatrixt_dvec(y, drs, x);
	mul_gfspmatrixt_gfvec(gy, gA, gx, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[SpMV A^T*x] max rel err (GPU float vs CPU double): %.3e\n", max_relerr_vec(y, gy));

	int reps = 100;
	double t0, t1, gpu_s, cpu_s, flop = 2.0 * (double)nnz;
	t0 = wtime();
	for(int r = 0; r < reps; r++) mul_gfspmatrix_gfvec(gy, gA, gx, nbg, ntb);
	cudaDeviceSynchronize();
	t1 = wtime(); gpu_s = (t1 - t0) / reps;
	t0 = wtime();
	for(int r = 0; r < reps; r++) mul_drsmatrix_dvec(y, drs, x);
	t1 = wtime(); cpu_s = (t1 - t0) / reps;
	printf("[perf      ] SpMV GPU: %.4e s (%.2f GFLOPS) | CPU(double): %.4e s (%.2f GFLOPS) | speedup x%.1f\n",
		gpu_s, flop / gpu_s * 1.0e-9, cpu_s, flop / cpu_s * 1.0e-9, cpu_s / gpu_s);

	free(row_ptr); free(col_idx); free(val);
	free_dmatrix(dense); free_dvector(x); free_dvector(y); free_fvector(xf);
	free_gfspmatrix_dev(gA); free_gfvector_dev(gx); free_gfvector_dev(gy);
	cudaDeviceReset();
	return 0;
}
