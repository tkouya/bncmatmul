/********************************************************************************/
/* test_gqdsparse.cu: verify & benchmark qd GPU sparse SpMV (gqdsparse)           */
/*   GPU CSR SpMV (A*x, A^T*x) vs CPU qd DENSE matvec (mul_qdmatrix_qdvec / _t).    */
/*   CSR is built directly from a sparse-pattern dense matrix (the CPU *RS sparse  */
/*   converters/SpMV are buggy for td/qd -- init_set_qdrsmatrix_qdmatrix segfaults  */
/*   and the serial td SpMV is inaccurate -- so we reference the verified dense     */
/*   matvec instead).                                                              */
/********************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <sys/time.h>

#include "gddlinear.h"
#include "gqdsparse.h"

static double wtime(void) { struct timeval tv; gettimeofday(&tv, NULL); return (double)tv.tv_sec + (double)tv.tv_usec * 1.0e-6; }

extern "C" void mul_qdmatrix_qdvec(QDVector, QDMatrix, QDVector);
extern "C" void mul_qdmatrixt_qdvec(QDVector, QDMatrix, QDVector);

#define SPSIZE QDSIZE          /* number of limbs (sed: TDSIZE/QDSIZE/DSSIZE/...) */
typedef double sp_scalar;      /* limb scalar (sed: float for s-family) */

// build standard CSR (gqd_real values) directly from a dense QDMatrix
static void dense_to_csr(QDMatrix a, long int n, long int **rp_o, long int **ci_o, gqd_real **v_o, long int *nnz_o)
{
	long int i, j, k = 0, nnz = 0;
	for(i = 0; i < n; i++)
		for(j = 0; j < n; j++)
		{
			qdfloat _z = get_qdmatrix_ij_qdfloat(a, i, j); sp_scalar *c = _z.val;
			int nz = 0; for(int l = 0; l < SPSIZE; l++) if(c[l] != 0) nz = 1;
			if(nz) nnz++;
		}
	long int *rp = (long int *)malloc((n + 1) * sizeof(long int));
	long int *ci = (long int *)malloc(nnz * sizeof(long int));
	gqd_real *vv = (gqd_real *)malloc(nnz * sizeof(gqd_real));
	rp[0] = 0;
	for(i = 0; i < n; i++)
	{
		for(j = 0; j < n; j++)
		{
			qdfloat _z = get_qdmatrix_ij_qdfloat(a, i, j); sp_scalar *c = _z.val;
			int nz = 0; for(int l = 0; l < SPSIZE; l++) if(c[l] != 0) nz = 1;
			if(nz) { ci[k] = j; sp_scalar *pv = (sp_scalar *)&vv[k]; for(int l = 0; l < SPSIZE; l++) pv[l] = c[l]; k++; }
		}
		rp[i + 1] = k;
	}
	*rp_o = rp; *ci_o = ci; *v_o = vv; *nnz_o = nnz;
}

static void transpose_csr(long int n, long int nnz, const long int *rp, const long int *ci, const gqd_real *v,
                          long int **trp_o, long int **tci_o, gqd_real **tv_o)
{
	long int i, k;
	long int *trp = (long int *)calloc(n + 1, sizeof(long int));
	long int *tci = (long int *)malloc(nnz * sizeof(long int));
	gqd_real *tv = (gqd_real *)malloc(nnz * sizeof(gqd_real));
	for(k = 0; k < nnz; k++) trp[ci[k] + 1]++;
	for(i = 0; i < n; i++) trp[i + 1] += trp[i];
	long int *off = (long int *)malloc(n * sizeof(long int));
	for(i = 0; i < n; i++) off[i] = trp[i];
	for(i = 0; i < n; i++)
		for(k = rp[i]; k < rp[i + 1]; k++) { long int c = ci[k], p = off[c]++; tci[p] = i; tv[p] = v[k]; }
	free(off);
	*trp_o = trp; *tci_o = tci; *tv_o = tv;
}

static double max_relerr_vec(QDVector ref, gqd_real *gpu_dev)
{
	long int i, n = ref->dim;
	gqd_real *h = (gqd_real *)malloc(n * sizeof(gqd_real));
	double maxe = 0.0;
	cudaMemcpy((void *)h, (void *)gpu_dev, (size_t)(sizeof(gqd_real) * n), cudaMemcpyDeviceToHost);
	for(i = 0; i < n; i++)
	{
		double cv[4] = {0,0,0,0}, gv[4] = {0,0,0,0};
		qdfloat _z = get_qdvector_i_qdfloat(ref, i); sp_scalar *c = _z.val;
		sp_scalar *pg = (sp_scalar *)&h[i];
		for(int l = 0; l < SPSIZE; l++) { cv[l] = c[l]; gv[l] = pg[l]; }
		qd_real cc(cv[0], cv[1], cv[2], cv[3]), gg(gv[0], gv[1], gv[2], gv[3]);
		qd_real d = cc - gg, num = abs(d), den = abs(cc);
		double e = (den == qd_real(0.0)) ? to_double(num) : to_double(num / den);
		if(e > maxe) maxe = e;
	}
	free(h);
	return maxe;
}

int main(int argc, char *argv[])
{
	long int dim = 1000, i;
	int nbg = 64, ntb = 128;
	if(argc >= 2) dim = atol(argv[1]);
	fpu_fix_start(NULL);
	cudaSetDevice(0);

	QDMatrix dense = init_qdmatrix(dim, dim);
	QDVector x = init_qdvector(dim), y = init_qdvector(dim);
	for(i = 0; i < dim; i++)
	{
		sp_scalar d2[SPSIZE] = {2.0}, dm[SPSIZE] = {-1.0}, dh[SPSIZE] = {0.5};
		set_qdmatrix_ij(dense, i, i, d2);
		if(i > 0)       set_qdmatrix_ij(dense, i, i - 1, dm);
		if(i < dim - 1) set_qdmatrix_ij(dense, i, i + 1, dm);
		if((i * 7 + 3) % dim != i && (i * 7 + 3) % dim != i - 1 && (i * 7 + 3) % dim != i + 1)
			set_qdmatrix_ij(dense, i, (i * 7 + 3) % dim, dh);
		sp_scalar vx[SPSIZE] = {0.0}; vx[0] = (sp_scalar)sqrt((double)(i + 1));
		set_qdvector_i(x, i, vx);
	}

	long int *rp, *ci, nnz; gqd_real *val;
	dense_to_csr(dense, dim, &rp, &ci, &val, &nnz);

	GQDSPMatrix gA = init_gqdspmatrix_dev(dim, dim, nnz, rp, ci, val);
	gqd_real *hx = (gqd_real *)malloc(dim * sizeof(gqd_real));
	for(i = 0; i < dim; i++) { qdfloat _z = get_qdvector_i_qdfloat(x, i); sp_scalar *c = _z.val; sp_scalar *p = (sp_scalar *)&hx[i]; for(int l = 0; l < SPSIZE; l++) p[l] = c[l]; }
	gqd_real *gx, *gy;
	cudaMalloc((void **)&gx, dim * sizeof(gqd_real));
	cudaMalloc((void **)&gy, dim * sizeof(gqd_real));
	cudaMemcpy((void *)gx, (void *)hx, dim * sizeof(gqd_real), cudaMemcpyHostToDevice);

	printf("===== qd GPU sparse SpMV (gqdsparse), dim = %ld, nnz = %ld =====\n", dim, nnz);

	mul_qdmatrix_qdvec(y, dense, x);                 // reference A*x (verified dense matvec)
	mul_gqdspmatrix(gy, gA, gx, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[SpMV  A*x ] max rel err (GPU sparse vs CPU dense, qd): %.3e\n", max_relerr_vec(y, gy));

	long int *trp, *tci; gqd_real *tval;
	transpose_csr(dim, nnz, rp, ci, val, &trp, &tci, &tval);
	GQDSPMatrix gAt = init_gqdspmatrix_dev(dim, dim, nnz, trp, tci, tval);
	mul_qdmatrixt_qdvec(y, dense, x);                // reference A^T*x
	mul_gqdspmatrix(gy, gAt, gx, nbg, ntb);
	cudaDeviceSynchronize();
	printf("[SpMV A^T*x] max rel err (GPU sparse vs CPU dense, qd): %.3e\n", max_relerr_vec(y, gy));

	int reps = 200;
	double t0, t1, gpu_s;
	t0 = wtime(); for(int r = 0; r < reps; r++) mul_gqdspmatrix(gy, gA, gx, nbg, ntb); cudaDeviceSynchronize(); t1 = wtime(); gpu_s = (t1 - t0) / reps;
	printf("[perf      ] SpMV GPU: %.4e s (%.2f Gnz/s)\n", gpu_s, (double)nnz / gpu_s * 1.0e-9);

	cudaDeviceReset();
	return 0;
}
