/* test_sparse_float.c - verify FRSMatrix SpMV (A*x and A^T*x) vs dense reference. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "flinear.h"
#include "bncsparse.h"

#ifndef BACKEND_NAME
#define BACKEND_NAME "scalar"
#endif

int main(int argc, char **argv)
{
	long n = (argc >= 2) ? atol(argv[1]) : 137;   /* non-multiple of SIMD width */
	long i, j;

	FMatrix A = init_fmatrix(n, n);
	FVector x = init_fvector(n);
	FVector y_dense = init_fvector(n), yt_dense = init_fvector(n);
	FVector y_sp = init_fvector(n), yt_sp = init_fvector(n);

	/* build a sparse-ish dense matrix: keep ~30% of entries, varied row counts */
	for(i = 0; i < n; i++)
	{
		for(j = 0; j < n; j++)
		{
			float v = 0.0f;
			if(((i * 31 + j * 17) % 10) < 3)               /* ~30% density */
				v = (float)(((i * 7 + j * 13) % 97) * 0.01 + 0.05);
			set_fmatrix_ij(A, i, j, v);
		}
		set_fvector_i(x, i, (float)(((i * 29 + 5) % 83) * 0.01 + 0.1));
	}

	/* dense references */
	mul_fmatrix_dvec(y_dense, A, x);   /* legacy name: FMatrix * FVector */
	for(i = 0; i < n; i++) set_fvector_i(yt_dense, i, 0.0f);
	for(i = 0; i < n; i++)
		for(j = 0; j < n; j++)
			yt_dense->element[j] += get_fmatrix_ij(A, i, j) * x->element[i]; /* A^T x */

	/* sparse */
	FRSMatrix As = init_set_frsmatrix_fmatrix(A);
	mul_frsmatrix_fvec(y_sp, As, x);
	mul_frsmatrixt_fvec(yt_sp, As, x);

	/* compare */
	double maxrel = 0.0, maxrel_t = 0.0;
	for(i = 0; i < n; i++)
	{
		double d = fabs((double)y_sp->element[i] - (double)y_dense->element[i]);
		double r = d / (fabs((double)y_dense->element[i]) + 1e-30);
		if(r > maxrel) maxrel = r;
		double dt = fabs((double)yt_sp->element[i] - (double)yt_dense->element[i]);
		double rt = dt / (fabs((double)yt_dense->element[i]) + 1e-30);
		if(rt > maxrel_t) maxrel_t = rt;
	}

	printf("VERIFY,%s,n=%ld,Ax_maxrelerr=%.3e,Atx_maxrelerr=%.3e,nnz=%ld -> %s\n",
	       BACKEND_NAME, n, maxrel, maxrel_t, As->nzero_total_num,
	       (maxrel < 1e-5 && maxrel_t < 1e-5) ? "OK" : "FAIL");

	free_frsmatrix(As);
	free_fmatrix(A); free_fvector(x);
	free_fvector(y_dense); free_fvector(yt_dense); free_fvector(y_sp); free_fvector(yt_sp);
	return 0;
}
