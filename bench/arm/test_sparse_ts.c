/* test_sparse_ts.c - verify TSRSMatrix SpMV (A*x and A^T*x) vs dense reference. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "tslinear.h"
#include "bncsparse.h"

#ifndef BACKEND_NAME
#define BACKEND_NAME "scalar"
#endif

static double dval(float v[TSSIZE]) { return (double)v[0] + (double)v[1]; }

int main(int argc, char **argv)
{
	long n = (argc >= 2) ? atol(argv[1]) : 137;
	long i, j, c;
	float tmp[TSSIZE];

	TSMatrix A = init_tsmatrix(n, n);
	TSVector x = init_tsvector(n);
	TSVector y_dense = init_tsvector(n);
	TSVector y_sp = init_tsvector(n), yt_sp = init_tsvector(n);

	/* sparse-ish dense matrix */
	for(i = 0; i < n; i++)
	{
		for(j = 0; j < n; j++)
		{
			float v = 0.0f;
			if(((i * 31 + j * 17) % 10) < 3)
				v = (float)(((i * 7 + j * 13) % 97) * 0.01 + 0.05);
			set_tsmatrix_ij_f(A, i, j, v);
		}
		{ float xv[TSSIZE] = {0}; xv[0] = (float)(((i * 29 + 5) % 83) * 0.01 + 0.1); set_tsvector_i(x, i, xv); }
	}

	mul_tsmatrix_tsvec(y_dense, A, x);

	/* dense A^T x reference (as double) */
	double *ytref = (double *)calloc(n, sizeof(double));
	for(i = 0; i < n; i++)
		for(j = 0; j < n; j++)
		{
			float *aij = get_tsmatrix_ij(A, i, j);
			float *xi  = get_tsvector_i(x, i);
			ytref[j] += dval(aij) * dval(xi);
		}

	TSRSMatrix As = init_set_tsrsmatrix_tsmatrix(A);
	mul_tsrsmatrix_tsvec(y_sp, As, x);
	mul_tsrsmatrixt_tsvec(yt_sp, As, x);

	double maxrel = 0.0, maxrel_t = 0.0;
	for(i = 0; i < n; i++)
	{
		double ds = dval(get_tsvector_i(y_sp, i));
		double dd = dval(get_tsvector_i(y_dense, i));
		double r = fabs(ds - dd) / (fabs(dd) + 1e-30);
		if(r > maxrel) maxrel = r;
		double dt = dval(get_tsvector_i(yt_sp, i));
		double rt = fabs(dt - ytref[i]) / (fabs(ytref[i]) + 1e-30);
		if(rt > maxrel_t) maxrel_t = rt;
	}

	printf("VERIFY,ts,%s,n=%ld,Ax_maxrelerr=%.3e,Atx_maxrelerr=%.3e,nnz=%ld -> %s\n",
	       BACKEND_NAME, n, maxrel, maxrel_t, As->nzero_total_num,
	       (maxrel < 1e-5 && maxrel_t < 1e-5) ? "OK" : "FAIL");

	(void)c; (void)tmp; free(ytref);
	return 0;
}
