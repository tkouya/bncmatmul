/* test_sparse_cqs.c - verify CQSRSMatrix SpMV (A*x) vs dense complex reference. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "cqslinear.h"
#include "bncsparse.h"

#ifndef BACKEND_NAME
#define BACKEND_NAME "scalar"
#endif

static double dre(cqsfloat *v){ double s=0; for(int c=0;c<QSSIZE;c++) s+=(double)v->val_re[c]; return s; }
static double dim_(cqsfloat *v){ double s=0; for(int c=0;c<QSSIZE;c++) s+=(double)v->val_im[c]; return s; }

int main(int argc, char **argv)
{
	long n = (argc >= 2) ? atol(argv[1]) : 137;
	long i, j;

	CQSMatrix A = init_cqsmatrix(n, n);
	CQSVector x = init_cqsvector(n);
	CQSVector y_dense = init_cqsvector(n);
	CQSVector y_sp = init_cqsvector(n);

	for(i = 0; i < n; i++)
	{
		for(j = 0; j < n; j++)
		{
			float _Complex v = 0.0f;
			if(((i * 31 + j * 17) % 10) < 3)
				v = (float)(((i*7+j*13)%97)*0.01+0.05) + (float)(((i*3+j*5)%41)*0.01+0.05) * I;
			set_cqsmatrix_ij_cd(A, i, j, v);
		}
		set_cqsvector_i_cd(x, i, (float)(((i*29+5)%83)*0.01+0.1) + (float)(((i*2+1)%31)*0.01+0.05)*I);
	}

	mul_cqsmatrix_cqsvec_4m(y_dense, A, x);

	CQSRSMatrix As = init_set_cqsrsmatrix_cqsmatrix(A);
	mul_cqsrsmatrix_cqsvec(y_sp, As, x);

	double maxrel = 0.0;
	for(i = 0; i < n; i++)
	{
		cqsfloat *s = get_cqsvector_i(y_sp, i);
		double sre = dre(s), sim = dim_(s);
		cqsfloat *d = get_cqsvector_i(y_dense, i);
		double dr = dre(d), di = dim_(d);
		double mag = sqrt(dr*dr + di*di) + 1e-30;
		double err = sqrt((sre-dr)*(sre-dr) + (sim-di)*(sim-di)) / mag;
		if(err > maxrel) maxrel = err;
	}

	printf("VERIFY,cqs,%s,n=%ld,Ax_maxrelerr=%.3e,nnz=%ld -> %s\n",
	       BACKEND_NAME, n, maxrel, As->re->nzero_total_num, (maxrel < 1e-5) ? "OK" : "FAIL");
	return 0;
}
