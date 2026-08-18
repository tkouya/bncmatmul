/* test_sparse_cds.c - verify CDSRSMatrix SpMV (A*x) vs dense complex reference. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "cdslinear.h"
#include "bncsparse.h"

#ifndef BACKEND_NAME
#define BACKEND_NAME "scalar"
#endif

static double dre(cdsfloat *v){ double s=0; for(int c=0;c<DSSIZE;c++) s+=(double)v->val_re[c]; return s; }
static double dim_(cdsfloat *v){ double s=0; for(int c=0;c<DSSIZE;c++) s+=(double)v->val_im[c]; return s; }

int main(int argc, char **argv)
{
	long n = (argc >= 2) ? atol(argv[1]) : 137;
	long i, j;

	CDSMatrix A = init_cdsmatrix(n, n);
	CDSVector x = init_cdsvector(n);
	CDSVector y_dense = init_cdsvector(n);
	CDSVector y_sp = init_cdsvector(n);

	for(i = 0; i < n; i++)
	{
		for(j = 0; j < n; j++)
		{
			float _Complex v = 0.0f;
			if(((i * 31 + j * 17) % 10) < 3)
				v = (float)(((i*7+j*13)%97)*0.01+0.05) + (float)(((i*3+j*5)%41)*0.01+0.05) * I;
			set_cdsmatrix_ij_cd(A, i, j, v);
		}
		set_cdsvector_i_cd(x, i, (float)(((i*29+5)%83)*0.01+0.1) + (float)(((i*2+1)%31)*0.01+0.05)*I);
	}

	mul_cdsmatrix_cdsvec_4m(y_dense, A, x);

	CDSRSMatrix As = init_set_cdsrsmatrix_cdsmatrix(A);
	mul_cdsrsmatrix_cdsvec(y_sp, As, x);

	double maxrel = 0.0;
	for(i = 0; i < n; i++)
	{
		cdsfloat *s = get_cdsvector_i(y_sp, i);
		double sre = dre(s), sim = dim_(s);
		cdsfloat *d = get_cdsvector_i(y_dense, i);
		double dr = dre(d), di = dim_(d);
		double mag = sqrt(dr*dr + di*di) + 1e-30;
		double err = sqrt((sre-dr)*(sre-dr) + (sim-di)*(sim-di)) / mag;
		if(err > maxrel) maxrel = err;
	}

	printf("VERIFY,cds,%s,n=%ld,Ax_maxrelerr=%.3e,nnz=%ld -> %s\n",
	       BACKEND_NAME, n, maxrel, As->re->nzero_total_num, (maxrel < 1e-5) ? "OK" : "FAIL");
	return 0;
}
