#include "ddlinear.h"

/* DD */

/* 1. Hilbert Matrix */
void hilbert_ddmatrix(DDMatrix a, long int dim)
{
	long int i, j;
#ifdef __cplusplus
	static dd_real tmp;
#else // __cplusplus
	double tmp[DDSIZE];
#endif // __cplusplus

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(hilbert_ddmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(hilbert_ddmatrix)\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			rdd_set_ui(tmp, (unsigned long)(i + j + 1));
			rdd_ui_div(tmp, 1UL, tmp);
			set_ddmatrix_ij(a, i, j, tmp);
		}
	}
}


/* 2. Lotkin Matrix */
void lotkin_ddmatrix(DDMatrix a, long int dim)
{
	long int i, j;
#ifdef __cplusplus
	static dd_real tmp;
#else // __cplusplus
	double tmp[DDSIZE];
#endif // __cplusplus

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(lotkin_ddmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(lotkin_ddmatrix)\n");
		return;
	}

	/* Lotkin Matrix */
	for(i = 0; i < a->col_dim; i++)
	{
		rdd_set_ui(tmp, 1UL);
		set_ddmatrix_ij(a, 0, i, tmp);
	}

	for(i = 1; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			rdd_set_ui(tmp, (unsigned long)(i + j + 1));
			rdd_ui_div(tmp, 1UL, tmp);
			set_ddmatrix_ij(a, i, j, tmp);
		}
	}
}

/* 3. Frank Matrix */
void frank_ddmatrix(DDMatrix a, long int dim)
{
	long int i, j;
#ifdef __cplusplus
	static dd_real tmp;
#else // __cplusplus
	double tmp[DDSIZE];
#endif // __cplusplus

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(frank_ddmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(frank_ddmatrix)\n");
		return;
	}

	/* Frank Matrix */
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			if(i < j)
			{
				rdd_set_ui(tmp, (unsigned long)(a->col_dim - j));
				set_ddmatrix_ij(a, i, j, tmp);
			}
			else
			{
				rdd_set_ui(tmp, (unsigned long)(a->col_dim - i));
				set_ddmatrix_ij(a, i, j, tmp);
			}
		}
	}
}

/* 4. Tridiagonal Matrix */
void tridiag_ddmatrix(DDMatrix a, DDVector low_subdiag, DDVector diag, DDVector up_subdiag, long int dim)
{
	long int i, j;
#ifdef __cplusplus
	static dd_real tmp;
#else // __cplusplus
	double tmp[DDSIZE];
#endif // __cplusplus

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(tridiag_ddmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(tridiag_ddmatrix)\n");
		return;
	}

	/* Tridiagonal Matrix */
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < i - 1; j++)
		{
			rdd_set_ui(tmp, 0UL);
			set_ddmatrix_ij(a, i, j, tmp);
		}
		for(j = i + 2; j < a->col_dim; j++)
		{
			rdd_set_ui(tmp, 0UL);
			set_ddmatrix_ij(a, i, j, tmp);
		}
	}

	set_ddmatrix_ij(a, 0, 0, get_ddvector_i(diag, 0));
	set_ddmatrix_ij(a, 0, 1, get_ddvector_i(up_subdiag , 0));
	for(i = 1; i < a->row_dim - 1; i++)
	{
		set_ddmatrix_ij(a, i, i - 1, get_ddvector_i(low_subdiag, i));
		set_ddmatrix_ij(a, i, i    , get_ddvector_i(diag, i));
		set_ddmatrix_ij(a, i, i + 1, get_ddvector_i(up_subdiag , i));
	}
	i = a->row_dim - 1;
	set_ddmatrix_ij(a, i, i - 1, get_ddvector_i(low_subdiag , i));
	set_ddmatrix_ij(a, i, i    , get_ddvector_i(diag, i));

}


/* 5. Integer Symmetrix Random Matrix */
void int_sym_rand_ddmatrix(DDMatrix mat, long int max, long int seed, long int dim)
{
	long int i, j;
#ifdef __cplusplus
	static dd_real tmp;
#else // __cplusplus
	double tmp[DDSIZE];
#endif // __cplusplus

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(int_sym_rand_ddmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(int_sym_rand_ddmatrix)\n");
		return;
	}

	srand((int)seed);
	for(i = 0; i < dim; i++)
		for(j = i; j < dim; j++)
		{
			rdd_set_ui(tmp, (unsigned long)(rand() % max));
			set_ddmatrix_ij(mat, i, j, tmp);
		}

	for(i = 0; i < dim; i++)
		for(j = 0; j < i; j++)
			set_ddmatrix_ij(mat, i, j, get_ddmatrix_ij(mat, j, i));

}

/* 6. Integer Unsymmetrix Random Matrix */
void int_unsym_rand_ddmatrix(DDMatrix mat, long int max, long int seed, long int dim)
{
	long int i, j;
#ifdef __cplusplus
	static dd_real tmp;
#else // __cplusplus
	double tmp[DDSIZE];
#endif // __cplusplus

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(int_unsym_rand_ddmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(int_unsym_rand_ddmatrix)\n");
		return;
	}

	srand((int)seed);
	for(i = 0; i < dim; i++)
		for(j = 0; j < dim; j++)
		{
			rdd_set_ui(tmp, (unsigned long)(rand() % max));
			set_ddmatrix_ij(mat, i, j, tmp);
		}
}

/* 7. Real Diagonal Matrix */
void diag_ddmatrix(DDMatrix mat, DDVector diag, long int dim)
{
	long int i;

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(diag_ddmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(diag_ddmatrix)\n");
		return;
	}

	set0_ddmatrix(mat);
	for(i = 0; i < dim; i++)
		set_ddmatrix_ij(mat, i, i, get_ddvector_i(diag, i));

}

/* 8. Toeplitz Matrix */
#ifdef __cplusplus
void toeplitz_ddmatrix(DDMatrix mat, dd_real gamma_param, long int dim)
#else // __cplusplus
void toeplitz_ddmatrix(DDMatrix mat, double gamma_param[DDSIZE], long int dim)
#endif // __cplusplus
{
	long int i, j;
#ifdef __cplusplus
	static dd_real tmp;
#else // __cplusplus
	double tmp[DDSIZE];
#endif // __cplusplus

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(toeplitz_ddmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(toeplitz_ddmatrix)\n");
		return;
	}

	for(i = 0; i < dim; i++)
	{
		rdd_set_ui(tmp, 0UL);
		for(j = 0; j < dim; j++)
			set_ddmatrix_ij(mat, i, j, tmp);

		if(i >= 2)
			set_ddmatrix_ij(mat, i, i - 2, gamma_param);

		rdd_set_ui(tmp, 1UL);
		if(i <= (dim - 2))
			set_ddmatrix_ij(mat, i, i + 1, tmp);

		rdd_set_ui(tmp, 2UL);
		set_ddmatrix_ij(mat, i, i, tmp);
	}
}


// n!
#ifdef __cplusplus
void ddfactorial(dd_real *ret, long int n)
{
	rdd_set_d(*ret, 1.0);

	if(n > 0)
	{
		do
		{
			rdd_mul_d(*ret, *ret, (double)n);
		} while(n-- > 1);
	}

	return;
}
#else
void ddfactorial(double ret[DDSIZE], long int n)
{
	rdd_set_d(ret, 1.0);

	if(n > 0)
	{
		do
		{
			rdd_mul_d(ret, ret, (double)n);
		} while(n-- > 1);
	}

	return;
}
#endif // __cplusplus

// 9. Pascal Matrix
void pascal_ddmatrix(DDMatrix ret, long int dim)
{
	long int i, j;
	static double element[DDSIZE], ifac[DDSIZE], jfac[DDSIZE], ipjfac[DDSIZE], tmp[DDSIZE];

	for(i = 0; i < ret->row_dim; i++)
	{
		for(j = 0; j < ret->col_dim; j++)
		{
			//element = dfactorial(i + j) / (dfactorial(i) * dfactorial(j));
			ddfactorial(ifac, i);
			ddfactorial(jfac, j);
			rdd_mul(tmp, ifac, jfac);
			ddfactorial(ipjfac, i + j);
			rdd_mul(element, tmp, ipjfac);
			set_ddmatrix_ij(ret, i, j, element);
		}
	}
}

// 10. I - randmatrix
void im_rand_ddmatrix(DDMatrix ret, unsigned long seed)
{
	long int i, j;
	double element[DDSIZE];

	set0_dd(element);

	// set seed
	srand(seed);

	// element := (-1)^rand() * rand() / RAND_MAX;
	for(i = 0; i < ret->row_dim; i++)
	{
		for(j = 0; j < ret->col_dim; j++)
		{
			//printf("%ld == %ld\n", i, j);
		#ifdef __cplusplus
			element = (double)rand();
			element /= (double)RAND_MAX;
			if((rand() % 2) != 0)
				element = -element;
		#else
			element[0] = (double)rand();
			element[1] = 0.0;
			rdd_div_d(element, element, (double)RAND_MAX);
			if((rand() % 2) != 0)
				rdd_neg(element, element);
		#endif
			set_ddmatrix_ij(ret, i, j, element);
		}

		//printf("%ld == %ld\n", i, i);
		// element[i][i] := 1 - element[i][i];
		#ifdef __cplusplus
			//element = 1.0 - get_ddmatrix_ij(ret, i, i);
			element += i+1;
		#else
			rdd_add_ui(element, get_ddmatrix_ij(ret, i, i), (unsigned long)(i + 1));
		#endif
		set_ddmatrix_ij(ret, i, i, element);
	}
}
