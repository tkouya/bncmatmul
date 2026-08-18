#include "tdlinear.h"

/* TD */

/* 1. Hilbert Matrix */
void hilbert_tdmatrix(TDMatrix a, long int dim)
{
	long int i, j;
	double tmp[TDSIZE];

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(hilbert_tdmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(hilbert_tdmatrix)\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			rtd_set_ui(tmp, (unsigned long)(i + j + 1));
			rtd_ui_div(tmp, 1UL, tmp);
			set_tdmatrix_ij(a, i, j, tmp);
		}
	}
}


/* 2. Lotkin Matrix */
void lotkin_tdmatrix(TDMatrix a, long int dim)
{
	long int i, j;
	double tmp[TDSIZE];

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(lotkin_tdmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(lotkin_tdmatrix)\n");
		return;
	}

	/* Lotkin Matrix */
	for(i = 0; i < a->col_dim; i++)
	{
		rtd_set_ui(tmp, 1UL);
		set_tdmatrix_ij(a, 0, i, tmp);
	}

	for(i = 1; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			rtd_set_ui(tmp, (unsigned long)(i + j + 1));
			rtd_ui_div(tmp, 1UL, tmp);
			set_tdmatrix_ij(a, i, j, tmp);
		}
	}
}

/* 3. Frank Matrix */
void frank_tdmatrix(TDMatrix a, long int dim)
{
	long int i, j;
	double tmp[TDSIZE];

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(frank_tdmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(frank_tdmatrix)\n");
		return;
	}

	/* Frank Matrix */
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			if(i < j)
			{
				rtd_set_ui(tmp, (unsigned long)(a->col_dim - j));
				set_tdmatrix_ij(a, i, j, tmp);
			}
			else
			{
				rtd_set_ui(tmp, (unsigned long)(a->col_dim - i));
				set_tdmatrix_ij(a, i, j, tmp);
			}
		}
	}
}

/* 4. Tridiagonal Matrix */
void tridiag_tdmatrix(TDMatrix a, TDVector low_subdiag, TDVector diag, TDVector up_subdiag, long int dim)
{
	long int i, j;
	double tmp[TDSIZE];

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(tridiag_tdmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(tridiag_tdmatrix)\n");
		return;
	}

	/* Tridiagonal Matrix */
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < i - 1; j++)
		{
			rtd_set_ui(tmp, 0UL);
			set_tdmatrix_ij(a, i, j, tmp);
		}
		for(j = i + 2; j < a->col_dim; j++)
		{
			rtd_set_ui(tmp, 0UL);
			set_tdmatrix_ij(a, i, j, tmp);
		}
	}

	set_tdmatrix_ij(a, 0, 0, get_tdvector_i(diag, 0));
	set_tdmatrix_ij(a, 0, 1, get_tdvector_i(up_subdiag , 0));
	for(i = 1; i < a->row_dim - 1; i++)
	{
		set_tdmatrix_ij(a, i, i - 1, get_tdvector_i(low_subdiag, i));
		set_tdmatrix_ij(a, i, i    , get_tdvector_i(diag, i));
		set_tdmatrix_ij(a, i, i + 1, get_tdvector_i(up_subdiag , i));
	}
	i = a->row_dim - 1;
	set_tdmatrix_ij(a, i, i - 1, get_tdvector_i(low_subdiag , i));
	set_tdmatrix_ij(a, i, i    , get_tdvector_i(diag, i));

}


/* 5. Integer Symmetrix Random Matrix */
void int_sym_rand_tdmatrix(TDMatrix mat, long int max, long int seed, long int dim)
{
	long int i, j;
	double tmp[TDSIZE];

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(int_sym_rand_tdmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(int_sym_rand_tdmatrix)\n");
		return;
	}

	srand((int)seed);
	for(i = 0; i < dim; i++)
		for(j = i; j < dim; j++)
		{
			rtd_set_ui(tmp, (unsigned long)(rand() % max));
			set_tdmatrix_ij(mat, i, j, tmp);
		}

	for(i = 0; i < dim; i++)
		for(j = 0; j < i; j++)
			set_tdmatrix_ij(mat, i, j, get_tdmatrix_ij(mat, j, i));

}

/* 6. Integer Unsymmetrix Random Matrix */
void int_unsym_rand_tdmatrix(TDMatrix mat, long int max, long int seed, long int dim)
{
	long int i, j;
	double tmp[TDSIZE];

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(int_unsym_rand_tdmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(int_unsym_rand_tdmatrix)\n");
		return;
	}

	srand((int)seed);
	for(i = 0; i < dim; i++)
		for(j = 0; j < dim; j++)
		{
			rtd_set_ui(tmp, (unsigned long)(rand() % max));
			set_tdmatrix_ij(mat, i, j, tmp);
		}
}

/* 7. Real Diagonal Matrix */
void diag_tdmatrix(TDMatrix mat, TDVector diag, long int dim)
{
	long int i;

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(diag_tdmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(diag_tdmatrix)\n");
		return;
	}

	set0_tdmatrix(mat);
	for(i = 0; i < dim; i++)
		set_tdmatrix_ij(mat, i, i, get_tdvector_i(diag, i));

}

/* 8. Toeplitz Matrix */
void toeplitz_tdmatrix(TDMatrix mat, double gamma_param[TDSIZE], long int dim)
{
	long int i, j;
	double tmp[TDSIZE];

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(toeplitz_tdmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(toeplitz_tdmatrix)\n");
		return;
	}

	for(i = 0; i < dim; i++)
	{
		rtd_set_ui(tmp, 0UL);
		for(j = 0; j < dim; j++)
			set_tdmatrix_ij(mat, i, j, tmp);

		if(i >= 2)
			set_tdmatrix_ij(mat, i, i - 2, gamma_param);

		rtd_set_ui(tmp, 1UL);
		if(i <= (dim - 2))
			set_tdmatrix_ij(mat, i, i + 1, tmp);

		rtd_set_ui(tmp, 2UL);
		set_tdmatrix_ij(mat, i, i, tmp);
	}
}

// n!
#ifdef __cplusplus
void tdfactorial(td_real *ret, long int n)
{
	rtd_set_d(*ret, 1.0);

	if(n > 0)
	{
		do
		{
			rtd_mul_d(*ret, *ret, (double)n);
		} while(n-- > 1);
	}

	return;
}
#else
void tdfactorial(double ret[TDSIZE], long int n)
{
	rtd_set_d(ret, 1.0);

	if(n > 0)
	{
		do
		{
			rtd_mul_d(ret, ret, (double)n);
		} while(n-- > 1);
	}

	return;
}
#endif // __cplusplus

// 9.Pascal Matrix
void pascal_tdmatrix(TDMatrix ret, long int dim)
{
	long int i, j;
#ifdef __cplusplus
	static td_real element, ifac, jfac, ipjfac, tmp;
#else // __cplusplus
	static double element[TDSIZE], ifac[TDSIZE], jfac[TDSIZE], ipjfac[TDSIZE], tmp[TDSIZE];
#endif // __cplusplus

	for(i = 0; i < ret->row_dim; i++)
	{
		for(j = 0; j < ret->col_dim; j++)
		{
			//element = dfactorial(i + j) / (dfactorial(i) * dfactorial(j));
#ifdef __cplusplus
			tdfactorial(&ifac, i);
			tdfactorial(&jfac, j);
			rtd_mul(tmp, ifac, jfac);
			tdfactorial(&ipjfac, i + j);
			rtd_mul(element, tmp, ipjfac);
			set_tdmatrix_ij(ret, i, j, element);
#else // __cplusplus
			tdfactorial(ifac, i);
			tdfactorial(jfac, j);
			rtd_mul(tmp, ifac, jfac);
			tdfactorial(ipjfac, i + j);
			rtd_mul(element, tmp, ipjfac);
			set_tdmatrix_ij(ret, i, j, element);
#endif // __cplusplus
		}
	}
}

// 10. I - randmatrix
void im_rand_tdmatrix(TDMatrix ret, unsigned long seed)
{
	long int i, j;
#ifdef __cplusplus
	td_real element;
#else
	double element[TDSIZE];
#endif // __cplusplus

	set0_td(element);

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
			element[2] = 0.0;
			rtd_div_d(element, element, (double)RAND_MAX);
			if((rand() % 2) != 0)
				rtd_neg(element, element);
		#endif

			set_tdmatrix_ij(ret, i, j, element);
		}

		//printf("%ld == %ld\n", i, i);
		// element[i][i] := 1 - element[i][i];
		#ifdef __cplusplus
			//element = 1.0 - get_tdmatrix_ij(ret, i, i);
			element = get_mpfmatrix_ij(ret, i, i) + (i + 1);
		#else
			//rtd_ui_div(element, 1UL, get_tdmatrix_ij(ret, i, i));
			rtd_add_ui(element, get_tdmatrix_ij(ret, i, i), (unsigned long)(i + 1));
		#endif
		set_tdmatrix_ij(ret, i, i, element);
	}
}
