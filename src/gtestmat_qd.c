#include "qdlinear.h"

/* QD */

/* 1. Hilbert Matrix */
void hilbert_qdmatrix(QDMatrix a, long int dim)
{
	long int i, j;
#ifdef __cplusplus
	static qd_real tmp;
#else // __cplusplus
	double tmp[QDSIZE];
#endif // __cplusplus

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(hilbert_qdmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(hilbert_qdmatrix)\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			rqd_set_ui(tmp, (unsigned long)(i + j + 1));
			rqd_ui_div(tmp, 1UL, tmp);
			set_qdmatrix_ij(a, i, j, tmp);
		}
	}
}


/* 2. Lotkin Matrix */
void lotkin_qdmatrix(QDMatrix a, long int dim)
{
	long int i, j;
#ifdef __cplusplus
	static qd_real tmp;
#else // __cplusplus
	double tmp[QDSIZE];
#endif // __cplusplus

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(lotkin_qdmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(lotkin_qdmatrix)\n");
		return;
	}

	/* Lotkin Matrix */
	for(i = 0; i < a->col_dim; i++)
	{
		rqd_set_ui(tmp, 1UL);
		set_qdmatrix_ij(a, 0, i, tmp);
	}

	for(i = 1; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			rqd_set_ui(tmp, (unsigned long)(i + j + 1));
			rqd_ui_div(tmp, 1UL, tmp);
			set_qdmatrix_ij(a, i, j, tmp);
		}
	}
}

/* 3. Frank Matrix */
void frank_qdmatrix(QDMatrix a, long int dim)
{
	long int i, j;
#ifdef __cplusplus
	static qd_real tmp;
#else // __cplusplus
	double tmp[QDSIZE];
#endif // __cplusplus

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(frank_qdmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(frank_qdmatrix)\n");
		return;
	}

	/* Frank Matrix */
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			if(i < j)
			{
				rqd_set_ui(tmp, (unsigned long)(a->col_dim - j));
				set_qdmatrix_ij(a, i, j, tmp);
			}
			else
			{
				rqd_set_ui(tmp, (unsigned long)(a->col_dim - i));
				set_qdmatrix_ij(a, i, j, tmp);
			}
		}
	}
}

/* 4. Tridiagonal Matrix */
void tridiag_qdmatrix(QDMatrix a, QDVector low_subdiag, QDVector diag, QDVector up_subdiag, long int dim)
{
	long int i, j;
#ifdef __cplusplus
	static qd_real tmp;
#else // __cplusplus
	double tmp[QDSIZE];
#endif // __cplusplus

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(tridiag_qdmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(tridiag_qdmatrix)\n");
		return;
	}

	/* Tridiagonal Matrix */
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < i - 1; j++)
		{
			rqd_set_ui(tmp, 0UL);
			set_qdmatrix_ij(a, i, j, tmp);
		}
		for(j = i + 2; j < a->col_dim; j++)
		{
			rqd_set_ui(tmp, 0UL);
			set_qdmatrix_ij(a, i, j, tmp);
		}
	}

	set_qdmatrix_ij(a, 0, 0, get_qdvector_i(diag, 0));
	set_qdmatrix_ij(a, 0, 1, get_qdvector_i(up_subdiag , 0));
	for(i = 1; i < a->row_dim - 1; i++)
	{
		set_qdmatrix_ij(a, i, i - 1, get_qdvector_i(low_subdiag, i));
		set_qdmatrix_ij(a, i, i    , get_qdvector_i(diag, i));
		set_qdmatrix_ij(a, i, i + 1, get_qdvector_i(up_subdiag , i));
	}
	i = a->row_dim - 1;
	set_qdmatrix_ij(a, i, i - 1, get_qdvector_i(low_subdiag , i));
	set_qdmatrix_ij(a, i, i    , get_qdvector_i(diag, i));

}


/* 5. Integer Symmetrix Random Matrix */
void int_sym_rand_qdmatrix(QDMatrix mat, long int max, long int seed, long int dim)
{
	long int i, j;
#ifdef __cplusplus
	static qd_real tmp;
#else // __cplusplus
	double tmp[QDSIZE];
#endif // __cplusplus

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(int_sym_rand_qdmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(int_sym_rand_qdmatrix)\n");
		return;
	}

	srand((int)seed);
	for(i = 0; i < dim; i++)
		for(j = i; j < dim; j++)
		{
			rqd_set_ui(tmp, (unsigned long)(rand() % max));
			set_qdmatrix_ij(mat, i, j, tmp);
		}

	for(i = 0; i < dim; i++)
		for(j = 0; j < i; j++)
			set_qdmatrix_ij(mat, i, j, get_qdmatrix_ij(mat, j, i));

}

/* 6. Integer Unsymmetrix Random Matrix */
void int_unsym_rand_qdmatrix(QDMatrix mat, long int max, long int seed, long int dim)
{
	long int i, j;
#ifdef __cplusplus
	static qd_real tmp;
#else // __cplusplus
	double tmp[QDSIZE];
#endif // __cplusplus

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(int_unsym_rand_qdmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(int_unsym_rand_qdmatrix)\n");
		return;
	}

	srand((int)seed);
	for(i = 0; i < dim; i++)
		for(j = 0; j < dim; j++)
		{
			rqd_set_ui(tmp, (unsigned long)(rand() % max));
			set_qdmatrix_ij(mat, i, j, tmp);
		}
}

/* 7. Real Diagonal Matrix */
void diag_qdmatrix(QDMatrix mat, QDVector diag, long int dim)
{
	long int i;

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(diag_qdmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(diag_qdmatrix)\n");
		return;
	}

	set0_qdmatrix(mat);
	for(i = 0; i < dim; i++)
		set_qdmatrix_ij(mat, i, i, get_qdvector_i(diag, i));

}

/* 8. Toeplitz Matrix */
#ifdef __cplusplus
void toeplitz_qdmatrix(QDMatrix mat, qd_real gamma_param, long int dim)
#else // __cplusplus
void toeplitz_qdmatrix(QDMatrix mat, double gamma_param[QDSIZE], long int dim)
#endif // __cplusplus
{
	long int i, j;
#ifdef __cplusplus
	qd_real tmp;
#else
	double tmp[QDSIZE];
#endif // __cplusplus

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(toeplitz_qdmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(toeplitz_qdmatrix)\n");
		return;
	}

	for(i = 0; i < dim; i++)
	{
		rqd_set_ui(tmp, 0UL);
		for(j = 0; j < dim; j++)
			set_qdmatrix_ij(mat, i, j, tmp);

		if(i >= 2)
			set_qdmatrix_ij(mat, i, i - 2, gamma_param);

		rqd_set_ui(tmp, 1UL);
		if(i <= (dim - 2))
			set_qdmatrix_ij(mat, i, i + 1, tmp);

		rqd_set_ui(tmp, 2UL);
		set_qdmatrix_ij(mat, i, i, tmp);
	}
}

// n!
#ifdef __cplusplus
void qdfactorial(qd_real *ret, long int n)
{
	rqd_set_d(*ret, 1.0);

	if(n > 0)
	{
		do
		{
			rqd_mul_d(*ret, *ret, (double)n);
		} while(n-- > 1);
	}

	return;
}
#else
void qdfactorial(double ret[QDSIZE], long int n)
{
	rqd_set_d(ret, 1.0);

	if(n > 0)
	{
		do
		{
			rqd_mul_d(ret, ret, (double)n);
		} while(n-- > 1);
	}

	return;
}
#endif // __cplusplus

// 9. Pascal Matrix
void pascal_qdmatrix(QDMatrix ret, long int dim)
{
	long int i, j;
#ifdef __cplusplus
	static qd_real element, ifac, jfac, ipjfac, tmp;
#else // __cplusplus
	static double element[QDSIZE], ifac[QDSIZE], jfac[QDSIZE], ipjfac[QDSIZE], tmp[QDSIZE];
#endif // __cplusplus

	for(i = 0; i < ret->row_dim; i++)
	{
		for(j = 0; j < ret->col_dim; j++)
		{
			//element = dfactorial(i + j) / (dfactorial(i) * dfactorial(j));
#ifdef __cplusplus
			qdfactorial(&ifac, i);
			qdfactorial(&jfac, j);
			rqd_mul(tmp, ifac, jfac);
			qdfactorial(&ipjfac, i + j);
			rqd_mul(element, tmp, ipjfac);
			set_qdmatrix_ij(ret, i, j, element);
#else // __cplusplus
			qdfactorial(ifac, i);
			qdfactorial(jfac, j);
			rqd_mul(tmp, ifac, jfac);
			qdfactorial(ipjfac, i + j);
			rqd_mul(element, tmp, ipjfac);
			set_qdmatrix_ij(ret, i, j, element);
#endif // __cplusplus
		}
	}
}

// 10. I - randmatrix
void im_rand_qdmatrix(QDMatrix ret, unsigned long seed)
{
	long int i, j;
#ifdef __cplusplus
	qd_real element;
#else
	double element[QDSIZE];
#endif

	set0_qd(element);

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
			element[3] = 0.0;
			rqd_div_d(element, element, (double)RAND_MAX);
			if((rand() % 2) != 0)
				rqd_neg(element, element);
		#endif

			set_qdmatrix_ij(ret, i, j, element);
		}

		//printf("%ld == %ld\n", i, i);
		// element[i][i] := 1 - element[i][i];
		#ifdef __cplusplus
			//element = 1.0 - get_qdmatrix_ij(ret, i, i);
			element = (i + 1) + get_qdmatrix_ij(ret, i, i);
		#else
			//rqd_ui_div(element, 1UL, get_qdmatrix_ij(ret, i, i));
			rqd_add_ui(element, get_qdmatrix_ij(ret, i, i), (unsigned long)(i + 1));
		#endif
		set_qdmatrix_ij(ret, i, i, element);
	}
}

