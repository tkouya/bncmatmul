/********************************************************************************/
/* gtestmat.c:                                                                  */
/* Copyright (C) 2003-2022 Tomonori Kouya                                       */
/*                                                                              */
/* Version 0.1: 2005.06/23 append arg "dim"                                     */
/*                                                                              */
/* This program is free software: you can redistribute it and/or modify it      */
/* under the terms of the GNU Lesser General Public License as published by the */
/* Free Software Foundation, either version 3 of the License or any later       */
/* version.                                                                     */
/*                                                                              */
/* This program is distributed in the hope that it will be useful, but WITHOUT  */
/* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        */
/* FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License */
/* for more details.                                                            */
/*                                                                              */
/* You should have received a copy of the GNU Lesser General Public License     */
/* along with this program.  If not, see <http://www.gnu.org/licenses/>.        */
/*                                                                              */
/********************************************************************************/
/************************************************/
/* Get Test Matrices                            */
/*                                              */
/* 1. Hilbert Matrix                            */
/* 2. Lotkin Matrix                             */
/* 3. Frank Matrix                              */
/* 4. Tridiagonal Matrix                        */
/* 5. Integer Symmetrix Random Matrix           */
/* 6. Integer Unsymmetrix Random Matrix         */
/* 7. Diagonal Matrix                           */
/* 8. Toeplitz Matrix                           */
/************************************************/

#include <stdio.h>
#include <stdlib.h>

//#include "gtestmat.h"
#include "dlinear.h"
#include "mpflinear.h"

/* Double */

/* 1. Hilbert Matrix */
void hilbert_dmatrix(DMatrix a, long int dim)
{
	long int i, j;
	double tmp;

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(hilbert_dmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(hilbert_dmatrix)\n");
		return;
	}

	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
			set_dmatrix_ij(a, i, j, 1.0 / (i + j + 1));
	}
}


/* 2. Lotkin Matrix */
void lotkin_dmatrix(DMatrix a, long int dim)
{
	long int i, j;
	double tmp;

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(lotkin_dmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(lotkin_dmatrix)\n");
		return;
	}
	/* Lotkin Matrix */
	for(i = 0; i < a->col_dim; i++)
		set_dmatrix_ij(a, 0, i, 1.0);

	for(i = 1; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
			set_dmatrix_ij(a, i, j, 1.0 / (i + j + 1));
	}
}

/* 3. Frank Matrix */
void frank_dmatrix(DMatrix a, long int dim)
{
	long int i, j;
	double tmp;

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(frank_dmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(frank_dmatrix)\n");
		return;
	}

	/* Frank Matrix */
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			if(i < j)
				set_dmatrix_ij(a, i, j, (double)(dim - j));
			else
				set_dmatrix_ij(a, i, j, (double)(dim - i));
		}
	}
}

/* 4. Tridiagonal Matrix */
void tridiag_dmatrix(DMatrix a, DVector low_subdiag, DVector diag, DVector up_subdiag, long int dim)
{
	long int i, j;
	double tmp;

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(tridiag_dmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(tridiag_dmatrix)\n");
		return;
	}

	/* Tridiagonal Matrix */
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < i - 1; j++)
			set_dmatrix_ij(a, i, j, 0.0);
		for(j = i + 2; j < dim; j++)
			set_dmatrix_ij(a, i, j, 0.0);
	}

	set_dmatrix_ij(a, 0, 0, get_dvector_i(diag, 0));
	set_dmatrix_ij(a, 0, 1, get_dvector_i(up_subdiag , 0));
	for(i = 1; i < dim - 1; i++)
	{
		set_dmatrix_ij(a, i, i - 1, get_dvector_i(low_subdiag, i));
		set_dmatrix_ij(a, i, i    , get_dvector_i(diag, i));
		set_dmatrix_ij(a, i, i + 1, get_dvector_i(up_subdiag , i));
	}
	i = dim - 1;
	set_dmatrix_ij(a, i, i - 1, get_dvector_i(low_subdiag , i));
	set_dmatrix_ij(a, i, i    , get_dvector_i(diag, i));
}


/* 5. Integer Symmetrix Random Matrix */
void int_sym_rand_dmatrix(DMatrix mat, long int max, long int seed, long int dim)
{
	long int i, j;

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(int_ sym_rand_dmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(int_sym_rand_dmatrix)\n");
		return;
	}

	srand((int)seed);
	for(i = 0; i < dim; i++)
		for(j = i; j < dim; j++)
			set_dmatrix_ij(mat, i, j, (double)(rand() % max));

	for(i = 0; i < dim; i++)
		for(j = 0; j < i; j++)
			set_dmatrix_ij(mat, i, j, get_dmatrix_ij(mat, j, i));
}

/* 6. Integer Unsymmetrix Random Matrix */
void int_unsym_rand_dmatrix(DMatrix mat, long int max, long int seed, long int dim)
{
	long int i, j;

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(int_unsym_rand_dmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(int_unsym_rand_dmatrix)\n");
		return;
	}

	srand((int)seed);
	for(i = 0; i < dim; i++)
		for(j = 0; j < dim; j++)
			set_dmatrix_ij(mat, i, j, (double)(rand() % max));
}

/* 7. Real Diagonal Matrix */
void diag_dmatrix(DMatrix mat, DVector diag, long int dim)
{
	long int i, j;

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(diag_dmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(diag_dmatrix)\n");
		return;
	}

	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
			set_dmatrix_ij(mat, i, j, 0.0);
		set_dmatrix_ij(mat, i, i, get_dvector_i(diag, i));
	}

}

/* 8. Toeplitz Matrix */
void toeplitz_dmatrix(DMatrix mat, double gamma_param, long int dim)
{
	long int i, j;

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(toeplitz_dmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(toeplitz_dmatrix)\n");
		return;
	}

	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
			set_dmatrix_ij(mat, i, j, 0.0);

		if(i >= 2)
			set_dmatrix_ij(mat, i, i - 2, gamma_param);

		if(i <= (dim - 2))
			set_dmatrix_ij(mat, i, i + 1, 1.0);

		set_dmatrix_ij(mat, i, i, 2.0);
	}
}

/* MPF */
#ifdef USE_GMP

/* 1. Hilbert Matrix */
void hilbert_mpfmatrix(MPFMatrix a, long int dim)
{
	long int i, j;
	mpf_t tmp;

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(hilbert_mpfmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(hilbert_mpfmatrix)\n");
		return;
	}

	mpf_init2(tmp, a->prec);

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			mpf_set_ui(tmp, (unsigned long)(i + j + 1));
			mpf_ui_div(tmp, 1UL, tmp);
			set_mpfmatrix_ij(a, i, j, tmp);
		}
	}

	mpf_clear(tmp);
}


/* 2. Lotkin Matrix */
void lotkin_mpfmatrix(MPFMatrix a, long int dim)
{
	long int i, j;
	mpf_t tmp;

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(lotkin_mpfmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(lotkin_mpfmatrix)\n");
		return;
	}

	mpf_init2(tmp, a->prec);

	/* Lotkin Matrix */
	for(i = 0; i < a->col_dim; i++)
	{
		mpf_set_ui(tmp, 1UL);
		set_mpfmatrix_ij(a, 0, i, tmp);
	}

	for(i = 1; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			mpf_set_ui(tmp, (unsigned long)(i + j + 1));
			mpf_ui_div(tmp, 1UL, tmp);
			set_mpfmatrix_ij(a, i, j, tmp);
		}
	}

	mpf_clear(tmp);
}

/* 3. Frank Matrix */
void frank_mpfmatrix(MPFMatrix a, long int dim)
{
	long int i, j;
	mpf_t tmp;

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(frank_mpfmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(frank_mpfmatrix)\n");
		return;
	}

	mpf_init2(tmp, a->prec);

	/* Frank Matrix */
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			if(i < j)
			{
				mpf_set_ui(tmp, (unsigned long)(a->col_dim - j));
				set_mpfmatrix_ij(a, i, j, tmp);
			}
			else
			{
				mpf_set_ui(tmp, (unsigned long)(a->col_dim - i));
				set_mpfmatrix_ij(a, i, j, tmp);
			}
		}
	}

	mpf_clear(tmp);
}

/* 4. Tridiagonal Matrix */
void tridiag_mpfmatrix(MPFMatrix a, MPFVector low_subdiag, MPFVector diag, MPFVector up_subdiag, long int dim)
{
	long int i, j;
	mpf_t tmp;

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(tridiag_mpfmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(tridiag_mpfmatrix)\n");
		return;
	}

	mpf_init2(tmp, a->prec);

	/* Tridiagonal Matrix */
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < i - 1; j++)
		{
			mpf_set_ui(tmp, 0UL);
			set_mpfmatrix_ij(a, i, j, tmp);
		}
		for(j = i + 2; j < a->col_dim; j++)
		{
			mpf_set_ui(tmp, 0UL);
			set_mpfmatrix_ij(a, i, j, tmp);
		}
	}

	set_mpfmatrix_ij(a, 0, 0, get_mpfvector_i(diag, 0));
	set_mpfmatrix_ij(a, 0, 1, get_mpfvector_i(up_subdiag , 0));
	for(i = 1; i < a->row_dim - 1; i++)
	{
		set_mpfmatrix_ij(a, i, i - 1, get_mpfvector_i(low_subdiag, i));
		set_mpfmatrix_ij(a, i, i    , get_mpfvector_i(diag, i));
		set_mpfmatrix_ij(a, i, i + 1, get_mpfvector_i(up_subdiag , i));
	}
	i = a->row_dim - 1;
	set_mpfmatrix_ij(a, i, i - 1, get_mpfvector_i(low_subdiag , i));
	set_mpfmatrix_ij(a, i, i    , get_mpfvector_i(diag, i));

	mpf_clear(tmp);

}


/* 5. Integer Symmetrix Random Matrix */
void int_sym_rand_mpfmatrix(MPFMatrix mat, long int max, long int seed, long int dim)
{
	long int i, j;
	mpf_t tmp;

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(int_sym_rand_mpfmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(int_sym_rand_mpfmatrix)\n");
		return;
	}

	mpf_init2(tmp, mat->prec);

	srand((int)seed);
	for(i = 0; i < dim; i++)
		for(j = i; j < dim; j++)
		{
			mpf_set_ui(tmp, (unsigned long)(rand() % max));
			set_mpfmatrix_ij(mat, i, j, tmp);
		}

	for(i = 0; i < dim; i++)
		for(j = 0; j < i; j++)
			set_mpfmatrix_ij(mat, i, j, get_mpfmatrix_ij(mat, j, i));

	mpf_clear(tmp);

}

/* 6. Integer Unsymmetrix Random Matrix */
void int_unsym_rand_mpfmatrix(MPFMatrix mat, long int max, long int seed, long int dim)
{
	long int i, j;
	mpf_t tmp;

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(int_unsym_rand_mpfmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(int_unsym_rand_mpfmatrix)\n");
		return;
	}

	mpf_init2(tmp, mat->prec);

	srand((int)seed);
	for(i = 0; i < dim; i++)
		for(j = 0; j < dim; j++)
		{
			mpf_set_ui(tmp, (unsigned long)(rand() % max));
			set_mpfmatrix_ij(mat, i, j, tmp);
		}

	mpf_clear(tmp);

}

/* 7. Real Diagonal Matrix */
void diag_mpfmatrix(MPFMatrix mat, MPFVector diag, long int dim)
{
	long int i;

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(diag_mpfmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(diag_mpfmatrix)\n");
		return;
	}

	set0_mpfmatrix(mat);
	for(i = 0; i < dim; i++)
		set_mpfmatrix_ij(mat, i, i, get_mpfvector_i(diag, i));

}

/* 8. Toeplitz Matrix */
void toeplitz_mpfmatrix(MPFMatrix mat, mpf_t gamma_param, long int dim)
{
	long int i, j;
	mpf_t tmp;

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(toeplitz_mpfmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(toeplitz_mpfmatrix)\n");
		return;
	}

	mpf_init2(tmp, mat->prec);

	for(i = 0; i < dim; i++)
	{
		mpf_set_ui(tmp, 0UL);
		for(j = 0; j < dim; j++)
			set_mpfmatrix_ij(mat, i, j, tmp);

		if(i >= 2)
			set_mpfmatrix_ij(mat, i, i - 2, gamma_param);

		mpf_set_ui(tmp, 1UL);
		if(i <= (dim - 2))
			set_mpfmatrix_ij(mat, i, i + 1, tmp);

		mpf_set_ui(tmp, 2UL);
		set_mpfmatrix_ij(mat, i, i, tmp);
	}

	mpf_clear(tmp);
}
// n!
void mpf_factorial(mpf_t ret, long int n)
{
	mpf_set_ui(ret, 1UL);

	if(n > 0)
	{
		do
		{
			mpf_mul_ui(ret, ret, n);;
		} while(n-- > 1);
	}

	return;
}

// 9. Pascal Matrix
void pascal_mpfmatrix(MPFMatrix ret, long int dim)
{
	long int i, j;
	mpf_t element, tmp;

	mpf_init2(element, ret->prec);
	mpf_init2(tmp, ret->prec);

	for(i = 0; i < ret->row_dim; i++)
	{
		for(j = 0; j < ret->col_dim; j++)
		{
			//element = dfactorial(i + j) / (dfactorial(i) * dfactorial(j));
			mpf_factorial(element, i + j);
			mpf_factorial(tmp, i);
			mpf_mul(element, element, tmp);
			mpf_factorial(tmp, j);
			mpf_mul(element, element, tmp);

			set_mpfmatrix_ij(ret, i, j, element);
		}
	}

	mpf_clear(element);
	mpf_clear(tmp);
}

// 10. I - randmatrix
void im_rand_mpfmatrix(MPFMatrix ret, unsigned long seed)
{
	long int i, j;
	mpf_t element;

	mpf_init2(element, ret->prec);

	// set seed
	srand(seed);

	// element := (-1)^rand() * rand() / RAND_MAX;
	for(i = 0; i < ret->row_dim; i++)
	{
		for(j = 0; j < ret->col_dim; j++)
		{
			//printf("%ld == %ld\n", i, j);
			mpf_set_d(element, (double)(rand()));
			mpf_div_ui(element, element, (unsigned long)RAND_MAX);
			if((rand() % 2) != 0)
				mpf_neg(element, element);

			set_mpfmatrix_ij(ret, i, j, element);
		}

		//printf("%ld == %ld\n", i, i);
		// element[i][i] := 1 - element[i][i];
		mpf_ui_sub(element, 1UL, get_mpfmatrix_ij(ret, i, i));
		set_mpfmatrix_ij(ret, i, i, element);
	}

	mpf_clear(element);
}
#endif // USE_GMP
