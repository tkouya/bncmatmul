#include "mpflinear.h"

#ifdef USE_GMP
/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                  (Multi-Precision)       */
/*                                                          */
/*                 ver. 0.0 2000.02.28 (Mon) Tomonori Kouya */
/*                 ver. 0.1 2000.07.05 (Wed) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int MPFLUdecomp(MPFMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       MPFMatrix a: Matrix (given by user)                */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	mpf_t tmp, axii;

	dim = a->col_dim;
	mpf_init2(tmp, a->prec);
	mpf_init2(axii, a->prec);
	for(i = 0; i < dim; i++)
	{
		mpf_abs(axii, get_mpfmatrix_ij(a, i, i));
		if(mpf_cmp_ui(axii, 0UL) == 0)
		{
			mpf_clear(tmp);
			mpf_clear(axii);
			fprintf(stderr, "%ld : Error! (MPFLUdecomp)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			mpf_div(tmp, get_mpfmatrix_ij(a, j, i), get_mpfmatrix_ij(a, i, i));
			set_mpfmatrix_ij(a, j, i, tmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				mpf_mul(tmp, get_mpfmatrix_ij(a, j, i), get_mpfmatrix_ij(a, i, k));
				mpf_sub(tmp, get_mpfmatrix_ij(a, j, k), tmp);
				set_mpfmatrix_ij(a, j, k, tmp);
			}
		}
	}

	mpf_clear(tmp);
	mpf_clear(axii);
	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                                  (Muiti-Precision)       */
/*                                                          */
/*                 ver. 0.0 2000.02.28 (Mon) Tomonori Kouya */
/*                 ver. 0.1 2000.07.05 (Wed) Tomonori Kouya */
/*                 ver. 0.1 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveMPFLS(MPFVector answer, MPFMatrix lu, MPFVector b)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       MPFMatrix lu: LU decomposed Matrix (given by user) */
/*       MPFVector b: constant vector (given by user)       */
/*       MPFVector answer: Solution for linear system       */
/*       long int dim: Dimension of Matrix (given by user)  */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	mpf_t tmp;

	dim = answer->dim;

	subst_mpfvector(answer, b);
	mpf_init2(tmp, answer->prec);

/* Forward */
	for(i = 0; i < dim; i++)
	{
		mpf_abs(tmp, get_mpfmatrix_ij(lu, i, i));
		if(mpf_cmp_ui(tmp, 0UL) == 0)
		{
			mpf_clear(tmp);
			fprintf(stderr, "Unable to solve the linear system!(SolveMPFLS, %ld)\n", i);
			return -1;
		}

		mpf_set(tmp, get_mpfvector_i(answer, i));
		for(j = (i + 1); j < dim; j++)
		{
			mpf_mul(tmp, get_mpfmatrix_ij(lu, j, i), get_mpfvector_i(answer, i));
			mpf_sub(tmp, get_mpfvector_i(answer, j), tmp);
			set_mpfvector_i(answer, j, tmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			mpf_mul(tmp, get_mpfmatrix_ij(lu, i, j), get_mpfvector_i(answer, j));
			mpf_sub(tmp, get_mpfvector_i(answer, i),  tmp);
			set_mpfvector_i(answer, i, tmp);
		}
		mpf_div(tmp, get_mpfvector_i(answer, i), get_mpfmatrix_ij(lu, i, i));
		set_mpfvector_i(answer, i, tmp);
	}

	mpf_clear(tmp);
	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                 (Multi-Precision)        */
/*                                 (Partial Pivoting)       */
/*                                                          */
/*                 ver. 0.0 2000.02.28 (Thu) Tomonori Kouya */
/*                 ver. 0.1 2000.07.05 (Wed) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int MPFLUdecompP(MPFMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       MPFMatrix a: Matrix (given by user)                */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
	mpf_t tmp, axii;

	dim = a->col_dim;

	mpf_init2(tmp, a->prec);
	mpf_init2(axii, a->prec);
	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		mpf_abs(axii, get_mpfmatrix_ij(a, ch[i], i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			mpf_abs(tmp, get_mpfmatrix_ij(a, ch[j], i));
			if(mpf_cmp(tmp, axii) > 0)
			{
				imax = j;
				mpf_set(axii, tmp);
			}
		}

		if(mpf_cmp_ui(axii, 0UL) == 0)
		{
			mpf_clear(tmp);
			mpf_clear(axii);
			fprintf(stderr, "%ld : Error! MPFLUdecompP!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;
		}

		for(j = (i + 1); j < dim; j++)
		{
			mpf_div(tmp, get_mpfmatrix_ij(a, ch[j], i), get_mpfmatrix_ij(a, ch[i], i));
			set_mpfmatrix_ij(a, ch[j], i, tmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				mpf_mul(tmp, get_mpfmatrix_ij(a, ch[j], i), get_mpfmatrix_ij(a, ch[i], k));
				mpf_sub(tmp, get_mpfmatrix_ij(a, ch[j], k), tmp);
				set_mpfmatrix_ij(a, ch[j], k, tmp);
			}
		}
	}

	mpf_clear(tmp);
	mpf_clear(axii);
	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                                 (Multi-Precision)        */
/*                                 (Partial Pivoting)       */
/*                                                          */
/*                 ver. 0.0 2000.02.28 (Mon) Tomonori Kouya */
/*                 ver. 0.1 2000.07.05 (Wed) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                 ver. 0.3 2012-07-18 (Web) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveMPFLSP(MPFVector answer, MPFMatrix lu, MPFVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       MPFMatrix lu[]: LU decomposed Matrix(given by user)*/
/*       MPFVector b[]: constant vector (given by user)     */
/*       MPFVector answer[]: Solution for linear system     */
/*       long int ch: Row order (given by user)             */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	mpf_t tmp;

	mpf_init2(tmp, answer->prec);
	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_mpfvector_i(answer, i, get_mpfvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		if(mpf_cmp_ui(get_mpfmatrix_ij(lu, ch[i], i), 0UL) == 0)
		{
			mpf_clear(tmp);
			fprintf(stderr, "Unable to solve the linear system!(SolveMPFLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			mpf_mul(tmp, get_mpfmatrix_ij(lu, ch[j], i), get_mpfvector_i(answer, i));
			mpf_sub(tmp, get_mpfvector_i(answer, j), tmp);
			set_mpfvector_i(answer, j, tmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			
			mpf_mul(tmp, get_mpfmatrix_ij(lu, ch[i], j), get_mpfvector_i(answer, j));
			mpf_sub(tmp, get_mpfvector_i(answer, i), tmp);
			set_mpfvector_i(answer, i, tmp);
		}
		mpf_div(tmp, get_mpfvector_i(answer, i), get_mpfmatrix_ij(lu, ch[i], i));
		set_mpfvector_i(answer, i, tmp);
	}

	mpf_clear(tmp); // Fix! 2012-07-18 by T.Kouya

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix        */
/*                                 (Multi-Precision)        */
/*                                 (Complete Pivoting)      */
/*                                                          */
/*                 ver. 0.0 2000.02.28 (Mon) Tomonori Kouya */
/*                 ver. 0.1 2000.07.05 (Wed) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int MPFLUdecompC(MPFMatrix a, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       MPFMatrix a[]: Matrix (given by user)              */
/*       long int row_ch[]: Row order                       */
/*       long int col_ch[]: Column order                    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*  row_ch[]: Row order                                     */
/*  col_ch[]: Column order                                  */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	mpf_t tmp, axii;

	mpf_init2(tmp, a->prec);
	mpf_init2(axii, a->prec);
	dim = a->col_dim;

	for(i = 0; i < dim; i++)
	{
		row_ch[i] = i;
		col_ch[i] = i;
	}

	for(i = 0; i < dim; i++)
	{
		/* Complete Pivoting */
		mpf_abs(axii, get_mpfmatrix_ij(a, row_ch[i], col_ch[i]));
		imax = i;
		jmax = i;
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				mpf_abs(tmp, get_mpfmatrix_ij(a, row_ch[j], col_ch[k]));
				if(mpf_cmp(tmp, axii) > 0)
				{
					imax = j;
					jmax = k;
					mpf_set(axii, tmp);
				}
			}
		}

		if(mpf_cmp_ui(axii, 0UL) == 0)
		{
			mpf_clear(tmp);
			mpf_clear(axii);
			fprintf(stderr, "%ld : Error! (MPFLUdecompC)!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = row_ch[imax];
			row_ch[imax] = row_ch[i];
			row_ch[i] = itmp;
		}
		if(jmax != i)
		{
			itmp = col_ch[jmax];
			col_ch[jmax] = col_ch[i];
			col_ch[i] = itmp;
		}

		for(j = (i + 1); j < dim; j++)
		{
			mpf_div(tmp, get_mpfmatrix_ij(a, row_ch[j], col_ch[i]),  get_mpfmatrix_ij(a, row_ch[i], col_ch[i]));
			set_mpfmatrix_ij(a, row_ch[j], col_ch[i], tmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				mpf_mul(tmp, get_mpfmatrix_ij(a, row_ch[j], col_ch[i]), get_mpfmatrix_ij(a, row_ch[i], col_ch[k]));
				mpf_sub(tmp, get_mpfmatrix_ij(a, row_ch[j], col_ch[k]), tmp);
				set_mpfmatrix_ij(a, row_ch[j], col_ch[k], tmp);
			}
		}
	}

	mpf_clear(tmp);
	mpf_clear(axii);
	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                                 (Multi-Precision)        */
/*                                 (Complete Pivoting)      */
/*                                                          */
/*                 ver. 0.0 2000.02.28 (Mon) Tomonori Kouya */
/*                 ver. 0.1 2000.07.05 (Wed) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveMPFLSC(MPFVector answer, MPFMatrix lu, MPFVector b, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       MPFMatrix lu: LU decomposed Matrix (given by user) */
/*       MPFVector b: constant vector (given by user)       */
/*       MPFVector answer: Solution for linear system       */
/*       long int row_ch[]: Row order (given by user)       */
/*       long int col_ch[]: Column order (given by user)    */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	mpf_t tmp;

	mpf_init2(tmp, answer->prec);
	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_mpfvector_i(answer, col_ch[i], get_mpfvector_i(b, row_ch[i]));

/* Forward */
	for(i = 0; i < dim; i++)
	{
		if(mpf_cmp_ui(get_mpfmatrix_ij(lu, row_ch[i], col_ch[i]), 0UL) == 0)
		{
			mpf_clear(tmp);
			fprintf(stderr, "Unable to solve the linear system!(SolveMPFLSC, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			
			mpf_mul(tmp, get_mpfmatrix_ij(lu, row_ch[j], col_ch[i]), get_mpfvector_i(answer, col_ch[i]));
			mpf_sub(tmp, get_mpfvector_i(answer, col_ch[j]), tmp);
			set_mpfvector_i(answer, col_ch[j], tmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			mpf_mul(tmp, get_mpfmatrix_ij(lu, row_ch[i], col_ch[j]), get_mpfvector_i(answer, col_ch[j]));
			mpf_sub(tmp, get_mpfvector_i(answer, col_ch[i]), tmp);
			set_mpfvector_i(answer, col_ch[i], tmp);
		}
		mpf_div(tmp, get_mpfvector_i(answer, col_ch[i]), get_mpfmatrix_ij(lu, row_ch[i], col_ch[i]));
		set_mpfvector_i(answer, col_ch[i], tmp);
	}

	mpf_clear(tmp);
	return 0;
}
#endif // USE_GMP
