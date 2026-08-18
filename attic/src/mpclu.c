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
int MPCLUdecomp(CMPFMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CMPFMatrix a: Matrix (given by user)                */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	mpf_t abs_axii;
	mpc_t tmp;

	dim = a->col_dim;
	mpc_init2(tmp, a->prec);
	mpf_init2(axii, a->prec);
	for(i = 0; i < dim; i++)
	{
		mpc_abs(axii, get_cmpfmatrix_ij(a, i, i));
		if(mpf_cmp_ui(axii, 0UL) == 0)
		{
			mpc_clear(tmp);
			mpf_clear(axii);
			fprintf(stderr, "%ld : Error! (MPCLUdecomp)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			mpc_div(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, i));
			set_cmpfmatrix_ij(a, j, i, tmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				mpc_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				mpc_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
				set_cmpfmatrix_ij(a, j, k, tmp);
			}
		}
	}

	mpc_clear(tmp);
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
int SolveMPCLS(CMPFVector answer, CMPFMatrix lu, CMPFVector b)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CMPFMatrix lu: LU decomposed Matrix (given by user) */
/*       CMPFVector b: constant vector (given by user)       */
/*       CMPFVector answer: Solution for linear system       */
/*       long int dim: Dimension of Matrix (given by user)  */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	mpf_t abs_luii;
	mpc_t tmp;

	dim = answer->dim;

	subst_cmpfvector(answer, b);
	mpf_init2(abs_luii, answer->prec);
	mpc_init2(tmp, answer->prec);

/* Forward */
	for(i = 0; i < dim; i++)
	{
		mpc_abs(abs_luii, get_cmpfmatrix_ij(lu, i, i));
		if(mpf_cmp_ui(luii, 0UL) == 0)
		{
			mpf_clear(abs_luii);
			mpc_clear(tmp);
			fprintf(stderr, "Unable to solve the linear system!(SolveMPCLS, %ld)\n", i);
			return -1;
		}

		mpc_set(tmp, get_cmpfvector_i(answer, i));
		for(j = (i + 1); j < dim; j++)
		{
			mpc_mul(tmp, get_cmpfmatrix_ij(lu, j, i), get_cmpfvector_i(answer, i));
			mpc_sub(tmp, get_cmpfvector_i(answer, j), tmp);
			set_cmpfvector_i(answer, j, tmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			mpc_mul(tmp, get_cmpfmatrix_ij(lu, i, j), get_cmpfvector_i(answer, j));
			mpc_sub(tmp, get_cmpfvector_i(answer, i),  tmp);
			set_cmpfvector_i(answer, i, tmp);
		}
		mpc_div(tmp, get_cmpfvector_i(answer, i), get_cmpfmatrix_ij(lu, i, i));
		set_cmpfvector_i(answer, i, tmp);
	}

	mpf_clear(abs_luii);
	mpc_clear(tmp);
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
int MPCLUdecompP(CMPFMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CMPFMatrix a: Matrix (given by user)                */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
	mpc_t tmp;
	mpf_t abs_axii, abs_tmp;

	dim = a->col_dim;

	mpc_init2(tmp, a->prec);
	mpf_init2(abs_axii, a->prec);
	mpf_init2(abs_tmp, a->prec);

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		mpc_abs(abs_axii, get_cmpfmatrix_ij(a, ch[i], i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			mpc_abs(abs_tmp, get_cmpfmatrix_ij(a, ch[j], i));
			if(mpf_cmp(abs_tmp, axii) > 0)
			{
				imax = j;
				mpf_set(abs_axii, abs_tmp);
			}
		}

		if(mpf_cmp_ui(abs_axii, 0UL) == 0)
		{
			mpc_clear(tmp);
			mpf_clear(abs_axii);
			mpf_clear(abs_tmp);
			fprintf(stderr, "%ld : Error! MPCLUdecompP!\n", i);
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
			mpc_div(tmp, get_cmpfmatrix_ij(a, ch[j], i), get_cmpfmatrix_ij(a, ch[i], i));
			set_cmpfmatrix_ij(a, ch[j], i, tmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				mpc_mul(tmp, get_cmpfmatrix_ij(a, ch[j], i), get_cmpfmatrix_ij(a, ch[i], k));
				mpc_sub(tmp, get_cmpfmatrix_ij(a, ch[j], k), tmp);
				set_cmpfmatrix_ij(a, ch[j], k, tmp);
			}
		}
	}

	mpc_clear(tmp);
	mpf_clear(abs_axii);
	mpf_clear(abs_tmp);

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
int SolveMPCLSP(CMPFVector answer, CMPFMatrix lu, CMPFVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CMPFMatrix lu[]: LU decomposed Matrix(given by user)*/
/*       CMPFVector b[]: constant vector (given by user)     */
/*       CMPFVector answer[]: Solution for linear system     */
/*       long int ch: Row order (given by user)             */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	mpc_t tmp;

	mpc_init2(tmp, answer->prec);
	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_cmpfvector_i(answer, i, get_cmpfvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		if(mpc_cmp_si_si(get_cmpfmatrix_ij(lu, ch[i], i), 0L, 0L) == 0)
		{
			mpc_clear(tmp);
			fprintf(stderr, "Unable to solve the linear system!(SolveMPCLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			mpc_mul(tmp, get_cmpfmatrix_ij(lu, ch[j], i), get_cmpfvector_i(answer, i));
			mpc_sub(tmp, get_cmpfvector_i(answer, j), tmp);
			set_cmpfvector_i(answer, j, tmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			
			mpc_mul(tmp, get_cmpfmatrix_ij(lu, ch[i], j), get_cmpfvector_i(answer, j));
			mpc_sub(tmp, get_cmpfvector_i(answer, i), tmp);
			set_cmpfvector_i(answer, i, tmp);
		}
		mpc_div(tmp, get_cmpfvector_i(answer, i), get_cmpfmatrix_ij(lu, ch[i], i));
		set_cmpfvector_i(answer, i, tmp);
	}

	mpc_clear(tmp); // Fix! 2012-07-18 by T.Kouya

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
int MPCLUdecompC(CMPFMatrix a, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CMPFMatrix a[]: Matrix (given by user)              */
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
	mpc_t tmp;
	mpf_t abs_axii, abs_tmp;

	mpc_init2(tmp, a->prec);
	mpf_init2(abs_axii, a->prec);
	mpf_init2(abs_tmp, a->prec);

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
	{
		row_ch[i] = i;
		col_ch[i] = i;
	}

	for(i = 0; i < dim; i++)
	{
		/* Complete Pivoting */
		mpc_abs(abs_axii, get_cmpfmatrix_ij(a, row_ch[i], col_ch[i]));
		imax = i;
		jmax = i;
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				mpc_abs(abs_tmp, get_cmpfmatrix_ij(a, row_ch[j], col_ch[k]));
				if(mpf_cmp(abs_tmp, abs_axii) > 0)
				{
					imax = j;
					jmax = k;
					mpf_set(abs_axii, abs_tmp);
				}
			}
		}

		if(mpf_cmp_ui(abs_axii, 0UL) == 0)
		{
			mpc_clear(tmp);
			mpf_clear(abs_axii);
			mpf_clear(abs_tmp);

			fprintf(stderr, "%ld : Error! (MPCLUdecompC)!\n", i);
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
			mpc_div(tmp, get_cmpfmatrix_ij(a, row_ch[j], col_ch[i]),  get_cmpfmatrix_ij(a, row_ch[i], col_ch[i]));
			set_cmpfmatrix_ij(a, row_ch[j], col_ch[i], tmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				mpc_mul(tmp, get_cmpfmatrix_ij(a, row_ch[j], col_ch[i]), get_cmpfmatrix_ij(a, row_ch[i], col_ch[k]));
				mpc_sub(tmp, get_cmpfmatrix_ij(a, row_ch[j], col_ch[k]), tmp);
				set_cmpfmatrix_ij(a, row_ch[j], col_ch[k], tmp);
			}
		}
	}

	mpc_clear(tmp);
	mpf_clear(abs_axii);
	mpf_clear(abs_tmp);

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
int SolveMPCLSC(CMPFVector answer, CMPFMatrix lu, CMPFVector b, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CMPFMatrix lu: LU decomposed Matrix (given by user) */
/*       CMPFVector b: constant vector (given by user)       */
/*       CMPFVector answer: Solution for linear system       */
/*       long int row_ch[]: Row order (given by user)       */
/*       long int col_ch[]: Column order (given by user)    */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	mpc_t tmp;

	mpc_init2(tmp, answer->prec);
	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_cmpfvector_i(answer, col_ch[i], get_cmpfvector_i(b, row_ch[i]));

/* Forward */
	for(i = 0; i < dim; i++)
	{
		if(mpc_cmp_si_si(get_cmpfmatrix_ij(lu, row_ch[i], col_ch[i]), 0L, 0L) == 0)
		{
			mpc_clear(tmp);
			fprintf(stderr, "Unable to solve the linear system!(SolveMPCLSC, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			
			mpc_mul(tmp, get_cmpfmatrix_ij(lu, row_ch[j], col_ch[i]), get_cmpfvector_i(answer, col_ch[i]));
			mpc_sub(tmp, get_cmpfvector_i(answer, col_ch[j]), tmp);
			set_cmpfvector_i(answer, col_ch[j], tmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			mpc_mul(tmp, get_cmpfmatrix_ij(lu, row_ch[i], col_ch[j]), get_cmpfvector_i(answer, col_ch[j]));
			mpc_sub(tmp, get_cmpfvector_i(answer, col_ch[i]), tmp);
			set_cmpfvector_i(answer, col_ch[i], tmp);
		}
		mpc_div(tmp, get_cmpfvector_i(answer, col_ch[i]), get_cmpfmatrix_ij(lu, row_ch[i], col_ch[i]));
		set_cmpfvector_i(answer, col_ch[i], tmp);
	}

	mpc_clear(tmp);
	return 0;
}
#endif // USE_GMP
