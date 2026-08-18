/********************************************************************************/
/* cmpflu.c:                                                                    */
/* Copyright (C) 2024 Tomonori Kouya                                            */
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
/* stdio.h */
#ifndef _STDIO_H
#include <stdio.h>
#endif

/* math.h */
#ifndef _MATH_H
#include <math.h>
#endif

// #include "bnc.h"
#include "cmpflinear.h"

#ifdef USE_GMP
/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                (Complex Multi-Precision) */
/*                                                          */
/*                 Ver. 0.0 2011-11-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CMPFLUdecomp(CMPFMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CMPFMatrix a: Matrix (given by user)               */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	mpf_t axii;
	mpc_t tmp;

	dim = a->col_dim;
	mpc_init2(tmp, a->prec);
	mpf_init2(axii, a->prec);
	for(i = 0; i < dim; i++)
	{
		//mpf_abs(axii, get_mpfmatrix_ij(a, i, i));
		mpc_abs(axii, get_cmpfmatrix_ij(a, i, i), get_bnc_default_rounding_mode());
		//printf("a%ld_%ld = ", i, i); mpfr_out_str(stdout, 10, 15, axii, MPFR_RNDN); printf("\n");
		if(mpf_cmp_ui(axii, 0UL) == 0)
		{
			mpc_clear(tmp);
			mpf_clear(axii);
			fprintf(stderr, "%ld : Error! (CMPFLUdecomp)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			//mpf_div(tmp, get_mpfmatrix_ij(a, j, i), get_mpfmatrix_ij(a, i, i));
			mpc_div(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, i), get_bnc_default_rounding_mode());
			set_cmpfmatrix_ij(a, j, i, tmp);
			//printf("a%ld_%ld = ", j, i); mpc_out_str(stdout, 10, 15, tmp, MPC_RNDNN); printf("\n");
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				//mpf_mul(tmp, get_mpfmatrix_ij(a, j, i), get_mpfmatrix_ij(a, i, k));
				mpc_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k), get_bnc_default_rounding_mode());
				//printf("a%ld_%ld = ", j, k); mpc_out_str(stdout, 10, 15, tmp, MPC_RNDNN); printf("\n");
				//mpf_sub(tmp, get_mpfmatrix_ij(a, j, k), tmp);
				mpc_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp, get_bnc_default_rounding_mode());
				set_cmpfmatrix_ij(a, j, k, tmp);
				//printf("a%ld_%ld = ", j, k); mpc_out_str(stdout, 10, 15, tmp, MPC_RNDNN); printf("\n");
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
/*                                (Complex Muiti-Precision) */
/*                                                          */
/*                 Ver. 0.0 2011-11-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveCMPFLS(CMPFVector answer, CMPFMatrix lu, CMPFVector b)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      CMPFMatrix lu: LU decomposed Matrix (given by user) */
/*      CMPFVector b: constant vector (given by user)       */
/*      CMPFVector answer: Solution for linear system       */
/*      long int dim: Dimension of Matrix (given by user)   */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	mpf_t tmp;
	mpc_t ctmp;

	dim = answer->dim;

	subst_cmpfvector(answer, b);
	mpf_init2(tmp, answer->prec);
	mpc_init2(ctmp, answer->prec);

/* Forward */
	for(i = 0; i < dim; i++)
	{
		mpc_abs(tmp, get_cmpfmatrix_ij(lu, i, i), get_bnc_default_rounding_mode());
		if(mpf_cmp_ui(tmp, 0UL) == 0)
		{
			mpf_clear(tmp);
			mpc_clear(ctmp);
			fprintf(stderr, "Unable to solve the linear system!(SolveCMPFLS, %ld)\n", i);
			return -1;
		}

		mpc_set(ctmp, get_cmpfvector_i(answer, i), get_bnc_default_rounding_mode());
		for(j = (i + 1); j < dim; j++)
		{
			//mpf_mul(tmp, get_mpfmatrix_ij(lu, j, i), get_mpfvector_i(answer, i));
			//mpf_sub(tmp, get_mpfvector_i(answer, j), tmp);
			mpc_mul(ctmp, get_cmpfmatrix_ij(lu, j, i), get_cmpfvector_i(answer, i), get_bnc_default_rounding_mode());
			mpc_sub(ctmp, get_cmpfvector_i(answer, j), ctmp, get_bnc_default_rounding_mode());
			set_cmpfvector_i(answer, j, ctmp);
		}
		//printf("f %ld = ", i); mpc_out_str(stdout, 10, 15, get_cmpfvector_i(answer, i), MPC_RNDNN); printf("\n");
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			//mpf_mul(tmp, get_mpfmatrix_ij(lu, i, j), get_mpfvector_i(answer, j));
			//mpf_sub(tmp, get_mpfvector_i(answer, i),  tmp);
			mpc_mul(ctmp, get_cmpfmatrix_ij(lu, i, j), get_cmpfvector_i(answer, j), get_bnc_default_rounding_mode());
			mpc_sub(ctmp, get_cmpfvector_i(answer, i),  ctmp, get_bnc_default_rounding_mode());
			set_cmpfvector_i(answer, i, ctmp);
		}
		//mpf_div(tmp, get_mpfvector_i(answer, i), get_mpfmatrix_ij(lu, i, i));
		mpc_div(ctmp, get_cmpfvector_i(answer, i), get_cmpfmatrix_ij(lu, i, i), get_bnc_default_rounding_mode());
		set_cmpfvector_i(answer, i, ctmp);
		//printf("b %ld = ", i); mpc_out_str(stdout, 10, 15, get_cmpfvector_i(answer, i), MPC_RNDNN); printf("\n");
	}

	mpf_clear(tmp);
	mpc_clear(ctmp);
	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                (Complex Multi-Precision) */
/*                                (Partial Pivoting)        */
/*                                                          */
/*                 Ver. 0.0 2011-11-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CMPFLUdecompP(CMPFMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CMPFMatrix a: Matrix (given by user)               */
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
	mpc_t ctmp;

	dim = a->col_dim;

	mpf_init2(tmp, a->prec);
	mpf_init2(axii, a->prec);
	mpc_init2(ctmp, a->prec);
	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		//mpf_abs(axii, get_mpfmatrix_ij(a, ch[i], i));
		mpc_abs(axii, get_cmpfmatrix_ij(a, ch[i], i), get_bnc_default_rounding_mode());
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			//mpf_abs(tmp, get_mpfmatrix_ij(a, ch[j], i));
			mpc_abs(tmp, get_cmpfmatrix_ij(a, ch[j], i), get_bnc_default_rounding_mode());
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
			mpc_clear(ctmp);
			fprintf(stderr, "%ld : Error! CMPFLUdecompP!\n", i);

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
			//mpf_div(tmp, get_mpfmatrix_ij(a, ch[j], i), get_mpfmatrix_ij(a, ch[i], i));
			mpc_div(ctmp, get_cmpfmatrix_ij(a, ch[j], i), get_cmpfmatrix_ij(a, ch[i], i), get_bnc_default_rounding_mode());
			set_cmpfmatrix_ij(a, ch[j], i, ctmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				//mpf_mul(tmp, get_mpfmatrix_ij(a, ch[j], i), get_mpfmatrix_ij(a, ch[i], k));
				//mpf_sub(tmp, get_mpfmatrix_ij(a, ch[j], k), tmp);
				mpc_mul(ctmp, get_cmpfmatrix_ij(a, ch[j], i), get_cmpfmatrix_ij(a, ch[i], k), get_bnc_default_rounding_mode());
				mpc_sub(ctmp, get_cmpfmatrix_ij(a, ch[j], k), ctmp, get_bnc_default_rounding_mode());
				set_cmpfmatrix_ij(a, ch[j], k, ctmp);
			}
		}
	}

	mpf_clear(tmp);
	mpf_clear(axii);
	mpc_clear(ctmp);

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                               (Complex Multi-Precision)  */
/*                               (Partial Pivoting)         */
/*                                                          */
/*                 Ver. 0.0 2011-11-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveCMPFLSP(CMPFVector answer, CMPFMatrix lu, CMPFVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*     CMPFMatrix lu[]: LU decomposed Matrix(given by user) */
/*     CMPFVector b[]: constant vector (given by user)      */
/*     CMPFVector answer[]: Solution for linear system      */
/*     long int ch: Row order (given by user)               */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	mpf_t tmp;
	mpc_t ctmp;

	mpf_init2(tmp, answer->prec);
	mpc_init2(ctmp, answer->prec);
	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_cmpfvector_i(answer, i, get_cmpfvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		mpc_abs(tmp, get_cmpfmatrix_ij(lu, ch[i], i), get_bnc_default_rounding_mode());
		if(mpf_cmp_ui(tmp, 0UL) == 0)
		{
			mpf_clear(tmp);
			mpc_clear(ctmp);
			fprintf(stderr, "Unable to solve the linear system!(SolveCMPFLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			//mpf_mul(tmp, get_mpfmatrix_ij(lu, ch[j], i), get_mpfvector_i(answer, i));
			//mpf_sub(tmp, get_mpfvector_i(answer, j), tmp);
			mpc_mul(ctmp, get_cmpfmatrix_ij(lu, ch[j], i), get_cmpfvector_i(answer, i), get_bnc_default_rounding_mode());
			mpc_sub(ctmp, get_cmpfvector_i(answer, j), ctmp, get_bnc_default_rounding_mode());
			set_cmpfvector_i(answer, j, ctmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			//mpf_mul(tmp, get_mpfmatrix_ij(lu, ch[i], j), get_mpfvector_i(answer, j));
			//mpf_sub(tmp, get_mpfvector_i(answer, i), tmp);
			mpc_mul(ctmp, get_cmpfmatrix_ij(lu, ch[i], j), get_cmpfvector_i(answer, j), get_bnc_default_rounding_mode());
			mpc_sub(ctmp, get_cmpfvector_i(answer, i), ctmp, get_bnc_default_rounding_mode());
			set_cmpfvector_i(answer, i, ctmp);
		}
		//mpf_div(tmp, get_mpfvector_i(answer, i), get_mpfmatrix_ij(lu, ch[i], i));
		mpc_div(ctmp, get_cmpfvector_i(answer, i), get_cmpfmatrix_ij(lu, ch[i], i), get_bnc_default_rounding_mode());
		set_cmpfvector_i(answer, i, ctmp);
	}

	mpf_init(tmp);
	mpc_clear(ctmp);

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                (Complex Multi-Precision) */
/*                                (Complete Pivoting)       */
/*                                                          */
/*                 Ver. 0.0 2011-11-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CMPFLUdecompC(CMPFMatrix a, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CMPFMatrix a[]: Matrix (given by user)             */
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
	mpc_t ctmp;

	mpf_init2(tmp, a->prec);
	mpf_init2(axii, a->prec);
	mpc_init2(ctmp, a->prec);
	dim = a->col_dim;

	for(i = 0; i < dim; i++)
	{
		row_ch[i] = i;
		col_ch[i] = i;
	}

	for(i = 0; i < dim; i++)
	{
		/* Complete Pivoting */
		//mpf_abs(axii, get_mpfmatrix_ij(a, row_ch[i], col_ch[i]));
		mpc_abs(axii, get_cmpfmatrix_ij(a, row_ch[i], col_ch[i]), get_bnc_default_rounding_mode());
		imax = i;
		jmax = i;
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				//mpf_abs(tmp, get_mpfmatrix_ij(a, row_ch[j], col_ch[k]));
				mpc_abs(tmp, get_cmpfmatrix_ij(a, row_ch[j], col_ch[k]), get_bnc_default_rounding_mode());
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
			mpc_clear(ctmp);
			fprintf(stderr, "%ld : Error! (CMPFLUdecompC)!\n", i);
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
			//mpf_div(tmp, get_mpfmatrix_ij(a, row_ch[j], col_ch[i]),  get_mpfmatrix_ij(a, row_ch[i], col_ch[i]));
			mpc_div(ctmp, get_cmpfmatrix_ij(a, row_ch[j], col_ch[i]),  get_cmpfmatrix_ij(a, row_ch[i], col_ch[i]), get_bnc_default_rounding_mode());
			set_cmpfmatrix_ij(a, row_ch[j], col_ch[i], ctmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				//mpf_mul(tmp, get_mpfmatrix_ij(a, row_ch[j], col_ch[i]), get_mpfmatrix_ij(a, row_ch[i], col_ch[k]));
				//mpf_sub(tmp, get_mpfmatrix_ij(a, row_ch[j], col_ch[k]), tmp);
				mpc_mul(ctmp, get_cmpfmatrix_ij(a, row_ch[j], col_ch[i]), get_cmpfmatrix_ij(a, row_ch[i], col_ch[k]), get_bnc_default_rounding_mode());
				mpc_sub(ctmp, get_cmpfmatrix_ij(a, row_ch[j], col_ch[k]), ctmp, get_bnc_default_rounding_mode());
				set_cmpfmatrix_ij(a, row_ch[j], col_ch[k], ctmp);
			}
		}
	}

	mpf_clear(tmp);
	mpf_clear(axii);
	mpc_clear(ctmp);

	return 0;
}

/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                                (Complex Multi-Precision) */
/*                                (Complete Pivoting)       */
/*                                                          */
/*                 Ver. 0.0 2011-11-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveCMPFLSC(CMPFVector answer, CMPFMatrix lu, CMPFVector b, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      CMPFMatrix lu: LU decomposed Matrix (given by user) */
/*      CMPFVector b: constant vector (given by user)       */
/*      CMPFVector answer: Solution for linear system       */
/*      long int row_ch[]: Row order (given by user)        */
/*      long int col_ch[]: Column order (given by user)     */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	mpf_t tmp;
	mpc_t ctmp;

	mpf_init2(tmp, answer->prec);
	mpc_init2(ctmp, answer->prec);
	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_cmpfvector_i(answer, col_ch[i], get_cmpfvector_i(b, row_ch[i]));

/* Forward */
	for(i = 0; i < dim; i++)
	{
		mpc_abs(tmp, get_cmpfmatrix_ij(lu, row_ch[i], col_ch[i]), get_bnc_default_rounding_mode());
		{
			mpf_clear(tmp);
			mpc_clear(ctmp);
			fprintf(stderr, "Unable to solve the linear system!(SolveCMPFLSC, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			//mpf_mul(tmp, get_mpfmatrix_ij(lu, row_ch[j], col_ch[i]), get_mpfvector_i(answer, col_ch[i]));
			//mpf_sub(tmp, get_mpfvector_i(answer, col_ch[j]), tmp);
			mpc_mul(ctmp, get_cmpfmatrix_ij(lu, row_ch[j], col_ch[i]), get_cmpfvector_i(answer, col_ch[i]), get_bnc_default_rounding_mode());
			mpc_sub(ctmp, get_cmpfvector_i(answer, col_ch[j]), ctmp, get_bnc_default_rounding_mode());
			set_cmpfvector_i(answer, col_ch[j], ctmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			//mpf_mul(tmp, get_mpfmatrix_ij(lu, row_ch[i], col_ch[j]), get_mpfvector_i(answer, col_ch[j]));
			//mpf_sub(tmp, get_mpfvector_i(answer, col_ch[i]), tmp);
			mpc_mul(ctmp, get_cmpfmatrix_ij(lu, row_ch[i], col_ch[j]), get_cmpfvector_i(answer, col_ch[j]), get_bnc_default_rounding_mode());
			mpc_sub(ctmp, get_cmpfvector_i(answer, col_ch[i]), ctmp, get_bnc_default_rounding_mode());
			set_cmpfvector_i(answer, col_ch[i], ctmp);
		}
		//mpf_div(tmp, get_mpfvector_i(answer, col_ch[i]), get_mpfmatrix_ij(lu, row_ch[i], col_ch[i]));
		mpc_div(ctmp, get_cmpfvector_i(answer, col_ch[i]), get_cmpfmatrix_ij(lu, row_ch[i], col_ch[i]), get_bnc_default_rounding_mode());
		set_cmpfvector_i(answer, col_ch[i], ctmp);
	}

	mpf_clear(tmp);
	mpc_clear(ctmp);

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
int CMPFLUdecompPM(CMPFMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CMPFMatrix a: Matrix (given by user)               */
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
	mpf_t abs_tmp, axii;

	dim = a->col_dim;

	mpc_init2(tmp, a->prec);
	mpf_init2(axii, a->prec);
	mpf_init2(abs_tmp, a->prec);

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		mpc_abs(axii, get_cmpfmatrix_ij(a, i, i), get_bnc_default_rounding_mode());
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			mpc_abs(abs_tmp, get_cmpfmatrix_ij(a, j, i), get_bnc_default_rounding_mode());
			if(mpf_cmp(abs_tmp, axii) > 0)
			{
				imax = j;
				mpf_set(axii, abs_tmp);
			}
		}

		if(mpf_cmp_ui(axii, 0UL) == 0)
		{
			mpc_clear(tmp);
			mpf_clear(axii);
			mpf_clear(abs_tmp);
			fprintf(stderr, "%ld : Error! CMPFLUdecompP!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			row_swap_cmpfmatrix(a, i, imax, 0, a->col_dim);
		}

		for(j = (i + 1); j < dim; j++)
		{
			mpc_div(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, i), get_bnc_default_rounding_mode());
			set_cmpfmatrix_ij(a, j, i, tmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				mpc_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k), get_bnc_default_rounding_mode());
				mpc_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp, get_bnc_default_rounding_mode());
				set_cmpfmatrix_ij(a, j, k, tmp);
			}
		}
	}

	mpc_clear(tmp);
	mpf_clear(axii);
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
int SolveCMPFLSPM(CMPFVector answer, CMPFMatrix lu, CMPFVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      CMPFMatrix lu[]: LU decomposed Matrix(given by user)*/
/*      CMPFVector b[]: constant vector (given by user)     */
/*      CMPFVector answer[]: Solution for linear system     */
/*      long int ch: Row order (given by user)              */
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
		if(mpc_cmp_si_si(get_cmpfmatrix_ij(lu, i, i), 0L, 0L) == 0)
		{
			mpc_clear(tmp);
			fprintf(stderr, "Unable to solve the linear system!(SolveCMPFLSPM, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			mpc_mul(tmp, get_cmpfmatrix_ij(lu, j, i), get_cmpfvector_i(answer, i), get_bnc_default_rounding_mode());
			mpc_sub(tmp, get_cmpfvector_i(answer, j), tmp, get_bnc_default_rounding_mode());
			set_cmpfvector_i(answer, j, tmp);
		}
	}

/* Backward */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			
			mpc_mul(tmp, get_cmpfmatrix_ij(lu, i, j), get_cmpfvector_i(answer, j), get_bnc_default_rounding_mode());
			mpc_sub(tmp, get_cmpfvector_i(answer, i), tmp, get_bnc_default_rounding_mode());
			set_cmpfvector_i(answer, i, tmp);
		}
		mpc_div(tmp, get_cmpfvector_i(answer, i), get_cmpfmatrix_ij(lu, i, i), get_bnc_default_rounding_mode());
		set_cmpfvector_i(answer, i, tmp);
	}

	mpc_clear(tmp); // Fix! 2012-07-18 by T.Kouya

	return 0;
}
#endif // USE_GMP

