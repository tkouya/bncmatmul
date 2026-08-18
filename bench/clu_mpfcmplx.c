/********************************************************************************/
/* clu.c:                                                                       */
/* Copyright (C) 2011-2012 Tomonori Kouya                                       */
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

#include "bnc.h"

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Complex Double Precision)       */
/*                                                          */
/*                 Ver. 0.0 2011-11-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CDLUdecomp(CDMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CDMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	double dtmp, dmaxii;

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
	{
		dmaxii = cabs(get_cdmatrix_ij(a, i, i));
		if(dmaxii == 0.0)
		{
			fprintf(stderr, "%ld : Error! (CDLUdecomp)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
			set_cdmatrix_ij(a, j, i, get_cdmatrix_ij(a, j, i) / get_cdmatrix_ij(a, i, i));
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
				set_cdmatrix_ij(a, j, k, get_cdmatrix_ij(a, j, k) - get_cdmatrix_ij(a, j, i) * get_cdmatrix_ij(a, i, k));
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                      (Complex Double Precision)          */
/*                                                          */
/*                 Ver. 0.0 2011-11-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveCDLS(CDVector answer, CDMatrix lu, CDVector b)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      CDMatrix lu: LU decomposed Matrix (given by user)   */
/*      CDVector b: constant vector (given by user)         */
/*      CDVector answer: Solution for linear system         */
/*      long int dim: Dimension of Matrix (given by user)   */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	double dtmp;

	dim = answer->dim;

	subst_cdvector(answer, b);

/* Forward */
	for(i = 0; i < dim; i++)
	{
		if(cabs(get_cdmatrix_ij(lu, i, i)) == 0.0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCDLS, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
			set_cdvector_i(answer, j, get_cdvector_i(answer, j) - get_cdmatrix_ij(lu, j, i) * get_cdvector_i(answer, i));
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
			set_cdvector_i(answer, i, get_cdvector_i(answer, i) - get_cdmatrix_ij(lu, i, j) * get_cdvector_i(answer, j));
		set_cdvector_i(answer, i, get_cdvector_i(answer, i) / get_cdmatrix_ij(lu, i, i));
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                               (Complex Double Precision) */
/*                               (Partial Pivoting)         */
/*                                                          */
/*                 Ver. 0.0 2011-11-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CDLUdecompP(CDMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CDMatrix a: Matrix (given by user)                 */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
	double dtmp, dmaxii;

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		dmaxii = cabs(get_cdmatrix_ij(a, ch[i], i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			dtmp = cabs(get_cdmatrix_ij(a, ch[j], i));
			if(dtmp > dmaxii)
			{
				imax = j;
				dmaxii = dtmp;
			}
		}

		if(dmaxii == 0.0)
		{
			fprintf(stderr, "%ld : Error! CDLUdecompP!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;
		}

		for(j = (i + 1); j < dim; j++)
			set_cdmatrix_ij(a, ch[j], i, get_cdmatrix_ij(a, ch[j], i) / get_cdmatrix_ij(a, ch[i], i));
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
				set_cdmatrix_ij(a, ch[j], k, get_cdmatrix_ij(a, ch[j], k) - get_cdmatrix_ij(a, ch[j], i) * get_cdmatrix_ij(a, ch[i], k));
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                               (Complex Double Precision) */
/*                               (Partial Pivoting)         */
/*                                                          */
/*                 Ver. 0.0 2011-11-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveCDLSP(CDVector answer, CDMatrix lu, CDVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      CDMatrix lu[]: LU decomposed Matrix (given by user) */
/*      CDVector b[]: constant vector (given by user)       */
/*      CDVector answer[]: Solution for linear system       */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	double dtmp;

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_cdvector_i(answer, i, get_cdvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		if(cabs(get_cdmatrix_ij(lu, ch[i], i)) == 0.0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCDLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
			set_cdvector_i(answer, j, get_cdvector_i(answer, j) - get_cdmatrix_ij(lu, ch[j], i) * get_cdvector_i(answer, i));
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
			set_cdvector_i(answer, i, get_cdvector_i(answer, i) - get_cdmatrix_ij(lu, ch[i], j) * get_cdvector_i(answer, j));
		set_cdvector_i(answer, i, get_cdvector_i(answer, i) / get_cdmatrix_ij(lu, ch[i], i));
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                               (Complex Double Precision) */
/*                               (Complete Pivoting)        */
/*                                                          */
/*                 Ver. 0.0 2011-11-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CDLUdecompC(CDMatrix a, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CDMatrix a[]: Matrix (given by user)               */
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
	double dtmp, dmaxii;

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
	{
		row_ch[i] = i;
		col_ch[i] = i;
	}

	for(i = 0; i < dim; i++)
	{
		/* Complete Pivoting */
		dmaxii = cabs(get_cdmatrix_ij(a, row_ch[i], col_ch[i]));
		imax = i;
		jmax = i;
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				dtmp = cabs(get_cdmatrix_ij(a, row_ch[j], col_ch[k]));
				if(dtmp > dmaxii)
				{
					imax = j;
					jmax = k;
					dmaxii = dtmp;
				}
			}
		}

		if(dmaxii == 0.0)
		{
			fprintf(stderr, "%ld : Error! (CDLUdecompC)!\n", i);
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
			set_cdmatrix_ij(a, row_ch[j], col_ch[i], get_cdmatrix_ij(a, row_ch[j], col_ch[i]) / get_cdmatrix_ij(a, row_ch[i], col_ch[i]));
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
				set_cdmatrix_ij(a, row_ch[j], col_ch[k], get_cdmatrix_ij(a, row_ch[j], col_ch[k]) - get_cdmatrix_ij(a, row_ch[j], col_ch[i]) * get_cdmatrix_ij(a, row_ch[i], col_ch[k]));
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                               (Complex Double Precision) */
/*                               (Complete Pivoting)        */
/*                                                          */
/*                 Ver. 0.0 2011-11-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveCDLSC(CDVector answer, CDMatrix lu, CDVector b, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CDMatrix lu: LU decomposed Matrix (given by user)  */
/*       CDVector b: constant vector (given by user)        */
/*       CDVector answer: Solution for linear system        */
/*       long int row_ch[]: Row order (given by user)       */
/*       long int col_ch[]: Column order (given by user)    */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	double dtmp;

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_cdvector_i(answer, col_ch[i], get_cdvector_i(b, row_ch[i]));

/* Forward */
	for(i = 0; i < dim; i++)
	{
		if(cabs(get_cdmatrix_ij(lu, row_ch[i], col_ch[i])) == 0.0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveCDLSC, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
			set_cdvector_i(answer, col_ch[j], get_cdvector_i(answer, col_ch[j]) - get_cdmatrix_ij(lu, row_ch[j], col_ch[i]) * get_cdvector_i(answer, col_ch[i]));
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
			set_cdvector_i(answer, col_ch[i], get_cdvector_i(answer, col_ch[i]) - get_cdmatrix_ij(lu, row_ch[i], col_ch[j]) * get_cdvector_i(answer, col_ch[j]));
		set_cdvector_i(answer, col_ch[i], get_cdvector_i(answer, col_ch[i]) / get_cdmatrix_ij(lu, row_ch[i], col_ch[i]));
	}

	return 0;
}

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
	MPFCmplx tmp;

	dim = a->col_dim;
	tmp = init2_mpfcmplx(a->prec);
	mpf_init2(axii, a->prec);
	for(i = 0; i < dim; i++)
	{
		//mpf_abs(axii, get_mpfmatrix_ij(a, i, i));
		abs_mpfcmplx(axii, get_cmpfmatrix_ij(a, i, i));
		if(mpf_cmp_ui(axii, 0UL) == 0)
		{
			free_mpfcmplx(tmp);
			mpf_clear(axii);
			fprintf(stderr, "%ld : Error! (CMPFLUdecomp)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			//mpf_div(tmp, get_mpfmatrix_ij(a, j, i), get_mpfmatrix_ij(a, i, i));
			div_mpfcmplx(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, i));
			set_cmpfmatrix_ij(a, j, i, tmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				//mpf_mul(tmp, get_mpfmatrix_ij(a, j, i), get_mpfmatrix_ij(a, i, k));
				mul_mpfcmplx(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_mpfmatrix_ij(a, j, k), tmp);
				sub_mpfcmplx(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
				set_cmpfmatrix_ij(a, j, k, tmp);
			}
		}
	}

	free_mpfcmplx(tmp);
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
	MPFCmplx ctmp;

	dim = answer->dim;

	subst_cmpfvector(answer, b);
	mpf_init2(tmp, answer->prec);
	ctmp = init2_mpfcmplx(answer->prec);

/* Forward */
	for(i = 0; i < dim; i++)
	{
		abs_mpfcmplx(tmp, get_cmpfmatrix_ij(lu, i, i));
		if(mpf_cmp_ui(tmp, 0UL) == 0)
		{
			mpf_clear(tmp);
			free_mpfcmplx(ctmp);
			fprintf(stderr, "Unable to solve the linear system!(SolveCMPFLS, %ld)\n", i);
			return -1;
		}

		subst_mpfcmplx(ctmp, get_cmpfvector_i(answer, i));
		for(j = (i + 1); j < dim; j++)
		{
			//mpf_mul(tmp, get_mpfmatrix_ij(lu, j, i), get_mpfvector_i(answer, i));
			//mpf_sub(tmp, get_mpfvector_i(answer, j), tmp);
			mul_mpfcmplx(ctmp, get_cmpfmatrix_ij(lu, j, i), get_cmpfvector_i(answer, i));
			sub_mpfcmplx(ctmp, get_cmpfvector_i(answer, j), ctmp);
			set_cmpfvector_i(answer, j, ctmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			//mpf_mul(tmp, get_mpfmatrix_ij(lu, i, j), get_mpfvector_i(answer, j));
			//mpf_sub(tmp, get_mpfvector_i(answer, i),  tmp);
			mul_mpfcmplx(ctmp, get_cmpfmatrix_ij(lu, i, j), get_cmpfvector_i(answer, j));
			sub_mpfcmplx(ctmp, get_cmpfvector_i(answer, i),  ctmp);
			set_cmpfvector_i(answer, i, ctmp);
		}
		//mpf_div(tmp, get_mpfvector_i(answer, i), get_mpfmatrix_ij(lu, i, i));
		div_mpfcmplx(ctmp, get_cmpfvector_i(answer, i), get_cmpfmatrix_ij(lu, i, i));
		set_cmpfvector_i(answer, i, ctmp);
	}

	mpf_clear(tmp);
	free_mpfcmplx(ctmp);
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
	MPFCmplx ctmp;

	dim = a->col_dim;

	mpf_init2(tmp, a->prec);
	mpf_init2(axii, a->prec);
	ctmp = init2_mpfcmplx(a->prec);
	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		//mpf_abs(axii, get_mpfmatrix_ij(a, ch[i], i));
		abs_mpfcmplx(axii, get_cmpfmatrix_ij(a, ch[i], i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			//mpf_abs(tmp, get_mpfmatrix_ij(a, ch[j], i));
			abs_mpfcmplx(tmp, get_cmpfmatrix_ij(a, ch[j], i));
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
			free_mpfcmplx(ctmp);
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
			div_mpfcmplx(ctmp, get_cmpfmatrix_ij(a, ch[j], i), get_cmpfmatrix_ij(a, ch[i], i));
			set_cmpfmatrix_ij(a, ch[j], i, ctmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				//mpf_mul(tmp, get_mpfmatrix_ij(a, ch[j], i), get_mpfmatrix_ij(a, ch[i], k));
				//mpf_sub(tmp, get_mpfmatrix_ij(a, ch[j], k), tmp);
				mul_mpfcmplx(ctmp, get_cmpfmatrix_ij(a, ch[j], i), get_cmpfmatrix_ij(a, ch[i], k));
				sub_mpfcmplx(ctmp, get_cmpfmatrix_ij(a, ch[j], k), ctmp);
				set_cmpfmatrix_ij(a, ch[j], k, ctmp);
			}
		}
	}

	mpf_clear(tmp);
	mpf_clear(axii);
	free_mpfcmplx(ctmp);

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
	MPFCmplx ctmp;

	mpf_init2(tmp, answer->prec);
	ctmp = init2_mpfcmplx(answer->prec);
	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_cmpfvector_i(answer, i, get_cmpfvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		abs_mpfcmplx(tmp, get_cmpfmatrix_ij(lu, ch[i], i));
		if(mpf_cmp_ui(tmp, 0UL) == 0)
		{
			mpf_clear(tmp);
			free_mpfcmplx(ctmp);
			fprintf(stderr, "Unable to solve the linear system!(SolveCMPFLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			//mpf_mul(tmp, get_mpfmatrix_ij(lu, ch[j], i), get_mpfvector_i(answer, i));
			//mpf_sub(tmp, get_mpfvector_i(answer, j), tmp);
			mul_mpfcmplx(ctmp, get_cmpfmatrix_ij(lu, ch[j], i), get_cmpfvector_i(answer, i));
			sub_mpfcmplx(ctmp, get_cmpfvector_i(answer, j), ctmp);
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
			mul_mpfcmplx(ctmp, get_cmpfmatrix_ij(lu, ch[i], j), get_cmpfvector_i(answer, j));
			sub_mpfcmplx(ctmp, get_cmpfvector_i(answer, i), ctmp);
			set_cmpfvector_i(answer, i, ctmp);
		}
		//mpf_div(tmp, get_mpfvector_i(answer, i), get_mpfmatrix_ij(lu, ch[i], i));
		div_mpfcmplx(ctmp, get_cmpfvector_i(answer, i), get_cmpfmatrix_ij(lu, ch[i], i));
		set_cmpfvector_i(answer, i, ctmp);
	}

	mpf_init(tmp);
	free_mpfcmplx(ctmp);

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
	MPFCmplx ctmp;

	mpf_init2(tmp, a->prec);
	mpf_init2(axii, a->prec);
	ctmp = init2_mpfcmplx(a->prec);
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
		abs_mpfcmplx(axii, get_cmpfmatrix_ij(a, row_ch[i], col_ch[i]));
		imax = i;
		jmax = i;
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				//mpf_abs(tmp, get_mpfmatrix_ij(a, row_ch[j], col_ch[k]));
				abs_mpfcmplx(tmp, get_cmpfmatrix_ij(a, row_ch[j], col_ch[k]));
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
			free_mpfcmplx(ctmp);
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
			div_mpfcmplx(ctmp, get_cmpfmatrix_ij(a, row_ch[j], col_ch[i]),  get_cmpfmatrix_ij(a, row_ch[i], col_ch[i]));
			set_cmpfmatrix_ij(a, row_ch[j], col_ch[i], ctmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				//mpf_mul(tmp, get_mpfmatrix_ij(a, row_ch[j], col_ch[i]), get_mpfmatrix_ij(a, row_ch[i], col_ch[k]));
				//mpf_sub(tmp, get_mpfmatrix_ij(a, row_ch[j], col_ch[k]), tmp);
				mul_mpfcmplx(ctmp, get_cmpfmatrix_ij(a, row_ch[j], col_ch[i]), get_cmpfmatrix_ij(a, row_ch[i], col_ch[k]));
				sub_mpfcmplx(ctmp, get_cmpfmatrix_ij(a, row_ch[j], col_ch[k]), ctmp);
				set_cmpfmatrix_ij(a, row_ch[j], col_ch[k], ctmp);
			}
		}
	}

	mpf_clear(tmp);
	mpf_clear(axii);
	free_mpfcmplx(ctmp);

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
	MPFCmplx ctmp;

	mpf_init2(tmp, answer->prec);
	ctmp = init2_mpfcmplx(answer->prec);
	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_cmpfvector_i(answer, col_ch[i], get_cmpfvector_i(b, row_ch[i]));

/* Forward */
	for(i = 0; i < dim; i++)
	{
		abs_mpfcmplx(tmp, get_cmpfmatrix_ij(lu, row_ch[i], col_ch[i]));
		if(mpf_cmp_ui(tmp, 0UL) == 0)
		{
			mpf_clear(tmp);
			free_mpfcmplx(ctmp);
			fprintf(stderr, "Unable to solve the linear system!(SolveCMPFLSC, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			//mpf_mul(tmp, get_mpfmatrix_ij(lu, row_ch[j], col_ch[i]), get_mpfvector_i(answer, col_ch[i]));
			//mpf_sub(tmp, get_mpfvector_i(answer, col_ch[j]), tmp);
			mul_mpfcmplx(ctmp, get_cmpfmatrix_ij(lu, row_ch[j], col_ch[i]), get_cmpfvector_i(answer, col_ch[i]));
			sub_mpfcmplx(ctmp, get_cmpfvector_i(answer, col_ch[j]), ctmp);
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
			mul_mpfcmplx(ctmp, get_cmpfmatrix_ij(lu, row_ch[i], col_ch[j]), get_cmpfvector_i(answer, col_ch[j]));
			sub_mpfcmplx(ctmp, get_cmpfvector_i(answer, col_ch[i]), ctmp);
			set_cmpfvector_i(answer, col_ch[i], ctmp);
		}
		//mpf_div(tmp, get_mpfvector_i(answer, col_ch[i]), get_mpfmatrix_ij(lu, row_ch[i], col_ch[i]));
		div_mpfcmplx(ctmp, get_cmpfvector_i(answer, col_ch[i]), get_cmpfmatrix_ij(lu, row_ch[i], col_ch[i]));
		set_cmpfvector_i(answer, col_ch[i], ctmp);
	}

	mpf_clear(tmp);
	free_mpfcmplx(ctmp);

	return 0;
}
#endif

