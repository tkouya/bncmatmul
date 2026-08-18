/********************************************************************************/
/* cdlu.c:                                                                      */
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
#include "cdlinear.h"

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
