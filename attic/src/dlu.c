/********************************************************************************/
/* lu.c:                                                                        */
/* Copyright (C) 2003- Tomonori Kouya                                           */
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

//#include "bnc.h"
#include "dlinear.h"

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 1997.10.24 (Fri) Tomonori Kouya */
/*                 ver. 0.1 2000.02.24 (Thu) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int DLUdecomp(DMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DMatrix a: Matrix (given by user)                 */
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
		dmaxii = fabs(get_dmatrix_ij(a, i, i));
		if(dmaxii == 0.0)
		{
			fprintf(stderr, "%ld : Error! (DLUdecomp)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
			set_dmatrix_ij(a, j, i, get_dmatrix_ij(a, j, i) / get_dmatrix_ij(a, i, i));
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
				set_dmatrix_ij(a, j, k, get_dmatrix_ij(a, j, k) - get_dmatrix_ij(a, j, i) * get_dmatrix_ij(a, i, k));
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 1997.10.24 (Fri) Tomonori Kouya */
/*                 ver. 0.1 2000.02.24 (Thu) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveDLS(DVector answer, DMatrix lu, DVector b)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DMatrix lu: LU decomposed Matrix (given by user)   */
/*       DVector b: constant vector (given by user)         */
/*       DVector answer: Solution for linear system         */
/*       long int dim: Dimension of Matrix (given by user)  */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	double dtmp;

	dim = answer->dim;

	subst_dvector(answer, b);

/* Forward */
	for(i = 0; i < dim; i++)
	{
		if(get_dmatrix_ij(lu, i, i) == 0.0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveDLS, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
			set_dvector_i(answer, j, get_dvector_i(answer, j) - get_dmatrix_ij(lu, j, i) * get_dvector_i(answer, i));
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
			set_dvector_i(answer, i, get_dvector_i(answer, i) - get_dmatrix_ij(lu, i, j) * get_dvector_i(answer, j));
		set_dvector_i(answer, i, get_dvector_i(answer, i) / get_dmatrix_ij(lu, i, i));
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                 (Double Precision)       */
/*                                 (Partial Pivoting)       */
/*                                                          */
/*                 ver. 0.0 1997.10.24 (Fri) Tomonori Kouya */
/*                 ver. 0.1 2000.02.24 (Thu) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int DLUdecompP(DMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DMatrix a: Matrix (given by user)                  */
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
		dmaxii = fabs(get_dmatrix_ij(a, ch[i], i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			dtmp = fabs(get_dmatrix_ij(a, ch[j], i));
			if(dtmp > dmaxii)
			{
				imax = j;
				dmaxii = dtmp;
			}
		}

		if(dmaxii == 0.0)
		{
			fprintf(stderr, "%ld : Error! DLUdecompP!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;
		}

		for(j = (i + 1); j < dim; j++)
			set_dmatrix_ij(a, ch[j], i, get_dmatrix_ij(a, ch[j], i) / get_dmatrix_ij(a, ch[i], i));
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
				set_dmatrix_ij(a, ch[j], k, get_dmatrix_ij(a, ch[j], k) - get_dmatrix_ij(a, ch[j], i) * get_dmatrix_ij(a, ch[i], k));
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                                 (Double Precision)       */
/*                                 (Partial Pivoting)       */
/*                                                          */
/*                 ver. 0.0 1997.10.24 (Fri) Tomonori Kouya */
/*                 ver. 0.1 2000.02.24 (Thu) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveDLSP(DVector answer, DMatrix lu, DVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DMatrix lu[]: LU decomposed Matrix (given by user) */
/*       DVector b[]: constant vector (given by user)       */
/*       DVector answer[]: Solution for linear system       */
/*       long int ch: Row order (given by user)             */
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
		set_dvector_i(answer, i, get_dvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		if(get_dmatrix_ij(lu, ch[i], i) == 0.0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveDLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
			set_dvector_i(answer, j, get_dvector_i(answer, j) - get_dmatrix_ij(lu, ch[j], i) * get_dvector_i(answer, i));
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
			set_dvector_i(answer, i, get_dvector_i(answer, i) - get_dmatrix_ij(lu, ch[i], j) * get_dvector_i(answer, j));
		set_dvector_i(answer, i, get_dvector_i(answer, i) / get_dmatrix_ij(lu, ch[i], i));
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                 (Double Precision)       */
/*                                 (Complete Pivoting)      */
/*                                                          */
/*                 ver. 0.0 1997.10.24 (Fri) Tomonori Kouya */
/*                 ver. 0.1 2000.02.24 (Thu) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int DLUdecompC(DMatrix a, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DMatrix a[]: Matrix (given by user)                */
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
		dmaxii = fabs(get_dmatrix_ij(a, row_ch[i], col_ch[i]));
		imax = i;
		jmax = i;
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				dtmp = fabs(get_dmatrix_ij(a, row_ch[j], col_ch[k]));
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
			fprintf(stderr, "%ld : Error! (DLUdecompC)!\n", i);
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
			set_dmatrix_ij(a, row_ch[j], col_ch[i], get_dmatrix_ij(a, row_ch[j], col_ch[i]) / get_dmatrix_ij(a, row_ch[i], col_ch[i]));
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
				set_dmatrix_ij(a, row_ch[j], col_ch[k], get_dmatrix_ij(a, row_ch[j], col_ch[k]) - get_dmatrix_ij(a, row_ch[j], col_ch[i]) * get_dmatrix_ij(a, row_ch[i], col_ch[k]));
		}
	}

	return 0;
}
