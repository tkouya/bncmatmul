/********************************************************************************/
/* ddlu.c:                                                                      */
/* Copyright (C) 2015 Tomonori Kouya                                            */
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
#include "ddlinear.h"

// DD

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Double-Double Precision)        */
/*                                                          */
/*                 Ver. 0.0 2015-02-23 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int DDLUdecomp(DDMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DDMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
#ifdef __cplusplus
	dd_real dtmp, dtmp1, dmaxii;
#else // __cplusplus
	static double dtmp[DDSIZE], dtmp1[DDSIZE], dmaxii[DDSIZE];
#endif // __cplusplus

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
	{
		rdd_abs(dmaxii, get_ddmatrix_ij(a, i, i));
		//printf("a%ld_%ld = ", i, i); rdd_out_str(dmaxii); printf("\n");
		if(rdd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (DDLUdecomp)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rdd_div(dtmp, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, i));
			set_ddmatrix_ij(a, j, i, dtmp);
			//printf("a%ld_%ld = ", j, i); rdd_out_str(dtmp); printf("\n");
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rdd_mul(dtmp1, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(dtmp, get_ddmatrix_ij(a, j, k), dtmp1);
				set_ddmatrix_ij(a, j, k, dtmp);
				//printf("a%ld_%ld= ", j, k); rdd_out_str(dtmp); printf("\n");
			}
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
int SolveDDLS(DDVector answer, DDMatrix lu, DDVector b)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      DDMatrix lu: LU decomposed Matrix (given by user)   */
/*      DDVector b: constant vector (given by user)         */
/*      DDVector answer: Solution for linear system         */
/*      long int dim: Dimension of Matrix (given by user)   */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
#ifdef __cplusplus
	static dd_real dtmp, dtmp1;
#else // __cplusplus
	static double dtmp[DDSIZE], dtmp1[DDSIZE];
#endif // __cplusplus

	dim = answer->dim;

	subst_ddvector(answer, b);

/* Forward */
	for(i = 0; i < dim; i++)
	{
		//printf("f %ld = ", i); rdd_out_str(get_ddvector_i(answer, i)); printf("\n");
		rdd_abs(dtmp, get_ddmatrix_ij(lu, i, i));
		if(rdd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveDDLS, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rdd_mul(dtmp1, get_ddmatrix_ij(lu, j, i), get_ddvector_i(answer, i));
			rdd_sub(dtmp, get_ddvector_i(answer, j), dtmp1);
			set_ddvector_i(answer, j, dtmp);
		}
		//printf("f %ld = ", i); rdd_out_str(get_ddvector_i(answer, i)); printf("\n");
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rdd_mul(dtmp1, get_ddmatrix_ij(lu, i, j), get_ddvector_i(answer, j));
			rdd_sub(dtmp, get_ddvector_i(answer, i), dtmp1);
			set_ddvector_i(answer, i, dtmp);
		}
		rdd_div(dtmp, get_ddvector_i(answer, i), get_ddmatrix_ij(lu, i, i));
		set_ddvector_i(answer, i, dtmp);
		//printf("b %ld = ", i); rdd_out_str(get_ddvector_i(answer, i)); printf("\n");
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                               (Double-Double Precision)  */
/*                               (Partial Pivoting)         */
/*                                                          */
/*                 Ver. 0.0 2015-02-23 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int DDLUdecompP(DDMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DDMatrix a: Matrix (given by user)                 */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
#ifdef __cplusplus
	static dd_real dtmp, dtmp1, dmaxii;
#else // __cplusplus
	static double dtmp[DDSIZE], dtmp1[DDSIZE], dmaxii[DDSIZE];
#endif // __cplusplus

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		rdd_abs(dmaxii, get_ddmatrix_ij(a, ch[i], i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rdd_abs(dtmp, get_ddmatrix_ij(a, ch[j], i));
			if(rdd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rdd_set(dmaxii, dtmp);
			}
		}

		if(rdd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! DDLUdecompP!\n", i);
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
			rdd_div(dtmp, get_ddmatrix_ij(a, ch[j], i), get_ddmatrix_ij(a, ch[i], i));
			set_ddmatrix_ij(a, ch[j], i, dtmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rdd_mul(dtmp1, get_ddmatrix_ij(a, ch[j], i), get_ddmatrix_ij(a, ch[i], k));
				rdd_sub(dtmp, get_ddmatrix_ij(a, ch[j], k), dtmp1);
				set_ddmatrix_ij(a, ch[j], k, dtmp);
			}
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                               (Double-Double Precision)  */
/*                               (Partial Pivoting)         */
/*                                                          */
/*                 Ver. 0.0 2011-11-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveDDLSP(DDVector answer, DDMatrix lu, DDVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      DDMatrix lu[]: LU decomposed Matrix (given by user) */
/*      DDVector b[]: constant vector (given by user)       */
/*      DDVector answer[]: Solution for linear system       */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
#ifdef __cplusplus
	static dd_real dtmp, dtmp1;
#else // __cplusplus
	static double dtmp[DDSIZE], dtmp1[DDSIZE];
#endif // __cplusplus

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_ddvector_i(answer, i, get_ddvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rdd_abs(dtmp, get_ddmatrix_ij(lu, ch[i], i));
		if(rdd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveDDLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rdd_mul(dtmp1, get_ddmatrix_ij(lu, ch[j], i), get_ddvector_i(answer, i));
			rdd_sub(dtmp, get_ddvector_i(answer, j), dtmp1);
			set_ddvector_i(answer, j, dtmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rdd_mul(dtmp1, get_ddmatrix_ij(lu, ch[i], j), get_ddvector_i(answer, j));
			rdd_sub(dtmp, get_ddvector_i(answer, i), dtmp1);
			set_ddvector_i(answer, i, dtmp);
		}
		rdd_div(dtmp, get_ddvector_i(answer, i), get_ddmatrix_ij(lu, ch[i], i));
		set_ddvector_i(answer, i, dtmp);
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                               (Double-Double Precision)  */
/*                               (Complete Pivoting)        */
/*                                                          */
/*                 Ver. 0.0 2015-02-23 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int DDLUdecompC(DDMatrix a, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DDMatrix a[]: Matrix (given by user)               */
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
#ifdef __cplusplus
	static dd_real dtmp, dtmp1, dmaxii;
#else // __cplusplus
	static double dtmp[DDSIZE], dtmp1[DDSIZE], dmaxii[DDSIZE];
#endif // __cplusplus

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
	{
		row_ch[i] = i;
		col_ch[i] = i;
	}

	for(i = 0; i < dim; i++)
	{
		/* Complete Pivoting */
		rdd_abs(dmaxii, get_ddmatrix_ij(a, row_ch[i], col_ch[i]));
		imax = i;
		jmax = i;
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rdd_abs(dtmp, get_ddmatrix_ij(a, row_ch[j], col_ch[k]));
				if(rdd_cmp(dtmp, dmaxii) > 0)
				{
					imax = j;
					jmax = k;
					rdd_set(dmaxii, dtmp);
				}
			}
		}

		if(rdd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (DDLUdecompC)!\n", i);
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
			rdd_div(dtmp, get_ddmatrix_ij(a, row_ch[j], col_ch[i]), get_ddmatrix_ij(a, row_ch[i], col_ch[i]));
			set_ddmatrix_ij(a, row_ch[j], col_ch[i], dtmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rdd_mul(dtmp1, get_ddmatrix_ij(a, row_ch[j], col_ch[i]), get_ddmatrix_ij(a, row_ch[i], col_ch[k]));
				rdd_sub(dtmp, get_ddmatrix_ij(a, row_ch[j], col_ch[k]), dtmp1);
				set_ddmatrix_ij(a, row_ch[j], col_ch[k], dtmp);
			}
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                               (Double-Double Precision)  */
/*                               (Complete Pivoting)        */
/*                                                          */
/*                 Ver. 0.0 2015-02-23 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveDDLSC(DDVector answer, DDMatrix lu, DDVector b, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DDMatrix lu: LU decomposed Matrix (given by user)  */
/*       DDVector b: constant vector (given by user)        */
/*       DDVector answer: Solution for linear system        */
/*       long int row_ch[]: Row order (given by user)       */
/*       long int col_ch[]: Column order (given by user)    */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
#ifdef __cplusplus
	static dd_real dtmp, dtmp1;
#else // __cplusplus
	static double dtmp[DDSIZE], dtmp1[DDSIZE];
#endif // __cplusplus

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_ddvector_i(answer, col_ch[i], get_ddvector_i(b, row_ch[i]));

/* Forward */
	for(i = 0; i < dim; i++)
	{
		rdd_abs(dtmp, get_ddmatrix_ij(lu, row_ch[i], col_ch[i]));
		if(rdd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveDDLSC, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rdd_mul(dtmp1, get_ddmatrix_ij(lu, row_ch[j], col_ch[i]), get_ddvector_i(answer, col_ch[i]));
			rdd_sub(dtmp, get_ddvector_i(answer, col_ch[j]), dtmp1);
			set_ddvector_i(answer, col_ch[j],  dtmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rdd_mul(dtmp1, get_ddmatrix_ij(lu, row_ch[i], col_ch[j]), get_ddvector_i(answer, col_ch[j]));
			rdd_sub(dtmp, get_ddvector_i(answer, col_ch[i]), dtmp1);
			set_ddvector_i(answer, col_ch[i], dtmp);
		}
		rdd_div(dtmp, get_ddvector_i(answer, col_ch[i]), get_ddmatrix_ij(lu, row_ch[i], col_ch[i]));
		set_ddvector_i(answer, col_ch[i], dtmp);
	}

	return 0;
}




/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                                 (Double Precision)       */
/*                                 (Complete Pivoting)      */
/*                                                          */
/*                 ver. 0.0 1997.10.24 (Fri) Tomonori Kouya */
/*                 ver. 0.1 2000.02.24 (Thu) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveDLSC(DVector answer, DMatrix lu, DVector b, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DMatrix lu: LU decomposed Matrix (given by user)   */
/*       DVector b: constant vector (given by user)         */
/*       DVector answer: Solution for linear system         */
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
		set_dvector_i(answer, col_ch[i], get_dvector_i(b, row_ch[i]));

/* Forward */
	for(i = 0; i < dim; i++)
	{
		if(get_dmatrix_ij(lu, row_ch[i], col_ch[i]) == 0.0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveDLSC, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
			set_dvector_i(answer, col_ch[j], get_dvector_i(answer, col_ch[j]) - get_dmatrix_ij(lu, row_ch[j], col_ch[i]) * get_dvector_i(answer, col_ch[i]));
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
			set_dvector_i(answer, col_ch[i], get_dvector_i(answer, col_ch[i]) - get_dmatrix_ij(lu, row_ch[i], col_ch[j]) * get_dvector_i(answer, col_ch[j]));
		set_dvector_i(answer, col_ch[i], get_dvector_i(answer, col_ch[i]) / get_dmatrix_ij(lu, row_ch[i], col_ch[i]));
	}

	return 0;
}


/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                          (Double double Precision)       */
/*          (Partial Pivoting with real swap of rows)       */
/*                                                          */
/*                 Ver. 0.0 2021-02-17 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int DDLUdecompPM(DDMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DDMatrix a: Matrix (given by user)                 */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
	static double dtmp[DDSIZE], dtmp1[DDSIZE], dmaxii[DDSIZE];

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		// partial pivoting
		rdd_abs(dmaxii, get_ddmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rdd_abs(dtmp, get_ddmatrix_ij(a, j, i));
			if(rdd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rdd_set(dmaxii, dtmp);
			}
		}

		if(rdd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! DDLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			row_swap_ddmatrix(a, i, imax, 0, a->col_dim);
		}

		for(j = (i + 1); j < dim; j++)
		{
			rdd_div(dtmp, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, i));
			set_ddmatrix_ij(a, j, i, dtmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rdd_mul(dtmp1, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(dtmp, get_ddmatrix_ij(a, j, k), dtmp1);
				set_ddmatrix_ij(a, j, k, dtmp);
			}
		}
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                (LU Decomposed Square Dense Matrix)       */
/*                          (Double double Precision)       */
/*          (Partial Pivoting with real swap of rows)       */
/*                                                          */
/*                 Ver. 0.0 2021-02-17 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveDDLSPM(DDVector answer, DDMatrix lu, DDVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      DDMatrix lu[]: LU decomposed Matrix (given by user) */
/*      DDVector b[]: constant vector (given by user)       */
/*      DDVector answer[]: Solution for linear system       */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static double dtmp[DDSIZE], dtmp1[DDSIZE];

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_ddvector_i(answer, i, get_ddvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rdd_abs(dtmp, get_ddmatrix_ij(lu, i, i));
		if(rdd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveDDLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rdd_mul(dtmp1, get_ddmatrix_ij(lu, j, i), get_ddvector_i(answer, i));
			rdd_sub(dtmp, get_ddvector_i(answer, j), dtmp1);
			set_ddvector_i(answer, j, dtmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rdd_mul(dtmp1, get_ddmatrix_ij(lu, i, j), get_ddvector_i(answer, j));
			rdd_sub(dtmp, get_ddvector_i(answer, i), dtmp1);
			set_ddvector_i(answer, i, dtmp);
		}
		rdd_div(dtmp, get_ddvector_i(answer, i), get_ddmatrix_ij(lu, i, i));
		set_ddvector_i(answer, i, dtmp);
	}

	return 0;
}