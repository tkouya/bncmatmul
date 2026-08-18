
/********************************************************************************/
/* tdlu.c:                                                                      */
/* Copyright (C) 2015-2023 Tomonori Kouya                                       */
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
#include "tdlinear.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// TD
// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_tdmatrix(TDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
	long int i, j, true_end;
	double tmp[TDSIZE];

	true_end = (col_end > mat->col_dim) ? mat->col_dim : col_end;

	for(i = col_start; i < true_end; i++)
	{
		rtd_set(tmp, get_tdmatrix_ij(mat, row_index0, i));
		set_tdmatrix_ij(mat, row_index0, i, get_tdmatrix_ij(mat, row_index1, i));
		set_tdmatrix_ij(mat, row_index1, i, tmp);
	}
}


/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Octuple double Precision)       */
/*                                                          */
/*                 Ver. 0.0 2015-02-23 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int TDLUdecomp(TDMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       TDMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	static double dtmp[TDSIZE], dtmp1[TDSIZE], dmaxii[TDSIZE];

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
	{
		rtd_abs(dmaxii, get_tdmatrix_ij(a, i, i));
		//printf("a%ld_%ld = ", i, i); rtd_out_str(dmaxii); printf("\n");
		if(rtd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (TDLUdecomp)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rtd_div(dtmp, get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, i));
			set_tdmatrix_ij(a, j, i, dtmp);
			//printf("a%ld_%ld = ", j, i); rtd_out_str(dtmp); printf("\n");
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rtd_mul(dtmp1, get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, k));
				rtd_sub(dtmp, get_tdmatrix_ij(a, j, k), dtmp1);
				set_tdmatrix_ij(a, j, k, dtmp);
				//printf("a%ld_%ld= ", j, k); rtd_out_str(dtmp); printf("\n");
			}
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                         (Octuple double Precision)       */
/*                                                          */
/*                 Ver. 0.0 2011-11-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveTDLS(TDVector answer, TDMatrix lu, TDVector b)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      TDMatrix lu: LU decomposed Matrix (given by user)   */
/*      TDVector b: constant vector (given by user)         */
/*      TDVector answer: Solution for linear system         */
/*      long int dim: Dimension of Matrix (given by user)   */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static double dtmp[TDSIZE], dtmp1[TDSIZE];

	dim = answer->dim;

	subst_tdvector(answer, b);

/* Forward */
	for(i = 0; i < dim; i++)
	{
		//printf("f %ld = ", i); rtd_out_str(get_tdvector_i(answer, i)); printf("\n");
		rtd_abs(dtmp, get_tdmatrix_ij(lu, i, i));
		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveTDLS, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rtd_mul(dtmp1, get_tdmatrix_ij(lu, j, i), get_tdvector_i(answer, i));
			rtd_sub(dtmp, get_tdvector_i(answer, j), dtmp1);
			set_tdvector_i(answer, j, dtmp);
		}
		//printf("f %ld = ", i); rtd_out_str(get_tdvector_i(answer, i)); printf("\n");
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rtd_mul(dtmp1, get_tdmatrix_ij(lu, i, j), get_tdvector_i(answer, j));
			rtd_sub(dtmp, get_tdvector_i(answer, i), dtmp1);
			set_tdvector_i(answer, i, dtmp);
		}
		rtd_div(dtmp, get_tdvector_i(answer, i), get_tdmatrix_ij(lu, i, i));
		set_tdvector_i(answer, i, dtmp);
		//printf("b %ld = ", i); rtd_out_str(get_tdvector_i(answer, i)); printf("\n");
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Octuple double Precision)       */
/*                               (Partial Pivoting)         */
/*                                                          */
/*                 Ver. 0.0 2015-02-23 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int TDLUdecompP(TDMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       TDMatrix a: Matrix (given by user)                 */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
	static double dtmp[TDSIZE], dtmp1[TDSIZE], dmaxii[TDSIZE];

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		rtd_abs(dmaxii, get_tdmatrix_ij(a, ch[i], i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rtd_abs(dtmp, get_tdmatrix_ij(a, ch[j], i));
			if(rtd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rtd_set(dmaxii, dtmp);
			}
		}

		if(rtd_cmp_ui(dmaxii, 0UL) == 0)
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
			rtd_div(dtmp, get_tdmatrix_ij(a, ch[j], i), get_tdmatrix_ij(a, ch[i], i));
			set_tdmatrix_ij(a, ch[j], i, dtmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rtd_mul(dtmp1, get_tdmatrix_ij(a, ch[j], i), get_tdmatrix_ij(a, ch[i], k));
				rtd_sub(dtmp, get_tdmatrix_ij(a, ch[j], k), dtmp1);
				set_tdmatrix_ij(a, ch[j], k, dtmp);
			}
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                         (Octuple double Precision)       */
/*                               (Partial Pivoting)         */
/*                                                          */
/*                 Ver. 0.0 2011-11-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveTDLSP(TDVector answer, TDMatrix lu, TDVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      TDMatrix lu[]: LU decomposed Matrix (given by user) */
/*      TDVector b[]: constant vector (given by user)       */
/*      TDVector answer[]: Solution for linear system       */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static double dtmp[TDSIZE], dtmp1[TDSIZE];

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_tdvector_i(answer, i, get_tdvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rtd_abs(dtmp, get_tdmatrix_ij(lu, ch[i], i));
		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveTDLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rtd_mul(dtmp1, get_tdmatrix_ij(lu, ch[j], i), get_tdvector_i(answer, i));
			rtd_sub(dtmp, get_tdvector_i(answer, j), dtmp1);
			set_tdvector_i(answer, j, dtmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rtd_mul(dtmp1, get_tdmatrix_ij(lu, ch[i], j), get_tdvector_i(answer, j));
			rtd_sub(dtmp, get_tdvector_i(answer, i), dtmp1);
			set_tdvector_i(answer, i, dtmp);
		}
		rtd_div(dtmp, get_tdvector_i(answer, i), get_tdmatrix_ij(lu, ch[i], i));
		set_tdvector_i(answer, i, dtmp);
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Octuple double Precision)       */
/*                               (Complete Pivoting)        */
/*                                                          */
/*                 Ver. 0.0 2015-02-23 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int TDLUdecompC(TDMatrix a, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       TDMatrix a[]: Matrix (given by user)               */
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
	static double dtmp[TDSIZE], dtmp1[TDSIZE], dmaxii[TDSIZE];

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
	{
		row_ch[i] = i;
		col_ch[i] = i;
	}

	for(i = 0; i < dim; i++)
	{
		/* Complete Pivoting */
		rtd_abs(dmaxii, get_tdmatrix_ij(a, row_ch[i], col_ch[i]));
		imax = i;
		jmax = i;
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rtd_abs(dtmp, get_tdmatrix_ij(a, row_ch[j], col_ch[k]));
				if(rtd_cmp(dtmp, dmaxii) > 0)
				{
					imax = j;
					jmax = k;
					rtd_set(dmaxii, dtmp);
				}
			}
		}

		if(rtd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (TDLUdecompC)!\n", i);
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
			rtd_div(dtmp, get_tdmatrix_ij(a, row_ch[j], col_ch[i]), get_tdmatrix_ij(a, row_ch[i], col_ch[i]));
			set_tdmatrix_ij(a, row_ch[j], col_ch[i], dtmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rtd_mul(dtmp1, get_tdmatrix_ij(a, row_ch[j], col_ch[i]), get_tdmatrix_ij(a, row_ch[i], col_ch[k]));
				rtd_sub(dtmp, get_tdmatrix_ij(a, row_ch[j], col_ch[k]), dtmp1);
				set_tdmatrix_ij(a, row_ch[j], col_ch[k], dtmp);
			}
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                         (Octuple double Precision)       */
/*                               (Complete Pivoting)        */
/*                                                          */
/*                 Ver. 0.0 2015-02-23 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveTDLSC(TDVector answer, TDMatrix lu, TDVector b, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       TDMatrix lu: LU decomposed Matrix (given by user)  */
/*       TDVector b: constant vector (given by user)        */
/*       TDVector answer: Solution for linear system        */
/*       long int row_ch[]: Row order (given by user)       */
/*       long int col_ch[]: Column order (given by user)    */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static double dtmp[TDSIZE], dtmp1[TDSIZE];

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_tdvector_i(answer, col_ch[i], get_tdvector_i(b, row_ch[i]));

/* Forward */
	for(i = 0; i < dim; i++)
	{
		rtd_abs(dtmp, get_tdmatrix_ij(lu, row_ch[i], col_ch[i]));
		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveTDLSC, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rtd_mul(dtmp1, get_tdmatrix_ij(lu, row_ch[j], col_ch[i]), get_tdvector_i(answer, col_ch[i]));
			rtd_sub(dtmp, get_tdvector_i(answer, col_ch[j]), dtmp1);
			set_tdvector_i(answer, col_ch[j],  dtmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rtd_mul(dtmp1, get_tdmatrix_ij(lu, row_ch[i], col_ch[j]), get_tdvector_i(answer, col_ch[j]));
			rtd_sub(dtmp, get_tdvector_i(answer, col_ch[i]), dtmp1);
			set_tdvector_i(answer, col_ch[i], dtmp);
		}
		rtd_div(dtmp, get_tdvector_i(answer, col_ch[i]), get_tdmatrix_ij(lu, row_ch[i], col_ch[i]));
		set_tdvector_i(answer, col_ch[i], dtmp);
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                          (triple double Precision)       */
/*          (Partial Pivoting with real swap of rows)       */
/*                                                          */
/*                 Ver. 0.0 2021-02-17 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int TDLUdecompPM(TDMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       TDMatrix a: Matrix (given by user)                 */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
	static double dtmp[TDSIZE], dtmp1[TDSIZE], dmaxii[TDSIZE];
// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m256d dtmp256[TDSIZE], aji256[TDSIZE], ajk256[TDSIZE], aik256[TDSIZE];
#elif defined(__AVX512F__) // __AVX512F__
#else // normal
#endif // __AVX2__


#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	printf("TDLUdecompPM AVX2 enabled!\n");
#elif defined(__AVX512F__) // __AVX512F__
	printf("TDLUdecompPM AVX-512 enabled!\n");
#else
	printf("TDLUdecompPM normal!\n");
#endif // __AVX2__

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		// partial pivoting
		rtd_abs(dmaxii, get_tdmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rtd_abs(dtmp, get_tdmatrix_ij(a, j, i));
			if(rtd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rtd_set(dmaxii, dtmp);
			}
		}

		if(rtd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! TDLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			row_swap_tdmatrix(a, i, imax, 0, a->col_dim);
		}

		for(j = (i + 1); j < dim; j++)
		{
			rtd_div(dtmp, get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, i));
			set_tdmatrix_ij(a, j, i, dtmp);
		}
// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->real_col_dim;

		//printf("real_row_dim, real_col_dim, dim, i, dim_start, dim_end = %ld, %ld, %ld, %ld, %ld, %ld\n", a->real_row_dim, a->real_col_dim, dim, i, dim_start, dim_end);
		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->real_col_dim + i;
			aji256[0] = _mm256_set_pd(
                a->element[0][index_ji],
                a->element[0][index_ji],
                a->element[0][index_ji],
                a->element[0][index_ji]
            );
			aji256[1] = _mm256_set_pd(
                a->element[1][index_ji],
                a->element[1][index_ji],
                a->element[1][index_ji],
                a->element[1][index_ji]
            );
			aji256[2] = _mm256_set_pd(
                a->element[2][index_ji],
                a->element[2][index_ji],
                a->element[2][index_ji],
                a->element[2][index_ji]
            );

			// head
			//printf("start j, k= %ld, %ld, ", j, i + 1);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			//for(k = (i + 1); k < dim; k++)
			{
				rtd_mul(dtmp1, get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, k));
				rtd_sub(dtmp, get_tdmatrix_ij(a, j, k), dtmp1);
				set_tdmatrix_ij(a, j, k, dtmp);
			}
			//printf("head k_start, k = %ld, %ld, ", k_start, k);
//#if 0
			// middle : SIMD
			k_end = dim_end;
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				//rdd_mul(dtmp1, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				aik256[0] = _mm256_load_pd(&(a->element[0][index_ik]));
				aik256[1] = _mm256_load_pd(&(a->element[1][index_ik]));
				aik256[2] = _mm256_load_pd(&(a->element[2][index_ik]));				
				_bncavx2_rtd_mul(dtmp256, aji256, aik256);
				//printf(" -- mul -- ");

				index_jk = j * a->real_col_dim + k;
				//rdd_sub(dtmp, get_ddmatrix_ij(a, j, k), dtmp1);
				ajk256[0] = _mm256_load_pd(&(a->element[0][index_jk]));
				ajk256[1] = _mm256_load_pd(&(a->element[1][index_jk]));
				ajk256[2] = _mm256_load_pd(&(a->element[2][index_jk]));
				_bncavx2_rtd_sub(dtmp256, ajk256, dtmp256);
				//printf(" -- sub -- ");

				//set_ddmatrix_ij(a, j, k, dtmp);
				_mm256_store_pd(&(a->element[0][index_jk]), dtmp256[0]);
				_mm256_store_pd(&(a->element[1][index_jk]), dtmp256[1]);
				_mm256_store_pd(&(a->element[2][index_jk]), dtmp256[2]);
			}
			//printf(", %ld middle", k);
		}
#elif defined(__AVX512F__) // __AVX512F__
#else // others
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rtd_mul(dtmp1, get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, k));
				rtd_sub(dtmp, get_tdmatrix_ij(a, j, k), dtmp1);
				set_tdmatrix_ij(a, j, k, dtmp);
			}
		}
#endif // __AVX2__
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                (LU Decomposed Square Dense Matrix)       */
/*                         (Octuple double Precision)       */
/*          (Partial Pivoting with real swap of rows)       */
/*                                                          */
/*                 Ver. 0.0 2021-02-17 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveTDLSPM(TDVector answer, TDMatrix lu, TDVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      TDMatrix lu[]: LU decomposed Matrix (given by user) */
/*      TDVector b[]: constant vector (given by user)       */
/*      TDVector answer[]: Solution for linear system       */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static double dtmp[TDSIZE], dtmp1[TDSIZE];

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_tdvector_i(answer, i, get_tdvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rtd_abs(dtmp, get_tdmatrix_ij(lu, i, i));
		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveTDLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rtd_mul(dtmp1, get_tdmatrix_ij(lu, j, i), get_tdvector_i(answer, i));
			rtd_sub(dtmp, get_tdvector_i(answer, j), dtmp1);
			set_tdvector_i(answer, j, dtmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rtd_mul(dtmp1, get_tdmatrix_ij(lu, i, j), get_tdvector_i(answer, j));
			rtd_sub(dtmp, get_tdvector_i(answer, i), dtmp1);
			set_tdvector_i(answer, i, dtmp);
		}
		rtd_div(dtmp, get_tdvector_i(answer, i), get_tdmatrix_ij(lu, i, i));
		set_tdvector_i(answer, i, dtmp);
	}

	return 0;
}

#ifdef __cplusplus
} // extern "C"
#endif