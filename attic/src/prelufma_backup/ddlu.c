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
// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m256d dtmp256[DDSIZE], aji256[DDSIZE], ajk256[DDSIZE], aik256[DDSIZE];
#elif defined(__AVX512F__) // __AVX512F__
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m512d dtmp512[DDSIZE], aji512[DDSIZE], ajk512[DDSIZE], aik512[DDSIZE];
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	
	svfloat64_t dtmp_neon_0, dtmp_neon_1;
	svfloat64_t aji_neon_0, aji_neon_1;
	svfloat64_t ajk_neon_0, ajk_neon_1;
	svfloat64_t aik_neon_0, aik_neon_1;
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON)  // ARM Neon
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	float64x2_t dtmp_neon[DDSIZE], aji_neon[DDSIZE], ajk_neon[DDSIZE], aik_neon[DDSIZE];
#endif // __AVX2__

	dim = a->col_dim;

	for(i = 0; i < a->row_dim; i++)
		ch[i] = i;

	for(i = 0; i < a->row_dim; i++)
	{
		// partial pivoting
		rdd_abs(dmaxii, get_ddmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < a->row_dim; j++)
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

			// head
			//printf("start j, k= %ld, %ld, ", j, i + 1);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			//for(k = (i + 1); k < dim; k++)
			{
				rdd_mul(dtmp1, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(dtmp, get_ddmatrix_ij(a, j, k), dtmp1);
				set_ddmatrix_ij(a, j, k, dtmp);
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
				_bncavx2_rdd_mul(dtmp256, aji256, aik256);
				//printf(" -- mul -- ");

				index_jk = j * a->real_col_dim + k;
				//rdd_sub(dtmp, get_ddmatrix_ij(a, j, k), dtmp1);
				ajk256[0] = _mm256_load_pd(&(a->element[0][index_jk]));
				ajk256[1] = _mm256_load_pd(&(a->element[1][index_jk]));
				_bncavx2_rdd_sub(dtmp256, ajk256, dtmp256);
				//printf(" -- sub -- ");

				//set_ddmatrix_ij(a, j, k, dtmp);
				_mm256_store_pd(&(a->element[0][index_jk]), dtmp256[0]);
				_mm256_store_pd(&(a->element[1][index_jk]), dtmp256[1]);
			}
			//printf(", %ld middle", k);
		}
#elif defined(__AVX512F__) // __AVX512F__
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->real_col_dim;

		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->real_col_dim + i;
			aji512[0] = _mm512_set1_pd(a->element[0][index_ji]);
			aji512[1] = _mm512_set1_pd(a->element[1][index_ji]);

			// head
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rdd_mul(dtmp1, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(dtmp, get_ddmatrix_ij(a, j, k), dtmp1);
				set_ddmatrix_ij(a, j, k, dtmp);
			}

			// middle : SIMD
			k_end = dim_end;
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				aik512[0] = _mm512_load_pd(&(a->element[0][index_ik]));
				aik512[1] = _mm512_load_pd(&(a->element[1][index_ik]));
				_bncavx512_rdd_mul(dtmp512, aji512, aik512);

				index_jk = j * a->real_col_dim + k;
				ajk512[0] = _mm512_load_pd(&(a->element[0][index_jk]));
				ajk512[1] = _mm512_load_pd(&(a->element[1][index_jk]));
				_bncavx512_rdd_sub(dtmp512, ajk512, dtmp512);

				_mm512_store_pd(&(a->element[0][index_jk]), dtmp512[0]);
				_mm512_store_pd(&(a->element[1][index_jk]), dtmp512[1]);
			}
		}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
		//dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_NEON_WIDTH) * _BNC_NEON_WIDTH;
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->real_col_dim;

		//printf("NEON: real_row_dim, real_col_dim, dim, i, dim_start, dim_end = %ld, %ld, %ld, %ld, %ld, %ld\n", a->real_row_dim, a->real_col_dim, dim, i, dim_start, dim_end);
		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->real_col_dim + i;
			// aji を全要素に複製 (broadcast)
			aji_neon_0 = svdup_f64(a->element[0][index_ji]);
			aji_neon_1 = svdup_f64(a->element[1][index_ji]);

			// head: アライメント境界までスカラー処理
			//printf("start j, k= %ld, %ld, ", j, i + 1);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rdd_mul(dtmp1, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(dtmp, get_ddmatrix_ij(a, j, k), dtmp1);
				set_ddmatrix_ij(a, j, k, dtmp);
			}
			//printf("head k_start, k = %ld, %ld, ", k_start, k);

			// middle : SIMD (Neon)
			k_end = dim_end;
			for(k = dim_start; k < dim_end; k += (long int)svcntd())
			{
		svbool_t pg = svwhilelt_b64_s64((int64_t)k, (int64_t)(dim_end));
				index_ik = i * a->real_col_dim + k;
				//rdd_mul(dtmp1, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				aik_neon_0 = svld1_f64(pg, &(a->element[0][index_ik]));
				aik_neon_1 = svld1_f64(pg, &(a->element[1][index_ik]));
				_bncsve2_rdd_mul(pg, &dtmp_neon_0, &dtmp_neon_1, aji_neon_0, aji_neon_1, aik_neon_0, aik_neon_1);
				//printf(" -- mul -- ");

				index_jk = j * a->real_col_dim + k;
				//rdd_sub(dtmp, get_ddmatrix_ij(a, j, k), dtmp1);
				ajk_neon_0 = svld1_f64(pg, &(a->element[0][index_jk]));
				ajk_neon_1 = svld1_f64(pg, &(a->element[1][index_jk]));
				_bncsve2_rdd_neg(pg, &dtmp_neon_0, &dtmp_neon_1, dtmp_neon_0, dtmp_neon_1);
		_bncsve2_rdd_add(pg, &dtmp_neon_0, &dtmp_neon_1, ajk_neon_0, ajk_neon_1, dtmp_neon_0, dtmp_neon_1);
				//printf(" -- sub -- ");

				//set_ddmatrix_ij(a, j, k, dtmp);
				svst1_f64(pg, &(a->element[0][index_jk]), dtmp_neon_0);
				svst1_f64(pg, &(a->element[1][index_jk]), dtmp_neon_1);
			}
			//printf(", %ld middle", k);
		}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
		//dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_NEON_WIDTH) * _BNC_NEON_WIDTH;
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->real_col_dim;

		//printf("NEON: real_row_dim, real_col_dim, dim, i, dim_start, dim_end = %ld, %ld, %ld, %ld, %ld, %ld\n", a->real_row_dim, a->real_col_dim, dim, i, dim_start, dim_end);
		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->real_col_dim + i;
			// aji を全要素に複製 (broadcast)
			aji_neon[0] = vdupq_n_f64(a->element[0][index_ji]);
			aji_neon[1] = vdupq_n_f64(a->element[1][index_ji]);

			// head: アライメント境界までスカラー処理
			//printf("start j, k= %ld, %ld, ", j, i + 1);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rdd_mul(dtmp1, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(dtmp, get_ddmatrix_ij(a, j, k), dtmp1);
				set_ddmatrix_ij(a, j, k, dtmp);
			}
			//printf("head k_start, k = %ld, %ld, ", k_start, k);

			// middle : SIMD (Neon)
			k_end = dim_end;
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				//rdd_mul(dtmp1, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				aik_neon[0] = vld1q_f64(&(a->element[0][index_ik]));
				aik_neon[1] = vld1q_f64(&(a->element[1][index_ik]));
				_bncneon_rdd_mul(dtmp_neon, aji_neon, aik_neon);
				//printf(" -- mul -- ");

				index_jk = j * a->real_col_dim + k;
				//rdd_sub(dtmp, get_ddmatrix_ij(a, j, k), dtmp1);
				ajk_neon[0] = vld1q_f64(&(a->element[0][index_jk]));
				ajk_neon[1] = vld1q_f64(&(a->element[1][index_jk]));
				_bncneon_rdd_sub(dtmp_neon, ajk_neon, dtmp_neon);
				//printf(" -- sub -- ");

				//set_ddmatrix_ij(a, j, k, dtmp);
				vst1q_f64(&(a->element[0][index_jk]), dtmp_neon[0]);
				vst1q_f64(&(a->element[1][index_jk]), dtmp_neon[1]);
			}
			//printf(", %ld middle", k);
		}
#else // others
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rdd_mul(dtmp1, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(dtmp, get_ddmatrix_ij(a, j, k), dtmp1);
				set_ddmatrix_ij(a, j, k, dtmp);
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

/* Backward */
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