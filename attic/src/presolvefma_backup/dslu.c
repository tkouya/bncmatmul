/********************************************************************************/
/* dslu.c: LU decomposition and solvers for DS (pair of floats) precision       */
/* Copyright (C) 2015-2026 Tomonori Kouya                                       */
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

#include "dslinear.h"

// DS

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                        (Double-Single Precision)         */
/*                                                          */
/************************************************************/
int DSLUdecomp(DSMatrix a)
{
	long int i, j, k, dim;
	static float dtmp[DSSIZE], dtmp1[DSSIZE], dmaxii[DSSIZE];
#ifdef BNC_USE_NEW_FMA
	static float neg_aji[DSSIZE];
#endif // BNC_USE_NEW_FMA

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
	{
		rds_abs(dmaxii, get_dsmatrix_ij(a, i, i));
		if(rds_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (DSLUdecomp)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rds_div(dtmp, get_dsmatrix_ij(a, j, i), get_dsmatrix_ij(a, i, i));
			set_dsmatrix_ij(a, j, i, dtmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
#ifdef BNC_USE_NEW_FMA
			rds_neg(neg_aji, get_dsmatrix_ij(a, j, i));
#endif // BNC_USE_NEW_FMA
			for(k = (i + 1); k < dim; k++)
			{
#ifdef BNC_USE_NEW_FMA
				rds_fma(dtmp, neg_aji, get_dsmatrix_ij(a, i, k), get_dsmatrix_ij(a, j, k));
#else // BNC_USE_NEW_FMA
				rds_mul(dtmp1, get_dsmatrix_ij(a, j, i), get_dsmatrix_ij(a, i, k));
				rds_sub(dtmp, get_dsmatrix_ij(a, j, k), dtmp1);
#endif // BNC_USE_NEW_FMA
				set_dsmatrix_ij(a, j, k, dtmp);
			}
		}
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                        (Double-Single Precision)         */
/*                        (Partial Pivoting)                */
/*                                                          */
/************************************************************/
int DSLUdecompP(DSMatrix a, long int ch[])
{
	long int i, j, k, imax, itmp, dim;
	static float dtmp[DSSIZE], dtmp1[DSSIZE], dmaxii[DSSIZE];
#ifdef BNC_USE_NEW_FMA
	static float neg_aji[DSSIZE];
#endif // BNC_USE_NEW_FMA

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		rds_abs(dmaxii, get_dsmatrix_ij(a, ch[i], i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rds_abs(dtmp, get_dsmatrix_ij(a, ch[j], i));
			if(rds_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rds_set(dmaxii, dtmp);
			}
		}

		if(rds_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! DSLUdecompP!\n", i);
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
			rds_div(dtmp, get_dsmatrix_ij(a, ch[j], i), get_dsmatrix_ij(a, ch[i], i));
			set_dsmatrix_ij(a, ch[j], i, dtmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
#ifdef BNC_USE_NEW_FMA
			rds_neg(neg_aji, get_dsmatrix_ij(a, ch[j], i));
#endif // BNC_USE_NEW_FMA
			for(k = (i + 1); k < dim; k++)
			{
#ifdef BNC_USE_NEW_FMA
				rds_fma(dtmp, neg_aji, get_dsmatrix_ij(a, ch[i], k), get_dsmatrix_ij(a, ch[j], k));
#else // BNC_USE_NEW_FMA
				rds_mul(dtmp1, get_dsmatrix_ij(a, ch[j], i), get_dsmatrix_ij(a, ch[i], k));
				rds_sub(dtmp, get_dsmatrix_ij(a, ch[j], k), dtmp1);
#endif // BNC_USE_NEW_FMA
				set_dsmatrix_ij(a, ch[j], k, dtmp);
			}
		}
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                        (Double-Single Precision)         */
/*                        (Complete Pivoting)               */
/*                                                          */
/************************************************************/
int DSLUdecompC(DSMatrix a, long int row_ch[], long int col_ch[])
{
	long int i, j, k, imax, jmax, itmp, dim;
	static float dtmp[DSSIZE], dtmp1[DSSIZE], dmaxii[DSSIZE];
#ifdef BNC_USE_NEW_FMA
	static float neg_aji[DSSIZE];
#endif // BNC_USE_NEW_FMA

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
	{
		row_ch[i] = i;
		col_ch[i] = i;
	}

	for(i = 0; i < dim; i++)
	{
		/* Complete Pivoting */
		rds_abs(dmaxii, get_dsmatrix_ij(a, row_ch[i], col_ch[i]));
		imax = i;
		jmax = i;
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rds_abs(dtmp, get_dsmatrix_ij(a, row_ch[j], col_ch[k]));
				if(rds_cmp(dtmp, dmaxii) > 0)
				{
					imax = j;
					jmax = k;
					rds_set(dmaxii, dtmp);
				}
			}
		}

		if(rds_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (DSLUdecompC)!\n", i);
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
			rds_div(dtmp, get_dsmatrix_ij(a, row_ch[j], col_ch[i]), get_dsmatrix_ij(a, row_ch[i], col_ch[i]));
			set_dsmatrix_ij(a, row_ch[j], col_ch[i], dtmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
#ifdef BNC_USE_NEW_FMA
			rds_neg(neg_aji, get_dsmatrix_ij(a, row_ch[j], col_ch[i]));
#endif // BNC_USE_NEW_FMA
			for(k = (i + 1); k < dim; k++)
			{
#ifdef BNC_USE_NEW_FMA
				rds_fma(dtmp, neg_aji, get_dsmatrix_ij(a, row_ch[i], col_ch[k]), get_dsmatrix_ij(a, row_ch[j], col_ch[k]));
#else // BNC_USE_NEW_FMA
				rds_mul(dtmp1, get_dsmatrix_ij(a, row_ch[j], col_ch[i]), get_dsmatrix_ij(a, row_ch[i], col_ch[k]));
				rds_sub(dtmp, get_dsmatrix_ij(a, row_ch[j], col_ch[k]), dtmp1);
#endif // BNC_USE_NEW_FMA
				set_dsmatrix_ij(a, row_ch[j], col_ch[k], dtmp);
			}
		}
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                        (Double-Single Precision)         */
/*          (Partial Pivoting with real swap of rows)       */
/*                                                          */
/************************************************************/
int DSLUdecompPM(DSMatrix a, long int ch[])
{
	long int i, j, k, imax, itmp, dim;
	static float dtmp[DSSIZE], dtmp1[DSSIZE], dmaxii[DSSIZE];
#ifdef BNC_USE_NEW_FMA
	float neg_aji[DSSIZE];
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m256 dtmp256[DSSIZE], aji256[DSSIZE], ajk256[DSSIZE], aik256[DSSIZE];
#elif defined(__AVX512F__) // __AVX512F__
	long int k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m512 dtmp512[DSSIZE], aji512[DSSIZE], ajk512[DSSIZE], aik512[DSSIZE];
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	long int k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	svfloat32_t dtmp_neon_0, dtmp_neon_1;
	svfloat32_t aji_neon_0, aji_neon_1;
	svfloat32_t ajk_neon_0, ajk_neon_1;
	svfloat32_t aik_neon_0, aik_neon_1;
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	long int k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	float32x4_t dtmp_neon[DSSIZE], aji_neon[DSSIZE], ajk_neon[DSSIZE], aik_neon[DSSIZE];
#else // others
#endif // __AVX2__
#endif // BNC_USE_NEW_FMA

	dim = a->col_dim;

	for(i = 0; i < a->row_dim; i++)
		ch[i] = i;

	for(i = 0; i < a->row_dim; i++)
	{
		// partial pivoting
		rds_abs(dmaxii, get_dsmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < a->row_dim; j++)
		{
			rds_abs(dtmp, get_dsmatrix_ij(a, j, i));
			if(rds_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rds_set(dmaxii, dtmp);
			}
		}

		if(rds_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! DSLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			row_swap_dsmatrix(a, i, imax, 0, a->col_dim);
		}

		for(j = (i + 1); j < dim; j++)
		{
			rds_div(dtmp, get_dsmatrix_ij(a, j, i), get_dsmatrix_ij(a, i, i));
			set_dsmatrix_ij(a, j, i, dtmp);
		}
#ifdef BNC_USE_NEW_FMA
// Fused update: a_jk := (-a_ji) * a_ik + a_jk in one branch-free FMA (arXiv:2607.11391)
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_S_WIDTH) * _BNC_S_WIDTH;
		dim_end = a->real_col_dim;

		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->real_col_dim + i;
			neg_aji[0] = -(a->element[0][index_ji]);
			neg_aji[1] = -(a->element[1][index_ji]);
			aji256[0] = _mm256_set1_ps(neg_aji[0]);
			aji256[1] = _mm256_set1_ps(neg_aji[1]);

			// head
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rds_fma(dtmp, neg_aji, get_dsmatrix_ij(a, i, k), get_dsmatrix_ij(a, j, k));
				set_dsmatrix_ij(a, j, k, dtmp);
			}

			// middle : SIMD
			for(k = dim_start; k < dim_end; k += _BNC_S_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				index_jk = j * a->real_col_dim + k;
				aik256[0] = _mm256_load_ps(&(a->element[0][index_ik]));
				aik256[1] = _mm256_load_ps(&(a->element[1][index_ik]));
				ajk256[0] = _mm256_load_ps(&(a->element[0][index_jk]));
				ajk256[1] = _mm256_load_ps(&(a->element[1][index_jk]));
				_bncavx2_dwfmaf(dtmp256, aji256, aik256, ajk256);
				_mm256_store_ps(&(a->element[0][index_jk]), dtmp256[0]);
				_mm256_store_ps(&(a->element[1][index_jk]), dtmp256[1]);
			}
		}
#elif defined(__AVX512F__) // __AVX512F__
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_S_WIDTH) * _BNC_S_WIDTH;
		dim_end = a->real_col_dim;

		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->real_col_dim + i;
			neg_aji[0] = -(a->element[0][index_ji]);
			neg_aji[1] = -(a->element[1][index_ji]);
			aji512[0] = _mm512_set1_ps(neg_aji[0]);
			aji512[1] = _mm512_set1_ps(neg_aji[1]);

			// head
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rds_fma(dtmp, neg_aji, get_dsmatrix_ij(a, i, k), get_dsmatrix_ij(a, j, k));
				set_dsmatrix_ij(a, j, k, dtmp);
			}

			// middle : SIMD
			for(k = dim_start; k < dim_end; k += _BNC_S_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				index_jk = j * a->real_col_dim + k;
				aik512[0] = _mm512_load_ps(&(a->element[0][index_ik]));
				aik512[1] = _mm512_load_ps(&(a->element[1][index_ik]));
				ajk512[0] = _mm512_load_ps(&(a->element[0][index_jk]));
				ajk512[1] = _mm512_load_ps(&(a->element[1][index_jk]));
				_bncavx512_dwfmaf(dtmp512, aji512, aik512, ajk512);
				_mm512_store_ps(&(a->element[0][index_jk]), dtmp512[0]);
				_mm512_store_ps(&(a->element[1][index_jk]), dtmp512[1]);
			}
		}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_S_WIDTH) * _BNC_S_WIDTH;
		dim_end = a->real_col_dim;

		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->real_col_dim + i;
			neg_aji[0] = -(a->element[0][index_ji]);
			neg_aji[1] = -(a->element[1][index_ji]);
			aji_neon_0 = svdup_f32(neg_aji[0]);
			aji_neon_1 = svdup_f32(neg_aji[1]);

			// head
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rds_fma(dtmp, neg_aji, get_dsmatrix_ij(a, i, k), get_dsmatrix_ij(a, j, k));
				set_dsmatrix_ij(a, j, k, dtmp);
			}

			// middle : SIMD (SVE2)
			for(k = dim_start; k < dim_end; k += (long int)svcntw())
			{
				svbool_t pg = svwhilelt_b32_s64((int64_t)k, (int64_t)(dim_end));
				index_ik = i * a->real_col_dim + k;
				index_jk = j * a->real_col_dim + k;
				aik_neon_0 = svld1_f32(pg, &(a->element[0][index_ik]));
				aik_neon_1 = svld1_f32(pg, &(a->element[1][index_ik]));
				ajk_neon_0 = svld1_f32(pg, &(a->element[0][index_jk]));
				ajk_neon_1 = svld1_f32(pg, &(a->element[1][index_jk]));
				_bncsve2_dwfmaf(pg, &dtmp_neon_0, &dtmp_neon_1, aji_neon_0, aji_neon_1, aik_neon_0, aik_neon_1, ajk_neon_0, ajk_neon_1);
				svst1_f32(pg, &(a->element[0][index_jk]), dtmp_neon_0);
				svst1_f32(pg, &(a->element[1][index_jk]), dtmp_neon_1);
			}
		}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_S_WIDTH) * _BNC_S_WIDTH;
		dim_end = a->real_col_dim;

		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->real_col_dim + i;
			neg_aji[0] = -(a->element[0][index_ji]);
			neg_aji[1] = -(a->element[1][index_ji]);
			aji_neon[0] = vdupq_n_f32(neg_aji[0]);
			aji_neon[1] = vdupq_n_f32(neg_aji[1]);

			// head
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rds_fma(dtmp, neg_aji, get_dsmatrix_ij(a, i, k), get_dsmatrix_ij(a, j, k));
				set_dsmatrix_ij(a, j, k, dtmp);
			}

			// middle : SIMD (Neon)
			for(k = dim_start; k < dim_end; k += _BNC_S_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				index_jk = j * a->real_col_dim + k;
				aik_neon[0] = vld1q_f32(&(a->element[0][index_ik]));
				aik_neon[1] = vld1q_f32(&(a->element[1][index_ik]));
				ajk_neon[0] = vld1q_f32(&(a->element[0][index_jk]));
				ajk_neon[1] = vld1q_f32(&(a->element[1][index_jk]));
				_bncneon_dwfmaf(dtmp_neon, aji_neon, aik_neon, ajk_neon);
				vst1q_f32(&(a->element[0][index_jk]), dtmp_neon[0]);
				vst1q_f32(&(a->element[1][index_jk]), dtmp_neon[1]);
			}
		}
#else // others
		for(j = (i + 1); j < dim; j++)
		{
			rds_neg(neg_aji, get_dsmatrix_ij(a, j, i));
			for(k = (i + 1); k < dim; k++)
			{
				rds_fma(dtmp, neg_aji, get_dsmatrix_ij(a, i, k), get_dsmatrix_ij(a, j, k));
				set_dsmatrix_ij(a, j, k, dtmp);
			}
		}
#endif // __AVX2__
#else // BNC_USE_NEW_FMA
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rds_mul(dtmp1, get_dsmatrix_ij(a, j, i), get_dsmatrix_ij(a, i, k));
				rds_sub(dtmp, get_dsmatrix_ij(a, j, k), dtmp1);
				set_dsmatrix_ij(a, j, k, dtmp);
			}
		}
#endif // BNC_USE_NEW_FMA
	}

	return 0;
}

/************************************************************/
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                        (Double-Single Precision)         */
/************************************************************/
int SolveDSLS(DSVector answer, DSMatrix lu, DSVector b)
{
	long int i, j, dim;
	static float dtmp[DSSIZE], dtmp1[DSSIZE];

	dim = answer->dim;

	subst_dsvector(answer, b);

/* Forward */
	for(i = 0; i < dim; i++)
	{
		rds_abs(dtmp, get_dsmatrix_ij(lu, i, i));
		if(rds_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveDSLS, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rds_mul(dtmp1, get_dsmatrix_ij(lu, j, i), get_dsvector_i(answer, i));
			rds_sub(dtmp, get_dsvector_i(answer, j), dtmp1);
			set_dsvector_i(answer, j, dtmp);
		}
	}

/* Backward */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rds_mul(dtmp1, get_dsmatrix_ij(lu, i, j), get_dsvector_i(answer, j));
			rds_sub(dtmp, get_dsvector_i(answer, i), dtmp1);
			set_dsvector_i(answer, i, dtmp);
		}
		rds_div(dtmp, get_dsvector_i(answer, i), get_dsmatrix_ij(lu, i, i));
		set_dsvector_i(answer, i, dtmp);
	}

	return 0;
}

/************************************************************/
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                        (Double-Single Precision)         */
/*                        (Partial Pivoting)                */
/************************************************************/
int SolveDSLSP(DSVector answer, DSMatrix lu, DSVector b, long int ch[])
{
	long int i, j, dim;
	static float dtmp[DSSIZE], dtmp1[DSSIZE];

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_dsvector_i(answer, i, get_dsvector_i(b, ch[i]));

/* Forward */
	for(i = 0; i < dim; i++)
	{
		rds_abs(dtmp, get_dsmatrix_ij(lu, ch[i], i));
		if(rds_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveDSLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rds_mul(dtmp1, get_dsmatrix_ij(lu, ch[j], i), get_dsvector_i(answer, i));
			rds_sub(dtmp, get_dsvector_i(answer, j), dtmp1);
			set_dsvector_i(answer, j, dtmp);
		}
	}

/* Backward */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rds_mul(dtmp1, get_dsmatrix_ij(lu, ch[i], j), get_dsvector_i(answer, j));
			rds_sub(dtmp, get_dsvector_i(answer, i), dtmp1);
			set_dsvector_i(answer, i, dtmp);
		}
		rds_div(dtmp, get_dsvector_i(answer, i), get_dsmatrix_ij(lu, ch[i], i));
		set_dsvector_i(answer, i, dtmp);
	}

	return 0;
}

/************************************************************/
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                        (Double-Single Precision)         */
/*                        (Complete Pivoting)               */
/************************************************************/
int SolveDSLSC(DSVector answer, DSMatrix lu, DSVector b, long int row_ch[], long int col_ch[])
{
	long int i, j, dim;
	static float dtmp[DSSIZE], dtmp1[DSSIZE];

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_dsvector_i(answer, col_ch[i], get_dsvector_i(b, row_ch[i]));

/* Forward */
	for(i = 0; i < dim; i++)
	{
		rds_abs(dtmp, get_dsmatrix_ij(lu, row_ch[i], col_ch[i]));
		if(rds_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveDSLSC, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rds_mul(dtmp1, get_dsmatrix_ij(lu, row_ch[j], col_ch[i]), get_dsvector_i(answer, col_ch[i]));
			rds_sub(dtmp, get_dsvector_i(answer, col_ch[j]), dtmp1);
			set_dsvector_i(answer, col_ch[j], dtmp);
		}
	}

/* Backward */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rds_mul(dtmp1, get_dsmatrix_ij(lu, row_ch[i], col_ch[j]), get_dsvector_i(answer, col_ch[j]));
			rds_sub(dtmp, get_dsvector_i(answer, col_ch[i]), dtmp1);
			set_dsvector_i(answer, col_ch[i], dtmp);
		}
		rds_div(dtmp, get_dsvector_i(answer, col_ch[i]), get_dsmatrix_ij(lu, row_ch[i], col_ch[i]));
		set_dsvector_i(answer, col_ch[i], dtmp);
	}

	return 0;
}

/************************************************************/
/*                 Solver for Linear System                 */
/*                (LU Decomposed Square Dense Matrix)       */
/*                        (Double-Single Precision)         */
/*          (Partial Pivoting with real swap of rows)       */
/************************************************************/
int SolveDSLSPM(DSVector answer, DSMatrix lu, DSVector b, long int ch[])
{
	long int i, j, dim;
	static float dtmp[DSSIZE], dtmp1[DSSIZE];

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_dsvector_i(answer, i, get_dsvector_i(b, ch[i]));

/* Forward */
	for(i = 0; i < dim; i++)
	{
		rds_abs(dtmp, get_dsmatrix_ij(lu, i, i));
		if(rds_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveDSLSPM, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rds_mul(dtmp1, get_dsmatrix_ij(lu, j, i), get_dsvector_i(answer, i));
			rds_sub(dtmp, get_dsvector_i(answer, j), dtmp1);
			set_dsvector_i(answer, j, dtmp);
		}
	}

/* Backward */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rds_mul(dtmp1, get_dsmatrix_ij(lu, i, j), get_dsvector_i(answer, j));
			rds_sub(dtmp, get_dsvector_i(answer, i), dtmp1);
			set_dsvector_i(answer, i, dtmp);
		}
		rds_div(dtmp, get_dsvector_i(answer, i), get_dsmatrix_ij(lu, i, i));
		set_dsvector_i(answer, i, dtmp);
	}

	return 0;
}
