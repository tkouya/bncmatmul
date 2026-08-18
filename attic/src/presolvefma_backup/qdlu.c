/********************************************************************************/
/* qdlu.c:                                                                      */
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
#include "qdlinear.h"

// QD

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_qdmatrix(QDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
	long int i, j, true_end;
	double tmp[QDSIZE];
	int thread_index;

	true_end = (col_end > mat->col_dim) ? mat->col_dim : col_end;

	for(i = col_start; i < true_end; i++)
	{
		rqd_set(tmp, get_qdmatrix_ij(mat, row_index0, i));
		set_qdmatrix_ij(mat, row_index0, i, get_qdmatrix_ij(mat, row_index1, i));
		set_qdmatrix_ij(mat, row_index1, i, tmp);
	}
}

// QD

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Octuple double Precision)       */
/*                                                          */
/*                 Ver. 0.0 2015-02-23 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int QDLUdecomp(QDMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       QDMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	static double dtmp[QDSIZE], dtmp1[QDSIZE], dmaxii[QDSIZE];
#ifdef BNC_USE_NEW_FMA
	static double neg_aji[QDSIZE];
#endif // BNC_USE_NEW_FMA

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
	{
		rqd_abs(dmaxii, get_qdmatrix_ij(a, i, i));
		if(rqd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (QDLUdecomp)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rqd_div(dtmp, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, i));
			set_qdmatrix_ij(a, j, i, dtmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
#ifdef BNC_USE_NEW_FMA
			rqd_neg(neg_aji, get_qdmatrix_ij(a, j, i));
#endif // BNC_USE_NEW_FMA
			for(k = (i + 1); k < dim; k++)
			{
#ifdef BNC_USE_NEW_FMA
				rqd_fma(dtmp, neg_aji, get_qdmatrix_ij(a, i, k), get_qdmatrix_ij(a, j, k));
#else // BNC_USE_NEW_FMA
				rqd_mul(dtmp1, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, k));
				rqd_sub(dtmp, get_qdmatrix_ij(a, j, k), dtmp1);
#endif // BNC_USE_NEW_FMA
				set_qdmatrix_ij(a, j, k, dtmp);
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
int SolveQDLS(QDVector answer, QDMatrix lu, QDVector b)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      QDMatrix lu: LU decomposed Matrix (given by user)   */
/*      QDVector b: constant vector (given by user)         */
/*      QDVector answer: Solution for linear system         */
/*      long int dim: Dimension of Matrix (given by user)   */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static double dtmp[QDSIZE], dtmp1[QDSIZE];

	dim = answer->dim;

	subst_qdvector(answer, b);

/* Forward */
	for(i = 0; i < dim; i++)
	{
		rqd_abs(dtmp, get_qdmatrix_ij(lu, i, i));
		if(rqd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveQDLS, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rqd_mul(dtmp1, get_qdmatrix_ij(lu, j, i), get_qdvector_i(answer, i));
			rqd_sub(dtmp, get_qdvector_i(answer, j), dtmp1);
			set_qdvector_i(answer, j, dtmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rqd_mul(dtmp1, get_qdmatrix_ij(lu, i, j), get_qdvector_i(answer, j));
			rqd_sub(dtmp, get_qdvector_i(answer, i), dtmp1);
			set_qdvector_i(answer, i, dtmp);
		}
		rqd_div(dtmp, get_qdvector_i(answer, i), get_qdmatrix_ij(lu, i, i));
		set_qdvector_i(answer, i, dtmp);
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
int QDLUdecompP(QDMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       QDMatrix a: Matrix (given by user)                 */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
	static double dtmp[QDSIZE], dtmp1[QDSIZE], dmaxii[QDSIZE];
#ifdef BNC_USE_NEW_FMA
	static double neg_aji[QDSIZE];
#endif // BNC_USE_NEW_FMA

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		rqd_abs(dmaxii, get_qdmatrix_ij(a, ch[i], i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rqd_abs(dtmp, get_qdmatrix_ij(a, ch[j], i));
			if(rqd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rqd_set(dmaxii, dtmp);
			}
		}

		if(rqd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! QDLUdecompP!\n", i);
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
			rqd_div(dtmp, get_qdmatrix_ij(a, ch[j], i), get_qdmatrix_ij(a, ch[i], i));
			set_qdmatrix_ij(a, ch[j], i, dtmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
#ifdef BNC_USE_NEW_FMA
			rqd_neg(neg_aji, get_qdmatrix_ij(a, ch[j], i));
#endif // BNC_USE_NEW_FMA
			for(k = (i + 1); k < dim; k++)
			{
#ifdef BNC_USE_NEW_FMA
				rqd_fma(dtmp, neg_aji, get_qdmatrix_ij(a, ch[i], k), get_qdmatrix_ij(a, ch[j], k));
#else // BNC_USE_NEW_FMA
				rqd_mul(dtmp1, get_qdmatrix_ij(a, ch[j], i), get_qdmatrix_ij(a, ch[i], k));
				rqd_sub(dtmp, get_qdmatrix_ij(a, ch[j], k), dtmp1);
#endif // BNC_USE_NEW_FMA
				set_qdmatrix_ij(a, ch[j], k, dtmp);
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
int SolveQDLSP(QDVector answer, QDMatrix lu, QDVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      QDMatrix lu[]: LU decomposed Matrix (given by user) */
/*      QDVector b[]: constant vector (given by user)       */
/*      QDVector answer[]: Solution for linear system       */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static double dtmp[QDSIZE], dtmp1[QDSIZE];

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_qdvector_i(answer, i, get_qdvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rqd_abs(dtmp, get_qdmatrix_ij(lu, ch[i], i));
		if(rqd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveQDLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rqd_mul(dtmp1, get_qdmatrix_ij(lu, ch[j], i), get_qdvector_i(answer, i));
			rqd_sub(dtmp, get_qdvector_i(answer, j), dtmp1);
			set_qdvector_i(answer, j, dtmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rqd_mul(dtmp1, get_qdmatrix_ij(lu, ch[i], j), get_qdvector_i(answer, j));
			rqd_sub(dtmp, get_qdvector_i(answer, i), dtmp1);
			set_qdvector_i(answer, i, dtmp);
		}
		rqd_div(dtmp, get_qdvector_i(answer, i), get_qdmatrix_ij(lu, ch[i], i));
		set_qdvector_i(answer, i, dtmp);
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
int QDLUdecompC(QDMatrix a, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       QDMatrix a[]: Matrix (given by user)               */
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
	static double dtmp[QDSIZE], dtmp1[QDSIZE], dmaxii[QDSIZE];
#ifdef BNC_USE_NEW_FMA
	static double neg_aji[QDSIZE];
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
		rqd_abs(dmaxii, get_qdmatrix_ij(a, row_ch[i], col_ch[i]));
		imax = i;
		jmax = i;
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rqd_abs(dtmp, get_qdmatrix_ij(a, row_ch[j], col_ch[k]));
				if(rqd_cmp(dtmp, dmaxii) > 0)
				{
					imax = j;
					jmax = k;
					rqd_set(dmaxii, dtmp);
				}
			}
		}

		if(rqd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (QDLUdecompC)!\n", i);
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
			rqd_div(dtmp, get_qdmatrix_ij(a, row_ch[j], col_ch[i]), get_qdmatrix_ij(a, row_ch[i], col_ch[i]));
			set_qdmatrix_ij(a, row_ch[j], col_ch[i], dtmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
#ifdef BNC_USE_NEW_FMA
			rqd_neg(neg_aji, get_qdmatrix_ij(a, row_ch[j], col_ch[i]));
#endif // BNC_USE_NEW_FMA
			for(k = (i + 1); k < dim; k++)
			{
#ifdef BNC_USE_NEW_FMA
				rqd_fma(dtmp, neg_aji, get_qdmatrix_ij(a, row_ch[i], col_ch[k]), get_qdmatrix_ij(a, row_ch[j], col_ch[k]));
#else // BNC_USE_NEW_FMA
				rqd_mul(dtmp1, get_qdmatrix_ij(a, row_ch[j], col_ch[i]), get_qdmatrix_ij(a, row_ch[i], col_ch[k]));
				rqd_sub(dtmp, get_qdmatrix_ij(a, row_ch[j], col_ch[k]), dtmp1);
#endif // BNC_USE_NEW_FMA
				set_qdmatrix_ij(a, row_ch[j], col_ch[k], dtmp);
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
int SolveQDLSC(QDVector answer, QDMatrix lu, QDVector b, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       QDMatrix lu: LU decomposed Matrix (given by user)  */
/*       QDVector b: constant vector (given by user)        */
/*       QDVector answer: Solution for linear system        */
/*       long int row_ch[]: Row order (given by user)       */
/*       long int col_ch[]: Column order (given by user)    */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static double dtmp[QDSIZE], dtmp1[QDSIZE];

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_qdvector_i(answer, col_ch[i], get_qdvector_i(b, row_ch[i]));

/* Forward */
	for(i = 0; i < dim; i++)
	{
		rqd_abs(dtmp, get_qdmatrix_ij(lu, row_ch[i], col_ch[i]));
		if(rqd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveQDLSC, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rqd_mul(dtmp1, get_qdmatrix_ij(lu, row_ch[j], col_ch[i]), get_qdvector_i(answer, col_ch[i]));
			rqd_sub(dtmp, get_qdvector_i(answer, col_ch[j]), dtmp1);
			set_qdvector_i(answer, col_ch[j],  dtmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rqd_mul(dtmp1, get_qdmatrix_ij(lu, row_ch[i], col_ch[j]), get_qdvector_i(answer, col_ch[j]));
			rqd_sub(dtmp, get_qdvector_i(answer, col_ch[i]), dtmp1);
			set_qdvector_i(answer, col_ch[i], dtmp);
		}
		rqd_div(dtmp, get_qdvector_i(answer, col_ch[i]), get_qdmatrix_ij(lu, row_ch[i], col_ch[i]));
		set_qdvector_i(answer, col_ch[i], dtmp);
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Octuple double Precision)       */
/*          (Partial Pivoting with real swap of rows)       */
/*                                                          */
/*                 Ver. 0.0 2021-02-17 (Wed) Tomonori Kouya */
/*                 Ver. 0.1 2025-12-19 (Thu) Neon/AVX512    */
/*                                                          */
/************************************************************/
int QDLUdecompPM(QDMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       QDMatrix a: Matrix (given by user)                 */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
	static double dtmp[QDSIZE], dtmp1[QDSIZE], dmaxii[QDSIZE];
// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m256d dtmp256[QDSIZE], aji256[QDSIZE], ajk256[QDSIZE], aik256[QDSIZE];
#elif defined(__AVX512F__) // __AVX512F__
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m512d dtmp512[QDSIZE], aji512[QDSIZE], ajk512[QDSIZE], aik512[QDSIZE];
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	
	svfloat64_t dtmp_neon_0, dtmp_neon_1, dtmp_neon_2, dtmp_neon_3;
	svfloat64_t aji_neon_0, aji_neon_1, aji_neon_2, aji_neon_3;
	svfloat64_t ajk_neon_0, ajk_neon_1, ajk_neon_2, ajk_neon_3;
	svfloat64_t aik_neon_0, aik_neon_1, aik_neon_2, aik_neon_3;
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	float64x2_t dtmp_neon[QDSIZE], aji_neon[QDSIZE], ajk_neon[QDSIZE], aik_neon[QDSIZE];
#else // normal
#endif // __AVX2__
#ifdef BNC_USE_NEW_FMA
	double neg_aji[QDSIZE];
#endif // BNC_USE_NEW_FMA


#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	printf("QDLUdecompPM AVX2 enabled!\n");
#elif defined(__AVX512F__) // __AVX512F__
	printf("QDLUdecompPM AVX-512 enabled!\n");
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	printf("QDLUdecompPM Arm Neon enabled!\n");
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	printf("QDLUdecompPM Arm Neon enabled!\n");
#else
	printf("QDLUdecompPM normal!\n");
#endif // __AVX2__

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		// partial pivoting
		rqd_abs(dmaxii, get_qdmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rqd_abs(dtmp, get_qdmatrix_ij(a, j, i));
			if(rqd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rqd_set(dmaxii, dtmp);
			}
		}

		if(rqd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! QDLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			row_swap_qdmatrix(a, i, imax, 0, a->col_dim);
		}

		for(j = (i + 1); j < dim; j++)
		{
			rqd_div(dtmp, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, i));
			set_qdmatrix_ij(a, j, i, dtmp);
		}
// SIMD : for copy & paste
#ifdef BNC_USE_NEW_FMA
// Fused update: a_jk := (-a_ji) * a_ik + a_jk in one branch-free FMA (arXiv:2607.11391)
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->real_col_dim;

		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->real_col_dim + i;
			neg_aji[0] = -(a->element[0][index_ji]);
			neg_aji[1] = -(a->element[1][index_ji]);
			neg_aji[2] = -(a->element[2][index_ji]);
			neg_aji[3] = -(a->element[3][index_ji]);
			aji256[0] = _mm256_set1_pd(neg_aji[0]);
			aji256[1] = _mm256_set1_pd(neg_aji[1]);
			aji256[2] = _mm256_set1_pd(neg_aji[2]);
			aji256[3] = _mm256_set1_pd(neg_aji[3]);

			// head
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rqd_fma(dtmp, neg_aji, get_qdmatrix_ij(a, i, k), get_qdmatrix_ij(a, j, k));
				set_qdmatrix_ij(a, j, k, dtmp);
			}

			// middle : SIMD
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				index_jk = j * a->real_col_dim + k;
				aik256[0] = _mm256_load_pd(&(a->element[0][index_ik]));
				aik256[1] = _mm256_load_pd(&(a->element[1][index_ik]));
				aik256[2] = _mm256_load_pd(&(a->element[2][index_ik]));
				aik256[3] = _mm256_load_pd(&(a->element[3][index_ik]));
				ajk256[0] = _mm256_load_pd(&(a->element[0][index_jk]));
				ajk256[1] = _mm256_load_pd(&(a->element[1][index_jk]));
				ajk256[2] = _mm256_load_pd(&(a->element[2][index_jk]));
				ajk256[3] = _mm256_load_pd(&(a->element[3][index_jk]));
				_bncavx2_qwfma(dtmp256, aji256, aik256, ajk256);
				_mm256_store_pd(&(a->element[0][index_jk]), dtmp256[0]);
				_mm256_store_pd(&(a->element[1][index_jk]), dtmp256[1]);
				_mm256_store_pd(&(a->element[2][index_jk]), dtmp256[2]);
				_mm256_store_pd(&(a->element[3][index_jk]), dtmp256[3]);
			}
		}
#elif defined(__AVX512F__) // __AVX512F__
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->real_col_dim;

		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->real_col_dim + i;
			neg_aji[0] = -(a->element[0][index_ji]);
			neg_aji[1] = -(a->element[1][index_ji]);
			neg_aji[2] = -(a->element[2][index_ji]);
			neg_aji[3] = -(a->element[3][index_ji]);
			aji512[0] = _mm512_set1_pd(neg_aji[0]);
			aji512[1] = _mm512_set1_pd(neg_aji[1]);
			aji512[2] = _mm512_set1_pd(neg_aji[2]);
			aji512[3] = _mm512_set1_pd(neg_aji[3]);

			// head
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rqd_fma(dtmp, neg_aji, get_qdmatrix_ij(a, i, k), get_qdmatrix_ij(a, j, k));
				set_qdmatrix_ij(a, j, k, dtmp);
			}

			// middle : SIMD
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				index_jk = j * a->real_col_dim + k;
				aik512[0] = _mm512_load_pd(&(a->element[0][index_ik]));
				aik512[1] = _mm512_load_pd(&(a->element[1][index_ik]));
				aik512[2] = _mm512_load_pd(&(a->element[2][index_ik]));
				aik512[3] = _mm512_load_pd(&(a->element[3][index_ik]));
				ajk512[0] = _mm512_load_pd(&(a->element[0][index_jk]));
				ajk512[1] = _mm512_load_pd(&(a->element[1][index_jk]));
				ajk512[2] = _mm512_load_pd(&(a->element[2][index_jk]));
				ajk512[3] = _mm512_load_pd(&(a->element[3][index_jk]));
				_bncavx512_qwfma(dtmp512, aji512, aik512, ajk512);
				_mm512_store_pd(&(a->element[0][index_jk]), dtmp512[0]);
				_mm512_store_pd(&(a->element[1][index_jk]), dtmp512[1]);
				_mm512_store_pd(&(a->element[2][index_jk]), dtmp512[2]);
				_mm512_store_pd(&(a->element[3][index_jk]), dtmp512[3]);
			}
		}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->real_col_dim;

		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->real_col_dim + i;
			neg_aji[0] = -(a->element[0][index_ji]);
			neg_aji[1] = -(a->element[1][index_ji]);
			neg_aji[2] = -(a->element[2][index_ji]);
			neg_aji[3] = -(a->element[3][index_ji]);
			aji_neon_0 = svdup_f64(neg_aji[0]);
			aji_neon_1 = svdup_f64(neg_aji[1]);
			aji_neon_2 = svdup_f64(neg_aji[2]);
			aji_neon_3 = svdup_f64(neg_aji[3]);

			// head
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rqd_fma(dtmp, neg_aji, get_qdmatrix_ij(a, i, k), get_qdmatrix_ij(a, j, k));
				set_qdmatrix_ij(a, j, k, dtmp);
			}

			// middle : SIMD (SVE2)
			for(k = dim_start; k < dim_end; k += (long int)svcntd())
			{
				svbool_t pg = svwhilelt_b64_s64((int64_t)k, (int64_t)(dim_end));
				index_ik = i * a->real_col_dim + k;
				index_jk = j * a->real_col_dim + k;
				aik_neon_0 = svld1_f64(pg, &(a->element[0][index_ik]));
				aik_neon_1 = svld1_f64(pg, &(a->element[1][index_ik]));
				aik_neon_2 = svld1_f64(pg, &(a->element[2][index_ik]));
				aik_neon_3 = svld1_f64(pg, &(a->element[3][index_ik]));
				ajk_neon_0 = svld1_f64(pg, &(a->element[0][index_jk]));
				ajk_neon_1 = svld1_f64(pg, &(a->element[1][index_jk]));
				ajk_neon_2 = svld1_f64(pg, &(a->element[2][index_jk]));
				ajk_neon_3 = svld1_f64(pg, &(a->element[3][index_jk]));
				_bncsve2_qwfma(pg, &dtmp_neon_0, &dtmp_neon_1, &dtmp_neon_2, &dtmp_neon_3, aji_neon_0, aji_neon_1, aji_neon_2, aji_neon_3, aik_neon_0, aik_neon_1, aik_neon_2, aik_neon_3, ajk_neon_0, ajk_neon_1, ajk_neon_2, ajk_neon_3);
				svst1_f64(pg, &(a->element[0][index_jk]), dtmp_neon_0);
				svst1_f64(pg, &(a->element[1][index_jk]), dtmp_neon_1);
				svst1_f64(pg, &(a->element[2][index_jk]), dtmp_neon_2);
				svst1_f64(pg, &(a->element[3][index_jk]), dtmp_neon_3);
			}
		}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->real_col_dim;

		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->real_col_dim + i;
			neg_aji[0] = -(a->element[0][index_ji]);
			neg_aji[1] = -(a->element[1][index_ji]);
			neg_aji[2] = -(a->element[2][index_ji]);
			neg_aji[3] = -(a->element[3][index_ji]);
			aji_neon[0] = vdupq_n_f64(neg_aji[0]);
			aji_neon[1] = vdupq_n_f64(neg_aji[1]);
			aji_neon[2] = vdupq_n_f64(neg_aji[2]);
			aji_neon[3] = vdupq_n_f64(neg_aji[3]);

			// head
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rqd_fma(dtmp, neg_aji, get_qdmatrix_ij(a, i, k), get_qdmatrix_ij(a, j, k));
				set_qdmatrix_ij(a, j, k, dtmp);
			}

			// middle : SIMD (Neon)
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				index_jk = j * a->real_col_dim + k;
				aik_neon[0] = vld1q_f64(&(a->element[0][index_ik]));
				aik_neon[1] = vld1q_f64(&(a->element[1][index_ik]));
				aik_neon[2] = vld1q_f64(&(a->element[2][index_ik]));
				aik_neon[3] = vld1q_f64(&(a->element[3][index_ik]));
				ajk_neon[0] = vld1q_f64(&(a->element[0][index_jk]));
				ajk_neon[1] = vld1q_f64(&(a->element[1][index_jk]));
				ajk_neon[2] = vld1q_f64(&(a->element[2][index_jk]));
				ajk_neon[3] = vld1q_f64(&(a->element[3][index_jk]));
				_bncneon_qwfma(dtmp_neon, aji_neon, aik_neon, ajk_neon);
				vst1q_f64(&(a->element[0][index_jk]), dtmp_neon[0]);
				vst1q_f64(&(a->element[1][index_jk]), dtmp_neon[1]);
				vst1q_f64(&(a->element[2][index_jk]), dtmp_neon[2]);
				vst1q_f64(&(a->element[3][index_jk]), dtmp_neon[3]);
			}
		}
#else // others
		for(j = (i + 1); j < dim; j++)
		{
			rqd_neg(neg_aji, get_qdmatrix_ij(a, j, i));
			for(k = (i + 1); k < dim; k++)
			{
				rqd_fma(dtmp, neg_aji, get_qdmatrix_ij(a, i, k), get_qdmatrix_ij(a, j, k));
				set_qdmatrix_ij(a, j, k, dtmp);
			}
		}
#endif // __AVX2__
#else // BNC_USE_NEW_FMA
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
			aji256[3] = _mm256_set_pd(
                a->element[3][index_ji],
                a->element[3][index_ji],
                a->element[3][index_ji],
                a->element[3][index_ji]
            );

			// head
			//printf("start j, k= %ld, %ld, ", j, i + 1);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			//for(k = (i + 1); k < dim; k++)
			{
				rqd_mul(dtmp1, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, k));
				rqd_sub(dtmp, get_qdmatrix_ij(a, j, k), dtmp1);
				set_qdmatrix_ij(a, j, k, dtmp);
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
				aik256[3] = _mm256_load_pd(&(a->element[3][index_ik]));				
				_bncavx2_rqd_mul(dtmp256, aji256, aik256);
				//printf(" -- mul -- ");

				index_jk = j * a->real_col_dim + k;
				//rdd_sub(dtmp, get_ddmatrix_ij(a, j, k), dtmp1);
				ajk256[0] = _mm256_load_pd(&(a->element[0][index_jk]));
				ajk256[1] = _mm256_load_pd(&(a->element[1][index_jk]));
				ajk256[2] = _mm256_load_pd(&(a->element[2][index_jk]));
				ajk256[3] = _mm256_load_pd(&(a->element[3][index_jk]));
				_bncavx2_rqd_sub(dtmp256, ajk256, dtmp256);
				//printf(" -- sub -- ");

				//set_ddmatrix_ij(a, j, k, dtmp);
				_mm256_store_pd(&(a->element[0][index_jk]), dtmp256[0]);
				_mm256_store_pd(&(a->element[1][index_jk]), dtmp256[1]);
				_mm256_store_pd(&(a->element[2][index_jk]), dtmp256[2]);
				_mm256_store_pd(&(a->element[3][index_jk]), dtmp256[3]);
			}
			//printf(", %ld middle", k);
		}
#elif defined(__AVX512F__) // __AVX512F__
		// AVX-512 processes 8 doubles at a time
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->real_col_dim;

		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->real_col_dim + i;
			// Broadcast a[j][i] to all elements in AVX-512 registers
			aji512[0] = _mm512_set1_pd(a->element[0][index_ji]);
			aji512[1] = _mm512_set1_pd(a->element[1][index_ji]);
			aji512[2] = _mm512_set1_pd(a->element[2][index_ji]);
			aji512[3] = _mm512_set1_pd(a->element[3][index_ji]);

			// head: process elements before aligned boundary
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rqd_mul(dtmp1, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, k));
				rqd_sub(dtmp, get_qdmatrix_ij(a, j, k), dtmp1);
				set_qdmatrix_ij(a, j, k, dtmp);
			}

			// middle: SIMD processing with AVX-512
			k_end = dim_end;
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				// Load a[i][k] elements
				aik512[0] = _mm512_load_pd(&(a->element[0][index_ik]));
				aik512[1] = _mm512_load_pd(&(a->element[1][index_ik]));
				aik512[2] = _mm512_load_pd(&(a->element[2][index_ik]));
				aik512[3] = _mm512_load_pd(&(a->element[3][index_ik]));
				
				// Multiply: dtmp = a[j][i] * a[i][k]
				_bncavx512_rqd_mul(dtmp512, aji512, aik512);

				index_jk = j * a->real_col_dim + k;
				// Load a[j][k] elements
				ajk512[0] = _mm512_load_pd(&(a->element[0][index_jk]));
				ajk512[1] = _mm512_load_pd(&(a->element[1][index_jk]));
				ajk512[2] = _mm512_load_pd(&(a->element[2][index_jk]));
				ajk512[3] = _mm512_load_pd(&(a->element[3][index_jk]));
				
				// Subtract: dtmp = a[j][k] - dtmp
				_bncavx512_rqd_sub(dtmp512, ajk512, dtmp512);

				// Store result back to a[j][k]
				_mm512_store_pd(&(a->element[0][index_jk]), dtmp512[0]);
				_mm512_store_pd(&(a->element[1][index_jk]), dtmp512[1]);
				_mm512_store_pd(&(a->element[2][index_jk]), dtmp512[2]);
				_mm512_store_pd(&(a->element[3][index_jk]), dtmp512[3]);
			}
		}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
		// Neon processes 2 doubles at a time
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->real_col_dim;

		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->real_col_dim + i;
			// Broadcast a[j][i] to all elements in Neon registers
			aji_neon_0 = svdup_f64(a->element[0][index_ji]);
			aji_neon_1 = svdup_f64(a->element[1][index_ji]);
			aji_neon_2 = svdup_f64(a->element[2][index_ji]);
			aji_neon_3 = svdup_f64(a->element[3][index_ji]);

			// head: process elements before aligned boundary
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rqd_mul(dtmp1, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, k));
				rqd_sub(dtmp, get_qdmatrix_ij(a, j, k), dtmp1);
				set_qdmatrix_ij(a, j, k, dtmp);
			}

			// middle: SIMD processing with Neon
			k_end = dim_end;
			for(k = dim_start; k < dim_end; k += (long int)svcntd())
			{
		svbool_t pg = svwhilelt_b64_s64((int64_t)k, (int64_t)(dim_end));
				index_ik = i * a->real_col_dim + k;
				// Load a[i][k] elements
				aik_neon_0 = svld1_f64(pg, &(a->element[0][index_ik]));
				aik_neon_1 = svld1_f64(pg, &(a->element[1][index_ik]));
				aik_neon_2 = svld1_f64(pg, &(a->element[2][index_ik]));
				aik_neon_3 = svld1_f64(pg, &(a->element[3][index_ik]));
				
				// Multiply: dtmp = a[j][i] * a[i][k]
				_bncsve2_rqd_mul(pg, &dtmp_neon_0, &dtmp_neon_1, &dtmp_neon_2, &dtmp_neon_3, aji_neon_0, aji_neon_1, aji_neon_2, aji_neon_3, aik_neon_0, aik_neon_1, aik_neon_2, aik_neon_3);

				index_jk = j * a->real_col_dim + k;
				// Load a[j][k] elements
				ajk_neon_0 = svld1_f64(pg, &(a->element[0][index_jk]));
				ajk_neon_1 = svld1_f64(pg, &(a->element[1][index_jk]));
				ajk_neon_2 = svld1_f64(pg, &(a->element[2][index_jk]));
				ajk_neon_3 = svld1_f64(pg, &(a->element[3][index_jk]));
				
				// Subtract: dtmp = a[j][k] - dtmp
				_bncsve2_rqd_neg(pg, &dtmp_neon_0, &dtmp_neon_1, &dtmp_neon_2, &dtmp_neon_3, dtmp_neon_0, dtmp_neon_1, dtmp_neon_2, dtmp_neon_3);
		_bncsve2_rqd_add(pg, &dtmp_neon_0, &dtmp_neon_1, &dtmp_neon_2, &dtmp_neon_3, ajk_neon_0, ajk_neon_1, ajk_neon_2, ajk_neon_3, dtmp_neon_0, dtmp_neon_1, dtmp_neon_2, dtmp_neon_3);

				// Store result back to a[j][k]
				svst1_f64(pg, &(a->element[0][index_jk]), dtmp_neon_0);
				svst1_f64(pg, &(a->element[1][index_jk]), dtmp_neon_1);
				svst1_f64(pg, &(a->element[2][index_jk]), dtmp_neon_2);
				svst1_f64(pg, &(a->element[3][index_jk]), dtmp_neon_3);
			}
		}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
		// Neon processes 2 doubles at a time
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->real_col_dim;

		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->real_col_dim + i;
			// Broadcast a[j][i] to all elements in Neon registers
			aji_neon[0] = vdupq_n_f64(a->element[0][index_ji]);
			aji_neon[1] = vdupq_n_f64(a->element[1][index_ji]);
			aji_neon[2] = vdupq_n_f64(a->element[2][index_ji]);
			aji_neon[3] = vdupq_n_f64(a->element[3][index_ji]);

			// head: process elements before aligned boundary
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rqd_mul(dtmp1, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, k));
				rqd_sub(dtmp, get_qdmatrix_ij(a, j, k), dtmp1);
				set_qdmatrix_ij(a, j, k, dtmp);
			}

			// middle: SIMD processing with Neon
			k_end = dim_end;
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				// Load a[i][k] elements
				aik_neon[0] = vld1q_f64(&(a->element[0][index_ik]));
				aik_neon[1] = vld1q_f64(&(a->element[1][index_ik]));
				aik_neon[2] = vld1q_f64(&(a->element[2][index_ik]));
				aik_neon[3] = vld1q_f64(&(a->element[3][index_ik]));
				
				// Multiply: dtmp = a[j][i] * a[i][k]
				_bncneon_rqd_mul(dtmp_neon, aji_neon, aik_neon);

				index_jk = j * a->real_col_dim + k;
				// Load a[j][k] elements
				ajk_neon[0] = vld1q_f64(&(a->element[0][index_jk]));
				ajk_neon[1] = vld1q_f64(&(a->element[1][index_jk]));
				ajk_neon[2] = vld1q_f64(&(a->element[2][index_jk]));
				ajk_neon[3] = vld1q_f64(&(a->element[3][index_jk]));
				
				// Subtract: dtmp = a[j][k] - dtmp
				_bncneon_rqd_sub(dtmp_neon, ajk_neon, dtmp_neon);

				// Store result back to a[j][k]
				vst1q_f64(&(a->element[0][index_jk]), dtmp_neon[0]);
				vst1q_f64(&(a->element[1][index_jk]), dtmp_neon[1]);
				vst1q_f64(&(a->element[2][index_jk]), dtmp_neon[2]);
				vst1q_f64(&(a->element[3][index_jk]), dtmp_neon[3]);
			}
		}
#else // others
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rqd_mul(dtmp1, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, k));
				rqd_sub(dtmp, get_qdmatrix_ij(a, j, k), dtmp1);
				set_qdmatrix_ij(a, j, k, dtmp);
			}
		}
#endif // __AVX2__
#endif // BNC_USE_NEW_FMA
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
int SolveQDLSPM(QDVector answer, QDMatrix lu, QDVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      QDMatrix lu[]: LU decomposed Matrix (given by user) */
/*      QDVector b[]: constant vector (given by user)       */
/*      QDVector answer[]: Solution for linear system       */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static double dtmp[QDSIZE], dtmp1[QDSIZE];

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_qdvector_i(answer, i, get_qdvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rqd_abs(dtmp, get_qdmatrix_ij(lu, i, i));
		if(rqd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveQDLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rqd_mul(dtmp1, get_qdmatrix_ij(lu, j, i), get_qdvector_i(answer, i));
			rqd_sub(dtmp, get_qdvector_i(answer, j), dtmp1);
			set_qdvector_i(answer, j, dtmp);
		}
	}

/* Backward */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rqd_mul(dtmp1, get_qdmatrix_ij(lu, i, j), get_qdvector_i(answer, j));
			rqd_sub(dtmp, get_qdvector_i(answer, i), dtmp1);
			set_qdvector_i(answer, i, dtmp);
		}
		rqd_div(dtmp, get_qdvector_i(answer, i), get_qdmatrix_ij(lu, i, i));
		set_qdvector_i(answer, i, dtmp);
	}

	return 0;
}

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
