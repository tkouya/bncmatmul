
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

#ifdef BNC_USE_NEW_FMA
/************************************************************/
/* sum := sum_{k=start}^{end-1} lu(row, k) * v(k)           */
/* (fused branch-free FMA dot product for LU solvers)       */
/************************************************************/
static void _bnc_tdsolve_dot(double sum[TDSIZE], TDMatrix lu, long int row, long int start, long int end, TDVector v)
{
	long int k;
	double rtmp[TDSIZE], vtmp[TDSIZE];
	double *row_e[TDSIZE], *vec_e[TDSIZE];

	row_e[0] = &(lu->element[0][row * lu->real_col_dim]);
	row_e[1] = &(lu->element[1][row * lu->real_col_dim]);
	row_e[2] = &(lu->element[2][row * lu->real_col_dim]);
	vec_e[0] = &(v->element[0][0]);
	vec_e[1] = &(v->element[1][0]);
	vec_e[2] = &(v->element[2][0]);
	sum[0] = (double)0.0;
	sum[1] = (double)0.0;
	sum[2] = (double)0.0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	{
		__m256d vacc[TDSIZE], vr[TDSIZE], vv[TDSIZE];
		double red[TDSIZE][8];
		long int k_simd_end = start + ((end - start) / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		long int w;

		vacc[0] = _mm256_setzero_pd();
		vacc[1] = _mm256_setzero_pd();
		vacc[2] = _mm256_setzero_pd();
		for(k = start; k < k_simd_end; k += _BNC_D_WIDTH)
		{
			vr[0] = _mm256_loadu_pd(&(row_e[0][k]));
			vr[1] = _mm256_loadu_pd(&(row_e[1][k]));
			vr[2] = _mm256_loadu_pd(&(row_e[2][k]));
			vv[0] = _mm256_loadu_pd(&(vec_e[0][k]));
			vv[1] = _mm256_loadu_pd(&(vec_e[1][k]));
			vv[2] = _mm256_loadu_pd(&(vec_e[2][k]));
			_bncavx2_twfma(vacc, vr, vv, vacc);
		}
		_mm256_storeu_pd(&(red[0][0]), vacc[0]);
		_mm256_storeu_pd(&(red[1][0]), vacc[1]);
		_mm256_storeu_pd(&(red[2][0]), vacc[2]);
		for(w = 0; w < _BNC_D_WIDTH; w++)
		{
			vtmp[0] = red[0][w];
			vtmp[1] = red[1][w];
			vtmp[2] = red[2][w];
			rtd_add(sum, sum, vtmp);
		}
		for(k = k_simd_end; k < end; k++)
		{
			rtmp[0] = row_e[0][k];
			rtmp[1] = row_e[1][k];
			rtmp[2] = row_e[2][k];
			vtmp[0] = vec_e[0][k];
			vtmp[1] = vec_e[1][k];
			vtmp[2] = vec_e[2][k];
			rtd_fma(sum, rtmp, vtmp, sum);
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	{
		__m512d vacc[TDSIZE], vr[TDSIZE], vv[TDSIZE];
		double red[TDSIZE][8];
		long int k_simd_end = start + ((end - start) / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		long int w;

		vacc[0] = _mm512_setzero_pd();
		vacc[1] = _mm512_setzero_pd();
		vacc[2] = _mm512_setzero_pd();
		for(k = start; k < k_simd_end; k += _BNC_D_WIDTH)
		{
			vr[0] = _mm512_loadu_pd(&(row_e[0][k]));
			vr[1] = _mm512_loadu_pd(&(row_e[1][k]));
			vr[2] = _mm512_loadu_pd(&(row_e[2][k]));
			vv[0] = _mm512_loadu_pd(&(vec_e[0][k]));
			vv[1] = _mm512_loadu_pd(&(vec_e[1][k]));
			vv[2] = _mm512_loadu_pd(&(vec_e[2][k]));
			_bncavx512_twfma(vacc, vr, vv, vacc);
		}
		_mm512_storeu_pd(&(red[0][0]), vacc[0]);
		_mm512_storeu_pd(&(red[1][0]), vacc[1]);
		_mm512_storeu_pd(&(red[2][0]), vacc[2]);
		for(w = 0; w < _BNC_D_WIDTH; w++)
		{
			vtmp[0] = red[0][w];
			vtmp[1] = red[1][w];
			vtmp[2] = red[2][w];
			rtd_add(sum, sum, vtmp);
		}
		for(k = k_simd_end; k < end; k++)
		{
			rtmp[0] = row_e[0][k];
			rtmp[1] = row_e[1][k];
			rtmp[2] = row_e[2][k];
			vtmp[0] = vec_e[0][k];
			vtmp[1] = vec_e[1][k];
			vtmp[2] = vec_e[2][k];
			rtd_fma(sum, rtmp, vtmp, sum);
		}
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	{
		svbool_t pg = svptrue_b64();
		svfloat64_t acc0, acc1, acc2;
		svfloat64_t r0, r1, r2;
		svfloat64_t v0, v1, v2;
		double red[TDSIZE][64];
		long int vl = (long int)svcntd();
		long int k_simd_end = start + ((end - start) / vl) * vl;
		long int w;

		acc0 = svdup_f64(0.0);
		acc1 = svdup_f64(0.0);
		acc2 = svdup_f64(0.0);
		for(k = start; k < k_simd_end; k += vl)
		{
			r0 = svld1_f64(pg, &(row_e[0][k]));
			r1 = svld1_f64(pg, &(row_e[1][k]));
			r2 = svld1_f64(pg, &(row_e[2][k]));
			v0 = svld1_f64(pg, &(vec_e[0][k]));
			v1 = svld1_f64(pg, &(vec_e[1][k]));
			v2 = svld1_f64(pg, &(vec_e[2][k]));
			_bncsve2_twfma(pg, &acc0, &acc1, &acc2, r0, r1, r2, v0, v1, v2, acc0, acc1, acc2);
		}
		svst1_f64(pg, &(red[0][0]), acc0);
		svst1_f64(pg, &(red[1][0]), acc1);
		svst1_f64(pg, &(red[2][0]), acc2);
		for(w = 0; w < vl; w++)
		{
			vtmp[0] = red[0][w];
			vtmp[1] = red[1][w];
			vtmp[2] = red[2][w];
			rtd_add(sum, sum, vtmp);
		}
		for(k = k_simd_end; k < end; k++)
		{
			rtmp[0] = row_e[0][k];
			rtmp[1] = row_e[1][k];
			rtmp[2] = row_e[2][k];
			vtmp[0] = vec_e[0][k];
			vtmp[1] = vec_e[1][k];
			vtmp[2] = vec_e[2][k];
			rtd_fma(sum, rtmp, vtmp, sum);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	{
		float64x2_t vacc[TDSIZE], vr[TDSIZE], vv[TDSIZE];
		double red[TDSIZE][8];
		long int k_simd_end = start + ((end - start) / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		long int w;

		vacc[0] = vdupq_n_f64(0.0);
		vacc[1] = vdupq_n_f64(0.0);
		vacc[2] = vdupq_n_f64(0.0);
		for(k = start; k < k_simd_end; k += _BNC_D_WIDTH)
		{
			vr[0] = vld1q_f64(&(row_e[0][k]));
			vr[1] = vld1q_f64(&(row_e[1][k]));
			vr[2] = vld1q_f64(&(row_e[2][k]));
			vv[0] = vld1q_f64(&(vec_e[0][k]));
			vv[1] = vld1q_f64(&(vec_e[1][k]));
			vv[2] = vld1q_f64(&(vec_e[2][k]));
			_bncneon_twfma(vacc, vr, vv, vacc);
		}
		vst1q_f64(&(red[0][0]), vacc[0]);
		vst1q_f64(&(red[1][0]), vacc[1]);
		vst1q_f64(&(red[2][0]), vacc[2]);
		for(w = 0; w < _BNC_D_WIDTH; w++)
		{
			vtmp[0] = red[0][w];
			vtmp[1] = red[1][w];
			vtmp[2] = red[2][w];
			rtd_add(sum, sum, vtmp);
		}
		for(k = k_simd_end; k < end; k++)
		{
			rtmp[0] = row_e[0][k];
			rtmp[1] = row_e[1][k];
			rtmp[2] = row_e[2][k];
			vtmp[0] = vec_e[0][k];
			vtmp[1] = vec_e[1][k];
			vtmp[2] = vec_e[2][k];
			rtd_fma(sum, rtmp, vtmp, sum);
		}
	}
#else // others
	for(k = start; k < end; k++)
	{
		rtmp[0] = row_e[0][k];
		rtmp[1] = row_e[1][k];
		rtmp[2] = row_e[2][k];
		vtmp[0] = vec_e[0][k];
		vtmp[1] = vec_e[1][k];
		vtmp[2] = vec_e[2][k];
		rtd_fma(sum, rtmp, vtmp, sum);
	}
#endif // __AVX2__
}
#endif // BNC_USE_NEW_FMA


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
#ifdef BNC_USE_NEW_FMA
	static double neg_aji[TDSIZE];
#endif // BNC_USE_NEW_FMA

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
#ifdef BNC_USE_NEW_FMA
			rtd_neg(neg_aji, get_tdmatrix_ij(a, j, i));
#endif // BNC_USE_NEW_FMA
			for(k = (i + 1); k < dim; k++)
			{
#ifdef BNC_USE_NEW_FMA
				rtd_fma(dtmp, neg_aji, get_tdmatrix_ij(a, i, k), get_tdmatrix_ij(a, j, k));
#else // BNC_USE_NEW_FMA
				rtd_mul(dtmp1, get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, k));
				rtd_sub(dtmp, get_tdmatrix_ij(a, j, k), dtmp1);
#endif // BNC_USE_NEW_FMA
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

#ifdef BNC_USE_NEW_FMA
/* fused branch-free FMA + SIMD forward/backward substitution */
	for(i = 0; i < dim; i++)
	{
		rtd_abs(dtmp, get_tdmatrix_ij(lu, i, i));
		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveTDLS, %ld)\n", i);
			return -1;
		}
	}

/* Forward (row-oriented) */
	for(i = 1; i < dim; i++)
	{
		_bnc_tdsolve_dot(dtmp1, lu, i, 0, i, answer);
		rtd_sub(dtmp, get_tdvector_i(answer, i), dtmp1);
		set_tdvector_i(answer, i, dtmp);
	}

/* Backward (row-oriented) */
	for(i = (dim - 1); i >= 0; i--)
	{
		_bnc_tdsolve_dot(dtmp1, lu, i, i + 1, dim, answer);
		rtd_sub(dtmp, get_tdvector_i(answer, i), dtmp1);
		rtd_div(dtmp, dtmp, get_tdmatrix_ij(lu, i, i));
		set_tdvector_i(answer, i, dtmp);
	}
#else // BNC_USE_NEW_FMA
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

#endif // BNC_USE_NEW_FMA
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
#ifdef BNC_USE_NEW_FMA
	static double neg_aji[TDSIZE];
#endif // BNC_USE_NEW_FMA

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
#ifdef BNC_USE_NEW_FMA
			rtd_neg(neg_aji, get_tdmatrix_ij(a, ch[j], i));
#endif // BNC_USE_NEW_FMA
			for(k = (i + 1); k < dim; k++)
			{
#ifdef BNC_USE_NEW_FMA
				rtd_fma(dtmp, neg_aji, get_tdmatrix_ij(a, ch[i], k), get_tdmatrix_ij(a, ch[j], k));
#else // BNC_USE_NEW_FMA
				rtd_mul(dtmp1, get_tdmatrix_ij(a, ch[j], i), get_tdmatrix_ij(a, ch[i], k));
				rtd_sub(dtmp, get_tdmatrix_ij(a, ch[j], k), dtmp1);
#endif // BNC_USE_NEW_FMA
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

	
#ifdef BNC_USE_NEW_FMA
/* fused branch-free FMA + SIMD forward/backward substitution */
	for(i = 0; i < dim; i++)
	{
		rtd_abs(dtmp, get_tdmatrix_ij(lu, ch[i], i));
		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveTDLSP, %ld)\n", i);
			return -1;
		}
	}

/* Forward (row-oriented) */
	for(i = 1; i < dim; i++)
	{
		_bnc_tdsolve_dot(dtmp1, lu, ch[i], 0, i, answer);
		rtd_sub(dtmp, get_tdvector_i(answer, i), dtmp1);
		set_tdvector_i(answer, i, dtmp);
	}

/* Backward (row-oriented) */
	for(i = (dim - 1); i >= 0; i--)
	{
		_bnc_tdsolve_dot(dtmp1, lu, ch[i], i + 1, dim, answer);
		rtd_sub(dtmp, get_tdvector_i(answer, i), dtmp1);
		rtd_div(dtmp, dtmp, get_tdmatrix_ij(lu, ch[i], i));
		set_tdvector_i(answer, i, dtmp);
	}
#else // BNC_USE_NEW_FMA
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

#endif // BNC_USE_NEW_FMA
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
#ifdef BNC_USE_NEW_FMA
	static double neg_aji[TDSIZE];
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
#ifdef BNC_USE_NEW_FMA
			rtd_neg(neg_aji, get_tdmatrix_ij(a, row_ch[j], col_ch[i]));
#endif // BNC_USE_NEW_FMA
			for(k = (i + 1); k < dim; k++)
			{
#ifdef BNC_USE_NEW_FMA
				rtd_fma(dtmp, neg_aji, get_tdmatrix_ij(a, row_ch[i], col_ch[k]), get_tdmatrix_ij(a, row_ch[j], col_ch[k]));
#else // BNC_USE_NEW_FMA
				rtd_mul(dtmp1, get_tdmatrix_ij(a, row_ch[j], col_ch[i]), get_tdmatrix_ij(a, row_ch[i], col_ch[k]));
				rtd_sub(dtmp, get_tdmatrix_ij(a, row_ch[j], col_ch[k]), dtmp1);
#endif // BNC_USE_NEW_FMA
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

#ifdef BNC_USE_NEW_FMA
/* fused branch-free FMA substitution (complete pivoting: scalar) */
	{
	static double ntmp[TDSIZE];

/* Forward */
	for(i = 0; i < dim; i++)
	{
		rtd_abs(dtmp, get_tdmatrix_ij(lu, row_ch[i], col_ch[i]));
		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveTDLSC, %ld)\n", i);
			return -1;
		}

		rtd_neg(ntmp, get_tdvector_i(answer, col_ch[i]));
		for(j = (i + 1); j < dim; j++)
		{
			rtd_fma(dtmp, get_tdmatrix_ij(lu, row_ch[j], col_ch[i]), ntmp, get_tdvector_i(answer, col_ch[j]));
			set_tdvector_i(answer, col_ch[j], dtmp);
		}
	}

/* Backward */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rtd_neg(ntmp, get_tdmatrix_ij(lu, row_ch[i], col_ch[j]));
			rtd_fma(dtmp, ntmp, get_tdvector_i(answer, col_ch[j]), get_tdvector_i(answer, col_ch[i]));
			set_tdvector_i(answer, col_ch[i], dtmp);
		}
		rtd_div(dtmp, get_tdvector_i(answer, col_ch[i]), get_tdmatrix_ij(lu, row_ch[i], col_ch[i]));
		set_tdvector_i(answer, col_ch[i], dtmp);
	}
	}
#else // BNC_USE_NEW_FMA
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

#endif // BNC_USE_NEW_FMA
	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                          (triple double Precision)       */
/*          (Partial Pivoting with real swap of rows)       */
/*                                                          */
/*                 Ver. 0.0 2021-02-17 (Wed) Tomonori Kouya */
/*                 Ver. 0.1 2025-12-19 (Thu) Arm Neon Added */
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
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m512d dtmp512[TDSIZE], aji512[TDSIZE], ajk512[TDSIZE], aik512[TDSIZE];
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	
	svfloat64_t dtmp_neon_0, dtmp_neon_1, dtmp_neon_2;
	svfloat64_t aji_neon_0, aji_neon_1, aji_neon_2;
	svfloat64_t ajk_neon_0, ajk_neon_1, ajk_neon_2;
	svfloat64_t aik_neon_0, aik_neon_1, aik_neon_2;
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	float64x2_t dtmp_neon[TDSIZE], aji_neon[TDSIZE], ajk_neon[TDSIZE], aik_neon[TDSIZE];
#else // normal
#endif // __AVX2__
#ifdef BNC_USE_NEW_FMA
	double neg_aji[TDSIZE];
#endif // BNC_USE_NEW_FMA


#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	printf("TDLUdecompPM AVX2 enabled!\n");
#elif defined(__AVX512F__) // __AVX512F__
	printf("TDLUdecompPM AVX-512 enabled!\n");
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	printf("TDLUdecompPM Arm Neon enabled!\n");
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	printf("TDLUdecompPM Arm Neon enabled!\n");
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
			aji256[0] = _mm256_set1_pd(neg_aji[0]);
			aji256[1] = _mm256_set1_pd(neg_aji[1]);
			aji256[2] = _mm256_set1_pd(neg_aji[2]);

			// head
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rtd_fma(dtmp, neg_aji, get_tdmatrix_ij(a, i, k), get_tdmatrix_ij(a, j, k));
				set_tdmatrix_ij(a, j, k, dtmp);
			}

			// middle : SIMD
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				index_jk = j * a->real_col_dim + k;
				aik256[0] = _mm256_load_pd(&(a->element[0][index_ik]));
				aik256[1] = _mm256_load_pd(&(a->element[1][index_ik]));
				aik256[2] = _mm256_load_pd(&(a->element[2][index_ik]));
				ajk256[0] = _mm256_load_pd(&(a->element[0][index_jk]));
				ajk256[1] = _mm256_load_pd(&(a->element[1][index_jk]));
				ajk256[2] = _mm256_load_pd(&(a->element[2][index_jk]));
				_bncavx2_twfma(dtmp256, aji256, aik256, ajk256);
				_mm256_store_pd(&(a->element[0][index_jk]), dtmp256[0]);
				_mm256_store_pd(&(a->element[1][index_jk]), dtmp256[1]);
				_mm256_store_pd(&(a->element[2][index_jk]), dtmp256[2]);
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
			aji512[0] = _mm512_set1_pd(neg_aji[0]);
			aji512[1] = _mm512_set1_pd(neg_aji[1]);
			aji512[2] = _mm512_set1_pd(neg_aji[2]);

			// head
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rtd_fma(dtmp, neg_aji, get_tdmatrix_ij(a, i, k), get_tdmatrix_ij(a, j, k));
				set_tdmatrix_ij(a, j, k, dtmp);
			}

			// middle : SIMD
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				index_jk = j * a->real_col_dim + k;
				aik512[0] = _mm512_load_pd(&(a->element[0][index_ik]));
				aik512[1] = _mm512_load_pd(&(a->element[1][index_ik]));
				aik512[2] = _mm512_load_pd(&(a->element[2][index_ik]));
				ajk512[0] = _mm512_load_pd(&(a->element[0][index_jk]));
				ajk512[1] = _mm512_load_pd(&(a->element[1][index_jk]));
				ajk512[2] = _mm512_load_pd(&(a->element[2][index_jk]));
				_bncavx512_twfma(dtmp512, aji512, aik512, ajk512);
				_mm512_store_pd(&(a->element[0][index_jk]), dtmp512[0]);
				_mm512_store_pd(&(a->element[1][index_jk]), dtmp512[1]);
				_mm512_store_pd(&(a->element[2][index_jk]), dtmp512[2]);
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
			aji_neon_0 = svdup_f64(neg_aji[0]);
			aji_neon_1 = svdup_f64(neg_aji[1]);
			aji_neon_2 = svdup_f64(neg_aji[2]);

			// head
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rtd_fma(dtmp, neg_aji, get_tdmatrix_ij(a, i, k), get_tdmatrix_ij(a, j, k));
				set_tdmatrix_ij(a, j, k, dtmp);
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
				ajk_neon_0 = svld1_f64(pg, &(a->element[0][index_jk]));
				ajk_neon_1 = svld1_f64(pg, &(a->element[1][index_jk]));
				ajk_neon_2 = svld1_f64(pg, &(a->element[2][index_jk]));
				_bncsve2_twfma(pg, &dtmp_neon_0, &dtmp_neon_1, &dtmp_neon_2, aji_neon_0, aji_neon_1, aji_neon_2, aik_neon_0, aik_neon_1, aik_neon_2, ajk_neon_0, ajk_neon_1, ajk_neon_2);
				svst1_f64(pg, &(a->element[0][index_jk]), dtmp_neon_0);
				svst1_f64(pg, &(a->element[1][index_jk]), dtmp_neon_1);
				svst1_f64(pg, &(a->element[2][index_jk]), dtmp_neon_2);
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
			aji_neon[0] = vdupq_n_f64(neg_aji[0]);
			aji_neon[1] = vdupq_n_f64(neg_aji[1]);
			aji_neon[2] = vdupq_n_f64(neg_aji[2]);

			// head
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rtd_fma(dtmp, neg_aji, get_tdmatrix_ij(a, i, k), get_tdmatrix_ij(a, j, k));
				set_tdmatrix_ij(a, j, k, dtmp);
			}

			// middle : SIMD (Neon)
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				index_jk = j * a->real_col_dim + k;
				aik_neon[0] = vld1q_f64(&(a->element[0][index_ik]));
				aik_neon[1] = vld1q_f64(&(a->element[1][index_ik]));
				aik_neon[2] = vld1q_f64(&(a->element[2][index_ik]));
				ajk_neon[0] = vld1q_f64(&(a->element[0][index_jk]));
				ajk_neon[1] = vld1q_f64(&(a->element[1][index_jk]));
				ajk_neon[2] = vld1q_f64(&(a->element[2][index_jk]));
				_bncneon_twfma(dtmp_neon, aji_neon, aik_neon, ajk_neon);
				vst1q_f64(&(a->element[0][index_jk]), dtmp_neon[0]);
				vst1q_f64(&(a->element[1][index_jk]), dtmp_neon[1]);
				vst1q_f64(&(a->element[2][index_jk]), dtmp_neon[2]);
			}
		}
#else // others
		for(j = (i + 1); j < dim; j++)
		{
			rtd_neg(neg_aji, get_tdmatrix_ij(a, j, i));
			for(k = (i + 1); k < dim; k++)
			{
				rtd_fma(dtmp, neg_aji, get_tdmatrix_ij(a, i, k), get_tdmatrix_ij(a, j, k));
				set_tdmatrix_ij(a, j, k, dtmp);
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

			// head: process elements before aligned boundary
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rtd_mul(dtmp1, get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, k));
				rtd_sub(dtmp, get_tdmatrix_ij(a, j, k), dtmp1);
				set_tdmatrix_ij(a, j, k, dtmp);
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
				
				// Multiply: dtmp = a[j][i] * a[i][k]
				_bncavx512_rtd_mul(dtmp512, aji512, aik512);

				index_jk = j * a->real_col_dim + k;
				// Load a[j][k] elements
				ajk512[0] = _mm512_load_pd(&(a->element[0][index_jk]));
				ajk512[1] = _mm512_load_pd(&(a->element[1][index_jk]));
				ajk512[2] = _mm512_load_pd(&(a->element[2][index_jk]));
				
				// Subtract: dtmp = a[j][k] - dtmp
				_bncavx512_rtd_sub(dtmp512, ajk512, dtmp512);

				// Store result back to a[j][k]
				_mm512_store_pd(&(a->element[0][index_jk]), dtmp512[0]);
				_mm512_store_pd(&(a->element[1][index_jk]), dtmp512[1]);
				_mm512_store_pd(&(a->element[2][index_jk]), dtmp512[2]);
			}
		}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
		// Neon processes 2 doubles at a time (_BNC_D_WIDTH_NEON should be 2)
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->real_col_dim;

		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->real_col_dim + i;
			// Broadcast a[j][i] to all elements in Neon registers
			aji_neon_0 = svdup_f64(a->element[0][index_ji]);
			aji_neon_1 = svdup_f64(a->element[1][index_ji]);
			aji_neon_2 = svdup_f64(a->element[2][index_ji]);

			// head: process elements before aligned boundary
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rtd_mul(dtmp1, get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, k));
				rtd_sub(dtmp, get_tdmatrix_ij(a, j, k), dtmp1);
				set_tdmatrix_ij(a, j, k, dtmp);
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
				
				// Multiply: dtmp = a[j][i] * a[i][k]
				_bncsve2_rtd_mul(pg, &dtmp_neon_0, &dtmp_neon_1, &dtmp_neon_2, aji_neon_0, aji_neon_1, aji_neon_2, aik_neon_0, aik_neon_1, aik_neon_2);

				index_jk = j * a->real_col_dim + k;
				// Load a[j][k] elements
				ajk_neon_0 = svld1_f64(pg, &(a->element[0][index_jk]));
				ajk_neon_1 = svld1_f64(pg, &(a->element[1][index_jk]));
				ajk_neon_2 = svld1_f64(pg, &(a->element[2][index_jk]));
				
				// Subtract: dtmp = a[j][k] - dtmp
				_bncsve2_rtd_neg(pg, &dtmp_neon_0, &dtmp_neon_1, &dtmp_neon_2, dtmp_neon_0, dtmp_neon_1, dtmp_neon_2);
		_bncsve2_rtd_add(pg, &dtmp_neon_0, &dtmp_neon_1, &dtmp_neon_2, ajk_neon_0, ajk_neon_1, ajk_neon_2, dtmp_neon_0, dtmp_neon_1, dtmp_neon_2);

				// Store result back to a[j][k]
				svst1_f64(pg, &(a->element[0][index_jk]), dtmp_neon_0);
				svst1_f64(pg, &(a->element[1][index_jk]), dtmp_neon_1);
				svst1_f64(pg, &(a->element[2][index_jk]), dtmp_neon_2);
			}
		}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
		// Neon processes 2 doubles at a time (_BNC_D_WIDTH_NEON should be 2)
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->real_col_dim;

		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->real_col_dim + i;
			// Broadcast a[j][i] to all elements in Neon registers
			aji_neon[0] = vdupq_n_f64(a->element[0][index_ji]);
			aji_neon[1] = vdupq_n_f64(a->element[1][index_ji]);
			aji_neon[2] = vdupq_n_f64(a->element[2][index_ji]);

			// head: process elements before aligned boundary
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rtd_mul(dtmp1, get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, k));
				rtd_sub(dtmp, get_tdmatrix_ij(a, j, k), dtmp1);
				set_tdmatrix_ij(a, j, k, dtmp);
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
				
				// Multiply: dtmp = a[j][i] * a[i][k]
				_bncneon_rtd_mul(dtmp_neon, aji_neon, aik_neon);

				index_jk = j * a->real_col_dim + k;
				// Load a[j][k] elements
				ajk_neon[0] = vld1q_f64(&(a->element[0][index_jk]));
				ajk_neon[1] = vld1q_f64(&(a->element[1][index_jk]));
				ajk_neon[2] = vld1q_f64(&(a->element[2][index_jk]));
				
				// Subtract: dtmp = a[j][k] - dtmp
				_bncneon_rtd_sub(dtmp_neon, ajk_neon, dtmp_neon);

				// Store result back to a[j][k]
				vst1q_f64(&(a->element[0][index_jk]), dtmp_neon[0]);
				vst1q_f64(&(a->element[1][index_jk]), dtmp_neon[1]);
				vst1q_f64(&(a->element[2][index_jk]), dtmp_neon[2]);
			}
		}
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

	
#ifdef BNC_USE_NEW_FMA
/* fused branch-free FMA + SIMD forward/backward substitution */
	for(i = 0; i < dim; i++)
	{
		rtd_abs(dtmp, get_tdmatrix_ij(lu, i, i));
		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveTDLSP, %ld)\n", i);
			return -1;
		}
	}

/* Forward (row-oriented) */
	for(i = 1; i < dim; i++)
	{
		_bnc_tdsolve_dot(dtmp1, lu, i, 0, i, answer);
		rtd_sub(dtmp, get_tdvector_i(answer, i), dtmp1);
		set_tdvector_i(answer, i, dtmp);
	}

/* Backward (row-oriented) */
	for(i = (dim - 1); i >= 0; i--)
	{
		_bnc_tdsolve_dot(dtmp1, lu, i, i + 1, dim, answer);
		rtd_sub(dtmp, get_tdvector_i(answer, i), dtmp1);
		rtd_div(dtmp, dtmp, get_tdmatrix_ij(lu, i, i));
		set_tdvector_i(answer, i, dtmp);
	}
#else // BNC_USE_NEW_FMA
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

#endif // BNC_USE_NEW_FMA
	return 0;
}

#ifdef __cplusplus
} // extern "C"
#endif