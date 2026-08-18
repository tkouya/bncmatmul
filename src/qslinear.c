/********************************************************************************/
/* qslinear.c: Quadruple Single precision Linear Computation Library            */
/* Copyright (C) 2020 Tomonori Kouya                                            */
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
//#include "rds.h"
#include "qslinear.h"

#ifdef BNC_USE_NEW_FMA
/************************************************************/
/* sum := sum_{k=start}^{end-1} lu(row, k) * v(k)           */
/* (fused branch-free FMA dot product for LU solvers)       */
/************************************************************/
static void _bnc_qssolve_dot(float sum[QSSIZE], QSMatrix lu, long int row, long int start, long int end, QSVector v)
{
	long int k;
	float rtmp[QSSIZE], vtmp[QSSIZE];
	float *row_e[QSSIZE], *vec_e[QSSIZE];

	row_e[0] = &(lu->element[0][row * lu->real_col_dim]);
	row_e[1] = &(lu->element[1][row * lu->real_col_dim]);
	row_e[2] = &(lu->element[2][row * lu->real_col_dim]);
	row_e[3] = &(lu->element[3][row * lu->real_col_dim]);
	vec_e[0] = &(v->element[0][0]);
	vec_e[1] = &(v->element[1][0]);
	vec_e[2] = &(v->element[2][0]);
	vec_e[3] = &(v->element[3][0]);
	sum[0] = (float)0.0;
	sum[1] = (float)0.0;
	sum[2] = (float)0.0;
	sum[3] = (float)0.0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	{
		__m256 vacc[QSSIZE], vr[QSSIZE], vv[QSSIZE];
		float red[QSSIZE][16];
		long int k_simd_end = start + ((end - start) / _BNC_S_WIDTH) * _BNC_S_WIDTH;
		long int w;

		vacc[0] = _mm256_setzero_ps();
		vacc[1] = _mm256_setzero_ps();
		vacc[2] = _mm256_setzero_ps();
		vacc[3] = _mm256_setzero_ps();
		for(k = start; k < k_simd_end; k += _BNC_S_WIDTH)
		{
			vr[0] = _mm256_loadu_ps(&(row_e[0][k]));
			vr[1] = _mm256_loadu_ps(&(row_e[1][k]));
			vr[2] = _mm256_loadu_ps(&(row_e[2][k]));
			vr[3] = _mm256_loadu_ps(&(row_e[3][k]));
			vv[0] = _mm256_loadu_ps(&(vec_e[0][k]));
			vv[1] = _mm256_loadu_ps(&(vec_e[1][k]));
			vv[2] = _mm256_loadu_ps(&(vec_e[2][k]));
			vv[3] = _mm256_loadu_ps(&(vec_e[3][k]));
			_bncavx2_qwfmaf(vacc, vr, vv, vacc);
		}
		_mm256_storeu_ps(&(red[0][0]), vacc[0]);
		_mm256_storeu_ps(&(red[1][0]), vacc[1]);
		_mm256_storeu_ps(&(red[2][0]), vacc[2]);
		_mm256_storeu_ps(&(red[3][0]), vacc[3]);
		for(w = 0; w < _BNC_S_WIDTH; w++)
		{
			vtmp[0] = red[0][w];
			vtmp[1] = red[1][w];
			vtmp[2] = red[2][w];
			vtmp[3] = red[3][w];
			rqs_add(sum, sum, vtmp);
		}
		for(k = k_simd_end; k < end; k++)
		{
			rtmp[0] = row_e[0][k];
			rtmp[1] = row_e[1][k];
			rtmp[2] = row_e[2][k];
			rtmp[3] = row_e[3][k];
			vtmp[0] = vec_e[0][k];
			vtmp[1] = vec_e[1][k];
			vtmp[2] = vec_e[2][k];
			vtmp[3] = vec_e[3][k];
			rqs_fma(sum, rtmp, vtmp, sum);
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	{
		__m512 vacc[QSSIZE], vr[QSSIZE], vv[QSSIZE];
		float red[QSSIZE][16];
		long int k_simd_end = start + ((end - start) / _BNC_S_WIDTH) * _BNC_S_WIDTH;
		long int w;

		vacc[0] = _mm512_setzero_ps();
		vacc[1] = _mm512_setzero_ps();
		vacc[2] = _mm512_setzero_ps();
		vacc[3] = _mm512_setzero_ps();
		for(k = start; k < k_simd_end; k += _BNC_S_WIDTH)
		{
			vr[0] = _mm512_loadu_ps(&(row_e[0][k]));
			vr[1] = _mm512_loadu_ps(&(row_e[1][k]));
			vr[2] = _mm512_loadu_ps(&(row_e[2][k]));
			vr[3] = _mm512_loadu_ps(&(row_e[3][k]));
			vv[0] = _mm512_loadu_ps(&(vec_e[0][k]));
			vv[1] = _mm512_loadu_ps(&(vec_e[1][k]));
			vv[2] = _mm512_loadu_ps(&(vec_e[2][k]));
			vv[3] = _mm512_loadu_ps(&(vec_e[3][k]));
			_bncavx512_qwfmaf(vacc, vr, vv, vacc);
		}
		_mm512_storeu_ps(&(red[0][0]), vacc[0]);
		_mm512_storeu_ps(&(red[1][0]), vacc[1]);
		_mm512_storeu_ps(&(red[2][0]), vacc[2]);
		_mm512_storeu_ps(&(red[3][0]), vacc[3]);
		for(w = 0; w < _BNC_S_WIDTH; w++)
		{
			vtmp[0] = red[0][w];
			vtmp[1] = red[1][w];
			vtmp[2] = red[2][w];
			vtmp[3] = red[3][w];
			rqs_add(sum, sum, vtmp);
		}
		for(k = k_simd_end; k < end; k++)
		{
			rtmp[0] = row_e[0][k];
			rtmp[1] = row_e[1][k];
			rtmp[2] = row_e[2][k];
			rtmp[3] = row_e[3][k];
			vtmp[0] = vec_e[0][k];
			vtmp[1] = vec_e[1][k];
			vtmp[2] = vec_e[2][k];
			vtmp[3] = vec_e[3][k];
			rqs_fma(sum, rtmp, vtmp, sum);
		}
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	{
		svbool_t pg = svptrue_b32();
		svfloat32_t acc0, acc1, acc2, acc3;
		svfloat32_t r0, r1, r2, r3;
		svfloat32_t v0, v1, v2, v3;
		float red[QSSIZE][64];
		long int vl = (long int)svcntw();
		long int k_simd_end = start + ((end - start) / vl) * vl;
		long int w;

		acc0 = svdup_f32(0.0);
		acc1 = svdup_f32(0.0);
		acc2 = svdup_f32(0.0);
		acc3 = svdup_f32(0.0);
		for(k = start; k < k_simd_end; k += vl)
		{
			r0 = svld1_f32(pg, &(row_e[0][k]));
			r1 = svld1_f32(pg, &(row_e[1][k]));
			r2 = svld1_f32(pg, &(row_e[2][k]));
			r3 = svld1_f32(pg, &(row_e[3][k]));
			v0 = svld1_f32(pg, &(vec_e[0][k]));
			v1 = svld1_f32(pg, &(vec_e[1][k]));
			v2 = svld1_f32(pg, &(vec_e[2][k]));
			v3 = svld1_f32(pg, &(vec_e[3][k]));
			_bncsve2_qwfmaf(pg, &acc0, &acc1, &acc2, &acc3, r0, r1, r2, r3, v0, v1, v2, v3, acc0, acc1, acc2, acc3);
		}
		svst1_f32(pg, &(red[0][0]), acc0);
		svst1_f32(pg, &(red[1][0]), acc1);
		svst1_f32(pg, &(red[2][0]), acc2);
		svst1_f32(pg, &(red[3][0]), acc3);
		for(w = 0; w < vl; w++)
		{
			vtmp[0] = red[0][w];
			vtmp[1] = red[1][w];
			vtmp[2] = red[2][w];
			vtmp[3] = red[3][w];
			rqs_add(sum, sum, vtmp);
		}
		for(k = k_simd_end; k < end; k++)
		{
			rtmp[0] = row_e[0][k];
			rtmp[1] = row_e[1][k];
			rtmp[2] = row_e[2][k];
			rtmp[3] = row_e[3][k];
			vtmp[0] = vec_e[0][k];
			vtmp[1] = vec_e[1][k];
			vtmp[2] = vec_e[2][k];
			vtmp[3] = vec_e[3][k];
			rqs_fma(sum, rtmp, vtmp, sum);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	{
		float32x4_t vacc[QSSIZE], vr[QSSIZE], vv[QSSIZE];
		float red[QSSIZE][16];
		long int k_simd_end = start + ((end - start) / _BNC_S_WIDTH) * _BNC_S_WIDTH;
		long int w;

		vacc[0] = vdupq_n_f32(0.0f);
		vacc[1] = vdupq_n_f32(0.0f);
		vacc[2] = vdupq_n_f32(0.0f);
		vacc[3] = vdupq_n_f32(0.0f);
		for(k = start; k < k_simd_end; k += _BNC_S_WIDTH)
		{
			vr[0] = vld1q_f32(&(row_e[0][k]));
			vr[1] = vld1q_f32(&(row_e[1][k]));
			vr[2] = vld1q_f32(&(row_e[2][k]));
			vr[3] = vld1q_f32(&(row_e[3][k]));
			vv[0] = vld1q_f32(&(vec_e[0][k]));
			vv[1] = vld1q_f32(&(vec_e[1][k]));
			vv[2] = vld1q_f32(&(vec_e[2][k]));
			vv[3] = vld1q_f32(&(vec_e[3][k]));
			_bncneon_qwfmaf(vacc, vr, vv, vacc);
		}
		vst1q_f32(&(red[0][0]), vacc[0]);
		vst1q_f32(&(red[1][0]), vacc[1]);
		vst1q_f32(&(red[2][0]), vacc[2]);
		vst1q_f32(&(red[3][0]), vacc[3]);
		for(w = 0; w < _BNC_S_WIDTH; w++)
		{
			vtmp[0] = red[0][w];
			vtmp[1] = red[1][w];
			vtmp[2] = red[2][w];
			vtmp[3] = red[3][w];
			rqs_add(sum, sum, vtmp);
		}
		for(k = k_simd_end; k < end; k++)
		{
			rtmp[0] = row_e[0][k];
			rtmp[1] = row_e[1][k];
			rtmp[2] = row_e[2][k];
			rtmp[3] = row_e[3][k];
			vtmp[0] = vec_e[0][k];
			vtmp[1] = vec_e[1][k];
			vtmp[2] = vec_e[2][k];
			vtmp[3] = vec_e[3][k];
			rqs_fma(sum, rtmp, vtmp, sum);
		}
	}
#else // others
	for(k = start; k < end; k++)
	{
		rtmp[0] = row_e[0][k];
		rtmp[1] = row_e[1][k];
		rtmp[2] = row_e[2][k];
		rtmp[3] = row_e[3][k];
		vtmp[0] = vec_e[0][k];
		vtmp[1] = vec_e[1][k];
		vtmp[2] = vec_e[2][k];
		vtmp[3] = vec_e[3][k];
		rqs_fma(sum, rtmp, vtmp, sum);
	}
#endif // __AVX2__
}
#endif // BNC_USE_NEW_FMA


#ifdef USE_GMP
#include "gmp.h"
#include "mpfr.h"
#include "mpfr_dtq_sd.h"
#include "mpflinear.h"
#endif //USE_GMP//


#if defined(USE_GMP) && defined(USE_MPFR)
// Frobenius norm
qsfloat qsnormf(qsfloat array[], int dim)
{
    int i;
    qsfloat ret, tmp;
    mpfr_t mpfr_ret;

    rqs_set_ui(ret.val, 0UL);
    for(i = 0; i < dim; i++)
    {
        rqs_mul(tmp.val, array[i].val, array[i].val);
        rqs_add(ret.val, ret.val, tmp.val);
    }
    //printf("ret.val = "); rqs_out_str(ret.val); printf("\n");
//  rqs_sqrt(ret, ret);
    mpfr_init2(mpfr_ret, 128);
    mpfr_set_qs(mpfr_ret, ret.val, MPFR_RNDN);
    mpfr_sqrt(mpfr_ret, mpfr_ret, MPFR_RNDN);
    mpfr_get_qs(ret.val, mpfr_ret, MPFR_RNDN);
    mpfr_clear(mpfr_ret);
    return ret;
}

// generate a text vector: vec(i) := sqrt(sqrt_seed) * (i + 1)
/* void set_test_qsvector(qsfloat vec[], int sqrt_seed, int dim)
{
    int i, j;
    qsfloat ddsqrt;
    mpfr_t mpfrsqrt;

    // ddsqrt := sqrt(sqrt_seed)
    mpfr_init2(mpfrsqrt, 128);
    mpfr_sqrt_ui(mpfrsqrt, (unsigned long)sqrt_seed, MPFR_RNDN);
    mpfr_get_ds(ddsqrt.val, mpfrsqrt, MPFR_RNDN);
//    rqs_set_ui(ddsqrt.val, sqrt_seed);
    //rqs_sqrt(ddsqrt.val, ddsqrt.val);
    //rqs_sqrt_ui(ddsqrt.val, sqrt_seed);

    //printf("set_test_qsmatrix: coef = "); rqs_out_str(ddsqrt.val); printf("\n");

    for(i = 0; i < dim; i++)
    {
        //printf("%5d: ", i);
        rqs_set_ui(vec[i].val, i + 1);
        rqs_mul(vec[i].val, vec[i].val, ddsqrt.val);
    }
} */

//#if 0
// qsrel_diff
qsfloat qsrel_diff(qsfloat a, qsfloat b)
{
    qsfloat rel_diff, abs_a;

    //rel_diff = fabs(a - b);
    rqs_sub(rel_diff.val, a.val, b.val);
    rqs_abs(rel_diff.val, rel_diff.val);

    //if(a != 0.0)
    if(rqs_cmp_ui(a.val, 0UL) != 0)
    {
//        rel_diff /= fabs(a);
        rqs_abs(abs_a.val, a.val);
        rqs_div(rel_diff.val, rel_diff.val, a.val);
    }

    return rel_diff;
}
//#endif // 0

qsfloat qsrel_diff_array(qsfloat approx_a[], qsfloat approx_b[], int dim, int print_flag)
{
    int i;
    qsfloat rel_min, rel_max, rel_ave, rel_diff;

    rel_diff = qsrel_diff(approx_a[0], approx_b[0]);
    rqs_set(rel_min.val, rel_diff.val);
    rqs_set(rel_max.val, rel_diff.val);
    rqs_set(rel_ave.val, rel_diff.val);

    for(i = 1; i < dim; i++)
    {
        rel_diff = qsrel_diff(approx_a[i], approx_b[i]);
        if(rqs_cmp(rel_diff.val, rel_min.val) < 0) rqs_set(rel_min.val, rel_diff.val);
        if(rqs_cmp(rel_diff.val, rel_max.val) > 0) rqs_set(rel_max.val, rel_diff.val);
        //rel_ave += rel_diff;
        rqs_add(rel_ave.val, rel_ave.val, rel_diff.val);
    }
    //rel_ave /= (qdfloat)dim;
    rqs_div_ui(rel_ave.val, rel_ave.val, (unsigned long)dim);

    if(print_flag == 1)
    {
        printf("max_rel_diff, min_rel_diff, ave_rel_diff:"); rqs_out_str(rel_max.val); printf(" "); rqs_out_str(rel_min.val);  printf(" "); rqs_out_str(rel_ave.val); printf("\n"); 
    }

    return rel_max;
}
#endif // defined(USE_GMP) && defined(USE_MPFR)

// initialize QSVector
QSVector init_qsvector(long int dimension)
{
	QSVector ret = NULL;
	long int i, real_dim;

	if(dimension <= 0)
	{
		fprintf(stderr, "ERROR: init_qsvector\n");
		return ret;
	}

	ret = (QSVector)BNC_MALLOC(sizeof(qsvector));
	if(ret == NULL)
		return ret;

	// real_dim is the nearest positive multiplier of _BNC_S_WIDTH
	real_dim = (long int)ceil((float)(dimension) / (float)_BNC_S_WIDTH) * _BNC_S_WIDTH;

	ret->element[0] = (float *)BNC_CALLOC(real_dim, sizeof(float));
	if(ret->element[0] == NULL)
	{ 	free(ret);
		return NULL;
	}
	ret->element[1] = (float *)BNC_CALLOC(real_dim, sizeof(float));
	if(ret->element[1] == NULL)
	{
		free(ret->element[0]);
		free(ret);
		return NULL;
	}
	ret->element[2] = (float *)BNC_CALLOC(real_dim, sizeof(float));
	if(ret->element[2] == NULL)
	{
		free(ret->element[0]);
		free(ret->element[1]);
		free(ret);
		return NULL;
	}
	ret->element[3] = (float *)BNC_CALLOC(real_dim, sizeof(float));
	if(ret->element[3] == NULL)
	{
		free(ret->element[0]);
		free(ret->element[1]);
		free(ret->element[2]);
		free(ret);
		return NULL;
	}

	/* All 0 */
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 zero4;

	zero4 = _mm256_setzero_ps();
	for(i = 0; i < real_dim; i += _BNC_S_WIDTH)
	{
		_mm256_store_ps(&(ret->element[0][i]), zero4);
		_mm256_store_ps(&(ret->element[1][i]), zero4);
		_mm256_store_ps(&(ret->element[2][i]), zero4);
		_mm256_store_ps(&(ret->element[3][i]), zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero4;

	zero4 = _mm512_setzero_ps();
	for(i = 0; i < real_dim; i += _BNC_S_WIDTH)
	{
		_mm512_store_ps(&(ret->element[0][i]), zero4);
		_mm512_store_ps(&(ret->element[1][i]), zero4);
		_mm512_store_ps(&(ret->element[2][i]), zero4);
		_mm512_store_ps(&(ret->element[3][i]), zero4);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t zero4;

	zero4 = svdup_f32(0.0f);
	for(i = 0; i < real_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(real_dim));
		svst1_f32(pg, &(ret->element[0][i]), zero4);
		svst1_f32(pg, &(ret->element[1][i]), zero4);
		svst1_f32(pg, &(ret->element[2][i]), zero4);
		svst1_f32(pg, &(ret->element[3][i]), zero4);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t zero4;

	zero4 = vdupq_n_f32(0.0f);
	for(i = 0; i < real_dim; i += _BNC_S_WIDTH)
	{
		vst1q_f32(&(ret->element[0][i]), zero4);
		vst1q_f32(&(ret->element[1][i]), zero4);
		vst1q_f32(&(ret->element[2][i]), zero4);
		vst1q_f32(&(ret->element[3][i]), zero4);
	}
#else // others
	for(i = 0; i < dimension; i++)
	{
		ret->element[0][i] = 0.0f;
		ret->element[1][i] = 0.0f;
		ret->element[2][i] = 0.0f;
		ret->element[3][i] = 0.0f;
	}
#endif // __AVX2__

	ret->dim = dimension;
	ret->real_dim = real_dim;

	return ret;
}

// free QSVector
void free_qsvector(QSVector vec)
{
    long int i;
    for(i = 0; i < QSSIZE; i++)
        free(vec->element[i]);

    free(vec);
}

// QSVector vec -> qsfloat array
void set_qsfloat_qsvec(qsfloat ret[], int ret_dim, QSVector vec)
{
    int index, j, dim;

    dim = (ret_dim < vec->dim) ? ret_dim : vec->dim;

    for(index = 0; index < dim; index++)
    {
        for(j = 0; j < QSSIZE; j++)
            ret[index].val[j] = vec->element[j][index];
    }
}

// qsfloat array -> QSVector ret
void set_qsvector_qsfloat(QSVector ret, qsfloat array[], int array_dim)
{
    int index, j, dim;

    dim = (ret->dim < array_dim) ? ret->dim : array_dim;

    for(index = 0; index < dim; index++)
    {
        for(j = 0; j < QSSIZE; j++)
            ret->element[j][index] = array[index].val[j];
    }
}

// print qsvector
void print_qsvector(QSVector vec)
{
	long int index;

	for(index = 0; index < vec->dim; index++)
	{
		printf("%4ld: ", index);
		//c_ds_write((vec->element + index * QSSIZE));
		RQS_OUT_STR(GET_QSVECTOR_I(vec, index));
	}
}

// set a zero vector
void set0_qsvector(QSVector vec)
{
	long int i;

	/* All 0 */
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 zero4;

	zero4 = _mm256_setzero_ps();
	for(i = 0; i < vec->real_dim; i += _BNC_S_WIDTH)
	{
		_mm256_store_ps(&vec->element[0][i], zero4);
		_mm256_store_ps(&vec->element[1][i], zero4);
		_mm256_store_ps(&vec->element[2][i], zero4);
		_mm256_store_ps(&vec->element[3][i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero4;

	zero4 = _mm512_setzero_ps();
	for(i = 0; i < vec->real_dim; i += _BNC_S_WIDTH)
	{
		_mm512_store_ps(&vec->element[0][i], zero4);
		_mm512_store_ps(&vec->element[1][i], zero4);
		_mm512_store_ps(&vec->element[2][i], zero4);
		_mm512_store_ps(&vec->element[3][i], zero4);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t zero4;

	zero4 = svdup_f32(0.0f);
	for(i = 0; i < vec->real_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(vec->real_dim));
		svst1_f32(pg, &vec->element[0][i], zero4);
		svst1_f32(pg, &vec->element[1][i], zero4);
		svst1_f32(pg, &vec->element[2][i], zero4);
		svst1_f32(pg, &vec->element[3][i], zero4);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t zero4;

	zero4 = vdupq_n_f32(0.0f);
	for(i = 0; i < vec->real_dim; i += _BNC_S_WIDTH)
	{
		vst1q_f32(&vec->element[0][i], zero4);
		vst1q_f32(&vec->element[1][i], zero4);
		vst1q_f32(&vec->element[2][i], zero4);
		vst1q_f32(&vec->element[3][i], zero4);
	}
#else // others
	for(i = 0; i < vec->dim; i++)
	{
		vec->element[0][i] = 0.0f;
		vec->element[1][i] = 0.0f;
		vec->element[2][i] = 0.0f;
		vec->element[3][i] = 0.0f;
	}
#endif // __AVX2__
}

// set_qsvector_i_str
void set_qsvector_i_str(QSVector vec, long int index, const char *str)
{
	float tmp[QSSIZE];

	//rqs_get_str(tmp, str);
	rqs_set_str(tmp, str);

	set_qsvector_i(vec, index, tmp);
}

/*************************************************/
/* Vector Calculations for QSVector               */
/*
void add_qsvector(QSVector c, QSVector a, QSVector b)
void add2_qsvector(QSVector c, QSVector a)
void sub_qsvector(QSVector c, QSVector a, QSVector b)
void sub2_qsvector(QSVector c, DVector a)
void cmul_qsvector(QSVector c, float val[QSSIZE], QSVector a)
void cmul2_qsvector(QSVector c, float val[QSSIZE])
void add_cmul_qsvector(QSVector c, QSVector a, float val[QSSIZE], QSVector b)
float ip_qsvector(QSVector a, QSVector b)
float norm1_qsvector(QSVector a)
float norm2_qsvector(QSVector a)
float normi_qsvector(QSVector a)
void subst_qsvector(QSVector c, QSVector a)
*/
/*************************************************/
/* c = a + b */
void add_qsvector(QSVector c, QSVector a, QSVector b)
{
    long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_qsvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256 in_ret[QSSIZE], in_a_val[QSSIZE], in_b_val[QSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_ps(&(a->element[1][index]));
        in_a_val[2] = _mm256_load_ps(&(a->element[2][index]));
        in_a_val[3] = _mm256_load_ps(&(a->element[3][index]));
        in_b_val[0] = _mm256_load_ps(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_ps(&(b->element[1][index]));
        in_b_val[2] = _mm256_load_ps(&(b->element[2][index]));
        in_b_val[3] = _mm256_load_ps(&(b->element[3][index]));

        _bncavx2_rqs_add(in_ret, in_a_val, in_b_val);

        _mm256_store_ps(&(c->element[0][index]), in_ret[0]);
        _mm256_store_ps(&(c->element[1][index]), in_ret[1]);
        _mm256_store_ps(&(c->element[2][index]), in_ret[2]);
        _mm256_store_ps(&(c->element[3][index]), in_ret[3]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512 in_ret[QSSIZE], in_a_val[QSSIZE], in_b_val[QSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm512_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_ps(&(a->element[1][index]));
        in_a_val[2] = _mm512_load_ps(&(a->element[2][index]));
        in_a_val[3] = _mm512_load_ps(&(a->element[3][index]));
        in_b_val[0] = _mm512_load_ps(&(b->element[0][index]));
        in_b_val[1] = _mm512_load_ps(&(b->element[1][index]));
        in_b_val[2] = _mm512_load_ps(&(b->element[2][index]));
        in_b_val[3] = _mm512_load_ps(&(b->element[3][index]));

        _bncavx512_rqs_add(in_ret, in_a_val, in_b_val);

        _mm512_store_ps(&(c->element[0][index]), in_ret[0]);
        _mm512_store_ps(&(c->element[1][index]), in_ret[1]);
        _mm512_store_ps(&(c->element[2][index]), in_ret[2]);
        _mm512_store_ps(&(c->element[3][index]), in_ret[3]);
   }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
    
	svfloat32_t in_ret_0, in_ret_1, in_ret_2, in_ret_3;
	svfloat32_t in_a_val_0, in_a_val_1, in_a_val_2, in_a_val_3;
	svfloat32_t in_b_val_0, in_b_val_1, in_b_val_2, in_b_val_3;

    for(index = 0; index < c->real_dim; index += (long int)svcntw())
    {
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(c->real_dim));
        in_a_val_0 = svld1_f32(pg, &(a->element[0][index]));
        in_a_val_1 = svld1_f32(pg, &(a->element[1][index]));
        in_a_val_2 = svld1_f32(pg, &(a->element[2][index]));
        in_a_val_3 = svld1_f32(pg, &(a->element[3][index]));
        in_b_val_0 = svld1_f32(pg, &(b->element[0][index]));
        in_b_val_1 = svld1_f32(pg, &(b->element[1][index]));
        in_b_val_2 = svld1_f32(pg, &(b->element[2][index]));
        in_b_val_3 = svld1_f32(pg, &(b->element[3][index]));

        _bncsve2_rqs_add(pg, &in_ret_0, &in_ret_1, &in_ret_2, &in_ret_3, in_a_val_0, in_a_val_1, in_a_val_2, in_a_val_3, in_b_val_0, in_b_val_1, in_b_val_2, in_b_val_3);

        svst1_f32(pg, &(c->element[0][index]), in_ret_0);
        svst1_f32(pg, &(c->element[1][index]), in_ret_1);
        svst1_f32(pg, &(c->element[2][index]), in_ret_2);
        svst1_f32(pg, &(c->element[3][index]), in_ret_3);
   }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float32x4_t in_ret[QSSIZE], in_a_val[QSSIZE], in_b_val[QSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = vld1q_f32(&(a->element[0][index]));
        in_a_val[1] = vld1q_f32(&(a->element[1][index]));
        in_a_val[2] = vld1q_f32(&(a->element[2][index]));
        in_a_val[3] = vld1q_f32(&(a->element[3][index]));
        in_b_val[0] = vld1q_f32(&(b->element[0][index]));
        in_b_val[1] = vld1q_f32(&(b->element[1][index]));
        in_b_val[2] = vld1q_f32(&(b->element[2][index]));
        in_b_val[3] = vld1q_f32(&(b->element[3][index]));

        _bncneon_rqs_add(in_ret, in_a_val, in_b_val);

        vst1q_f32(&(c->element[0][index]), in_ret[0]);
        vst1q_f32(&(c->element[1][index]), in_ret[1]);
        vst1q_f32(&(c->element[2][index]), in_ret[2]);
        vst1q_f32(&(c->element[3][index]), in_ret[3]);
   }
#else // others
	float tmp[QSSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rqs_add(tmp, get_qsvector_i(a, i),  get_qsvector_i(b, i));
		set_qsvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c += a */
void add2_qsvector(QSVector c, QSVector a)
{
	long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: add2_qsvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256 in_ret[QSSIZE], in_a_val[QSSIZE], tmp4[QSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_ps(&(a->element[1][index]));
        in_a_val[2] = _mm256_load_ps(&(a->element[2][index]));
        in_a_val[3] = _mm256_load_ps(&(a->element[3][index]));
		in_ret[0] = _mm256_load_ps(&(c->element[0][index]));
        in_ret[1] = _mm256_load_ps(&(c->element[1][index]));
        in_ret[2] = _mm256_load_ps(&(c->element[2][index]));
        in_ret[3] = _mm256_load_ps(&(c->element[3][index]));

        _bncavx2_rqs_add(tmp4, in_ret, in_a_val);

        _mm256_store_ps(&(c->element[0][index]), tmp4[0]);
        _mm256_store_ps(&(c->element[1][index]), tmp4[1]);
        _mm256_store_ps(&(c->element[2][index]), tmp4[2]);
        _mm256_store_ps(&(c->element[3][index]), tmp4[3]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512 in_ret[QSSIZE], in_a_val[QSSIZE], tmp4[QSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm512_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_ps(&(a->element[1][index]));
        in_a_val[2] = _mm512_load_ps(&(a->element[2][index]));
        in_a_val[3] = _mm512_load_ps(&(a->element[3][index]));
		in_ret[0] = _mm512_load_ps(&(c->element[0][index]));
        in_ret[1] = _mm512_load_ps(&(c->element[1][index]));
        in_ret[2] = _mm512_load_ps(&(c->element[2][index]));
        in_ret[3] = _mm512_load_ps(&(c->element[3][index]));

        _bncavx512_rqs_add(tmp4, in_ret, in_a_val);

        _mm512_store_ps(&(c->element[0][index]), tmp4[0]);
        _mm512_store_ps(&(c->element[1][index]), tmp4[1]);
        _mm512_store_ps(&(c->element[2][index]), tmp4[2]);
        _mm512_store_ps(&(c->element[3][index]), tmp4[3]);
   }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
    
	svfloat32_t in_ret_0, in_ret_1, in_ret_2, in_ret_3;
	svfloat32_t in_a_val_0, in_a_val_1, in_a_val_2, in_a_val_3;
	svfloat32_t tmp4_0, tmp4_1, tmp4_2, tmp4_3;

    for(index = 0; index < c->real_dim; index += (long int)svcntw())
    {
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(c->real_dim));
        in_a_val_0 = svld1_f32(pg, &(a->element[0][index]));
        in_a_val_1 = svld1_f32(pg, &(a->element[1][index]));
        in_a_val_2 = svld1_f32(pg, &(a->element[2][index]));
        in_a_val_3 = svld1_f32(pg, &(a->element[3][index]));
		in_ret_0 = svld1_f32(pg, &(c->element[0][index]));
        in_ret_1 = svld1_f32(pg, &(c->element[1][index]));
        in_ret_2 = svld1_f32(pg, &(c->element[2][index]));
        in_ret_3 = svld1_f32(pg, &(c->element[3][index]));

        _bncsve2_rqs_add(pg, &tmp4_0, &tmp4_1, &tmp4_2, &tmp4_3, in_ret_0, in_ret_1, in_ret_2, in_ret_3, in_a_val_0, in_a_val_1, in_a_val_2, in_a_val_3);

        svst1_f32(pg, &(c->element[0][index]), tmp4_0);
        svst1_f32(pg, &(c->element[1][index]), tmp4_1);
        svst1_f32(pg, &(c->element[2][index]), tmp4_2);
        svst1_f32(pg, &(c->element[3][index]), tmp4_3);
   }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float32x4_t in_ret[QSSIZE], in_a_val[QSSIZE], tmp4[QSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = vld1q_f32(&(a->element[0][index]));
        in_a_val[1] = vld1q_f32(&(a->element[1][index]));
        in_a_val[2] = vld1q_f32(&(a->element[2][index]));
        in_a_val[3] = vld1q_f32(&(a->element[3][index]));
		in_ret[0] = vld1q_f32(&(c->element[0][index]));
        in_ret[1] = vld1q_f32(&(c->element[1][index]));
        in_ret[2] = vld1q_f32(&(c->element[2][index]));
        in_ret[3] = vld1q_f32(&(c->element[3][index]));

        _bncneon_rqs_add(tmp4, in_ret, in_a_val);

        vst1q_f32(&(c->element[0][index]), tmp4[0]);
        vst1q_f32(&(c->element[1][index]), tmp4[1]);
        vst1q_f32(&(c->element[2][index]), tmp4[2]);
        vst1q_f32(&(c->element[3][index]), tmp4[3]);
   }
#else // others
	float tmp[QSSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rqs_add(tmp, get_qsvector_i(c, i), get_qsvector_i(a, i));
		set_qsvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c = a - b */
void sub_qsvector(QSVector c, QSVector a, QSVector b)
{
    long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_qsvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256 in_ret[QSSIZE], in_a_val[QSSIZE], in_b_val[QSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_ps(&(a->element[1][index]));
        in_a_val[2] = _mm256_load_ps(&(a->element[2][index]));
        in_a_val[3] = _mm256_load_ps(&(a->element[3][index]));
        in_b_val[0] = _mm256_load_ps(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_ps(&(b->element[1][index]));
        in_b_val[2] = _mm256_load_ps(&(b->element[2][index]));
        in_b_val[3] = _mm256_load_ps(&(b->element[3][index]));

        _bncavx2_rqs_sub(in_ret, in_a_val, in_b_val);

        _mm256_store_ps(&(c->element[0][index]), in_ret[0]);
        _mm256_store_ps(&(c->element[1][index]), in_ret[1]);
        _mm256_store_ps(&(c->element[2][index]), in_ret[2]);
        _mm256_store_ps(&(c->element[3][index]), in_ret[3]);
  }
#elif defined(__AVX512F__) // __AVX512F__
    __m512 in_ret[QSSIZE], in_a_val[QSSIZE], in_b_val[QSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm512_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_ps(&(a->element[1][index]));
        in_a_val[2] = _mm512_load_ps(&(a->element[2][index]));
        in_a_val[3] = _mm512_load_ps(&(a->element[3][index]));
        in_b_val[0] = _mm512_load_ps(&(b->element[0][index]));
        in_b_val[1] = _mm512_load_ps(&(b->element[1][index]));
        in_b_val[2] = _mm512_load_ps(&(b->element[2][index]));
        in_b_val[3] = _mm512_load_ps(&(b->element[3][index]));

        _bncavx512_rqs_sub(in_ret, in_a_val, in_b_val);

        _mm512_store_ps(&(c->element[0][index]), in_ret[0]);
        _mm512_store_ps(&(c->element[1][index]), in_ret[1]);
        _mm512_store_ps(&(c->element[2][index]), in_ret[2]);
        _mm512_store_ps(&(c->element[3][index]), in_ret[3]);
  }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
    
	svfloat32_t in_ret_0, in_ret_1, in_ret_2, in_ret_3;
	svfloat32_t in_a_val_0, in_a_val_1, in_a_val_2, in_a_val_3;
	svfloat32_t in_b_val_0, in_b_val_1, in_b_val_2, in_b_val_3;

    for(index = 0; index < c->real_dim; index += (long int)svcntw())
    {
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(c->real_dim));
        in_a_val_0 = svld1_f32(pg, &(a->element[0][index]));
        in_a_val_1 = svld1_f32(pg, &(a->element[1][index]));
        in_a_val_2 = svld1_f32(pg, &(a->element[2][index]));
        in_a_val_3 = svld1_f32(pg, &(a->element[3][index]));
        in_b_val_0 = svld1_f32(pg, &(b->element[0][index]));
        in_b_val_1 = svld1_f32(pg, &(b->element[1][index]));
        in_b_val_2 = svld1_f32(pg, &(b->element[2][index]));
        in_b_val_3 = svld1_f32(pg, &(b->element[3][index]));

        _bncsve2_rqs_neg(pg, &in_ret_0, &in_ret_1, &in_ret_2, &in_ret_3, in_b_val_0, in_b_val_1, in_b_val_2, in_b_val_3);
		_bncsve2_rqs_add(pg, &in_ret_0, &in_ret_1, &in_ret_2, &in_ret_3, in_a_val_0, in_a_val_1, in_a_val_2, in_a_val_3, in_ret_0, in_ret_1, in_ret_2, in_ret_3);

        svst1_f32(pg, &(c->element[0][index]), in_ret_0);
        svst1_f32(pg, &(c->element[1][index]), in_ret_1);
        svst1_f32(pg, &(c->element[2][index]), in_ret_2);
        svst1_f32(pg, &(c->element[3][index]), in_ret_3);
  }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float32x4_t in_ret[QSSIZE], in_a_val[QSSIZE], in_b_val[QSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = vld1q_f32(&(a->element[0][index]));
        in_a_val[1] = vld1q_f32(&(a->element[1][index]));
        in_a_val[2] = vld1q_f32(&(a->element[2][index]));
        in_a_val[3] = vld1q_f32(&(a->element[3][index]));
        in_b_val[0] = vld1q_f32(&(b->element[0][index]));
        in_b_val[1] = vld1q_f32(&(b->element[1][index]));
        in_b_val[2] = vld1q_f32(&(b->element[2][index]));
        in_b_val[3] = vld1q_f32(&(b->element[3][index]));

        _bncneon_rqs_sub(in_ret, in_a_val, in_b_val);

        vst1q_f32(&(c->element[0][index]), in_ret[0]);
        vst1q_f32(&(c->element[1][index]), in_ret[1]);
        vst1q_f32(&(c->element[2][index]), in_ret[2]);
        vst1q_f32(&(c->element[3][index]), in_ret[3]);
  }
#else // others
	float tmp[QSSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rqs_sub(tmp, get_qsvector_i(a, i),  get_qsvector_i(b, i));
		set_qsvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c -= a */
void sub2_qsvector(QSVector c, QSVector a)
{
	long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: add2_qsvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256 in_ret[QSSIZE], in_a_val[QSSIZE], tmp4[QSSIZE];

	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_ps(&(a->element[1][index]));
        in_a_val[2] = _mm256_load_ps(&(a->element[2][index]));
        in_a_val[3] = _mm256_load_ps(&(a->element[3][index]));
        in_ret[0] = _mm256_load_ps(&(c->element[0][index]));
        in_ret[1] = _mm256_load_ps(&(c->element[1][index]));
        in_ret[2] = _mm256_load_ps(&(c->element[2][index]));
        in_ret[3] = _mm256_load_ps(&(c->element[3][index]));

        _bncavx2_rqs_sub(tmp4, in_ret, in_a_val);

        _mm256_store_ps(&(c->element[0][index]), tmp4[0]);
        _mm256_store_ps(&(c->element[1][index]), tmp4[1]);
        _mm256_store_ps(&(c->element[2][index]), tmp4[2]);
        _mm256_store_ps(&(c->element[3][index]), tmp4[3]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512 in_ret[QSSIZE], in_a_val[QSSIZE], tmp4[QSSIZE];

	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm512_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_ps(&(a->element[1][index]));
        in_a_val[2] = _mm512_load_ps(&(a->element[2][index]));
        in_a_val[3] = _mm512_load_ps(&(a->element[3][index]));
        in_ret[0] = _mm512_load_ps(&(c->element[0][index]));
        in_ret[1] = _mm512_load_ps(&(c->element[1][index]));
        in_ret[2] = _mm512_load_ps(&(c->element[2][index]));
        in_ret[3] = _mm512_load_ps(&(c->element[3][index]));

        _bncavx512_rqs_sub(tmp4, in_ret, in_a_val);

        _mm512_store_ps(&(c->element[0][index]), tmp4[0]);
        _mm512_store_ps(&(c->element[1][index]), tmp4[1]);
        _mm512_store_ps(&(c->element[2][index]), tmp4[2]);
        _mm512_store_ps(&(c->element[3][index]), tmp4[3]);
   }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
    
	svfloat32_t in_ret_0, in_ret_1, in_ret_2, in_ret_3;
	svfloat32_t in_a_val_0, in_a_val_1, in_a_val_2, in_a_val_3;
	svfloat32_t tmp4_0, tmp4_1, tmp4_2, tmp4_3;

	for(index = 0; index < c->real_dim; index += (long int)svcntw())
    {
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(c->real_dim));
        in_a_val_0 = svld1_f32(pg, &(a->element[0][index]));
        in_a_val_1 = svld1_f32(pg, &(a->element[1][index]));
        in_a_val_2 = svld1_f32(pg, &(a->element[2][index]));
        in_a_val_3 = svld1_f32(pg, &(a->element[3][index]));
        in_ret_0 = svld1_f32(pg, &(c->element[0][index]));
        in_ret_1 = svld1_f32(pg, &(c->element[1][index]));
        in_ret_2 = svld1_f32(pg, &(c->element[2][index]));
        in_ret_3 = svld1_f32(pg, &(c->element[3][index]));

        _bncsve2_rqs_neg(pg, &tmp4_0, &tmp4_1, &tmp4_2, &tmp4_3, in_a_val_0, in_a_val_1, in_a_val_2, in_a_val_3);
		_bncsve2_rqs_add(pg, &tmp4_0, &tmp4_1, &tmp4_2, &tmp4_3, in_ret_0, in_ret_1, in_ret_2, in_ret_3, tmp4_0, tmp4_1, tmp4_2, tmp4_3);

        svst1_f32(pg, &(c->element[0][index]), tmp4_0);
        svst1_f32(pg, &(c->element[1][index]), tmp4_1);
        svst1_f32(pg, &(c->element[2][index]), tmp4_2);
        svst1_f32(pg, &(c->element[3][index]), tmp4_3);
   }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float32x4_t in_ret[QSSIZE], in_a_val[QSSIZE], tmp4[QSSIZE];

	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = vld1q_f32(&(a->element[0][index]));
        in_a_val[1] = vld1q_f32(&(a->element[1][index]));
        in_a_val[2] = vld1q_f32(&(a->element[2][index]));
        in_a_val[3] = vld1q_f32(&(a->element[3][index]));
        in_ret[0] = vld1q_f32(&(c->element[0][index]));
        in_ret[1] = vld1q_f32(&(c->element[1][index]));
        in_ret[2] = vld1q_f32(&(c->element[2][index]));
        in_ret[3] = vld1q_f32(&(c->element[3][index]));

        _bncneon_rqs_sub(tmp4, in_ret, in_a_val);

        vst1q_f32(&(c->element[0][index]), tmp4[0]);
        vst1q_f32(&(c->element[1][index]), tmp4[1]);
        vst1q_f32(&(c->element[2][index]), tmp4[2]);
        vst1q_f32(&(c->element[3][index]), tmp4[3]);
   }
#else // others
	float tmp[QSSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rqs_sub(tmp, get_qsvector_i(c, i), get_qsvector_i(a, i));
		set_qsvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c = val * a */
void cmul_qsvector(QSVector c, float val[QSSIZE], QSVector a)
{
    long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: cmul_dvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4[QSSIZE], c4[QSSIZE], val4[QSSIZE];

	val4[0] = _mm256_set1_ps(val[0]);
	val4[1] = _mm256_set1_ps(val[1]);
	val4[2] = _mm256_set1_ps(val[2]);
	val4[3] = _mm256_set1_ps(val[3]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		//set_qsvector_i(c, i, val * get_qsvector_i(a, i));
		a4[0] = _mm256_load_ps(&(a->element[0][index]));
		a4[1] = _mm256_load_ps(&(a->element[1][index]));
		a4[2] = _mm256_load_ps(&(a->element[2][index]));
		a4[3] = _mm256_load_ps(&(a->element[3][index]));

		_bncavx2_rqs_mul(c4, val4, a4);

		_mm256_store_ps(&(c->element[0][index]), c4[0]);
		_mm256_store_ps(&(c->element[1][index]), c4[1]);
		_mm256_store_ps(&(c->element[2][index]), c4[2]);
		_mm256_store_ps(&(c->element[3][index]), c4[3]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a4[QSSIZE], c4[QSSIZE], val4[QSSIZE];

	val4[0] = _mm512_set1_ps(val[0]);
	val4[1] = _mm512_set1_ps(val[1]);
	val4[2] = _mm512_set1_ps(val[2]);
	val4[3] = _mm512_set1_ps(val[3]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		//set_qsvector_i(c, i, val * get_qsvector_i(a, i));
		a4[0] = _mm512_load_ps(&(a->element[0][index]));
		a4[1] = _mm512_load_ps(&(a->element[1][index]));
		a4[2] = _mm512_load_ps(&(a->element[2][index]));
		a4[3] = _mm512_load_ps(&(a->element[3][index]));

		_bncavx512_rqs_mul(c4, val4, a4);

		_mm512_store_ps(&(c->element[0][index]), c4[0]);
		_mm512_store_ps(&(c->element[1][index]), c4[1]);
		_mm512_store_ps(&(c->element[2][index]), c4[2]);
		_mm512_store_ps(&(c->element[3][index]), c4[3]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t a4_0, a4_1, a4_2, a4_3;
	svfloat32_t c4_0, c4_1, c4_2, c4_3;
	svfloat32_t val4_0, val4_1, val4_2, val4_3;

	val4_0 = svdup_f32(val[0]);
	val4_1 = svdup_f32(val[1]);
	val4_2 = svdup_f32(val[2]);
	val4_3 = svdup_f32(val[3]);
	for(index = 0; index < c->real_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(c->real_dim));
		//set_qsvector_i(c, i, val * get_qsvector_i(a, i));
		a4_0 = svld1_f32(pg, &(a->element[0][index]));
		a4_1 = svld1_f32(pg, &(a->element[1][index]));
		a4_2 = svld1_f32(pg, &(a->element[2][index]));
		a4_3 = svld1_f32(pg, &(a->element[3][index]));

		_bncsve2_rqs_mul(pg, &c4_0, &c4_1, &c4_2, &c4_3, val4_0, val4_1, val4_2, val4_3, a4_0, a4_1, a4_2, a4_3);

		svst1_f32(pg, &(c->element[0][index]), c4_0);
		svst1_f32(pg, &(c->element[1][index]), c4_1);
		svst1_f32(pg, &(c->element[2][index]), c4_2);
		svst1_f32(pg, &(c->element[3][index]), c4_3);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a4[QSSIZE], c4[QSSIZE], val4[QSSIZE];

	val4[0] = vdupq_n_f32(val[0]);
	val4[1] = vdupq_n_f32(val[1]);
	val4[2] = vdupq_n_f32(val[2]);
	val4[3] = vdupq_n_f32(val[3]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		//set_qsvector_i(c, i, val * get_qsvector_i(a, i));
		a4[0] = vld1q_f32(&(a->element[0][index]));
		a4[1] = vld1q_f32(&(a->element[1][index]));
		a4[2] = vld1q_f32(&(a->element[2][index]));
		a4[3] = vld1q_f32(&(a->element[3][index]));

		_bncneon_rqs_mul(c4, val4, a4);

		vst1q_f32(&(c->element[0][index]), c4[0]);
		vst1q_f32(&(c->element[1][index]), c4[1]);
		vst1q_f32(&(c->element[2][index]), c4[2]);
		vst1q_f32(&(c->element[3][index]), c4[3]);
	}
#else // others
	float tmp[QSSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rqs_mul(tmp, val, get_qsvector_i(a, i));
		set_qsvector_i(c, i, tmp);
	}
#endif // __AVX2__

}

/* c *= val */
void cmul2_qsvector(QSVector c, float val[QSSIZE])
{
	long int i, index;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 c4[QSSIZE], val4[QSSIZE], tmp4[QSSIZE];

	val4[0] = _mm256_set1_ps(val[0]);
	val4[1] = _mm256_set1_ps(val[1]);
	val4[2] = _mm256_set1_ps(val[2]);
	val4[3] = _mm256_set1_ps(val[3]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		//set_qsvector_i(c, i, val * get_qsvector_i(a, i));
		c4[0] = _mm256_load_ps(&(c->element[0][index]));
		c4[1] = _mm256_load_ps(&(c->element[1][index]));
		c4[2] = _mm256_load_ps(&(c->element[2][index]));
		c4[3] = _mm256_load_ps(&(c->element[3][index]));

		_bncavx2_rqs_mul(tmp4, val4, c4);

		_mm256_store_ps(&(c->element[0][index]), tmp4[0]);
		_mm256_store_ps(&(c->element[1][index]), tmp4[1]);
		_mm256_store_ps(&(c->element[2][index]), tmp4[2]);
		_mm256_store_ps(&(c->element[3][index]), tmp4[3]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 c4[QSSIZE], val4[QSSIZE], tmp4[QSSIZE];

	val4[0] = _mm512_set1_ps(val[0]);
	val4[1] = _mm512_set1_ps(val[1]);
	val4[2] = _mm512_set1_ps(val[2]);
	val4[3] = _mm512_set1_ps(val[3]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		//set_qsvector_i(c, i, val * get_qsvector_i(a, i));
		c4[0] = _mm512_load_ps(&(c->element[0][index]));
		c4[1] = _mm512_load_ps(&(c->element[1][index]));
		c4[2] = _mm512_load_ps(&(c->element[2][index]));
		c4[3] = _mm512_load_ps(&(c->element[3][index]));

		_bncavx512_rqs_mul(tmp4, val4, c4);

		_mm512_store_ps(&(c->element[0][index]), tmp4[0]);
		_mm512_store_ps(&(c->element[1][index]), tmp4[1]);
		_mm512_store_ps(&(c->element[2][index]), tmp4[2]);
		_mm512_store_ps(&(c->element[3][index]), tmp4[3]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t c4_0, c4_1, c4_2, c4_3;
	svfloat32_t val4_0, val4_1, val4_2, val4_3;
	svfloat32_t tmp4_0, tmp4_1, tmp4_2, tmp4_3;

	val4_0 = svdup_f32(val[0]);
	val4_1 = svdup_f32(val[1]);
	val4_2 = svdup_f32(val[2]);
	val4_3 = svdup_f32(val[3]);
	for(index = 0; index < c->real_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(c->real_dim));
		//set_qsvector_i(c, i, val * get_qsvector_i(a, i));
		c4_0 = svld1_f32(pg, &(c->element[0][index]));
		c4_1 = svld1_f32(pg, &(c->element[1][index]));
		c4_2 = svld1_f32(pg, &(c->element[2][index]));
		c4_3 = svld1_f32(pg, &(c->element[3][index]));

		_bncsve2_rqs_mul(pg, &tmp4_0, &tmp4_1, &tmp4_2, &tmp4_3, val4_0, val4_1, val4_2, val4_3, c4_0, c4_1, c4_2, c4_3);

		svst1_f32(pg, &(c->element[0][index]), tmp4_0);
		svst1_f32(pg, &(c->element[1][index]), tmp4_1);
		svst1_f32(pg, &(c->element[2][index]), tmp4_2);
		svst1_f32(pg, &(c->element[3][index]), tmp4_3);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t c4[QSSIZE], val4[QSSIZE], tmp4[QSSIZE];

	val4[0] = vdupq_n_f32(val[0]);
	val4[1] = vdupq_n_f32(val[1]);
	val4[2] = vdupq_n_f32(val[2]);
	val4[3] = vdupq_n_f32(val[3]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		//set_qsvector_i(c, i, val * get_qsvector_i(a, i));
		c4[0] = vld1q_f32(&(c->element[0][index]));
		c4[1] = vld1q_f32(&(c->element[1][index]));
		c4[2] = vld1q_f32(&(c->element[2][index]));
		c4[3] = vld1q_f32(&(c->element[3][index]));

		_bncneon_rqs_mul(tmp4, val4, c4);

		vst1q_f32(&(c->element[0][index]), tmp4[0]);
		vst1q_f32(&(c->element[1][index]), tmp4[1]);
		vst1q_f32(&(c->element[2][index]), tmp4[2]);
		vst1q_f32(&(c->element[3][index]), tmp4[3]);
	}
#else // others
	float tmp[QSSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rqs_mul(tmp, val, get_qsvector_i(c, i));
		set_qsvector_i(c, i, tmp);
	}
#endif // __AVX2__	float tmp[QSSIZE];

}

/* c = a + val * b */
void add_cmul_qsvector(QSVector c, QSVector a, float val[QSSIZE], QSVector b)
{
	long int i, index;
	int _k;
	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_cmul_qsvector\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4[QSSIZE], b4[QSSIZE], c4[QSSIZE], val4[QSSIZE];
	for(_k = 0; _k < QSSIZE; _k++) val4[_k] = _mm256_set1_ps(val[_k]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		for(_k = 0; _k < QSSIZE; _k++){ a4[_k] = _mm256_load_ps(&(a->element[_k][index])); b4[_k] = _mm256_load_ps(&(b->element[_k][index])); }
		_bncavx2_rqs_mul(c4, val4, b4);
		_bncavx2_rqs_add(c4, a4, c4);
		for(_k = 0; _k < QSSIZE; _k++) _mm256_store_ps(&(c->element[_k][index]), c4[_k]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a4[QSSIZE], b4[QSSIZE], c4[QSSIZE], val4[QSSIZE];
	for(_k = 0; _k < QSSIZE; _k++) val4[_k] = _mm512_set1_ps(val[_k]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		for(_k = 0; _k < QSSIZE; _k++){ a4[_k] = _mm512_load_ps(&(a->element[_k][index])); b4[_k] = _mm512_load_ps(&(b->element[_k][index])); }
		_bncavx512_rqs_mul(c4, val4, b4);
		_bncavx512_rqs_add(c4, a4, c4);
		for(_k = 0; _k < QSSIZE; _k++) _mm512_store_ps(&(c->element[_k][index]), c4[_k]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t sa_0, sa_1, sa_2, sa_3, sb_0, sb_1, sb_2, sb_3, sc_0, sc_1, sc_2, sc_3, sv_0, sv_1, sv_2, sv_3;
	sv_0 = svdup_f32(val[0]); sv_1 = svdup_f32(val[1]); sv_2 = svdup_f32(val[2]); sv_3 = svdup_f32(val[3]);
	for(index = 0; index < c->real_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)c->real_dim);
		sa_0 = svld1_f32(pg, &(a->element[0][index])); sb_0 = svld1_f32(pg, &(b->element[0][index]));
		sa_1 = svld1_f32(pg, &(a->element[1][index])); sb_1 = svld1_f32(pg, &(b->element[1][index]));
		sa_2 = svld1_f32(pg, &(a->element[2][index])); sb_2 = svld1_f32(pg, &(b->element[2][index]));
		sa_3 = svld1_f32(pg, &(a->element[3][index])); sb_3 = svld1_f32(pg, &(b->element[3][index]));
		_bncsve2_rqs_mul(pg, &sc_0, &sc_1, &sc_2, &sc_3, sv_0, sv_1, sv_2, sv_3, sb_0, sb_1, sb_2, sb_3);
		_bncsve2_rqs_add(pg, &sc_0, &sc_1, &sc_2, &sc_3, sa_0, sa_1, sa_2, sa_3, sc_0, sc_1, sc_2, sc_3);
		svst1_f32(pg, &(c->element[0][index]), sc_0);
		svst1_f32(pg, &(c->element[1][index]), sc_1);
		svst1_f32(pg, &(c->element[2][index]), sc_2);
		svst1_f32(pg, &(c->element[3][index]), sc_3);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t na[QSSIZE], nb[QSSIZE], nc[QSSIZE], nv[QSSIZE];
	for(_k = 0; _k < QSSIZE; _k++) nv[_k] = vdupq_n_f32(val[_k]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		for(_k = 0; _k < QSSIZE; _k++){ na[_k] = vld1q_f32(&(a->element[_k][index])); nb[_k] = vld1q_f32(&(b->element[_k][index])); }
		_bncneon_rqs_mul(nc, nv, nb);
		_bncneon_rqs_add(nc, na, nc);
		for(_k = 0; _k < QSSIZE; _k++) vst1q_f32(&(c->element[_k][index]), nc[_k]);
	}
#else // scalar
	float tmp[QSSIZE];
	for(i = 0; i < c->dim; i++)
	{
		rqs_mul(tmp, val, get_qsvector_i(b, i));
		rqs_add(tmp, tmp, get_qsvector_i(a, i));
		set_qsvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

void sub_cmul_qsvector(QSVector c, QSVector a, float val[QSSIZE], QSVector b)
{
	long int i, index;
	int _k;
	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_cmul_qsvector\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4[QSSIZE], b4[QSSIZE], c4[QSSIZE], val4[QSSIZE];
	for(_k = 0; _k < QSSIZE; _k++) val4[_k] = _mm256_set1_ps(val[_k]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		for(_k = 0; _k < QSSIZE; _k++){ a4[_k] = _mm256_load_ps(&(a->element[_k][index])); b4[_k] = _mm256_load_ps(&(b->element[_k][index])); }
		_bncavx2_rqs_mul(c4, val4, b4);
		_bncavx2_rqs_sub(c4, a4, c4);
		for(_k = 0; _k < QSSIZE; _k++) _mm256_store_ps(&(c->element[_k][index]), c4[_k]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a4[QSSIZE], b4[QSSIZE], c4[QSSIZE], val4[QSSIZE];
	for(_k = 0; _k < QSSIZE; _k++) val4[_k] = _mm512_set1_ps(val[_k]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		for(_k = 0; _k < QSSIZE; _k++){ a4[_k] = _mm512_load_ps(&(a->element[_k][index])); b4[_k] = _mm512_load_ps(&(b->element[_k][index])); }
		_bncavx512_rqs_mul(c4, val4, b4);
		_bncavx512_rqs_sub(c4, a4, c4);
		for(_k = 0; _k < QSSIZE; _k++) _mm512_store_ps(&(c->element[_k][index]), c4[_k]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t sa_0, sa_1, sa_2, sa_3, sb_0, sb_1, sb_2, sb_3, sc_0, sc_1, sc_2, sc_3, sv_0, sv_1, sv_2, sv_3;
	sv_0 = svdup_f32(val[0]); sv_1 = svdup_f32(val[1]); sv_2 = svdup_f32(val[2]); sv_3 = svdup_f32(val[3]);
	for(index = 0; index < c->real_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)c->real_dim);
		sa_0 = svld1_f32(pg, &(a->element[0][index])); sb_0 = svld1_f32(pg, &(b->element[0][index]));
		sa_1 = svld1_f32(pg, &(a->element[1][index])); sb_1 = svld1_f32(pg, &(b->element[1][index]));
		sa_2 = svld1_f32(pg, &(a->element[2][index])); sb_2 = svld1_f32(pg, &(b->element[2][index]));
		sa_3 = svld1_f32(pg, &(a->element[3][index])); sb_3 = svld1_f32(pg, &(b->element[3][index]));
		_bncsve2_rqs_mul(pg, &sc_0, &sc_1, &sc_2, &sc_3, sv_0, sv_1, sv_2, sv_3, sb_0, sb_1, sb_2, sb_3);
		_bncsve2_rqs_neg(pg, &sc_0, &sc_1, &sc_2, &sc_3, sc_0, sc_1, sc_2, sc_3);
		_bncsve2_rqs_add(pg, &sc_0, &sc_1, &sc_2, &sc_3, sa_0, sa_1, sa_2, sa_3, sc_0, sc_1, sc_2, sc_3);
		svst1_f32(pg, &(c->element[0][index]), sc_0);
		svst1_f32(pg, &(c->element[1][index]), sc_1);
		svst1_f32(pg, &(c->element[2][index]), sc_2);
		svst1_f32(pg, &(c->element[3][index]), sc_3);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t na[QSSIZE], nb[QSSIZE], nc[QSSIZE], nv[QSSIZE];
	for(_k = 0; _k < QSSIZE; _k++) nv[_k] = vdupq_n_f32(val[_k]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		for(_k = 0; _k < QSSIZE; _k++){ na[_k] = vld1q_f32(&(a->element[_k][index])); nb[_k] = vld1q_f32(&(b->element[_k][index])); }
		_bncneon_rqs_mul(nc, nv, nb);
		_bncneon_rqs_sub(nc, na, nc);
		for(_k = 0; _k < QSSIZE; _k++) vst1q_f32(&(c->element[_k][index]), nc[_k]);
	}
#else // scalar
	float tmp[QSSIZE];
	for(i = 0; i < c->dim; i++)
	{
		rqs_mul(tmp, val, get_qsvector_i(b, i));
		rqs_sub(tmp, get_qsvector_i(a, i), tmp);
		set_qsvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* (a, b) */
void ip_qsvector(float ret[QSSIZE], QSVector a, QSVector b)
{
	long int i, index;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: ip_qsvector\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4[QSSIZE], b4[QSSIZE], ret4[QSSIZE], tmp4[QSSIZE];

	_bncavx2_set0_qs(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		a4[0] = _mm256_load_ps(&(a->element[0][index]));
		a4[1] = _mm256_load_ps(&(a->element[1][index]));
		a4[2] = _mm256_load_ps(&(a->element[2][index]));
		a4[3] = _mm256_load_ps(&(a->element[3][index]));
		b4[0] = _mm256_load_ps(&(b->element[0][index]));
		b4[1] = _mm256_load_ps(&(b->element[1][index]));
		b4[2] = _mm256_load_ps(&(b->element[2][index]));
		b4[3] = _mm256_load_ps(&(b->element[3][index]));

//		rqs_mul(tmp, get_qsvector_i(a, i), get_qsvector_i(b, i));
//		rqs_add(ret, ret, tmp);
		_bncavx2_rqs_mul(tmp4, a4, b4);
		_bncavx2_rqs_add(ret4, ret4, tmp4);
	}
	_bncavx2_rqs_sum256(ret, ret4);
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a4[QSSIZE], b4[QSSIZE], ret4[QSSIZE], tmp4[QSSIZE];

	_bncavx512_set0_qs(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		a4[0] = _mm512_load_ps(&(a->element[0][index]));
		a4[1] = _mm512_load_ps(&(a->element[1][index]));
		a4[2] = _mm512_load_ps(&(a->element[2][index]));
		a4[3] = _mm512_load_ps(&(a->element[3][index]));
		b4[0] = _mm512_load_ps(&(b->element[0][index]));
		b4[1] = _mm512_load_ps(&(b->element[1][index]));
		b4[2] = _mm512_load_ps(&(b->element[2][index]));
		b4[3] = _mm512_load_ps(&(b->element[3][index]));

//		rqs_mul(tmp, get_qsvector_i(a, i), get_qsvector_i(b, i));
//		rqs_add(ret, ret, tmp);
		_bncavx512_rqs_mul(tmp4, a4, b4);
		_bncavx512_rqs_add(ret4, ret4, tmp4);
	}
	_bncavx512_rqs_sum512(ret, ret4);
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t a4_0, a4_1, a4_2, a4_3;
	svfloat32_t b4_0, b4_1, b4_2, b4_3;
	svfloat32_t ret4_0, ret4_1, ret4_2, ret4_3;
	svfloat32_t tmp4_0, tmp4_1, tmp4_2, tmp4_3;

	_bncsve2_rqs_set0(&ret4_0, &ret4_1, &ret4_2, &ret4_3);
	for(index = 0; index < a->real_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(a->real_dim));
		a4_0 = svld1_f32(pg, &(a->element[0][index]));
		a4_1 = svld1_f32(pg, &(a->element[1][index]));
		a4_2 = svld1_f32(pg, &(a->element[2][index]));
		a4_3 = svld1_f32(pg, &(a->element[3][index]));
		b4_0 = svld1_f32(pg, &(b->element[0][index]));
		b4_1 = svld1_f32(pg, &(b->element[1][index]));
		b4_2 = svld1_f32(pg, &(b->element[2][index]));
		b4_3 = svld1_f32(pg, &(b->element[3][index]));

//		rqs_mul(tmp, get_qsvector_i(a, i), get_qsvector_i(b, i));
//		rqs_add(ret, ret, tmp);
		_bncsve2_rqs_mul(svptrue_b32(), &tmp4_0, &tmp4_1, &tmp4_2, &tmp4_3, a4_0, a4_1, a4_2, a4_3, b4_0, b4_1, b4_2, b4_3);
		_bncsve2_rqs_add(svptrue_b32(), &ret4_0, &ret4_1, &ret4_2, &ret4_3, ret4_0, ret4_1, ret4_2, ret4_3, tmp4_0, tmp4_1, tmp4_2, tmp4_3);
	}
	_bncsve2_rqs_sum128(svptrue_b32(), ret, ret4_0, ret4_1, ret4_2, ret4_3);
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a4[QSSIZE], b4[QSSIZE], ret4[QSSIZE], tmp4[QSSIZE];

	_bncneon_set0_qs(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		a4[0] = vld1q_f32(&(a->element[0][index]));
		a4[1] = vld1q_f32(&(a->element[1][index]));
		a4[2] = vld1q_f32(&(a->element[2][index]));
		a4[3] = vld1q_f32(&(a->element[3][index]));
		b4[0] = vld1q_f32(&(b->element[0][index]));
		b4[1] = vld1q_f32(&(b->element[1][index]));
		b4[2] = vld1q_f32(&(b->element[2][index]));
		b4[3] = vld1q_f32(&(b->element[3][index]));

//		rqs_mul(tmp, get_qsvector_i(a, i), get_qsvector_i(b, i));
//		rqs_add(ret, ret, tmp);
		_bncneon_rqs_mul(tmp4, a4, b4);
		_bncneon_rqs_add(ret4, ret4, tmp4);
	}
	_bncneon_rqs_sum128f(ret, ret4);
#else // others
	float tmp[QSSIZE];

	set0_qs(ret);
	for(i = 0; i < a->dim; i++)
	{
		rqs_mul(tmp, get_qsvector_i(a, i), get_qsvector_i(b, i));
		rqs_add(ret, ret, tmp);
	}
#endif // __AVX2__

	return;
}

/* c := a */
void subst_qsvector(QSVector c, QSVector a)
{
	long int i;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
//	for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i));
		_mm256_store_ps(&(c->element[0][i]), _mm256_load_ps(&(a->element[0][i])));
		_mm256_store_ps(&(c->element[1][i]), _mm256_load_ps(&(a->element[1][i])));
		_mm256_store_ps(&(c->element[2][i]), _mm256_load_ps(&(a->element[2][i])));
		_mm256_store_ps(&(c->element[3][i]), _mm256_load_ps(&(a->element[3][i])));
	}
#elif defined(__AVX512F__) // __AVX512F__
//	for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i));
		_mm512_store_ps(&(c->element[0][i]), _mm512_load_ps(&(a->element[0][i])));
		_mm512_store_ps(&(c->element[1][i]), _mm512_load_ps(&(a->element[1][i])));
		_mm512_store_ps(&(c->element[2][i]), _mm512_load_ps(&(a->element[2][i])));
		_mm512_store_ps(&(c->element[3][i]), _mm512_load_ps(&(a->element[3][i])));
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
//	for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(a->real_dim));
		//set_dvector_i(c, i, get_dvector_i(a, i));
		svst1_f32(pg, &(c->element[0][i]), svld1_f32(pg, &(a->element[0][i])));
		svst1_f32(pg, &(c->element[1][i]), svld1_f32(pg, &(a->element[1][i])));
		svst1_f32(pg, &(c->element[2][i]), svld1_f32(pg, &(a->element[2][i])));
		svst1_f32(pg, &(c->element[3][i]), svld1_f32(pg, &(a->element[3][i])));
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
//	for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i));
		vst1q_f32(&(c->element[0][i]), vld1q_f32(&(a->element[0][i])));
		vst1q_f32(&(c->element[1][i]), vld1q_f32(&(a->element[1][i])));
		vst1q_f32(&(c->element[2][i]), vld1q_f32(&(a->element[2][i])));
		vst1q_f32(&(c->element[3][i]), vld1q_f32(&(a->element[3][i])));
	}
#else // others
	for(i = 0; i < a->dim; i++)
		set_qsvector_i(c, i, get_qsvector_i(a, i));
#endif // __AVX2__
}

/* c := -a */
void neg_qsvector(QSVector c, QSVector a)
{
	long int i;
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 tmp[QSSIZE];

	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//rqs_neg(tmp, get_qsvector_i(a, i));
		//set_qsvector_i(c, i, tmp);
		tmp[0] = _bncavx2_fneg(_mm256_load_ps(&(a->element[0][i])));
		tmp[1] = _bncavx2_fneg(_mm256_load_ps(&(a->element[1][i])));
		tmp[2] = _bncavx2_fneg(_mm256_load_ps(&(a->element[2][i])));
		tmp[3] = _bncavx2_fneg(_mm256_load_ps(&(a->element[3][i])));
		_mm256_store_ps(&(c->element[0][i]), tmp[0]);
		_mm256_store_ps(&(c->element[1][i]), tmp[1]);
		_mm256_store_ps(&(c->element[2][i]), tmp[2]);
		_mm256_store_ps(&(c->element[3][i]), tmp[3]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 tmp[QSSIZE];

	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//rqs_neg(tmp, get_qsvector_i(a, i));
		//set_qsvector_i(c, i, tmp);
		tmp[0] = _bncavx512_fneg(_mm512_load_ps(&(a->element[0][i])));
		tmp[1] = _bncavx512_fneg(_mm512_load_ps(&(a->element[1][i])));
		tmp[2] = _bncavx512_fneg(_mm512_load_ps(&(a->element[2][i])));
		tmp[3] = _bncavx512_fneg(_mm512_load_ps(&(a->element[3][i])));
		_mm512_store_ps(&(c->element[0][i]), tmp[0]);
		_mm512_store_ps(&(c->element[1][i]), tmp[1]);
		_mm512_store_ps(&(c->element[2][i]), tmp[2]);
		_mm512_store_ps(&(c->element[3][i]), tmp[3]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t tmp_0, tmp_1, tmp_2, tmp_3;

	for(i = 0; i < a->real_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(a->real_dim));
		//rqs_neg(tmp, get_qsvector_i(a, i));
		//set_qsvector_i(c, i, tmp);
		tmp_0 = svneg_f32_x(pg, svld1_f32(pg, &(a->element[0][i])));
		tmp_1 = svneg_f32_x(pg, svld1_f32(pg, &(a->element[1][i])));
		tmp_2 = svneg_f32_x(pg, svld1_f32(pg, &(a->element[2][i])));
		tmp_3 = svneg_f32_x(pg, svld1_f32(pg, &(a->element[3][i])));
		svst1_f32(pg, &(c->element[0][i]), tmp_0);
		svst1_f32(pg, &(c->element[1][i]), tmp_1);
		svst1_f32(pg, &(c->element[2][i]), tmp_2);
		svst1_f32(pg, &(c->element[3][i]), tmp_3);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t tmp[QSSIZE];

	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//rqs_neg(tmp, get_qsvector_i(a, i));
		//set_qsvector_i(c, i, tmp);
		tmp[0] = vnegq_f32(vld1q_f32(&(a->element[0][i])));
		tmp[1] = vnegq_f32(vld1q_f32(&(a->element[1][i])));
		tmp[2] = vnegq_f32(vld1q_f32(&(a->element[2][i])));
		tmp[3] = vnegq_f32(vld1q_f32(&(a->element[3][i])));
		vst1q_f32(&(c->element[0][i]), tmp[0]);
		vst1q_f32(&(c->element[1][i]), tmp[1]);
		vst1q_f32(&(c->element[2][i]), tmp[2]);
		vst1q_f32(&(c->element[3][i]), tmp[3]);
	}
#else // others
	float tmp[QSSIZE];

	for(i = 0; i < a->dim; i++)
	{
		rqs_neg(tmp, get_qsvector_i(a, i));
		set_qsvector_i(c, i, tmp);
	}
#endif // __AVX2__
}


/* ||a||_1 */
void norm1_qsvector(float ret[QSSIZE], QSVector a)
{
	long int i, index, dim;

	dim = a->dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 vec4[QSSIZE], ret4[QSSIZE], tmp4[QSSIZE];

	_bncavx2_set0_qs(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		vec4[0] = _mm256_load_ps(&(a->element[0][index]));
		vec4[1] = _mm256_load_ps(&(a->element[1][index]));
		vec4[2] = _mm256_load_ps(&(a->element[2][index]));
		vec4[3] = _mm256_load_ps(&(a->element[3][index]));

		//rqs_abs(tmp, get_qsvector_i(a, i));
		//rqs_add(ret, ret, tmp);
		_bncavx2_rqs_abs(tmp4, vec4);
		_bncavx2_rqs_add(ret4, ret4, tmp4);
	}
	_bncavx2_rqs_abssum256(ret, ret4);
#elif defined(__AVX512F__) // __AVX512F__
	__m512 vec4[QSSIZE], ret4[QSSIZE], tmp4[QSSIZE];

	_bncavx512_set0_qs(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		vec4[0] = _mm512_load_ps(&(a->element[0][index]));
		vec4[1] = _mm512_load_ps(&(a->element[1][index]));
		vec4[2] = _mm512_load_ps(&(a->element[2][index]));
		vec4[3] = _mm512_load_ps(&(a->element[3][index]));

		//rqs_abs(tmp, get_qsvector_i(a, i));
		//rqs_add(ret, ret, tmp);
		_bncavx512_rqs_abs(tmp4, vec4);
		_bncavx512_rqs_add(ret4, ret4, tmp4);
	}
	_bncavx512_rqs_abssum512(ret, ret4);
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t vec4_0, vec4_1, vec4_2, vec4_3;
	svfloat32_t ret4_0, ret4_1, ret4_2, ret4_3;
	svfloat32_t tmp4_0, tmp4_1, tmp4_2, tmp4_3;

	_bncsve2_rqs_set0(&ret4_0, &ret4_1, &ret4_2, &ret4_3);
	for(index = 0; index < a->real_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(a->real_dim));
		vec4_0 = svld1_f32(pg, &(a->element[0][index]));
		vec4_1 = svld1_f32(pg, &(a->element[1][index]));
		vec4_2 = svld1_f32(pg, &(a->element[2][index]));
		vec4_3 = svld1_f32(pg, &(a->element[3][index]));

		//rqs_abs(tmp, get_qsvector_i(a, i));
		//rqs_add(ret, ret, tmp);
		_bncsve2_rqs_abs(svptrue_b32(), &tmp4_0, &tmp4_1, &tmp4_2, &tmp4_3, vec4_0, vec4_1, vec4_2, vec4_3);
		_bncsve2_rqs_add(svptrue_b32(), &ret4_0, &ret4_1, &ret4_2, &ret4_3, ret4_0, ret4_1, ret4_2, ret4_3, tmp4_0, tmp4_1, tmp4_2, tmp4_3);
	}
	_bncsve2_rqs_abssum128(svptrue_b32(), ret, ret4_0, ret4_1, ret4_2, ret4_3);
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t vec4[QSSIZE], ret4[QSSIZE], tmp4[QSSIZE];

	_bncneon_set0_qs(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		vec4[0] = vld1q_f32(&(a->element[0][index]));
		vec4[1] = vld1q_f32(&(a->element[1][index]));
		vec4[2] = vld1q_f32(&(a->element[2][index]));
		vec4[3] = vld1q_f32(&(a->element[3][index]));

		//rqs_abs(tmp, get_qsvector_i(a, i));
		//rqs_add(ret, ret, tmp);
		_bncneon_rqs_abs(tmp4, vec4);
		_bncneon_rqs_add(ret4, ret4, tmp4);
	}
	_bncneon_rqs_abssum128f(ret, ret4);
#else // others
	float tmp[QSSIZE];

	set0_qs(ret);
	for(i = 0; i < a->dim; i++)
	{
		rqs_abs(tmp, get_qsvector_i(a, i));
		rqs_add(ret, ret, tmp);
	}
#endif // __AVX2__

	return;
}

/* ||a||_infty */
void normi_qsvector(float ret[QSSIZE], QSVector a)
{
	float tmp[QSSIZE];
	long int i;

	rqs_abs(ret, get_qsvector_i(a, 0));
	for(i = 1; i < a->dim; i++)
	{
		rqs_abs(tmp, get_qsvector_i(a, i));
		if(rqs_cmp(ret, tmp) < 0)
			rqs_set(ret, tmp);
	}

	return;
}

// Euclid norm
void norm2_qsvector(float ret[QSSIZE], QSVector vec)
{
	long int i, index, dim;

	dim = vec->dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 vec4[QSSIZE], ret4[QSSIZE], tmp4[QSSIZE];

	_bncavx2_set0_qs(ret4);
	for(index = 0; index < vec->real_dim; index += _BNC_S_WIDTH)
	{
		vec4[0] = _mm256_load_ps(&(vec->element[0][index]));
		vec4[1] = _mm256_load_ps(&(vec->element[1][index]));
		vec4[2] = _mm256_load_ps(&(vec->element[2][index]));
		vec4[3] = _mm256_load_ps(&(vec->element[3][index]));

//		rqs_mul(tmp, get_qsvector_i(vec, i), get_qsvector_i(vec, i));
//		rqs_add(ret, ret, tmp);
		_bncavx2_rqs_mul(tmp4, vec4, vec4);
		_bncavx2_rqs_add(ret4, ret4, tmp4);
	}
	_bncavx2_rqs_norm256(ret, ret4);
#elif defined(__AVX512F__) // __AVX512F__
	__m512 vec4[QSSIZE], ret4[QSSIZE], tmp4[QSSIZE];

	_bncavx512_set0_qs(ret4);
	for(index = 0; index < vec->real_dim; index += _BNC_S_WIDTH)
	{
		vec4[0] = _mm512_load_ps(&(vec->element[0][index]));
		vec4[1] = _mm512_load_ps(&(vec->element[1][index]));
		vec4[2] = _mm512_load_ps(&(vec->element[2][index]));
		vec4[3] = _mm512_load_ps(&(vec->element[3][index]));

//		rqs_mul(tmp, get_qsvector_i(vec, i), get_qsvector_i(vec, i));
//		rqs_add(ret, ret, tmp);
		_bncavx512_rqs_mul(tmp4, vec4, vec4);
		_bncavx512_rqs_add(ret4, ret4, tmp4);
	}
	_bncavx512_rqs_norm512(ret, ret4);
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t vec4_0, vec4_1, vec4_2, vec4_3;
	svfloat32_t ret4_0, ret4_1, ret4_2, ret4_3;
	svfloat32_t tmp4_0, tmp4_1, tmp4_2, tmp4_3;

	_bncsve2_rqs_set0(&ret4_0, &ret4_1, &ret4_2, &ret4_3);
	for(index = 0; index < vec->real_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(vec->real_dim));
		vec4_0 = svld1_f32(pg, &(vec->element[0][index]));
		vec4_1 = svld1_f32(pg, &(vec->element[1][index]));
		vec4_2 = svld1_f32(pg, &(vec->element[2][index]));
		vec4_3 = svld1_f32(pg, &(vec->element[3][index]));

//		rqs_mul(tmp, get_qsvector_i(vec, i), get_qsvector_i(vec, i));
//		rqs_add(ret, ret, tmp);
		_bncsve2_rqs_mul(svptrue_b32(), &tmp4_0, &tmp4_1, &tmp4_2, &tmp4_3, vec4_0, vec4_1, vec4_2, vec4_3, vec4_0, vec4_1, vec4_2, vec4_3);
		_bncsve2_rqs_add(svptrue_b32(), &ret4_0, &ret4_1, &ret4_2, &ret4_3, ret4_0, ret4_1, ret4_2, ret4_3, tmp4_0, tmp4_1, tmp4_2, tmp4_3);
	}
	_bncsve2_rqs_norm128(svptrue_b32(), ret, ret4_0, ret4_1, ret4_2, ret4_3);
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t vec4[QSSIZE], ret4[QSSIZE], tmp4[QSSIZE];

	_bncneon_set0_qs(ret4);
	for(index = 0; index < vec->real_dim; index += _BNC_S_WIDTH)
	{
		vec4[0] = vld1q_f32(&(vec->element[0][index]));
		vec4[1] = vld1q_f32(&(vec->element[1][index]));
		vec4[2] = vld1q_f32(&(vec->element[2][index]));
		vec4[3] = vld1q_f32(&(vec->element[3][index]));

//		rqs_mul(tmp, get_qsvector_i(vec, i), get_qsvector_i(vec, i));
//		rqs_add(ret, ret, tmp);
		_bncneon_rqs_mul(tmp4, vec4, vec4);
		_bncneon_rqs_add(ret4, ret4, tmp4);
	}
	_bncneon_rqs_norm128f(ret, ret4);
#else // others
	float tmp[QSSIZE];

	//c_ds_copy_d((float)0.0, tmp);
	//c_ds_copy_d((float)0.0, ret);
	rqs_set0(tmp);
	rqs_set0(ret);

	for(i = 0; i < dim ; i++)
	{
		//c_qs_sqr(GET_QSVECTOR_I(vec, i), tmp);
		//c_qs_add(tmp, ret, ret);
		rqs_mul(tmp, get_qsvector_i(vec, i), get_qsvector_i(vec, i));
		rqs_add(ret, ret, tmp);
	}

	//c_qs_sqrt(ret, tmp);
	//c_qs_copy(tmp, ret);
	rqs_sqrt(tmp, ret);
	rqs_set(ret, tmp);
#endif // __AVX2__
}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__

/* add */
void _bncavx2_qsadd(qsfloat ret[], qsfloat a[], qsfloat b[], int dim)
{
    int i, index, div, rem, unit = _BNC_S_WIDTH;
    __m256 in_ret[QSSIZE], in_a_val[QSSIZE], in_b_val[QSSIZE];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_set_ps(
            a[index + 7].val[0],
            a[index + 6].val[0],
            a[index + 5].val[0],
            a[index + 4].val[0],
            a[index + 3].val[0],
            a[index + 2].val[0],
            a[index + 1].val[0],
            a[index    ].val[0]
        );
        in_a_val[1] = _mm256_set_ps(
			a[index + 7].val[1],
            a[index + 6].val[1],
            a[index + 5].val[1],
            a[index + 4].val[1],
            a[index + 3].val[1],
            a[index + 2].val[1],
            a[index + 1].val[1],
            a[index    ].val[1]
        );
        in_a_val[2] = _mm256_set_ps(
        	a[index + 7].val[2],
            a[index + 6].val[2],
            a[index + 5].val[2],
            a[index + 4].val[2],
            a[index + 3].val[2],
            a[index + 2].val[2],
            a[index + 1].val[2],
            a[index    ].val[2]
        );
        in_a_val[3] = _mm256_set_ps(
 			a[index + 7].val[3],
            a[index + 6].val[3],
            a[index + 5].val[3],
            a[index + 4].val[3],
            a[index + 3].val[3],
            a[index + 2].val[3],
            a[index + 1].val[3],
            a[index    ].val[3]
        );
        in_b_val[0] = _mm256_set_ps(
            b[index + 7].val[0],
            b[index + 6].val[0],
            b[index + 5].val[0],
            b[index + 4].val[0],
            b[index + 3].val[0],
            b[index + 2].val[0],
            b[index + 1].val[0],
            b[index    ].val[0]
        );
        in_b_val[1] = _mm256_set_ps(
            b[index + 7].val[1],
            b[index + 6].val[1],
            b[index + 5].val[1],
            b[index + 4].val[1],
            b[index + 3].val[1],
            b[index + 2].val[1],
            b[index + 1].val[1],
            b[index    ].val[1]
        );
        in_b_val[2] = _mm256_set_ps(
            b[index + 7].val[2],
            b[index + 6].val[2],
            b[index + 5].val[2],
            b[index + 4].val[2],
            b[index + 3].val[2],
            b[index + 2].val[2],
            b[index + 1].val[2],
            b[index    ].val[2]
        );
        in_b_val[3] = _mm256_set_ps(
            b[index + 7].val[3],
            b[index + 6].val[3],
            b[index + 5].val[3],
            b[index + 4].val[3],
            b[index + 3].val[3],
            b[index + 2].val[3],
            b[index + 1].val[3],
            b[index    ].val[3]
        );

        _bncavx2_rqs_add(in_ret, in_a_val, in_b_val);

        ret[index    ].val[0] = in_ret[0][0];
        ret[index + 1].val[0] = in_ret[0][1];
        ret[index + 2].val[0] = in_ret[0][2];
        ret[index + 3].val[0] = in_ret[0][3];
        ret[index + 4].val[0] = in_ret[0][4];
        ret[index + 5].val[0] = in_ret[0][5];
        ret[index + 6].val[0] = in_ret[0][6];
        ret[index + 7].val[0] = in_ret[0][7];

        ret[index    ].val[1] = in_ret[1][0];
        ret[index + 1].val[1] = in_ret[1][1];
        ret[index + 2].val[1] = in_ret[1][2];
        ret[index + 3].val[1] = in_ret[1][3];
        ret[index + 4].val[1] = in_ret[1][4];
        ret[index + 5].val[1] = in_ret[1][5];
        ret[index + 6].val[1] = in_ret[1][6];
        ret[index + 7].val[1] = in_ret[1][7];

        ret[index    ].val[2] = in_ret[2][0];
        ret[index + 1].val[2] = in_ret[2][1];
        ret[index + 2].val[2] = in_ret[2][2];
        ret[index + 3].val[2] = in_ret[2][3];
        ret[index + 4].val[2] = in_ret[2][4];
        ret[index + 5].val[2] = in_ret[2][5];
        ret[index + 6].val[2] = in_ret[2][6];
        ret[index + 7].val[2] = in_ret[2][7];

        ret[index    ].val[3] = in_ret[3][0];
        ret[index + 1].val[3] = in_ret[3][1];
        ret[index + 2].val[3] = in_ret[3][2];
        ret[index + 3].val[3] = in_ret[3][3];
        ret[index + 4].val[3] = in_ret[3][4];
        ret[index + 5].val[3] = in_ret[3][5];
        ret[index + 6].val[3] = in_ret[3][6];
        ret[index + 7].val[3] = in_ret[3][7];
   }
}

void _bncavx2_qsvadd(QSVector ret, QSVector a, QSVector b, int dim)
{
    int i, index, div, rem, unit = _BNC_S_WIDTH;
    __m256 in_ret[QSSIZE], in_a_val[QSSIZE], in_b_val[QSSIZE];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_ps(&(a->element[1][index]));
        in_a_val[2] = _mm256_load_ps(&(a->element[2][index]));
        in_a_val[3] = _mm256_load_ps(&(a->element[3][index]));

        in_b_val[0] = _mm256_load_ps(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_ps(&(b->element[1][index]));
        in_b_val[2] = _mm256_load_ps(&(b->element[2][index]));
        in_b_val[3] = _mm256_load_ps(&(b->element[3][index]));

        _bncavx2_rqs_add(in_ret, in_a_val, in_b_val);

        _mm256_store_ps(&ret->element[0][index], in_ret[0]);
        _mm256_store_ps(&ret->element[1][index], in_ret[1]);
        _mm256_store_ps(&ret->element[2][index], in_ret[2]);
        _mm256_store_ps(&ret->element[3][index], in_ret[3]);
   }

}

/* mul */
void _bncavx2_qsmul(qsfloat ret[], qsfloat a[], qsfloat b[], int dim)
{
    int i, index, div, rem, unit = _BNC_S_WIDTH;
    __m256 in_ret[QSSIZE], in_a_val[QSSIZE], in_b_val[QSSIZE];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_set_ps(
            a[index + 7].val[0],
            a[index + 6].val[0],
            a[index + 5].val[0],
            a[index + 4].val[0],
            a[index + 3].val[0],
            a[index + 2].val[0],
            a[index + 1].val[0],
            a[index    ].val[0]
        );
        in_a_val[1] = _mm256_set_ps(
            a[index + 7].val[1],
            a[index + 6].val[1],
            a[index + 5].val[1],
            a[index + 4].val[1],
            a[index + 3].val[1],
            a[index + 2].val[1],
            a[index + 1].val[1],
            a[index    ].val[1]
        );
        in_a_val[2] = _mm256_set_ps(
            a[index + 7].val[2],
            a[index + 6].val[2],
            a[index + 5].val[2],
            a[index + 4].val[2],
            a[index + 3].val[2],
            a[index + 2].val[2],
            a[index + 1].val[2],
            a[index    ].val[2]
        );
        in_a_val[3] = _mm256_set_ps(
            a[index + 7].val[3],
            a[index + 6].val[3],
            a[index + 5].val[3],
            a[index + 4].val[3],
            a[index + 3].val[3],
            a[index + 2].val[3],
            a[index + 1].val[3],
            a[index    ].val[3]
        );
        in_b_val[0] = _mm256_set_ps(
            b[index + 7].val[0],
            b[index + 6].val[0],
            b[index + 5].val[0],
            b[index + 4].val[0],
            b[index + 3].val[0],
            b[index + 2].val[0],
            b[index + 1].val[0],
            b[index    ].val[0]
        );
        in_b_val[1] = _mm256_set_ps(
            b[index + 7].val[1],
            b[index + 6].val[1],
            b[index + 5].val[1],
            b[index + 4].val[1],
            b[index + 3].val[1],
            b[index + 2].val[1],
            b[index + 1].val[1],
            b[index    ].val[1]
        );
        in_b_val[2] = _mm256_set_ps(
            b[index + 7].val[2],
            b[index + 6].val[2],
            b[index + 5].val[2],
            b[index + 4].val[2],
            b[index + 3].val[2],
            b[index + 2].val[2],
            b[index + 1].val[2],
            b[index    ].val[2]
        );
        in_b_val[3] = _mm256_set_ps(
            b[index + 7].val[3],
            b[index + 6].val[3],
            b[index + 5].val[3],
            b[index + 4].val[3],
            b[index + 3].val[3],
            b[index + 2].val[3],
            b[index + 1].val[3],
            b[index    ].val[3]
        );

        _bncavx2_rqs_mul(in_ret, in_a_val, in_b_val);

        ret[index    ].val[0] = in_ret[0][0];
        ret[index + 1].val[0] = in_ret[0][1];
        ret[index + 2].val[0] = in_ret[0][2];
        ret[index + 3].val[0] = in_ret[0][3];
        ret[index + 4].val[0] = in_ret[0][4];
        ret[index + 5].val[0] = in_ret[0][5];
        ret[index + 6].val[0] = in_ret[0][6];
        ret[index + 7].val[0] = in_ret[0][7];

        ret[index    ].val[1] = in_ret[1][0];
        ret[index + 1].val[1] = in_ret[1][1];
        ret[index + 2].val[1] = in_ret[1][2];
        ret[index + 3].val[1] = in_ret[1][3];
        ret[index + 4].val[1] = in_ret[1][4];
        ret[index + 5].val[1] = in_ret[1][5];
        ret[index + 6].val[1] = in_ret[1][6];
        ret[index + 7].val[1] = in_ret[1][7];

        ret[index    ].val[2] = in_ret[2][0];
        ret[index + 1].val[2] = in_ret[2][1];
        ret[index + 2].val[2] = in_ret[2][2];
        ret[index + 3].val[2] = in_ret[2][3];
        ret[index + 4].val[2] = in_ret[2][4];
        ret[index + 5].val[2] = in_ret[2][5];
        ret[index + 6].val[2] = in_ret[2][6];
        ret[index + 7].val[2] = in_ret[2][7];

        ret[index    ].val[3] = in_ret[3][0];
        ret[index + 1].val[3] = in_ret[3][1];
        ret[index + 2].val[3] = in_ret[3][2];
        ret[index + 3].val[3] = in_ret[3][3];
        ret[index + 4].val[3] = in_ret[3][4];
        ret[index + 5].val[3] = in_ret[3][5];
        ret[index + 6].val[3] = in_ret[3][6];
        ret[index + 7].val[3] = in_ret[3][7];

   }
}

void _bncavx2_qsvmul(QSVector ret, QSVector a, QSVector b, int dim)
{
    int i, index, div, rem, unit = _BNC_S_WIDTH;
    __m256 in_ret[QSSIZE], in_a_val[QSSIZE], in_b_val[QSSIZE];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_ps(&(a->element[1][index]));
        in_a_val[2] = _mm256_load_ps(&(a->element[2][index]));
        in_a_val[3] = _mm256_load_ps(&(a->element[3][index]));

        in_b_val[0] = _mm256_load_ps(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_ps(&(b->element[1][index]));
        in_b_val[2] = _mm256_load_ps(&(b->element[2][index]));
        in_b_val[3] = _mm256_load_ps(&(b->element[3][index]));

        _bncavx2_rqs_mul(in_ret, in_a_val, in_b_val);

        _mm256_store_ps(&ret->element[0][index], in_ret[0]);
        _mm256_store_ps(&ret->element[1][index], in_ret[1]);
        _mm256_store_ps(&ret->element[2][index], in_ret[2]);
        _mm256_store_ps(&ret->element[3][index], in_ret[3]);
   }

}

/* div */
void _bncavx2_qsdiv(qsfloat ret[], qsfloat a[], qsfloat b[], int dim)
{
    int i, index, div, rem, unit = _BNC_S_WIDTH;
    __m256 in_ret[QSSIZE], in_a_val[QSSIZE], in_b_val[QSSIZE];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_set_ps(
            a[index + 7].val[0],
            a[index + 6].val[0],
            a[index + 5].val[0],
            a[index + 4].val[0],
            a[index + 3].val[0],
            a[index + 2].val[0],
            a[index + 1].val[0],
            a[index    ].val[0]
        );
        in_a_val[1] = _mm256_set_ps(
            a[index + 7].val[1],
            a[index + 6].val[1],
            a[index + 5].val[1],
            a[index + 4].val[1],
            a[index + 3].val[1],
            a[index + 2].val[1],
            a[index + 1].val[1],
            a[index    ].val[1]
        );
        in_a_val[2] = _mm256_set_ps(
            a[index + 7].val[2],
            a[index + 6].val[2],
            a[index + 5].val[2],
            a[index + 4].val[2],
            a[index + 3].val[2],
            a[index + 2].val[2],
            a[index + 1].val[2],
            a[index    ].val[2]
        );
        in_a_val[3] = _mm256_set_ps(
            a[index + 7].val[3],
            a[index + 6].val[3],
            a[index + 5].val[3],
            a[index + 4].val[3],
            a[index + 3].val[3],
            a[index + 2].val[3],
            a[index + 1].val[3],
            a[index    ].val[3]
        );
        in_b_val[0] = _mm256_set_ps(
            b[index + 7].val[0],
            b[index + 6].val[0],
            b[index + 5].val[0],
            b[index + 4].val[0],
            b[index + 3].val[0],
            b[index + 2].val[0],
            b[index + 1].val[0],
            b[index    ].val[0]
        );
        in_b_val[1] = _mm256_set_ps(
            b[index + 7].val[1],
            b[index + 6].val[1],
            b[index + 5].val[1],
            b[index + 4].val[1],
            b[index + 3].val[1],
            b[index + 2].val[1],
            b[index + 1].val[1],
            b[index    ].val[1]
        );
        in_b_val[2] = _mm256_set_ps(
            b[index + 7].val[2],
            b[index + 6].val[2],
            b[index + 5].val[2],
            b[index + 4].val[2],
            b[index + 3].val[2],
            b[index + 2].val[2],
            b[index + 1].val[2],
            b[index    ].val[2]
        );
        in_b_val[3] = _mm256_set_ps(
            b[index + 7].val[3],
            b[index + 6].val[3],
            b[index + 5].val[3],
            b[index + 4].val[3],
            b[index + 3].val[3],
            b[index + 2].val[3],
            b[index + 1].val[3],
            b[index    ].val[3]
        );

        _bncavx2_rqs_div(in_ret, in_a_val, in_b_val);

        ret[index    ].val[0] = in_ret[0][0];
        ret[index + 1].val[0] = in_ret[0][1];
        ret[index + 2].val[0] = in_ret[0][2];
        ret[index + 3].val[0] = in_ret[0][3];
        ret[index + 4].val[0] = in_ret[0][4];
        ret[index + 5].val[0] = in_ret[0][5];
        ret[index + 6].val[0] = in_ret[0][6];
        ret[index + 7].val[0] = in_ret[0][7];

        ret[index    ].val[1] = in_ret[1][0];
        ret[index + 1].val[1] = in_ret[1][1];
        ret[index + 2].val[1] = in_ret[1][2];
        ret[index + 3].val[1] = in_ret[1][3];
        ret[index + 4].val[1] = in_ret[1][4];
        ret[index + 5].val[1] = in_ret[1][5];
        ret[index + 6].val[1] = in_ret[1][6];
        ret[index + 7].val[1] = in_ret[1][7];

        ret[index    ].val[2] = in_ret[2][0];
        ret[index + 1].val[2] = in_ret[2][1];
        ret[index + 2].val[2] = in_ret[2][2];
        ret[index + 3].val[2] = in_ret[2][3];
        ret[index + 4].val[2] = in_ret[2][4];
        ret[index + 5].val[2] = in_ret[2][5];
        ret[index + 6].val[2] = in_ret[2][6];
        ret[index + 7].val[2] = in_ret[2][7];

        ret[index    ].val[3] = in_ret[3][0];
        ret[index + 1].val[3] = in_ret[3][1];
        ret[index + 2].val[3] = in_ret[3][2];
        ret[index + 3].val[3] = in_ret[3][3];
        ret[index + 4].val[3] = in_ret[3][4];
        ret[index + 5].val[3] = in_ret[3][5];
        ret[index + 6].val[3] = in_ret[3][6];
        ret[index + 7].val[3] = in_ret[3][7];
   }
}

void _bncavx2_qsvdiv(QSVector ret, QSVector a, QSVector b, int dim)
{
    int i, index, div, rem, unit = _BNC_S_WIDTH;
    __m256 in_ret[QSSIZE], in_a_val[QSSIZE], in_b_val[QSSIZE];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_ps(&(a->element[1][index]));
        in_a_val[2] = _mm256_load_ps(&(a->element[2][index]));
        in_a_val[3] = _mm256_load_ps(&(a->element[3][index]));

        in_b_val[0] = _mm256_load_ps(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_ps(&(b->element[1][index]));
        in_b_val[2] = _mm256_load_ps(&(b->element[2][index]));
        in_b_val[3] = _mm256_load_ps(&(b->element[3][index]));

        _bncavx2_rqs_div(in_ret, in_a_val, in_b_val);

        _mm256_store_ps(&ret->element[0][index], in_ret[0]);
        _mm256_store_ps(&ret->element[1][index], in_ret[1]);
        _mm256_store_ps(&ret->element[2][index], in_ret[2]);
        _mm256_store_ps(&ret->element[3][index], in_ret[3]);
   }
}
#endif // if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__

// set a zero matrix
//void set0_qsmatrix(QSMatrix mat)
void set0_qsmatrix(QSMatrix mat)
{
	long int i;
	long int real_total_dim;

	real_total_dim = mat->real_row_dim * mat->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 zero4;

	zero4 = _mm256_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm256_store_ps(&(mat->element[0][i]), zero4);
		_mm256_store_ps(&(mat->element[1][i]), zero4);
		_mm256_store_ps(&(mat->element[2][i]), zero4);
		_mm256_store_ps(&(mat->element[3][i]), zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero4;

	zero4 = _mm512_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm512_store_ps(&(mat->element[0][i]), zero4);
		_mm512_store_ps(&(mat->element[1][i]), zero4);
		_mm512_store_ps(&(mat->element[2][i]), zero4);
		_mm512_store_ps(&(mat->element[3][i]), zero4);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t zero4;

	zero4 = svdup_f32(0.0f);
	for(i = 0; i < real_total_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(real_total_dim));
		svst1_f32(pg, &(mat->element[0][i]), zero4);
		svst1_f32(pg, &(mat->element[1][i]), zero4);
		svst1_f32(pg, &(mat->element[2][i]), zero4);
		svst1_f32(pg, &(mat->element[3][i]), zero4);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t zero4;

	zero4 = vdupq_n_f32(0.0f);
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		vst1q_f32(&(mat->element[0][i]), zero4);
		vst1q_f32(&(mat->element[1][i]), zero4);
		vst1q_f32(&(mat->element[2][i]), zero4);
		vst1q_f32(&(mat->element[3][i]), zero4);
	}
#else // others
	for(i = 0; i < real_total_dim; i++)
	{
		mat->element[0][i] = 0.0;
		mat->element[1][i] = 0.0;	
		mat->element[2][i] = 0.0;	
		mat->element[3][i] = 0.0;	
	}
#endif // __AVX2__
}

// initialize qsvector
QSMatrix init_qsmatrix(long int row_dim, long int col_dim)
{
	long int row_index, col_index, i;
	long int real_row_dim, real_col_dim, real_total_dim;
	QSMatrix ret = NULL;

	ret = (QSMatrix)BNC_MALLOC(sizeof(qsmatrix));
	if(ret == NULL)
		return ret;

	// real_dim is the nearest positive multiplier of _BNC_S_WIDTH
	real_row_dim = (long int)ceil((float)(row_dim) / (float)_BNC_S_WIDTH) * _BNC_S_WIDTH;
	real_col_dim = (long int)ceil((float)(col_dim) / (float)_BNC_S_WIDTH) * _BNC_S_WIDTH;
	real_total_dim = real_row_dim * real_col_dim;

	//printf("init_qsmatrix(%ld, %ld) %ld calloc\n", row_dim, col_dim, real_total_dim);
	ret->element[0] = (float *)BNC_CALLOC(real_total_dim, sizeof(float));
	if(ret->element[0] == NULL)
	{ 	free(ret);
		return NULL;
	}
	ret->element[1] = (float *)BNC_CALLOC(real_total_dim, sizeof(float));
	if(ret->element[1] == NULL)
	{
		free(ret->element[0]);
		free(ret);
		return NULL;
	}
	ret->element[2] = (float *)BNC_CALLOC(real_total_dim, sizeof(float));
	if(ret->element[2] == NULL)
	{
		free(ret->element[0]);
		free(ret->element[1]);
		free(ret);
		return NULL;
	}
	ret->element[3] = (float *)BNC_CALLOC(real_total_dim, sizeof(float));
	if(ret->element[3] == NULL)
	{
		free(ret->element[0]);
		free(ret->element[1]);
		free(ret->element[2]);
		free(ret);
		return NULL;
	}

	//printf("init_qsmatrix(%ld, %ld) calloc\n", row_dim, col_dim);
	/* All 0 */
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 zero4;

	zero4 = _mm256_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm256_store_ps(&ret->element[0][i], zero4);
		_mm256_store_ps(&ret->element[1][i], zero4);
		_mm256_store_ps(&ret->element[2][i], zero4);
		_mm256_store_ps(&ret->element[3][i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero4;

	zero4 = _mm512_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm512_store_ps(&ret->element[0][i], zero4);
		_mm512_store_ps(&ret->element[1][i], zero4);
		_mm512_store_ps(&ret->element[2][i], zero4);
		_mm512_store_ps(&ret->element[3][i], zero4);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t zero4;

	zero4 = svdup_f32(0.0f);
	for(i = 0; i < real_total_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(real_total_dim));
		svst1_f32(pg, &ret->element[0][i], zero4);
		svst1_f32(pg, &ret->element[1][i], zero4);
		svst1_f32(pg, &ret->element[2][i], zero4);
		svst1_f32(pg, &ret->element[3][i], zero4);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t zero4;

	zero4 = vdupq_n_f32(0.0f);
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		vst1q_f32(&ret->element[0][i], zero4);
		vst1q_f32(&ret->element[1][i], zero4);
		vst1q_f32(&ret->element[2][i], zero4);
		vst1q_f32(&ret->element[3][i], zero4);
	}
#else // others
	for(i = 0; i < real_total_dim; i++)
	{
		ret->element[0][i] = 0.0;
		ret->element[1][i] = 0.0;
		ret->element[2][i] = 0.0;
		ret->element[3][i] = 0.0;
	}
#endif // __AVX2__

	ret->real_row_dim = real_row_dim;
	ret->real_col_dim = real_col_dim;
	ret->row_dim = row_dim;
	ret->col_dim = col_dim;

	return ret;
}

// free qsvector
void free_qsmatrix(QSMatrix mat)
{
	long int i;

	for(i = 0; i < QSSIZE; i++)
		free(mat->element[i]);

	free(mat);
}

// print qsvector
void print_qsmatrix(QSMatrix mat)
{
	long int row_index, col_index;

	for(row_index = 0; row_index < mat->row_dim; row_index++)
	{
		for(col_index = 0; col_index < mat->col_dim; col_index++)
		{
			printf("%ld, %ld: ", row_index, col_index);
	//		c_qs_write((vec->element + index * QSSIZE));
			RQS_OUT_STR(GET_QSMATRIX_IJ(mat, row_index, col_index));
		}
	}
}

// QSMatrix mat -> qsfloat array
void set_qsfloat_qsmat(qsfloat ret[], int ret_dim, QSMatrix mat)
{
    int total_index, j, total_dim;
	long int row_index, col_index;

    total_dim = (ret_dim < (mat->row_dim * mat->col_dim)) ? ret_dim : (mat->row_dim * mat->col_dim);

	total_index = 0;
    for(row_index = 0; row_index < mat->row_dim; row_index++)
    {
		for(col_index = 0; (col_index < mat->col_dim) && (total_index < total_dim); col_index++)
		{
			for(j = 0; j < QSSIZE; j++)
				ret[total_index].val[j] = mat->element[j][(row_index * mat->real_col_dim) + col_index];

			total_index++;
		}
    }
}

// qsfloat array -> QDmatrix ret
void set_qsmatrix_qsfloat(QSMatrix ret, qsfloat array[], int array_dim)
{
    int total_index, j, total_dim;
	long int row_index, col_index;

    total_dim = (array_dim < (ret->row_dim * ret->col_dim)) ? array_dim : (ret->row_dim * ret->col_dim);

 	total_index = 0;
    for(row_index = 0; row_index < ret->row_dim; row_index++)
    {
		for(col_index = 0; (col_index < ret->col_dim) && (total_index < total_dim); col_index++)
		{
			for(j = 0; j < QSSIZE; j++)
				ret->element[j][(row_index * ret->real_col_dim) + col_index] = array[total_index].val[j];

			total_index++;
		}
    }
}

// matrix multiplication
// ret := A * B
void mul_qsmatrix(QSMatrix ret, QSMatrix a, QSMatrix b)
{
	long int i, j, k;

	/* dimension check */
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: mul_qsmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret->row_dim, ret->col_dim, a->row_dim, a->col_dim, b->row_dim, b->col_dim);
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long real_row_dim, real_col_dim, real_mid_dim;
	float cijval[8][QSSIZE];
    __m256 cij[QSSIZE], aik[QSSIZE], bkj[QSSIZE], tmp_mul[QSSIZE];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j++)
        {
            //rqs_set_ui(cij.val, 0UL);
            cij[0] = _mm256_setzero_ps();
            cij[1] = _mm256_setzero_ps();
            cij[2] = _mm256_setzero_ps();
            cij[3] = _mm256_setzero_ps();
			for(k = 0; k < real_mid_dim; k += _BNC_S_WIDTH)
            {
            /*
                aik[0].val[0] = a->element[0][i * mid_dim + k];
                aik[1].val[0] = a->element[0][i * mid_dim + k + 1];
                aik[2].val[0] = a->element[0][i * mid_dim + k + 2];
                aik[3].val[0] = a->element[0][i * mid_dim + k + 3];
            */
            
			    aik[0] = _mm256_load_ps(&(a->element[0][i * real_mid_dim + k]));
            /*    aik[0] = _mm256_set_ps(
                    a->element[0][i * real_mid_dim + k],
                    a->element[0][i * real_mid_dim + k + 1],
                    a->element[0][i * real_mid_dim + k + 2],
                    a->element[0][i * real_mid_dim + k + 3]
                );
            */
            /*
                aik[0].val[1] = a->element[1][i * mid_dim + k];
                aik[1].val[1] = a->element[1][i * mid_dim + k + 1];
                aik[2].val[1] = a->element[1][i * mid_dim + k + 2];
                aik[3].val[1] = a->element[1][i * mid_dim + k + 3];
            */
                aik[1] = _mm256_load_ps(&(a->element[1][i * real_mid_dim + k]));
            /*    aik[1] = _mm256_set_ps(
                    a->element[1][i * real_mid_dim + k],
                    a->element[1][i * real_mid_dim + k + 1],
                    a->element[1][i * real_mid_dim + k + 2],
                    a->element[1][i * real_mid_dim + k + 3]
                );
            */
                aik[2] = _mm256_load_ps(&(a->element[2][i * real_mid_dim + k]));
                aik[3] = _mm256_load_ps(&(a->element[3][i * real_mid_dim + k]));
		    /*   aik[2] = _mm256_set_ps(
                    a->element[2][i * real_mid_dim + k],
                    a->element[2][i * real_mid_dim + k + 1],
                    a->element[2][i * real_mid_dim + k + 2],
                    a->element[2][i * real_mid_dim + k + 3]
                );
                aik[3] = _mm256_set_ps(
                    a->element[3][i * real_mid_dim + k],
                    a->element[3][i * real_mid_dim + k + 1],
                    a->element[3][i * real_mid_dim + k + 2],
                    a->element[3][i * real_mid_dim + k + 3]
                );
            */    
            /*
                bkj[0].val[0] = b->element[0][k * col_dim + j];
                bkj[1].val[0] = b->element[0][(k + 1) * col_dim + j];
                bkj[2].val[0] = b->element[0][(k + 2) * col_dim + j];
                bkj[3].val[0] = b->element[0][(k + 3) * col_dim + j];
            */
                bkj[0] = _mm256_set_ps(
                    b->element[0][(k + 7) * real_col_dim + j],
                    b->element[0][(k + 6) * real_col_dim + j],
                    b->element[0][(k + 5) * real_col_dim + j],
                    b->element[0][(k + 4) * real_col_dim + j],
                    b->element[0][(k + 3) * real_col_dim + j],
                    b->element[0][(k + 2) * real_col_dim + j],
                    b->element[0][(k + 1) * real_col_dim + j],
                    b->element[0][(k    ) * real_col_dim + j]
                );
            /*
                bkj[0].val[1] = b->element[1][k * col_dim + j];
                bkj[1].val[1] = b->element[1][(k + 1) * col_dim + j];
                bkj[2].val[1] = b->element[1][(k + 2) * col_dim + j];
                bkj[3].val[1] = b->element[1][(k + 3) * col_dim + j];
            */
            
                bkj[1] = _mm256_set_ps(
                    b->element[1][(k + 7) * real_col_dim + j],
                    b->element[1][(k + 6) * real_col_dim + j],
                    b->element[1][(k + 5) * real_col_dim + j],
                    b->element[1][(k + 4) * real_col_dim + j],
                    b->element[1][(k + 3) * real_col_dim + j],
                    b->element[1][(k + 2) * real_col_dim + j],
                    b->element[1][(k + 1) * real_col_dim + j],
                    b->element[1][(k    ) * real_col_dim + j]
                );
                bkj[2] = _mm256_set_ps(
                    b->element[2][(k + 7) * real_col_dim + j],
                    b->element[2][(k + 6) * real_col_dim + j],
                    b->element[2][(k + 5) * real_col_dim + j],
                    b->element[2][(k + 4) * real_col_dim + j],
                    b->element[2][(k + 3) * real_col_dim + j],
                    b->element[2][(k + 2) * real_col_dim + j],
                    b->element[2][(k + 1) * real_col_dim + j],
                    b->element[2][(k    ) * real_col_dim + j]
                );
                bkj[3] = _mm256_set_ps(
                    b->element[3][(k + 7) * real_col_dim + j],
                    b->element[3][(k + 6) * real_col_dim + j],
                    b->element[3][(k + 5) * real_col_dim + j],
                    b->element[3][(k + 4) * real_col_dim + j],
                    b->element[3][(k + 3) * real_col_dim + j],
                    b->element[3][(k + 2) * real_col_dim + j],
                    b->element[3][(k + 1) * real_col_dim + j],
                    b->element[3][(k    ) * real_col_dim + j]
                );

            /*
                rqs_mul(tmp_mul[0].val, aik[0].val, bkj[0].val);
                rqs_mul(tmp_mul[1].val, aik[1].val, bkj[1].val);
                rqs_mul(tmp_mul[2].val, aik[2].val, bkj[2].val);
                rqs_mul(tmp_mul[3].val, aik[3].val, bkj[3].val);
            */
                _bncavx2_rqs_mul(tmp_mul, aik, bkj);

            /*
                rqs_add(cij.val, cij.val, tmp_mul[0].val);
                rqs_add(cij.val, cij.val, tmp_mul[1].val);
                rqs_add(cij.val, cij.val, tmp_mul[2].val);
                rqs_add(cij.val, cij.val, tmp_mul[3].val);
            */
                _bncavx2_rqs_add(cij, cij, tmp_mul);
            }

            cijval[0][0] = cij[0][0]; cijval[0][1] = cij[1][0]; cijval[0][2] = cij[2][0]; cijval[0][3] = cij[3][0];
            cijval[1][0] = cij[0][1]; cijval[1][1] = cij[1][1]; cijval[1][2] = cij[2][1]; cijval[1][3] = cij[3][1];
            cijval[2][0] = cij[0][2]; cijval[2][1] = cij[1][2]; cijval[2][2] = cij[2][2]; cijval[2][3] = cij[3][2];
            cijval[3][0] = cij[0][3]; cijval[3][1] = cij[1][3]; cijval[3][2] = cij[2][3]; cijval[3][3] = cij[3][3];
            cijval[4][0] = cij[0][4]; cijval[4][1] = cij[1][4]; cijval[4][2] = cij[2][4]; cijval[4][3] = cij[3][4];
            cijval[5][0] = cij[0][5]; cijval[5][1] = cij[1][5]; cijval[5][2] = cij[2][5]; cijval[5][3] = cij[3][5];
            cijval[6][0] = cij[0][6]; cijval[6][1] = cij[1][6]; cijval[6][2] = cij[2][6]; cijval[6][3] = cij[3][6];
            cijval[7][0] = cij[0][7]; cijval[7][1] = cij[1][7]; cijval[7][2] = cij[2][7]; cijval[7][3] = cij[3][7];

            rqs_add(cijval[0], cijval[0], cijval[1]);
            rqs_add(cijval[0], cijval[0], cijval[2]);
            rqs_add(cijval[0], cijval[0], cijval[3]);
            rqs_add(cijval[0], cijval[0], cijval[4]);
            rqs_add(cijval[0], cijval[0], cijval[5]);
            rqs_add(cijval[0], cijval[0], cijval[6]);
            rqs_add(cijval[0], cijval[0], cijval[7]);

            ret->element[0][i * real_col_dim + j] = cijval[0][0];
            ret->element[1][i * real_col_dim + j] = cijval[0][1];
            ret->element[2][i * real_col_dim + j] = cijval[0][2];
            ret->element[3][i * real_col_dim + j] = cijval[0][3];
       }
    }
#elif defined(__AVX512F__) // __AVX512F__
	long real_row_dim, real_col_dim, real_mid_dim;
	float cijval[16][QSSIZE];
    __m512 cij[QSSIZE], aik[QSSIZE], bkj[QSSIZE], tmp_mul[QSSIZE];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j++)
        {
            //rqs_set_ui(cij.val, 0UL);
            cij[0] = _mm512_setzero_ps();
            cij[1] = _mm512_setzero_ps();
            cij[2] = _mm512_setzero_ps();
            cij[3] = _mm512_setzero_ps();
			for(k = 0; k < real_mid_dim; k += _BNC_S_WIDTH)
            {
            /*
                aik[0].val[0] = a->element[0][i * mid_dim + k];
                aik[1].val[0] = a->element[0][i * mid_dim + k + 1];
                aik[2].val[0] = a->element[0][i * mid_dim + k + 2];
                aik[3].val[0] = a->element[0][i * mid_dim + k + 3];
            */
            
			    aik[0] = _mm512_load_ps(&(a->element[0][i * real_mid_dim + k]));
            /*    aik[0] = _mm512_set_ps(
                    a->element[0][i * real_mid_dim + k],
                    a->element[0][i * real_mid_dim + k + 1],
                    a->element[0][i * real_mid_dim + k + 2],
                    a->element[0][i * real_mid_dim + k + 3]
                );
            */
            /*
                aik[0].val[1] = a->element[1][i * mid_dim + k];
                aik[1].val[1] = a->element[1][i * mid_dim + k + 1];
                aik[2].val[1] = a->element[1][i * mid_dim + k + 2];
                aik[3].val[1] = a->element[1][i * mid_dim + k + 3];
            */
                aik[1] = _mm512_load_ps(&(a->element[1][i * real_mid_dim + k]));
            /*    aik[1] = _mm512_set_ps(
                    a->element[1][i * real_mid_dim + k],
                    a->element[1][i * real_mid_dim + k + 1],
                    a->element[1][i * real_mid_dim + k + 2],
                    a->element[1][i * real_mid_dim + k + 3]
                );
            */
                aik[2] = _mm512_load_ps(&(a->element[2][i * real_mid_dim + k]));
                aik[3] = _mm512_load_ps(&(a->element[3][i * real_mid_dim + k]));
		    /*   aik[2] = _mm512_set_ps(
                    a->element[2][i * real_mid_dim + k],
                    a->element[2][i * real_mid_dim + k + 1],
                    a->element[2][i * real_mid_dim + k + 2],
                    a->element[2][i * real_mid_dim + k + 3]
                );
                aik[3] = _mm512_set_ps(
                    a->element[3][i * real_mid_dim + k],
                    a->element[3][i * real_mid_dim + k + 1],
                    a->element[3][i * real_mid_dim + k + 2],
                    a->element[3][i * real_mid_dim + k + 3]
                );
            */    
            /*
                bkj[0].val[0] = b->element[0][k * col_dim + j];
                bkj[1].val[0] = b->element[0][(k + 1) * col_dim + j];
                bkj[2].val[0] = b->element[0][(k + 2) * col_dim + j];
                bkj[3].val[0] = b->element[0][(k + 3) * col_dim + j];
            */
                bkj[0] = _mm512_set_ps(
                    b->element[0][(k + 15) * real_col_dim + j],
                    b->element[0][(k + 14) * real_col_dim + j],
                    b->element[0][(k + 13) * real_col_dim + j],
                    b->element[0][(k + 12) * real_col_dim + j],
                    b->element[0][(k + 11) * real_col_dim + j],
                    b->element[0][(k + 10) * real_col_dim + j],
                    b->element[0][(k + 9) * real_col_dim + j],
                    b->element[0][(k + 8) * real_col_dim + j],
                    b->element[0][(k + 7) * real_col_dim + j],
                    b->element[0][(k + 6) * real_col_dim + j],
                    b->element[0][(k + 5) * real_col_dim + j],
                    b->element[0][(k + 4) * real_col_dim + j],
                    b->element[0][(k + 3) * real_col_dim + j],
                    b->element[0][(k + 2) * real_col_dim + j],
                    b->element[0][(k + 1) * real_col_dim + j],
                    b->element[0][(k + 0) * real_col_dim + j]
                );
            /*
                bkj[0].val[1] = b->element[1][k * col_dim + j];
                bkj[1].val[1] = b->element[1][(k + 1) * col_dim + j];
                bkj[2].val[1] = b->element[1][(k + 2) * col_dim + j];
                bkj[3].val[1] = b->element[1][(k + 3) * col_dim + j];
            */
            
                bkj[1] = _mm512_set_ps(
                    b->element[1][(k + 15) * real_col_dim + j],
                    b->element[1][(k + 14) * real_col_dim + j],
                    b->element[1][(k + 13) * real_col_dim + j],
                    b->element[1][(k + 12) * real_col_dim + j],
                    b->element[1][(k + 11) * real_col_dim + j],
                    b->element[1][(k + 10) * real_col_dim + j],
                    b->element[1][(k + 9) * real_col_dim + j],
                    b->element[1][(k + 8) * real_col_dim + j],
                    b->element[1][(k + 7) * real_col_dim + j],
                    b->element[1][(k + 6) * real_col_dim + j],
                    b->element[1][(k + 5) * real_col_dim + j],
                    b->element[1][(k + 4) * real_col_dim + j],
                    b->element[1][(k + 3) * real_col_dim + j],
                    b->element[1][(k + 2) * real_col_dim + j],
                    b->element[1][(k + 1) * real_col_dim + j],
                    b->element[1][(k + 0) * real_col_dim + j]
                );
                bkj[2] = _mm512_set_ps(
                    b->element[2][(k + 15) * real_col_dim + j],
                    b->element[2][(k + 14) * real_col_dim + j],
                    b->element[2][(k + 13) * real_col_dim + j],
                    b->element[2][(k + 12) * real_col_dim + j],
                    b->element[2][(k + 11) * real_col_dim + j],
                    b->element[2][(k + 10) * real_col_dim + j],
                    b->element[2][(k + 9) * real_col_dim + j],
                    b->element[2][(k + 8) * real_col_dim + j],
                    b->element[2][(k + 7) * real_col_dim + j],
                    b->element[2][(k + 6) * real_col_dim + j],
                    b->element[2][(k + 5) * real_col_dim + j],
                    b->element[2][(k + 4) * real_col_dim + j],
                    b->element[2][(k + 3) * real_col_dim + j],
                    b->element[2][(k + 2) * real_col_dim + j],
                    b->element[2][(k + 1) * real_col_dim + j],
                    b->element[2][(k + 0) * real_col_dim + j]
                );
                bkj[3] = _mm512_set_ps(
                    b->element[3][(k + 15) * real_col_dim + j],
                    b->element[3][(k + 14) * real_col_dim + j],
                    b->element[3][(k + 13) * real_col_dim + j],
                    b->element[3][(k + 12) * real_col_dim + j],
                    b->element[3][(k + 11) * real_col_dim + j],
                    b->element[3][(k + 10) * real_col_dim + j],
                    b->element[3][(k + 9) * real_col_dim + j],
                    b->element[3][(k + 8) * real_col_dim + j],
                    b->element[3][(k + 7) * real_col_dim + j],
                    b->element[3][(k + 6) * real_col_dim + j],
                    b->element[3][(k + 5) * real_col_dim + j],
                    b->element[3][(k + 4) * real_col_dim + j],
                    b->element[3][(k + 3) * real_col_dim + j],
                    b->element[3][(k + 2) * real_col_dim + j],
                    b->element[3][(k + 1) * real_col_dim + j],
                    b->element[3][(k + 0) * real_col_dim + j]
                );

            /*
                rqs_mul(tmp_mul[0].val, aik[0].val, bkj[0].val);
                rqs_mul(tmp_mul[1].val, aik[1].val, bkj[1].val);
                rqs_mul(tmp_mul[2].val, aik[2].val, bkj[2].val);
                rqs_mul(tmp_mul[3].val, aik[3].val, bkj[3].val);
            */
                _bncavx512_rqs_mul(tmp_mul, aik, bkj);

            /*
                rqs_add(cij.val, cij.val, tmp_mul[0].val);
                rqs_add(cij.val, cij.val, tmp_mul[1].val);
                rqs_add(cij.val, cij.val, tmp_mul[2].val);
                rqs_add(cij.val, cij.val, tmp_mul[3].val);
            */
                _bncavx512_rqs_add(cij, cij, tmp_mul);
            }

            cijval[0][0] = cij[0][0]; cijval[0][1] = cij[1][0]; cijval[0][2] = cij[2][0]; cijval[0][3] = cij[3][0];
            cijval[1][0] = cij[0][1]; cijval[1][1] = cij[1][1]; cijval[1][2] = cij[2][1]; cijval[1][3] = cij[3][1];
            cijval[2][0] = cij[0][2]; cijval[2][1] = cij[1][2]; cijval[2][2] = cij[2][2]; cijval[2][3] = cij[3][2];
            cijval[3][0] = cij[0][3]; cijval[3][1] = cij[1][3]; cijval[3][2] = cij[2][3]; cijval[3][3] = cij[3][3];
            cijval[4][0] = cij[0][4]; cijval[4][1] = cij[1][4]; cijval[4][2] = cij[2][4]; cijval[4][3] = cij[3][4];
            cijval[5][0] = cij[0][5]; cijval[5][1] = cij[1][5]; cijval[5][2] = cij[2][5]; cijval[5][3] = cij[3][5];
            cijval[6][0] = cij[0][6]; cijval[6][1] = cij[1][6]; cijval[6][2] = cij[2][6]; cijval[6][3] = cij[3][6];
            cijval[7][0] = cij[0][7]; cijval[7][1] = cij[1][7]; cijval[7][2] = cij[2][7]; cijval[7][3] = cij[3][7];
            cijval[8][0] = cij[0][8]; cijval[8][1] = cij[1][8]; cijval[8][2] = cij[2][8]; cijval[8][3] = cij[3][8];
            cijval[9][0] = cij[0][9]; cijval[9][1] = cij[1][9]; cijval[9][2] = cij[2][9]; cijval[9][3] = cij[3][9];
            cijval[10][0] = cij[0][10]; cijval[10][1] = cij[1][10]; cijval[10][2] = cij[2][10]; cijval[10][3] = cij[3][10];
            cijval[11][0] = cij[0][11]; cijval[11][1] = cij[1][11]; cijval[11][2] = cij[2][11]; cijval[11][3] = cij[3][11];
            cijval[12][0] = cij[0][12]; cijval[12][1] = cij[1][12]; cijval[12][2] = cij[2][12]; cijval[12][3] = cij[3][12];
            cijval[13][0] = cij[0][13]; cijval[13][1] = cij[1][13]; cijval[13][2] = cij[2][13]; cijval[13][3] = cij[3][13];
            cijval[14][0] = cij[0][14]; cijval[14][1] = cij[1][14]; cijval[14][2] = cij[2][14]; cijval[14][3] = cij[3][14];
            cijval[15][0] = cij[0][15]; cijval[15][1] = cij[1][15]; cijval[15][2] = cij[2][15]; cijval[15][3] = cij[3][15];

            rqs_add(cijval[0], cijval[0], cijval[1]);
            rqs_add(cijval[0], cijval[0], cijval[2]);
            rqs_add(cijval[0], cijval[0], cijval[3]);
            rqs_add(cijval[0], cijval[0], cijval[4]);
            rqs_add(cijval[0], cijval[0], cijval[5]);
            rqs_add(cijval[0], cijval[0], cijval[6]);
            rqs_add(cijval[0], cijval[0], cijval[7]);
            rqs_add(cijval[0], cijval[0], cijval[8]);
            rqs_add(cijval[0], cijval[0], cijval[9]);
            rqs_add(cijval[0], cijval[0], cijval[10]);
            rqs_add(cijval[0], cijval[0], cijval[11]);
            rqs_add(cijval[0], cijval[0], cijval[12]);
            rqs_add(cijval[0], cijval[0], cijval[13]);
            rqs_add(cijval[0], cijval[0], cijval[14]);
            rqs_add(cijval[0], cijval[0], cijval[15]);

            ret->element[0][i * real_col_dim + j] = cijval[0][0];
            ret->element[1][i * real_col_dim + j] = cijval[0][1];
            ret->element[2][i * real_col_dim + j] = cijval[0][2];
            ret->element[3][i * real_col_dim + j] = cijval[0][3];
       }
    }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic)
	{
		long real_row_dim = a->real_row_dim;
		long real_col_dim = b->real_col_dim;
		long real_mid_dim = a->real_col_dim;
		long vl = (long)svcntw();

		for(i = 0; i < real_row_dim; i++){
			for(j = 0; j < real_col_dim; j += vl){
				svbool_t pg = svwhilelt_b32_s32((int32_t)j, (int32_t)real_col_dim);
				svfloat32_t cij0, cij1, cij2, cij3;
				_bncsve2_rqs_set0(&cij0, &cij1, &cij2, &cij3);
				for(k = 0; k < real_mid_dim; k++){
					svfloat32_t aik0 = svdup_n_f32(a->element[0][i*real_mid_dim + k]);
					svfloat32_t aik1 = svdup_n_f32(a->element[1][i*real_mid_dim + k]);
					svfloat32_t aik2 = svdup_n_f32(a->element[2][i*real_mid_dim + k]);
					svfloat32_t aik3 = svdup_n_f32(a->element[3][i*real_mid_dim + k]);
					svfloat32_t bkj0 = svld1_f32(pg, &(b->element[0][k*real_col_dim + j]));
					svfloat32_t bkj1 = svld1_f32(pg, &(b->element[1][k*real_col_dim + j]));
					svfloat32_t bkj2 = svld1_f32(pg, &(b->element[2][k*real_col_dim + j]));
					svfloat32_t bkj3 = svld1_f32(pg, &(b->element[3][k*real_col_dim + j]));
					svfloat32_t t0, t1, t2, t3;
					_bncsve2_rqs_mul(pg, &t0, &t1, &t2, &t3,
					                 aik0, aik1, aik2, aik3,
					                 bkj0, bkj1, bkj2, bkj3);
					_bncsve2_rqs_add(pg, &cij0, &cij1, &cij2, &cij3,
					                 cij0, cij1, cij2, cij3,
					                 t0, t1, t2, t3);
				}
				svst1_f32(pg, &(ret->element[0][i*real_col_dim + j]), cij0);
				svst1_f32(pg, &(ret->element[1][i*real_col_dim + j]), cij1);
				svst1_f32(pg, &(ret->element[2][i*real_col_dim + j]), cij2);
				svst1_f32(pg, &(ret->element[3][i*real_col_dim + j]), cij3);
			}
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	long real_row_dim = a->real_row_dim;
	long real_col_dim = b->real_col_dim;
	long real_mid_dim = a->real_col_dim;
	float32x4_t cij[QSSIZE], aik[QSSIZE], bkj[QSSIZE], tmp[QSSIZE];

	for(i = 0; i < real_row_dim; i += 1){
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH){
			_bncneon_set0_qs(cij);
			for(k = 0; k < real_mid_dim; k += 1){
				aik[0] = vdupq_n_f32(a->element[0][i*real_mid_dim + k]);
				aik[1] = vdupq_n_f32(a->element[1][i*real_mid_dim + k]);
				aik[2] = vdupq_n_f32(a->element[2][i*real_mid_dim + k]);
				aik[3] = vdupq_n_f32(a->element[3][i*real_mid_dim + k]);
				bkj[0] = vld1q_f32(&(b->element[0][k*real_col_dim + j]));
				bkj[1] = vld1q_f32(&(b->element[1][k*real_col_dim + j]));
				bkj[2] = vld1q_f32(&(b->element[2][k*real_col_dim + j]));
				bkj[3] = vld1q_f32(&(b->element[3][k*real_col_dim + j]));
				_bncneon_rqs_mul(tmp, aik, bkj);
				_bncneon_rqs_add(cij, cij, tmp);
			}
			vst1q_f32(&(ret->element[0][i*real_col_dim + j]), cij[0]);
			vst1q_f32(&(ret->element[1][i*real_col_dim + j]), cij[1]);
			vst1q_f32(&(ret->element[2][i*real_col_dim + j]), cij[2]);
			vst1q_f32(&(ret->element[3][i*real_col_dim + j]), cij[3]);
		}
	}
#else // __AVX2__
	long row_dim, col_dim, mid_dim;
	float tmp[QSSIZE], ret_ij[QSSIZE];

	//printf("Non SIMD mul_qsmatrix(%ld, %ld)\n", ret->row_dim, ret->col_dim);
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = a->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			//rqs_set0(GET_QSMATRIX_IJ(ret, i, j));
			rqs_set0(ret_ij);
			for(k = 0; k < mid_dim; k++)
			{
				rqs_mul(tmp, GET_QSMATRIX_IJ(a, i, k), GET_QSMATRIX_IJ(b, k, j));
				//rqs_add(GET_QSMATRIX_IJ(ret, i, j), tmp, GET_QSMATRIX_IJ(ret, i, j));
				rqs_add(ret_ij, tmp, ret_ij);
			}
			set_qsmatrix_ij(ret, i, j, ret_ij);
		}
	}
	//printf("end(%ld, %ld)\n", ret->row_dim, ret->col_dim);
#endif // __AVX2__

}

// Frobenius norm
void normf_qsmatrix(float ret[QSSIZE], QSMatrix mat)
{
	long int i;
	long int real_total_dim;
	float tmp[QSSIZE];

	real_total_dim = mat->real_row_dim * mat->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 ret4[QSSIZE], mat4[QSSIZE], tmp4[QSSIZE];

	ret4[0] = _mm256_setzero_ps();
	ret4[1] = _mm256_setzero_ps();
	ret4[2] = _mm256_setzero_ps();
	ret4[3] = _mm256_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		mat4[0] = _mm256_load_ps(&mat->element[0][i]);
		mat4[1] = _mm256_load_ps(&mat->element[1][i]);
		mat4[2] = _mm256_load_ps(&mat->element[2][i]);
		mat4[3] = _mm256_load_ps(&mat->element[3][i]);

		// tmp4 := mat4[i]^2
		// ret4 += tmp4
		_bncavx2_rqs_mul(tmp4, mat4, mat4);
		_bncavx2_rqs_add(ret4, ret4, tmp4);
	}

	_bncavx2_rqs_sum256(ret, ret4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512 ret4[QSSIZE], mat4[QSSIZE], tmp4[QSSIZE];

	ret4[0] = _mm512_setzero_ps();
	ret4[1] = _mm512_setzero_ps();
	ret4[2] = _mm512_setzero_ps();
	ret4[3] = _mm512_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		mat4[0] = _mm512_load_ps(&mat->element[0][i]);
		mat4[1] = _mm512_load_ps(&mat->element[1][i]);
		mat4[2] = _mm512_load_ps(&mat->element[2][i]);
		mat4[3] = _mm512_load_ps(&mat->element[3][i]);

		// tmp4 := mat4[i]^2
		// ret4 += tmp4
		_bncavx512_rqs_mul(tmp4, mat4, mat4);
		_bncavx512_rqs_add(ret4, ret4, tmp4);
	}

	_bncavx512_rqs_sum512(ret, ret4);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t ret4_0, ret4_1, ret4_2, ret4_3;
	svfloat32_t mat4_0, mat4_1, mat4_2, mat4_3;
	svfloat32_t tmp4_0, tmp4_1, tmp4_2, tmp4_3;

	ret4_0 = svdup_f32(0.0f);
	ret4_1 = svdup_f32(0.0f);
	ret4_2 = svdup_f32(0.0f);
	ret4_3 = svdup_f32(0.0f);
	for(i = 0; i < real_total_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(real_total_dim));
		mat4_0 = svld1_f32(pg, &mat->element[0][i]);
		mat4_1 = svld1_f32(pg, &mat->element[1][i]);
		mat4_2 = svld1_f32(pg, &mat->element[2][i]);
		mat4_3 = svld1_f32(pg, &mat->element[3][i]);

		// tmp4 := mat4[i]^2
		// ret4 += tmp4
		_bncsve2_rqs_mul(svptrue_b32(), &tmp4_0, &tmp4_1, &tmp4_2, &tmp4_3, mat4_0, mat4_1, mat4_2, mat4_3, mat4_0, mat4_1, mat4_2, mat4_3);
		_bncsve2_rqs_add(svptrue_b32(), &ret4_0, &ret4_1, &ret4_2, &ret4_3, ret4_0, ret4_1, ret4_2, ret4_3, tmp4_0, tmp4_1, tmp4_2, tmp4_3);
	}

	_bncsve2_rqs_sum128(svptrue_b32(), ret, ret4_0, ret4_1, ret4_2, ret4_3);

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t ret4[QSSIZE], mat4[QSSIZE], tmp4[QSSIZE];

	ret4[0] = vdupq_n_f32(0.0f);
	ret4[1] = vdupq_n_f32(0.0f);
	ret4[2] = vdupq_n_f32(0.0f);
	ret4[3] = vdupq_n_f32(0.0f);
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		mat4[0] = vld1q_f32(&mat->element[0][i]);
		mat4[1] = vld1q_f32(&mat->element[1][i]);
		mat4[2] = vld1q_f32(&mat->element[2][i]);
		mat4[3] = vld1q_f32(&mat->element[3][i]);

		// tmp4 := mat4[i]^2
		// ret4 += tmp4
		_bncneon_rqs_mul(tmp4, mat4, mat4);
		_bncneon_rqs_add(ret4, ret4, tmp4);
	}

	_bncneon_rqs_sum128f(ret, ret4);

#else // others
	float mat1[QSSIZE];

	rqs_set0(ret);
	for(i = 0; i < real_total_dim; i++)
	{
		mat1[0] = mat->element[0][i];
		mat1[1] = mat->element[1][i];
		mat1[2] = mat->element[2][i];

		// tmp := mat1[i]^2
		// ret += tmp
		rqs_mul(tmp, mat1, mat1);
		rqs_add(ret, ret, tmp);
	}

#endif // __AVX2__

	rqs_sqrt(tmp, ret);
	rqs_set(ret, tmp);

}

// print normf
void print_normf_qsmatrix(const char *str, QSMatrix mat)
{
	static float tmp[QSSIZE];

	normf_qsmatrix(tmp, mat);

	if(str != NULL)
		printf("%s(%ld, %ld)", str, mat->row_dim, mat->col_dim);

	rqs_out_str(tmp); printf("\n");
}

/*************************************************/
/* Matrix Caluculations for QSMatrix            */
/*
void normf_qsmatrix(float ret[QSSIZE], QSMatrix mat)
void norm1_qsmatrix(float ret[QSSIZE], QSMatrix mat)
void normi_qsmatrix(float ret[QSSIZE], QSMatrix mat)
void add_qsmatrix(QSMatrix c, QSMatrix a, QSMatrix b);
void sub_qsmatrix(QSMatrix c, QSMatrix a, QSMatrix b);
void mul_qsmatrix(QSMatrix c, QSMatrix a, QSMatrix b);
void mul_qsmatrix_dsvec(QSVector v, QSMatrix a, QSVector vb)
void mul_qsmatrixt_dsvec(QSVector v, QSMatrix a, QSVector vb)
void transpose_qsmatrix(QSMatrix c, QSMatrix a);
void inv_qsmatrix(QSMatrix a);
void subst_mpfmatrux(QSMatrix c, QSMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_qsmatrix(float ret[QSSIZE], QSMatrix mat)
{
	long int i, j;
	float tmp[QSSIZE], sum[QSSIZE];

	set0_qs(ret);
	for(i = 0; i < mat->row_dim; i++)
	{
		set0_qs(sum);
		for(j = 0; j < mat->col_dim; j++)
		{
			rqs_abs(tmp, get_qsmatrix_ij(mat, i, j));
			rqs_add(sum, sum, tmp);
		}
		if(rqs_cmp(ret, sum) < 0)
			rqs_set(ret, sum);
	}

	return;
}

/* 1 Norm of Matrix */
void norm1_qsmatrix(float ret[QSSIZE], QSMatrix mat)
{
	long int i, j;
	float tmp[QSSIZE], sum[QSSIZE];

	rqs_set_ui(ret, 0UL);

	for(j = 0; j < mat->col_dim; j++)
	{
		rqs_set_ui(sum, 0UL);
		for(i = 0; i < mat->row_dim; i++)
		{
			rqs_abs(tmp, get_qsmatrix_ij(mat, i, j));
			rqs_add(sum, sum, tmp);
		}
		if(rqs_cmp(ret, sum) < 0)
			rqs_set(ret, sum);
	}

	return;
}

/* c := a + b */
void add_qsmatrix(QSMatrix c, QSMatrix a, QSMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_qsmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_qsmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 tmp4[QSSIZE], aij4[QSSIZE], bij4[QSSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = _mm256_load_ps(&(a->element[0][index]));
		aij4[1] = _mm256_load_ps(&(a->element[1][index]));
		aij4[2] = _mm256_load_ps(&(a->element[2][index]));
		aij4[3] = _mm256_load_ps(&(a->element[3][index]));

		bij4[0] = _mm256_load_ps(&(b->element[0][index]));
		bij4[1] = _mm256_load_ps(&(b->element[1][index]));
		bij4[2] = _mm256_load_ps(&(b->element[2][index]));
		bij4[3] = _mm256_load_ps(&(b->element[3][index]));

		_bncavx2_rqs_add(tmp4, aij4, bij4);

		_mm256_store_ps(&(c->element[0][index]), tmp4[0]);
		_mm256_store_ps(&(c->element[1][index]), tmp4[1]); 
		_mm256_store_ps(&(c->element[2][index]), tmp4[2]); 
		_mm256_store_ps(&(c->element[3][index]), tmp4[3]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 tmp4[QSSIZE], aij4[QSSIZE], bij4[QSSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = _mm512_load_ps(&(a->element[0][index]));
		aij4[1] = _mm512_load_ps(&(a->element[1][index]));
		aij4[2] = _mm512_load_ps(&(a->element[2][index]));
		aij4[3] = _mm512_load_ps(&(a->element[3][index]));

		bij4[0] = _mm512_load_ps(&(b->element[0][index]));
		bij4[1] = _mm512_load_ps(&(b->element[1][index]));
		bij4[2] = _mm512_load_ps(&(b->element[2][index]));
		bij4[3] = _mm512_load_ps(&(b->element[3][index]));

		_bncavx512_rqs_add(tmp4, aij4, bij4);

		_mm512_store_ps(&(c->element[0][index]), tmp4[0]);
		_mm512_store_ps(&(c->element[1][index]), tmp4[1]); 
		_mm512_store_ps(&(c->element[2][index]), tmp4[2]); 
		_mm512_store_ps(&(c->element[3][index]), tmp4[3]); 
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t tmp4_0, tmp4_1, tmp4_2, tmp4_3;
	svfloat32_t aij4_0, aij4_1, aij4_2, aij4_3;
	svfloat32_t bij4_0, bij4_1, bij4_2, bij4_3;

	for(index = 0; index < real_total_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(real_total_dim));
		aij4_0 = svld1_f32(pg, &(a->element[0][index]));
		aij4_1 = svld1_f32(pg, &(a->element[1][index]));
		aij4_2 = svld1_f32(pg, &(a->element[2][index]));
		aij4_3 = svld1_f32(pg, &(a->element[3][index]));

		bij4_0 = svld1_f32(pg, &(b->element[0][index]));
		bij4_1 = svld1_f32(pg, &(b->element[1][index]));
		bij4_2 = svld1_f32(pg, &(b->element[2][index]));
		bij4_3 = svld1_f32(pg, &(b->element[3][index]));

		_bncsve2_rqs_add(pg, &tmp4_0, &tmp4_1, &tmp4_2, &tmp4_3, aij4_0, aij4_1, aij4_2, aij4_3, bij4_0, bij4_1, bij4_2, bij4_3);

		svst1_f32(pg, &(c->element[0][index]), tmp4_0);
		svst1_f32(pg, &(c->element[1][index]), tmp4_1); 
		svst1_f32(pg, &(c->element[2][index]), tmp4_2); 
		svst1_f32(pg, &(c->element[3][index]), tmp4_3); 
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t tmp4[QSSIZE], aij4[QSSIZE], bij4[QSSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = vld1q_f32(&(a->element[0][index]));
		aij4[1] = vld1q_f32(&(a->element[1][index]));
		aij4[2] = vld1q_f32(&(a->element[2][index]));
		aij4[3] = vld1q_f32(&(a->element[3][index]));

		bij4[0] = vld1q_f32(&(b->element[0][index]));
		bij4[1] = vld1q_f32(&(b->element[1][index]));
		bij4[2] = vld1q_f32(&(b->element[2][index]));
		bij4[3] = vld1q_f32(&(b->element[3][index]));

		_bncneon_rqs_add(tmp4, aij4, bij4);

		vst1q_f32(&(c->element[0][index]), tmp4[0]);
		vst1q_f32(&(c->element[1][index]), tmp4[1]); 
		vst1q_f32(&(c->element[2][index]), tmp4[2]); 
		vst1q_f32(&(c->element[3][index]), tmp4[3]); 
	}
#else // others
	float tmp[QSSIZE], aij[QSSIZE], bij[QSSIZE];

	for(index = 0; index < real_total_dim; index++)
	{
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];
		aij[2] = a->element[2][index];
		aij[3] = a->element[3][index];

		bij[0] = b->element[0][index];
		bij[1] = b->element[1][index];
		bij[2] = b->element[2][index];
		bij[3] = b->element[3][index];

		rqs_add(tmp, aij, bij);

		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1]; 
		c->element[2][index] = tmp[2]; 
		c->element[3][index] = tmp[3]; 
	}
#endif // __AVX2__
/*
	float tmp[QSSIZE];

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rqs_add(tmp, get_qsmatrix_ij(a, i, j), get_qsmatrix_ij(b, i, j));
			set_qsmatrix_ij(c, i, j, tmp);
		}
	}
*/
}

/* c := a - b */
void sub_qsmatrix(QSMatrix c, QSMatrix a, QSMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: sub_qsmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: sub_qsmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

/*
	float tmp[QSSIZE];
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rqs_sub(tmp, get_qsmatrix_ij(a, i, j), get_qsmatrix_ij(b, i, j));
			set_qsmatrix_ij(c, i, j, tmp);
		}
	}
*/

// for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 tmp4[QSSIZE], aij4[QSSIZE], bij4[QSSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = _mm256_load_ps(&(a->element[0][index]));
		aij4[1] = _mm256_load_ps(&(a->element[1][index]));
		aij4[2] = _mm256_load_ps(&(a->element[2][index]));
		aij4[3] = _mm256_load_ps(&(a->element[3][index]));

		bij4[0] = _mm256_load_ps(&(b->element[0][index]));
		bij4[1] = _mm256_load_ps(&(b->element[1][index]));
		bij4[2] = _mm256_load_ps(&(b->element[2][index]));
		bij4[3] = _mm256_load_ps(&(b->element[3][index]));

		_bncavx2_rqs_sub(tmp4, aij4, bij4);

		_mm256_store_ps(&(c->element[0][index]), tmp4[0]);
		_mm256_store_ps(&(c->element[1][index]), tmp4[1]); 
		_mm256_store_ps(&(c->element[2][index]), tmp4[2]); 
		_mm256_store_ps(&(c->element[3][index]), tmp4[3]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 tmp4[QSSIZE], aij4[QSSIZE], bij4[QSSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = _mm512_load_ps(&(a->element[0][index]));
		aij4[1] = _mm512_load_ps(&(a->element[1][index]));
		aij4[2] = _mm512_load_ps(&(a->element[2][index]));
		aij4[3] = _mm512_load_ps(&(a->element[3][index]));

		bij4[0] = _mm512_load_ps(&(b->element[0][index]));
		bij4[1] = _mm512_load_ps(&(b->element[1][index]));
		bij4[2] = _mm512_load_ps(&(b->element[2][index]));
		bij4[3] = _mm512_load_ps(&(b->element[3][index]));

		_bncavx512_rqs_sub(tmp4, aij4, bij4);

		_mm512_store_ps(&(c->element[0][index]), tmp4[0]);
		_mm512_store_ps(&(c->element[1][index]), tmp4[1]); 
		_mm512_store_ps(&(c->element[2][index]), tmp4[2]); 
		_mm512_store_ps(&(c->element[3][index]), tmp4[3]); 
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t tmp4_0, tmp4_1, tmp4_2, tmp4_3;
	svfloat32_t aij4_0, aij4_1, aij4_2, aij4_3;
	svfloat32_t bij4_0, bij4_1, bij4_2, bij4_3;

	for(index = 0; index < real_total_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(real_total_dim));
		aij4_0 = svld1_f32(pg, &(a->element[0][index]));
		aij4_1 = svld1_f32(pg, &(a->element[1][index]));
		aij4_2 = svld1_f32(pg, &(a->element[2][index]));
		aij4_3 = svld1_f32(pg, &(a->element[3][index]));

		bij4_0 = svld1_f32(pg, &(b->element[0][index]));
		bij4_1 = svld1_f32(pg, &(b->element[1][index]));
		bij4_2 = svld1_f32(pg, &(b->element[2][index]));
		bij4_3 = svld1_f32(pg, &(b->element[3][index]));

		_bncsve2_rqs_neg(pg, &tmp4_0, &tmp4_1, &tmp4_2, &tmp4_3, bij4_0, bij4_1, bij4_2, bij4_3);
		_bncsve2_rqs_add(pg, &tmp4_0, &tmp4_1, &tmp4_2, &tmp4_3, aij4_0, aij4_1, aij4_2, aij4_3, tmp4_0, tmp4_1, tmp4_2, tmp4_3);

		svst1_f32(pg, &(c->element[0][index]), tmp4_0);
		svst1_f32(pg, &(c->element[1][index]), tmp4_1); 
		svst1_f32(pg, &(c->element[2][index]), tmp4_2); 
		svst1_f32(pg, &(c->element[3][index]), tmp4_3); 
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t tmp4[QSSIZE], aij4[QSSIZE], bij4[QSSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = vld1q_f32(&(a->element[0][index]));
		aij4[1] = vld1q_f32(&(a->element[1][index]));
		aij4[2] = vld1q_f32(&(a->element[2][index]));
		aij4[3] = vld1q_f32(&(a->element[3][index]));

		bij4[0] = vld1q_f32(&(b->element[0][index]));
		bij4[1] = vld1q_f32(&(b->element[1][index]));
		bij4[2] = vld1q_f32(&(b->element[2][index]));
		bij4[3] = vld1q_f32(&(b->element[3][index]));

		_bncneon_rqs_sub(tmp4, aij4, bij4);

		vst1q_f32(&(c->element[0][index]), tmp4[0]);
		vst1q_f32(&(c->element[1][index]), tmp4[1]); 
		vst1q_f32(&(c->element[2][index]), tmp4[2]); 
		vst1q_f32(&(c->element[3][index]), tmp4[3]); 
	}
#else // others
	float tmp[QSSIZE], aij[QSSIZE], bij[QSSIZE];

	for(index = 0; index < real_total_dim; index++)
	{
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];
		aij[2] = a->element[2][index];
		aij[3] = a->element[3][index];

		bij[0] = b->element[0][index];
		bij[1] = b->element[1][index];
		bij[2] = b->element[2][index];
		bij[3] = b->element[3][index];

		rqs_sub(tmp, aij, bij);

		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1]; 
		c->element[2][index] = tmp[2]; 
		c->element[3][index] = tmp[3]; 
	}
#endif // __AVX2__
}

/* c := sc * a */
void cmul_qsmatrix(QSMatrix c, float sc[QSSIZE], QSMatrix a)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_qsmatrix(row_dim is differnt! : c->(%ld, %ld), a->(%ld, %ld)\n", c->row_dim, c->col_dim, a->row_dim, a->col_dim);
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_qsmatrix(col_dim is differnt! : c->(%ld, %ld), a->(%ld, %ld)\n", c->row_dim, c->col_dim, a->row_dim, a->col_dim);
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

// for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 tmp4[QSSIZE], sc4[QSSIZE], aij4[QSSIZE];

	sc4[0] = _mm256_set1_ps(sc[0]);
	sc4[1] = _mm256_set1_ps(sc[1]);
	sc4[2] = _mm256_set1_ps(sc[2]);
	sc4[3] = _mm256_set1_ps(sc[3]);

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = _mm256_load_ps(&(a->element[0][index]));
		aij4[1] = _mm256_load_ps(&(a->element[1][index]));
		aij4[2] = _mm256_load_ps(&(a->element[2][index]));
		aij4[3] = _mm256_load_ps(&(a->element[3][index]));

		_bncavx2_rqs_mul(tmp4, sc4, aij4);

		_mm256_store_ps(&(c->element[0][index]), tmp4[0]);
		_mm256_store_ps(&(c->element[1][index]), tmp4[1]); 
		_mm256_store_ps(&(c->element[2][index]), tmp4[2]); 
		_mm256_store_ps(&(c->element[3][index]), tmp4[3]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 tmp4[QSSIZE], sc4[QSSIZE], aij4[QSSIZE];

	sc4[0] = _mm512_set1_ps(sc[0]);
	sc4[1] = _mm512_set1_ps(sc[1]);
	sc4[2] = _mm512_set1_ps(sc[2]);
	sc4[3] = _mm512_set1_ps(sc[3]);

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = _mm512_load_ps(&(a->element[0][index]));
		aij4[1] = _mm512_load_ps(&(a->element[1][index]));
		aij4[2] = _mm512_load_ps(&(a->element[2][index]));
		aij4[3] = _mm512_load_ps(&(a->element[3][index]));

		_bncavx512_rqs_mul(tmp4, sc4, aij4);

		_mm512_store_ps(&(c->element[0][index]), tmp4[0]);
		_mm512_store_ps(&(c->element[1][index]), tmp4[1]); 
		_mm512_store_ps(&(c->element[2][index]), tmp4[2]); 
		_mm512_store_ps(&(c->element[3][index]), tmp4[3]); 
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t tmp4_0, tmp4_1, tmp4_2, tmp4_3;
	svfloat32_t sc4_0, sc4_1, sc4_2, sc4_3;
	svfloat32_t aij4_0, aij4_1, aij4_2, aij4_3;

	sc4_0 = svdup_f32(sc[0]);
	sc4_1 = svdup_f32(sc[1]);
	sc4_2 = svdup_f32(sc[2]);
	sc4_3 = svdup_f32(sc[3]);

	for(index = 0; index < real_total_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(real_total_dim));
		aij4_0 = svld1_f32(pg, &(a->element[0][index]));
		aij4_1 = svld1_f32(pg, &(a->element[1][index]));
		aij4_2 = svld1_f32(pg, &(a->element[2][index]));
		aij4_3 = svld1_f32(pg, &(a->element[3][index]));

		_bncsve2_rqs_mul(pg, &tmp4_0, &tmp4_1, &tmp4_2, &tmp4_3, sc4_0, sc4_1, sc4_2, sc4_3, aij4_0, aij4_1, aij4_2, aij4_3);

		svst1_f32(pg, &(c->element[0][index]), tmp4_0);
		svst1_f32(pg, &(c->element[1][index]), tmp4_1); 
		svst1_f32(pg, &(c->element[2][index]), tmp4_2); 
		svst1_f32(pg, &(c->element[3][index]), tmp4_3); 
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t tmp4[QSSIZE], sc4[QSSIZE], aij4[QSSIZE];

	sc4[0] = vdupq_n_f32(sc[0]);
	sc4[1] = vdupq_n_f32(sc[1]);
	sc4[2] = vdupq_n_f32(sc[2]);
	sc4[3] = vdupq_n_f32(sc[3]);

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = vld1q_f32(&(a->element[0][index]));
		aij4[1] = vld1q_f32(&(a->element[1][index]));
		aij4[2] = vld1q_f32(&(a->element[2][index]));
		aij4[3] = vld1q_f32(&(a->element[3][index]));

		_bncneon_rqs_mul(tmp4, sc4, aij4);

		vst1q_f32(&(c->element[0][index]), tmp4[0]);
		vst1q_f32(&(c->element[1][index]), tmp4[1]); 
		vst1q_f32(&(c->element[2][index]), tmp4[2]); 
		vst1q_f32(&(c->element[3][index]), tmp4[3]); 
	}
#else // others
	float tmp[QSSIZE], aij[QSSIZE];

	for(index = 0; index < real_total_dim; index++)
	{
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];
		aij[2] = a->element[2][index];
		aij[3] = a->element[3][index];

		rqs_mul(tmp, sc, aij);

		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1]; 
		c->element[2][index] = tmp[2]; 
		c->element[3][index] = tmp[3]; 
	}
#endif // __AVX2__
/*
	float tmp[QSSIZE];

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rqs_mul(tmp, sc, get_qsmatrix_ij(a, i, j));
			set_qsmatrix_ij(c, i, j, tmp);
		}
	}
*/
}

/* c = a^T */
void transpose_qsmatrix(QSMatrix c, QSMatrix a)
{
	long int i, j, index;
	long int real_row_dim, real_col_dim;

	/* Check Dimentions */
	if((c->row_dim != a->col_dim) || (c->col_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: transpose_dmatrix\n");
		return;
	}

	real_row_dim = c->real_row_dim;
	real_col_dim = c->real_col_dim;
// AVX2
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 aji4[QSSIZE];

	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, j, i));
			aji4[0] = _mm256_set_ps(
				a->element[0][(j + 7) * real_col_dim + i],
				a->element[0][(j + 6) * real_col_dim + i],
				a->element[0][(j + 5) * real_col_dim + i],
				a->element[0][(j + 4) * real_col_dim + i],
				a->element[0][(j + 3) * real_col_dim + i],
				a->element[0][(j + 2) * real_col_dim + i],
				a->element[0][(j + 1) * real_col_dim + i],
				a->element[0][(j    ) * real_col_dim + i]
			);
			aji4[1] = _mm256_set_ps(
				a->element[1][(j + 7) * real_col_dim + i],
				a->element[1][(j + 6) * real_col_dim + i],
				a->element[1][(j + 5) * real_col_dim + i],
				a->element[1][(j + 4) * real_col_dim + i],
				a->element[1][(j + 3) * real_col_dim + i],
				a->element[1][(j + 2) * real_col_dim + i],
				a->element[1][(j + 1) * real_col_dim + i],
				a->element[1][(j    ) * real_col_dim + i]
			);
			aji4[2] = _mm256_set_ps(
				a->element[2][(j + 7) * real_col_dim + i],
				a->element[2][(j + 6) * real_col_dim + i],
				a->element[2][(j + 5) * real_col_dim + i],
				a->element[2][(j + 4) * real_col_dim + i],
				a->element[2][(j + 3) * real_col_dim + i],
				a->element[2][(j + 2) * real_col_dim + i],
				a->element[2][(j + 1) * real_col_dim + i],
				a->element[2][(j    ) * real_col_dim + i]
			);
			aji4[3] = _mm256_set_ps(
				a->element[3][(j + 7) * real_col_dim + i],
				a->element[3][(j + 6) * real_col_dim + i],
				a->element[3][(j + 5) * real_col_dim + i],
				a->element[3][(j + 4) * real_col_dim + i],
				a->element[3][(j + 3) * real_col_dim + i],
				a->element[3][(j + 2) * real_col_dim + i],
				a->element[3][(j + 1) * real_col_dim + i],
				a->element[3][(j    ) * real_col_dim + i]
			);
			index = i * real_col_dim + j;
			_mm256_store_ps(&(c->element[0][index]), aji4[0]);
			_mm256_store_ps(&(c->element[1][index]), aji4[1]);
			_mm256_store_ps(&(c->element[2][index]), aji4[2]);
			_mm256_store_ps(&(c->element[3][index]), aji4[3]);
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 aji4[QSSIZE];

	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, j, i));
			aji4[0] = _mm512_set_ps(
                    a->element[0][(j + 15) * real_col_dim + i],
                    a->element[0][(j + 14) * real_col_dim + i],
                    a->element[0][(j + 13) * real_col_dim + i],
                    a->element[0][(j + 12) * real_col_dim + i],
                    a->element[0][(j + 11) * real_col_dim + i],
                    a->element[0][(j + 10) * real_col_dim + i],
                    a->element[0][(j + 9) * real_col_dim + i],
                    a->element[0][(j + 8) * real_col_dim + i],
                    a->element[0][(j + 7) * real_col_dim + i],
                    a->element[0][(j + 6) * real_col_dim + i],
                    a->element[0][(j + 5) * real_col_dim + i],
                    a->element[0][(j + 4) * real_col_dim + i],
                    a->element[0][(j + 3) * real_col_dim + i],
                    a->element[0][(j + 2) * real_col_dim + i],
                    a->element[0][(j + 1) * real_col_dim + i],
                    a->element[0][(j + 0) * real_col_dim + i]
                );
			aji4[1] = _mm512_set_ps(
                    a->element[1][(j + 15) * real_col_dim + i],
                    a->element[1][(j + 14) * real_col_dim + i],
                    a->element[1][(j + 13) * real_col_dim + i],
                    a->element[1][(j + 12) * real_col_dim + i],
                    a->element[1][(j + 11) * real_col_dim + i],
                    a->element[1][(j + 10) * real_col_dim + i],
                    a->element[1][(j + 9) * real_col_dim + i],
                    a->element[1][(j + 8) * real_col_dim + i],
                    a->element[1][(j + 7) * real_col_dim + i],
                    a->element[1][(j + 6) * real_col_dim + i],
                    a->element[1][(j + 5) * real_col_dim + i],
                    a->element[1][(j + 4) * real_col_dim + i],
                    a->element[1][(j + 3) * real_col_dim + i],
                    a->element[1][(j + 2) * real_col_dim + i],
                    a->element[1][(j + 1) * real_col_dim + i],
                    a->element[1][(j + 0) * real_col_dim + i]
                );
			aji4[2] = _mm512_set_ps(
                    a->element[2][(j + 15) * real_col_dim + i],
                    a->element[2][(j + 14) * real_col_dim + i],
                    a->element[2][(j + 13) * real_col_dim + i],
                    a->element[2][(j + 12) * real_col_dim + i],
                    a->element[2][(j + 11) * real_col_dim + i],
                    a->element[2][(j + 10) * real_col_dim + i],
                    a->element[2][(j + 9) * real_col_dim + i],
                    a->element[2][(j + 8) * real_col_dim + i],
                    a->element[2][(j + 7) * real_col_dim + i],
                    a->element[2][(j + 6) * real_col_dim + i],
                    a->element[2][(j + 5) * real_col_dim + i],
                    a->element[2][(j + 4) * real_col_dim + i],
                    a->element[2][(j + 3) * real_col_dim + i],
                    a->element[2][(j + 2) * real_col_dim + i],
                    a->element[2][(j + 1) * real_col_dim + i],
                    a->element[2][(j + 0) * real_col_dim + i]
                );
			aji4[3] = _mm512_set_ps(
                    a->element[3][(j + 15) * real_col_dim + i],
                    a->element[3][(j + 14) * real_col_dim + i],
                    a->element[3][(j + 13) * real_col_dim + i],
                    a->element[3][(j + 12) * real_col_dim + i],
                    a->element[3][(j + 11) * real_col_dim + i],
                    a->element[3][(j + 10) * real_col_dim + i],
                    a->element[3][(j + 9) * real_col_dim + i],
                    a->element[3][(j + 8) * real_col_dim + i],
                    a->element[3][(j + 7) * real_col_dim + i],
                    a->element[3][(j + 6) * real_col_dim + i],
                    a->element[3][(j + 5) * real_col_dim + i],
                    a->element[3][(j + 4) * real_col_dim + i],
                    a->element[3][(j + 3) * real_col_dim + i],
                    a->element[3][(j + 2) * real_col_dim + i],
                    a->element[3][(j + 1) * real_col_dim + i],
                    a->element[3][(j + 0) * real_col_dim + i]
                );
			index = i * real_col_dim + j;
			_mm512_store_ps(&(c->element[0][index]), aji4[0]);
			_mm512_store_ps(&(c->element[1][index]), aji4[1]);
			_mm512_store_ps(&(c->element[2][index]), aji4[2]);
			_mm512_store_ps(&(c->element[3][index]), aji4[3]);
		}
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, gather)
    svfloat32_t aji_0, aji_1, aji_2, aji_3;
    svint32_t vidx;
    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j += (long int)svcntw())
        {
            svbool_t pg = svwhilelt_b32_s64((int64_t)j, (int64_t)real_col_dim);
            vidx = svindex_s32((int32_t)(j * real_col_dim + i), (int32_t)real_col_dim);
            aji_0 = svld1_gather_s32index_f32(pg, a->element[0], vidx);
            aji_1 = svld1_gather_s32index_f32(pg, a->element[1], vidx);
            aji_2 = svld1_gather_s32index_f32(pg, a->element[2], vidx);
            aji_3 = svld1_gather_s32index_f32(pg, a->element[3], vidx);
            index = i * real_col_dim + j;
            svst1_f32(pg, &(c->element[0][index]), aji_0);
            svst1_f32(pg, &(c->element[1][index]), aji_1);
            svst1_f32(pg, &(c->element[2][index]), aji_2);
            svst1_f32(pg, &(c->element[3][index]), aji_3);
        }
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    long int idx0;
    for(i = 0; i < real_row_dim; i++)
        for(j = 0; j < real_col_dim; j++)
        {
            idx0 = j * real_col_dim + i; index = i * real_col_dim + j;
            c->element[0][index] = a->element[0][idx0];
            c->element[1][index] = a->element[1][idx0];
            c->element[2][index] = a->element[2][idx0];
            c->element[3][index] = a->element[3][idx0];
        }
#else // others
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_qsmatrix_ij(c, i, j, get_qsmatrix_ij(a, j, i));
	}
#endif // AVX2
}

/* c := a */
void subst_qsmatrix(QSMatrix c, QSMatrix a)
{
	long int i, j, index;
	long int real_row_dim, real_col_dim;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_qsmatrix\n");
		return;
	}
	real_row_dim = c->real_row_dim;
	real_col_dim = c->real_col_dim;

// AVX2
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j));
			index = i * real_col_dim + j;
			_mm256_store_ps(&(c->element[0][index]), _mm256_load_ps(&(a->element[0][index])));
			_mm256_store_ps(&(c->element[1][index]), _mm256_load_ps(&(a->element[1][index])));
			_mm256_store_ps(&(c->element[2][index]), _mm256_load_ps(&(a->element[2][index])));
			_mm256_store_ps(&(c->element[3][index]), _mm256_load_ps(&(a->element[3][index])));
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j));
			index = i * real_col_dim + j;
			_mm512_store_ps(&(c->element[0][index]), _mm512_load_ps(&(a->element[0][index])));
			_mm512_store_ps(&(c->element[1][index]), _mm512_load_ps(&(a->element[1][index])));
			_mm512_store_ps(&(c->element[2][index]), _mm512_load_ps(&(a->element[2][index])));
			_mm512_store_ps(&(c->element[3][index]), _mm512_load_ps(&(a->element[3][index])));
		}
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += (long int)svcntw())
		{
		svbool_t pg = svwhilelt_b32_s64((int64_t)j, (int64_t)(real_col_dim));
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j));
			index = i * real_col_dim + j;
			svst1_f32(pg, &(c->element[0][index]), svld1_f32(pg, &(a->element[0][index])));
			svst1_f32(pg, &(c->element[1][index]), svld1_f32(pg, &(a->element[1][index])));
			svst1_f32(pg, &(c->element[2][index]), svld1_f32(pg, &(a->element[2][index])));
			svst1_f32(pg, &(c->element[3][index]), svld1_f32(pg, &(a->element[3][index])));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j));
			index = i * real_col_dim + j;
			vst1q_f32(&(c->element[0][index]), vld1q_f32(&(a->element[0][index])));
			vst1q_f32(&(c->element[1][index]), vld1q_f32(&(a->element[1][index])));
			vst1q_f32(&(c->element[2][index]), vld1q_f32(&(a->element[2][index])));
			vst1q_f32(&(c->element[3][index]), vld1q_f32(&(a->element[3][index])));
		}
	}
#else // others
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_qsmatrix_ij(c, i, j, get_qsmatrix_ij(a, i, j));
		}
	}
#endif // AVX2
}

/* c := I */
void setI_qsmatrix(QSMatrix c)
{
	long int i, j;
	long int real_total_dim;
	float tmp1[QSSIZE];

	real_total_dim = c->real_row_dim * c->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 zero4;

	zero4 = _mm256_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm256_store_ps(&c->element[0][i], zero4);
		_mm256_store_ps(&c->element[1][i], zero4);
		_mm256_store_ps(&c->element[2][i], zero4);
		_mm256_store_ps(&c->element[3][i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero4;

	zero4 = _mm512_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm512_store_ps(&c->element[0][i], zero4);
		_mm512_store_ps(&c->element[1][i], zero4);
		_mm512_store_ps(&c->element[2][i], zero4);
		_mm512_store_ps(&c->element[3][i], zero4);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t zero4;

	zero4 = svdup_f32(0.0f);
	for(i = 0; i < real_total_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(real_total_dim));
		svst1_f32(pg, &c->element[0][i], zero4);
		svst1_f32(pg, &c->element[1][i], zero4);
		svst1_f32(pg, &c->element[2][i], zero4);
		svst1_f32(pg, &c->element[3][i], zero4);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t zero4;

	zero4 = vdupq_n_f32(0.0f);
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		vst1q_f32(&c->element[0][i], zero4);
		vst1q_f32(&c->element[1][i], zero4);
		vst1q_f32(&c->element[2][i], zero4);
		vst1q_f32(&c->element[3][i], zero4);
	}
#else // others
	for(i = 0; i < real_total_dim; i++)
	{
		c->element[0][i] = 0.0;
		c->element[1][i] = 0.0;	
		c->element[2][i] = 0.0;	
		c->element[3][i] = 0.0;	
	}
#endif // __AVX2__

	rqs_set_ui(tmp1, 1UL);

	for(i = 0; i < c->row_dim; i++)
	{
		if(i < c->col_dim)
			set_qsmatrix_ij(c, i, i, tmp1);
	}
}

/* v := a * vb */
void mul_qsmatrix_qsvec(QSVector v, QSMatrix a, QSVector vb)
{
	long int i, j;
	float tmp[QSSIZE], tmp1[QSSIZE];

	/* Check Dimension */
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: mul_qsmatrix_qsvec\n");
		return;
	}
// SIMD
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int ij_index, real_col_dim;
	__m256 tmp4[QSSIZE], tmp1_4[QSSIZE];
	__m256 aij4[QSSIZE], vbj4[QSSIZE];

	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->row_dim; i++)
	{
		//rqs_set_ui(tmp, 0UL);
		_bncavx2_set0_qs(tmp4);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij4[0] = _mm256_load_ps(&(a->element[0][ij_index]));
			aij4[1] = _mm256_load_ps(&(a->element[1][ij_index]));
			aij4[2] = _mm256_load_ps(&(a->element[2][ij_index]));
			aij4[3] = _mm256_load_ps(&(a->element[3][ij_index]));

			vbj4[0] = _mm256_load_ps(&(vb->element[0][j]));
			vbj4[1] = _mm256_load_ps(&(vb->element[1][j]));
			vbj4[2] = _mm256_load_ps(&(vb->element[2][j]));
			vbj4[3] = _mm256_load_ps(&(vb->element[3][j]));

			//rqs_mul(tmp1, get_qsmatrix_ij(a, i, j), get_qsvector_i(vb, j));
			//rqs_add(tmp, tmp, tmp1);
			_bncavx2_rqs_mul(tmp1_4, aij4, vbj4);
			_bncavx2_rqs_add(tmp4, tmp4, tmp1_4);
		}
		//set_qsvector_i(v, i, tmp);
		_bncavx2_rqs_sum256(tmp, tmp4);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
		v->element[2][i] = tmp[2];
		v->element[3][i] = tmp[3];
	}

#elif defined(__AVX512F__) // __AVX512F__
	long int ij_index, real_col_dim;
	__m512 tmp4[QSSIZE], tmp1_4[QSSIZE];
	__m512 aij4[QSSIZE], vbj4[QSSIZE];

	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->row_dim; i++)
	{
		//rqs_set_ui(tmp, 0UL);
		_bncavx512_set0_qs(tmp4);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij4[0] = _mm512_load_ps(&(a->element[0][ij_index]));
			aij4[1] = _mm512_load_ps(&(a->element[1][ij_index]));
			aij4[2] = _mm512_load_ps(&(a->element[2][ij_index]));
			aij4[3] = _mm512_load_ps(&(a->element[3][ij_index]));

			vbj4[0] = _mm512_load_ps(&(vb->element[0][j]));
			vbj4[1] = _mm512_load_ps(&(vb->element[1][j]));
			vbj4[2] = _mm512_load_ps(&(vb->element[2][j]));
			vbj4[3] = _mm512_load_ps(&(vb->element[3][j]));

			//rqs_mul(tmp1, get_qsmatrix_ij(a, i, j), get_qsvector_i(vb, j));
			//rqs_add(tmp, tmp, tmp1);
			_bncavx512_rqs_mul(tmp1_4, aij4, vbj4);
			_bncavx512_rqs_add(tmp4, tmp4, tmp1_4);
		}
		//set_qsvector_i(v, i, tmp);
		_bncavx512_rqs_sum512(tmp, tmp4);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
		v->element[2][i] = tmp[2];
		v->element[3][i] = tmp[3];
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic)
	{
		long real_col_dim = a->real_col_dim;
		long vl = (long)svcntw();
		for(i = 0; i < a->row_dim; i++)
		{
			svfloat32_t acc0, acc1, acc2, acc3;
			_bncsve2_rqs_set0(&acc0, &acc1, &acc2, &acc3);
			for(j = 0; j < real_col_dim; j += vl)
			{
				svbool_t pg = svwhilelt_b32_s32((int32_t)j, (int32_t)real_col_dim);
				long ij = i * real_col_dim + j;
				svfloat32_t a0 = svld1_f32(pg, &(a->element[0][ij]));
				svfloat32_t a1 = svld1_f32(pg, &(a->element[1][ij]));
				svfloat32_t a2v= svld1_f32(pg, &(a->element[2][ij]));
				svfloat32_t a3 = svld1_f32(pg, &(a->element[3][ij]));
				svfloat32_t b0 = svld1_f32(pg, &(vb->element[0][j]));
				svfloat32_t b1 = svld1_f32(pg, &(vb->element[1][j]));
				svfloat32_t b2 = svld1_f32(pg, &(vb->element[2][j]));
				svfloat32_t b3 = svld1_f32(pg, &(vb->element[3][j]));
				svfloat32_t t0, t1, t2, t3;
				_bncsve2_rqs_mul(pg, &t0, &t1, &t2, &t3, a0, a1, a2v, a3, b0, b1, b2, b3);
				_bncsve2_rqs_add(pg, &acc0, &acc1, &acc2, &acc3, acc0, acc1, acc2, acc3, t0, t1, t2, t3);
			}
			{
				long _L, _vl = (long)svcntw();
				float _la0[64], _la1[64], _la2[64], _la3[64];
				svst1_f32(svptrue_b32(), _la0, acc0);
				svst1_f32(svptrue_b32(), _la1, acc1);
				svst1_f32(svptrue_b32(), _la2, acc2);
				svst1_f32(svptrue_b32(), _la3, acc3);
				rqs_set_ui(tmp, 0UL);
				for(_L = 0; _L < _vl; _L++)
				{
					float _lane[QSSIZE] = { _la0[_L], _la1[_L], _la2[_L], _la3[_L] };
					rqs_add(tmp, tmp, _lane);
				}
			}
			v->element[0][i] = tmp[0];
			v->element[1][i] = tmp[1];
			v->element[2][i] = tmp[2];
			v->element[3][i] = tmp[3];
		}
	}

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	{
		long int ij_index, real_col_dim = a->real_col_dim;
		float32x4_t tmp4[QSSIZE], tmp1_4[QSSIZE], aij4[QSSIZE], vbj4[QSSIZE];
		for(i = 0; i < a->row_dim; i++)
		{
			_bncneon_set0_qs(tmp4);
			for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
			{
				ij_index = i * real_col_dim + j;
				aij4[0] = vld1q_f32(&(a->element[0][ij_index]));
				aij4[1] = vld1q_f32(&(a->element[1][ij_index]));
				aij4[2] = vld1q_f32(&(a->element[2][ij_index]));
				aij4[3] = vld1q_f32(&(a->element[3][ij_index]));
				vbj4[0] = vld1q_f32(&(vb->element[0][j]));
				vbj4[1] = vld1q_f32(&(vb->element[1][j]));
				vbj4[2] = vld1q_f32(&(vb->element[2][j]));
				vbj4[3] = vld1q_f32(&(vb->element[3][j]));
				_bncneon_rqs_mul(tmp1_4, aij4, vbj4);
				_bncneon_rqs_add(tmp4, tmp4, tmp1_4);
			}
			{
				float _l0[QSSIZE]={vgetq_lane_f32(tmp4[0],0),vgetq_lane_f32(tmp4[1],0),vgetq_lane_f32(tmp4[2],0),vgetq_lane_f32(tmp4[3],0)};
				float _l1[QSSIZE]={vgetq_lane_f32(tmp4[0],1),vgetq_lane_f32(tmp4[1],1),vgetq_lane_f32(tmp4[2],1),vgetq_lane_f32(tmp4[3],1)};
				float _l2[QSSIZE]={vgetq_lane_f32(tmp4[0],2),vgetq_lane_f32(tmp4[1],2),vgetq_lane_f32(tmp4[2],2),vgetq_lane_f32(tmp4[3],2)};
				float _l3[QSSIZE]={vgetq_lane_f32(tmp4[0],3),vgetq_lane_f32(tmp4[1],3),vgetq_lane_f32(tmp4[2],3),vgetq_lane_f32(tmp4[3],3)};
				rqs_set(tmp,_l0); rqs_add(tmp,tmp,_l1); rqs_add(tmp,tmp,_l2); rqs_add(tmp,tmp,_l3);
			}
			v->element[0][i] = tmp[0];
			v->element[1][i] = tmp[1];
			v->element[2][i] = tmp[2];
			v->element[3][i] = tmp[3];
		}
	}

#else // others

	for(i = 0; i < a->row_dim; i++)
	{
		rqs_set_ui(tmp, 0UL);
		for(j = 0; j < a->col_dim; j++)
		{
			rqs_mul(tmp1, get_qsmatrix_ij(a, i, j), get_qsvector_i(vb, j));
			rqs_add(tmp, tmp, tmp1);
		}
		set_qsvector_i(v, i, tmp);
	}
#endif // __AVX2__
}

/* v := a^T * vb */
void mul_qsmatrixt_qsvec(QSVector v, QSMatrix a, QSVector vb)
{
	long int i, j;
	float tmp[QSSIZE], tmp1[QSSIZE];

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_qsmatrixt_dsvec\n");
		return;
	}
// SIMD
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int ji_index, real_row_dim, real_col_dim;
	__m256 tmp4[QSSIZE], tmp1_4[QSSIZE];
	__m256 aij4[QSSIZE], vbj4[QSSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->col_dim; i++)
	{
		//rqs_set_ui(tmp, 0UL);
		_bncavx2_set0_qs(tmp4);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_row_dim; j += _BNC_S_WIDTH)
		{
			aij4[0] = _mm256_set_ps(
				a->element[0][(j + 7) * real_col_dim + i],
				a->element[0][(j + 6) * real_col_dim + i],
				a->element[0][(j + 5) * real_col_dim + i],
				a->element[0][(j + 4) * real_col_dim + i],
				a->element[0][(j + 3) * real_col_dim + i],
				a->element[0][(j + 2) * real_col_dim + i],
				a->element[0][(j + 1) * real_col_dim + i],
				a->element[0][(j    ) * real_col_dim + i]
			);
			aij4[1] = _mm256_set_ps(
				a->element[1][(j + 7) * real_col_dim + i],
				a->element[1][(j + 6) * real_col_dim + i],
				a->element[1][(j + 5) * real_col_dim + i],
				a->element[1][(j + 4) * real_col_dim + i],
				a->element[1][(j + 3) * real_col_dim + i],
				a->element[1][(j + 2) * real_col_dim + i],
				a->element[1][(j + 1) * real_col_dim + i],
				a->element[1][(j    ) * real_col_dim + i]
			);
			aij4[2] = _mm256_set_ps(
				a->element[2][(j + 7) * real_col_dim + i],
				a->element[2][(j + 6) * real_col_dim + i],
				a->element[2][(j + 5) * real_col_dim + i],
				a->element[2][(j + 4) * real_col_dim + i],
				a->element[2][(j + 3) * real_col_dim + i],
				a->element[2][(j + 2) * real_col_dim + i],
				a->element[2][(j + 1) * real_col_dim + i],
				a->element[2][(j    ) * real_col_dim + i]
			);
			aij4[3] = _mm256_set_ps(
				a->element[3][(j + 7) * real_col_dim + i],
				a->element[3][(j + 6) * real_col_dim + i],
				a->element[3][(j + 5) * real_col_dim + i],
				a->element[3][(j + 4) * real_col_dim + i],
				a->element[3][(j + 3) * real_col_dim + i],
				a->element[3][(j + 2) * real_col_dim + i],
				a->element[3][(j + 1) * real_col_dim + i],
				a->element[3][(j    ) * real_col_dim + i]
			);
			vbj4[0] = _mm256_load_ps(&(vb->element[0][j]));
			vbj4[1] = _mm256_load_ps(&(vb->element[1][j]));
			vbj4[2] = _mm256_load_ps(&(vb->element[2][j]));
			vbj4[3] = _mm256_load_ps(&(vb->element[3][j]));

			//rqs_mul(tmp1, get_qsmatrix_ij(a, i, j), get_qsvector_i(vb, j));
			//rqs_add(tmp, tmp, tmp1);
			_bncavx2_rqs_mul(tmp1_4, aij4, vbj4);
			_bncavx2_rqs_add(tmp4, tmp4, tmp1_4);
		}
		//set_tdvector_i(v, i, tmp);
		_bncavx2_rqs_sum256(tmp, tmp4);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
		v->element[2][i] = tmp[2];
		v->element[3][i] = tmp[3];
	}

#elif defined(__AVX512F__) // __AVX512F__
	long int ji_index, real_row_dim, real_col_dim;
	__m512 tmp4[QSSIZE], tmp1_4[QSSIZE];
	__m512 aij4[QSSIZE], vbj4[QSSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->col_dim; i++)
	{
		//rqs_set_ui(tmp, 0UL);
		_bncavx512_set0_qs(tmp4);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_row_dim; j += _BNC_S_WIDTH)
		{
			aij4[0] = _mm512_set_ps(
                    a->element[0][(j + 15) * real_col_dim + i],
                    a->element[0][(j + 14) * real_col_dim + i],
                    a->element[0][(j + 13) * real_col_dim + i],
                    a->element[0][(j + 12) * real_col_dim + i],
                    a->element[0][(j + 11) * real_col_dim + i],
                    a->element[0][(j + 10) * real_col_dim + i],
                    a->element[0][(j + 9) * real_col_dim + i],
                    a->element[0][(j + 8) * real_col_dim + i],
                    a->element[0][(j + 7) * real_col_dim + i],
                    a->element[0][(j + 6) * real_col_dim + i],
                    a->element[0][(j + 5) * real_col_dim + i],
                    a->element[0][(j + 4) * real_col_dim + i],
                    a->element[0][(j + 3) * real_col_dim + i],
                    a->element[0][(j + 2) * real_col_dim + i],
                    a->element[0][(j + 1) * real_col_dim + i],
                    a->element[0][(j + 0) * real_col_dim + i]
                );
			aij4[1] = _mm512_set_ps(
                    a->element[1][(j + 15) * real_col_dim + i],
                    a->element[1][(j + 14) * real_col_dim + i],
                    a->element[1][(j + 13) * real_col_dim + i],
                    a->element[1][(j + 12) * real_col_dim + i],
                    a->element[1][(j + 11) * real_col_dim + i],
                    a->element[1][(j + 10) * real_col_dim + i],
                    a->element[1][(j + 9) * real_col_dim + i],
                    a->element[1][(j + 8) * real_col_dim + i],
                    a->element[1][(j + 7) * real_col_dim + i],
                    a->element[1][(j + 6) * real_col_dim + i],
                    a->element[1][(j + 5) * real_col_dim + i],
                    a->element[1][(j + 4) * real_col_dim + i],
                    a->element[1][(j + 3) * real_col_dim + i],
                    a->element[1][(j + 2) * real_col_dim + i],
                    a->element[1][(j + 1) * real_col_dim + i],
                    a->element[1][(j + 0) * real_col_dim + i]
                );
			aij4[2] = _mm512_set_ps(
                    a->element[2][(j + 15) * real_col_dim + i],
                    a->element[2][(j + 14) * real_col_dim + i],
                    a->element[2][(j + 13) * real_col_dim + i],
                    a->element[2][(j + 12) * real_col_dim + i],
                    a->element[2][(j + 11) * real_col_dim + i],
                    a->element[2][(j + 10) * real_col_dim + i],
                    a->element[2][(j + 9) * real_col_dim + i],
                    a->element[2][(j + 8) * real_col_dim + i],
                    a->element[2][(j + 7) * real_col_dim + i],
                    a->element[2][(j + 6) * real_col_dim + i],
                    a->element[2][(j + 5) * real_col_dim + i],
                    a->element[2][(j + 4) * real_col_dim + i],
                    a->element[2][(j + 3) * real_col_dim + i],
                    a->element[2][(j + 2) * real_col_dim + i],
                    a->element[2][(j + 1) * real_col_dim + i],
                    a->element[2][(j + 0) * real_col_dim + i]
                );
			aij4[3] = _mm512_set_ps(
                    a->element[3][(j + 15) * real_col_dim + i],
                    a->element[3][(j + 14) * real_col_dim + i],
                    a->element[3][(j + 13) * real_col_dim + i],
                    a->element[3][(j + 12) * real_col_dim + i],
                    a->element[3][(j + 11) * real_col_dim + i],
                    a->element[3][(j + 10) * real_col_dim + i],
                    a->element[3][(j + 9) * real_col_dim + i],
                    a->element[3][(j + 8) * real_col_dim + i],
                    a->element[3][(j + 7) * real_col_dim + i],
                    a->element[3][(j + 6) * real_col_dim + i],
                    a->element[3][(j + 5) * real_col_dim + i],
                    a->element[3][(j + 4) * real_col_dim + i],
                    a->element[3][(j + 3) * real_col_dim + i],
                    a->element[3][(j + 2) * real_col_dim + i],
                    a->element[3][(j + 1) * real_col_dim + i],
                    a->element[3][(j + 0) * real_col_dim + i]
                );
			vbj4[0] = _mm512_load_ps(&(vb->element[0][j]));
			vbj4[1] = _mm512_load_ps(&(vb->element[1][j]));
			vbj4[2] = _mm512_load_ps(&(vb->element[2][j]));
			vbj4[3] = _mm512_load_ps(&(vb->element[3][j]));

			//rqs_mul(tmp1, get_qsmatrix_ij(a, i, j), get_qsvector_i(vb, j));
			//rqs_add(tmp, tmp, tmp1);
			_bncavx512_rqs_mul(tmp1_4, aij4, vbj4);
			_bncavx512_rqs_add(tmp4, tmp4, tmp1_4);
		}
		//set_tdvector_i(v, i, tmp);
		_bncavx512_rqs_sum512(tmp, tmp4);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
		v->element[2][i] = tmp[2];
		v->element[3][i] = tmp[3];
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic)
	{
		long real_col_dim = a->real_col_dim, real_row_dim = a->real_row_dim;
		long vl = (long)svcntw();
		for(i = 0; i < a->col_dim; i++)
		{
			svfloat32_t acc0, acc1, acc2, acc3;
			_bncsve2_rqs_set0(&acc0, &acc1, &acc2, &acc3);
			for(j = 0; j < real_row_dim; j += vl)
			{
				svbool_t pg = svwhilelt_b32_s32((int32_t)j, (int32_t)real_row_dim);
				svint32_t idx = svindex_s32((int32_t)(j * real_col_dim + i), (int32_t)real_col_dim);
				svfloat32_t a0 = svld1_gather_s32index_f32(pg, a->element[0], idx);
				svfloat32_t a1 = svld1_gather_s32index_f32(pg, a->element[1], idx);
				svfloat32_t a2v= svld1_gather_s32index_f32(pg, a->element[2], idx);
				svfloat32_t a3 = svld1_gather_s32index_f32(pg, a->element[3], idx);
				svfloat32_t b0 = svld1_f32(pg, &(vb->element[0][j]));
				svfloat32_t b1 = svld1_f32(pg, &(vb->element[1][j]));
				svfloat32_t b2 = svld1_f32(pg, &(vb->element[2][j]));
				svfloat32_t b3 = svld1_f32(pg, &(vb->element[3][j]));
				svfloat32_t t0, t1, t2, t3;
				_bncsve2_rqs_mul(pg, &t0, &t1, &t2, &t3, a0, a1, a2v, a3, b0, b1, b2, b3);
				_bncsve2_rqs_add(pg, &acc0, &acc1, &acc2, &acc3, acc0, acc1, acc2, acc3, t0, t1, t2, t3);
			}
			{
				long _L, _vl = (long)svcntw();
				float _la0[64], _la1[64], _la2[64], _la3[64];
				svst1_f32(svptrue_b32(), _la0, acc0);
				svst1_f32(svptrue_b32(), _la1, acc1);
				svst1_f32(svptrue_b32(), _la2, acc2);
				svst1_f32(svptrue_b32(), _la3, acc3);
				rqs_set_ui(tmp, 0UL);
				for(_L = 0; _L < _vl; _L++)
				{
					float _lane[QSSIZE] = { _la0[_L], _la1[_L], _la2[_L], _la3[_L] };
					rqs_add(tmp, tmp, _lane);
				}
			}
			v->element[0][i] = tmp[0];
			v->element[1][i] = tmp[1];
			v->element[2][i] = tmp[2];
			v->element[3][i] = tmp[3];
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	{
		long real_col_dim = a->real_col_dim, real_row_dim = a->real_row_dim;
		float32x4_t tmp4[QSSIZE], tmp1_4[QSSIZE], aij4[QSSIZE], vbj4[QSSIZE];
		for(i = 0; i < a->col_dim; i++)
		{
			_bncneon_set0_qs(tmp4);
			for(j = 0; j < real_row_dim; j += _BNC_S_WIDTH)
			{
				aij4[0] = (float32x4_t){ a->element[0][(j)*real_col_dim+i], a->element[0][(j+1)*real_col_dim+i], a->element[0][(j+2)*real_col_dim+i], a->element[0][(j+3)*real_col_dim+i] };
				aij4[1] = (float32x4_t){ a->element[1][(j)*real_col_dim+i], a->element[1][(j+1)*real_col_dim+i], a->element[1][(j+2)*real_col_dim+i], a->element[1][(j+3)*real_col_dim+i] };
				aij4[2] = (float32x4_t){ a->element[2][(j)*real_col_dim+i], a->element[2][(j+1)*real_col_dim+i], a->element[2][(j+2)*real_col_dim+i], a->element[2][(j+3)*real_col_dim+i] };
				aij4[3] = (float32x4_t){ a->element[3][(j)*real_col_dim+i], a->element[3][(j+1)*real_col_dim+i], a->element[3][(j+2)*real_col_dim+i], a->element[3][(j+3)*real_col_dim+i] };
				vbj4[0] = vld1q_f32(&(vb->element[0][j]));
				vbj4[1] = vld1q_f32(&(vb->element[1][j]));
				vbj4[2] = vld1q_f32(&(vb->element[2][j]));
				vbj4[3] = vld1q_f32(&(vb->element[3][j]));
				_bncneon_rqs_mul(tmp1_4, aij4, vbj4);
				_bncneon_rqs_add(tmp4, tmp4, tmp1_4);
			}
			{
				float _l0[QSSIZE]={vgetq_lane_f32(tmp4[0],0),vgetq_lane_f32(tmp4[1],0),vgetq_lane_f32(tmp4[2],0),vgetq_lane_f32(tmp4[3],0)};
				float _l1[QSSIZE]={vgetq_lane_f32(tmp4[0],1),vgetq_lane_f32(tmp4[1],1),vgetq_lane_f32(tmp4[2],1),vgetq_lane_f32(tmp4[3],1)};
				float _l2[QSSIZE]={vgetq_lane_f32(tmp4[0],2),vgetq_lane_f32(tmp4[1],2),vgetq_lane_f32(tmp4[2],2),vgetq_lane_f32(tmp4[3],2)};
				float _l3[QSSIZE]={vgetq_lane_f32(tmp4[0],3),vgetq_lane_f32(tmp4[1],3),vgetq_lane_f32(tmp4[2],3),vgetq_lane_f32(tmp4[3],3)};
				rqs_set(tmp,_l0); rqs_add(tmp,tmp,_l1); rqs_add(tmp,tmp,_l2); rqs_add(tmp,tmp,_l3);
			}
			v->element[0][i] = tmp[0];
			v->element[1][i] = tmp[1];
			v->element[2][i] = tmp[2];
			v->element[3][i] = tmp[3];
		}
	}
#else // others

	for(i = 0; i < a->col_dim; i++)
	{
		set0_qs(tmp);
		for(j = 0; j < a->row_dim; j++)
		{
			rqs_mul(tmp1, get_qsmatrix_ij(a, j, i), get_qsvector_i(vb, j));
			rqs_add(tmp, tmp, tmp1);
		}
		set_qsvector_i(v, i, tmp);
	}
#endif // __AVX2__
}

/* a = a^(-1) */
/* square matrix only */
void inv_qsmatrix(QSMatrix a)
{
	long int i, j, k, dim;
	float tmp[QSSIZE], aii[QSSIZE];

	/* Check Dimensions */
	if(a->row_dim != a->col_dim)
	{
		fprintf(stderr, "ERROR: inv_qsmatrix\n");
		return;
	}

	dim = a->row_dim;

	for(i = 0; i < dim; i++)
	{
		if(rqs_cmp_ui(get_qsmatrix_ij(a, i, i), 0UL) == 0) 
		{
			fprintf(stderr, "ERROR: inv_qsmatrix: Pivot(%ld,%ld) is zero.\n", i, i);
			return;
		}

		rqs_ui_div(aii, 1UL, get_qsmatrix_ij(a, i, i));
		set_qsmatrix_ij(a, i, i, aii);

		for(j = 0; j < i; j++)
		{
			rqs_mul(tmp, get_qsmatrix_ij(a, i, j), aii);
			set_qsmatrix_ij(a, i, j, tmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			rqs_mul(tmp, get_qsmatrix_ij(a, i, j), aii);
			set_qsmatrix_ij(a, i, j, tmp);
		}

		for(j = 0; j < i; j++)
		{
			for(k = 0; k < i; k++)
			{
				rqs_mul(tmp, get_qsmatrix_ij(a, j, i), get_qsmatrix_ij(a, i, k));
				rqs_sub(tmp, get_qsmatrix_ij(a, j, k), tmp);
				set_qsmatrix_ij(a, j, k, tmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				rqs_mul(tmp, get_qsmatrix_ij(a, j, i), get_qsmatrix_ij(a, i, k));
				rqs_sub(tmp, get_qsmatrix_ij(a, j, k), tmp);
				set_qsmatrix_ij(a, j, k, tmp);
			}
		}

		for(j = i + 1; j < dim; j++)
		{
			for(k = 0; k < i; k++)
			{
				rqs_mul(tmp, get_qsmatrix_ij(a, j, i), get_qsmatrix_ij(a, i, k));
				rqs_sub(tmp, get_qsmatrix_ij(a, j, k), tmp);
				set_qsmatrix_ij(a, j, k, tmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				rqs_mul(tmp, get_qsmatrix_ij(a, j, i), get_qsmatrix_ij(a, i, k));
				rqs_sub(tmp, get_qsmatrix_ij(a, j, k), tmp);
				set_qsmatrix_ij(a, j, k, tmp);
			}
		}

		for(j = 0; j < i; j++)
		{
			rqs_neg(tmp, aii); /* tmp := -aii */
			rqs_mul(tmp, tmp, get_qsmatrix_ij(a, j, i));
			set_qsmatrix_ij(a, j, i, tmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			rqs_neg(tmp, aii); /* tmp := -aii */
			rqs_mul(tmp, tmp, get_qsmatrix_ij(a, j, i));
			set_qsmatrix_ij(a, j, i, tmp);
		}
	}
}

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpf)a */
void subst_mpfvector_qsvec(MPFVector c, QSVector a)
{
	long int i;
	mpf_t tmp;

	mpf_init2(tmp, c->prec);

	for(i = 0; i < a->dim; i++)
	{
		mpf_set_qs(tmp, get_qsvector_i(a, i));
		set_mpfvector_i(c, i, tmp);
	}

	mpf_clear(tmp);

}

/* c := (dd)a */
void subst_qsvector_mpfvec(QSVector c, MPFVector a)
{
	long int i;
	float tmp[QSSIZE];

	for(i = 0; i < a->dim; i++)
	{
		mpf_get_qs(tmp, get_mpfvector_i(a, i));
		set_qsvector_i(c, i, tmp);
	}

}
/* c := (mpf)a */
void subst_mpfmatrix_qsmat(MPFMatrix c, QSMatrix a)
{
	long int i, j;
	mpf_t tmp;

	mpf_init2(tmp, c->prec);

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_mpfmatrix_dsmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			mpf_set_qs(tmp, get_qsmatrix_ij(a, i, j));
			set_mpfmatrix_ij(c, i, j, tmp);
		}
	}

	mpf_clear(tmp);
}

/* c := (dd)a */
void subst_qsmatrix_mpfmat(QSMatrix c, MPFMatrix a)
{
	long int i, j;
	float tmp[QSSIZE];

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_qsmatrix_mpfmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			mpf_get_qs(tmp, get_mpfmatrix_ij(a, i, j));
			set_qsmatrix_ij(c, i, j, tmp);
		}
	}
}

/* Normwise relative error of vector */
void relerr_qsvector_mpfvec(float relerr[QSSIZE], QSVector approx_vec, MPFVector true_vec, int norm_type)
{
	unsigned long prec;
	mpf_t mpf_relerr, norm_true_vec, norm_diff_vec;
	MPFVector diff_vec, mpf_approx_vec;

	prec = true_vec->prec;

	diff_vec = init2_mpfvector(approx_vec->dim, prec);
	mpf_approx_vec = init2_mpfvector(approx_vec->dim, prec);
	mpf_init2(mpf_relerr, prec);
	mpf_init2(norm_true_vec, prec);
	mpf_init2(norm_diff_vec, prec);

	// mpf_approx_vec := (mpf)approx_vec
	subst_mpfvector_qsvec(mpf_approx_vec, approx_vec);

	// diff_vec := approx_vec - true_vec
	sub_mpfvector(diff_vec, mpf_approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_mpfvector(norm_diff_vec, diff_vec);
			normi_mpfvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_mpfvector(norm_diff_vec, diff_vec);
			norm1_mpfvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_mpfvector(norm_diff_vec, diff_vec);
			norm2_mpfvector(norm_true_vec, true_vec);
			break;
	}

	// mpf_relerr := ||approx_vec - true_vec||
	mpf_set(mpf_relerr, norm_diff_vec);

	if(mpf_cmp_ui(norm_true_vec, 0UL) != 0)
		mpf_div(mpf_relerr, norm_diff_vec, norm_true_vec);

	// relerr := (DD)mpf_relerr
	//mpf_get_ds(relerr, mpf_relerr);
	mpf_get_ds(relerr, mpf_relerr);

	free_mpfvector(diff_vec);
	free_mpfvector(mpf_approx_vec);
	mpf_clear(norm_diff_vec);
	mpf_clear(norm_true_vec);
	mpf_clear(mpf_relerr);

	return;
}

/* Elementwise relative errors of vector */
void relerr_element_qsvector_mpf(float max_relerr[QSSIZE], float min_relerr[QSSIZE], float norm_relerr[QSSIZE], QSVector approx_vec, MPFVector true_vec, int norm_type)
{
	unsigned long prec;
	long int i;
	mpf_t abs_true_vec, abs_diff_vec, norm_diff_vec, norm_true_vec, mpf_max_relerr, mpf_min_relerr, mpf_norm_relerr;
	MPFVector mpf_approx_vec, diff_vec;

	prec = true_vec->prec;

	diff_vec = init2_mpfvector(approx_vec->dim, prec);
	mpf_approx_vec = init2_mpfvector(approx_vec->dim, prec);
	mpf_init2(abs_true_vec, prec);
	mpf_init2(abs_diff_vec, prec);
	mpf_init2(norm_diff_vec, prec);
	mpf_init2(norm_true_vec, prec);
	mpf_init2(mpf_max_relerr, prec);
	mpf_init2(mpf_min_relerr, prec);
	mpf_init2(mpf_norm_relerr, prec);

	// mpf_approx_vec := (mpf)approx_vec
	subst_mpfvector_qsvec(mpf_approx_vec, approx_vec);

	// diff_vec := approx_vec - true_vec
	sub_mpfvector(diff_vec, mpf_approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_mpfvector(norm_diff_vec, diff_vec);
			normi_mpfvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_mpfvector(norm_diff_vec, diff_vec);
			norm1_mpfvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_mpfvector(norm_diff_vec, diff_vec);
			norm2_mpfvector(norm_true_vec, true_vec);
			break;
	}

	mpf_set(mpf_norm_relerr, norm_diff_vec);
	if(mpf_cmp_ui(norm_true_vec, 0UL) != 0)
		mpf_div(mpf_norm_relerr, norm_diff_vec, norm_true_vec);

	// relative errors of each elements
	mpf_set_ui(mpf_max_relerr, 0UL);
	normi_mpfvector(mpf_min_relerr, diff_vec);
	for(i = 0; i < approx_vec->dim; i++)
	{
		mpf_abs(abs_diff_vec, get_mpfvector_i(diff_vec, i));
		mpf_abs(abs_true_vec, get_mpfvector_i(true_vec, i));
		if(mpf_cmp_ui(abs_true_vec, 0UL) != 0)
			mpf_div(abs_diff_vec, abs_diff_vec, abs_true_vec);
		
		if(mpf_cmp(mpf_max_relerr, abs_diff_vec) < 0)
			mpf_set(mpf_max_relerr, abs_diff_vec);
		if(mpf_cmp(mpf_min_relerr, abs_diff_vec) > 0)
			mpf_set(mpf_min_relerr, abs_diff_vec);
	}

	// relerr := (DD)mpf_relerr
	mpf_get_qs(max_relerr, mpf_max_relerr);
	mpf_get_qs(min_relerr, mpf_min_relerr);
	mpf_get_qs(norm_relerr, mpf_norm_relerr);

	free_mpfvector(diff_vec);// Fix! 2012-06-03 by T.Kouya
	free_mpfvector(mpf_approx_vec);
	mpf_clear(abs_true_vec);
	mpf_clear(abs_diff_vec);
	mpf_clear(norm_true_vec);
	mpf_clear(norm_diff_vec);
	mpf_clear(mpf_max_relerr);
	mpf_clear(mpf_min_relerr);
	mpf_clear(mpf_norm_relerr);

	return;
}
#endif // USE_GMP

#ifdef __BNC_DLINEAR_H__
/* c := (dd)a */
void subst_qsvector_dvec(QSVector c, DVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
	{
		set_qsvector_i_f(c, i, get_dvector_i(a, i));
	}
}

/* c := (d)a */
void subst_dvector_qsvec(DVector c, QSVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
	{
		c->element[i] = rqs_get_f(get_qsvector_i(a, i));
	}
}
#endif // __BNC_DLINEAR_H__

#ifdef __BNC_DLINEAR_H__
/* c := (dd)a */
void subst_qsmatrix_dmat(QSMatrix c, DMatrix a)
{
	long int i, j;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_qsmatrix_dmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_qsmatrix_ij_f(c, i, j, get_dmatrix_ij(a, i, j));
		}
	}
}
#endif // __BNC_DLINEAR_H__

#ifdef USE_TDLINEAR

/* c := a */
void subst_qsmatrix_tdmat(QSMatrix c, TDMatrix a)
{
	long int i, j, ij_index;
	float qd_c_ij[QSSIZE];

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_qsmatrix_tdmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			ij_index = i * (a->real_col_dim) + j;
//			rqs_sub(tmp, get_qsmatrix_ij(a, i, j), get_qsmatrix_ij(b, i, j));
			qd_c_ij[0] = a->element[0][ij_index];
			qd_c_ij[1] = a->element[1][ij_index];
			qd_c_ij[2] = a->element[2][ij_index];
			qd_c_ij[3] = 0.0;
			set_qsmatrix_ij(c, i, j, qd_c_ij);
		}
	}
}

/* (QSMatrix)c := (TDMatrix)a - (TDMatrix)b */
void sub_qsmatrix_tdmat_tdmat(QSMatrix c, TDMatrix a, TDMatrix b)
{
	long int i, j, row_dim, col_dim, ij_index;
	float tmp[QSSIZE], qd_a_ij[QSSIZE], qd_b_ij[QSSIZE];

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: sub_qsmatrix_tdmat_tdmat\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: sub_qsmatrix_tdmat_tdmat\n");
		return;
	}
	col_dim = c->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			ij_index = i * (a->real_col_dim) + j;
//			rqs_sub(tmp, get_qsmatrix_ij(a, i, j), get_qsmatrix_ij(b, i, j));
			qd_a_ij[0] = a->element[0][ij_index];
			qd_a_ij[1] = a->element[1][ij_index];
			qd_a_ij[2] = a->element[2][ij_index];
			qd_a_ij[3] = 0.0;
			qd_b_ij[0] = b->element[0][ij_index];
			qd_b_ij[1] = b->element[1][ij_index];
			qd_b_ij[2] = b->element[2][ij_index];
			qd_b_ij[3] = 0.0;
			rqs_sub(tmp, qd_a_ij, qd_b_ij);
			set_qsmatrix_ij(c, i, j, tmp);
		}
	}
}
#endif // USE_TDLINEAR

/* Normwise relative error of vector */
void relerr_qsvector(float relerr[QSSIZE], QSVector approx_vec, QSVector true_vec, int norm_type)
{
	float norm_true_vec[QSSIZE], norm_diff_vec[QSSIZE];
	QSVector diff_vec;

	diff_vec = init_qsvector(approx_vec->dim);

	// diff_vec := approx_vec - true_vec
	sub_qsvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_qsvector(norm_diff_vec, diff_vec);
			normi_qsvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_qsvector(norm_diff_vec, diff_vec);
			norm1_qsvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_qsvector(norm_diff_vec, diff_vec);
			norm2_qsvector(norm_true_vec, true_vec);
			break;
	}

	if(rqs_cmp_ui(norm_true_vec, 0UL) != 0)
		rqs_div(relerr, norm_diff_vec, norm_true_vec);

	free_qsvector(diff_vec);

	return;
}

/* Elementwise relative errors of vector */
void relerr_element_qsvector(float max_relerr[QSSIZE], float min_relerr[QSSIZE], float norm_relerr[QSSIZE], QSVector approx_vec, QSVector true_vec, int norm_type)
{
	float abs_true_vec[QSSIZE], abs_diff_vec[QSSIZE], norm_diff_vec[QSSIZE], norm_true_vec[QSSIZE];
	long int i;
	QSVector diff_vec;

	diff_vec = init_qsvector(approx_vec->dim);

	// diff_vec := approx_vec - true_vec
	sub_qsvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_qsvector(norm_diff_vec, diff_vec);
			normi_qsvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_qsvector(norm_diff_vec, diff_vec);
			norm1_qsvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_qsvector(norm_diff_vec, diff_vec);
			norm2_qsvector(norm_true_vec, true_vec);
			break;
	}

	rqs_set(norm_relerr, norm_diff_vec);
	if(rqs_cmp_ui(norm_true_vec, 0UL) != 0)
		rqs_div(norm_relerr, norm_diff_vec, norm_true_vec);

	// relative errors of each elements
	rqs_set_ui(max_relerr, 0UL);
	normi_qsvector(min_relerr, diff_vec);
	for(i = 0; i < approx_vec->dim; i++)
	{
		rqs_abs(abs_diff_vec, get_qsvector_i(diff_vec, i));
		rqs_abs(abs_true_vec, get_qsvector_i(true_vec, i));
		if(rqs_cmp_ui(abs_true_vec, 0UL) != 0)
			rqs_div(abs_diff_vec, abs_diff_vec, abs_true_vec);
		
		if(rqs_cmp(max_relerr, abs_diff_vec) < 0)
			rqs_set(max_relerr, abs_diff_vec);
		if(rqs_cmp(min_relerr, abs_diff_vec) > 0)
			rqs_set(min_relerr, abs_diff_vec);
	}

	free_qsvector(diff_vec);// Fix! 2012-06-03 by T.Kouya

	return;
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_qsmatrix(QSMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
	long int i, j, true_end;
	float tmp[QSSIZE];
	int thread_index;

	true_end = (col_end > mat->col_dim) ? mat->col_dim : col_end;

	for(i = col_start; i < true_end; i++)
	{
		rqs_set(tmp, get_qsmatrix_ij(mat, row_index0, i));
		set_qsmatrix_ij(mat, row_index0, i, get_qsmatrix_ij(mat, row_index1, i));
		set_qsmatrix_ij(mat, row_index1, i, tmp);
	}
}

// TD

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Octuple float Precision)       */
/*                                                          */
/*                 Ver. 0.0 2015-02-23 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int QSLUdecomp(QSMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       QSMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	static float dtmp[QSSIZE], dtmp1[QSSIZE], dmaxii[QSSIZE];
#ifdef BNC_USE_NEW_FMA
	static float neg_aji[QSSIZE];
#endif // BNC_USE_NEW_FMA

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
	{
		rqs_abs(dmaxii, get_qsmatrix_ij(a, i, i));
		if(rqs_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (QSLUdecomp)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rqs_div(dtmp, get_qsmatrix_ij(a, j, i), get_qsmatrix_ij(a, i, i));
			set_qsmatrix_ij(a, j, i, dtmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
#ifdef BNC_USE_NEW_FMA
			rqs_neg(neg_aji, get_qsmatrix_ij(a, j, i));
#endif // BNC_USE_NEW_FMA
			for(k = (i + 1); k < dim; k++)
			{
#ifdef BNC_USE_NEW_FMA
				rqs_fma(dtmp, neg_aji, get_qsmatrix_ij(a, i, k), get_qsmatrix_ij(a, j, k));
#else // BNC_USE_NEW_FMA
				rqs_mul(dtmp1, get_qsmatrix_ij(a, j, i), get_qsmatrix_ij(a, i, k));
				rqs_sub(dtmp, get_qsmatrix_ij(a, j, k), dtmp1);
#endif // BNC_USE_NEW_FMA
				set_qsmatrix_ij(a, j, k, dtmp);
			}
		}
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                         (Octuple float Precision)       */
/*                                                          */
/*                 Ver. 0.0 2011-11-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveQSLS(QSVector answer, QSMatrix lu, QSVector b)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      QSMatrix lu: LU decomposed Matrix (given by user)   */
/*      QSVector b: constant vector (given by user)         */
/*      QSVector answer: Solution for linear system         */
/*      long int dim: Dimension of Matrix (given by user)   */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static float dtmp[QSSIZE], dtmp1[QSSIZE];

	dim = answer->dim;

	subst_qsvector(answer, b);

#ifdef BNC_USE_NEW_FMA
/* fused branch-free FMA + SIMD forward/backward substitution */
	for(i = 0; i < dim; i++)
	{
		rqs_abs(dtmp, get_qsmatrix_ij(lu, i, i));
		if(rqs_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveQSLS, %ld)\n", i);
			return -1;
		}
	}

/* Forward (row-oriented) */
	for(i = 1; i < dim; i++)
	{
		_bnc_qssolve_dot(dtmp1, lu, i, 0, i, answer);
		rqs_sub(dtmp, get_qsvector_i(answer, i), dtmp1);
		set_qsvector_i(answer, i, dtmp);
	}

/* Backward (row-oriented) */
	for(i = (dim - 1); i >= 0; i--)
	{
		_bnc_qssolve_dot(dtmp1, lu, i, i + 1, dim, answer);
		rqs_sub(dtmp, get_qsvector_i(answer, i), dtmp1);
		rqs_div(dtmp, dtmp, get_qsmatrix_ij(lu, i, i));
		set_qsvector_i(answer, i, dtmp);
	}
#else // BNC_USE_NEW_FMA
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rqs_abs(dtmp, get_qsmatrix_ij(lu, i, i));
		if(rqs_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveQSLS, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rqs_mul(dtmp1, get_qsmatrix_ij(lu, j, i), get_qsvector_i(answer, i));
			rqs_sub(dtmp, get_qsvector_i(answer, j), dtmp1);
			set_qsvector_i(answer, j, dtmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rqs_mul(dtmp1, get_qsmatrix_ij(lu, i, j), get_qsvector_i(answer, j));
			rqs_sub(dtmp, get_qsvector_i(answer, i), dtmp1);
			set_qsvector_i(answer, i, dtmp);
		}
		rqs_div(dtmp, get_qsvector_i(answer, i), get_qsmatrix_ij(lu, i, i));
		set_qsvector_i(answer, i, dtmp);
	}

#endif // BNC_USE_NEW_FMA
	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Octuple float Precision)       */
/*                               (Partial Pivoting)         */
/*                                                          */
/*                 Ver. 0.0 2015-02-23 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int QSLUdecompP(QSMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       QSMatrix a: Matrix (given by user)                 */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
	static float dtmp[QSSIZE], dtmp1[QSSIZE], dmaxii[QSSIZE];
#ifdef BNC_USE_NEW_FMA
	static float neg_aji[QSSIZE];
#endif // BNC_USE_NEW_FMA

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		rqs_abs(dmaxii, get_qsmatrix_ij(a, ch[i], i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rqs_abs(dtmp, get_qsmatrix_ij(a, ch[j], i));
			if(rqs_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rqs_set(dmaxii, dtmp);
			}
		}

		if(rqs_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! QSLUdecompP!\n", i);
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
			rqs_div(dtmp, get_qsmatrix_ij(a, ch[j], i), get_qsmatrix_ij(a, ch[i], i));
			set_qsmatrix_ij(a, ch[j], i, dtmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
#ifdef BNC_USE_NEW_FMA
			rqs_neg(neg_aji, get_qsmatrix_ij(a, ch[j], i));
#endif // BNC_USE_NEW_FMA
			for(k = (i + 1); k < dim; k++)
			{
#ifdef BNC_USE_NEW_FMA
				rqs_fma(dtmp, neg_aji, get_qsmatrix_ij(a, ch[i], k), get_qsmatrix_ij(a, ch[j], k));
#else // BNC_USE_NEW_FMA
				rqs_mul(dtmp1, get_qsmatrix_ij(a, ch[j], i), get_qsmatrix_ij(a, ch[i], k));
				rqs_sub(dtmp, get_qsmatrix_ij(a, ch[j], k), dtmp1);
#endif // BNC_USE_NEW_FMA
				set_qsmatrix_ij(a, ch[j], k, dtmp);
			}
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                         (Octuple float Precision)       */
/*                               (Partial Pivoting)         */
/*                                                          */
/*                 Ver. 0.0 2011-11-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveQSLSP(QSVector answer, QSMatrix lu, QSVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      QSMatrix lu[]: LU decomposed Matrix (given by user) */
/*      QSVector b[]: constant vector (given by user)       */
/*      QSVector answer[]: Solution for linear system       */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static float dtmp[QSSIZE], dtmp1[QSSIZE];

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_qsvector_i(answer, i, get_qsvector_i(b, ch[i]));

	
#ifdef BNC_USE_NEW_FMA
/* fused branch-free FMA + SIMD forward/backward substitution */
	for(i = 0; i < dim; i++)
	{
		rqs_abs(dtmp, get_qsmatrix_ij(lu, ch[i], i));
		if(rqs_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveQSLSP, %ld)\n", i);
			return -1;
		}
	}

/* Forward (row-oriented) */
	for(i = 1; i < dim; i++)
	{
		_bnc_qssolve_dot(dtmp1, lu, ch[i], 0, i, answer);
		rqs_sub(dtmp, get_qsvector_i(answer, i), dtmp1);
		set_qsvector_i(answer, i, dtmp);
	}

/* Backward (row-oriented) */
	for(i = (dim - 1); i >= 0; i--)
	{
		_bnc_qssolve_dot(dtmp1, lu, ch[i], i + 1, dim, answer);
		rqs_sub(dtmp, get_qsvector_i(answer, i), dtmp1);
		rqs_div(dtmp, dtmp, get_qsmatrix_ij(lu, ch[i], i));
		set_qsvector_i(answer, i, dtmp);
	}
#else // BNC_USE_NEW_FMA
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rqs_abs(dtmp, get_qsmatrix_ij(lu, ch[i], i));
		if(rqs_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveQSLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rqs_mul(dtmp1, get_qsmatrix_ij(lu, ch[j], i), get_qsvector_i(answer, i));
			rqs_sub(dtmp, get_qsvector_i(answer, j), dtmp1);
			set_qsvector_i(answer, j, dtmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rqs_mul(dtmp1, get_qsmatrix_ij(lu, ch[i], j), get_qsvector_i(answer, j));
			rqs_sub(dtmp, get_qsvector_i(answer, i), dtmp1);
			set_qsvector_i(answer, i, dtmp);
		}
		rqs_div(dtmp, get_qsvector_i(answer, i), get_qsmatrix_ij(lu, ch[i], i));
		set_qsvector_i(answer, i, dtmp);
	}

#endif // BNC_USE_NEW_FMA
	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Octuple float Precision)       */
/*                               (Complete Pivoting)        */
/*                                                          */
/*                 Ver. 0.0 2015-02-23 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int QSLUdecompC(QSMatrix a, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       QSMatrix a[]: Matrix (given by user)               */
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
	static float dtmp[QSSIZE], dtmp1[QSSIZE], dmaxii[QSSIZE];
#ifdef BNC_USE_NEW_FMA
	static float neg_aji[QSSIZE];
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
		rqs_abs(dmaxii, get_qsmatrix_ij(a, row_ch[i], col_ch[i]));
		imax = i;
		jmax = i;
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rqs_abs(dtmp, get_qsmatrix_ij(a, row_ch[j], col_ch[k]));
				if(rqs_cmp(dtmp, dmaxii) > 0)
				{
					imax = j;
					jmax = k;
					rqs_set(dmaxii, dtmp);
				}
			}
		}

		if(rqs_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (QSLUdecompC)!\n", i);
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
			rqs_div(dtmp, get_qsmatrix_ij(a, row_ch[j], col_ch[i]), get_qsmatrix_ij(a, row_ch[i], col_ch[i]));
			set_qsmatrix_ij(a, row_ch[j], col_ch[i], dtmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
#ifdef BNC_USE_NEW_FMA
			rqs_neg(neg_aji, get_qsmatrix_ij(a, row_ch[j], col_ch[i]));
#endif // BNC_USE_NEW_FMA
			for(k = (i + 1); k < dim; k++)
			{
#ifdef BNC_USE_NEW_FMA
				rqs_fma(dtmp, neg_aji, get_qsmatrix_ij(a, row_ch[i], col_ch[k]), get_qsmatrix_ij(a, row_ch[j], col_ch[k]));
#else // BNC_USE_NEW_FMA
				rqs_mul(dtmp1, get_qsmatrix_ij(a, row_ch[j], col_ch[i]), get_qsmatrix_ij(a, row_ch[i], col_ch[k]));
				rqs_sub(dtmp, get_qsmatrix_ij(a, row_ch[j], col_ch[k]), dtmp1);
#endif // BNC_USE_NEW_FMA
				set_qsmatrix_ij(a, row_ch[j], col_ch[k], dtmp);
			}
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                         (Octuple float Precision)       */
/*                               (Complete Pivoting)        */
/*                                                          */
/*                 Ver. 0.0 2015-02-23 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveQSLSC(QSVector answer, QSMatrix lu, QSVector b, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       QSMatrix lu: LU decomposed Matrix (given by user)  */
/*       QSVector b: constant vector (given by user)        */
/*       QSVector answer: Solution for linear system        */
/*       long int row_ch[]: Row order (given by user)       */
/*       long int col_ch[]: Column order (given by user)    */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static float dtmp[QSSIZE], dtmp1[QSSIZE];

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_qsvector_i(answer, col_ch[i], get_qsvector_i(b, row_ch[i]));

#ifdef BNC_USE_NEW_FMA
/* fused branch-free FMA substitution (complete pivoting: scalar) */
	{
	static float ntmp[QSSIZE];

/* Forward */
	for(i = 0; i < dim; i++)
	{
		rqs_abs(dtmp, get_qsmatrix_ij(lu, row_ch[i], col_ch[i]));
		if(rqs_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveQSLSC, %ld)\n", i);
			return -1;
		}

		rqs_neg(ntmp, get_qsvector_i(answer, col_ch[i]));
		for(j = (i + 1); j < dim; j++)
		{
			rqs_fma(dtmp, get_qsmatrix_ij(lu, row_ch[j], col_ch[i]), ntmp, get_qsvector_i(answer, col_ch[j]));
			set_qsvector_i(answer, col_ch[j], dtmp);
		}
	}

/* Backward */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rqs_neg(ntmp, get_qsmatrix_ij(lu, row_ch[i], col_ch[j]));
			rqs_fma(dtmp, ntmp, get_qsvector_i(answer, col_ch[j]), get_qsvector_i(answer, col_ch[i]));
			set_qsvector_i(answer, col_ch[i], dtmp);
		}
		rqs_div(dtmp, get_qsvector_i(answer, col_ch[i]), get_qsmatrix_ij(lu, row_ch[i], col_ch[i]));
		set_qsvector_i(answer, col_ch[i], dtmp);
	}
	}
#else // BNC_USE_NEW_FMA
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rqs_abs(dtmp, get_qsmatrix_ij(lu, row_ch[i], col_ch[i]));
		if(rqs_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveQSLSC, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rqs_mul(dtmp1, get_qsmatrix_ij(lu, row_ch[j], col_ch[i]), get_qsvector_i(answer, col_ch[i]));
			rqs_sub(dtmp, get_qsvector_i(answer, col_ch[j]), dtmp1);
			set_qsvector_i(answer, col_ch[j],  dtmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rqs_mul(dtmp1, get_qsmatrix_ij(lu, row_ch[i], col_ch[j]), get_qsvector_i(answer, col_ch[j]));
			rqs_sub(dtmp, get_qsvector_i(answer, col_ch[i]), dtmp1);
			set_qsvector_i(answer, col_ch[i], dtmp);
		}
		rqs_div(dtmp, get_qsvector_i(answer, col_ch[i]), get_qsmatrix_ij(lu, row_ch[i], col_ch[i]));
		set_qsvector_i(answer, col_ch[i], dtmp);
	}

#endif // BNC_USE_NEW_FMA
	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                         (Octuple float Precision)       */
/*          (Partial Pivoting with real swap of rows)       */
/*                                                          */
/*                 Ver. 0.0 2021-02-17 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int QSLUdecompPM(QSMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       QSMatrix a: Matrix (given by user)                 */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
	static float dtmp[QSSIZE], dtmp1[QSSIZE], dmaxii[QSSIZE];
#ifdef BNC_USE_NEW_FMA
	float neg_aji[QSSIZE];
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m256 dtmp256[QSSIZE], aji256[QSSIZE], ajk256[QSSIZE], aik256[QSSIZE];
#elif defined(__AVX512F__) // __AVX512F__
	long int k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m512 dtmp512[QSSIZE], aji512[QSSIZE], ajk512[QSSIZE], aik512[QSSIZE];
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	long int k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	svfloat32_t dtmp_neon_0, dtmp_neon_1, dtmp_neon_2, dtmp_neon_3;
	svfloat32_t aji_neon_0, aji_neon_1, aji_neon_2, aji_neon_3;
	svfloat32_t ajk_neon_0, ajk_neon_1, ajk_neon_2, ajk_neon_3;
	svfloat32_t aik_neon_0, aik_neon_1, aik_neon_2, aik_neon_3;
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	long int k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	float32x4_t dtmp_neon[QSSIZE], aji_neon[QSSIZE], ajk_neon[QSSIZE], aik_neon[QSSIZE];
#else // others
#endif // __AVX2__
#endif // BNC_USE_NEW_FMA

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		// partial pivoting
		rqs_abs(dmaxii, get_qsmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rqs_abs(dtmp, get_qsmatrix_ij(a, j, i));
			if(rqs_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rqs_set(dmaxii, dtmp);
			}
		}

		if(rqs_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! QSLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			row_swap_qsmatrix(a, i, imax, 0, a->col_dim);
		}

		for(j = (i + 1); j < dim; j++)
		{
			rqs_div(dtmp, get_qsmatrix_ij(a, j, i), get_qsmatrix_ij(a, i, i));
			set_qsmatrix_ij(a, j, i, dtmp);
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
			neg_aji[2] = -(a->element[2][index_ji]);
			neg_aji[3] = -(a->element[3][index_ji]);
			aji256[0] = _mm256_set1_ps(neg_aji[0]);
			aji256[1] = _mm256_set1_ps(neg_aji[1]);
			aji256[2] = _mm256_set1_ps(neg_aji[2]);
			aji256[3] = _mm256_set1_ps(neg_aji[3]);

			// head
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rqs_fma(dtmp, neg_aji, get_qsmatrix_ij(a, i, k), get_qsmatrix_ij(a, j, k));
				set_qsmatrix_ij(a, j, k, dtmp);
			}

			// middle : SIMD
			for(k = dim_start; k < dim_end; k += _BNC_S_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				index_jk = j * a->real_col_dim + k;
				aik256[0] = _mm256_load_ps(&(a->element[0][index_ik]));
				aik256[1] = _mm256_load_ps(&(a->element[1][index_ik]));
				aik256[2] = _mm256_load_ps(&(a->element[2][index_ik]));
				aik256[3] = _mm256_load_ps(&(a->element[3][index_ik]));
				ajk256[0] = _mm256_load_ps(&(a->element[0][index_jk]));
				ajk256[1] = _mm256_load_ps(&(a->element[1][index_jk]));
				ajk256[2] = _mm256_load_ps(&(a->element[2][index_jk]));
				ajk256[3] = _mm256_load_ps(&(a->element[3][index_jk]));
				_bncavx2_qwfmaf(dtmp256, aji256, aik256, ajk256);
				_mm256_store_ps(&(a->element[0][index_jk]), dtmp256[0]);
				_mm256_store_ps(&(a->element[1][index_jk]), dtmp256[1]);
				_mm256_store_ps(&(a->element[2][index_jk]), dtmp256[2]);
				_mm256_store_ps(&(a->element[3][index_jk]), dtmp256[3]);
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
			neg_aji[2] = -(a->element[2][index_ji]);
			neg_aji[3] = -(a->element[3][index_ji]);
			aji512[0] = _mm512_set1_ps(neg_aji[0]);
			aji512[1] = _mm512_set1_ps(neg_aji[1]);
			aji512[2] = _mm512_set1_ps(neg_aji[2]);
			aji512[3] = _mm512_set1_ps(neg_aji[3]);

			// head
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rqs_fma(dtmp, neg_aji, get_qsmatrix_ij(a, i, k), get_qsmatrix_ij(a, j, k));
				set_qsmatrix_ij(a, j, k, dtmp);
			}

			// middle : SIMD
			for(k = dim_start; k < dim_end; k += _BNC_S_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				index_jk = j * a->real_col_dim + k;
				aik512[0] = _mm512_load_ps(&(a->element[0][index_ik]));
				aik512[1] = _mm512_load_ps(&(a->element[1][index_ik]));
				aik512[2] = _mm512_load_ps(&(a->element[2][index_ik]));
				aik512[3] = _mm512_load_ps(&(a->element[3][index_ik]));
				ajk512[0] = _mm512_load_ps(&(a->element[0][index_jk]));
				ajk512[1] = _mm512_load_ps(&(a->element[1][index_jk]));
				ajk512[2] = _mm512_load_ps(&(a->element[2][index_jk]));
				ajk512[3] = _mm512_load_ps(&(a->element[3][index_jk]));
				_bncavx512_qwfmaf(dtmp512, aji512, aik512, ajk512);
				_mm512_store_ps(&(a->element[0][index_jk]), dtmp512[0]);
				_mm512_store_ps(&(a->element[1][index_jk]), dtmp512[1]);
				_mm512_store_ps(&(a->element[2][index_jk]), dtmp512[2]);
				_mm512_store_ps(&(a->element[3][index_jk]), dtmp512[3]);
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
			neg_aji[2] = -(a->element[2][index_ji]);
			neg_aji[3] = -(a->element[3][index_ji]);
			aji_neon_0 = svdup_f32(neg_aji[0]);
			aji_neon_1 = svdup_f32(neg_aji[1]);
			aji_neon_2 = svdup_f32(neg_aji[2]);
			aji_neon_3 = svdup_f32(neg_aji[3]);

			// head
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rqs_fma(dtmp, neg_aji, get_qsmatrix_ij(a, i, k), get_qsmatrix_ij(a, j, k));
				set_qsmatrix_ij(a, j, k, dtmp);
			}

			// middle : SIMD (SVE2)
			for(k = dim_start; k < dim_end; k += (long int)svcntw())
			{
				svbool_t pg = svwhilelt_b32_s64((int64_t)k, (int64_t)(dim_end));
				index_ik = i * a->real_col_dim + k;
				index_jk = j * a->real_col_dim + k;
				aik_neon_0 = svld1_f32(pg, &(a->element[0][index_ik]));
				aik_neon_1 = svld1_f32(pg, &(a->element[1][index_ik]));
				aik_neon_2 = svld1_f32(pg, &(a->element[2][index_ik]));
				aik_neon_3 = svld1_f32(pg, &(a->element[3][index_ik]));
				ajk_neon_0 = svld1_f32(pg, &(a->element[0][index_jk]));
				ajk_neon_1 = svld1_f32(pg, &(a->element[1][index_jk]));
				ajk_neon_2 = svld1_f32(pg, &(a->element[2][index_jk]));
				ajk_neon_3 = svld1_f32(pg, &(a->element[3][index_jk]));
				_bncsve2_qwfmaf(pg, &dtmp_neon_0, &dtmp_neon_1, &dtmp_neon_2, &dtmp_neon_3, aji_neon_0, aji_neon_1, aji_neon_2, aji_neon_3, aik_neon_0, aik_neon_1, aik_neon_2, aik_neon_3, ajk_neon_0, ajk_neon_1, ajk_neon_2, ajk_neon_3);
				svst1_f32(pg, &(a->element[0][index_jk]), dtmp_neon_0);
				svst1_f32(pg, &(a->element[1][index_jk]), dtmp_neon_1);
				svst1_f32(pg, &(a->element[2][index_jk]), dtmp_neon_2);
				svst1_f32(pg, &(a->element[3][index_jk]), dtmp_neon_3);
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
			neg_aji[2] = -(a->element[2][index_ji]);
			neg_aji[3] = -(a->element[3][index_ji]);
			aji_neon[0] = vdupq_n_f32(neg_aji[0]);
			aji_neon[1] = vdupq_n_f32(neg_aji[1]);
			aji_neon[2] = vdupq_n_f32(neg_aji[2]);
			aji_neon[3] = vdupq_n_f32(neg_aji[3]);

			// head
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rqs_fma(dtmp, neg_aji, get_qsmatrix_ij(a, i, k), get_qsmatrix_ij(a, j, k));
				set_qsmatrix_ij(a, j, k, dtmp);
			}

			// middle : SIMD (Neon)
			for(k = dim_start; k < dim_end; k += _BNC_S_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				index_jk = j * a->real_col_dim + k;
				aik_neon[0] = vld1q_f32(&(a->element[0][index_ik]));
				aik_neon[1] = vld1q_f32(&(a->element[1][index_ik]));
				aik_neon[2] = vld1q_f32(&(a->element[2][index_ik]));
				aik_neon[3] = vld1q_f32(&(a->element[3][index_ik]));
				ajk_neon[0] = vld1q_f32(&(a->element[0][index_jk]));
				ajk_neon[1] = vld1q_f32(&(a->element[1][index_jk]));
				ajk_neon[2] = vld1q_f32(&(a->element[2][index_jk]));
				ajk_neon[3] = vld1q_f32(&(a->element[3][index_jk]));
				_bncneon_qwfmaf(dtmp_neon, aji_neon, aik_neon, ajk_neon);
				vst1q_f32(&(a->element[0][index_jk]), dtmp_neon[0]);
				vst1q_f32(&(a->element[1][index_jk]), dtmp_neon[1]);
				vst1q_f32(&(a->element[2][index_jk]), dtmp_neon[2]);
				vst1q_f32(&(a->element[3][index_jk]), dtmp_neon[3]);
			}
		}
#else // others
		for(j = (i + 1); j < dim; j++)
		{
			rqs_neg(neg_aji, get_qsmatrix_ij(a, j, i));
			for(k = (i + 1); k < dim; k++)
			{
				rqs_fma(dtmp, neg_aji, get_qsmatrix_ij(a, i, k), get_qsmatrix_ij(a, j, k));
				set_qsmatrix_ij(a, j, k, dtmp);
			}
		}
#endif // __AVX2__
#else // BNC_USE_NEW_FMA
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rqs_mul(dtmp1, get_qsmatrix_ij(a, j, i), get_qsmatrix_ij(a, i, k));
				rqs_sub(dtmp, get_qsmatrix_ij(a, j, k), dtmp1);
				set_qsmatrix_ij(a, j, k, dtmp);
			}
		}
#endif // BNC_USE_NEW_FMA
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                (LU Decomposed Square Dense Matrix)       */
/*                         (Octuple float Precision)       */
/*          (Partial Pivoting with real swap of rows)       */
/*                                                          */
/*                 Ver. 0.0 2021-02-17 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveQSLSPM(QSVector answer, QSMatrix lu, QSVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      QSMatrix lu[]: LU decomposed Matrix (given by user) */
/*      QSVector b[]: constant vector (given by user)       */
/*      QSVector answer[]: Solution for linear system       */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static float dtmp[QSSIZE], dtmp1[QSSIZE];

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_qsvector_i(answer, i, get_qsvector_i(b, ch[i]));

	
#ifdef BNC_USE_NEW_FMA
/* fused branch-free FMA + SIMD forward/backward substitution */
	for(i = 0; i < dim; i++)
	{
		rqs_abs(dtmp, get_qsmatrix_ij(lu, i, i));
		if(rqs_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveQSLSP, %ld)\n", i);
			return -1;
		}
	}

/* Forward (row-oriented) */
	for(i = 1; i < dim; i++)
	{
		_bnc_qssolve_dot(dtmp1, lu, i, 0, i, answer);
		rqs_sub(dtmp, get_qsvector_i(answer, i), dtmp1);
		set_qsvector_i(answer, i, dtmp);
	}

/* Backward (row-oriented) */
	for(i = (dim - 1); i >= 0; i--)
	{
		_bnc_qssolve_dot(dtmp1, lu, i, i + 1, dim, answer);
		rqs_sub(dtmp, get_qsvector_i(answer, i), dtmp1);
		rqs_div(dtmp, dtmp, get_qsmatrix_ij(lu, i, i));
		set_qsvector_i(answer, i, dtmp);
	}
#else // BNC_USE_NEW_FMA
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rqs_abs(dtmp, get_qsmatrix_ij(lu, i, i));
		if(rqs_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveQSLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rqs_mul(dtmp1, get_qsmatrix_ij(lu, j, i), get_qsvector_i(answer, i));
			rqs_sub(dtmp, get_qsvector_i(answer, j), dtmp1);
			set_qsvector_i(answer, j, dtmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rqs_mul(dtmp1, get_qsmatrix_ij(lu, i, j), get_qsvector_i(answer, j));
			rqs_sub(dtmp, get_qsvector_i(answer, i), dtmp1);
			set_qsvector_i(answer, i, dtmp);
		}
		rqs_div(dtmp, get_qsvector_i(answer, i), get_qsmatrix_ij(lu, i, i));
		set_qsvector_i(answer, i, dtmp);
	}

#endif // BNC_USE_NEW_FMA
	return 0;
}


#ifdef __cplusplus
} // extern "C"
#endif
