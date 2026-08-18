/********************************************************************************/
/* tslinear.c: Triple-single precision Linear Computation Library               */
/* Copyright (C) 2021 Tomonori Kouya                                            */
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
#include "tslinear.h"

#ifdef BNC_USE_NEW_FMA
/************************************************************/
/* sum := sum_{k=start}^{end-1} lu(row, k) * v(k)           */
/* (fused branch-free FMA dot product for LU solvers)       */
/************************************************************/
static void _bnc_tssolve_dot(float sum[TSSIZE], TSMatrix lu, long int row, long int start, long int end, TSVector v)
{
	long int k;
	float rtmp[TSSIZE], vtmp[TSSIZE];
	float *row_e[TSSIZE], *vec_e[TSSIZE];

	row_e[0] = &(lu->element[0][row * lu->real_col_dim]);
	row_e[1] = &(lu->element[1][row * lu->real_col_dim]);
	row_e[2] = &(lu->element[2][row * lu->real_col_dim]);
	vec_e[0] = &(v->element[0][0]);
	vec_e[1] = &(v->element[1][0]);
	vec_e[2] = &(v->element[2][0]);
	sum[0] = (float)0.0;
	sum[1] = (float)0.0;
	sum[2] = (float)0.0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	{
		__m256 vacc[TSSIZE], vr[TSSIZE], vv[TSSIZE];
		float red[TSSIZE][16];
		long int k_simd_end = start + ((end - start) / _BNC_S_WIDTH) * _BNC_S_WIDTH;
		long int w;

		vacc[0] = _mm256_setzero_ps();
		vacc[1] = _mm256_setzero_ps();
		vacc[2] = _mm256_setzero_ps();
		for(k = start; k < k_simd_end; k += _BNC_S_WIDTH)
		{
			vr[0] = _mm256_loadu_ps(&(row_e[0][k]));
			vr[1] = _mm256_loadu_ps(&(row_e[1][k]));
			vr[2] = _mm256_loadu_ps(&(row_e[2][k]));
			vv[0] = _mm256_loadu_ps(&(vec_e[0][k]));
			vv[1] = _mm256_loadu_ps(&(vec_e[1][k]));
			vv[2] = _mm256_loadu_ps(&(vec_e[2][k]));
			_bncavx2_twfmaf(vacc, vr, vv, vacc);
		}
		_mm256_storeu_ps(&(red[0][0]), vacc[0]);
		_mm256_storeu_ps(&(red[1][0]), vacc[1]);
		_mm256_storeu_ps(&(red[2][0]), vacc[2]);
		for(w = 0; w < _BNC_S_WIDTH; w++)
		{
			vtmp[0] = red[0][w];
			vtmp[1] = red[1][w];
			vtmp[2] = red[2][w];
			rts_add(sum, sum, vtmp);
		}
		for(k = k_simd_end; k < end; k++)
		{
			rtmp[0] = row_e[0][k];
			rtmp[1] = row_e[1][k];
			rtmp[2] = row_e[2][k];
			vtmp[0] = vec_e[0][k];
			vtmp[1] = vec_e[1][k];
			vtmp[2] = vec_e[2][k];
			rts_fma(sum, rtmp, vtmp, sum);
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	{
		__m512 vacc[TSSIZE], vr[TSSIZE], vv[TSSIZE];
		float red[TSSIZE][16];
		long int k_simd_end = start + ((end - start) / _BNC_S_WIDTH) * _BNC_S_WIDTH;
		long int w;

		vacc[0] = _mm512_setzero_ps();
		vacc[1] = _mm512_setzero_ps();
		vacc[2] = _mm512_setzero_ps();
		for(k = start; k < k_simd_end; k += _BNC_S_WIDTH)
		{
			vr[0] = _mm512_loadu_ps(&(row_e[0][k]));
			vr[1] = _mm512_loadu_ps(&(row_e[1][k]));
			vr[2] = _mm512_loadu_ps(&(row_e[2][k]));
			vv[0] = _mm512_loadu_ps(&(vec_e[0][k]));
			vv[1] = _mm512_loadu_ps(&(vec_e[1][k]));
			vv[2] = _mm512_loadu_ps(&(vec_e[2][k]));
			_bncavx512_twfmaf(vacc, vr, vv, vacc);
		}
		_mm512_storeu_ps(&(red[0][0]), vacc[0]);
		_mm512_storeu_ps(&(red[1][0]), vacc[1]);
		_mm512_storeu_ps(&(red[2][0]), vacc[2]);
		for(w = 0; w < _BNC_S_WIDTH; w++)
		{
			vtmp[0] = red[0][w];
			vtmp[1] = red[1][w];
			vtmp[2] = red[2][w];
			rts_add(sum, sum, vtmp);
		}
		for(k = k_simd_end; k < end; k++)
		{
			rtmp[0] = row_e[0][k];
			rtmp[1] = row_e[1][k];
			rtmp[2] = row_e[2][k];
			vtmp[0] = vec_e[0][k];
			vtmp[1] = vec_e[1][k];
			vtmp[2] = vec_e[2][k];
			rts_fma(sum, rtmp, vtmp, sum);
		}
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	{
		svbool_t pg = svptrue_b32();
		svfloat32_t acc0, acc1, acc2;
		svfloat32_t r0, r1, r2;
		svfloat32_t v0, v1, v2;
		float red[TSSIZE][64];
		long int vl = (long int)svcntw();
		long int k_simd_end = start + ((end - start) / vl) * vl;
		long int w;

		acc0 = svdup_f32(0.0);
		acc1 = svdup_f32(0.0);
		acc2 = svdup_f32(0.0);
		for(k = start; k < k_simd_end; k += vl)
		{
			r0 = svld1_f32(pg, &(row_e[0][k]));
			r1 = svld1_f32(pg, &(row_e[1][k]));
			r2 = svld1_f32(pg, &(row_e[2][k]));
			v0 = svld1_f32(pg, &(vec_e[0][k]));
			v1 = svld1_f32(pg, &(vec_e[1][k]));
			v2 = svld1_f32(pg, &(vec_e[2][k]));
			_bncsve2_twfmaf(pg, &acc0, &acc1, &acc2, r0, r1, r2, v0, v1, v2, acc0, acc1, acc2);
		}
		svst1_f32(pg, &(red[0][0]), acc0);
		svst1_f32(pg, &(red[1][0]), acc1);
		svst1_f32(pg, &(red[2][0]), acc2);
		for(w = 0; w < vl; w++)
		{
			vtmp[0] = red[0][w];
			vtmp[1] = red[1][w];
			vtmp[2] = red[2][w];
			rts_add(sum, sum, vtmp);
		}
		for(k = k_simd_end; k < end; k++)
		{
			rtmp[0] = row_e[0][k];
			rtmp[1] = row_e[1][k];
			rtmp[2] = row_e[2][k];
			vtmp[0] = vec_e[0][k];
			vtmp[1] = vec_e[1][k];
			vtmp[2] = vec_e[2][k];
			rts_fma(sum, rtmp, vtmp, sum);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	{
		float32x4_t vacc[TSSIZE], vr[TSSIZE], vv[TSSIZE];
		float red[TSSIZE][16];
		long int k_simd_end = start + ((end - start) / _BNC_S_WIDTH) * _BNC_S_WIDTH;
		long int w;

		vacc[0] = vdupq_n_f32(0.0f);
		vacc[1] = vdupq_n_f32(0.0f);
		vacc[2] = vdupq_n_f32(0.0f);
		for(k = start; k < k_simd_end; k += _BNC_S_WIDTH)
		{
			vr[0] = vld1q_f32(&(row_e[0][k]));
			vr[1] = vld1q_f32(&(row_e[1][k]));
			vr[2] = vld1q_f32(&(row_e[2][k]));
			vv[0] = vld1q_f32(&(vec_e[0][k]));
			vv[1] = vld1q_f32(&(vec_e[1][k]));
			vv[2] = vld1q_f32(&(vec_e[2][k]));
			_bncneon_twfmaf(vacc, vr, vv, vacc);
		}
		vst1q_f32(&(red[0][0]), vacc[0]);
		vst1q_f32(&(red[1][0]), vacc[1]);
		vst1q_f32(&(red[2][0]), vacc[2]);
		for(w = 0; w < _BNC_S_WIDTH; w++)
		{
			vtmp[0] = red[0][w];
			vtmp[1] = red[1][w];
			vtmp[2] = red[2][w];
			rts_add(sum, sum, vtmp);
		}
		for(k = k_simd_end; k < end; k++)
		{
			rtmp[0] = row_e[0][k];
			rtmp[1] = row_e[1][k];
			rtmp[2] = row_e[2][k];
			vtmp[0] = vec_e[0][k];
			vtmp[1] = vec_e[1][k];
			vtmp[2] = vec_e[2][k];
			rts_fma(sum, rtmp, vtmp, sum);
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
		rts_fma(sum, rtmp, vtmp, sum);
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
tsfloat tsnormf(tsfloat array[], int dim)
{
    int i;
    tsfloat ret, tmp;
    mpfr_t mpfr_ret;

    rts_set_ui(ret.val, 0UL);
    for(i = 0; i < dim; i++)
    {
        rts_mul(tmp.val, array[i].val, array[i].val);
        rts_add(ret.val, ret.val, tmp.val);
    }
    //printf("ret.val = "); rts_out_str(ret.val); printf("\n");
//  rts_sqrt(ret, ret);
    mpfr_init2(mpfr_ret, 128);
    mpfr_set_ts(mpfr_ret, ret.val, MPFR_RNDN);
    mpfr_sqrt(mpfr_ret, mpfr_ret, MPFR_RNDN);
    mpfr_get_ts(ret.val, mpfr_ret, MPFR_RNDN);
    mpfr_clear(mpfr_ret);
    return ret;
}

// generate a text vector: vec(i) := sqrt(sqrt_seed) * (i + 1)
/*void set_test_tsvector(tsfloat vec[], int sqrt_seed, int dim)
{
    int i, j;
    tsfloat ddsqrt;
    mpfr_t mpfrsqrt;

    // ddsqrt := sqrt(sqrt_seed)
    mpfr_init2(mpfrsqrt, 128);
    mpfr_sqrt_ui(mpfrsqrt, (unsigned long)sqrt_seed, MPFR_RNDN);
    mpfr_get_ts(ddsqrt.val, mpfrsqrt, MPFR_RNDN);
//    rts_set_ui(ddsqrt.val, sqrt_seed);
    //rts_sqrt(ddsqrt.val, ddsqrt.val);
    //rts_sqrt_ui(ddsqrt.val, sqrt_seed);

    //printf("set_test_tsmatrix: coef = "); rts_out_str(ddsqrt.val); printf("\n");

    for(i = 0; i < dim; i++)
    {
        //printf("%5d: ", i);
        rts_set_ui(vec[i].val, i + 1);
        rts_mul(vec[i].val, vec[i].val, ddsqrt.val);
    }
}*/

//#if 0
// tsrel_diff
tsfloat tsrel_diff(tsfloat a, tsfloat b)
{
    tsfloat rel_diff, abs_a;

    //rel_diff = fabs(a - b);
    rts_sub(rel_diff.val, a.val, b.val);
    rts_abs(rel_diff.val, rel_diff.val);

    //if(a != 0.0)
    if(rts_cmp_ui(a.val, 0UL) != 0)
    {
//        rel_diff /= fabs(a);
        rts_abs(abs_a.val, a.val);
        rts_div(rel_diff.val, rel_diff.val, a.val);
    }

    return rel_diff;
}
//#endif // 0

tsfloat tsrel_diff_array(tsfloat approx_a[], tsfloat approx_b[], int dim, int print_flag)
{
    int i;
    tsfloat rel_min, rel_max, rel_ave, rel_diff;
    mpfr_t mpfr_tmp;

    rel_diff = tsrel_diff(approx_a[0], approx_b[0]);
    rts_set(rel_min.val, rel_diff.val);
    rts_set(rel_max.val, rel_diff.val);
    rts_set(rel_ave.val, rel_diff.val);

    for(i = 1; i < dim; i++)
    {
        rel_diff = tsrel_diff(approx_a[i], approx_b[i]);
        if(rts_cmp(rel_diff.val, rel_min.val) < 0) rts_set(rel_min.val, rel_diff.val);
        if(rts_cmp(rel_diff.val, rel_max.val) > 0) rts_set(rel_max.val, rel_diff.val);
        //rel_ave += rel_diff;
        rts_add(rel_ave.val, rel_ave.val, rel_diff.val);
    }
    //rel_ave /= (cddfloat)dim;
    //rtd_div_ui(rel_ave.val, rel_ave.val, (unsigned long)dim);
    mpfr_init2(mpfr_tmp, 256);
    mpfr_set_ts(mpfr_tmp, rel_ave.val, MPFR_RNDN);
    mpfr_div_ui(mpfr_tmp, mpfr_tmp, (unsigned long)dim, MPFR_RNDN);
    mpfr_get_ts(rel_ave.val, mpfr_tmp, MPFR_RNDN);
    mpfr_clear(mpfr_tmp);

    if(print_flag == 1)
    {
        printf("max_rel_diff, min_rel_diff, ave_rel_diff:"); rts_out_str(rel_max.val); printf(" "); rts_out_str(rel_min.val);  printf(" "); rts_out_str(rel_ave.val); printf("\n"); 
    }

    return rel_max;
}
#endif // defined(USE_GMP) && defined(USE_MPFR)



// initialize TSVector
TSVector init_tsvector(long int dimension)
{
	TSVector ret = NULL;
	long int i, real_dim;

	if(dimension <= 0)
	{
		fprintf(stderr, "ERROR: init_tsvector\n");
		return ret;
	}

	ret = (TSVector)BNC_MALLOC(sizeof(tsvector));
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

	/* All 0 */
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 zero4;

	zero4 = _mm256_setzero_ps();
	for(i = 0; i < real_dim; i += _BNC_S_WIDTH)
	{
		_mm256_store_ps(&ret->element[0][i], zero4);
		_mm256_store_ps(&ret->element[1][i], zero4);
		_mm256_store_ps(&ret->element[2][i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero4;

	zero4 = _mm512_setzero_ps();
	for(i = 0; i < real_dim; i += _BNC_S_WIDTH)
	{
		_mm512_store_ps(&ret->element[0][i], zero4);
		_mm512_store_ps(&ret->element[1][i], zero4);
		_mm512_store_ps(&ret->element[2][i], zero4);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t zero4;

	zero4 = svdup_f32(0.0f);
	for(i = 0; i < real_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(real_dim));
		svst1_f32(pg, &ret->element[0][i], zero4);
		svst1_f32(pg, &ret->element[1][i], zero4);
		svst1_f32(pg, &ret->element[2][i], zero4);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t zero4;

	zero4 = vdupq_n_f32(0.0f);
	for(i = 0; i < real_dim; i += _BNC_S_WIDTH)
	{
		vst1q_f32(&ret->element[0][i], zero4);
		vst1q_f32(&ret->element[1][i], zero4);
		vst1q_f32(&ret->element[2][i], zero4);
	}
#else // others
	for(i = 0; i < dimension; i++)
	{
		ret->element[0][i] = 0.0f;
		ret->element[1][i] = 0.0f;
		ret->element[2][i] = 0.0f;
	}
#endif // __AVX2__

	ret->dim = dimension;
	ret->real_dim = real_dim;

	return ret;
}

// free TSVector
void free_tsvector(TSVector vec)
{
    long int i;
    for(i = 0; i < TSSIZE; i++)
        free(vec->element[i]);

    free(vec);
}

// TSVector vec -> tsfloat array
void set_tsfloat_tsvec(tsfloat ret[], int ret_dim, TSVector vec)
{
    int index, j, dim;

    dim = (ret_dim < vec->dim) ? ret_dim : vec->dim;

    for(index = 0; index < dim; index++)
    {
        for(j = 0; j < TSSIZE; j++)
            ret[index].val[j] = vec->element[j][index];
    }
}

// tsfloat array -> TSVector ret
void set_tsvector_tsfloat(TSVector ret, tsfloat array[], int array_dim)
{
    int index, j, dim;

    dim = (ret->dim < array_dim) ? ret->dim : array_dim;

    for(index = 0; index < dim; index++)
    {
        for(j = 0; j < TSSIZE; j++)
            ret->element[j][index] = array[index].val[j];
    }
}

// print tsvector
void print_tsvector(TSVector vec)
{
	long int index;

	for(index = 0; index < vec->dim; index++)
	{
		printf("%4ld: ", index);
		//c_dd_write((vec->element + index * TSSIZE));
		c_ts_write(GET_TSVECTOR_I(vec, index));
	}
}

// set a zero vector
void set0_tsvector(TSVector vec)
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
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero4;

	zero4 = _mm512_setzero_ps();
	for(i = 0; i < vec->real_dim; i += _BNC_S_WIDTH)
	{
		_mm512_store_ps(&vec->element[0][i], zero4);
		_mm512_store_ps(&vec->element[1][i], zero4);
		_mm512_store_ps(&vec->element[2][i], zero4);
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
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t zero4;

	zero4 = vdupq_n_f32(0.0f);
	for(i = 0; i < vec->real_dim; i += _BNC_S_WIDTH)
	{
		vst1q_f32(&vec->element[0][i], zero4);
		vst1q_f32(&vec->element[1][i], zero4);
		vst1q_f32(&vec->element[2][i], zero4);
	}
#else // others
	for(i = 0; i < vec->dim; i++)
	{
		vec->element[0][i] = 0.0f;
		vec->element[1][i] = 0.0f;
		vec->element[2][i] = 0.0f;
	}
#endif // __AVX2__
}

// set_tsvector_i_str
void set_tsvector_i_str(TSVector vec, long int index, const char *str)
{
	float tmp[TSSIZE];

	//rts_get_str(tmp, str);
	rts_set_str(tmp, str);

	set_tsvector_i(vec, index, tmp);
}

/*************************************************/
/* Vector Calculations for TSVector               */
/*
void add_tsvector(TSVector c, TSVector a, TSVector b)
void add2_tsvector(TSVector c, TSVector a)
void sub_tsvector(TSVector c, TSVector a, TSVector b)
void sub2_tsvector(TSVector c, FVector a)
void cmul_tsvector(TSVector c, float val[TSSIZE], TSVector a)
void cmul2_tsvector(TSVector c, float val[TSSIZE])
void add_cmul_tsvector(TSVector c, TSVector a, float val[TSSIZE], TSVector b)
float ip_tsvector(TSVector a, TSVector b)
float norm1_tsvector(TSVector a)
float norm2_tsvector(TSVector a)
float normi_tsvector(TSVector a)
void subst_tsvector(TSVector c, TSVector a)
*/
/*************************************************/
/* c = a + b */
void add_tsvector(TSVector c, TSVector a, TSVector b)
{
    long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_tsvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256 in_ret[TSSIZE], in_a_val[TSSIZE], in_b_val[TSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_ps(&(a->element[1][index]));
        in_a_val[2] = _mm256_load_ps(&(a->element[2][index]));
        in_b_val[0] = _mm256_load_ps(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_ps(&(b->element[1][index]));
        in_b_val[2] = _mm256_load_ps(&(b->element[2][index]));

        _bncavx2_rts_add(in_ret, in_a_val, in_b_val);

        _mm256_store_ps(&(c->element[0][index]), in_ret[0]);
        _mm256_store_ps(&(c->element[1][index]), in_ret[1]);
        _mm256_store_ps(&(c->element[2][index]), in_ret[2]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512 in_ret[TSSIZE], in_a_val[TSSIZE], in_b_val[TSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm512_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_ps(&(a->element[1][index]));
        in_a_val[2] = _mm512_load_ps(&(a->element[2][index]));
        in_b_val[0] = _mm512_load_ps(&(b->element[0][index]));
        in_b_val[1] = _mm512_load_ps(&(b->element[1][index]));
        in_b_val[2] = _mm512_load_ps(&(b->element[2][index]));

        _bncavx512_rts_add(in_ret, in_a_val, in_b_val);

        _mm512_store_ps(&(c->element[0][index]), in_ret[0]);
        _mm512_store_ps(&(c->element[1][index]), in_ret[1]);
        _mm512_store_ps(&(c->element[2][index]), in_ret[2]);
   }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
    
	svfloat32_t in_ret_0, in_ret_1, in_ret_2;
	svfloat32_t in_a_val_0, in_a_val_1, in_a_val_2;
	svfloat32_t in_b_val_0, in_b_val_1, in_b_val_2;

    for(index = 0; index < c->real_dim; index += (long int)svcntw())
    {
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(c->real_dim));
        in_a_val_0 = svld1_f32(pg, &(a->element[0][index]));
        in_a_val_1 = svld1_f32(pg, &(a->element[1][index]));
        in_a_val_2 = svld1_f32(pg, &(a->element[2][index]));
        in_b_val_0 = svld1_f32(pg, &(b->element[0][index]));
        in_b_val_1 = svld1_f32(pg, &(b->element[1][index]));
        in_b_val_2 = svld1_f32(pg, &(b->element[2][index]));

        _bncsve2_rts_add(pg, &in_ret_0, &in_ret_1, &in_ret_2, in_a_val_0, in_a_val_1, in_a_val_2, in_b_val_0, in_b_val_1, in_b_val_2);

        svst1_f32(pg, &(c->element[0][index]), in_ret_0);
        svst1_f32(pg, &(c->element[1][index]), in_ret_1);
        svst1_f32(pg, &(c->element[2][index]), in_ret_2);
   }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float32x4_t in_ret[TSSIZE], in_a_val[TSSIZE], in_b_val[TSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = vld1q_f32(&(a->element[0][index]));
        in_a_val[1] = vld1q_f32(&(a->element[1][index]));
        in_a_val[2] = vld1q_f32(&(a->element[2][index]));
        in_b_val[0] = vld1q_f32(&(b->element[0][index]));
        in_b_val[1] = vld1q_f32(&(b->element[1][index]));
        in_b_val[2] = vld1q_f32(&(b->element[2][index]));

        _bncneon_rts_add(in_ret, in_a_val, in_b_val);

        vst1q_f32(&(c->element[0][index]), in_ret[0]);
        vst1q_f32(&(c->element[1][index]), in_ret[1]);
        vst1q_f32(&(c->element[2][index]), in_ret[2]);
   }
#else // others
	float tmp[TSSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rts_add(tmp, get_tsvector_i(a, i),  get_tsvector_i(b, i));
		set_tsvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c += a */
void add2_tsvector(TSVector c, TSVector a)
{
	long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: add2_tsvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256 in_ret[TSSIZE], in_a_val[TSSIZE], tmp4[TSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_ps(&(a->element[1][index]));
        in_a_val[2] = _mm256_load_ps(&(a->element[2][index]));
        in_ret[0] = _mm256_load_ps(&(c->element[0][index]));
        in_ret[1] = _mm256_load_ps(&(c->element[1][index]));
        in_ret[2] = _mm256_load_ps(&(c->element[2][index]));

        _bncavx2_rts_add(tmp4, in_ret, in_a_val);

        _mm256_store_ps(&(c->element[0][index]), tmp4[0]);
        _mm256_store_ps(&(c->element[1][index]), tmp4[1]);
        _mm256_store_ps(&(c->element[2][index]), tmp4[2]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512 in_ret[TSSIZE], in_a_val[TSSIZE], tmp4[TSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm512_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_ps(&(a->element[1][index]));
        in_a_val[2] = _mm512_load_ps(&(a->element[2][index]));
        in_ret[0] = _mm512_load_ps(&(c->element[0][index]));
        in_ret[1] = _mm512_load_ps(&(c->element[1][index]));
        in_ret[2] = _mm512_load_ps(&(c->element[2][index]));

        _bncavx512_rts_add(tmp4, in_ret, in_a_val);

        _mm512_store_ps(&(c->element[0][index]), tmp4[0]);
        _mm512_store_ps(&(c->element[1][index]), tmp4[1]);
        _mm512_store_ps(&(c->element[2][index]), tmp4[2]);
   }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
    
	svfloat32_t in_ret_0, in_ret_1, in_ret_2;
	svfloat32_t in_a_val_0, in_a_val_1, in_a_val_2;
	svfloat32_t tmp4_0, tmp4_1, tmp4_2;

    for(index = 0; index < c->real_dim; index += (long int)svcntw())
    {
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(c->real_dim));
        in_a_val_0 = svld1_f32(pg, &(a->element[0][index]));
        in_a_val_1 = svld1_f32(pg, &(a->element[1][index]));
        in_a_val_2 = svld1_f32(pg, &(a->element[2][index]));
        in_ret_0 = svld1_f32(pg, &(c->element[0][index]));
        in_ret_1 = svld1_f32(pg, &(c->element[1][index]));
        in_ret_2 = svld1_f32(pg, &(c->element[2][index]));

        _bncsve2_rts_add(pg, &tmp4_0, &tmp4_1, &tmp4_2, in_ret_0, in_ret_1, in_ret_2, in_a_val_0, in_a_val_1, in_a_val_2);

        svst1_f32(pg, &(c->element[0][index]), tmp4_0);
        svst1_f32(pg, &(c->element[1][index]), tmp4_1);
        svst1_f32(pg, &(c->element[2][index]), tmp4_2);
   }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float32x4_t in_ret[TSSIZE], in_a_val[TSSIZE], tmp4[TSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = vld1q_f32(&(a->element[0][index]));
        in_a_val[1] = vld1q_f32(&(a->element[1][index]));
        in_a_val[2] = vld1q_f32(&(a->element[2][index]));
        in_ret[0] = vld1q_f32(&(c->element[0][index]));
        in_ret[1] = vld1q_f32(&(c->element[1][index]));
        in_ret[2] = vld1q_f32(&(c->element[2][index]));

        _bncneon_rts_add(tmp4, in_ret, in_a_val);

        vst1q_f32(&(c->element[0][index]), tmp4[0]);
        vst1q_f32(&(c->element[1][index]), tmp4[1]);
        vst1q_f32(&(c->element[2][index]), tmp4[2]);
   }
#else // others
	float tmp[TSSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rts_add(tmp, get_tsvector_i(c, i), get_tsvector_i(a, i));
		set_tsvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c = a - b */
void sub_tsvector(TSVector c, TSVector a, TSVector b)
{
    long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_tsvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256 in_ret[TSSIZE], in_a_val[TSSIZE], in_b_val[TSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_ps(&(a->element[1][index]));
        in_a_val[2] = _mm256_load_ps(&(a->element[2][index]));
        in_b_val[0] = _mm256_load_ps(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_ps(&(b->element[1][index]));
        in_b_val[2] = _mm256_load_ps(&(b->element[2][index]));

        _bncavx2_rts_sub(in_ret, in_a_val, in_b_val);

        _mm256_store_ps(&(c->element[0][index]), in_ret[0]);
        _mm256_store_ps(&(c->element[1][index]), in_ret[1]);
        _mm256_store_ps(&(c->element[2][index]), in_ret[2]);
  }
#elif defined(__AVX512F__) // __AVX512F__
    __m512 in_ret[TSSIZE], in_a_val[TSSIZE], in_b_val[TSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm512_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_ps(&(a->element[1][index]));
        in_a_val[2] = _mm512_load_ps(&(a->element[2][index]));
        in_b_val[0] = _mm512_load_ps(&(b->element[0][index]));
        in_b_val[1] = _mm512_load_ps(&(b->element[1][index]));
        in_b_val[2] = _mm512_load_ps(&(b->element[2][index]));

        _bncavx512_rts_sub(in_ret, in_a_val, in_b_val);

        _mm512_store_ps(&(c->element[0][index]), in_ret[0]);
        _mm512_store_ps(&(c->element[1][index]), in_ret[1]);
        _mm512_store_ps(&(c->element[2][index]), in_ret[2]);
  }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
    
	svfloat32_t in_ret_0, in_ret_1, in_ret_2;
	svfloat32_t in_a_val_0, in_a_val_1, in_a_val_2;
	svfloat32_t in_b_val_0, in_b_val_1, in_b_val_2;

    for(index = 0; index < c->real_dim; index += (long int)svcntw())
    {
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(c->real_dim));
        in_a_val_0 = svld1_f32(pg, &(a->element[0][index]));
        in_a_val_1 = svld1_f32(pg, &(a->element[1][index]));
        in_a_val_2 = svld1_f32(pg, &(a->element[2][index]));
        in_b_val_0 = svld1_f32(pg, &(b->element[0][index]));
        in_b_val_1 = svld1_f32(pg, &(b->element[1][index]));
        in_b_val_2 = svld1_f32(pg, &(b->element[2][index]));

        _bncsve2_rts_neg(pg, &in_ret_0, &in_ret_1, &in_ret_2, in_b_val_0, in_b_val_1, in_b_val_2);
		_bncsve2_rts_add(pg, &in_ret_0, &in_ret_1, &in_ret_2, in_a_val_0, in_a_val_1, in_a_val_2, in_ret_0, in_ret_1, in_ret_2);

        svst1_f32(pg, &(c->element[0][index]), in_ret_0);
        svst1_f32(pg, &(c->element[1][index]), in_ret_1);
        svst1_f32(pg, &(c->element[2][index]), in_ret_2);
  }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float32x4_t in_ret[TSSIZE], in_a_val[TSSIZE], in_b_val[TSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = vld1q_f32(&(a->element[0][index]));
        in_a_val[1] = vld1q_f32(&(a->element[1][index]));
        in_a_val[2] = vld1q_f32(&(a->element[2][index]));
        in_b_val[0] = vld1q_f32(&(b->element[0][index]));
        in_b_val[1] = vld1q_f32(&(b->element[1][index]));
        in_b_val[2] = vld1q_f32(&(b->element[2][index]));

        _bncneon_rts_sub(in_ret, in_a_val, in_b_val);

        vst1q_f32(&(c->element[0][index]), in_ret[0]);
        vst1q_f32(&(c->element[1][index]), in_ret[1]);
        vst1q_f32(&(c->element[2][index]), in_ret[2]);
  }
#else // others
	float tmp[TSSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rts_sub(tmp, get_tsvector_i(a, i),  get_tsvector_i(b, i));
		set_tsvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c -= a */
void sub2_tsvector(TSVector c, TSVector a)
{
	long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: add2_tsvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256 in_ret[TSSIZE], in_a_val[TSSIZE], tmp4[TSSIZE];

	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_ps(&(a->element[1][index]));
        in_a_val[2] = _mm256_load_ps(&(a->element[2][index]));
        in_ret[0] = _mm256_load_ps(&(c->element[0][index]));
        in_ret[1] = _mm256_load_ps(&(c->element[1][index]));
        in_ret[2] = _mm256_load_ps(&(c->element[2][index]));

        _bncavx2_rts_sub(tmp4, in_ret, in_a_val);

        _mm256_store_ps(&(c->element[0][index]), tmp4[0]);
        _mm256_store_ps(&(c->element[1][index]), tmp4[1]);
        _mm256_store_ps(&(c->element[2][index]), tmp4[2]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512 in_ret[TSSIZE], in_a_val[TSSIZE], tmp4[TSSIZE];

	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm512_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_ps(&(a->element[1][index]));
        in_a_val[2] = _mm512_load_ps(&(a->element[2][index]));
        in_ret[0] = _mm512_load_ps(&(c->element[0][index]));
        in_ret[1] = _mm512_load_ps(&(c->element[1][index]));
        in_ret[2] = _mm512_load_ps(&(c->element[2][index]));

        _bncavx512_rts_sub(tmp4, in_ret, in_a_val);

        _mm512_store_ps(&(c->element[0][index]), tmp4[0]);
        _mm512_store_ps(&(c->element[1][index]), tmp4[1]);
        _mm512_store_ps(&(c->element[2][index]), tmp4[2]);
   }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
    
	svfloat32_t in_ret_0, in_ret_1, in_ret_2;
	svfloat32_t in_a_val_0, in_a_val_1, in_a_val_2;
	svfloat32_t tmp4_0, tmp4_1, tmp4_2;

	for(index = 0; index < c->real_dim; index += (long int)svcntw())
    {
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(c->real_dim));
        in_a_val_0 = svld1_f32(pg, &(a->element[0][index]));
        in_a_val_1 = svld1_f32(pg, &(a->element[1][index]));
        in_a_val_2 = svld1_f32(pg, &(a->element[2][index]));
        in_ret_0 = svld1_f32(pg, &(c->element[0][index]));
        in_ret_1 = svld1_f32(pg, &(c->element[1][index]));
        in_ret_2 = svld1_f32(pg, &(c->element[2][index]));

        _bncsve2_rts_neg(pg, &tmp4_0, &tmp4_1, &tmp4_2, in_a_val_0, in_a_val_1, in_a_val_2);
		_bncsve2_rts_add(pg, &tmp4_0, &tmp4_1, &tmp4_2, in_ret_0, in_ret_1, in_ret_2, tmp4_0, tmp4_1, tmp4_2);

        svst1_f32(pg, &(c->element[0][index]), tmp4_0);
        svst1_f32(pg, &(c->element[1][index]), tmp4_1);
        svst1_f32(pg, &(c->element[2][index]), tmp4_2);
   }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float32x4_t in_ret[TSSIZE], in_a_val[TSSIZE], tmp4[TSSIZE];

	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = vld1q_f32(&(a->element[0][index]));
        in_a_val[1] = vld1q_f32(&(a->element[1][index]));
        in_a_val[2] = vld1q_f32(&(a->element[2][index]));
        in_ret[0] = vld1q_f32(&(c->element[0][index]));
        in_ret[1] = vld1q_f32(&(c->element[1][index]));
        in_ret[2] = vld1q_f32(&(c->element[2][index]));

        _bncneon_rts_sub(tmp4, in_ret, in_a_val);

        vst1q_f32(&(c->element[0][index]), tmp4[0]);
        vst1q_f32(&(c->element[1][index]), tmp4[1]);
        vst1q_f32(&(c->element[2][index]), tmp4[2]);
   }
#else // others
	float tmp[TSSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rts_sub(tmp, get_tsvector_i(c, i), get_tsvector_i(a, i));
		set_tsvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c = val * a */
void cmul_tsvector(TSVector c, float val[TSSIZE], TSVector a)
{
    long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: cmul_fvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4[TSSIZE], c4[TSSIZE], val4[TSSIZE];

	val4[0] = _mm256_set1_ps(val[0]);
	val4[1] = _mm256_set1_ps(val[1]);
	val4[2] = _mm256_set1_ps(val[2]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		//set_tsvector_i(c, i, val * get_tsvector_i(a, i));
		a4[0] = _mm256_load_ps(&(a->element[0][index]));
		a4[1] = _mm256_load_ps(&(a->element[1][index]));
		a4[2] = _mm256_load_ps(&(a->element[2][index]));

		_bncavx2_rts_mul(c4, val4, a4);

		_mm256_store_ps(&(c->element[0][index]), c4[0]);
		_mm256_store_ps(&(c->element[1][index]), c4[1]);
		_mm256_store_ps(&(c->element[2][index]), c4[2]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a4[TSSIZE], c4[TSSIZE], val4[TSSIZE];

	val4[0] = _mm512_set1_ps(val[0]);
	val4[1] = _mm512_set1_ps(val[1]);
	val4[2] = _mm512_set1_ps(val[2]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		//set_tsvector_i(c, i, val * get_tsvector_i(a, i));
		a4[0] = _mm512_load_ps(&(a->element[0][index]));
		a4[1] = _mm512_load_ps(&(a->element[1][index]));
		a4[2] = _mm512_load_ps(&(a->element[2][index]));

		_bncavx512_rts_mul(c4, val4, a4);

		_mm512_store_ps(&(c->element[0][index]), c4[0]);
		_mm512_store_ps(&(c->element[1][index]), c4[1]);
		_mm512_store_ps(&(c->element[2][index]), c4[2]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t a4_0, a4_1, a4_2;
	svfloat32_t c4_0, c4_1, c4_2;
	svfloat32_t val4_0, val4_1, val4_2;

	val4_0 = svdup_f32(val[0]);
	val4_1 = svdup_f32(val[1]);
	val4_2 = svdup_f32(val[2]);
	for(index = 0; index < c->real_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(c->real_dim));
		//set_tsvector_i(c, i, val * get_tsvector_i(a, i));
		a4_0 = svld1_f32(pg, &(a->element[0][index]));
		a4_1 = svld1_f32(pg, &(a->element[1][index]));
		a4_2 = svld1_f32(pg, &(a->element[2][index]));

		_bncsve2_rts_mul(pg, &c4_0, &c4_1, &c4_2, val4_0, val4_1, val4_2, a4_0, a4_1, a4_2);

		svst1_f32(pg, &(c->element[0][index]), c4_0);
		svst1_f32(pg, &(c->element[1][index]), c4_1);
		svst1_f32(pg, &(c->element[2][index]), c4_2);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a4[TSSIZE], c4[TSSIZE], val4[TSSIZE];

	val4[0] = vdupq_n_f32(val[0]);
	val4[1] = vdupq_n_f32(val[1]);
	val4[2] = vdupq_n_f32(val[2]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		//set_tsvector_i(c, i, val * get_tsvector_i(a, i));
		a4[0] = vld1q_f32(&(a->element[0][index]));
		a4[1] = vld1q_f32(&(a->element[1][index]));
		a4[2] = vld1q_f32(&(a->element[2][index]));

		_bncneon_rts_mul(c4, val4, a4);

		vst1q_f32(&(c->element[0][index]), c4[0]);
		vst1q_f32(&(c->element[1][index]), c4[1]);
		vst1q_f32(&(c->element[2][index]), c4[2]);
	}
#else // others
	float tmp[TSSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rts_mul(tmp, val, get_tsvector_i(a, i));
		set_tsvector_i(c, i, tmp);
	}
#endif // __AVX2__

}

/* c *= val */
void cmul2_tsvector(TSVector c, float val[TSSIZE])
{
	long int i, index;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 c4[TSSIZE], val4[TSSIZE], tmp4[TSSIZE];

	val4[0] = _mm256_set1_ps(val[0]);
	val4[1] = _mm256_set1_ps(val[1]);
	val4[2] = _mm256_set1_ps(val[2]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		//set_tsvector_i(c, i, val * get_tsvector_i(a, i));
		c4[0] = _mm256_load_ps(&(c->element[0][index]));
		c4[1] = _mm256_load_ps(&(c->element[1][index]));
		c4[2] = _mm256_load_ps(&(c->element[2][index]));

		_bncavx2_rts_mul(tmp4, val4, c4);

		_mm256_store_ps(&(c->element[0][index]), tmp4[0]);
		_mm256_store_ps(&(c->element[1][index]), tmp4[1]);
		_mm256_store_ps(&(c->element[2][index]), tmp4[2]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 c4[TSSIZE], val4[TSSIZE], tmp4[TSSIZE];

	val4[0] = _mm512_set1_ps(val[0]);
	val4[1] = _mm512_set1_ps(val[1]);
	val4[2] = _mm512_set1_ps(val[2]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		//set_tsvector_i(c, i, val * get_tsvector_i(a, i));
		c4[0] = _mm512_load_ps(&(c->element[0][index]));
		c4[1] = _mm512_load_ps(&(c->element[1][index]));
		c4[2] = _mm512_load_ps(&(c->element[2][index]));

		_bncavx512_rts_mul(tmp4, val4, c4);

		_mm512_store_ps(&(c->element[0][index]), tmp4[0]);
		_mm512_store_ps(&(c->element[1][index]), tmp4[1]);
		_mm512_store_ps(&(c->element[2][index]), tmp4[2]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t c4_0, c4_1, c4_2;
	svfloat32_t val4_0, val4_1, val4_2;
	svfloat32_t tmp4_0, tmp4_1, tmp4_2;

	val4_0 = svdup_f32(val[0]);
	val4_1 = svdup_f32(val[1]);
	val4_2 = svdup_f32(val[2]);
	for(index = 0; index < c->real_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(c->real_dim));
		//set_tsvector_i(c, i, val * get_tsvector_i(a, i));
		c4_0 = svld1_f32(pg, &(c->element[0][index]));
		c4_1 = svld1_f32(pg, &(c->element[1][index]));
		c4_2 = svld1_f32(pg, &(c->element[2][index]));

		_bncsve2_rts_mul(pg, &tmp4_0, &tmp4_1, &tmp4_2, val4_0, val4_1, val4_2, c4_0, c4_1, c4_2);

		svst1_f32(pg, &(c->element[0][index]), tmp4_0);
		svst1_f32(pg, &(c->element[1][index]), tmp4_1);
		svst1_f32(pg, &(c->element[2][index]), tmp4_2);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t c4[TSSIZE], val4[TSSIZE], tmp4[TSSIZE];

	val4[0] = vdupq_n_f32(val[0]);
	val4[1] = vdupq_n_f32(val[1]);
	val4[2] = vdupq_n_f32(val[2]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		//set_tsvector_i(c, i, val * get_tsvector_i(a, i));
		c4[0] = vld1q_f32(&(c->element[0][index]));
		c4[1] = vld1q_f32(&(c->element[1][index]));
		c4[2] = vld1q_f32(&(c->element[2][index]));

		_bncneon_rts_mul(tmp4, val4, c4);

		vst1q_f32(&(c->element[0][index]), tmp4[0]);
		vst1q_f32(&(c->element[1][index]), tmp4[1]);
		vst1q_f32(&(c->element[2][index]), tmp4[2]);
	}
#else // others
	float tmp[TSSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rts_mul(tmp, val, get_tsvector_i(c, i));
		set_tsvector_i(c, i, tmp);
	}
#endif // __AVX2__	float tmp[TSSIZE];

}

/* c = a + val * b */
void add_cmul_tsvector(TSVector c, TSVector a, float val[TSSIZE], TSVector b)
{
	long int i, index;
	int _k;
	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_cmul_tsvector\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4[TSSIZE], b4[TSSIZE], c4[TSSIZE], val4[TSSIZE];
	for(_k = 0; _k < TSSIZE; _k++) val4[_k] = _mm256_set1_ps(val[_k]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		for(_k = 0; _k < TSSIZE; _k++){ a4[_k] = _mm256_load_ps(&(a->element[_k][index])); b4[_k] = _mm256_load_ps(&(b->element[_k][index])); }
		_bncavx2_rts_mul(c4, val4, b4);
		_bncavx2_rts_add(c4, a4, c4);
		for(_k = 0; _k < TSSIZE; _k++) _mm256_store_ps(&(c->element[_k][index]), c4[_k]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a4[TSSIZE], b4[TSSIZE], c4[TSSIZE], val4[TSSIZE];
	for(_k = 0; _k < TSSIZE; _k++) val4[_k] = _mm512_set1_ps(val[_k]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		for(_k = 0; _k < TSSIZE; _k++){ a4[_k] = _mm512_load_ps(&(a->element[_k][index])); b4[_k] = _mm512_load_ps(&(b->element[_k][index])); }
		_bncavx512_rts_mul(c4, val4, b4);
		_bncavx512_rts_add(c4, a4, c4);
		for(_k = 0; _k < TSSIZE; _k++) _mm512_store_ps(&(c->element[_k][index]), c4[_k]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t sa_0, sa_1, sa_2, sb_0, sb_1, sb_2, sc_0, sc_1, sc_2, sv_0, sv_1, sv_2;
	sv_0 = svdup_f32(val[0]); sv_1 = svdup_f32(val[1]); sv_2 = svdup_f32(val[2]);
	for(index = 0; index < c->real_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)c->real_dim);
		sa_0 = svld1_f32(pg, &(a->element[0][index])); sb_0 = svld1_f32(pg, &(b->element[0][index]));
		sa_1 = svld1_f32(pg, &(a->element[1][index])); sb_1 = svld1_f32(pg, &(b->element[1][index]));
		sa_2 = svld1_f32(pg, &(a->element[2][index])); sb_2 = svld1_f32(pg, &(b->element[2][index]));
		_bncsve2_rts_mul(pg, &sc_0, &sc_1, &sc_2, sv_0, sv_1, sv_2, sb_0, sb_1, sb_2);
		_bncsve2_rts_add(pg, &sc_0, &sc_1, &sc_2, sa_0, sa_1, sa_2, sc_0, sc_1, sc_2);
		svst1_f32(pg, &(c->element[0][index]), sc_0);
		svst1_f32(pg, &(c->element[1][index]), sc_1);
		svst1_f32(pg, &(c->element[2][index]), sc_2);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t na[TSSIZE], nb[TSSIZE], nc[TSSIZE], nv[TSSIZE];
	for(_k = 0; _k < TSSIZE; _k++) nv[_k] = vdupq_n_f32(val[_k]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		for(_k = 0; _k < TSSIZE; _k++){ na[_k] = vld1q_f32(&(a->element[_k][index])); nb[_k] = vld1q_f32(&(b->element[_k][index])); }
		_bncneon_rts_mul(nc, nv, nb);
		_bncneon_rts_add(nc, na, nc);
		for(_k = 0; _k < TSSIZE; _k++) vst1q_f32(&(c->element[_k][index]), nc[_k]);
	}
#else // scalar
	float tmp[TSSIZE];
	for(i = 0; i < c->dim; i++)
	{
		rts_mul(tmp, val, get_tsvector_i(b, i));
		rts_add(tmp, tmp, get_tsvector_i(a, i));
		set_tsvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

void sub_cmul_tsvector(TSVector c, TSVector a, float val[TSSIZE], TSVector b)
{
	long int i, index;
	int _k;
	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_cmul_tsvector\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4[TSSIZE], b4[TSSIZE], c4[TSSIZE], val4[TSSIZE];
	for(_k = 0; _k < TSSIZE; _k++) val4[_k] = _mm256_set1_ps(val[_k]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		for(_k = 0; _k < TSSIZE; _k++){ a4[_k] = _mm256_load_ps(&(a->element[_k][index])); b4[_k] = _mm256_load_ps(&(b->element[_k][index])); }
		_bncavx2_rts_mul(c4, val4, b4);
		_bncavx2_rts_sub(c4, a4, c4);
		for(_k = 0; _k < TSSIZE; _k++) _mm256_store_ps(&(c->element[_k][index]), c4[_k]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a4[TSSIZE], b4[TSSIZE], c4[TSSIZE], val4[TSSIZE];
	for(_k = 0; _k < TSSIZE; _k++) val4[_k] = _mm512_set1_ps(val[_k]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		for(_k = 0; _k < TSSIZE; _k++){ a4[_k] = _mm512_load_ps(&(a->element[_k][index])); b4[_k] = _mm512_load_ps(&(b->element[_k][index])); }
		_bncavx512_rts_mul(c4, val4, b4);
		_bncavx512_rts_sub(c4, a4, c4);
		for(_k = 0; _k < TSSIZE; _k++) _mm512_store_ps(&(c->element[_k][index]), c4[_k]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t sa_0, sa_1, sa_2, sb_0, sb_1, sb_2, sc_0, sc_1, sc_2, sv_0, sv_1, sv_2;
	sv_0 = svdup_f32(val[0]); sv_1 = svdup_f32(val[1]); sv_2 = svdup_f32(val[2]);
	for(index = 0; index < c->real_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)c->real_dim);
		sa_0 = svld1_f32(pg, &(a->element[0][index])); sb_0 = svld1_f32(pg, &(b->element[0][index]));
		sa_1 = svld1_f32(pg, &(a->element[1][index])); sb_1 = svld1_f32(pg, &(b->element[1][index]));
		sa_2 = svld1_f32(pg, &(a->element[2][index])); sb_2 = svld1_f32(pg, &(b->element[2][index]));
		_bncsve2_rts_mul(pg, &sc_0, &sc_1, &sc_2, sv_0, sv_1, sv_2, sb_0, sb_1, sb_2);
		_bncsve2_rts_neg(pg, &sc_0, &sc_1, &sc_2, sc_0, sc_1, sc_2);
		_bncsve2_rts_add(pg, &sc_0, &sc_1, &sc_2, sa_0, sa_1, sa_2, sc_0, sc_1, sc_2);
		svst1_f32(pg, &(c->element[0][index]), sc_0);
		svst1_f32(pg, &(c->element[1][index]), sc_1);
		svst1_f32(pg, &(c->element[2][index]), sc_2);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t na[TSSIZE], nb[TSSIZE], nc[TSSIZE], nv[TSSIZE];
	for(_k = 0; _k < TSSIZE; _k++) nv[_k] = vdupq_n_f32(val[_k]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		for(_k = 0; _k < TSSIZE; _k++){ na[_k] = vld1q_f32(&(a->element[_k][index])); nb[_k] = vld1q_f32(&(b->element[_k][index])); }
		_bncneon_rts_mul(nc, nv, nb);
		_bncneon_rts_sub(nc, na, nc);
		for(_k = 0; _k < TSSIZE; _k++) vst1q_f32(&(c->element[_k][index]), nc[_k]);
	}
#else // scalar
	float tmp[TSSIZE];
	for(i = 0; i < c->dim; i++)
	{
		rts_mul(tmp, val, get_tsvector_i(b, i));
		rts_sub(tmp, get_tsvector_i(a, i), tmp);
		set_tsvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* (a, b) */
void ip_tsvector(float ret[TSSIZE], TSVector a, TSVector b)
{
	long int i, index;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: ip_tsvector\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4[TSSIZE], b4[TSSIZE], ret4[TSSIZE], tmp4[TSSIZE];

	_bncavx2_set0_ts(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		a4[0] = _mm256_load_ps(&(a->element[0][index]));
		a4[1] = _mm256_load_ps(&(a->element[1][index]));
		a4[2] = _mm256_load_ps(&(a->element[2][index]));
		b4[0] = _mm256_load_ps(&(b->element[0][index]));
		b4[1] = _mm256_load_ps(&(b->element[1][index]));
		b4[2] = _mm256_load_ps(&(b->element[2][index]));

//		rts_mul(tmp, get_tsvector_i(a, i), get_tsvector_i(b, i));
//		rts_add(ret, ret, tmp);
		_bncavx2_rts_mul(tmp4, a4, b4);
		_bncavx2_rts_add(ret4, ret4, tmp4);
	}
	_bncavx2_rts_sum256(ret, ret4);
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a4[TSSIZE], b4[TSSIZE], ret4[TSSIZE], tmp4[TSSIZE];

	_bncavx512_set0_ts(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		a4[0] = _mm512_load_ps(&(a->element[0][index]));
		a4[1] = _mm512_load_ps(&(a->element[1][index]));
		a4[2] = _mm512_load_ps(&(a->element[2][index]));
		b4[0] = _mm512_load_ps(&(b->element[0][index]));
		b4[1] = _mm512_load_ps(&(b->element[1][index]));
		b4[2] = _mm512_load_ps(&(b->element[2][index]));

//		rts_mul(tmp, get_tsvector_i(a, i), get_tsvector_i(b, i));
//		rts_add(ret, ret, tmp);
		_bncavx512_rts_mul(tmp4, a4, b4);
		_bncavx512_rts_add(ret4, ret4, tmp4);
	}
	_bncavx512_rts_sum512(ret, ret4);
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t a4_0, a4_1, a4_2;
	svfloat32_t b4_0, b4_1, b4_2;
	svfloat32_t ret4_0, ret4_1, ret4_2;
	svfloat32_t tmp4_0, tmp4_1, tmp4_2;

	_bncsve2_rts_set0(&ret4_0, &ret4_1, &ret4_2);
	for(index = 0; index < a->real_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(a->real_dim));
		a4_0 = svld1_f32(pg, &(a->element[0][index]));
		a4_1 = svld1_f32(pg, &(a->element[1][index]));
		a4_2 = svld1_f32(pg, &(a->element[2][index]));
		b4_0 = svld1_f32(pg, &(b->element[0][index]));
		b4_1 = svld1_f32(pg, &(b->element[1][index]));
		b4_2 = svld1_f32(pg, &(b->element[2][index]));

//		rts_mul(tmp, get_tsvector_i(a, i), get_tsvector_i(b, i));
//		rts_add(ret, ret, tmp);
		_bncsve2_rts_mul(svptrue_b32(), &tmp4_0, &tmp4_1, &tmp4_2, a4_0, a4_1, a4_2, b4_0, b4_1, b4_2);
		_bncsve2_rts_add(svptrue_b32(), &ret4_0, &ret4_1, &ret4_2, ret4_0, ret4_1, ret4_2, tmp4_0, tmp4_1, tmp4_2);
	}
	_bncsve2_rts_sum128(svptrue_b32(), ret, ret4_0, ret4_1, ret4_2);
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a4[TSSIZE], b4[TSSIZE], ret4[TSSIZE], tmp4[TSSIZE];

	_bncneon_set0_ts(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		a4[0] = vld1q_f32(&(a->element[0][index]));
		a4[1] = vld1q_f32(&(a->element[1][index]));
		a4[2] = vld1q_f32(&(a->element[2][index]));
		b4[0] = vld1q_f32(&(b->element[0][index]));
		b4[1] = vld1q_f32(&(b->element[1][index]));
		b4[2] = vld1q_f32(&(b->element[2][index]));

//		rts_mul(tmp, get_tsvector_i(a, i), get_tsvector_i(b, i));
//		rts_add(ret, ret, tmp);
		_bncneon_rts_mul(tmp4, a4, b4);
		_bncneon_rts_add(ret4, ret4, tmp4);
	}
	_bncneon_rts_sum128f(ret, ret4);
#else // others
	float tmp[TSSIZE];

	set0_ts(ret);
	for(i = 0; i < a->dim; i++)
	{
		rts_mul(tmp, get_tsvector_i(a, i), get_tsvector_i(b, i));
		rts_add(ret, ret, tmp);
	}
#endif // __AVX2__

	return;
}

/* c := a */
void subst_tsvector(TSVector c, TSVector a)
{
	long int i;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
//	for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, get_fvector_i(a, i));
		_mm256_store_ps(&(c->element[0][i]), _mm256_load_ps(&(a->element[0][i])));
		_mm256_store_ps(&(c->element[1][i]), _mm256_load_ps(&(a->element[1][i])));
		_mm256_store_ps(&(c->element[2][i]), _mm256_load_ps(&(a->element[2][i])));
	}
#elif defined(__AVX512F__) // __AVX512F__
//	for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, get_fvector_i(a, i));
		_mm512_store_ps(&(c->element[0][i]), _mm512_load_ps(&(a->element[0][i])));
		_mm512_store_ps(&(c->element[1][i]), _mm512_load_ps(&(a->element[1][i])));
		_mm512_store_ps(&(c->element[2][i]), _mm512_load_ps(&(a->element[2][i])));
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
//	for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(a->real_dim));
		//set_fvector_i(c, i, get_fvector_i(a, i));
		svst1_f32(pg, &(c->element[0][i]), svld1_f32(pg, &(a->element[0][i])));
		svst1_f32(pg, &(c->element[1][i]), svld1_f32(pg, &(a->element[1][i])));
		svst1_f32(pg, &(c->element[2][i]), svld1_f32(pg, &(a->element[2][i])));
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
//	for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, get_fvector_i(a, i));
		vst1q_f32(&(c->element[0][i]), vld1q_f32(&(a->element[0][i])));
		vst1q_f32(&(c->element[1][i]), vld1q_f32(&(a->element[1][i])));
		vst1q_f32(&(c->element[2][i]), vld1q_f32(&(a->element[2][i])));
	}
#else // others
	for(i = 0; i < a->dim; i++)
		set_tsvector_i(c, i, get_tsvector_i(a, i));
#endif // __AVX2__
}

/* c := -a */
void neg_tsvector(TSVector c, TSVector a)
{
	long int i;
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 tmp[TSSIZE];

	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//rts_neg(tmp, get_tsvector_i(a, i));
		//set_tsvector_i(c, i, tmp);
		tmp[0] = _bncavx2_fneg(_mm256_load_ps(&(a->element[0][i])));
		tmp[1] = _bncavx2_fneg(_mm256_load_ps(&(a->element[1][i])));
		tmp[2] = _bncavx2_fneg(_mm256_load_ps(&(a->element[2][i])));
		_mm256_store_ps(&(c->element[0][i]), tmp[0]);
		_mm256_store_ps(&(c->element[1][i]), tmp[1]);
		_mm256_store_ps(&(c->element[2][i]), tmp[2]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 tmp[TSSIZE];

	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//rts_neg(tmp, get_tsvector_i(a, i));
		//set_tsvector_i(c, i, tmp);
		tmp[0] = _bncavx512_fneg(_mm512_load_ps(&(a->element[0][i])));
		tmp[1] = _bncavx512_fneg(_mm512_load_ps(&(a->element[1][i])));
		tmp[2] = _bncavx512_fneg(_mm512_load_ps(&(a->element[2][i])));
		_mm512_store_ps(&(c->element[0][i]), tmp[0]);
		_mm512_store_ps(&(c->element[1][i]), tmp[1]);
		_mm512_store_ps(&(c->element[2][i]), tmp[2]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t tmp_0, tmp_1, tmp_2;

	for(i = 0; i < a->real_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(a->real_dim));
		//rts_neg(tmp, get_tsvector_i(a, i));
		//set_tsvector_i(c, i, tmp);
		tmp_0 = svneg_f32_x(pg, svld1_f32(pg, &(a->element[0][i])));
		tmp_1 = svneg_f32_x(pg, svld1_f32(pg, &(a->element[1][i])));
		tmp_2 = svneg_f32_x(pg, svld1_f32(pg, &(a->element[2][i])));
		svst1_f32(pg, &(c->element[0][i]), tmp_0);
		svst1_f32(pg, &(c->element[1][i]), tmp_1);
		svst1_f32(pg, &(c->element[2][i]), tmp_2);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t tmp[TSSIZE];

	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//rts_neg(tmp, get_tsvector_i(a, i));
		//set_tsvector_i(c, i, tmp);
		tmp[0] = vnegq_f32(vld1q_f32(&(a->element[0][i])));
		tmp[1] = vnegq_f32(vld1q_f32(&(a->element[1][i])));
		tmp[2] = vnegq_f32(vld1q_f32(&(a->element[2][i])));
		vst1q_f32(&(c->element[0][i]), tmp[0]);
		vst1q_f32(&(c->element[1][i]), tmp[1]);
		vst1q_f32(&(c->element[2][i]), tmp[2]);
	}
#else // others
	float tmp[TSSIZE];

	for(i = 0; i < a->dim; i++)
	{
		rts_neg(tmp, get_tsvector_i(a, i));
		set_tsvector_i(c, i, tmp);
	}
#endif // __AVX2__
}


/* ||a||_1 */
void norm1_tsvector(float ret[TSSIZE], TSVector a)
{
	long int i, index, dim;

	dim = a->dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 vec4[TSSIZE], ret4[TSSIZE], tmp4[TSSIZE];

	_bncavx2_set0_ts(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		vec4[0] = _mm256_load_ps(&(a->element[0][index]));
		vec4[1] = _mm256_load_ps(&(a->element[1][index]));
		vec4[2] = _mm256_load_ps(&(a->element[2][index]));

		//rts_abs(tmp, get_tsvector_i(a, i));
		//rts_add(ret, ret, tmp);
		_bncavx2_rts_abs(tmp4, vec4);
		_bncavx2_rts_add(ret4, ret4, tmp4);
	}
	_bncavx2_rts_abssum256(ret, ret4);
#elif defined(__AVX512F__) // __AVX512F__
	__m512 vec4[TSSIZE], ret4[TSSIZE], tmp4[TSSIZE];

	_bncavx512_set0_ts(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		vec4[0] = _mm512_load_ps(&(a->element[0][index]));
		vec4[1] = _mm512_load_ps(&(a->element[1][index]));
		vec4[2] = _mm512_load_ps(&(a->element[2][index]));

		//rts_abs(tmp, get_tsvector_i(a, i));
		//rts_add(ret, ret, tmp);
		_bncavx512_rts_abs(tmp4, vec4);
		_bncavx512_rts_add(ret4, ret4, tmp4);
	}
	_bncavx512_rts_abssum512(ret, ret4);
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t vec4_0, vec4_1, vec4_2;
	svfloat32_t ret4_0, ret4_1, ret4_2;
	svfloat32_t tmp4_0, tmp4_1, tmp4_2;

	_bncsve2_rts_set0(&ret4_0, &ret4_1, &ret4_2);
	for(index = 0; index < a->real_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(a->real_dim));
		vec4_0 = svld1_f32(pg, &(a->element[0][index]));
		vec4_1 = svld1_f32(pg, &(a->element[1][index]));
		vec4_2 = svld1_f32(pg, &(a->element[2][index]));

		//rts_abs(tmp, get_tsvector_i(a, i));
		//rts_add(ret, ret, tmp);
		_bncsve2_rts_abs(svptrue_b32(), &tmp4_0, &tmp4_1, &tmp4_2, vec4_0, vec4_1, vec4_2);
		_bncsve2_rts_add(svptrue_b32(), &ret4_0, &ret4_1, &ret4_2, ret4_0, ret4_1, ret4_2, tmp4_0, tmp4_1, tmp4_2);
	}
	_bncsve2_rts_abssum128(svptrue_b32(), ret, ret4_0, ret4_1, ret4_2);
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t vec4[TSSIZE], ret4[TSSIZE], tmp4[TSSIZE];

	_bncneon_set0_ts(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		vec4[0] = vld1q_f32(&(a->element[0][index]));
		vec4[1] = vld1q_f32(&(a->element[1][index]));
		vec4[2] = vld1q_f32(&(a->element[2][index]));

		//rts_abs(tmp, get_tsvector_i(a, i));
		//rts_add(ret, ret, tmp);
		_bncneon_rts_abs(tmp4, vec4);
		_bncneon_rts_add(ret4, ret4, tmp4);
	}
	_bncneon_rts_abssum128f(ret, ret4);
#else // others
	float tmp[TSSIZE];

	set0_ts(ret);
	for(i = 0; i < a->dim; i++)
	{
		rts_abs(tmp, get_tsvector_i(a, i));
		rts_add(ret, ret, tmp);
	}
#endif // __AVX2__

	return;
}

/* ||a||_infty */
void normi_tsvector(float ret[TSSIZE], TSVector a)
{
	float tmp[TSSIZE];
	long int i;

	rts_abs(ret, get_tsvector_i(a, 0));
	for(i = 1; i < a->dim; i++)
	{
		rts_abs(tmp, get_tsvector_i(a, i));
		if(rts_cmp(ret, tmp) < 0)
			rts_set(ret, tmp);
	}

	return;
}

// Euclid norm
void norm2_tsvector(float ret[TSSIZE], TSVector vec)
{
	long int i, index, dim;

	dim = vec->dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 vec4[TSSIZE], ret4[TSSIZE], tmp4[TSSIZE];

	_bncavx2_set0_ts(ret4);
	for(index = 0; index < vec->real_dim; index += _BNC_S_WIDTH)
	{
		vec4[0] = _mm256_load_ps(&(vec->element[0][index]));
		vec4[1] = _mm256_load_ps(&(vec->element[1][index]));
		vec4[2] = _mm256_load_ps(&(vec->element[2][index]));

//		rts_mul(tmp, get_tsvector_i(vec, i), get_tsvector_i(vec, i));
//		rts_add(ret, ret, tmp);
		_bncavx2_rts_mul(tmp4, vec4, vec4);
		_bncavx2_rts_add(ret4, ret4, tmp4);
	}
	_bncavx2_rts_norm256(ret, ret4);
#elif defined(__AVX512F__) // __AVX512F__
	__m512 vec4[TSSIZE], ret4[TSSIZE], tmp4[TSSIZE];

	_bncavx512_set0_ts(ret4);
	for(index = 0; index < vec->real_dim; index += _BNC_S_WIDTH)
	{
		vec4[0] = _mm512_load_ps(&(vec->element[0][index]));
		vec4[1] = _mm512_load_ps(&(vec->element[1][index]));
		vec4[2] = _mm512_load_ps(&(vec->element[2][index]));

//		rts_mul(tmp, get_tsvector_i(vec, i), get_tsvector_i(vec, i));
//		rts_add(ret, ret, tmp);
		_bncavx512_rts_mul(tmp4, vec4, vec4);
		_bncavx512_rts_add(ret4, ret4, tmp4);
	}
	_bncavx512_rts_norm512(ret, ret4);
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t vec4_0, vec4_1, vec4_2;
	svfloat32_t ret4_0, ret4_1, ret4_2;
	svfloat32_t tmp4_0, tmp4_1, tmp4_2;

	_bncsve2_rts_set0(&ret4_0, &ret4_1, &ret4_2);
	for(index = 0; index < vec->real_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(vec->real_dim));
		vec4_0 = svld1_f32(pg, &(vec->element[0][index]));
		vec4_1 = svld1_f32(pg, &(vec->element[1][index]));
		vec4_2 = svld1_f32(pg, &(vec->element[2][index]));

//		rts_mul(tmp, get_tsvector_i(vec, i), get_tsvector_i(vec, i));
//		rts_add(ret, ret, tmp);
		_bncsve2_rts_mul(svptrue_b32(), &tmp4_0, &tmp4_1, &tmp4_2, vec4_0, vec4_1, vec4_2, vec4_0, vec4_1, vec4_2);
		_bncsve2_rts_add(svptrue_b32(), &ret4_0, &ret4_1, &ret4_2, ret4_0, ret4_1, ret4_2, tmp4_0, tmp4_1, tmp4_2);
	}
	_bncsve2_rts_norm128(svptrue_b32(), ret, ret4_0, ret4_1, ret4_2);
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t vec4[TSSIZE], ret4[TSSIZE], tmp4[TSSIZE];

	_bncneon_set0_ts(ret4);
	for(index = 0; index < vec->real_dim; index += _BNC_S_WIDTH)
	{
		vec4[0] = vld1q_f32(&(vec->element[0][index]));
		vec4[1] = vld1q_f32(&(vec->element[1][index]));
		vec4[2] = vld1q_f32(&(vec->element[2][index]));

//		rts_mul(tmp, get_tsvector_i(vec, i), get_tsvector_i(vec, i));
//		rts_add(ret, ret, tmp);
		_bncneon_rts_mul(tmp4, vec4, vec4);
		_bncneon_rts_add(ret4, ret4, tmp4);
	}
	_bncneon_rts_norm128f(ret, ret4);
#else // others
	float tmp[TSSIZE];

	//c_dd_copy_d((float)0.0, tmp);
	//c_dd_copy_d((float)0.0, ret);
	rts_set0(tmp);
	rts_set0(ret);

	for(i = 0; i < dim ; i++)
	{
		//c_dd_sqr(GET_TSVECTOR_I(vec, i), tmp);
		//c_dd_add(tmp, ret, ret);
		rts_mul(tmp, get_tsvector_i(vec, i), get_tsvector_i(vec, i));
		rts_add(ret, ret, tmp);
	}

	//c_ts_sqrt(ret, tmp);
	//c_ts_copy(tmp, ret);
	rts_sqrt(tmp, ret);
	rts_set(ret, tmp);
#endif // __AVX2__
}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
void _bncavx2_tsadd(tsfloat ret[], tsfloat a[], tsfloat b[], int dim)
{
    int i, index, div, rem, unit = _BNC_S_WIDTH;
    __m256 in_ret[3], in_a_val[3], in_b_val[3], in_z[6], in_e[6];

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

#ifdef USE_RTS_ADD
        _bncavx2_rts_addt(in_ret, in_a_val, in_b_val);
#else // USE_RTS_ADD
        _bncavx2_rts_addq(in_ret, in_a_val, in_b_val);
#endif // USE_RTS_ADD//

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
    }
}

void _bncavx2_tsvadd(TSVector ret, TSVector a, TSVector b, int dim)
{
    int i, index, div, rem, unit = _BNC_S_WIDTH;
    __m256 in_ret[3], in_a_val[3], in_b_val[3], in_z[6], in_e[6];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_ps(&(a->element[1][index]));
        in_a_val[2] = _mm256_load_ps(&(a->element[2][index]));
        in_b_val[0] = _mm256_load_ps(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_ps(&(b->element[1][index]));
        in_b_val[2] = _mm256_load_ps(&(b->element[2][index]));

#ifdef USE_RTS_ADD
        _bncavx2_rts_addt(in_ret, in_a_val, in_b_val);
#else // USE_RTS_ADD
        _bncavx2_rts_addq(in_ret, in_a_val, in_b_val);
#endif // USE_RTS_ADD

        _mm256_store_ps(&ret->element[0][index], in_ret[0]);
        _mm256_store_ps(&ret->element[1][index], in_ret[1]);
        _mm256_store_ps(&ret->element[2][index], in_ret[2]);
   }
}

void _bncavx2_tsmul(tsfloat ret[], tsfloat a[], tsfloat b[], int dim)
{
    int i, index, div, rem, unit = _BNC_S_WIDTH;
    __m256 in_a_val[3], in_b_val[3], in_ret[3];
	__m256 z00[2], z01[2], z10[2];
	__m256 in_b[3], in_c, z[3], e[4], temp[4];

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

/*
        z00[0] = _bncavx2_dtwo_prod(in_a_val[0], in_b_val[0], &z00[1]);
        z01[0] = _bncavx2_dtwo_prod(in_a_val[0], in_b_val[1], &z01[1]);
        z10[0] = _bncavx2_dtwo_prod(in_a_val[1], in_b_val[0], &z10[1]);

        z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];

        _bncavx2_vec_sum(in_b, z, 3);
        in_c = _mm256_fmadd_ps(in_a_val[1], in_b_val[1], in_b[2]);

        z[0] = _mm256_fmadd_ps(in_a_val[0], in_b_val[2], z10[1]);
        z[1] = _mm256_fmadd_ps(in_a_val[2], in_b_val[0], z01[1]);
        z[2] = _mm256_add_ps(z[0], z[1]);
        temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1]; 
        temp[3] = _mm256_add_ps(in_c, z[2]);
        _bncavx2_vec_sum(e, temp, 4);
        in_ret[0] = e[0];
        _bncavx2_vseb(&in_ret[1], 2, &e[1], 3);
*/
        _bncavx2_rts_mul(in_ret, in_a_val, in_b_val);
//        _bncavx2_rts_mulq(in_ret, in_a_val, in_b_val);

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
        ret[index + 4].val[0] = in_ret[0][4];
        ret[index + 5].val[0] = in_ret[0][5];
        ret[index + 6].val[0] = in_ret[0][6];
        ret[index + 7].val[0] = in_ret[0][7];

    }
}

void _bncavx2_tsvmul(TSVector ret, TSVector a, TSVector b, int dim)
{
    int i, index, div, rem, unit = _BNC_S_WIDTH;
    __m256 in_a_val[3], in_b_val[3], in_ret[3];
	__m256 z00[2], z01[2], z10[2];
	__m256 in_b[3], in_c, z[3], e[4], temp[4];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_ps(&(a->element[1][index]));
        in_a_val[2] = _mm256_load_ps(&(a->element[2][index]));
        in_b_val[0] = _mm256_load_ps(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_ps(&(b->element[1][index]));
        in_b_val[2] = _mm256_load_ps(&(b->element[2][index]));

        _bncavx2_rts_mul(in_ret, in_a_val, in_b_val);
        //_bncavx2_rts_mulq(in_ret, in_a_val, in_b_val);

        _mm256_store_ps(&ret->element[0][index], in_ret[0]);
        _mm256_store_ps(&ret->element[1][index], in_ret[1]);
        _mm256_store_ps(&ret->element[2][index], in_ret[2]);
   }
}

/* tddiv */
void _bncavx2_tsdiv(tsfloat ret[], tsfloat a[], tsfloat b[], int dim)
{
    int i, index, div, rem, unit = _BNC_S_WIDTH;
    __m256 in_a_val[3], in_b_val[3], in_ret[3];
	__m256 z00[2], z01[2], z10[2];
	__m256 in_b[3], in_c, z[3], e[4], temp[4];

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

/*
        z00[0] = _bncavx2_dtwo_prod(in_a_val[0], in_b_val[0], &z00[1]);
        z01[0] = _bncavx2_dtwo_prod(in_a_val[0], in_b_val[1], &z01[1]);
        z10[0] = _bncavx2_dtwo_prod(in_a_val[1], in_b_val[0], &z10[1]);

        z[0] = z00[1]; z[1] = z01[0]; z[2] = z10[0];

        _bncavx2_vec_sum(in_b, z, 3);
        in_c = _mm256_fmadd_ps(in_a_val[1], in_b_val[1], in_b[2]);

        z[0] = _mm256_fmadd_ps(in_a_val[0], in_b_val[2], z10[1]);
        z[1] = _mm256_fmadd_ps(in_a_val[2], in_b_val[0], z01[1]);
        z[2] = _mm256_add_ps(z[0], z[1]);
        temp[0] = z00[0]; temp[1] = in_b[0]; temp[2] = in_b[1]; 
        temp[3] = _mm256_add_ps(in_c, z[2]);
        _bncavx2_vec_sum(e, temp, 4);
        in_ret[0] = e[0];
        _bncavx2_vseb(&in_ret[1], 2, &e[1], 3);
*/
        _bncavx2_rts_div(in_ret, in_a_val, in_b_val);
//        _bncavx2_rts_mulq(in_ret, in_a_val, in_b_val);

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

    }
}

void _bncavx2_tsvdiv(TSVector ret, TSVector a, TSVector b, int dim)
{
    int i, index, div, rem, unit = _BNC_S_WIDTH;
    __m256 in_a_val[3], in_b_val[3], in_ret[3];
	__m256 z00[2], z01[2], z10[2];
	__m256 in_b[3], in_c, z[3], e[4], temp[4];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_ps(&(a->element[1][index]));
        in_a_val[2] = _mm256_load_ps(&(a->element[2][index]));
        in_b_val[0] = _mm256_load_ps(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_ps(&(b->element[1][index]));
        in_b_val[2] = _mm256_load_ps(&(b->element[2][index]));

        _bncavx2_rts_div(in_ret, in_a_val, in_b_val);
        //_bncavx2_rts_mulq(in_ret, in_a_val, in_b_val);

        _mm256_store_ps(&ret->element[0][index], in_ret[0]);
        _mm256_store_ps(&ret->element[1][index], in_ret[1]);
        _mm256_store_ps(&ret->element[2][index], in_ret[2]);
   }
}
#endif // if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__

// set a zero matrix
//void set0_tsmatrix(TSMatrix mat)
void set0_tsmatrix(TSMatrix mat)
{
	long int i;
	long int real_total_dim;

	real_total_dim = mat->real_row_dim * mat->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 zero4;

	zero4 = _mm256_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm256_store_ps(&mat->element[0][i], zero4);
		_mm256_store_ps(&mat->element[1][i], zero4);
		_mm256_store_ps(&mat->element[2][i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero4;

	zero4 = _mm512_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm512_store_ps(&mat->element[0][i], zero4);
		_mm512_store_ps(&mat->element[1][i], zero4);
		_mm512_store_ps(&mat->element[2][i], zero4);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t zero4;

	zero4 = svdup_f32(0.0f);
	for(i = 0; i < real_total_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(real_total_dim));
		svst1_f32(pg, &mat->element[0][i], zero4);
		svst1_f32(pg, &mat->element[1][i], zero4);
		svst1_f32(pg, &mat->element[2][i], zero4);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t zero4;

	zero4 = vdupq_n_f32(0.0f);
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		vst1q_f32(&mat->element[0][i], zero4);
		vst1q_f32(&mat->element[1][i], zero4);
		vst1q_f32(&mat->element[2][i], zero4);
	}
#else // others
	for(i = 0; i < real_total_dim; i++)
	{
		mat->element[0][i] = 0.0f;
		mat->element[1][i] = 0.0f;	
		mat->element[2][i] = 0.0f;	
	}
#endif // __AVX2__
}

// initialize tsvector
TSMatrix init_tsmatrix(long int row_dim, long int col_dim)
{
	long int row_index, col_index, i;
	long int real_row_dim, real_col_dim, real_total_dim;
	TSMatrix ret = NULL;

	ret = (TSMatrix)BNC_MALLOC(sizeof(tsmatrix));
	if(ret == NULL)
		return ret;

	// real_dim is the nearest positive multiplier of _BNC_S_WIDTH
	real_row_dim = (long int)ceil((float)(row_dim) / (float)_BNC_S_WIDTH) * _BNC_S_WIDTH;
	real_col_dim = (long int)ceil((float)(col_dim) / (float)_BNC_S_WIDTH) * _BNC_S_WIDTH;
	real_total_dim = real_row_dim * real_col_dim;

	//printf("init_tsmatrix(%ld, %ld) %ld calloc\n", row_dim, col_dim, real_total_dim);
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

	//printf("init_tsmatrix(%ld, %ld) calloc\n", row_dim, col_dim);
	/* All 0 */
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 zero4;

	zero4 = _mm256_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm256_store_ps(&ret->element[0][i], zero4);
		_mm256_store_ps(&ret->element[1][i], zero4);
		_mm256_store_ps(&ret->element[2][i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero4;

	zero4 = _mm512_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm512_store_ps(&ret->element[0][i], zero4);
		_mm512_store_ps(&ret->element[1][i], zero4);
		_mm512_store_ps(&ret->element[2][i], zero4);
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
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t zero4;

	zero4 = vdupq_n_f32(0.0f);
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		vst1q_f32(&ret->element[0][i], zero4);
		vst1q_f32(&ret->element[1][i], zero4);
		vst1q_f32(&ret->element[2][i], zero4);
	}
#else // others
	for(i = 0; i < real_total_dim; i++)
	{
		ret->element[0][i] = 0.0f;
		ret->element[1][i] = 0.0f;
		ret->element[2][i] = 0.0f;
	}
#endif // __AVX2__

	ret->real_row_dim = real_row_dim;
	ret->real_col_dim = real_col_dim;
	ret->row_dim = row_dim;
	ret->col_dim = col_dim;

	return ret;
}

// free tsvector
void free_tsmatrix(TSMatrix mat)
{
	long int i;

	for(i = 0; i < TSSIZE; i++)
		free(mat->element[i]);

	free(mat);
}

// print tsvector
void print_tsmatrix(TSMatrix mat)
{
	long int row_index, col_index;

	for(row_index = 0; row_index < mat->row_dim; row_index++)
	{
		for(col_index = 0; col_index < mat->col_dim; col_index++)
		{
			printf("%ld, %ld: ", row_index, col_index);
	//		c_dd_write((vec->element + index * TSSIZE));
			c_ts_write(GET_TSMATRIX_IJ(mat, row_index, col_index));
		}
	}
}

// TSMatrix mat -> tsfloat array
void set_tsfloat_tsmat(tsfloat ret[], int ret_dim, TSMatrix mat)
{
    int total_index, j, total_dim;
	long int row_index, col_index;

    total_dim = (ret_dim < (mat->row_dim * mat->col_dim)) ? ret_dim : (mat->row_dim * mat->col_dim);

	total_index = 0;
    for(row_index = 0; row_index < mat->row_dim; row_index++)
    {
		for(col_index = 0; (col_index < mat->col_dim) && (total_index < total_dim); col_index++)
		{
			for(j = 0; j < TSSIZE; j++)
				ret[total_index].val[j] = mat->element[j][(row_index * mat->real_col_dim) + col_index];

			total_index++;
		}
    }
}

// tsfloat array -> TDmatrix ret
void set_tsmatrix_tsfloat(TSMatrix ret, tsfloat array[], int array_dim)
{
    int total_index, j, total_dim;
	long int row_index, col_index;

    total_dim = (array_dim < (ret->row_dim * ret->col_dim)) ? array_dim : (ret->row_dim * ret->col_dim);

 	total_index = 0;
    for(row_index = 0; row_index < ret->row_dim; row_index++)
    {
		for(col_index = 0; (col_index < ret->col_dim) && (total_index < total_dim); col_index++)
		{
			for(j = 0; j < TSSIZE; j++)
				ret->element[j][(row_index * ret->real_col_dim) + col_index] = array[total_index].val[j];

			total_index++;
		}
    }
}


// matrix multiplication
// ret := A * B
void mul_tsmatrix(TSMatrix ret, TSMatrix a, TSMatrix b)
{
	long int i, j, k;

	/* dimension check */
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: mul_tsmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret->row_dim, ret->col_dim, a->row_dim, a->col_dim, b->row_dim, b->col_dim);
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long real_row_dim, real_col_dim, real_mid_dim;
	//float cijval[4][TSSIZE];
	float cijval[8][TSSIZE];
    __m256 cij[TSSIZE], aik[TSSIZE], bkj[TSSIZE], tmp_mul[TSSIZE];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j++)
        {
            //rts_set_ui(cij.val, 0UL);
            cij[0] = _mm256_setzero_ps();
            cij[1] = _mm256_setzero_ps();
            cij[2] = _mm256_setzero_ps();
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
                aik[2] = _mm256_load_ps(&(a->element[2][i * real_mid_dim + k]));
                
            /*    aik[1] = _mm256_set_ps(
                    a->element[1][i * real_mid_dim + k],
                    a->element[1][i * real_mid_dim + k + 1],
                    a->element[1][i * real_mid_dim + k + 2],
                    a->element[1][i * real_mid_dim + k + 3]
                );
                aik[2] = _mm256_set_ps(
                    a->element[2][i * real_mid_dim + k],
                    a->element[2][i * real_mid_dim + k + 1],
                    a->element[2][i * real_mid_dim + k + 2],
                    a->element[2][i * real_mid_dim + k + 3]
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

            /*
                rts_mul(tmp_mul[0].val, aik[0].val, bkj[0].val);
                rts_mul(tmp_mul[1].val, aik[1].val, bkj[1].val);
                rts_mul(tmp_mul[2].val, aik[2].val, bkj[2].val);
                rts_mul(tmp_mul[3].val, aik[3].val, bkj[3].val);
            */
                _bncavx2_rts_mul(tmp_mul, aik, bkj);

            /*
                rts_add(cij.val, cij.val, tmp_mul[0].val);
                rts_add(cij.val, cij.val, tmp_mul[1].val);
                rts_add(cij.val, cij.val, tmp_mul[2].val);
                rts_add(cij.val, cij.val, tmp_mul[3].val);
            */
                _bncavx2_rts_add(cij, cij, tmp_mul);
            }

            cijval[0][0] = cij[0][0]; cijval[0][1] = cij[1][0]; cijval[0][2] = cij[2][0];
            cijval[1][0] = cij[0][1]; cijval[1][1] = cij[1][1]; cijval[1][2] = cij[2][1];
            cijval[2][0] = cij[0][2]; cijval[2][1] = cij[1][2]; cijval[2][2] = cij[2][2];
            cijval[3][0] = cij[0][3]; cijval[3][1] = cij[1][3]; cijval[3][2] = cij[2][3];
            cijval[4][0] = cij[0][4]; cijval[4][1] = cij[1][4]; cijval[4][2] = cij[2][4];
            cijval[5][0] = cij[0][5]; cijval[5][1] = cij[1][5]; cijval[5][2] = cij[2][5];
            cijval[6][0] = cij[0][6]; cijval[6][1] = cij[1][6]; cijval[6][2] = cij[2][6];
            cijval[7][0] = cij[0][7]; cijval[7][1] = cij[1][7]; cijval[7][2] = cij[2][7];
            rts_add(cijval[0], cijval[0], cijval[1]);
            rts_add(cijval[0], cijval[0], cijval[2]);
            rts_add(cijval[0], cijval[0], cijval[3]);
            rts_add(cijval[0], cijval[0], cijval[4]);
            rts_add(cijval[0], cijval[0], cijval[5]);
            rts_add(cijval[0], cijval[0], cijval[6]);
            rts_add(cijval[0], cijval[0], cijval[7]);

            ret->element[0][i * real_col_dim + j] = cijval[0][0];
            ret->element[1][i * real_col_dim + j] = cijval[0][1];
            ret->element[2][i * real_col_dim + j] = cijval[0][2];
        }
    }
#elif defined(__AVX512F__) // __AVX512F__
	long real_row_dim, real_col_dim, real_mid_dim;
	//float cijval[4][TSSIZE];
	float cijval[16][TSSIZE];
    __m512 cij[TSSIZE], aik[TSSIZE], bkj[TSSIZE], tmp_mul[TSSIZE];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j++)
        {
            //rts_set_ui(cij.val, 0UL);
            cij[0] = _mm512_setzero_ps();
            cij[1] = _mm512_setzero_ps();
            cij[2] = _mm512_setzero_ps();
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
                aik[2] = _mm512_load_ps(&(a->element[2][i * real_mid_dim + k]));
                
            /*    aik[1] = _mm512_set_ps(
                    a->element[1][i * real_mid_dim + k],
                    a->element[1][i * real_mid_dim + k + 1],
                    a->element[1][i * real_mid_dim + k + 2],
                    a->element[1][i * real_mid_dim + k + 3]
                );
                aik[2] = _mm512_set_ps(
                    a->element[2][i * real_mid_dim + k],
                    a->element[2][i * real_mid_dim + k + 1],
                    a->element[2][i * real_mid_dim + k + 2],
                    a->element[2][i * real_mid_dim + k + 3]
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

            /*
                rts_mul(tmp_mul[0].val, aik[0].val, bkj[0].val);
                rts_mul(tmp_mul[1].val, aik[1].val, bkj[1].val);
                rts_mul(tmp_mul[2].val, aik[2].val, bkj[2].val);
                rts_mul(tmp_mul[3].val, aik[3].val, bkj[3].val);
            */
                _bncavx512_rts_mul(tmp_mul, aik, bkj);

            /*
                rts_add(cij.val, cij.val, tmp_mul[0].val);
                rts_add(cij.val, cij.val, tmp_mul[1].val);
                rts_add(cij.val, cij.val, tmp_mul[2].val);
                rts_add(cij.val, cij.val, tmp_mul[3].val);
            */
                _bncavx512_rts_add(cij, cij, tmp_mul);
            }

            cijval[0][0] = cij[0][0]; cijval[0][1] = cij[1][0]; cijval[0][2] = cij[2][0];
            cijval[1][0] = cij[0][1]; cijval[1][1] = cij[1][1]; cijval[1][2] = cij[2][1];
            cijval[2][0] = cij[0][2]; cijval[2][1] = cij[1][2]; cijval[2][2] = cij[2][2];
            cijval[3][0] = cij[0][3]; cijval[3][1] = cij[1][3]; cijval[3][2] = cij[2][3];
            cijval[4][0] = cij[0][4]; cijval[4][1] = cij[1][4]; cijval[4][2] = cij[2][4];
            cijval[5][0] = cij[0][5]; cijval[5][1] = cij[1][5]; cijval[5][2] = cij[2][5];
            cijval[6][0] = cij[0][6]; cijval[6][1] = cij[1][6]; cijval[6][2] = cij[2][6];
            cijval[7][0] = cij[0][7]; cijval[7][1] = cij[1][7]; cijval[7][2] = cij[2][7];
            cijval[8][0] = cij[0][8]; cijval[8][1] = cij[1][8]; cijval[8][2] = cij[2][8];
            cijval[9][0] = cij[0][9]; cijval[9][1] = cij[1][9]; cijval[9][2] = cij[2][9];
            cijval[10][0] = cij[0][10]; cijval[10][1] = cij[1][10]; cijval[10][2] = cij[2][10];
            cijval[11][0] = cij[0][11]; cijval[11][1] = cij[1][11]; cijval[11][2] = cij[2][11];
            cijval[12][0] = cij[0][12]; cijval[12][1] = cij[1][12]; cijval[12][2] = cij[2][12];
            cijval[13][0] = cij[0][13]; cijval[13][1] = cij[1][13]; cijval[13][2] = cij[2][13];
            cijval[14][0] = cij[0][14]; cijval[14][1] = cij[1][14]; cijval[14][2] = cij[2][14];
            cijval[15][0] = cij[0][15]; cijval[15][1] = cij[1][15]; cijval[15][2] = cij[2][15];
            rts_add(cijval[0], cijval[0], cijval[1]);
            rts_add(cijval[0], cijval[0], cijval[2]);
            rts_add(cijval[0], cijval[0], cijval[3]);
            rts_add(cijval[0], cijval[0], cijval[4]);
            rts_add(cijval[0], cijval[0], cijval[5]);
            rts_add(cijval[0], cijval[0], cijval[6]);
            rts_add(cijval[0], cijval[0], cijval[7]);
            rts_add(cijval[0], cijval[0], cijval[8]);
            rts_add(cijval[0], cijval[0], cijval[9]);
            rts_add(cijval[0], cijval[0], cijval[10]);
            rts_add(cijval[0], cijval[0], cijval[11]);
            rts_add(cijval[0], cijval[0], cijval[12]);
            rts_add(cijval[0], cijval[0], cijval[13]);
            rts_add(cijval[0], cijval[0], cijval[14]);
            rts_add(cijval[0], cijval[0], cijval[15]);

            ret->element[0][i * real_col_dim + j] = cijval[0][0];
            ret->element[1][i * real_col_dim + j] = cijval[0][1];
            ret->element[2][i * real_col_dim + j] = cijval[0][2];
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
				svfloat32_t cij0, cij1, cij2;
				_bncsve2_rts_set0(&cij0, &cij1, &cij2);
				for(k = 0; k < real_mid_dim; k++){
					svfloat32_t aik0 = svdup_n_f32(a->element[0][i*real_mid_dim + k]);
					svfloat32_t aik1 = svdup_n_f32(a->element[1][i*real_mid_dim + k]);
					svfloat32_t aik2 = svdup_n_f32(a->element[2][i*real_mid_dim + k]);
					svfloat32_t bkj0 = svld1_f32(pg, &(b->element[0][k*real_col_dim + j]));
					svfloat32_t bkj1 = svld1_f32(pg, &(b->element[1][k*real_col_dim + j]));
					svfloat32_t bkj2 = svld1_f32(pg, &(b->element[2][k*real_col_dim + j]));
					svfloat32_t t0, t1, t2;
					_bncsve2_rts_mul(pg, &t0, &t1, &t2, aik0, aik1, aik2, bkj0, bkj1, bkj2);
					_bncsve2_rts_add(pg, &cij0, &cij1, &cij2, cij0, cij1, cij2, t0, t1, t2);
				}
				svst1_f32(pg, &(ret->element[0][i*real_col_dim + j]), cij0);
				svst1_f32(pg, &(ret->element[1][i*real_col_dim + j]), cij1);
				svst1_f32(pg, &(ret->element[2][i*real_col_dim + j]), cij2);
			}
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	long real_row_dim = a->real_row_dim;
	long real_col_dim = b->real_col_dim;
	long real_mid_dim = a->real_col_dim;
	float32x4_t cij[TSSIZE], aik[TSSIZE], bkj[TSSIZE], tmp[TSSIZE];

	for(i = 0; i < real_row_dim; i += 1){
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH){
			_bncneon_set0_ts(cij);
			for(k = 0; k < real_mid_dim; k += 1){
				aik[0] = vdupq_n_f32(a->element[0][i*real_mid_dim + k]);
				aik[1] = vdupq_n_f32(a->element[1][i*real_mid_dim + k]);
				aik[2] = vdupq_n_f32(a->element[2][i*real_mid_dim + k]);
				bkj[0] = vld1q_f32(&(b->element[0][k*real_col_dim + j]));
				bkj[1] = vld1q_f32(&(b->element[1][k*real_col_dim + j]));
				bkj[2] = vld1q_f32(&(b->element[2][k*real_col_dim + j]));
				_bncneon_rts_mul(tmp, aik, bkj);
				_bncneon_rts_add(cij, cij, tmp);
			}
			vst1q_f32(&(ret->element[0][i*real_col_dim + j]), cij[0]);
			vst1q_f32(&(ret->element[1][i*real_col_dim + j]), cij[1]);
			vst1q_f32(&(ret->element[2][i*real_col_dim + j]), cij[2]);
		}
	}
#else // __AVX2__
	long row_dim, col_dim, mid_dim;
	float tmp[TSSIZE], ret_ij[TSSIZE];

	//printf("Non SIMD mul_tsmatrix(%ld, %ld)\n", ret->row_dim, ret->col_dim);
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = a->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			//rts_set0(GET_TSMATRIX_IJ(ret, i, j));
			rts_set0(ret_ij);
			for(k = 0; k < mid_dim; k++)
			{
				rts_mul(tmp, GET_TSMATRIX_IJ(a, i, k), GET_TSMATRIX_IJ(b, k, j));
				//rts_add(GET_TSMATRIX_IJ(ret, i, j), tmp, GET_TSMATRIX_IJ(ret, i, j));
				rts_add(ret_ij, tmp, ret_ij);
			}
			set_tsmatrix_ij(ret, i, j, ret_ij);
		}
	}
	//printf("end(%ld, %ld)\n", ret->row_dim, ret->col_dim);
#endif // __AVX2__

}

// Frobenius norm
void normf_tsmatrix(float ret[TSSIZE], TSMatrix mat)
{
	long int i;
	long int real_total_dim;
	float tmp[TSSIZE];

	real_total_dim = mat->real_row_dim * mat->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 ret4[TSSIZE], mat4[TSSIZE], tmp4[TSSIZE];

	ret4[0] = _mm256_setzero_ps();
	ret4[1] = _mm256_setzero_ps();
	ret4[2] = _mm256_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		mat4[0] = _mm256_load_ps(&mat->element[0][i]);
		mat4[1] = _mm256_load_ps(&mat->element[1][i]);
		mat4[2] = _mm256_load_ps(&mat->element[2][i]);

		// tmp4 := mat4[i]^2
		// ret4 += tmp4
		_bncavx2_rts_mul(tmp4, mat4, mat4);
		_bncavx2_rts_add(ret4, ret4, tmp4);
	}

	_bncavx2_rts_sum256(ret, ret4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512 ret4[TSSIZE], mat4[TSSIZE], tmp4[TSSIZE];

	ret4[0] = _mm512_setzero_ps();
	ret4[1] = _mm512_setzero_ps();
	ret4[2] = _mm512_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		mat4[0] = _mm512_load_ps(&mat->element[0][i]);
		mat4[1] = _mm512_load_ps(&mat->element[1][i]);
		mat4[2] = _mm512_load_ps(&mat->element[2][i]);

		// tmp4 := mat4[i]^2
		// ret4 += tmp4
		_bncavx512_rts_mul(tmp4, mat4, mat4);
		_bncavx512_rts_add(ret4, ret4, tmp4);
	}

	_bncavx512_rts_sum512(ret, ret4);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t ret4_0, ret4_1, ret4_2;
	svfloat32_t mat4_0, mat4_1, mat4_2;
	svfloat32_t tmp4_0, tmp4_1, tmp4_2;

	ret4_0 = svdup_f32(0.0f);
	ret4_1 = svdup_f32(0.0f);
	ret4_2 = svdup_f32(0.0f);
	for(i = 0; i < real_total_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(real_total_dim));
		mat4_0 = svld1_f32(pg, &mat->element[0][i]);
		mat4_1 = svld1_f32(pg, &mat->element[1][i]);
		mat4_2 = svld1_f32(pg, &mat->element[2][i]);

		// tmp4 := mat4[i]^2
		// ret4 += tmp4
		_bncsve2_rts_mul(svptrue_b32(), &tmp4_0, &tmp4_1, &tmp4_2, mat4_0, mat4_1, mat4_2, mat4_0, mat4_1, mat4_2);
		_bncsve2_rts_add(svptrue_b32(), &ret4_0, &ret4_1, &ret4_2, ret4_0, ret4_1, ret4_2, tmp4_0, tmp4_1, tmp4_2);
	}

	_bncsve2_rts_sum128(svptrue_b32(), ret, ret4_0, ret4_1, ret4_2);

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t ret4[TSSIZE], mat4[TSSIZE], tmp4[TSSIZE];

	ret4[0] = vdupq_n_f32(0.0f);
	ret4[1] = vdupq_n_f32(0.0f);
	ret4[2] = vdupq_n_f32(0.0f);
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		mat4[0] = vld1q_f32(&mat->element[0][i]);
		mat4[1] = vld1q_f32(&mat->element[1][i]);
		mat4[2] = vld1q_f32(&mat->element[2][i]);

		// tmp4 := mat4[i]^2
		// ret4 += tmp4
		_bncneon_rts_mul(tmp4, mat4, mat4);
		_bncneon_rts_add(ret4, ret4, tmp4);
	}

	_bncneon_rts_sum128f(ret, ret4);

#else // others
	float mat1[TSSIZE];

	rts_set0(ret);
	for(i = 0; i < real_total_dim; i++)
	{
		mat1[0] = mat->element[0][i];
		mat1[1] = mat->element[1][i];
		mat1[2] = mat->element[2][i];

		// tmp := mat1[i]^2
		// ret += tmp
		rts_mul(tmp, mat1, mat1);
		rts_add(ret, ret, tmp);
	}

#endif // __AVX2__

	rts_sqrt(tmp, ret);
	rts_set(ret, tmp);

}

// print normf
void print_normf_tsmatrix(const char *str, TSMatrix mat)
{
	static float tmp[TSSIZE];

	normf_tsmatrix(tmp, mat);

	if(str != NULL)
		printf("%s(%ld, %ld)", str, mat->row_dim, mat->col_dim);

	rts_out_str(tmp); printf("\n");
}

/*************************************************/
/* Matrix Caluculations for TSMatrix            */
/*
void normf_tsmatrix(float ret[TSSIZE], TSMatrix mat)
void norm1_tsmatrix(float ret[TSSIZE], TSMatrix mat)
void normi_tsmatrix(float ret[TSSIZE], TSMatrix mat)
void add_tsmatrix(TSMatrix c, TSMatrix a, TSMatrix b);
void sub_tsmatrix(TSMatrix c, TSMatrix a, TSMatrix b);
void mul_tsmatrix(TSMatrix c, TSMatrix a, TSMatrix b);
void mul_tsmatrix_tsvec(TSVector v, TSMatrix a, TSVector vb)
void mul_tsmatrixt_tsvec(TSVector v, TSMatrix a, TSVector vb)
void transpose_tsmatrix(TSMatrix c, TSMatrix a);
void inv_tsmatrix(TSMatrix a);
void subst_mpfmatrux(TSMatrix c, TSMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_tsmatrix(float ret[TSSIZE], TSMatrix mat)
{
	long int i, j;
	float tmp[TSSIZE], sum[TSSIZE];

	set0_ts(ret);
	for(i = 0; i < mat->row_dim; i++)
	{
		set0_ts(sum);
		for(j = 0; j < mat->col_dim; j++)
		{
			rts_abs(tmp, get_tsmatrix_ij(mat, i, j));
			rts_add(sum, sum, tmp);
		}
		if(rts_cmp(ret, sum) < 0)
			rts_set(ret, sum);
	}

	return;
}

/* 1 Norm of Matrix */
void norm1_tsmatrix(float ret[TSSIZE], TSMatrix mat)
{
	long int i, j;
	float tmp[TSSIZE], sum[TSSIZE];

	rts_set_ui(ret, 0UL);

	for(j = 0; j < mat->col_dim; j++)
	{
		rts_set_ui(sum, 0UL);
		for(i = 0; i < mat->row_dim; i++)
		{
			rts_abs(tmp, get_tsmatrix_ij(mat, i, j));
			rts_add(sum, sum, tmp);
		}
		if(rts_cmp(ret, sum) < 0)
			rts_set(ret, sum);
	}

	return;
}

/* c := a + b */
void add_tsmatrix(TSMatrix c, TSMatrix a, TSMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_tsmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_tsmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 tmp4[TSSIZE], aij4[TSSIZE], bij4[TSSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = _mm256_load_ps(&(a->element[0][index]));
		aij4[1] = _mm256_load_ps(&(a->element[1][index]));
		aij4[2] = _mm256_load_ps(&(a->element[2][index]));
		bij4[0] = _mm256_load_ps(&(b->element[0][index]));
		bij4[1] = _mm256_load_ps(&(b->element[1][index]));
		bij4[2] = _mm256_load_ps(&(b->element[2][index]));

		_bncavx2_rts_add(tmp4, aij4, bij4);

		_mm256_store_ps(&(c->element[0][index]), tmp4[0]);
		_mm256_store_ps(&(c->element[1][index]), tmp4[1]); 
		_mm256_store_ps(&(c->element[2][index]), tmp4[2]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 tmp4[TSSIZE], aij4[TSSIZE], bij4[TSSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = _mm512_load_ps(&(a->element[0][index]));
		aij4[1] = _mm512_load_ps(&(a->element[1][index]));
		aij4[2] = _mm512_load_ps(&(a->element[2][index]));
		bij4[0] = _mm512_load_ps(&(b->element[0][index]));
		bij4[1] = _mm512_load_ps(&(b->element[1][index]));
		bij4[2] = _mm512_load_ps(&(b->element[2][index]));

		_bncavx512_rts_add(tmp4, aij4, bij4);

		_mm512_store_ps(&(c->element[0][index]), tmp4[0]);
		_mm512_store_ps(&(c->element[1][index]), tmp4[1]); 
		_mm512_store_ps(&(c->element[2][index]), tmp4[2]); 
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t tmp4_0, tmp4_1, tmp4_2;
	svfloat32_t aij4_0, aij4_1, aij4_2;
	svfloat32_t bij4_0, bij4_1, bij4_2;

	for(index = 0; index < real_total_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(real_total_dim));
		aij4_0 = svld1_f32(pg, &(a->element[0][index]));
		aij4_1 = svld1_f32(pg, &(a->element[1][index]));
		aij4_2 = svld1_f32(pg, &(a->element[2][index]));
		bij4_0 = svld1_f32(pg, &(b->element[0][index]));
		bij4_1 = svld1_f32(pg, &(b->element[1][index]));
		bij4_2 = svld1_f32(pg, &(b->element[2][index]));

		_bncsve2_rts_add(pg, &tmp4_0, &tmp4_1, &tmp4_2, aij4_0, aij4_1, aij4_2, bij4_0, bij4_1, bij4_2);

		svst1_f32(pg, &(c->element[0][index]), tmp4_0);
		svst1_f32(pg, &(c->element[1][index]), tmp4_1); 
		svst1_f32(pg, &(c->element[2][index]), tmp4_2); 
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t tmp4[TSSIZE], aij4[TSSIZE], bij4[TSSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = vld1q_f32(&(a->element[0][index]));
		aij4[1] = vld1q_f32(&(a->element[1][index]));
		aij4[2] = vld1q_f32(&(a->element[2][index]));
		bij4[0] = vld1q_f32(&(b->element[0][index]));
		bij4[1] = vld1q_f32(&(b->element[1][index]));
		bij4[2] = vld1q_f32(&(b->element[2][index]));

		_bncneon_rts_add(tmp4, aij4, bij4);

		vst1q_f32(&(c->element[0][index]), tmp4[0]);
		vst1q_f32(&(c->element[1][index]), tmp4[1]); 
		vst1q_f32(&(c->element[2][index]), tmp4[2]); 
	}
#else // others
	float tmp[TSSIZE], aij[TSSIZE], bij[TSSIZE];

	for(index = 0; index < real_total_dim; index++)
	{
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];
		aij[2] = a->element[2][index];
		bij[0] = b->element[0][index];
		bij[1] = b->element[1][index];
		bij[2] = b->element[2][index];

		rts_add(tmp, aij, bij);

		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1]; 
		c->element[2][index] = tmp[2]; 
	}
#endif // __AVX2__
/*
	float tmp[TSSIZE];

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rts_add(tmp, get_tsmatrix_ij(a, i, j), get_tsmatrix_ij(b, i, j));
			set_tsmatrix_ij(c, i, j, tmp);
		}
	}
*/
}

/* c := a - b */
void sub_tsmatrix(TSMatrix c, TSMatrix a, TSMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: sub_tsmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: sub_tsmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

/*
	float tmp[TSSIZE];
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rts_sub(tmp, get_tsmatrix_ij(a, i, j), get_tsmatrix_ij(b, i, j));
			set_tsmatrix_ij(c, i, j, tmp);
		}
	}
*/

// for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 tmp4[TSSIZE], aij4[TSSIZE], bij4[TSSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = _mm256_load_ps(&(a->element[0][index]));
		aij4[1] = _mm256_load_ps(&(a->element[1][index]));
		aij4[2] = _mm256_load_ps(&(a->element[2][index]));
		bij4[0] = _mm256_load_ps(&(b->element[0][index]));
		bij4[1] = _mm256_load_ps(&(b->element[1][index]));
		bij4[2] = _mm256_load_ps(&(b->element[2][index]));

		_bncavx2_rts_sub(tmp4, aij4, bij4);

		_mm256_store_ps(&(c->element[0][index]), tmp4[0]);
		_mm256_store_ps(&(c->element[1][index]), tmp4[1]); 
		_mm256_store_ps(&(c->element[2][index]), tmp4[2]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 tmp4[TSSIZE], aij4[TSSIZE], bij4[TSSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = _mm512_load_ps(&(a->element[0][index]));
		aij4[1] = _mm512_load_ps(&(a->element[1][index]));
		aij4[2] = _mm512_load_ps(&(a->element[2][index]));
		bij4[0] = _mm512_load_ps(&(b->element[0][index]));
		bij4[1] = _mm512_load_ps(&(b->element[1][index]));
		bij4[2] = _mm512_load_ps(&(b->element[2][index]));

		_bncavx512_rts_sub(tmp4, aij4, bij4);

		_mm512_store_ps(&(c->element[0][index]), tmp4[0]);
		_mm512_store_ps(&(c->element[1][index]), tmp4[1]); 
		_mm512_store_ps(&(c->element[2][index]), tmp4[2]); 
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t tmp4_0, tmp4_1, tmp4_2;
	svfloat32_t aij4_0, aij4_1, aij4_2;
	svfloat32_t bij4_0, bij4_1, bij4_2;

	for(index = 0; index < real_total_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(real_total_dim));
		aij4_0 = svld1_f32(pg, &(a->element[0][index]));
		aij4_1 = svld1_f32(pg, &(a->element[1][index]));
		aij4_2 = svld1_f32(pg, &(a->element[2][index]));
		bij4_0 = svld1_f32(pg, &(b->element[0][index]));
		bij4_1 = svld1_f32(pg, &(b->element[1][index]));
		bij4_2 = svld1_f32(pg, &(b->element[2][index]));

		_bncsve2_rts_neg(pg, &tmp4_0, &tmp4_1, &tmp4_2, bij4_0, bij4_1, bij4_2);
		_bncsve2_rts_add(pg, &tmp4_0, &tmp4_1, &tmp4_2, aij4_0, aij4_1, aij4_2, tmp4_0, tmp4_1, tmp4_2);

		svst1_f32(pg, &(c->element[0][index]), tmp4_0);
		svst1_f32(pg, &(c->element[1][index]), tmp4_1); 
		svst1_f32(pg, &(c->element[2][index]), tmp4_2); 
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t tmp4[TSSIZE], aij4[TSSIZE], bij4[TSSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = vld1q_f32(&(a->element[0][index]));
		aij4[1] = vld1q_f32(&(a->element[1][index]));
		aij4[2] = vld1q_f32(&(a->element[2][index]));
		bij4[0] = vld1q_f32(&(b->element[0][index]));
		bij4[1] = vld1q_f32(&(b->element[1][index]));
		bij4[2] = vld1q_f32(&(b->element[2][index]));

		_bncneon_rts_sub(tmp4, aij4, bij4);

		vst1q_f32(&(c->element[0][index]), tmp4[0]);
		vst1q_f32(&(c->element[1][index]), tmp4[1]); 
		vst1q_f32(&(c->element[2][index]), tmp4[2]); 
	}
#else // others
	float tmp[TSSIZE], aij[TSSIZE], bij[TSSIZE];

	for(index = 0; index < real_total_dim; index++)
	{
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];
		aij[2] = a->element[2][index];
		bij[0] = b->element[0][index];
		bij[1] = b->element[1][index];
		bij[2] = b->element[2][index];

		rts_sub(tmp, aij, bij);

		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1]; 
		c->element[2][index] = tmp[2]; 
	}
#endif // __AVX2__
}

/* c := sc * a */
void cmul_tsmatrix(TSMatrix c, float sc[TSSIZE], TSMatrix a)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_tsmatrix(row_dim is differnt! : c->(%ld, %ld), a->(%ld, %ld)\n", c->row_dim, c->col_dim, a->row_dim, a->col_dim);
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_tsmatrix(col_dim is differnt! : c->(%ld, %ld), a->(%ld, %ld)\n", c->row_dim, c->col_dim, a->row_dim, a->col_dim);
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

// for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 tmp4[TSSIZE], sc4[TSSIZE], aij4[TSSIZE];

	sc4[0] = _mm256_set1_ps(sc[0]);
	sc4[1] = _mm256_set1_ps(sc[1]);
	sc4[2] = _mm256_set1_ps(sc[2]);

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = _mm256_load_ps(&(a->element[0][index]));
		aij4[1] = _mm256_load_ps(&(a->element[1][index]));
		aij4[2] = _mm256_load_ps(&(a->element[2][index]));

		_bncavx2_rts_mul(tmp4, sc4, aij4);

		_mm256_store_ps(&(c->element[0][index]), tmp4[0]);
		_mm256_store_ps(&(c->element[1][index]), tmp4[1]); 
		_mm256_store_ps(&(c->element[2][index]), tmp4[2]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 tmp4[TSSIZE], sc4[TSSIZE], aij4[TSSIZE];

	sc4[0] = _mm512_set1_ps(sc[0]);
	sc4[1] = _mm512_set1_ps(sc[1]);
	sc4[2] = _mm512_set1_ps(sc[2]);

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = _mm512_load_ps(&(a->element[0][index]));
		aij4[1] = _mm512_load_ps(&(a->element[1][index]));
		aij4[2] = _mm512_load_ps(&(a->element[2][index]));

		_bncavx512_rts_mul(tmp4, sc4, aij4);

		_mm512_store_ps(&(c->element[0][index]), tmp4[0]);
		_mm512_store_ps(&(c->element[1][index]), tmp4[1]); 
		_mm512_store_ps(&(c->element[2][index]), tmp4[2]); 
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t tmp4_0, tmp4_1, tmp4_2;
	svfloat32_t sc4_0, sc4_1, sc4_2;
	svfloat32_t aij4_0, aij4_1, aij4_2;

	sc4_0 = svdup_f32(sc[0]);
	sc4_1 = svdup_f32(sc[1]);
	sc4_2 = svdup_f32(sc[2]);

	for(index = 0; index < real_total_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(real_total_dim));
		aij4_0 = svld1_f32(pg, &(a->element[0][index]));
		aij4_1 = svld1_f32(pg, &(a->element[1][index]));
		aij4_2 = svld1_f32(pg, &(a->element[2][index]));

		_bncsve2_rts_mul(pg, &tmp4_0, &tmp4_1, &tmp4_2, sc4_0, sc4_1, sc4_2, aij4_0, aij4_1, aij4_2);

		svst1_f32(pg, &(c->element[0][index]), tmp4_0);
		svst1_f32(pg, &(c->element[1][index]), tmp4_1); 
		svst1_f32(pg, &(c->element[2][index]), tmp4_2); 
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t tmp4[TSSIZE], sc4[TSSIZE], aij4[TSSIZE];

	sc4[0] = vdupq_n_f32(sc[0]);
	sc4[1] = vdupq_n_f32(sc[1]);
	sc4[2] = vdupq_n_f32(sc[2]);

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = vld1q_f32(&(a->element[0][index]));
		aij4[1] = vld1q_f32(&(a->element[1][index]));
		aij4[2] = vld1q_f32(&(a->element[2][index]));

		_bncneon_rts_mul(tmp4, sc4, aij4);

		vst1q_f32(&(c->element[0][index]), tmp4[0]);
		vst1q_f32(&(c->element[1][index]), tmp4[1]); 
		vst1q_f32(&(c->element[2][index]), tmp4[2]); 
	}
#else // others
	float tmp[TSSIZE], aij[TSSIZE];

	for(index = 0; index < real_total_dim; index++)
	{
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];
		aij[2] = a->element[2][index];

		rts_mul(tmp, sc, aij);

		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1]; 
		c->element[2][index] = tmp[2]; 
	}
#endif // __AVX2__
/*
	float tmp[TSSIZE];

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rts_mul(tmp, sc, get_tsmatrix_ij(a, i, j));
			set_tsmatrix_ij(c, i, j, tmp);
		}
	}
*/
}

/* c = a^T */
void transpose_tsmatrix(TSMatrix c, TSMatrix a)
{
	long int i, j, index;
	long int real_row_dim, real_col_dim;

	/* Check Dimentions */
	if((c->row_dim != a->col_dim) || (c->col_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: transpose_fmatrix\n");
		return;
	}

	real_row_dim = c->real_row_dim;
	real_col_dim = c->real_col_dim;
// AVX2
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 aji4[TSSIZE];

	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			//set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, j, i));
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
				a->element[1][(j + 7) * real_col_dim + i],
				a->element[1][(j + 6) * real_col_dim + i],
				a->element[1][(j + 5) * real_col_dim + i],
				a->element[1][(j + 4) * real_col_dim + i],
				a->element[2][(j + 3) * real_col_dim + i],
				a->element[2][(j + 2) * real_col_dim + i],
				a->element[2][(j + 1) * real_col_dim + i],
				a->element[2][(j    ) * real_col_dim + i]
			);
			index = i * real_col_dim + j;
			_mm256_store_ps(&(c->element[0][index]), aji4[0]);
			_mm256_store_ps(&(c->element[1][index]), aji4[1]);
			_mm256_store_ps(&(c->element[2][index]), aji4[2]);
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 aji4[TSSIZE];

	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			//set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, j, i));
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
			index = i * real_col_dim + j;
			_mm512_store_ps(&(c->element[0][index]), aji4[0]);
			_mm512_store_ps(&(c->element[1][index]), aji4[1]);
			_mm512_store_ps(&(c->element[2][index]), aji4[2]);
		}
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, gather)
    svfloat32_t aji_0, aji_1, aji_2;
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
            index = i * real_col_dim + j;
            svst1_f32(pg, &(c->element[0][index]), aji_0);
            svst1_f32(pg, &(c->element[1][index]), aji_1);
            svst1_f32(pg, &(c->element[2][index]), aji_2);
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
        }
#else // others
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_tsmatrix_ij(c, i, j, get_tsmatrix_ij(a, j, i));
	}
#endif // AVX2
}

/* c := a */
void subst_tsmatrix(TSMatrix c, TSMatrix a)
{
	long int i, j, index;
	long int real_row_dim, real_col_dim;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_tsmatrix\n");
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
			//set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, i, j));
			index = i * real_col_dim + j;
			_mm256_store_ps(&(c->element[0][index]), _mm256_load_ps(&(a->element[0][index])));
			_mm256_store_ps(&(c->element[1][index]), _mm256_load_ps(&(a->element[1][index])));
			_mm256_store_ps(&(c->element[2][index]), _mm256_load_ps(&(a->element[2][index])));
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			//set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, i, j));
			index = i * real_col_dim + j;
			_mm512_store_ps(&(c->element[0][index]), _mm512_load_ps(&(a->element[0][index])));
			_mm512_store_ps(&(c->element[1][index]), _mm512_load_ps(&(a->element[1][index])));
			_mm512_store_ps(&(c->element[2][index]), _mm512_load_ps(&(a->element[2][index])));
		}
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += (long int)svcntw())
		{
		svbool_t pg = svwhilelt_b32_s64((int64_t)j, (int64_t)(real_col_dim));
			//set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, i, j));
			index = i * real_col_dim + j;
			svst1_f32(pg, &(c->element[0][index]), svld1_f32(pg, &(a->element[0][index])));
			svst1_f32(pg, &(c->element[1][index]), svld1_f32(pg, &(a->element[1][index])));
			svst1_f32(pg, &(c->element[2][index]), svld1_f32(pg, &(a->element[2][index])));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			//set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, i, j));
			index = i * real_col_dim + j;
			vst1q_f32(&(c->element[0][index]), vld1q_f32(&(a->element[0][index])));
			vst1q_f32(&(c->element[1][index]), vld1q_f32(&(a->element[1][index])));
			vst1q_f32(&(c->element[2][index]), vld1q_f32(&(a->element[2][index])));
		}
	}
#else // others
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_tsmatrix_ij(c, i, j, get_tsmatrix_ij(a, i, j));
		}
	}
#endif // AVX2
}

/* c := I */
void setI_tsmatrix(TSMatrix c)
{
	long int i, j;
	long int real_total_dim;
	float tmp1[TSSIZE];

	real_total_dim = c->real_row_dim * c->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 zero4;

	zero4 = _mm256_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm256_store_ps(&c->element[0][i], zero4);
		_mm256_store_ps(&c->element[1][i], zero4);
		_mm256_store_ps(&c->element[2][i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero4;

	zero4 = _mm512_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm512_store_ps(&c->element[0][i], zero4);
		_mm512_store_ps(&c->element[1][i], zero4);
		_mm512_store_ps(&c->element[2][i], zero4);
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
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t zero4;

	zero4 = vdupq_n_f32(0.0f);
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		vst1q_f32(&c->element[0][i], zero4);
		vst1q_f32(&c->element[1][i], zero4);
		vst1q_f32(&c->element[2][i], zero4);
	}
#else // others
	for(i = 0; i < real_total_dim; i++)
	{
		c->element[0][i] = 0.0f;
		c->element[1][i] = 0.0f;	
		c->element[2][i] = 0.0f;	
	}
#endif // __AVX2__

	rts_set_ui(tmp1, 1UL);

	for(i = 0; i < c->row_dim; i++)
	{
		if(i < c->col_dim)
			set_tsmatrix_ij(c, i, i, tmp1);
	}
}

/* v := a * vb */
void mul_tsmatrix_tsvec(TSVector v, TSMatrix a, TSVector vb)
{
	long int i, j;
	float tmp[TSSIZE], tmp1[TSSIZE];

	/* Check Dimension */
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: mul_tsmatrix_tsvec\n");
		return;
	}

// SIMD
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int ij_index, real_col_dim;
	__m256 tmp4[TSSIZE], tmp1_4[TSSIZE];
	__m256 aij4[TSSIZE], vbj4[TSSIZE];

	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->row_dim; i++)
	{
		//rts_set_ui(tmp, 0UL);
		_bncavx2_set0_ts(tmp4);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij4[0] = _mm256_load_ps(&(a->element[0][ij_index]));
			aij4[1] = _mm256_load_ps(&(a->element[1][ij_index]));
			aij4[2] = _mm256_load_ps(&(a->element[2][ij_index]));
			vbj4[0] = _mm256_load_ps(&(vb->element[0][j]));
			vbj4[1] = _mm256_load_ps(&(vb->element[1][j]));
			vbj4[2] = _mm256_load_ps(&(vb->element[2][j]));

			//rts_mul(tmp1, get_tsmatrix_ij(a, i, j), get_tsvector_i(vb, j));
			//rts_add(tmp, tmp, tmp1);
			_bncavx2_rts_mul(tmp1_4, aij4, vbj4);
			_bncavx2_rts_add(tmp4, tmp4, tmp1_4);
		}
		//set_tsvector_i(v, i, tmp);
		_bncavx2_rts_sum256(tmp, tmp4);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
		v->element[2][i] = tmp[2];
	}

#elif defined(__AVX512F__) // __AVX512F__
	long int ij_index, real_col_dim;
	__m512 tmp4[TSSIZE], tmp1_4[TSSIZE];
	__m512 aij4[TSSIZE], vbj4[TSSIZE];

	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->row_dim; i++)
	{
		//rts_set_ui(tmp, 0UL);
		_bncavx512_set0_ts(tmp4);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij4[0] = _mm512_load_ps(&(a->element[0][ij_index]));
			aij4[1] = _mm512_load_ps(&(a->element[1][ij_index]));
			aij4[2] = _mm512_load_ps(&(a->element[2][ij_index]));
			vbj4[0] = _mm512_load_ps(&(vb->element[0][j]));
			vbj4[1] = _mm512_load_ps(&(vb->element[1][j]));
			vbj4[2] = _mm512_load_ps(&(vb->element[2][j]));

			//rts_mul(tmp1, get_tsmatrix_ij(a, i, j), get_tsvector_i(vb, j));
			//rts_add(tmp, tmp, tmp1);
			_bncavx512_rts_mul(tmp1_4, aij4, vbj4);
			_bncavx512_rts_add(tmp4, tmp4, tmp1_4);
		}
		//set_tsvector_i(v, i, tmp);
		_bncavx512_rts_sum512(tmp, tmp4);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
		v->element[2][i] = tmp[2];
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic)
	{
		long real_col_dim = a->real_col_dim;
		long vl = (long)svcntw();
		for(i = 0; i < a->row_dim; i++)
		{
			svfloat32_t acc0, acc1, acc2;
			_bncsve2_rts_set0(&acc0, &acc1, &acc2);
			for(j = 0; j < real_col_dim; j += vl)
			{
				svbool_t pg = svwhilelt_b32_s32((int32_t)j, (int32_t)real_col_dim);
				long ij = i * real_col_dim + j;
				svfloat32_t a0 = svld1_f32(pg, &(a->element[0][ij]));
				svfloat32_t a1 = svld1_f32(pg, &(a->element[1][ij]));
				svfloat32_t a2v= svld1_f32(pg, &(a->element[2][ij]));
				svfloat32_t b0 = svld1_f32(pg, &(vb->element[0][j]));
				svfloat32_t b1 = svld1_f32(pg, &(vb->element[1][j]));
				svfloat32_t b2 = svld1_f32(pg, &(vb->element[2][j]));
				svfloat32_t t0, t1, t2;
				_bncsve2_rts_mul(pg, &t0, &t1, &t2, a0, a1, a2v, b0, b1, b2);
				_bncsve2_rts_add(pg, &acc0, &acc1, &acc2, acc0, acc1, acc2, t0, t1, t2);
			}
			{
				long _L, _vl = (long)svcntw();
				float _la0[64], _la1[64], _la2[64];
				svst1_f32(svptrue_b32(), _la0, acc0);
				svst1_f32(svptrue_b32(), _la1, acc1);
				svst1_f32(svptrue_b32(), _la2, acc2);
				rts_set_ui(tmp, 0UL);
				for(_L = 0; _L < _vl; _L++)
				{
					float _lane[TSSIZE] = { _la0[_L], _la1[_L], _la2[_L] };
					rts_add(tmp, tmp, _lane);
				}
			}
			v->element[0][i] = tmp[0];
			v->element[1][i] = tmp[1];
			v->element[2][i] = tmp[2];
		}
	}

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	{
		long int ij_index, real_col_dim = a->real_col_dim;
		float32x4_t tmp4[TSSIZE], tmp1_4[TSSIZE], aij4[TSSIZE], vbj4[TSSIZE];
		for(i = 0; i < a->row_dim; i++)
		{
			_bncneon_set0_ts(tmp4);
			for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
			{
				ij_index = i * real_col_dim + j;
				aij4[0] = vld1q_f32(&(a->element[0][ij_index]));
				aij4[1] = vld1q_f32(&(a->element[1][ij_index]));
				aij4[2] = vld1q_f32(&(a->element[2][ij_index]));
				vbj4[0] = vld1q_f32(&(vb->element[0][j]));
				vbj4[1] = vld1q_f32(&(vb->element[1][j]));
				vbj4[2] = vld1q_f32(&(vb->element[2][j]));
				_bncneon_rts_mul(tmp1_4, aij4, vbj4);
				_bncneon_rts_add(tmp4, tmp4, tmp1_4);
			}
			{
				float _l0[TSSIZE]={vgetq_lane_f32(tmp4[0],0),vgetq_lane_f32(tmp4[1],0),vgetq_lane_f32(tmp4[2],0)};
				float _l1[TSSIZE]={vgetq_lane_f32(tmp4[0],1),vgetq_lane_f32(tmp4[1],1),vgetq_lane_f32(tmp4[2],1)};
				float _l2[TSSIZE]={vgetq_lane_f32(tmp4[0],2),vgetq_lane_f32(tmp4[1],2),vgetq_lane_f32(tmp4[2],2)};
				float _l3[TSSIZE]={vgetq_lane_f32(tmp4[0],3),vgetq_lane_f32(tmp4[1],3),vgetq_lane_f32(tmp4[2],3)};
				rts_set(tmp,_l0); rts_add(tmp,tmp,_l1); rts_add(tmp,tmp,_l2); rts_add(tmp,tmp,_l3);
			}
			v->element[0][i] = tmp[0];
			v->element[1][i] = tmp[1];
			v->element[2][i] = tmp[2];
		}
	}

#else // others

	for(i = 0; i < a->row_dim; i++)
	{
		rts_set_ui(tmp, 0UL);
		for(j = 0; j < a->col_dim; j++)
		{
			rts_mul(tmp1, get_tsmatrix_ij(a, i, j), get_tsvector_i(vb, j));
			rts_add(tmp, tmp, tmp1);
		}
		set_tsvector_i(v, i, tmp);
	}
#endif // __AVX2__
}

/* v := a^T * vb */
void mul_tsmatrixt_tsvec(TSVector v, TSMatrix a, TSVector vb)
{
	long int i, j;
	float tmp[TSSIZE], tmp1[TSSIZE];

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_tsmatrixt_tsvec\n");
		return;
	}
// SIMD
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int ji_index, real_row_dim, real_col_dim;
	__m256 tmp4[TSSIZE], tmp1_4[TSSIZE];
	__m256 aij4[TSSIZE], vbj4[TSSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->col_dim; i++)
	{
		//rts_set_ui(tmp, 0UL);
		_bncavx2_set0_ts(tmp4);
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
			vbj4[0] = _mm256_load_ps(&(vb->element[0][j]));
			vbj4[1] = _mm256_load_ps(&(vb->element[1][j]));
			vbj4[2] = _mm256_load_ps(&(vb->element[2][j]));

			//rts_mul(tmp1, get_tsmatrix_ij(a, i, j), get_tsvector_i(vb, j));
			//rts_add(tmp, tmp, tmp1);
			_bncavx2_rts_mul(tmp1_4, aij4, vbj4);
			_bncavx2_rts_add(tmp4, tmp4, tmp1_4);
		}
		//set_tsvector_i(v, i, tmp);
		_bncavx2_rts_sum256(tmp, tmp4);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
		v->element[2][i] = tmp[2];
	}

#elif defined(__AVX512F__) // __AVX512F__
	long int ji_index, real_row_dim, real_col_dim;
	__m512 tmp4[TSSIZE], tmp1_4[TSSIZE];
	__m512 aij4[TSSIZE], vbj4[TSSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->col_dim; i++)
	{
		//rts_set_ui(tmp, 0UL);
		_bncavx512_set0_ts(tmp4);
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
			vbj4[0] = _mm512_load_ps(&(vb->element[0][j]));
			vbj4[1] = _mm512_load_ps(&(vb->element[1][j]));
			vbj4[2] = _mm512_load_ps(&(vb->element[2][j]));

			//rts_mul(tmp1, get_tsmatrix_ij(a, i, j), get_tsvector_i(vb, j));
			//rts_add(tmp, tmp, tmp1);
			_bncavx512_rts_mul(tmp1_4, aij4, vbj4);
			_bncavx512_rts_add(tmp4, tmp4, tmp1_4);
		}
		//set_tsvector_i(v, i, tmp);
		_bncavx512_rts_sum512(tmp, tmp4);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
		v->element[2][i] = tmp[2];
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic)
	{
		long real_col_dim = a->real_col_dim, real_row_dim = a->real_row_dim;
		long vl = (long)svcntw();
		for(i = 0; i < a->col_dim; i++)
		{
			svfloat32_t acc0, acc1, acc2;
			_bncsve2_rts_set0(&acc0, &acc1, &acc2);
			for(j = 0; j < real_row_dim; j += vl)
			{
				svbool_t pg = svwhilelt_b32_s32((int32_t)j, (int32_t)real_row_dim);
				svint32_t idx = svindex_s32((int32_t)(j * real_col_dim + i), (int32_t)real_col_dim);
				svfloat32_t a0 = svld1_gather_s32index_f32(pg, a->element[0], idx);
				svfloat32_t a1 = svld1_gather_s32index_f32(pg, a->element[1], idx);
				svfloat32_t a2v= svld1_gather_s32index_f32(pg, a->element[2], idx);
				svfloat32_t b0 = svld1_f32(pg, &(vb->element[0][j]));
				svfloat32_t b1 = svld1_f32(pg, &(vb->element[1][j]));
				svfloat32_t b2 = svld1_f32(pg, &(vb->element[2][j]));
				svfloat32_t t0, t1, t2;
				_bncsve2_rts_mul(pg, &t0, &t1, &t2, a0, a1, a2v, b0, b1, b2);
				_bncsve2_rts_add(pg, &acc0, &acc1, &acc2, acc0, acc1, acc2, t0, t1, t2);
			}
			{
				long _L, _vl = (long)svcntw();
				float _la0[64], _la1[64], _la2[64];
				svst1_f32(svptrue_b32(), _la0, acc0);
				svst1_f32(svptrue_b32(), _la1, acc1);
				svst1_f32(svptrue_b32(), _la2, acc2);
				rts_set_ui(tmp, 0UL);
				for(_L = 0; _L < _vl; _L++)
				{
					float _lane[TSSIZE] = { _la0[_L], _la1[_L], _la2[_L] };
					rts_add(tmp, tmp, _lane);
				}
			}
			v->element[0][i] = tmp[0];
			v->element[1][i] = tmp[1];
			v->element[2][i] = tmp[2];
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	{
		long real_col_dim = a->real_col_dim, real_row_dim = a->real_row_dim;
		float32x4_t tmp4[TSSIZE], tmp1_4[TSSIZE], aij4[TSSIZE], vbj4[TSSIZE];
		for(i = 0; i < a->col_dim; i++)
		{
			_bncneon_set0_ts(tmp4);
			for(j = 0; j < real_row_dim; j += _BNC_S_WIDTH)
			{
				aij4[0] = (float32x4_t){ a->element[0][(j)*real_col_dim+i], a->element[0][(j+1)*real_col_dim+i], a->element[0][(j+2)*real_col_dim+i], a->element[0][(j+3)*real_col_dim+i] };
				aij4[1] = (float32x4_t){ a->element[1][(j)*real_col_dim+i], a->element[1][(j+1)*real_col_dim+i], a->element[1][(j+2)*real_col_dim+i], a->element[1][(j+3)*real_col_dim+i] };
				aij4[2] = (float32x4_t){ a->element[2][(j)*real_col_dim+i], a->element[2][(j+1)*real_col_dim+i], a->element[2][(j+2)*real_col_dim+i], a->element[2][(j+3)*real_col_dim+i] };
				vbj4[0] = vld1q_f32(&(vb->element[0][j]));
				vbj4[1] = vld1q_f32(&(vb->element[1][j]));
				vbj4[2] = vld1q_f32(&(vb->element[2][j]));
				_bncneon_rts_mul(tmp1_4, aij4, vbj4);
				_bncneon_rts_add(tmp4, tmp4, tmp1_4);
			}
			{
				float _l0[TSSIZE]={vgetq_lane_f32(tmp4[0],0),vgetq_lane_f32(tmp4[1],0),vgetq_lane_f32(tmp4[2],0)};
				float _l1[TSSIZE]={vgetq_lane_f32(tmp4[0],1),vgetq_lane_f32(tmp4[1],1),vgetq_lane_f32(tmp4[2],1)};
				float _l2[TSSIZE]={vgetq_lane_f32(tmp4[0],2),vgetq_lane_f32(tmp4[1],2),vgetq_lane_f32(tmp4[2],2)};
				float _l3[TSSIZE]={vgetq_lane_f32(tmp4[0],3),vgetq_lane_f32(tmp4[1],3),vgetq_lane_f32(tmp4[2],3)};
				rts_set(tmp,_l0); rts_add(tmp,tmp,_l1); rts_add(tmp,tmp,_l2); rts_add(tmp,tmp,_l3);
			}
			v->element[0][i] = tmp[0];
			v->element[1][i] = tmp[1];
			v->element[2][i] = tmp[2];
		}
	}
#else // others
	for(i = 0; i < a->col_dim; i++)
	{
		set0_ts(tmp);
		for(j = 0; j < a->row_dim; j++)
		{
			rts_mul(tmp1, get_tsmatrix_ij(a, j, i), get_tsvector_i(vb, j));
			rts_add(tmp, tmp, tmp1);
		}
		set_tsvector_i(v, i, tmp);
	}
#endif // __AVX2__
}

/* a = a^(-1) */
/* square matrix only */
void inv_tsmatrix(TSMatrix a)
{
	long int i, j, k, dim;
	float tmp[TSSIZE], aii[TSSIZE];

	/* Check Dimensions */
	if(a->row_dim != a->col_dim)
	{
		fprintf(stderr, "ERROR: inv_tsmatrix\n");
		return;
	}

	dim = a->row_dim;

	for(i = 0; i < dim; i++)
	{
		if(rts_cmp_ui(get_tsmatrix_ij(a, i, i), 0UL) == 0) 
		{
			fprintf(stderr, "ERROR: inv_tsmatrix: Pivot(%ld,%ld) is zero.\n", i, i);
			return;
		}

		rts_ui_div(aii, 1UL, get_tsmatrix_ij(a, i, i));
		set_tsmatrix_ij(a, i, i, aii);

		for(j = 0; j < i; j++)
		{
			rts_mul(tmp, get_tsmatrix_ij(a, i, j), aii);
			set_tsmatrix_ij(a, i, j, tmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			rts_mul(tmp, get_tsmatrix_ij(a, i, j), aii);
			set_tsmatrix_ij(a, i, j, tmp);
		}

		for(j = 0; j < i; j++)
		{
			for(k = 0; k < i; k++)
			{
				rts_mul(tmp, get_tsmatrix_ij(a, j, i), get_tsmatrix_ij(a, i, k));
				rts_sub(tmp, get_tsmatrix_ij(a, j, k), tmp);
				set_tsmatrix_ij(a, j, k, tmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				rts_mul(tmp, get_tsmatrix_ij(a, j, i), get_tsmatrix_ij(a, i, k));
				rts_sub(tmp, get_tsmatrix_ij(a, j, k), tmp);
				set_tsmatrix_ij(a, j, k, tmp);
			}
		}

		for(j = i + 1; j < dim; j++)
		{
			for(k = 0; k < i; k++)
			{
				rts_mul(tmp, get_tsmatrix_ij(a, j, i), get_tsmatrix_ij(a, i, k));
				rts_sub(tmp, get_tsmatrix_ij(a, j, k), tmp);
				set_tsmatrix_ij(a, j, k, tmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				rts_mul(tmp, get_tsmatrix_ij(a, j, i), get_tsmatrix_ij(a, i, k));
				rts_sub(tmp, get_tsmatrix_ij(a, j, k), tmp);
				set_tsmatrix_ij(a, j, k, tmp);
			}
		}

		for(j = 0; j < i; j++)
		{
			rts_neg(tmp, aii); /* tmp := -aii */
			rts_mul(tmp, tmp, get_tsmatrix_ij(a, j, i));
			set_tsmatrix_ij(a, j, i, tmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			rts_neg(tmp, aii); /* tmp := -aii */
			rts_mul(tmp, tmp, get_tsmatrix_ij(a, j, i));
			set_tsmatrix_ij(a, j, i, tmp);
		}
	}
}

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpf)a */
void subst_mpfvector_tsvec(MPFVector c, TSVector a)
{
	long int i;
	mpf_t tmp;

	mpf_init2(tmp, c->prec);

	for(i = 0; i < a->dim; i++)
	{
		mpf_set_ts(tmp, get_tsvector_i(a, i));
		set_mpfvector_i(c, i, tmp);
	}

	mpf_clear(tmp);

}

/* c := (dd)a */
void subst_tsvector_mpfvec(TSVector c, MPFVector a)
{
	long int i;
	float tmp[TSSIZE];

	for(i = 0; i < a->dim; i++)
	{
		mpf_get_ts(tmp, get_mpfvector_i(a, i));
		set_tsvector_i(c, i, tmp);
	}

}
/* c := (mpf)a */
void subst_mpfmatrix_tsmat(MPFMatrix c, TSMatrix a)
{
	long int i, j;
	mpf_t tmp;

	mpf_init2(tmp, c->prec);

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_mpfmatrix_ddmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			mpf_set_ts(tmp, get_tsmatrix_ij(a, i, j));
			set_mpfmatrix_ij(c, i, j, tmp);
		}
	}

	mpf_clear(tmp);
}

/* c := (dd)a */
void subst_tsmatrix_mpfmat(TSMatrix c, MPFMatrix a)
{
	long int i, j;
	float tmp[TSSIZE];

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_tsmatrix_mpfmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			mpf_get_ts(tmp, get_mpfmatrix_ij(a, i, j));
			set_tsmatrix_ij(c, i, j, tmp);
		}
	}
}

/* Normwise relative error of vector */
void relerr_tsvector_mpfvec(float relerr[TSSIZE], TSVector approx_vec, MPFVector true_vec, int norm_type)
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
	subst_mpfvector_tsvec(mpf_approx_vec, approx_vec);

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
	//mpf_get_ts(relerr, mpf_relerr);
	mpf_get_ts(relerr, mpf_relerr);

	free_mpfvector(diff_vec);
	free_mpfvector(mpf_approx_vec);
	mpf_clear(norm_diff_vec);
	mpf_clear(norm_true_vec);
	mpf_clear(mpf_relerr);

	return;
}

/* Elementwise relative errors of vector */
void relerr_element_tsvector_mpf(float max_relerr[TSSIZE], float min_relerr[TSSIZE], float norm_relerr[TSSIZE], TSVector approx_vec, MPFVector true_vec, int norm_type)
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
	subst_mpfvector_tsvec(mpf_approx_vec, approx_vec);

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
	mpf_get_ts(max_relerr, mpf_max_relerr);
	mpf_get_ts(min_relerr, mpf_min_relerr);
	mpf_get_ts(norm_relerr, mpf_norm_relerr);

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

/* c := (dd)a */
void subst_tsvector_fvec(TSVector c, FVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
	{
		set_tsvector_i_f(c, i, get_fvector_i(a, i));
	}
}

/* c := (d)a */
void subst_fvector_tsvec(FVector c, TSVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
	{
		c->element[i] = rts_get_f(get_tsvector_i(a, i));
	}
}


/* c := (dd)a */
void subst_tsmatrix_fmat(TSMatrix c, FMatrix a)
{
	long int i, j;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_tsmatrix_fmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_tsmatrix_ij_f(c, i, j, get_fmatrix_ij(a, i, j));
		}
	}
}

/* c := (d)a */
void subst_fmatrix_tsmat(FMatrix c, TSMatrix a)
{
	long int i, j, ij_index;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_fmatrix_tsmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			ij_index = i * (c->col_dim) + j;
			c->element[ij_index] = rts_get_f(get_tsmatrix_ij(a, i, j));
		}
	}
}

/* Normwise relative error of vector */
void relerr_tsvector(float relerr[TSSIZE], TSVector approx_vec, TSVector true_vec, int norm_type)
{
	float norm_true_vec[TSSIZE], norm_diff_vec[TSSIZE];
	TSVector diff_vec;

	diff_vec = init_tsvector(approx_vec->dim);

	// diff_vec := approx_vec - true_vec
	sub_tsvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_tsvector(norm_diff_vec, diff_vec);
			normi_tsvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_tsvector(norm_diff_vec, diff_vec);
			norm1_tsvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_tsvector(norm_diff_vec, diff_vec);
			norm2_tsvector(norm_true_vec, true_vec);
			break;
	}

	if(rts_cmp_ui(norm_true_vec, 0UL) != 0)
		rts_div(relerr, norm_diff_vec, norm_true_vec);

	free_tsvector(diff_vec);

	return;
}

/* Elementwise relative errors of vector */
void relerr_element_tsvector(float max_relerr[TSSIZE], float min_relerr[TSSIZE], float norm_relerr[TSSIZE], TSVector approx_vec, TSVector true_vec, int norm_type)
{
	float abs_true_vec[TSSIZE], abs_diff_vec[TSSIZE], norm_diff_vec[TSSIZE], norm_true_vec[TSSIZE];
	long int i;
	TSVector diff_vec;

	diff_vec = init_tsvector(approx_vec->dim);

	// diff_vec := approx_vec - true_vec
	sub_tsvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_tsvector(norm_diff_vec, diff_vec);
			normi_tsvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_tsvector(norm_diff_vec, diff_vec);
			norm1_tsvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_tsvector(norm_diff_vec, diff_vec);
			norm2_tsvector(norm_true_vec, true_vec);
			break;
	}

	rts_set(norm_relerr, norm_diff_vec);
	if(rts_cmp_ui(norm_true_vec, 0UL) != 0)
		rts_div(norm_relerr, norm_diff_vec, norm_true_vec);

	// relative errors of each elements
	rts_set_ui(max_relerr, 0UL);
	normi_tsvector(min_relerr, diff_vec);
	for(i = 0; i < approx_vec->dim; i++)
	{
		rts_abs(abs_diff_vec, get_tsvector_i(diff_vec, i));
		rts_abs(abs_true_vec, get_tsvector_i(true_vec, i));
		if(rts_cmp_ui(abs_true_vec, 0UL) != 0)
			rts_div(abs_diff_vec, abs_diff_vec, abs_true_vec);
		
		if(rts_cmp(max_relerr, abs_diff_vec) < 0)
			rts_set(max_relerr, abs_diff_vec);
		if(rts_cmp(min_relerr, abs_diff_vec) > 0)
			rts_set(min_relerr, abs_diff_vec);
	}

	free_tsvector(diff_vec);// Fix! 2012-06-03 by T.Kouya

	return;
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_tsmatrix(TSMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
	long int i, j, true_end;
	float tmp[TSSIZE];

	true_end = (col_end > mat->col_dim) ? mat->col_dim : col_end;

	for(i = col_start; i < true_end; i++)
	{
		rts_set(tmp, get_tsmatrix_ij(mat, row_index0, i));
		set_tsmatrix_ij(mat, row_index0, i, get_tsmatrix_ij(mat, row_index1, i));
		set_tsmatrix_ij(mat, row_index1, i, tmp);
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
int TSLUdecomp(TSMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       TSMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	static float dtmp[TSSIZE], dtmp1[TSSIZE], dmaxii[TSSIZE];
#ifdef BNC_USE_NEW_FMA
	static float neg_aji[TSSIZE];
#endif // BNC_USE_NEW_FMA

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
	{
		rts_abs(dmaxii, get_tsmatrix_ij(a, i, i));
		//printf("a%ld_%ld = ", i, i); rts_out_str(dmaxii); printf("\n");
		if(rts_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (TSLUdecomp)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rts_div(dtmp, get_tsmatrix_ij(a, j, i), get_tsmatrix_ij(a, i, i));
			set_tsmatrix_ij(a, j, i, dtmp);
			//printf("a%ld_%ld = ", j, i); rts_out_str(dtmp); printf("\n");
		}
		for(j = (i + 1); j < dim; j++)
		{
#ifdef BNC_USE_NEW_FMA
			rts_neg(neg_aji, get_tsmatrix_ij(a, j, i));
#endif // BNC_USE_NEW_FMA
			for(k = (i + 1); k < dim; k++)
			{
#ifdef BNC_USE_NEW_FMA
				rts_fma(dtmp, neg_aji, get_tsmatrix_ij(a, i, k), get_tsmatrix_ij(a, j, k));
#else // BNC_USE_NEW_FMA
				rts_mul(dtmp1, get_tsmatrix_ij(a, j, i), get_tsmatrix_ij(a, i, k));
				rts_sub(dtmp, get_tsmatrix_ij(a, j, k), dtmp1);
#endif // BNC_USE_NEW_FMA
				set_tsmatrix_ij(a, j, k, dtmp);
				//printf("a%ld_%ld= ", j, k); rts_out_str(dtmp); printf("\n");
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
int SolveTSLS(TSVector answer, TSMatrix lu, TSVector b)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      TSMatrix lu: LU decomposed Matrix (given by user)   */
/*      TSVector b: constant vector (given by user)         */
/*      TSVector answer: Solution for linear system         */
/*      long int dim: Dimension of Matrix (given by user)   */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static float dtmp[TSSIZE], dtmp1[TSSIZE];

	dim = answer->dim;

	subst_tsvector(answer, b);

#ifdef BNC_USE_NEW_FMA
/* fused branch-free FMA + SIMD forward/backward substitution */
	for(i = 0; i < dim; i++)
	{
		rts_abs(dtmp, get_tsmatrix_ij(lu, i, i));
		if(rts_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveTSLS, %ld)\n", i);
			return -1;
		}
	}

/* Forward (row-oriented) */
	for(i = 1; i < dim; i++)
	{
		_bnc_tssolve_dot(dtmp1, lu, i, 0, i, answer);
		rts_sub(dtmp, get_tsvector_i(answer, i), dtmp1);
		set_tsvector_i(answer, i, dtmp);
	}

/* Backward (row-oriented) */
	for(i = (dim - 1); i >= 0; i--)
	{
		_bnc_tssolve_dot(dtmp1, lu, i, i + 1, dim, answer);
		rts_sub(dtmp, get_tsvector_i(answer, i), dtmp1);
		rts_div(dtmp, dtmp, get_tsmatrix_ij(lu, i, i));
		set_tsvector_i(answer, i, dtmp);
	}
#else // BNC_USE_NEW_FMA
/* Forward */
	for(i = 0; i < dim; i++)
	{
		//printf("f %ld = ", i); rts_out_str(get_tsvector_i(answer, i)); printf("\n");
		rts_abs(dtmp, get_tsmatrix_ij(lu, i, i));
		if(rts_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveTSLS, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rts_mul(dtmp1, get_tsmatrix_ij(lu, j, i), get_tsvector_i(answer, i));
			rts_sub(dtmp, get_tsvector_i(answer, j), dtmp1);
			set_tsvector_i(answer, j, dtmp);
		}
		//printf("f %ld = ", i); rts_out_str(get_tsvector_i(answer, i)); printf("\n");
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rts_mul(dtmp1, get_tsmatrix_ij(lu, i, j), get_tsvector_i(answer, j));
			rts_sub(dtmp, get_tsvector_i(answer, i), dtmp1);
			set_tsvector_i(answer, i, dtmp);
		}
		rts_div(dtmp, get_tsvector_i(answer, i), get_tsmatrix_ij(lu, i, i));
		set_tsvector_i(answer, i, dtmp);
		//printf("b %ld = ", i); rts_out_str(get_tsvector_i(answer, i)); printf("\n");
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
int TSLUdecompP(TSMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       TSMatrix a: Matrix (given by user)                 */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
	static float dtmp[TSSIZE], dtmp1[TSSIZE], dmaxii[TSSIZE];
#ifdef BNC_USE_NEW_FMA
	static float neg_aji[TSSIZE];
#endif // BNC_USE_NEW_FMA

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		rts_abs(dmaxii, get_tsmatrix_ij(a, ch[i], i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rts_abs(dtmp, get_tsmatrix_ij(a, ch[j], i));
			if(rts_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rts_set(dmaxii, dtmp);
			}
		}

		if(rts_cmp_ui(dmaxii, 0UL) == 0)
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
			rts_div(dtmp, get_tsmatrix_ij(a, ch[j], i), get_tsmatrix_ij(a, ch[i], i));
			set_tsmatrix_ij(a, ch[j], i, dtmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
#ifdef BNC_USE_NEW_FMA
			rts_neg(neg_aji, get_tsmatrix_ij(a, ch[j], i));
#endif // BNC_USE_NEW_FMA
			for(k = (i + 1); k < dim; k++)
			{
#ifdef BNC_USE_NEW_FMA
				rts_fma(dtmp, neg_aji, get_tsmatrix_ij(a, ch[i], k), get_tsmatrix_ij(a, ch[j], k));
#else // BNC_USE_NEW_FMA
				rts_mul(dtmp1, get_tsmatrix_ij(a, ch[j], i), get_tsmatrix_ij(a, ch[i], k));
				rts_sub(dtmp, get_tsmatrix_ij(a, ch[j], k), dtmp1);
#endif // BNC_USE_NEW_FMA
				set_tsmatrix_ij(a, ch[j], k, dtmp);
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
int SolveTSLSP(TSVector answer, TSMatrix lu, TSVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      TSMatrix lu[]: LU decomposed Matrix (given by user) */
/*      TSVector b[]: constant vector (given by user)       */
/*      TSVector answer[]: Solution for linear system       */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static float dtmp[TSSIZE], dtmp1[TSSIZE];

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_tsvector_i(answer, i, get_tsvector_i(b, ch[i]));

	
#ifdef BNC_USE_NEW_FMA
/* fused branch-free FMA + SIMD forward/backward substitution */
	for(i = 0; i < dim; i++)
	{
		rts_abs(dtmp, get_tsmatrix_ij(lu, ch[i], i));
		if(rts_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveTSLSP, %ld)\n", i);
			return -1;
		}
	}

/* Forward (row-oriented) */
	for(i = 1; i < dim; i++)
	{
		_bnc_tssolve_dot(dtmp1, lu, ch[i], 0, i, answer);
		rts_sub(dtmp, get_tsvector_i(answer, i), dtmp1);
		set_tsvector_i(answer, i, dtmp);
	}

/* Backward (row-oriented) */
	for(i = (dim - 1); i >= 0; i--)
	{
		_bnc_tssolve_dot(dtmp1, lu, ch[i], i + 1, dim, answer);
		rts_sub(dtmp, get_tsvector_i(answer, i), dtmp1);
		rts_div(dtmp, dtmp, get_tsmatrix_ij(lu, ch[i], i));
		set_tsvector_i(answer, i, dtmp);
	}
#else // BNC_USE_NEW_FMA
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rts_abs(dtmp, get_tsmatrix_ij(lu, ch[i], i));
		if(rts_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveTSLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rts_mul(dtmp1, get_tsmatrix_ij(lu, ch[j], i), get_tsvector_i(answer, i));
			rts_sub(dtmp, get_tsvector_i(answer, j), dtmp1);
			set_tsvector_i(answer, j, dtmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rts_mul(dtmp1, get_tsmatrix_ij(lu, ch[i], j), get_tsvector_i(answer, j));
			rts_sub(dtmp, get_tsvector_i(answer, i), dtmp1);
			set_tsvector_i(answer, i, dtmp);
		}
		rts_div(dtmp, get_tsvector_i(answer, i), get_tsmatrix_ij(lu, ch[i], i));
		set_tsvector_i(answer, i, dtmp);
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
int TSLUdecompC(TSMatrix a, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       TSMatrix a[]: Matrix (given by user)               */
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
	static float dtmp[TSSIZE], dtmp1[TSSIZE], dmaxii[TSSIZE];
#ifdef BNC_USE_NEW_FMA
	static float neg_aji[TSSIZE];
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
		rts_abs(dmaxii, get_tsmatrix_ij(a, row_ch[i], col_ch[i]));
		imax = i;
		jmax = i;
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rts_abs(dtmp, get_tsmatrix_ij(a, row_ch[j], col_ch[k]));
				if(rts_cmp(dtmp, dmaxii) > 0)
				{
					imax = j;
					jmax = k;
					rts_set(dmaxii, dtmp);
				}
			}
		}

		if(rts_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (TSLUdecompC)!\n", i);
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
			rts_div(dtmp, get_tsmatrix_ij(a, row_ch[j], col_ch[i]), get_tsmatrix_ij(a, row_ch[i], col_ch[i]));
			set_tsmatrix_ij(a, row_ch[j], col_ch[i], dtmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
#ifdef BNC_USE_NEW_FMA
			rts_neg(neg_aji, get_tsmatrix_ij(a, row_ch[j], col_ch[i]));
#endif // BNC_USE_NEW_FMA
			for(k = (i + 1); k < dim; k++)
			{
#ifdef BNC_USE_NEW_FMA
				rts_fma(dtmp, neg_aji, get_tsmatrix_ij(a, row_ch[i], col_ch[k]), get_tsmatrix_ij(a, row_ch[j], col_ch[k]));
#else // BNC_USE_NEW_FMA
				rts_mul(dtmp1, get_tsmatrix_ij(a, row_ch[j], col_ch[i]), get_tsmatrix_ij(a, row_ch[i], col_ch[k]));
				rts_sub(dtmp, get_tsmatrix_ij(a, row_ch[j], col_ch[k]), dtmp1);
#endif // BNC_USE_NEW_FMA
				set_tsmatrix_ij(a, row_ch[j], col_ch[k], dtmp);
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
int SolveTSLSC(TSVector answer, TSMatrix lu, TSVector b, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       TSMatrix lu: LU decomposed Matrix (given by user)  */
/*       TSVector b: constant vector (given by user)        */
/*       TSVector answer: Solution for linear system        */
/*       long int row_ch[]: Row order (given by user)       */
/*       long int col_ch[]: Column order (given by user)    */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static float dtmp[TSSIZE], dtmp1[TSSIZE];

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_tsvector_i(answer, col_ch[i], get_tsvector_i(b, row_ch[i]));

#ifdef BNC_USE_NEW_FMA
/* fused branch-free FMA substitution (complete pivoting: scalar) */
	{
	static float ntmp[TSSIZE];

/* Forward */
	for(i = 0; i < dim; i++)
	{
		rts_abs(dtmp, get_tsmatrix_ij(lu, row_ch[i], col_ch[i]));
		if(rts_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveTSLSC, %ld)\n", i);
			return -1;
		}

		rts_neg(ntmp, get_tsvector_i(answer, col_ch[i]));
		for(j = (i + 1); j < dim; j++)
		{
			rts_fma(dtmp, get_tsmatrix_ij(lu, row_ch[j], col_ch[i]), ntmp, get_tsvector_i(answer, col_ch[j]));
			set_tsvector_i(answer, col_ch[j], dtmp);
		}
	}

/* Backward */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rts_neg(ntmp, get_tsmatrix_ij(lu, row_ch[i], col_ch[j]));
			rts_fma(dtmp, ntmp, get_tsvector_i(answer, col_ch[j]), get_tsvector_i(answer, col_ch[i]));
			set_tsvector_i(answer, col_ch[i], dtmp);
		}
		rts_div(dtmp, get_tsvector_i(answer, col_ch[i]), get_tsmatrix_ij(lu, row_ch[i], col_ch[i]));
		set_tsvector_i(answer, col_ch[i], dtmp);
	}
	}
#else // BNC_USE_NEW_FMA
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rts_abs(dtmp, get_tsmatrix_ij(lu, row_ch[i], col_ch[i]));
		if(rts_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveTSLSC, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rts_mul(dtmp1, get_tsmatrix_ij(lu, row_ch[j], col_ch[i]), get_tsvector_i(answer, col_ch[i]));
			rts_sub(dtmp, get_tsvector_i(answer, col_ch[j]), dtmp1);
			set_tsvector_i(answer, col_ch[j],  dtmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rts_mul(dtmp1, get_tsmatrix_ij(lu, row_ch[i], col_ch[j]), get_tsvector_i(answer, col_ch[j]));
			rts_sub(dtmp, get_tsvector_i(answer, col_ch[i]), dtmp1);
			set_tsvector_i(answer, col_ch[i], dtmp);
		}
		rts_div(dtmp, get_tsvector_i(answer, col_ch[i]), get_tsmatrix_ij(lu, row_ch[i], col_ch[i]));
		set_tsvector_i(answer, col_ch[i], dtmp);
	}

#endif // BNC_USE_NEW_FMA
	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                          (triple float Precision)       */
/*          (Partial Pivoting with real swap of rows)       */
/*                                                          */
/*                 Ver. 0.0 2021-02-17 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int TSLUdecompPM(TSMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       TSMatrix a: Matrix (given by user)                 */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
	static float dtmp[TSSIZE], dtmp1[TSSIZE], dmaxii[TSSIZE];
#ifdef BNC_USE_NEW_FMA
	float neg_aji[TSSIZE];
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m256 dtmp256[TSSIZE], aji256[TSSIZE], ajk256[TSSIZE], aik256[TSSIZE];
#elif defined(__AVX512F__) // __AVX512F__
	long int k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m512 dtmp512[TSSIZE], aji512[TSSIZE], ajk512[TSSIZE], aik512[TSSIZE];
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	long int k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	svfloat32_t dtmp_neon_0, dtmp_neon_1, dtmp_neon_2;
	svfloat32_t aji_neon_0, aji_neon_1, aji_neon_2;
	svfloat32_t ajk_neon_0, ajk_neon_1, ajk_neon_2;
	svfloat32_t aik_neon_0, aik_neon_1, aik_neon_2;
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	long int k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	float32x4_t dtmp_neon[TSSIZE], aji_neon[TSSIZE], ajk_neon[TSSIZE], aik_neon[TSSIZE];
#else // others
#endif // __AVX2__
#endif // BNC_USE_NEW_FMA

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		// partial pivoting
		rts_abs(dmaxii, get_tsmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rts_abs(dtmp, get_tsmatrix_ij(a, j, i));
			if(rts_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rts_set(dmaxii, dtmp);
			}
		}

		if(rts_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! TSLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			row_swap_tsmatrix(a, i, imax, 0, a->col_dim);
		}

		for(j = (i + 1); j < dim; j++)
		{
			rts_div(dtmp, get_tsmatrix_ij(a, j, i), get_tsmatrix_ij(a, i, i));
			set_tsmatrix_ij(a, j, i, dtmp);
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
			aji256[0] = _mm256_set1_ps(neg_aji[0]);
			aji256[1] = _mm256_set1_ps(neg_aji[1]);
			aji256[2] = _mm256_set1_ps(neg_aji[2]);

			// head
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rts_fma(dtmp, neg_aji, get_tsmatrix_ij(a, i, k), get_tsmatrix_ij(a, j, k));
				set_tsmatrix_ij(a, j, k, dtmp);
			}

			// middle : SIMD
			for(k = dim_start; k < dim_end; k += _BNC_S_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				index_jk = j * a->real_col_dim + k;
				aik256[0] = _mm256_load_ps(&(a->element[0][index_ik]));
				aik256[1] = _mm256_load_ps(&(a->element[1][index_ik]));
				aik256[2] = _mm256_load_ps(&(a->element[2][index_ik]));
				ajk256[0] = _mm256_load_ps(&(a->element[0][index_jk]));
				ajk256[1] = _mm256_load_ps(&(a->element[1][index_jk]));
				ajk256[2] = _mm256_load_ps(&(a->element[2][index_jk]));
				_bncavx2_twfmaf(dtmp256, aji256, aik256, ajk256);
				_mm256_store_ps(&(a->element[0][index_jk]), dtmp256[0]);
				_mm256_store_ps(&(a->element[1][index_jk]), dtmp256[1]);
				_mm256_store_ps(&(a->element[2][index_jk]), dtmp256[2]);
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
			aji512[0] = _mm512_set1_ps(neg_aji[0]);
			aji512[1] = _mm512_set1_ps(neg_aji[1]);
			aji512[2] = _mm512_set1_ps(neg_aji[2]);

			// head
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rts_fma(dtmp, neg_aji, get_tsmatrix_ij(a, i, k), get_tsmatrix_ij(a, j, k));
				set_tsmatrix_ij(a, j, k, dtmp);
			}

			// middle : SIMD
			for(k = dim_start; k < dim_end; k += _BNC_S_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				index_jk = j * a->real_col_dim + k;
				aik512[0] = _mm512_load_ps(&(a->element[0][index_ik]));
				aik512[1] = _mm512_load_ps(&(a->element[1][index_ik]));
				aik512[2] = _mm512_load_ps(&(a->element[2][index_ik]));
				ajk512[0] = _mm512_load_ps(&(a->element[0][index_jk]));
				ajk512[1] = _mm512_load_ps(&(a->element[1][index_jk]));
				ajk512[2] = _mm512_load_ps(&(a->element[2][index_jk]));
				_bncavx512_twfmaf(dtmp512, aji512, aik512, ajk512);
				_mm512_store_ps(&(a->element[0][index_jk]), dtmp512[0]);
				_mm512_store_ps(&(a->element[1][index_jk]), dtmp512[1]);
				_mm512_store_ps(&(a->element[2][index_jk]), dtmp512[2]);
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
			aji_neon_0 = svdup_f32(neg_aji[0]);
			aji_neon_1 = svdup_f32(neg_aji[1]);
			aji_neon_2 = svdup_f32(neg_aji[2]);

			// head
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rts_fma(dtmp, neg_aji, get_tsmatrix_ij(a, i, k), get_tsmatrix_ij(a, j, k));
				set_tsmatrix_ij(a, j, k, dtmp);
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
				ajk_neon_0 = svld1_f32(pg, &(a->element[0][index_jk]));
				ajk_neon_1 = svld1_f32(pg, &(a->element[1][index_jk]));
				ajk_neon_2 = svld1_f32(pg, &(a->element[2][index_jk]));
				_bncsve2_twfmaf(pg, &dtmp_neon_0, &dtmp_neon_1, &dtmp_neon_2, aji_neon_0, aji_neon_1, aji_neon_2, aik_neon_0, aik_neon_1, aik_neon_2, ajk_neon_0, ajk_neon_1, ajk_neon_2);
				svst1_f32(pg, &(a->element[0][index_jk]), dtmp_neon_0);
				svst1_f32(pg, &(a->element[1][index_jk]), dtmp_neon_1);
				svst1_f32(pg, &(a->element[2][index_jk]), dtmp_neon_2);
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
			aji_neon[0] = vdupq_n_f32(neg_aji[0]);
			aji_neon[1] = vdupq_n_f32(neg_aji[1]);
			aji_neon[2] = vdupq_n_f32(neg_aji[2]);

			// head
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rts_fma(dtmp, neg_aji, get_tsmatrix_ij(a, i, k), get_tsmatrix_ij(a, j, k));
				set_tsmatrix_ij(a, j, k, dtmp);
			}

			// middle : SIMD (Neon)
			for(k = dim_start; k < dim_end; k += _BNC_S_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				index_jk = j * a->real_col_dim + k;
				aik_neon[0] = vld1q_f32(&(a->element[0][index_ik]));
				aik_neon[1] = vld1q_f32(&(a->element[1][index_ik]));
				aik_neon[2] = vld1q_f32(&(a->element[2][index_ik]));
				ajk_neon[0] = vld1q_f32(&(a->element[0][index_jk]));
				ajk_neon[1] = vld1q_f32(&(a->element[1][index_jk]));
				ajk_neon[2] = vld1q_f32(&(a->element[2][index_jk]));
				_bncneon_twfmaf(dtmp_neon, aji_neon, aik_neon, ajk_neon);
				vst1q_f32(&(a->element[0][index_jk]), dtmp_neon[0]);
				vst1q_f32(&(a->element[1][index_jk]), dtmp_neon[1]);
				vst1q_f32(&(a->element[2][index_jk]), dtmp_neon[2]);
			}
		}
#else // others
		for(j = (i + 1); j < dim; j++)
		{
			rts_neg(neg_aji, get_tsmatrix_ij(a, j, i));
			for(k = (i + 1); k < dim; k++)
			{
				rts_fma(dtmp, neg_aji, get_tsmatrix_ij(a, i, k), get_tsmatrix_ij(a, j, k));
				set_tsmatrix_ij(a, j, k, dtmp);
			}
		}
#endif // __AVX2__
#else // BNC_USE_NEW_FMA
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				rts_mul(dtmp1, get_tsmatrix_ij(a, j, i), get_tsmatrix_ij(a, i, k));
				rts_sub(dtmp, get_tsmatrix_ij(a, j, k), dtmp1);
				set_tsmatrix_ij(a, j, k, dtmp);
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
int SolveTSLSPM(TSVector answer, TSMatrix lu, TSVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*      TSMatrix lu[]: LU decomposed Matrix (given by user) */
/*      TSVector b[]: constant vector (given by user)       */
/*      TSVector answer[]: Solution for linear system       */
/*      long int ch: Row order (given by user)              */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	static float dtmp[TSSIZE], dtmp1[TSSIZE];

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_tsvector_i(answer, i, get_tsvector_i(b, ch[i]));

	
#ifdef BNC_USE_NEW_FMA
/* fused branch-free FMA + SIMD forward/backward substitution */
	for(i = 0; i < dim; i++)
	{
		rts_abs(dtmp, get_tsmatrix_ij(lu, i, i));
		if(rts_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveTSLSP, %ld)\n", i);
			return -1;
		}
	}

/* Forward (row-oriented) */
	for(i = 1; i < dim; i++)
	{
		_bnc_tssolve_dot(dtmp1, lu, i, 0, i, answer);
		rts_sub(dtmp, get_tsvector_i(answer, i), dtmp1);
		set_tsvector_i(answer, i, dtmp);
	}

/* Backward (row-oriented) */
	for(i = (dim - 1); i >= 0; i--)
	{
		_bnc_tssolve_dot(dtmp1, lu, i, i + 1, dim, answer);
		rts_sub(dtmp, get_tsvector_i(answer, i), dtmp1);
		rts_div(dtmp, dtmp, get_tsmatrix_ij(lu, i, i));
		set_tsvector_i(answer, i, dtmp);
	}
#else // BNC_USE_NEW_FMA
/* Forward */
	for(i = 0; i < dim; i++)
	{
		rts_abs(dtmp, get_tsmatrix_ij(lu, i, i));
		if(rts_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveTSLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			rts_mul(dtmp1, get_tsmatrix_ij(lu, j, i), get_tsvector_i(answer, i));
			rts_sub(dtmp, get_tsvector_i(answer, j), dtmp1);
			set_tsvector_i(answer, j, dtmp);
		}
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			rts_mul(dtmp1, get_tsmatrix_ij(lu, i, j), get_tsvector_i(answer, j));
			rts_sub(dtmp, get_tsvector_i(answer, i), dtmp1);
			set_tsvector_i(answer, i, dtmp);
		}
		rts_div(dtmp, get_tsvector_i(answer, i), get_tsmatrix_ij(lu, i, i));
		set_tsvector_i(answer, i, dtmp);
	}

#endif // BNC_USE_NEW_FMA
	return 0;
}

#ifdef __cplusplus
} // extern "C"
#endif
