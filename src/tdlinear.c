/********************************************************************************/
/* tdlinear.c: Triple-double and Quadruple precision Linear Computation Library */
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
//#include "rdd.h"
#include "tdlinear.h"

#ifdef USE_GMP
#include "gmp.h"
#include "mpfr.h"
#include "mpfr_dtq_sd.h"
#include "mpflinear.h"
#endif //USE_GMP//

#if defined(USE_GMP) && defined(USE_MPFR)
// Frobenius norm
tdfloat tdnormf(tdfloat array[], int dim)
{
    int i;
    tdfloat ret, tmp;
    mpfr_t mpfr_ret;

    rtd_set_ui(ret.val, 0UL);
    for(i = 0; i < dim; i++)
    {
        rtd_mul(tmp.val, array[i].val, array[i].val);
        rtd_add(ret.val, ret.val, tmp.val);
    }
    //printf("ret.val = "); rtd_out_str(ret.val); printf("\n");
//  rtd_sqrt(ret, ret);
    mpfr_init2(mpfr_ret, 128);
    mpfr_set_td(mpfr_ret, ret.val, MPFR_RNDN);
    mpfr_sqrt(mpfr_ret, mpfr_ret, MPFR_RNDN);
    mpfr_get_td(ret.val, mpfr_ret, MPFR_RNDN);
    mpfr_clear(mpfr_ret);
    return ret;
}

// generate a text vector: vec(i) := sqrt(sqrt_seed) * (i + 1)
/*void set_test_tdvector(tdfloat vec[], int sqrt_seed, int dim)
{
    int i, j;
    tdfloat ddsqrt;
    mpfr_t mpfrsqrt;

    // ddsqrt := sqrt(sqrt_seed)
    mpfr_init2(mpfrsqrt, 128);
    mpfr_sqrt_ui(mpfrsqrt, (unsigned long)sqrt_seed, MPFR_RNDN);
    mpfr_get_dd(ddsqrt.val, mpfrsqrt, MPFR_RNDN);
//    rtd_set_ui(ddsqrt.val, sqrt_seed);
    //rtd_sqrt(ddsqrt.val, ddsqrt.val);
    //rtd_sqrt_ui(ddsqrt.val, sqrt_seed);

    //printf("set_test_tdmatrix: coef = "); rtd_out_str(ddsqrt.val); printf("\n");

    for(i = 0; i < dim; i++)
    {
        //printf("%5d: ", i);
        rtd_set_ui(vec[i].val, i + 1);
        rtd_mul(vec[i].val, vec[i].val, ddsqrt.val);
    }
}*/

//#if 0
// tdrel_diff
tdfloat tdrel_diff(tdfloat a, tdfloat b)
{
    tdfloat rel_diff, abs_a;

    //rel_diff = fabs(a - b);
    rtd_sub(rel_diff.val, a.val, b.val);
    rtd_abs(rel_diff.val, rel_diff.val);

    //if(a != 0.0)
    if(rtd_cmp_ui(a.val, 0UL) != 0)
    {
//        rel_diff /= fabs(a);
        rtd_abs(abs_a.val, a.val);
        rtd_div(rel_diff.val, rel_diff.val, a.val);
    }

    return rel_diff;
}
//#endif // 0

tdfloat tdrel_diff_array(tdfloat approx_a[], tdfloat approx_b[], int dim, int print_flag)
{
    int i;
    tdfloat rel_min, rel_max, rel_ave, rel_diff;
    mpfr_t mpfr_tmp;

    rel_diff = tdrel_diff(approx_a[0], approx_b[0]);
    rtd_set(rel_min.val, rel_diff.val);
    rtd_set(rel_max.val, rel_diff.val);
    rtd_set(rel_ave.val, rel_diff.val);

    for(i = 1; i < dim; i++)
    {
        rel_diff = tdrel_diff(approx_a[i], approx_b[i]);
        if(rtd_cmp(rel_diff.val, rel_min.val) < 0) rtd_set(rel_min.val, rel_diff.val);
        if(rtd_cmp(rel_diff.val, rel_max.val) > 0) rtd_set(rel_max.val, rel_diff.val);
        //rel_ave += rel_diff;
        rtd_add(rel_ave.val, rel_ave.val, rel_diff.val);
    }
    //rel_ave /= (cddfloat)dim;
    //rtd_div_ui(rel_ave.val, rel_ave.val, (unsigned long)dim);
    mpfr_init2(mpfr_tmp, 256);
    mpfr_set_td(mpfr_tmp, rel_ave.val, MPFR_RNDN);
    mpfr_div_ui(mpfr_tmp, mpfr_tmp, (unsigned long)dim, MPFR_RNDN);
    mpfr_get_td(rel_ave.val, mpfr_tmp, MPFR_RNDN);
    mpfr_clear(mpfr_tmp);

    if(print_flag == 1)
    {
        printf("max_rel_diff, min_rel_diff, ave_rel_diff:"); rtd_out_str(rel_max.val); printf(" "); rtd_out_str(rel_min.val);  printf(" "); rtd_out_str(rel_ave.val); printf("\n"); 
    }

    return rel_max;
}
#endif // defined(USE_GMP) && defined(USE_MPFR)

// initialize TDVector
TDVector init_tdvector(long int dimension)
{
	TDVector ret = NULL;
	long int i, real_dim;

	if(dimension <= 0)
	{
		fprintf(stderr, "ERROR: init_tdvector\n");
		return ret;
	}

	//ret = (TDVector)BNC_MALLOC(sizeof(tdvector));
	ret = (TDVector)malloc(sizeof(tdvector)); // Fix it!: 2024-04-26(Fri) T.Kouya
	if(ret == NULL)
		return ret;

	// real_dim is the nearest positive multiplier of _BNC_D_WIDTH
	real_dim = (long int)ceil((double)(dimension) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;

	ret->element[0] = (double *)BNC_CALLOC(real_dim, sizeof(double));
	if(ret->element[0] == NULL)
	{ 	free(ret);
		return NULL;
	}
	ret->element[1] = (double *)BNC_CALLOC(real_dim, sizeof(double));
	if(ret->element[1] == NULL)
	{
		free(ret->element[0]);
		free(ret);
		return NULL;
	}
	ret->element[2] = (double *)BNC_CALLOC(real_dim, sizeof(double));
	if(ret->element[2] == NULL)
	{
		free(ret->element[0]);
		free(ret->element[1]);
		free(ret);
		return NULL;
	}

	/* All 0 */
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	for(i = 0; i < real_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&(ret->element[0][i]), zero4);
		_mm256_store_pd(&(ret->element[1][i]), zero4);
		_mm256_store_pd(&(ret->element[2][i]), zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(i = 0; i < real_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&(ret->element[0][i]), zero8);
		_mm512_store_pd(&(ret->element[1][i]), zero8);
		_mm512_store_pd(&(ret->element[2][i]), zero8);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t zero2;

    zero2 = svdup_f64(0.0);
    for(i = 0; i < real_dim; i += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(real_dim));
        svst1_f64(pg, &(ret->element[0][i]), zero2);
        svst1_f64(pg, &(ret->element[1][i]), zero2);
        svst1_f64(pg, &(ret->element[2][i]), zero2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm NEON
    float64x2_t zero2;

    zero2 = vdupq_n_f64(0.0);
    for(i = 0; i < real_dim; i += _BNC_D_WIDTH)
    {
        vst1q_f64(&(ret->element[0][i]), zero2);
        vst1q_f64(&(ret->element[1][i]), zero2);
        vst1q_f64(&(ret->element[2][i]), zero2);
    }
#else // others
	for(i = 0; i < dimension; i++)
	{
		ret->element[0][i] = 0.0;
		ret->element[1][i] = 0.0;
		ret->element[2][i] = 0.0;
	}
#endif // __AVX2__

	ret->dim = dimension;
	ret->real_dim = real_dim;

	return ret;
}

// free TDVector
void free_tdvector(TDVector vec)
{
    long int i;
    for(i = 0; i < TDSIZE; i++)
        free(vec->element[i]);

    free(vec);
}

// TDVector vec -> tdfloat array
void set_tdfloat_tdvec(tdfloat ret[], int ret_dim, TDVector vec)
{
    int index, j, dim;

    dim = (ret_dim < vec->dim) ? ret_dim : vec->dim;

    for(index = 0; index < dim; index++)
    {
        for(j = 0; j < TDSIZE; j++)
            ret[index].val[j] = vec->element[j][index];
    }
}

// tdfloat array -> TDVector ret
void set_tdvector_tdfloat(TDVector ret, tdfloat array[], int array_dim)
{
    int index, j, dim;

    dim = (ret->dim < array_dim) ? ret->dim : array_dim;

    for(index = 0; index < dim; index++)
    {
        for(j = 0; j < TDSIZE; j++)
            ret->element[j][index] = array[index].val[j];
    }
}

// print tdvector
void print_tdvector(TDVector vec)
{
	long int index;

	for(index = 0; index < vec->dim; index++)
	{
		printf("%4ld: ", index);
		//c_dd_write((vec->element + index * TDSIZE));
		//c_td_write(GET_TDVECTOR_I(vec, index));
		rtd_out_str(get_tdvector_i(vec, index));
		printf("\n");
	}
}

// set a zero vector
void set0_tdvector(TDVector vec)
{
	long int i;

	/* All 0 */
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	for(i = 0; i < vec->real_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&vec->element[0][i], zero4);
		_mm256_store_pd(&vec->element[1][i], zero4);
		_mm256_store_pd(&vec->element[2][i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(i = 0; i < vec->real_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&vec->element[0][i], zero8);
		_mm512_store_pd(&vec->element[1][i], zero8);
		_mm512_store_pd(&vec->element[2][i], zero8);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t zero2;

    zero2 = svdup_f64(0.0);
    for(i = 0; i < vec->real_dim; i += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(vec->real_dim));
        svst1_f64(pg, &vec->element[0][i], zero2);
        svst1_f64(pg, &vec->element[1][i], zero2);
        svst1_f64(pg, &vec->element[2][i], zero2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm NEON
    float64x2_t zero2;

    zero2 = vdupq_n_f64(0.0);
    for(i = 0; i < vec->real_dim; i += _BNC_D_WIDTH)
    {
        vst1q_f64(&vec->element[0][i], zero2);
        vst1q_f64(&vec->element[1][i], zero2);
        vst1q_f64(&vec->element[2][i], zero2);
    }
#else // others
	for(i = 0; i < vec->dim; i++)
	{
		vec->element[0][i] = 0.0;
		vec->element[1][i] = 0.0;
		vec->element[2][i] = 0.0;
	}
#endif // __AVX2__
}

// set_tdvector_i_str
void set_tdvector_i_str(TDVector vec, long int index, const char *str)
{
	double tmp[TDSIZE];

	//rtd_get_str(tmp, str);
	rtd_set_str(tmp, str);

	set_tdvector_i(vec, index, tmp);
}

/*************************************************/
/* Vector Calculations for TDVector               */
/*
void add_tdvector(TDVector c, TDVector a, TDVector b)
void add2_tdvector(TDVector c, TDVector a)
void sub_tdvector(TDVector c, TDVector a, TDVector b)
void sub2_tdvector(TDVector c, DVector a)
void cmul_tdvector(TDVector c, double val[TDSIZE], TDVector a)
void cmul2_tdvector(TDVector c, double val[TDSIZE])
void add_cmul_tdvector(TDVector c, TDVector a, double val[TDSIZE], TDVector b)
double ip_tdvector(TDVector a, TDVector b)
double norm1_tdvector(TDVector a)
double norm2_tdvector(TDVector a)
double normi_tdvector(TDVector a)
void subst_tdvector(TDVector c, TDVector a)
*/
/*************************************************/
/* c = a + b */
void add_tdvector(TDVector c, TDVector a, TDVector b)
{
    long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_tdvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256d in_ret[TDSIZE], in_a_val[TDSIZE], in_b_val[TDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_pd(&(a->element[1][index]));
        in_a_val[2] = _mm256_load_pd(&(a->element[2][index]));
        in_b_val[0] = _mm256_load_pd(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_pd(&(b->element[1][index]));
        in_b_val[2] = _mm256_load_pd(&(b->element[2][index]));

        _bncavx2_rtd_add(in_ret, in_a_val, in_b_val);

        _mm256_store_pd(&(c->element[0][index]), in_ret[0]);
        _mm256_store_pd(&(c->element[1][index]), in_ret[1]);
        _mm256_store_pd(&(c->element[2][index]), in_ret[2]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512d in_ret[TDSIZE], in_a_val[TDSIZE], in_b_val[TDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm512_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_pd(&(a->element[1][index]));
        in_a_val[2] = _mm512_load_pd(&(a->element[2][index]));
        in_b_val[0] = _mm512_load_pd(&(b->element[0][index]));
        in_b_val[1] = _mm512_load_pd(&(b->element[1][index]));
        in_b_val[2] = _mm512_load_pd(&(b->element[2][index]));

        _bncavx512_rtd_add(in_ret, in_a_val, in_b_val);

        _mm512_store_pd(&(c->element[0][index]), in_ret[0]);
        _mm512_store_pd(&(c->element[1][index]), in_ret[1]);
        _mm512_store_pd(&(c->element[2][index]), in_ret[2]);
   }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t in_ret_0, in_ret_1, in_ret_2;
	svfloat64_t in_a_val_0, in_a_val_1, in_a_val_2;
	svfloat64_t in_b_val_0, in_b_val_1, in_b_val_2;

    for(index = 0; index < c->real_dim; index += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        in_a_val_0 = svld1_f64(pg, &(a->element[0][index]));
        in_a_val_1 = svld1_f64(pg, &(a->element[1][index]));
        in_a_val_2 = svld1_f64(pg, &(a->element[2][index]));
        in_b_val_0 = svld1_f64(pg, &(b->element[0][index]));
        in_b_val_1 = svld1_f64(pg, &(b->element[1][index]));
        in_b_val_2 = svld1_f64(pg, &(b->element[2][index]));

        _bncsve2_rtd_add(pg, &in_ret_0, &in_ret_1, &in_ret_2, in_a_val_0, in_a_val_1, in_a_val_2, in_b_val_0, in_b_val_1, in_b_val_2);

        svst1_f64(pg, &(c->element[0][index]), in_ret_0);
        svst1_f64(pg, &(c->element[1][index]), in_ret_1);
        svst1_f64(pg, &(c->element[2][index]), in_ret_2);
   }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm NEON
    float64x2_t in_ret[TDSIZE], in_a_val[TDSIZE], in_b_val[TDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = vld1q_f64(&(a->element[0][index]));
        in_a_val[1] = vld1q_f64(&(a->element[1][index]));
        in_a_val[2] = vld1q_f64(&(a->element[2][index]));
        in_b_val[0] = vld1q_f64(&(b->element[0][index]));
        in_b_val[1] = vld1q_f64(&(b->element[1][index]));
        in_b_val[2] = vld1q_f64(&(b->element[2][index]));

        _bncneon_rtd_add(in_ret, in_a_val, in_b_val);

        vst1q_f64(&(c->element[0][index]), in_ret[0]);
        vst1q_f64(&(c->element[1][index]), in_ret[1]);
        vst1q_f64(&(c->element[2][index]), in_ret[2]);
   }
#else // others
	double tmp[TDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rtd_add(tmp, get_tdvector_i(a, i),  get_tdvector_i(b, i));
		set_tdvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c += a */
void add2_tdvector(TDVector c, TDVector a)
{
	long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: add2_tdvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256d in_ret[TDSIZE], in_a_val[TDSIZE], tmp4[TDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_pd(&(a->element[1][index]));
        in_a_val[2] = _mm256_load_pd(&(a->element[2][index]));
        in_ret[0] = _mm256_load_pd(&(c->element[0][index]));
        in_ret[1] = _mm256_load_pd(&(c->element[1][index]));
        in_ret[2] = _mm256_load_pd(&(c->element[2][index]));

        _bncavx2_rtd_add(tmp4, in_ret, in_a_val);

        _mm256_store_pd(&(c->element[0][index]), tmp4[0]);
        _mm256_store_pd(&(c->element[1][index]), tmp4[1]);
        _mm256_store_pd(&(c->element[2][index]), tmp4[2]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512d in_ret[TDSIZE], in_a_val[TDSIZE], tmp8[TDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm512_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_pd(&(a->element[1][index]));
        in_a_val[2] = _mm512_load_pd(&(a->element[2][index]));
        in_ret[0] = _mm512_load_pd(&(c->element[0][index]));
        in_ret[1] = _mm512_load_pd(&(c->element[1][index]));
        in_ret[2] = _mm512_load_pd(&(c->element[2][index]));

        _bncavx512_rtd_add(tmp8, in_ret, in_a_val);

        _mm512_store_pd(&(c->element[0][index]), tmp8[0]);
        _mm512_store_pd(&(c->element[1][index]), tmp8[1]);
        _mm512_store_pd(&(c->element[2][index]), tmp8[2]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t in_ret_0, in_ret_1, in_ret_2;
	svfloat64_t in_a_val_0, in_a_val_1, in_a_val_2;
	svfloat64_t tmp2_0, tmp2_1, tmp2_2;

    for(index = 0; index < c->real_dim; index += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        in_a_val_0 = svld1_f64(pg, &(a->element[0][index]));
        in_a_val_1 = svld1_f64(pg, &(a->element[1][index]));
        in_a_val_2 = svld1_f64(pg, &(a->element[2][index]));
        in_ret_0 = svld1_f64(pg, &(c->element[0][index]));
        in_ret_1 = svld1_f64(pg, &(c->element[1][index]));
        in_ret_2 = svld1_f64(pg, &(c->element[2][index]));

        _bncsve2_rtd_add(pg, &tmp2_0, &tmp2_1, &tmp2_2, in_ret_0, in_ret_1, in_ret_2, in_a_val_0, in_a_val_1, in_a_val_2);

        svst1_f64(pg, &(c->element[0][index]), tmp2_0);
        svst1_f64(pg, &(c->element[1][index]), tmp2_1);
        svst1_f64(pg, &(c->element[2][index]), tmp2_2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm NEON
    float64x2_t in_ret[TDSIZE], in_a_val[TDSIZE], tmp2[TDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = vld1q_f64(&(a->element[0][index]));
        in_a_val[1] = vld1q_f64(&(a->element[1][index]));
        in_a_val[2] = vld1q_f64(&(a->element[2][index]));
        in_ret[0] = vld1q_f64(&(c->element[0][index]));
        in_ret[1] = vld1q_f64(&(c->element[1][index]));
        in_ret[2] = vld1q_f64(&(c->element[2][index]));

        _bncneon_rtd_add(tmp2, in_ret, in_a_val);

        vst1q_f64(&(c->element[0][index]), tmp2[0]);
        vst1q_f64(&(c->element[1][index]), tmp2[1]);
        vst1q_f64(&(c->element[2][index]), tmp2[2]);
    }
#else // others
	double tmp[TDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rtd_add(tmp, get_tdvector_i(c, i), get_tdvector_i(a, i));
		set_tdvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c = a - b */
void sub_tdvector(TDVector c, TDVector a, TDVector b)
{
    long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_tdvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256d in_ret[TDSIZE], in_a_val[TDSIZE], in_b_val[TDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_pd(&(a->element[1][index]));
        in_a_val[2] = _mm256_load_pd(&(a->element[2][index]));
        in_b_val[0] = _mm256_load_pd(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_pd(&(b->element[1][index]));
        in_b_val[2] = _mm256_load_pd(&(b->element[2][index]));

        _bncavx2_rtd_sub(in_ret, in_a_val, in_b_val);

        _mm256_store_pd(&(c->element[0][index]), in_ret[0]);
        _mm256_store_pd(&(c->element[1][index]), in_ret[1]);
        _mm256_store_pd(&(c->element[2][index]), in_ret[2]);
  }
#elif defined(__AVX512F__) // __AVX512F__
    __m512d in_ret[TDSIZE], in_a_val[TDSIZE], in_b_val[TDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm512_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_pd(&(a->element[1][index]));
        in_a_val[2] = _mm512_load_pd(&(a->element[2][index]));
        in_b_val[0] = _mm512_load_pd(&(b->element[0][index]));
        in_b_val[1] = _mm512_load_pd(&(b->element[1][index]));
        in_b_val[2] = _mm512_load_pd(&(b->element[2][index]));

        _bncavx512_rtd_sub(in_ret, in_a_val, in_b_val);

        _mm512_store_pd(&(c->element[0][index]), in_ret[0]);
        _mm512_store_pd(&(c->element[1][index]), in_ret[1]);
        _mm512_store_pd(&(c->element[2][index]), in_ret[2]);
   }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t in_ret_0, in_ret_1, in_ret_2;
	svfloat64_t in_a_val_0, in_a_val_1, in_a_val_2;
	svfloat64_t in_b_val_0, in_b_val_1, in_b_val_2;

    for(index = 0; index < c->real_dim; index += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        in_a_val_0 = svld1_f64(pg, &(a->element[0][index]));
        in_a_val_1 = svld1_f64(pg, &(a->element[1][index]));
        in_a_val_2 = svld1_f64(pg, &(a->element[2][index]));
        in_b_val_0 = svld1_f64(pg, &(b->element[0][index]));
        in_b_val_1 = svld1_f64(pg, &(b->element[1][index]));
        in_b_val_2 = svld1_f64(pg, &(b->element[2][index]));

        _bncsve2_rtd_neg(pg, &in_ret_0, &in_ret_1, &in_ret_2, in_b_val_0, in_b_val_1, in_b_val_2);
		_bncsve2_rtd_add(pg, &in_ret_0, &in_ret_1, &in_ret_2, in_a_val_0, in_a_val_1, in_a_val_2, in_ret_0, in_ret_1, in_ret_2);

        svst1_f64(pg, &(c->element[0][index]), in_ret_0);
        svst1_f64(pg, &(c->element[1][index]), in_ret_1);
        svst1_f64(pg, &(c->element[2][index]), in_ret_2);
   }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm NEON
    float64x2_t in_ret[TDSIZE], in_a_val[TDSIZE], in_b_val[TDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = vld1q_f64(&(a->element[0][index]));
        in_a_val[1] = vld1q_f64(&(a->element[1][index]));
        in_a_val[2] = vld1q_f64(&(a->element[2][index]));
        in_b_val[0] = vld1q_f64(&(b->element[0][index]));
        in_b_val[1] = vld1q_f64(&(b->element[1][index]));
        in_b_val[2] = vld1q_f64(&(b->element[2][index]));

        _bncneon_rtd_sub(in_ret, in_a_val, in_b_val);

        vst1q_f64(&(c->element[0][index]), in_ret[0]);
        vst1q_f64(&(c->element[1][index]), in_ret[1]);
        vst1q_f64(&(c->element[2][index]), in_ret[2]);
   }
#else // others
	double tmp[TDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rtd_sub(tmp, get_tdvector_i(a, i),  get_tdvector_i(b, i));
		set_tdvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c -= a */
void sub2_tdvector(TDVector c, TDVector a)
{
	long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: sub2_tdvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256d in_ret[TDSIZE], in_a_val[TDSIZE], tmp4[TDSIZE];

	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_pd(&(a->element[1][index]));
        in_a_val[2] = _mm256_load_pd(&(a->element[2][index]));
        in_ret[0] = _mm256_load_pd(&(c->element[0][index]));
        in_ret[1] = _mm256_load_pd(&(c->element[1][index]));
        in_ret[2] = _mm256_load_pd(&(c->element[2][index]));

        _bncavx2_rtd_sub(tmp4, in_ret, in_a_val);

        _mm256_store_pd(&(c->element[0][index]), tmp4[0]);
        _mm256_store_pd(&(c->element[1][index]), tmp4[1]);
        _mm256_store_pd(&(c->element[2][index]), tmp4[2]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512d in_ret[TDSIZE], in_a_val[TDSIZE], tmp8[TDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm512_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_pd(&(a->element[1][index]));
        in_a_val[2] = _mm512_load_pd(&(a->element[2][index]));
        in_ret[0] = _mm512_load_pd(&(c->element[0][index]));
        in_ret[1] = _mm512_load_pd(&(c->element[1][index]));
        in_ret[2] = _mm512_load_pd(&(c->element[2][index]));

        _bncavx512_rtd_sub(tmp8, in_ret, in_a_val);

        _mm512_store_pd(&(c->element[0][index]), tmp8[0]);
        _mm512_store_pd(&(c->element[1][index]), tmp8[1]);
        _mm512_store_pd(&(c->element[2][index]), tmp8[2]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t in_ret_0, in_ret_1, in_ret_2;
	svfloat64_t in_a_val_0, in_a_val_1, in_a_val_2;
	svfloat64_t tmp2_0, tmp2_1, tmp2_2;

    for(index = 0; index < c->real_dim; index += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        in_a_val_0 = svld1_f64(pg, &(a->element[0][index]));
        in_a_val_1 = svld1_f64(pg, &(a->element[1][index]));
        in_a_val_2 = svld1_f64(pg, &(a->element[2][index]));
        in_ret_0 = svld1_f64(pg, &(c->element[0][index]));
        in_ret_1 = svld1_f64(pg, &(c->element[1][index]));
        in_ret_2 = svld1_f64(pg, &(c->element[2][index]));

        _bncsve2_rtd_neg(pg, &tmp2_0, &tmp2_1, &tmp2_2, in_a_val_0, in_a_val_1, in_a_val_2);
		_bncsve2_rtd_add(pg, &tmp2_0, &tmp2_1, &tmp2_2, in_ret_0, in_ret_1, in_ret_2, tmp2_0, tmp2_1, tmp2_2);

        svst1_f64(pg, &(c->element[0][index]), tmp2_0);
        svst1_f64(pg, &(c->element[1][index]), tmp2_1);
        svst1_f64(pg, &(c->element[2][index]), tmp2_2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm NEON
    float64x2_t in_ret[TDSIZE], in_a_val[TDSIZE], tmp2[TDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = vld1q_f64(&(a->element[0][index]));
        in_a_val[1] = vld1q_f64(&(a->element[1][index]));
        in_a_val[2] = vld1q_f64(&(a->element[2][index]));
        in_ret[0] = vld1q_f64(&(c->element[0][index]));
        in_ret[1] = vld1q_f64(&(c->element[1][index]));
        in_ret[2] = vld1q_f64(&(c->element[2][index]));

        _bncneon_rtd_sub(tmp2, in_ret, in_a_val);

        vst1q_f64(&(c->element[0][index]), tmp2[0]);
        vst1q_f64(&(c->element[1][index]), tmp2[1]);
        vst1q_f64(&(c->element[2][index]), tmp2[2]);
    }
#else // others
	double tmp[TDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rtd_sub(tmp, get_tdvector_i(c, i), get_tdvector_i(a, i));
		set_tdvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c = val * a */
void cmul_tdvector(TDVector c, double val[TDSIZE], TDVector a)
{
    long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: cmul_dvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[TDSIZE], c4[TDSIZE], val4[TDSIZE];

	val4[0] = _mm256_set1_pd(val[0]);
	val4[1] = _mm256_set1_pd(val[1]);
	val4[2] = _mm256_set1_pd(val[2]);
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		//set_tdvector_i(c, i, val * get_tdvector_i(a, i));
		a4[0] = _mm256_load_pd(&(a->element[0][index]));
		a4[1] = _mm256_load_pd(&(a->element[1][index]));
		a4[2] = _mm256_load_pd(&(a->element[2][index]));

		_bncavx2_rtd_mul(c4, val4, a4);

		_mm256_store_pd(&(c->element[0][index]), c4[0]);
		_mm256_store_pd(&(c->element[1][index]), c4[1]);
		_mm256_store_pd(&(c->element[2][index]), c4[2]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[TDSIZE], c8[TDSIZE], val8[TDSIZE];

	val8[0] = _mm512_set1_pd(val[0]);
	val8[1] = _mm512_set1_pd(val[1]);
	val8[2] = _mm512_set1_pd(val[2]);
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		//set_tdvector_i(c, i, val * get_tdvector_i(a, i));
		a8[0] = _mm512_load_pd(&(a->element[0][index]));
		a8[1] = _mm512_load_pd(&(a->element[1][index]));
		a8[2] = _mm512_load_pd(&(a->element[2][index]));

		_bncavx512_rtd_mul(c8, val8, a8);

		_mm512_store_pd(&(c->element[0][index]), c8[0]);
		_mm512_store_pd(&(c->element[1][index]), c8[1]);
		_mm512_store_pd(&(c->element[2][index]), c8[2]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t a2_0, a2_1, a2_2;
	svfloat64_t c2_0, c2_1, c2_2;
	svfloat64_t val2_0, val2_1, val2_2;

    val2_0 = svdup_f64(val[0]);
    val2_1 = svdup_f64(val[1]);
    val2_2 = svdup_f64(val[2]);
    for(index = 0; index < c->real_dim; index += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        a2_0 = svld1_f64(pg, &(a->element[0][index]));
        a2_1 = svld1_f64(pg, &(a->element[1][index]));
        a2_2 = svld1_f64(pg, &(a->element[2][index]));

        _bncsve2_rtd_mul(pg, &c2_0, &c2_1, &c2_2, val2_0, val2_1, val2_2, a2_0, a2_1, a2_2);

        svst1_f64(pg, &(c->element[0][index]), c2_0);
        svst1_f64(pg, &(c->element[1][index]), c2_1);
        svst1_f64(pg, &(c->element[2][index]), c2_2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm NEON
    float64x2_t a2[TDSIZE], c2[TDSIZE], val2[TDSIZE];

    val2[0] = vdupq_n_f64(val[0]);
    val2[1] = vdupq_n_f64(val[1]);
    val2[2] = vdupq_n_f64(val[2]);
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        a2[0] = vld1q_f64(&(a->element[0][index]));
        a2[1] = vld1q_f64(&(a->element[1][index]));
        a2[2] = vld1q_f64(&(a->element[2][index]));

        _bncneon_rtd_mul(c2, val2, a2);

        vst1q_f64(&(c->element[0][index]), c2[0]);
        vst1q_f64(&(c->element[1][index]), c2[1]);
        vst1q_f64(&(c->element[2][index]), c2[2]);
    }
#else // others
	double tmp[TDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rtd_mul(tmp, val, get_tdvector_i(a, i));
		set_tdvector_i(c, i, tmp);
	}
#endif // __AVX2__

}

/* c *= val */
void cmul2_tdvector(TDVector c, double val[TDSIZE])
{
	long int i, index;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d c4[TDSIZE], val4[TDSIZE], tmp4[TDSIZE];

	val4[0] = _mm256_set1_pd(val[0]);
	val4[1] = _mm256_set1_pd(val[1]);
	val4[2] = _mm256_set1_pd(val[2]);
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		//set_tdvector_i(c, i, val * get_tdvector_i(a, i));
		c4[0] = _mm256_load_pd(&(c->element[0][index]));
		c4[1] = _mm256_load_pd(&(c->element[1][index]));
		c4[2] = _mm256_load_pd(&(c->element[2][index]));

		_bncavx2_rtd_mul(tmp4, val4, c4);

		_mm256_store_pd(&(c->element[0][index]), tmp4[0]);
		_mm256_store_pd(&(c->element[1][index]), tmp4[1]);
		_mm256_store_pd(&(c->element[2][index]), tmp4[2]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d c8[TDSIZE], val8[TDSIZE], tmp8[TDSIZE];

	val8[0] = _mm512_set1_pd(val[0]);
	val8[1] = _mm512_set1_pd(val[1]);
	val8[2] = _mm512_set1_pd(val[2]);
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		//set_tdvector_i(c, i, val * get_tdvector_i(a, i));
		c8[0] = _mm512_load_pd(&(c->element[0][index]));
		c8[1] = _mm512_load_pd(&(c->element[1][index]));
		c8[2] = _mm512_load_pd(&(c->element[2][index]));

		_bncavx512_rtd_mul(tmp8, val8, c8);

		_mm512_store_pd(&(c->element[0][index]), tmp8[0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[1]);
		_mm512_store_pd(&(c->element[2][index]), tmp8[2]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t c2_0, c2_1, c2_2;
	svfloat64_t val2_0, val2_1, val2_2;
	svfloat64_t tmp2_0, tmp2_1, tmp2_2;

    val2_0 = svdup_f64(val[0]);
    val2_1 = svdup_f64(val[1]);
    val2_2 = svdup_f64(val[2]);
    for(index = 0; index < c->real_dim; index += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        c2_0 = svld1_f64(pg, &(c->element[0][index]));
        c2_1 = svld1_f64(pg, &(c->element[1][index]));
        c2_2 = svld1_f64(pg, &(c->element[2][index]));

        _bncsve2_rtd_mul(pg, &tmp2_0, &tmp2_1, &tmp2_2, val2_0, val2_1, val2_2, c2_0, c2_1, c2_2);

        svst1_f64(pg, &(c->element[0][index]), tmp2_0);
        svst1_f64(pg, &(c->element[1][index]), tmp2_1);
        svst1_f64(pg, &(c->element[2][index]), tmp2_2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm NEON
    float64x2_t c2[TDSIZE], val2[TDSIZE], tmp2[TDSIZE];

    val2[0] = vdupq_n_f64(val[0]);
    val2[1] = vdupq_n_f64(val[1]);
    val2[2] = vdupq_n_f64(val[2]);
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        c2[0] = vld1q_f64(&(c->element[0][index]));
        c2[1] = vld1q_f64(&(c->element[1][index]));
        c2[2] = vld1q_f64(&(c->element[2][index]));

        _bncneon_rtd_mul(tmp2, val2, c2);

        vst1q_f64(&(c->element[0][index]), tmp2[0]);
        vst1q_f64(&(c->element[1][index]), tmp2[1]);
        vst1q_f64(&(c->element[2][index]), tmp2[2]);
    }
#else // others
	double tmp[TDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rtd_mul(tmp, val, get_tdvector_i(c, i));
		set_tdvector_i(c, i, tmp);
	}
#endif // __AVX2__	double tmp[TDSIZE];

}

/* c = a + val * b */
void add_cmul_tdvector(TDVector c, TDVector a, double val[TDSIZE], TDVector b)
{
	long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_cmul_tdvector\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[TDSIZE], b4[TDSIZE], c4[TDSIZE], val4[TDSIZE];

	val4[0] = _mm256_set1_pd(val[0]);
	val4[1] = _mm256_set1_pd(val[1]);
	val4[2] = _mm256_set1_pd(val[2]);
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		a4[0] = _mm256_load_pd(&(a->element[0][index]));
		a4[1] = _mm256_load_pd(&(a->element[1][index]));
		a4[2] = _mm256_load_pd(&(a->element[2][index]));
		b4[0] = _mm256_load_pd(&(b->element[0][index]));
		b4[1] = _mm256_load_pd(&(b->element[1][index]));
		b4[2] = _mm256_load_pd(&(b->element[2][index]));

//		rtd_mul(tmp, val, get_tdvector_i(b, i));
//		rtd_add(tmp, tmp, get_tdvector_i(a, i));
		_bncavx2_rtd_mul(c4, val4, b4);
		_bncavx2_rtd_add(c4, a4, c4);

		_mm256_store_pd(&(c->element[0][index]), c4[0]);
		_mm256_store_pd(&(c->element[1][index]), c4[1]);
		_mm256_store_pd(&(c->element[2][index]), c4[2]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[TDSIZE], b8[TDSIZE], c8[TDSIZE], val8[TDSIZE];

	val8[0] = _mm512_set1_pd(val[0]);
	val8[1] = _mm512_set1_pd(val[1]);
	val8[2] = _mm512_set1_pd(val[2]);
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		a8[0] = _mm512_load_pd(&(a->element[0][index]));
		a8[1] = _mm512_load_pd(&(a->element[1][index]));
		a8[2] = _mm512_load_pd(&(a->element[2][index]));
		b8[0] = _mm512_load_pd(&(b->element[0][index]));
		b8[1] = _mm512_load_pd(&(b->element[1][index]));
		b8[2] = _mm512_load_pd(&(b->element[2][index]));

//		rtd_mul(tmp, val, get_tdvector_i(b, i));
//		rtd_add(tmp, tmp, get_tdvector_i(a, i));
		_bncavx512_rtd_mul(c8, val8, b8);
		_bncavx512_rtd_add(c8, a8, c8);

		_mm512_store_pd(&(c->element[0][index]), c8[0]);
		_mm512_store_pd(&(c->element[1][index]), c8[1]);
		_mm512_store_pd(&(c->element[2][index]), c8[2]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t a2_0, a2_1, a2_2;
	svfloat64_t b2_0, b2_1, b2_2;
	svfloat64_t c2_0, c2_1, c2_2;
	svfloat64_t val2_0, val2_1, val2_2;

    val2_0 = svdup_f64(val[0]);
    val2_1 = svdup_f64(val[1]);
    val2_2 = svdup_f64(val[2]);
    for(index = 0; index < c->real_dim; index += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        a2_0 = svld1_f64(pg, &(a->element[0][index]));
        a2_1 = svld1_f64(pg, &(a->element[1][index]));
        a2_2 = svld1_f64(pg, &(a->element[2][index]));
        b2_0 = svld1_f64(pg, &(b->element[0][index]));
        b2_1 = svld1_f64(pg, &(b->element[1][index]));
        b2_2 = svld1_f64(pg, &(b->element[2][index]));

        _bncsve2_rtd_mul(pg, &c2_0, &c2_1, &c2_2, val2_0, val2_1, val2_2, b2_0, b2_1, b2_2);
        _bncsve2_rtd_add(pg, &c2_0, &c2_1, &c2_2, a2_0, a2_1, a2_2, c2_0, c2_1, c2_2);

        svst1_f64(pg, &(c->element[0][index]), c2_0);
        svst1_f64(pg, &(c->element[1][index]), c2_1);
        svst1_f64(pg, &(c->element[2][index]), c2_2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm NEON
    float64x2_t a2[TDSIZE], b2[TDSIZE], c2[TDSIZE], val2[TDSIZE];

    val2[0] = vdupq_n_f64(val[0]);
    val2[1] = vdupq_n_f64(val[1]);
    val2[2] = vdupq_n_f64(val[2]);
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        a2[0] = vld1q_f64(&(a->element[0][index]));
        a2[1] = vld1q_f64(&(a->element[1][index]));
        a2[2] = vld1q_f64(&(a->element[2][index]));
        b2[0] = vld1q_f64(&(b->element[0][index]));
        b2[1] = vld1q_f64(&(b->element[1][index]));
        b2[2] = vld1q_f64(&(b->element[2][index]));

        _bncneon_rtd_mul(c2, val2, b2);
        _bncneon_rtd_add(c2, a2, c2);

        vst1q_f64(&(c->element[0][index]), c2[0]);
        vst1q_f64(&(c->element[1][index]), c2[1]);
        vst1q_f64(&(c->element[2][index]), c2[2]);
    }
#else // others
	double tmp[TDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rtd_mul(tmp, val, get_tdvector_i(b, i));
		rtd_add(tmp, tmp, get_tdvector_i(a, i));
		set_tdvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

// 2025-02-19(Wed)
/* c = a - val * b */
void sub_cmul_tdvector(TDVector c, TDVector a, double val[TDSIZE], TDVector b)
{
	long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_cmul_tdvector\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[TDSIZE], b4[TDSIZE], c4[TDSIZE], val4[TDSIZE];

	val4[0] = _mm256_set1_pd(val[0]);
	val4[1] = _mm256_set1_pd(val[1]);
	val4[2] = _mm256_set1_pd(val[2]);
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		a4[0] = _mm256_load_pd(&(a->element[0][index]));
		a4[1] = _mm256_load_pd(&(a->element[1][index]));
		a4[2] = _mm256_load_pd(&(a->element[2][index]));
		b4[0] = _mm256_load_pd(&(b->element[0][index]));
		b4[1] = _mm256_load_pd(&(b->element[1][index]));
		b4[2] = _mm256_load_pd(&(b->element[2][index]));

//		rtd_mul(tmp, val, get_tdvector_i(b, i));
//		rtd_add(tmp, tmp, get_tdvector_i(a, i));
		_bncavx2_rtd_mul(c4, val4, b4);
		//_bncavx2_rtd_add(c4, a4, c4);
		_bncavx2_rtd_sub(c4, a4, c4);

		_mm256_store_pd(&(c->element[0][index]), c4[0]);
		_mm256_store_pd(&(c->element[1][index]), c4[1]);
		_mm256_store_pd(&(c->element[2][index]), c4[2]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[TDSIZE], b8[TDSIZE], c8[TDSIZE], val8[TDSIZE];

	val8[0] = _mm512_set1_pd(val[0]);
	val8[1] = _mm512_set1_pd(val[1]);
	val8[2] = _mm512_set1_pd(val[2]);
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		a8[0] = _mm512_load_pd(&(a->element[0][index]));
		a8[1] = _mm512_load_pd(&(a->element[1][index]));
		a8[2] = _mm512_load_pd(&(a->element[2][index]));
		b8[0] = _mm512_load_pd(&(b->element[0][index]));
		b8[1] = _mm512_load_pd(&(b->element[1][index]));
		b8[2] = _mm512_load_pd(&(b->element[2][index]));

//		rtd_mul(tmp, val, get_tdvector_i(b, i));
//		rtd_add(tmp, tmp, get_tdvector_i(a, i));
		_bncavx512_rtd_mul(c8, val8, b8);
		//_bncavx512_rtd_add(c8, a8, c8);
		_bncavx512_rtd_sub(c8, a8, c8);

		_mm512_store_pd(&(c->element[0][index]), c8[0]);
		_mm512_store_pd(&(c->element[1][index]), c8[1]);
		_mm512_store_pd(&(c->element[2][index]), c8[2]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t a2_0, a2_1, a2_2;
	svfloat64_t b2_0, b2_1, b2_2;
	svfloat64_t c2_0, c2_1, c2_2;
	svfloat64_t val2_0, val2_1, val2_2;

    val2_0 = svdup_f64(val[0]);
    val2_1 = svdup_f64(val[1]);
    val2_2 = svdup_f64(val[2]);
    for(index = 0; index < c->real_dim; index += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        a2_0 = svld1_f64(pg, &(a->element[0][index]));
        a2_1 = svld1_f64(pg, &(a->element[1][index]));
        a2_2 = svld1_f64(pg, &(a->element[2][index]));
        b2_0 = svld1_f64(pg, &(b->element[0][index]));
        b2_1 = svld1_f64(pg, &(b->element[1][index]));
        b2_2 = svld1_f64(pg, &(b->element[2][index]));

        _bncsve2_rtd_mul(pg, &c2_0, &c2_1, &c2_2, val2_0, val2_1, val2_2, b2_0, b2_1, b2_2);     // val * b
        _bncsve2_rtd_neg(pg, &c2_0, &c2_1, &c2_2, c2_0, c2_1, c2_2);
		_bncsve2_rtd_add(pg, &c2_0, &c2_1, &c2_2, a2_0, a2_1, a2_2, c2_0, c2_1, c2_2);       // a - (val * b)

        svst1_f64(pg, &(c->element[0][index]), c2_0);
        svst1_f64(pg, &(c->element[1][index]), c2_1);
        svst1_f64(pg, &(c->element[2][index]), c2_2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm NEON
    float64x2_t a2[TDSIZE], b2[TDSIZE], c2[TDSIZE], val2[TDSIZE];

    val2[0] = vdupq_n_f64(val[0]);
    val2[1] = vdupq_n_f64(val[1]);
    val2[2] = vdupq_n_f64(val[2]);
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        a2[0] = vld1q_f64(&(a->element[0][index]));
        a2[1] = vld1q_f64(&(a->element[1][index]));
        a2[2] = vld1q_f64(&(a->element[2][index]));
        b2[0] = vld1q_f64(&(b->element[0][index]));
        b2[1] = vld1q_f64(&(b->element[1][index]));
        b2[2] = vld1q_f64(&(b->element[2][index]));

        _bncneon_rtd_mul(c2, val2, b2);     // val * b
        _bncneon_rtd_sub(c2, a2, c2);       // a - (val * b)

        vst1q_f64(&(c->element[0][index]), c2[0]);
        vst1q_f64(&(c->element[1][index]), c2[1]);
        vst1q_f64(&(c->element[2][index]), c2[2]);
    }
#else // others
	double tmp[TDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rtd_mul(tmp, val, get_tdvector_i(b, i));
		//rtd_add(tmp, tmp, get_tdvector_i(a, i));
		rtd_sub(tmp, get_tdvector_i(a, i), tmp);
		set_tdvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* (a, b) */
void ip_tdvector(double ret[TDSIZE], TDVector a, TDVector b)
{
	long int i, index;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: ip_tdvector\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[TDSIZE], b4[TDSIZE], ret4[TDSIZE], tmp4[TDSIZE];

	_bncavx2_set0_td(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		a4[0] = _mm256_load_pd(&(a->element[0][index]));
		a4[1] = _mm256_load_pd(&(a->element[1][index]));
		a4[2] = _mm256_load_pd(&(a->element[2][index]));
		b4[0] = _mm256_load_pd(&(b->element[0][index]));
		b4[1] = _mm256_load_pd(&(b->element[1][index]));
		b4[2] = _mm256_load_pd(&(b->element[2][index]));

//		rtd_mul(tmp, get_tdvector_i(a, i), get_tdvector_i(b, i));
//		rtd_add(ret, ret, tmp);
		_bncavx2_rtd_mul(tmp4, a4, b4);
		_bncavx2_rtd_add(ret4, ret4, tmp4);
	}
	_bncavx2_rtd_sum256d(ret, ret4);
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[TDSIZE], b8[TDSIZE], ret8[TDSIZE], tmp8[TDSIZE];

	_bncavx512_set0_td(ret8);
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		a8[0] = _mm512_load_pd(&(a->element[0][index]));
		a8[1] = _mm512_load_pd(&(a->element[1][index]));
		a8[2] = _mm512_load_pd(&(a->element[2][index]));
		b8[0] = _mm512_load_pd(&(b->element[0][index]));
		b8[1] = _mm512_load_pd(&(b->element[1][index]));
		b8[2] = _mm512_load_pd(&(b->element[2][index]));

//		rtd_mul(tmp, get_tdvector_i(a, i), get_tdvector_i(b, i));
//		rtd_add(ret, ret, tmp);
		_bncavx512_rtd_mul(tmp8, a8, b8);
		_bncavx512_rtd_add(ret8, ret8, tmp8);
	}
	_bncavx512_rtd_sum512d(ret, ret8);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)

    
	svfloat64_t a2_0, a2_1, a2_2;
	svfloat64_t b2_0, b2_1, b2_2;
	svfloat64_t ret2_0, ret2_1, ret2_2;
	svfloat64_t tmp2_0, tmp2_1, tmp2_2;

    _bncsve2_rtd_set0(&ret2_0, &ret2_1, &ret2_2);
    for(index = 0; index < a->real_dim; index += (long int)svcntd())
    {
    		svbool_t pg = svwhilelt_b64_s32((int)index, (int)a->real_dim);
        a2_0 = svld1_f64(pg, &(a->element[0][index]));
        a2_1 = svld1_f64(pg, &(a->element[1][index]));
        a2_2 = svld1_f64(pg, &(a->element[2][index]));
        b2_0 = svld1_f64(pg, &(b->element[0][index]));
        b2_1 = svld1_f64(pg, &(b->element[1][index]));
        b2_2 = svld1_f64(pg, &(b->element[2][index]));

        _bncsve2_rtd_mul(svptrue_b64(), &tmp2_0, &tmp2_1, &tmp2_2, a2_0, a2_1, a2_2, b2_0, b2_1, b2_2);
        _bncsve2_rtd_add(svptrue_b64(), &ret2_0, &ret2_1, &ret2_2, ret2_0, ret2_1, ret2_2, tmp2_0, tmp2_1, tmp2_2);
    }
    _bncsve2_rtd_sum128d(svptrue_b64(), ret, ret2_0, ret2_1, ret2_2);
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)

    float64x2_t a2[TDSIZE], b2[TDSIZE], ret2[TDSIZE], tmp2[TDSIZE];

    _bncneon_set0_td(ret2);
    for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
    {
        a2[0] = vld1q_f64(&(a->element[0][index]));
        a2[1] = vld1q_f64(&(a->element[1][index]));
        a2[2] = vld1q_f64(&(a->element[2][index]));
        b2[0] = vld1q_f64(&(b->element[0][index]));
        b2[1] = vld1q_f64(&(b->element[1][index]));
        b2[2] = vld1q_f64(&(b->element[2][index]));

        _bncneon_rtd_mul(tmp2, a2, b2);
        _bncneon_rtd_add(ret2, ret2, tmp2);
    }
    _bncneon_rtd_sum128d(ret, ret2);
#else // others
	double tmp[TDSIZE];

	set0_td(ret);
	for(i = 0; i < a->dim; i++)
	{
		rtd_mul(tmp, get_tdvector_i(a, i), get_tdvector_i(b, i));
		rtd_add(ret, ret, tmp);
	}
#endif // __AVX2__

	return;
}

/* c := a */
void subst_tdvector(TDVector c, TDVector a)
{
	long int i;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
//	for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i));
		_mm256_store_pd(&(c->element[0][i]), _mm256_load_pd(&(a->element[0][i])));
		_mm256_store_pd(&(c->element[1][i]), _mm256_load_pd(&(a->element[1][i])));
		_mm256_store_pd(&(c->element[2][i]), _mm256_load_pd(&(a->element[2][i])));
	}
#elif defined(__AVX512F__) // __AVX512F__
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i));
		_mm512_store_pd(&(c->element[0][i]), _mm512_load_pd(&(a->element[0][i])));
		_mm512_store_pd(&(c->element[1][i]), _mm512_load_pd(&(a->element[1][i])));
		_mm512_store_pd(&(c->element[2][i]), _mm512_load_pd(&(a->element[2][i])));
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)

    for(i = 0; i < a->real_dim; i += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(a->real_dim));
        svst1_f64(pg, &(c->element[0][i]), svld1_f64(pg, &(a->element[0][i])));
        svst1_f64(pg, &(c->element[1][i]), svld1_f64(pg, &(a->element[1][i])));
        svst1_f64(pg, &(c->element[2][i]), svld1_f64(pg, &(a->element[2][i])));
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)

    for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
    {
        vst1q_f64(&(c->element[0][i]), vld1q_f64(&(a->element[0][i])));
        vst1q_f64(&(c->element[1][i]), vld1q_f64(&(a->element[1][i])));
        vst1q_f64(&(c->element[2][i]), vld1q_f64(&(a->element[2][i])));
    }
#else // others
	for(i = 0; i < a->dim; i++)
		set_tdvector_i(c, i, get_tdvector_i(a, i));
#endif // __AVX2__
}

/* c := -a */
void neg_tdvector(TDVector c, TDVector a)
{
	long int i;
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp[TDSIZE];

	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//rtd_neg(tmp, get_tdvector_i(a, i));
		//set_tdvector_i(c, i, tmp);
		tmp[0] = _bncavx2_dneg(_mm256_load_pd(&(a->element[0][i])));
		tmp[1] = _bncavx2_dneg(_mm256_load_pd(&(a->element[1][i])));
		tmp[2] = _bncavx2_dneg(_mm256_load_pd(&(a->element[2][i])));
		_mm256_store_pd(&(c->element[0][i]), tmp[0]);
		_mm256_store_pd(&(c->element[1][i]), tmp[1]);
		_mm256_store_pd(&(c->element[2][i]), tmp[2]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp[TDSIZE];

	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//rtd_neg(tmp, get_tdvector_i(a, i));
		//set_tdvector_i(c, i, tmp);
		tmp[0] = _bncavx512_dneg(_mm512_load_pd(&(a->element[0][i])));
		tmp[1] = _bncavx512_dneg(_mm512_load_pd(&(a->element[1][i])));
		tmp[2] = _bncavx512_dneg(_mm512_load_pd(&(a->element[2][i])));
		_mm512_store_pd(&(c->element[0][i]), tmp[0]);
		_mm512_store_pd(&(c->element[1][i]), tmp[1]);
		_mm512_store_pd(&(c->element[2][i]), tmp[2]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)

    
	svfloat64_t tmp2_0, tmp2_1, tmp2_2;

    for(i = 0; i < a->real_dim; i += (long int)svcntd())
    {
    		svbool_t pg = svwhilelt_b64_s32((int)i, (int)a->real_dim);
        tmp2_0 = svld1_f64(pg, &(a->element[0][i]));
        tmp2_1 = svld1_f64(pg, &(a->element[1][i]));
        tmp2_2 = svld1_f64(pg, &(a->element[2][i]));

        _bncsve2_rtd_neg(svptrue_b64(), &tmp2_0, &tmp2_1, &tmp2_2, tmp2_0, tmp2_1, tmp2_2);

        svst1_f64(pg, &(c->element[0][i]), tmp2_0);
        svst1_f64(pg, &(c->element[1][i]), tmp2_1);
        svst1_f64(pg, &(c->element[2][i]), tmp2_2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)

    float64x2_t tmp2[TDSIZE];

    for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
    {
        tmp2[0] = vld1q_f64(&(a->element[0][i]));
        tmp2[1] = vld1q_f64(&(a->element[1][i]));
        tmp2[2] = vld1q_f64(&(a->element[2][i]));

        _bncneon_rtd_neg(tmp2, tmp2);

        vst1q_f64(&(c->element[0][i]), tmp2[0]);
        vst1q_f64(&(c->element[1][i]), tmp2[1]);
        vst1q_f64(&(c->element[2][i]), tmp2[2]);
    }
#else // others
	double tmp[TDSIZE];

	for(i = 0; i < a->dim; i++)
	{
		rtd_neg(tmp, get_tdvector_i(a, i));
		set_tdvector_i(c, i, tmp);
	}
#endif // __AVX2__
}


/* ||a||_1 */
void norm1_tdvector(double ret[TDSIZE], TDVector a)
{
	long int i, index, dim;

	dim = a->dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d vec4[TDSIZE], ret4[TDSIZE], tmp4[TDSIZE];

	_bncavx2_set0_td(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		vec4[0] = _mm256_load_pd(&(a->element[0][index]));
		vec4[1] = _mm256_load_pd(&(a->element[1][index]));
		vec4[2] = _mm256_load_pd(&(a->element[2][index]));

		//rtd_abs(tmp, get_tdvector_i(a, i));
		//rtd_add(ret, ret, tmp);
		_bncavx2_rtd_abs(tmp4, vec4);
		_bncavx2_rtd_add(ret4, ret4, tmp4);
	}
	_bncavx2_rtd_abssum256d(ret, ret4);
#elif defined(__AVX512F__) // __AVX512F__
	__m512d vec8[TDSIZE], ret8[TDSIZE], tmp8[TDSIZE];

	_bncavx512_set0_td(ret8);
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		vec8[0] = _mm512_load_pd(&(a->element[0][index]));
		vec8[1] = _mm512_load_pd(&(a->element[1][index]));
		vec8[2] = _mm512_load_pd(&(a->element[2][index]));

		//rtd_abs(tmp, get_tdvector_i(a, i));
		//rtd_add(ret, ret, tmp);
		_bncavx512_rtd_abs(tmp8, vec8);
		_bncavx512_rtd_add(ret8, ret8, tmp8);
	}
	_bncavx512_rtd_abssum512d(ret, ret8);
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)

    
	svfloat64_t vec2_0, vec2_1, vec2_2;
	svfloat64_t ret2_0, ret2_1, ret2_2;
	svfloat64_t tmp2_0, tmp2_1, tmp2_2;

    _bncsve2_rtd_set0(&ret2_0, &ret2_1, &ret2_2);
    for(index = 0; index < a->real_dim; index += (long int)svcntd())
    {
    		svbool_t pg = svwhilelt_b64_s32((int)index, (int)a->real_dim);
        vec2_0 = svld1_f64(pg, &(a->element[0][index]));
        vec2_1 = svld1_f64(pg, &(a->element[1][index]));
        vec2_2 = svld1_f64(pg, &(a->element[2][index]));

        _bncsve2_rtd_abs(svptrue_b64(), &tmp2_0, &tmp2_1, &tmp2_2, vec2_0, vec2_1, vec2_2);
        _bncsve2_rtd_add(svptrue_b64(), &ret2_0, &ret2_1, &ret2_2, ret2_0, ret2_1, ret2_2, tmp2_0, tmp2_1, tmp2_2);
    }
    _bncsve2_rtd_abssum128d(svptrue_b64(), ret, ret2_0, ret2_1, ret2_2);
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)

    float64x2_t vec2[TDSIZE], ret2[TDSIZE], tmp2[TDSIZE];

    _bncneon_set0_td(ret2);
    for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
    {
        vec2[0] = vld1q_f64(&(a->element[0][index]));
        vec2[1] = vld1q_f64(&(a->element[1][index]));
        vec2[2] = vld1q_f64(&(a->element[2][index]));

        _bncneon_rtd_abs(tmp2, vec2);
        _bncneon_rtd_add(ret2, ret2, tmp2);
    }
    _bncneon_rtd_abssum128d(ret, ret2);
#else // others
	double tmp[TDSIZE];

	set0_td(ret);
	for(i = 0; i < a->dim; i++)
	{
		rtd_abs(tmp, get_tdvector_i(a, i));
		rtd_add(ret, ret, tmp);
	}
#endif // __AVX2__

	return;
}

/* ||a||_infty */
void normi_tdvector(double ret[TDSIZE], TDVector a)
{
	double tmp[TDSIZE];
	long int i;

	rtd_abs(ret, get_tdvector_i(a, 0));
	for(i = 1; i < a->dim; i++)
	{
		rtd_abs(tmp, get_tdvector_i(a, i));
		if(rtd_cmp(ret, tmp) < 0)
			rtd_set(ret, tmp);
	}

	return;
}

// Euclid norm
void norm2_tdvector(double ret[TDSIZE], TDVector vec)
{
	long int i, index, dim, real_dim;
	double tmp[TDSIZE];

	dim = vec->dim;
	real_dim = vec->real_dim;
	set0_td(tmp);

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d vec4[TDSIZE], ret4[TDSIZE], tmp4[TDSIZE];

	_bncavx2_set0_td(ret4);
	for(index = 0; index < vec->real_dim; index += _BNC_D_WIDTH)
	{
		vec4[0] = _mm256_load_pd(&(vec->element[0][index]));
		vec4[1] = _mm256_load_pd(&(vec->element[1][index]));
		vec4[2] = _mm256_load_pd(&(vec->element[2][index]));

		_bncavx2_rtd_mul(tmp4, vec4, vec4);
		_bncavx2_rtd_add(ret4, ret4, tmp4);
	}
	//_bncavx2_rtd_norm256d(ret, ret4);
	_bncavx2_rtd_sum256d(tmp, ret4);
#elif defined(__AVX512F__) // __AVX512F__
	__m512d vec8[TDSIZE], ret8[TDSIZE], tmp8[TDSIZE];

	_bncavx512_set0_td(ret8);
	for(index = 0; index < vec->real_dim; index += _BNC_D_WIDTH)
	{
		vec8[0] = _mm512_load_pd(&(vec->element[0][index]));
		vec8[1] = _mm512_load_pd(&(vec->element[1][index]));
		vec8[2] = _mm512_load_pd(&(vec->element[2][index]));

		_bncavx512_rtd_mul(tmp8, vec8, vec8);
		_bncavx512_rtd_add(ret8, ret8, tmp8);
	}
	//_bncavx512_rtd_norm512d(ret, ret8);
	_bncavx512_rtd_sum512d(tmp, ret8);
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)

    
	svfloat64_t vec2_0, vec2_1, vec2_2;
	svfloat64_t ret2_0, ret2_1, ret2_2;
	svfloat64_t tmp2_0, tmp2_1, tmp2_2;

    _bncsve2_rtd_set0(&ret2_0, &ret2_1, &ret2_2);
    for(index = 0; index < vec->real_dim; index += (long int)svcntd())
    {
    		svbool_t pg = svwhilelt_b64_s32((int)index, (int)vec->real_dim);
        vec2_0 = svld1_f64(pg, &(vec->element[0][index]));
        vec2_1 = svld1_f64(pg, &(vec->element[1][index]));
        vec2_2 = svld1_f64(pg, &(vec->element[2][index]));

        _bncsve2_rtd_mul(svptrue_b64(), &tmp2_0, &tmp2_1, &tmp2_2, vec2_0, vec2_1, vec2_2, vec2_0, vec2_1, vec2_2);
        _bncsve2_rtd_add(svptrue_b64(), &ret2_0, &ret2_1, &ret2_2, ret2_0, ret2_1, ret2_2, tmp2_0, tmp2_1, tmp2_2);
    }
    _bncsve2_rtd_sum128d(svptrue_b64(), tmp, ret2_0, ret2_1, ret2_2);
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)

    float64x2_t vec2[TDSIZE], ret2[TDSIZE], tmp2[TDSIZE];

    _bncneon_set0_td(ret2);
    for(index = 0; index < vec->real_dim; index += _BNC_D_WIDTH)
    {
        vec2[0] = vld1q_f64(&(vec->element[0][index]));
        vec2[1] = vld1q_f64(&(vec->element[1][index]));
        vec2[2] = vld1q_f64(&(vec->element[2][index]));

        _bncneon_rtd_mul(tmp2, vec2, vec2);
        _bncneon_rtd_add(ret2, ret2, tmp2);
    }
    _bncneon_rtd_sum128d(tmp, ret2);
#else // others
	//c_dd_copy_d((double)0.0, tmp);
	//c_dd_copy_d((double)0.0, ret);
	rtd_set0(tmp);
	rtd_set0(ret);

	for(i = 0; i < dim ; i++)
	{
		//c_dd_sqr(GET_TDVECTOR_I(vec, i), tmp);
		//c_dd_add(tmp, ret, ret);
		rtd_mul(ret, get_tdvector_i(vec, i), get_tdvector_i(vec, i));
		rtd_add(tmp, tmp, ret);
	}

	//c_td_sqrt(ret, tmp);
	//c_td_copy(tmp, ret);
	//rtd_set(tmp, ret);
#endif // __AVX2__

#ifdef USE_GMP
	rtd_sqrt_mpfr(ret, tmp);
#else // USE_GMP
	rtd_sqrt(ret, tmp);
#endif // USE_GMP

}

// set a zero matrix
//void set0_tdmatrix(TDMatrix mat)
void set0_tdmatrix(TDMatrix mat)
{
	long int i;
	long int real_total_dim;

	real_total_dim = mat->real_row_dim * mat->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&(mat->element[0][i]), zero4);
		_mm256_store_pd(&(mat->element[1][i]), zero4);
		_mm256_store_pd(&(mat->element[2][i]), zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&(mat->element[0][i]), zero8);
		_mm512_store_pd(&(mat->element[1][i]), zero8);
		_mm512_store_pd(&(mat->element[2][i]), zero8);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)

    svfloat64_t zero2 = svdup_f64(0.0);

    for(i = 0; i < real_total_dim; i += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(real_total_dim));
        svst1_f64(pg, &(mat->element[0][i]), zero2);
        svst1_f64(pg, &(mat->element[1][i]), zero2);
        svst1_f64(pg, &(mat->element[2][i]), zero2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)

    float64x2_t zero2 = vdupq_n_f64(0.0);

    for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
    {
        vst1q_f64(&(mat->element[0][i]), zero2);
        vst1q_f64(&(mat->element[1][i]), zero2);
        vst1q_f64(&(mat->element[2][i]), zero2);
    }
#else // others
	for(i = 0; i < real_total_dim; i++)
	{
		mat->element[0][i] = 0.0;
		mat->element[1][i] = 0.0;	
		mat->element[2][i] = 0.0;	
	}
#endif // __AVX2__
}

// initialize tdvector
TDMatrix init_tdmatrix(long int row_dim, long int col_dim)
{
	long int row_index, col_index, i;
	long int real_row_dim, real_col_dim, real_total_dim;
	TDMatrix ret = NULL;

	//ret = (TDMatrix)BNC_MALLOC(sizeof(tdmatrix));
	ret = (TDMatrix)malloc(sizeof(tdmatrix));
	if(ret == NULL)
		return ret;

	// real_dim is the nearest positive multiplier of _BNC_D_WIDTH
	real_row_dim = (long int)ceil((double)(row_dim) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
	real_col_dim = (long int)ceil((double)(col_dim) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
	real_total_dim = real_row_dim * real_col_dim;

	//printf("init_tdmatrix(%ld, %ld) %ld calloc\n", row_dim, col_dim, real_total_dim);
	ret->element[0] = (double *)BNC_CALLOC(real_total_dim, sizeof(double));
	if(ret->element[0] == NULL)
	{ 	free(ret);
		return NULL;
	}
	ret->element[1] = (double *)BNC_CALLOC(real_total_dim, sizeof(double));
	if(ret->element[1] == NULL)
	{
		free(ret->element[0]);
		free(ret);
		return NULL;
	}
	ret->element[2] = (double *)BNC_CALLOC(real_total_dim, sizeof(double));
	if(ret->element[2] == NULL)
	{
		free(ret->element[0]);
		free(ret->element[1]);
		free(ret);
		return NULL;
	}

	//printf("init_tdmatrix(%ld, %ld) calloc\n", row_dim, col_dim);
	/* All 0 */
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&(ret->element[0][i]), zero4);
		_mm256_store_pd(&(ret->element[1][i]), zero4);
		_mm256_store_pd(&(ret->element[2][i]), zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&(ret->element[0][i]), zero8);
		_mm512_store_pd(&(ret->element[1][i]), zero8);
		_mm512_store_pd(&(ret->element[2][i]), zero8);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)

    svfloat64_t zero2 = svdup_f64(0.0);

    for(i = 0; i < real_total_dim; i += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(real_total_dim));
        svst1_f64(pg, &(ret->element[0][i]), zero2);
        svst1_f64(pg, &(ret->element[1][i]), zero2);
        svst1_f64(pg, &(ret->element[2][i]), zero2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)

    float64x2_t zero2 = vdupq_n_f64(0.0);

    for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
    {
        vst1q_f64(&(ret->element[0][i]), zero2);
        vst1q_f64(&(ret->element[1][i]), zero2);
        vst1q_f64(&(ret->element[2][i]), zero2);
    }
#else // others
	for(i = 0; i < real_total_dim; i++)
	{
		ret->element[0][i] = 0.0;
		ret->element[1][i] = 0.0;
		ret->element[2][i] = 0.0;
	}
#endif // __AVX2__

	ret->real_row_dim = real_row_dim;
	ret->real_col_dim = real_col_dim;
	ret->row_dim = row_dim;
	ret->col_dim = col_dim;

	return ret;
}

// free tdvector
void free_tdmatrix(TDMatrix mat)
{
	long int i;

	for(i = 0; i < TDSIZE; i++)
		free(mat->element[i]);

	free(mat);
}

// print tdvector
void print_tdmatrix(TDMatrix mat)
{
	long int row_index, col_index;

	for(row_index = 0; row_index < mat->row_dim; row_index++)
	{
		for(col_index = 0; col_index < mat->col_dim; col_index++)
		{
			printf("%ld, %ld: ", row_index, col_index);
	//		c_dd_write((vec->element + index * TDSIZE));
			//c_td_write(GET_TDMATRIX_IJ(mat, row_index, col_index));
			rtd_out_str(GET_TDMATRIX_IJ(mat, row_index, col_index));
			printf("\n");
		}
	}
}

// TDMatrix mat -> tdfloat array
void set_tdfloat_tdmat(tdfloat ret[], int ret_dim, TDMatrix mat)
{
    int total_index, j, total_dim;
	long int row_index, col_index;

    total_dim = (ret_dim < (mat->row_dim * mat->col_dim)) ? ret_dim : (mat->row_dim * mat->col_dim);

	total_index = 0;
    for(row_index = 0; row_index < mat->row_dim; row_index++)
    {
		for(col_index = 0; (col_index < mat->col_dim) && (total_index < total_dim); col_index++)
		{
			for(j = 0; j < TDSIZE; j++)
				ret[total_index].val[j] = mat->element[j][(row_index * mat->real_col_dim) + col_index];

			total_index++;
		}
    }
}

// tdfloat array -> TDmatrix ret
void set_tdmatrix_tdfloat(TDMatrix ret, tdfloat array[], int array_dim)
{
    int total_index, j, total_dim;
	long int row_index, col_index;

    total_dim = (array_dim < (ret->row_dim * ret->col_dim)) ? array_dim : (ret->row_dim * ret->col_dim);

 	total_index = 0;
    for(row_index = 0; row_index < ret->row_dim; row_index++)
    {
		for(col_index = 0; (col_index < ret->col_dim) && (total_index < total_dim); col_index++)
		{
			for(j = 0; j < TDSIZE; j++)
				ret->element[j][(row_index * ret->real_col_dim) + col_index] = array[total_index].val[j];

			total_index++;
		}
    }
}


// matrix multiplication
// ret := A * B
void mul_tdmatrix(TDMatrix ret, TDMatrix a, TDMatrix b)
{
	long int i, j, k;

	/* dimension check */
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: mul_tdmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret->row_dim, ret->col_dim, a->row_dim, a->col_dim, b->row_dim, b->col_dim);
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
//#ifdef BNC_USE_GCC
	long real_row_dim, real_col_dim, real_mid_dim;
	double cijval[4][TDSIZE];
    __m256d cij[TDSIZE], aik[TDSIZE], bkj[TDSIZE], tmp_mul[TDSIZE];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j++)
        {
            //rtd_set_ui(cij.val, 0UL);
            cij[0] = _mm256_setzero_pd();
            cij[1] = _mm256_setzero_pd();
            cij[2] = _mm256_setzero_pd();
			for(k = 0; k < real_mid_dim; k += _BNC_D_WIDTH)
            {
            /*
                aik[0].val[0] = a->element[0][i * mid_dim + k];
                aik[1].val[0] = a->element[0][i * mid_dim + k + 1];
                aik[2].val[0] = a->element[0][i * mid_dim + k + 2];
                aik[3].val[0] = a->element[0][i * mid_dim + k + 3];
            */
                aik[0] = _mm256_load_pd(&(a->element[0][i * real_mid_dim + k]));
              
            /*    aik[0] = _mm256_set_pd(
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
                aik[1] = _mm256_load_pd(&(a->element[1][i * real_mid_dim + k]));
                aik[2] = _mm256_load_pd(&(a->element[2][i * real_mid_dim + k]));
                
            /*    aik[1] = _mm256_set_pd(
                    a->element[1][i * real_mid_dim + k],
                    a->element[1][i * real_mid_dim + k + 1],
                    a->element[1][i * real_mid_dim + k + 2],
                    a->element[1][i * real_mid_dim + k + 3]
                );
                aik[2] = _mm256_set_pd(
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
                bkj[0] = _mm256_set_pd(
                    //b->element[0][ k      * real_col_dim + j],
                    //b->element[0][(k + 1) * real_col_dim + j],
                    //b->element[0][(k + 2) * real_col_dim + j],
                    //b->element[0][(k + 3) * real_col_dim + j]
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
            
                bkj[1] = _mm256_set_pd(
                    //b->element[1][ k      * real_col_dim + j],
                    //b->element[1][(k + 1) * real_col_dim + j],
                    //b->element[1][(k + 2) * real_col_dim + j],
                    //b->element[1][(k + 3) * real_col_dim + j]
                    b->element[1][(k + 3) * real_col_dim + j],
                    b->element[1][(k + 2) * real_col_dim + j],
                    b->element[1][(k + 1) * real_col_dim + j],
                    b->element[1][(k    ) * real_col_dim + j]
                );
                bkj[2] = _mm256_set_pd(
                    //b->element[2][ k      * real_col_dim + j],
                    //b->element[2][(k + 1) * real_col_dim + j],
                    //b->element[2][(k + 2) * real_col_dim + j],
                    //b->element[2][(k + 3) * real_col_dim + j]
                    b->element[2][(k + 3) * real_col_dim + j],
                    b->element[2][(k + 2) * real_col_dim + j],
                    b->element[2][(k + 1) * real_col_dim + j],
                    b->element[2][(k    ) * real_col_dim + j]
                );

            /*
                rtd_mul(tmp_mul[0].val, aik[0].val, bkj[0].val);
                rtd_mul(tmp_mul[1].val, aik[1].val, bkj[1].val);
                rtd_mul(tmp_mul[2].val, aik[2].val, bkj[2].val);
                rtd_mul(tmp_mul[3].val, aik[3].val, bkj[3].val);
            */
                _bncavx2_rtd_mul(tmp_mul, aik, bkj);

            /*
                rtd_add(cij.val, cij.val, tmp_mul[0].val);
                rtd_add(cij.val, cij.val, tmp_mul[1].val);
                rtd_add(cij.val, cij.val, tmp_mul[2].val);
                rtd_add(cij.val, cij.val, tmp_mul[3].val);
            */
                _bncavx2_rtd_add(cij, cij, tmp_mul);
            }

            cijval[0][0] = cij[0][0]; cijval[0][1] = cij[1][0]; cijval[0][2] = cij[2][0];
            cijval[1][0] = cij[0][1]; cijval[1][1] = cij[1][1]; cijval[1][2] = cij[2][1];
            cijval[2][0] = cij[0][2]; cijval[2][1] = cij[1][2]; cijval[2][2] = cij[2][2];
            cijval[3][0] = cij[0][3]; cijval[3][1] = cij[1][3]; cijval[3][2] = cij[2][3];
            rtd_add(cijval[0], cijval[0], cijval[1]);
            rtd_add(cijval[0], cijval[0], cijval[2]);
            rtd_add(cijval[0], cijval[0], cijval[3]);

            ret->element[0][i * real_col_dim + j] = cijval[0][0];
            ret->element[1][i * real_col_dim + j] = cijval[0][1];
            ret->element[2][i * real_col_dim + j] = cijval[0][2];
        }
    }
#elif defined(__AVX512F__) // __AVX512F__
	long real_row_dim, real_col_dim, real_mid_dim;
	double cijval[8][TDSIZE];
    __m512d cij[TDSIZE], aik[TDSIZE], bkj[TDSIZE], tmp_mul[TDSIZE];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j++)
        {
            //rtd_set_ui(cij.val, 0UL);
            cij[0] = _mm512_setzero_pd();
            cij[1] = _mm512_setzero_pd();
            cij[2] = _mm512_setzero_pd();
            for(k = 0; k < real_mid_dim; k += _BNC_D_WIDTH)
            {
            /*
                aik[0].val[0] = a->element[0][i * mid_dim + k];
                aik[1].val[0] = a->element[0][i * mid_dim + k + 1];
                aik[2].val[0] = a->element[0][i * mid_dim + k + 2];
                aik[3].val[0] = a->element[0][i * mid_dim + k + 3];
            */
            //    aik[0] = _mm256_load_pd(&(a->element[0][i * mid_dim + k]));
                
                aik[0] = _mm512_set_pd(
                    a->element[0][i * real_mid_dim + k],
                    a->element[0][i * real_mid_dim + k + 1],
                    a->element[0][i * real_mid_dim + k + 2],
                    a->element[0][i * real_mid_dim + k + 3],
                    a->element[0][i * real_mid_dim + k + 4],
                    a->element[0][i * real_mid_dim + k + 5],
                    a->element[0][i * real_mid_dim + k + 6],
                    a->element[0][i * real_mid_dim + k + 7]
                );
            
            /*
                aik[0].val[1] = a->element[1][i * mid_dim + k];
                aik[1].val[1] = a->element[1][i * mid_dim + k + 1];
                aik[2].val[1] = a->element[1][i * mid_dim + k + 2];
                aik[3].val[1] = a->element[1][i * mid_dim + k + 3];
            */
            //    aik[1] = _mm256_load_pd(&(a->element[1][i * mid_dim + k]));
                
                aik[1] = _mm512_set_pd(
                    a->element[1][i * real_mid_dim + k],
                    a->element[1][i * real_mid_dim + k + 1],
                    a->element[1][i * real_mid_dim + k + 2],
                    a->element[1][i * real_mid_dim + k + 3],
                    a->element[1][i * real_mid_dim + k + 4],
                    a->element[1][i * real_mid_dim + k + 5],
                    a->element[1][i * real_mid_dim + k + 6],
                    a->element[1][i * real_mid_dim + k + 7]
                );
                aik[2] = _mm512_set_pd(
                    a->element[2][i * real_mid_dim + k],
                    a->element[2][i * real_mid_dim + k + 1],
                    a->element[2][i * real_mid_dim + k + 2],
                    a->element[2][i * real_mid_dim + k + 3],
                    a->element[2][i * real_mid_dim + k + 4],
                    a->element[2][i * real_mid_dim + k + 5],
                    a->element[2][i * real_mid_dim + k + 6],
                    a->element[2][i * real_mid_dim + k + 7]
                );
                
            /*
                bkj[0].val[0] = b->element[0][k * col_dim + j];
                bkj[1].val[0] = b->element[0][(k + 1) * col_dim + j];
                bkj[2].val[0] = b->element[0][(k + 2) * col_dim + j];
                bkj[3].val[0] = b->element[0][(k + 3) * col_dim + j];
            */
                bkj[0] = _mm512_set_pd(
                    b->element[0][ k      * real_col_dim + j],
                    b->element[0][(k + 1) * real_col_dim + j],
                    b->element[0][(k + 2) * real_col_dim + j],
                    b->element[0][(k + 3) * real_col_dim + j],
                    b->element[0][(k + 4) * real_col_dim + j],
                    b->element[0][(k + 5) * real_col_dim + j],
                    b->element[0][(k + 6) * real_col_dim + j],
                    b->element[0][(k + 7) * real_col_dim + j]
                );
            /*
                bkj[0].val[1] = b->element[1][k * col_dim + j];
                bkj[1].val[1] = b->element[1][(k + 1) * col_dim + j];
                bkj[2].val[1] = b->element[1][(k + 2) * col_dim + j];
                bkj[3].val[1] = b->element[1][(k + 3) * col_dim + j];
            */
            
                bkj[1] = _mm512_set_pd(
                    b->element[1][ k      * real_col_dim + j],
                    b->element[1][(k + 1) * real_col_dim + j],
                    b->element[1][(k + 2) * real_col_dim + j],
                    b->element[1][(k + 3) * real_col_dim + j],
                    b->element[1][(k + 4) * real_col_dim + j],
                    b->element[1][(k + 5) * real_col_dim + j],
                    b->element[1][(k + 6) * real_col_dim + j],
                    b->element[1][(k + 7) * real_col_dim + j]
                );
                bkj[2] = _mm512_set_pd(
                    b->element[2][ k      * real_col_dim + j],
                    b->element[2][(k + 1) * real_col_dim + j],
                    b->element[2][(k + 2) * real_col_dim + j],
                    b->element[2][(k + 3) * real_col_dim + j],
                    b->element[2][(k + 4) * real_col_dim + j],
                    b->element[2][(k + 5) * real_col_dim + j],
                    b->element[2][(k + 6) * real_col_dim + j],
                    b->element[2][(k + 7) * real_col_dim + j]
                );

            /*
                rtd_mul(tmp_mul[0].val, aik[0].val, bkj[0].val);
                rtd_mul(tmp_mul[1].val, aik[1].val, bkj[1].val);
                rtd_mul(tmp_mul[2].val, aik[2].val, bkj[2].val);
                rtd_mul(tmp_mul[3].val, aik[3].val, bkj[3].val);
            */
                _bncavx512_rtd_mul(tmp_mul, aik, bkj);

            /*
                rtd_add(cij.val, cij.val, tmp_mul[0].val);
                rtd_add(cij.val, cij.val, tmp_mul[1].val);
                rtd_add(cij.val, cij.val, tmp_mul[2].val);
                rtd_add(cij.val, cij.val, tmp_mul[3].val);
            */
                _bncavx512_rtd_add(cij, cij, tmp_mul);
            }

            cijval[0][0] = cij[0][0]; cijval[0][1] = cij[1][0]; cijval[0][2] = cij[2][0];
            cijval[1][0] = cij[0][1]; cijval[1][1] = cij[1][1]; cijval[1][2] = cij[2][1];
            cijval[2][0] = cij[0][2]; cijval[2][1] = cij[1][2]; cijval[2][2] = cij[2][2];
            cijval[3][0] = cij[0][3]; cijval[3][1] = cij[1][3]; cijval[3][2] = cij[2][3];
            cijval[4][0] = cij[0][4]; cijval[4][1] = cij[1][4]; cijval[4][2] = cij[2][4];
            cijval[5][0] = cij[0][5]; cijval[5][1] = cij[1][5]; cijval[5][2] = cij[2][5];
            cijval[6][0] = cij[0][6]; cijval[6][1] = cij[1][6]; cijval[6][2] = cij[2][6];
            cijval[7][0] = cij[0][7]; cijval[7][1] = cij[1][7]; cijval[7][2] = cij[2][7];
            rtd_add(cijval[0], cijval[0], cijval[1]);
            rtd_add(cijval[0], cijval[0], cijval[2]);
            rtd_add(cijval[0], cijval[0], cijval[3]);
            rtd_add(cijval[0], cijval[0], cijval[4]);
            rtd_add(cijval[0], cijval[0], cijval[5]);
            rtd_add(cijval[0], cijval[0], cijval[6]);
            rtd_add(cijval[0], cijval[0], cijval[7]);

            ret->element[0][i * real_col_dim + j] = cijval[0][0];
            ret->element[1][i * real_col_dim + j] = cijval[0][1];
            ret->element[2][i * real_col_dim + j] = cijval[0][2];
        }
    }

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic)
	/* Outer-product GEMM: broadcast a[i][k] across all lanes, load svcntd()
	 * consecutive columns of b at row k, FMA into cij, store cij back to ret.
	 * Mirrors the SVE2 _bncsve2_rdd kernel above; uses per-limb-variable API. */
	{
		long real_row_dim = a->real_row_dim;
		long real_col_dim = b->real_col_dim;
		long real_mid_dim = a->real_col_dim;
		long vl = (long)svcntd();

		for(i = 0; i < real_row_dim; i++){
			for(j = 0; j < real_col_dim; j += vl){
				svbool_t pg = svwhilelt_b64_s64((int64_t)j, (int64_t)real_col_dim);
				svfloat64_t cij0, cij1, cij2;
				_bncsve2_rtd_set0(&cij0, &cij1, &cij2);
				for(k = 0; k < real_mid_dim; k++){
					svfloat64_t aik0 = svdup_n_f64(a->element[0][i*real_mid_dim + k]);
					svfloat64_t aik1 = svdup_n_f64(a->element[1][i*real_mid_dim + k]);
					svfloat64_t aik2 = svdup_n_f64(a->element[2][i*real_mid_dim + k]);
					svfloat64_t bkj0 = svld1_f64(pg, &(b->element[0][k*real_col_dim + j]));
					svfloat64_t bkj1 = svld1_f64(pg, &(b->element[1][k*real_col_dim + j]));
					svfloat64_t bkj2 = svld1_f64(pg, &(b->element[2][k*real_col_dim + j]));
					svfloat64_t t0, t1, t2;
					_bncsve2_rtd_mul(pg, &t0, &t1, &t2, aik0, aik1, aik2, bkj0, bkj1, bkj2);
					_bncsve2_rtd_add(pg, &cij0, &cij1, &cij2, cij0, cij1, cij2, t0, t1, t2);
				}
				svst1_f64(pg, &(ret->element[0][i*real_col_dim + j]), cij0);
				svst1_f64(pg, &(ret->element[1][i*real_col_dim + j]), cij1);
				svst1_f64(pg, &(ret->element[2][i*real_col_dim + j]), cij2);
			}
		}
	}

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	/* NEON (float64x2_t): outer-product GEMM, matches _bncomp_mul_tdmatrix() and
	 * the SVE2 path so serial vs parallel TD multiplies agree bitwise.
	 * (Replaces a dot-product formulation whose final _bncneon_rtd_sum128d
	 *  horizontal reduction was dropping the 3rd TD limb, leaving rel.err at
	 *  DD-eps level instead of TD-eps.) */
	long real_row_dim = ret->real_row_dim;
	long real_col_dim = ret->real_col_dim;
	long real_mid_dim = a->real_col_dim;
	float64x2_t cij[TDSIZE], aik[TDSIZE], bkj[TDSIZE], tmp_mul[TDSIZE];

	for(i = 0; i < real_row_dim; i++)
	{
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			_bncneon_set0_td(cij);
			for(k = 0; k < real_mid_dim; k++)
			{
				aik[0] = vdupq_n_f64(a->element[0][i * real_mid_dim + k]);
				aik[1] = vdupq_n_f64(a->element[1][i * real_mid_dim + k]);
				aik[2] = vdupq_n_f64(a->element[2][i * real_mid_dim + k]);
				bkj[0] = vld1q_f64(&(b->element[0][k * real_col_dim + j]));
				bkj[1] = vld1q_f64(&(b->element[1][k * real_col_dim + j]));
				bkj[2] = vld1q_f64(&(b->element[2][k * real_col_dim + j]));
				_bncneon_rtd_mul(tmp_mul, aik, bkj);
				_bncneon_rtd_add(cij, cij, tmp_mul);
			}
			vst1q_f64(&(ret->element[0][i * real_col_dim + j]), cij[0]);
			vst1q_f64(&(ret->element[1][i * real_col_dim + j]), cij[1]);
			vst1q_f64(&(ret->element[2][i * real_col_dim + j]), cij[2]);
		}
	}

#else // __AVX2__
	long row_dim, col_dim, mid_dim;
	double tmp[TDSIZE], ret_ij[TDSIZE];

	//printf("Non SIMD mul_tdmatrix(%ld, %ld)\n", ret->row_dim, ret->col_dim);
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = a->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			//rtd_set0(GET_TDMATRIX_IJ(ret, i, j));
			rtd_set0(ret_ij);
			for(k = 0; k < mid_dim; k++)
			{
				rtd_mul(tmp, GET_TDMATRIX_IJ(a, i, k), GET_TDMATRIX_IJ(b, k, j));
				//rtd_add(GET_TDMATRIX_IJ(ret, i, j), tmp, GET_TDMATRIX_IJ(ret, i, j));
				rtd_add(ret_ij, tmp, ret_ij);
			}
			set_tdmatrix_ij(ret, i, j, ret_ij);
		}
	}
	//printf("end(%ld, %ld)\n", ret->row_dim, ret->col_dim);
#endif // __AVX2__

}

// Frobenius norm
void normf_tdmatrix(double ret[TDSIZE], TDMatrix mat)
{
	long int i;
	long int real_total_dim;
	double tmp[TDSIZE];

	real_total_dim = mat->real_row_dim * mat->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d ret4[TDSIZE], mat4[TDSIZE], tmp4[TDSIZE];

	ret4[0] = _mm256_setzero_pd();
	ret4[1] = _mm256_setzero_pd();
	ret4[2] = _mm256_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		mat4[0] = _mm256_load_pd(&(mat->element[0][i]));
		mat4[1] = _mm256_load_pd(&(mat->element[1][i]));
		mat4[2] = _mm256_load_pd(&(mat->element[2][i]));

		// tmp4 := mat4[i]^2
		// ret4 += tmp4
		_bncavx2_rtd_mul(tmp4, mat4, mat4);
		_bncavx2_rtd_add(ret4, ret4, tmp4);
	}

	_bncavx2_rtd_sum256d(ret, ret4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512d ret8[TDSIZE], mat8[TDSIZE], tmp8[TDSIZE];

	ret8[0] = _mm512_setzero_pd();
	ret8[1] = _mm512_setzero_pd();
	ret8[2] = _mm512_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		mat8[0] = _mm512_load_pd(&(mat->element[0][i]));
		mat8[1] = _mm512_load_pd(&(mat->element[1][i]));
		mat8[2] = _mm512_load_pd(&(mat->element[2][i]));

		// tmp8 := mat8[i]^2
		// ret8 += tmp8
		_bncavx512_rtd_mul(tmp8, mat8, mat8);
		_bncavx512_rtd_add(ret8, ret8, tmp8);
	}

	_bncavx512_rtd_sum512d(ret, ret8);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)

    
	svfloat64_t ret2_0, ret2_1, ret2_2;
	svfloat64_t mat2_0, mat2_1, mat2_2;
	svfloat64_t tmp2_0, tmp2_1, tmp2_2;

    _bncsve2_rtd_set0(&ret2_0, &ret2_1, &ret2_2);
    for(i = 0; i < real_total_dim; i += (long int)svcntd())
    {
    		svbool_t pg = svwhilelt_b64_s32((int)i, (int)real_total_dim);
        mat2_0 = svld1_f64(pg, &(mat->element[0][i]));
        mat2_1 = svld1_f64(pg, &(mat->element[1][i]));
        mat2_2 = svld1_f64(pg, &(mat->element[2][i]));

        _bncsve2_rtd_mul(svptrue_b64(), &tmp2_0, &tmp2_1, &tmp2_2, mat2_0, mat2_1, mat2_2, mat2_0, mat2_1, mat2_2);
        _bncsve2_rtd_add(svptrue_b64(), &ret2_0, &ret2_1, &ret2_2, ret2_0, ret2_1, ret2_2, tmp2_0, tmp2_1, tmp2_2);
    }

    _bncsve2_rtd_sum128d(svptrue_b64(), ret, ret2_0, ret2_1, ret2_2);

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)

    float64x2_t ret2[TDSIZE], mat2[TDSIZE], tmp2[TDSIZE];

    _bncneon_set0_td(ret2);
    for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
    {
        mat2[0] = vld1q_f64(&(mat->element[0][i]));
        mat2[1] = vld1q_f64(&(mat->element[1][i]));
        mat2[2] = vld1q_f64(&(mat->element[2][i]));

        _bncneon_rtd_mul(tmp2, mat2, mat2);
        _bncneon_rtd_add(ret2, ret2, tmp2);
    }

    _bncneon_rtd_sum128d(ret, ret2);

	#else // others
	double mat1[TDSIZE];

	rtd_set0(ret);
	for(i = 0; i < real_total_dim; i++)
	{
		mat1[0] = mat->element[0][i];
		mat1[1] = mat->element[1][i];
		mat1[2] = mat->element[2][i];

		// tmp := mat1[i]^2
		// ret += tmp
		rtd_mul(tmp, mat1, mat1);
		rtd_add(ret, ret, tmp);
	}

#endif // __AVX2__
#ifdef USE_GMP
	rtd_sqrt_mpfr(tmp, ret);   /* tmp := sqrt(sum-of-squares in ret) */
#else // USE_GMP
	rtd_sqrt(tmp, ret);
#endif // USE_GMP
	rtd_set(ret, tmp);

}

// print normf
void print_normf_tdmatrix(const char *str, TDMatrix mat)
{
	static double tmp[TDSIZE];

	normf_tdmatrix(tmp, mat);

	if(str != NULL)
		printf("%s(%ld, %ld)", str, mat->row_dim, mat->col_dim);

	rtd_out_str(tmp); printf("\n");
}

/*************************************************/
/* Matrix Caluculations for TDMatrix            */
/*
void normf_tdmatrix(double ret[TDSIZE], TDMatrix mat)
void norm1_tdmatrix(double ret[TDSIZE], TDMatrix mat)
void normi_tdmatrix(double ret[TDSIZE], TDMatrix mat)
void add_tdmatrix(TDMatrix c, TDMatrix a, TDMatrix b);
void sub_tdmatrix(TDMatrix c, TDMatrix a, TDMatrix b);
void mul_tdmatrix(TDMatrix c, TDMatrix a, TDMatrix b);
void mul_tdmatrix_tdvec(TDVector v, TDMatrix a, TDVector vb)
void mul_tdmatrixt_tdvec(TDVector v, TDMatrix a, TDVector vb)
void transpose_tdmatrix(TDMatrix c, TDMatrix a);
void inv_tdmatrix(TDMatrix a);
void subst_tdmatrix(TDMatrix c, TDMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_tdmatrix(double ret[TDSIZE], TDMatrix mat)
{
	long int i, j;
	double tmp[TDSIZE], sum[TDSIZE];

	set0_td(ret);
	for(i = 0; i < mat->row_dim; i++)
	{
		set0_td(sum);
		for(j = 0; j < mat->col_dim; j++)
		{
			rtd_abs(tmp, get_tdmatrix_ij(mat, i, j));
			rtd_add(sum, sum, tmp);
		}
		if(rtd_cmp(ret, sum) < 0)
			rtd_set(ret, sum);
	}

	return;
}

/* 1 Norm of Matrix */
void norm1_tdmatrix(double ret[TDSIZE], TDMatrix mat)
{
	long int i, j;
	double tmp[TDSIZE], sum[TDSIZE];

	rtd_set_ui(ret, 0UL);

	for(j = 0; j < mat->col_dim; j++)
	{
		rtd_set_ui(sum, 0UL);
		for(i = 0; i < mat->row_dim; i++)
		{
			rtd_abs(tmp, get_tdmatrix_ij(mat, i, j));
			rtd_add(sum, sum, tmp);
		}
		if(rtd_cmp(ret, sum) < 0)
			rtd_set(ret, sum);
	}

	return;
}

/* c := a + b */
void add_tdmatrix(TDMatrix c, TDMatrix a, TDMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_tdmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_tdmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
//#ifdef BNC_USE_GCC
//#if 0
	__m256d tmp4[TDSIZE], aij4[TDSIZE], bij4[TDSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij4[0] = _mm256_load_pd(&(a->element[0][index]));
		aij4[1] = _mm256_load_pd(&(a->element[1][index]));
		aij4[2] = _mm256_load_pd(&(a->element[2][index]));
		bij4[0] = _mm256_load_pd(&(b->element[0][index]));
		bij4[1] = _mm256_load_pd(&(b->element[1][index]));
		bij4[2] = _mm256_load_pd(&(b->element[2][index]));

		_bncavx2_rtd_add(tmp4, aij4, bij4);

		_mm256_store_pd(&(c->element[0][index]), tmp4[0]);
		_mm256_store_pd(&(c->element[1][index]), tmp4[1]); 
		_mm256_store_pd(&(c->element[2][index]), tmp4[2]); 
	}
//#endif // 0
//#else // BNC_USE_GCC
#if 0
	double tmp[TDSIZE], aij[TDSIZE], bij[TDSIZE];

	for(index = 0; index < real_total_dim; index++)
	{
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];
		aij[2] = a->element[2][index];
		bij[0] = b->element[0][index];
		bij[1] = b->element[1][index];
		bij[2] = b->element[2][index];

		rtd_add(tmp, aij, bij);

		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1]; 
		c->element[2][index] = tmp[2]; 
	}
#endif // 0
//#endif // BNC_USE_GCC
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[TDSIZE], aij8[TDSIZE], bij8[TDSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij8[0] = _mm512_load_pd(&(a->element[0][index]));
		aij8[1] = _mm512_load_pd(&(a->element[1][index]));
		aij8[2] = _mm512_load_pd(&(a->element[2][index]));
		bij8[0] = _mm512_load_pd(&(b->element[0][index]));
		bij8[1] = _mm512_load_pd(&(b->element[1][index]));
		bij8[2] = _mm512_load_pd(&(b->element[2][index]));

		_bncavx512_rtd_add(tmp8, aij8, bij8);

		_mm512_store_pd(&(c->element[0][index]), tmp8[0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[1]); 
		_mm512_store_pd(&(c->element[2][index]), tmp8[2]); 
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)

    
	svfloat64_t tmp2_0, tmp2_1, tmp2_2;
	svfloat64_t aij2_0, aij2_1, aij2_2;
	svfloat64_t bij2_0, bij2_1, bij2_2;

    for(index = 0; index < real_total_dim; index += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(real_total_dim));
        aij2_0 = svld1_f64(pg, &(a->element[0][index]));
        aij2_1 = svld1_f64(pg, &(a->element[1][index]));
        aij2_2 = svld1_f64(pg, &(a->element[2][index]));
        bij2_0 = svld1_f64(pg, &(b->element[0][index]));
        bij2_1 = svld1_f64(pg, &(b->element[1][index]));
        bij2_2 = svld1_f64(pg, &(b->element[2][index]));

        _bncsve2_rtd_add(pg, &tmp2_0, &tmp2_1, &tmp2_2, aij2_0, aij2_1, aij2_2, bij2_0, bij2_1, bij2_2);

        svst1_f64(pg, &(c->element[0][index]), tmp2_0);
        svst1_f64(pg, &(c->element[1][index]), tmp2_1); 
        svst1_f64(pg, &(c->element[2][index]), tmp2_2); 
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)

    float64x2_t tmp2[TDSIZE], aij2[TDSIZE], bij2[TDSIZE];

    for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
    {
        aij2[0] = vld1q_f64(&(a->element[0][index]));
        aij2[1] = vld1q_f64(&(a->element[1][index]));
        aij2[2] = vld1q_f64(&(a->element[2][index]));
        bij2[0] = vld1q_f64(&(b->element[0][index]));
        bij2[1] = vld1q_f64(&(b->element[1][index]));
        bij2[2] = vld1q_f64(&(b->element[2][index]));

        _bncneon_rtd_add(tmp2, aij2, bij2);

        vst1q_f64(&(c->element[0][index]), tmp2[0]);
        vst1q_f64(&(c->element[1][index]), tmp2[1]); 
        vst1q_f64(&(c->element[2][index]), tmp2[2]); 
    }
#else // others
	double tmp[TDSIZE], aij[TDSIZE], bij[TDSIZE];

	for(index = 0; index < real_total_dim; index++)
	{
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];
		aij[2] = a->element[2][index];
		bij[0] = b->element[0][index];
		bij[1] = b->element[1][index];
		bij[2] = b->element[2][index];

		rtd_add(tmp, aij, bij);

		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1]; 
		c->element[2][index] = tmp[2]; 
	}
#endif // __AVX2__
/*
	double tmp[TDSIZE];

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rtd_add(tmp, get_tdmatrix_ij(a, i, j), get_tdmatrix_ij(b, i, j));
			set_tdmatrix_ij(c, i, j, tmp);
		}
	}
*/
}

/* c := a - b */
void sub_tdmatrix(TDMatrix c, TDMatrix a, TDMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: sub_tdmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: sub_tdmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

/*
	double tmp[TDSIZE];
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rtd_sub(tmp, get_tdmatrix_ij(a, i, j), get_tdmatrix_ij(b, i, j));
			set_tdmatrix_ij(c, i, j, tmp);
		}
	}
*/

// for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[TDSIZE], aij4[TDSIZE], bij4[TDSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij4[0] = _mm256_load_pd(&(a->element[0][index]));
		aij4[1] = _mm256_load_pd(&(a->element[1][index]));
		aij4[2] = _mm256_load_pd(&(a->element[2][index]));
		bij4[0] = _mm256_load_pd(&(b->element[0][index]));
		bij4[1] = _mm256_load_pd(&(b->element[1][index]));
		bij4[2] = _mm256_load_pd(&(b->element[2][index]));

		_bncavx2_rtd_sub(tmp4, aij4, bij4);

		_mm256_store_pd(&(c->element[0][index]), tmp4[0]);
		_mm256_store_pd(&(c->element[1][index]), tmp4[1]); 
		_mm256_store_pd(&(c->element[2][index]), tmp4[2]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[TDSIZE], aij8[TDSIZE], bij8[TDSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij8[0] = _mm512_load_pd(&(a->element[0][index]));
		aij8[1] = _mm512_load_pd(&(a->element[1][index]));
		aij8[2] = _mm512_load_pd(&(a->element[2][index]));
		bij8[0] = _mm512_load_pd(&(b->element[0][index]));
		bij8[1] = _mm512_load_pd(&(b->element[1][index]));
		bij8[2] = _mm512_load_pd(&(b->element[2][index]));

		_bncavx512_rtd_sub(tmp8, aij8, bij8);

		_mm512_store_pd(&(c->element[0][index]), tmp8[0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[1]); 
		_mm512_store_pd(&(c->element[2][index]), tmp8[2]); 
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)

    
	svfloat64_t tmp2_0, tmp2_1, tmp2_2;
	svfloat64_t aij2_0, aij2_1, aij2_2;
	svfloat64_t bij2_0, bij2_1, bij2_2;

    for(index = 0; index < real_total_dim; index += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(real_total_dim));
        aij2_0 = svld1_f64(pg, &(a->element[0][index]));
        aij2_1 = svld1_f64(pg, &(a->element[1][index]));
        aij2_2 = svld1_f64(pg, &(a->element[2][index]));
        bij2_0 = svld1_f64(pg, &(b->element[0][index]));
        bij2_1 = svld1_f64(pg, &(b->element[1][index]));
        bij2_2 = svld1_f64(pg, &(b->element[2][index]));

        _bncsve2_rtd_neg(pg, &tmp2_0, &tmp2_1, &tmp2_2, bij2_0, bij2_1, bij2_2);
		_bncsve2_rtd_add(pg, &tmp2_0, &tmp2_1, &tmp2_2, aij2_0, aij2_1, aij2_2, tmp2_0, tmp2_1, tmp2_2);

        svst1_f64(pg, &(c->element[0][index]), tmp2_0);
        svst1_f64(pg, &(c->element[1][index]), tmp2_1); 
        svst1_f64(pg, &(c->element[2][index]), tmp2_2); 
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)

    float64x2_t tmp2[TDSIZE], aij2[TDSIZE], bij2[TDSIZE];

    for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
    {
        aij2[0] = vld1q_f64(&(a->element[0][index]));
        aij2[1] = vld1q_f64(&(a->element[1][index]));
        aij2[2] = vld1q_f64(&(a->element[2][index]));
        bij2[0] = vld1q_f64(&(b->element[0][index]));
        bij2[1] = vld1q_f64(&(b->element[1][index]));
        bij2[2] = vld1q_f64(&(b->element[2][index]));

        _bncneon_rtd_sub(tmp2, aij2, bij2);

        vst1q_f64(&(c->element[0][index]), tmp2[0]);
        vst1q_f64(&(c->element[1][index]), tmp2[1]); 
        vst1q_f64(&(c->element[2][index]), tmp2[2]); 
    }
#else // others
	double tmp[TDSIZE], aij[TDSIZE], bij[TDSIZE];

	for(index = 0; index < real_total_dim; index++)
	{
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];
		aij[2] = a->element[2][index];
		bij[0] = b->element[0][index];
		bij[1] = b->element[1][index];
		bij[2] = b->element[2][index];

		rtd_sub(tmp, aij, bij);

		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1]; 
		c->element[2][index] = tmp[2]; 
	}
#endif // __AVX2__
}

/* c := sc * a */
void cmul_tdmatrix(TDMatrix c, double sc[TDSIZE], TDMatrix a)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_tdmatrix(row_dim is differnt! : c->(%ld, %ld), a->(%ld, %ld)\n", c->row_dim, c->col_dim, a->row_dim, a->col_dim);
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_tdmatrix(col_dim is differnt! : c->(%ld, %ld), a->(%ld, %ld)\n", c->row_dim, c->col_dim, a->row_dim, a->col_dim);
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

// for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[TDSIZE], sc4[TDSIZE], aij4[TDSIZE];

	sc4[0] = _mm256_set1_pd(sc[0]);
	sc4[1] = _mm256_set1_pd(sc[1]);
	sc4[2] = _mm256_set1_pd(sc[2]);

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij4[0] = _mm256_load_pd(&(a->element[0][index]));
		aij4[1] = _mm256_load_pd(&(a->element[1][index]));
		aij4[2] = _mm256_load_pd(&(a->element[2][index]));

		_bncavx2_rtd_mul(tmp4, sc4, aij4);

		_mm256_store_pd(&(c->element[0][index]), tmp4[0]);
		_mm256_store_pd(&(c->element[1][index]), tmp4[1]); 
		_mm256_store_pd(&(c->element[2][index]), tmp4[2]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[TDSIZE], aij8[TDSIZE], sc8[TDSIZE];

	sc8[0] = _mm512_set1_pd(sc[0]);
	sc8[1] = _mm512_set1_pd(sc[1]);
	sc8[2] = _mm512_set1_pd(sc[2]);

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij8[0] = _mm512_load_pd(&(a->element[0][index]));
		aij8[1] = _mm512_load_pd(&(a->element[1][index]));
		aij8[2] = _mm512_load_pd(&(a->element[2][index]));

		_bncavx512_rtd_add(tmp8, sc8, aij8);

		_mm512_store_pd(&(c->element[0][index]), tmp8[0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[1]); 
		_mm512_store_pd(&(c->element[2][index]), tmp8[2]); 
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)

    
	svfloat64_t tmp2_0, tmp2_1, tmp2_2;
	svfloat64_t aij2_0, aij2_1, aij2_2;
	svfloat64_t sc2_0, sc2_1, sc2_2;

    sc2_0 = svdup_f64(sc[0]);
    sc2_1 = svdup_f64(sc[1]);
    sc2_2 = svdup_f64(sc[2]);

    for(index = 0; index < real_total_dim; index += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(real_total_dim));
        aij2_0 = svld1_f64(pg, &(a->element[0][index]));
        aij2_1 = svld1_f64(pg, &(a->element[1][index]));
        aij2_2 = svld1_f64(pg, &(a->element[2][index]));

        _bncsve2_rtd_mul(pg, &tmp2_0, &tmp2_1, &tmp2_2, sc2_0, sc2_1, sc2_2, aij2_0, aij2_1, aij2_2);

        svst1_f64(pg, &(c->element[0][index]), tmp2_0);
        svst1_f64(pg, &(c->element[1][index]), tmp2_1); 
        svst1_f64(pg, &(c->element[2][index]), tmp2_2); 
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)

    float64x2_t tmp2[TDSIZE], aij2[TDSIZE], sc2[TDSIZE];

    sc2[0] = vdupq_n_f64(sc[0]);
    sc2[1] = vdupq_n_f64(sc[1]);
    sc2[2] = vdupq_n_f64(sc[2]);

    for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
    {
        aij2[0] = vld1q_f64(&(a->element[0][index]));
        aij2[1] = vld1q_f64(&(a->element[1][index]));
        aij2[2] = vld1q_f64(&(a->element[2][index]));

        _bncneon_rtd_mul(tmp2, sc2, aij2);

        vst1q_f64(&(c->element[0][index]), tmp2[0]);
        vst1q_f64(&(c->element[1][index]), tmp2[1]); 
        vst1q_f64(&(c->element[2][index]), tmp2[2]); 
    }
#else // others
	double tmp[TDSIZE], aij[TDSIZE];

	for(index = 0; index < real_total_dim; index++)
	{
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];
		aij[2] = a->element[2][index];

		rtd_mul(tmp, sc, aij);

		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1]; 
		c->element[2][index] = tmp[2]; 
	}
#endif // __AVX2__
/*
	double tmp[TDSIZE];

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rtd_mul(tmp, sc, get_tdmatrix_ij(a, i, j));
			set_tdmatrix_ij(c, i, j, tmp);
		}
	}
*/
}

/* c = a^T */
void transpose_tdmatrix(TDMatrix c, TDMatrix a)
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
	__m256d aji4[TDSIZE];

	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, j, i));
			aji4[0] = _mm256_set_pd(
				a->element[0][(j + 3) * real_col_dim + i],
				a->element[0][(j + 2) * real_col_dim + i],
				a->element[0][(j + 1) * real_col_dim + i],
				a->element[0][(j    ) * real_col_dim + i]
			);
			aji4[1] = _mm256_set_pd(
				a->element[1][(j + 3) * real_col_dim + i],
				a->element[1][(j + 2) * real_col_dim + i],
				a->element[1][(j + 1) * real_col_dim + i],
				a->element[1][(j    ) * real_col_dim + i]
			);
			aji4[2] = _mm256_set_pd(
				a->element[2][(j + 3) * real_col_dim + i],
				a->element[2][(j + 2) * real_col_dim + i],
				a->element[2][(j + 1) * real_col_dim + i],
				a->element[2][(j    ) * real_col_dim + i]
			);
			index = i * real_col_dim + j;
			_mm256_store_pd(&(c->element[0][index]), aji4[0]);
			_mm256_store_pd(&(c->element[1][index]), aji4[1]);
			_mm256_store_pd(&(c->element[2][index]), aji4[2]);
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d aji8[TDSIZE];

	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, j, i));
			aji8[0] = _mm512_set_pd(
				a->element[0][(j + 7) * real_col_dim + i],
				a->element[0][(j + 6) * real_col_dim + i],
				a->element[0][(j + 5) * real_col_dim + i],
				a->element[0][(j + 4) * real_col_dim + i],
				a->element[0][(j + 3) * real_col_dim + i],
				a->element[0][(j + 2) * real_col_dim + i],
				a->element[0][(j + 1) * real_col_dim + i],
				a->element[0][(j    ) * real_col_dim + i]
			);
			aji8[1] = _mm512_set_pd(
				a->element[1][(j + 7) * real_col_dim + i],
				a->element[1][(j + 6) * real_col_dim + i],
				a->element[1][(j + 5) * real_col_dim + i],
				a->element[1][(j + 4) * real_col_dim + i],
				a->element[1][(j + 3) * real_col_dim + i],
				a->element[1][(j + 2) * real_col_dim + i],
				a->element[1][(j + 1) * real_col_dim + i],
				a->element[1][(j    ) * real_col_dim + i]
			);
			aji8[2] = _mm512_set_pd(
				a->element[2][(j + 7) * real_col_dim + i],
				a->element[2][(j + 6) * real_col_dim + i],
				a->element[2][(j + 5) * real_col_dim + i],
				a->element[2][(j + 4) * real_col_dim + i],
				a->element[2][(j + 3) * real_col_dim + i],
				a->element[2][(j + 2) * real_col_dim + i],
				a->element[2][(j + 1) * real_col_dim + i],
				a->element[2][(j    ) * real_col_dim + i]
			);
			index = i * real_col_dim + j;
			_mm512_store_pd(&(c->element[0][index]), aji8[0]);
			_mm512_store_pd(&(c->element[1][index]), aji8[1]);
			_mm512_store_pd(&(c->element[2][index]), aji8[2]);
		}
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, gather)
    svfloat64_t aji2_0, aji2_1, aji2_2;
    svint64_t vidx;
    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j += (long int)svcntd())
        {
            svbool_t pg = svwhilelt_b64_s64((int64_t)j, (int64_t)real_col_dim);
            vidx = svindex_s64((int64_t)(j * real_col_dim + i), (int64_t)real_col_dim);
            aji2_0 = svld1_gather_s64index_f64(pg, a->element[0], vidx);
            aji2_1 = svld1_gather_s64index_f64(pg, a->element[1], vidx);
            aji2_2 = svld1_gather_s64index_f64(pg, a->element[2], vidx);
            index = i * real_col_dim + j;
            svst1_f64(pg, &(c->element[0][index]), aji2_0);
            svst1_f64(pg, &(c->element[1][index]), aji2_1);
            svst1_f64(pg, &(c->element[2][index]), aji2_2);
        }
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)

    float64x2_t aji2[TDSIZE];

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
        {
            /* aji2[k] : 行 j, j+1 の列 i をロードして、行ベクトル c(i, j:j+1) に格納 */
            aji2[0] = (float64x2_t){
                a->element[0][(j    ) * real_col_dim + i],
                a->element[0][(j + 1) * real_col_dim + i]
            };
            aji2[1] = (float64x2_t){
                a->element[1][(j    ) * real_col_dim + i],
                a->element[1][(j + 1) * real_col_dim + i]
            };
            aji2[2] = (float64x2_t){
                a->element[2][(j    ) * real_col_dim + i],
                a->element[2][(j + 1) * real_col_dim + i]
            };
            index = i * real_col_dim + j;
            vst1q_f64(&(c->element[0][index]), aji2[0]);
            vst1q_f64(&(c->element[1][index]), aji2[1]);
            vst1q_f64(&(c->element[2][index]), aji2[2]);
        }
    }
#else // others
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_tdmatrix_ij(c, i, j, get_tdmatrix_ij(a, j, i));
	}
#endif // AVX2
}

/* c := a */
void subst_tdmatrix(TDMatrix c, TDMatrix a)
{
	long int i, j, index;
	long int real_row_dim, real_col_dim;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_tdmatrix\n");
		return;
	}
	real_row_dim = c->real_row_dim;
	real_col_dim = c->real_col_dim;

// AVX2
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j));
			index = i * real_col_dim + j;
			_mm256_store_pd(&(c->element[0][index]), _mm256_load_pd(&(a->element[0][index])));
			_mm256_store_pd(&(c->element[1][index]), _mm256_load_pd(&(a->element[1][index])));
			_mm256_store_pd(&(c->element[2][index]), _mm256_load_pd(&(a->element[2][index])));
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j));
			index = i * real_col_dim + j;
			_mm512_store_pd(&(c->element[0][index]), _mm512_load_pd(&(a->element[0][index])));
			_mm512_store_pd(&(c->element[1][index]), _mm512_load_pd(&(a->element[1][index])));
			_mm512_store_pd(&(c->element[2][index]), _mm512_load_pd(&(a->element[2][index])));
		}
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j += (long int)svcntd())
        {
		svbool_t pg = svwhilelt_b64_s64((int64_t)j, (int64_t)(real_col_dim));
            index = i * real_col_dim + j;
            svst1_f64(pg, &(c->element[0][index]), svld1_f64(pg, &(a->element[0][index])));
            svst1_f64(pg, &(c->element[1][index]), svld1_f64(pg, &(a->element[1][index])));
            svst1_f64(pg, &(c->element[2][index]), svld1_f64(pg, &(a->element[2][index])));
        }
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
        {
            index = i * real_col_dim + j;
            vst1q_f64(&(c->element[0][index]), vld1q_f64(&(a->element[0][index])));
            vst1q_f64(&(c->element[1][index]), vld1q_f64(&(a->element[1][index])));
            vst1q_f64(&(c->element[2][index]), vld1q_f64(&(a->element[2][index])));
        }
    }
#else // others
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_tdmatrix_ij(c, i, j, get_tdmatrix_ij(a, i, j));
		}
	}
#endif // AVX2
}

/* c := -a */
void neg_tdmatrix(TDMatrix c, TDMatrix a)
{
	long int i, j, index;
	long int real_row_dim, real_col_dim;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: neg_tdmatrix\n");
		return;
	}
	real_row_dim = c->real_row_dim;
	real_col_dim = c->real_col_dim;

// AVX2
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp[TDSIZE];

	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j));
			index = i * real_col_dim + j;
			tmp[0] = _bncavx2_dneg(_mm256_load_pd(&(a->element[0][index])));
			tmp[1] = _bncavx2_dneg(_mm256_load_pd(&(a->element[1][index])));
			tmp[2] = _bncavx2_dneg(_mm256_load_pd(&(a->element[2][index])));
			_mm256_store_pd(&(c->element[0][index]), tmp[0]); //_mm256_load_pd(&(a->element[0][index])));
			_mm256_store_pd(&(c->element[1][index]), tmp[1]); //_mm256_load_pd(&(a->element[1][index])));
			_mm256_store_pd(&(c->element[2][index]), tmp[2]); //_mm256_load_pd(&(a->element[2][index])));
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp[TDSIZE];

	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j));
			index = i * real_col_dim + j;
			tmp[0] = _bncavx512_dneg(_mm512_load_pd(&(a->element[0][index])));
			tmp[1] = _bncavx512_dneg(_mm512_load_pd(&(a->element[1][index])));
			tmp[2] = _bncavx512_dneg(_mm512_load_pd(&(a->element[2][index])));
			_mm512_store_pd(&(c->element[0][index]), tmp[0]); //_mm512_load_pd(&(a->element[0][index])));
			_mm512_store_pd(&(c->element[1][index]), tmp[1]); //_mm512_load_pd(&(a->element[1][index])));
			_mm512_store_pd(&(c->element[2][index]), tmp[2]); //_mm512_load_pd(&(a->element[1][index])));
		}
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)

    
	svfloat64_t tmp2_0, tmp2_1, tmp2_2;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j += (long int)svcntd())
        {
        		svbool_t pg = svwhilelt_b64_s32((int)j, (int)real_col_dim);
            index = i * real_col_dim + j;
            tmp2_0 = svld1_f64(pg, &(a->element[0][index]));
            tmp2_1 = svld1_f64(pg, &(a->element[1][index]));
            tmp2_2 = svld1_f64(pg, &(a->element[2][index]));

            _bncsve2_rtd_neg(svptrue_b64(), &tmp2_0, &tmp2_1, &tmp2_2, tmp2_0, tmp2_1, tmp2_2);

            svst1_f64(pg, &(c->element[0][index]), tmp2_0);
            svst1_f64(pg, &(c->element[1][index]), tmp2_1);
            svst1_f64(pg, &(c->element[2][index]), tmp2_2);
        }
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)

    float64x2_t tmp2[TDSIZE];

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
        {
            index = i * real_col_dim + j;
            tmp2[0] = vld1q_f64(&(a->element[0][index]));
            tmp2[1] = vld1q_f64(&(a->element[1][index]));
            tmp2[2] = vld1q_f64(&(a->element[2][index]));

            _bncneon_rtd_neg(tmp2, tmp2);

            vst1q_f64(&(c->element[0][index]), tmp2[0]);
            vst1q_f64(&(c->element[1][index]), tmp2[1]);
            vst1q_f64(&(c->element[2][index]), tmp2[2]);
        }
    }
#else // others
	double tmp[TDSIZE];

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			rtd_neg(tmp, get_tdmatrix_ij(a, i, j));
			set_tdmatrix_ij(c, i, j, tmp); //get_tdmatrix_ij(a, i, j));
		}
	}
#endif // AVX2
}

/* c := I */
void setI_tdmatrix(TDMatrix c)
{
	long int i, j;
	long int real_total_dim;
	double tmp1[TDSIZE];

	real_total_dim = c->real_row_dim * c->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&(c->element[0][i]), zero4);
		_mm256_store_pd(&(c->element[1][i]), zero4);
		_mm256_store_pd(&(c->element[2][i]), zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&(c->element[0][i]), zero8);
		_mm512_store_pd(&(c->element[1][i]), zero8);
		_mm512_store_pd(&(c->element[2][i]), zero8);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)

    svfloat64_t zero2 = svdup_f64(0.0);

    for(i = 0; i < real_total_dim; i += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(real_total_dim));
        svst1_f64(pg, &(c->element[0][i]), zero2);
        svst1_f64(pg, &(c->element[1][i]), zero2);
        svst1_f64(pg, &(c->element[2][i]), zero2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)

    float64x2_t zero2 = vdupq_n_f64(0.0);

    for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
    {
        vst1q_f64(&(c->element[0][i]), zero2);
        vst1q_f64(&(c->element[1][i]), zero2);
        vst1q_f64(&(c->element[2][i]), zero2);
    }
#else // others
	for(i = 0; i < real_total_dim; i++)
	{
		c->element[0][i] = 0.0;
		c->element[1][i] = 0.0;	
		c->element[2][i] = 0.0;	
	}
#endif // __AVX2__

	rtd_set_ui(tmp1, 1UL);

	for(i = 0; i < c->row_dim; i++)
	{
		if(i < c->col_dim)
			set_tdmatrix_ij(c, i, i, tmp1);
	}
}

/* v := a * vb */
void mul_tdmatrix_tdvec(TDVector v, TDMatrix a, TDVector vb)
{
	long int i, j;
	double tmp[TDSIZE], tmp1[TDSIZE];

	/* Check Dimension */
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: mul_tdmatrix_tdvec\n");
		return;
	}

// SIMD
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int ij_index, real_col_dim;
	__m256d tmp4[TDSIZE], tmp1_4[TDSIZE];
	__m256d aij4[TDSIZE], vbj4[TDSIZE];

	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->row_dim; i++)
	{
		//rtd_set_ui(tmp, 0UL);
		_bncavx2_set0_td(tmp4);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij4[0] = _mm256_load_pd(&(a->element[0][ij_index]));
			aij4[1] = _mm256_load_pd(&(a->element[1][ij_index]));
			aij4[2] = _mm256_load_pd(&(a->element[2][ij_index]));
			vbj4[0] = _mm256_load_pd(&(vb->element[0][j]));
			vbj4[1] = _mm256_load_pd(&(vb->element[1][j]));
			vbj4[2] = _mm256_load_pd(&(vb->element[2][j]));

			//rtd_mul(tmp1, get_tdmatrix_ij(a, i, j), get_tdvector_i(vb, j));
			//rtd_add(tmp, tmp, tmp1);
			_bncavx2_rtd_mul(tmp1_4, aij4, vbj4);
			_bncavx2_rtd_add(tmp4, tmp4, tmp1_4);
		}
		//set_tdvector_i(v, i, tmp);
		_bncavx2_rtd_sum256d(tmp, tmp4);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
		v->element[2][i] = tmp[2];
	}

#elif defined(__AVX512F__) // __AVX512F__
	long int ij_index, real_col_dim;
	__m512d tmp8[TDSIZE], tmp1_8[TDSIZE];
	__m512d aij8[TDSIZE], vbj8[TDSIZE];

	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->row_dim; i++)
	{
		//rdd_set_ui(tmp, 0UL);
		_bncavx512_set0_td(tmp8);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij8[0] = _mm512_load_pd(&(a->element[0][ij_index]));
			aij8[1] = _mm512_load_pd(&(a->element[1][ij_index]));
			aij8[2] = _mm512_load_pd(&(a->element[2][ij_index]));
			vbj8[0] = _mm512_load_pd(&(vb->element[0][j]));
			vbj8[1] = _mm512_load_pd(&(vb->element[1][j]));
			vbj8[2] = _mm512_load_pd(&(vb->element[2][j]));

			//rtd_mul(tmp1, get_tdmatrix_ij(a, i, j), get_tdvector_i(vb, j));
			//rtd_add(tmp, tmp, tmp1);
			_bncavx512_rtd_mul(tmp1_8, aij8, vbj8);
			_bncavx512_rtd_add(tmp8, tmp8, tmp1_8);
		}
		//set_tdvector_i(v, i, tmp);
		_bncavx512_rtd_sum512d(tmp, tmp8);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
		v->element[2][i] = tmp[2];
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)

    long int ij_index, real_col_dim;
    
	svfloat64_t tmp2_0, tmp2_1, tmp2_2;
	svfloat64_t tmp1_2_0, tmp1_2_1, tmp1_2_2;
    
	svfloat64_t aij2_0, aij2_1, aij2_2;
	svfloat64_t vbj2_0, vbj2_1, vbj2_2;

    real_col_dim = a->real_col_dim;

    for(i = 0; i < a->row_dim; i++)
    {
        _bncsve2_rtd_set0(&tmp2_0, &tmp2_1, &tmp2_2);
        for(j = 0; j < real_col_dim; j += (long int)svcntd())
        {
        		svbool_t pg = svwhilelt_b64_s32((int)j, (int)real_col_dim);
            ij_index = i * real_col_dim + j;
            aij2_0 = svld1_f64(pg, &(a->element[0][ij_index]));
            aij2_1 = svld1_f64(pg, &(a->element[1][ij_index]));
            aij2_2 = svld1_f64(pg, &(a->element[2][ij_index]));
            vbj2_0 = svld1_f64(pg, &(vb->element[0][j]));
            vbj2_1 = svld1_f64(pg, &(vb->element[1][j]));
            vbj2_2 = svld1_f64(pg, &(vb->element[2][j]));

            _bncsve2_rtd_mul(svptrue_b64(), &tmp1_2_0, &tmp1_2_1, &tmp1_2_2, aij2_0, aij2_1, aij2_2, vbj2_0, vbj2_1, vbj2_2);
            _bncsve2_rtd_add(svptrue_b64(), &tmp2_0, &tmp2_1, &tmp2_2, tmp2_0, tmp2_1, tmp2_2, tmp1_2_0, tmp1_2_1, tmp1_2_2);
        }
        _bncsve2_rtd_sum128d(svptrue_b64(), tmp, tmp2_0, tmp2_1, tmp2_2);
        v->element[0][i] = tmp[0];
        v->element[1][i] = tmp[1];
        v->element[2][i] = tmp[2];
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)

    long int ij_index, real_col_dim;
    float64x2_t tmp2[TDSIZE], tmp1_2[TDSIZE];
    float64x2_t aij2[TDSIZE], vbj2[TDSIZE];

    real_col_dim = a->real_col_dim;

    for(i = 0; i < a->row_dim; i++)
    {
        _bncneon_set0_td(tmp2);
        for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
        {
            ij_index = i * real_col_dim + j;
            aij2[0] = vld1q_f64(&(a->element[0][ij_index]));
            aij2[1] = vld1q_f64(&(a->element[1][ij_index]));
            aij2[2] = vld1q_f64(&(a->element[2][ij_index]));
            vbj2[0] = vld1q_f64(&(vb->element[0][j]));
            vbj2[1] = vld1q_f64(&(vb->element[1][j]));
            vbj2[2] = vld1q_f64(&(vb->element[2][j]));

            _bncneon_rtd_mul(tmp1_2, aij2, vbj2);
            _bncneon_rtd_add(tmp2, tmp2, tmp1_2);
        }
        _bncneon_rtd_sum128d(tmp, tmp2);
        v->element[0][i] = tmp[0];
        v->element[1][i] = tmp[1];
        v->element[2][i] = tmp[2];
    }
#else // others

	for(i = 0; i < a->row_dim; i++)
	{
		rtd_set_ui(tmp, 0UL);
		for(j = 0; j < a->col_dim; j++)
		{
			rtd_mul(tmp1, get_tdmatrix_ij(a, i, j), get_tdvector_i(vb, j));
			rtd_add(tmp, tmp, tmp1);
		}
		set_tdvector_i(v, i, tmp);
	}
#endif // __AVX2__
}

/* v := a^T * vb */
void mul_tdmatrixt_tdvec(TDVector v, TDMatrix a, TDVector vb)
{
	long int i, j;
	double tmp[TDSIZE], tmp1[TDSIZE];

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_tdmatrixt_tdvec\n");
		return;
	}
// SIMD
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int ji_index, real_row_dim, real_col_dim;
	__m256d tmp4[TDSIZE], tmp1_4[TDSIZE];
	__m256d aij4[TDSIZE], vbj4[TDSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->col_dim; i++)
	{
		//rtd_set_ui(tmp, 0UL);
		_bncavx2_set0_td(tmp4);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_row_dim; j += _BNC_D_WIDTH)
		{
			aij4[0] = _mm256_set_pd(
				a->element[0][(j + 3) * real_col_dim + i],
				a->element[0][(j + 2) * real_col_dim + i],
				a->element[0][(j + 1) * real_col_dim + i],
				a->element[0][(j    ) * real_col_dim + i]
			);
			aij4[1] = _mm256_set_pd(
				a->element[1][(j + 3) * real_col_dim + i],
				a->element[1][(j + 2) * real_col_dim + i],
				a->element[1][(j + 1) * real_col_dim + i],
				a->element[1][(j    ) * real_col_dim + i]
			);
			aij4[2] = _mm256_set_pd(
				a->element[2][(j + 3) * real_col_dim + i],
				a->element[2][(j + 2) * real_col_dim + i],
				a->element[2][(j + 1) * real_col_dim + i],
				a->element[2][(j    ) * real_col_dim + i]
			);
			vbj4[0] = _mm256_load_pd(&(vb->element[0][j]));
			vbj4[1] = _mm256_load_pd(&(vb->element[1][j]));
			vbj4[2] = _mm256_load_pd(&(vb->element[2][j]));

			//rtd_mul(tmp1, get_tdmatrix_ij(a, i, j), get_tdvector_i(vb, j));
			//rtd_add(tmp, tmp, tmp1);
			_bncavx2_rtd_mul(tmp1_4, aij4, vbj4);
			_bncavx2_rtd_add(tmp4, tmp4, tmp1_4);
		}
		//set_tdvector_i(v, i, tmp);
		_bncavx2_rtd_sum256d(tmp, tmp4);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
		v->element[2][i] = tmp[2];
	}

#elif defined(__AVX512F__) // __AVX512F__
	long int ji_index, real_row_dim, real_col_dim;
	__m512d tmp8[TDSIZE], tmp1_8[TDSIZE];
	__m512d aij8[TDSIZE], vbj8[TDSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->col_dim; i++)
	{
		//rdd_set_ui(tmp, 0UL);
		_bncavx512_set0_td(tmp8);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_row_dim; j += _BNC_D_WIDTH)
		{
			aij8[0] = _mm512_set_pd(
				a->element[0][(j + 7) * real_col_dim + i],
				a->element[0][(j + 6) * real_col_dim + i],
				a->element[0][(j + 5) * real_col_dim + i],
				a->element[0][(j + 4) * real_col_dim + i],
				a->element[0][(j + 3) * real_col_dim + i],
				a->element[0][(j + 2) * real_col_dim + i],
				a->element[0][(j + 1) * real_col_dim + i],
				a->element[0][(j    ) * real_col_dim + i]
			);
			aij8[1] = _mm512_set_pd(
				a->element[1][(j + 7) * real_col_dim + i],
				a->element[1][(j + 6) * real_col_dim + i],
				a->element[1][(j + 5) * real_col_dim + i],
				a->element[1][(j + 4) * real_col_dim + i],
				a->element[1][(j + 3) * real_col_dim + i],
				a->element[1][(j + 2) * real_col_dim + i],
				a->element[1][(j + 1) * real_col_dim + i],
				a->element[1][(j    ) * real_col_dim + i]
			);
			aij8[2] = _mm512_set_pd(
				a->element[2][(j + 7) * real_col_dim + i],
				a->element[2][(j + 6) * real_col_dim + i],
				a->element[2][(j + 5) * real_col_dim + i],
				a->element[2][(j + 4) * real_col_dim + i],
				a->element[2][(j + 3) * real_col_dim + i],
				a->element[2][(j + 2) * real_col_dim + i],
				a->element[2][(j + 1) * real_col_dim + i],
				a->element[2][(j    ) * real_col_dim + i]
			);
			vbj8[0] = _mm512_load_pd(&(vb->element[0][j]));
			vbj8[1] = _mm512_load_pd(&(vb->element[1][j]));
			vbj8[2] = _mm512_load_pd(&(vb->element[2][j]));

			//rtd_mul(tmp1, get_tdmatrix_ij(a, i, j), get_tdvector_i(vb, j));
			//rtd_add(tmp, tmp, tmp1);
			_bncavx512_rtd_mul(tmp1_8, aij8, vbj8);
			_bncavx512_rtd_add(tmp8, tmp8, tmp1_8);
		}
		//set_tdvector_i(v, i, tmp);
		_bncavx512_rtd_sum512d(tmp, tmp8); // Fix! 2025-12-26(Fri) T.Kouya
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
		v->element[2][i] = tmp[2];
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, gather)
    long int real_row_dim, real_col_dim;
    svfloat64_t tmp2_0, tmp2_1, tmp2_2;
    svfloat64_t tmp1_2_0, tmp1_2_1, tmp1_2_2;
    svfloat64_t aij2_0, aij2_1, aij2_2;
    svfloat64_t vbj2_0, vbj2_1, vbj2_2;
    svint64_t vidx;

    real_row_dim = a->real_row_dim;
    real_col_dim = a->real_col_dim;

    for(i = 0; i < a->col_dim; i++)
    {
        _bncsve2_rtd_set0(&tmp2_0, &tmp2_1, &tmp2_2);
        for(j = 0; j < real_row_dim; j += (long int)svcntd())
        {
            svbool_t pg = svwhilelt_b64_s64((int64_t)j, (int64_t)real_row_dim);
            vidx = svindex_s64((int64_t)(j * real_col_dim + i), (int64_t)real_col_dim);
            aij2_0 = svld1_gather_s64index_f64(pg, a->element[0], vidx);
            aij2_1 = svld1_gather_s64index_f64(pg, a->element[1], vidx);
            aij2_2 = svld1_gather_s64index_f64(pg, a->element[2], vidx);
            vbj2_0 = svld1_f64(pg, &(vb->element[0][j]));
            vbj2_1 = svld1_f64(pg, &(vb->element[1][j]));
            vbj2_2 = svld1_f64(pg, &(vb->element[2][j]));
            _bncsve2_rtd_mul(svptrue_b64(), &tmp1_2_0, &tmp1_2_1, &tmp1_2_2, aij2_0, aij2_1, aij2_2, vbj2_0, vbj2_1, vbj2_2);
            _bncsve2_rtd_add(svptrue_b64(), &tmp2_0, &tmp2_1, &tmp2_2, tmp2_0, tmp2_1, tmp2_2, tmp1_2_0, tmp1_2_1, tmp1_2_2);
        }
        _bncsve2_rtd_sum128d(svptrue_b64(), tmp, tmp2_0, tmp2_1, tmp2_2);
        v->element[0][i] = tmp[0];
        v->element[1][i] = tmp[1];
        v->element[2][i] = tmp[2];
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)

    long int real_row_dim, real_col_dim;
    float64x2_t tmp2[TDSIZE], tmp1_2[TDSIZE];
    float64x2_t aij2[TDSIZE], vbj2[TDSIZE];

    real_row_dim = a->real_row_dim;
    real_col_dim = a->real_col_dim;

    for(i = 0; i < a->col_dim; i++)
    {
        _bncneon_set0_td(tmp2);
        for(j = 0; j < real_row_dim; j += _BNC_D_WIDTH)
        {
            /* a(j,i), a(j+1,i) を1レジスタにパック */
            aij2[0] = (float64x2_t){
                a->element[0][(j    ) * real_col_dim + i],
                a->element[0][(j + 1) * real_col_dim + i]
            };
            aij2[1] = (float64x2_t){
                a->element[1][(j    ) * real_col_dim + i],
                a->element[1][(j + 1) * real_col_dim + i]
            };
            aij2[2] = (float64x2_t){
                a->element[2][(j    ) * real_col_dim + i],
                a->element[2][(j + 1) * real_col_dim + i]
            };

            vbj2[0] = vld1q_f64(&(vb->element[0][j]));
            vbj2[1] = vld1q_f64(&(vb->element[1][j]));
            vbj2[2] = vld1q_f64(&(vb->element[2][j]));

            _bncneon_rtd_mul(tmp1_2, aij2, vbj2);
            _bncneon_rtd_add(tmp2, tmp2, tmp1_2);
        }
        _bncneon_rtd_sum128d(tmp, tmp2);
        v->element[0][i] = tmp[0];
        v->element[1][i] = tmp[1];
        v->element[2][i] = tmp[2];
    }

#else // others
	for(i = 0; i < a->col_dim; i++)
	{
		set0_td(tmp);
		for(j = 0; j < a->row_dim; j++)
		{
			rtd_mul(tmp1, get_tdmatrix_ij(a, j, i), get_tdvector_i(vb, j));
			rtd_add(tmp, tmp, tmp1);
		}
		set_tdvector_i(v, i, tmp);
	}
#endif // __AVX2__
}

/* a = a^(-1) */
/* square matrix only */
void inv_tdmatrix(TDMatrix a)
{
	long int i, j, k, dim;
	double tmp[TDSIZE], aii[TDSIZE];

	/* Check Dimensions */
	if(a->row_dim != a->col_dim)
	{
		fprintf(stderr, "ERROR: inv_tdmatrix\n");
		return;
	}

	dim = a->row_dim;

	for(i = 0; i < dim; i++)
	{
		if(rtd_cmp_ui(get_tdmatrix_ij(a, i, i), 0UL) == 0) 
		{
			fprintf(stderr, "ERROR: inv_tdmatrix: Pivot(%ld,%ld) is zero.\n", i, i);
			return;
		}

		rtd_ui_div(aii, 1UL, get_tdmatrix_ij(a, i, i));
		set_tdmatrix_ij(a, i, i, aii);

		for(j = 0; j < i; j++)
		{
			rtd_mul(tmp, get_tdmatrix_ij(a, i, j), aii);
			set_tdmatrix_ij(a, i, j, tmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			rtd_mul(tmp, get_tdmatrix_ij(a, i, j), aii);
			set_tdmatrix_ij(a, i, j, tmp);
		}

		for(j = 0; j < i; j++)
		{
			for(k = 0; k < i; k++)
			{
				rtd_mul(tmp, get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, k));
				rtd_sub(tmp, get_tdmatrix_ij(a, j, k), tmp);
				set_tdmatrix_ij(a, j, k, tmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				rtd_mul(tmp, get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, k));
				rtd_sub(tmp, get_tdmatrix_ij(a, j, k), tmp);
				set_tdmatrix_ij(a, j, k, tmp);
			}
		}

		for(j = i + 1; j < dim; j++)
		{
			for(k = 0; k < i; k++)
			{
				rtd_mul(tmp, get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, k));
				rtd_sub(tmp, get_tdmatrix_ij(a, j, k), tmp);
				set_tdmatrix_ij(a, j, k, tmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				rtd_mul(tmp, get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, k));
				rtd_sub(tmp, get_tdmatrix_ij(a, j, k), tmp);
				set_tdmatrix_ij(a, j, k, tmp);
			}
		}

		for(j = 0; j < i; j++)
		{
			rtd_neg(tmp, aii); /* tmp := -aii */
			rtd_mul(tmp, tmp, get_tdmatrix_ij(a, j, i));
			set_tdmatrix_ij(a, j, i, tmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			rtd_neg(tmp, aii); /* tmp := -aii */
			rtd_mul(tmp, tmp, get_tdmatrix_ij(a, j, i));
			set_tdmatrix_ij(a, j, i, tmp);
		}
	}
}

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpf)a */
void subst_mpfvector_tdvec(MPFVector c, TDVector a)
{
	long int i;
	mpf_t tmp;

	mpf_init2(tmp, c->prec);

	for(i = 0; i < a->dim; i++)
	{
		mpf_set_td(tmp, get_tdvector_i(a, i));
		set_mpfvector_i(c, i, tmp);
	}

	mpf_clear(tmp);

}

/* c := (dd)a */
void subst_tdvector_mpfvec(TDVector c, MPFVector a)
{
	long int i;
	double tmp[TDSIZE];

	for(i = 0; i < a->dim; i++)
	{
		mpf_get_td(tmp, get_mpfvector_i(a, i));
		set_tdvector_i(c, i, tmp);
	}

}
/* c := (mpf)a */
void subst_mpfmatrix_tdmat(MPFMatrix c, TDMatrix a)
{
	long int i, j;
	mpf_t tmp;

	mpf_init2(tmp, c->prec);

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_mpfmatrixt_tdmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			mpf_set_td(tmp, get_tdmatrix_ij(a, i, j));
			set_mpfmatrix_ij(c, i, j, tmp);
		}
	}

	mpf_clear(tmp);
}

/* c := (dd)a */
void subst_tdmatrix_mpfmat(TDMatrix c, MPFMatrix a)
{
	long int i, j;
	double tmp[TDSIZE];

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_tdmatrix_mpfmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			mpf_get_td(tmp, get_mpfmatrix_ij(a, i, j));
			set_tdmatrix_ij(c, i, j, tmp);
		}
	}
}

/* Normwise relative error of vector */
void relerr_tdvector_mpfvec(double relerr[TDSIZE], TDVector approx_vec, MPFVector true_vec, int norm_type)
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
	subst_mpfvector_tdvec(mpf_approx_vec, approx_vec);

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
	//mpf_get_dd(relerr, mpf_relerr);
	mpf_get_td(relerr, mpf_relerr);

	free_mpfvector(diff_vec);
	free_mpfvector(mpf_approx_vec);
	mpf_clear(norm_diff_vec);
	mpf_clear(norm_true_vec);
	mpf_clear(mpf_relerr);

	return;
}

/* Elementwise relative errors of vector */
void relerr_element_tdvector_mpf(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double norm_relerr[TDSIZE], TDVector approx_vec, MPFVector true_vec, int norm_type)
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
	subst_mpfvector_tdvec(mpf_approx_vec, approx_vec);

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
	mpf_get_td(max_relerr, mpf_max_relerr);
	mpf_get_td(min_relerr, mpf_min_relerr);
	mpf_get_td(norm_relerr, mpf_norm_relerr);

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

//---------------------------------/
// TD <-> double
//---------------------------------/
/* c := (dd)a */
void subst_tdvector_dvec(TDVector c, DVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
	{
		set_tdvector_i_d(c, i, get_dvector_i(a, i));
	}
}

/* c := (d)a */
void subst_dvector_tdvec(DVector c, TDVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
	{
		c->element[i] = rtd_get_d(get_tdvector_i(a, i));
	}
}


/* c := (dd)a */
void subst_tdmatrix_dmat(TDMatrix c, DMatrix a)
{
	long int i, j;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_tdmatrix_dmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_tdmatrix_ij_d(c, i, j, get_dmatrix_ij(a, i, j));
		}
	}
}

/* c := (d)a */
void subst_dmatrix_tdmat(DMatrix c, TDMatrix a)
{
	long int i, j, ij_index;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_dmatrix_tdmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			ij_index = i * (c->col_dim) + j;
			c->element[ij_index] = rtd_get_d(get_tdmatrix_ij(a, i, j));
		}
	}
}

//---------------------------------/
// TD <-> DD
//---------------------------------/
#ifdef USE_DDLINEAR
/* c := (dd)a */
void subst_tdvector_ddvec(TDVector c, DDVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
	{
		set_tdvector_i_dd(c, i, get_ddvector_i(a, i));
	}
}

/* c := (dd)a */
void subst_ddvector_tdvec(DDVector c, TDVector a)
{
	long int i;
	ddfloat val;

	for(i = 0; i < a->dim; i++)
	{
		set_ddvector_i(c, i, get_tdvector_i_dd(a, i));
	}
}


/* c := (dd)a */
void subst_tdmatrix_ddmat(TDMatrix c, DDMatrix a)
{
	long int i, j;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_tdmatrix_ddmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_tdmatrix_ij_dd(c, i, j, get_ddmatrix_ij(a, i, j));
		}
	}
}

/* c := (dd)a */
void subst_ddmatrix_tdmat(DDMatrix c, TDMatrix a)
{
	long int i, j, ij_index;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_ddmatrix_tdmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			//ij_index = i * (c->col_dim) + j;
			//c->element[ij_index] = rtd_get_dd(get_tdmatrix_ij(a, i, j));
			set_ddmatrix_ij(c, i, j, get_tdmatrix_ij_dd(a, i, j));
		}
	}
}
#endif // USE_DDLINEAR

/* Normwise relative error of vector */
void relerr_tdvector(double relerr[TDSIZE], TDVector approx_vec, TDVector true_vec, int norm_type)
{
	double norm_true_vec[TDSIZE], norm_diff_vec[TDSIZE];
	TDVector diff_vec;

	diff_vec = init_tdvector(approx_vec->dim);

	// diff_vec := approx_vec - true_vec
	sub_tdvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_tdvector(norm_diff_vec, diff_vec);
			normi_tdvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_tdvector(norm_diff_vec, diff_vec);
			norm1_tdvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_tdvector(norm_diff_vec, diff_vec);
			norm2_tdvector(norm_true_vec, true_vec);
			break;
	}

	if(rtd_cmp_ui(norm_true_vec, 0UL) != 0)
		rtd_div(relerr, norm_diff_vec, norm_true_vec);

	free_tdvector(diff_vec);

	return;
}

/* Elementwise relative errors of vector */
void relerr_element_tdvector(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double norm_relerr[TDSIZE], TDVector approx_vec, TDVector true_vec, int norm_type)
{
	double abs_true_vec[TDSIZE], abs_diff_vec[TDSIZE], norm_diff_vec[TDSIZE], norm_true_vec[TDSIZE];
	long int i;
	TDVector diff_vec;

	diff_vec = init_tdvector(approx_vec->dim);

	// diff_vec := approx_vec - true_vec
	sub_tdvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_tdvector(norm_diff_vec, diff_vec);
			normi_tdvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_tdvector(norm_diff_vec, diff_vec);
			norm1_tdvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_tdvector(norm_diff_vec, diff_vec);
			norm2_tdvector(norm_true_vec, true_vec);
			break;
	}

	rtd_set(norm_relerr, norm_diff_vec);
	if(rtd_cmp_ui(norm_true_vec, 0UL) != 0)
		rtd_div(norm_relerr, norm_diff_vec, norm_true_vec);

	// relative errors of each elements
	rtd_set_ui(max_relerr, 0UL);
	normi_tdvector(min_relerr, diff_vec);
	for(i = 0; i < approx_vec->dim; i++)
	{
		rtd_abs(abs_diff_vec, get_tdvector_i(diff_vec, i));
		rtd_abs(abs_true_vec, get_tdvector_i(true_vec, i));
		if(rtd_cmp_ui(abs_true_vec, 0UL) != 0)
			rtd_div(abs_diff_vec, abs_diff_vec, abs_true_vec);
		
		if(rtd_cmp(max_relerr, abs_diff_vec) < 0)
			rtd_set(max_relerr, abs_diff_vec);
		if(rtd_cmp(min_relerr, abs_diff_vec) > 0)
			rtd_set(min_relerr, abs_diff_vec);
	}

	free_tdvector(diff_vec);// Fix! 2012-06-03 by T.Kouya

	return;
}

// TD

#if 0
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
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	float64x2_t dtmp256[TDSIZE], aji256[TDSIZE], ajk256[TDSIZE], aik256[TDSIZE];
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	svfloat64_t dtmp_0, dtmp_1, dtmp_2;
	svfloat64_t aji_0, aji_1, aji_2;
	svfloat64_t ajk_0, ajk_1, ajk_2;
	svfloat64_t aik_0, aik_1, aik_2;
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
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->real_col_dim;
		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->real_col_dim + i;
			aji256[0] = vdupq_n_f64(a->element[0][index_ji]);
			aji256[1] = vdupq_n_f64(a->element[1][index_ji]);
			aji256[2] = vdupq_n_f64(a->element[2][index_ji]);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rtd_mul(dtmp1, get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, k));
				rtd_sub(dtmp, get_tdmatrix_ij(a, j, k), dtmp1);
				set_tdmatrix_ij(a, j, k, dtmp);
			}
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				aik256[0] = vld1q_f64(&(a->element[0][index_ik]));
				aik256[1] = vld1q_f64(&(a->element[1][index_ik]));
				aik256[2] = vld1q_f64(&(a->element[2][index_ik]));
				_bncneon_rtd_mul(dtmp256, aji256, aik256);
				index_jk = j * a->real_col_dim + k;
				ajk256[0] = vld1q_f64(&(a->element[0][index_jk]));
				ajk256[1] = vld1q_f64(&(a->element[1][index_jk]));
				ajk256[2] = vld1q_f64(&(a->element[2][index_jk]));
				_bncneon_rtd_sub(dtmp256, ajk256, dtmp256);
				vst1q_f64(&(a->element[0][index_jk]), dtmp256[0]);
				vst1q_f64(&(a->element[1][index_jk]), dtmp256[1]);
				vst1q_f64(&(a->element[2][index_jk]), dtmp256[2]);
			}
		}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->real_col_dim;
		for(j = (i + 1); j < dim; j++)
		{
			index_ji = j * a->real_col_dim + i;
			aji_0 = svdup_f64(a->element[0][index_ji]);
			aji_1 = svdup_f64(a->element[1][index_ji]);
			aji_2 = svdup_f64(a->element[2][index_ji]);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			{
				rtd_mul(dtmp1, get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, k));
				rtd_sub(dtmp, get_tdmatrix_ij(a, j, k), dtmp1);
				set_tdmatrix_ij(a, j, k, dtmp);
			}
			for(k = dim_start; k < dim_end; k += (long int)svcntd())
			{
				svbool_t pg = svwhilelt_b64_s64((int64_t)k, (int64_t)dim_end);
				index_ik = i * a->real_col_dim + k;
				aik_0 = svld1_f64(pg, &(a->element[0][index_ik]));
				aik_1 = svld1_f64(pg, &(a->element[1][index_ik]));
				aik_2 = svld1_f64(pg, &(a->element[2][index_ik]));
				_bncsve2_rtd_mul(svptrue_b64(), &dtmp_0, &dtmp_1, &dtmp_2, aji_0, aji_1, aji_2, aik_0, aik_1, aik_2);
				index_jk = j * a->real_col_dim + k;
				ajk_0 = svld1_f64(pg, &(a->element[0][index_jk]));
				ajk_1 = svld1_f64(pg, &(a->element[1][index_jk]));
				ajk_2 = svld1_f64(pg, &(a->element[2][index_jk]));
				_bncsve2_rtd_neg(svptrue_b64(), &dtmp_0, &dtmp_1, &dtmp_2, dtmp_0, dtmp_1, dtmp_2);
				_bncsve2_rtd_add(svptrue_b64(), &dtmp_0, &dtmp_1, &dtmp_2, ajk_0, ajk_1, ajk_2, dtmp_0, dtmp_1, dtmp_2);
				svst1_f64(pg, &(a->element[0][index_jk]), dtmp_0);
				svst1_f64(pg, &(a->element[1][index_jk]), dtmp_1);
				svst1_f64(pg, &(a->element[2][index_jk]), dtmp_2);
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

/* Backward */
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
#endif // 0

#ifdef __cplusplus
} // extern "C"
#endif
