/********************************************************************************/
/* ddlinear.c: Double-double precision Linear Computation Library               */
/* Copyright (C) 2015-2025 Tomonori Kouya                                       */
/* Helped by ChatGPT 5 and Claude                                               */
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
#include "ddlinear.h"

// #ifndef _BNC_D_WIDTH
//     #define _BNC_D_WIDTH 2
// #endif // _BNC_D_WIDTH

#ifdef USE_GMP
#include "gmp.h"
#include "mpfr.h"
#include "mpfr_dtq_sd.h"
#include "mpflinear.h"
#endif //USE_GMP//

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// ddrel_diff
ddfloat ddrel_diff(ddfloat a, ddfloat b)
{
    ddfloat rel_diff, abs_a;

    //rel_diff = fabs(a - b);
    rdd_sub(rel_diff.val, a.val, b.val);
    rdd_abs(rel_diff.val, rel_diff.val);

    //if(a != 0.0)
    if(rdd_cmp_ui(a.val, 0UL) != 0)
    {
//        rel_diff /= fabs(a);
        rdd_abs(abs_a.val, a.val);
        rdd_div(rel_diff.val, rel_diff.val, a.val);
    }

    return rel_diff;
}

ddfloat ddrel_diff_array(ddfloat approx_a[], ddfloat approx_b[], int dim, int print_flag)
{
    int i;
    ddfloat rel_min, rel_max, rel_ave, rel_diff;

    rel_diff = ddrel_diff(approx_a[0], approx_b[0]);
    rdd_set(rel_min.val, rel_diff.val);
    rdd_set(rel_max.val, rel_diff.val);
    rdd_set(rel_ave.val, rel_diff.val);

    for(i = 1; i < dim; i++)
    {
        rel_diff = ddrel_diff(approx_a[i], approx_b[i]);
        if(rdd_cmp(rel_diff.val, rel_min.val) < 0) rdd_set(rel_min.val, rel_diff.val);
        if(rdd_cmp(rel_diff.val, rel_max.val) > 0) rdd_set(rel_max.val, rel_diff.val);
        //rel_ave += rel_diff;
        rdd_add(rel_ave.val, rel_ave.val, rel_diff.val);
    }
    //rel_ave /= (ddfloat)dim;
    rdd_div_ui(rel_ave.val, rel_ave.val, (unsigned long)dim);

    if(print_flag == 1)
    {
        printf("max_rel_diff, min_rel_diff, ave_rel_diff:"); rdd_out_str(rel_max.val); printf(" "); rdd_out_str(rel_min.val);  printf(" "); rdd_out_str(rel_ave.val); printf("\n"); 
    }

    return rel_max;
}

#if defined(USE_GMP) && defined(USE_MPFR)
// Frobenius norm
ddfloat ddnormf(ddfloat array[], int dim)
{
    int i;
    ddfloat ret, tmp;
    mpfr_t mpfr_ret;

    rdd_set_ui(ret.val, 0UL);
    for(i = 0; i < dim; i++)
    {
        rdd_mul(tmp.val, array[i].val, array[i].val);
        rdd_add(ret.val, ret.val, tmp.val);
    }
    //printf("ret.val = "); rdd_out_str(ret.val); printf("\n");
//  rdd_sqrt(ret, ret);
    mpfr_init2(mpfr_ret, 128);
    mpfr_set_dd(mpfr_ret, ret.val, MPFR_RNDN);
    mpfr_sqrt(mpfr_ret, mpfr_ret, MPFR_RNDN);
    mpfr_get_dd(ret.val, mpfr_ret, MPFR_RNDN);
    mpfr_clear(mpfr_ret);
    return ret;
}

// generate a text vector: vec(i) := sqrt(sqrt_seed) * (i + 1)
/* void set_test_ddvector(ddfloat vec[], int sqrt_seed, int dim)
{
    int i, j;
    ddfloat ddsqrt;
    mpfr_t mpfrsqrt;

    // ddsqrt := sqrt(sqrt_seed)
    mpfr_init2(mpfrsqrt, 128);
    mpfr_sqrt_ui(mpfrsqrt, (unsigned long)sqrt_seed, MPFR_RNDN);
    mpfr_get_dd(ddsqrt.val, mpfrsqrt, MPFR_RNDN);
//    rdd_set_ui(ddsqrt.val, sqrt_seed);
    //rdd_sqrt(ddsqrt.val, ddsqrt.val);
    //rdd_sqrt_ui(ddsqrt.val, sqrt_seed);

    //printf("set_test_ddmatrix: coef = "); rdd_out_str(ddsqrt.val); printf("\n");

    for(i = 0; i < dim; i++)
    {
        //printf("%5d: ", i);
        rdd_set_ui(vec[i].val, i + 1);
        rdd_mul(vec[i].val, vec[i].val, ddsqrt.val);
    }
} */
#endif // defined(USE_GMP) && defined(USE_MPFR)

// initialize DDVector
DDVector init_ddvector(int dimension)
{
	DDVector ret = NULL;
	long int i, real_dim;

	if(dimension <= 0)
	{
		fprintf(stderr, "ERROR: init_ddvector\n");
		return ret;
	}

	//ret = (DDVector)BNC_MALLOC(sizeof(ddvector));
	ret = (DDVector)malloc(sizeof(ddvector)); // Fix it!: 2024-04-26(Fri) T.Kouya
	if(ret == NULL)
		return ret;

	// real_dim is the nearest positive multiplier of _BNC_D_WIDTH
	real_dim = (long int)ceil((double)(dimension) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;

	ret->element[0] = (double *)BNC_CALLOC(real_dim, sizeof(double));
	if(ret->element[0] == NULL)
	{ 	
		free(ret);
		return NULL;
	}
	ret->element[1] = (double *)BNC_CALLOC(real_dim, sizeof(double));
	if(ret->element[1] == NULL)
	{
		free(ret->element[0]);
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
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(i = 0; i < real_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&(ret->element[0][i]), zero8);
		_mm512_store_pd(&(ret->element[1][i]), zero8);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	svfloat64_t z2 = svdup_f64(0.0);

	for(i = 0; i < real_dim; i += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(real_dim));
        svst1_f64(pg, &(ret->element[0][i]), z2);
        svst1_f64(pg, &(ret->element[1][i]), z2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon|| defined(__aarch64__)
	float64x2_t z2 = vdupq_n_f64(0.0);

	for(i = 0; i < real_dim; i += _BNC_D_WIDTH){
        vst1q_f64(&(ret->element[0][i]), z2);
        vst1q_f64(&(ret->element[1][i]), z2);
    }
#else // others
	for(i = 0; i < dimension; i++)
	{
		ret->element[0][i] = 0.0;
		ret->element[1][i] = 0.0;
	}
#endif // __AVX2__

	ret->dim = dimension;
	ret->real_dim = real_dim;

	return ret;
}

// free DDVector
void free_ddvector(DDVector vec)
{
    int i;

    for(i = 0; i < DDSIZE; i++)
        free(vec->element[i]);

    free(vec);
}

// DDVector vec -> ddfloat array
void set_ddfloat_ddvec(ddfloat ret[], int ret_dim, DDVector vec)
{
    int index, j, dim;

    dim = (ret_dim < vec->dim) ? ret_dim : vec->dim;

    for(index = 0; index < dim; index++)
    {
        for(j = 0; j < DDSIZE; j++)
            ret[index].val[j] = vec->element[j][index];
    }
}

// ddfloat array -> DDVector ret
void set_ddvector_ddfloat(DDVector ret, ddfloat array[], int array_dim)
{
    int index, j, dim;

    dim = (ret->dim < array_dim) ? ret->dim : array_dim;

    for(index = 0; index < dim; index++)
    {
        for(j = 0; j < DDSIZE; j++)
            ret->element[j][index] = array[index].val[j];
    }
}

// print ddvector
void print_ddvector(DDVector vec)
{
	long int index;

	for(index = 0; index < vec->dim; index++)
	{
		printf("%4ld: ", index);
		//c_dd_write((vec->element + index * DDSIZE));
		rdd_out_str(GET_DDVECTOR_I(vec, index));
		printf("\n");
	}
}

// set a zero vector
void set0_ddvector(DDVector vec)
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
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(i = 0; i < vec->real_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&vec->element[0][i], zero8);
		_mm512_store_pd(&vec->element[1][i], zero8);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	svfloat64_t z2 = svdup_f64(0.0);

	for(i = 0; i < vec->real_dim; i += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(vec->real_dim));
        svst1_f64(pg, &vec->element[0][i], z2);
        svst1_f64(pg, &vec->element[1][i], z2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon|| defined(__aarch64__)
	float64x2_t z2 = vdupq_n_f64(0.0);

	for(i = 0; i < vec->real_dim; i += _BNC_D_WIDTH){
        vst1q_f64(&vec->element[0][i], z2);
        vst1q_f64(&vec->element[1][i], z2);
    }
#else // others
	for(i = 0; i < vec->dim; i++)
	{
		vec->element[0][i] = 0.0;
		vec->element[1][i] = 0.0;
	}
#endif // __AVX2__
}

// set_ddvector_i_str
void set_ddvector_i_str(DDVector vec, long int index, const char *str)
{
	double tmp[DDSIZE];

	//rdd_get_str(tmp, str);
	rdd_set_str(tmp, str);

	set_ddvector_i(vec, index, tmp);
}

/*************************************************/
/* Vector Calculations for DDVector               */
/*
void add_ddvector(DDVector c, DDVector a, DDVector b)
void add2_ddvector(DDVector c, DDVector a)
void sub_ddvector(DDVector c, DDVector a, DDVector b)
void sub2_ddvector(DDVector c, DVector a)
void cmul_ddvector(DDVector c, double val[DDSIZE], DDVector a)
void cmul2_ddvector(DDVector c, double val[DDSIZE])
void add_cmul_ddvector(DDVector c, DDVector a, double val[DDSIZE], DDVector b)
double ip_ddvector(DDVector a, DDVector b)
double norm1_ddvector(DDVector a)
double norm2_ddvector(DDVector a)
double normi_ddvector(DDVector a)
void subst_ddvector(DDVector c, DDVector a)
*/
/*************************************************/
/* c = a + b */
void add_ddvector(DDVector c, DDVector a, DDVector b)
{
    long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_ddvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256d in_ret[DDSIZE], in_a_val[DDSIZE], in_b_val[DDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_pd(&(a->element[1][index]));
        in_b_val[0] = _mm256_load_pd(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_pd(&(b->element[1][index]));

        _bncavx2_rdd_add(in_ret, in_a_val, in_b_val);

        _mm256_store_pd(&(c->element[0][index]), in_ret[0]);
        _mm256_store_pd(&(c->element[1][index]), in_ret[1]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512d in_ret[DDSIZE], in_a_val[DDSIZE], in_b_val[DDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm512_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_pd(&(a->element[1][index]));
        in_b_val[0] = _mm512_load_pd(&(b->element[0][index]));
        in_b_val[1] = _mm512_load_pd(&(b->element[1][index]));

        _bncavx512_rdd_add(in_ret, in_a_val, in_b_val);

        _mm512_store_pd(&(c->element[0][index]), in_ret[0]);
        _mm512_store_pd(&(c->element[1][index]), in_ret[1]);
   }

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	
	svfloat64_t r_0, r_1;
	svfloat64_t ax_0, ax_1;
	svfloat64_t bx_0, bx_1;

	for(index = 0; index < c->real_dim; index += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        ax_0 = svld1_f64(pg, &(a->element[0][index]));
        ax_1 = svld1_f64(pg, &(a->element[1][index]));
        bx_0 = svld1_f64(pg, &(b->element[0][index]));
        bx_1 = svld1_f64(pg, &(b->element[1][index]));

		_bncsve2_rdd_add(pg, &r_0, &r_1, ax_0, ax_1, bx_0, bx_1);

		svst1_f64(pg, &(c->element[0][index]), r_0);
        svst1_f64(pg, &(c->element[1][index]), r_1);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon|| defined(__aarch64__)
	float64x2_t r[DDSIZE], ax[DDSIZE], bx[DDSIZE];

	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0] = vld1q_f64(&(a->element[0][index]));
        ax[1] = vld1q_f64(&(a->element[1][index]));
        bx[0] = vld1q_f64(&(b->element[0][index]));
        bx[1] = vld1q_f64(&(b->element[1][index]));

		_bncneon_rdd_add(r, ax, bx);

		vst1q_f64(&(c->element[0][index]), r[0]);
        vst1q_f64(&(c->element[1][index]), r[1]);
    }
#else // others
	double tmp[DDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rdd_add(tmp, get_ddvector_i(a, i),  get_ddvector_i(b, i));
		set_ddvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c += a */
void add2_ddvector(DDVector c, DDVector a)
{
	long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: add2_ddvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256d in_ret[DDSIZE], in_a_val[DDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_pd(&(a->element[1][index]));
        in_ret[0] = _mm256_load_pd(&(c->element[0][index]));
        in_ret[1] = _mm256_load_pd(&(c->element[1][index]));

        _bncavx2_rdd_add(in_ret, in_ret, in_a_val);

        _mm256_store_pd(&(c->element[0][index]), in_ret[0]);
        _mm256_store_pd(&(c->element[1][index]), in_ret[1]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512d in_ret[DDSIZE], in_a_val[DDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm512_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_pd(&(a->element[1][index]));
        in_ret[0] = _mm512_load_pd(&(c->element[0][index]));
        in_ret[1] = _mm512_load_pd(&(c->element[1][index]));

        _bncavx512_rdd_add(in_ret, in_ret, in_a_val);

        _mm512_store_pd(&(c->element[0][index]), in_ret[0]);
        _mm512_store_pd(&(c->element[1][index]), in_ret[1]);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	
	svfloat64_t r_0, r_1;
	svfloat64_t ax_0, ax_1;

	for(index = 0; index < c->real_dim; index += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        ax_0 = svld1_f64(pg, &(a->element[0][index]));
        ax_1 = svld1_f64(pg, &(a->element[1][index]));
        r_0  = svld1_f64(pg, &(c->element[0][index]));
        r_1  = svld1_f64(pg, &(c->element[1][index]));

		_bncsve2_rdd_add(pg, &r_0, &r_1, r_0, r_1, ax_0, ax_1);

		svst1_f64(pg, &(c->element[0][index]), r_0);
        svst1_f64(pg, &(c->element[1][index]), r_1);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon|| defined(__aarch64__)
	float64x2_t r[DDSIZE], ax[DDSIZE];

	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0] = vld1q_f64(&(a->element[0][index]));
        ax[1] = vld1q_f64(&(a->element[1][index]));
        r[0]  = vld1q_f64(&(c->element[0][index]));
        r[1]  = vld1q_f64(&(c->element[1][index]));

		_bncneon_rdd_add(r, r, ax);

		vst1q_f64(&(c->element[0][index]), r[0]);
        vst1q_f64(&(c->element[1][index]), r[1]);
    }
#else // others
	double tmp[DDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rdd_add(tmp, get_ddvector_i(c, i), get_ddvector_i(a, i));
		set_ddvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c = a - b */
void sub_ddvector(DDVector c, DDVector a, DDVector b)
{
    long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_ddvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256d in_ret[DDSIZE], in_a_val[DDSIZE], in_b_val[DDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_pd(&(a->element[1][index]));
        in_b_val[0] = _mm256_load_pd(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_pd(&(b->element[1][index]));

        _bncavx2_rdd_sub(in_ret, in_a_val, in_b_val);

        _mm256_store_pd(&(c->element[0][index]), in_ret[0]);
        _mm256_store_pd(&(c->element[1][index]), in_ret[1]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512d in_ret[DDSIZE], in_a_val[DDSIZE], in_b_val[DDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm512_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_pd(&(a->element[1][index]));
        in_b_val[0] = _mm512_load_pd(&(b->element[0][index]));
        in_b_val[1] = _mm512_load_pd(&(b->element[1][index]));

        _bncavx512_rdd_sub(in_ret, in_a_val, in_b_val);

        _mm512_store_pd(&(c->element[0][index]), in_ret[0]);
        _mm512_store_pd(&(c->element[1][index]), in_ret[1]);
   }

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	
	svfloat64_t r_0, r_1;
	svfloat64_t ax_0, ax_1;
	svfloat64_t bx_0, bx_1;

	for(index = 0; index < c->real_dim; index += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        ax_0 = svld1_f64(pg, &(a->element[0][index]));
        ax_1 = svld1_f64(pg, &(a->element[1][index]));
        bx_0 = svld1_f64(pg, &(b->element[0][index]));
        bx_1 = svld1_f64(pg, &(b->element[1][index]));

		_bncsve2_rdd_neg(pg, &r_0, &r_1, bx_0, bx_1);
		_bncsve2_rdd_add(pg, &r_0, &r_1, ax_0, ax_1, r_0, r_1);

		svst1_f64(pg, &(c->element[0][index]), r_0);
        svst1_f64(pg, &(c->element[1][index]), r_1);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon|| defined(__aarch64__)
	float64x2_t r[DDSIZE], ax[DDSIZE], bx[DDSIZE];

	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0] = vld1q_f64(&(a->element[0][index]));
        ax[1] = vld1q_f64(&(a->element[1][index]));
        bx[0] = vld1q_f64(&(b->element[0][index]));
        bx[1] = vld1q_f64(&(b->element[1][index]));

		_bncneon_rdd_sub(r, ax, bx);

		vst1q_f64(&(c->element[0][index]), r[0]);
        vst1q_f64(&(c->element[1][index]), r[1]);
    }
#else // others
	double tmp[DDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rdd_sub(tmp, get_ddvector_i(a, i),  get_ddvector_i(b, i));
		set_ddvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c -= a */
void sub2_ddvector(DDVector c, DDVector a)
{
	long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: add2_ddvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256d in_ret[DDSIZE], in_a_val[DDSIZE];

	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_pd(&(a->element[1][index]));
        in_ret[0] = _mm256_load_pd(&(c->element[0][index]));
        in_ret[1] = _mm256_load_pd(&(c->element[1][index]));

        _bncavx2_rdd_sub(in_ret, in_ret, in_a_val);

        _mm256_store_pd(&(c->element[0][index]), in_ret[0]);
        _mm256_store_pd(&(c->element[1][index]), in_ret[1]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512d in_ret[DDSIZE], in_a_val[DDSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
        in_a_val[0] = _mm512_load_pd(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_pd(&(a->element[1][index]));
        in_ret[0] = _mm512_load_pd(&(c->element[0][index]));
        in_ret[1] = _mm512_load_pd(&(c->element[1][index]));

        _bncavx512_rdd_sub(in_ret, in_ret, in_a_val);

        _mm512_store_pd(&(c->element[0][index]), in_ret[0]);
        _mm512_store_pd(&(c->element[1][index]), in_ret[1]);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	
	svfloat64_t r_0, r_1;
	svfloat64_t ax_0, ax_1;

	for(index = 0; index < c->real_dim; index += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        ax_0 = svld1_f64(pg, &(a->element[0][index]));
        ax_1 = svld1_f64(pg, &(a->element[1][index]));
        r_0  = svld1_f64(pg, &(c->element[0][index]));
        r_1  = svld1_f64(pg, &(c->element[1][index]));

		_bncsve2_rdd_neg(pg, &r_0, &r_1, ax_0, ax_1);
		_bncsve2_rdd_add(pg, &r_0, &r_1, r_0, r_1, r_0, r_1);

		svst1_f64(pg, &(c->element[0][index]), r_0);
        svst1_f64(pg, &(c->element[1][index]), r_1);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon|| defined(__aarch64__)
	float64x2_t r[DDSIZE], ax[DDSIZE];

	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        ax[0] = vld1q_f64(&(a->element[0][index]));
        ax[1] = vld1q_f64(&(a->element[1][index]));
        r[0]  = vld1q_f64(&(c->element[0][index]));
        r[1]  = vld1q_f64(&(c->element[1][index]));

		_bncneon_rdd_sub(r, r, ax);

		vst1q_f64(&(c->element[0][index]), r[0]);
        vst1q_f64(&(c->element[1][index]), r[1]);
    }
#else // others
	double tmp[DDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rdd_sub(tmp, get_ddvector_i(c, i), get_ddvector_i(a, i));
		set_ddvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c = val * a */
void cmul_ddvector(DDVector c, double val[DDSIZE], DDVector a)
{
    long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: cmul_dvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[DDSIZE], c4[DDSIZE], val4[DDSIZE];

	val4[0] = _mm256_set1_pd(val[0]);
	val4[1] = _mm256_set1_pd(val[1]);
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		//set_ddvector_i(c, i, val * get_ddvector_i(a, i));
		a4[0] = _mm256_load_pd(&(a->element[0][index]));
		a4[1] = _mm256_load_pd(&(a->element[1][index]));

		_bncavx2_rdd_mul(c4, val4, a4);

		_mm256_store_pd(&(c->element[0][index]), c4[0]);
		_mm256_store_pd(&(c->element[1][index]), c4[1]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[DDSIZE], c8[DDSIZE], val8[DDSIZE];

	val8[0] = _mm512_set1_pd(val[0]);
	val8[1] = _mm512_set1_pd(val[1]);
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		//set_ddvector_i(c, i, val * get_ddvector_i(a, i));
		a8[0] = _mm512_load_pd(&(a->element[0][index]));
		a8[1] = _mm512_load_pd(&(a->element[1][index]));

		_bncavx512_rdd_mul(c8, val8, a8);

		_mm512_store_pd(&(c->element[0][index]), c8[0]);
		_mm512_store_pd(&(c->element[1][index]), c8[1]);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	
	svfloat64_t a2_0, a2_1;
	svfloat64_t c2_0, c2_1;
	svfloat64_t v2_0, v2_1;

	v2_0 = svdup_f64(val[0]);
    v2_1 = svdup_f64(val[1]);
    for(index = 0; index < c->real_dim; index += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        a2_0 = svld1_f64(pg, &(a->element[0][index]));
        a2_1 = svld1_f64(pg, &(a->element[1][index]));

		_bncsve2_rdd_mul(pg, &c2_0, &c2_1, v2_0, v2_1, a2_0, a2_1);

		svst1_f64(pg, &(c->element[0][index]), c2_0);
        svst1_f64(pg, &(c->element[1][index]), c2_1);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon|| defined(__aarch64__)
	float64x2_t a2[DDSIZE], c2[DDSIZE], v2[DDSIZE];

	v2[0] = vdupq_n_f64(val[0]);
    v2[1] = vdupq_n_f64(val[1]);
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        a2[0] = vld1q_f64(&(a->element[0][index]));
        a2[1] = vld1q_f64(&(a->element[1][index]));

		_bncneon_rdd_mul(c2, v2, a2);

		vst1q_f64(&(c->element[0][index]), c2[0]);
        vst1q_f64(&(c->element[1][index]), c2[1]);
    }
#else // others
	double tmp[DDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rdd_mul(tmp, val, get_ddvector_i(a, i));
		set_ddvector_i(c, i, tmp);
	}
#endif // __AVX2__

}

/* c *= val */
void cmul2_ddvector(DDVector c, double val[DDSIZE])
{
	long int i, index;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d c4[DDSIZE], val4[DDSIZE];

	val4[0] = _mm256_set1_pd(val[0]);
	val4[1] = _mm256_set1_pd(val[1]);
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		//set_ddvector_i(c, i, val * get_ddvector_i(a, i));
		c4[0] = _mm256_load_pd(&(c->element[0][index]));
		c4[1] = _mm256_load_pd(&(c->element[1][index]));

		_bncavx2_rdd_mul(c4, val4, c4);

		_mm256_store_pd(&(c->element[0][index]), c4[0]);
		_mm256_store_pd(&(c->element[1][index]), c4[1]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d c8[DDSIZE], val8[DDSIZE];

	val8[0] = _mm512_set1_pd(val[0]);
	val8[1] = _mm512_set1_pd(val[1]);
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		//set_ddvector_i(c, i, val * get_ddvector_i(a, i));
		c8[0] = _mm512_load_pd(&(c->element[0][index]));
		c8[1] = _mm512_load_pd(&(c->element[1][index]));

		_bncavx512_rdd_mul(c8, val8, c8);

		_mm512_store_pd(&(c->element[0][index]), c8[0]);
		_mm512_store_pd(&(c->element[1][index]), c8[1]);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	
	svfloat64_t c2_0, c2_1;
	svfloat64_t v2_0, v2_1;

	v2_0 = svdup_f64(val[0]);
    v2_1 = svdup_f64(val[1]);

	for(index = 0; index < c->real_dim; index += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        c2_0 = svld1_f64(pg, &(c->element[0][index]));
        c2_1 = svld1_f64(pg, &(c->element[1][index]));

		_bncsve2_rdd_mul(pg, &c2_0, &c2_1, v2_0, v2_1, c2_0, c2_1);

		svst1_f64(pg, &(c->element[0][index]), c2_0);
        svst1_f64(pg, &(c->element[1][index]), c2_1);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon|| defined(__aarch64__)
	float64x2_t c2[DDSIZE], v2[DDSIZE];

	v2[0] = vdupq_n_f64(val[0]);
    v2[1] = vdupq_n_f64(val[1]);

	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        c2[0] = vld1q_f64(&(c->element[0][index]));
        c2[1] = vld1q_f64(&(c->element[1][index]));

		_bncneon_rdd_mul(c2, v2, c2);

		vst1q_f64(&(c->element[0][index]), c2[0]);
        vst1q_f64(&(c->element[1][index]), c2[1]);
    }
#else // others
	double tmp[DDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rdd_mul(tmp, val, get_ddvector_i(c, i));
		set_ddvector_i(c, i, tmp);
	}
#endif // __AVX2__	double tmp[DDSIZE];

}

/* c = a + val * b */
void add_cmul_ddvector(DDVector c, DDVector a, double val[DDSIZE], DDVector b)
{
	long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_cmul_ddvector\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[DDSIZE], b4[DDSIZE], c4[DDSIZE], val4[DDSIZE];

	val4[0] = _mm256_set1_pd(val[0]);
	val4[1] = _mm256_set1_pd(val[1]);
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		a4[0] = _mm256_load_pd(&(a->element[0][index]));
		a4[1] = _mm256_load_pd(&(a->element[1][index]));
		b4[0] = _mm256_load_pd(&(b->element[0][index]));
		b4[1] = _mm256_load_pd(&(b->element[1][index]));

//		rdd_mul(tmp, val, get_ddvector_i(b, i));
//		rdd_add(tmp, tmp, get_ddvector_i(a, i));
		_bncavx2_rdd_mul(c4, val4, b4);
		_bncavx2_rdd_add(c4, a4, c4);

		_mm256_store_pd(&(c->element[0][index]), c4[0]);
		_mm256_store_pd(&(c->element[1][index]), c4[1]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[DDSIZE], b8[DDSIZE], c8[DDSIZE], val8[DDSIZE];

	val8[0] = _mm512_set1_pd(val[0]);
	val8[1] = _mm512_set1_pd(val[1]);
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		a8[0] = _mm512_load_pd(&(a->element[0][index]));
		a8[1] = _mm512_load_pd(&(a->element[1][index]));
		b8[0] = _mm512_load_pd(&(b->element[0][index]));
		b8[1] = _mm512_load_pd(&(b->element[1][index]));

//		rdd_mul(tmp, val, get_ddvector_i(b, i));
//		rdd_add(tmp, tmp, get_ddvector_i(a, i));
		_bncavx512_rdd_mul(c8, val8, b8);
		_bncavx512_rdd_add(c8, a8, c8);

		_mm512_store_pd(&(c->element[0][index]), c8[0]);
		_mm512_store_pd(&(c->element[1][index]), c8[1]);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	
	svfloat64_t a2_0, a2_1;
	svfloat64_t b2_0, b2_1;
	svfloat64_t c2_0, c2_1;
	svfloat64_t v2_0, v2_1;

	v2_0 = svdup_f64(val[0]);
    v2_1 = svdup_f64(val[1]);
    for(index = 0; index < c->real_dim; index += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        a2_0 = svld1_f64(pg, &(a->element[0][index]));
        a2_1 = svld1_f64(pg, &(a->element[1][index]));
        b2_0 = svld1_f64(pg, &(b->element[0][index]));
        b2_1 = svld1_f64(pg, &(b->element[1][index]));

		_bncsve2_rdd_mul(pg, &c2_0, &c2_1, v2_0, v2_1, b2_0, b2_1);
        _bncsve2_rdd_add(pg, &c2_0, &c2_1, a2_0, a2_1, c2_0, c2_1);

		svst1_f64(pg, &(c->element[0][index]), c2_0);
        svst1_f64(pg, &(c->element[1][index]), c2_1);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon|| defined(__aarch64__)
	float64x2_t a2[DDSIZE], b2[DDSIZE], c2[DDSIZE], v2[DDSIZE];

	v2[0] = vdupq_n_f64(val[0]);
    v2[1] = vdupq_n_f64(val[1]);
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        a2[0] = vld1q_f64(&(a->element[0][index]));
        a2[1] = vld1q_f64(&(a->element[1][index]));
        b2[0] = vld1q_f64(&(b->element[0][index]));
        b2[1] = vld1q_f64(&(b->element[1][index]));

		_bncneon_rdd_mul(c2, v2, b2);
        _bncneon_rdd_add(c2, a2, c2);

		vst1q_f64(&(c->element[0][index]), c2[0]);
        vst1q_f64(&(c->element[1][index]), c2[1]);
    }
#else // others
	double tmp[DDSIZE];

	rdd_set0(tmp);
	for(i = 0; i < c->dim; i++)
	{
		//rdd_mul(tmp, val, get_ddvector_i(b, i));
		//rdd_add(tmp, tmp, get_ddvector_i(a, i));
		// 2026-03-23(Mon) T.Kouya
		// tmp := val * b + a
		rdd_fma(tmp, val, get_ddvector_i(b, i), get_ddvector_i(a, i));

		set_ddvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c = a - val * b */
void sub_cmul_ddvector(DDVector c, DDVector a, double val[DDSIZE], DDVector b)
{
	long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_cmul_ddvector\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[DDSIZE], b4[DDSIZE], c4[DDSIZE], val4[DDSIZE];

	val4[0] = _mm256_set1_pd(val[0]);
	val4[1] = _mm256_set1_pd(val[1]);
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		a4[0] = _mm256_load_pd(&(a->element[0][index]));
		a4[1] = _mm256_load_pd(&(a->element[1][index]));
		b4[0] = _mm256_load_pd(&(b->element[0][index]));
		b4[1] = _mm256_load_pd(&(b->element[1][index]));

//		rdd_mul(tmp, val, get_ddvector_i(b, i));
//		rdd_add(tmp, tmp, get_ddvector_i(a, i));
		_bncavx2_rdd_mul(c4, val4, b4);
		//_bncavx2_rdd_add(c4, a4, c4);
		_bncavx2_rdd_sub(c4, a4, c4);

		_mm256_store_pd(&(c->element[0][index]), c4[0]);
		_mm256_store_pd(&(c->element[1][index]), c4[1]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[DDSIZE], b8[DDSIZE], c8[DDSIZE], val8[DDSIZE];

	val8[0] = _mm512_set1_pd(val[0]);
	val8[1] = _mm512_set1_pd(val[1]);
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		a8[0] = _mm512_load_pd(&(a->element[0][index]));
		a8[1] = _mm512_load_pd(&(a->element[1][index]));
		b8[0] = _mm512_load_pd(&(b->element[0][index]));
		b8[1] = _mm512_load_pd(&(b->element[1][index]));

//		rdd_mul(tmp, val, get_ddvector_i(b, i));
//		rdd_add(tmp, tmp, get_ddvector_i(a, i));
		_bncavx512_rdd_mul(c8, val8, b8);
		//_bncavx512_rdd_add(c8, a8, c8);
		_bncavx512_rdd_sub(c8, a8, c8);

		_mm512_store_pd(&(c->element[0][index]), c8[0]);
		_mm512_store_pd(&(c->element[1][index]), c8[1]);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	
	svfloat64_t a2_0, a2_1;
	svfloat64_t b2_0, b2_1;
	svfloat64_t c2_0, c2_1;
	svfloat64_t v2_0, v2_1;

	v2_0 = svdup_f64(val[0]);
    v2_1 = svdup_f64(val[1]);
    for(index = 0; index < c->real_dim; index += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(c->real_dim));
        a2_0 = svld1_f64(pg, &(a->element[0][index]));
        a2_1 = svld1_f64(pg, &(a->element[1][index]));
        b2_0 = svld1_f64(pg, &(b->element[0][index]));
        b2_1 = svld1_f64(pg, &(b->element[1][index]));

		_bncsve2_rdd_mul(pg, &c2_0, &c2_1, v2_0, v2_1, b2_0, b2_1);
        _bncsve2_rdd_neg(pg, &c2_0, &c2_1, c2_0, c2_1);
		_bncsve2_rdd_add(pg, &c2_0, &c2_1, a2_0, a2_1, c2_0, c2_1);

		svst1_f64(pg, &(c->element[0][index]), c2_0);
        svst1_f64(pg, &(c->element[1][index]), c2_1);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon|| defined(__aarch64__)
	float64x2_t a2[DDSIZE], b2[DDSIZE], c2[DDSIZE], v2[DDSIZE];

	v2[0] = vdupq_n_f64(val[0]);
    v2[1] = vdupq_n_f64(val[1]);
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH){
        a2[0] = vld1q_f64(&(a->element[0][index]));
        a2[1] = vld1q_f64(&(a->element[1][index]));
        b2[0] = vld1q_f64(&(b->element[0][index]));
        b2[1] = vld1q_f64(&(b->element[1][index]));

		_bncneon_rdd_mul(c2, v2, b2);
        _bncneon_rdd_sub(c2, a2, c2);

		vst1q_f64(&(c->element[0][index]), c2[0]);
        vst1q_f64(&(c->element[1][index]), c2[1]);
    }
#else // others
	double tmp[DDSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rdd_mul(tmp, val, get_ddvector_i(b, i));
		//rdd_add(tmp, tmp, get_ddvector_i(a, i));
		rdd_sub(tmp, get_ddvector_i(a, i), tmp);
		set_ddvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* (a, b) */
void ip_ddvector(double ret[DDSIZE], DDVector a, DDVector b)
{
	long int i, index;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: ip_ddvector\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[DDSIZE], b4[DDSIZE], ret4[DDSIZE], tmp4[DDSIZE];

	_bncavx2_set0_dd(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		a4[0] = _mm256_load_pd(&(a->element[0][index]));
		a4[1] = _mm256_load_pd(&(a->element[1][index]));
		b4[0] = _mm256_load_pd(&(b->element[0][index]));
		b4[1] = _mm256_load_pd(&(b->element[1][index]));

//		rdd_mul(tmp, get_ddvector_i(a, i), get_ddvector_i(b, i));
//		rdd_add(ret, ret, tmp);
		_bncavx2_rdd_mul(tmp4, a4, b4);
		_bncavx2_rdd_add(ret4, ret4, tmp4);
	}
	_bncavx2_rdd_sum256d(ret, ret4);
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[DDSIZE], b8[DDSIZE], ret8[DDSIZE], tmp8[DDSIZE];

	_bncavx512_set0_dd(ret8);
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		a8[0] = _mm512_load_pd(&(a->element[0][index]));
		a8[1] = _mm512_load_pd(&(a->element[1][index]));
		b8[0] = _mm512_load_pd(&(b->element[0][index]));
		b8[1] = _mm512_load_pd(&(b->element[1][index]));

//		rdd_mul(tmp, get_ddvector_i(a, i), get_ddvector_i(b, i));
//		rdd_add(ret, ret, tmp);
		_bncavx512_rdd_mul(tmp8, a8, b8);
		_bncavx512_rdd_add(ret8, ret8, tmp8);
	}
	_bncavx2_rdd_sum512d(ret, ret8);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat64_t a2_0, a2_1;
	svfloat64_t b2_0, b2_1;
	svfloat64_t acc_0, acc_1;
	svfloat64_t tmp_0, tmp_1;

	_bncsve2_rdd_set0(&acc_0, &acc_1);
    for(index = 0; index < a->real_dim; index += (long int)svcntd()){
    		svbool_t pg = svwhilelt_b64_s32((int)index, (int)a->real_dim);
        a2_0 = svld1_f64(pg, &(a->element[0][index]));
        a2_1 = svld1_f64(pg, &(a->element[1][index]));
        b2_0 = svld1_f64(pg, &(b->element[0][index]));
        b2_1 = svld1_f64(pg, &(b->element[1][index]));

		_bncsve2_rdd_mul(svptrue_b64(), &tmp_0, &tmp_1, a2_0, a2_1, b2_0, b2_1);
        _bncsve2_rdd_add(svptrue_b64(), &acc_0, &acc_1, acc_0, acc_1, tmp_0, tmp_1);
    }
    _bncsve2_rdd_sum128d(svptrue_b64(), ret, acc_0, acc_1);
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon|| defined(__aarch64__)
	float64x2_t a2[DDSIZE], b2[DDSIZE], acc[DDSIZE], tmp[DDSIZE];

	_bncneon_set0_dd(acc);
    for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH){
        a2[0] = vld1q_f64(&(a->element[0][index]));
        a2[1] = vld1q_f64(&(a->element[1][index]));
        b2[0] = vld1q_f64(&(b->element[0][index]));
        b2[1] = vld1q_f64(&(b->element[1][index]));

		_bncneon_rdd_mul(tmp, a2, b2);
        _bncneon_rdd_add(acc, acc, tmp);
    }
    _bncneon_rdd_sum128d(ret, acc);
#else // others
	double tmp[DDSIZE];

	set0_dd(ret);
	for(i = 0; i < a->dim; i++)
	{
		//rdd_mul(tmp, get_ddvector_i(a, i), get_ddvector_i(b, i));
		//rdd_add(ret, ret, tmp);
		rdd_fma(ret, get_ddvector_i(a, i), get_ddvector_i(b, i), ret);
	}
#endif // __AVX2__

	return;
}

/* c := a */
void subst_ddvector(DDVector c, DDVector a)
{
	long int i;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
//	for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i));
		_mm256_store_pd(&(c->element[0][i]), _mm256_load_pd(&(a->element[0][i])));
		_mm256_store_pd(&(c->element[1][i]), _mm256_load_pd(&(a->element[1][i])));
	}
#elif defined(__AVX512F__) // __AVX512F__
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i));
		_mm512_store_pd(&(c->element[0][i]), _mm512_load_pd(&(a->element[0][i])));
		_mm512_store_pd(&(c->element[1][i]), _mm512_load_pd(&(a->element[1][i])));
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	for(i = 0; i < a->real_dim; i += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(a->real_dim));
        svfloat64_t t0 = svld1_f64(pg, &(a->element[0][i]));
        svfloat64_t t1 = svld1_f64(pg, &(a->element[1][i]));
        svst1_f64(pg, &(c->element[0][i]), t0);
        svst1_f64(pg, &(c->element[1][i]), t1);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon|| defined(__aarch64__)
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH){
        float64x2_t t0 = vld1q_f64(&(a->element[0][i]));
        float64x2_t t1 = vld1q_f64(&(a->element[1][i]));
        vst1q_f64(&(c->element[0][i]), t0);
        vst1q_f64(&(c->element[1][i]), t1);
    }
#else // others
	for(i = 0; i < a->dim; i++)
		set_ddvector_i(c, i, get_ddvector_i(a, i));
#endif // __AVX2__
}

/* c := -a */
void neg_ddvector(DDVector c, DDVector a)
{
	long int i;
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp[DDSIZE];

	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//rdd_neg(tmp, get_ddvector_i(a, i));
		//set_ddvector_i(c, i, tmp);
		tmp[0] = _bncavx2_dneg(_mm256_load_pd(&(a->element[0][i])));
		tmp[1] = _bncavx2_dneg(_mm256_load_pd(&(a->element[1][i])));
		_mm256_store_pd(&(c->element[0][i]), tmp[0]);
		_mm256_store_pd(&(c->element[1][i]), tmp[1]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp[DDSIZE];

	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//rdd_neg(tmp, get_ddvector_i(a, i));
		//set_ddvector_i(c, i, tmp);
		tmp[0] = _bncavx512_dneg(_mm512_load_pd(&(a->element[0][i])));
		tmp[1] = _bncavx512_dneg(_mm512_load_pd(&(a->element[1][i])));
		_mm512_store_pd(&(c->element[0][i]), tmp[0]);
		_mm512_store_pd(&(c->element[1][i]), tmp[1]);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	
	svfloat64_t t_0, t_1;

	for(i = 0; i < a->real_dim; i += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(a->real_dim));
        t_0 = svneg_f64_x(pg, svld1_f64(pg, &(a->element[0][i])));
        t_1 = svneg_f64_x(pg, svld1_f64(pg, &(a->element[1][i])));
        svst1_f64(pg, &(c->element[0][i]), t_0);
        svst1_f64(pg, &(c->element[1][i]), t_1);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon|| defined(__aarch64__)
	float64x2_t t[DDSIZE];

	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH){
        t[0] = vnegq_f64(vld1q_f64(&(a->element[0][i])));
        t[1] = vnegq_f64(vld1q_f64(&(a->element[1][i])));
        vst1q_f64(&(c->element[0][i]), t[0]);
        vst1q_f64(&(c->element[1][i]), t[1]);
    }
#else // others
	double tmp[DDSIZE];

	for(i = 0; i < a->dim; i++)
	{
		rdd_neg(tmp, get_ddvector_i(a, i));
		set_ddvector_i(c, i, tmp);
	}
#endif // __AVX2__
}


/* ||a||_1 */
void norm1_ddvector(double ret[DDSIZE], DDVector a)
{
	long int i, index, dim;

	dim = a->dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d vec4[DDSIZE], ret4[DDSIZE], tmp4[DDSIZE];

	_bncavx2_set0_dd(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		vec4[0] = _mm256_load_pd(&(a->element[0][index]));
		vec4[1] = _mm256_load_pd(&(a->element[1][index]));

		//rdd_abs(tmp, get_ddvector_i(a, i));
		//rdd_add(ret, ret, tmp);
		_bncavx2_rdd_abs(tmp4, vec4);
		_bncavx2_rdd_add(ret4, ret4, tmp4);
	}
	_bncavx2_rdd_abssum256d(ret, ret4);
#elif defined(__AVX512F__) // __AVX512F__
	__m512d vec8[DDSIZE], ret8[DDSIZE], tmp8[DDSIZE];

	_bncavx512_set0_dd(ret8);
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		vec8[0] = _mm512_load_pd(&(a->element[0][index]));
		vec8[1] = _mm512_load_pd(&(a->element[1][index]));

		//rdd_abs(tmp, get_ddvector_i(a, i));
		//rdd_add(ret, ret, tmp);
		_bncavx512_rdd_abs(tmp8, vec8);
		_bncavx512_rdd_add(ret8, ret8, tmp8);
	}
	_bncavx512_rdd_abssum512d(ret, ret8);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat64_t v_0, v_1;
	svfloat64_t acc_0, acc_1;
	svfloat64_t t_0, t_1;

	_bncsve2_rdd_set0(&acc_0, &acc_1);
    for(index = 0; index < a->real_dim; index += (long int)svcntd()){
    		svbool_t pg = svwhilelt_b64_s32((int)index, (int)a->real_dim);
        v_0 = svld1_f64(pg, &(a->element[0][index]));
        v_1 = svld1_f64(pg, &(a->element[1][index]));
        _bncsve2_rdd_abs(svptrue_b64(), &t_0, &t_1, v_0, v_1);
        _bncsve2_rdd_add(svptrue_b64(), &acc_0, &acc_1, acc_0, acc_1, t_0, t_1);
    }
    _bncsve2_rdd_abssum128d(svptrue_b64(), ret, acc_0, acc_1);
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon|| defined(__aarch64__)
	float64x2_t v[DDSIZE], acc[DDSIZE], t[DDSIZE];

	_bncneon_set0_dd(acc);
    for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH){
        v[0] = vld1q_f64(&(a->element[0][index]));
        v[1] = vld1q_f64(&(a->element[1][index]));
        _bncneon_rdd_abs(t, v);
        _bncneon_rdd_add(acc, acc, t);
    }
    _bncneon_rdd_abssum128d(ret, acc);
#else // others
	double tmp[DDSIZE];

	set0_dd(ret);
	for(i = 0; i < a->dim; i++)
	{
		rdd_abs(tmp, get_ddvector_i(a, i));
		rdd_add(ret, ret, tmp);
	}
#endif // __AVX2__

	return;
}

/* ||a||_infty */
void normi_ddvector(double ret[DDSIZE], DDVector a)
{
	double tmp[DDSIZE];
	long int i;

	rdd_abs(ret, get_ddvector_i(a, 0));
	for(i = 1; i < a->dim; i++)
	{
		rdd_abs(tmp, get_ddvector_i(a, i));
		if(rdd_cmp(ret, tmp) < 0)
			rdd_set(ret, tmp);
	}

	return;
}

// Euclid norm
void norm2_ddvector(double ret[DDSIZE], DDVector vec)
{
	long int i, index, dim;
	double tmp[DDSIZE];

	dim = vec->dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d vec4[DDSIZE], ret4[DDSIZE], tmp4[DDSIZE];

	_bncavx2_set0_dd(ret4);
	for(index = 0; index < vec->real_dim; index += _BNC_D_WIDTH)
	{
		vec4[0] = _mm256_load_pd(&(vec->element[0][index]));
		vec4[1] = _mm256_load_pd(&(vec->element[1][index]));

//		rdd_mul(tmp, get_ddvector_i(vec, i), get_ddvector_i(vec, i));
//		rdd_add(ret, ret, tmp);
		_bncavx2_rdd_mul(tmp4, vec4, vec4);
		_bncavx2_rdd_add(ret4, ret4, tmp4);
	}
	//_bncavx2_rdd_norm256d(ret, ret4);
	_bncavx2_rdd_sum256d(tmp, ret4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512d vec8[DDSIZE], ret8[DDSIZE], tmp8[DDSIZE];

	_bncavx512_set0_dd(ret8);
	for(index = 0; index < vec->real_dim; index += _BNC_D_WIDTH)
	{
		vec8[0] = _mm512_load_pd(&(vec->element[0][index]));
		vec8[1] = _mm512_load_pd(&(vec->element[1][index]));

//		rdd_mul(tmp, get_ddvector_i(a, i), get_ddvector_i(b, i));
//		rdd_add(ret, ret, tmp);
		_bncavx512_rdd_mul(tmp8, vec8, vec8);
		_bncavx512_rdd_add(ret8, ret8, tmp8);
	}
	//_bncavx512_rdd_norm512d(ret, ret8);
	_bncavx512_rdd_norm512d(tmp, ret8);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat64_t v_0, v_1;
	svfloat64_t acc_0, acc_1;
	svfloat64_t t_0, t_1; //double tmp[DDSIZE];

	_bncsve2_rdd_set0(&acc_0, &acc_1);
    for(index = 0; index < vec->real_dim; index += (long int)svcntd()){
    		svbool_t pg = svwhilelt_b64_s32((int)index, (int)vec->real_dim);
        v_0 = svld1_f64(pg, &(vec->element[0][index]));
        v_1 = svld1_f64(pg, &(vec->element[1][index]));
        _bncsve2_rdd_mul(svptrue_b64(), &t_0, &t_1, v_0, v_1, v_0, v_1);
        _bncsve2_rdd_add(svptrue_b64(), &acc_0, &acc_1, acc_0, acc_1, t_0, t_1);
    }
    _bncsve2_rdd_sum128d(svptrue_b64(), tmp, acc_0, acc_1);
    //rdd_sqrt(ret, tmp);
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon|| defined(__aarch64__)
	float64x2_t v[DDSIZE], acc[DDSIZE], t[DDSIZE]; //double tmp[DDSIZE];

	_bncneon_set0_dd(acc);
    for(index = 0; index < vec->real_dim; index += _BNC_D_WIDTH){
        v[0] = vld1q_f64(&(vec->element[0][index]));
        v[1] = vld1q_f64(&(vec->element[1][index]));
        _bncneon_rdd_mul(t, v, v);
        _bncneon_rdd_add(acc, acc, t);
    }
    _bncneon_rdd_sum128d(tmp, acc);
    //rdd_sqrt(ret, tmp);
#else // others
	//c_dd_copy_d((double)0.0, tmp);
	//c_dd_copy_d((double)0.0, ret);
	rdd_set0(tmp);
	rdd_set0(ret);

	for(i = 0; i < dim ; i++)
	{
		//c_dd_sqr(GET_DDVECTOR_I(vec, i), tmp);
		//c_dd_add(tmp, ret, ret);
		rdd_mul(ret, get_ddvector_i(vec, i), get_ddvector_i(vec, i));
		rdd_add(tmp, tmp, ret);
	}

	//c_dd_sqrt(ret, tmp);
	//c_dd_copy(tmp, ret);
	//rdd_set(ret, tmp);
#endif // __AVX2__

	rdd_sqrt(ret, tmp);
}

// set a zero matrix
//void set0_ddmatrix(DDMatrix mat)
void set0_ddmatrix(DDMatrix mat)
{
	long int i;
	long int real_total_dim;

	real_total_dim = mat->real_row_dim * mat->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&mat->element[0][i], zero4);
		_mm256_store_pd(&mat->element[1][i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&mat->element[0][i], zero8);
		_mm512_store_pd(&mat->element[1][i], zero8);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	svfloat64_t z2 = svdup_f64(0.0);

	for(i = 0; i < real_total_dim; i += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(real_total_dim));
        svst1_f64(pg, &mat->element[0][i], z2);
        svst1_f64(pg, &mat->element[1][i], z2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon|| defined(__aarch64__)
	float64x2_t z2 = vdupq_n_f64(0.0);

	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH){
        vst1q_f64(&mat->element[0][i], z2);
        vst1q_f64(&mat->element[1][i], z2);
    }
#else // others
	for(i = 0; i < real_total_dim; i++)
	{
		mat->element[0][i] = 0.0;
		mat->element[1][i] = 0.0;	
	}
#endif // __AVX2__
}

// initialize ddvector
DDMatrix init_ddmatrix(long int row_dim, long int col_dim)
{
	long int row_index, col_index, i;
	long int real_row_dim, real_col_dim, real_total_dim;
	DDMatrix ret = NULL;

	ret = (DDMatrix)BNC_MALLOC(sizeof(ddmatrix));
	if(ret == NULL)
		return ret;

	// real_dim is the nearest positive multiplier of _BNC_D_WIDTH
	real_row_dim = (long int)ceil((double)(row_dim) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
	real_col_dim = (long int)ceil((double)(col_dim) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
	real_total_dim = real_row_dim * real_col_dim;

	//printf("init_ddmatrix(%ld, %ld) %ld calloc\n", row_dim, col_dim, real_total_dim);
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

	//printf("init_ddmatrix(%ld, %ld) calloc\n", row_dim, col_dim);
	/* All 0 */
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&ret->element[0][i], zero4);
		_mm256_store_pd(&ret->element[1][i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&ret->element[0][i], zero8);
		_mm512_store_pd(&ret->element[1][i], zero8);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	svfloat64_t z2 = svdup_f64(0.0);

	for(i = 0; i < real_total_dim; i += (long int)svcntd()){
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(real_total_dim));
        svst1_f64(pg, &ret->element[0][i], z2);
        svst1_f64(pg, &ret->element[1][i], z2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon|| defined(__aarch64__)
	float64x2_t z2 = vdupq_n_f64(0.0);

	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH){
        vst1q_f64(&ret->element[0][i], z2);
        vst1q_f64(&ret->element[1][i], z2);
    }
#else // others
	for(i = 0; i < real_total_dim; i++)
	{
		ret->element[0][i] = 0.0;
		ret->element[1][i] = 0.0;
	}
#endif // __AVX2__

	ret->real_row_dim = real_row_dim;
	ret->real_col_dim = real_col_dim;
	ret->row_dim = row_dim;
	ret->col_dim = col_dim;

	return ret;
}

// free ddvector
void free_ddmatrix(DDMatrix mat)
{
	long int i;

	for(i = 0; i < DDSIZE; i++)
		free(mat->element[i]);

	free(mat);
}

// print ddmatrix
void print_ddmatrix(DDMatrix mat)
{
	long int row_index, col_index;

	for(row_index = 0; row_index < mat->row_dim; row_index++)
	{
		for(col_index = 0; col_index < mat->col_dim; col_index++)
		{
			printf("%ld, %ld: ", row_index, col_index);
	//		c_dd_write((vec->element + index * DDSIZE));
			//c_dd_write(GET_DDMATRIX_IJ(mat, row_index, col_index));
			rdd_out_str(GET_DDMATRIX_IJ(mat, row_index, col_index));
		}
	}
}

// print ddmatrix2
#if 0
void print_ddmatrix2(DDMatrix mat1, DDMatrix mat2)
{
	long int row_index, col_index;
	long int row_dim, col_dim;

    row_dim = (mat1->re->row_dim <= mat2->re->row_dim) ? mat1->re->row_dim : mat2->re->row_dim;
    col_dim = (mat1->re->col_dim <= mat2->re->col_dim) ? mat1->re->col_dim : mat2->re->col_dim;

	for(row_index = 0; row_index < row_dim; row_index++)
	{
		for(col_index = 0; col_index < col_dim; col_index++)
		{
			printf("%ld, %ld: ", row_index, col_index);
	//		c_dd_write((vec->element + index * DDSIZE));
			//c_dd_write(GET_DDMATRIX_IJ(mat, row_index, col_index));
			rdd_out_str(GET_DDMATRIX_IJ(mat1, row_index, col_index));
			printf(", ");
			rdd_out_str(GET_DDMATRIX_IJ(mat2, row_index, col_index));
			printf("\n");
		}
	}
}
#endif // 0


// DDMatrix mat -> ddfloat array
void set_ddfloat_ddmat(ddfloat ret[], int ret_dim, DDMatrix mat)
{
    int total_index, j, total_dim;
	long int row_index, col_index;

    total_dim = (ret_dim < (mat->row_dim * mat->col_dim)) ? ret_dim : (mat->row_dim * mat->col_dim);

	total_index = 0;
    for(row_index = 0; row_index < mat->row_dim; row_index++)
    {
		for(col_index = 0; (col_index < mat->col_dim) && (total_index < total_dim); col_index++)
		{
			for(j = 0; j < DDSIZE; j++)
				ret[total_index].val[j] = mat->element[j][(row_index * mat->real_col_dim) + col_index];

			total_index++;
		}
    }
}

// ddfloat array -> DDmatrix ret
void set_ddmatrix_ddfloat(DDMatrix ret, ddfloat array[], int array_dim)
{
    int total_index, j, total_dim;
	long int row_index, col_index;

    total_dim = (array_dim < (ret->row_dim * ret->col_dim)) ? array_dim : (ret->row_dim * ret->col_dim);

 	total_index = 0;
    for(row_index = 0; row_index < ret->row_dim; row_index++)
    {
		for(col_index = 0; (col_index < ret->col_dim) && (total_index < total_dim); col_index++)
		{
			for(j = 0; j < DDSIZE; j++)
				ret->element[j][(row_index * ret->real_col_dim) + col_index] = array[total_index].val[j];

			total_index++;
		}
    }
}

// small matrix multiplication
#include "small_ddmatrix.c"

// matrix multiplication
// ret := A * B
void mul_ddmatrix(DDMatrix ret, DDMatrix a, DDMatrix b)
{
	long int i, j, k;

	/* dimension check */
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: mul_ddmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret->row_dim, ret->col_dim, a->row_dim, a->col_dim, b->row_dim, b->col_dim);
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long real_row_dim, real_col_dim, real_mid_dim;
	double cijval[4][DDSIZE];
    __m256d cij[DDSIZE], aik[DDSIZE], bkj[DDSIZE], tmp_mul[DDSIZE];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j++)
        {
            //rdd_set_ui(cij.val, 0UL);
            cij[0] = _mm256_setzero_pd();
            cij[1] = _mm256_setzero_pd();
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
                
            /*    aik[1] = _mm256_set_pd(
                    a->element[1][i * real_mid_dim + k],
                    a->element[1][i * real_mid_dim + k + 1],
                    a->element[1][i * real_mid_dim + k + 2],
                    a->element[1][i * real_mid_dim + k + 3]
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

            /*
                rdd_mul(tmp_mul[0].val, aik[0].val, bkj[0].val);
                rdd_mul(tmp_mul[1].val, aik[1].val, bkj[1].val);
                rdd_mul(tmp_mul[2].val, aik[2].val, bkj[2].val);
                rdd_mul(tmp_mul[3].val, aik[3].val, bkj[3].val);
            */
                _bncavx2_rdd_mul(tmp_mul, aik, bkj);

            /*
                rdd_add(cij.val, cij.val, tmp_mul[0].val);
                rdd_add(cij.val, cij.val, tmp_mul[1].val);
                rdd_add(cij.val, cij.val, tmp_mul[2].val);
                rdd_add(cij.val, cij.val, tmp_mul[3].val);
            */
                _bncavx2_rdd_add(cij, cij, tmp_mul);
            }

            cijval[0][0] = cij[0][0]; cijval[0][1] = cij[1][0];
            cijval[1][0] = cij[0][1]; cijval[1][1] = cij[1][1];
            cijval[2][0] = cij[0][2]; cijval[2][1] = cij[1][2];
            cijval[3][0] = cij[0][3]; cijval[3][1] = cij[1][3];
            rdd_add(cijval[0], cijval[0], cijval[1]);
            rdd_add(cijval[0], cijval[0], cijval[2]);
            rdd_add(cijval[0], cijval[0], cijval[3]);

            ret->element[0][i * real_col_dim + j] = cijval[0][0];
            ret->element[1][i * real_col_dim + j] = cijval[0][1];
        }
    }
#elif defined(__AVX512F__) // __AVX512F__
	long real_row_dim, real_col_dim, real_mid_dim;
	double cijval[8][DDSIZE];
    __m512d cij[DDSIZE], aik[DDSIZE], bkj[DDSIZE], tmp_mul[DDSIZE];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j++)
        {
            //rdd_set_ui(cij.val, 0UL);
            cij[0] = _mm512_setzero_pd();
            cij[1] = _mm512_setzero_pd();
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

            /*
                rdd_mul(tmp_mul[0].val, aik[0].val, bkj[0].val);
                rdd_mul(tmp_mul[1].val, aik[1].val, bkj[1].val);
                rdd_mul(tmp_mul[2].val, aik[2].val, bkj[2].val);
                rdd_mul(tmp_mul[3].val, aik[3].val, bkj[3].val);
            */
                _bncavx512_rdd_mul(tmp_mul, aik, bkj);

            /*
                rdd_add(cij.val, cij.val, tmp_mul[0].val);
                rdd_add(cij.val, cij.val, tmp_mul[1].val);
                rdd_add(cij.val, cij.val, tmp_mul[2].val);
                rdd_add(cij.val, cij.val, tmp_mul[3].val);
            */
                _bncavx512_rdd_add(cij, cij, tmp_mul);
            }

            cijval[0][0] = cij[0][0]; cijval[0][1] = cij[1][0];
            cijval[1][0] = cij[0][1]; cijval[1][1] = cij[1][1];
            cijval[2][0] = cij[0][2]; cijval[2][1] = cij[1][2];
            cijval[3][0] = cij[0][3]; cijval[3][1] = cij[1][3];
            cijval[4][0] = cij[0][4]; cijval[4][1] = cij[1][4];
            cijval[5][0] = cij[0][5]; cijval[5][1] = cij[1][5];
            cijval[6][0] = cij[0][6]; cijval[6][1] = cij[1][6];
            cijval[7][0] = cij[0][7]; cijval[7][1] = cij[1][7];
            rdd_add(cijval[0], cijval[0], cijval[1]);
            rdd_add(cijval[0], cijval[0], cijval[2]);
            rdd_add(cijval[0], cijval[0], cijval[3]);
            rdd_add(cijval[0], cijval[0], cijval[4]);
            rdd_add(cijval[0], cijval[0], cijval[5]);
            rdd_add(cijval[0], cijval[0], cijval[6]);
            rdd_add(cijval[0], cijval[0], cijval[7]);

            ret->element[0][i * real_col_dim + j] = cijval[0][0];
            ret->element[1][i * real_col_dim + j] = cijval[0][1];
        }
    }

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic)
    {
        long real_row_dim = a->real_row_dim;
        long real_col_dim = b->real_col_dim;
        long real_mid_dim = a->real_col_dim;
        long vl = (long)svcntd();   /* double lanes per vector (VL-agnostic) */

        for(i = 0; i < real_row_dim; i++){
            for(j = 0; j < real_col_dim; j += vl){
                svbool_t pg = svwhilelt_b64_s64((int64_t)j, (int64_t)real_col_dim);
                svfloat64_t cij0, cij1;
                _bncsve2_rdd_set0(&cij0, &cij1);
                for(k = 0; k < real_mid_dim; k++){
                    svfloat64_t aik0 = svdup_n_f64(a->element[0][i*real_mid_dim + k]);
                    svfloat64_t aik1 = svdup_n_f64(a->element[1][i*real_mid_dim + k]);
                    svfloat64_t bkj0 = svld1_f64(pg, &(b->element[0][k*real_col_dim + j]));
                    svfloat64_t bkj1 = svld1_f64(pg, &(b->element[1][k*real_col_dim + j]));
                    svfloat64_t t0, t1;
                    _bncsve2_rdd_mul(pg, &t0, &t1, aik0, aik1, bkj0, bkj1);
                    _bncsve2_rdd_add(pg, &cij0, &cij1, cij0, cij1, t0, t1);
                }
                svst1_f64(pg, &(ret->element[0][i*real_col_dim + j]), cij0);
                svst1_f64(pg, &(ret->element[1][i*real_col_dim + j]), cij1);
            }
        }
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon|| defined(__aarch64__)
	long real_row_dim = a->real_row_dim;
    long real_col_dim = b->real_col_dim;
    long real_mid_dim = a->real_col_dim;
    float64x2_t cij[DDSIZE], aik[DDSIZE], bkj[DDSIZE], tmp[DDSIZE];

    for(i = 0; i < ret->real_row_dim; i += 1){
        for(j = 0; j < ret->real_col_dim; j += _BNC_D_WIDTH){
            _bncneon_set0_dd(cij);
            for(k = 0; k < real_mid_dim; k += 1){
                aik[0] = vdupq_n_f64(a->element[0][i*real_mid_dim + k]);
                aik[1] = vdupq_n_f64(a->element[1][i*real_mid_dim + k]);
                bkj[0] = vld1q_f64(&(b->element[0][k*real_col_dim + j]));
                bkj[1] = vld1q_f64(&(b->element[1][k*real_col_dim + j]));
                _bncneon_rdd_mul(tmp, aik, bkj);
                _bncneon_rdd_add(cij, cij, tmp);
            }
            vst1q_f64(&(ret->element[0][i*ret->real_col_dim + j]), cij[0]);
            vst1q_f64(&(ret->element[1][i*ret->real_col_dim + j]), cij[1]);
        }
    }
#else // __AVX2__
	long row_dim, col_dim, mid_dim;
	double tmp[DDSIZE], ret_ij[DDSIZE];

	// row_dim, col_dim <= 32
/*	if((ret->row_dim <= 32) && (ret->col_dim <= 32))
	{
		mul_small_ddmatrix_32x32(ret, a, b);
		return;
	}
*/
	//printf("Non SIMD mul_ddmatrix(%ld, %ld)\n", ret->row_dim, ret->col_dim);
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = a->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			//rdd_set0(GET_DDMATRIX_IJ(ret, i, j));
			rdd_set0(ret_ij);
			for(k = 0; k < mid_dim; k++)
			{
				//rdd_mul(tmp, GET_DDMATRIX_IJ(a, i, k), GET_DDMATRIX_IJ(b, k, j));
				//rdd_add(GET_DDMATRIX_IJ(ret, i, j), tmp, GET_DDMATRIX_IJ(ret, i, j));
				//rdd_add(ret_ij, tmp, ret_ij);
				rdd_fma(ret_ij, get_ddmatrix_ij(a, i, k), get_ddmatrix_ij(b, k, j), ret_ij);
			}
			set_ddmatrix_ij(ret, i, j, ret_ij);
		}
	}
	//printf("end(%ld, %ld)\n", ret->row_dim, ret->col_dim);
#endif // __AVX2__

}

// Frobenius norm
void normf_ddmatrix(double ret[DDSIZE], DDMatrix mat)
{
	long int i;
	long int real_total_dim;
	double tmp[DDSIZE];

	real_total_dim = mat->real_row_dim * mat->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d ret4[DDSIZE], mat4[DDSIZE], tmp4[DDSIZE];

	ret4[0] = _mm256_setzero_pd();
	ret4[1] = _mm256_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		mat4[0] = _mm256_load_pd(&(mat->element[0][i]));
		mat4[1] = _mm256_load_pd(&(mat->element[1][i]));

		// tmp4 := mat4[i]^2
		// ret4 += tmp4
		_bncavx2_rdd_mul(tmp4, mat4, mat4);
		_bncavx2_rdd_add(ret4, ret4, tmp4);
	}

	_bncavx2_rdd_sum256d(ret, ret4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512d ret8[DDSIZE], mat8[DDSIZE], tmp8[DDSIZE];

	ret8[0] = _mm512_setzero_pd();
	ret8[1] = _mm512_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		mat8[0] = _mm512_load_pd(&(mat->element[0][i]));
		mat8[1] = _mm512_load_pd(&(mat->element[1][i]));

		// tmp8 := mat8[i]^2
		// ret8 += tmp8
		_bncavx512_rdd_mul(tmp8, mat8, mat8);
		_bncavx512_rdd_add(ret8, ret8, tmp8);
	}

	_bncavx2_rdd_sum512d(ret, ret8);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)

    
	svfloat64_t ret2_0, ret2_1;
	svfloat64_t mat2_0, mat2_1;
	svfloat64_t tmp2_0, tmp2_1;

    /* ret2 := 0 */
    _bncsve2_rdd_set0(&ret2_0, &ret2_1);

    for(i = 0; i < real_total_dim; i += (long int)svcntd())
    {
    		svbool_t pg = svwhilelt_b64_s32((int)i, (int)real_total_dim);
        /* mat2 := mat の DD 要素 (hi, lo) をロード */
        mat2_0 = svld1_f64(pg, &(mat->element[0][i]));
        mat2_1 = svld1_f64(pg, &(mat->element[1][i]));

        /* tmp2 := mat2^2, ret2 += tmp2 */
        _bncsve2_rdd_mul(svptrue_b64(), &tmp2_0, &tmp2_1, mat2_0, mat2_1, mat2_0, mat2_1);
        _bncsve2_rdd_add(svptrue_b64(), &ret2_0, &ret2_1, ret2_0, ret2_1, tmp2_0, tmp2_1);
    }

    /* ret := sum(ret2_0, ret2_1) (2-lane の DD をスカラ DD に集約) */
    _bncsve2_rdd_sum128d(svptrue_b64(), ret, ret2_0, ret2_1);

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon|| defined(__aarch64__)

    float64x2_t ret2[DDSIZE], mat2[DDSIZE], tmp2[DDSIZE];

    /* ret2 := 0 */
    _bncneon_rdd_set0(ret2);

    for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
    {
        /* mat2 := mat の DD 要素 (hi, lo) をロード */
        mat2[0] = vld1q_f64(&(mat->element[0][i]));
        mat2[1] = vld1q_f64(&(mat->element[1][i]));

        /* tmp2 := mat2^2, ret2 += tmp2 */
        _bncneon_rdd_mul(tmp2, mat2, mat2);
        _bncneon_rdd_add(ret2, ret2, tmp2);
    }

    /* ret := sum(ret2[0], ret2[1]) (2-lane の DD をスカラ DD に集約) */
    _bncneon_rdd_sum128d(ret, ret2);

#else // others
	double mat1[DDSIZE];

	rdd_set0(ret);
	for(i = 0; i < real_total_dim; i++)
	{
		mat1[0] = mat->element[0][i];
		mat1[1] = mat->element[1][i];

		// tmp := mat1[i]^2
		// ret += tmp
		rdd_mul(tmp, mat1, mat1);
		rdd_add(ret, ret, tmp);
	}

#endif // __AVX2__

	rdd_sqrt(tmp, ret);
	rdd_set(ret, tmp);

}

// print normf
void print_normf_ddmatrix(const char *str, DDMatrix mat)
{
	static double tmp[DDSIZE];

	normf_ddmatrix(tmp, mat);

	if(str != NULL)
		printf("%s(%ld, %ld)", str, mat->row_dim, mat->col_dim);

	rdd_out_str(tmp); printf("\n");
}

/*************************************************/
/* Matrix Caluculations for DDMatrix            */
/*
void normf_ddmatrix(double ret[DDSIZE], DDMatrix mat)
void norm1_ddmatrix(double ret[DDSIZE], DDMatrix mat)
void normi_ddmatrix(double ret[DDSIZE], DDMatrix mat)
void add_ddmatrix(DDMatrix c, DDMatrix a, DDMatrix b);
void sub_ddmatrix(DDMatrix c, DDMatrix a, DDMatrix b);
void mul_ddmatrix(DDMatrix c, DDMatrix a, DDMatrix b);
void mul_ddmatrix_ddvec(DDVector v, DDMatrix a, DDVector vb)
void mul_ddmatrixt_ddvec(DDVector v, DDMatrix a, DDVector vb)
void transpose_ddmatrix(DDMatrix c, DDMatrix a);
void inv_ddmatrix(DDMatrix a);
void subst_ddmatrix(DDMatrix c, DDMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_ddmatrix(double ret[DDSIZE], DDMatrix mat)
{
	long int i, j;
	double tmp[DDSIZE], sum[DDSIZE];

	set0_dd(ret);
	for(i = 0; i < mat->row_dim; i++)
	{
		set0_dd(sum);
		for(j = 0; j < mat->col_dim; j++)
		{
			rdd_abs(tmp, get_ddmatrix_ij(mat, i, j));
			rdd_add(sum, sum, tmp);
		}
		if(rdd_cmp(ret, sum) < 0)
			rdd_set(ret, sum);
	}

	return;
}

/* 1 Norm of Matrix */
void norm1_ddmatrix(double ret[DDSIZE], DDMatrix mat)
{
	long int i, j;
	double tmp[DDSIZE], sum[DDSIZE];

	rdd_set_ui(ret, 0UL);

	for(j = 0; j < mat->col_dim; j++)
	{
		rdd_set_ui(sum, 0UL);
		for(i = 0; i < mat->row_dim; i++)
		{
			rdd_abs(tmp, get_ddmatrix_ij(mat, i, j));
			rdd_add(sum, sum, tmp);
		}
		if(rdd_cmp(ret, sum) < 0)
			rdd_set(ret, sum);
	}

	return;
}

/* c := a + b */
void add_ddmatrix(DDMatrix c, DDMatrix a, DDMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_ddmatrix: row_dim(c, a, b) = %ld, %ld, %ld\n", c->row_dim, a->row_dim, b->row_dim);
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_ddmatrix: col_dim(c, a, b) = %ld, %ld, %ld\n", c->col_dim, a->col_dim, b->col_dim);
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[DDSIZE], aij4[DDSIZE], bij4[DDSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij4[0] = _mm256_load_pd(&(a->element[0][index]));
		aij4[1] = _mm256_load_pd(&(a->element[1][index]));
		bij4[0] = _mm256_load_pd(&(b->element[0][index]));
		bij4[1] = _mm256_load_pd(&(b->element[1][index]));

		_bncavx2_rdd_add(tmp4, aij4, bij4);

		_mm256_store_pd(&(c->element[0][index]), tmp4[0]);
		_mm256_store_pd(&(c->element[1][index]), tmp4[1]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[DDSIZE], aij8[DDSIZE], bij8[DDSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij8[0] = _mm512_load_pd(&(a->element[0][index]));
		aij8[1] = _mm512_load_pd(&(a->element[1][index]));
		bij8[0] = _mm512_load_pd(&(b->element[0][index]));
		bij8[1] = _mm512_load_pd(&(b->element[1][index]));

		_bncavx512_rdd_add(tmp8, aij8, bij8);

		_mm512_store_pd(&(c->element[0][index]), tmp8[0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[1]); 
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t tmp2_0, tmp2_1;
	svfloat64_t aij2_0, aij2_1;
	svfloat64_t bij2_0, bij2_1;

    for(index = 0; index < real_total_dim; index += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(real_total_dim));
        aij2_0 = svld1_f64(pg, &(a->element[0][index]));
        aij2_1 = svld1_f64(pg, &(a->element[1][index]));
        bij2_0 = svld1_f64(pg, &(b->element[0][index]));
        bij2_1 = svld1_f64(pg, &(b->element[1][index]));

        _bncsve2_rdd_add(pg, &tmp2_0, &tmp2_1, aij2_0, aij2_1, bij2_0, bij2_1);

        svst1_f64(pg, &(c->element[0][index]), tmp2_0);
        svst1_f64(pg, &(c->element[1][index]), tmp2_1); 
    }

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t tmp2[DDSIZE], aij2[DDSIZE], bij2[DDSIZE];

    for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
    {
        aij2[0] = vld1q_f64(&(a->element[0][index]));
        aij2[1] = vld1q_f64(&(a->element[1][index]));
        bij2[0] = vld1q_f64(&(b->element[0][index]));
        bij2[1] = vld1q_f64(&(b->element[1][index]));

        _bncneon_rdd_add(tmp2, aij2, bij2);

        vst1q_f64(&(c->element[0][index]), tmp2[0]);
        vst1q_f64(&(c->element[1][index]), tmp2[1]); 
    }

#else // others
	double tmp[DDSIZE], aij[DDSIZE], bij[DDSIZE];

	for(index = 0; index < real_total_dim; index++)
	{
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];
		bij[0] = b->element[0][index];
		bij[1] = b->element[1][index];

		rdd_add(tmp, aij, bij);

		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1]; 
	}
#endif // __AVX2__
/*
	double tmp[DDSIZE];

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rdd_add(tmp, get_ddmatrix_ij(a, i, j), get_ddmatrix_ij(b, i, j));
			set_ddmatrix_ij(c, i, j, tmp);
		}
	}
*/
}

/* c := a - b */
void sub_ddmatrix(DDMatrix c, DDMatrix a, DDMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_ddmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: sub_ddmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

/*
	double tmp[DDSIZE];
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rdd_sub(tmp, get_ddmatrix_ij(a, i, j), get_ddmatrix_ij(b, i, j));
			set_ddmatrix_ij(c, i, j, tmp);
		}
	}
*/

// for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[DDSIZE], aij4[DDSIZE], bij4[DDSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij4[0] = _mm256_load_pd(&(a->element[0][index]));
		aij4[1] = _mm256_load_pd(&(a->element[1][index]));
		bij4[0] = _mm256_load_pd(&(b->element[0][index]));
		bij4[1] = _mm256_load_pd(&(b->element[1][index]));

		_bncavx2_rdd_sub(tmp4, aij4, bij4);

		_mm256_store_pd(&(c->element[0][index]), tmp4[0]);
		_mm256_store_pd(&(c->element[1][index]), tmp4[1]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[DDSIZE], aij8[DDSIZE], bij8[DDSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij8[0] = _mm512_load_pd(&(a->element[0][index]));
		aij8[1] = _mm512_load_pd(&(a->element[1][index]));
		bij8[0] = _mm512_load_pd(&(b->element[0][index]));
		bij8[1] = _mm512_load_pd(&(b->element[1][index]));

		_bncavx512_rdd_sub(tmp8, aij8, bij8);

		_mm512_store_pd(&(c->element[0][index]), tmp8[0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[1]); 
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t tmp2_0, tmp2_1;
	svfloat64_t aij2_0, aij2_1;
	svfloat64_t bij2_0, bij2_1;

    for(index = 0; index < real_total_dim; index += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(real_total_dim));
        aij2_0 = svld1_f64(pg, &(a->element[0][index]));
        aij2_1 = svld1_f64(pg, &(a->element[1][index]));
        bij2_0 = svld1_f64(pg, &(b->element[0][index]));
        bij2_1 = svld1_f64(pg, &(b->element[1][index]));

        _bncsve2_rdd_neg(pg, &tmp2_0, &tmp2_1, bij2_0, bij2_1);
		_bncsve2_rdd_add(pg, &tmp2_0, &tmp2_1, aij2_0, aij2_1, tmp2_0, tmp2_1);

        svst1_f64(pg, &(c->element[0][index]), tmp2_0);
        svst1_f64(pg, &(c->element[1][index]), tmp2_1); 
    }

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t tmp2[DDSIZE], aij2[DDSIZE], bij2[DDSIZE];

    for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
    {
        aij2[0] = vld1q_f64(&(a->element[0][index]));
        aij2[1] = vld1q_f64(&(a->element[1][index]));
        bij2[0] = vld1q_f64(&(b->element[0][index]));
        bij2[1] = vld1q_f64(&(b->element[1][index]));

        _bncneon_rdd_sub(tmp2, aij2, bij2);

        vst1q_f64(&(c->element[0][index]), tmp2[0]);
        vst1q_f64(&(c->element[1][index]), tmp2[1]); 
    }

#else // others
	double tmp[DDSIZE], aij[DDSIZE], bij[DDSIZE];

	for(index = 0; index < real_total_dim; index++)
	{
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];
		bij[0] = b->element[0][index];
		bij[1] = b->element[1][index];

		rdd_sub(tmp, aij, bij);

		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1]; 
	}
#endif // __AVX2__
}

/* c := sc * a */
void cmul_ddmatrix(DDMatrix c, double sc[DDSIZE], DDMatrix a)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_ddmatrix(row_dim is differnt! : c->(%ld, %ld), a->(%ld, %ld)\n", c->row_dim, c->col_dim, a->row_dim, a->col_dim);
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_ddmatrix(col_dim is differnt! : c->(%ld, %ld), a->(%ld, %ld)\n", c->row_dim, c->col_dim, a->row_dim, a->col_dim);
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

// for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[DDSIZE], sc4[DDSIZE], aij4[DDSIZE];

	sc4[0] = _mm256_set1_pd(sc[0]);
	sc4[1] = _mm256_set1_pd(sc[1]);

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij4[0] = _mm256_load_pd(&(a->element[0][index]));
		aij4[1] = _mm256_load_pd(&(a->element[1][index]));

		_bncavx2_rdd_mul(tmp4, sc4, aij4);

		_mm256_store_pd(&(c->element[0][index]), tmp4[0]);
		_mm256_store_pd(&(c->element[1][index]), tmp4[1]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[DDSIZE], aij8[DDSIZE], sc8[DDSIZE];

	sc8[0] = _mm512_set1_pd(sc[0]);
	sc8[1] = _mm512_set1_pd(sc[1]);

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		aij8[0] = _mm512_load_pd(&(a->element[0][index]));
		aij8[1] = _mm512_load_pd(&(a->element[1][index]));

		_bncavx512_rdd_add(tmp8, sc8, aij8);

		_mm512_store_pd(&(c->element[0][index]), tmp8[0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[1]); 
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
    
	svfloat64_t tmp2_0, tmp2_1;
	svfloat64_t sc2_0, sc2_1;
	svfloat64_t aij2_0, aij2_1;

    /* dd スカラ sc を 2-lane NEON ベクトルにブロードキャスト */
    sc2_0 = svdup_f64(sc[0]); sc2_1 = svdup_f64(sc[1]);

    real_total_dim = real_row_dim * real_col_dim;

    for(index = 0; index < real_total_dim; index += (long int)svcntd())
    {
    		svbool_t pg = svwhilelt_b64_s32((int)index, (int)real_total_dim);
        aij2_0 = svld1_f64(pg, &(a->element[0][index]));
        aij2_1 = svld1_f64(pg, &(a->element[1][index]));

        _bncsve2_rdd_mul(svptrue_b64(), &tmp2_0, &tmp2_1, sc2_0, sc2_1, aij2_0, aij2_1);

        svst1_f64(pg, &(c->element[0][index]), tmp2_0);
        svst1_f64(pg, &(c->element[1][index]), tmp2_1); 
    }

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t tmp2[DDSIZE], sc2[DDSIZE], aij2[DDSIZE];

    /* dd スカラ sc を 2-lane NEON ベクトルにブロードキャスト */
    _bncneon_rdd_set1_dd(sc2, sc);

    real_total_dim = real_row_dim * real_col_dim;

    for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
    {
        aij2[0] = vld1q_f64(&(a->element[0][index]));
        aij2[1] = vld1q_f64(&(a->element[1][index]));

        _bncneon_rdd_mul(tmp2, sc2, aij2);

        vst1q_f64(&(c->element[0][index]), tmp2[0]);
        vst1q_f64(&(c->element[1][index]), tmp2[1]); 
    }

#else // others
	double tmp[DDSIZE], aij[DDSIZE];

	for(index = 0; index < real_total_dim; index++)
	{
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];

		rdd_mul(tmp, sc, aij);

		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1]; 
	}
#endif // __AVX2__
/*
	double tmp[DDSIZE];

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rdd_mul(tmp, sc, get_ddmatrix_ij(a, i, j));
			set_ddmatrix_ij(c, i, j, tmp);
		}
	}
*/
}

/* c = a^T */
void transpose_ddmatrix(DDMatrix c, DDMatrix a)
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
	__m256d aji4[DDSIZE];

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
			index = i * real_col_dim + j;
			_mm256_store_pd(&(c->element[0][index]), aji4[0]);
			_mm256_store_pd(&(c->element[1][index]), aji4[1]);
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d aji8[DDSIZE];

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
			index = i * real_col_dim + j;
			_mm512_store_pd(&(c->element[0][index]), aji8[0]);
			_mm512_store_pd(&(c->element[1][index]), aji8[1]);
		}
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, gather)
    svfloat64_t aji2_0, aji2_1;
    svint64_t vidx;
    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j += (long int)svcntd())
        {
            svbool_t pg = svwhilelt_b64_s64((int64_t)j, (int64_t)real_col_dim);
            vidx = svindex_s64((int64_t)(j * real_col_dim + i), (int64_t)real_col_dim);
            aji2_0 = svld1_gather_s64index_f64(pg, a->element[0], vidx);
            aji2_1 = svld1_gather_s64index_f64(pg, a->element[1], vidx);
            index = i * real_col_dim + j;
            svst1_f64(pg, &(c->element[0][index]), aji2_0);
            svst1_f64(pg, &(c->element[1][index]), aji2_1);
        }
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t aji2[DDSIZE];
    float64x2_t v0, v1;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
        {
            /* a(j,i), a(j+1,i) を 2-lane ベクトルに詰める */
            v0 = vdupq_n_f64(a->element[0][(j    ) * real_col_dim + i]);
            v1 = vdupq_n_f64(a->element[1][(j    ) * real_col_dim + i]);
            v0 = vsetq_lane_f64(a->element[0][(j + 1) * real_col_dim + i], v0, 1);
            v1 = vsetq_lane_f64(a->element[1][(j + 1) * real_col_dim + i], v1, 1);

            aji2[0] = v0;
            aji2[1] = v1;

            index = i * real_col_dim + j;
            vst1q_f64(&(c->element[0][index]), aji2[0]);
            vst1q_f64(&(c->element[1][index]), aji2[1]);
        }
    }
#else // others
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_ddmatrix_ij(c, i, j, get_ddmatrix_ij(a, j, i));
	}
#endif // AVX2
}

/* c := a */
void subst_ddmatrix(DDMatrix c, DDMatrix a)
{
	long int i, j, index;
	long int real_row_dim, real_col_dim;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_ddmatrix\n");
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
		}
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t v0;
	svfloat64_t v1;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j += (long int)svcntd())
        {
		svbool_t pg = svwhilelt_b64_s64((int64_t)j, (int64_t)(real_col_dim));
            index = i * real_col_dim + j;

            v0 = svld1_f64(pg, &(a->element[0][index]));
            v1 = svld1_f64(pg, &(a->element[1][index]));

            svst1_f64(pg, &(c->element[0][index]), v0);
            svst1_f64(pg, &(c->element[1][index]), v1);
        }
    }

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t v0, v1;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
        {
            index = i * real_col_dim + j;

            v0 = vld1q_f64(&(a->element[0][index]));
            v1 = vld1q_f64(&(a->element[1][index]));

            vst1q_f64(&(c->element[0][index]), v0);
            vst1q_f64(&(c->element[1][index]), v1);
        }
    }

#else // others
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_ddmatrix_ij(c, i, j, get_ddmatrix_ij(a, i, j));
		}
	}
#endif // AVX2
}

/* c := -a */
void neg_ddmatrix(DDMatrix c, DDMatrix a)
{
	long int i, j, index;
	long int real_row_dim, real_col_dim;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: neg_ddmatrix\n");
		return;
	}
	real_row_dim = c->real_row_dim;
	real_col_dim = c->real_col_dim;

// AVX2
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp[DDSIZE];

	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j));
			index = i * real_col_dim + j;
			tmp[0] = _bncavx2_dneg(_mm256_load_pd(&(a->element[0][index])));
			tmp[1] = _bncavx2_dneg(_mm256_load_pd(&(a->element[1][index])));
			_mm256_store_pd(&(c->element[0][i * real_col_dim + j]), tmp[0]); //_mm256_load_pd(&(a->element[0][index])));
			_mm256_store_pd(&(c->element[1][i * real_col_dim + j]), tmp[1]); //_mm256_load_pd(&(a->element[1][index])));
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp[DDSIZE];

	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j));
			index = i * real_col_dim + j;
			tmp[0] = _bncavx512_dneg(_mm512_load_pd(&(a->element[0][index])));
			tmp[1] = _bncavx512_dneg(_mm512_load_pd(&(a->element[1][index])));
			_mm512_store_pd(&(c->element[0][i * real_col_dim + j]), tmp[0]); //_mm512_load_pd(&(a->element[0][index])));
			_mm512_store_pd(&(c->element[1][i * real_col_dim + j]), tmp[1]); //_mm512_load_pd(&(a->element[1][index])));
		}
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
    
	svfloat64_t val2_0, val2_1;
	svfloat64_t tmp2_0, tmp2_1;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j += (long int)svcntd())
        {
        		svbool_t pg = svwhilelt_b64_s32((int)j, (int)real_col_dim);
            index = i * real_col_dim + j;

            val2_0 = svld1_f64(pg, &(a->element[0][index]));
            val2_1 = svld1_f64(pg, &(a->element[1][index]));

            _bncsve2_rdd_neg(svptrue_b64(), &tmp2_0, &tmp2_1, val2_0, val2_1);

            svst1_f64(pg, &(c->element[0][index]), tmp2_0);
            svst1_f64(pg, &(c->element[1][index]), tmp2_1);
        }
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t val2[DDSIZE], tmp2[DDSIZE];

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
        {
            index = i * real_col_dim + j;

            val2[0] = vld1q_f64(&(a->element[0][index]));
            val2[1] = vld1q_f64(&(a->element[1][index]));

            _bncneon_rdd_neg(tmp2, val2);

            vst1q_f64(&(c->element[0][index]), tmp2[0]);
            vst1q_f64(&(c->element[1][index]), tmp2[1]);
        }
    }
#else // others
	double tmp[DDSIZE];

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			rdd_neg(tmp, get_ddmatrix_ij(a, i, j));
			set_ddmatrix_ij(c, i, j, tmp); //get_ddmatrix_ij(a, i, j));
		}
	}
#endif // AVX2
}

/* c := I */
void setI_ddmatrix(DDMatrix c)
{
	long int i, j;
	long int real_total_dim;
	double tmp1[DDSIZE];

	real_total_dim = c->real_row_dim * c->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&c->element[0][i], zero4);
		_mm256_store_pd(&c->element[1][i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&c->element[0][i], zero8);
		_mm512_store_pd(&c->element[1][i], zero8);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    
	svfloat64_t zero2;

    zero2 = svdup_f64(0.0);
    for(i = 0; i < real_total_dim; i += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(real_total_dim));
        svst1_f64(pg, &c->element[0][i], zero2);
        svst1_f64(pg, &c->element[1][i], zero2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t zero2;

    zero2 = vdupq_n_f64(0.0);
    for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
    {
        vst1q_f64(&c->element[0][i], zero2);
        vst1q_f64(&c->element[1][i], zero2);
    }
#else // others
	for(i = 0; i < real_total_dim; i++)
	{
		c->element[0][i] = 0.0;
		c->element[1][i] = 0.0;	
	}
#endif // __AVX2__

	rdd_set_ui(tmp1, 1UL);

	for(i = 0; i < c->row_dim; i++)
	{
		if(i < c->col_dim)
			set_ddmatrix_ij(c, i, i, tmp1);
	}
}

/* v := a * vb */
void mul_ddmatrix_ddvec(DDVector v, DDMatrix a, DDVector vb)
{
	long int i, j;
	double tmp[DDSIZE], tmp1[DDSIZE];

	/* Check Dimension */
	//if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	if((v->dim < a->row_dim) || (vb->dim < a->col_dim))
	{
		fprintf(stderr, "ERROR: mul_ddmatrix_ddvec\n");
		return;
	}

// SIMD
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int ij_index, real_col_dim;
	__m256d tmp4[DDSIZE], tmp1_4[DDSIZE];
	__m256d aij4[DDSIZE], vbj4[DDSIZE];

	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->row_dim; i++)
	{
		//rdd_set_ui(tmp, 0UL);
		_bncavx2_set0_dd(tmp4);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij4[0] = _mm256_load_pd(&(a->element[0][ij_index]));
			aij4[1] = _mm256_load_pd(&(a->element[1][ij_index]));
			vbj4[0] = _mm256_load_pd(&(vb->element[0][j]));
			vbj4[1] = _mm256_load_pd(&(vb->element[1][j]));

			//rdd_mul(tmp1, get_ddmatrix_ij(a, i, j), get_ddvector_i(vb, j));
			//rdd_add(tmp, tmp, tmp1);
			_bncavx2_rdd_mul(tmp1_4, aij4, vbj4);
			_bncavx2_rdd_add(tmp4, tmp4, tmp1_4);
		}
		//set_ddvector_i(v, i, tmp);
		_bncavx2_rdd_sum256d(tmp, tmp4);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int ij_index, real_col_dim;
	__m512d tmp8[DDSIZE], tmp1_8[DDSIZE];
	__m512d aij8[DDSIZE], vbj8[DDSIZE];

	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->row_dim; i++)
	{
		//rdd_set_ui(tmp, 0UL);
		_bncavx512_set0_dd(tmp8);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij8[0] = _mm512_load_pd(&(a->element[0][ij_index]));
			aij8[1] = _mm512_load_pd(&(a->element[1][ij_index]));
			vbj8[0] = _mm512_load_pd(&(vb->element[0][j]));
			vbj8[1] = _mm512_load_pd(&(vb->element[1][j]));

			//rdd_mul(tmp1, get_ddmatrix_ij(a, i, j), get_ddvector_i(vb, j));
			//rdd_add(tmp, tmp, tmp1);
			_bncavx512_rdd_mul(tmp1_8, aij8, vbj8);
			_bncavx512_rdd_add(tmp8, tmp8, tmp1_8);
		}
		//set_ddvector_i(v, i, tmp);
		_bncavx512_rdd_sum512d(tmp, tmp8);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
    long int ij_index, real_col_dim;
    
	svfloat64_t tmp2_0, tmp2_1;
	svfloat64_t tmp1_2_0, tmp1_2_1;
    
	svfloat64_t aij2_0, aij2_1;
	svfloat64_t vbj2_0, vbj2_1;

    real_col_dim = a->real_col_dim;

    for(i = 0; i < a->row_dim; i++)
    {
        _bncsve2_rdd_set0(&tmp2_0, &tmp2_1);

        for(j = 0; j < real_col_dim; j += (long int)svcntd())
        {
        		svbool_t pg = svwhilelt_b64_s32((int)j, (int)real_col_dim);
            ij_index = i * real_col_dim + j;

            aij2_0 = svld1_f64(pg, &(a->element[0][ij_index]));
            aij2_1 = svld1_f64(pg, &(a->element[1][ij_index]));
            vbj2_0 = svld1_f64(pg, &(vb->element[0][j]));
            vbj2_1 = svld1_f64(pg, &(vb->element[1][j]));

            _bncsve2_rdd_mul(svptrue_b64(), &tmp1_2_0, &tmp1_2_1, aij2_0, aij2_1, vbj2_0, vbj2_1);
            _bncsve2_rdd_add(svptrue_b64(), &tmp2_0, &tmp2_1, tmp2_0, tmp2_1, tmp1_2_0, tmp1_2_1);
        }

        _bncsve2_rdd_sum128d(svptrue_b64(), tmp, tmp2_0, tmp2_1);
        v->element[0][i] = tmp[0];
        v->element[1][i] = tmp[1];
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    long int ij_index, real_col_dim;
    float64x2_t tmp2[DDSIZE], tmp1_2[DDSIZE];
    float64x2_t aij2[DDSIZE], vbj2[DDSIZE];

    real_col_dim = a->real_col_dim;

    for(i = 0; i < a->row_dim; i++)
    {
        _bncneon_set0_dd(tmp2);

        for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
        {
            ij_index = i * real_col_dim + j;

            aij2[0] = vld1q_f64(&(a->element[0][ij_index]));
            aij2[1] = vld1q_f64(&(a->element[1][ij_index]));
            vbj2[0] = vld1q_f64(&(vb->element[0][j]));
            vbj2[1] = vld1q_f64(&(vb->element[1][j]));

            _bncneon_rdd_mul(tmp1_2, aij2, vbj2);
            _bncneon_rdd_add(tmp2, tmp2, tmp1_2);
        }

        _bncneon_rdd_sum128d(tmp, tmp2);
        v->element[0][i] = tmp[0];
        v->element[1][i] = tmp[1];
    }
#else // others

	for(i = 0; i < a->row_dim; i++)
	{
		rdd_set_ui(tmp, 0UL);
		for(j = 0; j < a->col_dim; j++)
		{
			// rdd_mul(tmp1, get_ddmatrix_ij(a, i, j), get_ddvector_i(vb, j));
			// rdd_add(tmp, tmp, tmp1);
			rdd_fma(tmp, get_ddmatrix_ij(a, i, j), get_ddvector_i(vb, j), tmp);
		}
		set_ddvector_i(v, i, tmp);
	}
#endif // __AVX2__

}

/* v := a^T * vb */
void mul_ddmatrixt_ddvec(DDVector v, DDMatrix a, DDVector vb)
{
	long int i, j;
	double tmp[DDSIZE], tmp1[DDSIZE];

	/* Check Dimension */
	//if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	if((v->dim < a->col_dim) || (vb->dim < a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_ddmatrixt_ddvec\n");
		return;
	}

// SIMD
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int ji_index, real_row_dim, real_col_dim;
	__m256d tmp4[DDSIZE], tmp1_4[DDSIZE];
	__m256d aij4[DDSIZE], vbj4[DDSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->col_dim; i++)
	{
		//rdd_set_ui(tmp, 0UL);
		_bncavx2_set0_dd(tmp4);
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
			vbj4[0] = _mm256_load_pd(&(vb->element[0][j]));
			vbj4[1] = _mm256_load_pd(&(vb->element[1][j]));

			//rdd_mul(tmp1, get_ddmatrix_ij(a, i, j), get_ddvector_i(vb, j));
			//rdd_add(tmp, tmp, tmp1);
			_bncavx2_rdd_mul(tmp1_4, aij4, vbj4);
			_bncavx2_rdd_add(tmp4, tmp4, tmp1_4);
		}
		//set_ddvector_i(v, i, tmp);
		_bncavx2_rdd_sum256d(tmp, tmp4);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int real_row_dim, real_col_dim;
	__m512d tmp8[DDSIZE], tmp1_8[DDSIZE];
	__m512d aij8[DDSIZE], vbj8[DDSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->col_dim; i++)
	{
		//rdd_set_ui(tmp, 0UL);
		_bncavx512_set0_dd(tmp8);
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
			vbj8[0] = _mm512_load_pd(&(vb->element[0][j]));
			vbj8[1] = _mm512_load_pd(&(vb->element[1][j]));

			//rdd_mul(tmp1, get_ddmatrix_ij(a, i, j), get_ddvector_i(vb, j));
			//rdd_add(tmp, tmp, tmp1);
			_bncavx512_rdd_mul(tmp1_8, aij8, vbj8);
			_bncavx512_rdd_add(tmp8, tmp8, tmp1_8);
		}
		//set_ddvector_i(v, i, tmp);
		_bncavx512_rdd_sum512d(tmp, tmp8);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, gather)
    long int real_row_dim, real_col_dim;
    svfloat64_t tmp2_0, tmp2_1;
    svfloat64_t tmp1_2_0, tmp1_2_1;
    svfloat64_t aij2_0, aij2_1;
    svfloat64_t vbj2_0, vbj2_1;
    svint64_t vidx;

    real_row_dim = a->real_row_dim;
    real_col_dim = a->real_col_dim;

    for(i = 0; i < a->col_dim; i++)
    {
        _bncsve2_rdd_set0(&tmp2_0, &tmp2_1);
        for(j = 0; j < real_row_dim; j += (long int)svcntd())
        {
            svbool_t pg = svwhilelt_b64_s64((int64_t)j, (int64_t)real_row_dim);
            vidx = svindex_s64((int64_t)(j * real_col_dim + i), (int64_t)real_col_dim);
            aij2_0 = svld1_gather_s64index_f64(pg, a->element[0], vidx);
            aij2_1 = svld1_gather_s64index_f64(pg, a->element[1], vidx);
            vbj2_0 = svld1_f64(pg, &(vb->element[0][j]));
            vbj2_1 = svld1_f64(pg, &(vb->element[1][j]));
            _bncsve2_rdd_mul(svptrue_b64(), &tmp1_2_0, &tmp1_2_1, aij2_0, aij2_1, vbj2_0, vbj2_1);
            _bncsve2_rdd_add(svptrue_b64(), &tmp2_0, &tmp2_1, tmp2_0, tmp2_1, tmp1_2_0, tmp1_2_1);
        }
        _bncsve2_rdd_sum128d(svptrue_b64(), tmp, tmp2_0, tmp2_1);
        v->element[0][i] = tmp[0];
        v->element[1][i] = tmp[1];
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    long int real_row_dim, real_col_dim;
    float64x2_t tmp2[DDSIZE], tmp1_2[DDSIZE];
    float64x2_t aij2[DDSIZE], vbj2[DDSIZE];
    float64x2_t v0h, v0l;

    real_row_dim = a->real_row_dim;
    real_col_dim = a->real_col_dim;

    for(i = 0; i < a->col_dim; i++)
    {
        _bncneon_set0_dd(tmp2);

        for(j = 0; j < real_row_dim; j += _BNC_D_WIDTH)
        {
            /* a(j,i), a(j+1,i) を 2-lane dd ベクトルに詰める */
            v0h = vdupq_n_f64(a->element[0][(j    ) * real_col_dim + i]);
            v0l = vdupq_n_f64(a->element[1][(j    ) * real_col_dim + i]);
            v0h = vsetq_lane_f64(a->element[0][(j + 1) * real_col_dim + i], v0h, 1);
            v0l = vsetq_lane_f64(a->element[1][(j + 1) * real_col_dim + i], v0l, 1);

            aij2[0] = v0h;
            aij2[1] = v0l;

            vbj2[0] = vld1q_f64(&(vb->element[0][j]));
            vbj2[1] = vld1q_f64(&(vb->element[1][j]));

            _bncneon_rdd_mul(tmp1_2, aij2, vbj2);
            _bncneon_rdd_add(tmp2, tmp2, tmp1_2);
        }

        _bncneon_rdd_sum128d(tmp, tmp2);
        v->element[0][i] = tmp[0];
        v->element[1][i] = tmp[1];
    }
#else // others

	for(i = 0; i < a->col_dim; i++)
	{
		set0_dd(tmp);
		for(j = 0; j < a->row_dim; j++)
		{
			//rdd_mul(tmp1, get_ddmatrix_ij(a, j, i), get_ddvector_i(vb, j));
			//rdd_add(tmp, tmp, tmp1);
			rdd_fma(tmp, get_ddmatrix_ij(a, j, i), get_ddvector_i(vb, j), tmp);
		}
		set_ddvector_i(v, i, tmp);
	}
#endif // __AVX2__
}

/* a = a^(-1) */
/* square matrix only */
void inv_ddmatrix(DDMatrix a)
{
	long int i, j, k, dim;
	double tmp[DDSIZE], aii[DDSIZE];

	/* Check Dimensions */
	if(a->row_dim != a->col_dim)
	{
		fprintf(stderr, "ERROR: inv_ddmatrix\n");
		return;
	}

	dim = a->row_dim;

	for(i = 0; i < dim; i++)
	{
		if(rdd_cmp_ui(get_ddmatrix_ij(a, i, i), 0UL) == 0) 
		{
			fprintf(stderr, "ERROR: inv_ddmatrix: Pivot(%ld,%ld) is zero.\n", i, i);
			return;
		}

		rdd_ui_div(aii, 1UL, get_ddmatrix_ij(a, i, i));
		set_ddmatrix_ij(a, i, i, aii);

		for(j = 0; j < i; j++)
		{
			rdd_mul(tmp, get_ddmatrix_ij(a, i, j), aii);
			set_ddmatrix_ij(a, i, j, tmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			rdd_mul(tmp, get_ddmatrix_ij(a, i, j), aii);
			set_ddmatrix_ij(a, i, j, tmp);
		}

		for(j = 0; j < i; j++)
		{
			for(k = 0; k < i; k++)
			{
				rdd_mul(tmp, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(tmp, get_ddmatrix_ij(a, j, k), tmp);
				set_ddmatrix_ij(a, j, k, tmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				rdd_mul(tmp, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(tmp, get_ddmatrix_ij(a, j, k), tmp);
				set_ddmatrix_ij(a, j, k, tmp);
			}
		}

		for(j = i + 1; j < dim; j++)
		{
			for(k = 0; k < i; k++)
			{
				rdd_mul(tmp, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(tmp, get_ddmatrix_ij(a, j, k), tmp);
				set_ddmatrix_ij(a, j, k, tmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				rdd_mul(tmp, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(tmp, get_ddmatrix_ij(a, j, k), tmp);
				set_ddmatrix_ij(a, j, k, tmp);
			}
		}

		for(j = 0; j < i; j++)
		{
			rdd_neg(tmp, aii); /* tmp := -aii */
			rdd_mul(tmp, tmp, get_ddmatrix_ij(a, j, i));
			set_ddmatrix_ij(a, j, i, tmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			rdd_neg(tmp, aii); /* tmp := -aii */
			rdd_mul(tmp, tmp, get_ddmatrix_ij(a, j, i));
			set_ddmatrix_ij(a, j, i, tmp);
		}
	}
}

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpf)a */
void subst_mpfvector_ddvec(MPFVector c, DDVector a)
{
	long int i;
	mpf_t tmp;

	mpf_init2(tmp, c->prec);

	for(i = 0; i < a->dim; i++)
	{
		mpf_set_dd(tmp, get_ddvector_i(a, i));
		set_mpfvector_i(c, i, tmp);
	}

	mpf_clear(tmp);

}

/* c := (dd)a */
void subst_ddvector_mpfvec(DDVector c, MPFVector a)
{
	long int i;
	double tmp[DDSIZE];

	for(i = 0; i < a->dim; i++)
	{
		mpf_get_dd(tmp, get_mpfvector_i(a, i));
		set_ddvector_i(c, i, tmp);
	}

}

/* init_set_ddvector_mpfvec */
DDVector init_set_ddvector_mpfvec(MPFVector c)
{
	long int i;
	mpf_t tmp;
	DDVector ret = NULL;

	// initialize
	ret = init_ddvector(c->dim);
	if(ret == NULL) return ret;

	subst_ddvector_mpfvec(ret, c);

	return ret;
}

/* c := (mpf)a */
void subst_mpfmatrix_ddmat(MPFMatrix c, DDMatrix a)
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
			mpf_set_dd(tmp, get_ddmatrix_ij(a, i, j));
			set_mpfmatrix_ij(c, i, j, tmp);
		}
	}

	mpf_clear(tmp);
}

/* c := (dd)a */
void subst_ddmatrix_mpfmat(DDMatrix c, MPFMatrix a)
{
	long int i, j;
	double tmp[DDSIZE];

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_ddmatrix_mpfmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			mpf_get_dd(tmp, get_mpfmatrix_ij(a, i, j));
			set_ddmatrix_ij(c, i, j, tmp);
		}
	}
}

/* init_set_ddmatrix_mpfmat */
DDMatrix init_set_ddmatrix_mpfmat(MPFMatrix a)
{
	long int i;
	mpf_t tmp;
	DDMatrix ret = NULL;

	// initialize
	ret = init_ddmatrix(a->row_dim, a->col_dim);
	if(ret == NULL) return ret;

	subst_ddmatrix_mpfmat(ret, a);

	return ret;
}

/* Normwise relative error of vector */
void relerr_ddvector_mpfvec(double relerr[DDSIZE], DDVector approx_vec, MPFVector true_vec, int norm_type)
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
	subst_mpfvector_ddvec(mpf_approx_vec, approx_vec);

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
	mpf_get_dd(relerr, mpf_relerr);

	free_mpfvector(diff_vec);
	free_mpfvector(mpf_approx_vec);
	mpf_clear(norm_diff_vec);
	mpf_clear(norm_true_vec);
	mpf_clear(mpf_relerr);

	return;
}

/* Elementwise relative errors of vector */
void relerr_element_ddvector_mpf(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double norm_relerr[DDSIZE], DDVector approx_vec, MPFVector true_vec, int norm_type)
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
	subst_mpfvector_ddvec(mpf_approx_vec, approx_vec);

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
	mpf_get_dd(max_relerr, mpf_max_relerr);
	mpf_get_dd(min_relerr, mpf_min_relerr);
	mpf_get_dd(norm_relerr, mpf_norm_relerr);

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
void subst_ddvector_dvec(DDVector c, DVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
	{
		set_ddvector_i_d(c, i, get_dvector_i(a, i));
	}
}

/* c := (d)a */
void subst_dvector_ddvec(DVector c, DDVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
	{
		c->element[i] = rdd_get_d(get_ddvector_i(a, i));
	}
}


/* c := (dd)a */
void subst_ddmatrix_dmat(DDMatrix c, DMatrix a)
{
	long int i, j;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_ddmatrix_dmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_ddmatrix_ij_d(c, i, j, get_dmatrix_ij(a, i, j));
		}
	}
}

/* c := (d)a */
void subst_dmatrix_ddmat(DMatrix c, DDMatrix a)
{
	long int i, j, ij_index;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_dmatrix_ddmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			//ij_index = i * (c->col_dim) + j; // 2022-11-24(Tue) Fixed! by T.Kouya
			ij_index = i * (c->real_col_dim) + j;
			c->element[ij_index] = rdd_get_d(get_ddmatrix_ij(a, i, j));
		}
	}
}

/* Normwise relative error of vector */
void relerr_ddvector(double relerr[DDSIZE], DDVector approx_vec, DDVector true_vec, int norm_type)
{
	double norm_true_vec[DDSIZE], norm_diff_vec[DDSIZE];
	DDVector diff_vec;

	diff_vec = init_ddvector(approx_vec->dim);

	// diff_vec := approx_vec - true_vec
	sub_ddvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_ddvector(norm_diff_vec, diff_vec);
			normi_ddvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_ddvector(norm_diff_vec, diff_vec);
			norm1_ddvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_ddvector(norm_diff_vec, diff_vec);
			norm2_ddvector(norm_true_vec, true_vec);
			break;
	}

	if(rdd_cmp_ui(norm_true_vec, 0UL) != 0)
		rdd_div(relerr, norm_diff_vec, norm_true_vec);

	free_ddvector(diff_vec);

	return;
}

/* Elementwise relative errors of vector */
void relerr_element_ddvector(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double norm_relerr[DDSIZE], DDVector approx_vec, DDVector true_vec, int norm_type)
{
	double abs_true_vec[DDSIZE], abs_diff_vec[DDSIZE], norm_diff_vec[DDSIZE], norm_true_vec[DDSIZE];
	long int i;
	DDVector diff_vec;

	diff_vec = init_ddvector(approx_vec->dim);

	// diff_vec := approx_vec - true_vec
	sub_ddvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_ddvector(norm_diff_vec, diff_vec);
			normi_ddvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_ddvector(norm_diff_vec, diff_vec);
			norm1_ddvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_ddvector(norm_diff_vec, diff_vec);
			norm2_ddvector(norm_true_vec, true_vec);
			break;
	}

	rdd_set(norm_relerr, norm_diff_vec);
	if(rdd_cmp_ui(norm_true_vec, 0UL) != 0)
		rdd_div(norm_relerr, norm_diff_vec, norm_true_vec);

	// relative errors of each elements
	rdd_set_ui(max_relerr, 0UL);
	normi_ddvector(min_relerr, diff_vec);
	for(i = 0; i < approx_vec->dim; i++)
	{
		rdd_abs(abs_diff_vec, get_ddvector_i(diff_vec, i));
		rdd_abs(abs_true_vec, get_ddvector_i(true_vec, i));
		if(rdd_cmp_ui(abs_true_vec, 0UL) != 0)
			rdd_div(abs_diff_vec, abs_diff_vec, abs_true_vec);
		
		if(rdd_cmp(max_relerr, abs_diff_vec) < 0)
			rdd_set(max_relerr, abs_diff_vec);
		if(rdd_cmp(min_relerr, abs_diff_vec) > 0)
			rdd_set(min_relerr, abs_diff_vec);
	}

	free_ddvector(diff_vec);// Fix! 2012-06-03 by T.Kouya

	return;
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_ddmatrix(DDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
// SIMD
//#if 0
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int i, j, true_end, true_end_start, true_end_end, real_col_dim, index0, index1;
	double tmp[DDSIZE];
	__m256d tmp256[DDSIZE];

	true_end = (col_end > mat->col_dim) ? mat->col_dim : col_end;
	true_end_start = (long int)ceil((double)col_start / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
	true_end_end = (long int)floor((double)true_end / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;

	real_col_dim = mat->real_col_dim;

	for(i = col_start; i < true_end_start; i++)
	{
		rdd_set(tmp, get_ddmatrix_ij(mat, row_index0, i));
		set_ddmatrix_ij(mat, row_index0, i, get_ddmatrix_ij(mat, row_index1, i));
		set_ddmatrix_ij(mat, row_index1, i, tmp);
	}
	//for(i = col_start; i < true_end; i += _BNC_D_WIDTH)
	for(i = true_end_start; i < true_end_end; i += _BNC_D_WIDTH)
	{
		//rdd_set(tmp, get_ddmatrix_ij(mat, row_index0, i));
		index0 = row_index0 * real_col_dim + i;
		tmp256[0] = _mm256_load_pd(&(mat->element[0][index0]));
		tmp256[1] = _mm256_load_pd(&(mat->element[1][index0]));

		//set_ddmatrix_ij(mat, row_index0, i, get_ddmatrix_ij(mat, row_index1, i));
		index1 = row_index1 * real_col_dim + i;
		_mm256_store_pd(&(mat->element[0][index0]), _mm256_load_pd(&(mat->element[0][index1])));
		_mm256_store_pd(&(mat->element[1][index0]), _mm256_load_pd(&(mat->element[1][index1])));

		//set_ddmatrix_ij(mat, row_index1, i, tmp);
		_mm256_store_pd(&(mat->element[0][index1]), tmp256[0]);
		_mm256_store_pd(&(mat->element[1][index1]), tmp256[1]);
	}
	for(i = true_end_end; i < true_end; i++)
	{
		rdd_set(tmp, get_ddmatrix_ij(mat, row_index0, i));
		set_ddmatrix_ij(mat, row_index0, i, get_ddmatrix_ij(mat, row_index1, i));
		set_ddmatrix_ij(mat, row_index1, i, tmp);
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int i, j, true_end, true_end_quo, true_end_rem, real_col_dim, index0, index1;
	double tmp[DDSIZE];
	__m512d tmp512[DDSIZE];

	true_end = (col_end > mat->col_dim) ? mat->col_dim : col_end;
	true_end_quo = (true_end / _BNC_D_WIDTH) * _BNC_D_WIDTH;
	true_end_rem = true_end - true_end_quo * _BNC_D_WIDTH;

	real_col_dim = mat->real_col_dim;

	//for(i = col_start; i < true_end; i += _BNC_D_WIDTH)
	for(i = col_start; i < true_end_quo; i += _BNC_D_WIDTH)
	{
		//rdd_set(tmp, get_ddmatrix_ij(mat, row_index0, i));
		index0 = row_index0 * real_col_dim + i;
		tmp512[0] = _mm512_load_pd(&(mat->element[0][index0]));
		tmp512[1] = _mm512_load_pd(&(mat->element[1][index0]));

		//set_ddmatrix_ij(mat, row_index0, i, get_ddmatrix_ij(mat, row_index1, i));
		index1 = row_index1 * real_col_dim + i;
		_mm512_store_pd(&(mat->element[0][index0]), _mm512_load_pd(&(mat->element[0][index1])));
		_mm512_store_pd(&(mat->element[1][index0]), _mm512_load_pd(&(mat->element[1][index1])));

		//set_ddmatrix_ij(mat, row_index1, i, tmp);
		_mm512_store_pd(&(mat->element[0][index1]), tmp512[0]);
		_mm512_store_pd(&(mat->element[1][index1]), tmp512[1]);
	}
	for(i = true_end_quo; i < true_end; i++)
	{
		rdd_set(tmp, get_ddmatrix_ij(mat, row_index0, i));
		set_ddmatrix_ij(mat, row_index0, i, get_ddmatrix_ij(mat, row_index1, i));
		set_ddmatrix_ij(mat, row_index1, i, tmp);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    long int i, true_end, true_end_start, true_end_end, real_col_dim, index0, index1;
    double tmp[DDSIZE];
    
	svfloat64_t tmp2_0, tmp2_1;

    true_end       = (col_end > mat->col_dim) ? mat->col_dim : col_end;
    true_end_start = (long int)ceil((double)col_start / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
    true_end_end   = (long int)floor((double)true_end   / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;

    real_col_dim = mat->real_col_dim;

    /* 先頭のアラインされていない部分をスカラで交換 */
    for(i = col_start; i < true_end_start; i++)
    {
        rdd_set(tmp, get_ddmatrix_ij(mat, row_index0, i));
        set_ddmatrix_ij(mat, row_index0, i, get_ddmatrix_ij(mat, row_index1, i));
        set_ddmatrix_ij(mat, row_index1, i, tmp);
    }

    /* アラインされた中央部分を NEON で 2 要素ずつ交換 */
    for(i = true_end_start; i < true_end_end; i += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(true_end_end));
        index0   = row_index0 * real_col_dim + i;
        tmp2_0  = svld1_f64(pg, &(mat->element[0][index0]));
        tmp2_1  = svld1_f64(pg, &(mat->element[1][index0]));

        index1   = row_index1 * real_col_dim + i;
        svst1_f64(pg, &(mat->element[0][index0]), svld1_f64(pg, &(mat->element[0][index1])));
        svst1_f64(pg, &(mat->element[1][index0]), svld1_f64(pg, &(mat->element[1][index1])));

        svst1_f64(pg, &(mat->element[0][index1]), tmp2_0);
        svst1_f64(pg, &(mat->element[1][index1]), tmp2_1);
    }

    /* 末尾の余りをスカラで交換 */
    for(i = true_end_end; i < true_end; i++)
    {
        rdd_set(tmp, get_ddmatrix_ij(mat, row_index0, i));
        set_ddmatrix_ij(mat, row_index0, i, get_ddmatrix_ij(mat, row_index1, i));
        set_ddmatrix_ij(mat, row_index1, i, tmp);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    long int i, true_end, true_end_start, true_end_end, real_col_dim, index0, index1;
    double tmp[DDSIZE];
    float64x2_t tmp2[DDSIZE];

    true_end       = (col_end > mat->col_dim) ? mat->col_dim : col_end;
    true_end_start = (long int)ceil((double)col_start / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
    true_end_end   = (long int)floor((double)true_end   / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;

    real_col_dim = mat->real_col_dim;

    /* 先頭のアラインされていない部分をスカラで交換 */
    for(i = col_start; i < true_end_start; i++)
    {
        rdd_set(tmp, get_ddmatrix_ij(mat, row_index0, i));
        set_ddmatrix_ij(mat, row_index0, i, get_ddmatrix_ij(mat, row_index1, i));
        set_ddmatrix_ij(mat, row_index1, i, tmp);
    }

    /* アラインされた中央部分を NEON で 2 要素ずつ交換 */
    for(i = true_end_start; i < true_end_end; i += _BNC_D_WIDTH)
    {
        index0   = row_index0 * real_col_dim + i;
        tmp2[0]  = vld1q_f64(&(mat->element[0][index0]));
        tmp2[1]  = vld1q_f64(&(mat->element[1][index0]));

        index1   = row_index1 * real_col_dim + i;
        vst1q_f64(&(mat->element[0][index0]), vld1q_f64(&(mat->element[0][index1])));
        vst1q_f64(&(mat->element[1][index0]), vld1q_f64(&(mat->element[1][index1])));

        vst1q_f64(&(mat->element[0][index1]), tmp2[0]);
        vst1q_f64(&(mat->element[1][index1]), tmp2[1]);
    }

    /* 末尾の余りをスカラで交換 */
    for(i = true_end_end; i < true_end; i++)
    {
        rdd_set(tmp, get_ddmatrix_ij(mat, row_index0, i));
        set_ddmatrix_ij(mat, row_index0, i, get_ddmatrix_ij(mat, row_index1, i));
        set_ddmatrix_ij(mat, row_index1, i, tmp);
    }
#else // No SIMD
//#endif //if 0
	long int i, j, true_end;
	double tmp[DDSIZE];

	true_end = (col_end > mat->col_dim) ? mat->col_dim : col_end;

	for(i = col_start; i < true_end; i++)
	{
		rdd_set(tmp, get_ddmatrix_ij(mat, row_index0, i));
		set_ddmatrix_ij(mat, row_index0, i, get_ddmatrix_ij(mat, row_index1, i));
		set_ddmatrix_ij(mat, row_index1, i, tmp);
	}
#endif // __AVX2__
}


#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus
