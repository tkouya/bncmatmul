/********************************************************************************/
/* dslinear.c: Double-single precision Linear Computation Library               */
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
//#include "rdd.h"
#include "dslinear.h"

#ifdef USE_GMP
#include "gmp.h"
#include "mpfr.h"
#include "mpfr_dtq_sd.h"
#include "mpflinear.h"
#endif //USE_GMP//

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// dsrel_diff
dsfloat dsrel_diff(dsfloat a, dsfloat b)
{
    dsfloat rel_diff, abs_a;

    //rel_diff = fabs(a - b);
    rds_sub(rel_diff.val, a.val, b.val);
    rds_abs(rel_diff.val, rel_diff.val);

    //if(a != 0.0)
    if(rds_cmp_ui(a.val, 0UL) != 0)
    {
//        rel_diff /= fabs(a);
        rds_abs(abs_a.val, a.val);
        rds_div(rel_diff.val, rel_diff.val, a.val);
    }

    return rel_diff;
}

dsfloat dsrel_diff_array(dsfloat approx_a[], dsfloat approx_b[], int dim, int print_flag)
{
    int i;
    dsfloat rel_min, rel_max, rel_ave, rel_diff;

    rel_diff = dsrel_diff(approx_a[0], approx_b[0]);
    rds_set(rel_min.val, rel_diff.val);
    rds_set(rel_max.val, rel_diff.val);
    rds_set(rel_ave.val, rel_diff.val);

    for(i = 1; i < dim; i++)
    {
        rel_diff = dsrel_diff(approx_a[i], approx_b[i]);
        if(rds_cmp(rel_diff.val, rel_min.val) < 0) rds_set(rel_min.val, rel_diff.val);
        if(rds_cmp(rel_diff.val, rel_max.val) > 0) rds_set(rel_max.val, rel_diff.val);
        //rel_ave += rel_diff;
        rds_add(rel_ave.val, rel_ave.val, rel_diff.val);
    }
    //rel_ave /= (dsfloat)dim;
    rds_div_ui(rel_ave.val, rel_ave.val, (unsigned long)dim);

    if(print_flag == 1)
    {
        printf("max_rel_diff, min_rel_diff, ave_rel_diff:"); rds_out_str(rel_max.val); printf(" "); rds_out_str(rel_min.val);  printf(" "); rds_out_str(rel_ave.val); printf("\n"); 
    }

    return rel_max;
}

#if defined(USE_GMP) && defined(USE_MPFR)
// Frobenius norm
dsfloat dsnormf(dsfloat array[], int dim)
{
    int i;
    dsfloat ret, tmp;
    mpfr_t mpfr_ret;

    rds_set_ui(ret.val, 0UL);
    for(i = 0; i < dim; i++)
    {
        rds_mul(tmp.val, array[i].val, array[i].val);
        rds_add(ret.val, ret.val, tmp.val);
    }
    //printf("ret.val = "); rds_out_str(ret.val); printf("\n");
//  rds_sqrt(ret, ret);
    mpfr_init2(mpfr_ret, 128);
    mpfr_set_ds(mpfr_ret, ret.val, MPFR_RNDN);
    mpfr_sqrt(mpfr_ret, mpfr_ret, MPFR_RNDN);
    mpfr_get_ds(ret.val, mpfr_ret, MPFR_RNDN);
    mpfr_clear(mpfr_ret);
    return ret;
}

// generate a text vector: vec(i) := sqrt(sqrt_seed) * (i + 1)
/* void set_test_dsvector(dsfloat vec[], int sqrt_seed, int dim)
{
    int i, j;
    dsfloat ddsqrt;
    mpfr_t mpfrsqrt;

    // ddsqrt := sqrt(sqrt_seed)
    mpfr_init2(mpfrsqrt, 128);
    mpfr_sqrt_ui(mpfrsqrt, (unsigned long)sqrt_seed, MPFR_RNDN);
    mpfr_get_ds(ddsqrt.val, mpfrsqrt, MPFR_RNDN);
//    rds_set_ui(ddsqrt.val, sqrt_seed);
    //rds_sqrt(ddsqrt.val, ddsqrt.val);
    //rds_sqrt_ui(ddsqrt.val, sqrt_seed);

    //printf("set_test_dsmatrix: coef = "); rds_out_str(ddsqrt.val); printf("\n");

    for(i = 0; i < dim; i++)
    {
        //printf("%5d: ", i);
        rds_set_ui(vec[i].val, i + 1);
        rds_mul(vec[i].val, vec[i].val, ddsqrt.val);
    }
} */
#endif // defined(USE_GMP) && defined(USE_MPFR)

// initialize DSVector
DSVector init_dsvector(int dimension)
{
	DSVector ret = NULL;
	long int i, real_dim;

	if(dimension <= 0)
	{
		fprintf(stderr, "ERROR: init_dsvector\n");
		return ret;
	}

	ret = (DSVector)BNC_MALLOC(sizeof(dsvector));
	if(ret == NULL)
		return ret;

	// real_dim is the nearest positive multiplier of _BNC_S_WIDTH
	real_dim = (long int)ceil((float)(dimension) / (float)_BNC_S_WIDTH) * _BNC_S_WIDTH;

	ret->element[0] = (float *)BNC_CALLOC(real_dim, sizeof(float));
	if(ret->element[0] == NULL)
	{ 	
		free(ret);
		return NULL;
	}
	ret->element[1] = (float *)BNC_CALLOC(real_dim, sizeof(float));
	if(ret->element[1] == NULL)
	{
		free(ret->element[0]);
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
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero4;

	zero4 = _mm512_setzero_ps();
	for(i = 0; i < real_dim; i += _BNC_S_WIDTH)
	{
		_mm512_store_ps(&(ret->element[0][i]), zero4);
		_mm512_store_ps(&(ret->element[1][i]), zero4);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t zero4;

	zero4 = svdup_f32(0.0f);
	for(i = 0; i < real_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(real_dim));
		svst1_f32(pg, &(ret->element[0][i]), zero4);
		svst1_f32(pg, &(ret->element[1][i]), zero4);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t zero4;

	zero4 = vdupq_n_f32(0.0f);
	for(i = 0; i < real_dim; i += _BNC_S_WIDTH)
	{
		vst1q_f32(&(ret->element[0][i]), zero4);
		vst1q_f32(&(ret->element[1][i]), zero4);
	}
#else // others
	for(i = 0; i < dimension; i++)
	{
		ret->element[0][i] = 0.0f;
		ret->element[1][i] = 0.0f;
	}
#endif // __AVX2__

	ret->dim = dimension;
	ret->real_dim = real_dim;

	return ret;
}

// free DSVector
void free_dsvector(DSVector vec)
{
    int i;

    for(i = 0; i < DSSIZE; i++)
        free(vec->element[i]);

    free(vec);
}

// DSVector vec -> dsfloat array
void set_dsfloat_dsvec(dsfloat ret[], int ret_dim, DSVector vec)
{
    int index, j, dim;

    dim = (ret_dim < vec->dim) ? ret_dim : vec->dim;

    for(index = 0; index < dim; index++)
    {
        for(j = 0; j < DSSIZE; j++)
            ret[index].val[j] = vec->element[j][index];
    }
}

// dsfloat array -> DSVector ret
void set_dsvector_dsfloat(DSVector ret, dsfloat array[], int array_dim)
{
    int index, j, dim;

    dim = (ret->dim < array_dim) ? ret->dim : array_dim;

    for(index = 0; index < dim; index++)
    {
        for(j = 0; j < DSSIZE; j++)
            ret->element[j][index] = array[index].val[j];
    }
}

// print dsvector
void print_dsvector(DSVector vec)
{
	long int index;

	for(index = 0; index < vec->dim; index++)
	{
		printf("%4ld: ", index);
		//c_ds_write((vec->element + index * DSSIZE));
		c_ds_write(GET_DSVECTOR_I(vec, index));
	}
}

// set a zero vector
void set0_dsvector(DSVector vec)
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
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero4;

	zero4 = _mm512_setzero_ps();
	for(i = 0; i < vec->real_dim; i += _BNC_S_WIDTH)
	{
		_mm512_store_ps(&vec->element[0][i], zero4);
		_mm512_store_ps(&vec->element[1][i], zero4);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t zero4;

	zero4 = svdup_f32(0.0f);
	for(i = 0; i < vec->real_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(vec->real_dim));
		svst1_f32(pg, &vec->element[0][i], zero4);
		svst1_f32(pg, &vec->element[1][i], zero4);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t zero4;

	zero4 = vdupq_n_f32(0.0f);
	for(i = 0; i < vec->real_dim; i += _BNC_S_WIDTH)
	{
		vst1q_f32(&vec->element[0][i], zero4);
		vst1q_f32(&vec->element[1][i], zero4);
	}
#else // others
	for(i = 0; i < vec->dim; i++)
	{
		vec->element[0][i] = 0.0f;
		vec->element[1][i] = 0.0f;
	}
#endif // __AVX2__
}

// set_dsvector_i_str
void set_dsvector_i_str(DSVector vec, long int index, const char *str)
{
	float tmp[DSSIZE];

	//rds_get_str(tmp, str);
	rds_set_str(tmp, str);

	set_dsvector_i(vec, index, tmp);
}

/*************************************************/
/* Vector Calculations for DSVector               */
/*
void add_dsvector(DSVector c, DSVector a, DSVector b)
void add2_dsvector(DSVector c, DSVector a)
void sub_dsvector(DSVector c, DSVector a, DSVector b)
void sub2_dsvector(DSVector c, FVector a)
void cmul_dsvector(DSVector c, float val[DSSIZE], DSVector a)
void cmul2_dsvector(DSVector c, float val[DSSIZE])
void add_cmul_dsvector(DSVector c, DSVector a, float val[DSSIZE], DSVector b)
float ip_dsvector(DSVector a, DSVector b)
float norm1_dsvector(DSVector a)
float norm2_dsvector(DSVector a)
float normi_dsvector(DSVector a)
void subst_dsvector(DSVector c, DSVector a)
*/
/*************************************************/
/* c = a + b */
void add_dsvector(DSVector c, DSVector a, DSVector b)
{
    long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_dsvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256 in_ret[DSSIZE], in_a_val[DSSIZE], in_b_val[DSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_ps(&(a->element[1][index]));
        in_b_val[0] = _mm256_load_ps(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_ps(&(b->element[1][index]));

        _bncavx2_rds_add(in_ret, in_a_val, in_b_val);

        _mm256_store_ps(&(c->element[0][index]), in_ret[0]);
        _mm256_store_ps(&(c->element[1][index]), in_ret[1]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512 in_ret[DSSIZE], in_a_val[DSSIZE], in_b_val[DSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm512_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_ps(&(a->element[1][index]));
        in_b_val[0] = _mm512_load_ps(&(b->element[0][index]));
        in_b_val[1] = _mm512_load_ps(&(b->element[1][index]));

        _bncavx512_rds_add(in_ret, in_a_val, in_b_val);

        _mm512_store_ps(&(c->element[0][index]), in_ret[0]);
        _mm512_store_ps(&(c->element[1][index]), in_ret[1]);
   }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
    
	svfloat32_t in_ret_0, in_ret_1;
	svfloat32_t in_a_val_0, in_a_val_1;
	svfloat32_t in_b_val_0, in_b_val_1;

    for(index = 0; index < c->real_dim; index += (long int)svcntw())
    {
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(c->real_dim));
        in_a_val_0 = svld1_f32(pg, &(a->element[0][index]));
        in_a_val_1 = svld1_f32(pg, &(a->element[1][index]));
        in_b_val_0 = svld1_f32(pg, &(b->element[0][index]));
        in_b_val_1 = svld1_f32(pg, &(b->element[1][index]));

        _bncsve2_rds_add(pg, &in_ret_0, &in_ret_1, in_a_val_0, in_a_val_1, in_b_val_0, in_b_val_1);

        svst1_f32(pg, &(c->element[0][index]), in_ret_0);
        svst1_f32(pg, &(c->element[1][index]), in_ret_1);
   }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float32x4_t in_ret[DSSIZE], in_a_val[DSSIZE], in_b_val[DSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = vld1q_f32(&(a->element[0][index]));
        in_a_val[1] = vld1q_f32(&(a->element[1][index]));
        in_b_val[0] = vld1q_f32(&(b->element[0][index]));
        in_b_val[1] = vld1q_f32(&(b->element[1][index]));

        _bncneon_rds_add(in_ret, in_a_val, in_b_val);

        vst1q_f32(&(c->element[0][index]), in_ret[0]);
        vst1q_f32(&(c->element[1][index]), in_ret[1]);
   }
#else // others
	float tmp[DSSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rds_add(tmp, get_dsvector_i(a, i),  get_dsvector_i(b, i));
		set_dsvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c += a */
void add2_dsvector(DSVector c, DSVector a)
{
	long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: add2_dsvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256 in_ret[DSSIZE], in_a_val[DSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_ps(&(a->element[1][index]));
        in_ret[0] = _mm256_load_ps(&(c->element[0][index]));
        in_ret[1] = _mm256_load_ps(&(c->element[1][index]));

        _bncavx2_rds_add(in_ret, in_ret, in_a_val);

        _mm256_store_ps(&(c->element[0][index]), in_ret[0]);
        _mm256_store_ps(&(c->element[1][index]), in_ret[1]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512 in_ret[DSSIZE], in_a_val[DSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm512_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_ps(&(a->element[1][index]));
        in_ret[0] = _mm512_load_ps(&(c->element[0][index]));
        in_ret[1] = _mm512_load_ps(&(c->element[1][index]));

        _bncavx512_rds_add(in_ret, in_ret, in_a_val);

        _mm512_store_ps(&(c->element[0][index]), in_ret[0]);
        _mm512_store_ps(&(c->element[1][index]), in_ret[1]);
   }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
    
	svfloat32_t in_ret_0, in_ret_1;
	svfloat32_t in_a_val_0, in_a_val_1;

    for(index = 0; index < c->real_dim; index += (long int)svcntw())
    {
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(c->real_dim));
        in_a_val_0 = svld1_f32(pg, &(a->element[0][index]));
        in_a_val_1 = svld1_f32(pg, &(a->element[1][index]));
        in_ret_0 = svld1_f32(pg, &(c->element[0][index]));
        in_ret_1 = svld1_f32(pg, &(c->element[1][index]));

        _bncsve2_rds_add(pg, &in_ret_0, &in_ret_1, in_ret_0, in_ret_1, in_a_val_0, in_a_val_1);

        svst1_f32(pg, &(c->element[0][index]), in_ret_0);
        svst1_f32(pg, &(c->element[1][index]), in_ret_1);
   }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float32x4_t in_ret[DSSIZE], in_a_val[DSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = vld1q_f32(&(a->element[0][index]));
        in_a_val[1] = vld1q_f32(&(a->element[1][index]));
        in_ret[0] = vld1q_f32(&(c->element[0][index]));
        in_ret[1] = vld1q_f32(&(c->element[1][index]));

        _bncneon_rds_add(in_ret, in_ret, in_a_val);

        vst1q_f32(&(c->element[0][index]), in_ret[0]);
        vst1q_f32(&(c->element[1][index]), in_ret[1]);
   }
#else // others
	float tmp[DSSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rds_add(tmp, get_dsvector_i(c, i), get_dsvector_i(a, i));
		set_dsvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c = a - b */
void sub_dsvector(DSVector c, DSVector a, DSVector b)
{
    long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_dsvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256 in_ret[DSSIZE], in_a_val[DSSIZE], in_b_val[DSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_ps(&(a->element[1][index]));
        in_b_val[0] = _mm256_load_ps(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_ps(&(b->element[1][index]));

        _bncavx2_rds_sub(in_ret, in_a_val, in_b_val);

        _mm256_store_ps(&(c->element[0][index]), in_ret[0]);
        _mm256_store_ps(&(c->element[1][index]), in_ret[1]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512 in_ret[DSSIZE], in_a_val[DSSIZE], in_b_val[DSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm512_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_ps(&(a->element[1][index]));
        in_b_val[0] = _mm512_load_ps(&(b->element[0][index]));
        in_b_val[1] = _mm512_load_ps(&(b->element[1][index]));

        _bncavx512_rds_sub(in_ret, in_a_val, in_b_val);

        _mm512_store_ps(&(c->element[0][index]), in_ret[0]);
        _mm512_store_ps(&(c->element[1][index]), in_ret[1]);
   }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
    
	svfloat32_t in_ret_0, in_ret_1;
	svfloat32_t in_a_val_0, in_a_val_1;
	svfloat32_t in_b_val_0, in_b_val_1;

    for(index = 0; index < c->real_dim; index += (long int)svcntw())
    {
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(c->real_dim));
        in_a_val_0 = svld1_f32(pg, &(a->element[0][index]));
        in_a_val_1 = svld1_f32(pg, &(a->element[1][index]));
        in_b_val_0 = svld1_f32(pg, &(b->element[0][index]));
        in_b_val_1 = svld1_f32(pg, &(b->element[1][index]));

        _bncsve2_rds_neg(pg, &in_ret_0, &in_ret_1, in_b_val_0, in_b_val_1);
		_bncsve2_rds_add(pg, &in_ret_0, &in_ret_1, in_a_val_0, in_a_val_1, in_ret_0, in_ret_1);

        svst1_f32(pg, &(c->element[0][index]), in_ret_0);
        svst1_f32(pg, &(c->element[1][index]), in_ret_1);
   }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float32x4_t in_ret[DSSIZE], in_a_val[DSSIZE], in_b_val[DSSIZE];

    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = vld1q_f32(&(a->element[0][index]));
        in_a_val[1] = vld1q_f32(&(a->element[1][index]));
        in_b_val[0] = vld1q_f32(&(b->element[0][index]));
        in_b_val[1] = vld1q_f32(&(b->element[1][index]));

        _bncneon_rds_sub(in_ret, in_a_val, in_b_val);

        vst1q_f32(&(c->element[0][index]), in_ret[0]);
        vst1q_f32(&(c->element[1][index]), in_ret[1]);
   }
#else // others
	float tmp[DSSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rds_sub(tmp, get_dsvector_i(a, i),  get_dsvector_i(b, i));
		set_dsvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c -= a */
void sub2_dsvector(DSVector c, DSVector a)
{
	long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: add2_dsvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256 in_ret[DSSIZE], in_a_val[DSSIZE];

	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_ps(&(a->element[1][index]));
        in_ret[0] = _mm256_load_ps(&(c->element[0][index]));
        in_ret[1] = _mm256_load_ps(&(c->element[1][index]));

        _bncavx2_rds_sub(in_ret, in_ret, in_a_val);

        _mm256_store_ps(&(c->element[0][index]), in_ret[0]);
        _mm256_store_ps(&(c->element[1][index]), in_ret[1]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512 in_ret[DSSIZE], in_a_val[DSSIZE];

	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = _mm512_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm512_load_ps(&(a->element[1][index]));
        in_ret[0] = _mm512_load_ps(&(c->element[0][index]));
        in_ret[1] = _mm512_load_ps(&(c->element[1][index]));

        _bncavx512_rds_sub(in_ret, in_ret, in_a_val);

        _mm512_store_ps(&(c->element[0][index]), in_ret[0]);
        _mm512_store_ps(&(c->element[1][index]), in_ret[1]);
   }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
    
	svfloat32_t in_ret_0, in_ret_1;
	svfloat32_t in_a_val_0, in_a_val_1;

	for(index = 0; index < c->real_dim; index += (long int)svcntw())
    {
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(c->real_dim));
        in_a_val_0 = svld1_f32(pg, &(a->element[0][index]));
        in_a_val_1 = svld1_f32(pg, &(a->element[1][index]));
        in_ret_0 = svld1_f32(pg, &(c->element[0][index]));
        in_ret_1 = svld1_f32(pg, &(c->element[1][index]));

        _bncsve2_rds_neg(pg, &in_ret_0, &in_ret_1, in_a_val_0, in_a_val_1);
		_bncsve2_rds_add(pg, &in_ret_0, &in_ret_1, in_ret_0, in_ret_1, in_ret_0, in_ret_1);

        svst1_f32(pg, &(c->element[0][index]), in_ret_0);
        svst1_f32(pg, &(c->element[1][index]), in_ret_1);
   }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float32x4_t in_ret[DSSIZE], in_a_val[DSSIZE];

	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
        in_a_val[0] = vld1q_f32(&(a->element[0][index]));
        in_a_val[1] = vld1q_f32(&(a->element[1][index]));
        in_ret[0] = vld1q_f32(&(c->element[0][index]));
        in_ret[1] = vld1q_f32(&(c->element[1][index]));

        _bncneon_rds_sub(in_ret, in_ret, in_a_val);

        vst1q_f32(&(c->element[0][index]), in_ret[0]);
        vst1q_f32(&(c->element[1][index]), in_ret[1]);
   }
#else // others
	float tmp[DSSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rds_sub(tmp, get_dsvector_i(c, i), get_dsvector_i(a, i));
		set_dsvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* c = val * a */
void cmul_dsvector(DSVector c, float val[DSSIZE], DSVector a)
{
    long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: cmul_dvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4[DSSIZE], c4[DSSIZE], val4[DSSIZE];

	val4[0] = _mm256_set1_ps(val[0]);
	val4[1] = _mm256_set1_ps(val[1]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		//set_dsvector_i(c, i, val * get_dsvector_i(a, i));
		a4[0] = _mm256_load_ps(&(a->element[0][index]));
		a4[1] = _mm256_load_ps(&(a->element[1][index]));

		_bncavx2_rds_mul(c4, val4, a4);

		_mm256_store_ps(&(c->element[0][index]), c4[0]);
		_mm256_store_ps(&(c->element[1][index]), c4[1]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a4[DSSIZE], c4[DSSIZE], val4[DSSIZE];

	val4[0] = _mm512_set1_ps(val[0]);
	val4[1] = _mm512_set1_ps(val[1]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		//set_dsvector_i(c, i, val * get_dsvector_i(a, i));
		a4[0] = _mm512_load_ps(&(a->element[0][index]));
		a4[1] = _mm512_load_ps(&(a->element[1][index]));

		_bncavx512_rds_mul(c4, val4, a4);

		_mm512_store_ps(&(c->element[0][index]), c4[0]);
		_mm512_store_ps(&(c->element[1][index]), c4[1]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t a4_0, a4_1;
	svfloat32_t c4_0, c4_1;
	svfloat32_t val4_0, val4_1;

	val4_0 = svdup_f32(val[0]);
	val4_1 = svdup_f32(val[1]);
	for(index = 0; index < c->real_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(c->real_dim));
		//set_dsvector_i(c, i, val * get_dsvector_i(a, i));
		a4_0 = svld1_f32(pg, &(a->element[0][index]));
		a4_1 = svld1_f32(pg, &(a->element[1][index]));

		_bncsve2_rds_mul(pg, &c4_0, &c4_1, val4_0, val4_1, a4_0, a4_1);

		svst1_f32(pg, &(c->element[0][index]), c4_0);
		svst1_f32(pg, &(c->element[1][index]), c4_1);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a4[DSSIZE], c4[DSSIZE], val4[DSSIZE];

	val4[0] = vdupq_n_f32(val[0]);
	val4[1] = vdupq_n_f32(val[1]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		//set_dsvector_i(c, i, val * get_dsvector_i(a, i));
		a4[0] = vld1q_f32(&(a->element[0][index]));
		a4[1] = vld1q_f32(&(a->element[1][index]));

		_bncneon_rds_mul(c4, val4, a4);

		vst1q_f32(&(c->element[0][index]), c4[0]);
		vst1q_f32(&(c->element[1][index]), c4[1]);
	}
#else // others
	float tmp[DSSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rds_mul(tmp, val, get_dsvector_i(a, i));
		set_dsvector_i(c, i, tmp);
	}
#endif // __AVX2__

}

/* c *= val */
void cmul2_dsvector(DSVector c, float val[DSSIZE])
{
	long int i, index;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 c4[DSSIZE], val4[DSSIZE];

	val4[0] = _mm256_set1_ps(val[0]);
	val4[1] = _mm256_set1_ps(val[1]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		//set_dsvector_i(c, i, val * get_dsvector_i(a, i));
		c4[0] = _mm256_load_ps(&(c->element[0][index]));
		c4[1] = _mm256_load_ps(&(c->element[1][index]));

		_bncavx2_rds_mul(c4, val4, c4);

		_mm256_store_ps(&(c->element[0][index]), c4[0]);
		_mm256_store_ps(&(c->element[1][index]), c4[1]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 c4[DSSIZE], val4[DSSIZE];

	val4[0] = _mm512_set1_ps(val[0]);
	val4[1] = _mm512_set1_ps(val[1]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		//set_dsvector_i(c, i, val * get_dsvector_i(a, i));
		c4[0] = _mm512_load_ps(&(c->element[0][index]));
		c4[1] = _mm512_load_ps(&(c->element[1][index]));

		_bncavx512_rds_mul(c4, val4, c4);

		_mm512_store_ps(&(c->element[0][index]), c4[0]);
		_mm512_store_ps(&(c->element[1][index]), c4[1]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t c4_0, c4_1;
	svfloat32_t val4_0, val4_1;

	val4_0 = svdup_f32(val[0]);
	val4_1 = svdup_f32(val[1]);
	for(index = 0; index < c->real_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(c->real_dim));
		//set_dsvector_i(c, i, val * get_dsvector_i(a, i));
		c4_0 = svld1_f32(pg, &(c->element[0][index]));
		c4_1 = svld1_f32(pg, &(c->element[1][index]));

		_bncsve2_rds_mul(pg, &c4_0, &c4_1, val4_0, val4_1, c4_0, c4_1);

		svst1_f32(pg, &(c->element[0][index]), c4_0);
		svst1_f32(pg, &(c->element[1][index]), c4_1);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t c4[DSSIZE], val4[DSSIZE];

	val4[0] = vdupq_n_f32(val[0]);
	val4[1] = vdupq_n_f32(val[1]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		//set_dsvector_i(c, i, val * get_dsvector_i(a, i));
		c4[0] = vld1q_f32(&(c->element[0][index]));
		c4[1] = vld1q_f32(&(c->element[1][index]));

		_bncneon_rds_mul(c4, val4, c4);

		vst1q_f32(&(c->element[0][index]), c4[0]);
		vst1q_f32(&(c->element[1][index]), c4[1]);
	}
#else // others
	float tmp[DSSIZE];

	for(i = 0; i < c->dim; i++)
	{
		rds_mul(tmp, val, get_dsvector_i(c, i));
		set_dsvector_i(c, i, tmp);
	}
#endif // __AVX2__	float tmp[DSSIZE];

}

/* c = a + val * b */
void add_cmul_dsvector(DSVector c, DSVector a, float val[DSSIZE], DSVector b)
{
	long int i, index;
	int _k;
	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_cmul_dsvector\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4[DSSIZE], b4[DSSIZE], c4[DSSIZE], val4[DSSIZE];
	for(_k = 0; _k < DSSIZE; _k++) val4[_k] = _mm256_set1_ps(val[_k]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		for(_k = 0; _k < DSSIZE; _k++){ a4[_k] = _mm256_load_ps(&(a->element[_k][index])); b4[_k] = _mm256_load_ps(&(b->element[_k][index])); }
		_bncavx2_rds_mul(c4, val4, b4);
		_bncavx2_rds_add(c4, a4, c4);
		for(_k = 0; _k < DSSIZE; _k++) _mm256_store_ps(&(c->element[_k][index]), c4[_k]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a4[DSSIZE], b4[DSSIZE], c4[DSSIZE], val4[DSSIZE];
	for(_k = 0; _k < DSSIZE; _k++) val4[_k] = _mm512_set1_ps(val[_k]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		for(_k = 0; _k < DSSIZE; _k++){ a4[_k] = _mm512_load_ps(&(a->element[_k][index])); b4[_k] = _mm512_load_ps(&(b->element[_k][index])); }
		_bncavx512_rds_mul(c4, val4, b4);
		_bncavx512_rds_add(c4, a4, c4);
		for(_k = 0; _k < DSSIZE; _k++) _mm512_store_ps(&(c->element[_k][index]), c4[_k]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t sa_0, sa_1, sb_0, sb_1, sc_0, sc_1, sv_0, sv_1;
	sv_0 = svdup_f32(val[0]); sv_1 = svdup_f32(val[1]);
	for(index = 0; index < c->real_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)c->real_dim);
		sa_0 = svld1_f32(pg, &(a->element[0][index])); sb_0 = svld1_f32(pg, &(b->element[0][index]));
		sa_1 = svld1_f32(pg, &(a->element[1][index])); sb_1 = svld1_f32(pg, &(b->element[1][index]));
		_bncsve2_rds_mul(pg, &sc_0, &sc_1, sv_0, sv_1, sb_0, sb_1);
		_bncsve2_rds_add(pg, &sc_0, &sc_1, sa_0, sa_1, sc_0, sc_1);
		svst1_f32(pg, &(c->element[0][index]), sc_0);
		svst1_f32(pg, &(c->element[1][index]), sc_1);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t na[DSSIZE], nb[DSSIZE], nc[DSSIZE], nv[DSSIZE];
	for(_k = 0; _k < DSSIZE; _k++) nv[_k] = vdupq_n_f32(val[_k]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		for(_k = 0; _k < DSSIZE; _k++){ na[_k] = vld1q_f32(&(a->element[_k][index])); nb[_k] = vld1q_f32(&(b->element[_k][index])); }
		_bncneon_rds_mul(nc, nv, nb);
		_bncneon_rds_add(nc, na, nc);
		for(_k = 0; _k < DSSIZE; _k++) vst1q_f32(&(c->element[_k][index]), nc[_k]);
	}
#else // scalar
	float tmp[DSSIZE];
	for(i = 0; i < c->dim; i++)
	{
		rds_mul(tmp, val, get_dsvector_i(b, i));
		rds_add(tmp, tmp, get_dsvector_i(a, i));
		set_dsvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

void sub_cmul_dsvector(DSVector c, DSVector a, float val[DSSIZE], DSVector b)
{
	long int i, index;
	int _k;
	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_cmul_dsvector\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4[DSSIZE], b4[DSSIZE], c4[DSSIZE], val4[DSSIZE];
	for(_k = 0; _k < DSSIZE; _k++) val4[_k] = _mm256_set1_ps(val[_k]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		for(_k = 0; _k < DSSIZE; _k++){ a4[_k] = _mm256_load_ps(&(a->element[_k][index])); b4[_k] = _mm256_load_ps(&(b->element[_k][index])); }
		_bncavx2_rds_mul(c4, val4, b4);
		_bncavx2_rds_sub(c4, a4, c4);
		for(_k = 0; _k < DSSIZE; _k++) _mm256_store_ps(&(c->element[_k][index]), c4[_k]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a4[DSSIZE], b4[DSSIZE], c4[DSSIZE], val4[DSSIZE];
	for(_k = 0; _k < DSSIZE; _k++) val4[_k] = _mm512_set1_ps(val[_k]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		for(_k = 0; _k < DSSIZE; _k++){ a4[_k] = _mm512_load_ps(&(a->element[_k][index])); b4[_k] = _mm512_load_ps(&(b->element[_k][index])); }
		_bncavx512_rds_mul(c4, val4, b4);
		_bncavx512_rds_sub(c4, a4, c4);
		for(_k = 0; _k < DSSIZE; _k++) _mm512_store_ps(&(c->element[_k][index]), c4[_k]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t sa_0, sa_1, sb_0, sb_1, sc_0, sc_1, sv_0, sv_1;
	sv_0 = svdup_f32(val[0]); sv_1 = svdup_f32(val[1]);
	for(index = 0; index < c->real_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)c->real_dim);
		sa_0 = svld1_f32(pg, &(a->element[0][index])); sb_0 = svld1_f32(pg, &(b->element[0][index]));
		sa_1 = svld1_f32(pg, &(a->element[1][index])); sb_1 = svld1_f32(pg, &(b->element[1][index]));
		_bncsve2_rds_mul(pg, &sc_0, &sc_1, sv_0, sv_1, sb_0, sb_1);
		_bncsve2_rds_neg(pg, &sc_0, &sc_1, sc_0, sc_1);
		_bncsve2_rds_add(pg, &sc_0, &sc_1, sa_0, sa_1, sc_0, sc_1);
		svst1_f32(pg, &(c->element[0][index]), sc_0);
		svst1_f32(pg, &(c->element[1][index]), sc_1);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t na[DSSIZE], nb[DSSIZE], nc[DSSIZE], nv[DSSIZE];
	for(_k = 0; _k < DSSIZE; _k++) nv[_k] = vdupq_n_f32(val[_k]);
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		for(_k = 0; _k < DSSIZE; _k++){ na[_k] = vld1q_f32(&(a->element[_k][index])); nb[_k] = vld1q_f32(&(b->element[_k][index])); }
		_bncneon_rds_mul(nc, nv, nb);
		_bncneon_rds_sub(nc, na, nc);
		for(_k = 0; _k < DSSIZE; _k++) vst1q_f32(&(c->element[_k][index]), nc[_k]);
	}
#else // scalar
	float tmp[DSSIZE];
	for(i = 0; i < c->dim; i++)
	{
		rds_mul(tmp, val, get_dsvector_i(b, i));
		rds_sub(tmp, get_dsvector_i(a, i), tmp);
		set_dsvector_i(c, i, tmp);
	}
#endif // __AVX2__
}

/* (a, b) */
void ip_dsvector(float ret[DSSIZE], DSVector a, DSVector b)
{
	long int i, index;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: ip_dsvector\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4[DSSIZE], b4[DSSIZE], ret4[DSSIZE], tmp4[DSSIZE];

	_bncavx2_set0_ds(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		a4[0] = _mm256_load_ps(&(a->element[0][index]));
		a4[1] = _mm256_load_ps(&(a->element[1][index]));
		b4[0] = _mm256_load_ps(&(b->element[0][index]));
		b4[1] = _mm256_load_ps(&(b->element[1][index]));

//		rds_mul(tmp, get_dsvector_i(a, i), get_dsvector_i(b, i));
//		rds_add(ret, ret, tmp);
		_bncavx2_rds_mul(tmp4, a4, b4);
		_bncavx2_rds_add(ret4, ret4, tmp4);
	}
	_bncavx2_rds_sum256(ret, ret4);
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a4[DSSIZE], b4[DSSIZE], ret4[DSSIZE], tmp4[DSSIZE];

	_bncavx512_set0_ds(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		a4[0] = _mm512_load_ps(&(a->element[0][index]));
		a4[1] = _mm512_load_ps(&(a->element[1][index]));
		b4[0] = _mm512_load_ps(&(b->element[0][index]));
		b4[1] = _mm512_load_ps(&(b->element[1][index]));

//		rds_mul(tmp, get_dsvector_i(a, i), get_dsvector_i(b, i));
//		rds_add(ret, ret, tmp);
		_bncavx512_rds_mul(tmp4, a4, b4);
		_bncavx512_rds_add(ret4, ret4, tmp4);
	}
	_bncavx512_rds_sum512(ret, ret4);
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t a4_0, a4_1;
	svfloat32_t b4_0, b4_1;
	svfloat32_t ret4_0, ret4_1;
	svfloat32_t tmp4_0, tmp4_1;

	_bncsve2_rds_set0(&ret4_0, &ret4_1);
	for(index = 0; index < a->real_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(a->real_dim));
		a4_0 = svld1_f32(pg, &(a->element[0][index]));
		a4_1 = svld1_f32(pg, &(a->element[1][index]));
		b4_0 = svld1_f32(pg, &(b->element[0][index]));
		b4_1 = svld1_f32(pg, &(b->element[1][index]));

//		rds_mul(tmp, get_dsvector_i(a, i), get_dsvector_i(b, i));
//		rds_add(ret, ret, tmp);
		_bncsve2_rds_mul(svptrue_b32(), &tmp4_0, &tmp4_1, a4_0, a4_1, b4_0, b4_1);
		_bncsve2_rds_add(svptrue_b32(), &ret4_0, &ret4_1, ret4_0, ret4_1, tmp4_0, tmp4_1);
	}
	_bncsve2_rds_sum128(svptrue_b32(), ret, ret4_0, ret4_1);
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a4[DSSIZE], b4[DSSIZE], ret4[DSSIZE], tmp4[DSSIZE];

	_bncneon_set0_ds(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		a4[0] = vld1q_f32(&(a->element[0][index]));
		a4[1] = vld1q_f32(&(a->element[1][index]));
		b4[0] = vld1q_f32(&(b->element[0][index]));
		b4[1] = vld1q_f32(&(b->element[1][index]));

//		rds_mul(tmp, get_dsvector_i(a, i), get_dsvector_i(b, i));
//		rds_add(ret, ret, tmp);
		_bncneon_rds_mul(tmp4, a4, b4);
		_bncneon_rds_add(ret4, ret4, tmp4);
	}
	_bncneon_rds_sum128f(ret, ret4);
#else // others
	float tmp[DSSIZE];

	set0_ds(ret);
	for(i = 0; i < a->dim; i++)
	{
		rds_mul(tmp, get_dsvector_i(a, i), get_dsvector_i(b, i));
		rds_add(ret, ret, tmp);
	}
#endif // __AVX2__

	return;
}

/* c := a */
void subst_dsvector(DSVector c, DSVector a)
{
	long int i;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
//	for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i));
		_mm256_store_ps(&(c->element[0][i]), _mm256_load_ps(&(a->element[0][i])));
		_mm256_store_ps(&(c->element[1][i]), _mm256_load_ps(&(a->element[1][i])));
	}
#elif defined(__AVX512F__) // __AVX512F__
//	for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i));
		_mm512_store_ps(&(c->element[0][i]), _mm512_load_ps(&(a->element[0][i])));
		_mm512_store_ps(&(c->element[1][i]), _mm512_load_ps(&(a->element[1][i])));
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
//	for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(a->real_dim));
		//set_dvector_i(c, i, get_dvector_i(a, i));
		svst1_f32(pg, &(c->element[0][i]), svld1_f32(pg, &(a->element[0][i])));
		svst1_f32(pg, &(c->element[1][i]), svld1_f32(pg, &(a->element[1][i])));
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
//	for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i));
		vst1q_f32(&(c->element[0][i]), vld1q_f32(&(a->element[0][i])));
		vst1q_f32(&(c->element[1][i]), vld1q_f32(&(a->element[1][i])));
	}
#else // others
	for(i = 0; i < a->dim; i++)
		set_dsvector_i(c, i, get_dsvector_i(a, i));
#endif // __AVX2__
}

/* c := -a */
void neg_dsvector(DSVector c, DSVector a)
{
	long int i;
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 tmp[DSSIZE];

	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//rds_neg(tmp, get_dsvector_i(a, i));
		//set_dsvector_i(c, i, tmp);
		tmp[0] = _bncavx2_fneg(_mm256_load_ps(&(a->element[0][i])));
		tmp[1] = _bncavx2_fneg(_mm256_load_ps(&(a->element[1][i])));
		_mm256_store_ps(&(c->element[0][i]), tmp[0]);
		_mm256_store_ps(&(c->element[1][i]), tmp[1]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 tmp[DSSIZE];

	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//rds_neg(tmp, get_dsvector_i(a, i));
		//set_dsvector_i(c, i, tmp);
		tmp[0] = _bncavx512_fneg(_mm512_load_ps(&(a->element[0][i])));
		tmp[1] = _bncavx512_fneg(_mm512_load_ps(&(a->element[1][i])));
		_mm512_store_ps(&(c->element[0][i]), tmp[0]);
		_mm512_store_ps(&(c->element[1][i]), tmp[1]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t tmp_0, tmp_1;

	for(i = 0; i < a->real_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(a->real_dim));
		//rds_neg(tmp, get_dsvector_i(a, i));
		//set_dsvector_i(c, i, tmp);
		tmp_0 = svneg_f32_x(pg, svld1_f32(pg, &(a->element[0][i])));
		tmp_1 = svneg_f32_x(pg, svld1_f32(pg, &(a->element[1][i])));
		svst1_f32(pg, &(c->element[0][i]), tmp_0);
		svst1_f32(pg, &(c->element[1][i]), tmp_1);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t tmp[DSSIZE];

	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//rds_neg(tmp, get_dsvector_i(a, i));
		//set_dsvector_i(c, i, tmp);
		tmp[0] = vnegq_f32(vld1q_f32(&(a->element[0][i])));
		tmp[1] = vnegq_f32(vld1q_f32(&(a->element[1][i])));
		vst1q_f32(&(c->element[0][i]), tmp[0]);
		vst1q_f32(&(c->element[1][i]), tmp[1]);
	}
#else // others
	float tmp[DSSIZE];

	for(i = 0; i < a->dim; i++)
	{
		rds_neg(tmp, get_dsvector_i(a, i));
		set_dsvector_i(c, i, tmp);
	}
#endif // __AVX2__
}


/* ||a||_1 */
void norm1_dsvector(float ret[DSSIZE], DSVector a)
{
	long int i, index, dim;

	dim = a->dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 vec4[DSSIZE], ret4[DSSIZE], tmp4[DSSIZE];

	_bncavx2_set0_ds(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		vec4[0] = _mm256_load_ps(&(a->element[0][index]));
		vec4[1] = _mm256_load_ps(&(a->element[1][index]));

		//rds_abs(tmp, get_dsvector_i(a, i));
		//rds_add(ret, ret, tmp);
		_bncavx2_rds_abs(tmp4, vec4);
		_bncavx2_rds_add(ret4, ret4, tmp4);
	}
	_bncavx2_rds_abssum256(ret, ret4);
#elif defined(__AVX512F__) // __AVX512F__
	__m512 vec4[DSSIZE], ret4[DSSIZE], tmp4[DSSIZE];

	_bncavx512_set0_ds(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		vec4[0] = _mm512_load_ps(&(a->element[0][index]));
		vec4[1] = _mm512_load_ps(&(a->element[1][index]));

		//rds_abs(tmp, get_dsvector_i(a, i));
		//rds_add(ret, ret, tmp);
		_bncavx512_rds_abs(tmp4, vec4);
		_bncavx512_rds_add(ret4, ret4, tmp4);
	}
	_bncavx512_rds_abssum512(ret, ret4);
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t vec4_0, vec4_1;
	svfloat32_t ret4_0, ret4_1;
	svfloat32_t tmp4_0, tmp4_1;

	_bncsve2_rds_set0(&ret4_0, &ret4_1);
	for(index = 0; index < a->real_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(a->real_dim));
		vec4_0 = svld1_f32(pg, &(a->element[0][index]));
		vec4_1 = svld1_f32(pg, &(a->element[1][index]));

		//rds_abs(tmp, get_dsvector_i(a, i));
		//rds_add(ret, ret, tmp);
		_bncsve2_rds_abs(svptrue_b32(), &tmp4_0, &tmp4_1, vec4_0, vec4_1);
		_bncsve2_rds_add(svptrue_b32(), &ret4_0, &ret4_1, ret4_0, ret4_1, tmp4_0, tmp4_1);
	}
	_bncsve2_rds_abssum128(svptrue_b32(), ret, ret4_0, ret4_1);
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t vec4[DSSIZE], ret4[DSSIZE], tmp4[DSSIZE];

	_bncneon_set0_ds(ret4);
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		vec4[0] = vld1q_f32(&(a->element[0][index]));
		vec4[1] = vld1q_f32(&(a->element[1][index]));

		//rds_abs(tmp, get_dsvector_i(a, i));
		//rds_add(ret, ret, tmp);
		_bncneon_rds_abs(tmp4, vec4);
		_bncneon_rds_add(ret4, ret4, tmp4);
	}
	_bncneon_rds_abssum128f(ret, ret4);
#else // others
	float tmp[DSSIZE];

	set0_ds(ret);
	for(i = 0; i < a->dim; i++)
	{
		rds_abs(tmp, get_dsvector_i(a, i));
		rds_add(ret, ret, tmp);
	}
#endif // __AVX2__

	return;
}

/* ||a||_infty */
void normi_dsvector(float ret[DSSIZE], DSVector a)
{
	float tmp[DSSIZE];
	long int i;

	rds_abs(ret, get_dsvector_i(a, 0));
	for(i = 1; i < a->dim; i++)
	{
		rds_abs(tmp, get_dsvector_i(a, i));
		if(rds_cmp(ret, tmp) < 0)
			rds_set(ret, tmp);
	}

	return;
}

// Euclid norm
void norm2_dsvector(float ret[DSSIZE], DSVector vec)
{
	long int i, index, dim;

	dim = vec->dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 vec4[DSSIZE], ret4[DSSIZE], tmp4[DSSIZE];

	_bncavx2_set0_ds(ret4);
	for(index = 0; index < vec->real_dim; index += _BNC_S_WIDTH)
	{
		vec4[0] = _mm256_load_ps(&(vec->element[0][index]));
		vec4[1] = _mm256_load_ps(&(vec->element[1][index]));

//		rds_mul(tmp, get_dsvector_i(vec, i), get_dsvector_i(vec, i));
//		rds_add(ret, ret, tmp);
		_bncavx2_rds_mul(tmp4, vec4, vec4);
		_bncavx2_rds_add(ret4, ret4, tmp4);
	}
	_bncavx2_rds_norm256(ret, ret4);
#elif defined(__AVX512F__) // __AVX512F__
	__m512 vec4[DSSIZE], ret4[DSSIZE], tmp4[DSSIZE];

	_bncavx512_set0_ds(ret4);
	for(index = 0; index < vec->real_dim; index += _BNC_S_WIDTH)
	{
		vec4[0] = _mm512_load_ps(&(vec->element[0][index]));
		vec4[1] = _mm512_load_ps(&(vec->element[1][index]));

//		rds_mul(tmp, get_dsvector_i(vec, i), get_dsvector_i(vec, i));
//		rds_add(ret, ret, tmp);
		_bncavx512_rds_mul(tmp4, vec4, vec4);
		_bncavx512_rds_add(ret4, ret4, tmp4);
	}
	_bncavx512_rds_norm512(ret, ret4);
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t vec4_0, vec4_1;
	svfloat32_t ret4_0, ret4_1;
	svfloat32_t tmp4_0, tmp4_1;

	_bncsve2_rds_set0(&ret4_0, &ret4_1);
	for(index = 0; index < vec->real_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(vec->real_dim));
		vec4_0 = svld1_f32(pg, &(vec->element[0][index]));
		vec4_1 = svld1_f32(pg, &(vec->element[1][index]));

//		rds_mul(tmp, get_dsvector_i(vec, i), get_dsvector_i(vec, i));
//		rds_add(ret, ret, tmp);
		_bncsve2_rds_mul(svptrue_b32(), &tmp4_0, &tmp4_1, vec4_0, vec4_1, vec4_0, vec4_1);
		_bncsve2_rds_add(svptrue_b32(), &ret4_0, &ret4_1, ret4_0, ret4_1, tmp4_0, tmp4_1);
	}
	_bncsve2_rds_norm128(svptrue_b32(), ret, ret4_0, ret4_1);
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t vec4[DSSIZE], ret4[DSSIZE], tmp4[DSSIZE];

	_bncneon_set0_ds(ret4);
	for(index = 0; index < vec->real_dim; index += _BNC_S_WIDTH)
	{
		vec4[0] = vld1q_f32(&(vec->element[0][index]));
		vec4[1] = vld1q_f32(&(vec->element[1][index]));

//		rds_mul(tmp, get_dsvector_i(vec, i), get_dsvector_i(vec, i));
//		rds_add(ret, ret, tmp);
		_bncneon_rds_mul(tmp4, vec4, vec4);
		_bncneon_rds_add(ret4, ret4, tmp4);
	}
	_bncneon_rds_norm128f(ret, ret4);
#else // others
	float tmp[DSSIZE];

	//c_ds_copy_d((float)0.0, tmp);
	//c_ds_copy_d((float)0.0, ret);
	rds_set0(tmp);
	rds_set0(ret);

	for(i = 0; i < dim ; i++)
	{
		//c_ds_sqr(GET_DSVECTOR_I(vec, i), tmp);
		//c_ds_add(tmp, ret, ret);
		rds_mul(tmp, get_dsvector_i(vec, i), get_dsvector_i(vec, i));
		rds_add(ret, ret, tmp);
	}

	//c_ds_sqrt(ret, tmp);
	//c_ds_copy(tmp, ret);
	rds_sqrt(tmp, ret);
	rds_set(ret, tmp);
#endif // __AVX2__
}


#if defined(__AVX2__)
void _bncavx2_mm256_function_for_rem(dsfloat ret[], int index, int rem, __m256 (* avx2_func)(__m256, __m256, __m256 *), float a[], float b[], int dim)
{
    __m256 in_a, in_b, s, e;
    float in_s[8], in_e[8];

    if((index + rem)>= dim)
    {
        fprintf(stderr, "ERROR: index + rem = %d + %d is larger than dim = %d\n", index, rem, dim);
        return;
    }
    if(rem >= 8)
    {
        fprintf(stderr, "ERROR: rem = %d is larger than 8\n", rem);
        return;
    }

    // rem != 0
    if(rem > 0)
    {
        //index = div * unit;
//        printf(" %d ", index);
        // load a, b, c to in_a, in_b, in_c
        if(rem == 1)
        {
            in_a = _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, a[index]);
            in_b = _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, b[index]);

            //s = two_sum(a, b, &e);
            //s = _bncavx2_ftwo_sum(in_a, in_b, &e);
            s = avx2_func(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_ps(in_s, s);
            _mm256_storeu_ps(in_e, e);
            ret[index].val[0] = in_s[0]; ret[index].val[1] = in_e[0];
        }
        else if(rem == 2)
        {
            in_a = _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, a[index + 1], a[index]);
            in_b = _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, b[index + 1], b[index]);

            //s = two_sum(a, b, &e);
            s = avx2_func(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_ps(in_s, s);
            _mm256_storeu_ps(in_e, e);
            ret[index    ].val[0] = in_s[0]; ret[index    ].val[1] = in_e[0];
            ret[index + 1].val[0] = in_s[1]; ret[index + 1].val[1] = in_e[1];
        }
        else if(rem == 3)
        {
            in_a = _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, b[index + 2], b[index + 1], b[index]);

            //s = two_sum(a, b, &e);
            s = avx2_func(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_ps(in_s, s);
            _mm256_storeu_ps(in_e, e);
            ret[index    ].val[0] = in_s[0]; ret[index    ].val[1] = in_e[0];
            ret[index + 1].val[0] = in_s[1]; ret[index + 1].val[1] = in_e[1];
            ret[index + 2].val[0] = in_s[2]; ret[index + 2].val[1] = in_e[2];
        }
        else if(rem == 4)
        {
            in_a = _mm256_set_ps(0.0, 0.0, 0.0, 0.0, a[index + 3], a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_ps(0.0, 0.0, 0.0, 0.0, b[index + 3], b[index + 2], b[index + 1], b[index]);

            //s = two_sum(a, b, &e);
            s = avx2_func(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_ps(in_s, s);
            _mm256_storeu_ps(in_e, e);
            ret[index    ].val[0] = in_s[0]; ret[index    ].val[1] = in_e[0];
            ret[index + 1].val[0] = in_s[1]; ret[index + 1].val[1] = in_e[1];
            ret[index + 2].val[0] = in_s[2]; ret[index + 2].val[1] = in_e[2];
            ret[index + 3].val[0] = in_s[3]; ret[index + 3].val[1] = in_e[3];
        }
        else if(rem == 5)
        {
            in_a = _mm256_set_ps(0.0, 0.0, 0.0, a[index + 4], a[index + 3], a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_ps(0.0, 0.0, 0.0, b[index + 4], b[index + 3], b[index + 2], b[index + 1], b[index]);

            //s = two_sum(a, b, &e);
            s = avx2_func(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_ps(in_s, s);
            _mm256_storeu_ps(in_e, e);
            ret[index    ].val[0] = in_s[0]; ret[index    ].val[1] = in_e[0];
            ret[index + 1].val[0] = in_s[1]; ret[index + 1].val[1] = in_e[1];
            ret[index + 2].val[0] = in_s[2]; ret[index + 2].val[1] = in_e[2];
            ret[index + 3].val[0] = in_s[3]; ret[index + 3].val[1] = in_e[3];
            ret[index + 4].val[0] = in_s[4]; ret[index + 4].val[1] = in_e[4];
        }
        else if(rem == 6)
        {
            in_a = _mm256_set_ps(0.0, 0.0, a[index + 5], a[index + 4], a[index + 3], a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_ps(0.0, 0.0, b[index + 5], b[index + 4], b[index + 3], b[index + 2], b[index + 1], b[index]);

            //s = two_sum(a, b, &e);
            s = avx2_func(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_ps(in_s, s);
            _mm256_storeu_ps(in_e, e);
            ret[index    ].val[0] = in_s[0]; ret[index    ].val[1] = in_e[0];
            ret[index + 1].val[0] = in_s[1]; ret[index + 1].val[1] = in_e[1];
            ret[index + 2].val[0] = in_s[2]; ret[index + 2].val[1] = in_e[2];
            ret[index + 3].val[0] = in_s[3]; ret[index + 3].val[1] = in_e[3];
            ret[index + 4].val[0] = in_s[4]; ret[index + 4].val[1] = in_e[4];
            ret[index + 5].val[0] = in_s[5]; ret[index + 5].val[1] = in_e[5];
        }
        else if(rem == 7)
        {
            in_a = _mm256_set_ps(0.0, a[index + 6], a[index + 5], a[index + 4], a[index + 3], a[index + 2], a[index + 1], a[index]);
            in_b = _mm256_set_ps(0.0, b[index + 6], b[index + 5], b[index + 4], b[index + 3], b[index + 2], b[index + 1], b[index]);

            //s = two_sum(a, b, &e);
            s = avx2_func(in_a, in_b, &e);

            //	return dd_real(s, e);
            //c[0] = s; c[1] = e;
            _mm256_storeu_ps(in_s, s);
            _mm256_storeu_ps(in_e, e);
            ret[index    ].val[0] = in_s[0]; ret[index    ].val[1] = in_e[0];
            ret[index + 1].val[0] = in_s[1]; ret[index + 1].val[1] = in_e[1];
            ret[index + 2].val[0] = in_s[2]; ret[index + 2].val[1] = in_e[2];
            ret[index + 3].val[0] = in_s[3]; ret[index + 3].val[1] = in_e[3];
            ret[index + 4].val[0] = in_s[4]; ret[index + 4].val[1] = in_e[4];
            ret[index + 5].val[0] = in_s[5]; ret[index + 5].val[1] = in_e[5];
            ret[index + 6].val[0] = in_s[6]; ret[index + 6].val[1] = in_e[6];
        }
    }
}

/* float-float = float + float */
void _bncavx2_dsadd_f_f(dsfloat ret[], float a[], float b[], int dim)
{
    int i, index, div, rem, unit = _BNC_S_WIDTH;// 4;
    __m256 in_ret, in_a, in_b, s, e;
    //float in_s[4], in_e[4];
    float in_s[8], in_e[8];


    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a = _mm256_load_ps(&a[index]);
        in_b = _mm256_load_ps(&b[index]);

       	//s = two_sum(a, b, &e);
        s = _bncavx2_ftwo_sum(in_a, in_b, &e);

        //	return dd_real(s, e);
    	//c[0] = s; c[1] = e;
        //_mm256_store_ps(&ret[index].val[0], s);
        //_mm256_store_ps(&ret[index].val[1], e);

        _mm256_storeu_ps(in_s, s);
        _mm256_storeu_ps(in_e, e);
        ret[index    ].val[0] = in_s[0]; ret[index    ].val[1] = in_e[0];
        ret[index + 1].val[0] = in_s[1]; ret[index + 1].val[1] = in_e[1];
        ret[index + 2].val[0] = in_s[2]; ret[index + 2].val[1] = in_e[2];
        ret[index + 3].val[0] = in_s[3]; ret[index + 3].val[1] = in_e[3];
        ret[index + 4].val[0] = in_s[4]; ret[index + 4].val[1] = in_e[4];
        ret[index + 5].val[0] = in_s[5]; ret[index + 5].val[1] = in_e[5];
        ret[index + 6].val[0] = in_s[6]; ret[index + 6].val[1] = in_e[6];
        ret[index + 7].val[0] = in_s[7]; ret[index + 7].val[1] = in_e[7];
    }
   //printf("set in_a, in_b, in_c\n");
    index = div * unit;

    _bncavx2_mm256_function_for_rem(ret, index, rem, _bncavx2_ftwo_sum, a, b, dim);

//    printf("\n");
}

/* float-float = float + float */
//void _bncavx2_ddvadd_d_d(dsvector *ret, float a[], float b[], int dim)
void _bncavx2_dsvadd_f_f(DSVector ret, float a[], float b[], int dim)
{
    int i, index, div, rem, unit = _BNC_S_WIDTH; //4;
    __m256 in_ret, in_a, in_b, s, e;
    float in_s[4], in_e[4];
	dsfloat ret_rem[8];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a = _mm256_load_ps(&a[index]);
        in_b = _mm256_load_ps(&b[index]);

       	//s = two_sum(a, b, &e);
        s = _bncavx2_ftwo_sum(in_a, in_b, &e);

        //	return dd_real(s, e);
    	//c[0] = s; c[1] = e;
        _mm256_store_ps(&ret->element[0][index], s);
        _mm256_store_ps(&ret->element[1][index], e);
        /*
        _mm256_storeu_ps(in_s, s);
        _mm256_storeu_ps(in_e, e);
        ret[index    ].val[0] = in_s[0]; ret[index    ].val[1] = in_e[0];
        ret[index + 1].val[0] = in_s[1]; ret[index + 1].val[1] = in_e[1];
        ret[index + 2].val[0] = in_s[2]; ret[index + 2].val[1] = in_e[2];
        ret[index + 3].val[0] = in_s[3]; ret[index + 3].val[1] = in_e[3];
        */
    }

  //printf("set in_a, in_b, in_c\n");
    index = div * unit;

    _bncavx2_mm256_function_for_rem(ret_rem, 0, rem, _bncavx2_ftwo_sum, a, b, dim);
	for(i = 0; i < rem; i++)
		set_dsvector_i(ret, index + i, ret_rem[i].val);
	
}

/* add */
void _bncavx2_dsvadd(DSVector ret, DSVector a, DSVector b, int dim)
{
	/* This one satisfies IEEE style error bound, 
		due to K. Briggs and W. Kahan.                   */
    int i, index, div, rem, unit = _BNC_S_WIDTH; //4;
    __m256 in_ret[2], in_a_val[2], in_b_val[2], s1, s2, t1, t2;
    float in_s1[4], in_s2[4], in_t1[4], in_t2[4];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_ps(&(a->element[1][index]));
        in_b_val[0] = _mm256_load_ps(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_ps(&(b->element[1][index]));

        _bncavx2_rds_add(in_ret, in_a_val, in_b_val);

        _mm256_store_ps(&ret->element[0][index], in_ret[0]);
        _mm256_store_ps(&ret->element[1][index], in_ret[1]);
   }
}


void _bncavx2_dsadd(dsfloat ret[], dsfloat a[], dsfloat b[], int dim)
{
	/* This one satisfies IEEE style error bound, 
		due to K. Briggs and W. Kahan.                   */
    int i, index, div, rem, unit = _BNC_S_WIDTH; //4;
    __m256 in_ret[2], in_a_val[2], in_b_val[2];

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

        _bncavx2_rds_add(in_ret, in_a_val, in_b_val);

        ret[index    ].val[0] = in_ret[0][0]; ret[index    ].val[1] = in_ret[1][0];
        ret[index + 1].val[0] = in_ret[0][1]; ret[index + 1].val[1] = in_ret[1][1];
        ret[index + 2].val[0] = in_ret[0][2]; ret[index + 2].val[1] = in_ret[1][2];
        ret[index + 3].val[0] = in_ret[0][3]; ret[index + 3].val[1] = in_ret[1][3];
        ret[index + 4].val[0] = in_ret[0][4]; ret[index + 4].val[1] = in_ret[1][4];
        ret[index + 5].val[0] = in_ret[0][5]; ret[index + 5].val[1] = in_ret[1][5];
        ret[index + 6].val[0] = in_ret[0][6]; ret[index + 6].val[1] = in_ret[1][6];
        ret[index + 7].val[0] = in_ret[0][7]; ret[index + 7].val[1] = in_ret[1][7];
   }

}

/* float-float = float * float */
void _bncavx2_dsvmul_f_f(DSVector ret, float a[], float b[], int dim)
{
    int i, index, div, rem, unit = _BNC_S_WIDTH; //4;
    __m256 in_ret, in_a, in_b, s, e;
    float in_s[4], in_e[4];
	dsfloat ret_rem[8];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a = _mm256_load_ps(&a[index]);
        in_b = _mm256_load_ps(&b[index]);

       	//s = two_prod(a, b, &e);
        s = _bncavx2_ftwo_prod(in_a, in_b, &e);

        //	return dd_real(s, e);
    	//c[0] = s; c[1] = e;
        _mm256_store_ps(&ret->element[0][index], s);
        _mm256_store_ps(&ret->element[1][index], e);
        /*
        _mm256_storeu_ps(in_s, s);
        _mm256_storeu_ps(in_e, e);
        ret[index    ].val[0] = in_s[0]; ret[index    ].val[1] = in_e[0];
        ret[index + 1].val[0] = in_s[1]; ret[index + 1].val[1] = in_e[1];
        ret[index + 2].val[0] = in_s[2]; ret[index + 2].val[1] = in_e[2];
        ret[index + 3].val[0] = in_s[3]; ret[index + 3].val[1] = in_e[3];
        */
    }

   //printf("set in_a, in_b, in_c\n");
    //_bncavx2_mm256_function_for_rem(ret, index, rem, _bncavx2_ftwo_prod, a, b, dim);
    _bncavx2_mm256_function_for_rem(ret_rem, 0, rem, _bncavx2_ftwo_sum, a, b, dim);
	for(i = 0; i < rem; i++)
		set_dsvector_i(ret, index + i, ret_rem[i].val);

//    printf("\n");
}

/* ddmul */
void _bncavx2_dsmul(dsfloat ret[], dsfloat a[], dsfloat b[], int dim)
{
	/* This one satisfies IEEE style error bound, 
		due to K. Briggs and W. Kahan.                   */
    int i, index, div, rem, unit = _BNC_S_WIDTH; //4;
    __m256 in_ret[2], in_a_val[2], in_b_val[2];

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

        _bncavx2_rds_mul(in_ret, in_a_val, in_b_val);

        ret[index    ].val[0] = in_ret[0][0]; ret[index    ].val[1] = in_ret[1][0];
        ret[index + 1].val[0] = in_ret[0][1]; ret[index + 1].val[1] = in_ret[1][1];
        ret[index + 2].val[0] = in_ret[0][2]; ret[index + 2].val[1] = in_ret[1][2];
        ret[index + 3].val[0] = in_ret[0][3]; ret[index + 3].val[1] = in_ret[1][3];
        ret[index + 4].val[0] = in_ret[0][4]; ret[index + 4].val[1] = in_ret[1][4];
        ret[index + 5].val[0] = in_ret[0][5]; ret[index + 5].val[1] = in_ret[1][5];
        ret[index + 6].val[0] = in_ret[0][6]; ret[index + 6].val[1] = in_ret[1][6];
        ret[index + 7].val[0] = in_ret[0][7]; ret[index + 7].val[1] = in_ret[1][7];
    }

}

/* ddvmul */
void _bncavx2_dsvmul(DSVector ret, DSVector a, DSVector b, int dim)
{
    int i, index, div, rem, unit = _BNC_S_WIDTH; //4;
    __m256 in_ret[2], in_a_val[2], in_b_val[2];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_ps(&(a->element[1][index]));
        in_b_val[0] = _mm256_load_ps(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_ps(&(b->element[1][index]));

        _bncavx2_rds_mul(in_ret, in_a_val, in_b_val);

        _mm256_store_ps(&ret->element[0][index], in_ret[0]);
        _mm256_store_ps(&ret->element[1][index], in_ret[1]);
   }
}
/* dddiv */
void _bncavx2_dsdiv(dsfloat ret[], dsfloat a[], dsfloat b[], int dim)
{
	/* This one satisfies IEEE style error bound, 
		due to K. Briggs and W. Kahan.                   */
    int i, index, div, rem, unit = _BNC_S_WIDTH; //4;
    __m256 in_ret[2], in_a_val[2], in_b_val[2];

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

        _bncavx2_rds_div(in_ret, in_a_val, in_b_val);

        ret[index    ].val[0] = in_ret[0][0]; ret[index    ].val[1] = in_ret[1][0];
        ret[index + 1].val[0] = in_ret[0][1]; ret[index + 1].val[1] = in_ret[1][1];
        ret[index + 2].val[0] = in_ret[0][2]; ret[index + 2].val[1] = in_ret[1][2];
        ret[index + 3].val[0] = in_ret[0][3]; ret[index + 3].val[1] = in_ret[1][3];
        ret[index + 4].val[0] = in_ret[0][4]; ret[index + 4].val[1] = in_ret[1][4];
        ret[index + 5].val[0] = in_ret[0][5]; ret[index + 5].val[1] = in_ret[1][5];
        ret[index + 6].val[0] = in_ret[0][6]; ret[index + 6].val[1] = in_ret[1][6];
        ret[index + 7].val[0] = in_ret[0][7]; ret[index + 7].val[1] = in_ret[1][7];
   }

}

/* div */
void _bncavx2_dsvdiv(DSVector ret, DSVector a, DSVector b, int dim)
{
    int i, index, div, rem, unit = _BNC_S_WIDTH; //4;
    __m256 in_ret[2], in_a_val[2], in_b_val[2];

    div = dim / unit;
    rem = dim % unit;

    //for(i = 0; i < div; i++)
    for(index = 0; index < dim; index += unit)
    {
        in_a_val[0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[1] = _mm256_load_ps(&(a->element[1][index]));
        in_b_val[0] = _mm256_load_ps(&(b->element[0][index]));
        in_b_val[1] = _mm256_load_ps(&(b->element[1][index]));

        _bncavx2_rds_div(in_ret, in_a_val, in_b_val);

        _mm256_store_ps(&ret->element[0][index], in_ret[0]);
        _mm256_store_ps(&ret->element[1][index], in_ret[1]);
   }
}

#endif // __AVX2__

// set a zero matrix
//void set0_dsmatrix(DSMatrix mat)
void set0_dsmatrix(DSMatrix mat)
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
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero4;

	zero4 = _mm512_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm512_store_ps(&mat->element[0][i], zero4);
		_mm512_store_ps(&mat->element[1][i], zero4);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t zero4;

	zero4 = svdup_f32(0.0f);
	for(i = 0; i < real_total_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(real_total_dim));
		svst1_f32(pg, &mat->element[0][i], zero4);
		svst1_f32(pg, &mat->element[1][i], zero4);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t zero4;

	zero4 = vdupq_n_f32(0.0f);
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		vst1q_f32(&mat->element[0][i], zero4);
		vst1q_f32(&mat->element[1][i], zero4);
	}
#else // others
	for(i = 0; i < real_total_dim; i++)
	{
		mat->element[0][i] = 0.0;
		mat->element[1][i] = 0.0;	
	}
#endif // __AVX2__
}

// initialize dsvector
DSMatrix init_dsmatrix(long int row_dim, long int col_dim)
{
	long int row_index, col_index, i;
	long int real_row_dim, real_col_dim, real_total_dim;
	DSMatrix ret = NULL;

	ret = (DSMatrix)BNC_MALLOC(sizeof(dsmatrix));
	if(ret == NULL)
		return ret;

	// real_dim is the nearest positive multiplier of _BNC_S_WIDTH
	real_row_dim = (long int)ceil((float)(row_dim) / (float)_BNC_S_WIDTH) * _BNC_S_WIDTH;
	real_col_dim = (long int)ceil((float)(col_dim) / (float)_BNC_S_WIDTH) * _BNC_S_WIDTH;
	real_total_dim = real_row_dim * real_col_dim;

	//printf("init_dsmatrix(%ld, %ld) %ld calloc\n", row_dim, col_dim, real_total_dim);
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

	//printf("init_dsmatrix(%ld, %ld) calloc\n", row_dim, col_dim);
	/* All 0 */
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 zero4;

	zero4 = _mm256_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm256_store_ps(&ret->element[0][i], zero4);
		_mm256_store_ps(&ret->element[1][i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero4;

	zero4 = _mm512_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm512_store_ps(&ret->element[0][i], zero4);
		_mm512_store_ps(&ret->element[1][i], zero4);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t zero4;

	zero4 = svdup_f32(0.0f);
	for(i = 0; i < real_total_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(real_total_dim));
		svst1_f32(pg, &ret->element[0][i], zero4);
		svst1_f32(pg, &ret->element[1][i], zero4);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t zero4;

	zero4 = vdupq_n_f32(0.0f);
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		vst1q_f32(&ret->element[0][i], zero4);
		vst1q_f32(&ret->element[1][i], zero4);
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

// free dsvector
void free_dsmatrix(DSMatrix mat)
{
	long int i;

	for(i = 0; i < DSSIZE; i++)
		free(mat->element[i]);

	free(mat);
}

// print dsvector
void print_dsmatrix(DSMatrix mat)
{
	long int row_index, col_index;

	for(row_index = 0; row_index < mat->row_dim; row_index++)
	{
		for(col_index = 0; col_index < mat->col_dim; col_index++)
		{
			printf("%ld, %ld: ", row_index, col_index);
	//		c_ds_write((vec->element + index * DSSIZE));
			c_ds_write(GET_DSMATRIX_IJ(mat, row_index, col_index));
		}
	}
}

// DSMatrix mat -> dsfloat array
void set_dsfloat_dsmat(dsfloat ret[], int ret_dim, DSMatrix mat)
{
    int total_index, j, total_dim;
	long int row_index, col_index;

    total_dim = (ret_dim < (mat->row_dim * mat->col_dim)) ? ret_dim : (mat->row_dim * mat->col_dim);

	total_index = 0;
    for(row_index = 0; row_index < mat->row_dim; row_index++)
    {
		for(col_index = 0; (col_index < mat->col_dim) && (total_index < total_dim); col_index++)
		{
			for(j = 0; j < DSSIZE; j++)
				ret[total_index].val[j] = mat->element[j][(row_index * mat->real_col_dim) + col_index];

			total_index++;
		}
    }
}

// dsfloat array -> DDmatrix ret
void set_dsmatrix_dsfloat(DSMatrix ret, dsfloat array[], int array_dim)
{
    int total_index, j, total_dim;
	long int row_index, col_index;

    total_dim = (array_dim < (ret->row_dim * ret->col_dim)) ? array_dim : (ret->row_dim * ret->col_dim);

 	total_index = 0;
    for(row_index = 0; row_index < ret->row_dim; row_index++)
    {
		for(col_index = 0; (col_index < ret->col_dim) && (total_index < total_dim); col_index++)
		{
			for(j = 0; j < DSSIZE; j++)
				ret->element[j][(row_index * ret->real_col_dim) + col_index] = array[total_index].val[j];

			total_index++;
		}
    }
}


// matrix multiplication
// ret := A * B
void mul_dsmatrix(DSMatrix ret, DSMatrix a, DSMatrix b)
{
	long int i, j, k;

	/* dimension check */
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: mul_dsmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret->row_dim, ret->col_dim, a->row_dim, a->col_dim, b->row_dim, b->col_dim);
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long real_row_dim, real_col_dim, real_mid_dim;
	//float cijval[4][DSSIZE];
	float cijval[8][DSSIZE];
    __m256 cij[DSSIZE], aik[DSSIZE], bkj[DSSIZE], tmp_mul[DSSIZE];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j++)
        {
            //rds_set_ui(cij.val, 0UL);
            cij[0] = _mm256_setzero_ps();
            cij[1] = _mm256_setzero_ps();
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

            /*
                rds_mul(tmp_mul[0].val, aik[0].val, bkj[0].val);
                rds_mul(tmp_mul[1].val, aik[1].val, bkj[1].val);
                rds_mul(tmp_mul[2].val, aik[2].val, bkj[2].val);
                rds_mul(tmp_mul[3].val, aik[3].val, bkj[3].val);
            */
                _bncavx2_rds_mul(tmp_mul, aik, bkj);

            /*
                rds_add(cij.val, cij.val, tmp_mul[0].val);
                rds_add(cij.val, cij.val, tmp_mul[1].val);
                rds_add(cij.val, cij.val, tmp_mul[2].val);
                rds_add(cij.val, cij.val, tmp_mul[3].val);
            */
                _bncavx2_rds_add(cij, cij, tmp_mul);
            }

            cijval[0][0] = cij[0][0]; cijval[0][1] = cij[1][0];
            cijval[1][0] = cij[0][1]; cijval[1][1] = cij[1][1];
            cijval[2][0] = cij[0][2]; cijval[2][1] = cij[1][2];
            cijval[3][0] = cij[0][3]; cijval[3][1] = cij[1][3];
            cijval[4][0] = cij[0][4]; cijval[4][1] = cij[1][4];
            cijval[5][0] = cij[0][5]; cijval[5][1] = cij[1][5];
            cijval[6][0] = cij[0][6]; cijval[6][1] = cij[1][6];
            cijval[7][0] = cij[0][7]; cijval[7][1] = cij[1][7];
            rds_add(cijval[0], cijval[0], cijval[1]);
            rds_add(cijval[0], cijval[0], cijval[2]);
            rds_add(cijval[0], cijval[0], cijval[3]);
            rds_add(cijval[0], cijval[0], cijval[4]);
            rds_add(cijval[0], cijval[0], cijval[5]);
            rds_add(cijval[0], cijval[0], cijval[6]);
            rds_add(cijval[0], cijval[0], cijval[7]);

            ret->element[0][i * real_col_dim + j] = cijval[0][0];
            ret->element[1][i * real_col_dim + j] = cijval[0][1];
        }
    }
#elif defined(__AVX512F__) // __AVX512F__
	long real_row_dim, real_col_dim, real_mid_dim;
	//float cijval[4][DSSIZE];
	float cijval[16][DSSIZE];
    __m512 cij[DSSIZE], aik[DSSIZE], bkj[DSSIZE], tmp_mul[DSSIZE];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j++)
        {
            //rds_set_ui(cij.val, 0UL);
            cij[0] = _mm512_setzero_ps();
            cij[1] = _mm512_setzero_ps();
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

            /*
                rds_mul(tmp_mul[0].val, aik[0].val, bkj[0].val);
                rds_mul(tmp_mul[1].val, aik[1].val, bkj[1].val);
                rds_mul(tmp_mul[2].val, aik[2].val, bkj[2].val);
                rds_mul(tmp_mul[3].val, aik[3].val, bkj[3].val);
            */
                _bncavx512_rds_mul(tmp_mul, aik, bkj);

            /*
                rds_add(cij.val, cij.val, tmp_mul[0].val);
                rds_add(cij.val, cij.val, tmp_mul[1].val);
                rds_add(cij.val, cij.val, tmp_mul[2].val);
                rds_add(cij.val, cij.val, tmp_mul[3].val);
            */
                _bncavx512_rds_add(cij, cij, tmp_mul);
            }

            cijval[0][0] = cij[0][0]; cijval[0][1] = cij[1][0];
            cijval[1][0] = cij[0][1]; cijval[1][1] = cij[1][1];
            cijval[2][0] = cij[0][2]; cijval[2][1] = cij[1][2];
            cijval[3][0] = cij[0][3]; cijval[3][1] = cij[1][3];
            cijval[4][0] = cij[0][4]; cijval[4][1] = cij[1][4];
            cijval[5][0] = cij[0][5]; cijval[5][1] = cij[1][5];
            cijval[6][0] = cij[0][6]; cijval[6][1] = cij[1][6];
            cijval[7][0] = cij[0][7]; cijval[7][1] = cij[1][7];
            cijval[8][0] = cij[0][8]; cijval[8][1] = cij[1][8];
            cijval[9][0] = cij[0][9]; cijval[9][1] = cij[1][9];
            cijval[10][0] = cij[0][10]; cijval[10][1] = cij[1][10];
            cijval[11][0] = cij[0][11]; cijval[11][1] = cij[1][11];
            cijval[12][0] = cij[0][12]; cijval[12][1] = cij[1][12];
            cijval[13][0] = cij[0][13]; cijval[13][1] = cij[1][13];
            cijval[14][0] = cij[0][14]; cijval[14][1] = cij[1][14];
            cijval[15][0] = cij[0][15]; cijval[15][1] = cij[1][15];
            rds_add(cijval[0], cijval[0], cijval[1]);
            rds_add(cijval[0], cijval[0], cijval[2]);
            rds_add(cijval[0], cijval[0], cijval[3]);
            rds_add(cijval[0], cijval[0], cijval[4]);
            rds_add(cijval[0], cijval[0], cijval[5]);
            rds_add(cijval[0], cijval[0], cijval[6]);
            rds_add(cijval[0], cijval[0], cijval[7]);
            rds_add(cijval[0], cijval[0], cijval[8]);
            rds_add(cijval[0], cijval[0], cijval[9]);
            rds_add(cijval[0], cijval[0], cijval[10]);
            rds_add(cijval[0], cijval[0], cijval[11]);
            rds_add(cijval[0], cijval[0], cijval[12]);
            rds_add(cijval[0], cijval[0], cijval[13]);
            rds_add(cijval[0], cijval[0], cijval[14]);
            rds_add(cijval[0], cijval[0], cijval[15]);

            ret->element[0][i * real_col_dim + j] = cijval[0][0];
            ret->element[1][i * real_col_dim + j] = cijval[0][1];
        }
    }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic)
	{
		long real_row_dim = a->real_row_dim;
		long real_col_dim = b->real_col_dim;
		long real_mid_dim = a->real_col_dim;
		long vl = (long)svcntw();   /* float lanes per vector (VL-agnostic) */

		for(i = 0; i < real_row_dim; i++){
			for(j = 0; j < real_col_dim; j += vl){
				svbool_t pg = svwhilelt_b32_s32((int32_t)j, (int32_t)real_col_dim);
				svfloat32_t cij0, cij1;
				_bncsve2_rds_set0(&cij0, &cij1);
				for(k = 0; k < real_mid_dim; k++){
					svfloat32_t aik0 = svdup_n_f32(a->element[0][i*real_mid_dim + k]);
					svfloat32_t aik1 = svdup_n_f32(a->element[1][i*real_mid_dim + k]);
					svfloat32_t bkj0 = svld1_f32(pg, &(b->element[0][k*real_col_dim + j]));
					svfloat32_t bkj1 = svld1_f32(pg, &(b->element[1][k*real_col_dim + j]));
					svfloat32_t t0, t1;
					_bncsve2_rds_mul(pg, &t0, &t1, aik0, aik1, bkj0, bkj1);
					_bncsve2_rds_add(pg, &cij0, &cij1, cij0, cij1, t0, t1);
				}
				svst1_f32(pg, &(ret->element[0][i*real_col_dim + j]), cij0);
				svst1_f32(pg, &(ret->element[1][i*real_col_dim + j]), cij1);
			}
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	long real_row_dim = a->real_row_dim;
	long real_col_dim = b->real_col_dim;
	long real_mid_dim = a->real_col_dim;
	float32x4_t cij[DSSIZE], aik[DSSIZE], bkj[DSSIZE], tmp[DSSIZE];

	for(i = 0; i < real_row_dim; i += 1){
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH){
			_bncneon_set0_ds(cij);
			for(k = 0; k < real_mid_dim; k += 1){
				aik[0] = vdupq_n_f32(a->element[0][i*real_mid_dim + k]);
				aik[1] = vdupq_n_f32(a->element[1][i*real_mid_dim + k]);
				bkj[0] = vld1q_f32(&(b->element[0][k*real_col_dim + j]));
				bkj[1] = vld1q_f32(&(b->element[1][k*real_col_dim + j]));
				_bncneon_rds_mul(tmp, aik, bkj);
				_bncneon_rds_add(cij, cij, tmp);
			}
			vst1q_f32(&(ret->element[0][i*real_col_dim + j]), cij[0]);
			vst1q_f32(&(ret->element[1][i*real_col_dim + j]), cij[1]);
		}
	}
#else // __AVX2__
	long row_dim, col_dim, mid_dim;
	float tmp[DSSIZE], ret_ij[DSSIZE];

	//printf("Non SIMD mul_dsmatrix(%ld, %ld)\n", ret->row_dim, ret->col_dim);
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = a->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			//rds_set0(GET_DSMATRIX_IJ(ret, i, j));
			rds_set0(ret_ij);
			for(k = 0; k < mid_dim; k++)
			{
				rds_mul(tmp, GET_DSMATRIX_IJ(a, i, k), GET_DSMATRIX_IJ(b, k, j));
				//rds_add(GET_DSMATRIX_IJ(ret, i, j), tmp, GET_DSMATRIX_IJ(ret, i, j));
				rds_add(ret_ij, tmp, ret_ij);
			}
			set_dsmatrix_ij(ret, i, j, ret_ij);
		}
	}
	//printf("end(%ld, %ld)\n", ret->row_dim, ret->col_dim);
#endif // __AVX2__

}

// Frobenius norm
void normf_dsmatrix(float ret[DSSIZE], DSMatrix mat)
{
	long int i;
	long int real_total_dim;
	float tmp[DSSIZE];

	real_total_dim = mat->real_row_dim * mat->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 ret4[DSSIZE], mat4[DSSIZE], tmp4[DSSIZE];

	ret4[0] = _mm256_setzero_ps();
	ret4[1] = _mm256_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		mat4[0] = _mm256_load_ps(&(mat->element[0][i]));
		mat4[1] = _mm256_load_ps(&(mat->element[1][i]));

		// tmp4 := mat4[i]^2
		// ret4 += tmp4
		_bncavx2_rds_mul(tmp4, mat4, mat4);
		_bncavx2_rds_add(ret4, ret4, tmp4);
	}

	_bncavx2_rds_sum256(ret, ret4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512 ret4[DSSIZE], mat4[DSSIZE], tmp4[DSSIZE];

	ret4[0] = _mm512_setzero_ps();
	ret4[1] = _mm512_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		mat4[0] = _mm512_load_ps(&(mat->element[0][i]));
		mat4[1] = _mm512_load_ps(&(mat->element[1][i]));

		// tmp4 := mat4[i]^2
		// ret4 += tmp4
		_bncavx512_rds_mul(tmp4, mat4, mat4);
		_bncavx512_rds_add(ret4, ret4, tmp4);
	}

	_bncavx512_rds_sum512(ret, ret4);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t ret4_0, ret4_1;
	svfloat32_t mat4_0, mat4_1;
	svfloat32_t tmp4_0, tmp4_1;

	ret4_0 = svdup_f32(0.0f);
	ret4_1 = svdup_f32(0.0f);
	for(i = 0; i < real_total_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(real_total_dim));
		mat4_0 = svld1_f32(pg, &(mat->element[0][i]));
		mat4_1 = svld1_f32(pg, &(mat->element[1][i]));

		// tmp4 := mat4[i]^2
		// ret4 += tmp4
		_bncsve2_rds_mul(svptrue_b32(), &tmp4_0, &tmp4_1, mat4_0, mat4_1, mat4_0, mat4_1);
		_bncsve2_rds_add(svptrue_b32(), &ret4_0, &ret4_1, ret4_0, ret4_1, tmp4_0, tmp4_1);
	}

	_bncsve2_rds_sum128(svptrue_b32(), ret, ret4_0, ret4_1);

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t ret4[DSSIZE], mat4[DSSIZE], tmp4[DSSIZE];

	ret4[0] = vdupq_n_f32(0.0f);
	ret4[1] = vdupq_n_f32(0.0f);
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		mat4[0] = vld1q_f32(&(mat->element[0][i]));
		mat4[1] = vld1q_f32(&(mat->element[1][i]));

		// tmp4 := mat4[i]^2
		// ret4 += tmp4
		_bncneon_rds_mul(tmp4, mat4, mat4);
		_bncneon_rds_add(ret4, ret4, tmp4);
	}

	_bncneon_rds_sum128f(ret, ret4);

#else // others
	float mat1[DSSIZE];

	rds_set0(ret);
	for(i = 0; i < real_total_dim; i++)
	{
		mat1[0] = mat->element[0][i];
		mat1[1] = mat->element[1][i];

		// tmp := mat1[i]^2
		// ret += tmp
		rds_mul(tmp, mat1, mat1);
		rds_add(ret, ret, tmp);
	}

#endif // __AVX2__

	rds_sqrt(tmp, ret);
	rds_set(ret, tmp);

}

// print normf
void print_normf_dsmatrix(const char *str, DSMatrix mat)
{
	static float tmp[DSSIZE];

	normf_dsmatrix(tmp, mat);

	if(str != NULL)
		printf("%s(%ld, %ld)", str, mat->row_dim, mat->col_dim);

	rds_out_str(tmp); printf("\n");
}

/*************************************************/
/* Matrix Caluculations for DSMatrix            */
/*
void normf_dsmatrix(float ret[DSSIZE], DSMatrix mat)
void norm1_dsmatrix(float ret[DSSIZE], DSMatrix mat)
void normi_dsmatrix(float ret[DSSIZE], DSMatrix mat)
void add_dsmatrix(DSMatrix c, DSMatrix a, DSMatrix b);
void sub_dsmatrix(DSMatrix c, DSMatrix a, DSMatrix b);
void mul_dsmatrix(DSMatrix c, DSMatrix a, DSMatrix b);
void mul_dsmatrix_dsvec(DSVector v, DSMatrix a, DSVector vb)
void mul_dsmatrixt_dsvec(DSVector v, DSMatrix a, DSVector vb)
void transpose_dsmatrix(DSMatrix c, DSMatrix a);
void inv_dsmatrix(DSMatrix a);
void subst_mpfmatrux(DSMatrix c, DSMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_dsmatrix(float ret[DSSIZE], DSMatrix mat)
{
	long int i, j;
	float tmp[DSSIZE], sum[DSSIZE];

	set0_ds(ret);
	for(i = 0; i < mat->row_dim; i++)
	{
		set0_ds(sum);
		for(j = 0; j < mat->col_dim; j++)
		{
			rds_abs(tmp, get_dsmatrix_ij(mat, i, j));
			rds_add(sum, sum, tmp);
		}
		if(rds_cmp(ret, sum) < 0)
			rds_set(ret, sum);
	}

	return;
}

/* 1 Norm of Matrix */
void norm1_dsmatrix(float ret[DSSIZE], DSMatrix mat)
{
	long int i, j;
	float tmp[DSSIZE], sum[DSSIZE];

	rds_set_ui(ret, 0UL);

	for(j = 0; j < mat->col_dim; j++)
	{
		rds_set_ui(sum, 0UL);
		for(i = 0; i < mat->row_dim; i++)
		{
			rds_abs(tmp, get_dsmatrix_ij(mat, i, j));
			rds_add(sum, sum, tmp);
		}
		if(rds_cmp(ret, sum) < 0)
			rds_set(ret, sum);
	}

	return;
}

/* c := a + b */
void add_dsmatrix(DSMatrix c, DSMatrix a, DSMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_dsmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_dsmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 tmp4[DSSIZE], aij4[DSSIZE], bij4[DSSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = _mm256_load_ps(&(a->element[0][index]));
		aij4[1] = _mm256_load_ps(&(a->element[1][index]));
		bij4[0] = _mm256_load_ps(&(b->element[0][index]));
		bij4[1] = _mm256_load_ps(&(b->element[1][index]));

		_bncavx2_rds_add(tmp4, aij4, bij4);

		_mm256_store_ps(&(c->element[0][index]), tmp4[0]);
		_mm256_store_ps(&(c->element[1][index]), tmp4[1]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 tmp4[DSSIZE], aij4[DSSIZE], bij4[DSSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = _mm512_load_ps(&(a->element[0][index]));
		aij4[1] = _mm512_load_ps(&(a->element[1][index]));
		bij4[0] = _mm512_load_ps(&(b->element[0][index]));
		bij4[1] = _mm512_load_ps(&(b->element[1][index]));

		_bncavx512_rds_add(tmp4, aij4, bij4);

		_mm512_store_ps(&(c->element[0][index]), tmp4[0]);
		_mm512_store_ps(&(c->element[1][index]), tmp4[1]); 
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t tmp4_0, tmp4_1;
	svfloat32_t aij4_0, aij4_1;
	svfloat32_t bij4_0, bij4_1;

	for(index = 0; index < real_total_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(real_total_dim));
		aij4_0 = svld1_f32(pg, &(a->element[0][index]));
		aij4_1 = svld1_f32(pg, &(a->element[1][index]));
		bij4_0 = svld1_f32(pg, &(b->element[0][index]));
		bij4_1 = svld1_f32(pg, &(b->element[1][index]));

		_bncsve2_rds_add(pg, &tmp4_0, &tmp4_1, aij4_0, aij4_1, bij4_0, bij4_1);

		svst1_f32(pg, &(c->element[0][index]), tmp4_0);
		svst1_f32(pg, &(c->element[1][index]), tmp4_1); 
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t tmp4[DSSIZE], aij4[DSSIZE], bij4[DSSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = vld1q_f32(&(a->element[0][index]));
		aij4[1] = vld1q_f32(&(a->element[1][index]));
		bij4[0] = vld1q_f32(&(b->element[0][index]));
		bij4[1] = vld1q_f32(&(b->element[1][index]));

		_bncneon_rds_add(tmp4, aij4, bij4);

		vst1q_f32(&(c->element[0][index]), tmp4[0]);
		vst1q_f32(&(c->element[1][index]), tmp4[1]); 
	}
#else // others
	float tmp[DSSIZE], aij[DSSIZE], bij[DSSIZE];

	for(index = 0; index < real_total_dim; index++)
	{
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];
		bij[0] = b->element[0][index];
		bij[1] = b->element[1][index];

		rds_add(tmp, aij, bij);

		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1]; 
	}
#endif // __AVX2__
/*
	float tmp[DSSIZE];

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rds_add(tmp, get_dsmatrix_ij(a, i, j), get_dsmatrix_ij(b, i, j));
			set_dsmatrix_ij(c, i, j, tmp);
		}
	}
*/
}

/* c := a - b */
void sub_dsmatrix(DSMatrix c, DSMatrix a, DSMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_dsmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_dsmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

/*
	float tmp[DSSIZE];
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rds_sub(tmp, get_dsmatrix_ij(a, i, j), get_dsmatrix_ij(b, i, j));
			set_dsmatrix_ij(c, i, j, tmp);
		}
	}
*/

// for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 tmp4[DSSIZE], aij4[DSSIZE], bij4[DSSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = _mm256_load_ps(&(a->element[0][index]));
		aij4[1] = _mm256_load_ps(&(a->element[1][index]));
		bij4[0] = _mm256_load_ps(&(b->element[0][index]));
		bij4[1] = _mm256_load_ps(&(b->element[1][index]));

		_bncavx2_rds_sub(tmp4, aij4, bij4);

		_mm256_store_ps(&(c->element[0][index]), tmp4[0]);
		_mm256_store_ps(&(c->element[1][index]), tmp4[1]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 tmp4[DSSIZE], aij4[DSSIZE], bij4[DSSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = _mm512_load_ps(&(a->element[0][index]));
		aij4[1] = _mm512_load_ps(&(a->element[1][index]));
		bij4[0] = _mm512_load_ps(&(b->element[0][index]));
		bij4[1] = _mm512_load_ps(&(b->element[1][index]));

		_bncavx512_rds_sub(tmp4, aij4, bij4);

		_mm512_store_ps(&(c->element[0][index]), tmp4[0]);
		_mm512_store_ps(&(c->element[1][index]), tmp4[1]); 
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t tmp4_0, tmp4_1;
	svfloat32_t aij4_0, aij4_1;
	svfloat32_t bij4_0, bij4_1;

	for(index = 0; index < real_total_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(real_total_dim));
		aij4_0 = svld1_f32(pg, &(a->element[0][index]));
		aij4_1 = svld1_f32(pg, &(a->element[1][index]));
		bij4_0 = svld1_f32(pg, &(b->element[0][index]));
		bij4_1 = svld1_f32(pg, &(b->element[1][index]));

		_bncsve2_rds_neg(pg, &tmp4_0, &tmp4_1, bij4_0, bij4_1);
		_bncsve2_rds_add(pg, &tmp4_0, &tmp4_1, aij4_0, aij4_1, tmp4_0, tmp4_1);

		svst1_f32(pg, &(c->element[0][index]), tmp4_0);
		svst1_f32(pg, &(c->element[1][index]), tmp4_1); 
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t tmp4[DSSIZE], aij4[DSSIZE], bij4[DSSIZE];

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = vld1q_f32(&(a->element[0][index]));
		aij4[1] = vld1q_f32(&(a->element[1][index]));
		bij4[0] = vld1q_f32(&(b->element[0][index]));
		bij4[1] = vld1q_f32(&(b->element[1][index]));

		_bncneon_rds_sub(tmp4, aij4, bij4);

		vst1q_f32(&(c->element[0][index]), tmp4[0]);
		vst1q_f32(&(c->element[1][index]), tmp4[1]); 
	}
#else // others
	float tmp[DSSIZE], aij[DSSIZE], bij[DSSIZE];

	for(index = 0; index < real_total_dim; index++)
	{
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];
		bij[0] = b->element[0][index];
		bij[1] = b->element[1][index];

		rds_sub(tmp, aij, bij);

		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1]; 
	}
#endif // __AVX2__
}

/* c := sc * a */
void cmul_dsmatrix(DSMatrix c, float sc[DSSIZE], DSMatrix a)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_dsmatrix(row_dim is differnt! : c->(%ld, %ld), a->(%ld, %ld)\n", c->row_dim, c->col_dim, a->row_dim, a->col_dim);
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_dsmatrix(col_dim is differnt! : c->(%ld, %ld), a->(%ld, %ld)\n", c->row_dim, c->col_dim, a->row_dim, a->col_dim);
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

// for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 tmp4[DSSIZE], sc4[DSSIZE], aij4[DSSIZE];

	sc4[0] = _mm256_set1_ps(sc[0]);
	sc4[1] = _mm256_set1_ps(sc[1]);

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = _mm256_load_ps(&(a->element[0][index]));
		aij4[1] = _mm256_load_ps(&(a->element[1][index]));

		_bncavx2_rds_mul(tmp4, sc4, aij4);

		_mm256_store_ps(&(c->element[0][index]), tmp4[0]);
		_mm256_store_ps(&(c->element[1][index]), tmp4[1]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 tmp4[DSSIZE], sc4[DSSIZE], aij4[DSSIZE];

	sc4[0] = _mm512_set1_ps(sc[0]);
	sc4[1] = _mm512_set1_ps(sc[1]);

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = _mm512_load_ps(&(a->element[0][index]));
		aij4[1] = _mm512_load_ps(&(a->element[1][index]));

		_bncavx512_rds_mul(tmp4, sc4, aij4);

		_mm512_store_ps(&(c->element[0][index]), tmp4[0]);
		_mm512_store_ps(&(c->element[1][index]), tmp4[1]); 
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t tmp4_0, tmp4_1;
	svfloat32_t sc4_0, sc4_1;
	svfloat32_t aij4_0, aij4_1;

	sc4_0 = svdup_f32(sc[0]);
	sc4_1 = svdup_f32(sc[1]);

	for(index = 0; index < real_total_dim; index += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(real_total_dim));
		aij4_0 = svld1_f32(pg, &(a->element[0][index]));
		aij4_1 = svld1_f32(pg, &(a->element[1][index]));

		_bncsve2_rds_mul(pg, &tmp4_0, &tmp4_1, sc4_0, sc4_1, aij4_0, aij4_1);

		svst1_f32(pg, &(c->element[0][index]), tmp4_0);
		svst1_f32(pg, &(c->element[1][index]), tmp4_1); 
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t tmp4[DSSIZE], sc4[DSSIZE], aij4[DSSIZE];

	sc4[0] = vdupq_n_f32(sc[0]);
	sc4[1] = vdupq_n_f32(sc[1]);

	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		aij4[0] = vld1q_f32(&(a->element[0][index]));
		aij4[1] = vld1q_f32(&(a->element[1][index]));

		_bncneon_rds_mul(tmp4, sc4, aij4);

		vst1q_f32(&(c->element[0][index]), tmp4[0]);
		vst1q_f32(&(c->element[1][index]), tmp4[1]); 
	}
#else // others
	float tmp[DSSIZE], aij[DSSIZE];

	for(index = 0; index < real_total_dim; index++)
	{
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];

		rds_mul(tmp, sc, aij);

		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1]; 
	}
#endif // __AVX2__
/*
	float tmp[DSSIZE];

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rds_mul(tmp, sc, get_dsmatrix_ij(a, i, j));
			set_dsmatrix_ij(c, i, j, tmp);
		}
	}
*/
}

/* c = a^T */
void transpose_dsmatrix(DSMatrix c, DSMatrix a)
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
	__m256 aji4[DSSIZE];

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
			index = i * real_col_dim + j;
			_mm256_store_ps(&(c->element[0][index]), aji4[0]);
			_mm256_store_ps(&(c->element[1][index]), aji4[1]);
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 aji4[DSSIZE];

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
			index = i * real_col_dim + j;
			_mm512_store_ps(&(c->element[0][index]), aji4[0]);
			_mm512_store_ps(&(c->element[1][index]), aji4[1]);
		}
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, gather)
    svfloat32_t aji_0, aji_1;
    svint32_t vidx;
    for(i = 0; i < real_row_dim; i++)
    {
        for(j = 0; j < real_col_dim; j += (long int)svcntw())
        {
            svbool_t pg = svwhilelt_b32_s64((int64_t)j, (int64_t)real_col_dim);
            vidx = svindex_s32((int32_t)(j * real_col_dim + i), (int32_t)real_col_dim);
            aji_0 = svld1_gather_s32index_f32(pg, a->element[0], vidx);
            aji_1 = svld1_gather_s32index_f32(pg, a->element[1], vidx);
            index = i * real_col_dim + j;
            svst1_f32(pg, &(c->element[0][index]), aji_0);
            svst1_f32(pg, &(c->element[1][index]), aji_1);
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
        }
#else // others
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_dsmatrix_ij(c, i, j, get_dsmatrix_ij(a, j, i));
	}
#endif // AVX2
}

/* c := a */
void subst_dsmatrix(DSMatrix c, DSMatrix a)
{
	long int i, j, index;
	long int real_row_dim, real_col_dim;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_dsmatrix\n");
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
			_mm256_store_ps(&(c->element[0][i * real_col_dim + j]), _mm256_load_ps(&(a->element[0][index])));
			_mm256_store_ps(&(c->element[1][i * real_col_dim + j]), _mm256_load_ps(&(a->element[1][index])));
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
			_mm512_store_ps(&(c->element[0][i * real_col_dim + j]), _mm512_load_ps(&(a->element[0][index])));
			_mm512_store_ps(&(c->element[1][i * real_col_dim + j]), _mm512_load_ps(&(a->element[1][index])));
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
			svst1_f32(pg, &(c->element[0][i * real_col_dim + j]), svld1_f32(pg, &(a->element[0][index])));
			svst1_f32(pg, &(c->element[1][i * real_col_dim + j]), svld1_f32(pg, &(a->element[1][index])));
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
			vst1q_f32(&(c->element[0][i * real_col_dim + j]), vld1q_f32(&(a->element[0][index])));
			vst1q_f32(&(c->element[1][i * real_col_dim + j]), vld1q_f32(&(a->element[1][index])));
		}
	}
#else // others
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_dsmatrix_ij(c, i, j, get_dsmatrix_ij(a, i, j));
		}
	}
#endif // AVX2
}

/* c := I */
void setI_dsmatrix(DSMatrix c)
{
	long int i, j;
	long int real_total_dim;
	float tmp1[DSSIZE];

	real_total_dim = c->real_row_dim * c->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 zero4;

	zero4 = _mm256_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm256_store_ps(&c->element[0][i], zero4);
		_mm256_store_ps(&c->element[1][i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero4;

	zero4 = _mm512_setzero_ps();
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm512_store_ps(&c->element[0][i], zero4);
		_mm512_store_ps(&c->element[1][i], zero4);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	
	svfloat32_t zero4;

	zero4 = svdup_f32(0.0f);
	for(i = 0; i < real_total_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(real_total_dim));
		svst1_f32(pg, &c->element[0][i], zero4);
		svst1_f32(pg, &c->element[1][i], zero4);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t zero4;

	zero4 = vdupq_n_f32(0.0f);
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		vst1q_f32(&c->element[0][i], zero4);
		vst1q_f32(&c->element[1][i], zero4);
	}
#else // others
	for(i = 0; i < real_total_dim; i++)
	{
		c->element[0][i] = 0.0;
		c->element[1][i] = 0.0;	
	}
#endif // __AVX2__

	rds_set_ui(tmp1, 1UL);

	for(i = 0; i < c->row_dim; i++)
	{
		if(i < c->col_dim)
			set_dsmatrix_ij(c, i, i, tmp1);
	}
}

/* v := a * vb */
void mul_dsmatrix_dsvec(DSVector v, DSMatrix a, DSVector vb)
{
	long int i, j;
	float tmp[DSSIZE], tmp1[DSSIZE];

	/* Check Dimension */
	//if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	if((v->dim < a->row_dim) || (vb->dim < a->col_dim))
	{
		fprintf(stderr, "ERROR: mul_dsmatrix_dsvec\n");
		return;
	}

// SIMD
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int ij_index, real_col_dim;
	__m256 tmp4[DSSIZE], tmp1_4[DSSIZE];
	__m256 aij4[DSSIZE], vbj4[DSSIZE];

	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->row_dim; i++)
	{
		//rds_set_ui(tmp, 0UL);
		_bncavx2_set0_ds(tmp4);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij4[0] = _mm256_load_ps(&(a->element[0][ij_index]));
			aij4[1] = _mm256_load_ps(&(a->element[1][ij_index]));
			vbj4[0] = _mm256_load_ps(&(vb->element[0][j]));
			vbj4[1] = _mm256_load_ps(&(vb->element[1][j]));

			//rds_mul(tmp1, get_dsmatrix_ij(a, i, j), get_dsvector_i(vb, j));
			//rds_add(tmp, tmp, tmp1);
			_bncavx2_rds_mul(tmp1_4, aij4, vbj4);
			_bncavx2_rds_add(tmp4, tmp4, tmp1_4);
		}
		//set_dsvector_i(v, i, tmp);
		_bncavx2_rds_sum256(tmp, tmp4);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int ij_index, real_col_dim;
	__m512 tmp4[DSSIZE], tmp1_4[DSSIZE];
	__m512 aij4[DSSIZE], vbj4[DSSIZE];

	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->row_dim; i++)
	{
		//rds_set_ui(tmp, 0UL);
		_bncavx512_set0_ds(tmp4);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij4[0] = _mm512_load_ps(&(a->element[0][ij_index]));
			aij4[1] = _mm512_load_ps(&(a->element[1][ij_index]));
			vbj4[0] = _mm512_load_ps(&(vb->element[0][j]));
			vbj4[1] = _mm512_load_ps(&(vb->element[1][j]));

			//rds_mul(tmp1, get_dsmatrix_ij(a, i, j), get_dsvector_i(vb, j));
			//rds_add(tmp, tmp, tmp1);
			_bncavx512_rds_mul(tmp1_4, aij4, vbj4);
			_bncavx512_rds_add(tmp4, tmp4, tmp1_4);
		}
		//set_dsvector_i(v, i, tmp);
		_bncavx512_rds_sum512(tmp, tmp4);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic)
	{
		long real_col_dim = a->real_col_dim;
		long vl = (long)svcntw();
		for(i = 0; i < a->row_dim; i++)
		{
			svfloat32_t acc0, acc1;
			_bncsve2_rds_set0(&acc0, &acc1);
			for(j = 0; j < real_col_dim; j += vl)
			{
				svbool_t pg = svwhilelt_b32_s32((int32_t)j, (int32_t)real_col_dim);
				long ij = i * real_col_dim + j;
				svfloat32_t a0 = svld1_f32(pg, &(a->element[0][ij]));
				svfloat32_t a1 = svld1_f32(pg, &(a->element[1][ij]));
				svfloat32_t b0 = svld1_f32(pg, &(vb->element[0][j]));
				svfloat32_t b1 = svld1_f32(pg, &(vb->element[1][j]));
				svfloat32_t t0, t1;
				_bncsve2_rds_mul(pg, &t0, &t1, a0, a1, b0, b1);
				_bncsve2_rds_add(pg, &acc0, &acc1, acc0, acc1, t0, t1);
			}
			{
				long _L, _vl = (long)svcntw();
				float _la0[64], _la1[64];
				svst1_f32(svptrue_b32(), _la0, acc0);
				svst1_f32(svptrue_b32(), _la1, acc1);
				rds_set_ui(tmp, 0UL);
				for(_L = 0; _L < _vl; _L++)
				{
					float _lane[DSSIZE] = { _la0[_L], _la1[_L] };
					rds_add(tmp, tmp, _lane);
				}
			}
			v->element[0][i] = tmp[0];
			v->element[1][i] = tmp[1];
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	{
		long int ij_index, real_col_dim = a->real_col_dim;
		float32x4_t tmp4[DSSIZE], tmp1_4[DSSIZE], aij4[DSSIZE], vbj4[DSSIZE];
		for(i = 0; i < a->row_dim; i++)
		{
			_bncneon_set0_ds(tmp4);
			for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
			{
				ij_index = i * real_col_dim + j;
				aij4[0] = vld1q_f32(&(a->element[0][ij_index]));
				aij4[1] = vld1q_f32(&(a->element[1][ij_index]));
				vbj4[0] = vld1q_f32(&(vb->element[0][j]));
				vbj4[1] = vld1q_f32(&(vb->element[1][j]));
				_bncneon_rds_mul(tmp1_4, aij4, vbj4);
				_bncneon_rds_add(tmp4, tmp4, tmp1_4);
			}
			/* horizontal DS sum across the 4 NEON lanes */
			{
				float _l0[DSSIZE] = {vgetq_lane_f32(tmp4[0],0), vgetq_lane_f32(tmp4[1],0)};
				float _l1[DSSIZE] = {vgetq_lane_f32(tmp4[0],1), vgetq_lane_f32(tmp4[1],1)};
				float _l2[DSSIZE] = {vgetq_lane_f32(tmp4[0],2), vgetq_lane_f32(tmp4[1],2)};
				float _l3[DSSIZE] = {vgetq_lane_f32(tmp4[0],3), vgetq_lane_f32(tmp4[1],3)};
				rds_set(tmp, _l0); rds_add(tmp, tmp, _l1);
				rds_add(tmp, tmp, _l2); rds_add(tmp, tmp, _l3);
			}
			v->element[0][i] = tmp[0];
			v->element[1][i] = tmp[1];
		}
	}
#else // others

	for(i = 0; i < a->row_dim; i++)
	{
		rds_set_ui(tmp, 0UL);
		for(j = 0; j < a->col_dim; j++)
		{
			rds_mul(tmp1, get_dsmatrix_ij(a, i, j), get_dsvector_i(vb, j));
			rds_add(tmp, tmp, tmp1);
		}
		set_dsvector_i(v, i, tmp);
	}
#endif // __AVX2__

}

/* v := a^T * vb */
void mul_dsmatrixt_dsvec(DSVector v, DSMatrix a, DSVector vb)
{
	long int i, j;
	float tmp[DSSIZE], tmp1[DSSIZE];

	/* Check Dimension */
	//if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	if((v->dim < a->col_dim) || (vb->dim < a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_dsmatrixt_dsvec\n");
		return;
	}

// SIMD
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int ji_index, real_row_dim, real_col_dim;
	__m256 tmp4[DSSIZE], tmp1_4[DSSIZE];
	__m256 aij4[DSSIZE], vbj4[DSSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->col_dim; i++)
	{
		//rds_set_ui(tmp, 0UL);
		_bncavx2_set0_ds(tmp4);
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
			vbj4[0] = _mm256_load_ps(&(vb->element[0][j]));
			vbj4[1] = _mm256_load_ps(&(vb->element[1][j]));

			//rds_mul(tmp1, get_dsmatrix_ij(a, i, j), get_dsvector_i(vb, j));
			//rds_add(tmp, tmp, tmp1);
			_bncavx2_rds_mul(tmp1_4, aij4, vbj4);
			_bncavx2_rds_add(tmp4, tmp4, tmp1_4);
		}
		//set_dsvector_i(v, i, tmp);
		_bncavx2_rds_sum256(tmp, tmp4);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int ji_index, real_row_dim, real_col_dim;
	__m512 tmp4[DSSIZE], tmp1_4[DSSIZE];
	__m512 aij4[DSSIZE], vbj4[DSSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	for(i = 0; i < a->col_dim; i++)
	{
		//rds_set_ui(tmp, 0UL);
		_bncavx512_set0_ds(tmp4);
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
			vbj4[0] = _mm512_load_ps(&(vb->element[0][j]));
			vbj4[1] = _mm512_load_ps(&(vb->element[1][j]));

			//rds_mul(tmp1, get_dsmatrix_ij(a, i, j), get_dsvector_i(vb, j));
			//rds_add(tmp, tmp, tmp1);
			_bncavx512_rds_mul(tmp1_4, aij4, vbj4);
			_bncavx512_rds_add(tmp4, tmp4, tmp1_4);
		}
		//set_dsvector_i(v, i, tmp);
		_bncavx512_rds_sum512(tmp, tmp4);
		v->element[0][i] = tmp[0];
		v->element[1][i] = tmp[1];
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic)
	{
		long real_col_dim = a->real_col_dim, real_row_dim = a->real_row_dim;
		long vl = (long)svcntw();
		for(i = 0; i < a->col_dim; i++)
		{
			svfloat32_t acc0, acc1;
			_bncsve2_rds_set0(&acc0, &acc1);
			for(j = 0; j < real_row_dim; j += vl)
			{
				svbool_t pg = svwhilelt_b32_s32((int32_t)j, (int32_t)real_row_dim);
				svint32_t idx = svindex_s32((int32_t)(j * real_col_dim + i), (int32_t)real_col_dim);
				svfloat32_t a0 = svld1_gather_s32index_f32(pg, a->element[0], idx);
				svfloat32_t a1 = svld1_gather_s32index_f32(pg, a->element[1], idx);
				svfloat32_t b0 = svld1_f32(pg, &(vb->element[0][j]));
				svfloat32_t b1 = svld1_f32(pg, &(vb->element[1][j]));
				svfloat32_t t0, t1;
				_bncsve2_rds_mul(pg, &t0, &t1, a0, a1, b0, b1);
				_bncsve2_rds_add(pg, &acc0, &acc1, acc0, acc1, t0, t1);
			}
			{
				long _L, _vl = (long)svcntw();
				float _la0[64], _la1[64];
				svst1_f32(svptrue_b32(), _la0, acc0);
				svst1_f32(svptrue_b32(), _la1, acc1);
				rds_set_ui(tmp, 0UL);
				for(_L = 0; _L < _vl; _L++)
				{
					float _lane[DSSIZE] = { _la0[_L], _la1[_L] };
					rds_add(tmp, tmp, _lane);
				}
			}
			v->element[0][i] = tmp[0];
			v->element[1][i] = tmp[1];
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	{
		long real_col_dim = a->real_col_dim, real_row_dim = a->real_row_dim;
		float32x4_t tmp4[DSSIZE], tmp1_4[DSSIZE], aij4[DSSIZE], vbj4[DSSIZE];
		for(i = 0; i < a->col_dim; i++)
		{
			_bncneon_set0_ds(tmp4);
			for(j = 0; j < real_row_dim; j += _BNC_S_WIDTH)
			{
				aij4[0] = (float32x4_t){ a->element[0][(j)*real_col_dim+i], a->element[0][(j+1)*real_col_dim+i], a->element[0][(j+2)*real_col_dim+i], a->element[0][(j+3)*real_col_dim+i] };
				aij4[1] = (float32x4_t){ a->element[1][(j)*real_col_dim+i], a->element[1][(j+1)*real_col_dim+i], a->element[1][(j+2)*real_col_dim+i], a->element[1][(j+3)*real_col_dim+i] };
				vbj4[0] = vld1q_f32(&(vb->element[0][j]));
				vbj4[1] = vld1q_f32(&(vb->element[1][j]));
				_bncneon_rds_mul(tmp1_4, aij4, vbj4);
				_bncneon_rds_add(tmp4, tmp4, tmp1_4);
			}
			{
				float _l0[DSSIZE]={vgetq_lane_f32(tmp4[0],0),vgetq_lane_f32(tmp4[1],0)};
				float _l1[DSSIZE]={vgetq_lane_f32(tmp4[0],1),vgetq_lane_f32(tmp4[1],1)};
				float _l2[DSSIZE]={vgetq_lane_f32(tmp4[0],2),vgetq_lane_f32(tmp4[1],2)};
				float _l3[DSSIZE]={vgetq_lane_f32(tmp4[0],3),vgetq_lane_f32(tmp4[1],3)};
				rds_set(tmp,_l0); rds_add(tmp,tmp,_l1); rds_add(tmp,tmp,_l2); rds_add(tmp,tmp,_l3);
			}
			v->element[0][i] = tmp[0];
			v->element[1][i] = tmp[1];
		}
	}
#else // others

	for(i = 0; i < a->col_dim; i++)
	{
		set0_ds(tmp);
		for(j = 0; j < a->row_dim; j++)
		{
			rds_mul(tmp1, get_dsmatrix_ij(a, j, i), get_dsvector_i(vb, j));
			rds_add(tmp, tmp, tmp1);
		}
		set_dsvector_i(v, i, tmp);
	}
#endif // __AVX2__
}

/* a = a^(-1) */
/* square matrix only */
void inv_dsmatrix(DSMatrix a)
{
	long int i, j, k, dim;
	float tmp[DSSIZE], aii[DSSIZE];

	/* Check Dimensions */
	if(a->row_dim != a->col_dim)
	{
		fprintf(stderr, "ERROR: inv_dsmatrix\n");
		return;
	}

	dim = a->row_dim;

	for(i = 0; i < dim; i++)
	{
		if(rds_cmp_ui(get_dsmatrix_ij(a, i, i), 0UL) == 0) 
		{
			fprintf(stderr, "ERROR: inv_dsmatrix: Pivot(%ld,%ld) is zero.\n", i, i);
			return;
		}

		rds_ui_div(aii, 1UL, get_dsmatrix_ij(a, i, i));
		set_dsmatrix_ij(a, i, i, aii);

		for(j = 0; j < i; j++)
		{
			rds_mul(tmp, get_dsmatrix_ij(a, i, j), aii);
			set_dsmatrix_ij(a, i, j, tmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			rds_mul(tmp, get_dsmatrix_ij(a, i, j), aii);
			set_dsmatrix_ij(a, i, j, tmp);
		}

		for(j = 0; j < i; j++)
		{
			for(k = 0; k < i; k++)
			{
				rds_mul(tmp, get_dsmatrix_ij(a, j, i), get_dsmatrix_ij(a, i, k));
				rds_sub(tmp, get_dsmatrix_ij(a, j, k), tmp);
				set_dsmatrix_ij(a, j, k, tmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				rds_mul(tmp, get_dsmatrix_ij(a, j, i), get_dsmatrix_ij(a, i, k));
				rds_sub(tmp, get_dsmatrix_ij(a, j, k), tmp);
				set_dsmatrix_ij(a, j, k, tmp);
			}
		}

		for(j = i + 1; j < dim; j++)
		{
			for(k = 0; k < i; k++)
			{
				rds_mul(tmp, get_dsmatrix_ij(a, j, i), get_dsmatrix_ij(a, i, k));
				rds_sub(tmp, get_dsmatrix_ij(a, j, k), tmp);
				set_dsmatrix_ij(a, j, k, tmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				rds_mul(tmp, get_dsmatrix_ij(a, j, i), get_dsmatrix_ij(a, i, k));
				rds_sub(tmp, get_dsmatrix_ij(a, j, k), tmp);
				set_dsmatrix_ij(a, j, k, tmp);
			}
		}

		for(j = 0; j < i; j++)
		{
			rds_neg(tmp, aii); /* tmp := -aii */
			rds_mul(tmp, tmp, get_dsmatrix_ij(a, j, i));
			set_dsmatrix_ij(a, j, i, tmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			rds_neg(tmp, aii); /* tmp := -aii */
			rds_mul(tmp, tmp, get_dsmatrix_ij(a, j, i));
			set_dsmatrix_ij(a, j, i, tmp);
		}
	}
}

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpf)a */
void subst_mpfvector_dsvec(MPFVector c, DSVector a)
{
	long int i;
	mpf_t tmp;

	mpf_init2(tmp, c->prec);

	for(i = 0; i < a->dim; i++)
	{
		mpf_set_ds(tmp, get_dsvector_i(a, i));
		set_mpfvector_i(c, i, tmp);
	}

	mpf_clear(tmp);

}

/* c := (dd)a */
void subst_dsvector_mpfvec(DSVector c, MPFVector a)
{
	long int i;
	float tmp[DSSIZE];

	for(i = 0; i < a->dim; i++)
	{
		mpf_get_ds(tmp, get_mpfvector_i(a, i));
		set_dsvector_i(c, i, tmp);
	}

}
/* c := (mpf)a */
void subst_mpfmatrix_dsmat(MPFMatrix c, DSMatrix a)
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
			mpf_set_ds(tmp, get_dsmatrix_ij(a, i, j));
			set_mpfmatrix_ij(c, i, j, tmp);
		}
	}

	mpf_clear(tmp);
}

/* c := (dd)a */
void subst_dsmatrix_mpfmat(DSMatrix c, MPFMatrix a)
{
	long int i, j;
	float tmp[DSSIZE];

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_dsmatrix_mpfmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			mpf_get_ds(tmp, get_mpfmatrix_ij(a, i, j));
			set_dsmatrix_ij(c, i, j, tmp);
		}
	}
}

/* Normwise relative error of vector */
void relerr_dsvector_mpfvec(float relerr[DSSIZE], DSVector approx_vec, MPFVector true_vec, int norm_type)
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
	subst_mpfvector_dsvec(mpf_approx_vec, approx_vec);

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
	mpf_get_ds(relerr, mpf_relerr);

	free_mpfvector(diff_vec);
	free_mpfvector(mpf_approx_vec);
	mpf_clear(norm_diff_vec);
	mpf_clear(norm_true_vec);
	mpf_clear(mpf_relerr);

	return;
}

/* Elementwise relative errors of vector */
void relerr_element_dsvector_mpf(float max_relerr[DSSIZE], float min_relerr[DSSIZE], float norm_relerr[DSSIZE], DSVector approx_vec, MPFVector true_vec, int norm_type)
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
	subst_mpfvector_dsvec(mpf_approx_vec, approx_vec);

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
	mpf_get_ds(max_relerr, mpf_max_relerr);
	mpf_get_ds(min_relerr, mpf_min_relerr);
	mpf_get_ds(norm_relerr, mpf_norm_relerr);

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
void subst_dsvector_dvec(DSVector c, FVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
	{
		set_dsvector_i_d(c, i, get_dvector_i(a, i));
	}
}

/* c := (d)a */
void subst_fvector_dsvec(FVector c, DSVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
	{
		c->element[i] = rds_get_f(get_dsvector_i(a, i));
	}
}


/* c := (dd)a */
void subst_dsmatrix_fmat(DSMatrix c, FMatrix a)
{
	long int i, j;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_dsmatrix_fmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_dsmatrix_ij_f(c, i, j, get_fmatrix_ij(a, i, j));
		}
	}
}

/* c := (d)a */
void subst_fmatrix_dsmat(FMatrix c, DSMatrix a)
{
	long int i, j, ij_index;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_fmatrix_dsmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			ij_index = i * (c->col_dim) + j;
			c->element[ij_index] = rds_get_f(get_dsmatrix_ij(a, i, j));
		}
	}
}

/* Normwise relative error of vector */
void relerr_dsvector(float relerr[DSSIZE], DSVector approx_vec, DSVector true_vec, int norm_type)
{
	float norm_true_vec[DSSIZE], norm_diff_vec[DSSIZE];
	DSVector diff_vec;

	diff_vec = init_dsvector(approx_vec->dim);

	// diff_vec := approx_vec - true_vec
	sub_dsvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_dsvector(norm_diff_vec, diff_vec);
			normi_dsvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_dsvector(norm_diff_vec, diff_vec);
			norm1_dsvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_dsvector(norm_diff_vec, diff_vec);
			norm2_dsvector(norm_true_vec, true_vec);
			break;
	}

	if(rds_cmp_ui(norm_true_vec, 0UL) != 0)
		rds_div(relerr, norm_diff_vec, norm_true_vec);

	free_dsvector(diff_vec);

	return;
}

/* Elementwise relative errors of vector */
void relerr_element_dsvector(float max_relerr[DSSIZE], float min_relerr[DSSIZE], float norm_relerr[DSSIZE], DSVector approx_vec, DSVector true_vec, int norm_type)
{
	float abs_true_vec[DSSIZE], abs_diff_vec[DSSIZE], norm_diff_vec[DSSIZE], norm_true_vec[DSSIZE];
	long int i;
	DSVector diff_vec;

	diff_vec = init_dsvector(approx_vec->dim);

	// diff_vec := approx_vec - true_vec
	sub_dsvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_dsvector(norm_diff_vec, diff_vec);
			normi_dsvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_dsvector(norm_diff_vec, diff_vec);
			norm1_dsvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_dsvector(norm_diff_vec, diff_vec);
			norm2_dsvector(norm_true_vec, true_vec);
			break;
	}

	rds_set(norm_relerr, norm_diff_vec);
	if(rds_cmp_ui(norm_true_vec, 0UL) != 0)
		rds_div(norm_relerr, norm_diff_vec, norm_true_vec);

	// relative errors of each elements
	rds_set_ui(max_relerr, 0UL);
	normi_dsvector(min_relerr, diff_vec);
	for(i = 0; i < approx_vec->dim; i++)
	{
		rds_abs(abs_diff_vec, get_dsvector_i(diff_vec, i));
		rds_abs(abs_true_vec, get_dsvector_i(true_vec, i));
		if(rds_cmp_ui(abs_true_vec, 0UL) != 0)
			rds_div(abs_diff_vec, abs_diff_vec, abs_true_vec);
		
		if(rds_cmp(max_relerr, abs_diff_vec) < 0)
			rds_set(max_relerr, abs_diff_vec);
		if(rds_cmp(min_relerr, abs_diff_vec) > 0)
			rds_set(min_relerr, abs_diff_vec);
	}

	free_dsvector(diff_vec);// Fix! 2012-06-03 by T.Kouya

	return;
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_dsmatrix(DSMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
	long int i, j, true_end;
	float tmp[DSSIZE];

	true_end = (col_end > mat->col_dim) ? mat->col_dim : col_end;

	for(i = col_start; i < true_end; i++)
	{
		rds_set(tmp, get_dsmatrix_ij(mat, row_index0, i));
		set_dsmatrix_ij(mat, row_index0, i, get_dsmatrix_ij(mat, row_index1, i));
		set_dsmatrix_ij(mat, row_index1, i, tmp);
	}
}

#ifdef __cplusplus
} // extern "C"
#endif
