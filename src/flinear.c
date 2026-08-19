//******************************************************************************
// flinear.c : Single Precision Basic Linear Algebra 
// Copyright (C) 2021 Tomonori Kouya
// 
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License or any later
// version.
// 
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// 
//******************************************************************************
#include "flinear.h"

#if defined (__cplusplus)
extern "C" {
#endif // defined (__cplusplus)

float frel_diff_array(float approx_a[], float approx_b[], int dim, int print_flag)
{
    int i;
    float rel_min, rel_max, rel_ave, rel_diff;

    rel_diff = frel_diff(approx_a[0], approx_b[0]);
    rel_min = rel_diff;
    rel_max = rel_diff;
    rel_ave = rel_diff;

    for(i = 1; i < dim; i++)
    {
        rel_diff = frel_diff(approx_a[i], approx_b[i]);
        if(rel_diff < rel_min) rel_min = rel_diff;
        if(rel_diff > rel_max) rel_max = rel_diff;
        rel_ave += rel_diff;
    }
    rel_ave /= (float)dim;

    if(print_flag == 1)
        printf("max_rel_diff, min_rel_diff, ave_rel_diff: %5.2e, %5.2e, %5.2e\n", rel_max, rel_min, rel_ave);

    return rel_max;
}

/*************************************************/
/* Vector Calculations for FVector               */
/*
FVector init_fvector(long int dimension)
void free_fvector(FVector vec)
void add_fvector(FVector c, FVector a, FVector b)
void add2_fvector(FVector c, FVector a)
void sub_fvector(FVector c, FVector a, FVector b)
void sub2_fvector(FVector c, FVector a)
void cmul_fvector(FVector c, float val, FVector a)
void cmul2_fvector(FVector c, float val)
void add_cmul_fvector(FVector c, FVector a, float val, FVector b)
float ip_fvector(FVector a, FVector b)
float norm1_fvector(FVector a)
float norm2_fvector(FVector a)
float normi_fvector(FVector a)
void subst_fvector(FVector c, FVector a)
*/
/*************************************************/

FVector init_fvector(long int dimension)
{
	FVector ret = NULL;
	long int i, real_dim;

	if(dimension <= 0)
	{
		fprintf(stderr, "ERROR: init_fvector\n");
		return ret;
	}

	ret = (FVector)BNC_MALLOC(sizeof(fvector));
	if(ret == NULL)
		return ret;

	// real_dim is the nearest positive multiplier of _BNC_S_WIDTH
	real_dim = (long int)ceil((float)dimension / (float)_BNC_S_WIDTH) * _BNC_S_WIDTH;

//	ret->element = (float *)BNC_CALLOC(dimension, sizeof(float));
	ret->element = (float *)BNC_CALLOC(real_dim, sizeof(float));
	if(ret->element == NULL)
		return ret;

	/* All 0 */
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 zero8;

	zero8 = _mm256_setzero_ps();
	for(i = 0; i < real_dim; i += _BNC_S_WIDTH)
		_mm256_store_ps((ret->element + i), zero8);

#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero8;

	zero8 = _mm512_setzero_ps();
	for(i = 0; i < real_dim; i += _BNC_S_WIDTH)
		_mm512_store_ps((ret->element + i), zero8);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t zero8;

	zero8 = svdup_f32(0.0f);
	for(i = 0; i < real_dim; i += (long int)svcntw()) {
			svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(real_dim));
		svst1_f32(pg, (ret->element + i), zero8);
		}

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t zero8;

	zero8 = vdupq_n_f32(0.0f);
	for(i = 0; i < real_dim; i += _BNC_S_WIDTH)
		vst1q_f32((ret->element + i), zero8);

#else // others
	for(i = 0; i < dimension; i++)
		*(ret->element + i) = 0.0;
#endif // __AVX2__

	ret->dim = dimension;
	ret->real_dim = real_dim;

	return ret;
}

void free_fvector(FVector vec)
{
	if(vec == NULL)
		return;

	if(vec->element != NULL)
		free(vec->element);

	free(vec);
}

/* c = a + b */
void add_fvector(FVector c, FVector a, FVector b)
{
	long int i;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_fvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4, b4, c4;

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, get_fvector_i(a, i) + get_fvector_i(b, i));

		a4 = _mm256_load_ps(&get_fvector_i(a, i));
		b4 = _mm256_load_ps(&get_fvector_i(b, i));
		c4 = _mm256_add_ps(a4, b4);
		_mm256_store_ps(&get_fvector_i(c, i), c4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a8, b8, c8;

	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, get_fvector_i(a, i) + get_fvector_i(b, i));

		a8 = _mm512_load_ps(&get_fvector_i(a, i));
		b8 = _mm512_load_ps(&get_fvector_i(b, i));
		c8 = _mm512_add_ps(a8, b8);
		_mm512_store_ps(&get_fvector_i(c, i), c8);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t a4, b4, c4;

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += (long int)svcntw())
	{
			svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(c->real_dim));
		//set_fvector_i(c, i, get_fvector_i(a, i) + get_fvector_i(b, i));

		a4 = svld1_f32(pg, &get_fvector_i(a, i));
		b4 = svld1_f32(pg, &get_fvector_i(b, i));
		c4 = svadd_f32_x(pg, a4, b4);
		svst1_f32(pg, &get_fvector_i(c, i), c4);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a4, b4, c4;

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, get_fvector_i(a, i) + get_fvector_i(b, i));

		a4 = vld1q_f32(&get_fvector_i(a, i));
		b4 = vld1q_f32(&get_fvector_i(b, i));
		c4 = vaddq_f32(a4, b4);
		vst1q_f32(&get_fvector_i(c, i), c4);
	}
#else // others
	for(i = 0; i < c->dim; i++)
		set_fvector_i(c, i, get_fvector_i(a, i) + get_fvector_i(b, i));
#endif // __AVX2__

}

/* c += a */
void add2_fvector(FVector c, FVector a)
{
	long int i;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: add2_fvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4, c4;

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, get_fvector_i(c, i) + get_fvector_i(a, i));

		a4 = _mm256_load_ps(&get_fvector_i(a, i));
		c4 = _mm256_load_ps(&get_fvector_i(c, i));
		c4 = _mm256_add_ps(c4, a4);
		_mm256_store_ps(&get_fvector_i(c, i), c4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a8, c8;

	//for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, get_fvector_i(c, i) + get_fvector_i(a, i));

		a8 = _mm512_load_ps(&get_fvector_i(a, i));
		c8 = _mm512_load_ps(&get_fvector_i(c, i));
		c8 = _mm512_add_ps(c8, a8);
		_mm512_store_ps(&get_fvector_i(c, i), c8);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t a4, c4;

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += (long int)svcntw())
	{
			svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(c->real_dim));
		//set_fvector_i(c, i, get_fvector_i(c, i) + get_fvector_i(a, i));

		a4 = svld1_f32(pg, &get_fvector_i(a, i));
		c4 = svld1_f32(pg, &get_fvector_i(c, i));
		c4 = svadd_f32_x(pg, c4, a4);
		svst1_f32(pg, &get_fvector_i(c, i), c4);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a4, c4;

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, get_fvector_i(c, i) + get_fvector_i(a, i));

		a4 = vld1q_f32(&get_fvector_i(a, i));
		c4 = vld1q_f32(&get_fvector_i(c, i));
		c4 = vaddq_f32(c4, a4);
		vst1q_f32(&get_fvector_i(c, i), c4);
	}
#else // others
	for(i = 0; i < c->dim; i++)
		set_fvector_i(c, i, get_fvector_i(c, i) + get_fvector_i(a, i));
#endif // __AVX2__

}

/* c = a - b */
void sub_fvector(FVector c, FVector a, FVector b)
{
	long int i;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_fvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4, b4, c4;

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, get_fvector_i(a, i) - get_fvector_i(b, i));

		a4 = _mm256_load_ps(&get_fvector_i(a, i));
		b4 = _mm256_load_ps(&get_fvector_i(b, i));
		c4 = _mm256_sub_ps(a4, b4);
		_mm256_store_ps(&get_fvector_i(c, i), c4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a8, b8, c8;

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, get_fvector_i(a, i) - get_fvector_i(b, i));

		a8 = _mm512_load_ps(&get_fvector_i(a, i));
		b8 = _mm512_load_ps(&get_fvector_i(b, i));
		c8 = _mm512_sub_ps(a8, b8);
		_mm512_store_ps(&get_fvector_i(c, i), c8);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t a4, b4, c4;

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += (long int)svcntw())
	{
			svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(c->real_dim));
		//set_fvector_i(c, i, get_fvector_i(a, i) - get_fvector_i(b, i));

		a4 = svld1_f32(pg, &get_fvector_i(a, i));
		b4 = svld1_f32(pg, &get_fvector_i(b, i));
		c4 = svsub_f32_x(pg, a4, b4);
		svst1_f32(pg, &get_fvector_i(c, i), c4);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a4, b4, c4;

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, get_fvector_i(a, i) - get_fvector_i(b, i));

		a4 = vld1q_f32(&get_fvector_i(a, i));
		b4 = vld1q_f32(&get_fvector_i(b, i));
		c4 = vsubq_f32(a4, b4);
		vst1q_f32(&get_fvector_i(c, i), c4);
	}
#else // others
	for(i = 0; i < c->dim; i++)
		set_fvector_i(c, i, get_fvector_i(a, i) - get_fvector_i(b, i));
#endif // __AVX2__

}

/* c -= a */
void sub2_fvector(FVector c, FVector a)
{
	long int i;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: sub2_fvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4, c4;

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, get_fvector_i(c, i) - get_fvector_i(a, i));

		a4 = _mm256_load_ps(&get_fvector_i(a, i));
		c4 = _mm256_load_ps(&get_fvector_i(c, i));
		c4 = _mm256_sub_ps(c4, a4);
		_mm256_store_ps(&get_fvector_i(c, i), c4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a8, c8;

	//for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, get_fvector_i(c, i) - get_fvector_i(a, i));

		a8 = _mm512_load_ps(&get_fvector_i(a, i));
		c8 = _mm512_load_ps(&get_fvector_i(c, i));
		c8 = _mm512_sub_ps(c8, a8);
		_mm512_store_ps(&get_fvector_i(c, i), c8);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t a4, c4;

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += (long int)svcntw())
	{
			svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(c->real_dim));
		//set_fvector_i(c, i, get_fvector_i(c, i) - get_fvector_i(a, i));

		a4 = svld1_f32(pg, &get_fvector_i(a, i));
		c4 = svld1_f32(pg, &get_fvector_i(c, i));
		c4 = svsub_f32_x(pg, c4, a4);
		svst1_f32(pg, &get_fvector_i(c, i), c4);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a4, c4;

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, get_fvector_i(c, i) - get_fvector_i(a, i));

		a4 = vld1q_f32(&get_fvector_i(a, i));
		c4 = vld1q_f32(&get_fvector_i(c, i));
		c4 = vsubq_f32(c4, a4);
		vst1q_f32(&get_fvector_i(c, i), c4);
	}
#else // others
	for(i = 0; i < c->dim; i++)
		set_fvector_i(c, i, get_fvector_i(c, i) - get_fvector_i(a, i));
#endif // __AVX2__

}

/* c = val * a */
void cmul_fvector(FVector c, float val, FVector a)
{
	long int i;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: cmul_fvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4, c4, val4;

//	val4 = _mm256_set_ps(val, val, val, val);
	val4 = _mm256_set1_ps(val);

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, val * get_fvector_i(a, i));

		a4 = _mm256_load_ps(&get_fvector_i(a, i));
		c4 = _mm256_mul_ps(val4, a4);
		_mm256_store_ps(&get_fvector_i(c, i), c4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a8, c8, val8;

	val8 = _mm512_set_ps(val, val, val, val, val, val, val, val, val, val, val, val, val, val, val, val);

	//for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, val * get_fvector_i(a, i));

		a8 = _mm512_load_ps(&get_fvector_i(a, i));
		c8 = _mm512_mul_ps(val8, a8);
		_mm512_store_ps(&get_fvector_i(c, i), c8);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t a4, c4, val4;

//	val4 = svdup_f32(val);
	val4 = svdup_f32(val);

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += (long int)svcntw())
	{
			svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(c->real_dim));
		//set_fvector_i(c, i, val * get_fvector_i(a, i));

		a4 = svld1_f32(pg, &get_fvector_i(a, i));
		c4 = svmul_f32_x(pg, val4, a4);
		svst1_f32(pg, &get_fvector_i(c, i), c4);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a4, c4, val4;

//	val4 = vdupq_n_f32(val);
	val4 = vdupq_n_f32(val);

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, val * get_fvector_i(a, i));

		a4 = vld1q_f32(&get_fvector_i(a, i));
		c4 = vmulq_f32(val4, a4);
		vst1q_f32(&get_fvector_i(c, i), c4);
	}
#else // others
	for(i = 0; i < c->dim; i++)
		set_fvector_i(c, i, val * get_fvector_i(a, i));
#endif // __AVX2__

}

/* c *= val */
void cmul2_fvector(FVector c, float val)
{
	long int i;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 c4, val4;

	val4 = _mm256_set_ps(val, val, val, val, val, val, val, val);

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, val * get_fvector_i(a, i));

		c4 = _mm256_load_ps(&get_fvector_i(c, i));
		c4 = _mm256_mul_ps(val4, c4);
		_mm256_store_ps(&get_fvector_i(c, i), c4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 c8, val8;

	val8 = _mm512_set_ps(val, val, val, val, val, val, val, val, val, val, val, val, val, val, val, val);

	//for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, val * get_fvector_i(a, i));

		c8 = _mm512_load_ps(&get_fvector_i(c, i));
		c8 = _mm512_mul_ps(val8, c8);
		_mm512_store_ps(&get_fvector_i(c, i), c8);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t c4, val4;

	val4 = svdup_f32(val);

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += (long int)svcntw())
	{
			svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(c->real_dim));
		//set_fvector_i(c, i, val * get_fvector_i(a, i));

		c4 = svld1_f32(pg, &get_fvector_i(c, i));
		c4 = svmul_f32_x(pg, val4, c4);
		svst1_f32(pg, &get_fvector_i(c, i), c4);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t c4, val4;

	val4 = vdupq_n_f32(val);

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, val * get_fvector_i(a, i));

		c4 = vld1q_f32(&get_fvector_i(c, i));
		c4 = vmulq_f32(val4, c4);
		vst1q_f32(&get_fvector_i(c, i), c4);
	}
#else // others
	for(i = 0; i < c->dim; i++)
		set_fvector_i(c, i, val * get_fvector_i(c, i));
#endif // __AVX2__

}

/* c = a + val * b */
void add_cmul_fvector(FVector c, FVector a, float val, FVector b)
{
	long int i;
	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_cmul_fvector\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4, b4, c4, val4;
	val4 = _mm256_set1_ps(val);
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		a4 = _mm256_load_ps(&get_fvector_i(a, i));
		b4 = _mm256_load_ps(&get_fvector_i(b, i));
		c4 = _mm256_fmadd_ps(val4, b4, a4);
		_mm256_store_ps(&get_fvector_i(c, i), c4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a8, b8, c8, val8;
	val8 = _mm512_set1_ps(val);
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		a8 = _mm512_load_ps(&get_fvector_i(a, i));
		b8 = _mm512_load_ps(&get_fvector_i(b, i));
		c8 = _mm512_fmadd_ps(val8, b8, a8);
		_mm512_store_ps(&get_fvector_i(c, i), c8);
	}
#elif defined(__ARM_SVE2_BROKEN_LINEAR_DISABLED) // Arm SVE2 (disabled: falls through to NEON)
	svfloat32_t a2, b2, c2, v2;
	v2 = svdup_f32(val);
	for(i = 0; i < c->real_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s32((int)i, (int)c->real_dim);
		a2 = svld1_f32(pg, &get_fvector_i(a, i));
		b2 = svld1_f32(pg, &get_fvector_i(b, i));
		c2 = svmad_f32_x(pg, a2, v2, b2);
		svst1_f32(pg, &get_fvector_i(c, i), c2);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a2, b2, c2, v2;
	v2 = vdupq_n_f32(val);
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		a2 = vld1q_f32(&get_fvector_i(a, i));
		b2 = vld1q_f32(&get_fvector_i(b, i));
		c2 = vfmaq_f32(a2, v2, b2);
		vst1q_f32(&get_fvector_i(c, i), c2);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t a4, b4, c4, val4;
	val4 = svdup_f32(val);
	for(i = 0; i < c->real_dim; i += (long int)svcntw())
	{
			svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(c->real_dim));
		a4 = svld1_f32(pg, &get_fvector_i(a, i));
		b4 = svld1_f32(pg, &get_fvector_i(b, i));
		c4 = svmla_f32_m(pg, a4, val4, b4);
		svst1_f32(pg, &get_fvector_i(c, i), c4);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a4, b4, c4, val4;
	val4 = vdupq_n_f32(val);
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		a4 = vld1q_f32(&get_fvector_i(a, i));
		b4 = vld1q_f32(&get_fvector_i(b, i));
		c4 = vfmaq_f32(a4, val4, b4);
		vst1q_f32(&get_fvector_i(c, i), c4);
	}
#else // scalar
	for(i = 0; i < c->dim; i++)
		set_fvector_i(c, i, get_fvector_i(a, i) + val * get_fvector_i(b, i));
#endif // __AVX2__
}

void sub_cmul_fvector(FVector c, FVector a, float val, FVector b)
{
	long int i;
	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_cmul_fvector\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4, b4, c4, val4;
	val4 = _mm256_set1_ps(val);
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		a4 = _mm256_load_ps(&get_fvector_i(a, i));
		b4 = _mm256_load_ps(&get_fvector_i(b, i));
		c4 = _mm256_fnmadd_ps(val4, b4, a4);
		_mm256_store_ps(&get_fvector_i(c, i), c4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a8, b8, c8, val8;
	val8 = _mm512_set1_ps(val);
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		a8 = _mm512_load_ps(&get_fvector_i(a, i));
		b8 = _mm512_load_ps(&get_fvector_i(b, i));
		c8 = _mm512_fnmadd_ps(val8, b8, a8);
		_mm512_store_ps(&get_fvector_i(c, i), c8);
	}
#elif defined(__ARM_SVE2_BROKEN_LINEAR_DISABLED) // Arm SVE2 (disabled: falls through to NEON)
	svfloat32_t a2, b2, c2, v2;
	v2 = svdup_f32(val);
	for(i = 0; i < c->real_dim; i += (long int)svcntw())
	{
		svbool_t pg = svwhilelt_b32_s32((int)i, (int)c->real_dim);
		a2 = svld1_f32(pg, &get_fvector_i(a, i));
		b2 = svld1_f32(pg, &get_fvector_i(b, i));
		c2 = svmsb_f32_x(pg, a2, v2, b2);
		svst1_f32(pg, &get_fvector_i(c, i), c2);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a2, b2, c2, v2;
	v2 = vdupq_n_f32(val);
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		a2 = vld1q_f32(&get_fvector_i(a, i));
		b2 = vld1q_f32(&get_fvector_i(b, i));
		c2 = vfmsq_f32(a2, v2, b2);
		vst1q_f32(&get_fvector_i(c, i), c2);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t a4, b4, c4, val4;
	val4 = svdup_f32(val);
	for(i = 0; i < c->real_dim; i += (long int)svcntw())
	{
			svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(c->real_dim));
		a4 = svld1_f32(pg, &get_fvector_i(a, i));
		b4 = svld1_f32(pg, &get_fvector_i(b, i));
		c4 = svmls_f32_m(pg, a4, val4, b4);
		svst1_f32(pg, &get_fvector_i(c, i), c4);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a4, b4, c4, val4;
	val4 = vdupq_n_f32(val);
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		a4 = vld1q_f32(&get_fvector_i(a, i));
		b4 = vld1q_f32(&get_fvector_i(b, i));
		c4 = vfmsq_f32(a4, val4, b4);
		vst1q_f32(&get_fvector_i(c, i), c4);
	}
#else // scalar
	for(i = 0; i < c->dim; i++)
		set_fvector_i(c, i, get_fvector_i(a, i) - val * get_fvector_i(b, i));
#endif // __AVX2__
}

/* (a, b) */
float ip_fvector(FVector a, FVector b)
{
	float ret = 0.0;
	long int i;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: ip_fvector\n");
		return 0;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4, b4, tmp4;

	// tmp4 ;= 0
	tmp4 = _mm256_setzero_ps();

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//tmp += get_fvector_i(a, i) * get_fvector_i(b, i);

		a4 = _mm256_load_ps(&get_fvector_i(a, i));
		b4 = _mm256_load_ps(&get_fvector_i(b, i));

		// tmp4 += a4 * b4
		tmp4 = _mm256_fmadd_ps(a4, b4, tmp4);
	}

	ret = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3] + tmp4[4] + tmp4[5] + tmp4[6] + tmp4[7];

#elif defined(__AVX512F__) // __AVX512F__
	__m512 a8, b8, tmp8;

	// tmp8 := 0
	tmp8 = _mm512_setzero_ps();

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//tmp += get_fvector_i(a, i) * get_fvector_i(b, i);

		a8 = _mm512_load_ps(&get_fvector_i(a, i));
		b8 = _mm512_load_ps(&get_fvector_i(b, i));

		// c8 += a8 * b8
		tmp8 = _mm512_fmadd_ps(a8, b8, tmp8);

	}

	ret = tmp8[0] + tmp8[1] + tmp8[2] + tmp8[3] + tmp8[4] + tmp8[5] + tmp8[6] + tmp8[7] + tmp8[8] + tmp8[9] + tmp8[10] + tmp8[11] + tmp8[12] + tmp8[13] + tmp8[14] + tmp8[15];

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t a4, b4, tmp4;

	// tmp4 ;= 0
	tmp4 = svdup_f32(0.0f);

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += (long int)svcntw())
	{
			svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(a->real_dim));
		//tmp += get_fvector_i(a, i) * get_fvector_i(b, i);

		a4 = svld1_f32(pg, &get_fvector_i(a, i));
		b4 = svld1_f32(pg, &get_fvector_i(b, i));

		// tmp4 += a4 * b4
		tmp4 = svmla_f32_m(pg, tmp4, a4, b4);
	}

	ret = svaddv_f32(svptrue_b32(), tmp4);

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a4, b4, tmp4;

	// tmp4 ;= 0
	tmp4 = vdupq_n_f32(0.0f);

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//tmp += get_fvector_i(a, i) * get_fvector_i(b, i);

		a4 = vld1q_f32(&get_fvector_i(a, i));
		b4 = vld1q_f32(&get_fvector_i(b, i));

		// tmp4 += a4 * b4
		tmp4 = vfmaq_f32(tmp4, a4, b4);
	}

	ret = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3];

#else // others
	for(i = 0; i < a->dim; i++)
		ret += get_fvector_i(a, i) * get_fvector_i(b, i);
#endif // __AVX2__

	return ret;
}


/* ||a||_1 */
float norm1_fvector(FVector a)
{
	float ret = 0.0;
	long int i;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4, tmp4;

	// tmp4 ;= 0
	tmp4 = _mm256_setzero_ps();

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//ret += fabsf(get_fvector_i(a, i));

		a4 = _bncavx2_fabsf(_mm256_load_ps(&get_fvector_i(a, i)));

		// tmp4 += fabsf(a4)
		tmp4 = _mm256_add_ps(a4, tmp4);
	}

	ret = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3] + tmp4[4] + tmp4[5] + tmp4[6] + tmp4[7];

#elif defined(__AVX512F__) // __AVX512F__
	__m512 a8, tmp8;

	// tmp8 := 0
	tmp8 = _mm512_setzero_ps();

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//ret += fabsf(get_fvector_i(a, i));

		a8 = _bncavx512_fabsf(_mm512_load_ps(&get_fvector_i(a, i)));

		// c8 += fabsf(a8)
		tmp8 = _mm512_add_ps(a8, tmp8);

	}

	ret = tmp8[0] + tmp8[1] + tmp8[2] + tmp8[3] + tmp8[4] + tmp8[5] + tmp8[6] + tmp8[7] + tmp8[8] + tmp8[9] + tmp8[10] + tmp8[11] + tmp8[12] + tmp8[13] + tmp8[14] + tmp8[15];

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t a4, tmp4;

	// tmp4 ;= 0
	tmp4 = svdup_f32(0.0f);

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += (long int)svcntw())
	{
			svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(a->real_dim));
		//ret += fabsf(get_fvector_i(a, i));

		a4 = svabs_f32_x(pg, svld1_f32(pg, &get_fvector_i(a, i)));

		// tmp4 += fabsf(a4)
		tmp4 = svadd_f32_x(pg, a4, tmp4);
	}

	ret = svaddv_f32(svptrue_b32(), tmp4);

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a4, tmp4;

	// tmp4 ;= 0
	tmp4 = vdupq_n_f32(0.0f);

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//ret += fabsf(get_fvector_i(a, i));

		a4 = vabsq_f32(vld1q_f32(&get_fvector_i(a, i)));

		// tmp4 += fabsf(a4)
		tmp4 = vaddq_f32(a4, tmp4);
	}

	ret = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3];

#else // others
	for(i = 0; i < a->dim; i++)
		ret += fabsf(get_fvector_i(a, i));
#endif // __AVX2__

	return ret;
}

/* ||a||_2 */
float norm2_fvector(FVector a)
{
	float ret = 0.0;
	long int i;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4, tmp4;

	// tmp4 ;= 0
	tmp4 = _mm256_setzero_ps();

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//ret += get_fvector_i(a, i) * get_fvector_i(a, i);

		a4 = _mm256_load_ps(&get_fvector_i(a, i));

		// tmp4 += a4 * a4
		tmp4 = _mm256_fmadd_ps(a4, a4, tmp4);
	}

	ret = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3] + tmp4[4] + tmp4[5] + tmp4[6] + tmp4[7];

#elif defined(__AVX512F__) // __AVX512F__
	__m512 a8, b8, tmp8;

	// tmp8 := 0
	tmp8 = _mm512_setzero_ps();

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//ret += get_fvector_i(a, i) * get_fvector_i(a, i);

		a8 = _mm512_load_ps(&get_fvector_i(a, i));

		// c8 += a8 * a8
		tmp8 = _mm512_fmadd_ps(a8, a8, tmp8);

	}

	ret = tmp8[0] + tmp8[1] + tmp8[2] + tmp8[3] + tmp8[4] + tmp8[5] + tmp8[6] + tmp8[7] + tmp8[8] + tmp8[9] + tmp8[10] + tmp8[11] + tmp8[12] + tmp8[13] + tmp8[14] + tmp8[15];

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t a4, tmp4;

	// tmp4 ;= 0
	tmp4 = svdup_f32(0.0f);

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += (long int)svcntw())
	{
			svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(a->real_dim));
		//ret += get_fvector_i(a, i) * get_fvector_i(a, i);

		a4 = svld1_f32(pg, &get_fvector_i(a, i));

		// tmp4 += a4 * a4
		tmp4 = svmla_f32_m(pg, tmp4, a4, a4);
	}

	ret = svaddv_f32(svptrue_b32(), tmp4);

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a4, tmp4;

	// tmp4 ;= 0
	tmp4 = vdupq_n_f32(0.0f);

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//ret += get_fvector_i(a, i) * get_fvector_i(a, i);

		a4 = vld1q_f32(&get_fvector_i(a, i));

		// tmp4 += a4 * a4
		tmp4 = vfmaq_f32(tmp4, a4, a4);
	}

	ret = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3];

#else // others
	for(i = 0; i < a->dim; i++)
		ret += get_fvector_i(a, i) * get_fvector_i(a, i);
#endif // __AVX2__

	return sqrt(ret);
}

/* ||a||_infty */
float normi_fvector(FVector a)
{
	float ret, tmp;
	long int i;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4, ret4, tmp4;

	// tmp4 ;= 0
	ret4 = _mm256_setzero_ps();

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//tmp = fabsf(get_fvector_i(a, i));
		//if(ret < tmp)
		//	ret = tmp;

		tmp4 = _bncavx2_fabsf(_mm256_load_ps(&get_fvector_i(a, i)));

		// ret4 := max(ret4, tmp4)
		ret4 = _mm256_max_ps(ret4, tmp4);
	}

	ret = ret4[0];
	if(ret < ret4[1]) ret = ret4[1];
	if(ret < ret4[2]) ret = ret4[2];
	if(ret < ret4[3]) ret = ret4[3];

#elif defined(__AVX512F__) // __AVX512F__
	__m512 a8, ret8, tmp8;

	// tmp8 := 0
	tmp8 = _mm512_setzero_ps();

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//if(ret < tmp)
		//	ret = tmp;

		tmp8 = _bncavx512_fabsf(_mm512_load_ps(&get_fvector_i(a, i)));

		// ret8 := max(ret8, tmp8)
		ret8 = _mm512_max_ps(ret8, tmp8);
	}

	ret = ret8[0];
	if(ret < ret8[1]) ret = ret8[1];
	if(ret < ret8[2]) ret = ret8[2];
	if(ret < ret8[3]) ret = ret8[3];
	if(ret < ret8[4]) ret = ret8[4];
	if(ret < ret8[5]) ret = ret8[5];
	if(ret < ret8[6]) ret = ret8[6];
	if(ret < ret8[7]) ret = ret8[7];

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t a4, ret4, tmp4;

	// tmp4 ;= 0
	ret4 = svdup_f32(0.0f);

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += (long int)svcntw())
	{
			svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(a->real_dim));
		//tmp = fabsf(get_fvector_i(a, i));
		//if(ret < tmp)
		//	ret = tmp;

		tmp4 = svabs_f32_x(pg, svld1_f32(pg, &get_fvector_i(a, i)));

		// ret4 := max(ret4, tmp4)
		ret4 = svmax_f32_x(pg, ret4, tmp4);
	}

	ret = svmaxv_f32(svptrue_b32(), ret4);

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a4, ret4, tmp4;

	// tmp4 ;= 0
	ret4 = vdupq_n_f32(0.0f);

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//tmp = fabsf(get_fvector_i(a, i));
		//if(ret < tmp)
		//	ret = tmp;

		tmp4 = vabsq_f32(vld1q_f32(&get_fvector_i(a, i)));

		// ret4 := max(ret4, tmp4)
		ret4 = vmaxq_f32(ret4, tmp4);
	}

	ret = ret4[0];
	if(ret < ret4[1]) ret = ret4[1];
	if(ret < ret4[2]) ret = ret4[2];
	if(ret < ret4[3]) ret = ret4[3];

#else // others
	ret = fabsf(get_fvector_i(a, 0));
	for(i = 1; i < a->dim; i++)
	{
		tmp = fabsf(get_fvector_i(a, i));
		if(ret < tmp)
			ret = tmp;
	}
#endif // __AVX2__

	return ret;
}

/* c := a */
void subst_fvector(FVector c, FVector a)
{
	long int i;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
//	for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, get_fvector_i(a, i));
		_mm256_store_ps(&get_fvector_i(c, i), _mm256_load_ps(&get_fvector_i(a, i)));
	}
#elif defined(__AVX512F__) // __AVX512F__
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, get_fvector_i(a, i));
		_mm512_store_ps(&get_fvector_i(c, i), _mm512_load_ps(&get_fvector_i(a, i)));
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
//	for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += (long int)svcntw())
	{
			svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(a->real_dim));
		//set_fvector_i(c, i, get_fvector_i(a, i));
		svst1_f32(pg, &get_fvector_i(c, i), svld1_f32(pg, &get_fvector_i(a, i)));
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
//	for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, get_fvector_i(a, i));
		vst1q_f32(&get_fvector_i(c, i), vld1q_f32(&get_fvector_i(a, i)));
	}
#else // others
	for(i = 0; i < a->dim; i++)
		set_fvector_i(c, i, get_fvector_i(a, i));
#endif // __AVX2__
}

/* c := 0 */
void set0_fvector(FVector c)
{
	long int i;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 zero4;

	zero4 = _mm256_setzero_ps();

	//for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, (float)0);
		_mm256_store_ps(&get_fvector_i(c, i), zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero8;

	zero8 = _mm512_setzero_ps();

	//for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, (float)0);
		_mm512_store_ps(&get_fvector_i(c, i), zero8);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t zero4;

	zero4 = svdup_f32(0.0f);

	//for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += (long int)svcntw())
	{
			svbool_t pg = svwhilelt_b32_s64((int64_t)i, (int64_t)(c->real_dim));
		//set_fvector_i(c, i, (float)0);
		svst1_f32(pg, &get_fvector_i(c, i), zero4);
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t zero4;

	zero4 = vdupq_n_f32(0.0f);

	//for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_S_WIDTH)
	{
		//set_fvector_i(c, i, (float)0);
		vst1q_f32(&get_fvector_i(c, i), zero4);
	}
#else // others
	for(i = 0; i < c->dim; i++)
		set_fvector_i(c, i, (float)0);
#endif // __AVX2__
}

/* append 2005.07/12 */
/*
	ret(index_start) = src(src_index_start)
	 ...
	ret(index_end  ) = src(src_index_end)
*/
void copy_fvector_ij(FVector ret, long int index_start, long int index_end, FVector src, long int src_index_start, long int src_index_end)
{
	long int i, itmp;

	if((src_index_end - src_index_start) != (index_end - index_start))
	{
		fprintf(stderr, "Invalid index!(copy_fvector_ij)\n");
		return;
	}

	for(i = 0; i <= (index_end - index_start); i++)
	{
		set_fvector_i(ret, index_start + i, get_fvector_i(src, src_index_start + i));
//		printf("%d <----------------------------------> %d\n", index_start + i, src_index_start + i);
	}
}


// old
//#define get_fmatrix_ij(mat, i, j) ( *((mat)->element + (i) * (mat)->col_dim + (j)) )
//#define set_fmatrix_ij(mat, i, j, val) ( *((mat)->element + (i) * (mat)->col_dim + (j)) = (val) )

// new
#define get_fmatrix_ij(mat, i, j) ( *((mat)->element + (i) * (mat)->real_col_dim + (j)) )
#define set_fmatrix_ij(mat, i, j, val) ( *((mat)->element + (i) * (mat)->real_col_dim + (j)) = (val) )


/*************************************************/
/* Matrix Caluculations for FMatrix              */
/*
FMatrix init_fmatrix(long int row_dimension, long int col_dimension)
void free_fmatrix(FMatrix mat)
float normf_fmatrix(FMatrix mat)
float normi_fmatrix(FMatrix mat)
float norm1_fmatrix(FMatrix mat)
void add_fmatrix(FMatrix c, FMatrix a, FMatrix b);
void sub_fmatrix(FMatrix c, FMatrix a, FMatrix b);
void mul_fmatrix(FMatrix c, FMatrix a, FMatrix b);
void transpose_fmatrix(FMatrix c, FMatrix a);
void mul_fmatrix_dvec(FVector v, FMatrix a, FVector vb)
void mul_fmatrixt_dvec(FVector v, FMatrix a, FVector vb)
void inv_fmatrix(FMatrix a);
void subst_dmatrux(FMatrix c, FMatrix a);
*/
/*************************************************/
FMatrix init_fmatrix(long int row_dimension, long int col_dimension)
{
	FMatrix ret = NULL;
	long int i, j;
	long real_row_dim, real_col_dim;

	if(row_dimension <= 0 || col_dimension <= 0)
	{
		fprintf(stderr, "ERROR: init_fmatrix\n");
		return ret;
	}

	ret = (FMatrix)BNC_MALLOC(sizeof(fmatrix)); // was sizeof(FMatrix) (pointer size) -> heap overflow
	if(ret == NULL)
		return ret;

	//ret->element = (float *)BNC_CALLOC(row_dimension * col_dimension, sizeof(float));

	// real_dim is the nearest positive multiplier of _BNC_S_WIDTH
	real_row_dim = (long int)ceil((float)row_dimension / (float)_BNC_S_WIDTH) * _BNC_S_WIDTH;
	//real_row_dim = row_dimension; // row-major way
	real_col_dim = (long int)ceil((float)col_dimension / (float)_BNC_S_WIDTH) * _BNC_S_WIDTH;
	//printf("row, real_row, col, real_col = %ld, %ld, %ld, %ld\n", row_dimension, real_row_dim, col_dimension, real_col_dim);

	ret->element = (float *)BNC_CALLOC(real_row_dim * real_col_dim, sizeof(float));

	if(ret->element == NULL)
		return ret;

	/* All 0 */
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 zero4;

	zero4 = _mm256_setzero_ps();
	for(i = 0; i < real_row_dim; i++)
	{
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
			_mm256_store_ps((ret->element + i * real_col_dim + j), zero4);
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero8;

	zero8 = _mm512_setzero_ps();
	for(i = 0; i < real_row_dim; i++)
	{
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
			_mm512_store_ps((ret->element + i * real_col_dim + j), zero8);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t zero4;

	zero4 = svdup_f32(0.0f);
	for(i = 0; i < real_row_dim; i++)
	{
		for(j = 0; j < real_col_dim; j += (long int)svcntw()) {
			svbool_t pg = svwhilelt_b32_s64((int64_t)j, (int64_t)(real_col_dim));
			svst1_f32(pg, (ret->element + i * real_col_dim + j), zero4);
			}
	}

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t zero4;

	zero4 = vdupq_n_f32(0.0f);
	for(i = 0; i < real_row_dim; i++)
	{
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
			vst1q_f32((ret->element + i * real_col_dim + j), zero4);
	}

#else // others

	for(i = 0; i < real_row_dim; i++)
		for(j = 0; j < real_col_dim; j++)
			*(ret->element + i * real_col_dim + j) = (float)0.0;

#endif // __AVX2__

	ret->row_dim = row_dimension;
	ret->col_dim = col_dimension;

	ret->real_row_dim = real_row_dim;
	ret->real_col_dim = real_col_dim;


	return ret;
}

void free_fmatrix(FMatrix mat)
{
	if(mat == NULL)
		return;

	if(mat->element != NULL)
		free(mat->element);

	free(mat);
}

/*************************************************/
/* Matrix Caluculations for FMatrix              */
/*
float normf_fmatrix(FMatrix mat)
float normi_fmatrix(FMatrix mat)
float norm1_fmatrix(FMatrix mat)
void add_fmatrix(FMatrix c, FMatrix a, FMatrix b);
void sub_fmatrix(FMatrix c, FMatrix a, FMatrix b);
void mul_fmatrix(FMatrix c, FMatrix a, FMatrix b);
void transpose_fmatrix(FMatrix c, FMatrix a);
void mul_fmatrix_dvec(FVector v, FMatrix a, FVector vb)
void mul_fmatrixt_dvec(FVector v, FMatrix a, FVector vb)
void inv_fmatrix(FMatrix a);
void subst_dmatrux(FMatrix c, FMatrix a);
*/
/*************************************************/
/* Frobenius Norm of Matrix */
float normf_fmatrix(FMatrix mat)
{
	long int i, j;
	float ret;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 mat4, tmp4;

	// tmp4 ;= 0
	tmp4 = _mm256_setzero_ps();

	for(i = 0; i < mat->row_dim; i++)
	{
		//for(j = 0; j < mat->col_dim; j++)
		for(j = 0; j < mat->real_col_dim; j += _BNC_S_WIDTH)
		{
			//ret += (get_fmatrix_ij(mat, i, j) * get_fmatrix_ij(mat, i, j));

			mat4 = _mm256_load_ps(&get_fmatrix_ij(mat, i, j));

			// tmp4 += mat4 * mat4
			tmp4 = _mm256_fmadd_ps(mat4, mat4, tmp4);
		}
	}

	ret = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3] + tmp4[4] + tmp4[5] + tmp4[6] + tmp4[7];

#elif defined(__AVX512F__) // __AVX512F__
	__m512 mat8, tmp8;

	// tmp4 ;= 0
	tmp8 = _mm512_setzero_ps();

	for(i = 0; i < mat->row_dim; i++)
	{
		//for(j = 0; j < mat->col_dim; j++)
		for(j = 0; j < mat->real_col_dim; j += _BNC_S_WIDTH)
		{
			//ret += (get_fmatrix_ij(mat, i, j) * get_fmatrix_ij(mat, i, j));

			mat8 = _mm512_load_ps(&get_fmatrix_ij(mat, i, j));

			// tmp8 += mat8 * mat8
			tmp8 = _mm512_fmadd_ps(mat8, mat8, tmp8);
		}
	}

	ret = tmp8[0] + tmp8[1] + tmp8[2] + tmp8[3] + tmp8[4] + tmp8[5] + tmp8[6] + tmp8[7] + tmp8[8] + tmp8[9] + tmp8[10] + tmp8[11] + tmp8[12] + tmp8[13] + tmp8[14] + tmp8[15];

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t mat4, tmp4;

	// tmp4 ;= 0
	tmp4 = svdup_f32(0.0f);

	for(i = 0; i < mat->row_dim; i++)
	{
		//for(j = 0; j < mat->col_dim; j++)
		for(j = 0; j < mat->real_col_dim; j += (long int)svcntw())
		{
			svbool_t pg = svwhilelt_b32_s64((int64_t)j, (int64_t)(mat->real_col_dim));
			//ret += (get_fmatrix_ij(mat, i, j) * get_fmatrix_ij(mat, i, j));

			mat4 = svld1_f32(pg, &get_fmatrix_ij(mat, i, j));

			// tmp4 += mat4 * mat4
			tmp4 = svmla_f32_m(pg, tmp4, mat4, mat4);
		}
	}

	ret = svaddv_f32(svptrue_b32(), tmp4);

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t mat4, tmp4;

	// tmp4 ;= 0
	tmp4 = vdupq_n_f32(0.0f);

	for(i = 0; i < mat->row_dim; i++)
	{
		//for(j = 0; j < mat->col_dim; j++)
		for(j = 0; j < mat->real_col_dim; j += _BNC_S_WIDTH)
		{
			//ret += (get_fmatrix_ij(mat, i, j) * get_fmatrix_ij(mat, i, j));

			mat4 = vld1q_f32(&get_fmatrix_ij(mat, i, j));

			// tmp4 += mat4 * mat4
			tmp4 = vfmaq_f32(tmp4, mat4, mat4);
		}
	}

	ret = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3];

#else // others
	ret = 0.0;
	for(i = 0; i < mat->row_dim; i++)
		for(j = 0; j < mat->col_dim; j++)
			ret += (get_fmatrix_ij(mat, i, j) * get_fmatrix_ij(mat, i, j));
#endif // __AVX2__

	ret = sqrt(ret);

	return ret;
}

/* Frobenius Norm of Matrix: array type */
float normf_fmatrix_array(float mat[], int row_dim, int col_dim)
{
	int i, j;
	float ret;

	ret = 0.0;
	for(i = 0; i < row_dim; i++)
		for(j = 0; j < col_dim; j++)
			ret += mat[i * col_dim + j] * mat[i * col_dim + j];

	ret = sqrt(ret);

	return ret;
}

/* Infinity Norm of Matrix */
float normi_fmatrix(FMatrix mat)
{
	long int i, j;
	float ret, sum;

	ret = 0.0;
	for(i = 0; i < mat->row_dim; i++)
	{
		sum = 0.0;
		for(j = 0; j < mat->col_dim; j++)
			sum += fabsf(get_fmatrix_ij(mat, i, j));
		if(ret < sum)
			ret = sum;
	}

	return ret;
}

/* 1 Norm of Matrix */
float norm1_fmatrix(FMatrix mat)
{
	long int i, j;
	float ret, sum;

	ret = 0.0;
	for(j = 0; j < mat->col_dim; j++)
	{
		sum = 0.0;
		for(i = 0; i < mat->row_dim; i++)
			sum += fabsf(get_fmatrix_ij(mat, i, j));
		if(ret < sum)
			ret = sum;
	}

	return ret;
}

/* c = a + b */
void add_fmatrix(FMatrix c, FMatrix a, FMatrix b)
{
	long int i, j, row_dim, col_dim, real_col_dim;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_fmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_fmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4, b4, tmp4;

	for(i = 0; i < row_dim; i++)
	{
//		for(j = 0; j < col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			a4 = _mm256_load_ps(&get_fmatrix_ij(a, i, j));
			b4 = _mm256_load_ps(&get_fmatrix_ij(b, i, j));
			//set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, i, j) + get_fmatrix_ij(b, i, j));
			tmp4 = _mm256_add_ps(a4, b4);
			_mm256_store_ps(&get_fmatrix_ij(c, i, j), tmp4);
		}
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512 a8, b8, tmp8;

	for(i = 0; i < row_dim; i++)
	{
//		for(j = 0; j < col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			a8 = _mm512_load_ps(&get_fmatrix_ij(a, i, j));
			b8 = _mm512_load_ps(&get_fmatrix_ij(b, i, j));
			//set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, i, j) + get_fmatrix_ij(b, i, j));
			tmp8 = _mm512_add_ps(a8, b8);
			_mm512_store_ps(&get_fmatrix_ij(c, i, j), tmp8);
		}
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t a4, b4, tmp4;

	for(i = 0; i < row_dim; i++)
	{
//		for(j = 0; j < col_dim; j++)
		for(j = 0; j < real_col_dim; j += (long int)svcntw())
		{
			svbool_t pg = svwhilelt_b32_s64((int64_t)j, (int64_t)(real_col_dim));
			a4 = svld1_f32(pg, &get_fmatrix_ij(a, i, j));
			b4 = svld1_f32(pg, &get_fmatrix_ij(b, i, j));
			//set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, i, j) + get_fmatrix_ij(b, i, j));
			tmp4 = svadd_f32_x(pg, a4, b4);
			svst1_f32(pg, &get_fmatrix_ij(c, i, j), tmp4);
		}
	}

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a4, b4, tmp4;

	for(i = 0; i < row_dim; i++)
	{
//		for(j = 0; j < col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			a4 = vld1q_f32(&get_fmatrix_ij(a, i, j));
			b4 = vld1q_f32(&get_fmatrix_ij(b, i, j));
			//set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, i, j) + get_fmatrix_ij(b, i, j));
			tmp4 = vaddq_f32(a4, b4);
			vst1q_f32(&get_fmatrix_ij(c, i, j), tmp4);
		}
	}

#else // others
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
			set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, i, j) + get_fmatrix_ij(b, i, j));
	}
#endif // __AVX2__

}

/* c = a - b */
void sub_fmatrix(FMatrix c, FMatrix a, FMatrix b)
{
	long int i, j, row_dim, col_dim, real_col_dim;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: sub_fmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: sub_fmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4, b4, tmp4;

	for(i = 0; i < row_dim; i++)
	{
//		for(j = 0; j < col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			a4 = _mm256_load_ps(&get_fmatrix_ij(a, i, j));
			b4 = _mm256_load_ps(&get_fmatrix_ij(b, i, j));
			//set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, i, j) - get_fmatrix_ij(b, i, j));
			tmp4 = _mm256_sub_ps(a4, b4);
			_mm256_store_ps(&get_fmatrix_ij(c, i, j), tmp4);
		}
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512 a8, b8, tmp8;

	for(i = 0; i < row_dim; i++)
	{
//		for(j = 0; j < col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			a8 = _mm512_load_ps(&get_fmatrix_ij(a, i, j));
			b8 = _mm512_load_ps(&get_fmatrix_ij(b, i, j));
			//set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, i, j) - get_fmatrix_ij(b, i, j));
			tmp8 = _mm512_sub_ps(a8, b8);
			_mm512_store_ps(&get_fmatrix_ij(c, i, j), tmp8);
		}
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t a4, b4, tmp4;

	for(i = 0; i < row_dim; i++)
	{
//		for(j = 0; j < col_dim; j++)
		for(j = 0; j < real_col_dim; j += (long int)svcntw())
		{
			svbool_t pg = svwhilelt_b32_s64((int64_t)j, (int64_t)(real_col_dim));
			a4 = svld1_f32(pg, &get_fmatrix_ij(a, i, j));
			b4 = svld1_f32(pg, &get_fmatrix_ij(b, i, j));
			//set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, i, j) - get_fmatrix_ij(b, i, j));
			tmp4 = svsub_f32_x(pg, a4, b4);
			svst1_f32(pg, &get_fmatrix_ij(c, i, j), tmp4);
		}
	}

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a4, b4, tmp4;

	for(i = 0; i < row_dim; i++)
	{
//		for(j = 0; j < col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			a4 = vld1q_f32(&get_fmatrix_ij(a, i, j));
			b4 = vld1q_f32(&get_fmatrix_ij(b, i, j));
			//set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, i, j) - get_fmatrix_ij(b, i, j));
			tmp4 = vsubq_f32(a4, b4);
			vst1q_f32(&get_fmatrix_ij(c, i, j), tmp4);
		}
	}

#else // others
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
			set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, i, j) - get_fmatrix_ij(b, i, j));
	}
#endif // __AVX2__
}

/* c = sc * a */
void cmul_fmatrix(FMatrix c, float sc, FMatrix a)
{
	long int i, j, row_dim, col_dim, real_col_dim;

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_fmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_fmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4, sc4, c4;

	sc4 = _mm256_set_ps(sc, sc, sc, sc, sc, sc, sc, sc);

	for(i = 0; i < row_dim; i++)
	{
//		for(j = 0; j < col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			//set_fmatrix_ij(c, i, j, sc * get_fmatrix_ij(a, i, j));
			a4 = _mm256_load_ps(&get_fmatrix_ij(a, i, j));
			c4 = _mm256_mul_ps(sc4, a4);
			_mm256_store_ps(&get_fmatrix_ij(c, i, j), c4);
		}
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512 a8, sc8, c8;

	sc8 = _mm512_set_ps(sc, sc, sc, sc, sc, sc, sc, sc, sc, sc, sc, sc, sc, sc, sc, sc);

	for(i = 0; i < row_dim; i++)
	{
//		for(j = 0; j < col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			//set_fmatrix_ij(c, i, j, sc * get_fmatrix_ij(a, i, j));
			a8 = _mm512_load_ps(&get_fmatrix_ij(a, i, j));
			c8 = _mm512_mul_ps(sc8, a8);
			_mm512_store_ps(&get_fmatrix_ij(c, i, j), c8);
		}
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t a4, sc4, c4;

	sc4 = svdup_f32(sc);

	for(i = 0; i < row_dim; i++)
	{
//		for(j = 0; j < col_dim; j++)
		for(j = 0; j < real_col_dim; j += (long int)svcntw())
		{
			svbool_t pg = svwhilelt_b32_s64((int64_t)j, (int64_t)(real_col_dim));
			//set_fmatrix_ij(c, i, j, sc * get_fmatrix_ij(a, i, j));
			a4 = svld1_f32(pg, &get_fmatrix_ij(a, i, j));
			c4 = svmul_f32_x(pg, sc4, a4);
			svst1_f32(pg, &get_fmatrix_ij(c, i, j), c4);
		}
	}

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a4, sc4, c4;

	sc4 = vdupq_n_f32(sc);

	for(i = 0; i < row_dim; i++)
	{
//		for(j = 0; j < col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			//set_fmatrix_ij(c, i, j, sc * get_fmatrix_ij(a, i, j));
			a4 = vld1q_f32(&get_fmatrix_ij(a, i, j));
			c4 = vmulq_f32(sc4, a4);
			vst1q_f32(&get_fmatrix_ij(c, i, j), c4);
		}
	}

#else // others
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
			set_fmatrix_ij(c, i, j, sc * get_fmatrix_ij(a, i, j));
	}
#endif // __AVX2__
}

/* c = a * b */
void mul_fmatrix(FMatrix c, FMatrix a, FMatrix b)
{
	long int i, j, k;
	float tmp;

	/* dimension check */
	if((c->row_dim != a->row_dim) || (c->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: mul_fmatrix\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4, b4, tmp4;

	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
		{
			//tmp = 0.0;
			tmp4 = _mm256_setzero_ps();

			//for(k = 0; k < a->col_dim; k++)
			for(k = 0; k < a->real_col_dim; k += _BNC_S_WIDTH)
			{
				a4 = _mm256_load_ps(&get_fmatrix_ij(a, i, k));
				b4 = _mm256_set_ps(
					get_fmatrix_ij(b, k + 7, j),
					get_fmatrix_ij(b, k + 6, j),
					get_fmatrix_ij(b, k + 5, j),
					get_fmatrix_ij(b, k + 4, j),
					get_fmatrix_ij(b, k + 3, j),
					get_fmatrix_ij(b, k + 2, j),
					get_fmatrix_ij(b, k + 1, j),
					get_fmatrix_ij(b, k    , j)
				);
				tmp4 = _mm256_fmadd_ps(a4, b4, tmp4);
			}
			set_fmatrix_ij(c, i, j, tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3] + tmp4[4] + tmp4[5] + tmp4[6] + tmp4[7]);
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a8, b8, tmp8;

	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
		{
			//tmp = 0.0;
			tmp8 = _mm512_setzero_ps();

			//for(k = 0; k < a->col_dim; k++)
			for(k = 0; k < a->real_col_dim; k += _BNC_S_WIDTH)
			{
				a8 = _mm512_load_ps(&get_fmatrix_ij(a, i, k));
				b8 = _mm512_set_ps(
					get_fmatrix_ij(b, k + 15, j),
					get_fmatrix_ij(b, k + 14, j),
					get_fmatrix_ij(b, k + 13, j),
					get_fmatrix_ij(b, k + 12, j),
					get_fmatrix_ij(b, k + 11, j),
					get_fmatrix_ij(b, k + 10, j),
					get_fmatrix_ij(b, k + 9, j),
					get_fmatrix_ij(b, k + 8, j),
					get_fmatrix_ij(b, k + 7, j),
					get_fmatrix_ij(b, k + 6, j),
					get_fmatrix_ij(b, k + 5, j),
					get_fmatrix_ij(b, k + 4, j),
					get_fmatrix_ij(b, k + 3, j),
					get_fmatrix_ij(b, k + 2, j),
					get_fmatrix_ij(b, k + 1, j),
					get_fmatrix_ij(b, k    , j)
				);
				tmp8 = _mm512_fmadd_ps(a8, b8, tmp8);
			}
			set_fmatrix_ij(c, i, j, tmp8[0] + tmp8[1] + tmp8[2] + tmp8[3] + tmp8[4] + tmp8[5] + tmp8[6] + tmp8[7] + tmp8[8] + tmp8[9] + tmp8[10] + tmp8[11] + tmp8[12] + tmp8[13] + tmp8[14] + tmp8[15]);
		}
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, gather)
	{
	svfloat32_t a_v, b_v, tmp;
	svint32_t bidx;
	for(i = 0; i < c->row_dim; i++)
		for(j = 0; j < c->col_dim; j++)
		{
			tmp = svdup_f32(0.0);
			for(k = 0; k < a->real_col_dim; k += (long int)svcntw())
			{
				svbool_t pg = svwhilelt_b32_s64((int64_t)k, (int64_t)a->real_col_dim);
				a_v = svld1_f32(pg, &get_fmatrix_ij(a, i, k));
				bidx = svindex_s32((int32_t)(k * b->real_col_dim + j), (int32_t)b->real_col_dim);
				b_v = svld1_gather_s32index_f32(pg, b->element, bidx);
				tmp = svmla_f32_m(pg, tmp, a_v, b_v);
			}
			set_fmatrix_ij(c, i, j, svaddv_f32(svptrue_b32(), tmp));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	{
	float32x4_t acc, av, bv;
	double sum; long int k;
	for(i = 0; i < c->row_dim; i++)
		for(j = 0; j < c->col_dim; j++)
		{
			acc = vdupq_n_f32(0.0);
			for(k = 0; k <= a->real_col_dim - 4; k += 4)
			{
				av = vld1q_f32(&get_fmatrix_ij(a, i, k));
				bv = (float32x4_t){ get_fmatrix_ij(b, k + 0, j), get_fmatrix_ij(b, k + 1, j), get_fmatrix_ij(b, k + 2, j), get_fmatrix_ij(b, k + 3, j) };
				acc = vfmaq_f32(acc, av, bv);
			}
			sum = vaddvq_f32(acc);
			for(; k < a->col_dim; k++) sum += get_fmatrix_ij(a, i, k) * get_fmatrix_ij(b, k, j);
			set_fmatrix_ij(c, i, j, sum);
		}
	}
#else // others
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
		{
			tmp = 0.0;
			for(k = 0; k < a->col_dim; k++)
				tmp += get_fmatrix_ij(a, i, k) * get_fmatrix_ij(b, k, j);
			set_fmatrix_ij(c, i, j, tmp);
		}
	}

#endif // __AVX2__
}

/* c = a^T */
void transpose_fmatrix(FMatrix c, FMatrix a)
{
	long int i, j;
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
	__m256 aji4;

	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			//set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, j, i));
			aji4 = _mm256_set_ps(
				get_fmatrix_ij(a, j + 7, i),
				get_fmatrix_ij(a, j + 6, i),
				get_fmatrix_ij(a, j + 5, i),
				get_fmatrix_ij(a, j + 4, i),
				get_fmatrix_ij(a, j + 3, i),
				get_fmatrix_ij(a, j + 2, i),
				get_fmatrix_ij(a, j + 1, i),
				get_fmatrix_ij(a, j    , i)
			);
			_mm256_store_ps(&get_fmatrix_ij(c, i, j), aji4);
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 aji8;

	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			//set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, j, i));
			aji8 = _mm512_set_ps(
				get_fmatrix_ij(a, j + 15, i),
				get_fmatrix_ij(a, j + 14, i),
				get_fmatrix_ij(a, j + 13, i),
				get_fmatrix_ij(a, j + 12, i),
				get_fmatrix_ij(a, j + 11, i),
				get_fmatrix_ij(a, j + 10, i),
				get_fmatrix_ij(a, j + 9, i),
				get_fmatrix_ij(a, j + 8, i),
				get_fmatrix_ij(a, j + 7, i),
				get_fmatrix_ij(a, j + 6, i),
				get_fmatrix_ij(a, j + 5, i),
				get_fmatrix_ij(a, j + 4, i),
				get_fmatrix_ij(a, j + 3, i),
				get_fmatrix_ij(a, j + 2, i),
				get_fmatrix_ij(a, j + 1, i),
				get_fmatrix_ij(a, j    , i)
			);
			_mm512_store_ps(&get_fmatrix_ij(c, i, j), aji8);
		}
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, gather)
	{
	svfloat32_t aji;
	svint32_t vidx;
	for(i = 0; i < real_row_dim; i++)
		for(j = 0; j < real_col_dim; j += (long int)svcntw())
		{
			svbool_t pg = svwhilelt_b32_s64((int64_t)j, (int64_t)real_col_dim);
			vidx = svindex_s32((int32_t)(j * a->real_col_dim + i), (int32_t)a->real_col_dim);
			aji = svld1_gather_s32index_f32(pg, a->element, vidx);
			svst1_f32(pg, &get_fmatrix_ij(c, i, j), aji);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	{
	for(i = 0; i < real_row_dim; i++)
		for(j = 0; j < real_col_dim; j++)
			set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, j, i));
	}
#else // others
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, j, i));
	}
#endif // __AVX2__	
}

/* c := a */
void subst_fmatrix(FMatrix c, FMatrix a)
{
	long int i, j;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_fmatrix\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	for(i = 0; i < a->row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < a->real_col_dim; j += _BNC_S_WIDTH)
		{
			//set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, i, j));
			_mm256_store_ps(&get_fmatrix_ij(c, i, j), _mm256_load_ps(&get_fmatrix_ij(a, i, j)));
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	for(i = 0; i < a->row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < a->real_col_dim; j += _BNC_S_WIDTH)
		{
			//set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, i, j));
			_mm512_store_ps(&get_fmatrix_ij(c, i, j), _mm512_load_ps(&get_fmatrix_ij(a, i, j)));
		}
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	for(i = 0; i < a->row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < a->real_col_dim; j += (long int)svcntw())
		{
			svbool_t pg = svwhilelt_b32_s64((int64_t)j, (int64_t)(a->real_col_dim));
			//set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, i, j));
			svst1_f32(pg, &get_fmatrix_ij(c, i, j), svld1_f32(pg, &get_fmatrix_ij(a, i, j)));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	for(i = 0; i < a->row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < a->real_col_dim; j += _BNC_S_WIDTH)
		{
			//set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, i, j));
			vst1q_f32(&get_fmatrix_ij(c, i, j), vld1q_f32(&get_fmatrix_ij(a, i, j)));
		}
	}
#else // others
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_fmatrix_ij(c, i, j, get_fmatrix_ij(a, i, j));
		}
	}
#endif // __AVX2__
}

/* c := 0 */
void set0_fmatrix(FMatrix c)
{
	long int index, total_dim;

	total_dim = c->real_row_dim * c->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 zero4;

	zero4 = _mm256_setzero_ps();
	for(index = 0; index < total_dim; index += _BNC_S_WIDTH)
		_mm256_store_ps(&(c->element[index]), zero4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero8;

	zero8 = _mm512_setzero_ps();
	for(index = 0; index < total_dim; index += _BNC_S_WIDTH)
		_mm512_store_ps(&(c->element[index]), zero8);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t zero4;

	zero4 = svdup_f32(0.0f);
	for(index = 0; index < total_dim; index += (long int)svcntw()) {
			svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(total_dim));
		svst1_f32(pg, &(c->element[index]), zero4);
		}

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t zero4;

	zero4 = vdupq_n_f32(0.0f);
	for(index = 0; index < total_dim; index += _BNC_S_WIDTH)
		vst1q_f32(&(c->element[index]), zero4);

#else // others
	for(index = 0; index < total_dim; index++)
		c->element[index] = 0.0;

#endif // __AVX2__
/*
	for(i = 0; i < c->real_row_dim; i++)
	{
		for(j = 0; j < c->real_col_dim; j++)
			set_fmatrix_ij(c, i, j, 0.0);
	}
*/
}

/* c := I */
void setI_fmatrix(FMatrix c)
{
	long int index, total_dim;

	total_dim = c->real_row_dim * c->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 zero4;

	zero4 = _mm256_setzero_ps();
	for(index = 0; index < total_dim; index += _BNC_S_WIDTH)
		_mm256_store_ps(&(c->element[index]), zero4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero8;

	zero8 = _mm512_setzero_ps();
	for(index = 0; index < total_dim; index += _BNC_S_WIDTH)
		_mm512_store_ps(&(c->element[index]), zero8);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t zero4;

	zero4 = svdup_f32(0.0f);
	for(index = 0; index < total_dim; index += (long int)svcntw()) {
			svbool_t pg = svwhilelt_b32_s64((int64_t)index, (int64_t)(total_dim));
		svst1_f32(pg, &(c->element[index]), zero4);
		}

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t zero4;

	zero4 = vdupq_n_f32(0.0f);
	for(index = 0; index < total_dim; index += _BNC_S_WIDTH)
		vst1q_f32(&(c->element[index]), zero4);

#else // others
	for(index = 0; index < total_dim; index++)
		c->element[index] = 0.0;

#endif // __AVX2__
	for(index = 0; index < c->row_dim; index++)
	{
		if(index < c->col_dim)
			set_fmatrix_ij(c, index, index, 1.0);
	}
}

/* v = a * vb */
void mul_fmatrix_dvec(FVector v, FMatrix a, FVector vb)
{
	long int i, j;
	float tmp;

	/* Check Dimension */
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: mul_fmatrix_dvec\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4, vb4, tmp4;

	//printf("v = %d, a->real_row_dim, real_col_dim = %d, %d, vb = %d\n", v->real_dim, a->real_row_dim, a->real_col_dim, vb->real_dim);
	for(i = 0; i < a->row_dim; i++)
	{
		//tmp = 0.0;
		tmp4 = _mm256_setzero_ps();

		//for(j = 0; j < a->row_dim; j++)
		for(j = 0; j < a->real_col_dim; j += _BNC_S_WIDTH)
		{
			//tmp += get_fmatrix_ij(a, i, j) * get_fvector_i(vb, j);

			a4  = _mm256_load_ps(&get_fmatrix_ij(a, i, j));
			vb4 = _mm256_load_ps(&get_fvector_i(vb, j));
			tmp4 = _mm256_fmadd_ps(a4, vb4, tmp4);
		}
		//set_fvector_i(v, i, tmp);
		//_mm256_store_ps(&get_fvector_i(v, i), tmp4);
		set_fvector_i(v, i, tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3] + tmp4[4] + tmp4[5] + tmp4[6] + tmp4[7]);

	}
	//printf("v = %d, a->real_row_dim, real_col_dim = %d, %d, vb = %d\n", v->real_dim, a->real_row_dim, a->real_col_dim, vb->real_dim);
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a8, vb8, tmp8;

	for(i = 0; i < a->row_dim; i++)
	{
		//tmp = 0.0;
		tmp8 = _mm512_setzero_ps();

		//for(j = 0; j < a->row_dim; j++)
		for(j = 0; j < a->real_col_dim; j += _BNC_S_WIDTH)
		{
			//tmp += get_fmatrix_ij(a, i, j) * get_fvector_i(vb, j);

			a8  = _mm512_load_ps(&get_fmatrix_ij(a, i, j));
			vb8 = _mm512_load_ps(&get_fvector_i(vb, j));
			tmp8 = _mm512_fmadd_ps(a8, vb8, tmp8);
		}
		//set_fvector_i(v, i, tmp);
		//_mm512_store_ps(&get_fvector_i(v, i), tmp8);
		set_fvector_i(v, i, tmp8[0] + tmp8[1] + tmp8[2] + tmp8[3] + tmp8[4] + tmp8[5] + tmp8[6] + tmp8[7] + tmp8[8] + tmp8[9] + tmp8[10] + tmp8[11] + tmp8[12] + tmp8[13] + tmp8[14] + tmp8[15]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat32_t a4, vb4, tmp4;

	//printf("v = %d, a->real_row_dim, real_col_dim = %d, %d, vb = %d\n", v->real_dim, a->real_row_dim, a->real_col_dim, vb->real_dim);
	for(i = 0; i < a->row_dim; i++)
	{
		//tmp = 0.0;
		tmp4 = svdup_f32(0.0f);

		//for(j = 0; j < a->row_dim; j++)
		for(j = 0; j < a->real_col_dim; j += (long int)svcntw())
		{
			svbool_t pg = svwhilelt_b32_s64((int64_t)j, (int64_t)(a->real_col_dim));
			//tmp += get_fmatrix_ij(a, i, j) * get_fvector_i(vb, j);

			a4  = svld1_f32(pg, &get_fmatrix_ij(a, i, j));
			vb4 = svld1_f32(pg, &get_fvector_i(vb, j));
			tmp4 = svmla_f32_m(pg, tmp4, a4, vb4);
		}
		//set_fvector_i(v, i, tmp);
		//svst1_f32(pg, &get_fvector_i(v, i), tmp4);
		set_fvector_i(v, i, svaddv_f32(svptrue_b32(), tmp4));

	}
	//printf("v = %d, a->real_row_dim, real_col_dim = %d, %d, vb = %d\n", v->real_dim, a->real_row_dim, a->real_col_dim, vb->real_dim);
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a4, vb4, tmp4;

	//printf("v = %d, a->real_row_dim, real_col_dim = %d, %d, vb = %d\n", v->real_dim, a->real_row_dim, a->real_col_dim, vb->real_dim);
	for(i = 0; i < a->row_dim; i++)
	{
		//tmp = 0.0;
		tmp4 = vdupq_n_f32(0.0f);

		//for(j = 0; j < a->row_dim; j++)
		for(j = 0; j < a->real_col_dim; j += _BNC_S_WIDTH)
		{
			//tmp += get_fmatrix_ij(a, i, j) * get_fvector_i(vb, j);

			a4  = vld1q_f32(&get_fmatrix_ij(a, i, j));
			vb4 = vld1q_f32(&get_fvector_i(vb, j));
			tmp4 = vfmaq_f32(tmp4, a4, vb4);
		}
		//set_fvector_i(v, i, tmp);
		//vst1q_f32(&get_fvector_i(v, i), tmp4);
		set_fvector_i(v, i, tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3]);

	}
	//printf("v = %d, a->real_row_dim, real_col_dim = %d, %d, vb = %d\n", v->real_dim, a->real_row_dim, a->real_col_dim, vb->real_dim);
#else // others
	for(i = 0; i < a->row_dim; i++)
	{
		tmp = 0.0;
		for(j = 0; j < a->col_dim; j++)
			tmp += get_fmatrix_ij(a, i, j) * get_fvector_i(vb, j);
		set_fvector_i(v, i, tmp);
	}
#endif // __AVX2__
}

/* v = a^T * vb */
//void mul_fmatrixt_dvec(FVector v, FMatrix a, FVector vb)
void mul_fmatrixt_dvec_old(FVector v, FMatrix a, FVector vb)
{
	long int i, j;
	float tmp;

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_fmatrixt_dvec\n");
		return;
	}
	for(i = 0; i < a->col_dim; i++)
	{
		tmp = 0.0;
		for(j = 0; j < a->row_dim; j++)
			tmp += get_fmatrix_ij(a, j, i) * get_fvector_i(vb, j);
		set_fvector_i(v, i, tmp);
	}
}
/* v = a^T * vb */
// Dangerous!
//void mul_fmatrixt_dvec_simd(FVector v, FMatrix a, FVector vb)
void mul_fmatrixt_dvec(FVector v, FMatrix a, FVector vb)
{
	long int i, j;
	float tmp;

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_fmatrixt_dvec\n");
		return;
	}
// Dangerous!
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4, vb4, tmp4;

	for(i = 0; i < a->col_dim; i++)
	{
		//tmp = 0.0;
		tmp4 = _mm256_setzero_ps();

		//for(j = 0; j < a->row_dim; j++)
		for(j = 0; j < a->real_row_dim; j += _BNC_S_WIDTH)
		{
			//tmp += get_fmatrix_ij(a, j, i) * get_fvector_i(vb, j);

			//a4  = _mm256_load_ps(&get_fmatrix_ij(a, j, i));
			a4 = _mm256_set_ps(
				get_fmatrix_ij(a, j + 7, i),
				get_fmatrix_ij(a, j + 6, i),
				get_fmatrix_ij(a, j + 5, i),
				get_fmatrix_ij(a, j + 4, i),
				get_fmatrix_ij(a, j + 3, i),
				get_fmatrix_ij(a, j + 2, i),
				get_fmatrix_ij(a, j + 1, i),
				get_fmatrix_ij(a, j    , i)
			);
			vb4 = _mm256_load_ps(&get_fvector_i(vb, j));
			tmp4 = _mm256_fmadd_ps(a4, vb4, tmp4);
		}
		//set_fvector_i(v, i, tmp);
		//_mm256_store_ps(&get_fvector_i(v, i), tmp4);
		set_fvector_i(v, i, tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3] + tmp4[4] + tmp4[5] + tmp4[6] + tmp4[7]);
	}
// Dangerous!
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a8, vb8, tmp8;

	for(i = 0; i < a->col_dim; i++)
	{
		//tmp = 0.0;
		tmp8 = _mm512_setzero_ps();

		//for(j = 0; j < a->row_dim; j++)
		for(j = 0; j < a->real_row_dim; j += _BNC_S_WIDTH)
		{
			//tmp += get_fmatrix_ij(a, j, i) * get_fvector_i(vb, j);

			//a8  = _mm512_load_ps(&get_fmatrix_ij(a, j, i));
			a8 = _mm512_set_ps(
				get_fmatrix_ij(a, j + 15, i),
				get_fmatrix_ij(a, j + 14, i),
				get_fmatrix_ij(a, j + 13, i),
				get_fmatrix_ij(a, j + 12, i),
				get_fmatrix_ij(a, j + 11, i),
				get_fmatrix_ij(a, j + 10, i),
				get_fmatrix_ij(a, j + 9, i),
				get_fmatrix_ij(a, j + 8, i),
				get_fmatrix_ij(a, j + 7, i),
				get_fmatrix_ij(a, j + 6, i),
				get_fmatrix_ij(a, j + 5, i),
				get_fmatrix_ij(a, j + 4, i),
				get_fmatrix_ij(a, j + 3, i),
				get_fmatrix_ij(a, j + 2, i),
				get_fmatrix_ij(a, j + 1, i),
				get_fmatrix_ij(a, j    , i)
			);
			vb8 = _mm512_load_ps(&get_fvector_i(vb, j));
			tmp8 = _mm512_fmadd_ps(a8, vb8, tmp8);
		}
		//set_fvector_i(v, i, tmp);
		//_mm512_store_ps(&get_fvector_i(v, i), tmp8);
		set_fvector_i(v, i, tmp8[0] + tmp8[1] + tmp8[2] + tmp8[3] + tmp8[4] + tmp8[5] + tmp8[6] + tmp8[7] + tmp8[8] + tmp8[9] + tmp8[10] + tmp8[11] + tmp8[12] + tmp8[13] + tmp8[14] + tmp8[15]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, gather)
	{
	svfloat32_t a_v, vb_v, tmp;
	svint32_t aidx;
	for(i = 0; i < a->col_dim; i++)
	{
		tmp = svdup_f32(0.0);
		for(j = 0; j < a->real_row_dim; j += (long int)svcntw())
		{
			svbool_t pg = svwhilelt_b32_s64((int64_t)j, (int64_t)a->real_row_dim);
			aidx = svindex_s32((int32_t)(j * a->real_col_dim + i), (int32_t)a->real_col_dim);
			a_v = svld1_gather_s32index_f32(pg, a->element, aidx);
			vb_v = svld1_f32(pg, &get_fvector_i(vb, j));
			tmp = svmla_f32_m(pg, tmp, a_v, vb_v);
		}
		set_fvector_i(v, i, svaddv_f32(svptrue_b32(), tmp));
	}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	{
	float32x4_t acc, av, vbv;
	double sum; long int j;
	for(i = 0; i < a->col_dim; i++)
	{
		acc = vdupq_n_f32(0.0);
		for(j = 0; j <= a->real_row_dim - 4; j += 4)
		{
			av = (float32x4_t){ get_fmatrix_ij(a, j + 0, i), get_fmatrix_ij(a, j + 1, i), get_fmatrix_ij(a, j + 2, i), get_fmatrix_ij(a, j + 3, i) };
			vbv = vld1q_f32(&get_fvector_i(vb, j));
			acc = vfmaq_f32(acc, av, vbv);
		}
		sum = vaddvq_f32(acc);
		for(; j < a->row_dim; j++) sum += get_fmatrix_ij(a, j, i) * get_fvector_i(vb, j);
		set_fvector_i(v, i, sum);
	}
	}
#else // others
	for(i = 0; i < a->col_dim; i++)
	{
		tmp = 0.0;
		for(j = 0; j < a->row_dim; j++)
			tmp += get_fmatrix_ij(a, j, i) * get_fvector_i(vb, j);
		set_fvector_i(v, i, tmp);
	}
#endif // __AVX2__
}

/* a = a^(-1) */
/* square matrix only */
void inv_fmatrix(FMatrix a)
{
	long int i, j, k, dim;
	float tmp, aii;

	/* Check Dimensions */
	if(a->row_dim != a->col_dim)
	{
		fprintf(stderr, "ERROR: inv_fmatrix\n");
		return;
	}

	dim = a->row_dim;

	for(i = 0; i < dim; i++)
	{
		if(get_fmatrix_ij(a, i, i) == 0.0)
		{
			fprintf(stderr, "ERROR: inv_fmatrix: Pivot(%ld,%ld) is zero.\n", i, i);
			return;
		}

		aii = 1.0 / get_fmatrix_ij(a, i, i);
		set_fmatrix_ij(a, i, i, aii);

		for(j = 0; j < i; j++)
			set_fmatrix_ij(a, i, j, get_fmatrix_ij(a, i, j)* aii);
		for(j = i + 1; j < dim; j++)
			set_fmatrix_ij(a, i, j, get_fmatrix_ij(a, i, j) * aii);

		for(j = 0; j < i; j++)
		{
			for(k = 0; k < i; k++)
				set_fmatrix_ij(a, j, k, get_fmatrix_ij(a, j, k) - get_fmatrix_ij(a, j, i) * get_fmatrix_ij(a, i, k));
			for(k = i + 1; k < dim; k++)
				set_fmatrix_ij(a, j, k, get_fmatrix_ij(a, j, k) - get_fmatrix_ij(a, j, i) * get_fmatrix_ij(a, i, k));
		}

		for(j = i + 1; j < dim; j++)
		{
			for(k = 0; k < i; k++)
				set_fmatrix_ij(a, j, k, get_fmatrix_ij(a, j, k) - get_fmatrix_ij(a, j, i) * get_fmatrix_ij(a, i, k));
			for(k = i + 1; k < dim; k++)
				set_fmatrix_ij(a, j, k, get_fmatrix_ij(a, j, k) - get_fmatrix_ij(a, j, i) * get_fmatrix_ij(a, i, k));
		}

		for(j = 0; j < i; j++)
			set_fmatrix_ij(a, j, i, get_fmatrix_ij(a, j, i) * -aii);
		for(j = i + 1; j < dim; j++)
			set_fmatrix_ij(a, j, i, get_fmatrix_ij(a, j, i) * -aii);	}
}

/* Double */

/* 1. Hilbert Matrix */
void hilbert_fmatrix(FMatrix a, long int dim)
{
	long int i, j;
	float tmp;

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(hilbert_fmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(hilbert_fmatrix)\n");
		return;
	}

	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
			set_fmatrix_ij(a, i, j, 1.0 / (i + j + 1));
	}
}


/* 2. Lotkin Matrix */
void lotkin_fmatrix(FMatrix a, long int dim)
{
	long int i, j;
	float tmp;

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(lotkin_fmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(lotkin_fmatrix)\n");
		return;
	}
	/* Lotkin Matrix */
	for(i = 0; i < a->col_dim; i++)
		set_fmatrix_ij(a, 0, i, 1.0);

	for(i = 1; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
			set_fmatrix_ij(a, i, j, 1.0 / (i + j + 1));
	}
}

/* 3. Frank Matrix */
void frank_fmatrix(FMatrix a, long int dim)
{
	long int i, j;
	float tmp;

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(frank_fmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(frank_fmatrix)\n");
		return;
	}

	/* Frank Matrix */
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			if(i < j)
				set_fmatrix_ij(a, i, j, (float)(dim - j));
			else
				set_fmatrix_ij(a, i, j, (float)(dim - i));
		}
	}
}

/* 4. Tridiagonal Matrix */
void tridiag_fmatrix(FMatrix a, FVector low_subdiag, FVector diag, FVector up_subdiag, long int dim)
{
	long int i, j;
	float tmp;

	if(dim > a->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(tridiag_fmatrix)\n");
		return;
	}
	if(dim > a->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(tridiag_fmatrix)\n");
		return;
	}

	/* Tridiagonal Matrix */
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < i - 1; j++)
			set_fmatrix_ij(a, i, j, 0.0);
		for(j = i + 2; j < dim; j++)
			set_fmatrix_ij(a, i, j, 0.0);
	}

	set_fmatrix_ij(a, 0, 0, get_fvector_i(diag, 0));
	set_fmatrix_ij(a, 0, 1, get_fvector_i(up_subdiag , 0));
	for(i = 1; i < dim - 1; i++)
	{
		set_fmatrix_ij(a, i, i - 1, get_fvector_i(low_subdiag, i));
		set_fmatrix_ij(a, i, i    , get_fvector_i(diag, i));
		set_fmatrix_ij(a, i, i + 1, get_fvector_i(up_subdiag , i));
	}
	i = dim - 1;
	set_fmatrix_ij(a, i, i - 1, get_fvector_i(low_subdiag , i));
	set_fmatrix_ij(a, i, i    , get_fvector_i(diag, i));
}


/* 5. Integer Symmetrix Random Matrix */
void int_sym_rand_fmatrix(FMatrix mat, long int max, long int seed, long int dim)
{
	long int i, j;

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(int_ sym_rand_fmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(int_sym_rand_fmatrix)\n");
		return;
	}

	srand((int)seed);
	for(i = 0; i < dim; i++)
		for(j = i; j < dim; j++)
			set_fmatrix_ij(mat, i, j, (float)(rand() % max));

	for(i = 0; i < dim; i++)
		for(j = 0; j < i; j++)
			set_fmatrix_ij(mat, i, j, get_fmatrix_ij(mat, j, i));
}

/* 6. Integer Unsymmetrix Random Matrix */
void int_unsym_rand_fmatrix(FMatrix mat, long int max, long int seed, long int dim)
{
	long int i, j;

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(int_unsym_rand_fmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(int_unsym_rand_fmatrix)\n");
		return;
	}

	srand((int)seed);
	for(i = 0; i < dim; i++)
		for(j = 0; j < dim; j++)
			set_fmatrix_ij(mat, i, j, (float)(rand() % max));
}

/* 7. Real Diagonal Matrix */
void diag_fmatrix(FMatrix mat, FVector diag, long int dim)
{
	long int i, j;

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(diag_fmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(diag_fmatrix)\n");
		return;
	}

	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
			set_fmatrix_ij(mat, i, j, 0.0);
		set_fmatrix_ij(mat, i, i, get_fvector_i(diag, i));
	}

}

/* 8. Toeplitz Matrix */
void toeplitz_fmatrix(FMatrix mat, float gamma_param, long int dim)
{
	long int i, j;

	if(dim > mat->row_dim)
	{
		fprintf(stderr, "dim is too large than row_dim!(toeplitz_fmatrix)\n");
		return;
	}
	if(dim > mat->col_dim)
	{
		fprintf(stderr, "dim is too large than col_dim!(toeplitz_fmatrix)\n");
		return;
	}

	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
			set_fmatrix_ij(mat, i, j, 0.0);

		if(i >= 2)
			set_fmatrix_ij(mat, i, i - 2, gamma_param);

		if(i <= (dim - 2))
			set_fmatrix_ij(mat, i, i + 1, 1.0);

		set_fmatrix_ij(mat, i, i, 2.0);
	}
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 1997.10.24 (Fri) Tomonori Kouya */
/*                 ver. 0.1 2000.02.24 (Thu) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int FLUdecomp(FMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       FMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	float dtmp, dmaxii;

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
	{
		dmaxii = fabsf(get_fmatrix_ij(a, i, i));
		if(dmaxii == 0.0)
		{
			fprintf(stderr, "%ld : Error! (DLUdecomp)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
			set_fmatrix_ij(a, j, i, get_fmatrix_ij(a, j, i) / get_fmatrix_ij(a, i, i));
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
				set_fmatrix_ij(a, j, k, get_fmatrix_ij(a, j, k) - get_fmatrix_ij(a, j, i) * get_fmatrix_ij(a, i, k));
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 1997.10.24 (Fri) Tomonori Kouya */
/*                 ver. 0.1 2000.02.24 (Thu) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveFLS(FVector answer, FMatrix lu, FVector b)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       FMatrix lu: LU decomposed Matrix (given by user)   */
/*       FVector b: constant vector (given by user)         */
/*       FVector answer: Solution for linear system         */
/*       long int dim: Dimension of Matrix (given by user)  */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	float dtmp;

	dim = answer->dim;

	subst_fvector(answer, b);

/* Forward */
	for(i = 0; i < dim; i++)
	{
		if(get_fmatrix_ij(lu, i, i) == 0.0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveDLS, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
			set_fvector_i(answer, j, get_fvector_i(answer, j) - get_fmatrix_ij(lu, j, i) * get_fvector_i(answer, i));
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
			set_fvector_i(answer, i, get_fvector_i(answer, i) - get_fmatrix_ij(lu, i, j) * get_fvector_i(answer, j));
		set_fvector_i(answer, i, get_fvector_i(answer, i) / get_fmatrix_ij(lu, i, i));
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                 (Double Precision)       */
/*                                 (Partial Pivoting)       */
/*                                                          */
/*                 ver. 0.0 1997.10.24 (Fri) Tomonori Kouya */
/*                 ver. 0.1 2000.02.24 (Thu) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int FLUdecompP(FMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       FMatrix a: Matrix (given by user)                  */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
	float dtmp, dmaxii;

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		dmaxii = fabsf(get_fmatrix_ij(a, ch[i], i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			dtmp = fabsf(get_fmatrix_ij(a, ch[j], i));
			if(dtmp > dmaxii)
			{
				imax = j;
				dmaxii = dtmp;
			}
		}

		if(dmaxii == 0.0)
		{
			fprintf(stderr, "%ld : Error! DLUdecompP!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;
		}

		for(j = (i + 1); j < dim; j++)
			set_fmatrix_ij(a, ch[j], i, get_fmatrix_ij(a, ch[j], i) / get_fmatrix_ij(a, ch[i], i));
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
				set_fmatrix_ij(a, ch[j], k, get_fmatrix_ij(a, ch[j], k) - get_fmatrix_ij(a, ch[j], i) * get_fmatrix_ij(a, ch[i], k));
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                                 (Double Precision)       */
/*                                 (Partial Pivoting)       */
/*                                                          */
/*                 ver. 0.0 1997.10.24 (Fri) Tomonori Kouya */
/*                 ver. 0.1 2000.02.24 (Thu) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveFLSP(FVector answer, FMatrix lu, FVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       FMatrix lu[]: LU decomposed Matrix (given by user) */
/*       FVector b[]: constant vector (given by user)       */
/*       FVector answer[]: Solution for linear system       */
/*       long int ch: Row order (given by user)             */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	float dtmp;

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_fvector_i(answer, i, get_fvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		if(get_fmatrix_ij(lu, ch[i], i) == 0.0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveDLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
			set_fvector_i(answer, j, get_fvector_i(answer, j) - get_fmatrix_ij(lu, ch[j], i) * get_fvector_i(answer, i));
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
			set_fvector_i(answer, i, get_fvector_i(answer, i) - get_fmatrix_ij(lu, ch[i], j) * get_fvector_i(answer, j));
		set_fvector_i(answer, i, get_fvector_i(answer, i) / get_fmatrix_ij(lu, ch[i], i));
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                 (Double Precision)       */
/*                                 (Complete Pivoting)      */
/*                                                          */
/*                 ver. 0.0 1997.10.24 (Fri) Tomonori Kouya */
/*                 ver. 0.1 2000.02.24 (Thu) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int FLUdecompC(FMatrix a, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       FMatrix a[]: Matrix (given by user)                */
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
	float dtmp, dmaxii;

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
	{
		row_ch[i] = i;
		col_ch[i] = i;
	}

	for(i = 0; i < dim; i++)
	{
		/* Complete Pivoting */
		dmaxii = fabsf(get_fmatrix_ij(a, row_ch[i], col_ch[i]));
		imax = i;
		jmax = i;
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				dtmp = fabsf(get_fmatrix_ij(a, row_ch[j], col_ch[k]));
				if(dtmp > dmaxii)
				{
					imax = j;
					jmax = k;
					dmaxii = dtmp;
				}
			}
		}

		if(dmaxii == 0.0)
		{
			fprintf(stderr, "%ld : Error! (DLUdecompC)!\n", i);
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
			set_fmatrix_ij(a, row_ch[j], col_ch[i], get_fmatrix_ij(a, row_ch[j], col_ch[i]) / get_fmatrix_ij(a, row_ch[i], col_ch[i]));
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
				set_fmatrix_ij(a, row_ch[j], col_ch[k], get_fmatrix_ij(a, row_ch[j], col_ch[k]) - get_fmatrix_ij(a, row_ch[j], col_ch[i]) * get_fmatrix_ij(a, row_ch[i], col_ch[k]));
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                                 (Double Precision)       */
/*                                 (Complete Pivoting)      */
/*                                                          */
/*                 ver. 0.0 1997.10.24 (Fri) Tomonori Kouya */
/*                 ver. 0.1 2000.02.24 (Thu) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveFLSC(FVector answer, FMatrix lu, FVector b, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       FMatrix lu: LU decomposed Matrix (given by user)   */
/*       FVector b: constant vector (given by user)         */
/*       FVector answer: Solution for linear system         */
/*       long int row_ch[]: Row order (given by user)       */
/*       long int col_ch[]: Column order (given by user)    */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	float dtmp;

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_fvector_i(answer, col_ch[i], get_fvector_i(b, row_ch[i]));

/* Forward */
	for(i = 0; i < dim; i++)
	{
		if(get_fmatrix_ij(lu, row_ch[i], col_ch[i]) == 0.0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveDLSC, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
			set_fvector_i(answer, col_ch[j], get_fvector_i(answer, col_ch[j]) - get_fmatrix_ij(lu, row_ch[j], col_ch[i]) * get_fvector_i(answer, col_ch[i]));
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
			set_fvector_i(answer, col_ch[i], get_fvector_i(answer, col_ch[i]) - get_fmatrix_ij(lu, row_ch[i], col_ch[j]) * get_fvector_i(answer, col_ch[j]));
		set_fvector_i(answer, col_ch[i], get_fvector_i(answer, col_ch[i]) / get_fmatrix_ij(lu, row_ch[i], col_ch[i]));
	}

	return 0;
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_fmatrix(FMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
	long int i, j, true_end;
	float tmp;

	true_end = (col_end > mat->col_dim) ? mat->col_dim : col_end;

	for(i = col_start; i < true_end; i++)
	{
		tmp = get_fmatrix_ij(mat, row_index0, i);
		set_fmatrix_ij(mat, row_index0, i, get_fmatrix_ij(mat, row_index1, i));
		set_fmatrix_ij(mat, row_index1, i, tmp);
	}
}

#if defined (__cplusplus)
}
#endif // defined (__cplusplus)
