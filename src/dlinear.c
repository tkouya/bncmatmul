//******************************************************************************
// dlinear.c : Double Precision Basic Linear Algebra 
// Copyright (C) 2020 Tomonori Kouya
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
#include "dlinear.h"
#include "rdd.h"
//#include "bncavx.h"

#if defined (__cplusplus)
extern "C" {
#endif // defined (__cplusplus)

double drel_diff_array(double approx_a[], double approx_b[], int dim, int print_flag)
{
    int i;
    double rel_min, rel_max, rel_ave, rel_diff;

    rel_diff = drel_diff(approx_a[0], approx_b[0]);
    rel_min = rel_diff;
    rel_max = rel_diff;
    rel_ave = rel_diff;

    for(i = 1; i < dim; i++)
    {
        rel_diff = drel_diff(approx_a[i], approx_b[i]);
        if(rel_diff < rel_min) rel_min = rel_diff;
        if(rel_diff > rel_max) rel_max = rel_diff;
        rel_ave += rel_diff;
    }
    rel_ave /= (double)dim;

    if(print_flag == 1)
        printf("max_rel_diff, min_rel_diff, ave_rel_diff: %5.2e, %5.2e, %5.2e\n", rel_max, rel_min, rel_ave);

    return rel_max;
}

/*************************************************/
/* Vector Calculations for DVector               */
/*
DVector init_dvector(long int dimension)
void free_dvector(DVector vec)
void add_dvector(DVector c, DVector a, DVector b)
void add2_dvector(DVector c, DVector a)
void sub_dvector(DVector c, DVector a, DVector b)
void sub2_dvector(DVector c, DVector a)
void cmul_dvector(DVector c, double val, DVector a)
void cmul2_dvector(DVector c, double val)
void add_cmul_dvector(DVector c, DVector a, double val, DVector b)
double ip_dvector(DVector a, DVector b)
double norm1_dvector(DVector a)
double norm2_dvector(DVector a)
double normi_dvector(DVector a)
void subst_dvector(DVector c, DVector a)
*/
/*************************************************/

DVector init_dvector(long int dimension)
{
	DVector ret = NULL;
	long int i, real_dim;

	if(dimension <= 0)
	{
		fprintf(stderr, "ERROR: init_dvector\n");
		return ret;
	}

	ret = (DVector)BNC_MALLOC(sizeof(dvector));
	if(ret == NULL)
		return ret;

	// real_dim is the nearest positive multiplier of _BNC_D_WIDTH
	real_dim = (long int)ceil((double)dimension / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;

//	ret->element = (double *)BNC_CALLOC(dimension, sizeof(double));
	ret->element = (double *)BNC_CALLOC(real_dim, sizeof(double));
	if(ret->element == NULL)
		return ret;

	/* All 0 */
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	for(i = 0; i < real_dim; i += _BNC_D_WIDTH)
		_mm256_store_pd((ret->element + i), zero4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(i = 0; i < real_dim; i += _BNC_D_WIDTH)
		_mm512_store_pd((ret->element + i), zero8);

// 2025-08-07(Thu)
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    svfloat64_t zero2 = svdup_f64(0.0);

	for(i = 0; i < real_dim; i += (long int)svcntd()) {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(real_dim));
        svst1_f64(pg, &(ret->element[i]), zero2);
        }

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    float64x2_t zero2 = vdupq_n_f64(0.0);

	for(i = 0; i < real_dim; i += 2)
        vst1q_f64(&(ret->element[i]), zero2);

#else // others
	for(i = 0; i < dimension; i++)
		*(ret->element + i) = 0.0;
#endif // __AVX2__

	ret->dim = dimension;
	ret->real_dim = real_dim;

	return ret;
}

void free_dvector(DVector vec)
{
	if(vec == NULL)
		return;

	if(vec->element != NULL)
		free(vec->element);

	free(vec);
}

// print_dvector
void print_dvector(DVector vec)
{
    long int i;

    for(i = 0; i < vec->dim; i++)
        printf("%5ld %25.17e\n", i, get_dvector_i(vec, i));
}

/* c = a + b */
void add_dvector(DVector c, DVector a, DVector b)
{
	long int i;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_dvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, b4, c4;

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i) + get_dvector_i(b, i));

		a4 = _mm256_load_pd(&get_dvector_i(a, i));
		b4 = _mm256_load_pd(&get_dvector_i(b, i));
		c4 = _mm256_add_pd(a4, b4);
		_mm256_store_pd(&get_dvector_i(c, i), c4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, b8, c8;

	for(i = 0; i < c->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i) + get_dvector_i(b, i));

		a8 = _mm512_load_pd(&get_dvector_i(a, i));
		b8 = _mm512_load_pd(&get_dvector_i(b, i));
		c8 = _mm512_add_pd(a8, b8);
		_mm512_store_pd(&get_dvector_i(c, i), c8);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    svfloat64_t a2, b2, c2;
    for(i = 0; i < c->real_dim; i += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(c->real_dim));
        a2 = svld1_f64(pg, &get_dvector_i(a, i));
        b2 = svld1_f64(pg, &get_dvector_i(b, i));
        c2 = svadd_f64_x(pg, a2, b2);
        svst1_f64(pg, &get_dvector_i(c, i), c2);
    }

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon|| defined(__aarch64__) && defined(BNC_ENABLE_NEON) // Arm Neon// __ARM_NEON
    float64x2_t a2, b2, c2;
    for(i = 0; i < c->real_dim; i += 2)
    {
        a2 = vld1q_f64(&get_dvector_i(a, i));
        b2 = vld1q_f64(&get_dvector_i(b, i));
        c2 = vaddq_f64(a2, b2);
        vst1q_f64(&get_dvector_i(c, i), c2);
    }

#else // others
	for(i = 0; i < c->dim; i++)
		set_dvector_i(c, i, get_dvector_i(a, i) + get_dvector_i(b, i));
#endif // __AVX2__

}

/* c += a */
void add2_dvector(DVector c, DVector a)
{
	long int i;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: add2_dvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, c4;

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(c, i) + get_dvector_i(a, i));

		a4 = _mm256_load_pd(&get_dvector_i(a, i));
		c4 = _mm256_load_pd(&get_dvector_i(c, i));
		c4 = _mm256_add_pd(c4, a4);
		_mm256_store_pd(&get_dvector_i(c, i), c4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, c8;

	//for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(c, i) + get_dvector_i(a, i));

		a8 = _mm512_load_pd(&get_dvector_i(a, i));
		c8 = _mm512_load_pd(&get_dvector_i(c, i));
		c8 = _mm512_add_pd(c8, a8);
		_mm512_store_pd(&get_dvector_i(c, i), c8);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    for (long i = 0; i < c->real_dim; i += (long int)svcntd()) {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(c->real_dim));
        svfloat64_t c2 = svld1_f64(pg, &c->element[i]);
        svfloat64_t a2 = svld1_f64(pg, &a->element[i]);
        c2 = svadd_f64_x(pg, c2, a2);
        svst1_f64(pg, &c->element[i], c2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
    for (long i = 0; i < c->real_dim; i += 2) {
        float64x2_t c2 = vld1q_f64(&c->element[i]);
        float64x2_t a2 = vld1q_f64(&a->element[i]);
        c2 = vaddq_f64(c2, a2);
        vst1q_f64(&c->element[i], c2);
    }
#else // others
	for(i = 0; i < c->dim; i++)
		set_dvector_i(c, i, get_dvector_i(c, i) + get_dvector_i(a, i));
#endif // __AVX2__

}

/* c = a - b */
void sub_dvector(DVector c, DVector a, DVector b)
{
	long int i;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_dvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, b4, c4;

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i) - get_dvector_i(b, i));

		a4 = _mm256_load_pd(&get_dvector_i(a, i));
		b4 = _mm256_load_pd(&get_dvector_i(b, i));
		c4 = _mm256_sub_pd(a4, b4);
		_mm256_store_pd(&get_dvector_i(c, i), c4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, b8, c8;

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i) - get_dvector_i(b, i));

		a8 = _mm512_load_pd(&get_dvector_i(a, i));
		b8 = _mm512_load_pd(&get_dvector_i(b, i));
		c8 = _mm512_sub_pd(a8, b8);
		_mm512_store_pd(&get_dvector_i(c, i), c8);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    for (long i = 0; i < c->real_dim; i += (long int)svcntd()) {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(c->real_dim));
        svfloat64_t a2 = svld1_f64(pg, &a->element[i]);
        svfloat64_t b2 = svld1_f64(pg, &b->element[i]);
        svfloat64_t c2 = svsub_f64_x(pg, a2, b2);
        svst1_f64(pg, &c->element[i], c2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
    for (long i = 0; i < c->real_dim; i += 2) {
        float64x2_t a2 = vld1q_f64(&a->element[i]);
        float64x2_t b2 = vld1q_f64(&b->element[i]);
        float64x2_t c2 = vsubq_f64(a2, b2);
        vst1q_f64(&c->element[i], c2);
    }
#else // others
	for(i = 0; i < c->dim; i++)
		set_dvector_i(c, i, get_dvector_i(a, i) - get_dvector_i(b, i));
#endif // __AVX2__

}

/* c -= a */
void sub2_dvector(DVector c, DVector a)
{
	long int i;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: sub2_dvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, c4;

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(c, i) - get_dvector_i(a, i));

		a4 = _mm256_load_pd(&get_dvector_i(a, i));
		c4 = _mm256_load_pd(&get_dvector_i(c, i));
		c4 = _mm256_sub_pd(c4, a4);
		_mm256_store_pd(&get_dvector_i(c, i), c4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, c8;

	//for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(c, i) - get_dvector_i(a, i));

		a8 = _mm512_load_pd(&get_dvector_i(a, i));
		c8 = _mm512_load_pd(&get_dvector_i(c, i));
		c8 = _mm512_sub_pd(c8, a8);
		_mm512_store_pd(&get_dvector_i(c, i), c8);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    for (long i = 0; i < c->real_dim; i += (long int)svcntd()) {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(c->real_dim));
        svfloat64_t c2 = svld1_f64(pg, &c->element[i]);
        svfloat64_t a2 = svld1_f64(pg, &a->element[i]);
        c2 = svsub_f64_x(pg, c2, a2);
        svst1_f64(pg, &c->element[i], c2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
    for (long i = 0; i < c->real_dim; i += 2) {
        float64x2_t c2 = vld1q_f64(&c->element[i]);
        float64x2_t a2 = vld1q_f64(&a->element[i]);
        c2 = vsubq_f64(c2, a2);
        vst1q_f64(&c->element[i], c2);
    }
#else // others
	for(i = 0; i < c->dim; i++)
		set_dvector_i(c, i, get_dvector_i(c, i) - get_dvector_i(a, i));
#endif // __AVX2__

}

/* c = val * a */
void cmul_dvector(DVector c, double val, DVector a)
{
	long int i;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: cmul_dvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, c4, val4;

//	val4 = _mm256_set_pd(val, val, val, val);
	val4 = _mm256_set1_pd(val);

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, val * get_dvector_i(a, i));

		a4 = _mm256_load_pd(&get_dvector_i(a, i));
		c4 = _mm256_mul_pd(val4, a4);
		_mm256_store_pd(&get_dvector_i(c, i), c4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, c8, val8;

	val8 = _mm512_set_pd(val, val, val, val, val, val, val, val);

	//for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, val * get_dvector_i(a, i));

		a8 = _mm512_load_pd(&get_dvector_i(a, i));
		c8 = _mm512_mul_pd(val8, a8);
		_mm512_store_pd(&get_dvector_i(c, i), c8);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    svfloat64_t val2 = svdup_f64(val);
    for (long i = 0; i < c->real_dim; i += (long int)svcntd()) {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(c->real_dim));
        svfloat64_t a2 = svld1_f64(pg, &a->element[i]);
        svfloat64_t c2 = svmul_f64_x(pg, val2, a2);
        svst1_f64(pg, &c->element[i], c2);
    }

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
    float64x2_t val2 = vdupq_n_f64(val);
    for (long i = 0; i < c->real_dim; i += 2) {
        float64x2_t a2 = vld1q_f64(&a->element[i]);
        float64x2_t c2 = vmulq_f64(val2, a2);
        vst1q_f64(&c->element[i], c2);
    }

#else // others
	for(i = 0; i < c->dim; i++)
		set_dvector_i(c, i, val * get_dvector_i(a, i));
#endif // __AVX2__

}

/* c *= val */
void cmul2_dvector(DVector c, double val)
{
	long int i;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d c4, val4;

	val4 = _mm256_set_pd(val, val, val, val);

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, val * get_dvector_i(a, i));

		c4 = _mm256_load_pd(&get_dvector_i(c, i));
		c4 = _mm256_mul_pd(val4, c4);
		_mm256_store_pd(&get_dvector_i(c, i), c4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d c8, val8;

	val8 = _mm512_set_pd(val, val, val, val, val, val, val, val);

	//for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, val * get_dvector_i(a, i));

		c8 = _mm512_load_pd(&get_dvector_i(c, i));
		c8 = _mm512_mul_pd(val8, c8);
		_mm512_store_pd(&get_dvector_i(c, i), c8);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    svfloat64_t val2 = svdup_f64(val);
    for (long i = 0; i < c->real_dim; i += (long int)svcntd()) {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(c->real_dim));
        svfloat64_t c2 = svld1_f64(pg, &c->element[i]);
        c2 = svmul_f64_x(pg, val2, c2);
        svst1_f64(pg, &c->element[i], c2);
    }

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
    float64x2_t val2 = vdupq_n_f64(val);
    for (long i = 0; i < c->real_dim; i += 2) {
        float64x2_t c2 = vld1q_f64(&c->element[i]);
        c2 = vmulq_f64(val2, c2);
        vst1q_f64(&c->element[i], c2);
    }

#else // others
	for(i = 0; i < c->dim; i++)
		set_dvector_i(c, i, val * get_dvector_i(c, i));
#endif // __AVX2__

}

/* c = a + val * b */
void add_cmul_dvector(DVector c, DVector a, double val, DVector b)
{
	long int i;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_cmul_dvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, b4, c4, val4;

	val4 = _mm256_set_pd(val, val, val, val);

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i) + val * get_dvector_i(b, i));

		a4 = _mm256_load_pd(&get_dvector_i(a, i));
		b4 = _mm256_load_pd(&get_dvector_i(b, i));

		// c4 := (val4 * b4) + a4
		c4 = _mm256_fmadd_pd(val4, b4, a4);

		_mm256_store_pd(&get_dvector_i(c, i), c4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, b8, c8, val8;

	val8 = _mm512_set_pd(val, val, val, val, val, val, val, val);

	//for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i) + val * get_dvector_i(b, i));

		a8 = _mm512_load_pd(&get_dvector_i(a, i));
		b8 = _mm512_load_pd(&get_dvector_i(b, i));

		// c8 := (val8 * b8) + a8
		c8 = _mm512_fmadd_pd(val8, b8, a8);

		_mm512_store_pd(&get_dvector_i(c, i), c8);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    svfloat64_t val2 = svdup_f64(val);
    for (long i = 0; i < c->real_dim; i += (long int)svcntd()) {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(c->real_dim));
        svfloat64_t a2 = svld1_f64(pg, &a->element[i]);
        svfloat64_t b2 = svld1_f64(pg, &b->element[i]);
        svfloat64_t c2 = svmla_f64_x(pg, a2, b2, val2);
        svst1_f64(pg, &c->element[i], c2);
    }

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
    float64x2_t val2 = vdupq_n_f64(val);
    for (long i = 0; i < c->real_dim; i += 2) {
        float64x2_t a2 = vld1q_f64(&a->element[i]);
        float64x2_t b2 = vld1q_f64(&b->element[i]);
        float64x2_t c2 = vfmaq_f64(a2, b2, val2);
        vst1q_f64(&c->element[i], c2);
    }

#else // others
	for(i = 0; i < c->dim; i++)
		set_dvector_i(c, i, get_dvector_i(a, i) + val * get_dvector_i(b, i));
#endif // __AVX2__

}

// 2025-02-19(Wed) T.Kouya
/* c = a - val * b */
void sub_cmul_dvector(DVector c, DVector a, double val, DVector b)
{
	long int i;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_cmul_dvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, b4, c4, val4;

	val4 = _mm256_set_pd(val, val, val, val);

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i) + val * get_dvector_i(b, i));

		a4 = _mm256_load_pd(&get_dvector_i(a, i));
		b4 = _mm256_load_pd(&get_dvector_i(b, i));

		// c4 := a4 - (val4 * b4)
		//c4 = _mm256_fmadd_pd(val4, b4, a4);
		c4 = _mm256_sub_pd(a4, _mm256_mul_pd(val4, b4));

		_mm256_store_pd(&get_dvector_i(c, i), c4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, b8, c8, val8;

	val8 = _mm512_set_pd(val, val, val, val, val, val, val, val);

	//for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i) + val * get_dvector_i(b, i));

		a8 = _mm512_load_pd(&get_dvector_i(a, i));
		b8 = _mm512_load_pd(&get_dvector_i(b, i));

		// c8 := a8 - (val8 * b8)
		//c8 = _mm512_fmadd_pd(val8, b8, a8);
		c8 = _mm512_sub_pd(a8, _mm512_mul_pd(val8, b8));

		_mm512_store_pd(&get_dvector_i(c, i), c8);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    svfloat64_t val2 = svdup_f64(val);
    for (long i = 0; i < c->real_dim; i += (long int)svcntd()) {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(c->real_dim));
        svfloat64_t a2 = svld1_f64(pg, &a->element[i]);
        svfloat64_t b2 = svld1_f64(pg, &b->element[i]);
        svfloat64_t c2 = svmls_f64_x(pg, a2, b2, val2);
        svst1_f64(pg, &c->element[i], c2);
    }

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
    float64x2_t val2 = vdupq_n_f64(val);
    for (long i = 0; i < c->real_dim; i += 2) {
        float64x2_t a2 = vld1q_f64(&a->element[i]);
        float64x2_t b2 = vld1q_f64(&b->element[i]);
        float64x2_t c2 = vmlsq_f64(a2, b2, val2);
        vst1q_f64(&c->element[i], c2);
    }

#else // others
	for(i = 0; i < c->dim; i++)
		set_dvector_i(c, i, get_dvector_i(a, i) - val * get_dvector_i(b, i));
#endif // __AVX2__

}


/* (a, b) */
double ip_dvector(DVector a, DVector b)
{
	double ret = 0.0;
	long int i;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: ip_dvector\n");
		return 0;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, b4, tmp4;

	// tmp4 ;= 0
	tmp4 = _mm256_setzero_pd();

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//tmp += get_dvector_i(a, i) * get_dvector_i(b, i);

		a4 = _mm256_load_pd(&get_dvector_i(a, i));
		b4 = _mm256_load_pd(&get_dvector_i(b, i));

		// tmp4 += a4 * b4
		tmp4 = _mm256_fmadd_pd(a4, b4, tmp4);
	}

	ret = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3];

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, b8, tmp8;

	// tmp8 := 0
	tmp8 = _mm512_setzero_pd();

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//tmp += get_dvector_i(a, i) * get_dvector_i(b, i);

		a8 = _mm512_load_pd(&get_dvector_i(a, i));
		b8 = _mm512_load_pd(&get_dvector_i(b, i));

		// c8 += a8 * b8
		tmp8 = _mm512_fmadd_pd(a8, b8, tmp8);

	}

	ret = tmp8[0] + tmp8[1] + tmp8[2] + tmp8[3] + tmp8[4] + tmp8[5] + tmp8[6] + tmp8[7];

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat64_t a4, b4, tmp4;

	// tmp4 ;= 0
	tmp4 = svdup_f64(0.0);

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += (long int)svcntd())
	{
			svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(a->real_dim));
		//tmp += get_dvector_i(a, i) * get_dvector_i(b, i);

		a4 = svld1_f64(pg, &get_dvector_i(a, i));
		b4 = svld1_f64(pg, &get_dvector_i(b, i));

		// tmp4 += a4 * b4
		tmp4 = svmla_f64_m(pg, tmp4, a4, b4);
	}

	ret = svaddv_f64(svptrue_b64(), tmp4);

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
    float64x2_t sum2 = vdupq_n_f64(0.0);
    for (long i = 0; i < a->real_dim; i += 2) {
        float64x2_t a2 = vld1q_f64(&a->element[i]);
        float64x2_t b2 = vld1q_f64(&b->element[i]);
        sum2 = vfmaq_f64(sum2, a2, b2);
    }
    ret = vgetq_lane_f64(sum2, 0) + vgetq_lane_f64(sum2, 1);

#else // others
	for(i = 0; i < a->dim; i++)
		ret += get_dvector_i(a, i) * get_dvector_i(b, i);
#endif // __AVX2__

	return ret;
}


/* ||a||_1 */
double norm1_dvector(DVector a)
{
	double ret = 0.0;
	long int i;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, tmp4;

	// tmp4 ;= 0
	tmp4 = _mm256_setzero_pd();

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//ret += fabs(get_dvector_i(a, i));

		a4 = _bncavx2_fabs(_mm256_load_pd(&get_dvector_i(a, i)));

		// tmp4 += fabs(a4)
		tmp4 = _mm256_add_pd(a4, tmp4);
	}

	ret = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3];

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, tmp8;

	// tmp8 := 0
	tmp8 = _mm512_setzero_pd();

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//ret += fabs(get_dvector_i(a, i));

		a8 = _bncavx512_fabs(_mm512_load_pd(&get_dvector_i(a, i)));

		// c8 += fabs(a8)
		tmp8 = _mm512_add_pd(a8, tmp8);

	}

	ret = tmp8[0] + tmp8[1] + tmp8[2] + tmp8[3] + tmp8[4] + tmp8[5] + tmp8[6] + tmp8[7];

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat64_t a4, tmp4;

	// tmp4 ;= 0
	tmp4 = svdup_f64(0.0);

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += (long int)svcntd())
	{
			svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(a->real_dim));
		//ret += fabs(get_dvector_i(a, i));

		a4 = svabs_f64_x(pg, svld1_f64(pg, &get_dvector_i(a, i)));

		// tmp4 += fabs(a4)
		tmp4 = svadd_f64_x(pg, a4, tmp4);
	}

	ret = svaddv_f64(svptrue_b64(), tmp4);

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
    float64x2_t sum2 = vdupq_n_f64(0.0);
    for (long i = 0; i < a->real_dim; i += 2) {
        float64x2_t a2 = vabsq_f64(vld1q_f64(&a->element[i]));
        sum2 = vaddq_f64(sum2, a2);
    }
    ret = vgetq_lane_f64(sum2, 0) + vgetq_lane_f64(sum2, 1);

#else // others
	for(i = 0; i < a->dim; i++)
		ret += fabs(get_dvector_i(a, i));
#endif // __AVX2__

	return ret;
}

/* ||a||_2 */
double norm2_dvector(DVector a)
{
	double ret = 0.0;
	long int i;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, tmp4;

	// tmp4 := 0
	tmp4 = _mm256_setzero_pd();

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//ret += get_dvector_i(a, i) * get_dvector_i(a, i);

		a4 = _mm256_load_pd(&get_dvector_i(a, i));

		// tmp4 += a4 * a4
		tmp4 = _mm256_fmadd_pd(a4, a4, tmp4);
	}

	ret = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3];

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, b8, tmp8;

	// tmp8 := 0
	tmp8 = _mm512_setzero_pd();

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//ret += get_dvector_i(a, i) * get_dvector_i(a, i);

		a8 = _mm512_load_pd(&get_dvector_i(a, i));

		// c8 += a8 * a8
		tmp8 = _mm512_fmadd_pd(a8, a8, tmp8);

	}

	ret = tmp8[0] + tmp8[1] + tmp8[2] + tmp8[3] + tmp8[4] + tmp8[5] + tmp8[6] + tmp8[7];

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat64_t a4, tmp4;

	// tmp4 := 0
	tmp4 = svdup_f64(0.0);

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += (long int)svcntd())
	{
			svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(a->real_dim));
		//ret += get_dvector_i(a, i) * get_dvector_i(a, i);

		a4 = svld1_f64(pg, &get_dvector_i(a, i));

		// tmp4 += a4 * a4
		tmp4 = svmla_f64_m(pg, tmp4, a4, a4);
	}

	ret = svaddv_f64(svptrue_b64(), tmp4);

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
    float64x2_t sum2 = vdupq_n_f64(0.0);
    for (long i = 0; i < a->real_dim; i += 2) {
        float64x2_t a2 = vld1q_f64(&a->element[i]);
        sum2 = vfmaq_f64(sum2, a2, a2);
    }
    ret = vgetq_lane_f64(sum2, 0) + vgetq_lane_f64(sum2, 1);

#else // others
	for(i = 0; i < a->dim; i++)
		ret += get_dvector_i(a, i) * get_dvector_i(a, i);
#endif // __AVX2__

	return sqrt(ret);
}

/* ||a||_infty */
double normi_dvector(DVector a)
{
	double ret, tmp;
	long int i;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, ret4, tmp4;

	// tmp4 ;= 0
	ret4 = _mm256_setzero_pd();

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//tmp = fabs(get_dvector_i(a, i));
		//if(ret < tmp)
		//	ret = tmp;

		tmp4 = _bncavx2_fabs(_mm256_load_pd(&get_dvector_i(a, i)));

		// ret4 := max(ret4, tmp4)
		ret4 = _mm256_max_pd(ret4, tmp4);
	}

	ret = ret4[0];
	if(ret < ret4[1]) ret = ret4[1];
	if(ret < ret4[2]) ret = ret4[2];
	if(ret < ret4[3]) ret = ret4[3];

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, ret8, tmp8;

	// tmp8 := 0
	tmp8 = _mm512_setzero_pd();

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//if(ret < tmp)
		//	ret = tmp;

		tmp8 = _bncavx512_fabs(_mm512_load_pd(&get_dvector_i(a, i)));

		// ret8 := max(ret8, tmp8)
		ret8 = _mm512_max_pd(ret8, tmp8);
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
	svfloat64_t a4, ret4, tmp4;

	// tmp4 ;= 0
	ret4 = svdup_f64(0.0);

	//for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += (long int)svcntd())
	{
			svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(a->real_dim));
		//tmp = fabs(get_dvector_i(a, i));
		//if(ret < tmp)
		//	ret = tmp;

		tmp4 = svabs_f64_x(pg, svld1_f64(pg, &get_dvector_i(a, i)));

		// ret4 := max(ret4, tmp4)
		ret4 = svmax_f64_x(pg, ret4, tmp4);
	}

	ret = svmaxv_f64(svptrue_b64(), ret4);

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
    float64x2_t max2 = vdupq_n_f64(0.0);
    for (long i = 0; i < a->real_dim; i += 2) {
        float64x2_t a2 = vabsq_f64(vld1q_f64(&a->element[i]));
        max2 = vmaxq_f64(max2, a2);
    }
    ret = fmax(vgetq_lane_f64(max2, 0), vgetq_lane_f64(max2, 1));

#else // others
	ret = fabs(get_dvector_i(a, 0));
	for(i = 1; i < a->dim; i++)
	{
		tmp = fabs(get_dvector_i(a, i));
		if(ret < tmp)
			ret = tmp;
	}
#endif // __AVX2__

	return ret;
}

/* c := a */
void subst_dvector(DVector c, DVector a)
{
	long int i;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
//	for(i = 0; i < a->dim; i++)
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i));
		_mm256_store_pd(&get_dvector_i(c, i), _mm256_load_pd(&get_dvector_i(a, i)));
	}
#elif defined(__AVX512F__) // __AVX512F__
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i));
		_mm512_store_pd(&get_dvector_i(c, i), _mm512_load_pd(&get_dvector_i(a, i)));
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    for (long i = 0; i < a->real_dim; i += (long int)svcntd()) {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(a->real_dim));
        svfloat64_t a2 = svld1_f64(pg, &a->element[i]);
        svst1_f64(pg, &c->element[i], a2);
    }

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
    for (long i = 0; i < a->real_dim; i += 2) {
        float64x2_t a2 = vld1q_f64(&a->element[i]);
        vst1q_f64(&c->element[i], a2);
    }

#else // others
	for(i = 0; i < a->dim; i++)
		set_dvector_i(c, i, get_dvector_i(a, i));
#endif // __AVX2__
}

/* c := 0 */
void set0_dvector(DVector c)
{
	long int i;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();

	//for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, (double)0);
		_mm256_store_pd(&get_dvector_i(c, i), zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();

	//for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, (double)0);
		_mm512_store_pd(&get_dvector_i(c, i), zero8);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    svfloat64_t zero2 = svdup_f64(0.0);
    for (long i = 0; i < c->real_dim; i += (long int)svcntd()) {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(c->real_dim));
        svst1_f64(pg, &c->element[i], zero2);
        }

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
    float64x2_t zero2 = vdupq_n_f64(0.0);
    for (long i = 0; i < c->real_dim; i += 2)
        vst1q_f64(&c->element[i], zero2);

#else // others
	for(i = 0; i < c->dim; i++)
		set_dvector_i(c, i, (double)0);
#endif // __AVX2__
}

/* append 2005.07/12 */
/*
	ret(index_start) = src(src_index_start)
	 ...
	ret(index_end  ) = src(src_index_end)
*/
void copy_dvector_ij(DVector ret, long int index_start, long int index_end, DVector src, long int src_index_start, long int src_index_end)
{
	long int i, itmp;

	if((src_index_end - src_index_start) != (index_end - index_start))
	{
		fprintf(stderr, "Invalid index!(copy_dvector_ij)\n");
		return;
	}

	for(i = 0; i <= (index_end - index_start); i++)
	{
		set_dvector_i(ret, index_start + i, get_dvector_i(src, src_index_start + i));
//		printf("%d <----------------------------------> %d\n", index_start + i, src_index_start + i);
	}
}


// 2022-11-17(Thu) T.Kouya
// absmax_dvector
double absmax_dvector(long int *max_index, DVector vec)
{
    long int i, max_i, dim = vec->dim;
    double ret, abs_val;

    max_i = 0;
    ret = 0.0;
    for(i = 0; i < dim; i++)
    {
        abs_val = fabs(get_dvector_i(vec, i));
        if(ret < abs_val)
        {
            ret = abs_val;
            max_i = i;
        }
    }

    if(max_index != NULL)
        *max_index = max_i;

    return ret;
}

// old
//#define get_dmatrix_ij(mat, i, j) ( *((mat)->element + (i) * (mat)->col_dim + (j)) )
//#define set_dmatrix_ij(mat, i, j, val) ( *((mat)->element + (i) * (mat)->col_dim + (j)) = (val) )

// new
#define get_dmatrix_ij(mat, i, j) ( *((mat)->element + (i) * (mat)->real_col_dim + (j)) )
#define set_dmatrix_ij(mat, i, j, val) ( *((mat)->element + (i) * (mat)->real_col_dim + (j)) = (val) )


/*************************************************/
/* Matrix Caluculations for DMatrix              */
/*
DMatrix init_dmatrix(long int row_dimension, long int col_dimension)
void free_dmatrix(DMatrix mat)
double normf_dmatrix(DMatrix mat)
double normi_dmatrix(DMatrix mat)
double norm1_dmatrix(DMatrix mat)
void add_dmatrix(DMatrix c, DMatrix a, DMatrix b);
void sub_dmatrix(DMatrix c, DMatrix a, DMatrix b);
void mul_dmatrix(DMatrix c, DMatrix a, DMatrix b);
void transpose_dmatrix(DMatrix c, DMatrix a);
void mul_dmatrix_dvec(DVector v, DMatrix a, DVector vb)
void mul_dmatrixt_dvec(DVector v, DMatrix a, DVector vb)
void inv_dmatrix(DMatrix a);
void subst_dmatrux(DMatrix c, DMatrix a);
*/
/*************************************************/
DMatrix init_dmatrix(long int row_dimension, long int col_dimension)
{
	DMatrix ret = NULL;
	long int i, j;
	long real_row_dim, real_col_dim, real_total_dim;

	if(row_dimension <= 0 || col_dimension <= 0)
	{
		fprintf(stderr, "ERROR: init_dmatrix\n");
		return ret;
	}

	ret = (DMatrix)BNC_MALLOC(sizeof(dmatrix));
	if(ret == NULL)
		return ret;

	//ret->element = (double *)BNC_CALLOC(row_dimension * col_dimension, sizeof(double));

	// real_dim is the nearest positive multiplier of _BNC_D_WIDTH
	real_row_dim = (long int)ceil((double)row_dimension / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
	//real_row_dim = row_dimension; // row-major way
	real_col_dim = (long int)ceil((double)col_dimension / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
	//printf("row, real_row, col, real_col = %ld, %ld, %ld, %ld\n", row_dimension, real_row_dim, col_dimension, real_col_dim);
	real_total_dim = real_row_dim * real_col_dim;

	ret->element = (double *)BNC_CALLOC(real_total_dim, sizeof(double));

	if(ret->element == NULL)
	{
		free(ret);
		return ret;
	}

	/* All 0 */
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
/*	
	for(i = 0; i < real_row_dim; i++)
	{
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
			_mm256_store_pd((ret->element + i * real_col_dim + j), zero4);
	}
*/
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
		_mm256_store_pd(&(ret->element[i]), zero4);


#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
/*
	for(i = 0; i < real_row_dim; i++)
	{
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
			_mm512_store_pd((ret->element + i * real_col_dim + j), zero8);
	}
*/
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
		_mm512_store_pd(&(ret->element[i]), zero8);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    svfloat64_t zero2 = svdup_f64(0.0);

	for (i = 0; i < real_total_dim; i += (long int)svcntd()) {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(real_total_dim));
        svst1_f64(pg, &ret->element[i], zero2);
    }

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
    float64x2_t zero2 = vdupq_n_f64(0.0);

	for (i = 0; i < real_total_dim; i += 2) {
        vst1q_f64(&ret->element[i], zero2);
    }

#else // others

/*	for(i = 0; i < real_row_dim; i++)
		for(j = 0; j < real_col_dim; j++)
			*(ret->element + i * real_col_dim + j) = (double)0.0;
*/
	for(i = 0; i < real_total_dim; i++)
		ret->element[i] = 0.0;

#endif // __AVX2__

	ret->row_dim = row_dimension;
	ret->col_dim = col_dimension;

	ret->real_row_dim = real_row_dim;
	ret->real_col_dim = real_col_dim;


	return ret;
}

void free_dmatrix(DMatrix mat)
{
	if(mat == NULL)
		return;

	if(mat->element != NULL)
		free(mat->element);

	free(mat);
}

// print_dmatrix
void print_dmatrix(DMatrix mat)
{
    long int i, j;

    for(i = 0; i < mat->row_dim; i++)
    {
        printf("%5ld: ", i);
        for(j = 0; j < mat->col_dim; j++)
            printf("%25.17e ", get_dmatrix_ij(mat, i, j));
        printf("\n");
    }
}

/*************************************************/
/* Matrix Caluculations for DMatrix              */
/*
double normf_dmatrix(DMatrix mat)
double normi_dmatrix(DMatrix mat)
double norm1_dmatrix(DMatrix mat)
void add_dmatrix(DMatrix c, DMatrix a, DMatrix b);
void sub_dmatrix(DMatrix c, DMatrix a, DMatrix b);
void mul_dmatrix(DMatrix c, DMatrix a, DMatrix b);
void transpose_dmatrix(DMatrix c, DMatrix a);
void mul_dmatrix_dvec(DVector v, DMatrix a, DVector vb)
void mul_dmatrixt_dvec(DVector v, DMatrix a, DVector vb)
void inv_dmatrix(DMatrix a);
void subst_dmatrux(DMatrix c, DMatrix a);
*/
/*************************************************/
/* Frobenius Norm of Matrix */
double normf_dmatrix(DMatrix mat)
{
	long int i, j;
	double ret;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d mat4, tmp4;

	// tmp4 ;= 0
	tmp4 = _mm256_setzero_pd();

	for(i = 0; i < mat->row_dim; i++)
	{
		//for(j = 0; j < mat->col_dim; j++)
		for(j = 0; j < mat->real_col_dim; j += _BNC_D_WIDTH)
		{
			//ret += (get_dmatrix_ij(mat, i, j) * get_dmatrix_ij(mat, i, j));

			mat4 = _mm256_load_pd(&get_dmatrix_ij(mat, i, j));

			// tmp4 += mat4 * mat4
			tmp4 = _mm256_fmadd_pd(mat4, mat4, tmp4);
		}
	}

	ret = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3];

#elif defined(__AVX512F__) // __AVX512F__
	__m512d mat8, tmp8;

	// tmp4 ;= 0
	tmp8 = _mm512_setzero_pd();

	for(i = 0; i < mat->row_dim; i++)
	{
		//for(j = 0; j < mat->col_dim; j++)
		for(j = 0; j < mat->real_col_dim; j += _BNC_D_WIDTH)
		{
			//ret += (get_dmatrix_ij(mat, i, j) * get_dmatrix_ij(mat, i, j));

			mat8 = _mm512_load_pd(&get_dmatrix_ij(mat, i, j));

			// tmp8 += mat8 * mat8
			tmp8 = _mm512_fmadd_pd(mat8, mat8, tmp8);
		}
	}

	ret = tmp8[0] + tmp8[1] + tmp8[2] + tmp8[3] + tmp8[4] + tmp8[5] + tmp8[6] + tmp8[7];

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat64_t mat4, tmp4;

	// tmp4 ;= 0
	tmp4 = svdup_f64(0.0);

	for(i = 0; i < mat->row_dim; i++)
	{
		//for(j = 0; j < mat->col_dim; j++)
		for(j = 0; j < mat->real_col_dim; j += (long int)svcntd())
		{
			svbool_t pg = svwhilelt_b64_s64((int64_t)j, (int64_t)(mat->real_col_dim));
			//ret += (get_dmatrix_ij(mat, i, j) * get_dmatrix_ij(mat, i, j));

			mat4 = svld1_f64(pg, &get_dmatrix_ij(mat, i, j));

			// tmp4 += mat4 * mat4
			tmp4 = svmla_f64_m(pg, tmp4, mat4, mat4);
		}
	}

	ret = svaddv_f64(svptrue_b64(), tmp4);

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
    float64x2_t mat2, tmp2;

	tmp2 = vdupq_n_f64(0.0);
	for(i = 0; i < mat->row_dim; i++)
	{
		//for(j = 0; j < mat->col_dim; j++)
		for(j = 0; j < mat->real_col_dim; j += _BNC_D_WIDTH)
		{
	    	mat2 = vld1q_f64(&mat->element[i]);

			// tmp2 += mat2 * mat2
			tmp2 = vfmaq_f64(tmp2, mat2, mat2);
		}
    }

	ret = vgetq_lane_f64(tmp2, 0) + vgetq_lane_f64(tmp2, 1);

#else // others
	ret = 0.0;
	for(i = 0; i < mat->row_dim; i++)
		for(j = 0; j < mat->col_dim; j++)
			ret += (get_dmatrix_ij(mat, i, j) * get_dmatrix_ij(mat, i, j));
#endif // __AVX2__

	ret = sqrt(ret);

	return ret;
}

/* Frobenius Norm of Matrix: array type */
double normf_dmatrix_array(double mat[], int row_dim, int col_dim)
{
	int i, j;
	double ret;

	ret = 0.0;
	for(i = 0; i < row_dim; i++)
		for(j = 0; j < col_dim; j++)
			ret += mat[i * col_dim + j] * mat[i * col_dim + j];

	ret = sqrt(ret);

	return ret;
}

/* Infinity Norm of Matrix */
double normi_dmatrix(DMatrix mat)
{
	long int i, j;
	double ret, sum;

	ret = 0.0;
	for(i = 0; i < mat->row_dim; i++)
	{
		sum = 0.0;
		for(j = 0; j < mat->col_dim; j++)
			sum += fabs(get_dmatrix_ij(mat, i, j));
		if(ret < sum)
			ret = sum;
	}

	return ret;
}

/* 1 Norm of Matrix */
double norm1_dmatrix(DMatrix mat)
{
	long int i, j;
	double ret, sum;

	ret = 0.0;
	for(j = 0; j < mat->col_dim; j++)
	{
		sum = 0.0;
		for(i = 0; i < mat->row_dim; i++)
			sum += fabs(get_dmatrix_ij(mat, i, j));
		if(ret < sum)
			ret = sum;
	}

	return ret;
}

/* c = a + b */
void add_dmatrix(DMatrix c, DMatrix a, DMatrix b)
{
	long int i, j, index, row_dim, col_dim, real_col_dim, real_total_dim;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_dmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_dmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = c->real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, b4, tmp4;

/*
	for(i = 0; i < row_dim; i++)
	{
//		for(j = 0; j < col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			a4 = _mm256_load_pd(&get_dmatrix_ij(a, i, j));
			b4 = _mm256_load_pd(&get_dmatrix_ij(b, i, j));
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j) + get_dmatrix_ij(b, i, j));
			tmp4 = _mm256_add_pd(a4, b4);
			_mm256_store_pd(&get_dmatrix_ij(c, i, j), tmp4);
		}
	}
*/
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		a4 = _mm256_load_pd(&(a->element[index]));
		b4 = _mm256_load_pd(&(b->element[index]));

		_mm256_store_pd(&(c->element[index]), _mm256_add_pd(a4, b4));
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, b8, tmp8;

/*
	for(i = 0; i < row_dim; i++)
	{
//		for(j = 0; j < col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			a8 = _mm512_load_pd(&get_dmatrix_ij(a, i, j));
			b8 = _mm512_load_pd(&get_dmatrix_ij(b, i, j));
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j) + get_dmatrix_ij(b, i, j));
			tmp8 = _mm512_add_pd(a8, b8);
			_mm512_store_pd(&get_dmatrix_ij(c, i, j), tmp8);
		}
	}
*/
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		a8 = _mm512_load_pd(&(a->element[index]));
		b8 = _mm512_load_pd(&(b->element[index]));

		_mm512_store_pd(&(c->element[index]), _mm512_add_pd(a8, b8));
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    for (long i = 0; i < real_total_dim; i += (long int)svcntd()) {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(real_total_dim));
        svfloat64_t a2 = svld1_f64(pg, &a->element[i]);
        svfloat64_t b2 = svld1_f64(pg, &b->element[i]);
        svfloat64_t c2 = svadd_f64_x(pg, a2, b2);
        svst1_f64(pg, &c->element[i], c2);
    }

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
    for (long i = 0; i < real_total_dim; i += 2) {
        float64x2_t a2 = vld1q_f64(&a->element[i]);
        float64x2_t b2 = vld1q_f64(&b->element[i]);
        float64x2_t c2 = vaddq_f64(a2, b2);
        vst1q_f64(&c->element[i], c2);
    }

#else // others
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
			set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j) + get_dmatrix_ij(b, i, j));
	}
#endif // __AVX2__

}

/* c = a - b */
void sub_dmatrix(DMatrix c, DMatrix a, DMatrix b)
{
	long int i, j, index, row_dim, col_dim, real_col_dim, real_total_dim;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: sub_dmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: sub_dmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = c->real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, b4, tmp4;
/*
	for(i = 0; i < row_dim; i++)
	{
//		for(j = 0; j < col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			a4 = _mm256_load_pd(&get_dmatrix_ij(a, i, j));
			b4 = _mm256_load_pd(&get_dmatrix_ij(b, i, j));
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j) - get_dmatrix_ij(b, i, j));
			tmp4 = _mm256_sub_pd(a4, b4);
			_mm256_store_pd(&get_dmatrix_ij(c, i, j), tmp4);
		}
	}
	*/
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		a4 = _mm256_load_pd(&(a->element[index]));
		b4 = _mm256_load_pd(&(b->element[index]));

		_mm256_store_pd(&(c->element[index]), _mm256_sub_pd(a4, b4));
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, b8, tmp8;

/*
	for(i = 0; i < row_dim; i++)
	{
//		for(j = 0; j < col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			a8 = _mm512_load_pd(&get_dmatrix_ij(a, i, j));
			b8 = _mm512_load_pd(&get_dmatrix_ij(b, i, j));
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j) - get_dmatrix_ij(b, i, j));
			tmp8 = _mm512_sub_pd(a8, b8);
			_mm512_store_pd(&get_dmatrix_ij(c, i, j), tmp8);
		}
	}
*/
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		a8 = _mm512_load_pd(&(a->element[index]));
		b8 = _mm512_load_pd(&(b->element[index]));

		_mm512_store_pd(&(c->element[index]), _mm512_sub_pd(a8, b8));
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    for (long i = 0; i < real_total_dim; i += (long int)svcntd()) {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(real_total_dim));
        svfloat64_t a2 = svld1_f64(pg, &a->element[i]);
        svfloat64_t b2 = svld1_f64(pg, &b->element[i]);
        svfloat64_t c2 = svsub_f64_x(pg, a2, b2);
        svst1_f64(pg, &c->element[i], c2);
    }
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
    for (long i = 0; i < real_total_dim; i += 2) {
        float64x2_t a2 = vld1q_f64(&a->element[i]);
        float64x2_t b2 = vld1q_f64(&b->element[i]);
        float64x2_t c2 = vsubq_f64(a2, b2);
        vst1q_f64(&c->element[i], c2);
    }
#else // others
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
			set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j) - get_dmatrix_ij(b, i, j));
	}
#endif // __AVX2__
}

/* c = sc * a */
void cmul_dmatrix(DMatrix c, double sc, DMatrix a)
{
	long int i, j, row_dim, col_dim, real_col_dim;

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_dmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_dmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, sc4, c4;

	sc4 = _mm256_set_pd(sc, sc, sc, sc);

	for(i = 0; i < row_dim; i++)
	{
//		for(j = 0; j < col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, sc * get_dmatrix_ij(a, i, j));
			a4 = _mm256_load_pd(&get_dmatrix_ij(a, i, j));
			c4 = _mm256_mul_pd(sc4, a4);
			_mm256_store_pd(&get_dmatrix_ij(c, i, j), c4);
		}
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, sc8, c8;

	sc8 = _mm512_set_pd(sc, sc, sc, sc, sc, sc, sc, sc);

	for(i = 0; i < row_dim; i++)
	{
//		for(j = 0; j < col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, sc * get_dmatrix_ij(a, i, j));
			a8 = _mm512_load_pd(&get_dmatrix_ij(a, i, j));
			c8 = _mm512_mul_pd(sc8, a8);
			_mm512_store_pd(&get_dmatrix_ij(c, i, j), c8);
		}
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    svfloat64_t val2 = svdup_f64(sc);

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < real_col_dim; j += (long int)svcntd())
		{
		svbool_t pg = svwhilelt_b64_s64((int64_t)j, (int64_t)(real_col_dim));
			svfloat64_t a2 = svld1_f64(pg, &get_dmatrix_ij(a, i, j));
			svfloat64_t c2 = svmul_f64_x(pg, val2, a2);
			svst1_f64(pg, &get_dmatrix_ij(c, i, j), c2);
		}
    }

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
    float64x2_t val2 = vdupq_n_f64(sc);

	for(i = 0; i < row_dim; i++)
	{
//		for(j = 0; j < col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			float64x2_t a2 = vld1q_f64(&get_dmatrix_ij(a, i, j));
			float64x2_t c2 = vmulq_f64(val2, a2);
			vst1q_f64(&get_dmatrix_ij(c, i, j), c2);
		}
    }

#else // others
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
			set_dmatrix_ij(c, i, j, sc * get_dmatrix_ij(a, i, j));
	}
#endif // __AVX2__
}

/* c = a * b */
void mul_dmatrix(DMatrix c, DMatrix a, DMatrix b)
{
	long int i, j, k;
	double tmp;

	/* dimension check */
	if((c->row_dim != a->row_dim) || (c->col_dim != b->row_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: mul_dmatrix\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, b4, tmp4;

	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
		{
			//tmp = 0.0;
			tmp4 = _mm256_setzero_pd();

			//for(k = 0; k < a->col_dim; k++)
			for(k = 0; k < a->real_col_dim; k += _BNC_D_WIDTH)
			{
				a4 = _mm256_load_pd(&get_dmatrix_ij(a, i, k));
				b4 = _mm256_set_pd(
					get_dmatrix_ij(b, k + 3, j),
					get_dmatrix_ij(b, k + 2, j),
					get_dmatrix_ij(b, k + 1, j),
					get_dmatrix_ij(b, k    , j)
				);
				tmp4 = _mm256_fmadd_pd(a4, b4, tmp4);
			}
			set_dmatrix_ij(c, i, j, tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3]);
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, b8, tmp8;

	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
		{
			//tmp = 0.0;
			tmp8 = _mm512_setzero_pd();

			//for(k = 0; k < a->col_dim; k++)
			for(k = 0; k < a->real_col_dim; k += _BNC_D_WIDTH)
			{
				a8 = _mm512_load_pd(&get_dmatrix_ij(a, i, k));
				b8 = _mm512_set_pd(
					get_dmatrix_ij(b, k + 7, j),
					get_dmatrix_ij(b, k + 6, j),
					get_dmatrix_ij(b, k + 5, j),
					get_dmatrix_ij(b, k + 4, j),
					get_dmatrix_ij(b, k + 3, j),
					get_dmatrix_ij(b, k + 2, j),
					get_dmatrix_ij(b, k + 1, j),
					get_dmatrix_ij(b, k    , j)
				);
				tmp8 = _mm512_fmadd_pd(a8, b8, tmp8);
			}
			set_dmatrix_ij(c, i, j, tmp8[0] + tmp8[1] + tmp8[2] + tmp8[3] + tmp8[4] + tmp8[5] + tmp8[6] + tmp8[7]);
		}
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, gather)
	{
	svfloat64_t a_v, b_v, tmp;
	svint64_t bidx;
	for(i = 0; i < c->row_dim; i++)
		for(j = 0; j < c->col_dim; j++)
		{
			tmp = svdup_f64(0.0);
			for(k = 0; k < a->real_col_dim; k += (long int)svcntd())
			{
				svbool_t pg = svwhilelt_b64_s64((int64_t)k, (int64_t)a->real_col_dim);
				a_v = svld1_f64(pg, &get_dmatrix_ij(a, i, k));
				bidx = svindex_s64((int64_t)(k * b->real_col_dim + j), (int64_t)b->real_col_dim);
				b_v = svld1_gather_s64index_f64(pg, b->element, bidx);
				tmp = svmla_f64_m(pg, tmp, a_v, b_v);
			}
			set_dmatrix_ij(c, i, j, svaddv_f64(svptrue_b64(), tmp));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
    for (long i = 0; i < c->row_dim; i++) {
        for (long j = 0; j < c->col_dim; j++) {
            double sum = 0.0;

            float64x2_t acc = vdupq_n_f64(0.0);
            long k;
            for (k = 0; k <= a->real_col_dim - 2; k += 2) {
                float64x2_t va = vld1q_f64(&a->element[i * a->real_col_dim + k]);
                float64x2_t vb = { b->element[k * b->real_col_dim + j], b->element[(k + 1) * b->real_col_dim + j] };
                acc = vfmaq_f64(acc, va, vb);
            }
            sum = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);
            for (; k < a->col_dim; k++) {
                sum += a->element[i * a->real_col_dim + k] * b->element[k * b->real_col_dim + j];
            }
            c->element[i * c->real_col_dim + j] = sum;
        }
    }

#else // others
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
		{
			tmp = 0.0;
			for(k = 0; k < a->col_dim; k++)
				tmp += get_dmatrix_ij(a, i, k) * get_dmatrix_ij(b, k, j);
			set_dmatrix_ij(c, i, j, tmp);
		}
	}

#endif // __AVX2__
}

/* c = a^T */
void transpose_dmatrix(DMatrix c, DMatrix a)
{
	long int i, j;
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
	__m256d aji4;

	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, j, i));
			aji4 = _mm256_set_pd(
				get_dmatrix_ij(a, j + 3, i),
				get_dmatrix_ij(a, j + 2, i),
				get_dmatrix_ij(a, j + 1, i),
				get_dmatrix_ij(a, j    , i)
			);
			_mm256_store_pd(&get_dmatrix_ij(c, i, j), aji4);
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d aji8;

	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, j, i));
			aji8 = _mm512_set_pd(
				get_dmatrix_ij(a, j + 7, i),
				get_dmatrix_ij(a, j + 6, i),
				get_dmatrix_ij(a, j + 5, i),
				get_dmatrix_ij(a, j + 4, i),
				get_dmatrix_ij(a, j + 3, i),
				get_dmatrix_ij(a, j + 2, i),
				get_dmatrix_ij(a, j + 1, i),
				get_dmatrix_ij(a, j    , i)
			);
			_mm512_store_pd(&get_dmatrix_ij(c, i, j), aji8);
		}
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, gather)
	{
	svfloat64_t aji;
	svint64_t vidx;
	for(i = 0; i < real_row_dim; i++)
		for(j = 0; j < real_col_dim; j += (long int)svcntd())
		{
			svbool_t pg = svwhilelt_b64_s64((int64_t)j, (int64_t)real_col_dim);
			vidx = svindex_s64((int64_t)(j * a->real_col_dim + i), (int64_t)a->real_col_dim);
			aji = svld1_gather_s64index_f64(pg, a->element, vidx);
			svst1_f64(pg, &get_dmatrix_ij(c, i, j), aji);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
    for (i = 0; i < real_row_dim; i++) {
        for (j = 0; j < real_col_dim; j += 2) {
            if (j + 1 < a->col_dim) {
                float64x2_t v = vld1q_f64(&a->element[i * a->real_col_dim + j]);

				c->element[j * c->real_col_dim + i]     = vgetq_lane_f64(v, 0);
                c->element[(j + 1) * c->real_col_dim + i] = vgetq_lane_f64(v, 1);
            } else {
                c->element[j * c->real_col_dim + i] = a->element[i * a->real_col_dim + j];
            }
        }
    }

#else // others
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, j, i));
	}
#endif // __AVX2__	
}

/* c := a */
void subst_dmatrix(DMatrix c, DMatrix a)
{
	long int i, j, index;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_dmatrix\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int real_total_dim = a->real_row_dim * a->real_col_dim;
/*
	for(i = 0; i < a->row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < a->real_col_dim; j += _BNC_D_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j));
			_mm256_store_pd(&get_dmatrix_ij(c, i, j), _mm256_load_pd(&get_dmatrix_ij(a, i, j)));
		}
	}
*/
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
		_mm256_store_pd(&c->element[index], _mm256_load_pd(&(a->element[index])));

#elif defined(__AVX512F__) // __AVX512F__
	long int real_total_dim = a->real_row_dim * a->real_col_dim;
/*
	for(i = 0; i < a->row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < a->real_col_dim; j += _BNC_D_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j));
			_mm512_store_pd(&get_dmatrix_ij(c, i, j), _mm512_load_pd(&get_dmatrix_ij(a, i, j)));
		}
	}
*/
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
		_mm512_store_pd(&c->element[index], _mm512_load_pd(&(a->element[index])));

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
	long int real_total_dim = a->real_row_dim * a->real_col_dim;

	for(index = 0; index < real_total_dim; index += (long int)svcntd()) {
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(real_total_dim));
        svst1_f64(pg, &c->element[index], svld1_f64(pg, &a->element[index]));
        }

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
	long int real_total_dim = a->real_row_dim * a->real_col_dim;

	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
        vst1q_f64(&c->element[index], vld1q_f64(&a->element[index]));

#else // others
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j));
		}
	}
#endif // __AVX2__
}

/* c := 0 */
void set0_dmatrix(DMatrix c)
{
	long int index, total_dim;

	total_dim = c->real_row_dim * c->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	for(index = 0; index < total_dim; index += _BNC_D_WIDTH)
		_mm256_store_pd(&(c->element[index]), zero4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(index = 0; index < total_dim; index += _BNC_D_WIDTH)
		_mm512_store_pd(&(c->element[index]), zero8);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    svfloat64_t zero = svdup_f64(0.0);
	for(index = 0; index < total_dim; index += (long int)svcntd()) {
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(total_dim));
		svst1_f64(pg, &(c->element[index]), zero);
		}

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
    float64x2_t zero = vdupq_n_f64(0.0);
	for(index = 0; index < total_dim; index += _BNC_D_WIDTH)
		vst1q_f64(&(c->element[index]), zero);

#else // others
	for(index = 0; index < total_dim; index++)
		c->element[index] = 0.0;

#endif // __AVX2__
/*
	for(i = 0; i < c->real_row_dim; i++)
	{
		for(j = 0; j < c->real_col_dim; j++)
			set_dmatrix_ij(c, i, j, 0.0);
	}
*/
}

/* c := I */
void setI_dmatrix(DMatrix c)
{
	long int index, total_dim;

	total_dim = c->real_row_dim * c->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	for(index = 0; index < total_dim; index += _BNC_D_WIDTH)
		_mm256_store_pd(&(c->element[index]), zero4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(index = 0; index < total_dim; index += _BNC_D_WIDTH)
		_mm512_store_pd(&(c->element[index]), zero8);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)
    svfloat64_t zero = svdup_f64(0.0);

	for(index = 0; index < total_dim; index += (long int)svcntd()) {
		svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)(total_dim));
		svst1_f64(pg, &(c->element[index]), zero);
		}

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
    float64x2_t zero = vdupq_n_f64(0.0);

	for(index = 0; index < total_dim; index += _BNC_D_WIDTH)
		vst1q_f64(&(c->element[index]), zero);

#else // others
	for(index = 0; index < total_dim; index++)
		c->element[index] = 0.0;

#endif // __AVX2__
	for(index = 0; index < c->row_dim; index++)
	{
		if(index < c->col_dim)
			set_dmatrix_ij(c, index, index, 1.0);
	}
}

/* v = a * vb */
void mul_dmatrix_dvec(DVector v, DMatrix a, DVector vb)
{
	long int i, j;
	double tmp;

	/* Check Dimension */
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: mul_dmatrix_dvec\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, vb4, tmp4;

	//printf("v = %d, a->real_row_dim, real_col_dim = %d, %d, vb = %d\n", v->real_dim, a->real_row_dim, a->real_col_dim, vb->real_dim);
	for(i = 0; i < a->row_dim; i++)
	{
		//tmp = 0.0;
		tmp4 = _mm256_setzero_pd();

		//for(j = 0; j < a->row_dim; j++)
		for(j = 0; j < a->real_col_dim; j += _BNC_D_WIDTH)
		{
			//tmp += get_dmatrix_ij(a, i, j) * get_dvector_i(vb, j);

			a4  = _mm256_load_pd(&get_dmatrix_ij(a, i, j));
			vb4 = _mm256_load_pd(&get_dvector_i(vb, j));
			tmp4 = _mm256_fmadd_pd(a4, vb4, tmp4);
		}
		//set_dvector_i(v, i, tmp);
		//_mm256_store_pd(&get_dvector_i(v, i), tmp4);
		set_dvector_i(v, i, tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3]);

	}
	//printf("v = %d, a->real_row_dim, real_col_dim = %d, %d, vb = %d\n", v->real_dim, a->real_row_dim, a->real_col_dim, vb->real_dim);
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, vb8, tmp8;

	for(i = 0; i < a->row_dim; i++)
	{
		//tmp = 0.0;
		tmp8 = _mm512_setzero_pd();

		//for(j = 0; j < a->row_dim; j++)
		for(j = 0; j < a->real_col_dim; j += _BNC_D_WIDTH)
		{
			//tmp += get_dmatrix_ij(a, i, j) * get_dvector_i(vb, j);

			a8  = _mm512_load_pd(&get_dmatrix_ij(a, i, j));
			vb8 = _mm512_load_pd(&get_dvector_i(vb, j));
			tmp8 = _mm512_fmadd_pd(a8, vb8, tmp8);
		}
		//set_dvector_i(v, i, tmp);
		//_mm512_store_pd(&get_dvector_i(v, i), tmp8);
		set_dvector_i(v, i, tmp8[0] + tmp8[1] + tmp8[2] + tmp8[3] + tmp8[4] + tmp8[5] + tmp8[6] + tmp8[7]);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native)
	svfloat64_t a4, vb4, tmp4;

	//printf("v = %d, a->real_row_dim, real_col_dim = %d, %d, vb = %d\n", v->real_dim, a->real_row_dim, a->real_col_dim, vb->real_dim);
	for(i = 0; i < a->row_dim; i++)
	{
		//tmp = 0.0;
		tmp4 = svdup_f64(0.0);

		//for(j = 0; j < a->row_dim; j++)
		for(j = 0; j < a->real_col_dim; j += (long int)svcntd())
		{
			svbool_t pg = svwhilelt_b64_s64((int64_t)j, (int64_t)(a->real_col_dim));
			//tmp += get_dmatrix_ij(a, i, j) * get_dvector_i(vb, j);

			a4  = svld1_f64(pg, &get_dmatrix_ij(a, i, j));
			vb4 = svld1_f64(pg, &get_dvector_i(vb, j));
			tmp4 = svmla_f64_m(pg, tmp4, a4, vb4);
		}
		//set_dvector_i(v, i, tmp);
		//svst1_f64(pg, &get_dvector_i(v, i), tmp4);
		set_dvector_i(v, i, svaddv_f64(svptrue_b64(), tmp4));

	}
	//printf("v = %d, a->real_row_dim, real_col_dim = %d, %d, vb = %d\n", v->real_dim, a->real_row_dim, a->real_col_dim, vb->real_dim);
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
    float64x2_t acc2, va2, vb2;

	for (i = 0; i < a->row_dim; i++) {
		acc2 = vdupq_n_f64(0.0);
		for (j = 0; j <= a->real_col_dim; j += 2) {
            va2 = vld1q_f64(&(get_dmatrix_ij(a, i, j)));
            vb2 = vld1q_f64(&(get_dvector_i(vb, j)));
            acc2 = vfmaq_f64(acc2, va2, vb2);
        }
        //for (; j < a->col_dim; j++) {
        //    sum += a->element[i * a->real_col_dim + j] * b->element[j];
        //}
		set_dvector_i(v, i, vgetq_lane_f64(acc2, 0) + vgetq_lane_f64(acc2, 1));
	}
#else // others
	for(i = 0; i < a->row_dim; i++)
	{
		tmp = 0.0;
		for(j = 0; j < a->col_dim; j++)
			tmp += get_dmatrix_ij(a, i, j) * get_dvector_i(vb, j);
		set_dvector_i(v, i, tmp);
	}
#endif // __AVX2__
}

/* v = a^T * vb */
//void mul_dmatrixt_dvec(DVector v, DMatrix a, DVector vb)
void mul_dmatrixt_dvec_old(DVector v, DMatrix a, DVector vb)
{
	long int i, j;
	double tmp;

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_dmatrixt_dvec\n");
		return;
	}
	for(i = 0; i < a->col_dim; i++)
	{
		tmp = 0.0;
		for(j = 0; j < a->row_dim; j++)
			tmp += get_dmatrix_ij(a, j, i) * get_dvector_i(vb, j);
		set_dvector_i(v, i, tmp);
	}
}
/* v = a^T * vb */
// Dangerous!
//void mul_dmatrixt_dvec_simd(DVector v, DMatrix a, DVector vb)
void mul_dmatrixt_dvec(DVector v, DMatrix a, DVector vb)
{
	long int i, j;
	double tmp;

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_dmatrixt_dvec\n");
		return;
	}
// Dangerous!
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, vb4, tmp4;

	for(i = 0; i < a->col_dim; i++)
	{
		//tmp = 0.0;
		tmp4 = _mm256_setzero_pd();

		//for(j = 0; j < a->row_dim; j++)
		for(j = 0; j < a->real_row_dim; j += _BNC_D_WIDTH)
		{
			//tmp += get_dmatrix_ij(a, j, i) * get_dvector_i(vb, j);

			//a4  = _mm256_load_pd(&get_dmatrix_ij(a, j, i));
			a4 = _mm256_set_pd(
				get_dmatrix_ij(a, j + 3, i),
				get_dmatrix_ij(a, j + 2, i),
				get_dmatrix_ij(a, j + 1, i),
				get_dmatrix_ij(a, j    , i)
			);
			vb4 = _mm256_load_pd(&get_dvector_i(vb, j));
			tmp4 = _mm256_fmadd_pd(a4, vb4, tmp4);
		}
		//set_dvector_i(v, i, tmp);
		//_mm256_store_pd(&get_dvector_i(v, i), tmp4);
		set_dvector_i(v, i, tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3]);
	}
// Dangerous!
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, vb8, tmp8;

	for(i = 0; i < a->col_dim; i++)
	{
		//tmp = 0.0;
		tmp8 = _mm512_setzero_pd();

		//for(j = 0; j < a->row_dim; j++)
		for(j = 0; j < a->real_row_dim; j += _BNC_D_WIDTH)
		{
			//tmp += get_dmatrix_ij(a, j, i) * get_dvector_i(vb, j);

			//a8  = _mm512_load_pd(&get_dmatrix_ij(a, j, i));
			a8 = _mm512_set_pd(
				get_dmatrix_ij(a, j + 7, i),
				get_dmatrix_ij(a, j + 6, i),
				get_dmatrix_ij(a, j + 5, i),
				get_dmatrix_ij(a, j + 4, i),
				get_dmatrix_ij(a, j + 3, i),
				get_dmatrix_ij(a, j + 2, i),
				get_dmatrix_ij(a, j + 1, i),
				get_dmatrix_ij(a, j    , i)
			);
			vb8 = _mm512_load_pd(&get_dvector_i(vb, j));
			tmp8 = _mm512_fmadd_pd(a8, vb8, tmp8);
		}
		//set_dvector_i(v, i, tmp);
		//_mm512_store_pd(&get_dvector_i(v, i), tmp8);
		set_dvector_i(v, i, tmp8[0] + tmp8[1] + tmp8[2] + tmp8[3] + tmp8[4] + tmp8[5] + tmp8[6] + tmp8[7]);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, gather)
	{
	svfloat64_t a_v, vb_v, tmp;
	svint64_t aidx;
	for(i = 0; i < a->col_dim; i++)
	{
		tmp = svdup_f64(0.0);
		for(j = 0; j < a->real_row_dim; j += (long int)svcntd())
		{
			svbool_t pg = svwhilelt_b64_s64((int64_t)j, (int64_t)a->real_row_dim);
			aidx = svindex_s64((int64_t)(j * a->real_col_dim + i), (int64_t)a->real_col_dim);
			a_v = svld1_gather_s64index_f64(pg, a->element, aidx);
			vb_v = svld1_f64(pg, &get_dvector_i(vb, j));
			tmp = svmla_f64_m(pg, tmp, a_v, vb_v);
		}
		set_dvector_i(v, i, svaddv_f64(svptrue_b64(), tmp));
	}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)
	double sum;
	float64x2_t acc2, va2, vb2;

	for (i = 0; i < a->col_dim; i++) {
		sum = 0.0;
        acc2 = vdupq_n_f64(0.0);
        for (j = 0; j <= a->real_row_dim; j += 2) {
            va2 = vsetq_lane_f64(get_dmatrix_ij(a, j    , i), va2, 0);
			va2 = vsetq_lane_f64(get_dmatrix_ij(a, j + 1, i), va2, 1);
            vb2 = vld1q_f64(&vb->element[j]);
            acc2 = vfmaq_f64(acc2, va2, vb2);
        }
		set_dvector_i(v, i, vgetq_lane_f64(acc2, 0) + vgetq_lane_f64(acc2, 1));
	}
#else // others
	for(i = 0; i < a->col_dim; i++)
	{
		tmp = 0.0;
		for(j = 0; j < a->row_dim; j++)
			tmp += get_dmatrix_ij(a, j, i) * get_dvector_i(vb, j);
		set_dvector_i(v, i, tmp);
	}
#endif // __AVX2__
}

/* a = a^(-1) */
/* square matrix only */
void inv_dmatrix(DMatrix a)
{
	long int i, j, k, dim;
	double tmp, aii;

	/* Check Dimensions */
	if(a->row_dim != a->col_dim)
	{
		fprintf(stderr, "ERROR: inv_dmatrix\n");
		return;
	}

	dim = a->row_dim;

	for(i = 0; i < dim; i++)
	{
		if(get_dmatrix_ij(a, i, i) == 0.0)
		{
			fprintf(stderr, "ERROR: inv_dmatrix: Pivot(%ld,%ld) is zero.\n", i, i);
			return;
		}

		aii = 1.0 / get_dmatrix_ij(a, i, i);
		set_dmatrix_ij(a, i, i, aii);

		for(j = 0; j < i; j++)
			set_dmatrix_ij(a, i, j, get_dmatrix_ij(a, i, j)* aii);
		for(j = i + 1; j < dim; j++)
			set_dmatrix_ij(a, i, j, get_dmatrix_ij(a, i, j) * aii);

		for(j = 0; j < i; j++)
		{
			for(k = 0; k < i; k++)
				set_dmatrix_ij(a, j, k, get_dmatrix_ij(a, j, k) - get_dmatrix_ij(a, j, i) * get_dmatrix_ij(a, i, k));
			for(k = i + 1; k < dim; k++)
				set_dmatrix_ij(a, j, k, get_dmatrix_ij(a, j, k) - get_dmatrix_ij(a, j, i) * get_dmatrix_ij(a, i, k));
		}

		for(j = i + 1; j < dim; j++)
		{
			for(k = 0; k < i; k++)
				set_dmatrix_ij(a, j, k, get_dmatrix_ij(a, j, k) - get_dmatrix_ij(a, j, i) * get_dmatrix_ij(a, i, k));
			for(k = i + 1; k < dim; k++)
				set_dmatrix_ij(a, j, k, get_dmatrix_ij(a, j, k) - get_dmatrix_ij(a, j, i) * get_dmatrix_ij(a, i, k));
		}

		for(j = 0; j < i; j++)
			set_dmatrix_ij(a, j, i, get_dmatrix_ij(a, j, i) * -aii);
		for(j = i + 1; j < dim; j++)
			set_dmatrix_ij(a, j, i, get_dmatrix_ij(a, j, i) * -aii);
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
int DLUdecomp(DMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	double dtmp, dmaxii;

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
	{
		dmaxii = fabs(get_dmatrix_ij(a, i, i));
		if(dmaxii == 0.0)
		{
			fprintf(stderr, "%ld : Error! (DLUdecomp)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
			set_dmatrix_ij(a, j, i, get_dmatrix_ij(a, j, i) / get_dmatrix_ij(a, i, i));
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
				set_dmatrix_ij(a, j, k, get_dmatrix_ij(a, j, k) - get_dmatrix_ij(a, j, i) * get_dmatrix_ij(a, i, k));
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
int SolveDLS(DVector answer, DMatrix lu, DVector b)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DMatrix lu: LU decomposed Matrix (given by user)   */
/*       DVector b: constant vector (given by user)         */
/*       DVector answer: Solution for linear system         */
/*       long int dim: Dimension of Matrix (given by user)  */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	double dtmp;

	dim = answer->dim;

	subst_dvector(answer, b);

/* Forward */
	for(i = 0; i < dim; i++)
	{
		if(get_dmatrix_ij(lu, i, i) == 0.0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveDLS, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
			set_dvector_i(answer, j, get_dvector_i(answer, j) - get_dmatrix_ij(lu, j, i) * get_dvector_i(answer, i));
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
			set_dvector_i(answer, i, get_dvector_i(answer, i) - get_dmatrix_ij(lu, i, j) * get_dvector_i(answer, j));
		set_dvector_i(answer, i, get_dvector_i(answer, i) / get_dmatrix_ij(lu, i, i));
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
int DLUdecompP(DMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DMatrix a: Matrix (given by user)                  */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
	double dtmp, dmaxii;

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		dmaxii = fabs(get_dmatrix_ij(a, ch[i], i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			dtmp = fabs(get_dmatrix_ij(a, ch[j], i));
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
			set_dmatrix_ij(a, ch[j], i, get_dmatrix_ij(a, ch[j], i) / get_dmatrix_ij(a, ch[i], i));
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
				set_dmatrix_ij(a, ch[j], k, get_dmatrix_ij(a, ch[j], k) - get_dmatrix_ij(a, ch[j], i) * get_dmatrix_ij(a, ch[i], k));
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
int SolveDLSP(DVector answer, DMatrix lu, DVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DMatrix lu[]: LU decomposed Matrix (given by user) */
/*       DVector b[]: constant vector (given by user)       */
/*       DVector answer[]: Solution for linear system       */
/*       long int ch: Row order (given by user)             */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	double dtmp;

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_dvector_i(answer, i, get_dvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		if(get_dmatrix_ij(lu, ch[i], i) == 0.0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveDLSP, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
			set_dvector_i(answer, j, get_dvector_i(answer, j) - get_dmatrix_ij(lu, ch[j], i) * get_dvector_i(answer, i));
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
			set_dvector_i(answer, i, get_dvector_i(answer, i) - get_dmatrix_ij(lu, ch[i], j) * get_dvector_i(answer, j));
		set_dvector_i(answer, i, get_dvector_i(answer, i) / get_dmatrix_ij(lu, ch[i], i));
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
int DLUdecompC(DMatrix a, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DMatrix a[]: Matrix (given by user)                */
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
	double dtmp, dmaxii;

	dim = a->col_dim;

	for(i = 0; i < dim; i++)
	{
		row_ch[i] = i;
		col_ch[i] = i;
	}

	for(i = 0; i < dim; i++)
	{
		/* Complete Pivoting */
		dmaxii = fabs(get_dmatrix_ij(a, row_ch[i], col_ch[i]));
		imax = i;
		jmax = i;
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				dtmp = fabs(get_dmatrix_ij(a, row_ch[j], col_ch[k]));
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
			set_dmatrix_ij(a, row_ch[j], col_ch[i], get_dmatrix_ij(a, row_ch[j], col_ch[i]) / get_dmatrix_ij(a, row_ch[i], col_ch[i]));
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
				set_dmatrix_ij(a, row_ch[j], col_ch[k], get_dmatrix_ij(a, row_ch[j], col_ch[k]) - get_dmatrix_ij(a, row_ch[j], col_ch[i]) * get_dmatrix_ij(a, row_ch[i], col_ch[k]));
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
int SolveDLSC(DVector answer, DMatrix lu, DVector b, long int row_ch[], long int col_ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DMatrix lu: LU decomposed Matrix (given by user)   */
/*       DVector b: constant vector (given by user)         */
/*       DVector answer: Solution for linear system         */
/*       long int row_ch[]: Row order (given by user)       */
/*       long int col_ch[]: Column order (given by user)    */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	double dtmp;

	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_dvector_i(answer, col_ch[i], get_dvector_i(b, row_ch[i]));

/* Forward */
	for(i = 0; i < dim; i++)
	{
		if(get_dmatrix_ij(lu, row_ch[i], col_ch[i]) == 0.0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveDLSC, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
			set_dvector_i(answer, col_ch[j], get_dvector_i(answer, col_ch[j]) - get_dmatrix_ij(lu, row_ch[j], col_ch[i]) * get_dvector_i(answer, col_ch[i]));
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
			set_dvector_i(answer, col_ch[i], get_dvector_i(answer, col_ch[i]) - get_dmatrix_ij(lu, row_ch[i], col_ch[j]) * get_dvector_i(answer, col_ch[j]));
		set_dvector_i(answer, col_ch[i], get_dvector_i(answer, col_ch[i]) / get_dmatrix_ij(lu, row_ch[i], col_ch[i]));
	}

	return 0;
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_dmatrix(DMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
	long int i, j, true_end;
	double tmp;

	true_end = (col_end > mat->col_dim) ? mat->col_dim : col_end;

	for(i = col_start; i < true_end; i++)
	{
		tmp = get_dmatrix_ij(mat, row_index0, i);
		set_dmatrix_ij(mat, row_index0, i, get_dmatrix_ij(mat, row_index1, i));
		set_dmatrix_ij(mat, row_index1, i, tmp);
	}
}

// 2022-11-17(Thu) T.Kouya
// absmax_row_dmatrix
double absmax_row_dmatrix(long int *max_j, long int row_index, DMatrix mat)
{
    long int j, max_index = 0;
    double mu, abs_aij;

	mu = fabs(get_dmatrix_ij(mat, row_index, 0));
    //rqd_abs(mu, get_qdmatrix_ij(mat, row_index, 0));

	for(j = 1; j < mat->col_dim; j++)
	{
		//rqd_abs(abs_aij, get_qdmatrix_ij(mat, row_index, j));
		abs_aij = fabs(get_dmatrix_ij(mat, row_index, j));
		//if(rqd_cmp(abs_aij, mu) > 0)
		if(abs_aij > mu)
        {
			mu = abs_aij;
            //rqd_set(mu, abs_aij);
            max_index = j;
        }
	}

    if(max_j != NULL)
        *max_j = max_index;

    return mu;
    //return;
}

// 2022-11-17(Thu) T.Kouya
// absmax_col_dmatrix
double absmax_col_dmatrix(long int *max_i, long int col_index, DMatrix mat)
{
    long int i, max_index = 0;
    double mu, abs_aij;

	mu = fabs(get_dmatrix_ij(mat, 0, col_index));
    //rqd_abs(mu, get_qdmatrix_ij(mat, 0, col_index));
	for(i = 1; i < mat->row_dim; i++)
	{
		//rqd_abs(abs_aij, get_qdmatrix_ij(mat, i, col_index));
        abs_aij = fabs(get_dmatrix_ij(mat, i, col_index));
		//if(rqd_cmp(abs_aij, mu) > 0)
		if(abs_aij > mu)
        {
			//rqd_set(mu, abs_aij);
            //abs_aij = mu;
            mu = abs_aij; // Fix! 2022-11-16(Wed)
            max_index = i;
        }
	}

    if(max_i != NULL)
        *max_i = max_index;

    //return;
    return mu;
}

// 2026-02-03(Tue) T.Kouya
// absmax_dmatrix
double absmax_dmatrix(long int *max_i, long int *max_j, DMatrix mat)
{
    long int i, j, max_row_index = 0, max_col_index = 0;
    double mu, abs_aij;

	mu = fabs(get_dmatrix_ij(mat, 0, 0));
	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
		{
			//rqd_abs(abs_aij, get_qdmatrix_ij(mat, i, col_index));
			abs_aij = fabs(get_dmatrix_ij(mat, i, j));
			//if(rqd_cmp(abs_aij, mu) > 0)
			if(abs_aij > mu)
			{
				//rqd_set(mu, abs_aij);
				//abs_aij = mu;
				mu = abs_aij; // Fix! 2022-11-16(Wed)
				max_row_index = i;
				max_col_index = j;
			}
		}
	}

    if(max_i != NULL)
        *max_i = max_row_index;
	if(max_j != NULL)
		*max_j = max_col_index;

    //return;
    return mu;
}

// Appended in 2024-05-09 T.Kouya
// from lanczos.c

/* c = a - sc * b */
void subcmul_dvector(DVector c, DVector a, double sc, DVector b)
{
	long int i;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: subcmul_dvector\n");
		return;
	}

	for(i = 0; i < c->dim; i++)
		set_dvector_i(c, i, get_dvector_i(a, i) - sc * get_dvector_i(b, i));

}

/* mat := (vec[0] vec[1] ... vec[n]) */
void subst_dmatrix_dvec(DMatrix mat, DVector vec[])
{
	long int i, j;

	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
			set_dmatrix_ij(mat, i, j, get_dvector_i(vec[j], i));
	}
}

/* (c, e) = two_add(a, b) */
void add_dvector_err(DVector c, DVector c_err, DVector a, DVector b)
{
	long int i;
	double ret, ret_err;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim) || (c_err->dim != c->dim) || (c_err->dim != a->dim) || (c_err->dim != b->dim))

	{
		fprintf(stderr, "ERROR: add_dvector_err\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, b4, c4, c4_err;

//	for(i = 0; i < c->dim; i++)
	for(i = 0; i < c->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i) + get_dvector_i(b, i));

		a4 = _mm256_load_pd(&get_dvector_i(a, i));
		b4 = _mm256_load_pd(&get_dvector_i(b, i));
		//c4 = _mm256_add_pd(a4, b4);
		c4 = _bncavx2_dtwo_sum(a4, b4, &c4_err);
		_mm256_store_pd(&get_dvector_i(c, i), c4);
		_mm256_store_pd(&get_dvector_i(c_err, i), c4_err);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, b8, c8, c8_err;

	for(i = 0; i < c->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i) + get_dvector_i(b, i));

		a8 = _mm512_load_pd(&get_dvector_i(a, i));
		b8 = _mm512_load_pd(&get_dvector_i(b, i));
		//c8 = _mm512_add_pd(a8, b8);
		c8 = _bncavx512_dtwo_sum(a8, b8, &c8_err);
		_mm512_store_pd(&get_dvector_i(c, i), c8);
		_mm512_store_pd(&get_dvector_i(c_err, i), c8_err);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (native, derived from NEON)

    svfloat64_t a2, b2, s2, e2;
    svfloat64_t bb2, t1, t2;

    /* NEON では 128-bit ベクトル = 2 x double
       _BNC_D_WIDTH は Arm では 2 が定義されている前提 */
    for(i = 0; i < c->real_dim; i += (long int)svcntd())
    {
		svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)(c->real_dim));
        a2 = svld1_f64(pg, &get_dvector_i(a, i));
        b2 = svld1_f64(pg, &get_dvector_i(b, i));

        s2 = svadd_f64_x(pg, a2, b2);

        bb2 = svsub_f64_x(pg, s2, a2);

        t1 = svsub_f64_x(pg, a2, svsub_f64_x(pg, s2, bb2));

        t2 = svsub_f64_x(pg, b2, bb2);

        e2 = svadd_f64_x(pg, t1, t2);

        svst1_f64(pg, &get_dvector_i(c, i),     s2);
        svst1_f64(pg, &get_dvector_i(c_err, i), e2);
    }

#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon&& defined(BNC_ENABLE_NEON)

    float64x2_t a2, b2, s2, e2;
    float64x2_t bb2, t1, t2;

    /* NEON では 128-bit ベクトル = 2 x double
       _BNC_D_WIDTH は Arm では 2 が定義されている前提 */
    for(i = 0; i < c->real_dim; i += _BNC_D_WIDTH)
    {
        // load a[i], a[i+1], b[i], b[i+1]
        a2 = vld1q_f64(&get_dvector_i(a, i));
        b2 = vld1q_f64(&get_dvector_i(b, i));

        // two_sum: s = a + b
        s2 = vaddq_f64(a2, b2);

        // bb = s - a
        bb2 = vsubq_f64(s2, a2);

        // t1 = a - (s - bb)
        t1 = vsubq_f64(a2, vsubq_f64(s2, bb2));

        // t2 = b - bb
        t2 = vsubq_f64(b2, bb2);

        // err = t1 + t2
        e2 = vaddq_f64(t1, t2);

        // store
        vst1q_f64(&get_dvector_i(c, i),     s2);
        vst1q_f64(&get_dvector_i(c_err, i), e2);
    }

#else // others
	for(i = 0; i < c->dim; i++)
	{
		ret = two_sum(get_dvector_i(a, i), get_dvector_i(b, i), &ret_err);
		//set_dvector_i(c, i, get_dvector_i(a, i) + get_dvector_i(b, i));
		set_dvector_i(c, i, ret);
		set_dvector_i(c_err, i, ret_err);
	}
#endif // __AVX2__

}

#if defined (__cplusplus)
}
#endif // defined (__cplusplus)
