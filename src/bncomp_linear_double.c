/********************************************************************************/
/* bncomp_linear_double.c: Parallelized Double Precision Linear Computation     */
/* Library with OpenMP                                                          */
/* Copyright (C) 2024 Tomonori Kouya                                            */
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
// BNCpack with OpenMP
#include "bncomp.h"
#include "oz_scheme.h" // split_dmatrix

// ARM NEON support added to bncomp_linear_double.c
// Add this include at the top of the file
//#if defined(__ARM_NEON)
//#include "_bncneon_d.h"
//#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//---------------------------------------
// Double - Vector operations with NEON
//---------------------------------------

/* c := a */
void _bncomp_subst_dvector(DVector c, DVector a)
{
	long int i;

#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl=(long)svcntd(); long _N=a->real_dim; long _ix;
		for(_ix=0;_ix<_N;_ix+=_vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svst1_f64(_pg, &(c->element[_ix]), svld1_f64(_pg, &(a->element[_ix])));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	#pragma omp parallel for
	for(i = 0; i < a->real_dim; i += 2) // NEON processes 2 doubles at a time
	{
		vst1q_f64(&(c->element[i]), vld1q_f64(&(a->element[i])));
	}
#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	#pragma omp parallel for
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&(c->element[i]), _mm256_load_pd(&(a->element[i])));
	}
#elif defined(__AVX512F__) // __AVX512F__
	#pragma omp parallel for
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&(c->element[i]), _mm512_load_pd(&(a->element[i])));
	}
#else // others
	#pragma omp parallel for
	for(i = 0; i < a->dim; i++)
		set_dvector_i(c, i, get_dvector_i(a, i));
#endif
}

/* c = a + b */
void _bncomp_add_dvector(DVector c, DVector a, DVector b)
{
	int thread_index;
	long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_dvector\n");
		return;
	}

#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl=(long)svcntd(); long _N=c->real_dim; long _ix;
		for(_ix=0;_ix<_N;_ix+=_vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svst1_f64(_pg, &(c->element[_ix]), svadd_f64_x(_pg, svld1_f64(_pg,&(a->element[_ix])), svld1_f64(_pg,&(b->element[_ix]))));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
    float64x2_t in_ret[BNCOMP_MAX_NUM_THREADS], in_a_val[BNCOMP_MAX_NUM_THREADS], in_b_val[BNCOMP_MAX_NUM_THREADS];

 	#pragma omp parallel for private(thread_index)
    for(index = 0; index < c->real_dim; index += 2)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index] = vld1q_f64(&(a->element[index]));
        in_b_val[thread_index] = vld1q_f64(&(b->element[index]));

        in_ret[thread_index] = vaddq_f64(in_a_val[thread_index], in_b_val[thread_index]);

        vst1q_f64(&(c->element[index]), in_ret[thread_index]);
    }
#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256d in_ret[BNCOMP_MAX_NUM_THREADS], in_a_val[BNCOMP_MAX_NUM_THREADS], in_b_val[BNCOMP_MAX_NUM_THREADS];

 	#pragma omp parallel for private(thread_index)
   for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index] = _mm256_load_pd(&(a->element[index]));
        in_b_val[thread_index] = _mm256_load_pd(&(b->element[index]));

        in_ret[thread_index] = _mm256_add_pd(in_a_val[thread_index], in_b_val[thread_index]);

        _mm256_store_pd(&(c->element[index]), in_ret[thread_index]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512d in_ret[BNCOMP_MAX_NUM_THREADS], in_a_val[BNCOMP_MAX_NUM_THREADS], in_b_val[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel for private(thread_index)
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index] = _mm512_load_pd(&(a->element[index]));
        in_b_val[thread_index] = _mm512_load_pd(&(b->element[index]));

        in_ret[thread_index] = _mm512_add_pd(in_a_val[thread_index], in_b_val[thread_index]);

        _mm512_store_pd(&(c->element[index]), in_ret[thread_index]);
   }
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

        tmp[thread_index] = get_dvector_i(a, i) + get_dvector_i(b, i);
		set_dvector_i(c, i, tmp[thread_index]);
	}
#endif
}

/* c = a - b */
void _bncomp_sub_dvector(DVector c, DVector a, DVector b)
{
	int thread_index;
	long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: _bncomp_sub_dvector\n");
		return;
	}
#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl=(long)svcntd(); long _N=c->real_dim; long _ix;
		for(_ix=0;_ix<_N;_ix+=_vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svst1_f64(_pg, &(c->element[_ix]), svsub_f64_x(_pg, svld1_f64(_pg,&(a->element[_ix])), svld1_f64(_pg,&(b->element[_ix]))));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
    float64x2_t in_ret[BNCOMP_MAX_NUM_THREADS], in_a_val[BNCOMP_MAX_NUM_THREADS], in_b_val[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel for private(thread_index)
    for(index = 0; index < c->real_dim; index += 2)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index] = vld1q_f64(&(a->element[index]));
        in_b_val[thread_index] = vld1q_f64(&(b->element[index]));

        in_ret[thread_index] = vsubq_f64(in_a_val[thread_index], in_b_val[thread_index]);

        vst1q_f64(&(c->element[index]), in_ret[thread_index]);
   }
#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256d in_ret[BNCOMP_MAX_NUM_THREADS], in_a_val[BNCOMP_MAX_NUM_THREADS], in_b_val[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel for private(thread_index)
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index] = _mm256_load_pd(&(a->element[index]));
        in_b_val[thread_index] = _mm256_load_pd(&(b->element[index]));

        in_ret[thread_index] = _mm256_sub_pd(in_a_val[thread_index], in_b_val[thread_index]);

        _mm256_store_pd(&(c->element[index]), in_ret[thread_index]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512d in_ret[BNCOMP_MAX_NUM_THREADS], in_a_val[BNCOMP_MAX_NUM_THREADS], in_b_val[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index] = _mm512_load_pd(&(a->element[index]));
        in_b_val[thread_index] = _mm512_load_pd(&(b->element[index]));

        in_ret[thread_index] = _mm512_sub_pd(in_a_val[thread_index], in_b_val[thread_index]);

        _mm512_store_pd(&(c->element[index]), in_ret[thread_index]);
   }
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

		tmp[thread_index] = get_dvector_i(a, i) - get_dvector_i(b, i);
		set_dvector_i(c, i, tmp[thread_index]);
	}
#endif
}

/* c = val * a */
void _bncomp_cmul_dvector(DVector c, double val, DVector a)
{
	int thread_index;
	long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: _bncomp_cmul_dvector\n");
		return;
	}

#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl=(long)svcntd(); long _N=a->real_dim; long _ix;
		for(_ix=0;_ix<_N;_ix+=_vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svst1_f64(_pg, &(c->element[_ix]), svmul_f64_x(_pg, svdup_n_f64(val), svld1_f64(_pg,&(a->element[_ix]))));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	float64x2_t a2[BNCOMP_MAX_NUM_THREADS], c2[BNCOMP_MAX_NUM_THREADS], val2[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		val2[thread_index] = vdupq_n_f64(val);
	}
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += 2)
	{
		thread_index = omp_get_thread_num();

		a2[thread_index] = vld1q_f64(&(a->element[index]));
		c2[thread_index] = vmulq_f64(val2[thread_index], a2[thread_index]);
		vst1q_f64(&(c->element[index]), c2[thread_index]);
	}
#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[BNCOMP_MAX_NUM_THREADS], c4[BNCOMP_MAX_NUM_THREADS], val4[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		val4[thread_index] = _mm256_set1_pd(val);
	}
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a4[thread_index] = _mm256_load_pd(&(a->element[index]));
		c4[thread_index] = _mm256_mul_pd(val4[thread_index], a4[thread_index]);
		_mm256_store_pd(&(c->element[index]), c4[thread_index]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[BNCOMP_MAX_NUM_THREADS], c8[BNCOMP_MAX_NUM_THREADS], val8[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		val8[thread_index] = _mm512_set1_pd(val);
	}

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a8[thread_index] = _mm512_load_pd(&(a->element[index]));
		c8[thread_index] = _mm512_mul_pd(val8[thread_index], a8[thread_index]);
		_mm512_store_pd(&(c->element[index]), c8[thread_index]);
	}
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

		tmp[thread_index] = val * get_dvector_i(a, i);
		set_dvector_i(c, i, tmp[thread_index]);
	}
#endif
}

/* (a, b) - inner product */
void _bncomp_ip_dvector(double ret, DVector a, DVector b)
{
	int thread_index;
	long int i, index;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: _bncomp_ip_dvector\n");
		return;
	}
#if (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	float64x2_t a2[BNCOMP_MAX_NUM_THREADS], b2[BNCOMP_MAX_NUM_THREADS], ret2, tmp2[BNCOMP_MAX_NUM_THREADS];

	ret2 = vdupq_n_f64(0.0);
	
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < a->real_dim; index += 2)
	{
		thread_index = omp_get_thread_num();

		a2[thread_index] = vld1q_f64(&(a->element[index]));
		b2[thread_index] = vld1q_f64(&(b->element[index]));

		tmp2[thread_index] = vmulq_f64(a2[thread_index], b2[thread_index]);
		#pragma omp critical
			ret2 = vaddq_f64(ret2, tmp2[thread_index]);
	}

	ret = _bncneon_dsum128(ret2);

#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[BNCOMP_MAX_NUM_THREADS], b4[BNCOMP_MAX_NUM_THREADS], ret4, tmp4[BNCOMP_MAX_NUM_THREADS];

	ret4 = _mm256_setzero_pd();
	
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a4[thread_index] = _mm256_load_pd(&(a->element[index]));
		b4[thread_index] = _mm256_load_pd(&(b->element[index]));

		tmp4[thread_index] = _mm256_mul_pd(a4[thread_index], b4[thread_index]);
		#pragma omp critical
			ret4 = _mm256_add_pd(ret4, tmp4[thread_index]);
	}

	ret = _bncavx2_dsum256d(ret4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[BNCOMP_MAX_NUM_THREADS], b8[BNCOMP_MAX_NUM_THREADS], ret8, tmp8[BNCOMP_MAX_NUM_THREADS];

	ret8 = _mm512_setzero_pd();
	
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a8[thread_index] = _mm512_load_pd(&(a->element[index]));
		b8[thread_index] = _mm512_load_pd(&(b->element[index]));

		tmp8[thread_index] = _mm512_mul_pd(a8[thread_index], b8[thread_index]);
		#pragma omp critical
			ret8 = _mm512_add_pd(ret8, tmp8[thread_index]);
	}

	ret = _bncavx2_dsum512d(ret8);

#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS];

	ret = 0.0;

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < a->dim; i++)
	{
		thread_index = omp_get_thread_num();

		tmp[thread_index] = get_dvector_i(a, i) * get_dvector_i(b, i);
	#pragma omp critical
		ret += tmp[thread_index];
	}
#endif

	return;
}

//---------------------------------------
// Double - Matrix operations with NEON
//---------------------------------------

/* c := a + b */
void _bncomp_add_dmatrix(DMatrix c, DMatrix a, DMatrix b)
{
	int thread_index;
	long int index, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim;

	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_dmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_dmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl=(long)svcntd(); long _N=c->real_row_dim * c->real_col_dim; long _ix;
		for(_ix=0;_ix<_N;_ix+=_vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svst1_f64(_pg, &(c->element[_ix]), svadd_f64_x(_pg, svld1_f64(_pg,&(a->element[_ix])), svld1_f64(_pg,&(b->element[_ix]))));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	float64x2_t tmp2[BNCOMP_MAX_NUM_THREADS], aij2[BNCOMP_MAX_NUM_THREADS], bij2[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += 2)
	{
		thread_index = omp_get_thread_num();
		
		aij2[thread_index] = vld1q_f64(&(a->element[index]));
		bij2[thread_index] = vld1q_f64(&(b->element[index]));

        tmp2[thread_index] = vaddq_f64(aij2[thread_index], bij2[thread_index]);

		vst1q_f64(&(c->element[index]), tmp2[thread_index]);
	}
#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[BNCOMP_MAX_NUM_THREADS], aij4[BNCOMP_MAX_NUM_THREADS], bij4[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij4[thread_index] = _mm256_load_pd(&(a->element[index]));
		bij4[thread_index] = _mm256_load_pd(&(b->element[index]));

        tmp4[thread_index] = _mm256_add_pd(aij4[thread_index], bij4[thread_index]);

		_mm256_store_pd(&(c->element[index]), tmp4[thread_index]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[BNCOMP_MAX_NUM_THREADS], aij8[BNCOMP_MAX_NUM_THREADS], bij8[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij8[thread_index] = _mm512_load_pd(&(a->element[index]));
		bij8[thread_index] = _mm512_load_pd(&(b->element[index]));

		tmp8[thread_index] = _mm512_add_pd(aij8[thread_index], bij8[thread_index]);

		_mm512_store_pd(&(c->element[index]), tmp8[thread_index]);
	}
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS], aij[BNCOMP_MAX_NUM_THREADS], bij[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index++)
	{
		thread_index = omp_get_thread_num();
		
		aij[thread_index] = a->element[index];
		bij[thread_index] = b->element[index];

		tmp[thread_index] = aij[thread_index] + bij[thread_index];

		c->element[index] = tmp[thread_index];
	}
#endif
}

/* c := a - b */
void _bncomp_sub_dmatrix(DMatrix c, DMatrix a, DMatrix b)
{
	int thread_index;
	long int index, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim;

	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_dmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_dmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl=(long)svcntd(); long _N=c->real_row_dim * c->real_col_dim; long _ix;
		for(_ix=0;_ix<_N;_ix+=_vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svst1_f64(_pg, &(c->element[_ix]), svsub_f64_x(_pg, svld1_f64(_pg,&(a->element[_ix])), svld1_f64(_pg,&(b->element[_ix]))));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	float64x2_t tmp2[BNCOMP_MAX_NUM_THREADS], aij2[BNCOMP_MAX_NUM_THREADS], bij2[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += 2)
	{
		thread_index = omp_get_thread_num();
		
		aij2[thread_index] = vld1q_f64(&(a->element[index]));
		bij2[thread_index] = vld1q_f64(&(b->element[index]));

		tmp2[thread_index] = vsubq_f64(aij2[thread_index], bij2[thread_index]);

		vst1q_f64(&(c->element[index]), tmp2[thread_index]);
	}
#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[BNCOMP_MAX_NUM_THREADS], aij4[BNCOMP_MAX_NUM_THREADS], bij4[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij4[thread_index] = _mm256_load_pd(&(a->element[index]));
		bij4[thread_index] = _mm256_load_pd(&(b->element[index]));

		tmp4[thread_index] = _mm256_sub_pd(aij4[thread_index], bij4[thread_index]);

		_mm256_store_pd(&(c->element[index]), tmp4[thread_index]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[BNCOMP_MAX_NUM_THREADS], aij8[BNCOMP_MAX_NUM_THREADS], bij8[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij8[thread_index] = _mm512_load_pd(&(a->element[index]));
		bij8[thread_index] = _mm512_load_pd(&(b->element[index]));

		tmp8[thread_index] = _mm512_sub_pd(aij8[thread_index], bij8[thread_index]);

		_mm512_store_pd(&(c->element[index]), tmp8[thread_index]);
	}
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS], aij[BNCOMP_MAX_NUM_THREADS], bij[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index++)
	{
		thread_index = omp_get_thread_num();
		
		aij[thread_index]= a->element[index];
		bij[thread_index]= b->element[index];

		tmp[thread_index] = aij[thread_index] - bij[thread_index];

		c->element[index] = tmp[thread_index];
	}
#endif
}

/* c := sc * a */
void _bncomp_cmul_dmatrix(DMatrix c, double sc, DMatrix a)
{
	long int i, j, index, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim;
	int thread_index;

	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_dmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_dmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl=(long)svcntd(); long _N=a->real_row_dim * a->real_col_dim; long _ix;
		for(_ix=0;_ix<_N;_ix+=_vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svst1_f64(_pg, &(c->element[_ix]), svmul_f64_x(_pg, svdup_n_f64(sc), svld1_f64(_pg,&(a->element[_ix]))));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	float64x2_t tmp2[BNCOMP_MAX_NUM_THREADS], aij2[BNCOMP_MAX_NUM_THREADS], sc2[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		sc2[thread_index] = vdupq_n_f64(sc);
	}
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += 2)
	{
		thread_index = omp_get_thread_num();
		
		aij2[thread_index] = vld1q_f64(&(a->element[index]));

		tmp2[thread_index] = vmulq_f64(sc2[thread_index], aij2[thread_index]);

		vst1q_f64(&(c->element[index]), tmp2[thread_index]);
	}
#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[BNCOMP_MAX_NUM_THREADS], aij4[BNCOMP_MAX_NUM_THREADS], sc4[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		sc4[thread_index] = _mm256_set1_pd(sc);
	}
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij4[thread_index] = _mm256_load_pd(&(a->element[index]));

		tmp4[thread_index] = _mm256_sub_pd(sc4[thread_index], aij4[thread_index]);

		_mm256_store_pd(&(c->element[index]), tmp4[thread_index]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[BNCOMP_MAX_NUM_THREADS], aij8[BNCOMP_MAX_NUM_THREADS], sc8[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		sc8[thread_index] = _mm512_set1_pd(sc);
	}
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij8[thread_index] = _mm512_load_pd(&(a->element[index]));

		tmp8[thread_index] = _mm512_mul_pd(sc8[thread_index], aij8[thread_index]);

		_mm512_store_pd(&(c->element[index]), tmp8[thread_index]);
	}
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS], aij[BNCOMP_MAX_NUM_THREADS];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index++)
	{
		thread_index = omp_get_thread_num();
		
		aij[thread_index] = a->element[index];

		tmp[thread_index] = sc * aij[thread_index];

		c->element[index] = tmp[thread_index];
	}
#endif
}

/* c := a */
void _bncomp_subst_dmatrix(DMatrix c, DMatrix a)
{
	long int i, j, index;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_dmatrix\n");
		return;
	}

#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl=(long)svcntd(); long _N=c->real_row_dim * c->real_col_dim; long _ix;
		for(_ix=0;_ix<_N;_ix+=_vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svst1_f64(_pg, &(c->element[_ix]), svld1_f64(_pg, &(a->element[_ix])));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	long int real_row_dim, real_col_dim;

	real_row_dim = c->real_row_dim;
	real_col_dim = c->real_col_dim;

	#pragma omp parallel for private(j, index)
	for(i = 0; i < real_row_dim; i++)
	{
		for(j = 0; j < real_col_dim; j += 2)
		{
			index = i * real_col_dim + j;
			vst1q_f64(&(c->element[i * real_col_dim + j]), vld1q_f64(&(a->element[index])));
		}
	}
#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int real_row_dim, real_col_dim;

	real_row_dim = c->real_row_dim;
	real_col_dim = c->real_col_dim;

	#pragma omp parallel for private(j, index)
	for(i = 0; i < real_row_dim; i++)
	{
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			index = i * real_col_dim + j;
			_mm256_store_pd(&(c->element[i * real_col_dim + j]), _mm256_load_pd(&(a->element[index])));
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int real_row_dim, real_col_dim;

	real_row_dim = c->real_row_dim;
	real_col_dim = c->real_col_dim;

	#pragma omp parallel for private(j, index)
	for(i = 0; i < real_row_dim; i++)
	{
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			index = i * real_col_dim + j;
			_mm512_store_pd(&(c->element[i * real_col_dim + j]), _mm512_load_pd(&(a->element[index])));
		}
	}
#else // others
	#pragma omp parallel for private(j)
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j));
		}
	}
#endif
}

/* c = a * b */
//void _bncomp_mul_dmatrix(DMatrix ret, DMatrix a, DMatrix b)
//{
//	mul_dmatrix(ret, a, b);
//}
void _bncomp_mul_dmatrix(DMatrix ret, DMatrix a, DMatrix b)
{
	int thread_num, thread_index;
	long int i, j, k;
	long row_dim, col_dim, mid_dim;

	/* dimension check */
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_dmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret->row_dim, ret->col_dim, a->row_dim, a->col_dim, b->row_dim, b->col_dim);
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long real_row_dim, real_col_dim, real_mid_dim;
	double cijval[BNCOMP_MAX_NUM_THREADS][4];
    __m256d cij[BNCOMP_MAX_NUM_THREADS], aik[BNCOMP_MAX_NUM_THREADS], bkj[BNCOMP_MAX_NUM_THREADS], tmp_mul[BNCOMP_MAX_NUM_THREADS];

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		//set0_d(tmp[thread_index]);
	}

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, i, j, k, aik, bkj, cij, cijval, tmp_mul)
    for(i = 0; i < real_row_dim; i++)
    {
		thread_index = omp_get_thread_num();

        for(j = 0; j < real_col_dim; j++)
        {
            //rdd_set_ui(cij.val, 0UL);
            cij[thread_index] = _mm256_setzero_pd();
            for(k = 0; k < real_mid_dim; k += _BNC_D_WIDTH)
            {
            /*
                aik[0].val[0] = a->element[0][i * mid_dim + k];
                aik[1].val[0] = a->element[0][i * mid_dim + k + 1];
                aik[2].val[0] = a->element[0][i * mid_dim + k + 2];
                aik[3].val[0] = a->element[0][i * mid_dim + k + 3];
            */
                aik[thread_index] = _mm256_load_pd(&(a->element[i * real_mid_dim + k]));
                
            /*    aik[thread_index][0] = _mm256_set_pd(
                    a->element[0][i * real_mid_dim + k],
                    a->element[0][i * real_mid_dim + k + 1],
                    a->element[0][i * real_mid_dim + k + 2],
                    a->element[0][i * real_mid_dim + k + 3]
                );
            */
                
            /*    aik[thread_index][1] = _mm256_set_pd(
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
                bkj[thread_index] = _mm256_set_pd(
                //    b->element[0][ k      * real_col_dim + j],
                //    b->element[0][(k + 1) * real_col_dim + j],
                //    b->element[0][(k + 2) * real_col_dim + j],
                //    b->element[0][(k + 3) * real_col_dim + j]
                    b->element[(k + 3) * real_col_dim + j],
                    b->element[(k + 2) * real_col_dim + j],
                    b->element[(k + 1) * real_col_dim + j],
                    b->element[(k    ) * real_col_dim + j]
                );
            /*
                bkj[0].val[1] = b->element[1][k * col_dim + j];
                bkj[1].val[1] = b->element[1][(k + 1) * col_dim + j];
                bkj[2].val[1] = b->element[1][(k + 2) * col_dim + j];
                bkj[3].val[1] = b->element[1][(k + 3) * col_dim + j];
            */

            /*
                rdd_mul(tmp_mul[0].val, aik[0].val, bkj[0].val);
                rdd_mul(tmp_mul[1].val, aik[1].val, bkj[1].val);
                rdd_mul(tmp_mul[2].val, aik[2].val, bkj[2].val);
                rdd_mul(tmp_mul[3].val, aik[3].val, bkj[3].val);
            */
               // _bncavx2_rdd_mul(tmp_mul[thread_index], aik[thread_index], bkj[thread_index]);
               tmp_mul[thread_index] = _mm256_mul_pd(aik[thread_index], bkj[thread_index]);

            /*
                rdd_add(cij.val, cij.val, tmp_mul[0].val);
                rdd_add(cij.val, cij.val, tmp_mul[1].val);
                rdd_add(cij.val, cij.val, tmp_mul[2].val);
                rdd_add(cij.val, cij.val, tmp_mul[3].val);
            */
				#pragma omp critical // needed for Intel Compiler
				cij[thread_index] = _mm256_add_pd(cij[thread_index], tmp_mul[thread_index]);
            }

            /*
			[thread_index][0] = cij[thread_index][0];
            cijval[thread_index][1] = cij[thread_index][0];
            cijval[thread_index][2] = cij[thread_index][0];
            cijval[thread_index][3] = cij[thread_index][0];
            cijval[thread_index][0] = cijval[thread_index][0] + cijval[thread_index][1];
            cijval[thread_index][0] = cijval[thread_index][0] + cijval[thread_index][2];
            cijval[thread_index][0] = cijval[thread_index][0] + cijval[thread_index][3];
			*/

            ret->element[i * real_col_dim + j] = _bncavx2_dsum256d(cij[thread_index]); // cijval[thread_index][0];
        }
    }
#elif defined(__AVX512F__) // __AVX512F__
	long real_row_dim, real_col_dim, real_mid_dim;
	double cijval[BNCOMP_MAX_NUM_THREADS][8];
    __m512d cij[BNCOMP_MAX_NUM_THREADS], aik[BNCOMP_MAX_NUM_THREADS], bkj[BNCOMP_MAX_NUM_THREADS], tmp_mul[BNCOMP_MAX_NUM_THREADS];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, i, j, k, aik, bkj, cij, cijval, tmp_mul)
    for(i = 0; i < real_row_dim; i++)
    {
		thread_index = omp_get_thread_num();

        for(j = 0; j < real_col_dim; j++)
        {
            //rdd_set_ui(cij.val, 0UL);
            cij[thread_index] = _mm512_setzero_pd();

            for(k = 0; k < real_mid_dim; k += _BNC_D_WIDTH)
            {
                aik[thread_index] = _mm512_load_pd(&(a->element[i * real_mid_dim + k]));

                bkj[thread_index] = _mm512_set_pd(
                    b->element[(k + 7) * real_col_dim + j],
                    b->element[(k + 6) * real_col_dim + j],
                    b->element[(k + 5) * real_col_dim + j],
                    b->element[(k + 4) * real_col_dim + j],
                    b->element[(k + 3) * real_col_dim + j],
                    b->element[(k + 2) * real_col_dim + j],
                    b->element[(k + 1) * real_col_dim + j],
                    b->element[(k    ) * real_col_dim + j]
                );

                //_bncavx512_rdd_mul(tmp_mul[thread_index], aik[thread_index], bkj[thread_index]);
				tmp_mul[thread_index] = _mm512_mul_pd(aik[thread_index], bkj[thread_index]);
				#pragma omp critical // needed for Intel Compiler
                	cij[thread_index] = _mm512_add_pd(cij[thread_index], tmp_mul[thread_index]);
            }

            /*
			cijval[thread_index][0] = cij[thread_index][0];
            cijval[thread_index][1] = cij[thread_index][1];
			cijval[thread_index][2] = cij[thread_index][2];
            cijval[thread_index][3] = cij[thread_index][3];
            cijval[thread_index][4] = cij[thread_index][4];
            cijval[thread_index][5] = cij[thread_index][5];
            cijval[thread_index][6] = cij[thread_index][6];
            cijval[thread_index][7] = cij[thread_index][7];

			cijval[thread_index] += , cijval[thread_index][1]);
            cijval[thread_index] += , cijval[thread_index][2]);
            cijval[thread_index] += , cijval[thread_index][3]);
            cijval[thread_index] += , cijval[thread_index][4]);
            cijval[thread_index] += , cijval[thread_index][5]);
            cijval[thread_index] += , cijval[thread_index][6]);
            cijval[thread_index] += , cijval[thread_index][7]);
			*/
            ret->element[i * real_col_dim + j] = _bncavx2_dsum512d(cij[thread_index]); //cijval[thread_index][0][0];
        }
    }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP)
	{
		long _vl = (long)svcntd();
		long _rr = ret->real_row_dim, _rc = ret->real_col_dim, _rm = a->real_col_dim;
		#pragma omp parallel for private(j, k)
		for(i = 0; i < _rr; i++)
		{
			for(j = 0; j < _rc; j += _vl)
			{
				svbool_t _pg = svwhilelt_b64_s64((int64_t)j, (int64_t)_rc);
				svfloat64_t _cij = svdup_n_f64(0.0);
				for(k = 0; k < _rm; k++)
				{
					svfloat64_t _aik = svdup_n_f64(a->element[i * _rm + k]);
					svfloat64_t _bkj = svld1_f64(_pg, &(b->element[k * _rc + j]));
					_cij = svmla_f64_x(_pg, _cij, _aik, _bkj);
				}
				svst1_f64(_pg, &(ret->element[i * _rc + j]), _cij);
			}
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	long real_row_dim, real_col_dim, real_mid_dim;
	double cijval[BNCOMP_MAX_NUM_THREADS], bkj_tmp0[2];
    float64x2_t cij[BNCOMP_MAX_NUM_THREADS], aik[BNCOMP_MAX_NUM_THREADS], bkj[BNCOMP_MAX_NUM_THREADS], tmp_mul[BNCOMP_MAX_NUM_THREADS];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, i, j, k, aik, bkj, cij, cijval, tmp_mul, bkj_tmp0)
    for(i = 0; i < real_row_dim; i++)
    {
		thread_index = omp_get_thread_num();

        for(j = 0; j < real_col_dim; j++)
        {
            cij[thread_index] = vdupq_n_f64(0.0);
            for(k = 0; k < real_mid_dim; k += _BNC_D_WIDTH)
            {
                aik[thread_index] = vld1q_f64(&(a->element[i * real_mid_dim + k]));

                // Neonは2要素なので個別にセット
                bkj_tmp0[0] = b->element[(k    ) * real_col_dim + j];
                bkj_tmp0[1] = b->element[(k + 1) * real_col_dim + j];
                bkj[thread_index] = vld1q_f64(bkj_tmp0);

                tmp_mul[thread_index] = vmulq_f64(aik[thread_index], bkj[thread_index]);
				#pragma omp critical
                	cij[thread_index] = vaddq_f64(cij[thread_index], tmp_mul[thread_index]);
            }

            //cijval[thread_index] = vgetq_lane_f64(cij[thread_index], 0) + vgetq_lane_f64(cij[thread_index], 1);
			cijval[thread_index] = vaddvq_f64(cij[thread_index]);

            ret->element[i * real_col_dim + j] = cijval[thread_index];
        }
    }
#else // __AVX2__
	double tmp[BNCOMP_MAX_NUM_THREADS], ret_ij[BNCOMP_MAX_NUM_THREADS];

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		tmp[thread_index] = 0.0;
	}

	//printf("Non SIMD mul_dmatrix(%ld, %ld)\n", ret->row_dim, ret->col_dim);
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = a->col_dim;

	// Fix!: 2022-09-29(Thu) T.Kouya
	#pragma omp parallel for private(thread_index, i, j, k, ret_ij, tmp)
	for(i = 0; i < row_dim; i++)
	{

		thread_index = omp_get_thread_num();

		//#pragma omp parallel for private(thread_index, k)
		for(j = 0; j < col_dim; j++)
		{
			//thread_index = omp_get_thread_num();
			//#pragma omp critical
			ret_ij[thread_index] = 0.0;
			for(k = 0; k < mid_dim; k++)
			{
				//rdd_mul(tmp[thread_index], GET_DMATRIX_IJ(a, i, k), GET_DMATRIX_IJ(b, k, j));
				tmp[thread_index] = get_dmatrix_ij(a, i, k) * get_dmatrix_ij(b, k, j);
				#pragma omp critical // needed for Intel Compiler
				ret_ij[thread_index] = tmp[thread_index] + ret_ij[thread_index];
			}
			
			set_dmatrix_ij(ret, i, j, ret_ij[thread_index]);
		}
	}	//printf("end(%ld, %ld)\n", ret->row_dim, ret->col_dim);
#endif // __AVX2__
}

/* c := I */
void _bncomp_setI_dmatrix(DMatrix c)
{
	long int i, real_total_dim;
	double tmp0, tmp1;

	tmp0 = 0.0;
	tmp1 = 1.0;

	real_total_dim = c->real_row_dim * c->real_col_dim;

#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl=(long)svcntd(); long _N=c->real_row_dim * c->real_col_dim; long _ix;
		for(_ix=0;_ix<_N;_ix+=_vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svst1_f64(_pg, &(c->element[_ix]), svdup_n_f64(0.0));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	float64x2_t zero2;

	zero2 = vdupq_n_f64(0.0);
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += 2)
	{
		vst1q_f64(&c->element[i], zero2);
	}
#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&c->element[i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&c->element[i], zero8);
	}
#else // others
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i++)
	{
		c->element[i] = 0.0;
	}
#endif

	tmp1 = 1.0;

	#pragma omp parallel for
	for(i = 0; i < c->row_dim; i++)
	{
		if(i < c->col_dim)
			set_dmatrix_ij(c, i, i, tmp1);
	}
}

void _bncomp_set0_dmatrix(DMatrix mat)
{
	long int i, real_total_dim;

	real_total_dim = mat->real_row_dim * mat->real_col_dim;

#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl=(long)svcntd(); long _N=mat->real_row_dim * mat->real_col_dim; long _ix;
		for(_ix=0;_ix<_N;_ix+=_vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svst1_f64(_pg, &(mat->element[_ix]), svdup_n_f64(0.0));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	float64x2_t zero2;

	zero2 = vdupq_n_f64(0.0);
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += 2)
	{
		vst1q_f64(&mat->element[i], zero2);
	}
#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&mat->element[i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&mat->element[i], zero8);
	}
#else // others
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i++)
	{
		mat->element[i] = 0.0;
	}
#endif
}

/* v := a * vb */
void _bncomp_mul_dmatrix_dvec(DVector v, DMatrix a, DVector vb)
{
	long int i, j, row_dim;
	double tmp[BNCOMP_MAX_NUM_THREADS], tmp1[BNCOMP_MAX_NUM_THREADS];
	int thread_index;

	if((v->dim < a->row_dim) || (vb->dim < a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_dmatrix_dvec\n");
		return;
	}

	row_dim = a->row_dim;

#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP)
	{
		long _vl = (long)svcntd();
		long _rc = a->real_col_dim;
		#pragma omp parallel for private(j)
		for(i = 0; i < row_dim; i++)
		{
			svfloat64_t _acc = svdup_n_f64(0.0);
			for(j = 0; j < _rc; j += _vl)
			{
				svbool_t _pg = svwhilelt_b64_s64((int64_t)j, (int64_t)_rc);
				svfloat64_t _aij = svld1_f64(_pg, &(a->element[i * _rc + j]));
				svfloat64_t _vbj = svld1_f64(_pg, &(vb->element[j]));
				_acc = svmla_f64_x(_pg, _acc, _aij, _vbj);
			}
			v->element[i] = svaddv_f64(svptrue_b64(), _acc);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	long int ij_index, real_row_dim, real_col_dim;
	float64x2_t tmp2[BNCOMP_MAX_NUM_THREADS], tmp1_2[BNCOMP_MAX_NUM_THREADS];
	float64x2_t aij2[BNCOMP_MAX_NUM_THREADS], vbj2[BNCOMP_MAX_NUM_THREADS];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j, ij_index)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		tmp2[thread_index] = vdupq_n_f64(0.0);
		for(j = 0; j < real_col_dim; j += 2)
		{
			ij_index = i * real_col_dim + j;
			aij2[thread_index] = vld1q_f64(&(a->element[ij_index]));
			vbj2[thread_index] = vld1q_f64(&(vb->element[j]));

			tmp2[thread_index] = vfmaq_f64(tmp2[thread_index], aij2[thread_index], vbj2[thread_index]);
		}
		v->element[i] = _bncneon_dsum128(tmp2[thread_index]);
	}
#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int ij_index, real_col_dim;
	__m256d tmp4[BNCOMP_MAX_NUM_THREADS], tmp1_4[BNCOMP_MAX_NUM_THREADS];
	__m256d aij4[BNCOMP_MAX_NUM_THREADS], vbj4[BNCOMP_MAX_NUM_THREADS];

	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j, ij_index)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		tmp4[thread_index] = _mm256_setzero_pd();
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij4[thread_index] = _mm256_load_pd(&(a->element[ij_index]));
			vbj4[thread_index] = _mm256_load_pd(&(vb->element[j]));

			tmp4[thread_index] = _mm256_fmadd_pd(aij4[thread_index], vbj4[thread_index], tmp4[thread_index]);
		}
		v->element[i] = _bncavx2_dsum256d(tmp4[thread_index]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int ij_index, real_col_dim;
	__m512d tmp8[BNCOMP_MAX_NUM_THREADS], tmp1_8[BNCOMP_MAX_NUM_THREADS];
	__m512d aij8[BNCOMP_MAX_NUM_THREADS], vbj8[BNCOMP_MAX_NUM_THREADS];

	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j, ij_index)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		tmp8[thread_index] = _mm512_setzero_pd();
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			thread_index = omp_get_thread_num();

			ij_index = i * real_col_dim + j;
			aij8[thread_index] = _mm512_load_pd(&(a->element[ij_index]));
			vbj8[thread_index] = _mm512_load_pd(&(vb->element[j]));

			tmp8[thread_index] = _mm512_fmadd_pd(aij8[thread_index], vbj8[thread_index], tmp8[thread_index]);
		}
		v->element[i] = _bncavx2_dsum512d(tmp8[thread_index]);
	}
#else // others
	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		tmp[thread_index] = 0.0;
		for(j = 0; j < a->col_dim; j++)
		{
            tmp[thread_index] += get_dmatrix_ij(a, i, j) * get_dvector_i(vb, j);
		}
		set_dvector_i(v, i, tmp[thread_index]);
	}
#endif
}

/* v := a^T * vb */
void _bncomp_mul_dmatrixt_dvec(DVector v, DMatrix a, DVector vb)
{
	long int i, j, col_dim;
	double tmp[BNCOMP_MAX_NUM_THREADS], tmp1[BNCOMP_MAX_NUM_THREADS];
	int thread_index;

	if((v->dim < a->col_dim) || (vb->dim < a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_dmatrixt_dvec\n");
		return;
	}

	col_dim = a->col_dim;

#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP)
	{
		long _vl = (long)svcntd();
		long _rr = a->real_row_dim, _rc = a->real_col_dim;
		#pragma omp parallel for private(j)
		for(i = 0; i < col_dim; i++)
		{
			svfloat64_t _acc = svdup_n_f64(0.0);
			for(j = 0; j < _rr; j += _vl)
			{
				svbool_t _pg = svwhilelt_b64_s64((int64_t)j, (int64_t)_rr);
				svint64_t _idx = svindex_s64((int64_t)(j * _rc + i), (int64_t)_rc);
				svfloat64_t _aji = svld1_gather_s64index_f64(_pg, a->element, _idx);
				svfloat64_t _vbj = svld1_f64(_pg, &(vb->element[j]));
				_acc = svmla_f64_x(_pg, _acc, _aji, _vbj);
			}
			v->element[i] = svaddv_f64(svptrue_b64(), _acc);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	long int real_row_dim, real_col_dim;
	float64x2_t tmp2[BNCOMP_MAX_NUM_THREADS], tmp1_2[BNCOMP_MAX_NUM_THREADS];
	float64x2_t aij2[BNCOMP_MAX_NUM_THREADS], vbj2[BNCOMP_MAX_NUM_THREADS];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < col_dim; i++)
	{
		thread_index = omp_get_thread_num();

		tmp2[thread_index] = vdupq_n_f64(0.0);

		for(j = 0; j < real_row_dim; j += 2)
		{
			// Load transposed elements
			double temp[2];
			temp[0] = a->element[(j    ) * real_col_dim + i];
			temp[1] = a->element[(j + 1) * real_col_dim + i];
			aij2[thread_index] = vld1q_f64(temp);
			
			vbj2[thread_index] = vld1q_f64(&(vb->element[j]));

			tmp2[thread_index] = vfmaq_f64(tmp2[thread_index], aij2[thread_index], vbj2[thread_index]);
		}
		v->element[i] = _bncneon_dsum128(tmp2[thread_index]);
	}
#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int real_row_dim, real_col_dim;
	__m256d tmp4[BNCOMP_MAX_NUM_THREADS], tmp1_4[BNCOMP_MAX_NUM_THREADS];
	__m256d aij4[BNCOMP_MAX_NUM_THREADS], vbj4[BNCOMP_MAX_NUM_THREADS];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < col_dim; i++)
	{
		thread_index = omp_get_thread_num();

		tmp4[thread_index] = _mm256_setzero_pd();

		for(j = 0; j < real_row_dim; j += _BNC_D_WIDTH)
		{
			aij4[thread_index] = _mm256_set_pd(
				a->element[(j + 3) * real_col_dim + i],
				a->element[(j + 2) * real_col_dim + i],
				a->element[(j + 1) * real_col_dim + i],
				a->element[(j    ) * real_col_dim + i]
			);
			vbj4[thread_index] = _mm256_load_pd(&(vb->element[j]));

			tmp4[thread_index] = _mm256_fmadd_pd(aij4[thread_index], vbj4[thread_index], tmp4[thread_index]);
		}
		v->element[i] = _bncavx2_dsum256d(tmp4[thread_index]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int real_row_dim, real_col_dim;
	__m512d tmp8[BNCOMP_MAX_NUM_THREADS], tmp1_8[BNCOMP_MAX_NUM_THREADS];
	__m512d aij8[BNCOMP_MAX_NUM_THREADS], vbj8[BNCOMP_MAX_NUM_THREADS];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < col_dim; i++)
	{
		thread_index = omp_get_thread_num();

		tmp8[thread_index] = _mm512_setzero_pd();
		for(j = 0; j < real_row_dim; j += _BNC_D_WIDTH)
		{
			thread_index = omp_get_thread_num();

			aij8[thread_index] = _mm512_set_pd(
				a->element[(j + 7) * real_col_dim + i],
				a->element[(j + 6) * real_col_dim + i],
				a->element[(j + 5) * real_col_dim + i],
				a->element[(j + 4) * real_col_dim + i],
				a->element[(j + 3) * real_col_dim + i],
				a->element[(j + 2) * real_col_dim + i],
				a->element[(j + 1) * real_col_dim + i],
				a->element[(j    ) * real_col_dim + i]
			);
			vbj8[thread_index] = _mm512_load_pd(&(vb->element[j]));

			tmp8[thread_index] = _mm512_fmadd_pd(aij8[thread_index], vbj8[thread_index], tmp8[thread_index]);
		}
		v->element[i] = _bncavx2_dsum512d(tmp8[thread_index]);

    }
#else // others
	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < a->col_dim; i++)
	{
		thread_index = omp_get_thread_num();

        tmp[thread_index] = 0.0;
		for(j = 0; j < a->row_dim; j++)
		{
            tmp[thread_index] += get_dmatrix_ij(a, j, i) * get_dvector_i(vb, j);
		}
		set_dvector_i(v, i, tmp[thread_index]);
	}
#endif
}

// Matrix-vector multiplication: 4M
void _bncomp_mul_cdmatrix_cdvec(CDVector ret, CDMatrix mat, CDVector vec)
{
	DVector t1, t2, t3, t4, vec_re, vec_im, ret_re, ret_im;
    DMatrix mat_re, mat_im;

	mat_re = init_dmatrix(mat->row_dim, mat->col_dim);
	mat_im = init_dmatrix(mat->row_dim, mat->col_dim);
	separate_cdmatrix(mat_re, mat_im, mat);
	vec_re = init_dvector(vec->dim);
	vec_im = init_dvector(vec->dim);
	separate_cdvector(vec_re, vec_im, vec);

    t1 = init_dvector(vec->dim);
    t2 = init_dvector(vec->dim);
    t3 = init_dvector(vec->dim);
    t4 = init_dvector(vec->dim);
	ret_re = init_dvector(vec->dim);
    ret_im = init_dvector(vec->dim);

    _bncomp_mul_dmatrix_dvec(t1, mat_re, vec_re);
    _bncomp_mul_dmatrix_dvec(t2, mat_im, vec_im);
    _bncomp_sub_dvector(ret_re, t1, t2);

    //#ifdef USE_4M
        // 4M
        _bncomp_mul_dmatrix_dvec(t3, mat_im, vec_re);
        _bncomp_mul_dmatrix_dvec(t4, mat_re, vec_im);
        _bncomp_add_dvector(ret_im, t3, t4);
    //#else // USE_4M
    /*
        // 3M
        tmp_mat = init_dmatrix(mat->row_dim, mat->col_dim);
        add_dmatrix(tmp_mat, a->re, a->im);
        add_dvector(t3, vb->re, vb->im);
        mul_dmatrix_dvec(t4, tmp_mat, t3);
        sub_dvector(v->im, t4, t1);
        sub_dvector(v->im, v->im, t2);
        free_dmatrix(tmp_mat);
    */
    //#endif // USE_4M

	merge_cdvector(ret, ret_re, ret_im);

	free_dvector(ret_re);
	free_dvector(ret_im);
	free_dvector(vec_re);
	free_dvector(vec_im);
	free_dmatrix(mat_re);
	free_dmatrix(mat_im);
    free_dvector(t1);
    free_dvector(t2);
    free_dvector(t3);
    free_dvector(t4);
}

// Transposed Matrix-vector multiplication: 4M
void _bncomp_mul_cdmatrixt_cdvec(CDVector ret, CDMatrix mat, CDVector vec)
{
	DVector t1, t2, t3, t4, vec_re, vec_im, ret_re, ret_im;
    DMatrix mat_re, mat_im;

	mat_re = init_dmatrix(mat->row_dim, mat->col_dim);
	mat_im = init_dmatrix(mat->row_dim, mat->col_dim);
	separate_cdmatrix(mat_re, mat_im, mat);
	vec_re = init_dvector(vec->dim);
	vec_im = init_dvector(vec->dim);
	separate_cdvector(vec_re, vec_im, vec);

    t1 = init_dvector(vec->dim);
    t2 = init_dvector(vec->dim);
    t3 = init_dvector(vec->dim);
    t4 = init_dvector(vec->dim);
	ret_re = init_dvector(vec->dim);
    ret_im = init_dvector(vec->dim);

    _bncomp_mul_dmatrixt_dvec(t1, mat_re, vec_re);
    _bncomp_mul_dmatrixt_dvec(t2, mat_im, vec_im);
    _bncomp_sub_dvector(ret_re, t1, t2);

    //#ifdef USE_4M
        // 4M
        _bncomp_mul_dmatrixt_dvec(t3, mat_im, vec_re);
        _bncomp_mul_dmatrixt_dvec(t4, mat_re, vec_im);
        _bncomp_add_dvector(ret_im, t3, t4);
    //#else // USE_4M
    /*
        // 3M
        tmp_mat = init_ddmatrix(a->re->row_dim, a->re->col_dim);
        add_ddmatrix(tmp_mat, a->re, a->im);
        add_ddvector(t3, vb->re, vb->im);
        mul_ddmatrix_ddvec(t4, tmp_mat, t3);
        sub_ddvector(v->im, t4, t1);
        sub_ddvector(v->im, v->im, t2);
        free_ddmatrix(tmp_mat);
    */
    //#endif // USE_4M

	merge_cdvector(ret, ret_re, ret_im);

	free_dvector(ret_re);
	free_dvector(ret_im);
	free_dvector(vec_re);
	free_dvector(vec_im);
	free_dmatrix(mat_re);
	free_dmatrix(mat_im);
    free_dvector(t1);
    free_dvector(t2);
    free_dvector(t3);
    free_dvector(t4);
}

// Transposed conjugate Matrix-vector multiplication: 4M
void _bncomp_mul_cdmatrixs_cdvec(CDVector ret, CDMatrix mat, CDVector vec)
{
	DVector t1, t2, t3, t4, vec_re, vec_im, ret_re, ret_im;
    DMatrix mat_re, mat_im;

	mat_re = init_dmatrix(mat->row_dim, mat->col_dim);
	mat_im = init_dmatrix(mat->row_dim, mat->col_dim);
	separate_cdmatrix(mat_re, mat_im, mat);
	vec_re = init_dvector(vec->dim);
	vec_im = init_dvector(vec->dim);
	separate_cdvector(vec_re, vec_im, vec);

    t1 = init_dvector(vec->dim);
    t2 = init_dvector(vec->dim);
    t3 = init_dvector(vec->dim);
    t4 = init_dvector(vec->dim);
	ret_re = init_dvector(vec->dim);
    ret_im = init_dvector(vec->dim);

    _bncomp_mul_dmatrixt_dvec(t1, mat_re, vec_re);
    _bncomp_mul_dmatrixt_dvec(t2, mat_im, vec_im);
    _bncomp_add_dvector(ret_re, t1, t2);

    //#ifdef USE_4M
        // 4M
        _bncomp_mul_dmatrixt_dvec(t3, mat_im, vec_re);
        _bncomp_mul_dmatrixt_dvec(t4, mat_re, vec_im);
        _bncomp_sub_dvector(ret_im, t3, t4);
    //#else // USE_4M
    /*
        // 3M
        tmp_mat = init_ddmatrix(a->re->row_dim, a->re->col_dim);
        add_ddmatrix(tmp_mat, a->re, a->im);
        add_ddvector(t3, vb->re, vb->im);
        mul_ddmatrix_ddvec(t4, tmp_mat, t3);
        sub_ddvector(v->im, t4, t1);
        sub_ddvector(v->im, v->im, t2);
        free_ddmatrix(tmp_mat);
    */
    //#endif // USE_4M

	merge_cdvector(ret, ret_re, ret_im);

	free_dvector(ret_re);
	free_dvector(ret_im);
	free_dvector(vec_re);
	free_dvector(vec_im);
	free_dmatrix(mat_re);
	free_dmatrix(mat_im);
    free_dvector(t1);
    free_dvector(t2);
    free_dvector(t3);
    free_dvector(t4);
}

// Matrix multiplication based on Ozaki scheme
/*------------------------------------------------------------------------------*/
/* Matrix multiplication based on Ozaki scheme                                   */
/*                                                                               */
/* mul_dmatrix_oz() is OpenMP-parallel itself: it cuts the rows of ret into      */
/* blocks and gives one block at a time to a thread, which runs every slice      */
/* product for it with a single-threaded BLAS call and accumulates on the spot.  */
/* That parallelizes the splitting and the accumulation as well, whereas the     */
/* code that used to stand here parallelized only the outer slice loop and       */
/* pushed every accumulation through an omp critical -- which serialized the     */
/* expensive half and made the sum order depend on the thread interleaving.      */
/*                                                                               */
/* It also honours max_num_div_a / max_num_div_b, as every other precision does; */
/* the old body ignored both and always split the operands in two.               */
/*------------------------------------------------------------------------------*/
void _bncomp_mul_dmatrix_oz(DMatrix ret, DMatrix a, int max_num_div_a, DMatrix b, int max_num_div_b)
{
    mul_dmatrix_oz(ret, a, max_num_div_a, b, max_num_div_b);
}

// Fit dimension to be multiple of min_dim
void _bncomp_mul_cdmatrix_oz_3m(CDMatrix ret, CDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CDMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    DMatrix t1, t2, t3, t4;
	int max_num_div_apa, max_num_div_bpb;
	DMatrix a_re, a_im, b_re, b_im, ret_re, ret_im;

	a_re = init_dmatrix(a->row_dim, a->col_dim);
	a_im = init_dmatrix(a->row_dim, a->col_dim);
	b_re = init_dmatrix(b->row_dim, b->col_dim);
	b_im = init_dmatrix(b->row_dim, b->col_dim);
	separate_cdmatrix(a_re, a_im, a);
	separate_cdmatrix(b_re, b_im, b);
	ret_re = init_dmatrix(ret->row_dim, ret->col_dim);
	ret_im = init_dmatrix(ret->row_dim, ret->col_dim);

    t1 = init_dmatrix(ret->row_dim, ret->col_dim);
    t2 = init_dmatrix(ret->row_dim, ret->col_dim);
    t3 = init_dmatrix(ret->row_dim, ret->col_dim);
    t4 = init_dmatrix(ret->row_dim, ret->col_dim);

    _bncomp_mul_dmatrix_oz(t1, a_re, max_num_div_a_real, b_re, max_num_div_b_real);
    _bncomp_mul_dmatrix_oz(t2, a_im, max_num_div_a_image, b_im, max_num_div_a_image);
    _bncomp_sub_dmatrix(ret_re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        mul_dmatrix_oz(t3, a_im, max_num_div_a_image, b_re, max_num_div_a_real);
        mul_dmatrix_oz(t4, a_re, max_num_div_a_real, b_im, max_num_div_a_image);
        add_dmatrix(ret_im, t3, t4);
    #else // USE_4M
    */
        // 3M
        _bncomp_add_dmatrix(t3, a_re, a_im);
        _bncomp_add_dmatrix(t4, b_re, b_im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        _bncomp_mul_dmatrix_oz(ret_im, t3, max_num_div_apa, t4, max_num_div_bpb);
        _bncomp_sub_dmatrix(ret_im, ret_im, t1);
        _bncomp_sub_dmatrix(ret_im, ret_im, t2);
    //#endif // USE_4M

	merge_cdmatrix(ret, ret_re, ret_im);

	free_dmatrix(a_re);
	free_dmatrix(a_im);
	free_dmatrix(b_re);
	free_dmatrix(b_im);
	free_dmatrix(ret_re);
	free_dmatrix(ret_im);

    free_dmatrix(t1);
    free_dmatrix(t2);
    free_dmatrix(t3);
    free_dmatrix(t4);
}

// Fit dimension to be multiple of min_dim
void _bncomp_mul_cdmatrix_oz_4m(CDMatrix ret, CDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CDMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    DMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;
	DMatrix a_re, a_im, b_re, b_im, ret_re, ret_im;

	a_re = init_dmatrix(a->row_dim, a->col_dim);
	a_im = init_dmatrix(a->row_dim, a->col_dim);
	b_re = init_dmatrix(b->row_dim, b->col_dim);
	b_im = init_dmatrix(b->row_dim, b->col_dim);
	separate_cdmatrix(a_re, a_im, a);
	separate_cdmatrix(b_re, b_im, b);
	ret_re = init_dmatrix(ret->row_dim, ret->col_dim);
	ret_im = init_dmatrix(ret->row_dim, ret->col_dim);

    t1 = init_dmatrix(ret_re->row_dim, ret_re->col_dim);
    t2 = init_dmatrix(ret_re->row_dim, ret_re->col_dim);
    t3 = init_dmatrix(ret_re->row_dim, ret_re->col_dim);
    t4 = init_dmatrix(ret_re->row_dim, ret_re->col_dim);

    _bncomp_mul_dmatrix_oz(t1, a_re, max_num_div_a_real, b_re, max_num_div_b_real);
    _bncomp_mul_dmatrix_oz(t2, a_im, max_num_div_a_image, b_im, max_num_div_a_image);
    _bncomp_sub_dmatrix(ret_re, t1, t2);

    // 4M
    //#ifdef USE_4M
        _bncomp_mul_dmatrix_oz(t3, a_im, max_num_div_a_image, b_re, max_num_div_a_real);
        _bncomp_mul_dmatrix_oz(t4, a_re, max_num_div_a_real, b_im, max_num_div_a_image);
        _bncomp_add_dmatrix(ret_im, t3, t4);
    /*
    #else // USE_4M 
        // 3M
        add_dmatrix(t3, a_re, a_im);
        add_dmatrix(t4, b_re, b_im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        mul_dmatrix_oz(ret_im, t3, max_num_div_apa, t4, max_num_div_bpb);
        sub_dmatrix(ret_im, ret_im, t1);
        sub_dmatrix(ret_im, ret_im, t2);
    #endif // USE_4M
    */
	merge_cdmatrix(ret, ret_re, ret_im);

	free_dmatrix(a_re);
	free_dmatrix(a_im);
	free_dmatrix(b_re);
	free_dmatrix(b_im);
	free_dmatrix(ret_re);
	free_dmatrix(ret_im);

    free_dmatrix(t1);
    free_dmatrix(t2);
    free_dmatrix(t3);
    free_dmatrix(t4);
}

// Simple triple-loop-way matrix multiplication (3M)
void _bncomp_mul_cdmatrix_3m(CDMatrix ret, CDMatrix a, CDMatrix b)
{
    DMatrix t1, t2, t3, t4;
	DMatrix a_re, a_im, b_re, b_im, ret_re, ret_im;

	a_re = init_dmatrix(a->row_dim, a->col_dim);
	a_im = init_dmatrix(a->row_dim, a->col_dim);
	b_re = init_dmatrix(b->row_dim, b->col_dim);
	b_im = init_dmatrix(b->row_dim, b->col_dim);
	separate_cdmatrix(a_re, a_im, a);
	separate_cdmatrix(b_re, b_im, b);
	ret_re = init_dmatrix(ret->row_dim, ret->col_dim);
	ret_im = init_dmatrix(ret->row_dim, ret->col_dim);

 
    t1 = init_dmatrix(ret_re->row_dim, ret_re->col_dim);
    t2 = init_dmatrix(ret_re->row_dim, ret_re->col_dim);
    t3 = init_dmatrix(ret_re->row_dim, ret_re->col_dim);
    t4 = init_dmatrix(ret_re->row_dim, ret_re->col_dim);

    _bncomp_mul_dmatrix(t1, a_re, b_re);
    _bncomp_mul_dmatrix(t2, a_im, b_im);
    _bncomp_sub_dmatrix(ret_re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        _bncomp_mul_dmatrix(t3, a_im, b_re);
        _bncomp_mul_dmatrix(t4, a_re, b_im);
        _bncomp_add_dmatrix(ret_im, t3, t4);
    #else // USE_4M
    */
        // 3M
        _bncomp_add_dmatrix(t3, a_re, a_im);
        _bncomp_add_dmatrix(t4, b_re, b_im);
        _bncomp_mul_dmatrix(ret_im, t3, t4);
        _bncomp_sub_dmatrix(ret_im, ret_im, t1);
        _bncomp_sub_dmatrix(ret_im, ret_im, t2);
    //#endif // USE_4M
	merge_cdmatrix(ret, ret_re, ret_im);

	free_dmatrix(a_re);
	free_dmatrix(a_im);
	free_dmatrix(b_re);
	free_dmatrix(b_im);
	free_dmatrix(ret_re);
	free_dmatrix(ret_im);

    free_dmatrix(t1);
    free_dmatrix(t2);
    free_dmatrix(t3);
    free_dmatrix(t4);
}

// Simple triple-loop-way matrix multiplication (4M)
void _bncomp_mul_cdmatrix_4m(CDMatrix ret, CDMatrix a, CDMatrix b)
{
    DMatrix t1, t2, t3, t4;
 	DMatrix a_re, a_im, b_re, b_im, ret_re, ret_im;

	a_re = init_dmatrix(a->row_dim, a->col_dim);
	a_im = init_dmatrix(a->row_dim, a->col_dim);
	b_re = init_dmatrix(b->row_dim, b->col_dim);
	b_im = init_dmatrix(b->row_dim, b->col_dim);
	separate_cdmatrix(a_re, a_im, a);
	separate_cdmatrix(b_re, b_im, b);
	ret_re = init_dmatrix(ret->row_dim, ret->col_dim);
	ret_im = init_dmatrix(ret->row_dim, ret->col_dim);


    t1 = init_dmatrix(ret_re->row_dim, ret_re->col_dim);
    t2 = init_dmatrix(ret_re->row_dim, ret_re->col_dim);
    t3 = init_dmatrix(ret_re->row_dim, ret_re->col_dim);
    t4 = init_dmatrix(ret_re->row_dim, ret_re->col_dim);

    _bncomp_mul_dmatrix(t1, a_re, b_re);
    _bncomp_mul_dmatrix(t2, a_im, b_im);
    _bncomp_sub_dmatrix(ret_re, t1, t2);

    // 4M
    
    //#ifdef USE_4M
        _bncomp_mul_dmatrix(t3, a_im, b_re);
        _bncomp_mul_dmatrix(t4, a_re, b_im);
        _bncomp_add_dmatrix(ret_im, t3, t4);
    //#else // USE_4M
	/*
        // 3M
        _bncomp_add_dmatrix(t3, a_re, a_im);
        _bncomp_add_dmatrix(t4, b_re, b_im);
        _bncomp_mul_dmatrix(ret_im, t3, t4s);
        _bncomp_sub_dmatrix(ret_im, ret_im, t1);
        _bncomp_sub_dmatrix(ret_im, ret_im, t2);
	*/
    //#endif // USE_4M
	merge_cdmatrix(ret, ret_re, ret_im);

	free_dmatrix(a_re);
	free_dmatrix(a_im);
	free_dmatrix(b_re);
	free_dmatrix(b_im);
	free_dmatrix(ret_re);
	free_dmatrix(ret_im);

    free_dmatrix(t1);
    free_dmatrix(t2);
    free_dmatrix(t3);
    free_dmatrix(t4);
}

#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus