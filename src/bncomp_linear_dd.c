/********************************************************************************/
/* bncomp_linear_dd.c: Parallelized DD Precision Linear Computation Library     */
/*                                                                  with OpenMP */
/* Copyright (C) 2023 Tomonori Kouya                                            */
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
#include "matmul_strassen.h"

// bncomp_linear_dd.c with ARM NEON support
// Add this include at the top of the file
// #if defined(__ARM_NEON)
// #include "_bncneon_dd.h"
// #endif

//---------------------------------------
// Double-Double - Vector operations with NEON
//---------------------------------------

/* c := a */
void _bncomp_subst_ddvector(DDVector c, DDVector a)
{
	long int i;

#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = a->real_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svst1_f64(_pg, &(c->element[0][_ix]), svld1_f64(_pg, &(a->element[0][_ix])));
			svst1_f64(_pg, &(c->element[1][_ix]), svld1_f64(_pg, &(a->element[1][_ix])));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON)
	#pragma omp parallel for
	for(i = 0; i < a->real_dim; i += 2)
	{
		vst1q_f64(&(c->element[0][i]), vld1q_f64(&(a->element[0][i])));
		vst1q_f64(&(c->element[1][i]), vld1q_f64(&(a->element[1][i])));
	}
#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	#pragma omp parallel for
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&(c->element[0][i]), _mm256_load_pd(&(a->element[0][i])));
		_mm256_store_pd(&(c->element[1][i]), _mm256_load_pd(&(a->element[1][i])));
	}
#elif defined(__AVX512F__) // __AVX512F__
	#pragma omp parallel for
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&(c->element[0][i]), _mm512_load_pd(&(a->element[0][i])));
		_mm512_store_pd(&(c->element[1][i]), _mm512_load_pd(&(a->element[1][i])));
	}
#else // others
	#pragma omp parallel for
	for(i = 0; i < a->dim; i++)
		set_ddvector_i(c, i, get_ddvector_i(a, i));
#endif
}

/* c = a + b */
void _bncomp_add_ddvector(DDVector c, DDVector a, DDVector b)
{
	int thread_index;
	long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_ddvector\n");
		return;
	}

#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = c->real_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svfloat64_t _a0 = svld1_f64(_pg, &(a->element[0][_ix]));
			svfloat64_t _a1 = svld1_f64(_pg, &(a->element[1][_ix]));
			svfloat64_t _b0 = svld1_f64(_pg, &(b->element[0][_ix]));
			svfloat64_t _b1 = svld1_f64(_pg, &(b->element[1][_ix]));
			svfloat64_t _o0, _o1;
			_bncsve2_rdd_add(_pg, &_o0, &_o1, _a0, _a1, _b0, _b1);
			svst1_f64(_pg, &(c->element[0][_ix]), _o0);
			svst1_f64(_pg, &(c->element[1][_ix]), _o1);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON)
    float64x2_t in_ret[BNCOMP_MAX_NUM_THREADS][DDSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][DDSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][DDSIZE];

 	#pragma omp parallel for private(thread_index)
    for(index = 0; index < c->real_dim; index += 2)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = vld1q_f64(&(a->element[0][index]));
        in_a_val[thread_index][1] = vld1q_f64(&(a->element[1][index]));
        in_b_val[thread_index][0] = vld1q_f64(&(b->element[0][index]));
        in_b_val[thread_index][1] = vld1q_f64(&(b->element[1][index]));

        _bncneon_rdd_add(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        vst1q_f64(&(c->element[0][index]), in_ret[thread_index][0]);
        vst1q_f64(&(c->element[1][index]), in_ret[thread_index][1]);
    }
#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256d in_ret[BNCOMP_MAX_NUM_THREADS][DDSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][DDSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][DDSIZE];

 	#pragma omp parallel for private(thread_index)
   for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[thread_index][1] = _mm256_load_pd(&(a->element[1][index]));
        in_b_val[thread_index][0] = _mm256_load_pd(&(b->element[0][index]));
        in_b_val[thread_index][1] = _mm256_load_pd(&(b->element[1][index]));

        _bncavx2_rdd_add(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        _mm256_store_pd(&(c->element[0][index]), in_ret[thread_index][0]);
        _mm256_store_pd(&(c->element[1][index]), in_ret[thread_index][1]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512d in_ret[BNCOMP_MAX_NUM_THREADS][DDSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][DDSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	#pragma omp parallel for private(thread_index)
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = _mm512_load_pd(&(a->element[0][index]));
        in_a_val[thread_index][1] = _mm512_load_pd(&(a->element[1][index]));
        in_b_val[thread_index][0] = _mm512_load_pd(&(b->element[0][index]));
        in_b_val[thread_index][1] = _mm512_load_pd(&(b->element[1][index]));

        _bncavx512_rdd_add(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        _mm512_store_pd(&(c->element[0][index]), in_ret[thread_index][0]);
        _mm512_store_pd(&(c->element[1][index]), in_ret[thread_index][1]);
   }
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

		rdd_add(tmp[thread_index], get_ddvector_i(a, i), get_ddvector_i(b, i));
		set_ddvector_i(c, i, tmp[thread_index]);
	}
#endif
}

/* c = a - b */
void _bncomp_sub_ddvector(DDVector c, DDVector a, DDVector b)
{
	int thread_index;
	long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: _bncomp_sub_ddvector\n");
		return;
	}

#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = c->real_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svfloat64_t _a0 = svld1_f64(_pg, &(a->element[0][_ix]));
			svfloat64_t _a1 = svld1_f64(_pg, &(a->element[1][_ix]));
			svfloat64_t _b0 = svld1_f64(_pg, &(b->element[0][_ix]));
			svfloat64_t _b1 = svld1_f64(_pg, &(b->element[1][_ix]));
			_b0 = svneg_f64_x(_pg, _b0);
			_b1 = svneg_f64_x(_pg, _b1);
			svfloat64_t _o0, _o1;
			_bncsve2_rdd_add(_pg, &_o0, &_o1, _a0, _a1, _b0, _b1);
			svst1_f64(_pg, &(c->element[0][_ix]), _o0);
			svst1_f64(_pg, &(c->element[1][_ix]), _o1);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON)
    float64x2_t in_ret[BNCOMP_MAX_NUM_THREADS][DDSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][DDSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	#pragma omp parallel for private(thread_index)
    for(index = 0; index < c->real_dim; index += 2)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = vld1q_f64(&(a->element[0][index]));
        in_a_val[thread_index][1] = vld1q_f64(&(a->element[1][index]));
        in_b_val[thread_index][0] = vld1q_f64(&(b->element[0][index]));
        in_b_val[thread_index][1] = vld1q_f64(&(b->element[1][index]));

        _bncneon_rdd_sub(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        vst1q_f64(&(c->element[0][index]), in_ret[thread_index][0]);
        vst1q_f64(&(c->element[1][index]), in_ret[thread_index][1]);
   }
#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256d in_ret[BNCOMP_MAX_NUM_THREADS][DDSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][DDSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	#pragma omp parallel for private(thread_index)
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[thread_index][1] = _mm256_load_pd(&(a->element[1][index]));
        in_b_val[thread_index][0] = _mm256_load_pd(&(b->element[0][index]));
        in_b_val[thread_index][1] = _mm256_load_pd(&(b->element[1][index]));

        _bncavx2_rdd_sub(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        _mm256_store_pd(&(c->element[0][index]), in_ret[thread_index][0]);
        _mm256_store_pd(&(c->element[1][index]), in_ret[thread_index][1]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512d in_ret[BNCOMP_MAX_NUM_THREADS][DDSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][DDSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = _mm512_load_pd(&(a->element[0][index]));
        in_a_val[thread_index][1] = _mm512_load_pd(&(a->element[1][index]));
        in_b_val[thread_index][0] = _mm512_load_pd(&(b->element[0][index]));
        in_b_val[thread_index][1] = _mm512_load_pd(&(b->element[1][index]));

        _bncavx512_rdd_sub(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        _mm512_store_pd(&(c->element[0][index]), in_ret[thread_index][0]);
        _mm512_store_pd(&(c->element[1][index]), in_ret[thread_index][1]);
   }
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

		rdd_sub(tmp[thread_index], get_ddvector_i(a, i), get_ddvector_i(b, i));
		set_ddvector_i(c, i, tmp[thread_index]);
	}
#endif
}

/* c = val * a */
void _bncomp_cmul_ddvector(DDVector c, double val[DDSIZE], DDVector a)
{
	int thread_index;
	long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: _bncomp_cmul_ddvector\n");
		return;
	}

#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = a->real_dim; long _ix;
		svfloat64_t _v0 = svdup_n_f64(val[0]);
		svfloat64_t _v1 = svdup_n_f64(val[1]);
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svfloat64_t _a0 = svld1_f64(_pg, &(a->element[0][_ix]));
			svfloat64_t _a1 = svld1_f64(_pg, &(a->element[1][_ix]));
			svfloat64_t _o0, _o1;
			_bncsve2_rdd_mul(_pg, &_o0, &_o1, _v0, _v1, _a0, _a1);
			svst1_f64(_pg, &(c->element[0][_ix]), _o0);
			svst1_f64(_pg, &(c->element[1][_ix]), _o1);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float64x2_t a2[BNCOMP_MAX_NUM_THREADS][DDSIZE], c2[BNCOMP_MAX_NUM_THREADS][DDSIZE], val2[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		val2[thread_index][0] = vdupq_n_f64(val[0]);
		val2[thread_index][1] = vdupq_n_f64(val[1]);
	}
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += 2)
	{
		thread_index = omp_get_thread_num();

		a2[thread_index][0] = vld1q_f64(&(a->element[0][index]));
		a2[thread_index][1] = vld1q_f64(&(a->element[1][index]));

		_bncneon_rdd_mul(c2[thread_index], val2[thread_index], a2[thread_index]);

		vst1q_f64(&(c->element[0][index]), c2[thread_index][0]);
		vst1q_f64(&(c->element[1][index]), c2[thread_index][1]);
	}
#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[BNCOMP_MAX_NUM_THREADS][DDSIZE], c4[BNCOMP_MAX_NUM_THREADS][DDSIZE], val4[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		val4[thread_index][0] = _mm256_set1_pd(val[0]);
		val4[thread_index][1] = _mm256_set1_pd(val[1]);
	}
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a4[thread_index][0] = _mm256_load_pd(&(a->element[0][index]));
		a4[thread_index][1] = _mm256_load_pd(&(a->element[1][index]));

		_bncavx2_rdd_mul(c4[thread_index], val4[thread_index], a4[thread_index]);

		_mm256_store_pd(&(c->element[0][index]), c4[thread_index][0]);
		_mm256_store_pd(&(c->element[1][index]), c4[thread_index][1]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[BNCOMP_MAX_NUM_THREADS][DDSIZE], c8[BNCOMP_MAX_NUM_THREADS][DDSIZE], val8[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		val8[thread_index][0] = _mm512_set1_pd(val[0]);
		val8[thread_index][1] = _mm512_set1_pd(val[1]);
	}

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a8[thread_index][0] = _mm512_load_pd(&(a->element[0][index]));
		a8[thread_index][1] = _mm512_load_pd(&(a->element[1][index]));

		_bncavx512_rdd_mul(c8[thread_index], val8[thread_index], a8[thread_index]);

		_mm512_store_pd(&(c->element[0][index]), c8[thread_index][0]);
		_mm512_store_pd(&(c->element[1][index]), c8[thread_index][1]);
	}
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

		rdd_mul(tmp[thread_index], val, get_ddvector_i(a, i));
		set_ddvector_i(c, i, tmp[thread_index]);
	}
#endif
}

/* (a, b) - inner product */
void _bncomp_ip_ddvector(double ret[DDSIZE], DDVector a, DDVector b)
{
	int thread_index;
	long int i, index;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: _bncomp_ip_ddvector\n");
		return;
	}

#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = a->real_dim; long _ix;
		svfloat64_t _acc0 = svdup_n_f64(0.0);
		svfloat64_t _acc1 = svdup_n_f64(0.0);
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svfloat64_t _a0 = svld1_f64(_pg, &(a->element[0][_ix]));
			svfloat64_t _a1 = svld1_f64(_pg, &(a->element[1][_ix]));
			svfloat64_t _b0 = svld1_f64(_pg, &(b->element[0][_ix]));
			svfloat64_t _b1 = svld1_f64(_pg, &(b->element[1][_ix]));
			svfloat64_t _m0, _m1;
			_bncsve2_rdd_mul(_pg, &_m0, &_m1, _a0, _a1, _b0, _b1);
			_bncsve2_rdd_add(_pg, &_acc0, &_acc1, _acc0, _acc1, _m0, _m1);
		}
		{
			long _L, _vl = (long)svcntd();
			double _la0[64], _la1[64];
			svst1_f64(svptrue_b64(), _la0, _acc0);
			svst1_f64(svptrue_b64(), _la1, _acc1);
			rdd_set_ui(ret, 0UL);
			for(_L = 0; _L < _vl; _L++)
			{
				double _lane[DDSIZE] = { _la0[_L], _la1[_L] };
				rdd_add(ret, ret, _lane);
			}
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON)
	float64x2_t a2[BNCOMP_MAX_NUM_THREADS][DDSIZE], b2[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	float64x2_t ret2[DDSIZE], tmp2[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	_bncneon_rdd_set0(ret2);
	
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < a->real_dim; index += 2)
	{
		thread_index = omp_get_thread_num();

		a2[thread_index][0] = vld1q_f64(&(a->element[0][index]));
		a2[thread_index][1] = vld1q_f64(&(a->element[1][index]));
		b2[thread_index][0] = vld1q_f64(&(b->element[0][index]));
		b2[thread_index][1] = vld1q_f64(&(b->element[1][index]));

		_bncneon_rdd_mul(tmp2[thread_index], a2[thread_index], b2[thread_index]);
		
		#pragma omp critical
		_bncneon_rdd_add(ret2, ret2, tmp2[thread_index]);
	}

	_bncneon_rdd_sum128d(ret, ret2);

#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[BNCOMP_MAX_NUM_THREADS][DDSIZE], b4[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	__m256d ret4[DDSIZE], tmp4[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	_bncavx2_rdd_set0(ret4);
	
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a4[thread_index][0] = _mm256_load_pd(&(a->element[0][index]));
		a4[thread_index][1] = _mm256_load_pd(&(a->element[1][index]));
		b4[thread_index][0] = _mm256_load_pd(&(b->element[0][index]));
		b4[thread_index][1] = _mm256_load_pd(&(b->element[1][index]));

		_bncavx2_rdd_mul(tmp4[thread_index], a4[thread_index], b4[thread_index]);
		#pragma omp critical
			_bncavx2_rdd_add(ret4, ret4, tmp4[thread_index]);
	}

	_bncavx2_rdd_sum256d(ret, ret4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[BNCOMP_MAX_NUM_THREADS][DDSIZE], b8[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	__m512d ret8[DDSIZE], tmp8[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	_bncavx512_rdd_set0(ret8);
	
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a8[thread_index][0] = _mm512_load_pd(&(a->element[0][index]));
		a8[thread_index][1] = _mm512_load_pd(&(a->element[1][index]));
		b8[thread_index][0] = _mm512_load_pd(&(b->element[0][index]));
		b8[thread_index][1] = _mm512_load_pd(&(b->element[1][index]));

		_bncavx512_rdd_mul(tmp8[thread_index], a8[thread_index], b8[thread_index]);
		#pragma omp critical
			_bncavx512_rdd_add(ret8, ret8, tmp8[thread_index]);
	}

	_bncavx512_rdd_sum512d(ret, ret8);

#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	rdd_set_ui(ret, 0UL);

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < a->dim; i++)
	{
		thread_index = omp_get_thread_num();

		rdd_mul(tmp[thread_index], get_ddvector_i(a, i), get_ddvector_i(b, i));
	#pragma omp critical
		rdd_add(ret, ret, tmp[thread_index]);
	}
#endif

	return;
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void _bncomp_row_swap_ddmatrix(DDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
	long int i, j, true_end;
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	int thread_index;

	true_end = (col_end > mat->col_dim) ? mat->col_dim : col_end;

	#pragma omp parallel for private(thread_index)
	for(i = col_start; i < true_end; i++)
	{
		thread_index = omp_get_thread_num();

		rdd_set(tmp[thread_index], get_ddmatrix_ij(mat, row_index0, i));
		set_ddmatrix_ij(mat, row_index0, i, get_ddmatrix_ij(mat, row_index1, i));
		set_ddmatrix_ij(mat, row_index1, i, tmp[thread_index]);
	}
}


//--------
// Matrix
//--------

// Neon用のヘッダーとマクロ定義を追加
#if defined(__ARM_NEON) && defined(BNC_ENABLE_NEON)
#include <arm_neon.h>
//#define _BNC_NEON_WIDTH 2  // Neon は float64x2_t (2つのdouble)
#endif

/* c := a + b */
void _bncomp_add_ddmatrix(DDMatrix c, DDMatrix a, DDMatrix b)
{
	int thread_index;
	long int index, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_ddmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_ddmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[BNCOMP_MAX_NUM_THREADS][DDSIZE], aij4[BNCOMP_MAX_NUM_THREADS][DDSIZE], bij4[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij4[thread_index][0] = _mm256_load_pd(&(a->element[0][index]));
		aij4[thread_index][1] = _mm256_load_pd(&(a->element[1][index]));
		bij4[thread_index][0] = _mm256_load_pd(&(b->element[0][index]));
		bij4[thread_index][1] = _mm256_load_pd(&(b->element[1][index]));

		_bncavx2_rdd_add(tmp4[thread_index], aij4[thread_index], bij4[thread_index]);

		_mm256_store_pd(&(c->element[0][index]), tmp4[thread_index][0]);
		_mm256_store_pd(&(c->element[1][index]), tmp4[thread_index][1]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[BNCOMP_MAX_NUM_THREADS][DDSIZE], aij8[BNCOMP_MAX_NUM_THREADS][DDSIZE], bij8[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij8[thread_index][0] = _mm512_load_pd(&(a->element[0][index]));
		aij8[thread_index][1] = _mm512_load_pd(&(a->element[1][index]));
		bij8[thread_index][0] = _mm512_load_pd(&(b->element[0][index]));
		bij8[thread_index][1] = _mm512_load_pd(&(b->element[1][index]));

		_bncavx512_rdd_add(tmp8[thread_index], aij8[thread_index], bij8[thread_index]);

		_mm512_store_pd(&(c->element[0][index]), tmp8[thread_index][0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[thread_index][1]); 
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = c->real_row_dim * c->real_col_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svfloat64_t _a0 = svld1_f64(_pg, &(a->element[0][_ix]));
			svfloat64_t _a1 = svld1_f64(_pg, &(a->element[1][_ix]));
			svfloat64_t _b0 = svld1_f64(_pg, &(b->element[0][_ix]));
			svfloat64_t _b1 = svld1_f64(_pg, &(b->element[1][_ix]));
			svfloat64_t _o0, _o1;
			_bncsve2_rdd_add(_pg, &_o0, &_o1, _a0, _a1, _b0, _b1);
			svst1_f64(_pg, &(c->element[0][_ix]), _o0);
			svst1_f64(_pg, &(c->element[1][_ix]), _o1);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	float64x2_t tmp2[BNCOMP_MAX_NUM_THREADS][DDSIZE], aij2[BNCOMP_MAX_NUM_THREADS][DDSIZE], bij2[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij2[thread_index][0] = vld1q_f64(&(a->element[0][index]));
		aij2[thread_index][1] = vld1q_f64(&(a->element[1][index]));
		bij2[thread_index][0] = vld1q_f64(&(b->element[0][index]));
		bij2[thread_index][1] = vld1q_f64(&(b->element[1][index]));

		_bncneon_rdd_add(tmp2[thread_index], aij2[thread_index], bij2[thread_index]);

		vst1q_f64(&(c->element[0][index]), tmp2[thread_index][0]);
		vst1q_f64(&(c->element[1][index]), tmp2[thread_index][1]); 
	}
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE], aij[BNCOMP_MAX_NUM_THREADS][DDSIZE], bij[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index++)
	{
		thread_index = omp_get_thread_num();
		
		aij[thread_index][0] = a->element[0][index];
		aij[thread_index][1] = a->element[1][index];
		bij[thread_index][0] = b->element[0][index];
		bij[thread_index][1] = b->element[1][index];

		rdd_add(tmp[thread_index], aij[thread_index], bij[thread_index]);

		c->element[0][index] = tmp[thread_index][0];
		c->element[1][index] = tmp[thread_index][1]; 
	}
#endif // __AVX2__
}

/* c := a - b */
void _bncomp_sub_ddmatrix(DDMatrix c, DDMatrix a, DDMatrix b)
{
	int thread_index;
	long int index, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_ddmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_ddmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[BNCOMP_MAX_NUM_THREADS][DDSIZE], aij4[BNCOMP_MAX_NUM_THREADS][DDSIZE], bij4[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij4[thread_index][0] = _mm256_load_pd(&(a->element[0][index]));
		aij4[thread_index][1] = _mm256_load_pd(&(a->element[1][index]));
		bij4[thread_index][0] = _mm256_load_pd(&(b->element[0][index]));
		bij4[thread_index][1] = _mm256_load_pd(&(b->element[1][index]));

		_bncavx2_rdd_sub(tmp4[thread_index], aij4[thread_index], bij4[thread_index]);

		_mm256_store_pd(&(c->element[0][index]), tmp4[thread_index][0]);
		_mm256_store_pd(&(c->element[1][index]), tmp4[thread_index][1]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[BNCOMP_MAX_NUM_THREADS][DDSIZE], aij8[BNCOMP_MAX_NUM_THREADS][DDSIZE], bij8[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij8[thread_index][0] = _mm512_load_pd(&(a->element[0][index]));
		aij8[thread_index][1] = _mm512_load_pd(&(a->element[1][index]));
		bij8[thread_index][0] = _mm512_load_pd(&(b->element[0][index]));
		bij8[thread_index][1] = _mm512_load_pd(&(b->element[1][index]));

		_bncavx512_rdd_sub(tmp8[thread_index], aij8[thread_index], bij8[thread_index]);

		_mm512_store_pd(&(c->element[0][index]), tmp8[thread_index][0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[thread_index][1]); 
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = c->real_row_dim * c->real_col_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svfloat64_t _a0 = svld1_f64(_pg, &(a->element[0][_ix]));
			svfloat64_t _a1 = svld1_f64(_pg, &(a->element[1][_ix]));
			svfloat64_t _b0 = svld1_f64(_pg, &(b->element[0][_ix]));
			svfloat64_t _b1 = svld1_f64(_pg, &(b->element[1][_ix]));
			_b0 = svneg_f64_x(_pg, _b0);
			_b1 = svneg_f64_x(_pg, _b1);
			svfloat64_t _o0, _o1;
			_bncsve2_rdd_add(_pg, &_o0, &_o1, _a0, _a1, _b0, _b1);
			svst1_f64(_pg, &(c->element[0][_ix]), _o0);
			svst1_f64(_pg, &(c->element[1][_ix]), _o1);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	float64x2_t tmp2[BNCOMP_MAX_NUM_THREADS][DDSIZE], aij2[BNCOMP_MAX_NUM_THREADS][DDSIZE], bij2[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij2[thread_index][0] = vld1q_f64(&(a->element[0][index]));
		aij2[thread_index][1] = vld1q_f64(&(a->element[1][index]));
		bij2[thread_index][0] = vld1q_f64(&(b->element[0][index]));
		bij2[thread_index][1] = vld1q_f64(&(b->element[1][index]));

		_bncneon_rdd_sub(tmp2[thread_index], aij2[thread_index], bij2[thread_index]);

		vst1q_f64(&(c->element[0][index]), tmp2[thread_index][0]);
		vst1q_f64(&(c->element[1][index]), tmp2[thread_index][1]); 
	}
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE], aij[BNCOMP_MAX_NUM_THREADS][DDSIZE], bij[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index++)
	{
		thread_index = omp_get_thread_num();
		
		aij[thread_index][0] = a->element[0][index];
		aij[thread_index][1] = a->element[1][index];
		bij[thread_index][0] = b->element[0][index];
		bij[thread_index][1] = b->element[1][index];

		rdd_sub(tmp[thread_index], aij[thread_index], bij[thread_index]);

		c->element[0][index] = tmp[thread_index][0];
		c->element[1][index] = tmp[thread_index][1]; 
	}
#endif // __AVX2__
}

/* c := sc * a */
void _bncomp_cmul_ddmatrix(DDMatrix c, double sc[DDSIZE], DDMatrix a)
{
	long int i, j, index, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim;
	int thread_index;

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_ddmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_ddmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[BNCOMP_MAX_NUM_THREADS][DDSIZE], aij4[BNCOMP_MAX_NUM_THREADS][DDSIZE], sc4[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		sc4[thread_index][0] = _mm256_set1_pd(sc[0]);
		sc4[thread_index][1] = _mm256_set1_pd(sc[1]);
	}
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij4[thread_index][0] = _mm256_load_pd(&(a->element[0][index]));
		aij4[thread_index][1] = _mm256_load_pd(&(a->element[1][index]));

		_bncavx2_rdd_mul(tmp4[thread_index], sc4[thread_index], aij4[thread_index]);

		_mm256_store_pd(&(c->element[0][index]), tmp4[thread_index][0]);
		_mm256_store_pd(&(c->element[1][index]), tmp4[thread_index][1]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[BNCOMP_MAX_NUM_THREADS][DDSIZE], aij8[BNCOMP_MAX_NUM_THREADS][DDSIZE], sc8[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		sc8[thread_index][0] = _mm512_set1_pd(sc[0]);
		sc8[thread_index][1] = _mm512_set1_pd(sc[1]);
	}
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij8[thread_index][0] = _mm512_load_pd(&(a->element[0][index]));
		aij8[thread_index][1] = _mm512_load_pd(&(a->element[1][index]));

		_bncavx512_rdd_mul(tmp8[thread_index], sc8[thread_index], aij8[thread_index]);

		_mm512_store_pd(&(c->element[0][index]), tmp8[thread_index][0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[thread_index][1]); 
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = a->real_row_dim * a->real_col_dim; long _ix;
		svfloat64_t _v0 = svdup_n_f64(sc[0]);
		svfloat64_t _v1 = svdup_n_f64(sc[1]);
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svfloat64_t _a0 = svld1_f64(_pg, &(a->element[0][_ix]));
			svfloat64_t _a1 = svld1_f64(_pg, &(a->element[1][_ix]));
			svfloat64_t _o0, _o1;
			_bncsve2_rdd_mul(_pg, &_o0, &_o1, _v0, _v1, _a0, _a1);
			svst1_f64(_pg, &(c->element[0][_ix]), _o0);
			svst1_f64(_pg, &(c->element[1][_ix]), _o1);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	float64x2_t tmp2[BNCOMP_MAX_NUM_THREADS][DDSIZE], aij2[BNCOMP_MAX_NUM_THREADS][DDSIZE], sc2[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		sc2[thread_index][0] = vdupq_n_f64(sc[0]);
		sc2[thread_index][1] = vdupq_n_f64(sc[1]);
	}
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij2[thread_index][0] = vld1q_f64(&(a->element[0][index]));
		aij2[thread_index][1] = vld1q_f64(&(a->element[1][index]));

		_bncneon_rdd_mul(tmp2[thread_index], sc2[thread_index], aij2[thread_index]);

		vst1q_f64(&(c->element[0][index]), tmp2[thread_index][0]);
		vst1q_f64(&(c->element[1][index]), tmp2[thread_index][1]); 
	}
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE], aij[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index++)
	{
		thread_index = omp_get_thread_num();
		
		aij[thread_index][0] = a->element[0][index];
		aij[thread_index][1] = a->element[1][index];

		rdd_mul(tmp[thread_index], sc, aij[thread_index]);

		c->element[0][index] = tmp[thread_index][0];
		c->element[1][index] = tmp[thread_index][1]; 
	}
#endif // __AVX2__
}

/* c = a * b */
void _bncomp_mul_ddmatrix(DDMatrix ret, DDMatrix a, DDMatrix b)
{
	int thread_num, thread_index;
	long int i, j, k;
	long row_dim, col_dim, mid_dim;

	/* dimension check */
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_ddmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret->row_dim, ret->col_dim, a->row_dim, a->col_dim, b->row_dim, b->col_dim);
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long real_row_dim, real_col_dim, real_mid_dim;
	double cijval[BNCOMP_MAX_NUM_THREADS][4][DDSIZE];
    __m256d cij[BNCOMP_MAX_NUM_THREADS][DDSIZE], aik[BNCOMP_MAX_NUM_THREADS][DDSIZE], bkj[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp_mul[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
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
            cij[thread_index][0] = _mm256_setzero_pd();
            cij[thread_index][1] = _mm256_setzero_pd();
            for(k = 0; k < real_mid_dim; k += _BNC_D_WIDTH)
            {
                aik[thread_index][0] = _mm256_load_pd(&(a->element[0][i * real_mid_dim + k]));
                aik[thread_index][1] = _mm256_load_pd(&(a->element[1][i * real_mid_dim + k]));

                bkj[thread_index][0] = _mm256_set_pd(
                    b->element[0][(k + 3) * real_col_dim + j],
                    b->element[0][(k + 2) * real_col_dim + j],
                    b->element[0][(k + 1) * real_col_dim + j],
                    b->element[0][(k    ) * real_col_dim + j]
                );
                bkj[thread_index][1] = _mm256_set_pd(
                    b->element[1][(k + 3) * real_col_dim + j],
                    b->element[1][(k + 2) * real_col_dim + j],
                    b->element[1][(k + 1) * real_col_dim + j],
                    b->element[1][(k    ) * real_col_dim + j]
                );

                _bncavx2_rdd_mul(tmp_mul[thread_index], aik[thread_index], bkj[thread_index]);
				//#pragma omp critical
				_bncavx2_rdd_add(cij[thread_index], cij[thread_index], tmp_mul[thread_index]);
            }

            cijval[thread_index][0][0] = cij[thread_index][0][0]; cijval[thread_index][0][1] = cij[thread_index][1][0];
            cijval[thread_index][1][0] = cij[thread_index][0][1]; cijval[thread_index][1][1] = cij[thread_index][1][1];
            cijval[thread_index][2][0] = cij[thread_index][0][2]; cijval[thread_index][2][1] = cij[thread_index][1][2];
            cijval[thread_index][3][0] = cij[thread_index][0][3]; cijval[thread_index][3][1] = cij[thread_index][1][3];
            rdd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][1]);
            rdd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][2]);
            rdd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][3]);

            ret->element[0][i * real_col_dim + j] = cijval[thread_index][0][0];
            ret->element[1][i * real_col_dim + j] = cijval[thread_index][0][1];
        }
    }
#elif defined(__AVX512F__) // __AVX512F__
	long real_row_dim, real_col_dim, real_mid_dim;
	double cijval[BNCOMP_MAX_NUM_THREADS][8][DDSIZE];
    __m512d cij[BNCOMP_MAX_NUM_THREADS][DDSIZE], aik[BNCOMP_MAX_NUM_THREADS][DDSIZE], bkj[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp_mul[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, i, j, k, aik, bkj, cij, cijval, tmp_mul)
    for(i = 0; i < real_row_dim; i++)
    {
		thread_index = omp_get_thread_num();

        for(j = 0; j < real_col_dim; j++)
        {
            cij[thread_index][0] = _mm512_setzero_pd();
            cij[thread_index][1] = _mm512_setzero_pd();
            for(k = 0; k < real_mid_dim; k += _BNC_D_WIDTH)
            {
                aik[thread_index][0] = _mm512_load_pd(&(a->element[0][i * real_mid_dim + k]));
                aik[thread_index][1] = _mm512_load_pd(&(a->element[1][i * real_mid_dim + k]));

                bkj[thread_index][0] = _mm512_set_pd(
                    b->element[0][(k + 7) * real_col_dim + j],
                    b->element[0][(k + 6) * real_col_dim + j],
                    b->element[0][(k + 5) * real_col_dim + j],
                    b->element[0][(k + 4) * real_col_dim + j],
                    b->element[0][(k + 3) * real_col_dim + j],
                    b->element[0][(k + 2) * real_col_dim + j],
                    b->element[0][(k + 1) * real_col_dim + j],
                    b->element[0][(k    ) * real_col_dim + j]
                );
                bkj[thread_index][1] = _mm512_set_pd(
                    b->element[1][(k + 7) * real_col_dim + j],
                    b->element[1][(k + 6) * real_col_dim + j],
                    b->element[1][(k + 5) * real_col_dim + j],
                    b->element[1][(k + 4) * real_col_dim + j],
                    b->element[1][(k + 3) * real_col_dim + j],
                    b->element[1][(k + 2) * real_col_dim + j],
                    b->element[1][(k + 1) * real_col_dim + j],
                    b->element[1][(k    ) * real_col_dim + j]
                );
                _bncavx512_rdd_mul(tmp_mul[thread_index], aik[thread_index], bkj[thread_index]);
				//#pragma omp critical
                _bncavx512_rdd_add(cij[thread_index], cij[thread_index], tmp_mul[thread_index]);
            }

            cijval[thread_index][0][0] = cij[thread_index][0][0];
			cijval[thread_index][0][1] = cij[thread_index][1][0];
            cijval[thread_index][1][0] = cij[thread_index][0][1];
			cijval[thread_index][1][1] = cij[thread_index][1][1];
            cijval[thread_index][2][0] = cij[thread_index][0][2];
			cijval[thread_index][2][1] = cij[thread_index][1][2];
            cijval[thread_index][3][0] = cij[thread_index][0][3];
			cijval[thread_index][3][1] = cij[thread_index][1][3];
            cijval[thread_index][4][0] = cij[thread_index][0][4];
			cijval[thread_index][4][1] = cij[thread_index][1][4];
            cijval[thread_index][5][0] = cij[thread_index][0][5];
			cijval[thread_index][5][1] = cij[thread_index][1][5];
            cijval[thread_index][6][0] = cij[thread_index][0][6];
			cijval[thread_index][6][1] = cij[thread_index][1][6];
            cijval[thread_index][7][0] = cij[thread_index][0][7];
			cijval[thread_index][7][1] = cij[thread_index][1][7];

            rdd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][1]);
            rdd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][2]);
            rdd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][3]);
            rdd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][4]);
            rdd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][5]);
            rdd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][6]);
            rdd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][7]);

            ret->element[0][i * real_col_dim + j] = cijval[thread_index][0][0];
            ret->element[1][i * real_col_dim + j] = cijval[thread_index][0][1];
        }
    }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic) + OpenMP
	/* Outer-product GEMM with i parallelized; SVE locals live per-iteration
	 * (per-thread) on the automatic stack — no scratch arrays of svfloat64_t. */
	{
		long real_row_dim = ret->real_row_dim;
		long real_col_dim = ret->real_col_dim;
		long real_mid_dim = a->real_col_dim;
		long vl = (long)svcntd();

		#pragma omp parallel for private(thread_index, i, j, k)
		for(i = 0; i < real_row_dim; i++){
			thread_index = omp_get_thread_num();
			(void)thread_index;
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
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	/* NEON (float64x2_t): outer-product GEMM, mirrors serial mul_ddmatrix().
	 * Same per-cell k-accumulation order as serial → serial vs parallel results
	 * are bitwise identical (rel.err = 0). */
	long real_row_dim, real_col_dim, real_mid_dim;
	float64x2_t cij[BNCOMP_MAX_NUM_THREADS][DDSIZE], aik[BNCOMP_MAX_NUM_THREADS][DDSIZE], bkj[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp_mul[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, i, j, k, aik, bkj, cij, tmp_mul)
	for(i = 0; i < real_row_dim; i++)
	{
		thread_index = omp_get_thread_num();
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			_bncneon_set0_dd(cij[thread_index]);
			for(k = 0; k < real_mid_dim; k++)
			{
				aik[thread_index][0] = vdupq_n_f64(a->element[0][i * real_mid_dim + k]);
				aik[thread_index][1] = vdupq_n_f64(a->element[1][i * real_mid_dim + k]);
				bkj[thread_index][0] = vld1q_f64(&(b->element[0][k * real_col_dim + j]));
				bkj[thread_index][1] = vld1q_f64(&(b->element[1][k * real_col_dim + j]));
				_bncneon_rdd_mul(tmp_mul[thread_index], aik[thread_index], bkj[thread_index]);
				_bncneon_rdd_add(cij[thread_index], cij[thread_index], tmp_mul[thread_index]);
			}
			vst1q_f64(&(ret->element[0][i * real_col_dim + j]), cij[thread_index][0]);
			vst1q_f64(&(ret->element[1][i * real_col_dim + j]), cij[thread_index][1]);
		}
	}
#else // __AVX2__
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE], ret_ij[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_dd(tmp[thread_index]);
	}

	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = a->col_dim;

	#pragma omp parallel for private(thread_index, i, j, k, ret_ij, tmp)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		for(j = 0; j < col_dim; j++)
		{
			rdd_set0(ret_ij[thread_index]);
			for(k = 0; k < mid_dim; k++)
			{
				rdd_mul(tmp[thread_index], get_ddmatrix_ij(a, i, k), get_ddmatrix_ij(b, k, j));
				#pragma omp critical
				rdd_add(ret_ij[thread_index], tmp[thread_index], ret_ij[thread_index]);
			}
			
			set_ddmatrix_ij(ret, i, j, ret_ij[thread_index]);
		}
	}
#endif // __AVX2__
}


/* c := a */
void _bncomp_subst_ddmatrix(DDMatrix c, DDMatrix a)
{
	long int i, j, index;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_ddmatrix\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int real_row_dim, real_col_dim;

	real_row_dim = c->real_row_dim;
	real_col_dim = c->real_col_dim;

	#pragma omp parallel for private(j, index)
	for(i = 0; i < real_row_dim; i++)
	{
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			index = i * real_col_dim + j;
			_mm256_store_pd(&(c->element[0][i * real_col_dim + j]), _mm256_load_pd(&(a->element[0][index])));
			_mm256_store_pd(&(c->element[1][i * real_col_dim + j]), _mm256_load_pd(&(a->element[1][index])));
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
			_mm512_store_pd(&(c->element[0][i * real_col_dim + j]), _mm512_load_pd(&(a->element[0][index])));
			_mm512_store_pd(&(c->element[1][i * real_col_dim + j]), _mm512_load_pd(&(a->element[1][index])));
		}
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = c->real_row_dim * c->real_col_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svst1_f64(_pg, &(c->element[0][_ix]), svld1_f64(_pg, &(a->element[0][_ix])));
			svst1_f64(_pg, &(c->element[1][_ix]), svld1_f64(_pg, &(a->element[1][_ix])));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	long int real_row_dim, real_col_dim;

	real_row_dim = c->real_row_dim;
	real_col_dim = c->real_col_dim;

	#pragma omp parallel for private(j, index)
	for(i = 0; i < real_row_dim; i++)
	{
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			index = i * real_col_dim + j;
			vst1q_f64(&(c->element[0][i * real_col_dim + j]), vld1q_f64(&(a->element[0][index])));
			vst1q_f64(&(c->element[1][i * real_col_dim + j]), vld1q_f64(&(a->element[1][index])));
		}
	}
#else // others
	#pragma omp parallel for private(j)
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_ddmatrix_ij(c, i, j, get_ddmatrix_ij(a, i, j));
		}
	}
#endif // AVX2
}

/* c := I */
void _bncomp_setI_ddmatrix(DDMatrix c)
{
	long int i, real_total_dim;
	double tmp0[DDSIZE], tmp1[DDSIZE];

	rdd_set_ui(tmp0, 0UL);
	rdd_set_ui(tmp1, 1UL);

	real_total_dim = c->real_row_dim * c->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&c->element[0][i], zero4);
		_mm256_store_pd(&c->element[1][i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&c->element[0][i], zero8);
		_mm512_store_pd(&c->element[1][i], zero8);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = c->real_row_dim * c->real_col_dim; long _ix;
		svfloat64_t _z = svdup_n_f64(0.0);
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svst1_f64(_pg, &(c->element[0][_ix]), _z);
			svst1_f64(_pg, &(c->element[1][_ix]), _z);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	float64x2_t zero2;

	zero2 = vdupq_n_f64(0.0);
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		vst1q_f64(&c->element[0][i], zero2);
		vst1q_f64(&c->element[1][i], zero2);
	}
#else // others
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i++)
	{
		c->element[0][i] = 0.0;
		c->element[1][i] = 0.0;	
	}
#endif // __AVX2__

	rdd_set_ui(tmp1, 1UL);

	#pragma omp parallel for
	for(i = 0; i < c->row_dim; i++)
	{
		if(i < c->col_dim)
			set_ddmatrix_ij(c, i, i, tmp1);
	}
}

// set a zero matrix
void _bncomp_set0_ddmatrix(DDMatrix mat)
{
	long int i, real_total_dim;

	real_total_dim = mat->real_row_dim * mat->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&mat->element[0][i], zero4);
		_mm256_store_pd(&mat->element[1][i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&mat->element[0][i], zero8);
		_mm512_store_pd(&mat->element[1][i], zero8);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = mat->real_row_dim * mat->real_col_dim; long _ix;
		svfloat64_t _z = svdup_n_f64(0.0);
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svst1_f64(_pg, &(mat->element[0][_ix]), _z);
			svst1_f64(_pg, &(mat->element[1][_ix]), _z);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	float64x2_t zero2;

	zero2 = vdupq_n_f64(0.0);
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		vst1q_f64(&mat->element[0][i], zero2);
		vst1q_f64(&mat->element[1][i], zero2);
	}
#else // others
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i++)
	{
		mat->element[0][i] = 0.0;
		mat->element[1][i] = 0.0;	
	}
#endif // __AVX2__
}

/* v := a * vb */
void _bncomp_mul_ddmatrix_ddvec(DDVector v, DDMatrix a, DDVector vb)
{
	long int i, j, row_dim;
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp1[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	int thread_index;

	/* Check Dimension */
	if((v->dim < a->row_dim) || (vb->dim < a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_ddmatrix_ddvec\n");
		return;
	}

	row_dim = a->row_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int ij_index, real_row_dim, real_col_dim;
	__m256d tmp4[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp1_4[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	__m256d aij4[BNCOMP_MAX_NUM_THREADS][DDSIZE], vbj4[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j, ij_index)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		_bncavx2_set0_dd(tmp4[thread_index]);
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij4[thread_index][0] = _mm256_load_pd(&(a->element[0][ij_index]));
			aij4[thread_index][1] = _mm256_load_pd(&(a->element[1][ij_index]));
			vbj4[thread_index][0] = _mm256_load_pd(&(vb->element[0][j]));
			vbj4[thread_index][1] = _mm256_load_pd(&(vb->element[1][j]));

			_bncavx2_rdd_mul(tmp1_4[thread_index], aij4[thread_index], vbj4[thread_index]);
			_bncavx2_rdd_add(tmp4[thread_index], tmp4[thread_index], tmp1_4[thread_index]);
		}
		_bncavx2_rdd_sum256d(tmp[thread_index], tmp4[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int ij_index, real_col_dim;
	__m512d tmp8[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp1_8[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	__m512d aij8[BNCOMP_MAX_NUM_THREADS][DDSIZE], vbj8[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j, ij_index)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		_bncavx512_set0_dd(tmp8[thread_index]);
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			thread_index = omp_get_thread_num();

			ij_index = i * real_col_dim + j;
			aij8[thread_index][0] = _mm512_load_pd(&(a->element[0][ij_index]));
			aij8[thread_index][1] = _mm512_load_pd(&(a->element[1][ij_index]));
			vbj8[thread_index][0] = _mm512_load_pd(&(vb->element[0][j]));
			vbj8[thread_index][1] = _mm512_load_pd(&(vb->element[1][j]));

			_bncavx512_rdd_mul(tmp1_8[thread_index], aij8[thread_index], vbj8[thread_index]);
			_bncavx512_rdd_add(tmp8[thread_index], tmp8[thread_index], tmp1_8[thread_index]);
		}
		_bncavx512_rdd_sum512d(tmp[thread_index], tmp8[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic) + OpenMP
	/* SVE2 dot-product matvec: per row i (parallel), accumulate
	 * acc[lane] += A[i][j+lane] * vb[j+lane] across the j SIMD axis,
	 * then horizontally reduce acc across all VL lanes via store-and-add. */
	{
		long sve_real_col_dim = a->real_col_dim;
		long sve_vl = (long)svcntd();

		#pragma omp parallel for private(thread_index, i, j) firstprivate(sve_vl, sve_real_col_dim)
		for(i = 0; i < row_dim; i++)
		{
			thread_index = omp_get_thread_num();
			(void)thread_index;
			svfloat64_t acc0, acc1;
			_bncsve2_rdd_set0(&acc0, &acc1);

			for(j = 0; j < sve_real_col_dim; j += sve_vl)
			{
				svbool_t pg = svwhilelt_b64_s64((int64_t)j, (int64_t)sve_real_col_dim);
				svfloat64_t aij0 = svld1_f64(pg, &(a->element[0][i * sve_real_col_dim + j]));
				svfloat64_t aij1 = svld1_f64(pg, &(a->element[1][i * sve_real_col_dim + j]));
				svfloat64_t bj0  = svld1_f64(pg, &(vb->element[0][j]));
				svfloat64_t bj1  = svld1_f64(pg, &(vb->element[1][j]));
				svfloat64_t prod0, prod1;
				_bncsve2_rdd_mul(pg, &prod0, &prod1, aij0, aij1, bj0, bj1);
				_bncsve2_rdd_add(pg, &acc0, &acc1, acc0, acc1, prod0, prod1);
			}

			/* Horizontal sum: extract every lane, do multi-precision add */
			{
				double acc0_arr[sve_vl], acc1_arr[sve_vl];
				svst1_f64(svptrue_b64(), acc0_arr, acc0);
				svst1_f64(svptrue_b64(), acc1_arr, acc1);
				double sum_dd[DDSIZE] = { 0.0, 0.0 };
				long lane;
				for(lane = 0; lane < sve_vl; lane++)
				{
					double lane_dd[DDSIZE] = { acc0_arr[lane], acc1_arr[lane] };
					rdd_add(sum_dd, sum_dd, lane_dd);
				}
				v->element[0][i] = sum_dd[0];
				v->element[1][i] = sum_dd[1];
			}
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	long int ij_index, real_col_dim;
	float64x2_t tmp2[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp1_2[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	float64x2_t aij2[BNCOMP_MAX_NUM_THREADS][DDSIZE], vbj2[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j, ij_index)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		_bncneon_set0_dd(tmp2[thread_index]);
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij2[thread_index][0] = vld1q_f64(&(a->element[0][ij_index]));
			aij2[thread_index][1] = vld1q_f64(&(a->element[1][ij_index]));
			vbj2[thread_index][0] = vld1q_f64(&(vb->element[0][j]));
			vbj2[thread_index][1] = vld1q_f64(&(vb->element[1][j]));

			_bncneon_rdd_mul(tmp1_2[thread_index], aij2[thread_index], vbj2[thread_index]);
			_bncneon_rdd_add(tmp2[thread_index], tmp2[thread_index], tmp1_2[thread_index]);
		}
		_bncneon_rdd_sum128d(tmp[thread_index], tmp2[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
	}
#else // others
	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		rdd_set_ui(tmp[thread_index], 0UL);
		for(j = 0; j < a->col_dim; j++)
		{
			rdd_mul(tmp1[thread_index], get_ddmatrix_ij(a, i, j), get_ddvector_i(vb, j));
			rdd_add(tmp[thread_index], tmp[thread_index], tmp1[thread_index]);
		}
		set_ddvector_i(v, i, tmp[thread_index]);
	}
#endif // __AVX2__
}

/* v := a^T * vb */
void _bncomp_mul_ddmatrixt_ddvec(DDVector v, DDMatrix a, DDVector vb)
{
	long int i, j, col_dim;
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp1[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	int thread_index;

	/* Check Dimension */
	if((v->dim < a->col_dim) || (vb->dim < a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_ddmatrixt_ddvec\n");
		return;
	}

	col_dim = a->col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int real_row_dim, real_col_dim;
	__m256d tmp4[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp1_4[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	__m256d aij4[BNCOMP_MAX_NUM_THREADS][DDSIZE], vbj4[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < col_dim; i++)
	{
		thread_index = omp_get_thread_num();

		_bncavx2_set0_dd(tmp4[thread_index]);
		for(j = 0; j < real_row_dim; j += _BNC_D_WIDTH)
		{
			aij4[thread_index][0] = _mm256_set_pd(
				a->element[0][(j + 3) * real_col_dim + i],
				a->element[0][(j + 2) * real_col_dim + i],
				a->element[0][(j + 1) * real_col_dim + i],
				a->element[0][(j    ) * real_col_dim + i]
			);
			aij4[thread_index][1] = _mm256_set_pd(
				a->element[1][(j + 3) * real_col_dim + i],
				a->element[1][(j + 2) * real_col_dim + i],
				a->element[1][(j + 1) * real_col_dim + i],
				a->element[1][(j    ) * real_col_dim + i]
			);
			vbj4[thread_index][0] = _mm256_load_pd(&(vb->element[0][j]));
			vbj4[thread_index][1] = _mm256_load_pd(&(vb->element[1][j]));

			_bncavx2_rdd_mul(tmp1_4[thread_index], aij4[thread_index], vbj4[thread_index]);
			_bncavx2_rdd_add(tmp4[thread_index], tmp4[thread_index], tmp1_4[thread_index]);
		}
		_bncavx2_rdd_sum256d(tmp[thread_index], tmp4[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int real_row_dim, real_col_dim;
	__m512d tmp8[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp1_8[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	__m512d aij8[BNCOMP_MAX_NUM_THREADS][DDSIZE], vbj8[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < col_dim; i++)
	{
		thread_index = omp_get_thread_num();

		_bncavx512_set0_dd(tmp8[thread_index]);
		for(j = 0; j < real_row_dim; j += _BNC_D_WIDTH)
		{
			thread_index = omp_get_thread_num();

			aij8[thread_index][0] = _mm512_set_pd(
				a->element[0][(j + 7) * real_col_dim + i],
				a->element[0][(j + 6) * real_col_dim + i],
				a->element[0][(j + 5) * real_col_dim + i],
				a->element[0][(j + 4) * real_col_dim + i],
				a->element[0][(j + 3) * real_col_dim + i],
				a->element[0][(j + 2) * real_col_dim + i],
				a->element[0][(j + 1) * real_col_dim + i],
				a->element[0][(j    ) * real_col_dim + i]
			);
			aij8[thread_index][1] = _mm512_set_pd(
				a->element[1][(j + 7) * real_col_dim + i],
				a->element[1][(j + 6) * real_col_dim + i],
				a->element[1][(j + 5) * real_col_dim + i],
				a->element[1][(j + 4) * real_col_dim + i],
				a->element[1][(j + 3) * real_col_dim + i],
				a->element[1][(j + 2) * real_col_dim + i],
				a->element[1][(j + 1) * real_col_dim + i],
				a->element[1][(j    ) * real_col_dim + i]
			);
			vbj8[thread_index][0] = _mm512_load_pd(&(vb->element[0][j]));
			vbj8[thread_index][1] = _mm512_load_pd(&(vb->element[1][j]));

			_bncavx512_rdd_mul(tmp1_8[thread_index], aij8[thread_index], vbj8[thread_index]);
			_bncavx512_rdd_add(tmp8[thread_index], tmp8[thread_index], tmp1_8[thread_index]);
		}
		_bncavx512_rdd_sum512d(tmp[thread_index], tmp8[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic) + OpenMP
	/* SVE2 transpose-matvec: outer-product formulation.
	 * v[i] = sum_j A[j][i] * vb[j]  →  tile i over VL lanes (parallel),
	 * sequential j inner; A[j][i_tile] is contiguous (good for SIMD load). */
	{
		long sve_real_row_dim = a->real_row_dim;
		long sve_real_col_dim = a->real_col_dim;
		long sve_vl = (long)svcntd();

		#pragma omp parallel for private(thread_index, i, j) firstprivate(sve_vl, sve_real_row_dim, sve_real_col_dim)
		for(i = 0; i < sve_real_col_dim; i += sve_vl)
		{
			thread_index = omp_get_thread_num();
			(void)thread_index;
			svbool_t pg = svwhilelt_b64_s64((int64_t)i, (int64_t)sve_real_col_dim);
			svfloat64_t acc0, acc1;
			_bncsve2_rdd_set0(&acc0, &acc1);

			for(j = 0; j < sve_real_row_dim; j++)
			{
				svfloat64_t aji0 = svld1_f64(pg, &(a->element[0][j * sve_real_col_dim + i]));
				svfloat64_t aji1 = svld1_f64(pg, &(a->element[1][j * sve_real_col_dim + i]));
				svfloat64_t bj0  = svdup_n_f64(vb->element[0][j]);
				svfloat64_t bj1  = svdup_n_f64(vb->element[1][j]);
				svfloat64_t prod0, prod1;
				_bncsve2_rdd_mul(pg, &prod0, &prod1, aji0, aji1, bj0, bj1);
				_bncsve2_rdd_add(pg, &acc0, &acc1, acc0, acc1, prod0, prod1);
			}

			svst1_f64(pg, &(v->element[0][i]), acc0);
			svst1_f64(pg, &(v->element[1][i]), acc1);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	long int real_row_dim, real_col_dim;
	float64x2_t tmp2[BNCOMP_MAX_NUM_THREADS][DDSIZE], tmp1_2[BNCOMP_MAX_NUM_THREADS][DDSIZE];
	float64x2_t aij2[BNCOMP_MAX_NUM_THREADS][DDSIZE], vbj2[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < col_dim; i++)
	{
		thread_index = omp_get_thread_num();

		_bncneon_set0_dd(tmp2[thread_index]);
		for(j = 0; j < real_row_dim; j += _BNC_D_WIDTH)
		{
			// 転置行列アクセスのため個別に要素をセット
			double aij_tmp0[2] = {
				a->element[0][(j    ) * real_col_dim + i],
				a->element[0][(j + 1) * real_col_dim + i]
			};
			double aij_tmp1[2] = {
				a->element[1][(j    ) * real_col_dim + i],
				a->element[1][(j + 1) * real_col_dim + i]
			};
			aij2[thread_index][0] = vld1q_f64(aij_tmp0);
			aij2[thread_index][1] = vld1q_f64(aij_tmp1);
			
			vbj2[thread_index][0] = vld1q_f64(&(vb->element[0][j]));
			vbj2[thread_index][1] = vld1q_f64(&(vb->element[1][j]));

			_bncneon_rdd_mul(tmp1_2[thread_index], aij2[thread_index], vbj2[thread_index]);
			_bncneon_rdd_add(tmp2[thread_index], tmp2[thread_index], tmp1_2[thread_index]);
		}
		_bncneon_rdd_sum128d(tmp[thread_index], tmp2[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
	}
#else // others
	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < a->col_dim; i++)
	{
		thread_index = omp_get_thread_num();

		rdd_set_ui(tmp[thread_index], 0UL);
		for(j = 0; j < a->row_dim; j++)
		{
			rdd_mul(tmp1[thread_index], get_ddmatrix_ij(a, j, i), get_ddvector_i(vb, j));
			rdd_add(tmp[thread_index], tmp[thread_index], tmp1[thread_index]);
		}
		set_ddvector_i(v, i, tmp[thread_index]);
	}
#endif // __AVX2__
}

// Matrix multiplication based on Ozaki scheme
/*------------------------------------------------------------------------------*/
/* Matrix  multiplication based on Ozaki scheme                          */
/*                                                                               */
/* mul_ddmatrix_oz() is OpenMP-parallel itself: it cuts the rows of   */
/* ret into blocks and gives one block at a time to a thread, which runs every   */
/* slice product for it with a single-threaded BLAS call and accumulates in      */
/* DD on the spot.  That parallelizes the splitting and the accumulation as   */
/* well, whereas the code that used to stand here parallelized only the outer    */
/* slice loop and pushed every accumulation through an omp critical -- which     */
/* serialized the expensive half and made the sum order depend on the thread     */
/* interleaving.  Delegating is both faster and reproducible.                    */
/*------------------------------------------------------------------------------*/
void _bncomp_mul_ddmatrix_oz(DDMatrix ret, DDMatrix a, int max_num_div_a, DDMatrix b, int max_num_div_b)
{
    mul_ddmatrix_oz(ret, a, max_num_div_a, b, max_num_div_b);
}

#if 0

// Fit dimension to be multiple of min_dim
void _bncomp_mul_cddmatrix_oz_3m(CDDMatrix ret, CDDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CDDMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    DDMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_ddmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    _bncomp_mul_ddmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_a_image);
    _bncomp_sub_ddmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        mul_ddmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        mul_ddmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_a_image);
        add_ddmatrix(ret->im, t3, t4);
    #else // USE_4M
    */
        // 3M
        _bncomp_add_ddmatrix(t3, a->re, a->im);
        _bncomp_add_ddmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        _bncomp_mul_ddmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        _bncomp_sub_ddmatrix(ret->im, ret->im, t1);
        _bncomp_sub_ddmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_ddmatrix(t1);
    free_ddmatrix(t2);
    free_ddmatrix(t3);
    free_ddmatrix(t4);
}

// Fit dimension to be multiple of min_dim
void _bncomp_mul_cddmatrix_oz_4m(CDDMatrix ret, CDDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CDDMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    DDMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_ddmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    _bncomp_mul_ddmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_a_image);
    _bncomp_sub_ddmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        _bncomp_mul_ddmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        _bncomp_mul_ddmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_a_image);
        _bncomp_add_ddmatrix(ret->im, t3, t4);
    /*
    #else // USE_4M 
        // 3M
        add_ddmatrix(t3, a->re, a->im);
        add_ddmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        mul_ddmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        sub_ddmatrix(ret->im, ret->im, t1);
        sub_ddmatrix(ret->im, ret->im, t2);
    #endif // USE_4M
    */

    free_ddmatrix(t1);
    free_ddmatrix(t2);
    free_ddmatrix(t3);
    free_ddmatrix(t4);
}

// Simple triple-loop-way matrix multiplication (3M)
void _bncomp_mul_cddmatrix_3m(CDDMatrix ret, CDDMatrix a, CDDMatrix b)
{
    DDMatrix t1, t2, t3, t4;
 
    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_ddmatrix(t1, a->re, b->re);
    _bncomp_mul_ddmatrix(t2, a->im, b->im);
    _bncomp_sub_ddmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        _bncomp_mul_ddmatrix(t3, a->im, b->re);
        _bncomp_mul_ddmatrix(t4, a->re, b->im);
        _bncomp_add_ddmatrix(ret->im, t3, t4);
    #else // USE_4M
    */
        // 3M
        _bncomp_add_ddmatrix(t3, a->re, a->im);
        _bncomp_add_ddmatrix(t4, b->re, b->im);
        _bncomp_mul_ddmatrix(ret->im, t3, t4);
        _bncomp_sub_ddmatrix(ret->im, ret->im, t1);
        _bncomp_sub_ddmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_ddmatrix(t1);
    free_ddmatrix(t2);
    free_ddmatrix(t3);
    free_ddmatrix(t4);
}

// Simple triple-loop-way matrix multiplication (4M)
void _bncomp_mul_cddmatrix_4m(CDDMatrix ret, CDDMatrix a, CDDMatrix b)
{
    DDMatrix t1, t2, t3, t4;
 
    t1 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_ddmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_ddmatrix(t1, a->re, b->re);
    _bncomp_mul_ddmatrix(t2, a->im, b->im);
    _bncomp_sub_ddmatrix(ret->re, t1, t2);

    // 4M
    
    //#ifdef USE_4M
        _bncomp_mul_ddmatrix(t3, a->im, b->re);
        _bncomp_mul_ddmatrix(t4, a->re, b->im);
        _bncomp_add_ddmatrix(ret->im, t3, t4);
    //#else // USE_4M
	/*
        // 3M
        _bncomp_add_ddmatrix(t3, a->re, a->im);
        _bncomp_add_ddmatrix(t4, b->re, b->im);
        _bncomp_mul_ddmatrix(ret->im, t3, t4s);
        _bncomp_sub_ddmatrix(ret->im, ret->im, t1);
        _bncomp_sub_ddmatrix(ret->im, ret->im, t2);
	*/
    //#endif // USE_4M

    free_ddmatrix(t1);
    free_ddmatrix(t2);
    free_ddmatrix(t3);
    free_ddmatrix(t4);
}
#endif // 0

/****************************************************************************/
/* OpenMP-parallel LU decomposition (partial pivoting, SIMD inner kernel).  */
/* Mirrors DDLUdecompPM but parallelizes the O(n^3) trailing-submatrix       */
/* update over rows j (each below-pivot row is independent). Pivot search    */
/* and the multiplier column stay sequential. On SVE2 builds the inner       */
/* kernel uses the NEON path (arrays-of-SVE-types are unusable), exactly as  */
/* the serial routine does.                                                  */
/****************************************************************************/
int _bncomp_DDLUdecompPM(DDMatrix a, long int ch[])
{
	long int i, j, k, imax, itmp, dim, dim_start, dim_end;
	double dtmp[DDSIZE], dtmp1[DDSIZE], dmaxii[DDSIZE];

	dim = a->col_dim;
	for(i = 0; i < a->row_dim; i++) ch[i] = i;

	for(i = 0; i < a->row_dim; i++)
	{
		/* partial pivoting (sequential) */
		rdd_abs(dmaxii, get_ddmatrix_ij(a, i, i));
		imax = i;
		for(j = i + 1; j < a->row_dim; j++)
		{
			rdd_abs(dtmp, get_ddmatrix_ij(a, j, i));
			if(rdd_cmp(dtmp, dmaxii) > 0) { imax = j; rdd_set(dmaxii, dtmp); }
		}
		if(rdd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! _bncomp_DDLUdecompPM!\n", i);
			return -1;
		}
		if(imax != i)
		{
			itmp = ch[imax]; ch[imax] = ch[i]; ch[i] = itmp;
			row_swap_ddmatrix(a, i, imax, 0, a->col_dim);
		}
		/* multiplier column (sequential, O(n)) */
		for(j = i + 1; j < dim; j++)
		{
			rdd_div(dtmp, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, i));
			set_ddmatrix_ij(a, j, i, dtmp);
		}
		/* trailing-submatrix update : parallel over rows j */
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->real_col_dim;
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
		#pragma omp parallel for schedule(static) private(k)
		for(j = i + 1; j < dim; j++)
		{
			double _t[DDSIZE], _t1[DDSIZE];
			long int index_ji = j * a->real_col_dim + i, index_ik, index_jk, k_end;
			__m256d dtmp256[DDSIZE], aji256[DDSIZE], ajk256[DDSIZE], aik256[DDSIZE];
			aji256[0] = _mm256_set1_pd(a->element[0][index_ji]);
			aji256[1] = _mm256_set1_pd(a->element[1][index_ji]);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = i + 1; k < k_end; k++)
			{
				rdd_mul(_t1, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(_t, get_ddmatrix_ij(a, j, k), _t1);
				set_ddmatrix_ij(a, j, k, _t);
			}
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				aik256[0] = _mm256_load_pd(&(a->element[0][index_ik]));
				aik256[1] = _mm256_load_pd(&(a->element[1][index_ik]));
				_bncavx2_rdd_mul(dtmp256, aji256, aik256);
				index_jk = j * a->real_col_dim + k;
				ajk256[0] = _mm256_load_pd(&(a->element[0][index_jk]));
				ajk256[1] = _mm256_load_pd(&(a->element[1][index_jk]));
				_bncavx2_rdd_sub(dtmp256, ajk256, dtmp256);
				_mm256_store_pd(&(a->element[0][index_jk]), dtmp256[0]);
				_mm256_store_pd(&(a->element[1][index_jk]), dtmp256[1]);
			}
		}
#elif defined(__AVX512F__) // __AVX512F__
		#pragma omp parallel for schedule(static) private(k)
		for(j = i + 1; j < dim; j++)
		{
			double _t[DDSIZE], _t1[DDSIZE];
			long int index_ji = j * a->real_col_dim + i, index_ik, index_jk, k_end;
			__m512d dtmp512[DDSIZE], aji512[DDSIZE], ajk512[DDSIZE], aik512[DDSIZE];
			aji512[0] = _mm512_set1_pd(a->element[0][index_ji]);
			aji512[1] = _mm512_set1_pd(a->element[1][index_ji]);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = i + 1; k < k_end; k++)
			{
				rdd_mul(_t1, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(_t, get_ddmatrix_ij(a, j, k), _t1);
				set_ddmatrix_ij(a, j, k, _t);
			}
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				aik512[0] = _mm512_load_pd(&(a->element[0][index_ik]));
				aik512[1] = _mm512_load_pd(&(a->element[1][index_ik]));
				_bncavx512_rdd_mul(dtmp512, aji512, aik512);
				index_jk = j * a->real_col_dim + k;
				ajk512[0] = _mm512_load_pd(&(a->element[0][index_jk]));
				ajk512[1] = _mm512_load_pd(&(a->element[1][index_jk]));
				_bncavx512_rdd_sub(dtmp512, ajk512, dtmp512);
				_mm512_store_pd(&(a->element[0][index_jk]), dtmp512[0]);
				_mm512_store_pd(&(a->element[1][index_jk]), dtmp512[1]);
			}
		}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon (also used by SVE2 build)
		#pragma omp parallel for schedule(static) private(k)
		for(j = i + 1; j < dim; j++)
		{
			double _t[DDSIZE], _t1[DDSIZE];
			long int index_ji = j * a->real_col_dim + i, index_ik, index_jk, k_end;
			float64x2_t dtmp_n[DDSIZE], aji_n[DDSIZE], ajk_n[DDSIZE], aik_n[DDSIZE];
			aji_n[0] = vdupq_n_f64(a->element[0][index_ji]);
			aji_n[1] = vdupq_n_f64(a->element[1][index_ji]);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = i + 1; k < k_end; k++)
			{
				rdd_mul(_t1, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(_t, get_ddmatrix_ij(a, j, k), _t1);
				set_ddmatrix_ij(a, j, k, _t);
			}
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				aik_n[0] = vld1q_f64(&(a->element[0][index_ik]));
				aik_n[1] = vld1q_f64(&(a->element[1][index_ik]));
				_bncneon_rdd_mul(dtmp_n, aji_n, aik_n);
				index_jk = j * a->real_col_dim + k;
				ajk_n[0] = vld1q_f64(&(a->element[0][index_jk]));
				ajk_n[1] = vld1q_f64(&(a->element[1][index_jk]));
				_bncneon_rdd_sub(dtmp_n, ajk_n, dtmp_n);
				vst1q_f64(&(a->element[0][index_jk]), dtmp_n[0]);
				vst1q_f64(&(a->element[1][index_jk]), dtmp_n[1]);
			}
		}
#else // others
		#pragma omp parallel for schedule(static) private(k)
		for(j = i + 1; j < dim; j++)
		{
			double _t[DDSIZE], _t1[DDSIZE];
			for(k = i + 1; k < dim; k++)
			{
				rdd_mul(_t1, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(_t, get_ddmatrix_ij(a, j, k), _t1);
				set_ddmatrix_ij(a, j, k, _t);
			}
		}
#endif // __AVX2__
	}
	return 0;
}

/* OpenMP solver for an LU-decomposed system (partial pivoting).            */
/* The forward-substitution row updates are independent for a fixed column  */
/* and run in parallel; back substitution is a sequential reduction.        */
int _bncomp_SolveDDLSPM(DDVector answer, DDMatrix lu, DDVector b, long int ch[])
{
	long int i, j, dim;
	double dtmp[DDSIZE], dtmp1[DDSIZE];

	dim = answer->dim;
	for(i = 0; i < dim; i++) set_ddvector_i(answer, i, get_ddvector_i(b, ch[i]));

	/* Forward (parallel over j) */
	for(i = 0; i < dim; i++)
	{
		rdd_abs(dtmp, get_ddmatrix_ij(lu, i, i));
		if(rdd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(_bncomp_SolveDDLSPM, %ld)\n", i);
			return -1;
		}
		#pragma omp parallel for schedule(static)
		for(j = i + 1; j < dim; j++)
		{
			double _t[DDSIZE], _t1[DDSIZE];
			rdd_mul(_t1, get_ddmatrix_ij(lu, j, i), get_ddvector_i(answer, i));
			rdd_sub(_t, get_ddvector_i(answer, j), _t1);
			set_ddvector_i(answer, j, _t);
		}
	}

	/* Backward (sequential: reduction into answer[i]) */
	for(i = dim - 1; i >= 0; i--)
	{
		for(j = i + 1; j < dim; j++)
		{
			rdd_mul(dtmp1, get_ddmatrix_ij(lu, i, j), get_ddvector_i(answer, j));
			rdd_sub(dtmp, get_ddvector_i(answer, i), dtmp1);
			set_ddvector_i(answer, i, dtmp);
		}
		rdd_div(dtmp, get_ddvector_i(answer, i), get_ddmatrix_ij(lu, i, i));
		set_ddvector_i(answer, i, dtmp);
	}
	return 0;
}

