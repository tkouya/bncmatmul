/********************************************************************************/
/* bncomp_linear_td.c: Parallelized DD Precision Linear Computation Library     */
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
#include "matmul_strassen.h"
#include "bncomp.h"

//---------------------------------------
// TD
//---------------------------------------
//--------
// Vector
//--------
/* c := a */
void _bncomp_subst_tdvector(TDVector c, TDVector a)
{
	long int i;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
//	for(i = 0; i < a->dim; i++)
	#pragma omp parallel for
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//set_tvector_i(c, i, get_tvector_i(a, i));
		_mm256_store_pd(&(c->element[0][i]), _mm256_load_pd(&(a->element[0][i])));
		_mm256_store_pd(&(c->element[1][i]), _mm256_load_pd(&(a->element[1][i])));
		_mm256_store_pd(&(c->element[2][i]), _mm256_load_pd(&(a->element[2][i])));
	}
#elif defined(__AVX512F__) // __AVX512F__
	#pragma omp parallel for
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//set_tvector_i(c, i, get_tvector_i(a, i));
		_mm512_store_pd(&(c->element[0][i]), _mm512_load_pd(&(a->element[0][i])));
		_mm512_store_pd(&(c->element[1][i]), _mm512_load_pd(&(a->element[1][i])));
		_mm512_store_pd(&(c->element[2][i]), _mm512_load_pd(&(a->element[2][i])));
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = a->real_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svst1_f64(_pg, &(c->element[0][_ix]), svld1_f64(_pg, &(a->element[0][_ix])));
			svst1_f64(_pg, &(c->element[1][_ix]), svld1_f64(_pg, &(a->element[1][_ix])));
			svst1_f64(_pg, &(c->element[2][_ix]), svld1_f64(_pg, &(a->element[2][_ix])));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) 
	#pragma omp parallel for
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		vst1q_f64(&(c->element[0][i]), vld1q_f64(&(a->element[0][i])));
		vst1q_f64(&(c->element[1][i]), vld1q_f64(&(a->element[1][i])));
		vst1q_f64(&(c->element[2][i]), vld1q_f64(&(a->element[2][i])));
	}
#else // others
	#pragma omp parallel for
	for(i = 0; i < a->dim; i++)
		set_tdvector_i(c, i, get_tdvector_i(a, i));

#endif // __AVX2__
}

/* c = a + b */
void _bncomp_add_tdvector(TDVector c, TDVector a, TDVector b)
{
	int thread_index;
	long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_tdvector\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256d in_ret[BNCOMP_MAX_NUM_THREADS][TDSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][TDSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][TDSIZE];

 	#pragma omp parallel for private(thread_index)
   for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[thread_index][1] = _mm256_load_pd(&(a->element[1][index]));
        in_a_val[thread_index][2] = _mm256_load_pd(&(a->element[2][index]));
        in_b_val[thread_index][0] = _mm256_load_pd(&(b->element[0][index]));
        in_b_val[thread_index][1] = _mm256_load_pd(&(b->element[1][index]));
        in_b_val[thread_index][2] = _mm256_load_pd(&(b->element[2][index]));

        _bncavx2_rtd_add(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        _mm256_store_pd(&(c->element[0][index]), in_ret[thread_index][0]);
        _mm256_store_pd(&(c->element[1][index]), in_ret[thread_index][1]);
        _mm256_store_pd(&(c->element[2][index]), in_ret[thread_index][2]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512d in_ret[BNCOMP_MAX_NUM_THREADS][TDSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][TDSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	#pragma omp parallel for private(thread_index)
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = _mm512_load_pd(&(a->element[0][index]));
        in_a_val[thread_index][1] = _mm512_load_pd(&(a->element[1][index]));
        in_a_val[thread_index][2] = _mm512_load_pd(&(a->element[2][index]));
        in_b_val[thread_index][0] = _mm512_load_pd(&(b->element[0][index]));
        in_b_val[thread_index][1] = _mm512_load_pd(&(b->element[1][index]));
        in_b_val[thread_index][2] = _mm512_load_pd(&(b->element[2][index]));

        _bncavx512_rtd_add(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        _mm512_store_pd(&(c->element[0][index]), in_ret[thread_index][0]);
        _mm512_store_pd(&(c->element[1][index]), in_ret[thread_index][1]);
        _mm512_store_pd(&(c->element[2][index]), in_ret[thread_index][2]);
   }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = c->real_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svfloat64_t _a0 = svld1_f64(_pg, &(a->element[0][_ix]));
			svfloat64_t _a1 = svld1_f64(_pg, &(a->element[1][_ix]));
			svfloat64_t _a2 = svld1_f64(_pg, &(a->element[2][_ix]));
			svfloat64_t _b0 = svld1_f64(_pg, &(b->element[0][_ix]));
			svfloat64_t _b1 = svld1_f64(_pg, &(b->element[1][_ix]));
			svfloat64_t _b2 = svld1_f64(_pg, &(b->element[2][_ix]));
			svfloat64_t _o0, _o1, _o2;
			_bncsve2_rtd_add(_pg, &_o0, &_o1, &_o2, _a0, _a1, _a2, _b0, _b1, _b2);
			svst1_f64(_pg, &(c->element[0][_ix]), _o0);
			svst1_f64(_pg, &(c->element[1][_ix]), _o1);
			svst1_f64(_pg, &(c->element[2][_ix]), _o2);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) 
    float64x2_t in_ret[BNCOMP_MAX_NUM_THREADS][TDSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][TDSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][TDSIZE];

 	#pragma omp parallel for private(thread_index)
   for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = vld1q_f64(&(a->element[0][index]));
        in_a_val[thread_index][1] = vld1q_f64(&(a->element[1][index]));
        in_a_val[thread_index][2] = vld1q_f64(&(a->element[2][index]));
        in_b_val[thread_index][0] = vld1q_f64(&(b->element[0][index]));
        in_b_val[thread_index][1] = vld1q_f64(&(b->element[1][index]));
        in_b_val[thread_index][2] = vld1q_f64(&(b->element[2][index]));

        _bncneon_rtd_add(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        vst1q_f64(&(c->element[0][index]), in_ret[thread_index][0]);
        vst1q_f64(&(c->element[1][index]), in_ret[thread_index][1]);
        vst1q_f64(&(c->element[2][index]), in_ret[thread_index][2]);
   }
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

		rtd_add(tmp[thread_index], get_tdvector_i(a, i),  get_tdvector_i(b, i));
		set_tdvector_i(c, i, tmp[thread_index]);
	}
#endif // __AVX2__

}

/* c = a - b */
void _bncomp_sub_tdvector(TDVector c, TDVector a, TDVector b)
{
	int thread_index;
	long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: _bncomp_sub_tdvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256d in_ret[BNCOMP_MAX_NUM_THREADS][TDSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][TDSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	#pragma omp parallel for private(thread_index)
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[thread_index][1] = _mm256_load_pd(&(a->element[1][index]));
        in_a_val[thread_index][2] = _mm256_load_pd(&(a->element[2][index]));
        in_b_val[thread_index][0] = _mm256_load_pd(&(b->element[0][index]));
        in_b_val[thread_index][1] = _mm256_load_pd(&(b->element[1][index]));
        in_b_val[thread_index][2] = _mm256_load_pd(&(b->element[2][index]));

        _bncavx2_rtd_sub(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        _mm256_store_pd(&(c->element[0][index]), in_ret[thread_index][0]);
        _mm256_store_pd(&(c->element[1][index]), in_ret[thread_index][1]);
        _mm256_store_pd(&(c->element[2][index]), in_ret[thread_index][2]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512d in_ret[BNCOMP_MAX_NUM_THREADS][TDSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][TDSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = _mm512_load_pd(&(a->element[0][index]));
        in_a_val[thread_index][1] = _mm512_load_pd(&(a->element[1][index]));
        in_a_val[thread_index][2] = _mm512_load_pd(&(a->element[2][index]));
        in_b_val[thread_index][0] = _mm512_load_pd(&(b->element[0][index]));
        in_b_val[thread_index][1] = _mm512_load_pd(&(b->element[1][index]));
        in_b_val[thread_index][2] = _mm512_load_pd(&(b->element[2][index]));

        _bncavx512_rtd_sub(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        _mm512_store_pd(&(c->element[0][index]), in_ret[thread_index][0]);
        _mm512_store_pd(&(c->element[1][index]), in_ret[thread_index][1]);
        _mm512_store_pd(&(c->element[2][index]), in_ret[thread_index][2]);
   }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = c->real_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svfloat64_t _a0 = svld1_f64(_pg, &(a->element[0][_ix]));
			svfloat64_t _a1 = svld1_f64(_pg, &(a->element[1][_ix]));
			svfloat64_t _a2 = svld1_f64(_pg, &(a->element[2][_ix]));
			svfloat64_t _b0 = svld1_f64(_pg, &(b->element[0][_ix]));
			svfloat64_t _b1 = svld1_f64(_pg, &(b->element[1][_ix]));
			svfloat64_t _b2 = svld1_f64(_pg, &(b->element[2][_ix]));
			_b0 = svneg_f64_x(_pg, _b0);
			_b1 = svneg_f64_x(_pg, _b1);
			_b2 = svneg_f64_x(_pg, _b2);
			svfloat64_t _o0, _o1, _o2;
			_bncsve2_rtd_add(_pg, &_o0, &_o1, &_o2, _a0, _a1, _a2, _b0, _b1, _b2);
			svst1_f64(_pg, &(c->element[0][_ix]), _o0);
			svst1_f64(_pg, &(c->element[1][_ix]), _o1);
			svst1_f64(_pg, &(c->element[2][_ix]), _o2);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) 
    float64x2_t in_ret[BNCOMP_MAX_NUM_THREADS][TDSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][TDSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	#pragma omp parallel for private(thread_index)
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = vld1q_f64(&(a->element[0][index]));
        in_a_val[thread_index][1] = vld1q_f64(&(a->element[1][index]));
        in_a_val[thread_index][2] = vld1q_f64(&(a->element[2][index]));
        in_b_val[thread_index][0] = vld1q_f64(&(b->element[0][index]));
        in_b_val[thread_index][1] = vld1q_f64(&(b->element[1][index]));
        in_b_val[thread_index][2] = vld1q_f64(&(b->element[2][index]));

        _bncneon_rtd_sub(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        vst1q_f64(&(c->element[0][index]), in_ret[thread_index][0]);
        vst1q_f64(&(c->element[1][index]), in_ret[thread_index][1]);
        vst1q_f64(&(c->element[2][index]), in_ret[thread_index][2]);
   }
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

		rtd_sub(tmp[thread_index], get_tdvector_i(a, i), get_tdvector_i(b, i));
		set_tdvector_i(c, i, tmp[thread_index]);
	}
#endif // __AVX2__

}

/* c = val * a */
//#ifdef __cplusplus
//void _bncomp_cmul_tdvector(TDVector c, dd_real val, TDVector a)
//#else // __cplusplus
void _bncomp_cmul_tdvector(TDVector c, double val[TDSIZE], TDVector a)
//#endif // __cplusplus
{
	int thread_index;
	long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: cmul_tdvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[BNCOMP_MAX_NUM_THREADS][TDSIZE], c4[BNCOMP_MAX_NUM_THREADS][TDSIZE], val4[TDSIZE];

	val4[0] = _mm256_set1_pd(val[0]);
	val4[1] = _mm256_set1_pd(val[1]);
	val4[2] = _mm256_set1_pd(val[2]);

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		//set_ddvector_i(c, i, val * get_ddvector_i(a, i));
		a4[thread_index][0] = _mm256_load_pd(&(a->element[0][index]));
		a4[thread_index][1] = _mm256_load_pd(&(a->element[1][index]));
		a4[thread_index][2] = _mm256_load_pd(&(a->element[2][index]));

		_bncavx2_rtd_mul(c4[thread_index], val4, a4[thread_index]);
		//_bncavx2_rtd_mul(c4[thread_index], val4[thread_index], a4[thread_index]);

		_mm256_store_pd(&(c->element[0][index]), c4[thread_index][0]);
		_mm256_store_pd(&(c->element[1][index]), c4[thread_index][1]);
		_mm256_store_pd(&(c->element[2][index]), c4[thread_index][2]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[BNCOMP_MAX_NUM_THREADS][TDSIZE], c8[BNCOMP_MAX_NUM_THREADS][TDSIZE], val8[TDSIZE];

	val8[0] = _mm512_set1_pd(val[0]);
	val8[1] = _mm512_set1_pd(val[1]);
	val8[2] = _mm512_set1_pd(val[2]);

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		//set_ddvector_i(c, i, val * get_ddvector_i(a, i));
		a8[thread_index][0] = _mm512_load_pd(&(a->element[0][index]));
		a8[thread_index][1] = _mm512_load_pd(&(a->element[1][index]));
		a8[thread_index][2] = _mm512_load_pd(&(a->element[2][index]));

		_bncavx512_rtd_mul(c8[thread_index], val8, a8[thread_index]);

		_mm512_store_pd(&(c->element[0][index]), c8[thread_index][0]);
		_mm512_store_pd(&(c->element[1][index]), c8[thread_index][1]);
		_mm512_store_pd(&(c->element[2][index]), c8[thread_index][2]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = a->real_dim; long _ix;
		svfloat64_t _v0 = svdup_n_f64(val[0]);
		svfloat64_t _v1 = svdup_n_f64(val[1]);
		svfloat64_t _v2 = svdup_n_f64(val[2]);
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svfloat64_t _a0 = svld1_f64(_pg, &(a->element[0][_ix]));
			svfloat64_t _a1 = svld1_f64(_pg, &(a->element[1][_ix]));
			svfloat64_t _a2 = svld1_f64(_pg, &(a->element[2][_ix]));
			svfloat64_t _o0, _o1, _o2;
			_bncsve2_rtd_mul(_pg, &_o0, &_o1, &_o2, _v0, _v1, _v2, _a0, _a1, _a2);
			svst1_f64(_pg, &(c->element[0][_ix]), _o0);
			svst1_f64(_pg, &(c->element[1][_ix]), _o1);
			svst1_f64(_pg, &(c->element[2][_ix]), _o2);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) 
	float64x2_t a2[BNCOMP_MAX_NUM_THREADS][TDSIZE], c2[BNCOMP_MAX_NUM_THREADS][TDSIZE], val2[TDSIZE];

	val2[0] = vdupq_n_f64(val[0]);
	val2[1] = vdupq_n_f64(val[1]);
	val2[2] = vdupq_n_f64(val[2]);

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a2[thread_index][0] = vld1q_f64(&(a->element[0][index]));
		a2[thread_index][1] = vld1q_f64(&(a->element[1][index]));
		a2[thread_index][2] = vld1q_f64(&(a->element[2][index]));

		_bncneon_rtd_mul(c2[thread_index], val2, a2[thread_index]);

		vst1q_f64(&(c->element[0][index]), c2[thread_index][0]);
		vst1q_f64(&(c->element[1][index]), c2[thread_index][1]);
		vst1q_f64(&(c->element[2][index]), c2[thread_index][2]);
	}
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

		rtd_mul(tmp[thread_index], val, get_tdvector_i(a, i));
		set_tdvector_i(c, i, tmp[thread_index]);
	}
#endif // __AVX2__
}

/* (a, b) */
void _bncomp_ip_tdvector(double ret[TDSIZE], TDVector a, TDVector b)
{
	int thread_index;
	long int i, index;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: _bncomp_ip_tdvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[BNCOMP_MAX_NUM_THREADS][TDSIZE], b4[BNCOMP_MAX_NUM_THREADS][TDSIZE], ret4[TDSIZE], tmp4[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	_bncavx2_set0_td(ret4);
	
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a4[thread_index][0] = _mm256_load_pd(&(a->element[0][index]));
		a4[thread_index][1] = _mm256_load_pd(&(a->element[1][index]));
		a4[thread_index][2] = _mm256_load_pd(&(a->element[2][index]));
		b4[thread_index][0] = _mm256_load_pd(&(b->element[0][index]));
		b4[thread_index][1] = _mm256_load_pd(&(b->element[1][index]));
		b4[thread_index][2] = _mm256_load_pd(&(b->element[2][index]));

		_bncavx2_rtd_mul(tmp4[thread_index], a4[thread_index], b4[thread_index]);
		#pragma omp critical
			_bncavx2_rtd_add(ret4, ret4, tmp4[thread_index]);
	}

	_bncavx2_rtd_sum256d(ret, ret4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[BNCOMP_MAX_NUM_THREADS][TDSIZE], b8[BNCOMP_MAX_NUM_THREADS][TDSIZE], ret8[TDSIZE], tmp8[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	_bncavx512_set0_td(ret8);
	
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a8[thread_index][0] = _mm512_load_pd(&(a->element[0][index]));
		a8[thread_index][1] = _mm512_load_pd(&(a->element[1][index]));
		a8[thread_index][2] = _mm512_load_pd(&(a->element[2][index]));
		b8[thread_index][0] = _mm512_load_pd(&(b->element[0][index]));
		b8[thread_index][1] = _mm512_load_pd(&(b->element[1][index]));
		b8[thread_index][2] = _mm512_load_pd(&(b->element[2][index]));

		_bncavx512_rtd_mul(tmp8[thread_index], a8[thread_index], b8[thread_index]);
		#pragma omp critical
			_bncavx512_rtd_add(ret8, ret8, tmp8[thread_index]);
	}
	_bncavx512_rtd_sum512d(ret, ret8);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = a->real_dim; long _ix;
		svfloat64_t _acc0 = svdup_n_f64(0.0);
		svfloat64_t _acc1 = svdup_n_f64(0.0);
		svfloat64_t _acc2 = svdup_n_f64(0.0);
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svfloat64_t _a0 = svld1_f64(_pg, &(a->element[0][_ix]));
			svfloat64_t _a1 = svld1_f64(_pg, &(a->element[1][_ix]));
			svfloat64_t _a2 = svld1_f64(_pg, &(a->element[2][_ix]));
			svfloat64_t _b0 = svld1_f64(_pg, &(b->element[0][_ix]));
			svfloat64_t _b1 = svld1_f64(_pg, &(b->element[1][_ix]));
			svfloat64_t _b2 = svld1_f64(_pg, &(b->element[2][_ix]));
			svfloat64_t _m0, _m1, _m2;
			_bncsve2_rtd_mul(_pg, &_m0, &_m1, &_m2, _a0, _a1, _a2, _b0, _b1, _b2);
			_bncsve2_rtd_add(_pg, &_acc0, &_acc1, &_acc2, _acc0, _acc1, _acc2, _m0, _m1, _m2);
		}
		{
			long _L, _vl = (long)svcntd();
			double _la0[64], _la1[64], _la2[64];
			svst1_f64(svptrue_b64(), _la0, _acc0);
			svst1_f64(svptrue_b64(), _la1, _acc1);
			svst1_f64(svptrue_b64(), _la2, _acc2);
			rtd_set_ui(ret, 0UL);
			for(_L = 0; _L < _vl; _L++)
			{
				double _lane[TDSIZE] = { _la0[_L], _la1[_L], _la2[_L] };
				rtd_add(ret, ret, _lane);
			}
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) 
	float64x2_t a2[BNCOMP_MAX_NUM_THREADS][TDSIZE], b2[BNCOMP_MAX_NUM_THREADS][TDSIZE], ret2[TDSIZE], tmp2[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	_bncneon_set0_td(ret2);
	
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a2[thread_index][0] = vld1q_f64(&(a->element[0][index]));
		a2[thread_index][1] = vld1q_f64(&(a->element[1][index]));
		a2[thread_index][2] = vld1q_f64(&(a->element[2][index]));
		b2[thread_index][0] = vld1q_f64(&(b->element[0][index]));
		b2[thread_index][1] = vld1q_f64(&(b->element[1][index]));
		b2[thread_index][2] = vld1q_f64(&(b->element[2][index]));

		_bncneon_rtd_mul(tmp2[thread_index], a2[thread_index], b2[thread_index]);
		#pragma omp critical
			_bncneon_rtd_add(ret2, ret2, tmp2[thread_index]);
	}

	_bncneon_rtd_sum128d(ret, ret2);

#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	set0_td(ret);
	#pragma omp parallel for private(thread_index)
	for(i = 0; i < a->dim; i++)
	{
		thread_index = omp_get_thread_num();

		rtd_mul(tmp[thread_index], get_tdvector_i(a, i), get_tdvector_i(b, i));

#pragma omp critical
		rtd_add(ret, ret, tmp[thread_index]);

	}
#endif // __AVX2__

	return;
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void _bncomp_row_swap_tdmatrix(TDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
	long int i, j, true_end;
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE];
	int thread_index;

	true_end = (col_end > mat->col_dim) ? mat->col_dim : col_end;

	#pragma omp parallel for private(thread_index)
	for(i = col_start; i < true_end; i++)
	{
		thread_index = omp_get_thread_num();

		rtd_set(tmp[thread_index], get_tdmatrix_ij(mat, row_index0, i));
		set_tdmatrix_ij(mat, row_index0, i, get_tdmatrix_ij(mat, row_index1, i));
		set_tdmatrix_ij(mat, row_index1, i, tmp[thread_index]);
	}
}

//--------
// Matrix
//--------
/* c := a + b */
void _bncomp_add_tdmatrix(TDMatrix c, TDMatrix a, TDMatrix b)
{
	int thread_index;
	long int index, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_tdmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_tdmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[BNCOMP_MAX_NUM_THREADS][TDSIZE], aij4[BNCOMP_MAX_NUM_THREADS][TDSIZE], bij4[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij4[thread_index][0] = _mm256_load_pd(&(a->element[0][index]));
		aij4[thread_index][1] = _mm256_load_pd(&(a->element[1][index]));
		aij4[thread_index][2] = _mm256_load_pd(&(a->element[2][index]));
		bij4[thread_index][0] = _mm256_load_pd(&(b->element[0][index]));
		bij4[thread_index][1] = _mm256_load_pd(&(b->element[1][index]));
		bij4[thread_index][2] = _mm256_load_pd(&(b->element[2][index]));

		_bncavx2_rtd_add(tmp4[thread_index], aij4[thread_index], bij4[thread_index]);

		_mm256_store_pd(&(c->element[0][index]), tmp4[thread_index][0]);
		_mm256_store_pd(&(c->element[1][index]), tmp4[thread_index][1]); 
		_mm256_store_pd(&(c->element[2][index]), tmp4[thread_index][2]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[BNCOMP_MAX_NUM_THREADS][TDSIZE], aij8[BNCOMP_MAX_NUM_THREADS][TDSIZE], bij8[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij8[thread_index][0] = _mm512_load_pd(&(a->element[0][index]));
		aij8[thread_index][1] = _mm512_load_pd(&(a->element[1][index]));
		aij8[thread_index][2] = _mm512_load_pd(&(a->element[2][index]));
		bij8[thread_index][0] = _mm512_load_pd(&(b->element[0][index]));
		bij8[thread_index][1] = _mm512_load_pd(&(b->element[1][index]));
		bij8[thread_index][2] = _mm512_load_pd(&(b->element[2][index]));

		_bncavx512_rtd_add(tmp8[thread_index], aij8[thread_index], bij8[thread_index]);

		_mm512_store_pd(&(c->element[0][index]), tmp8[thread_index][0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[thread_index][1]); 
		_mm512_store_pd(&(c->element[2][index]), tmp8[thread_index][2]); 
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = c->real_row_dim * c->real_col_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svfloat64_t _a0 = svld1_f64(_pg, &(a->element[0][_ix]));
			svfloat64_t _a1 = svld1_f64(_pg, &(a->element[1][_ix]));
			svfloat64_t _a2 = svld1_f64(_pg, &(a->element[2][_ix]));
			svfloat64_t _b0 = svld1_f64(_pg, &(b->element[0][_ix]));
			svfloat64_t _b1 = svld1_f64(_pg, &(b->element[1][_ix]));
			svfloat64_t _b2 = svld1_f64(_pg, &(b->element[2][_ix]));
			svfloat64_t _o0, _o1, _o2;
			_bncsve2_rtd_add(_pg, &_o0, &_o1, &_o2, _a0, _a1, _a2, _b0, _b1, _b2);
			svst1_f64(_pg, &(c->element[0][_ix]), _o0);
			svst1_f64(_pg, &(c->element[1][_ix]), _o1);
			svst1_f64(_pg, &(c->element[2][_ix]), _o2);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) 
	float64x2_t tmp2[BNCOMP_MAX_NUM_THREADS][TDSIZE], aij2[BNCOMP_MAX_NUM_THREADS][TDSIZE], bij2[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij2[thread_index][0] = vld1q_f64(&(a->element[0][index]));
		aij2[thread_index][1] = vld1q_f64(&(a->element[1][index]));
		aij2[thread_index][2] = vld1q_f64(&(a->element[2][index]));
		bij2[thread_index][0] = vld1q_f64(&(b->element[0][index]));
		bij2[thread_index][1] = vld1q_f64(&(b->element[1][index]));
		bij2[thread_index][2] = vld1q_f64(&(b->element[2][index]));

		_bncneon_rtd_add(tmp2[thread_index], aij2[thread_index], bij2[thread_index]);

		vst1q_f64(&(c->element[0][index]), tmp2[thread_index][0]);
		vst1q_f64(&(c->element[1][index]), tmp2[thread_index][1]); 
		vst1q_f64(&(c->element[2][index]), tmp2[thread_index][2]); 
	}
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE], aij[BNCOMP_MAX_NUM_THREADS][TDSIZE], bij[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index++)
	{
		thread_index = omp_get_thread_num();
		
		aij[thread_index][0] = a->element[0][index];
		aij[thread_index][1] = a->element[1][index];
		aij[thread_index][2] = a->element[2][index];
		bij[thread_index][0] = b->element[0][index];
		bij[thread_index][1] = b->element[1][index];
		bij[thread_index][2] = b->element[2][index];

		rtd_add(tmp[thread_index], aij[thread_index], bij[thread_index]);

		c->element[0][index] = tmp[thread_index][0];
		c->element[1][index] = tmp[thread_index][1]; 
		c->element[2][index] = tmp[thread_index][2]; 
	}
#endif // __AVX2__
}

/* c := a - b */
void _bncomp_sub_tdmatrix(TDMatrix c, TDMatrix a, TDMatrix b)
{
	int thread_index;
	long int index, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_sub_tdmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_sub_tdmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[BNCOMP_MAX_NUM_THREADS][TDSIZE], aij4[BNCOMP_MAX_NUM_THREADS][TDSIZE], bij4[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij4[thread_index][0] = _mm256_load_pd(&(a->element[0][index]));
		aij4[thread_index][1] = _mm256_load_pd(&(a->element[1][index]));
		aij4[thread_index][2] = _mm256_load_pd(&(a->element[2][index]));
		bij4[thread_index][0] = _mm256_load_pd(&(b->element[0][index]));
		bij4[thread_index][1] = _mm256_load_pd(&(b->element[1][index]));
		bij4[thread_index][2] = _mm256_load_pd(&(b->element[2][index]));

		_bncavx2_rtd_sub(tmp4[thread_index], aij4[thread_index], bij4[thread_index]);

		_mm256_store_pd(&(c->element[0][index]), tmp4[thread_index][0]);
		_mm256_store_pd(&(c->element[1][index]), tmp4[thread_index][1]); 
		_mm256_store_pd(&(c->element[2][index]), tmp4[thread_index][2]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[BNCOMP_MAX_NUM_THREADS][TDSIZE], aij8[BNCOMP_MAX_NUM_THREADS][TDSIZE], bij8[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij8[thread_index][0] = _mm512_load_pd(&(a->element[0][index]));
		aij8[thread_index][1] = _mm512_load_pd(&(a->element[1][index]));
		aij8[thread_index][2] = _mm512_load_pd(&(a->element[2][index]));
		bij8[thread_index][0] = _mm512_load_pd(&(b->element[0][index]));
		bij8[thread_index][1] = _mm512_load_pd(&(b->element[1][index]));
		bij8[thread_index][2] = _mm512_load_pd(&(b->element[2][index]));

		_bncavx512_rtd_sub(tmp8[thread_index], aij8[thread_index], bij8[thread_index]);

		_mm512_store_pd(&(c->element[0][index]), tmp8[thread_index][0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[thread_index][1]); 
		_mm512_store_pd(&(c->element[2][index]), tmp8[thread_index][2]); 
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = c->real_row_dim * c->real_col_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svfloat64_t _a0 = svld1_f64(_pg, &(a->element[0][_ix]));
			svfloat64_t _a1 = svld1_f64(_pg, &(a->element[1][_ix]));
			svfloat64_t _a2 = svld1_f64(_pg, &(a->element[2][_ix]));
			svfloat64_t _b0 = svld1_f64(_pg, &(b->element[0][_ix]));
			svfloat64_t _b1 = svld1_f64(_pg, &(b->element[1][_ix]));
			svfloat64_t _b2 = svld1_f64(_pg, &(b->element[2][_ix]));
			_b0 = svneg_f64_x(_pg, _b0);
			_b1 = svneg_f64_x(_pg, _b1);
			_b2 = svneg_f64_x(_pg, _b2);
			svfloat64_t _o0, _o1, _o2;
			_bncsve2_rtd_add(_pg, &_o0, &_o1, &_o2, _a0, _a1, _a2, _b0, _b1, _b2);
			svst1_f64(_pg, &(c->element[0][_ix]), _o0);
			svst1_f64(_pg, &(c->element[1][_ix]), _o1);
			svst1_f64(_pg, &(c->element[2][_ix]), _o2);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) 
	float64x2_t tmp2[BNCOMP_MAX_NUM_THREADS][TDSIZE], aij2[BNCOMP_MAX_NUM_THREADS][TDSIZE], bij2[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij2[thread_index][0] = vld1q_f64(&(a->element[0][index]));
		aij2[thread_index][1] = vld1q_f64(&(a->element[1][index]));
		aij2[thread_index][2] = vld1q_f64(&(a->element[2][index]));
		bij2[thread_index][0] = vld1q_f64(&(b->element[0][index]));
		bij2[thread_index][1] = vld1q_f64(&(b->element[1][index]));
		bij2[thread_index][2] = vld1q_f64(&(b->element[2][index]));

		_bncneon_rtd_sub(tmp2[thread_index], aij2[thread_index], bij2[thread_index]);

		vst1q_f64(&(c->element[0][index]), tmp2[thread_index][0]);
		vst1q_f64(&(c->element[1][index]), tmp2[thread_index][1]); 
		vst1q_f64(&(c->element[2][index]), tmp2[thread_index][2]); 
	}
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE], aij[BNCOMP_MAX_NUM_THREADS][TDSIZE], bij[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index++)
	{
		thread_index = omp_get_thread_num();
		
		aij[thread_index][0] = a->element[0][index];
		aij[thread_index][1] = a->element[1][index];
		aij[thread_index][2] = a->element[2][index];
		bij[thread_index][0] = b->element[0][index];
		bij[thread_index][1] = b->element[1][index];
		bij[thread_index][2] = b->element[2][index];

		rtd_sub(tmp[thread_index], aij[thread_index], bij[thread_index]);

		c->element[0][index] = tmp[thread_index][0];
		c->element[1][index] = tmp[thread_index][1]; 
		c->element[2][index] = tmp[thread_index][2]; 
	}
#endif // __AVX2__
}

/* c := sc * a */
void _bncomp_cmul_tdmatrix(TDMatrix c, double sc[TDSIZE], TDMatrix a)
{
	long int i, j, index, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim;
	int thread_index;

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_tdmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_tdmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[BNCOMP_MAX_NUM_THREADS][TDSIZE], aij4[BNCOMP_MAX_NUM_THREADS][TDSIZE], sc4[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		sc4[thread_index][0] = _mm256_set1_pd(sc[0]);
		sc4[thread_index][1] = _mm256_set1_pd(sc[1]);
		sc4[thread_index][2] = _mm256_set1_pd(sc[2]);
	}
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij4[thread_index][0] = _mm256_load_pd(&(a->element[0][index]));
		aij4[thread_index][1] = _mm256_load_pd(&(a->element[1][index]));
		aij4[thread_index][2] = _mm256_load_pd(&(a->element[2][index]));

		_bncavx2_rtd_mul(tmp4[thread_index], sc4[thread_index], aij4[thread_index]);

		_mm256_store_pd(&(c->element[0][index]), tmp4[thread_index][0]);
		_mm256_store_pd(&(c->element[1][index]), tmp4[thread_index][1]); 
		_mm256_store_pd(&(c->element[2][index]), tmp4[thread_index][2]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[BNCOMP_MAX_NUM_THREADS][TDSIZE], aij8[BNCOMP_MAX_NUM_THREADS][TDSIZE], sc8[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		sc8[thread_index][0] = _mm512_set1_pd(sc[0]);
		sc8[thread_index][1] = _mm512_set1_pd(sc[1]);
		sc8[thread_index][2] = _mm512_set1_pd(sc[2]);
	}
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij8[thread_index][0] = _mm512_load_pd(&(a->element[0][index]));
		aij8[thread_index][1] = _mm512_load_pd(&(a->element[1][index]));
		aij8[thread_index][2] = _mm512_load_pd(&(a->element[2][index]));

		_bncavx512_rtd_mul(tmp8[thread_index], sc8[thread_index], aij8[thread_index]);

		_mm512_store_pd(&(c->element[0][index]), tmp8[thread_index][0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[thread_index][1]); 
		_mm512_store_pd(&(c->element[2][index]), tmp8[thread_index][2]); 
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = a->real_row_dim * a->real_col_dim; long _ix;
		svfloat64_t _v0 = svdup_n_f64(sc[0]);
		svfloat64_t _v1 = svdup_n_f64(sc[1]);
		svfloat64_t _v2 = svdup_n_f64(sc[2]);
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svfloat64_t _a0 = svld1_f64(_pg, &(a->element[0][_ix]));
			svfloat64_t _a1 = svld1_f64(_pg, &(a->element[1][_ix]));
			svfloat64_t _a2 = svld1_f64(_pg, &(a->element[2][_ix]));
			svfloat64_t _o0, _o1, _o2;
			_bncsve2_rtd_mul(_pg, &_o0, &_o1, &_o2, _v0, _v1, _v2, _a0, _a1, _a2);
			svst1_f64(_pg, &(c->element[0][_ix]), _o0);
			svst1_f64(_pg, &(c->element[1][_ix]), _o1);
			svst1_f64(_pg, &(c->element[2][_ix]), _o2);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) 
	float64x2_t tmp2[BNCOMP_MAX_NUM_THREADS][TDSIZE], aij2[BNCOMP_MAX_NUM_THREADS][TDSIZE], sc2[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		sc2[thread_index][0] = vdupq_n_f64(sc[0]);
		sc2[thread_index][1] = vdupq_n_f64(sc[1]);
		sc2[thread_index][2] = vdupq_n_f64(sc[2]);
	}
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij2[thread_index][0] = vld1q_f64(&(a->element[0][index]));
		aij2[thread_index][1] = vld1q_f64(&(a->element[1][index]));
		aij2[thread_index][2] = vld1q_f64(&(a->element[2][index]));

		_bncneon_rtd_mul(tmp2[thread_index], sc2[thread_index], aij2[thread_index]);

		vst1q_f64(&(c->element[0][index]), tmp2[thread_index][0]);
		vst1q_f64(&(c->element[1][index]), tmp2[thread_index][1]); 
		vst1q_f64(&(c->element[2][index]), tmp2[thread_index][2]); 
	}
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE], aij[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index++)
	{
		thread_index = omp_get_thread_num();
		
		aij[thread_index][0] = a->element[0][index];
		aij[thread_index][1] = a->element[1][index];
		aij[thread_index][2] = a->element[2][index];

		rtd_mul(tmp[thread_index], sc, aij[thread_index]);

		c->element[0][index] = tmp[thread_index][0];
		c->element[1][index] = tmp[thread_index][1]; 
		c->element[2][index] = tmp[thread_index][2]; 
	}
#endif // __AVX2__
}

/* c = a * b */
void _bncomp_mul_tdmatrix(TDMatrix ret, TDMatrix a, TDMatrix b)
{
	int thread_num, thread_index;
	long int i, j, k;
	long row_dim, col_dim, mid_dim;

	/* dimension check */
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_tdmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret->row_dim, ret->col_dim, a->row_dim, a->col_dim, b->row_dim, b->col_dim);
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
#ifdef BNC_USE_GCC
	long real_row_dim, real_col_dim, real_mid_dim;
	double cijval[BNCOMP_MAX_NUM_THREADS][4][TDSIZE];
    __m256d cij[BNCOMP_MAX_NUM_THREADS][TDSIZE], aik[BNCOMP_MAX_NUM_THREADS][TDSIZE], bkj[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp_mul[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		//set0_dd(tmp[thread_index]);
	}

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j, k, aik, bkj, cij, cijval, tmp_mul)
    for(i = 0; i < real_row_dim; i++)
    {
		thread_index = omp_get_thread_num();

        for(j = 0; j < real_col_dim; j++)
        {
            cij[thread_index][0] = _mm256_setzero_pd();
            cij[thread_index][1] = _mm256_setzero_pd();
            cij[thread_index][2] = _mm256_setzero_pd();

            for(k = 0; k < real_mid_dim; k += _BNC_D_WIDTH)
            {
                aik[thread_index][0] = _mm256_load_pd(&(a->element[0][i * real_mid_dim + k]));
                aik[thread_index][1] = _mm256_load_pd(&(a->element[1][i * real_mid_dim + k]));
                aik[thread_index][2] = _mm256_load_pd(&(a->element[2][i * real_mid_dim + k]));

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
				bkj[thread_index][2] = _mm256_set_pd(
                    b->element[2][(k + 3) * real_col_dim + j],
                    b->element[2][(k + 2) * real_col_dim + j],
                    b->element[2][(k + 1) * real_col_dim + j],
                    b->element[2][(k    ) * real_col_dim + j]
                );

                _bncavx2_rtd_mul(tmp_mul[thread_index], aik[thread_index], bkj[thread_index]);
                _bncavx2_rtd_add(cij[thread_index], cij[thread_index], tmp_mul[thread_index]);
            }

            cijval[thread_index][0][0] = cij[thread_index][0][0];
			cijval[thread_index][0][1] = cij[thread_index][1][0];
			cijval[thread_index][0][2] = cij[thread_index][2][0];

            cijval[thread_index][1][0] = cij[thread_index][0][1];
			cijval[thread_index][1][1] = cij[thread_index][1][1];
			cijval[thread_index][1][2] = cij[thread_index][2][1];

            cijval[thread_index][2][0] = cij[thread_index][0][2];
			cijval[thread_index][2][1] = cij[thread_index][1][2];
			cijval[thread_index][2][2] = cij[thread_index][2][2];

            cijval[thread_index][3][0] = cij[thread_index][0][3];
			cijval[thread_index][3][1] = cij[thread_index][1][3];
			cijval[thread_index][3][2] = cij[thread_index][2][3];

            rtd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][1]);
            rtd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][2]);
            rtd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][3]);

            ret->element[0][i * real_col_dim + j] = cijval[thread_index][0][0];
            ret->element[1][i * real_col_dim + j] = cijval[thread_index][0][1];
            ret->element[2][i * real_col_dim + j] = cijval[thread_index][0][2];
        }
    }
#else // BNC_USE_GCC
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE], ret_ij[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_td(tmp[thread_index]);
	}

	//printf("Non SIMD mul_tdmatrix(%ld, %ld)\n", ret->row_dim, ret->col_dim);
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = a->col_dim;

	#pragma omp parallel for private(thread_index, j, k, ret_ij, tmp)
	for(i = 0; i < row_dim; i++)
	{

		thread_index = omp_get_thread_num();

		//#pragma omp parallel for private(thread_index, k)
		for(j = 0; j < col_dim; j++)
		{
			//thread_index = omp_get_thread_num();
			//#pragma omp critical
			rtd_set0(ret_ij[thread_index]);
			for(k = 0; k < mid_dim; k++)
			{
				rtd_mul(tmp[thread_index], get_tdmatrix_ij(a, i, k), get_tdmatrix_ij(b, k, j));
				#pragma omp critical
				rtd_add(ret_ij[thread_index], tmp[thread_index], ret_ij[thread_index]);
			}
			
			set_tdmatrix_ij(ret, i, j, ret_ij[thread_index]);
		}
	}	//printf("end(%ld, %ld)\n", ret->row_dim, ret->col_dim);
#endif // BNC_USE_GCC
#elif defined(__AVX512F__) // __AVX512F__
	long real_row_dim, real_col_dim, real_mid_dim;
	double cijval[BNCOMP_MAX_NUM_THREADS][8][TDSIZE];
    __m512d cij[BNCOMP_MAX_NUM_THREADS][TDSIZE], aik[BNCOMP_MAX_NUM_THREADS][TDSIZE], bkj[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp_mul[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j, k, aik, bkj, cij, cijval, tmp_mul)
    for(i = 0; i < real_row_dim; i++)
    {
		thread_index = omp_get_thread_num();

		for(j = 0; j < real_col_dim; j++)
        {
            cij[thread_index][0] = _mm512_setzero_pd();
            cij[thread_index][1] = _mm512_setzero_pd();
            cij[thread_index][2] = _mm512_setzero_pd();

            for(k = 0; k < real_mid_dim; k += _BNC_D_WIDTH)
            {
                aik[thread_index][0] = _mm512_load_pd(&(a->element[0][i * real_mid_dim + k]));
                aik[thread_index][1] = _mm512_load_pd(&(a->element[1][i * real_mid_dim + k]));
                aik[thread_index][2] = _mm512_load_pd(&(a->element[2][i * real_mid_dim + k]));

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
                bkj[thread_index][2] = _mm512_set_pd(
                    b->element[2][(k + 7) * real_col_dim + j],
                    b->element[2][(k + 6) * real_col_dim + j],
                    b->element[2][(k + 5) * real_col_dim + j],
                    b->element[2][(k + 4) * real_col_dim + j],
                    b->element[2][(k + 3) * real_col_dim + j],
                    b->element[2][(k + 2) * real_col_dim + j],
                    b->element[2][(k + 1) * real_col_dim + j],
                    b->element[2][(k    ) * real_col_dim + j]
                );

                _bncavx512_rtd_mul(tmp_mul[thread_index], aik[thread_index], bkj[thread_index]);
                _bncavx512_rtd_add(cij[thread_index], cij[thread_index], tmp_mul[thread_index]);
            }

            cijval[thread_index][0][0] = cij[thread_index][0][0];
			cijval[thread_index][0][1] = cij[thread_index][1][0];
			cijval[thread_index][0][2] = cij[thread_index][2][0];

            cijval[thread_index][1][0] = cij[thread_index][0][1];
			cijval[thread_index][1][1] = cij[thread_index][1][1];
			cijval[thread_index][1][2] = cij[thread_index][2][1];

            cijval[thread_index][2][0] = cij[thread_index][0][2];
			cijval[thread_index][2][1] = cij[thread_index][1][2];
			cijval[thread_index][2][2] = cij[thread_index][2][2];

            cijval[thread_index][3][0] = cij[thread_index][0][3];
			cijval[thread_index][3][1] = cij[thread_index][1][3];
			cijval[thread_index][3][2] = cij[thread_index][2][3];

            cijval[thread_index][4][0] = cij[thread_index][0][4];
			cijval[thread_index][4][1] = cij[thread_index][1][4];
			cijval[thread_index][4][2] = cij[thread_index][2][4];

            cijval[thread_index][5][0] = cij[thread_index][0][5];
			cijval[thread_index][5][1] = cij[thread_index][1][5];
			cijval[thread_index][5][2] = cij[thread_index][2][5];

            cijval[thread_index][6][0] = cij[thread_index][0][6];
			cijval[thread_index][6][1] = cij[thread_index][1][6];
			cijval[thread_index][6][2] = cij[thread_index][2][6];

            cijval[thread_index][7][0] = cij[thread_index][0][7];
			cijval[thread_index][7][1] = cij[thread_index][1][7];
			cijval[thread_index][7][2] = cij[thread_index][2][7];

            rtd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][1]);
            rtd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][2]);
            rtd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][3]);
            rtd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][4]);
            rtd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][5]);
            rtd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][6]);
            rtd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][7]);

            ret->element[0][i * real_col_dim + j] = cijval[thread_index][0][0];
            ret->element[1][i * real_col_dim + j] = cijval[thread_index][0][1];
            ret->element[2][i * real_col_dim + j] = cijval[thread_index][0][2];
        }
    }

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic) + OpenMP
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
	#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	long real_row_dim, real_col_dim, real_mid_dim;
	double cijval[BNCOMP_MAX_NUM_THREADS][2][TDSIZE], bkj_tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE][2];
    float64x2_t cij[BNCOMP_MAX_NUM_THREADS][TDSIZE], aik[BNCOMP_MAX_NUM_THREADS][TDSIZE], bkj[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp_mul[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, i, j, k, aik, bkj, cij, cijval, tmp_mul)
    for(i = 0; i < real_row_dim; i++)
    {
		thread_index = omp_get_thread_num();

        for(j = 0; j < real_col_dim; j++)
        {
            cij[thread_index][0] = vdupq_n_f64(0.0);
            cij[thread_index][1] = vdupq_n_f64(0.0);
            cij[thread_index][2] = vdupq_n_f64(0.0);

			for(k = 0; k < real_mid_dim; k += _BNC_D_WIDTH)
            {
                aik[thread_index][0] = vld1q_f64(&(a->element[0][i * real_mid_dim + k]));
                aik[thread_index][1] = vld1q_f64(&(a->element[1][i * real_mid_dim + k]));
                aik[thread_index][2] = vld1q_f64(&(a->element[2][i * real_mid_dim + k]));

                // Neonは2要素なので個別にセット
                bkj_tmp[thread_index][0][0] = b->element[0][(k    ) * real_col_dim + j];
                bkj_tmp[thread_index][0][1] = b->element[0][(k + 1) * real_col_dim + j];

	            bkj_tmp[thread_index][1][0] = b->element[1][(k    ) * real_col_dim + j];
                bkj_tmp[thread_index][1][1] = b->element[1][(k + 1) * real_col_dim + j];

				bkj_tmp[thread_index][2][0] = b->element[2][(k    ) * real_col_dim + j];
                bkj_tmp[thread_index][2][1] = b->element[2][(k + 1) * real_col_dim + j];

                bkj[thread_index][0] = vld1q_f64(bkj_tmp[thread_index][0]);
                bkj[thread_index][1] = vld1q_f64(bkj_tmp[thread_index][1]);
                bkj[thread_index][2] = vld1q_f64(bkj_tmp[thread_index][2]);

				_bncneon_rtd_mul(tmp_mul[thread_index], aik[thread_index], bkj[thread_index]);
				//#pragma omp critical
                	_bncneon_rtd_add(cij[thread_index], cij[thread_index], tmp_mul[thread_index]);
            }

            cijval[thread_index][0][0] = vgetq_lane_f64(cij[thread_index][0], 0);
            cijval[thread_index][0][1] = vgetq_lane_f64(cij[thread_index][1], 0);
            cijval[thread_index][0][2] = vgetq_lane_f64(cij[thread_index][2], 0);

			cijval[thread_index][1][0] = vgetq_lane_f64(cij[thread_index][0], 1);
            cijval[thread_index][1][1] = vgetq_lane_f64(cij[thread_index][1], 1);
            cijval[thread_index][1][2] = vgetq_lane_f64(cij[thread_index][2], 1);

			/* BUG FIX: was rdd_add (DD = 2 limbs) — silently dropped the 3rd TD limb
			 * of lane 1, collapsing the result to ~DD precision (~1e-31 error).
			 * rtd_add correctly sums all 3 TD limbs across the 2 NEON lanes. */
			rtd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][1]);

            ret->element[0][i * real_col_dim + j] = cijval[thread_index][0][0];
            ret->element[1][i * real_col_dim + j] = cijval[thread_index][0][1];
            ret->element[2][i * real_col_dim + j] = cijval[thread_index][0][2];
        }
    }
#else // __AVX2__
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE], ret_ij[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_td(tmp[thread_index]);
	}

	//printf("Non SIMD mul_tdmatrix(%ld, %ld)\n", ret->row_dim, ret->col_dim);
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = a->col_dim;

	#pragma omp parallel for private(thread_index, i, j, k, ret_ij, tmp)
	for(i = 0; i < row_dim; i++)
	{

		thread_index = omp_get_thread_num();

		//#pragma omp parallel for private(thread_index, k)
		for(j = 0; j < col_dim; j++)
		{
			//thread_index = omp_get_thread_num();
			//#pragma omp critical
			rtd_set0(ret_ij[thread_index]);
			for(k = 0; k < mid_dim; k++)
			{
				rtd_mul(tmp[thread_index], get_tdmatrix_ij(a, i, k), get_tdmatrix_ij(b, k, j));
				#pragma omp critical
				rtd_add(ret_ij[thread_index], tmp[thread_index], ret_ij[thread_index]);
			}
			
			set_tdmatrix_ij(ret, i, j, ret_ij[thread_index]);
		}
	}	//printf("end(%ld, %ld)\n", ret->row_dim, ret->col_dim);
#endif // __AVX2__

}

/* c := a */
void _bncomp_subst_tdmatrix(TDMatrix c, TDMatrix a)
{
	long int index;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_tdmatrix\n");
		return;
	}

// AVX2
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int real_total_dim;

	real_total_dim = c->real_row_dim * c->real_col_dim;

	#pragma omp parallel for
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&(c->element[0][index]), _mm256_load_pd(&(a->element[0][index])));
		_mm256_store_pd(&(c->element[1][index]), _mm256_load_pd(&(a->element[1][index])));
		_mm256_store_pd(&(c->element[2][index]), _mm256_load_pd(&(a->element[2][index])));
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int real_total_dim;

	real_total_dim = c->real_row_dim * c->real_col_dim;

	#pragma omp parallel for
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&(c->element[0][index]), _mm512_load_pd(&(a->element[0][index])));
		_mm512_store_pd(&(c->element[1][index]), _mm512_load_pd(&(a->element[1][index])));
		_mm512_store_pd(&(c->element[2][index]), _mm512_load_pd(&(a->element[2][index])));
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = c->real_row_dim * c->real_col_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svst1_f64(_pg, &(c->element[0][_ix]), svld1_f64(_pg, &(a->element[0][_ix])));
			svst1_f64(_pg, &(c->element[1][_ix]), svld1_f64(_pg, &(a->element[1][_ix])));
			svst1_f64(_pg, &(c->element[2][_ix]), svld1_f64(_pg, &(a->element[2][_ix])));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) 
	long int real_total_dim;

	real_total_dim = c->real_row_dim * c->real_col_dim;

	#pragma omp parallel for
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		vst1q_f64(&(c->element[0][index]), vld1q_f64(&(a->element[0][index])));
		vst1q_f64(&(c->element[1][index]), vld1q_f64(&(a->element[1][index])));
		vst1q_f64(&(c->element[2][index]), vld1q_f64(&(a->element[2][index])));
	}
#else // others
	long int total_dim;

	/* element[] uses real_col_dim stride (SIMD padding); copy the full padded
	 * buffer (row i begins at i*real_col_dim). row_dim*col_dim misaligned it. */
	total_dim = c->real_row_dim * c->real_col_dim;

	#pragma omp parallel for
	for(index = 0; index < total_dim; index++)
	{
		c->element[0][index] = a->element[0][index];
		c->element[1][index] = a->element[1][index];
		c->element[2][index] = a->element[2][index];
	}

	//#pragma omp parallel for
	//for(index = 0; index < TDSIZE; index++)
	//	memcpy((void *)(c->element[index]), (void *)(a->element[index]), (size_t)(sizeof(double) * total_dim)); 

#endif // AVX2
}

/* c := I */
void _bncomp_setI_tdmatrix(TDMatrix c)
{
	long int i, real_total_dim;
	double tmp0[TDSIZE], tmp1[TDSIZE];

	rtd_set_ui(tmp0, 0UL);
	rtd_set_ui(tmp1, 1UL);

	real_total_dim = c->real_row_dim * c->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&(c->element[0][i]), zero4);
		_mm256_store_pd(&(c->element[1][i]), zero4);
		_mm256_store_pd(&(c->element[2][i]), zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&(c->element[0][i]), zero8);
		_mm512_store_pd(&(c->element[1][i]), zero8);
		_mm512_store_pd(&(c->element[2][i]), zero8);
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
			svst1_f64(_pg, &(c->element[2][_ix]), _z);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) 
	float64x2_t zero2;

	zero2 = vdupq_n_f64(0.0);
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		vst1q_f64(&(c->element[0][i]), zero2);
		vst1q_f64(&(c->element[1][i]), zero2);
		vst1q_f64(&(c->element[2][i]), zero2);
	}
#else // others
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i++)
	{
		c->element[0][i] = 0.0;
		c->element[1][i] = 0.0;	
		c->element[2][i] = 0.0;	
	}
#endif // __AVX2__

	rtd_set_ui(tmp1, 1UL);

	#pragma omp parallel for
	for(i = 0; i < c->row_dim; i++)
	{
		if(i < c->col_dim)
			set_tdmatrix_ij(c, i, i, tmp1);
	}

}

// set a zero matrix
//void set0_tdmatrix(TDMatrix mat)
void _bncomp_set0_tdmatrix(TDMatrix mat)
{
	long int i, real_total_dim;

	real_total_dim = mat->real_row_dim * mat->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&(mat->element[0][i]), zero4);
		_mm256_store_pd(&(mat->element[1][i]), zero4);
		_mm256_store_pd(&(mat->element[2][i]), zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&(mat->element[0][i]), zero8);
		_mm512_store_pd(&(mat->element[1][i]), zero8);
		_mm512_store_pd(&(mat->element[2][i]), zero8);
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
			svst1_f64(_pg, &(mat->element[2][_ix]), _z);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) 
	float64x2_t zero2;

	zero2 = vdupq_n_f64(0.0);
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		vst1q_f64(&(mat->element[0][i]), zero2);
		vst1q_f64(&(mat->element[1][i]), zero2);
		vst1q_f64(&(mat->element[2][i]), zero2);
	}
#else // others
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i++)
	{
		mat->element[0][i] = 0.0;
		mat->element[1][i] = 0.0;	
		mat->element[2][i] = 0.0;	
	}
#endif // __AVX2__
}

/* v := a * vb */
void _bncomp_mul_tdmatrix_tdvec(TDVector v, TDMatrix a, TDVector vb)
{
	long int i, j, row_dim;
	int thread_index;
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp1[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	/* Check Dimension */
	//if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	if((v->dim < a->row_dim) || (vb->dim < a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_tdmatrix_tdvec\n");
		return;
	}

	row_dim = a->row_dim;

// SIMD
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int ij_index, real_row_dim, real_col_dim;
	__m256d tmp4[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp1_4[BNCOMP_MAX_NUM_THREADS][TDSIZE];
	__m256d aij4[BNCOMP_MAX_NUM_THREADS][TDSIZE], vbj4[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j, ij_index)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		//rtd_set_ui(tmp, 0UL);
		_bncavx2_set0_td(tmp4[thread_index]);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij4[thread_index][0] = _mm256_load_pd(&(a->element[0][ij_index]));
			aij4[thread_index][1] = _mm256_load_pd(&(a->element[1][ij_index]));
			aij4[thread_index][2] = _mm256_load_pd(&(a->element[2][ij_index]));
			vbj4[thread_index][0] = _mm256_load_pd(&(vb->element[0][j]));
			vbj4[thread_index][1] = _mm256_load_pd(&(vb->element[1][j]));
			vbj4[thread_index][2] = _mm256_load_pd(&(vb->element[2][j]));

			//rtd_mul(tmp1, get_tdmatrix_ij(a, i, j), get_tdvector_i(vb, j));
			//rtd_add(tmp, tmp, tmp1);
			_bncavx2_rtd_mul(tmp1_4[thread_index], aij4[thread_index], vbj4[thread_index]);
			_bncavx2_rtd_add(tmp4[thread_index], tmp4[thread_index], tmp1_4[thread_index]);
		}
		//set_ddvector_i(v, i, tmp);
		_bncavx2_rtd_sum256d(tmp[thread_index], tmp4[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
		v->element[2][i] = tmp[thread_index][2];
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int ij_index, real_col_dim;
	__m512d tmp8[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp1_8[BNCOMP_MAX_NUM_THREADS][TDSIZE];
	__m512d aij8[BNCOMP_MAX_NUM_THREADS][TDSIZE], vbj8[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j, ij_index)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		//rtd_set_ui(tmp, 0UL);
		_bncavx512_set0_td(tmp8[thread_index]);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij8[thread_index][0] = _mm512_load_pd(&(a->element[0][ij_index]));
			aij8[thread_index][1] = _mm512_load_pd(&(a->element[1][ij_index]));
			aij8[thread_index][2] = _mm512_load_pd(&(a->element[2][ij_index]));
			vbj8[thread_index][0] = _mm512_load_pd(&(vb->element[0][j]));
			vbj8[thread_index][1] = _mm512_load_pd(&(vb->element[1][j]));
			vbj8[thread_index][2] = _mm512_load_pd(&(vb->element[2][j]));

			//rtd_mul(tmp1, get_tdmatrix_ij(a, i, j), get_tdvector_i(vb, j));
			//rtd_add(tmp, tmp, tmp1);
			_bncavx512_rtd_mul(tmp1_8[thread_index], aij8[thread_index], vbj8[thread_index]);
			_bncavx512_rtd_add(tmp8[thread_index], tmp8[thread_index], tmp1_8[thread_index]);
		}
		//set_ddvector_i(v, i, tmp);
		_bncavx512_rtd_sum512d(tmp[thread_index], tmp8[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
		v->element[2][i] = tmp[thread_index][2];
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic) + OpenMP
	/* SVE2 dot-product matvec for TD: 3-limb accumulator vectors. */
	{
		long sve_real_col_dim = a->real_col_dim;
		long sve_vl = (long)svcntd();

		#pragma omp parallel for private(thread_index, i, j) firstprivate(sve_vl, sve_real_col_dim)
		for(i = 0; i < row_dim; i++)
		{
			thread_index = omp_get_thread_num();
			(void)thread_index;
			svfloat64_t acc0, acc1, acc2;
			_bncsve2_rtd_set0(&acc0, &acc1, &acc2);

			for(j = 0; j < sve_real_col_dim; j += sve_vl)
			{
				svbool_t pg = svwhilelt_b64_s64((int64_t)j, (int64_t)sve_real_col_dim);
				svfloat64_t aij0 = svld1_f64(pg, &(a->element[0][i * sve_real_col_dim + j]));
				svfloat64_t aij1 = svld1_f64(pg, &(a->element[1][i * sve_real_col_dim + j]));
				svfloat64_t aij2_v = svld1_f64(pg, &(a->element[2][i * sve_real_col_dim + j]));
				svfloat64_t bj0  = svld1_f64(pg, &(vb->element[0][j]));
				svfloat64_t bj1  = svld1_f64(pg, &(vb->element[1][j]));
				svfloat64_t bj2  = svld1_f64(pg, &(vb->element[2][j]));
				svfloat64_t prod0, prod1, prod2;
				_bncsve2_rtd_mul(pg, &prod0, &prod1, &prod2,
				                 aij0, aij1, aij2_v, bj0, bj1, bj2);
				_bncsve2_rtd_add(pg, &acc0, &acc1, &acc2,
				                 acc0, acc1, acc2, prod0, prod1, prod2);
			}

			/* Horizontal sum across VL lanes via store-and-add */
			{
				double acc0_arr[sve_vl], acc1_arr[sve_vl], acc2_arr[sve_vl];
				svst1_f64(svptrue_b64(), acc0_arr, acc0);
				svst1_f64(svptrue_b64(), acc1_arr, acc1);
				svst1_f64(svptrue_b64(), acc2_arr, acc2);
				double sum_td[TDSIZE] = { 0.0, 0.0, 0.0 };
				long lane;
				for(lane = 0; lane < sve_vl; lane++)
				{
					double lane_td[TDSIZE] = { acc0_arr[lane], acc1_arr[lane], acc2_arr[lane] };
					rtd_add(sum_td, sum_td, lane_td);
				}
				v->element[0][i] = sum_td[0];
				v->element[1][i] = sum_td[1];
				v->element[2][i] = sum_td[2];
			}
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON)
	long int ij_index, real_col_dim;
	float64x2_t tmp2[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp1_2[BNCOMP_MAX_NUM_THREADS][TDSIZE];
	float64x2_t aij2[BNCOMP_MAX_NUM_THREADS][TDSIZE], vbj2[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j, ij_index)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		_bncneon_set0_td(tmp2[thread_index]);
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij2[thread_index][0] = vld1q_f64(&(a->element[0][ij_index]));
			aij2[thread_index][1] = vld1q_f64(&(a->element[1][ij_index]));
			aij2[thread_index][2] = vld1q_f64(&(a->element[2][ij_index]));
			vbj2[thread_index][0] = vld1q_f64(&(vb->element[0][j]));
			vbj2[thread_index][1] = vld1q_f64(&(vb->element[1][j]));
			vbj2[thread_index][2] = vld1q_f64(&(vb->element[2][j]));

			_bncneon_rtd_mul(tmp1_2[thread_index], aij2[thread_index], vbj2[thread_index]);
			_bncneon_rtd_add(tmp2[thread_index], tmp2[thread_index], tmp1_2[thread_index]);
		}
		_bncneon_rtd_sum128d(tmp[thread_index], tmp2[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
		v->element[2][i] = tmp[thread_index][2];
	}
#else // others
	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		rtd_set_ui(tmp[thread_index], 0UL);
		for(j = 0; j < a->col_dim; j++)
		{
			rtd_mul(tmp1[thread_index], get_tdmatrix_ij(a, i, j), get_tdvector_i(vb, j));
			rtd_add(tmp[thread_index], tmp[thread_index], tmp1[thread_index]);
		}
		set_tdvector_i(v, i, tmp[thread_index]);
	}
#endif // __AVX2__
}

/* v := a^T * vb */
void _bncomp_mul_tdmatrixt_tdvec(TDVector v, TDMatrix a, TDVector vb)
{
	long int i, j, col_dim;
	int thread_index;
	double tmp[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp1[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	/* Check Dimension */
	//if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	if((v->dim < a->col_dim) || (vb->dim < a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_tdmatrixt_ddvec\n");
		return;
	}

	col_dim = a->col_dim;

// SIMD
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int real_row_dim, real_col_dim;
	__m256d tmp4[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp1_4[BNCOMP_MAX_NUM_THREADS][TDSIZE];
	__m256d aij4[BNCOMP_MAX_NUM_THREADS][TDSIZE], vbj4[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < col_dim; i++)
	{
		thread_index = omp_get_thread_num();

		//rtd_set_ui(tmp, 0UL);
		_bncavx2_set0_td(tmp4[thread_index]);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_row_dim; j += _BNC_D_WIDTH)
		{
			//aij4[thread_index][0] = _mm256_load_pd(&(a->element[0][ij_index]));
			//aij4[thread_index][1] = _mm256_load_pd(&(a->element[1][ij_index]));
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
			aij4[thread_index][2] = _mm256_set_pd(
				a->element[2][(j + 3) * real_col_dim + i],
				a->element[2][(j + 2) * real_col_dim + i],
				a->element[2][(j + 1) * real_col_dim + i],
				a->element[2][(j    ) * real_col_dim + i]
			);
			vbj4[thread_index][0] = _mm256_load_pd(&(vb->element[0][j]));
			vbj4[thread_index][1] = _mm256_load_pd(&(vb->element[1][j]));
			vbj4[thread_index][2] = _mm256_load_pd(&(vb->element[2][j]));

			//rtd_mul(tmp1, get_tdmatrix_ij(a, i, j), get_tdvector_i(vb, j));
			//rtd_add(tmp, tmp, tmp1);
			_bncavx2_rtd_mul(tmp1_4[thread_index], aij4[thread_index], vbj4[thread_index]);
			_bncavx2_rtd_add(tmp4[thread_index], tmp4[thread_index], tmp1_4[thread_index]);
		}
		//set_tdvector_i(v, i, tmp);
		_bncavx2_rtd_sum256d(tmp[thread_index], tmp4[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
		v->element[2][i] = tmp[thread_index][2];
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int real_row_dim, real_col_dim;
	__m512d tmp8[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp1_8[BNCOMP_MAX_NUM_THREADS][TDSIZE];
	__m512d aij8[BNCOMP_MAX_NUM_THREADS][TDSIZE], vbj8[BNCOMP_MAX_NUM_THREADS][TDSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < col_dim; i++)
	{
		thread_index = omp_get_thread_num();

		//rtd_set_ui(tmp, 0UL);
		_bncavx512_set0_td(tmp8[thread_index]);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_row_dim; j += _BNC_D_WIDTH)
		{
			//aij8[thread_index][0] = _mm512_load_pd(&(a->element[0][ij_index]));
			//aij8[thread_index][1] = _mm512_load_pd(&(a->element[1][ij_index]));
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
			aij8[thread_index][2] = _mm512_set_pd(
				a->element[2][(j + 7) * real_col_dim + i],
				a->element[2][(j + 6) * real_col_dim + i],
				a->element[2][(j + 5) * real_col_dim + i],
				a->element[2][(j + 4) * real_col_dim + i],
				a->element[2][(j + 3) * real_col_dim + i],
				a->element[2][(j + 2) * real_col_dim + i],
				a->element[2][(j + 1) * real_col_dim + i],
				a->element[2][(j    ) * real_col_dim + i]
			);
			vbj8[thread_index][0] = _mm512_load_pd(&(vb->element[0][j]));
			vbj8[thread_index][1] = _mm512_load_pd(&(vb->element[1][j]));
			vbj8[thread_index][2] = _mm512_load_pd(&(vb->element[2][j]));

			//rtd_mul(tmp1, get_tdmatrix_ij(a, i, j), get_tdvector_i(vb, j));
			//rtd_add(tmp, tmp, tmp1);
			_bncavx512_rtd_mul(tmp1_8[thread_index], aij8[thread_index], vbj8[thread_index]);
			_bncavx512_rtd_add(tmp8[thread_index], tmp8[thread_index], tmp1_8[thread_index]);
		}
		//set_ddvector_i(v, i, tmp);
		_bncavx512_rtd_sum512d(tmp[thread_index], tmp8[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
		v->element[2][i] = tmp[thread_index][2];
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic) + OpenMP
	/* SVE2 transpose-matvec for TD: outer-product, parallel over i tiles. */
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
			svfloat64_t acc0, acc1, acc2;
			_bncsve2_rtd_set0(&acc0, &acc1, &acc2);

			for(j = 0; j < sve_real_row_dim; j++)
			{
				svfloat64_t aji0 = svld1_f64(pg, &(a->element[0][j * sve_real_col_dim + i]));
				svfloat64_t aji1 = svld1_f64(pg, &(a->element[1][j * sve_real_col_dim + i]));
				svfloat64_t aji2 = svld1_f64(pg, &(a->element[2][j * sve_real_col_dim + i]));
				svfloat64_t bj0  = svdup_n_f64(vb->element[0][j]);
				svfloat64_t bj1  = svdup_n_f64(vb->element[1][j]);
				svfloat64_t bj2  = svdup_n_f64(vb->element[2][j]);
				svfloat64_t prod0, prod1, prod2;
				_bncsve2_rtd_mul(pg, &prod0, &prod1, &prod2,
				                 aji0, aji1, aji2, bj0, bj1, bj2);
				_bncsve2_rtd_add(pg, &acc0, &acc1, &acc2,
				                 acc0, acc1, acc2, prod0, prod1, prod2);
			}

			svst1_f64(pg, &(v->element[0][i]), acc0);
			svst1_f64(pg, &(v->element[1][i]), acc1);
			svst1_f64(pg, &(v->element[2][i]), acc2);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON)
	long int real_row_dim, real_col_dim;
	float64x2_t tmp2[BNCOMP_MAX_NUM_THREADS][TDSIZE], tmp1_2[BNCOMP_MAX_NUM_THREADS][TDSIZE];
	float64x2_t aij2[BNCOMP_MAX_NUM_THREADS][TDSIZE], vbj2[BNCOMP_MAX_NUM_THREADS][TDSIZE];
	double aij_vals[BNCOMP_MAX_NUM_THREADS][2];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j, aij_vals)
	for(i = 0; i < col_dim; i++)
	{
		thread_index = omp_get_thread_num();

		_bncneon_set0_td(tmp2[thread_index]);
		for(j = 0; j < real_row_dim; j += _BNC_D_WIDTH)
		{
			// Gather elements for transpose operation
			aij_vals[thread_index][0] = a->element[0][j * real_col_dim + i];
			aij_vals[thread_index][1] = a->element[0][(j + 1) * real_col_dim + i];
			aij2[thread_index][0] = vld1q_f64(aij_vals[thread_index]);
			aij_vals[thread_index][0] = a->element[1][j * real_col_dim + i];
			aij_vals[thread_index][1] = a->element[1][(j + 1) * real_col_dim + i];
			aij2[thread_index][1] = vld1q_f64(aij_vals[thread_index]);
			aij_vals[thread_index][0] = a->element[2][j * real_col_dim + i];
			aij_vals[thread_index][1] = a->element[2][(j + 1) * real_col_dim + i];
			aij2[thread_index][2] = vld1q_f64(aij_vals[thread_index]);
			vbj2[thread_index][0] = vld1q_f64(&(vb->element[0][j]));
			vbj2[thread_index][1] = vld1q_f64(&(vb->element[1][j]));
			vbj2[thread_index][2] = vld1q_f64(&(vb->element[2][j]));

			_bncneon_rtd_mul(tmp1_2[thread_index], aij2[thread_index], vbj2[thread_index]);
			_bncneon_rtd_add(tmp2[thread_index], tmp2[thread_index], tmp1_2[thread_index]);
		}
		_bncneon_rtd_sum128d(tmp[thread_index], tmp2[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
		v->element[2][i] = tmp[thread_index][2];
	}

#else // others
	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < col_dim; i++)
	{
		thread_index = omp_get_thread_num();

		set0_td(tmp[thread_index]);
		for(j = 0; j < a->row_dim; j++)
		{
			rtd_mul(tmp1[thread_index], get_tdmatrix_ij(a, j, i), get_tdvector_i(vb, j));
			rtd_add(tmp[thread_index], tmp[thread_index], tmp1[thread_index]);
		}
		set_tdvector_i(v, i, tmp[thread_index]);
	}
#endif // __AVX2__
}

// Matrix multiplication based on Ozaki scheme
void _bncomp_mul_tdmatrix_oz(TDMatrix ret, TDMatrix a, int max_num_div_a, TDMatrix b, int max_num_div_b) //, int num_digits)
{
    int i, j;
    int real_num_div_a, real_num_div_b;
    long int row_dim = ret->row_dim, col_dim = ret->col_dim, mid_dim = a->col_dim;
    DMatrix *div_a, *div_b, *div_ret;//div_ret[BNCOMP_MAX_NUM_THREADS];
	int thread_index, thread_num;

    if(mid_dim != b->row_dim)
    {
        fprintf(stderr, "ERROR: mul_tdmatrix_oz mid_dim(a, b) = (%ld, %ld)!\n", mid_dim, b->row_dim);
        return;
    }

	thread_num = omp_get_num_threads();

    div_a = (DMatrix *)calloc(max_num_div_a, sizeof(DMatrix));
    div_b = (DMatrix *)calloc(max_num_div_b, sizeof(DMatrix));
    div_ret = (DMatrix *)calloc(max_num_div_a, sizeof(DMatrix));
    //div_ret = (DMatrix *)calloc(max_num_div_b, sizeof(DMatrix));
    //div_ret = (DMatrix *)calloc(thread_num, sizeof(DMatrix));

	#pragma omp parallel for
    for(i = 0; i < max_num_div_a; i++)
	{
        div_a[i] = init_dmatrix(row_dim, mid_dim);
		div_ret[i] = init_dmatrix(row_dim, col_dim);
	}

	#pragma omp parallel for
    for(i = 0; i < max_num_div_b; i++)
	{
        div_b[i] = init_dmatrix(mid_dim, col_dim);
		//div_ret[i] = init_dmatrix(row_dim, col_dim);
	}

/*/
	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
	    div_ret[thread_index] = init_dmatrix(row_dim, col_dim);
	}
*/

	#pragma omp parallel sections
	{
		#pragma omp section
		real_num_div_a = split_tdmatrix_dmat(div_a, max_num_div_a, a);
		//printf("split_tdmatrix_dmat(%d, %d)  ->%d\n", div_a[0]->real_row_dim, div_a[0]->real_col_dim, real_num_div_a);

		#pragma omp section
		real_num_div_b = split_tdmatrix_t_dmat(div_b, max_num_div_b, b);
		//printf("split_tdmatrix_t_dmat(%d, %d)->%d\n", div_b[0]->real_row_dim, div_b[0]->real_col_dim, real_num_div_b);
	}

    set0_tdmatrix(ret);
	#pragma omp parallel for private(i, j) // , div_ret)
    for(i = 0; i < real_num_div_a; i++)
    {
		//thread_index = omp_get_thread_num();
		//printf("-- start -- %d th thread --\n", thread_index);

        //for(j = 0; j < real_num_div_b; j++)
        for(j = 0; j < real_num_div_b - i; j++)
        {
            //printf("(i, j) = (%d, %d), %d, %d\n", i, j, div_b[j]->real_row_dim, div_b[j]->real_col_dim);
#ifdef USE_IMKL
            //set0_dmatrix(div_ret[thread_index]);
            //set0_dmatrix(div_ret[j]);
			set0_dmatrix(div_ret[i]);
            cblas_dgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasNoTrans,
                div_a[i]->real_row_dim, // m
                div_b[j]->real_col_dim, // n
                div_a[i]->real_col_dim, // k
                1.0,
                div_a[i]->element,
                div_a[i]->real_col_dim, // k
                div_b[j]->element,
                div_b[j]->real_col_dim, // n
                1.0,
                div_ret[i]->element,
                div_ret[i]->real_col_dim   // n
                //div_ret[j]->element,
                //div_ret[j]->real_col_dim   // n
                //div_ret[thread_index]->element,
                //div_ret[thread_index]->real_col_dim   // n
            );
#else // USE_IMKL
            //mul_dmatrix(div_ret[thread_index], div_a[i], div_b[j]);
            //mul_dmatrix(div_ret[j], div_a[i], div_b[j]);
            mul_dmatrix(div_ret[i], div_a[i], div_b[j]);
#endif // USE_IMKL
            //add_tdmatrix_dmat(ret, ret, div_ret[thread_index]);	 
            //add_tdmatrix_dmat(ret, ret, div_ret[j]);
			#pragma omp critical
				add_tdmatrix_dmat(ret, ret, div_ret[i]);
    	}
		//printf("-- end -- %d th thread --\n", thread_index);
    }

/*
	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
	    free_dmatrix(div_ret[thread_index]);
	}
*/
	#pragma omp parallel for
    for(i = 0; i < max_num_div_a; i++)
	{
        free_dmatrix(div_a[i]);
		free_dmatrix(div_ret[i]);
	}

	#pragma omp parallel for
    for(i = 0; i < max_num_div_b; i++)
	{
        free_dmatrix(div_b[i]);
		//free_dmatrix(div_ret[i]);
	}

    free(div_a);
    free(div_b);
	free(div_ret);

}

#if 0 

// Fit dimension to be multiple of min_dim
void _bncomp_mul_ctdmatrix_oz_3m(CTDMatrix ret, CTDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CTDMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    TDMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    t1 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_tdmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    _bncomp_mul_tdmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_a_image);
    _bncomp_sub_tdmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        mul_tdmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        mul_tdmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_a_image);
        add_tdmatrix(ret->im, t3, t4);
    #else // USE_4M
    */
        // 3M
        _bncomp_add_tdmatrix(t3, a->re, a->im);
        _bncomp_add_tdmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        _bncomp_mul_tdmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        _bncomp_sub_tdmatrix(ret->im, ret->im, t1);
        _bncomp_sub_tdmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_tdmatrix(t1);
    free_tdmatrix(t2);
    free_tdmatrix(t3);
    free_tdmatrix(t4);
}

// Fit dimension to be multiple of min_dim
void _bncomp_mul_ctdmatrix_oz_4m(CTDMatrix ret, CTDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CTDMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    TDMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    t1 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_tdmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    _bncomp_mul_tdmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_a_image);
    _bncomp_sub_tdmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        _bncomp_mul_tdmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        _bncomp_mul_tdmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_a_image);
        _bncomp_add_tdmatrix(ret->im, t3, t4);
    /*
    #else // USE_4M 
        // 3M
        add_tdmatrix(t3, a->re, a->im);
        add_tdmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        mul_tdmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        sub_tdmatrix(ret->im, ret->im, t1);
        sub_tdmatrix(ret->im, ret->im, t2);
    #endif // USE_4M
    */

    free_tdmatrix(t1);
    free_tdmatrix(t2);
    free_tdmatrix(t3);
    free_tdmatrix(t4);
}

// Simple triple-loop-way matrix multiplication (3M)
void _bncomp_mul_ctdmatrix_3m(CTDMatrix ret, CTDMatrix a, CTDMatrix b)
{
    TDMatrix t1, t2, t3, t4;
 
    t1 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_tdmatrix(t1, a->re, b->re);
    _bncomp_mul_tdmatrix(t2, a->im, b->im);
    _bncomp_sub_tdmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        _bncomp_mul_tdmatrix(t3, a->im, b->re);
        _bncomp_mul_tdmatrix(t4, a->re, b->im);
        _bncomp_add_tdmatrix(ret->im, t3, t4);
    #else // USE_4M
    */
        // 3M
        _bncomp_add_tdmatrix(t3, a->re, a->im);
        _bncomp_add_tdmatrix(t4, b->re, b->im);
        _bncomp_mul_tdmatrix(ret->im, t3, t4);
        _bncomp_sub_tdmatrix(ret->im, ret->im, t1);
        _bncomp_sub_tdmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_tdmatrix(t1);
    free_tdmatrix(t2);
    free_tdmatrix(t3);
    free_tdmatrix(t4);
}

// Simple triple-loop-way matrix multiplication (4M)
void _bncomp_mul_ctdmatrix_4m(CTDMatrix ret, CTDMatrix a, CTDMatrix b)
{
    TDMatrix t1, t2, t3, t4;
 
    t1 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_tdmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_tdmatrix(t1, a->re, b->re);
    _bncomp_mul_tdmatrix(t2, a->im, b->im);
    _bncomp_sub_tdmatrix(ret->re, t1, t2);

    // 4M
    
    //#ifdef USE_4M
        _bncomp_mul_tdmatrix(t3, a->im, b->re);
        _bncomp_mul_tdmatrix(t4, a->re, b->im);
        _bncomp_add_tdmatrix(ret->im, t3, t4);
    //#else // USE_4M
	/*
        // 3M
        _bncomp_add_tdmatrix(t3, a->re, a->im);
        _bncomp_add_tdmatrix(t4, b->re, b->im);
        _bncomp_mul_tdmatrix(ret->im, t3, t4s);
        _bncomp_sub_tdmatrix(ret->im, ret->im, t1);
        _bncomp_sub_tdmatrix(ret->im, ret->im, t2);
	*/
    //#endif // USE_4M

    free_tdmatrix(t1);
    free_tdmatrix(t2);
    free_tdmatrix(t3);
    free_tdmatrix(t4);
}
#endif // 0

/****************************************************************************/
/* OpenMP-parallel LU decomposition (partial pivoting), mirrors TDLUdecompPM */
/* with the O(n^3) trailing update parallelized over rows j. SVE2 build uses */
/* the NEON inner kernel (arrays-of-SVE-types unusable), as the serial does. */
/****************************************************************************/
int _bncomp_TDLUdecompPM(TDMatrix a, long int ch[])
{
	long int i, j, k, imax, itmp, dim, dim_start, dim_end;
	double dtmp[TDSIZE], dtmp1[TDSIZE], dmaxii[TDSIZE];

	dim = a->col_dim;
	for(i = 0; i < a->row_dim; i++) ch[i] = i;

	for(i = 0; i < a->row_dim; i++)
	{
		rtd_abs(dmaxii, get_tdmatrix_ij(a, i, i));
		imax = i;
		for(j = i + 1; j < a->row_dim; j++)
		{
			rtd_abs(dtmp, get_tdmatrix_ij(a, j, i));
			if(rtd_cmp(dtmp, dmaxii) > 0) { imax = j; rtd_set(dmaxii, dtmp); }
		}
		if(rtd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! _bncomp_TDLUdecompPM!\n", i);
			return -1;
		}
		if(imax != i)
		{
			itmp = ch[imax]; ch[imax] = ch[i]; ch[i] = itmp;
			row_swap_tdmatrix(a, i, imax, 0, a->col_dim);
		}
		for(j = i + 1; j < dim; j++)
		{
			rtd_div(dtmp, get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, i));
			set_tdmatrix_ij(a, j, i, dtmp);
		}
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->real_col_dim;
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
		#pragma omp parallel for schedule(static) private(k)
		for(j = i + 1; j < dim; j++)
		{
			double _t[TDSIZE], _t1[TDSIZE];
			long int index_ji = j * a->real_col_dim + i, index_ik, index_jk, k_end;
			__m256d dtmpv[TDSIZE], ajiv[TDSIZE], ajkv[TDSIZE], aikv[TDSIZE];
			ajiv[0] = _mm256_set1_pd(a->element[0][index_ji]);
			ajiv[1] = _mm256_set1_pd(a->element[1][index_ji]);
			ajiv[2] = _mm256_set1_pd(a->element[2][index_ji]);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = i + 1; k < k_end; k++)
			{
				rtd_mul(_t1, get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, k));
				rtd_sub(_t, get_tdmatrix_ij(a, j, k), _t1);
				set_tdmatrix_ij(a, j, k, _t);
			}
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				aikv[0] = _mm256_load_pd(&(a->element[0][index_ik]));
				aikv[1] = _mm256_load_pd(&(a->element[1][index_ik]));
				aikv[2] = _mm256_load_pd(&(a->element[2][index_ik]));
				_bncavx2_rtd_mul(dtmpv, ajiv, aikv);
				index_jk = j * a->real_col_dim + k;
				ajkv[0] = _mm256_load_pd(&(a->element[0][index_jk]));
				ajkv[1] = _mm256_load_pd(&(a->element[1][index_jk]));
				ajkv[2] = _mm256_load_pd(&(a->element[2][index_jk]));
				_bncavx2_rtd_sub(dtmpv, ajkv, dtmpv);
				_mm256_store_pd(&(a->element[0][index_jk]), dtmpv[0]);
				_mm256_store_pd(&(a->element[1][index_jk]), dtmpv[1]);
				_mm256_store_pd(&(a->element[2][index_jk]), dtmpv[2]);
			}
		}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon (also SVE2 build)
		#pragma omp parallel for schedule(static) private(k)
		for(j = i + 1; j < dim; j++)
		{
			double _t[TDSIZE], _t1[TDSIZE];
			long int index_ji = j * a->real_col_dim + i, index_ik, index_jk, k_end;
			float64x2_t dtmpv[TDSIZE], ajiv[TDSIZE], ajkv[TDSIZE], aikv[TDSIZE];
			ajiv[0] = vdupq_n_f64(a->element[0][index_ji]);
			ajiv[1] = vdupq_n_f64(a->element[1][index_ji]);
			ajiv[2] = vdupq_n_f64(a->element[2][index_ji]);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = i + 1; k < k_end; k++)
			{
				rtd_mul(_t1, get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, k));
				rtd_sub(_t, get_tdmatrix_ij(a, j, k), _t1);
				set_tdmatrix_ij(a, j, k, _t);
			}
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				aikv[0] = vld1q_f64(&(a->element[0][index_ik]));
				aikv[1] = vld1q_f64(&(a->element[1][index_ik]));
				aikv[2] = vld1q_f64(&(a->element[2][index_ik]));
				_bncneon_rtd_mul(dtmpv, ajiv, aikv);
				index_jk = j * a->real_col_dim + k;
				ajkv[0] = vld1q_f64(&(a->element[0][index_jk]));
				ajkv[1] = vld1q_f64(&(a->element[1][index_jk]));
				ajkv[2] = vld1q_f64(&(a->element[2][index_jk]));
				_bncneon_rtd_sub(dtmpv, ajkv, dtmpv);
				vst1q_f64(&(a->element[0][index_jk]), dtmpv[0]);
				vst1q_f64(&(a->element[1][index_jk]), dtmpv[1]);
				vst1q_f64(&(a->element[2][index_jk]), dtmpv[2]);
			}
		}
#else // others
		#pragma omp parallel for schedule(static) private(k)
		for(j = i + 1; j < dim; j++)
		{
			double _t[TDSIZE], _t1[TDSIZE];
			for(k = i + 1; k < dim; k++)
			{
				rtd_mul(_t1, get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, k));
				rtd_sub(_t, get_tdmatrix_ij(a, j, k), _t1);
				set_tdmatrix_ij(a, j, k, _t);
			}
		}
#endif // __AVX2__
	}
	return 0;
}

int _bncomp_SolveTDLSPM(TDVector answer, TDMatrix lu, TDVector b, long int ch[])
{
	long int i, j, dim;
	double dtmp[TDSIZE], dtmp1[TDSIZE];

	dim = answer->dim;
	for(i = 0; i < dim; i++) set_tdvector_i(answer, i, get_tdvector_i(b, ch[i]));

	for(i = 0; i < dim; i++)
	{
		rtd_abs(dtmp, get_tdmatrix_ij(lu, i, i));
		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(_bncomp_SolveTDLSPM, %ld)\n", i);
			return -1;
		}
		#pragma omp parallel for schedule(static)
		for(j = i + 1; j < dim; j++)
		{
			double _t[TDSIZE], _t1[TDSIZE];
			rtd_mul(_t1, get_tdmatrix_ij(lu, j, i), get_tdvector_i(answer, i));
			rtd_sub(_t, get_tdvector_i(answer, j), _t1);
			set_tdvector_i(answer, j, _t);
		}
	}
	for(i = dim - 1; i >= 0; i--)
	{
		for(j = i + 1; j < dim; j++)
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

