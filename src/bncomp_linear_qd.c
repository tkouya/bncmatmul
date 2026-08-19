/********************************************************************************/
/* bncomp_linear_qd.c: Parallelized DD Precision Linear Computation Library     */
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
// QD
//---------------------------------------
//--------
// Vector
//--------
/* c := a */
void _bncomp_subst_qdvector(QDVector c, QDVector a)
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
		_mm256_store_pd(&(c->element[3][i]), _mm256_load_pd(&(a->element[3][i])));
	}
#elif defined(__AVX512F__) // __AVX512F__
	#pragma omp parallel for
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//set_tvector_i(c, i, get_tvector_i(a, i));
		_mm512_store_pd(&(c->element[0][i]), _mm512_load_pd(&(a->element[0][i])));
		_mm512_store_pd(&(c->element[1][i]), _mm512_load_pd(&(a->element[1][i])));
		_mm512_store_pd(&(c->element[2][i]), _mm512_load_pd(&(a->element[2][i])));
		_mm512_store_pd(&(c->element[3][i]), _mm512_load_pd(&(a->element[3][i])));
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
			svst1_f64(_pg, &(c->element[3][_ix]), svld1_f64(_pg, &(a->element[3][_ix])));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	#pragma omp parallel for
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		// float64x2_tで2要素ずつロード・ストア
		float64x2_t tmp0 = vld1q_f64(&(a->element[0][i]));
		float64x2_t tmp1 = vld1q_f64(&(a->element[1][i]));
		float64x2_t tmp2 = vld1q_f64(&(a->element[2][i]));
		float64x2_t tmp3 = vld1q_f64(&(a->element[3][i]));
		
		vst1q_f64(&(c->element[0][i]), tmp0);
		vst1q_f64(&(c->element[1][i]), tmp1);
		vst1q_f64(&(c->element[2][i]), tmp2);
		vst1q_f64(&(c->element[3][i]), tmp3);
	}
#else // others
	#pragma omp parallel for
	for(i = 0; i < a->dim; i++)
		set_qdvector_i(c, i, get_qdvector_i(a, i));

#endif // __AVX2__

}

/* c = a + b */
void _bncomp_add_qdvector(QDVector c, QDVector a, QDVector b)
{
	int thread_index;
	long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_qdvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256d in_ret[BNCOMP_MAX_NUM_THREADS][QDSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][QDSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][QDSIZE];

 	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[thread_index][1] = _mm256_load_pd(&(a->element[1][index]));
        in_a_val[thread_index][2] = _mm256_load_pd(&(a->element[2][index]));
        in_a_val[thread_index][3] = _mm256_load_pd(&(a->element[3][index]));
        in_b_val[thread_index][0] = _mm256_load_pd(&(b->element[0][index]));
        in_b_val[thread_index][1] = _mm256_load_pd(&(b->element[1][index]));
        in_b_val[thread_index][2] = _mm256_load_pd(&(b->element[2][index]));
        in_b_val[thread_index][3] = _mm256_load_pd(&(b->element[3][index]));

        _bncavx2_rqd_add(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        _mm256_store_pd(&(c->element[0][index]), in_ret[thread_index][0]);
        _mm256_store_pd(&(c->element[1][index]), in_ret[thread_index][1]);
        _mm256_store_pd(&(c->element[2][index]), in_ret[thread_index][2]);
        _mm256_store_pd(&(c->element[3][index]), in_ret[thread_index][3]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512d in_ret[BNCOMP_MAX_NUM_THREADS][QDSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][QDSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	#pragma omp parallel for private(thread_index)
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = _mm512_load_pd(&(a->element[0][index]));
        in_a_val[thread_index][1] = _mm512_load_pd(&(a->element[1][index]));
        in_a_val[thread_index][2] = _mm512_load_pd(&(a->element[2][index]));
        in_a_val[thread_index][3] = _mm512_load_pd(&(a->element[3][index]));
        in_b_val[thread_index][0] = _mm512_load_pd(&(b->element[0][index]));
        in_b_val[thread_index][1] = _mm512_load_pd(&(b->element[1][index]));
        in_b_val[thread_index][2] = _mm512_load_pd(&(b->element[2][index]));
        in_b_val[thread_index][3] = _mm512_load_pd(&(b->element[3][index]));

        _bncavx512_rqd_add(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        _mm512_store_pd(&(c->element[0][index]), in_ret[thread_index][0]);
        _mm512_store_pd(&(c->element[1][index]), in_ret[thread_index][1]);
        _mm512_store_pd(&(c->element[2][index]), in_ret[thread_index][2]);
        _mm512_store_pd(&(c->element[3][index]), in_ret[thread_index][3]);
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
			svfloat64_t _a3 = svld1_f64(_pg, &(a->element[3][_ix]));
			svfloat64_t _b0 = svld1_f64(_pg, &(b->element[0][_ix]));
			svfloat64_t _b1 = svld1_f64(_pg, &(b->element[1][_ix]));
			svfloat64_t _b2 = svld1_f64(_pg, &(b->element[2][_ix]));
			svfloat64_t _b3 = svld1_f64(_pg, &(b->element[3][_ix]));
			svfloat64_t _o0, _o1, _o2, _o3;
			_bncsve2_rqd_add(_pg, &_o0, &_o1, &_o2, &_o3, _a0, _a1, _a2, _a3, _b0, _b1, _b2, _b3);
			svst1_f64(_pg, &(c->element[0][_ix]), _o0);
			svst1_f64(_pg, &(c->element[1][_ix]), _o1);
			svst1_f64(_pg, &(c->element[2][_ix]), _o2);
			svst1_f64(_pg, &(c->element[3][_ix]), _o3);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	float64x2_t in_ret[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	float64x2_t in_a_val[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	float64x2_t in_b_val[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		// a, b の値をロード (2要素ずつ)
		in_a_val[thread_index][0] = vld1q_f64(&(a->element[0][index]));
		in_a_val[thread_index][1] = vld1q_f64(&(a->element[1][index]));
		in_a_val[thread_index][2] = vld1q_f64(&(a->element[2][index]));
		in_a_val[thread_index][3] = vld1q_f64(&(a->element[3][index]));
		in_b_val[thread_index][0] = vld1q_f64(&(b->element[0][index]));
		in_b_val[thread_index][1] = vld1q_f64(&(b->element[1][index]));
		in_b_val[thread_index][2] = vld1q_f64(&(b->element[2][index]));
		in_b_val[thread_index][3] = vld1q_f64(&(b->element[3][index]));

		// Neon QD加算
		_bncneon_rqd_add(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

		// 結果をストア
		vst1q_f64(&(c->element[0][index]), in_ret[thread_index][0]);
		vst1q_f64(&(c->element[1][index]), in_ret[thread_index][1]);
		vst1q_f64(&(c->element[2][index]), in_ret[thread_index][2]);
		vst1q_f64(&(c->element[3][index]), in_ret[thread_index][3]);
	}
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();
		rqd_add(tmp[thread_index], get_qdvector_i(a, i),  get_qdvector_i(b, i));
		set_qdvector_i(c, i, tmp[thread_index]);
	}
#endif // __AVX2__
}

/* c = a - b */
void _bncomp_sub_qdvector(QDVector c, QDVector a, QDVector b)
{
	int thread_index;
	long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: _bncomp_sub_qdvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256d in_ret[BNCOMP_MAX_NUM_THREADS][QDSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][QDSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	#pragma omp parallel for private(thread_index)
    for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = _mm256_load_pd(&(a->element[0][index]));
        in_a_val[thread_index][1] = _mm256_load_pd(&(a->element[1][index]));
        in_a_val[thread_index][2] = _mm256_load_pd(&(a->element[2][index]));
        in_a_val[thread_index][3] = _mm256_load_pd(&(a->element[3][index]));
        in_b_val[thread_index][0] = _mm256_load_pd(&(b->element[0][index]));
        in_b_val[thread_index][1] = _mm256_load_pd(&(b->element[1][index]));
        in_b_val[thread_index][2] = _mm256_load_pd(&(b->element[2][index]));
        in_b_val[thread_index][3] = _mm256_load_pd(&(b->element[3][index]));

        _bncavx2_rqd_sub(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        _mm256_store_pd(&(c->element[0][index]), in_ret[thread_index][0]);
        _mm256_store_pd(&(c->element[1][index]), in_ret[thread_index][1]);
        _mm256_store_pd(&(c->element[2][index]), in_ret[thread_index][2]);
        _mm256_store_pd(&(c->element[3][index]), in_ret[thread_index][3]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512d in_ret[BNCOMP_MAX_NUM_THREADS][QDSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][QDSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = _mm512_load_pd(&(a->element[0][index]));
        in_a_val[thread_index][1] = _mm512_load_pd(&(a->element[1][index]));
        in_a_val[thread_index][2] = _mm512_load_pd(&(a->element[2][index]));
        in_a_val[thread_index][3] = _mm512_load_pd(&(a->element[3][index]));
        in_b_val[thread_index][0] = _mm512_load_pd(&(b->element[0][index]));
        in_b_val[thread_index][1] = _mm512_load_pd(&(b->element[1][index]));
        in_b_val[thread_index][2] = _mm512_load_pd(&(b->element[2][index]));
        in_b_val[thread_index][3] = _mm512_load_pd(&(b->element[3][index]));

        _bncavx512_rqd_sub(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        _mm512_store_pd(&(c->element[0][index]), in_ret[thread_index][0]);
        _mm512_store_pd(&(c->element[1][index]), in_ret[thread_index][1]);
        _mm512_store_pd(&(c->element[2][index]), in_ret[thread_index][2]);
        _mm512_store_pd(&(c->element[3][index]), in_ret[thread_index][3]);
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
			svfloat64_t _a3 = svld1_f64(_pg, &(a->element[3][_ix]));
			svfloat64_t _b0 = svld1_f64(_pg, &(b->element[0][_ix]));
			svfloat64_t _b1 = svld1_f64(_pg, &(b->element[1][_ix]));
			svfloat64_t _b2 = svld1_f64(_pg, &(b->element[2][_ix]));
			svfloat64_t _b3 = svld1_f64(_pg, &(b->element[3][_ix]));
			_b0 = svneg_f64_x(_pg, _b0);
			_b1 = svneg_f64_x(_pg, _b1);
			_b2 = svneg_f64_x(_pg, _b2);
			_b3 = svneg_f64_x(_pg, _b3);
			svfloat64_t _o0, _o1, _o2, _o3;
			_bncsve2_rqd_add(_pg, &_o0, &_o1, &_o2, &_o3, _a0, _a1, _a2, _a3, _b0, _b1, _b2, _b3);
			svst1_f64(_pg, &(c->element[0][_ix]), _o0);
			svst1_f64(_pg, &(c->element[1][_ix]), _o1);
			svst1_f64(_pg, &(c->element[2][_ix]), _o2);
			svst1_f64(_pg, &(c->element[3][_ix]), _o3);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	float64x2_t in_ret[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	float64x2_t in_a_val[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	float64x2_t in_b_val[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		// a, b の値をロード
		in_a_val[thread_index][0] = vld1q_f64(&(a->element[0][index]));
		in_a_val[thread_index][1] = vld1q_f64(&(a->element[1][index]));
		in_a_val[thread_index][2] = vld1q_f64(&(a->element[2][index]));
		in_a_val[thread_index][3] = vld1q_f64(&(a->element[3][index]));
		in_b_val[thread_index][0] = vld1q_f64(&(b->element[0][index]));
		in_b_val[thread_index][1] = vld1q_f64(&(b->element[1][index]));
		in_b_val[thread_index][2] = vld1q_f64(&(b->element[2][index]));
		in_b_val[thread_index][3] = vld1q_f64(&(b->element[3][index]));

		// Neon QD減算
		_bncneon_rqd_sub(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

		// 結果をストア
		vst1q_f64(&(c->element[0][index]), in_ret[thread_index][0]);
		vst1q_f64(&(c->element[1][index]), in_ret[thread_index][1]);
		vst1q_f64(&(c->element[2][index]), in_ret[thread_index][2]);
		vst1q_f64(&(c->element[3][index]), in_ret[thread_index][3]);
	}
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

		rqd_sub(tmp[thread_index], get_qdvector_i(a, i), get_qdvector_i(b, i));
		set_qdvector_i(c, i, tmp[thread_index]);
	}
#endif // __AVX2__

}

/* c = val * a */
void _bncomp_cmul_qdvector(QDVector c, double val[QDSIZE], QDVector a)
{
	int thread_index;
	long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: cmul_qdvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[BNCOMP_MAX_NUM_THREADS][QDSIZE], c4[BNCOMP_MAX_NUM_THREADS][QDSIZE], val4[QDSIZE];

	val4[0] = _mm256_set1_pd(val[0]);
	val4[1] = _mm256_set1_pd(val[1]);
	val4[2] = _mm256_set1_pd(val[2]);
	val4[3] = _mm256_set1_pd(val[3]);

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		//set_ddvector_i(c, i, val * get_ddvector_i(a, i));
		a4[thread_index][0] = _mm256_load_pd(&(a->element[0][index]));
		a4[thread_index][1] = _mm256_load_pd(&(a->element[1][index]));
		a4[thread_index][2] = _mm256_load_pd(&(a->element[2][index]));
		a4[thread_index][3] = _mm256_load_pd(&(a->element[3][index]));

		_bncavx2_rqd_mul(c4[thread_index], val4, a4[thread_index]);
		//_bncavx2_rqd_mul(c4[thread_index], val4[thread_index], a4[thread_index]);

		_mm256_store_pd(&(c->element[0][index]), c4[thread_index][0]);
		_mm256_store_pd(&(c->element[1][index]), c4[thread_index][1]);
		_mm256_store_pd(&(c->element[2][index]), c4[thread_index][2]);
		_mm256_store_pd(&(c->element[3][index]), c4[thread_index][3]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[BNCOMP_MAX_NUM_THREADS][QDSIZE], c8[BNCOMP_MAX_NUM_THREADS][QDSIZE], val8[QDSIZE];

	val8[0] = _mm512_set1_pd(val[0]);
	val8[1] = _mm512_set1_pd(val[1]);
	val8[2] = _mm512_set1_pd(val[2]);
	val8[3] = _mm512_set1_pd(val[3]);

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		//set_ddvector_i(c, i, val * get_ddvector_i(a, i));
		a8[thread_index][0] = _mm512_load_pd(&(a->element[0][index]));
		a8[thread_index][1] = _mm512_load_pd(&(a->element[1][index]));
		a8[thread_index][2] = _mm512_load_pd(&(a->element[2][index]));
		a8[thread_index][3] = _mm512_load_pd(&(a->element[3][index]));

		_bncavx512_rqd_mul(c8[thread_index], val8, a8[thread_index]);

		_mm512_store_pd(&(c->element[0][index]), c8[thread_index][0]);
		_mm512_store_pd(&(c->element[1][index]), c8[thread_index][1]);
		_mm512_store_pd(&(c->element[2][index]), c8[thread_index][2]);
		_mm512_store_pd(&(c->element[3][index]), c8[thread_index][3]);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = a->real_dim; long _ix;
		svfloat64_t _v0 = svdup_n_f64(val[0]);
		svfloat64_t _v1 = svdup_n_f64(val[1]);
		svfloat64_t _v2 = svdup_n_f64(val[2]);
		svfloat64_t _v3 = svdup_n_f64(val[3]);
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svfloat64_t _a0 = svld1_f64(_pg, &(a->element[0][_ix]));
			svfloat64_t _a1 = svld1_f64(_pg, &(a->element[1][_ix]));
			svfloat64_t _a2 = svld1_f64(_pg, &(a->element[2][_ix]));
			svfloat64_t _a3 = svld1_f64(_pg, &(a->element[3][_ix]));
			svfloat64_t _o0, _o1, _o2, _o3;
			_bncsve2_rqd_mul(_pg, &_o0, &_o1, &_o2, &_o3, _v0, _v1, _v2, _v3, _a0, _a1, _a2, _a3);
			svst1_f64(_pg, &(c->element[0][_ix]), _o0);
			svst1_f64(_pg, &(c->element[1][_ix]), _o1);
			svst1_f64(_pg, &(c->element[2][_ix]), _o2);
			svst1_f64(_pg, &(c->element[3][_ix]), _o3);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	float64x2_t in_ret[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	float64x2_t in_a_val[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	float64x2_t in_scalar_val[QDSIZE];

	// scalar を broadcast
	_bncneon_rqd_set1_qd(in_scalar_val, val); // scalar);

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		// a の値をロード
		in_a_val[thread_index][0] = vld1q_f64(&(a->element[0][index]));
		in_a_val[thread_index][1] = vld1q_f64(&(a->element[1][index]));
		in_a_val[thread_index][2] = vld1q_f64(&(a->element[2][index]));
		in_a_val[thread_index][3] = vld1q_f64(&(a->element[3][index]));

		// Neon QD乗算
		_bncneon_rqd_mul(in_ret[thread_index], in_scalar_val, in_a_val[thread_index]);

		// 結果をストア
		vst1q_f64(&(c->element[0][index]), in_ret[thread_index][0]);
		vst1q_f64(&(c->element[1][index]), in_ret[thread_index][1]);
		vst1q_f64(&(c->element[2][index]), in_ret[thread_index][2]);
		vst1q_f64(&(c->element[3][index]), in_ret[thread_index][3]);
	}
#else // others

	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

		rqd_mul(tmp[thread_index], val, get_qdvector_i(a, i));
		set_qdvector_i(c, i, tmp[thread_index]);
	}
#endif // __AVX2__

}

/* (a, b) */
void _bncomp_ip_qdvector(double ret[QDSIZE], QDVector a, QDVector b)
{
	int thread_index;
	long int i, index;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: _bncomp_ip_qdvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[BNCOMP_MAX_NUM_THREADS][QDSIZE], b4[BNCOMP_MAX_NUM_THREADS][QDSIZE], ret4[QDSIZE], tmp4[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	_bncavx2_set0_qd(ret4);
	
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a4[thread_index][0] = _mm256_load_pd(&(a->element[0][index]));
		a4[thread_index][1] = _mm256_load_pd(&(a->element[1][index]));
		a4[thread_index][2] = _mm256_load_pd(&(a->element[2][index]));
		a4[thread_index][3] = _mm256_load_pd(&(a->element[3][index]));
		b4[thread_index][0] = _mm256_load_pd(&(b->element[0][index]));
		b4[thread_index][1] = _mm256_load_pd(&(b->element[1][index]));
		b4[thread_index][2] = _mm256_load_pd(&(b->element[2][index]));
		b4[thread_index][3] = _mm256_load_pd(&(b->element[3][index]));

		_bncavx2_rqd_mul(tmp4[thread_index], a4[thread_index], b4[thread_index]);
		#pragma omp critical
			_bncavx2_rqd_add(ret4, ret4, tmp4[thread_index]);
	}

	_bncavx2_rqd_sum256d(ret, ret4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[BNCOMP_MAX_NUM_THREADS][QDSIZE], b8[BNCOMP_MAX_NUM_THREADS][QDSIZE], ret8[QDSIZE], tmp8[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	_bncavx512_set0_qd(ret8);
	
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a8[thread_index][0] = _mm512_load_pd(&(a->element[0][index]));
		a8[thread_index][1] = _mm512_load_pd(&(a->element[1][index]));
		a8[thread_index][2] = _mm512_load_pd(&(a->element[2][index]));
		a8[thread_index][3] = _mm512_load_pd(&(a->element[3][index]));
		b8[thread_index][0] = _mm512_load_pd(&(b->element[0][index]));
		b8[thread_index][1] = _mm512_load_pd(&(b->element[1][index]));
		b8[thread_index][2] = _mm512_load_pd(&(b->element[2][index]));
		b8[thread_index][3] = _mm512_load_pd(&(b->element[3][index]));

		_bncavx512_rqd_mul(tmp8[thread_index], a8[thread_index], b8[thread_index]);
		#pragma omp critical
			_bncavx512_rqd_add(ret8, ret8, tmp8[thread_index]);
	}
	_bncavx512_rqd_sum512d(ret, ret8);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = a->real_dim; long _ix;
		svfloat64_t _acc0 = svdup_n_f64(0.0);
		svfloat64_t _acc1 = svdup_n_f64(0.0);
		svfloat64_t _acc2 = svdup_n_f64(0.0);
		svfloat64_t _acc3 = svdup_n_f64(0.0);
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svfloat64_t _a0 = svld1_f64(_pg, &(a->element[0][_ix]));
			svfloat64_t _a1 = svld1_f64(_pg, &(a->element[1][_ix]));
			svfloat64_t _a2 = svld1_f64(_pg, &(a->element[2][_ix]));
			svfloat64_t _a3 = svld1_f64(_pg, &(a->element[3][_ix]));
			svfloat64_t _b0 = svld1_f64(_pg, &(b->element[0][_ix]));
			svfloat64_t _b1 = svld1_f64(_pg, &(b->element[1][_ix]));
			svfloat64_t _b2 = svld1_f64(_pg, &(b->element[2][_ix]));
			svfloat64_t _b3 = svld1_f64(_pg, &(b->element[3][_ix]));
			svfloat64_t _m0, _m1, _m2, _m3;
			_bncsve2_rqd_mul(_pg, &_m0, &_m1, &_m2, &_m3, _a0, _a1, _a2, _a3, _b0, _b1, _b2, _b3);
			_bncsve2_rqd_add(_pg, &_acc0, &_acc1, &_acc2, &_acc3, _acc0, _acc1, _acc2, _acc3, _m0, _m1, _m2, _m3);
		}
		{
			long _L, _vl = (long)svcntd();
			double _la0[64], _la1[64], _la2[64], _la3[64];
			svst1_f64(svptrue_b64(), _la0, _acc0);
			svst1_f64(svptrue_b64(), _la1, _acc1);
			svst1_f64(svptrue_b64(), _la2, _acc2);
			svst1_f64(svptrue_b64(), _la3, _acc3);
			rqd_set_ui(ret, 0UL);
			for(_L = 0; _L < _vl; _L++)
			{
				double _lane[QDSIZE] = { _la0[_L], _la1[_L], _la2[_L], _la3[_L] };
				rqd_add(ret, ret, _lane);
			}
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	float64x2_t in_a_val[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	float64x2_t in_b_val[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	float64x2_t in_mul[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	float64x2_t sum_neon[QDSIZE];

	// 和を0で初期化
	_bncneon_rqd_set0(sum_neon);

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		// a, b の値をロード
		in_a_val[thread_index][0] = vld1q_f64(&(a->element[0][index]));
		in_a_val[thread_index][1] = vld1q_f64(&(a->element[1][index]));
		in_a_val[thread_index][2] = vld1q_f64(&(a->element[2][index]));
		in_a_val[thread_index][3] = vld1q_f64(&(a->element[3][index]));
		in_b_val[thread_index][0] = vld1q_f64(&(b->element[0][index]));
		in_b_val[thread_index][1] = vld1q_f64(&(b->element[1][index]));
		in_b_val[thread_index][2] = vld1q_f64(&(b->element[2][index]));
		in_b_val[thread_index][3] = vld1q_f64(&(b->element[3][index]));

		// Neon QD乗算
		_bncneon_rqd_mul(in_mul[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

		// 累積加算 (critical sectionで保護)
		#pragma omp critical
		{
			_bncneon_rqd_add(sum_neon, sum_neon, in_mul[thread_index]);
		}
	}
	// fix! 2026-02-25(Wed) T.Kouya
	_bncneon_rqd_sum128d(ret, sum_neon);
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	set0_qd(ret);

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < a->dim; i++)
	{
		thread_index = omp_get_thread_num();

		rqd_mul(tmp[thread_index], get_qdvector_i(a, i), get_qdvector_i(b, i));

	#pragma omp critical
		rqd_add(ret, ret, tmp[thread_index]);

	}
#endif // __AVX2__

	return;
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void _bncomp_row_swap_qdmatrix(QDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
	long int i, j, true_end;
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	int thread_index;

	true_end = (col_end > mat->col_dim) ? mat->col_dim : col_end;

	#pragma omp parallel for private(thread_index)
	for(i = col_start; i < true_end; i++)
	{
		thread_index = omp_get_thread_num();

		rqd_set(tmp[thread_index], get_qdmatrix_ij(mat, row_index0, i));
		set_qdmatrix_ij(mat, row_index0, i, get_qdmatrix_ij(mat, row_index1, i));
		set_qdmatrix_ij(mat, row_index1, i, tmp[thread_index]);
	}
}

//--------
// Matrix
//--------
/* c := a + b */
void _bncomp_add_qdmatrix(QDMatrix c, QDMatrix a, QDMatrix b)
{
	int thread_index;
	long int index, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_qdmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_qdmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[BNCOMP_MAX_NUM_THREADS][QDSIZE], aij4[BNCOMP_MAX_NUM_THREADS][QDSIZE], bij4[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij4[thread_index][0] = _mm256_load_pd(&(a->element[0][index]));
		aij4[thread_index][1] = _mm256_load_pd(&(a->element[1][index]));
		aij4[thread_index][2] = _mm256_load_pd(&(a->element[2][index]));
		aij4[thread_index][3] = _mm256_load_pd(&(a->element[3][index]));
		bij4[thread_index][0] = _mm256_load_pd(&(b->element[0][index]));
		bij4[thread_index][1] = _mm256_load_pd(&(b->element[1][index]));
		bij4[thread_index][2] = _mm256_load_pd(&(b->element[2][index]));
		bij4[thread_index][3] = _mm256_load_pd(&(b->element[3][index]));

		_bncavx2_rqd_add(tmp4[thread_index], aij4[thread_index], bij4[thread_index]);

		_mm256_store_pd(&(c->element[0][index]), tmp4[thread_index][0]);
		_mm256_store_pd(&(c->element[1][index]), tmp4[thread_index][1]); 
		_mm256_store_pd(&(c->element[2][index]), tmp4[thread_index][2]); 
		_mm256_store_pd(&(c->element[3][index]), tmp4[thread_index][3]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[BNCOMP_MAX_NUM_THREADS][QDSIZE], aij8[BNCOMP_MAX_NUM_THREADS][QDSIZE], bij8[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij8[thread_index][0] = _mm512_load_pd(&(a->element[0][index]));
		aij8[thread_index][1] = _mm512_load_pd(&(a->element[1][index]));
		aij8[thread_index][2] = _mm512_load_pd(&(a->element[2][index]));
		aij8[thread_index][3] = _mm512_load_pd(&(a->element[3][index]));
		bij8[thread_index][0] = _mm512_load_pd(&(b->element[0][index]));
		bij8[thread_index][1] = _mm512_load_pd(&(b->element[1][index]));
		bij8[thread_index][2] = _mm512_load_pd(&(b->element[2][index]));
		bij8[thread_index][3] = _mm512_load_pd(&(b->element[3][index]));

		_bncavx512_rqd_add(tmp8[thread_index], aij8[thread_index], bij8[thread_index]);

		_mm512_store_pd(&(c->element[0][index]), tmp8[thread_index][0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[thread_index][1]); 
		_mm512_store_pd(&(c->element[2][index]), tmp8[thread_index][2]); 
		_mm512_store_pd(&(c->element[3][index]), tmp8[thread_index][3]); 
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
			svfloat64_t _a3 = svld1_f64(_pg, &(a->element[3][_ix]));
			svfloat64_t _b0 = svld1_f64(_pg, &(b->element[0][_ix]));
			svfloat64_t _b1 = svld1_f64(_pg, &(b->element[1][_ix]));
			svfloat64_t _b2 = svld1_f64(_pg, &(b->element[2][_ix]));
			svfloat64_t _b3 = svld1_f64(_pg, &(b->element[3][_ix]));
			svfloat64_t _o0, _o1, _o2, _o3;
			_bncsve2_rqd_add(_pg, &_o0, &_o1, &_o2, &_o3, _a0, _a1, _a2, _a3, _b0, _b1, _b2, _b3);
			svst1_f64(_pg, &(c->element[0][_ix]), _o0);
			svst1_f64(_pg, &(c->element[1][_ix]), _o1);
			svst1_f64(_pg, &(c->element[2][_ix]), _o2);
			svst1_f64(_pg, &(c->element[3][_ix]), _o3);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	float64x2_t tmp_neon[BNCOMP_MAX_NUM_THREADS][QDSIZE], aij_neon[BNCOMP_MAX_NUM_THREADS][QDSIZE], bij_neon[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	#pragma omp parallel for private(thread_index) // , tmp_neon, aij_neon, bij_neon)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		aij_neon[thread_index][0] = vld1q_f64(&(a->element[0][index]));
		aij_neon[thread_index][1] = vld1q_f64(&(a->element[1][index]));
		aij_neon[thread_index][2] = vld1q_f64(&(a->element[2][index]));
		aij_neon[thread_index][3] = vld1q_f64(&(a->element[3][index]));
		bij_neon[thread_index][0] = vld1q_f64(&(b->element[0][index]));
		bij_neon[thread_index][1] = vld1q_f64(&(b->element[1][index]));
		bij_neon[thread_index][2] = vld1q_f64(&(b->element[2][index]));
		bij_neon[thread_index][3] = vld1q_f64(&(b->element[3][index]));

		_bncneon_rqd_add(tmp_neon[thread_index], aij_neon[thread_index], bij_neon[thread_index]);

		vst1q_f64(&(c->element[0][index]), tmp_neon[thread_index][0]);
		vst1q_f64(&(c->element[1][index]), tmp_neon[thread_index][1]); 
		vst1q_f64(&(c->element[2][index]), tmp_neon[thread_index][2]); 
		vst1q_f64(&(c->element[3][index]), tmp_neon[thread_index][3]); 
	}
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE], aij[BNCOMP_MAX_NUM_THREADS][QDSIZE], bij[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index++)
	{
		thread_index = omp_get_thread_num();
		
		aij[thread_index][0] = a->element[0][index];
		aij[thread_index][1] = a->element[1][index];
		aij[thread_index][2] = a->element[2][index];
		aij[thread_index][3] = a->element[3][index];
		bij[thread_index][0] = b->element[0][index];
		bij[thread_index][1] = b->element[1][index];
		bij[thread_index][2] = b->element[2][index];
		bij[thread_index][3] = b->element[3][index];

		rqd_add(tmp[thread_index], aij[thread_index], bij[thread_index]);

		c->element[0][index] = tmp[thread_index][0];
		c->element[1][index] = tmp[thread_index][1]; 
		c->element[2][index] = tmp[thread_index][2]; 
		c->element[3][index] = tmp[thread_index][3]; 
	}
#endif // __AVX2__
}

/* c := a - b */
void _bncomp_sub_qdmatrix(QDMatrix c, QDMatrix a, QDMatrix b)
{
	int thread_index;
	long int index, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_sub_qdmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_sub_qdmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[BNCOMP_MAX_NUM_THREADS][QDSIZE], aij4[BNCOMP_MAX_NUM_THREADS][QDSIZE], bij4[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij4[thread_index][0] = _mm256_load_pd(&(a->element[0][index]));
		aij4[thread_index][1] = _mm256_load_pd(&(a->element[1][index]));
		aij4[thread_index][2] = _mm256_load_pd(&(a->element[2][index]));
		aij4[thread_index][3] = _mm256_load_pd(&(a->element[3][index]));
		bij4[thread_index][0] = _mm256_load_pd(&(b->element[0][index]));
		bij4[thread_index][1] = _mm256_load_pd(&(b->element[1][index]));
		bij4[thread_index][2] = _mm256_load_pd(&(b->element[2][index]));
		bij4[thread_index][3] = _mm256_load_pd(&(b->element[3][index]));

		_bncavx2_rqd_sub(tmp4[thread_index], aij4[thread_index], bij4[thread_index]);

		_mm256_store_pd(&(c->element[0][index]), tmp4[thread_index][0]);
		_mm256_store_pd(&(c->element[1][index]), tmp4[thread_index][1]); 
		_mm256_store_pd(&(c->element[2][index]), tmp4[thread_index][2]); 
		_mm256_store_pd(&(c->element[3][index]), tmp4[thread_index][3]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[BNCOMP_MAX_NUM_THREADS][QDSIZE], aij8[BNCOMP_MAX_NUM_THREADS][QDSIZE], bij8[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij8[thread_index][0] = _mm512_load_pd(&(a->element[0][index]));
		aij8[thread_index][1] = _mm512_load_pd(&(a->element[1][index]));
		aij8[thread_index][2] = _mm512_load_pd(&(a->element[2][index]));
		aij8[thread_index][3] = _mm512_load_pd(&(a->element[3][index]));
		bij8[thread_index][0] = _mm512_load_pd(&(b->element[0][index]));
		bij8[thread_index][1] = _mm512_load_pd(&(b->element[1][index]));
		bij8[thread_index][2] = _mm512_load_pd(&(b->element[2][index]));
		bij8[thread_index][3] = _mm512_load_pd(&(b->element[3][index]));

		_bncavx512_rqd_sub(tmp8[thread_index], aij8[thread_index], bij8[thread_index]);

		_mm512_store_pd(&(c->element[0][index]), tmp8[thread_index][0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[thread_index][1]); 
		_mm512_store_pd(&(c->element[2][index]), tmp8[thread_index][2]); 
		_mm512_store_pd(&(c->element[3][index]), tmp8[thread_index][3]); 
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
			svfloat64_t _a3 = svld1_f64(_pg, &(a->element[3][_ix]));
			svfloat64_t _b0 = svld1_f64(_pg, &(b->element[0][_ix]));
			svfloat64_t _b1 = svld1_f64(_pg, &(b->element[1][_ix]));
			svfloat64_t _b2 = svld1_f64(_pg, &(b->element[2][_ix]));
			svfloat64_t _b3 = svld1_f64(_pg, &(b->element[3][_ix]));
			_b0 = svneg_f64_x(_pg, _b0);
			_b1 = svneg_f64_x(_pg, _b1);
			_b2 = svneg_f64_x(_pg, _b2);
			_b3 = svneg_f64_x(_pg, _b3);
			svfloat64_t _o0, _o1, _o2, _o3;
			_bncsve2_rqd_add(_pg, &_o0, &_o1, &_o2, &_o3, _a0, _a1, _a2, _a3, _b0, _b1, _b2, _b3);
			svst1_f64(_pg, &(c->element[0][_ix]), _o0);
			svst1_f64(_pg, &(c->element[1][_ix]), _o1);
			svst1_f64(_pg, &(c->element[2][_ix]), _o2);
			svst1_f64(_pg, &(c->element[3][_ix]), _o3);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
// Arm Neon版
	float64x2_t tmp_neon[BNCOMP_MAX_NUM_THREADS][QDSIZE], aij_neon[BNCOMP_MAX_NUM_THREADS][QDSIZE], bij_neon[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	// 並列化の閾値判定（行数で判定）
	//if(row_dim >= PARALLEL_THRESHOLD_MATRIX)
	{
		#pragma omp parallel for private(tmp_neon, aij_neon, bij_neon)
		for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
		{
			thread_index = omp_get_thread_num();

			aij_neon[thread_index][0] = vld1q_f64(&(a->element[0][index]));
			aij_neon[thread_index][1] = vld1q_f64(&(a->element[1][index]));
			aij_neon[thread_index][2] = vld1q_f64(&(a->element[2][index]));
			aij_neon[thread_index][3] = vld1q_f64(&(a->element[3][index]));
			bij_neon[thread_index][0] = vld1q_f64(&(b->element[0][index]));
			bij_neon[thread_index][1] = vld1q_f64(&(b->element[1][index]));
			bij_neon[thread_index][2] = vld1q_f64(&(b->element[2][index]));
			bij_neon[thread_index][3] = vld1q_f64(&(b->element[3][index]));

			_bncneon_rqd_sub(tmp_neon[thread_index], aij_neon[thread_index], bij_neon[thread_index]);

			vst1q_f64(&(c->element[0][index]), tmp_neon[thread_index][0]);
			vst1q_f64(&(c->element[1][index]), tmp_neon[thread_index][1]); 
			vst1q_f64(&(c->element[2][index]), tmp_neon[thread_index][2]); 
			vst1q_f64(&(c->element[3][index]), tmp_neon[thread_index][3]); 
		}
	}
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE], aij[BNCOMP_MAX_NUM_THREADS][QDSIZE], bij[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index++)
	{
		thread_index = omp_get_thread_num();
		
		aij[thread_index][0] = a->element[0][index];
		aij[thread_index][1] = a->element[1][index];
		aij[thread_index][2] = a->element[2][index];
		aij[thread_index][3] = a->element[3][index];
		bij[thread_index][0] = b->element[0][index];
		bij[thread_index][1] = b->element[1][index];
		bij[thread_index][2] = b->element[2][index];
		bij[thread_index][3] = b->element[3][index];

		rqd_sub(tmp[thread_index], aij[thread_index], bij[thread_index]);

		c->element[0][index] = tmp[thread_index][0];
		c->element[1][index] = tmp[thread_index][1]; 
		c->element[2][index] = tmp[thread_index][2]; 
		c->element[3][index] = tmp[thread_index][3]; 
	}
#endif // __AVX2__
}

/* c := sc * a */
void _bncomp_cmul_qdmatrix(QDMatrix c, double sc[QDSIZE], QDMatrix a)
{
	long int i, j, index, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim;
	int thread_index;

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_qdmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_qdmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d tmp4[BNCOMP_MAX_NUM_THREADS][QDSIZE], aij4[BNCOMP_MAX_NUM_THREADS][QDSIZE], sc4[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		sc4[thread_index][0] = _mm256_set1_pd(sc[0]);
		sc4[thread_index][1] = _mm256_set1_pd(sc[1]);
		sc4[thread_index][2] = _mm256_set1_pd(sc[2]);
		sc4[thread_index][3] = _mm256_set1_pd(sc[3]);
	}
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij4[thread_index][0] = _mm256_load_pd(&(a->element[0][index]));
		aij4[thread_index][1] = _mm256_load_pd(&(a->element[1][index]));
		aij4[thread_index][2] = _mm256_load_pd(&(a->element[2][index]));
		aij4[thread_index][3] = _mm256_load_pd(&(a->element[3][index]));

		_bncavx2_rqd_mul(tmp4[thread_index], sc4[thread_index], aij4[thread_index]);

		_mm256_store_pd(&(c->element[0][index]), tmp4[thread_index][0]);
		_mm256_store_pd(&(c->element[1][index]), tmp4[thread_index][1]); 
		_mm256_store_pd(&(c->element[2][index]), tmp4[thread_index][2]); 
		_mm256_store_pd(&(c->element[3][index]), tmp4[thread_index][3]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d tmp8[BNCOMP_MAX_NUM_THREADS][QDSIZE], aij8[BNCOMP_MAX_NUM_THREADS][QDSIZE], sc8[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		sc8[thread_index][0] = _mm512_set1_pd(sc[0]);
		sc8[thread_index][1] = _mm512_set1_pd(sc[1]);
		sc8[thread_index][2] = _mm512_set1_pd(sc[2]);
		sc8[thread_index][3] = _mm512_set1_pd(sc[3]);
	}
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij8[thread_index][0] = _mm512_load_pd(&(a->element[0][index]));
		aij8[thread_index][1] = _mm512_load_pd(&(a->element[1][index]));
		aij8[thread_index][2] = _mm512_load_pd(&(a->element[2][index]));
		aij8[thread_index][3] = _mm512_load_pd(&(a->element[3][index]));

		_bncavx512_rqd_mul(tmp8[thread_index], sc8[thread_index], aij8[thread_index]);

		_mm512_store_pd(&(c->element[0][index]), tmp8[thread_index][0]);
		_mm512_store_pd(&(c->element[1][index]), tmp8[thread_index][1]); 
		_mm512_store_pd(&(c->element[2][index]), tmp8[thread_index][2]); 
		_mm512_store_pd(&(c->element[3][index]), tmp8[thread_index][3]); 
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntd(); long _N = a->real_row_dim * a->real_col_dim; long _ix;
		svfloat64_t _v0 = svdup_n_f64(sc[0]);
		svfloat64_t _v1 = svdup_n_f64(sc[1]);
		svfloat64_t _v2 = svdup_n_f64(sc[2]);
		svfloat64_t _v3 = svdup_n_f64(sc[3]);
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b64_s64((int64_t)_ix, (int64_t)_N);
			svfloat64_t _a0 = svld1_f64(_pg, &(a->element[0][_ix]));
			svfloat64_t _a1 = svld1_f64(_pg, &(a->element[1][_ix]));
			svfloat64_t _a2 = svld1_f64(_pg, &(a->element[2][_ix]));
			svfloat64_t _a3 = svld1_f64(_pg, &(a->element[3][_ix]));
			svfloat64_t _o0, _o1, _o2, _o3;
			_bncsve2_rqd_mul(_pg, &_o0, &_o1, &_o2, &_o3, _v0, _v1, _v2, _v3, _a0, _a1, _a2, _a3);
			svst1_f64(_pg, &(c->element[0][_ix]), _o0);
			svst1_f64(_pg, &(c->element[1][_ix]), _o1);
			svst1_f64(_pg, &(c->element[2][_ix]), _o2);
			svst1_f64(_pg, &(c->element[3][_ix]), _o3);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	float64x2_t tmp_neon[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	float64x2_t aij_neon[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	float64x2_t sc_neon[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		// スカラーをブロードキャスト
		_bncneon_rqd_set1_qd(sc_neon[thread_index], sc);
	}
	
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		// ロード
		aij_neon[thread_index][0] = vld1q_f64(&(a->element[0][index]));
		aij_neon[thread_index][1] = vld1q_f64(&(a->element[1][index]));
		aij_neon[thread_index][2] = vld1q_f64(&(a->element[2][index]));
		aij_neon[thread_index][3] = vld1q_f64(&(a->element[3][index]));

		// Neon QD乗算
		_bncneon_rqd_mul(tmp_neon[thread_index], sc_neon[thread_index], aij_neon[thread_index]);

		// ストア
		vst1q_f64(&(c->element[0][index]), tmp_neon[thread_index][0]);
		vst1q_f64(&(c->element[1][index]), tmp_neon[thread_index][1]); 
		vst1q_f64(&(c->element[2][index]), tmp_neon[thread_index][2]); 
		vst1q_f64(&(c->element[3][index]), tmp_neon[thread_index][3]); 
	}
#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE], aij[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index++)
	{
		thread_index = omp_get_thread_num();
		
		aij[thread_index][0] = a->element[0][index];
		aij[thread_index][1] = a->element[1][index];
		aij[thread_index][2] = a->element[2][index];
		aij[thread_index][3] = a->element[3][index];

		rqd_mul(tmp[thread_index], sc, aij[thread_index]);

		c->element[0][index] = tmp[thread_index][0];
		c->element[1][index] = tmp[thread_index][1]; 
		c->element[2][index] = tmp[thread_index][2]; 
		c->element[3][index] = tmp[thread_index][3]; 
	}
#endif // __AVX2__
}

/* c = a * b */
void _bncomp_mul_qdmatrix(QDMatrix ret, QDMatrix a, QDMatrix b)
{
	int thread_num, thread_index;
	long int i, j, k;
	long row_dim, col_dim, mid_dim;

	/* dimension check */
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_qdmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret->row_dim, ret->col_dim, a->row_dim, a->col_dim, b->row_dim, b->col_dim);
		return;
	}
	row_dim = a->row_dim;
	col_dim = b->col_dim;
	mid_dim = a->col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long real_row_dim, real_col_dim, real_mid_dim;
	double cijval[BNCOMP_MAX_NUM_THREADS][4][QDSIZE];
    __m256d cij[BNCOMP_MAX_NUM_THREADS][QDSIZE], aik[BNCOMP_MAX_NUM_THREADS][QDSIZE], bkj[BNCOMP_MAX_NUM_THREADS][QDSIZE], tmp_mul[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		//set0_dd(tmp[thread_index]);
	}

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

//#if 0 //
	#pragma omp parallel for private(thread_index, i, j, k, aik, bkj, cij, cijval, tmp_mul)
    for(i = 0; i < real_row_dim; i++)
    {
		thread_index = omp_get_thread_num();

        for(j = 0; j < real_col_dim; j++)
        {
            cij[thread_index][0] = _mm256_setzero_pd();
            cij[thread_index][1] = _mm256_setzero_pd();
            cij[thread_index][2] = _mm256_setzero_pd();
            cij[thread_index][3] = _mm256_setzero_pd();

            for(k = 0; k < real_mid_dim; k += _BNC_D_WIDTH)
            {
                aik[thread_index][0] = _mm256_load_pd(&(a->element[0][i * real_mid_dim + k]));
                aik[thread_index][1] = _mm256_load_pd(&(a->element[1][i * real_mid_dim + k]));
                aik[thread_index][2] = _mm256_load_pd(&(a->element[2][i * real_mid_dim + k]));
                aik[thread_index][3] = _mm256_load_pd(&(a->element[3][i * real_mid_dim + k]));

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
				bkj[thread_index][3] = _mm256_set_pd(
                    b->element[3][(k + 3) * real_col_dim + j],
                    b->element[3][(k + 2) * real_col_dim + j],
                    b->element[3][(k + 1) * real_col_dim + j],
                    b->element[3][(k    ) * real_col_dim + j]
                );

                _bncavx2_rqd_mul(tmp_mul[thread_index], aik[thread_index], bkj[thread_index]);
                _bncavx2_rqd_add(cij[thread_index], cij[thread_index], tmp_mul[thread_index]);
            }

            cijval[thread_index][0][0] = cij[thread_index][0][0];
			cijval[thread_index][0][1] = cij[thread_index][1][0];
			cijval[thread_index][0][2] = cij[thread_index][2][0];
			cijval[thread_index][0][3] = cij[thread_index][3][0];

            cijval[thread_index][1][0] = cij[thread_index][0][1];
			cijval[thread_index][1][1] = cij[thread_index][1][1];
			cijval[thread_index][1][2] = cij[thread_index][2][1];
			cijval[thread_index][1][3] = cij[thread_index][3][1];

            cijval[thread_index][2][0] = cij[thread_index][0][2];
			cijval[thread_index][2][1] = cij[thread_index][1][2];
			cijval[thread_index][2][2] = cij[thread_index][2][2];
			cijval[thread_index][2][3] = cij[thread_index][3][2];

            cijval[thread_index][3][0] = cij[thread_index][0][3];
			cijval[thread_index][3][1] = cij[thread_index][1][3];
			cijval[thread_index][3][2] = cij[thread_index][2][3];
			cijval[thread_index][3][3] = cij[thread_index][3][3];

            rqd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][1]);
            rqd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][2]);
            rqd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][3]);

            ret->element[0][i * real_col_dim + j] = cijval[thread_index][0][0];
            ret->element[1][i * real_col_dim + j] = cijval[thread_index][0][1];
            ret->element[2][i * real_col_dim + j] = cijval[thread_index][0][2];
            ret->element[3][i * real_col_dim + j] = cijval[thread_index][0][3];
        }
    }
//#else // 0
#if 0
	_bncomp_set0_qdmatrix(ret);
	// From codes of Rgemm
	#pragma omp parallel for private(thread_index, k, i)
	//for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
	for(j = 0; j < col_dim; j++)
	{
		thread_index = omp_get_thread_num();
		for(k = 0; k < real_mid_dim; k +=  _BNC_D_WIDTH)
		{
			//rqd_set(b_kj[thread_index], get_qdmatrix_ij(b, k, j));
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
			bkj[thread_index][3] = _mm256_set_pd(
                b->element[3][(k + 3) * real_col_dim + j],
                b->element[3][(k + 2) * real_col_dim + j],
                b->element[3][(k + 1) * real_col_dim + j],
                b->element[3][(k    ) * real_col_dim + j]
            );
			for(i = 0; i < row_dim; i++)
			{
				//rqd_mul(tmp[thread_index], get_qdmatrix_ij(a, i, k), b_kj[thread_index]);
                aik[thread_index][0] = _mm256_load_pd(&(a->element[0][i * real_mid_dim + k]));
                aik[thread_index][1] = _mm256_load_pd(&(a->element[1][i * real_mid_dim + k]));
                aik[thread_index][2] = _mm256_load_pd(&(a->element[2][i * real_mid_dim + k]));
                aik[thread_index][3] = _mm256_load_pd(&(a->element[3][i * real_mid_dim + k]));
                _bncavx2_rqd_mul(tmp_mul[thread_index], aik[thread_index], bkj[thread_index]);

				cijval[thread_index][0][0] = tmp_mul[thread_index][0][0];
				cijval[thread_index][0][1] = tmp_mul[thread_index][1][0];
				cijval[thread_index][0][2] = tmp_mul[thread_index][2][0];
				cijval[thread_index][0][3] = tmp_mul[thread_index][3][0];

				cijval[thread_index][1][0] = tmp_mul[thread_index][0][1];
				cijval[thread_index][1][1] = tmp_mul[thread_index][1][1];
				cijval[thread_index][1][2] = tmp_mul[thread_index][2][1];
				cijval[thread_index][1][3] = tmp_mul[thread_index][3][1];

				cijval[thread_index][2][0] = tmp_mul[thread_index][0][2];
				cijval[thread_index][2][1] = tmp_mul[thread_index][1][2];
				cijval[thread_index][2][2] = tmp_mul[thread_index][2][2];
				cijval[thread_index][2][3] = tmp_mul[thread_index][3][2];

				cijval[thread_index][3][0] = tmp_mul[thread_index][0][3];
				cijval[thread_index][3][1] = tmp_mul[thread_index][1][3];
				cijval[thread_index][3][2] = tmp_mul[thread_index][2][3];
				cijval[thread_index][3][3] = tmp_mul[thread_index][3][3];

				//rqd_add(get_qdmatrix_ij(ret, i, j), get_qdmatrix_ij(ret, i, j), tmp[thread_index]);
				rqd_add(get_qdmatrix_ij(ret, i, j), get_qdmatrix_ij(ret, i, j), cijval[thread_index][0]);
				rqd_add(get_qdmatrix_ij(ret, i, j), get_qdmatrix_ij(ret, i, j), cijval[thread_index][1]);
				rqd_add(get_qdmatrix_ij(ret, i, j), get_qdmatrix_ij(ret, i, j), cijval[thread_index][2]);
				rqd_add(get_qdmatrix_ij(ret, i, j), get_qdmatrix_ij(ret, i, j), cijval[thread_index][3]);
			}
		}
	}
	//printf("end(%ld, %ld)\n", ret->row_dim, ret->col_dim);#endif // 0
#endif //0
#elif defined(__AVX512F__) // __AVX512F__
	long real_row_dim, real_col_dim, real_mid_dim;
	double cijval[BNCOMP_MAX_NUM_THREADS][8][QDSIZE];
    __m512d cij[BNCOMP_MAX_NUM_THREADS][QDSIZE], aik[BNCOMP_MAX_NUM_THREADS][QDSIZE], bkj[BNCOMP_MAX_NUM_THREADS][QDSIZE], tmp_mul[BNCOMP_MAX_NUM_THREADS][QDSIZE];

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
            cij[thread_index][2] = _mm512_setzero_pd();
            cij[thread_index][3] = _mm512_setzero_pd();

            for(k = 0; k < real_mid_dim; k += _BNC_D_WIDTH)
            {
                aik[thread_index][0] = _mm512_load_pd(&(a->element[0][i * real_mid_dim + k]));
                aik[thread_index][1] = _mm512_load_pd(&(a->element[1][i * real_mid_dim + k]));
                aik[thread_index][2] = _mm512_load_pd(&(a->element[2][i * real_mid_dim + k]));
                aik[thread_index][3] = _mm512_load_pd(&(a->element[3][i * real_mid_dim + k]));

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
                bkj[thread_index][3] = _mm512_set_pd(
                    b->element[3][(k + 7) * real_col_dim + j],
                    b->element[3][(k + 6) * real_col_dim + j],
                    b->element[3][(k + 5) * real_col_dim + j],
                    b->element[3][(k + 4) * real_col_dim + j],
                    b->element[3][(k + 3) * real_col_dim + j],
                    b->element[3][(k + 2) * real_col_dim + j],
                    b->element[3][(k + 1) * real_col_dim + j],
                    b->element[3][(k    ) * real_col_dim + j]
                );

                _bncavx512_rqd_mul(tmp_mul[thread_index], aik[thread_index], bkj[thread_index]);
                _bncavx512_rqd_add(cij[thread_index], cij[thread_index], tmp_mul[thread_index]);
            }

            cijval[thread_index][0][0] = cij[thread_index][0][0];
			cijval[thread_index][0][1] = cij[thread_index][1][0];
			cijval[thread_index][0][2] = cij[thread_index][2][0];
			cijval[thread_index][0][3] = cij[thread_index][3][0];

            cijval[thread_index][1][0] = cij[thread_index][0][1];
			cijval[thread_index][1][1] = cij[thread_index][1][1];
			cijval[thread_index][1][2] = cij[thread_index][2][1];
			cijval[thread_index][1][3] = cij[thread_index][3][1];

            cijval[thread_index][2][0] = cij[thread_index][0][2];
			cijval[thread_index][2][1] = cij[thread_index][1][2];
			cijval[thread_index][2][2] = cij[thread_index][2][2];
			cijval[thread_index][2][3] = cij[thread_index][3][2];

            cijval[thread_index][3][0] = cij[thread_index][0][3];
			cijval[thread_index][3][1] = cij[thread_index][1][3];
			cijval[thread_index][3][2] = cij[thread_index][2][3];
			cijval[thread_index][3][3] = cij[thread_index][3][3];

            cijval[thread_index][4][0] = cij[thread_index][0][4];
			cijval[thread_index][4][1] = cij[thread_index][1][4];
			cijval[thread_index][4][2] = cij[thread_index][2][4];
			cijval[thread_index][4][3] = cij[thread_index][3][4];

            cijval[thread_index][5][0] = cij[thread_index][0][5];
			cijval[thread_index][5][1] = cij[thread_index][1][5];
			cijval[thread_index][5][2] = cij[thread_index][2][5];
			cijval[thread_index][5][3] = cij[thread_index][3][5];

            cijval[thread_index][6][0] = cij[thread_index][0][6];
			cijval[thread_index][6][1] = cij[thread_index][1][6];
			cijval[thread_index][6][2] = cij[thread_index][2][6];
			cijval[thread_index][6][3] = cij[thread_index][3][6];

            cijval[thread_index][7][0] = cij[thread_index][0][7];
			cijval[thread_index][7][1] = cij[thread_index][1][7];
			cijval[thread_index][7][2] = cij[thread_index][2][7];
			cijval[thread_index][7][3] = cij[thread_index][3][7];

            rqd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][1]);
            rqd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][2]);
            rqd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][3]);
            rqd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][4]);
            rqd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][5]);
            rqd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][6]);
            rqd_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][7]);

            ret->element[0][i * real_col_dim + j] = cijval[thread_index][0][0];
            ret->element[1][i * real_col_dim + j] = cijval[thread_index][0][1];
            ret->element[2][i * real_col_dim + j] = cijval[thread_index][0][2];
            ret->element[3][i * real_col_dim + j] = cijval[thread_index][0][3];
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
				svfloat64_t cij0, cij1, cij2, cij3;
				_bncsve2_rqd_set0(&cij0, &cij1, &cij2, &cij3);
				for(k = 0; k < real_mid_dim; k++){
					svfloat64_t aik0 = svdup_n_f64(a->element[0][i*real_mid_dim + k]);
					svfloat64_t aik1 = svdup_n_f64(a->element[1][i*real_mid_dim + k]);
					svfloat64_t aik2 = svdup_n_f64(a->element[2][i*real_mid_dim + k]);
					svfloat64_t aik3 = svdup_n_f64(a->element[3][i*real_mid_dim + k]);
					svfloat64_t bkj0 = svld1_f64(pg, &(b->element[0][k*real_col_dim + j]));
					svfloat64_t bkj1 = svld1_f64(pg, &(b->element[1][k*real_col_dim + j]));
					svfloat64_t bkj2 = svld1_f64(pg, &(b->element[2][k*real_col_dim + j]));
					svfloat64_t bkj3 = svld1_f64(pg, &(b->element[3][k*real_col_dim + j]));
					svfloat64_t t0, t1, t2, t3;
					_bncsve2_rqd_mul(pg, &t0, &t1, &t2, &t3,
					                 aik0, aik1, aik2, aik3,
					                 bkj0, bkj1, bkj2, bkj3);
					_bncsve2_rqd_add(pg, &cij0, &cij1, &cij2, &cij3,
					                 cij0, cij1, cij2, cij3,
					                 t0, t1, t2, t3);
				}
				svst1_f64(pg, &(ret->element[0][i*real_col_dim + j]), cij0);
				svst1_f64(pg, &(ret->element[1][i*real_col_dim + j]), cij1);
				svst1_f64(pg, &(ret->element[2][i*real_col_dim + j]), cij2);
				svst1_f64(pg, &(ret->element[3][i*real_col_dim + j]), cij3);
			}
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	long real_row_dim, real_col_dim, real_mid_dim;

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

	// 行列乗算は計算量が大きいので、並列化の効果が出やすい
	// ただし、小さい行列では逐次実行の方が速い
	//if(row_dim >= PARALLEL_THRESHOLD_MATRIX / 2)
	{
		// 並列版
		#pragma omp parallel for private(thread_index, i, j, k) schedule(dynamic)
		for(i = 0; i < real_row_dim; i++)
		{
			thread_index = omp_get_thread_num();
			double cijval[2][QDSIZE];
			float64x2_t cij[QDSIZE], aik[QDSIZE], bkj[QDSIZE], tmp_mul[QDSIZE];

			for(j = 0; j < real_col_dim; j++)
			{
				cij[0] = vdupq_n_f64(0.0);
				cij[1] = vdupq_n_f64(0.0);
				cij[2] = vdupq_n_f64(0.0);
				cij[3] = vdupq_n_f64(0.0);

				for(k = 0; k < real_mid_dim; k += _BNC_D_WIDTH)
				{
					aik[0] = vld1q_f64(&(a->element[0][i * real_mid_dim + k]));
					aik[1] = vld1q_f64(&(a->element[1][i * real_mid_dim + k]));
					aik[2] = vld1q_f64(&(a->element[2][i * real_mid_dim + k]));
					aik[3] = vld1q_f64(&(a->element[3][i * real_mid_dim + k]));

					bkj[0] = vsetq_lane_f64(
						b->element[0][(k    ) * real_col_dim + j],
						vsetq_lane_f64(b->element[0][(k + 1) * real_col_dim + j], vdupq_n_f64(0.0), 1),
						0
					);
					bkj[1] = vsetq_lane_f64(
						b->element[1][(k    ) * real_col_dim + j],
						vsetq_lane_f64(b->element[1][(k + 1) * real_col_dim + j], vdupq_n_f64(0.0), 1),
						0
					);
					bkj[2] = vsetq_lane_f64(
						b->element[2][(k    ) * real_col_dim + j],
						vsetq_lane_f64(b->element[2][(k + 1) * real_col_dim + j], vdupq_n_f64(0.0), 1),
						0
					);
					bkj[3] = vsetq_lane_f64(
						b->element[3][(k    ) * real_col_dim + j],
						vsetq_lane_f64(b->element[3][(k + 1) * real_col_dim + j], vdupq_n_f64(0.0), 1),
						0
					);

					_bncneon_rqd_mul(tmp_mul, aik, bkj);
					_bncneon_rqd_add(cij, cij, tmp_mul);
				}

				cijval[0][0] = vgetq_lane_f64(cij[0], 0);
				cijval[0][1] = vgetq_lane_f64(cij[1], 0);
				cijval[0][2] = vgetq_lane_f64(cij[2], 0);
				cijval[0][3] = vgetq_lane_f64(cij[3], 0);

				cijval[1][0] = vgetq_lane_f64(cij[0], 1);
				cijval[1][1] = vgetq_lane_f64(cij[1], 1);
				cijval[1][2] = vgetq_lane_f64(cij[2], 1);
				cijval[1][3] = vgetq_lane_f64(cij[3], 1);

				rqd_add(cijval[0], cijval[0], cijval[1]);

				ret->element[0][i * real_col_dim + j] = cijval[0][0];
				ret->element[1][i * real_col_dim + j] = cijval[0][1];
				ret->element[2][i * real_col_dim + j] = cijval[0][2];
				ret->element[3][i * real_col_dim + j] = cijval[0][3];
			}
		}
	}
#else // __AVX2__
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE], ret_ij[BNCOMP_MAX_NUM_THREADS][QDSIZE], b_kj[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_qd(tmp[thread_index]);
	}

	//printf("Non SIMD mul_qdmatrix(%ld, %ld)\n", ret->row_dim, ret->col_dim);
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = a->col_dim;

//#if 0
	#pragma omp parallel for private(thread_index, i, j, k, ret_ij, tmp)
	for(i = 0; i < row_dim; i++)
	{

		thread_index = omp_get_thread_num();

		//#pragma omp parallel for private(thread_index, j, k) shared(a, b, ret, tmp, mid_dim, col_dim)
		for(j = 0; j < col_dim; j++)
		{
			//thread_index = omp_get_thread_num();
			//#pragma omp critical
			rqd_set0(ret_ij[thread_index]);
			for(k = 0; k < mid_dim; k++)
			{
				//#pragma omp critical
				//{
				rqd_mul(tmp[thread_index], get_qdmatrix_ij(a, i, k), get_qdmatrix_ij(b, k, j));
				// fixed! 2024-02-22 T.Kouya
				#pragma omp critical
					rqd_add(ret_ij[thread_index], tmp[thread_index], ret_ij[thread_index]);
				//}
			}
			
			set_qdmatrix_ij(ret, i, j, ret_ij[thread_index]);
		}
	}	//printf("end(%ld, %ld)\n", ret->row_dim, ret->col_dim);
//#else // 0
/*	_bncomp_set0_qdmatrix(ret);
	// From codes of Rgemm
	#pragma omp parallel for private(thread_index, k, i)
	for(j = 0; j < col_dim; j++)
	{
		thread_index = omp_get_thread_num();
		for(k = 0; k < mid_dim; k++)
		{
			rqd_set(b_kj[thread_index], get_qdmatrix_ij(b, k, j));
			for(i = 0; i < row_dim; i++)
			{
				rqd_mul(tmp[thread_index], get_qdmatrix_ij(a, i, k), b_kj[thread_index]);
				rqd_add(get_qdmatrix_ij(ret, i, j), get_qdmatrix_ij(ret, i, j), tmp[thread_index]);
			}
		}
	}
	//printf("end(%ld, %ld)\n", ret->row_dim, ret->col_dim);#endif // 0
*/
//#endif //0
#endif // __AVX2__

}

/* c := a */
void _bncomp_subst_qdmatrix(QDMatrix c, QDMatrix a)
{
	long int index;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_qdmatrix\n");
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
		_mm256_store_pd(&(c->element[3][index]), _mm256_load_pd(&(a->element[3][index])));
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
		_mm512_store_pd(&(c->element[3][index]), _mm512_load_pd(&(a->element[3][index])));
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
			svst1_f64(_pg, &(c->element[3][_ix]), svld1_f64(_pg, &(a->element[3][_ix])));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	long int real_total_dim;

	real_total_dim = c->real_row_dim * c->real_col_dim;

	#pragma omp parallel for
	for(index = 0; index < real_total_dim; index += _BNC_D_WIDTH)
	{
		float64x2_t tmp0 = vld1q_f64(&(a->element[0][index]));
		float64x2_t tmp1 = vld1q_f64(&(a->element[1][index]));
		float64x2_t tmp2 = vld1q_f64(&(a->element[2][index]));
		float64x2_t tmp3 = vld1q_f64(&(a->element[3][index]));
		
		vst1q_f64(&(c->element[0][index]), tmp0);
		vst1q_f64(&(c->element[1][index]), tmp1);
		vst1q_f64(&(c->element[2][index]), tmp2);
		vst1q_f64(&(c->element[3][index]), tmp3);
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
		c->element[3][index] = a->element[3][index];
	}

	//#pragma omp parallel for
	//for(index = 0; index < QDSIZE; index++)
	//	memcpy((void *)(c->element[index]), (void *)(a->element[index]), (size_t)(sizeof(double) * total_dim)); 

#endif // AVX2
}

/* c := I */
void _bncomp_setI_qdmatrix(QDMatrix c)
{
	long int i, real_total_dim;
	double tmp0[QDSIZE], tmp1[QDSIZE];

	rqd_set_ui(tmp0, 0UL);
	rqd_set_ui(tmp1, 1UL);

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
		_mm256_store_pd(&(c->element[3][i]), zero4);
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
		_mm512_store_pd(&(c->element[3][i]), zero8);
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
			svst1_f64(_pg, &(c->element[3][_ix]), _z);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	float64x2_t zero_neon;

	zero_neon = vdupq_n_f64(0.0);
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		vst1q_f64(&(c->element[0][i]), zero_neon);
		vst1q_f64(&(c->element[1][i]), zero_neon);
		vst1q_f64(&(c->element[2][i]), zero_neon);
		vst1q_f64(&(c->element[3][i]), zero_neon);
	}
#else // others
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i++)
	{
		c->element[0][i] = 0.0;
		c->element[1][i] = 0.0;	
		c->element[2][i] = 0.0;	
		c->element[3][i] = 0.0;	
	}
#endif // __AVX2__

	rqd_set_ui(tmp1, 1UL);

	#pragma omp parallel for
	for(i = 0; i < c->row_dim; i++)
	{
		if(i < c->col_dim)
			set_qdmatrix_ij(c, i, i, tmp1);
	}

}

// set a zero matrix
//void set0_qdmatrix(QDMatrix mat)
void _bncomp_set0_qdmatrix(QDMatrix mat)
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
		_mm256_store_pd(&(mat->element[3][i]), zero4);
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
		_mm512_store_pd(&(mat->element[3][i]), zero8);
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
			svst1_f64(_pg, &(mat->element[3][_ix]), _z);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	float64x2_t zero_neon;

	zero_neon = vdupq_n_f64(0.0);
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_D_WIDTH)
	{
		vst1q_f64(&(mat->element[0][i]), zero_neon);
		vst1q_f64(&(mat->element[1][i]), zero_neon);
		vst1q_f64(&(mat->element[2][i]), zero_neon);
		vst1q_f64(&(mat->element[3][i]), zero_neon);
	}
#else // others
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i++)
	{
		mat->element[0][i] = 0.0;
		mat->element[1][i] = 0.0;	
		mat->element[2][i] = 0.0;	
		mat->element[3][i] = 0.0;	
	}
#endif // __AVX2__
}

/* v := a * vb */
void _bncomp_mul_qdmatrix_qdvec(QDVector v, QDMatrix a, QDVector vb)
{
	long int i, j, row_dim;
	int thread_index;
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE], tmp1[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	/* Check Dimension */
	//if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	if((v->dim < a->row_dim) || (vb->dim < a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_qdmatrix_qdvec\n");
		return;
	}

	row_dim = a->row_dim;

// SIMD
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int ij_index, real_row_dim, real_col_dim;
	__m256d tmp4[BNCOMP_MAX_NUM_THREADS][QDSIZE], tmp1_4[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	__m256d aij4[BNCOMP_MAX_NUM_THREADS][QDSIZE], vbj4[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j, ij_index)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		//rqd_set_ui(tmp, 0UL);
		_bncavx2_set0_qd(tmp4[thread_index]);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij4[thread_index][0] = _mm256_load_pd(&(a->element[0][ij_index]));
			aij4[thread_index][1] = _mm256_load_pd(&(a->element[1][ij_index]));
			aij4[thread_index][2] = _mm256_load_pd(&(a->element[2][ij_index]));
			aij4[thread_index][3] = _mm256_load_pd(&(a->element[3][ij_index]));
			vbj4[thread_index][0] = _mm256_load_pd(&(vb->element[0][j]));
			vbj4[thread_index][1] = _mm256_load_pd(&(vb->element[1][j]));
			vbj4[thread_index][2] = _mm256_load_pd(&(vb->element[2][j]));
			vbj4[thread_index][3] = _mm256_load_pd(&(vb->element[3][j]));

			//rqd_mul(tmp1, get_qdmatrix_ij(a, i, j), get_qdvector_i(vb, j));
			//rqd_add(tmp, tmp, tmp1);
			_bncavx2_rqd_mul(tmp1_4[thread_index], aij4[thread_index], vbj4[thread_index]);
			_bncavx2_rqd_add(tmp4[thread_index], tmp4[thread_index], tmp1_4[thread_index]);
		}
		//set_qdvector_i(v, i, tmp);
		_bncavx2_rqd_sum256d(tmp[thread_index], tmp4[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
		v->element[2][i] = tmp[thread_index][2];
		v->element[3][i] = tmp[thread_index][3];
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int ij_index, real_col_dim, real_row_dim;
	__m512d tmp8[BNCOMP_MAX_NUM_THREADS][QDSIZE], tmp1_8[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	__m512d aij8[BNCOMP_MAX_NUM_THREADS][QDSIZE], vbj8[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j, ij_index)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		//rqd_set_ui(tmp, 0UL);
		_bncavx512_set0_qd(tmp8[thread_index]);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij8[thread_index][0] = _mm512_load_pd(&(a->element[0][ij_index]));
			aij8[thread_index][1] = _mm512_load_pd(&(a->element[1][ij_index]));
			aij8[thread_index][2] = _mm512_load_pd(&(a->element[2][ij_index]));
			aij8[thread_index][3] = _mm512_load_pd(&(a->element[3][ij_index]));
			vbj8[thread_index][0] = _mm512_load_pd(&(vb->element[0][j]));
			vbj8[thread_index][1] = _mm512_load_pd(&(vb->element[1][j]));
			vbj8[thread_index][2] = _mm512_load_pd(&(vb->element[2][j]));
			vbj8[thread_index][3] = _mm512_load_pd(&(vb->element[3][j]));

			//rqd_mul(tmp1, get_tdmatrix_ij(a, i, j), get_tdvector_i(vb, j));
			//rqd_add(tmp, tmp, tmp1);
			_bncavx512_rqd_mul(tmp1_8[thread_index], aij8[thread_index], vbj8[thread_index]);
			_bncavx512_rqd_add(tmp8[thread_index], tmp8[thread_index], tmp1_8[thread_index]);
		}
		//set_ddvector_i(v, i, tmp);
		_bncavx512_rqd_sum512d(tmp[thread_index], tmp8[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
		v->element[2][i] = tmp[thread_index][2];
		v->element[3][i] = tmp[thread_index][3];
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic) + OpenMP
	/* SVE2 dot-product matvec for QD: 4-limb accumulator vectors. */
	{
		long sve_real_col_dim = a->real_col_dim;
		long sve_vl = (long)svcntd();

		#pragma omp parallel for private(thread_index, i, j) firstprivate(sve_vl, sve_real_col_dim)
		for(i = 0; i < row_dim; i++)
		{
			thread_index = omp_get_thread_num();
			(void)thread_index;
			svfloat64_t acc0, acc1, acc2, acc3;
			_bncsve2_rqd_set0(&acc0, &acc1, &acc2, &acc3);

			for(j = 0; j < sve_real_col_dim; j += sve_vl)
			{
				svbool_t pg = svwhilelt_b64_s64((int64_t)j, (int64_t)sve_real_col_dim);
				svfloat64_t aij0 = svld1_f64(pg, &(a->element[0][i * sve_real_col_dim + j]));
				svfloat64_t aij1 = svld1_f64(pg, &(a->element[1][i * sve_real_col_dim + j]));
				svfloat64_t aij2 = svld1_f64(pg, &(a->element[2][i * sve_real_col_dim + j]));
				svfloat64_t aij3 = svld1_f64(pg, &(a->element[3][i * sve_real_col_dim + j]));
				svfloat64_t bj0  = svld1_f64(pg, &(vb->element[0][j]));
				svfloat64_t bj1  = svld1_f64(pg, &(vb->element[1][j]));
				svfloat64_t bj2  = svld1_f64(pg, &(vb->element[2][j]));
				svfloat64_t bj3  = svld1_f64(pg, &(vb->element[3][j]));
				svfloat64_t prod0, prod1, prod2, prod3;
				_bncsve2_rqd_mul(pg, &prod0, &prod1, &prod2, &prod3,
				                 aij0, aij1, aij2, aij3, bj0, bj1, bj2, bj3);
				_bncsve2_rqd_add(pg, &acc0, &acc1, &acc2, &acc3,
				                 acc0, acc1, acc2, acc3, prod0, prod1, prod2, prod3);
			}

			/* Horizontal sum across VL lanes */
			{
				double acc0_arr[sve_vl], acc1_arr[sve_vl], acc2_arr[sve_vl], acc3_arr[sve_vl];
				svst1_f64(svptrue_b64(), acc0_arr, acc0);
				svst1_f64(svptrue_b64(), acc1_arr, acc1);
				svst1_f64(svptrue_b64(), acc2_arr, acc2);
				svst1_f64(svptrue_b64(), acc3_arr, acc3);
				double sum_qd[QDSIZE] = { 0.0, 0.0, 0.0, 0.0 };
				long lane;
				for(lane = 0; lane < sve_vl; lane++)
				{
					double lane_qd[QDSIZE] = { acc0_arr[lane], acc1_arr[lane], acc2_arr[lane], acc3_arr[lane] };
					rqd_add(sum_qd, sum_qd, lane_qd);
				}
				v->element[0][i] = sum_qd[0];
				v->element[1][i] = sum_qd[1];
				v->element[2][i] = sum_qd[2];
				v->element[3][i] = sum_qd[3];
			}
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	long int ij_index, real_row_dim, real_col_dim;
	float64x2_t tmp_neon[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	float64x2_t tmp1_neon[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	float64x2_t aij_neon[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	float64x2_t vbj_neon[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j, ij_index)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		// ゼロ初期化
		_bncneon_rqd_set0(tmp_neon[thread_index]);
		
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			
			// ロード
			aij_neon[thread_index][0] = vld1q_f64(&(a->element[0][ij_index]));
			aij_neon[thread_index][1] = vld1q_f64(&(a->element[1][ij_index]));
			aij_neon[thread_index][2] = vld1q_f64(&(a->element[2][ij_index]));
			aij_neon[thread_index][3] = vld1q_f64(&(a->element[3][ij_index]));
			vbj_neon[thread_index][0] = vld1q_f64(&(vb->element[0][j]));
			vbj_neon[thread_index][1] = vld1q_f64(&(vb->element[1][j]));
			vbj_neon[thread_index][2] = vld1q_f64(&(vb->element[2][j]));
			vbj_neon[thread_index][3] = vld1q_f64(&(vb->element[3][j]));

			// Neon QD乗算と累積加算
			_bncneon_rqd_mul(tmp1_neon[thread_index], aij_neon[thread_index], vbj_neon[thread_index]);
			_bncneon_rqd_add(tmp_neon[thread_index], tmp_neon[thread_index], tmp1_neon[thread_index]);
		}
		
		// float64x2_tの2要素を合計
		_bncneon_rqd_sum128d(tmp[thread_index], tmp_neon[thread_index]);
		
		// 結果を格納
		//set_qdvector_i(v, i, tmp[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
		v->element[2][i] = tmp[thread_index][2];
		v->element[3][i] = tmp[thread_index][3];
	}
#else // others
	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		rqd_set_ui(tmp[thread_index], 0UL);
		for(j = 0; j < a->col_dim; j++)
		{
			rqd_mul(tmp1[thread_index], get_qdmatrix_ij(a, i, j), get_qdvector_i(vb, j));
			rqd_add(tmp[thread_index], tmp[thread_index], tmp1[thread_index]);
		}
		set_qdvector_i(v, i, tmp[thread_index]);
	}
#endif // __AVX2__
}

/* v := a^T * vb */
void _bncomp_mul_qdmatrixt_qdvec(QDVector v, QDMatrix a, QDVector vb)
{
	long int i, j, col_dim;
	int thread_index;
	double tmp[BNCOMP_MAX_NUM_THREADS][QDSIZE], tmp1[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	/* Check Dimension */
	//if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	if((v->dim < a->col_dim) || (vb->dim < a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_qdmatrixt_qdvec\n");
		return;
	}

	col_dim = a->col_dim;

// SIMD
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int real_row_dim, real_col_dim;
	__m256d tmp4[BNCOMP_MAX_NUM_THREADS][QDSIZE], tmp1_4[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	__m256d aij4[BNCOMP_MAX_NUM_THREADS][QDSIZE], vbj4[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < col_dim; i++)
	{
		thread_index = omp_get_thread_num();

		//rqd_set_ui(tmp, 0UL);
		_bncavx2_set0_qd(tmp4[thread_index]);
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
			aij4[thread_index][3] = _mm256_set_pd(
				a->element[3][(j + 3) * real_col_dim + i],
				a->element[3][(j + 2) * real_col_dim + i],
				a->element[3][(j + 1) * real_col_dim + i],
				a->element[3][(j    ) * real_col_dim + i]
			);
			vbj4[thread_index][0] = _mm256_load_pd(&(vb->element[0][j]));
			vbj4[thread_index][1] = _mm256_load_pd(&(vb->element[1][j]));
			vbj4[thread_index][2] = _mm256_load_pd(&(vb->element[2][j]));
			vbj4[thread_index][3] = _mm256_load_pd(&(vb->element[3][j]));

			//rqd_mul(tmp1, get_qdmatrix_ij(a, i, j), get_qdvector_i(vb, j));
			//rqd_add(tmp, tmp, tmp1);
			_bncavx2_rqd_mul(tmp1_4[thread_index], aij4[thread_index], vbj4[thread_index]);
			_bncavx2_rqd_add(tmp4[thread_index], tmp4[thread_index], tmp1_4[thread_index]);
		}
		//set_tdvector_i(v, i, tmp);
		_bncavx2_rqd_sum256d(tmp[thread_index], tmp4[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
		v->element[2][i] = tmp[thread_index][2];
		v->element[3][i] = tmp[thread_index][3];
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int real_row_dim, real_col_dim;
	__m512d tmp8[BNCOMP_MAX_NUM_THREADS][QDSIZE], tmp1_8[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	__m512d aij8[BNCOMP_MAX_NUM_THREADS][QDSIZE], vbj8[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < col_dim; i++)
	{
		thread_index = omp_get_thread_num();

		//rqd_set_ui(tmp, 0UL);
		_bncavx512_set0_qd(tmp8[thread_index]);
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
			aij8[thread_index][3] = _mm512_set_pd(
				a->element[3][(j + 7) * real_col_dim + i],
				a->element[3][(j + 6) * real_col_dim + i],
				a->element[3][(j + 5) * real_col_dim + i],
				a->element[3][(j + 4) * real_col_dim + i],
				a->element[3][(j + 3) * real_col_dim + i],
				a->element[3][(j + 2) * real_col_dim + i],
				a->element[3][(j + 1) * real_col_dim + i],
				a->element[3][(j    ) * real_col_dim + i]
			);
			vbj8[thread_index][0] = _mm512_load_pd(&(vb->element[0][j]));
			vbj8[thread_index][1] = _mm512_load_pd(&(vb->element[1][j]));
			vbj8[thread_index][2] = _mm512_load_pd(&(vb->element[2][j]));
			vbj8[thread_index][3] = _mm512_load_pd(&(vb->element[3][j]));

			//rqd_mul(tmp1, get_tdmatrix_ij(a, i, j), get_tdvector_i(vb, j));
			//rqd_add(tmp, tmp, tmp1);
			_bncavx512_rqd_mul(tmp1_8[thread_index], aij8[thread_index], vbj8[thread_index]);
			_bncavx512_rqd_add(tmp8[thread_index], tmp8[thread_index], tmp1_8[thread_index]);
		}
		//set_ddvector_i(v, i, tmp);
		_bncavx512_rqd_sum512d(tmp[thread_index], tmp8[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
		v->element[2][i] = tmp[thread_index][2];
		v->element[3][i] = tmp[thread_index][3];
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic) + OpenMP
	/* SVE2 transpose-matvec for QD: outer-product, parallel over i tiles. */
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
			svfloat64_t acc0, acc1, acc2, acc3;
			_bncsve2_rqd_set0(&acc0, &acc1, &acc2, &acc3);

			for(j = 0; j < sve_real_row_dim; j++)
			{
				svfloat64_t aji0 = svld1_f64(pg, &(a->element[0][j * sve_real_col_dim + i]));
				svfloat64_t aji1 = svld1_f64(pg, &(a->element[1][j * sve_real_col_dim + i]));
				svfloat64_t aji2 = svld1_f64(pg, &(a->element[2][j * sve_real_col_dim + i]));
				svfloat64_t aji3 = svld1_f64(pg, &(a->element[3][j * sve_real_col_dim + i]));
				svfloat64_t bj0  = svdup_n_f64(vb->element[0][j]);
				svfloat64_t bj1  = svdup_n_f64(vb->element[1][j]);
				svfloat64_t bj2  = svdup_n_f64(vb->element[2][j]);
				svfloat64_t bj3  = svdup_n_f64(vb->element[3][j]);
				svfloat64_t prod0, prod1, prod2, prod3;
				_bncsve2_rqd_mul(pg, &prod0, &prod1, &prod2, &prod3,
				                 aji0, aji1, aji2, aji3, bj0, bj1, bj2, bj3);
				_bncsve2_rqd_add(pg, &acc0, &acc1, &acc2, &acc3,
				                 acc0, acc1, acc2, acc3, prod0, prod1, prod2, prod3);
			}

			svst1_f64(pg, &(v->element[0][i]), acc0);
			svst1_f64(pg, &(v->element[1][i]), acc1);
			svst1_f64(pg, &(v->element[2][i]), acc2);
			svst1_f64(pg, &(v->element[3][i]), acc3);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	long int real_row_dim, real_col_dim;
	float64x2_t tmp_neon[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	float64x2_t tmp1_neon[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	float64x2_t aij_neon[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	float64x2_t vbj_neon[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	double aij_vals[BNCOMP_MAX_NUM_THREADS][QDSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	// 列ごとに並列処理（転置なので列方向にアクセス）
	#pragma omp parallel for private(thread_index, j) // , aij_vals)
	for(i = 0; i < col_dim; i++)
	{
		thread_index = omp_get_thread_num();

		// ゼロ初期化
		_bncneon_rqd_set0(tmp_neon[thread_index]);
		
		// 行方向にループ（転置なので A[j][i] を A^T[i][j] として扱う）
		for(j = 0; j < real_row_dim; j += _BNC_D_WIDTH)
		{
			// Gather elements for transpose operation
			aij_vals[thread_index][0] = a->element[0][j * real_col_dim + i];
			aij_vals[thread_index][1] = a->element[0][(j + 1) * real_col_dim + i];
			aij_neon[thread_index][0] = vld1q_f64(aij_vals[thread_index]);

			aij_vals[thread_index][0] = a->element[1][j * real_col_dim + i];
			aij_vals[thread_index][1] = a->element[1][(j + 1) * real_col_dim + i];
			aij_neon[thread_index][1] = vld1q_f64(aij_vals[thread_index]);

			aij_vals[thread_index][0] = a->element[2][j * real_col_dim + i];
			aij_vals[thread_index][1] = a->element[2][(j + 1) * real_col_dim + i];
			aij_neon[thread_index][2] = vld1q_f64(aij_vals[thread_index]);			

			aij_vals[thread_index][0] = a->element[3][j * real_col_dim + i];
			aij_vals[thread_index][1] = a->element[3][(j + 1) * real_col_dim + i];
			aij_neon[thread_index][3] = vld1q_f64(aij_vals[thread_index]);			
			
			// vb[j:j+1] をロード（連続要素）
			vbj_neon[thread_index][0] = vld1q_f64(&(vb->element[0][j]));
			vbj_neon[thread_index][1] = vld1q_f64(&(vb->element[1][j]));
			vbj_neon[thread_index][2] = vld1q_f64(&(vb->element[2][j]));
			vbj_neon[thread_index][3] = vld1q_f64(&(vb->element[3][j]));

			// Neon QD乗算と累積加算
			_bncneon_rqd_mul(tmp1_neon[thread_index], aij_neon[thread_index], vbj_neon[thread_index]);
			_bncneon_rqd_add(tmp_neon[thread_index], tmp_neon[thread_index], tmp1_neon[thread_index]);
		}
		
		// float64x2_tの2要素を合計
		_bncneon_rqd_sum128d(tmp[thread_index], tmp_neon[thread_index]);
		
		// 結果を格納
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
		v->element[2][i] = tmp[thread_index][2];
		v->element[3][i] = tmp[thread_index][3];
	}
#else // others
	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < col_dim; i++)
	{
		thread_index = omp_get_thread_num();

		set0_qd(tmp[thread_index]);
		for(j = 0; j < a->row_dim; j++)
		{
			rqd_mul(tmp1[thread_index], get_qdmatrix_ij(a, j, i), get_qdvector_i(vb, j));
			rqd_add(tmp[thread_index], tmp[thread_index], tmp1[thread_index]);
		}
		set_qdvector_i(v, i, tmp[thread_index]);
	}
#endif // __AVX2__
}

// Matrix multiplication based on Ozaki scheme
/*------------------------------------------------------------------------------*/
/* Matrix  multiplication based on Ozaki scheme                          */
/*                                                                               */
/* mul_qdmatrix_oz() is OpenMP-parallel itself: it cuts the rows of   */
/* ret into blocks and gives one block at a time to a thread, which runs every   */
/* slice product for it with a single-threaded BLAS call and accumulates in      */
/* QD on the spot.  That parallelizes the splitting and the accumulation as   */
/* well, whereas the code that used to stand here parallelized only the outer    */
/* slice loop and pushed every accumulation through an omp critical -- which     */
/* serialized the expensive half and made the sum order depend on the thread     */
/* interleaving.  Delegating is both faster and reproducible.                    */
/*------------------------------------------------------------------------------*/
void _bncomp_mul_qdmatrix_oz(QDMatrix ret, QDMatrix a, int max_num_div_a, QDMatrix b, int max_num_div_b)
{
    mul_qdmatrix_oz(ret, a, max_num_div_a, b, max_num_div_b);
}

#if 0

// Fit dimension to be multiple of min_dim
void _bncomp_mul_cqdmatrix_oz_3m(CQDMatrix ret, CQDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CQDMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    QDMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_qdmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    _bncomp_mul_qdmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_a_image);
    _bncomp_sub_qdmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        mul_qdmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        mul_qdmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_a_image);
        add_qdmatrix(ret->im, t3, t4);
    #else // USE_4M
    */
        // 3M
        _bncomp_add_qdmatrix(t3, a->re, a->im);
        _bncomp_add_qdmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        _bncomp_mul_qdmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        _bncomp_sub_qdmatrix(ret->im, ret->im, t1);
        _bncomp_sub_qdmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_qdmatrix(t1);
    free_qdmatrix(t2);
    free_qdmatrix(t3);
    free_qdmatrix(t4);
}

// Fit dimension to be multiple of min_dim
void _bncomp_mul_cqdmatrix_oz_4m(CQDMatrix ret, CQDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CQDMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    QDMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_qdmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    _bncomp_mul_qdmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_a_image);
    _bncomp_sub_qdmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        _bncomp_mul_qdmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        _bncomp_mul_qdmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_a_image);
        _bncomp_add_qdmatrix(ret->im, t3, t4);
    /*
    #else // USE_4M 
        // 3M
        add_qdmatrix(t3, a->re, a->im);
        add_qdmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        mul_qdmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        sub_qdmatrix(ret->im, ret->im, t1);
        sub_qdmatrix(ret->im, ret->im, t2);
    #endif // USE_4M
    */

    free_qdmatrix(t1);
    free_qdmatrix(t2);
    free_qdmatrix(t3);
    free_qdmatrix(t4);
}

// Simple triple-loop-way matrix multiplication (3M)
void _bncomp_mul_cqdmatrix_3m(CQDMatrix ret, CQDMatrix a, CQDMatrix b)
{
    QDMatrix t1, t2, t3, t4;
 
    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_qdmatrix(t1, a->re, b->re);
    _bncomp_mul_qdmatrix(t2, a->im, b->im);
    _bncomp_sub_qdmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        _bncomp_mul_qdmatrix(t3, a->im, b->re);
        _bncomp_mul_qdmatrix(t4, a->re, b->im);
        _bncomp_add_qdmatrix(ret->im, t3, t4);
    #else // USE_4M
    */
        // 3M
        _bncomp_add_qdmatrix(t3, a->re, a->im);
        _bncomp_add_qdmatrix(t4, b->re, b->im);
        _bncomp_mul_qdmatrix(ret->im, t3, t4);
        _bncomp_sub_qdmatrix(ret->im, ret->im, t1);
        _bncomp_sub_qdmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_qdmatrix(t1);
    free_qdmatrix(t2);
    free_qdmatrix(t3);
    free_qdmatrix(t4);
}

// Simple triple-loop-way matrix multiplication (4M)
void _bncomp_mul_cqdmatrix_4m(CQDMatrix ret, CQDMatrix a, CQDMatrix b)
{
    QDMatrix t1, t2, t3, t4;
 
    t1 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_qdmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_qdmatrix(t1, a->re, b->re);
    _bncomp_mul_qdmatrix(t2, a->im, b->im);
    _bncomp_sub_qdmatrix(ret->re, t1, t2);

    // 4M
    
    //#ifdef USE_4M
        _bncomp_mul_qdmatrix(t3, a->im, b->re);
        _bncomp_mul_qdmatrix(t4, a->re, b->im);
        _bncomp_add_qdmatrix(ret->im, t3, t4);
    //#else // USE_4M
	/*
        // 3M
        _bncomp_add_qdmatrix(t3, a->re, a->im);
        _bncomp_add_qdmatrix(t4, b->re, b->im);
        _bncomp_mul_qdmatrix(ret->im, t3, t4s);
        _bncomp_sub_qdmatrix(ret->im, ret->im, t1);
        _bncomp_sub_qdmatrix(ret->im, ret->im, t2);
	*/
    //#endif // USE_4M

    free_qdmatrix(t1);
    free_qdmatrix(t2);
    free_qdmatrix(t3);
    free_qdmatrix(t4);
}
#endif // 0

/****************************************************************************/
/* OpenMP-parallel LU decomposition (partial pivoting), mirrors QDLUdecompPM */
/* with the O(n^3) trailing update parallelized over rows j. SVE2 build uses */
/* the NEON inner kernel (arrays-of-SVE-types unusable), as the serial does. */
/****************************************************************************/
int _bncomp_QDLUdecompPM(QDMatrix a, long int ch[])
{
	long int i, j, k, imax, itmp, dim, dim_start, dim_end;
	double dtmp[QDSIZE], dtmp1[QDSIZE], dmaxii[QDSIZE];

	dim = a->col_dim;
	for(i = 0; i < a->row_dim; i++) ch[i] = i;

	for(i = 0; i < a->row_dim; i++)
	{
		rqd_abs(dmaxii, get_qdmatrix_ij(a, i, i));
		imax = i;
		for(j = i + 1; j < a->row_dim; j++)
		{
			rqd_abs(dtmp, get_qdmatrix_ij(a, j, i));
			if(rqd_cmp(dtmp, dmaxii) > 0) { imax = j; rqd_set(dmaxii, dtmp); }
		}
		if(rqd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! _bncomp_QDLUdecompPM!\n", i);
			return -1;
		}
		if(imax != i)
		{
			itmp = ch[imax]; ch[imax] = ch[i]; ch[i] = itmp;
			row_swap_qdmatrix(a, i, imax, 0, a->col_dim);
		}
		for(j = i + 1; j < dim; j++)
		{
			rqd_div(dtmp, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, i));
			set_qdmatrix_ij(a, j, i, dtmp);
		}
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->real_col_dim;
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
		#pragma omp parallel for schedule(static) private(k)
		for(j = i + 1; j < dim; j++)
		{
			double _t[QDSIZE], _t1[QDSIZE];
			long int index_ji = j * a->real_col_dim + i, index_ik, index_jk, k_end;
			__m256d dtmpv[QDSIZE], ajiv[QDSIZE], ajkv[QDSIZE], aikv[QDSIZE];
			ajiv[0] = _mm256_set1_pd(a->element[0][index_ji]);
			ajiv[1] = _mm256_set1_pd(a->element[1][index_ji]);
			ajiv[2] = _mm256_set1_pd(a->element[2][index_ji]);
			ajiv[3] = _mm256_set1_pd(a->element[3][index_ji]);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = i + 1; k < k_end; k++)
			{
				rqd_mul(_t1, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, k));
				rqd_sub(_t, get_qdmatrix_ij(a, j, k), _t1);
				set_qdmatrix_ij(a, j, k, _t);
			}
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				aikv[0] = _mm256_load_pd(&(a->element[0][index_ik]));
				aikv[1] = _mm256_load_pd(&(a->element[1][index_ik]));
				aikv[2] = _mm256_load_pd(&(a->element[2][index_ik]));
				aikv[3] = _mm256_load_pd(&(a->element[3][index_ik]));
				_bncavx2_rqd_mul(dtmpv, ajiv, aikv);
				index_jk = j * a->real_col_dim + k;
				ajkv[0] = _mm256_load_pd(&(a->element[0][index_jk]));
				ajkv[1] = _mm256_load_pd(&(a->element[1][index_jk]));
				ajkv[2] = _mm256_load_pd(&(a->element[2][index_jk]));
				ajkv[3] = _mm256_load_pd(&(a->element[3][index_jk]));
				_bncavx2_rqd_sub(dtmpv, ajkv, dtmpv);
				_mm256_store_pd(&(a->element[0][index_jk]), dtmpv[0]);
				_mm256_store_pd(&(a->element[1][index_jk]), dtmpv[1]);
				_mm256_store_pd(&(a->element[2][index_jk]), dtmpv[2]);
				_mm256_store_pd(&(a->element[3][index_jk]), dtmpv[3]);
			}
		}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon (also SVE2 build)
		#pragma omp parallel for schedule(static) private(k)
		for(j = i + 1; j < dim; j++)
		{
			double _t[QDSIZE], _t1[QDSIZE];
			long int index_ji = j * a->real_col_dim + i, index_ik, index_jk, k_end;
			float64x2_t dtmpv[QDSIZE], ajiv[QDSIZE], ajkv[QDSIZE], aikv[QDSIZE];
			ajiv[0] = vdupq_n_f64(a->element[0][index_ji]);
			ajiv[1] = vdupq_n_f64(a->element[1][index_ji]);
			ajiv[2] = vdupq_n_f64(a->element[2][index_ji]);
			ajiv[3] = vdupq_n_f64(a->element[3][index_ji]);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = i + 1; k < k_end; k++)
			{
				rqd_mul(_t1, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, k));
				rqd_sub(_t, get_qdmatrix_ij(a, j, k), _t1);
				set_qdmatrix_ij(a, j, k, _t);
			}
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				aikv[0] = vld1q_f64(&(a->element[0][index_ik]));
				aikv[1] = vld1q_f64(&(a->element[1][index_ik]));
				aikv[2] = vld1q_f64(&(a->element[2][index_ik]));
				aikv[3] = vld1q_f64(&(a->element[3][index_ik]));
				_bncneon_rqd_mul(dtmpv, ajiv, aikv);
				index_jk = j * a->real_col_dim + k;
				ajkv[0] = vld1q_f64(&(a->element[0][index_jk]));
				ajkv[1] = vld1q_f64(&(a->element[1][index_jk]));
				ajkv[2] = vld1q_f64(&(a->element[2][index_jk]));
				ajkv[3] = vld1q_f64(&(a->element[3][index_jk]));
				_bncneon_rqd_sub(dtmpv, ajkv, dtmpv);
				vst1q_f64(&(a->element[0][index_jk]), dtmpv[0]);
				vst1q_f64(&(a->element[1][index_jk]), dtmpv[1]);
				vst1q_f64(&(a->element[2][index_jk]), dtmpv[2]);
				vst1q_f64(&(a->element[3][index_jk]), dtmpv[3]);
			}
		}
#else // others
		#pragma omp parallel for schedule(static) private(k)
		for(j = i + 1; j < dim; j++)
		{
			double _t[QDSIZE], _t1[QDSIZE];
			for(k = i + 1; k < dim; k++)
			{
				rqd_mul(_t1, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, k));
				rqd_sub(_t, get_qdmatrix_ij(a, j, k), _t1);
				set_qdmatrix_ij(a, j, k, _t);
			}
		}
#endif // __AVX2__
	}
	return 0;
}

int _bncomp_SolveQDLSPM(QDVector answer, QDMatrix lu, QDVector b, long int ch[])
{
	long int i, j, dim;
	double dtmp[QDSIZE], dtmp1[QDSIZE];

	dim = answer->dim;
	for(i = 0; i < dim; i++) set_qdvector_i(answer, i, get_qdvector_i(b, ch[i]));

	for(i = 0; i < dim; i++)
	{
		rqd_abs(dtmp, get_qdmatrix_ij(lu, i, i));
		if(rqd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(_bncomp_SolveQDLSPM, %ld)\n", i);
			return -1;
		}
		#pragma omp parallel for schedule(static)
		for(j = i + 1; j < dim; j++)
		{
			double _t[QDSIZE], _t1[QDSIZE];
			rqd_mul(_t1, get_qdmatrix_ij(lu, j, i), get_qdvector_i(answer, i));
			rqd_sub(_t, get_qdvector_i(answer, j), _t1);
			set_qdvector_i(answer, j, _t);
		}
	}
	for(i = dim - 1; i >= 0; i--)
	{
		for(j = i + 1; j < dim; j++)
		{
			rqd_mul(dtmp1, get_qdmatrix_ij(lu, i, j), get_qdvector_i(answer, j));
			rqd_sub(dtmp, get_qdvector_i(answer, i), dtmp1);
			set_qdvector_i(answer, i, dtmp);
		}
		rqd_div(dtmp, get_qdvector_i(answer, i), get_qdmatrix_ij(lu, i, i));
		set_qdvector_i(answer, i, dtmp);
	}
	return 0;
}

