/********************************************************************************/
/* bncomp_linear_qs.c: Parallelized DS Precision Linear Computation Library     */
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

/* Force-include NEON helpers used by the OMP+NEON / OMP+SVE2 (NEON fallback)
 * branches.  bncomp.h transitively pulls neon/bncneon.h via qslinear.h, but the
 * include chain depends on __ARM_SVE2 / __ARM_NEON macros — when the
 * outlined _omp_fn.0 references reduction helpers (sum128f, ...), the static
 * definition MUST be visible.  Including directly here guarantees that. */
#if (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON)
#include "neon/_bncneon_qs.h"
#endif

//---------------------------------------
// QS
//---------------------------------------
//--------
// Vector
//--------
/* c := a */
void _bncomp_subst_qsvector(QSVector c, QSVector a)
{
	long int i;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
//	for(i = 0; i < a->dim; i++)
	#pragma omp parallel for
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//set_tvector_i(c, i, get_tvector_i(a, i));
		_mm256_store_ps(&(c->element[0][i]), _mm256_load_ps(&(a->element[0][i])));
		_mm256_store_ps(&(c->element[1][i]), _mm256_load_ps(&(a->element[1][i])));
		_mm256_store_ps(&(c->element[2][i]), _mm256_load_ps(&(a->element[2][i])));
		_mm256_store_ps(&(c->element[3][i]), _mm256_load_ps(&(a->element[3][i])));
	}
#elif defined(__AVX512F__) // __AVX512F__
	#pragma omp parallel for
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		//set_tvector_i(c, i, get_tvector_i(a, i));
		_mm512_store_ps(&(c->element[0][i]), _mm512_load_ps(&(a->element[0][i])));
		_mm512_store_ps(&(c->element[1][i]), _mm512_load_ps(&(a->element[1][i])));
		_mm512_store_ps(&(c->element[2][i]), _mm512_load_ps(&(a->element[2][i])));
		_mm512_store_ps(&(c->element[3][i]), _mm512_load_ps(&(a->element[3][i])));
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntw(); long _N = a->real_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b32_s32((int32_t)_ix, (int32_t)_N);
			svst1_f32(_pg, &(c->element[0][_ix]), svld1_f32(_pg, &(a->element[0][_ix])));
			svst1_f32(_pg, &(c->element[1][_ix]), svld1_f32(_pg, &(a->element[1][_ix])));
			svst1_f32(_pg, &(c->element[2][_ix]), svld1_f32(_pg, &(a->element[2][_ix])));
			svst1_f32(_pg, &(c->element[3][_ix]), svld1_f32(_pg, &(a->element[3][_ix])));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	#pragma omp parallel for
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		// float32x4_tで2要素ずつロード・ストア
		float32x4_t tmp0 = vld1q_f32(&(a->element[0][i]));
		float32x4_t tmp1 = vld1q_f32(&(a->element[1][i]));
		float32x4_t tmp2 = vld1q_f32(&(a->element[2][i]));
		float32x4_t tmp3 = vld1q_f32(&(a->element[3][i]));
		
		vst1q_f32(&(c->element[0][i]), tmp0);
		vst1q_f32(&(c->element[1][i]), tmp1);
		vst1q_f32(&(c->element[2][i]), tmp2);
		vst1q_f32(&(c->element[3][i]), tmp3);
	}
#else // others
	#pragma omp parallel for
	for(i = 0; i < a->dim; i++)
		set_qsvector_i(c, i, get_qsvector_i(a, i));

#endif // __AVX2__

}

/* c = a + b */
void _bncomp_add_qsvector(QSVector c, QSVector a, QSVector b)
{
	int thread_index;
	long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_qsvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256 in_ret[BNCOMP_MAX_NUM_THREADS][QSSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][QSSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][QSSIZE];

 	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[thread_index][1] = _mm256_load_ps(&(a->element[1][index]));
        in_a_val[thread_index][2] = _mm256_load_ps(&(a->element[2][index]));
        in_a_val[thread_index][3] = _mm256_load_ps(&(a->element[3][index]));
        in_b_val[thread_index][0] = _mm256_load_ps(&(b->element[0][index]));
        in_b_val[thread_index][1] = _mm256_load_ps(&(b->element[1][index]));
        in_b_val[thread_index][2] = _mm256_load_ps(&(b->element[2][index]));
        in_b_val[thread_index][3] = _mm256_load_ps(&(b->element[3][index]));

        _bncavx2_rqs_add(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        _mm256_store_ps(&(c->element[0][index]), in_ret[thread_index][0]);
        _mm256_store_ps(&(c->element[1][index]), in_ret[thread_index][1]);
        _mm256_store_ps(&(c->element[2][index]), in_ret[thread_index][2]);
        _mm256_store_ps(&(c->element[3][index]), in_ret[thread_index][3]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512 in_ret[BNCOMP_MAX_NUM_THREADS][QSSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][QSSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	#pragma omp parallel for private(thread_index)
    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = _mm512_load_ps(&(a->element[0][index]));
        in_a_val[thread_index][1] = _mm512_load_ps(&(a->element[1][index]));
        in_a_val[thread_index][2] = _mm512_load_ps(&(a->element[2][index]));
        in_a_val[thread_index][3] = _mm512_load_ps(&(a->element[3][index]));
        in_b_val[thread_index][0] = _mm512_load_ps(&(b->element[0][index]));
        in_b_val[thread_index][1] = _mm512_load_ps(&(b->element[1][index]));
        in_b_val[thread_index][2] = _mm512_load_ps(&(b->element[2][index]));
        in_b_val[thread_index][3] = _mm512_load_ps(&(b->element[3][index]));

        _bncavx512_rqs_add(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        _mm512_store_ps(&(c->element[0][index]), in_ret[thread_index][0]);
        _mm512_store_ps(&(c->element[1][index]), in_ret[thread_index][1]);
        _mm512_store_ps(&(c->element[2][index]), in_ret[thread_index][2]);
        _mm512_store_ps(&(c->element[3][index]), in_ret[thread_index][3]);
   }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntw(); long _N = c->real_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b32_s32((int32_t)_ix, (int32_t)_N);
			svfloat32_t _a0 = svld1_f32(_pg, &(a->element[0][_ix]));
			svfloat32_t _a1 = svld1_f32(_pg, &(a->element[1][_ix]));
			svfloat32_t _a2 = svld1_f32(_pg, &(a->element[2][_ix]));
			svfloat32_t _a3 = svld1_f32(_pg, &(a->element[3][_ix]));
			svfloat32_t _b0 = svld1_f32(_pg, &(b->element[0][_ix]));
			svfloat32_t _b1 = svld1_f32(_pg, &(b->element[1][_ix]));
			svfloat32_t _b2 = svld1_f32(_pg, &(b->element[2][_ix]));
			svfloat32_t _b3 = svld1_f32(_pg, &(b->element[3][_ix]));
			svfloat32_t _o0, _o1, _o2, _o3;
			_bncsve2_rqs_add(_pg, &_o0, &_o1, &_o2, &_o3, _a0, _a1, _a2, _a3, _b0, _b1, _b2, _b3);
			svst1_f32(_pg, &(c->element[0][_ix]), _o0);
			svst1_f32(_pg, &(c->element[1][_ix]), _o1);
			svst1_f32(_pg, &(c->element[2][_ix]), _o2);
			svst1_f32(_pg, &(c->element[3][_ix]), _o3);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	float32x4_t in_ret[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	float32x4_t in_a_val[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	float32x4_t in_b_val[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();

		// a, b の値をロード (2要素ずつ)
		in_a_val[thread_index][0] = vld1q_f32(&(a->element[0][index]));
		in_a_val[thread_index][1] = vld1q_f32(&(a->element[1][index]));
		in_a_val[thread_index][2] = vld1q_f32(&(a->element[2][index]));
		in_a_val[thread_index][3] = vld1q_f32(&(a->element[3][index]));
		in_b_val[thread_index][0] = vld1q_f32(&(b->element[0][index]));
		in_b_val[thread_index][1] = vld1q_f32(&(b->element[1][index]));
		in_b_val[thread_index][2] = vld1q_f32(&(b->element[2][index]));
		in_b_val[thread_index][3] = vld1q_f32(&(b->element[3][index]));

		// Neon QS加算
		_bncneon_rqs_add(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

		// 結果をストア
		vst1q_f32(&(c->element[0][index]), in_ret[thread_index][0]);
		vst1q_f32(&(c->element[1][index]), in_ret[thread_index][1]);
		vst1q_f32(&(c->element[2][index]), in_ret[thread_index][2]);
		vst1q_f32(&(c->element[3][index]), in_ret[thread_index][3]);
	}
#else // others
	float tmp[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();
		rqs_add(tmp[thread_index], get_qsvector_i(a, i),  get_qsvector_i(b, i));
		set_qsvector_i(c, i, tmp[thread_index]);
	}
#endif // __AVX2__
}

/* c = a - b */
void _bncomp_sub_qsvector(QSVector c, QSVector a, QSVector b)
{
	int thread_index;
	long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: _bncomp_sub_qsvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256 in_ret[BNCOMP_MAX_NUM_THREADS][QSSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][QSSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	#pragma omp parallel for private(thread_index)
    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[thread_index][1] = _mm256_load_ps(&(a->element[1][index]));
        in_a_val[thread_index][2] = _mm256_load_ps(&(a->element[2][index]));
        in_a_val[thread_index][3] = _mm256_load_ps(&(a->element[3][index]));
        in_b_val[thread_index][0] = _mm256_load_ps(&(b->element[0][index]));
        in_b_val[thread_index][1] = _mm256_load_ps(&(b->element[1][index]));
        in_b_val[thread_index][2] = _mm256_load_ps(&(b->element[2][index]));
        in_b_val[thread_index][3] = _mm256_load_ps(&(b->element[3][index]));

        _bncavx2_rqs_sub(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        _mm256_store_ps(&(c->element[0][index]), in_ret[thread_index][0]);
        _mm256_store_ps(&(c->element[1][index]), in_ret[thread_index][1]);
        _mm256_store_ps(&(c->element[2][index]), in_ret[thread_index][2]);
        _mm256_store_ps(&(c->element[3][index]), in_ret[thread_index][3]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512 in_ret[BNCOMP_MAX_NUM_THREADS][QSSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][QSSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = _mm512_load_ps(&(a->element[0][index]));
        in_a_val[thread_index][1] = _mm512_load_ps(&(a->element[1][index]));
        in_a_val[thread_index][2] = _mm512_load_ps(&(a->element[2][index]));
        in_a_val[thread_index][3] = _mm512_load_ps(&(a->element[3][index]));
        in_b_val[thread_index][0] = _mm512_load_ps(&(b->element[0][index]));
        in_b_val[thread_index][1] = _mm512_load_ps(&(b->element[1][index]));
        in_b_val[thread_index][2] = _mm512_load_ps(&(b->element[2][index]));
        in_b_val[thread_index][3] = _mm512_load_ps(&(b->element[3][index]));

        _bncavx512_rqs_sub(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        _mm512_store_ps(&(c->element[0][index]), in_ret[thread_index][0]);
        _mm512_store_ps(&(c->element[1][index]), in_ret[thread_index][1]);
        _mm512_store_ps(&(c->element[2][index]), in_ret[thread_index][2]);
        _mm512_store_ps(&(c->element[3][index]), in_ret[thread_index][3]);
   }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntw(); long _N = c->real_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b32_s32((int32_t)_ix, (int32_t)_N);
			svfloat32_t _a0 = svld1_f32(_pg, &(a->element[0][_ix]));
			svfloat32_t _a1 = svld1_f32(_pg, &(a->element[1][_ix]));
			svfloat32_t _a2 = svld1_f32(_pg, &(a->element[2][_ix]));
			svfloat32_t _a3 = svld1_f32(_pg, &(a->element[3][_ix]));
			svfloat32_t _b0 = svld1_f32(_pg, &(b->element[0][_ix]));
			svfloat32_t _b1 = svld1_f32(_pg, &(b->element[1][_ix]));
			svfloat32_t _b2 = svld1_f32(_pg, &(b->element[2][_ix]));
			svfloat32_t _b3 = svld1_f32(_pg, &(b->element[3][_ix]));
			_b0 = svneg_f32_x(_pg, _b0);
			_b1 = svneg_f32_x(_pg, _b1);
			_b2 = svneg_f32_x(_pg, _b2);
			_b3 = svneg_f32_x(_pg, _b3);
			svfloat32_t _o0, _o1, _o2, _o3;
			_bncsve2_rqs_add(_pg, &_o0, &_o1, &_o2, &_o3, _a0, _a1, _a2, _a3, _b0, _b1, _b2, _b3);
			svst1_f32(_pg, &(c->element[0][_ix]), _o0);
			svst1_f32(_pg, &(c->element[1][_ix]), _o1);
			svst1_f32(_pg, &(c->element[2][_ix]), _o2);
			svst1_f32(_pg, &(c->element[3][_ix]), _o3);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	float32x4_t in_ret[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	float32x4_t in_a_val[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	float32x4_t in_b_val[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();

		// a, b の値をロード
		in_a_val[thread_index][0] = vld1q_f32(&(a->element[0][index]));
		in_a_val[thread_index][1] = vld1q_f32(&(a->element[1][index]));
		in_a_val[thread_index][2] = vld1q_f32(&(a->element[2][index]));
		in_a_val[thread_index][3] = vld1q_f32(&(a->element[3][index]));
		in_b_val[thread_index][0] = vld1q_f32(&(b->element[0][index]));
		in_b_val[thread_index][1] = vld1q_f32(&(b->element[1][index]));
		in_b_val[thread_index][2] = vld1q_f32(&(b->element[2][index]));
		in_b_val[thread_index][3] = vld1q_f32(&(b->element[3][index]));

		// Neon QS減算
		_bncneon_rqs_sub(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

		// 結果をストア
		vst1q_f32(&(c->element[0][index]), in_ret[thread_index][0]);
		vst1q_f32(&(c->element[1][index]), in_ret[thread_index][1]);
		vst1q_f32(&(c->element[2][index]), in_ret[thread_index][2]);
		vst1q_f32(&(c->element[3][index]), in_ret[thread_index][3]);
	}
#else // others
	float tmp[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

		rqs_sub(tmp[thread_index], get_qsvector_i(a, i), get_qsvector_i(b, i));
		set_qsvector_i(c, i, tmp[thread_index]);
	}
#endif // __AVX2__

}

/* c = val * a */
void _bncomp_cmul_qsvector(QSVector c, float val[QSSIZE], QSVector a)
{
	int thread_index;
	long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: cmul_qsvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4[BNCOMP_MAX_NUM_THREADS][QSSIZE], c4[BNCOMP_MAX_NUM_THREADS][QSSIZE], val4[QSSIZE];

	val4[0] = _mm256_set1_ps(val[0]);
	val4[1] = _mm256_set1_ps(val[1]);
	val4[2] = _mm256_set1_ps(val[2]);
	val4[3] = _mm256_set1_ps(val[3]);

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();

		//set_dsvector_i(c, i, val * get_dsvector_i(a, i));
		a4[thread_index][0] = _mm256_load_ps(&(a->element[0][index]));
		a4[thread_index][1] = _mm256_load_ps(&(a->element[1][index]));
		a4[thread_index][2] = _mm256_load_ps(&(a->element[2][index]));
		a4[thread_index][3] = _mm256_load_ps(&(a->element[3][index]));

		_bncavx2_rqs_mul(c4[thread_index], val4, a4[thread_index]);
		//_bncavx2_rqs_mul(c4[thread_index], val4[thread_index], a4[thread_index]);

		_mm256_store_ps(&(c->element[0][index]), c4[thread_index][0]);
		_mm256_store_ps(&(c->element[1][index]), c4[thread_index][1]);
		_mm256_store_ps(&(c->element[2][index]), c4[thread_index][2]);
		_mm256_store_ps(&(c->element[3][index]), c4[thread_index][3]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a8[BNCOMP_MAX_NUM_THREADS][QSSIZE], c8[BNCOMP_MAX_NUM_THREADS][QSSIZE], val8[QSSIZE];

	val8[0] = _mm512_set1_ps(val[0]);
	val8[1] = _mm512_set1_ps(val[1]);
	val8[2] = _mm512_set1_ps(val[2]);
	val8[3] = _mm512_set1_ps(val[3]);

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();

		//set_dsvector_i(c, i, val * get_dsvector_i(a, i));
		a8[thread_index][0] = _mm512_load_ps(&(a->element[0][index]));
		a8[thread_index][1] = _mm512_load_ps(&(a->element[1][index]));
		a8[thread_index][2] = _mm512_load_ps(&(a->element[2][index]));
		a8[thread_index][3] = _mm512_load_ps(&(a->element[3][index]));

		_bncavx512_rqs_mul(c8[thread_index], val8, a8[thread_index]);

		_mm512_store_ps(&(c->element[0][index]), c8[thread_index][0]);
		_mm512_store_ps(&(c->element[1][index]), c8[thread_index][1]);
		_mm512_store_ps(&(c->element[2][index]), c8[thread_index][2]);
		_mm512_store_ps(&(c->element[3][index]), c8[thread_index][3]);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntw(); long _N = a->real_dim; long _ix;
		svfloat32_t _v0 = svdup_n_f32(val[0]);
		svfloat32_t _v1 = svdup_n_f32(val[1]);
		svfloat32_t _v2 = svdup_n_f32(val[2]);
		svfloat32_t _v3 = svdup_n_f32(val[3]);
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b32_s32((int32_t)_ix, (int32_t)_N);
			svfloat32_t _a0 = svld1_f32(_pg, &(a->element[0][_ix]));
			svfloat32_t _a1 = svld1_f32(_pg, &(a->element[1][_ix]));
			svfloat32_t _a2 = svld1_f32(_pg, &(a->element[2][_ix]));
			svfloat32_t _a3 = svld1_f32(_pg, &(a->element[3][_ix]));
			svfloat32_t _o0, _o1, _o2, _o3;
			_bncsve2_rqs_mul(_pg, &_o0, &_o1, &_o2, &_o3, _v0, _v1, _v2, _v3, _a0, _a1, _a2, _a3);
			svst1_f32(_pg, &(c->element[0][_ix]), _o0);
			svst1_f32(_pg, &(c->element[1][_ix]), _o1);
			svst1_f32(_pg, &(c->element[2][_ix]), _o2);
			svst1_f32(_pg, &(c->element[3][_ix]), _o3);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	float32x4_t in_ret[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	float32x4_t in_a_val[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	float32x4_t in_scalar_val[QSSIZE];

	// scalar を broadcast
	_bncneon_rqs_set1_qs(in_scalar_val, val); // scalar);

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();

		// a の値をロード
		in_a_val[thread_index][0] = vld1q_f32(&(a->element[0][index]));
		in_a_val[thread_index][1] = vld1q_f32(&(a->element[1][index]));
		in_a_val[thread_index][2] = vld1q_f32(&(a->element[2][index]));
		in_a_val[thread_index][3] = vld1q_f32(&(a->element[3][index]));

		// Neon QS乗算
		_bncneon_rqs_mul(in_ret[thread_index], in_scalar_val, in_a_val[thread_index]);

		// 結果をストア
		vst1q_f32(&(c->element[0][index]), in_ret[thread_index][0]);
		vst1q_f32(&(c->element[1][index]), in_ret[thread_index][1]);
		vst1q_f32(&(c->element[2][index]), in_ret[thread_index][2]);
		vst1q_f32(&(c->element[3][index]), in_ret[thread_index][3]);
	}
#else // others

	float tmp[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

		rqs_mul(tmp[thread_index], val, get_qsvector_i(a, i));
		set_qsvector_i(c, i, tmp[thread_index]);
	}
#endif // __AVX2__

}

/* (a, b) */
void _bncomp_ip_qsvector(float ret[QSSIZE], QSVector a, QSVector b)
{
	int thread_index;
	long int i, index;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: _bncomp_ip_qsvector\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4[BNCOMP_MAX_NUM_THREADS][QSSIZE], b4[BNCOMP_MAX_NUM_THREADS][QSSIZE], ret4[QSSIZE], tmp4[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	_bncavx2_set0_qs(ret4);
	
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a4[thread_index][0] = _mm256_load_ps(&(a->element[0][index]));
		a4[thread_index][1] = _mm256_load_ps(&(a->element[1][index]));
		a4[thread_index][2] = _mm256_load_ps(&(a->element[2][index]));
		a4[thread_index][3] = _mm256_load_ps(&(a->element[3][index]));
		b4[thread_index][0] = _mm256_load_ps(&(b->element[0][index]));
		b4[thread_index][1] = _mm256_load_ps(&(b->element[1][index]));
		b4[thread_index][2] = _mm256_load_ps(&(b->element[2][index]));
		b4[thread_index][3] = _mm256_load_ps(&(b->element[3][index]));

		_bncavx2_rqs_mul(tmp4[thread_index], a4[thread_index], b4[thread_index]);
		#pragma omp critical
			_bncavx2_rqs_add(ret4, ret4, tmp4[thread_index]);
	}

	_bncavx2_rqs_sum256(ret, ret4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512 a8[BNCOMP_MAX_NUM_THREADS][QSSIZE], b8[BNCOMP_MAX_NUM_THREADS][QSSIZE], ret8[QSSIZE], tmp8[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	_bncavx512_set0_qs(ret8);
	
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a8[thread_index][0] = _mm512_load_ps(&(a->element[0][index]));
		a8[thread_index][1] = _mm512_load_ps(&(a->element[1][index]));
		a8[thread_index][2] = _mm512_load_ps(&(a->element[2][index]));
		a8[thread_index][3] = _mm512_load_ps(&(a->element[3][index]));
		b8[thread_index][0] = _mm512_load_ps(&(b->element[0][index]));
		b8[thread_index][1] = _mm512_load_ps(&(b->element[1][index]));
		b8[thread_index][2] = _mm512_load_ps(&(b->element[2][index]));
		b8[thread_index][3] = _mm512_load_ps(&(b->element[3][index]));

		_bncavx512_rqs_mul(tmp8[thread_index], a8[thread_index], b8[thread_index]);
		#pragma omp critical
			_bncavx512_rqs_add(ret8, ret8, tmp8[thread_index]);
	}
	_bncavx512_rqs_sum512(ret, ret8);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntw(); long _N = a->real_dim; long _ix;
		svfloat32_t _acc0 = svdup_n_f32(0.0);
		svfloat32_t _acc1 = svdup_n_f32(0.0);
		svfloat32_t _acc2 = svdup_n_f32(0.0);
		svfloat32_t _acc3 = svdup_n_f32(0.0);
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b32_s32((int32_t)_ix, (int32_t)_N);
			svfloat32_t _a0 = svld1_f32(_pg, &(a->element[0][_ix]));
			svfloat32_t _a1 = svld1_f32(_pg, &(a->element[1][_ix]));
			svfloat32_t _a2 = svld1_f32(_pg, &(a->element[2][_ix]));
			svfloat32_t _a3 = svld1_f32(_pg, &(a->element[3][_ix]));
			svfloat32_t _b0 = svld1_f32(_pg, &(b->element[0][_ix]));
			svfloat32_t _b1 = svld1_f32(_pg, &(b->element[1][_ix]));
			svfloat32_t _b2 = svld1_f32(_pg, &(b->element[2][_ix]));
			svfloat32_t _b3 = svld1_f32(_pg, &(b->element[3][_ix]));
			svfloat32_t _m0, _m1, _m2, _m3;
			_bncsve2_rqs_mul(_pg, &_m0, &_m1, &_m2, &_m3, _a0, _a1, _a2, _a3, _b0, _b1, _b2, _b3);
			_bncsve2_rqs_add(_pg, &_acc0, &_acc1, &_acc2, &_acc3, _acc0, _acc1, _acc2, _acc3, _m0, _m1, _m2, _m3);
		}
		{
			long _L, _vl = (long)svcntw();
			float _la0[64], _la1[64], _la2[64], _la3[64];
			svst1_f32(svptrue_b32(), _la0, _acc0);
			svst1_f32(svptrue_b32(), _la1, _acc1);
			svst1_f32(svptrue_b32(), _la2, _acc2);
			svst1_f32(svptrue_b32(), _la3, _acc3);
			rqs_set_ui(ret, 0UL);
			for(_L = 0; _L < _vl; _L++)
			{
				float _lane[QSSIZE] = { _la0[_L], _la1[_L], _la2[_L], _la3[_L] };
				rqs_add(ret, ret, _lane);
			}
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	float32x4_t in_a_val[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	float32x4_t in_b_val[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	float32x4_t in_mul[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	float32x4_t sum_neon[QSSIZE];

	// 和を0で初期化
	_bncneon_rqs_set0(sum_neon);

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();

		// a, b の値をロード
		in_a_val[thread_index][0] = vld1q_f32(&(a->element[0][index]));
		in_a_val[thread_index][1] = vld1q_f32(&(a->element[1][index]));
		in_a_val[thread_index][2] = vld1q_f32(&(a->element[2][index]));
		in_a_val[thread_index][3] = vld1q_f32(&(a->element[3][index]));
		in_b_val[thread_index][0] = vld1q_f32(&(b->element[0][index]));
		in_b_val[thread_index][1] = vld1q_f32(&(b->element[1][index]));
		in_b_val[thread_index][2] = vld1q_f32(&(b->element[2][index]));
		in_b_val[thread_index][3] = vld1q_f32(&(b->element[3][index]));

		// Neon QS乗算
		_bncneon_rqs_mul(in_mul[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

		// 累積加算 (critical sectionで保護)
		#pragma omp critical
		{
			_bncneon_rqs_add(sum_neon, sum_neon, in_mul[thread_index]);
		}
	}
	// fix! 2026-02-25(Wed) T.Kouya
	/* Inline 4-lane horizontal sum (was _bncneon_rqs_sum128f). */
	{
		float _l0[QSSIZE] = { vgetq_lane_f32(sum_neon[0], 0), vgetq_lane_f32(sum_neon[1], 0), vgetq_lane_f32(sum_neon[2], 0), vgetq_lane_f32(sum_neon[3], 0) };
		float _l1[QSSIZE] = { vgetq_lane_f32(sum_neon[0], 1), vgetq_lane_f32(sum_neon[1], 1), vgetq_lane_f32(sum_neon[2], 1), vgetq_lane_f32(sum_neon[3], 1) };
		float _l2[QSSIZE] = { vgetq_lane_f32(sum_neon[0], 2), vgetq_lane_f32(sum_neon[1], 2), vgetq_lane_f32(sum_neon[2], 2), vgetq_lane_f32(sum_neon[3], 2) };
		float _l3[QSSIZE] = { vgetq_lane_f32(sum_neon[0], 3), vgetq_lane_f32(sum_neon[1], 3), vgetq_lane_f32(sum_neon[2], 3), vgetq_lane_f32(sum_neon[3], 3) };
		rqs_set(ret, _l0);
		rqs_add(ret, ret, _l1);
		rqs_add(ret, ret, _l2);
		rqs_add(ret, ret, _l3);
	}
#else // others
	float tmp[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	set0_qs(ret);

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < a->dim; i++)
	{
		thread_index = omp_get_thread_num();

		rqs_mul(tmp[thread_index], get_qsvector_i(a, i), get_qsvector_i(b, i));

	#pragma omp critical
		rqs_add(ret, ret, tmp[thread_index]);

	}
#endif // __AVX2__

	return;
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void _bncomp_row_swap_qsmatrix(QSMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
	long int i, j, true_end;
	float tmp[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	int thread_index;

	true_end = (col_end > mat->col_dim) ? mat->col_dim : col_end;

	#pragma omp parallel for private(thread_index)
	for(i = col_start; i < true_end; i++)
	{
		thread_index = omp_get_thread_num();

		rqs_set(tmp[thread_index], get_qsmatrix_ij(mat, row_index0, i));
		set_qsmatrix_ij(mat, row_index0, i, get_qsmatrix_ij(mat, row_index1, i));
		set_qsmatrix_ij(mat, row_index1, i, tmp[thread_index]);
	}
}

//--------
// Matrix
//--------
/* c := a + b */
void _bncomp_add_qsmatrix(QSMatrix c, QSMatrix a, QSMatrix b)
{
	int thread_index;
	long int index, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_qsmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_qsmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 tmp4[BNCOMP_MAX_NUM_THREADS][QSSIZE], aij4[BNCOMP_MAX_NUM_THREADS][QSSIZE], bij4[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij4[thread_index][0] = _mm256_load_ps(&(a->element[0][index]));
		aij4[thread_index][1] = _mm256_load_ps(&(a->element[1][index]));
		aij4[thread_index][2] = _mm256_load_ps(&(a->element[2][index]));
		aij4[thread_index][3] = _mm256_load_ps(&(a->element[3][index]));
		bij4[thread_index][0] = _mm256_load_ps(&(b->element[0][index]));
		bij4[thread_index][1] = _mm256_load_ps(&(b->element[1][index]));
		bij4[thread_index][2] = _mm256_load_ps(&(b->element[2][index]));
		bij4[thread_index][3] = _mm256_load_ps(&(b->element[3][index]));

		_bncavx2_rqs_add(tmp4[thread_index], aij4[thread_index], bij4[thread_index]);

		_mm256_store_ps(&(c->element[0][index]), tmp4[thread_index][0]);
		_mm256_store_ps(&(c->element[1][index]), tmp4[thread_index][1]); 
		_mm256_store_ps(&(c->element[2][index]), tmp4[thread_index][2]); 
		_mm256_store_ps(&(c->element[3][index]), tmp4[thread_index][3]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 tmp8[BNCOMP_MAX_NUM_THREADS][QSSIZE], aij8[BNCOMP_MAX_NUM_THREADS][QSSIZE], bij8[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij8[thread_index][0] = _mm512_load_ps(&(a->element[0][index]));
		aij8[thread_index][1] = _mm512_load_ps(&(a->element[1][index]));
		aij8[thread_index][2] = _mm512_load_ps(&(a->element[2][index]));
		aij8[thread_index][3] = _mm512_load_ps(&(a->element[3][index]));
		bij8[thread_index][0] = _mm512_load_ps(&(b->element[0][index]));
		bij8[thread_index][1] = _mm512_load_ps(&(b->element[1][index]));
		bij8[thread_index][2] = _mm512_load_ps(&(b->element[2][index]));
		bij8[thread_index][3] = _mm512_load_ps(&(b->element[3][index]));

		_bncavx512_rqs_add(tmp8[thread_index], aij8[thread_index], bij8[thread_index]);

		_mm512_store_ps(&(c->element[0][index]), tmp8[thread_index][0]);
		_mm512_store_ps(&(c->element[1][index]), tmp8[thread_index][1]); 
		_mm512_store_ps(&(c->element[2][index]), tmp8[thread_index][2]); 
		_mm512_store_ps(&(c->element[3][index]), tmp8[thread_index][3]); 
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntw(); long _N = c->real_row_dim * c->real_col_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b32_s32((int32_t)_ix, (int32_t)_N);
			svfloat32_t _a0 = svld1_f32(_pg, &(a->element[0][_ix]));
			svfloat32_t _a1 = svld1_f32(_pg, &(a->element[1][_ix]));
			svfloat32_t _a2 = svld1_f32(_pg, &(a->element[2][_ix]));
			svfloat32_t _a3 = svld1_f32(_pg, &(a->element[3][_ix]));
			svfloat32_t _b0 = svld1_f32(_pg, &(b->element[0][_ix]));
			svfloat32_t _b1 = svld1_f32(_pg, &(b->element[1][_ix]));
			svfloat32_t _b2 = svld1_f32(_pg, &(b->element[2][_ix]));
			svfloat32_t _b3 = svld1_f32(_pg, &(b->element[3][_ix]));
			svfloat32_t _o0, _o1, _o2, _o3;
			_bncsve2_rqs_add(_pg, &_o0, &_o1, &_o2, &_o3, _a0, _a1, _a2, _a3, _b0, _b1, _b2, _b3);
			svst1_f32(_pg, &(c->element[0][_ix]), _o0);
			svst1_f32(_pg, &(c->element[1][_ix]), _o1);
			svst1_f32(_pg, &(c->element[2][_ix]), _o2);
			svst1_f32(_pg, &(c->element[3][_ix]), _o3);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	float32x4_t tmp_neon[BNCOMP_MAX_NUM_THREADS][QSSIZE], aij_neon[BNCOMP_MAX_NUM_THREADS][QSSIZE], bij_neon[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	#pragma omp parallel for private(thread_index) // , tmp_neon, aij_neon, bij_neon)
	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();

		aij_neon[thread_index][0] = vld1q_f32(&(a->element[0][index]));
		aij_neon[thread_index][1] = vld1q_f32(&(a->element[1][index]));
		aij_neon[thread_index][2] = vld1q_f32(&(a->element[2][index]));
		aij_neon[thread_index][3] = vld1q_f32(&(a->element[3][index]));
		bij_neon[thread_index][0] = vld1q_f32(&(b->element[0][index]));
		bij_neon[thread_index][1] = vld1q_f32(&(b->element[1][index]));
		bij_neon[thread_index][2] = vld1q_f32(&(b->element[2][index]));
		bij_neon[thread_index][3] = vld1q_f32(&(b->element[3][index]));

		_bncneon_rqs_add(tmp_neon[thread_index], aij_neon[thread_index], bij_neon[thread_index]);

		vst1q_f32(&(c->element[0][index]), tmp_neon[thread_index][0]);
		vst1q_f32(&(c->element[1][index]), tmp_neon[thread_index][1]); 
		vst1q_f32(&(c->element[2][index]), tmp_neon[thread_index][2]); 
		vst1q_f32(&(c->element[3][index]), tmp_neon[thread_index][3]); 
	}
#else // others
	float tmp[BNCOMP_MAX_NUM_THREADS][QSSIZE], aij[BNCOMP_MAX_NUM_THREADS][QSSIZE], bij[BNCOMP_MAX_NUM_THREADS][QSSIZE];

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

		rqs_add(tmp[thread_index], aij[thread_index], bij[thread_index]);

		c->element[0][index] = tmp[thread_index][0];
		c->element[1][index] = tmp[thread_index][1]; 
		c->element[2][index] = tmp[thread_index][2]; 
		c->element[3][index] = tmp[thread_index][3]; 
	}
#endif // __AVX2__
}

/* c := a - b */
void _bncomp_sub_qsmatrix(QSMatrix c, QSMatrix a, QSMatrix b)
{
	int thread_index;
	long int index, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_sub_qsmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_sub_qsmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 tmp4[BNCOMP_MAX_NUM_THREADS][QSSIZE], aij4[BNCOMP_MAX_NUM_THREADS][QSSIZE], bij4[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij4[thread_index][0] = _mm256_load_ps(&(a->element[0][index]));
		aij4[thread_index][1] = _mm256_load_ps(&(a->element[1][index]));
		aij4[thread_index][2] = _mm256_load_ps(&(a->element[2][index]));
		aij4[thread_index][3] = _mm256_load_ps(&(a->element[3][index]));
		bij4[thread_index][0] = _mm256_load_ps(&(b->element[0][index]));
		bij4[thread_index][1] = _mm256_load_ps(&(b->element[1][index]));
		bij4[thread_index][2] = _mm256_load_ps(&(b->element[2][index]));
		bij4[thread_index][3] = _mm256_load_ps(&(b->element[3][index]));

		_bncavx2_rqs_sub(tmp4[thread_index], aij4[thread_index], bij4[thread_index]);

		_mm256_store_ps(&(c->element[0][index]), tmp4[thread_index][0]);
		_mm256_store_ps(&(c->element[1][index]), tmp4[thread_index][1]); 
		_mm256_store_ps(&(c->element[2][index]), tmp4[thread_index][2]); 
		_mm256_store_ps(&(c->element[3][index]), tmp4[thread_index][3]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 tmp8[BNCOMP_MAX_NUM_THREADS][QSSIZE], aij8[BNCOMP_MAX_NUM_THREADS][QSSIZE], bij8[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij8[thread_index][0] = _mm512_load_ps(&(a->element[0][index]));
		aij8[thread_index][1] = _mm512_load_ps(&(a->element[1][index]));
		aij8[thread_index][2] = _mm512_load_ps(&(a->element[2][index]));
		aij8[thread_index][3] = _mm512_load_ps(&(a->element[3][index]));
		bij8[thread_index][0] = _mm512_load_ps(&(b->element[0][index]));
		bij8[thread_index][1] = _mm512_load_ps(&(b->element[1][index]));
		bij8[thread_index][2] = _mm512_load_ps(&(b->element[2][index]));
		bij8[thread_index][3] = _mm512_load_ps(&(b->element[3][index]));

		_bncavx512_rqs_sub(tmp8[thread_index], aij8[thread_index], bij8[thread_index]);

		_mm512_store_ps(&(c->element[0][index]), tmp8[thread_index][0]);
		_mm512_store_ps(&(c->element[1][index]), tmp8[thread_index][1]); 
		_mm512_store_ps(&(c->element[2][index]), tmp8[thread_index][2]); 
		_mm512_store_ps(&(c->element[3][index]), tmp8[thread_index][3]); 
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntw(); long _N = c->real_row_dim * c->real_col_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b32_s32((int32_t)_ix, (int32_t)_N);
			svfloat32_t _a0 = svld1_f32(_pg, &(a->element[0][_ix]));
			svfloat32_t _a1 = svld1_f32(_pg, &(a->element[1][_ix]));
			svfloat32_t _a2 = svld1_f32(_pg, &(a->element[2][_ix]));
			svfloat32_t _a3 = svld1_f32(_pg, &(a->element[3][_ix]));
			svfloat32_t _b0 = svld1_f32(_pg, &(b->element[0][_ix]));
			svfloat32_t _b1 = svld1_f32(_pg, &(b->element[1][_ix]));
			svfloat32_t _b2 = svld1_f32(_pg, &(b->element[2][_ix]));
			svfloat32_t _b3 = svld1_f32(_pg, &(b->element[3][_ix]));
			_b0 = svneg_f32_x(_pg, _b0);
			_b1 = svneg_f32_x(_pg, _b1);
			_b2 = svneg_f32_x(_pg, _b2);
			_b3 = svneg_f32_x(_pg, _b3);
			svfloat32_t _o0, _o1, _o2, _o3;
			_bncsve2_rqs_add(_pg, &_o0, &_o1, &_o2, &_o3, _a0, _a1, _a2, _a3, _b0, _b1, _b2, _b3);
			svst1_f32(_pg, &(c->element[0][_ix]), _o0);
			svst1_f32(_pg, &(c->element[1][_ix]), _o1);
			svst1_f32(_pg, &(c->element[2][_ix]), _o2);
			svst1_f32(_pg, &(c->element[3][_ix]), _o3);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
// Arm Neon版
	float32x4_t tmp_neon[BNCOMP_MAX_NUM_THREADS][QSSIZE], aij_neon[BNCOMP_MAX_NUM_THREADS][QSSIZE], bij_neon[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	// 並列化の閾値判定（行数で判定）
	//if(row_dim >= PARALLEL_THRESHOLD_MATRIX)
	{
		#pragma omp parallel for private(tmp_neon, aij_neon, bij_neon)
		for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
		{
			thread_index = omp_get_thread_num();

			aij_neon[thread_index][0] = vld1q_f32(&(a->element[0][index]));
			aij_neon[thread_index][1] = vld1q_f32(&(a->element[1][index]));
			aij_neon[thread_index][2] = vld1q_f32(&(a->element[2][index]));
			aij_neon[thread_index][3] = vld1q_f32(&(a->element[3][index]));
			bij_neon[thread_index][0] = vld1q_f32(&(b->element[0][index]));
			bij_neon[thread_index][1] = vld1q_f32(&(b->element[1][index]));
			bij_neon[thread_index][2] = vld1q_f32(&(b->element[2][index]));
			bij_neon[thread_index][3] = vld1q_f32(&(b->element[3][index]));

			_bncneon_rqs_sub(tmp_neon[thread_index], aij_neon[thread_index], bij_neon[thread_index]);

			vst1q_f32(&(c->element[0][index]), tmp_neon[thread_index][0]);
			vst1q_f32(&(c->element[1][index]), tmp_neon[thread_index][1]); 
			vst1q_f32(&(c->element[2][index]), tmp_neon[thread_index][2]); 
			vst1q_f32(&(c->element[3][index]), tmp_neon[thread_index][3]); 
		}
	}
#else // others
	float tmp[BNCOMP_MAX_NUM_THREADS][QSSIZE], aij[BNCOMP_MAX_NUM_THREADS][QSSIZE], bij[BNCOMP_MAX_NUM_THREADS][QSSIZE];

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

		rqs_sub(tmp[thread_index], aij[thread_index], bij[thread_index]);

		c->element[0][index] = tmp[thread_index][0];
		c->element[1][index] = tmp[thread_index][1]; 
		c->element[2][index] = tmp[thread_index][2]; 
		c->element[3][index] = tmp[thread_index][3]; 
	}
#endif // __AVX2__
}

/* c := sc * a */
void _bncomp_cmul_qsmatrix(QSMatrix c, float sc[QSSIZE], QSMatrix a)
{
	long int i, j, index, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim;
	int thread_index;

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_qsmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_qsmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 tmp4[BNCOMP_MAX_NUM_THREADS][QSSIZE], aij4[BNCOMP_MAX_NUM_THREADS][QSSIZE], sc4[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		sc4[thread_index][0] = _mm256_set1_ps(sc[0]);
		sc4[thread_index][1] = _mm256_set1_ps(sc[1]);
		sc4[thread_index][2] = _mm256_set1_ps(sc[2]);
		sc4[thread_index][3] = _mm256_set1_ps(sc[3]);
	}
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij4[thread_index][0] = _mm256_load_ps(&(a->element[0][index]));
		aij4[thread_index][1] = _mm256_load_ps(&(a->element[1][index]));
		aij4[thread_index][2] = _mm256_load_ps(&(a->element[2][index]));
		aij4[thread_index][3] = _mm256_load_ps(&(a->element[3][index]));

		_bncavx2_rqs_mul(tmp4[thread_index], sc4[thread_index], aij4[thread_index]);

		_mm256_store_ps(&(c->element[0][index]), tmp4[thread_index][0]);
		_mm256_store_ps(&(c->element[1][index]), tmp4[thread_index][1]); 
		_mm256_store_ps(&(c->element[2][index]), tmp4[thread_index][2]); 
		_mm256_store_ps(&(c->element[3][index]), tmp4[thread_index][3]); 
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 tmp8[BNCOMP_MAX_NUM_THREADS][QSSIZE], aij8[BNCOMP_MAX_NUM_THREADS][QSSIZE], sc8[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		sc8[thread_index][0] = _mm512_set1_ps(sc[0]);
		sc8[thread_index][1] = _mm512_set1_ps(sc[1]);
		sc8[thread_index][2] = _mm512_set1_ps(sc[2]);
		sc8[thread_index][3] = _mm512_set1_ps(sc[3]);
	}
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij8[thread_index][0] = _mm512_load_ps(&(a->element[0][index]));
		aij8[thread_index][1] = _mm512_load_ps(&(a->element[1][index]));
		aij8[thread_index][2] = _mm512_load_ps(&(a->element[2][index]));
		aij8[thread_index][3] = _mm512_load_ps(&(a->element[3][index]));

		_bncavx512_rqs_mul(tmp8[thread_index], sc8[thread_index], aij8[thread_index]);

		_mm512_store_ps(&(c->element[0][index]), tmp8[thread_index][0]);
		_mm512_store_ps(&(c->element[1][index]), tmp8[thread_index][1]); 
		_mm512_store_ps(&(c->element[2][index]), tmp8[thread_index][2]); 
		_mm512_store_ps(&(c->element[3][index]), tmp8[thread_index][3]); 
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntw(); long _N = a->real_row_dim * a->real_col_dim; long _ix;
		svfloat32_t _v0 = svdup_n_f32(sc[0]);
		svfloat32_t _v1 = svdup_n_f32(sc[1]);
		svfloat32_t _v2 = svdup_n_f32(sc[2]);
		svfloat32_t _v3 = svdup_n_f32(sc[3]);
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b32_s32((int32_t)_ix, (int32_t)_N);
			svfloat32_t _a0 = svld1_f32(_pg, &(a->element[0][_ix]));
			svfloat32_t _a1 = svld1_f32(_pg, &(a->element[1][_ix]));
			svfloat32_t _a2 = svld1_f32(_pg, &(a->element[2][_ix]));
			svfloat32_t _a3 = svld1_f32(_pg, &(a->element[3][_ix]));
			svfloat32_t _o0, _o1, _o2, _o3;
			_bncsve2_rqs_mul(_pg, &_o0, &_o1, &_o2, &_o3, _v0, _v1, _v2, _v3, _a0, _a1, _a2, _a3);
			svst1_f32(_pg, &(c->element[0][_ix]), _o0);
			svst1_f32(_pg, &(c->element[1][_ix]), _o1);
			svst1_f32(_pg, &(c->element[2][_ix]), _o2);
			svst1_f32(_pg, &(c->element[3][_ix]), _o3);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	float32x4_t tmp_neon[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	float32x4_t aij_neon[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	float32x4_t sc_neon[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		// スカラーをブロードキャスト
		_bncneon_rqs_set1_qs(sc_neon[thread_index], sc);
	}
	
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		// ロード
		aij_neon[thread_index][0] = vld1q_f32(&(a->element[0][index]));
		aij_neon[thread_index][1] = vld1q_f32(&(a->element[1][index]));
		aij_neon[thread_index][2] = vld1q_f32(&(a->element[2][index]));
		aij_neon[thread_index][3] = vld1q_f32(&(a->element[3][index]));

		// Neon QS乗算
		_bncneon_rqs_mul(tmp_neon[thread_index], sc_neon[thread_index], aij_neon[thread_index]);

		// ストア
		vst1q_f32(&(c->element[0][index]), tmp_neon[thread_index][0]);
		vst1q_f32(&(c->element[1][index]), tmp_neon[thread_index][1]); 
		vst1q_f32(&(c->element[2][index]), tmp_neon[thread_index][2]); 
		vst1q_f32(&(c->element[3][index]), tmp_neon[thread_index][3]); 
	}
#else // others
	float tmp[BNCOMP_MAX_NUM_THREADS][QSSIZE], aij[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index++)
	{
		thread_index = omp_get_thread_num();
		
		aij[thread_index][0] = a->element[0][index];
		aij[thread_index][1] = a->element[1][index];
		aij[thread_index][2] = a->element[2][index];
		aij[thread_index][3] = a->element[3][index];

		rqs_mul(tmp[thread_index], sc, aij[thread_index]);

		c->element[0][index] = tmp[thread_index][0];
		c->element[1][index] = tmp[thread_index][1]; 
		c->element[2][index] = tmp[thread_index][2]; 
		c->element[3][index] = tmp[thread_index][3]; 
	}
#endif // __AVX2__
}

/* c = a * b */
void _bncomp_mul_qsmatrix(QSMatrix ret, QSMatrix a, QSMatrix b)
{
	int thread_num, thread_index;
	long int i, j, k;
	long row_dim, col_dim, mid_dim;

	/* dimension check */
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_qsmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret->row_dim, ret->col_dim, a->row_dim, a->col_dim, b->row_dim, b->col_dim);
		return;
	}
	row_dim = a->row_dim;
	col_dim = b->col_dim;
	mid_dim = a->col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long real_row_dim, real_col_dim, real_mid_dim;
	float cijval[BNCOMP_MAX_NUM_THREADS][8][QSSIZE];
    __m256 cij[BNCOMP_MAX_NUM_THREADS][QSSIZE], aik[BNCOMP_MAX_NUM_THREADS][QSSIZE], bkj[BNCOMP_MAX_NUM_THREADS][QSSIZE], tmp_mul[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		//set0_ds(tmp[thread_index]);
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
            cij[thread_index][0] = _mm256_setzero_ps();
            cij[thread_index][1] = _mm256_setzero_ps();
            cij[thread_index][2] = _mm256_setzero_ps();
            cij[thread_index][3] = _mm256_setzero_ps();

            for(k = 0; k < real_mid_dim; k += _BNC_S_WIDTH)
            {
                aik[thread_index][0] = _mm256_load_ps(&(a->element[0][i * real_mid_dim + k]));
                aik[thread_index][1] = _mm256_load_ps(&(a->element[1][i * real_mid_dim + k]));
                aik[thread_index][2] = _mm256_load_ps(&(a->element[2][i * real_mid_dim + k]));
                aik[thread_index][3] = _mm256_load_ps(&(a->element[3][i * real_mid_dim + k]));

                bkj[thread_index][0] = _mm256_set_ps(
                    b->element[0][(k + 7) * real_col_dim + j],
                    b->element[0][(k + 6) * real_col_dim + j],
                    b->element[0][(k + 5) * real_col_dim + j],
                    b->element[0][(k + 4) * real_col_dim + j],
                    b->element[0][(k + 3) * real_col_dim + j],
                    b->element[0][(k + 2) * real_col_dim + j],
                    b->element[0][(k + 1) * real_col_dim + j],
                    b->element[0][(k    ) * real_col_dim + j]
                );
				bkj[thread_index][1] = _mm256_set_ps(
                    b->element[1][(k + 7) * real_col_dim + j],
                    b->element[1][(k + 6) * real_col_dim + j],
                    b->element[1][(k + 5) * real_col_dim + j],
                    b->element[1][(k + 4) * real_col_dim + j],
                    b->element[1][(k + 3) * real_col_dim + j],
                    b->element[1][(k + 2) * real_col_dim + j],
                    b->element[1][(k + 1) * real_col_dim + j],
                    b->element[1][(k    ) * real_col_dim + j]
                );
				bkj[thread_index][2] = _mm256_set_ps(
                    b->element[2][(k + 7) * real_col_dim + j],
                    b->element[2][(k + 6) * real_col_dim + j],
                    b->element[2][(k + 5) * real_col_dim + j],
                    b->element[2][(k + 4) * real_col_dim + j],
                    b->element[2][(k + 3) * real_col_dim + j],
                    b->element[2][(k + 2) * real_col_dim + j],
                    b->element[2][(k + 1) * real_col_dim + j],
                    b->element[2][(k    ) * real_col_dim + j]
                );
				bkj[thread_index][3] = _mm256_set_ps(
                    b->element[3][(k + 7) * real_col_dim + j],
                    b->element[3][(k + 6) * real_col_dim + j],
                    b->element[3][(k + 5) * real_col_dim + j],
                    b->element[3][(k + 4) * real_col_dim + j],
                    b->element[3][(k + 3) * real_col_dim + j],
                    b->element[3][(k + 2) * real_col_dim + j],
                    b->element[3][(k + 1) * real_col_dim + j],
                    b->element[3][(k    ) * real_col_dim + j]
                );

                _bncavx2_rqs_mul(tmp_mul[thread_index], aik[thread_index], bkj[thread_index]);
                _bncavx2_rqs_add(cij[thread_index], cij[thread_index], tmp_mul[thread_index]);
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

            rqs_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][1]);
            rqs_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][2]);
            rqs_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][3]);
            rqs_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][4]);
            rqs_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][5]);
            rqs_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][6]);
            rqs_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][7]);

            ret->element[0][i * real_col_dim + j] = cijval[thread_index][0][0];
            ret->element[1][i * real_col_dim + j] = cijval[thread_index][0][1];
            ret->element[2][i * real_col_dim + j] = cijval[thread_index][0][2];
            ret->element[3][i * real_col_dim + j] = cijval[thread_index][0][3];
        }
    }
//#else // 0
#if 0
	_bncomp_set0_qsmatrix(ret);
	// From codes of Rgemm
	#pragma omp parallel for private(thread_index, k, i)
	//for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
	for(j = 0; j < col_dim; j++)
	{
		thread_index = omp_get_thread_num();
		for(k = 0; k < real_mid_dim; k +=  _BNC_S_WIDTH)
		{
			//rqs_set(b_kj[thread_index], get_qsmatrix_ij(b, k, j));
			bkj[thread_index][0] = _mm256_set_ps(
                b->element[0][(k + 7) * real_col_dim + j],
                b->element[0][(k + 6) * real_col_dim + j],
                b->element[0][(k + 5) * real_col_dim + j],
                b->element[0][(k + 4) * real_col_dim + j],
                b->element[0][(k + 3) * real_col_dim + j],
                b->element[0][(k + 2) * real_col_dim + j],
                b->element[0][(k + 1) * real_col_dim + j],
                b->element[0][(k    ) * real_col_dim + j]
            );
			bkj[thread_index][1] = _mm256_set_ps(
                b->element[1][(k + 7) * real_col_dim + j],
                b->element[1][(k + 6) * real_col_dim + j],
                b->element[1][(k + 5) * real_col_dim + j],
                b->element[1][(k + 4) * real_col_dim + j],
                b->element[1][(k + 3) * real_col_dim + j],
                b->element[1][(k + 2) * real_col_dim + j],
                b->element[1][(k + 1) * real_col_dim + j],
                b->element[1][(k    ) * real_col_dim + j]
            );
			bkj[thread_index][2] = _mm256_set_ps(
                b->element[2][(k + 7) * real_col_dim + j],
                b->element[2][(k + 6) * real_col_dim + j],
                b->element[2][(k + 5) * real_col_dim + j],
                b->element[2][(k + 4) * real_col_dim + j],
                b->element[2][(k + 3) * real_col_dim + j],
                b->element[2][(k + 2) * real_col_dim + j],
                b->element[2][(k + 1) * real_col_dim + j],
                b->element[2][(k    ) * real_col_dim + j]
            );
			bkj[thread_index][3] = _mm256_set_ps(
                b->element[3][(k + 7) * real_col_dim + j],
                b->element[3][(k + 6) * real_col_dim + j],
                b->element[3][(k + 5) * real_col_dim + j],
                b->element[3][(k + 4) * real_col_dim + j],
                b->element[3][(k + 3) * real_col_dim + j],
                b->element[3][(k + 2) * real_col_dim + j],
                b->element[3][(k + 1) * real_col_dim + j],
                b->element[3][(k    ) * real_col_dim + j]
            );
			for(i = 0; i < row_dim; i++)
			{
				//rqs_mul(tmp[thread_index], get_qsmatrix_ij(a, i, k), b_kj[thread_index]);
                aik[thread_index][0] = _mm256_load_ps(&(a->element[0][i * real_mid_dim + k]));
                aik[thread_index][1] = _mm256_load_ps(&(a->element[1][i * real_mid_dim + k]));
                aik[thread_index][2] = _mm256_load_ps(&(a->element[2][i * real_mid_dim + k]));
                aik[thread_index][3] = _mm256_load_ps(&(a->element[3][i * real_mid_dim + k]));
                _bncavx2_rqs_mul(tmp_mul[thread_index], aik[thread_index], bkj[thread_index]);

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

				//rqs_add(get_qsmatrix_ij(ret, i, j), get_qsmatrix_ij(ret, i, j), tmp[thread_index]);
				rqs_add(get_qsmatrix_ij(ret, i, j), get_qsmatrix_ij(ret, i, j), cijval[thread_index][0]);
				rqs_add(get_qsmatrix_ij(ret, i, j), get_qsmatrix_ij(ret, i, j), cijval[thread_index][1]);
				rqs_add(get_qsmatrix_ij(ret, i, j), get_qsmatrix_ij(ret, i, j), cijval[thread_index][2]);
				rqs_add(get_qsmatrix_ij(ret, i, j), get_qsmatrix_ij(ret, i, j), cijval[thread_index][3]);
			}
		}
	}
	//printf("end(%ld, %ld)\n", ret->row_dim, ret->col_dim);#endif // 0
#endif //0
#elif defined(__AVX512F__) // __AVX512F__
	long real_row_dim, real_col_dim, real_mid_dim;
	float cijval[BNCOMP_MAX_NUM_THREADS][16][QSSIZE];
    __m512 cij[BNCOMP_MAX_NUM_THREADS][QSSIZE], aik[BNCOMP_MAX_NUM_THREADS][QSSIZE], bkj[BNCOMP_MAX_NUM_THREADS][QSSIZE], tmp_mul[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, i, j, k, aik, bkj, cij, cijval, tmp_mul)
    for(i = 0; i < real_row_dim; i++)
    {
		thread_index = omp_get_thread_num();

		for(j = 0; j < real_col_dim; j++)
        {
            cij[thread_index][0] = _mm512_setzero_ps();
            cij[thread_index][1] = _mm512_setzero_ps();
            cij[thread_index][2] = _mm512_setzero_ps();
            cij[thread_index][3] = _mm512_setzero_ps();

            for(k = 0; k < real_mid_dim; k += _BNC_S_WIDTH)
            {
                aik[thread_index][0] = _mm512_load_ps(&(a->element[0][i * real_mid_dim + k]));
                aik[thread_index][1] = _mm512_load_ps(&(a->element[1][i * real_mid_dim + k]));
                aik[thread_index][2] = _mm512_load_ps(&(a->element[2][i * real_mid_dim + k]));
                aik[thread_index][3] = _mm512_load_ps(&(a->element[3][i * real_mid_dim + k]));

                bkj[thread_index][0] = _mm512_set_ps(
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
                    b->element[0][(k    ) * real_col_dim + j]
                );
                bkj[thread_index][1] = _mm512_set_ps(
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
                    b->element[1][(k    ) * real_col_dim + j]
                );
                bkj[thread_index][2] = _mm512_set_ps(
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
                    b->element[2][(k    ) * real_col_dim + j]
                );
                bkj[thread_index][3] = _mm512_set_ps(
                    b->element[3][(k + 15) * real_col_dim + j],
                    b->element[3][(k + 14) * real_col_dim + j],
                    b->element[3][(k + 13) * real_col_dim + j],
                    b->element[3][(k + 12) * real_col_dim + j],
                    b->element[3][(k + 11) * real_col_dim + j],
                    b->element[3][(k + 10) * real_col_dim + j],
                    b->element[3][(k + 9) * real_col_dim + j],
                    b->element[3][(k + 8) * real_col_dim + j],
                    b->element[3][(k + 7) * real_col_dim + j],
                    b->element[3][(k + 6) * real_col_dim + j],
                    b->element[3][(k + 5) * real_col_dim + j],
                    b->element[3][(k + 4) * real_col_dim + j],
                    b->element[3][(k + 3) * real_col_dim + j],
                    b->element[3][(k + 2) * real_col_dim + j],
                    b->element[3][(k + 1) * real_col_dim + j],
                    b->element[3][(k    ) * real_col_dim + j]
                );

                _bncavx512_rqs_mul(tmp_mul[thread_index], aik[thread_index], bkj[thread_index]);
                _bncavx512_rqs_add(cij[thread_index], cij[thread_index], tmp_mul[thread_index]);
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

            cijval[thread_index][8][0] = cij[thread_index][0][8];
			cijval[thread_index][8][1] = cij[thread_index][1][8];
			cijval[thread_index][8][2] = cij[thread_index][2][8];
			cijval[thread_index][8][3] = cij[thread_index][3][8];

            cijval[thread_index][9][0] = cij[thread_index][0][9];
			cijval[thread_index][9][1] = cij[thread_index][1][9];
			cijval[thread_index][9][2] = cij[thread_index][2][9];
			cijval[thread_index][9][3] = cij[thread_index][3][9];

            cijval[thread_index][10][0] = cij[thread_index][0][10];
			cijval[thread_index][10][1] = cij[thread_index][1][10];
			cijval[thread_index][10][2] = cij[thread_index][2][10];
			cijval[thread_index][10][3] = cij[thread_index][3][10];

            cijval[thread_index][11][0] = cij[thread_index][0][11];
			cijval[thread_index][11][1] = cij[thread_index][1][11];
			cijval[thread_index][11][2] = cij[thread_index][2][11];
			cijval[thread_index][11][3] = cij[thread_index][3][11];

            cijval[thread_index][12][0] = cij[thread_index][0][12];
			cijval[thread_index][12][1] = cij[thread_index][1][12];
			cijval[thread_index][12][2] = cij[thread_index][2][12];
			cijval[thread_index][12][3] = cij[thread_index][3][12];

            cijval[thread_index][13][0] = cij[thread_index][0][13];
			cijval[thread_index][13][1] = cij[thread_index][1][13];
			cijval[thread_index][13][2] = cij[thread_index][2][13];
			cijval[thread_index][13][3] = cij[thread_index][3][13];

            cijval[thread_index][14][0] = cij[thread_index][0][14];
			cijval[thread_index][14][1] = cij[thread_index][1][14];
			cijval[thread_index][14][2] = cij[thread_index][2][14];
			cijval[thread_index][14][3] = cij[thread_index][3][14];

            cijval[thread_index][15][0] = cij[thread_index][0][15];
			cijval[thread_index][15][1] = cij[thread_index][1][15];
			cijval[thread_index][15][2] = cij[thread_index][2][15];
			cijval[thread_index][15][3] = cij[thread_index][3][15];

            rqs_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][1]);
            rqs_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][2]);
            rqs_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][3]);
            rqs_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][4]);
            rqs_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][5]);
            rqs_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][6]);
            rqs_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][7]);
            rqs_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][8]);
            rqs_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][9]);
            rqs_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][10]);
            rqs_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][11]);
            rqs_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][12]);
            rqs_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][13]);
            rqs_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][14]);
            rqs_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][15]);

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
		long vl = (long)svcntw();

		#pragma omp parallel for private(thread_index, i, j, k)
		for(i = 0; i < real_row_dim; i++){
			thread_index = omp_get_thread_num();
			(void)thread_index;
			for(j = 0; j < real_col_dim; j += vl){
				svbool_t pg = svwhilelt_b32_s32((int32_t)j, (int32_t)real_col_dim);
				svfloat32_t cij0, cij1, cij2, cij3;
				_bncsve2_rqs_set0(&cij0, &cij1, &cij2, &cij3);
				for(k = 0; k < real_mid_dim; k++){
					svfloat32_t aik0 = svdup_n_f32(a->element[0][i*real_mid_dim + k]);
					svfloat32_t aik1 = svdup_n_f32(a->element[1][i*real_mid_dim + k]);
					svfloat32_t aik2 = svdup_n_f32(a->element[2][i*real_mid_dim + k]);
					svfloat32_t aik3 = svdup_n_f32(a->element[3][i*real_mid_dim + k]);
					svfloat32_t bkj0 = svld1_f32(pg, &(b->element[0][k*real_col_dim + j]));
					svfloat32_t bkj1 = svld1_f32(pg, &(b->element[1][k*real_col_dim + j]));
					svfloat32_t bkj2 = svld1_f32(pg, &(b->element[2][k*real_col_dim + j]));
					svfloat32_t bkj3 = svld1_f32(pg, &(b->element[3][k*real_col_dim + j]));
					svfloat32_t t0, t1, t2, t3;
					_bncsve2_rqs_mul(pg, &t0, &t1, &t2, &t3,
					                 aik0, aik1, aik2, aik3,
					                 bkj0, bkj1, bkj2, bkj3);
					_bncsve2_rqs_add(pg, &cij0, &cij1, &cij2, &cij3,
					                 cij0, cij1, cij2, cij3,
					                 t0, t1, t2, t3);
				}
				svst1_f32(pg, &(ret->element[0][i*real_col_dim + j]), cij0);
				svst1_f32(pg, &(ret->element[1][i*real_col_dim + j]), cij1);
				svst1_f32(pg, &(ret->element[2][i*real_col_dim + j]), cij2);
				svst1_f32(pg, &(ret->element[3][i*real_col_dim + j]), cij3);
			}
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	/* NEON (float32x4_t): outer-product GEMM, mirroring serial mul_qsmatrix(). */
	long real_row_dim, real_col_dim, real_mid_dim;
	float32x4_t cij[BNCOMP_MAX_NUM_THREADS][QSSIZE], aik[BNCOMP_MAX_NUM_THREADS][QSSIZE], bkj[BNCOMP_MAX_NUM_THREADS][QSSIZE], tmp_mul[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, i, j, k, aik, bkj, cij, tmp_mul) schedule(dynamic)
	for(i = 0; i < real_row_dim; i++)
	{
		thread_index = omp_get_thread_num();
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			_bncneon_set0_qs(cij[thread_index]);
			for(k = 0; k < real_mid_dim; k++)
			{
				aik[thread_index][0] = vdupq_n_f32(a->element[0][i * real_mid_dim + k]);
				aik[thread_index][1] = vdupq_n_f32(a->element[1][i * real_mid_dim + k]);
				aik[thread_index][2] = vdupq_n_f32(a->element[2][i * real_mid_dim + k]);
				aik[thread_index][3] = vdupq_n_f32(a->element[3][i * real_mid_dim + k]);
				bkj[thread_index][0] = vld1q_f32(&(b->element[0][k * real_col_dim + j]));
				bkj[thread_index][1] = vld1q_f32(&(b->element[1][k * real_col_dim + j]));
				bkj[thread_index][2] = vld1q_f32(&(b->element[2][k * real_col_dim + j]));
				bkj[thread_index][3] = vld1q_f32(&(b->element[3][k * real_col_dim + j]));
				_bncneon_rqs_mul(tmp_mul[thread_index], aik[thread_index], bkj[thread_index]);
				_bncneon_rqs_add(cij[thread_index], cij[thread_index], tmp_mul[thread_index]);
			}
			vst1q_f32(&(ret->element[0][i * real_col_dim + j]), cij[thread_index][0]);
			vst1q_f32(&(ret->element[1][i * real_col_dim + j]), cij[thread_index][1]);
			vst1q_f32(&(ret->element[2][i * real_col_dim + j]), cij[thread_index][2]);
			vst1q_f32(&(ret->element[3][i * real_col_dim + j]), cij[thread_index][3]);
		}
	}
#else // __AVX2__
	float tmp[BNCOMP_MAX_NUM_THREADS][QSSIZE], ret_ij[BNCOMP_MAX_NUM_THREADS][QSSIZE], b_kj[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_qs(tmp[thread_index]);
	}

	//printf("Non SIMD mul_qsmatrix(%ld, %ld)\n", ret->row_dim, ret->col_dim);
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
			rqs_set0(ret_ij[thread_index]);
			for(k = 0; k < mid_dim; k++)
			{
				//#pragma omp critical
				//{
				rqs_mul(tmp[thread_index], get_qsmatrix_ij(a, i, k), get_qsmatrix_ij(b, k, j));
				// fixed! 2024-02-22 T.Kouya
				#pragma omp critical
					rqs_add(ret_ij[thread_index], tmp[thread_index], ret_ij[thread_index]);
				//}
			}
			
			set_qsmatrix_ij(ret, i, j, ret_ij[thread_index]);
		}
	}	//printf("end(%ld, %ld)\n", ret->row_dim, ret->col_dim);
//#else // 0
/*	_bncomp_set0_qsmatrix(ret);
	// From codes of Rgemm
	#pragma omp parallel for private(thread_index, k, i)
	for(j = 0; j < col_dim; j++)
	{
		thread_index = omp_get_thread_num();
		for(k = 0; k < mid_dim; k++)
		{
			rqs_set(b_kj[thread_index], get_qsmatrix_ij(b, k, j));
			for(i = 0; i < row_dim; i++)
			{
				rqs_mul(tmp[thread_index], get_qsmatrix_ij(a, i, k), b_kj[thread_index]);
				rqs_add(get_qsmatrix_ij(ret, i, j), get_qsmatrix_ij(ret, i, j), tmp[thread_index]);
			}
		}
	}
	//printf("end(%ld, %ld)\n", ret->row_dim, ret->col_dim);#endif // 0
*/
//#endif //0
#endif // __AVX2__

}

/* c := a */
void _bncomp_subst_qsmatrix(QSMatrix c, QSMatrix a)
{
	long int index;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_qsmatrix\n");
		return;
	}

// AVX2
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int real_total_dim;

	real_total_dim = c->real_row_dim * c->real_col_dim;

	#pragma omp parallel for
	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		_mm256_store_ps(&(c->element[0][index]), _mm256_load_ps(&(a->element[0][index])));
		_mm256_store_ps(&(c->element[1][index]), _mm256_load_ps(&(a->element[1][index])));
		_mm256_store_ps(&(c->element[2][index]), _mm256_load_ps(&(a->element[2][index])));
		_mm256_store_ps(&(c->element[3][index]), _mm256_load_ps(&(a->element[3][index])));
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int real_total_dim;

	real_total_dim = c->real_row_dim * c->real_col_dim;

	#pragma omp parallel for
	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		_mm512_store_ps(&(c->element[0][index]), _mm512_load_ps(&(a->element[0][index])));
		_mm512_store_ps(&(c->element[1][index]), _mm512_load_ps(&(a->element[1][index])));
		_mm512_store_ps(&(c->element[2][index]), _mm512_load_ps(&(a->element[2][index])));
		_mm512_store_ps(&(c->element[3][index]), _mm512_load_ps(&(a->element[3][index])));
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntw(); long _N = c->real_row_dim * c->real_col_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b32_s32((int32_t)_ix, (int32_t)_N);
			svst1_f32(_pg, &(c->element[0][_ix]), svld1_f32(_pg, &(a->element[0][_ix])));
			svst1_f32(_pg, &(c->element[1][_ix]), svld1_f32(_pg, &(a->element[1][_ix])));
			svst1_f32(_pg, &(c->element[2][_ix]), svld1_f32(_pg, &(a->element[2][_ix])));
			svst1_f32(_pg, &(c->element[3][_ix]), svld1_f32(_pg, &(a->element[3][_ix])));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	long int real_total_dim;

	real_total_dim = c->real_row_dim * c->real_col_dim;

	#pragma omp parallel for
	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		float32x4_t tmp0 = vld1q_f32(&(a->element[0][index]));
		float32x4_t tmp1 = vld1q_f32(&(a->element[1][index]));
		float32x4_t tmp2 = vld1q_f32(&(a->element[2][index]));
		float32x4_t tmp3 = vld1q_f32(&(a->element[3][index]));
		
		vst1q_f32(&(c->element[0][index]), tmp0);
		vst1q_f32(&(c->element[1][index]), tmp1);
		vst1q_f32(&(c->element[2][index]), tmp2);
		vst1q_f32(&(c->element[3][index]), tmp3);
	}
#else // others
	long int total_dim;

	total_dim = c->row_dim * c->col_dim;

	#pragma omp parallel for
	for(index = 0; index < total_dim; index++)
	{
		c->element[0][index] = a->element[0][index];
		c->element[1][index] = a->element[1][index];
		c->element[2][index] = a->element[2][index];
		c->element[3][index] = a->element[3][index];
	}

	//#pragma omp parallel for
	//for(index = 0; index < QSSIZE; index++)
	//	memcpy((void *)(c->element[index]), (void *)(a->element[index]), (size_t)(sizeof(float) * total_dim)); 

#endif // AVX2
}

/* c := I */
void _bncomp_setI_qsmatrix(QSMatrix c)
{
	long int i, real_total_dim;
	float tmp0[QSSIZE], tmp1[QSSIZE];

	rqs_set_ui(tmp0, 0UL);
	rqs_set_ui(tmp1, 1UL);

	real_total_dim = c->real_row_dim * c->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 zero4;

	zero4 = _mm256_setzero_ps();
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm256_store_ps(&(c->element[0][i]), zero4);
		_mm256_store_ps(&(c->element[1][i]), zero4);
		_mm256_store_ps(&(c->element[2][i]), zero4);
		_mm256_store_ps(&(c->element[3][i]), zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero8;

	zero8 = _mm512_setzero_ps();
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm512_store_ps(&(c->element[0][i]), zero8);
		_mm512_store_ps(&(c->element[1][i]), zero8);
		_mm512_store_ps(&(c->element[2][i]), zero8);
		_mm512_store_ps(&(c->element[3][i]), zero8);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntw(); long _N = c->real_row_dim * c->real_col_dim; long _ix;
		svfloat32_t _z = svdup_n_f32(0.0);
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b32_s32((int32_t)_ix, (int32_t)_N);
			svst1_f32(_pg, &(c->element[0][_ix]), _z);
			svst1_f32(_pg, &(c->element[1][_ix]), _z);
			svst1_f32(_pg, &(c->element[2][_ix]), _z);
			svst1_f32(_pg, &(c->element[3][_ix]), _z);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	float32x4_t zero_neon;

	zero_neon = vdupq_n_f32(0.0);
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		vst1q_f32(&(c->element[0][i]), zero_neon);
		vst1q_f32(&(c->element[1][i]), zero_neon);
		vst1q_f32(&(c->element[2][i]), zero_neon);
		vst1q_f32(&(c->element[3][i]), zero_neon);
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

	rqs_set_ui(tmp1, 1UL);

	#pragma omp parallel for
	for(i = 0; i < c->row_dim; i++)
	{
		if(i < c->col_dim)
			set_qsmatrix_ij(c, i, i, tmp1);
	}

}

// set a zero matrix
//void set0_qsmatrix(QSMatrix mat)
void _bncomp_set0_qsmatrix(QSMatrix mat)
{
	long int i, real_total_dim;

	real_total_dim = mat->real_row_dim * mat->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 zero4;

	zero4 = _mm256_setzero_ps();
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm256_store_ps(&(mat->element[0][i]), zero4);
		_mm256_store_ps(&(mat->element[1][i]), zero4);
		_mm256_store_ps(&(mat->element[2][i]), zero4);
		_mm256_store_ps(&(mat->element[3][i]), zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero8;

	zero8 = _mm512_setzero_ps();
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm512_store_ps(&(mat->element[0][i]), zero8);
		_mm512_store_ps(&(mat->element[1][i]), zero8);
		_mm512_store_ps(&(mat->element[2][i]), zero8);
		_mm512_store_ps(&(mat->element[3][i]), zero8);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntw(); long _N = mat->real_row_dim * mat->real_col_dim; long _ix;
		svfloat32_t _z = svdup_n_f32(0.0);
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b32_s32((int32_t)_ix, (int32_t)_N);
			svst1_f32(_pg, &(mat->element[0][_ix]), _z);
			svst1_f32(_pg, &(mat->element[1][_ix]), _z);
			svst1_f32(_pg, &(mat->element[2][_ix]), _z);
			svst1_f32(_pg, &(mat->element[3][_ix]), _z);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	float32x4_t zero_neon;

	zero_neon = vdupq_n_f32(0.0);
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		vst1q_f32(&(mat->element[0][i]), zero_neon);
		vst1q_f32(&(mat->element[1][i]), zero_neon);
		vst1q_f32(&(mat->element[2][i]), zero_neon);
		vst1q_f32(&(mat->element[3][i]), zero_neon);
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
void _bncomp_mul_qsmatrix_qsvec(QSVector v, QSMatrix a, QSVector vb)
{
	long int i, j, row_dim;
	int thread_index;
	float tmp[BNCOMP_MAX_NUM_THREADS][QSSIZE], tmp1[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	/* Check Dimension */
	//if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	if((v->dim < a->row_dim) || (vb->dim < a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_qsmatrix_qsvec\n");
		return;
	}

	row_dim = a->row_dim;

// SIMD
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int ij_index, real_row_dim, real_col_dim;
	__m256 tmp4[BNCOMP_MAX_NUM_THREADS][QSSIZE], tmp1_4[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	__m256 aij4[BNCOMP_MAX_NUM_THREADS][QSSIZE], vbj4[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j, ij_index)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		//rqs_set_ui(tmp, 0UL);
		_bncavx2_set0_qs(tmp4[thread_index]);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij4[thread_index][0] = _mm256_load_ps(&(a->element[0][ij_index]));
			aij4[thread_index][1] = _mm256_load_ps(&(a->element[1][ij_index]));
			aij4[thread_index][2] = _mm256_load_ps(&(a->element[2][ij_index]));
			aij4[thread_index][3] = _mm256_load_ps(&(a->element[3][ij_index]));
			vbj4[thread_index][0] = _mm256_load_ps(&(vb->element[0][j]));
			vbj4[thread_index][1] = _mm256_load_ps(&(vb->element[1][j]));
			vbj4[thread_index][2] = _mm256_load_ps(&(vb->element[2][j]));
			vbj4[thread_index][3] = _mm256_load_ps(&(vb->element[3][j]));

			//rqs_mul(tmp1, get_qsmatrix_ij(a, i, j), get_qsvector_i(vb, j));
			//rqs_add(tmp, tmp, tmp1);
			_bncavx2_rqs_mul(tmp1_4[thread_index], aij4[thread_index], vbj4[thread_index]);
			_bncavx2_rqs_add(tmp4[thread_index], tmp4[thread_index], tmp1_4[thread_index]);
		}
		//set_qsvector_i(v, i, tmp);
		_bncavx2_rqs_sum256(tmp[thread_index], tmp4[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
		v->element[2][i] = tmp[thread_index][2];
		v->element[3][i] = tmp[thread_index][3];
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int ij_index, real_col_dim, real_row_dim;
	__m512 tmp8[BNCOMP_MAX_NUM_THREADS][QSSIZE], tmp1_8[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	__m512 aij8[BNCOMP_MAX_NUM_THREADS][QSSIZE], vbj8[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j, ij_index)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		//rqs_set_ui(tmp, 0UL);
		_bncavx512_set0_qs(tmp8[thread_index]);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij8[thread_index][0] = _mm512_load_ps(&(a->element[0][ij_index]));
			aij8[thread_index][1] = _mm512_load_ps(&(a->element[1][ij_index]));
			aij8[thread_index][2] = _mm512_load_ps(&(a->element[2][ij_index]));
			aij8[thread_index][3] = _mm512_load_ps(&(a->element[3][ij_index]));
			vbj8[thread_index][0] = _mm512_load_ps(&(vb->element[0][j]));
			vbj8[thread_index][1] = _mm512_load_ps(&(vb->element[1][j]));
			vbj8[thread_index][2] = _mm512_load_ps(&(vb->element[2][j]));
			vbj8[thread_index][3] = _mm512_load_ps(&(vb->element[3][j]));

			//rqs_mul(tmp1, get_tsmatrix_ij(a, i, j), get_tsvector_i(vb, j));
			//rqs_add(tmp, tmp, tmp1);
			_bncavx512_rqs_mul(tmp1_8[thread_index], aij8[thread_index], vbj8[thread_index]);
			_bncavx512_rqs_add(tmp8[thread_index], tmp8[thread_index], tmp1_8[thread_index]);
		}
		//set_dsvector_i(v, i, tmp);
		_bncavx512_rqs_sum512(tmp[thread_index], tmp8[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
		v->element[2][i] = tmp[thread_index][2];
		v->element[3][i] = tmp[thread_index][3];
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic) + OpenMP
	/* SVE2 dot-product matvec for QS: 4-limb accumulator vectors. */
	{
		long sve_real_col_dim = a->real_col_dim;
		long sve_vl = (long)svcntw();

		#pragma omp parallel for private(thread_index, i, j) firstprivate(sve_vl, sve_real_col_dim)
		for(i = 0; i < row_dim; i++)
		{
			thread_index = omp_get_thread_num();
			(void)thread_index;
			svfloat32_t acc0, acc1, acc2, acc3;
			_bncsve2_rqs_set0(&acc0, &acc1, &acc2, &acc3);

			for(j = 0; j < sve_real_col_dim; j += sve_vl)
			{
				svbool_t pg = svwhilelt_b32_s32((int32_t)j, (int32_t)sve_real_col_dim);
				svfloat32_t aij0 = svld1_f32(pg, &(a->element[0][i * sve_real_col_dim + j]));
				svfloat32_t aij1 = svld1_f32(pg, &(a->element[1][i * sve_real_col_dim + j]));
				svfloat32_t aij2_v = svld1_f32(pg, &(a->element[2][i * sve_real_col_dim + j]));
				svfloat32_t aij3 = svld1_f32(pg, &(a->element[3][i * sve_real_col_dim + j]));
				svfloat32_t bj0  = svld1_f32(pg, &(vb->element[0][j]));
				svfloat32_t bj1  = svld1_f32(pg, &(vb->element[1][j]));
				svfloat32_t bj2  = svld1_f32(pg, &(vb->element[2][j]));
				svfloat32_t bj3  = svld1_f32(pg, &(vb->element[3][j]));
				svfloat32_t prod0, prod1, prod2, prod3;
				_bncsve2_rqs_mul(pg, &prod0, &prod1, &prod2, &prod3,
				                 aij0, aij1, aij2_v, aij3, bj0, bj1, bj2, bj3);
				_bncsve2_rqs_add(pg, &acc0, &acc1, &acc2, &acc3,
				                 acc0, acc1, acc2, acc3, prod0, prod1, prod2, prod3);
			}

			/* Horizontal sum across VL lanes */
			{
				float acc0_arr[sve_vl], acc1_arr[sve_vl], acc2_arr[sve_vl], acc3_arr[sve_vl];
				svst1_f32(svptrue_b32(), acc0_arr, acc0);
				svst1_f32(svptrue_b32(), acc1_arr, acc1);
				svst1_f32(svptrue_b32(), acc2_arr, acc2);
				svst1_f32(svptrue_b32(), acc3_arr, acc3);
				float sum_qs[QSSIZE] = { 0.0f, 0.0f, 0.0f, 0.0f };
				long lane;
				for(lane = 0; lane < sve_vl; lane++)
				{
					float lane_qs[QSSIZE] = { acc0_arr[lane], acc1_arr[lane], acc2_arr[lane], acc3_arr[lane] };
					rqs_add(sum_qs, sum_qs, lane_qs);
				}
				v->element[0][i] = sum_qs[0];
				v->element[1][i] = sum_qs[1];
				v->element[2][i] = sum_qs[2];
				v->element[3][i] = sum_qs[3];
			}
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	long int ij_index, real_row_dim, real_col_dim;
	float32x4_t tmp_neon[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	float32x4_t tmp1_neon[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	float32x4_t aij_neon[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	float32x4_t vbj_neon[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j, ij_index)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		// ゼロ初期化
		_bncneon_rqs_set0(tmp_neon[thread_index]);
		
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			
			// ロード
			aij_neon[thread_index][0] = vld1q_f32(&(a->element[0][ij_index]));
			aij_neon[thread_index][1] = vld1q_f32(&(a->element[1][ij_index]));
			aij_neon[thread_index][2] = vld1q_f32(&(a->element[2][ij_index]));
			aij_neon[thread_index][3] = vld1q_f32(&(a->element[3][ij_index]));
			vbj_neon[thread_index][0] = vld1q_f32(&(vb->element[0][j]));
			vbj_neon[thread_index][1] = vld1q_f32(&(vb->element[1][j]));
			vbj_neon[thread_index][2] = vld1q_f32(&(vb->element[2][j]));
			vbj_neon[thread_index][3] = vld1q_f32(&(vb->element[3][j]));

			// Neon QS乗算と累積加算
			_bncneon_rqs_mul(tmp1_neon[thread_index], aij_neon[thread_index], vbj_neon[thread_index]);
			_bncneon_rqs_add(tmp_neon[thread_index], tmp_neon[thread_index], tmp1_neon[thread_index]);
		}

		// float32x4_tの2要素を合計
		/* Inline 4-lane horizontal sum (was _bncneon_rqs_sum128f). */
		{
			float32x4_t *_s = tmp_neon[thread_index];
			float _l0[QSSIZE] = { vgetq_lane_f32(_s[0], 0), vgetq_lane_f32(_s[1], 0), vgetq_lane_f32(_s[2], 0), vgetq_lane_f32(_s[3], 0) };
			float _l1[QSSIZE] = { vgetq_lane_f32(_s[0], 1), vgetq_lane_f32(_s[1], 1), vgetq_lane_f32(_s[2], 1), vgetq_lane_f32(_s[3], 1) };
			float _l2[QSSIZE] = { vgetq_lane_f32(_s[0], 2), vgetq_lane_f32(_s[1], 2), vgetq_lane_f32(_s[2], 2), vgetq_lane_f32(_s[3], 2) };
			float _l3[QSSIZE] = { vgetq_lane_f32(_s[0], 3), vgetq_lane_f32(_s[1], 3), vgetq_lane_f32(_s[2], 3), vgetq_lane_f32(_s[3], 3) };
			rqs_set(tmp[thread_index], _l0);
			rqs_add(tmp[thread_index], tmp[thread_index], _l1);
			rqs_add(tmp[thread_index], tmp[thread_index], _l2);
			rqs_add(tmp[thread_index], tmp[thread_index], _l3);
		}

		// 結果を格納
		//set_qsvector_i(v, i, tmp[thread_index]);
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

		rqs_set_ui(tmp[thread_index], 0UL);
		for(j = 0; j < a->col_dim; j++)
		{
			rqs_mul(tmp1[thread_index], get_qsmatrix_ij(a, i, j), get_qsvector_i(vb, j));
			rqs_add(tmp[thread_index], tmp[thread_index], tmp1[thread_index]);
		}
		set_qsvector_i(v, i, tmp[thread_index]);
	}
#endif // __AVX2__
}

/* v := a^T * vb */
void _bncomp_mul_qsmatrixt_qsvec(QSVector v, QSMatrix a, QSVector vb)
{
	long int i, j, col_dim;
	int thread_index;
	float tmp[BNCOMP_MAX_NUM_THREADS][QSSIZE], tmp1[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	/* Check Dimension */
	//if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	if((v->dim < a->col_dim) || (vb->dim < a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_qsmatrixt_qsvec\n");
		return;
	}

	col_dim = a->col_dim;

// SIMD
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int real_row_dim, real_col_dim;
	__m256 tmp4[BNCOMP_MAX_NUM_THREADS][QSSIZE], tmp1_4[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	__m256 aij4[BNCOMP_MAX_NUM_THREADS][QSSIZE], vbj4[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < col_dim; i++)
	{
		thread_index = omp_get_thread_num();

		//rqs_set_ui(tmp, 0UL);
		_bncavx2_set0_qs(tmp4[thread_index]);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_row_dim; j += _BNC_S_WIDTH)
		{
			//aij4[thread_index][0] = _mm256_load_ps(&(a->element[0][ij_index]));
			//aij4[thread_index][1] = _mm256_load_ps(&(a->element[1][ij_index]));
			aij4[thread_index][0] = _mm256_set_ps(
				a->element[0][(j + 7) * real_col_dim + i],
				a->element[0][(j + 6) * real_col_dim + i],
				a->element[0][(j + 5) * real_col_dim + i],
				a->element[0][(j + 4) * real_col_dim + i],
				a->element[0][(j + 3) * real_col_dim + i],
				a->element[0][(j + 2) * real_col_dim + i],
				a->element[0][(j + 1) * real_col_dim + i],
				a->element[0][(j    ) * real_col_dim + i]
			);
			aij4[thread_index][1] = _mm256_set_ps(
				a->element[1][(j + 7) * real_col_dim + i],
				a->element[1][(j + 6) * real_col_dim + i],
				a->element[1][(j + 5) * real_col_dim + i],
				a->element[1][(j + 4) * real_col_dim + i],
				a->element[1][(j + 3) * real_col_dim + i],
				a->element[1][(j + 2) * real_col_dim + i],
				a->element[1][(j + 1) * real_col_dim + i],
				a->element[1][(j    ) * real_col_dim + i]
			);
			aij4[thread_index][2] = _mm256_set_ps(
				a->element[2][(j + 7) * real_col_dim + i],
				a->element[2][(j + 6) * real_col_dim + i],
				a->element[2][(j + 5) * real_col_dim + i],
				a->element[2][(j + 4) * real_col_dim + i],
				a->element[2][(j + 3) * real_col_dim + i],
				a->element[2][(j + 2) * real_col_dim + i],
				a->element[2][(j + 1) * real_col_dim + i],
				a->element[2][(j    ) * real_col_dim + i]
			);
			aij4[thread_index][3] = _mm256_set_ps(
				a->element[3][(j + 7) * real_col_dim + i],
				a->element[3][(j + 6) * real_col_dim + i],
				a->element[3][(j + 5) * real_col_dim + i],
				a->element[3][(j + 4) * real_col_dim + i],
				a->element[3][(j + 3) * real_col_dim + i],
				a->element[3][(j + 2) * real_col_dim + i],
				a->element[3][(j + 1) * real_col_dim + i],
				a->element[3][(j    ) * real_col_dim + i]
			);
			vbj4[thread_index][0] = _mm256_load_ps(&(vb->element[0][j]));
			vbj4[thread_index][1] = _mm256_load_ps(&(vb->element[1][j]));
			vbj4[thread_index][2] = _mm256_load_ps(&(vb->element[2][j]));
			vbj4[thread_index][3] = _mm256_load_ps(&(vb->element[3][j]));

			//rqs_mul(tmp1, get_qsmatrix_ij(a, i, j), get_qsvector_i(vb, j));
			//rqs_add(tmp, tmp, tmp1);
			_bncavx2_rqs_mul(tmp1_4[thread_index], aij4[thread_index], vbj4[thread_index]);
			_bncavx2_rqs_add(tmp4[thread_index], tmp4[thread_index], tmp1_4[thread_index]);
		}
		//set_tsvector_i(v, i, tmp);
		_bncavx2_rqs_sum256(tmp[thread_index], tmp4[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
		v->element[2][i] = tmp[thread_index][2];
		v->element[3][i] = tmp[thread_index][3];
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int real_row_dim, real_col_dim;
	__m512 tmp8[BNCOMP_MAX_NUM_THREADS][QSSIZE], tmp1_8[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	__m512 aij8[BNCOMP_MAX_NUM_THREADS][QSSIZE], vbj8[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < col_dim; i++)
	{
		thread_index = omp_get_thread_num();

		//rqs_set_ui(tmp, 0UL);
		_bncavx512_set0_qs(tmp8[thread_index]);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_row_dim; j += _BNC_S_WIDTH)
		{
			//aij8[thread_index][0] = _mm512_load_ps(&(a->element[0][ij_index]));
			//aij8[thread_index][1] = _mm512_load_ps(&(a->element[1][ij_index]));
			aij8[thread_index][0] = _mm512_set_ps(
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
				a->element[0][(j    ) * real_col_dim + i]
			);
			aij8[thread_index][1] = _mm512_set_ps(
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
				a->element[1][(j    ) * real_col_dim + i]
			);
			aij8[thread_index][2] = _mm512_set_ps(
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
				a->element[2][(j    ) * real_col_dim + i]
			);
			aij8[thread_index][3] = _mm512_set_ps(
				a->element[3][(j + 15) * real_col_dim + i],
				a->element[3][(j + 14) * real_col_dim + i],
				a->element[3][(j + 13) * real_col_dim + i],
				a->element[3][(j + 12) * real_col_dim + i],
				a->element[3][(j + 11) * real_col_dim + i],
				a->element[3][(j + 10) * real_col_dim + i],
				a->element[3][(j + 9) * real_col_dim + i],
				a->element[3][(j + 8) * real_col_dim + i],
				a->element[3][(j + 7) * real_col_dim + i],
				a->element[3][(j + 6) * real_col_dim + i],
				a->element[3][(j + 5) * real_col_dim + i],
				a->element[3][(j + 4) * real_col_dim + i],
				a->element[3][(j + 3) * real_col_dim + i],
				a->element[3][(j + 2) * real_col_dim + i],
				a->element[3][(j + 1) * real_col_dim + i],
				a->element[3][(j    ) * real_col_dim + i]
			);
			vbj8[thread_index][0] = _mm512_load_ps(&(vb->element[0][j]));
			vbj8[thread_index][1] = _mm512_load_ps(&(vb->element[1][j]));
			vbj8[thread_index][2] = _mm512_load_ps(&(vb->element[2][j]));
			vbj8[thread_index][3] = _mm512_load_ps(&(vb->element[3][j]));

			//rqs_mul(tmp1, get_tsmatrix_ij(a, i, j), get_tsvector_i(vb, j));
			//rqs_add(tmp, tmp, tmp1);
			_bncavx512_rqs_mul(tmp1_8[thread_index], aij8[thread_index], vbj8[thread_index]);
			_bncavx512_rqs_add(tmp8[thread_index], tmp8[thread_index], tmp1_8[thread_index]);
		}
		//set_dsvector_i(v, i, tmp);
		_bncavx512_rqs_sum512(tmp[thread_index], tmp8[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
		v->element[2][i] = tmp[thread_index][2];
		v->element[3][i] = tmp[thread_index][3];
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic) + OpenMP
	/* SVE2 transpose-matvec for QS: outer-product, parallel over i tiles. */
	{
		long sve_real_row_dim = a->real_row_dim;
		long sve_real_col_dim = a->real_col_dim;
		long sve_vl = (long)svcntw();

		#pragma omp parallel for private(thread_index, i, j) firstprivate(sve_vl, sve_real_row_dim, sve_real_col_dim)
		for(i = 0; i < sve_real_col_dim; i += sve_vl)
		{
			thread_index = omp_get_thread_num();
			(void)thread_index;
			svbool_t pg = svwhilelt_b32_s32((int32_t)i, (int32_t)sve_real_col_dim);
			svfloat32_t acc0, acc1, acc2, acc3;
			_bncsve2_rqs_set0(&acc0, &acc1, &acc2, &acc3);

			for(j = 0; j < sve_real_row_dim; j++)
			{
				svfloat32_t aji0 = svld1_f32(pg, &(a->element[0][j * sve_real_col_dim + i]));
				svfloat32_t aji1 = svld1_f32(pg, &(a->element[1][j * sve_real_col_dim + i]));
				svfloat32_t aji2_v = svld1_f32(pg, &(a->element[2][j * sve_real_col_dim + i]));
				svfloat32_t aji3 = svld1_f32(pg, &(a->element[3][j * sve_real_col_dim + i]));
				svfloat32_t bj0  = svdup_n_f32(vb->element[0][j]);
				svfloat32_t bj1  = svdup_n_f32(vb->element[1][j]);
				svfloat32_t bj2  = svdup_n_f32(vb->element[2][j]);
				svfloat32_t bj3  = svdup_n_f32(vb->element[3][j]);
				svfloat32_t prod0, prod1, prod2, prod3;
				_bncsve2_rqs_mul(pg, &prod0, &prod1, &prod2, &prod3,
				                 aji0, aji1, aji2_v, aji3, bj0, bj1, bj2, bj3);
				_bncsve2_rqs_add(pg, &acc0, &acc1, &acc2, &acc3,
				                 acc0, acc1, acc2, acc3, prod0, prod1, prod2, prod3);
			}

			svst1_f32(pg, &(v->element[0][i]), acc0);
			svst1_f32(pg, &(v->element[1][i]), acc1);
			svst1_f32(pg, &(v->element[2][i]), acc2);
			svst1_f32(pg, &(v->element[3][i]), acc3);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	// Arm Neon版
	long int real_row_dim, real_col_dim;
	float32x4_t tmp_neon[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	float32x4_t tmp1_neon[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	float32x4_t aij_neon[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	float32x4_t vbj_neon[BNCOMP_MAX_NUM_THREADS][QSSIZE];
	float aij_vals[BNCOMP_MAX_NUM_THREADS][QSSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	// 列ごとに並列処理（転置なので列方向にアクセス）
	#pragma omp parallel for private(thread_index, j) // , aij_vals)
	for(i = 0; i < col_dim; i++)
	{
		thread_index = omp_get_thread_num();

		// ゼロ初期化
		_bncneon_rqs_set0(tmp_neon[thread_index]);
		
		// 行方向にループ（転置なので A[j][i] を A^T[i][j] として扱う）
		for(j = 0; j < real_row_dim; j += _BNC_S_WIDTH)
		{
			// Gather elements for transpose operation
			aij_vals[thread_index][0] = a->element[0][j * real_col_dim + i];
			aij_vals[thread_index][1] = a->element[0][(j + 1) * real_col_dim + i];
			aij_neon[thread_index][0] = vld1q_f32(aij_vals[thread_index]);

			aij_vals[thread_index][0] = a->element[1][j * real_col_dim + i];
			aij_vals[thread_index][1] = a->element[1][(j + 1) * real_col_dim + i];
			aij_neon[thread_index][1] = vld1q_f32(aij_vals[thread_index]);

			aij_vals[thread_index][0] = a->element[2][j * real_col_dim + i];
			aij_vals[thread_index][1] = a->element[2][(j + 1) * real_col_dim + i];
			aij_neon[thread_index][2] = vld1q_f32(aij_vals[thread_index]);			

			aij_vals[thread_index][0] = a->element[3][j * real_col_dim + i];
			aij_vals[thread_index][1] = a->element[3][(j + 1) * real_col_dim + i];
			aij_neon[thread_index][3] = vld1q_f32(aij_vals[thread_index]);			
			
			// vb[j:j+1] をロード（連続要素）
			vbj_neon[thread_index][0] = vld1q_f32(&(vb->element[0][j]));
			vbj_neon[thread_index][1] = vld1q_f32(&(vb->element[1][j]));
			vbj_neon[thread_index][2] = vld1q_f32(&(vb->element[2][j]));
			vbj_neon[thread_index][3] = vld1q_f32(&(vb->element[3][j]));

			// Neon QS乗算と累積加算
			_bncneon_rqs_mul(tmp1_neon[thread_index], aij_neon[thread_index], vbj_neon[thread_index]);
			_bncneon_rqs_add(tmp_neon[thread_index], tmp_neon[thread_index], tmp1_neon[thread_index]);
		}

		// float32x4_tの2要素を合計
		/* Inline 4-lane horizontal sum (was _bncneon_rqs_sum128f). */
		{
			float32x4_t *_s = tmp_neon[thread_index];
			float _l0[QSSIZE] = { vgetq_lane_f32(_s[0], 0), vgetq_lane_f32(_s[1], 0), vgetq_lane_f32(_s[2], 0), vgetq_lane_f32(_s[3], 0) };
			float _l1[QSSIZE] = { vgetq_lane_f32(_s[0], 1), vgetq_lane_f32(_s[1], 1), vgetq_lane_f32(_s[2], 1), vgetq_lane_f32(_s[3], 1) };
			float _l2[QSSIZE] = { vgetq_lane_f32(_s[0], 2), vgetq_lane_f32(_s[1], 2), vgetq_lane_f32(_s[2], 2), vgetq_lane_f32(_s[3], 2) };
			float _l3[QSSIZE] = { vgetq_lane_f32(_s[0], 3), vgetq_lane_f32(_s[1], 3), vgetq_lane_f32(_s[2], 3), vgetq_lane_f32(_s[3], 3) };
			rqs_set(tmp[thread_index], _l0);
			rqs_add(tmp[thread_index], tmp[thread_index], _l1);
			rqs_add(tmp[thread_index], tmp[thread_index], _l2);
			rqs_add(tmp[thread_index], tmp[thread_index], _l3);
		}

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

		set0_qs(tmp[thread_index]);
		for(j = 0; j < a->row_dim; j++)
		{
			rqs_mul(tmp1[thread_index], get_qsmatrix_ij(a, j, i), get_qsvector_i(vb, j));
			rqs_add(tmp[thread_index], tmp[thread_index], tmp1[thread_index]);
		}
		set_qsvector_i(v, i, tmp[thread_index]);
	}
#endif // __AVX2__
}

#if 0  /* Ozaki-scheme parallel matmul disabled for the float (ds/ts/qs) layer:
        * requires split_qsmatrix_dmat / split_qsmatrix_t_dmat / add_qsmatrix_dmat,
        * which live in qs_oz_scheme.c (not implemented). */
// Matrix multiplication based on Ozaki scheme
void _bncomp_mul_qsmatrix_oz(QSMatrix ret, QSMatrix a, int max_num_div_a, QSMatrix b, int max_num_div_b) //, int num_digits)
{
    int i, j;
    int real_num_div_a, real_num_div_b;
    long int row_dim = ret->row_dim, col_dim = ret->col_dim, mid_dim = a->col_dim;
    DMatrix *div_a, *div_b, *div_ret;//div_ret[BNCOMP_MAX_NUM_THREADS];
	int thread_index, thread_num;

    if(mid_dim != b->row_dim)
    {
        fprintf(stderr, "ERROR: mul_qsmatrix_oz mid_dim(a, b) = (%ld, %ld)!\n", mid_dim, b->row_dim);
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
		real_num_div_a = split_qsmatrix_dmat(div_a, max_num_div_a, a);
		//printf("split_qsmatrix_dmat(%d, %d)  ->%d\n", div_a[0]->real_row_dim, div_a[0]->real_col_dim, real_num_div_a);

		#pragma omp section
		real_num_div_b = split_qsmatrix_t_dmat(div_b, max_num_div_b, b);
		//printf("split_qsmatrix_t_dmat(%d, %d)->%d\n", div_b[0]->real_row_dim, div_b[0]->real_col_dim, real_num_div_b);
	}

    set0_qsmatrix(ret);
	#pragma omp parallel for private(i, j) //collapse(2) private(thread_index, j) // , div_ret)
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
            //add_qsmatrix_dmat(ret, ret, div_ret[thread_index]);	 
            //add_qsmatrix_dmat(ret, ret, div_ret[j]);
			#pragma omp critical
				add_qsmatrix_dmat(ret, ret, div_ret[i]);
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
#endif  /* end: _bncomp_mul_qsmatrix_oz (Ozaki scheme) disabled */

#if 0

// Fit dimension to be multiple of min_dim
void _bncomp_mul_cqsmatrix_oz_3m(CQSMatrix ret, CQSMatrix a, int max_num_div_a_real, int max_num_div_a_image, CQSMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    QSMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    t1 = init_qsmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qsmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qsmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_qsmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_qsmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    _bncomp_mul_qsmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_a_image);
    _bncomp_sub_qsmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        mul_qsmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        mul_qsmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_a_image);
        add_qsmatrix(ret->im, t3, t4);
    #else // USE_4M
    */
        // 3M
        _bncomp_add_qsmatrix(t3, a->re, a->im);
        _bncomp_add_qsmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        _bncomp_mul_qsmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        _bncomp_sub_qsmatrix(ret->im, ret->im, t1);
        _bncomp_sub_qsmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_qsmatrix(t1);
    free_qsmatrix(t2);
    free_qsmatrix(t3);
    free_qsmatrix(t4);
}

// Fit dimension to be multiple of min_dim
void _bncomp_mul_cqsmatrix_oz_4m(CQSMatrix ret, CQSMatrix a, int max_num_div_a_real, int max_num_div_a_image, CQSMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    QSMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    t1 = init_qsmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qsmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qsmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_qsmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_qsmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    _bncomp_mul_qsmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_a_image);
    _bncomp_sub_qsmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        _bncomp_mul_qsmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        _bncomp_mul_qsmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_a_image);
        _bncomp_add_qsmatrix(ret->im, t3, t4);
    /*
    #else // USE_4M 
        // 3M
        add_qsmatrix(t3, a->re, a->im);
        add_qsmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        mul_qsmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        sub_qsmatrix(ret->im, ret->im, t1);
        sub_qsmatrix(ret->im, ret->im, t2);
    #endif // USE_4M
    */

    free_qsmatrix(t1);
    free_qsmatrix(t2);
    free_qsmatrix(t3);
    free_qsmatrix(t4);
}

// Simple triple-loop-way matrix multiplication (3M)
void _bncomp_mul_cqsmatrix_3m(CQSMatrix ret, CQSMatrix a, CQSMatrix b)
{
    QSMatrix t1, t2, t3, t4;
 
    t1 = init_qsmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qsmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qsmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_qsmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_qsmatrix(t1, a->re, b->re);
    _bncomp_mul_qsmatrix(t2, a->im, b->im);
    _bncomp_sub_qsmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        _bncomp_mul_qsmatrix(t3, a->im, b->re);
        _bncomp_mul_qsmatrix(t4, a->re, b->im);
        _bncomp_add_qsmatrix(ret->im, t3, t4);
    #else // USE_4M
    */
        // 3M
        _bncomp_add_qsmatrix(t3, a->re, a->im);
        _bncomp_add_qsmatrix(t4, b->re, b->im);
        _bncomp_mul_qsmatrix(ret->im, t3, t4);
        _bncomp_sub_qsmatrix(ret->im, ret->im, t1);
        _bncomp_sub_qsmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_qsmatrix(t1);
    free_qsmatrix(t2);
    free_qsmatrix(t3);
    free_qsmatrix(t4);
}

// Simple triple-loop-way matrix multiplication (4M)
void _bncomp_mul_cqsmatrix_4m(CQSMatrix ret, CQSMatrix a, CQSMatrix b)
{
    QSMatrix t1, t2, t3, t4;
 
    t1 = init_qsmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_qsmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_qsmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_qsmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_qsmatrix(t1, a->re, b->re);
    _bncomp_mul_qsmatrix(t2, a->im, b->im);
    _bncomp_sub_qsmatrix(ret->re, t1, t2);

    // 4M
    
    //#ifdef USE_4M
        _bncomp_mul_qsmatrix(t3, a->im, b->re);
        _bncomp_mul_qsmatrix(t4, a->re, b->im);
        _bncomp_add_qsmatrix(ret->im, t3, t4);
    //#else // USE_4M
	/*
        // 3M
        _bncomp_add_qsmatrix(t3, a->re, a->im);
        _bncomp_add_qsmatrix(t4, b->re, b->im);
        _bncomp_mul_qsmatrix(ret->im, t3, t4s);
        _bncomp_sub_qsmatrix(ret->im, ret->im, t1);
        _bncomp_sub_qsmatrix(ret->im, ret->im, t2);
	*/
    //#endif // USE_4M

    free_qsmatrix(t1);
    free_qsmatrix(t2);
    free_qsmatrix(t3);
    free_qsmatrix(t4);
}
#endif // 0

