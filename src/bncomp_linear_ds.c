/********************************************************************************/
/* bncomp_linear_ds.c: Parallelized DS Precision Linear Computation Library     */
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

// bncomp_linear_ds.c with ARM NEON support
// Ads this include at the top of the file
// #if defined(__ARM_NEON)
/* Force-include NEON helpers used by the OMP+NEON / OMP+SVE2 (NEON fallback)
 * branches.  bncomp.h transitively pulls neon/bncneon.h via dslinear.h, but the
 * include chain depends on __ARM_SVE2 / __ARM_NEON macros — when the
 * outlined _omp_fn.0 references reduction helpers (sum128f, ...), the static
 * definition MUST be visible.  Including directly here guarantees that. */
#if (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON)
#include "neon/_bncneon_ds.h"
#endif
// #endif

//---------------------------------------
// Double-Double - Vector operations with NEON
//---------------------------------------

/* c := a */
void _bncomp_subst_dsvector(DSVector c, DSVector a)
{
	long int i;

#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntw(); long _N = a->real_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b32_s32((int32_t)_ix, (int32_t)_N);
			svst1_f32(_pg, &(c->element[0][_ix]), svld1_f32(_pg, &(a->element[0][_ix])));
			svst1_f32(_pg, &(c->element[1][_ix]), svld1_f32(_pg, &(a->element[1][_ix])));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON)
	#pragma omp parallel for
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		vst1q_f32(&(c->element[0][i]), vld1q_f32(&(a->element[0][i])));
		vst1q_f32(&(c->element[1][i]), vld1q_f32(&(a->element[1][i])));
	}
#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	#pragma omp parallel for
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		_mm256_store_ps(&(c->element[0][i]), _mm256_load_ps(&(a->element[0][i])));
		_mm256_store_ps(&(c->element[1][i]), _mm256_load_ps(&(a->element[1][i])));
	}
#elif defined(__AVX512F__) // __AVX512F__
	#pragma omp parallel for
	for(i = 0; i < a->real_dim; i += _BNC_S_WIDTH)
	{
		_mm512_store_ps(&(c->element[0][i]), _mm512_load_ps(&(a->element[0][i])));
		_mm512_store_ps(&(c->element[1][i]), _mm512_load_ps(&(a->element[1][i])));
	}
#else // others
	#pragma omp parallel for
	for(i = 0; i < a->dim; i++)
		set_dsvector_i(c, i, get_dsvector_i(a, i));
#endif
}

/* c = a + b */
void _bncomp_add_dsvector(DSVector c, DSVector a, DSVector b)
{
	int thread_index;
	long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_dsvector\n");
		return;
	}

#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntw(); long _N = c->real_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b32_s32((int32_t)_ix, (int32_t)_N);
			svfloat32_t _a0 = svld1_f32(_pg, &(a->element[0][_ix]));
			svfloat32_t _a1 = svld1_f32(_pg, &(a->element[1][_ix]));
			svfloat32_t _b0 = svld1_f32(_pg, &(b->element[0][_ix]));
			svfloat32_t _b1 = svld1_f32(_pg, &(b->element[1][_ix]));
			svfloat32_t _o0, _o1;
			_bncsve2_rds_add(_pg, &_o0, &_o1, _a0, _a1, _b0, _b1);
			svst1_f32(_pg, &(c->element[0][_ix]), _o0);
			svst1_f32(_pg, &(c->element[1][_ix]), _o1);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON)
    float32x4_t in_ret[BNCOMP_MAX_NUM_THREADS][DSSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][DSSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][DSSIZE];

 	#pragma omp parallel for private(thread_index)
    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = vld1q_f32(&(a->element[0][index]));
        in_a_val[thread_index][1] = vld1q_f32(&(a->element[1][index]));
        in_b_val[thread_index][0] = vld1q_f32(&(b->element[0][index]));
        in_b_val[thread_index][1] = vld1q_f32(&(b->element[1][index]));

        _bncneon_rds_add(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        vst1q_f32(&(c->element[0][index]), in_ret[thread_index][0]);
        vst1q_f32(&(c->element[1][index]), in_ret[thread_index][1]);
    }
#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256 in_ret[BNCOMP_MAX_NUM_THREADS][DSSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][DSSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][DSSIZE];

 	#pragma omp parallel for private(thread_index)
   for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[thread_index][1] = _mm256_load_ps(&(a->element[1][index]));
        in_b_val[thread_index][0] = _mm256_load_ps(&(b->element[0][index]));
        in_b_val[thread_index][1] = _mm256_load_ps(&(b->element[1][index]));

        _bncavx2_rds_add(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        _mm256_store_ps(&(c->element[0][index]), in_ret[thread_index][0]);
        _mm256_store_ps(&(c->element[1][index]), in_ret[thread_index][1]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512 in_ret[BNCOMP_MAX_NUM_THREADS][DSSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][DSSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	#pragma omp parallel for private(thread_index)
    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = _mm512_load_ps(&(a->element[0][index]));
        in_a_val[thread_index][1] = _mm512_load_ps(&(a->element[1][index]));
        in_b_val[thread_index][0] = _mm512_load_ps(&(b->element[0][index]));
        in_b_val[thread_index][1] = _mm512_load_ps(&(b->element[1][index]));

        _bncavx512_rds_add(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        _mm512_store_ps(&(c->element[0][index]), in_ret[thread_index][0]);
        _mm512_store_ps(&(c->element[1][index]), in_ret[thread_index][1]);
   }
#else // others
	float tmp[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

		rds_add(tmp[thread_index], get_dsvector_i(a, i), get_dsvector_i(b, i));
		set_dsvector_i(c, i, tmp[thread_index]);
	}
#endif
}

/* c = a - b */
void _bncomp_sub_dsvector(DSVector c, DSVector a, DSVector b)
{
	int thread_index;
	long int i, index;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: _bncomp_sub_dsvector\n");
		return;
	}

#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntw(); long _N = c->real_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b32_s32((int32_t)_ix, (int32_t)_N);
			svfloat32_t _a0 = svld1_f32(_pg, &(a->element[0][_ix]));
			svfloat32_t _a1 = svld1_f32(_pg, &(a->element[1][_ix]));
			svfloat32_t _b0 = svld1_f32(_pg, &(b->element[0][_ix]));
			svfloat32_t _b1 = svld1_f32(_pg, &(b->element[1][_ix]));
			_b0 = svneg_f32_x(_pg, _b0);
			_b1 = svneg_f32_x(_pg, _b1);
			svfloat32_t _o0, _o1;
			_bncsve2_rds_add(_pg, &_o0, &_o1, _a0, _a1, _b0, _b1);
			svst1_f32(_pg, &(c->element[0][_ix]), _o0);
			svst1_f32(_pg, &(c->element[1][_ix]), _o1);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON)
    float32x4_t in_ret[BNCOMP_MAX_NUM_THREADS][DSSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][DSSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	#pragma omp parallel for private(thread_index)
    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = vld1q_f32(&(a->element[0][index]));
        in_a_val[thread_index][1] = vld1q_f32(&(a->element[1][index]));
        in_b_val[thread_index][0] = vld1q_f32(&(b->element[0][index]));
        in_b_val[thread_index][1] = vld1q_f32(&(b->element[1][index]));

        _bncneon_rds_sub(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        vst1q_f32(&(c->element[0][index]), in_ret[thread_index][0]);
        vst1q_f32(&(c->element[1][index]), in_ret[thread_index][1]);
   }
#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
    __m256 in_ret[BNCOMP_MAX_NUM_THREADS][DSSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][DSSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	#pragma omp parallel for private(thread_index)
    for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = _mm256_load_ps(&(a->element[0][index]));
        in_a_val[thread_index][1] = _mm256_load_ps(&(a->element[1][index]));
        in_b_val[thread_index][0] = _mm256_load_ps(&(b->element[0][index]));
        in_b_val[thread_index][1] = _mm256_load_ps(&(b->element[1][index]));

        _bncavx2_rds_sub(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        _mm256_store_ps(&(c->element[0][index]), in_ret[thread_index][0]);
        _mm256_store_ps(&(c->element[1][index]), in_ret[thread_index][1]);
   }
#elif defined(__AVX512F__) // __AVX512F__
    __m512 in_ret[BNCOMP_MAX_NUM_THREADS][DSSIZE], in_a_val[BNCOMP_MAX_NUM_THREADS][DSSIZE], in_b_val[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
    {
		thread_index = omp_get_thread_num();

        in_a_val[thread_index][0] = _mm512_load_ps(&(a->element[0][index]));
        in_a_val[thread_index][1] = _mm512_load_ps(&(a->element[1][index]));
        in_b_val[thread_index][0] = _mm512_load_ps(&(b->element[0][index]));
        in_b_val[thread_index][1] = _mm512_load_ps(&(b->element[1][index]));

        _bncavx512_rds_sub(in_ret[thread_index], in_a_val[thread_index], in_b_val[thread_index]);

        _mm512_store_ps(&(c->element[0][index]), in_ret[thread_index][0]);
        _mm512_store_ps(&(c->element[1][index]), in_ret[thread_index][1]);
   }
#else // others
	float tmp[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

		rds_sub(tmp[thread_index], get_dsvector_i(a, i), get_dsvector_i(b, i));
		set_dsvector_i(c, i, tmp[thread_index]);
	}
#endif
}

/* c = val * a */
void _bncomp_cmul_dsvector(DSVector c, float val[DSSIZE], DSVector a)
{
	int thread_index;
	long int i, index;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: _bncomp_cmul_dsvector\n");
		return;
	}

#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntw(); long _N = a->real_dim; long _ix;
		svfloat32_t _v0 = svdup_n_f32(val[0]);
		svfloat32_t _v1 = svdup_n_f32(val[1]);
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b32_s32((int32_t)_ix, (int32_t)_N);
			svfloat32_t _a0 = svld1_f32(_pg, &(a->element[0][_ix]));
			svfloat32_t _a1 = svld1_f32(_pg, &(a->element[1][_ix]));
			svfloat32_t _o0, _o1;
			_bncsve2_rds_mul(_pg, &_o0, &_o1, _v0, _v1, _a0, _a1);
			svst1_f32(_pg, &(c->element[0][_ix]), _o0);
			svst1_f32(_pg, &(c->element[1][_ix]), _o1);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a2[BNCOMP_MAX_NUM_THREADS][DSSIZE], c2[BNCOMP_MAX_NUM_THREADS][DSSIZE], val2[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		val2[thread_index][0] = vdupq_n_f32(val[0]);
		val2[thread_index][1] = vdupq_n_f32(val[1]);
	}
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a2[thread_index][0] = vld1q_f32(&(a->element[0][index]));
		a2[thread_index][1] = vld1q_f32(&(a->element[1][index]));

		_bncneon_rds_mul(c2[thread_index], val2[thread_index], a2[thread_index]);

		vst1q_f32(&(c->element[0][index]), c2[thread_index][0]);
		vst1q_f32(&(c->element[1][index]), c2[thread_index][1]);
	}
#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4[BNCOMP_MAX_NUM_THREADS][DSSIZE], c4[BNCOMP_MAX_NUM_THREADS][DSSIZE], val4[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		val4[thread_index][0] = _mm256_set1_ps(val[0]);
		val4[thread_index][1] = _mm256_set1_ps(val[1]);
	}
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a4[thread_index][0] = _mm256_load_ps(&(a->element[0][index]));
		a4[thread_index][1] = _mm256_load_ps(&(a->element[1][index]));

		_bncavx2_rds_mul(c4[thread_index], val4[thread_index], a4[thread_index]);

		_mm256_store_ps(&(c->element[0][index]), c4[thread_index][0]);
		_mm256_store_ps(&(c->element[1][index]), c4[thread_index][1]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 a8[BNCOMP_MAX_NUM_THREADS][DSSIZE], c8[BNCOMP_MAX_NUM_THREADS][DSSIZE], val8[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		val8[thread_index][0] = _mm512_set1_ps(val[0]);
		val8[thread_index][1] = _mm512_set1_ps(val[1]);
	}

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < c->real_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a8[thread_index][0] = _mm512_load_ps(&(a->element[0][index]));
		a8[thread_index][1] = _mm512_load_ps(&(a->element[1][index]));

		_bncavx512_rds_mul(c8[thread_index], val8[thread_index], a8[thread_index]);

		_mm512_store_ps(&(c->element[0][index]), c8[thread_index][0]);
		_mm512_store_ps(&(c->element[1][index]), c8[thread_index][1]);
	}
#else // others
	float tmp[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < c->dim; i++)
	{
		thread_index = omp_get_thread_num();

		rds_mul(tmp[thread_index], val, get_dsvector_i(a, i));
		set_dsvector_i(c, i, tmp[thread_index]);
	}
#endif
}

/* (a, b) - inner product */
void _bncomp_ip_dsvector(float ret[DSSIZE], DSVector a, DSVector b)
{
	int thread_index;
	long int i, index;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: _bncomp_ip_dsvector\n");
		return;
	}

#if defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntw(); long _N = a->real_dim; long _ix;
		svfloat32_t _acc0 = svdup_n_f32(0.0);
		svfloat32_t _acc1 = svdup_n_f32(0.0);
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b32_s32((int32_t)_ix, (int32_t)_N);
			svfloat32_t _a0 = svld1_f32(_pg, &(a->element[0][_ix]));
			svfloat32_t _a1 = svld1_f32(_pg, &(a->element[1][_ix]));
			svfloat32_t _b0 = svld1_f32(_pg, &(b->element[0][_ix]));
			svfloat32_t _b1 = svld1_f32(_pg, &(b->element[1][_ix]));
			svfloat32_t _m0, _m1;
			_bncsve2_rds_mul(_pg, &_m0, &_m1, _a0, _a1, _b0, _b1);
			_bncsve2_rds_add(_pg, &_acc0, &_acc1, _acc0, _acc1, _m0, _m1);
		}
		{
			long _L, _vl = (long)svcntw();
			float _la0[64], _la1[64];
			svst1_f32(svptrue_b32(), _la0, _acc0);
			svst1_f32(svptrue_b32(), _la1, _acc1);
			rds_set_ui(ret, 0UL);
			for(_L = 0; _L < _vl; _L++)
			{
				float _lane[DSSIZE] = { _la0[_L], _la1[_L] };
				rds_add(ret, ret, _lane);
			}
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	float32x4_t a2[BNCOMP_MAX_NUM_THREADS][DSSIZE], b2[BNCOMP_MAX_NUM_THREADS][DSSIZE];
	float32x4_t ret2[DSSIZE], tmp2[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	_bncneon_rds_set0(ret2);
	
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a2[thread_index][0] = vld1q_f32(&(a->element[0][index]));
		a2[thread_index][1] = vld1q_f32(&(a->element[1][index]));
		b2[thread_index][0] = vld1q_f32(&(b->element[0][index]));
		b2[thread_index][1] = vld1q_f32(&(b->element[1][index]));

		_bncneon_rds_mul(tmp2[thread_index], a2[thread_index], b2[thread_index]);
		
		#pragma omp critical
		_bncneon_rds_add(ret2, ret2, tmp2[thread_index]);
	}

	/* Inline 4-lane horizontal sum (was _bncneon_rds_sum128f).
	 * Inlined here because GCC's OpenMP outline pass + static-symbol
	 * pruning made the helper an undefined reference in the .o. */
	{
		float _l0[DSSIZE] = { vgetq_lane_f32(ret2[0], 0), vgetq_lane_f32(ret2[1], 0) };
		float _l1[DSSIZE] = { vgetq_lane_f32(ret2[0], 1), vgetq_lane_f32(ret2[1], 1) };
		float _l2[DSSIZE] = { vgetq_lane_f32(ret2[0], 2), vgetq_lane_f32(ret2[1], 2) };
		float _l3[DSSIZE] = { vgetq_lane_f32(ret2[0], 3), vgetq_lane_f32(ret2[1], 3) };
		rds_set(ret, _l0);
		rds_add(ret, ret, _l1);
		rds_add(ret, ret, _l2);
		rds_add(ret, ret, _l3);
	}

#elif defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a4[BNCOMP_MAX_NUM_THREADS][DSSIZE], b4[BNCOMP_MAX_NUM_THREADS][DSSIZE];
	__m256 ret4[DSSIZE], tmp4[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	_bncavx2_rds_set0(ret4);

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a4[thread_index][0] = _mm256_load_ps(&(a->element[0][index]));
		a4[thread_index][1] = _mm256_load_ps(&(a->element[1][index]));
		b4[thread_index][0] = _mm256_load_ps(&(b->element[0][index]));
		b4[thread_index][1] = _mm256_load_ps(&(b->element[1][index]));

		_bncavx2_rds_mul(tmp4[thread_index], a4[thread_index], b4[thread_index]);
		#pragma omp critical
			_bncavx2_rds_add(ret4, ret4, tmp4[thread_index]);
	}

	_bncavx2_rds_sum256(ret, ret4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512 a8[BNCOMP_MAX_NUM_THREADS][DSSIZE], b8[BNCOMP_MAX_NUM_THREADS][DSSIZE];
	__m512 ret8[DSSIZE], tmp8[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	_bncavx512_rds_set0(ret8);

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < a->real_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a8[thread_index][0] = _mm512_load_ps(&(a->element[0][index]));
		a8[thread_index][1] = _mm512_load_ps(&(a->element[1][index]));
		b8[thread_index][0] = _mm512_load_ps(&(b->element[0][index]));
		b8[thread_index][1] = _mm512_load_ps(&(b->element[1][index]));

		_bncavx512_rds_mul(tmp8[thread_index], a8[thread_index], b8[thread_index]);
		#pragma omp critical
			_bncavx512_rds_add(ret8, ret8, tmp8[thread_index]);
	}

	_bncavx512_rds_sum512(ret, ret8);

#else // others
	float tmp[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	rds_set_ui(ret, 0UL);

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < a->dim; i++)
	{
		thread_index = omp_get_thread_num();

		rds_mul(tmp[thread_index], get_dsvector_i(a, i), get_dsvector_i(b, i));
	#pragma omp critical
		rds_add(ret, ret, tmp[thread_index]);
	}
#endif

	return;
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void _bncomp_row_swap_dsmatrix(DSMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
	long int i, j, true_end;
	float tmp[BNCOMP_MAX_NUM_THREADS][DSSIZE];
	int thread_index;

	true_end = (col_end > mat->col_dim) ? mat->col_dim : col_end;

	#pragma omp parallel for private(thread_index)
	for(i = col_start; i < true_end; i++)
	{
		thread_index = omp_get_thread_num();

		rds_set(tmp[thread_index], get_dsmatrix_ij(mat, row_index0, i));
		set_dsmatrix_ij(mat, row_index0, i, get_dsmatrix_ij(mat, row_index1, i));
		set_dsmatrix_ij(mat, row_index1, i, tmp[thread_index]);
	}
}


//--------
// Matrix
//--------

// Neon用のヘッダーとマクロ定義を追加
#if defined(__ARM_NEON) && defined(BNC_ENABLE_NEON)
#include <arm_neon.h>
//#define _BNC_NEON_WIDTH 2  // Neon は float32x4_t (2つのfloat)
#endif

/* c := a + b */
void _bncomp_add_dsmatrix(DSMatrix c, DSMatrix a, DSMatrix b)
{
	int thread_index;
	long int index, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_dsmatrix\n");
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
	__m256 tmp4[BNCOMP_MAX_NUM_THREADS][DSSIZE], aij4[BNCOMP_MAX_NUM_THREADS][DSSIZE], bij4[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();

		aij4[thread_index][0] = _mm256_load_ps(&(a->element[0][index]));
		aij4[thread_index][1] = _mm256_load_ps(&(a->element[1][index]));
		bij4[thread_index][0] = _mm256_load_ps(&(b->element[0][index]));
		bij4[thread_index][1] = _mm256_load_ps(&(b->element[1][index]));

		_bncavx2_rds_add(tmp4[thread_index], aij4[thread_index], bij4[thread_index]);

		_mm256_store_ps(&(c->element[0][index]), tmp4[thread_index][0]);
		_mm256_store_ps(&(c->element[1][index]), tmp4[thread_index][1]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 tmp8[BNCOMP_MAX_NUM_THREADS][DSSIZE], aij8[BNCOMP_MAX_NUM_THREADS][DSSIZE], bij8[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();

		aij8[thread_index][0] = _mm512_load_ps(&(a->element[0][index]));
		aij8[thread_index][1] = _mm512_load_ps(&(a->element[1][index]));
		bij8[thread_index][0] = _mm512_load_ps(&(b->element[0][index]));
		bij8[thread_index][1] = _mm512_load_ps(&(b->element[1][index]));

		_bncavx512_rds_add(tmp8[thread_index], aij8[thread_index], bij8[thread_index]);

		_mm512_store_ps(&(c->element[0][index]), tmp8[thread_index][0]);
		_mm512_store_ps(&(c->element[1][index]), tmp8[thread_index][1]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntw(); long _N = c->real_row_dim * c->real_col_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b32_s32((int32_t)_ix, (int32_t)_N);
			svfloat32_t _a0 = svld1_f32(_pg, &(a->element[0][_ix]));
			svfloat32_t _a1 = svld1_f32(_pg, &(a->element[1][_ix]));
			svfloat32_t _b0 = svld1_f32(_pg, &(b->element[0][_ix]));
			svfloat32_t _b1 = svld1_f32(_pg, &(b->element[1][_ix]));
			svfloat32_t _o0, _o1;
			_bncsve2_rds_add(_pg, &_o0, &_o1, _a0, _a1, _b0, _b1);
			svst1_f32(_pg, &(c->element[0][_ix]), _o0);
			svst1_f32(_pg, &(c->element[1][_ix]), _o1);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	float32x4_t tmp2[BNCOMP_MAX_NUM_THREADS][DSSIZE], aij2[BNCOMP_MAX_NUM_THREADS][DSSIZE], bij2[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij2[thread_index][0] = vld1q_f32(&(a->element[0][index]));
		aij2[thread_index][1] = vld1q_f32(&(a->element[1][index]));
		bij2[thread_index][0] = vld1q_f32(&(b->element[0][index]));
		bij2[thread_index][1] = vld1q_f32(&(b->element[1][index]));

		_bncneon_rds_add(tmp2[thread_index], aij2[thread_index], bij2[thread_index]);

		vst1q_f32(&(c->element[0][index]), tmp2[thread_index][0]);
		vst1q_f32(&(c->element[1][index]), tmp2[thread_index][1]); 
	}
#else // others
	float tmp[BNCOMP_MAX_NUM_THREADS][DSSIZE], aij[BNCOMP_MAX_NUM_THREADS][DSSIZE], bij[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index++)
	{
		thread_index = omp_get_thread_num();
		
		aij[thread_index][0] = a->element[0][index];
		aij[thread_index][1] = a->element[1][index];
		bij[thread_index][0] = b->element[0][index];
		bij[thread_index][1] = b->element[1][index];

		rds_add(tmp[thread_index], aij[thread_index], bij[thread_index]);

		c->element[0][index] = tmp[thread_index][0];
		c->element[1][index] = tmp[thread_index][1]; 
	}
#endif // __AVX2__
}

/* c := a - b */
void _bncomp_sub_dsmatrix(DSMatrix c, DSMatrix a, DSMatrix b)
{
	int thread_index;
	long int index, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_dsmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_add_dsmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 tmp4[BNCOMP_MAX_NUM_THREADS][DSSIZE], aij4[BNCOMP_MAX_NUM_THREADS][DSSIZE], bij4[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();

		aij4[thread_index][0] = _mm256_load_ps(&(a->element[0][index]));
		aij4[thread_index][1] = _mm256_load_ps(&(a->element[1][index]));
		bij4[thread_index][0] = _mm256_load_ps(&(b->element[0][index]));
		bij4[thread_index][1] = _mm256_load_ps(&(b->element[1][index]));

		_bncavx2_rds_sub(tmp4[thread_index], aij4[thread_index], bij4[thread_index]);

		_mm256_store_ps(&(c->element[0][index]), tmp4[thread_index][0]);
		_mm256_store_ps(&(c->element[1][index]), tmp4[thread_index][1]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 tmp8[BNCOMP_MAX_NUM_THREADS][DSSIZE], aij8[BNCOMP_MAX_NUM_THREADS][DSSIZE], bij8[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();

		aij8[thread_index][0] = _mm512_load_ps(&(a->element[0][index]));
		aij8[thread_index][1] = _mm512_load_ps(&(a->element[1][index]));
		bij8[thread_index][0] = _mm512_load_ps(&(b->element[0][index]));
		bij8[thread_index][1] = _mm512_load_ps(&(b->element[1][index]));

		_bncavx512_rds_sub(tmp8[thread_index], aij8[thread_index], bij8[thread_index]);

		_mm512_store_ps(&(c->element[0][index]), tmp8[thread_index][0]);
		_mm512_store_ps(&(c->element[1][index]), tmp8[thread_index][1]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntw(); long _N = c->real_row_dim * c->real_col_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b32_s32((int32_t)_ix, (int32_t)_N);
			svfloat32_t _a0 = svld1_f32(_pg, &(a->element[0][_ix]));
			svfloat32_t _a1 = svld1_f32(_pg, &(a->element[1][_ix]));
			svfloat32_t _b0 = svld1_f32(_pg, &(b->element[0][_ix]));
			svfloat32_t _b1 = svld1_f32(_pg, &(b->element[1][_ix]));
			_b0 = svneg_f32_x(_pg, _b0);
			_b1 = svneg_f32_x(_pg, _b1);
			svfloat32_t _o0, _o1;
			_bncsve2_rds_add(_pg, &_o0, &_o1, _a0, _a1, _b0, _b1);
			svst1_f32(_pg, &(c->element[0][_ix]), _o0);
			svst1_f32(_pg, &(c->element[1][_ix]), _o1);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	float32x4_t tmp2[BNCOMP_MAX_NUM_THREADS][DSSIZE], aij2[BNCOMP_MAX_NUM_THREADS][DSSIZE], bij2[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij2[thread_index][0] = vld1q_f32(&(a->element[0][index]));
		aij2[thread_index][1] = vld1q_f32(&(a->element[1][index]));
		bij2[thread_index][0] = vld1q_f32(&(b->element[0][index]));
		bij2[thread_index][1] = vld1q_f32(&(b->element[1][index]));

		_bncneon_rds_sub(tmp2[thread_index], aij2[thread_index], bij2[thread_index]);

		vst1q_f32(&(c->element[0][index]), tmp2[thread_index][0]);
		vst1q_f32(&(c->element[1][index]), tmp2[thread_index][1]); 
	}
#else // others
	float tmp[BNCOMP_MAX_NUM_THREADS][DSSIZE], aij[BNCOMP_MAX_NUM_THREADS][DSSIZE], bij[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index++)
	{
		thread_index = omp_get_thread_num();
		
		aij[thread_index][0] = a->element[0][index];
		aij[thread_index][1] = a->element[1][index];
		bij[thread_index][0] = b->element[0][index];
		bij[thread_index][1] = b->element[1][index];

		rds_sub(tmp[thread_index], aij[thread_index], bij[thread_index]);

		c->element[0][index] = tmp[thread_index][0];
		c->element[1][index] = tmp[thread_index][1]; 
	}
#endif // __AVX2__
}

/* c := sc * a */
void _bncomp_cmul_dsmatrix(DSMatrix c, float sc[DSSIZE], DSMatrix a)
{
	long int i, j, index, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim;
	int thread_index;

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_dsmatrix\n");
		return;
	}
	row_dim = c->row_dim;
	real_row_dim = c->real_row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_dsmatrix\n");
		return;
	}
	col_dim = c->col_dim;
	real_col_dim = c->real_col_dim;

	real_total_dim = real_row_dim * real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 tmp4[BNCOMP_MAX_NUM_THREADS][DSSIZE], aij4[BNCOMP_MAX_NUM_THREADS][DSSIZE], sc4[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		sc4[thread_index][0] = _mm256_set1_ps(sc[0]);
		sc4[thread_index][1] = _mm256_set1_ps(sc[1]);
	}
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();

		aij4[thread_index][0] = _mm256_load_ps(&(a->element[0][index]));
		aij4[thread_index][1] = _mm256_load_ps(&(a->element[1][index]));

		_bncavx2_rds_mul(tmp4[thread_index], sc4[thread_index], aij4[thread_index]);

		_mm256_store_ps(&(c->element[0][index]), tmp4[thread_index][0]);
		_mm256_store_ps(&(c->element[1][index]), tmp4[thread_index][1]);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 tmp8[BNCOMP_MAX_NUM_THREADS][DSSIZE], aij8[BNCOMP_MAX_NUM_THREADS][DSSIZE], sc8[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		sc8[thread_index][0] = _mm512_set1_ps(sc[0]);
		sc8[thread_index][1] = _mm512_set1_ps(sc[1]);
	}
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();

		aij8[thread_index][0] = _mm512_load_ps(&(a->element[0][index]));
		aij8[thread_index][1] = _mm512_load_ps(&(a->element[1][index]));

		_bncavx512_rds_mul(tmp8[thread_index], sc8[thread_index], aij8[thread_index]);

		_mm512_store_ps(&(c->element[0][index]), tmp8[thread_index][0]);
		_mm512_store_ps(&(c->element[1][index]), tmp8[thread_index][1]);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntw(); long _N = a->real_row_dim * a->real_col_dim; long _ix;
		svfloat32_t _v0 = svdup_n_f32(sc[0]);
		svfloat32_t _v1 = svdup_n_f32(sc[1]);
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b32_s32((int32_t)_ix, (int32_t)_N);
			svfloat32_t _a0 = svld1_f32(_pg, &(a->element[0][_ix]));
			svfloat32_t _a1 = svld1_f32(_pg, &(a->element[1][_ix]));
			svfloat32_t _o0, _o1;
			_bncsve2_rds_mul(_pg, &_o0, &_o1, _v0, _v1, _a0, _a1);
			svst1_f32(_pg, &(c->element[0][_ix]), _o0);
			svst1_f32(_pg, &(c->element[1][_ix]), _o1);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	float32x4_t tmp2[BNCOMP_MAX_NUM_THREADS][DSSIZE], aij2[BNCOMP_MAX_NUM_THREADS][DSSIZE], sc2[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	#pragma omp parallel
	{
		thread_index = omp_get_thread_num();
		sc2[thread_index][0] = vdupq_n_f32(sc[0]);
		sc2[thread_index][1] = vdupq_n_f32(sc[1]);
	}
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index += _BNC_S_WIDTH)
	{
		thread_index = omp_get_thread_num();
		
		aij2[thread_index][0] = vld1q_f32(&(a->element[0][index]));
		aij2[thread_index][1] = vld1q_f32(&(a->element[1][index]));

		_bncneon_rds_mul(tmp2[thread_index], sc2[thread_index], aij2[thread_index]);

		vst1q_f32(&(c->element[0][index]), tmp2[thread_index][0]);
		vst1q_f32(&(c->element[1][index]), tmp2[thread_index][1]); 
	}
#else // others
	float tmp[BNCOMP_MAX_NUM_THREADS][DSSIZE], aij[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	#pragma omp parallel for private(thread_index)
	for(index = 0; index < real_total_dim; index++)
	{
		thread_index = omp_get_thread_num();
		
		aij[thread_index][0] = a->element[0][index];
		aij[thread_index][1] = a->element[1][index];

		rds_mul(tmp[thread_index], sc, aij[thread_index]);

		c->element[0][index] = tmp[thread_index][0];
		c->element[1][index] = tmp[thread_index][1]; 
	}
#endif // __AVX2__
}

/* c = a * b */
void _bncomp_mul_dsmatrix(DSMatrix ret, DSMatrix a, DSMatrix b)
{
	int thread_num, thread_index;
	long int i, j, k;
	long row_dim, col_dim, mid_dim;

	/* dimension check */
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_dsmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret->row_dim, ret->col_dim, a->row_dim, a->col_dim, b->row_dim, b->col_dim);
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long real_row_dim, real_col_dim, real_mid_dim;
	float cijval[BNCOMP_MAX_NUM_THREADS][8][DSSIZE];
    __m256 cij[BNCOMP_MAX_NUM_THREADS][DSSIZE], aik[BNCOMP_MAX_NUM_THREADS][DSSIZE], bkj[BNCOMP_MAX_NUM_THREADS][DSSIZE], tmp_mul[BNCOMP_MAX_NUM_THREADS][DSSIZE];

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
            cij[thread_index][0] = _mm256_setzero_ps();
            cij[thread_index][1] = _mm256_setzero_ps();
            for(k = 0; k < real_mid_dim; k += _BNC_S_WIDTH)
            {
                aik[thread_index][0] = _mm256_load_ps(&(a->element[0][i * real_mid_dim + k]));
                aik[thread_index][1] = _mm256_load_ps(&(a->element[1][i * real_mid_dim + k]));

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

                _bncavx2_rds_mul(tmp_mul[thread_index], aik[thread_index], bkj[thread_index]);
				//#pragma omp critical
				_bncavx2_rds_add(cij[thread_index], cij[thread_index], tmp_mul[thread_index]);
            }

            cijval[thread_index][0][0] = cij[thread_index][0][0]; cijval[thread_index][0][1] = cij[thread_index][1][0];
            cijval[thread_index][1][0] = cij[thread_index][0][1]; cijval[thread_index][1][1] = cij[thread_index][1][1];
            cijval[thread_index][2][0] = cij[thread_index][0][2]; cijval[thread_index][2][1] = cij[thread_index][1][2];
            cijval[thread_index][3][0] = cij[thread_index][0][3]; cijval[thread_index][3][1] = cij[thread_index][1][3];
            cijval[thread_index][4][0] = cij[thread_index][0][4]; cijval[thread_index][4][1] = cij[thread_index][1][4];
            cijval[thread_index][5][0] = cij[thread_index][0][5]; cijval[thread_index][5][1] = cij[thread_index][1][5];
            cijval[thread_index][6][0] = cij[thread_index][0][6]; cijval[thread_index][6][1] = cij[thread_index][1][6];
            cijval[thread_index][7][0] = cij[thread_index][0][7]; cijval[thread_index][7][1] = cij[thread_index][1][7];
            rds_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][1]);
            rds_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][2]);
            rds_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][3]);
            rds_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][4]);
            rds_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][5]);
            rds_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][6]);
            rds_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][7]);

            ret->element[0][i * real_col_dim + j] = cijval[thread_index][0][0];
            ret->element[1][i * real_col_dim + j] = cijval[thread_index][0][1];
        }
    }
#elif defined(__AVX512F__) // __AVX512F__
	long real_row_dim, real_col_dim, real_mid_dim;
	float cijval[BNCOMP_MAX_NUM_THREADS][16][DSSIZE];
    __m512 cij[BNCOMP_MAX_NUM_THREADS][DSSIZE], aik[BNCOMP_MAX_NUM_THREADS][DSSIZE], bkj[BNCOMP_MAX_NUM_THREADS][DSSIZE], tmp_mul[BNCOMP_MAX_NUM_THREADS][DSSIZE];

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
            for(k = 0; k < real_mid_dim; k += _BNC_S_WIDTH)
            {
                aik[thread_index][0] = _mm512_load_ps(&(a->element[0][i * real_mid_dim + k]));
                aik[thread_index][1] = _mm512_load_ps(&(a->element[1][i * real_mid_dim + k]));

                bkj[thread_index][0] = _mm512_set_ps(
                    b->element[0][(k + 15) * real_col_dim + j],
                    b->element[0][(k + 14) * real_col_dim + j],
                    b->element[0][(k + 13) * real_col_dim + j],
                    b->element[0][(k + 12) * real_col_dim + j],
                    b->element[0][(k + 11) * real_col_dim + j],
                    b->element[0][(k + 10) * real_col_dim + j],
                    b->element[0][(k +  9) * real_col_dim + j],
                    b->element[0][(k +  8) * real_col_dim + j],
                    b->element[0][(k +  7) * real_col_dim + j],
                    b->element[0][(k +  6) * real_col_dim + j],
                    b->element[0][(k +  5) * real_col_dim + j],
                    b->element[0][(k +  4) * real_col_dim + j],
                    b->element[0][(k +  3) * real_col_dim + j],
                    b->element[0][(k +  2) * real_col_dim + j],
                    b->element[0][(k +  1) * real_col_dim + j],
                    b->element[0][(k     ) * real_col_dim + j]
                );
                bkj[thread_index][1] = _mm512_set_ps(
                    b->element[1][(k + 15) * real_col_dim + j],
                    b->element[1][(k + 14) * real_col_dim + j],
                    b->element[1][(k + 13) * real_col_dim + j],
                    b->element[1][(k + 12) * real_col_dim + j],
                    b->element[1][(k + 11) * real_col_dim + j],
                    b->element[1][(k + 10) * real_col_dim + j],
                    b->element[1][(k +  9) * real_col_dim + j],
                    b->element[1][(k +  8) * real_col_dim + j],
                    b->element[1][(k +  7) * real_col_dim + j],
                    b->element[1][(k +  6) * real_col_dim + j],
                    b->element[1][(k +  5) * real_col_dim + j],
                    b->element[1][(k +  4) * real_col_dim + j],
                    b->element[1][(k +  3) * real_col_dim + j],
                    b->element[1][(k +  2) * real_col_dim + j],
                    b->element[1][(k +  1) * real_col_dim + j],
                    b->element[1][(k     ) * real_col_dim + j]
                );
                _bncavx512_rds_mul(tmp_mul[thread_index], aik[thread_index], bkj[thread_index]);
				//#pragma omp critical
                _bncavx512_rds_add(cij[thread_index], cij[thread_index], tmp_mul[thread_index]);
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
            cijval[thread_index][8][0] = cij[thread_index][0][8];
			cijval[thread_index][8][1] = cij[thread_index][1][8];
            cijval[thread_index][9][0] = cij[thread_index][0][9];
			cijval[thread_index][9][1] = cij[thread_index][1][9];
            cijval[thread_index][10][0] = cij[thread_index][0][10];
			cijval[thread_index][10][1] = cij[thread_index][1][10];
            cijval[thread_index][11][0] = cij[thread_index][0][11];
			cijval[thread_index][11][1] = cij[thread_index][1][11];
            cijval[thread_index][12][0] = cij[thread_index][0][12];
			cijval[thread_index][12][1] = cij[thread_index][1][12];
            cijval[thread_index][13][0] = cij[thread_index][0][13];
			cijval[thread_index][13][1] = cij[thread_index][1][13];
            cijval[thread_index][14][0] = cij[thread_index][0][14];
			cijval[thread_index][14][1] = cij[thread_index][1][14];
            cijval[thread_index][15][0] = cij[thread_index][0][15];
			cijval[thread_index][15][1] = cij[thread_index][1][15];

            rds_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][1]);
            rds_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][2]);
            rds_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][3]);
            rds_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][4]);
            rds_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][5]);
            rds_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][6]);
            rds_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][7]);
            rds_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][8]);
            rds_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][9]);
            rds_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][10]);
            rds_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][11]);
            rds_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][12]);
            rds_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][13]);
            rds_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][14]);
            rds_add(cijval[thread_index][0], cijval[thread_index][0], cijval[thread_index][15]);

            ret->element[0][i * real_col_dim + j] = cijval[thread_index][0][0];
            ret->element[1][i * real_col_dim + j] = cijval[thread_index][0][1];
        }
    }
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic) + OpenMP
	/* SVE2 (VL-agnostic): same outer-product GEMM as the serial path, with
	 * the outer i loop parallelized.  SVE "sizeless" types CANNOT be
	 * BNCOMP_MAX_NUM_THREADS-arrayed scratch — instead every SVE local lives
	 * on the per-iteration (hence per-thread) stack, which is correct and
	 * automatic. */
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
	/* NEON (float32x4_t): outer-product GEMM — one a[i][k] broadcast times
	 * _BNC_S_WIDTH consecutive columns of b at a time.  Mirrors the serial
	 * mul_dsmatrix() NEON path; per-thread scratch is private so threads do not
	 * touch the same float32x4_t[]. */
	long real_row_dim, real_col_dim, real_mid_dim;
	float32x4_t cij[BNCOMP_MAX_NUM_THREADS][DSSIZE], aik[BNCOMP_MAX_NUM_THREADS][DSSIZE], bkj[BNCOMP_MAX_NUM_THREADS][DSSIZE], tmp_mul[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	real_row_dim = ret->real_row_dim;
	real_col_dim = ret->real_col_dim;
	real_mid_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, i, j, k, aik, bkj, cij, tmp_mul)
	for(i = 0; i < real_row_dim; i++)
	{
		thread_index = omp_get_thread_num();
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			_bncneon_set0_ds(cij[thread_index]);
			for(k = 0; k < real_mid_dim; k++)
			{
				aik[thread_index][0] = vdupq_n_f32(a->element[0][i * real_mid_dim + k]);
				aik[thread_index][1] = vdupq_n_f32(a->element[1][i * real_mid_dim + k]);
				bkj[thread_index][0] = vld1q_f32(&(b->element[0][k * real_col_dim + j]));
				bkj[thread_index][1] = vld1q_f32(&(b->element[1][k * real_col_dim + j]));
				_bncneon_rds_mul(tmp_mul[thread_index], aik[thread_index], bkj[thread_index]);
				_bncneon_rds_add(cij[thread_index], cij[thread_index], tmp_mul[thread_index]);
			}
			vst1q_f32(&(ret->element[0][i * real_col_dim + j]), cij[thread_index][0]);
			vst1q_f32(&(ret->element[1][i * real_col_dim + j]), cij[thread_index][1]);
		}
	}
#else // __AVX2__
	float tmp[BNCOMP_MAX_NUM_THREADS][DSSIZE], ret_ij[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_ds(tmp[thread_index]);
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
			rds_set0(ret_ij[thread_index]);
			for(k = 0; k < mid_dim; k++)
			{
				rds_mul(tmp[thread_index], get_dsmatrix_ij(a, i, k), get_dsmatrix_ij(b, k, j));
				#pragma omp critical
				rds_add(ret_ij[thread_index], tmp[thread_index], ret_ij[thread_index]);
			}
			
			set_dsmatrix_ij(ret, i, j, ret_ij[thread_index]);
		}
	}
#endif // __AVX2__
}


/* c := a */
void _bncomp_subst_dsmatrix(DSMatrix c, DSMatrix a)
{
	long int i, j, index;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_dsmatrix\n");
		return;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int real_row_dim, real_col_dim;

	real_row_dim = c->real_row_dim;
	real_col_dim = c->real_col_dim;

	#pragma omp parallel for private(j, index)
	for(i = 0; i < real_row_dim; i++)
	{
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			index = i * real_col_dim + j;
			_mm256_store_ps(&(c->element[0][i * real_col_dim + j]), _mm256_load_ps(&(a->element[0][index])));
			_mm256_store_ps(&(c->element[1][i * real_col_dim + j]), _mm256_load_ps(&(a->element[1][index])));
		}
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int real_row_dim, real_col_dim;

	real_row_dim = c->real_row_dim;
	real_col_dim = c->real_col_dim;

	#pragma omp parallel for private(j, index)
	for(i = 0; i < real_row_dim; i++)
	{
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			index = i * real_col_dim + j;
			_mm512_store_ps(&(c->element[0][i * real_col_dim + j]), _mm512_load_ps(&(a->element[0][index])));
			_mm512_store_ps(&(c->element[1][i * real_col_dim + j]), _mm512_load_ps(&(a->element[1][index])));
		}
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic, OpenMP-safe)
	{
		long _vl = (long)svcntw(); long _N = c->real_row_dim * c->real_col_dim; long _ix;
		for(_ix = 0; _ix < _N; _ix += _vl)
		{
			svbool_t _pg = svwhilelt_b32_s32((int32_t)_ix, (int32_t)_N);
			svst1_f32(_pg, &(c->element[0][_ix]), svld1_f32(_pg, &(a->element[0][_ix])));
			svst1_f32(_pg, &(c->element[1][_ix]), svld1_f32(_pg, &(a->element[1][_ix])));
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	long int real_row_dim, real_col_dim;

	real_row_dim = c->real_row_dim;
	real_col_dim = c->real_col_dim;

	#pragma omp parallel for private(j, index)
	for(i = 0; i < real_row_dim; i++)
	{
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			index = i * real_col_dim + j;
			vst1q_f32(&(c->element[0][i * real_col_dim + j]), vld1q_f32(&(a->element[0][index])));
			vst1q_f32(&(c->element[1][i * real_col_dim + j]), vld1q_f32(&(a->element[1][index])));
		}
	}
#else // others
	#pragma omp parallel for private(j)
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
void _bncomp_setI_dsmatrix(DSMatrix c)
{
	long int i, real_total_dim;
	float tmp0[DSSIZE], tmp1[DSSIZE];

	rds_set_ui(tmp0, 0UL);
	rds_set_ui(tmp1, 1UL);

	real_total_dim = c->real_row_dim * c->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 zero4;

	zero4 = _mm256_setzero_ps();
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm256_store_ps(&c->element[0][i], zero4);
		_mm256_store_ps(&c->element[1][i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero8;

	zero8 = _mm512_setzero_ps();
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm512_store_ps(&c->element[0][i], zero8);
		_mm512_store_ps(&c->element[1][i], zero8);
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
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	float32x4_t zero2;

	zero2 = vdupq_n_f32(0.0);
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		vst1q_f32(&c->element[0][i], zero2);
		vst1q_f32(&c->element[1][i], zero2);
	}
#else // others
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i++)
	{
		c->element[0][i] = 0.0;
		c->element[1][i] = 0.0;	
	}
#endif // __AVX2__

	rds_set_ui(tmp1, 1UL);

	#pragma omp parallel for
	for(i = 0; i < c->row_dim; i++)
	{
		if(i < c->col_dim)
			set_dsmatrix_ij(c, i, i, tmp1);
	}
}

// set a zero matrix
void _bncomp_set0_dsmatrix(DSMatrix mat)
{
	long int i, real_total_dim;

	real_total_dim = mat->real_row_dim * mat->real_col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 zero4;

	zero4 = _mm256_setzero_ps();
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm256_store_ps(&mat->element[0][i], zero4);
		_mm256_store_ps(&mat->element[1][i], zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512 zero8;

	zero8 = _mm512_setzero_ps();
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		_mm512_store_ps(&mat->element[0][i], zero8);
		_mm512_store_ps(&mat->element[1][i], zero8);
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
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon && defined(BNC_ENABLE_NEON)
	float32x4_t zero2;

	zero2 = vdupq_n_f32(0.0);
	#pragma omp parallel for
	for(i = 0; i < real_total_dim; i += _BNC_S_WIDTH)
	{
		vst1q_f32(&mat->element[0][i], zero2);
		vst1q_f32(&mat->element[1][i], zero2);
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
void _bncomp_mul_dsmatrix_dsvec(DSVector v, DSMatrix a, DSVector vb)
{
	long int i, j, row_dim;
	float tmp[BNCOMP_MAX_NUM_THREADS][DSSIZE], tmp1[BNCOMP_MAX_NUM_THREADS][DSSIZE];
	int thread_index;

	/* Check Dimension */
	if((v->dim < a->row_dim) || (vb->dim < a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_dsmatrix_dsvec\n");
		return;
	}

	row_dim = a->row_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int ij_index, real_row_dim, real_col_dim;
	__m256 tmp4[BNCOMP_MAX_NUM_THREADS][DSSIZE], tmp1_4[BNCOMP_MAX_NUM_THREADS][DSSIZE];
	__m256 aij4[BNCOMP_MAX_NUM_THREADS][DSSIZE], vbj4[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j, ij_index)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		_bncavx2_set0_ds(tmp4[thread_index]);
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij4[thread_index][0] = _mm256_load_ps(&(a->element[0][ij_index]));
			aij4[thread_index][1] = _mm256_load_ps(&(a->element[1][ij_index]));
			vbj4[thread_index][0] = _mm256_load_ps(&(vb->element[0][j]));
			vbj4[thread_index][1] = _mm256_load_ps(&(vb->element[1][j]));

			_bncavx2_rds_mul(tmp1_4[thread_index], aij4[thread_index], vbj4[thread_index]);
			_bncavx2_rds_add(tmp4[thread_index], tmp4[thread_index], tmp1_4[thread_index]);
		}
		_bncavx2_rds_sum256(tmp[thread_index], tmp4[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int ij_index, real_col_dim;
	__m512 tmp8[BNCOMP_MAX_NUM_THREADS][DSSIZE], tmp1_8[BNCOMP_MAX_NUM_THREADS][DSSIZE];
	__m512 aij8[BNCOMP_MAX_NUM_THREADS][DSSIZE], vbj8[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j, ij_index)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		_bncavx512_set0_ds(tmp8[thread_index]);
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			thread_index = omp_get_thread_num();

			ij_index = i * real_col_dim + j;
			aij8[thread_index][0] = _mm512_load_ps(&(a->element[0][ij_index]));
			aij8[thread_index][1] = _mm512_load_ps(&(a->element[1][ij_index]));
			vbj8[thread_index][0] = _mm512_load_ps(&(vb->element[0][j]));
			vbj8[thread_index][1] = _mm512_load_ps(&(vb->element[1][j]));

			_bncavx512_rds_mul(tmp1_8[thread_index], aij8[thread_index], vbj8[thread_index]);
			_bncavx512_rds_add(tmp8[thread_index], tmp8[thread_index], tmp1_8[thread_index]);
		}
		_bncavx512_rds_sum512(tmp[thread_index], tmp8[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic) + OpenMP
	/* SVE2 dot-product matvec for DS (float-based). */
	{
		long sve_real_col_dim = a->real_col_dim;
		long sve_vl = (long)svcntw();   /* 32-bit lanes */

		#pragma omp parallel for private(thread_index, i, j) firstprivate(sve_vl, sve_real_col_dim)
		for(i = 0; i < row_dim; i++)
		{
			thread_index = omp_get_thread_num();
			(void)thread_index;
			svfloat32_t acc0, acc1;
			_bncsve2_rds_set0(&acc0, &acc1);

			for(j = 0; j < sve_real_col_dim; j += sve_vl)
			{
				svbool_t pg = svwhilelt_b32_s32((int32_t)j, (int32_t)sve_real_col_dim);
				svfloat32_t aij0 = svld1_f32(pg, &(a->element[0][i * sve_real_col_dim + j]));
				svfloat32_t aij1 = svld1_f32(pg, &(a->element[1][i * sve_real_col_dim + j]));
				svfloat32_t bj0  = svld1_f32(pg, &(vb->element[0][j]));
				svfloat32_t bj1  = svld1_f32(pg, &(vb->element[1][j]));
				svfloat32_t prod0, prod1;
				_bncsve2_rds_mul(pg, &prod0, &prod1, aij0, aij1, bj0, bj1);
				_bncsve2_rds_add(pg, &acc0, &acc1, acc0, acc1, prod0, prod1);
			}

			/* Horizontal sum across VL lanes */
			{
				float acc0_arr[sve_vl], acc1_arr[sve_vl];
				svst1_f32(svptrue_b32(), acc0_arr, acc0);
				svst1_f32(svptrue_b32(), acc1_arr, acc1);
				float sum_ds[DSSIZE] = { 0.0f, 0.0f };
				long lane;
				for(lane = 0; lane < sve_vl; lane++)
				{
					float lane_ds[DSSIZE] = { acc0_arr[lane], acc1_arr[lane] };
					rds_add(sum_ds, sum_ds, lane_ds);
				}
				v->element[0][i] = sum_ds[0];
				v->element[1][i] = sum_ds[1];
			}
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	long int ij_index, real_col_dim;
	float32x4_t tmp2[BNCOMP_MAX_NUM_THREADS][DSSIZE], tmp1_2[BNCOMP_MAX_NUM_THREADS][DSSIZE];
	float32x4_t aij2[BNCOMP_MAX_NUM_THREADS][DSSIZE], vbj2[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j, ij_index)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		_bncneon_set0_ds(tmp2[thread_index]);
		for(j = 0; j < real_col_dim; j += _BNC_S_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij2[thread_index][0] = vld1q_f32(&(a->element[0][ij_index]));
			aij2[thread_index][1] = vld1q_f32(&(a->element[1][ij_index]));
			vbj2[thread_index][0] = vld1q_f32(&(vb->element[0][j]));
			vbj2[thread_index][1] = vld1q_f32(&(vb->element[1][j]));

			_bncneon_rds_mul(tmp1_2[thread_index], aij2[thread_index], vbj2[thread_index]);
			_bncneon_rds_add(tmp2[thread_index], tmp2[thread_index], tmp1_2[thread_index]);
		}
		/* Inline 4-lane horizontal sum (was _bncneon_rds_sum128f). */
		{
			float32x4_t *_s = tmp2[thread_index];
			float _l0[DSSIZE] = { vgetq_lane_f32(_s[0], 0), vgetq_lane_f32(_s[1], 0) };
			float _l1[DSSIZE] = { vgetq_lane_f32(_s[0], 1), vgetq_lane_f32(_s[1], 1) };
			float _l2[DSSIZE] = { vgetq_lane_f32(_s[0], 2), vgetq_lane_f32(_s[1], 2) };
			float _l3[DSSIZE] = { vgetq_lane_f32(_s[0], 3), vgetq_lane_f32(_s[1], 3) };
			rds_set(tmp[thread_index], _l0);
			rds_add(tmp[thread_index], tmp[thread_index], _l1);
			rds_add(tmp[thread_index], tmp[thread_index], _l2);
			rds_add(tmp[thread_index], tmp[thread_index], _l3);
		}
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
	}
#else // others
	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < row_dim; i++)
	{
		thread_index = omp_get_thread_num();

		rds_set_ui(tmp[thread_index], 0UL);
		for(j = 0; j < a->col_dim; j++)
		{
			rds_mul(tmp1[thread_index], get_dsmatrix_ij(a, i, j), get_dsvector_i(vb, j));
			rds_add(tmp[thread_index], tmp[thread_index], tmp1[thread_index]);
		}
		set_dsvector_i(v, i, tmp[thread_index]);
	}
#endif // __AVX2__
}

/* v := a^T * vb */
void _bncomp_mul_dsmatrixt_dsvec(DSVector v, DSMatrix a, DSVector vb)
{
	long int i, j, col_dim;
	float tmp[BNCOMP_MAX_NUM_THREADS][DSSIZE], tmp1[BNCOMP_MAX_NUM_THREADS][DSSIZE];
	int thread_index;

	/* Check Dimension */
	if((v->dim < a->col_dim) || (vb->dim < a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_dsmatrixt_dsvec\n");
		return;
	}

	col_dim = a->col_dim;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int real_row_dim, real_col_dim;
	__m256 tmp4[BNCOMP_MAX_NUM_THREADS][DSSIZE], tmp1_4[BNCOMP_MAX_NUM_THREADS][DSSIZE];
	__m256 aij4[BNCOMP_MAX_NUM_THREADS][DSSIZE], vbj4[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < col_dim; i++)
	{
		thread_index = omp_get_thread_num();

		_bncavx2_set0_ds(tmp4[thread_index]);
		for(j = 0; j < real_row_dim; j += _BNC_S_WIDTH)
		{
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
			vbj4[thread_index][0] = _mm256_load_ps(&(vb->element[0][j]));
			vbj4[thread_index][1] = _mm256_load_ps(&(vb->element[1][j]));

			_bncavx2_rds_mul(tmp1_4[thread_index], aij4[thread_index], vbj4[thread_index]);
			_bncavx2_rds_add(tmp4[thread_index], tmp4[thread_index], tmp1_4[thread_index]);
		}
		_bncavx2_rds_sum256(tmp[thread_index], tmp4[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
	}
#elif defined(__AVX512F__) // __AVX512F__
	long int real_row_dim, real_col_dim;
	__m512 tmp8[BNCOMP_MAX_NUM_THREADS][DSSIZE], tmp1_8[BNCOMP_MAX_NUM_THREADS][DSSIZE];
	__m512 aij8[BNCOMP_MAX_NUM_THREADS][DSSIZE], vbj8[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < col_dim; i++)
	{
		thread_index = omp_get_thread_num();

		_bncavx512_set0_ds(tmp8[thread_index]);
		for(j = 0; j < real_row_dim; j += _BNC_S_WIDTH)
		{
			thread_index = omp_get_thread_num();

			aij8[thread_index][0] = _mm512_set_ps(
				a->element[0][(j + 15) * real_col_dim + i],
				a->element[0][(j + 14) * real_col_dim + i],
				a->element[0][(j + 13) * real_col_dim + i],
				a->element[0][(j + 12) * real_col_dim + i],
				a->element[0][(j + 11) * real_col_dim + i],
				a->element[0][(j + 10) * real_col_dim + i],
				a->element[0][(j +  9) * real_col_dim + i],
				a->element[0][(j +  8) * real_col_dim + i],
				a->element[0][(j +  7) * real_col_dim + i],
				a->element[0][(j +  6) * real_col_dim + i],
				a->element[0][(j +  5) * real_col_dim + i],
				a->element[0][(j +  4) * real_col_dim + i],
				a->element[0][(j +  3) * real_col_dim + i],
				a->element[0][(j +  2) * real_col_dim + i],
				a->element[0][(j +  1) * real_col_dim + i],
				a->element[0][(j     ) * real_col_dim + i]
			);
			aij8[thread_index][1] = _mm512_set_ps(
				a->element[1][(j + 15) * real_col_dim + i],
				a->element[1][(j + 14) * real_col_dim + i],
				a->element[1][(j + 13) * real_col_dim + i],
				a->element[1][(j + 12) * real_col_dim + i],
				a->element[1][(j + 11) * real_col_dim + i],
				a->element[1][(j + 10) * real_col_dim + i],
				a->element[1][(j +  9) * real_col_dim + i],
				a->element[1][(j +  8) * real_col_dim + i],
				a->element[1][(j +  7) * real_col_dim + i],
				a->element[1][(j +  6) * real_col_dim + i],
				a->element[1][(j +  5) * real_col_dim + i],
				a->element[1][(j +  4) * real_col_dim + i],
				a->element[1][(j +  3) * real_col_dim + i],
				a->element[1][(j +  2) * real_col_dim + i],
				a->element[1][(j +  1) * real_col_dim + i],
				a->element[1][(j     ) * real_col_dim + i]
			);
			vbj8[thread_index][0] = _mm512_load_ps(&(vb->element[0][j]));
			vbj8[thread_index][1] = _mm512_load_ps(&(vb->element[1][j]));

			_bncavx512_rds_mul(tmp1_8[thread_index], aij8[thread_index], vbj8[thread_index]);
			_bncavx512_rds_add(tmp8[thread_index], tmp8[thread_index], tmp1_8[thread_index]);
		}
		_bncavx512_rds_sum512(tmp[thread_index], tmp8[thread_index]);
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2 (VL-agnostic) + OpenMP
	/* SVE2 transpose-matvec for DS: outer-product, parallel over i tiles. */
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
			svfloat32_t acc0, acc1;
			_bncsve2_rds_set0(&acc0, &acc1);

			for(j = 0; j < sve_real_row_dim; j++)
			{
				svfloat32_t aji0 = svld1_f32(pg, &(a->element[0][j * sve_real_col_dim + i]));
				svfloat32_t aji1 = svld1_f32(pg, &(a->element[1][j * sve_real_col_dim + i]));
				svfloat32_t bj0  = svdup_n_f32(vb->element[0][j]);
				svfloat32_t bj1  = svdup_n_f32(vb->element[1][j]);
				svfloat32_t prod0, prod1;
				_bncsve2_rds_mul(pg, &prod0, &prod1, aji0, aji1, bj0, bj1);
				_bncsve2_rds_add(pg, &acc0, &acc1, acc0, acc1, prod0, prod1);
			}

			svst1_f32(pg, &(v->element[0][i]), acc0);
			svst1_f32(pg, &(v->element[1][i]), acc1);
		}
	}
#elif (defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON) // Arm Neon
	long int real_row_dim, real_col_dim;
	float32x4_t tmp2[BNCOMP_MAX_NUM_THREADS][DSSIZE], tmp1_2[BNCOMP_MAX_NUM_THREADS][DSSIZE];
	float32x4_t aij2[BNCOMP_MAX_NUM_THREADS][DSSIZE], vbj2[BNCOMP_MAX_NUM_THREADS][DSSIZE];

	real_row_dim = a->real_row_dim;
	real_col_dim = a->real_col_dim;

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < col_dim; i++)
	{
		thread_index = omp_get_thread_num();

		_bncneon_set0_ds(tmp2[thread_index]);
		for(j = 0; j < real_row_dim; j += _BNC_S_WIDTH)
		{
			// 転置行列アクセスのため個別に要素をセット
			float aij_tmp0[2] = {
				a->element[0][(j    ) * real_col_dim + i],
				a->element[0][(j + 1) * real_col_dim + i]
			};
			float aij_tmp1[2] = {
				a->element[1][(j    ) * real_col_dim + i],
				a->element[1][(j + 1) * real_col_dim + i]
			};
			aij2[thread_index][0] = vld1q_f32(aij_tmp0);
			aij2[thread_index][1] = vld1q_f32(aij_tmp1);
			
			vbj2[thread_index][0] = vld1q_f32(&(vb->element[0][j]));
			vbj2[thread_index][1] = vld1q_f32(&(vb->element[1][j]));

			_bncneon_rds_mul(tmp1_2[thread_index], aij2[thread_index], vbj2[thread_index]);
			_bncneon_rds_add(tmp2[thread_index], tmp2[thread_index], tmp1_2[thread_index]);
		}
		/* Inline 4-lane horizontal sum (was _bncneon_rds_sum128f). */
		{
			float32x4_t *_s = tmp2[thread_index];
			float _l0[DSSIZE] = { vgetq_lane_f32(_s[0], 0), vgetq_lane_f32(_s[1], 0) };
			float _l1[DSSIZE] = { vgetq_lane_f32(_s[0], 1), vgetq_lane_f32(_s[1], 1) };
			float _l2[DSSIZE] = { vgetq_lane_f32(_s[0], 2), vgetq_lane_f32(_s[1], 2) };
			float _l3[DSSIZE] = { vgetq_lane_f32(_s[0], 3), vgetq_lane_f32(_s[1], 3) };
			rds_set(tmp[thread_index], _l0);
			rds_add(tmp[thread_index], tmp[thread_index], _l1);
			rds_add(tmp[thread_index], tmp[thread_index], _l2);
			rds_add(tmp[thread_index], tmp[thread_index], _l3);
		}
		v->element[0][i] = tmp[thread_index][0];
		v->element[1][i] = tmp[thread_index][1];
	}
#else // others
	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < a->col_dim; i++)
	{
		thread_index = omp_get_thread_num();

		rds_set_ui(tmp[thread_index], 0UL);
		for(j = 0; j < a->row_dim; j++)
		{
			rds_mul(tmp1[thread_index], get_dsmatrix_ij(a, j, i), get_dsvector_i(vb, j));
			rds_add(tmp[thread_index], tmp[thread_index], tmp1[thread_index]);
		}
		set_dsvector_i(v, i, tmp[thread_index]);
	}
#endif // __AVX2__
}

#if 0  /* Ozaki-scheme parallel matmul disabled for the float (ds/ts/qs) layer:
        * requires split_dsmatrix_dmat / split_dsmatrix_t_dmat / add_dsmatrix_dmat,
        * which live in ds_oz_scheme.c (not implemented). */
// Matrix multiplication based on Ozaki scheme
void _bncomp_mul_dsmatrix_oz(DSMatrix ret, DSMatrix a, int max_num_div_a, DSMatrix b, int max_num_div_b) //, int num_digits)
{
    int i, j;
    int real_num_div_a, real_num_div_b;
    long int row_dim = ret->row_dim, col_dim = ret->col_dim, mid_dim = a->col_dim;
    DMatrix *div_a, *div_b, *div_ret;//div_ret[BNCOMP_MAX_NUM_THREADS];
	int thread_index, thread_num;

    if(mid_dim != b->row_dim)
    {
        fprintf(stderr, "ERROR: mul_dsmatrix_oz mid_dim(a, b) = (%ld, %ld)!\n", mid_dim, b->row_dim);
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
		real_num_div_a = split_dsmatrix_dmat(div_a, max_num_div_a, a);
		//printf("split_dsmatrix_dmat(%d, %d)  ->%d\n", div_a[0]->real_row_dim, div_a[0]->real_col_dim, real_num_div_a);

		#pragma omp section
		real_num_div_b = split_dsmatrix_t_dmat(div_b, max_num_div_b, b);
		//printf("split_dsmatrix_t_dmat(%d, %d)->%d\n", div_b[0]->real_row_dim, div_b[0]->real_col_dim, real_num_div_b);
	}

    set0_dsmatrix(ret);
	//#pragma omp parallel for private(thread_index, j) // , div_ret)
	#pragma omp parallel for private(i, j) //collapse(2) private(j) // , div_ret)
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
            //add_dsmatrix_dmat(ret, ret, div_ret[thread_index]);	 
            //add_dsmatrix_dmat(ret, ret, div_ret[j]);
			#pragma omp critical
				add_dsmatrix_dmat(ret, ret, div_ret[i]);
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
#endif  /* end: _bncomp_mul_dsmatrix_oz (Ozaki scheme) disabled */

#if 0

// Fit dimension to be multiple of min_dim
void _bncomp_mul_cdsmatrix_oz_3m(CDSMatrix ret, CDSMatrix a, int max_num_div_a_real, int max_num_div_a_image, CDSMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    DSMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    t1 = init_dsmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_dsmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_dsmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_dsmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_dsmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    _bncomp_mul_dsmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_a_image);
    _bncomp_sub_dsmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        mul_dsmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        mul_dsmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_a_image);
        add_dsmatrix(ret->im, t3, t4);
    #else // USE_4M
    */
        // 3M
        _bncomp_add_dsmatrix(t3, a->re, a->im);
        _bncomp_add_dsmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        _bncomp_mul_dsmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        _bncomp_sub_dsmatrix(ret->im, ret->im, t1);
        _bncomp_sub_dsmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_dsmatrix(t1);
    free_dsmatrix(t2);
    free_dsmatrix(t3);
    free_dsmatrix(t4);
}

// Fit dimension to be multiple of min_dim
void _bncomp_mul_cdsmatrix_oz_4m(CDSMatrix ret, CDSMatrix a, int max_num_div_a_real, int max_num_div_a_image, CDSMatrix b, int max_num_div_b_real, int max_num_div_b_image)
{
    DSMatrix t1, t2, t3, t4;
    int max_num_div_apa, max_num_div_bpb;

    t1 = init_dsmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_dsmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_dsmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_dsmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_dsmatrix_oz(t1, a->re, max_num_div_a_real, b->re, max_num_div_b_real);
    _bncomp_mul_dsmatrix_oz(t2, a->im, max_num_div_a_image, b->im, max_num_div_a_image);
    _bncomp_sub_dsmatrix(ret->re, t1, t2);

    // 4M
    //#ifdef USE_4M
        _bncomp_mul_dsmatrix_oz(t3, a->im, max_num_div_a_image, b->re, max_num_div_a_real);
        _bncomp_mul_dsmatrix_oz(t4, a->re, max_num_div_a_real, b->im, max_num_div_a_image);
        _bncomp_add_dsmatrix(ret->im, t3, t4);
    /*
    #else // USE_4M 
        // 3M
        add_dsmatrix(t3, a->re, a->im);
        add_dsmatrix(t4, b->re, b->im);
        max_num_div_apa = (max_num_div_a_real < max_num_div_a_image) ? max_num_div_a_image : max_num_div_a_real;
        max_num_div_bpb = (max_num_div_b_real < max_num_div_b_image) ? max_num_div_b_image : max_num_div_b_real;
        mul_dsmatrix_oz(ret->im, t3, max_num_div_apa, t4, max_num_div_bpb);
        sub_dsmatrix(ret->im, ret->im, t1);
        sub_dsmatrix(ret->im, ret->im, t2);
    #endif // USE_4M
    */

    free_dsmatrix(t1);
    free_dsmatrix(t2);
    free_dsmatrix(t3);
    free_dsmatrix(t4);
}

// Simple triple-loop-way matrix multiplication (3M)
void _bncomp_mul_cdsmatrix_3m(CDSMatrix ret, CDSMatrix a, CDSMatrix b)
{
    DSMatrix t1, t2, t3, t4;
 
    t1 = init_dsmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_dsmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_dsmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_dsmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_dsmatrix(t1, a->re, b->re);
    _bncomp_mul_dsmatrix(t2, a->im, b->im);
    _bncomp_sub_dsmatrix(ret->re, t1, t2);

    // 4M
    /*
    #ifdef USE_4M
        _bncomp_mul_dsmatrix(t3, a->im, b->re);
        _bncomp_mul_dsmatrix(t4, a->re, b->im);
        _bncomp_add_dsmatrix(ret->im, t3, t4);
    #else // USE_4M
    */
        // 3M
        _bncomp_add_dsmatrix(t3, a->re, a->im);
        _bncomp_add_dsmatrix(t4, b->re, b->im);
        _bncomp_mul_dsmatrix(ret->im, t3, t4);
        _bncomp_sub_dsmatrix(ret->im, ret->im, t1);
        _bncomp_sub_dsmatrix(ret->im, ret->im, t2);
    //#endif // USE_4M

    free_dsmatrix(t1);
    free_dsmatrix(t2);
    free_dsmatrix(t3);
    free_dsmatrix(t4);
}

// Simple triple-loop-way matrix multiplication (4M)
void _bncomp_mul_cdsmatrix_4m(CDSMatrix ret, CDSMatrix a, CDSMatrix b)
{
    DSMatrix t1, t2, t3, t4;
 
    t1 = init_dsmatrix(ret->re->row_dim, ret->re->col_dim);
    t2 = init_dsmatrix(ret->re->row_dim, ret->re->col_dim);
    t3 = init_dsmatrix(ret->re->row_dim, ret->re->col_dim);
    t4 = init_dsmatrix(ret->re->row_dim, ret->re->col_dim);

    _bncomp_mul_dsmatrix(t1, a->re, b->re);
    _bncomp_mul_dsmatrix(t2, a->im, b->im);
    _bncomp_sub_dsmatrix(ret->re, t1, t2);

    // 4M
    
    //#ifdef USE_4M
        _bncomp_mul_dsmatrix(t3, a->im, b->re);
        _bncomp_mul_dsmatrix(t4, a->re, b->im);
        _bncomp_add_dsmatrix(ret->im, t3, t4);
    //#else // USE_4M
	/*
        // 3M
        _bncomp_add_dsmatrix(t3, a->re, a->im);
        _bncomp_add_dsmatrix(t4, b->re, b->im);
        _bncomp_mul_dsmatrix(ret->im, t3, t4s);
        _bncomp_sub_dsmatrix(ret->im, ret->im, t1);
        _bncomp_sub_dsmatrix(ret->im, ret->im, t2);
	*/
    //#endif // USE_4M

    free_dsmatrix(t1);
    free_dsmatrix(t2);
    free_dsmatrix(t3);
    free_dsmatrix(t4);
}
#endif // 0

