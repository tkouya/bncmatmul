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

//---------------------------------------
// DD
//---------------------------------------

#ifdef USE_DDLINEAR
//--------
// Vector
//--------
/* c := a */
void _bncomp_subst_ddvector(DDVector c, DDVector a)
{
	long int i;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
//	for(i = 0; i < a->dim; i++)
	#pragma omp parallel for
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i));
		_mm256_store_pd(&(c->element[0][i]), _mm256_load_pd(&(a->element[0][i])));
		_mm256_store_pd(&(c->element[1][i]), _mm256_load_pd(&(a->element[1][i])));
	}
#elif defined(__AVX512F__) // __AVX512F__
	#pragma omp parallel for
	for(i = 0; i < a->real_dim; i += _BNC_D_WIDTH)
	{
		//set_dvector_i(c, i, get_dvector_i(a, i));
		_mm512_store_pd(&(c->element[0][i]), _mm512_load_pd(&(a->element[0][i])));
		_mm512_store_pd(&(c->element[1][i]), _mm512_load_pd(&(a->element[1][i])));
	}
#else // others
	#pragma omp parallel for
	for(i = 0; i < a->dim; i++)
		set_ddvector_i(c, i, get_ddvector_i(a, i));
#endif // __AVX2__
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

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
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

		rdd_add(tmp[thread_index], get_ddvector_i(a, i),  get_ddvector_i(b, i));
		set_ddvector_i(c, i, tmp[thread_index]);
	}
#endif // __AVX2__
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
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
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

		rdd_sub(tmp[thread_index], get_ddvector_i(a, i),  get_ddvector_i(b, i));
		set_ddvector_i(c, i, tmp[thread_index]);
	}
#endif // __AVX2__
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

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
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

		//set_ddvector_i(c, i, val * get_ddvector_i(a, i));
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

		//set_ddvector_i(c, i, val * get_ddvector_i(a, i));
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
#endif // __AVX2__
}

/* (a, b) */
void _bncomp_ip_ddvector(double ret[DDSIZE], DDVector a, DDVector b)
{
	int thread_index;
	long int i, index;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: _bncomp_ip_ddvector\n");
		return;
	}
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[BNCOMP_MAX_NUM_THREADS][DDSIZE], b4[BNCOMP_MAX_NUM_THREADS][DDSIZE], ret4[DDSIZE], tmp4[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	_bncavx2_set0_dd(ret4);
	
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a4[thread_index][0] = _mm256_load_pd(&(a->element[0][index]));
		a4[thread_index][1] = _mm256_load_pd(&(a->element[1][index]));
		b4[thread_index][0] = _mm256_load_pd(&(b->element[0][index]));
		b4[thread_index][1] = _mm256_load_pd(&(b->element[1][index]));

//		rdd_mul(tmp, get_ddvector_i(a, i), get_ddvector_i(b, i));
//		rdd_add(ret, ret, tmp);
		_bncavx2_rdd_mul(tmp4[thread_index], a4[thread_index], b4[thread_index]);
		#pragma omp critical
			_bncavx2_rdd_add(ret4, ret4, tmp4[thread_index]);
	}

	_bncavx2_rdd_sum256d(ret, ret4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[BNCOMP_MAX_NUM_THREADS][DDSIZE], b8[BNCOMP_MAX_NUM_THREADS][DDSIZE], ret8[DDSIZE], tmp8[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	_bncavx512_set0_dd(ret8);
	
	#pragma omp parallel for private(thread_index)
	for(index = 0; index < a->real_dim; index += _BNC_D_WIDTH)
	{
		thread_index = omp_get_thread_num();

		a8[thread_index][0] = _mm512_load_pd(&(a->element[0][index]));
		a8[thread_index][1] = _mm512_load_pd(&(a->element[1][index]));
		b8[thread_index][0] = _mm512_load_pd(&(b->element[0][index]));
		b8[thread_index][1] = _mm512_load_pd(&(b->element[1][index]));

//		rdd_mul(tmp, get_ddvector_i(a, i), get_ddvector_i(b, i));
//		rdd_add(ret, ret, tmp);
		_bncavx512_rdd_mul(tmp8[thread_index], a8[thread_index], b8[thread_index]);
		#pragma omp critical
			_bncavx512_rdd_add(ret8, ret8, tmp8[thread_index]);
	}
	_bncavx2_rdd_sum512d(ret, ret8);

#else // others
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	set0_dd(ret);

	#pragma omp parallel for private(thread_index)
	for(i = 0; i < a->dim; i++)
	{
		thread_index = omp_get_thread_num();

		rdd_mul(tmp[thread_index], get_ddvector_i(a, i), get_ddvector_i(b, i));
	#pragma omp critical
		rdd_add(ret, ret, tmp[thread_index]);
	}
#endif // __AVX2__

	return;
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void _bncomp_row_swap_ddmatrix(DDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
// SIMD
#if 0
//#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	int thread_index;
	long int i, j, true_end, true_end_start, true_end_end, real_col_dim, index0, index1;
	double tmp[DDSIZE];
	__m256d tmp256[BNCOMP_MAX_NUM_THREADS][DDSIZE];

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
	#pragma omp parallel for private(thread_index, index0, index1)
	for(i = true_end_start; i < true_end_end; i += _BNC_D_WIDTH)
	{
		//rdd_set(tmp, get_ddmatrix_ij(mat, row_index0, i));
		index0 = row_index0 * real_col_dim + i;
		tmp256[thread_index][0] = _mm256_load_pd(&(mat->element[0][index0]));
		tmp256[thread_index][1] = _mm256_load_pd(&(mat->element[1][index0]));

		//set_ddmatrix_ij(mat, row_index0, i, get_ddmatrix_ij(mat, row_index1, i));
		index1 = row_index1 * real_col_dim + i;
		_mm256_store_pd(&(mat->element[0][index0]), _mm256_load_pd(&(mat->element[0][index1])));
		_mm256_store_pd(&(mat->element[1][index0]), _mm256_load_pd(&(mat->element[1][index1])));

		//set_ddmatrix_ij(mat, row_index1, i, tmp);
		_mm256_store_pd(&(mat->element[0][index1]), tmp256[thread_index][0]);
		_mm256_store_pd(&(mat->element[1][index1]), tmp256[thread_index][1]);
	}

	for(i = true_end_end; i < true_end; i++)
	{
		rdd_set(tmp, get_ddmatrix_ij(mat, row_index0, i));
		set_ddmatrix_ij(mat, row_index0, i, get_ddmatrix_ij(mat, row_index1, i));
		set_ddmatrix_ij(mat, row_index1, i, tmp);
	}
//#elif defined(__AVX512F__) // __AVX512F__
//#else // No SIMD
#endif // 0
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
//#endif // __AVX2__
}

//--------
// Matrix
//--------
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
//void _bncomp_mul_ddmatrix(DDMatrix ret, DDMatrix a, DDMatrix b)
//{
//	mul_ddmatrix(ret, a, b);
//}
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
		//set0_dd(tmp[thread_index]);
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
            cij[thread_index][0] = _mm256_setzero_pd();
            cij[thread_index][1] = _mm256_setzero_pd();
            for(k = 0; k < real_mid_dim; k += _BNC_D_WIDTH)
            {
            /*
                aik[0].val[0] = a->element[0][i * mid_dim + k];
                aik[1].val[0] = a->element[0][i * mid_dim + k + 1];
                aik[2].val[0] = a->element[0][i * mid_dim + k + 2];
                aik[3].val[0] = a->element[0][i * mid_dim + k + 3];
            */
                aik[thread_index][0] = _mm256_load_pd(&(a->element[0][i * real_mid_dim + k]));
                
            /*    aik[thread_index][0] = _mm256_set_pd(
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
                aik[thread_index][1] = _mm256_load_pd(&(a->element[1][i * real_mid_dim + k]));
                
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
                bkj[thread_index][0] = _mm256_set_pd(
                //    b->element[0][ k      * real_col_dim + j],
                //    b->element[0][(k + 1) * real_col_dim + j],
                //    b->element[0][(k + 2) * real_col_dim + j],
                //    b->element[0][(k + 3) * real_col_dim + j]
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
            
                bkj[thread_index][1] = _mm256_set_pd(
                //    b->element[1][ k      * real_col_dim + j],
                //    b->element[1][(k + 1) * real_col_dim + j],
                //    b->element[1][(k + 2) * real_col_dim + j],
                //    b->element[1][(k + 3) * real_col_dim + j]
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
                _bncavx2_rdd_mul(tmp_mul[thread_index], aik[thread_index], bkj[thread_index]);

            /*
                rdd_add(cij.val, cij.val, tmp_mul[0].val);
                rdd_add(cij.val, cij.val, tmp_mul[1].val);
                rdd_add(cij.val, cij.val, tmp_mul[2].val);
                rdd_add(cij.val, cij.val, tmp_mul[3].val);
            */
				#pragma omp critical // needed for Intel Compiler
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
            //rdd_set_ui(cij.val, 0UL);
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
				#pragma omp critical // needed for Intel Compiler
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
#else // __AVX2__
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE], ret_ij[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_dd(tmp[thread_index]);
	}

	//printf("Non SIMD mul_ddmatrix(%ld, %ld)\n", ret->row_dim, ret->col_dim);
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
			rdd_set0(ret_ij[thread_index]);
			for(k = 0; k < mid_dim; k++)
			{
				//rdd_mul(tmp[thread_index], GET_DDMATRIX_IJ(a, i, k), GET_DDMATRIX_IJ(b, k, j));
				rdd_mul(tmp[thread_index], get_ddmatrix_ij(a, i, k), get_ddmatrix_ij(b, k, j));
				#pragma omp critical // needed for Intel Compiler
				rdd_add(ret_ij[thread_index], tmp[thread_index], ret_ij[thread_index]);
			}
			
			set_ddmatrix_ij(ret, i, j, ret_ij[thread_index]);
		}
	}	//printf("end(%ld, %ld)\n", ret->row_dim, ret->col_dim);
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

// AVX2
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int real_row_dim, real_col_dim;

	real_row_dim = c->real_row_dim;
	real_col_dim = c->real_col_dim;

	#pragma omp parallel for private(j, index)
	for(i = 0; i < real_row_dim; i++)
	{
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j));
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
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			//set_dmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j));
			index = i * real_col_dim + j;
			_mm512_store_pd(&(c->element[0][i * real_col_dim + j]), _mm512_load_pd(&(a->element[0][index])));
			_mm512_store_pd(&(c->element[1][i * real_col_dim + j]), _mm512_load_pd(&(a->element[1][index])));
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
//void set0_ddmatrix(DDMatrix mat)
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
	//if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	if((v->dim < a->row_dim) || (vb->dim < a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_ddmatrix_ddvec\n");
		return;
	}

	row_dim = a->row_dim;

// SIMD
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

		//rdd_set_ui(tmp, 0UL);
		_bncavx2_set0_dd(tmp4[thread_index]);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			ij_index = i * real_col_dim + j;
			aij4[thread_index][0] = _mm256_load_pd(&(a->element[0][ij_index]));
			aij4[thread_index][1] = _mm256_load_pd(&(a->element[1][ij_index]));
			vbj4[thread_index][0] = _mm256_load_pd(&(vb->element[0][j]));
			vbj4[thread_index][1] = _mm256_load_pd(&(vb->element[1][j]));

			//rdd_mul(tmp1, get_ddmatrix_ij(a, i, j), get_ddvector_i(vb, j));
			//rdd_add(tmp, tmp, tmp1);
			_bncavx2_rdd_mul(tmp1_4[thread_index], aij4[thread_index], vbj4[thread_index]);
			_bncavx2_rdd_add(tmp4[thread_index], tmp4[thread_index], tmp1_4[thread_index]);
		}
		//set_ddvector_i(v, i, tmp);
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

		//rdd_set_ui(tmp, 0UL);
		_bncavx512_set0_dd(tmp8[thread_index]);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_col_dim; j += _BNC_D_WIDTH)
		{
			thread_index = omp_get_thread_num();

			ij_index = i * real_col_dim + j;
			aij8[thread_index][0] = _mm512_load_pd(&(a->element[0][ij_index]));
			aij8[thread_index][1] = _mm512_load_pd(&(a->element[1][ij_index]));
			vbj8[thread_index][0] = _mm512_load_pd(&(vb->element[0][j]));
			vbj8[thread_index][1] = _mm512_load_pd(&(vb->element[1][j]));

			//rdd_mul(tmp1, get_ddmatrix_ij(a, i, j), get_ddvector_i(vb, j));
			//rdd_add(tmp, tmp, tmp1);
			_bncavx512_rdd_mul(tmp1_8[thread_index], aij8[thread_index], vbj8[thread_index]);
			_bncavx512_rdd_add(tmp8[thread_index], tmp8[thread_index], tmp1_8[thread_index]);
		}
		//set_ddvector_i(v, i, tmp);
		_bncavx512_rdd_sum512d(tmp[thread_index], tmp8[thread_index]);
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
	//if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	if((v->dim < a->col_dim) || (vb->dim < a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncomp_mul_ddmatrixt_ddvec\n");
		return;
	}

	col_dim = a->col_dim;

// SIMD
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

		//rdd_set_ui(tmp, 0UL);
		_bncavx2_set0_dd(tmp4[thread_index]);
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
			vbj4[thread_index][0] = _mm256_load_pd(&(vb->element[0][j]));
			vbj4[thread_index][1] = _mm256_load_pd(&(vb->element[1][j]));

			//rdd_mul(tmp1, get_ddmatrix_ij(a, i, j), get_ddvector_i(vb, j));
			//rdd_add(tmp, tmp, tmp1);
			_bncavx2_rdd_mul(tmp1_4[thread_index], aij4[thread_index], vbj4[thread_index]);
			_bncavx2_rdd_add(tmp4[thread_index], tmp4[thread_index], tmp1_4[thread_index]);
		}
		//set_ddvector_i(v, i, tmp);
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

		//rdd_set_ui(tmp, 0UL);
		_bncavx512_set0_dd(tmp8[thread_index]);
		//for(j = 0; j < a->col_dim; j++)
		for(j = 0; j < real_row_dim; j += _BNC_D_WIDTH)
		{
			thread_index = omp_get_thread_num();

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
			vbj8[thread_index][0] = _mm512_load_pd(&(vb->element[0][j]));
			vbj8[thread_index][1] = _mm512_load_pd(&(vb->element[1][j]));

			//rdd_mul(tmp1, get_ddmatrix_ij(a, i, j), get_ddvector_i(vb, j));
			//rdd_add(tmp, tmp, tmp1);
			_bncavx512_rdd_mul(tmp1_8[thread_index], aij8[thread_index], vbj8[thread_index]);
			_bncavx512_rdd_add(tmp8[thread_index], tmp8[thread_index], tmp1_8[thread_index]);
		}
		//set_ddvector_i(v, i, tmp);
		_bncavx512_rdd_sum512d(tmp[thread_index], tmp8[thread_index]);
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
void _bncomp_mul_ddmatrix_oz(DDMatrix ret, DDMatrix a, int max_num_div_a, DDMatrix b, int max_num_div_b) //, int num_digits)
{
    int i, j;
    int real_num_div_a, real_num_div_b;
    long int row_dim = ret->row_dim, col_dim = ret->col_dim, mid_dim = a->col_dim;
    DMatrix *div_a, *div_b, *div_ret;//div_ret[BNCOMP_MAX_NUM_THREADS];
	int thread_index, thread_num;

    if(mid_dim != b->row_dim)
    {
        fprintf(stderr, "ERROR: mul_ddmatrix_oz mid_dim(a, b) = (%ld, %ld)!\n", mid_dim, b->row_dim);
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
		real_num_div_a = split_ddmatrix_dmat(div_a, max_num_div_a, a);
		//printf("split_ddmatrix_dmat(%d, %d)  ->%d\n", div_a[0]->real_row_dim, div_a[0]->real_col_dim, real_num_div_a);

		#pragma omp section
		real_num_div_b = split_ddmatrix_t_dmat(div_b, max_num_div_b, b);
		//printf("split_ddmatrix_t_dmat(%d, %d)->%d\n", div_b[0]->real_row_dim, div_b[0]->real_col_dim, real_num_div_b);
	}

    set0_ddmatrix(ret);
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
            //add_ddmatrix_dmat(ret, ret, div_ret[thread_index]);	 
            //add_ddmatrix_dmat(ret, ret, div_ret[j]);
			#pragma omp critical
				add_ddmatrix_dmat(ret, ret, div_ret[i]);
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

#endif // USE_DDLINEAR
