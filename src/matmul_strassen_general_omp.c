/********************************************************************************/
/* matmul_strassen_general_omp.c:                                               */
/* Copyright (C) 2014-2020 Tomonori Kouya                                       */
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
#include <stdio.h>
#include <math.h>

//#include "bnc.h"
#include "bncomp.h"

#ifdef USE_IMKL
	#include "mkl.h"
	#include "mkl_cblas.h" // for Intel Math Kernel Library
#endif

#include "matmul_strassen.h"

// count the number of computations
long int _bncomp_num_addsub_mul_mpfmatrix_strassen;	// addition and subtraction
long int _bncomp_num_mul_mul_mpfmatrix_strassen;		// multiplication

// MPF & MPFR
#ifdef USE_GMP

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_add_mpfmatrix_partial(MPFMatrix ret, long int ret_index[4], MPFMatrix mat_a, long int mat_a_index[4], MPFMatrix mat_b, long int mat_b_index[4])
{
	int thread_num, thread_index;
	unsigned long int prec;
	long int i, j;
	long int ret_i[BNCOMP_MAX_NUM_THREADS], ret_j[BNCOMP_MAX_NUM_THREADS], a_i[BNCOMP_MAX_NUM_THREADS], a_j[BNCOMP_MAX_NUM_THREADS], b_i[BNCOMP_MAX_NUM_THREADS], b_j[BNCOMP_MAX_NUM_THREADS];
	long int imax, jmax;
	mpf_t tmp_val[BNCOMP_MAX_NUM_THREADS];

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	prec = prec_mpfmatrix(ret);
	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp_val[thread_index], prec);
	}

	//#pragma omp parallel for schedule(static) private(thread_index, j)
	#pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < imax; i++)
	{
		thread_index = omp_get_thread_num();

		ret_i[thread_index] = ret_index[0] + i;
		a_i[thread_index] = mat_a_index[0] + i;
		b_i[thread_index] = mat_b_index[0] + i;
		//printf("i: %ld %ld %ld\n", ret_i, a_i, b_i);
		for(j = 0; j < jmax; j++)
		{
			ret_j[thread_index] = ret_index[2] + j;
			a_j[thread_index] = mat_a_index[2] + j;
			b_j[thread_index] = mat_b_index[2] + j;
			//printf("j: %ld %ld %ld\n", ret_j, a_j, b_j);

			//tmp_val = get_mpfmatrix_ij(mat_a, a_i, a_j) + get_mpfmatrix_ij(mat_b, b_i, b_j);
			mpf_add(tmp_val[thread_index], get_mpfmatrix_ij(mat_a, a_i[thread_index], a_j[thread_index]), get_mpfmatrix_ij(mat_b, b_i[thread_index], b_j[thread_index]));
			set_mpfmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index], tmp_val[thread_index]);
		}
	}

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp_val[thread_index]);
	}
}

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_sub_mpfmatrix_partial(MPFMatrix ret, long int ret_index[4], MPFMatrix mat_a, long int mat_a_index[4], MPFMatrix mat_b, long int mat_b_index[4])
{
	int thread_num, thread_index;
	unsigned long int prec;
	long int i, j;
	long int ret_i[BNCOMP_MAX_NUM_THREADS], ret_j[BNCOMP_MAX_NUM_THREADS], a_i[BNCOMP_MAX_NUM_THREADS], a_j[BNCOMP_MAX_NUM_THREADS], b_i[BNCOMP_MAX_NUM_THREADS], b_j[BNCOMP_MAX_NUM_THREADS];
	long int imax, jmax;
	mpf_t tmp_val[BNCOMP_MAX_NUM_THREADS];

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	prec = prec_mpfmatrix(ret);
	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp_val[thread_index], prec);
	}

	#pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < imax; i++)
	{

		thread_index = omp_get_thread_num();

		ret_i[thread_index] = ret_index[0] + i;
		a_i[thread_index] = mat_a_index[0] + i;
		b_i[thread_index] = mat_b_index[0] + i;
		for(j = 0; j < jmax; j++)
		{
			ret_j[thread_index] = ret_index[2] + j;
			a_j[thread_index] = mat_a_index[2] + j;
			b_j[thread_index] = mat_b_index[2] + j;

			//tmp_val = get_mpfmatrix_ij(mat_a, a_i, a_j) - get_mpfmatrix_ij(mat_b, b_i, b_j);
			mpf_sub(tmp_val[thread_index], get_mpfmatrix_ij(mat_a, a_i[thread_index], a_j[thread_index]), get_mpfmatrix_ij(mat_b, b_i[thread_index], b_j[thread_index]));
			set_mpfmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index], tmp_val[thread_index]);
		}
	}

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp_val[thread_index]);
	}
}

#ifdef DDEBUG
// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_mpfmatrix_partial(MPFMatrix ret, long int ret_index[4], MPFMatrix mat_a, long int mat_a_index[4])
{
	long int i, j, ret_i, ret_j, a_i, a_j;
	long int imax, jmax;

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	for(i = 0; i < imax; i++)
	{
		ret_i = ret_index[0] + i;
		a_i = mat_a_index[0] + i;
		for(j = 0; j < jmax; j++)
		{
			ret_j = ret_index[2] + j;
			a_j = mat_a_index[2] + j;

			set_mpfmatrix_ij(ret, ret_i, ret_j, get_mpfmatrix_ij(mat_a, a_i, a_j));
		}
	}
}

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_mpfmatrix_partial_checked(MPFMatrix ret, long int ret_index[4], MPFMatrix mat_a, long int mat_a_index[4])
{
	long int i, j, ret_i, ret_j, a_i, a_j;
	long int imax, jmax;

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	for(i = 0; i < imax; i++)
	{
		ret_i = ret_index[0] + i;
		a_i = mat_a_index[0] + i;
		if((ret_i >= 0) && (ret_i < ret->row_dim))
		{
			for(j = 0; j < jmax; j++)
			{
				ret_j = ret_index[2] + j;
				a_j = mat_a_index[2] + j;
				if((ret_j >= 0) && (ret_j < ret->col_dim))
				{
					if((a_i >= 0) && (a_i < mat_a->row_dim) && (a_j >= 0) && (a_j < mat_a->col_dim))
						set_mpfmatrix_ij(ret, ret_i, ret_j, get_mpfmatrix_ij(mat_a, a_i, a_j));
					else
						set_mpfmatrix_ij_ui(ret, ret_i, ret_j, 0UL); // Padding
					//printf("Warning: ret_index = %d, %d, %d, %d\n", ret_i, ret_j, a_i, a_j);
				}
			}
		}
	}
}
#endif // DDEBUG

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_subst_mpfmatrix_partial(MPFMatrix ret, long int ret_index[4], MPFMatrix mat_a, long int mat_a_index[4])
{
	int thread_num, thread_index;
	long int i, j, ret_i[BNCOMP_MAX_NUM_THREADS], ret_j[BNCOMP_MAX_NUM_THREADS], a_i[BNCOMP_MAX_NUM_THREADS], a_j[BNCOMP_MAX_NUM_THREADS];
	long int imax, jmax;

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	thread_num = omp_get_num_threads();

	#pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < imax; i++)
	{
		thread_index = omp_get_thread_num();

		ret_i[thread_index] = ret_index[0] + i;
		a_i[thread_index] = mat_a_index[0] + i;
		for(j = 0; j < jmax; j++)
		{
			ret_j[thread_index] = ret_index[2] + j;
			a_j[thread_index] = mat_a_index[2] + j;

			set_mpfmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index], get_mpfmatrix_ij(mat_a, a_i[thread_index], a_j[thread_index]));
		}
	}
}

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_subst_mpfmatrix_partial_checked(MPFMatrix ret, long int ret_index[4], MPFMatrix mat_a, long int mat_a_index[4])
{
	int thread_num, thread_index;
	long int i, j, ret_i[BNCOMP_MAX_NUM_THREADS], ret_j[BNCOMP_MAX_NUM_THREADS], a_i[BNCOMP_MAX_NUM_THREADS], a_j[BNCOMP_MAX_NUM_THREADS];
	long int imax, jmax;

	thread_num = omp_get_num_threads();

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	#pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < imax; i++)
	{
		thread_index = omp_get_thread_num();

		ret_i[thread_index] = ret_index[0] + i;
		a_i[thread_index] = mat_a_index[0] + i;
		if((ret_i[thread_index] >= 0) && (ret_i[thread_index] < ret->row_dim))
		{
			for(j = 0; j < jmax; j++)
			{
				ret_j[thread_index] = ret_index[2] + j;
				a_j[thread_index] = mat_a_index[2] + j;
				if((ret_j[thread_index] >= 0) && (ret_j[thread_index] < ret->col_dim))
				{
					if((a_i[thread_index] >= 0) && (a_i[thread_index] < mat_a->row_dim) && (a_j[thread_index] >= 0) && (a_j[thread_index] < mat_a->col_dim))
						set_mpfmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index], get_mpfmatrix_ij(mat_a, a_i[thread_index], a_j[thread_index]));
					else
						set_mpfmatrix_ij_ui(ret, ret_i[thread_index], ret_j[thread_index], 0UL); // Padding
					//printf("Warning: ret_index = %d, %d, %d, %d\n", ret_i, ret_j, a_i, a_j);
				}
			}
		}
	}
}

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_neg_mpfmatrix_partial(MPFMatrix ret, long int ret_index[4], MPFMatrix mat_a, long int mat_a_index[4])
{
	int thread_num, thread_index;
	long int i, j, ret_i[BNCOMP_MAX_NUM_THREADS], ret_j[BNCOMP_MAX_NUM_THREADS], a_i[BNCOMP_MAX_NUM_THREADS], a_j[BNCOMP_MAX_NUM_THREADS];
	long int imax, jmax;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS];
	unsigned long prec;

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	prec = prec_mpfmatrix(ret);
	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp[thread_index], prec);
	}

	#pragma omp parallel for schedule(static) private(thread_index, j)
	for(i = 0; i < imax; i++)
	{
		thread_index = omp_get_thread_num();

		ret_i[thread_index] = ret_index[0] + i;
		a_i[thread_index] = mat_a_index[0] + i;
		for(j = 0; j < jmax; j++)
		{
			ret_j[thread_index] = ret_index[2] + j;
			a_j[thread_index] = mat_a_index[2] + j;
			mpf_neg(tmp[thread_index], get_mpfmatrix_ij(mat_a, a_i[thread_index], a_j[thread_index]));
			set_mpfmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index], tmp[thread_index]);
		}
	}

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp[thread_index]);
	}
}

/* c = a * b */
void _bncomp_mul_mpfmatrix_simple(MPFMatrix c, MPFMatrix a, MPFMatrix b)
{
	_bncomp_mul_mpfmatrix(c, a, b);
}

// Block matrix multiplicaiton
void _bncomp_mul_mpfmatrix_block(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim)
{
	int thread_num, thread_index;
	unsigned long prec;
	int row_padding_flag = 0, col_padding_flag = 0, mid_padding_flag = 0;
	long i, j, k, row_dim, col_dim, mid_dim;
	long int num_div_row, num_div_col, num_div_mid, max_num_div;
	long int **mat_a_index, small_mat_a_index[4];
	long int **mat_b_index, small_mat_b_index[4];
	long int **ret_index, small_ret_index[4];
	//MPFMatrix small_ret[1024], small_mat_a[1024], small_mat_b[1024], small_tmp_mat;
	MPFMatrix *small_ret, *small_mat_a, *small_mat_b, *small_tmp_mat;

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(_bncomp_mul_mpfmatrix_block)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	// normal matrix multiplication in case of ret_dim <= 4 
	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	{
		_bncomp_mul_mpfmatrix_simple(ret, mat_a, mat_b);
		return;
	}

	// Number of division of matrix
	num_div_row = (ret->row_dim) / min_dim;
	if((ret->row_dim % min_dim) >= 1)
	{
		row_padding_flag = 1;
		num_div_row++;
	}

	num_div_mid = mid_dim / min_dim;
	if((mid_dim % min_dim) >= 1)
	{
		mid_padding_flag = 1;
		num_div_mid++;
	}

	num_div_col = (ret->col_dim) / min_dim;
	if((ret->col_dim % min_dim) >= 1)
	{
		col_padding_flag = 1;
		num_div_col++;
	}

	max_num_div = (num_div_row > num_div_mid) ? num_div_row : num_div_mid;
	max_num_div = (max_num_div > num_div_col) ? max_num_div : num_div_col;

	// initialize
	prec = prec_mpfmatrix(ret);
	thread_num = omp_get_num_threads();

	mat_a_index = (long int **)calloc(max_num_div, sizeof(long int *));
	mat_b_index = (long int **)calloc(max_num_div, sizeof(long int *));
	ret_index = (long int **)calloc(max_num_div, sizeof(long int *));

	#pragma omp parallel for
	for(i = 0; i < max_num_div; i++)
	{
		mat_a_index[i] = (long int *)calloc(4, sizeof(long int));
		mat_b_index[i] = (long int *)calloc(4, sizeof(long int));
		ret_index[i] = (long int *)calloc(4, sizeof(long int));
	}

	small_ret = (MPFMatrix *)calloc(sizeof(MPFMatrix), num_div_col);
	small_mat_a = (MPFMatrix *)calloc(sizeof(MPFMatrix), num_div_mid);
	small_mat_b = (MPFMatrix *)calloc(sizeof(MPFMatrix), num_div_mid);
	small_tmp_mat = (MPFMatrix *)calloc(sizeof(MPFMatrix), num_div_mid);

	#pragma omp parallel for
	for(i = 0; i < num_div_col; i++)
		small_ret[i] = init2_mpfmatrix(min_dim, min_dim, prec);

	#pragma omp parallel for
	for(i = 0; i < num_div_mid; i++)
	{
		small_mat_a[i] = init2_mpfmatrix(min_dim, min_dim, prec);
		small_mat_b[i] = init2_mpfmatrix(min_dim, min_dim, prec);
		small_tmp_mat[i] = init2_mpfmatrix(min_dim, min_dim, prec);
	}

	// mail loop
	small_mat_a_index[0] = 0;
	small_mat_a_index[1] = min_dim;
	small_mat_a_index[2] = 0;
	small_mat_a_index[3] = min_dim;

	small_mat_b_index[0] = 0;
	small_mat_b_index[1] = min_dim;
	small_mat_b_index[2] = 0;
	small_mat_b_index[3] = min_dim;

	small_ret_index[0] = 0;
	small_ret_index[1] = min_dim;
	small_ret_index[2] = 0;
	small_ret_index[3] = min_dim;

	// mail loop
	for(i = 0; i < num_div_row; i++)
	{
		#pragma omp parallel for
		for(j = 0; j < num_div_mid; j++)
		{
			// copy matrices
			mat_a_index[j][0] = i * min_dim;
			mat_a_index[j][1] = (i + 1) * min_dim;
			mat_a_index[j][2] = j * min_dim;
			mat_a_index[j][3] = (j + 1) * min_dim;
			subst_mpfmatrix_partial_checked(small_mat_a[j], small_mat_a_index, mat_a, mat_a_index[j]);
		}

		for(j = 0; j < num_div_col; j++)
		{
			set0_mpfmatrix(small_ret[j]);

			#pragma omp parallel for
			for(k = 0; k < num_div_mid; k++)
			{
				// copy matrices
				mat_b_index[k][0] = k * min_dim;
				mat_b_index[k][1] = (k + 1) * min_dim;
				mat_b_index[k][2] = j * min_dim;
				mat_b_index[k][3] = (j + 1) * min_dim;
				subst_mpfmatrix_partial_checked(small_mat_b[k], small_mat_b_index, mat_b, mat_b_index[k]);
				//_bncomp_subst_mpfmatrix_partial_checked(small_mat_b[k], small_mat_b_index, mat_b, mat_b_index[k]);
				// ret[j] += small_mat_a[i][k] * small_mat_b[k][j];
				mul_mpfmatrix(small_tmp_mat[k], small_mat_a[k], small_mat_b[k]);
			}

			for(k = 0; k < num_div_mid; k++)
				_bncomp_add_mpfmatrix(small_ret[j], small_ret[j], small_tmp_mat[k]);

			ret_index[j][0] = i * min_dim;
			ret_index[j][1] = (i + 1) * min_dim;
			ret_index[j][2] = j * min_dim;
			ret_index[j][3] = (j + 1) * min_dim;
			//subst_mpfmatrix_partial_checked(ret, ret_index[j], small_ret[j], small_ret_index);
			_bncomp_subst_mpfmatrix_partial_checked(ret, ret_index[j], small_ret[j], small_ret_index);
		}
	}

	#pragma omp parallel for
	for(i = 0; i < max_num_div; i++)
	{
		free(mat_a_index[i]);
		free(mat_b_index[i]);
		free(ret_index[i]);
	}
	free(mat_a_index);
	free(mat_b_index);
	free(ret_index);

	#pragma omp parallel for
	for(i = 0; i < num_div_col; i++)
		free_mpfmatrix(small_ret[i]);

	#pragma omp parallel for
	for(i = 0; i < num_div_mid; i++)
	{
		free_mpfmatrix(small_mat_a[i]);
		free_mpfmatrix(small_mat_b[i]);
		free_mpfmatrix(small_tmp_mat[i]);
	}
	free(small_ret);
	free(small_mat_a);
	free(small_mat_b);
	free(small_tmp_mat);
}

// Block matrix multiplicaiton(poor, so obosolete)
void _bncomp_mul_mpfmatrix_block_old(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim)
{
	int thread_num, thread_index;
	unsigned long prec;
	int row_padding_flag = 0, col_padding_flag = 0, mid_padding_flag = 0;
	long i, j, k, row_dim, col_dim, mid_dim;
	long int num_div_row, num_div_col, num_div_mid;
	long int **mat_a_index, **small_mat_a_index;
	long int **mat_b_index, **small_mat_b_index;
	long int **ret_index, **small_ret_index;
	//DMatrix small_ret[1024], small_mat_a[1024], small_mat_b[1024], small_tmp_mat;
	MPFMatrix *small_ret, *small_mat_a, *small_mat_b, *small_tmp_mat;

	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;

	// normal matrix multiplication in case of ret_dim <= 4 
	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	{
		_bncomp_mul_mpfmatrix_simple(ret, mat_a, mat_b);
		return;
	}

	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_mpfmatrix_block)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}
	
	// Number of division of matrix
	num_div_row = (ret->row_dim) / min_dim;
	if((ret->row_dim % min_dim) >= 1)
	{
		row_padding_flag = 1;
		num_div_row++;
	}

	num_div_mid = mid_dim / min_dim;
	if((mid_dim % min_dim) >= 1)
	{
		mid_padding_flag = 1;
		num_div_mid++;
	}

	num_div_col = (ret->col_dim) / min_dim;
	if((ret->col_dim % min_dim) >= 1)
	{
		col_padding_flag = 1;
		num_div_col++;
	}

	// initialize
	prec = prec_mpfmatrix(ret);
	thread_num = omp_get_num_threads();

	ret_index = (long int **)calloc(num_div_col, sizeof(long int *));
	small_ret_index = (long int **)calloc(num_div_col, sizeof(long int *));

	mat_a_index = (long int **)calloc(num_div_mid, sizeof(long int *));
	mat_b_index = (long int **)calloc(num_div_mid, sizeof(long int *));
	small_mat_a_index = (long int **)calloc(num_div_mid, sizeof(long int *));
	small_mat_b_index = (long int **)calloc(num_div_mid, sizeof(long int *));

	// initialize
	small_ret = (MPFMatrix *)calloc(sizeof(MPFMatrix), num_div_col);
	small_tmp_mat = (MPFMatrix *)calloc(sizeof(MPFMatrix), num_div_mid);
	small_mat_a = (MPFMatrix *)calloc(sizeof(MPFMatrix), num_div_mid);
	small_mat_b = (MPFMatrix *)calloc(sizeof(MPFMatrix), num_div_mid);

	#pragma omp parallel for
	for(i = 0; i < num_div_col; i++)
	{
		ret_index[i] = (long int *)calloc(4, sizeof(long int));
		small_ret_index[i] = (long int *)calloc(4, sizeof(long int));

		small_ret[i] = init2_mpfmatrix(min_dim, min_dim, prec);
	}

	#pragma omp parallel for
	for(i = 0; i < num_div_mid; i++)
	{
		mat_a_index[i] = (long int *)calloc(4, sizeof(long int));
		mat_b_index[i] = (long int *)calloc(4, sizeof(long int));
		small_mat_a_index[i] = (long int *)calloc(4, sizeof(long int));
		small_mat_b_index[i] = (long int *)calloc(4, sizeof(long int));

		small_mat_a[i] = init2_mpfmatrix(min_dim, min_dim, prec);
		small_mat_b[i] = init2_mpfmatrix(min_dim, min_dim, prec);
		small_tmp_mat[i] = init2_mpfmatrix(min_dim, min_dim, prec);
	}

	// mail loop
	//#pragma omp parallel for schedule(static) private(thread_index, j, k)
	for(i = 0; i < num_div_row; i++)
	{
		#pragma omp parallel for shared(i, min_dim)
		for(j = 0; j < num_div_mid; j++)
		{
			// copy matrices
			mat_a_index[j][0] = i * min_dim;
			mat_a_index[j][1] = (i + 1) * min_dim;
			mat_a_index[j][2] = j * min_dim;
			mat_a_index[j][3] = (j + 1) * min_dim;
			small_mat_a_index[j][0] = 0;
			small_mat_a_index[j][1] = min_dim;
			small_mat_a_index[j][2] = 0;
			small_mat_a_index[j][3] = min_dim;
			//subst_mpfmatrix_partial(small_mat_a[j], small_mat_a_index, mat_a, mat_a_index);
			subst_mpfmatrix_partial_checked(small_mat_a[j], small_mat_a_index[j], mat_a, mat_a_index[j]);
		}

		#pragma omp parallel for schedule(static) private(thread_index, k) shared(min_dim)
		for(j = 0; j < num_div_col; j++)
		{
			set0_mpfmatrix(small_ret[j]);

			//#pragma omp parallel for shared(j, min_dim) // ordered
			#pragma omp critical
			for(k = 0; k < num_div_mid; k++)
			{
				// copy matrices
				mat_b_index[k][0] = k * min_dim;
				mat_b_index[k][1] = (k + 1) * min_dim;
				mat_b_index[k][2] = j * min_dim;
				mat_b_index[k][3] = (j + 1) * min_dim;
				small_mat_b_index[k][0] = 0;
				small_mat_b_index[k][1] = min_dim;
				small_mat_b_index[k][2] = 0;
				small_mat_b_index[k][3] = min_dim;
				
				//subst_mpfmatrix_partial(small_mat_b[k], small_mat_b_index, mat_b, mat_b_index);
				//#pragma omp critical
				{
					subst_mpfmatrix_partial_checked(small_mat_b[k], small_mat_b_index[k], mat_b, mat_b_index[k]);
					// ret[j] += small_mat_a[i][k] * small_mat_b[k][j];
					mul_mpfmatrix(small_tmp_mat[k], small_mat_a[k], small_mat_b[k]);
					add_mpfmatrix(small_ret[j], small_ret[j], small_tmp_mat[k]);
				}
			}

			

			ret_index[j][0] = i * min_dim;
			ret_index[j][1] = (i + 1) * min_dim;
			ret_index[j][2] = j * min_dim;
			ret_index[j][3] = (j + 1) * min_dim;
			small_ret_index[j][0] = 0;
			small_ret_index[j][1] = min_dim;
			small_ret_index[j][2] = 0;
			small_ret_index[j][3] = min_dim;
			//subst_mpfmatrix_partial(ret, ret_index, small_ret[j], small_ret_index);
			subst_mpfmatrix_partial_checked(ret, ret_index[j], small_ret[j], small_ret_index[j]);
		}
	}

	// free
	#pragma omp parallel for
	for(i = 0; i < num_div_col; i++)
	{
		free(ret_index[i]);
		free(small_ret_index[i]);
		free_mpfmatrix(small_ret[i]);
		free_mpfmatrix(small_tmp_mat[i]);
	}

	#pragma omp parallel for
	for(i = 0; i < num_div_mid; i++)
	{
		free(mat_a_index[i]);
		free(mat_b_index[i]);
		free(small_mat_a_index[i]);
		free(small_mat_b_index[i]);
		free_mpfmatrix(small_mat_a[i]);
		free_mpfmatrix(small_mat_b[i]);
	}

	free(ret_index);
	free(small_ret_index);
	free(mat_a_index);
	free(mat_b_index);
	free(small_mat_a_index);
	free(small_mat_b_index);
	free(small_ret);
	free(small_mat_a);
	free(small_mat_b);
	free(small_tmp_mat);
}

// Padding to 2-powered dimensional matrix
MPFMatrix _bncomp_init_static_padding_mpfmatrix_strassen(MPFMatrix orig_mat)
{
	int thread_num, thread_index;
	unsigned long prec;
	MPFMatrix ret = NULL;
	long int ret_row_dim, ret_col_dim, min_dim, i, j;

	if(orig_mat == NULL)
	{
		fprintf(stderr, "Warning: orig_mat is empty!(padding_mpfmatrix_strassen)\n");
		return NULL;
	}

	prec = orig_mat->prec;

	ret_row_dim = (long int)pow(2.0, ceil(mylog2((double)orig_mat->row_dim)));
	ret_col_dim = (long int)pow(2.0, ceil(mylog2((double)orig_mat->col_dim)));

	//printf("Padding: row_dim %ld -> %ld, col_dim %ld -> %ld\n", orig_mat->row_dim, ret_row_dim, orig_mat->col_dim, ret_col_dim);

	ret = init2_mpfmatrix(ret_row_dim, ret_col_dim, prec);
	if(ret == NULL)
	{
		fprintf(stderr, "Warning: padding matrix cannot be allocated!(padding_mpfmatrix_strassen)\n");
		return NULL;
	}

	// ret = [ A | 0 ]
	//       [---+---]
	//       [ 0 | I ]
	#pragma omp parallel for schedule(static) private(j)
	for(i = 0; i < orig_mat->row_dim; i++)
	{
		for(j = 0; j < orig_mat->col_dim; j++)
			set_mpfmatrix_ij(ret, i, j, get_mpfmatrix_ij(orig_mat, i, j));
	}

	min_dim = (ret_row_dim < ret_col_dim) ? ret_row_dim : ret_col_dim;
	#pragma omp parallel for
	for(i = orig_mat->row_dim; i < min_dim; i++)
		set_mpfmatrix_ij_ui(ret, i, i, 1UL);

	return ret;
}

// Padding to even dimensional matrix
MPFMatrix _bncomp_init_dynamic_padding_mpfmatrix_strassen(MPFMatrix orig_mat)
{
	int thread_num, thread_index;
	unsigned long prec;
	MPFMatrix ret = NULL;
	long int ret_row_dim, ret_col_dim, min_dim, i, j;

	if(orig_mat == NULL)
	{
		fprintf(stderr, "Warning: orig_mat is empty!(padding_mpfmatrix_strassen)\n");
		return NULL;
	}

	prec = orig_mat->prec;
	ret_row_dim = orig_mat->row_dim;
	ret_col_dim = orig_mat->col_dim;

	if((ret_row_dim % 2) == 1)
		ret_row_dim++;
	if((ret_col_dim % 2) == 1)
		ret_col_dim++;

//	printf("Padding: row_dim %ld -> %ld, col_dim %ld -> %ld\n", orig_mat->row_dim, ret_row_dim, orig_mat->col_dim, ret_col_dim);

	ret = init2_mpfmatrix(ret_row_dim, ret_col_dim, prec);
	if(ret == NULL)
	{
		fprintf(stderr, "Warning: padding matrix cannot be allocated!(padding_mpfmatrix_strassen)\n");
		return NULL;
	}

	// ret = [ A | 0 ]
	//       [---+---]
	//       [ 0 | I ]
	#pragma omp parallel for schedule(static) private(j)
	for(i = 0; i < orig_mat->row_dim; i++)
	{
		for(j = 0; j < orig_mat->col_dim; j++)
			set_mpfmatrix_ij(ret, i, j, get_mpfmatrix_ij(orig_mat, i, j));
	}

/*	min_dim = (ret_row_dim < ret_col_dim) ? ret_row_dim : ret_col_dim;
	for(i = orig_mat->row_dim; i < min_dim; i++)
		set_mpfmatrix_ij_ui(ret, i, i, 1UL);
*/
	return ret;
}

// Padding to even dimensional matrix
MPFMatrix _bncomp_init_dynamic_padding_mpfmatrix_strassen2(MPFMatrix orig_mat, long int min_dim)
{
	int thread_num, thread_index;
	unsigned long prec;
	MPFMatrix ret = NULL;
	long int ret_row_dim, ret_col_dim, i, j;

	if(orig_mat == NULL)
	{
		fprintf(stderr, "Warning: orig_mat is empty!(padding_mpfmatrix_strassen)\n");
		return NULL;
	}

	prec = orig_mat->prec;
	ret_row_dim = orig_mat->row_dim;
	ret_col_dim = orig_mat->col_dim;

	if((ret_row_dim % min_dim) >= 1)
		ret_row_dim = ((ret_row_dim / min_dim) + 1) * min_dim;
	if((ret_col_dim % min_dim) >= 1)
		ret_col_dim = ((ret_col_dim / min_dim) + 1) * min_dim;

//	printf("Padding: row_dim %ld -> %ld, col_dim %ld -> %ld\n", orig_mat->row_dim, ret_row_dim, orig_mat->col_dim, ret_col_dim);

	ret = init2_mpfmatrix(ret_row_dim, ret_col_dim, prec);
	if(ret == NULL)
	{
		fprintf(stderr, "Warning: padding matrix cannot be allocated!(padding_mpfmatrix_strassen)\n");
		return NULL;
	}

	// ret = [ A | 0 ]
	//       [---+---]
	//       [ 0 | I ]
	#pragma omp parallel for schedule(static) private(j)
	for(i = 0; i < orig_mat->row_dim; i++)
	{
		for(j = 0; j < orig_mat->col_dim; j++)
			set_mpfmatrix_ij(ret, i, j, get_mpfmatrix_ij(orig_mat, i, j));
	}
/*
	min_dim = (ret_row_dim < ret_col_dim) ? ret_row_dim : ret_col_dim;
	for(i = orig_mat->row_dim; i < min_dim; i++)
		set_mpfmatrix_ij_ui(ret, i, i, 1UL);
*/
	return ret;
}

// Strassen's Algorithm with static padding
void _bncomp_mul_mpfmatrix_strassen_odd_padding(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim)
{
	long int tmp_ret_index[4], ret_index[4];
	MPFMatrix tmp_ret, tmp_mat_a, tmp_mat_b;

	// padding
#ifdef USE_STATIC_PADDING
	tmp_ret = _bncomp_init_static_padding_mpfmatrix_strassen(ret);
	tmp_mat_a = _bncomp_init_static_padding_mpfmatrix_strassen(mat_a);
	tmp_mat_b = _bncomp_init_static_padding_mpfmatrix_strassen(mat_b);
#else
//	tmp_ret = _bncomp_init_dynamic_padding_mpfmatrix_strassen(ret);
//	tmp_mat_a = _bncomp_init_dynamic_padding_mpfmatrix_strassen(mat_a);
//	tmp_mat_b = _bncomp_init_dynamic_padding_mpfmatrix_strassen(mat_b);
	tmp_ret = _bncomp_init_dynamic_padding_mpfmatrix_strassen2(ret, min_dim);
	tmp_mat_a = _bncomp_init_dynamic_padding_mpfmatrix_strassen2(mat_a, min_dim);
	tmp_mat_b = _bncomp_init_dynamic_padding_mpfmatrix_strassen2(mat_b, min_dim);
	//tmp_ret = init_dynamic_padding_mpfmatrix_strassen2(ret, min_dim);
	//tmp_mat_a = init_dynamic_padding_mpfmatrix_strassen2(mat_a, min_dim);
	//tmp_mat_b = init_dynamic_padding_mpfmatrix_strassen2(mat_b, min_dim);
#endif

	// strassen
#ifdef USE_WINOGRAD
	_bncomp_mul_mpfmatrix_winograd_even_psec(tmp_ret, tmp_mat_a, tmp_mat_b, min_dim);
//	_bncomp_mul_mpfmatrix_winograd_even(tmp_ret, tmp_mat_a, tmp_mat_b, min_dim);
#else
	_bncomp_mul_mpfmatrix_strassen_even_psec(tmp_ret, tmp_mat_a, tmp_mat_b, min_dim);
//	_bncomp_mul_mpfmatrix_strassen_even(tmp_ret, tmp_mat_a, tmp_mat_b, min_dim);
#endif

	//printf("tmp_ret->row_dim, col_dim: %ld, %ld\n", tmp_ret->row_dim, tmp_ret->col_dim);
	//printf("    ret->row_dim, col_dim: %ld, %ld\n", ret->row_dim, ret->col_dim);

	// substitute
	tmp_ret_index[0] = 0;
	tmp_ret_index[1] = ret->row_dim;
	tmp_ret_index[2] = 0;
	tmp_ret_index[3] = ret->col_dim;
	ret_index[0] = 0;
	ret_index[1] = ret->row_dim;
	ret_index[2] = 0;
	ret_index[3] = ret->col_dim;

	_bncomp_subst_mpfmatrix_partial(ret, ret_index, tmp_ret, tmp_ret_index);
	//subst_mpfmatrix_partial(ret, ret_index, tmp_ret, tmp_ret_index);

	// free
	free_mpfmatrix(tmp_ret);
	free_mpfmatrix(tmp_mat_a);
	free_mpfmatrix(tmp_mat_b);

}

// clear counter
void _bncomp_reset_num_mul_mpfmatrix_strassen(void)
{
	_bncomp_num_addsub_mul_mpfmatrix_strassen = 0;
	_bncomp_num_mul_mul_mpfmatrix_strassen = 0;
}

// get counters
void _bncomp_get_num_mul_mpfmatrix_strassen(long int *_bncomp_num_addsub, long int *_bncomp_num_mul)
{
	//printf("num_addsub_mul_mpfmatrix_strassen: %ld\n", num_addsub_mul_mpfmatrix_strassen);
	//printf("num_mul_mul_mpfmatrix_strassen   : %ld\n", num_mul_mul_mpfmatrix_strassen);

	if(_bncomp_num_addsub != NULL)
		*_bncomp_num_addsub = _bncomp_num_addsub_mul_mpfmatrix_strassen;
	if(_bncomp_num_mul != NULL)
		*_bncomp_num_mul = _bncomp_num_mul_mul_mpfmatrix_strassen;
}

// print counters
void _bncomp_print_num_mul_mpfmatrix_strassen(long int *_bncomp_num_addsub, long int *_bncomp_num_mul)
{
	printf("_bncomp_num_addsub_mul_mpfmatrix_strassen: %ld\n", _bncomp_num_addsub_mul_mpfmatrix_strassen);
	printf("_bncomp_num_mul_mul_mpfmatrix_strassen   : %ld\n", _bncomp_num_mul_mul_mpfmatrix_strassen);

	_bncomp_get_num_mul_mpfmatrix_strassen(_bncomp_num_addsub, _bncomp_num_mul);

}

// Fit dimension to be multiple of min_dim
void _bncomp_mul_mpfmatrix_strassen(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim)
{
	long int row_k, col_k, mid_row_k, mid_col_k;
	long int row_dim, col_dim, mid_dim;
//	MPFVector diag_left, diag_right;

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_mpfmatrix_strassen)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	// normal matrix multiplication in case of ret_dim <= 4 
	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	{
		//mul_mpfmatrix(ret, mat_a, mat_b);

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		_bncomp_num_mul_mul_mpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		_bncomp_mul_mpfmatrix_simple(ret, mat_a, mat_b);
		//mul_mpfmatrix_simple(ret, mat_a, mat_b);
		return;
	}

	// scaling 
//	diag_left = init2_mpfvector(row_dim, mat_a->prec);
//	diag_right = init2_mpfvector(col_dim, mat_b->prec);

//	left_scaling_mpfmatrix(mat_a, diag_left, NULL);
//	right_scaling_mpfmatrix(mat_b, diag_right, NULL);

	// dynamic peeling in case of odd dim
	// [ A11   a12 ] [ B11   b12 ] = [ A11*B11 + a12 * b21^T   A11*b12 + a12 * b22    ]
	// [ a21^T a22 ] [ b21^T b22 ]   [ a21^T*B11 + a22 * b21^T a21^T * b12 + a22 * b22]
	if((ret->row_dim % 2 == 1) || (ret->col_dim % 2 == 1) || (mid_dim % 2 == 1))
	{
#ifdef PEELING_ONLY
		_bncomp_mul_mpfmatrix_strassen_odd_peeling(ret, mat_a, mat_b, min_dim);
#elif PADDING_ONLY
		_bncomp_mul_mpfmatrix_strassen_odd_padding(ret, mat_a, mat_b, min_dim);
#else
		row_k = (long int)floor(mylog2((double)ret->row_dim));
		col_k = (long int)floor(mylog2((double)ret->col_dim));

		mid_row_k = (long int)pow(2.0, row_k - 1) * 3;
		mid_col_k = (long int)pow(2.0, col_k - 1) * 3;

		//printf("2^%ld <= %ld <= 2^%ld\n", row_k, ret->row_dim, row_k + 1);

		// dynamic peeling
		//if(ret->row_dim < mid_row_k)
		if((ret->row_dim % min_dim) < (min_dim / 2))
		{
			//mul_mpfmatrix_strassen_odd_peeling(ret, mat_a, mat_b, min_dim);
			_bncomp_mul_mpfmatrix_strassen_odd_peeling(ret, mat_a, mat_b, min_dim);
		}
		// padding
		else
			_bncomp_mul_mpfmatrix_strassen_odd_padding(ret, mat_a, mat_b, min_dim);
#endif
	}
	// normal strassen algorithm in case of even dim
	else
	{
		//printf("%ld is even -> ", ret->row_dim);
#ifdef USE_WINOGRAD
		_bncomp_mul_mpfmatrix_winograd_even_psec(ret, mat_a, mat_b, min_dim);
		//_bncomp_mul_mpfmatrix_winograd_even(ret, mat_a, mat_b, min_dim);
#else
		_bncomp_mul_mpfmatrix_strassen_even_psec(ret, mat_a, mat_b, min_dim);
		//_bncomp_mul_mpfmatrix_strassen_even(ret, mat_a, mat_b, min_dim);
#endif
		//printf("end\n");
	}

//	mul_mpfmatrix_ddiag_mat(ret, diag_left, 0, ret, diag_right, 0);

//	free_mpfvector(diag_left);
//	free_mpfvector(diag_right);

}

// Strassen's Algorithm with Dynamic peeling
void _bncomp_mul_mpfmatrix_strassen_odd_peeling(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim)
{
	unsigned long prec;
	long int i, j, row_dim, row_dim_h, col_dim, col_dim_h, mid_dim, mid_dim_h, tmp_dim_h, tmp_dim;
	MPFMatrix mat_a11, mat_b11, mat_c11, mat_tmp;
	MPFVector vec_a12, vec_a21, vec_b12, vec_b21, vec_c12, vec_c21, vec_tmp12, vec_tmp21;
	mpf_t a22, b22, c22, tmp;

	// initialize
	prec = ret->prec;
	//printf("odd_peeling...\n");

	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_mpfmatrix_strassen_odd_peeling)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	if((row_dim % 2) >= 1)
		row_dim_h = row_dim - 1;
	else
		row_dim_h = row_dim;

	if((mid_dim % 2) >= 1)
		mid_dim_h = mid_dim - 1;
	else
		mid_dim_h = mid_dim;

	if((col_dim % 2) >= 1)
		col_dim_h = col_dim - 1;
	else
		col_dim_h = col_dim;	// normal matrix multiplication in case of ret_dim <= 4 

	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		_bncomp_num_mul_mul_mpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		//mul_mpfmatrix_simple(ret, mat_a, mat_b);
		_bncomp_mul_mpfmatrix_simple(ret, mat_a, mat_b);
		return;
	}

	// tmp_dim_h = mid_dim_h or col_dim_h
	tmp_dim_h = mat_b->col_dim - 1;
	if(mid_dim_h < tmp_dim_h)
		tmp_dim_h = mid_dim_h;

	// Initialize
	mat_a11 = init2_mpfmatrix(row_dim_h, mid_dim_h, prec);
	mat_b11 = init2_mpfmatrix(mid_dim_h, col_dim_h, prec);
	mat_c11 = init2_mpfmatrix(row_dim_h, col_dim_h, prec);
	mat_tmp = init2_mpfmatrix(row_dim_h, col_dim_h, prec);

	vec_a12 = NULL;
	vec_b21 = NULL;
	vec_b12 = NULL;
	vec_a21 = NULL;
	vec_c12 = init2_mpfvector(row_dim_h, prec);
	vec_c21 = init2_mpfvector(col_dim_h, prec);
	vec_tmp12 = init2_mpfvector(row_dim_h, prec);
	vec_tmp21 = init2_mpfvector(col_dim_h, prec);

	mpf_init2(a22, prec); mpf_set_ui(a22, 0UL);
	mpf_init2(b22, prec); mpf_set_ui(b22, 0UL);
	mpf_init2(c22, prec);
	mpf_init2(tmp, prec);

	// set matrix elements to mat_a11
	#pragma omp parallel for schedule(static) private(j)
	for(i = 0; i < row_dim_h; i++)
	{
		for(j = 0; j < mid_dim_h; j++)
			set_mpfmatrix_ij(mat_a11, i, j, get_mpfmatrix_ij(mat_a, i, j));
	}

	// set matrix elements to vec_b11
	#pragma omp parallel for schedule(static) private(j)
	for(i = 0; i < mid_dim_h; i++)
	{
		for(j = 0; j < col_dim_h; j++)
			set_mpfmatrix_ij(mat_b11, i, j, get_mpfmatrix_ij(mat_b, i, j));
	}

	// set matrix elements to vec_a12 and vec_b21
	if(mid_dim_h < mid_dim)
	{
		//printf("set vec_a12, b21\n");
		vec_a12 = init2_mpfvector(row_dim_h, prec);
		vec_b21 = init2_mpfvector(col_dim_h, prec); // fix!: 2014-03-19 by T.Kouya

		#pragma omp parallel for
		for(i = 0; i < row_dim_h; i++)
			set_mpfvector_i(vec_a12, i, get_mpfmatrix_ij(mat_a, i, mat_a->col_dim - 1));

		//printf("set vec_a12\n");

		#pragma omp parallel for
		for(i = 0; i < col_dim_h; i++)
			set_mpfvector_i(vec_b21, i, get_mpfmatrix_ij(mat_b, mat_b->row_dim - 1, i));

		//printf("set vec_b21\n");
	}

	// set matrix elements to vec_a21
	if(row_dim_h < row_dim)
	{
		//printf("set vec_a21, a22\n");
		vec_a21 = init2_mpfvector(mid_dim_h, prec);

		#pragma omp parallel for
		for(i = 0; i < mid_dim_h; i++)
			set_mpfvector_i(vec_a21, i, get_mpfmatrix_ij(mat_a, mat_a->row_dim - 1, i));

		//mpf_init2(a22, prec);
		if(mid_dim_h < mid_dim)
			mpf_set(a22, get_mpfmatrix_ij(mat_a, mat_a->row_dim - 1, mat_a->col_dim - 1));

	}

	// set matrix elements to vec_b12
	if(col_dim_h < col_dim)
	{
		//printf("set vec_a12, b22\n");
		vec_b12 = init2_mpfvector(mid_dim_h, prec);

		#pragma omp parallel for
		for(i = 0; i < mid_dim_h; i++)
			set_mpfvector_i(vec_b12, i, get_mpfmatrix_ij(mat_b, i, mat_b->col_dim - 1));

		//mpf_init2(b22, prec);
		if(mid_dim_h < mid_dim)
			mpf_set(b22, get_mpfmatrix_ij(mat_b, mat_b->row_dim - 1, mat_b->col_dim - 1));
	}

	// dynamic peeling in case of odd dim
	// [ A11   a12 ] [ B11   b12 ] = [ A11*B11 + a12 * b21^T   A11*b12 + a12 * b22     ]
	// [ a21^T a22 ] [ b21^T b22 ]   [ a21^T*B11 + a22 * b21^T a21^T * b12 + a22 * b22 ]

	//printf("starting C11 = A11 * B11...\n");

	// C11 = A11 * B11
#ifdef USE_WINOGRAD
	_bncomp_mul_mpfmatrix_winograd_even_psec(mat_c11, mat_a11, mat_b11, min_dim);
	//_bncomp_mul_mpfmatrix_winograd_even(mat_c11, mat_a11, mat_b11, min_dim);
#else
	_bncomp_mul_mpfmatrix_strassen_even_psec(mat_c11, mat_a11, mat_b11, min_dim);
	//_bncomp_mul_mpfmatrix_strassen_even(mat_c11, mat_a11, mat_b11, min_dim);
#endif

	//printf("C11 = A11 * B11\n");

	// C11 += a12 * b21^T
	if((vec_a12 != NULL) && (vec_b21 != NULL))
	{
		//printf("starting C11 += a12 * b21...\n");

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;
		_bncomp_num_mul_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

		#pragma omp parallel for schedule(static) private(j)
		for(i = 0; i < row_dim_h; i++)
		{
			for(j = 0; j < col_dim_h; j++)
			{
				mpf_mul(get_mpfmatrix_ij(mat_tmp, i, j), get_mpfvector_i(vec_a12, i), get_mpfvector_i(vec_b21, j));
			}
		}
		//add_mpfmatrix(mat_c11, mat_c11, mat_tmp);
		_bncomp_add_mpfmatrix(mat_c11, mat_c11, mat_tmp);
		//printf("C11 += a12 * b21...\n");
	}

	#pragma omp parallel for schedule(static) private(j)
	for(i = 0; i < row_dim_h; i++)
		for(j = 0; j < col_dim_h; j++) 
			set_mpfmatrix_ij(ret, i, j, get_mpfmatrix_ij(mat_c11, i, j));

	// c12 := A11 * b12
	if(vec_b12 != NULL)
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mat_a11->row_dim * mat_a11->col_dim;
		_bncomp_num_mul_mul_mpfmatrix_strassen += mat_a11->row_dim * mat_a11->col_dim;

		//mul_mpfmatrix_mpfvec(vec_c12, mat_a11, vec_b12);
		_bncomp_mul_mpfmatrix_mpfvec(vec_c12, mat_a11, vec_b12);

		//printf("c12 = A11 * b12\n");
	}

 	// c12 += b22 * a12
 	if((vec_a12 != NULL) && (mpf_cmp_ui(b22, 0UL) != 0))
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += vec_c12->dim;
		_bncomp_num_mul_mul_mpfmatrix_strassen += vec_tmp12->dim;

		_bncomp_cmul_mpfvector(vec_tmp12, b22, vec_a12, vec_tmp12->dim);
		_bncomp_add_mpfvector(vec_c12, vec_c12, vec_tmp12, vec_c12->dim);
		//cmul_mpfvector(vec_tmp12, b22, vec_a12);
		//add_mpfvector(vec_c12, vec_c12, vec_tmp12);
	}
	//printf("c12 += b22 * a12\n");

	// Fix! 2016-08-18 by T.Kouya
	if((vec_b12 != NULL) || ((vec_a12 != NULL) && (mpf_cmp_ui(b22, 0UL) != 0)))
	{
		#pragma omp parallel for
		for(i = 0; i < row_dim_h; i++) // Fix! 2016-08-18 by T.Kouya
			set_mpfmatrix_ij(ret, i, ret->col_dim - 1, get_mpfvector_i(vec_c12, i));
	}

	//printf("vec_c12\n");

	// c21 := a21^T*B11
	if(vec_a21 != NULL)
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mat_b11->row_dim * mat_b11->col_dim;
		_bncomp_num_mul_mul_mpfmatrix_strassen += mat_b11->row_dim * mat_b11->col_dim;

		//mul_mpfmatrixt_mpfvec(vec_c21, mat_b11, vec_a21);
		_bncomp_mul_mpfmatrixt_mpfvec(vec_c21, mat_b11, vec_a21);
	}
	//printf("c21 = a21^T * B11\n");

	// c21 += a22 * b21^T
	if((vec_b21 != NULL) && (mpf_cmp_ui(a22, 0UL) != 0))
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += vec_c21->dim;
		_bncomp_num_mul_mul_mpfmatrix_strassen += vec_tmp21->dim;

		_bncomp_cmul_mpfvector(vec_tmp21, a22, vec_b21, vec_tmp21->dim);
		_bncomp_add_mpfvector(vec_c21, vec_c21, vec_tmp21, vec_c21->dim);
		//cmul_mpfvector(vec_tmp21, a22, vec_b21);
		//add_mpfvector(vec_c21, vec_c21, vec_tmp21);
	}
	//printf("c21 += a22 * b21^T\n");
	if(vec_a21 != NULL)
	{
		#pragma omp parallel for
		for(i = 0; i < col_dim_h; i++)
			set_mpfmatrix_ij(ret, ret->row_dim - 1, i, get_mpfvector_i(vec_c21, i));
	}
	//printf("c21\n");

	// c22 := a21^T * b12
	if((vec_a21 != NULL) && (vec_b12 != NULL))
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += vec_a21->dim;
		_bncomp_num_mul_mul_mpfmatrix_strassen += vec_tmp21->dim;

		_bncomp_ip_mpfvector(c22, vec_a21, vec_b12);
		//ip_mpfvector(c22, vec_a21, vec_b12);
	}
	//printf("c22 += a21^T * b12\n");

	// c22 += a22 * b22
	if((mpf_cmp_ui(a22, 0UL) != 0) || (mpf_cmp_ui(b22, 0UL) != 0))
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += 1;
		_bncomp_num_mul_mul_mpfmatrix_strassen += 1;

		mpf_mul(tmp, a22, b22);
		mpf_add(c22, c22, tmp);
	}
	//printf("c22 += a22 * b22\n");

	if((vec_a21 != NULL) && (vec_b12 != NULL))
		set_mpfmatrix_ij(ret, ret->row_dim - 1, ret->col_dim - 1, c22);

	//printf("c22\n");

	// free
	free_mpfmatrix(mat_a11);
	free_mpfmatrix(mat_b11);
	free_mpfmatrix(mat_c11);
	//printf("free_mpfmatrix a11, b11, c11\n");
	free_mpfmatrix(mat_tmp);

	//printf("free_mpfmatrix\n");

	if(vec_a12 != NULL)
		free_mpfvector(vec_a12);

	if(vec_b21 != NULL)
		free_mpfvector(vec_b21);

	if(vec_a21 != NULL)
		free_mpfvector(vec_a21);

	if(vec_b12 != NULL)
		free_mpfvector(vec_b12);

	//printf("free_mpfvector\n");

	free_mpfvector(vec_c12);
	free_mpfvector(vec_c21);
	free_mpfvector(vec_tmp12);
	free_mpfvector(vec_tmp21);

	//printf("free_mpfvector2\n");

	mpf_clear(a22);
	mpf_clear(b22);
	mpf_clear(c22);
	mpf_clear(tmp);
	//printf("mpfclear\n");
	//printf("...odd_peeling end\n");
}

// Strassen's Algorithm
void _bncomp_mul_mpfmatrix_strassen_even(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim)
{
	unsigned long prec;
//	long int min_dim = 4; // = 2^2
	MPFMatrix mat_p[7], mat_tmp_a[7], mat_tmp_b[7], mat_tmp_c[4];
	long int row_dim_h, row_dim, col_dim_h, col_dim, mid_dim_h, mid_dim;
	long int ret_index[4], mat_tmp_a_index[4], mat_tmp_b_index[4], mat_ar_index[7][4], mat_al_index[7][4], mat_br_index[7][4], mat_bl_index[7][4], mat_c_index[4][4];
	long int i;

	// initialize
	prec = ret->prec;
	//printf("strassen_even start...\n");

	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_mpfmatrix_strassen_even)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	row_dim_h = row_dim / 2;
	col_dim_h = col_dim / 2;
	mid_dim_h = mid_dim / 2;

	//printf("mat_a: %ld, %ld\n", mat_a->row_dim, mat_a->col_dim);
	//printf("mat_b: %ld, %ld\n", mat_b->row_dim, mat_b->col_dim);
	//printf("row_dim_h, col_dim_h, mid_dim_h: %ld, %ld, %ld\n", row_dim_h, col_dim_h, mid_dim_h);

	// normal matrix multiplication in case of ret_dim <= 4 
//	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	if((ret->row_dim <= min_dim) || (ret->col_dim <= min_dim) || (mid_dim <= min_dim))
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		_bncomp_num_mul_mul_mpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		//mul_mpfmatrix_simple(ret, mat_a, mat_b);
		_bncomp_mul_mpfmatrix_simple(ret, mat_a, mat_b);
		return;
	}

	for(i = 0; i < 7; i++)
	{
		mat_p[i] = init2_mpfmatrix(row_dim_h, col_dim_h, prec);
		mat_tmp_a[i] = init2_mpfmatrix(row_dim_h, mid_dim_h, prec);
		mat_tmp_b[i] = init2_mpfmatrix(mid_dim_h, col_dim_h, prec);
	}
	for(i = 0; i < 4; i++)
		mat_tmp_c[i] = init2_mpfmatrix(row_dim_h, col_dim_h, prec);

	ret_index[0] = 0;
	ret_index[1] = row_dim_h;
	ret_index[2] = 0;
	ret_index[3] = col_dim_h;

	mat_tmp_a_index[0] = 0;
	mat_tmp_a_index[1] = row_dim_h;
	mat_tmp_a_index[2] = 0;
	mat_tmp_a_index[3] = mid_dim_h;

	mat_tmp_b_index[0] = 0;
	mat_tmp_b_index[1] = mid_dim_h;
	mat_tmp_b_index[2] = 0;
	mat_tmp_b_index[3] = col_dim_h;

	// -------------------------------
	// P1 := (A11 + A22) * (B11 + B22)
	//--------------------------------

	// A11 + A22
	mat_ar_index[0][0] = 0;
	mat_ar_index[0][1] = row_dim_h;
	mat_ar_index[0][2] = 0;
	mat_ar_index[0][3] = mid_dim_h;

	mat_al_index[0][0] = row_dim_h;
	mat_al_index[0][1] = row_dim;
	mat_al_index[0][2] = mid_dim_h;
	mat_al_index[0][3] = mid_dim;
//	printf("P1a:\n");

	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

	//add_mpfmatrix_partial(mat_tmp_a[0], mat_tmp_a_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);
	_bncomp_add_mpfmatrix_partial(mat_tmp_a[0], mat_tmp_a_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);

	// B11 + B22
	mat_br_index[0][0] = 0;
	mat_br_index[0][1] = mid_dim_h;
	mat_br_index[0][2] = 0;
	mat_br_index[0][3] = col_dim_h;

	mat_bl_index[0][0] = mid_dim_h;
	mat_bl_index[0][1] = mid_dim;
	mat_bl_index[0][2] = col_dim_h;
	mat_bl_index[0][3] = col_dim;
//	printf("P1b:\n");

	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

	//add_mpfmatrix_partial(mat_tmp_b[0], mat_tmp_b_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);
	_bncomp_add_mpfmatrix_partial(mat_tmp_b[0], mat_tmp_b_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);

	// P1 = tmp_a * tmp_b
//	printf("P1:\n");
	_bncomp_mul_mpfmatrix_strassen(mat_p[0], mat_tmp_a[0], mat_tmp_b[0], min_dim);

	// -------------------------------
	// P2 := (A21 + A22) * B11
	// -------------------------------

	// A21 + A22
	mat_ar_index[1][0] = row_dim_h;
	mat_ar_index[1][1] = row_dim;
	mat_ar_index[1][2] = 0;
	mat_ar_index[1][3] = mid_dim_h;

	mat_al_index[1][0] = row_dim_h;
	mat_al_index[1][1] = row_dim;
	mat_al_index[1][2] = mid_dim_h;
	mat_al_index[1][3] = mid_dim;

	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

	//add_mpfmatrix_partial(mat_tmp_a[1], mat_tmp_a_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);
	_bncomp_add_mpfmatrix_partial(mat_tmp_a[1], mat_tmp_a_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);

	// B11
	mat_br_index[1][0] = 0;
	mat_br_index[1][1] = mid_dim_h;
	mat_br_index[1][2] = 0;
	mat_br_index[1][3] = col_dim_h;
	//subst_mpfmatrix_partial(mat_tmp_b[1], mat_tmp_b_index, mat_b, mat_br_index[1]);
	_bncomp_subst_mpfmatrix_partial(mat_tmp_b[1], mat_tmp_b_index, mat_b, mat_br_index[1]);

	// P2 = tmp_a * tmp_b
	//printf("P2:\n");
	_bncomp_mul_mpfmatrix_strassen(mat_p[1], mat_tmp_a[1], mat_tmp_b[1], min_dim);

	// -------------------------------
	// P3 := A11 * (B12 - B22)
	// -------------------------------

	// A11
	mat_ar_index[2][0] = 0;
	mat_ar_index[2][1] = row_dim_h;
	mat_ar_index[2][2] = 0;
	mat_ar_index[2][3] = mid_dim_h;
	//subst_mpfmatrix_partial(mat_tmp_a[2], mat_tmp_a_index, mat_a, mat_ar_index[2]);
	_bncomp_subst_mpfmatrix_partial(mat_tmp_a[2], mat_tmp_a_index, mat_a, mat_ar_index[2]);

	// B12 - B22
	mat_br_index[2][0] = 0;
	mat_br_index[2][1] = mid_dim_h;
	mat_br_index[2][2] = col_dim_h;
	mat_br_index[2][3] = col_dim;

	mat_bl_index[2][0] = mid_dim_h;
	mat_bl_index[2][1] = mid_dim;
	mat_bl_index[2][2] = col_dim_h;
	mat_bl_index[2][3] = col_dim;

	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

	//sub_mpfmatrix_partial(mat_tmp_b[2], mat_tmp_b_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);
	_bncomp_sub_mpfmatrix_partial(mat_tmp_b[2], mat_tmp_b_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);

	// P3 = tmp_a * tmp_b
	//printf("P3:\n");
	_bncomp_mul_mpfmatrix_strassen(mat_p[2], mat_tmp_a[2], mat_tmp_b[2], min_dim);

	// -------------------------------
	// P4 := A22 * (B21 - B11)
	// -------------------------------

	// A22
	mat_ar_index[3][0] = row_dim_h;
	mat_ar_index[3][1] = row_dim;
	mat_ar_index[3][2] = mid_dim_h;
	mat_ar_index[3][3] = mid_dim;

	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

	//subst_mpfmatrix_partial(mat_tmp_a[3], mat_tmp_a_index, mat_a, mat_ar_index[3]);
	_bncomp_subst_mpfmatrix_partial(mat_tmp_a[3], mat_tmp_a_index, mat_a, mat_ar_index[3]);

	// B21 - B11
	mat_br_index[3][0] = mid_dim_h;
	mat_br_index[3][1] = mid_dim;
	mat_br_index[3][2] = 0;
	mat_br_index[3][3] = col_dim_h;

	mat_bl_index[3][0] = 0;
	mat_bl_index[3][1] = mid_dim_h;
	mat_bl_index[3][2] = 0;
	mat_bl_index[3][3] = col_dim_h;
	//sub_mpfmatrix_partial(mat_tmp_b[3], mat_tmp_b_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);
	_bncomp_sub_mpfmatrix_partial(mat_tmp_b[3], mat_tmp_b_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);

	// P4 = tmp_a * tmp_b
	//printf("P4:\n");
	_bncomp_mul_mpfmatrix_strassen(mat_p[3], mat_tmp_a[3], mat_tmp_b[3], min_dim);

	// -------------------------------
	// P5 := (A11 + A12) * B22
	// -------------------------------

	// A11 + A12
	mat_ar_index[4][0] = 0;
	mat_ar_index[4][1] = row_dim_h;
	mat_ar_index[4][2] = 0;
	mat_ar_index[4][3] = mid_dim_h;

	mat_al_index[4][0] = 0;
	mat_al_index[4][1] = row_dim_h;
	mat_al_index[4][2] = mid_dim_h;
	mat_al_index[4][3] = mid_dim;

	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

	//add_mpfmatrix_partial(mat_tmp_a[4], mat_tmp_a_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);
	_bncomp_add_mpfmatrix_partial(mat_tmp_a[4], mat_tmp_a_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);

	// B22
	mat_br_index[4][0] = mid_dim_h;
	mat_br_index[4][1] = mid_dim;
	mat_br_index[4][2] = col_dim_h;
	mat_br_index[4][3] = col_dim;
	//subst_mpfmatrix_partial(mat_tmp_b[4], mat_tmp_b_index, mat_b, mat_br_index[4]);
	_bncomp_subst_mpfmatrix_partial(mat_tmp_b[4], mat_tmp_b_index, mat_b, mat_br_index[4]);

	// P5 = tmp_a * tmp_b
	//printf("P5:\n");
	_bncomp_mul_mpfmatrix_strassen(mat_p[4], mat_tmp_a[4], mat_tmp_b[4], min_dim);

	// -------------------------------
	// P6 := (A21 - A11) * (B11 + B12)
	// -------------------------------
	// A21 - A11
	mat_ar_index[5][0] = row_dim_h;
	mat_ar_index[5][1] = row_dim;
	mat_ar_index[5][2] = 0;
	mat_ar_index[5][3] = mid_dim_h;

	mat_al_index[5][0] = 0;
	mat_al_index[5][1] = row_dim_h;
	mat_al_index[5][2] = 0;
	mat_al_index[5][3] = mid_dim_h;

	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

	//sub_mpfmatrix_partial(mat_tmp_a[5], mat_tmp_a_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);
	_bncomp_sub_mpfmatrix_partial(mat_tmp_a[5], mat_tmp_a_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);

	// B11 + B12
	mat_br_index[5][0] = 0;
	mat_br_index[5][1] = mid_dim_h;
	mat_br_index[5][2] = 0;
	mat_br_index[5][3] = col_dim_h;

	mat_bl_index[5][0] = 0;
	mat_bl_index[5][1] = mid_dim_h;
	mat_bl_index[5][2] = col_dim_h;
	mat_bl_index[5][3] = col_dim;

	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

	//add_mpfmatrix_partial(mat_tmp_b[5], mat_tmp_b_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);
	_bncomp_add_mpfmatrix_partial(mat_tmp_b[5], mat_tmp_b_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);

	// P6 = tmp_a * tmp_b
	//printf("P6:\n");
	_bncomp_mul_mpfmatrix_strassen(mat_p[5], mat_tmp_a[5], mat_tmp_b[5], min_dim);

	// -------------------------------
	// P7 := (A12 - A22) * (B21 + B22)
	// -------------------------------

	// A12 - A22
	mat_ar_index[6][0] = 0;
	mat_ar_index[6][1] = row_dim_h;
	mat_ar_index[6][2] = mid_dim_h;
	mat_ar_index[6][3] = mid_dim;

	mat_al_index[6][0] = row_dim_h;
	mat_al_index[6][1] = row_dim;
	mat_al_index[6][2] = mid_dim_h;
	mat_al_index[6][3] = mid_dim;

	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

	//sub_mpfmatrix_partial(mat_tmp_a[6], mat_tmp_a_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);
	_bncomp_sub_mpfmatrix_partial(mat_tmp_a[6], mat_tmp_a_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);

	// B21 + B22
	mat_br_index[6][0] = mid_dim_h;
	mat_br_index[6][1] = mid_dim;
	mat_br_index[6][2] = 0;
	mat_br_index[6][3] = col_dim_h;

	mat_bl_index[6][0] = mid_dim_h;
	mat_bl_index[6][1] = mid_dim;
	mat_bl_index[6][2] = col_dim_h;
	mat_bl_index[6][3] = col_dim;

	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

	//add_mpfmatrix_partial(mat_tmp_b[6], mat_tmp_b_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);
	_bncomp_add_mpfmatrix_partial(mat_tmp_b[6], mat_tmp_b_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);

	// P7 = tmp_a * tmp_b
	//printf("P7:\n");
	_bncomp_mul_mpfmatrix_strassen(mat_p[6], mat_tmp_a[6], mat_tmp_b[6], min_dim);

	// -------------------------------
	// C11 := P1 + P4 - P5 + P7
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += 3 * row_dim_h * col_dim_h;

	//add_mpfmatrix(mat_tmp_c[0], mat_p[0], mat_p[3]);
	//sub_mpfmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[4]);
	//add_mpfmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[6]);
	_bncomp_add_mpfmatrix(mat_tmp_c[0], mat_p[0], mat_p[3]);
	_bncomp_sub_mpfmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[4]);
	_bncomp_add_mpfmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[6]);
	mat_c_index[0][0] = 0;
	mat_c_index[0][1] = row_dim_h;
	mat_c_index[0][2] = 0;
	mat_c_index[0][3] = col_dim_h;

	//subst_mpfmatrix_partial(ret, mat_c_index[0], mat_tmp_c[0], ret_index);
	_bncomp_subst_mpfmatrix_partial(ret, mat_c_index[0], mat_tmp_c[0], ret_index);

	// -------------------------------
	// C12 := P3 + P5
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

	//add_mpfmatrix(mat_tmp_c[1], mat_p[2], mat_p[4]);
	_bncomp_add_mpfmatrix(mat_tmp_c[1], mat_p[2], mat_p[4]);
	mat_c_index[1][0] = 0;
	mat_c_index[1][1] = row_dim_h;
	mat_c_index[1][2] = col_dim_h;
	mat_c_index[1][3] = col_dim;
	//subst_mpfmatrix_partial(ret, mat_c_index[1], mat_tmp_c[1], ret_index);
	_bncomp_subst_mpfmatrix_partial(ret, mat_c_index[1], mat_tmp_c[1], ret_index);

	// -------------------------------
	// C21 := P2 + P4
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

	//add_mpfmatrix(mat_tmp_c[2], mat_p[1], mat_p[3]);
	_bncomp_add_mpfmatrix(mat_tmp_c[2], mat_p[1], mat_p[3]);
	mat_c_index[2][0] = row_dim_h;
	mat_c_index[2][1] = row_dim;
	mat_c_index[2][2] = 0;
	mat_c_index[2][3] = col_dim_h;
	//subst_mpfmatrix_partial(ret, mat_c_index[2], mat_tmp_c[2], ret_index);
	_bncomp_subst_mpfmatrix_partial(ret, mat_c_index[2], mat_tmp_c[2], ret_index);

	// -------------------------------
	// C22 := P1 + P3 - P2 + P6
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += 3 * row_dim_h * col_dim_h;

	//add_mpfmatrix(mat_tmp_c[3], mat_p[0], mat_p[2]);
	//sub_mpfmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[1]);
	//add_mpfmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[5]);
	_bncomp_add_mpfmatrix(mat_tmp_c[3], mat_p[0], mat_p[2]);
	_bncomp_sub_mpfmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[1]);
	_bncomp_add_mpfmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[5]);
	mat_c_index[3][0] = row_dim_h;
	mat_c_index[3][1] = row_dim;
	mat_c_index[3][2] = col_dim_h;
	mat_c_index[3][3] = col_dim;
	//subst_mpfmatrix_partial(ret, mat_c_index[3], mat_tmp_c[3], ret_index);
	_bncomp_subst_mpfmatrix_partial(ret, mat_c_index[3], mat_tmp_c[3], ret_index);

	// free
	for(i = 0; i < 7; i++)
	{
		free_mpfmatrix(mat_p[i]);
		free_mpfmatrix(mat_tmp_a[i]);
		free_mpfmatrix(mat_tmp_b[i]);
	}
	for(i = 0; i < 4; i++)
		free_mpfmatrix(mat_tmp_c[i]);

	//printf("...strassen_even end\n");
}

// Strassen's Algorithm with parallelized sections
void _bncomp_mul_mpfmatrix_strassen_even_psec(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim)
{
	unsigned long prec;

	int thread_num, thread_index;
//	long int min_dim = 4; // = 2^2
	MPFMatrix mat_p[7], mat_tmp_a[7], mat_tmp_b[7], mat_tmp_c[4];
	long int row_dim_h, row_dim, col_dim_h, col_dim, mid_dim_h, mid_dim;
	long int ret_index[4], mat_tmp_a_index[4], mat_tmp_b_index[4], mat_ar_index[7][4], mat_al_index[7][4], mat_br_index[7][4], mat_bl_index[7][4], mat_c_index[4][4];
	long int i;

	// initialize
	prec = ret->prec;

	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_mpfmatrix_strassen_even)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	row_dim_h = row_dim / 2;
	col_dim_h = col_dim / 2;
	mid_dim_h = mid_dim / 2;

	//printf("mat_a: %ld, %ld\n", mat_a->row_dim, mat_a->col_dim);
	//printf("mat_b: %ld, %ld\n", mat_b->row_dim, mat_b->col_dim);
//	printf("row_dim_h, col_dim_h, mid_dim_h: %ld, %ld, %ld\n", row_dim_h, col_dim_h, mid_dim_h);

	// normal matrix multiplication in case of ret_dim <= 4 
//	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	if((ret->row_dim <= min_dim) || (ret->col_dim <= min_dim) || (mid_dim <= min_dim))
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		_bncomp_num_mul_mul_mpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		//mul_mpfmatrix(ret, mat_a, mat_b);
		_bncomp_mul_mpfmatrix_simple(ret, mat_a, mat_b);
		return;
	}

	#pragma omp parallel for
	for(i = 0; i < 7; i++)
	{
		mat_p[i] = init2_mpfmatrix(row_dim_h, col_dim_h, prec);
		mat_tmp_a[i] = init2_mpfmatrix(row_dim_h, mid_dim_h, prec);
		mat_tmp_b[i] = init2_mpfmatrix(mid_dim_h, col_dim_h, prec);
	}
	
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
		mat_tmp_c[i] = init2_mpfmatrix(row_dim_h, col_dim_h, prec);

	ret_index[0] = 0;
	ret_index[1] = row_dim_h;
	ret_index[2] = 0;
	ret_index[3] = col_dim_h;

	mat_tmp_a_index[0] = 0;
	mat_tmp_a_index[1] = row_dim_h;
	mat_tmp_a_index[2] = 0;
	mat_tmp_a_index[3] = mid_dim_h;

	mat_tmp_b_index[0] = 0;
	mat_tmp_b_index[1] = mid_dim_h;
	mat_tmp_b_index[2] = 0;
	mat_tmp_b_index[3] = col_dim_h;

//#pragma omp sections //num_threads(7)
//#pragma omp parallel sections shared(mat_tmp_a, mat_tmp_b, mat_a, mat_b, mat_br_index, mat_bl_index, mat_ar_index, mat_al_index, row_dim, col_dim, mid_dim, row_dim_h, col_dim_h, mid_dim_h, mat_tmp_a_index, mat_tmp_b_index, ret_index) //num_threads(7)
//#pragma omp parallel sections //num_threads(7)
//#pragma omp parallel sections shared(mat_tmp_a, mat_tmp_b, mat_a, mat_b, mat_br_index, mat_bl_index, mat_ar_index, mat_al_index, row_dim, col_dim, mid_dim, row_dim_h, col_dim_h, mid_dim_h, mat_tmp_a_index, mat_tmp_b_index, ret_index) //num_threads(7)
//#pragma omp parallel sections
//{
	// -------------------------------
	// P1 := (A11 + A22) * (B11 + B22)
	//--------------------------------
	//#pragma omp section
	#pragma omp task
	{

		// A11 + A22
		mat_ar_index[0][0] = 0;
		mat_ar_index[0][1] = row_dim_h;
		mat_ar_index[0][2] = 0;
		mat_ar_index[0][3] = mid_dim_h;

		mat_al_index[0][0] = row_dim_h;
		mat_al_index[0][1] = row_dim;
		mat_al_index[0][2] = mid_dim_h;
		mat_al_index[0][3] = mid_dim;
	//	printf("P1a:\n");

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		add_mpfmatrix_partial(mat_tmp_a[0], mat_tmp_a_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);
	//	_bncomp_add_mpfmatrix_partial(mat_tmp_a[0], mat_tmp_a_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);

		// B11 + B22
		mat_br_index[0][0] = 0;
		mat_br_index[0][1] = mid_dim_h;
		mat_br_index[0][2] = 0;
		mat_br_index[0][3] = col_dim_h;

		mat_bl_index[0][0] = mid_dim_h;
		mat_bl_index[0][1] = mid_dim;
		mat_bl_index[0][2] = col_dim_h;
		mat_bl_index[0][3] = col_dim;
	//	printf("P1b:\n");

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		add_mpfmatrix_partial(mat_tmp_b[0], mat_tmp_b_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);
	//	_bncomp_add_mpfmatrix_partial(mat_tmp_b[0], mat_tmp_b_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);

		// P1 = tmp_a * tmp_b
		//printf("P1: \n");
		_bncomp_mul_mpfmatrix_strassen(mat_p[0], mat_tmp_a[0], mat_tmp_b[0], min_dim);

	}

	// -------------------------------
	// P2 := (A21 + A22) * B11
	// -------------------------------
	//#pragma omp section
	#pragma omp task
	{

		// A21 + A22
		mat_ar_index[1][0] = row_dim_h;
		mat_ar_index[1][1] = row_dim;
		mat_ar_index[1][2] = 0;
		mat_ar_index[1][3] = mid_dim_h;

		mat_al_index[1][0] = row_dim_h;
		mat_al_index[1][1] = row_dim;
		mat_al_index[1][2] = mid_dim_h;
		mat_al_index[1][3] = mid_dim;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		add_mpfmatrix_partial(mat_tmp_a[1], mat_tmp_a_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);
	//	_bncomp_add_mpfmatrix_partial(mat_tmp_a[1], mat_tmp_a_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);

		// B11
		mat_br_index[1][0] = 0;
		mat_br_index[1][1] = mid_dim_h;
		mat_br_index[1][2] = 0;
		mat_br_index[1][3] = col_dim_h;
		subst_mpfmatrix_partial(mat_tmp_b[1], mat_tmp_b_index, mat_b, mat_br_index[1]);
	//	_bncomp_subst_mpfmatrix_partial(mat_tmp_b[1], mat_tmp_b_index, mat_b, mat_br_index[1]);

		// P2 = tmp_a * tmp_b
		//printf("P2: \n");
		_bncomp_mul_mpfmatrix_strassen(mat_p[1], mat_tmp_a[1], mat_tmp_b[1], min_dim);
	}

	// -------------------------------
	// P3 := A11 * (B12 - B22)
	// -------------------------------
	//#pragma omp section
	#pragma omp task
	{

		// A11
		mat_ar_index[2][0] = 0;
		mat_ar_index[2][1] = row_dim_h;
		mat_ar_index[2][2] = 0;
		mat_ar_index[2][3] = mid_dim_h;
		subst_mpfmatrix_partial(mat_tmp_a[2], mat_tmp_a_index, mat_a, mat_ar_index[2]);
	//	_bncomp_subst_mpfmatrix_partial(mat_tmp_a[2], mat_tmp_a_index, mat_a, mat_ar_index[2]);

		// B12 - B22
		mat_br_index[2][0] = 0;
		mat_br_index[2][1] = mid_dim_h;
		mat_br_index[2][2] = col_dim_h;
		mat_br_index[2][3] = col_dim;

		mat_bl_index[2][0] = mid_dim_h;
		mat_bl_index[2][1] = mid_dim;
		mat_bl_index[2][2] = col_dim_h;
		mat_bl_index[2][3] = col_dim;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		sub_mpfmatrix_partial(mat_tmp_b[2], mat_tmp_b_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);
	//	_bncomp_sub_mpfmatrix_partial(mat_tmp_b[2], mat_tmp_b_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);

		// P3 = tmp_a * tmp_b
		//printf("P3: \n");
		_bncomp_mul_mpfmatrix_strassen(mat_p[2], mat_tmp_a[2], mat_tmp_b[2], min_dim);

	}

	// -------------------------------
	// P4 := A22 * (B21 - B11)
	// -------------------------------
	//#pragma omp section
	#pragma omp task
	{

		// A22
		mat_ar_index[3][0] = row_dim_h;
		mat_ar_index[3][1] = row_dim;
		mat_ar_index[3][2] = mid_dim_h;
		mat_ar_index[3][3] = mid_dim;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		subst_mpfmatrix_partial(mat_tmp_a[3], mat_tmp_a_index, mat_a, mat_ar_index[3]);
	//	_bncomp_subst_mpfmatrix_partial(mat_tmp_a[3], mat_tmp_a_index, mat_a, mat_ar_index[3]);

		// B21 - B11
		mat_br_index[3][0] = mid_dim_h;
		mat_br_index[3][1] = mid_dim;
		mat_br_index[3][2] = 0;
		mat_br_index[3][3] = col_dim_h;

		mat_bl_index[3][0] = 0;
		mat_bl_index[3][1] = mid_dim_h;
		mat_bl_index[3][2] = 0;
		mat_bl_index[3][3] = col_dim_h;
		sub_mpfmatrix_partial(mat_tmp_b[3], mat_tmp_b_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);
	//	_bncomp_sub_mpfmatrix_partial(mat_tmp_b[3], mat_tmp_b_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);

		// P4 = tmp_a * tmp_b
		//printf("P4: \n");
		_bncomp_mul_mpfmatrix_strassen(mat_p[3], mat_tmp_a[3], mat_tmp_b[3], min_dim);

	}

	// -------------------------------
	// P5 := (A11 + A12) * B22
	// -------------------------------
	//#pragma omp section
	#pragma omp task
	{

		// A11 + A12
		mat_ar_index[4][0] = 0;
		mat_ar_index[4][1] = row_dim_h;
		mat_ar_index[4][2] = 0;
		mat_ar_index[4][3] = mid_dim_h;

		mat_al_index[4][0] = 0;
		mat_al_index[4][1] = row_dim_h;
		mat_al_index[4][2] = mid_dim_h;
		mat_al_index[4][3] = mid_dim;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		add_mpfmatrix_partial(mat_tmp_a[4], mat_tmp_a_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);
	//	_bncomp_add_mpfmatrix_partial(mat_tmp_a[4], mat_tmp_a_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);

		// B22
		mat_br_index[4][0] = mid_dim_h;
		mat_br_index[4][1] = mid_dim;
		mat_br_index[4][2] = col_dim_h;
		mat_br_index[4][3] = col_dim;
		subst_mpfmatrix_partial(mat_tmp_b[4], mat_tmp_b_index, mat_b, mat_br_index[4]);
	//	_bncomp_subst_mpfmatrix_partial(mat_tmp_b[4], mat_tmp_b_index, mat_b, mat_br_index[4]);

		// P5 = tmp_a * tmp_b
		//printf("P5: \n");
		_bncomp_mul_mpfmatrix_strassen(mat_p[4], mat_tmp_a[4], mat_tmp_b[4], min_dim);
	}

	// -------------------------------
	// P6 := (A21 - A11) * (B11 + B12)
	// -------------------------------
	//#pragma omp section
	#pragma omp task
	{

		// A21 - A11
		mat_ar_index[5][0] = row_dim_h;
		mat_ar_index[5][1] = row_dim;
		mat_ar_index[5][2] = 0;
		mat_ar_index[5][3] = mid_dim_h;

		mat_al_index[5][0] = 0;
		mat_al_index[5][1] = row_dim_h;
		mat_al_index[5][2] = 0;
		mat_al_index[5][3] = mid_dim_h;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		sub_mpfmatrix_partial(mat_tmp_a[5], mat_tmp_a_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);
	//	_bncomp_sub_mpfmatrix_partial(mat_tmp_a[5], mat_tmp_a_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);

		// B11 + B12
		mat_br_index[5][0] = 0;
		mat_br_index[5][1] = mid_dim_h;
		mat_br_index[5][2] = 0;
		mat_br_index[5][3] = col_dim_h;

		mat_bl_index[5][0] = 0;
		mat_bl_index[5][1] = mid_dim_h;
		mat_bl_index[5][2] = col_dim_h;
		mat_bl_index[5][3] = col_dim;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		add_mpfmatrix_partial(mat_tmp_b[5], mat_tmp_b_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);
	//	_bncomp_add_mpfmatrix_partial(mat_tmp_b[5], mat_tmp_b_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);

		// P6 = tmp_a * tmp_b
		//printf("P6: \n");
		_bncomp_mul_mpfmatrix_strassen(mat_p[5], mat_tmp_a[5], mat_tmp_b[5], min_dim);

	}

	// -------------------------------
	// P7 := (A12 - A22) * (B21 + B22)
	// -------------------------------
	//#pragma omp section
	#pragma omp task
	{

		// A12 - A22
		mat_ar_index[6][0] = 0;
		mat_ar_index[6][1] = row_dim_h;
		mat_ar_index[6][2] = mid_dim_h;
		mat_ar_index[6][3] = mid_dim;

		mat_al_index[6][0] = row_dim_h;
		mat_al_index[6][1] = row_dim;
		mat_al_index[6][2] = mid_dim_h;
		mat_al_index[6][3] = mid_dim;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		sub_mpfmatrix_partial(mat_tmp_a[6], mat_tmp_a_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);
	//	_bncomp_sub_mpfmatrix_partial(mat_tmp_a[6], mat_tmp_a_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);

		// B21 + B22
		mat_br_index[6][0] = mid_dim_h;
		mat_br_index[6][1] = mid_dim;
		mat_br_index[6][2] = 0;
		mat_br_index[6][3] = col_dim_h;

		mat_bl_index[6][0] = mid_dim_h;
		mat_bl_index[6][1] = mid_dim;
		mat_bl_index[6][2] = col_dim_h;
		mat_bl_index[6][3] = col_dim;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		add_mpfmatrix_partial(mat_tmp_b[6], mat_tmp_b_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);
	//	_bncomp_add_mpfmatrix_partial(mat_tmp_b[6], mat_tmp_b_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);

		// P7 = tmp_a * tmp_b
		//printf("P7: \n");
		_bncomp_mul_mpfmatrix_strassen(mat_p[6], mat_tmp_a[6], mat_tmp_b[6], min_dim);
	}

//} // pragma omp parallel sections

#pragma omp taskwait


//#pragma omp sections
#pragma omp parallel sections
{
	// -------------------------------
	// C11 := P1 + P4 - P5 + P7
	// -------------------------------
	#pragma omp section
	{

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += 3 * row_dim_h * col_dim_h;

		add_mpfmatrix(mat_tmp_c[0], mat_p[0], mat_p[3]);
		sub_mpfmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[4]);
		add_mpfmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[6]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[0], mat_p[0], mat_p[3]);
		//_bncomp_sub_mpfmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[4]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[6]);
		mat_c_index[0][0] = 0;
		mat_c_index[0][1] = row_dim_h;
		mat_c_index[0][2] = 0;
		mat_c_index[0][3] = col_dim_h;

		subst_mpfmatrix_partial(ret, mat_c_index[0], mat_tmp_c[0], ret_index);
		//_bncomp_subst_mpfmatrix_partial(ret, mat_c_index[0], mat_tmp_c[0], ret_index);
	}

	// -------------------------------
	// C12 := P3 + P5
	// -------------------------------
	#pragma omp section
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

		add_mpfmatrix(mat_tmp_c[1], mat_p[2], mat_p[4]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[1], mat_p[2], mat_p[4]);
		mat_c_index[1][0] = 0;
		mat_c_index[1][1] = row_dim_h;
		mat_c_index[1][2] = col_dim_h;
		mat_c_index[1][3] = col_dim;
		subst_mpfmatrix_partial(ret, mat_c_index[1], mat_tmp_c[1], ret_index);
		//_bncomp_subst_mpfmatrix_partial(ret, mat_c_index[1], mat_tmp_c[1], ret_index);
	}

	// -------------------------------
	// C21 := P2 + P4
	// -------------------------------
	#pragma omp section
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

		add_mpfmatrix(mat_tmp_c[2], mat_p[1], mat_p[3]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[2], mat_p[1], mat_p[3]);
		mat_c_index[2][0] = row_dim_h;
		mat_c_index[2][1] = row_dim;
		mat_c_index[2][2] = 0;
		mat_c_index[2][3] = col_dim_h;
		subst_mpfmatrix_partial(ret, mat_c_index[2], mat_tmp_c[2], ret_index);
		//_bncomp_subst_mpfmatrix_partial(ret, mat_c_index[2], mat_tmp_c[2], ret_index);
	}

	// -------------------------------
	// C22 := P1 + P3 - P2 + P6
	// -------------------------------
	#pragma omp section
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += 3 * row_dim_h * col_dim_h;

		add_mpfmatrix(mat_tmp_c[3], mat_p[0], mat_p[2]);
		sub_mpfmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[1]);
		add_mpfmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[5]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[3], mat_p[0], mat_p[2]);
		//_bncomp_sub_mpfmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[1]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[5]);
		mat_c_index[3][0] = row_dim_h;
		mat_c_index[3][1] = row_dim;
		mat_c_index[3][2] = col_dim_h;
		mat_c_index[3][3] = col_dim;
		subst_mpfmatrix_partial(ret, mat_c_index[3], mat_tmp_c[3], ret_index);
		//_bncomp_subst_mpfmatrix_partial(ret, mat_c_index[3], mat_tmp_c[3], ret_index);
	}

} // pragma omp parallel sections

	// free
	#pragma omp parallel for
	for(i = 0; i < 7; i++)
	{
		free_mpfmatrix(mat_p[i]);
		free_mpfmatrix(mat_tmp_a[i]);
		free_mpfmatrix(mat_tmp_b[i]);
	}

	#pragma omp parallel for
	for(i = 0; i < 4; i++)
		free_mpfmatrix(mat_tmp_c[i]);
}


// Winograd Variant of Strassen's Algorithm
void _bncomp_mul_mpfmatrix_winograd_even(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim)
{
//	long int min_dim = 4; // = 2^2
	unsigned long prec;
	MPFMatrix mat_s[8], mat_m[7], mat_t[2], mat_tmp_a[4], mat_tmp_b[4], mat_tmp_c[4];
	long int row_dim_h, row_dim, col_dim_h, col_dim, mid_dim, mid_dim_h;
	long int ret_index[4], mat_tmp_a_index[4], mat_tmp_b_index[4];
	long int mat_a_index[4][4], mat_b_index[4][4], mat_c_index[4][4];
	long int i;

	// initialize
	prec = ret->prec;
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_mpfmatrix_winograd_even)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	row_dim_h = row_dim / 2;
	col_dim_h = col_dim / 2;
	mid_dim_h = mid_dim / 2;

	// normal matrix multiplication in case of ret_dim <= 4 
//	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	if((ret->row_dim <= min_dim) || (ret->col_dim <= min_dim) || (mid_dim <= min_dim))
	{
		_bncomp_mul_mpfmatrix_simple(ret, mat_a, mat_b);

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		_bncomp_num_mul_mul_mpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		return;
	}

	for(i = 0; i < 4; i++)
	{
		mat_s[i] = init2_mpfmatrix(row_dim_h, mid_dim_h, prec);
		mat_tmp_a[i] = init2_mpfmatrix(row_dim_h, mid_dim_h, prec);

		mat_s[i + 4] = init2_mpfmatrix(mid_dim_h, col_dim_h, prec);
		mat_tmp_b[i] = init2_mpfmatrix(mid_dim_h, col_dim_h, prec);

		mat_tmp_c[i] = init2_mpfmatrix(row_dim_h, col_dim_h, prec);
	}
	for(i = 0; i < 7; i++)
		mat_m[i] = init2_mpfmatrix(row_dim_h, col_dim_h, prec);

	mat_t[0] = init2_mpfmatrix(row_dim_h, col_dim_h, prec);
	mat_t[1] = init2_mpfmatrix(row_dim_h, col_dim_h, prec);

	ret_index[0] = 0;
	ret_index[1] = row_dim_h;
	ret_index[2] = 0;
	ret_index[3] = col_dim_h;

	// indeces for A
	mat_a_index[0][0] = 0        ; mat_a_index[0][1] = row_dim_h; mat_a_index[0][2] = 0        ; mat_a_index[0][3] = mid_dim_h;
	mat_a_index[1][0] = 0        ; mat_a_index[1][1] = row_dim_h; mat_a_index[1][2] = mid_dim_h; mat_a_index[1][3] = mid_dim;
	mat_a_index[2][0] = row_dim_h; mat_a_index[2][1] = row_dim  ; mat_a_index[2][2] = 0        ; mat_a_index[2][3] = mid_dim_h;
	mat_a_index[3][0] = row_dim_h; mat_a_index[3][1] = row_dim  ; mat_a_index[3][2] = mid_dim_h; mat_a_index[3][3] = mid_dim;

	// indeces for B
	mat_b_index[0][0] = 0        ; mat_b_index[0][1] = mid_dim_h; mat_b_index[0][2] = 0        ; mat_b_index[0][3] = col_dim_h;
	mat_b_index[1][0] = 0        ; mat_b_index[1][1] = mid_dim_h; mat_b_index[1][2] = col_dim_h; mat_b_index[1][3] = col_dim;
	mat_b_index[2][0] = mid_dim_h; mat_b_index[2][1] = mid_dim  ; mat_b_index[2][2] = 0        ; mat_b_index[2][3] = col_dim_h;
	mat_b_index[3][0] = mid_dim_h; mat_b_index[3][1] = mid_dim  ; mat_b_index[3][2] = col_dim_h; mat_b_index[3][3] = col_dim;

	// indeces for C
	mat_c_index[0][0] = 0        ; mat_c_index[0][1] = row_dim_h; mat_c_index[0][2] = 0        ; mat_c_index[0][3] = col_dim_h;
	mat_c_index[1][0] = 0        ; mat_c_index[1][1] = row_dim_h; mat_c_index[1][2] = col_dim_h; mat_c_index[1][3] = col_dim;
	mat_c_index[2][0] = row_dim_h; mat_c_index[2][1] = row_dim  ; mat_c_index[2][2] = 0        ; mat_c_index[2][3] = col_dim_h;
	mat_c_index[3][0] = row_dim_h; mat_c_index[3][1] = row_dim  ; mat_c_index[3][2] = col_dim_h; mat_c_index[3][3] = col_dim;

	mat_tmp_a_index[0] = 0;
	mat_tmp_a_index[1] = row_dim_h;
	mat_tmp_a_index[2] = 0;
	mat_tmp_a_index[3] = mid_dim_h;

	mat_tmp_b_index[0] = 0;
	mat_tmp_b_index[1] = mid_dim_h;
	mat_tmp_b_index[2] = 0;
	mat_tmp_b_index[3] = col_dim_h;

	// mat_tmp_a[0], [1], [2], [3] := A11, A12, A21, A22
	// mat_tmp_b[0], [1], [2], [3] := B11, B12, B21, B22
	for(i = 0; i < 4; i++)
	{
		_bncomp_subst_mpfmatrix_partial(mat_tmp_a[i], mat_tmp_a_index, mat_a, mat_a_index[i]);
		_bncomp_subst_mpfmatrix_partial(mat_tmp_b[i], mat_tmp_b_index, mat_b, mat_b_index[i]);
	}
//	printf("subst a, b...\n");

	// -------------------------------
	// S1 := A21 + A22
	//--------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

	_bncomp_add_mpfmatrix(mat_s[0], mat_tmp_a[2], mat_tmp_a[3]);

	// -------------------------------
	// S2 := S1 - A11
	//--------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

	_bncomp_sub_mpfmatrix(mat_s[1], mat_s[0], mat_tmp_a[0]);

	// -------------------------------
	// S3 := A11 - A21
	//--------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

	_bncomp_sub_mpfmatrix(mat_s[2], mat_tmp_a[0], mat_tmp_a[2]);

	// -------------------------------
	// S4 := A12 - S2
	//--------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

	_bncomp_sub_mpfmatrix(mat_s[3], mat_tmp_a[1], mat_s[1]);

	// -------------------------------
	// S5 := B12 - B11
	//--------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

	_bncomp_sub_mpfmatrix(mat_s[4], mat_tmp_b[1], mat_tmp_b[0]);

	// -------------------------------
	// S6 := B22 - S5
	//--------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

	_bncomp_sub_mpfmatrix(mat_s[5], mat_tmp_b[3], mat_s[4]);

	// -------------------------------
	// S7 := B22 - B12
	//--------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

	_bncomp_sub_mpfmatrix(mat_s[6], mat_tmp_b[3], mat_tmp_b[1]);

	// -------------------------------
	// S8 := S6 - B21
	//--------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

	_bncomp_sub_mpfmatrix(mat_s[7], mat_s[5], mat_tmp_b[2]);

//	printf("s...\n");

	// -------------------------------
	// M1 := S2 * S6
	// -------------------------------
	_bncomp_mul_mpfmatrix_strassen(mat_m[0], mat_s[1], mat_s[5], min_dim);

	// -------------------------------
	// M2 := A11 * B11
	// -------------------------------
	_bncomp_mul_mpfmatrix_strassen(mat_m[1], mat_tmp_a[0], mat_tmp_b[0], min_dim);

	// -------------------------------
	// M3 := A12 * B21
	// -------------------------------
	_bncomp_mul_mpfmatrix_strassen(mat_m[2], mat_tmp_a[1], mat_tmp_b[2], min_dim);

	// -------------------------------
	// M4 := S3 * S7
	// -------------------------------
	_bncomp_mul_mpfmatrix_strassen(mat_m[3], mat_s[2], mat_s[6], min_dim);

	// -------------------------------
	// M5 := S1 * S5
	// -------------------------------
	_bncomp_mul_mpfmatrix_strassen(mat_m[4], mat_s[0], mat_s[4], min_dim);

	// -------------------------------
	// M6 := S4 * B22
	// -------------------------------
	_bncomp_mul_mpfmatrix_strassen(mat_m[5], mat_s[3], mat_tmp_b[3], min_dim);

	// -------------------------------
	// M7 := A22 * S8
	// -------------------------------
	_bncomp_mul_mpfmatrix_strassen(mat_m[6], mat_tmp_a[3], mat_s[7], min_dim);

//	printf("m...\n");

	// -------------------------------
	// T1 := M1 + M2
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

	_bncomp_add_mpfmatrix(mat_t[0], mat_m[0], mat_m[1]);

	// -------------------------------
	// T2 := T1 + M4
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

	_bncomp_add_mpfmatrix(mat_t[1], mat_t[0], mat_m[3]);

//	printf("t...\n");

	// -------------------------------
	// C11 := M2 + M3
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

	_bncomp_add_mpfmatrix(mat_tmp_c[0], mat_m[1], mat_m[2]);

	// -------------------------------
	// C12 := T1 + M5 + M6
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += 2 * row_dim_h * col_dim_h;

	_bncomp_add_mpfmatrix(mat_tmp_c[1], mat_t[0], mat_m[4]);
	_bncomp_add_mpfmatrix(mat_tmp_c[1], mat_tmp_c[1], mat_m[5]);

	// -------------------------------
	// C21 := T2 - M7
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

	_bncomp_sub_mpfmatrix(mat_tmp_c[2], mat_t[1], mat_m[6]);

	// -------------------------------
	// C22 := T2 + M5
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

	_bncomp_add_mpfmatrix(mat_tmp_c[3], mat_t[1], mat_m[4]);

//	printf("c...\n");

	// -------------------------------
	// RET := [C11 C12]
	//        [C21 C22]
	// -------------------------------
	for(i = 0; i < 4; i++)
		_bncomp_subst_mpfmatrix_partial(ret, mat_c_index[i], mat_tmp_c[i], ret_index);

//	printf("set...\n");


	// free
	for(i = 0; i < 4; i++)
	{
		free_mpfmatrix(mat_s[i]);
		free_mpfmatrix(mat_tmp_a[i]);
		free_mpfmatrix(mat_s[i + 4]);
		free_mpfmatrix(mat_tmp_b[i]);
		free_mpfmatrix(mat_tmp_c[i]);
	}
	for(i = 0; i < 7; i++)
		free_mpfmatrix(mat_m[i]);
	
	free_mpfmatrix(mat_t[0]);
	free_mpfmatrix(mat_t[1]);
}

// Winograd Variant of Strassen's Algorithm with parallelized sections
void _bncomp_mul_mpfmatrix_winograd_even_psec(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim)
{
	unsigned long prec;

	int thread_num, thread_index;
//	long int min_dim = 4; // = 2^2
	MPFMatrix mat_s[8], mat_m[7], mat_t[2], mat_tmp_a[4], mat_tmp_b[4], mat_tmp_c[4];
	long int row_dim_h, row_dim, col_dim_h, col_dim, mid_dim, mid_dim_h;
	long int ret_index[4], mat_tmp_a_index[4], mat_tmp_b_index[4];
	long int mat_a_index[4][4], mat_b_index[4][4], mat_c_index[4][4];
	long int i;

	// initialize
	prec = ret->prec;

	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_mpfmatrix_winograd_even)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	row_dim_h = row_dim / 2;
	col_dim_h = col_dim / 2;
	mid_dim_h = mid_dim / 2;

	// normal matrix multiplication in case of ret_dim <= 4 
//	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	if((ret->row_dim <= min_dim) || (ret->col_dim <= min_dim) || (mid_dim <= min_dim))
	{
		mul_mpfmatrix_simple(ret, mat_a, mat_b);
		//_bncomp_mul_mpfmatrix_simple(ret, mat_a, mat_b);

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		_bncomp_num_mul_mul_mpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		return;
	}

	#pragma omp parallel for
	for(i = 0; i < 4; i++)
	{
		mat_s[i] = init2_mpfmatrix(row_dim_h, mid_dim_h, prec);
		mat_tmp_a[i] = init2_mpfmatrix(row_dim_h, mid_dim_h, prec);

		mat_s[i + 4] = init2_mpfmatrix(mid_dim_h, col_dim_h, prec);
		mat_tmp_b[i] = init2_mpfmatrix(mid_dim_h, col_dim_h, prec);

		mat_tmp_c[i] = init2_mpfmatrix(row_dim_h, col_dim_h, prec);
	}
	#pragma omp parallel for
	for(i = 0; i < 7; i++)
		mat_m[i] = init2_mpfmatrix(row_dim_h, col_dim_h, prec);

	mat_t[0] = init2_mpfmatrix(row_dim_h, col_dim_h, prec);
	mat_t[1] = init2_mpfmatrix(row_dim_h, col_dim_h, prec);

	ret_index[0] = 0;
	ret_index[1] = row_dim_h;
	ret_index[2] = 0;
	ret_index[3] = col_dim_h;

	// indeces for A
	mat_a_index[0][0] = 0        ; mat_a_index[0][1] = row_dim_h; mat_a_index[0][2] = 0        ; mat_a_index[0][3] = mid_dim_h;
	mat_a_index[1][0] = 0        ; mat_a_index[1][1] = row_dim_h; mat_a_index[1][2] = mid_dim_h; mat_a_index[1][3] = mid_dim;
	mat_a_index[2][0] = row_dim_h; mat_a_index[2][1] = row_dim  ; mat_a_index[2][2] = 0        ; mat_a_index[2][3] = mid_dim_h;
	mat_a_index[3][0] = row_dim_h; mat_a_index[3][1] = row_dim  ; mat_a_index[3][2] = mid_dim_h; mat_a_index[3][3] = mid_dim;

	// indeces for B
	mat_b_index[0][0] = 0        ; mat_b_index[0][1] = mid_dim_h; mat_b_index[0][2] = 0        ; mat_b_index[0][3] = col_dim_h;
	mat_b_index[1][0] = 0        ; mat_b_index[1][1] = mid_dim_h; mat_b_index[1][2] = col_dim_h; mat_b_index[1][3] = col_dim;
	mat_b_index[2][0] = mid_dim_h; mat_b_index[2][1] = mid_dim  ; mat_b_index[2][2] = 0        ; mat_b_index[2][3] = col_dim_h;
	mat_b_index[3][0] = mid_dim_h; mat_b_index[3][1] = mid_dim  ; mat_b_index[3][2] = col_dim_h; mat_b_index[3][3] = col_dim;

	// indeces for C
	mat_c_index[0][0] = 0        ; mat_c_index[0][1] = row_dim_h; mat_c_index[0][2] = 0        ; mat_c_index[0][3] = col_dim_h;
	mat_c_index[1][0] = 0        ; mat_c_index[1][1] = row_dim_h; mat_c_index[1][2] = col_dim_h; mat_c_index[1][3] = col_dim;
	mat_c_index[2][0] = row_dim_h; mat_c_index[2][1] = row_dim  ; mat_c_index[2][2] = 0        ; mat_c_index[2][3] = col_dim_h;
	mat_c_index[3][0] = row_dim_h; mat_c_index[3][1] = row_dim  ; mat_c_index[3][2] = col_dim_h; mat_c_index[3][3] = col_dim;

	mat_tmp_a_index[0] = 0;
	mat_tmp_a_index[1] = row_dim_h;
	mat_tmp_a_index[2] = 0;
	mat_tmp_a_index[3] = mid_dim_h;

	mat_tmp_b_index[0] = 0;
	mat_tmp_b_index[1] = mid_dim_h;
	mat_tmp_b_index[2] = 0;
	mat_tmp_b_index[3] = col_dim_h;

	// mat_tmp_a[0], [1], [2], [3] := A11, A12, A21, A22
	// mat_tmp_b[0], [1], [2], [3] := B11, B12, B21, B22
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
	{
		subst_mpfmatrix_partial(mat_tmp_a[i], mat_tmp_a_index, mat_a, mat_a_index[i]);
		subst_mpfmatrix_partial(mat_tmp_b[i], mat_tmp_b_index, mat_b, mat_b_index[i]);
		//_bncomp_subst_mpfmatrix_partial(mat_tmp_a[i], mat_tmp_a_index, mat_a, mat_a_index[i]);
		//_bncomp_subst_mpfmatrix_partial(mat_tmp_b[i], mat_tmp_b_index, mat_b, mat_b_index[i]);
	}
//	printf("subst a, b...\n");

// 4 threads
#pragma omp parallel sections
{
	#pragma omp section
	{
		// -------------------------------
		// S1 := A21 + A22
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		add_mpfmatrix(mat_s[0], mat_tmp_a[2], mat_tmp_a[3]);
		//_bncomp_add_mpfmatrix(mat_s[0], mat_tmp_a[2], mat_tmp_a[3]);
	}

	#pragma omp section
	{
		// -------------------------------
		// S3 := A11 - A21
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		sub_mpfmatrix(mat_s[2], mat_tmp_a[0], mat_tmp_a[2]);
		//_bncomp_sub_mpfmatrix(mat_s[2], mat_tmp_a[0], mat_tmp_a[2]);
	}

	#pragma omp section
	{
		// -------------------------------
		// S5 := B12 - B11
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		sub_mpfmatrix(mat_s[4], mat_tmp_b[1], mat_tmp_b[0]);
		//_bncomp_sub_mpfmatrix(mat_s[4], mat_tmp_b[1], mat_tmp_b[0]);
	}

	#pragma omp section
	{
		// -------------------------------
		// S7 := B22 - B12
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		sub_mpfmatrix(mat_s[6], mat_tmp_b[3], mat_tmp_b[1]);
		//_bncomp_sub_mpfmatrix(mat_s[6], mat_tmp_b[3], mat_tmp_b[1]);
	}
} // pragma omp parallel sections

// 2 threads
#pragma omp parallel sections
{
	#pragma omp section
	{
		// -------------------------------
		// S2 := S1 - A11
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		sub_mpfmatrix(mat_s[1], mat_s[0], mat_tmp_a[0]);
		//_bncomp_sub_mpfmatrix(mat_s[1], mat_s[0], mat_tmp_a[0]);

		// -------------------------------
		// S4 := A12 - S2
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		sub_mpfmatrix(mat_s[3], mat_tmp_a[1], mat_s[1]);
		//_bncomp_sub_mpfmatrix(mat_s[3], mat_tmp_a[1], mat_s[1]);
	}

	#pragma omp section
	{
		// -------------------------------
		// S6 := B22 - S5
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		sub_mpfmatrix(mat_s[5], mat_tmp_b[3], mat_s[4]);
		//_bncomp_sub_mpfmatrix(mat_s[5], mat_tmp_b[3], mat_s[4]);

		// -------------------------------
		// S8 := S6 - B21
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		sub_mpfmatrix(mat_s[7], mat_s[5], mat_tmp_b[2]);
		//_bncomp_sub_mpfmatrix(mat_s[7], mat_s[5], mat_tmp_b[2]);
	}

} // pragma omp parallel sections

//	printf("s...\n");

// 7 threads
//#pragma omp parallel sections
//{
	//#pragma omp section
	#pragma omp task
	{
		// -------------------------------
		// M1 := S2 * S6
		// -------------------------------
		_bncomp_mul_mpfmatrix_strassen(mat_m[0], mat_s[1], mat_s[5], min_dim);
	}

	//#pragma omp section
	#pragma omp task
	{
		// -------------------------------
		// M2 := A11 * B11
		// -------------------------------
		_bncomp_mul_mpfmatrix_strassen(mat_m[1], mat_tmp_a[0], mat_tmp_b[0], min_dim);
	}

	//#pragma omp section
	#pragma omp task
	{
		// -------------------------------
		// M3 := A12 * B21
		// -------------------------------
		_bncomp_mul_mpfmatrix_strassen(mat_m[2], mat_tmp_a[1], mat_tmp_b[2], min_dim);
	}

	//#pragma omp section
	#pragma omp task
	{
		// -------------------------------
		// M4 := S3 * S7
		// -------------------------------
		_bncomp_mul_mpfmatrix_strassen(mat_m[3], mat_s[2], mat_s[6], min_dim);
	}

	//#pragma omp section
	#pragma omp task
	{
		// -------------------------------
		// M5 := S1 * S5
		// -------------------------------
		_bncomp_mul_mpfmatrix_strassen(mat_m[4], mat_s[0], mat_s[4], min_dim);
	}

	//#pragma omp section
	#pragma omp task
	{
		// -------------------------------
		// M6 := S4 * B22
		// -------------------------------
		_bncomp_mul_mpfmatrix_strassen(mat_m[5], mat_s[3], mat_tmp_b[3], min_dim);
	}

	//#pragma omp section
	#pragma omp task
	{
		// -------------------------------
		// M7 := A22 * S8
		// -------------------------------
		_bncomp_mul_mpfmatrix_strassen(mat_m[6], mat_tmp_a[3], mat_s[7], min_dim);
	}
//} // pragma omp parallel sections

	#pragma omp taskwait

//	printf("m...\n");

	// -------------------------------
	// T1 := M1 + M2
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

	//add_mpfmatrix(mat_t[0], mat_m[0], mat_m[1]);
	_bncomp_add_mpfmatrix(mat_t[0], mat_m[0], mat_m[1]);

	// -------------------------------
	// T2 := T1 + M4
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

	//add_mpfmatrix(mat_t[1], mat_t[0], mat_m[3]);
	_bncomp_add_mpfmatrix(mat_t[1], mat_t[0], mat_m[3]);

//	printf("t...\n");

// 4 threads
#pragma omp parallel sections
{
	#pragma omp section
	{
		// -------------------------------
		// C11 := M2 + M3
		// -------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

		add_mpfmatrix(mat_tmp_c[0], mat_m[1], mat_m[2]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[0], mat_m[1], mat_m[2]);
	}

	#pragma omp section
	{
		// -------------------------------
		// C12 := T1 + M5 + M6
		// -------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += 2 * row_dim_h * col_dim_h;

		add_mpfmatrix(mat_tmp_c[1], mat_t[0], mat_m[4]);
		add_mpfmatrix(mat_tmp_c[1], mat_tmp_c[1], mat_m[5]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[1], mat_t[0], mat_m[4]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[1], mat_tmp_c[1], mat_m[5]);
	}

	#pragma omp section
	{
		// -------------------------------
		// C21 := T2 - M7
		// -------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

		sub_mpfmatrix(mat_tmp_c[2], mat_t[1], mat_m[6]);
		//_bncomp_sub_mpfmatrix(mat_tmp_c[2], mat_t[1], mat_m[6]);
	}

	#pragma omp section
	{
		// -------------------------------
		// C22 := T2 + M5
		// -------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

		add_mpfmatrix(mat_tmp_c[3], mat_t[1], mat_m[4]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[3], mat_t[1], mat_m[4]);
	}
} // pragma omp parallel sections

//	printf("c...\n");

	// -------------------------------
	// RET := [C11 C12]
	//        [C21 C22]
	// -------------------------------
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
		subst_mpfmatrix_partial(ret, mat_c_index[i], mat_tmp_c[i], ret_index);
		//_bncomp_subst_mpfmatrix_partial(ret, mat_c_index[i], mat_tmp_c[i], ret_index);

//	printf("set...\n");


	// free
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
	{
		free_mpfmatrix(mat_s[i]);
		free_mpfmatrix(mat_tmp_a[i]);
		free_mpfmatrix(mat_s[i + 4]);
		free_mpfmatrix(mat_tmp_b[i]);
		free_mpfmatrix(mat_tmp_c[i]);
	}

	#pragma omp parallel for
	for(i = 0; i < 7; i++)
		free_mpfmatrix(mat_m[i]);
	
	free_mpfmatrix(mat_t[0]);
	free_mpfmatrix(mat_t[1]);
}

// Matrix multiplicaiton with Strassen's algorithm (Nonrecursive version in the area of Strassen's algorithms)
void _bncomp_mul_mpfmatrix_strassen_nonrec(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim)
{
	unsigned long prec;

	int thread_num, thread_index;
	int row_padding_flag = 0, col_padding_flag = 0, mid_padding_flag = 0;
	long i, j, k, row_dim, col_dim, mid_dim, max_dim;
	long int small_row_dim, small_col_dim, small_mid_dim;
	long int num_div_row, num_div_col, num_div_mid;
	long int max_num_div, min_num_div, strassen_num_div, strassen_pow;
	MPFMatrix a11, a12, a21, a22;
	MPFMatrix b11, b12, b21, b22;
	MPFMatrix ret11, ret12, ret21, ret22, tmp_mat;
	long int ret_index[4], mat_a_index[4][4], mat_b_index[4][4];
	long int a11_row_dim, a11_col_dim, a11_index[4];
	long int a12_row_dim, a12_col_dim, a12_index[4];
	long int a21_row_dim, a21_col_dim, a21_index[4];
	long int a22_row_dim, a22_col_dim, a22_index[4];
	long int b11_row_dim, b11_col_dim, b11_index[4];
	long int b12_row_dim, b12_col_dim, b12_index[4];
	long int b21_row_dim, b21_col_dim, b21_index[4];
	long int b22_row_dim, b22_col_dim, b22_index[4];
	long int ret11_row_dim, ret11_col_dim, ret11_index[4];
	long int ret12_row_dim, ret12_col_dim, ret12_index[4];
	long int ret21_row_dim, ret21_col_dim, ret21_index[4];
	long int ret22_row_dim, ret22_col_dim, ret22_index[4];

	// initialize
	prec = ret->prec;

	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(_bncomp_mul_mpfmatrix_strassen_nonrec)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	// normal matrix multiplication in case of ret_dim <= 4 
	//if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	if((ret->row_dim <= min_dim) || (ret->col_dim <= min_dim) || (mid_dim <= min_dim))
	{
		_bncomp_mul_mpfmatrix_simple(ret, mat_a, mat_b);
		return;
	}

	// size(A) > size(B) ?
	if(row_dim < col_dim) // size(A) < size(B)
	{
		if(row_dim > mid_dim)
		{
			small_row_dim = min_dim;
			small_col_dim = small_row_dim;
			small_mid_dim = (long int)floor((double)min_dim * ((double)mid_dim / (double)row_dim));
			if(small_mid_dim < 2)
				small_mid_dim = 2;
		}
		else
		{
			small_row_dim = (long int)floor((double)min_dim * ((double)row_dim / (double)mid_dim));
			small_mid_dim = min_dim;
			if(small_row_dim < 2)
				small_row_dim = 2;
			small_col_dim = small_row_dim;
		}
	}
	else  // size(A) > size(B)
	{
		if(mid_dim > col_dim)
		{
			small_mid_dim = min_dim;
			small_col_dim = (long int)floor((double)min_dim * ((double)col_dim / (double)mid_dim));
			if(small_col_dim < 2)
				small_col_dim = 2;
			small_row_dim = small_col_dim;
		}
		else
		{
			small_mid_dim = (long int)floor((double)min_dim * ((double)mid_dim / (double)col_dim));
			small_col_dim = min_dim;
			small_row_dim = small_col_dim;
			if(small_mid_dim < 2)
				small_mid_dim = 2;
		}
	}

	// Number of division of matrix
	num_div_row = (ret->row_dim) / small_row_dim;
	if((ret->row_dim % small_row_dim) >= 1)
	{
		row_padding_flag = 1;
		num_div_row++;
	}

	num_div_mid = mid_dim / small_mid_dim;
	if((mid_dim % small_mid_dim) >= 1)
	{
		mid_padding_flag = 1;
		num_div_mid++;
	}

	num_div_col = (ret->col_dim) / small_col_dim;
	if((ret->col_dim % small_col_dim) >= 1)
	{
		col_padding_flag = 1;
		num_div_col++;
	}

	// max_num_div = max(num_div_row, num_div_col, num_div_mid);
	// mix_num_div = mix(num_div_row, num_div_col, num_div_mid);
	max_num_div = (num_div_row > num_div_col) ? num_div_row : num_div_col;
	max_num_div = (max_num_div > num_div_mid) ? max_num_div : num_div_mid;
	min_num_div = (num_div_row < num_div_col) ? num_div_row : num_div_col;
	min_num_div = (min_num_div < num_div_mid) ? min_num_div : num_div_mid;

	// 2^strasse_num_div indicate dimensions of area to use strassen's algorithm
	strassen_pow = (long int)mylog2(min_num_div);
	strassen_num_div = (long int)pow(2.0, (double)strassen_pow);

	//print
/*	printf("row_dim, mid_dim, col_dim = %ld, %ld, %ld\n", row_dim, mid_dim, col_dim);
	printf("small_row_dim, small_mid_dim, small_col_dim = %ld, %ld, %ld\n", small_row_dim, small_mid_dim, small_col_dim);
	printf("num_div_row, num_div_mid, num_div_col = %ld, %ld, %ld\n", num_div_row, num_div_mid, num_div_col);
	printf("min_num_div, strassen_pow, strassen_num_div = %ld, %ld, %ld\n", min_num_div, strassen_pow, strassen_num_div);
*/
	// use block algorithm
	if(strassen_num_div < 2)
	{
		//_bncomp_mul_mpfmatrix_block(ret, mat_a, mat_b, min_dim);
		_bncomp_mul_mpfmatrix_simple(ret, mat_a, mat_b);
		return;
	}

	// separate A -> [A11 A12]
	//               [A21 A22]
	a11_row_dim = strassen_num_div * small_row_dim;
	a11_col_dim = strassen_num_div * small_mid_dim;

	a12_row_dim = strassen_num_div * small_row_dim;
	a12_col_dim = (num_div_mid - strassen_num_div) * small_mid_dim;

	a21_row_dim = (num_div_row - strassen_num_div) * small_row_dim;
	a21_col_dim = strassen_num_div * small_mid_dim;

	a22_row_dim = (num_div_row - strassen_num_div) * small_row_dim;
	a22_col_dim = (num_div_mid - strassen_num_div) * small_mid_dim;

	// Strassen Area
	a11 = init2_mpfmatrix(a11_row_dim, a11_col_dim, prec);
	a11_index[0] = mat_a_index[0][0] = 0;
	a11_index[1] = mat_a_index[0][1] = a11_row_dim;
	a11_index[2] = mat_a_index[0][2] = 0;
	a11_index[3] = mat_a_index[0][3] = a11_col_dim;
	_bncomp_subst_mpfmatrix_partial_checked(a11, a11_index, mat_a, mat_a_index[0]);

	a12 = NULL;
	if(a12_col_dim > 0)
	{
		a12 = init2_mpfmatrix(a12_row_dim, a12_col_dim, prec);
		a12_index[0] = 0;
		a12_index[1] = a12_row_dim;
		a12_index[2] = 0;
		a12_index[3] = a12_col_dim;
		mat_a_index[1][0] = 0;
		mat_a_index[1][1] = a12_row_dim;
		mat_a_index[1][2] = a11_col_dim;
		mat_a_index[1][3] = a11_col_dim + a12_col_dim;
		_bncomp_subst_mpfmatrix_partial_checked(a12, a12_index, mat_a, mat_a_index[1]);
	}

	a21 = NULL;
	if(a21_row_dim > 0)
	{
		a21 = init2_mpfmatrix(a21_row_dim, a21_col_dim, prec);
		a21_index[0] = 0;
		a21_index[1] = a21_row_dim;
		a21_index[2] = 0;
		a21_index[3] = a21_col_dim;
		mat_a_index[2][0] = a11_row_dim;
		mat_a_index[2][1] = a11_row_dim + a21_row_dim;
		mat_a_index[2][2] = 0;
		mat_a_index[2][3] = a21_col_dim;
		_bncomp_subst_mpfmatrix_partial_checked(a21, a21_index, mat_a, mat_a_index[2]);
	}

	a22 = NULL;
	if((a22_row_dim > 0) && (a22_col_dim > 0))
	{
		a22 = init2_mpfmatrix(a22_row_dim, a22_col_dim, prec);
		a22_index[0] = 0;
		a22_index[1] = a22_row_dim;
		a22_index[2] = 0;
		a22_index[3] = a22_col_dim;
		mat_a_index[3][0] = a11_row_dim;
		mat_a_index[3][1] = a11_row_dim + a22_row_dim;
		mat_a_index[3][2] = a11_col_dim;
		mat_a_index[3][3] = a11_col_dim + a22_col_dim;
		_bncomp_subst_mpfmatrix_partial_checked(a22, a22_index, mat_a, mat_a_index[3]);
	}

	// separate B -> [B11 B12]
	//               [B21 B22]
	b11_row_dim = strassen_num_div * small_mid_dim;
	b11_col_dim = strassen_num_div * small_col_dim;

	b12_row_dim = strassen_num_div * small_mid_dim;
	b12_col_dim = (num_div_col - strassen_num_div) * small_col_dim;

	b21_row_dim = (num_div_mid - strassen_num_div) * small_mid_dim;
	b21_col_dim = strassen_num_div * small_col_dim;

	b22_row_dim = (num_div_mid - strassen_num_div) * small_mid_dim;
	b22_col_dim = (num_div_col - strassen_num_div) * small_col_dim;

	// Strassen Area
	b11 = init2_mpfmatrix(b11_row_dim, b11_col_dim, prec);
	b11_index[0] = mat_b_index[0][0] = 0;
	b11_index[1] = mat_b_index[0][1] = b11_row_dim;
	b11_index[2] = mat_b_index[0][2] = 0;
	b11_index[3] = mat_b_index[0][3] = b11_col_dim;
	_bncomp_subst_mpfmatrix_partial_checked(b11, b11_index, mat_b, mat_b_index[0]);

	b12 = NULL;
	if(b12_col_dim > 0)
	{
		b12 = init2_mpfmatrix(b12_row_dim, b12_col_dim, prec);
		b12_index[0] = 0;
		b12_index[1] = b12_row_dim;
		b12_index[2] = 0;
		b12_index[3] = b12_col_dim;
		mat_b_index[1][0] = 0;
		mat_b_index[1][1] = b12_row_dim;
		mat_b_index[1][2] = b11_col_dim;
		mat_b_index[1][3] = b11_col_dim + b12_col_dim;
		_bncomp_subst_mpfmatrix_partial_checked(b12, b12_index, mat_b, mat_b_index[1]);
	}

	b21 = NULL;
	if(b21_row_dim > 0)
	{
		b21 = init2_mpfmatrix(b21_row_dim, b21_col_dim, prec);
		b21_index[0] = 0;
		b21_index[1] = b21_row_dim;
		b21_index[2] = 0;
		b21_index[3] = b21_col_dim;
		mat_b_index[2][0] = b11_row_dim;
		mat_b_index[2][1] = b11_row_dim + b21_row_dim;
		mat_b_index[2][2] = 0;
		mat_b_index[2][3] = b21_col_dim;
		_bncomp_subst_mpfmatrix_partial_checked(b21, b21_index, mat_b, mat_b_index[2]);
	}

	b22 = NULL;
	if((b22_row_dim > 0) && (b22_col_dim > 0))
	{
		b22 = init2_mpfmatrix(b22_row_dim, b22_col_dim, prec);
		b22_index[0] = 0;
		b22_index[1] = b22_row_dim;
		b22_index[2] = 0;
		b22_index[3] = b22_col_dim;
		mat_b_index[3][0] = b11_row_dim;
		mat_b_index[3][1] = b11_row_dim + b22_row_dim;
		mat_b_index[3][2] = b11_col_dim;
		mat_b_index[3][3] = b11_col_dim + b22_col_dim;
		_bncomp_subst_mpfmatrix_partial_checked(b22, b22_index, mat_b, mat_b_index[3]);
	}	

	// collect [ret11 ret12] -> ret
	//         [ret21 ret22]

	// ret11 := a11 * b11 + a12 * b21
	//printf("ret11\n");
	ret11_row_dim = strassen_num_div * small_row_dim;
	ret11_col_dim = strassen_num_div * small_col_dim;

	ret11 = init2_mpfmatrix(ret11_row_dim, ret11_col_dim, prec);

	// Strassen Area
	// a11 * b11
#ifdef USE_WINOGRAD
//	_bncomp_mul_mpfmatrix_winograd_even(ret11, a11, b11, min_dim);
	_bncomp_mul_mpfmatrix_winograd_even2(ret11, a11, b11, min_dim);
#else
	//#pragma omp parallel
//	_bncomp_mul_mpfmatrix_strassen_even3(ret11, a11, b11, min_dim, 0);
	_bncomp_mul_mpfmatrix_strassen_even2(ret11, a11, b11, min_dim, 0);
//	_bncomp_mul_mpfmatrix_strassen_even(ret11, a11, b11, min_dim);
#endif
	//_bncomp_mul_mpfmatrix_block(ret11, a11, b11, min_dim);
	//_bncomp_mul_mpfmatrix_simple(ret11, a11, b11);
	
	// ret11 += a12 * b21
	if((a12 != NULL) && (b21 != NULL))
	{
		tmp_mat = init2_mpfmatrix(ret11_row_dim, ret11_col_dim, prec);

		//_bncomp_mul_mpfmatrix_block(tmp_mat, a12, b21, min_dim);
		_bncomp_mul_mpfmatrix_simple(tmp_mat, a12, b21);
		_bncomp_add_mpfmatrix(ret11, ret11, tmp_mat);

		free_mpfmatrix(tmp_mat);
	}

	// ret := ret11
	ret11_index[0] = ret_index[0] = 0;
	ret11_index[1] = ret_index[1] = ret11_row_dim;
	ret11_index[2] = ret_index[2] = 0;
	ret11_index[3] = ret_index[3] = ret11_col_dim;
	_bncomp_subst_mpfmatrix_partial_checked(ret, ret_index, ret11, ret11_index);

	free_mpfmatrix(ret11);

	// ret12 := a11 * b12 + a12 * b22
	//printf("ret12\n");
	ret12_row_dim = strassen_num_div * small_row_dim;
	ret12_col_dim = (num_div_col - strassen_num_div) * small_col_dim;

	ret12 = NULL;
	if(ret12_col_dim > 0)
	{
		ret12 = init2_mpfmatrix(ret12_row_dim, ret12_col_dim, prec);

		// a11 * b12
		if(b12 != NULL)
			_bncomp_mul_mpfmatrix_simple(ret12, a11, b12);
		//	_bncomp_mul_mpfmatrix_block(ret12, a11, b12, min_dim);

		// ret12 += a12 * b22
		if((a12 != NULL) && (b22 != NULL))
		{
			tmp_mat = init2_mpfmatrix(ret12_row_dim, ret12_col_dim, prec);

		//	_bncomp_mul_mpfmatrix_block(tmp_mat, a12, b22, min_dim);
			_bncomp_mul_mpfmatrix_simple(tmp_mat, a12, b22);
			_bncomp_add_mpfmatrix(ret12, ret12, tmp_mat);

			free_mpfmatrix(tmp_mat);
		}

		//ret := ret12
		ret12_index[0] = 0;
		ret12_index[1] = ret12_row_dim;
		ret12_index[2] = 0;
		ret12_index[3] = ret12_col_dim;
		ret_index[0] = 0;
		ret_index[1] = ret12_row_dim;
		ret_index[2] = ret11_col_dim;
		ret_index[3] = ret11_col_dim + ret12_col_dim;
		_bncomp_subst_mpfmatrix_partial_checked(ret, ret_index, ret12, ret12_index);

		free_mpfmatrix(ret12);
	}

	// ret21 := a21 * b11 + a22 * b21
	//printf("ret21\n");
	ret21_row_dim = (num_div_row - strassen_num_div) * small_row_dim;
	ret21_col_dim = strassen_num_div * small_col_dim;

	ret21 = NULL;
	if(ret21_row_dim > 0)
	{
		ret21 = init2_mpfmatrix(ret21_row_dim, ret21_col_dim, prec);

		// a21 * b11
		if(a21 != NULL)
			_bncomp_mul_mpfmatrix_simple(ret21, a21, b11);
		//	_bncomp_mul_mpfmatrix_block(ret21, a21, b11, min_dim);

		// ret21 += a22 * b21
		if((a22 != NULL) && (b21 != NULL))
		{
			tmp_mat = init2_mpfmatrix(ret21_row_dim, ret21_col_dim, prec);
			
			//_bncomp_mul_mpfmatrix_block(tmp_mat, a22, b21, min_dim);
			_bncomp_mul_mpfmatrix_simple(tmp_mat, a22, b21);
			_bncomp_add_mpfmatrix(ret21, ret21, tmp_mat);

			free_mpfmatrix(tmp_mat);
		}

		// ret := ret21
		ret21_index[0] = 0;
		ret21_index[1] = ret21_row_dim;
		ret21_index[2] = 0;
		ret21_index[3] = ret21_col_dim;
		ret_index[0] = ret11_row_dim;
		ret_index[1] = ret11_row_dim + ret21_row_dim;
		ret_index[2] = 0;
		ret_index[3] = ret21_col_dim;
		_bncomp_subst_mpfmatrix_partial_checked(ret, ret_index, ret21, ret21_index);

		free_mpfmatrix(ret21);
	}

	// ret22 := a21 * b12 + a22 * b22
	//printf("ret22\n");
	ret22_row_dim = (num_div_row - strassen_num_div) * small_row_dim;
	ret22_col_dim = (num_div_col - strassen_num_div) * small_col_dim;

	ret22 = NULL;
	if((ret22_row_dim > 0) && (ret22_col_dim > 0))
	{
		ret22 = init2_mpfmatrix(ret22_row_dim, ret22_col_dim, prec);

		// a21 * b12
		if((a21 != NULL) && (b12 != NULL))
			_bncomp_mul_mpfmatrix_simple(ret22, a21, b12);
		//	_bncomp_mul_mpfmatrix_block(ret22, a21, b12, min_dim);

		// ret22 += a22 * b22
		if((a22 != NULL) && (b22 != NULL))
		{
			tmp_mat = init2_mpfmatrix(ret22_row_dim, ret22_col_dim, prec);
			
			//_bncomp_mul_mpfmatrix_block(tmp_mat, a22, b22, min_dim);
			_bncomp_mul_mpfmatrix_simple(tmp_mat, a22, b22);
			_bncomp_add_mpfmatrix(ret22, ret22, tmp_mat);

			free_mpfmatrix(tmp_mat);
		}

		// ret := ret22
		ret22_index[0] = 0;
		ret22_index[1] = ret22_row_dim;
		ret22_index[2] = 0;
		ret22_index[3] = ret22_col_dim;
		ret_index[0] = ret11_row_dim;
		ret_index[1] = ret11_row_dim + ret22_row_dim;
		ret_index[2] = ret11_col_dim;
		ret_index[3] = ret11_col_dim + ret22_col_dim;
		_bncomp_subst_mpfmatrix_partial_checked(ret, ret_index, ret22, ret22_index);

		free_mpfmatrix(ret22);
	}
/*
	printf("a11_row_dim, a11_col_dim, b11_row_dim, b11_col_dim = %ld, %ld, %ld, %ld\n", a11_row_dim, a11_col_dim, b11_row_dim, b11_col_dim);
	printf("a12_row_dim, a12_col_dim, b12_row_dim, b12_col_dim = %ld, %ld, %ld, %ld\n", a12_row_dim, a12_col_dim, b12_row_dim, b12_col_dim);
	printf("a21_row_dim, a21_col_dim, b21_row_dim, b21_col_dim = %ld, %ld, %ld, %ld\n", a21_row_dim, a21_col_dim, b21_row_dim, b21_col_dim);
	printf("a22_row_dim, a22_col_dim, b22_row_dim, b22_col_dim = %ld, %ld, %ld, %ld\n", a22_row_dim, a22_col_dim, b22_row_dim, b22_col_dim);
	printf("ret11_row_dim, ret11_col_dim, ret12_row_dim, ret12_col_dim = %ld, %ld, %ld, %ld\n", ret11_row_dim, ret11_col_dim, ret12_row_dim, ret12_col_dim);
	printf("ret21_row_dim, ret21_col_dim, ret22_row_dim, ret22_col_dim = %ld, %ld, %ld, %ld\n", ret21_row_dim, ret21_col_dim, ret22_row_dim, ret22_col_dim);
*/
	// free a11, a12, a12, a22, b11, b12, b21, b22
	free_mpfmatrix(a11);
	free_mpfmatrix(b11);
	if(a12 != NULL) free_mpfmatrix(a12);
	if(a21 != NULL) free_mpfmatrix(a21);
	if(a22 != NULL) free_mpfmatrix(a22);
	if(b12 != NULL) free_mpfmatrix(b12);
	if(b21 != NULL) free_mpfmatrix(b21);
	if(b22 != NULL) free_mpfmatrix(b22);

}

// Strassen's Algorithm (parallelized sections)
void _bncomp_mul_mpfmatrix_strassen_even2(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim, long int rec_num)
{
	unsigned long prec;

	int thread_num, thread_index;
//	long int min_dim = 4; // = 2^2
	MPFMatrix mat_p[7], mat_tmp_a[7], mat_tmp_b[7], mat_tmp_c[4];
	long int row_dim_h, row_dim, col_dim_h, col_dim, mid_dim_h, mid_dim;
	long int ret_index[4], mat_tmp_a_index[4], mat_tmp_b_index[4], mat_ar_index[7][4], mat_al_index[7][4], mat_br_index[7][4], mat_bl_index[7][4], mat_c_index[4][4];
	long int i;

	// initialize
	prec = ret->prec;

	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_mpfmatrix_strassen_even)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	row_dim_h = row_dim / 2;
	col_dim_h = col_dim / 2;
	mid_dim_h = mid_dim / 2;

//	printf("mat_a: %ld, %ld\n", mat_a->row_dim, mat_a->col_dim);
//	printf("mat_b: %ld, %ld\n", mat_b->row_dim, mat_b->col_dim);
//	printf("row_dim_h, col_dim_h, mid_dim_h: %ld, %ld, %ld\n", row_dim_h, col_dim_h, mid_dim_h);

	// normal matrix multiplication in case of ret_dim <= 4 
//	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	if((ret->row_dim <= min_dim) || (ret->col_dim <= min_dim) || (mid_dim <= min_dim))
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		_bncomp_num_mul_mul_mpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		//mul_mpfmatrix(ret, mat_a, mat_b);
		_bncomp_mul_mpfmatrix_simple(ret, mat_a, mat_b);
		return;
	}

	#pragma omp parallel for
	for(i = 0; i < 7; i++)
	{
		mat_p[i] = init2_mpfmatrix(row_dim_h, col_dim_h, prec);
		mat_tmp_a[i] = init2_mpfmatrix(row_dim_h, mid_dim_h, prec);
		mat_tmp_b[i] = init2_mpfmatrix(mid_dim_h, col_dim_h, prec);
	}
	
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
		mat_tmp_c[i] = init2_mpfmatrix(row_dim_h, col_dim_h, prec);

	ret_index[0] = 0;
	ret_index[1] = row_dim_h;
	ret_index[2] = 0;
	ret_index[3] = col_dim_h;

	mat_tmp_a_index[0] = 0;
	mat_tmp_a_index[1] = row_dim_h;
	mat_tmp_a_index[2] = 0;
	mat_tmp_a_index[3] = mid_dim_h;

	mat_tmp_b_index[0] = 0;
	mat_tmp_b_index[1] = mid_dim_h;
	mat_tmp_b_index[2] = 0;
	mat_tmp_b_index[3] = col_dim_h;

//#pragma omp sections //num_threads(7)
//#pragma omp parallel sections shared(mat_tmp_a, mat_tmp_b, mat_a, mat_b, mat_br_index, mat_bl_index, mat_ar_index, mat_al_index, row_dim, col_dim, mid_dim, row_dim_h, col_dim_h, mid_dim_h, mat_tmp_a_index, mat_tmp_b_index, ret_index) //num_threads(7)
//#pragma omp sections //num_threads(7)
#pragma omp parallel sections shared(mat_tmp_a, mat_tmp_b, mat_a, mat_b, mat_br_index, mat_bl_index, mat_ar_index, mat_al_index, row_dim, col_dim, mid_dim, row_dim_h, col_dim_h, mid_dim_h, mat_tmp_a_index, mat_tmp_b_index, ret_index) //num_threads(7)
{
	// -------------------------------
	// P1 := (A11 + A22) * (B11 + B22)
	//--------------------------------
	#pragma omp section
	{

		// A11 + A22
		mat_ar_index[0][0] = 0;
		mat_ar_index[0][1] = row_dim_h;
		mat_ar_index[0][2] = 0;
		mat_ar_index[0][3] = mid_dim_h;

		mat_al_index[0][0] = row_dim_h;
		mat_al_index[0][1] = row_dim;
		mat_al_index[0][2] = mid_dim_h;
		mat_al_index[0][3] = mid_dim;
	//	printf("P1a:\n");

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		add_mpfmatrix_partial(mat_tmp_a[0], mat_tmp_a_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);
	//	_bncomp_add_mpfmatrix_partial(mat_tmp_a[0], mat_tmp_a_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);

		// B11 + B22
		mat_br_index[0][0] = 0;
		mat_br_index[0][1] = mid_dim_h;
		mat_br_index[0][2] = 0;
		mat_br_index[0][3] = col_dim_h;

		mat_bl_index[0][0] = mid_dim_h;
		mat_bl_index[0][1] = mid_dim;
		mat_bl_index[0][2] = col_dim_h;
		mat_bl_index[0][3] = col_dim;
	//	printf("P1b:\n");

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		add_mpfmatrix_partial(mat_tmp_b[0], mat_tmp_b_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);
	//	_bncomp_add_mpfmatrix_partial(mat_tmp_b[0], mat_tmp_b_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);

		// P1 = tmp_a * tmp_b
		//printf("P1: %ld, ", rec_num);
		_bncomp_mul_mpfmatrix_strassen_even2(mat_p[0], mat_tmp_a[0], mat_tmp_b[0], min_dim, rec_num + 1);

	}

	// -------------------------------
	// P2 := (A21 + A22) * B11
	// -------------------------------
	#pragma omp section
	{

		// A21 + A22
		mat_ar_index[1][0] = row_dim_h;
		mat_ar_index[1][1] = row_dim;
		mat_ar_index[1][2] = 0;
		mat_ar_index[1][3] = mid_dim_h;

		mat_al_index[1][0] = row_dim_h;
		mat_al_index[1][1] = row_dim;
		mat_al_index[1][2] = mid_dim_h;
		mat_al_index[1][3] = mid_dim;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		add_mpfmatrix_partial(mat_tmp_a[1], mat_tmp_a_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);
	//	_bncomp_add_mpfmatrix_partial(mat_tmp_a[1], mat_tmp_a_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);

		// B11
		mat_br_index[1][0] = 0;
		mat_br_index[1][1] = mid_dim_h;
		mat_br_index[1][2] = 0;
		mat_br_index[1][3] = col_dim_h;
		subst_mpfmatrix_partial(mat_tmp_b[1], mat_tmp_b_index, mat_b, mat_br_index[1]);
	//	_bncomp_subst_mpfmatrix_partial(mat_tmp_b[1], mat_tmp_b_index, mat_b, mat_br_index[1]);

		// P2 = tmp_a * tmp_b
		//printf("P2: %ld, ", rec_num);
		_bncomp_mul_mpfmatrix_strassen_even2(mat_p[1], mat_tmp_a[1], mat_tmp_b[1], min_dim, rec_num + 1);

	}

	// -------------------------------
	// P3 := A11 * (B12 - B22)
	// -------------------------------
	#pragma omp section
	{

		// A11
		mat_ar_index[2][0] = 0;
		mat_ar_index[2][1] = row_dim_h;
		mat_ar_index[2][2] = 0;
		mat_ar_index[2][3] = mid_dim_h;
		subst_mpfmatrix_partial(mat_tmp_a[2], mat_tmp_a_index, mat_a, mat_ar_index[2]);
	//	_bncomp_subst_mpfmatrix_partial(mat_tmp_a[2], mat_tmp_a_index, mat_a, mat_ar_index[2]);

		// B12 - B22
		mat_br_index[2][0] = 0;
		mat_br_index[2][1] = mid_dim_h;
		mat_br_index[2][2] = col_dim_h;
		mat_br_index[2][3] = col_dim;

		mat_bl_index[2][0] = mid_dim_h;
		mat_bl_index[2][1] = mid_dim;
		mat_bl_index[2][2] = col_dim_h;
		mat_bl_index[2][3] = col_dim;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		sub_mpfmatrix_partial(mat_tmp_b[2], mat_tmp_b_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);
	//	_bncomp_sub_mpfmatrix_partial(mat_tmp_b[2], mat_tmp_b_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);

		// P3 = tmp_a * tmp_b
		//printf("P3: %ld, ", rec_num);
		_bncomp_mul_mpfmatrix_strassen_even2(mat_p[2], mat_tmp_a[2], mat_tmp_b[2], min_dim, rec_num + 1);

	}

	// -------------------------------
	// P4 := A22 * (B21 - B11)
	// -------------------------------
	#pragma omp section
	{

		// A22
		mat_ar_index[3][0] = row_dim_h;
		mat_ar_index[3][1] = row_dim;
		mat_ar_index[3][2] = mid_dim_h;
		mat_ar_index[3][3] = mid_dim;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		subst_mpfmatrix_partial(mat_tmp_a[3], mat_tmp_a_index, mat_a, mat_ar_index[3]);
	//	_bncomp_subst_mpfmatrix_partial(mat_tmp_a[3], mat_tmp_a_index, mat_a, mat_ar_index[3]);

		// B21 - B11
		mat_br_index[3][0] = mid_dim_h;
		mat_br_index[3][1] = mid_dim;
		mat_br_index[3][2] = 0;
		mat_br_index[3][3] = col_dim_h;

		mat_bl_index[3][0] = 0;
		mat_bl_index[3][1] = mid_dim_h;
		mat_bl_index[3][2] = 0;
		mat_bl_index[3][3] = col_dim_h;
		sub_mpfmatrix_partial(mat_tmp_b[3], mat_tmp_b_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);
	//	_bncomp_sub_mpfmatrix_partial(mat_tmp_b[3], mat_tmp_b_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);

		// P4 = tmp_a * tmp_b
		//printf("P4: %ld, ", rec_num);
		_bncomp_mul_mpfmatrix_strassen_even2(mat_p[3], mat_tmp_a[3], mat_tmp_b[3], min_dim, rec_num + 1);

	}

	// -------------------------------
	// P5 := (A11 + A12) * B22
	// -------------------------------
	#pragma omp section
	{

		// A11 + A12
		mat_ar_index[4][0] = 0;
		mat_ar_index[4][1] = row_dim_h;
		mat_ar_index[4][2] = 0;
		mat_ar_index[4][3] = mid_dim_h;

		mat_al_index[4][0] = 0;
		mat_al_index[4][1] = row_dim_h;
		mat_al_index[4][2] = mid_dim_h;
		mat_al_index[4][3] = mid_dim;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		add_mpfmatrix_partial(mat_tmp_a[4], mat_tmp_a_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);
	//	_bncomp_add_mpfmatrix_partial(mat_tmp_a[4], mat_tmp_a_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);

		// B22
		mat_br_index[4][0] = mid_dim_h;
		mat_br_index[4][1] = mid_dim;
		mat_br_index[4][2] = col_dim_h;
		mat_br_index[4][3] = col_dim;
		subst_mpfmatrix_partial(mat_tmp_b[4], mat_tmp_b_index, mat_b, mat_br_index[4]);
	//	_bncomp_subst_mpfmatrix_partial(mat_tmp_b[4], mat_tmp_b_index, mat_b, mat_br_index[4]);

		// P5 = tmp_a * tmp_b
		//printf("P5: %ld, ", rec_num);
		_bncomp_mul_mpfmatrix_strassen_even2(mat_p[4], mat_tmp_a[4], mat_tmp_b[4], min_dim, rec_num + 1);

	}

	// -------------------------------
	// P6 := (A21 - A11) * (B11 + B12)
	// -------------------------------
	#pragma omp section
	{

		// A21 - A11
		mat_ar_index[5][0] = row_dim_h;
		mat_ar_index[5][1] = row_dim;
		mat_ar_index[5][2] = 0;
		mat_ar_index[5][3] = mid_dim_h;

		mat_al_index[5][0] = 0;
		mat_al_index[5][1] = row_dim_h;
		mat_al_index[5][2] = 0;
		mat_al_index[5][3] = mid_dim_h;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		sub_mpfmatrix_partial(mat_tmp_a[5], mat_tmp_a_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);
	//	_bncomp_sub_mpfmatrix_partial(mat_tmp_a[5], mat_tmp_a_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);

		// B11 + B12
		mat_br_index[5][0] = 0;
		mat_br_index[5][1] = mid_dim_h;
		mat_br_index[5][2] = 0;
		mat_br_index[5][3] = col_dim_h;

		mat_bl_index[5][0] = 0;
		mat_bl_index[5][1] = mid_dim_h;
		mat_bl_index[5][2] = col_dim_h;
		mat_bl_index[5][3] = col_dim;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		add_mpfmatrix_partial(mat_tmp_b[5], mat_tmp_b_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);
	//	_bncomp_add_mpfmatrix_partial(mat_tmp_b[5], mat_tmp_b_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);

		// P6 = tmp_a * tmp_b
		//printf("P6: %ld, ", rec_num);
		_bncomp_mul_mpfmatrix_strassen_even2(mat_p[5], mat_tmp_a[5], mat_tmp_b[5], min_dim, rec_num + 1);

	}

	// -------------------------------
	// P7 := (A12 - A22) * (B21 + B22)
	// -------------------------------
	#pragma omp section
	{

		// A12 - A22
		mat_ar_index[6][0] = 0;
		mat_ar_index[6][1] = row_dim_h;
		mat_ar_index[6][2] = mid_dim_h;
		mat_ar_index[6][3] = mid_dim;

		mat_al_index[6][0] = row_dim_h;
		mat_al_index[6][1] = row_dim;
		mat_al_index[6][2] = mid_dim_h;
		mat_al_index[6][3] = mid_dim;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		sub_mpfmatrix_partial(mat_tmp_a[6], mat_tmp_a_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);
	//	_bncomp_sub_mpfmatrix_partial(mat_tmp_a[6], mat_tmp_a_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);

		// B21 + B22
		mat_br_index[6][0] = mid_dim_h;
		mat_br_index[6][1] = mid_dim;
		mat_br_index[6][2] = 0;
		mat_br_index[6][3] = col_dim_h;

		mat_bl_index[6][0] = mid_dim_h;
		mat_bl_index[6][1] = mid_dim;
		mat_bl_index[6][2] = col_dim_h;
		mat_bl_index[6][3] = col_dim;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		add_mpfmatrix_partial(mat_tmp_b[6], mat_tmp_b_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);
	//	_bncomp_add_mpfmatrix_partial(mat_tmp_b[6], mat_tmp_b_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);

		// P7 = tmp_a * tmp_b
		//printf("P7: %ld\n", rec_num);
		_bncomp_mul_mpfmatrix_strassen_even2(mat_p[6], mat_tmp_a[6], mat_tmp_b[6], min_dim, rec_num + 1);

	}
} // pragma omp parallel sections


//#pragma omp sections
#pragma omp parallel sections
{
	// -------------------------------
	// C11 := P1 + P4 - P5 + P7
	// -------------------------------
	#pragma omp section
	{

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += 3 * row_dim_h * col_dim_h;

		add_mpfmatrix(mat_tmp_c[0], mat_p[0], mat_p[3]);
		sub_mpfmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[4]);
		add_mpfmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[6]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[0], mat_p[0], mat_p[3]);
		//_bncomp_sub_mpfmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[4]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[6]);
		mat_c_index[0][0] = 0;
		mat_c_index[0][1] = row_dim_h;
		mat_c_index[0][2] = 0;
		mat_c_index[0][3] = col_dim_h;

		subst_mpfmatrix_partial(ret, mat_c_index[0], mat_tmp_c[0], ret_index);
		//_bncomp_subst_mpfmatrix_partial(ret, mat_c_index[0], mat_tmp_c[0], ret_index);
	}

	// -------------------------------
	// C12 := P3 + P5
	// -------------------------------
	#pragma omp section
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

		add_mpfmatrix(mat_tmp_c[1], mat_p[2], mat_p[4]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[1], mat_p[2], mat_p[4]);
		mat_c_index[1][0] = 0;
		mat_c_index[1][1] = row_dim_h;
		mat_c_index[1][2] = col_dim_h;
		mat_c_index[1][3] = col_dim;
		subst_mpfmatrix_partial(ret, mat_c_index[1], mat_tmp_c[1], ret_index);
		//_bncomp_subst_mpfmatrix_partial(ret, mat_c_index[1], mat_tmp_c[1], ret_index);
	}

	// -------------------------------
	// C21 := P2 + P4
	// -------------------------------
	#pragma omp section
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

		add_mpfmatrix(mat_tmp_c[2], mat_p[1], mat_p[3]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[2], mat_p[1], mat_p[3]);
		mat_c_index[2][0] = row_dim_h;
		mat_c_index[2][1] = row_dim;
		mat_c_index[2][2] = 0;
		mat_c_index[2][3] = col_dim_h;
		subst_mpfmatrix_partial(ret, mat_c_index[2], mat_tmp_c[2], ret_index);
		//_bncomp_subst_mpfmatrix_partial(ret, mat_c_index[2], mat_tmp_c[2], ret_index);
	}

	// -------------------------------
	// C22 := P1 + P3 - P2 + P6
	// -------------------------------
	#pragma omp section
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += 3 * row_dim_h * col_dim_h;

		add_mpfmatrix(mat_tmp_c[3], mat_p[0], mat_p[2]);
		sub_mpfmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[1]);
		add_mpfmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[5]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[3], mat_p[0], mat_p[2]);
		//_bncomp_sub_mpfmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[1]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[5]);
		mat_c_index[3][0] = row_dim_h;
		mat_c_index[3][1] = row_dim;
		mat_c_index[3][2] = col_dim_h;
		mat_c_index[3][3] = col_dim;
		subst_mpfmatrix_partial(ret, mat_c_index[3], mat_tmp_c[3], ret_index);
		//_bncomp_subst_mpfmatrix_partial(ret, mat_c_index[3], mat_tmp_c[3], ret_index);
	}

} // pragma omp parallel sections

	// free
	#pragma omp parallel for
	for(i = 0; i < 7; i++)
	{
		free_mpfmatrix(mat_p[i]);
		free_mpfmatrix(mat_tmp_a[i]);
		free_mpfmatrix(mat_tmp_b[i]);
	}
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
		free_mpfmatrix(mat_tmp_c[i]);
}

// Strassen's Algorithm (parallelizable tasks)
void _bncomp_mul_mpfmatrix_strassen_even3(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim, long int rec_num)
{
	unsigned long prec;

	int thread_num, thread_index;
//	long int min_dim = 4; // = 2^2
	MPFMatrix mat_p[7], mat_tmp_a[7], mat_tmp_b[7], mat_tmp_c[4];
	long int row_dim_h, row_dim, col_dim_h, col_dim, mid_dim_h, mid_dim;
	long int ret_index[4], mat_tmp_a_index[4], mat_tmp_b_index[4], mat_ar_index[7][4], mat_al_index[7][4], mat_br_index[7][4], mat_bl_index[7][4], mat_c_index[4][4];
	long int i;

	// initialize
	prec = ret->prec;

	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_mpfmatrix_strassen_even)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	row_dim_h = row_dim / 2;
	col_dim_h = col_dim / 2;
	mid_dim_h = mid_dim / 2;

//	printf("mat_a: %ld, %ld\n", mat_a->row_dim, mat_a->col_dim);
//	printf("mat_b: %ld, %ld\n", mat_b->row_dim, mat_b->col_dim);
//	printf("row_dim_h, col_dim_h, mid_dim_h: %ld, %ld, %ld\n", row_dim_h, col_dim_h, mid_dim_h);

	// normal matrix multiplication in case of ret_dim <= 4 
//	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	if((ret->row_dim <= min_dim) || (ret->col_dim <= min_dim) || (mid_dim <= min_dim))
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		_bncomp_num_mul_mul_mpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		//mul_mpfmatrix(ret, mat_a, mat_b);
		_bncomp_mul_mpfmatrix_simple(ret, mat_a, mat_b);
		return;
	}

	#pragma omp parallel for
	for(i = 0; i < 7; i++)
	{
		mat_p[i] = init2_mpfmatrix(row_dim_h, col_dim_h, prec);
		mat_tmp_a[i] = init2_mpfmatrix(row_dim_h, mid_dim_h, prec);
		mat_tmp_b[i] = init2_mpfmatrix(mid_dim_h, col_dim_h, prec);
	}
	
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
		mat_tmp_c[i] = init2_mpfmatrix(row_dim_h, col_dim_h, prec);

	ret_index[0] = 0;
	ret_index[1] = row_dim_h;
	ret_index[2] = 0;
	ret_index[3] = col_dim_h;

	mat_tmp_a_index[0] = 0;
	mat_tmp_a_index[1] = row_dim_h;
	mat_tmp_a_index[2] = 0;
	mat_tmp_a_index[3] = mid_dim_h;

	mat_tmp_b_index[0] = 0;
	mat_tmp_b_index[1] = mid_dim_h;
	mat_tmp_b_index[2] = 0;
	mat_tmp_b_index[3] = col_dim_h;

//#pragma omp parallel
{
	// -------------------------------
	// P1 := (A11 + A22) * (B11 + B22)
	//--------------------------------
	//#pragma omp task
	{

		// A11 + A22
		mat_ar_index[0][0] = 0;
		mat_ar_index[0][1] = row_dim_h;
		mat_ar_index[0][2] = 0;
		mat_ar_index[0][3] = mid_dim_h;

		mat_al_index[0][0] = row_dim_h;
		mat_al_index[0][1] = row_dim;
		mat_al_index[0][2] = mid_dim_h;
		mat_al_index[0][3] = mid_dim;
	//	printf("P1a:\n");

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		add_mpfmatrix_partial(mat_tmp_a[0], mat_tmp_a_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);
	//	_bncomp_add_mpfmatrix_partial(mat_tmp_a[0], mat_tmp_a_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);

		// B11 + B22
		mat_br_index[0][0] = 0;
		mat_br_index[0][1] = mid_dim_h;
		mat_br_index[0][2] = 0;
		mat_br_index[0][3] = col_dim_h;

		mat_bl_index[0][0] = mid_dim_h;
		mat_bl_index[0][1] = mid_dim;
		mat_bl_index[0][2] = col_dim_h;
		mat_bl_index[0][3] = col_dim;
	//	printf("P1b:\n");

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		add_mpfmatrix_partial(mat_tmp_b[0], mat_tmp_b_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);
	//	_bncomp_add_mpfmatrix_partial(mat_tmp_b[0], mat_tmp_b_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);

		// P1 = tmp_a * tmp_b
		//printf("P1: %ld, ", rec_num);
		#pragma omp task
		_bncomp_mul_mpfmatrix_strassen_even3(mat_p[0], mat_tmp_a[0], mat_tmp_b[0], min_dim, rec_num + 1);

	}

	// -------------------------------
	// P2 := (A21 + A22) * B11
	// -------------------------------
	//#pragma omp task
	{

		// A21 + A22
		mat_ar_index[1][0] = row_dim_h;
		mat_ar_index[1][1] = row_dim;
		mat_ar_index[1][2] = 0;
		mat_ar_index[1][3] = mid_dim_h;

		mat_al_index[1][0] = row_dim_h;
		mat_al_index[1][1] = row_dim;
		mat_al_index[1][2] = mid_dim_h;
		mat_al_index[1][3] = mid_dim;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		add_mpfmatrix_partial(mat_tmp_a[1], mat_tmp_a_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);
	//	_bncomp_add_mpfmatrix_partial(mat_tmp_a[1], mat_tmp_a_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);

		// B11
		mat_br_index[1][0] = 0;
		mat_br_index[1][1] = mid_dim_h;
		mat_br_index[1][2] = 0;
		mat_br_index[1][3] = col_dim_h;
		subst_mpfmatrix_partial(mat_tmp_b[1], mat_tmp_b_index, mat_b, mat_br_index[1]);
	//	_bncomp_subst_mpfmatrix_partial(mat_tmp_b[1], mat_tmp_b_index, mat_b, mat_br_index[1]);

		// P2 = tmp_a * tmp_b
		//printf("P2: %ld, ", rec_num);
		#pragma omp task
		_bncomp_mul_mpfmatrix_strassen_even3(mat_p[1], mat_tmp_a[1], mat_tmp_b[1], min_dim, rec_num + 1);

	}

	// -------------------------------
	// P3 := A11 * (B12 - B22)
	// -------------------------------
	//#pragma omp task
	{

		// A11
		mat_ar_index[2][0] = 0;
		mat_ar_index[2][1] = row_dim_h;
		mat_ar_index[2][2] = 0;
		mat_ar_index[2][3] = mid_dim_h;
		subst_mpfmatrix_partial(mat_tmp_a[2], mat_tmp_a_index, mat_a, mat_ar_index[2]);
	//	_bncomp_subst_mpfmatrix_partial(mat_tmp_a[2], mat_tmp_a_index, mat_a, mat_ar_index[2]);

		// B12 - B22
		mat_br_index[2][0] = 0;
		mat_br_index[2][1] = mid_dim_h;
		mat_br_index[2][2] = col_dim_h;
		mat_br_index[2][3] = col_dim;

		mat_bl_index[2][0] = mid_dim_h;
		mat_bl_index[2][1] = mid_dim;
		mat_bl_index[2][2] = col_dim_h;
		mat_bl_index[2][3] = col_dim;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		sub_mpfmatrix_partial(mat_tmp_b[2], mat_tmp_b_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);
	//	_bncomp_sub_mpfmatrix_partial(mat_tmp_b[2], mat_tmp_b_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);

		// P3 = tmp_a * tmp_b
		//printf("P3: %ld, ", rec_num);
		#pragma omp task
		_bncomp_mul_mpfmatrix_strassen_even3(mat_p[2], mat_tmp_a[2], mat_tmp_b[2], min_dim, rec_num + 1);

	}

	// -------------------------------
	// P4 := A22 * (B21 - B11)
	// -------------------------------
	//#pragma omp task
	{

		// A22
		mat_ar_index[3][0] = row_dim_h;
		mat_ar_index[3][1] = row_dim;
		mat_ar_index[3][2] = mid_dim_h;
		mat_ar_index[3][3] = mid_dim;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		subst_mpfmatrix_partial(mat_tmp_a[3], mat_tmp_a_index, mat_a, mat_ar_index[3]);
	//	_bncomp_subst_mpfmatrix_partial(mat_tmp_a[3], mat_tmp_a_index, mat_a, mat_ar_index[3]);

		// B21 - B11
		mat_br_index[3][0] = mid_dim_h;
		mat_br_index[3][1] = mid_dim;
		mat_br_index[3][2] = 0;
		mat_br_index[3][3] = col_dim_h;

		mat_bl_index[3][0] = 0;
		mat_bl_index[3][1] = mid_dim_h;
		mat_bl_index[3][2] = 0;
		mat_bl_index[3][3] = col_dim_h;
		sub_mpfmatrix_partial(mat_tmp_b[3], mat_tmp_b_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);
	//	_bncomp_sub_mpfmatrix_partial(mat_tmp_b[3], mat_tmp_b_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);

		// P4 = tmp_a * tmp_b
		//printf("P4: %ld, ", rec_num);
		#pragma omp task
		_bncomp_mul_mpfmatrix_strassen_even3(mat_p[3], mat_tmp_a[3], mat_tmp_b[3], min_dim, rec_num + 1);

	}

	// -------------------------------
	// P5 := (A11 + A12) * B22
	// -------------------------------
	//#pragma omp task
	{

		// A11 + A12
		mat_ar_index[4][0] = 0;
		mat_ar_index[4][1] = row_dim_h;
		mat_ar_index[4][2] = 0;
		mat_ar_index[4][3] = mid_dim_h;

		mat_al_index[4][0] = 0;
		mat_al_index[4][1] = row_dim_h;
		mat_al_index[4][2] = mid_dim_h;
		mat_al_index[4][3] = mid_dim;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		add_mpfmatrix_partial(mat_tmp_a[4], mat_tmp_a_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);
	//	_bncomp_add_mpfmatrix_partial(mat_tmp_a[4], mat_tmp_a_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);

		// B22
		mat_br_index[4][0] = mid_dim_h;
		mat_br_index[4][1] = mid_dim;
		mat_br_index[4][2] = col_dim_h;
		mat_br_index[4][3] = col_dim;
		subst_mpfmatrix_partial(mat_tmp_b[4], mat_tmp_b_index, mat_b, mat_br_index[4]);
	//	_bncomp_subst_mpfmatrix_partial(mat_tmp_b[4], mat_tmp_b_index, mat_b, mat_br_index[4]);

		// P5 = tmp_a * tmp_b
		//printf("P5: %ld, ", rec_num);
		#pragma omp task
		_bncomp_mul_mpfmatrix_strassen_even3(mat_p[4], mat_tmp_a[4], mat_tmp_b[4], min_dim, rec_num + 1);

	}

	// -------------------------------
	// P6 := (A21 - A11) * (B11 + B12)
	// -------------------------------
	//#pragma omp task
	{

		// A21 - A11
		mat_ar_index[5][0] = row_dim_h;
		mat_ar_index[5][1] = row_dim;
		mat_ar_index[5][2] = 0;
		mat_ar_index[5][3] = mid_dim_h;

		mat_al_index[5][0] = 0;
		mat_al_index[5][1] = row_dim_h;
		mat_al_index[5][2] = 0;
		mat_al_index[5][3] = mid_dim_h;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		sub_mpfmatrix_partial(mat_tmp_a[5], mat_tmp_a_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);
	//	_bncomp_sub_mpfmatrix_partial(mat_tmp_a[5], mat_tmp_a_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);

		// B11 + B12
		mat_br_index[5][0] = 0;
		mat_br_index[5][1] = mid_dim_h;
		mat_br_index[5][2] = 0;
		mat_br_index[5][3] = col_dim_h;

		mat_bl_index[5][0] = 0;
		mat_bl_index[5][1] = mid_dim_h;
		mat_bl_index[5][2] = col_dim_h;
		mat_bl_index[5][3] = col_dim;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		add_mpfmatrix_partial(mat_tmp_b[5], mat_tmp_b_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);
	//	_bncomp_add_mpfmatrix_partial(mat_tmp_b[5], mat_tmp_b_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);

		// P6 = tmp_a * tmp_b
		//printf("P6: %ld, ", rec_num);
		#pragma omp task
		_bncomp_mul_mpfmatrix_strassen_even3(mat_p[5], mat_tmp_a[5], mat_tmp_b[5], min_dim, rec_num + 1);

	}

	// -------------------------------
	// P7 := (A12 - A22) * (B21 + B22)
	// -------------------------------
	//#pragma omp task
	{

		// A12 - A22
		mat_ar_index[6][0] = 0;
		mat_ar_index[6][1] = row_dim_h;
		mat_ar_index[6][2] = mid_dim_h;
		mat_ar_index[6][3] = mid_dim;

		mat_al_index[6][0] = row_dim_h;
		mat_al_index[6][1] = row_dim;
		mat_al_index[6][2] = mid_dim_h;
		mat_al_index[6][3] = mid_dim;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		sub_mpfmatrix_partial(mat_tmp_a[6], mat_tmp_a_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);
	//	_bncomp_sub_mpfmatrix_partial(mat_tmp_a[6], mat_tmp_a_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);

		// B21 + B22
		mat_br_index[6][0] = mid_dim_h;
		mat_br_index[6][1] = mid_dim;
		mat_br_index[6][2] = 0;
		mat_br_index[6][3] = col_dim_h;

		mat_bl_index[6][0] = mid_dim_h;
		mat_bl_index[6][1] = mid_dim;
		mat_bl_index[6][2] = col_dim_h;
		mat_bl_index[6][3] = col_dim;

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		add_mpfmatrix_partial(mat_tmp_b[6], mat_tmp_b_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);
	//	_bncomp_add_mpfmatrix_partial(mat_tmp_b[6], mat_tmp_b_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);

		// P7 = tmp_a * tmp_b
		//printf("P7: %ld\n", rec_num);
		#pragma omp task
		_bncomp_mul_mpfmatrix_strassen_even3(mat_p[6], mat_tmp_a[6], mat_tmp_b[6], min_dim, rec_num + 1);

	}
} // pragma omp parallel sections

#pragma omp taskwait

//#pragma omp sections
#pragma omp parallel sections
{
	// -------------------------------
	// C11 := P1 + P4 - P5 + P7
	// -------------------------------
	#pragma omp section
	{

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += 3 * row_dim_h * col_dim_h;

		add_mpfmatrix(mat_tmp_c[0], mat_p[0], mat_p[3]);
		sub_mpfmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[4]);
		add_mpfmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[6]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[0], mat_p[0], mat_p[3]);
		//_bncomp_sub_mpfmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[4]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[6]);
		mat_c_index[0][0] = 0;
		mat_c_index[0][1] = row_dim_h;
		mat_c_index[0][2] = 0;
		mat_c_index[0][3] = col_dim_h;

		subst_mpfmatrix_partial(ret, mat_c_index[0], mat_tmp_c[0], ret_index);
		//_bncomp_subst_mpfmatrix_partial(ret, mat_c_index[0], mat_tmp_c[0], ret_index);
	}

	// -------------------------------
	// C12 := P3 + P5
	// -------------------------------
	#pragma omp section
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

		add_mpfmatrix(mat_tmp_c[1], mat_p[2], mat_p[4]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[1], mat_p[2], mat_p[4]);
		mat_c_index[1][0] = 0;
		mat_c_index[1][1] = row_dim_h;
		mat_c_index[1][2] = col_dim_h;
		mat_c_index[1][3] = col_dim;
		subst_mpfmatrix_partial(ret, mat_c_index[1], mat_tmp_c[1], ret_index);
		//_bncomp_subst_mpfmatrix_partial(ret, mat_c_index[1], mat_tmp_c[1], ret_index);
	}

	// -------------------------------
	// C21 := P2 + P4
	// -------------------------------
	#pragma omp section
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

		add_mpfmatrix(mat_tmp_c[2], mat_p[1], mat_p[3]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[2], mat_p[1], mat_p[3]);
		mat_c_index[2][0] = row_dim_h;
		mat_c_index[2][1] = row_dim;
		mat_c_index[2][2] = 0;
		mat_c_index[2][3] = col_dim_h;
		subst_mpfmatrix_partial(ret, mat_c_index[2], mat_tmp_c[2], ret_index);
		//_bncomp_subst_mpfmatrix_partial(ret, mat_c_index[2], mat_tmp_c[2], ret_index);
	}

	// -------------------------------
	// C22 := P1 + P3 - P2 + P6
	// -------------------------------
	#pragma omp section
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += 3 * row_dim_h * col_dim_h;

		add_mpfmatrix(mat_tmp_c[3], mat_p[0], mat_p[2]);
		sub_mpfmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[1]);
		add_mpfmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[5]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[3], mat_p[0], mat_p[2]);
		//_bncomp_sub_mpfmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[1]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[5]);
		mat_c_index[3][0] = row_dim_h;
		mat_c_index[3][1] = row_dim;
		mat_c_index[3][2] = col_dim_h;
		mat_c_index[3][3] = col_dim;
		subst_mpfmatrix_partial(ret, mat_c_index[3], mat_tmp_c[3], ret_index);
		//_bncomp_subst_mpfmatrix_partial(ret, mat_c_index[3], mat_tmp_c[3], ret_index);
	}

} // pragma omp parallel sections

	// free
	#pragma omp parallel for
	for(i = 0; i < 7; i++)
	{
		free_mpfmatrix(mat_p[i]);
		free_mpfmatrix(mat_tmp_a[i]);
		free_mpfmatrix(mat_tmp_b[i]);
	}
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
		free_mpfmatrix(mat_tmp_c[i]);
}

// Winograd Variant of Strassen's Algorithm (parallelized sections)
void _bncomp_mul_mpfmatrix_winograd_even2(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim)
{
	unsigned long prec;

	int thread_num, thread_index;
//	long int min_dim = 4; // = 2^2
	MPFMatrix mat_s[8], mat_m[7], mat_t[2], mat_tmp_a[4], mat_tmp_b[4], mat_tmp_c[4];
	long int row_dim_h, row_dim, col_dim_h, col_dim, mid_dim, mid_dim_h;
	long int ret_index[4], mat_tmp_a_index[4], mat_tmp_b_index[4];
	long int mat_a_index[4][4], mat_b_index[4][4], mat_c_index[4][4];
	long int i;

	// initialize
	prec = ret->prec;

	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_mpfmatrix_winograd_even)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	row_dim_h = row_dim / 2;
	col_dim_h = col_dim / 2;
	mid_dim_h = mid_dim / 2;

	// normal matrix multiplication in case of ret_dim <= 4 
//	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	if((ret->row_dim <= min_dim) || (ret->col_dim <= min_dim) || (mid_dim <= min_dim))
	{
		mul_mpfmatrix_simple(ret, mat_a, mat_b);
		//_bncomp_mul_mpfmatrix_simple(ret, mat_a, mat_b);

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		_bncomp_num_mul_mul_mpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		return;
	}

	#pragma omp parallel for
	for(i = 0; i < 4; i++)
	{
		mat_s[i] = init2_mpfmatrix(row_dim_h, mid_dim_h, prec);
		mat_tmp_a[i] = init2_mpfmatrix(row_dim_h, mid_dim_h, prec);

		mat_s[i + 4] = init2_mpfmatrix(mid_dim_h, col_dim_h, prec);
		mat_tmp_b[i] = init2_mpfmatrix(mid_dim_h, col_dim_h, prec);

		mat_tmp_c[i] = init2_mpfmatrix(row_dim_h, col_dim_h, prec);
	}
	#pragma omp parallel for
	for(i = 0; i < 7; i++)
		mat_m[i] = init2_mpfmatrix(row_dim_h, col_dim_h, prec);

	mat_t[0] = init2_mpfmatrix(row_dim_h, col_dim_h, prec);
	mat_t[1] = init2_mpfmatrix(row_dim_h, col_dim_h, prec);

	ret_index[0] = 0;
	ret_index[1] = row_dim_h;
	ret_index[2] = 0;
	ret_index[3] = col_dim_h;

	// indeces for A
	mat_a_index[0][0] = 0        ; mat_a_index[0][1] = row_dim_h; mat_a_index[0][2] = 0        ; mat_a_index[0][3] = mid_dim_h;
	mat_a_index[1][0] = 0        ; mat_a_index[1][1] = row_dim_h; mat_a_index[1][2] = mid_dim_h; mat_a_index[1][3] = mid_dim;
	mat_a_index[2][0] = row_dim_h; mat_a_index[2][1] = row_dim  ; mat_a_index[2][2] = 0        ; mat_a_index[2][3] = mid_dim_h;
	mat_a_index[3][0] = row_dim_h; mat_a_index[3][1] = row_dim  ; mat_a_index[3][2] = mid_dim_h; mat_a_index[3][3] = mid_dim;

	// indeces for B
	mat_b_index[0][0] = 0        ; mat_b_index[0][1] = mid_dim_h; mat_b_index[0][2] = 0        ; mat_b_index[0][3] = col_dim_h;
	mat_b_index[1][0] = 0        ; mat_b_index[1][1] = mid_dim_h; mat_b_index[1][2] = col_dim_h; mat_b_index[1][3] = col_dim;
	mat_b_index[2][0] = mid_dim_h; mat_b_index[2][1] = mid_dim  ; mat_b_index[2][2] = 0        ; mat_b_index[2][3] = col_dim_h;
	mat_b_index[3][0] = mid_dim_h; mat_b_index[3][1] = mid_dim  ; mat_b_index[3][2] = col_dim_h; mat_b_index[3][3] = col_dim;

	// indeces for C
	mat_c_index[0][0] = 0        ; mat_c_index[0][1] = row_dim_h; mat_c_index[0][2] = 0        ; mat_c_index[0][3] = col_dim_h;
	mat_c_index[1][0] = 0        ; mat_c_index[1][1] = row_dim_h; mat_c_index[1][2] = col_dim_h; mat_c_index[1][3] = col_dim;
	mat_c_index[2][0] = row_dim_h; mat_c_index[2][1] = row_dim  ; mat_c_index[2][2] = 0        ; mat_c_index[2][3] = col_dim_h;
	mat_c_index[3][0] = row_dim_h; mat_c_index[3][1] = row_dim  ; mat_c_index[3][2] = col_dim_h; mat_c_index[3][3] = col_dim;

	mat_tmp_a_index[0] = 0;
	mat_tmp_a_index[1] = row_dim_h;
	mat_tmp_a_index[2] = 0;
	mat_tmp_a_index[3] = mid_dim_h;

	mat_tmp_b_index[0] = 0;
	mat_tmp_b_index[1] = mid_dim_h;
	mat_tmp_b_index[2] = 0;
	mat_tmp_b_index[3] = col_dim_h;

	// mat_tmp_a[0], [1], [2], [3] := A11, A12, A21, A22
	// mat_tmp_b[0], [1], [2], [3] := B11, B12, B21, B22
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
	{
		subst_mpfmatrix_partial(mat_tmp_a[i], mat_tmp_a_index, mat_a, mat_a_index[i]);
		subst_mpfmatrix_partial(mat_tmp_b[i], mat_tmp_b_index, mat_b, mat_b_index[i]);
		//_bncomp_subst_mpfmatrix_partial(mat_tmp_a[i], mat_tmp_a_index, mat_a, mat_a_index[i]);
		//_bncomp_subst_mpfmatrix_partial(mat_tmp_b[i], mat_tmp_b_index, mat_b, mat_b_index[i]);
	}
//	printf("subst a, b...\n");

// 4 threads
#pragma omp parallel sections
{
	#pragma omp section
	{
		// -------------------------------
		// S1 := A21 + A22
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		add_mpfmatrix(mat_s[0], mat_tmp_a[2], mat_tmp_a[3]);
		//_bncomp_add_mpfmatrix(mat_s[0], mat_tmp_a[2], mat_tmp_a[3]);
	}

	#pragma omp section
	{
		// -------------------------------
		// S3 := A11 - A21
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		sub_mpfmatrix(mat_s[2], mat_tmp_a[0], mat_tmp_a[2]);
		//_bncomp_sub_mpfmatrix(mat_s[2], mat_tmp_a[0], mat_tmp_a[2]);
	}

	#pragma omp section
	{
		// -------------------------------
		// S5 := B12 - B11
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		sub_mpfmatrix(mat_s[4], mat_tmp_b[1], mat_tmp_b[0]);
		//_bncomp_sub_mpfmatrix(mat_s[4], mat_tmp_b[1], mat_tmp_b[0]);
	}

	#pragma omp section
	{
		// -------------------------------
		// S7 := B22 - B12
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		sub_mpfmatrix(mat_s[6], mat_tmp_b[3], mat_tmp_b[1]);
		//_bncomp_sub_mpfmatrix(mat_s[6], mat_tmp_b[3], mat_tmp_b[1]);
	}
} // pragma omp parallel sections

// 2 threads
#pragma omp parallel sections
{
	#pragma omp section
	{
		// -------------------------------
		// S2 := S1 - A11
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		sub_mpfmatrix(mat_s[1], mat_s[0], mat_tmp_a[0]);
		//_bncomp_sub_mpfmatrix(mat_s[1], mat_s[0], mat_tmp_a[0]);

		// -------------------------------
		// S4 := A12 - S2
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * mid_dim_h;

		sub_mpfmatrix(mat_s[3], mat_tmp_a[1], mat_s[1]);
		//_bncomp_sub_mpfmatrix(mat_s[3], mat_tmp_a[1], mat_s[1]);
	}

	#pragma omp section
	{
		// -------------------------------
		// S6 := B22 - S5
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		sub_mpfmatrix(mat_s[5], mat_tmp_b[3], mat_s[4]);
		//_bncomp_sub_mpfmatrix(mat_s[5], mat_tmp_b[3], mat_s[4]);

		// -------------------------------
		// S8 := S6 - B21
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += mid_dim_h * col_dim_h;

		sub_mpfmatrix(mat_s[7], mat_s[5], mat_tmp_b[2]);
		//_bncomp_sub_mpfmatrix(mat_s[7], mat_s[5], mat_tmp_b[2]);
	}

} // pragma omp parallel sections

//	printf("s...\n");

// 7 threads
#pragma omp parallel sections
{
	#pragma omp section
	{
		// -------------------------------
		// M1 := S2 * S6
		// -------------------------------
		_bncomp_mul_mpfmatrix_winograd_even2(mat_m[0], mat_s[1], mat_s[5], min_dim);
	}

	#pragma omp section
	{
		// -------------------------------
		// M2 := A11 * B11
		// -------------------------------
		_bncomp_mul_mpfmatrix_winograd_even2(mat_m[1], mat_tmp_a[0], mat_tmp_b[0], min_dim);
	}

	#pragma omp section
	{
		// -------------------------------
		// M3 := A12 * B21
		// -------------------------------
		_bncomp_mul_mpfmatrix_winograd_even2(mat_m[2], mat_tmp_a[1], mat_tmp_b[2], min_dim);
	}

	#pragma omp section
	{
		// -------------------------------
		// M4 := S3 * S7
		// -------------------------------
		_bncomp_mul_mpfmatrix_winograd_even2(mat_m[3], mat_s[2], mat_s[6], min_dim);
	}

	#pragma omp section
	{
		// -------------------------------
		// M5 := S1 * S5
		// -------------------------------
		_bncomp_mul_mpfmatrix_winograd_even2(mat_m[4], mat_s[0], mat_s[4], min_dim);
	}

	#pragma omp section
	{
		// -------------------------------
		// M6 := S4 * B22
		// -------------------------------
		_bncomp_mul_mpfmatrix_winograd_even2(mat_m[5], mat_s[3], mat_tmp_b[3], min_dim);
	}

	#pragma omp section
	{
		// -------------------------------
		// M7 := A22 * S8
		// -------------------------------
		_bncomp_mul_mpfmatrix_winograd_even2(mat_m[6], mat_tmp_a[3], mat_s[7], min_dim);
	}
} // pragma omp parallel sections

//	printf("m...\n");

	// -------------------------------
	// T1 := M1 + M2
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

	//add_mpfmatrix(mat_t[0], mat_m[0], mat_m[1]);
	_bncomp_add_mpfmatrix(mat_t[0], mat_m[0], mat_m[1]);

	// -------------------------------
	// T2 := T1 + M4
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

	//add_mpfmatrix(mat_t[1], mat_t[0], mat_m[3]);
	_bncomp_add_mpfmatrix(mat_t[1], mat_t[0], mat_m[3]);

//	printf("t...\n");

// 4 threads
#pragma omp parallel sections
{
	#pragma omp section
	{
		// -------------------------------
		// C11 := M2 + M3
		// -------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

		add_mpfmatrix(mat_tmp_c[0], mat_m[1], mat_m[2]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[0], mat_m[1], mat_m[2]);
	}

	#pragma omp section
	{
		// -------------------------------
		// C12 := T1 + M5 + M6
		// -------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += 2 * row_dim_h * col_dim_h;

		add_mpfmatrix(mat_tmp_c[1], mat_t[0], mat_m[4]);
		add_mpfmatrix(mat_tmp_c[1], mat_tmp_c[1], mat_m[5]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[1], mat_t[0], mat_m[4]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[1], mat_tmp_c[1], mat_m[5]);
	}

	#pragma omp section
	{
		// -------------------------------
		// C21 := T2 - M7
		// -------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

		sub_mpfmatrix(mat_tmp_c[2], mat_t[1], mat_m[6]);
		//_bncomp_sub_mpfmatrix(mat_tmp_c[2], mat_t[1], mat_m[6]);
	}

	#pragma omp section
	{
		// -------------------------------
		// C22 := T2 + M5
		// -------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_mpfmatrix_strassen += row_dim_h * col_dim_h;

		add_mpfmatrix(mat_tmp_c[3], mat_t[1], mat_m[4]);
		//_bncomp_add_mpfmatrix(mat_tmp_c[3], mat_t[1], mat_m[4]);
	}
} // pragma omp parallel sections

//	printf("c...\n");

	// -------------------------------
	// RET := [C11 C12]
	//        [C21 C22]
	// -------------------------------
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
		subst_mpfmatrix_partial(ret, mat_c_index[i], mat_tmp_c[i], ret_index);
		//_bncomp_subst_mpfmatrix_partial(ret, mat_c_index[i], mat_tmp_c[i], ret_index);

//	printf("set...\n");


	// free
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
	{
		free_mpfmatrix(mat_s[i]);
		free_mpfmatrix(mat_tmp_a[i]);
		free_mpfmatrix(mat_s[i + 4]);
		free_mpfmatrix(mat_tmp_b[i]);
		free_mpfmatrix(mat_tmp_c[i]);
	}
	#pragma omp parallel for
	for(i = 0; i < 7; i++)
		free_mpfmatrix(mat_m[i]);
	
	free_mpfmatrix(mat_t[0]);
	free_mpfmatrix(mat_t[1]);
}
#endif // USE_GMP

// The following main function is for debugging
#ifdef DEBUG

#include "get_secv.h"

int main(int argc, char *argv[])
{
	int num_threads;
	long int i, j, row_dim, col_dim, mid_dim;
	unsigned long prec = 128;
	mpf_t ddtmp[4];
	MPFMatrix mpfc, mpfa, mpfb, mpfc_normal, mpfc_block, mpfc_nonrec, mpfc_tmp;
	MPFVector mpfdiag_left, mpfdiag_right;
	double stime, etime[4], reldiff[4];

//	dim = 128;
	if(argc <= 5)
	{
		fprintf(stderr, "Usage: %s [prec] [row_dim] [col_dim] [mid_dim] [#thread]\n", argv[0]);
		return 0;
	}

	prec = atol(argv[1]);
	if(prec <= 0)
		prec = 128;

	row_dim = atol(argv[2]);
	if(row_dim <= 0)
		return 0;

	col_dim = atol(argv[3]);
	if(col_dim <= 0)
		return 0;

	mid_dim = atol(argv[4]);
	if(mid_dim <= 0)
		return 0;

	if(argc >= 6)
	{
		num_threads = (unsigned long)atol(argv[5]);
		if(num_threads < 1)
			num_threads = 1;
	}

/* double-double precision */
//	set_bncomp_num_threads(4);
	set_bncomp_num_threads(num_threads);
//	printf("OpenMP #Threads = %ld\n", omp_get_num_threads());

	set_bnc_default_prec(prec);

	mpf_init2(ddtmp[0], prec); mpf_set_ui(ddtmp[0], 0UL);
	mpf_init2(ddtmp[1], prec); mpf_set_ui(ddtmp[1], 0UL);
	mpf_init2(ddtmp[2], prec); mpf_set_ui(ddtmp[2], 0UL);
	mpf_init2(ddtmp[3], prec); mpf_set_ui(ddtmp[3], 0UL);

	mpfc = init_mpfmatrix(row_dim, col_dim);
	mpfc_tmp = init_mpfmatrix(row_dim, col_dim);
	mpfc_normal = init_mpfmatrix(row_dim, col_dim);
	mpfc_block = init_mpfmatrix(row_dim, col_dim);
	mpfc_nonrec = init_mpfmatrix(row_dim, col_dim);
	mpfa = init_mpfmatrix(row_dim, mid_dim);
	mpfb = init_mpfmatrix(mid_dim, col_dim);

//	mpfdiag_left = init_ddvector(dim);
//	mpfdiag_right = init_ddvector(dim);

	// set A
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < mid_dim; j++)
		{
			mpf_set_d(ddtmp[0], (double)rand());
			if(rand() % 2 != 0)
				mpf_neg(ddtmp[0], ddtmp[0]);

			mpf_set_d(ddtmp[1], (double)rand());
			mpf_ui_div(ddtmp[1], 1UL, ddtmp[1]);
			if(rand() % 2 != 0)
				mpf_neg(ddtmp[1], ddtmp[1]);

			set_mpfmatrix_ij(mpfa, i, j, ddtmp[0]);
		}
	}

	// set B
	for(i = 0; i < mid_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			mpf_set_d(ddtmp[0], (double)rand());
			if(rand() % 2 != 0)
				mpf_neg(ddtmp[0], ddtmp[0]);

			mpf_set_d(ddtmp[1], (double)rand());
			mpf_ui_div(ddtmp[1], 1UL, ddtmp[1]);
			if(rand() % 2 != 0)
				mpf_neg(ddtmp[1], ddtmp[1]);

			set_mpfmatrix_ij(mpfb, i, j, ddtmp[1]);
		}
	}

	// normal matrix mul
	printf("_bncomp_mul_mpfmatrix_simple...");
	stime = get_real_secv();
	//_bncomp_mul_mpfmatrix(mpfc_normal, mpfa, mpfb);
	mul_mpfmatrix(mpfc_normal, mpfa, mpfb);
	etime[0] = get_real_secv() - stime;
	printf("end\n");

	//left_scaling_mpfmatrix(mpfa, mpfa, mpfdiag_left, NULL);
	//right_scaling_mpfmatrix(mpfb, mpfb, mpfdiag_right, NULL);

	// blocked matrix mul
	printf("_bncomp_mul_mpfmatrix_block...");
	//stime = get_secv();
	stime = get_real_secv();
	_bncomp_mul_mpfmatrix_block(mpfc_block, mpfa, mpfb, 32);
	//etime[1] = get_secv() - stime;
	etime[1] = get_real_secv() - stime;
	printf("end\n");fflush(stdout); 

	// Strassen 
	fflush(stdout); printf("_bncomp_mul_mpfmatrix_strassen...");
	stime = get_real_secv();
	_bncomp_mul_mpfmatrix_strassen(mpfc, mpfa, mpfb, 8);
//	mul_mpfmatrix_strassen(mpfc, mpfa, mpfb, 16);
//	mul_mpfmatrix_strassen(mpfc, mpfa, mpfb, 32);
	etime[2] = get_real_secv() - stime;
	printf("end\n"); fflush(stdout); 

	// Strassen w/o recursion
	fflush(stdout); printf("_bncomp_mul_mpfmatrix_strassen_nonrec...");
	stime = get_real_secv();
	_bncomp_mul_mpfmatrix_strassen_nonrec(mpfc_nonrec, mpfa, mpfb, 8);
	etime[3] = get_real_secv() - stime;
	printf("end\n");

	//mul_mpfmatrix_dddiag(mpfc, mpfdiag_left, 0, mpfc, mpfdiag_right, 0);

	// difference
	sub_mpfmatrix(mpfc_tmp  , mpfc_normal, mpfc);
	sub_mpfmatrix(mpfc      , mpfc_normal, mpfc_block);
	sub_mpfmatrix(mpfc_block, mpfc_normal, mpfc_nonrec);

	// print
	printf("row_dim, col_dim, mid_dim: %ld, %ld, %ld\n", row_dim, col_dim, mid_dim);
	printf("normal         : %f\n", etime[0]);
	printf("block          : %f\n", etime[1]);
	printf("strassen       : %f\n", etime[2]);
	printf("strassen_nonrec: %f\n", etime[3]);

	normi_mpfmatrix(ddtmp[0], mpfc_tmp);
	normi_mpfmatrix(ddtmp[1], mpfc);
	normi_mpfmatrix(ddtmp[2], mpfc_block);
	normi_mpfmatrix(ddtmp[3], mpfc_normal);

	mpf_div(ddtmp[0], ddtmp[0], ddtmp[3]);
	mpf_div(ddtmp[1], ddtmp[1], ddtmp[3]);
	mpf_div(ddtmp[2], ddtmp[2], ddtmp[3]);
	printf("||reldiff_block   ||       : "); mpf_out_str(stdout, 10, 20, ddtmp[1]); printf("\n");
	printf("||reldiff_strassen||       : "); mpf_out_str(stdout, 10, 20, ddtmp[0]); printf("\n");
	printf("||reldiff_strassen_nonrec||: "); mpf_out_str(stdout, 10, 20, ddtmp[2]); printf("\n");

/* Inverse */

//	frank_mpfmatrix(mpfa, dim);
//	frank_mpfmatrix(mpfb, dim);
//	lotkin_mpfmatrix(mpfa, dim);
//	lotkin_mpfmatrix(mpfb, dim);

	free_mpfmatrix(mpfc);
	free_mpfmatrix(mpfc_tmp);
	free_mpfmatrix(mpfc_normal);
	free_mpfmatrix(mpfc_block);
	free_mpfmatrix(mpfc_nonrec);
	free_mpfmatrix(mpfa);
	free_mpfmatrix(mpfb);

//	free_ddvector(mpfdiag_left);
//	free_ddvector(mpfdiag_right);

	return 0;
}
#endif // DEBUG
