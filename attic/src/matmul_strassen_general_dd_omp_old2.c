/********************************************************************************/
/* matmul_strassen_general_dd_omp.c:                                            */
/* Copyright (C) 2015-2016 Tomonori Kouya                                       */
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
//#include "ddlinear.h"

#include "bncomp.h"

#ifdef USE_IMKL
	#include "mkl.h"
	#include "mkl_cblas.h" // for Intel Math Kernel Library
#endif

#include "matmul_strassen.h"

// count the number of computations
long int _bncomp_num_addsub_mul_ddmatrix_strassen;	// addition and subtraction
long int _bncomp_num_mul_mul_ddmatrix_strassen;		// multiplication

// Total number of threads currently used in _bncomp_matmul_ddmatrix_strassen
#define MAX_TOTAL_NUM_THREADS (2)
//#define MAX_TOTAL_NUM_THREADS (4)
//#define MAX_TOTAL_NUM_THREADS (12)
//#define MAX_TOTAL_NUM_THREADS (12 * 2)
//#define MAX_TOTAL_NUM_THREADS (12 * 4)
int _bncomp_total_num_threads_matmul_ddmatrix_strassen = 0; // -> _bncomp_total_num_threads[MATMUL_DDMATRIX_STRASSEN] in matmul_general.h;

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_add_ddmatrix_partial(DDMatrix ret, long int ret_index[4], DDMatrix mat_a, long int mat_a_index[4], DDMatrix mat_b, long int mat_b_index[4])
{
	int thread_num, thread_index;
	long int i, j;
	long int ret_i[BNCOMP_MAX_NUM_THREADS], ret_j[BNCOMP_MAX_NUM_THREADS], a_i[BNCOMP_MAX_NUM_THREADS], a_j[BNCOMP_MAX_NUM_THREADS], b_i[BNCOMP_MAX_NUM_THREADS], b_j[BNCOMP_MAX_NUM_THREADS];
	long int imax, jmax;
#ifdef __cplusplus
	static dd_real tmp_val[BNCOMP_MAX_NUM_THREADS];
#else // __cplusplus
	static double tmp_val[BNCOMP_MAX_NUM_THREADS][DDSIZE];
#endif // __cpplusplus

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_dd(tmp_val[thread_index]);
	}

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < imax; i++)
	{
		thread_index = omp_get_thread_num();

		ret_i [thread_index]= ret_index[0] + i;
		a_i[thread_index] = mat_a_index[0] + i;
		b_i[thread_index] = mat_b_index[0] + i;
		//printf("i: %ld %ld %ld\n", ret_i, a_i, b_i);
		for(j = 0; j < jmax; j++)
		{
			thread_index = omp_get_thread_num();

			ret_j[thread_index] = ret_index[2] + j;
			a_j[thread_index] = mat_a_index[2] + j;
			b_j[thread_index] = mat_b_index[2] + j;
			//printf("j: %ld %ld %ld\n", ret_j, a_j, b_j);

			//tmp_val = get_ddmatrix_ij(mat_a, a_i, a_j) + get_ddmatrix_ij(mat_b, b_i, b_j);
			rdd_add(tmp_val[thread_index], get_ddmatrix_ij(mat_a, a_i[thread_index], a_j[thread_index]), get_ddmatrix_ij(mat_b, b_i[thread_index], b_j[thread_index]));
			set_ddmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index], tmp_val[thread_index]);
		}
	}
}

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_sub_ddmatrix_partial(DDMatrix ret, long int ret_index[4], DDMatrix mat_a, long int mat_a_index[4], DDMatrix mat_b, long int mat_b_index[4])
{
	int thread_num, thread_index;
	long int i, j;
	long int ret_i[BNCOMP_MAX_NUM_THREADS], ret_j[BNCOMP_MAX_NUM_THREADS], a_i[BNCOMP_MAX_NUM_THREADS], a_j[BNCOMP_MAX_NUM_THREADS], b_i[BNCOMP_MAX_NUM_THREADS], b_j[BNCOMP_MAX_NUM_THREADS];
	long int imax, jmax;
#ifdef __cplusplus
	static dd_real tmp_val[BNCOMP_MAX_NUM_THREADS];
#else // __cplusplus
	double tmp_val[BNCOMP_MAX_NUM_THREADS][DDSIZE];
#endif // _cplusplus

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_dd(tmp_val[thread_index]);
	}

	#pragma omp parallel for private(thread_index, j)
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

			//tmp_val = get_ddmatrix_ij(mat_a, a_i, a_j) - get_ddmatrix_ij(mat_b, b_i, b_j);
			rdd_sub(tmp_val[thread_index], get_ddmatrix_ij(mat_a, a_i[thread_index], a_j[thread_index]), get_ddmatrix_ij(mat_b, b_i[thread_index], b_j[thread_index]));
			set_ddmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index], tmp_val[thread_index]);
		}
	}
}

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_subst_ddmatrix_partial(DDMatrix ret, long int ret_index[4], DDMatrix mat_a, long int mat_a_index[4])
{
	int thread_num, thread_index;
	long int i, j;
	long int ret_i[BNCOMP_MAX_NUM_THREADS], ret_j[BNCOMP_MAX_NUM_THREADS], a_i[BNCOMP_MAX_NUM_THREADS], a_j[BNCOMP_MAX_NUM_THREADS];
	long int imax, jmax;

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	thread_num = omp_get_num_threads();

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < imax; i++)
	{
		thread_index = omp_get_thread_num();

		ret_i[thread_index] = ret_index[0] + i;
		a_i[thread_index] = mat_a_index[0] + i;
		for(j = 0; j < jmax; j++)
		{
			ret_j[thread_index] = ret_index[2] + j;
			a_j[thread_index] = mat_a_index[2] + j;

			set_ddmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index], get_ddmatrix_ij(mat_a, a_i[thread_index], a_j[thread_index]));
		}
	}
}

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_subst_ddmatrix_partial_checked(DDMatrix ret, long int ret_index[4], DDMatrix mat_a, long int mat_a_index[4])
{
	int thread_num, thread_index;
	long int i, j;
	long int ret_i[BNCOMP_MAX_NUM_THREADS], ret_j[BNCOMP_MAX_NUM_THREADS], a_i[BNCOMP_MAX_NUM_THREADS], a_j[BNCOMP_MAX_NUM_THREADS];
	long int imax, jmax;
#ifdef __cplusplus
	static dd_real dd_tmp;
#else // __cplusplus
	double *ptr_ddtmp;
#endif // _cplusplus

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	#pragma omp parallel for private(thread_index, j)
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
					{
					#ifdef __cplusplus
						set_ddmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index], get_ddmatrix_ij(mat_a, a_i[thread_index], a_j[thread_index]));
					#else // __cplusplus
						ptr_ddtmp = get_ddmatrix_ij(mat_a, a_i[thread_index], a_j[thread_index]);
						set_ddmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index], ptr_ddtmp);
					#endif // __cplusplus
					}
					else
						set0_ddmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index]); // Padding
					//printf("Warning: ret_index = %d, %d, %d, %d\n", ret_i, ret_j, a_i, a_j);
				}
			}
		}
	}
}

// reverse sign
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_neg_ddmatrix_partial(DDMatrix ret, long int ret_index[4], DDMatrix mat_a, long int mat_a_index[4])
{
	int thread_num, thread_index;
	long int i, j;
	long int ret_i[BNCOMP_MAX_NUM_THREADS], ret_j[BNCOMP_MAX_NUM_THREADS], a_i[BNCOMP_MAX_NUM_THREADS], a_j[BNCOMP_MAX_NUM_THREADS];
	long int imax, jmax;
#ifdef __cplusplus
	dd_real tmp[BNCOMP_MAX_NUM_THREADS];
#else
	double tmp[BNCOMP_MAX_NUM_THREADS][DDSIZE];
#endif // __cplusplus

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_dd(tmp[thread_index]);
	}

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < imax; i++)
	{
		thread_index = omp_get_thread_num();

		ret_i[thread_index] = ret_index[0] + i;
		a_i[thread_index] = mat_a_index[0] + i;
		for(j = 0; j < jmax; j++)
		{
			ret_j[thread_index] = ret_index[2] + j;
			a_j[thread_index] = mat_a_index[2] + j;
#ifdef __cplusplus
			set_ddmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index], -get_ddmatrix_ij(mat_a, a_i[thread_index], a_j[thread_index]));
#else // __cplusplus
			rdd_neg(tmp[thread_index], get_ddmatrix_ij(mat_a, a_i[thread_index], a_j[thread_index]));
			set_ddmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index], tmp[thread_index]);
#endif // __cplusplus
		}
	}
}

/* c = a * b */
//void _bncomp_mul_ddmatrix_simple(DDMatrix ret, DDMatrix a, DDMatrix b)
//{
//	_bncomp_mul_ddmatrix(ret, a, b);
//}

// Block matrix multiplicaiton
void _bncomp_mul_ddmatrix_block(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim)
{
	int thread_num, thread_index;
	int row_padding_flag = 0, col_padding_flag = 0, mid_padding_flag = 0;
	long i, j, k, row_dim, col_dim, mid_dim;
	long int num_div_row, num_div_col, num_div_mid, max_num_div;
	long int **mat_a_index, small_mat_a_index[4];
	long int **mat_b_index, small_mat_b_index[4];
	long int **ret_index, small_ret_index[4];
	//DMatrix small_ret[1024], small_mat_a[1024], small_mat_b[1024], small_tmp_mat;
	DDMatrix *small_ret, *small_mat_a, *small_mat_b, *small_tmp_mat;

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(_bncomp_mul_ddmatrix_block)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	// normal matrix multiplication in case of ret_dim <= 4 
	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	{
		_bncomp_mul_ddmatrix_simple(ret, mat_a, mat_b);
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

	small_ret = (DDMatrix *)calloc(num_div_col, sizeof(DDMatrix));
	small_mat_a = (DDMatrix *)calloc(num_div_mid, sizeof(DDMatrix));
	small_mat_b = (DDMatrix *)calloc(num_div_mid, sizeof(DDMatrix));
	small_tmp_mat = (DDMatrix *)calloc(num_div_mid, sizeof(DDMatrix));

	#pragma omp parallel for
	for(i = 0; i < num_div_col; i++)
		small_ret[i] = init_ddmatrix(min_dim, min_dim);

	#pragma omp parallel for
	for(i = 0; i < num_div_mid; i++)
	{
		small_mat_a[i] = init_ddmatrix(min_dim, min_dim);
		small_mat_b[i] = init_ddmatrix(min_dim, min_dim);
		small_tmp_mat[i] = init_ddmatrix(min_dim, min_dim);
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
			subst_ddmatrix_partial_checked(small_mat_a[j], small_mat_a_index, mat_a, mat_a_index[j]);

			//printf("small_mat_a[%ld]:\n", j); print_ddmatrix(small_mat_a[j]);
		}

		for(j = 0; j < num_div_col; j++)
		{
			set0_ddmatrix(small_ret[j]);

			#pragma omp parallel for
			for(k = 0; k < num_div_mid; k++)
			{
				// copy matrices
				mat_b_index[k][0] = k * min_dim;
				mat_b_index[k][1] = (k + 1) * min_dim;
				mat_b_index[k][2] = j * min_dim;
				mat_b_index[k][3] = (j + 1) * min_dim;
				subst_ddmatrix_partial_checked(small_mat_b[k], small_mat_b_index, mat_b, mat_b_index[k]);
				//_bncomp_subst_ddmatrix_partial_checked(small_mat_b[k], small_mat_b_index, mat_b, mat_b_index[k]);
				// ret[j] += small_mat_a[i][k] * small_mat_b[k][j];
				mul_ddmatrix(small_tmp_mat[k], small_mat_a[k], small_mat_b[k]);
			}

			//for(k = 0; k < num_div_mid; k++)
			//	_bncomp_add_ddmatrix(small_ret[j], small_ret[j], small_tmp_mat[k]);
			for(k = 0; k < num_div_mid; k++)
				add_ddmatrix(small_ret[j], small_ret[j], small_tmp_mat[k]);

			ret_index[j][0] = i * min_dim;
			ret_index[j][1] = (i + 1) * min_dim;
			ret_index[j][2] = j * min_dim;
			ret_index[j][3] = (j + 1) * min_dim;
			subst_ddmatrix_partial_checked(ret, ret_index[j], small_ret[j], small_ret_index);
			//_bncomp_subst_ddmatrix_partial_checked(ret, ret_index[j], small_ret[j], small_ret_index);
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
		free_ddmatrix(small_ret[i]);

	#pragma omp parallel for
	for(i = 0; i < num_div_mid; i++)
	{
		free_ddmatrix(small_mat_a[i]);
		free_ddmatrix(small_mat_b[i]);
		free_ddmatrix(small_tmp_mat[i]);
	}
	free(small_ret);
	free(small_mat_a);
	free(small_mat_b);
	free(small_tmp_mat);
}

// Block matrix multiplicaiton(poor, so obosolete)
void _bncomp_mul_ddmatrix_block_old(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim)
{
	int thread_num, thread_index;
	int row_padding_flag = 0, col_padding_flag = 0, mid_padding_flag = 0;
	long i, j, k, row_dim, col_dim, mid_dim;
	long int num_div_row, num_div_col, num_div_mid;
	long int mat_a_index[BNCOMP_MAX_NUM_THREADS][4], small_mat_a_index[BNCOMP_MAX_NUM_THREADS][4];
	long int mat_b_index[BNCOMP_MAX_NUM_THREADS][4], small_mat_b_index[BNCOMP_MAX_NUM_THREADS][4];
	long int ret_index[BNCOMP_MAX_NUM_THREADS][4], small_ret_index[BNCOMP_MAX_NUM_THREADS][4];
	//DMatrix small_ret[1024], small_mat_a[1024], small_mat_b[1024], small_tmp_mat;
	DDMatrix *small_ret, *small_mat_a, *small_mat_b, small_tmp_mat[BNCOMP_MAX_NUM_THREADS];

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(_bncomp_mul_ddmatrix_block)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}
	

	// normal matrix multiplication in case of ret_dim <= 4 
	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	{
		_bncomp_mul_ddmatrix_simple(ret, mat_a, mat_b);
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
	thread_num = omp_get_num_threads();

	small_ret = (DDMatrix *)calloc(sizeof(DDMatrix), num_div_col);
	small_mat_a = (DDMatrix *)calloc(sizeof(DDMatrix), num_div_mid);
	small_mat_b = (DDMatrix *)calloc(sizeof(DDMatrix), num_div_mid);

	#pragma omp parallel for
	for(i = 0; i < num_div_col; i++)
		small_ret[i] = init_ddmatrix(min_dim, min_dim);

	#pragma omp parallel for
	for(i = 0; i < num_div_mid; i++)
	{
		small_mat_a[i] = init_ddmatrix(min_dim, min_dim);
		small_mat_b[i] = init_ddmatrix(min_dim, min_dim);
	}

	//small_tmp_mat = init_ddmatrix(min_dim, min_dim);
	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		small_tmp_mat[thread_index] = init_ddmatrix(min_dim, min_dim);
	}

	// mail loop
	//#pragma omp parallel for private(thread_index, j, k)
	for(i = 0; i < num_div_row; i++)
	{
		//mat_a_index[0] = i * min_dim;
		//mat_a_index[1] = (i + 1) * min_dim;
		#pragma omp parallel private(thread_index)
		{
			thread_index = omp_get_thread_num();
			mat_a_index[thread_index][0] = i * min_dim;
			mat_a_index[thread_index][1] = (i + 1) * min_dim;
		}

		#pragma omp parallel for private(thread_index)
		for(j = 0; j < num_div_mid; j++)
		{
			thread_index = omp_get_thread_num();

			// copy matrices
			mat_a_index[thread_index][2] = j * min_dim;
			mat_a_index[thread_index][3] = (j + 1) * min_dim;
			small_mat_a_index[thread_index][0] = 0;
			small_mat_a_index[thread_index][1] = min_dim;
			small_mat_a_index[thread_index][2] = 0;
			small_mat_a_index[thread_index][3] = min_dim;
			//subst_ddmatrix_partial(small_mat_a[j], small_mat_a_index, mat_a, mat_a_index);
			subst_ddmatrix_partial_checked(small_mat_a[j], small_mat_a_index[thread_index], mat_a, mat_a_index[thread_index]);
		}

		#pragma omp parallel for private(thread_index, k)
		for(j = 0; j < num_div_col; j++)
		{
			thread_index = omp_get_thread_num();

			set0_ddmatrix(small_ret[j]);
			mat_b_index[thread_index][2] = j * min_dim;
			mat_b_index[thread_index][3] = (j + 1) * min_dim;

			#pragma omp critical
			for(k = 0; k < num_div_mid; k++)
			{
				// copy matrices
				mat_b_index[thread_index][0] = k * min_dim;
				mat_b_index[thread_index][1] = (k + 1) * min_dim;
				small_mat_b_index[thread_index][0] = 0;
				small_mat_b_index[thread_index][1] = min_dim;
				small_mat_b_index[thread_index][2] = 0;
				small_mat_b_index[thread_index][3] = min_dim;
				//subst_ddmatrix_partial(small_mat_b[k], small_mat_b_index, mat_b, mat_b_index);
				subst_ddmatrix_partial_checked(small_mat_b[k], small_mat_b_index[thread_index], mat_b, mat_b_index[thread_index]);

				// ret[j] += small_mat_a[i][k] * small_mat_b[k][j];
				mul_ddmatrix(small_tmp_mat[thread_index], small_mat_a[k], small_mat_b[k]);
				add_ddmatrix(small_ret[j], small_ret[j], small_tmp_mat[thread_index]);
			}
			ret_index[thread_index][0] = i * min_dim;
			ret_index[thread_index][1] = (i + 1) * min_dim;
			ret_index[thread_index][2] = j * min_dim;
			ret_index[thread_index][3] = (j + 1) * min_dim;
			small_ret_index[thread_index][0] = 0;
			small_ret_index[thread_index][1] = min_dim;
			small_ret_index[thread_index][2] = 0;
			small_ret_index[thread_index][3] = min_dim;
			//subst_ddmatrix_partial(ret, ret_index, small_ret[j], small_ret_index);
			subst_ddmatrix_partial_checked(ret, ret_index[thread_index], small_ret[j], small_ret_index[thread_index]);
		}
	}

	// free
	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		free_ddmatrix(small_tmp_mat[thread_index]);
	}

	#pragma omp parallel for
	for(i = 0; i < num_div_col; i++)
		free_ddmatrix(small_ret[i]);

	#pragma omp parallel for
	for(i = 0; i < num_div_mid; i++)
	{
		free_ddmatrix(small_mat_a[i]);
		free_ddmatrix(small_mat_b[i]);
	}
	free(small_ret);
	free(small_mat_a);
	free(small_mat_b);
}

// Padding to 2-powered dimensional matrix
DDMatrix _bncomp_init_static_padding_ddmatrix_strassen(DDMatrix orig_mat)
{
	int thread_num, thread_index;
	DDMatrix ret = NULL;
	long int ret_row_dim, ret_col_dim, min_dim, i, j;

	if(orig_mat == NULL)
	{
		fprintf(stderr, "Warning: orig_mat is empty!(padding_ddmatrix_strassen)\n");
		return NULL;
	}

	ret_row_dim = (long int)pow(2.0, ceil(mylog2((double)orig_mat->row_dim)));
	ret_col_dim = (long int)pow(2.0, ceil(mylog2((double)orig_mat->col_dim)));

	//printf("Padding: row_dim %ld -> %ld, col_dim %ld -> %ld\n", orig_mat->row_dim, ret_row_dim, orig_mat->col_dim, ret_col_dim);

	ret = init_ddmatrix(ret_row_dim, ret_col_dim);
	if(ret == NULL)
	{
		fprintf(stderr, "Warning: padding matrix cannot be allocated!(padding_ddmatrix_strassen)\n");
		return NULL;
	}

	// ret = [ A | 0 ]
	//       [---+---]
	//       [ 0 | I ]
	#pragma omp parallel for private(j)
	for(i = 0; i < orig_mat->row_dim; i++)
	{
		for(j = 0; j < orig_mat->col_dim; j++)
			set_ddmatrix_ij(ret, i, j, get_ddmatrix_ij(orig_mat, i, j));
	}

	min_dim = (ret_row_dim < ret_col_dim) ? ret_row_dim : ret_col_dim;
	#pragma omp parallel for
	for(i = orig_mat->row_dim; i < min_dim; i++)
		set_ddmatrix_ij_ui(ret, i, i, 1UL);

	return ret;
}

// Padding to even dimensional matrix
DDMatrix _bncomp_init_dynamic_padding_ddmatrix_strassen(DDMatrix orig_mat)
{
	int thread_num, thread_index;
	DDMatrix ret = NULL;
	long int ret_row_dim, ret_col_dim, min_dim, i, j;

	if(orig_mat == NULL)
	{
		fprintf(stderr, "Warning: orig_mat is empty!(padding_ddmatrix_strassen)\n");
		return NULL;
	}

	ret_row_dim = orig_mat->row_dim;
	ret_col_dim = orig_mat->col_dim;

	if((ret_row_dim % 2) == 1)
		ret_row_dim++;
	if((ret_col_dim % 2) == 1)
		ret_col_dim++;

//	printf("Padding: row_dim %ld -> %ld, col_dim %ld -> %ld\n", orig_mat->row_dim, ret_row_dim, orig_mat->col_dim, ret_col_dim);

	ret = init_ddmatrix(ret_row_dim, ret_col_dim);
	if(ret == NULL)
	{
		fprintf(stderr, "Warning: padding matrix cannot be allocated!(padding_ddmatrix_strassen)\n");
		return NULL;
	}

	// ret = [ A | 0 ]
	//       [---+---]
	//       [ 0 | I ]
	#pragma omp parallel for private(j)
	for(i = 0; i < orig_mat->row_dim; i++)
	{
		for(j = 0; j < orig_mat->col_dim; j++)
			set_ddmatrix_ij(ret, i, j, get_ddmatrix_ij(orig_mat, i, j));
	}

/*	min_dim = (ret_row_dim < ret_col_dim) ? ret_row_dim : ret_col_dim;
	for(i = orig_mat->row_dim; i < min_dim; i++)
		set_ddmatrix_ij_ui(ret, i, i, 1UL);
*/
	return ret;
}

// Padding to even dimensional matrix
DDMatrix _bncomp_init_dynamic_padding_ddmatrix_strassen2(DDMatrix orig_mat, long int min_dim)
{
	int thread_num, thread_index;
	DDMatrix ret = NULL;
	long int ret_row_dim, ret_col_dim, i, j;

	if(orig_mat == NULL)
	{
		fprintf(stderr, "Warning: orig_mat is empty!(padding_ddmatrix_strassen)\n");
		return NULL;
	}

	ret_row_dim = orig_mat->row_dim;
	ret_col_dim = orig_mat->col_dim;

	if((ret_row_dim % min_dim) >= 1)
		ret_row_dim = ((ret_row_dim / min_dim) + 1) * min_dim;
	if((ret_col_dim % min_dim) >= 1)
		ret_col_dim = ((ret_col_dim / min_dim) + 1) * min_dim;

//	printf("Padding: row_dim %ld -> %ld, col_dim %ld -> %ld\n", orig_mat->row_dim, ret_row_dim, orig_mat->col_dim, ret_col_dim);

	ret = init_ddmatrix(ret_row_dim, ret_col_dim);
	if(ret == NULL)
	{
		fprintf(stderr, "Warning: padding matrix cannot be allocated!(padding_ddmatrix_strassen)\n");
		return NULL;
	}

	// ret = [ A | 0 ]
	//       [---+---]
	//       [ 0 | I ]
	#pragma omp parallel for private(j)
	for(i = 0; i < orig_mat->row_dim; i++)
	{
		for(j = 0; j < orig_mat->col_dim; j++)
			set_ddmatrix_ij(ret, i, j, get_ddmatrix_ij(orig_mat, i, j));
	}
/*
	min_dim = (ret_row_dim < ret_col_dim) ? ret_row_dim : ret_col_dim;
	for(i = orig_mat->row_dim; i < min_dim; i++)
		set_ddmatrix_ij_ui(ret, i, i, 1UL);
*/
	return ret;
}

// Strassen's Algorithm with static padding
void _bncomp_mul_ddmatrix_strassen_odd_padding(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim)
{
	int thread_num, thread_index;
	long int tmp_ret_index[4], ret_index[4];
	DDMatrix tmp_ret, tmp_mat_a, tmp_mat_b;

	// padding
#ifdef USE_STATIC_PADDING
	tmp_ret = _bncomp_init_static_padding_ddmatrix_strassen(ret);
	tmp_mat_a = _bncomp_init_static_padding_ddmatrix_strassen(mat_a);
	tmp_mat_b = _bncomp_init_static_padding_ddmatrix_strassen(mat_b);
#else
//	tmp_ret = _bncomp_init_dynamic_padding_ddmatrix_strassen(ret);
//	tmp_mat_a = _bncomp_init_dynamic_padding_ddmatrix_strassen(mat_a);
//	tmp_mat_b = _bncomp_init_dynamic_padding_ddmatrix_strassen(mat_b);
	tmp_ret = _bncomp_init_dynamic_padding_ddmatrix_strassen2(ret, min_dim);
	tmp_mat_a = _bncomp_init_dynamic_padding_ddmatrix_strassen2(mat_a, min_dim);
	tmp_mat_b = _bncomp_init_dynamic_padding_ddmatrix_strassen2(mat_b, min_dim);
#endif

	// strassen
#ifdef USE_WINOGRAD
//	_bncomp_mul_ddmatrix_winograd_even(tmp_ret, tmp_mat_a, tmp_mat_b, min_dim);
	_bncomp_mul_ddmatrix_winograd_even_psec(tmp_ret, tmp_mat_a, tmp_mat_b, min_dim);
#else
//	_bncomp_mul_ddmatrix_strassen_even(tmp_ret, tmp_mat_a, tmp_mat_b, min_dim);
	_bncomp_mul_ddmatrix_strassen_even_psec(tmp_ret, tmp_mat_a, tmp_mat_b, min_dim);
#endif

//	printf("tmp_ret->row_dim, col_dim: %ld, %ld\n", tmp_ret->row_dim, tmp_ret->col_dim);
//	printf("    ret->row_dim, col_dim: %ld, %ld\n", ret->row_dim, ret->col_dim);

	// substitute
	tmp_ret_index[0] = 0;
	tmp_ret_index[1] = ret->row_dim;
	tmp_ret_index[2] = 0;
	tmp_ret_index[3] = ret->col_dim;
	ret_index[0] = 0;
	ret_index[1] = ret->row_dim;
	ret_index[2] = 0;
	ret_index[3] = ret->col_dim;

	_bncomp_subst_ddmatrix_partial(ret, ret_index, tmp_ret, tmp_ret_index);

	// free
	free_ddmatrix(tmp_ret);
	free_ddmatrix(tmp_mat_a);
	free_ddmatrix(tmp_mat_b);
}

// clear counter
void _bncomp_reset_num_mul_ddmatrix_strassen(void)
{
	_bncomp_num_addsub_mul_ddmatrix_strassen = 0;
	_bncomp_num_mul_mul_ddmatrix_strassen = 0;
	_bncomp_total_num_threads_matmul_ddmatrix_strassen = 0;
}

// get counters
void _bncomp_get_num_mul_ddmatrix_strassen(long int *num_addsub, long int *num_mul)
{
	//printf("num_addsub_mul_ddmatrix_strassen: %ld\n", num_addsub_mul_ddmatrix_strassen);
	//printf("num_mul_mul_ddmatrix_strassen   : %ld\n", num_mul_mul_ddmatrix_strassen);

	if(num_addsub != NULL)
		*num_addsub = _bncomp_num_addsub_mul_ddmatrix_strassen;
	if(num_mul != NULL)
		*num_mul = _bncomp_num_mul_mul_ddmatrix_strassen;
}

// print counters
void _bncomp_print_num_mul_ddmatrix_strassen(long int *num_addsub, long int *num_mul)
{
	printf("_bncomp_num_addsub_mul_ddmatrix_strassen: %ld\n", _bncomp_num_addsub_mul_ddmatrix_strassen);
	printf("_bncomp_num_mul_mul_ddmatrix_strassen   : %ld\n", _bncomp_num_mul_mul_ddmatrix_strassen);

	_bncomp_get_num_mul_ddmatrix_strassen(num_addsub, num_mul);

}

// Fit dimension to be multiple of min_dim
void _bncomp_mul_ddmatrix_strassen(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim)
{
	int thread_num, thread_index;
	long int row_k, col_k, mid_row_k, mid_col_k;
	long int row_dim, col_dim, mid_dim;
//	DDVector diag_left, diag_right;

	thread_num = omp_get_num_threads();

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_ddmatrix_strassen)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	// normal matrix multiplication in case of ret_dim <= 4 
	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	{
		//mul_ddmatrix(ret, mat_a, mat_b);

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		_bncomp_num_mul_mul_ddmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		//mul_ddmatrix_simple(ret, mat_a, mat_b);
		_bncomp_mul_ddmatrix_simple(ret, mat_a, mat_b);
		//print_normf_ddmatrix("||ret1||_F = ", ret);

	/*	if(_bncomp_total_num_threads_matmul_ddmatrix_strassen >= MAX_TOTAL_NUM_THREADS)
		{
			_bncomp_total_num_threads_matmul_ddmatrix_strassen++;
			mul_ddmatrix(ret, mat_a, mat_b);
			_bncomp_total_num_threads_matmul_ddmatrix_strassen--;
		}
		else
		{
			_bncomp_total_num_threads_matmul_ddmatrix_strassen += thread_num;
			_bncomp_mul_ddmatrix_simple(ret, mat_a, mat_b);
			_bncomp_total_num_threads_matmul_ddmatrix_strassen -= thread_num;
		}
	*/
		return;
	}

	// scaling 
//	diag_left = init2_ddvector(row_dim, mat_a->prec);
//	diag_right = init2_ddvector(col_dim, mat_b->prec);

//	left_scaling_ddmatrix(mat_a, diag_left, NULL);
//	right_scaling_ddmatrix(mat_b, diag_right, NULL);

	// dynamic peeling in case of odd dim
	// [ A11   a12 ] [ B11   b12 ] = [ A11*B11 + a12 * b21^T   A11*b12 + a12 * b22    ]
	// [ a21^T a22 ] [ b21^T b22 ]   [ a21^T*B11 + a22 * b21^T a21^T * b12 + a22 * b22]
	if((ret->row_dim % 2 == 1) || (ret->col_dim % 2 == 1) || (mid_dim % 2 == 1))
	{
#ifdef PEELING_ONLY
		_bncomp_mul_ddmatrix_strassen_odd_peeling(ret, mat_a, mat_b, min_dim);

		/*if(_bncomp_total_num_threads_matmul_ddmatrix_strassen >= MAX_TOTAL_NUM_THREADS)
		{
			_bncomp_total_num_threads_matmul_ddmatrix_strassen++;
			mul_ddmatrix_strassen_odd_peeling(ret, mat_a, mat_b, min_dim);
			_bncomp_total_num_threads_matmul_ddmatrix_strassen--;
		}
		else
		{
			_bncomp_total_num_threads_matmul_ddmatrix_strassen += thread_num;
			_bncomp_mul_ddmatrix_strassen_odd_peeling(ret, mat_a, mat_b, min_dim);
			_bncomp_total_num_threads_matmul_ddmatrix_strassen -= thread_num;
		}
		*/
#elif PADDING_ONLY
		_bncomp_mul_ddmatrix_strassen_odd_padding(ret, mat_a, mat_b, min_dim);

		/* if(_bncomp_total_num_threads_matmul_ddmatrix_strassen >= MAX_TOTAL_NUM_THREADS)
		{
			_bncomp_total_num_threads_matmul_ddmatrix_strassen++;
			mul_ddmatrix_strassen_odd_padding(ret, mat_a, mat_b, min_dim);
			_bncomp_total_num_threads_matmul_ddmatrix_strassen--;
		}
		else
		{
			_bncomp_total_num_threads_matmul_ddmatrix_strassen += thread_num;
			_bncomp_mul_ddmatrix_strassen_odd_padding(ret, mat_a, mat_b, min_dim);
			_bncomp_total_num_threads_matmul_ddmatrix_strassen -= thread_num;
		}
		*/
#else
		row_k = (long int)floor(mylog2((double)(ret->row_dim)));
		col_k = (long int)floor(mylog2((double)(ret->col_dim)));

		mid_row_k = (long int)pow(2.0, row_k - 1) * 3;
		mid_col_k = (long int)pow(2.0, col_k - 1) * 3;

		//printf("2^%ld <= %ld <= 2^%ld\n", row_k, ret->row_dim, row_k + 1);

		// dynamic peeling
		//if(ret->row_dim < mid_row_k)
		if((ret->row_dim % min_dim) < (min_dim / 2))
		{
			//printf("peeling\n");
			_bncomp_mul_ddmatrix_strassen_odd_peeling(ret, mat_a, mat_b, min_dim);
			//mul_ddmatrix_strassen_odd_peeling(ret, mat_a, mat_b, min_dim);

			/* if(_bncomp_total_num_threads_matmul_ddmatrix_strassen >= MAX_TOTAL_NUM_THREADS)
			{
				_bncomp_total_num_threads_matmul_ddmatrix_strassen++;
				mul_ddmatrix_strassen_odd_peeling(ret, mat_a, mat_b, min_dim);
				_bncomp_total_num_threads_matmul_ddmatrix_strassen--;
			}
			else
			{
				_bncomp_total_num_threads_matmul_ddmatrix_strassen += thread_num;
				_bncomp_mul_ddmatrix_strassen_odd_peeling(ret, mat_a, mat_b, min_dim);
				_bncomp_total_num_threads_matmul_ddmatrix_strassen -= thread_num;
			}
			*/
		}
		// padding
		else
		{
			//printf("padding\n");
			_bncomp_mul_ddmatrix_strassen_odd_padding(ret, mat_a, mat_b, min_dim);
			//mul_ddmatrix_strassen_odd_padding(ret, mat_a, mat_b, min_dim);

			/* if(_bncomp_total_num_threads_matmul_ddmatrix_strassen >= MAX_TOTAL_NUM_THREADS)
			{
				_bncomp_total_num_threads_matmul_ddmatrix_strassen++;
				mul_ddmatrix_strassen_odd_padding(ret, mat_a, mat_b, min_dim);
				_bncomp_total_num_threads_matmul_ddmatrix_strassen--;
			}
			else
			{
				_bncomp_total_num_threads_matmul_ddmatrix_strassen += thread_num;
				_bncomp_mul_ddmatrix_strassen_odd_padding(ret, mat_a, mat_b, min_dim);
				_bncomp_total_num_threads_matmul_ddmatrix_strassen -= thread_num;
			}
			*/
		}
#endif

		//printf("end\n");
	}
	// normal strassen algorithm in case of even dim
	else
	{
		//printf("(%d, %d) is even -> ", ret->row_dim, ret->col_dim);
		//printf("%d is even -> ", ret->row_dim);
#ifdef USE_WINOGRAD
		//_bncomp_mul_ddmatrix_winograd_even(ret, mat_a, mat_b, min_dim);
		_bncomp_mul_ddmatrix_winograd_even_psec(ret, mat_a, mat_b, min_dim);

		/* if(_bncomp_total_num_threads_matmul_ddmatrix_strassen >= MAX_TOTAL_NUM_THREADS)
		{
			_bncomp_total_num_threads_matmul_ddmatrix_strassen++;
			mul_ddmatrix_winograd_even(ret, mat_a, mat_b, min_dim);
			_bncomp_total_num_threads_matmul_ddmatrix_strassen--;
		}
		else
		{
			_bncomp_total_num_threads_matmul_ddmatrix_strassen += thread_num;
			_bncomp_mul_ddmatrix_winograd_even(ret, mat_a, mat_b, min_dim);
			_bncomp_total_num_threads_matmul_ddmatrix_strassen -= thread_num;
		}
		*/
#else
		//_bncomp_mul_ddmatrix_strassen_even(ret, mat_a, mat_b, min_dim);
		_bncomp_mul_ddmatrix_strassen_even_psec(ret, mat_a, mat_b, min_dim);

		/* if(_bncomp_total_num_threads_matmul_ddmatrix_strassen >= MAX_TOTAL_NUM_THREADS)
		{
			_bncomp_total_num_threads_matmul_ddmatrix_strassen++;
			mul_ddmatrix_strassen_even(ret, mat_a, mat_b, min_dim);
			_bncomp_total_num_threads_matmul_ddmatrix_strassen--;
		}
		else
		{
			_bncomp_total_num_threads_matmul_ddmatrix_strassen += thread_num;
			_bncomp_mul_ddmatrix_strassen_even(ret, mat_a, mat_b, min_dim);
			_bncomp_total_num_threads_matmul_ddmatrix_strassen -= thread_num;
		}
		*/
#endif
		//printf("end\n");
	}

//	mul_ddmatrix_ddiag_mat(ret, diag_left, 0, ret, diag_right, 0);

//	free_ddvector(diag_left);
//	free_ddvector(diag_right);

}

// Strassen's Algorithm with Dynamic peeling
void _bncomp_mul_ddmatrix_strassen_odd_peeling(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim)
{
	int thread_num, thread_index;
	long int i, j, row_dim, row_dim_h, col_dim, col_dim_h, mid_dim, mid_dim_h, tmp_dim_h, tmp_dim;
	DDMatrix mat_a11, mat_b11, mat_c11, mat_tmp;
	DDVector vec_a12, vec_a21, vec_b12, vec_b21, vec_c12, vec_c21, vec_tmp12, vec_tmp21;
	static double a22[DDSIZE], b22[DDSIZE], c22[DDSIZE], tmp[DDSIZE];
	static double tmp_array[BNCOMP_MAX_NUM_THREADS][DDSIZE];

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_ddmatrix_strassen_odd_peeling)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	set0_dd(a22);
	set0_dd(b22);
	set0_dd(c22);
	set0_dd(tmp);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		_bncomp_num_mul_mul_ddmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		//mul_ddmatrix(ret, mat_a, mat_b);
		_bncomp_mul_ddmatrix_simple(ret, mat_a, mat_b);

		return;
	}

	// tmp_dim_h = mid_dim_h or col_dim_h
	tmp_dim_h = mat_b->col_dim - 1;
	if(mid_dim_h < tmp_dim_h)
		tmp_dim_h = mid_dim_h;

	// Initialize
	mat_a11 = init_ddmatrix(row_dim_h, mid_dim_h);
	mat_b11 = init_ddmatrix(mid_dim_h, col_dim_h);
	mat_c11 = init_ddmatrix(row_dim_h, col_dim_h);
	mat_tmp = init_ddmatrix(row_dim_h, col_dim_h);

	vec_a12 = NULL;
	vec_b21 = NULL;
	vec_b12 = NULL;
	vec_a21 = NULL;
	vec_c12 = init_ddvector(row_dim_h);
	vec_c21 = init_ddvector(col_dim_h);
	vec_tmp12 = init_ddvector(row_dim_h);
	vec_tmp21 = init_ddvector(col_dim_h);

	// set matrix elements to mat_a11
	#pragma omp parallel for private(j)
	for(i = 0; i < row_dim_h; i++)
	{
		for(j = 0; j < mid_dim_h; j++)
			set_ddmatrix_ij(mat_a11, i, j, get_ddmatrix_ij(mat_a, i, j));
	}

	// set matrix elements to vec_b11
	#pragma omp parallel for private(j)
	for(i = 0; i < mid_dim_h; i++)
	{
		for(j = 0; j < col_dim_h; j++)
			set_ddmatrix_ij(mat_b11, i, j, get_ddmatrix_ij(mat_b, i, j));
	}

	// set matrix elements to vec_a12 and vec_b21
	if(mid_dim_h < mid_dim)
	{
		//printf("set vec_a12, b21\n");
		vec_a12 = init_ddvector(row_dim_h);
		vec_b21 = init_ddvector(col_dim_h); // fix!: 2014-03-19 by T.Kouya
		#pragma omp parallel for
		for(i = 0; i < row_dim_h; i++)
			set_ddvector_i(vec_a12, i, get_ddmatrix_ij(mat_a, i, mat_a->col_dim - 1));

		//printf("set vec_a12\n");

		#pragma omp parallel for
		for(i = 0; i < col_dim_h; i++)
			set_ddvector_i(vec_b21, i, get_ddmatrix_ij(mat_b, mat_b->row_dim - 1, i));

		//printf("set vec_b21\n");
	}

	// set matrix elements to vec_a21
	if(row_dim_h < row_dim)
	{
		//printf("set vec_a21, a22\n");
		vec_a21 = init_ddvector(mid_dim_h);
		#pragma omp parallel for
		for(i = 0; i < mid_dim_h; i++)
			set_ddvector_i(vec_a21, i, get_ddmatrix_ij(mat_a, mat_a->row_dim - 1, i));

		set0_dd(a22);
		if(mid_dim_h < mid_dim)
			rdd_set(a22, get_ddmatrix_ij(mat_a, mat_a->row_dim - 1, mat_a->col_dim - 1));

	}

	// set matrix elements to vec_b12
	if(col_dim_h < col_dim)
	{
		//printf("set vec_a12, b22\n");
		vec_b12 = init_ddvector(mid_dim_h);

		#pragma omp parallel for
		for(i = 0; i < mid_dim_h; i++)
			set_ddvector_i(vec_b12, i, get_ddmatrix_ij(mat_b, i, mat_b->col_dim - 1));

		set0_dd(b22);
		if(mid_dim_h < mid_dim)
			rdd_set(b22, get_ddmatrix_ij(mat_b, mat_b->row_dim - 1, mat_b->col_dim - 1));
	}

	// dynamic peeling in case of odd dim
	// [ A11   a12 ] [ B11   b12 ] = [ A11*B11 + a12 * b21^T   A11*b12 + a12 * b22     ]
	// [ a21^T a22 ] [ b21^T b22 ]   [ a21^T*B11 + a22 * b21^T a21^T * b12 + a22 * b22 ]

	//printf("starting C11 = A11 * B11...\n");

	// C11 = A11 * B11
#ifdef USE_WINOGRAD
	//_bncomp_mul_ddmatrix_winograd_even(mat_c11, mat_a11, mat_b11, min_dim);
	_bncomp_mul_ddmatrix_winograd_even_psec(mat_c11, mat_a11, mat_b11, min_dim);
#else
//	_bncomp_mul_ddmatrix_strassen_even(mat_c11, mat_a11, mat_b11, min_dim);
	_bncomp_mul_ddmatrix_strassen_even_psec(mat_c11, mat_a11, mat_b11, min_dim);
#endif

	//printf("C11 = A11 * B11\n");

	// C11 += a12 * b21^T
	if((vec_a12 != NULL) && (vec_b21 != NULL))
	{
		//printf("starting C11 += a12 * b21...\n");

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;
		_bncomp_num_mul_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

		#pragma omp parallel for private(j)
		for(i = 0; i < row_dim_h; i++)
		{
			thread_index = omp_get_thread_num();
			for(j = 0; j < col_dim_h; j++)
			{
				//rdd_mul(get_ddmatrix_ij(mat_tmp, i, j), get_ddvector_i(vec_a12, i), get_ddvector_i(vec_b21, j));
				rdd_mul(tmp_array[thread_index], get_ddvector_i(vec_a12, i), get_ddvector_i(vec_b21, j));
				set_ddmatrix_ij(mat_tmp, i, j, tmp_array[thread_index]);
			}
		}
		_bncomp_add_ddmatrix(mat_c11, mat_c11, mat_tmp);
		//printf("C11 += a12 * b21...\n");
	}

	#pragma omp parallel for private(j)
	for(i = 0; i < row_dim_h; i++)
	{
		for(j = 0; j < col_dim_h; j++) 
			set_ddmatrix_ij(ret, i, j, get_ddmatrix_ij(mat_c11, i, j));
	}

	// c12 := A11 * b12
	if(vec_b12 != NULL)
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += mat_a11->row_dim * mat_a11->col_dim;
		_bncomp_num_mul_mul_ddmatrix_strassen += mat_a11->row_dim * mat_a11->col_dim;

		//printf("c12(%ld:%ld) = A11(%ld:%ld, %ld:%ld) * b12(%ld:%ld)\n", vec_c12->dim, vec_c12->real_dim, mat_a11->row_dim, mat_a11->real_row_dim, mat_a11->col_dim, mat_a11->real_col_dim, vec_b12->dim, vec_b12->real_dim);
		_bncomp_mul_ddmatrix_ddvec(vec_c12, mat_a11, vec_b12);
		//mul_ddmatrix_ddvec(vec_c12, mat_a11, vec_b12);

		//printf("c12(%ld) = A11(%ld, %ld) * b12(%ld)\n", vec_c12->dim, mat_a11->row_dim, mat_a11->col_dim, vec_b12->dim);
	}

 	// c12 += b22 * a12
 	if((vec_a12 != NULL) && (rdd_cmp_ui(b22, 0UL) != 0))
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += vec_c12->dim;
		_bncomp_num_mul_mul_ddmatrix_strassen += vec_tmp12->dim;

		_bncomp_cmul_ddvector(vec_tmp12, b22, vec_a12);
		_bncomp_add_ddvector(vec_c12, vec_c12, vec_tmp12);
	}
	//printf("c12 += b22 * a12\n");

	// Fix! 2016-08-18 by T.Kouya
	if((vec_b12 != NULL) || ((vec_a12 != NULL) && (rdd_cmp_ui(b22, 0UL) != 0)))
	{
		#pragma omp parallel for
		for(i = 0; i < row_dim_h; i++)	// Fix! 2016-08-18 by T.Kouya
			set_ddmatrix_ij(ret, i, ret->col_dim - 1, get_ddvector_i(vec_c12, i));
	}

	//printf("vec_c12\n");

	// c21 := a21^T*B11
	if(vec_a21 != NULL)
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += mat_b11->row_dim * mat_b11->col_dim;
		_bncomp_num_mul_mul_ddmatrix_strassen += mat_b11->row_dim * mat_b11->col_dim;

		//printf("t: c21(%ld:%ld) = B11(%ld:%ld, %ld:%ld) * a21(%ld:%ld)\n", vec_c21->dim, vec_c21->real_dim, mat_b11->row_dim, mat_b11->real_row_dim, mat_b11->col_dim, mat_b11->real_col_dim, vec_a21->dim, vec_a21->real_dim);
		_bncomp_mul_ddmatrixt_ddvec(vec_c21, mat_b11, vec_a21);
		//mul_ddmatrixt_ddvec(vec_c21, mat_b11, vec_a21);
		//printf("t: c21(%ld:%ld) = B11(%ld:%ld, %ld:%ld) * a21(%ld:%ld)\n", vec_c21->dim, vec_c21->real_dim, mat_b11->row_dim, mat_b11->real_row_dim, mat_b11->col_dim, mat_b11->real_col_dim, vec_a21->dim, vec_a21->real_dim);
	}
	//printf("c21 = a21^T * B11\n");

	// c21 += a22 * b21^T
	if((vec_b21 != NULL) && (rdd_cmp_ui(a22, 0UL) != 0))
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += vec_c21->dim;
		_bncomp_num_mul_mul_ddmatrix_strassen += vec_tmp21->dim;

		_bncomp_cmul_ddvector(vec_tmp21, a22, vec_b21);
		_bncomp_add_ddvector(vec_c21, vec_c21, vec_tmp21);
	}
	//printf("c21 += a22 * b21^T\n");
	if(vec_a21 != NULL)
	{
		#pragma omp parallel for
		for(i = 0; i < col_dim_h; i++)
			set_ddmatrix_ij(ret, ret->row_dim - 1, i, get_ddvector_i(vec_c21, i));
	}
	//printf("c21\n");

	// c22 := a21^T * b12
	if((vec_a21 != NULL) && (vec_b12 != NULL))
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += vec_a21->dim;
		_bncomp_num_mul_mul_ddmatrix_strassen += vec_tmp21->dim;

#ifdef __cplusplus
		_bncomp_ip_ddvector(&c22, vec_a21, vec_b12);
#else
		_bncomp_ip_ddvector(c22, vec_a21, vec_b12);
#endif // __cplusplus
	}
	//printf("c22 += a21^T * b12\n");

	// c22 += a22 * b22
	if((rdd_cmp_ui(a22, 0UL) != 0) || (rdd_cmp_ui(b22, 0UL) != 0))
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += 1;
		_bncomp_num_mul_mul_ddmatrix_strassen += 1;

		rdd_mul(tmp, a22, b22);
		rdd_add(c22, c22, tmp);
	}
	//printf("c22 += a22 * b22\n");

	if((vec_a21 != NULL) && (vec_b12 != NULL))
		set_ddmatrix_ij(ret, ret->row_dim - 1, ret->col_dim - 1, c22);

	//printf("c22\n");

	// free
	free_ddmatrix(mat_a11);
	free_ddmatrix(mat_b11);
	free_ddmatrix(mat_c11);
	//printf("free_ddmatrix a11, b11, c11\n");
	free_ddmatrix(mat_tmp);

	//printf("free_ddmatrix\n");

	if(vec_a12 != NULL)
		free_ddvector(vec_a12);

	if(vec_b21 != NULL)
		free_ddvector(vec_b21);

	if(vec_a21 != NULL)
		free_ddvector(vec_a21);

	if(vec_b12 != NULL)
		free_ddvector(vec_b12);

	//printf("free_ddvector\n");

	free_ddvector(vec_c12);
	free_ddvector(vec_c21);
	free_ddvector(vec_tmp12);
	free_ddvector(vec_tmp21);

}

// Strassen's Algorithm
void _bncomp_mul_ddmatrix_strassen_even(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim)
{
	int thread_num, thread_index;
//	long int min_dim = 4; // = 2^2
	DDMatrix mat_p[7], mat_tmp_a[7], mat_tmp_b[7], mat_tmp_c[4];
	long int row_dim_h, row_dim, col_dim_h, col_dim, mid_dim_h, mid_dim;
	long int ret_index[4], mat_tmp_a_index[4], mat_tmp_b_index[4], mat_ar_index[7][4], mat_al_index[7][4], mat_br_index[7][4], mat_bl_index[7][4], mat_c_index[4][4];
	long int i;

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_ddmatrix_strassen_even)\n", mat_a->col_dim, mat_b->row_dim);
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
		_bncomp_num_addsub_mul_ddmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		_bncomp_num_mul_mul_ddmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		//mul_ddmatrix(ret, mat_a, mat_b);
		_bncomp_mul_ddmatrix_simple(ret, mat_a, mat_b);
		return;
	}

	for(i = 0; i < 7; i++)
	{
		mat_p[i] = init_ddmatrix(row_dim_h, col_dim_h);
		mat_tmp_a[i] = init_ddmatrix(row_dim_h, mid_dim_h);
		mat_tmp_b[i] = init_ddmatrix(mid_dim_h, col_dim_h);
	}
	for(i = 0; i < 4; i++)
		mat_tmp_c[i] = init_ddmatrix(row_dim_h, col_dim_h);

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
	_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

//	add_ddmatrix_partial(mat_tmp_a[0], ret_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);
	_bncomp_add_ddmatrix_partial(mat_tmp_a[0], mat_tmp_a_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);

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
	_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

//	add_ddmatrix_partial(mat_tmp_b[0], ret_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);
	_bncomp_add_ddmatrix_partial(mat_tmp_b[0], mat_tmp_b_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);

	// P1 = tmp_a * tmp_b
	printf("P1:\n");
	_bncomp_mul_ddmatrix_strassen(mat_p[0], mat_tmp_a[0], mat_tmp_b[0], min_dim);

	print_ddmatrix(mat_p[0]);

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
	_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

//	add_ddmatrix_partial(mat_tmp_a[1], ret_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);
	_bncomp_add_ddmatrix_partial(mat_tmp_a[1], mat_tmp_a_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);

	// B11
	mat_br_index[1][0] = 0;
	mat_br_index[1][1] = mid_dim_h;
	mat_br_index[1][2] = 0;
	mat_br_index[1][3] = col_dim_h;
//	subst_ddmatrix_partial(mat_tmp_b[1], ret_index, mat_b, mat_br_index[1]);
	_bncomp_subst_ddmatrix_partial(mat_tmp_b[1], mat_tmp_b_index, mat_b, mat_br_index[1]);

	// P2 = tmp_a * tmp_b
	//printf("P2:\n");
	_bncomp_mul_ddmatrix_strassen(mat_p[1], mat_tmp_a[1], mat_tmp_b[1], min_dim);
	print_ddmatrix(mat_p[1]);

	// -------------------------------
	// P3 := A11 * (B12 - B22)
	// -------------------------------

	// A11
	mat_ar_index[2][0] = 0;
	mat_ar_index[2][1] = row_dim_h;
	mat_ar_index[2][2] = 0;
	mat_ar_index[2][3] = mid_dim_h;
//	subst_ddmatrix_partial(mat_tmp_a[2], ret_index, mat_a, mat_ar_index[2]);
	_bncomp_subst_ddmatrix_partial(mat_tmp_a[2], mat_tmp_a_index, mat_a, mat_ar_index[2]);

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
	_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

//	sub_ddmatrix_partial(mat_tmp_b[2], ret_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);
	_bncomp_sub_ddmatrix_partial(mat_tmp_b[2], mat_tmp_b_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);

	// P3 = tmp_a * tmp_b
	//printf("P3:\n");
	_bncomp_mul_ddmatrix_strassen(mat_p[2], mat_tmp_a[2], mat_tmp_b[2], min_dim);
	print_ddmatrix(mat_p[2]);

	// -------------------------------
	// P4 := A22 * (B21 - B11)
	// -------------------------------

	// A22
	mat_ar_index[3][0] = row_dim_h;
	mat_ar_index[3][1] = row_dim;
	mat_ar_index[3][2] = mid_dim_h;
	mat_ar_index[3][3] = mid_dim;

	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

//	subst_ddmatrix_partial(mat_tmp_a[3], ret_index, mat_a, mat_ar_index[3]);
	_bncomp_subst_ddmatrix_partial(mat_tmp_a[3], mat_tmp_a_index, mat_a, mat_ar_index[3]);

	// B21 - B11
	mat_br_index[3][0] = mid_dim_h;
	mat_br_index[3][1] = mid_dim;
	mat_br_index[3][2] = 0;
	mat_br_index[3][3] = col_dim_h;

	mat_bl_index[3][0] = 0;
	mat_bl_index[3][1] = mid_dim_h;
	mat_bl_index[3][2] = 0;
	mat_bl_index[3][3] = col_dim_h;
//	sub_ddmatrix_partial(mat_tmp_b[3], ret_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);
	_bncomp_sub_ddmatrix_partial(mat_tmp_b[3], mat_tmp_b_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);

	// P4 = tmp_a * tmp_b
	//printf("P4:\n");
	_bncomp_mul_ddmatrix_strassen(mat_p[3], mat_tmp_a[3], mat_tmp_b[3], min_dim);
	print_ddmatrix(mat_p[3]);

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
	_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

//	add_ddmatrix_partial(mat_tmp_a[4], ret_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);
	_bncomp_add_ddmatrix_partial(mat_tmp_a[4], mat_tmp_a_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);

	// B22
	mat_br_index[4][0] = mid_dim_h;
	mat_br_index[4][1] = mid_dim;
	mat_br_index[4][2] = col_dim_h;
	mat_br_index[4][3] = col_dim;
//	subst_ddmatrix_partial(mat_tmp_b[4], ret_index, mat_b, mat_br_index[4]);
	_bncomp_subst_ddmatrix_partial(mat_tmp_b[4], mat_tmp_b_index, mat_b, mat_br_index[4]);

	// P5 = tmp_a * tmp_b
	//printf("P5:\n");
	_bncomp_mul_ddmatrix_strassen(mat_p[4], mat_tmp_a[4], mat_tmp_b[4], min_dim);
	print_ddmatrix(mat_p[4]);

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
	_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

//	sub_ddmatrix_partial(mat_tmp_a[5], ret_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);
	_bncomp_sub_ddmatrix_partial(mat_tmp_a[5], mat_tmp_a_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);

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
	_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

//	add_ddmatrix_partial(mat_tmp_b[5], ret_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);
	_bncomp_add_ddmatrix_partial(mat_tmp_b[5], mat_tmp_b_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);

	// P6 = tmp_a * tmp_b
	//printf("P6:\n");
	_bncomp_mul_ddmatrix_strassen(mat_p[5], mat_tmp_a[5], mat_tmp_b[5], min_dim);
	print_ddmatrix(mat_p[5]);

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
	_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

//	sub_ddmatrix_partial(mat_tmp_a[6], ret_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);
	_bncomp_sub_ddmatrix_partial(mat_tmp_a[6], mat_tmp_a_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);

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
	_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

//	add_ddmatrix_partial(mat_tmp_b[6], ret_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);
	_bncomp_add_ddmatrix_partial(mat_tmp_b[6], mat_tmp_b_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);

	// P7 = tmp_a * tmp_b
	//printf("P7:\n");
	_bncomp_mul_ddmatrix_strassen(mat_p[6], mat_tmp_a[6], mat_tmp_b[6], min_dim);
	print_ddmatrix(mat_p[6]);

	// -------------------------------
	// C11 := P1 + P4 - P5 + P7
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += 3 * row_dim_h * col_dim_h;

	//add_ddmatrix(mat_tmp_c[0], mat_p[0], mat_p[3]);
	//sub_ddmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[4]);
	//add_ddmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[6]);
	_bncomp_add_ddmatrix(mat_tmp_c[0], mat_p[0], mat_p[3]);
	_bncomp_sub_ddmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[4]);
	_bncomp_add_ddmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[6]);
	mat_c_index[0][0] = 0;
	mat_c_index[0][1] = row_dim_h;
	mat_c_index[0][2] = 0;
	mat_c_index[0][3] = col_dim_h;

	//subst_ddmatrix_partial(ret, mat_c_index[0], mat_tmp_c[0], ret_index);
	_bncomp_subst_ddmatrix_partial(ret, mat_c_index[0], mat_tmp_c[0], ret_index);

	// -------------------------------
	// C12 := P3 + P5
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

	//add_ddmatrix(mat_tmp_c[1], mat_p[2], mat_p[4]);
	_bncomp_add_ddmatrix(mat_tmp_c[1], mat_p[2], mat_p[4]);
	mat_c_index[1][0] = 0;
	mat_c_index[1][1] = row_dim_h;
	mat_c_index[1][2] = col_dim_h;
	mat_c_index[1][3] = col_dim;
	//subst_ddmatrix_partial(ret, mat_c_index[1], mat_tmp_c[1], ret_index);
	_bncomp_subst_ddmatrix_partial(ret, mat_c_index[1], mat_tmp_c[1], ret_index);

	// -------------------------------
	// C21 := P2 + P4
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

	//add_ddmatrix(mat_tmp_c[2], mat_p[1], mat_p[3]);
	_bncomp_add_ddmatrix(mat_tmp_c[2], mat_p[1], mat_p[3]);
	mat_c_index[2][0] = row_dim_h;
	mat_c_index[2][1] = row_dim;
	mat_c_index[2][2] = 0;
	mat_c_index[2][3] = col_dim_h;
	//subst_ddmatrix_partial(ret, mat_c_index[2], mat_tmp_c[2], ret_index);
	_bncomp_subst_ddmatrix_partial(ret, mat_c_index[2], mat_tmp_c[2], ret_index);

	// -------------------------------
	// C22 := P1 + P3 - P2 + P6
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += 3 * row_dim_h * col_dim_h;

	//add_ddmatrix(mat_tmp_c[3], mat_p[0], mat_p[2]);
	//sub_ddmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[1]);
	//add_ddmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[5]);
	_bncomp_add_ddmatrix(mat_tmp_c[3], mat_p[0], mat_p[2]);
	_bncomp_sub_ddmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[1]);
	_bncomp_add_ddmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[5]);
	mat_c_index[3][0] = row_dim_h;
	mat_c_index[3][1] = row_dim;
	mat_c_index[3][2] = col_dim_h;
	mat_c_index[3][3] = col_dim;
	//subst_ddmatrix_partial(ret, mat_c_index[3], mat_tmp_c[3], ret_index);
	_bncomp_subst_ddmatrix_partial(ret, mat_c_index[3], mat_tmp_c[3], ret_index);

	// free
	for(i = 0; i < 7; i++)
	{
		free_ddmatrix(mat_p[i]);
		free_ddmatrix(mat_tmp_a[i]);
		free_ddmatrix(mat_tmp_b[i]);
	}
	for(i = 0; i < 4; i++)
		free_ddmatrix(mat_tmp_c[i]);
}

// Strassen's Algorithm with parallized sections
/*void _bncomp_mul_ddmatrix_strassen_even_psec(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim)
{
	mul_ddmatrix_strassen_even(ret, mat_a, mat_b, min_dim);
}
*/
void _bncomp_mul_ddmatrix_strassen_even_psec(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim)
{
	int thread_num, thread_index;
//	long int min_dim = 4; // = 2^2
	DDMatrix mat_p[7], mat_tmp_a[7], mat_tmp_b[7], mat_tmp_c[4];
	long int row_dim_h, row_dim, col_dim_h, col_dim, mid_dim_h, mid_dim;
	long int ret_index[4], mat_tmp_a_index[4], mat_tmp_b_index[4], mat_ar_index[7][4], mat_al_index[7][4], mat_br_index[7][4], mat_bl_index[7][4], mat_c_index[4][4];
	long int i;

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_ddmatrix_strassen_even)\n", mat_a->col_dim, mat_b->row_dim);
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
		_bncomp_num_addsub_mul_ddmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		_bncomp_num_mul_mul_ddmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		//mul_ddmatrix(ret, mat_a, mat_b);
		_bncomp_mul_ddmatrix_simple(ret, mat_a, mat_b);
		//print_normf_ddmatrix("||ret1||_F = ", ret);
		return;
	}

	#pragma omp parallel for
	for(i = 0; i < 7; i++)
	{
		mat_p[i] = init_ddmatrix(row_dim_h, col_dim_h);
		mat_tmp_a[i] = init_ddmatrix(row_dim_h, mid_dim_h);
		mat_tmp_b[i] = init_ddmatrix(mid_dim_h, col_dim_h);
	}
	
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
		mat_tmp_c[i] = init_ddmatrix(row_dim_h, col_dim_h);

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
//#pragma omp parallel sections //shared(mat_tmp_a, mat_tmp_b, mat_a, mat_b, mat_br_index, mat_bl_index, mat_ar_index, mat_al_index, row_dim, col_dim, mid_dim, row_dim_h, col_dim_h, mid_dim_h, mat_tmp_a_index, mat_tmp_b_index, ret_index) //num_threads(7)
#pragma omp parallel sections //num_threads(7)
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
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		add_ddmatrix_partial(mat_tmp_a[0], mat_tmp_a_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);
	//	_bncomp_add_ddmatrix_partial(mat_tmp_a[0], mat_tmp_a_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		add_ddmatrix_partial(mat_tmp_b[0], mat_tmp_b_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);
	//	_bncomp_add_ddmatrix_partial(mat_tmp_b[0], mat_tmp_b_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);

		// P1 = tmp_a * tmp_b
		//printf("P1: %d, ", omp_get_thread_num());
		_bncomp_mul_ddmatrix_strassen(mat_p[0], mat_tmp_a[0], mat_tmp_b[0], min_dim);
		//print_normf_ddmatrix("||P1||_F = ", mat_p[0]);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		add_ddmatrix_partial(mat_tmp_a[1], mat_tmp_a_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);
	//	_bncomp_add_ddmatrix_partial(mat_tmp_a[1], mat_tmp_a_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);

		// B11
		mat_br_index[1][0] = 0;
		mat_br_index[1][1] = mid_dim_h;
		mat_br_index[1][2] = 0;
		mat_br_index[1][3] = col_dim_h;
		subst_ddmatrix_partial(mat_tmp_b[1], mat_tmp_b_index, mat_b, mat_br_index[1]);
	//	_bncomp_subst_ddmatrix_partial(mat_tmp_b[1], mat_tmp_b_index, mat_b, mat_br_index[1]);

		// P2 = tmp_a * tmp_b
		//printf("P2: %d, ", omp_get_thread_num());
		_bncomp_mul_ddmatrix_strassen(mat_p[1], mat_tmp_a[1], mat_tmp_b[1], min_dim);
		//print_normf_ddmatrix("||P2||_F = ", mat_p[1]);
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
		subst_ddmatrix_partial(mat_tmp_a[2], mat_tmp_a_index, mat_a, mat_ar_index[2]);
	//	_bncomp_subst_ddmatrix_partial(mat_tmp_a[2], mat_tmp_a_index, mat_a, mat_ar_index[2]);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		sub_ddmatrix_partial(mat_tmp_b[2], mat_tmp_b_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);
	//	_bncomp_sub_ddmatrix_partial(mat_tmp_b[2], mat_tmp_b_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);

		// P3 = tmp_a * tmp_b
		//printf("P3: %d, ", omp_get_thread_num());
		_bncomp_mul_ddmatrix_strassen(mat_p[2], mat_tmp_a[2], mat_tmp_b[2], min_dim);
		//print_normf_ddmatrix("||P3||_F = ", mat_p[2]);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		subst_ddmatrix_partial(mat_tmp_a[3], mat_tmp_a_index, mat_a, mat_ar_index[3]);
	//	_bncomp_subst_ddmatrix_partial(mat_tmp_a[3], mat_tmp_a_index, mat_a, mat_ar_index[3]);

		// B21 - B11
		mat_br_index[3][0] = mid_dim_h;
		mat_br_index[3][1] = mid_dim;
		mat_br_index[3][2] = 0;
		mat_br_index[3][3] = col_dim_h;

		mat_bl_index[3][0] = 0;
		mat_bl_index[3][1] = mid_dim_h;
		mat_bl_index[3][2] = 0;
		mat_bl_index[3][3] = col_dim_h;
		sub_ddmatrix_partial(mat_tmp_b[3], mat_tmp_b_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);
	//	_bncomp_sub_ddmatrix_partial(mat_tmp_b[3], mat_tmp_b_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);

		// P4 = tmp_a * tmp_b
		//printf("P4: %d, ", omp_get_thread_num());
		_bncomp_mul_ddmatrix_strassen(mat_p[3], mat_tmp_a[3], mat_tmp_b[3], min_dim);
		//print_normf_ddmatrix("||P4||_F = ", mat_p[3]);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		add_ddmatrix_partial(mat_tmp_a[4], mat_tmp_a_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);
	//	_bncomp_add_ddmatrix_partial(mat_tmp_a[4], mat_tmp_a_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);

		// B22
		mat_br_index[4][0] = mid_dim_h;
		mat_br_index[4][1] = mid_dim;
		mat_br_index[4][2] = col_dim_h;
		mat_br_index[4][3] = col_dim;
		subst_ddmatrix_partial(mat_tmp_b[4], mat_tmp_b_index, mat_b, mat_br_index[4]);
	//	_bncomp_subst_ddmatrix_partial(mat_tmp_b[4], mat_tmp_b_index, mat_b, mat_br_index[4]);

		// P5 = tmp_a * tmp_b
		//printf("P5: %d, ", omp_get_thread_num());
		_bncomp_mul_ddmatrix_strassen(mat_p[4], mat_tmp_a[4], mat_tmp_b[4], min_dim);
		//print_normf_ddmatrix("||P5||_F = ", mat_p[4]);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		sub_ddmatrix_partial(mat_tmp_a[5], mat_tmp_a_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);
	//	_bncomp_sub_ddmatrix_partial(mat_tmp_a[5], mat_tmp_a_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		add_ddmatrix_partial(mat_tmp_b[5], mat_tmp_b_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);
	//	_bncomp_add_ddmatrix_partial(mat_tmp_b[5], mat_tmp_b_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);

		// P6 = tmp_a * tmp_b
		//printf("P6: %d, ", omp_get_thread_num());
		_bncomp_mul_ddmatrix_strassen(mat_p[5], mat_tmp_a[5], mat_tmp_b[5], min_dim);
		//print_normf_ddmatrix("||P6||_F = ", mat_p[5]);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		sub_ddmatrix_partial(mat_tmp_a[6], mat_tmp_a_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);
	//	_bncomp_sub_ddmatrix_partial(mat_tmp_a[6], mat_tmp_a_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		add_ddmatrix_partial(mat_tmp_b[6], mat_tmp_b_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);
	//	_bncomp_add_ddmatrix_partial(mat_tmp_b[6], mat_tmp_b_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);

		// P7 = tmp_a * tmp_b
		//printf("P7: %d", omp_get_thread_num());
		_bncomp_mul_ddmatrix_strassen(mat_p[6], mat_tmp_a[6], mat_tmp_b[6], min_dim);
		//print_normf_ddmatrix("||P7||_F = ", mat_p[6]);

	}
} // pragma omp parallel sections

//printf("\n");

//#pragma omp barrier

//#pragma omp sections
#pragma omp parallel sections
{
	// -------------------------------
	// C11 := P1 + P4 - P5 + P7
	// -------------------------------
	#pragma omp section
	{

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += 3 * row_dim_h * col_dim_h;

		add_ddmatrix(mat_tmp_c[0], mat_p[0], mat_p[3]);

		sub_ddmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[4]);
		add_ddmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[6]);
		//_bncomp_add_ddmatrix(mat_tmp_c[0], mat_p[0], mat_p[3]);
		//_bncomp_sub_ddmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[4]);
		//_bncomp_add_ddmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[6]);
		mat_c_index[0][0] = 0;
		mat_c_index[0][1] = row_dim_h;
		mat_c_index[0][2] = 0;
		mat_c_index[0][3] = col_dim_h;

		subst_ddmatrix_partial(ret, mat_c_index[0], mat_tmp_c[0], ret_index);
		//_bncomp_subst_ddmatrix_partial(ret, mat_c_index[0], mat_tmp_c[0], ret_index);
		//print_normf_ddmatrix("||c11||_F = ", ret);
	}

	// -------------------------------
	// C12 := P3 + P5
	// -------------------------------
	#pragma omp section
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

		add_ddmatrix(mat_tmp_c[1], mat_p[2], mat_p[4]);
		//_bncomp_add_ddmatrix(mat_tmp_c[1], mat_p[2], mat_p[4]);
		mat_c_index[1][0] = 0;
		mat_c_index[1][1] = row_dim_h;
		mat_c_index[1][2] = col_dim_h;
		mat_c_index[1][3] = col_dim;
		subst_ddmatrix_partial(ret, mat_c_index[1], mat_tmp_c[1], ret_index);
		//_bncomp_subst_ddmatrix_partial(ret, mat_c_index[1], mat_tmp_c[1], ret_index);
		//print_normf_ddmatrix("||c12||_F = ", ret);
	}

	// -------------------------------
	// C21 := P2 + P4
	// -------------------------------
	#pragma omp section
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

		add_ddmatrix(mat_tmp_c[2], mat_p[1], mat_p[3]);
		//_bncomp_add_ddmatrix(mat_tmp_c[2], mat_p[1], mat_p[3]);
		mat_c_index[2][0] = row_dim_h;
		mat_c_index[2][1] = row_dim;
		mat_c_index[2][2] = 0;
		mat_c_index[2][3] = col_dim_h;
		subst_ddmatrix_partial(ret, mat_c_index[2], mat_tmp_c[2], ret_index);
		//_bncomp_subst_ddmatrix_partial(ret, mat_c_index[2], mat_tmp_c[2], ret_index);
		//print_normf_ddmatrix("||c21||_F = ", ret);
	}

	// -------------------------------
	// C22 := P1 + P3 - P2 + P6
	// -------------------------------
	#pragma omp section
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += 3 * row_dim_h * col_dim_h;

		add_ddmatrix(mat_tmp_c[3], mat_p[0], mat_p[2]);
		sub_ddmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[1]);
		add_ddmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[5]);
		//_bncomp_add_ddmatrix(mat_tmp_c[3], mat_p[0], mat_p[2]);
		//_bncomp_sub_ddmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[1]);
		//_bncomp_add_ddmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[5]);
		mat_c_index[3][0] = row_dim_h;
		mat_c_index[3][1] = row_dim;
		mat_c_index[3][2] = col_dim_h;
		mat_c_index[3][3] = col_dim;
		subst_ddmatrix_partial(ret, mat_c_index[3], mat_tmp_c[3], ret_index);
		//_bncomp_subst_ddmatrix_partial(ret, mat_c_index[3], mat_tmp_c[3], ret_index);
		//print_normf_ddmatrix("||c22||_F = ", ret);
	}

} // pragma omp parallel sections

	// free
	#pragma omp parallel for
	for(i = 0; i < 7; i++)
	{
		free_ddmatrix(mat_p[i]);
		free_ddmatrix(mat_tmp_a[i]);
		free_ddmatrix(mat_tmp_b[i]);
	}
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
		free_ddmatrix(mat_tmp_c[i]);

//#pragma omp barrier

}

// Winograd Variant of Strassen's Algorithm
void _bncomp_mul_ddmatrix_winograd_even(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim)
{
	int thread_num, thread_index;
//	long int min_dim = 4; // = 2^2
	DDMatrix mat_s[8], mat_m[7], mat_t[2], mat_tmp_a[4], mat_tmp_b[4], mat_tmp_c[4];
	long int row_dim_h, row_dim, col_dim_h, col_dim, mid_dim, mid_dim_h;
	long int ret_index[4], mat_tmp_a_index[4], mat_tmp_b_index[4];
	long int mat_a_index[4][4], mat_b_index[4][4], mat_c_index[4][4];
	long int i;

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_ddmatrix_winograd_even)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	row_dim_h = row_dim / 2;
	col_dim_h = col_dim / 2;
	mid_dim_h = mid_dim / 2;

	// normal matrix multiplication in case of ret_dim <= 4 
//	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	if((ret->row_dim <= min_dim) || (ret->col_dim <= min_dim) || (mid_dim <= min_dim))
	{
		mul_ddmatrix_simple(ret, mat_a, mat_b);
		//_bncomp_mul_ddmatrix_simple(ret, mat_a, mat_b);

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		_bncomp_num_mul_mul_ddmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		return;
	}

	for(i = 0; i < 4; i++)
	{
		mat_s[i] = init_ddmatrix(row_dim_h, mid_dim_h);
		mat_tmp_a[i] = init_ddmatrix(row_dim_h, mid_dim_h);

		mat_s[i + 4] = init_ddmatrix(mid_dim_h, col_dim_h);
		mat_tmp_b[i] = init_ddmatrix(mid_dim_h, col_dim_h);

		mat_tmp_c[i] = init_ddmatrix(row_dim_h, col_dim_h);
	}
	for(i = 0; i < 7; i++)
		mat_m[i] = init_ddmatrix(row_dim_h, col_dim_h);

	mat_t[0] = init_ddmatrix(row_dim_h, col_dim_h);
	mat_t[1] = init_ddmatrix(row_dim_h, col_dim_h);

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
		_bncomp_subst_ddmatrix_partial(mat_tmp_a[i], mat_tmp_a_index, mat_a, mat_a_index[i]);
		_bncomp_subst_ddmatrix_partial(mat_tmp_b[i], mat_tmp_b_index, mat_b, mat_b_index[i]);
	}
//	printf("subst a, b...\n");

	// -------------------------------
	// S1 := A21 + A22
	//--------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

	_bncomp_add_ddmatrix(mat_s[0], mat_tmp_a[2], mat_tmp_a[3]);

	// -------------------------------
	// S2 := S1 - A11
	//--------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

	_bncomp_sub_ddmatrix(mat_s[1], mat_s[0], mat_tmp_a[0]);

	// -------------------------------
	// S3 := A11 - A21
	//--------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

	_bncomp_sub_ddmatrix(mat_s[2], mat_tmp_a[0], mat_tmp_a[2]);

	// -------------------------------
	// S4 := A12 - S2
	//--------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

	_bncomp_sub_ddmatrix(mat_s[3], mat_tmp_a[1], mat_s[1]);

	// -------------------------------
	// S5 := B12 - B11
	//--------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

	_bncomp_sub_ddmatrix(mat_s[4], mat_tmp_b[1], mat_tmp_b[0]);

	// -------------------------------
	// S6 := B22 - S5
	//--------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

	_bncomp_sub_ddmatrix(mat_s[5], mat_tmp_b[3], mat_s[4]);

	// -------------------------------
	// S7 := B22 - B12
	//--------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

	_bncomp_sub_ddmatrix(mat_s[6], mat_tmp_b[3], mat_tmp_b[1]);

	// -------------------------------
	// S8 := S6 - B21
	//--------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

	_bncomp_sub_ddmatrix(mat_s[7], mat_s[5], mat_tmp_b[2]);

//	printf("s...\n");

	// -------------------------------
	// M1 := S2 * S6
	// -------------------------------
	_bncomp_mul_ddmatrix_strassen(mat_m[0], mat_s[1], mat_s[5], min_dim);

	// -------------------------------
	// M2 := A11 * B11
	// -------------------------------
	_bncomp_mul_ddmatrix_strassen(mat_m[1], mat_tmp_a[0], mat_tmp_b[0], min_dim);

	// -------------------------------
	// M3 := A12 * B21
	// -------------------------------
	_bncomp_mul_ddmatrix_strassen(mat_m[2], mat_tmp_a[1], mat_tmp_b[2], min_dim);

	// -------------------------------
	// M4 := S3 * S7
	// -------------------------------
	_bncomp_mul_ddmatrix_strassen(mat_m[3], mat_s[2], mat_s[6], min_dim);

	// -------------------------------
	// M5 := S1 * S5
	// -------------------------------
	_bncomp_mul_ddmatrix_strassen(mat_m[4], mat_s[0], mat_s[4], min_dim);

	// -------------------------------
	// M6 := S4 * B22
	// -------------------------------
	_bncomp_mul_ddmatrix_strassen(mat_m[5], mat_s[3], mat_tmp_b[3], min_dim);

	// -------------------------------
	// M7 := A22 * S8
	// -------------------------------
	_bncomp_mul_ddmatrix_strassen(mat_m[6], mat_tmp_a[3], mat_s[7], min_dim);

//	printf("m...\n");

	// -------------------------------
	// T1 := M1 + M2
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

	_bncomp_add_ddmatrix(mat_t[0], mat_m[0], mat_m[1]);

	// -------------------------------
	// T2 := T1 + M4
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

	_bncomp_add_ddmatrix(mat_t[1], mat_t[0], mat_m[3]);

//	printf("t...\n");

	// -------------------------------
	// C11 := M2 + M3
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

	_bncomp_add_ddmatrix(mat_tmp_c[0], mat_m[1], mat_m[2]);

	// -------------------------------
	// C12 := T1 + M5 + M6
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += 2 * row_dim_h * col_dim_h;

	_bncomp_add_ddmatrix(mat_tmp_c[1], mat_t[0], mat_m[4]);
	_bncomp_add_ddmatrix(mat_tmp_c[1], mat_tmp_c[1], mat_m[5]);

	// -------------------------------
	// C21 := T2 - M7
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

	_bncomp_sub_ddmatrix(mat_tmp_c[2], mat_t[1], mat_m[6]);

	// -------------------------------
	// C22 := T2 + M5
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

	_bncomp_add_ddmatrix(mat_tmp_c[3], mat_t[1], mat_m[4]);

//	printf("c...\n");

	// -------------------------------
	// RET := [C11 C12]
	//        [C21 C22]
	// -------------------------------
	for(i = 0; i < 4; i++)
		_bncomp_subst_ddmatrix_partial(ret, mat_c_index[i], mat_tmp_c[i], ret_index);

//	printf("set...\n");


	// free
	for(i = 0; i < 4; i++)
	{
		free_ddmatrix(mat_s[i]);
		free_ddmatrix(mat_tmp_a[i]);
		free_ddmatrix(mat_s[i + 4]);
		free_ddmatrix(mat_tmp_b[i]);
		free_ddmatrix(mat_tmp_c[i]);
	}
	for(i = 0; i < 7; i++)
		free_ddmatrix(mat_m[i]);
	
	free_ddmatrix(mat_t[0]);
	free_ddmatrix(mat_t[1]);
}

// Winograd Variant of Strassen's Algorithm with parallelied sections
void _bncomp_mul_ddmatrix_winograd_even_psec(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim)
{
	int thread_num, thread_index;
//	long int min_dim = 4; // = 2^2
	DDMatrix mat_s[8], mat_m[7], mat_t[2], mat_tmp_a[4], mat_tmp_b[4], mat_tmp_c[4];
	long int row_dim_h, row_dim, col_dim_h, col_dim, mid_dim, mid_dim_h;
	long int ret_index[4], mat_tmp_a_index[4], mat_tmp_b_index[4];
	long int mat_a_index[4][4], mat_b_index[4][4], mat_c_index[4][4];
	long int i;

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_ddmatrix_winograd_even)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	row_dim_h = row_dim / 2;
	col_dim_h = col_dim / 2;
	mid_dim_h = mid_dim / 2;

	// normal matrix multiplication in case of ret_dim <= 4 
//	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	if((ret->row_dim <= min_dim) || (ret->col_dim <= min_dim) || (mid_dim <= min_dim))
	{
		mul_ddmatrix_simple(ret, mat_a, mat_b);
		//_bncomp_mul_ddmatrix_simple(ret, mat_a, mat_b);

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		_bncomp_num_mul_mul_ddmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		return;
	}

	#pragma omp parallel for
	for(i = 0; i < 4; i++)
	{
		mat_s[i] = init_ddmatrix(row_dim_h, mid_dim_h);
		mat_tmp_a[i] = init_ddmatrix(row_dim_h, mid_dim_h);

		mat_s[i + 4] = init_ddmatrix(mid_dim_h, col_dim_h);
		mat_tmp_b[i] = init_ddmatrix(mid_dim_h, col_dim_h);

		mat_tmp_c[i] = init_ddmatrix(row_dim_h, col_dim_h);
	}
	#pragma omp parallel for
	for(i = 0; i < 7; i++)
		mat_m[i] = init_ddmatrix(row_dim_h, col_dim_h);

	mat_t[0] = init_ddmatrix(row_dim_h, col_dim_h);
	mat_t[1] = init_ddmatrix(row_dim_h, col_dim_h);

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
		subst_ddmatrix_partial(mat_tmp_a[i], mat_tmp_a_index, mat_a, mat_a_index[i]);
		subst_ddmatrix_partial(mat_tmp_b[i], mat_tmp_b_index, mat_b, mat_b_index[i]);
		//_bncomp_subst_ddmatrix_partial(mat_tmp_a[i], mat_tmp_a_index, mat_a, mat_a_index[i]);
		//_bncomp_subst_ddmatrix_partial(mat_tmp_b[i], mat_tmp_b_index, mat_b, mat_b_index[i]);
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
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		add_ddmatrix(mat_s[0], mat_tmp_a[2], mat_tmp_a[3]);
		//_bncomp_add_ddmatrix(mat_s[0], mat_tmp_a[2], mat_tmp_a[3]);
	}

	#pragma omp section
	{
		// -------------------------------
		// S3 := A11 - A21
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		sub_ddmatrix(mat_s[2], mat_tmp_a[0], mat_tmp_a[2]);
		//_bncomp_sub_ddmatrix(mat_s[2], mat_tmp_a[0], mat_tmp_a[2]);
	}

	#pragma omp section
	{
		// -------------------------------
		// S5 := B12 - B11
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		sub_ddmatrix(mat_s[4], mat_tmp_b[1], mat_tmp_b[0]);
		//_bncomp_sub_ddmatrix(mat_s[4], mat_tmp_b[1], mat_tmp_b[0]);
	}

	#pragma omp section
	{
		// -------------------------------
		// S7 := B22 - B12
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		sub_ddmatrix(mat_s[6], mat_tmp_b[3], mat_tmp_b[1]);
		//_bncomp_sub_ddmatrix(mat_s[6], mat_tmp_b[3], mat_tmp_b[1]);
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
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		sub_ddmatrix(mat_s[1], mat_s[0], mat_tmp_a[0]);
		//_bncomp_sub_ddmatrix(mat_s[1], mat_s[0], mat_tmp_a[0]);

		// -------------------------------
		// S4 := A12 - S2
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		sub_ddmatrix(mat_s[3], mat_tmp_a[1], mat_s[1]);
		//_bncomp_sub_ddmatrix(mat_s[3], mat_tmp_a[1], mat_s[1]);
	}

	#pragma omp section
	{
		// -------------------------------
		// S6 := B22 - S5
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		sub_ddmatrix(mat_s[5], mat_tmp_b[3], mat_s[4]);
		//_bncomp_sub_ddmatrix(mat_s[5], mat_tmp_b[3], mat_s[4]);

		// -------------------------------
		// S8 := S6 - B21
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		sub_ddmatrix(mat_s[7], mat_s[5], mat_tmp_b[2]);
		//_bncomp_sub_ddmatrix(mat_s[7], mat_s[5], mat_tmp_b[2]);
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
		_bncomp_mul_ddmatrix_strassen(mat_m[0], mat_s[1], mat_s[5], min_dim);
	}

	#pragma omp section
	{
		// -------------------------------
		// M2 := A11 * B11
		// -------------------------------
		_bncomp_mul_ddmatrix_strassen(mat_m[1], mat_tmp_a[0], mat_tmp_b[0], min_dim);
	}

	#pragma omp section
	{
		// -------------------------------
		// M3 := A12 * B21
		// -------------------------------
		_bncomp_mul_ddmatrix_strassen(mat_m[2], mat_tmp_a[1], mat_tmp_b[2], min_dim);
	}

	#pragma omp section
	{
		// -------------------------------
		// M4 := S3 * S7
		// -------------------------------
		_bncomp_mul_ddmatrix_strassen(mat_m[3], mat_s[2], mat_s[6], min_dim);
	}

	#pragma omp section
	{
		// -------------------------------
		// M5 := S1 * S5
		// -------------------------------
		_bncomp_mul_ddmatrix_strassen(mat_m[4], mat_s[0], mat_s[4], min_dim);
	}

	#pragma omp section
	{
		// -------------------------------
		// M6 := S4 * B22
		// -------------------------------
		_bncomp_mul_ddmatrix_strassen(mat_m[5], mat_s[3], mat_tmp_b[3], min_dim);
	}

	#pragma omp section
	{
		// -------------------------------
		// M7 := A22 * S8
		// -------------------------------
		_bncomp_mul_ddmatrix_strassen(mat_m[6], mat_tmp_a[3], mat_s[7], min_dim);
	}
} // pragma omp parallel sections

//	printf("m...\n");

	// -------------------------------
	// T1 := M1 + M2
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

	//add_ddmatrix(mat_t[0], mat_m[0], mat_m[1]);
	_bncomp_add_ddmatrix(mat_t[0], mat_m[0], mat_m[1]);

	// -------------------------------
	// T2 := T1 + M4
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

	//add_ddmatrix(mat_t[1], mat_t[0], mat_m[3]);
	_bncomp_add_ddmatrix(mat_t[1], mat_t[0], mat_m[3]);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

		add_ddmatrix(mat_tmp_c[0], mat_m[1], mat_m[2]);
		//_bncomp_add_ddmatrix(mat_tmp_c[0], mat_m[1], mat_m[2]);
	}

	#pragma omp section
	{
		// -------------------------------
		// C12 := T1 + M5 + M6
		// -------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += 2 * row_dim_h * col_dim_h;

		add_ddmatrix(mat_tmp_c[1], mat_t[0], mat_m[4]);
		add_ddmatrix(mat_tmp_c[1], mat_tmp_c[1], mat_m[5]);
		//_bncomp_add_ddmatrix(mat_tmp_c[1], mat_t[0], mat_m[4]);
		//_bncomp_add_ddmatrix(mat_tmp_c[1], mat_tmp_c[1], mat_m[5]);
	}

	#pragma omp section
	{
		// -------------------------------
		// C21 := T2 - M7
		// -------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

		sub_ddmatrix(mat_tmp_c[2], mat_t[1], mat_m[6]);
		//_bncomp_sub_ddmatrix(mat_tmp_c[2], mat_t[1], mat_m[6]);
	}

	#pragma omp section
	{
		// -------------------------------
		// C22 := T2 + M5
		// -------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

		add_ddmatrix(mat_tmp_c[3], mat_t[1], mat_m[4]);
		//_bncomp_add_ddmatrix(mat_tmp_c[3], mat_t[1], mat_m[4]);
	}
} // pragma omp parallel sections

//	printf("c...\n");

	// -------------------------------
	// RET := [C11 C12]
	//        [C21 C22]
	// -------------------------------
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
	{
		subst_ddmatrix_partial(ret, mat_c_index[i], mat_tmp_c[i], ret_index);
		//_bncomp_subst_ddmatrix_partial(ret, mat_c_index[i], mat_tmp_c[i], ret_index);
	}

//	printf("set...\n");


	// free
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
	{
		free_ddmatrix(mat_s[i]);
		free_ddmatrix(mat_tmp_a[i]);
		free_ddmatrix(mat_s[i + 4]);
		free_ddmatrix(mat_tmp_b[i]);
		free_ddmatrix(mat_tmp_c[i]);
	}

	#pragma omp parallel for
	for(i = 0; i < 7; i++)
		free_ddmatrix(mat_m[i]);
	
	free_ddmatrix(mat_t[0]);
	free_ddmatrix(mat_t[1]);
}


// Matrix multiplicaiton with Strassen's algorithm (Nonrecursive version in the area of Strassen's algorithms)
void _bncomp_mul_ddmatrix_strassen_nonrec(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim)
{
	int thread_num, thread_index;
	int row_padding_flag = 0, col_padding_flag = 0, mid_padding_flag = 0;
	long i, j, k, row_dim, col_dim, mid_dim, max_dim;
	long int small_row_dim, small_col_dim, small_mid_dim;
	long int num_div_row, num_div_col, num_div_mid;
	long int max_num_div, min_num_div, strassen_num_div, strassen_pow;
	DDMatrix a11, a12, a21, a22;
	DDMatrix b11, b12, b21, b22;
	DDMatrix ret11, ret12, ret21, ret22, tmp_mat;
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
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(_bncomp_mul_ddmatrix_strassen_nonrec)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	// normal matrix multiplication in case of ret_dim <= 4 
	//if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	if((ret->row_dim <= min_dim) || (ret->col_dim <= min_dim) || (mid_dim <= min_dim))
	{
		_bncomp_mul_ddmatrix_simple(ret, mat_a, mat_b);
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
		//_bncomp_mul_ddmatrix_block(ret, mat_a, mat_b, min_dim);
		_bncomp_mul_ddmatrix_simple(ret, mat_a, mat_b);
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
	a11 = init_ddmatrix(a11_row_dim, a11_col_dim);
	a11_index[0] = mat_a_index[0][0] = 0;
	a11_index[1] = mat_a_index[0][1] = a11_row_dim;
	a11_index[2] = mat_a_index[0][2] = 0;
	a11_index[3] = mat_a_index[0][3] = a11_col_dim;
	_bncomp_subst_ddmatrix_partial_checked(a11, a11_index, mat_a, mat_a_index[0]);

	a12 = NULL;
	if(a12_col_dim > 0)
	{
		a12 = init_ddmatrix(a12_row_dim, a12_col_dim);
		a12_index[0] = 0;
		a12_index[1] = a12_row_dim;
		a12_index[2] = 0;
		a12_index[3] = a12_col_dim;
		mat_a_index[1][0] = 0;
		mat_a_index[1][1] = a12_row_dim;
		mat_a_index[1][2] = a11_col_dim;
		mat_a_index[1][3] = a11_col_dim + a12_col_dim;
		_bncomp_subst_ddmatrix_partial_checked(a12, a12_index, mat_a, mat_a_index[1]);
	}

	a21 = NULL;
	if(a21_row_dim > 0)
	{
		a21 = init_ddmatrix(a21_row_dim, a21_col_dim);
		a21_index[0] = 0;
		a21_index[1] = a21_row_dim;
		a21_index[2] = 0;
		a21_index[3] = a21_col_dim;
		mat_a_index[2][0] = a11_row_dim;
		mat_a_index[2][1] = a11_row_dim + a21_row_dim;
		mat_a_index[2][2] = 0;
		mat_a_index[2][3] = a21_col_dim;
		_bncomp_subst_ddmatrix_partial_checked(a21, a21_index, mat_a, mat_a_index[2]);
	}

	a22 = NULL;
	if((a22_row_dim > 0) && (a22_col_dim > 0))
	{
		a22 = init_ddmatrix(a22_row_dim, a22_col_dim);
		a22_index[0] = 0;
		a22_index[1] = a22_row_dim;
		a22_index[2] = 0;
		a22_index[3] = a22_col_dim;
		mat_a_index[3][0] = a11_row_dim;
		mat_a_index[3][1] = a11_row_dim + a22_row_dim;
		mat_a_index[3][2] = a11_col_dim;
		mat_a_index[3][3] = a11_col_dim + a22_col_dim;
		_bncomp_subst_ddmatrix_partial_checked(a22, a22_index, mat_a, mat_a_index[3]);
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
	b11 = init_ddmatrix(b11_row_dim, b11_col_dim);
	b11_index[0] = mat_b_index[0][0] = 0;
	b11_index[1] = mat_b_index[0][1] = b11_row_dim;
	b11_index[2] = mat_b_index[0][2] = 0;
	b11_index[3] = mat_b_index[0][3] = b11_col_dim;
	_bncomp_subst_ddmatrix_partial_checked(b11, b11_index, mat_b, mat_b_index[0]);

	b12 = NULL;
	if(b12_col_dim > 0)
	{
		b12 = init_ddmatrix(b12_row_dim, b12_col_dim);
		b12_index[0] = 0;
		b12_index[1] = b12_row_dim;
		b12_index[2] = 0;
		b12_index[3] = b12_col_dim;
		mat_b_index[1][0] = 0;
		mat_b_index[1][1] = b12_row_dim;
		mat_b_index[1][2] = b11_col_dim;
		mat_b_index[1][3] = b11_col_dim + b12_col_dim;
		_bncomp_subst_ddmatrix_partial_checked(b12, b12_index, mat_b, mat_b_index[1]);
	}

	b21 = NULL;
	if(b21_row_dim > 0)
	{
		b21 = init_ddmatrix(b21_row_dim, b21_col_dim);
		b21_index[0] = 0;
		b21_index[1] = b21_row_dim;
		b21_index[2] = 0;
		b21_index[3] = b21_col_dim;
		mat_b_index[2][0] = b11_row_dim;
		mat_b_index[2][1] = b11_row_dim + b21_row_dim;
		mat_b_index[2][2] = 0;
		mat_b_index[2][3] = b21_col_dim;
		_bncomp_subst_ddmatrix_partial_checked(b21, b21_index, mat_b, mat_b_index[2]);
	}

	b22 = NULL;
	if((b22_row_dim > 0) && (b22_col_dim > 0))
	{
		b22 = init_ddmatrix(b22_row_dim, b22_col_dim);
		b22_index[0] = 0;
		b22_index[1] = b22_row_dim;
		b22_index[2] = 0;
		b22_index[3] = b22_col_dim;
		mat_b_index[3][0] = b11_row_dim;
		mat_b_index[3][1] = b11_row_dim + b22_row_dim;
		mat_b_index[3][2] = b11_col_dim;
		mat_b_index[3][3] = b11_col_dim + b22_col_dim;
		_bncomp_subst_ddmatrix_partial_checked(b22, b22_index, mat_b, mat_b_index[3]);
	}	

	// collect [ret11 ret12] -> ret
	//         [ret21 ret22]

	// ret11 := a11 * b11 + a12 * b21
	//printf("ret11\n");
	ret11_row_dim = strassen_num_div * small_row_dim;
	ret11_col_dim = strassen_num_div * small_col_dim;

	ret11 = init_ddmatrix(ret11_row_dim, ret11_col_dim);

	// Strassen Area
	// a11 * b11
#ifdef USE_WINOGRAD
//	_bncomp_mul_ddmatrix_winograd_even(ret11, a11, b11, min_dim);
	_bncomp_mul_ddmatrix_winograd_even2(ret11, a11, b11, min_dim);
#else
	//#pragma omp parallel
//	_bncomp_mul_ddmatrix_strassen_even3(ret11, a11, b11, min_dim, 0);
	_bncomp_mul_ddmatrix_strassen_even2(ret11, a11, b11, min_dim, 0);
//	_bncomp_mul_ddmatrix_strassen_even(ret11, a11, b11, min_dim);
#endif
	//_bncomp_mul_ddmatrix_block(ret11, a11, b11, min_dim);
	//_bncomp_mul_ddmatrix_simple(ret11, a11, b11);
	
	// ret11 += a12 * b21
	if((a12 != NULL) && (b21 != NULL))
	{
		tmp_mat = init_ddmatrix(ret11_row_dim, ret11_col_dim);

		//_bncomp_mul_ddmatrix_block(tmp_mat, a12, b21, min_dim);
		_bncomp_mul_ddmatrix_simple(tmp_mat, a12, b21);
		_bncomp_add_ddmatrix(ret11, ret11, tmp_mat);

		free_ddmatrix(tmp_mat);
	}

	// ret := ret11
	ret11_index[0] = ret_index[0] = 0;
	ret11_index[1] = ret_index[1] = ret11_row_dim;
	ret11_index[2] = ret_index[2] = 0;
	ret11_index[3] = ret_index[3] = ret11_col_dim;
	_bncomp_subst_ddmatrix_partial_checked(ret, ret_index, ret11, ret11_index);

	free_ddmatrix(ret11);

	// ret12 := a11 * b12 + a12 * b22
	//printf("ret12\n");
	ret12_row_dim = strassen_num_div * small_row_dim;
	ret12_col_dim = (num_div_col - strassen_num_div) * small_col_dim;

	ret12 = NULL;
	if(ret12_col_dim > 0)
	{
		ret12 = init_ddmatrix(ret12_row_dim, ret12_col_dim);

		// a11 * b12
		if(b12 != NULL)
			_bncomp_mul_ddmatrix_simple(ret12, a11, b12);
		//	_bncomp_mul_ddmatrix_block(ret12, a11, b12, min_dim);

		// ret12 += a12 * b22
		if((a12 != NULL) && (b22 != NULL))
		{
			tmp_mat = init_ddmatrix(ret12_row_dim, ret12_col_dim);

		//	_bncomp_mul_ddmatrix_block(tmp_mat, a12, b22, min_dim);
			_bncomp_mul_ddmatrix_simple(tmp_mat, a12, b22);
			_bncomp_add_ddmatrix(ret12, ret12, tmp_mat);

			free_ddmatrix(tmp_mat);
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
		_bncomp_subst_ddmatrix_partial_checked(ret, ret_index, ret12, ret12_index);

		free_ddmatrix(ret12);
	}

	// ret21 := a21 * b11 + a22 * b21
	//printf("ret21\n");
	ret21_row_dim = (num_div_row - strassen_num_div) * small_row_dim;
	ret21_col_dim = strassen_num_div * small_col_dim;

	ret21 = NULL;
	if(ret21_row_dim > 0)
	{
		ret21 = init_ddmatrix(ret21_row_dim, ret21_col_dim);

		// a21 * b11
		if(a21 != NULL)
			_bncomp_mul_ddmatrix_simple(ret21, a21, b11);
		//	_bncomp_mul_ddmatrix_block(ret21, a21, b11, min_dim);

		// ret21 += a22 * b21
		if((a22 != NULL) && (b21 != NULL))
		{
			tmp_mat = init_ddmatrix(ret21_row_dim, ret21_col_dim);
			
			//_bncomp_mul_ddmatrix_block(tmp_mat, a22, b21, min_dim);
			_bncomp_mul_ddmatrix_simple(tmp_mat, a22, b21);
			_bncomp_add_ddmatrix(ret21, ret21, tmp_mat);

			free_ddmatrix(tmp_mat);
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
		_bncomp_subst_ddmatrix_partial_checked(ret, ret_index, ret21, ret21_index);

		free_ddmatrix(ret21);
	}

	// ret22 := a21 * b12 + a22 * b22
	//printf("ret22\n");
	ret22_row_dim = (num_div_row - strassen_num_div) * small_row_dim;
	ret22_col_dim = (num_div_col - strassen_num_div) * small_col_dim;

	ret22 = NULL;
	if((ret22_row_dim > 0) && (ret22_col_dim > 0))
	{
		ret22 = init_ddmatrix(ret22_row_dim, ret22_col_dim);

		// a21 * b12
		if((a21 != NULL) && (b12 != NULL))
			_bncomp_mul_ddmatrix_simple(ret22, a21, b12);
		//	_bncomp_mul_ddmatrix_block(ret22, a21, b12, min_dim);

		// ret22 += a22 * b22
		if((a22 != NULL) && (b22 != NULL))
		{
			tmp_mat = init_ddmatrix(ret22_row_dim, ret22_col_dim);
			
			//_bncomp_mul_ddmatrix_block(tmp_mat, a22, b22, min_dim);
			_bncomp_mul_ddmatrix_simple(tmp_mat, a22, b22);
			_bncomp_add_ddmatrix(ret22, ret22, tmp_mat);

			free_ddmatrix(tmp_mat);
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
		_bncomp_subst_ddmatrix_partial_checked(ret, ret_index, ret22, ret22_index);

		free_ddmatrix(ret22);
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
	free_ddmatrix(a11);
	free_ddmatrix(b11);
	if(a12 != NULL) free_ddmatrix(a12);
	if(a21 != NULL) free_ddmatrix(a21);
	if(a22 != NULL) free_ddmatrix(a22);
	if(b12 != NULL) free_ddmatrix(b12);
	if(b21 != NULL) free_ddmatrix(b21);
	if(b22 != NULL) free_ddmatrix(b22);

}

// Strassen's Algorithm (parallelized sections)
void _bncomp_mul_ddmatrix_strassen_even2(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim, long int rec_num)
{
	int thread_num, thread_index;
//	long int min_dim = 4; // = 2^2
	DDMatrix mat_p[7], mat_tmp_a[7], mat_tmp_b[7], mat_tmp_c[4];
	long int row_dim_h, row_dim, col_dim_h, col_dim, mid_dim_h, mid_dim;
	long int ret_index[4], mat_tmp_a_index[4], mat_tmp_b_index[4], mat_ar_index[7][4], mat_al_index[7][4], mat_br_index[7][4], mat_bl_index[7][4], mat_c_index[4][4];
	long int i;

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_ddmatrix_strassen_even)\n", mat_a->col_dim, mat_b->row_dim);
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
		_bncomp_num_addsub_mul_ddmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		_bncomp_num_mul_mul_ddmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		//mul_ddmatrix(ret, mat_a, mat_b);
		_bncomp_mul_ddmatrix_simple(ret, mat_a, mat_b);
		return;
	}

	#pragma omp parallel for
	for(i = 0; i < 7; i++)
	{
		mat_p[i] = init_ddmatrix(row_dim_h, col_dim_h);
		mat_tmp_a[i] = init_ddmatrix(row_dim_h, mid_dim_h);
		mat_tmp_b[i] = init_ddmatrix(mid_dim_h, col_dim_h);
	}
	
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
		mat_tmp_c[i] = init_ddmatrix(row_dim_h, col_dim_h);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		add_ddmatrix_partial(mat_tmp_a[0], mat_tmp_a_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);
	//	_bncomp_add_ddmatrix_partial(mat_tmp_a[0], mat_tmp_a_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		add_ddmatrix_partial(mat_tmp_b[0], mat_tmp_b_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);
	//	_bncomp_add_ddmatrix_partial(mat_tmp_b[0], mat_tmp_b_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);

		// P1 = tmp_a * tmp_b
		//printf("P1: %ld, ", rec_num);
		_bncomp_mul_ddmatrix_strassen_even2(mat_p[0], mat_tmp_a[0], mat_tmp_b[0], min_dim, rec_num + 1);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		add_ddmatrix_partial(mat_tmp_a[1], mat_tmp_a_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);
	//	_bncomp_add_ddmatrix_partial(mat_tmp_a[1], mat_tmp_a_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);

		// B11
		mat_br_index[1][0] = 0;
		mat_br_index[1][1] = mid_dim_h;
		mat_br_index[1][2] = 0;
		mat_br_index[1][3] = col_dim_h;
		subst_ddmatrix_partial(mat_tmp_b[1], mat_tmp_b_index, mat_b, mat_br_index[1]);
	//	_bncomp_subst_ddmatrix_partial(mat_tmp_b[1], mat_tmp_b_index, mat_b, mat_br_index[1]);

		// P2 = tmp_a * tmp_b
		//printf("P2: %ld, ", rec_num);
		_bncomp_mul_ddmatrix_strassen_even2(mat_p[1], mat_tmp_a[1], mat_tmp_b[1], min_dim, rec_num + 1);

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
		subst_ddmatrix_partial(mat_tmp_a[2], mat_tmp_a_index, mat_a, mat_ar_index[2]);
	//	_bncomp_subst_ddmatrix_partial(mat_tmp_a[2], mat_tmp_a_index, mat_a, mat_ar_index[2]);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		sub_ddmatrix_partial(mat_tmp_b[2], mat_tmp_b_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);
	//	_bncomp_sub_ddmatrix_partial(mat_tmp_b[2], mat_tmp_b_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);

		// P3 = tmp_a * tmp_b
		//printf("P3: %ld, ", rec_num);
		_bncomp_mul_ddmatrix_strassen_even2(mat_p[2], mat_tmp_a[2], mat_tmp_b[2], min_dim, rec_num + 1);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		subst_ddmatrix_partial(mat_tmp_a[3], mat_tmp_a_index, mat_a, mat_ar_index[3]);
	//	_bncomp_subst_ddmatrix_partial(mat_tmp_a[3], mat_tmp_a_index, mat_a, mat_ar_index[3]);

		// B21 - B11
		mat_br_index[3][0] = mid_dim_h;
		mat_br_index[3][1] = mid_dim;
		mat_br_index[3][2] = 0;
		mat_br_index[3][3] = col_dim_h;

		mat_bl_index[3][0] = 0;
		mat_bl_index[3][1] = mid_dim_h;
		mat_bl_index[3][2] = 0;
		mat_bl_index[3][3] = col_dim_h;
		sub_ddmatrix_partial(mat_tmp_b[3], mat_tmp_b_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);
	//	_bncomp_sub_ddmatrix_partial(mat_tmp_b[3], mat_tmp_b_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);

		// P4 = tmp_a * tmp_b
		//printf("P4: %ld, ", rec_num);
		_bncomp_mul_ddmatrix_strassen_even2(mat_p[3], mat_tmp_a[3], mat_tmp_b[3], min_dim, rec_num + 1);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		add_ddmatrix_partial(mat_tmp_a[4], mat_tmp_a_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);
	//	_bncomp_add_ddmatrix_partial(mat_tmp_a[4], mat_tmp_a_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);

		// B22
		mat_br_index[4][0] = mid_dim_h;
		mat_br_index[4][1] = mid_dim;
		mat_br_index[4][2] = col_dim_h;
		mat_br_index[4][3] = col_dim;
		subst_ddmatrix_partial(mat_tmp_b[4], mat_tmp_b_index, mat_b, mat_br_index[4]);
	//	_bncomp_subst_ddmatrix_partial(mat_tmp_b[4], mat_tmp_b_index, mat_b, mat_br_index[4]);

		// P5 = tmp_a * tmp_b
		//printf("P5: %ld, ", rec_num);
		_bncomp_mul_ddmatrix_strassen_even2(mat_p[4], mat_tmp_a[4], mat_tmp_b[4], min_dim, rec_num + 1);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		sub_ddmatrix_partial(mat_tmp_a[5], mat_tmp_a_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);
	//	_bncomp_sub_ddmatrix_partial(mat_tmp_a[5], mat_tmp_a_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		add_ddmatrix_partial(mat_tmp_b[5], mat_tmp_b_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);
	//	_bncomp_add_ddmatrix_partial(mat_tmp_b[5], mat_tmp_b_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);

		// P6 = tmp_a * tmp_b
		//printf("P6: %ld, ", rec_num);
		_bncomp_mul_ddmatrix_strassen_even2(mat_p[5], mat_tmp_a[5], mat_tmp_b[5], min_dim, rec_num + 1);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		sub_ddmatrix_partial(mat_tmp_a[6], mat_tmp_a_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);
	//	_bncomp_sub_ddmatrix_partial(mat_tmp_a[6], mat_tmp_a_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		add_ddmatrix_partial(mat_tmp_b[6], mat_tmp_b_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);
	//	_bncomp_add_ddmatrix_partial(mat_tmp_b[6], mat_tmp_b_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);

		// P7 = tmp_a * tmp_b
		//printf("P7: %ld\n", rec_num);
		_bncomp_mul_ddmatrix_strassen_even2(mat_p[6], mat_tmp_a[6], mat_tmp_b[6], min_dim, rec_num + 1);

	}
} // pragma omp parallel sections

#pragma omp barrier

//#pragma omp sections
#pragma omp parallel sections
{
	// -------------------------------
	// C11 := P1 + P4 - P5 + P7
	// -------------------------------
	#pragma omp section
	{

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += 3 * row_dim_h * col_dim_h;

		add_ddmatrix(mat_tmp_c[0], mat_p[0], mat_p[3]);
		sub_ddmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[4]);
		add_ddmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[6]);
		//_bncomp_add_ddmatrix(mat_tmp_c[0], mat_p[0], mat_p[3]);
		//_bncomp_sub_ddmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[4]);
		//_bncomp_add_ddmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[6]);
		mat_c_index[0][0] = 0;
		mat_c_index[0][1] = row_dim_h;
		mat_c_index[0][2] = 0;
		mat_c_index[0][3] = col_dim_h;

		subst_ddmatrix_partial(ret, mat_c_index[0], mat_tmp_c[0], ret_index);
		//_bncomp_subst_ddmatrix_partial(ret, mat_c_index[0], mat_tmp_c[0], ret_index);
	}

	// -------------------------------
	// C12 := P3 + P5
	// -------------------------------
	#pragma omp section
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

		add_ddmatrix(mat_tmp_c[1], mat_p[2], mat_p[4]);
		//_bncomp_add_ddmatrix(mat_tmp_c[1], mat_p[2], mat_p[4]);
		mat_c_index[1][0] = 0;
		mat_c_index[1][1] = row_dim_h;
		mat_c_index[1][2] = col_dim_h;
		mat_c_index[1][3] = col_dim;
		subst_ddmatrix_partial(ret, mat_c_index[1], mat_tmp_c[1], ret_index);
		//_bncomp_subst_ddmatrix_partial(ret, mat_c_index[1], mat_tmp_c[1], ret_index);
	}

	// -------------------------------
	// C21 := P2 + P4
	// -------------------------------
	#pragma omp section
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

		add_ddmatrix(mat_tmp_c[2], mat_p[1], mat_p[3]);
		//_bncomp_add_ddmatrix(mat_tmp_c[2], mat_p[1], mat_p[3]);
		mat_c_index[2][0] = row_dim_h;
		mat_c_index[2][1] = row_dim;
		mat_c_index[2][2] = 0;
		mat_c_index[2][3] = col_dim_h;
		subst_ddmatrix_partial(ret, mat_c_index[2], mat_tmp_c[2], ret_index);
		//_bncomp_subst_ddmatrix_partial(ret, mat_c_index[2], mat_tmp_c[2], ret_index);
	}

	// -------------------------------
	// C22 := P1 + P3 - P2 + P6
	// -------------------------------
	#pragma omp section
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += 3 * row_dim_h * col_dim_h;

		add_ddmatrix(mat_tmp_c[3], mat_p[0], mat_p[2]);
		sub_ddmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[1]);
		add_ddmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[5]);
		//_bncomp_add_ddmatrix(mat_tmp_c[3], mat_p[0], mat_p[2]);
		//_bncomp_sub_ddmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[1]);
		//_bncomp_add_ddmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[5]);
		mat_c_index[3][0] = row_dim_h;
		mat_c_index[3][1] = row_dim;
		mat_c_index[3][2] = col_dim_h;
		mat_c_index[3][3] = col_dim;
		subst_ddmatrix_partial(ret, mat_c_index[3], mat_tmp_c[3], ret_index);
		//_bncomp_subst_ddmatrix_partial(ret, mat_c_index[3], mat_tmp_c[3], ret_index);
	}

} // pragma omp parallel sections

	// free
	#pragma omp parallel for
	for(i = 0; i < 7; i++)
	{
		free_ddmatrix(mat_p[i]);
		free_ddmatrix(mat_tmp_a[i]);
		free_ddmatrix(mat_tmp_b[i]);
	}
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
		free_ddmatrix(mat_tmp_c[i]);
}

// Strassen's Algorithm (parallelizable tasks)
void _bncomp_mul_ddmatrix_strassen_even3(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim, long int rec_num)
{
	int thread_num, thread_index;
//	long int min_dim = 4; // = 2^2
	DDMatrix mat_p[7], mat_tmp_a[7], mat_tmp_b[7], mat_tmp_c[4];
	long int row_dim_h, row_dim, col_dim_h, col_dim, mid_dim_h, mid_dim;
	long int ret_index[4], mat_tmp_a_index[4], mat_tmp_b_index[4], mat_ar_index[7][4], mat_al_index[7][4], mat_br_index[7][4], mat_bl_index[7][4], mat_c_index[4][4];
	long int i;

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_ddmatrix_strassen_even)\n", mat_a->col_dim, mat_b->row_dim);
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
		_bncomp_num_addsub_mul_ddmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		_bncomp_num_mul_mul_ddmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		//mul_ddmatrix(ret, mat_a, mat_b);
		_bncomp_mul_ddmatrix_simple(ret, mat_a, mat_b);
		return;
	}

	#pragma omp parallel for
	for(i = 0; i < 7; i++)
	{
		mat_p[i] = init_ddmatrix(row_dim_h, col_dim_h);
		mat_tmp_a[i] = init_ddmatrix(row_dim_h, mid_dim_h);
		mat_tmp_b[i] = init_ddmatrix(mid_dim_h, col_dim_h);
	}
	
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
		mat_tmp_c[i] = init_ddmatrix(row_dim_h, col_dim_h);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		add_ddmatrix_partial(mat_tmp_a[0], mat_tmp_a_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);
	//	_bncomp_add_ddmatrix_partial(mat_tmp_a[0], mat_tmp_a_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		add_ddmatrix_partial(mat_tmp_b[0], mat_tmp_b_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);
	//	_bncomp_add_ddmatrix_partial(mat_tmp_b[0], mat_tmp_b_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);

		// P1 = tmp_a * tmp_b
		//printf("P1: %ld, ", rec_num);
		#pragma omp task
		_bncomp_mul_ddmatrix_strassen_even3(mat_p[0], mat_tmp_a[0], mat_tmp_b[0], min_dim, rec_num + 1);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		add_ddmatrix_partial(mat_tmp_a[1], mat_tmp_a_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);
	//	_bncomp_add_ddmatrix_partial(mat_tmp_a[1], mat_tmp_a_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);

		// B11
		mat_br_index[1][0] = 0;
		mat_br_index[1][1] = mid_dim_h;
		mat_br_index[1][2] = 0;
		mat_br_index[1][3] = col_dim_h;
		subst_ddmatrix_partial(mat_tmp_b[1], mat_tmp_b_index, mat_b, mat_br_index[1]);
	//	_bncomp_subst_ddmatrix_partial(mat_tmp_b[1], mat_tmp_b_index, mat_b, mat_br_index[1]);

		// P2 = tmp_a * tmp_b
		//printf("P2: %ld, ", rec_num);
		#pragma omp task
		_bncomp_mul_ddmatrix_strassen_even3(mat_p[1], mat_tmp_a[1], mat_tmp_b[1], min_dim, rec_num + 1);

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
		subst_ddmatrix_partial(mat_tmp_a[2], mat_tmp_a_index, mat_a, mat_ar_index[2]);
	//	_bncomp_subst_ddmatrix_partial(mat_tmp_a[2], mat_tmp_a_index, mat_a, mat_ar_index[2]);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		sub_ddmatrix_partial(mat_tmp_b[2], mat_tmp_b_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);
	//	_bncomp_sub_ddmatrix_partial(mat_tmp_b[2], mat_tmp_b_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);

		// P3 = tmp_a * tmp_b
		//printf("P3: %ld, ", rec_num);
		#pragma omp task
		_bncomp_mul_ddmatrix_strassen_even3(mat_p[2], mat_tmp_a[2], mat_tmp_b[2], min_dim, rec_num + 1);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		subst_ddmatrix_partial(mat_tmp_a[3], mat_tmp_a_index, mat_a, mat_ar_index[3]);
	//	_bncomp_subst_ddmatrix_partial(mat_tmp_a[3], mat_tmp_a_index, mat_a, mat_ar_index[3]);

		// B21 - B11
		mat_br_index[3][0] = mid_dim_h;
		mat_br_index[3][1] = mid_dim;
		mat_br_index[3][2] = 0;
		mat_br_index[3][3] = col_dim_h;

		mat_bl_index[3][0] = 0;
		mat_bl_index[3][1] = mid_dim_h;
		mat_bl_index[3][2] = 0;
		mat_bl_index[3][3] = col_dim_h;
		sub_ddmatrix_partial(mat_tmp_b[3], mat_tmp_b_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);
	//	_bncomp_sub_ddmatrix_partial(mat_tmp_b[3], mat_tmp_b_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);

		// P4 = tmp_a * tmp_b
		//printf("P4: %ld, ", rec_num);
		#pragma omp task
		_bncomp_mul_ddmatrix_strassen_even3(mat_p[3], mat_tmp_a[3], mat_tmp_b[3], min_dim, rec_num + 1);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		add_ddmatrix_partial(mat_tmp_a[4], mat_tmp_a_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);
	//	_bncomp_add_ddmatrix_partial(mat_tmp_a[4], mat_tmp_a_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);

		// B22
		mat_br_index[4][0] = mid_dim_h;
		mat_br_index[4][1] = mid_dim;
		mat_br_index[4][2] = col_dim_h;
		mat_br_index[4][3] = col_dim;
		subst_ddmatrix_partial(mat_tmp_b[4], mat_tmp_b_index, mat_b, mat_br_index[4]);
	//	_bncomp_subst_ddmatrix_partial(mat_tmp_b[4], mat_tmp_b_index, mat_b, mat_br_index[4]);

		// P5 = tmp_a * tmp_b
		//printf("P5: %ld, ", rec_num);
		#pragma omp task
		_bncomp_mul_ddmatrix_strassen_even3(mat_p[4], mat_tmp_a[4], mat_tmp_b[4], min_dim, rec_num + 1);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		sub_ddmatrix_partial(mat_tmp_a[5], mat_tmp_a_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);
	//	_bncomp_sub_ddmatrix_partial(mat_tmp_a[5], mat_tmp_a_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		add_ddmatrix_partial(mat_tmp_b[5], mat_tmp_b_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);
	//	_bncomp_add_ddmatrix_partial(mat_tmp_b[5], mat_tmp_b_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);

		// P6 = tmp_a * tmp_b
		//printf("P6: %ld, ", rec_num);
		#pragma omp task
		_bncomp_mul_ddmatrix_strassen_even3(mat_p[5], mat_tmp_a[5], mat_tmp_b[5], min_dim, rec_num + 1);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		sub_ddmatrix_partial(mat_tmp_a[6], mat_tmp_a_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);
	//	_bncomp_sub_ddmatrix_partial(mat_tmp_a[6], mat_tmp_a_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		add_ddmatrix_partial(mat_tmp_b[6], mat_tmp_b_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);
	//	_bncomp_add_ddmatrix_partial(mat_tmp_b[6], mat_tmp_b_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);

		// P7 = tmp_a * tmp_b
		//printf("P7: %ld\n", rec_num);
		#pragma omp task
		_bncomp_mul_ddmatrix_strassen_even3(mat_p[6], mat_tmp_a[6], mat_tmp_b[6], min_dim, rec_num + 1);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += 3 * row_dim_h * col_dim_h;

		add_ddmatrix(mat_tmp_c[0], mat_p[0], mat_p[3]);
		sub_ddmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[4]);
		add_ddmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[6]);
		//_bncomp_add_ddmatrix(mat_tmp_c[0], mat_p[0], mat_p[3]);
		//_bncomp_sub_ddmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[4]);
		//_bncomp_add_ddmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[6]);
		mat_c_index[0][0] = 0;
		mat_c_index[0][1] = row_dim_h;
		mat_c_index[0][2] = 0;
		mat_c_index[0][3] = col_dim_h;

		subst_ddmatrix_partial(ret, mat_c_index[0], mat_tmp_c[0], ret_index);
		//_bncomp_subst_ddmatrix_partial(ret, mat_c_index[0], mat_tmp_c[0], ret_index);
	}

	// -------------------------------
	// C12 := P3 + P5
	// -------------------------------
	#pragma omp section
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

		add_ddmatrix(mat_tmp_c[1], mat_p[2], mat_p[4]);
		//_bncomp_add_ddmatrix(mat_tmp_c[1], mat_p[2], mat_p[4]);
		mat_c_index[1][0] = 0;
		mat_c_index[1][1] = row_dim_h;
		mat_c_index[1][2] = col_dim_h;
		mat_c_index[1][3] = col_dim;
		subst_ddmatrix_partial(ret, mat_c_index[1], mat_tmp_c[1], ret_index);
		//_bncomp_subst_ddmatrix_partial(ret, mat_c_index[1], mat_tmp_c[1], ret_index);
	}

	// -------------------------------
	// C21 := P2 + P4
	// -------------------------------
	#pragma omp section
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

		add_ddmatrix(mat_tmp_c[2], mat_p[1], mat_p[3]);
		//_bncomp_add_ddmatrix(mat_tmp_c[2], mat_p[1], mat_p[3]);
		mat_c_index[2][0] = row_dim_h;
		mat_c_index[2][1] = row_dim;
		mat_c_index[2][2] = 0;
		mat_c_index[2][3] = col_dim_h;
		subst_ddmatrix_partial(ret, mat_c_index[2], mat_tmp_c[2], ret_index);
		//_bncomp_subst_ddmatrix_partial(ret, mat_c_index[2], mat_tmp_c[2], ret_index);
	}

	// -------------------------------
	// C22 := P1 + P3 - P2 + P6
	// -------------------------------
	#pragma omp section
	{
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += 3 * row_dim_h * col_dim_h;

		add_ddmatrix(mat_tmp_c[3], mat_p[0], mat_p[2]);
		sub_ddmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[1]);
		add_ddmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[5]);
		//_bncomp_add_ddmatrix(mat_tmp_c[3], mat_p[0], mat_p[2]);
		//_bncomp_sub_ddmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[1]);
		//_bncomp_add_ddmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[5]);
		mat_c_index[3][0] = row_dim_h;
		mat_c_index[3][1] = row_dim;
		mat_c_index[3][2] = col_dim_h;
		mat_c_index[3][3] = col_dim;
		subst_ddmatrix_partial(ret, mat_c_index[3], mat_tmp_c[3], ret_index);
		//_bncomp_subst_ddmatrix_partial(ret, mat_c_index[3], mat_tmp_c[3], ret_index);
	}

} // pragma omp parallel sections

	// free
	#pragma omp parallel for
	for(i = 0; i < 7; i++)
	{
		free_ddmatrix(mat_p[i]);
		free_ddmatrix(mat_tmp_a[i]);
		free_ddmatrix(mat_tmp_b[i]);
	}
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
		free_ddmatrix(mat_tmp_c[i]);
}

// Winograd Variant of Strassen's Algorithm (parallelied sections)
void _bncomp_mul_ddmatrix_winograd_even2(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim)
{
	int thread_num, thread_index;
//	long int min_dim = 4; // = 2^2
	DDMatrix mat_s[8], mat_m[7], mat_t[2], mat_tmp_a[4], mat_tmp_b[4], mat_tmp_c[4];
	long int row_dim_h, row_dim, col_dim_h, col_dim, mid_dim, mid_dim_h;
	long int ret_index[4], mat_tmp_a_index[4], mat_tmp_b_index[4];
	long int mat_a_index[4][4], mat_b_index[4][4], mat_c_index[4][4];
	long int i;

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_ddmatrix_winograd_even)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	row_dim_h = row_dim / 2;
	col_dim_h = col_dim / 2;
	mid_dim_h = mid_dim / 2;

	// normal matrix multiplication in case of ret_dim <= 4 
//	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	if((ret->row_dim <= min_dim) || (ret->col_dim <= min_dim) || (mid_dim <= min_dim))
	{
		mul_ddmatrix_simple(ret, mat_a, mat_b);
		//_bncomp_mul_ddmatrix_simple(ret, mat_a, mat_b);

		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		_bncomp_num_mul_mul_ddmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		return;
	}

	#pragma omp parallel for
	for(i = 0; i < 4; i++)
	{
		mat_s[i] = init_ddmatrix(row_dim_h, mid_dim_h);
		mat_tmp_a[i] = init_ddmatrix(row_dim_h, mid_dim_h);

		mat_s[i + 4] = init_ddmatrix(mid_dim_h, col_dim_h);
		mat_tmp_b[i] = init_ddmatrix(mid_dim_h, col_dim_h);

		mat_tmp_c[i] = init_ddmatrix(row_dim_h, col_dim_h);
	}
	#pragma omp parallel for
	for(i = 0; i < 7; i++)
		mat_m[i] = init_ddmatrix(row_dim_h, col_dim_h);

	mat_t[0] = init_ddmatrix(row_dim_h, col_dim_h);
	mat_t[1] = init_ddmatrix(row_dim_h, col_dim_h);

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
		subst_ddmatrix_partial(mat_tmp_a[i], mat_tmp_a_index, mat_a, mat_a_index[i]);
		subst_ddmatrix_partial(mat_tmp_b[i], mat_tmp_b_index, mat_b, mat_b_index[i]);
		//_bncomp_subst_ddmatrix_partial(mat_tmp_a[i], mat_tmp_a_index, mat_a, mat_a_index[i]);
		//_bncomp_subst_ddmatrix_partial(mat_tmp_b[i], mat_tmp_b_index, mat_b, mat_b_index[i]);
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
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		add_ddmatrix(mat_s[0], mat_tmp_a[2], mat_tmp_a[3]);
		//_bncomp_add_ddmatrix(mat_s[0], mat_tmp_a[2], mat_tmp_a[3]);
	}

	#pragma omp section
	{
		// -------------------------------
		// S3 := A11 - A21
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		sub_ddmatrix(mat_s[2], mat_tmp_a[0], mat_tmp_a[2]);
		//_bncomp_sub_ddmatrix(mat_s[2], mat_tmp_a[0], mat_tmp_a[2]);
	}

	#pragma omp section
	{
		// -------------------------------
		// S5 := B12 - B11
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		sub_ddmatrix(mat_s[4], mat_tmp_b[1], mat_tmp_b[0]);
		//_bncomp_sub_ddmatrix(mat_s[4], mat_tmp_b[1], mat_tmp_b[0]);
	}

	#pragma omp section
	{
		// -------------------------------
		// S7 := B22 - B12
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		sub_ddmatrix(mat_s[6], mat_tmp_b[3], mat_tmp_b[1]);
		//_bncomp_sub_ddmatrix(mat_s[6], mat_tmp_b[3], mat_tmp_b[1]);
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
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		sub_ddmatrix(mat_s[1], mat_s[0], mat_tmp_a[0]);
		//_bncomp_sub_ddmatrix(mat_s[1], mat_s[0], mat_tmp_a[0]);

		// -------------------------------
		// S4 := A12 - S2
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * mid_dim_h;

		sub_ddmatrix(mat_s[3], mat_tmp_a[1], mat_s[1]);
		//_bncomp_sub_ddmatrix(mat_s[3], mat_tmp_a[1], mat_s[1]);
	}

	#pragma omp section
	{
		// -------------------------------
		// S6 := B22 - S5
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		sub_ddmatrix(mat_s[5], mat_tmp_b[3], mat_s[4]);
		//_bncomp_sub_ddmatrix(mat_s[5], mat_tmp_b[3], mat_s[4]);

		// -------------------------------
		// S8 := S6 - B21
		//--------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += mid_dim_h * col_dim_h;

		sub_ddmatrix(mat_s[7], mat_s[5], mat_tmp_b[2]);
		//_bncomp_sub_ddmatrix(mat_s[7], mat_s[5], mat_tmp_b[2]);
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
		_bncomp_mul_ddmatrix_winograd_even2(mat_m[0], mat_s[1], mat_s[5], min_dim);
	}

	#pragma omp section
	{
		// -------------------------------
		// M2 := A11 * B11
		// -------------------------------
		_bncomp_mul_ddmatrix_winograd_even2(mat_m[1], mat_tmp_a[0], mat_tmp_b[0], min_dim);
	}

	#pragma omp section
	{
		// -------------------------------
		// M3 := A12 * B21
		// -------------------------------
		_bncomp_mul_ddmatrix_winograd_even2(mat_m[2], mat_tmp_a[1], mat_tmp_b[2], min_dim);
	}

	#pragma omp section
	{
		// -------------------------------
		// M4 := S3 * S7
		// -------------------------------
		_bncomp_mul_ddmatrix_winograd_even2(mat_m[3], mat_s[2], mat_s[6], min_dim);
	}

	#pragma omp section
	{
		// -------------------------------
		// M5 := S1 * S5
		// -------------------------------
		_bncomp_mul_ddmatrix_winograd_even2(mat_m[4], mat_s[0], mat_s[4], min_dim);
	}

	#pragma omp section
	{
		// -------------------------------
		// M6 := S4 * B22
		// -------------------------------
		_bncomp_mul_ddmatrix_winograd_even2(mat_m[5], mat_s[3], mat_tmp_b[3], min_dim);
	}

	#pragma omp section
	{
		// -------------------------------
		// M7 := A22 * S8
		// -------------------------------
		_bncomp_mul_ddmatrix_winograd_even2(mat_m[6], mat_tmp_a[3], mat_s[7], min_dim);
	}
} // pragma omp parallel sections

//	printf("m...\n");

	// -------------------------------
	// T1 := M1 + M2
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

	//add_ddmatrix(mat_t[0], mat_m[0], mat_m[1]);
	_bncomp_add_ddmatrix(mat_t[0], mat_m[0], mat_m[1]);

	// -------------------------------
	// T2 := T1 + M4
	// -------------------------------
	// counting the number of arithmetic
	_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

	//add_ddmatrix(mat_t[1], mat_t[0], mat_m[3]);
	_bncomp_add_ddmatrix(mat_t[1], mat_t[0], mat_m[3]);

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
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

		add_ddmatrix(mat_tmp_c[0], mat_m[1], mat_m[2]);
		//_bncomp_add_ddmatrix(mat_tmp_c[0], mat_m[1], mat_m[2]);
	}

	#pragma omp section
	{
		// -------------------------------
		// C12 := T1 + M5 + M6
		// -------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += 2 * row_dim_h * col_dim_h;

		add_ddmatrix(mat_tmp_c[1], mat_t[0], mat_m[4]);
		add_ddmatrix(mat_tmp_c[1], mat_tmp_c[1], mat_m[5]);
		//_bncomp_add_ddmatrix(mat_tmp_c[1], mat_t[0], mat_m[4]);
		//_bncomp_add_ddmatrix(mat_tmp_c[1], mat_tmp_c[1], mat_m[5]);
	}

	#pragma omp section
	{
		// -------------------------------
		// C21 := T2 - M7
		// -------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

		sub_ddmatrix(mat_tmp_c[2], mat_t[1], mat_m[6]);
		//_bncomp_sub_ddmatrix(mat_tmp_c[2], mat_t[1], mat_m[6]);
	}

	#pragma omp section
	{
		// -------------------------------
		// C22 := T2 + M5
		// -------------------------------
		// counting the number of arithmetic
		_bncomp_num_addsub_mul_ddmatrix_strassen += row_dim_h * col_dim_h;

		add_ddmatrix(mat_tmp_c[3], mat_t[1], mat_m[4]);
		//_bncomp_add_ddmatrix(mat_tmp_c[3], mat_t[1], mat_m[4]);
	}
} // pragma omp parallel sections

//	printf("c...\n");

	// -------------------------------
	// RET := [C11 C12]
	//        [C21 C22]
	// -------------------------------
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
		subst_ddmatrix_partial(ret, mat_c_index[i], mat_tmp_c[i], ret_index);
		//_bncomp_subst_ddmatrix_partial(ret, mat_c_index[i], mat_tmp_c[i], ret_index);

//	printf("set...\n");


	// free
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
	{
		free_ddmatrix(mat_s[i]);
		free_ddmatrix(mat_tmp_a[i]);
		free_ddmatrix(mat_s[i + 4]);
		free_ddmatrix(mat_tmp_b[i]);
		free_ddmatrix(mat_tmp_c[i]);
	}
	#pragma omp parallel for
	for(i = 0; i < 7; i++)
		free_ddmatrix(mat_m[i]);
	
	free_ddmatrix(mat_t[0]);
	free_ddmatrix(mat_t[1]);
}

// The following main function is for debugging
#ifdef DEBUG

#include "get_secv.h"

int main(int argc, char *argv[])
{
	int num_threads;
	long int i, j, row_dim, col_dim, mid_dim, min_dim = 32;
	unsigned long prec = 128;
	double ddtmp[4][DDSIZE];
	DDMatrix mpfc, mpfa, mpfb, mpfc_normal, mpfc_block, mpfc_tmp;
	DDVector mpfdiag_left, mpfdiag_right;
	double stime, etime[4], reldiff[4];

//	dim = 128;
	if(argc <= 4)
	{
		fprintf(stderr, "Usage: %s [row_dim] [col_dim] [mid_dim] [min_dim] [#thread]\n", argv[0]);
		return 0;
	}
	row_dim = atol(argv[1]);
	if(row_dim <= 0)
		return 0;

	col_dim = atol(argv[2]);
	if(col_dim <= 0)
		return 0;

	mid_dim = atol(argv[3]);
	if(mid_dim <= 0)
		return 0;

	if(argc >= 5)
	{
		min_dim = atol(argv[4]);
		if(min_dim < 1)
			min_dim = 1;
	}

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

	set0_dd(ddtmp[0]);
	set0_dd(ddtmp[1]);

	mpfc = init_ddmatrix(row_dim, col_dim);
	mpfc_tmp = init_ddmatrix(row_dim, col_dim);
	mpfc_normal = init_ddmatrix(row_dim, col_dim);
	mpfc_block = init_ddmatrix(row_dim, col_dim);
	mpfa = init_ddmatrix(row_dim, mid_dim);
	mpfb = init_ddmatrix(mid_dim, col_dim);

//	mpfdiag_left = init_ddvector(dim);
//	mpfdiag_right = init_ddvector(dim);

	// set A
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < mid_dim; j++)
		{
			rdd_set_d(ddtmp[0], (double)rand());
			if(rand() % 2 != 0)
				rdd_neg(ddtmp[0], ddtmp[0]);

			rdd_set_d(ddtmp[1], (double)rand());
			rdd_ui_div(ddtmp[1], 1UL, ddtmp[1]);
			if(rand() % 2 != 0)
				rdd_neg(ddtmp[1], ddtmp[1]);

			set_ddmatrix_ij(mpfa, i, j, ddtmp[0]);
		}
	}

	// set B
	for(i = 0; i < mid_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rdd_set_d(ddtmp[0], (double)rand());
			if(rand() % 2 != 0)
				rdd_neg(ddtmp[0], ddtmp[0]);

			rdd_set_d(ddtmp[1], (double)rand());
			rdd_ui_div(ddtmp[1], 1UL, ddtmp[1]);
			if(rand() % 2 != 0)
				rdd_neg(ddtmp[1], ddtmp[1]);

			set_ddmatrix_ij(mpfb, i, j, ddtmp[1]);
		}
	}

	// normal matrix mul
	printf("_bncomp_mul_ddmatrix_simple...\n");
	stime = get_real_secv();
	_bncomp_mul_ddmatrix(mpfc_normal, mpfa, mpfb);
	etime[0] = get_real_secv() - stime;

	//left_scaling_ddmatrix(mpfa, mpfa, mpfdiag_left, NULL);
	//right_scaling_ddmatrix(mpfb, mpfb, mpfdiag_right, NULL);

	// blocked matrix mul
	printf("_bncomp_mul_ddmatrix_block(%ld)...\n", min_dim);
	//stime = get_secv();
	stime = get_real_secv();
	_bncomp_mul_ddmatrix_block(mpfc_block, mpfa, mpfb, min_dim);
	//etime[1] = get_secv() - stime;
	etime[1] = get_real_secv() - stime;

	// Strassen 
	printf("_bncomp_mul_ddmatrix_strassen(%ld)...\n", min_dim);
	stime = get_real_secv();
	_bncomp_mul_ddmatrix_strassen(mpfc, mpfa, mpfb, min_dim);
//	mul_ddmatrix_strassen(mpfc, mpfa, mpfb, 16);
//	mul_ddmatrix_strassen(mpfc, mpfa, mpfb, 32);
	etime[2] = get_real_secv() - stime;

	//mul_ddmatrix_dddiag(mpfc, mpfdiag_left, 0, mpfc, mpfdiag_right, 0);

	// difference
	sub_ddmatrix(mpfc_tmp  , mpfc_normal, mpfc);
	sub_ddmatrix(mpfc      , mpfc_normal, mpfc_block);

	// print
	printf("row_dim, col_dim, mid_dim: %ld, %ld, %ld\n", row_dim, col_dim, mid_dim);
	printf("normal         : %f\n", etime[0]);
	printf("block          : %f\n", etime[1]);
	printf("strassen       : %f\n", etime[2]);

	normi_ddmatrix(ddtmp[0], mpfc_tmp);
	normi_ddmatrix(ddtmp[1], mpfc);
	normi_ddmatrix(ddtmp[2], mpfc_block);
	normi_ddmatrix(ddtmp[3], mpfc_normal);

	rdd_div(ddtmp[0], ddtmp[0], ddtmp[3]);
	rdd_div(ddtmp[1], ddtmp[1], ddtmp[3]);
	printf("||reldiff_block   ||       : "); rdd_out_str(ddtmp[1]); printf("\n");
	printf("||reldiff_strassen||       : "); rdd_out_str(ddtmp[0]); printf("\n");

/* Inverse */

//	frank_ddmatrix(mpfa, dim);
//	frank_ddmatrix(mpfb, dim);
//	lotkin_ddmatrix(mpfa, dim);
//	lotkin_ddmatrix(mpfb, dim);

	free_ddmatrix(mpfc);
	free_ddmatrix(mpfc_tmp);
	free_ddmatrix(mpfc_normal);
	free_ddmatrix(mpfc_block);
	free_ddmatrix(mpfa);
	free_ddmatrix(mpfb);

//	free_ddvector(mpfdiag_left);
//	free_ddvector(mpfdiag_right);

	return 0;
}
#endif // DEBUG
