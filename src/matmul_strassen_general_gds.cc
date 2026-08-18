/********************************************************************************/
/* matmul_strassen_general_gds.cc:                                              */
/* Copyright (C) 2015 Tomonori Kouya                                            */
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
#include <iostream>   /* preload libstdc++ <iosfwd>/<string> before extern "C" headers */
#include <string>
#include <cmath>
#ifdef DEBUG
	#include <stdio.h>
	#include <math.h>
#endif // DEBUG
#include "bncuda.h" // BNCUDA library
#include "gdslinear.h" // G[D,Q]DVector, G[D,Q]DMatrix

//#include "bnc.h"
#include "bncomp.h"
#include "dslinear.h"

#include "matmul_strassen.h"

// count the number of computations
long int _bncuda_num_addsub_mul_gdsmatrix_strassen;	// addition and subtraction
long int _bncuda_num_mul_mul_gdsmatrix_strassen;		// multiplication

// Total number of threads currently used in _bncuda_matmul_gdsmatrix_strassen
#define MAX_TOTAL_NUM_THREADS (2)
//#define MAX_TOTAL_NUM_THREADS (4)
//#define MAX_TOTAL_NUM_THREADS (12)
//#define MAX_TOTAL_NUM_THREADS (12 * 2)
//#define MAX_TOTAL_NUM_THREADS (12 * 4)
int _bncuda_total_num_threads_matmul_gdsmatrix_strassen = 0; // -> _bncuda_total_num_threads[MATMUL_DDMATRIX_STRASSEN] in matmul_general.h;

/* c = a * b */
//void _bncuda_mul_gdsmatrix_simple(GDSMatrix ret, GDSMatrix a, GDSMatrix b)
//{
//	_bncuda_mul_gdsmatrix(ret, a, b);
//}

// Block matrix multiplicaiton
void _bncuda_mul_gdsmatrix_block(GDSMatrix ret_dev, GDSMatrix mat_a_dev, GDSMatrix mat_b_dev, long int min_dim, int num_blocks_per_grid, int num_threads_per_block)
{
	int thread_num, thread_index;
	int row_padding_flag = 0, col_padding_flag = 0, mid_padding_flag = 0;
	long i, j, k, row_dim, col_dim, mid_dim;
	long int num_div_row, num_div_col, num_div_mid, max_num_div;
	long int **mat_a_index, small_mat_a_index[4];
	long int **mat_b_index, small_mat_b_index[4];
	long int **ret_index, small_ret_index[4];
	long int **mat_a_index_dev, *small_mat_a_index_dev;
	long int **mat_b_index_dev, *small_mat_b_index_dev;
	long int **ret_index_dev, *small_ret_index_dev;
	GDSMatrix *small_ret_dev, *small_mat_a_dev, *small_mat_b_dev, *small_tmp_mat_dev;

	// initialize
	row_dim = ret_dev->row_dim;
	col_dim = ret_dev->col_dim;
	mid_dim = mat_a_dev->col_dim;
	if(mid_dim != mat_b_dev->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(_bncuda_mul_gdsmatrix_block)\n", mat_a_dev->col_dim, mat_b_dev->row_dim);
		return;
	}

	// normal matrix multiplication in case of ret_dim <= 4 
	if((ret_dev->row_dim <= min_dim) && (ret_dev->col_dim <= min_dim) && (mid_dim <= min_dim))
	{
		_bncuda_mul_gdsmatrix_simple(ret_dev, mat_a_dev, mat_b_dev, num_blocks_per_grid, num_threads_per_block);
		return;
	}

	// Number of division of matrix
	num_div_row = (ret_dev->row_dim) / min_dim;
	if((ret_dev->row_dim % min_dim) >= 1)
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

	num_div_col = (ret_dev->col_dim) / min_dim;
	if((ret_dev->col_dim % min_dim) >= 1)
	{
		col_padding_flag = 1;
		num_div_col++;
	}

	max_num_div = (num_div_row > num_div_mid) ? num_div_row : num_div_mid;
	max_num_div = (max_num_div > num_div_col) ? max_num_div : num_div_col;

	// initialize
#ifdef _OPENMP
	thread_num = omp_get_num_threads();
#else
	thread_num = 1;
#endif // _OPENMP

	mat_a_index = (long int **)calloc(max_num_div, sizeof(long int *));
	mat_b_index = (long int **)calloc(max_num_div, sizeof(long int *));
	ret_index = (long int **)calloc(max_num_div, sizeof(long int *));

	mat_a_index_dev = (long int **)calloc(max_num_div, sizeof(long int *));
	mat_b_index_dev = (long int **)calloc(max_num_div, sizeof(long int *));
	ret_index_dev = (long int **)calloc(max_num_div, sizeof(long int *));

	#pragma omp parallel for
	for(i = 0; i < max_num_div; i++)
	{
		mat_a_index[i] = (long int *)calloc(4, sizeof(long int));
		mat_b_index[i] = (long int *)calloc(4, sizeof(long int));
		ret_index[i] = (long int *)calloc(4, sizeof(long int));

		mat_a_index_dev[i] = _bncuda_init_l_array(4);
		mat_b_index_dev[i] = _bncuda_init_l_array(4);
		ret_index_dev[i] = _bncuda_init_l_array(4);
	}

	small_ret_dev = (GDSMatrix *)calloc(sizeof(GDSMatrix), num_div_col);
	small_mat_a_dev = (GDSMatrix *)calloc(sizeof(GDSMatrix), num_div_mid);
	small_mat_b_dev = (GDSMatrix *)calloc(sizeof(GDSMatrix), num_div_mid);
	small_tmp_mat_dev = (GDSMatrix *)calloc(sizeof(GDSMatrix), num_div_mid);

	#pragma omp parallel for
	for(i = 0; i < num_div_col; i++)
		small_ret_dev[i] = init_gdsmatrix_dev(min_dim, min_dim);

	#pragma omp parallel for
	for(i = 0; i < num_div_mid; i++)
	{
		small_mat_a_dev[i] = init_gdsmatrix_dev(min_dim, min_dim);
		small_mat_b_dev[i] = init_gdsmatrix_dev(min_dim, min_dim);
		small_tmp_mat_dev[i] = init_gdsmatrix_dev(min_dim, min_dim);
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

	// Host to Device
	small_mat_a_index_dev = _bncuda_init_l_array(4);
	small_mat_b_index_dev = _bncuda_init_l_array(4);
	small_ret_index_dev = _bncuda_init_l_array(4);

//	_bncuda_set_l_array(small_mat_a_index_dev, small_mat_a_index, 4);
//	_bncuda_set_l_array(small_mat_b_index_dev, small_mat_b_index, 4);
//	_bncuda_set_l_array(small_ret_index_dev, small_ret_index, 4);

	//printf("main loop....\n");

	// mail loop
	for(i = 0; i < num_div_row; i++)
	{
		//printf("start i = %ld \n", i);

		#pragma omp parallel for
		for(j = 0; j < num_div_mid; j++)
		{
			// copy matrices
			mat_a_index[j][0] = i * min_dim;
			mat_a_index[j][1] = (i + 1) * min_dim;
			mat_a_index[j][2] = j * min_dim;
			mat_a_index[j][3] = (j + 1) * min_dim;
			//_bncuda_set_l_array(mat_a_index_dev[j], mat_a_index[j], 4);

			//subst_gdsmatrix_partial_checked(small_mat_a[j], small_mat_a_index, mat_a, mat_a_index[j]);
			//_bncuda_subst_gdsmatrix_partial_checked(small_mat_a_dev[j], small_mat_a_index_dev, mat_a_dev, mat_a_index_dev[j], num_blocks_per_grid, num_threads_per_block);
			_bncuda_subst_gdsmatrix_partial_checked(small_mat_a_dev[j], small_mat_a_index, mat_a_dev, mat_a_index[j], num_blocks_per_grid, num_threads_per_block);
	
		}

		//printf("... _bncuda_subst_gdsmatrix_partial_checked ... ", i);

		for(j = 0; j < num_div_col; j++)
		{
			_bncuda_set0_gdsmatrix(small_ret_dev[j], num_blocks_per_grid, num_threads_per_block);

			#pragma omp parallel for
			for(k = 0; k < num_div_mid; k++)
			{
				// copy matrices
				mat_b_index[k][0] = k * min_dim;
				mat_b_index[k][1] = (k + 1) * min_dim;
				mat_b_index[k][2] = j * min_dim;
				mat_b_index[k][3] = (j + 1) * min_dim;
				//_bncuda_set_l_array(mat_b_index_dev[k], mat_b_index[k], 4);

				//_bncuda_subst_gdsmatrix_partial_checked(small_mat_b_dev[k], small_mat_b_index_dev, mat_b_dev, mat_b_index_dev[k], num_blocks_per_grid, num_threads_per_block);
				_bncuda_subst_gdsmatrix_partial_checked(small_mat_b_dev[k], small_mat_b_index, mat_b_dev, mat_b_index[k], num_blocks_per_grid, num_threads_per_block);
				//_bncuda_subst_gdsmatrix_partial_checked(small_mat_b[k], small_mat_b_index, mat_b, mat_b_index[k]);
				// ret[j] += small_mat_a[i][k] * small_mat_b[k][j];
				mul_gdsmatrix_dev(small_tmp_mat_dev[k], small_mat_a_dev[k], small_mat_b_dev[k], num_blocks_per_grid, num_threads_per_block);
			}

			for(k = 0; k < num_div_mid; k++)
				add_gdsmatrix_dev(small_ret_dev[j], small_ret_dev[j], small_tmp_mat_dev[k], num_blocks_per_grid, num_threads_per_block);

			ret_index[j][0] = i * min_dim;
			ret_index[j][1] = (i + 1) * min_dim;
			ret_index[j][2] = j * min_dim;
			ret_index[j][3] = (j + 1) * min_dim;
			//_bncuda_set_l_array(ret_index_dev[j], ret_index[j], 4);

			//subst_gdsmatrix_partial_checked(ret, ret_index[j], small_ret[j], small_ret_index);
			//_bncuda_subst_gdsmatrix_partial_checked(ret_dev, ret_index_dev[j], small_ret_dev[j], small_ret_index_dev, num_blocks_per_grid, num_threads_per_block);
			_bncuda_subst_gdsmatrix_partial_checked(ret_dev, ret_index[j], small_ret_dev[j], small_ret_index, num_blocks_per_grid, num_threads_per_block);
		}

		//printf("....end \n");

	}

	#pragma omp parallel for
	for(i = 0; i < max_num_div; i++)
	{
		free(mat_a_index[i]);
		free(mat_b_index[i]);
		free(ret_index[i]);

		_bncuda_free_l_array(mat_a_index_dev[i]);
		_bncuda_free_l_array(mat_b_index_dev[i]);
		_bncuda_free_l_array(ret_index_dev[i]);
	}

	free(mat_a_index);
	free(mat_b_index);
	free(ret_index);

	free(mat_a_index_dev);
	free(mat_b_index_dev);
	free(ret_index_dev);

	#pragma omp parallel for
	for(i = 0; i < num_div_col; i++)
		free_gdsmatrix_dev(small_ret_dev[i]);

	#pragma omp parallel for
	for(i = 0; i < num_div_mid; i++)
	{
		free_gdsmatrix_dev(small_mat_a_dev[i]);
		free_gdsmatrix_dev(small_mat_b_dev[i]);
		free_gdsmatrix_dev(small_tmp_mat_dev[i]);
	}

	free(small_ret_dev);
	free(small_mat_a_dev);
	free(small_mat_b_dev);
	free(small_tmp_mat_dev);

	_bncuda_free_l_array(small_mat_a_index_dev);
	_bncuda_free_l_array(small_mat_b_index_dev);
	_bncuda_free_l_array(small_ret_index_dev);

}

// Padding to 2-powered dimensional matrix
GDSMatrix _bncuda_init_static_padding_gdsmatrix_strassen(GDSMatrix orig_mat)
{
/***********************************************************
	int thread_num, thread_index;
	GDSMatrix ret = NULL;
	long int ret_row_dim, ret_col_dim, min_dim, i, j;

	if(orig_mat == NULL)
	{
		fprintf(stderr, "Warning: orig_mat is empty!(padding_gdsmatrix_strassen)\n");
		return NULL;
	}

	ret_row_dim = (long int)pow(2.0, ceil(mylog2((double)orig_mat->row_dim)));
	ret_col_dim = (long int)pow(2.0, ceil(mylog2((double)orig_mat->col_dim)));

	//printf("Padding: row_dim %ld -> %ld, col_dim %ld -> %ld\n", orig_mat->row_dim, ret_row_dim, orig_mat->col_dim, ret_col_dim);

	ret = init_gdsmatrix(ret_row_dim, ret_col_dim);
	if(ret == NULL)
	{
		fprintf(stderr, "Warning: padding matrix cannot be allocated!(padding_gdsmatrix_strassen)\n");
		return NULL;
	}

	// ret = [ A | 0 ]
	//       [---+---]
	//       [ 0 | I ]
	#pragma omp parallel for private(j)
	for(i = 0; i < orig_mat->row_dim; i++)
	{
		for(j = 0; j < orig_mat->col_dim; j++)
			set_gdsmatrix_ij(ret, i, j, get_gdsmatrix_ij(orig_mat, i, j));
	}

	min_dim = (ret_row_dim < ret_col_dim) ? ret_row_dim : ret_col_dim;
	#pragma omp parallel for
	for(i = orig_mat->row_dim; i < min_dim; i++)
		set_gdsmatrix_ij_ui(ret, i, i, 1UL);

	return ret;
*************************************************************/
}

// Padding to even dimensional matrix
GDSMatrix _bncuda_init_dynamic_padding_gdsmatrix_strassen(GDSMatrix orig_mat)
{
/***********************************************************
	int thread_num, thread_index;
	GDSMatrix ret = NULL;
	long int ret_row_dim, ret_col_dim, min_dim, i, j;

	if(orig_mat == NULL)
	{
		fprintf(stderr, "Warning: orig_mat is empty!(padding_gdsmatrix_strassen)\n");
		return NULL;
	}

	ret_row_dim = orig_mat->row_dim;
	ret_col_dim = orig_mat->col_dim;

	if((ret_row_dim % 2) == 1)
		ret_row_dim++;
	if((ret_col_dim % 2) == 1)
		ret_col_dim++;

//	printf("Padding: row_dim %ld -> %ld, col_dim %ld -> %ld\n", orig_mat->row_dim, ret_row_dim, orig_mat->col_dim, ret_col_dim);

	ret = init_gdsmatrix(ret_row_dim, ret_col_dim);
	if(ret == NULL)
	{
		fprintf(stderr, "Warning: padding matrix cannot be allocated!(padding_gdsmatrix_strassen)\n");
		return NULL;
	}

	// ret = [ A | 0 ]
	//       [---+---]
	//       [ 0 | I ]
	#pragma omp parallel for private(j)
	for(i = 0; i < orig_mat->row_dim; i++)
	{
		for(j = 0; j < orig_mat->col_dim; j++)
			set_gdsmatrix_ij(ret, i, j, get_gdsmatrix_ij(orig_mat, i, j));
	}

//	min_dim = (ret_row_dim < ret_col_dim) ? ret_row_dim : ret_col_dim;
//	for(i = orig_mat->row_dim; i < min_dim; i++)
//		set_gdsmatrix_ij_ui(ret, i, i, 1UL);

	return ret;
*************************************************************/
}

// Padding to even dimensional matrix
GDSMatrix _bncuda_init_dynamic_padding_gdsmatrix_strassen2(GDSMatrix orig_mat_dev, long int min_dim)
{
	int thread_num, thread_index;
	GDSMatrix ret_dev = NULL;
	long int ret_row_dim, ret_col_dim, i, j;

	if(orig_mat_dev == NULL)
	{
		fprintf(stderr, "Warning: orig_mat is empty!(padding_gdsmatrix_strassen)\n");
		return NULL;
	}

	ret_row_dim = orig_mat_dev->row_dim;
	ret_col_dim = orig_mat_dev->col_dim;

	if((ret_row_dim % min_dim) >= 1)
		ret_row_dim = ((ret_row_dim / min_dim) + 1) * min_dim;
	if((ret_col_dim % min_dim) >= 1)
		ret_col_dim = ((ret_col_dim / min_dim) + 1) * min_dim;

//	printf("Padding: row_dim %ld -> %ld, col_dim %ld -> %ld\n", orig_mat->row_dim, ret_row_dim, orig_mat->col_dim, ret_col_dim);

	ret_dev = init_gdsmatrix_dev(ret_row_dim, ret_col_dim);
	if(ret_dev == NULL)
	{
		fprintf(stderr, "Warning: padding matrix cannot be allocated!(padding_gdsmatrix_strassen)\n");
		return NULL;
	}

	// ret = [ A | 0 ]
	//       [---+---]
	//       [ 0 | 0 ]
	cudaMemcpy(ret_dev, orig_mat_dev, sizeof(gds_real) * (orig_mat_dev->row_dim * orig_mat_dev->col_dim), cudaMemcpyDeviceToDevice);
/*************************************************************
	#pragma omp parallel for private(j)
	for(i = 0; i < orig_mat->row_dim; i++)
	{
		for(j = 0; j < orig_mat->col_dim; j++)
			set_gdsmatrix_ij(ret, i, j, get_gdsmatrix_ij(orig_mat, i, j));
	}
*************************************************************/

//	min_dim = (ret_row_dim < ret_col_dim) ? ret_row_dim : ret_col_dim;
//	for(i = orig_mat->row_dim; i < min_dim; i++)
//		set_gdsmatrix_ij_ui(ret, i, i, 1UL);

	return ret_dev;
}

// Strassen's Algorithm with static padding
void _bncuda_mul_gdsmatrix_strassen_odd_padding(GDSMatrix ret_dev, GDSMatrix mat_a_dev, GDSMatrix mat_b_dev, long int min_dim, int num_blocks_per_grid, int num_threads_per_block)
{
	int thread_num, thread_index;
	long int tmp_ret_index[4], ret_index[4];
	long int *tmp_ret_index_dev, *ret_index_dev;
	GDSMatrix tmp_ret_dev, tmp_mat_a_dev, tmp_mat_b_dev;

	// padding
#ifdef USE_STATIC_PADDING
	tmp_ret = _bncuda_init_static_padding_gdsmatrix_strassen(ret);
	tmp_mat_a = _bncuda_init_static_padding_gdsmatrix_strassen(mat_a);
	tmp_mat_b = _bncuda_init_static_padding_gdsmatrix_strassen(mat_b);
#else
//	tmp_ret = _bncuda_init_dynamic_padding_gdsmatrix_strassen(ret);
//	tmp_mat_a = _bncuda_init_dynamic_padding_gdsmatrix_strassen(mat_a);
//	tmp_mat_b = _bncuda_init_dynamic_padding_gdsmatrix_strassen(mat_b);
	tmp_ret_dev = _bncuda_init_dynamic_padding_gdsmatrix_strassen2(ret_dev, min_dim);
	tmp_mat_a_dev = _bncuda_init_dynamic_padding_gdsmatrix_strassen2(mat_a_dev, min_dim);
	tmp_mat_b_dev = _bncuda_init_dynamic_padding_gdsmatrix_strassen2(mat_b_dev, min_dim);
#endif

	ret_index_dev = _bncuda_init_l_array(4);
	tmp_ret_index_dev = _bncuda_init_l_array(4);

	// strassen
#ifdef USE_WINOGRAD
//	_bncuda_mul_gdsmatrix_winograd_even(tmp_ret_dev, tmp_mat_a_dev, tmp_mat_b_dev, min_dim, num_blocks_per_grid, num_threads_per_block);
	_bncuda_mul_gdsmatrix_winograd_even_psec(tmp_ret_dev, tmp_mat_a_dev, tmp_mat_b_dev, min_dim, num_blocks_per_grid, num_threads_per_block);
#else
//	_bncuda_mul_gdsmatrix_strassen_even(tmp_ret_dev, tmp_mat_a_dev, tmp_mat_b_dev, min_dim, num_blocks_per_grid, num_threads_per_block);
	_bncuda_mul_gdsmatrix_strassen_even_psec(tmp_ret_dev, tmp_mat_a_dev, tmp_mat_b_dev, min_dim, num_blocks_per_grid, num_threads_per_block);
#endif

//	printf("tmp_ret->row_dim, col_dim: %ld, %ld\n", tmp_ret->row_dim, tmp_ret->col_dim);
//	printf("    ret->row_dim, col_dim: %ld, %ld\n", ret->row_dim, ret->col_dim);

	// substitute
	tmp_ret_index[0] = 0;
	tmp_ret_index[1] = ret_dev->row_dim;
	tmp_ret_index[2] = 0;
	tmp_ret_index[3] = ret_dev->col_dim;
	ret_index[0] = 0;
	ret_index[1] = ret_dev->row_dim;
	ret_index[2] = 0;
	ret_index[3] = ret_dev->col_dim;

	_bncuda_set_l_array(ret_index_dev, ret_index, 4);
	_bncuda_set_l_array(tmp_ret_index_dev, tmp_ret_index, 4);
	_bncuda_subst_gdsmatrix_partial(ret_dev, ret_index_dev, tmp_ret_dev, tmp_ret_index_dev, num_blocks_per_grid, num_threads_per_block);

	// free
	_bncuda_free_l_array(tmp_ret_index_dev);
	_bncuda_free_l_array(ret_index_dev);

	free_gdsmatrix(tmp_ret_dev);
	free_gdsmatrix(tmp_mat_a_dev);
	free_gdsmatrix(tmp_mat_b_dev);
}

// clear counter
void _bncuda_reset_num_mul_gdsmatrix_strassen(void)
{
	_bncuda_num_addsub_mul_gdsmatrix_strassen = 0;
	_bncuda_num_mul_mul_gdsmatrix_strassen = 0;
	_bncuda_total_num_threads_matmul_gdsmatrix_strassen = 0;
}

// get counters
void _bncuda_get_num_mul_gdsmatrix_strassen(long int *num_addsub, long int *num_mul)
{
	//printf("num_addsub_mul_gdsmatrix_strassen: %ld\n", num_addsub_mul_gdsmatrix_strassen);
	//printf("num_mul_mul_gdsmatrix_strassen   : %ld\n", num_mul_mul_gdsmatrix_strassen);

	if(num_addsub != NULL)
		*num_addsub = _bncuda_num_addsub_mul_gdsmatrix_strassen;
	if(num_mul != NULL)
		*num_mul = _bncuda_num_mul_mul_gdsmatrix_strassen;
}

// print counters
void _bncuda_print_num_mul_gdsmatrix_strassen(long int *num_addsub, long int *num_mul)
{
	printf("_bncuda_num_addsub_mul_gdsmatrix_strassen: %ld\n", _bncuda_num_addsub_mul_gdsmatrix_strassen);
	printf("_bncuda_num_mul_mul_gdsmatrix_strassen   : %ld\n", _bncuda_num_mul_mul_gdsmatrix_strassen);

	_bncuda_get_num_mul_gdsmatrix_strassen(num_addsub, num_mul);

}

// Fit dimension to be multiple of min_dim
void _bncuda_mul_gdsmatrix_strassen(GDSMatrix ret, GDSMatrix mat_a, GDSMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block)
{
	int thread_num, thread_index;
	long int row_k, col_k, mid_row_k, mid_col_k;
	long int row_dim, col_dim, mid_dim;
//	DSVector diag_left, diag_right;

#ifdef _OPENMP
	thread_num = omp_get_num_threads();
#else
	thread_num = 1;
#endif

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_gdsmatrix_strassen)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	// normal matrix multiplication in case of ret_dim <= 4 
	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	{
		//mul_gdsmatrix(ret, mat_a, mat_b);

		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		_bncuda_num_mul_mul_gdsmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		_bncuda_mul_gdsmatrix_simple(ret, mat_a, mat_b, num_blocks_per_grid, num_threads_per_block);

		return;
	}

	// scaling 
//	diag_left = init2_dsvector(row_dim, mat_a->prec);
//	diag_right = init2_dsvector(col_dim, mat_b->prec);

//	left_scaling_gdsmatrix(mat_a, diag_left, NULL);
//	right_scaling_gdsmatrix(mat_b, diag_right, NULL);

	// dynamic peeling in case of odd dim
	// [ A11   a12 ] [ B11   b12 ] = [ A11*B11 + a12 * b21^T   A11*b12 + a12 * b22    ]
	// [ a21^T a22 ] [ b21^T b22 ]   [ a21^T*B11 + a22 * b21^T a21^T * b12 + a22 * b22]
	if((ret->row_dim % 2 == 1) || (ret->col_dim % 2 == 1) || (mid_dim % 2 == 1))
	{
//#ifdef PEELING_ONLY
//		_bncuda_mul_gdsmatrix_strassen_odd_peeling(ret, mat_a, mat_b, min_dim, num_blocks_per_grid, num_threads_per_block);

//#elif PADDING_ONLY
		_bncuda_mul_gdsmatrix_strassen_odd_padding(ret, mat_a, mat_b, min_dim, num_blocks_per_grid, num_threads_per_block);

//#else
#if 0
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
			_bncuda_mul_gdsmatrix_strassen_odd_peeling(ret, mat_a, mat_b, min_dim, num_blocks_per_grid, num_threads_per_block);

		}
		// padding
		else
		{
			//printf("padding\n");
			_bncuda_mul_gdsmatrix_strassen_odd_padding(ret, mat_a, mat_b, min_dim, num_blocks_per_grid, num_threads_per_block);

		}
#endif // 0
//#endif

		//printf("end\n");
	}
	// normal strassen algorithm in case of even dim
	else
	{
		//printf("%d is even -> ", ret->row_dim);
#ifdef USE_WINOGRAD
		//_bncuda_mul_gdsmatrix_winograd_even(ret, mat_a, mat_b, min_dim);
		_bncuda_mul_gdsmatrix_winograd_even_psec(ret, mat_a, mat_b, min_dim, num_blocks_per_grid, num_threads_per_block);

#else
		//_bncuda_mul_gdsmatrix_strassen_even(ret, mat_a, mat_b, min_dim);
		_bncuda_mul_gdsmatrix_strassen_even_psec(ret, mat_a, mat_b, min_dim, num_blocks_per_grid, num_threads_per_block);

#endif
		//printf("end\n");
	}

//	mul_gdsmatrix_ddiag_mat(ret, diag_left, 0, ret, diag_right, 0);

//	free_dsvector(diag_left);
//	free_dsvector(diag_right);

}

// Strassen's Algorithm with Dynamic peeling
void _bncuda_mul_gdsmatrix_strassen_odd_peeling(GDSMatrix ret, GDSMatrix mat_a, GDSMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block)
{
/***********************************************************
	int thread_num, thread_index;
	long int i, j, row_dim, row_dim_h, col_dim, col_dim_h, mid_dim, mid_dim_h, tmp_dim_h, tmp_dim;
	GDSMatrix mat_a11, mat_b11, mat_c11, mat_tmp;
	GDSVector vec_a12, vec_a21, vec_b12, vec_b21, vec_c12, vec_c21, vec_tmp12, vec_tmp21;
	gds_real *a22, *b22, *c22, *tmp;

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_gdsmatrix_strassen_odd_peeling)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	a22 = _bncuda_init_gds();
	b22 = _bncuda_init_gds();
	c22 = _bncuda_init_gds();
	tmp = _bncuda_init_gds();

	set0_gds_dev(a22);
	set0_gds_dev(b22);
	set0_gds_dev(c22);
	set0_gds_dev(tmp);

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
		_bncuda_num_addsub_mul_gdsmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		_bncuda_num_mul_mul_gdsmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		//mul_gdsmatrix(ret, mat_a, mat_b);
		_bncuda_mul_gdsmatrix_simple(ret, mat_a, mat_b, num_blocks_per_grid, num_threads_per_block);

		return;
	}

	// tmp_dim_h = mid_dim_h or col_dim_h
	tmp_dim_h = mat_b->col_dim - 1;
	if(mid_dim_h < tmp_dim_h)
		tmp_dim_h = mid_dim_h;

	// Initialize
	mat_a11 = init_gdsmatrix_dev(row_dim_h, mid_dim_h);
	mat_b11 = init_gdsmatrix_dev(mid_dim_h, col_dim_h);
	mat_c11 = init_gdsmatrix_dev(row_dim_h, col_dim_h);
	mat_tmp = init_gdsmatrix_dev(row_dim_h, col_dim_h);

	vec_a12 = NULL;
	vec_b21 = NULL;
	vec_b12 = NULL;
	vec_a21 = NULL;
	vec_c12 = init_gdsvector_dev(row_dim_h);
	vec_c21 = init_gdsvector_dev(col_dim_h);
	vec_tmp12 = init_gdsvector_dev(row_dim_h);
	vec_tmp21 = init_gdsvector_dev(col_dim_h);

	// set matrix elements to mat_a11
	#pragma omp parallel for private(j)
	for(i = 0; i < row_dim_h; i++)
	{
		for(j = 0; j < mid_dim_h; j++)
			set_gdsmatrix_ij_dev(mat_a11, i, j, get_gdsmatrix_ij_dev(mat_a, i, j));
	}

	// set matrix elements to vec_b11
	#pragma omp parallel for private(j)
	for(i = 0; i < mid_dim_h; i++)
	{
		for(j = 0; j < col_dim_h; j++)
			set_gdsmatrix_ij_dev(mat_b11, i, j, get_gdsmatrix_ij_dev(mat_b, i, j));
	}

	// set matrix elements to vec_a12 and vec_b21
	if(mid_dim_h < mid_dim)
	{
		//printf("set vec_a12, b21\n");
		vec_a12 = init_dsvector_dev(row_dim_h);
		vec_b21 = init_dsvector_dev(col_dim_h); // fix!: 2014-03-19 by T.Kouya
		#pragma omp parallel for
		for(i = 0; i < row_dim_h; i++)
			set_dsvector_i_dev(vec_a12, i, get_gdsmatrix_ij_dev(mat_a, i, mat_a->col_dim - 1));

		//printf("set vec_a12\n");

		#pragma omp parallel for
		for(i = 0; i < col_dim_h; i++)
			set_dsvector_i_dev(vec_b21, i, get_gdsmatrix_ij_dev(mat_b, mat_b->row_dim - 1, i));

		//printf("set vec_b21\n");
	}

	// set matrix elements to vec_a21
	if(row_dim_h < row_dim)
	{
		//printf("set vec_a21, a22\n");
		vec_a21 = init_dsvector_dev(mid_dim_h);
		#pragma omp parallel for
		for(i = 0; i < mid_dim_h; i++)
			set_dsvector_i_dev(vec_a21, i, get_gdsmatrix_ij_dev(mat_a, mat_a->row_dim - 1, i));

		set0_dd_dev(a22);
		if(mid_dim_h < mid_dim)
			a22 = get_gdsmatrix_ij_dev(mat_a, mat_a->row_dim - 1, mat_a->col_dim - 1);

	}

	// set matrix elements to vec_b12
	if(col_dim_h < col_dim)
	{
		//printf("set vec_a12, b22\n");
		vec_b12 = init_dsvector_dev(mid_dim_h);

		#pragma omp parallel for
		for(i = 0; i < mid_dim_h; i++)
			set_dsvector_i_dev(vec_b12, i, get_gdsmatrix_ij_dev(mat_b, i, mat_b->col_dim - 1));

		set0_dd(b22);
		if(mid_dim_h < mid_dim)
			b22 = get_gdsmatrix_ij_dev(mat_b, mat_b->row_dim - 1, mat_b->col_dim - 1);
	}

	// dynamic peeling in case of odd dim
	// [ A11   a12 ] [ B11   b12 ] = [ A11*B11 + a12 * b21^T   A11*b12 + a12 * b22     ]
	// [ a21^T a22 ] [ b21^T b22 ]   [ a21^T*B11 + a22 * b21^T a21^T * b12 + a22 * b22 ]

	//printf("starting C11 = A11 * B11...\n");

	// C11 = A11 * B11
#ifdef USE_WINOGRAD
	//_bncuda_mul_gdsmatrix_winograd_even(mat_c11, mat_a11, mat_b11, min_dim);
	_bncuda_mul_gdsmatrix_winograd_even_psec(mat_c11, mat_a11, mat_b11, min_dim);
#else
//	_bncuda_mul_gdsmatrix_strassen_even(mat_c11, mat_a11, mat_b11, min_dim);
	_bncuda_mul_gdsmatrix_strassen_even_psec(mat_c11, mat_a11, mat_b11, min_dim);
#endif

	//printf("C11 = A11 * B11\n");

	// C11 += a12 * b21^T
	if((vec_a12 != NULL) && (vec_b21 != NULL))
	{
		//printf("starting C11 += a12 * b21...\n");

		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += row_dim_h * col_dim_h;
		_bncuda_num_mul_mul_gdsmatrix_strassen += row_dim_h * col_dim_h;

		#pragma omp parallel for private(j)
		for(i = 0; i < row_dim_h; i++)
		{
			for(j = 0; j < col_dim_h; j++)
			{
				rds_mul(get_gdsmatrix_ij_dev(mat_tmp, i, j), get_gdsvector_i_dev(vec_a12, i), get_gdsvector_i_dev(vec_b21, j));
			}
		}
		_bncuda_add_gdsmatrix(mat_c11, mat_c11, mat_tmp);
		//printf("C11 += a12 * b21...\n");
	}

	#pragma omp parallel for private(j)
	for(i = 0; i < row_dim_h; i++)
		for(j = 0; j < col_dim_h; j++) 
			set_gdsmatrix_ij_dev(ret, i, j, get_gdsmatrix_ij_dev(mat_c11, i, j));

	// c12 := A11 * b12
	if(vec_b12 != NULL)
	{
		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += mat_a11->row_dim * mat_a11->col_dim;
		_bncuda_num_mul_mul_gdsmatrix_strassen += mat_a11->row_dim * mat_a11->col_dim;

		_bncuda_mul_gdsmatrix_gdsvec(vec_c12, mat_a11, vec_b12);
		//mul_gdsmatrix_dsvec(vec_c12, mat_a11, vec_b12);

		//printf("c12 = A11 * b12\n");
	}

 	// c12 += b22 * a12
 	if((vec_a12 != NULL) && (rds_cmp_ui(b22, 0UL) != 0))
	{
		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += vec_c12->dim;
		_bncuda_num_mul_mul_gdsmatrix_strassen += vec_tmp12->dim;

		_bncuda_cmul_gdsvector(vec_tmp12, b22, vec_a12);
		_bncuda_add_gdsvector(vec_c12, vec_c12, vec_tmp12);
	}
	//printf("c12 += b22 * a12\n");

	if(vec_b12 != NULL)
	{
		#pragma omp parallel for
		for(i = 0; i < col_dim_h; i++)
			set_gdsmatrix_ij_dev(ret, i, ret->col_dim - 1, get_gdsvector_i_dev(vec_c12, i));
	}

	//printf("vec_c12\n");

	// c21 := a21^T*B11
	if(vec_a21 != NULL)
	{
		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += mat_b11->row_dim * mat_b11->col_dim;
		_bncuda_num_mul_mul_gdsmatrix_strassen += mat_b11->row_dim * mat_b11->col_dim;

		_bncuda_mul_gdsmatrixt_gdsvec_dev(vec_c21, mat_b11, vec_a21);
	}
	//printf("c21 = a21^T * B11\n");

	// c21 += a22 * b21^T
	if((vec_b21 != NULL) && (rds_cmp_ui(a22, 0UL) != 0))
	{
		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += vec_c21->dim;
		_bncuda_num_mul_mul_gdsmatrix_strassen += vec_tmp21->dim;

		_bncuda_cmul_gdsvector(vec_tmp21, a22, vec_b21);
		_bncuda_add_gdsvector(vec_c21, vec_c21, vec_tmp21);
	}
	//printf("c21 += a22 * b21^T\n");
	if(vec_a21 != NULL)
	{
		#pragma omp parallel for
		for(i = 0; i < col_dim_h; i++)
			set_gdsmatrix_ij(ret, ret->row_dim - 1, i, get_dsvector_i(vec_c21, i));
	}
	//printf("c21\n");

	// c22 := a21^T * b12
	if((vec_a21 != NULL) && (vec_b12 != NULL))
	{
		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += vec_a21->dim;
		_bncuda_num_mul_mul_gdsmatrix_strassen += vec_tmp21->dim;

		_bncuda_ip_gdsvector(&c22, vec_a21, vec_b12);
	}
	//printf("c22 += a21^T * b12\n");

	// c22 += a22 * b22
	if((rds_cmp_ui(a22, 0UL) != 0) || (rds_cmp_ui(b22, 0UL) != 0))
	{
		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += 1;
		_bncuda_num_mul_mul_gdsmatrix_strassen += 1;

		rds_mul(tmp, a22, b22);
		rds_add(c22, c22, tmp);
	}
	//printf("c22 += a22 * b22\n");

	if((vec_a21 != NULL) && (vec_b12 != NULL))
		set_gdsmatrix_ij_dev(ret, ret->row_dim - 1, ret->col_dim - 1, c22);

	//printf("c22\n");

	// free
	free_gdsmatrix_dev(mat_a11);
	free_gdsmatrix_dev(mat_b11);
	free_gdsmatrix_dev(mat_c11);
	//printf("free_gdsmatrix a11, b11, c11\n");
	free_gdsmatrix_dev(mat_tmp);

	//printf("free_gdsmatrix\n");

	if(vec_a12 != NULL)
		free_dsvector_dev(vec_a12);

	if(vec_b21 != NULL)
		free_dsvector_dev(vec_b21);

	if(vec_a21 != NULL)
		free_dsvector_dev(vec_a21);

	if(vec_b12 != NULL)
		free_dsvector_dev(vec_b12);

	//printf("free_dsvector\n");

	free_dsvector_dev(vec_c12);
	free_dsvector_dev(vec_c21);
	free_dsvector_dev(vec_tmp12);
	free_dsvector_dev(vec_tmp21);

	_bncuda_free_gds(a22);
	_bncuda_free_gds(b22);
	_bncuda_free_gds(c22);
	_bncuda_free_gds(tmp);

	//printf("free_dsvector2\n");
*************************************************************/
}

// Strassen's Algorithm with parallized sections
void _bncuda_mul_gdsmatrix_strassen_even_psec(GDSMatrix ret, GDSMatrix mat_a, GDSMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block)
{
	int thread_num, thread_index;
//	long int min_dim = 4; // = 2^2
	GDSMatrix mat_p[7], mat_tmp_a[7], mat_tmp_b[7], mat_tmp_c[4];
	long int row_dim_h, row_dim, col_dim_h, col_dim, mid_dim_h, mid_dim;
	long int ret_index[4], mat_tmp_a_index[4], mat_tmp_b_index[4], mat_ar_index[7][4], mat_al_index[7][4], mat_br_index[7][4], mat_bl_index[7][4], mat_c_index[4][4];
	long int *ret_index_dev, *mat_tmp_a_index_dev, *mat_tmp_b_index_dev, *mat_ar_index_dev[7], *mat_al_index_dev[7], *mat_br_index_dev[7], *mat_bl_index_dev[7], *mat_c_index_dev[4];
	long int i;

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_gdsmatrix_strassen_even)\n", mat_a->col_dim, mat_b->row_dim);
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
		_bncuda_num_addsub_mul_gdsmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		_bncuda_num_mul_mul_gdsmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		//mul_gdsmatrix(ret, mat_a, mat_b);
		_bncuda_mul_gdsmatrix_simple(ret, mat_a, mat_b, num_blocks_per_grid, num_threads_per_block);
		return;
	}

	#pragma omp parallel for
	for(i = 0; i < 7; i++)
	{
		mat_p[i] = init_gdsmatrix_dev(row_dim_h, col_dim_h);
		mat_tmp_a[i] = init_gdsmatrix_dev(row_dim_h, mid_dim_h);
		mat_tmp_b[i] = init_gdsmatrix_dev(mid_dim_h, col_dim_h);

	//	mat_ar_index_dev[i] = _bncuda_init_l_array(4);
	//	mat_al_index_dev[i] = _bncuda_init_l_array(4);
	//	mat_br_index_dev[i] = _bncuda_init_l_array(4);
	//	mat_bl_index_dev[i] = _bncuda_init_l_array(4);
	}
	
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
	{
		mat_tmp_c[i] = init_gdsmatrix_dev(row_dim_h, col_dim_h);
	//	mat_c_index_dev[i]  = _bncuda_init_l_array(4);
	}

	//ret_index_dev = _bncuda_init_l_array(4);
	//mat_tmp_a_index_dev = _bncuda_init_l_array(4);
	//mat_tmp_b_index_dev = _bncuda_init_l_array(4);

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

//	_bncuda_set_l_array(ret_index_dev, ret_index, 4);
//	_bncuda_set_l_array(mat_tmp_a_index_dev, mat_tmp_a_index, 4);
//	_bncuda_set_l_array(mat_tmp_b_index_dev, mat_tmp_b_index, 4);

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

		//_bncuda_set_l_array(mat_ar_index_dev[0], mat_ar_index[0], 4);
		//_bncuda_set_l_array(mat_al_index_dev[0], mat_al_index[0], 4);

		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += row_dim_h * mid_dim_h;

	//	add_gdsmatrix_partial(mat_tmp_a[0], mat_tmp_a_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);
		//_bncuda_add_gdsmatrix_partial(mat_tmp_a[0], mat_tmp_a_index_dev, mat_a, mat_ar_index_dev[0], mat_a, mat_al_index_dev[0], 1, 1);
		_bncuda_add_gdsmatrix_partial(mat_tmp_a[0], mat_tmp_a_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0], 1, 1);

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

		//_bncuda_set_l_array(mat_br_index_dev[0], mat_br_index[0], 4);
		//_bncuda_set_l_array(mat_bl_index_dev[0], mat_bl_index[0], 4);

		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += mid_dim_h * col_dim_h;

	//	add_gdsmatrix_partial(mat_tmp_b[0], mat_tmp_b_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);
		//_bncuda_add_gdsmatrix_partial(mat_tmp_b[0], mat_tmp_b_index_dev, mat_b, mat_br_index_dev[0], mat_b, mat_bl_index_dev[0], 1, 1);
		_bncuda_add_gdsmatrix_partial(mat_tmp_b[0], mat_tmp_b_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0], 1, 1);

		// P1 = tmp_a * tmp_b
		//printf("P1: %d, ", omp_get_thread_num());
		_bncuda_mul_gdsmatrix_strassen(mat_p[0], mat_tmp_a[0], mat_tmp_b[0], min_dim, num_blocks_per_grid, num_threads_per_block);

		//print_gdsmatrix_dev(mat_p[0]);
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

		//_bncuda_set_l_array(mat_ar_index_dev[1], mat_ar_index[1], 4);
		//_bncuda_set_l_array(mat_al_index_dev[1], mat_al_index[1], 4);

		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += row_dim_h * mid_dim_h;

	//	add_gdsmatrix_partial(mat_tmp_a[1], mat_tmp_a_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);
		//_bncuda_add_gdsmatrix_partial(mat_tmp_a[1], mat_tmp_a_index_dev, mat_a, mat_ar_index_dev[1], mat_a, mat_al_index_dev[1], 1, 1);
		_bncuda_add_gdsmatrix_partial(mat_tmp_a[1], mat_tmp_a_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1], 1, 1);

		// B11
		mat_br_index[1][0] = 0;
		mat_br_index[1][1] = mid_dim_h;
		mat_br_index[1][2] = 0;
		mat_br_index[1][3] = col_dim_h;

		//_bncuda_set_l_array(mat_br_index_dev[1], mat_br_index[1], 4);

	//	subst_gdsmatrix_partial(mat_tmp_b[1], mat_tmp_b_index, mat_b, mat_br_index[1]);
		//_bncuda_subst_gdsmatrix_partial(mat_tmp_b[1], mat_tmp_b_index_dev, mat_b, mat_br_index_dev[1], 1, 1);
		_bncuda_subst_gdsmatrix_partial(mat_tmp_b[1], mat_tmp_b_index, mat_b, mat_br_index[1], 1, 1);

		// P2 = tmp_a * tmp_b
		//printf("P2: %d, ", omp_get_thread_num());
		_bncuda_mul_gdsmatrix_strassen(mat_p[1], mat_tmp_a[1], mat_tmp_b[1], min_dim, num_blocks_per_grid, num_threads_per_block);

		//print_gdsmatrix_dev(mat_p[1]);

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

		//_bncuda_set_l_array(mat_ar_index_dev[2], mat_ar_index[2], 4);

	//	subst_gdsmatrix_partial(mat_tmp_a[2], mat_tmp_a_index, mat_a, mat_ar_index[2]);
		//_bncuda_subst_gdsmatrix_partial(mat_tmp_a[2], mat_tmp_a_index_dev, mat_a, mat_ar_index_dev[2], 1, 1);
		_bncuda_subst_gdsmatrix_partial(mat_tmp_a[2], mat_tmp_a_index, mat_a, mat_ar_index[2], 1, 1);

		// B12 - B22
		mat_br_index[2][0] = 0;
		mat_br_index[2][1] = mid_dim_h;
		mat_br_index[2][2] = col_dim_h;
		mat_br_index[2][3] = col_dim;

		mat_bl_index[2][0] = mid_dim_h;
		mat_bl_index[2][1] = mid_dim;
		mat_bl_index[2][2] = col_dim_h;
		mat_bl_index[2][3] = col_dim;

		//_bncuda_set_l_array(mat_br_index_dev[2], mat_br_index[2], 4);
		//_bncuda_set_l_array(mat_bl_index_dev[2], mat_bl_index[2], 4);

		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += mid_dim_h * col_dim_h;

	//	sub_gdsmatrix_partial(mat_tmp_b[2], mat_tmp_b_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);
		//_bncuda_sub_gdsmatrix_partial(mat_tmp_b[2], mat_tmp_b_index_dev, mat_b, mat_br_index_dev[2], mat_b, mat_bl_index_dev[2], 1, 1);
		_bncuda_sub_gdsmatrix_partial(mat_tmp_b[2], mat_tmp_b_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2], 1, 1);

		// P3 = tmp_a * tmp_b
		//printf("P3: %d, ", omp_get_thread_num());
		_bncuda_mul_gdsmatrix_strassen(mat_p[2], mat_tmp_a[2], mat_tmp_b[2], min_dim, num_blocks_per_grid, num_threads_per_block);

		//print_gdsmatrix_dev(mat_p[2]);
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

		//_bncuda_set_l_array(mat_ar_index_dev[3], mat_ar_index[3], 4);

		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += mid_dim_h * col_dim_h;

	//	subst_gdsmatrix_partial(mat_tmp_a[3], mat_tmp_a_index, mat_a, mat_ar_index[3]);
		//_bncuda_subst_gdsmatrix_partial(mat_tmp_a[3], mat_tmp_a_index_dev, mat_a, mat_ar_index_dev[3], 1, 1);
		_bncuda_subst_gdsmatrix_partial(mat_tmp_a[3], mat_tmp_a_index, mat_a, mat_ar_index[3], 1, 1);

		// B21 - B11
		mat_br_index[3][0] = mid_dim_h;
		mat_br_index[3][1] = mid_dim;
		mat_br_index[3][2] = 0;
		mat_br_index[3][3] = col_dim_h;

		mat_bl_index[3][0] = 0;
		mat_bl_index[3][1] = mid_dim_h;
		mat_bl_index[3][2] = 0;
		mat_bl_index[3][3] = col_dim_h;

		//_bncuda_set_l_array(mat_br_index_dev[3], mat_br_index[3], 4);
		//_bncuda_set_l_array(mat_bl_index_dev[3], mat_bl_index[3], 4);

	//	sub_gdsmatrix_partial(mat_tmp_b[3], mat_tmp_b_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);
		//_bncuda_sub_gdsmatrix_partial(mat_tmp_b[3], mat_tmp_b_index_dev, mat_b, mat_br_index_dev[3], mat_b, mat_bl_index_dev[3], 1, 1);
		_bncuda_sub_gdsmatrix_partial(mat_tmp_b[3], mat_tmp_b_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3], 1, 1);

		// P4 = tmp_a * tmp_b
		//printf("P4: %d, ", omp_get_thread_num());
		_bncuda_mul_gdsmatrix_strassen(mat_p[3], mat_tmp_a[3], mat_tmp_b[3], min_dim, num_blocks_per_grid, num_threads_per_block);

		//print_gdsmatrix_dev(mat_p[3]);
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

		//_bncuda_set_l_array(mat_ar_index_dev[4], mat_ar_index[4], 4);
		//_bncuda_set_l_array(mat_al_index_dev[4], mat_al_index[4], 4);

		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += row_dim_h * mid_dim_h;

	//	add_gdsmatrix_partial(mat_tmp_a[4], mat_tmp_a_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);
		//_bncuda_add_gdsmatrix_partial(mat_tmp_a[4], mat_tmp_a_index_dev, mat_a, mat_ar_index_dev[4], mat_a, mat_al_index_dev[4], 1, 1);
		_bncuda_add_gdsmatrix_partial(mat_tmp_a[4], mat_tmp_a_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4], 1, 1);

		// B22
		mat_br_index[4][0] = mid_dim_h;
		mat_br_index[4][1] = mid_dim;
		mat_br_index[4][2] = col_dim_h;
		mat_br_index[4][3] = col_dim;

		//_bncuda_set_l_array(mat_br_index_dev[4], mat_br_index[4], 4);

	//	subst_gdsmatrix_partial(mat_tmp_b[4], mat_tmp_b_index, mat_b, mat_br_index[4]);
		//_bncuda_subst_gdsmatrix_partial(mat_tmp_b[4], mat_tmp_b_index_dev, mat_b, mat_br_index_dev[4], 1, 1);
		_bncuda_subst_gdsmatrix_partial(mat_tmp_b[4], mat_tmp_b_index, mat_b, mat_br_index[4], 1, 1);

		// P5 = tmp_a * tmp_b
		//printf("P5: %d, ", omp_get_thread_num());
		_bncuda_mul_gdsmatrix_strassen(mat_p[4], mat_tmp_a[4], mat_tmp_b[4], min_dim, num_blocks_per_grid, num_threads_per_block);

		//print_gdsmatrix_dev(mat_p[4]);
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

		//_bncuda_set_l_array(mat_ar_index_dev[5], mat_ar_index[5], 4);
		//_bncuda_set_l_array(mat_al_index_dev[5], mat_al_index[5], 4);

		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += row_dim_h * mid_dim_h;

	//	sub_gdsmatrix_partial(mat_tmp_a[5], mat_tmp_a_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);
		//_bncuda_sub_gdsmatrix_partial(mat_tmp_a[5], mat_tmp_a_index_dev, mat_a, mat_ar_index_dev[5], mat_a, mat_al_index_dev[5], 1, 1);
		_bncuda_sub_gdsmatrix_partial(mat_tmp_a[5], mat_tmp_a_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5], 1, 1);

		// B11 + B12
		mat_br_index[5][0] = 0;
		mat_br_index[5][1] = mid_dim_h;
		mat_br_index[5][2] = 0;
		mat_br_index[5][3] = col_dim_h;

		mat_bl_index[5][0] = 0;
		mat_bl_index[5][1] = mid_dim_h;
		mat_bl_index[5][2] = col_dim_h;
		mat_bl_index[5][3] = col_dim;

		//_bncuda_set_l_array(mat_br_index_dev[5], mat_br_index[5], 4);
		//_bncuda_set_l_array(mat_bl_index_dev[5], mat_bl_index[5], 4);

		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += mid_dim_h * col_dim_h;

	//	add_gdsmatrix_partial(mat_tmp_b[5], mat_tmp_b_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);
		//_bncuda_add_gdsmatrix_partial(mat_tmp_b[5], mat_tmp_b_index_dev, mat_b, mat_br_index_dev[5], mat_b, mat_bl_index_dev[5], 1, 1);
		_bncuda_add_gdsmatrix_partial(mat_tmp_b[5], mat_tmp_b_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5], 1, 1);

		// P6 = tmp_a * tmp_b
		//printf("P6: %d, ", omp_get_thread_num());
		_bncuda_mul_gdsmatrix_strassen(mat_p[5], mat_tmp_a[5], mat_tmp_b[5], min_dim, num_blocks_per_grid, num_threads_per_block);

		//print_gdsmatrix_dev(mat_p[5]);
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

		//_bncuda_set_l_array(mat_ar_index_dev[6], mat_ar_index[6], 4);
		//_bncuda_set_l_array(mat_al_index_dev[6], mat_al_index[6], 4);

		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += row_dim_h * mid_dim_h;

	//	sub_gdsmatrix_partial(mat_tmp_a[6], mat_tmp_a_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);
		//_bncuda_sub_gdsmatrix_partial(mat_tmp_a[6], mat_tmp_a_index_dev, mat_a, mat_ar_index_dev[6], mat_a, mat_al_index_dev[6], 1, 1);
		_bncuda_sub_gdsmatrix_partial(mat_tmp_a[6], mat_tmp_a_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6], 1, 1);

		// B21 + B22
		mat_br_index[6][0] = mid_dim_h;
		mat_br_index[6][1] = mid_dim;
		mat_br_index[6][2] = 0;
		mat_br_index[6][3] = col_dim_h;

		mat_bl_index[6][0] = mid_dim_h;
		mat_bl_index[6][1] = mid_dim;
		mat_bl_index[6][2] = col_dim_h;
		mat_bl_index[6][3] = col_dim;

		//_bncuda_set_l_array(mat_br_index_dev[6], mat_br_index[6], 4);
		//_bncuda_set_l_array(mat_bl_index_dev[6], mat_bl_index[6], 4);

		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += mid_dim_h * col_dim_h;

	//	add_gdsmatrix_partial(mat_tmp_b[6], mat_tmp_b_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);
		//_bncuda_add_gdsmatrix_partial(mat_tmp_b[6], mat_tmp_b_index_dev, mat_b, mat_br_index_dev[6], mat_b, mat_bl_index_dev[6], 1, 1);
		_bncuda_add_gdsmatrix_partial(mat_tmp_b[6], mat_tmp_b_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6], 1, 1);

		// P7 = tmp_a * tmp_b
		//printf("P7: %d", omp_get_thread_num());
		_bncuda_mul_gdsmatrix_strassen(mat_p[6], mat_tmp_a[6], mat_tmp_b[6], min_dim, num_blocks_per_grid, num_threads_per_block);

		//print_gdsmatrix_dev(mat_p[6]);
	}
} // pragma omp parallel sections

//printf("\n");

//#pragma omp sections
#pragma omp parallel sections
{
	// -------------------------------
	// C11 := P1 + P4 - P5 + P7
	// -------------------------------
	#pragma omp section
	{

		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += 3 * row_dim_h * col_dim_h;

		add_gdsmatrix_dev(mat_tmp_c[0], mat_p[0], mat_p[3], num_blocks_per_grid, num_threads_per_block);
		sub_gdsmatrix_dev(mat_tmp_c[0], mat_tmp_c[0], mat_p[4], num_blocks_per_grid, num_threads_per_block);
		add_gdsmatrix_dev(mat_tmp_c[0], mat_tmp_c[0], mat_p[6], num_blocks_per_grid, num_threads_per_block);
		//_bncuda_add_gdsmatrix(mat_tmp_c[0], mat_p[0], mat_p[3]);
		//_bncuda_sub_gdsmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[4]);
		//_bncuda_add_gdsmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[6]);
		mat_c_index[0][0] = 0;
		mat_c_index[0][1] = row_dim_h;
		mat_c_index[0][2] = 0;
		mat_c_index[0][3] = col_dim_h;

		//_bncuda_set_l_array(mat_c_index_dev[0], mat_c_index[0], 4);

		//subst_gdsmatrix_partial(ret, mat_c_index[0], mat_tmp_c[0], ret_index);
		//_bncuda_subst_gdsmatrix_partial(ret, mat_c_index_dev[0], mat_tmp_c[0], ret_index_dev, 1, 1);
		_bncuda_subst_gdsmatrix_partial(ret, mat_c_index[0], mat_tmp_c[0], ret_index, num_blocks_per_grid, num_threads_per_block);
	}

	// -------------------------------
	// C12 := P3 + P5
	// -------------------------------
	#pragma omp section
	{
		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += row_dim_h * col_dim_h;

		add_gdsmatrix_dev(mat_tmp_c[1], mat_p[2], mat_p[4], num_blocks_per_grid, num_threads_per_block);
		//_bncuda_add_gdsmatrix(mat_tmp_c[1], mat_p[2], mat_p[4]);
		mat_c_index[1][0] = 0;
		mat_c_index[1][1] = row_dim_h;
		mat_c_index[1][2] = col_dim_h;
		mat_c_index[1][3] = col_dim;

		//_bncuda_set_l_array(mat_c_index_dev[1], mat_c_index[1], 4);

		//subst_gdsmatrix_partial(ret, mat_c_index[1], mat_tmp_c[1], ret_index);
		_bncuda_subst_gdsmatrix_partial(ret, mat_c_index[1], mat_tmp_c[1], ret_index, num_blocks_per_grid, num_threads_per_block);
	}

	// -------------------------------
	// C21 := P2 + P4
	// -------------------------------
	#pragma omp section
	{
		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += row_dim_h * col_dim_h;

		add_gdsmatrix_dev(mat_tmp_c[2], mat_p[1], mat_p[3], num_blocks_per_grid, num_threads_per_block);
		//_bncuda_add_gdsmatrix(mat_tmp_c[2], mat_p[1], mat_p[3]);
		mat_c_index[2][0] = row_dim_h;
		mat_c_index[2][1] = row_dim;
		mat_c_index[2][2] = 0;
		mat_c_index[2][3] = col_dim_h;

		//_bncuda_set_l_array(mat_c_index_dev[2], mat_c_index[2], 4);

		//subst_gdsmatrix_partial(ret, mat_c_index[2], mat_tmp_c[2], ret_index);
		_bncuda_subst_gdsmatrix_partial(ret, mat_c_index[2], mat_tmp_c[2], ret_index, num_blocks_per_grid, num_threads_per_block);
	}

	// -------------------------------
	// C22 := P1 + P3 - P2 + P6
	// -------------------------------
	#pragma omp section
	{
		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += 3 * row_dim_h * col_dim_h;

		add_gdsmatrix_dev(mat_tmp_c[3], mat_p[0], mat_p[2], 1, 1);
		sub_gdsmatrix_dev(mat_tmp_c[3], mat_tmp_c[3], mat_p[1], 1, 1);
		add_gdsmatrix_dev(mat_tmp_c[3], mat_tmp_c[3], mat_p[5], 1, 1);
		//_bncuda_add_gdsmatrix(mat_tmp_c[3], mat_p[0], mat_p[2], 1, 1);
		//_bncuda_sub_gdsmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[1]);
		//_bncuda_add_gdsmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[5]);
		mat_c_index[3][0] = row_dim_h;
		mat_c_index[3][1] = row_dim;
		mat_c_index[3][2] = col_dim_h;
		mat_c_index[3][3] = col_dim;

		//_bncuda_set_l_array(mat_c_index_dev[3], mat_c_index[3], 4);

		//subst_gdsmatrix_partial(ret, mat_c_index[3], mat_tmp_c[3], ret_index);
		_bncuda_subst_gdsmatrix_partial(ret, mat_c_index[3], mat_tmp_c[3], ret_index, num_blocks_per_grid, num_threads_per_block);
	}

} // pragma omp parallel sections

	// free
	#pragma omp parallel for
	for(i = 0; i < 7; i++)
	{
		free_gdsmatrix_dev(mat_p[i]);
		free_gdsmatrix_dev(mat_tmp_a[i]);
		free_gdsmatrix_dev(mat_tmp_b[i]);

	//	_bncuda_free_l_array(mat_ar_index_dev[i]);
	//	_bncuda_free_l_array(mat_al_index_dev[i]);
	//	_bncuda_free_l_array(mat_br_index_dev[i]);
	//	_bncuda_free_l_array(mat_bl_index_dev[i]);
	}

	#pragma omp parallel for
	for(i = 0; i < 4; i++)
	{
		free_gdsmatrix_dev(mat_tmp_c[i]);
	//	_bncuda_free_l_array(mat_c_index_dev[i]);
	}

	//_bncuda_free_l_array(ret_index_dev);
	//_bncuda_free_l_array(mat_tmp_a_index_dev);
	//_bncuda_free_l_array(mat_tmp_b_index_dev);

}

// Winograd Variant of Strassen's Algorithm with parallelied sections
void _bncuda_mul_gdsmatrix_winograd_even_psec(GDSMatrix ret, GDSMatrix mat_a, GDSMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block)
{
	int thread_num, thread_index;
//	long int min_dim = 4; // = 2^2
	GDSMatrix mat_s[8], mat_m[7], mat_t[2], mat_tmp_a[4], mat_tmp_b[4], mat_tmp_c[4];
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
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_gdsmatrix_winograd_even)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	row_dim_h = row_dim / 2;
	col_dim_h = col_dim / 2;
	mid_dim_h = mid_dim / 2;

	// normal matrix multiplication in case of ret_dim <= 4 
//	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	if((ret->row_dim <= min_dim) || (ret->col_dim <= min_dim) || (mid_dim <= min_dim))
	{
		//mul_gdsmatrix_simple(ret, mat_a, mat_b);
		_bncuda_mul_gdsmatrix_simple(ret, mat_a, mat_b, num_blocks_per_grid, num_threads_per_block);

		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		_bncuda_num_mul_mul_gdsmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		return;
	}

	#pragma omp parallel for
	for(i = 0; i < 4; i++)
	{
		mat_s[i] = init_gdsmatrix_dev(row_dim_h, mid_dim_h);
		mat_tmp_a[i] = init_gdsmatrix_dev(row_dim_h, mid_dim_h);

		mat_s[i + 4] = init_gdsmatrix_dev(mid_dim_h, col_dim_h);
		mat_tmp_b[i] = init_gdsmatrix_dev(mid_dim_h, col_dim_h);

		mat_tmp_c[i] = init_gdsmatrix_dev(row_dim_h, col_dim_h);
	}
	#pragma omp parallel for
	for(i = 0; i < 7; i++)
		mat_m[i] = init_gdsmatrix_dev(row_dim_h, col_dim_h);

	mat_t[0] = init_gdsmatrix_dev(row_dim_h, col_dim_h);
	mat_t[1] = init_gdsmatrix_dev(row_dim_h, col_dim_h);

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
		//subst_gdsmatrix_partial(mat_tmp_a[i], mat_tmp_a_index, mat_a, mat_a_index[i]);
		//subst_gdsmatrix_partial(mat_tmp_b[i], mat_tmp_b_index, mat_b, mat_b_index[i]);
		_bncuda_subst_gdsmatrix_partial(mat_tmp_a[i], mat_tmp_a_index, mat_a, mat_a_index[i], 1, 1);
		_bncuda_subst_gdsmatrix_partial(mat_tmp_b[i], mat_tmp_b_index, mat_b, mat_b_index[i], 1, 1);
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
		_bncuda_num_addsub_mul_gdsmatrix_strassen += row_dim_h * mid_dim_h;

		//add_gdsmatrix(mat_s[0], mat_tmp_a[2], mat_tmp_a[3]);
		add_gdsmatrix_dev(mat_s[0], mat_tmp_a[2], mat_tmp_a[3], 1, 1);
	}

	#pragma omp section
	{
		// -------------------------------
		// S3 := A11 - A21
		//--------------------------------
		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += row_dim_h * mid_dim_h;

		//sub_gdsmatrix(mat_s[2], mat_tmp_a[0], mat_tmp_a[2]);
		sub_gdsmatrix_dev(mat_s[2], mat_tmp_a[0], mat_tmp_a[2], 1, 1);
	}

	#pragma omp section
	{
		// -------------------------------
		// S5 := B12 - B11
		//--------------------------------
		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += mid_dim_h * col_dim_h;

		//sub_gdsmatrix(mat_s[4], mat_tmp_b[1], mat_tmp_b[0]);
		sub_gdsmatrix_dev(mat_s[4], mat_tmp_b[1], mat_tmp_b[0], 1, 1);
	}

	#pragma omp section
	{
		// -------------------------------
		// S7 := B22 - B12
		//--------------------------------
		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += mid_dim_h * col_dim_h;

		//sub_gdsmatrix(mat_s[6], mat_tmp_b[3], mat_tmp_b[1]);
		sub_gdsmatrix_dev(mat_s[6], mat_tmp_b[3], mat_tmp_b[1], 1, 1);
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
		_bncuda_num_addsub_mul_gdsmatrix_strassen += row_dim_h * mid_dim_h;

		//sub_gdsmatrix(mat_s[1], mat_s[0], mat_tmp_a[0]);
		sub_gdsmatrix_dev(mat_s[1], mat_s[0], mat_tmp_a[0], 1, 1);

		// -------------------------------
		// S4 := A12 - S2
		//--------------------------------
		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += row_dim_h * mid_dim_h;

		//sub_gdsmatrix(mat_s[3], mat_tmp_a[1], mat_s[1]);
		sub_gdsmatrix_dev(mat_s[3], mat_tmp_a[1], mat_s[1], 1, 1);
	}

	#pragma omp section
	{
		// -------------------------------
		// S6 := B22 - S5
		//--------------------------------
		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += mid_dim_h * col_dim_h;

		//sub_gdsmatrix(mat_s[5], mat_tmp_b[3], mat_s[4]);
		sub_gdsmatrix_dev(mat_s[5], mat_tmp_b[3], mat_s[4], 1, 1);

		// -------------------------------
		// S8 := S6 - B21
		//--------------------------------
		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += mid_dim_h * col_dim_h;

		//sub_gdsmatrix(mat_s[7], mat_s[5], mat_tmp_b[2]);
		sub_gdsmatrix_dev(mat_s[7], mat_s[5], mat_tmp_b[2], 1, 1);
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
		_bncuda_mul_gdsmatrix_strassen(mat_m[0], mat_s[1], mat_s[5], min_dim, num_blocks_per_grid, num_threads_per_block);
	}

	#pragma omp section
	{
		// -------------------------------
		// M2 := A11 * B11
		// -------------------------------
		_bncuda_mul_gdsmatrix_strassen(mat_m[1], mat_tmp_a[0], mat_tmp_b[0], min_dim, num_blocks_per_grid, num_threads_per_block);
	}

	#pragma omp section
	{
		// -------------------------------
		// M3 := A12 * B21
		// -------------------------------
		_bncuda_mul_gdsmatrix_strassen(mat_m[2], mat_tmp_a[1], mat_tmp_b[2], min_dim, num_blocks_per_grid, num_threads_per_block);
	}

	#pragma omp section
	{
		// -------------------------------
		// M4 := S3 * S7
		// -------------------------------
		_bncuda_mul_gdsmatrix_strassen(mat_m[3], mat_s[2], mat_s[6], min_dim, num_blocks_per_grid, num_threads_per_block);
	}

	#pragma omp section
	{
		// -------------------------------
		// M5 := S1 * S5
		// -------------------------------
		_bncuda_mul_gdsmatrix_strassen(mat_m[4], mat_s[0], mat_s[4], min_dim, num_blocks_per_grid, num_threads_per_block);
	}

	#pragma omp section
	{
		// -------------------------------
		// M6 := S4 * B22
		// -------------------------------
		_bncuda_mul_gdsmatrix_strassen(mat_m[5], mat_s[3], mat_tmp_b[3], min_dim, num_blocks_per_grid, num_threads_per_block);
	}

	#pragma omp section
	{
		// -------------------------------
		// M7 := A22 * S8
		// -------------------------------
		_bncuda_mul_gdsmatrix_strassen(mat_m[6], mat_tmp_a[3], mat_s[7], min_dim, num_blocks_per_grid, num_threads_per_block);
	}
} // pragma omp parallel sections

//	printf("m...\n");

	// -------------------------------
	// T1 := M1 + M2
	// -------------------------------
	// counting the number of arithmetic
	_bncuda_num_addsub_mul_gdsmatrix_strassen += row_dim_h * col_dim_h;

	//add_gdsmatrix(mat_t[0], mat_m[0], mat_m[1]);
	add_gdsmatrix_dev(mat_t[0], mat_m[0], mat_m[1], num_blocks_per_grid, num_threads_per_block);

	// -------------------------------
	// T2 := T1 + M4
	// -------------------------------
	// counting the number of arithmetic
	_bncuda_num_addsub_mul_gdsmatrix_strassen += row_dim_h * col_dim_h;

	//add_gdsmatrix(mat_t[1], mat_t[0], mat_m[3]);
	add_gdsmatrix_dev(mat_t[1], mat_t[0], mat_m[3], num_blocks_per_grid, num_threads_per_block);

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
		_bncuda_num_addsub_mul_gdsmatrix_strassen += row_dim_h * col_dim_h;

		//add_gdsmatrix(mat_tmp_c[0], mat_m[1], mat_m[2]);
		add_gdsmatrix_dev(mat_tmp_c[0], mat_m[1], mat_m[2], num_blocks_per_grid, num_threads_per_block);
	}

	#pragma omp section
	{
		// -------------------------------
		// C12 := T1 + M5 + M6
		// -------------------------------
		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += 2 * row_dim_h * col_dim_h;

		//add_gdsmatrix(mat_tmp_c[1], mat_t[0], mat_m[4]);
		//add_gdsmatrix(mat_tmp_c[1], mat_tmp_c[1], mat_m[5]);
		add_gdsmatrix_dev(mat_tmp_c[1], mat_t[0], mat_m[4], 1, 1);
		add_gdsmatrix_dev(mat_tmp_c[1], mat_tmp_c[1], mat_m[5], num_blocks_per_grid, num_threads_per_block);
	}

	#pragma omp section
	{
		// -------------------------------
		// C21 := T2 - M7
		// -------------------------------
		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += row_dim_h * col_dim_h;

		//sub_gdsmatrix(mat_tmp_c[2], mat_t[1], mat_m[6]);
		sub_gdsmatrix_dev(mat_tmp_c[2], mat_t[1], mat_m[6], num_blocks_per_grid, num_threads_per_block);
	}

	#pragma omp section
	{
		// -------------------------------
		// C22 := T2 + M5
		// -------------------------------
		// counting the number of arithmetic
		_bncuda_num_addsub_mul_gdsmatrix_strassen += row_dim_h * col_dim_h;

		//add_gdsmatrix(mat_tmp_c[3], mat_t[1], mat_m[4]);
		add_gdsmatrix_dev(mat_tmp_c[3], mat_t[1], mat_m[4], num_blocks_per_grid, num_threads_per_block);
	}
} // pragma omp parallel sections

//	printf("c...\n");

	// -------------------------------
	// RET := [C11 C12]
	//        [C21 C22]
	// -------------------------------
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
		_bncuda_subst_gdsmatrix_partial(ret, mat_c_index[i], mat_tmp_c[i], ret_index, num_blocks_per_grid, num_threads_per_block);
		//subst_gdsmatrix_partial(ret, mat_c_index[i], mat_tmp_c[i], ret_index);

//	printf("set...\n");


	// free
	#pragma omp parallel for
	for(i = 0; i < 4; i++)
	{
		free_gdsmatrix_dev(mat_s[i]);
		free_gdsmatrix_dev(mat_tmp_a[i]);
		free_gdsmatrix_dev(mat_s[i + 4]);
		free_gdsmatrix_dev(mat_tmp_b[i]);
		free_gdsmatrix_dev(mat_tmp_c[i]);
	}
	#pragma omp parallel for
	for(i = 0; i < 7; i++)
		free_gdsmatrix_dev(mat_m[i]);
	
	free_gdsmatrix_dev(mat_t[0]);
	free_gdsmatrix_dev(mat_t[1]);
}

// The following main function is for debugging
#ifdef DEBUG

int main(int argc, char *argv[])
{
	int omp_num_threads = 1, num_threads = 1, num_blocks = 1;
	long int i, j, row_dim, col_dim, mid_dim, min_dim = 32;
	unsigned long prec = 128;
	dsfloat ddtmp[4];
	gds_real *ddtmp_dev;
	float cuda_etime[4];
	cudaEvent_t start, stop;

	DSMatrix dda, ddb, ddc;
	GDSMatrix mpfc, mpfa, mpfb, mpfc_normal, mpfc_block, mpfc_nonrec, mpfc_tmp;
	GDSVector mpfdiag_left, mpfdiag_right;
	double stime, etime[4], reldiff[4];

//	dim = 128;
	if(argc <= 4)
	{
		fprintf(stderr, "Usage: %s [row_dim] [col_dim] [mid_dim] [min_dim] [#omp_threads] [#cuda_block] [#cuda_thread] \n", argv[0]);
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
		omp_num_threads = (unsigned long)atol(argv[5]);
		if(omp_num_threads < 1)
			omp_num_threads = 1;
	}

	if(argc >= 7)
	{
		num_blocks = (unsigned long)atol(argv[6]);
		if(num_blocks < 1)
			num_blocks = 1;
	}

	if(argc >= 8)
	{
		num_threads = (unsigned long)atol(argv[7]);
		if(num_threads < 1)
			num_threads = 1;
	}

/* double-double precision */
	set_bncomp_num_threads(omp_num_threads);
//	num_threads = 64;
//	num_blocks = 8;
	printf("OpenMP #Threads = %ld\n", omp_get_num_threads());
	printf("CUDA #Blocks x #Threads = %ld x %ld \n", num_blocks, num_threads);

	// Initialize QD library
	fpu_fix_start(NULL);

	// Initialize GQS library to prepare to be available GDS type
	GDSStart();

	ddtmp_dev = _bncuda_init_gds_array(4);
	set0_dd(ddtmp[0]);
	set0_dd(ddtmp[1]);

	dda = init_dsmatrix(row_dim, col_dim);
	ddb = init_dsmatrix(row_dim, col_dim);
	ddc = init_dsmatrix(row_dim, col_dim);

	mpfc = init_gdsmatrix_dev(row_dim, col_dim);
	mpfc_tmp = init_gdsmatrix_dev(row_dim, col_dim);
	mpfc_normal = init_gdsmatrix_dev(row_dim, col_dim);
	mpfc_block = init_gdsmatrix_dev(row_dim, col_dim);
	mpfc_nonrec = init_gdsmatrix(row_dim, col_dim);
	mpfa = init_gdsmatrix_dev(row_dim, mid_dim);
	mpfb = init_gdsmatrix_dev(mid_dim, col_dim);

//	mpfdiag_left = init_dsvector(dim);
//	mpfdiag_right = init_dsvector(dim);

	// set A
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < mid_dim; j++)
		{
			rds_set_d(ddtmp[0], (double)rand());
			if(rand() % 2 != 0)
				rds_neg(ddtmp[0], ddtmp[0]);

			rds_set_d(ddtmp[1], (double)rand());
			rds_ui_div(ddtmp[1], 1UL, ddtmp[1]);
			if(rand() % 2 != 0)
				rds_neg(ddtmp[1], ddtmp[1]);

			set_gdsmatrix_ij_ds_dev(mpfa, i, j, ddtmp[0]);
			set_dsmatrix_ij(dda, i, j, ddtmp[0]);
		}
	}

	// set B
	for(i = 0; i < mid_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rds_set_d(ddtmp[0], (double)rand());
			if(rand() % 2 != 0)
				rds_neg(ddtmp[0], ddtmp[0]);

			rds_set_d(ddtmp[1], (double)rand());
			rds_ui_div(ddtmp[1], 1UL, ddtmp[1]);
			if(rand() % 2 != 0)
				rds_neg(ddtmp[1], ddtmp[1]);

			set_gdsmatrix_ij_ds_dev(mpfb, i, j, ddtmp[1]);
			set_dsmatrix_ij(ddb, i, j, ddtmp[1]);
		}
	}

	// normal matrix mul
	printf("_bncuda_mul_gdsmatrix_simple...\n");
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	stime = get_real_secv();
	_bncuda_mul_gdsmatrix_simple(mpfc_normal, mpfa, mpfb, num_blocks, num_threads);
	etime[0] = get_real_secv() - stime;
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&cuda_etime[0], start, stop); // Unit: ms
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//left_scaling_gdsmatrix(mpfa, mpfa, mpfdiag_left, NULL);
	//right_scaling_gdsmatrix(mpfb, mpfb, mpfdiag_right, NULL);

	// blocked matrix mul
	printf("_bncuda_mul_gdsmatrix_block(%ld)...\n", min_dim);
	//stime = get_secv();
	stime = get_real_secv();
	_bncuda_mul_gdsmatrix_block(mpfc_block, mpfa, mpfb, min_dim, num_blocks, num_threads);
	//etime[1] = get_secv() - stime;
	etime[1] = get_real_secv() - stime;

	// Strassen 
	printf("_bncuda_mul_gdsmatrix_strassen(%ld)...\n", min_dim);
	stime = get_real_secv();
	_bncuda_mul_gdsmatrix_strassen(mpfc, mpfa, mpfb, min_dim, num_blocks, num_threads);
//	mul_gdsmatrix_strassen(mpfc, mpfa, mpfb, 16);
//	mul_gdsmatrix_strassen(mpfc, mpfa, mpfb, 32);
	etime[2] = get_real_secv() - stime;

	//mul_gdsmatrix_dddiag(mpfc, mpfdiag_left, 0, mpfc, mpfdiag_right, 0);

/*
	printf("Simple (%ld x %ld): \n", mpfc_normal->row_dim, mpfc_normal->col_dim);
	print_gdsmatrix_dev(mpfc_normal);
	printf("block (%ld x %ld): \n", mpfc_block->row_dim, mpfc_block->col_dim);
	print_gdsmatrix_dev(mpfc_block);
	printf("Strassen (%ld x %ld): \n", mpfc->row_dim, mpfc->col_dim);
	print_gdsmatrix_dev(mpfc);
*/
	// difference
//	sub_gdsmatrix_dev(mpfc_nonrec , mpfc_normal, mpfc_normal, num_blocks, num_threads);
	sub_gdsmatrix_dev(mpfc_tmp  , mpfc_normal, mpfc, num_blocks, num_threads);
	sub_gdsmatrix_dev(mpfc      , mpfc_normal, mpfc_block, num_blocks, num_threads);

	// print
	printf("row_dim, col_dim, mid_dim: %ld, %ld, %ld\n", row_dim, col_dim, mid_dim);
	printf("normal         : %f (%f)\n", cuda_etime[0] / 1000.0, etime[0]);
	printf("block          : %f\n", etime[1]);
	printf("strassen       : %f\n", etime[2]);

	normf_gdsmatrix_dev(&ddtmp_dev[0], mpfc_tmp, num_blocks, num_threads);
	normf_gdsmatrix_dev(&ddtmp_dev[1], mpfc, num_blocks, num_threads);
	//normf_gdsmatrix_dev(&ddtmp_dev[1], mpfc_nonrec, num_blocks, num_threads);
	normf_gdsmatrix_dev(&ddtmp_dev[2], mpfc_block, num_blocks, num_threads);
	normf_gdsmatrix_dev(&ddtmp_dev[3], mpfc_normal, num_blocks, num_threads);

	gds2ds_dev(&ddtmp[0], &ddtmp_dev[0]);
	gds2ds_dev(&ddtmp[1], &ddtmp_dev[1]);
	gds2ds_dev(&ddtmp[2], &ddtmp_dev[2]);
	gds2ds_dev(&ddtmp[3], &ddtmp_dev[3]);

	rds_div(ddtmp[0], ddtmp[0], ddtmp[3]);
	rds_div(ddtmp[1], ddtmp[1], ddtmp[3]);
	printf("||C               ||       : "); rds_out_str(ddtmp[3]); printf("\n");
	printf("||C_block         ||       : "); rds_out_str(ddtmp[2]); printf("\n");
	printf("||reldiff_block   ||       : "); rds_out_str(ddtmp[1]); printf("\n");
	printf("||reldiff_strassen||       : "); rds_out_str(ddtmp[0]); printf("\n");

/* Inverse */

//	frank_gdsmatrix(mpfa, dim);
//	frank_gdsmatrix(mpfb, dim);
//	lotkin_gdsmatrix(mpfa, dim);
//	lotkin_gdsmatrix(mpfb, dim);

	free_gdsmatrix_dev(mpfc);
	free_gdsmatrix_dev(mpfc_tmp);
	free_gdsmatrix_dev(mpfc_normal);
	free_gdsmatrix_dev(mpfc_block);
	free_gdsmatrix_dev(mpfc_nonrec);
	free_gdsmatrix_dev(mpfa);
	free_gdsmatrix_dev(mpfb);

//	free_dsvector(mpfdiag_left);
//	free_dsvector(mpfdiag_right);

	free_dsmatrix(dda);
	free_dsmatrix(ddb);
	free_dsmatrix(ddc);

	_bncuda_free_gds_array(ddtmp_dev);

	// End of GQS
	GDSEnd();

	return 0;
}
#endif // DEBUG
