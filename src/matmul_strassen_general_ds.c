/********************************************************************************/
/* matmul_strassen_general_ds.c:                                                */
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
#include <stdio.h>
#include <math.h>

//#include "bnc.h"
//#include "dslinear.h"

#ifdef USE_IMKL
	#include "mkl.h"
	#include "mkl_cblas.h" // for Intel Math Kernel Library
#endif

#include "matmul_strassen.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// count the number of computations
long int num_addsub_mul_dsmatrix_strassen;	// addition and subtraction
long int num_mul_mul_dsmatrix_strassen;		// multiplication

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void add_dsmatrix_partial(DSMatrix ret, long int ret_index[4], DSMatrix mat_a, long int mat_a_index[4], DSMatrix mat_b, long int mat_b_index[4])
{
	long int i, j, ret_i, ret_j, a_i, a_j, b_i, b_j;
	long int imax, jmax;
#ifdef __cplusplus
	static float tmp_val[DSSIZE];
#else // __cplusplus
	float tmp_val[DSSIZE];
#endif // __cpplusplus

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	set0_ds(tmp_val); // <-- Fix! 2015-06-17 by T.Kouya

	for(i = 0; i < imax; i++)
	{
		ret_i = ret_index[0] + i;
		a_i = mat_a_index[0] + i;
		b_i = mat_b_index[0] + i;
		//printf("i: %ld %ld %ld\n", ret_i, a_i, b_i);
		for(j = 0; j < jmax; j++)
		{
			ret_j = ret_index[2] + j;
			a_j = mat_a_index[2] + j;
			b_j = mat_b_index[2] + j;
			//printf("j: %ld %ld %ld\n", ret_j, a_j, b_j);

			//tmp_val = get_dsmatrix_ij(mat_a, a_i, a_j) + get_dsmatrix_ij(mat_b, b_i, b_j);
			rds_add(tmp_val, get_dsmatrix_ij(mat_a, a_i, a_j), get_dsmatrix_ij(mat_b, b_i, b_j));
			set_dsmatrix_ij(ret, ret_i, ret_j, tmp_val);
		}
	}
}

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void sub_dsmatrix_partial(DSMatrix ret, long int ret_index[4], DSMatrix mat_a, long int mat_a_index[4], DSMatrix mat_b, long int mat_b_index[4])
{
	long int i, j, ret_i, ret_j, a_i, a_j, b_i, b_j;
	long int imax, jmax;
#ifdef __cplusplus
	static float tmp_val[DSSIZE];
#else // __cplusplus
	float tmp_val[DSSIZE];
#endif // _cplusplus

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	set0_ds(tmp_val);

	for(i = 0; i < imax; i++)
	{
		ret_i = ret_index[0] + i;
		a_i = mat_a_index[0] + i;
		b_i = mat_b_index[0] + i;
		for(j = 0; j < jmax; j++)
		{
			ret_j = ret_index[2] + j;
			a_j = mat_a_index[2] + j;
			b_j = mat_b_index[2] + j;

			//tmp_val = get_dsmatrix_ij(mat_a, a_i, a_j) - get_dsmatrix_ij(mat_b, b_i, b_j);
			rds_sub(tmp_val, get_dsmatrix_ij(mat_a, a_i, a_j), get_dsmatrix_ij(mat_b, b_i, b_j));
			set_dsmatrix_ij(ret, ret_i, ret_j, tmp_val);
		}
	}
}

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_dsmatrix_partial(DSMatrix ret, long int ret_index[4], DSMatrix mat_a, long int mat_a_index[4])
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

			set_dsmatrix_ij(ret, ret_i, ret_j, get_dsmatrix_ij(mat_a, a_i, a_j));
		}
	}
}

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_dsmatrix_partial_checked(DSMatrix ret, long int ret_index[4], DSMatrix mat_a, long int mat_a_index[4])
{
	long int i, j, ret_i, ret_j, a_i, a_j;
	long int imax, jmax;
#ifdef __cplusplus
	static float ds_tmp[DSSIZE];
#else // __cplusplus
	float *ptr_dstmp;
#endif // _cplusplus

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
					{
					#ifdef __cplusplus
						set_dsmatrix_ij(ret, ret_i, ret_j, get_dsmatrix_ij(mat_a, a_i, a_j));
					#else // __cplusplus
						//ptr_dstmp = get_dsmatrix_ij(mat_a, a_i, a_j);
						//set_dsmatrix_ij(ret, ret_i, ret_j, ptr_dstmp);
						set_dsmatrix_ij(ret, ret_i, ret_j, get_dsmatrix_ij(mat_a, a_i, a_j));
					#endif // __cplusplus
					}
					else
						set0_dsmatrix_ij(ret, ret_i, ret_j); // Padding
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
void neg_dsmatrix_partial(DSMatrix ret, long int ret_index[4], DSMatrix mat_a, long int mat_a_index[4])
{
	long int i, j, ret_i, ret_j, a_i, a_j;
	long int imax, jmax;
#ifdef __cplusplus
	float tmp[DSSIZE];
#else
	float tmp[DSSIZE];
#endif // __cplusplus

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	set0_ds(tmp);

	for(i = 0; i < imax; i++)
	{
		ret_i = ret_index[0] + i;
		a_i = mat_a_index[0] + i;
		for(j = 0; j < jmax; j++)
		{
			ret_j = ret_index[2] + j;
			a_j = mat_a_index[2] + j;
#ifdef __cplusplus
			set_dsmatrix_ij(ret, ret_i, ret_j, -get_dsmatrix_ij(mat_a, a_i, a_j));
#else // __cplusplus
			rds_neg(tmp, get_dsmatrix_ij(mat_a, a_i, a_j));
			set_dsmatrix_ij(ret, ret_i, ret_j, tmp);
#endif // __cplusplus
		}
	}
}

/* c = a * b */
/* void mul_dsmatrix_simple(DSMatrix c, DSMatrix a, DSMatrix b)
{
	mul_dsmatrix(c, a, b);
}
*/

// Block matrix multiplicaiton
void mul_dsmatrix_block(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim)
{
	int row_padding_flag = 0, col_padding_flag = 0, mid_padding_flag = 0;
	long i, j, k, row_dim, col_dim, mid_dim;
	long int num_div_row, num_div_col, num_div_mid;
	long int mat_a_index[4], small_mat_a_index[4], mat_b_index[4], small_mat_b_index[4], ret_index[4], small_ret_index[4];
	//DMatrix small_ret[1024], small_mat_a[1024], small_mat_b[1024], small_tmp_mat;
	DSMatrix *small_ret, *small_mat_a, *small_mat_b, small_tmp_mat;

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_dsmatrix_block)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}
	

	// normal matrix multiplication in case of ret_dim <= 4 
	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	{
		mul_dsmatrix_simple(ret, mat_a, mat_b);
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
	small_ret = (DSMatrix *)calloc(sizeof(DSMatrix), num_div_col);
	small_mat_a = (DSMatrix *)calloc(sizeof(DSMatrix), num_div_mid);
	small_mat_b = (DSMatrix *)calloc(sizeof(DSMatrix), num_div_mid);
	for(i = 0; i < num_div_col; i++)
		small_ret[i] = init_dsmatrix(min_dim, min_dim);

	for(i = 0; i < num_div_mid; i++)
	{
		small_mat_a[i] = init_dsmatrix(min_dim, min_dim);
		small_mat_b[i] = init_dsmatrix(min_dim, min_dim);
	}
	small_tmp_mat = init_dsmatrix(min_dim, min_dim);

	// mail loop
	for(i = 0; i < num_div_row; i++)
	{
		mat_a_index[0] = i * min_dim;
		mat_a_index[1] = (i + 1) * min_dim;
		for(j = 0; j < num_div_mid; j++)
		{
			// copy matrices
			mat_a_index[2] = j * min_dim;
			mat_a_index[3] = (j + 1) * min_dim;
			small_mat_a_index[0] = 0;
			small_mat_a_index[1] = min_dim;
			small_mat_a_index[2] = 0;
			small_mat_a_index[3] = min_dim;
			//subst_dsmatrix_partial(small_mat_a[j], small_mat_a_index, mat_a, mat_a_index);
			subst_dsmatrix_partial_checked(small_mat_a[j], small_mat_a_index, mat_a, mat_a_index);
		}

		for(j = 0; j < num_div_col; j++)
		{

			set0_dsmatrix(small_ret[j]);
			mat_b_index[2] = j * min_dim;
			mat_b_index[3] = (j + 1) * min_dim;
			for(k = 0; k < num_div_mid; k++)
			{
				// copy matrices
				mat_b_index[0] = k * min_dim;
				mat_b_index[1] = (k + 1) * min_dim;
				small_mat_b_index[0] = 0;
				small_mat_b_index[1] = min_dim;
				small_mat_b_index[2] = 0;
				small_mat_b_index[3] = min_dim;
				//subst_dsmatrix_partial(small_mat_b[k], small_mat_b_index, mat_b, mat_b_index);
				subst_dsmatrix_partial_checked(small_mat_b[k], small_mat_b_index, mat_b, mat_b_index);

				// ret[j] += small_mat_a[i][k] * small_mat_b[k][j];
				mul_dsmatrix(small_tmp_mat, small_mat_a[k], small_mat_b[k]);
				add_dsmatrix(small_ret[j], small_ret[j], small_tmp_mat);
			}
			ret_index[0] = i * min_dim;
			ret_index[1] = (i + 1) * min_dim;
			ret_index[2] = j * min_dim;
			ret_index[3] = (j + 1) * min_dim;
			small_ret_index[0] = 0;
			small_ret_index[1] = min_dim;
			small_ret_index[2] = 0;
			small_ret_index[3] = min_dim;
			//subst_dsmatrix_partial(ret, ret_index, small_ret[j], small_ret_index);
			subst_dsmatrix_partial_checked(ret, ret_index, small_ret[j], small_ret_index);
		}
	}

	// free
	free_dsmatrix(small_tmp_mat);
	for(i = 0; i < num_div_col; i++)
		free_dsmatrix(small_ret[i]);

	for(i = 0; i < num_div_mid; i++)
	{
		free_dsmatrix(small_mat_a[i]);
		free_dsmatrix(small_mat_b[i]);
	}
	free(small_ret);
	free(small_mat_a);
	free(small_mat_b);
}

// Padding to 2-powered dimensional matrix
DSMatrix init_static_padding_dsmatrix_strassen(DSMatrix orig_mat)
{
	DSMatrix ret = NULL;
	long int ret_row_dim, ret_col_dim, min_dim, i, j;

	if(orig_mat == NULL)
	{
		fprintf(stderr, "Warning: orig_mat is empty!(padding_dsmatrix_strassen)\n");
		return NULL;
	}

	ret_row_dim = (long int)pow(2.0, ceil(mylog2((double)orig_mat->row_dim)));
	ret_col_dim = (long int)pow(2.0, ceil(mylog2((double)orig_mat->col_dim)));

	//printf("Padding: row_dim %ld -> %ld, col_dim %ld -> %ld\n", orig_mat->row_dim, ret_row_dim, orig_mat->col_dim, ret_col_dim);

	ret = init_dsmatrix(ret_row_dim, ret_col_dim);
	if(ret == NULL)
	{
		fprintf(stderr, "Warning: padding matrix cannot be allocated!(padding_dsmatrix_strassen)\n");
		return NULL;
	}

	// ret = [ A | 0 ]
	//       [---+---]
	//       [ 0 | I ]
	for(i = 0; i < orig_mat->row_dim; i++)
	{
		for(j = 0; j < orig_mat->col_dim; j++)
			set_dsmatrix_ij(ret, i, j, get_dsmatrix_ij(orig_mat, i, j));
	}

	min_dim = (ret_row_dim < ret_col_dim) ? ret_row_dim : ret_col_dim;
	for(i = orig_mat->row_dim; i < min_dim; i++)
		set_dsmatrix_ij_ui(ret, i, i, 1UL);

	return ret;
}

// Padding to even dimensional matrix
DSMatrix init_dynamic_padding_dsmatrix_strassen(DSMatrix orig_mat)
{
	DSMatrix ret = NULL;
	long int ret_row_dim, ret_col_dim, min_dim, i, j;

	if(orig_mat == NULL)
	{
		fprintf(stderr, "Warning: orig_mat is empty!(padding_dsmatrix_strassen)\n");
		return NULL;
	}

	ret_row_dim = orig_mat->row_dim;
	ret_col_dim = orig_mat->col_dim;

	if((ret_row_dim % 2) == 1)
		ret_row_dim++;
	if((ret_col_dim % 2) == 1)
		ret_col_dim++;

//	printf("Padding: row_dim %ld -> %ld, col_dim %ld -> %ld\n", orig_mat->row_dim, ret_row_dim, orig_mat->col_dim, ret_col_dim);

	ret = init_dsmatrix(ret_row_dim, ret_col_dim);
	if(ret == NULL)
	{
		fprintf(stderr, "Warning: padding matrix cannot be allocated!(padding_dsmatrix_strassen)\n");
		return NULL;
	}

	// ret = [ A | 0 ]
	//       [---+---]
	//       [ 0 | I ]
	for(i = 0; i < orig_mat->row_dim; i++)
	{
		for(j = 0; j < orig_mat->col_dim; j++)
			set_dsmatrix_ij(ret, i, j, get_dsmatrix_ij(orig_mat, i, j));
	}

/*	min_dim = (ret_row_dim < ret_col_dim) ? ret_row_dim : ret_col_dim;
	for(i = orig_mat->row_dim; i < min_dim; i++)
		set_dsmatrix_ij_ui(ret, i, i, 1UL);
*/
	return ret;
}

// Padding to even dimensional matrix
DSMatrix init_dynamic_padding_dsmatrix_strassen2(DSMatrix orig_mat, long int min_dim)
{
	DSMatrix ret = NULL;
	long int ret_row_dim, ret_col_dim, i, j;

	if(orig_mat == NULL)
	{
		fprintf(stderr, "Warning: orig_mat is empty!(padding_dsmatrix_strassen)\n");
		return NULL;
	}

	ret_row_dim = orig_mat->row_dim;
	ret_col_dim = orig_mat->col_dim;

	if((ret_row_dim % min_dim) >= 1)
		ret_row_dim = ((ret_row_dim / min_dim) + 1) * min_dim;
	if((ret_col_dim % min_dim) >= 1)
		ret_col_dim = ((ret_col_dim / min_dim) + 1) * min_dim;

//	printf("Padding: row_dim %ld -> %ld, col_dim %ld -> %ld\n", orig_mat->row_dim, ret_row_dim, orig_mat->col_dim, ret_col_dim);

	ret = init_dsmatrix(ret_row_dim, ret_col_dim);
	if(ret == NULL)
	{
		fprintf(stderr, "Warning: padding matrix cannot be allocated!(padding_dsmatrix_strassen)\n");
		return NULL;
	}

	// ret = [ A | 0 ]
	//       [---+---]
	//       [ 0 | I ]
	for(i = 0; i < orig_mat->row_dim; i++)
	{
		for(j = 0; j < orig_mat->col_dim; j++)
			set_dsmatrix_ij(ret, i, j, get_dsmatrix_ij(orig_mat, i, j));
	}
/*
	min_dim = (ret_row_dim < ret_col_dim) ? ret_row_dim : ret_col_dim;
	for(i = orig_mat->row_dim; i < min_dim; i++)
		set_dsmatrix_ij_ui(ret, i, i, 1UL);
*/
	return ret;
}

// Strassen's Algorithm with static padding
void mul_dsmatrix_strassen_odd_padding(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim)
{
	long int tmp_ret_index[4], ret_index[4];
	DSMatrix tmp_ret, tmp_mat_a, tmp_mat_b;

	// padding
#ifdef USE_STATIC_PADDING
	tmp_ret = init_static_padding_dsmatrix_strassen(ret);
	tmp_mat_a = init_static_padding_dsmatrix_strassen(mat_a);
	tmp_mat_b = init_static_padding_dsmatrix_strassen(mat_b);
#else
//	tmp_ret = init_dynamic_padding_dsmatrix_strassen(ret);
//	tmp_mat_a = init_dynamic_padding_dsmatrix_strassen(mat_a);
//	tmp_mat_b = init_dynamic_padding_dsmatrix_strassen(mat_b);
	tmp_ret = init_dynamic_padding_dsmatrix_strassen2(ret, min_dim);
	tmp_mat_a = init_dynamic_padding_dsmatrix_strassen2(mat_a, min_dim);
	tmp_mat_b = init_dynamic_padding_dsmatrix_strassen2(mat_b, min_dim);
#endif

	// strassen
#ifdef USE_WINOGRAD
	mul_dsmatrix_winograd_even(tmp_ret, tmp_mat_a, tmp_mat_b, min_dim);
#else
	mul_dsmatrix_strassen_even(tmp_ret, tmp_mat_a, tmp_mat_b, min_dim);
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

	subst_dsmatrix_partial(ret, ret_index, tmp_ret, tmp_ret_index);

	// free
	free_dsmatrix(tmp_ret);
	free_dsmatrix(tmp_mat_a);
	free_dsmatrix(tmp_mat_b);
}

// clear counter
void reset_num_mul_dsmatrix_strassen(void)
{
	num_addsub_mul_dsmatrix_strassen = 0;
	num_mul_mul_dsmatrix_strassen = 0;
}

// get counters
void get_num_mul_dsmatrix_strassen(long int *num_addsub, long int *num_mul)
{
	//printf("num_addsub_mul_dsmatrix_strassen: %ld\n", num_addsub_mul_dsmatrix_strassen);
	//printf("num_mul_mul_dsmatrix_strassen   : %ld\n", num_mul_mul_dsmatrix_strassen);

	if(num_addsub != NULL)
		*num_addsub = num_addsub_mul_dsmatrix_strassen;
	if(num_mul != NULL)
		*num_mul = num_mul_mul_dsmatrix_strassen;
}

// print counters
void print_num_mul_dsmatrix_strassen(long int *num_addsub, long int *num_mul)
{
	printf("num_addsub_mul_dsmatrix_strassen: %ld\n", num_addsub_mul_dsmatrix_strassen);
	printf("num_mul_mul_dsmatrix_strassen   : %ld\n", num_mul_mul_dsmatrix_strassen);

	get_num_mul_dsmatrix_strassen(num_addsub, num_mul);

}

// Fit dimension to be multiple of min_dim
void mul_dsmatrix_strassen(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim)
{
	long int row_k, col_k, mid_row_k, mid_col_k;
	long int row_dim, col_dim, mid_dim;
//	DSVector diag_left, diag_right;

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_dsmatrix_strassen)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	// normal matrix multiplication in case of ret_dim <= 4 
	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	{
		//mul_dsmatrix(ret, mat_a, mat_b);

		// counting the number of arithmetic
		num_addsub_mul_dsmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		num_mul_mul_dsmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		mul_dsmatrix_simple(ret, mat_a, mat_b);
		return;
	}

	// scaling 
//	diag_left = init2_dsvector(row_dim, mat_a->prec);
//	diag_right = init2_dsvector(col_dim, mat_b->prec);

//	left_scaling_dsmatrix(mat_a, diag_left, NULL);
//	right_scaling_dsmatrix(mat_b, diag_right, NULL);

	// dynamic peeling in case of odd dim
	// [ A11   a12 ] [ B11   b12 ] = [ A11*B11 + a12 * b21^T   A11*b12 + a12 * b22    ]
	// [ a21^T a22 ] [ b21^T b22 ]   [ a21^T*B11 + a22 * b21^T a21^T * b12 + a22 * b22]
	if((ret->row_dim % 2 == 1) || (ret->col_dim % 2 == 1) || (mid_dim % 2 == 1))
	{
#ifdef PEELING_ONLY
		mul_dsmatrix_strassen_odd_peeling(ret, mat_a, mat_b, min_dim);
#elif PADDING_ONLY
		mul_dsmatrix_strassen_odd_padding(ret, mat_a, mat_b, min_dim);
#else
//		row_k = (long int)floor(mylog2((double)ret->row_dim));
//		col_k = (long int)floor(mylog2((double)ret->col_dim));
		row_k = (long int)mylog2((double)ret->row_dim);
		col_k = (long int)mylog2((double)ret->col_dim);

		mid_row_k = (long int)pow(2.0, (double)(row_k - 1)) * 3;
		mid_col_k = (long int)pow(2.0, (double)(col_k - 1)) * 3;

		//printf("2^%ld <= %ld <= 2^%ld\n", row_k, ret->row_dim, row_k + 1);

		// dynamic peeling
		//if(ret->row_dim < mid_row_k)
		if((ret->row_dim % min_dim) < (min_dim / 2))
		{
			//printf("peeling\n");
			mul_dsmatrix_strassen_odd_peeling(ret, mat_a, mat_b, min_dim);
		}
		// padding
		else
		{
			//printf("padding\n");
			mul_dsmatrix_strassen_odd_padding(ret, mat_a, mat_b, min_dim);
		}
#endif

		//printf("end\n");
	}
	// normal strassen algorithm in case of even dim
	else
	{
		//printf("%d is even -> ", ret->row_dim);
#ifdef USE_WINOGRAD
		mul_dsmatrix_winograd_even(ret, mat_a, mat_b, min_dim);
#else
		mul_dsmatrix_strassen_even(ret, mat_a, mat_b, min_dim);
#endif
		//printf("end\n");
	}

//	mul_dsmatrix_dsiag_mat(ret, diag_left, 0, ret, diag_right, 0);

//	free_dsvector(diag_left);
//	free_dsvector(diag_right);

}

// Strassen's Algorithm with Dynamic peeling
void mul_dsmatrix_strassen_odd_peeling(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim)
{
	long int i, j, row_dim, row_dim_h, col_dim, col_dim_h, mid_dim, mid_dim_h, tmp_dim_h, tmp_dim;
	DSMatrix mat_a11, mat_b11, mat_c11, mat_tmp;
	DSVector vec_a12, vec_a21, vec_b12, vec_b21, vec_c12, vec_c21, vec_tmp12, vec_tmp21;
#ifdef __cplusplus
	static float a22[DSSIZE], b22[DSSIZE], c22[DSSIZE], tmp[DSSIZE];
#else // __cplusplus
	static float a22[DSSIZE], b22[DSSIZE], c22[DSSIZE], tmp[DSSIZE];
#endif // __cplusplus

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_dsmatrix_strassen_odd_peeling)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	set0_ds(a22);
	set0_ds(b22);
	set0_ds(c22);
	set0_ds(tmp);

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
		num_addsub_mul_dsmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		num_mul_mul_dsmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		//mul_dsmatrix(ret, mat_a, mat_b);
		mul_dsmatrix_simple(ret, mat_a, mat_b);
		return;
	}

	// tmp_dim_h = mid_dim_h or col_dim_h
	tmp_dim_h = mat_b->col_dim - 1;
	if(mid_dim_h < tmp_dim_h)
		tmp_dim_h = mid_dim_h;

	// Initialize
	mat_a11 = init_dsmatrix(row_dim_h, mid_dim_h);
	mat_b11 = init_dsmatrix(mid_dim_h, col_dim_h);
	mat_c11 = init_dsmatrix(row_dim_h, col_dim_h);
	mat_tmp = init_dsmatrix(row_dim_h, col_dim_h);

	vec_a12 = NULL;
	vec_b21 = NULL;
	vec_b12 = NULL;
	vec_a21 = NULL;
	vec_c12 = init_dsvector(row_dim_h);
	vec_c21 = init_dsvector(col_dim_h);
	vec_tmp12 = init_dsvector(row_dim_h);
	vec_tmp21 = init_dsvector(col_dim_h);

	// set matrix elements to mat_a11
	for(i = 0; i < row_dim_h; i++)
	{
		for(j = 0; j < mid_dim_h; j++)
			set_dsmatrix_ij(mat_a11, i, j, get_dsmatrix_ij(mat_a, i, j));
	}

	// set matrix elements to vec_b11
	for(i = 0; i < mid_dim_h; i++)
	{
		for(j = 0; j < col_dim_h; j++)
			set_dsmatrix_ij(mat_b11, i, j, get_dsmatrix_ij(mat_b, i, j));
	}

	// set matrix elements to vec_a12 and vec_b21
	if(mid_dim_h < mid_dim)
	{
		//printf("set vec_a12, b21\n");
		vec_a12 = init_dsvector(row_dim_h);
		vec_b21 = init_dsvector(col_dim_h); // fix!: 2014-03-19 by T.Kouya
		for(i = 0; i < row_dim_h; i++)
			set_dsvector_i(vec_a12, i, get_dsmatrix_ij(mat_a, i, mat_a->col_dim - 1));

		//printf("set vec_a12\n");

		for(i = 0; i < col_dim_h; i++)
			set_dsvector_i(vec_b21, i, get_dsmatrix_ij(mat_b, mat_b->row_dim - 1, i));

		//printf("set vec_b21\n");
	}

	// set matrix elements to vec_a21
	if(row_dim_h < row_dim)
	{
		//printf("set vec_a21, a22\n");
		vec_a21 = init_dsvector(mid_dim_h);
		for(i = 0; i < mid_dim_h; i++)
			set_dsvector_i(vec_a21, i, get_dsmatrix_ij(mat_a, mat_a->row_dim - 1, i));

		set0_ds(a22);
		if(mid_dim_h < mid_dim)
			rds_set(a22, get_dsmatrix_ij(mat_a, mat_a->row_dim - 1, mat_a->col_dim - 1));

	}

	// set matrix elements to vec_b12
	if(col_dim_h < col_dim)
	{
		//printf("set vec_a12, b22\n");
		vec_b12 = init_dsvector(mid_dim_h);
		for(i = 0; i < mid_dim_h; i++)
			set_dsvector_i(vec_b12, i, get_dsmatrix_ij(mat_b, i, mat_b->col_dim - 1));

		set0_ds(b22);
		if(mid_dim_h < mid_dim)
			rds_set(b22, get_dsmatrix_ij(mat_b, mat_b->row_dim - 1, mat_b->col_dim - 1));
	}

	// dynamic peeling in case of odd dim
	// [ A11   a12 ] [ B11   b12 ] = [ A11*B11 + a12 * b21^T   A11*b12 + a12 * b22     ]
	// [ a21^T a22 ] [ b21^T b22 ]   [ a21^T*B11 + a22 * b21^T a21^T * b12 + a22 * b22 ]

	//printf("starting C11 = A11 * B11...\n");

	// C11 = A11 * B11
#ifdef USE_WINOGRAD
	mul_dsmatrix_winograd_even(mat_c11, mat_a11, mat_b11, min_dim);
#else
	mul_dsmatrix_strassen_even(mat_c11, mat_a11, mat_b11, min_dim);
#endif

	//printf("C11 = A11 * B11\n");

	// C11 += a12 * b21^T
	if((vec_a12 != NULL) && (vec_b21 != NULL))
	{
		//printf("starting C11 += a12 * b21...\n");

		// counting the number of arithmetic
		num_addsub_mul_dsmatrix_strassen += row_dim_h * col_dim_h;
		num_mul_mul_dsmatrix_strassen += row_dim_h * col_dim_h;

		for(i = 0; i < row_dim_h; i++)
		{
			for(j = 0; j < col_dim_h; j++)
			{
				//rds_mul(get_dsmatrix_ij(mat_tmp, i, j), get_dsvector_i(vec_a12, i), get_dsvector_i(vec_b21, j));
				rds_mul(tmp, get_dsvector_i(vec_a12, i), get_dsvector_i(vec_b21, j));
				set_dsmatrix_ij(mat_tmp, i, j, tmp);
			}
		}
		add_dsmatrix(mat_c11, mat_c11, mat_tmp);
		//printf("C11 += a12 * b21...\n");
	}

	for(i = 0; i < row_dim_h; i++)
		for(j = 0; j < col_dim_h; j++) 
			set_dsmatrix_ij(ret, i, j, get_dsmatrix_ij(mat_c11, i, j));

	// c12 := A11 * b12
	if(vec_b12 != NULL)
	{
		// counting the number of arithmetic
		num_addsub_mul_dsmatrix_strassen += mat_a11->row_dim * mat_a11->col_dim;
		num_mul_mul_dsmatrix_strassen += mat_a11->row_dim * mat_a11->col_dim;

		mul_dsmatrix_dsvec(vec_c12, mat_a11, vec_b12);

		//printf("c12 = A11 * b12\n");
	}

 	// c12 += b22 * a12
 	if((vec_a12 != NULL) && (rds_cmp_ui(b22, 0UL) != 0))
	{
		// counting the number of arithmetic
		num_addsub_mul_dsmatrix_strassen += vec_c12->dim;
		num_mul_mul_dsmatrix_strassen += vec_tmp12->dim;

		cmul_dsvector(vec_tmp12, b22, vec_a12);
		add_dsvector(vec_c12, vec_c12, vec_tmp12);
	}
	//printf("c12 += b22 * a12\n");

	// Fix! 2016-08-18 by T.Kouya
	if((vec_b12 != NULL) || ((vec_a12 != NULL) && (rds_cmp_ui(b22, 0UL) != 0)))
	{
		for(i = 0; i < row_dim_h; i++) // Fix! 2016-08-18 by T.Kouya
			set_dsmatrix_ij(ret, i, ret->col_dim - 1, get_dsvector_i(vec_c12, i));
	}

	//printf("vec_c12\n");

	// c21 := a21^T*B11
	if(vec_a21 != NULL)
	{
		// counting the number of arithmetic
		num_addsub_mul_dsmatrix_strassen += mat_b11->row_dim * mat_b11->col_dim;
		num_mul_mul_dsmatrix_strassen += mat_b11->row_dim * mat_b11->col_dim;

		mul_dsmatrixt_dsvec(vec_c21, mat_b11, vec_a21);
	}
	//printf("c21 = a21^T * B11\n");

	// c21 += a22 * b21^T
	if((vec_b21 != NULL) && (rds_cmp_ui(a22, 0UL) != 0))
	{
		// counting the number of arithmetic
		num_addsub_mul_dsmatrix_strassen += vec_c21->dim;
		num_mul_mul_dsmatrix_strassen += vec_tmp21->dim;

		cmul_dsvector(vec_tmp21, a22, vec_b21);
		add_dsvector(vec_c21, vec_c21, vec_tmp21);
	}
	//printf("c21 += a22 * b21^T\n");
	if(vec_a21 != NULL)
	{
		for(i = 0; i < col_dim_h; i++)
			set_dsmatrix_ij(ret, ret->row_dim - 1, i, get_dsvector_i(vec_c21, i));
	}
	//printf("c21\n");

	// c22 := a21^T * b12
	if((vec_a21 != NULL) && (vec_b12 != NULL))
	{
		// counting the number of arithmetic
		num_addsub_mul_dsmatrix_strassen += vec_a21->dim;
		num_mul_mul_dsmatrix_strassen += vec_tmp21->dim;

#ifdef __cplusplus
		ip_dsvector(c22, vec_a21, vec_b12);
#else
		ip_dsvector(c22, vec_a21, vec_b12);
#endif // __cplusplus
	}
	//printf("c22 += a21^T * b12\n");

	// c22 += a22 * b22
	if((rds_cmp_ui(a22, 0UL) != 0) || (rds_cmp_ui(b22, 0UL) != 0))
	{
		// counting the number of arithmetic
		num_addsub_mul_dsmatrix_strassen += 1;
		num_mul_mul_dsmatrix_strassen += 1;

		rds_mul(tmp, a22, b22);
		rds_add(c22, c22, tmp);
	}
	//printf("c22 += a22 * b22\n");

	if((vec_a21 != NULL) && (vec_b12 != NULL))
		set_dsmatrix_ij(ret, ret->row_dim - 1, ret->col_dim - 1, c22);

	//printf("c22\n");

	// free
	free_dsmatrix(mat_a11);
	free_dsmatrix(mat_b11);
	free_dsmatrix(mat_c11);
	//printf("free_dsmatrix a11, b11, c11\n");
	free_dsmatrix(mat_tmp);

	//printf("free_dsmatrix\n");

	if(vec_a12 != NULL)
		free_dsvector(vec_a12);

	if(vec_b21 != NULL)
		free_dsvector(vec_b21);

	if(vec_a21 != NULL)
		free_dsvector(vec_a21);

	if(vec_b12 != NULL)
		free_dsvector(vec_b12);

	//printf("free_dsvector\n");

	free_dsvector(vec_c12);
	free_dsvector(vec_c21);
	free_dsvector(vec_tmp12);
	free_dsvector(vec_tmp21);

	//printf("free_dsvector2\n");
}

// Strassen's Algorithm
void mul_dsmatrix_strassen_even(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim)
{
//	long int min_dim = 4; // = 2^2
	DSMatrix mat_p[7], mat_tmp_a[7], mat_tmp_b[7], mat_tmp_c[4];
	long int row_dim_h, row_dim, col_dim_h, col_dim, mid_dim_h, mid_dim;
	long int ret_index[4], mat_tmp_a_index[4], mat_tmp_b_index[4], mat_ar_index[7][4], mat_al_index[7][4], mat_br_index[7][4], mat_bl_index[7][4], mat_c_index[4][4];
	long int i;

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_dsmatrix_strassen_even)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	row_dim_h = row_dim / 2;
	col_dim_h = col_dim / 2;
	mid_dim_h = mid_dim / 2;

/*	printf("mat_a: %ld, %ld\n", mat_a->row_dim, mat_a->col_dim);
	printf("mat_b: %ld, %ld\n", mat_b->row_dim, mat_b->col_dim);
	printf("row_dim_h, col_dim_h, mid_dim_h: %ld, %ld, %ld\n", row_dim_h, col_dim_h, mid_dim_h);
*/
	// normal matrix multiplication in case of ret_dim <= 4 
//	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	if((ret->row_dim <= min_dim) || (ret->col_dim <= min_dim) || (mid_dim <= min_dim))
	{
		// counting the number of arithmetic
		num_addsub_mul_dsmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		num_mul_mul_dsmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		//mul_dsmatrix(ret, mat_a, mat_b);
		mul_dsmatrix_simple(ret, mat_a, mat_b);
		return;
	}

	for(i = 0; i < 7; i++)
	{
		mat_p[i] = init_dsmatrix(row_dim_h, col_dim_h);
		mat_tmp_a[i] = init_dsmatrix(row_dim_h, mid_dim_h);
		mat_tmp_b[i] = init_dsmatrix(mid_dim_h, col_dim_h);
	}
	for(i = 0; i < 4; i++)
		mat_tmp_c[i] = init_dsmatrix(row_dim_h, col_dim_h);

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
	num_addsub_mul_dsmatrix_strassen += row_dim_h * mid_dim_h;

//	add_dsmatrix_partial(mat_tmp_a[0], ret_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);
	add_dsmatrix_partial(mat_tmp_a[0], mat_tmp_a_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);

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
	num_addsub_mul_dsmatrix_strassen += mid_dim_h * col_dim_h;

//	add_dsmatrix_partial(mat_tmp_b[0], ret_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);
	add_dsmatrix_partial(mat_tmp_b[0], mat_tmp_b_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);

	// P1 = tmp_a * tmp_b
//	printf("P1:\n");
	mul_dsmatrix_strassen(mat_p[0], mat_tmp_a[0], mat_tmp_b[0], min_dim);

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
	num_addsub_mul_dsmatrix_strassen += row_dim_h * mid_dim_h;

//	add_dsmatrix_partial(mat_tmp_a[1], ret_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);
	add_dsmatrix_partial(mat_tmp_a[1], mat_tmp_a_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);

	// B11
	mat_br_index[1][0] = 0;
	mat_br_index[1][1] = mid_dim_h;
	mat_br_index[1][2] = 0;
	mat_br_index[1][3] = col_dim_h;
//	subst_dsmatrix_partial(mat_tmp_b[1], ret_index, mat_b, mat_br_index[1]);
	subst_dsmatrix_partial(mat_tmp_b[1], mat_tmp_b_index, mat_b, mat_br_index[1]);

	// P2 = tmp_a * tmp_b
	//printf("P2:\n");
	mul_dsmatrix_strassen(mat_p[1], mat_tmp_a[1], mat_tmp_b[1], min_dim);

	// -------------------------------
	// P3 := A11 * (B12 - B22)
	// -------------------------------

	// A11
	mat_ar_index[2][0] = 0;
	mat_ar_index[2][1] = row_dim_h;
	mat_ar_index[2][2] = 0;
	mat_ar_index[2][3] = mid_dim_h;
//	subst_dsmatrix_partial(mat_tmp_a[2], ret_index, mat_a, mat_ar_index[2]);
	subst_dsmatrix_partial(mat_tmp_a[2], mat_tmp_a_index, mat_a, mat_ar_index[2]);

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
	num_addsub_mul_dsmatrix_strassen += mid_dim_h * col_dim_h;

//	sub_dsmatrix_partial(mat_tmp_b[2], ret_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);
	sub_dsmatrix_partial(mat_tmp_b[2], mat_tmp_b_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);

	// P3 = tmp_a * tmp_b
	//printf("P3:\n");
	mul_dsmatrix_strassen(mat_p[2], mat_tmp_a[2], mat_tmp_b[2], min_dim);

	// -------------------------------
	// P4 := A22 * (B21 - B11)
	// -------------------------------

	// A22
	mat_ar_index[3][0] = row_dim_h;
	mat_ar_index[3][1] = row_dim;
	mat_ar_index[3][2] = mid_dim_h;
	mat_ar_index[3][3] = mid_dim;

	// counting the number of arithmetic
	num_addsub_mul_dsmatrix_strassen += mid_dim_h * col_dim_h;

//	subst_dsmatrix_partial(mat_tmp_a[3], ret_index, mat_a, mat_ar_index[3]);
	subst_dsmatrix_partial(mat_tmp_a[3], mat_tmp_a_index, mat_a, mat_ar_index[3]);

	// B21 - B11
	mat_br_index[3][0] = mid_dim_h;
	mat_br_index[3][1] = mid_dim;
	mat_br_index[3][2] = 0;
	mat_br_index[3][3] = col_dim_h;

	mat_bl_index[3][0] = 0;
	mat_bl_index[3][1] = mid_dim_h;
	mat_bl_index[3][2] = 0;
	mat_bl_index[3][3] = col_dim_h;
//	sub_dsmatrix_partial(mat_tmp_b[3], ret_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);
	sub_dsmatrix_partial(mat_tmp_b[3], mat_tmp_b_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);

	// P4 = tmp_a * tmp_b
	//printf("P4:\n");
	mul_dsmatrix_strassen(mat_p[3], mat_tmp_a[3], mat_tmp_b[3], min_dim);

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
	num_addsub_mul_dsmatrix_strassen += row_dim_h * mid_dim_h;

//	add_dsmatrix_partial(mat_tmp_a[4], ret_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);
	add_dsmatrix_partial(mat_tmp_a[4], mat_tmp_a_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);

	// B22
	mat_br_index[4][0] = mid_dim_h;
	mat_br_index[4][1] = mid_dim;
	mat_br_index[4][2] = col_dim_h;
	mat_br_index[4][3] = col_dim;
//	subst_dsmatrix_partial(mat_tmp_b[4], ret_index, mat_b, mat_br_index[4]);
	subst_dsmatrix_partial(mat_tmp_b[4], mat_tmp_b_index, mat_b, mat_br_index[4]);

	// P5 = tmp_a * tmp_b
	//printf("P5:\n");
	mul_dsmatrix_strassen(mat_p[4], mat_tmp_a[4], mat_tmp_b[4], min_dim);

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
	num_addsub_mul_dsmatrix_strassen += row_dim_h * mid_dim_h;

//	sub_dsmatrix_partial(mat_tmp_a[5], ret_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);
	sub_dsmatrix_partial(mat_tmp_a[5], mat_tmp_a_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);

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
	num_addsub_mul_dsmatrix_strassen += mid_dim_h * col_dim_h;

//	add_dsmatrix_partial(mat_tmp_b[5], ret_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);
	add_dsmatrix_partial(mat_tmp_b[5], mat_tmp_b_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);

	// P6 = tmp_a * tmp_b
	//printf("P6:\n");
	mul_dsmatrix_strassen(mat_p[5], mat_tmp_a[5], mat_tmp_b[5], min_dim);

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
	num_addsub_mul_dsmatrix_strassen += row_dim_h * mid_dim_h;

//	sub_dsmatrix_partial(mat_tmp_a[6], ret_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);
	sub_dsmatrix_partial(mat_tmp_a[6], mat_tmp_a_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);

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
	num_addsub_mul_dsmatrix_strassen += mid_dim_h * col_dim_h;

//	add_dsmatrix_partial(mat_tmp_b[6], ret_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);
	add_dsmatrix_partial(mat_tmp_b[6], mat_tmp_b_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);

	// P7 = tmp_a * tmp_b
	//printf("P7:\n");
	mul_dsmatrix_strassen(mat_p[6], mat_tmp_a[6], mat_tmp_b[6], min_dim);

	// -------------------------------
	// C11 := P1 + P4 - P5 + P7
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dsmatrix_strassen += 3 * row_dim_h * col_dim_h;

	add_dsmatrix(mat_tmp_c[0], mat_p[0], mat_p[3]);
	sub_dsmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[4]);
	add_dsmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[6]);
	mat_c_index[0][0] = 0;
	mat_c_index[0][1] = row_dim_h;
	mat_c_index[0][2] = 0;
	mat_c_index[0][3] = col_dim_h;

	subst_dsmatrix_partial(ret, mat_c_index[0], mat_tmp_c[0], ret_index);

	// -------------------------------
	// C12 := P3 + P5
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dsmatrix_strassen += row_dim_h * col_dim_h;

	add_dsmatrix(mat_tmp_c[1], mat_p[2], mat_p[4]);
	mat_c_index[1][0] = 0;
	mat_c_index[1][1] = row_dim_h;
	mat_c_index[1][2] = col_dim_h;
	mat_c_index[1][3] = col_dim;
	subst_dsmatrix_partial(ret, mat_c_index[1], mat_tmp_c[1], ret_index);

	// -------------------------------
	// C21 := P2 + P4
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dsmatrix_strassen += row_dim_h * col_dim_h;

	add_dsmatrix(mat_tmp_c[2], mat_p[1], mat_p[3]);
	mat_c_index[2][0] = row_dim_h;
	mat_c_index[2][1] = row_dim;
	mat_c_index[2][2] = 0;
	mat_c_index[2][3] = col_dim_h;
	subst_dsmatrix_partial(ret, mat_c_index[2], mat_tmp_c[2], ret_index);

	// -------------------------------
	// C22 := P1 + P3 - P2 + P6
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dsmatrix_strassen += 3 * row_dim_h * col_dim_h;

	add_dsmatrix(mat_tmp_c[3], mat_p[0], mat_p[2]);
	sub_dsmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[1]);
	add_dsmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[5]);
	mat_c_index[3][0] = row_dim_h;
	mat_c_index[3][1] = row_dim;
	mat_c_index[3][2] = col_dim_h;
	mat_c_index[3][3] = col_dim;
	subst_dsmatrix_partial(ret, mat_c_index[3], mat_tmp_c[3], ret_index);

	// free
	for(i = 0; i < 7; i++)
	{
		free_dsmatrix(mat_p[i]);
		free_dsmatrix(mat_tmp_a[i]);
		free_dsmatrix(mat_tmp_b[i]);
	}
	for(i = 0; i < 4; i++)
		free_dsmatrix(mat_tmp_c[i]);
}

// Winograd Variant of Strassen's Algorithm
void mul_dsmatrix_winograd_even(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim)
{
//	long int min_dim = 4; // = 2^2
	DSMatrix mat_s[8], mat_m[7], mat_t[2], mat_tmp_a[4], mat_tmp_b[4], mat_tmp_c[4];
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
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_dsmatrix_winograd_even)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	row_dim_h = row_dim / 2;
	col_dim_h = col_dim / 2;
	mid_dim_h = mid_dim / 2;

	// normal matrix multiplication in case of ret_dim <= 4 
//	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	if((ret->row_dim <= min_dim) || (ret->col_dim <= min_dim) || (mid_dim <= min_dim))
	{
		mul_dsmatrix_simple(ret, mat_a, mat_b);

		// counting the number of arithmetic
		num_addsub_mul_dsmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		num_mul_mul_dsmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		return;
	}

	for(i = 0; i < 4; i++)
	{
		mat_s[i] = init_dsmatrix(row_dim_h, mid_dim_h);
		mat_tmp_a[i] = init_dsmatrix(row_dim_h, mid_dim_h);

		mat_s[i + 4] = init_dsmatrix(mid_dim_h, col_dim_h);
		mat_tmp_b[i] = init_dsmatrix(mid_dim_h, col_dim_h);

		mat_tmp_c[i] = init_dsmatrix(row_dim_h, col_dim_h);
	}
	for(i = 0; i < 7; i++)
		mat_m[i] = init_dsmatrix(row_dim_h, col_dim_h);

	mat_t[0] = init_dsmatrix(row_dim_h, col_dim_h);
	mat_t[1] = init_dsmatrix(row_dim_h, col_dim_h);

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
		subst_dsmatrix_partial(mat_tmp_a[i], mat_tmp_a_index, mat_a, mat_a_index[i]);
		subst_dsmatrix_partial(mat_tmp_b[i], mat_tmp_b_index, mat_b, mat_b_index[i]);
	}
//	printf("subst a, b...\n");

	// -------------------------------
	// S1 := A21 + A22
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dsmatrix_strassen += row_dim_h * mid_dim_h;

	add_dsmatrix(mat_s[0], mat_tmp_a[2], mat_tmp_a[3]);

	// -------------------------------
	// S2 := S1 - A11
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dsmatrix_strassen += row_dim_h * mid_dim_h;

	sub_dsmatrix(mat_s[1], mat_s[0], mat_tmp_a[0]);

	// -------------------------------
	// S3 := A11 - A21
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dsmatrix_strassen += row_dim_h * mid_dim_h;

	sub_dsmatrix(mat_s[2], mat_tmp_a[0], mat_tmp_a[2]);

	// -------------------------------
	// S4 := A12 - S2
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dsmatrix_strassen += row_dim_h * mid_dim_h;

	sub_dsmatrix(mat_s[3], mat_tmp_a[1], mat_s[1]);

	// -------------------------------
	// S5 := B12 - B11
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dsmatrix_strassen += mid_dim_h * col_dim_h;

	sub_dsmatrix(mat_s[4], mat_tmp_b[1], mat_tmp_b[0]);

	// -------------------------------
	// S6 := B22 - S5
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dsmatrix_strassen += mid_dim_h * col_dim_h;

	sub_dsmatrix(mat_s[5], mat_tmp_b[3], mat_s[4]);

	// -------------------------------
	// S7 := B22 - B12
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dsmatrix_strassen += mid_dim_h * col_dim_h;

	sub_dsmatrix(mat_s[6], mat_tmp_b[3], mat_tmp_b[1]);

	// -------------------------------
	// S8 := S6 - B21
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dsmatrix_strassen += mid_dim_h * col_dim_h;

	sub_dsmatrix(mat_s[7], mat_s[5], mat_tmp_b[2]);

//	printf("s...\n");

	// -------------------------------
	// M1 := S2 * S6
	// -------------------------------
	mul_dsmatrix_strassen(mat_m[0], mat_s[1], mat_s[5], min_dim);

	// -------------------------------
	// M2 := A11 * B11
	// -------------------------------
	mul_dsmatrix_strassen(mat_m[1], mat_tmp_a[0], mat_tmp_b[0], min_dim);

	// -------------------------------
	// M3 := A12 * B21
	// -------------------------------
	mul_dsmatrix_strassen(mat_m[2], mat_tmp_a[1], mat_tmp_b[2], min_dim);

	// -------------------------------
	// M4 := S3 * S7
	// -------------------------------
	mul_dsmatrix_strassen(mat_m[3], mat_s[2], mat_s[6], min_dim);

	// -------------------------------
	// M5 := S1 * S5
	// -------------------------------
	mul_dsmatrix_strassen(mat_m[4], mat_s[0], mat_s[4], min_dim);

	// -------------------------------
	// M6 := S4 * B22
	// -------------------------------
	mul_dsmatrix_strassen(mat_m[5], mat_s[3], mat_tmp_b[3], min_dim);

	// -------------------------------
	// M7 := A22 * S8
	// -------------------------------
	mul_dsmatrix_strassen(mat_m[6], mat_tmp_a[3], mat_s[7], min_dim);

//	printf("m...\n");

	// -------------------------------
	// T1 := M1 + M2
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dsmatrix_strassen += row_dim_h * col_dim_h;

	add_dsmatrix(mat_t[0], mat_m[0], mat_m[1]);

	// -------------------------------
	// T2 := T1 + M4
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dsmatrix_strassen += row_dim_h * col_dim_h;

	add_dsmatrix(mat_t[1], mat_t[0], mat_m[3]);

//	printf("t...\n");

	// -------------------------------
	// C11 := M2 + M3
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dsmatrix_strassen += row_dim_h * col_dim_h;

	add_dsmatrix(mat_tmp_c[0], mat_m[1], mat_m[2]);

	// -------------------------------
	// C12 := T1 + M5 + M6
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dsmatrix_strassen += 2 * row_dim_h * col_dim_h;

	add_dsmatrix(mat_tmp_c[1], mat_t[0], mat_m[4]);
	add_dsmatrix(mat_tmp_c[1], mat_tmp_c[1], mat_m[5]);

	// -------------------------------
	// C21 := T2 - M7
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dsmatrix_strassen += row_dim_h * col_dim_h;

	sub_dsmatrix(mat_tmp_c[2], mat_t[1], mat_m[6]);

	// -------------------------------
	// C22 := T2 + M5
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dsmatrix_strassen += row_dim_h * col_dim_h;

	add_dsmatrix(mat_tmp_c[3], mat_t[1], mat_m[4]);

//	printf("c...\n");

	// -------------------------------
	// RET := [C11 C12]
	//        [C21 C22]
	// -------------------------------
	for(i = 0; i < 4; i++)
		subst_dsmatrix_partial(ret, mat_c_index[i], mat_tmp_c[i], ret_index);

//	printf("set...\n");


	// free
	for(i = 0; i < 4; i++)
	{
		free_dsmatrix(mat_s[i]);
		free_dsmatrix(mat_tmp_a[i]);
		free_dsmatrix(mat_s[i + 4]);
		free_dsmatrix(mat_tmp_b[i]);
		free_dsmatrix(mat_tmp_c[i]);
	}
	for(i = 0; i < 7; i++)
		free_dsmatrix(mat_m[i]);
	
	free_dsmatrix(mat_t[0]);
	free_dsmatrix(mat_t[1]);
}

// Computattion of Inverse Matrix by using Strassen's Algorithm
void inv_dsmatrix_strassen_even(DSMatrix ret, DSMatrix mat_a, long int min_dim)
{
	DSMatrix mat_p[6], mat_tmp_a[4], mat_p3p6, mat_p6p2, mat_p3p6p2;
	long int dim_h, dim;
	long int ret_index[4], mat_a_index[4][4];
	long int i;

	// Square matrix only
	if((ret->row_dim != ret->col_dim) || (mat_a->row_dim != mat_a->col_dim) || (ret->col_dim != mat_a->row_dim))
	{
		fprintf(stderr, "ERROR: Impossible to get inverse matrix for rectangle matrix!(inv_dsmatrix_strassen_even)\n");
		return;
	}

	// initialize
	dim = ret->row_dim;
	dim_h = dim / 2;

	// normal matrix multiplication in case of ret_dim <= 4 
	//printf("P0: %ld -> %ld start \n", dim, dim_h);
	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim))
	{
		subst_dsmatrix(ret, mat_a);
		//print_dsmatrix(mat_a);
		inv_dsmatrix(ret);

		return;
	}
	//printf("P0: %ld -> %ld end\n", dim, dim_h);

	mat_p3p6 = init_dsmatrix(dim_h, dim_h);
	mat_p6p2 = init_dsmatrix(dim_h, dim_h);
	mat_p3p6p2 = init_dsmatrix(dim_h, dim_h);

	for(i = 0; i < 6; i++)
		mat_p[i] = init_dsmatrix(dim_h, dim_h);

	for(i = 0; i < 4; i++)
		mat_tmp_a[i] = init_dsmatrix(dim_h, dim_h);

	ret_index[0] = 0;
	ret_index[1] = dim_h;
	ret_index[2] = 0;
	ret_index[3] = dim_h;

	// -------------------------------
	// P1 := A11^(-1)
	//--------------------------------

	// A11
	mat_a_index[0][0] = 0;
	mat_a_index[0][1] = dim_h;
	mat_a_index[0][2] = 0;
	mat_a_index[0][3] = dim_h;

	subst_dsmatrix_partial(mat_tmp_a[0], ret_index, mat_a, mat_a_index[0]);

	// P1 = tmp_a * tmp_b
	//printf("P1: %ld -> %ld start \n", dim, dim_h);
	inv_dsmatrix_strassen_even(mat_p[0], mat_tmp_a[0], min_dim);
	//inv_dsmatrix_strassen(mat_p[0], mat_tmp_a[0], min_dim);
	//printf("P1: %ld -> %ld end \n", dim, dim_h);

	// -------------------------------
	// P2 := A21 * P1
	// -------------------------------

	// A21
	mat_a_index[2][0] = dim_h;
	mat_a_index[2][1] = dim;
	mat_a_index[2][2] = 0;
	mat_a_index[2][3] = dim_h;

	subst_dsmatrix_partial(mat_tmp_a[2], ret_index, mat_a, mat_a_index[2]);

	// P2 := A21 * P1
	mul_dsmatrix_strassen(mat_p[1], mat_tmp_a[2], mat_p[0], min_dim);

	// -------------------------------
	// P3 := P1 * A12
	// -------------------------------

	// A12
	mat_a_index[1][0] = 0;
	mat_a_index[1][1] = dim_h;
	mat_a_index[1][2] = dim_h;
	mat_a_index[1][3] = dim;

	subst_dsmatrix_partial(mat_tmp_a[1], ret_index, mat_a, mat_a_index[1]);

	// P3 = P1 * A12
	mul_dsmatrix_strassen(mat_p[2], mat_p[0], mat_tmp_a[1], min_dim);

	// -------------------------------
	// P4 := A21 * P3
	// -------------------------------
	mul_dsmatrix_strassen(mat_p[3], mat_tmp_a[2], mat_p[2], min_dim);

	// -------------------------------
	// P5 := P4 - A22
	// -------------------------------

	// A22
	mat_a_index[3][0] = dim_h;
	mat_a_index[3][1] = dim;
	mat_a_index[3][2] = dim_h;
	mat_a_index[3][3] = dim;
	subst_dsmatrix_partial(mat_tmp_a[3], ret_index, mat_a, mat_a_index[3]);

	// P5 = P4 - A22
	sub_dsmatrix(mat_p[4], mat_p[3], mat_tmp_a[3]);

	// -------------------------------
	// P6 := P5^(-1)
	// -------------------------------

	// P6 = tmp_a * tmp_b
	//printf("P6: %ld -> %ld start\n", dim, dim_h);
	inv_dsmatrix_strassen_even(mat_p[5], mat_p[4], min_dim);
	//inv_dsmatrix_strassen(mat_p[5], mat_p[5], min_dim);
	//printf("P6: %ld -> %ld end\n", dim, dim_h);

	// -------------------------------
	// P3P6 := P3 * P6
	// -------------------------------
	mul_dsmatrix_strassen(mat_p3p6, mat_p[2], mat_p[5], min_dim);

	// -------------------------------
	// P6P2 := P6 * P2
	// -------------------------------
	mul_dsmatrix_strassen(mat_p6p2, mat_p[5], mat_p[1], min_dim);

	// -------------------------------
	// P3P6P2 := P6 * P2
	// -------------------------------
	mul_dsmatrix_strassen(mat_p3p6p2, mat_p3p6, mat_p[1], min_dim);

	// -------------------------------
	// RET11 := P1 - P3 * P6 * P2
	// -------------------------------
	sub_dsmatrix_partial(ret, mat_a_index[0], mat_p[0], ret_index, mat_p3p6p2, ret_index);

	// -------------------------------
	// RET12 := P3 * P6
	// -------------------------------
	subst_dsmatrix_partial(ret, mat_a_index[1], mat_p3p6, ret_index);

	// -------------------------------
	// RET21 := P6 * P2
	// -------------------------------
	subst_dsmatrix_partial(ret, mat_a_index[2], mat_p6p2, ret_index);

	// -------------------------------
	// RET22 := -P6
	// -------------------------------
	neg_dsmatrix_partial(ret, mat_a_index[3], mat_p[5], ret_index);

	// free
	free_dsmatrix(mat_p3p6);
	free_dsmatrix(mat_p6p2);
	free_dsmatrix(mat_p3p6p2);

	for(i = 0; i < 6; i++)
		free_dsmatrix(mat_p[i]);

	for(i = 0; i < 4; i++)
		free_dsmatrix(mat_tmp_a[i]);
}

/* scaling with the absolute maximum element in the row */
/* ret := scaling_diag_mat * org_mat -> ||ret|| \approx 1 */
void left_scaling_dsmatrix(DSMatrix ret, DSMatrix org_mat, DSVector scaling_diag_mat, long int *ret_col_index)
{
	long int i, j, row_dim, col_dim, absmax_col_index;
#ifdef __cplusplus
	static float absmax_val[DSSIZE], tmp_abs[DSSIZE];
#else // __cplusplus
	static float absmax_val[DSSIZE], tmp_abs[DSSIZE];
#endif // __cplusplus

	row_dim = ret->row_dim;
	col_dim = ret->col_dim;

	// dimension check
	if(row_dim != org_mat->row_dim)
	{
		fprintf(stderr, "Warning: Dimension of row of ret (ret->row_dim = %ld) is not the same as the org_mat's (org_mat->row_dim = %ld)\n", row_dim, org_mat->row_dim);
		if(row_dim > org_mat->row_dim)
			row_dim = org_mat->row_dim;
	}

	if(scaling_diag_mat != NULL)
	{
		if(row_dim != scaling_diag_mat->dim)
		{
			fprintf(stderr, "Warning: Dimension of row of ret (ret->row_dim = %ld) is not the same as the scaling_diag_mat's (scaling_diag_mat->dim = %ld)\n", row_dim, scaling_diag_mat->dim);
			if(row_dim > scaling_diag_mat->dim)
				row_dim = scaling_diag_mat->dim;
		}
	}

	if(col_dim != org_mat->col_dim)
	{
		fprintf(stderr, "Warning: Dimension of column of ret (ret->col_dim = %ld) is not the same as the org_mat's (org_mat->col_dim = %ld)\n", col_dim, org_mat->col_dim);
		if(col_dim > org_mat->col_dim)
			col_dim = org_mat->col_dim;
	}

	// search & scaling
	for(i = 0; i < row_dim; i++)
	{
		// search absmax value
		absmax_col_index = 0;
		//absmax_val = fabs(get_dsmatrix_ij(org_mat, i, 0));
		rds_abs(absmax_val, get_dsmatrix_ij(org_mat, i, 0));
		for(j = 1; j < col_dim; j++)
		{
			//tmp_abs = fabs(get_dsmatrix_ij(org_mat, i, j));
			rds_abs(tmp_abs, get_dsmatrix_ij(org_mat, i, j));
			if(rds_cmp(absmax_val, tmp_abs) < 0)
			{
				absmax_col_index = j;
				//absmax_val = tmp_abs;
				rds_set(absmax_val, tmp_abs);
			}
		}
		
		if(scaling_diag_mat != NULL)
			set_dsvector_i(scaling_diag_mat, i, absmax_val);
		if(ret_col_index != NULL)
			ret_col_index[i] = absmax_col_index;
		
		// scaling
		if(rds_cmp_ui(absmax_val, 0UL) == 0)
		{
			for(j = 0; j < col_dim; j++)
				set_dsmatrix_ij(ret, i, j, get_dsmatrix_ij(org_mat, i, j));
		}
		else
		{
			for(j = 0; j < col_dim; j++)
			{
				//tmp_abs = get_dsmatrix_ij(org_mat, i, j) / absmax_val;
				rds_div(tmp_abs, get_dsmatrix_ij(org_mat, i, j), absmax_val);
				set_dsmatrix_ij(ret, i, j, tmp_abs);
			}
		}
	}

}

/* scaling with the absolute maximum element in the column */
/* ret := org_mat * scaling_diag_mat -> ||ret|| \approx 1 */
void right_scaling_dsmatrix(DSMatrix ret, DSMatrix org_mat, DSVector scaling_diag_mat, long int *ret_row_index)
{
	long int i, j, row_dim, col_dim, absmax_row_index;
#ifdef __cplusplus
	static float absmax_val[DSSIZE], tmp_abs[DSSIZE];
#else // __cplusplus
	static float absmax_val[DSSIZE], tmp_abs[DSSIZE];
#endif // __cplusplus

	row_dim = ret->row_dim;
	col_dim = ret->col_dim;

	// dimension check
	if(row_dim != org_mat->row_dim)
	{
		fprintf(stderr, "Warning: Dimension of row of ret (ret->row_dim = %ld) is not the same as the org_mat's (org_mat->row_dim = %ld)\n", row_dim, org_mat->row_dim);
		if(row_dim > org_mat->row_dim)
			row_dim = org_mat->row_dim;
	}

	if(scaling_diag_mat != NULL)
	{
		if(col_dim != scaling_diag_mat->dim)
		{
			fprintf(stderr, "Warning: Dimension of col of ret (ret->row_dim = %ld) is not the same as the scaling_diag_mat's (scaling_diag_mat->dim = %ld)\n", col_dim, scaling_diag_mat->dim);
			if(col_dim > scaling_diag_mat->dim)
				col_dim = scaling_diag_mat->dim;
		}
	}

	if(col_dim != org_mat->col_dim)
	{
		fprintf(stderr, "Warning: Dimension of column of ret (ret->col_dim = %ld) is not the same as the org_mat's (org_mat->col_dim = %ld)\n", col_dim, org_mat->col_dim);
		if(col_dim > org_mat->col_dim)
			col_dim = org_mat->col_dim;
	}

	// search & scaling
	for(i = 0; i < col_dim; i++)
	{
		// search absmax value
		absmax_row_index = 0;
		//absmax_val = fabs(get_dsmatrix_ij(org_mat, 0, i));
		rds_abs(absmax_val, get_dsmatrix_ij(org_mat, 0, i));
		for(j = 1; j < row_dim; j++)
		{
			//tmp_abs = fabs(get_dsmatrix_ij(org_mat, j, i));
			rds_abs(tmp_abs, get_dsmatrix_ij(org_mat, j, i));
			if(rds_cmp(absmax_val, tmp_abs) < 0)
			{
				absmax_row_index = j;
				rds_set(absmax_val, tmp_abs);
			}
		}

		if(scaling_diag_mat != NULL)
			set_dsvector_i(scaling_diag_mat, i, absmax_val);
		if(ret_row_index != NULL)
			ret_row_index[i] = absmax_row_index;
		
		// scaling
		if(rds_cmp_ui(absmax_val, 0UL) == 0)
		{
			for(j = 0; j < row_dim; j++)
				set_dsmatrix_ij(ret, j, i, get_dsmatrix_ij(org_mat, j, i));
		}
		else
		{
			for(j = 0; j < row_dim; j++)
			{
				//tmp_abs = get_dsmatrix_ij(org_mat, j, i) / absmax_val;
				rds_div(tmp_abs, get_dsmatrix_ij(org_mat, j, i), absmax_val);
				set_dsmatrix_ij(ret, j, i, tmp_abs);
			}
		}
	}

}

/* multiply square matrix by diagonal matrix      */
/* ret = left_diag_mat^(-left_inv_flat) * org_mat * right_diag_mat^(-right_inv_flat) (left_diag_mat != null, right_diag_mat != null) */
/* ret = left_diag_mat^(-left_inv_flat) * org_mat * right_diag_mat^(-right_inv_flat) (left_diag_mat != null, right_diag_mat != null) */
/* ret = left_diag_mat^(-left_inv_flat) * org_mat                                    (left_diag_mat != null, right_diag_mat == null) */
/* ret =                                  org_mat * right_diag_mat^(-right_inv_flat) (left_diag_mat == null, right_diag_mat != null) */
void mul_dsmatrix_dsdiag(DSMatrix ret, DSVector left_diag_mat, int left_inv_flag, DSMatrix org_mat, DSVector right_diag_mat, int right_inv_flag)
{
	long int i, j, row_dim, col_dim;
#ifdef __cplusplus
	static float tmp[DSSIZE];
#else // __cplusplus
	static float tmp[DSSIZE];
#endif // __cplusplus

	row_dim = ret->row_dim;
	col_dim = ret->col_dim;

	/* ret := left_diag_mat * org_mat */
	if(left_diag_mat != NULL)
	{
		// dimension check
		if(row_dim != org_mat->row_dim)
		{
			fprintf(stderr, "ERROR: Dimension of row of ret (ret->row_dim = %ld) is not the same as the org_mat's (org_mat->row_dim = %ld)\n", row_dim, org_mat->row_dim);

			return;
		}

		// dimension check
		if(row_dim != left_diag_mat->dim)
		{
			fprintf(stderr, "ERROR: Dimension of row of ret (ret->row_dim = %ld) is not the same as the left_diag_mat's (left_diag_mat->dim = %ld)\n", row_dim, left_diag_mat->dim);

			return;
		}

		// multiplication
		for(i = 0; i < row_dim; i++)
		{
			// left_diag_mat * org_mat
			if(left_inv_flag == 0)
			{
				for(j = 0; j < col_dim; j++)
				{
					//tmp = get_dsvector_i(left_diag_mat, i) * get_dsmatrix_ij(org_mat, i, j);
					rds_mul(tmp, get_dsvector_i(left_diag_mat, i), get_dsmatrix_ij(org_mat, i, j));
					set_dsmatrix_ij(ret, i, j, tmp);
				}
			}
			// left_diag_mat^(-1) * org_mat
			else
			{
				for(j = 0; j < col_dim; j++)
				{
					//tmp = get_dsmatrix_ij(org_mat, i, j) / get_dvector_i(left_diag_mat, i);
					rds_div(tmp, get_dsmatrix_ij(org_mat, i, j), get_dsvector_i(left_diag_mat, i));
					set_dsmatrix_ij(ret, i, j, tmp);
				}
			}
		}
	}

	/* ret := ret * right_diag_mt */
	if(right_diag_mat != NULL)
	{
		if(col_dim != org_mat->col_dim)
		{
			fprintf(stderr, "Error: Dimension of column of ret (ret->col_dim = %ld) is not the same as the org_mat's (org_mat->col_dim = %ld)\n", col_dim, org_mat->col_dim);

			return;
		}

		if(col_dim != right_diag_mat->dim)
		{
			fprintf(stderr, "Warning: Dimension of column of ret (ret->col_dim = %ld) is not the same as the right_diag_mat's (right_diag_mat->dim = %ld)\n", col_dim, right_diag_mat->dim);

			return;
		}

		// multiplication
		for(i = 0; i < row_dim; i++)
		{
			// org_mat * right_diag_mat
			if(right_inv_flag == 0)
			{
				for(j = 0; j < col_dim; j++)
				{
					//tmp = get_dsmatrix_ij(ret, i, j) * get_dvector_i(right_diag_mat, j);
					rds_mul(tmp, get_dsmatrix_ij(ret, i, j), get_dsvector_i(right_diag_mat, j));
					set_dsmatrix_ij(ret, i, j, tmp);
				}
			}
			// org_mat * right_diag_mat^(-1)
			else
			{
				for(j = 0; j < col_dim; j++)
				{
					//tmp = get_dsmatrix_ij(ret, i, j) / get_dvector_i(right_diag_mat, j);
					rds_div(tmp, get_dsmatrix_ij(ret, i, j), get_dsvector_i(right_diag_mat, j));
					set_dsmatrix_ij(ret, i, j, tmp);
				}
			}
		}
	}

}
#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus

// The following main function is for debugging
#ifdef DEBUG

#include "get_secv.h"

int main(int argc, char *argv[])
{
	long int i, j, dim;
	unsigned long prec = 128;
	float dstmp[3][DSSIZE];
	DSMatrix dsc, dsa, dsb, dsc_normal, dsc_block;
	DSVector dsdiag_left, dsdiag_right;
	double stime, etime[4], reldiff[4];

//	dim = 128;
	if(argc <= 1)
	{
		fprintf(stderr, "Usage: %s [dim]\n", argv[0]);
		return 0;
	}
	dim = atol(argv[1]);
	if(dim <= 0)
		return 0;

/* double-single precision */

	set0_ds(dstmp[0]);
	set0_ds(dstmp[1]);

	dsc = init_dsmatrix(dim, dim);
	dsc_normal = init_dsmatrix(dim, dim);
	dsc_block = init_dsmatrix(dim, dim);
	dsa = init_dsmatrix(dim, dim);
	dsb = init_dsmatrix(dim, dim);

	dsdiag_left = init_dsvector(dim);
	dsdiag_right = init_dsvector(dim);

	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			rds_set_d(dstmp[0], (double)rand());
			if(rand() % 2 != 0)
				rds_neg(dstmp[0], dstmp[0]);

			rds_set_d(dstmp[1], (double)rand());
			rds_ui_div(dstmp[1], 1UL, dstmp[1]);
			if(rand() % 2 != 0)
				rds_neg(dstmp[1], dstmp[1]);

			set_dsmatrix_ij(dsa, i, j, dstmp[0]);
			set_dsmatrix_ij(dsb, i, j, dstmp[1]);
		}
	}
//	printf("a(%ld x %ld) -> (%ld, %ld)\n", dsa->row_dim, dsa->col_dim, dsa->real_row_dim, dsa->real_col_dim);
//	printf("b(%ld x %ld) -> (%ld, %ld)\n", dsb->row_dim, dsb->col_dim, dsb->real_row_dim, dsb->real_col_dim);
//	printf("c(%ld x %ld) -> (%ld, %ld)\n", dsc->row_dim, dsc->col_dim, dsc->real_row_dim, dsc->real_col_dim);


	// normal matrix mul
//	printf("Normal matmul(%ld x %ld) ...", dsa->row_dim, dsa->col_dim);
	stime = get_secv();
	mul_dsmatrix(dsc_normal, dsa, dsb);
	etime[0] = get_secv() - stime;
//	printf("end\n");

//	left_scaling_dsmatrix(dsa, dsa, dsdiag_left, NULL);
//	right_scaling_dsmatrix(dsb, dsb, dsdiag_right, NULL);

	// Strassen 
	stime = get_secv();
//	mul_dsmatrix_strassen(dsc, dsa, dsb, 8);
//	mul_dsmatrix_strassen(dsc, dsa, dsb, 16);
	mul_dsmatrix_strassen(dsc, dsa, dsb, 32);
	etime[1] = get_secv() - stime;

//	mul_dsmatrix_dsdiag(dsc, dsdiag_left, 0, dsc, dsdiag_right, 0);

	// difference
	sub_dsmatrix(dsc, dsc_normal, dsc);

	// Block
	stime = get_secv();
	mul_dsmatrix_block(dsc_block, dsa, dsb, 32);
//	mul_dsmatrix_block(dsc_block, dsa, dsb, 1024);
	etime[2] = get_secv() - stime;

	// difference
	sub_dsmatrix(dsc_block, dsc_normal, dsc_block);

	// print
	printf("dim        : %ld\n", dim);
	printf("strassen   : %f\n", etime[1]);
	printf("block      : %f\n", etime[2]);
	printf("normal     : %f\n", etime[0]);
	normi_dsmatrix(dstmp[0], dsc);
	normi_dsmatrix(dstmp[1], dsc_block);
	normi_dsmatrix(dstmp[2], dsc_normal);
	rds_div(dstmp[0], dstmp[0], dstmp[2]);
	rds_div(dstmp[1], dstmp[1], dstmp[2]);
	printf("||reldiff_strassen||: "); rds_out_str(dstmp[0]); printf("\n");
	printf("||reldiff_block   ||: "); rds_out_str(dstmp[1]); printf("\n");

/* Inverse */

	frank_dsmatrix(dsa, dim);
	frank_dsmatrix(dsb, dim);
//	lotkin_dsmatrix(dsa, dim);
//	lotkin_dsmatrix(dsb, dim);

	// normal inverse
	printf("Normal Inverse...\n");
	stime = get_secv();
	inv_dsmatrix(dsa);
	etime[1] = get_secv() - stime;

	// Strassen 
	printf("Strassen Inverse...\n");
	stime = get_secv();
	inv_dsmatrix_strassen_even(dsc, dsb, 32);
	etime[0] = get_secv() - stime;

//	print_dmatrix(dc);

	// LU
	printf("LU decompsition...\n");
	stime = get_secv();
	DSLUdecomp(dsb);
	etime[2] = get_secv() - stime;

	// difference
	sub_dsmatrix(dsb, dsa, dsc);
	normi_dsmatrix(dstmp[0], dsb);
	normi_dsmatrix(dstmp[1], dsa);
	rds_div(dstmp[0], dstmp[0], dstmp[1]);

	// print
	printf("dim        : %ld\n", dim);
	printf("LU         : %f\n", etime[2]);
	printf("strassen   : %f\n", etime[0]);
	printf("normal     : %f\n", etime[1]);
	printf("||a^(-1)|| strass : ");
		normi_dsmatrix(dstmp[1], dsc);
		rds_out_str(dstmp[1]);
	printf("\n");
	printf("||a^(-1)||        : ");
		normi_dsmatrix(dstmp[1], dsa);
		rds_out_str(dstmp[1]);
	printf("\n");
	printf("||reldiff_strass||: ");
		rds_out_str(dstmp[0]);
	printf("\n");

	free_dsmatrix(dsc);
	free_dsmatrix(dsc_normal);
	free_dsmatrix(dsc_block);
	free_dsmatrix(dsa);
	free_dsmatrix(dsb);

	free_dsvector(dsdiag_left);
	free_dsvector(dsdiag_right);

	return 0;
}
#endif // DEBUG
