/********************************************************************************/
/* matmul_strassen_general_mpc_omp.c:                                           */
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
#include <stdio.h>
#include <math.h>

#include "bncomp.h"

#include "matmul_strassen.h"

// count the number of computations
long int _bncomp_num_addsub_mul_cmpfmatrix_strassen;	// addition and subtraction
long int _bncomp_num_mul_mul_cmpfmatrix_strassen;	// multiplication


// MPF & MPFR
#ifdef USE_GMP

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_add_cmpfmatrix_partial(CMPFMatrix ret, long int ret_index[4], CMPFMatrix mat_a, long int mat_a_index[4], CMPFMatrix mat_b, long int mat_b_index[4])
{
	unsigned long prec;
	long int i, j, ret_i, ret_j, a_i, a_j, b_i, b_j;
	long int imax, jmax;
	mpc_t tmp_val;

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	prec = ret->prec;
	mpc_init2(tmp_val, prec);
	//tmp_val = init2_mpc_t(prec);

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

			//tmp_val = get_cmpfmatrix_ij(mat_a, a_i, a_j) + get_cmpfmatrix_ij(mat_b, b_i, b_j);
			mpc_add(tmp_val, get_cmpfmatrix_ij(mat_a, a_i, a_j), get_cmpfmatrix_ij(mat_b, b_i, b_j), get_bnc_default_rounding_mode());
			set_cmpfmatrix_ij(ret, ret_i, ret_j, tmp_val);
		}
	}

	mpc_clear(tmp_val);
}

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void sub_cmpfmatrix_partial(CMPFMatrix ret, long int ret_index[4], CMPFMatrix mat_a, long int mat_a_index[4], CMPFMatrix mat_b, long int mat_b_index[4])
{
	unsigned long prec;
	long int i, j, ret_i, ret_j, a_i, a_j, b_i, b_j;
	long int imax, jmax;
	mpc_t tmp_val;

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	prec = ret->prec;
	mpc_init2(tmp_val, prec);
	//tmp_val = init2_mpc_t(prec);

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

			//tmp_val = get_cmpfmatrix_ij(mat_a, a_i, a_j) - get_cmpfmatrix_ij(mat_b, b_i, b_j);
			mpc_sub(tmp_val, get_cmpfmatrix_ij(mat_a, a_i, a_j), get_cmpfmatrix_ij(mat_b, b_i, b_j), get_bnc_default_rounding_mode());
			set_cmpfmatrix_ij(ret, ret_i, ret_j, tmp_val);
		}
	}

	mpc_clear(tmp_val);
}

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_cmpfmatrix_partial(CMPFMatrix ret, long int ret_index[4], CMPFMatrix mat_a, long int mat_a_index[4])
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

			set_cmpfmatrix_ij(ret, ret_i, ret_j, get_cmpfmatrix_ij(mat_a, a_i, a_j));
		}
	}
}

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_cmpfmatrix_partial_checked(CMPFMatrix ret, long int ret_index[4], CMPFMatrix mat_a, long int mat_a_index[4])
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
						set_cmpfmatrix_ij(ret, ret_i, ret_j, get_cmpfmatrix_ij(mat_a, a_i, a_j));
					else
						set_cmpfmatrix_ij_ui(ret, ret_i, ret_j, 0UL); // Padding
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
void neg_cmpfmatrix_partial(CMPFMatrix ret, long int ret_index[4], CMPFMatrix mat_a, long int mat_a_index[4])
{
	long int i, j, ret_i, ret_j, a_i, a_j;
	long int imax, jmax;
	mpc_t tmp;

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	mpc_init2(tmp, ret->prec);
	//tmp = init2_mpc_t(ret->prec);

	for(i = 0; i < imax; i++)
	{
		ret_i = ret_index[0] + i;
		a_i = mat_a_index[0] + i;
		for(j = 0; j < jmax; j++)
		{
			ret_j = ret_index[2] + j;
			a_j = mat_a_index[2] + j;
			mpc_neg(tmp, get_cmpfmatrix_ij(mat_a, a_i, a_j), get_bnc_default_rounding_mode());
			//neg_mpc_t(tmp, get_cmpfmatrix_ij(mat_a, a_i, a_j));
			set_cmpfmatrix_ij(ret, ret_i, ret_j, tmp);
		}
	}

	mpc_clear(tmp);
}

/* c = a * b */
void mul_cmpfmatrix_simple(CMPFMatrix c, CMPFMatrix a, CMPFMatrix b)
{
	long int i, j, k;
	mpc_t tmp, tmp2;

	/* dimension check */
	if(c->row_dim != a->row_dim)
	{
		fprintf(stderr, "ERROR: mul_cmpfmatrix_simple(c->row_dim, a->row_dim) = (%ld, %ld)\n", c->row_dim, a->row_dim);
		return;
	}
	if(c->col_dim != b->col_dim)
	{
		fprintf(stderr, "ERROR: mul_cmpfmatrix_simple(c->col_dim, b->col_dim) = (%ld, %ld)\n", c->col_dim, b->col_dim);
		return;
	}
	if(a->col_dim != b->row_dim)
	{
		fprintf(stderr, "ERROR: mul_cmpfmatrix_simple(a->col_dim, b->col_dim) = (%ld, %ld)\n", a->col_dim, b->row_dim);
		return;
	}

	mpc_init2(tmp , c->prec);
	mpc_init2(tmp2, c->prec);
	//tmp = init2_mpc_t(c->prec);
	//tmp2 = init2_mpc_t(c->prec);

	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
		{
			mpc_set_ui_ui(tmp2, 0UL, 0UL, get_bnc_default_rounding_mode());
            //set0_mpc_t(tmp2);
			for(k = 0; k < a->col_dim; k++)
			{
				mpc_mul(tmp, get_cmpfmatrix_ij(a, i, k), get_cmpfmatrix_ij(b, k, j), get_bnc_default_rounding_mode());
				mpc_add(tmp2, tmp2, tmp, get_bnc_default_rounding_mode());
			}
			set_cmpfmatrix_ij(c, i, j, tmp2);
		}
	}

	mpc_clear(tmp);
	mpc_clear(tmp2);
}

// Block matrix multiplicaiton
void mul_cmpfmatrix_block(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim)
{
	unsigned long prec;
	int row_padding_flag = 0, col_padding_flag = 0, mid_padding_flag = 0;
	long i, j, k, row_dim, col_dim, mid_dim;
	long int num_div_row, num_div_col, num_div_mid;
	long int mat_a_index[4], small_mat_a_index[4], mat_b_index[4], small_mat_b_index[4], ret_index[4], small_ret_index[4];
	//DMatrix small_ret[1024], small_mat_a[1024], small_mat_b[1024], small_tmp_mat;
	CMPFMatrix *small_ret, *small_mat_a, *small_mat_b, small_tmp_mat;

	// initialize
	prec = ret->prec;
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_cmpfmatrix_block)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}
	

	// normal matrix multiplication in case of ret_dim <= 4 
	if(((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim)) || (min_dim <= 0))
	{
		mul_cmpfmatrix_simple(ret, mat_a, mat_b);
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
	small_ret = (CMPFMatrix *)calloc(sizeof(CMPFMatrix), num_div_col);
	small_mat_a = (CMPFMatrix *)calloc(sizeof(CMPFMatrix), num_div_mid);
	small_mat_b = (CMPFMatrix *)calloc(sizeof(CMPFMatrix), num_div_mid);
	for(i = 0; i < num_div_col; i++)
		small_ret[i] = init2_cmpfmatrix(min_dim, min_dim, prec);

	for(i = 0; i < num_div_mid; i++)
	{
		small_mat_a[i] = init2_cmpfmatrix(min_dim, min_dim, prec);
		small_mat_b[i] = init2_cmpfmatrix(min_dim, min_dim, prec);
	}
	small_tmp_mat = init2_cmpfmatrix(min_dim, min_dim, prec);

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
			//subst_cmpfmatrix_partial(small_mat_a[j], small_mat_a_index, mat_a, mat_a_index);
			subst_cmpfmatrix_partial_checked(small_mat_a[j], small_mat_a_index, mat_a, mat_a_index);
		}

		for(j = 0; j < num_div_col; j++)
		{

			set0_cmpfmatrix(small_ret[j]);
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
				//subst_cmpfmatrix_partial(small_mat_b[k], small_mat_b_index, mat_b, mat_b_index);
				subst_cmpfmatrix_partial_checked(small_mat_b[k], small_mat_b_index, mat_b, mat_b_index);

				// ret[j] += small_mat_a[i][k] * small_mat_b[k][j];
				mul_cmpfmatrix(small_tmp_mat, small_mat_a[k], small_mat_b[k]);
				add_cmpfmatrix(small_ret[j], small_ret[j], small_tmp_mat);
			}
			ret_index[0] = i * min_dim;
			ret_index[1] = (i + 1) * min_dim;
			ret_index[2] = j * min_dim;
			ret_index[3] = (j + 1) * min_dim;
			small_ret_index[0] = 0;
			small_ret_index[1] = min_dim;
			small_ret_index[2] = 0;
			small_ret_index[3] = min_dim;
			//subst_cmpfmatrix_partial(ret, ret_index, small_ret[j], small_ret_index);
			subst_cmpfmatrix_partial_checked(ret, ret_index, small_ret[j], small_ret_index);
		}
	}

	// free
	free_cmpfmatrix(small_tmp_mat);
	for(i = 0; i < num_div_col; i++)
		free_cmpfmatrix(small_ret[i]);

	for(i = 0; i < num_div_mid; i++)
	{
		free_cmpfmatrix(small_mat_a[i]);
		free_cmpfmatrix(small_mat_b[i]);
	}
	free(small_ret);
	free(small_mat_a);
	free(small_mat_b);
}

// Padding to 2-powered dimensional matrix
CMPFMatrix init_static_padding_cmpfmatrix_strassen(CMPFMatrix orig_mat)
{
	unsigned long prec;
	CMPFMatrix ret = NULL;
	long int ret_row_dim, ret_col_dim, min_dim, i, j;

	if(orig_mat == NULL)
	{
		fprintf(stderr, "Warning: orig_mat is empty!(padding_cmpfmatrix_strassen)\n");
		return NULL;
	}

	prec = orig_mat->prec;

	ret_row_dim = (long int)pow(2.0, ceil(mylog2((double)orig_mat->row_dim)));
	ret_col_dim = (long int)pow(2.0, ceil(mylog2((double)orig_mat->col_dim)));

	//printf("Padding: row_dim %ld -> %ld, col_dim %ld -> %ld\n", orig_mat->row_dim, ret_row_dim, orig_mat->col_dim, ret_col_dim);

	ret = init2_cmpfmatrix(ret_row_dim, ret_col_dim, prec);
	if(ret == NULL)
	{
		fprintf(stderr, "Warning: padding matrix cannot be allocated!(padding_cmpfmatrix_strassen)\n");
		return NULL;
	}

	// ret = [ A | 0 ]
	//       [---+---]
	//       [ 0 | I ]
	for(i = 0; i < orig_mat->row_dim; i++)
	{
		for(j = 0; j < orig_mat->col_dim; j++)
			set_cmpfmatrix_ij(ret, i, j, get_cmpfmatrix_ij(orig_mat, i, j));
	}

	min_dim = (ret_row_dim < ret_col_dim) ? ret_row_dim : ret_col_dim;
	for(i = orig_mat->row_dim; i < min_dim; i++)
		set_cmpfmatrix_ij_ui(ret, i, i, 1UL);

	return ret;
}

// Padding to even dimensional matrix
CMPFMatrix init_dynamic_padding_cmpfmatrix_strassen(CMPFMatrix orig_mat)
{
	unsigned long prec;
	CMPFMatrix ret = NULL;
	long int ret_row_dim, ret_col_dim, min_dim, i, j;

	if(orig_mat == NULL)
	{
		fprintf(stderr, "Warning: orig_mat is empty!(padding_cmpfmatrix_strassen)\n");
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

	ret = init2_cmpfmatrix(ret_row_dim, ret_col_dim, prec);
	if(ret == NULL)
	{
		fprintf(stderr, "Warning: padding matrix cannot be allocated!(padding_cmpfmatrix_strassen)\n");
		return NULL;
	}

	// ret = [ A | 0 ]
	//       [---+---]
	//       [ 0 | I ]
	for(i = 0; i < orig_mat->row_dim; i++)
	{
		for(j = 0; j < orig_mat->col_dim; j++)
			set_cmpfmatrix_ij(ret, i, j, get_cmpfmatrix_ij(orig_mat, i, j));
	}

/*	min_dim = (ret_row_dim < ret_col_dim) ? ret_row_dim : ret_col_dim;
	for(i = orig_mat->row_dim; i < min_dim; i++)
		set_cmpfmatrix_ij_ui(ret, i, i, 1UL);
*/
	return ret;
}

// Padding to even dimensional matrix
CMPFMatrix init_dynamic_padding_cmpfmatrix_strassen2(CMPFMatrix orig_mat, long int min_dim)
{
	unsigned long prec;
	CMPFMatrix ret = NULL;
	long int ret_row_dim, ret_col_dim, i, j;

	if(orig_mat == NULL)
	{
		fprintf(stderr, "Warning: orig_mat is empty!(padding_cmpfmatrix_strassen)\n");
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

	ret = init2_cmpfmatrix(ret_row_dim, ret_col_dim, prec);
	if(ret == NULL)
	{
		fprintf(stderr, "Warning: padding matrix cannot be allocated!(padding_cmpfmatrix_strassen)\n");
		return NULL;
	}

	// ret = [ A | 0 ]
	//       [---+---]
	//       [ 0 | I ]
	for(i = 0; i < orig_mat->row_dim; i++)
	{
		for(j = 0; j < orig_mat->col_dim; j++)
			set_cmpfmatrix_ij(ret, i, j, get_cmpfmatrix_ij(orig_mat, i, j));
	}
/*
	min_dim = (ret_row_dim < ret_col_dim) ? ret_row_dim : ret_col_dim;
	for(i = orig_mat->row_dim; i < min_dim; i++)
		set_cmpfmatrix_ij_ui(ret, i, i, 1UL);
*/
	return ret;
}

// Strassen's Algorithm with static padding
void mul_cmpfmatrix_strassen_odd_padding(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim)
{
	long int tmp_ret_index[4], ret_index[4];
	CMPFMatrix tmp_ret, tmp_mat_a, tmp_mat_b;

	//printf("start mul_cmpfmatrix_strassen_odd_padding\n");

	// padding
#ifdef USE_STATIC_PADDING
	tmp_ret = init_static_padding_cmpfmatrix_strassen(ret);
	tmp_mat_a = init_static_padding_cmpfmatrix_strassen(mat_a);
	tmp_mat_b = init_static_padding_cmpfmatrix_strassen(mat_b);
#else
//	tmp_ret = init_dynamic_padding_cmpfmatrix_strassen(ret);
//	tmp_mat_a = init_dynamic_padding_cmpfmatrix_strassen(mat_a);
//	tmp_mat_b = init_dynamic_padding_cmpfmatrix_strassen(mat_b);
	tmp_ret = init_dynamic_padding_cmpfmatrix_strassen2(ret, min_dim);
	tmp_mat_a = init_dynamic_padding_cmpfmatrix_strassen2(mat_a, min_dim);
	tmp_mat_b = init_dynamic_padding_cmpfmatrix_strassen2(mat_b, min_dim);
#endif

	// strassen
#ifdef USE_WINOGRAD
	mul_cmpfmatrix_winograd_even(tmp_ret, tmp_mat_a, tmp_mat_b, min_dim);
#else
	mul_cmpfmatrix_strassen_even(tmp_ret, tmp_mat_a, tmp_mat_b, min_dim);
#endif

	// substitute
	tmp_ret_index[0] = 0;
	tmp_ret_index[1] = ret->row_dim;
	tmp_ret_index[2] = 0;
	tmp_ret_index[3] = ret->col_dim;
	ret_index[0] = 0;
	ret_index[1] = ret->row_dim;
	ret_index[2] = 0;
	ret_index[3] = ret->col_dim;

	subst_cmpfmatrix_partial(ret, ret_index, tmp_ret, tmp_ret_index);

	// free
	free_cmpfmatrix(tmp_ret);
	free_cmpfmatrix(tmp_mat_a);
	free_cmpfmatrix(tmp_mat_b);

	//printf("end of mul_cmpfmatrix_strassen_odd_padding\n");

}

// clear counter
void reset_num_mul_cmpfmatrix_strassen(void)
{
	num_addsub_mul_cmpfmatrix_strassen = 0;
	num_mul_mul_cmpfmatrix_strassen = 0;
}

// get counters
void get_num_mul_cmpfmatrix_strassen(long int *num_addsub, long int *num_mul)
{
	//printf("num_addsub_mul_cmpfmatrix_strassen: %ld\n", num_addsub_mul_cmpfmatrix_strassen);
	//printf("num_mul_mul_cmpfmatrix_strassen   : %ld\n", num_mul_mul_cmpfmatrix_strassen);

	if(num_addsub != NULL)
		*num_addsub = num_addsub_mul_cmpfmatrix_strassen;
	if(num_mul != NULL)
		*num_mul = num_mul_mul_cmpfmatrix_strassen;
}

// print counters
void print_num_mul_cmpfmatrix_strassen(long int *num_addsub, long int *num_mul)
{
	printf("num_addsub_mul_cmpfmatrix_strassen: %ld\n", num_addsub_mul_cmpfmatrix_strassen);
	printf("num_mul_mul_cmpfmatrix_strassen   : %ld\n", num_mul_mul_cmpfmatrix_strassen);

	get_num_mul_cmpfmatrix_strassen(num_addsub, num_mul);

}

// Fit dimension to be multiple of min_dim
void mul_cmpfmatrix_strassen(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim)
{
	long int row_k, col_k, mid_row_k, mid_col_k;
	long int row_dim, col_dim, mid_dim;
//	CMPFVector diag_left, diag_right;

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_cmpfmatrix_strassen)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	// normal matrix multiplication in case of ret_dim <= 4 
	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	{
		//mul_cmpfmatrix(ret, mat_a, mat_b);

		// counting the number of arithmetic
		num_addsub_mul_cmpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		num_mul_mul_cmpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		mul_cmpfmatrix_simple(ret, mat_a, mat_b);
		return;
	}

	// scaling 
//	diag_left = init2_cmpfvector(row_dim, mat_a->prec);
//	diag_right = init2_cmpfvector(col_dim, mat_b->prec);

//	left_scaling_cmpfmatrix(mat_a, diag_left, NULL);
//	right_scaling_cmpfmatrix(mat_b, diag_right, NULL);

	// dynamic peeling in case of odd dim
	// [ A11   a12 ] [ B11   b12 ] = [ A11*B11 + a12 * b21^T   A11*b12 + a12 * b22    ]
	// [ a21^T a22 ] [ b21^T b22 ]   [ a21^T*B11 + a22 * b21^T a21^T * b12 + a22 * b22]
	if((ret->row_dim % 2 == 1) || (ret->col_dim % 2 == 1) || (mid_dim % 2 == 1))
	{
		//printf("%d, %d or %d are odd -> ", ret->row_dim, ret->col_dim, mid_dim);
#ifdef PEELING_ONLY
		mul_cmpfmatrix_strassen_odd_peeling(ret, mat_a, mat_b, min_dim);
#elif PADDING_ONLY
		mul_cmpfmatrix_strassen_odd_padding(ret, mat_a, mat_b, min_dim);
#else
		row_k = (long int)floor(mylog2((double)ret->row_dim));
		col_k = (long int)floor(mylog2((double)ret->col_dim));

		mid_row_k = (long int)pow(2.0, row_k - 1) * 3;
		mid_col_k = (long int)pow(2.0, col_k - 1) * 3;

		//printf("2^%ld <= %ld <= 2^%ld\n", row_k, ret->row_dim, row_k + 1);

		// dynamic peeling
		//if(ret->row_dim < mid_row_k)
		if((ret->row_dim % min_dim) < (min_dim / 2))
			mul_cmpfmatrix_strassen_odd_peeling(ret, mat_a, mat_b, min_dim);
		// padding
		else
			mul_cmpfmatrix_strassen_odd_padding(ret, mat_a, mat_b, min_dim);

		//printf("end\n");
#endif
	}
	// normal strassen algorithm in case of even dim
	else
	{
		//printf("%d is even -> ", ret->row_dim);
#ifdef USE_WINOGRAD
		mul_cmpfmatrix_winograd_even(ret, mat_a, mat_b, min_dim);
#else
		mul_cmpfmatrix_strassen_even(ret, mat_a, mat_b, min_dim);
#endif
		//printf("end\n");
	}

//	mul_cmpfmatrix_ddiag_mat(ret, diag_left, 0, ret, diag_right, 0);

//	free_cmpfvector(diag_left);
//	free_cmpfvector(diag_right);

}

// Strassen's Algorithm with Dynamic peeling
void mul_cmpfmatrix_strassen_odd_peeling(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim)
{
	unsigned long prec;
	long int i, j, row_dim, row_dim_h, col_dim, col_dim_h, mid_dim, mid_dim_h, tmp_dim_h, tmp_dim;
	CMPFMatrix mat_a11, mat_b11, mat_c11, mat_tmp;
	CMPFVector vec_a12, vec_a21, vec_b12, vec_b21, vec_c12, vec_c21, vec_tmp12, vec_tmp21;
	mpc_t a22, b22, c22, tmp;
    mpf_t abs_a22, abs_b22;

	// initialize
	prec = ret->prec;
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_cmpfmatrix_strassen_odd_peeling)\n", mat_a->col_dim, mat_b->row_dim);
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
		num_addsub_mul_cmpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		num_mul_mul_cmpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		//mul_cmpfmatrix(ret, mat_a, mat_b);
		mul_cmpfmatrix_simple(ret, mat_a, mat_b);
		return;
	}

	//printf("start mul_cmpfmatrix_strassen_odd_peeling\n");

	// tmp_dim_h = mid_dim_h or col_dim_h
	tmp_dim_h = mat_b->col_dim - 1;
	if(mid_dim_h < tmp_dim_h)
		tmp_dim_h = mid_dim_h;

	// Initialize
	mat_a11 = init2_cmpfmatrix(row_dim_h, mid_dim_h, prec);
	mat_b11 = init2_cmpfmatrix(mid_dim_h, col_dim_h, prec);
	mat_c11 = init2_cmpfmatrix(row_dim_h, col_dim_h, prec);
	mat_tmp = init2_cmpfmatrix(row_dim_h, col_dim_h, prec);

	vec_a12 = NULL;
	vec_b21 = NULL;
	vec_b12 = NULL;
	vec_a21 = NULL;
	vec_c12 = init2_cmpfvector(row_dim_h, prec);
	vec_c21 = init2_cmpfvector(col_dim_h, prec);
	vec_tmp12 = init2_cmpfvector(row_dim_h, prec);
	vec_tmp21 = init2_cmpfvector(col_dim_h, prec);

	mpc_init2(a22, prec); mpc_set_ui_ui(a22, 0UL, 0UL, get_bnc_default_rounding_mode());
	mpc_init2(b22, prec); mpc_set_ui_ui(b22, 0UL, 0UL, get_bnc_default_rounding_mode());
	mpc_init2(c22, prec);
	mpc_init2(tmp, prec);
	//a22 = init2_mpc_t(prec);
	//b22 = init2_mpc_t(prec);
	//c22 = init2_mpc_t(prec);
	//tmp = init2_mpc_t(prec);
    mpf_init2(abs_a22, prec);
    mpf_init2(abs_b22, prec);

	// set matrix elements to mat_a11
	for(i = 0; i < row_dim_h; i++)
	{
		for(j = 0; j < mid_dim_h; j++)
			set_cmpfmatrix_ij(mat_a11, i, j, get_cmpfmatrix_ij(mat_a, i, j));
	}

	// set matrix elements to vec_b11
	for(i = 0; i < mid_dim_h; i++)
	{
		for(j = 0; j < col_dim_h; j++)
			set_cmpfmatrix_ij(mat_b11, i, j, get_cmpfmatrix_ij(mat_b, i, j));
	}

	// set matrix elements to vec_a12 and vec_b21
	if(mid_dim_h < mid_dim)
	{
		//printf("set vec_a12, b21\n");
		vec_a12 = init2_cmpfvector(row_dim_h, prec);
		vec_b21 = init2_cmpfvector(col_dim_h, prec); // fix!: 2014-03-19 by T.Kouya
		for(i = 0; i < row_dim_h; i++)
			set_cmpfvector_i(vec_a12, i, get_cmpfmatrix_ij(mat_a, i, mat_a->col_dim - 1));

		//printf("set vec_a12\n");

		for(i = 0; i < col_dim_h; i++)
			set_cmpfvector_i(vec_b21, i, get_cmpfmatrix_ij(mat_b, mat_b->row_dim - 1, i));

		//printf("set vec_b21\n");
	}

	// set matrix elements to vec_a21
	if(row_dim_h < row_dim)
	{
		vec_a21 = init2_cmpfvector(mid_dim_h, prec);
		for(i = 0; i < mid_dim_h; i++)
			set_cmpfvector_i(vec_a21, i, get_cmpfmatrix_ij(mat_a, mat_a->row_dim - 1, i));

		mpc_init2(a22, prec);
        //a22 = init2_mpc_t(prec);
		if(mid_dim_h < mid_dim)
			mpc_set(a22, get_cmpfmatrix_ij(mat_a, mat_a->row_dim - 1, mat_a->col_dim - 1), get_bnc_default_rounding_mode());
			//mpf_set(a22, get_cmpfmatrix_ij(mat_a, mat_a->row_dim - 1, mat_a->col_dim - 1));

		//printf("set vec_a21, a22\n");
	}

	// set matrix elements to vec_b12
	if(col_dim_h < col_dim)
	{
		vec_b12 = init2_cmpfvector(mid_dim_h, prec);
		for(i = 0; i < mid_dim_h; i++)
			set_cmpfvector_i(vec_b12, i, get_cmpfmatrix_ij(mat_b, i, mat_b->col_dim - 1));

		mpc_init2(b22, prec);
		//b22 = init2_mpc_t(prec);
		if(mid_dim_h < mid_dim)
			mpc_set(b22, get_cmpfmatrix_ij(mat_b, mat_b->row_dim - 1, mat_b->col_dim - 1), get_bnc_default_rounding_mode());
			//mpf_set(b22, get_cmpfmatrix_ij(mat_b, mat_b->row_dim - 1, mat_b->col_dim - 1));


		//printf("set vec_a12, b22\n");
	}

	// dynamic peeling in case of odd dim
	// [ A11   a12 ] [ B11   b12 ] = [ A11  * B11 + a12 * b21^T  A11   * b12 + a12 * b22 ]
	// [ a21^T a22 ] [ b21^T b22 ]   [ a21^T* B11 + a22 * b21^T  a21^T * b12 + a22 * b22 ]

	//printf("starting C11 = A11 * B11...\n");

	// C11 = A11 * B11
#ifdef USE_WINOGRAD
	mul_cmpfmatrix_winograd_even(mat_c11, mat_a11, mat_b11, min_dim);
#else
	mul_cmpfmatrix_strassen_even(mat_c11, mat_a11, mat_b11, min_dim);
#endif

	//printf("end of C11 = A11 * B11\n");

	// C11 += a12 * b21^T
	if((vec_a12 != NULL) && (vec_b21 != NULL))
	{
		//printf("starting C11 += a12 * b21...\n");

		// counting the number of arithmetic
		num_addsub_mul_cmpfmatrix_strassen += row_dim_h * col_dim_h;
		num_mul_mul_cmpfmatrix_strassen += row_dim_h * col_dim_h;

		for(i = 0; i < row_dim_h; i++)
		{
			for(j = 0; j < col_dim_h; j++)
			{
				mpc_mul(get_cmpfmatrix_ij(mat_tmp, i, j), get_cmpfvector_i(vec_a12, i), get_cmpfvector_i(vec_b21, j), get_bnc_default_rounding_mode());
			}
		}
		add_cmpfmatrix(mat_c11, mat_c11, mat_tmp);

		//printf("end of C11 += a12 * b21...\n");
	}

	for(i = 0; i < row_dim_h; i++)
		for(j = 0; j < col_dim_h; j++) 
			set_cmpfmatrix_ij(ret, i, j, get_cmpfmatrix_ij(mat_c11, i, j));

	// c12 := A11 * b12
	if(vec_b12 != NULL)
	{
		// counting the number of arithmetic
		num_addsub_mul_cmpfmatrix_strassen += mat_a11->row_dim * mat_a11->col_dim;
		num_mul_mul_cmpfmatrix_strassen += mat_a11->row_dim * mat_a11->col_dim;

		mul_cmpfmatrix_cmpfvec(vec_c12, mat_a11, vec_b12);

		//printf("c12 = A11 * b12\n");
	}

 	// c12 += b22 * a12
    mpc_abs(abs_b22, b22, get_bnc_default_rounding_mode());
 	if((vec_a12 != NULL) && (mpf_cmp_ui(abs_b22, 0UL) != 0))
	{
		// counting the number of arithmetic
		num_addsub_mul_cmpfmatrix_strassen += vec_c12->dim;
		num_mul_mul_cmpfmatrix_strassen += vec_tmp12->dim;

		cmul_cmpfvector(vec_tmp12, b22, vec_a12);
		add_cmpfvector(vec_c12, vec_c12, vec_tmp12);

		//printf("c12 += b22 * a12\n");
	}

	// Fix! 2016-08-18 by T.Kouya
    mpc_abs(abs_b22, b22, get_bnc_default_rounding_mode());
	if((vec_b12 != NULL) || ((vec_a12 != NULL) && (mpf_cmp_ui(abs_b22, 0UL) != 0))) 
	{
		for(i = 0; i < row_dim_h; i++) // Fix! 2016-08-18 by T.Kouya
			set_cmpfmatrix_ij(ret, i, ret->col_dim - 1, get_cmpfvector_i(vec_c12, i));
	}

	//printf("vec_c12\n");

	// c21 := a21^T*B11
	if(vec_a21 != NULL)
	{
		// counting the number of arithmetic
		num_addsub_mul_cmpfmatrix_strassen += mat_b11->row_dim * mat_b11->col_dim;
		num_mul_mul_cmpfmatrix_strassen += mat_b11->row_dim * mat_b11->col_dim;

		mul_cmpfmatrixt_cmpfvec(vec_c21, mat_b11, vec_a21);
	}
	//printf("c21 = a21^T * B11\n");

	// c21 += a22 * b21^T
    mpc_abs(abs_a22, a22, get_bnc_default_rounding_mode());
	if((vec_b21 != NULL) && (mpf_cmp_ui(abs_a22, 0UL) != 0))
	{
		// counting the number of arithmetic
		num_addsub_mul_cmpfmatrix_strassen += vec_c21->dim;
		num_mul_mul_cmpfmatrix_strassen += vec_tmp21->dim;

		cmul_cmpfvector(vec_tmp21, a22, vec_b21);
		add_cmpfvector(vec_c21, vec_c21, vec_tmp21);
	}
	//printf("c21 += a22 * b21^T\n");

	if(vec_a21 != NULL)
	{
		for(i = 0; i < col_dim_h; i++)
			set_cmpfmatrix_ij(ret, ret->row_dim - 1, i, get_cmpfvector_i(vec_c21, i));
	}
	//printf("c21\n");

	// c22 := a21^T * b12
	if((vec_a21 != NULL) && (vec_b12 != NULL))
	{
		// counting the number of arithmetic
		num_addsub_mul_cmpfmatrix_strassen += vec_a21->dim;
		num_mul_mul_cmpfmatrix_strassen += vec_tmp21->dim;

		ip_cmpfvector(c22, vec_a21, vec_b12);
	}
	//printf("c22 += a21^T * b12\n");

	// c22 += a22 * b22
    mpc_abs(abs_a22, a22, get_bnc_default_rounding_mode());
    mpc_abs(abs_b22, b22, get_bnc_default_rounding_mode());
	if((mpf_cmp_ui(abs_a22, 0UL) != 0) || (mpf_cmp_ui(abs_b22, 0UL) != 0))
	{
		// counting the number of arithmetic
		num_addsub_mul_cmpfmatrix_strassen += 1;
		num_mul_mul_cmpfmatrix_strassen += 1;

		mpc_mul(tmp, a22, b22, get_bnc_default_rounding_mode());
		mpc_add(c22, c22, tmp, get_bnc_default_rounding_mode());
	}
	//printf("c22 += a22 * b22\n");

	if((vec_a21 != NULL) && (vec_b12 != NULL))
		set_cmpfmatrix_ij(ret, ret->row_dim - 1, ret->col_dim - 1, c22);

	//printf("c22\n");

	// free
	free_cmpfmatrix(mat_a11);
	free_cmpfmatrix(mat_b11);
	free_cmpfmatrix(mat_c11);
	//printf("free_cmpfmatrix a11, b11, c11\n");
	free_cmpfmatrix(mat_tmp);

	//printf("free_cmpfmatrix\n");

	if(vec_a12 != NULL)
		free_cmpfvector(vec_a12);

	if(vec_b21 != NULL)
		free_cmpfvector(vec_b21);

	if(vec_a21 != NULL)
		free_cmpfvector(vec_a21);

	if(vec_b12 != NULL)
		free_cmpfvector(vec_b12);

	//printf("free_cmpfvector\n");

	free_cmpfvector(vec_c12);
	free_cmpfvector(vec_c21);
	free_cmpfvector(vec_tmp12);
	free_cmpfvector(vec_tmp21);

	//printf("free_cmpfvector2\n");

	mpc_clear(a22);
	mpc_clear(b22);
	mpc_clear(c22);
	mpc_clear(tmp);
    mpf_clear(abs_a22);
    mpf_clear(abs_b22);
	//printf("mpfclear\n");
	//printf("end of mul_cmpfmatrix_strassen_odd_peeling\n");
}

// Strassen's Algorithm
void mul_cmpfmatrix_strassen_even(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim)
{
	unsigned long prec;
//	long int min_dim = 4; // = 2^2
	CMPFMatrix mat_p[7], mat_tmp_a[7], mat_tmp_b[7], mat_tmp_c[4];
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
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_cmpfmatrix_strassen_even)\n", mat_a->col_dim, mat_b->row_dim);
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
		num_addsub_mul_cmpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		num_mul_mul_cmpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		//mul_cmpfmatrix(ret, mat_a, mat_b);
		mul_cmpfmatrix_simple(ret, mat_a, mat_b);
		return;
	}

	for(i = 0; i < 7; i++)
	{
		mat_p[i] = init2_cmpfmatrix(row_dim_h, col_dim_h, prec);
		mat_tmp_a[i] = init2_cmpfmatrix(row_dim_h, mid_dim_h, prec);
		mat_tmp_b[i] = init2_cmpfmatrix(mid_dim_h, col_dim_h, prec);
	}
	for(i = 0; i < 4; i++)
		mat_tmp_c[i] = init2_cmpfmatrix(row_dim_h, col_dim_h, prec);

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
	num_addsub_mul_cmpfmatrix_strassen += row_dim_h * mid_dim_h;

//	add_cmpfmatrix_partial(mat_tmp_a[0], ret_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);
	add_cmpfmatrix_partial(mat_tmp_a[0], mat_tmp_a_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);

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
	num_addsub_mul_cmpfmatrix_strassen += mid_dim_h * col_dim_h;

//	add_cmpfmatrix_partial(mat_tmp_b[0], ret_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);
	add_cmpfmatrix_partial(mat_tmp_b[0], mat_tmp_b_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);

	// P1 = tmp_a * tmp_b
//	printf("P1:\n");
	mul_cmpfmatrix_strassen(mat_p[0], mat_tmp_a[0], mat_tmp_b[0], min_dim);

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
	num_addsub_mul_cmpfmatrix_strassen += row_dim_h * mid_dim_h;

//	add_cmpfmatrix_partial(mat_tmp_a[1], ret_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);
	add_cmpfmatrix_partial(mat_tmp_a[1], mat_tmp_a_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);

	// B11
	mat_br_index[1][0] = 0;
	mat_br_index[1][1] = mid_dim_h;
	mat_br_index[1][2] = 0;
	mat_br_index[1][3] = col_dim_h;
//	subst_cmpfmatrix_partial(mat_tmp_b[1], ret_index, mat_b, mat_br_index[1]);
	subst_cmpfmatrix_partial(mat_tmp_b[1], mat_tmp_b_index, mat_b, mat_br_index[1]);

	// P2 = tmp_a * tmp_b
	//printf("P2:\n");
	mul_cmpfmatrix_strassen(mat_p[1], mat_tmp_a[1], mat_tmp_b[1], min_dim);

	// -------------------------------
	// P3 := A11 * (B12 - B22)
	// -------------------------------

	// A11
	mat_ar_index[2][0] = 0;
	mat_ar_index[2][1] = row_dim_h;
	mat_ar_index[2][2] = 0;
	mat_ar_index[2][3] = mid_dim_h;
//	subst_cmpfmatrix_partial(mat_tmp_a[2], ret_index, mat_a, mat_ar_index[2]);
	subst_cmpfmatrix_partial(mat_tmp_a[2], mat_tmp_a_index, mat_a, mat_ar_index[2]);

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
	num_addsub_mul_cmpfmatrix_strassen += mid_dim_h * col_dim_h;

//	sub_cmpfmatrix_partial(mat_tmp_b[2], ret_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);
	sub_cmpfmatrix_partial(mat_tmp_b[2], mat_tmp_b_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);

	// P3 = tmp_a * tmp_b
	//printf("P3:\n");
	mul_cmpfmatrix_strassen(mat_p[2], mat_tmp_a[2], mat_tmp_b[2], min_dim);

	// -------------------------------
	// P4 := A22 * (B21 - B11)
	// -------------------------------

	// A22
	mat_ar_index[3][0] = row_dim_h;
	mat_ar_index[3][1] = row_dim;
	mat_ar_index[3][2] = mid_dim_h;
	mat_ar_index[3][3] = mid_dim;

	// counting the number of arithmetic
	num_addsub_mul_cmpfmatrix_strassen += mid_dim_h * col_dim_h;

//	subst_cmpfmatrix_partial(mat_tmp_a[3], ret_index, mat_a, mat_ar_index[3]);
	subst_cmpfmatrix_partial(mat_tmp_a[3], mat_tmp_a_index, mat_a, mat_ar_index[3]);

	// B21 - B11
	mat_br_index[3][0] = mid_dim_h;
	mat_br_index[3][1] = mid_dim;
	mat_br_index[3][2] = 0;
	mat_br_index[3][3] = col_dim_h;

	mat_bl_index[3][0] = 0;
	mat_bl_index[3][1] = mid_dim_h;
	mat_bl_index[3][2] = 0;
	mat_bl_index[3][3] = col_dim_h;
//	sub_cmpfmatrix_partial(mat_tmp_b[3], ret_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);
	sub_cmpfmatrix_partial(mat_tmp_b[3], mat_tmp_b_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);

	// P4 = tmp_a * tmp_b
	//printf("P4:\n");
	mul_cmpfmatrix_strassen(mat_p[3], mat_tmp_a[3], mat_tmp_b[3], min_dim);

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
	num_addsub_mul_cmpfmatrix_strassen += row_dim_h * mid_dim_h;

//	add_cmpfmatrix_partial(mat_tmp_a[4], ret_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);
	add_cmpfmatrix_partial(mat_tmp_a[4], mat_tmp_a_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);

	// B22
	mat_br_index[4][0] = mid_dim_h;
	mat_br_index[4][1] = mid_dim;
	mat_br_index[4][2] = col_dim_h;
	mat_br_index[4][3] = col_dim;
//	subst_cmpfmatrix_partial(mat_tmp_b[4], ret_index, mat_b, mat_br_index[4]);
	subst_cmpfmatrix_partial(mat_tmp_b[4], mat_tmp_b_index, mat_b, mat_br_index[4]);

	// P5 = tmp_a * tmp_b
	//printf("P5:\n");
	mul_cmpfmatrix_strassen(mat_p[4], mat_tmp_a[4], mat_tmp_b[4], min_dim);

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
	num_addsub_mul_cmpfmatrix_strassen += row_dim_h * mid_dim_h;

//	sub_cmpfmatrix_partial(mat_tmp_a[5], ret_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);
	sub_cmpfmatrix_partial(mat_tmp_a[5], mat_tmp_a_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);

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
	num_addsub_mul_cmpfmatrix_strassen += mid_dim_h * col_dim_h;

//	add_cmpfmatrix_partial(mat_tmp_b[5], ret_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);
	add_cmpfmatrix_partial(mat_tmp_b[5], mat_tmp_b_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);

	// P6 = tmp_a * tmp_b
	//printf("P6:\n");
	mul_cmpfmatrix_strassen(mat_p[5], mat_tmp_a[5], mat_tmp_b[5], min_dim);

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
	num_addsub_mul_cmpfmatrix_strassen += row_dim_h * mid_dim_h;

//	sub_cmpfmatrix_partial(mat_tmp_a[6], ret_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);
	sub_cmpfmatrix_partial(mat_tmp_a[6], mat_tmp_a_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);

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
	num_addsub_mul_cmpfmatrix_strassen += mid_dim_h * col_dim_h;

//	add_cmpfmatrix_partial(mat_tmp_b[6], ret_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);
	add_cmpfmatrix_partial(mat_tmp_b[6], mat_tmp_b_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);

	// P7 = tmp_a * tmp_b
	//printf("P7:\n");
	mul_cmpfmatrix_strassen(mat_p[6], mat_tmp_a[6], mat_tmp_b[6], min_dim);

	// -------------------------------
	// C11 := P1 + P4 - P5 + P7
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_cmpfmatrix_strassen += 3 * row_dim_h * col_dim_h;

	add_cmpfmatrix(mat_tmp_c[0], mat_p[0], mat_p[3]);
	sub_cmpfmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[4]);
	add_cmpfmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[6]);
	mat_c_index[0][0] = 0;
	mat_c_index[0][1] = row_dim_h;
	mat_c_index[0][2] = 0;
	mat_c_index[0][3] = col_dim_h;

	subst_cmpfmatrix_partial(ret, mat_c_index[0], mat_tmp_c[0], ret_index);

	// -------------------------------
	// C12 := P3 + P5
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_cmpfmatrix_strassen += row_dim_h * col_dim_h;

	add_cmpfmatrix(mat_tmp_c[1], mat_p[2], mat_p[4]);
	mat_c_index[1][0] = 0;
	mat_c_index[1][1] = row_dim_h;
	mat_c_index[1][2] = col_dim_h;
	mat_c_index[1][3] = col_dim;
	subst_cmpfmatrix_partial(ret, mat_c_index[1], mat_tmp_c[1], ret_index);

	// -------------------------------
	// C21 := P2 + P4
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_cmpfmatrix_strassen += row_dim_h * col_dim_h;

	add_cmpfmatrix(mat_tmp_c[2], mat_p[1], mat_p[3]);
	mat_c_index[2][0] = row_dim_h;
	mat_c_index[2][1] = row_dim;
	mat_c_index[2][2] = 0;
	mat_c_index[2][3] = col_dim_h;
	subst_cmpfmatrix_partial(ret, mat_c_index[2], mat_tmp_c[2], ret_index);

	// -------------------------------
	// C22 := P1 + P3 - P2 + P6
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_cmpfmatrix_strassen += 3 * row_dim_h * col_dim_h;

	add_cmpfmatrix(mat_tmp_c[3], mat_p[0], mat_p[2]);
	sub_cmpfmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[1]);
	add_cmpfmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[5]);
	mat_c_index[3][0] = row_dim_h;
	mat_c_index[3][1] = row_dim;
	mat_c_index[3][2] = col_dim_h;
	mat_c_index[3][3] = col_dim;
	subst_cmpfmatrix_partial(ret, mat_c_index[3], mat_tmp_c[3], ret_index);

	// free
	for(i = 0; i < 7; i++)
	{
		free_cmpfmatrix(mat_p[i]);
		free_cmpfmatrix(mat_tmp_a[i]);
		free_cmpfmatrix(mat_tmp_b[i]);
	}
	for(i = 0; i < 4; i++)
		free_cmpfmatrix(mat_tmp_c[i]);
}

// Winograd Variant of Strassen's Algorithm
void mul_cmpfmatrix_winograd_even(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim)
{
//	long int min_dim = 4; // = 2^2
	unsigned long prec;
	CMPFMatrix mat_s[8], mat_m[7], mat_t[2], mat_tmp_a[4], mat_tmp_b[4], mat_tmp_c[4];
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
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_cmpfmatrix_winograd_even)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	row_dim_h = row_dim / 2;
	col_dim_h = col_dim / 2;
	mid_dim_h = mid_dim / 2;

	// normal matrix multiplication in case of ret_dim <= 4 
//	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	if((ret->row_dim <= min_dim) || (ret->col_dim <= min_dim) || (mid_dim <= min_dim))
	{
		mul_cmpfmatrix_simple(ret, mat_a, mat_b);

		// counting the number of arithmetic
		num_addsub_mul_cmpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		num_mul_mul_cmpfmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		return;
	}

	for(i = 0; i < 4; i++)
	{
		mat_s[i] = init2_cmpfmatrix(row_dim_h, mid_dim_h, prec);
		mat_tmp_a[i] = init2_cmpfmatrix(row_dim_h, mid_dim_h, prec);

		mat_s[i + 4] = init2_cmpfmatrix(mid_dim_h, col_dim_h, prec);
		mat_tmp_b[i] = init2_cmpfmatrix(mid_dim_h, col_dim_h, prec);

		mat_tmp_c[i] = init2_cmpfmatrix(row_dim_h, col_dim_h, prec);
	}
	for(i = 0; i < 7; i++)
		mat_m[i] = init2_cmpfmatrix(row_dim_h, col_dim_h, prec);

	mat_t[0] = init2_cmpfmatrix(row_dim_h, col_dim_h, prec);
	mat_t[1] = init2_cmpfmatrix(row_dim_h, col_dim_h, prec);

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
		subst_cmpfmatrix_partial(mat_tmp_a[i], mat_tmp_a_index, mat_a, mat_a_index[i]);
		subst_cmpfmatrix_partial(mat_tmp_b[i], mat_tmp_b_index, mat_b, mat_b_index[i]);
	}
//	printf("subst a, b...\n");

	// -------------------------------
	// S1 := A21 + A22
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_cmpfmatrix_strassen += row_dim_h * mid_dim_h;

	add_cmpfmatrix(mat_s[0], mat_tmp_a[2], mat_tmp_a[3]);

	// -------------------------------
	// S2 := S1 - A11
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_cmpfmatrix_strassen += row_dim_h * mid_dim_h;

	sub_cmpfmatrix(mat_s[1], mat_s[0], mat_tmp_a[0]);

	// -------------------------------
	// S3 := A11 - A21
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_cmpfmatrix_strassen += row_dim_h * mid_dim_h;

	sub_cmpfmatrix(mat_s[2], mat_tmp_a[0], mat_tmp_a[2]);

	// -------------------------------
	// S4 := A12 - S2
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_cmpfmatrix_strassen += row_dim_h * mid_dim_h;

	sub_cmpfmatrix(mat_s[3], mat_tmp_a[1], mat_s[1]);

	// -------------------------------
	// S5 := B12 - B11
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_cmpfmatrix_strassen += mid_dim_h * col_dim_h;

	sub_cmpfmatrix(mat_s[4], mat_tmp_b[1], mat_tmp_b[0]);

	// -------------------------------
	// S6 := B22 - S5
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_cmpfmatrix_strassen += mid_dim_h * col_dim_h;

	sub_cmpfmatrix(mat_s[5], mat_tmp_b[3], mat_s[4]);

	// -------------------------------
	// S7 := B22 - B12
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_cmpfmatrix_strassen += mid_dim_h * col_dim_h;

	sub_cmpfmatrix(mat_s[6], mat_tmp_b[3], mat_tmp_b[1]);

	// -------------------------------
	// S8 := S6 - B21
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_cmpfmatrix_strassen += mid_dim_h * col_dim_h;

	sub_cmpfmatrix(mat_s[7], mat_s[5], mat_tmp_b[2]);

//	printf("s...\n");

	// -------------------------------
	// M1 := S2 * S6
	// -------------------------------
	mul_cmpfmatrix_strassen(mat_m[0], mat_s[1], mat_s[5], min_dim);

	// -------------------------------
	// M2 := A11 * B11
	// -------------------------------
	mul_cmpfmatrix_strassen(mat_m[1], mat_tmp_a[0], mat_tmp_b[0], min_dim);

	// -------------------------------
	// M3 := A12 * B21
	// -------------------------------
	mul_cmpfmatrix_strassen(mat_m[2], mat_tmp_a[1], mat_tmp_b[2], min_dim);

	// -------------------------------
	// M4 := S3 * S7
	// -------------------------------
	mul_cmpfmatrix_strassen(mat_m[3], mat_s[2], mat_s[6], min_dim);

	// -------------------------------
	// M5 := S1 * S5
	// -------------------------------
	mul_cmpfmatrix_strassen(mat_m[4], mat_s[0], mat_s[4], min_dim);

	// -------------------------------
	// M6 := S4 * B22
	// -------------------------------
	mul_cmpfmatrix_strassen(mat_m[5], mat_s[3], mat_tmp_b[3], min_dim);

	// -------------------------------
	// M7 := A22 * S8
	// -------------------------------
	mul_cmpfmatrix_strassen(mat_m[6], mat_tmp_a[3], mat_s[7], min_dim);

//	printf("m...\n");

	// -------------------------------
	// T1 := M1 + M2
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_cmpfmatrix_strassen += row_dim_h * col_dim_h;

	add_cmpfmatrix(mat_t[0], mat_m[0], mat_m[1]);

	// -------------------------------
	// T2 := T1 + M4
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_cmpfmatrix_strassen += row_dim_h * col_dim_h;

	add_cmpfmatrix(mat_t[1], mat_t[0], mat_m[3]);

//	printf("t...\n");

	// -------------------------------
	// C11 := M2 + M3
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_cmpfmatrix_strassen += row_dim_h * col_dim_h;

	add_cmpfmatrix(mat_tmp_c[0], mat_m[1], mat_m[2]);

	// -------------------------------
	// C12 := T1 + M5 + M6
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_cmpfmatrix_strassen += 2 * row_dim_h * col_dim_h;

	add_cmpfmatrix(mat_tmp_c[1], mat_t[0], mat_m[4]);
	add_cmpfmatrix(mat_tmp_c[1], mat_tmp_c[1], mat_m[5]);

	// -------------------------------
	// C21 := T2 - M7
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_cmpfmatrix_strassen += row_dim_h * col_dim_h;

	sub_cmpfmatrix(mat_tmp_c[2], mat_t[1], mat_m[6]);

	// -------------------------------
	// C22 := T2 + M5
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_cmpfmatrix_strassen += row_dim_h * col_dim_h;

	add_cmpfmatrix(mat_tmp_c[3], mat_t[1], mat_m[4]);

//	printf("c...\n");

	// -------------------------------
	// RET := [C11 C12]
	//        [C21 C22]
	// -------------------------------
	for(i = 0; i < 4; i++)
		subst_cmpfmatrix_partial(ret, mat_c_index[i], mat_tmp_c[i], ret_index);

//	printf("set...\n");


	// free
	for(i = 0; i < 4; i++)
	{
		free_cmpfmatrix(mat_s[i]);
		free_cmpfmatrix(mat_tmp_a[i]);
		free_cmpfmatrix(mat_s[i + 4]);
		free_cmpfmatrix(mat_tmp_b[i]);
		free_cmpfmatrix(mat_tmp_c[i]);
	}
	for(i = 0; i < 7; i++)
		free_cmpfmatrix(mat_m[i]);
	
	free_cmpfmatrix(mat_t[0]);
	free_cmpfmatrix(mat_t[1]);
}

// Computattion of Inverse Matrix by using Strassen's Algorithm
void inv_cmpfmatrix_strassen_even(CMPFMatrix ret, CMPFMatrix mat_a, long int min_dim)
{
	unsigned long prec;
	CMPFMatrix mat_p[6], mat_tmp_a[4], mat_p3p6, mat_p6p2, mat_p3p6p2;
	long int dim_h, dim;
	long int ret_index[4], mat_a_index[4][4];
	long int i;

	// Square matrix only
	if((ret->row_dim != ret->col_dim) || (mat_a->row_dim != mat_a->col_dim) || (ret->col_dim != mat_a->row_dim))
	{
		fprintf(stderr, "ERROR: Impossible to get inverse matrix for rectangle matrix!(inv_cmpfmatrix_strassen_even)\n");
		return;
	}

	// initialize
	prec = ret->prec;
	dim = ret->row_dim;
	dim_h = dim / 2;

	// normal matrix multiplication in case of ret_dim <= 4 
	//printf("P0: %ld -> %ld start \n", dim, dim_h);
	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim))
	{
		subst_cmpfmatrix(ret, mat_a);
		//print_cmpfmatrix(mat_a);
		inv_cmpfmatrix(ret);

		return;
	}
	//printf("P0: %ld -> %ld end\n", dim, dim_h);

	mat_p3p6 = init2_cmpfmatrix(dim_h, dim_h, prec);
	mat_p6p2 = init2_cmpfmatrix(dim_h, dim_h, prec);
	mat_p3p6p2 = init2_cmpfmatrix(dim_h, dim_h, prec);

	for(i = 0; i < 6; i++)
		mat_p[i] = init2_cmpfmatrix(dim_h, dim_h, prec);

	for(i = 0; i < 4; i++)
		mat_tmp_a[i] = init2_cmpfmatrix(dim_h, dim_h, prec);

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

	subst_cmpfmatrix_partial(mat_tmp_a[0], ret_index, mat_a, mat_a_index[0]);

	// P1 = tmp_a * tmp_b
	//printf("P1: %ld -> %ld start \n", dim, dim_h);
	inv_cmpfmatrix_strassen_even(mat_p[0], mat_tmp_a[0], min_dim);
	//inv_cmpfmatrix_strassen(mat_p[0], mat_tmp_a[0], min_dim);
	//printf("P1: %ld -> %ld end \n", dim, dim_h);

	// -------------------------------
	// P2 := A21 * P1
	// -------------------------------

	// A21
	mat_a_index[2][0] = dim_h;
	mat_a_index[2][1] = dim;
	mat_a_index[2][2] = 0;
	mat_a_index[2][3] = dim_h;

	subst_cmpfmatrix_partial(mat_tmp_a[2], ret_index, mat_a, mat_a_index[2]);

	// P2 := A21 * P1
	mul_cmpfmatrix_strassen(mat_p[1], mat_tmp_a[2], mat_p[0], min_dim);

	// -------------------------------
	// P3 := P1 * A12
	// -------------------------------

	// A12
	mat_a_index[1][0] = 0;
	mat_a_index[1][1] = dim_h;
	mat_a_index[1][2] = dim_h;
	mat_a_index[1][3] = dim;

	subst_cmpfmatrix_partial(mat_tmp_a[1], ret_index, mat_a, mat_a_index[1]);

	// P3 = P1 * A12
	mul_cmpfmatrix_strassen(mat_p[2], mat_p[0], mat_tmp_a[1], min_dim);

	// -------------------------------
	// P4 := A21 * P3
	// -------------------------------
	mul_cmpfmatrix_strassen(mat_p[3], mat_tmp_a[2], mat_p[2], min_dim);

	// -------------------------------
	// P5 := P4 - A22
	// -------------------------------

	// A22
	mat_a_index[3][0] = dim_h;
	mat_a_index[3][1] = dim;
	mat_a_index[3][2] = dim_h;
	mat_a_index[3][3] = dim;
	subst_cmpfmatrix_partial(mat_tmp_a[3], ret_index, mat_a, mat_a_index[3]);

	// P5 = P4 - A22
	sub_cmpfmatrix(mat_p[4], mat_p[3], mat_tmp_a[3]);

	// -------------------------------
	// P6 := P5^(-1)
	// -------------------------------

	// P6 = tmp_a * tmp_b
	//printf("P6: %ld -> %ld start\n", dim, dim_h);
	inv_cmpfmatrix_strassen_even(mat_p[5], mat_p[4], min_dim);
	//inv_cmpfmatrix_strassen(mat_p[5], mat_p[5], min_dim);
	//printf("P6: %ld -> %ld end\n", dim, dim_h);

	// -------------------------------
	// P3P6 := P3 * P6
	// -------------------------------
	mul_cmpfmatrix_strassen(mat_p3p6, mat_p[2], mat_p[5], min_dim);

	// -------------------------------
	// P6P2 := P6 * P2
	// -------------------------------
	mul_cmpfmatrix_strassen(mat_p6p2, mat_p[5], mat_p[1], min_dim);

	// -------------------------------
	// P3P6P2 := P6 * P2
	// -------------------------------
	mul_cmpfmatrix_strassen(mat_p3p6p2, mat_p3p6, mat_p[1], min_dim);

	// -------------------------------
	// RET11 := P1 - P3 * P6 * P2
	// -------------------------------
	sub_cmpfmatrix_partial(ret, mat_a_index[0], mat_p[0], ret_index, mat_p3p6p2, ret_index);

	// -------------------------------
	// RET12 := P3 * P6
	// -------------------------------
	subst_cmpfmatrix_partial(ret, mat_a_index[1], mat_p3p6, ret_index);

	// -------------------------------
	// RET21 := P6 * P2
	// -------------------------------
	subst_cmpfmatrix_partial(ret, mat_a_index[2], mat_p6p2, ret_index);

	// -------------------------------
	// RET22 := -P6
	// -------------------------------
	neg_cmpfmatrix_partial(ret, mat_a_index[3], mat_p[5], ret_index);

	// free
	free_cmpfmatrix(mat_p3p6);
	free_cmpfmatrix(mat_p6p2);
	free_cmpfmatrix(mat_p3p6p2);

	for(i = 0; i < 6; i++)
		free_cmpfmatrix(mat_p[i]);

	for(i = 0; i < 4; i++)
		free_cmpfmatrix(mat_tmp_a[i]);
}

/* scaling with the absolute maximum element in the row */
/* ret := scaling_diag_mat * org_mat -> ||ret|| \approx 1 */
void left_scaling_cmpfmatrix(CMPFMatrix ret, CMPFMatrix org_mat, CMPFVector scaling_diag_mat, long int *ret_col_index)
{
	unsigned long prec;
	long int i, j, row_dim, col_dim, absmax_col_index;
	mpf_t absmax_val, tmp_abs;
    mpc_t ctmp;

	prec = ret->prec;
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;

	// initialize
	mpf_init2(absmax_val, prec);
	mpf_init2(tmp_abs, prec);
    mpc_init2(ctmp, prec);

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
		//absmax_val = fabs(get_cmpfmatrix_ij(org_mat, i, 0));
		mpc_abs(absmax_val, get_cmpfmatrix_ij(org_mat, i, 0), get_bnc_default_rounding_mode());
		//abs_mpc_t(absmax_val, get_cmpfmatrix_ij(org_mat, i, 0));
		for(j = 1; j < col_dim; j++)
		{
			//tmp_abs = fabs(get_cmpfmatrix_ij(org_mat, i, j));
			mpc_abs(tmp_abs, get_cmpfmatrix_ij(org_mat, i, j), get_bnc_default_rounding_mode());
            //abs_mpc_t(tmp_abs, get_cmpfmatrix_ij(org_mat, i, j));
			if(mpf_cmp(absmax_val, tmp_abs) < 0)
			{
				absmax_col_index = j;
				//absmax_val = tmp_abs;
				mpf_set(absmax_val, tmp_abs);
			}
		}
		
		if(scaling_diag_mat != NULL)
			set_cmpfvector_i_real(scaling_diag_mat, i, absmax_val);
			//set_cmpfvector_i(scaling_diag_mat, i, absmax_val);

		if(ret_col_index != NULL)
			ret_col_index[i] = absmax_col_index;
		
		// scaling
		if(mpf_cmp_ui(absmax_val, 0UL) == 0)
		{
			for(j = 0; j < col_dim; j++)
				set_cmpfmatrix_ij(ret, i, j, get_cmpfmatrix_ij(org_mat, i, j));
		}
		else
		{
			for(j = 0; j < col_dim; j++)
			{
				//tmp_abs = get_cmpfmatrix_ij(org_mat, i, j) / absmax_val;
				mpc_div_fr(ctmp, get_cmpfmatrix_ij(org_mat, i, j), absmax_val, get_bnc_default_rounding_mode());
				set_cmpfmatrix_ij(ret, i, j, ctmp);
			}
		}
	}

	// clear
	mpf_clear(absmax_val);
	mpf_clear(tmp_abs);
    mpc_clear(ctmp);

}

/* scaling with the absolute maximum element in the column */
/* ret := org_mat * scaling_diag_mat -> ||ret|| \approx 1 */
void right_scaling_cmpfmatrix(CMPFMatrix ret, CMPFMatrix org_mat, CMPFVector scaling_diag_mat, long int *ret_row_index)
{
	unsigned long prec;
	long int i, j, row_dim, col_dim, absmax_row_index;
	mpf_t absmax_val, tmp_abs;
    mpc_t tmp;

	prec = ret->prec;
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;

	// initialize
	mpf_init2(absmax_val, prec);
	mpf_init2(tmp_abs, prec);
    mpc_init2(tmp, prec);

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
		//absmax_val = fabs(get_cmpfmatrix_ij(org_mat, 0, i));
		mpc_abs(absmax_val, get_cmpfmatrix_ij(org_mat, 0, i), get_bnc_default_rounding_mode());
		//abs_mpc_t(absmax_val, get_cmpfmatrix_ij(org_mat, 0, i));
		for(j = 1; j < row_dim; j++)
		{
			//tmp_abs = fabs(get_cmpfmatrix_ij(org_mat, j, i));
			mpc_abs(tmp_abs, get_cmpfmatrix_ij(org_mat, j, i), get_bnc_default_rounding_mode());
			//abs_mpc_t(tmp_abs, get_cmpfmatrix_ij(org_mat, j, i));
			if(mpf_cmp(absmax_val, tmp_abs) < 0)
			{
				absmax_row_index = j;
				mpf_set(absmax_val, tmp_abs);
			}
		}

		if(scaling_diag_mat != NULL)
			set_cmpfvector_i_real(scaling_diag_mat, i, absmax_val);
			//set_cmpfvector_i(scaling_diag_mat, i, absmax_val);

		if(ret_row_index != NULL)
			ret_row_index[i] = absmax_row_index;
		
		// scaling
		if(mpf_cmp_ui(absmax_val, 0UL) == 0)
		{
			for(j = 0; j < row_dim; j++)
				set_cmpfmatrix_ij(ret, j, i, get_cmpfmatrix_ij(org_mat, j, i));
		}
		else
		{
			for(j = 0; j < row_dim; j++)
			{
				//tmp_abs = get_cmpfmatrix_ij(org_mat, j, i) / absmax_val;
				mpc_div_fr(tmp, get_cmpfmatrix_ij(org_mat, j, i), absmax_val, get_bnc_default_rounding_mode());
				set_cmpfmatrix_ij(ret, j, i, tmp);
			}
		}
	}

	// clear
    mpc_clear(tmp);
	mpf_clear(absmax_val);
	mpf_clear(tmp_abs);

}

/* multiply square matrix by diagonal matrix      */
/* ret = left_diag_mat^(-left_inv_flat) * org_mat * right_diag_mat^(-right_inv_flat) (left_diag_mat != null, right_diag_mat != null) */
/* ret = left_diag_mat^(-left_inv_flat) * org_mat * right_diag_mat^(-right_inv_flat) (left_diag_mat != null, right_diag_mat != null) */
/* ret = left_diag_mat^(-left_inv_flat) * org_mat                                    (left_diag_mat != null, right_diag_mat == null) */
/* ret =                                  org_mat * right_diag_mat^(-right_inv_flat) (left_diag_mat == null, right_diag_mat != null) */
void mul_cmpfmatrix_mpfdiag(CMPFMatrix ret, CMPFVector left_diag_mat, int left_inv_flag, CMPFMatrix org_mat, CMPFVector right_diag_mat, int right_inv_flag)
{
	unsigned long prec;
	long int i, j, row_dim, col_dim;
	mpc_t tmp;

	prec = ret->prec;
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;

	mpc_init2(tmp, prec);

	/* ret := left_diag_mat * org_mat */
	if(left_diag_mat != NULL)
	{
		// dimension check
		if(row_dim != org_mat->row_dim)
		{
			fprintf(stderr, "ERROR: Dimension of row of ret (ret->row_dim = %ld) is not the same as the org_mat's (org_mat->row_dim = %ld)\n", row_dim, org_mat->row_dim);

			mpc_clear(tmp);
			return;
		}

		// dimension check
		if(row_dim != left_diag_mat->dim)
		{
			fprintf(stderr, "ERROR: Dimension of row of ret (ret->row_dim = %ld) is not the same as the left_diag_mat's (left_diag_mat->dim = %ld)\n", row_dim, left_diag_mat->dim);

			mpc_clear(tmp);
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
					//tmp = get_cmpfvector_i(left_diag_mat, i) * get_cmpfmatrix_ij(org_mat, i, j);
					mpc_mul(tmp, get_cmpfvector_i(left_diag_mat, i), get_cmpfmatrix_ij(org_mat, i, j), get_bnc_default_rounding_mode());
					set_cmpfmatrix_ij(ret, i, j, tmp);
				}
			}
			// left_diag_mat^(-1) * org_mat
			else
			{
				for(j = 0; j < col_dim; j++)
				{
					//tmp = get_cmpfmatrix_ij(org_mat, i, j) / get_dvector_i(left_diag_mat, i);
					mpc_div(tmp, get_cmpfmatrix_ij(org_mat, i, j), get_cmpfvector_i(left_diag_mat, i), get_bnc_default_rounding_mode());
					set_cmpfmatrix_ij(ret, i, j, tmp);
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

			mpc_clear(tmp);
			return;
		}

		if(col_dim != right_diag_mat->dim)
		{
			fprintf(stderr, "Warning: Dimension of column of ret (ret->col_dim = %ld) is not the same as the right_diag_mat's (right_diag_mat->dim = %ld)\n", col_dim, right_diag_mat->dim);

			mpc_clear(tmp);
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
					//tmp = get_cmpfmatrix_ij(ret, i, j) * get_dvector_i(right_diag_mat, j);
					mpc_mul(tmp, get_cmpfmatrix_ij(ret, i, j), get_cmpfvector_i(right_diag_mat, j), get_bnc_default_rounding_mode());
					set_cmpfmatrix_ij(ret, i, j, tmp);
				}
			}
			// org_mat * right_diag_mat^(-1)
			else
			{
				for(j = 0; j < col_dim; j++)
				{
					//tmp = get_cmpfmatrix_ij(ret, i, j) / get_dvector_i(right_diag_mat, j);
					mpc_div(tmp, get_cmpfmatrix_ij(ret, i, j), get_cmpfvector_i(right_diag_mat, j), get_bnc_default_rounding_mode());
					set_cmpfmatrix_ij(ret, i, j, tmp);
				}
			}
		}
	}

	mpc_clear(tmp);
}


#endif // USE_GMP