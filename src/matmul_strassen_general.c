/********************************************************************************/
/* matmul_strassen_general.c:                                                   */
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
//#include "dlinear.h"

#ifdef USE_IMKL
	#include "mkl.h"
	#include "mkl_cblas.h" // for Intel Math Kernel Library
#endif

#include "matmul_strassen.h"

// count the number of computations
long int num_addsub_mul_dmatrix_strassen;	// addition and subtraction
long int num_mul_mul_dmatrix_strassen;		// multiplication

/* c = a * b */
void mul_dmatrix_simple(DMatrix c, DMatrix a, DMatrix b)
{
	long int i, j, k;
	double tmp;

	/* dimension check */
	if((c->row_dim != a->row_dim) || (c->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: mul_dmatrix_simple\n");
		return;
	}

	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
		{
			tmp = 0.0;
			for(k = 0; k < a->col_dim; k++)
				tmp += get_dmatrix_ij(a, i, k) * get_dmatrix_ij(b, k, j);
			set_dmatrix_ij(c, i, j, tmp);
		}
	}
}

#ifdef __AVX2__ //__AVX2__
// square matrix multiplication (row-major order)
void _bncinner_dmatmul_simple_avx2(double ret[], double mat_a[], double mat_b[], int dim)
{
	int i, j, k, ij_index;
	__m256d ret4, mat_a4, mat_b4;

	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			ij_index = i * dim + j; // row-major order
			//ret[ij_index] = 0.0;
			ret4 = _mm256_setzero_pd();

			for(k = 0; k < dim; k += 4) // AVX2
			{
				//ret[ij_index] += mat_a[i * dim + k] * mat_b[k * dim + j];
				mat_a4 = _mm256_load_pd(&(mat_a[i * dim + k]));
				mat_b4 = _mm256_set_pd(
					mat_b[(k + 3) * dim + j], mat_b[(k + 2) * dim + j], mat_b[(k + 1) * dim + j], mat_b[(k + 0) * dim + j]
					//mat_b[(k + 0) * dim + j], mat_b[(k + 1) * dim + j], mat_b[(k + 2) * dim + j], mat_b[(k + 3) * dim + j]
				);
				ret4 = _mm256_fmadd_pd(mat_a4, mat_b4, ret4);
			}
			ret[ij_index    ] = ret4[0] + ret4[1] + ret4[2] + ret4[3]; 
		}
	}
}
#endif // __AVX2__

#ifdef __AVX512F__ // __AVX512F__
// square matrix multiplication (row-major order)
void _bncinner_dmatmul_simple_avx512(double ret[], double mat_a[], double mat_b[], int dim)
{
	int i, j, k, ij_index;
	//__m256d ret4, mat_a4, mat_b4;
	__m512d ret8, mat_a8, mat_b8;

	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			ij_index = i * dim + j; // row-major order
			//ret[ij_index] = 0.0;
			//ret4 = _mm256_setzero_pd();
			ret8 = _mm512_setzero_pd();

			for(k = 0; k < dim; k += 8) // AVX-512
			{
				//ret[ij_index] += mat_a[i * dim + k] * mat_b[k * dim + j];
				//mat_a4 = _mm256_load_pd(&(mat_a[i * dim + k]));
				//mat_b4 = _mm256_set_pd(
				//	mat_b[(k + 3) * dim + j], mat_b[(k + 2) * dim + j], mat_b[(k + 1) * dim + j], mat_b[(k + 0) * dim + j]
				//);
				//ret4 = _mm256_fmadd_pd(mat_a4, mat_b4, ret4);
				mat_a8 = _mm512_load_pd(&(mat_a[i * dim + k]));
				mat_b8 = _mm512_set_pd(
					mat_b[(k + 7) * dim + j], mat_b[(k + 6) * dim + j], mat_b[(k + 5) * dim + j], mat_b[(k + 4) * dim + j],
					mat_b[(k + 3) * dim + j], mat_b[(k + 2) * dim + j], mat_b[(k + 1) * dim + j], mat_b[(k + 0) * dim + j]
				);
				ret8 = _mm512_fmadd_pd(mat_a8, mat_b8, ret8);
			}
			//ret[ij_index    ] = ret4[0] + ret4[1] + ret4[2] + ret4[3]; 
			ret[ij_index] = ret8[0] + ret8[1] + ret8[2] + ret8[3] + ret8[4] + ret8[5] + ret8[6] + ret8[7]; 
		}
	}
}
#endif // __AVX512F__

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void add_dmatrix_partial(DMatrix ret, long int ret_index[4], DMatrix mat_a, long int mat_a_index[4], DMatrix mat_b, long int mat_b_index[4])
{
	long int i, j, ret_i, ret_j, a_i, a_j, b_i, b_j;
	long int imax, jmax;
	double tmp_val;

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

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

			tmp_val = get_dmatrix_ij(mat_a, a_i, a_j) + get_dmatrix_ij(mat_b, b_i, b_j);
			set_dmatrix_ij(ret, ret_i, ret_j, tmp_val);
		}
	}
}

// partial sub
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void sub_dmatrix_partial(DMatrix ret, long int ret_index[4], DMatrix mat_a, long int mat_a_index[4], DMatrix mat_b, long int mat_b_index[4])
{
	long int i, j, ret_i, ret_j, a_i, a_j, b_i, b_j;
	long int imax, jmax;
	double tmp_val;

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

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

			tmp_val = get_dmatrix_ij(mat_a, a_i, a_j) - get_dmatrix_ij(mat_b, b_i, b_j);
			set_dmatrix_ij(ret, ret_i, ret_j, tmp_val);
		}
	}
}

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_dmatrix_partial(DMatrix ret, long int ret_index[4], DMatrix mat_a, long int mat_a_index[4])
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
			set_dmatrix_ij(ret, ret_i, ret_j, get_dmatrix_ij(mat_a, a_i, a_j));
		}
	}
}

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void neg_dmatrix_partial(DMatrix ret, long int ret_index[4], DMatrix mat_a, long int mat_a_index[4])
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
			set_dmatrix_ij(ret, ret_i, ret_j, -get_dmatrix_ij(mat_a, a_i, a_j));
		}
	}
}

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_dmatrix_partial_checked(DMatrix ret, long int ret_index[4], DMatrix mat_a, long int mat_a_index[4])
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
						set_dmatrix_ij(ret, ret_i, ret_j, get_dmatrix_ij(mat_a, a_i, a_j));
					else
						set_dmatrix_ij(ret, ret_i, ret_j, 0.0); // Padding
					//printf("Warning: ret_index = %d, %d, %d, %d\n", ret_i, ret_j, a_i, a_j);
				}
			}
		}
	}
}

// Padding to 2-powered dimensional matrix
DMatrix init_static_padding_dmatrix_strassen(DMatrix orig_mat)
{
	DMatrix ret = NULL;
	long int ret_row_dim, ret_col_dim, min_dim, i, j;

	if(orig_mat == NULL)
	{
		fprintf(stderr, "Warning: orig_mat is empty!(padding_dmatrix_strassen)\n");
		return NULL;
	}

	ret_row_dim = (long int)pow(2.0, ceil(mylog2((double)orig_mat->row_dim)));
	ret_col_dim = (long int)pow(2.0, ceil(mylog2((double)orig_mat->col_dim)));

	//printf("Padding: row_dim %ld -> %ld, col_dim %ld -> %ld\n", orig_mat->row_dim, ret_row_dim, orig_mat->col_dim, ret_col_dim);

	ret = init_dmatrix(ret_row_dim, ret_col_dim);
	if(ret == NULL)
	{
		fprintf(stderr, "Warning: padding matrix cannot be allocated!(padding_dmatrix_strassen)\n");
		return NULL;
	}

	// ret = [ A | 0 ]
	//       [---+---]
	//       [ 0 | I ]
	for(i = 0; i < orig_mat->row_dim; i++)
	{
		for(j = 0; j < orig_mat->col_dim; j++)
			set_dmatrix_ij(ret, i, j, get_dmatrix_ij(orig_mat, i, j));
	}

/*	min_dim = (ret_row_dim < ret_col_dim) ? ret_row_dim : ret_col_dim;
	for(i = orig_mat->row_dim; i < min_dim; i++)
		set_dmatrix_ij(ret, i, i, 1.0);
*/
	return ret;
}

// Padding to min_dim times dimensional matrix
DMatrix init_static_padding2_dmatrix_strassen(DMatrix orig_mat, long int min_dim)
{
	DMatrix ret = NULL;
	long int ret_row_dim, ret_col_dim, row_col_min_dim, i, j;

	if(orig_mat == NULL)
	{
		fprintf(stderr, "Warning: orig_mat is empty!(padding2_dmatrix_strassen)\n");
		return NULL;
	}

//	ret_row_dim = (long int)pow(2.0, ceil(mylog2((double)orig_mat->row_dim)));
//	ret_col_dim = (long int)pow(2.0, ceil(mylog2((double)orig_mat->col_dim)));
	ret_row_dim = (long int)ceil((double)(orig_mat->row_dim) / (double)min_dim) * min_dim;
	ret_col_dim = (long int)ceil((double)(orig_mat->col_dim) / (double)min_dim) * min_dim;

	//printf("Padding2: row_dim %ld -> %ld, col_dim %ld -> %ld\n", orig_mat->row_dim, ret_row_dim, orig_mat->col_dim, ret_col_dim);

	ret = init_dmatrix(ret_row_dim, ret_col_dim);
	if(ret == NULL)
	{
		fprintf(stderr, "Warning: padding matrix cannot be allocated!(padding2_dmatrix_strassen)\n");
		return NULL;
	}

	// ret = [ A | 0 ]
	//       [---+---]
	//       [ 0 | I ]
	for(i = 0; i < orig_mat->row_dim; i++)
	{
		for(j = 0; j < orig_mat->col_dim; j++)
			set_dmatrix_ij(ret, i, j, get_dmatrix_ij(orig_mat, i, j));
	}

/*	row_col_min_dim = (ret_row_dim < ret_col_dim) ? ret_row_dim : ret_col_dim;
	for(i = orig_mat->row_dim; i < row_col_min_dim; i++)
		set_dmatrix_ij(ret, i, i, 1.0);
*/
	return ret;
}

// Dynamic Padding to even dimensional matrix
DMatrix init_dynamic_padding_dmatrix_strassen(DMatrix orig_mat)
{
	DMatrix ret = NULL;
	long int ret_row_dim, ret_col_dim, min_dim, i, j;

	if(orig_mat == NULL)
	{
		fprintf(stderr, "Warning: orig_mat is empty!(padding_dmatrix_strassen)\n");
		return NULL;
	}

	ret_row_dim = orig_mat->row_dim;
	ret_col_dim = orig_mat->col_dim;

	if((ret_row_dim % 2) == 1)
		ret_row_dim++;
	if((ret_col_dim % 2) == 1)
		ret_col_dim++;

	//printf("Padding: row_dim %ld -> %ld, col_dim %ld -> %ld\n", orig_mat->row_dim, ret_row_dim, orig_mat->col_dim, ret_col_dim);

	ret = init_dmatrix(ret_row_dim, ret_col_dim);
	if(ret == NULL)
	{
		fprintf(stderr, "Warning: padding matrix cannot be allocated!(padding_dmatrix_strassen)\n");
		return NULL;
	}

	// ret = [ A | 0 ]
	//       [---+---]
	//       [ 0 | 1 ]
	for(i = 0; i < orig_mat->row_dim; i++)
	{
		for(j = 0; j < orig_mat->col_dim; j++)
			set_dmatrix_ij(ret, i, j, get_dmatrix_ij(orig_mat, i, j));
	}

	min_dim = (ret_row_dim < ret_col_dim) ? ret_row_dim : ret_col_dim;
	for(i = orig_mat->row_dim; i < min_dim; i++)
		set_dmatrix_ij(ret, i, i, 1.0);

	return ret;
}

// Strassen's Algorithm with static padding
void mul_dmatrix_strassen_odd_padding(DMatrix ret, DMatrix mat_a, DMatrix mat_b, long int min_dim)
{
	long int tmp_ret_index[4], ret_index[4];
	DMatrix tmp_ret, tmp_mat_a, tmp_mat_b;

	// padding
#ifdef STATIC_PADDING
	tmp_ret = init_static_padding_dmatrix_strassen(ret);
	tmp_mat_a = init_static_padding_dmatrix_strassen(mat_a);
	tmp_mat_b = init_static_padding_dmatrix_strassen(mat_b);
#elif STATIC_PADDING2
	tmp_ret = init_static_padding2_dmatrix_strassen(ret, min_dim);
	tmp_mat_a = init_static_padding2_dmatrix_strassen(mat_a, min_dim);
	tmp_mat_b = init_static_padding2_dmatrix_strassen(mat_b, min_dim);
#else
	tmp_ret = init_dynamic_padding_dmatrix_strassen(ret);
	tmp_mat_a = init_dynamic_padding_dmatrix_strassen(mat_a);
	tmp_mat_b = init_dynamic_padding_dmatrix_strassen(mat_b);
#endif

	// strassen
#ifdef USE_WINOGRAD
	mul_dmatrix_winograd_even(tmp_ret, tmp_mat_a, tmp_mat_b, min_dim);
#else
	mul_dmatrix_strassen_even(tmp_ret, tmp_mat_a, tmp_mat_b, min_dim);
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

//	subst_dmatrix_partial(ret, ret_index, tmp_ret, tmp_ret_index);
	subst_dmatrix_partial_checked(ret, ret_index, tmp_ret, tmp_ret_index);

	// free
	free_dmatrix(tmp_ret);
	free_dmatrix(tmp_mat_a);
	free_dmatrix(tmp_mat_b);
}

// clear counters
void reset_num_mul_dmatrix_strassen(void)
{
	num_addsub_mul_dmatrix_strassen = 0;
	num_mul_mul_dmatrix_strassen = 0;
}

// print counters
void print_num_mul_dmatrix_strassen(long int *num_addsub, long int *num_mul)
{
	printf("num_addsub_mul_dmatrix_strassen: %ld\n", num_addsub_mul_dmatrix_strassen);
	printf("num_mul_mul_dmatrix_strassen   : %ld\n", num_mul_mul_dmatrix_strassen);

	if(num_addsub != NULL)
		*num_addsub = num_addsub_mul_dmatrix_strassen;
	if(num_mul != NULL)
		*num_mul = num_mul_mul_dmatrix_strassen;
}


// Fit dimension to be multiple of min_dim
void mul_dmatrix_strassen(DMatrix ret, DMatrix mat_a, DMatrix mat_b, long int min_dim)
{
	long int row_k, col_k, mid_row_k, mid_col_k;
	long int row_dim, col_dim, mid_dim;
	DVector diag_left, diag_right;

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_dmatrix_strassen)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	// normal matrix multiplication in case of ret_dim <= 4 
	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	{
#ifdef USE_IMKL
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row_dim, col_dim, mid_dim, 1.0, mat_a->element, mat_a->row_dim, mat_b->element, mat_b->row_dim, 0.0, ret->element, ret->row_dim);
#else
		mul_dmatrix(ret, mat_a, mat_b);
#endif

		// counting the number of arithmetic
		num_addsub_mul_dmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		num_mul_mul_dmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		return;
	}

//	diag_left = init_dvector(row_dim);
//	diag_right = init_dvector(col_dim);

//	left_scaling_dmatrix(mat_a, mat_a, diag_left, NULL);
//	right_scaling_dmatrix(mat_b, mat_b, diag_right, NULL);

	// dynamic peeling in case of odd dim
	// [ A11   a12 ] [ B11   b12 ] = [ A11*B11 + a12 * b21^T   A11*b12 + a12 * b22    ]
	// [ a21^T a22 ] [ b21^T b22 ]   [ a21^T*B11 + a22 * b21^T a21^T * b12 + a22 * b22]
	if((ret->row_dim % 2 == 1) || (ret->col_dim % 2 == 1) || (mid_dim % 2 == 1))
	{
#ifdef PEELING_ONLY
		mul_dmatrix_strassen_odd_peeling(ret, mat_a, mat_b, min_dim);
#elif PADDING_ONLY
		mul_dmatrix_strassen_odd_padding(ret, mat_a, mat_b, min_dim);
#else
		row_k = (long int)floor(mylog2((double)ret->row_dim));
		col_k = (long int)floor(mylog2((double)ret->col_dim));

		mid_row_k = (long int)pow(2.0, row_k - 1) * 3;
		mid_col_k = (long int)pow(2.0, col_k - 1) * 3;

		//printf("2^%ld <= %ld <= 2^%ld\n", row_k, ret->row_dim, row_k + 1);

		// dynamic peeling
		//if(ret->row_dim < mid_row_k)
		if((ret->row_dim % min_dim) < (min_dim / 2))
			mul_dmatrix_strassen_odd_peeling(ret, mat_a, mat_b, min_dim);
		// padding
		else
			mul_dmatrix_strassen_odd_padding(ret, mat_a, mat_b, min_dim);
#endif
	}
	// normal strassen algorithm in case of even dim
	else
	{
		//printf("%d is even -> ", ret->row_dim);
#ifdef USE_WINOGRAD
		mul_dmatrix_winograd_even(ret, mat_a, mat_b, min_dim);
#else
		mul_dmatrix_strassen_even(ret, mat_a, mat_b, min_dim);
#endif
		//printf("end\n");
	}

//	mul_dmatrix_ddiag_mat(ret, diag_left, 0, ret, diag_right, 0);

//	free_dvector(diag_left);
//	free_dvector(diag_right);
}

// Strassen's Algorithm
void mul_dmatrix_strassen_odd_peeling(DMatrix ret, DMatrix mat_a, DMatrix mat_b, long int min_dim)
{
	long int i, j, row_dim, row_dim_h, col_dim, col_dim_h, mid_dim, mid_dim_h;
	DMatrix mat_a11, mat_b11, mat_c11, mat_tmp;
	DVector vec_a12, vec_a21, vec_b12, vec_b21, vec_c12, vec_c21, vec_tmp12, vec_tmp21;
	double a22, b22, c22, tmp;

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_dmatrix_strassen_odd_peeling)\n", mat_a->col_dim, mat_b->row_dim);
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
		col_dim_h = col_dim;


	// normal matrix multiplication in case of ret_dim <= 4 
	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	{
#ifdef USE_IMKL
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row_dim, col_dim, mid_dim, 1.0, mat_a->element, mat_a->row_dim, mat_b->element, mat_b->row_dim, 0.0, ret->element, ret->row_dim);
#else
		mul_dmatrix(ret, mat_a, mat_b);
#endif
		return;
	}

	// Initialize
	mat_a11 = init_dmatrix(row_dim_h, mid_dim_h);
	mat_b11 = init_dmatrix(mid_dim_h, col_dim_h);
	mat_c11 = init_dmatrix(row_dim_h, col_dim_h);
	mat_tmp = init_dmatrix(row_dim_h, col_dim_h);

	vec_a12 = init_dvector(row_dim_h);
	vec_a21 = init_dvector(mid_dim_h);
	vec_b12 = init_dvector(mid_dim_h);
	vec_b21 = init_dvector(col_dim_h);
	vec_c12 = init_dvector(row_dim_h);
	vec_c21 = init_dvector(col_dim_h);
	vec_tmp12 = init_dvector(row_dim_h);
	vec_tmp21 = init_dvector(col_dim_h);

	// set matrix elements to mat_a11, mat_b11, vec_a12, vec_b12, vec_a21, vec_b21, a22, b22
	for(i = 0; i < row_dim_h; i++)
	{
		set_dvector_i(vec_a12, i, get_dmatrix_ij(mat_a, i, mat_a->col_dim - 1));
		for(j = 0; j < mid_dim_h; j++)
			set_dmatrix_ij(mat_a11, i, j, get_dmatrix_ij(mat_a, i, j));
	}

	for(i = 0; i < mid_dim_h; i++)
	{
		set_dvector_i(vec_b12, i, get_dmatrix_ij(mat_b, i, mat_b->col_dim - 1));
		set_dvector_i(vec_a21, i, get_dmatrix_ij(mat_a, mat_a->row_dim - 1, i));
		for(j = 0; j < col_dim_h; j++)
			set_dmatrix_ij(mat_b11, i, j, get_dmatrix_ij(mat_b, i, j));
	}

	for(i = 0; i < col_dim_h; i++)
		set_dvector_i(vec_b21, i, get_dmatrix_ij(mat_b, mat_b->row_dim - 1, i));

	a22 = get_dmatrix_ij(mat_a, mat_a->row_dim - 1, mat_a->col_dim - 1);
	b22 = get_dmatrix_ij(mat_b, mat_b->row_dim - 1, mat_b->col_dim - 1);

	// dynamic peeling in case of odd dim
	// [ A11   a12 ] [ B11   b12 ] = [ A11*B11 + a12 * b21^T   A11*b12 + a12 * b22     ]
	// [ a21^T a22 ] [ b21^T b22 ]   [ a21^T*B11 + a22 * b21^T a21^T * b12 + a22 * b22 ]

	// A11 * B11
	mul_dmatrix_strassen(mat_c11, mat_a11, mat_b11, min_dim);

	// a12 * b21^T
	for(i = 0; i < row_dim_h; i++)
	{
		for(j = 0; j < col_dim_h; j++)
		{
;			set_dmatrix_ij(mat_tmp, i, j, get_dvector_i(vec_a12, i) * get_dvector_i(vec_b21, j));
		}
	}
	
	add_dmatrix(mat_c11, mat_c11, mat_tmp);
	for(i = 0; i < row_dim_h; i++)
		for(j = 0; j < col_dim_h; j++)
			set_dmatrix_ij(ret, i, j, get_dmatrix_ij(mat_c11, i, j));

	// A11 * b12 + b22 * a12
	mul_dmatrix_dvec(vec_c12, mat_a11, vec_b12);
	cmul_dvector(vec_tmp12, b22, vec_a12);
	add_dvector(vec_c12, vec_c12, vec_tmp12);
	for(i = 0; i < row_dim_h; i++)
		set_dmatrix_ij(ret, i, ret->col_dim - 1, get_dvector_i(vec_c12, i));

	// a21^T*B11 + a22 * b21^T
	mul_dmatrixt_dvec(vec_c21, mat_b11, vec_a21);
	cmul_dvector(vec_tmp21, a22, vec_b21);
	add_dvector(vec_c21, vec_c21, vec_tmp21);
	for(i = 0; i < col_dim_h; i++)
		set_dmatrix_ij(ret, ret->row_dim - 1, i, get_dvector_i(vec_c21, i));

	// a21^T * b12 + a22 * b22
	c22 = ip_dvector(vec_a21, vec_b12);
	c22 += a22 * b22;
	set_dmatrix_ij(ret, ret->row_dim - 1, ret->col_dim - 1, c22);

	// free
	free_dmatrix(mat_a11);
	free_dmatrix(mat_b11);
	free_dmatrix(mat_c11);
	free_dmatrix(mat_tmp);

	free_dvector(vec_a12);
	free_dvector(vec_a21);
	free_dvector(vec_b12);
	free_dvector(vec_b21);
	free_dvector(vec_c12);
	free_dvector(vec_c21);
	free_dvector(vec_tmp12);
	free_dvector(vec_tmp21);
}


// Strassen's Algorithm
void mul_dmatrix_strassen_even(DMatrix ret, DMatrix mat_a, DMatrix mat_b, long int min_dim)
{
//	long int min_dim = 4; // = 2^2
	DMatrix mat_p[7], mat_tmp_a[7], mat_tmp_b[7], mat_tmp_c[4];
	long int row_dim_h, row_dim, col_dim_h, col_dim, mid_dim, mid_dim_h;
	long int ret_index[4], mat_tmp_a_index[4], mat_tmp_b_index[4], mat_ar_index[7][4], mat_al_index[7][4], mat_br_index[7][4], mat_bl_index[7][4], mat_c_index[4][4];
	long int i;

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_dmatrix_strassen_even)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	row_dim_h = row_dim / 2;
	col_dim_h = col_dim / 2;
	mid_dim_h = mid_dim / 2;

	// normal matrix multiplication in case of ret_dim <= 4 
//	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	if((ret->row_dim <= min_dim) || (ret->col_dim <= min_dim) || (mid_dim <= min_dim))
	{
#ifdef USE_IMKL
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row_dim, col_dim, mid_dim, 1.0, mat_a->element, mat_a->row_dim, mat_b->element, mat_b->row_dim, 0.0, ret->element, ret->row_dim);
#else
		mul_dmatrix(ret, mat_a, mat_b);
#endif
		// counting the number of arithmetic
		num_addsub_mul_dmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		num_mul_mul_dmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		return;
	}

	for(i = 0; i < 7; i++)
	{
		mat_p[i] = init_dmatrix(row_dim_h, col_dim_h);
		mat_tmp_a[i] = init_dmatrix(row_dim_h, mid_dim_h);
		mat_tmp_b[i] = init_dmatrix(mid_dim_h, col_dim_h);
	}
	for(i = 0; i < 4; i++)
		mat_tmp_c[i] = init_dmatrix(row_dim_h, col_dim_h);

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
	//printf("P1a:\n");

	// counting the number of arithmetic
	num_addsub_mul_dmatrix_strassen += row_dim_h * mid_dim_h;

//	add_dmatrix_partial(mat_tmp_a[0], ret_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);
	add_dmatrix_partial(mat_tmp_a[0], mat_tmp_a_index, mat_a, mat_ar_index[0], mat_a, mat_al_index[0]);

	// B11 + B22
	mat_br_index[0][0] = 0;
	mat_br_index[0][1] = mid_dim_h;
	mat_br_index[0][2] = 0;
	mat_br_index[0][3] = col_dim_h;

	mat_bl_index[0][0] = mid_dim_h;
	mat_bl_index[0][1] = mid_dim;
	mat_bl_index[0][2] = col_dim_h;
	mat_bl_index[0][3] = col_dim;
	//printf("P1b:\n");

	// counting the number of arithmetic
	num_addsub_mul_dmatrix_strassen += mid_dim_h * col_dim_h;

//	add_dmatrix_partial(mat_tmp_b[0], ret_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);
	add_dmatrix_partial(mat_tmp_b[0], mat_tmp_b_index, mat_b, mat_br_index[0], mat_b, mat_bl_index[0]);

	// P1 = tmp_a * tmp_b
	//printf("P1:\n");
	mul_dmatrix_strassen(mat_p[0], mat_tmp_a[0], mat_tmp_b[0], min_dim);

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
	num_addsub_mul_dmatrix_strassen += row_dim_h * mid_dim_h;

//	add_dmatrix_partial(mat_tmp_a[1], ret_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);
	add_dmatrix_partial(mat_tmp_a[1], mat_tmp_a_index, mat_a, mat_ar_index[1], mat_a, mat_al_index[1]);

	// B11
	mat_br_index[1][0] = 0;
	mat_br_index[1][1] = mid_dim_h;
	mat_br_index[1][2] = 0;
	mat_br_index[1][3] = col_dim_h;
//	subst_dmatrix_partial(mat_tmp_b[1], ret_index, mat_b, mat_br_index[1]);
	subst_dmatrix_partial(mat_tmp_b[1], mat_tmp_b_index, mat_b, mat_br_index[1]);

	// P2 = tmp_a * tmp_b
	//printf("P2:\n");
	mul_dmatrix_strassen(mat_p[1], mat_tmp_a[1], mat_tmp_b[1], min_dim);

	// -------------------------------
	// P3 := A11 * (B12 - B22)
	// -------------------------------

	// A11
	mat_ar_index[2][0] = 0;
	mat_ar_index[2][1] = row_dim_h;
	mat_ar_index[2][2] = 0;
	mat_ar_index[2][3] = mid_dim_h;
//	subst_dmatrix_partial(mat_tmp_a[2], ret_index, mat_a, mat_ar_index[2]);
	subst_dmatrix_partial(mat_tmp_a[2], mat_tmp_a_index, mat_a, mat_ar_index[2]);

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
	num_addsub_mul_dmatrix_strassen += mid_dim_h * col_dim_h;

//	sub_dmatrix_partial(mat_tmp_b[2], ret_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);
	sub_dmatrix_partial(mat_tmp_b[2], mat_tmp_b_index, mat_b, mat_br_index[2], mat_b, mat_bl_index[2]);

	// P3 = tmp_a * tmp_b
	//printf("P3:\n");
	mul_dmatrix_strassen(mat_p[2], mat_tmp_a[2], mat_tmp_b[2], min_dim);

	// -------------------------------
	// P4 := A22 * (B21 - B11)
	// -------------------------------

	// A22
	mat_ar_index[3][0] = row_dim_h;
	mat_ar_index[3][1] = row_dim;
	mat_ar_index[3][2] = mid_dim_h;
	mat_ar_index[3][3] = mid_dim;
//	subst_dmatrix_partial(mat_tmp_a[3], ret_index, mat_a, mat_ar_index[3]);
	subst_dmatrix_partial(mat_tmp_a[3], mat_tmp_a_index, mat_a, mat_ar_index[3]);

	// B21 - B11
	mat_br_index[3][0] = mid_dim_h;
	mat_br_index[3][1] = mid_dim;
	mat_br_index[3][2] = 0;
	mat_br_index[3][3] = col_dim_h;

	mat_bl_index[3][0] = 0;
	mat_bl_index[3][1] = mid_dim_h;
	mat_bl_index[3][2] = 0;
	mat_bl_index[3][3] = col_dim_h;

	// counting the number of arithmetic
	num_addsub_mul_dmatrix_strassen += mid_dim_h * col_dim_h;

//	sub_dmatrix_partial(mat_tmp_b[3], ret_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);
	sub_dmatrix_partial(mat_tmp_b[3], mat_tmp_b_index, mat_b, mat_br_index[3], mat_b, mat_bl_index[3]);

	// P4 = tmp_a * tmp_b
	//printf("P4:\n");
	mul_dmatrix_strassen(mat_p[3], mat_tmp_a[3], mat_tmp_b[3], min_dim);

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
	num_addsub_mul_dmatrix_strassen += row_dim_h * mid_dim_h;

//	add_dmatrix_partial(mat_tmp_a[4], ret_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);
	add_dmatrix_partial(mat_tmp_a[4], mat_tmp_a_index, mat_a, mat_ar_index[4], mat_a, mat_al_index[4]);

	// B22
	mat_br_index[4][0] = mid_dim_h;
	mat_br_index[4][1] = mid_dim;
	mat_br_index[4][2] = col_dim_h;
	mat_br_index[4][3] = col_dim;
//	subst_dmatrix_partial(mat_tmp_b[4], ret_index, mat_b, mat_br_index[4]);
	subst_dmatrix_partial(mat_tmp_b[4], mat_tmp_b_index, mat_b, mat_br_index[4]);

	// P5 = tmp_a * tmp_b
	//printf("P5:\n");
	mul_dmatrix_strassen(mat_p[4], mat_tmp_a[4], mat_tmp_b[4], min_dim);

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
	num_addsub_mul_dmatrix_strassen += row_dim_h * mid_dim_h;

//	sub_dmatrix_partial(mat_tmp_a[5], ret_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);
	sub_dmatrix_partial(mat_tmp_a[5], mat_tmp_a_index, mat_a, mat_ar_index[5], mat_a, mat_al_index[5]);

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
	num_addsub_mul_dmatrix_strassen += mid_dim_h * col_dim_h;

//	add_dmatrix_partial(mat_tmp_b[5], ret_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);
	add_dmatrix_partial(mat_tmp_b[5], mat_tmp_b_index, mat_b, mat_br_index[5], mat_b, mat_bl_index[5]);

	// P6 = tmp_a * tmp_b
	//printf("P6:\n");
	mul_dmatrix_strassen(mat_p[5], mat_tmp_a[5], mat_tmp_b[5], min_dim);

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
	num_addsub_mul_dmatrix_strassen += row_dim_h * mid_dim_h;

//	sub_dmatrix_partial(mat_tmp_a[6], ret_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);
	sub_dmatrix_partial(mat_tmp_a[6], mat_tmp_a_index, mat_a, mat_ar_index[6], mat_a, mat_al_index[6]);

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
	num_addsub_mul_dmatrix_strassen += mid_dim_h * col_dim_h;

//	add_dmatrix_partial(mat_tmp_b[6], ret_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);
	add_dmatrix_partial(mat_tmp_b[6], mat_tmp_b_index, mat_b, mat_br_index[6], mat_b, mat_bl_index[6]);

	// P7 = tmp_a * tmp_b
	//printf("P7:\n");
	mul_dmatrix_strassen(mat_p[6], mat_tmp_a[6], mat_tmp_b[6], min_dim);

	// -------------------------------
	// C11 := P1 + P4 - P5 + P7
	// -------------------------------

	// counting the number of arithmetic
	num_addsub_mul_dmatrix_strassen += 3 * row_dim_h * col_dim_h;

	add_dmatrix(mat_tmp_c[0], mat_p[0], mat_p[3]);
	sub_dmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[4]);
	add_dmatrix(mat_tmp_c[0], mat_tmp_c[0], mat_p[6]);
	mat_c_index[0][0] = 0;
	mat_c_index[0][1] = row_dim_h;
	mat_c_index[0][2] = 0;
	mat_c_index[0][3] = col_dim_h;
	subst_dmatrix_partial(ret, mat_c_index[0], mat_tmp_c[0], ret_index);

	// -------------------------------
	// C12 := P3 + P5
	// -------------------------------

	// counting the number of arithmetic
	num_addsub_mul_dmatrix_strassen += row_dim_h * col_dim_h;

	add_dmatrix(mat_tmp_c[1], mat_p[2], mat_p[4]);
	mat_c_index[1][0] = 0;
	mat_c_index[1][1] = row_dim_h;
	mat_c_index[1][2] = col_dim_h;
	mat_c_index[1][3] = col_dim;
	subst_dmatrix_partial(ret, mat_c_index[1], mat_tmp_c[1], ret_index);

	// -------------------------------
	// C21 := P2 + P4
	// -------------------------------

	// counting the number of arithmetic
	num_addsub_mul_dmatrix_strassen += row_dim_h * col_dim_h;

	add_dmatrix(mat_tmp_c[2], mat_p[1], mat_p[3]);
	mat_c_index[2][0] = row_dim_h;
	mat_c_index[2][1] = row_dim;
	mat_c_index[2][2] = 0;
	mat_c_index[2][3] = col_dim_h;
	subst_dmatrix_partial(ret, mat_c_index[2], mat_tmp_c[2], ret_index);

	// -------------------------------
	// C22 := P1 + P3 - P2 + P6
	// -------------------------------

	// counting the number of arithmetic
	num_addsub_mul_dmatrix_strassen += 3 * row_dim_h * col_dim_h;

	add_dmatrix(mat_tmp_c[3], mat_p[0], mat_p[2]);
	sub_dmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[1]);
	add_dmatrix(mat_tmp_c[3], mat_tmp_c[3], mat_p[5]);
	mat_c_index[3][0] = row_dim_h;
	mat_c_index[3][1] = row_dim;
	mat_c_index[3][2] = col_dim_h;
	mat_c_index[3][3] = col_dim;
	subst_dmatrix_partial(ret, mat_c_index[3], mat_tmp_c[3], ret_index);

	// free
	for(i = 0; i < 7; i++)
	{
		free_dmatrix(mat_p[i]);
		free_dmatrix(mat_tmp_a[i]);
		free_dmatrix(mat_tmp_b[i]);
	}
	for(i = 0; i < 4; i++)
		free_dmatrix(mat_tmp_c[i]);
}

// Winograd Variant of Strassen's Algorithm
void mul_dmatrix_winograd_even(DMatrix ret, DMatrix mat_a, DMatrix mat_b, long int min_dim)
{
//	long int min_dim = 4; // = 2^2
	DMatrix mat_s[8], mat_m[7], mat_t[2], mat_tmp_a[4], mat_tmp_b[4], mat_tmp_c[4];
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
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_dmatrix_winograd_even)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	row_dim_h = row_dim / 2;
	col_dim_h = col_dim / 2;
	mid_dim_h = mid_dim / 2;

	// normal matrix multiplication in case of ret_dim <= 4 
//	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	if((ret->row_dim <= min_dim) || (ret->col_dim <= min_dim) || (mid_dim <= min_dim))
	{
#ifdef USE_IMKL
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row_dim, col_dim, mid_dim, 1.0, mat_a->element, mat_a->row_dim, mat_b->element, mat_b->row_dim, 0.0, ret->element, ret->row_dim);
#else
		mul_dmatrix(ret, mat_a, mat_b);
#endif
		// counting the number of arithmetic
		num_addsub_mul_dmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);
		num_mul_mul_dmatrix_strassen += mat_a->col_dim * (mat_a->row_dim * mat_b->col_dim);

		return;
	}

	for(i = 0; i < 4; i++)
	{
		mat_s[i] = init_dmatrix(row_dim_h, mid_dim_h);
		mat_tmp_a[i] = init_dmatrix(row_dim_h, mid_dim_h);

		mat_s[i+4] = init_dmatrix(mid_dim_h, col_dim_h);
		mat_tmp_b[i] = init_dmatrix(mid_dim_h, col_dim_h);

		mat_tmp_c[i] = init_dmatrix(row_dim_h, col_dim_h);
	}
	for(i = 0; i < 7; i++)
		mat_m[i] = init_dmatrix(row_dim_h, col_dim_h);

	mat_t[0] = init_dmatrix(row_dim_h, col_dim_h);
	mat_t[1] = init_dmatrix(row_dim_h, col_dim_h);

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
		subst_dmatrix_partial(mat_tmp_a[i], mat_tmp_a_index, mat_a, mat_a_index[i]);
		subst_dmatrix_partial(mat_tmp_b[i], mat_tmp_b_index, mat_b, mat_b_index[i]);
	}
//	printf("subst a, b...\n");

	// -------------------------------
	// S1 := A21 + A22
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dmatrix_strassen += row_dim_h * mid_dim_h;

	add_dmatrix(mat_s[0], mat_tmp_a[2], mat_tmp_a[3]);

	// -------------------------------
	// S2 := S1 - A11
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dmatrix_strassen += row_dim_h * mid_dim_h;

	sub_dmatrix(mat_s[1], mat_s[0], mat_tmp_a[0]);

	// -------------------------------
	// S3 := A11 - A21
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dmatrix_strassen += row_dim_h * mid_dim_h;

	sub_dmatrix(mat_s[2], mat_tmp_a[0], mat_tmp_a[2]);

	// -------------------------------
	// S4 := A12 - S2
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dmatrix_strassen += row_dim_h * mid_dim_h;

	sub_dmatrix(mat_s[3], mat_tmp_a[1], mat_s[1]);

	// -------------------------------
	// S5 := B12 - B11
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dmatrix_strassen += mid_dim_h * col_dim_h;

	sub_dmatrix(mat_s[4], mat_tmp_b[1], mat_tmp_b[0]);

	// -------------------------------
	// S6 := B22 - S5
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dmatrix_strassen += mid_dim_h * col_dim_h;

	sub_dmatrix(mat_s[5], mat_tmp_b[3], mat_s[4]);

	// -------------------------------
	// S7 := B22 - B12
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dmatrix_strassen += mid_dim_h * col_dim_h;

	sub_dmatrix(mat_s[6], mat_tmp_b[3], mat_tmp_b[1]);

	// -------------------------------
	// S8 := S6 - B21
	//--------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dmatrix_strassen += mid_dim_h * col_dim_h;

	sub_dmatrix(mat_s[7], mat_s[5], mat_tmp_b[2]);

//	printf("s...\n");

	// -------------------------------
	// M1 := S2 * S6
	// -------------------------------
	mul_dmatrix_strassen(mat_m[0], mat_s[1], mat_s[5], min_dim);

	// -------------------------------
	// M2 := A11 * B11
	// -------------------------------
	mul_dmatrix_strassen(mat_m[1], mat_tmp_a[0], mat_tmp_b[0], min_dim);

	// -------------------------------
	// M3 := A12 * B21
	// -------------------------------
	mul_dmatrix_strassen(mat_m[2], mat_tmp_a[1], mat_tmp_b[2], min_dim);

	// -------------------------------
	// M4 := S3 * S7
	// -------------------------------
	mul_dmatrix_strassen(mat_m[3], mat_s[2], mat_s[6], min_dim);

	// -------------------------------
	// M5 := S1 * S5
	// -------------------------------
	mul_dmatrix_strassen(mat_m[4], mat_s[0], mat_s[4], min_dim);

	// -------------------------------
	// M6 := S4 * B22
	// -------------------------------
	mul_dmatrix_strassen(mat_m[5], mat_s[3], mat_tmp_b[3], min_dim);

	// -------------------------------
	// M7 := A22 * S8
	// -------------------------------
	mul_dmatrix_strassen(mat_m[6], mat_tmp_a[3], mat_s[7], min_dim);

//	printf("m...\n");

	// -------------------------------
	// T1 := M1 + M2
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dmatrix_strassen += row_dim_h * col_dim_h;

	add_dmatrix(mat_t[0], mat_m[0], mat_m[1]);

	// -------------------------------
	// T2 := T1 + M4
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dmatrix_strassen += row_dim_h * col_dim_h;

	add_dmatrix(mat_t[1], mat_t[0], mat_m[3]);

//	printf("t...\n");

	// -------------------------------
	// C11 := M2 + M3
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dmatrix_strassen += row_dim_h * col_dim_h;

	add_dmatrix(mat_tmp_c[0], mat_m[1], mat_m[2]);

	// -------------------------------
	// C12 := T1 + M5 + M6
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dmatrix_strassen += 2 * row_dim_h * col_dim_h;

	add_dmatrix(mat_tmp_c[1], mat_t[0], mat_m[4]);
	add_dmatrix(mat_tmp_c[1], mat_tmp_c[1], mat_m[5]);

	// -------------------------------
	// C21 := T2 - M7
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dmatrix_strassen += row_dim_h * col_dim_h;

	sub_dmatrix(mat_tmp_c[2], mat_t[1], mat_m[6]);

	// -------------------------------
	// C22 := T2 + M5
	// -------------------------------
	// counting the number of arithmetic
	num_addsub_mul_dmatrix_strassen += row_dim_h * col_dim_h;

	add_dmatrix(mat_tmp_c[3], mat_t[1], mat_m[4]);

//	printf("c...\n");

	// -------------------------------
	// RET := [C11 C12]
	//        [C21 C22]
	// -------------------------------
	for(i = 0; i < 4; i++)
		subst_dmatrix_partial(ret, mat_c_index[i], mat_tmp_c[i], ret_index);

//	printf("set...\n");


	// free
	for(i = 0; i < 4; i++)
	{
		free_dmatrix(mat_s[i]);
		free_dmatrix(mat_tmp_a[i]);
		free_dmatrix(mat_s[i + 4]);
		free_dmatrix(mat_tmp_b[i]);
		free_dmatrix(mat_tmp_c[i]);
	}
	for(i = 0; i < 7; i++)
		free_dmatrix(mat_m[i]);
	
	free_dmatrix(mat_t[0]);
	free_dmatrix(mat_t[1]);
}

// Block matrix multiplicaiton
void mul_dmatrix_block(DMatrix ret, DMatrix mat_a, DMatrix mat_b, long int min_dim)
{
	int row_padding_flag = 0, col_padding_flag = 0, mid_padding_flag = 0;
	long i, j, k, row_dim, col_dim, mid_dim;
	long int num_div_row, num_div_col, num_div_mid;
	long int mat_a_index[4], small_mat_a_index[4], mat_b_index[4], small_mat_b_index[4], ret_index[4], small_ret_index[4];
	DMatrix *small_ret, *small_mat_a, *small_mat_b, small_tmp_mat;

	// initialize
	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = mat_a->col_dim;
	if(mid_dim != mat_b->row_dim)
	{
		fprintf(stderr, "ERROR: mat_a's col_dim %ld does not just fit mat_b's row_dim %ld!(mul_dmatrix_block)\n", mat_a->col_dim, mat_b->row_dim);
		return;
	}

	// normal matrix multiplication in case of ret_dim <= 4 
	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim) && (mid_dim <= min_dim))
	{
#ifdef USE_IMKL
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, row_dim, col_dim, mid_dim, 1.0, mat_a->element, mat_a->row_dim, mat_b->element, mat_b->row_dim, 0.0, ret->element, ret->row_dim);
//		mul_dmatrix(ret, mat_a, mat_b);
#else
		mul_dmatrix(ret, mat_a, mat_b);
#endif
		return;
	}

	// Number of division of matrix
	num_div_row = (ret->row_dim) / min_dim;
	if((num_div_row % min_dim) >= 1)
	{
		row_padding_flag = 1;
		num_div_row++;
	}

	num_div_mid = mid_dim / min_dim;
	if((num_div_row % min_dim) >= 1)
	{
		mid_padding_flag = 1;
		num_div_mid++;
	}

	num_div_col = (ret->col_dim) / min_dim;
	if((num_div_col % min_dim) >= 1)
	{
		col_padding_flag = 1;
		num_div_col++;
	}

	// initialize
	small_ret = (DMatrix *)calloc(sizeof(DMatrix), num_div_col);
	small_mat_a = (DMatrix *)calloc(sizeof(DMatrix), num_div_mid);
	small_mat_b = (DMatrix *)calloc(sizeof(DMatrix), num_div_mid);
	for(i = 0; i < num_div_col; i++)
		small_ret[i] = init_dmatrix(min_dim, min_dim);
	for(i = 0; i < num_div_mid; i++)
	{
		small_mat_a[i] = init_dmatrix(min_dim, min_dim);
		small_mat_b[i] = init_dmatrix(min_dim, min_dim);
	}
	small_tmp_mat = init_dmatrix(min_dim, min_dim);

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
			//subst_dmatrix_partial(small_mat_a[j], small_mat_a_index, mat_a, mat_a_index);
			subst_dmatrix_partial_checked(small_mat_a[j], small_mat_a_index, mat_a, mat_a_index);
		}

		for(j = 0; j < num_div_col; j++)
		{

			set0_dmatrix(small_ret[j]);
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
				//subst_dmatrix_partial(small_mat_b[k], small_mat_b_index, mat_b, mat_b_index);
				subst_dmatrix_partial_checked(small_mat_b[k], small_mat_b_index, mat_b, mat_b_index);

				// ret[j] += small_mat_a[i][k] * small_mat_b[k][j];
#ifdef USE_IMKL
				cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, min_dim, min_dim, min_dim, 1.0, small_mat_a[k]->element, min_dim, small_mat_b[k]->element, min_dim, 0.0, small_tmp_mat->element, min_dim);
//				mul_dmatrix(small_tmp_mat, small_mat_a[k], small_mat_b[k]);
#else
				mul_dmatrix(small_tmp_mat, small_mat_a[k], small_mat_b[k]);
#endif
				add_dmatrix(small_ret[j], small_ret[j], small_tmp_mat);
			}
			ret_index[0] = i * min_dim;
			ret_index[1] = (i + 1) * min_dim;
			ret_index[2] = j * min_dim;
			ret_index[3] = (j + 1) * min_dim;
			small_ret_index[0] = 0;
			small_ret_index[1] = min_dim;
			small_ret_index[2] = 0;
			small_ret_index[3] = min_dim;
			//subst_dmatrix_partial(ret, ret_index, small_ret[j], small_ret_index);
			subst_dmatrix_partial_checked(ret, ret_index, small_ret[j], small_ret_index);
		}
	}

	// free
	free_dmatrix(small_tmp_mat);
	for(i = 0; i < num_div_col; i++)
		free_dmatrix(small_ret[i]);

	for(i = 0; i < num_div_mid; i++)
	{
		free_dmatrix(small_mat_a[i]);
		free_dmatrix(small_mat_b[i]);
	}
	free(small_ret);
	free(small_mat_a);
	free(small_mat_b);
}

// Computattion of Inverse Matrix by using Strassen's Algorithm
void inv_dmatrix_strassen_even(DMatrix ret, DMatrix mat_a, long int min_dim)
{
//	long int min_dim = 4; // = 2^2
	DMatrix mat_p[6], mat_tmp_a[4], mat_p3p6, mat_p6p2, mat_p3p6p2;
	long int dim_h, dim;
	long int ret_index[4], mat_a_index[4][4];
	long int i;

	// Square matrix only
	if((ret->row_dim != ret->col_dim) || (mat_a->row_dim != mat_a->col_dim) || (ret->col_dim != mat_a->row_dim))
	{
		fprintf(stderr, "ERROR: Impossible to get inverse matrix for rectangle matrix!(inv_dmatrix_strassen_even)\n");
		return;
	}

	// initialize
	dim = ret->row_dim;
	dim_h = dim / 2;

	// normal matrix multiplication in case of ret_dim <= 4 
	//printf("P0: %ld -> %ld start \n", dim, dim_h);
	if((ret->row_dim <= min_dim) && (ret->col_dim <= min_dim))
	{
		subst_dmatrix(ret, mat_a);
		//print_dmatrix(mat_a);
		inv_dmatrix(ret);

		return;
	}
	//printf("P0: %ld -> %ld end\n", dim, dim_h);

	mat_p3p6 = init_dmatrix(dim_h, dim_h);
	mat_p6p2 = init_dmatrix(dim_h, dim_h);
	mat_p3p6p2 = init_dmatrix(dim_h, dim_h);

	for(i = 0; i < 6; i++)
		mat_p[i] = init_dmatrix(dim_h, dim_h);

	for(i = 0; i < 4; i++)
		mat_tmp_a[i] = init_dmatrix(dim_h, dim_h);

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

	subst_dmatrix_partial(mat_tmp_a[0], ret_index, mat_a, mat_a_index[0]);

	// P1 = tmp_a * tmp_b
	//printf("P1: %ld -> %ld start \n", dim, dim_h);
	inv_dmatrix_strassen_even(mat_p[0], mat_tmp_a[0], min_dim);
	//inv_dmatrix_strassen(mat_p[0], mat_tmp_a[0], min_dim);
	//printf("P1: %ld -> %ld end \n", dim, dim_h);

	// -------------------------------
	// P2 := A21 * P1
	// -------------------------------

	// A21
	mat_a_index[2][0] = dim_h;
	mat_a_index[2][1] = dim;
	mat_a_index[2][2] = 0;
	mat_a_index[2][3] = dim_h;

	subst_dmatrix_partial(mat_tmp_a[2], ret_index, mat_a, mat_a_index[2]);

	// P2 := A21 * P1
	mul_dmatrix_strassen(mat_p[1], mat_tmp_a[2], mat_p[0], min_dim);

	// -------------------------------
	// P3 := P1 * A12
	// -------------------------------

	// A12
	mat_a_index[1][0] = 0;
	mat_a_index[1][1] = dim_h;
	mat_a_index[1][2] = dim_h;
	mat_a_index[1][3] = dim;

	subst_dmatrix_partial(mat_tmp_a[1], ret_index, mat_a, mat_a_index[1]);

	// P3 = P1 * A12
	mul_dmatrix_strassen(mat_p[2], mat_p[0], mat_tmp_a[1], min_dim);

	// -------------------------------
	// P4 := A21 * P3
	// -------------------------------
	mul_dmatrix_strassen(mat_p[3], mat_tmp_a[2], mat_p[2], min_dim);

	// -------------------------------
	// P5 := P4 - A22
	// -------------------------------

	// A22
	mat_a_index[3][0] = dim_h;
	mat_a_index[3][1] = dim;
	mat_a_index[3][2] = dim_h;
	mat_a_index[3][3] = dim;
	subst_dmatrix_partial(mat_tmp_a[3], ret_index, mat_a, mat_a_index[3]);

	// P5 = P4 - A22
	sub_dmatrix(mat_p[4], mat_p[3], mat_tmp_a[3]);

	// -------------------------------
	// P6 := P5^(-1)
	// -------------------------------

	// P6 = tmp_a * tmp_b
	//printf("P6: %ld -> %ld start\n", dim, dim_h);
	inv_dmatrix_strassen_even(mat_p[5], mat_p[4], min_dim);
	//inv_dmatrix_strassen(mat_p[5], mat_p[5], min_dim);
	//printf("P6: %ld -> %ld end\n", dim, dim_h);

	// -------------------------------
	// P3P6 := P3 * P6
	// -------------------------------
	mul_dmatrix_strassen(mat_p3p6, mat_p[2], mat_p[5], min_dim);

	// -------------------------------
	// P6P2 := P6 * P2
	// -------------------------------
	mul_dmatrix_strassen(mat_p6p2, mat_p[5], mat_p[1], min_dim);

	// -------------------------------
	// P3P6P2 := P6 * P2
	// -------------------------------
	mul_dmatrix_strassen(mat_p3p6p2, mat_p3p6, mat_p[1], min_dim);

	// -------------------------------
	// RET11 := P1 - P3 * P6 * P2
	// -------------------------------
	sub_dmatrix_partial(ret, mat_a_index[0], mat_p[0], ret_index, mat_p3p6p2, ret_index);

	// -------------------------------
	// RET12 := P3 * P6
	// -------------------------------
	subst_dmatrix_partial(ret, mat_a_index[1], mat_p3p6, ret_index);

	// -------------------------------
	// RET21 := P6 * P2
	// -------------------------------
	subst_dmatrix_partial(ret, mat_a_index[2], mat_p6p2, ret_index);

	// -------------------------------
	// RET22 := -P6
	// -------------------------------
	neg_dmatrix_partial(ret, mat_a_index[3], mat_p[5], ret_index);

	// free
	free_dmatrix(mat_p3p6);
	free_dmatrix(mat_p6p2);
	free_dmatrix(mat_p3p6p2);

	for(i = 0; i < 6; i++)
		free_dmatrix(mat_p[i]);

	for(i = 0; i < 4; i++)
		free_dmatrix(mat_tmp_a[i]);
}

/* scaling with the absolute maximum element in the row */
/* ret := scaling_diag_mat * org_mat -> ||ret|| \approx 1 */
void left_scaling_dmatrix(DMatrix ret, DMatrix org_mat, DVector scaling_diag_mat, long int *ret_col_index)
{
	long int i, j, row_dim, col_dim, absmax_col_index;
	double absmax_val, tmp_abs;

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
		absmax_val = fabs(get_dmatrix_ij(org_mat, i, 0));
		for(j = 1; j < col_dim; j++)
		{
			tmp_abs = fabs(get_dmatrix_ij(org_mat, i, j));
			if(absmax_val < tmp_abs)
			{
				absmax_col_index = j;
				absmax_val = tmp_abs;
			}
		}
		
		if(scaling_diag_mat != NULL)
			set_dvector_i(scaling_diag_mat, i, absmax_val);
		if(ret_col_index != NULL)
			ret_col_index[i] = absmax_col_index;
		
		// scaling
		if(absmax_val == 0.0)
		{
			for(j = 0; j < col_dim; j++)
				set_dmatrix_ij(ret, i, j, get_dmatrix_ij(org_mat, i, j));
		}
		else
		{
			for(j = 0; j < col_dim; j++)
			{
				tmp_abs = get_dmatrix_ij(org_mat, i, j) / absmax_val;
				set_dmatrix_ij(ret, i, j, tmp_abs);
			}
		}
	}
}

/* scaling with the absolute maximum element in the column */
/* ret := org_mat * scaling_diag_mat -> ||ret|| \approx 1 */
void right_scaling_dmatrix(DMatrix ret, DMatrix org_mat, DVector scaling_diag_mat, long int *ret_row_index)
{
	long int i, j, row_dim, col_dim, absmax_row_index;
	double absmax_val, tmp_abs;

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
		if(col_dim != org_mat->col_dim)
		{
			fprintf(stderr, "Warning: Dimension of column of ret (ret->col_dim = %ld) is not the same as the org_mat's (org_mat->col_dim = %ld)\n", col_dim, org_mat->col_dim);
			if(col_dim > org_mat->col_dim)
				col_dim = org_mat->col_dim;
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
		absmax_val = fabs(get_dmatrix_ij(org_mat, 0, i));
		for(j = 1; j < row_dim; j++)
		{
			tmp_abs = fabs(get_dmatrix_ij(org_mat, j, i));
			if(absmax_val < tmp_abs)
			{
				absmax_row_index = j;
				absmax_val = tmp_abs;
			}
		}
		
		if(scaling_diag_mat != NULL)
			set_dvector_i(scaling_diag_mat, i, absmax_val);
		if(ret_row_index != NULL)
			ret_row_index[i] = absmax_row_index;
		
		// scaling
		if(absmax_val == 0.0)
		{
			for(j = 0; j < row_dim; j++)
				set_dmatrix_ij(ret, j, i, get_dmatrix_ij(org_mat, j, i));
		}
		else
		{
			for(j = 0; j < row_dim; j++)
			{
				tmp_abs = get_dmatrix_ij(org_mat, j, i) / absmax_val;
				set_dmatrix_ij(ret, j, i, tmp_abs);
			}
		}
	}
}

/* multiply square matrix by diagonal matrix      */
/* ret = left_diag_mat^(-left_inv_flat) * org_mat * right_diag_mat^(-right_inv_flat) (left_diag_mat != null, right_diag_mat != null) */
/* ret = left_diag_mat^(-left_inv_flat) * org_mat * right_diag_mat^(-right_inv_flat) (left_diag_mat != null, right_diag_mat != null) */
/* ret = left_diag_mat^(-left_inv_flat) * org_mat                                    (left_diag_mat != null, right_diag_mat == null) */
/* ret =                                  org_mat * right_diag_mat^(-right_inv_flat) (left_diag_mat == null, right_diag_mat != null) */
void mul_dmatrix_ddiag(DMatrix ret, DVector left_diag_mat, int left_inv_flag, DMatrix org_mat, DVector right_diag_mat, int right_inv_flag)
{
	long int i, j, row_dim, col_dim;
	double tmp;

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
					tmp = get_dvector_i(left_diag_mat, i) * get_dmatrix_ij(org_mat, i, j);
					set_dmatrix_ij(ret, i, j, tmp);
				}
			}
			// left_diag_mat^(-1) * org_mat
			else
			{
				for(j = 0; j < col_dim; j++)
				{
					tmp = get_dmatrix_ij(org_mat, i, j) / get_dvector_i(left_diag_mat, i);
					set_dmatrix_ij(ret, i, j, tmp);
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
					tmp = get_dmatrix_ij(ret, i, j) * get_dvector_i(right_diag_mat, j);
					set_dmatrix_ij(ret, i, j, tmp);
				}
			}
			// org_mat * right_diag_mat^(-1)
			else
			{
				for(j = 0; j < col_dim; j++)
				{
					tmp = get_dmatrix_ij(ret, i, j) / get_dvector_i(right_diag_mat, j);
					set_dmatrix_ij(ret, i, j, tmp);
				}
			}
		}
	}
}

// The following main function is for debugging
#ifdef DEBUG

#include "get_secv.h"

int main(int argc, char *argv[])
{
	long int i, j, dim;
	DMatrix dc, da, db, dc_normal, dc_imkl, dc_block;
	DVector ddiag_left, ddiag_right;
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

/* Double precision */

	dc = init_dmatrix(dim, dim);
	dc_normal = init_dmatrix(dim, dim);
	dc_imkl = init_dmatrix(dim, dim);
	dc_block = init_dmatrix(dim, dim);
	da = init_dmatrix(dim, dim);
	db = init_dmatrix(dim, dim);

	ddiag_left = init_dvector(dim);
	ddiag_right = init_dvector(dim);

	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			set_dmatrix_ij(da, i, j, (double)rand() * pow(-1.0, rand()));
			set_dmatrix_ij(db, i, j, 1.0 / (double)rand() * pow(-1.0, rand()));
		}
	}

#ifdef USE_IMKL
	stime = get_secv();
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim, dim, dim, 1.0, da->element, dim, db->element, dim, 0.0, dc_imkl->element, dim);
	etime[2] = get_secv() - stime;
#endif

	// normal matrix mul
	stime = get_secv();
	mul_dmatrix(dc_normal, da, db);
	etime[1] = get_secv() - stime;

	// Strassen 

	left_scaling_dmatrix(da, da, ddiag_left, NULL);
	right_scaling_dmatrix(db, db, ddiag_right, NULL);

//	print_dvector(ddiag_left);
//	print_dvector(ddiag_right);

	stime = get_secv();
	mul_dmatrix_strassen(dc, da, db, 4);
//	mul_dmatrix_strassen(dc, da, db, 64);
//	mul_dmatrix_strassen(dc, da, db, 128);
//	mul_dmatrix_strassen(dc, da, db, 256);
//	mul_dmatrix_strassen(dc, da, db, 512);
//	mul_dmatrix_strassen(dc, da, db, 1024);
	etime[0] = get_secv() - stime;

	mul_dmatrix_ddiag(dc, ddiag_left, 0, dc, ddiag_right, 0);

	mul_dmatrix_ddiag(da, ddiag_left, 0, da, NULL, 0);
	mul_dmatrix_ddiag(db, NULL, 0, db, ddiag_right, 0);

	// Block
	stime = get_secv();
	mul_dmatrix_block(dc_block, da, db, 128);
//	mul_dmatrix_block(dc_block, da, db, 1024);
	etime[3] = get_secv() - stime;

	// difference
#ifdef USE_IMKL
	sub_dmatrix(da, dc_imkl, dc_normal);
	reldiff[2] = normi_dmatrix(da) / normi_dmatrix(dc_normal);
#endif

	sub_dmatrix(da, dc, dc_normal);
	reldiff[0] = normi_dmatrix(da) / normi_dmatrix(dc_normal);
	sub_dmatrix(da, dc_block, dc_normal);
	reldiff[3] = normi_dmatrix(da) / normi_dmatrix(dc_normal);

	// print
	printf("dim        : %ld\n", dim);
	printf("normal     : %f\n", etime[1]);
#ifdef USE_IMKL
	printf("Intel IMKL : %f\n", etime[2]);
#endif
	printf("block      : %f\n", etime[3]);
	printf("strassen   : %f\n", etime[0]);
#ifdef USE_IMKL
	printf("||reldiff_imkl||  : %e\n", reldiff[2]);
#endif
	printf("||reldiff_blockl||: %e\n", reldiff[3]);
	printf("||reldiff_strass||: %e\n", reldiff[0]);

/* Inverse */

	frank_dmatrix(da, dim);
	frank_dmatrix(db, dim);
//	lotkin_dmatrix(da, dim);
//	lotkin_dmatrix(db, dim);

	// normal inverse
	printf("Normal Inverse...\n");
	stime = get_secv();
	inv_dmatrix(da);
	etime[1] = get_secv() - stime;

//	print_dmatrix(da);

	// Strassen 
	printf("Strassen Inverse...\n");
	stime = get_secv();
	inv_dmatrix_strassen_even(dc, db, 32);
	etime[0] = get_secv() - stime;

//	print_dmatrix(dc);

	// LU
	printf("LU decompsition...\n");
	stime = get_secv();
	DLUdecomp(db);
	etime[2] = get_secv() - stime;

	// difference
	sub_dmatrix(db, da, dc);
	reldiff[0] = normi_dmatrix(db) / normi_dmatrix(da);

	// print
	printf("dim        : %ld\n", dim);
	printf("LU         : %f\n", etime[2]);
	printf("strassen   : %f\n", etime[0]);
	printf("normal     : %f\n", etime[1]);
	printf("||a^(-1)|| strass : %e\n", normi_dmatrix(dc));
	printf("||a^(-1)||        : %e\n", normi_dmatrix(da));
	printf("||reldiff_strass||: %e\n", reldiff[0]);

	free_dmatrix(dc);
	free_dmatrix(dc_normal);
	free_dmatrix(dc_imkl);
	free_dmatrix(dc_block);
	free_dmatrix(da);
	free_dmatrix(db);

	free_dvector(ddiag_left);
	free_dvector(ddiag_right);

	return 0;
}
#endif // DEBUG
