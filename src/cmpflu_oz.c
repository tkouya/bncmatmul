/********************************************************************************/
/* cmpflu_oz.c:                                                                 */
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
/* stdio.h */
#ifndef _STDIO_H
#include <stdio.h>
#endif

/* math.h */
#ifndef _MATH_H
#include <math.h>
#endif

//#include "bnc.h"

#ifdef USE_IMKL
	#include "mkl.h"
	#include "mkl_cblas.h" // for Intel Math Kernel Library
#endif

#include "bncomp.h"
#include "matmul_strassen.h"

//#include "lu_bench.h"
#include "get_secv.h"

#ifdef USE_GMP

// (4) D22 := L21 * U12
// (5) A22 := A22 - D22
//
//    start_index
// +--+-----+--------+
// |  +-----+--------+
// |  |\ U11|  U12   |
// |  |L11\ |        |
// +  +-----+--------+
// |  |     |        |
// |  |     |        |
// |  | L21 |  A22   |
// |  |     |        |
// +--+-----+--------+
int CMPFLUdecomp_a22_oz(CMPFMatrix a, CMPFMatrix d22, CMPFMatrix l21, CMPFMatrix u12, long int start_index, long int min_dim, int max_num_div)
{
	long int i, j, k, imax, jmax, itmp, dim, index[4], d22_index[4];
	CMPFVector diag_left, diag_right;

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// d22 := l21 * u12
#ifdef USE_OZ_4M
	mul_cmpfmatrix_oz_4m(d22, l21, max_num_div, max_num_div, u12, max_num_div, max_num_div);
#else // USE_OZ_4M
	mul_cmpfmatrix_oz_3m(d22, l21, max_num_div, max_num_div, u12, max_num_div, max_num_div);
#endif // USE_OZ_4M

	// a22 := a22 - d22
	index[0] = imax;
	index[1] = dim;
	index[2] = jmax;
	index[3] = dim;
	d22_index[0] = 0;
	d22_index[1] = d22->row_dim;
	d22_index[2] = 0;
	d22_index[3] = d22->col_dim;

	sub_cmpfmatrix_partial(a, index, a, index, d22, d22_index);

	return 0;
}
#ifdef _OPENMP
int CMPFLUdecomp_a22_oz_omp(CMPFMatrix a, CMPFMatrix d22, CMPFMatrix l21, CMPFMatrix u12, long int start_index, long int min_dim, int max_num_div)
{
	long int i, j, k, imax, jmax, itmp, dim, index[4], d22_index[4];
	CMPFVector diag_left, diag_right;

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// d22 := l21 * u12
	_bncomp_mul_cmpfmatrix_oz_3m(d22, l21, max_num_div, max_num_div, u12, max_num_div, max_num_div);

	// a22 := a22 - d22
	index[0] = imax;
	index[1] = dim;
	index[2] = jmax;
	index[3] = dim;
	d22_index[0] = 0;
	d22_index[1] = d22->row_dim;
	d22_index[2] = 0;
	d22_index[3] = d22->col_dim;

	//sub_cmpfmatrix_partial(a, index, a, index, d22, d22_index);
	_bncomp_sub_cmpfmatrix_partial(a, index, a, index, d22, d22_index);

	return 0;
}
#endif // _OPENMP

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                    (MPF Precision)       */
/*                                                          */
/*                 ver. 0.0 1997.10.24 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CMPFMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
int CMPFLUdecomp_oz(CMPFMatrix a, long int min_dim, int max_num_div)
{
	long int i, row_dim, col_dim, dim;
	CMPFMatrix l21, u12, d22;

	dim = a->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		CMPFLUdecomp(a);
		return 0;
	}

	// dim > min_dim
	for(i = 0; i < dim; i += min_dim)
	{
		// Initialize
		row_dim = dim - (i + min_dim);
		if(row_dim <= 0) 
		{
			// (1) L11 * U11 = A11
			if((dim - i) > 0)
				CMPFLUdecomp_square(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		CMPFLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_dmatrix(a);

/*		l21 = init2_cmpfmatrix(row_dim, min_dim, a->prec + (a->prec / 4));
		u12 = init2_cmpfmatrix(min_dim, col_dim, a->prec + (a->prec / 4));
		d22 = init2_cmpfmatrix(row_dim, col_dim, a->prec + (a->prec / 4));
*/		l21 = init2_cmpfmatrix(row_dim, min_dim, a->prec);
		u12 = init2_cmpfmatrix(min_dim, col_dim, a->prec);
		d22 = init2_cmpfmatrix(row_dim, col_dim, a->prec);

		// (2) Solve L21 * U11 = A21
		CMPFLUdecomp_l21(l21, a, i, min_dim);
		//print_dmatrix(a);
		//printf("L21:\n");
		//print_dmatrix(l21);

		// (3) Solve L11 * U12 = A12
		CMPFLUdecomp_u12(u12, a, i, min_dim);
		//print_dmatrix(a);
		//printf("U12:\n");
		//print_dmatrix(u12);

		//printf("A:\n");
		//print_dmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		CMPFLUdecomp_a22_oz(a, d22, l21, u12, i, min_dim, max_num_div);
		//printf("D22:\n");
		//print_dmatrix(d22);
		//print_dmatrix(a);

		// clear
		free_cmpfmatrix(l21);
		free_cmpfmatrix(u12);
		free_cmpfmatrix(d22);

	}

	return 0;
}
/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                    (MPF Precision)       */
/*                                                          */
/*                 ver. 0.0 1997.10.24 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CMPFMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
int CMPFLUdecomp_ozPM(CMPFMatrix a, long int ch[], long int min_dim, int max_num_div)
{
	long int i, j, row_dim, col_dim, dim, imax, itmp;
	CMPFMatrix l21, u12, d22;
	mpc_t tmp;
    mpf_t axii, abs_tmp;
	int flag = 1;

	// bug case
	if((a->row_dim == 1024) && (min_dim == 32)) flag = 1;

	dim = a->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		CMPFLUdecompPM(a, ch);
		return 0;
	}

	mpc_init2(tmp, a->prec);
    mpf_init2(abs_tmp, a->prec);
	mpf_init2(axii, a->prec);
	for(i = 0; i < dim; i++)
		ch[i] = i;

	// dim > min_dim
	for(i = 0; i < dim; i += min_dim)
	{
		// partial pivoting
		mpc_abs(axii, get_cmpfmatrix_ij(a, i, i), get_bnc_default_rounding_mode());
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			mpc_abs(abs_tmp, get_cmpfmatrix_ij(a, j, i), get_bnc_default_rounding_mode());
			if(mpf_cmp(abs_tmp, axii) > 0)
			{
				imax = j;
				mpf_set(axii, abs_tmp);
			}
		}

		if(mpf_cmp_ui(axii, 0UL) == 0)
		{
			mpc_clear(tmp);
            mpf_clear(abs_tmp);
			mpf_clear(axii);
			fprintf(stderr, "%ld : Error! CMPFLUdecompP!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			//printf("row_swap_cmpfmatrix run! %ld <-> %ld\n", ch[i], ch[imax]);
			row_swap_cmpfmatrix(a, i, imax, 0, a->col_dim); // Fix! 2022-04-11(Mon) T.Kouya
			//row_swap_cmpfmatrix(a, ch[i], ch[imax], 0, a->col_dim);
		}

		// Initialize
		row_dim = dim - (i + min_dim);
		if(row_dim <= 0) 
		{
			//if(flag == 1) printf("row_dim = %d!\n", row_dim);
			// (1) L11 * U11 = A11
			if((dim - i) > 0)
				CMPFLUdecomp_square(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		CMPFLUdecomp_square(a, i, min_dim);

		//if(flag == 1) printf("i = %ld\n", i);
		//print_dmatrix(a);

/*		l21 = init2_cmpfmatrix(row_dim, min_dim, a->prec + (a->prec / 4));
		u12 = init2_cmpfmatrix(min_dim, col_dim, a->prec + (a->prec / 4));
		d22 = init2_cmpfmatrix(row_dim, col_dim, a->prec + (a->prec / 4));
*/		l21 = init2_cmpfmatrix(row_dim, min_dim, a->prec);
		u12 = init2_cmpfmatrix(min_dim, col_dim, a->prec);
		d22 = init2_cmpfmatrix(row_dim, col_dim, a->prec);

		// (2) Solve L21 * U11 = A21
		CMPFLUdecomp_l21(l21, a, i, min_dim);
		//print_dmatrix(a);
		//if(flag == 1) printf("L21:\n");
		//if(flag == 1) { normf_cmpfmatrix(tmp, l21); mpc_out_str(stdout, 10, 0, tmp); printf("\n"); } //print_cmpfmatrix(l21); }

		// (3) Solve L11 * U12 = A12
		CMPFLUdecomp_u12(u12, a, i, min_dim);
		//print_dmatrix(a);
		//if(flag == 1) printf("U12:\n");
		//if(flag == 1) { normf_cmpfmatrix(tmp, u12); mpc_out_str(stdout, 10, 0, tmp); printf("\n"); } //print_cmpfmatrix(u12); }

		//printf("A:\n");
		//print_dmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		CMPFLUdecomp_a22_oz(a, d22, l21, u12, i, min_dim, max_num_div);
		//if(flag == 1) printf("D22:\n");
		//if(flag == 1) { normf_cmpfmatrix(tmp, d22); mpc_out_str(stdout, 10, 0, tmp); printf("\n"); } //print_cmpfmatrix(d22);
		//print_dmatrix(a);

		// clear
		free_cmpfmatrix(l21);
		free_cmpfmatrix(u12);
		free_cmpfmatrix(d22);

	}

	mpc_clear(tmp);
    mpf_clear(abs_tmp);
	mpf_clear(axii);

	return 0;
}

#ifdef _OPENMP
int CMPFLUdecomp_oz_omp(CMPFMatrix a, long int min_dim, int max_num_div)
{
	long int i, row_dim, col_dim, dim;
	CMPFMatrix l21, u12, d22;

	dim = a->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		//CMPFLUdecomp(a);
		CMPFLUdecomp_omp(a);
		return 0;
	}

	// dim > min_dim
	for(i = 0; i < dim; i += min_dim)
	{
		// Initialize
		row_dim = dim - (i + min_dim);
		if(row_dim <= 0) 
		{
			// (1) L11 * U11 = A11
			if((dim - i) > 0)
				CMPFLUdecomp_square_omp(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		//CMPFLUdecomp_square(a, i, min_dim);
		CMPFLUdecomp_square_omp(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_dmatrix(a);

/*		l21 = init2_cmpfmatrix(row_dim, min_dim, a->prec + (a->prec / 4));
		u12 = init2_cmpfmatrix(min_dim, col_dim, a->prec + (a->prec / 4));
		d22 = init2_cmpfmatrix(row_dim, col_dim, a->prec + (a->prec / 4));
*/		l21 = init2_cmpfmatrix(row_dim, min_dim, a->prec);
		u12 = init2_cmpfmatrix(min_dim, col_dim, a->prec);
		d22 = init2_cmpfmatrix(row_dim, col_dim, a->prec);

		// (2) Solve L21 * U11 = A21
		//CMPFLUdecomp_l21(l21, a, i, min_dim);
		CMPFLUdecomp_l21_omp(l21, a, i, min_dim);
		//print_dmatrix(a);
		//printf("L21:\n");
		//print_dmatrix(l21);

		// (3) Solve L11 * U12 = A12
		//CMPFLUdecomp_u12(u12, a, i, min_dim);
		CMPFLUdecomp_u12_omp(u12, a, i, min_dim);
		//print_dmatrix(a);
		//printf("U12:\n");
		//print_dmatrix(u12);

		//printf("A:\n");
		//print_dmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		CMPFLUdecomp_a22_oz_omp(a, d22, l21, u12, i, min_dim, max_num_div);
		//printf("D22:\n");
		//print_dmatrix(d22);
		//print_dmatrix(a);

		// clear
		free_cmpfmatrix(l21);
		free_cmpfmatrix(u12);
		free_cmpfmatrix(d22);

	}

	return 0;
}

int CMPFLUdecomp_ozPM_omp(CMPFMatrix a, long int ch[], long int min_dim, int max_num_div)
{
	long int i, j, row_dim, col_dim, dim, imax, itmp;
	CMPFMatrix l21, u12, d22;
	mpc_t tmp;
    mpf_t axii, abs_tmp;

	dim = a->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		CMPFLUdecompPM(a, ch);
		return 0;
	}

	mpc_init2(tmp, a->prec);
    mpf_init2(abs_tmp, a->prec);
	mpf_init2(axii, a->prec);

	for(i = 0; i < dim; i++)
		ch[i] = i;

	// dim > min_dim
	for(i = 0; i < dim; i += min_dim)
	{
		// partial pivoting
		mpc_abs(axii, get_cmpfmatrix_ij(a, i, i), get_bnc_default_rounding_mode());
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			mpc_abs(abs_tmp, get_cmpfmatrix_ij(a, j, i), get_bnc_default_rounding_mode());
			if(mpf_cmp(abs_tmp, axii) > 0)
			{
				imax = j;
				mpf_set(axii, abs_tmp);
			}
		}

		if(mpf_cmp_ui(axii, 0UL) == 0)
		{
			mpc_clear(tmp);
            mpf_clear(abs_tmp);
			mpf_clear(axii);
			fprintf(stderr, "%ld : Error! CMPFLUdecompPM_omp!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			_bncomp_row_swap_cmpfmatrix(a, i, imax, 0, a->col_dim); // Fix! 2022-04-12(Mon) T.Kouya
			//_bncomp_row_swap_cmpfmatrix(a, ch[i], ch[imax], 0, a->col_dim);
		}

		// Initialize
		row_dim = dim - (i + min_dim);
		if(row_dim <= 0) 
		{
			// (1) L11 * U11 = A11
			if((dim - i) > 0)
				CMPFLUdecomp_square_omp(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		//CMPFLUdecomp_square(a, i, min_dim);
		CMPFLUdecomp_square_omp(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_dmatrix(a);

/*		l21 = init2_cmpfmatrix(row_dim, min_dim, a->prec + (a->prec / 4));
		u12 = init2_cmpfmatrix(min_dim, col_dim, a->prec + (a->prec / 4));
		d22 = init2_cmpfmatrix(row_dim, col_dim, a->prec + (a->prec / 4));
*/		l21 = init2_cmpfmatrix(row_dim, min_dim, a->prec);
		u12 = init2_cmpfmatrix(min_dim, col_dim, a->prec);
		d22 = init2_cmpfmatrix(row_dim, col_dim, a->prec);

		// (2) Solve L21 * U11 = A21
		//CMPFLUdecomp_l21(l21, a, i, min_dim);
		CMPFLUdecomp_l21_omp(l21, a, i, min_dim);
		//print_dmatrix(a);
		//printf("L21:\n");
		//print_dmatrix(l21);

		// (3) Solve L11 * U12 = A12
		//CMPFLUdecomp_u12(u12, a, i, min_dim);
		CMPFLUdecomp_u12_omp(u12, a, i, min_dim);
		//print_dmatrix(a);
		//printf("U12:\n");
		//print_dmatrix(u12);

		//printf("A:\n");
		//print_dmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		CMPFLUdecomp_a22_oz_omp(a, d22, l21, u12, i, min_dim, max_num_div);
		//printf("D22:\n");
		//print_dmatrix(d22);
		//print_dmatrix(a);

		// clear
		free_cmpfmatrix(l21);
		free_cmpfmatrix(u12);
		free_cmpfmatrix(d22);

	}

	mpc_clear(tmp);
	mpf_clear(axii);

	return 0;
}
#endif // _OPENMP
#endif // USE_GMP