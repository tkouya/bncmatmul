/********************************************************************************/
/* ddlu_oz.c:                                                                   */
/* Copyright (C) 2022 Tomonori Kouya                                            */
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
//#include "ddlinear.h"
#include "bncomp.h"
#include "matmul_strassen.h"

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
//
int DDLUdecomp_a22_oz(DDMatrix a, DDMatrix d22, DDMatrix l21, DDMatrix u12, long int start_index, long int min_dim, int max_num_div)
{
	long int i, j, k, imax, jmax, itmp, dim, index[4], d22_index[4];

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// d22 := l21 * u12
	mul_ddmatrix_oz(d22, l21, max_num_div, u12, max_num_div);

	// a22 := a22 - d22
	index[0] = imax;
	index[1] = dim;
	index[2] = jmax;
	index[3] = dim;
	d22_index[0] = 0;
	d22_index[1] = d22->row_dim;
	d22_index[2] = 0;
	d22_index[3] = d22->col_dim;

	sub_ddmatrix_partial(a, index, a, index, d22, d22_index);

	return 0;
}

#ifdef _OPENMP
int DDLUdecomp_a22_oz_omp(DDMatrix a, DDMatrix d22, DDMatrix l21, DDMatrix u12, long int start_index, long int min_dim, int max_num_div)
{
	long int i, j, k, imax, jmax, itmp, dim, index[4], d22_index[4];

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// d22 := l21 * u12
	_bncomp_mul_ddmatrix_oz(d22, l21, max_num_div, u12, max_num_div);

	// a22 := a22 - d22
	index[0] = imax;
	index[1] = dim;
	index[2] = jmax;
	index[3] = dim;
	d22_index[0] = 0;
	d22_index[1] = d22->row_dim;
	d22_index[2] = 0;
	d22_index[3] = d22->col_dim;

	//sub_ddmatrix_partial(a, index, a, index, d22, d22_index);
	_bncomp_sub_ddmatrix_partial(a, index, a, index, d22, d22_index);

	return 0;
}
#endif // _OPENMP

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                 (Quadraple Precision)    */
/*                                                          */
/*                 ver. 0.0 2015-02-25 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int DDLUdecomp_oz(DDMatrix a, long int min_dim, int max_num_div)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DDMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, row_dim, col_dim, dim;
	DDMatrix l21, u12, d22;

	dim = a->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		DDLUdecomp(a);
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
				DDLUdecomp_square(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		DDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_ddmatrix(a);

		l21 = init_ddmatrix(row_dim, min_dim);
		u12 = init_ddmatrix(min_dim, col_dim);
		d22 = init_ddmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		DDLUdecomp_l21(l21, a, i, min_dim);
		//print_ddmatrix(a);
		//printf("L21:\n");
		//print_ddmatrix(l21);

		// (3) Solve L11 * U12 = A12
		DDLUdecomp_u12(u12, a, i, min_dim);
		//print_ddmatrix(a);
		//printf("U12:\n");
		//print_ddmatrix(u12);

		//printf("A:\n");
		//print_ddmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		DDLUdecomp_a22_oz(a, d22, l21, u12, i, min_dim, max_num_div);
		//printf("D22:\n");
		//print_ddmatrix(d22);
		//print_ddmatrix(a);

		// clear
		free_ddmatrix(l21);
		free_ddmatrix(u12);
		free_ddmatrix(d22);

	}

	return 0;
}


/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                 (Quadraple Precision)    */
/*                                                          */
/*                 ver. 0.0 2015-02-25 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int DDLUdecomp_ozPM(DDMatrix a, long int ch[], long int min_dim, int max_num_div)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DDMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, row_dim, col_dim, dim, imax, itmp;
	DDMatrix l21, u12, d22;
	static double dtmp[DDSIZE], dtmp1[DDSIZE], dmaxii[DDSIZE];

	dim = a->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		DDLUdecompPM(a, ch);
		return 0;
	}

	for(i = 0; i < dim; i++)
		ch[i] = i;

	// dim > min_dim
	for(i = 0; i < dim; i += min_dim)
	{
		// partial pivoting
		rdd_abs(dmaxii, get_ddmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rdd_abs(dtmp, get_ddmatrix_ij(a, j, i));
			if(rdd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rdd_set(dmaxii, dtmp);
			}
		}

		if(rdd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! DDLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			row_swap_ddmatrix(a, i, imax, 0, a->col_dim);
		}

		// Initialize
		row_dim = dim - (i + min_dim);
		if(row_dim <= 0) 
		{
			// (1) L11 * U11 = A11
			if((dim - i) > 0)
				DDLUdecomp_square(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		DDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_ddmatrix(a);

		l21 = init_ddmatrix(row_dim, min_dim);
		u12 = init_ddmatrix(min_dim, col_dim);
		d22 = init_ddmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		DDLUdecomp_l21(l21, a, i, min_dim);
		//print_ddmatrix(a);
		//printf("L21:\n");
		//print_ddmatrix(l21);

		// (3) Solve L11 * U12 = A12
		DDLUdecomp_u12(u12, a, i, min_dim);
		//print_ddmatrix(a);
		//printf("U12:\n");
		//print_ddmatrix(u12);

		//printf("A:\n");
		//print_ddmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		DDLUdecomp_a22_oz(a, d22, l21, u12, i, min_dim, max_num_div);
		//printf("D22:\n");
		//print_ddmatrix(d22);
		//print_ddmatrix(a);

		// clear
		free_ddmatrix(l21);
		free_ddmatrix(u12);
		free_ddmatrix(d22);

	}

	return 0;
}

#ifdef _OPENMP
/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                 (Quadraple Precision)    */
/*                                                          */
/*                 ver. 0.0 2015-02-25 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int DDLUdecomp_oz_omp(DDMatrix a, long int min_dim, int max_num_div)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DDMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, row_dim, col_dim, dim;
	DDMatrix l21, u12, d22;

	dim = a->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		DDLUdecomp_omp(a);
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
				DDLUdecomp_square_omp(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		DDLUdecomp_square_omp(a, i, min_dim);
		//DDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_ddmatrix(a);

		l21 = init_ddmatrix(row_dim, min_dim);
		u12 = init_ddmatrix(min_dim, col_dim);
		d22 = init_ddmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		DDLUdecomp_l21_omp(l21, a, i, min_dim);
		//DDLUdecomp_l21(l21, a, i, min_dim);
		//print_ddmatrix(a);
		//printf("L21:\n");
		//print_ddmatrix(l21);

		// (3) Solve L11 * U12 = A12
		DDLUdecomp_u12_omp(u12, a, i, min_dim);
		//DDLUdecomp_u12(u12, a, i, min_dim);
		//print_ddmatrix(a);
		//printf("U12:\n");
		//print_ddmatrix(u12);

		//printf("A:\n");
		//print_ddmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		DDLUdecomp_a22_oz_omp(a, d22, l21, u12, i, min_dim, max_num_div);
		//DDLUdecomp_a22(a, d22, l21, u12, i, min_dim);
		//printf("D22:\n");
		//print_ddmatrix(d22);
		//print_ddmatrix(a);

		// clear
		free_ddmatrix(l21);
		free_ddmatrix(u12);
		free_ddmatrix(d22);

	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                 (Quadraple Precision)    */
/*                                                          */
/*                 ver. 0.0 2015-02-25 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int DDLUdecomp_ozPM_omp(DDMatrix a, long int ch[], long int min_dim, int max_num_div)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DDMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, row_dim, col_dim, dim, imax, itmp;
	DDMatrix l21, u12, d22;
	double dtmp[DDSIZE], dmaxii[DDSIZE];

	dim = a->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		DDLUdecompPM_omp(a, ch);
		return 0;
	}

	for(i = 0; i < dim; i++)
		ch[i] = i;

	// dim > min_dim
	for(i = 0; i < dim; i += min_dim)
	{
		// partial pivoting
		rdd_abs(dmaxii, get_ddmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rdd_abs(dtmp, get_ddmatrix_ij(a, j, i));
			if(rdd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rdd_set(dmaxii, dtmp);
			}
		}

		if(rdd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! DDLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			_bncomp_row_swap_ddmatrix(a, i, imax, 0, a->col_dim);
		}

		// Initialize
		row_dim = dim - (i + min_dim);
		if(row_dim <= 0) 
		{
			// (1) L11 * U11 = A11
			if((dim - i) > 0)
				DDLUdecomp_square_omp(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		DDLUdecomp_square_omp(a, i, min_dim);
		//DDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_ddmatrix(a);

		l21 = init_ddmatrix(row_dim, min_dim);
		u12 = init_ddmatrix(min_dim, col_dim);
		d22 = init_ddmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		DDLUdecomp_l21_omp(l21, a, i, min_dim);
		//DDLUdecomp_l21(l21, a, i, min_dim);
		//print_ddmatrix(a);
		//printf("L21:\n");
		//print_ddmatrix(l21);

		// (3) Solve L11 * U12 = A12
		DDLUdecomp_u12_omp(u12, a, i, min_dim);
		//DDLUdecomp_u12(u12, a, i, min_dim);
		//print_ddmatrix(a);
		//printf("U12:\n");
		//print_ddmatrix(u12);

		//printf("A:\n");
		//print_ddmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		//DDLUdecomp_a22_omp(a, d22, l21, u12, i, min_dim);
		DDLUdecomp_a22_oz_omp(a, d22, l21, u12, i, min_dim, max_num_div);
		//printf("D22:\n");
		//print_ddmatrix(d22);
		//print_ddmatrix(a);

		// clear
		free_ddmatrix(l21);
		free_ddmatrix(u12);
		free_ddmatrix(d22);

	}

	return 0;
}
#endif // _OPENMP
