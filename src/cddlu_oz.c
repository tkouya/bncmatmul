/********************************************************************************/
/* cddlu_oz.c:                                                                  */
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
int CDDLUdecomp_a22_oz(CDDMatrix a, CDDMatrix d22, CDDMatrix l21, CDDMatrix u12, long int start_index, long int min_dim, int max_num_div)
{
	long int i, j, k, imax, jmax, itmp, dim, index[4], d22_index[4];

	dim = a->re->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// d22 := l21 * u12
	mul_cddmatrix_oz(d22, l21, max_num_div, max_num_div, u12, max_num_div, max_num_div);

	// a22 := a22 - d22
	index[0] = imax;
	index[1] = dim;
	index[2] = jmax;
	index[3] = dim;
	d22_index[0] = 0;
	d22_index[1] = d22->re->row_dim;
	d22_index[2] = 0;
	d22_index[3] = d22->re->col_dim;

	sub_cddmatrix_partial(a, index, a, index, d22, d22_index);

	return 0;
}

#ifdef _OPENMP
int CDDLUdecomp_a22_oz_omp(CDDMatrix a, CDDMatrix d22, CDDMatrix l21, CDDMatrix u12, long int start_index, long int min_dim, int max_num_div)
{
	long int i, j, k, imax, jmax, itmp, dim, index[4], d22_index[4];

	dim = a->re->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// d22 := l21 * u12
	//_bncomp_mul_cddmatrix_oz(d22, l21, max_num_div, u12, max_num_div);
	_bncomp_mul_cddmatrix_oz(d22, l21, max_num_div, max_num_div, u12, max_num_div, max_num_div);

	// a22 := a22 - d22
	index[0] = imax;
	index[1] = dim;
	index[2] = jmax;
	index[3] = dim;
	d22_index[0] = 0;
	d22_index[1] = d22->re->row_dim;
	d22_index[2] = 0;
	d22_index[3] = d22->re->col_dim;

	//sub_cddmatrix_partial(a, index, a, index, d22, d22_index);
	_bncomp_sub_cddmatrix_partial(a, index, a, index, d22, d22_index);

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
int CDDLUdecomp_oz(CDDMatrix a, long int min_dim, int max_num_div)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CDDMatrix a: Matrix (given by user)                */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, row_dim, col_dim, dim;
	CDDMatrix l21, u12, d22;

	dim = a->re->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		CDDLUdecomp(a);
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
				CDDLUdecomp_square(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		CDDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_cddmatrix(a);

		l21 = init_cddmatrix(row_dim, min_dim);
		u12 = init_cddmatrix(min_dim, col_dim);
		d22 = init_cddmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		CDDLUdecomp_l21(l21, a, i, min_dim);
		//print_cddmatrix(a);
		//printf("L21:\n");
		//print_cddmatrix(l21);

		// (3) Solve L11 * U12 = A12
		CDDLUdecomp_u12(u12, a, i, min_dim);
		//print_cddmatrix(a);
		//printf("U12:\n");
		//print_cddmatrix(u12);

		//printf("A:\n");
		//print_cddmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		CDDLUdecomp_a22_oz(a, d22, l21, u12, i, min_dim, max_num_div);
		//printf("D22:\n");
		//print_cddmatrix(d22);
		//print_cddmatrix(a);

		// clear
		free_cddmatrix(l21);
		free_cddmatrix(u12);
		free_cddmatrix(d22);

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
int CDDLUdecomp_ozPM(CDDMatrix a, long int ch[], long int min_dim, int max_num_div)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CDDMatrix a: Matrix (given by user)                */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, row_dim, col_dim, dim, imax, itmp;
	CDDMatrix l21, u12, d22;
	//static 
	double dtmp[DDSIZE], dtmp1[DDSIZE], dmaxii[DDSIZE];

	dim = a->re->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		CDDLUdecompPM(a, ch);
		return 0;
	}

	for(i = 0; i < dim; i++)
		ch[i] = i;

	// dim > min_dim
	for(i = 0; i < dim; i += min_dim)
	{
		// partial pivoting
		rcdd_abs_dd(dmaxii, get_cddmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rcdd_abs_dd(dtmp, get_cddmatrix_ij(a, j, i));
			if(rdd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rdd_set(dmaxii, dtmp);
			}
		}

		if(rdd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! CDDLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			row_swap_cddmatrix(a, i, imax, 0, a->re->col_dim);
		}

		// Initialize
		row_dim = dim - (i + min_dim);
		if(row_dim <= 0) 
		{
			// (1) L11 * U11 = A11
			if((dim - i) > 0)
				CDDLUdecomp_square(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		CDDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_cddmatrix(a);

		l21 = init_cddmatrix(row_dim, min_dim);
		u12 = init_cddmatrix(min_dim, col_dim);
		d22 = init_cddmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		CDDLUdecomp_l21(l21, a, i, min_dim);
		//print_cddmatrix(a);
		//printf("L21:\n");
		//print_cddmatrix(l21);

		// (3) Solve L11 * U12 = A12
		CDDLUdecomp_u12(u12, a, i, min_dim);
		//print_cddmatrix(a);
		//printf("U12:\n");
		//print_cddmatrix(u12);

		//printf("A:\n");
		//print_cddmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		CDDLUdecomp_a22_oz(a, d22, l21, u12, i, min_dim, max_num_div);
		//printf("D22:\n");
		//print_cddmatrix(d22);
		//print_cddmatrix(a);

		// clear
		free_cddmatrix(l21);
		free_cddmatrix(u12);
		free_cddmatrix(d22);

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
int CDDLUdecomp_oz_omp(CDDMatrix a, long int min_dim, int max_num_div)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CDDMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, row_dim, col_dim, dim;
	CDDMatrix l21, u12, d22;

	dim = a->re->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		CDDLUdecomp_omp(a);
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
				CDDLUdecomp_square_omp(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		CDDLUdecomp_square_omp(a, i, min_dim);
		//CDDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_cddmatrix(a);

		l21 = init_cddmatrix(row_dim, min_dim);
		u12 = init_cddmatrix(min_dim, col_dim);
		d22 = init_cddmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		CDDLUdecomp_l21_omp(l21, a, i, min_dim);
		//CDDLUdecomp_l21(l21, a, i, min_dim);
		//print_cddmatrix(a);
		//printf("L21:\n");
		//print_cddmatrix(l21);

		// (3) Solve L11 * U12 = A12
		CDDLUdecomp_u12_omp(u12, a, i, min_dim);
		//CDDLUdecomp_u12(u12, a, i, min_dim);
		//print_cddmatrix(a);
		//printf("U12:\n");
		//print_cddmatrix(u12);

		//printf("A:\n");
		//print_cddmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		CDDLUdecomp_a22_oz_omp(a, d22, l21, u12, i, min_dim, max_num_div);
		//CDDLUdecomp_a22(a, d22, l21, u12, i, min_dim);
		//printf("D22:\n");
		//print_cddmatrix(d22);
		//print_cddmatrix(a);

		// clear
		free_cddmatrix(l21);
		free_cddmatrix(u12);
		free_cddmatrix(d22);

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
int CDDLUdecomp_ozPM_omp(CDDMatrix a, long int ch[], long int min_dim, int max_num_div)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CDDMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, row_dim, col_dim, dim, imax, itmp;
	CDDMatrix l21, u12, d22;
	double dtmp[DDSIZE], dmaxii[DDSIZE];

	dim = a->re->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		CDDLUdecompPM_omp(a, ch);
		return 0;
	}

	for(i = 0; i < dim; i++)
		ch[i] = i;

	// dim > min_dim
	for(i = 0; i < dim; i += min_dim)
	{
		// partial pivoting
		rcdd_abs_dd(dmaxii, get_cddmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rcdd_abs_dd(dtmp, get_cddmatrix_ij(a, j, i));
			if(rdd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rdd_set(dmaxii, dtmp);
			}
		}

		if(rdd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! CDDLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			_bncomp_row_swap_cddmatrix(a, i, imax, 0, a->re->col_dim);
		}

		// Initialize
		row_dim = dim - (i + min_dim);
		if(row_dim <= 0) 
		{
			// (1) L11 * U11 = A11
			if((dim - i) > 0)
				CDDLUdecomp_square_omp(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		CDDLUdecomp_square_omp(a, i, min_dim);
		//CDDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_cddmatrix(a);

		l21 = init_cddmatrix(row_dim, min_dim);
		u12 = init_cddmatrix(min_dim, col_dim);
		d22 = init_cddmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		CDDLUdecomp_l21_omp(l21, a, i, min_dim);
		//CDDLUdecomp_l21(l21, a, i, min_dim);
		//print_cddmatrix(a);
		//printf("L21:\n");
		//print_cddmatrix(l21);

		// (3) Solve L11 * U12 = A12
		CDDLUdecomp_u12_omp(u12, a, i, min_dim);
		//CDDLUdecomp_u12(u12, a, i, min_dim);
		//print_cddmatrix(a);
		//printf("U12:\n");
		//print_cddmatrix(u12);

		//printf("A:\n");
		//print_cddmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		//CDDLUdecomp_a22_omp(a, d22, l21, u12, i, min_dim);
		CDDLUdecomp_a22_oz_omp(a, d22, l21, u12, i, min_dim, max_num_div);
		//printf("D22:\n");
		//print_cddmatrix(d22);
		//print_cddmatrix(a);

		// clear
		free_cddmatrix(l21);
		free_cddmatrix(u12);
		free_cddmatrix(d22);

	}

	return 0;
}
#endif // _OPENMP
