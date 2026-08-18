/********************************************************************************/
/* ctdlu_oz.c: Complex Triple-double Precision LU decomposition                 */ 
/*                                                        with Ozaki scheme MM  */
/*                                                                              */
/* Copyright (C) 2024 Tomonori Kouya                                            */
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

#include "ctdlinear.h"

#include "matmul_strassen.h"
#include "bncomp.h"

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
int CTDLUdecomp_a22_oz(CTDMatrix a, CTDMatrix d22, CTDMatrix l21, CTDMatrix u12, long int start_index, long int min_dim, int max_num_div)
{
	long int i, j, k, imax, jmax, itmp, dim, index[4], d22_index[4];

	dim = a->re->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// d22 := l21 * u12
//ifdef USE_OZ
	//#ifdef OZ_MAX_NUM_DIV
	//int max_num_div = OZ_MAX_NUM_DIV;
	//#else
	//int max_num_div = 10;
	//#endif // OZ_MAX_NUM_DIV
//	mul_ctdmatrix_strassen(d22, l21, u12, min_dim);
	mul_ctdmatrix_oz(d22, l21, max_num_div, max_num_div, u12, max_num_div, max_num_div);
//#else
//	mul_ctdmatrix_simple(d22, l21, u12);
//#endif

	// a22 := a22 - d22
	index[0] = imax;
	index[1] = dim;
	index[2] = jmax;
	index[3] = dim;
	d22_index[0] = 0;
	d22_index[1] = d22->re->row_dim;
	d22_index[2] = 0;
	d22_index[3] = d22->re->col_dim;

	sub_ctdmatrix_partial(a, index, a, index, d22, d22_index);

	return 0;
}

#ifdef _OPENMP
int CTDLUdecomp_a22_oz_omp(CTDMatrix a, CTDMatrix d22, CTDMatrix l21, CTDMatrix u12, long int start_index, long int min_dim, int max_num_div)
{
	long int i, j, k, imax, jmax, itmp, dim, index[4], d22_index[4];

	dim = a->re->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// d22 := l21 * u12
	//mul_ctdmatrix_oz(d22, l21, max_num_div, max_num_div, u12, max_num_div, max_num_div);
	_bncomp_mul_ctdmatrix_oz(d22, l21, max_num_div, max_num_div, u12, max_num_div, max_num_div);

	// a22 := a22 - d22
	index[0] = imax;
	index[1] = dim;
	index[2] = jmax;
	index[3] = dim;
	d22_index[0] = 0;
	d22_index[1] = d22->re->row_dim;
	d22_index[2] = 0;
	d22_index[3] = d22->re->col_dim;

	//sub_ctdmatrix_partial(a, index, a, index, d22, d22_index);
	_bncomp_sub_ctdmatrix_partial(a, index, a, index, d22, d22_index);

	return 0;
}
#endif // _OPENMP

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                (Triple-double Precision) */
/*                                                          */
/*                 ver. 0.0 2024-01-31 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CTDLUdecomp_oz(CTDMatrix a, long int min_dim, int max_num_div)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CTDMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, row_dim, col_dim, dim;
	CTDMatrix l21, u12, d22;

	dim = a->re->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		CTDLUdecomp(a);
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
				CTDLUdecomp_square(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		CTDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_ctdmatrix(a);

		l21 = init_ctdmatrix(row_dim, min_dim);
		u12 = init_ctdmatrix(min_dim, col_dim);
		d22 = init_ctdmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		CTDLUdecomp_l21(l21, a, i, min_dim);
		//print_ctdmatrix(a);
		//printf("L21:\n");
		//print_ctdmatrix(l21);

		// (3) Solve L11 * U12 = A12
		CTDLUdecomp_u12(u12, a, i, min_dim);
		//print_ctdmatrix(a);
		//printf("U12:\n");
		//print_ctdmatrix(u12);

		//printf("A:\n");
		//print_ctdmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		CTDLUdecomp_a22_oz(a, d22, l21, u12, i, min_dim, max_num_div);
		//printf("D22:\n");
		//print_ctdmatrix(d22);
		//print_ctdmatrix(a);

		// clear
		free_ctdmatrix(l21);
		free_ctdmatrix(u12);
		free_ctdmatrix(d22);

	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                (Triple-double Precision) */
/*                                                          */
/*                 ver. 0.0 2024-01-31 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CTDLUdecomp_ozPM(CTDMatrix a, long int ch[], long int min_dim, int max_num_div)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CTDMatrix a: Matrix (given by user)                 */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, row_dim, col_dim, dim, imax, itmp;
	CTDMatrix l21, u12, d22;
	double dtmp[TDSIZE], dtmp1[TDSIZE], dmaxii[TDSIZE];
    ctdfloat ctmp, ctmp1;

	dim = a->re->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		CTDLUdecompPM(a, ch);
		return 0;
	}

	for(i = 0; i < dim; i++)
		ch[i] = i;

	// dim > min_dim
	for(i = 0; i < dim; i += min_dim)
	{
		// partial pivoting
		rctd_abs_td(dmaxii, get_ctdmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rctd_abs_td(dtmp, get_ctdmatrix_ij(a, j, i));
			if(rtd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rtd_set(dmaxii, dtmp);
			}
		}

		if(rtd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! CTDLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			row_swap_ctdmatrix(a, i, imax, 0, a->re->col_dim);
		}

		// Initialize
		row_dim = dim - (i + min_dim);
		if(row_dim <= 0) 
		{
			// (1) L11 * U11 = A11
			if((dim - i) > 0)
				CTDLUdecomp_square(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		CTDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_ctdmatrix(a);

		l21 = init_ctdmatrix(row_dim, min_dim);
		u12 = init_ctdmatrix(min_dim, col_dim);
		d22 = init_ctdmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		CTDLUdecomp_l21(l21, a, i, min_dim);
		//print_ctdmatrix(a);
		//printf("L21:\n");
		//print_ctdmatrix(l21);

		// (3) Solve L11 * U12 = A12
		CTDLUdecomp_u12(u12, a, i, min_dim);
		//print_ctdmatrix(a);
		//printf("U12:\n");
		//print_ctdmatrix(u12);

		//printf("A:\n");
		//print_ctdmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		CTDLUdecomp_a22_oz(a, d22, l21, u12, i, min_dim, max_num_div);
		//printf("D22:\n");
		//print_ctdmatrix(d22);
		//print_ctdmatrix(a);

		// clear
		free_ctdmatrix(l21);
		free_ctdmatrix(u12);
		free_ctdmatrix(d22);

	}

	return 0;
}

#ifdef _OPENMP
/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                       (Complex Triple-double Precision)  */
/*                                                          */
/*                 ver. 0.0 2024-01-31 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CTDLUdecomp_oz_omp(CTDMatrix a, long int min_dim, int max_num_div)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CTDMatrix a: Matrix (given by user)                */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, row_dim, col_dim, dim;
	CTDMatrix l21, u12, d22;

	dim = a->re->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		CTDLUdecomp_omp(a);
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
				CTDLUdecomp_square_omp(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		CTDLUdecomp_square_omp(a, i, min_dim);
		//CTDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_ctdmatrix(a);

		l21 = init_ctdmatrix(row_dim, min_dim);
		u12 = init_ctdmatrix(min_dim, col_dim);
		d22 = init_ctdmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		CTDLUdecomp_l21_omp(l21, a, i, min_dim);
		//CTDLUdecomp_l21(l21, a, i, min_dim);
		//print_ctdmatrix(a);
		//printf("L21:\n");
		//print_ctdmatrix(l21);

		// (3) Solve L11 * U12 = A12
		CTDLUdecomp_u12_omp(u12, a, i, min_dim);
		//CTDLUdecomp_u12(u12, a, i, min_dim);
		//print_ctdmatrix(a);
		//printf("U12:\n");
		//print_ctdmatrix(u12);

		//printf("A:\n");
		//print_ctdmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		CTDLUdecomp_a22_oz_omp(a, d22, l21, u12, i, min_dim, max_num_div);
		//CTDLUdecomp_a22(a, d22, l21, u12, i, min_dim);
		//printf("D22:\n");
		//print_ctdmatrix(d22);
		//print_ctdmatrix(a);

		// clear
		free_ctdmatrix(l21);
		free_ctdmatrix(u12);
		free_ctdmatrix(d22);

	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                (Triple-double Precision) */
/*                                                          */
/*                 ver. 0.0 2024-01-31 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CTDLUdecomp_ozPM_omp(CTDMatrix a, long int ch[], long int min_dim, int max_num_div)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CTDMatrix a: Matrix (given by user)                */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, row_dim, col_dim, dim, imax, itmp;
	CTDMatrix l21, u12, d22;
	double dtmp[TDSIZE], dmaxii[TDSIZE];
    ctdfloat ctmp;

	dim = a->re->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		CTDLUdecompPM_omp(a, ch);
		return 0;
	}

	for(i = 0; i < dim; i++)
		ch[i] = i;

	// dim > min_dim
	for(i = 0; i < dim; i += min_dim)
	{
		// partial pivoting
		rctd_abs_td(dmaxii, get_ctdmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rctd_abs_td(dtmp, get_ctdmatrix_ij(a, j, i));
			if(rtd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rtd_set(dmaxii, dtmp);
			}
		}

		if(rtd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! CTDLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			_bncomp_row_swap_ctdmatrix(a, i, imax, 0, a->re->col_dim);
		}

		// Initialize
		row_dim = dim - (i + min_dim);
		if(row_dim <= 0) 
		{
			// (1) L11 * U11 = A11
			if((dim - i) > 0)
				CTDLUdecomp_square_omp(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		CTDLUdecomp_square_omp(a, i, min_dim);
		//CTDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_ctdmatrix(a);

		l21 = init_ctdmatrix(row_dim, min_dim);
		u12 = init_ctdmatrix(min_dim, col_dim);
		d22 = init_ctdmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		CTDLUdecomp_l21_omp(l21, a, i, min_dim);
		//CTDLUdecomp_l21(l21, a, i, min_dim);
		//print_ctdmatrix(a);
		//printf("L21:\n");
		//print_ctdmatrix(l21);

		// (3) Solve L11 * U12 = A12
		CTDLUdecomp_u12_omp(u12, a, i, min_dim);
		//CTDLUdecomp_u12(u12, a, i, min_dim);
		//print_ctdmatrix(a);
		//printf("U12:\n");
		//print_ctdmatrix(u12);

		//printf("A:\n");
		//print_ctdmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		CTDLUdecomp_a22_oz_omp(a, d22, l21, u12, i, min_dim, max_num_div);
		//CTDLUdecomp_a22(a, d22, l21, u12, i, min_dim);
		//printf("D22:\n");
		//print_ctdmatrix(d22);
		//print_ctdmatrix(a);

		// clear
		free_ctdmatrix(l21);
		free_ctdmatrix(u12);
		free_ctdmatrix(d22);

	}

	return 0;
}
#endif // _OPENMP
