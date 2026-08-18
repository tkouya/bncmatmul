/********************************************************************************/
/* qdlu_oz.c: Octuple Precision LU decomposition with Ozaki scheme MM           */
/*                                                                              */
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

#include "qdlinear.h"

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
int QDLUdecomp_a22_oz(QDMatrix a, QDMatrix d22, QDMatrix l21, QDMatrix u12, long int start_index, long int min_dim, int max_num_div)
{
	long int i, j, k, imax, jmax, itmp, dim, index[4], d22_index[4];

	dim = a->col_dim;

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
//	mul_qdmatrix_strassen(d22, l21, u12, min_dim);
	mul_qdmatrix_oz(d22, l21, max_num_div, u12, max_num_div);
//#else
//	mul_qdmatrix_simple(d22, l21, u12);
//#endif

	// a22 := a22 - d22
	index[0] = imax;
	index[1] = dim;
	index[2] = jmax;
	index[3] = dim;
	d22_index[0] = 0;
	d22_index[1] = d22->row_dim;
	d22_index[2] = 0;
	d22_index[3] = d22->col_dim;

	sub_qdmatrix_partial(a, index, a, index, d22, d22_index);

	return 0;
}

#ifdef _OPENMP
int QDLUdecomp_a22_oz_omp(QDMatrix a, QDMatrix d22, QDMatrix l21, QDMatrix u12, long int start_index, long int min_dim, int max_num_div)
{
	long int i, j, k, imax, jmax, itmp, dim, index[4], d22_index[4];

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// d22 := l21 * u12
	_bncomp_mul_qdmatrix_oz(d22, l21, u12, min_dim, max_num_div);

	// a22 := a22 - d22
	index[0] = imax;
	index[1] = dim;
	index[2] = jmax;
	index[3] = dim;
	d22_index[0] = 0;
	d22_index[1] = d22->row_dim;
	d22_index[2] = 0;
	d22_index[3] = d22->col_dim;

	//sub_qdmatrix_partial(a, index, a, index, d22, d22_index);
	_bncomp_sub_qdmatrix_partial(a, index, a, index, d22, d22_index);

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
int QDLUdecomp_oz(QDMatrix a, long int min_dim, int max_num_div)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       QDMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, row_dim, col_dim, dim;
	QDMatrix l21, u12, d22;

	dim = a->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		QDLUdecomp(a);
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
				QDLUdecomp_square(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		QDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_qdmatrix(a);

		l21 = init_qdmatrix(row_dim, min_dim);
		u12 = init_qdmatrix(min_dim, col_dim);
		d22 = init_qdmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		QDLUdecomp_l21(l21, a, i, min_dim);
		//print_qdmatrix(a);
		//printf("L21:\n");
		//print_qdmatrix(l21);

		// (3) Solve L11 * U12 = A12
		QDLUdecomp_u12(u12, a, i, min_dim);
		//print_qdmatrix(a);
		//printf("U12:\n");
		//print_qdmatrix(u12);

		//printf("A:\n");
		//print_qdmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		QDLUdecomp_a22_oz(a, d22, l21, u12, i, min_dim, max_num_div);
		//printf("D22:\n");
		//print_qdmatrix(d22);
		//print_qdmatrix(a);

		// clear
		free_qdmatrix(l21);
		free_qdmatrix(u12);
		free_qdmatrix(d22);

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
int QDLUdecomp_ozPM(QDMatrix a, long int ch[], long int min_dim, int max_num_div)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       QDMatrix a: Matrix (given by user)                 */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, row_dim, col_dim, dim, imax, itmp;
	QDMatrix l21, u12, d22;
	static double dtmp[QDSIZE], dtmp1[QDSIZE], dmaxii[QDSIZE];

	dim = a->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		QDLUdecompPM(a, ch);
		return 0;
	}

	for(i = 0; i < dim; i++)
		ch[i] = i;

	// dim > min_dim
	for(i = 0; i < dim; i += min_dim)
	{
		// partial pivoting
		rqd_abs(dmaxii, get_qdmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rqd_abs(dtmp, get_qdmatrix_ij(a, j, i));
			if(rqd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rqd_set(dmaxii, dtmp);
			}
		}

		if(rqd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! QDLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			row_swap_qdmatrix(a, i, imax, 0, a->col_dim);
		}

		// Initialize
		row_dim = dim - (i + min_dim);
		if(row_dim <= 0) 
		{
			// (1) L11 * U11 = A11
			if((dim - i) > 0)
				QDLUdecomp_square(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		QDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_qdmatrix(a);

		l21 = init_qdmatrix(row_dim, min_dim);
		u12 = init_qdmatrix(min_dim, col_dim);
		d22 = init_qdmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		QDLUdecomp_l21(l21, a, i, min_dim);
		//print_qdmatrix(a);
		//printf("L21:\n");
		//print_qdmatrix(l21);

		// (3) Solve L11 * U12 = A12
		QDLUdecomp_u12(u12, a, i, min_dim);
		//print_qdmatrix(a);
		//printf("U12:\n");
		//print_qdmatrix(u12);

		//printf("A:\n");
		//print_qdmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		QDLUdecomp_a22_oz(a, d22, l21, u12, i, min_dim, max_num_div);
		//printf("D22:\n");
		//print_qdmatrix(d22);
		//print_qdmatrix(a);

		// clear
		free_qdmatrix(l21);
		free_qdmatrix(u12);
		free_qdmatrix(d22);

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
int QDLUdecomp_oz_omp(QDMatrix a, long int min_dim, int max_num_div)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       QDMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, row_dim, col_dim, dim;
	QDMatrix l21, u12, d22;

	dim = a->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		QDLUdecomp_omp(a);
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
				QDLUdecomp_square_omp(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		QDLUdecomp_square_omp(a, i, min_dim);
		//QDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_qdmatrix(a);

		l21 = init_qdmatrix(row_dim, min_dim);
		u12 = init_qdmatrix(min_dim, col_dim);
		d22 = init_qdmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		QDLUdecomp_l21_omp(l21, a, i, min_dim);
		//QDLUdecomp_l21(l21, a, i, min_dim);
		//print_qdmatrix(a);
		//printf("L21:\n");
		//print_qdmatrix(l21);

		// (3) Solve L11 * U12 = A12
		QDLUdecomp_u12_omp(u12, a, i, min_dim);
		//QDLUdecomp_u12(u12, a, i, min_dim);
		//print_qdmatrix(a);
		//printf("U12:\n");
		//print_qdmatrix(u12);

		//printf("A:\n");
		//print_qdmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		QDLUdecomp_a22_oz_omp(a, d22, l21, u12, i, min_dim, max_num_div);
		//QDLUdecomp_a22(a, d22, l21, u12, i, min_dim);
		//printf("D22:\n");
		//print_qdmatrix(d22);
		//print_qdmatrix(a);

		// clear
		free_qdmatrix(l21);
		free_qdmatrix(u12);
		free_qdmatrix(d22);

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
int QDLUdecomp_ozPM_omp(QDMatrix a, long int ch[], long int min_dim, int max_num_div)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       QDMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, row_dim, col_dim, dim, imax, itmp;
	QDMatrix l21, u12, d22;
	double dtmp[QDSIZE], dmaxii[QDSIZE];

	dim = a->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		QDLUdecompPM_omp(a, ch);
		return 0;
	}

	for(i = 0; i < dim; i++)
		ch[i] = i;

	// dim > min_dim
	for(i = 0; i < dim; i += min_dim)
	{
		// partial pivoting
		rqd_abs(dmaxii, get_qdmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rqd_abs(dtmp, get_qdmatrix_ij(a, j, i));
			if(rqd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rqd_set(dmaxii, dtmp);
			}
		}

		if(rqd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! QDLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			_bncomp_row_swap_qdmatrix(a, i, imax, 0, a->col_dim);
		}

		// Initialize
		row_dim = dim - (i + min_dim);
		if(row_dim <= 0) 
		{
			// (1) L11 * U11 = A11
			if((dim - i) > 0)
				QDLUdecomp_square_omp(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		QDLUdecomp_square_omp(a, i, min_dim);
		//QDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_qdmatrix(a);

		l21 = init_qdmatrix(row_dim, min_dim);
		u12 = init_qdmatrix(min_dim, col_dim);
		d22 = init_qdmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		QDLUdecomp_l21_omp(l21, a, i, min_dim);
		//QDLUdecomp_l21(l21, a, i, min_dim);
		//print_qdmatrix(a);
		//printf("L21:\n");
		//print_qdmatrix(l21);

		// (3) Solve L11 * U12 = A12
		QDLUdecomp_u12_omp(u12, a, i, min_dim);
		//QDLUdecomp_u12(u12, a, i, min_dim);
		//print_qdmatrix(a);
		//printf("U12:\n");
		//print_qdmatrix(u12);

		//printf("A:\n");
		//print_qdmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		QDLUdecomp_a22_oz_omp(a, d22, l21, u12, i, min_dim, max_num_div);
		//QDLUdecomp_a22(a, d22, l21, u12, i, min_dim);
		//printf("D22:\n");
		//print_qdmatrix(d22);
		//print_qdmatrix(a);

		// clear
		free_qdmatrix(l21);
		free_qdmatrix(u12);
		free_qdmatrix(d22);

	}

	return 0;
}
#endif // _OPENMP
