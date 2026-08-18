/********************************************************************************/
/* ddlu_strassen.c:                                                             */
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

// DD
// (1) L11 * U11 = A11
//
//    start_index
// +--+-----+--------+
// |  +-----+        |
// |  |\ U11|        |
// |  |L11\ |        |
// +  +-----+        |
// |  min_dim        |
// |                 |
// |                 |
// |                 |
// +-----------+-----+
//
int DDLUdecomp_square(DDMatrix a, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim;
	static double dtmp[DDSIZE], dtmp1[DDSIZE], dmaxii[DDSIZE];

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// dim > min_dim
	for(i = start_index; i < imax; i++)
	{
		rdd_abs(dmaxii, get_ddmatrix_ij(a, i, i));
		if(rdd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (DDLUdecomp_square)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < jmax; j++)
		{
			rdd_div(dtmp, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, i));
			set_ddmatrix_ij(a, j, i, dtmp);
		}

		for(j = (i + 1); j < jmax; j++)
		{
			for(k = (i + 1); k < jmax; k++)
			{
				//get_dmatrix_ij(a, j, k) - get_dmatrix_ij(a, j, i) * get_dmatrix_ij(a, i, k)
				rdd_mul(dtmp1, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(dtmp, get_ddmatrix_ij(a, j, k), dtmp1);
				set_ddmatrix_ij(a, j, k, dtmp);
			}
		}
	}

	return 0;
}

#ifdef _OPENMP
int DDLUdecomp_square_omp(DDMatrix a, long int start_index, long int min_dim)
{
	int thread_index, thread_num;
	long int i, j, k, imax, jmax, itmp, dim;
	static double dtmp[128][DDSIZE], dtmp1[128][DDSIZE], dmaxii[DDSIZE];

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();

		set0_dd(dtmp[thread_index]);
		set0_dd(dtmp1[thread_index]);
	}

	set0_dd(dmaxii);

	// dim > min_dim
	for(i = start_index; i < imax; i++)
	{
		rdd_abs(dmaxii, get_ddmatrix_ij(a, i, i));
		if(rdd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (DDLUdecomp_square)!\n", i);
			return -1;
		}

		#pragma omp parallel for private(thread_index)
		for(j = (i + 1); j < jmax; j++)
		{
			thread_index = omp_get_thread_num();

			rdd_div(dtmp[thread_index], get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, i));
			set_ddmatrix_ij(a, j, i, dtmp[thread_index]);
		}

		#pragma omp parallel for private(thread_index, k)
		for(j = (i + 1); j < jmax; j++)
		{
			thread_index = omp_get_thread_num();

			for(k = (i + 1); k < jmax; k++)
			{
				//get_dmatrix_ij(a, j, k) - get_dmatrix_ij(a, j, i) * get_dmatrix_ij(a, i, k)
				rdd_mul(dtmp1[thread_index], get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(dtmp[thread_index], get_ddmatrix_ij(a, j, k), dtmp1[thread_index]);
				set_ddmatrix_ij(a, j, k, dtmp[thread_index]);
			}
		}
	}

	return 0;
}
#endif // _OPENMP

// (2) Solve L21 * U11 = A21
//
//    start_index
// +--+-----+--------+
// |  +-----+        |
// |  |\ U11|        |
// |  |L11\ |        |
// +  +-----+        |
// |  |     |        |
// |  |     |        |
// |  | L21 |        |
// |  |     |        |
// +--+-----+--------+
//
int DDLUdecomp_l21(DDMatrix l21, DDMatrix a, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim;
	static double dtmp[DDSIZE], dtmp1[DDSIZE], dmaxii[DDSIZE];

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// dim > min_dim
	for(i = imax; i < dim; i++)
	{
		rdd_abs(dmaxii, get_ddmatrix_ij(a, i, i));
		if(rdd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (DDLUdecomp_l21)!\n", i);
			return -1;
		}

		for(j = start_index; j < jmax; j++)
		{
			rdd_set(dtmp, get_ddmatrix_ij(a, i, j));
			for(k = start_index; k < j; k++)
			{
				//dtmp -= get_dmatrix_ij(a, i, k) * get_dmatrix_ij(a, k, j);
				rdd_mul(dtmp1, get_ddmatrix_ij(a, i, k), get_ddmatrix_ij(a, k, j));
				rdd_sub(dtmp, dtmp, dtmp1);
			}
			rdd_div(dtmp1, dtmp, get_ddmatrix_ij(a, j, j));
			rdd_set(dtmp, dtmp1);
			
			//printf("l21(i              , j       ) = (%ld, %ld), %ld, %ld\n", i, j, l21->row_dim, l21->col_dim);
			//printf("l21(i - start_index, j - imax) = (%ld, %ld), %ld, %ld\n", i - imax, j - start_index, l21->row_dim, l21->col_dim);
			//printf("(i              , j       ) = (%ld, %ld) %25.17e, %25.17e\n", i, j, dtmp, get_dmatrix_ij(a, j, j));
			set_ddmatrix_ij(a  , i              , j       , dtmp);
			set_ddmatrix_ij(l21, i - imax, j - start_index, dtmp);
		}
	}

	return 0;
}

#ifdef _OPENMP
int DDLUdecomp_l21_omp(DDMatrix l21, DDMatrix a, long int start_index, long int min_dim)
{
	int thread_index, thread_num;
	long int i, j, k, imax, jmax, itmp, dim;
	static double dtmp[128][DDSIZE], dtmp1[128][DDSIZE], dmaxii[128][DDSIZE];

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		
		set0_dd(dtmp[thread_index]);
		set0_dd(dtmp1[thread_index]);
		set0_dd(dmaxii[thread_index]);
	}

	//set0_dd(dmaxii);

	// dim > min_dim
	#pragma omp parallel for private(thread_index, j, k)
	for(i = imax; i < dim; i++)
	{
		thread_index = omp_get_thread_num();

		rdd_abs(dmaxii[thread_index], get_ddmatrix_ij(a, i, i));
		if(rdd_cmp_ui(dmaxii[thread_index], 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (DDLUdecomp_l21_omp)!\n", i);
			//return -1;
		}

		for(j = start_index; j < jmax; j++)
		{
			rdd_set(dtmp[thread_index], get_ddmatrix_ij(a, i, j));
			for(k = start_index; k < j; k++)
			{
				//dtmp -= get_dmatrix_ij(a, i, k) * get_dmatrix_ij(a, k, j);
				rdd_mul(dtmp1[thread_index], get_ddmatrix_ij(a, i, k), get_ddmatrix_ij(a, k, j));
				rdd_sub(dtmp[thread_index], dtmp[thread_index], dtmp1[thread_index]);
			}
			rdd_div(dtmp1[thread_index], dtmp[thread_index], get_ddmatrix_ij(a, j, j));
			rdd_set(dtmp[thread_index], dtmp1[thread_index]);
			
			//printf("l21(i              , j       ) = (%ld, %ld), %ld, %ld\n", i, j, l21->row_dim, l21->col_dim);
			//printf("l21(i - start_index, j - imax) = (%ld, %ld), %ld, %ld\n", i - imax, j - start_index, l21->row_dim, l21->col_dim);
			//printf("(i              , j       ) = (%ld, %ld) %25.17e, %25.17e\n", i, j, dtmp, get_dmatrix_ij(a, j, j));
			set_ddmatrix_ij(a  , i              , j       , dtmp[thread_index]);
			set_ddmatrix_ij(l21, i - imax, j - start_index, dtmp[thread_index]);
		}
	}

	return 0;
}
#endif // _OPENMP

// (3) Solve L11 * U12 = A21
//
//    start_index
// +--+-----+--------+
// |  +-----+--------+
// |  |\ U11|  U12   |
// |  |L11\ |        |
// +  +-----+--------+
// |  |     |        |
// |  |     |        |
// |  | L21 |        |
// |  |     |        |
// +--+-----+--------+
//
int DDLUdecomp_u12(DDMatrix u12, DDMatrix a, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim;
	static double dtmp[DDSIZE], dtmp1[DDSIZE], dmaxii[DDSIZE];

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// dim > min_dim
	for(i = imax; i < dim; i++)
	{
		rdd_abs(dmaxii, get_ddmatrix_ij(a, i, i));
		if(rdd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (DDLUdecomp_u12)!\n", i);
			return -1;
		}

		for(j = start_index; j < jmax; j++)
		{
			rdd_set(dtmp, get_ddmatrix_ij(a, j, i));
			//printf("(j              , i              ) = (%ld, %ld) %25.17e\n", j, i , dtmp);
			for(k = start_index; k < j; k++)
			{
				//dtmp -= get_ddmatrix_ij(a, j, k) * get_ddmatrix_ij(a, k, i);
				rdd_mul(dtmp1, get_ddmatrix_ij(a, j, k), get_ddmatrix_ij(a, k, i));
				rdd_sub(dtmp, dtmp, dtmp1);
				//printf("(j - start_index, k - start_index) = (%ld, %ld)\n", j - start_index, k - start_index);
			}
			//printf("u12(j              , i       ) = (%ld, %ld) %ld, %ld\n", j, i, u12->row_dim, u12->col_dim);
			//printf("u12(j - start_index, i - imax) = (%ld, %ld) %ld, %ld\n", j - start_index, i - imax, u12->row_dim, u12->col_dim);
			set_ddmatrix_ij(a  , j              , i       , dtmp);
			set_ddmatrix_ij(u12, j - start_index, i - imax, dtmp);
		}
	}

	return 0;
}

#ifdef _OPENMP
int DDLUdecomp_u12_omp(DDMatrix u12, DDMatrix a, long int start_index, long int min_dim)
{
	int thread_index;
	long int i, j, k, imax, jmax, itmp, dim;
	static double dtmp[128][DDSIZE], dtmp1[128][DDSIZE], dmaxii[128][DDSIZE];

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();

		set0_dd(dtmp[thread_index]);
		set0_dd(dtmp1[thread_index]);
		set0_dd(dmaxii[thread_index]);
	}

	// dim > min_dim
	#pragma omp parallel for private(thread_index, j, k)
	for(i = imax; i < dim; i++)
	{
		thread_index = omp_get_thread_num();

		rdd_abs(dmaxii[thread_index], get_ddmatrix_ij(a, i, i));
		if(rdd_cmp_ui(dmaxii[thread_index], 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (DDLUdecomp_u12_omp)!\n", i);
			//return -1;
		}

		for(j = start_index; j < jmax; j++)
		{
			rdd_set(dtmp[thread_index], get_ddmatrix_ij(a, j, i));
			//printf("(j              , i              ) = (%ld, %ld) %25.17e\n", j, i , dtmp);
			for(k = start_index; k < j; k++)
			{
				//dtmp -= get_ddmatrix_ij(a, j, k) * get_ddmatrix_ij(a, k, i);
				rdd_mul(dtmp1[thread_index], get_ddmatrix_ij(a, j, k), get_ddmatrix_ij(a, k, i));
				rdd_sub(dtmp[thread_index], dtmp[thread_index], dtmp1[thread_index]);
				//printf("(j - start_index, k - start_index) = (%ld, %ld)\n", j - start_index, k - start_index);
			}
			//printf("u12(j              , i       ) = (%ld, %ld) %ld, %ld\n", j, i, u12->row_dim, u12->col_dim);
			//printf("u12(j - start_index, i - imax) = (%ld, %ld) %ld, %ld\n", j - start_index, i - imax, u12->row_dim, u12->col_dim);
			set_ddmatrix_ij(a  , j              , i       , dtmp[thread_index]);
			set_ddmatrix_ij(u12, j - start_index, i - imax, dtmp[thread_index]);
		}
	}

	return 0;
}
#endif // _OPENMP


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
#ifndef STRASSEN_MIN_DIM
#define STRASSEN_MIN_DIM 32
#endif // STRASSEN_MIN_DIM
int DDLUdecomp_a22(DDMatrix a, DDMatrix d22, DDMatrix l21, DDMatrix u12, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim, index[4], d22_index[4];

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// d22 := l21 * u12
#ifdef USE_BLOCK
//	mul_ddmatrix_block(d22, l21, u12, min_dim);
	mul_ddmatrix_block(d22, l21, u12, STRASSEN_MIN_DIM);
#elif USE_STRASSEN
//	mul_ddmatrix_strassen(d22, l21, u12, min_dim);
	mul_ddmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM);
#elif USE_WINOGRAD
//	mul_ddmatrix_strassen(d22, l21, u12, min_dim);
	mul_ddmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM);
#else
	mul_ddmatrix_simple(d22, l21, u12);
#endif

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
int DDLUdecomp_a22_omp(DDMatrix a, DDMatrix d22, DDMatrix l21, DDMatrix u12, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim, index[4], d22_index[4];

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// d22 := l21 * u12
#ifdef USE_BLOCK
	mul_ddmatrix_block(d22, l21, u12, min_dim);
	//_bncomp_mul_ddmatrix_block(d22, l21, u12, STRASSEN_MIN_DIM);
//	_bncomp_mul_ddmatrix_block_old(d22, l21, u12, STRASSEN_MIN_DIM);
#elif USE_STRASSEN
//	mul_ddmatrix_strassen(d22, l21, u12, min_dim);
	_bncomp_mul_ddmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM);
	//print_normf_ddmatrix("||d22||_F = ", d22);
	//print_normf_ddmatrix("||l21||_F = ", l21);
	//print_normf_ddmatrix("||u12||_F = ", u12);
#elif USE_WINOGRAD
//	mul_ddmatrix_strassen(d22, l21, u12, min_dim);
	_bncomp_mul_ddmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM);
#else
	_bncomp_mul_ddmatrix_simple(d22, l21, u12);
#endif

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
int DDLUdecomp_strassen(DDMatrix a, long int min_dim)
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
		DDLUdecomp_a22(a, d22, l21, u12, i, min_dim);
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
int DDLUdecomp_strassenPM(DDMatrix a, long int ch[], long int min_dim)
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
		DDLUdecomp_a22(a, d22, l21, u12, i, min_dim);
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
/*                                  (dd_real)               */
/*                                                          */
/*                 ver. 0.0 2015-07-07 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
int DDLUdecomp_omp(DDMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*        DDMatrix a: Matrix (given by user)                */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	double tmp[128][DDSIZE], axii[DDSIZE];
	int thread_index;

	dim = a->col_dim;
	set0_dd(axii);

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_dd(tmp[thread_index]);
	}

	for(i = 0; i < dim; i++)
	{
		rdd_abs(axii, get_ddmatrix_ij(a, i, i));

		if(rdd_cmp_ui(axii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (DDLUdecomp_omp)!\n", i);
			return -1;
		}

		#pragma omp parallel for private(thread_index)
		for(j = (i + 1); j < dim; j++)
		{
			thread_index = omp_get_thread_num();

			rdd_div(tmp[thread_index], get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, i));
			set_ddmatrix_ij(a, j, i, tmp[thread_index]);
		}

		#pragma omp parallel for private(thread_index, k)
		for(j = (i + 1); j < dim; j++)
		{
			thread_index = omp_get_thread_num();

			for(k = (i + 1); k < dim; k++)
			{
				rdd_mul(tmp[thread_index], get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(tmp[thread_index], get_ddmatrix_ij(a, j, k), tmp[thread_index]);
				set_ddmatrix_ij(a, j, k, tmp[thread_index]);
			}
		}
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                (Triple double precision) */
/*                                                          */
/*                 ver. 0.0 2021-02-17 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int DDLUdecompPM_omp(DDMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*        DDMatrix a: Matrix (given by user)                */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	double tmp[128][DDSIZE], tmp1[128][DDSIZE], axii[DDSIZE];
	int thread_index;
// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m256d dtmp256[128][DDSIZE], aji256[128][DDSIZE], ajk256[128][DDSIZE], aik256[128][DDSIZE];
#elif defined(__AVX512F__) // __AVX512F__
#endif // __AVX2__

	dim = a->col_dim;
	set0_dd(axii);

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_dd(tmp[thread_index]);
	}

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		// partial pivoting
		rdd_abs(axii, get_ddmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rdd_abs(tmp[0], get_ddmatrix_ij(a, j, i));
			if(rdd_cmp(tmp[0], axii) > 0)
			{
				imax = j;
				rdd_set(axii, tmp[0]);
			}
		}

		if(rdd_cmp_ui(axii, 0UL) == 0)
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

		#pragma omp parallel for private(thread_index)
		for(j = (i + 1); j < dim; j++)
		{
			thread_index = omp_get_thread_num();

			rdd_div(tmp[thread_index], get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, i));
			set_ddmatrix_ij(a, j, i, tmp[thread_index]);
		}

// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->real_col_dim;

		//printf("real_row_dim, real_col_dim, dim, i, dim_start, dim_end = %ld, %ld, %ld, %ld, %ld, %ld\n", a->real_row_dim, a->real_col_dim, dim, i, dim_start, dim_end);
		#pragma omp parallel for private(thread_index, k, k_start, k_end, index_ji, index_jk, index_ik)
		for(j = (i + 1); j < dim; j++)
		{
			thread_index = omp_get_thread_num();

			index_ji = j * a->real_col_dim + i;
			aji256[thread_index][0] = _mm256_set_pd(
                a->element[0][index_ji],
                a->element[0][index_ji],
                a->element[0][index_ji],
                a->element[0][index_ji]
            );
			aji256[thread_index][1] = _mm256_set_pd(
                a->element[1][index_ji],
                a->element[1][index_ji],
                a->element[1][index_ji],
                a->element[1][index_ji]
            );

			// head
			//printf("start j, k= %ld, %ld, ", j, i + 1);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			//for(k = (i + 1); k < dim; k++)
			{
				rdd_mul(tmp[thread_index], get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(tmp[thread_index], get_ddmatrix_ij(a, j, k), tmp[thread_index]);
				set_ddmatrix_ij(a, j, k, tmp[thread_index]);
			}
			//printf("head k_start, k = %ld, %ld, ", k_start, k);

			// middle : SIMD
			k_end = dim_end;
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->real_col_dim + k;
				//rdd_mul(dtmp1, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				aik256[thread_index][0] = _mm256_load_pd(&(a->element[0][index_ik]));
				aik256[thread_index][1] = _mm256_load_pd(&(a->element[1][index_ik]));
				_bncavx2_rdd_mul(dtmp256[thread_index], aji256[thread_index], aik256[thread_index]);
				//printf(" -- mul -- ");

				index_jk = j * a->real_col_dim + k;
				//rdd_sub(dtmp, get_ddmatrix_ij(a, j, k), dtmp1);
				ajk256[thread_index][0] = _mm256_load_pd(&(a->element[0][index_jk]));
				ajk256[thread_index][1] = _mm256_load_pd(&(a->element[1][index_jk]));
				_bncavx2_rdd_sub(dtmp256[thread_index], ajk256[thread_index], dtmp256[thread_index]);
				//printf(" -- sub -- ");

				//set_ddmatrix_ij(a, j, k, dtmp);
				_mm256_store_pd(&(a->element[0][index_jk]), dtmp256[thread_index][0]);
				_mm256_store_pd(&(a->element[1][index_jk]), dtmp256[thread_index][1]);
			}
			//printf(", %ld middle", k);
		}
#elif defined(__AVX512F__) // __AVX512F__
#else // others
		#pragma omp parallel for private(thread_index, k)
		for(j = (i + 1); j < dim; j++)
		{
			thread_index = omp_get_thread_num();

			for(k = (i + 1); k < dim; k++)
			{
				rdd_mul(tmp[thread_index], get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				rdd_sub(tmp[thread_index], get_ddmatrix_ij(a, j, k), tmp[thread_index]);
				set_ddmatrix_ij(a, j, k, tmp[thread_index]);
			}
		}
#endif // __AVX2__
	}

	return 0;
}
#endif // _OPENMP


#ifdef _OPENMP
/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                 (Quadraple Precision)    */
/*                                                          */
/*                 ver. 0.0 2015-02-25 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int DDLUdecomp_strassen_omp(DDMatrix a, long int min_dim)
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
		DDLUdecomp_a22_omp(a, d22, l21, u12, i, min_dim);
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
int DDLUdecomp_strassenPM_omp(DDMatrix a, long int ch[], long int min_dim)
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
		DDLUdecomp_a22(a, d22, l21, u12, i, min_dim);
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
