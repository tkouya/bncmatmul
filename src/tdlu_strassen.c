/********************************************************************************/
/* tdlu_strassen.c: Triple Precision LU decomposition with Strassen MM          */
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

#include "tdlinear.h"

#include "matmul_strassen.h"
#include "bncomp.h"


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
int TDLUdecomp_square(TDMatrix a, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim;
#ifdef __cplusplus
	static td_real dtmp, dtmp1, dmaxii;
#else // __cplusplus
	static double dtmp[TDSIZE], dtmp1[TDSIZE], dmaxii[TDSIZE];
#endif // __cplusplus

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// dim > min_dim
	for(i = start_index; i < imax; i++)
	{
		rtd_abs(dmaxii, get_tdmatrix_ij(a, i, i));
		if(rtd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (TDLUdecomp_square)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < jmax; j++)
		{
			rtd_div(dtmp, get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, i));
			set_tdmatrix_ij(a, j, i, dtmp);
		}

		for(j = (i + 1); j < jmax; j++)
		{
			for(k = (i + 1); k < jmax; k++)
			{
				//get_dmatrix_ij(a, j, k) - get_dmatrix_ij(a, j, i) * get_dmatrix_ij(a, i, k)
				rtd_mul(dtmp1, get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, k));
				rtd_sub(dtmp, get_tdmatrix_ij(a, j, k), dtmp1);
				set_tdmatrix_ij(a, j, k, dtmp);
			}
		}
	}

	return 0;
}

#ifdef _OPENMP
int TDLUdecomp_square_omp(TDMatrix a, long int start_index, long int min_dim)
{
	int thread_index, thread_num;
	long int i, j, k, imax, jmax, itmp, dim;
#ifdef __cplusplus
	static td_real dtmp[128], dtmp1[128], dmaxii;
#else // __cplusplus
	static double dtmp[128][TDSIZE], dtmp1[128][TDSIZE], dmaxii[TDSIZE];
#endif // __cplusplus

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();

		set0_td(dtmp[thread_index]);
		set0_td(dtmp1[thread_index]);
	}

	set0_td(dmaxii);

	// dim > min_dim
	for(i = start_index; i < imax; i++)
	{
		rtd_abs(dmaxii, get_tdmatrix_ij(a, i, i));
		if(rtd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (TDLUdecomp_square)!\n", i);
			return -1;
		}

		#pragma omp parallel for private(thread_index)
		for(j = (i + 1); j < jmax; j++)
		{
			thread_index = omp_get_thread_num();

			rtd_div(dtmp[thread_index], get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, i));
			set_tdmatrix_ij(a, j, i, dtmp[thread_index]);
		}

		#pragma omp parallel for private(thread_index, k)
		for(j = (i + 1); j < jmax; j++)
		{
			thread_index = omp_get_thread_num();

			for(k = (i + 1); k < jmax; k++)
			{
				//get_dmatrix_ij(a, j, k) - get_dmatrix_ij(a, j, i) * get_dmatrix_ij(a, i, k)
				rtd_mul(dtmp1[thread_index], get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, k));
				rtd_sub(dtmp[thread_index], get_tdmatrix_ij(a, j, k), dtmp1[thread_index]);
				set_tdmatrix_ij(a, j, k, dtmp[thread_index]);
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
int TDLUdecomp_l21(TDMatrix l21, TDMatrix a, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim;
#ifdef __cplusplus
	static td_real dtmp, dtmp1, dmaxii;
#else // __cplusplus
	static double dtmp[TDSIZE], dtmp1[TDSIZE], dmaxii[TDSIZE];
#endif // __cplusplus

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// dim > min_dim
	for(i = imax; i < dim; i++)
	{
		rtd_abs(dmaxii, get_tdmatrix_ij(a, i, i));
		if(rtd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (TDLUdecomp_l21)!\n", i);
			return -1;
		}

		for(j = start_index; j < jmax; j++)
		{
			rtd_set(dtmp, get_tdmatrix_ij(a, i, j));
			for(k = start_index; k < j; k++)
			{
				//dtmp -= get_dmatrix_ij(a, i, k) * get_dmatrix_ij(a, k, j);
				rtd_mul(dtmp1, get_tdmatrix_ij(a, i, k), get_tdmatrix_ij(a, k, j));
				rtd_sub(dtmp, dtmp, dtmp1);
			}
			rtd_div(dtmp1, dtmp, get_tdmatrix_ij(a, j, j));
			rtd_set(dtmp, dtmp1);
			
			//printf("l21(i              , j       ) = (%ld, %ld), %ld, %ld\n", i, j, l21->row_dim, l21->col_dim);
			//printf("l21(i - start_index, j - imax) = (%ld, %ld), %ld, %ld\n", i - imax, j - start_index, l21->row_dim, l21->col_dim);
			//printf("(i              , j       ) = (%ld, %ld) %25.17e, %25.17e\n", i, j, dtmp, get_dmatrix_ij(a, j, j));
			set_tdmatrix_ij(a  , i              , j       , dtmp);
			set_tdmatrix_ij(l21, i - imax, j - start_index, dtmp);
		}
	}

	return 0;
}

#ifdef _OPENMP
int TDLUdecomp_l21_omp(TDMatrix l21, TDMatrix a, long int start_index, long int min_dim)
{
	int thread_index, thread_num;
	long int i, j, k, imax, jmax, itmp, dim;
#ifdef __cplusplus
	static td_real dtmp[128], dtmp1[128], dmaxii[128];
#else // __cplusplus
	static double dtmp[128][TDSIZE], dtmp1[128][TDSIZE], dmaxii[128][TDSIZE];
#endif // __cplusplus

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		
		set0_td(dtmp[thread_index]);
		set0_td(dtmp1[thread_index]);
		set0_td(dmaxii[thread_index]);
	}
	//set0_td(dmaxii);

	// dim > min_dim
	#pragma omp parallel for private(thread_index, j, k)
	for(i = imax; i < dim; i++)
	{
		thread_index = omp_get_thread_num();

		rtd_abs(dmaxii[thread_index], get_tdmatrix_ij(a, i, i));
		if(rtd_cmp_ui(dmaxii[thread_index], 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (TDLUdecomp_l21_omp)!\n", i);
			//return -1;
		}

		for(j = start_index; j < jmax; j++)
		{
			rtd_set(dtmp[thread_index], get_tdmatrix_ij(a, i, j));
			for(k = start_index; k < j; k++)
			{
				//dtmp -= get_dmatrix_ij(a, i, k) * get_dmatrix_ij(a, k, j);
				rtd_mul(dtmp1[thread_index], get_tdmatrix_ij(a, i, k), get_tdmatrix_ij(a, k, j));
				rtd_sub(dtmp[thread_index], dtmp[thread_index], dtmp1[thread_index]);
			}
			rtd_div(dtmp1[thread_index], dtmp[thread_index], get_tdmatrix_ij(a, j, j));
			rtd_set(dtmp[thread_index], dtmp1[thread_index]);
			
			//printf("l21(i              , j       ) = (%ld, %ld), %ld, %ld\n", i, j, l21->row_dim, l21->col_dim);
			//printf("l21(i - start_index, j - imax) = (%ld, %ld), %ld, %ld\n", i - imax, j - start_index, l21->row_dim, l21->col_dim);
			//printf("(i              , j       ) = (%ld, %ld) %25.17e, %25.17e\n", i, j, dtmp, get_dmatrix_ij(a, j, j));
			set_tdmatrix_ij(a  , i              , j       , dtmp[thread_index]);
			set_tdmatrix_ij(l21, i - imax, j - start_index, dtmp[thread_index]);
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
int TDLUdecomp_u12(TDMatrix u12, TDMatrix a, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim;
#ifdef __cplusplus
	static td_real dtmp, dtmp1, dmaxii;
#else // __cplusplus
	static double dtmp[TDSIZE], dtmp1[TDSIZE], dmaxii[TDSIZE];
#endif // __cplusplus

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// dim > min_dim
	for(i = imax; i < dim; i++)
	{
		rtd_abs(dmaxii, get_tdmatrix_ij(a, i, i));
		if(rtd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (TDLUdecomp_u12)!\n", i);
			return -1;
		}

		for(j = start_index; j < jmax; j++)
		{
			rtd_set(dtmp, get_tdmatrix_ij(a, j, i));
			//printf("(j              , i              ) = (%ld, %ld) %25.17e\n", j, i , dtmp);
			for(k = start_index; k < j; k++)
			{
				//dtmp -= get_tdmatrix_ij(a, j, k) * get_tdmatrix_ij(a, k, i);
				rtd_mul(dtmp1, get_tdmatrix_ij(a, j, k), get_tdmatrix_ij(a, k, i));
				rtd_sub(dtmp, dtmp, dtmp1);
				//printf("(j - start_index, k - start_index) = (%ld, %ld)\n", j - start_index, k - start_index);
			}
			//printf("u12(j              , i       ) = (%ld, %ld) %ld, %ld\n", j, i, u12->row_dim, u12->col_dim);
			//printf("u12(j - start_index, i - imax) = (%ld, %ld) %ld, %ld\n", j - start_index, i - imax, u12->row_dim, u12->col_dim);
			set_tdmatrix_ij(a  , j              , i       , dtmp);
			set_tdmatrix_ij(u12, j - start_index, i - imax, dtmp);
		}
	}

	return 0;
}

#ifdef _OPENMP
int TDLUdecomp_u12_omp(TDMatrix u12, TDMatrix a, long int start_index, long int min_dim)
{
	int thread_index;
	long int i, j, k, imax, jmax, itmp, dim;
#ifdef __cplusplus
	static td_real dtmp[128], dtmp1[128], dmaxii;
#else // __cplusplus
	static double dtmp[128][TDSIZE], dtmp1[128][TDSIZE], dmaxii[128][TDSIZE];
#endif // __cplusplus

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();

		set0_td(dtmp[thread_index]);
		set0_td(dtmp1[thread_index]);
	}

	// dim > min_dim
	#pragma omp parallel for private(thread_index, j, k)
	for(i = imax; i < dim; i++)
	{
		thread_index = omp_get_thread_num();

		rtd_abs(dmaxii[thread_index], get_tdmatrix_ij(a, i, i));
		if(rtd_cmp_ui(dmaxii[thread_index], 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (TDLUdecomp_u12_omp)!\n", i);
			//return -1;
		}

		for(j = start_index; j < jmax; j++)
		{
			rtd_set(dtmp[thread_index], get_tdmatrix_ij(a, j, i));
			//printf("(j              , i              ) = (%ld, %ld) %25.17e\n", j, i , dtmp);
			for(k = start_index; k < j; k++)
			{
				//dtmp -= get_tdmatrix_ij(a, j, k) * get_tdmatrix_ij(a, k, i);
				rtd_mul(dtmp1[thread_index], get_tdmatrix_ij(a, j, k), get_tdmatrix_ij(a, k, i));
				rtd_sub(dtmp[thread_index], dtmp[thread_index], dtmp1[thread_index]);
				//printf("(j - start_index, k - start_index) = (%ld, %ld)\n", j - start_index, k - start_index);
			}
			//printf("u12(j              , i       ) = (%ld, %ld) %ld, %ld\n", j, i, u12->row_dim, u12->col_dim);
			//printf("u12(j - start_index, i - imax) = (%ld, %ld) %ld, %ld\n", j - start_index, i - imax, u12->row_dim, u12->col_dim);
			set_tdmatrix_ij(a  , j              , i       , dtmp[thread_index]);
			set_tdmatrix_ij(u12, j - start_index, i - imax, dtmp[thread_index]);
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
int TDLUdecomp_a22(TDMatrix a, TDMatrix d22, TDMatrix l21, TDMatrix u12, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim, index[4], d22_index[4];

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// d22 := l21 * u12
#ifdef USE_BLOCK
//	mul_tdmatrix_block(d22, l21, u12, min_dim);
	mul_tdmatrix_block(d22, l21, u12, STRASSEN_MIN_DIM);
#elif USE_STRASSEN
//	mul_tdmatrix_strassen(d22, l21, u12, min_dim);
	mul_tdmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM);
#elif USE_WINOGRAD
//	mul_tdmatrix_strassen(d22, l21, u12, min_dim);
	mul_tdmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM);
#else
	mul_tdmatrix_simple(d22, l21, u12);
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

	sub_tdmatrix_partial(a, index, a, index, d22, d22_index);

	return 0;
}

#ifdef _OPENMP
int TDLUdecomp_a22_omp(TDMatrix a, TDMatrix d22, TDMatrix l21, TDMatrix u12, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim, index[4], d22_index[4];

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// d22 := l21 * u12
#ifdef USE_BLOCK
//	mul_tdmatrix_block(d22, l21, u12, STRASSEN_MIN_DIM);
	_bncomp_mul_tdmatrix_block(d22, l21, u12, STRASSEN_MIN_DIM);
//	_bncomp_mul_tdmatrix_block_old(d22, l21, u12, STRASSEN_MIN_DIM);
#elif USE_STRASSEN
//	mul_tdmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM);
	_bncomp_mul_tdmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM);
#elif USE_WINOGRAD
//	mul_tdmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM);
	_bncomp_mul_tdmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM);
#else
	_bncomp_mul_tdmatrix_simple(d22, l21, u12);
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

	//sub_tdmatrix_partial(a, index, a, index, d22, d22_index);
	_bncomp_sub_tdmatrix_partial(a, index, a, index, d22, d22_index);

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
int TDLUdecomp_strassen(TDMatrix a, long int min_dim)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       TDMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, row_dim, col_dim, dim;
	TDMatrix l21, u12, d22;

	dim = a->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		TDLUdecomp(a);
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
				TDLUdecomp_square(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		TDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_tdmatrix(a);

		l21 = init_tdmatrix(row_dim, min_dim);
		u12 = init_tdmatrix(min_dim, col_dim);
		d22 = init_tdmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		TDLUdecomp_l21(l21, a, i, min_dim);
		//print_tdmatrix(a);
		//printf("L21:\n");
		//print_tdmatrix(l21);

		// (3) Solve L11 * U12 = A12
		TDLUdecomp_u12(u12, a, i, min_dim);
		//print_tdmatrix(a);
		//printf("U12:\n");
		//print_tdmatrix(u12);

		//printf("A:\n");
		//print_tdmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		TDLUdecomp_a22(a, d22, l21, u12, i, min_dim);
		//printf("D22:\n");
		//print_tdmatrix(d22);
		//print_tdmatrix(a);

		// clear
		free_tdmatrix(l21);
		free_tdmatrix(u12);
		free_tdmatrix(d22);

	}

	return 0;
}


/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                (Triple double Precision) */
/*                                                          */
/*                 ver. 0.0 2015-02-25 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int TDLUdecomp_strassenPM(TDMatrix a, long int ch[], long int min_dim)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       TDMatrix a: Matrix (given by user)                 */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, row_dim, col_dim, dim, imax, itmp;
	TDMatrix l21, u12, d22;
	static double dtmp[TDSIZE], dtmp1[TDSIZE], dmaxii[TDSIZE];

	dim = a->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		TDLUdecompPM(a, ch);
		return 0;
	}

	for(i = 0; i < dim; i++)
		ch[i] = i;

	// dim > min_dim
	for(i = 0; i < dim; i += min_dim)
	{
		// partial pivoting
		rtd_abs(dmaxii, get_tdmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rtd_abs(dtmp, get_tdmatrix_ij(a, j, i));
			if(rtd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rtd_set(dmaxii, dtmp);
			}
		}

		if(rtd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! TDLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			row_swap_tdmatrix(a, i, imax, 0, a->col_dim);
		}

		// Initialize
		row_dim = dim - (i + min_dim);
		if(row_dim <= 0) 
		{
			// (1) L11 * U11 = A11
			if((dim - i) > 0)
				TDLUdecomp_square(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		TDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_tdmatrix(a);

		l21 = init_tdmatrix(row_dim, min_dim);
		u12 = init_tdmatrix(min_dim, col_dim);
		d22 = init_tdmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		TDLUdecomp_l21(l21, a, i, min_dim);
		//print_tdmatrix(a);
		//printf("L21:\n");
		//print_tdmatrix(l21);

		// (3) Solve L11 * U12 = A12
		TDLUdecomp_u12(u12, a, i, min_dim);
		//print_tdmatrix(a);
		//printf("U12:\n");
		//print_tdmatrix(u12);

		//printf("A:\n");
		//print_tdmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		TDLUdecomp_a22(a, d22, l21, u12, i, min_dim);
		//printf("D22:\n");
		//print_tdmatrix(d22);
		//print_tdmatrix(a);

		// clear
		free_tdmatrix(l21);
		free_tdmatrix(u12);
		free_tdmatrix(d22);

	}

	return 0;
}


#ifdef _OPENMP
/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                  (td_real)               */
/*                                                          */
/*                 ver. 0.0 2015-07-07 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
int TDLUdecomp_omp(TDMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*        TDMatrix a: Matrix (given by user)                */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
#ifdef __cplusplus
	td_real tmp[128], axii;
#else // __cplusplus
	double tmp[128][TDSIZE], axii[TDSIZE];
#endif // __cplusplus
	int thread_index;

	dim = a->col_dim;
	set0_td(axii);

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_td(tmp[thread_index]);
	}

	for(i = 0; i < dim; i++)
	{
		rtd_abs(axii, get_tdmatrix_ij(a, i, i));

		if(rtd_cmp_ui(axii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (TDLUdecomp_omp)!\n", i);
			return -1;
		}

		#pragma omp parallel for private(thread_index)
		for(j = (i + 1); j < dim; j++)
		{
			thread_index = omp_get_thread_num();

			rtd_div(tmp[thread_index], get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, i));
			set_tdmatrix_ij(a, j, i, tmp[thread_index]);
		}

		#pragma omp parallel for private(thread_index, k)
		for(j = (i + 1); j < dim; j++)
		{
			thread_index = omp_get_thread_num();

			for(k = (i + 1); k < dim; k++)
			{
				rtd_mul(tmp[thread_index], get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, k));
				rtd_sub(tmp[thread_index], get_tdmatrix_ij(a, j, k), tmp[thread_index]);
				set_tdmatrix_ij(a, j, k, tmp[thread_index]);
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
int TDLUdecompPM_omp(TDMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*        TDMatrix a: Matrix (given by user)                */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	double tmp[128][TDSIZE], axii[TDSIZE];
// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m256d dtmp256[128][TDSIZE], aji256[128][TDSIZE], ajk256[128][TDSIZE], aik256[128][TDSIZE];
#elif defined(__AVX512F__) // __AVX512F__
#endif // __AVX2__
	int thread_index;

	dim = a->col_dim;
	set0_td(axii);

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_td(tmp[thread_index]);
	}

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		// partial pivoting
		rtd_abs(axii, get_tdmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rtd_abs(tmp[0], get_tdmatrix_ij(a, j, i));
			if(rtd_cmp(tmp[0], axii) > 0)
			{
				imax = j;
				rtd_set(axii, tmp[0]);
			}
		}

		if(rtd_cmp_ui(axii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! TDLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			_bncomp_row_swap_tdmatrix(a, i, imax, 0, a->col_dim);
		}

		#pragma omp parallel for private(thread_index)
		for(j = (i + 1); j < dim; j++)
		{
			thread_index = omp_get_thread_num();

			rtd_div(tmp[thread_index], get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, i));
			set_tdmatrix_ij(a, j, i, tmp[thread_index]);
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
			aji256[thread_index][2] = _mm256_set_pd(
                a->element[2][index_ji],
                a->element[2][index_ji],
                a->element[2][index_ji],
                a->element[2][index_ji]
            );

			// head
			//printf("start j, k= %ld, %ld, ", j, i + 1);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			//for(k = (i + 1); k < dim; k++)
			{
				rtd_mul(tmp[thread_index], get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, k));
				rtd_sub(tmp[thread_index], get_tdmatrix_ij(a, j, k), tmp[thread_index]);
				set_tdmatrix_ij(a, j, k, tmp[thread_index]);
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
				aik256[thread_index][2] = _mm256_load_pd(&(a->element[2][index_ik]));
				_bncavx2_rtd_mul(dtmp256[thread_index], aji256[thread_index], aik256[thread_index]);
				//printf(" -- mul -- ");

				index_jk = j * a->real_col_dim + k;
				//rdd_sub(dtmp, get_ddmatrix_ij(a, j, k), dtmp1);
				ajk256[thread_index][0] = _mm256_load_pd(&(a->element[0][index_jk]));
				ajk256[thread_index][1] = _mm256_load_pd(&(a->element[1][index_jk]));
				ajk256[thread_index][2] = _mm256_load_pd(&(a->element[2][index_jk]));
				_bncavx2_rtd_sub(dtmp256[thread_index], ajk256[thread_index], dtmp256[thread_index]);
				//printf(" -- sub -- ");

				//set_ddmatrix_ij(a, j, k, dtmp);
				_mm256_store_pd(&(a->element[0][index_jk]), dtmp256[thread_index][0]);
				_mm256_store_pd(&(a->element[1][index_jk]), dtmp256[thread_index][1]);
				_mm256_store_pd(&(a->element[2][index_jk]), dtmp256[thread_index][2]);
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
				rtd_mul(tmp[thread_index], get_tdmatrix_ij(a, j, i), get_tdmatrix_ij(a, i, k));
				rtd_sub(tmp[thread_index], get_tdmatrix_ij(a, j, k), tmp[thread_index]);
				set_tdmatrix_ij(a, j, k, tmp[thread_index]);
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
int TDLUdecomp_strassen_omp(TDMatrix a, long int min_dim)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       TDMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, row_dim, col_dim, dim;
	TDMatrix l21, u12, d22;

	dim = a->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		TDLUdecomp_omp(a);
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
				TDLUdecomp_square_omp(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		TDLUdecomp_square_omp(a, i, min_dim);
		//TDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_tdmatrix(a);

		l21 = init_tdmatrix(row_dim, min_dim);
		u12 = init_tdmatrix(min_dim, col_dim);
		d22 = init_tdmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		TDLUdecomp_l21_omp(l21, a, i, min_dim);
		//TDLUdecomp_l21(l21, a, i, min_dim);
		//print_tdmatrix(a);
		//printf("L21:\n");
		//print_tdmatrix(l21);

		// (3) Solve L11 * U12 = A12
		TDLUdecomp_u12_omp(u12, a, i, min_dim);
		//TDLUdecomp_u12(u12, a, i, min_dim);
		//print_tdmatrix(a);
		//printf("U12:\n");
		//print_tdmatrix(u12);

		//printf("A:\n");
		//print_tdmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		TDLUdecomp_a22_omp(a, d22, l21, u12, i, min_dim);
		//TDLUdecomp_a22(a, d22, l21, u12, i, min_dim);
		//printf("D22:\n");
		//print_tdmatrix(d22);
		//print_tdmatrix(a);

		// clear
		free_tdmatrix(l21);
		free_tdmatrix(u12);
		free_tdmatrix(d22);

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
int TDLUdecomp_strassenPM_omp(TDMatrix a, long int ch[], long int min_dim)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       TDMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, row_dim, col_dim, dim, imax, itmp;
	TDMatrix l21, u12, d22;
	double dtmp[TDSIZE], dmaxii[TDSIZE];

	dim = a->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		TDLUdecompPM_omp(a, ch);
		return 0;
	}

	for(i = 0; i < dim; i++)
		ch[i] = i;

	// dim > min_dim
	for(i = 0; i < dim; i += min_dim)
	{
		// partial pivoting
		rtd_abs(dmaxii, get_tdmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rtd_abs(dtmp, get_tdmatrix_ij(a, j, i));
			if(rtd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rtd_set(dmaxii, dtmp);
			}
		}

		if(rtd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! TDLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			_bncomp_row_swap_tdmatrix(a, i, imax, 0, a->col_dim);
		}

		// Initialize
		row_dim = dim - (i + min_dim);
		if(row_dim <= 0) 
		{
			// (1) L11 * U11 = A11
			if((dim - i) > 0)
				TDLUdecomp_square_omp(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		TDLUdecomp_square_omp(a, i, min_dim);
		//TDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_tdmatrix(a);

		l21 = init_tdmatrix(row_dim, min_dim);
		u12 = init_tdmatrix(min_dim, col_dim);
		d22 = init_tdmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		TDLUdecomp_l21_omp(l21, a, i, min_dim);
		//TDLUdecomp_l21(l21, a, i, min_dim);
		//print_tdmatrix(a);
		//printf("L21:\n");
		//print_tdmatrix(l21);

		// (3) Solve L11 * U12 = A12
		TDLUdecomp_u12_omp(u12, a, i, min_dim);
		//TDLUdecomp_u12(u12, a, i, min_dim);
		//print_tdmatrix(a);
		//printf("U12:\n");
		//print_tdmatrix(u12);

		//printf("A:\n");
		//print_tdmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		TDLUdecomp_a22_omp(a, d22, l21, u12, i, min_dim);
		//TDLUdecomp_a22(a, d22, l21, u12, i, min_dim);
		//printf("D22:\n");
		//print_tdmatrix(d22);
		//print_tdmatrix(a);

		// clear
		free_tdmatrix(l21);
		free_tdmatrix(u12);
		free_tdmatrix(d22);

	}

	return 0;
}
#endif // _OPENMP