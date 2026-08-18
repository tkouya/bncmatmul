/********************************************************************************/
/* mpflu_strassen.c:                                                            */
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

#ifdef USE_IMKL
	#include "mkl.h"
	#include "mkl_cblas.h" // for Intel Math Kernel Library
#endif

#include "bncomp.h"
#include "matmul_strassen.h"

//#include "lu_bench.h"
#include "get_secv.h"

#ifdef USE_GMP

//#include "serial_lu_bench.c"

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
int MPFLUdecomp_square(MPFMatrix a, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim;
	mpf_t dtmp, dmaxii;

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	mpf_init2(dtmp, a->prec);
	mpf_init2(dmaxii, a->prec);

	// dim > min_dim
	for(i = start_index; i < imax; i++)
	{
		mpf_abs(dmaxii, get_mpfmatrix_ij(a, i, i));
		if(mpf_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (MPFLUdecomp_square)!\n", i);
			mpf_clear(dtmp);
			mpf_clear(dmaxii);
			return -1;
		}

		for(j = (i + 1); j < jmax; j++)
		{
			mpf_div(dtmp, get_mpfmatrix_ij(a, j, i), get_mpfmatrix_ij(a, i, i));
			set_mpfmatrix_ij(a, j, i, dtmp);
		}

		for(j = (i + 1); j < jmax; j++)
		{
			for(k = (i + 1); k < jmax; k++)
			{
				mpf_mul(dtmp, get_mpfmatrix_ij(a, j, i), get_mpfmatrix_ij(a, i, k));
				mpf_sub(dtmp, get_mpfmatrix_ij(a, j, k), dtmp);
				set_mpfmatrix_ij(a, j, k, dtmp);
			}
		}
	}

	mpf_clear(dtmp);
	mpf_clear(dmaxii);

	return 0;
}
#ifdef _OPENMP
int MPFLUdecomp_square_omp(MPFMatrix a, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim;
	mpf_t dtmp[128], dmaxii;
	int thread_index;

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(dtmp[thread_index], a->prec);
		//mpf_init2(dmaxii[thread_index], a->prec);
	}
	mpf_init2(dmaxii, a->prec);

	// dim > min_dim
	for(i = start_index; i < imax; i++)
	{
		mpf_abs(dmaxii, get_mpfmatrix_ij(a, i, i));
		if(mpf_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld (start_index = %ld, min_dim = %ld): Error! (MPFLUdecomp_square)!\n", i, start_index, min_dim);
			mpf_clear(dmaxii);
			#pragma omp parallel private(thread_index)
			{
				thread_index = omp_get_thread_num();
				mpf_clear(dtmp[thread_index]);
			}
			return -1;
		}

		#pragma omp parallel for private(thread_index)
		for(j = (i + 1); j < jmax; j++)
		{
			thread_index = omp_get_thread_num();

			mpf_div(dtmp[thread_index], get_mpfmatrix_ij(a, j, i), get_mpfmatrix_ij(a, i, i));
			set_mpfmatrix_ij(a, j, i, dtmp[thread_index]);
		}

		#pragma omp parallel for private(thread_index, k)
		for(j = (i + 1); j < jmax; j++)
		{
			thread_index = omp_get_thread_num();

			for(k = (i + 1); k < jmax; k++)
			{
				mpf_mul(dtmp[thread_index], get_mpfmatrix_ij(a, j, i), get_mpfmatrix_ij(a, i, k));
				mpf_sub(dtmp[thread_index], get_mpfmatrix_ij(a, j, k), dtmp[thread_index]);
				set_mpfmatrix_ij(a, j, k, dtmp[thread_index]);
			}
		}
	}

	mpf_clear(dmaxii);
	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(dtmp[thread_index]);
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
int MPFLUdecomp_l21(MPFMatrix l21, MPFMatrix a, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim;
	mpf_t dtmp, dmaxii;

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	mpf_init2(dmaxii, l21->prec);
	mpf_init2(dtmp, l21->prec);

	// dim > min_dim
	for(i = imax; i < dim; i++)
	{
		mpf_abs(dmaxii, get_mpfmatrix_ij(a, i, i));
		if(mpf_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (MPFLUdecomp_l21)!\n", i);
			mpf_clear(dtmp);
			mpf_clear(dmaxii);
			return -1;
		}

		for(j = start_index; j < jmax; j++)
		{
			mpf_set(dtmp, get_mpfmatrix_ij(a, i, j));
			for(k = start_index; k < j; k++)
			{
				//dtmp -= get_dmatrix_ij(a, i, k) * get_dmatrix_ij(a, k, j);
				mpf_mul(dmaxii, get_mpfmatrix_ij(a, i, k), get_mpfmatrix_ij(a, k, j));
				mpf_sub(dtmp, dtmp, dmaxii);
			}
			mpf_div(dtmp, dtmp, get_mpfmatrix_ij(a, j, j));
			
			//printf("(i - start_index, j - imax) = (%ld, %ld), %ld, %ld\n", i - imax, j - start_index, l21->row_dim, l21->col_dim);
			//printf("(i              , j       ) = (%ld, %ld) %25.17e, %25.17e\n", i, j, dtmp, get_dmatrix_ij(a, j, j));
			set_mpfmatrix_ij(a  , i              , j       , dtmp);
			set_mpfmatrix_ij(l21, i - imax, j - start_index, dtmp);
		}
	}

	mpf_clear(dtmp);
	mpf_clear(dmaxii);

	return 0;
}

#ifdef _OPENMP
int MPFLUdecomp_l21_omp(MPFMatrix l21, MPFMatrix a, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim;
	mpf_t dtmp[128], dtmp1[128], dmaxii[128];
	int thread_index;

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	//mpf_init2(dmaxii, l21->prec);
	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();

		mpf_init2(dtmp[thread_index], l21->prec);
		mpf_init2(dtmp1[thread_index], l21->prec);
		mpf_init2(dmaxii[thread_index], l21->prec);
	}

	// dim > min_dim
	#pragma omp parallel for private(thread_index, j, k)
	for(i = imax; i < dim; i++)
	{
		thread_index = omp_get_thread_num();

		mpf_abs(dmaxii[thread_index], get_mpfmatrix_ij(a, i, i));
		if(mpf_cmp_ui(dmaxii[thread_index], 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (MPFLUdecomp_l21_omp)!\n", i);
			//mpf_clear(dmaxii);
			#pragma omp parallel private(thread_index)
			{
				thread_index = omp_get_thread_num();
				mpf_clear(dtmp[thread_index]);
 				mpf_clear(dtmp1[thread_index]);
				mpf_clear(dmaxii[thread_index]);
			}
			//return -1;
		}

		//#pragma omp parallel for private(thread_index, k)
		for(j = start_index; j < jmax; j++)
		{
			//thread_index = omp_get_thread_num();

			mpf_set(dtmp[thread_index], get_mpfmatrix_ij(a, i, j));
			for(k = start_index; k < j; k++)
			{
				//thread_index = omp_get_thread_num();

				//dtmp -= get_dmatrix_ij(a, i, k) * get_dmatrix_ij(a, k, j);
				mpf_mul(dtmp1[thread_index], get_mpfmatrix_ij(a, i, k), get_mpfmatrix_ij(a, k, j));
				mpf_sub(dtmp[thread_index], dtmp[thread_index], dtmp1[thread_index]);
			}
			mpf_div(dtmp[thread_index], dtmp[thread_index], get_mpfmatrix_ij(a, j, j));
			
			//printf("(i - start_index, j - imax) = (%ld, %ld), %ld, %ld\n", i - imax, j - start_index, l21->row_dim, l21->col_dim);
			//printf("(i              , j       ) = (%ld, %ld) %25.17e, %25.17e\n", i, j, dtmp, get_dmatrix_ij(a, j, j));
			set_mpfmatrix_ij(a  , i              , j       , dtmp[thread_index]);
			set_mpfmatrix_ij(l21, i - imax, j - start_index, dtmp[thread_index]);
		}
	}

	//mpf_clear(dmaxii);
	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(dtmp[thread_index]);
		mpf_clear(dtmp1[thread_index]);
		mpf_clear(dmaxii[thread_index]);
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
int MPFLUdecomp_u12(MPFMatrix u12, MPFMatrix a, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim;
	mpf_t dtmp, dmaxii;

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	mpf_init2(dmaxii, u12->prec);
	mpf_init2(dtmp, u12->prec);

	// dim > min_dim
	for(i = imax; i < dim; i++)
	{
		mpf_abs(dmaxii, get_mpfmatrix_ij(a, i, i));
		if(mpf_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (MPFLUdecomp_u12)!\n", i);
			mpf_clear(dtmp);
			mpf_clear(dmaxii);
			return -1;
		}

		for(j = start_index; j < jmax; j++)
		{
			mpf_set(dtmp, get_mpfmatrix_ij(a, j, i));
			//printf("(j              , i              ) = (%ld, %ld) %25.17e\n", j, i , dtmp);
			for(k = start_index; k < j; k++)
			{
				//dtmp -= get_dmatrix_ij(a, j, k) * get_dmatrix_ij(a, k, i);
				mpf_mul(dmaxii, get_mpfmatrix_ij(a, j, k), get_mpfmatrix_ij(a, k, i));
				mpf_sub(dtmp, dtmp, dmaxii);
				//printf("(j - start_index, k - start_index) = (%ld, %ld)\n", j - start_index, k - start_index);
			}
			//printf("(j              , i              ) = (%ld, %ld) %25.17e\n", j, i , dtmp);
			set_mpfmatrix_ij(a  , j              , i       , dtmp);
			set_mpfmatrix_ij(u12, j - start_index, i - imax, dtmp);
		}
	}

	mpf_clear(dtmp);
	mpf_clear(dmaxii);

	return 0;
}
#ifdef _OPENMP
int MPFLUdecomp_u12_omp(MPFMatrix u12, MPFMatrix a, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim;
	mpf_t dtmp[128], dtmp1[128], dmaxii[128];
	int thread_index;

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	//mpf_init2(dmaxii, u12->prec);
	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(dtmp[thread_index], a->prec);
		mpf_init2(dtmp1[thread_index], a->prec);
		mpf_init2(dmaxii[thread_index], a->prec);
	}

	// dim > min_dim
	#pragma omp parallel for private(thread_index, j, k)
	for(i = imax; i < dim; i++)
	{
		thread_index = omp_get_thread_num();

		mpf_abs(dmaxii[thread_index], get_mpfmatrix_ij(a, i, i));
		if(mpf_cmp_ui(dmaxii[thread_index], 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (MPFLUdecomp_u12_omp)!\n", i);
			//mpf_clear(dmaxii);
			#pragma omp parallel private(thread_index)
			{
				thread_index = omp_get_thread_num();
				mpf_clear(dtmp[thread_index]);
				mpf_clear(dtmp1[thread_index]);
				mpf_clear(dmaxii[thread_index]);
			}
			//return -1;
		}

		//#pragma omp parallel for private(thread_index, k)
		for(j = start_index; j < jmax; j++)
		{
			//thread_index = omp_get_thread_num();

			mpf_set(dtmp[thread_index], get_mpfmatrix_ij(a, j, i));
			//printf("(j              , i              ) = (%ld, %ld) %25.17e\n", j, i , dtmp);
			for(k = start_index; k < j; k++)
			{
				//dtmp -= get_dmatrix_ij(a, j, k) * get_dmatrix_ij(a, k, i);
				mpf_mul(dtmp1[thread_index], get_mpfmatrix_ij(a, j, k), get_mpfmatrix_ij(a, k, i));
				mpf_sub(dtmp[thread_index], dtmp[thread_index], dtmp1[thread_index]);
				//printf("(j - start_index, k - start_index) = (%ld, %ld)\n", j - start_index, k - start_index);
			}
			//printf("(j              , i              ) = (%ld, %ld) %25.17e\n", j, i , dtmp);
			set_mpfmatrix_ij(a  , j              , i       , dtmp[thread_index]);
			set_mpfmatrix_ij(u12, j - start_index, i - imax, dtmp[thread_index]);
		}
	}

	//mpf_clear(dmaxii);
	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(dtmp[thread_index]);
		mpf_clear(dtmp1[thread_index]);
		mpf_clear(dmaxii[thread_index]);
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
int MPFLUdecomp_a22(MPFMatrix a, MPFMatrix d22, MPFMatrix l21, MPFMatrix u12, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim, index[4], d22_index[4];
	MPFVector diag_left, diag_right;

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// d22 := l21 * u12
#if defined(USE_STRASSEN) || defined(USE_WINOGRAD)
#ifdef USE_SCALING
	diag_left = init2_mpfvector(l21->row_dim, d22->prec);
	diag_right = init2_mpfvector(u12->col_dim, d22->prec);

	left_scaling_mpfmatrix(l21, l21, diag_left, NULL);
	right_scaling_mpfmatrix(u12, u12, diag_right, NULL);
#endif

//	mul_mpfmatrix_strassen(d22, l21, u12, min_dim);
//	mul_mpfmatrix_strassen(d22, l21, u12, min_dim / 2);
//	mul_mpfmatrix_strassen(d22, l21, u12, min_dim / 4);
	mul_mpfmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM);

#ifdef USE_SCALING
	mul_mpfmatrix_mpfdiag(d22, diag_left, 0, d22, diag_right, 0); 

	mul_mpfmatrix_mpfdiag(l21, diag_left, 1, l21, NULL, 0);
	mul_mpfmatrix_mpfdiag(u12, NULL, 0, u12, diag_right, 1);

	free_mpfvector(diag_left);
	free_mpfvector(diag_right);
#endif

#elif USE_BLOCK
	mul_mpfmatrix_block(d22, l21, u12, STRASSEN_MIN_DIM);
#else
	mul_mpfmatrix_simple(d22, l21, u12);
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

	sub_mpfmatrix_partial(a, index, a, index, d22, d22_index);

	return 0;
}
#ifdef _OPENMP
int MPFLUdecomp_a22_omp(MPFMatrix a, MPFMatrix d22, MPFMatrix l21, MPFMatrix u12, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim, index[4], d22_index[4];
	MPFVector diag_left, diag_right;

	dim = a->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// d22 := l21 * u12
#if defined(USE_STRASSEN) || defined(USE_WINOGRAD)
#ifdef USE_SCALING
	diag_left = init2_mpfvector(l21->row_dim, d22->prec);
	diag_right = init2_mpfvector(u12->col_dim, d22->prec);

	left_scaling_mpfmatrix(l21, l21, diag_left, NULL);
	right_scaling_mpfmatrix(u12, u12, diag_right, NULL);
#endif

//	mul_mpfmatrix_strassen(d22, l21, u12, min_dim);
//	mul_mpfmatrix_strassen(d22, l21, u12, min_dim / 2);
//	mul_mpfmatrix_strassen(d22, l21, u12, min_dim / 4);
//	mul_mpfmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM);
	_bncomp_mul_mpfmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM);

#ifdef USE_SCALING
	mul_mpfmatrix_mpfdiag(d22, diag_left, 0, d22, diag_right, 0); 

	mul_mpfmatrix_mpfdiag(l21, diag_left, 1, l21, NULL, 0);
	mul_mpfmatrix_mpfdiag(u12, NULL, 0, u12, diag_right, 1);

	free_mpfvector(diag_left);
	free_mpfvector(diag_right);
#endif

#elif USE_BLOCK
	//mul_mpfmatrix_block(d22, l21, u12, STRASSEN_MIN_DIM);
	_bncomp_mul_mpfmatrix_block(d22, l21, u12, STRASSEN_MIN_DIM);
#else
	_bncomp_mul_mpfmatrix_simple(d22, l21, u12);
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

	//sub_mpfmatrix_partial(a, index, a, index, d22, d22_index);
	_bncomp_sub_mpfmatrix_partial(a, index, a, index, d22, d22_index);

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
/*       MPFMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
int MPFLUdecomp_strassen(MPFMatrix a, long int min_dim)
{
	long int i, row_dim, col_dim, dim;
	MPFMatrix l21, u12, d22;

	dim = a->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		MPFLUdecomp(a);
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
				MPFLUdecomp_square(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		MPFLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_dmatrix(a);

/*		l21 = init2_mpfmatrix(row_dim, min_dim, a->prec + (a->prec / 4));
		u12 = init2_mpfmatrix(min_dim, col_dim, a->prec + (a->prec / 4));
		d22 = init2_mpfmatrix(row_dim, col_dim, a->prec + (a->prec / 4));
*/		l21 = init2_mpfmatrix(row_dim, min_dim, a->prec);
		u12 = init2_mpfmatrix(min_dim, col_dim, a->prec);
		d22 = init2_mpfmatrix(row_dim, col_dim, a->prec);

		// (2) Solve L21 * U11 = A21
		MPFLUdecomp_l21(l21, a, i, min_dim);
		//print_dmatrix(a);
		//printf("L21:\n");
		//print_dmatrix(l21);

		// (3) Solve L11 * U12 = A12
		MPFLUdecomp_u12(u12, a, i, min_dim);
		//print_dmatrix(a);
		//printf("U12:\n");
		//print_dmatrix(u12);

		//printf("A:\n");
		//print_dmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		MPFLUdecomp_a22(a, d22, l21, u12, i, min_dim);
		//printf("D22:\n");
		//print_dmatrix(d22);
		//print_dmatrix(a);

		// clear
		free_mpfmatrix(l21);
		free_mpfmatrix(u12);
		free_mpfmatrix(d22);

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
/*       MPFMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
int MPFLUdecomp_strassenPM(MPFMatrix a, long int ch[], long int min_dim)
{
	long int i, j, row_dim, col_dim, dim, imax, itmp;
	MPFMatrix l21, u12, d22;
	mpf_t tmp, axii;
	int flag = 1;

	// bug case
	if((a->row_dim == 1024) && (min_dim == 32)) flag = 1;

	dim = a->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		MPFLUdecompPM(a, ch);
		return 0;
	}

	mpf_init2(tmp, a->prec);
	mpf_init2(axii, a->prec);
	for(i = 0; i < dim; i++)
		ch[i] = i;

	// dim > min_dim
	for(i = 0; i < dim; i += min_dim)
	{
		// partial pivoting
		mpf_abs(axii, get_mpfmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			mpf_abs(tmp, get_mpfmatrix_ij(a, j, i));
			if(mpf_cmp(tmp, axii) > 0)
			{
				imax = j;
				mpf_set(axii, tmp);
			}
		}

		if(mpf_cmp_ui(axii, 0UL) == 0)
		{
			mpf_clear(tmp);
			mpf_clear(axii);
			fprintf(stderr, "%ld : Error! MPFLUdecompP!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			//printf("row_swap_mpfmatrix run! %ld <-> %ld\n", ch[i], ch[imax]);
			row_swap_mpfmatrix(a, i, imax, 0, a->col_dim); // Fix! 2022-04-11(Mon) T.Kouya
			//row_swap_mpfmatrix(a, ch[i], ch[imax], 0, a->col_dim);
		}

		// Initialize
		row_dim = dim - (i + min_dim);
		if(row_dim <= 0) 
		{
			//if(flag == 1) printf("row_dim = %d!\n", row_dim);
			// (1) L11 * U11 = A11
			if((dim - i) > 0)
				MPFLUdecomp_square(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		MPFLUdecomp_square(a, i, min_dim);

		//if(flag == 1) printf("i = %ld\n", i);
		//print_dmatrix(a);

/*		l21 = init2_mpfmatrix(row_dim, min_dim, a->prec + (a->prec / 4));
		u12 = init2_mpfmatrix(min_dim, col_dim, a->prec + (a->prec / 4));
		d22 = init2_mpfmatrix(row_dim, col_dim, a->prec + (a->prec / 4));
*/		l21 = init2_mpfmatrix(row_dim, min_dim, a->prec);
		u12 = init2_mpfmatrix(min_dim, col_dim, a->prec);
		d22 = init2_mpfmatrix(row_dim, col_dim, a->prec);

		// (2) Solve L21 * U11 = A21
		MPFLUdecomp_l21(l21, a, i, min_dim);
		//print_dmatrix(a);
		//if(flag == 1) printf("L21:\n");
		//if(flag == 1) { normf_mpfmatrix(tmp, l21); mpf_out_str(stdout, 10, 0, tmp); printf("\n"); } //print_mpfmatrix(l21); }

		// (3) Solve L11 * U12 = A12
		MPFLUdecomp_u12(u12, a, i, min_dim);
		//print_dmatrix(a);
		//if(flag == 1) printf("U12:\n");
		//if(flag == 1) { normf_mpfmatrix(tmp, u12); mpf_out_str(stdout, 10, 0, tmp); printf("\n"); } //print_mpfmatrix(u12); }

		//printf("A:\n");
		//print_dmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		MPFLUdecomp_a22(a, d22, l21, u12, i, min_dim);
		//if(flag == 1) printf("D22:\n");
		//if(flag == 1) { normf_mpfmatrix(tmp, d22); mpf_out_str(stdout, 10, 0, tmp); printf("\n"); } //print_mpfmatrix(d22);
		//print_dmatrix(a);

		// clear
		free_mpfmatrix(l21);
		free_mpfmatrix(u12);
		free_mpfmatrix(d22);

	}

	mpf_clear(tmp);
	mpf_clear(axii);

	return 0;
}

#ifdef _OPENMP
int MPFLUdecomp_strassen_omp(MPFMatrix a, long int min_dim)
{
	long int i, row_dim, col_dim, dim;
	MPFMatrix l21, u12, d22;

	dim = a->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		//MPFLUdecomp(a);
		MPFLUdecomp_omp(a);
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
				MPFLUdecomp_square_omp(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		//MPFLUdecomp_square(a, i, min_dim);
		MPFLUdecomp_square_omp(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_dmatrix(a);

/*		l21 = init2_mpfmatrix(row_dim, min_dim, a->prec + (a->prec / 4));
		u12 = init2_mpfmatrix(min_dim, col_dim, a->prec + (a->prec / 4));
		d22 = init2_mpfmatrix(row_dim, col_dim, a->prec + (a->prec / 4));
*/		l21 = init2_mpfmatrix(row_dim, min_dim, a->prec);
		u12 = init2_mpfmatrix(min_dim, col_dim, a->prec);
		d22 = init2_mpfmatrix(row_dim, col_dim, a->prec);

		// (2) Solve L21 * U11 = A21
		//MPFLUdecomp_l21(l21, a, i, min_dim);
		MPFLUdecomp_l21_omp(l21, a, i, min_dim);
		//print_dmatrix(a);
		//printf("L21:\n");
		//print_dmatrix(l21);

		// (3) Solve L11 * U12 = A12
		//MPFLUdecomp_u12(u12, a, i, min_dim);
		MPFLUdecomp_u12_omp(u12, a, i, min_dim);
		//print_dmatrix(a);
		//printf("U12:\n");
		//print_dmatrix(u12);

		//printf("A:\n");
		//print_dmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		MPFLUdecomp_a22_omp(a, d22, l21, u12, i, min_dim);
		//printf("D22:\n");
		//print_dmatrix(d22);
		//print_dmatrix(a);

		// clear
		free_mpfmatrix(l21);
		free_mpfmatrix(u12);
		free_mpfmatrix(d22);

	}

	return 0;
}

int MPFLUdecomp_strassenPM_omp(MPFMatrix a, long int ch[], long int min_dim)
{
	long int i, j, row_dim, col_dim, dim, imax, itmp;
	MPFMatrix l21, u12, d22;
	mpf_t tmp, axii;

	dim = a->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		MPFLUdecompPM(a, ch);
		return 0;
	}

	mpf_init2(tmp, a->prec);
	mpf_init2(axii, a->prec);
	for(i = 0; i < dim; i++)
		ch[i] = i;

	// dim > min_dim
	for(i = 0; i < dim; i += min_dim)
	{
		// partial pivoting
		mpf_abs(axii, get_mpfmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			mpf_abs(tmp, get_mpfmatrix_ij(a, j, i));
			if(mpf_cmp(tmp, axii) > 0)
			{
				imax = j;
				mpf_set(axii, tmp);
			}
		}

		if(mpf_cmp_ui(axii, 0UL) == 0)
		{
			mpf_clear(tmp);
			mpf_clear(axii);
			fprintf(stderr, "%ld : Error! MPFLUdecompPM_omp!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			_bncomp_row_swap_mpfmatrix(a, i, imax, 0, a->col_dim); // Fix! 2022-04-12(Mon) T.Kouya
			//_bncomp_row_swap_mpfmatrix(a, ch[i], ch[imax], 0, a->col_dim);
		}

		// Initialize
		row_dim = dim - (i + min_dim);
		if(row_dim <= 0) 
		{
			// (1) L11 * U11 = A11
			if((dim - i) > 0)
				MPFLUdecomp_square_omp(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		//MPFLUdecomp_square(a, i, min_dim);
		MPFLUdecomp_square_omp(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_dmatrix(a);

/*		l21 = init2_mpfmatrix(row_dim, min_dim, a->prec + (a->prec / 4));
		u12 = init2_mpfmatrix(min_dim, col_dim, a->prec + (a->prec / 4));
		d22 = init2_mpfmatrix(row_dim, col_dim, a->prec + (a->prec / 4));
*/		l21 = init2_mpfmatrix(row_dim, min_dim, a->prec);
		u12 = init2_mpfmatrix(min_dim, col_dim, a->prec);
		d22 = init2_mpfmatrix(row_dim, col_dim, a->prec);

		// (2) Solve L21 * U11 = A21
		//MPFLUdecomp_l21(l21, a, i, min_dim);
		MPFLUdecomp_l21_omp(l21, a, i, min_dim);
		//print_dmatrix(a);
		//printf("L21:\n");
		//print_dmatrix(l21);

		// (3) Solve L11 * U12 = A12
		//MPFLUdecomp_u12(u12, a, i, min_dim);
		MPFLUdecomp_u12_omp(u12, a, i, min_dim);
		//print_dmatrix(a);
		//printf("U12:\n");
		//print_dmatrix(u12);

		//printf("A:\n");
		//print_dmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		MPFLUdecomp_a22_omp(a, d22, l21, u12, i, min_dim);
		//printf("D22:\n");
		//print_dmatrix(d22);
		//print_dmatrix(a);

		// clear
		free_mpfmatrix(l21);
		free_mpfmatrix(u12);
		free_mpfmatrix(d22);

	}

	mpf_clear(tmp);
	mpf_clear(axii);

	return 0;
}
#endif // _OPENMP

#ifdef _OPENMP
/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                  (Multi-Precision)       */
/*                                                          */
/*                 ver. 0.0 2000.02.28 (Mon) Tomonori Kouya */
/*                 ver. 0.1 2000.07.05 (Wed) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int MPFLUdecomp_omp(MPFMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       MPFMatrix a: Matrix (given by user)                */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	mpf_t tmp[128], axii;
	int thread_index;

	dim = a->col_dim;
	mpf_init2(axii, a->prec);
	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp[thread_index], a->prec);
	}

	for(i = 0; i < dim; i++)
	{
		mpf_abs(axii, get_mpfmatrix_ij(a, i, i));
		if(mpf_cmp_ui(axii, 0UL) == 0)
		{
			mpf_clear(axii);
			fprintf(stderr, "%ld : Error! (MPFLUdecomp)!\n", i);
			#pragma omp parallel private(thread_index)
			{
				thread_index = omp_get_thread_num();
				mpf_clear(tmp[thread_index]);
			}
			return -1;
		}

		#pragma omp parallel for private(thread_index)
		for(j = (i + 1); j < dim; j++)
		{
			thread_index = omp_get_thread_num();

			mpf_div(tmp[thread_index], get_mpfmatrix_ij(a, j, i), get_mpfmatrix_ij(a, i, i));
			set_mpfmatrix_ij(a, j, i, tmp[thread_index]);
		}

		#pragma omp parallel for private(thread_index, k)
		for(j = (i + 1); j < dim; j++)
		{
			thread_index = omp_get_thread_num();

			for(k = (i + 1); k < dim; k++)
			{
				mpf_mul(tmp[thread_index], get_mpfmatrix_ij(a, j, i), get_mpfmatrix_ij(a, i, k));
				mpf_sub(tmp[thread_index], get_mpfmatrix_ij(a, j, k), tmp[thread_index]);
				set_mpfmatrix_ij(a, j, k, tmp[thread_index]);
			}
		}
	}

	mpf_clear(axii);
	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp[thread_index]);
	}

	return 0;
}
#endif // _OPENMP

#endif // def USE_GMP