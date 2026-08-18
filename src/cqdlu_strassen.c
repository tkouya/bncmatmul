/********************************************************************************/
/* cqdlu_strassen.c: Complex Quad-double Precision LU decomposition             */
/*                                                             with Strassen MM */
/*                                                                              */
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

#include "cqdlinear.h"

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
int CQDLUdecomp_square(CQDMatrix a, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim;
	double dtmp[QDSIZE], dtmp1[QDSIZE], dmaxii[QDSIZE];
    cqdfloat ctmp, ctmp1;

	dim = a->re->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// dim > min_dim
	for(i = start_index; i < imax; i++)
	{
		rcqd_abs_qd(dmaxii, get_cqdmatrix_ij(a, i, i));
		if(rqd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (CQDLUdecomp_square)!\n", i);
			return -1;
		}

		for(j = (i + 1); j < jmax; j++)
		{
			rcqd_div(&ctmp, get_cqdmatrix_ij(a, j, i), get_cqdmatrix_ij(a, i, i));
			set_cqdmatrix_ij(a, j, i, &ctmp);
		}

		for(j = (i + 1); j < jmax; j++)
		{
			for(k = (i + 1); k < jmax; k++)
			{
				//get_dmatrix_ij(a, j, k) - get_dmatrix_ij(a, j, i) * get_dmatrix_ij(a, i, k)
				rcqd_mul(&ctmp1, get_cqdmatrix_ij(a, j, i), get_cqdmatrix_ij(a, i, k));
				rcqd_sub(&ctmp, get_cqdmatrix_ij(a, j, k), &ctmp1);
				set_cqdmatrix_ij(a, j, k, &ctmp);
			}
		}
	}

	return 0;
}

#ifdef USE_OMP
int CQDLUdecomp_square_omp(CQDMatrix a, long int start_index, long int min_dim)
{
	int thread_index, thread_num;
	long int i, j, k, imax, jmax, itmp, dim;
	double dtmp[128][QDSIZE], dtmp1[128][QDSIZE], dmaxii[QDSIZE];
    cqdfloat ctmp[128], ctmp1[128];

	dim = a->re->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();

		rcqd_set0(&ctmp[thread_index]);
		rcqd_set0(&ctmp1[thread_index]);
	}

	set0_qd(dmaxii);

	// dim > min_dim
	for(i = start_index; i < imax; i++)
	{
		rcqd_abs_qd(dmaxii, get_cqdmatrix_ij(a, i, i));
		if(rqd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (CQDLUdecomp_square_omp)!\n", i);
			return -1;
		}

		#pragma omp parallel for private(thread_index)
		for(j = (i + 1); j < jmax; j++)
		{
			thread_index = omp_get_thread_num();

			rcqd_div(&ctmp[thread_index], get_cqdmatrix_ij(a, j, i), get_cqdmatrix_ij(a, i, i));
			set_cqdmatrix_ij(a, j, i, &ctmp[thread_index]);
		}

		#pragma omp parallel for private(thread_index, k)
		for(j = (i + 1); j < jmax; j++)
		{
			thread_index = omp_get_thread_num();

			for(k = (i + 1); k < jmax; k++)
			{
				//get_dmatrix_ij(a, j, k) - get_dmatrix_ij(a, j, i) * get_dmatrix_ij(a, i, k)
				rcqd_mul(&ctmp1[thread_index], get_cqdmatrix_ij(a, j, i), get_cqdmatrix_ij(a, i, k));
				rcqd_sub(&ctmp[thread_index], get_cqdmatrix_ij(a, j, k), &ctmp1[thread_index]);
				set_cqdmatrix_ij(a, j, k, &ctmp[thread_index]);
			}
		}
	}

	return 0;
}
#endif // USE_OMP

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
int CQDLUdecomp_l21(CQDMatrix l21, CQDMatrix a, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim;
	double dtmp[QDSIZE], dtmp1[QDSIZE], dmaxii[QDSIZE];
    cqdfloat ctmp, ctmp1;

	dim = a->re->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// dim > min_dim
	for(i = imax; i < dim; i++)
	{
		rcqd_abs_qd(dmaxii, get_cqdmatrix_ij(a, i, i));
		if(rqd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (CQDLUdecomp_l21)!\n", i);
			return -1;
		}

		for(j = start_index; j < jmax; j++)
		{
			rcqd_set(&ctmp, get_cqdmatrix_ij(a, i, j));
			for(k = start_index; k < j; k++)
			{
				//dtmp -= get_dmatrix_ij(a, i, k) * get_dmatrix_ij(a, k, j);
				rcqd_mul(&ctmp1, get_cqdmatrix_ij(a, i, k), get_cqdmatrix_ij(a, k, j));
				rcqd_sub(&ctmp, &ctmp, &ctmp1);
			}
			rcqd_div(&ctmp1, &ctmp, get_cqdmatrix_ij(a, j, j));
			rcqd_set(&ctmp, &ctmp1);
			
			//printf("l21(i              , j       ) = (%ld, %ld), %ld, %ld\n", i, j, l21->row_dim, l21->col_dim);
			//printf("l21(i - start_index, j - imax) = (%ld, %ld), %ld, %ld\n", i - imax, j - start_index, l21->row_dim, l21->col_dim);
			//printf("(i              , j       ) = (%ld, %ld) %25.17e, %25.17e\n", i, j, dtmp, get_dmatrix_ij(a, j, j));
			set_cqdmatrix_ij(a  , i              , j       , &ctmp);
			set_cqdmatrix_ij(l21, i - imax, j - start_index, &ctmp);
		}
	}

	return 0;
}

#ifdef USE_OMP
int CQDLUdecomp_l21_omp(CQDMatrix l21, CQDMatrix a, long int start_index, long int min_dim)
{
	int thread_index, thread_num;
	long int i, j, k, imax, jmax, itmp, dim;
	double dtmp[128][QDSIZE], dtmp1[128][QDSIZE], dmaxii[128][QDSIZE];
    cqdfloat ctmp[128], ctmp1[128];

	dim = a->re->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		
		rcqd_set0(&ctmp[thread_index]);
		rcqd_set0(&ctmp1[thread_index]);
		set0_qd(dmaxii[thread_index]);
	}

	//set0_qd(dmaxii);

	// dim > min_dim
	#pragma omp parallel for private(thread_index, j, k)
	for(i = imax; i < dim; i++)
	{
		thread_index = omp_get_thread_num();

		rcqd_abs_qd(dmaxii[thread_index], get_cqdmatrix_ij(a, i, i));
		if(rqd_cmp_ui(dmaxii[thread_index], 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (CQDLUdecomp_l21_omp)!\n", i);
			//return -1;
		}

		for(j = start_index; j < jmax; j++)
		{
			rcqd_set(&ctmp[thread_index], get_cqdmatrix_ij(a, i, j));
			for(k = start_index; k < j; k++)
			{
				//dtmp -= get_dmatrix_ij(a, i, k) * get_dmatrix_ij(a, k, j);
				rcqd_mul(&ctmp1[thread_index], get_cqdmatrix_ij(a, i, k), get_cqdmatrix_ij(a, k, j));
				rcqd_sub(&ctmp[thread_index], &ctmp[thread_index], &ctmp1[thread_index]);
			}
			rcqd_div(&ctmp1[thread_index], &ctmp[thread_index], get_cqdmatrix_ij(a, j, j));
			rcqd_set(&ctmp[thread_index], &ctmp1[thread_index]);
			
			//printf("l21(i              , j       ) = (%ld, %ld), %ld, %ld\n", i, j, l21->row_dim, l21->col_dim);
			//printf("l21(i - start_index, j - imax) = (%ld, %ld), %ld, %ld\n", i - imax, j - start_index, l21->row_dim, l21->col_dim);
			//printf("(i              , j       ) = (%ld, %ld) %25.17e, %25.17e\n", i, j, dtmp, get_dmatrix_ij(a, j, j));
			set_cqdmatrix_ij(a  , i              , j       , &ctmp[thread_index]);
			set_cqdmatrix_ij(l21, i - imax, j - start_index, &ctmp[thread_index]);
		}
	}

	return 0;
}
#endif // USE_OMP

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
int CQDLUdecomp_u12(CQDMatrix u12, CQDMatrix a, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim;
	double dtmp[QDSIZE], dtmp1[QDSIZE], dmaxii[QDSIZE];
    cqdfloat ctmp, ctmp1;

	dim = a->re->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// dim > min_dim
	for(i = imax; i < dim; i++)
	{
		rcqd_abs_qd(dmaxii, get_cqdmatrix_ij(a, i, i));
		if(rqd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (CQDLUdecomp_u12)!\n", i);
			return -1;
		}

		for(j = start_index; j < jmax; j++)
		{
			rcqd_set(&ctmp, get_cqdmatrix_ij(a, j, i));
			//printf("(j              , i              ) = (%ld, %ld) %25.17e\n", j, i , dtmp);
			for(k = start_index; k < j; k++)
			{
				//dtmp -= get_cqdmatrix_ij(a, j, k) * get_cqdmatrix_ij(a, k, i);
				rcqd_mul(&ctmp1, get_cqdmatrix_ij(a, j, k), get_cqdmatrix_ij(a, k, i));
				rcqd_sub(&ctmp, &ctmp, &ctmp1);
				//printf("(j - start_index, k - start_index) = (%ld, %ld)\n", j - start_index, k - start_index);
			}
			//printf("u12(j              , i       ) = (%ld, %ld) %ld, %ld\n", j, i, u12->row_dim, u12->col_dim);
			//printf("u12(j - start_index, i - imax) = (%ld, %ld) %ld, %ld\n", j - start_index, i - imax, u12->row_dim, u12->col_dim);
			set_cqdmatrix_ij(a  , j              , i       , &ctmp);
			set_cqdmatrix_ij(u12, j - start_index, i - imax, &ctmp);
		}
	}

	return 0;
}

#ifdef USE_OMP
int CQDLUdecomp_u12_omp(CQDMatrix u12, CQDMatrix a, long int start_index, long int min_dim)
{
	int thread_index;
	long int i, j, k, imax, jmax, itmp, dim;
	double dtmp[128][QDSIZE], dtmp1[128][QDSIZE], dmaxii[128][QDSIZE];
    cqdfloat ctmp[128], ctmp1[128];

	dim = a->re->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();

		rcqd_set0(&ctmp[thread_index]);
		rcqd_set0(&ctmp1[thread_index]);
		set0_qd(dmaxii[thread_index]);
	}

	// dim > min_dim
	#pragma omp parallel for private(thread_index, j, k)
	for(i = imax; i < dim; i++)
	{
		thread_index = omp_get_thread_num();

		rcqd_abs_qd(dmaxii[thread_index], get_cqdmatrix_ij(a, i, i));
		if(rqd_cmp_ui(dmaxii[thread_index], 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (CQDLUdecomp_u12_omp)!\n", i);
			//return -1;
		}

		for(j = start_index; j < jmax; j++)
		{
			rcqd_set(&ctmp[thread_index], get_cqdmatrix_ij(a, j, i));
			//printf("(j              , i              ) = (%ld, %ld) %25.17e\n", j, i , dtmp);
			for(k = start_index; k < j; k++)
			{
				//dtmp -= get_cqdmatrix_ij(a, j, k) * get_cqdmatrix_ij(a, k, i);
				rcqd_mul(&ctmp1[thread_index], get_cqdmatrix_ij(a, j, k), get_cqdmatrix_ij(a, k, i));
				rcqd_sub(&ctmp[thread_index], &ctmp[thread_index], &ctmp1[thread_index]);
				//printf("(j - start_index, k - start_index) = (%ld, %ld)\n", j - start_index, k - start_index);
			}
			//printf("u12(j              , i       ) = (%ld, %ld) %ld, %ld\n", j, i, u12->row_dim, u12->col_dim);
			//printf("u12(j - start_index, i - imax) = (%ld, %ld) %ld, %ld\n", j - start_index, i - imax, u12->row_dim, u12->col_dim);
			set_cqdmatrix_ij(a  , j              , i       , &ctmp[thread_index]);
			set_cqdmatrix_ij(u12, j - start_index, i - imax, &ctmp[thread_index]);
		}
	}

	return 0;
}
#endif // USE_OMP

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
int CQDLUdecomp_a22(CQDMatrix a, CQDMatrix d22, CQDMatrix l21, CQDMatrix u12, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim, index[4], d22_index[4];

	dim = a->re->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// d22 := l21 * u12
#ifdef USE_BLOCK
//	mul_cqdmatrix_block(d22, l21, u12, min_dim);
	mul_cqdmatrix_block(d22, l21, u12, STRASSEN_MIN_DIM);
#elif USE_STRASSEN
//	mul_cqdmatrix_strassen(d22, l21, u12, min_dim);
	mul_cqdmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM);
#elif USE_WINOGRAD
//	mul_cqdmatrix_strassen(d22, l21, u12, min_dim);
	mul_cqdmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM);
/* #elif USE_OZ
	#ifdef OZ_MAX_NUM_DIV
	int max_num_div = OZ_MAX_NUM_DIV;
	#else
	int max_num_div = 10;
	#endif // OZ_MAX_NUM_DIV
//	mul_cqdmatrix_strassen(d22, l21, u12, min_dim);
	mul_cqdmatrix_oz(d22, l21, max_num_div, u12, max_num_div);
*/
#elif USE_SIMPLE // USE_BLOCK
	mul_cqdmatrix_simple(d22, l21, u12);
#else // USE_BLOCK
	mul_cqdmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM);
#endif // USE_BLOCK

	// a22 := a22 - d22
	index[0] = imax;
	index[1] = dim;
	index[2] = jmax;
	index[3] = dim;
	d22_index[0] = 0;
	d22_index[1] = d22->re->row_dim;
	d22_index[2] = 0;
	d22_index[3] = d22->re->col_dim;

	sub_cqdmatrix_partial(a, index, a, index, d22, d22_index);

	return 0;
}

#ifdef USE_OMP
int CQDLUdecomp_a22_omp(CQDMatrix a, CQDMatrix d22, CQDMatrix l21, CQDMatrix u12, long int start_index, long int min_dim)
{
	long int i, j, k, imax, jmax, itmp, dim, index[4], d22_index[4];

	dim = a->re->col_dim;

	imax = start_index + min_dim;
	if(imax > dim) imax = dim;

	jmax = imax;

	// d22 := l21 * u12
#ifdef USE_BLOCK
//	mul_cqdmatrix_block(d22, l21, u12, STRASSEN_MIN_DIM);
	_bncomp_mul_cqdmatrix_block(d22, l21, u12, STRASSEN_MIN_DIM);
//	_bncomp_mul_cqdmatrix_block_old(d22, l21, u12, STRASSEN_MIN_DIM);
#elif USE_STRASSEN
	//printf("-Strassen: \n");
	mul_cqdmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM);
	//_bncomp_mul_cqdmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM);
	//_bncomp_mul_cqdmatrix_strassen_4m(d22, l21, u12, STRASSEN_MIN_DIM);
#elif USE_WINOGRAD
//	mul_cqdmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM);
	_bncomp_mul_cqdmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM);
#elif USE_SIMPLE // USE_BLOCK
	//printf("-Simple: \n");
	_bncomp_mul_cqdmatrix_simple(d22, l21, u12);
#else // USE_BLOCK
	//printf("-Default Strassen: \n");
	//mul_cqdmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM); // Fix! 2024-02-05 (Mon)T.Kouya
	_bncomp_mul_cqdmatrix_strassen(d22, l21, u12, STRASSEN_MIN_DIM); // Fix! 2024-02-05 (Mon)T.Kouya
	//mul_cqdmatrix_simple(d22, l21, u12);
	//_bncomp_mul_cqdmatrix_simple(d22, l21, u12);
#endif // USE_BLOCK

	// a22 := a22 - d22
	index[0] = imax;
	index[1] = dim;
	index[2] = jmax;
	index[3] = dim;
	d22_index[0] = 0;
	d22_index[1] = d22->re->row_dim;
	d22_index[2] = 0;
	d22_index[3] = d22->re->col_dim;

	//sub_cqdmatrix_partial(a, index, a, index, d22, d22_index);
	_bncomp_sub_cqdmatrix_partial(a, index, a, index, d22, d22_index); // OK!

	return 0;
}
#endif // USE_OMP

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                 (Quadraple Precision)    */
/*                                                          */
/*                 ver. 0.0 2015-02-25 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CQDLUdecomp_strassen(CQDMatrix a, long int min_dim)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CQDMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, row_dim, col_dim, dim;
	CQDMatrix l21, u12, d22;

	dim = a->re->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		CQDLUdecomp(a);
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
				CQDLUdecomp_square(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		CQDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_cqdmatrix(a);

		l21 = init_cqdmatrix(row_dim, min_dim);
		u12 = init_cqdmatrix(min_dim, col_dim);
		d22 = init_cqdmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		CQDLUdecomp_l21(l21, a, i, min_dim);
		//print_cqdmatrix(a);
		//printf("L21:\n");
		//print_cqdmatrix(l21);

		// (3) Solve L11 * U12 = A12
		CQDLUdecomp_u12(u12, a, i, min_dim);
		//print_cqdmatrix(a);
		//printf("U12:\n");
		//print_cqdmatrix(u12);

		//printf("A:\n");
		//print_cqdmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		CQDLUdecomp_a22(a, d22, l21, u12, i, min_dim);
		//printf("D22:\n");
		//print_cqdmatrix(d22);
		//print_cqdmatrix(a);

		// clear
		free_cqdmatrix(l21);
		free_cqdmatrix(u12);
		free_cqdmatrix(d22);

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
int CQDLUdecomp_strassenPM(CQDMatrix a, long int ch[], long int min_dim)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CQDMatrix a: Matrix (given by user)                 */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, row_dim, col_dim, dim, imax, itmp;
	CQDMatrix l21, u12, d22;
	double dtmp[QDSIZE], dtmp1[QDSIZE], dmaxii[QDSIZE];
    cqdfloat ctmp, ctmp1;

	dim = a->re->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		CQDLUdecompPM(a, ch);
		return 0;
	}

	for(i = 0; i < dim; i++)
		ch[i] = i;

	// dim > min_dim
	for(i = 0; i < dim; i += min_dim)
	{
		// partial pivoting
		rcqd_abs_qd(dmaxii, get_cqdmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rcqd_abs_qd(dtmp, get_cqdmatrix_ij(a, j, i));
			if(rqd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rqd_set(dmaxii, dtmp);
			}
		}

		if(rqd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! CQDLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			row_swap_cqdmatrix(a, i, imax, 0, a->re->col_dim);
		}

		// Initialize
		row_dim = dim - (i + min_dim);
		if(row_dim <= 0) 
		{
			// (1) L11 * U11 = A11
			if((dim - i) > 0)
				CQDLUdecomp_square(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		CQDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_cqdmatrix(a);

		l21 = init_cqdmatrix(row_dim, min_dim);
		u12 = init_cqdmatrix(min_dim, col_dim);
		d22 = init_cqdmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		CQDLUdecomp_l21(l21, a, i, min_dim);
		//print_cqdmatrix(a);
		//printf("L21:\n");
		//print_cqdmatrix(l21);

		// (3) Solve L11 * U12 = A12
		CQDLUdecomp_u12(u12, a, i, min_dim);
		//print_cqdmatrix(a);
		//printf("U12:\n");
		//print_cqdmatrix(u12);

		//printf("A:\n");
		//print_cqdmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		CQDLUdecomp_a22(a, d22, l21, u12, i, min_dim);
		//printf("D22:\n");
		//print_cqdmatrix(d22);
		//print_cqdmatrix(a);

		// clear
		free_cqdmatrix(l21);
		free_cqdmatrix(u12);
		free_cqdmatrix(d22);

	}

	return 0;
}


#ifdef USE_OMP
/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                  (qd_real)               */
/*                                                          */
/*                 ver. 0.0 2015-07-07 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CQDLUdecomp_omp(CQDMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*        CQDMatrix a: Matrix (given by user)                */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	double tmp[128][QDSIZE], axii[QDSIZE];
    cqdfloat ctmp[128];
	int thread_index;

	dim = a->re->col_dim;
	set0_qd(axii);

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_cqd(&ctmp[thread_index]);
	}

	for(i = 0; i < dim; i++)
	{
		rcqd_abs_qd(axii, get_cqdmatrix_ij(a, i, i));

		if(rqd_cmp_ui(axii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (CQDLUdecomp_omp)!\n", i);
			return -1;
		}

		#pragma omp parallel for private(thread_index)
		for(j = (i + 1); j < dim; j++)
		{
			thread_index = omp_get_thread_num();

			rcqd_div(&ctmp[thread_index], get_cqdmatrix_ij(a, j, i), get_cqdmatrix_ij(a, i, i));
			set_cqdmatrix_ij(a, j, i, &ctmp[thread_index]);
		}

		#pragma omp parallel for private(thread_index, k)
		for(j = (i + 1); j < dim; j++)
		{
			thread_index = omp_get_thread_num();

			for(k = (i + 1); k < dim; k++)
			{
				rcqd_mul(&ctmp[thread_index], get_cqdmatrix_ij(a, j, i), get_cqdmatrix_ij(a, i, k));
				rcqd_sub(&ctmp[thread_index], get_cqdmatrix_ij(a, j, k), &ctmp[thread_index]);
				set_cqdmatrix_ij(a, j, k, &ctmp[thread_index]);
			}
		}
	}

	return 0;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                  (qd_real)               */
/*                                                          */
/*                 ver. 0.0 2015-07-07 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CQDLUdecompPM_omp(CQDMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*        CQDMatrix a: Matrix (given by user)                */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, jmax, itmp, dim;
	double tmp[128][QDSIZE], axii[QDSIZE];
    cqdfloat ctmp[128];
	int thread_index;
// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	long int k_start, k_end, dim_start, dim_end, index_ji, index_ik, index_jk;
	__m256d dtmp256_re[BNCOMP_MAX_NUM_THREADS][QDSIZE], aji256_re[BNCOMP_MAX_NUM_THREADS][QDSIZE], ajk256_re[BNCOMP_MAX_NUM_THREADS][QDSIZE], aik256_re[BNCOMP_MAX_NUM_THREADS][QDSIZE];
	__m256d dtmp256_im[BNCOMP_MAX_NUM_THREADS][QDSIZE], aji256_im[BNCOMP_MAX_NUM_THREADS][QDSIZE], ajk256_im[BNCOMP_MAX_NUM_THREADS][QDSIZE], aik256_im[BNCOMP_MAX_NUM_THREADS][QDSIZE];
#elif defined(__AVX512F__) // __AVX512F__
#endif // __AVX2__

	dim = a->re->col_dim;
	set0_qd(axii);

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_cqd(&ctmp[thread_index]);
	}

	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		// partial pivoting
		rcqd_abs_qd(axii, get_cqdmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rcqd_abs_qd(tmp[0], get_cqdmatrix_ij(a, j, i));
			if(rqd_cmp(tmp[0], axii) > 0)
			{
				imax = j;
				rqd_set(axii, tmp[0]);
			}
		}

		if(rqd_cmp_ui(axii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! CQDLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			_bncomp_row_swap_cqdmatrix(a, i, imax, 0, a->re->col_dim);
		}

		#pragma omp parallel for private(thread_index)
		for(j = (i + 1); j < dim; j++)
		{
			thread_index = omp_get_thread_num();

			rcqd_div(&ctmp[thread_index], get_cqdmatrix_ij(a, j, i), get_cqdmatrix_ij(a, i, i));
			set_cqdmatrix_ij(a, j, i, &ctmp[thread_index]);
		}
// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
		dim_start = (long int)ceil((double)(i + 1) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		dim_end = a->re->real_col_dim;

		//printf("real_row_dim, real_col_dim, dim, i, dim_start, dim_end = %ld, %ld, %ld, %ld, %ld, %ld\n", a->real_row_dim, a->real_col_dim, dim, i, dim_start, dim_end);
		#pragma omp parallel for private(thread_index, k, k_start, k_end, index_ji, index_jk, index_ik)
		for(j = (i + 1); j < dim; j++)
		{
			thread_index = omp_get_thread_num();

			index_ji = j * a->re->real_col_dim + i;

			// Real part
			aji256_re[thread_index][0] = _mm256_set_pd(
                a->re->element[0][index_ji],
                a->re->element[0][index_ji],
                a->re->element[0][index_ji],
                a->re->element[0][index_ji]
            );
			aji256_re[thread_index][1] = _mm256_set_pd(
                a->re->element[1][index_ji],
                a->re->element[1][index_ji],
                a->re->element[1][index_ji],
                a->re->element[1][index_ji]
            );
			aji256_re[thread_index][2] = _mm256_set_pd(
                a->re->element[2][index_ji],
                a->re->element[2][index_ji],
                a->re->element[2][index_ji],
                a->re->element[2][index_ji]
            );
			aji256_re[thread_index][3] = _mm256_set_pd(
                a->re->element[3][index_ji],
                a->re->element[3][index_ji],
                a->re->element[3][index_ji],
                a->re->element[3][index_ji]
            );

			// Imaginary part
			aji256_im[thread_index][0] = _mm256_set_pd(
                a->im->element[0][index_ji],
                a->im->element[0][index_ji],
                a->im->element[0][index_ji],
                a->im->element[0][index_ji]
            );
			aji256_im[thread_index][1] = _mm256_set_pd(
                a->im->element[1][index_ji],
                a->im->element[1][index_ji],
                a->im->element[1][index_ji],
                a->im->element[1][index_ji]
            );
			aji256_im[thread_index][2] = _mm256_set_pd(
                a->im->element[2][index_ji],
                a->im->element[2][index_ji],
                a->im->element[2][index_ji],
                a->im->element[2][index_ji]
            );
			aji256_im[thread_index][3] = _mm256_set_pd(
                a->im->element[3][index_ji],
                a->im->element[3][index_ji],
                a->im->element[3][index_ji],
                a->im->element[3][index_ji]
            );

			// head
			//printf("start j, k= %ld, %ld, ", j, i + 1);
			k_end = (dim_start > dim) ? dim : dim_start;
			for(k = (i + 1); k < k_end; k++)
			//for(k = (i + 1); k < dim; k++)
			{
				rcqd_mul(&ctmp[thread_index], get_cqdmatrix_ij(a, j, i), get_cqdmatrix_ij(a, i, k));
				rcqd_sub(&ctmp[thread_index], get_cqdmatrix_ij(a, j, k), &ctmp[thread_index]);
				set_cqdmatrix_ij(a, j, k, &ctmp[thread_index]);
			}
			//printf("head k_start, k = %ld, %ld, ", k_start, k);

			// middle : SIMD
			k_end = dim_end;
			for(k = dim_start; k < dim_end; k += _BNC_D_WIDTH)
			{
				index_ik = i * a->re->real_col_dim + k;
				//rdd_mul(dtmp1, get_ddmatrix_ij(a, j, i), get_ddmatrix_ij(a, i, k));
				aik256_re[thread_index][0] = _mm256_load_pd(&(a->re->element[0][index_ik]));
				aik256_re[thread_index][1] = _mm256_load_pd(&(a->re->element[1][index_ik]));
				aik256_re[thread_index][2] = _mm256_load_pd(&(a->re->element[2][index_ik]));
				aik256_re[thread_index][3] = _mm256_load_pd(&(a->re->element[3][index_ik]));

				aik256_im[thread_index][0] = _mm256_load_pd(&(a->im->element[0][index_ik]));
				aik256_im[thread_index][1] = _mm256_load_pd(&(a->im->element[1][index_ik]));
				aik256_im[thread_index][2] = _mm256_load_pd(&(a->im->element[2][index_ik]));
				aik256_im[thread_index][3] = _mm256_load_pd(&(a->im->element[3][index_ik]));

				_bncavx2_rcqd_mul(
					dtmp256_re[thread_index], dtmp256_im[thread_index],
					aji256_re[thread_index] , aji256_im[thread_index],
					aik256_re[thread_index] , aik256_im[thread_index]
				);
				//printf(" -- mul -- ");

				index_jk = j * a->re->real_col_dim + k;
				//rdd_sub(dtmp, get_ddmatrix_ij(a, j, k), dtmp1);
				ajk256_re[thread_index][0] = _mm256_load_pd(&(a->re->element[0][index_jk]));
				ajk256_re[thread_index][1] = _mm256_load_pd(&(a->re->element[1][index_jk]));
				ajk256_re[thread_index][2] = _mm256_load_pd(&(a->re->element[2][index_jk]));
				ajk256_re[thread_index][3] = _mm256_load_pd(&(a->re->element[3][index_jk]));

				ajk256_im[thread_index][0] = _mm256_load_pd(&(a->im->element[0][index_jk]));
				ajk256_im[thread_index][1] = _mm256_load_pd(&(a->im->element[1][index_jk]));
				ajk256_im[thread_index][2] = _mm256_load_pd(&(a->im->element[2][index_jk]));
				ajk256_im[thread_index][3] = _mm256_load_pd(&(a->im->element[3][index_jk]));

				_bncavx2_rcqd_sub(
					dtmp256_re[thread_index], dtmp256_im[thread_index],
					 ajk256_re[thread_index],  ajk256_im[thread_index],
					dtmp256_re[thread_index], dtmp256_im[thread_index]
				);
				//printf(" -- sub -- ");

				//set_ddmatrix_ij(a, j, k, dtmp);
				_mm256_store_pd(&(a->re->element[0][index_jk]), dtmp256_re[thread_index][0]);
				_mm256_store_pd(&(a->re->element[1][index_jk]), dtmp256_re[thread_index][1]);
				_mm256_store_pd(&(a->re->element[2][index_jk]), dtmp256_re[thread_index][2]);
				_mm256_store_pd(&(a->re->element[3][index_jk]), dtmp256_re[thread_index][3]);

				_mm256_store_pd(&(a->im->element[0][index_jk]), dtmp256_im[thread_index][0]);
				_mm256_store_pd(&(a->im->element[1][index_jk]), dtmp256_im[thread_index][1]);
				_mm256_store_pd(&(a->im->element[2][index_jk]), dtmp256_im[thread_index][2]);
				_mm256_store_pd(&(a->im->element[3][index_jk]), dtmp256_im[thread_index][3]);

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
				rcqd_mul(&ctmp[thread_index], get_cqdmatrix_ij(a, j, i), get_cqdmatrix_ij(a, i, k));
				rcqd_sub(&ctmp[thread_index], get_cqdmatrix_ij(a, j, k), &ctmp[thread_index]);
				set_cqdmatrix_ij(a, j, k, &ctmp[thread_index]);
			}
		}
#endif // __AVX2__
	}

	return 0;
}
#endif // USE_OMP

#ifdef USE_OMP
/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                 (Quadraple Precision)    */
/*                                                          */
/*                 ver. 0.0 2015-02-25 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int CQDLUdecomp_strassen_omp(CQDMatrix a, long int min_dim)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CQDMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, row_dim, col_dim, dim;
	CQDMatrix l21, u12, d22;

	dim = a->re->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		CQDLUdecomp_omp(a);
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
				CQDLUdecomp_square_omp(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		CQDLUdecomp_square_omp(a, i, min_dim);
		//CQDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_cqdmatrix(a);

		l21 = init_cqdmatrix(row_dim, min_dim);
		u12 = init_cqdmatrix(min_dim, col_dim);
		d22 = init_cqdmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		CQDLUdecomp_l21_omp(l21, a, i, min_dim);
		//CQDLUdecomp_l21(l21, a, i, min_dim);
		//print_cqdmatrix(a);
		//printf("L21:\n");
		//print_cqdmatrix(l21);

		// (3) Solve L11 * U12 = A12
		CQDLUdecomp_u12_omp(u12, a, i, min_dim);
		//CQDLUdecomp_u12(u12, a, i, min_dim);
		//print_cqdmatrix(a);
		//printf("U12:\n");
		//print_cqdmatrix(u12);

		//printf("A:\n");
		//print_cqdmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		CQDLUdecomp_a22_omp(a, d22, l21, u12, i, min_dim);
		//CQDLUdecomp_a22(a, d22, l21, u12, i, min_dim);
		//printf("D22:\n");
		//print_cqdmatrix(d22);
		//print_cqdmatrix(a);

		// clear
		free_cqdmatrix(l21);
		free_cqdmatrix(u12);
		free_cqdmatrix(d22);

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
int CQDLUdecomp_strassenPM_omp(CQDMatrix a, long int ch[], long int min_dim)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       CQDMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, row_dim, col_dim, dim, imax, itmp;
	CQDMatrix l21, u12, d22;
	double dtmp[QDSIZE], dmaxii[QDSIZE];
    cqdfloat ctmp;

	dim = a->re->col_dim;

	// dim <= min_dim
	if(dim <= min_dim)
	{
		CQDLUdecompPM_omp(a, ch);
		return 0;
	}

	for(i = 0; i < dim; i++)
		ch[i] = i;

	// dim > min_dim
	for(i = 0; i < dim; i += min_dim)
	{
		// partial pivoting
		rcqd_abs_qd(dmaxii, get_cqdmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			rcqd_abs_qd(dtmp, get_cqdmatrix_ij(a, j, i));
			if(rqd_cmp(dtmp, dmaxii) > 0)
			{
				imax = j;
				rqd_set(dmaxii, dtmp);
			}
		}

		if(rqd_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! CQDLUdecompPM!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			_bncomp_row_swap_cqdmatrix(a, i, imax, 0, a->re->col_dim);
		}

		// Initialize
		row_dim = dim - (i + min_dim);
		if(row_dim <= 0) 
		{
			// (1) L11 * U11 = A11
			if((dim - i) > 0)
				CQDLUdecomp_square_omp(a, i, dim - i);

			break;
		}

		col_dim = row_dim;

		//printf("%2ld : (row_dim, col_dim, min_dim) = (%ld, %ld, %ld)\n", i, row_dim, col_dim, min_dim);

		// (1) L11 * U11 = A11
		CQDLUdecomp_square_omp(a, i, min_dim); // OK!
		//CQDLUdecomp_square(a, i, min_dim);

		//printf("i = %ld\n", i);
		//print_cqdmatrix(a);

		l21 = init_cqdmatrix(row_dim, min_dim);
		u12 = init_cqdmatrix(min_dim, col_dim);
		d22 = init_cqdmatrix(row_dim, col_dim);

		// (2) Solve L21 * U11 = A21
		CQDLUdecomp_l21_omp(l21, a, i, min_dim); // Fix! 2024-02-05(Mon) T.Kouya
		//CQDLUdecomp_l21(l21, a, i, min_dim);
		//print_cqdmatrix(a);
		//printf("L21:\n");
		//print_cqdmatrix(l21);

		// (3) Solve L11 * U12 = A12
		CQDLUdecomp_u12_omp(u12, a, i, min_dim); // OK!
		//CQDLUdecomp_u12(u12, a, i, min_dim);
		//print_cqdmatrix(a);
		//printf("U12:\n");
		//print_cqdmatrix(u12);

		//printf("A:\n");
		//print_cqdmatrix(a);

		// (4) D22 := L21 * U12
		// (5) A22 := A22 - D22
		CQDLUdecomp_a22_omp(a, d22, l21, u12, i, min_dim); // x
		//CQDLUdecomp_a22(a, d22, l21, u12, i, min_dim);
		//printf("D22:\n");
		//print_cqdmatrix(d22);
		//print_cqdmatrix(a);

		// clear
		free_cqdmatrix(l21);
		free_cqdmatrix(u12);
		free_cqdmatrix(d22);

	}

	return 0;
}
#endif // USE_OMP
