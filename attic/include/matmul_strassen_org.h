/********************************************************************************/
/* matmul_strassen.h: Fast Matrix Multiplication based on double, double-double */
/*                    and Multiple precision floating-point arithmetics         */
/* Copyright (C) 2014-2015 Tomonori Kouya                                       */
/*                                                                              */
/* Version 0.19 : First implementation of parallelized matrix mutiplication     */
/* Version 0.2  : DD and QD precision have been supported                       */
/* Version 0.21 : Bug fix in mul_mpfmatrix_strassen_odd_peeling                 */
/* Version 0.3b0: AVX2 and AVX-512 is applied for any codes                     */
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
// define _BNC_MATMUL_STRASSEN_H
#ifndef _BNC_MATMUL_STRASSEN_H
#define _BNC_MATMUL_STRASSEN_H

#ifdef __cplusplus
#include <cstdio>
#include <cmath>
#else // __cplusplus
#include <stdio.h>
#include <math.h>
#endif // __cplusplus

//#ifdef USE_DDLINEAR
#include "dlinear.h"

#ifdef USE_IMKL
	#include "mkl.h"
	#include "mkl_cblas.h" // for Intel Math Kernel Library
#endif // USE_IMKL

#ifdef USE_DDLINEAR
	#include "ddlinear.h" // double-double and quadratic double precision
	#ifdef USE_TDLINEAR
		#include "tdlinear.h"
		#ifdef USE_QDLINEAR
			#include "qdlinear.h"
		#endif // USE_QDLINEAR
	#endif // USE_TDLINEAR
#endif // USE_DDLINEAR

#ifdef USE_CUDA
	#include "gddlinear.h"
#endif // USE_CUDA
//#endif // USE_DDLINEAR

#ifdef USE_GMP
	#include "gmp.h"
	#ifdef USE_MPFR
		#include "mpfr.h"
	#endif // USE_MPFR
	#include "mpflinear.h"
#endif // USE_GMP

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#ifndef _BNC_DEFAULT_MIN_DIM_STRASSEN
//	#define _BNC_DEFAULT_MIN_DIM_STRASSEN 2
//	#define _BNC_DEFAULT_MIN_DIM_STRASSEN 4
//	#define _BNC_DEFAULT_MIN_DIM_STRASSEN 8
//	#define _BNC_DEFAULT_MIN_DIM_STRASSEN 16
	#define _BNC_DEFAULT_MIN_DIM_STRASSEN 32
//	#define _BNC_DEFAULT_MIN_DIM_STRASSEN 64
//	#define _BNC_DEFAULT_MIN_DIM_STRASSEN 128
//	#define _BNC_DEFAULT_MIN_DIM_STRASSEN 256
#endif // _BNC_DEFAULT_MIN_DIM_STRASSEN

// log2(x)
//#define mylog2(x) (log((double)(x)) / log((double)2.0)) // <-- fix! 2015-06-16(Tue)
static inline double mylog2(double x)
{
	return log10(x) / log10(2.0);
}

// GFlops
static inline double matmul_gflops(double comp_sec, int dim)
{
	return 2.0 * (double)dim * (double)dim * (double)dim / comp_sec / 1024.0 / 1024.0 / 1024.0;
}

// GB of Double prec. Square matrix
static inline int byte_double_sqmat(int dim)
{
	return sizeof(double) * dim * dim;
}

// variables to count the number of arithmetic
// double precision
#ifndef __NUM_ADDSUB_MUL_DMATRIX_STRASSEN
#define __NUM_ADDSUB_MUL_DMATRIX_STRASSEN
extern long int num_addsub_mul_dmatrix_strassen;	// addition and subtraction
extern long int num_mul_mul_dmatrix_strassen;		// multiplication
#endif // __NUM_ADDSUB_MUL_DMATRIX_STRASSEN

// clear counter
void reset_num_mul_dmatrix_strassen(void);

// print counters
void print_num_mul_dmatrix_strassen(long int *num_addsub, long int *num_mul);

// multiple precision
#ifdef USE_GMP
#ifndef __NUM_ADDSUB_MUL_MPFMATRIX_STRASSEN
#define __NUM_ADDSUB_MUL_MPFMATRIX_STRASSEN
extern long int num_addsub_mul_mpfmatrix_strassen;	// addition and subtraction
extern long int num_mul_mul_mpfmatrix_strassen;		// multiplication
#endif // __NUM_ADDSUB_MUL_MPFMATRIX_STRASSEN

// clear counter
void reset_num_mul_mpfmatrix_strassen(void);

// get counters
void get_num_mul_mpfmatrix_strassen(long int *num_addsub, long int *num_mul);

// print counters
void print_num_mul_mpfmatrix_strassen(long int *num_addsub, long int *num_mul);

// multiple precision(OpenMP)
#ifndef ___BNCOMP_NUM_ADDSUB_MUL_MPFMATRIX_STRASSEN
#define ___BNCOMP_NUM_ADDSUB_MUL_MPFMATRIX_STRASSEN
extern long int _bncomp_num_addsub_mul_mpfmatrix_strassen;	// addition and subtraction
extern long int _bncomp_num_mul_mul_mpfmatrix_strassen;		// multiplication
#endif // ___BNCOMP_NUM_ADDSUB_MUL_MPFMATRIX_STRASSEN

// clear counter
void _bncomp_reset_num_mul_mpfmatrix_strassen(void);

// get counters
void _bncomp_get_num_mul_mpfmatrix_strassen(long int *_bncomp_num_addsub, long int *_bncomp_num_mul);

// print counters
void _bncomp_print_num_mul_mpfmatrix_strassen(long int *_bncomp_num_addsub, long int *_bncomp_num_mul);
#endif // USE_GMP

//#ifdef USE_DDLINEAR
// dd precision
#ifndef __NUM_ADDSUB_MUL_DDMATRIX_STRASSEN
#define __NUM_ADDSUB_MUL_DDMATRIX_STRASSEN
extern long int num_addsub_mul_ddmatrix_strassen;	// addition and subtraction
extern long int num_mul_mul_ddmatrix_strassen;		// multiplication
#endif // __NUM_ADDSUB_MUL_DDMATRIX_STRASSEN

// clear counter
void reset_num_mul_ddmatrix_strassen(void);

// get counters
void get_num_mul_ddmatrix_strassen(long int *num_addsub, long int *num_mul);

// print counters
void print_num_mul_ddmatrix_strassen(long int *num_addsub, long int *num_mul);

// dd precision(OpenMP)
#ifndef ___BNCOMP_NUM_ADDSUB_MUL_DDMATRIX_STRASSEN
#define ___BNCOMP_NUM_ADDSUB_MUL_DDMATRIX_STRASSEN
extern long int _bncomp_num_addsub_mul_ddmatrix_strassen;	// addition and subtraction
extern long int _bncomp_num_mul_mul_ddmatrix_strassen;		// multiplication
#endif // ___BNCOMP_NUM_ADDSUB_MUL_DDMATRIX_STRASSEN

// clear counter
void _bncomp_reset_num_mul_ddmatrix_strassen(void);

// get counters
void _bncomp_get_num_mul_ddmatrix_strassen(long int *num_addsub, long int *num_mul);

// print counters
void _bncomp_print_num_mul_ddmatrix_strassen(long int *num_addsub, long int *num_mul);

// td precision
#ifndef __NUM_ADDSUB_MUL_DDMATRIX_STRASSEN
#define __NUM_ADDSUB_MUL_DDMATRIX_STRASSEN
extern long int num_addsub_mul_tdmatrix_strassen;	// addition and subtraction
extern long int num_mul_mul_tdmatrix_strassen;		// multiplication
#endif // __NUM_ADDSUB_MUL_DDMATRIX_STRASSEN

// clear counter
void reset_num_mul_tdmatrix_strassen(void);

// get counters
void get_num_mul_tdmatrix_strassen(long int *num_addsub, long int *num_mul);

// print counters
void print_num_mul_tdmatrix_strassen(long int *num_addsub, long int *num_mul);

// dd precision(OpenMP)
#ifndef ___BNCOMP_NUM_ADDSUB_MUL_DDMATRIX_STRASSEN
#define ___BNCOMP_NUM_ADDSUB_MUL_DDMATRIX_STRASSEN
extern long int _bncomp_num_addsub_mul_tdmatrix_strassen;	// addition and subtraction
extern long int _bncomp_num_mul_mul_tdmatrix_strassen;		// multiplication
#endif // ___BNCOMP_NUM_ADDSUB_MUL_DDMATRIX_STRASSEN

// clear counter
void _bncomp_reset_num_mul_tdmatrix_strassen(void);

// get counters
void _bncomp_get_num_mul_tdmatrix_strassen(long int *num_addsub, long int *num_mul);

// print counters
void _bncomp_print_num_mul_tdmatrix_strassen(long int *num_addsub, long int *num_mul);

// qd precision
#ifndef __NUM_ADDSUB_MUL_QDMATRIX_STRASSEN
#define __NUM_ADDSUB_MUL_QDMATRIX_STRASSEN
extern long int num_addsub_mul_qdmatrix_strassen;	// addition and subtraction
extern long int num_mul_mul_qdmatrix_strassen;		// multiplication
#endif // __NUM_ADDSUB_MUL_QDMATRIX_STRASSEN

// clear counter
void reset_num_mul_qdmatrix_strassen(void);

// get counters
void get_num_mul_qdmatrix_strassen(long int *num_addsub, long int *num_mul);

// print counters
void print_num_mul_qdmatrix_strassen(long int *num_addsub, long int *num_mul);
//#endif // DD_LINEAR

/* c = a * b */
void mul_dmatrix_simple(DMatrix c, DMatrix a, DMatrix b);

//#ifdef DD_LINEAR
// qd precision(OpenMP)
#ifndef ___BNCOMP_NUM_ADDSUB_MUL_QDMATRIX_STRASSEN
#define ___BNCOMP_NUM_ADDSUB_MUL_QDMATRIX_STRASSEN
// count the number of computations
extern long int _bncomp_num_addsub_mul_qdmatrix_strassen;	// addition and subtraction
extern long int _bncomp_num_mul_mul_qdmatrix_strassen;		// multiplication
#endif // ___BNCOMP_NUM_ADDSUB_MUL_QDMATRIX_STRASSEN

// clear counter
void _bncomp_reset_num_mul_qdmatrix_strassen(void);

// get counters
void _bncomp_get_num_mul_qdmatrix_strassen(long int *num_addsub, long int *num_mul);

// print counters
void _bncomp_print_num_mul_qdmatrix_strassen(long int *num_addsub, long int *num_mul);
//#endif // DD_LINEAR

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void add_dmatrix_partial(DMatrix ret, long int ret_index[4], DMatrix mat_a, long int mat_a_index[4], DMatrix mat_b, long int mat_b_index[4]);

// partial sub
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void sub_dmatrix_partial(DMatrix ret, long int ret_index[4], DMatrix mat_a, long int mat_a_index[4], DMatrix mat_b, long int mat_b_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_dmatrix_partial(DMatrix ret, long int ret_index[4], DMatrix mat_a, long int mat_a_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_dmatrix_partial_checked(DMatrix ret, long int ret_index[4], DMatrix mat_a, long int mat_a_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void neg_dmatrix_partial(DMatrix ret, long int ret_index[4], DMatrix mat_a, long int mat_a_index[4]);

// Padding to 2-powered dimensional matrix
DMatrix init_static_padding_dmatrix_strassen(DMatrix orig_mat);

// Padding to even dimensional matrix
DMatrix init_dynamic_padding_dmatrix_strassen(DMatrix orig_mat);

// Strassen's Algorithm with static padding
void mul_dmatrix_strassen_odd_padding(DMatrix ret, DMatrix mat_a, DMatrix mat_b, long int min_dim);

// Strassen's Algorithm with dynamic peeling
void mul_dmatrix_strassen(DMatrix ret, DMatrix mat_a, DMatrix mat_b, long int min_dim);

// Strassen's Algorithm
void mul_dmatrix_strassen_odd_peeling(DMatrix ret, DMatrix mat_a, DMatrix mat_b, long int min_dim);

// Strassen's Algorithm
void mul_dmatrix_strassen_even(DMatrix ret, DMatrix mat_a, DMatrix mat_b, long int min_dim);

// Winograd Variant of Strassen's Algorithm
void mul_dmatrix_winograd_even(DMatrix ret, DMatrix mat_a, DMatrix mat_b, long int min_dim);

// Block matrix multiplicaiton
void mul_dmatrix_block(DMatrix ret, DMatrix mat_a, DMatrix mat_b, long int min_dim);

// Computattion of Inverse Matrix by using Strassen's Algorithm
void inv_dmatrix_strassen_even(DMatrix ret, DMatrix mat_a, long int min_dim);

// MPF & MPFR
#ifdef USE_GMP

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void add_mpfmatrix_partial(MPFMatrix ret, long int ret_index[4], MPFMatrix mat_a, long int mat_a_index[4], MPFMatrix mat_b, long int mat_b_index[4]);

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void sub_mpfmatrix_partial(MPFMatrix ret, long int ret_index[4], MPFMatrix mat_a, long int mat_a_index[4], MPFMatrix mat_b, long int mat_b_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_mpfmatrix_partial(MPFMatrix ret, long int ret_index[4], MPFMatrix mat_a, long int mat_a_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_mpfmatrix_partial_checked(MPFMatrix ret, long int ret_index[4], MPFMatrix mat_a, long int mat_a_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void neg_mpfmatrix_partial(MPFMatrix ret, long int ret_index[4], MPFMatrix mat_a, long int mat_a_index[4]);

/* c = a * b */
void mul_mpfmatrix_simple(MPFMatrix c, MPFMatrix a, MPFMatrix b);

// Block matrix multiplicaiton
void mul_mpfmatrix_block(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim);

// Padding to 2-powered dimensional matrix
MPFMatrix init_static_padding_mpfmatrix_strassen(MPFMatrix orig_mat);

// Padding to even dimensional matrix
MPFMatrix init_dynamic_padding_mpfmatrix_strassen(MPFMatrix orig_mat);

// Strassen's Algorithm with static or dynamic padding
void mul_mpfmatrix_strassen_odd_padding(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim);

// Fit dimension to be multiple of min_dim
void mul_mpfmatrix_strassen(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim);

// Strassen's Algorithm (even dimension)
void mul_mpfmatrix_strassen_even(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim);

// Winograd Variant of Strassen's Algorithm
void mul_mpfmatrix_winograd_even(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim);

// Strassen's Algorithm with dynamic peeling
void mul_mpfmatrix_strassen_odd_peeling(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim);

// Computation of Inverse Matrix by using Strassen's Algorithm
void inv_mpfmatrix_strassen_even(MPFMatrix ret, MPFMatrix mat_a, long int min_dim);

#endif // USE_GMP

/*********************/
/* DD and QD         */
/*********************/
#ifdef USE_DDLINEAR

///////////////
///// DD //////
///////////////

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void add_ddmatrix_partial(DDMatrix ret, long int ret_index[4], DDMatrix mat_a, long int mat_a_index[4], DDMatrix mat_b, long int mat_b_index[4]);

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void sub_ddmatrix_partial(DDMatrix ret, long int ret_index[4], DDMatrix mat_a, long int mat_a_index[4], DDMatrix mat_b, long int mat_b_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_ddmatrix_partial(DDMatrix ret, long int ret_index[4], DDMatrix mat_a, long int mat_a_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_ddmatrix_partial_checked(DDMatrix ret, long int ret_index[4], DDMatrix mat_a, long int mat_a_index[4]);

// reverse sign
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void neg_ddmatrix_partial(DDMatrix ret, long int ret_index[4], DDMatrix mat_a, long int mat_a_index[4]);

/* c = a * b */
#define mul_ddmatrix_simple(c, a, b) mul_ddmatrix(c, a, b)

// Block matrix multiplicaiton
void mul_ddmatrix_block(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim);

// Padding to 2-powered dimensional matrix
DDMatrix init_static_padding_ddmatrix_strassen(DDMatrix orig_mat);

// Padding to even dimensional matrix
DDMatrix init_dynamic_padding_ddmatrix_strassen(DDMatrix orig_mat);

// Padding to even dimensional matrix
DDMatrix init_dynamic_padding_ddmatrix_strassen2(DDMatrix orig_mat, long int min_dim);

// Strassen's Algorithm with static padding
void mul_ddmatrix_strassen_odd_padding(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim);

// Fit dimension to be multiple of min_dim
void mul_ddmatrix_strassen(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim);

// Strassen's Algorithm with Dynamic peeling
void mul_ddmatrix_strassen_odd_peeling(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim);

// Strassen's Algorithm
void mul_ddmatrix_strassen_even(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim);

// Winograd Variant of Strassen's Algorithm
void mul_ddmatrix_winograd_even(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim);

// Computattion of Inverse Matrix by using Strassen's Algorithm
void inv_ddmatrix_strassen_even(DDMatrix ret, DDMatrix mat_a, long int min_dim);

/* scaling with the absolute maximum element in the row */
/* ret := scaling_diag_mat * org_mat -> ||ret|| \approx 1 */
void left_scaling_ddmatrix(DDMatrix ret, DDMatrix org_mat, DDVector scaling_diag_mat, long int *ret_col_index);

/* scaling with the absolute maximum element in the column */
/* ret := org_mat * scaling_diag_mat -> ||ret|| \approx 1 */
void right_scaling_ddmatrix(DDMatrix ret, DDMatrix org_mat, DDVector scaling_diag_mat, long int *ret_row_index);

/* multiply square matrix by diagonal matrix      */
/* ret = left_diag_mat^(-left_inv_flat) * org_mat * right_diag_mat^(-right_inv_flat) (left_diag_mat != null, right_diag_mat != null) */
/* ret = left_diag_mat^(-left_inv_flat) * org_mat * right_diag_mat^(-right_inv_flat) (left_diag_mat != null, right_diag_mat != null) */
/* ret = left_diag_mat^(-left_inv_flat) * org_mat                                    (left_diag_mat != null, right_diag_mat == null) */
/* ret =                                  org_mat * right_diag_mat^(-right_inv_flat) (left_diag_mat == null, right_diag_mat != null) */
void mul_ddmatrix_mpfdiag(DDMatrix ret, DDVector left_diag_mat, int left_inv_flag, DDMatrix org_mat, DDVector right_diag_mat, int right_inv_flag);

//--------------------------------------
// OpenMP
//--------------------------------------

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_add_ddmatrix_partial(DDMatrix ret, long int ret_index[4], DDMatrix mat_a, long int mat_a_index[4], DDMatrix mat_b, long int mat_b_index[4]);

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_sub_ddmatrix_partial(DDMatrix ret, long int ret_index[4], DDMatrix mat_a, long int mat_a_index[4], DDMatrix mat_b, long int mat_b_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_subst_ddmatrix_partial(DDMatrix ret, long int ret_index[4], DDMatrix mat_a, long int mat_a_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_subst_ddmatrix_partial_checked(DDMatrix ret, long int ret_index[4], DDMatrix mat_a, long int mat_a_index[4]);

// reverse sign
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_neg_ddmatrix_partial(DDMatrix ret, long int ret_index[4], DDMatrix mat_a, long int mat_a_index[4]);

/* c = a * b */
#define _bncomp_mul_ddmatrix_simple(c, a, b) _bncomp_mul_ddmatrix(c, a, b)
//void _bncomp_mul_ddmatrix_simple(DDMatrix ret, DDMatrix a, DDMatrix b);

// Block matrix multiplicaiton
void _bncomp_mul_ddmatrix_block(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim);

// Padding to 2-powered dimensional matrix
DDMatrix _bncomp_init_static_padding_ddmatrix_strassen(DDMatrix orig_mat);

// Padding to even dimensional matrix
DDMatrix _bncomp_init_dynamic_padding_ddmatrix_strassen(DDMatrix orig_mat);

// Padding to even dimensional matrix
DDMatrix _bncomp_init_dynamic_padding_ddmatrix_strassen2(DDMatrix orig_mat, long int min_dim);

// Strassen's Algorithm with static padding
void _bncomp_mul_ddmatrix_strassen_odd_padding(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim);

// Fit dimension to be multiple of min_dim
void _bncomp_mul_ddmatrix_strassen(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim);

// Strassen's Algorithm with Dynamic peeling
void _bncomp_mul_ddmatrix_strassen_odd_peeling(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim);

// Matrix multiplicaiton with Strassen's algorithm (Nonrecursive version in the area of Strassen's algorithms)
void _bncomp_mul_ddmatrix_strassen_nonrec(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim);

// Strassen's Algorithm
void _bncomp_mul_ddmatrix_strassen_even(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim);

// Strassen's Algorithm with parallized sections
void _bncomp_mul_ddmatrix_strassen_even_psec(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim);

// Strassen's Algorithm (parallelized sections)
void _bncomp_mul_ddmatrix_strassen_even2(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim, long int rec_num);

// Strassen's Algorithm (parallelizable tasks)
void _bncomp_mul_ddmatrix_strassen_even3(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim, long int rec_num);

// Winograd Variant of Strassen's Algorithm
void _bncomp_mul_ddmatrix_winograd_even(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim);

// Winograd Variant of Strassen's Algorithm with parallelied sections
void _bncomp_mul_ddmatrix_winograd_even_psec(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim);

// Winograd Variant of Strassen's Algorithm (parallelied sections)
void _bncomp_mul_ddmatrix_winograd_even2(DDMatrix ret, DDMatrix mat_a, DDMatrix mat_b, long int min_dim);

//--------------------------------------
// CUDA
//--------------------------------------
#ifdef USE_CUDA

#define mul_gddmatrix_simple(c, a, b, num_blocks, num_threads) mul_gddmatrix_dev((c), (a), (b), (num_blocks), (num_threads))
#define _bncuda_mul_gddmatrix_simple(c, a, b, num_blocks, num_threads) mul_gddmatrix_dev((c), (a), (b), (num_blocks), (num_threads))
#define _bncuda_set0_gddmatrix(a, num_blocks, num_threads) set0_gddmatrix_dev((a), (num_blocks), (num_threads))

// Suppose that all arguments are allocated on GPU
void _bncuda_add_gddmatrix_partial(GDDMatrix ret_dev, long int ret_index_dev[4], GDDMatrix mat_a_dev, long int mat_a_index_dev[4], GDDMatrix mat_b_dev, long int mat_b_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);


// partial sub
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncuda_sub_gddmatrix_partial(GDDMatrix ret_dev, long int ret_index_dev[4], GDDMatrix mat_a_dev, long int mat_a_index_dev[4], GDDMatrix mat_b_dev, long int mat_b_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncuda_subst_gddmatrix_partial(GDDMatrix ret_dev, long int ret_index_dev[4], GDDMatrix mat_a_dev, long int mat_a_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncuda_subst_gddmatrix_partial_checked(GDDMatrix ret_dev, long int ret_index_dev[4], GDDMatrix mat_a_dev, long int mat_a_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);

// reverse sign
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncuda_neg_gddmatrix_partial(GDDMatrix ret_dev, long int ret_index_dev[4], GDDMatrix mat_a_dev, long int mat_a_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);

// Block matrix multiplicaiton
void _bncuda_mul_gddmatrix_block(GDDMatrix ret_dev, GDDMatrix mat_a_dev, GDDMatrix mat_b_dev, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);

// Padding to even dimensional matrix
GDDMatrix _bncuda_init_dynamic_padding_gddmatrix_strassen2(GDDMatrix orig_mat_dev, long int min_dim);

// Strassen's Algorithm with static padding
void _bncuda_mul_gddmatrix_strassen_odd_padding(GDDMatrix ret_dev, GDDMatrix mat_a_dev, GDDMatrix mat_b_dev, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);

// Strassen's Algorithm with Dynamic peeling
void _bncuda_mul_gddmatrix_strassen_odd_peeling(GDDMatrix ret, GDDMatrix mat_a, GDDMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);

// clear counter
void _bncuda_reset_num_mul_gddmatrix_strassen(void);

// get counters
void _bncuda_get_num_mul_gddmatrix_strassen(long int *num_addsub, long int *num_mul);

// print counters
void _bncuda_print_num_mul_gddmatrix_strassen(long int *num_addsub, long int *num_mul);

// Fit dimension to be multiple of min_dim
void _bncuda_mul_gddmatrix_strassen(GDDMatrix ret, GDDMatrix mat_a, GDDMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);

// Strassen's Algorithm with parallized sections
void _bncuda_mul_gddmatrix_strassen_even_psec(GDDMatrix ret, GDDMatrix mat_a, GDDMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);

// Winograd Variant of Strassen's Algorithm with parallelied sections
void _bncuda_mul_gddmatrix_winograd_even_psec(GDDMatrix ret, GDDMatrix mat_a, GDDMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);

#endif // USE_CUDA

///////////////
///// QD //////
///////////////
#ifdef USE_QDLINEAR
// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void add_qdmatrix_partial(QDMatrix ret, long int ret_index[4], QDMatrix mat_a, long int mat_a_index[4], QDMatrix mat_b, long int mat_b_index[4]);

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void sub_qdmatrix_partial(QDMatrix ret, long int ret_index[4], QDMatrix mat_a, long int mat_a_index[4], QDMatrix mat_b, long int mat_b_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_qdmatrix_partial(QDMatrix ret, long int ret_index[4], QDMatrix mat_a, long int mat_a_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_qdmatrix_partial_checked(QDMatrix ret, long int ret_index[4], QDMatrix mat_a, long int mat_a_index[4]);

// reverse sign
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void neg_qdmatrix_partial(QDMatrix ret, long int ret_index[4], QDMatrix mat_a, long int mat_a_index[4]);

/* c = a * b */
#define mul_qdmatrix_simple(c, a, b) mul_qdmatrix(c, a, b)

// Block matrix multiplicaiton
void mul_qdmatrix_block(QDMatrix ret, QDMatrix mat_a, QDMatrix mat_b, long int min_dim);

// Padding to 2-powered dimensional matrix
QDMatrix init_static_padding_qdmatrix_strassen(QDMatrix orig_mat);

// Padding to even dimensional matrix
QDMatrix init_dynamic_padding_qdmatrix_strassen(QDMatrix orig_mat);

// Padding to even dimensional matrix
QDMatrix init_dynamic_padding_qdmatrix_strassen2(QDMatrix orig_mat, long int min_dim);

// Strassen's Algorithm with static padding
void mul_qdmatrix_strassen_odd_padding(QDMatrix ret, QDMatrix mat_a, QDMatrix mat_b, long int min_dim);

// Fit dimension to be multiple of min_dim
void mul_qdmatrix_strassen(QDMatrix ret, QDMatrix mat_a, QDMatrix mat_b, long int min_dim);

// Strassen's Algorithm with Dynamic peeling
void mul_qdmatrix_strassen_odd_peeling(QDMatrix ret, QDMatrix mat_a, QDMatrix mat_b, long int min_dim);

// Strassen's Algorithm
void mul_qdmatrix_strassen_even(QDMatrix ret, QDMatrix mat_a, QDMatrix mat_b, long int min_dim);

// Winograd Variant of Strassen's Algorithm
void mul_qdmatrix_winograd_even(QDMatrix ret, QDMatrix mat_a, QDMatrix mat_b, long int min_dim);

// Computattion of Inverse Matrix by using Strassen's Algorithm
void inv_qdmatrix_strassen_even(QDMatrix ret, QDMatrix mat_a, long int min_dim);

/* scaling with the absolute maximum element in the row */
/* ret := scaling_diag_mat * org_mat -> ||ret|| \approx 1 */
void left_scaling_qdmatrix(QDMatrix ret, QDMatrix org_mat, QDVector scaling_diag_mat, long int *ret_col_index);

/* scaling with the absolute maximum element in the column */
/* ret := org_mat * scaling_diag_mat -> ||ret|| \approx 1 */
void right_scaling_qdmatrix(QDMatrix ret, QDMatrix org_mat, QDVector scaling_diag_mat, long int *ret_row_index);

/* multiply square matrix by diagonal matrix      */
/* ret = left_diag_mat^(-left_inv_flat) * org_mat * right_diag_mat^(-right_inv_flat) (left_diag_mat != null, right_diag_mat != null) */
/* ret = left_diag_mat^(-left_inv_flat) * org_mat * right_diag_mat^(-right_inv_flat) (left_diag_mat != null, right_diag_mat != null) */
/* ret = left_diag_mat^(-left_inv_flat) * org_mat                                    (left_diag_mat != null, right_diag_mat == null) */
/* ret =                                  org_mat * right_diag_mat^(-right_inv_flat) (left_diag_mat == null, right_diag_mat != null) */
void mul_qdmatrix_qddiag(QDMatrix ret, QDVector left_diag_mat, int left_inv_flag, QDMatrix org_mat, QDVector right_diag_mat, int right_inv_flag);

//--------------------------------------
// OpenMP
//--------------------------------------

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_add_qdmatrix_partial(QDMatrix ret, long int ret_index[4], QDMatrix mat_a, long int mat_a_index[4], QDMatrix mat_b, long int mat_b_index[4]);

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_sub_qdmatrix_partial(QDMatrix ret, long int ret_index[4], QDMatrix mat_a, long int mat_a_index[4], QDMatrix mat_b, long int mat_b_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_subst_qdmatrix_partial(QDMatrix ret, long int ret_index[4], QDMatrix mat_a, long int mat_a_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_subst_qdmatrix_partial_checked(QDMatrix ret, long int ret_index[4], QDMatrix mat_a, long int mat_a_index[4]);

// reverse sign
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_neg_qdmatrix_partial(QDMatrix ret, long int ret_index[4], QDMatrix mat_a, long int mat_a_index[4]);

/* c = a * b */
#define _bncomp_mul_qdmatrix_simple(c, a, b) _bncomp_mul_qdmatrix(c, a, b)
//void _bncomp_mul_qdmatrix_simple(QDMatrix ret, QDMatrix a, QDMatrix b);

// Block matrix multiplicaiton
void _bncomp_mul_qdmatrix_block(QDMatrix ret, QDMatrix mat_a, QDMatrix mat_b, long int min_dim);

// Padding to 2-powered dimensional matrix
QDMatrix _bncomp_init_static_padding_qdmatrix_strassen(QDMatrix orig_mat);

// Padding to even dimensional matrix
QDMatrix _bncomp_init_dynamic_padding_qdmatrix_strassen(QDMatrix orig_mat);

// Padding to even dimensional matrix
QDMatrix _bncomp_init_dynamic_padding_qdmatrix_strassen2(QDMatrix orig_mat, long int min_dim);

// Strassen's Algorithm with static padding
void _bncomp_mul_qdmatrix_strassen_odd_padding(QDMatrix ret, QDMatrix mat_a, QDMatrix mat_b, long int min_dim);

// Fit dimension to be multiple of min_dim
void _bncomp_mul_qdmatrix_strassen(QDMatrix ret, QDMatrix mat_a, QDMatrix mat_b, long int min_dim);

// Strassen's Algorithm with Dynamic peeling
void _bncomp_mul_qdmatrix_strassen_odd_peeling(QDMatrix ret, QDMatrix mat_a, QDMatrix mat_b, long int min_dim);

// Matrix multiplicaiton with Strassen's algorithm (Nonrecursive version)
void _bncomp_mul_qdmatrix_strassen_nonrec(QDMatrix ret, QDMatrix mat_a, QDMatrix mat_b, long int min_dim);

// Strassen's Algorithm
void _bncomp_mul_qdmatrix_strassen_even(QDMatrix ret, QDMatrix mat_a, QDMatrix mat_b, long int min_dim);

// Strassen's Algorithm with parallelized sections
void _bncomp_mul_qdmatrix_strassen_even_psec(QDMatrix ret, QDMatrix mat_a, QDMatrix mat_b, long int min_dim);

// Strassen's Algorithm (parallelized sections)
void _bncomp_mul_qdmatrix_strassen_even2(QDMatrix ret, QDMatrix mat_a, QDMatrix mat_b, long int min_dim, long int rec_num);

// Strassen's Algorithm (parallelizable tasks)
void _bncomp_mul_qdmatrix_strassen_even3(QDMatrix ret, QDMatrix mat_a, QDMatrix mat_b, long int min_dim, long int rec_num);

// Winograd Variant of Strassen's Algorithm
void _bncomp_mul_qdmatrix_winograd_even(QDMatrix ret, QDMatrix mat_a, QDMatrix mat_b, long int min_dim);

// Winograd Variant of Strassen's Algorithm with parallelized sections
void _bncomp_mul_qdmatrix_winograd_even_psec(QDMatrix ret, QDMatrix mat_a, QDMatrix mat_b, long int min_dim);

// Winograd Variant of Strassen's Algorithm (parallelized sections)
void _bncomp_mul_qdmatrix_winograd_even2(QDMatrix ret, QDMatrix mat_a, QDMatrix mat_b, long int min_dim);
#endif // USE_QDLINEAR

///////////////
///// TD //////
///////////////
#ifdef USE_TDLINEAR

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void add_tdmatrix_partial(TDMatrix ret, long int ret_index[4], TDMatrix mat_a, long int mat_a_index[4], TDMatrix mat_b, long int mat_b_index[4]);

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void sub_tdmatrix_partial(TDMatrix ret, long int ret_index[4], TDMatrix mat_a, long int mat_a_index[4], TDMatrix mat_b, long int mat_b_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_tdmatrix_partial(TDMatrix ret, long int ret_index[4], TDMatrix mat_a, long int mat_a_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_tdmatrix_partial_checked(TDMatrix ret, long int ret_index[4], TDMatrix mat_a, long int mat_a_index[4]);

// reverse sign
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void neg_tdmatrix_partial(TDMatrix ret, long int ret_index[4], TDMatrix mat_a, long int mat_a_index[4]);

/* c = a * b */
#define mul_tdmatrix_simple(c, a, b) mul_tdmatrix(c, a, b)

// Block matrix multiplicaiton
void mul_tdmatrix_block(TDMatrix ret, TDMatrix mat_a, TDMatrix mat_b, long int min_dim);

// Padding to 2-powered dimensional matrix
TDMatrix init_static_padding_tdmatrix_strassen(TDMatrix orig_mat);

// Padding to even dimensional matrix
TDMatrix init_dynamic_padding_tdmatrix_strassen(TDMatrix orig_mat);

// Padding to even dimensional matrix
TDMatrix init_dynamic_padding_tdmatrix_strassen2(TDMatrix orig_mat, long int min_dim);

// Strassen's Algorithm with static padding
void mul_tdmatrix_strassen_odd_padding(TDMatrix ret, TDMatrix mat_a, TDMatrix mat_b, long int min_dim);

// Fit dimension to be multiple of min_dim
void mul_tdmatrix_strassen(TDMatrix ret, TDMatrix mat_a, TDMatrix mat_b, long int min_dim);

// Strassen's Algorithm with Dynamic peeling
void mul_tdmatrix_strassen_odd_peeling(TDMatrix ret, TDMatrix mat_a, TDMatrix mat_b, long int min_dim);

// Strassen's Algorithm
void mul_tdmatrix_strassen_even(TDMatrix ret, TDMatrix mat_a, TDMatrix mat_b, long int min_dim);

// Winograd Variant of Strassen's Algorithm
void mul_tdmatrix_winograd_even(TDMatrix ret, TDMatrix mat_a, TDMatrix mat_b, long int min_dim);

// Computattion of Inverse Matrix by using Strassen's Algorithm
void inv_tdmatrix_strassen_even(TDMatrix ret, TDMatrix mat_a, long int min_dim);

/* scaling with the absolute maximum element in the row */
/* ret := scaling_diag_mat * org_mat -> ||ret|| \approx 1 */
void left_scaling_tdmatrix(TDMatrix ret, TDMatrix org_mat, TDVector scaling_diag_mat, long int *ret_col_index);

/* scaling with the absolute maximum element in the column */
/* ret := org_mat * scaling_diag_mat -> ||ret|| \approx 1 */
void right_scaling_tdmatrix(TDMatrix ret, TDMatrix org_mat, TDVector scaling_diag_mat, long int *ret_row_index);

/* multiply square matrix by diagonal matrix      */
/* ret = left_diag_mat^(-left_inv_flat) * org_mat * right_diag_mat^(-right_inv_flat) (left_diag_mat != null, right_diag_mat != null) */
/* ret = left_diag_mat^(-left_inv_flat) * org_mat * right_diag_mat^(-right_inv_flat) (left_diag_mat != null, right_diag_mat != null) */
/* ret = left_diag_mat^(-left_inv_flat) * org_mat                                    (left_diag_mat != null, right_diag_mat == null) */
/* ret =                                  org_mat * right_diag_mat^(-right_inv_flat) (left_diag_mat == null, right_diag_mat != null) */
void mul_tdmatrix_tddiag(TDMatrix ret, TDVector left_diag_mat, int left_inv_flag, TDMatrix org_mat, TDVector right_diag_mat, int right_inv_flag);

//--------------------------------------
// OpenMP
//--------------------------------------

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_add_tdmatrix_partial(TDMatrix ret, long int ret_index[4], TDMatrix mat_a, long int mat_a_index[4], TDMatrix mat_b, long int mat_b_index[4]);

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_sub_tdmatrix_partial(TDMatrix ret, long int ret_index[4], TDMatrix mat_a, long int mat_a_index[4], TDMatrix mat_b, long int mat_b_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_subst_tdmatrix_partial(TDMatrix ret, long int ret_index[4], TDMatrix mat_a, long int mat_a_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_subst_tdmatrix_partial_checked(TDMatrix ret, long int ret_index[4], TDMatrix mat_a, long int mat_a_index[4]);

// reverse sign
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_neg_tdmatrix_partial(TDMatrix ret, long int ret_index[4], TDMatrix mat_a, long int mat_a_index[4]);

/* c = a * b */
#define _bncomp_mul_tdmatrix_simple(c, a, b) _bncomp_mul_tdmatrix(c, a, b)
//void _bncomp_mul_tdmatrix_simple(TDMatrix ret, TDMatrix a, TDMatrix b);

// Block matrix multiplicaiton
void _bncomp_mul_tdmatrix_block(TDMatrix ret, TDMatrix mat_a, TDMatrix mat_b, long int min_dim);

// Padding to 2-powered dimensional matrix
TDMatrix _bncomp_init_static_padding_tdmatrix_strassen(TDMatrix orig_mat);

// Padding to even dimensional matrix
TDMatrix _bncomp_init_dynamic_padding_tdmatrix_strassen(TDMatrix orig_mat);

// Padding to even dimensional matrix
TDMatrix _bncomp_init_dynamic_padding_tdmatrix_strassen2(TDMatrix orig_mat, long int min_dim);

// Strassen's Algorithm with static padding
void _bncomp_mul_tdmatrix_strassen_odd_padding(TDMatrix ret, TDMatrix mat_a, TDMatrix mat_b, long int min_dim);

// Fit dimension to be multiple of min_dim
void _bncomp_mul_tdmatrix_strassen(TDMatrix ret, TDMatrix mat_a, TDMatrix mat_b, long int min_dim);

// Strassen's Algorithm with Dynamic peeling
void _bncomp_mul_tdmatrix_strassen_odd_peeling(TDMatrix ret, TDMatrix mat_a, TDMatrix mat_b, long int min_dim);

// Matrix multiplicaiton with Strassen's algorithm (Nonrecursive version)
void _bncomp_mul_tdmatrix_strassen_nonrec(TDMatrix ret, TDMatrix mat_a, TDMatrix mat_b, long int min_dim);

// Strassen's Algorithm
void _bncomp_mul_tdmatrix_strassen_even(TDMatrix ret, TDMatrix mat_a, TDMatrix mat_b, long int min_dim);

// Strassen's Algorithm with parallelized sections
void _bncomp_mul_tdmatrix_strassen_even_psec(TDMatrix ret, TDMatrix mat_a, TDMatrix mat_b, long int min_dim);

// Strassen's Algorithm (parallelized sections)
void _bncomp_mul_tdmatrix_strassen_even2(TDMatrix ret, TDMatrix mat_a, TDMatrix mat_b, long int min_dim, long int rec_num);

// Strassen's Algorithm (parallelizable tasks)
void _bncomp_mul_tdmatrix_strassen_even3(TDMatrix ret, TDMatrix mat_a, TDMatrix mat_b, long int min_dim, long int rec_num);

// Winograd Variant of Strassen's Algorithm
void _bncomp_mul_tdmatrix_winograd_even(TDMatrix ret, TDMatrix mat_a, TDMatrix mat_b, long int min_dim);

// Winograd Variant of Strassen's Algorithm with parallelized sections
void _bncomp_mul_tdmatrix_winograd_even_psec(TDMatrix ret, TDMatrix mat_a, TDMatrix mat_b, long int min_dim);

// Winograd Variant of Strassen's Algorithm (parallelized sections)
void _bncomp_mul_tdmatrix_winograd_even2(TDMatrix ret, TDMatrix mat_a, TDMatrix mat_b, long int min_dim);
#endif // USE_TDLINEAR
#endif // USE_DDLINEAR

//--------------------------------------
// CUDA
//--------------------------------------
#ifdef USE_CUDA

#define mul_gqdmatrix_simple(c, a, b, num_blocks, num_threads) mul_gqdmatrix_dev((c), (a), (b), (num_blocks), (num_threads))
#define _bncuda_mul_gqdmatrix_simple(c, a, b, num_blocks, num_threads) mul_gqdmatrix_dev((c), (a), (b), (num_blocks), (num_threads))
#define _bncuda_set0_gqdmatrix(a, num_blocks, num_threads) set0_gqdmatrix_dev((a), (num_blocks), (num_threads))

// Suppose that all arguments are allocated on GPU
void _bncuda_add_gqdmatrix_partial(GQDMatrix ret_dev, long int ret_index_dev[4], GQDMatrix mat_a_dev, long int mat_a_index_dev[4], GQDMatrix mat_b_dev, long int mat_b_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);

// partial sub
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncuda_sub_gqdmatrix_partial(GQDMatrix ret_dev, long int ret_index_dev[4], GQDMatrix mat_a_dev, long int mat_a_index_dev[4], GQDMatrix mat_b_dev, long int mat_b_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncuda_subst_gqdmatrix_partial(GQDMatrix ret_dev, long int ret_index_dev[4], GQDMatrix mat_a_dev, long int mat_a_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncuda_subst_gqdmatrix_partial_checked(GQDMatrix ret_dev, long int ret_index_dev[4], GQDMatrix mat_a_dev, long int mat_a_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);

// reverse sign
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncuda_neg_gqdmatrix_partial(GQDMatrix ret_dev, long int ret_index_dev[4], GQDMatrix mat_a_dev, long int mat_a_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);

// Block matrix multiplicaiton
void _bncuda_mul_gqdmatrix_block(GQDMatrix ret_dev, GQDMatrix mat_a_dev, GQDMatrix mat_b_dev, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);

// Padding to even dimensional matrix
GQDMatrix _bncuda_init_dynamic_padding_gqdmatrix_strassen2(GQDMatrix orig_mat_dev, long int min_dim);

// Strassen's Algorithm with static padding
void _bncuda_mul_gqdmatrix_strassen_odd_padding(GQDMatrix ret_dev, GQDMatrix mat_a_dev, GQDMatrix mat_b_dev, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);

// Strassen's Algorithm with Dynamic peeling
void _bncuda_mul_gqdmatrix_strassen_odd_peeling(GQDMatrix ret, GQDMatrix mat_a, GQDMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);

// clear counter
void _bncuda_reset_num_mul_gqdmatrix_strassen(void);

// get counters
void _bncuda_get_num_mul_gqdmatrix_strassen(long int *num_addsub, long int *num_mul);

// print counters
void _bncuda_print_num_mul_gqdmatrix_strassen(long int *num_addsub, long int *num_mul);

// Fit dimension to be multiple of min_dim
void _bncuda_mul_gqdmatrix_strassen(GQDMatrix ret, GQDMatrix mat_a, GQDMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);

// Strassen's Algorithm with parallized sections
void _bncuda_mul_gqdmatrix_strassen_even_psec(GQDMatrix ret, GQDMatrix mat_a, GQDMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);

// Winograd Variant of Strassen's Algorithm with parallelied sections
void _bncuda_mul_gqdmatrix_winograd_even_psec(GQDMatrix ret, GQDMatrix mat_a, GQDMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);

#endif // USE_CUDA

//--------------------------------------
// OpenMP
//--------------------------------------
#ifdef USE_GMP
// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_add_mpfmatrix_partial(MPFMatrix ret, long int ret_index[4], MPFMatrix mat_a, long int mat_a_index[4], MPFMatrix mat_b, long int mat_b_index[4]);

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_sub_mpfmatrix_partial(MPFMatrix ret, long int ret_index[4], MPFMatrix mat_a, long int mat_a_index[4], MPFMatrix mat_b, long int mat_b_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_subst_mpfmatrix_partial(MPFMatrix ret, long int ret_index[4], MPFMatrix mat_a, long int mat_a_index[4]);


// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_subst_mpfmatrix_partial_checked(MPFMatrix ret, long int ret_index[4], MPFMatrix mat_a, long int mat_a_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_neg_mpfmatrix_partial(MPFMatrix ret, long int ret_index[4], MPFMatrix mat_a, long int mat_a_index[4]);

/* c = a * b */
void _bncomp_mul_mpfmatrix_simple(MPFMatrix c, MPFMatrix a, MPFMatrix b);

// Block matrix multiplicaiton
void _bncomp_mul_mpfmatrix_block(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim);

// Padding to 2-powered dimensional matrix
MPFMatrix _bncomp_init_static_padding_mpfmatrix_strassen(MPFMatrix orig_mat);

// Padding to even dimensional matrix
MPFMatrix _bncomp_init_dynamic_padding_mpfmatrix_strassen(MPFMatrix orig_mat);

// Padding to even dimensional matrix
MPFMatrix _bncomp_init_dynamic_padding_mpfmatrix_strassen2(MPFMatrix orig_mat, long int min_dim);

// Strassen's Algorithm with static padding
void _bncomp_mul_mpfmatrix_strassen_odd_padding(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim);

// clear counter
void _bncomp_reset_num_mul_mpfmatrix_strassen(void);

// get counters
void _bncomp_get_num_mul_mpfmatrix_strassen(long int *_bncomp_num_addsub, long int *_bncomp_num_mul);

// print counters
void _bncomp_print_num_mul_mpfmatrix_strassen(long int *_bncomp_num_addsub, long int *_bncomp_num_mul);

// Fit dimension to be multiple of min_dim
void _bncomp_mul_mpfmatrix_strassen(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim);

// Strassen's Algorithm with Dynamic peeling
void _bncomp_mul_mpfmatrix_strassen_odd_peeling(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim);

// Matrix multiplicaiton with Strassen's algorithm (Nonrecursive version in the area of Strassen's algorithms)
void _bncomp_mul_mpfmatrix_strassen_nonrec(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim);

// Strassen's Algorithm
void _bncomp_mul_mpfmatrix_strassen_even(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim);

// Strassen's Algorithm with parallelized sections
void _bncomp_mul_mpfmatrix_strassen_even_psec(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim);

// Strassen's Algorithm (parallelized sections)
void _bncomp_mul_mpfmatrix_strassen_even2(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim, long int rec_num);

// Strassen's Algorithm (parallelizable tasks)
void _bncomp_mul_mpfmatrix_strassen_even3(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim, long int rec_num);

// Winograd Variant of Strassen's Algorithm
void _bncomp_mul_mpfmatrix_winograd_even(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim);

// Winograd Variant of Strassen's Algorithm with parallelized sections
void _bncomp_mul_mpfmatrix_winograd_even_psec(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim);

// Winograd Variant of Strassen's Algorithm (parallelized sections)
void _bncomp_mul_mpfmatrix_winograd_even2(MPFMatrix ret, MPFMatrix mat_a, MPFMatrix mat_b, long int min_dim);

#endif // USE_GMP

// -------------------
// benchmark_tools.c
// -------------------
// relative errors for double precision matrix
void relerr3_dmatrix(double *max_relerr, double *min_relerr, double *norm_relerr, DMatrix mat, DMatrix mat_true, int kind_of_norm);

#ifdef USE_DDLINEAR
	// ANSI C
	// relative errors for DD or QD matrix
	void relerr3_ddmatrix(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double norm_relerr[DDSIZE], DDMatrix mat, DDMatrix mat_true, int kind_of_norm);
#ifdef USE_TDLINEAR
	void relerr3_tdmatrix(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double norm_relerr[TDSIZE], TDMatrix mat, TDMatrix mat_true, int kind_of_norm);
#endif // USE_TDLINEAR
#ifdef USE_QDLINEAR
	void relerr3_qdmatrix(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double norm_relerr[QDSIZE], QDMatrix mat, QDMatrix mat_true, int kind_of_norm);
#endif // USE_QDLINEAR
#endif // USE_DDLINEAR

#ifdef USE_GMP
// relative errors for MPF matrix
void relerr3_mpfmatrix(mpf_t max_relerr, mpf_t min_relerr, mpf_t norm_relerr, MPFMatrix mat, MPFMatrix mat_true, int kind_of_norm);
#endif // USE_GMP

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // define _BNC_MATMUL_STRASSEN_H
