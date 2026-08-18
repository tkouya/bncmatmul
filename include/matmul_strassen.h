/********************************************************************************/
/* matmul_strassen.h: Fast Matrix Multiplication based on double, double-double */
/*                    and Multiple precision floating-point arithmetics         */
/* Copyright (C) 2014-2022 Tomonori Kouya                                       */
/*                                                                              */
/* Version 0.19 : First implementation of parallelized matrix mutiplication     */
/* Version 0.2  : DD and QD precision have been supported                       */
/* Version 0.21 : Bug fix in mul_mpfmatrix_strassen_odd_peeling                 */
/* Version 0.3b0: AVX2 and AVX-512 are applied for any codes                    */
/* Version 0.3  : bnc_print_env_all has been implemented                        */
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
#include <ccomplex>
#else // __cplusplus
#include <stdio.h>
#include <math.h>
#include <complex.h>
#endif // __cplusplus

//#ifdef USE_DDLINEAR
#include "dlinear.h"

#ifdef USE_IMKL
	#include "mkl.h"
	#include "mkl_cblas.h" // for Intel Math Kernel Library
#endif // USE_IMKL

#ifdef USE_DDLINEAR
	#include "ddlinear.h" // double-double and quadratic double precision
	#include "cddlinear.h" // Complex DD precision
#endif // USE_DDLINEAR
#ifdef USE_TDLINEAR
	#include "tdlinear.h"
	#include "ctdlinear.h"
#endif // USE_TDLINEAR
#ifdef USE_QDLINEAR
	#include "qdlinear.h"
	#include "cqdlinear.h"
#endif // USE_QDLINEAR
#ifdef USE_DSLINEAR
	#include "dslinear.h" // double-single precision
#endif // USE_DSLINEAR
#ifdef USE_TSLINEAR
	#include "tslinear.h" // triple-single precision
#endif // USE_TSLINEAR
#ifdef USE_QSLINEAR
	#include "qslinear.h" // quad-single precision
#endif // USE_QSLINEAR

#ifdef USE_CUDA
	#include "gddlinear.h"
	#ifdef USE_TDLINEAR
		#include "gtdlinear.h"   // GTDMatrix / GTDVector for the GTD CUDA prototypes below
	#endif
	#ifdef USE_DSLINEAR
		#include "gdslinear.h"   // GDSMatrix / GQSMatrix for the GDS/GQS CUDA prototypes below
	#endif
	#ifdef USE_TSLINEAR
		#include "gtslinear.h"   // GTSMatrix for the GTS CUDA prototypes below
	#endif
	#ifdef USE_QSLINEAR
		#include "gqslinear.h"   // (forwards to gdslinear.h)
	#endif
#endif // USE_CUDA
//#endif // USE_DDLINEAR

#ifdef USE_GMP
	#include "gmp.h"
	#ifdef USE_MPFR
		#include "mpfr.h"
		#include "mpf2mpfr.h"
	#endif // USE_MPFR
	#include "mpflinear.h"
#endif // USE_GMP

// Complex linear computation
//#include "clinear.h"
#include "cdlinear.h"
#include "cmpflinear.h"

// OpenMP
#ifdef _OPENMP
#include "bncomp.h"
#endif // _OPENMP

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

#ifndef STRASSEN_MIN_DIM
#define STRASSEN_MIN_DIM _BNC_DEFAULT_MIN_DIM_STRASSEN
#endif // STRASSEN_MIN_DIM

// BNC common functions
//#include "bnc_common.h"

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

// multiple precision: Real matrix multiplication
#ifdef USE_GMP
#ifndef __NUM_ADDSUB_MUL_MPFMATRIX_STRASSEN
#define __NUM_ADDSUB_MUL_MPFMATRIX_STRASSEN
extern long int num_addsub_mul_mpfmatrix_strassen;	// addition and subtraction
extern long int num_mul_mul_mpfmatrix_strassen;		// multiplication
#endif // __NUM_ADDSUB_MUL_MPFMATRIX_STRASSEN

// multiple precision: Complex matrix multiplication
#ifndef __NUM_ADDSUB_MUL_CMPFMATRIX_STRASSEN
#define __NUM_ADDSUB_MUL_CMPFMATRIX_STRASSEN
extern long int num_addsub_mul_cmpfmatrix_strassen;	// addition and subtraction
extern long int num_mul_mul_cmpfmatrix_strassen;	// multiplication
#endif // __NUM_ADDSUB_MUL_CMPFMATRIX_STRASSEN

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

/************************************************/
/* Complex matrix multiplication                */
/************************************************/

// MPF & MPFR
#ifdef USE_GMP

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void add_cmpfmatrix_partial(CMPFMatrix ret, long int ret_index[4], CMPFMatrix mat_a, long int mat_a_index[4], CMPFMatrix mat_b, long int mat_b_index[4]);

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void sub_cmpfmatrix_partial(CMPFMatrix ret, long int ret_index[4], CMPFMatrix mat_a, long int mat_a_index[4], CMPFMatrix mat_b, long int mat_b_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_cmpfmatrix_partial(CMPFMatrix ret, long int ret_index[4], CMPFMatrix mat_a, long int mat_a_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_cmpfmatrix_partial_checked(CMPFMatrix ret, long int ret_index[4], CMPFMatrix mat_a, long int mat_a_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void neg_cmpfmatrix_partial(CMPFMatrix ret, long int ret_index[4], CMPFMatrix mat_a, long int mat_a_index[4]);

/* c = a * b */
void mul_cmpfmatrix_simple(CMPFMatrix c, CMPFMatrix a, CMPFMatrix b);

// Block matrix multiplicaiton
void mul_cmpfmatrix_block(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim);

// Padding to 2-powered dimensional matrix
CMPFMatrix init_static_padding_cmpfmatrix_strassen(CMPFMatrix orig_mat);

// Padding to even dimensional matrix
CMPFMatrix init_dynamic_padding_cmpfmatrix_strassen(CMPFMatrix orig_mat);

// Strassen's Algorithm with static or dynamic padding
void mul_cmpfmatrix_strassen_odd_padding(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim);

// Fit dimension to be multiple of min_dim
void mul_cmpfmatrix_strassen(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim);

// Strassen's Algorithm (even dimension)
void mul_cmpfmatrix_strassen_even(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim);

// Winograd Variant of Strassen's Algorithm
void mul_cmpfmatrix_winograd_even(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim);

// Strassen's Algorithm with dynamic peeling
void mul_cmpfmatrix_strassen_odd_peeling(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim);

// Computation of Inverse Matrix by using Strassen's Algorithm
void inv_cmpfmatrix_strassen_even(CMPFMatrix ret, CMPFMatrix mat_a, long int min_dim);

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

// Complex

/* c = a * b */
#define mul_cddmatrix_simple(c, a, b) mul_cddmatrix(c, a, b)

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void add_cddmatrix_partial(CDDMatrix ret, long int ret_index[4], CDDMatrix mat_a, long int mat_a_index[4], CDDMatrix mat_b, long int mat_b_index[4]);

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void sub_cddmatrix_partial(CDDMatrix ret, long int ret_index[4], CDDMatrix mat_a, long int mat_a_index[4], CDDMatrix mat_b, long int mat_b_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_cddmatrix_partial(CDDMatrix ret, long int ret_index[4], CDDMatrix mat_a, long int mat_a_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_cddmatrix_partial_checked(CDDMatrix ret, long int ret_index[4], CDDMatrix mat_a, long int mat_a_index[4]);

// reverse sign
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void neg_cddmatrix_partial(CDDMatrix ret, long int ret_index[4], CDDMatrix mat_a, long int mat_a_index[4]);

// Block matrix multiplicaiton
void mul_cddmatrix_block_3m(CDDMatrix ret, CDDMatrix mat_a, CDDMatrix mat_b, long int min_dim);
void mul_cddmatrix_block_4m(CDDMatrix ret, CDDMatrix mat_a, CDDMatrix mat_b, long int min_dim);
#ifdef USE_4M
#define mul_cddmatrix_block mul_cddmatrix_block_4m
#else // USE_4M
#define mul_cddmatrix_block mul_cddmatrix_block_3m
#endif // USE_4M

// Strassen
void mul_cddmatrix_strassen_3m(CDDMatrix ret, CDDMatrix mat_a, CDDMatrix mat_b, long int min_dim);
void mul_cddmatrix_strassen_4m(CDDMatrix ret, CDDMatrix mat_a, CDDMatrix mat_b, long int min_dim);
#ifdef USE_4M
#define mul_cddmatrix_strassen mul_cddmatrix_strassen_4m
#else // USE_4M
#define mul_cddmatrix_strassen mul_cddmatrix_strassen_3m
#endif // USE_4M

//--------------------------------------
// OpenMP
//--------------------------------------

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_add_ddmatrix_partial(DDMatrix ret, long int ret_index[4], DDMatrix mat_a, long int mat_a_index[4], DDMatrix mat_b, long int mat_b_index[4]);
void _bncomp_add_cddmatrix_partial(CDDMatrix ret, long int ret_index[4], CDDMatrix mat_a, long int mat_a_index[4], CDDMatrix mat_b, long int mat_b_index[4]);

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_sub_ddmatrix_partial(DDMatrix ret, long int ret_index[4], DDMatrix mat_a, long int mat_a_index[4], DDMatrix mat_b, long int mat_b_index[4]);
void _bncomp_sub_cddmatrix_partial(CDDMatrix ret, long int ret_index[4], CDDMatrix mat_a, long int mat_a_index[4], CDDMatrix mat_b, long int mat_b_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_subst_ddmatrix_partial(DDMatrix ret, long int ret_index[4], DDMatrix mat_a, long int mat_a_index[4]);
void _bncomp_subst_cddmatrix_partial(CDDMatrix ret, long int ret_index[4], CDDMatrix mat_a, long int mat_a_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_subst_ddmatrix_partial_checked(DDMatrix ret, long int ret_index[4], DDMatrix mat_a, long int mat_a_index[4]);
void _bncomp_subst_cddmatrix_partial_checked(CDDMatrix ret, long int ret_index[4], CDDMatrix mat_a, long int mat_a_index[4]);

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


// Complex with OpenMP

/* c = a * b */
#ifdef USE_4M
#define _bncomp_mul_cddmatrix(c, a, b) _bncomp_mul_cddmatrix_4m(c, a, b)
#else // USE_4M
#define _bncomp_mul_cddmatrix(c, a, b) _bncomp_mul_cddmatrix_3m(c, a, b)
#endif // USE_4M
#define _bncomp_mul_cddmatrix_simple(c, a, b) _bncomp_mul_cddmatrix(c, a, b)

// Block matrix multiplicaiton
void _bncomp_mul_cddmatrix_block_3m(CDDMatrix ret, CDDMatrix mat_a, CDDMatrix mat_b, long int min_dim);
void _bncomp_mul_cddmatrix_block_4m(CDDMatrix ret, CDDMatrix mat_a, CDDMatrix mat_b, long int min_dim);
#ifdef USE_4M
#define _bncomp_mul_cddmatrix_block  _bncomp_mul_cddmatrix_block_4m
#else // USE_4M
#define _bncomp_mul_cddmatrix_block  _bncomp_mul_cddmatrix_block_3m
#endif // USE_4M

// Strassen
void _bncomp_mul_cddmatrix_strassen_3m(CDDMatrix ret, CDDMatrix mat_a, CDDMatrix mat_b, long int min_dim);
void _bncomp_mul_cddmatrix_strassen_4m(CDDMatrix ret, CDDMatrix mat_a, CDDMatrix mat_b, long int min_dim);
#ifdef USE_4M
#define _bncomp_mul_cddmatrix_strassen _bncomp_mul_cddmatrix_strassen_4m
#else // USE_4M
#define _bncomp_mul_cddmatrix_strassen _bncomp_mul_cddmatrix_strassen_3m
#endif // USE_4M

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
#endif // USE_DDLINEAR

///////////////
///// QD //////
///////////////
#ifdef USE_QDLINEAR
#define mul_cqdmatrix_simple(c, a, b) mul_cqdmatrix_3m(c, a, b)

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void add_cqdmatrix_partial(CQDMatrix ret, long int ret_index[4], CQDMatrix mat_a, long int mat_a_index[4], CQDMatrix mat_b, long int mat_b_index[4]);

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void sub_cqdmatrix_partial(CQDMatrix ret, long int ret_index[4], CQDMatrix mat_a, long int mat_a_index[4], CQDMatrix mat_b, long int mat_b_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_cqdmatrix_partial(CQDMatrix ret, long int ret_index[4], CQDMatrix mat_a, long int mat_a_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_cqdmatrix_partial_checked(CQDMatrix ret, long int ret_index[4], CQDMatrix mat_a, long int mat_a_index[4]);

// reverse sign
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void neg_cqdmatrix_partial(CQDMatrix ret, long int ret_index[4], CQDMatrix mat_a, long int mat_a_index[4]);

// Block matrix multiplicaiton
void mul_cqdmatrix_block_3m(CQDMatrix ret, CQDMatrix mat_a, CQDMatrix mat_b, long int min_dim);
void mul_cqdmatrix_block_4m(CQDMatrix ret, CQDMatrix mat_a, CQDMatrix mat_b, long int min_dim);
#ifdef USE_4M
#define mul_cqdmatrix_block mul_cqdmatrix_block_4m
#else // USE_4M
#define mul_cqdmatrix_block mul_cqdmatrix_block_3m
#endif // USE_4M

// Strassen
void mul_cqdmatrix_strassen_3m(CQDMatrix ret, CQDMatrix mat_a, CQDMatrix mat_b, long int min_dim);
void mul_cqdmatrix_strassen_4m(CQDMatrix ret, CQDMatrix mat_a, CQDMatrix mat_b, long int min_dim);
#ifdef USE_4M
#define mul_cqdmatrix_strassen mul_cqdmatrix_strassen_4m
#else // USE_4M
#define mul_cqdmatrix_strassen mul_cqdmatrix_strassen_3m
#endif // USE_4M

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
void _bncomp_add_cqdmatrix_partial(CQDMatrix ret, long int ret_index[4], CQDMatrix mat_a, long int mat_a_index[4], CQDMatrix mat_b, long int mat_b_index[4]);

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_sub_qdmatrix_partial(QDMatrix ret, long int ret_index[4], QDMatrix mat_a, long int mat_a_index[4], QDMatrix mat_b, long int mat_b_index[4]);
void _bncomp_sub_cqdmatrix_partial(CQDMatrix ret, long int ret_index[4], CQDMatrix mat_a, long int mat_a_index[4], CQDMatrix mat_b, long int mat_b_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_subst_qdmatrix_partial(QDMatrix ret, long int ret_index[4], QDMatrix mat_a, long int mat_a_index[4]);
void _bncomp_subst_cqdmatrix_partial(CQDMatrix ret, long int ret_index[4], CQDMatrix mat_a, long int mat_a_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_subst_qdmatrix_partial_checked(QDMatrix ret, long int ret_index[4], QDMatrix mat_a, long int mat_a_index[4]);
void _bncomp_subst_cqdmatrix_partial_checked(CQDMatrix ret, long int ret_index[4], CQDMatrix mat_a, long int mat_a_index[4]);

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

// Complex with OpenMP

/* c = a * b */
#ifdef USE_4M
#define _bncomp_mul_cqdmatrix(c, a, b) _bncomp_mul_cqdmatrix_4m(c, a, b)
#else // USE_4M
#define _bncomp_mul_cqdmatrix(c, a, b) _bncomp_mul_cqdmatrix_3m(c, a, b)
#endif // USE_4M
#define _bncomp_mul_cqdmatrix_simple(c, a, b) _bncomp_mul_cqdmatrix(c, a, b)

// Block matrix multiplicaiton
void _bncomp_mul_cqdmatrix_block_3m(CQDMatrix ret, CQDMatrix mat_a, CQDMatrix mat_b, long int min_dim);
void _bncomp_mul_cqdmatrix_block_4m(CQDMatrix ret, CQDMatrix mat_a, CQDMatrix mat_b, long int min_dim);
#ifdef USE_4M
#define _bncomp_mul_cqdmatrix_block _bncomp_mul_cqdmatrix_block_4m
#else // USE_4M
#define _bncomp_mul_cqdmatrix_block _bncomp_mul_cqdmatrix_block_3m
#endif // USE_4M

// Strassen
void _bncomp_mul_cqdmatrix_strassen_3m(CQDMatrix ret, CQDMatrix mat_a, CQDMatrix mat_b, long int min_dim);
void _bncomp_mul_cqdmatrix_strassen_4m(CQDMatrix ret, CQDMatrix mat_a, CQDMatrix mat_b, long int min_dim);
#ifdef USE_4M
#define _bncomp_mul_cqdmatrix_strassen _bncomp_mul_cqdmatrix_strassen_4m
#else // USE_4M
#define _bncomp_mul_cqdmatrix_strassen _bncomp_mul_cqdmatrix_strassen_3m
#endif // USE_4M


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
void add_ctdmatrix_partial(CTDMatrix ret, long int ret_index[4], CTDMatrix mat_a, long int mat_a_index[4], CTDMatrix mat_b, long int mat_b_index[4]);

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void sub_ctdmatrix_partial(CTDMatrix ret, long int ret_index[4], CTDMatrix mat_a, long int mat_a_index[4], CTDMatrix mat_b, long int mat_b_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_ctdmatrix_partial(CTDMatrix ret, long int ret_index[4], CTDMatrix mat_a, long int mat_a_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void subst_ctdmatrix_partial_checked(CTDMatrix ret, long int ret_index[4], CTDMatrix mat_a, long int mat_a_index[4]);

// reverse sign
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void neg_ctdmatrix_partial(CTDMatrix ret, long int ret_index[4], CTDMatrix mat_a, long int mat_a_index[4]);

#ifdef USE_4M
#define mul_ctdmatrix_simple(c, a, b) mul_ctdmatrix_4m(c, a, b)
#else // USE_4M
#define mul_ctdmatrix_simple(c, a, b) mul_ctdmatrix_3m(c, a, b)
#endif // USE_4M

// Block matrix multiplicaiton
void mul_ctdmatrix_block_3m(CTDMatrix ret, CTDMatrix mat_a, CTDMatrix mat_b, long int min_dim);
void mul_ctdmatrix_block_4m(CTDMatrix ret, CTDMatrix mat_a, CTDMatrix mat_b, long int min_dim);
#ifdef USE_4M
#define mul_ctdmatrix_block mul_ctdmatrix_block_4m
#else // USE_4M
#define mul_ctdmatrix_block mul_ctdmatrix_block_3m
#endif // USE_4M

// Strassen
void mul_ctdmatrix_strassen_3m(CTDMatrix ret, CTDMatrix mat_a, CTDMatrix mat_b, long int min_dim);
void mul_ctdmatrix_strassen_4m(CTDMatrix ret, CTDMatrix mat_a, CTDMatrix mat_b, long int min_dim);
#ifdef USE_4M
#define mul_ctdmatrix_strassen mul_ctdmatrix_strassen_4m
#else // USE_4M
#define mul_ctdmatrix_strassen mul_ctdmatrix_strassen_3m
#endif // USE_4M

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
void _bncomp_add_ctdmatrix_partial(CTDMatrix ret, long int ret_index[4], CTDMatrix mat_a, long int mat_a_index[4], CTDMatrix mat_b, long int mat_b_index[4]);

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_sub_tdmatrix_partial(TDMatrix ret, long int ret_index[4], TDMatrix mat_a, long int mat_a_index[4], TDMatrix mat_b, long int mat_b_index[4]);
void _bncomp_sub_ctdmatrix_partial(CTDMatrix ret, long int ret_index[4], CTDMatrix mat_a, long int mat_a_index[4], CTDMatrix mat_b, long int mat_b_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_subst_tdmatrix_partial(TDMatrix ret, long int ret_index[4], TDMatrix mat_a, long int mat_a_index[4]);
void _bncomp_subst_ctdmatrix_partial(CTDMatrix ret, long int ret_index[4], CTDMatrix mat_a, long int mat_a_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_subst_tdmatrix_partial_checked(TDMatrix ret, long int ret_index[4], TDMatrix mat_a, long int mat_a_index[4]);
void _bncomp_subst_ctdmatrix_partial_checked(CTDMatrix ret, long int ret_index[4], CTDMatrix mat_a, long int mat_a_index[4]);

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

// Complex with OpenMP

/* c = a * b */
#ifdef USE_4M
#define _bncomp_mul_ctdmatrix(c, a, b) _bncomp_mul_ctdmatrix_4m(c, a, b)
#else // USE_4M
#define _bncomp_mul_ctdmatrix(c, a, b) _bncomp_mul_ctdmatrix_3m(c, a, b)
#endif // USE_4M
#define _bncomp_mul_ctdmatrix_simple(c, a, b) _bncomp_mul_ctdmatrix(c, a, b)

// Block matrix multiplicaiton
void _bncomp_mul_ctdmatrix_block_3m(CTDMatrix ret, CTDMatrix mat_a, CTDMatrix mat_b, long int min_dim);
void _bncomp_mul_ctdmatrix_block_4m(CTDMatrix ret, CTDMatrix mat_a, CTDMatrix mat_b, long int min_dim);
#ifdef USE_4M
#define _bncomp_mul_ctdmatrix_block _bncomp_mul_ctdmatrix_block_4m
#else // USE_4M
#define _bncomp_mul_ctdmatrix_block _bncomp_mul_ctdmatrix_block_3m
#endif // USE_4M

// Strassen
void _bncomp_mul_ctdmatrix_strassen_3m(CTDMatrix ret, CTDMatrix mat_a, CTDMatrix mat_b, long int min_dim);
void _bncomp_mul_ctdmatrix_strassen_4m(CTDMatrix ret, CTDMatrix mat_a, CTDMatrix mat_b, long int min_dim);
#ifdef USE_4M
#define _bncomp_mul_ctdmatrix_strassen _bncomp_mul_ctdmatrix_strassen_4m
#else // USE_4M
#define _bncomp_mul_ctdmatrix_strassen _bncomp_mul_ctdmatrix_strassen_3m
#endif // USE_4M

//--------------------------------------
// CUDA (GTD: triple-double on GPU, gdtq-0.0.2)
//--------------------------------------
#ifdef USE_CUDA

#define mul_gtdmatrix_simple(c, a, b, num_blocks, num_threads) mul_gtdmatrix_dev((c), (a), (b), (num_blocks), (num_threads))
#define _bncuda_mul_gtdmatrix_simple(c, a, b, num_blocks, num_threads) mul_gtdmatrix_dev((c), (a), (b), (num_blocks), (num_threads))
#define _bncuda_set0_gtdmatrix(a, num_blocks, num_threads) set0_gtdmatrix_dev((a), (num_blocks), (num_threads))

// Suppose that all arguments are allocated on GPU
void _bncuda_add_gtdmatrix_partial(GTDMatrix ret_dev, long int ret_index_dev[4], GTDMatrix mat_a_dev, long int mat_a_index_dev[4], GTDMatrix mat_b_dev, long int mat_b_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);

// partial sub
void _bncuda_sub_gtdmatrix_partial(GTDMatrix ret_dev, long int ret_index_dev[4], GTDMatrix mat_a_dev, long int mat_a_index_dev[4], GTDMatrix mat_b_dev, long int mat_b_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);

// partial set
void _bncuda_subst_gtdmatrix_partial(GTDMatrix ret_dev, long int ret_index_dev[4], GTDMatrix mat_a_dev, long int mat_a_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);

// partial set with bounds check
void _bncuda_subst_gtdmatrix_partial_checked(GTDMatrix ret_dev, long int ret_index_dev[4], GTDMatrix mat_a_dev, long int mat_a_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);

// reverse sign
void _bncuda_neg_gtdmatrix_partial(GTDMatrix ret_dev, long int ret_index_dev[4], GTDMatrix mat_a_dev, long int mat_a_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);

// Block matrix multiplicaiton
void _bncuda_mul_gtdmatrix_block(GTDMatrix ret_dev, GTDMatrix mat_a_dev, GTDMatrix mat_b_dev, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);

// Padding to even dimensional matrix
GTDMatrix _bncuda_init_dynamic_padding_gtdmatrix_strassen2(GTDMatrix orig_mat_dev, long int min_dim);

// Strassen's Algorithm with static padding
void _bncuda_mul_gtdmatrix_strassen_odd_padding(GTDMatrix ret_dev, GTDMatrix mat_a_dev, GTDMatrix mat_b_dev, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);

// Strassen's Algorithm with Dynamic peeling
void _bncuda_mul_gtdmatrix_strassen_odd_peeling(GTDMatrix ret, GTDMatrix mat_a, GTDMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);

// counters
void _bncuda_reset_num_mul_gtdmatrix_strassen(void);
void _bncuda_get_num_mul_gtdmatrix_strassen(long int *num_addsub, long int *num_mul);
void _bncuda_print_num_mul_gtdmatrix_strassen(long int *num_addsub, long int *num_mul);

// Fit dimension to be multiple of min_dim
void _bncuda_mul_gtdmatrix_strassen(GTDMatrix ret, GTDMatrix mat_a, GTDMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);

// Strassen's Algorithm with parallized sections
void _bncuda_mul_gtdmatrix_strassen_even_psec(GTDMatrix ret, GTDMatrix mat_a, GTDMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);

// Winograd Variant of Strassen's Algorithm with parallelied sections
void _bncuda_mul_gtdmatrix_winograd_even_psec(GTDMatrix ret, GTDMatrix mat_a, GTDMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);

#endif // USE_CUDA

#endif // USE_TDLINEAR

//==================================================
// float-based DS / TS / QS Strassen prototypes
//==================================================

///////////////
///// DS //////
///////////////
#ifdef USE_DSLINEAR

// count the number of computations
#ifndef __NUM_ADDSUB_MUL_DSMATRIX_STRASSEN
#define __NUM_ADDSUB_MUL_DSMATRIX_STRASSEN
extern long int num_addsub_mul_dsmatrix_strassen;	// addition and subtraction
extern long int num_mul_mul_dsmatrix_strassen;		// multiplication
#endif // __NUM_ADDSUB_MUL_DSMATRIX_STRASSEN

// clear / get / print counters
void reset_num_mul_dsmatrix_strassen(void);
void get_num_mul_dsmatrix_strassen(long int *num_addsub, long int *num_mul);
void print_num_mul_dsmatrix_strassen(long int *num_addsub, long int *num_mul);

// partial add / sub / set / set(checked) / negate
void add_dsmatrix_partial(DSMatrix ret, long int ret_index[4], DSMatrix mat_a, long int mat_a_index[4], DSMatrix mat_b, long int mat_b_index[4]);
void sub_dsmatrix_partial(DSMatrix ret, long int ret_index[4], DSMatrix mat_a, long int mat_a_index[4], DSMatrix mat_b, long int mat_b_index[4]);
void subst_dsmatrix_partial(DSMatrix ret, long int ret_index[4], DSMatrix mat_a, long int mat_a_index[4]);
void subst_dsmatrix_partial_checked(DSMatrix ret, long int ret_index[4], DSMatrix mat_a, long int mat_a_index[4]);
void neg_dsmatrix_partial(DSMatrix ret, long int ret_index[4], DSMatrix mat_a, long int mat_a_index[4]);

/* c = a * b */
#define mul_dsmatrix_simple(c, a, b) mul_dsmatrix(c, a, b)

// Block matrix multiplication
void mul_dsmatrix_block(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim);

// Padding helpers
DSMatrix init_static_padding_dsmatrix_strassen(DSMatrix orig_mat);
DSMatrix init_dynamic_padding_dsmatrix_strassen(DSMatrix orig_mat);
DSMatrix init_dynamic_padding_dsmatrix_strassen2(DSMatrix orig_mat, long int min_dim);

// Strassen / Winograd
void mul_dsmatrix_strassen_odd_padding(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim);
void mul_dsmatrix_strassen(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim);
void mul_dsmatrix_strassen_odd_peeling(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim);
void mul_dsmatrix_strassen_even(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim);
void mul_dsmatrix_winograd_even(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim);

// Inverse via Strassen
void inv_dsmatrix_strassen_even(DSMatrix ret, DSMatrix mat_a, long int min_dim);

// scaling with the absolute maximum element
void left_scaling_dsmatrix(DSMatrix ret, DSMatrix org_mat, DSVector scaling_diag_mat, long int *ret_col_index);
void right_scaling_dsmatrix(DSMatrix ret, DSMatrix org_mat, DSVector scaling_diag_mat, long int *ret_row_index);

// multiply square matrix by diagonal matrix
void mul_dsmatrix_dsdiag(DSMatrix ret, DSVector left_diag_mat, int left_inv_flag, DSMatrix org_mat, DSVector right_diag_mat, int right_inv_flag);

//--------------------------------------
// OpenMP
//--------------------------------------
#ifndef ___BNCOMP_NUM_ADDSUB_MUL_DSMATRIX_STRASSEN
#define ___BNCOMP_NUM_ADDSUB_MUL_DSMATRIX_STRASSEN
extern long int _bncomp_num_addsub_mul_dsmatrix_strassen;	// addition and subtraction
extern long int _bncomp_num_mul_mul_dsmatrix_strassen;		// multiplication
#endif // ___BNCOMP_NUM_ADDSUB_MUL_DSMATRIX_STRASSEN

void _bncomp_reset_num_mul_dsmatrix_strassen(void);
void _bncomp_get_num_mul_dsmatrix_strassen(long int *num_addsub, long int *num_mul);
void _bncomp_print_num_mul_dsmatrix_strassen(long int *num_addsub, long int *num_mul);

void _bncomp_add_dsmatrix_partial(DSMatrix ret, long int ret_index[4], DSMatrix mat_a, long int mat_a_index[4], DSMatrix mat_b, long int mat_b_index[4]);
void _bncomp_sub_dsmatrix_partial(DSMatrix ret, long int ret_index[4], DSMatrix mat_a, long int mat_a_index[4], DSMatrix mat_b, long int mat_b_index[4]);
void _bncomp_subst_dsmatrix_partial(DSMatrix ret, long int ret_index[4], DSMatrix mat_a, long int mat_a_index[4]);
void _bncomp_subst_dsmatrix_partial_checked(DSMatrix ret, long int ret_index[4], DSMatrix mat_a, long int mat_a_index[4]);
void _bncomp_neg_dsmatrix_partial(DSMatrix ret, long int ret_index[4], DSMatrix mat_a, long int mat_a_index[4]);

/* c = a * b */
#define _bncomp_mul_dsmatrix_simple(c, a, b) _bncomp_mul_dsmatrix(c, a, b)

void _bncomp_mul_dsmatrix_block(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim);

DSMatrix _bncomp_init_static_padding_dsmatrix_strassen(DSMatrix orig_mat);
DSMatrix _bncomp_init_dynamic_padding_dsmatrix_strassen(DSMatrix orig_mat);
DSMatrix _bncomp_init_dynamic_padding_dsmatrix_strassen2(DSMatrix orig_mat, long int min_dim);

void _bncomp_mul_dsmatrix_strassen_odd_padding(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim);
void _bncomp_mul_dsmatrix_strassen(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim);
void _bncomp_mul_dsmatrix_strassen_odd_peeling(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim);
void _bncomp_mul_dsmatrix_strassen_nonrec(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim);
void _bncomp_mul_dsmatrix_strassen_even(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim);
void _bncomp_mul_dsmatrix_strassen_even_psec(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim);
void _bncomp_mul_dsmatrix_strassen_even2(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim, long int rec_num);
void _bncomp_mul_dsmatrix_strassen_even3(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim, long int rec_num);
void _bncomp_mul_dsmatrix_winograd_even(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim);
void _bncomp_mul_dsmatrix_winograd_even_psec(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim);
void _bncomp_mul_dsmatrix_winograd_even2(DSMatrix ret, DSMatrix mat_a, DSMatrix mat_b, long int min_dim);

// NOTE: the following _bncomp_* primitives are referenced by matmul_strassen_general_ds_omp.c
// but live in the (not-yet-existing) bncomp_linear_ds.c float OpenMP layer.
void _bncomp_add_dsvector(DSVector c, DSVector a, DSVector b);
void _bncomp_cmul_dsvector(DSVector c, float val[DSSIZE], DSVector a);
void _bncomp_ip_dsvector(float ret[DSSIZE], DSVector a, DSVector b);
void _bncomp_add_dsmatrix(DSMatrix c, DSMatrix a, DSMatrix b);
void _bncomp_sub_dsmatrix(DSMatrix c, DSMatrix a, DSMatrix b);
void _bncomp_mul_dsmatrix(DSMatrix ret, DSMatrix a, DSMatrix b);
void _bncomp_mul_dsmatrix_dsvec(DSVector v, DSMatrix a, DSVector vb);
void _bncomp_mul_dsmatrixt_dsvec(DSVector v, DSMatrix a, DSVector vb);

#endif // USE_DSLINEAR


///////////////
///// TS //////
///////////////
#ifdef USE_TSLINEAR

// count the number of computations
#ifndef __NUM_ADDSUB_MUL_TSMATRIX_STRASSEN
#define __NUM_ADDSUB_MUL_TSMATRIX_STRASSEN
extern long int num_addsub_mul_tsmatrix_strassen;	// addition and subtraction
extern long int num_mul_mul_tsmatrix_strassen;		// multiplication
#endif // __NUM_ADDSUB_MUL_TSMATRIX_STRASSEN

// clear / get / print counters
void reset_num_mul_tsmatrix_strassen(void);
void get_num_mul_tsmatrix_strassen(long int *num_addsub, long int *num_mul);
void print_num_mul_tsmatrix_strassen(long int *num_addsub, long int *num_mul);

// partial add / sub / set / set(checked) / negate
void add_tsmatrix_partial(TSMatrix ret, long int ret_index[4], TSMatrix mat_a, long int mat_a_index[4], TSMatrix mat_b, long int mat_b_index[4]);
void sub_tsmatrix_partial(TSMatrix ret, long int ret_index[4], TSMatrix mat_a, long int mat_a_index[4], TSMatrix mat_b, long int mat_b_index[4]);
void subst_tsmatrix_partial(TSMatrix ret, long int ret_index[4], TSMatrix mat_a, long int mat_a_index[4]);
void subst_tsmatrix_partial_checked(TSMatrix ret, long int ret_index[4], TSMatrix mat_a, long int mat_a_index[4]);
void neg_tsmatrix_partial(TSMatrix ret, long int ret_index[4], TSMatrix mat_a, long int mat_a_index[4]);

/* c = a * b */
#define mul_tsmatrix_simple(c, a, b) mul_tsmatrix(c, a, b)

// Block matrix multiplication
void mul_tsmatrix_block(TSMatrix ret, TSMatrix mat_a, TSMatrix mat_b, long int min_dim);

// Padding helpers
TSMatrix init_static_padding_tsmatrix_strassen(TSMatrix orig_mat);
TSMatrix init_dynamic_padding_tsmatrix_strassen(TSMatrix orig_mat);
TSMatrix init_dynamic_padding_tsmatrix_strassen2(TSMatrix orig_mat, long int min_dim);

// Strassen / Winograd
void mul_tsmatrix_strassen_odd_padding(TSMatrix ret, TSMatrix mat_a, TSMatrix mat_b, long int min_dim);
void mul_tsmatrix_strassen(TSMatrix ret, TSMatrix mat_a, TSMatrix mat_b, long int min_dim);
void mul_tsmatrix_strassen_odd_peeling(TSMatrix ret, TSMatrix mat_a, TSMatrix mat_b, long int min_dim);
void mul_tsmatrix_strassen_even(TSMatrix ret, TSMatrix mat_a, TSMatrix mat_b, long int min_dim);
void mul_tsmatrix_winograd_even(TSMatrix ret, TSMatrix mat_a, TSMatrix mat_b, long int min_dim);

// Inverse via Strassen
void inv_tsmatrix_strassen_even(TSMatrix ret, TSMatrix mat_a, long int min_dim);

// scaling with the absolute maximum element
void left_scaling_tsmatrix(TSMatrix ret, TSMatrix org_mat, TSVector scaling_diag_mat, long int *ret_col_index);
void right_scaling_tsmatrix(TSMatrix ret, TSMatrix org_mat, TSVector scaling_diag_mat, long int *ret_row_index);

// multiply square matrix by diagonal matrix
void mul_tsmatrix_tsdiag(TSMatrix ret, TSVector left_diag_mat, int left_inv_flag, TSMatrix org_mat, TSVector right_diag_mat, int right_inv_flag);

//--------------------------------------
// OpenMP
//--------------------------------------
#ifndef ___BNCOMP_NUM_ADDSUB_MUL_TSMATRIX_STRASSEN
#define ___BNCOMP_NUM_ADDSUB_MUL_TSMATRIX_STRASSEN
extern long int _bncomp_num_addsub_mul_tsmatrix_strassen;	// addition and subtraction
extern long int _bncomp_num_mul_mul_tsmatrix_strassen;		// multiplication
#endif // ___BNCOMP_NUM_ADDSUB_MUL_TSMATRIX_STRASSEN

void _bncomp_reset_num_mul_tsmatrix_strassen(void);
void _bncomp_get_num_mul_tsmatrix_strassen(long int *num_addsub, long int *num_mul);
void _bncomp_print_num_mul_tsmatrix_strassen(long int *num_addsub, long int *num_mul);

void _bncomp_add_tsmatrix_partial(TSMatrix ret, long int ret_index[4], TSMatrix mat_a, long int mat_a_index[4], TSMatrix mat_b, long int mat_b_index[4]);
void _bncomp_sub_tsmatrix_partial(TSMatrix ret, long int ret_index[4], TSMatrix mat_a, long int mat_a_index[4], TSMatrix mat_b, long int mat_b_index[4]);
void _bncomp_subst_tsmatrix_partial(TSMatrix ret, long int ret_index[4], TSMatrix mat_a, long int mat_a_index[4]);
void _bncomp_subst_tsmatrix_partial_checked(TSMatrix ret, long int ret_index[4], TSMatrix mat_a, long int mat_a_index[4]);
void _bncomp_neg_tsmatrix_partial(TSMatrix ret, long int ret_index[4], TSMatrix mat_a, long int mat_a_index[4]);

/* c = a * b */
#define _bncomp_mul_tsmatrix_simple(c, a, b) _bncomp_mul_tsmatrix(c, a, b)

void _bncomp_mul_tsmatrix_block(TSMatrix ret, TSMatrix mat_a, TSMatrix mat_b, long int min_dim);

TSMatrix _bncomp_init_static_padding_tsmatrix_strassen(TSMatrix orig_mat);
TSMatrix _bncomp_init_dynamic_padding_tsmatrix_strassen(TSMatrix orig_mat);
TSMatrix _bncomp_init_dynamic_padding_tsmatrix_strassen2(TSMatrix orig_mat, long int min_dim);

void _bncomp_mul_tsmatrix_strassen_odd_padding(TSMatrix ret, TSMatrix mat_a, TSMatrix mat_b, long int min_dim);
void _bncomp_mul_tsmatrix_strassen(TSMatrix ret, TSMatrix mat_a, TSMatrix mat_b, long int min_dim);
void _bncomp_mul_tsmatrix_strassen_odd_peeling(TSMatrix ret, TSMatrix mat_a, TSMatrix mat_b, long int min_dim);
void _bncomp_mul_tsmatrix_strassen_nonrec(TSMatrix ret, TSMatrix mat_a, TSMatrix mat_b, long int min_dim);
void _bncomp_mul_tsmatrix_strassen_even(TSMatrix ret, TSMatrix mat_a, TSMatrix mat_b, long int min_dim);
void _bncomp_mul_tsmatrix_strassen_even_psec(TSMatrix ret, TSMatrix mat_a, TSMatrix mat_b, long int min_dim);
void _bncomp_mul_tsmatrix_strassen_even2(TSMatrix ret, TSMatrix mat_a, TSMatrix mat_b, long int min_dim, long int rec_num);
void _bncomp_mul_tsmatrix_strassen_even3(TSMatrix ret, TSMatrix mat_a, TSMatrix mat_b, long int min_dim, long int rec_num);
void _bncomp_mul_tsmatrix_winograd_even(TSMatrix ret, TSMatrix mat_a, TSMatrix mat_b, long int min_dim);
void _bncomp_mul_tsmatrix_winograd_even_psec(TSMatrix ret, TSMatrix mat_a, TSMatrix mat_b, long int min_dim);
void _bncomp_mul_tsmatrix_winograd_even2(TSMatrix ret, TSMatrix mat_a, TSMatrix mat_b, long int min_dim);

// NOTE: the following _bncomp_* primitives are referenced by matmul_strassen_general_ts_omp.c
// but live in the (not-yet-existing) bncomp_linear_ts.c float OpenMP layer.
void _bncomp_add_tsvector(TSVector c, TSVector a, TSVector b);
void _bncomp_cmul_tsvector(TSVector c, float val[TSSIZE], TSVector a);
void _bncomp_ip_tsvector(float ret[TSSIZE], TSVector a, TSVector b);
void _bncomp_add_tsmatrix(TSMatrix c, TSMatrix a, TSMatrix b);
void _bncomp_sub_tsmatrix(TSMatrix c, TSMatrix a, TSMatrix b);
void _bncomp_mul_tsmatrix(TSMatrix ret, TSMatrix a, TSMatrix b);
void _bncomp_mul_tsmatrix_tsvec(TSVector v, TSMatrix a, TSVector vb);
void _bncomp_mul_tsmatrixt_tsvec(TSVector v, TSMatrix a, TSVector vb);

#endif // USE_TSLINEAR


///////////////
///// QS //////
///////////////
#ifdef USE_QSLINEAR

// count the number of computations
#ifndef __NUM_ADDSUB_MUL_QSMATRIX_STRASSEN
#define __NUM_ADDSUB_MUL_QSMATRIX_STRASSEN
extern long int num_addsub_mul_qsmatrix_strassen;	// addition and subtraction
extern long int num_mul_mul_qsmatrix_strassen;		// multiplication
#endif // __NUM_ADDSUB_MUL_QSMATRIX_STRASSEN

// clear / get / print counters
void reset_num_mul_qsmatrix_strassen(void);
void get_num_mul_qsmatrix_strassen(long int *num_addsub, long int *num_mul);
void print_num_mul_qsmatrix_strassen(long int *num_addsub, long int *num_mul);

// partial add / sub / set / set(checked) / negate
void add_qsmatrix_partial(QSMatrix ret, long int ret_index[4], QSMatrix mat_a, long int mat_a_index[4], QSMatrix mat_b, long int mat_b_index[4]);
void sub_qsmatrix_partial(QSMatrix ret, long int ret_index[4], QSMatrix mat_a, long int mat_a_index[4], QSMatrix mat_b, long int mat_b_index[4]);
void subst_qsmatrix_partial(QSMatrix ret, long int ret_index[4], QSMatrix mat_a, long int mat_a_index[4]);
void subst_qsmatrix_partial_checked(QSMatrix ret, long int ret_index[4], QSMatrix mat_a, long int mat_a_index[4]);
void neg_qsmatrix_partial(QSMatrix ret, long int ret_index[4], QSMatrix mat_a, long int mat_a_index[4]);

/* c = a * b */
#define mul_qsmatrix_simple(c, a, b) mul_qsmatrix(c, a, b)

// Block matrix multiplication
void mul_qsmatrix_block(QSMatrix ret, QSMatrix mat_a, QSMatrix mat_b, long int min_dim);

// Padding helpers
QSMatrix init_static_padding_qsmatrix_strassen(QSMatrix orig_mat);
QSMatrix init_dynamic_padding_qsmatrix_strassen(QSMatrix orig_mat);
QSMatrix init_dynamic_padding_qsmatrix_strassen2(QSMatrix orig_mat, long int min_dim);

// Strassen / Winograd
void mul_qsmatrix_strassen_odd_padding(QSMatrix ret, QSMatrix mat_a, QSMatrix mat_b, long int min_dim);
void mul_qsmatrix_strassen(QSMatrix ret, QSMatrix mat_a, QSMatrix mat_b, long int min_dim);
void mul_qsmatrix_strassen_odd_peeling(QSMatrix ret, QSMatrix mat_a, QSMatrix mat_b, long int min_dim);
void mul_qsmatrix_strassen_even(QSMatrix ret, QSMatrix mat_a, QSMatrix mat_b, long int min_dim);
void mul_qsmatrix_winograd_even(QSMatrix ret, QSMatrix mat_a, QSMatrix mat_b, long int min_dim);

// Inverse via Strassen
void inv_qsmatrix_strassen_even(QSMatrix ret, QSMatrix mat_a, long int min_dim);

// scaling with the absolute maximum element
void left_scaling_qsmatrix(QSMatrix ret, QSMatrix org_mat, QSVector scaling_diag_mat, long int *ret_col_index);
void right_scaling_qsmatrix(QSMatrix ret, QSMatrix org_mat, QSVector scaling_diag_mat, long int *ret_row_index);

// multiply square matrix by diagonal matrix
void mul_qsmatrix_qsdiag(QSMatrix ret, QSVector left_diag_mat, int left_inv_flag, QSMatrix org_mat, QSVector right_diag_mat, int right_inv_flag);

//--------------------------------------
// OpenMP
//--------------------------------------
#ifndef ___BNCOMP_NUM_ADDSUB_MUL_QSMATRIX_STRASSEN
#define ___BNCOMP_NUM_ADDSUB_MUL_QSMATRIX_STRASSEN
extern long int _bncomp_num_addsub_mul_qsmatrix_strassen;	// addition and subtraction
extern long int _bncomp_num_mul_mul_qsmatrix_strassen;		// multiplication
#endif // ___BNCOMP_NUM_ADDSUB_MUL_QSMATRIX_STRASSEN

void _bncomp_reset_num_mul_qsmatrix_strassen(void);
void _bncomp_get_num_mul_qsmatrix_strassen(long int *num_addsub, long int *num_mul);
void _bncomp_print_num_mul_qsmatrix_strassen(long int *num_addsub, long int *num_mul);

void _bncomp_add_qsmatrix_partial(QSMatrix ret, long int ret_index[4], QSMatrix mat_a, long int mat_a_index[4], QSMatrix mat_b, long int mat_b_index[4]);
void _bncomp_sub_qsmatrix_partial(QSMatrix ret, long int ret_index[4], QSMatrix mat_a, long int mat_a_index[4], QSMatrix mat_b, long int mat_b_index[4]);
void _bncomp_subst_qsmatrix_partial(QSMatrix ret, long int ret_index[4], QSMatrix mat_a, long int mat_a_index[4]);
void _bncomp_subst_qsmatrix_partial_checked(QSMatrix ret, long int ret_index[4], QSMatrix mat_a, long int mat_a_index[4]);
void _bncomp_neg_qsmatrix_partial(QSMatrix ret, long int ret_index[4], QSMatrix mat_a, long int mat_a_index[4]);

/* c = a * b */
#define _bncomp_mul_qsmatrix_simple(c, a, b) _bncomp_mul_qsmatrix(c, a, b)

void _bncomp_mul_qsmatrix_block(QSMatrix ret, QSMatrix mat_a, QSMatrix mat_b, long int min_dim);

QSMatrix _bncomp_init_static_padding_qsmatrix_strassen(QSMatrix orig_mat);
QSMatrix _bncomp_init_dynamic_padding_qsmatrix_strassen(QSMatrix orig_mat);
QSMatrix _bncomp_init_dynamic_padding_qsmatrix_strassen2(QSMatrix orig_mat, long int min_dim);

void _bncomp_mul_qsmatrix_strassen_odd_padding(QSMatrix ret, QSMatrix mat_a, QSMatrix mat_b, long int min_dim);
void _bncomp_mul_qsmatrix_strassen(QSMatrix ret, QSMatrix mat_a, QSMatrix mat_b, long int min_dim);
void _bncomp_mul_qsmatrix_strassen_odd_peeling(QSMatrix ret, QSMatrix mat_a, QSMatrix mat_b, long int min_dim);
void _bncomp_mul_qsmatrix_strassen_nonrec(QSMatrix ret, QSMatrix mat_a, QSMatrix mat_b, long int min_dim);
void _bncomp_mul_qsmatrix_strassen_even(QSMatrix ret, QSMatrix mat_a, QSMatrix mat_b, long int min_dim);
void _bncomp_mul_qsmatrix_strassen_even_psec(QSMatrix ret, QSMatrix mat_a, QSMatrix mat_b, long int min_dim);
void _bncomp_mul_qsmatrix_strassen_even2(QSMatrix ret, QSMatrix mat_a, QSMatrix mat_b, long int min_dim, long int rec_num);
void _bncomp_mul_qsmatrix_strassen_even3(QSMatrix ret, QSMatrix mat_a, QSMatrix mat_b, long int min_dim, long int rec_num);
void _bncomp_mul_qsmatrix_winograd_even(QSMatrix ret, QSMatrix mat_a, QSMatrix mat_b, long int min_dim);
void _bncomp_mul_qsmatrix_winograd_even_psec(QSMatrix ret, QSMatrix mat_a, QSMatrix mat_b, long int min_dim);
void _bncomp_mul_qsmatrix_winograd_even2(QSMatrix ret, QSMatrix mat_a, QSMatrix mat_b, long int min_dim);

// NOTE: the following _bncomp_* primitives are referenced by matmul_strassen_general_qs_omp.c
// but live in the (not-yet-existing) bncomp_linear_qs.c float OpenMP layer.
void _bncomp_add_qsvector(QSVector c, QSVector a, QSVector b);
void _bncomp_cmul_qsvector(QSVector c, float val[QSSIZE], QSVector a);
void _bncomp_ip_qsvector(float ret[QSSIZE], QSVector a, QSVector b);
void _bncomp_add_qsmatrix(QSMatrix c, QSMatrix a, QSMatrix b);
void _bncomp_sub_qsmatrix(QSMatrix c, QSMatrix a, QSMatrix b);
void _bncomp_mul_qsmatrix(QSMatrix ret, QSMatrix a, QSMatrix b);
void _bncomp_mul_qsmatrix_qsvec(QSVector v, QSMatrix a, QSVector vb);
void _bncomp_mul_qsmatrixt_qsvec(QSVector v, QSMatrix a, QSVector vb);

#endif // USE_QSLINEAR


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
// CUDA (GDS: double-single on GPU, gdtq-0.0.2)
//--------------------------------------
#if defined(USE_CUDA) && defined(USE_DSLINEAR)

#define mul_gdsmatrix_simple(c, a, b, num_blocks, num_threads) mul_gdsmatrix_dev((c), (a), (b), (num_blocks), (num_threads))
#define _bncuda_mul_gdsmatrix_simple(c, a, b, num_blocks, num_threads) mul_gdsmatrix_dev((c), (a), (b), (num_blocks), (num_threads))
#define _bncuda_set0_gdsmatrix(a, num_blocks, num_threads) set0_gdsmatrix_dev((a), (num_blocks), (num_threads))

void _bncuda_add_gdsmatrix_partial(GDSMatrix ret_dev, long int ret_index_dev[4], GDSMatrix mat_a_dev, long int mat_a_index_dev[4], GDSMatrix mat_b_dev, long int mat_b_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_sub_gdsmatrix_partial(GDSMatrix ret_dev, long int ret_index_dev[4], GDSMatrix mat_a_dev, long int mat_a_index_dev[4], GDSMatrix mat_b_dev, long int mat_b_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_subst_gdsmatrix_partial(GDSMatrix ret_dev, long int ret_index_dev[4], GDSMatrix mat_a_dev, long int mat_a_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_subst_gdsmatrix_partial_checked(GDSMatrix ret_dev, long int ret_index_dev[4], GDSMatrix mat_a_dev, long int mat_a_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_neg_gdsmatrix_partial(GDSMatrix ret_dev, long int ret_index_dev[4], GDSMatrix mat_a_dev, long int mat_a_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_mul_gdsmatrix_block(GDSMatrix ret_dev, GDSMatrix mat_a_dev, GDSMatrix mat_b_dev, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);
GDSMatrix _bncuda_init_dynamic_padding_gdsmatrix_strassen2(GDSMatrix orig_mat_dev, long int min_dim);
void _bncuda_mul_gdsmatrix_strassen_odd_padding(GDSMatrix ret_dev, GDSMatrix mat_a_dev, GDSMatrix mat_b_dev, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_mul_gdsmatrix_strassen_odd_peeling(GDSMatrix ret, GDSMatrix mat_a, GDSMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_reset_num_mul_gdsmatrix_strassen(void);
void _bncuda_get_num_mul_gdsmatrix_strassen(long int *num_addsub, long int *num_mul);
void _bncuda_print_num_mul_gdsmatrix_strassen(long int *num_addsub, long int *num_mul);
void _bncuda_mul_gdsmatrix_strassen(GDSMatrix ret, GDSMatrix mat_a, GDSMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_mul_gdsmatrix_strassen_even_psec(GDSMatrix ret, GDSMatrix mat_a, GDSMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_mul_gdsmatrix_winograd_even_psec(GDSMatrix ret, GDSMatrix mat_a, GDSMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);

#endif // USE_CUDA && USE_DSLINEAR

//--------------------------------------
// CUDA (GTS: triple-single on GPU, gdtq-0.0.2)
//--------------------------------------
#if defined(USE_CUDA) && defined(USE_TSLINEAR)

#define mul_gtsmatrix_simple(c, a, b, num_blocks, num_threads) mul_gtsmatrix_dev((c), (a), (b), (num_blocks), (num_threads))
#define _bncuda_mul_gtsmatrix_simple(c, a, b, num_blocks, num_threads) mul_gtsmatrix_dev((c), (a), (b), (num_blocks), (num_threads))
#define _bncuda_set0_gtsmatrix(a, num_blocks, num_threads) set0_gtsmatrix_dev((a), (num_blocks), (num_threads))

void _bncuda_add_gtsmatrix_partial(GTSMatrix ret_dev, long int ret_index_dev[4], GTSMatrix mat_a_dev, long int mat_a_index_dev[4], GTSMatrix mat_b_dev, long int mat_b_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_sub_gtsmatrix_partial(GTSMatrix ret_dev, long int ret_index_dev[4], GTSMatrix mat_a_dev, long int mat_a_index_dev[4], GTSMatrix mat_b_dev, long int mat_b_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_subst_gtsmatrix_partial(GTSMatrix ret_dev, long int ret_index_dev[4], GTSMatrix mat_a_dev, long int mat_a_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_subst_gtsmatrix_partial_checked(GTSMatrix ret_dev, long int ret_index_dev[4], GTSMatrix mat_a_dev, long int mat_a_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_neg_gtsmatrix_partial(GTSMatrix ret_dev, long int ret_index_dev[4], GTSMatrix mat_a_dev, long int mat_a_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_mul_gtsmatrix_block(GTSMatrix ret_dev, GTSMatrix mat_a_dev, GTSMatrix mat_b_dev, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);
GTSMatrix _bncuda_init_dynamic_padding_gtsmatrix_strassen2(GTSMatrix orig_mat_dev, long int min_dim);
void _bncuda_mul_gtsmatrix_strassen_odd_padding(GTSMatrix ret_dev, GTSMatrix mat_a_dev, GTSMatrix mat_b_dev, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_mul_gtsmatrix_strassen_odd_peeling(GTSMatrix ret, GTSMatrix mat_a, GTSMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_reset_num_mul_gtsmatrix_strassen(void);
void _bncuda_get_num_mul_gtsmatrix_strassen(long int *num_addsub, long int *num_mul);
void _bncuda_print_num_mul_gtsmatrix_strassen(long int *num_addsub, long int *num_mul);
void _bncuda_mul_gtsmatrix_strassen(GTSMatrix ret, GTSMatrix mat_a, GTSMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_mul_gtsmatrix_strassen_even_psec(GTSMatrix ret, GTSMatrix mat_a, GTSMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_mul_gtsmatrix_winograd_even_psec(GTSMatrix ret, GTSMatrix mat_a, GTSMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);

#endif // USE_CUDA && USE_TSLINEAR

//--------------------------------------
// CUDA (GQS: quad-single on GPU, gdtq-0.0.2)
//--------------------------------------
#if defined(USE_CUDA) && defined(USE_QSLINEAR)

#define mul_gqsmatrix_simple(c, a, b, num_blocks, num_threads) mul_gqsmatrix_dev((c), (a), (b), (num_blocks), (num_threads))
#define _bncuda_mul_gqsmatrix_simple(c, a, b, num_blocks, num_threads) mul_gqsmatrix_dev((c), (a), (b), (num_blocks), (num_threads))
#define _bncuda_set0_gqsmatrix(a, num_blocks, num_threads) set0_gqsmatrix_dev((a), (num_blocks), (num_threads))

void _bncuda_add_gqsmatrix_partial(GQSMatrix ret_dev, long int ret_index_dev[4], GQSMatrix mat_a_dev, long int mat_a_index_dev[4], GQSMatrix mat_b_dev, long int mat_b_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_sub_gqsmatrix_partial(GQSMatrix ret_dev, long int ret_index_dev[4], GQSMatrix mat_a_dev, long int mat_a_index_dev[4], GQSMatrix mat_b_dev, long int mat_b_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_subst_gqsmatrix_partial(GQSMatrix ret_dev, long int ret_index_dev[4], GQSMatrix mat_a_dev, long int mat_a_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_subst_gqsmatrix_partial_checked(GQSMatrix ret_dev, long int ret_index_dev[4], GQSMatrix mat_a_dev, long int mat_a_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_neg_gqsmatrix_partial(GQSMatrix ret_dev, long int ret_index_dev[4], GQSMatrix mat_a_dev, long int mat_a_index_dev[4], int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_mul_gqsmatrix_block(GQSMatrix ret_dev, GQSMatrix mat_a_dev, GQSMatrix mat_b_dev, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);
GQSMatrix _bncuda_init_dynamic_padding_gqsmatrix_strassen2(GQSMatrix orig_mat_dev, long int min_dim);
void _bncuda_mul_gqsmatrix_strassen_odd_padding(GQSMatrix ret_dev, GQSMatrix mat_a_dev, GQSMatrix mat_b_dev, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_mul_gqsmatrix_strassen_odd_peeling(GQSMatrix ret, GQSMatrix mat_a, GQSMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_reset_num_mul_gqsmatrix_strassen(void);
void _bncuda_get_num_mul_gqsmatrix_strassen(long int *num_addsub, long int *num_mul);
void _bncuda_print_num_mul_gqsmatrix_strassen(long int *num_addsub, long int *num_mul);
void _bncuda_mul_gqsmatrix_strassen(GQSMatrix ret, GQSMatrix mat_a, GQSMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_mul_gqsmatrix_strassen_even_psec(GQSMatrix ret, GQSMatrix mat_a, GQSMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);
void _bncuda_mul_gqsmatrix_winograd_even_psec(GQSMatrix ret, GQSMatrix mat_a, GQSMatrix mat_b, long int min_dim, int num_blocks_per_grid, int num_threads_per_block);

#endif // USE_CUDA && USE_QSLINEAR

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

// Complex with OpenMP
// matmul_strassen_general_mpc_omp.c

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_add_cmpfmatrix_partial(CMPFMatrix ret, long int ret_index[4], CMPFMatrix mat_a, long int mat_a_index[4], CMPFMatrix mat_b, long int mat_b_index[4]);

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_sub_cmpfmatrix_partial(CMPFMatrix ret, long int ret_index[4], CMPFMatrix mat_a, long int mat_a_index[4], CMPFMatrix mat_b, long int mat_b_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_subst_cmpfmatrix_partial(CMPFMatrix ret, long int ret_index[4], CMPFMatrix mat_a, long int mat_a_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_subst_cmpfmatrix_partial_checked(CMPFMatrix ret, long int ret_index[4], CMPFMatrix mat_a, long int mat_a_index[4]);

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
void _bncomp_neg_cmpfmatrix_partial(CMPFMatrix ret, long int ret_index[4], CMPFMatrix mat_a, long int mat_a_index[4]);

// Block matrix multiplicaiton
void _bncomp_mul_cmpfmatrix_block(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim);

// Block matrix multiplicaiton(poor, so obosolete)
void _bncomp_mul_cmpfmatrix_block_old(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim);

// Padding to 2-powered dimensional matrix
CMPFMatrix _bncomp_init_static_padding_cmpfmatrix_strassen(CMPFMatrix orig_mat);

// Padding to even dimensional matrix
CMPFMatrix _bncomp_init_dynamic_padding_cmpfmatrix_strassen(CMPFMatrix orig_mat);

// Padding to even dimensional matrix
CMPFMatrix _bncomp_init_dynamic_padding_cmpfmatrix_strassen2(CMPFMatrix orig_mat, long int min_dim);

// Strassen's Algorithm with static padding
void _bncomp_mul_cmpfmatrix_strassen_odd_padding(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim);

// clear counter
void _bncomp_reset_num_mul_cmpfmatrix_strassen(void);

// get counters
void _bncomp_get_num_mul_cmpfmatrix_strassen(long int *_bncomp_num_addsub, long int *_bncomp_num_mul);

// print counters
void _bncomp_print_num_mul_cmpfmatrix_strassen(long int *_bncomp_num_addsub, long int *_bncomp_num_mul);

// get counters
void _bncomp_get_num_mul_cmpfmatrix_strassen(long int *_bncomp_num_addsub, long int *_bncomp_num_mul);

// Fit dimension to be multiple of min_dim
void _bncomp_mul_cmpfmatrix_strassen(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim);

// Strassen's Algorithm with Dynamic peeling
void _bncomp_mul_cmpfmatrix_strassen_odd_peeling(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim);

// Strassen's Algorithm
void _bncomp_mul_cmpfmatrix_strassen_even(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim);

// Strassen's Algorithm with parallelized sections
void _bncomp_mul_cmpfmatrix_strassen_even_psec(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim);

// Winograd Variant of Strassen's Algorithm
void _bncomp_mul_cmpfmatrix_winograd_even(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim);

// Winograd Variant of Strassen's Algorithm with parallelized sections
void _bncomp_mul_cmpfmatrix_winograd_even_psec(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim);

// Matrix multiplicaiton with Strassen's algorithm (Nonrecursive version in the area of Strassen's algorithms)
void _bncomp_mul_cmpfmatrix_strassen_nonrec(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim);

// Strassen's Algorithm (parallelized sections)
void _bncomp_mul_cmpfmatrix_strassen_even2(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim, long int rec_num);

// Strassen's Algorithm (parallelizable tasks)
void _bncomp_mul_cmpfmatrix_strassen_even3(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim, long int rec_num);

// Winograd Variant of Strassen's Algorithm (parallelized sections)
void _bncomp_mul_cmpfmatrix_winograd_even2(CMPFMatrix ret, CMPFMatrix mat_a, CMPFMatrix mat_b, long int min_dim);

#endif // USE_GMP

// -------------------
// Benchmark tools
// 2024-08-02 (Fri) T.Kouya
// -------------------
#include "benchmark_tools.h"

// -------------------
// Ozaki Scheme
// 2022-11-18(Fri) T.Kouya
// -------------------
#include "oz_scheme.h"

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // define _BNC_MATMUL_STRASSEN_H
