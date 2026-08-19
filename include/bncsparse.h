/********************************************************************************/
/*                                                                              */
/* bncsparse.h : Header file for sparse_double.c and sparse_mpf.c               */
/* Copyright (c) 2006-2011 Tomonori Kouya, All rights reserved.                 */
/*                                                                              */
/* 2007-11-13 Version 0.1: Bug fix of get_secv,                                 */
/*     append get_dvector_i, set_dvector_i, add_dvector                         */
/*     append nzero_row_dim[] and set_nzero_row_dim                             */
/*                                                                              */
/* 2011-06-28 Version 0.2: Reconstruction of sparse.h to be merged into BNCpack */
/* 2011-08-29 Version 0.3: Bug fix in sparse_double.c & sparse_mpf.c            */
/* 2011-03-20 Version 0.31:Add "zero_element" to *RSMatrix structure            */
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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//#include "bnc.h"
#include "dlinear.h"

// Intel Math Kernel
#ifdef USE_IMKL
	#include "mkl.h"
	#include "mkl_cblas.h" // for Intel Math Kernel Library
#endif // USE_IMKL

//#include "clinear.h"
//#include "mpflinear.h"
#include "cdlinear.h"
#include "cmpflinear.h"

#include "oz_scheme.h"

#ifndef __BNC_SPARSE_H__
#define __BNC_SPARSE_H__

#define ERROR   (-1)
#define SUCCESS (0)

#define EMPTY (-1)

#define LINE_BUF_LEN 128

/********************/
/* double precision */
/********************/

/* Sparse matrix struct */
/* Example:             */
/*      0 1 2 3 4       */
/* A = [a 0 b c 0]0     */
/*     [0 d 0 0 0]1     */
/*     [0 e f 0 0]2     */
/*     [0 0 0 g 0]3     */
/*     [0 0 h 0 i]4     */
/*                      */
/* <--> element = [a b c d e f g h i] */
/*      row_dim = 5, col_dim = 5      */
/*      nzero_index[0] = [0 2 3]    */
/*      nzero_index[1] = [4]        */
/*      nzero_index[2] = [1 2]      */
/*      nzero_index[3] = [3]        */
/*      nzero_index[4] = [2 4]      */
/*      nzero_col_dim[0] = 3 */
/*      nzero_col_dim[1] = 1 */
/*      nzero_col_dim[2] = 2 */
/*      nzero_col_dim[3] = 1 */
/*      nzero_col_dim[4] = 2 */
/*      nzero_total_num = 9 */
/*      nzero_row_dim[0] = 1 */
/*      nzero_row_dim[1] = 2 */
/*      nzero_row_dim[2] = 3 */
/*      nzero_row_dim[3] = 2 */
/*      nzero_row_dim[4] = 1 */

typedef struct {
	double *element;			// Elements of matrix
	long int row_dim, col_dim;	// Dimensions of Row and Column
	long int **nzero_index;		// Indeces of Non-zero elements
	long int *nzero_col_dim;	// Numbers of non-zero elements in i-th row
	long int *nzero_row_dim;	// Numbers of non-zero elements in i-th column
	long int nzero_total_num;	// Total number of non-zero elements
// ------------------------------------
// 2024-04-25(Thu) Appended by T.Kouya
// real_nzero_total_num := real_zero_col_dim[0] + ... + real_nzero_col_dim[row_dim - 1]
//      real_col_dim[i] := (nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH
// ------------------------------------
	long int real_nzero_total_num; // Real total number of non-zero elements;
	long int *real_nzero_col_dim; // Real numbers of non-zero elements in i-th row
//	long int *real_nzero_row_dim; // Real numbers of non-zero elements in i-th column
	double zero_element;        // Zero value (Append in 2012-03-20)
} drsmatrix;

/* Definition of "DRSMatrix" type */
typedef drsmatrix *DRSMatrix;

/* initialize DRSMatrix */
DRSMatrix init_drsmatrix(long int, long int *, long int);

/* Clear DRSMatrix */
void free_drsmatrix(DRSMatrix);

/* set nzero_row_dim automatically */
void set_nzero_row_dim(DRSMatrix);

/* Print DRSMatrix */
void print_drsmatrix(DRSMatrix);

/* Dense Matrix := Sparse Matrix */
void set_dmatrix_drsmatrix(DMatrix ret, DRSMatrix spmat);

/* initialize and substitute DRSMatrix from DMatrix */
DRSMatrix init_set_drsmatrix_dmatrix(DMatrix);

// 2024-07-30(Tue)
/* get the DRSMatrix ij-element */
double get_drsmatrix_ij(DRSMatrix mat, long int row_index, long int col_index);

// 2024-08-04 (Sun) T.Kouya
/* set the DRSMatrix ij-element */
void set_drsmatrix_ij(DRSMatrix mat, long int row_index, long int col_index, double val);

// 2024-07-30(Tue) T.Kouya
/* initialize and set DRSMatrix */
DRSMatrix init_set_drsmatrix(DRSMatrix spmat_org);

// 2024-07-30(Tue) T.Kouya
// spmat := 0
void set0_drsmatrix(DRSMatrix spmat);

/* Get variables to initialize DRSMatrix */
int get_vars_drsmatrix_fname(long int *, long int **, long int *, const char *);

/* Read URI linking data */
int fread_urilinkdat_fname(DRSMatrix, const char *);

// 2025-07-10(Thu) T.Kouya
#ifdef USE_IMKL
// ret_i_csr_start, ret_i_csr_end, ret_j_csr := (sparse_matrix_t)mat
void convert_indeces_drsmatrix_mkl_csrmat(MKL_INT *ret_i_csr_start, MKL_INT *ret_i_csr_end, MKL_INT *ret_j_csr, DRSMatrix mat);

// ret := (sparse_matrix_t)mat
sparse_status_t convert_drsmatrix_mkl_csrmat(sparse_matrix_t *ret, MKL_INT *ret_i_csr_start, MKL_INT *ret_i_csr_end, MKL_INT *ret_j_csr, DRSMatrix mat);

// ret := (sparse_matrix_t)mat
sparse_status_t subst_drsmatrix_mkl_csrmat(sparse_matrix_t *ret, MKL_INT *i_csr_start, MKL_INT *i_csr_end, MKL_INT *j_csr, DRSMatrix mat);
#endif // USE_IMKL

/* Multiply DRSMatrix * DVector */
int mul_drsmatrix_dvec(DVector, DRSMatrix, DVector);

/* Multiply DRSMatrix^T * DVector */
int mul_drsmatrixt_dvec(DVector, DRSMatrix, DVector);

/* Power Method for Randomly Sparse Matrices */
/* 	double *evec: the eigenvector for max eigenvalue */
/* 	double *drsmat: Randomly sparse matrix */
/* 	double reps, aeps: Relative and Absolute tolerance */
/* 	long int max_times: Maximum iterative times of Power method */
double dpower_rsmatrix(DVector, DRSMatrix, double, double, long int);

/* Scalar multiply of DVector */
int smul_dvector(DVector, double, DVector);

/* Select index of absolute maximum element and its value in DVector */
long int absmax_index_dvector(double *, DVector);

// 2024-07-30(Tue) T.Kouya
// ret := spmat_a
void subst_drsmatrix(DRSMatrix ret, DRSMatrix spmat_a);

// 2024-07-30(Tue) T.Kouya
// ret := spmat_a + spmat_b
void add_drsmatrix(DRSMatrix ret, DRSMatrix spmat_a, DRSMatrix spmat_b);

// 2024-07-30(Tue) T.Kouya
// ret := spmat_a - spmat_b
void sub_drsmatrix(DRSMatrix ret, DRSMatrix spmat_a, DRSMatrix spmat_b);

// 2024-07-30(Tue) T.Kouya
// ret := scaler * spmat_a
void cmul_drsmatrix(DRSMatrix ret, double scaler, DRSMatrix spmat_a);

// 2024-07-30(Tue) T.Kouya
// absmax_row_drsmatrix
double absmax_row_drsmatrix(long int *max_j, long int row_index, DRSMatrix mat);

// 2024-07-30(Tue) T.Kouya
// absmax_col_dmatrix
double absmax_col_drsmatrix(long int *max_i, long int col_index, DRSMatrix mat);

// 2024-08-01(Thu) T.Kouya
// Frobenius norm of mat
double normf_drsmatrix(DRSMatrix mat);

// 2024-07-30(Tue) T.Kouya
// SplitMat_A
void split_drsmatrix(DRSMatrix ret_high_mat, DRSMatrix ret_low_mat, DRSMatrix org_mat);

// 2024-08-05 (Mon) T.Kouya
// SplitMat_A
// row_shift holds num_div * row_dim exponents (may be NULL); see oz_scheme.h
int split_drsmatrix_drsmat_ex(DRSMatrix ret_mat[], long int row_shift[], int num_div, DRSMatrix org_mat);
int split_drsmatrix_drsmat(DRSMatrix ret_mat[], int num_div, DRSMatrix org_mat);

// 2024-08-04 (Sun) T.Kouya
// SplitMat_B
// return real_num_div
// col_shift holds num_div * col_dim exponents (may be NULL); see oz_scheme.h
int split_drsmatrix_t_drsmat_ex(DRSMatrix ret_mat[], long int col_shift[], int num_div, DRSMatrix org_mat);
int split_drsmatrix_t_drsmat(DRSMatrix ret_mat[], int num_div, DRSMatrix org_mat);

// Matrix-Vector multiplication based on Ozaki scheme
/*----------------------------------------------------------------*/
/* Blocked kernels shared by the sparse Ozaki-scheme routines      */
/*----------------------------------------------------------------*/
// offsets of each row into DRSMatrix::element[]; caller frees
long int *bnc_oz_sp_row_start(const long int *real_nzero_col_dim, long int row_dim);
// ret_block[0 .. num_rows) := a[first_row ...] * b
void bnc_oz_sp_dcsr_block(double *ret_block, DRSMatrix a, const long int *row_start, long int first_row, long int num_rows, DVector b);
// ret_block[0 .. num_rows) x b->col_dim := a[first_row ...] * b (dense b)
void bnc_oz_sp_dcsr_dmat_block(double *ret_block, long int ld_ret_block, DRSMatrix a, const long int *row_start, long int first_row, long int num_rows, DMatrix b);
// ret_block[0 .. col_dim) += a[first_row ...]^T * b
void bnc_oz_sp_dcsrt_block(double *ret_block, DRSMatrix a, const long int *row_start, long int first_row, long int num_rows, DVector b);

void mul_drsmatrix_dvec_oz(DVector ret, DRSMatrix a, int max_num_div_a, DVector vb, int max_num_div_vb);
// C := A(sparse) * B(dense) based on Ozaki scheme
void mul_drsmatrix_dmat_oz(DMatrix ret, DRSMatrix a, int max_num_div_a, DMatrix b, int max_num_div_b);

// 2024-08-02(Fri) T.Kouya
// Transposed Matrix-Vector multiplication based on Ozaki scheme
void mul_drsmatrixt_dvec_oz(DVector ret, DRSMatrix a, int max_num_div_a, DVector vb, int max_num_div_vb);

// Imcomplete LU decomposition; iLU0_drsmatrix
void iLU0_drsmatrix(DRSMatrix mat);

// iLU0_solve: iLU *  = b
void solve_iLU0_drsmatrix(DVector ret, DRSMatrix ilu, DVector b);

// iLU0t_solve: x^T * iLU = b^T
void solve_iLU0t_drsmatrix(DVector ret, DRSMatrix ilu, DVector b);

// Complex double sparse matrix
typedef struct{
	DRSMatrix re; // Real part
	DRSMatrix im; // Imaginary part
} cdrsmatrix;
typedef cdrsmatrix *CDRSMatrix;

/* initialize DRSMatrix */
CDRSMatrix init_cdrsmatrix(long int, long int *, long int);

/* Clear DRSMatrix */
void free_cdrsmatrix(CDRSMatrix);

/* Print DRSMatrix */
void print_cdrsmatrix(CDRSMatrix);

/* Dense Matrix := Sparse Matrix */
void set_cdmatrix_cdrsmatrix(CDMatrix ret, CDRSMatrix spmat);

// 2024-09-05(Thu) T.Kouya
// Frobenius norm of mat
double normf_cdrsmatrix(CDRSMatrix mat);

/* initialize and substitute DRSMatrix from DMatrix */
CDRSMatrix init_set_cdrsmatrix_cdmatrix(CDMatrix);

/* Initialize and substitute CDRSMatrix from CDRSMatrix */
CDRSMatrix init_set_cdrsmatrix(CDRSMatrix org_mat);

// 2024-11-07(Tue)
/* get the CDRSMatrix ij-element */
double _Complex get_cdrsmatrix_ij(CDRSMatrix mat, long int row_index, long int col_index);

// 2024-08-04 (Sun) T.Kouya
/* set the CDRSMatrix ij-element */
void set_cdrsmatrix_ij(CDRSMatrix mat, long int row_index, long int col_index, double _Complex val);

/* Multiply CDRSMatrix * CDVector */
int mul_cdrsmatrix_cdvec(CDVector, CDRSMatrix, CDVector);

/* Multiply CDRSMatrix^T * CDVector */
int mul_cdrsmatrixt_cdvec(CDVector, CDRSMatrix, CDVector);

/* Multiply conf(CDRSMatrix)^T * CDVector */
int mul_cdrsmatrixs_cdvec(CDVector, CDRSMatrix, CDVector);

// Imcomplete LU decomposition; iLU0_drsmatrix
void iLU0_cdrsmatrix(CDRSMatrix mat);

// iLU0_solve: iLU * x = b
void solve_iLU0_cdrsmatrix(CDVector ret, CDRSMatrix ilu, CDVector b);

// iLU0t_solve: x^T * iLU = b^T
void solve_iLU0t_cdrsmatrix(CDVector ret, CDRSMatrix ilu, CDVector b);

// iLU0s_solve: x^T * conj(iLU) = b^T
void solve_iLU0s_cdrsmatrix(CDVector ret, CDRSMatrix ilu, CDVector b);

// -------------
// Double-double
// ------------- 
#include "ddlinear.h"
#include "cddlinear.h"

// Double-double 
typedef struct {
	double *element[DDSIZE];	// Elements of matrix
	long int row_dim, col_dim;	// Dimensions of Row and Column
	long int **nzero_index;		// Indeces of Non-zero elements
	long int *nzero_col_dim;	// Numbers of non-zero elements in i-th row
	long int *nzero_row_dim;	// Numbers of non-zero elements in i-th column
	long int nzero_total_num;	// Total number of non-zero elements
// ------------------------------------
// 2024-04-25(Thu) Appended by T.Kouya
// real_nzero_total_num := real_zero_col_dim[0] + ... + real_nzero_col_dim[row_dim - 1]
//      real_col_dim[i] := (nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH
// ------------------------------------
	long int real_nzero_total_num; // Real total number of non-zero elements;
	long int *real_nzero_col_dim; // Real numbers of non-zero elements in i-th row
//	long int *real_nzero_row_dim; // Real numbers of non-zero elements in i-th column
} ddrsmatrix;
typedef ddrsmatrix *DDRSMatrix;

/* initialize DDRSMatrix */
DDRSMatrix init_ddrsmatrix(long int row_dim, long int *nzero_col_dim, long int nzero_total_num);

/* Clear DDRSMatrix */
void free_ddrsmatrix(DDRSMatrix mat);

/* set nzero_row_dim automatically */
void set_nzero_row_dim_dd(DDRSMatrix mat);

/* Print DDRSMatrix */
void print_ddrsmatrix(DDRSMatrix mat);

// 2024-09-05(Thu) T.Kouya
// Frobenius norm of mat
void normf_ddrsmatrix(double ret[DDSIZE], DDRSMatrix mat);

/* initialize and substitute DDRSMatrix from MPFMatrix */
DDRSMatrix init_set_ddrsmatrix_ddmatrix(DDMatrix org_mat);

// 2024-07-31(Wed) T.Kouya
/* get the DRSMatrix ij-element */
void get_ddrsmatrix_ij(double ret[DDSIZE], DDRSMatrix mat, long int row_index, long int col_index);

// 2024-08-04 (Sun) T.Kouya
/* set the DDRSMatrix ij-element */
void set_ddrsmatrix_ij(DDRSMatrix mat, long int row_index, long int col_index, double val[DDSIZE]);

// 2024-07-30(Tue) T.Kouya
// init and set DDRSMatrix
DDRSMatrix init_set_ddrsmatrix(DDRSMatrix org_sp);

// 2024-08-01(Thu) T.Kouya
// spmat := 0
void set0_ddrsmatrix(DDRSMatrix spmat);

// 2024-12-03(Tue) T.Kouya
// ret := spmat_a
void subst_ddrsmatrix(DDRSMatrix ret, DDRSMatrix spmat_a);

// 2024-08-01 T.Kouya
// init and set DRSMatrix from DDRSMatrix
DRSMatrix init_set_drsmatrix_ddrsmat(DDRSMatrix org_sp);

/* Get variables to initialize DDRSMatrix */
int get_vars_ddrsmatrix_fname(long int *ptr_row_dim, long int **ptr_nzero_col_dim, long int *ptr_nzero_total_num, const char *fname);

// 2025-07-10(Thu) T.Kouya
#ifdef USE_IMKL
// ret_i_csr_start, ret_i_csr_end, ret_j_csr := (sparse_matrix_t)mat
void convert_indeces_ddrsmatrix_mkl_csrmat(MKL_INT *ret_i_csr_start, MKL_INT *ret_i_csr_end, MKL_INT *ret_j_csr, DDRSMatrix mat);

// ret := (sparse_matrix_t)mat
sparse_status_t convert_ddrsmatrix_mkl_csrmat(sparse_matrix_t *ret[DDSIZE], MKL_INT *ret_i_csr_start, MKL_INT *ret_i_csr_end, MKL_INT *ret_j_csr, DDRSMatrix mat);

// ret := (sparse_matrix_t)mat
sparse_status_t subst_ddrsmatrix_mkl_csrmat(sparse_matrix_t *ret[DDSIZE], MKL_INT *i_csr_start, MKL_INT *i_csr_end, MKL_INT *j_csr, DDRSMatrix mat);
#endif // USE_IMKL

/* Multiply DDRSMatrix * DDVector */
int mul_ddrsmatrix_ddvec(DDVector ret, DDRSMatrix mat, DDVector vec);

/* Multiply DDRSMatrix^T * DDVector */
int mul_ddrsmatrixt_ddvec(DDVector ret, DDRSMatrix mat, DDVector vec);

/* Multiply DRSMatrix * DDVector */
int mul_drsmatrix_ddvec(DDVector ret, DRSMatrix mat, DDVector vec);

/* Multiply DRSMatrix^T * DDVector */
int mul_drsmatrixt_ddvec(DDVector ret, DRSMatrix mat, DDVector vec);

/* Power Method for Randomly Sparse Matrices */
/* 	DDVector *evec: the eigenvector for max eigenvalue */
/* 	DDRSMatrix *drsmat: Randomly sparse matrix */
/* 	double *reps, *aeps: Relative and Absolute tolerance */
/* 	long int max_times: Maximum iterative times of Power method */
void ddpower_sp(double max_eig[DDSIZE], DDVector evec, DDRSMatrix mat, double reps[DDSIZE], double aeps[DDSIZE], long int max_times);

/* Select index of absolute maximum element and its value in DDVector */
long int absmax_index_ddvector(double ret[DDSIZE], DDVector vec);

// 2024-07-30 (Tue)
/* c := (dd)a */
void subst_ddrsmatrix_drsmat(DDRSMatrix c, DRSMatrix a);

// 2024-07-30 (Tue)
/* c := (d)a */
void subst_drsmatrix_ddrsmat(DRSMatrix c, DDRSMatrix a);

// 2024-07-31(Ted) T.Kouya
// absmax_row_ddrsmatrix
void absmax_row_ddrsmatrix(double mu[DDSIZE], long int *max_j, long int row_index, DDRSMatrix mat);

// 2024-08-04 (SUN) T.Kouya
// absmax_col_ddrsmatrix
void absmax_col_ddrsmatrix(double mu[DDSIZE], long int *max_i, long int col_index, DDRSMatrix mat);

// 2024-07-31(Wed) T.Kouya
/* c := a + (double)b */
void add_ddrsmatrix_drsmat(DDRSMatrix c, DDRSMatrix a, DRSMatrix b);

// 2024-07-31(Wed) T.Kouya
/* c := a - (double)b */
void sub_ddrsmatrix_drsmat(DDRSMatrix c, DDRSMatrix a, DRSMatrix b);

// 2024-07-31(Wed) T.Kouya
// SplitMat_A
// return real_num_div
// row_shift holds num_div * row_dim exponents (may be NULL); see oz_scheme.h
int split_ddrsmatrix_drsmat_ex(DRSMatrix ret_mat[], long int row_shift[], int num_div, DDRSMatrix org_mat);
int split_ddrsmatrix_drsmat(DRSMatrix ret_mat[], int num_div, DDRSMatrix org_mat);

// 2024-08-04 (Sun) T.Kouya
// SplitMat_B
// return real_num_div
// col_shift holds num_div * col_dim exponents (may be NULL); see oz_scheme.h
int split_ddrsmatrix_t_drsmat_ex(DRSMatrix ret_mat[], long int col_shift[], int num_div, DDRSMatrix org_mat);
int split_ddrsmatrix_t_drsmat(DRSMatrix ret_mat[], int num_div, DDRSMatrix org_mat);

// Matrix-Vector multiplication based on Ozaki scheme
void mul_ddrsmatrix_ddvec_oz(DDVector ret, DDRSMatrix a, int max_num_div_a, DDVector vb, int max_num_div_vb); 
// C := A(sparse) * B(dense) based on Ozaki scheme
void mul_ddrsmatrix_ddmat_oz(DDMatrix ret, DDRSMatrix a, int max_num_div_a, DDMatrix b, int max_num_div_b);

// 2024-08-02(Fri) T.Kouya
// Transposed Matrix-Vector multiplication based on Ozaki scheme
void mul_ddrsmatrixt_ddvec_oz(DDVector ret, DDRSMatrix a, int max_num_div_a, DDVector vb, int max_num_div_vb);

// Incomplete LU decomposition; iLU0_drsmatrix
void iLU0_ddrsmatrix(DDRSMatrix mat);

// iLU0_solve: iLU * x = b
void solve_iLU0_ddrsmatrix(DDVector ret, DDRSMatrix ilu, DDVector b);

// iLU0t_solve: x^T * iLU = b^T
void solve_iLU0t_ddrsmatrix(DDVector ret, DDRSMatrix ilu, DDVector b);

// iLU0_solve: iLU * x = b
void solve_iLU0_drsmatrix_ddvec(DDVector ret, DRSMatrix ilu, DDVector b);

// iLU0t_solve: x^T * iLU = b^T
void solve_iLU0t_drsmatrix_ddvec(DDVector ret, DRSMatrix ilu, DDVector b);

// Complex DD sparse matrix
typedef struct{
	DDRSMatrix re; // Real part
	DDRSMatrix im; // Imaginary part
} cddrsmatrix;
typedef cddrsmatrix *CDDRSMatrix;

/* initialize CDRSMatrix */
CDDRSMatrix init_cddrsmatrix(long int row_dim, long int *nzero_col_dim, long int nzero_total_num);

/* Clear CDDRSMatrix */
void free_cddrsmatrix(CDDRSMatrix mat);

/* Print CDDRSMatrix */
void print_cddrsmatrix(CDDRSMatrix mat);

// 2024-09-05(Thu) T.Kouya
// Frobenius norm of mat
void normf_cddrsmatrix(double ret[DDSIZE], CDDRSMatrix mat);

// 2024-12-03 (Tue) T.Kouya
CDDRSMatrix init_set_cddrsmatrix(CDDRSMatrix org_mat);

/* Multiply CDDRSMatrix * CDDVector */
int mul_cddrsmatrix_cddvec(CDDVector ret, CDDRSMatrix mat, CDDVector vec);

/* Multiply CDDRSMatrix^T * CDDVector */
int mul_cddrsmatrixt_cddvec(CDDVector ret, CDDRSMatrix mat, CDDVector vec);

/* Multiply conj(CDDRSMatrix)^T * CDDVector */
int mul_cddrsmatrixs_cddvec(CDDVector ret, CDDRSMatrix mat, CDDVector vec);

/* Multiply CDRSMatrix * CDDVector */
int mul_cdrsmatrix_cddvec(CDDVector ret, CDRSMatrix mat, CDDVector vec);

/* Multiply CDRSMatrix^T * CDDVector */
int mul_cdrsmatrixt_cddvec(CDDVector ret, CDRSMatrix mat, CDDVector vec);

/* Multiply conj(CDRSMatrix)^T * CDDVector */
int mul_cdrsmatrixs_cddvec(CDDVector ret, CDRSMatrix mat, CDDVector vec);

// Imcomplete LU decomposition; iLU0_drsmatrix
void iLU0_cddrsmatrix(CDDRSMatrix mat);

// iLU0_solve: iLU * x = b
void solve_iLU0_cddrsmatrix(CDDVector ret, CDDRSMatrix ilu, CDDVector b);

// iLU0_solve: iLU * x = b
void solve_iLU0t_cddrsmatrix(CDDVector ret, CDDRSMatrix ilu, CDDVector b);

// iLU0s_solve: x^T * conj(iLU) = b^T
void solve_iLU0s_cddrsmatrix(CDDVector ret, CDDRSMatrix ilu, CDDVector b);

// iLU0_solve: iLU * x = b
void solve_iLU0_cdrsmatrix_cddvec(CDDVector ret, CDRSMatrix ilu, CDDVector b);

// iLU0_solve: iLU * x = b
void solve_iLU0t_cdrsmatrix_cddvec(CDDVector ret, CDRSMatrix ilu, CDDVector b);

// iLU0s_solve: x^T * conj(iLU) = b^T
void solve_iLU0s_cdrsmatrix_cddvec(CDDVector ret, CDRSMatrix ilu, CDDVector b);

// -------------
// Triple-double
// ------------- 
#include "tdlinear.h"
#include "ctdlinear.h"

// Triple-double 
typedef struct {
	double *element[TDSIZE];	// Elements of matrix
	long int row_dim, col_dim;	// Dimensions of Row and Column
	long int **nzero_index;		// Indeces of Non-zero elements
	long int *nzero_col_dim;	// Numbers of non-zero elements in i-th row
	long int *nzero_row_dim;	// Numbers of non-zero elements in i-th column
	long int nzero_total_num;	// Total number of non-zero elements
// ------------------------------------
// 2024-04-25(Thu) Appended by T.Kouya
// real_nzero_total_num := real_zero_col_dim[0] + ... + real_nzero_col_dim[row_dim - 1]
//      real_col_dim[i] := (nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH
// ------------------------------------
	long int real_nzero_total_num; // Real total number of non-zero elements;
	long int *real_nzero_col_dim; // Real numbers of non-zero elements in i-th row
//	long int *real_nzero_row_dim; // Real numbers of non-zero elements in i-th column
} tdrsmatrix;
typedef tdrsmatrix *TDRSMatrix;

/* initialize TDRSMatrix */
TDRSMatrix init_tdrsmatrix(long int row_dim, long int *nzero_col_dim, long int nzero_total_num);

/* Clear TDRSMatrix */
void free_tdrsmatrix(TDRSMatrix mat);

/* set nzero_row_dim automatically */
void set_nzero_row_dim_td(TDRSMatrix mat);

/* Print TDRSMatrix */
void print_tdrsmatrix(TDRSMatrix mat);

// 2024-08-01(Thu) T.Kouya
/* get the DRSMatrix ij-element */
void get_tdrsmatrix_ij(double ret[TDSIZE], TDRSMatrix mat, long int row_index, long int col_index);

// 2024-08-04 (Sun) T.Kouya
/* set the TDRSMatrix ij-element */
void set_tdrsmatrix_ij(TDRSMatrix mat, long int row_index, long int col_index, double val[TDSIZE]);

// 2024-07-30(Tue) T.Kouya
// init and set TDRSMatrix
TDRSMatrix init_set_tdrsmatrix(TDRSMatrix org_sp);

// 2024-08-01(Thu) T.Kouya
// spmat := 0
void set0_tdrsmatrix(TDRSMatrix spmat);

// 2024-12-03(Tue) T.Kouya
// ret := spmat_a
void subst_tdrsmatrix(TDRSMatrix ret, TDRSMatrix spmat_a);

// 2024-08-01 T.Kouya
// init and set DRSMatrix from TDRSMatrix
DRSMatrix init_set_drsmatrix_tdrsmat(TDRSMatrix org_sp);

/* initialize and substitute TDRSMatrix from MPFMatrix */
TDRSMatrix init_set_tdrsmatrix_tdmatrix(TDMatrix org_mat);

/* Get variables to initialize TDRSMatrix */
int get_vars_tdrsmatrix_fname(long int *ptr_row_dim, long int **ptr_nzero_col_dim, long int *ptr_nzero_total_num, const char *fname);

// 2025-07-10(Thu) T.Kouya
#ifdef USE_IMKL
// ret_i_csr_start, ret_i_csr_end, ret_j_csr := (sparse_matrix_t)mat
void convert_indeces_tdrsmatrix_mkl_csrmat(MKL_INT *ret_i_csr_start, MKL_INT *ret_i_csr_end, MKL_INT *ret_j_csr, TDRSMatrix mat);

// ret := (sparse_matrix_t)mat
sparse_status_t convert_tdrsmatrix_mkl_csrmat(sparse_matrix_t *ret[TDSIZE], MKL_INT *ret_i_csr_start, MKL_INT *ret_i_csr_end, MKL_INT *ret_j_csr, TDRSMatrix mat);

// ret := (sparse_matrix_t)mat
sparse_status_t subst_tdrsmatrix_mkl_csrmat(sparse_matrix_t *ret[TDSIZE], MKL_INT *i_csr_start, MKL_INT *i_csr_end, MKL_INT *j_csr, TDRSMatrix mat);
#endif // USE_IMKL

/* Multiply TDRSMatrix * TDVector */
int mul_tdrsmatrix_tdvec(TDVector ret, TDRSMatrix mat, TDVector vec);

/* Multiply TDRSMatrix^T * TDVector */
int mul_tdrsmatrixt_tdvec(TDVector ret, TDRSMatrix mat, TDVector vec);

/* Multiply DRSMatrix * TDVector */
int mul_drsmatrix_tdvec(TDVector ret, DRSMatrix mat, TDVector vec);

/* Multiply DRSMatrix^T * TDVector */
int mul_drsmatrixt_tdvec(TDVector ret, DRSMatrix mat, TDVector vec);

/* Power Method for Randomly Sparse Matrices */
/* 	TDVector *evec: the eigenvector for max eigenvalue */
/* 	TDRSMatrix *drsmat: Randomly sparse matrix */
/* 	double *reps, *aeps: Relative and Absolute tolerance */
/* 	long int max_times: Maximum iterative times of Power method */
void tdpower_sp(double max_eig[TDSIZE], TDVector evec, TDRSMatrix mat, double reps[TDSIZE], double aeps[TDSIZE], long int max_times);

/* Select index of absolute maximum element and its value in TDVector */
long int absmax_index_tdvector(double ret[TDSIZE], TDVector vec);

// 2024-07-30 (Tue)
/* c := (td)a */
void subst_tdrsmatrix_drsmat(TDRSMatrix c, DRSMatrix a);

// 2024-07-30 (Tue)
/* c := (d)a */
void subst_drsmatrix_tdrsmat(DRSMatrix c, TDRSMatrix a);

// 2024-07-31(Ted) T.Kouya
// absmax_row_tdrsmatrix
void absmax_row_tdrsmatrix(double mu[TDSIZE], long int *max_j, long int row_index, TDRSMatrix mat);

// 2024-08-04 (SUN) T.Kouya
// absmax_col_tdrsmatrix
void absmax_col_tdrsmatrix(double mu[TDSIZE], long int *max_i, long int col_index, TDRSMatrix mat);

// 2024-07-31(Wed) T.Kouya
/* c := a + (double)b */
void add_tdrsmatrix_drsmat(TDRSMatrix c, TDRSMatrix a, DRSMatrix b);

// 2024-07-31(Wed) T.Kouya
/* c := a - (double)b */
void sub_tdrsmatrix_drsmat(TDRSMatrix c, TDRSMatrix a, DRSMatrix b);

// 2024-07-31(Wed) T.Kouya
// SplitMat_A
// return real_num_div
// row_shift holds num_div * row_dim exponents (may be NULL); see oz_scheme.h
int split_tdrsmatrix_drsmat_ex(DRSMatrix ret_mat[], long int row_shift[], int num_div, TDRSMatrix org_mat);
int split_tdrsmatrix_drsmat(DRSMatrix ret_mat[], int num_div, TDRSMatrix org_mat);

// 2024-08-04 (Sun) T.Kouya
// SplitMat_B
// return real_num_div
// col_shift holds num_div * col_dim exponents (may be NULL); see oz_scheme.h
int split_tdrsmatrix_t_drsmat_ex(DRSMatrix ret_mat[], long int col_shift[], int num_div, TDRSMatrix org_mat);
int split_tdrsmatrix_t_drsmat(DRSMatrix ret_mat[], int num_div, TDRSMatrix org_mat);

// Matrix-Vector multiplication based on Ozaki scheme
void mul_tdrsmatrix_tdvec_oz(TDVector ret, TDRSMatrix a, int max_num_div_a, TDVector vb, int max_num_div_vb);
// C := A(sparse) * B(dense) based on Ozaki scheme
void mul_tdrsmatrix_tdmat_oz(TDMatrix ret, TDRSMatrix a, int max_num_div_a, TDMatrix b, int max_num_div_b);

// 2024-08-02(Fri) T.Kouya
// Transposed Matrix-Vector multiplication based on Ozaki scheme
void mul_tdrsmatrixt_tdvec_oz(TDVector ret, TDRSMatrix a, int max_num_div_a, TDVector vb, int max_num_div_vb);

// Incomplete LU decomposition; iLU0_tdrsmatrix
void iLU0_tdrsmatrix(TDRSMatrix mat);

// iLU0_solve: iLU * x = b
void solve_iLU0_tdrsmatrix(TDVector ret, TDRSMatrix ilu, TDVector b);

// iLU0t_solve: x^T * iLU = b^T
void solve_iLU0t_tdrsmatrix(TDVector ret, TDRSMatrix ilu, TDVector b);

// iLU0_solve: iLU * x = b
void solve_iLU0_drsmatrix_tdvec(TDVector ret, DRSMatrix ilu, TDVector b);

// iLU0t_solve: x^T * iLU = b^T
void solve_iLU0t_drsmatrix_tdvec(TDVector ret, DRSMatrix ilu, TDVector b);

// Complex TD sparse matrix
typedef struct{
	TDRSMatrix re; // Real part
	TDRSMatrix im; // Imaginary part
} ctdrsmatrix;
typedef ctdrsmatrix *CTDRSMatrix;

/* initialize CTDRSMatrix */
CTDRSMatrix init_ctdrsmatrix(long int row_dim, long int *nzero_col_dim, long int nzero_total_num);

/* Clear CTDRSMatrix */
void free_ctdrsmatrix(CTDRSMatrix mat);

/* Print CTDRSMatrix */
void print_ctdrsmatrix(CTDRSMatrix mat);

// 2024-12-03 (Tue) T.Kouya
CTDRSMatrix init_set_ctdrsmatrix(CTDRSMatrix org_mat);;

/* Multiply CTDRSMatrix * CTDVector */
int mul_ctdrsmatrix_ctdvec(CTDVector ret, CTDRSMatrix mat, CTDVector vec);

/* Multiply CTDRSMatrix^T * CTDVector */
int mul_ctdrsmatrixt_ctdvec(CTDVector ret, CTDRSMatrix mat, CTDVector vec);

/* Multiply conj(CTDRSMatrix)^T * CTDVector */
int mul_ctdrsmatrixs_ctdvec(CTDVector ret, CTDRSMatrix mat, CTDVector vec);

/* Multiply CDRSMatrix * CTDVector */
int mul_cdrsmatrix_ctdvec(CTDVector ret, CDRSMatrix mat, CTDVector vec);

/* Multiply CDRSMatrix^T * CTDVector */
int mul_cdrsmatrixt_ctdvec(CTDVector ret, CDRSMatrix mat, CTDVector vec);

/* Multiply conj(CDRSMatrix)^T * CTDVector */
int mul_cdrsmatrixs_ctdvec(CTDVector ret, CDRSMatrix mat, CTDVector vec);

// Imcomplete LU decomposition; iLU0_drsmatrix
void iLU0_ctdrsmatrix(CTDRSMatrix mat);

// iLU0_solve: iLU * x = b
void solve_iLU0_ctdrsmatrix(CTDVector ret, CTDRSMatrix ilu, CTDVector b);

// iLU0t_solve: x^t * iLU = b^t
void solve_iLU0t_ctdrsmatrix(CTDVector ret, CTDRSMatrix ilu, CTDVector b);

// iLU0s_solve: x^t * conj(iLU) = b^t
void solve_iLU0s_ctdrsmatrix(CTDVector ret, CTDRSMatrix ilu, CTDVector b);

// iLU0_solve: iLU * x = b
void solve_iLU0_cdrsmatrix_ctdvec(CTDVector ret, CDRSMatrix ilu, CTDVector b);

// iLU0_solve: x^t * iLU = b^t
void solve_iLU0t_cdrsmatrix_ctdvec(CTDVector ret, CDRSMatrix ilu, CTDVector b);

// iLU0_solve: x^t * conj(iLU) = b^t
void solve_iLU0s_cdrsmatrix_ctdvec(CTDVector ret, CDRSMatrix ilu, CTDVector b);


// ----------------
// Quadruple-double
// ---------------- 
#include "qdlinear.h"
#include "cqdlinear.h"

// Quadruple-double 
typedef struct {
	double *element[QDSIZE];	// Elements of matrix
	long int row_dim, col_dim;	// Dimensions of Row and Column
	long int **nzero_index;		// Indeces of Non-zero elements
	long int *nzero_col_dim;	// Numbers of non-zero elements in i-th row
	long int *nzero_row_dim;	// Numbers of non-zero elements in i-th column
	long int nzero_total_num;	// Total number of non-zero elements
// ------------------------------------
// 2024-04-25(Thu) Appended by T.Kouya
// real_nzero_total_num := real_zero_col_dim[0] + ... + real_nzero_col_dim[row_dim - 1]
//      real_col_dim[i] := (nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH
// ------------------------------------
	long int real_nzero_total_num; // Real total number of non-zero elements;
	long int *real_nzero_col_dim; // Real numbers of non-zero elements in i-th row
//	long int *real_nzero_row_dim; // Real numbers of non-zero elements in i-th column
} qdrsmatrix;
typedef qdrsmatrix *QDRSMatrix;

/* initialize QDRSMatrix */
QDRSMatrix init_qdrsmatrix(long int row_dim, long int *nzero_col_dim, long int nzero_total_num);

/* Clear QDRSMatrix */
void free_qdrsmatrix(QDRSMatrix mat);

/* set nzero_row_dim automatically */
void set_nzero_row_dim_qd(QDRSMatrix mat);

/* Print QDRSMatrix */
void print_qdrsmatrix(QDRSMatrix mat);

/* initialize and substitute QDRSMatrix from MPFMatrix */
QDRSMatrix init_set_qdrsmatrix_qdmatrix(QDMatrix org_mat);

// 2024-08-02(Fri) T.Kouya
/* get the DRSMatrix ij-element */
void get_qdrsmatrix_ij(double ret[QDSIZE], QDRSMatrix mat, long int row_index, long int col_index);

// 2024-08-04 (Sun) T.Kouya
/* set the QDRSMatrix ij-element */
void set_qdrsmatrix_ij(QDRSMatrix mat, long int row_index, long int col_index, double val[QDSIZE]);

// 2024-08-02 (Fri) T.Kouya
// init and set QDRSMatrix
QDRSMatrix init_set_qdrsmatrix(QDRSMatrix org_sp);

// 2024-08-01(Thu) T.Kouya
// spmat := 0
void set0_qdrsmatrix(QDRSMatrix spmat);

// 2024-12-03(Tue) T.Kouya
// ret := spmat_a
void subst_qdrsmatrix(QDRSMatrix ret, QDRSMatrix spmat_a);

// 2024-08-01 T.Kouya
// init and set DRSMatrix from QDRSMatrix
DRSMatrix init_set_drsmatrix_qdrsmat(QDRSMatrix org_sp);

#ifdef USE_GMP
// MPFRSMatrix -> QDRSMatrix
//QDRSMatrix init_set_qdrsmatrix_mpfrsmatrix(MPFRSMatrix org_sp);
#endif // USE_GMP

/* Get variables to initialize QDRSMatrix */
int get_vars_qdrsmatrix_fname(long int *ptr_row_dim, long int **ptr_nzero_col_dim, long int *ptr_nzero_total_num, const char *fname);

// 2025-07-10(Thu) T.Kouya
#ifdef USE_IMKL
// ret_i_csr_start, ret_i_csr_end, ret_j_csr := (sparse_matrix_t)mat
void convert_indeces_qdrsmatrix_mkl_csrmat(MKL_INT *ret_i_csr_start, MKL_INT *ret_i_csr_end, MKL_INT *ret_j_csr, QDRSMatrix mat);

// ret := (sparse_matrix_t)mat
sparse_status_t convert_qdrsmatrix_mkl_csrmat(sparse_matrix_t *ret[QDSIZE], MKL_INT *ret_i_csr_start, MKL_INT *ret_i_csr_end, MKL_INT *ret_j_csr, QDRSMatrix mat);

// ret := (sparse_matrix_t)mat
sparse_status_t subst_qdrsmatrix_mkl_csrmat(sparse_matrix_t *ret[QDSIZE], MKL_INT *i_csr_start, MKL_INT *i_csr_end, MKL_INT *j_csr, QDRSMatrix mat);
#endif // USE_IMKL

/* Multiply QDRSMatrix * QDVector */
int mul_qdrsmatrix_qdvec(QDVector ret, QDRSMatrix mat, QDVector vec);

/* Multiply QDRSMatrix^T * QDVector */
int mul_qdrsmatrixt_qdvec(QDVector ret, QDRSMatrix mat, QDVector vec);

/* Multiply DRSMatrix * QDVector */
int mul_drsmatrix_qdvec(QDVector ret, DRSMatrix mat, QDVector vec);

/* Multiply DRSMatrix^T * QDVector */
int mul_drsmatrixt_qdvec(QDVector ret, DRSMatrix mat, QDVector vec);

/* Power Method for Randomly Sparse Matrices */
/* 	QDVector *evec: the eigenvector for max eigenvalue */
/* 	QDRSMatrix *drsmat: Randomly sparse matrix */
/* 	double *reps, *aeps: Relative and Absolute tolerance */
/* 	long int max_times: Maximum iterative times of Power method */
void qdpower_sp(double max_eig[QDSIZE], QDVector evec, QDRSMatrix mat, double reps[QDSIZE], double aeps[QDSIZE], long int max_times);

/* Select index of absolute maximum element and its value in QDVector */
long int absmax_index_qdvector(double ret[QDSIZE], QDVector vec);

// 2024-08-02 (Fri) T.Kouya
/* c := (qd)a */
void subst_qdrsmatrix_drsmat(QDRSMatrix c, DRSMatrix a);

// 2024-08-02 (Fri) T.Kouya
/* c := (d)a */
void subst_drsmatrix_qdrsmat(DRSMatrix c, QDRSMatrix a);

// 2024-08-02 (Fri) T.Kouya
// absmax_row_qdrsmatrix
void absmax_row_qdrsmatrix(double mu[QDSIZE], long int *max_j, long int row_index, QDRSMatrix mat);

// 2024-08-04 (SUN) T.Kouya
// absmax_col_qdrsmatrix
void absmax_col_qdrsmatrix(double mu[QDSIZE], long int *max_i, long int col_index, QDRSMatrix mat);

// 2024-08-02 (Fri) T.Kouya
/* c := a + (double)b */
void add_qdrsmatrix_drsmat(QDRSMatrix c, QDRSMatrix a, DRSMatrix b);

// 2024-07-31(Wed) T.Kouya
/* c := a - (double)b */
void sub_qdrsmatrix_drsmat(QDRSMatrix c, QDRSMatrix a, DRSMatrix b);

// 2024-07-31(Wed) T.Kouya
// SplitMat_A
// return real_num_div
// row_shift holds num_div * row_dim exponents (may be NULL); see oz_scheme.h
int split_qdrsmatrix_drsmat_ex(DRSMatrix ret_mat[], long int row_shift[], int num_div, QDRSMatrix org_mat);
int split_qdrsmatrix_drsmat(DRSMatrix ret_mat[], int num_div, QDRSMatrix org_mat);

// 2024-08-04 (Sun) T.Kouya
// SplitMat_B
// return real_num_div
// col_shift holds num_div * col_dim exponents (may be NULL); see oz_scheme.h
int split_qdrsmatrix_t_drsmat_ex(DRSMatrix ret_mat[], long int col_shift[], int num_div, QDRSMatrix org_mat);
int split_qdrsmatrix_t_drsmat(DRSMatrix ret_mat[], int num_div, QDRSMatrix org_mat);

// Matrix-Vector multiplication based on Ozaki scheme
void mul_qdrsmatrix_qdvec_oz(QDVector ret, QDRSMatrix a, int max_num_div_a, QDVector vb, int max_num_div_vb);
// C := A(sparse) * B(dense) based on Ozaki scheme
void mul_qdrsmatrix_qdmat_oz(QDMatrix ret, QDRSMatrix a, int max_num_div_a, QDMatrix b, int max_num_div_b);

// 2024-08-02(Fri) T.Kouya
// Transposed Matrix-Vector multiplication based on Ozaki scheme
void mul_qdrsmatrixt_qdvec_oz(QDVector ret, QDRSMatrix a, int max_num_div_a, QDVector vb, int max_num_div_vb);

// Incomplete LU decomposition; iLU0_qdrsmatrix
void iLU0_qdrsmatrix(QDRSMatrix mat);

// iLU0_solve: iLU * x = b
void solve_iLU0_qdrsmatrix(QDVector ret, QDRSMatrix ilu, QDVector b);

// iLU0t_solve: x^T * iLU = b^T
void solve_iLU0t_qdrsmatrix(QDVector ret, QDRSMatrix ilu, QDVector b);

// iLU0_solve: iLU * x = b
void solve_iLU0_drsmatrix_qdvec(QDVector ret, DRSMatrix ilu, QDVector b);

// iLU0t_solve: x^T * iLU = b^T
void solve_iLU0t_drsmatrix_qdvec(QDVector ret, DRSMatrix ilu, QDVector b);

// Complex QD sparse matrix
typedef struct{
	QDRSMatrix re; // Real part
	QDRSMatrix im; // Imaginary part
} cqdrsmatrix;
typedef cqdrsmatrix *CQDRSMatrix;

/* initialize CQDRSMatrix */
CQDRSMatrix init_cqdrsmatrix(long int row_dim, long int *nzero_col_dim, long int nzero_total_num);

/* Clear CQDRSMatrix */
void free_cqdrsmatrix(CQDRSMatrix mat);

/* Print CQDRSMatrix */
void print_cqdrsmatrix(CQDRSMatrix mat);

// 2024-12-03 (Tue) T.Kouya
CQDRSMatrix init_set_cqdrsmatrix(CQDRSMatrix org_mat);

/* Multiply CQDRSMatrix * CQDVector */
int mul_cqdrsmatrix_cqdvec(CQDVector ret, CQDRSMatrix mat, CQDVector vec);

/* Multiply CQDRSMatrix^T * CQDVector */
int mul_cqdrsmatrixt_cqdvec(CQDVector ret, CQDRSMatrix mat, CQDVector vec);

/* Multiply conj(CQDRSMatrix)^T * CQDVector */
int mul_cqdrsmatrixs_cqdvec(CQDVector ret, CQDRSMatrix mat, CQDVector vec);

/* Multiply CDRSMatrix * CQDVector */
int mul_cdrsmatrix_cqdvec(CQDVector ret, CDRSMatrix mat, CQDVector vec);

/* Multiply CDRSMatrix^T * CQDVector */
int mul_cdrsmatrixt_cqdvec(CQDVector ret, CDRSMatrix mat, CQDVector vec);

/* Multiply conj(CDRSMatrix)^T * CQDVector */
int mul_cdrsmatrixs_cqdvec(CQDVector ret, CDRSMatrix mat, CQDVector vec);

// Imcomplete LU decomposition; iLU0_drsmatrix
void iLU0_cqdrsmatrix(CQDRSMatrix mat);

// iLU0_solve: iLU * x = b
void solve_iLU0_cqdrsmatrix(CQDVector ret, CQDRSMatrix ilu, CQDVector b);

// iLU0s_solve: x^t * iLU = b^t
void solve_iLU0t_cqdrsmatrix(CQDVector ret, CQDRSMatrix ilu, CQDVector b);

// iLU0s_solve: x^t * conj(iLU) = b^t
void solve_iLU0s_cqdrsmatrix(CQDVector ret, CQDRSMatrix ilu, CQDVector b);

// iLU0_solve: iLU * x = b
void solve_iLU0_cdrsmatrix_cqdvec(CQDVector ret, CDRSMatrix ilu, CQDVector b);

// iLU0t_solve: x^t * iLU = b^t
void solve_iLU0t_cdrsmatrix_cqdvec(CQDVector ret, CDRSMatrix ilu, CQDVector b);

// iLU0s_solve: x^t * conj(iLU) = b^t
void solve_iLU0s_cdrsmatrix_cqdvec(CQDVector ret, CDRSMatrix ilu, CQDVector b);


/* MPF */
#ifdef USE_GMP
//#include "mpflinear.h"
//#include "clinear.h"

/* Sparse matrix struct */
/* Example:             */
/*      0 1 2 3 4       */
/* A = [a 0 b c 0]0     */
/*     [0 d 0 0 0]1     */
/*     [0 e f 0 0]2     */
/*     [0 0 0 g 0]3     */
/*     [0 0 h 0 i]4     */
/*                      */
/* <--> element = [a b c d e f g h i] */
/*      row_dim = 5, col_dim = 5      */
/*      nzero_index[0] = [0 2 3]    */
/*      nzero_index[1] = [4]        */
/*      nzero_index[2] = [1 2]      */
/*      nzero_index[3] = [3]        */
/*      nzero_index[4] = [2 4]      */
/*      nzero_col_dim[0] = 3 */
/*      nzero_col_dim[1] = 1 */
/*      nzero_col_dim[2] = 2 */
/*      nzero_col_dim[3] = 1 */
/*      nzero_col_dim[4] = 2 */
/*      nzero_total_num =  9 */
/*      nzero_row_dim[0] = 1 */
/*      nzero_row_dim[1] = 2 */
/*      nzero_row_dim[2] = 3 */
/*      nzero_row_dim[3] = 2 */
/*      nzero_row_dim[4] = 1 */

typedef struct {
	unsigned long prec;
	mpf_t *element;				// Elements of matrix
	long int row_dim, col_dim;	// Dimensions of Row and Column
	long int **nzero_index;		// Indeces of Non-zero elements
	long int *nzero_col_dim;	// Numbers of non-zero elements in i-th row
	long int *nzero_row_dim;	// Numbers of non-zero elements in i-th column
	long int nzero_total_num;	// Total number of non-zero elements
	mpf_t zero_element;         // Zero value (Append in 2012-03-20)
} mpfrsmatrix;

/* Definition of "MPFRSMatrix" type */
typedef mpfrsmatrix *MPFRSMatrix;

/* initialize MPFRSMatrix */
MPFRSMatrix init_mpfrsmatrix(long int, long int *, long int);

/* initialize MPFRSMatrix with specific precision */
MPFRSMatrix init2_mpfrsmatrix(long int, long int *, long int, unsigned long prec);

/* Clear MPFRSMatrix */
void free_mpfrsmatrix(MPFRSMatrix);

/* set nzero_row_dim automatically */
void set_nzero_row_dim_mpf(MPFRSMatrix);

/* Print MPFRSMatrix */
void print_mpfrsmatrix(MPFRSMatrix);
// 2024-10-16(Wed) T.Kouya

/* get the MPFRSMatrix ij-element */
void get_mpfrsmatrix_ij(mpf_t ret, MPFRSMatrix mat, long int row_index, long int col_index);

// 2024-10-16 (Wed) T.Kouya
/* set the MPFRSMatrix ij-element */
void set_mpfrsmatrix_ij(MPFRSMatrix mat, long int row_index, long int col_index, mpf_t val);

// 2024-10-16(Wed) T.Kouya
// init and set MPFRSMatrix
MPFRSMatrix init_set_mpfrsmatrix(MPFRSMatrix org_sp);

// 2024-10-16(Wed) T.Kouya
// spmat := 0
void set0_mpfrsmatrix(MPFRSMatrix spmat);

/* Dense Matrix := Sparse Matrix */
void set_mpfmatrix_mpfrsmat(MPFMatrix ret, MPFRSMatrix spmat);

/* initialize and substitute MPFRSMatrix from MPFMatrix */
MPFRSMatrix init_set_mpfrsmatrix_mpfmatrix(MPFMatrix);

/* Get variables to initialize MPFRSMatrix */
int get_vars_mpfrsmatrix_fname(long int *, long int **, long int *, const char *);

/* Read URI linking data */
int fread_urilinkdat_fname_mpf(MPFRSMatrix, const char *);

/* Select index of absolute maximum element and its value in MPFVector */
long int absmax_index_mpfvector(mpf_t, MPFVector);

/* Multiply MPFRSMatrix * MPFVector */
int mul_mpfrsmatrix_mpfvec(MPFVector, MPFRSMatrix, MPFVector);

/* Multiply MPFRSMatrix^T * MPFVector */
int mul_mpfrsmatrixt_mpfvec(MPFVector, MPFRSMatrix, MPFVector);

/*----------------------------------------------------------------*/
/* Ozaki-scheme routines for sparse mpf_t matrices                 */
/* The *_ex forms hand back the power of two each row (SplitMat_A) */
/* or column (SplitMat_B) was scaled by; see oz_scheme.h.          */
/*----------------------------------------------------------------*/
// SplitMat_A; row_shift holds num_div * row_dim exponents (may be NULL)
int split_mpfrsmatrix_drsmat_ex(DRSMatrix ret_mat[], long int row_shift[], int num_div, MPFRSMatrix org_mat);
int split_mpfrsmatrix_drsmat(DRSMatrix ret_mat[], int num_div, MPFRSMatrix org_mat);
// SplitMat_B; col_shift holds num_div * col_dim exponents (may be NULL)
int split_mpfrsmatrix_t_drsmat_ex(DRSMatrix ret_mat[], long int col_shift[], int num_div, MPFRSMatrix org_mat);
int split_mpfrsmatrix_t_drsmat(DRSMatrix ret_mat[], int num_div, MPFRSMatrix org_mat);
// Matrix-Vector multiplication based on Ozaki scheme
void mul_mpfrsmatrix_mpfvec_oz(MPFVector ret, MPFRSMatrix a, int max_num_div_a, MPFVector vb, int max_num_div_vb);
// Transposed Matrix-Vector multiplication based on Ozaki scheme
void mul_mpfrsmatrixt_mpfvec_oz(MPFVector ret, MPFRSMatrix a, int max_num_div_a, MPFVector vb, int max_num_div_vb);
// C := A(sparse) * B(dense) based on Ozaki scheme
void mul_mpfrsmatrix_mpfmat_oz(MPFMatrix ret, MPFRSMatrix a, int max_num_div_a, MPFMatrix b, int max_num_div_b);

/* Multiply DRSMatrix * MPFVector */
int mul_drsmatrix_mpfvec(MPFVector ret, DRSMatrix mat, MPFVector vec);

/* Multiply DRSMatrix^T * MPFVector */
int mul_drsmatrixt_mpfvec(MPFVector ret, DRSMatrix mat, MPFVector vec);

/* Power Method for Randomly Sparse Matrices */
/* 	double *evec: the eigenvector for max eigenvalue */
/* 	double *drsmat: Randomly sparse matrix */
/* 	double reps, aeps: Relative and Absolute tolerance */
/* 	long int max_times: Maximum iterative times of Power method */
void mpfpower_rsmatrix(mpf_t, MPFVector, MPFRSMatrix, mpf_t, mpf_t, long int);

/* Scalar multiply of MPFVector */
int smul_mpfvector(MPFVector, mpf_t, MPFVector);

/* Select index of absolute maximum element and its value in MPFVector */
long int absmax_index_mpfvector(mpf_t, MPFVector);

// Incomplete LU decomposition; iLU0_drsmatrix
void iLU0_mpfrsmatrix(MPFRSMatrix mat);

// iLU0_solve: iLU * x = b
void solve_iLU0_mpfrsmatrix(MPFVector ret, MPFRSMatrix ilu, MPFVector b);

// iLU0t_solve: x^t * (iLU) = b^t
void solve_iLU0t_mpfrsmatrix(MPFVector ret, MPFRSMatrix ilu, MPFVector b);

// iLU0_solve: iLU * x = b
void solve_iLU0_drsmatrix_mpfvec(MPFVector ret, DRSMatrix ilu, MPFVector b);

// iLU0t_solve: x^t(iLU) = b^t
void solve_iLU0t_drsmatrix_mpfvec(MPFVector ret, DRSMatrix ilu, MPFVector b);

// ---------
// Complex 
// ---------

/* Sparse matrix struct */
/* Example:             */
/*      0 1 2 3 4       */
/* A = [a 0 b c 0]0     */
/*     [0 d 0 0 0]1     */
/*     [0 e f 0 0]2     */
/*     [0 0 0 g 0]3     */
/*     [0 0 h 0 i]4     */
/*                      */
/* <--> element = [a b c d e f g h i] */
/*      row_dim = 5, col_dim = 5      */
/*      nzero_index[0] = [0 2 3]    */
/*      nzero_index[1] = [4]        */
/*      nzero_index[2] = [1 2]      */
/*      nzero_index[3] = [3]        */
/*      nzero_index[4] = [2 4]      */
/*      nzero_col_dim[0] = 3 */
/*      nzero_col_dim[1] = 1 */
/*      nzero_col_dim[2] = 2 */
/*      nzero_col_dim[3] = 1 */
/*      nzero_col_dim[4] = 2 */
/*      nzero_total_num =  9 */
/*      nzero_row_dim[0] = 1 */
/*      nzero_row_dim[1] = 2 */
/*      nzero_row_dim[2] = 3 */
/*      nzero_row_dim[3] = 2 */
/*      nzero_row_dim[4] = 1 */

typedef struct {
	unsigned long prec;
	mpc_t *element;				// Elements of matrix
	long int row_dim, col_dim;	// Dimensions of Row and Column
	long int **nzero_index;		// Indeces of Non-zero elements
	long int *nzero_col_dim;	// Numbers of non-zero elements in i-th row
	long int *nzero_row_dim;	// Numbers of non-zero elements in i-th column
	long int nzero_total_num;	// Total number of non-zero elements
	mpc_t zero_element;         // Zero value (Append in 2012-03-20)
} cmpfrsmatrix;

/* Definition of "CMPFRSMatrix" type */
typedef cmpfrsmatrix *CMPFRSMatrix;

/* initialize CMPFRSMatrix */
CMPFRSMatrix init_cmpfrsmatrix(long int, long int *, long int);

/* initialize CMPFRSMatrix with specific precision */
CMPFRSMatrix init2_cmpfrsmatrix(long int, long int *, long int, unsigned long prec);

/* Clear CMPFRSMatrix */
void free_cmpfrsmatrix(CMPFRSMatrix);

/* set nzero_row_dim automatically */
void set_nzero_row_dim_cmpf(CMPFRSMatrix);

/* Print CMPFRSMatrix */
void print_cmpfrsmatrix(CMPFRSMatrix);

// 2024-09-05(Thu) T.Kouya
// Frobenius norm of mat
void normf_cmpfrsmatrix(mpf_t ret, CMPFRSMatrix mat);

// 2024-12-04(Tue)
/* get the CMPFRSMatrix ij-element */
mpc_ptr get_cmpfrsmatrix_ij(CMPFRSMatrix mat, long int row_index, long int col_index);

// 2024-12-04 (Tue) T.Kouya
/* set the CMPFRSMatrix ij-element */
void set_cmpfrsmatrix_ij(CMPFRSMatrix mat, long int row_index, long int col_index, mpc_t val);

// 2024-12-17 (Tue) T.Kouya
// init and set CMPFRSMatrix
CMPFRSMatrix init_set_cmpfrsmatrix(CMPFRSMatrix org_sp);

/* Dense Matrix := Sparse Matrix */
void set_cmpfmatrix_cmpfrsmat(CMPFMatrix ret, CMPFRSMatrix spmat);

/* initialize and substitute CMPFRSMatrix from CMPFMatrix */
CMPFRSMatrix init_set_cmpfrsmatrix_cmpfmatrix(CMPFMatrix);

/* Get variables to initialize CMPFRSMatrix */
int get_vars_cmpfrsmatrix_fname(long int *, long int **, long int *, const char *);

/* Multiply CMPFRSMatrix * CMPFVector */
int mul_cmpfrsmatrix_cmpfvec(CMPFVector, CMPFRSMatrix, CMPFVector);

/* Multiply CMPFRSMatrix^T * CMPFVector */
int mul_cmpfrsmatrixt_cmpfvec(CMPFVector, CMPFRSMatrix, CMPFVector);

/* Multiply conj(CMPFRSMatrix)^T * CMPFVector */
int mul_cmpfrsmatrixs_cmpfvec(CMPFVector, CMPFRSMatrix, CMPFVector);

/* Multiply CDRSMatrix * CMPFVector */
int mul_cdrsmatrix_cmpfvec(CMPFVector ret, CDRSMatrix mat, CMPFVector vec);

/* Multiply CDRSMatrix^T * MPFVector */
int mul_cdrsmatrixt_cmpfvec(CMPFVector ret, CDRSMatrix mat, CMPFVector vec);

/* Multiply conj(CDRSMatrix)^T * MPFVector */
int mul_cdrsmatrixs_cmpfvec(CMPFVector ret, CDRSMatrix mat, CMPFVector vec);

// Imcomplete LU decomposition; iLU0_drsmatrix
void iLU0_cmpfrsmatrix(CMPFRSMatrix mat);

// iLU0_solve: iLU * x = b
void solve_iLU0_cmpfrsmatrix(CMPFVector ret, CMPFRSMatrix ilu, CMPFVector b);

// iLU0t_solve: x^t * (iLU) = b^t
void solve_iLU0t_cmpfrsmatrix(CMPFVector ret, CMPFRSMatrix ilu, CMPFVector b);

// iLU0s_solve: x^t * conj(iLU) = b^t
void solve_iLU0s_cmpfrsmatrix(CMPFVector ret, CMPFRSMatrix ilu, CMPFVector b);

// iLU0_solve: iLU * x = b
void solve_iLU0_cdrsmatrix_cmpfvec(CMPFVector ret, CDRSMatrix ilu, CMPFVector b);

// iLU0s_solve: x^t * iLU = b^t
void solve_iLU0t_cdrsmatrix_cmpfvec(CMPFVector ret, CDRSMatrix ilu, CMPFVector b);

// iLU0s_solve: x^t * conj(iLU) = b^t
void solve_iLU0s_cdrsmatrix_cmpfvec(CMPFVector ret, CDRSMatrix ilu, CMPFVector b);

// MPFRSMatrix -> D, DD, TD, QD
// MPFRSMatrix -> DRSMatrix
DRSMatrix init_set_drsmatrix_mpfrsmatrix(MPFRSMatrix org_sp);
// CMPFRSMatrix -> CDRSMatrix
CDRSMatrix init_set_cdrsmatrix_cmpfrsmatrix(CMPFRSMatrix org_sp);

/* initialize and substitute DRSMatrix from MPFRSMatrix */
DDRSMatrix init_set_ddrsmatrix_mpfrsmatrix(MPFRSMatrix org_sp);
CDDRSMatrix init_set_cddrsmatrix_cmpfrsmatrix(CMPFRSMatrix org_sp);

// MPFRSMatrix -> TDRSMatrix
TDRSMatrix init_set_tdrsmatrix_mpfrsmatrix(MPFRSMatrix org_sp);
CTDRSMatrix init_set_ctdrsmatrix_cmpfrsmatrix(CMPFRSMatrix org_sp);

// MPFRSMatrix -> QDRSMatrix
QDRSMatrix init_set_qdrsmatrix_mpfrsmatrix(MPFRSMatrix org_sp);
CQDRSMatrix init_set_cqdrsmatrix_cmpfrsmatrix(CMPFRSMatrix org_sp);

#endif // USE_GMP

/***********************************************/
// Conjugate-Gradient Method (Sparse & Dense Version)
/**********************************************/
long int FCG(FVector answer, FMatrix a, FVector b, float reps, float aeps, long int maxtimes);
long int DCG_sp(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes);
long int bnc_DCG(DVector answer, DMatrix a, DVector b, double reps, double aeps, long int maxtimes);
// Complex
long int CDCOCG_sp(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes);
long int CDCOCG(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes);

// DD
long int DDCG_sp(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int DDCG_sp_d(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int DDCG(DDVector answer, DDMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
// Complex
long int CDDCOCG_sp(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int CDDCOCG_sp_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int CDDCOCG(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);

// TD
long int TDCG_sp(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int TDCG_sp_d(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int TDCG(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
// Complex
long int CTDCOCG_sp(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int CTDCOCG_sp_d(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int CTDCOCG(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);

// QD
long int QDCG_sp(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int QDCG_sp_d(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int QDCG(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
// Complex
long int CQDCOCG_sp(CQDVector answer, CQDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int CQDCOCG_sp_d(CQDVector answer, CDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int CQDCOCG(CQDVector answer, CQDMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);

// MPFR
#ifdef USE_GMP
long int MPFCG_sp(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int MPFCG_sp_d(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int MPFCG(MPFVector answer, MPFMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
// Complex
long int CMPFCOCG_sp(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int CMPFCOCG_sp_d(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int CMPFCOCG(CMPFVector answer, CMPFMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
#endif // USE_GMP

/********************************************/
/* Krylov Subspace Methods (Sparse Version) */
/********************************************/
long int DBiCG_sp(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes);
long int DCGS_sp(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes);
long int DBiCGSTAB_sp(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes);
long int DGPBiCG_sp(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes);
long int DBiCG(DVector answer, DMatrix a, DVector b, double reps, double aeps, long int maxtimes);
long int DCGS(DVector answer, DMatrix a, DVector b, double reps, double aeps, long int maxtimes);
long int DBiCGSTAB(DVector answer, DMatrix a, DVector b, double reps, double aeps, long int maxtimes);
long int DGPBiCG(DVector answer, DMatrix a, DVector b, double reps, double aeps, long int maxtimes);

// Preconditioning
long int DBiCG_sp_iLU0(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int DCGS_sp_iLU0(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int DBiCGSTAB_sp_iLU0(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int DGPBiCG_sp_iLU0(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes, DRSMatrix ilu, double *norm2_res_history);

// Complex
long int CDBiCG_sp(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes);
long int CDCGS_sp(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes);
long int CDBiCGSTAB_sp(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes);
long int CDGPBiCG_sp(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes);

// Complex and Preconditioning
long int CDBiCG_sp_iLU0(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes, CDRSMatrix ilu, double *norm2_res_history);
long int CDCGS_sp_iLU0(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes, CDRSMatrix ilu, double *norm2_res_history);
long int CDBiCGSTAB_sp_iLU0(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes, CDRSMatrix ilu, double *norm2_res_history);
long int CDGPBiCG_sp_iLU0(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes, CDRSMatrix ilu, double *norm2_res_history);

long int CDBiCG(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes);
long int CDCGS(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes);
long int CDBiCGSTAB(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes);
long int CDGPBiCG(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes);

// DD
// -----------
// Real 
// -----------
// Sparse
long int DDBiCG_sp(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int DDCGS_sp(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int DDBiCGSTAB_sp(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int DDGPBiCG_sp(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);

// Sparse & Preconditioning
long int DDBiCG_sp_iLU0(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, DDRSMatrix ilu, double *norm2_res_history);
long int DDCGS_sp_iLU0(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, DDRSMatrix ilu, double *norm2_res_history);
long int DDBiCGSTAB_sp_iLU0(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, DDRSMatrix ilu, double *norm2_res_history);
long int DDGPBiCG_sp_iLU0(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, DDRSMatrix ilu, double *norm2_res_history);

// Double Sparse
long int DDBiCG_sp_d(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int DDCGS_sp_d(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int DDBiCGSTAB_sp_d(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int DDGPBiCG_sp_d(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);

// Double Sparse & Preconditioning
long int DDBiCG_sp_d_iLU0(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int DDCGS_sp_d_iLU0(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int DDBiCGSTAB_sp_d_iLU0(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int DDGPBiCG_sp_d_iLU0(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);

// Dense
long int DDBiCG(DDVector answer, DDMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int DDCGS(DDVector answer, DDMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int DDBiCGSTAB(DDVector answer, DDMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int DDGPBiCG(DDVector answer, DDMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);

// -----------
// Complex
// -----------
// Sparse
long int CDDBiCG_sp(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int CDDCGS_sp(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int CDDBiCGSTAB_sp(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int CDDGPBiCG_sp(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);

// Sparse + Preconditioning
long int CDDBiCG_sp_iLU0(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDRSMatrix ilu, double *norm2_res_history);
long int CDDCGS_sp_iLU0(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDRSMatrix ilu, double *norm2_res_history);
long int CDDBiCGSTAB_sp_iLU0(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDRSMatrix ilu, double *norm2_res_history);
long int CDDGPBiCG_sp_iLU0(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDRSMatrix ilu, double *norm2_res_history);

// Double Sparse
long int CDDBiCG_sp_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int CDDCGS_sp_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int CDDBiCGSTAB_sp_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int CDDGPBiCG_sp_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);

// Double Sparse + Preconditioning
long int CDDBiCG_sp_d_iLU0(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history);
long int CDDCGS_sp_d_iLU0(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history);
long int CDDBiCGSTAB_sp_d_iLU0(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history);
long int CDDGPBiCG_sp_d_iLU0(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history);

// Dense
long int CDDBiCG(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int CDDCGS(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int CDDBiCGSTAB(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int CDDGPBiCG(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);

// TD
// Sparse
long int TDBiCG_sp(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int TDCGS_sp(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int TDBiCGSTAB_sp(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int TDGPBiCG_sp(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);

// Sparse & Precondioning
long int TDBiCG_sp_iLU0(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDRSMatrix ilu, double *norm2_res_history);
long int TDCGS_sp_iLU0(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDRSMatrix ilu, double *norm2_res_history);
long int TDBiCGSTAB_sp_iLU0(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDRSMatrix ilu, double *norm2_res_history);
long int TDGPBiCG_sp_iLU0(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDRSMatrix ilu, double *norm2_res_history);

// Double Sparse
long int TDBiCG_sp_d(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int TDCGS_sp_d(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int TDBiCGSTAB_sp_d(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int TDGPBiCG_sp_d(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);

// Double Sparse and Preconditioning
long int TDBiCG_sp_d_iLU0(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int TDCGS_sp_d_iLU0(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int TDBiCGSTAB_sp_d_iLU0(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int TDGPBiCG_sp_d_iLU0(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);

// Dense
long int TDBiCG(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int TDCGS(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int TDBiCGSTAB(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int TDGPBiCG(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);

// -----------
// Complex
// -----------
// Sparse
long int CTDBiCG_sp		(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int CTDCGS_sp		(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int CTDBiCGSTAB_sp	(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int CTDGPBiCG_sp	(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);

// Sparse + Preconditioning
long int CTDBiCG_sp_iLU0	(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CTDRSMatrix ilu, double *norm2_res_history);
long int CTDCGS_sp_iLU0		(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CTDRSMatrix ilu, double *norm2_res_history);
long int CTDBiCGSTAB_sp_iLU0(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CTDRSMatrix ilu, double *norm2_res_history);
long int CTDGPBiCG_sp_iLU0	(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CTDRSMatrix ilu, double *norm2_res_history);

// Double Sparse
long int CTDBiCG_sp_d	 (CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int CTDCGS_sp_d	 (CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int CTDBiCGSTAB_sp_d(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int CTDGPBiCG_sp_d	 (CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);

// Double Sparse + Preconditioning
long int CTDBiCG_sp_d_iLU0		(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history);
long int CTDCGS_sp_d_iLU0		(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history);
long int CTDBiCGSTAB_sp_d_iLU0	(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history);
long int CTDGPBiCG_sp_d_iLU0	(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history);

// Dense
long int CTDBiCG		(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int CTDCGS			(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int CTDBiCGSTAB	(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int CTDGPBiCG		(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);

// QD
// Sparse
long int QDBiCG_sp(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int QDCGS_sp(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int QDBiCGSTAB_sp(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int QDGPBiCG_sp(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);

// Sparse and Preconditioning
long int QDBiCG_sp_iLU0(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDRSMatrix ilu, double *norm2_res_history);
long int QDCGS_sp_iLU0(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDRSMatrix ilu, double *norm2_res_history);
long int QDBiCGSTAB_sp_iLU0(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDRSMatrix ilu, double *norm2_res_history);
long int QDGPBiCG_sp_iLU0(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDRSMatrix ilu, double *norm2_res_history);

// Double Sparse
long int QDBiCG_sp_d(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int QDCGS_sp_d(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int QDBiCGSTAB_sp_d(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int QDGPBiCG_sp_d(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);

// Double Sparse and Preconditioning
long int QDBiCG_sp_d_iLU0(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int QDCGS_sp_d_iLU0(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int QDBiCGSTAB_sp_d_iLU0(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int QDGPBiCG_sp_d_iLU0(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);

// Dense
long int QDBiCG(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int QDCGS(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int QDBiCGSTAB(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int QDGPBiCG(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);

// -----------
// Complex
// -----------
// Sparse
long int CQDBiCG_sp		(CQDVector answer, CQDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int CQDCGS_sp		(CQDVector answer, CQDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int CQDBiCGSTAB_sp	(CQDVector answer, CQDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int CQDGPBiCG_sp	(CQDVector answer, CQDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);

// Sparse + Preconditioning
long int CQDBiCG_sp_iLU0		(CQDVector answer, CQDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, CQDRSMatrix ilu, double *norm2_res_history);
long int CQDCGS_sp_iLU0			(CQDVector answer, CQDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, CQDRSMatrix ilu, double *norm2_res_history);
long int CQDBiCGSTAB_sp_iLU0	(CQDVector answer, CQDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, CQDRSMatrix ilu, double *norm2_res_history);
long int CQDGPBiCG_sp_iLU0		(CQDVector answer, CQDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, CQDRSMatrix ilu, double *norm2_res_history);

// Double Sparse
long int CQDBiCG_sp_d	 (CQDVector answer, CDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int CQDCGS_sp_d	 (CQDVector answer, CDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int CQDBiCGSTAB_sp_d(CQDVector answer, CDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int CQDGPBiCG_sp_d	 (CQDVector answer, CDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);

// Double Sparse
long int CQDBiCG_sp_d_iLU0	 	(CQDVector answer, CDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history);
long int CQDCGS_sp_d_iLU0	 	(CQDVector answer, CDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history);
long int CQDBiCGSTAB_sp_d_iLU0	(CQDVector answer, CDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history);
long int CQDGPBiCG_sp_d_iLU0	(CQDVector answer, CDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history);

// Dense
long int CQDBiCG		(CQDVector answer, CQDMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int CQDCGS			(CQDVector answer, CQDMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int CQDBiCGSTAB	(CQDVector answer, CQDMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int CQDGPBiCG		(CQDVector answer, CQDMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);


#ifdef USE_GMP
// ----------
// Real
// ----------
// SPARSE
long int MPFBiCG_sp(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int MPFCGS_sp(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int MPFBiCGSTAB_sp(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int MPFGPBiCG_sp(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes); 

// Preconditioning
long int MPFBiCG_sp_iLU0(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, MPFRSMatrix ilu, MPFVector norm2_res_history);
long int MPFCGS_sp_iLU0(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, MPFRSMatrix ilu, MPFVector norm2_res_history);
long int MPFBiCGSTAB_sp_iLU0(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, MPFRSMatrix ilu, MPFVector norm2_res_history);
long int MPFGPBiCG_sp_iLU0(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, MPFRSMatrix ilu, MPFVector norm2_res_history);

// Double Sparse
long int MPFBiCG_sp_d(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int MPFCGS_sp_d(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int MPFBiCGSTAB_sp_d(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int MPFGPBiCG_sp_d(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes); 

// Double Sparse and Preconditioning
long int MPFBiCG_sp_d_iLU0(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, DRSMatrix ilu, MPFVector norm2_res_history);
long int MPFCGS_sp_d_iLU0(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, DRSMatrix ilu, MPFVector norm2_res_history);
long int MPFBiCGSTAB_sp_d_iLU0(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, DRSMatrix ilu, MPFVector norm2_res_history);
long int MPFGPBiCG_sp_d_iLU0(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, DRSMatrix ilu, MPFVector norm2_res_history);

// Dense
long int MPFBiCG(MPFVector answer, MPFMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int MPFCGS(MPFVector answer, MPFMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int MPFBiCGSTAB(MPFVector answer, MPFMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int MPFGPBiCG(MPFVector answer, MPFMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);

// ----------
// Complex
// ----------
// SPARSE
long int CMPFBiCG_sp(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int CMPFCGS_sp(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int CMPFBiCGSTAB_sp(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int CMPFGPBiCG_sp(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);

// SPARSE + Preconditioning
long int CMPFBiCG_sp_iLU0		(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CMPFRSMatrix ilu, MPFVector norm2_res_history);
long int CMPFCGS_sp_iLU0		(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CMPFRSMatrix ilu, MPFVector norm2_res_history);
long int CMPFBiCGSTAB_sp_iLU0	(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CMPFRSMatrix ilu, MPFVector norm2_res_history);
long int CMPFGPBiCG_sp_iLU0		(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CMPFRSMatrix ilu, MPFVector norm2_res_history); 

// Double Sparse
long int CMPFBiCG_sp_d(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int CMPFCGS_sp_d(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int CMPFBiCGSTAB_sp_d(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int CMPFGPBiCG_sp_d(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes); 

// Double Sparse + Preconditioning
long int CMPFBiCG_sp_d_iLU0		(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CDRSMatrix ilu, MPFVector norm2_res_history);
long int CMPFCGS_sp_d_iLU0		(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CDRSMatrix ilu, MPFVector norm2_res_history);
long int CMPFBiCGSTAB_sp_d_iLU0	(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CDRSMatrix ilu, MPFVector norm2_res_history);
long int CMPFGPBiCG_sp_d_iLU0	(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CDRSMatrix ilu, MPFVector norm2_res_history); 

// Dense
long int CMPFBiCG(CMPFVector answer, CMPFMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int CMPFCGS(CMPFVector answer, CMPFMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int CMPFBiCGSTAB(CMPFVector answer, CMPFMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int CMPFGPBiCG(CMPFVector answer, CMPFMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
#endif // USE_GMP

// -----------
// lanczos.c
// -----------
long int dslancsoz_sp(DMatrix trimat, DVector *qvec, DRSMatrix mat_sp);
long int dslanczos(DMatrix trimat, DVector *qvec, DMatrix mat);
long int dlanczos_sp(DMatrix trimat, DVector *qvec, DVector *pvec, DRSMatrix mat_sp);
long int dlanczos(DMatrix trimat, DVector *qvec, DVector *pvec, DMatrix mat);

#ifdef USE_GMP
long int mpf_slanczos_sp(MPFMatrix trimat, MPFVector *qvec, MPFRSMatrix mat_sp);
long int mpf_slanczos(MPFMatrix trimat, MPFVector *qvec, MPFMatrix mat);
long int mpf_lanczos_sp(MPFMatrix trimat, MPFVector *qvec, MPFVector *pvec, MPFRSMatrix mat_sp);
long int mpf_lanczos(MPFMatrix trimat, MPFVector *qvec, MPFVector *pvec, MPFMatrix mat);
#endif // USE_GMP

// =====================================================================
// Float-based real sparse matrices (single / double-single / triple-single
// / quad-single) and SpMV.  Added 2026-06-15 (T.Kouya).
// SpMV implementations: scalar + AVX2 + AVX-512 + NEON + SVE2(=NEON).
// =====================================================================
#include "flinear.h"
typedef struct {
	float *element;
	long int row_dim, col_dim;
	long int **nzero_index;
	long int *nzero_col_dim;
	long int *nzero_row_dim;
	long int nzero_total_num;
	long int real_nzero_total_num;
	long int *real_nzero_col_dim;
} frsmatrix;
typedef frsmatrix *FRSMatrix;
FRSMatrix init_frsmatrix(long int, long int *, long int);
void free_frsmatrix(FRSMatrix);
void set0_frsmatrix(FRSMatrix);
void set_nzero_row_dim_f(FRSMatrix);
void set_frsmatrix_ij(FRSMatrix, long int, long int, float);
float get_frsmatrix_ij(FRSMatrix, long int, long int);
FRSMatrix init_set_frsmatrix_fmatrix(FMatrix);
int mul_frsmatrix_fvec(FVector, FRSMatrix, FVector);
int mul_frsmatrixt_fvec(FVector, FRSMatrix, FVector);

#include "dslinear.h"
typedef struct {
	float *element[DSSIZE];
	long int row_dim, col_dim;
	long int **nzero_index;
	long int *nzero_col_dim;
	long int *nzero_row_dim;
	long int nzero_total_num;
	long int real_nzero_total_num;
	long int *real_nzero_col_dim;
} dsrsmatrix;
typedef dsrsmatrix *DSRSMatrix;
DSRSMatrix init_dsrsmatrix(long int, long int *, long int);
void free_dsrsmatrix(DSRSMatrix);
void set0_dsrsmatrix(DSRSMatrix);
void set_nzero_row_dim_ds(DSRSMatrix);
void set_dsrsmatrix_ij(DSRSMatrix, long int, long int, float [DSSIZE]);
void get_dsrsmatrix_ij(float [DSSIZE], DSRSMatrix, long int, long int);
DSRSMatrix init_set_dsrsmatrix_dsmatrix(DSMatrix);
int mul_dsrsmatrix_dsvec(DSVector, DSRSMatrix, DSVector);
int mul_dsrsmatrixt_dsvec(DSVector, DSRSMatrix, DSVector);
/* complex double-single sparse: re/im pair of DSRSMatrix (SpMV reuses the real DS SpMV) */
#include "cdslinear.h"
typedef struct { DSRSMatrix re; DSRSMatrix im; } cdsrsmatrix;
typedef cdsrsmatrix *CDSRSMatrix;
CDSRSMatrix init_cdsrsmatrix(long int, long int *, long int);
void free_cdsrsmatrix(CDSRSMatrix);
void set0_cdsrsmatrix(CDSRSMatrix);
void set_cdsrsmatrix_ij(CDSRSMatrix, long int, long int, cdsfloat *);
void get_cdsrsmatrix_ij(cdsfloat *, CDSRSMatrix, long int, long int);
CDSRSMatrix init_set_cdsrsmatrix_cdsmatrix(CDSMatrix);
int mul_cdsrsmatrix_cdsvec(CDSVector, CDSRSMatrix, CDSVector);
int mul_cdsrsmatrixt_cdsvec(CDSVector, CDSRSMatrix, CDSVector);

#include "tslinear.h"
typedef struct {
	float *element[TSSIZE];
	long int row_dim, col_dim;
	long int **nzero_index;
	long int *nzero_col_dim;
	long int *nzero_row_dim;
	long int nzero_total_num;
	long int real_nzero_total_num;
	long int *real_nzero_col_dim;
} tsrsmatrix;
typedef tsrsmatrix *TSRSMatrix;
TSRSMatrix init_tsrsmatrix(long int, long int *, long int);
void free_tsrsmatrix(TSRSMatrix);
void set0_tsrsmatrix(TSRSMatrix);
void set_nzero_row_dim_ts(TSRSMatrix);
void set_tsrsmatrix_ij(TSRSMatrix, long int, long int, float [TSSIZE]);
void get_tsrsmatrix_ij(float [TSSIZE], TSRSMatrix, long int, long int);
TSRSMatrix init_set_tsrsmatrix_tsmatrix(TSMatrix);
int mul_tsrsmatrix_tsvec(TSVector, TSRSMatrix, TSVector);
int mul_tsrsmatrixt_tsvec(TSVector, TSRSMatrix, TSVector);
/* complex triple-single sparse: re/im pair of TSRSMatrix */
#include "ctslinear.h"
typedef struct { TSRSMatrix re; TSRSMatrix im; } ctsrsmatrix;
typedef ctsrsmatrix *CTSRSMatrix;
CTSRSMatrix init_ctsrsmatrix(long int, long int *, long int);
void free_ctsrsmatrix(CTSRSMatrix);
void set0_ctsrsmatrix(CTSRSMatrix);
void set_ctsrsmatrix_ij(CTSRSMatrix, long int, long int, ctsfloat *);
void get_ctsrsmatrix_ij(ctsfloat *, CTSRSMatrix, long int, long int);
CTSRSMatrix init_set_ctsrsmatrix_ctsmatrix(CTSMatrix);
int mul_ctsrsmatrix_ctsvec(CTSVector, CTSRSMatrix, CTSVector);
int mul_ctsrsmatrixt_ctsvec(CTSVector, CTSRSMatrix, CTSVector);

#include "qslinear.h"
typedef struct {
	float *element[QSSIZE];
	long int row_dim, col_dim;
	long int **nzero_index;
	long int *nzero_col_dim;
	long int *nzero_row_dim;
	long int nzero_total_num;
	long int real_nzero_total_num;
	long int *real_nzero_col_dim;
} qsrsmatrix;
typedef qsrsmatrix *QSRSMatrix;
QSRSMatrix init_qsrsmatrix(long int, long int *, long int);
void free_qsrsmatrix(QSRSMatrix);
void set0_qsrsmatrix(QSRSMatrix);
void set_nzero_row_dim_qs(QSRSMatrix);
void set_qsrsmatrix_ij(QSRSMatrix, long int, long int, float [QSSIZE]);
void get_qsrsmatrix_ij(float [QSSIZE], QSRSMatrix, long int, long int);
QSRSMatrix init_set_qsrsmatrix_qsmatrix(QSMatrix);
int mul_qsrsmatrix_qsvec(QSVector, QSRSMatrix, QSVector);
int mul_qsrsmatrixt_qsvec(QSVector, QSRSMatrix, QSVector);
/* complex quad-single sparse: re/im pair of QSRSMatrix */
#include "cqslinear.h"
typedef struct { QSRSMatrix re; QSRSMatrix im; } cqsrsmatrix;
typedef cqsrsmatrix *CQSRSMatrix;
CQSRSMatrix init_cqsrsmatrix(long int, long int *, long int);
void free_cqsrsmatrix(CQSRSMatrix);
void set0_cqsrsmatrix(CQSRSMatrix);
void set_cqsrsmatrix_ij(CQSRSMatrix, long int, long int, cqsfloat *);
void get_cqsrsmatrix_ij(cqsfloat *, CQSRSMatrix, long int, long int);
CQSRSMatrix init_set_cqsrsmatrix_cqsmatrix(CQSMatrix);
int mul_cqsrsmatrix_cqsvec(CQSVector, CQSRSMatrix, CQSVector);
int mul_cqsrsmatrixt_cqsvec(CQSVector, CQSRSMatrix, CQSVector);

/* End of __BNC_SPARSE_H__ */
#endif // __BNC_SPARSE_H__
