/********************************************************************************/
/* cmpclinear.c: MPC-based Vector, Matrix                                       */
/* Copyright (c) 2024 Tomonori Kouya                                            */
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
// define __BNC_CLINEAR_H__
#ifndef __BNC_CMPFLINEAR_H__
  #define __BNC_CMPFLINEAR_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h> // float _Complex & double _Complex

// Common defs
#include "bnc_common.h" // [F,D,MPF]Complx

// mpc_t
#include "mpc.h"
#define mpc_init(z) mpc_init2((z), (get_bnc_default_prec()))

//#include "bnc.h"
#include "dlinear.h"
#include "flinear.h"
#include "cdlinear.h"

#ifdef USE_GMP
#include "mpflinear.h"
#endif // USE_GMP

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#ifdef USE_GMP
// CMPFVector: multiple precision complex vector
typedef struct{
	unsigned long int prec;
#ifdef USE_MPFCMPLX
	mpfcmplx *element;
#else // USE_MPFCMPLX
	mpc_t *element;
#endif // USE_MPFCMPLX
	long int dim;
} cmpfvector;

typedef cmpfvector *CMPFVector;

// CMPFMatrix: multiple precision complex matrix
typedef struct{
	unsigned long int prec;
#ifdef USE_MPFCMPLX
	mpfcmplx *element;
#else // USE_MPFCMPLX
	mpc_t *element;
#endif // USE_MPFCMPLX
	long int row_dim, col_dim;
} cmpfmatrix;

typedef cmpfmatrix *CMPFMatrix;
#endif // USE_GMP

#ifdef USE_GMP
// ret := a * b
void _bnc_mpc_mul_cd(mpc_t ret, mpc_t a, double _Complex b);

CMPFVector init_cmpfvector(long int);
CMPFVector init2_cmpfvector(long int, unsigned long int);
void free_cmpfvector(CMPFVector);
void print_cmpfmatrix(CMPFMatrix);
#ifdef USE_MPFCMPLX
MPFCmplx get_cmpfvector_i(CMPFVector, long int);
MPFCmplx get_cmpfvector_i(CMPFVector, long int);
void set_cmpfvector_i(CMPFVector, long int, MPFCmplx);
#else // USE_MPFCMPLX
mpc_ptr get_cmpfvector_i(CMPFVector, long int);
mpc_ptr get_cmpfvector_i(CMPFVector, long int);
void set_cmpfvector_i(CMPFVector, long int, mpc_t);
#endif // USE_MPFCMPLX
void set_cmpfvector_i_real(CMPFVector vec, long int index, mpf_t val);
void set_cmpfvector_i_image(CMPFVector vec, long int index, mpf_t val);
#define set_cmpfvector_i_re set_cmpfvector_i_real
#define set_cmpfvector_i_im set_cmpfvector_i_image
void set_cmpfvector_i_d(CMPFVector, long int, double _Complex);
void set_cmpfvector_i_str(CMPFVector, long int, const char *, const char *, int);
void set_cmpfvector_i_ui(CMPFVector, long int, unsigned long);
unsigned long int prec_cmpfvector(CMPFVector);
unsigned long int get_prec_cmpfvector(CMPFVector);
unsigned long int minprec_cmpfvector(CMPFVector);
unsigned long int maxprec_cmpfvector(CMPFVector);
void print_cmpfvector(CMPFVector);
void add_cmpfvector(CMPFVector, CMPFVector, CMPFVector);
void add2_cmpfvector(CMPFVector, CMPFVector);
void sub_cmpfvector(CMPFVector, CMPFVector, CMPFVector);
void sub2_cmpfvector(CMPFVector, CMPFVector);
#ifdef USE_MPFCMPLX
void cmul_cmpfvector(CMPFVector, MPFCmplx, CMPFVector);
void cmul_cmpfvector_mpfvec(CMPFVector, MPFCmplx, MPFVector);
void cmul2_cmpfvector(CMPFVector, MPFCmplx);
void add_cmul_cmpfvector(CMPFVector, CMPFVector, MPFCmplx, CMPFVector);
void ip_cmpfvector(MPFCmplx, CMPFVector, CMPFVector);
#else // USE_MPFCMPLX
void cmul_cmpfvector(CMPFVector, mpc_t, CMPFVector);
void cmul_cmpfvector_4m(CMPFVector c, mpc_t val, CMPFVector a);
void cmul_cmpfvector_mpfvec(CMPFVector, mpc_t, MPFVector);
void cmul2_cmpfvector(CMPFVector, mpc_t);
void add_cmul_cmpfvector(CMPFVector, CMPFVector, mpc_t, CMPFVector);
void sub_cmul_cmpfvector(CMPFVector, CMPFVector, mpc_t, CMPFVector);
void ip_cmpfvector(mpc_t, CMPFVector, CMPFVector);
void dotp_cmpfvector(mpc_t, CMPFVector, CMPFVector);
#endif // USE_MPFCMPLX
void norm1_cmpfvector(mpf_t, CMPFVector);
void subst_cmpfvector(CMPFVector, CMPFVector);
void conj_cmpfvector(CMPFVector, CMPFVector);
void norm2_cmpfvector(mpf_t, CMPFVector);
void normi_cmpfvector(mpf_t, CMPFVector);
void subst_cmpfvector_mpfvec(CMPFVector, MPFVector);
void subst_mpfvector_image_cmpfvec(MPFVector, CMPFVector);
void subst_mpfvector_real_cmpfvec(MPFVector, CMPFVector);
void set0_cmpfvector(CMPFVector);
void copy_cmpfvector_ij(CMPFVector, long int, long int, CMPFVector, long int, long int);
// ret_real + ret_image * I := src
void separate_cmpfvector(MPFVector, MPFVector, CMPFVector);
// ret := src_real + src_image * I
void merge_cmpfvector(CMPFVector, MPFVector, MPFVector);

CMPFMatrix init_cmpfmatrix(long int, long int);
CMPFMatrix init2_cmpfmatrix(long int, long int, unsigned long);
void free_cmpfmatrix(CMPFMatrix);
#ifdef USE_MPFCMPLX
MPFCmplx get_cmpfmatrix_ij(CMPFMatrix, long int, long int);
void set_cmpfmatrix_ij(CMPFMatrix, long int, long int, MPFCmplx);
#else // USE_MPFCMPLX
mpc_ptr get_cmpfmatrix_ij(CMPFMatrix, long int, long int);
void set_cmpfmatrix_ij(CMPFMatrix, long int, long int, mpc_t);
#endif // USE_MPFCMPLX
void set_cmpfmatrix_ij_d(CMPFMatrix, long int, long int, double _Complex);
void set_cmpfmatrix_ij_str(CMPFMatrix, long int, long int, const char *, const char *, int);
void set_cmpfmatrix_ij_ui(CMPFMatrix, long int, long int, unsigned long );
unsigned long int maxprec_cmpfmatrix(CMPFMatrix);
unsigned long int get_prec_cmpfmatrix(CMPFMatrix);
unsigned long int prec_cmpfmatrix(CMPFMatrix);
unsigned long int minprec_cmpfmatrix(CMPFMatrix);
void print_cmpfmatrix(CMPFMatrix);
void normf_cmpfmatrix(mpf_t, CMPFMatrix);
void normi_cmpfmatrix(mpf_t, CMPFMatrix);
void norm1_cmpfmatrix(mpf_t, CMPFMatrix);
void add_cmpfmatrix(CMPFMatrix, CMPFMatrix, CMPFMatrix);
void sub_cmpfmatrix(CMPFMatrix, CMPFMatrix, CMPFMatrix);
#ifdef USE_MPFCMPLX
void cmul_cmpfmatrix(CMPFMatrix, MPFCmplx, CMPFMatrix);
#else // USE_MPFCMPLX
void cmul_cmpfmatrix(CMPFMatrix, mpc_t, CMPFMatrix);
#endif // USE_MPFCMPLX
void mul_cmpfmatrix(CMPFMatrix, CMPFMatrix, CMPFMatrix);
void transpose_cmpfmatrix(CMPFMatrix, CMPFMatrix);
void star_cmpfmatrix(CMPFMatrix, CMPFMatrix);
void subst_cmpfmatrix(CMPFMatrix, CMPFMatrix);
void conj_cmpfmatrix(CMPFMatrix, CMPFMatrix);
void neg_cmpfmatrix(CMPFMatrix, CMPFMatrix);
void subst_cmpfmatrix_mpfmat(CMPFMatrix, MPFMatrix);
void set0_cmpfmatrix(CMPFMatrix);
void setI_cmpfmatrix(CMPFMatrix);
void mul_cmpfmatrix_cmpfvec(CMPFVector, CMPFMatrix, CMPFVector);
void mul_cmpfmatrix_cmpfvec_4m(CMPFVector, CMPFMatrix, CMPFVector);
void mul_cmpfmatrixt_cmpfvec(CMPFVector, CMPFMatrix, CMPFVector);

/* v := conj(a)^T * vb */
void mul_cmpfmatrixs_cmpfvec(CMPFVector v, CMPFMatrix a, CMPFVector vb);
void inv_cmpfmatrix(CMPFMatrix);
/* append 2023-02-26 */
// ret_real + ret_image * I := src
void separate_cmpfmatrix(MPFMatrix, MPFMatrix, CMPFMatrix);
// ret := src_real + src_image * I
void merge_cmpfmatrix(CMPFMatrix, MPFMatrix, MPFMatrix);
/* vec := -vec */
void neg_cmpfvector(CMPFVector vec);
/* Normwise relative error of vector */
void relerr_cmpfvector(mpf_t relerr, CMPFVector approx_vec, CMPFVector true_vec, int norm_type);
/* Elementwise relative errors of vector */
void relerr_element_cmpfvector(mpf_t max_relerr, mpf_t min_relerr, mpf_t norm_relerr, 
CMPFVector approx_vec, CMPFVector true_vec, int norm_type);
// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_cmpfmatrix(CMPFMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

// 2024-08-02 (Fri) T.Kouya
// ret := (cmpf)a
void subst_cmpfvector_cdvec(CMPFVector ret, CDVector a);

// 2024-08-02 (Fri) T.Kouya
// ret := (cmpf)a
void subst_cmpfmatrix_cdmat(CMPFMatrix ret, CDMatrix a);

// 2026-02-03(Tue) T.Kouya
// absmax_cmpfvector
void absmax_cmpfvector(mpf_t ret, long int *max_index, CMPFVector vec);

// 2026-02-03(Tue) T.Kouya
// absmax_row_cmpfmatrix
void absmax_row_cmpfmatrix(mpf_t ret, long int *max_j, long int row_index, CMPFMatrix mat);

// 2022-11-17(Thu) T.Kouya
// absmax_col_cmpfmatrix
void absmax_col_cpfmatrix(mpf_t ret, long int *max_i, long int col_index, CMPFMatrix mat);

// 2026-02-03(Tue) T.Kouya
// absmax_cmpfmatrix
void absmax_cmpfmatrix(mpf_t ret, long int *max_i, long int *max_j, CMPFMatrix mat);

#endif // USE_GMP


/**********************************************/
/* cmpflu.c : LU decomposition of complex matrix */
/**********************************************/
#ifdef USE_GMP
int CMPFLUdecomp(CMPFMatrix);
int SolveCMPFLS(CMPFVector, CMPFMatrix, CMPFVector);
int CMPFLUdecompP(CMPFMatrix, long int[]);
int SolveCMPFLSP(CMPFVector, CMPFMatrix, CMPFVector, long int[]);
int CMPFLUdecompC(CMPFMatrix, long int[], long int[]);
int SolveCMPFLSC(CMPFVector, CMPFMatrix, CMPFVector, long int[], long int[]);
int CMPFLUdecompPM(CMPFMatrix a, long int ch[]);
int SolveCMPFLSPM(CMPFVector answer, CMPFMatrix lu, CMPFVector b, long int ch[]);
#endif // USE_GMP


/**********************************************/
/* cmpflu_strassen.c : LU decomposition of    */
/* complex matrix using Strassen algorithm    */
/**********************************************/
#ifdef USE_GMP
// (1) L11 * U11 = A11
int CMPFLUdecomp_square(CMPFMatrix a, long int start_index, long int min_dim);
// (2) Solve L21 * U11 = A21
int CMPFLUdecomp_l21(CMPFMatrix l21, CMPFMatrix a, long int start_index, long int min_dim);
// (3) Solve L11 * U12 = A21
int CMPFLUdecomp_u12(CMPFMatrix u12, CMPFMatrix a, long int start_index, long int min_dim);
// (4) D22 := L21 * U12
// (5) A22 := A22 - D22
int CMPFLUdecomp_a22(CMPFMatrix a, CMPFMatrix d22, CMPFMatrix l21, CMPFMatrix u12, long int start_index, long int min_dim);
int CMPFLUdecomp_strassen(CMPFMatrix a, long int min_dim);
int CMPFLUdecomp_strassenPM(CMPFMatrix a, long int ch[], long int min_dim);

#ifdef _OPENMP
int CMPFLUdecomp_square_omp(CMPFMatrix a, long int start_index, long int min_dim);
int CMPFLUdecomp_l21_omp(CMPFMatrix l21, CMPFMatrix a, long int start_index, long int min_dim);
int CMPFLUdecomp_u12_omp(CMPFMatrix u12, CMPFMatrix a, long int start_index, long int min_dim);
int CMPFLUdecomp_a22_omp(CMPFMatrix a, CMPFMatrix d22, CMPFMatrix l21, CMPFMatrix u12, long int start_index, long int min_dim);
int CMPFLUdecomp_strassen_omp(CMPFMatrix a, long int min_dim);
int CMPFLUdecomp_strassenPM_omp(CMPFMatrix a, long int ch[], long int min_dim);
int CMPFLUdecomp_omp(CMPFMatrix a);
int CMPFLUdecompPM_omp(CMPFMatrix a, long int ch[]); // 2024-02-13 (Mon) T.Kouya
#endif // _OPENMP

//----------------------------------/
// mpflu_oz.c
//----------------------------------/
int CMPFLUdecomp_a22_oz(CMPFMatrix a, CMPFMatrix d22, CMPFMatrix l21, CMPFMatrix u12, long int start_index, long int min_dim, int max_num_div);
int CMPFLUdecomp_oz(CMPFMatrix a, long int min_dim, int max_num_div);
int CMPFLUdecomp_ozPM(CMPFMatrix a, long int ch[], long int min_dim, int max_num_div);
#ifdef _OPENMP
int CMPFLUdecomp_a22_oz_omp(CMPFMatrix a, CMPFMatrix d22, CMPFMatrix l21, CMPFMatrix u12, long int start_index, long int min_dim, int max_num_div);
int CMPFLUdecomp_oz_omp(CMPFMatrix a, long int min_dim, int max_num_div);
int CMPFLUdecomp_ozPM_omp(CMPFMatrix a, long int ch[], long int min_dim, int max_num_div);
#endif // _OPENMP

//----------------------------------/
// fread_write.c
//----------------------------------/
void fread_cmpfmatrix(FILE *fp, CMPFMatrix mat);
void fread_cmpfmatrix_fname(const char *fname, CMPFMatrix mat);
void fwrite_cmpfmatrix(FILE *fp, CMPFMatrix mat);
void fwrite_cmpfmatrix_fname(const char *fname, CMPFMatrix mat);
void fread_cmpfvector(FILE *fp, CMPFVector vec);
void fread_cmpfvector_fname(const char *fname, CMPFVector vec);
void fwrite_cmpfvector(FILE *fp, CMPFVector vec);
void fwrite_cmpfvector_fname(const char *fname, CMPFVector vec);
void read_test_linear_eq_c(CMPFMatrix A, CMPFVector true_x, CMPFVector b, long int dim, const char *fname_A, const char *fname_true_x, const char *fname_b);
void write_test_linear_eq_c(CMPFMatrix A, CMPFVector true_x, CMPFVector b, long int dim, const char *fname_A, const char *fname_true_x, const char *fname_b);

#endif // USE_GMP

#ifdef __cplusplus
} //extern "C" {
#endif // __cplusplus

// define __BNC_CMPFLINEAR_H__
#endif // ifndef __BNC_CMPFLINEAR_H__
