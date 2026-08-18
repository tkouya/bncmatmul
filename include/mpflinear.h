/********************************************************************************/
/* mpflinear.c: Arbitrary precision Vector and Matrix computing                 */
/* Copyright (c) 2000-2011 Tomonori Kouya                                       */
/*                                                                              */
/* Version 0.1, 2005.07/11: append copy_*vector_ij                              */
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
// define __BNC_MPFLINEAR_H__
#ifndef __BNC_MPFLINEAR_H__
  #define __BNC_MPFLINEAR_H__

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Common defs
#include "bnc_common.h"

//#include "bnc.h"
#include "dlinear.h"
#include "flinear.h"

#ifdef USE_GMP
#include "gmp.h"
#ifdef USE_MPFR
#include "mpfr.h"
#include "mpf2mpfr.h" // mpf_t as mpfr_t 

// Original defines
#include "mpfr_dtq_sd.h"
#include "mpf_func_mpfr.h" // mpf_* functions from mpfr.h
//static mpfr_rnd_t bnc_default_rounding_mode = GMP_RNDN; // round to nearest
#endif // USE_MPFR
#endif // USE_GMP

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/*************************************************/
/* Functions for Vector Types                    */
/*                                               */
/* Initialize:                                   */
/*   MPFVector init_mpfvector(long int dimension)*/
/*   MPFVector init2_mpfvector(long int dimension, unsigned long mbits)*/
/* Free:                                         */
/*   void free_mpfvector(MPFVector vec)          */
/* Get & Set Values:                             */
/*   mpf_t *get_mpfvector_i(MPFVector vec, long int index) */
/*   void set_mpfvector_i(MPFVector vec, long int index, mpf_t val) */
/*   void set_mpfvector_i_d(MPFVector vec, long int index, double val) */
/* Output:                                       */
/*   void print_mpfvector(MPFVector vec)         */
/*   void print_fdmpfvector(FVector fv, DVector dv, MPFVector mpfv) */
/*************************************************/
#ifdef USE_GMP

typedef struct{
	unsigned long int prec;
	mpf_t *element;
//	mpf_ptr *element;
	long int dim;
	long int real_dim; // append in 2022-11-18(Fri) T.Kouya
} mpfvector;

typedef mpfvector *MPFVector;
MPFVector init_mpfvector(long int dimension);

/* mbits ... A number of at least bits of mantissa */
MPFVector init2_mpfvector(long int dimension, unsigned long int mbits);
#endif // USE_GMP

#ifdef USE_GMP
void free_mpfvector(MPFVector vec);
#endif // USE_GMP

#ifdef USE_GMP
#ifndef GET_VECTOR_I
mpf_ptr get_mpfvector_i(MPFVector vec, long int index);
#endif // GET_VECTOR_I
#define gmpfvi get_mpfvector_i // used in old BNClibrary
#endif // USE_GMP

#ifdef USE_GMP
void set_mpfvector_i(MPFVector vec, long int index, mpf_t val);
void set_mpfvector_i_d(MPFVector vec, long int index, double val);
#define smpfvi set_mpfvector_i // used in old BNClibrary

void set_mpfvector_i_str(MPFVector vec, long int index, const char *str, int base);
void set_mpfvector_i_ui(MPFVector vec, long int index, unsigned long val);

/* get precision of MPFVector */
unsigned long int prec_mpfvector(MPFVector vec);

/* search minimam precision in MPFVector */
unsigned long int minprec_mpfvector(MPFVector vec);

/* search maximam precision in MPFVector */
unsigned long int maxprec_mpfvector(MPFVector vec);
#endif // USE_GMP

#ifdef USE_GMP
void print_mpfvector(MPFVector vec);
#endif // USE_GMP

/*************************************************/
/* Function for Matrix Types                     */
/*                                               */
/* Initialize:                                   */
/*   MPFMatrix init_mpfmatrix                    */
/*   MPFMatrix init2_mpfmatrix                   */
/* Free:                                         */
/*   void free_mpfmatrix(MPFMatrix mat)          */
/* Get & Set:                                    */
/*   void set_mpfmatrix_ij                       */
/*   void set_mpfmatrix_ij_d                     */
/* Output:                                       */
/*   void print_mpfmatrix(MPFMatrix mat)         */
/*************************************************/

#ifdef USE_GMP
typedef struct{
	unsigned long int prec;
//	unsigned int type;
	mpf_t *element;
//	mpf_ptr *element;
	long int row_dim, col_dim;
	long int real_row_dim, real_col_dim; // 2022-11-18(Fri) T.Kouya
	void *element_block; // mantissa block
} mpfmatrix;

typedef mpfmatrix *MPFMatrix;

typedef struct{
	unsigned long int prec;
//	unsigned int type;
	mpf_t *element;
//	mpf_ptr *element;
	long int dim, upper_dim, lower_dim;
	mpf_t zero; // = 0
} mpfbmatrix;

typedef mpfbmatrix *MPFBMatrix;

MPFMatrix init_mpfmatrix(long int row_dimension, long int col_dimension);

/* mbits ... A number of at least bits of mantissa */
MPFMatrix init2_mpfmatrix(long int row_dimension, long int col_dimension, unsigned long mbits);
#endif // USE_GMP

#ifdef USE_GMP
void free_mpfmatrix(MPFMatrix mat);
#endif // USE_GMP


#ifdef USE_GMP
mpf_ptr get_mpfmatrix_ij(MPFMatrix mat, long int row_index, long int col_index);

void set_mpfmatrix_ij(MPFMatrix mat, long int row_index, long int col_index, mpf_t val);

void set_mpfmatrix_ij_d(MPFMatrix mat, long int row_index, long int col_index, double val);

void set_mpfmatrix_ij_str(MPFMatrix mat, long int row_index, long int col_index, const char *str, int base);

void set_mpfmatrix_ij_ui(MPFMatrix mat, long int row_index, long int col_index, unsigned long val);

// used in old BNClibrary
#define gmpfmij get_mpfmatrix_ij
#define smpfmij set_mpfmatrix_ij

/* get precision of MPFMatrix */
unsigned long int prec_mpfmatrix(MPFMatrix mat);

/* search minimam precision in MPFMatrix */
unsigned long int minprec_mpfmatrix(MPFMatrix mat);

/* search maximam precision in MPFMatrix */
unsigned long int maxprec_mpfmatrix(MPFMatrix mat);

void print_mpfmatrix(MPFMatrix mat);
#endif // USE_GMP

/*************************************************/
/* Vector Calculations for MPFVector             */
/*
void add_mpfvector(MPFVector c, MPFVector a, MPFVector b)
void add2_mpfvector(MPFVector c, MPFVector a)
void sub_mpfvector(MPFVector c, MPFVector a, MPFVector b)
void sub2_mpfvector(MPFVector c, MPFVector a)
void cmul_mpfvector(MPFVector c, mpf_t val, MPFVector a)
void cmul2_mpfvector(MPFVector c, mpf_t val)
void add_cmul_mpfvector(MPFVector c, MPFVector a, mpf_t val, MPFVector b)
void ip_mpfvector(mpf_t ret, MPFVector a, MPFVector b)
void norm1_mpfvector(mpf_t ret, MPFVector a)
void norm2_mpfvector(mpf_t ret, MPFVector a)
void normi_mpfvector(mpf_t ret, MPFVector a)
void subst_mpfvector(MPFVector c, MPFVector a)
*/
/*************************************************/
#ifdef USE_GMP
/* c = a + b */
void add_mpfvector(MPFVector c, MPFVector a, MPFVector b);

/* c += a */
void add2_mpfvector(MPFVector c, MPFVector a);

/* c = a - b */
void sub_mpfvector(MPFVector c, MPFVector a, MPFVector b);

/* c -= a */
void sub2_mpfvector(MPFVector c, MPFVector a);

/* c = val * a */
void cmul_mpfvector(MPFVector c, mpf_t val, MPFVector a);

/* c *= val */
void cmul2_mpfvector(MPFVector c, mpf_t val);

/* c = a + val * b */
void add_cmul_mpfvector(MPFVector c, MPFVector a, mpf_t val, MPFVector b);

/* ret = (a, b) */
void ip_mpfvector(mpf_t ret, MPFVector a, MPFVector b);

/* ret = ||a||_1 */
void norm1_mpfvector(mpf_t ret, MPFVector a);

/* ret := ||a||_2 */
void norm2_mpfvector(mpf_t ret, MPFVector a);

/* ||a||_infty */
void normi_mpfvector(mpf_t ret, MPFVector a);

/* c := a */
void subst_mpfvector(MPFVector c, MPFVector a);

/* c := 0 */
void set0_mpfvector(MPFVector c);

/* append 2005.07/12 */
/*
	ret(index_start) = src(src_index_start)
	 ...
	ret(index_end  ) = src(src_index_end)
*/
void copy_mpfvector_ij(MPFVector ret, long int index_start, long int index_end, MPFVector src, long int src_index_start, long int src_index_end);
#endif // USE_GMP


/*************************************************/
/* Matrix Caluculations for MPFMatrix            */
/*
void normf_mpfmatrix(mpf_t ret, MPFMatrix mat)
void norm1_mpfmatrix(mpf_t ret, MPFMatrix mat)
void normi_mpfmatrix(mpf_t ret, MPFMatrix mat)
void add_mpfmatrix(MPFMatrix c, MPFMatrix a, MPFMatrix b);
void sub_mpfmatrix(MPFMatrix c, MPFMatrix a, MPFMatrix b);
void mul_mpfmatrix(MPFMatrix c, MPFMatrix a, MPFMatrix b);
void mul_mpfmatrix_mpfvec(MPFVector v, MPFMatrix a, MPFVector vb)
void mul_mpfmatrixt_mpfvec(MPFVector v, MPFMatrix a, MPFVector vb)
void transpose_mpfmatrix(MPFMatrix c, MPFMatrix a);
void inv_mpfmatrix(MPFMatrix a);
void subst_mpfmatrux(MPFMatrix c, MPFMatrix a);
*/
/*************************************************/
#ifdef USE_GMP
/* Frobenius Norm of Matrix */
void normf_mpfmatrix(mpf_t ret, MPFMatrix mat);

/* Infinity Norm of Matrix */
void normi_mpfmatrix(mpf_t ret, MPFMatrix mat);

/* 1 Norm of Matrix */
void norm1_mpfmatrix(mpf_t ret, MPFMatrix mat);

/* c := a + b */
void add_mpfmatrix(MPFMatrix c, MPFMatrix a, MPFMatrix b);

/* c := a - b */
void sub_mpfmatrix(MPFMatrix c, MPFMatrix a, MPFMatrix b);

/* c := sc * a */
void cmul_mpfmatrix(MPFMatrix c, mpf_t sc, MPFMatrix a);

/* c = a * b */
void mul_mpfmatrix(MPFMatrix c, MPFMatrix a, MPFMatrix b);

/* c = a^T */
void transpose_mpfmatrix(MPFMatrix c, MPFMatrix a);

/* c := a */
void subst_mpfmatrix(MPFMatrix c, MPFMatrix a);

/* c := 0 */
void set0_mpfmatrix(MPFMatrix c);

/* c := I */
void setI_mpfmatrix(MPFMatrix c);

/* v := a * vb */
void mul_mpfmatrix_mpfvec(MPFVector v, MPFMatrix a, MPFVector vb);

/* v := a^T * vb */
void mul_mpfmatrixt_mpfvec(MPFVector v, MPFMatrix a, MPFVector vb);

/* a = a^(-1) */
/* square matrix only */
void inv_mpfmatrix(MPFMatrix a);

// Appended in 2024-05-09 T.Kouya
// from lanczos.c

/* c = a - sc * b */
void subcmul_mpfvector(MPFVector c, MPFVector a, mpf_t sc, MPFVector b);

/* mat := (vec[0] vec[1] ... vec[n]) */
void subst_mpfmatrix_mpfvec(MPFMatrix mat, MPFVector vec[]);

/* (mpf_t)c := (double)a */
void subst_mpfmatrix_dmat(MPFMatrix c, DMatrix a);

/* (mpf_t)c := (dd)a */
//void subst_mpfmatrix_ddmat(MPFMatrix c, DDMatrix a);

/* (mpf_t)c := (td)a */
//void subst_mpfmatrix_tdmat(MPFMatrix c, TDMatrix a);

/* (mpf_t)c := (qd)a */
//void subst_mpfmatrix_qdmat(MPFMatrix c, QDMatrix a);

// ----- linear_append.c ------


/* c := (double)a */
void subst_fmatrix_dmat(FMatrix c, DMatrix a);

/* c := (double)a^T */
void subst_fmatrix_dmat_trans(FMatrix c, DMatrix a);

/* c := (double)a */
void subst_fvector_dvec(FVector c, DVector a);

/* (double)c := (float)a */
void subst_dvector_fvec(DVector c, FVector a);


//#ifdef USE_DLINEAR
/* get residual in double precision */
/* r := b - A * x */
void residual_dmat_dvec(DVector r, DVector b, DMatrix mat, DVector x);

/* get residual in mpf_t precision */
/* r := b - A * x */
void residual_mpfmat_mpfvec(MPFVector r, MPFVector b, MPFMatrix mat, MPFVector x);

/* c := (mpf_t)a */
void subst_dmatrix_mpfmat(DMatrix c, MPFMatrix a);

/* c := (mpf_t)a^T */
void subst_dmatrix_mpfmat_trans(DMatrix c, MPFMatrix a);

/* c := (double)a */
void subst_dvector_mpfvec(DVector c, MPFVector a);

/* (mpf_t)c := (double)a */
void subst_mpfvector_dvec(MPFVector c, DVector a);
//#endif // if USE_DLINEAR

/* vec := -vec */
void neg_mpfvector(MPFVector vec);

/* Normwise relative error of vector */
void relerr_mpfvector(mpf_t relerr, MPFVector approx_vec, MPFVector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_mpfvector(mpf_t max_relerr, mpf_t min_relerr, mpf_t norm_relerr, MPFVector approx_vec, MPFVector true_vec, int norm_type);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_mpfmatrix(MPFMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

// 2026-02-03(Tue) T.Kouya
// absmax_mpfmatrix
void absmax_mpfmatrix(mpf_t ret, long int *max_i, long int *max_j, MPFMatrix mat);

//----------------------------------/
// mpflu.c
//----------------------------------/
int MPFLUdecomp(MPFMatrix a);
int SolveMPFLS(MPFVector answer, MPFMatrix lu, MPFVector b);
int MPFLUdecompP(MPFMatrix a, long int ch[]);
int SolveMPFLSP(MPFVector answer, MPFMatrix lu, MPFVector b, long int ch[]);
int MPFLUdecompC(MPFMatrix a, long int row_ch[], long int col_ch[]);
int SolveMPFLSC(MPFVector answer, MPFMatrix lu, MPFVector b, long int row_ch[], long int col_ch[]);
int MPFLUdecompPM(MPFMatrix a, long int ch[]);
int SolveMPFLSPM(MPFVector answer, MPFMatrix lu, MPFVector b, long int ch[]);

//----------------------------------/
// mpflu_strassen.c
//----------------------------------/
int MPFLUdecomp_square(MPFMatrix a, long int start_index, long int min_dim);
int MPFLUdecomp_l21(MPFMatrix l21, MPFMatrix a, long int start_index, long int min_dim);
int MPFLUdecomp_u12(MPFMatrix u12, MPFMatrix a, long int start_index, long int min_dim);
int MPFLUdecomp_a22(MPFMatrix a, MPFMatrix d22, MPFMatrix l21, MPFMatrix u12, long int start_index, long int min_dim);
int MPFLUdecomp_strassen(MPFMatrix a, long int min_dim);
int MPFLUdecomp_strassenPM(MPFMatrix a, long int ch[], long int min_dim);

#ifdef _OPENMP
int MPFLUdecomp_square_omp(MPFMatrix a, long int start_index, long int min_dim);
int MPFLUdecomp_l21_omp(MPFMatrix l21, MPFMatrix a, long int start_index, long int min_dim);
int MPFLUdecomp_u12_omp(MPFMatrix u12, MPFMatrix a, long int start_index, long int min_dim);
int MPFLUdecomp_a22_omp(MPFMatrix a, MPFMatrix d22, MPFMatrix l21, MPFMatrix u12, long int start_index, long int min_dim);
int MPFLUdecomp_strassen_omp(MPFMatrix a, long int min_dim);
int MPFLUdecomp_strassenPM_omp(MPFMatrix a, long int ch[], long int min_dim);
int MPFLUdecomp_omp(MPFMatrix a);
#endif // _OPENMP


//----------------------------------/
// mpflu_oz.c
//----------------------------------/
int MPFLUdecomp_a22_oz(MPFMatrix a, MPFMatrix d22, MPFMatrix l21, MPFMatrix u12, long int start_index, long int min_dim, int max_num_div);
int MPFLUdecomp_oz(MPFMatrix a, long int min_dim, int max_num_div);
int MPFLUdecomp_ozPM(MPFMatrix a, long int ch[], long int min_dim, int max_num_div);

#ifdef _OPENMP
int MPFLUdecomp_a22_oz_omp(MPFMatrix a, MPFMatrix d22, MPFMatrix l21, MPFMatrix u12, long int start_index, long int min_dim, int max_num_div);
int MPFLUdecomp_oz_omp(MPFMatrix a, long int min_dim, int max_num_div);
int MPFLUdecomp_ozPM_omp(MPFMatrix a, long int ch[], long int min_dim, int max_num_div);
#endif // _OPENMP

//----------------------------------/
// gtestmat.c
//----------------------------------/
/* 1. Hilbert Matrix */
void hilbert_mpfmatrix(MPFMatrix a, long int dim);

/* 2. Lotkin Matrix */
void lotkin_mpfmatrix(MPFMatrix a, long int dim);

/* 3. Frank Matrix */
void frank_mpfmatrix(MPFMatrix a, long int dim);

/* 4. Tridiagonal Matrix */
void tridiag_mpfmatrix(MPFMatrix a, MPFVector low_subdiag, MPFVector diag, MPFVector up_subdiag, long int dim);

/* 5. Integer Symmetrix Random Matrix */
void int_sym_rand_mpfmatrix(MPFMatrix mat, long int max, long int seed, long int dim);

/* 6. Integer Unsymmetrix Random Matrix */
void int_unsym_rand_mpfmatrix(MPFMatrix mat, long int max, long int seed, long int dim);

/* 7. Real Diagonal Matrix */
void diag_mpfmatrix(MPFMatrix mat, MPFVector diag, long int dim);

/* 8. Toeplitz Matrix */
void toeplitz_mpfmatrix(MPFMatrix mat, mpf_t gamma_param, long int dim);

// 9. Pascal Matrix
void pascal_mpfmatrix(MPFMatrix ret, long int dim);

// 10. I - randmatrix
void im_rand_mpfmatrix(MPFMatrix ret, unsigned long seed);

//----------------------------------/
// fread_write.c
//----------------------------------/
void fread_mpfmatrix(FILE *fp, MPFMatrix mat);
void fread_mpfmatrix_fname(const char *fname, MPFMatrix mat);
void fwrite_mpfmatrix(FILE *fp, MPFMatrix mat);
void fwrite_mpfmatrix_fname(const char *fname, MPFMatrix mat);
//void fread_mpfpolycoef(FILE *fp, MPFPoly p, long int maxdeg);
//void fread_mpfpolycoef_fname(const char *fname, MPFPoly p, long int maxdeg);
void fread_mpfvector(FILE *fp, MPFVector vec);
void fread_mpfvector_fname(const char *fname, MPFVector vec);
void fwrite_mpfvector(FILE *fp, MPFVector vec);
void fwrite_mpfvector_fname(const char *fname, MPFVector vec);

void read_test_linear_eq(MPFMatrix A, MPFVector true_x, MPFVector b, long int dim, const char *fname_A, const char *fname_true_x, const char *fname_b);

// write problem into file
void write_test_linear_eq(MPFMatrix A, MPFVector true_x, MPFVector b, long int dim, const char *fname_A, const char *fname_true_x, const char *fname_b);

#endif // USE_GMP

#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus

#endif // __BNC_MPFLINEAR_H__
