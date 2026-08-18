/********************************************************************************/
/* tdlinear.h: Triple Double precision Linear Computation Library               */
/* Copyright (C) 2020 Tomonori Kouya                                            */
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
#ifndef __BNC_TSLINEAR_H__
#define __BNC_TSLINEAR_H__

// Common defs
#include "bnc_common.h"

#include "rds.h"

#include "flinear.h" // Double precision linear computation
#ifdef USE_DSLINEAR
#include "dslinear.h" // DD precision linear computation
#endif // USE_DSLINEAR

#ifdef USE_GMP
#include "gmp.h"
#include "mpfr.h"
#include "mpfr_dtq_sd.h"
#include "mpflinear.h"
#endif //USE_GMP//

#if defined(__AVX2__) || defined(__AVX512F__)
#include "avx2/bncavx.h"
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2)
#include "sve2/bncsve2.h"
#include "neon/bncneon.h"   /* SVE2 build keeps NEON for the #elif fallback paths */
#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON)
#include "neon/bncneon.h"
#endif // defined(__AVX2__) || defined(__AVX512F__)

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//#ifndef __cplusplus
//#endif // __cplusplus

// qsrel_diff
tsfloat tsrel_diff(tsfloat a, tsfloat b);

tsfloat tsrel_diff_array(tsfloat approx_a[], tsfloat approx_b[], int dim, int print_flag);

// TD vector
typedef struct{
	long int dim; // dim <= real_dim
	long int real_dim; // multiplier of _BNC_D_WIDTH
	float *element[TSSIZE];
} tsvector;

typedef tsvector *TSVector;

// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
#elif defined(__AVX512F__) // __AVX512F__
#else // others
#endif // __AVX2__

// get_tsvector_i
static inline tsfloat get_tsvector_i_tsfloat(TSVector vec, long int index)
//static inline float *get_tsvector_i(TSVector vec, long int index)
{
	tsfloat ret;

	ret.val[0] = vec->element[0][index];
	ret.val[1] = vec->element[1][index];
	ret.val[2] = vec->element[2][index];

	return ret;
}

//	GET_TSVECTOR_I(vec, index))
#define GET_TSVECTOR_I(vec, index) ((get_tsvector_i_tsfloat((vec), (index)).val))
#define get_tsvector_i(vec, index) ((get_tsvector_i_tsfloat((vec), (index)).val))

// set_tsvector_i
static inline void set_tsvector_i(TSVector vec, long int index, float *val) // val[TSSIZE]
{
	float ret[TSSIZE];

	vec->element[0][index] = val[0];
	vec->element[1][index] = val[1];
	vec->element[2][index] = val[2];
}
#define SET_TSVECTOR_I(vec, index, val) set_tsvector_i((vec), (index), (val))

// set_tsvector_i_d
static inline void set_tsvector_i_f(TSVector vec, long int index, float val) // val
{
	float ret[TSSIZE];

	vec->element[0][index] = val;
	vec->element[1][index] = 0.0f;
	vec->element[2][index] = 0.0f;
}
#define SET_TSVECTOR_I_F(vec, index, val) set_tsvector_i_f((vec), (index), (val))

// set0_tsvector_i
static inline void set0_tsvector_i(TSVector vec, long int index)
{
	float ret[TSSIZE];

	vec->element[0][index] = 0.0f;
	vec->element[1][index] = 0.0f;
	vec->element[2][index] = 0.0f;
}
#define SET0_TSVECTOR_I(vec, index) set0_tsvector_i((vec), (index))

// set a zero vector
//void set0_tsvector(TSVector vec)
void set0_tsvector(TSVector vec);

#if defined(USE_GMP) && defined(USE_MPFR)
tsfloat tsrel_diff_array(tsfloat approx_a[], tsfloat approx_b[], int dim, int print_flag);

// Frobenius norm
tsfloat tsnormf(tsfloat array[], int dim);

// generate a text vector: vec(i) := sqrt(sqrt_seed) * (i + 1)
void set_test_tsvector(tsfloat vec[], int sqrt_seed, int dim);

// generate a text matrix: mat(i, j) := sqrt(sqrt_seed) * (i + j - 1)
void set_test_tsmatrix(tsfloat mat[], int sqrt_seed, int row_dim, int col_dim);

#endif // defined(USE_GMP) && defined(USE_MPFR)

// initialize tsvector
TSVector init_tsvector(long int dim);

// free tsvector
void free_tsvector(TSVector vec);

// TSVector vec -> tsfloat array
void set_tsfloat_tsvec(tsfloat ret[], int ret_dim, TSVector vec);

// tsfloat array -> TSVector ret
void set_tsvector_tsfloat(TSVector ret, tsfloat array[], int array_dim);

// print tsvector
void print_tsvector(TSVector vec);

// set_tsvector_i_str
void set_tsvector_i_str(TSVector vec, long int index, const char *str);

/*************************************************/
/* Vector Calculations for TSVector               */
/*
void add_tsvector(TSVector c, TSVector a, TSVector b)
void add2_tsvector(TSVector c, TSVector a)
void sub_tsvector(TSVector c, TSVector a, TSVector b)
void sub2_tsvector(TSVector c, DVector a)
void cmul_tsvector(TSVector c, float val[TSSIZE], TSVector a)
void cmul2_tsvector(TSVector c, float val[TSSIZE])
void add_cmul_tsvector(TSVector c, TSVector a, float val[TSSIZE], TSVector b)
float ip_tsvector(TSVector a, TSVector b)
float norm1_tsvector(TSVector a)
float norm2_tsvector(TSVector a)
float normi_tsvector(TSVector a)
void subst_tsvector(TSVector c, TSVector a)
*/
/*************************************************/
/* c = a + b */
void add_tsvector(TSVector c, TSVector a, TSVector b);

/* c += a */
void add2_tsvector(TSVector c, TSVector a);

/* c = a - b */
void sub_tsvector(TSVector c, TSVector a, TSVector b);

/* c -= a */
void sub2_tsvector(TSVector c, TSVector a);

/* c = val * a */
void cmul_tsvector(TSVector c, float val[TSSIZE], TSVector a);

/* c *= val */
void cmul2_tsvector(TSVector c, float val[TSSIZE]);

/* c = a + val * b */
void add_cmul_tsvector(TSVector c, TSVector a, float val[TSSIZE], TSVector b);
void sub_cmul_tsvector(TSVector c, TSVector a, float val[TSSIZE], TSVector b);

/* (a, b) */
void ip_tsvector(float ret[TSSIZE], TSVector a, TSVector b);

/* c := a */
void subst_tsvector(TSVector c, TSVector a);

/* c := -a */
void neg_tsvector(TSVector c, TSVector a);

/* ||a||_1 */
void norm1_tsvector(float ret[TSSIZE], TSVector a);

/* ||a||_infty */
void normi_tsvector(float ret[TSSIZE], TSVector a);

// Euclid norm
void norm2_tsvector(float ret[TSSIZE], TSVector vec);

// tdadd
void _bncavx2_tsadd(tsfloat ret[], tsfloat a[], tsfloat b[], int dim);
void _bncavx2_tsvadd(TSVector ret, TSVector a, TSVector b, int dim);

// tdmul
void _bncavx2_tsmul(tsfloat ret[], tsfloat a[], tsfloat b[], int dim);
void _bncavx2_tsvmul(TSVector ret, TSVector a, TSVector b, int dim);

/* tddiv */
void _bncavx2_tsdiv(tsfloat ret[], tsfloat a[], tsfloat b[], int dim);
void _bncavx2_tsvdiv(TSVector ret, TSVector a, TSVector b, int dim);

// TD matrix
typedef struct{
	long int row_dim, col_dim;
	long int real_row_dim, real_col_dim; // multiplier of _BNC_D_WIDTH
	float *element[TSSIZE];
} tsmatrix;

typedef tsmatrix *TSMatrix;

// old
//#define get_tsmatrix_ij(mat, i, j) ( *((mat)->element + (i) * (mat)->col_dim + (j)) )
//#define set_tsmatrix_ij(mat, i, j, val) ( *((mat)->element + (i) * (mat)->col_dim + (j)) = (val) )

// new
//#define get_tsmatrix_ij(mat, i, j) ( *((mat)->element + (i) * (mat)->real_col_dim + (j)) )
//#define set_tsmatrix_ij(mat, i, j, val) ( *((mat)->element + (i) * (mat)->real_col_dim + (j)) = (val) )

// get_tsmatrix_ij
static inline tsfloat get_tsmatrix_ij_tsfloat(TSMatrix mat, long int i, long int j)
{
	long int ij_index;
//	float ret[TSSIZE];
	tsfloat ret;

	ij_index = mat->real_col_dim * i + j;

	ret.val[0] = mat->element[0][ij_index];
	ret.val[1] = mat->element[1][ij_index];
	ret.val[2] = mat->element[2][ij_index];

	return ret;
} 
#define GET_TSMATRIX_IJ(mat, i, j) ((get_tsmatrix_ij_tsfloat((mat), (i), (j)).val))
#define get_tsmatrix_ij(mat, i, j) ((get_tsmatrix_ij_tsfloat((mat), (i), (j)).val))

// set_tsmatrix_ij
static inline void set_tsmatrix_ij(TSMatrix mat, long int i, long int j, float *val) // val[TSSIZE]
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[0][ij_index] = val[0];
	mat->element[1][ij_index] = val[1];
	mat->element[2][ij_index] = val[2];

	return;
} 
#define SET_TSMATRIX_IJ(mat, i, j, val) set_tsmatrix_ij((mat), (i), (j))

// set_tsmatrix_ij_d
static inline void set_tsmatrix_ij_f(TSMatrix mat, long int i, long int j, float val)
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[0][ij_index] = val;
	mat->element[1][ij_index] = 0.0f;
	mat->element[2][ij_index] = 0.0f;

	return;
} 
#define SET_TSMATRIX_IJ_F(mat, i, j, val) set_tsmatrix_ij_f((mat), (i), (j), (val))
#define SET_TSMATRIX_IJ_UI(mat, i, j, val) set_tsmatrix_ij_f((mat), (i), (j), (float)(val))
#define set_tsmatrix_ij_ui(mat, i, j, val) set_tsmatrix_ij_f((mat), (i), (j), (float)(val))

// set0_tsmatrix_ij
static inline void set0_tsmatrix_ij(TSMatrix mat, long int i, long int j)
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[0][ij_index] = 0.0f;
	mat->element[1][ij_index] = 0.0f;
	mat->element[2][ij_index] = 0.0f;

	return;
}
#define SET0_TSMATRIX_IJ(mat, i, j) set0_tsmatrix_ij((mat), (i), (j))

// set a zero matrix
//void set0_tsmatrix(TSMatrix mat)
void set0_tsmatrix(TSMatrix mat);

// initialize tsmatrix
TSMatrix init_tsmatrix(long int row_dim, long int col_dim);

// free tsmatrix
void free_tsmatrix(TSMatrix mat);

// print tsmatrix
void print_tsmatrix(TSMatrix mat);

// TSMatrix mat -> tsfloat array
void set_tsfloat_tsmat(tsfloat ret[], int ret_dim, TSMatrix mat);

// tsfloat array -> TDmatrix ret
void set_tsmatrix_tsfloat(TSMatrix ret, tsfloat array[], int array_dim);

/*************************************************/
/* Matrix Caluculations for TSMatrix            */
/*
void normf_tsmatrix(float ret[TSSIZE], TSMatrix mat)
void norm1_tsmatrix(float ret[TSSIZE], TSMatrix mat)
void normi_tsmatrix(float ret[TSSIZE], TSMatrix mat)
void add_tsmatrix(TSMatrix c, TSMatrix a, TSMatrix b);
void sub_tsmatrix(TSMatrix c, TSMatrix a, TSMatrix b);
void mul_tsmatrix(TSMatrix c, TSMatrix a, TSMatrix b);
void mul_tsmatrix_tsvec(TSVector v, TSMatrix a, TSVector vb)
void mul_tsmatrixt_tsvec(TSVector v, TSMatrix a, TSVector vb)
void transpose_tsmatrix(TSMatrix c, TSMatrix a);
void inv_tsmatrix(TSMatrix a);
void subst_mpfmatrux(TSMatrix c, TSMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_tsmatrix(float ret[TSSIZE], TSMatrix mat);

/* 1 Norm of Matrix */
void norm1_tsmatrix(float ret[TSSIZE], TSMatrix mat);

/* c := a + b */
void add_tsmatrix(TSMatrix c, TSMatrix a, TSMatrix b);

/* c := a - b */
void sub_tsmatrix(TSMatrix c, TSMatrix a, TSMatrix b);

/* (QDMatrix)c := (TSMatrix)a - (TSMatrix)b */
//void sub_qdmatrix_tsmat_tsmat(QDMatrix c, TSMatrix a, TSMatrix b);

/* c := sc * a */
void cmul_tsmatrix(TSMatrix c, float sc[TSSIZE], TSMatrix a);

/* c = a^T */
void transpose_tsmatrix(TSMatrix c, TSMatrix a);

/* c := a */
void subst_tsmatrix(TSMatrix c, TSMatrix a);

/* c := a */
//void subst_qdmatrix_tsmat(QDMatrix c, TSMatrix a);

/* c := I */
void setI_tsmatrix(TSMatrix c);

/* v := a * vb */
void mul_tsmatrix_tsvec(TSVector v, TSMatrix a, TSVector vb);

/* v := a^T * vb */
void mul_tsmatrixt_tsvec(TSVector v, TSMatrix a, TSVector vb);

/* a = a^(-1) */
/* square matrix only */
void inv_tsmatrix(TSMatrix a);

// matrix multiplication
// ret := A * B
void mul_tsmatrix(TSMatrix ret, TSMatrix a, TSMatrix b);

// Frobenius norm
void normf_tsmatrix(float ret[TSSIZE], TSMatrix mat);

/* Elementwise relative errors of vector */
void relerr_element_tsvector(float max_relerr[TSSIZE], float min_relerr[TSSIZE], float norm_relerr[TSSIZE], TSVector approx_vec, TSVector true_vec, int norm_type);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_tsmatrix(TSMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

// tsmatmul_tsvec
void tsmatmul_tsvec(TSVector ret, TSVector mat_a, TSVector mat_b, int row_dim, int mid_dim, int col_dim);

// tsmatmul_tsvec_ur4
void tsmatmul_tsvec_ur4(TSVector ret, TSVector mat_a, TSVector mat_b, int row_dim, int mid_dim, int col_dim);

// tsmatmul_tsvec_avx2
void tsmatmul_tsvec_avx2(TSVector ret, TSVector mat_a, TSVector mat_b, int row_dim, int mid_dim, int col_dim);

#ifdef __BNC_TDLINEAR_H__
//void _bncavx2_tdadd(tsfloat ret[], tsfloat a[], tsfloat b[], int dim)
//void _bncavx2_tdvadd(TSVector ret, TSVector a, TSVector b, int dim)
//void _bncavx2_tdmul(tsfloat ret[], tsfloat a[], tsfloat b[], int dim)
//void _bncavx2_tdvmul(TSVector ret, TSVector a, TSVector b, int dim)
//#include "tdv_addmul.c"
#endif // __BNC_TDLINEAR_H__

int TSLUdecomp(TSMatrix a);
int SolveTSLS(TSVector answer, TSMatrix lu, TSVector b);
int TSLUdecompP(TSMatrix a, long int ch[]);
int SolveTSLSP(TSVector answer, TSMatrix lu, TSVector b, long int ch[]);
int TSLUdecompPM(TSMatrix a, long int ch[]);
int SolveTSLSPM(TSVector answer, TSMatrix lu, TSVector b, long int ch[]);

/* 1. Hilbert Matrix */
void hilbert_tsmatrix(TSMatrix a, long int dim);

/* 2. Lotkin Matrix */
void lotkin_tsmatrix(TSMatrix a, long int dim);

/* 3. Frank Matrix */
void frank_tsmatrix(TSMatrix a, long int dim);

/* 4. Tridiagonal Matrix */
void tridiag_tsmatrix(TSMatrix a, TSVector low_subdiag, TSVector diag, TSVector up_subdiag, long int dim);

/* 5. Integer Symmetrix Random Matrix */
void int_sym_rand_tsmatrix(TSMatrix mat, long int max, long int seed, long int dim);

/* 6. Integer Unsymmetrix Random Matrix */
void int_unsym_rand_tsmatrix(TSMatrix mat, long int max, long int seed, long int dim);

/* 7. Real Diagonal Matrix */
void diag_tsmatrix(TSMatrix mat, TSVector diag, long int dim);

/* 8. Toeplitz Matrix */
void toeplitz_tsmatrix(TSMatrix mat, float gamma_param[TSSIZE], long int dim);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // define __BNC_TSLINEAR_H__
