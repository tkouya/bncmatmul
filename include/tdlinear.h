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
#ifndef __BNC_TDLINEAR_H__
  #define __BNC_TDLINEAR_H__

// Common defs
#include "bnc_common.h"

#include "rdd.h"

#include "dlinear.h" // Double precision linear computation
#include "ddlinear.h" // DD precision linear computation

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
#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // __ARM_NEON
#include "neon/bncneon.h"
#endif // defined(__AVX2__) || defined(__AVX512F__)

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//#ifndef __cplusplus
//#endif // __cplusplus

// TD vector
typedef struct{
	long int dim; // dim <= real_dim
	long int real_dim; // multiplier of _BNC_D_WIDTH
	double *element[TDSIZE];
} tdvector;

typedef tdvector *TDVector;

// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
#elif defined(__AVX512F__) // __AVX512F__
#else // others
#endif // __AVX2__

// get_tdvector_i
static inline tdfloat get_tdvector_i_tdfloat(TDVector vec, long int index)
//static inline double *get_tdvector_i(TDVector vec, long int index)
{
	tdfloat ret;

	ret.val[0] = vec->element[0][index];
	ret.val[1] = vec->element[1][index];
	ret.val[2] = vec->element[2][index];

	return ret;
}

//	GET_TDVECTOR_I(vec, index))
#define GET_TDVECTOR_I(vec, index) ((get_tdvector_i_tdfloat((vec), (index)).val))
#define get_tdvector_i(vec, index) ((get_tdvector_i_tdfloat((vec), (index)).val))

// get_tdvector_i_dd
static inline ddfloat get_tdvector_i_ddfloat(TDVector vec, long int index)
//static inline double *get_tdvector_i(TDVector vec, long int index)
{
	ddfloat ret;

	ret.val[0] = vec->element[0][index];
	ret.val[1] = vec->element[1][index];
	//ret.val[2] = vec->element[2][index];

	return ret;
}
//	GET_TDVECTOR_I(vec, index))
#define GET_TDVECTOR_I_DD(vec, index) ((get_tdvector_i_ddfloat((vec), (index)).val))
#define get_tdvector_i_dd(vec, index) ((get_tdvector_i_ddfloat((vec), (index)).val))

// set_tdvector_i
static inline void set_tdvector_i(TDVector vec, long int index, double *val) // val[TDSIZE]
{
	double ret[TDSIZE];

	vec->element[0][index] = val[0];
	vec->element[1][index] = val[1];
	vec->element[2][index] = val[2];
}
#define SET_TDVECTOR_I(vec, index, val) set_tdvector_i((vec), (index), (val))
// 2025-02-19(Wed) T.Kouya
#define subst_tdvector_i(ret, vec, index) rtd_set((ret), get_tdvector_i((vec), (index)))

// set_tdvector_i_d
static inline void set_tdvector_i_d(TDVector vec, long int index, double val) // val
{
	double ret[TDSIZE];

	vec->element[0][index] = val;
	vec->element[1][index] = 0.0;
	vec->element[2][index] = 0.0;
}
#define SET_TDVECTOR_I_D(vec, index, val) set_tdvector_i_d((vec), (index), (val))

// set_tdvector_i_d
static inline void set_tdvector_i_dd(TDVector vec, long int index, double val[DDSIZE]) // val
{
	double ret[TDSIZE];

	vec->element[0][index] = val[0];
	vec->element[1][index] = val[1];
	vec->element[2][index] = 0.0;
}
#define SET_TDVECTOR_I_DD(vec, index, val) set_tdvector_i_dd((vec), (index), (val))

// set0_tdvector_i
static inline void set0_tdvector_i(TDVector vec, long int index)
{
	double ret[TDSIZE];

	vec->element[0][index] = 0.0;
	vec->element[1][index] = 0.0;
	vec->element[2][index] = 0.0;
}
#define SET0_TDVECTOR_I(vec, index) set0_tdvector_i((vec), (index))

// set a zero vector
//void set0_tdvector(TDVector vec)
void set0_tdvector(TDVector vec);

// tdrel_diff
tdfloat tdrel_diff(tdfloat a, tdfloat b);

#if defined(USE_GMP) && defined(USE_MPFR)
tdfloat tdrel_diff_array(tdfloat approx_a[], tdfloat approx_b[], int dim, int print_flag);

// Frobenius norm
tdfloat tdnormf(tdfloat array[], int dim);

// generate a text vector: vec(i) := sqrt(sqrt_seed) * (i + 1)
void set_test_tdvector(tdfloat vec[], int sqrt_seed, int dim);

// generate a text matrix: mat(i, j) := sqrt(sqrt_seed) * (i + j - 1)
void set_test_tdmatrix(tdfloat mat[], int sqrt_seed, int row_dim, int col_dim);

#endif // defined(USE_GMP) && defined(USE_MPFR)

// initialize tdvector
TDVector init_tdvector(long int dim);

// free tdvector
void free_tdvector(TDVector vec);

// TDVector vec -> tdfloat array
void set_tdfloat_tdvec(tdfloat ret[], int ret_dim, TDVector vec);

// tdfloat array -> TDVector ret
void set_tdvector_tdfloat(TDVector ret, tdfloat array[], int array_dim);

// print tdvector
void print_tdvector(TDVector vec);

// set_tdvector_i_str
void set_tdvector_i_str(TDVector vec, long int index, const char *str);

/*************************************************/
/* Vector Calculations for TDVector               */
/*
void add_tdvector(TDVector c, TDVector a, TDVector b)
void add2_tdvector(TDVector c, TDVector a)
void sub_tdvector(TDVector c, TDVector a, TDVector b)
void sub2_tdvector(TDVector c, DVector a)
void cmul_tdvector(TDVector c, double val[TDSIZE], TDVector a)
void cmul2_tdvector(TDVector c, double val[TDSIZE])
void add_cmul_tdvector(TDVector c, TDVector a, double val[TDSIZE], TDVector b)
double ip_tdvector(TDVector a, TDVector b)
double norm1_tdvector(TDVector a)
double norm2_tdvector(TDVector a)
double normi_tdvector(TDVector a)
void subst_tdvector(TDVector c, TDVector a)
*/
/*************************************************/
/* c = a + b */
void add_tdvector(TDVector c, TDVector a, TDVector b);

/* c += a */
void add2_tdvector(TDVector c, TDVector a);

/* c = a - b */
void sub_tdvector(TDVector c, TDVector a, TDVector b);

/* c -= a */
void sub2_tdvector(TDVector c, TDVector a);

/* c = val * a */
void cmul_tdvector(TDVector c, double val[TDSIZE], TDVector a);

/* c *= val */
void cmul2_tdvector(TDVector c, double val[TDSIZE]);

/* c = a + val * b */
void add_cmul_tdvector(TDVector c, TDVector a, double val[TDSIZE], TDVector b);

/* c = a - val * b */
void sub_cmul_tdvector(TDVector c, TDVector a, double val[TDSIZE], TDVector b);

/* (a, b) */
void ip_tdvector(double ret[TDSIZE], TDVector a, TDVector b);

/* c := a */
void subst_tdvector(TDVector c, TDVector a);

/* c := -a */
void neg_tdvector(TDVector c, TDVector a);

/* ||a||_1 */
void norm1_tdvector(double ret[TDSIZE], TDVector a);

/* ||a||_infty */
void normi_tdvector(double ret[TDSIZE], TDVector a);

// Euclid norm
void norm2_tdvector(double ret[TDSIZE], TDVector vec);

// tdadd
void _bncavx2_tdadd(tdfloat ret[], tdfloat a[], tdfloat b[], int dim);
void _bncavx2_tdvadd(TDVector ret, TDVector a, TDVector b, int dim);

// tdmul
void _bncavx2_tdmul(tdfloat ret[], tdfloat a[], tdfloat b[], int dim);
void _bncavx2_tdvmul(TDVector ret, TDVector a, TDVector b, int dim);

/* tddiv */
void _bncavx2_tddiv(tdfloat ret[], tdfloat a[], tdfloat b[], int dim);
void _bncavx2_tdvdiv(TDVector ret, TDVector a, TDVector b, int dim);

// TD matrix
typedef struct{
	long int row_dim, col_dim;
	long int real_row_dim, real_col_dim; // multiplier of _BNC_D_WIDTH
	double *element[TDSIZE];
} tdmatrix;

typedef tdmatrix *TDMatrix;

// old
//#define get_tdmatrix_ij(mat, i, j) ( *((mat)->element + (i) * (mat)->col_dim + (j)) )
//#define set_tdmatrix_ij(mat, i, j, val) ( *((mat)->element + (i) * (mat)->col_dim + (j)) = (val) )

// new
//#define get_tdmatrix_ij(mat, i, j) ( *((mat)->element + (i) * (mat)->real_col_dim + (j)) )
//#define set_tdmatrix_ij(mat, i, j, val) ( *((mat)->element + (i) * (mat)->real_col_dim + (j)) = (val) )

// get_tdmatrix_ij
static inline tdfloat get_tdmatrix_ij_tdfloat(TDMatrix mat, long int i, long int j)
{
	long int ij_index;
//	double ret[TDSIZE];
	tdfloat ret;

	ij_index = mat->real_col_dim * i + j;

	ret.val[0] = mat->element[0][ij_index];
	ret.val[1] = mat->element[1][ij_index];
	ret.val[2] = mat->element[2][ij_index];

	return ret;
} 
#define GET_TDMATRIX_IJ(mat, i, j) ((get_tdmatrix_ij_tdfloat((mat), (i), (j)).val))
#define get_tdmatrix_ij(mat, i, j) ((get_tdmatrix_ij_tdfloat((mat), (i), (j)).val))

// get_tdmatrix_ij_dd
static inline ddfloat get_tdmatrix_ij_ddfloat(TDMatrix mat, long int i, long int j)
{
	long int ij_index;
//	double ret[TDSIZE];
	ddfloat ret;

	ij_index = mat->real_col_dim * i + j;

	ret.val[0] = mat->element[0][ij_index];
	ret.val[1] = mat->element[1][ij_index];
	//ret.val[2] = mat->element[2][ij_index];

	return ret;
} 
#define GET_TDMATRIX_IJ_DD(mat, i, j) ((get_tdmatrix_ij_tdfloat((mat), (i), (j)).val))
#define get_tdmatrix_ij_dd(mat, i, j) ((get_tdmatrix_ij_tdfloat((mat), (i), (j)).val))

// set_tdmatrix_ij
static inline void set_tdmatrix_ij(TDMatrix mat, long int i, long int j, double *val) // val[TDSIZE]
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[0][ij_index] = val[0];
	mat->element[1][ij_index] = val[1];
	mat->element[2][ij_index] = val[2];

	return;
} 
#define SET_TDMATRIX_IJ(mat, i, j, val) set_tdmatrix_ij((mat), (i), (j))

// set_tdmatrix_ij_d
static inline void set_tdmatrix_ij_d(TDMatrix mat, long int i, long int j, double val)
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[0][ij_index] = val;
	mat->element[1][ij_index] = 0.0;
	mat->element[2][ij_index] = 0.0;

	return;
} 
#define SET_TDMATRIX_IJ_D(mat, i, j, val) set_tdmatrix_ij_d((mat), (i), (j), (val))
#define SET_TDMATRIX_IJ_UI(mat, i, j, val) set_tdmatrix_ij_d((mat), (i), (j), (double)(val))
#define set_tdmatrix_ij_ui(mat, i, j, val) set_tdmatrix_ij_d((mat), (i), (j), (double)(val))

// set_tdmatrix_ij_dd
static inline void set_tdmatrix_ij_dd(TDMatrix mat, long int i, long int j, double *val)
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[0][ij_index] = val[0];
	mat->element[1][ij_index] = val[1];
	mat->element[2][ij_index] = 0.0;

	return;
} 
#define SET_TDMATRIX_IJ_DD(mat, i, j, val) set_tdmatrix_ij_dd((mat), (i), (j), (val))

// set0_tdmatrix_ij
static inline void set0_tdmatrix_ij(TDMatrix mat, long int i, long int j)
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[0][ij_index] = 0.0;
	mat->element[1][ij_index] = 0.0;
	mat->element[2][ij_index] = 0.0;

	return;
}
#define SET0_TDMATRIX_IJ(mat, i, j) set0_tdmatrix_ij((mat), (i), (j))

// set a zero matrix
//void set0_tdmatrix(TDMatrix mat)
void set0_tdmatrix(TDMatrix mat);

// initialize tdmatrix
TDMatrix init_tdmatrix(long int row_dim, long int col_dim);

// free tdmatrix
void free_tdmatrix(TDMatrix mat);

// print tdmatrix
void print_tdmatrix(TDMatrix mat);

// TDMatrix mat -> tdfloat array
void set_tdfloat_tdmat(tdfloat ret[], int ret_dim, TDMatrix mat);

// tdfloat array -> TDmatrix ret
void set_tdmatrix_tdfloat(TDMatrix ret, tdfloat array[], int array_dim);

/*************************************************/
/* Matrix Caluculations for TDMatrix            */
/*
void normf_tdmatrix(double ret[TDSIZE], TDMatrix mat)
void norm1_tdmatrix(double ret[TDSIZE], TDMatrix mat)
void normi_tdmatrix(double ret[TDSIZE], TDMatrix mat)
void add_tdmatrix(TDMatrix c, TDMatrix a, TDMatrix b);
void sub_tdmatrix(TDMatrix c, TDMatrix a, TDMatrix b);
void mul_tdmatrix(TDMatrix c, TDMatrix a, TDMatrix b);
void mul_tdmatrix_tdvec(TDVector v, TDMatrix a, TDVector vb)
void mul_tdmatrixt_tdvec(TDVector v, TDMatrix a, TDVector vb)
void transpose_tdmatrix(TDMatrix c, TDMatrix a);
void inv_tdmatrix(TDMatrix a);
void subst_mpfmatrux(TDMatrix c, TDMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_tdmatrix(double ret[TDSIZE], TDMatrix mat);

/* 1 Norm of Matrix */
void norm1_tdmatrix(double ret[TDSIZE], TDMatrix mat);

/* c := a + b */
void add_tdmatrix(TDMatrix c, TDMatrix a, TDMatrix b);

/* c := a - b */
void sub_tdmatrix(TDMatrix c, TDMatrix a, TDMatrix b);

/* (QDMatrix)c := (TDMatrix)a - (TDMatrix)b */
//void sub_qdmatrix_tdmat_tdmat(QDMatrix c, TDMatrix a, TDMatrix b);

/* c := sc * a */
void cmul_tdmatrix(TDMatrix c, double sc[TDSIZE], TDMatrix a);

/* c = a^T */
void transpose_tdmatrix(TDMatrix c, TDMatrix a);

/* c := a */
void subst_tdmatrix(TDMatrix c, TDMatrix a);

/* c := -a */
void neg_tdmatrix(TDMatrix c, TDMatrix a);

/* c := a */
//void subst_qdmatrix_tdmat(QDMatrix c, TDMatrix a);

/* c := I */
void setI_tdmatrix(TDMatrix c);

/* v := a * vb */
void mul_tdmatrix_tdvec(TDVector v, TDMatrix a, TDVector vb);

/* v := a^T * vb */
void mul_tdmatrixt_tdvec(TDVector v, TDMatrix a, TDVector vb);

/* a = a^(-1) */
/* square matrix only */
void inv_tdmatrix(TDMatrix a);

// matrix multiplication
// ret := A * B
void mul_tdmatrix(TDMatrix ret, TDMatrix a, TDMatrix b);

// Frobenius norm
void normf_tdmatrix(double ret[TDSIZE], TDMatrix mat);

void relerr_tdvector(double relerr[TDSIZE], TDVector approx_vec, TDVector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_tdvector(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double norm_relerr[TDSIZE], TDVector approx_vec, TDVector true_vec, int norm_type);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_tdmatrix(TDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

// tdmatmul_tdvec
void tdmatmul_tdvec(TDVector ret, TDVector mat_a, TDVector mat_b, int row_dim, int mid_dim, int col_dim);

// tdmatmul_tdvec_ur4
void tdmatmul_tdvec_ur4(TDVector ret, TDVector mat_a, TDVector mat_b, int row_dim, int mid_dim, int col_dim);

// tdmatmul_tdvec_avx2
void tdmatmul_tdvec_avx2(TDVector ret, TDVector mat_a, TDVector mat_b, int row_dim, int mid_dim, int col_dim);

#ifdef __BNC_TDLINEAR_H__
//void _bncavx2_tdadd(tdfloat ret[], tdfloat a[], tdfloat b[], int dim)
//void _bncavx2_tdvadd(TDVector ret, TDVector a, TDVector b, int dim)
//void _bncavx2_tdmul(tdfloat ret[], tdfloat a[], tdfloat b[], int dim)
//void _bncavx2_tdvmul(TDVector ret, TDVector a, TDVector b, int dim)
//#include "tdv_addmul.c"
#endif // __BNC_TDLINEAR_H__

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpf)a */
void subst_mpfvector_tdvec(MPFVector c, TDVector a);

/* c := (dd)a */
void subst_tdvector_mpfvec(TDVector c, MPFVector a);

/* c := (mpf)a */
void subst_mpfmatrix_tdmat(MPFMatrix c, TDMatrix a);

/* c := (dd)a */
void subst_tdmatrix_mpfmat(TDMatrix c, MPFMatrix a);

/* Normwise relative error of vector */
void relerr_tdvector_mpfvec(double relerr[TDSIZE], TDVector approx_vec, MPFVector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_tdvector_mpf(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double norm_relerr[TDSIZE], TDVector approx_vec, MPFVector true_vec, int norm_type);
#endif // USE_GMP

//---------------------------------/
// TD <-> double
//---------------------------------/
/* c := (dd)a */
void subst_tdvector_dvec(TDVector c, DVector a);

/* c := (d)a */
void subst_dvector_tdvec(DVector c, TDVector a);

/* c := (dd)a */
void subst_tdmatrix_dmat(TDMatrix c, DMatrix a);

/* c := (d)a */
void subst_dmatrix_tdmat(DMatrix c, TDMatrix a);

//---------------------------------/
// TD <-> DD
//---------------------------------/
/* c := (dd)a */
void subst_tdvector_ddvec(TDVector c, DDVector a);

/* c := (dd)a */
void subst_ddvector_tdvec(DDVector c, TDVector a);

/* c := (dd)a */
void subst_tdmatrix_ddmat(TDMatrix c, DDMatrix a);

/* c := (dd)a */
void subst_ddmatrix_tdmat(DDMatrix c, TDMatrix a);

//----------------------------------------/
// tdlinear.c
//----------------------------------------/
int TDLUdecomp(TDMatrix a);
int SolveTDLS(TDVector answer, TDMatrix lu, TDVector b);
int TDLUdecompP(TDMatrix a, long int ch[]);
int SolveTDLSP(TDVector answer, TDMatrix lu, TDVector b, long int ch[]);
int TDLUdecompPM(TDMatrix a, long int ch[]);
int SolveTDLSPM(TDVector answer, TDMatrix lu, TDVector b, long int ch[]);

//----------------------------------------/
// tdlu_strassen.c
//----------------------------------------/
int TDLUdecomp_square(TDMatrix a, long int start_index, long int min_dim);
int TDLUdecomp_l21(TDMatrix l21, TDMatrix a, long int start_index, long int min_dim);
int TDLUdecomp_u12(TDMatrix u12, TDMatrix a, long int start_index, long int min_dim);
int TDLUdecomp_a22(TDMatrix a, TDMatrix d22, TDMatrix l21, TDMatrix u12, long int start_index, long int min_dim);
int TDLUdecomp_strassen(TDMatrix a, long int min_dim);
int TDLUdecomp_strassenPM(TDMatrix a, long int ch[], long int min_dim);

#ifdef _OPENMP
int TDLUdecomp_square_omp(TDMatrix a, long int start_index, long int min_dim);
int TDLUdecomp_l21_omp(TDMatrix l21, TDMatrix a, long int start_index, long int min_dim);
int TDLUdecomp_u12_omp(TDMatrix u12, TDMatrix a, long int start_index, long int min_dim);
int TDLUdecomp_a22_omp(TDMatrix a, TDMatrix d22, TDMatrix l21, TDMatrix u12, long int start_index, long int min_dim);
int TDLUdecomp_omp(TDMatrix a);
int TDLUdecompPM_omp(TDMatrix a, long int ch[]);
int TDLUdecomp_strassen_omp(TDMatrix a, long int min_dim);
int TDLUdecomp_strassenPM_omp(TDMatrix a, long int ch[], long int min_dim);
#endif // _OPENMP

//----------------------------------------/
// tdlu_oz.c
//----------------------------------------/
int TDLUdecomp_a22_oz(TDMatrix a, TDMatrix d22, TDMatrix l21, TDMatrix u12, long int start_index, long int min_dim, int max_num_div);
int TDLUdecomp_oz(TDMatrix a, long int min_dim, int max_num_div);
int TDLUdecomp_ozPM(TDMatrix a, long int ch[], long int min_dim, int max_num_div);

#ifdef _OPENMP
int TDLUdecomp_a22_oz_omp(TDMatrix a, TDMatrix d22, TDMatrix l21, TDMatrix u12, long int start_index, long int min_dim, int max_num_div);
int TDLUdecomp_oz_omp(TDMatrix a, long int min_dim, int max_num_div);
int TDLUdecomp_ozPM_omp(TDMatrix a, long int ch[], long int min_dim, int max_num_div);
#endif // _OPENMP

//----------------------------------------/
// gtestmat_td.c
//----------------------------------------/
/* 1. Hilbert Matrix */
void hilbert_tdmatrix(TDMatrix a, long int dim);

/* 2. Lotkin Matrix */
void lotkin_tdmatrix(TDMatrix a, long int dim);

/* 3. Frank Matrix */
void frank_tdmatrix(TDMatrix a, long int dim);

/* 4. Tridiagonal Matrix */
void tridiag_tdmatrix(TDMatrix a, TDVector low_subdiag, TDVector diag, TDVector up_subdiag, long int dim);

/* 5. Integer Symmetrix Random Matrix */
void int_sym_rand_tdmatrix(TDMatrix mat, long int max, long int seed, long int dim);

/* 6. Integer Unsymmetrix Random Matrix */
void int_unsym_rand_tdmatrix(TDMatrix mat, long int max, long int seed, long int dim);

/* 7. Real Diagonal Matrix */
void diag_tdmatrix(TDMatrix mat, TDVector diag, long int dim);

/* 8. Toeplitz Matrix */
void toeplitz_tdmatrix(TDMatrix mat, double gamma_param[TDSIZE], long int dim);

// 9.Pascal Matrix
void pascal_tdmatrix(TDMatrix ret, long int dim);

// 10. I - randmatrix
void im_rand_tdmatrix(TDMatrix ret, unsigned long seed);

//----------------------------------------/
// fread_write.c
//----------------------------------------/
void read_test_linear_eq_td(TDMatrix A, TDVector true_x, TDVector b, long int dim, const char *fname_A, const char *fname_true_x, const char *fname_b);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // define __BNC_TDLINEAR_H__
