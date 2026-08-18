/********************************************************************************/
/* qdlinear.h: Quadruple Double precision Linear Computation Library            */
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
#ifndef __BNC_QDLINEAR_H__
  #define __BNC_QDLINEAR_H__

// Common defs
#include "bnc_common.h"

#include "rdd.h"

#include "dlinear.h"  // Double precision linear computation

#include "ddlinear.h" // DD precision linear computation

#include "tdlinear.h" // TD precision linear computation

#ifdef USE_GMP
#include "gmp.h"
#include "mpfr.h"
#include "mpfr_dtq_sd.h"
#include "mpflinear.h"
#endif //USE_GMP//

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//#ifndef __cplusplus
//#endif // __cplusplus

// QD vector
typedef struct{
	long int dim; // dim <= real_dim
	long int real_dim; // multiplier of _BNC_D_WIDTH
	double *element[QDSIZE];
} qdvector;

typedef qdvector *QDVector;

#ifndef _DEF_BNC_QDVECTOR
#define _DEF_BNC_QDVECTOR_
#endif // _DEF_BNC_QDVECTOR

#if defined(__AVX2__) || defined(__AVX512F__)
#include "avx2/bncavx.h"
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2)
#include "sve2/bncsve2.h"
#include "neon/bncneon.h"   /* SVE2 build keeps NEON for the #elif fallback paths */
#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // __ARM_NEON
#include "neon/bncneon.h"
#endif // defined(__AVX2__) || defined(__AVX512F__)

// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
#elif defined(__AVX512F__) // __AVX512F__
#else // others
#endif // __AVX2__

// get_qdvector_i
static inline qdfloat get_qdvector_i_qdfloat(QDVector vec, long int index)
//static inline double *get_qdvector_i(QDVector vec, long int index)
{
	qdfloat ret;

	ret.val[0] = vec->element[0][index];
	ret.val[1] = vec->element[1][index];
	ret.val[2] = vec->element[2][index];
	ret.val[3] = vec->element[3][index];

	return ret;
}

//	GET_QDVECTOR_I(vec, index))
#define GET_QDVECTOR_I(vec, index) ((get_qdvector_i_qdfloat((vec), (index)).val))
#define get_qdvector_i(vec, index) ((get_qdvector_i_qdfloat((vec), (index)).val))

// get_qdvector_i_dd
static inline ddfloat get_qdvector_i_ddfloat(QDVector vec, long int index)
//static inline double *get_qdvector_i(QDVector vec, long int index)
{
	ddfloat ret;

	ret.val[0] = vec->element[0][index];
	ret.val[1] = vec->element[1][index];
	//ret.val[2] = vec->element[2][index];
	//ret.val[3] = vec->element[3][index];

	return ret;
}
//	GET_QDVECTOR_I_DD(vec, index))
#define GET_QDVECTOR_I_DD(vec, index) ((get_qdvector_i_ddfloat((vec), (index)).val))
#define get_qdvector_i_dd(vec, index) ((get_qdvector_i_ddfloat((vec), (index)).val))

// get_qdvector_i_td
static inline tdfloat get_qdvector_i_tdfloat(QDVector vec, long int index)
//static inline double *get_qdvector_i(QDVector vec, long int index)
{
	tdfloat ret;

	ret.val[0] = vec->element[0][index];
	ret.val[1] = vec->element[1][index];
	ret.val[2] = vec->element[2][index];
	//ret.val[3] = vec->element[3][index];

	return ret;
}
//	GET_QDVECTOR_I_TD(vec, index))
#define GET_QDVECTOR_I_TD(vec, index) ((get_qdvector_i_tdfloat((vec), (index)).val))
#define get_qdvector_i_td(vec, index) ((get_qdvector_i_tdfloat((vec), (index)).val))

// set_qdvector_i
static inline void set_qdvector_i(QDVector vec, long int index, double *val) // val[QDSIZE]
{
	double ret[QDSIZE];

	vec->element[0][index] = val[0];
	vec->element[1][index] = val[1];
	vec->element[2][index] = val[2];
	vec->element[3][index] = val[3];
}
#define SET_QDVECTOR_I(vec, index, val) set_qdvector_i((vec), (index), (val))
// 2025-02-19(Wed) T.Kouya
#define subst_qdvector_i(ret, vec, index) rqd_set((ret), get_qdvector_i((vec), (index)))

// set_ddvector_i_d
static inline void set_qdvector_i_d(QDVector vec, long int index, double val) // val
{
	//double ret[QDSIZE];

	vec->element[0][index] = val;
	vec->element[1][index] = 0.0;
	vec->element[2][index] = 0.0;
	vec->element[3][index] = 0.0;
}
#define SET_QDVECTOR_I_D(vec, index, val) set_qdvector_i_d((vec), (index), (val))

// set_ddvector_i_dd
static inline void set_qdvector_i_dd(QDVector vec, long int index, double *val) // val
{
	//double ret[QDSIZE];

	vec->element[0][index] = val[0];
	vec->element[1][index] = val[1];
	vec->element[2][index] = 0.0;
	vec->element[3][index] = 0.0;
}
#define SET_QDVECTOR_I_DD(vec, index, val) set_qdvector_i_dd((vec), (index), (val))

// set_ddvector_i_td
static inline void set_qdvector_i_td(QDVector vec, long int index, double *val) // val
{
	//double ret[QDSIZE];

	vec->element[0][index] = val[0];
	vec->element[1][index] = val[1];
	vec->element[2][index] = val[2];
	vec->element[3][index] = 0.0;
}
#define SET_QDVECTOR_I_TD(vec, index, val) set_qdvector_i_td((vec), (index), (val))

// set0_qdvector_i
static inline void set0_qdvector_i(QDVector vec, long int index)
{
	double ret[QDSIZE];

	vec->element[0][index] = 0.0;
	vec->element[1][index] = 0.0;
	vec->element[2][index] = 0.0;
	vec->element[3][index] = 0.0;
}
#define SET0_QDVECTOR_I(vec, index) set0_qdvector_i((vec), (index))

// set a zero vector
//void set0_qdvector(QDVector vec)
void set0_qdvector(QDVector vec);

#if defined(USE_GMP) && defined(USE_MPFR)
qdfloat qdrel_diff_array(qdfloat approx_a[], qdfloat approx_b[], int dim, int print_flag);

// Frobenius norm
qdfloat qdnormf(qdfloat array[], int dim);

// generate a text vector: vec(i) := sqrt(sqrt_seed) * (i + 1)
void set_test_qdvector(qdfloat vec[], int sqrt_seed, int dim);

// generate a text matrix: mat(i, j) := sqrt(sqrt_seed) * (i + j - 1)
void set_test_qdmatrix(qdfloat mat[], int sqrt_seed, int row_dim, int col_dim);

#endif // defined(USE_GMP) && defined(USE_MPFR)

// initialize qdvector
QDVector init_qdvector(long int dim);

// free qdvector
void free_qdvector(QDVector vec);

// QDVector vec -> qdfloat array
void set_qdfloat_qdvec(qdfloat ret[], int ret_dim, QDVector vec);

// qdfloat array -> QDVector ret
void set_qdvector_qdfloat(QDVector ret, qdfloat array[], int array_dim);

// print ddvector
void print_qdvector(QDVector vec);

// set_qdvector_i_str
void set_qdvector_i_str(QDVector vec, long int index, const char *str);

/*************************************************/
/* Vector Calculations for QDVector               */
/*
void add_qdvector(QDVector c, QDVector a, QDVector b)
void add2_qdvector(QDVector c, QDVector a)
void sub_qdvector(QDVector c, QDVector a, QDVector b)
void sub2_qdvector(QDVector c, DVector a)
void cmul_qdvector(QDVector c, double val[QDSIZE], QDVector a)
void cmul2_qdvector(QDVector c, double val[QDSIZE])
void add_cmul_qdvector(QDVector c, QDVector a, double val[QDSIZE], QDVector b)
double ip_qdvector(QDVector a, QDVector b)
double norm1_qdvector(QDVector a)
double norm2_qdvector(QDVector a)
double normi_qdvector(QDVector a)
void subst_qdvector(QDVector c, QDVector a)
*/
/*************************************************/
/* c = a + b */
void add_qdvector(QDVector c, QDVector a, QDVector b);

/* c += a */
void add2_qdvector(QDVector c, QDVector a);

/* c = a - b */
void sub_qdvector(QDVector c, QDVector a, QDVector b);

/* c -= a */
void sub2_qdvector(QDVector c, QDVector a);

/* c = val * a */
void cmul_qdvector(QDVector c, double val[QDSIZE], QDVector a);

/* c *= val */
void cmul2_qdvector(QDVector c, double val[QDSIZE]);

/* c = a + val * b */
void add_cmul_qdvector(QDVector c, QDVector a, double val[QDSIZE], QDVector b);

/* c = a - val * b */
void sub_cmul_qdvector(QDVector c, QDVector a, double val[QDSIZE], QDVector b);

/* (a, b) */
void ip_qdvector(double ret[QDSIZE], QDVector a, QDVector b);

/* c := a */
void subst_qdvector(QDVector c, QDVector a);


/* c := -a */
void neg_qdvector(QDVector c, QDVector a);

/* ||a||_1 */
void norm1_qdvector(double ret[QDSIZE], QDVector a);

/* ||a||_infty */
void normi_qdvector(double ret[QDSIZE], QDVector a);

// Euclid norm
void norm2_qdvector(double ret[QDSIZE], QDVector vec);

/* add */
void _bncavx2_qdadd(qdfloat ret[], qdfloat a[], qdfloat b[], int dim);
void _bncavx2_qdvadd(QDVector ret, QDVector a, QDVector b, int dim);

/* mul */
void _bncavx2_qdmul(qdfloat ret[], qdfloat a[], qdfloat b[], int dim);
void _bncavx2_qdvmul(QDVector ret, QDVector a, QDVector b, int dim);

/* div */
void _bncavx2_qddiv(qdfloat ret[], qdfloat a[], qdfloat b[], int dim);
void _bncavx2_qdvdiv(QDVector ret, QDVector a, QDVector b, int dim);

// QD matrix
typedef struct{
	long int row_dim, col_dim;
	long int real_row_dim, real_col_dim; // multiplier of _BNC_D_WIDTH
	double *element[QDSIZE];
} qdmatrix;

typedef qdmatrix *QDMatrix;

// old
//#define get_qdmatrix_ij(mat, i, j) ( *((mat)->element + (i) * (mat)->col_dim + (j)) )
//#define set_qdmatrix_ij(mat, i, j, val) ( *((mat)->element + (i) * (mat)->col_dim + (j)) = (val) )

// new
//#define get_qdmatrix_ij(mat, i, j) ( *((mat)->element + (i) * (mat)->real_col_dim + (j)) )
//#define set_qdmatrix_ij(mat, i, j, val) ( *((mat)->element + (i) * (mat)->real_col_dim + (j)) = (val) )

// get_qdmatrix_ij
static inline qdfloat get_qdmatrix_ij_qdfloat(QDMatrix mat, long int i, long int j)
{
	long int ij_index;
//	double ret[QDSIZE];
	qdfloat ret;

	ij_index = mat->real_col_dim * i + j;

	ret.val[0] = mat->element[0][ij_index];
	ret.val[1] = mat->element[1][ij_index];
	ret.val[2] = mat->element[2][ij_index];
	ret.val[3] = mat->element[3][ij_index];

	return ret;
} 
#define GET_QDMATRIX_IJ(mat, i, j) ((get_qdmatrix_ij_qdfloat((mat), (i), (j)).val))
#define get_qdmatrix_ij(mat, i, j) ((get_qdmatrix_ij_qdfloat((mat), (i), (j)).val))

// get_qdmatrix_ij_dd
static inline ddfloat get_qdmatrix_ij_ddfloat(QDMatrix mat, long int i, long int j)
{
	long int ij_index;
//	double ret[QDSIZE];
	ddfloat ret;

	ij_index = mat->real_col_dim * i + j;

	ret.val[0] = mat->element[0][ij_index];
	ret.val[1] = mat->element[1][ij_index];
	//ret.val[2] = mat->element[2][ij_index];
	//ret.val[3] = mat->element[3][ij_index];

	return ret;
} 
#define GET_QDMATRIX_IJ_DD(mat, i, j) ((get_qdmatrix_ij_ddfloat((mat), (i), (j)).val))
#define get_qdmatrix_ij_dd(mat, i, j) ((get_qdmatrix_ij_ddfloat((mat), (i), (j)).val))

// get_qdmatrix_ij_td
static inline tdfloat get_qdmatrix_ij_tdfloat(QDMatrix mat, long int i, long int j)
{
	long int ij_index;
//	double ret[QDSIZE];
	tdfloat ret;

	ij_index = mat->real_col_dim * i + j;

	ret.val[0] = mat->element[0][ij_index];
	ret.val[1] = mat->element[1][ij_index];
	ret.val[2] = mat->element[2][ij_index];
	//ret.val[3] = mat->element[3][ij_index];

	return ret;
} 
#define GET_QDMATRIX_IJ_TD(mat, i, j) ((get_qdmatrix_ij_tdfloat((mat), (i), (j)).val))
#define get_qdmatrix_ij_td(mat, i, j) ((get_qdmatrix_ij_tdfloat((mat), (i), (j)).val))

// set_qdmatrix_ij
static inline void set_qdmatrix_ij(QDMatrix mat, long int i, long int j, double *val) // val[QDSIZE]
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[0][ij_index] = val[0];
	mat->element[1][ij_index] = val[1];
	mat->element[2][ij_index] = val[2];
	mat->element[3][ij_index] = val[3];

	return;
} 
#define SET_QDMATRIX_IJ(mat, i, j, val) set_qdmatrix_ij((mat), (i), (j))

// set_qdmatrix_ij_d
static inline void set_qdmatrix_ij_d(QDMatrix mat, long int i, long int j, double val)
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[0][ij_index] = val;
	mat->element[1][ij_index] = 0.0;
	mat->element[2][ij_index] = 0.0;
	mat->element[3][ij_index] = 0.0;

	return;
} 
#define SET_QDMATRIX_IJ_D(mat, i, j, val) set_qdmatrix_ij_d((mat), (i), (j), (val))
#define SET_QDMATRIX_IJ_UI(mat, i, j, val) set_qdmatrix_ij_d((mat), (i), (j), (double)(val))
#define set_qdmatrix_ij_ui(mat, i, j, val) set_qdmatrix_ij_d((mat), (i), (j), (double)(val))

// set_qdmatrix_ij_dd
static inline void set_qdmatrix_ij_dd(QDMatrix mat, long int i, long int j, double *val)
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[0][ij_index] = val[0];
	mat->element[1][ij_index] = val[1];
	mat->element[2][ij_index] = 0.0;
	mat->element[3][ij_index] = 0.0;

	return;
} 
#define SET_QDMATRIX_IJ_DD(mat, i, j, val) set_qdmatrix_ij_dd((mat), (i), (j), (val))

// set_qdmatrix_ij_td
static inline void set_qdmatrix_ij_td(QDMatrix mat, long int i, long int j, double *val)
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[0][ij_index] = val[0];
	mat->element[1][ij_index] = val[1];
	mat->element[2][ij_index] = val[2];
	mat->element[3][ij_index] = 0.0;

	return;
} 
#define SET_QDMATRIX_IJ_TD(mat, i, j, val) set_qdmatrix_ij_td((mat), (i), (j), (val))

// set0_qdmatrix_ij
static inline void set0_qdmatrix_ij(QDMatrix mat, long int i, long int j)
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[0][ij_index] = 0.0;
	mat->element[1][ij_index] = 0.0;
	mat->element[2][ij_index] = 0.0;
	mat->element[3][ij_index] = 0.0;

	return;
}
#define SET0_QDMATRIX_IJ(mat, i, j) set0_qdmatrix_ij((mat), (i), (j))

// set a zero matrix
//void set0_qdmatrix(QDMatrix mat)
void set0_qdmatrix(QDMatrix mat);

// initialize qdmatrix
QDMatrix init_qdmatrix(long int row_dim, long int col_dim);

// free qdmatrix
void free_qdmatrix(QDMatrix mat);

// print qdmatrix
void print_qdmatrix(QDMatrix mat);

// QDMatrix mat -> qdfloat array
void set_qdfloat_qdmat(qdfloat ret[], int ret_dim, QDMatrix mat);

// qdfloat array -> QDmatrix ret
void set_qdmatrix_qdfloat(QDMatrix ret, qdfloat array[], int array_dim);

/*************************************************/
/* Matrix Caluculations for QDMatrix            */
/*
void normf_qdmatrix(double ret[QDSIZE], QDMatrix mat)
void norm1_qdmatrix(double ret[QDSIZE], QDMatrix mat)
void normi_qdmatrix(double ret[QDSIZE], QDMatrix mat)
void add_qdmatrix(QDMatrix c, QDMatrix a, QDMatrix b);
void sub_qdmatrix(QDMatrix c, QDMatrix a, QDMatrix b);
void mul_qdmatrix(QDMatrix c, QDMatrix a, QDMatrix b);
void mul_qdmatrix_tdvec(QDVector v, QDMatrix a, QDVector vb)
void mul_qdmatrixt_tdvec(QDVector v, QDMatrix a, QDVector vb)
void transpose_qdmatrix(QDMatrix c, QDMatrix a);
void inv_qdmatrix(QDMatrix a);
void subst_qdmatrix(QDMatrix c, QDMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_qdmatrix(double ret[QDSIZE], QDMatrix mat);

/* 1 Norm of Matrix */
void norm1_qdmatrix(double ret[QDSIZE], QDMatrix mat);

/* c := a + b */
void add_qdmatrix(QDMatrix c, QDMatrix a, QDMatrix b);

/* c := a - b */
void sub_qdmatrix(QDMatrix c, QDMatrix a, QDMatrix b);

/* (QDMatrix)c := (QDMatrix)a - (QDMatrix)b */
//void sub_qdmatrix_tdmat_tdmat(QDMatrix c, QDMatrix a, QDMatrix b);

/* c := sc * a */
void cmul_qdmatrix(QDMatrix c, double sc[QDSIZE], QDMatrix a);

/* c = a^T */
void transpose_qdmatrix(QDMatrix c, QDMatrix a);

/* c := a */
void subst_qdmatrix(QDMatrix c, QDMatrix a);

/* c := -a */
void neg_qdmatrix(QDMatrix c, QDMatrix a);

/* c := a */
//void subst_qdmatrix_qdmat(QDMatrix c, QDMatrix a);

/* c := I */
void setI_qdmatrix(QDMatrix c);

/* v := a * vb */
void mul_qdmatrix_qdvec(QDVector v, QDMatrix a, QDVector vb);

/* v := a^T * vb */
void mul_qdmatrixt_qdvec(QDVector v, QDMatrix a, QDVector vb);

/* a = a^(-1) */
/* square matrix only */
void inv_qdmatrix(QDMatrix a);

// matrix multiplication
// ret := A * B
void mul_qdmatrix(QDMatrix ret, QDMatrix a, QDMatrix b);

// Frobenius norm
void normf_qdmatrix(double ret[QDSIZE], QDMatrix mat);

/* Normwise relative error of vector */
void relerr_qdvector(double relerr[QDSIZE], QDVector approx_vec, QDVector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_qdvector(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double norm_relerr[QDSIZE], QDVector approx_vec, QDVector true_vec, int norm_type);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_qdmatrix(QDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

// qdmatmul_tdvec
void qdmatmul_qdvec(QDVector ret, QDVector mat_a, QDVector mat_b, int row_dim, int mid_dim, int col_dim);

// qdmatmul_tdvec_ur4
void qdmatmul_qdvec_ur4(QDVector ret, QDVector mat_a, QDVector mat_b, int row_dim, int mid_dim, int col_dim);

// qdmatmul_tdvec_avx2
void qdmatmul_qdvec_avx2(QDVector ret, QDVector mat_a, QDVector mat_b, int row_dim, int mid_dim, int col_dim);

#ifdef __BNC_TDLINEAR_H__
//void _bncavx2_tdadd(tdfloat ret[], tdfloat a[], tdfloat b[], int dim)
//void _bncavx2_tdvadd(QDVector ret, QDVector a, QDVector b, int dim)
//void _bncavx2_tdmul(tdfloat ret[], tdfloat a[], tdfloat b[], int dim)
//void _bncavx2_tdvmul(QDVector ret, QDVector a, QDVector b, int dim)
//#include "tdv_addmul.c"
#endif // __BNC_TDLINEAR_H__

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpf)a */
void subst_mpfvector_qdvec(MPFVector c, QDVector a);

/* c := (dd)a */
void subst_qdvector_mpfvec(QDVector c, MPFVector a);

/* c := (mpf)a */
void subst_mpfmatrix_qdmat(MPFMatrix c, QDMatrix a);

/* c := (dd)a */
void subst_qdmatrix_mpfmat(QDMatrix c, MPFMatrix a);

/* Normwise relative error of vector */
void relerr_qdvector_mpfvec(double relerr[QDSIZE], QDVector approx_vec, MPFVector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_qdvector_mpf(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double norm_relerr[QDSIZE], QDVector approx_vec, MPFVector true_vec, int norm_type);
#endif // USE_GMP

/* c := (qd)a */
void subst_qdvector_dvec(QDVector c, DVector a);

/* c := (d)a */
void subst_dvector_qdvec(DVector c, QDVector a);

/* c := (qd)a */
void subst_qdmatrix_dmat(QDMatrix c, DMatrix a);

/* c := (d)a */
void subst_dmatrix_qdmat(DMatrix c, QDMatrix a);


/* c := (dd)a */
void subst_qdvector_ddvec(QDVector c, DDVector a);

/* c := (dd)a */
void subst_ddvector_qdvec(DDVector c, QDVector a);

/* c := (dd)a */
void subst_qdmatrix_ddmat(QDMatrix c, DDMatrix a);

/* c := (dd)a */
void subst_ddmatrix_qdmat(DDMatrix c, QDMatrix a);

/* c := (td)a */
void subst_tdvector_qdvec(TDVector c, QDVector a);

/* c := (qd)a */
void subst_qdvector_tdvec(QDVector c, TDVector a);

/* c := (TD)a */
void subst_tdmatrix_qdmat(TDMatrix c, QDMatrix a);

/* c := a */
void subst_qdmatrix_tdmat(QDMatrix c, TDMatrix a);

/* (QDMatrix)c := (TDMatrix)a - (TDMatrix)b */
void sub_qdmatrix_tdmat_tdmat(QDMatrix c, TDMatrix a, TDMatrix b);



int QDLUdecomp(QDMatrix a);
int SolveQDLS(QDVector answer, QDMatrix lu, QDVector b);
int QDLUdecompP(QDMatrix a, long int ch[]);
int SolveQDLSP(QDVector answer, QDMatrix lu, QDVector b, long int ch[]);
int QDLUdecompPM(QDMatrix a, long int ch[]);
int SolveQDLSPM(QDVector answer, QDMatrix lu, QDVector b, long int ch[]);

//--------------------------------/
// qdlu_strassen.c
//--------------------------------/
int QDLUdecomp_square(QDMatrix a, long int start_index, long int min_dim);
int QDLUdecomp_l21(QDMatrix l21, QDMatrix a, long int start_index, long int min_dim);
int QDLUdecomp_u12(QDMatrix u12, QDMatrix a, long int start_index, long int min_dim);
int QDLUdecomp_a22(QDMatrix a, QDMatrix d22, QDMatrix l21, QDMatrix u12, long int start_index, long int min_dim);
int QDLUdecomp_strassen(QDMatrix a, long int min_dim);
int QDLUdecomp_strassenPM(QDMatrix a, long int ch[], long int min_dim);

#ifdef _OPENMP
int QDLUdecomp_omp(QDMatrix a);
int QDLUdecomp_square_omp(QDMatrix a, long int start_index, long int min_dim);
int QDLUdecomp_l21_omp(QDMatrix l21, QDMatrix a, long int start_index, long int min_dim);
int QDLUdecomp_u12_omp(QDMatrix u12, QDMatrix a, long int start_index, long int min_dim);
int QDLUdecomp_a22_omp(QDMatrix a, QDMatrix d22, QDMatrix l21, QDMatrix u12, long int start_index, long int min_dim);
int QDLUdecomp_strassen_omp(QDMatrix a, long int min_dim);
int QDLUdecomp_strassenPM_omp(QDMatrix a, long int ch[], long int min_dim);
int QDLUdecompPM_omp(QDMatrix a, long int ch[]);
#endif // _OPENMP

//--------------------------------/
// qdlu_oz.c
//--------------------------------/
int QDLUdecomp_a22_oz(QDMatrix a, QDMatrix d22, QDMatrix l21, QDMatrix u12, long int start_index, long int min_dim, int max_num_div);
int QDLUdecomp_oz(QDMatrix a, long int min_dim, int max_num_div);
int QDLUdecomp_ozPM(QDMatrix a, long int ch[], long int min_dim, int max_num_div);

#ifdef _OPENMP
int QDLUdecomp_a22_oz_omp(QDMatrix a, QDMatrix d22, QDMatrix l21, QDMatrix u12, long int start_index, long int min_dim, int max_num_div);
int QDLUdecomp_oz_omp(QDMatrix a, long int min_dim, int max_num_div);
int QDLUdecomp_ozPM_omp(QDMatrix a, long int ch[], long int min_dim, int max_num_div);
#endif // _OPENMP

//--------------------------------/
// fread_write.c
//--------------------------------/
// 2021-07-15(Wed) T.Kouya
// read problem from file
void read_test_linear_eq_qd(QDMatrix A, QDVector true_x, QDVector b, long int dim, const char *fname_A, const char *fname_true_x, const char *fname_b);

//--------------------------------/
// gtestmat_qd.c
//--------------------------------/
/* 1. Hilbert Matrix */
void hilbert_qdmatrix(QDMatrix a, long int dim);

/* 2. Lotkin Matrix */
void lotkin_qdmatrix(QDMatrix a, long int dim);

/* 3. Frank Matrix */
void frank_qdmatrix(QDMatrix a, long int dim);

/* 4. Tridiagonal Matrix */
void tridiag_qdmatrix(QDMatrix a, QDVector low_subdiag, QDVector diag, QDVector up_subdiag, long int dim);

/* 5. Integer Symmetrix Random Matrix */
void int_sym_rand_qdmatrix(QDMatrix mat, long int max, long int seed, long int dim);

/* 6. Integer Unsymmetrix Random Matrix */
void int_unsym_rand_qdmatrix(QDMatrix mat, long int max, long int seed, long int dim);

/* 7. Real Diagonal Matrix */
void diag_qdmatrix(QDMatrix mat, QDVector diag, long int dim);

/* 8. Toeplitz Matrix */
void toeplitz_qdmatrix(QDMatrix mat, double gamma_param[QDSIZE], long int dim);

// 9. Pascal Matrix
void pascal_qdmatrix(QDMatrix ret, long int dim);

// 10. I - randmatrix
void im_rand_qdmatrix(QDMatrix ret, unsigned long seed);

/* c := a */
void subst_qdmatrix_tdmat(QDMatrix c, TDMatrix a);

/* (QDMatrix)c := (TDMatrix)a - (TDMatrix)b */
void sub_qdmatrix_tdmat_tdmat(QDMatrix c, TDMatrix a, TDMatrix b);

#ifdef __cplusplus
}
#endif

#endif // define __BNC_QDLINEAR_H__
