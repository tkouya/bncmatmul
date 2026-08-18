/********************************************************************************/
/* qslinear.h: Quadruple Single precision Linear Computation Library            */
/* Copyright (C) 2021 Tomonori Kouya                                            */
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
#ifndef __BNC_QSLINEAR_H__
#define __BNC_QSLINEAR_H__

// Common defs
#include "bnc_common.h"

#include "rds.h"

#include "flinear.h"  // Single precision linear computation

#include "dslinear.h" // DS precision linear computation

#include "tslinear.h" // TS precision linear computation

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

// qsrel_diff
/*inline static qsfloat qsrel_diff(qsfloat a, qsfloat b)
{
    qsfloat rel_diff, abs_a;

    //rel_diff = fabs(a - b);
    rqs_sub(rel_diff.val, a.val, b.val);
    rqs_abs(rel_diff.val, rel_diff.val);

    //if(a != 0.0)
    if(rqs_cmp_ui(a.val, 0UL) != 0)
    {
//        rel_diff /= fabs(a);
        rqs_abs(abs_a.val, a.val);
        rqs_div(rel_diff.val, rel_diff.val, a.val);
    }

    return rel_diff;
}
*/
qsfloat qsrel_diff_array(qsfloat approx_a[], qsfloat approx_b[], int dim, int print_flag);

// QS vector
typedef struct{
	long int dim; // dim <= real_dim
	long int real_dim; // multiplier of _BNC_S_WIDTH
	float *element[QSSIZE];
} qsvector;

typedef qsvector *QSVector;
#ifndef _DEF_BNC_QSVECTOR
#define _DEF_BNC_QSVECTOR_
#endif // _DEF_BNC_QSVECTOR

#if defined(__AVX2__) || defined(__AVX512F__)
#include "avx2/bncavx.h"
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2)
#include "sve2/bncsve2.h"
#include "neon/bncneon.h"   /* SVE2 build keeps NEON for the #elif fallback paths */
#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON)
#include "neon/bncneon.h"
#endif // defined(__AVX2__) || defined(__AVX512F__)

// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
#elif defined(__AVX512F__) // __AVX512F__
#else // others
#endif // __AVX2__

// get_qsvector_i
static inline qsfloat get_qsvector_i_qsfloat(QSVector vec, long int index)
//static inline float *get_qsvector_i(QSVector vec, long int index)
{
	qsfloat ret;

	ret.val[0] = vec->element[0][index];
	ret.val[1] = vec->element[1][index];
	ret.val[2] = vec->element[2][index];
	ret.val[3] = vec->element[3][index];

	return ret;
}

//	GET_QSVECTOR_I(vec, index))
#define GET_QSVECTOR_I(vec, index) ((get_qsvector_i_qsfloat((vec), (index)).val))
#define get_qsvector_i(vec, index) ((get_qsvector_i_qsfloat((vec), (index)).val))

// set_qsvector_i
static inline void set_qsvector_i(QSVector vec, long int index, float *val) // val[QSSIZE]
{
	float ret[QSSIZE];

	vec->element[0][index] = val[0];
	vec->element[1][index] = val[1];
	vec->element[2][index] = val[2];
	vec->element[3][index] = val[3];
}
#define SET_QSVECTOR_I(vec, index, val) set_qsvector_i((vec), (index), (val))

// set_ddvector_i_f
static inline void set_qsvector_i_f(QSVector vec, long int index, float val) // val
{
	float ret[QSSIZE];

	vec->element[0][index] = val;
	vec->element[1][index] = 0.0f;
	vec->element[2][index] = 0.0f;
	vec->element[3][index] = 0.0f;
}
#define SET_QSVECTOR_I_F(vec, index, val) set_qsvector_i_f((vec), (index), (val))

// set0_qsvector_i
static inline void set0_qsvector_i(QSVector vec, long int index)
{
	float ret[QSSIZE];

	vec->element[0][index] = 0.0f;
	vec->element[1][index] = 0.0f;
	vec->element[2][index] = 0.0f;
	vec->element[3][index] = 0.0f;
}
#define SET0_QSVECTOR_I(vec, index) set0_qsvector_i((vec), (index))

// set a zero vector
//void set0_qsvector(QSVector vec)
void set0_qsvector(QSVector vec);

// qsrel_diff
qsfloat qsrel_diff(qsfloat a, qsfloat b);

#if defined(USE_GMP) && defined(USE_MPFR)
qsfloat qsrel_diff_array(qsfloat approx_a[], qsfloat approx_b[], int dim, int print_flag);

// Frobenius norm
qsfloat qsnormf(qsfloat array[], int dim);

// generate a text vector: vec(i) := sqrt(sqrt_seed) * (i + 1)
void set_test_qsvector(qsfloat vec[], int sqrt_seed, int dim);

// generate a text matrix: mat(i, j) := sqrt(sqrt_seed) * (i + j - 1)
void set_test_qsmatrix(qsfloat mat[], int sqrt_seed, int row_dim, int col_dim);

#endif // defined(USE_GMP) && defined(USE_MPFR)

// initialize qsvector
QSVector init_qsvector(long int dim);

// free qsvector
void free_qsvector(QSVector vec);

// QSVector vec -> qsfloat array
void set_qsfloat_qsvec(qsfloat ret[], int ret_dim, QSVector vec);

// qsfloat array -> QSVector ret
void set_qsvector_qsfloat(QSVector ret, qsfloat array[], int array_dim);

// print ddvector
void print_qsvector(QSVector vec);

// set_qsvector_i_str
void set_qsvector_i_str(QSVector vec, long int index, const char *str);

/*************************************************/
/* Vector Calculations for QSVector               */
/*
void add_qsvector(QSVector c, QSVector a, QSVector b)
void add2_qsvector(QSVector c, QSVector a)
void sub_qsvector(QSVector c, QSVector a, QSVector b)
void sub2_qsvector(QSVector c, DVector a)
void cmul_qsvector(QSVector c, float val[QSSIZE], QSVector a)
void cmul2_qsvector(QSVector c, float val[QSSIZE])
void add_cmul_qsvector(QSVector c, QSVector a, float val[QSSIZE], QSVector b)
float ip_qsvector(QSVector a, QSVector b)
float norm1_qsvector(QSVector a)
float norm2_qsvector(QSVector a)
float normi_qsvector(QSVector a)
void subst_qsvector(QSVector c, QSVector a)
*/
/*************************************************/
/* c = a + b */
void add_qsvector(QSVector c, QSVector a, QSVector b);

/* c += a */
void add2_qsvector(QSVector c, QSVector a);

/* c = a - b */
void sub_qsvector(QSVector c, QSVector a, QSVector b);

/* c -= a */
void sub2_qsvector(QSVector c, QSVector a);

/* c = val * a */
void cmul_qsvector(QSVector c, float val[QSSIZE], QSVector a);

/* c *= val */
void cmul2_qsvector(QSVector c, float val[QSSIZE]);

/* c = a + val * b */
void add_cmul_qsvector(QSVector c, QSVector a, float val[QSSIZE], QSVector b);
void sub_cmul_qsvector(QSVector c, QSVector a, float val[QSSIZE], QSVector b);

/* (a, b) */
void ip_qsvector(float ret[QSSIZE], QSVector a, QSVector b);

/* c := a */
void subst_qsvector(QSVector c, QSVector a);

/* c := -a */
void neg_qsvector(QSVector c, QSVector a);

/* ||a||_1 */
void norm1_qsvector(float ret[QSSIZE], QSVector a);

/* ||a||_infty */
void normi_qsvector(float ret[QSSIZE], QSVector a);

// Euclid norm
void norm2_qsvector(float ret[QSSIZE], QSVector vec);

/* add */
void _bncavx2_qsadd(qsfloat ret[], qsfloat a[], qsfloat b[], int dim);
void _bncavx2_qsvadd(QSVector ret, QSVector a, QSVector b, int dim);

/* mul */
void _bncavx2_qsmul(qsfloat ret[], qsfloat a[], qsfloat b[], int dim);
void _bncavx2_qsvmul(QSVector ret, QSVector a, QSVector b, int dim);

/* div */
void _bncavx2_qsdiv(qsfloat ret[], qsfloat a[], qsfloat b[], int dim);
void _bncavx2_qsvdiv(QSVector ret, QSVector a, QSVector b, int dim);

// QS matrix
typedef struct{
	long int row_dim, col_dim;
	long int real_row_dim, real_col_dim; // multiplier of _BNC_S_WIDTH
	float *element[QSSIZE];
} qsmatrix;

typedef qsmatrix *QSMatrix;

// old
//#define get_qsmatrix_ij(mat, i, j) ( *((mat)->element + (i) * (mat)->col_dim + (j)) )
//#define set_qsmatrix_ij(mat, i, j, val) ( *((mat)->element + (i) * (mat)->col_dim + (j)) = (val) )

// new
//#define get_qsmatrix_ij(mat, i, j) ( *((mat)->element + (i) * (mat)->real_col_dim + (j)) )
//#define set_qsmatrix_ij(mat, i, j, val) ( *((mat)->element + (i) * (mat)->real_col_dim + (j)) = (val) )

// get_qsmatrix_ij
static inline qsfloat get_qsmatrix_ij_qsfloat(QSMatrix mat, long int i, long int j)
{
	long int ij_index;
//	float ret[QSSIZE];
	qsfloat ret;

	ij_index = mat->real_col_dim * i + j;

	ret.val[0] = mat->element[0][ij_index];
	ret.val[1] = mat->element[1][ij_index];
	ret.val[2] = mat->element[2][ij_index];
	ret.val[3] = mat->element[3][ij_index];

	return ret;
} 
#define GET_QSMATRIX_IJ(mat, i, j) ((get_qsmatrix_ij_qsfloat((mat), (i), (j)).val))
#define get_qsmatrix_ij(mat, i, j) ((get_qsmatrix_ij_qsfloat((mat), (i), (j)).val))

// set_qsmatrix_ij
static inline void set_qsmatrix_ij(QSMatrix mat, long int i, long int j, float *val) // val[QSSIZE]
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[0][ij_index] = val[0];
	mat->element[1][ij_index] = val[1];
	mat->element[2][ij_index] = val[2];
	mat->element[3][ij_index] = val[3];

	return;
} 
#define SET_QSMATRIX_IJ(mat, i, j, val) set_qsmatrix_ij((mat), (i), (j))

// set_qsmatrix_ij_d
static inline void set_qsmatrix_ij_f(QSMatrix mat, long int i, long int j, float val)
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[0][ij_index] = val;
	mat->element[1][ij_index] = 0.0f;
	mat->element[2][ij_index] = 0.0f;
	mat->element[3][ij_index] = 0.0f;

	return;
} 
#define SET_QSMATRIX_IJ_F(mat, i, j, val) set_qsmatrix_ij_f((mat), (i), (j), (val))
#define SET_QSMATRIX_IJ_UI(mat, i, j, val) set_qsmatrix_ij_f((mat), (i), (j), (float)(val))
#define set_qsmatrix_ij_ui(mat, i, j, val) set_qsmatrix_ij_f((mat), (i), (j), (float)(val))

// set0_qsmatrix_ij
static inline void set0_qsmatrix_ij(QSMatrix mat, long int i, long int j)
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[0][ij_index] = 0.0f;
	mat->element[1][ij_index] = 0.0f;
	mat->element[2][ij_index] = 0.0f;
	mat->element[3][ij_index] = 0.0f;

	return;
}
#define SET0_QSMATRIX_IJ(mat, i, j) set0_qsmatrix_ij((mat), (i), (j))

// set a zero matrix
//void set0_qsmatrix(QSMatrix mat)
void set0_qsmatrix(QSMatrix mat);

// initialize qsmatrix
QSMatrix init_qsmatrix(long int row_dim, long int col_dim);

// free qsmatrix
void free_qsmatrix(QSMatrix mat);

// print qsmatrix
void print_qsmatrix(QSMatrix mat);

// QSMatrix mat -> qsfloat array
void set_qsfloat_qsmat(qsfloat ret[], int ret_dim, QSMatrix mat);

// qsfloat array -> QDmatrix ret
void set_qsmatrix_qsfloat(QSMatrix ret, qsfloat array[], int array_dim);

/*************************************************/
/* Matrix Caluculations for QSMatrix            */
/*
void normf_qsmatrix(float ret[QSSIZE], QSMatrix mat)
void norm1_qsmatrix(float ret[QSSIZE], QSMatrix mat)
void normi_qsmatrix(float ret[QSSIZE], QSMatrix mat)
void add_qsmatrix(QSMatrix c, QSMatrix a, QSMatrix b);
void sub_qsmatrix(QSMatrix c, QSMatrix a, QSMatrix b);
void mul_qsmatrix(QSMatrix c, QSMatrix a, QSMatrix b);
void mul_qsmatrix_tdvec(QSVector v, QSMatrix a, QSVector vb)
void mul_qsmatrixt_tdvec(QSVector v, QSMatrix a, QSVector vb)
void transpose_qsmatrix(QSMatrix c, QSMatrix a);
void inv_qsmatrix(QSMatrix a);
void subst_mpfmatrux(QSMatrix c, QSMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_qsmatrix(float ret[QSSIZE], QSMatrix mat);

/* 1 Norm of Matrix */
void norm1_qsmatrix(float ret[QSSIZE], QSMatrix mat);

/* c := a + b */
void add_qsmatrix(QSMatrix c, QSMatrix a, QSMatrix b);

/* c := a - b */
void sub_qsmatrix(QSMatrix c, QSMatrix a, QSMatrix b);

/* (QSMatrix)c := (QSMatrix)a - (QSMatrix)b */
//void sub_qsmatrix_tdmat_tdmat(QSMatrix c, QSMatrix a, QSMatrix b);

/* c := sc * a */
void cmul_qsmatrix(QSMatrix c, float sc[QSSIZE], QSMatrix a);

/* c = a^T */
void transpose_qsmatrix(QSMatrix c, QSMatrix a);

/* c := a */
void subst_qsmatrix(QSMatrix c, QSMatrix a);

/* c := a */
//void subst_qsmatrix_qsmat(QSMatrix c, QSMatrix a);

/* c := I */
void setI_qsmatrix(QSMatrix c);

/* v := a * vb */
void mul_qsmatrix_qsvec(QSVector v, QSMatrix a, QSVector vb);

/* v := a^T * vb */
void mul_qsmatrixt_qsvec(QSVector v, QSMatrix a, QSVector vb);

/* a = a^(-1) */
/* square matrix only */
void inv_qsmatrix(QSMatrix a);

// matrix multiplication
// ret := A * B
void mul_qsmatrix(QSMatrix ret, QSMatrix a, QSMatrix b);

// Frobenius norm
void normf_qsmatrix(float ret[QSSIZE], QSMatrix mat);

/* Elementwise relative errors of vector */
void relerr_element_qsvector(float max_relerr[QSSIZE], float min_relerr[QSSIZE], float norm_relerr[QSSIZE], QSVector approx_vec, QSVector true_vec, int norm_type);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_qsmatrix(QSMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

// qsmatmul_tdvec
void qsmatmul_qsvec(QSVector ret, QSVector mat_a, QSVector mat_b, int row_dim, int mid_dim, int col_dim);

// qsmatmul_tdvec_ur4
void qsmatmul_qsvec_ur4(QSVector ret, QSVector mat_a, QSVector mat_b, int row_dim, int mid_dim, int col_dim);

// qsmatmul_tdvec_avx2
void qsmatmul_qsvec_avx2(QSVector ret, QSVector mat_a, QSVector mat_b, int row_dim, int mid_dim, int col_dim);

#ifdef __BNC_TDLINEAR_H__
//void _bncavx2_tdadd(tdfloat ret[], tdfloat a[], tdfloat b[], int dim)
//void _bncavx2_tdvadd(QSVector ret, QSVector a, QSVector b, int dim)
//void _bncavx2_tdmul(tdfloat ret[], tdfloat a[], tdfloat b[], int dim)
//void _bncavx2_tdvmul(QSVector ret, QSVector a, QSVector b, int dim)
//#include "tdv_addmul.c"
#endif // __BNC_TDLINEAR_H__

int QSLUdecomp(QSMatrix a);
int SolveQSLS(QSVector answer, QSMatrix lu, QSVector b);
int QSLUdecompP(QSMatrix a, long int ch[]);
int SolveQSLSP(QSVector answer, QSMatrix lu, QSVector b, long int ch[]);
int QSLUdecompPM(QSMatrix a, long int ch[]);
int SolveQSLSPM(QSVector answer, QSMatrix lu, QSVector b, long int ch[]);

/* 1. Hilbert Matrix */
void hilbert_qsmatrix(QSMatrix a, long int dim);

/* 2. Lotkin Matrix */
void lotkin_qsmatrix(QSMatrix a, long int dim);

/* 3. Frank Matrix */
void frank_qsmatrix(QSMatrix a, long int dim);

/* 4. Tridiagonal Matrix */
void tridiag_qsmatrix(QSMatrix a, QSVector low_subdiag, QSVector diag, QSVector up_subdiag, long int dim);

/* 5. Integer Symmetrix Random Matrix */
void int_sym_rand_qsmatrix(QSMatrix mat, long int max, long int seed, long int dim);

/* 6. Integer Unsymmetrix Random Matrix */
void int_unsym_rand_qsmatrix(QSMatrix mat, long int max, long int seed, long int dim);

/* 7. Real Diagonal Matrix */
void diag_qsmatrix(QSMatrix mat, QSVector diag, long int dim);

/* 8. Toeplitz Matrix */
void toeplitz_qsmatrix(QSMatrix mat, float gamma_param[QSSIZE], long int dim);

#include "tdlinear.h" // brings in TDMatrix type

/* c := a */
void subst_qsmatrix_tdmat(QSMatrix c, TDMatrix a);

/* (QSMatrix)c := (TDMatrix)a - (TDMatrix)b */
void sub_qsmatrix_tdmat_tdmat(QSMatrix c, TDMatrix a, TDMatrix b);

#ifdef __cplusplus
}
#endif

#endif // define __BNC_QSLINEAR_H__
