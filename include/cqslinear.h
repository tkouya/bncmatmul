/********************************************************************************/
/* cqdlinear.h: Quadruple-float precision Complex Linear Computation Library   */
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
// define __BNC_CQSLINEAR_H__
#ifndef __BNC_CQSLINEAR_H__
  #define __BNC_CQSLINEAR_H__

#include "rds.h"
#include "rcds.h" // Complex QD arithmetic

//#include "clinear.h"
#include "cdlinear.h"
#ifdef USE_DDLINEAR
#include "cdslinear.h"
#endif // USE_DDLINEAR
#ifdef USE_TDLINEAR
#include "ctslinear.h"
#endif // USE_TDLINEAR
#include "qslinear.h"
//#include "bmatrix.h"

#ifdef USE_GMP
#include "gmp.h"
#include "mpfr.h"
#include "mpfr_dtq_sd.h"
#include "mpflinear.h"
#include "cmpflinear.h"
#endif //USE_GMP//

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//#if defined(__AVX2__) || defined(__AVX512F__)
//#include "bncavx.h"
//#endif // defined(__AVX2__) || defined(__AVX512F__)

// CQD vector
typedef struct
{
    QSVector re; // Real part
    QSVector im; // Imaginary part

} cqsvector;

typedef cqsvector *CQSVector;

// SIMD: AVX2 and AVX-512
//#if defined(__AVX2__) || defined(__AVX512F__) // __AVX2__ or __AVX512F__
//#include "bncavx.h"
//#endif // defined(__AVX2__) || defined(__AVX512F__)

// SIMD : for copy & paste
#if 0 // __AVX2__ (disabled for single-complex)
#elif 0 // __AVX512F__ (disabled)
#else // others
#endif // __AVX2__

// get_cqsvector_i_cqsfloat
static inline cqsfloat get_cqsvector_i_cqsfloat(CQSVector vec, long int index)
{
	//static 
	cqsfloat ret;

	rqs_set(ret.val_re, get_qsvector_i(vec->re, index));
    rqs_set(ret.val_im, get_qsvector_i(vec->im, index));

	return ret;
}
// subst_cqsvector_i
// ret := get_cqsvector_i(vec, index)
static inline void subst_cqsvector_i(cqsfloat *ret, CQSVector vec, long int index)
{
	rqs_set(ret->val_re, get_qsvector_i(vec->re, index));
	rqs_set(ret->val_im, get_qsvector_i(vec->im, index));
} 
// Very dangerous!! 2024-04-18(Thu) T.Kouya
//	GET_CQDVECTOR_I(vec, index))
#define GET_CQDVECTOR_I(vec, index) ((cqsfloat *)&(get_cqsvector_i_cqsfloat((vec), (index))))
//#define get_cqsvector_i(vec, index) ((cqsfloat *)&(get_cqsvector_i_cqsfloat((vec), (index))))
static inline cqsfloat *get_cqsvector_i(CQSVector vec, long int index)
{
	//static is neccesary here! 2024-09-23 T.Kouya
	static cqsfloat ret, *ptr_ret;

	ptr_ret = &ret;

	rqs_set(ptr_ret->val_re, get_qsvector_i(vec->re, index));
	rqs_set(ptr_ret->val_im, get_qsvector_i(vec->im, index));

	return ptr_ret;
}

// set_cqsvector_i
static inline void set_cqsvector_i(CQSVector vec, long int index, cqsfloat *val) // val[QSSIZE]
{
    set_qsvector_i(vec->re, index, val->val_re);
    set_qsvector_i(vec->im, index, val->val_im);
}
#define SET_CQDVECTOR_I(vec, index, val) set_cqsvector_i((vec), (index), (val))

// set_cqsvector_i_d
static inline void set_cqsvector_i_d(CQSVector vec, long int index, float val) // val
{
    set_qsvector_i_f(vec->re, index, val);
    set0_qsvector_i(vec->im, index);
}
#define SET_CQDVECTOR_I_D(vec, index, val) set_cqsvector_i_d((vec), (index), (val))

// set_cqsvector_i_cd
static inline void set_cqsvector_i_cd(CQSVector vec, long int index, float _Complex val) // val
{
    set_qsvector_i_f(vec->re, index, __real__ val); //creal(val));
    set_qsvector_i_f(vec->im, index, __imag__ val); //cimag(val));
}

// MPLAPACK
#if defined(_QD_COMPLEX_H_) && defined(__cplusplus)
// set_cqsvector_i_cd
static inline void set_cqsvector_i_qd_complex(CQSVector vec, long int index, qd_complex val) // val
{
    set_qsvector_i(vec->re, index, val.real().x);
    set_qsvector_i(vec->im, index, val.imag().x);
}
#endif // _QD_COMPLEX_H_ & __cplusplus

// set0_cqsvector_i
static inline void set0_cqsvector_i(CQSVector vec, long int index)
{
	set0_qsvector_i(vec->re, index);
	set0_qsvector_i(vec->im, index);
}
#define SET0_CQDVECTOR_I(vec, index) set0_cqsvector_i((vec), (index))

// initialize CQSVector
CQSVector init_cqsvector(int dimension);

// free CQSVector
void free_cqsvector(CQSVector vec);

// CQSVector vec -> qsfloat array
void set_cqsfloat_cdsvec(cqsfloat ret[], int ret_dim, CQSVector vec);

// qsvector -> cqsvector
void set_cqsvector_qsvec(CQSVector ret, QSVector re_vec, QSVector im_vec);

// qsfloat array -> CQSVector ret
void set_cqsvector_qsfloat(CQSVector ret, qsfloat array[], int array_dim);

// print qsvector
void print_cqsvector(CQSVector vec);

// set a zero vector
void set0_cqsvector(CQSVector vec);

// set_cqsvector_i_str
void set_cqsvector_i_str(CQSVector vec, long int index, const char *str);

/*************************************************/
/* Vector Calculations for CQSVector               */
/*
void add_cqsvector(CQSVector c, CQSVector a, CQSVector b)
void add2_cqsvector(CQSVector c, CQSVector a)
void sub_cqsvector(CQSVector c, CQSVector a, CQSVector b)
void sub2_cqsvector(CQSVector c, CDVector a)
void cmul_cqsvector(CQSVector c, cqsfloat *val, CQSVector a)
void cmul2_cqsvector(CQSVector c, cqsfloat *val)
void add_cmul_cqsvector(CQSVector c, CQSVector a, cqsfloat *val, CQSVector b)
float ip_cqsvector(CQSVector a, CQSVector b)
float norm1_cqsvector(CQSVector a)
float norm2_cqsvector(CQSVector a)
float normi_cqsvector(CQSVector a)
void subst_cqsvector(CQSVector c, CQSVector a)
*/
/*************************************************/
/* c = a + b */
void add_cqsvector(CQSVector c, CQSVector a, CQSVector b);

/* c += a */
void add2_cqsvector(CQSVector c, CQSVector a);

/* c = a - b */
void sub_cqsvector(CQSVector c, CQSVector a, CQSVector b);

/* c -= a */
void sub2_cqsvector(CQSVector c, CQSVector a);

/* c = val * a */
void cmul_cqsvector_4m(CQSVector c, cqsfloat *val, CQSVector a);
void cmul_cqsvector_3m(CQSVector c, cqsfloat *val, CQSVector a);
#ifdef USE_4M
#define cmul_cqsvector cmul_cqsvector_4m
#else // USE_4M
//#define cmul_cqsvector cmul_cqsvector_3m
#define cmul_cqsvector cmul_cqsvector_4m // faster than 3m on DD arithmetic
#endif // USE_4M

/* c *= val */
void cmul2_cqsvector(CQSVector c, cqsfloat *val);

/* c = a + val * b */
void add_cmul_cqsvector(CQSVector c, CQSVector a, cqsfloat *val, CQSVector b);

/* c = a - val * b */
void sub_cmul_cqsvector(CQSVector c, CQSVector a, cqsfloat *val, CQSVector b);

/* (a, b) */
void ip_cqsvector(cqsfloat *ret, CQSVector a, CQSVector b);

/* a^T * b */
void dotp_cqsvector(cqsfloat *ret, CQSVector a, CQSVector b);

/* c := a */
void subst_cqsvector(CQSVector c, CQSVector a);

/* c := -a */
void neg_cqsvector(CQSVector c, CQSVector a);

/* ||a||_1 */
void norm1_cqsvector(float ret[QSSIZE], CQSVector a);

/* ||a||_infty */
void normi_cqsvector(float ret[QSSIZE], CQSVector a);

// Euclid norm
void norm2_cqsvector(float ret[QSSIZE], CQSVector vec);

// print cqsvector
void print_cqsvector(CQSVector vec);

// CQD matrix
typedef struct{
	QSMatrix re; // Real part
	QSMatrix im; // Imaginary part
} cqsmatrix;

typedef cqsmatrix *CQSMatrix;

// get_cqsmatrix_ij
static inline cqsfloat get_cqsmatrix_ij_cqsfloat(CQSMatrix mat, long int i, long int j)
{
	long int ij_index;
	//static 
	cqsfloat ret;

	ij_index = mat->re->real_col_dim * i + j;

	rqs_set(ret.val_re, get_qsmatrix_ij(mat->re, i, j));
	rqs_set(ret.val_im, get_qsmatrix_ij(mat->im, i, j));

	return ret;
} 
// subst_cqsmatrix_ij
// ret := get_cqsmatrix_ij(mat, i, j)
static inline void subst_cqsmatrix_ij(cqsfloat *ret, CQSMatrix mat, long int i, long int j)
{
	long int ij_index;

	rqs_set(ret->val_re, get_qsmatrix_ij(mat->re, i, j));
	rqs_set(ret->val_im, get_qsmatrix_ij(mat->im, i, j));

	//return ret;
} 
// Very dangerous!! 2024-04-18(Thu) T.Kouya
#define GET_CQDMATRIX_IJ(mat, i, j) (&(get_cqsmatrix_ij_cqsfloat((mat), (i), (j))))
//#define get_cqsmatrix_ij(mat, i, j) (&(get_cqsmatrix_ij_cqsfloat((mat), (i), (j))))
static inline cqsfloat *get_cqsmatrix_ij(CQSMatrix mat, long int i, long int j)
{
	long int ij_index;
	//static is neccesary here! 2024-09-23 T.Kouya
	static cqsfloat ret, *ptr_ret;

	ij_index = mat->re->real_col_dim * i + j;
	ptr_ret = &ret;

	rqs_set(ptr_ret->val_re, get_qsmatrix_ij(mat->re, i, j));
	rqs_set(ptr_ret->val_im, get_qsmatrix_ij(mat->im, i, j));

	return ptr_ret;
} 

// set_cqsmatrix_ij
static inline void set_cqsmatrix_ij(CQSMatrix mat, long int i, long int j, cqsfloat *val) // val[QSSIZE]
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_qsmatrix_ij(mat->re, i, j, val->val_re);
	set_qsmatrix_ij(mat->im, i, j, val->val_im);

	return;
} 
#define SET_CQDMATRIX_IJ(mat, i, j, val) set_cqsmatrix_ij((mat), (i), (j), (val))

// For MPBLAS
#ifdef _QD_COMPLEX_H_
static inline void set_cqsmatrix_ij_qd_complex(CQSMatrix mat, long int i, long int j, qd_complex val)
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_qsmatrix_ij(mat->re, i, j, val.real().x);
	set_qsmatrix_ij(mat->im, i, j, val.imag().x);

	return;
} 
#endif // _QD_COMPLEX_H_

// set_cqsmatrix_ij_d
static inline void set_cqsmatrix_ij_d(CQSMatrix mat, long int i, long int j, float val)
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_qsmatrix_ij_f(mat->re, i, j, val);
	set0_qsmatrix_ij(mat->im, i, j);

	return;
} 
#define SET_CQDMATRIX_IJ_D(mat, i, j, val) set_cqsmatrix_ij_d((mat), (i), (j), (val))
#define SET_CQDMATRIX_IJ_UI(mat, i, j, val) set_cqsmatrix_ij_d((mat), (i), (j), (float)(val))
#define set_cqsmatrix_ij_ui(mat, i, j, val) set_cqsmatrix_ij_d((mat), (i), (j), (float)(val))

// set_cqsmatrix_ij_qd
static inline void set_cqsmatrix_ij_qd(CQSMatrix mat, long int i, long int j, float val[QSSIZE])
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_qsmatrix_ij(mat->re, i, j, val);
	set0_qsmatrix_ij(mat->im, i, j);

	return;
} 

// set_cqsmatrix_ij_cd
static inline void set_cqsmatrix_ij_cd(CQSMatrix mat, long int i, long int j, float _Complex val)
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_qsmatrix_ij_f(mat->re, i, j, __real__ val); //creal(val));
	set_qsmatrix_ij_f(mat->im, i, j, __imag__ val); //cimag(val));

	return;
} 

// set0_cqsmatrix_ij
static inline void set0_cqsmatrix_ij(CQSMatrix mat, long int i, long int j)
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set0_qsmatrix_ij(mat->re, i, j);
	set0_qsmatrix_ij(mat->im, i, j);

	return;
}
#define SET0_CQDMATRIX_IJ(mat, i, j) set0_cqsmatrix_ij((mat), (i), (j))

// set a zero matrix
//void set0_cqsmatrix(CQSMatrix mat);
void set0_cqsmatrix(CQSMatrix mat);

// initialize qsvector
CQSMatrix init_cqsmatrix(long int row_dim, long int col_dim);

// free qsvector
void free_cqsmatrix(CQSMatrix mat);

// print qsvector
void print_cqsmatrix(CQSMatrix mat);

// CQSMatrix mat -> cqsfloat array
void set_cqsfloat_cdsmat(cqsfloat ret[], int ret_dim, CQSMatrix mat);

// qsmatrix -> cqsmatrix
void set_cqsmatrix_qsmat(CQSMatrix ret, QSMatrix re_mat, QSMatrix im_mat);

// cqsfloat array -> CQDmatrix ret
void set_cqsmatrix_cqsfloat(CQSMatrix ret, cqsfloat array[], int array_dim);

// matrix multiplication
// ret := A * B
void mul_cqsmatrix_4m(CQSMatrix ret, CQSMatrix a, CQSMatrix b);
void mul_cqsmatrix_3m(CQSMatrix ret, CQSMatrix a, CQSMatrix b);
#ifdef USE_4M
#define mul_cqsmatrix mul_cqsmatrix_4m
#else // USE_4M
#define mul_cqsmatrix mul_cqsmatrix_3m
#endif // USE_4M

// Frobenius norm
void normf_cqsmatrix(float ret[QSSIZE], CQSMatrix mat);

// print normf
void print_normf_cqsmatrix(const char *str, CQSMatrix mat);

/*************************************************/
/* Matrix Caluculations for CQSMatrix            */
/*
void normf_cqsmatrix(float ret[QSSIZE], CQSMatrix mat)
void norm1_cqsmatrix(float ret[QSSIZE], CQSMatrix mat)
void normi_cqsmatrix(float ret[QSSIZE], CQSMatrix mat)
void add_cqsmatrix(CQSMatrix c, CQSMatrix a, CQSMatrix b);
void sub_cqsmatrix(CQSMatrix c, CQSMatrix a, CQSMatrix b);
void mul_cqsmatrix(CQSMatrix c, CQSMatrix a, CQSMatrix b);
void mul_cqsmatrix_cqsvec(CQSVector v, CQSMatrix a, CQSVector vb)
void mul_cqsmatrixt_cqsvec(CQSVector v, CQSMatrix a, CQSVector vb)
void transpose_cqsmatrix(CQSMatrix c, CQSMatrix a);
void inv_cqsmatrix(CQSMatrix a);
void subst_cqsmatrix(CQSMatrix c, CQSMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_cqsmatrix(float ret[QSSIZE], CQSMatrix mat);

/* 1 Norm of Matrix */
void norm1_cqsmatrix(float ret[QSSIZE], CQSMatrix mat);

/* c := a + b */
void add_cqsmatrix(CQSMatrix c, CQSMatrix a, CQSMatrix b);

/* c := a - b */
void sub_cqsmatrix(CQSMatrix c, CQSMatrix a, CQSMatrix b);

/* c := sc * a */
void cmul_cqsmatrix(CQSMatrix c, cqsfloat *sc, CQSMatrix a);

/* c = a^T */
void transpose_cqsmatrix(CQSMatrix c, CQSMatrix a);

/* c := conj(a)^T */
void star_cqsmatrix(CQSMatrix c, CQSMatrix a);

/* c := a */
void subst_cqsmatrix(CQSMatrix c, CQSMatrix a);

/* c := conj(a) */
void conj_cqsmatrix(CQSMatrix c, CQSMatrix a);

/* c := -a */
void neg_cqsmatrix(CQSMatrix c, CQSMatrix a);

/* c := a */
void subst_cqsmatrix(CQSMatrix c, CQSMatrix a);

/* c := I */
void setI_cqsmatrix(CQSMatrix c);

/* v := a * vb */
void mul_cqsmatrix_cqsvec_4m(CQSVector v, CQSMatrix a, CQSVector vb);
void mul_cqsmatrix_cqsvec_3m(CQSVector v, CQSMatrix a, CQSVector vb);
#ifdef USE_4M
#define mul_cqsmatrix_cqsvec mul_cqsmatrix_cqsvec_4m
#else // USE_4M
#define mul_cqsmatrix_cqsvec mul_cqsmatrix_cqsvec_3m
#endif // USE_4M

/* v := a^T * vb */
void mul_cqsmatrixt_cqsvec(CQSVector v, CQSMatrix a, CQSVector vb);

/* v := conj(a)^T * vb */
void mul_cqsmatrixs_cqsvec(CQSVector v, CQSMatrix a, CQSVector vb);

/* a = a^(-1) */
/* square matrix only */
void inv_cqsmatrix(CQSMatrix a);

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpf)a */
void subst_cmpfvector_cqsvec(CMPFVector c, CQSVector a);

/* c := (dd)a */
void subst_cqsvector_cmpfvec(CQSVector c, CMPFVector a);

/* c := (mpf)a */
void subst_cmpfmatrix_cqsmat(CMPFMatrix c, CQSMatrix a);

/* c := (dd)a */
void subst_cqsmatrix_cmpfmat(CQSMatrix c, CMPFMatrix a);

/* Normwise relative error of vector */
void relerr_cqsvector_cmpfvec(float relerr[QSSIZE], CQSVector approx_vec, CMPFVector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_cqsvector_mpf(float max_relerr[QSSIZE], float min_relerr[QSSIZE], float norm_relerr[QSSIZE], CQSVector approx_vec, MPFVector true_vec, int norm_type);
#endif // USE_GMP

/* c := (dd)a */
void subst_cqsvector_cdvec(CQSVector c, CDVector a);

/* c := (d)a */
///void subst_cdvector_cdsvec(CDVector c, CQSVector a);

/* c := (dd)a */
void subst_cqsmatrix_cdmat(CQSMatrix c, CDMatrix a);

/* c := (d)a */
//void subst_cdmatrix_cdsmat(CDMatrix c, CQSMatrix a);


#ifdef USE_DDLINEAR
/* c := (qd)a */
void subst_cqsvector_cdsvec(CQSVector c, CDSVector a);

/* c := (dd)a */
void subst_cdsvector_cqsvec(CDSVector c, CQSVector a);

/* c := (qd)a */
void subst_cqsmatrix_cdsmat(CQSMatrix c, CDSMatrix a);

/* c := (dd)a */
void subst_cdsmatrix_cqsmat(CDSMatrix c, CQSMatrix a);

#endif // USE_DDLINEAR

#ifdef USE_TDLINEAR
/* c := (qd)a */
void subst_cqsvector_ctsvec(CQSVector c, CTSVector a);

/* c := (td)a */
void subst_ctsvector_cqsvec(CTSVector c, CQSVector a);

/* c := (qd)a */
void subst_cqsmatrix_ctsmat(CQSMatrix c, CTSMatrix a);

/* c := (td)a */
void subst_ctsmatrix_cqsmat(CTSMatrix c, CQSMatrix a);
#endif // USE_TDLINEAR

/* Normwise relative error of vector */
void relerr_cqsvector(float relerr[QSSIZE], CQSVector approx_vec, CQSVector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_cqsvector(float max_relerr[QSSIZE], float min_relerr[QSSIZE], float norm_relerr[QSSIZE], CQSVector approx_vec, CQSVector true_vec, int norm_type);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_cqsmatrix(CQSMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

// print_cqsmatrix
void print_cqsmatrix(CQSMatrix mat);

/**************************************/
/* cddlu.c                             */
/**************************************/
int CQSLUdecomp(CQSMatrix a);
int SolveCQSLS(CQSVector answer, CQSMatrix lu, CQSVector b);
int CQSLUdecompP(CQSMatrix a, long int ch[]);
int SolveCQSLSP(CQSVector answer, CQSMatrix lu, CQSVector b, long int ch[]);
int CQSLUdecompC(CQSMatrix a, long int row_ch[], long int col_ch[]);
int SolveCQSLSC(CQSVector answer, CQSMatrix lu, CQSVector b, long int row_ch[], long int col_ch[]);
int CQSLUdecompPM(CQSMatrix a, long int ch[]);
int SolveCQSLSPM(CQSVector answer, CQSMatrix lu, CQSVector b, long int ch[]);

//--------------------------------------/
// cddlu_strassen.c
//--------------------------------------/
int CQSLUdecomp_square(CQSMatrix a, long int start_index, long int min_dim);
int CQSLUdecomp_l21(CQSMatrix l21, CQSMatrix a, long int start_index, long int min_dim);
int CQSLUdecomp_u12(CQSMatrix u12, CQSMatrix a, long int start_index, long int min_dim);
int CQSLUdecomp_a22(CQSMatrix a, CQSMatrix d22, CQSMatrix l21, CQSMatrix u12, long int start_index, long int min_dim);
int CQSLUdecomp_strassen(CQSMatrix a, long int min_dim);
int CQSLUdecomp_strassenPM(CQSMatrix a, long int ch[], long int min_dim);

#ifdef _OPENMP
int CQSLUdecomp_square_omp(CQSMatrix a, long int start_index, long int min_dim);
int CQSLUdecomp_l21_omp(CQSMatrix l21, CQSMatrix a, long int start_index, long int min_dim);
int CQSLUdecomp_u12_omp(CQSMatrix u12, CQSMatrix a, long int start_index, long int min_dim);
int CQSLUdecomp_a22_omp(CQSMatrix a, CQSMatrix d22, CQSMatrix l21, CQSMatrix u12, long int start_index, long int min_dim);
int CQSLUdecomp_omp(CQSMatrix a);
int CQSLUdecompPM_omp(CQSMatrix a, long int ch[]);
int CQSLUdecomp_strassen_omp(CQSMatrix a, long int min_dim);
int CQSLUdecomp_strassenPM_omp(CQSMatrix a, long int ch[], long int min_dim);
#endif // _OPENMP

//--------------------------------------/
// cddlu_oz.c
//--------------------------------------/
int CQSLUdecomp_a22_oz(CQSMatrix a, CQSMatrix d22, CQSMatrix l21, CQSMatrix u12, long int start_index, long int min_dim, int max_num_div);
int CQSLUdecomp_oz(CQSMatrix a, long int min_dim, int max_num_div);
int CQSLUdecomp_ozPM(CQSMatrix a, long int ch[], long int min_dim, int max_num_div);

#ifdef _OPENMP
int CQSLUdecomp_a22_oz_omp(CQSMatrix a, CQSMatrix d22, CQSMatrix l21, CQSMatrix u12, long int start_index, long int min_dim, int max_num_div);
int CQSLUdecomp_oz_omp(CQSMatrix a, long int min_dim, int max_num_div);
int CQSLUdecomp_ozPM_omp(CQSMatrix a, long int ch[], long int min_dim, int max_num_div);
#endif // _OPENMP

/**************************************/
/* fread_write.c                      */
/**************************************/
/* undefined
void fread_qsmatrix(FILE *fp, QSMatrix mat);
void fread_qsmatrix(FILE *fp, QSMatrix mat);
void fread_qsmatrix_fname(const char *fname, QSMatrix mat);
void fwrite_qsmatrix(FILE *fp, QSMatrix mat);
void fwrite_qsmatrix_fname(const char *fname, QSMatrix mat);
//void fread_qdpolycoef(FILE *fp, QDPoly p, long int maxdeg);
//void fread_qdpolycoef_fname(const char *fname, QDPoly p, long int maxdeg);
void fread_qsvector(FILE *fp, QSVector vec);
void fread_qsvector_fname(const char *fname, QSVector vec);
void fwrite_qsvector(FILE *fp, QSVector vec);
void fwrite_qsvector_fname(const char *fname, QSVector vec);
*/
// 2023-12-15(Fri) T.Kouya
// NOTE: removed stray mis-named declaration `read_test_linear_eq_cqd(CQSMatrix,...)`
// (copy-paste leftover — never defined or called; the real read_test_linear_eq_cqd
// takes CQDMatrix and lives in cqdlinear.h). It collided by name with the cqd
// version, breaking any TU that includes both cqdlinear.h and cqslinear.h. 2026-06-16.

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __BNC_CQSLINEAR_H__
