/********************************************************************************/
/* cqdlinear.h: Quadruple-double precision Complex Linear Computation Library   */
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
// define __BNC_CQDLINEAR_H__
#ifndef __BNC_CQDLINEAR_H__
  #define __BNC_CQDLINEAR_H__

#include "rdd.h"
#include "rcdd.h" // Complex QD arithmetic

//#include "clinear.h"
#include "cdlinear.h"
#ifdef USE_DDLINEAR
#include "cddlinear.h"
#endif // USE_DDLINEAR
#ifdef USE_TDLINEAR
#include "ctdlinear.h"
#endif // USE_TDLINEAR
#include "qdlinear.h"
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
    QDVector re; // Real part
    QDVector im; // Imaginary part

} cqdvector;

typedef cqdvector *CQDVector;

// SIMD: AVX2 and AVX-512
//#if defined(__AVX2__) || defined(__AVX512F__) // __AVX2__ or __AVX512F__
//#include "bncavx.h"
//#endif // defined(__AVX2__) || defined(__AVX512F__)

// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
#elif defined(__AVX512F__) // __AVX512F__
#else // others
#endif // __AVX2__

// get_cqdvector_i_cqdfloat
static inline cqdfloat get_cqdvector_i_cqdfloat(CQDVector vec, long int index)
{
	//static 
	cqdfloat ret;

	rqd_set(ret.val_re, get_qdvector_i(vec->re, index));
    rqd_set(ret.val_im, get_qdvector_i(vec->im, index));

	return ret;
}
// subst_cqdvector_i
// ret := get_cqdvector_i(vec, index)
static inline void subst_cqdvector_i(cqdfloat *ret, CQDVector vec, long int index)
{
	rqd_set(ret->val_re, get_qdvector_i(vec->re, index));
	rqd_set(ret->val_im, get_qdvector_i(vec->im, index));
} 
// Very dangerous!! 2024-04-18(Thu) T.Kouya
//	GET_CQDVECTOR_I(vec, index))
#define GET_CQDVECTOR_I(vec, index) ((cqdfloat *)&(get_cqdvector_i_cqdfloat((vec), (index))))
//#define get_cqdvector_i(vec, index) ((cqdfloat *)&(get_cqdvector_i_cqdfloat((vec), (index))))
static inline cqdfloat *get_cqdvector_i(CQDVector vec, long int index)
{
	//static is neccesary here! 2024-09-23 T.Kouya
	static cqdfloat ret, *ptr_ret;

	ptr_ret = &ret;

	rqd_set(ptr_ret->val_re, get_qdvector_i(vec->re, index));
	rqd_set(ptr_ret->val_im, get_qdvector_i(vec->im, index));

	return ptr_ret;
}

// set_cqdvector_i
static inline void set_cqdvector_i(CQDVector vec, long int index, cqdfloat *val) // val[QDSIZE]
{
    set_qdvector_i(vec->re, index, val->val_re);
    set_qdvector_i(vec->im, index, val->val_im);
}
#define SET_CQDVECTOR_I(vec, index, val) set_cqdvector_i((vec), (index), (val))

// set_cqdvector_i_d
static inline void set_cqdvector_i_d(CQDVector vec, long int index, double val) // val
{
    set_qdvector_i_d(vec->re, index, val);
    set0_qdvector_i(vec->im, index);
}
#define SET_CQDVECTOR_I_D(vec, index, val) set_cqdvector_i_d((vec), (index), (val))

// set_cqdvector_i_cd
static inline void set_cqdvector_i_cd(CQDVector vec, long int index, double _Complex val) // val
{
    set_qdvector_i_d(vec->re, index, __real__ val); //creal(val));
    set_qdvector_i_d(vec->im, index, __imag__ val); //cimag(val));
}

// MPLAPACK
#if defined(_QD_COMPLEX_H_) && defined(__cplusplus)
// set_cqdvector_i_cd
static inline void set_cqdvector_i_qd_complex(CQDVector vec, long int index, qd_complex val) // val
{
    set_qdvector_i(vec->re, index, val.real().x);
    set_qdvector_i(vec->im, index, val.imag().x);
}
#endif // _QD_COMPLEX_H_ & __cplusplus

// set0_cqdvector_i
static inline void set0_cqdvector_i(CQDVector vec, long int index)
{
	set0_qdvector_i(vec->re, index);
	set0_qdvector_i(vec->im, index);
}
#define SET0_CQDVECTOR_I(vec, index) set0_cqdvector_i((vec), (index))

// initialize CQDVector
CQDVector init_cqdvector(int dimension);

// free CQDVector
void free_cqdvector(CQDVector vec);

// CQDVector vec -> qdfloat array
void set_cqdfloat_cddvec(cqdfloat ret[], int ret_dim, CQDVector vec);

// qdvector -> cqdvector
void set_cqdvector_qdvec(CQDVector ret, QDVector re_vec, QDVector im_vec);

// qdfloat array -> CQDVector ret
void set_cqdvector_qdfloat(CQDVector ret, qdfloat array[], int array_dim);

// print qdvector
void print_cqdvector(CQDVector vec);

// set a zero vector
void set0_cqdvector(CQDVector vec);

// set_cqdvector_i_str
void set_cqdvector_i_str(CQDVector vec, long int index, const char *str);

/*************************************************/
/* Vector Calculations for CQDVector               */
/*
void add_cqdvector(CQDVector c, CQDVector a, CQDVector b)
void add2_cqdvector(CQDVector c, CQDVector a)
void sub_cqdvector(CQDVector c, CQDVector a, CQDVector b)
void sub2_cqdvector(CQDVector c, CDVector a)
void cmul_cqdvector(CQDVector c, cqdfloat *val, CQDVector a)
void cmul2_cqdvector(CQDVector c, cqdfloat *val)
void add_cmul_cqdvector(CQDVector c, CQDVector a, cqdfloat *val, CQDVector b)
double ip_cqdvector(CQDVector a, CQDVector b)
double norm1_cqdvector(CQDVector a)
double norm2_cqdvector(CQDVector a)
double normi_cqdvector(CQDVector a)
void subst_cqdvector(CQDVector c, CQDVector a)
*/
/*************************************************/
/* c = a + b */
void add_cqdvector(CQDVector c, CQDVector a, CQDVector b);

/* c += a */
void add2_cqdvector(CQDVector c, CQDVector a);

/* c = a - b */
void sub_cqdvector(CQDVector c, CQDVector a, CQDVector b);

/* c -= a */
void sub2_cqdvector(CQDVector c, CQDVector a);

/* c = val * a */
void cmul_cqdvector_4m(CQDVector c, cqdfloat *val, CQDVector a);
void cmul_cqdvector_3m(CQDVector c, cqdfloat *val, CQDVector a);
#ifdef USE_4M
#define cmul_cqdvector cmul_cqdvector_4m
#else // USE_4M
//#define cmul_cqdvector cmul_cqdvector_3m
#define cmul_cqdvector cmul_cqdvector_4m // faster than 3m on DD arithmetic
#endif // USE_4M

/* c *= val */
void cmul2_cqdvector(CQDVector c, cqdfloat *val);

/* c = a + val * b */
void add_cmul_cqdvector(CQDVector c, CQDVector a, cqdfloat *val, CQDVector b);

/* c = a - val * b */
void sub_cmul_cqdvector(CQDVector c, CQDVector a, cqdfloat *val, CQDVector b);

/* (a, b) */
void ip_cqdvector(cqdfloat *ret, CQDVector a, CQDVector b);

/* a^T * b */
void dotp_cqdvector(cqdfloat *ret, CQDVector a, CQDVector b);

/* c := a */
void subst_cqdvector(CQDVector c, CQDVector a);

/* c := -a */
void neg_cqdvector(CQDVector c, CQDVector a);

/* ||a||_1 */
void norm1_cqdvector(double ret[QDSIZE], CQDVector a);

/* ||a||_infty */
void normi_cqdvector(double ret[QDSIZE], CQDVector a);

// Euclid norm
void norm2_cqdvector(double ret[QDSIZE], CQDVector vec);

// print cqdvector
void print_cqdvector(CQDVector vec);

// CQD matrix
typedef struct{
	QDMatrix re; // Real part
	QDMatrix im; // Imaginary part
} cqdmatrix;

typedef cqdmatrix *CQDMatrix;

// get_cqdmatrix_ij
static inline cqdfloat get_cqdmatrix_ij_cqdfloat(CQDMatrix mat, long int i, long int j)
{
	long int ij_index;
	//static 
	cqdfloat ret;

	ij_index = mat->re->real_col_dim * i + j;

	rqd_set(ret.val_re, get_qdmatrix_ij(mat->re, i, j));
	rqd_set(ret.val_im, get_qdmatrix_ij(mat->im, i, j));

	return ret;
} 
// subst_cqdmatrix_ij
// ret := get_cqdmatrix_ij(mat, i, j)
static inline void subst_cqdmatrix_ij(cqdfloat *ret, CQDMatrix mat, long int i, long int j)
{
	long int ij_index;

	rqd_set(ret->val_re, get_qdmatrix_ij(mat->re, i, j));
	rqd_set(ret->val_im, get_qdmatrix_ij(mat->im, i, j));

	//return ret;
} 
// Very dangerous!! 2024-04-18(Thu) T.Kouya
#define GET_CQDMATRIX_IJ(mat, i, j) (&(get_cqdmatrix_ij_cqdfloat((mat), (i), (j))))
//#define get_cqdmatrix_ij(mat, i, j) (&(get_cqdmatrix_ij_cqdfloat((mat), (i), (j))))
static inline cqdfloat *get_cqdmatrix_ij(CQDMatrix mat, long int i, long int j)
{
	long int ij_index;
	//static is neccesary here! 2024-09-23 T.Kouya
	static cqdfloat ret, *ptr_ret;

	ij_index = mat->re->real_col_dim * i + j;
	ptr_ret = &ret;

	rqd_set(ptr_ret->val_re, get_qdmatrix_ij(mat->re, i, j));
	rqd_set(ptr_ret->val_im, get_qdmatrix_ij(mat->im, i, j));

	return ptr_ret;
} 

// set_cqdmatrix_ij
static inline void set_cqdmatrix_ij(CQDMatrix mat, long int i, long int j, cqdfloat *val) // val[QDSIZE]
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_qdmatrix_ij(mat->re, i, j, val->val_re);
	set_qdmatrix_ij(mat->im, i, j, val->val_im);

	return;
} 
#define SET_CQDMATRIX_IJ(mat, i, j, val) set_cqdmatrix_ij((mat), (i), (j), (val))

// For MPBLAS
#ifdef _QD_COMPLEX_H_
static inline void set_cqdmatrix_ij_qd_complex(CQDMatrix mat, long int i, long int j, qd_complex val)
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_qdmatrix_ij(mat->re, i, j, val.real().x);
	set_qdmatrix_ij(mat->im, i, j, val.imag().x);

	return;
} 
#endif // _QD_COMPLEX_H_

// set_cqdmatrix_ij_d
static inline void set_cqdmatrix_ij_d(CQDMatrix mat, long int i, long int j, double val)
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_qdmatrix_ij_d(mat->re, i, j, val);
	set0_qdmatrix_ij(mat->im, i, j);

	return;
} 
#define SET_CQDMATRIX_IJ_D(mat, i, j, val) set_cqdmatrix_ij_d((mat), (i), (j), (val))
#define SET_CQDMATRIX_IJ_UI(mat, i, j, val) set_cqdmatrix_ij_d((mat), (i), (j), (double)(val))
#define set_cqdmatrix_ij_ui(mat, i, j, val) set_cqdmatrix_ij_d((mat), (i), (j), (double)(val))

// set_cqdmatrix_ij_qd
static inline void set_cqdmatrix_ij_qd(CQDMatrix mat, long int i, long int j, double val[QDSIZE])
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_qdmatrix_ij(mat->re, i, j, val);
	set0_qdmatrix_ij(mat->im, i, j);

	return;
} 

// set_cqdmatrix_ij_cd
static inline void set_cqdmatrix_ij_cd(CQDMatrix mat, long int i, long int j, double _Complex val)
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_qdmatrix_ij_d(mat->re, i, j, __real__ val); //creal(val));
	set_qdmatrix_ij_d(mat->im, i, j, __imag__ val); //cimag(val));

	return;
} 

// set0_cqdmatrix_ij
static inline void set0_cqdmatrix_ij(CQDMatrix mat, long int i, long int j)
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set0_qdmatrix_ij(mat->re, i, j);
	set0_qdmatrix_ij(mat->im, i, j);

	return;
}
#define SET0_CQDMATRIX_IJ(mat, i, j) set0_cqdmatrix_ij((mat), (i), (j))

// set a zero matrix
//void set0_cqdmatrix(CQDMatrix mat);
void set0_cqdmatrix(CQDMatrix mat);

// initialize qdvector
CQDMatrix init_cqdmatrix(long int row_dim, long int col_dim);

// free qdvector
void free_cqdmatrix(CQDMatrix mat);

// print qdvector
void print_cqdmatrix(CQDMatrix mat);

// CQDMatrix mat -> cqdfloat array
void set_cqdfloat_cddmat(cqdfloat ret[], int ret_dim, CQDMatrix mat);

// qdmatrix -> cqdmatrix
void set_cqdmatrix_qdmat(CQDMatrix ret, QDMatrix re_mat, QDMatrix im_mat);

// cqdfloat array -> CQDmatrix ret
void set_cqdmatrix_cqdfloat(CQDMatrix ret, cqdfloat array[], int array_dim);

// matrix multiplication
// ret := A * B
void mul_cqdmatrix_4m(CQDMatrix ret, CQDMatrix a, CQDMatrix b);
void mul_cqdmatrix_3m(CQDMatrix ret, CQDMatrix a, CQDMatrix b);
#ifdef USE_4M
#define mul_cqdmatrix mul_cqdmatrix_4m
#else // USE_4M
#define mul_cqdmatrix mul_cqdmatrix_3m
#endif // USE_4M

// Frobenius norm
void normf_cqdmatrix(double ret[QDSIZE], CQDMatrix mat);

// print normf
void print_normf_cqdmatrix(const char *str, CQDMatrix mat);

/*************************************************/
/* Matrix Caluculations for CQDMatrix            */
/*
void normf_cqdmatrix(double ret[QDSIZE], CQDMatrix mat)
void norm1_cqdmatrix(double ret[QDSIZE], CQDMatrix mat)
void normi_cqdmatrix(double ret[QDSIZE], CQDMatrix mat)
void add_cqdmatrix(CQDMatrix c, CQDMatrix a, CQDMatrix b);
void sub_cqdmatrix(CQDMatrix c, CQDMatrix a, CQDMatrix b);
void mul_cqdmatrix(CQDMatrix c, CQDMatrix a, CQDMatrix b);
void mul_cqdmatrix_cqdvec(CQDVector v, CQDMatrix a, CQDVector vb)
void mul_cqdmatrixt_cqdvec(CQDVector v, CQDMatrix a, CQDVector vb)
void transpose_cqdmatrix(CQDMatrix c, CQDMatrix a);
void inv_cqdmatrix(CQDMatrix a);
void subst_cqdmatrix(CQDMatrix c, CQDMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_cqdmatrix(double ret[QDSIZE], CQDMatrix mat);

/* 1 Norm of Matrix */
void norm1_cqdmatrix(double ret[QDSIZE], CQDMatrix mat);

/* c := a + b */
void add_cqdmatrix(CQDMatrix c, CQDMatrix a, CQDMatrix b);

/* c := a - b */
void sub_cqdmatrix(CQDMatrix c, CQDMatrix a, CQDMatrix b);

/* c := sc * a */
void cmul_cqdmatrix(CQDMatrix c, cqdfloat *sc, CQDMatrix a);

/* c = a^T */
void transpose_cqdmatrix(CQDMatrix c, CQDMatrix a);

/* c := conj(a)^T */
void star_cqdmatrix(CQDMatrix c, CQDMatrix a);

/* c := a */
void subst_cqdmatrix(CQDMatrix c, CQDMatrix a);

/* c := conj(a) */
void conj_cqdmatrix(CQDMatrix c, CQDMatrix a);

/* c := -a */
void neg_cqdmatrix(CQDMatrix c, CQDMatrix a);

/* c := a */
void subst_cqdmatrix(CQDMatrix c, CQDMatrix a);

/* c := I */
void setI_cqdmatrix(CQDMatrix c);

/* v := a * vb */
void mul_cqdmatrix_cqdvec_4m(CQDVector v, CQDMatrix a, CQDVector vb);
void mul_cqdmatrix_cqdvec_3m(CQDVector v, CQDMatrix a, CQDVector vb);
#ifdef USE_4M
#define mul_cqdmatrix_cqdvec mul_cqdmatrix_cqdvec_4m
#else // USE_4M
#define mul_cqdmatrix_cqdvec mul_cqdmatrix_cqdvec_3m
#endif // USE_4M

/* v := a^T * vb */
void mul_cqdmatrixt_cqdvec(CQDVector v, CQDMatrix a, CQDVector vb);

/* v := conj(a)^T * vb */
void mul_cqdmatrixs_cqdvec(CQDVector v, CQDMatrix a, CQDVector vb);

/* a = a^(-1) */
/* square matrix only */
void inv_cqdmatrix(CQDMatrix a);

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpf)a */
void subst_cmpfvector_cqdvec(CMPFVector c, CQDVector a);

/* c := (dd)a */
void subst_cqdvector_cmpfvec(CQDVector c, CMPFVector a);

/* c := (mpf)a */
void subst_cmpfmatrix_cqdmat(CMPFMatrix c, CQDMatrix a);

/* c := (dd)a */
void subst_cqdmatrix_cmpfmat(CQDMatrix c, CMPFMatrix a);

/* Normwise relative error of vector */
void relerr_cqdvector_cmpfvec(double relerr[QDSIZE], CQDVector approx_vec, CMPFVector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_cqdvector_mpf(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double norm_relerr[QDSIZE], CQDVector approx_vec, MPFVector true_vec, int norm_type);
#endif // USE_GMP

/* c := (dd)a */
void subst_cqdvector_cdvec(CQDVector c, CDVector a);

/* c := (d)a */
///void subst_cdvector_cddvec(CDVector c, CQDVector a);

/* c := (dd)a */
void subst_cqdmatrix_cdmat(CQDMatrix c, CDMatrix a);

/* c := (d)a */
//void subst_cdmatrix_cddmat(CDMatrix c, CQDMatrix a);


#ifdef USE_DDLINEAR
/* c := (qd)a */
void subst_cqdvector_cddvec(CQDVector c, CDDVector a);

/* c := (dd)a */
void subst_cddvector_cqdvec(CDDVector c, CQDVector a);

/* c := (qd)a */
void subst_cqdmatrix_cddmat(CQDMatrix c, CDDMatrix a);

/* c := (dd)a */
void subst_cddmatrix_cqdmat(CDDMatrix c, CQDMatrix a);

#endif // USE_DDLINEAR

#ifdef USE_TDLINEAR
/* c := (qd)a */
void subst_cqdvector_ctdvec(CQDVector c, CTDVector a);

/* c := (td)a */
void subst_ctdvector_cqdvec(CTDVector c, CQDVector a);

/* c := (qd)a */
void subst_cqdmatrix_ctdmat(CQDMatrix c, CTDMatrix a);

/* c := (td)a */
void subst_ctdmatrix_cqdmat(CTDMatrix c, CQDMatrix a);
#endif // USE_TDLINEAR

/* Normwise relative error of vector */
void relerr_cqdvector(double relerr[QDSIZE], CQDVector approx_vec, CQDVector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_cqdvector(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double norm_relerr[QDSIZE], CQDVector approx_vec, CQDVector true_vec, int norm_type);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_cqdmatrix(CQDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

// print_cqdmatrix
void print_cqdmatrix(CQDMatrix mat);

/**************************************/
/* cddlu.c                             */
/**************************************/
int CQDLUdecomp(CQDMatrix a);
int SolveCQDLS(CQDVector answer, CQDMatrix lu, CQDVector b);
int CQDLUdecompP(CQDMatrix a, long int ch[]);
int SolveCQDLSP(CQDVector answer, CQDMatrix lu, CQDVector b, long int ch[]);
int CQDLUdecompC(CQDMatrix a, long int row_ch[], long int col_ch[]);
int SolveCQDLSC(CQDVector answer, CQDMatrix lu, CQDVector b, long int row_ch[], long int col_ch[]);
int CQDLUdecompPM(CQDMatrix a, long int ch[]);
int SolveCQDLSPM(CQDVector answer, CQDMatrix lu, CQDVector b, long int ch[]);

//--------------------------------------/
// cddlu_strassen.c
//--------------------------------------/
int CQDLUdecomp_square(CQDMatrix a, long int start_index, long int min_dim);
int CQDLUdecomp_l21(CQDMatrix l21, CQDMatrix a, long int start_index, long int min_dim);
int CQDLUdecomp_u12(CQDMatrix u12, CQDMatrix a, long int start_index, long int min_dim);
int CQDLUdecomp_a22(CQDMatrix a, CQDMatrix d22, CQDMatrix l21, CQDMatrix u12, long int start_index, long int min_dim);
int CQDLUdecomp_strassen(CQDMatrix a, long int min_dim);
int CQDLUdecomp_strassenPM(CQDMatrix a, long int ch[], long int min_dim);

#ifdef _OPENMP
int CQDLUdecomp_square_omp(CQDMatrix a, long int start_index, long int min_dim);
int CQDLUdecomp_l21_omp(CQDMatrix l21, CQDMatrix a, long int start_index, long int min_dim);
int CQDLUdecomp_u12_omp(CQDMatrix u12, CQDMatrix a, long int start_index, long int min_dim);
int CQDLUdecomp_a22_omp(CQDMatrix a, CQDMatrix d22, CQDMatrix l21, CQDMatrix u12, long int start_index, long int min_dim);
int CQDLUdecomp_omp(CQDMatrix a);
int CQDLUdecompPM_omp(CQDMatrix a, long int ch[]);
int CQDLUdecomp_strassen_omp(CQDMatrix a, long int min_dim);
int CQDLUdecomp_strassenPM_omp(CQDMatrix a, long int ch[], long int min_dim);
#endif // _OPENMP

//--------------------------------------/
// cddlu_oz.c
//--------------------------------------/
int CQDLUdecomp_a22_oz(CQDMatrix a, CQDMatrix d22, CQDMatrix l21, CQDMatrix u12, long int start_index, long int min_dim, int max_num_div);
int CQDLUdecomp_oz(CQDMatrix a, long int min_dim, int max_num_div);
int CQDLUdecomp_ozPM(CQDMatrix a, long int ch[], long int min_dim, int max_num_div);

#ifdef _OPENMP
int CQDLUdecomp_a22_oz_omp(CQDMatrix a, CQDMatrix d22, CQDMatrix l21, CQDMatrix u12, long int start_index, long int min_dim, int max_num_div);
int CQDLUdecomp_oz_omp(CQDMatrix a, long int min_dim, int max_num_div);
int CQDLUdecomp_ozPM_omp(CQDMatrix a, long int ch[], long int min_dim, int max_num_div);
#endif // _OPENMP

/**************************************/
/* fread_write.c                      */
/**************************************/
/* undefined
void fread_qdmatrix(FILE *fp, QDMatrix mat);
void fread_qdmatrix(FILE *fp, QDMatrix mat);
void fread_qdmatrix_fname(const char *fname, QDMatrix mat);
void fwrite_qdmatrix(FILE *fp, QDMatrix mat);
void fwrite_qdmatrix_fname(const char *fname, QDMatrix mat);
//void fread_qdpolycoef(FILE *fp, QDPoly p, long int maxdeg);
//void fread_qdpolycoef_fname(const char *fname, QDPoly p, long int maxdeg);
void fread_qdvector(FILE *fp, QDVector vec);
void fread_qdvector_fname(const char *fname, QDVector vec);
void fwrite_qdvector(FILE *fp, QDVector vec);
void fwrite_qdvector_fname(const char *fname, QDVector vec);
*/
// 2023-12-15(Fri) T.Kouya
// read problem from file
void read_test_linear_eq_cqd(CQDMatrix A, CQDVector true_x, CQDVector b, long int dim, const char *fname_A, const char *fname_true_x, const char *fname_b);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __BNC_CQDLINEAR_H__
