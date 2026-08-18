/********************************************************************************/
/* cddlinear.h: Double-double precision Complex Linear Computation Library      */
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
// define __BNC_CDDLINEAR_H__
#ifndef __BNC_CDDLINEAR_H__
  #define __BNC_CDDLINEAR_H__

#include "rdd.h"
#include "rcdd.h" // Complex DD arithmetic

//#include "clinear.h"
#include "cdlinear.h"
#include "ddlinear.h"
//#include "bmatrix.h"

#ifdef USE_GMP
#include "gmp.h"
#include "mpfr.h"
#include "mpfr_dtq_sd.h"
#include "mpflinear.h"
//#include "clinear.h
#include "cmpflinear.h"
#endif //USE_GMP//

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//#if defined(__AVX2__) || defined(__AVX512F__)
//#include "bncavx.h"
//#endif // defined(__AVX2__) || defined(__AVX512F__)

// CDD vector
typedef struct
{
    DDVector re; // Real part
    DDVector im; // Imaginary part

} cddvector;

typedef cddvector *CDDVector;

// SIMD: AVX2 and AVX-512
//#if defined(__AVX2__) || defined(__AVX512F__) // __AVX2__ or __AVX512F__
//#include "bncavx.h"
//#endif // defined(__AVX2__) || defined(__AVX512F__)

// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
#elif defined(__AVX512F__) // __AVX512F__
#else // others
#endif // __AVX2__

// get_cddvector_i_cddfloat
static inline cddfloat get_cddvector_i_cddfloat(CDDVector vec, long int index)
{
	//static 
	cddfloat ret;

	rdd_set(ret.val_re, get_ddvector_i(vec->re, index));
    rdd_set(ret.val_im, get_ddvector_i(vec->im, index));

	return ret;
}
// subst_cddvector_i
// ret := get_cddvector_i(vec, index)
static inline void subst_cddvector_i(cddfloat *ret, CDDVector vec, long int index)
{
	rdd_set(ret->val_re, get_ddvector_i(vec->re, index));
	rdd_set(ret->val_im, get_ddvector_i(vec->im, index));
} 
// Very dangerous!! 2024-04-18(Thu) T.Kouya
//	GET_CDDVECTOR_I(vec, index))
#define GET_CDDVECTOR_I(vec, index) ((cddfloat *)&(get_cddvector_i_cddfloat((vec), (index))))
//#define get_cddvector_i(vec, index) ((cddfloat *)&(get_cddvector_i_cddfloat((vec), (index))))
static inline cddfloat *get_cddvector_i(CDDVector vec, long int index)
{
	//static is neccesary here! 2024-09-23 T.Kouya
	static cddfloat ret, *ptr_ret;

	ptr_ret = &ret;

	rdd_set(ptr_ret->val_re, get_ddvector_i(vec->re, index));
	rdd_set(ptr_ret->val_im, get_ddvector_i(vec->im, index));

	return ptr_ret;
}

// set_cddvector_i
static inline void set_cddvector_i(CDDVector vec, long int index, cddfloat *val) // val[DDSIZE]
{
    set_ddvector_i(vec->re, index, val->val_re);
    set_ddvector_i(vec->im, index, val->val_im);
}
#define SET_CDDVECTOR_I(vec, index, val) set_cddvector_i((vec), (index), (val))

// set_cddvector_i_d
static inline void set_cddvector_i_d(CDDVector vec, long int index, double val) // val
{
    set_ddvector_i_d(vec->re, index, val);
    set0_ddvector_i(vec->im, index);
}
#define SET_CDDVECTOR_I_D(vec, index, val) set_cddvector_i_d((vec), (index), (val))

// set_cddvector_i_cd
static inline void set_cddvector_i_cd(CDDVector vec, long int index, double _Complex val) // val
{
    set_ddvector_i_d(vec->re, index, __real__ val); //creal(val));
    set_ddvector_i_d(vec->im, index, __imag__ val); //cimag(val));
}

// MPLAPACK
#if defined(_DD_COMPLEX_H_) && defined(__cplusplus)
// set_cddvector_i_cd
static inline void set_cddvector_i_dd_complex(CDDVector vec, long int index, dd_complex val) // val
{
    set_ddvector_i(vec->re, index, val.real().x);
    set_ddvector_i(vec->im, index, val.imag().x);
}
#endif // _DD_COMPLEX_H_ & __cplusplus

// set0_cddvector_i
static inline void set0_cddvector_i(CDDVector vec, long int index)
{
	set0_ddvector_i(vec->re, index);
	set0_ddvector_i(vec->im, index);
}
#define SET0_CDDVECTOR_I(vec, index) set0_cddvector_i((vec), (index))

// initialize CDDVector
CDDVector init_cddvector(int dimension);

// free CDDVector
void free_cddvector(CDDVector vec);

// CDDVector vec -> ddfloat array
void set_cddfloat_cddvec(cddfloat ret[], int ret_dim, CDDVector vec);

// ddvector -> cddvector
void set_cddvector_ddvec(CDDVector ret, DDVector re_vec, DDVector im_vec);

// cddfloat array -> CDDVector ret
void set_cddvector_cddfloat(CDDVector ret, cddfloat array[], int array_dim);

// print ddvector
void print_cddvector(CDDVector vec);

// set a zero vector
void set0_cddvector(CDDVector vec);

// set_cddvector_i_str
void set_cddvector_i_str(CDDVector vec, long int index, const char *str);

/*************************************************/
/* Vector Calculations for CDDVector               */
/*
void add_cddvector(CDDVector c, CDDVector a, CDDVector b)
void add2_cddvector(CDDVector c, CDDVector a)
void sub_cddvector(CDDVector c, CDDVector a, CDDVector b)
void sub2_cddvector(CDDVector c, CDVector a)
void cmul_cddvector(CDDVector c, cddfloat *val, CDDVector a)
void cmul2_cddvector(CDDVector c, cddfloat *val)
void add_cmul_cddvector(CDDVector c, CDDVector a, cddfloat *val, CDDVector b)
double ip_cddvector(CDDVector a, CDDVector b)
double norm1_cddvector(CDDVector a)
double norm2_cddvector(CDDVector a)
double normi_cddvector(CDDVector a)
void subst_cddvector(CDDVector c, CDDVector a)
*/
/*************************************************/
/* c = a + b */
void add_cddvector(CDDVector c, CDDVector a, CDDVector b);

/* c += a */
void add2_cddvector(CDDVector c, CDDVector a);

/* c = a - b */
void sub_cddvector(CDDVector c, CDDVector a, CDDVector b);

/* c -= a */
void sub2_cddvector(CDDVector c, CDDVector a);

/* c = val * a */
void cmul_cddvector_4m(CDDVector c, cddfloat *val, CDDVector a);
void cmul_cddvector_3m(CDDVector c, cddfloat *val, CDDVector a);
#ifdef USE_4M
#define cmul_cddvector cmul_cddvector_4m
#else // USE_4M
//#define cmul_cddvector cmul_cddvector_3m
#define cmul_cddvector cmul_cddvector_4m // faster than 3m
#endif // USE_4M

/* c *= val */
void cmul2_cddvector(CDDVector c, cddfloat *val);

/* c = a + val * b */
void add_cmul_cddvector(CDDVector c, CDDVector a, cddfloat *val, CDDVector b);

/* c = a - val * b */
void sub_cmul_cddvector(CDDVector c, CDDVector a, cddfloat *val, CDDVector b);

/* (a, b) */
void ip_cddvector(cddfloat *ret, CDDVector a, CDDVector b);

/* a^T * b */
void dotp_cddvector(cddfloat *ret, CDDVector a, CDDVector b);

/* c := a */
void subst_cddvector(CDDVector c, CDDVector a);

/* c := conj(a) */
void conj_cddvector(CDDVector c, CDDVector a);

/* c := -a */
void neg_cddvector(CDDVector c, CDDVector a);

/* ||a||_1 */
void norm1_cddvector(double ret[DDSIZE], CDDVector a);

/* ||a||_infty */
void normi_cddvector(double ret[DDSIZE], CDDVector a);

// Euclid norm
void norm2_cddvector(double ret[DDSIZE], CDDVector vec);

// print cddvector
void print_cddvector(CDDVector vec);

// CDD matrix
typedef struct{
	DDMatrix re; // Real part
	DDMatrix im; // Imaginary part
} cddmatrix;

typedef cddmatrix *CDDMatrix;

// get_cddmatrix_ij
static inline cddfloat get_cddmatrix_ij_cddfloat(CDDMatrix mat, long int i, long int j)
{
	long int ij_index;
	//static -> illigal! 2023-12-16
	cddfloat ret;

	ij_index = mat->re->real_col_dim * i + j;

	rdd_set(ret.val_re, get_ddmatrix_ij(mat->re, i, j));
	rdd_set(ret.val_im, get_ddmatrix_ij(mat->im, i, j));

	return ret;
} 
// subst_cddmatrix_ij
// ret := get_cddmatrix_ij(mat, i, j)
static inline void subst_cddmatrix_ij(cddfloat *ret, CDDMatrix mat, long int i, long int j)
{
	long int ij_index;

	rdd_set(ret->val_re, get_ddmatrix_ij(mat->re, i, j));
	rdd_set(ret->val_im, get_ddmatrix_ij(mat->im, i, j));

	//return ret;
} 
// Very dangerous!! 2024-04-18(Thu) T.Kouya
#define GET_CDDMATRIX_IJ(mat, i, j) (&(get_cddmatrix_ij_cddfloat((mat), (i), (j))))
//#define get_cddmatrix_ij(mat, i, j) (&(get_cddmatrix_ij_cddfloat((mat), (i), (j))))
static inline cddfloat *get_cddmatrix_ij(CDDMatrix mat, long int i, long int j)
{
	long int ij_index;
	//static is neccesary here! 2024-09-23 T.Kouya
	static cddfloat ret, *ptr_ret;

	ij_index = mat->re->real_col_dim * i + j;
	ptr_ret = &ret;

	rdd_set(ptr_ret->val_re, get_ddmatrix_ij(mat->re, i, j));
	rdd_set(ptr_ret->val_im, get_ddmatrix_ij(mat->im, i, j));

	return ptr_ret;
} 

// set_cddmatrix_ij
static inline void set_cddmatrix_ij(CDDMatrix mat, long int i, long int j, cddfloat *val) // val[DDSIZE]
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_ddmatrix_ij(mat->re, i, j, val->val_re);
	set_ddmatrix_ij(mat->im, i, j, val->val_im);

	return;
} 
#define SET_CDDMATRIX_IJ(mat, i, j, val) set_cddmatrix_ij((mat), (i), (j), (val))

// For MPBLAS
#ifdef _DD_COMPLEX_H_
static inline void set_cddmatrix_ij_dd_complex(CDDMatrix mat, long int i, long int j, dd_complex val)
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_ddmatrix_ij(mat->re, i, j, val.real().x);
	set_ddmatrix_ij(mat->im, i, j, val.imag().x);
	//set_ddmatrix_ij(mat->re, i, j, val.re.x);
	//set_ddmatrix_ij(mat->im, i, j, val.im.x);

	return;
} 
#endif // _DD_COMPLEX_H_

// set_cddmatrix_ij_d
static inline void set_cddmatrix_ij_d(CDDMatrix mat, long int i, long int j, double val)
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_ddmatrix_ij_d(mat->re, i, j, val);
	set0_ddmatrix_ij(mat->im, i, j);

	return;
} 
#define SET_CDDMATRIX_IJ_D(mat, i, j, val) set_cddmatrix_ij_d((mat), (i), (j), (val))
#define SET_CDDMATRIX_IJ_UI(mat, i, j, val) set_cddmatrix_ij_d((mat), (i), (j), (double)(val))
#define set_cddmatrix_ij_ui(mat, i, j, val) set_cddmatrix_ij_d((mat), (i), (j), (double)(val))

// set_cddmatrix_ij_dd
static inline void set_cddmatrix_ij_dd(CDDMatrix mat, long int i, long int j, double val[DDSIZE])
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_ddmatrix_ij(mat->re, i, j, val);
	set0_ddmatrix_ij(mat->im, i, j);

	return;
} 

// set_cddmatrix_ij_cd
static inline void set_cddmatrix_ij_cd(CDDMatrix mat, long int i, long int j, double _Complex val)
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_ddmatrix_ij_d(mat->re, i, j, __real__ val); //creal(val));
	set_ddmatrix_ij_d(mat->im, i, j, __imag__ val); //cimag(val));

	return;
} 

// set0_cddmatrix_ij
static inline void set0_cddmatrix_ij(CDDMatrix mat, long int i, long int j)
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set0_ddmatrix_ij(mat->re, i, j);
	set0_ddmatrix_ij(mat->im, i, j);

	return;
}
#define SET0_CDDMATRIX_IJ(mat, i, j) set0_cddmatrix_ij((mat), (i), (j))

// set a zero matrix
//void set0_cddmatrix(CDDMatrix mat);
void set0_cddmatrix(CDDMatrix mat);

// initialize ddvector
CDDMatrix init_cddmatrix(long int row_dim, long int col_dim);

// free ddvector
void free_cddmatrix(CDDMatrix mat);

// print ddvector
void print_cddmatrix(CDDMatrix mat);

// CDDMatrix mat -> cddfloat array
void set_cddfloat_cddmat(cddfloat ret[], int ret_dim, CDDMatrix mat);

// ddmatrix -> cddmatrix
void set_cddmatrix_ddmat(CDDMatrix ret, DDMatrix re_mat, DDMatrix im_mat);

// cddfloat array -> CDDmatrix ret
void set_cddmatrix_cddfloat(CDDMatrix ret, cddfloat array[], int array_dim);

// matrix multiplication
// ret := A * B
void mul_cddmatrix_4m(CDDMatrix ret, CDDMatrix a, CDDMatrix b);
void mul_cddmatrix_3m(CDDMatrix ret, CDDMatrix a, CDDMatrix b);
#ifdef USE_4M
#define mul_cddmatrix mul_cddmatrix_4m
#else // USE_4M
#define mul_cddmatrix mul_cddmatrix_3m
#endif // USE_4M


// Frobenius norm
void normf_cddmatrix(double ret[DDSIZE], CDDMatrix mat);

// print normf
void print_normf_cddmatrix(const char *str, CDDMatrix mat);

/*************************************************/
/* Matrix Caluculations for CDDMatrix            */
/*
void normf_cddmatrix(double ret[DDSIZE], CDDMatrix mat)
void norm1_cddmatrix(double ret[DDSIZE], CDDMatrix mat)
void normi_cddmatrix(double ret[DDSIZE], CDDMatrix mat)
void add_cddmatrix(CDDMatrix c, CDDMatrix a, CDDMatrix b);
void sub_cddmatrix(CDDMatrix c, CDDMatrix a, CDDMatrix b);
void mul_cddmatrix(CDDMatrix c, CDDMatrix a, CDDMatrix b);
void mul_cddmatrix_ddvec(CDDVector v, CDDMatrix a, CDDVector vb)
void mul_cddmatrixt_ddvec(CDDVector v, CDDMatrix a, CDDVector vb)
void transpose_cddmatrix(CDDMatrix c, CDDMatrix a);
void inv_cddmatrix(CDDMatrix a);
void subst_cddmatrix(CDDMatrix c, CDDMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_cddmatrix(double ret[DDSIZE], CDDMatrix mat);

/* 1 Norm of Matrix */
void norm1_cddmatrix(double ret[DDSIZE], CDDMatrix mat);

/* c := a + b */
void add_cddmatrix(CDDMatrix c, CDDMatrix a, CDDMatrix b);

/* c := a - b */
void sub_cddmatrix(CDDMatrix c, CDDMatrix a, CDDMatrix b);

/* c := sc * a */
void cmul_cddmatrix(CDDMatrix c, cddfloat *sc, CDDMatrix a);

/* c = a^T */
void transpose_cddmatrix(CDDMatrix c, CDDMatrix a);

/* c := conj(a)^T */
void star_cddmatrix(CDDMatrix c, CDDMatrix a);

/* c := a */
void subst_cddmatrix(CDDMatrix c, CDDMatrix a);

/* c := conj(a) */
void conj_cddmatrix(CDDMatrix c, CDDMatrix a);

/* c := -a */
void neg_cddmatrix(CDDMatrix c, CDDMatrix a);

/* c := I */
void setI_cddmatrix(CDDMatrix c);

/* v := a * vb */
void mul_cddmatrix_cddvec_4m(CDDVector v, CDDMatrix a, CDDVector vb);
void mul_cddmatrix_cddvec_3m(CDDVector v, CDDMatrix a, CDDVector vb);
#ifdef USE_4M
#define mul_cddmatrix_cddvec mul_cddmatrix_cddvec_4m
#else // USE_4M
#define mul_cddmatrix_cddvec mul_cddmatrix_cddvec_3m
#endif // USE_4M

/* v := a^T * vb */
void mul_cddmatrixt_cddvec(CDDVector v, CDDMatrix a, CDDVector vb);

/* v := conj(a)^T * vb */
void mul_cddmatrixs_cddvec(CDDVector v, CDDMatrix a, CDDVector vb);

/* a = a^(-1) */
/* square matrix only */
void inv_cddmatrix(CDDMatrix a);

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpf)a */
void subst_cmpfvector_cddvec(CMPFVector c, CDDVector a);

/* c := (dd)a */
void subst_cddvector_cmpfvec(CDDVector c, CMPFVector a);

/* c := (mpf)a */
void subst_cmpfmatrix_cddmat(CMPFMatrix c, CDDMatrix a);

/* c := (dd)a */
void subst_cddmatrix_cmpfmat(CDDMatrix c, CMPFMatrix a);

/* Normwise relative error of vector */
void relerr_cddvector_cmpfvec(double relerr[DDSIZE], CDDVector approx_vec, CMPFVector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_cddvector_mpf(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double norm_relerr[DDSIZE], CDDVector approx_vec, MPFVector true_vec, int norm_type);
#endif // USE_GMP

/* c := (dd)a */
void subst_cddvector_cdvec(CDDVector c, CDVector a);

/* c := (d)a */
void subst_cdvector_cddvec(CDVector c, CDDVector a);

/* c := (dd)a */
void subst_cddmatrix_cdmat(CDDMatrix c, CDMatrix a);

/* c := (d)a */
void subst_cdmatrix_cddmat(CDMatrix c, CDDMatrix a);

/* Normwise relative error of vector */
void relerr_cddvector(double relerr[DDSIZE], CDDVector approx_vec, CDDVector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_cddvector(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double norm_relerr[DDSIZE], CDDVector approx_vec, CDDVector true_vec, int norm_type);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_cddmatrix(CDDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

// print_cddmatrix
void print_cddmatrix(CDDMatrix mat);

/**************************************/
/* cddlu.c                             */
/**************************************/
int CDDLUdecomp(CDDMatrix a);
int SolveCDDLS(CDDVector answer, CDDMatrix lu, CDDVector b);
int CDDLUdecompP(CDDMatrix a, long int ch[]);
int SolveCDDLSP(CDDVector answer, CDDMatrix lu, CDDVector b, long int ch[]);
int CDDLUdecompC(CDDMatrix a, long int row_ch[], long int col_ch[]);
int SolveCDDLSC(CDDVector answer, CDDMatrix lu, CDDVector b, long int row_ch[], long int col_ch[]);
int CDDLUdecompPM(CDDMatrix a, long int ch[]);
int SolveCDDLSPM(CDDVector answer, CDDMatrix lu, CDDVector b, long int ch[]);

//--------------------------------------/
// cddlu_strassen.c
//--------------------------------------/
int CDDLUdecomp_square(CDDMatrix a, long int start_index, long int min_dim);
int CDDLUdecomp_l21(CDDMatrix l21, CDDMatrix a, long int start_index, long int min_dim);
int CDDLUdecomp_u12(CDDMatrix u12, CDDMatrix a, long int start_index, long int min_dim);
int CDDLUdecomp_a22(CDDMatrix a, CDDMatrix d22, CDDMatrix l21, CDDMatrix u12, long int start_index, long int min_dim);
int CDDLUdecomp_strassen(CDDMatrix a, long int min_dim);
int CDDLUdecomp_strassenPM(CDDMatrix a, long int ch[], long int min_dim);

#ifdef _OPENMP
int CDDLUdecomp_square_omp(CDDMatrix a, long int start_index, long int min_dim);
int CDDLUdecomp_l21_omp(CDDMatrix l21, CDDMatrix a, long int start_index, long int min_dim);
int CDDLUdecomp_u12_omp(CDDMatrix u12, CDDMatrix a, long int start_index, long int min_dim);
int CDDLUdecomp_a22_omp(CDDMatrix a, CDDMatrix d22, CDDMatrix l21, CDDMatrix u12, long int start_index, long int min_dim);
int CDDLUdecomp_omp(CDDMatrix a);
int CDDLUdecompPM_omp(CDDMatrix a, long int ch[]);
int CDDLUdecomp_strassen_omp(CDDMatrix a, long int min_dim);
int CDDLUdecomp_strassenPM_omp(CDDMatrix a, long int ch[], long int min_dim);
#endif // _OPENMP

//--------------------------------------/
// cddlu_oz.c
//--------------------------------------/
int CDDLUdecomp_a22_oz(CDDMatrix a, CDDMatrix d22, CDDMatrix l21, CDDMatrix u12, long int start_index, long int min_dim, int max_num_div);
int CDDLUdecomp_oz(CDDMatrix a, long int min_dim, int max_num_div);
int CDDLUdecomp_ozPM(CDDMatrix a, long int ch[], long int min_dim, int max_num_div);

#ifdef _OPENMP
int CDDLUdecomp_a22_oz_omp(CDDMatrix a, CDDMatrix d22, CDDMatrix l21, CDDMatrix u12, long int start_index, long int min_dim, int max_num_div);
int CDDLUdecomp_oz_omp(CDDMatrix a, long int min_dim, int max_num_div);
int CDDLUdecomp_ozPM_omp(CDDMatrix a, long int ch[], long int min_dim, int max_num_div);
#endif // _OPENMP

/**************************************/
/* fread_write.c                      */
/**************************************/
/* undefined
void fread_ddmatrix(FILE *fp, DDMatrix mat);
void fread_ddmatrix(FILE *fp, DDMatrix mat);
void fread_ddmatrix_fname(const char *fname, DDMatrix mat);
void fwrite_ddmatrix(FILE *fp, DDMatrix mat);
void fwrite_ddmatrix_fname(const char *fname, DDMatrix mat);
//void fread_ddpolycoef(FILE *fp, DDPoly p, long int maxdeg);
//void fread_ddpolycoef_fname(const char *fname, DDPoly p, long int maxdeg);
void fread_ddvector(FILE *fp, DDVector vec);
void fread_ddvector_fname(const char *fname, DDVector vec);
void fwrite_ddvector(FILE *fp, DDVector vec);
void fwrite_ddvector_fname(const char *fname, DDVector vec);
*/

// defined
void read_test_linear_eq_cdd(CDDMatrix A, CDDVector true_x, CDDVector b, long int dim, const char *fname_A, const char *fname_true_x, const char *fname_b);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __BNC_DDLINEAR_H__
