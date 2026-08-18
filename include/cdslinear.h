/********************************************************************************/
/* cddlinear.h: Double-float precision Complex Linear Computation Library      */
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
// define __BNC_CDSLINEAR_H__
#ifndef __BNC_CDSLINEAR_H__
  #define __BNC_CDSLINEAR_H__

#include "rds.h"
#include "rcds.h" // Complex DD arithmetic

//#include "clinear.h"
#include "cdlinear.h"
#include "dslinear.h"
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
    DSVector re; // Real part
    DSVector im; // Imaginary part

} cdsvector;

typedef cdsvector *CDSVector;

// SIMD: AVX2 and AVX-512
//#if defined(__AVX2__) || defined(__AVX512F__) // __AVX2__ or __AVX512F__
//#include "bncavx.h"
//#endif // defined(__AVX2__) || defined(__AVX512F__)

// SIMD : for copy & paste
#if 0 // __AVX2__ (disabled for single-complex)
#elif 0 // __AVX512F__ (disabled)
#else // others
#endif // __AVX2__

// get_cdsvector_i_cdsfloat
static inline cdsfloat get_cdsvector_i_cdsfloat(CDSVector vec, long int index)
{
	//static 
	cdsfloat ret;

	rds_set(ret.val_re, get_dsvector_i(vec->re, index));
    rds_set(ret.val_im, get_dsvector_i(vec->im, index));

	return ret;
}
// subst_cdsvector_i
// ret := get_cdsvector_i(vec, index)
static inline void subst_cdsvector_i(cdsfloat *ret, CDSVector vec, long int index)
{
	rds_set(ret->val_re, get_dsvector_i(vec->re, index));
	rds_set(ret->val_im, get_dsvector_i(vec->im, index));
} 
// Very dangerous!! 2024-04-18(Thu) T.Kouya
//	GET_CDSVECTOR_I(vec, index))
#define GET_CDSVECTOR_I(vec, index) ((cdsfloat *)&(get_cdsvector_i_cdsfloat((vec), (index))))
//#define get_cdsvector_i(vec, index) ((cdsfloat *)&(get_cdsvector_i_cdsfloat((vec), (index))))
static inline cdsfloat *get_cdsvector_i(CDSVector vec, long int index)
{
	//static is neccesary here! 2024-09-23 T.Kouya
	static cdsfloat ret, *ptr_ret;

	ptr_ret = &ret;

	rds_set(ptr_ret->val_re, get_dsvector_i(vec->re, index));
	rds_set(ptr_ret->val_im, get_dsvector_i(vec->im, index));

	return ptr_ret;
}

// set_cdsvector_i
static inline void set_cdsvector_i(CDSVector vec, long int index, cdsfloat *val) // val[DSSIZE]
{
    set_dsvector_i(vec->re, index, val->val_re);
    set_dsvector_i(vec->im, index, val->val_im);
}
#define SET_CDSVECTOR_I(vec, index, val) set_cdsvector_i((vec), (index), (val))

// set_cdsvector_i_d
static inline void set_cdsvector_i_d(CDSVector vec, long int index, float val) // val
{
    set_dsvector_i_d(vec->re, index, val);
    set0_dsvector_i(vec->im, index);
}
#define SET_CDSVECTOR_I_D(vec, index, val) set_cdsvector_i_d((vec), (index), (val))

// set_cdsvector_i_cd
static inline void set_cdsvector_i_cd(CDSVector vec, long int index, float _Complex val) // val
{
    set_dsvector_i_d(vec->re, index, __real__ val); //creal(val));
    set_dsvector_i_d(vec->im, index, __imag__ val); //cimag(val));
}

// MPLAPACK
#if defined(_DD_COMPLEX_H_) && defined(__cplusplus)
// set_cdsvector_i_cd
static inline void set_cdsvector_i_dd_complex(CDSVector vec, long int index, dd_complex val) // val
{
    set_dsvector_i(vec->re, index, val.real().x);
    set_dsvector_i(vec->im, index, val.imag().x);
}
#endif // _DD_COMPLEX_H_ & __cplusplus

// set0_cdsvector_i
static inline void set0_cdsvector_i(CDSVector vec, long int index)
{
	set0_dsvector_i(vec->re, index);
	set0_dsvector_i(vec->im, index);
}
#define SET0_CDSVECTOR_I(vec, index) set0_cdsvector_i((vec), (index))

// initialize CDSVector
CDSVector init_cdsvector(int dimension);

// free CDSVector
void free_cdsvector(CDSVector vec);

// CDSVector vec -> dsfloat array
void set_cdsfloat_cdsvec(cdsfloat ret[], int ret_dim, CDSVector vec);

// dsvector -> cdsvector
void set_cdsvector_dsvec(CDSVector ret, DSVector re_vec, DSVector im_vec);

// cdsfloat array -> CDSVector ret
void set_cdsvector_cdsfloat(CDSVector ret, cdsfloat array[], int array_dim);

// print dsvector
void print_cdsvector(CDSVector vec);

// set a zero vector
void set0_cdsvector(CDSVector vec);

// set_cdsvector_i_str
void set_cdsvector_i_str(CDSVector vec, long int index, const char *str);

/*************************************************/
/* Vector Calculations for CDSVector               */
/*
void add_cdsvector(CDSVector c, CDSVector a, CDSVector b)
void add2_cdsvector(CDSVector c, CDSVector a)
void sub_cdsvector(CDSVector c, CDSVector a, CDSVector b)
void sub2_cdsvector(CDSVector c, CDVector a)
void cmul_cdsvector(CDSVector c, cdsfloat *val, CDSVector a)
void cmul2_cdsvector(CDSVector c, cdsfloat *val)
void add_cmul_cdsvector(CDSVector c, CDSVector a, cdsfloat *val, CDSVector b)
float ip_cdsvector(CDSVector a, CDSVector b)
float norm1_cdsvector(CDSVector a)
float norm2_cdsvector(CDSVector a)
float normi_cdsvector(CDSVector a)
void subst_cdsvector(CDSVector c, CDSVector a)
*/
/*************************************************/
/* c = a + b */
void add_cdsvector(CDSVector c, CDSVector a, CDSVector b);

/* c += a */
void add2_cdsvector(CDSVector c, CDSVector a);

/* c = a - b */
void sub_cdsvector(CDSVector c, CDSVector a, CDSVector b);

/* c -= a */
void sub2_cdsvector(CDSVector c, CDSVector a);

/* c = val * a */
void cmul_cdsvector_4m(CDSVector c, cdsfloat *val, CDSVector a);
void cmul_cdsvector_3m(CDSVector c, cdsfloat *val, CDSVector a);
#ifdef USE_4M
#define cmul_cdsvector cmul_cdsvector_4m
#else // USE_4M
//#define cmul_cdsvector cmul_cdsvector_3m
#define cmul_cdsvector cmul_cdsvector_4m // faster than 3m
#endif // USE_4M

/* c *= val */
void cmul2_cdsvector(CDSVector c, cdsfloat *val);

/* c = a + val * b */
void add_cmul_cdsvector(CDSVector c, CDSVector a, cdsfloat *val, CDSVector b);

/* c = a - val * b */
void sub_cmul_cdsvector(CDSVector c, CDSVector a, cdsfloat *val, CDSVector b);

/* (a, b) */
void ip_cdsvector(cdsfloat *ret, CDSVector a, CDSVector b);

/* a^T * b */
void dotp_cdsvector(cdsfloat *ret, CDSVector a, CDSVector b);

/* c := a */
void subst_cdsvector(CDSVector c, CDSVector a);

/* c := conj(a) */
void conj_cdsvector(CDSVector c, CDSVector a);

/* c := -a */
void neg_cdsvector(CDSVector c, CDSVector a);

/* ||a||_1 */
void norm1_cdsvector(float ret[DSSIZE], CDSVector a);

/* ||a||_infty */
void normi_cdsvector(float ret[DSSIZE], CDSVector a);

// Euclid norm
void norm2_cdsvector(float ret[DSSIZE], CDSVector vec);

// print cdsvector
void print_cdsvector(CDSVector vec);

// CDD matrix
typedef struct{
	DSMatrix re; // Real part
	DSMatrix im; // Imaginary part
} cdsmatrix;

typedef cdsmatrix *CDSMatrix;

// get_cdsmatrix_ij
static inline cdsfloat get_cdsmatrix_ij_cdsfloat(CDSMatrix mat, long int i, long int j)
{
	long int ij_index;
	//static -> illigal! 2023-12-16
	cdsfloat ret;

	ij_index = mat->re->real_col_dim * i + j;

	rds_set(ret.val_re, get_dsmatrix_ij(mat->re, i, j));
	rds_set(ret.val_im, get_dsmatrix_ij(mat->im, i, j));

	return ret;
} 
// subst_cdsmatrix_ij
// ret := get_cdsmatrix_ij(mat, i, j)
static inline void subst_cdsmatrix_ij(cdsfloat *ret, CDSMatrix mat, long int i, long int j)
{
	long int ij_index;

	rds_set(ret->val_re, get_dsmatrix_ij(mat->re, i, j));
	rds_set(ret->val_im, get_dsmatrix_ij(mat->im, i, j));

	//return ret;
} 
// Very dangerous!! 2024-04-18(Thu) T.Kouya
#define GET_CDSMATRIX_IJ(mat, i, j) (&(get_cdsmatrix_ij_cdsfloat((mat), (i), (j))))
//#define get_cdsmatrix_ij(mat, i, j) (&(get_cdsmatrix_ij_cdsfloat((mat), (i), (j))))
static inline cdsfloat *get_cdsmatrix_ij(CDSMatrix mat, long int i, long int j)
{
	long int ij_index;
	//static is neccesary here! 2024-09-23 T.Kouya
	static cdsfloat ret, *ptr_ret;

	ij_index = mat->re->real_col_dim * i + j;
	ptr_ret = &ret;

	rds_set(ptr_ret->val_re, get_dsmatrix_ij(mat->re, i, j));
	rds_set(ptr_ret->val_im, get_dsmatrix_ij(mat->im, i, j));

	return ptr_ret;
} 

// set_cdsmatrix_ij
static inline void set_cdsmatrix_ij(CDSMatrix mat, long int i, long int j, cdsfloat *val) // val[DSSIZE]
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_dsmatrix_ij(mat->re, i, j, val->val_re);
	set_dsmatrix_ij(mat->im, i, j, val->val_im);

	return;
} 
#define SET_CDSMATRIX_IJ(mat, i, j, val) set_cdsmatrix_ij((mat), (i), (j), (val))

// For MPBLAS
#ifdef _DD_COMPLEX_H_
static inline void set_cdsmatrix_ij_dd_complex(CDSMatrix mat, long int i, long int j, dd_complex val)
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_dsmatrix_ij(mat->re, i, j, val.real().x);
	set_dsmatrix_ij(mat->im, i, j, val.imag().x);
	//set_dsmatrix_ij(mat->re, i, j, val.re.x);
	//set_dsmatrix_ij(mat->im, i, j, val.im.x);

	return;
} 
#endif // _DD_COMPLEX_H_

// set_cdsmatrix_ij_d
static inline void set_cdsmatrix_ij_d(CDSMatrix mat, long int i, long int j, float val)
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_dsmatrix_ij_f(mat->re, i, j, val);
	set0_dsmatrix_ij(mat->im, i, j);

	return;
} 
#define SET_CDSMATRIX_IJ_D(mat, i, j, val) set_cdsmatrix_ij_d((mat), (i), (j), (val))
#define SET_CDSMATRIX_IJ_UI(mat, i, j, val) set_cdsmatrix_ij_d((mat), (i), (j), (float)(val))
#define set_cdsmatrix_ij_ui(mat, i, j, val) set_cdsmatrix_ij_d((mat), (i), (j), (float)(val))

// set_cdsmatrix_ij_dd
static inline void set_cdsmatrix_ij_dd(CDSMatrix mat, long int i, long int j, float val[DSSIZE])
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_dsmatrix_ij(mat->re, i, j, val);
	set0_dsmatrix_ij(mat->im, i, j);

	return;
} 

// set_cdsmatrix_ij_cd
static inline void set_cdsmatrix_ij_cd(CDSMatrix mat, long int i, long int j, float _Complex val)
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_dsmatrix_ij_f(mat->re, i, j, __real__ val); //creal(val));
	set_dsmatrix_ij_f(mat->im, i, j, __imag__ val); //cimag(val));

	return;
} 

// set0_cdsmatrix_ij
static inline void set0_cdsmatrix_ij(CDSMatrix mat, long int i, long int j)
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set0_dsmatrix_ij(mat->re, i, j);
	set0_dsmatrix_ij(mat->im, i, j);

	return;
}
#define SET0_CDSMATRIX_IJ(mat, i, j) set0_cdsmatrix_ij((mat), (i), (j))

// set a zero matrix
//void set0_cdsmatrix(CDSMatrix mat);
void set0_cdsmatrix(CDSMatrix mat);

// initialize dsvector
CDSMatrix init_cdsmatrix(long int row_dim, long int col_dim);

// free dsvector
void free_cdsmatrix(CDSMatrix mat);

// print dsvector
void print_cdsmatrix(CDSMatrix mat);

// CDSMatrix mat -> cdsfloat array
void set_cdsfloat_cdsmat(cdsfloat ret[], int ret_dim, CDSMatrix mat);

// dsmatrix -> cdsmatrix
void set_cdsmatrix_dsmat(CDSMatrix ret, DSMatrix re_mat, DSMatrix im_mat);

// cdsfloat array -> CDDmatrix ret
void set_cdsmatrix_cdsfloat(CDSMatrix ret, cdsfloat array[], int array_dim);

// matrix multiplication
// ret := A * B
void mul_cdsmatrix_4m(CDSMatrix ret, CDSMatrix a, CDSMatrix b);
void mul_cdsmatrix_3m(CDSMatrix ret, CDSMatrix a, CDSMatrix b);
#ifdef USE_4M
#define mul_cdsmatrix mul_cdsmatrix_4m
#else // USE_4M
#define mul_cdsmatrix mul_cdsmatrix_3m
#endif // USE_4M


// Frobenius norm
void normf_cdsmatrix(float ret[DSSIZE], CDSMatrix mat);

// print normf
void print_normf_cdsmatrix(const char *str, CDSMatrix mat);

/*************************************************/
/* Matrix Caluculations for CDSMatrix            */
/*
void normf_cdsmatrix(float ret[DSSIZE], CDSMatrix mat)
void norm1_cdsmatrix(float ret[DSSIZE], CDSMatrix mat)
void normi_cdsmatrix(float ret[DSSIZE], CDSMatrix mat)
void add_cdsmatrix(CDSMatrix c, CDSMatrix a, CDSMatrix b);
void sub_cdsmatrix(CDSMatrix c, CDSMatrix a, CDSMatrix b);
void mul_cdsmatrix(CDSMatrix c, CDSMatrix a, CDSMatrix b);
void mul_cdsmatrix_dsvec(CDSVector v, CDSMatrix a, CDSVector vb)
void mul_cdsmatrixt_dsvec(CDSVector v, CDSMatrix a, CDSVector vb)
void transpose_cdsmatrix(CDSMatrix c, CDSMatrix a);
void inv_cdsmatrix(CDSMatrix a);
void subst_cdsmatrix(CDSMatrix c, CDSMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_cdsmatrix(float ret[DSSIZE], CDSMatrix mat);

/* 1 Norm of Matrix */
void norm1_cdsmatrix(float ret[DSSIZE], CDSMatrix mat);

/* c := a + b */
void add_cdsmatrix(CDSMatrix c, CDSMatrix a, CDSMatrix b);

/* c := a - b */
void sub_cdsmatrix(CDSMatrix c, CDSMatrix a, CDSMatrix b);

/* c := sc * a */
void cmul_cdsmatrix(CDSMatrix c, cdsfloat *sc, CDSMatrix a);

/* c = a^T */
void transpose_cdsmatrix(CDSMatrix c, CDSMatrix a);

/* c := conj(a)^T */
void star_cdsmatrix(CDSMatrix c, CDSMatrix a);

/* c := a */
void subst_cdsmatrix(CDSMatrix c, CDSMatrix a);

/* c := conj(a) */
void conj_cdsmatrix(CDSMatrix c, CDSMatrix a);

/* c := -a */
void neg_cdsmatrix(CDSMatrix c, CDSMatrix a);

/* c := I */
void setI_cdsmatrix(CDSMatrix c);

/* v := a * vb */
void mul_cdsmatrix_cdsvec_4m(CDSVector v, CDSMatrix a, CDSVector vb);
void mul_cdsmatrix_cdsvec_3m(CDSVector v, CDSMatrix a, CDSVector vb);
#ifdef USE_4M
#define mul_cdsmatrix_cdsvec mul_cdsmatrix_cdsvec_4m
#else // USE_4M
#define mul_cdsmatrix_cdsvec mul_cdsmatrix_cdsvec_3m
#endif // USE_4M

/* v := a^T * vb */
void mul_cdsmatrixt_cdsvec(CDSVector v, CDSMatrix a, CDSVector vb);

/* v := conj(a)^T * vb */
void mul_cdsmatrixs_cdsvec(CDSVector v, CDSMatrix a, CDSVector vb);

/* a = a^(-1) */
/* square matrix only */
void inv_cdsmatrix(CDSMatrix a);

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpf)a */
void subst_cmpfvector_cdsvec(CMPFVector c, CDSVector a);

/* c := (dd)a */
void subst_cdsvector_cmpfvec(CDSVector c, CMPFVector a);

/* c := (mpf)a */
void subst_cmpfmatrix_cdsmat(CMPFMatrix c, CDSMatrix a);

/* c := (dd)a */
void subst_cdsmatrix_cmpfmat(CDSMatrix c, CMPFMatrix a);

/* Normwise relative error of vector */
void relerr_cdsvector_cmpfvec(float relerr[DSSIZE], CDSVector approx_vec, CMPFVector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_cdsvector_mpf(float max_relerr[DSSIZE], float min_relerr[DSSIZE], float norm_relerr[DSSIZE], CDSVector approx_vec, MPFVector true_vec, int norm_type);
#endif // USE_GMP

/* c := (dd)a */
void subst_cdsvector_cdvec(CDSVector c, CDVector a);

/* c := (d)a */
void subst_cdvector_cdsvec(CDVector c, CDSVector a);

/* c := (dd)a */
void subst_cdsmatrix_cdmat(CDSMatrix c, CDMatrix a);

/* c := (d)a */
void subst_cdmatrix_cdsmat(CDMatrix c, CDSMatrix a);

/* Normwise relative error of vector */
void relerr_cdsvector(float relerr[DSSIZE], CDSVector approx_vec, CDSVector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_cdsvector(float max_relerr[DSSIZE], float min_relerr[DSSIZE], float norm_relerr[DSSIZE], CDSVector approx_vec, CDSVector true_vec, int norm_type);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_cdsmatrix(CDSMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

// print_cdsmatrix
void print_cdsmatrix(CDSMatrix mat);

/**************************************/
/* cddlu.c                             */
/**************************************/
int CDSLUdecomp(CDSMatrix a);
int SolveCDSLS(CDSVector answer, CDSMatrix lu, CDSVector b);
int CDSLUdecompP(CDSMatrix a, long int ch[]);
int SolveCDSLSP(CDSVector answer, CDSMatrix lu, CDSVector b, long int ch[]);
int CDSLUdecompC(CDSMatrix a, long int row_ch[], long int col_ch[]);
int SolveCDSLSC(CDSVector answer, CDSMatrix lu, CDSVector b, long int row_ch[], long int col_ch[]);
int CDSLUdecompPM(CDSMatrix a, long int ch[]);
int SolveCDSLSPM(CDSVector answer, CDSMatrix lu, CDSVector b, long int ch[]);

//--------------------------------------/
// cddlu_strassen.c
//--------------------------------------/
int CDSLUdecomp_square(CDSMatrix a, long int start_index, long int min_dim);
int CDSLUdecomp_l21(CDSMatrix l21, CDSMatrix a, long int start_index, long int min_dim);
int CDSLUdecomp_u12(CDSMatrix u12, CDSMatrix a, long int start_index, long int min_dim);
int CDSLUdecomp_a22(CDSMatrix a, CDSMatrix d22, CDSMatrix l21, CDSMatrix u12, long int start_index, long int min_dim);
int CDSLUdecomp_strassen(CDSMatrix a, long int min_dim);
int CDSLUdecomp_strassenPM(CDSMatrix a, long int ch[], long int min_dim);

#ifdef _OPENMP
int CDSLUdecomp_square_omp(CDSMatrix a, long int start_index, long int min_dim);
int CDSLUdecomp_l21_omp(CDSMatrix l21, CDSMatrix a, long int start_index, long int min_dim);
int CDSLUdecomp_u12_omp(CDSMatrix u12, CDSMatrix a, long int start_index, long int min_dim);
int CDSLUdecomp_a22_omp(CDSMatrix a, CDSMatrix d22, CDSMatrix l21, CDSMatrix u12, long int start_index, long int min_dim);
int CDSLUdecomp_omp(CDSMatrix a);
int CDSLUdecompPM_omp(CDSMatrix a, long int ch[]);
int CDSLUdecomp_strassen_omp(CDSMatrix a, long int min_dim);
int CDSLUdecomp_strassenPM_omp(CDSMatrix a, long int ch[], long int min_dim);
#endif // _OPENMP

//--------------------------------------/
// cddlu_oz.c
//--------------------------------------/
int CDSLUdecomp_a22_oz(CDSMatrix a, CDSMatrix d22, CDSMatrix l21, CDSMatrix u12, long int start_index, long int min_dim, int max_num_div);
int CDSLUdecomp_oz(CDSMatrix a, long int min_dim, int max_num_div);
int CDSLUdecomp_ozPM(CDSMatrix a, long int ch[], long int min_dim, int max_num_div);

#ifdef _OPENMP
int CDSLUdecomp_a22_oz_omp(CDSMatrix a, CDSMatrix d22, CDSMatrix l21, CDSMatrix u12, long int start_index, long int min_dim, int max_num_div);
int CDSLUdecomp_oz_omp(CDSMatrix a, long int min_dim, int max_num_div);
int CDSLUdecomp_ozPM_omp(CDSMatrix a, long int ch[], long int min_dim, int max_num_div);
#endif // _OPENMP

/**************************************/
/* fread_write.c                      */
/**************************************/
/* undefined
void fread_dsmatrix(FILE *fp, DSMatrix mat);
void fread_dsmatrix(FILE *fp, DSMatrix mat);
void fread_dsmatrix_fname(const char *fname, DSMatrix mat);
void fwrite_dsmatrix(FILE *fp, DSMatrix mat);
void fwrite_dsmatrix_fname(const char *fname, DSMatrix mat);
//void fread_ddpolycoef(FILE *fp, DDPoly p, long int maxdeg);
//void fread_ddpolycoef_fname(const char *fname, DDPoly p, long int maxdeg);
void fread_dsvector(FILE *fp, DSVector vec);
void fread_dsvector_fname(const char *fname, DSVector vec);
void fwrite_dsvector(FILE *fp, DSVector vec);
void fwrite_dsvector_fname(const char *fname, DSVector vec);
*/

// NOTE: removed stray mis-named declaration `read_test_linear_eq_cdd(CDSMatrix,...)`
// (copy-paste leftover — never defined or called; the real read_test_linear_eq_cdd
// takes CDDMatrix and lives in cddlinear.h). It collided by name with the cdd
// version, breaking any TU that includes both cddlinear.h and cdslinear.h
// (e.g. the Automake build with USE_DDLINEAR + USE_DSLINEAR). 2026-06-16.

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __BNC_DDLINEAR_H__
