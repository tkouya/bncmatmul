/********************************************************************************/
/* ctdlinear.h: Triple-float precision Complex Linear Computation Library      */
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
// define __BNC_CTSLINEAR_H__
#ifndef __BNC_CTSLINEAR_H__
  #define __BNC_CTSLINEAR_H__

#include "rds.h"
#include "rcds.h" // Complex TD arithmetic

#include "cdlinear.h"
#include "cdslinear.h"
#include "tslinear.h"
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

// CTD vector
typedef struct
{
    TSVector re; // Real part
    TSVector im; // Imaginary part
} ctsvector;

typedef ctsvector *CTSVector;

// SIMD: AVX2 and AVX-512
//#if defined(__AVX2__) || defined(__AVX512F__) // __AVX2__ or __AVX512F__
//#include "bncavx.h"
//#endif // defined(__AVX2__) || defined(__AVX512F__)

// SIMD : for copy & paste
#if 0 // __AVX2__ (disabled for single-complex)
#elif 0 // __AVX512F__ (disabled)
#else // others
#endif // __AVX2__

// get_ctsvector_i_ctsfloat
static inline ctsfloat get_ctsvector_i_ctsfloat(CTSVector vec, long int index)
{
	//static 
	ctsfloat ret;

	rts_set(ret.val_re, get_tsvector_i(vec->re, index));
    rts_set(ret.val_im, get_tsvector_i(vec->im, index));

	return ret;
}
// subst_ctsvector_i
// ret := get_ctsvector_i(vec, index)
static inline void subst_ctsvector_i(ctsfloat *ret, CTSVector vec, long int index)
{
	rts_set(ret->val_re, get_tsvector_i(vec->re, index));
	rts_set(ret->val_im, get_tsvector_i(vec->im, index));
} 
// Very dangerous!! 2024-04-18(Thu) T.Kouya
//	GET_CDDVECTOR_I(vec, index))
#define GET_CTSVECTOR_I(vec, index) ((ctsfloat *)&(get_ctsvector_i_ctsfloat((vec), (index))))
//#define get_ctsvector_i(vec, index) ((ctsfloat *)&(get_ctsvector_i_ctsfloat((vec), (index))))
static inline ctsfloat *get_ctsvector_i(CTSVector vec, long int index)
{
	//static is neccesary here! 2024-09-23 T.Kouya
	static ctsfloat ret, *ptr_ret;

	ptr_ret = &ret;

	rts_set(ptr_ret->val_re, get_tsvector_i(vec->re, index));
	rts_set(ptr_ret->val_im, get_tsvector_i(vec->im, index));

	return ptr_ret;
}

// set_ctsvector_i
static inline void set_ctsvector_i(CTSVector vec, long int index, ctsfloat *val) // val[TSSIZE]
{
    set_tsvector_i(vec->re, index, val->val_re);
    set_tsvector_i(vec->im, index, val->val_im);
}
#define SET_CTSVECTOR_I(vec, index, val) set_ctsvector_i((vec), (index), (val))

// set_ctsvector_i_d
static inline void set_ctsvector_i_d(CTSVector vec, long int index, float val) // val
{
    set_tsvector_i_f(vec->re, index, val);
    set0_tsvector_i(vec->im, index);
}
#define SET_CTSVECTOR_I_D(vec, index, val) set_ctsvector_i_d((vec), (index), (val))

// set_ctsvector_i_cd
static inline void set_ctsvector_i_cd(CTSVector vec, long int index, float _Complex val) // val
{
    set_tsvector_i_f(vec->re, index, __real__ val); //creal(val));
    set_tsvector_i_f(vec->im, index, __imag__ val); //cimag(val));
}

// set0_ctsvector_i
static inline void set0_ctsvector_i(CTSVector vec, long int index)
{
	set0_tsvector_i(vec->re, index);
	set0_tsvector_i(vec->im, index);
}
#define SET0_CTSVECTOR_I(vec, index) set0_ctsvector_i((vec), (index))

// initialize CTSVector
CTSVector init_ctsvector(int dimension);

// free CTSVector
void free_ctsvector(CTSVector vec);

// CTSVector vec -> ctsfloat array
void set_ctsfloat_ctsvec(ctsfloat ret[], int ret_dim, CTSVector vec);

// tsvector -> ctsvector
void set_ctsvector_tsvec(CTSVector ret, TSVector re_vec, TSVector im_vec);

// ctsfloat array -> CTSVector ret
void set_ctsvector_ctsfloat(CTSVector ret, ctsfloat array[], int array_dim);

// print tsvector
void print_ctsvector(CTSVector vec);

// set a zero vector
void set0_ctsvector(CTSVector vec);

// set_ctsvector_i_str
void set_ctsvector_i_str(CTSVector vec, long int index, const char *str);

/*************************************************/
/* Vector Calculations for CTSVector               */
/*
void add_ctsvector(CTSVector c, CTSVector a, CTSVector b)
void add2_ctsvector(CTSVector c, CTSVector a)
void sub_ctsvector(CTSVector c, CTSVector a, CTSVector b)
void sub2_ctsvector(CTSVector c, CDVector a)
void cmul_ctsvector(CTSVector c, ctsfloat *val, CTSVector a)
void cmul2_ctsvector(CTSVector c, ctsfloat *val)
void add_cmul_ctsvector(CTSVector c, CTSVector a, ctsfloat *val, CTSVector b)
float ip_ctsvector(CTSVector a, CTSVector b)
float norm1_ctsvector(CTSVector a)
float norm2_ctsvector(CTSVector a)
float normi_ctsvector(CTSVector a)
void subst_ctsvector(CTSVector c, CTSVector a)
*/
/*************************************************/
/* c = a + b */
void add_ctsvector(CTSVector c, CTSVector a, CTSVector b);

/* c += a */
void add2_ctsvector(CTSVector c, CTSVector a);

/* c = a - b */
void sub_ctsvector(CTSVector c, CTSVector a, CTSVector b);

/* c -= a */
void sub2_ctsvector(CTSVector c, CTSVector a);

/* c = val * a */
void cmul_ctsvector_4m(CTSVector c, ctsfloat *val, CTSVector a);
void cmul_ctsvector_3m(CTSVector c, ctsfloat *val, CTSVector a);
#ifdef USE_4M
#define cmul_ctsvector cmul_ctsvector_4m
#else // USE_4M
//#define cmul_ctsvector cmul_ctsvector_3m
#define cmul_ctsvector cmul_ctsvector_4m // faster than 3m on TD arithemtic
#endif // USE_4M

/* c *= val */
void cmul2_ctsvector(CTSVector c, ctsfloat *val);

/* c = a + val * b */
void add_cmul_ctsvector(CTSVector c, CTSVector a, ctsfloat *val, CTSVector b);

/* c = a - val * b */
void sub_cmul_ctsvector(CTSVector c, CTSVector a, ctsfloat *val, CTSVector b);

/* (a, b) */
void ip_ctsvector(ctsfloat *ret, CTSVector a, CTSVector b);

/* a^T * b */
void dotp_ctsvector(ctsfloat *ret, CTSVector a, CTSVector b);

/* c := a */
void subst_ctsvector(CTSVector c, CTSVector a);

/* c := -a */
void neg_ctsvector(CTSVector c, CTSVector a);

/* ||a||_1 */
void norm1_ctsvector(float ret[TSSIZE], CTSVector a);

/* ||a||_infty */
void normi_ctsvector(float ret[TSSIZE], CTSVector a);

// Euclid norm
void norm2_ctsvector(float ret[TSSIZE], CTSVector vec);

// print ctsvector
void print_ctsvector(CTSVector vec);

// CTD matrix
typedef struct{
	TSMatrix re; // Real part
	TSMatrix im; // Imaginary part
} ctsmatrix;

typedef ctsmatrix *CTSMatrix;

// get_ctsmatrix_ij
static inline ctsfloat get_ctsmatrix_ij_ctsfloat(CTSMatrix mat, long int i, long int j)
{
	long int ij_index;
	//static 
	ctsfloat ret;

	ij_index = mat->re->real_col_dim * i + j;

	rts_set(ret.val_re, get_tsmatrix_ij(mat->re, i, j));
	rts_set(ret.val_im, get_tsmatrix_ij(mat->im, i, j));

	return ret;
}
// subst_ctsmatrix_ij
// ret := get_ctsmatrix_ij(mat, i, j)
static inline void subst_ctsmatrix_ij(ctsfloat *ret, CTSMatrix mat, long int i, long int j)
{
	long int ij_index;

	rts_set(ret->val_re, get_tsmatrix_ij(mat->re, i, j));
	rts_set(ret->val_im, get_tsmatrix_ij(mat->im, i, j));

	//return ret;
} 
// Very dangerous!! 2024-04-18(Thu) T.Kouya
#define GET_CTSMATRIX_IJ(mat, i, j) (&(get_ctsmatrix_ij_ctsfloat((mat), (i), (j))))
//#define get_ctsmatrix_ij(mat, i, j) (&(get_ctsmatrix_ij_ctsfloat((mat), (i), (j))))
static inline ctsfloat *get_ctsmatrix_ij(CTSMatrix mat, long int i, long int j)
{
	long int ij_index;
	//static is neccesary here! 2024-09-23 T.Kouya
	static ctsfloat ret, *ptr_ret;

	ij_index = mat->re->real_col_dim * i + j;
	ptr_ret = &ret;

	rts_set(ptr_ret->val_re, get_tsmatrix_ij(mat->re, i, j));
	rts_set(ptr_ret->val_im, get_tsmatrix_ij(mat->im, i, j));

	return ptr_ret;
} 

// set_ctsmatrix_ij
static inline void set_ctsmatrix_ij(CTSMatrix mat, long int i, long int j, ctsfloat *val) // val[TSSIZE]
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_tsmatrix_ij(mat->re, i, j, val->val_re);
	set_tsmatrix_ij(mat->im, i, j, val->val_im);

	return;
} 
#define SET_CTSMATRIX_IJ(mat, i, j, val) set_ctsmatrix_ij((mat), (i), (j), (val))

// set_ctsmatrix_ij_d
static inline void set_ctsmatrix_ij_d(CTSMatrix mat, long int i, long int j, float val)
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_tsmatrix_ij_f(mat->re, i, j, val);
	set0_tsmatrix_ij(mat->im, i, j);

	return;
} 
#define SET_CTSMATRIX_IJ_D(mat, i, j, val) set_ctsmatrix_ij_d((mat), (i), (j), (val))
#define SET_CTSMATRIX_IJ_UI(mat, i, j, val) set_ctsmatrix_ij_d((mat), (i), (j), (float)(val))
#define set_ctsmatrix_ij_ui(mat, i, j, val) set_ctsmatrix_ij_d((mat), (i), (j), (float)(val))

// set_ctsmatrix_ij_td
static inline void set_ctsmatrix_ij_td(CTSMatrix mat, long int i, long int j, float val[TSSIZE])
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_tsmatrix_ij(mat->re, i, j, val);
	set0_tsmatrix_ij(mat->im, i, j);

	return;
} 

// set_ctsmatrix_ij_cd
static inline void set_ctsmatrix_ij_cd(CTSMatrix mat, long int i, long int j, float _Complex val)
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_tsmatrix_ij_f(mat->re, i, j, __real__ val); //creal(val));
	set_tsmatrix_ij_f(mat->im, i, j, __imag__ val); //cimag(val));

	return;
} 

// set0_ctsmatrix_ij
static inline void set0_ctsmatrix_ij(CTSMatrix mat, long int i, long int j)
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set0_tsmatrix_ij(mat->re, i, j);
	set0_tsmatrix_ij(mat->im, i, j);

	return;
}
#define SET0_CTSMATRIX_IJ(mat, i, j) set0_ctsmatrix_ij((mat), (i), (j))

// set a zero matrix
//void set0_ctsmatrix(CTSMatrix mat);
void set0_ctsmatrix(CTSMatrix mat);

// initialize tsvector
CTSMatrix init_ctsmatrix(long int row_dim, long int col_dim);

// free tsvector
void free_ctsmatrix(CTSMatrix mat);

// print tsvector
void print_ctsmatrix(CTSMatrix mat);

// CTSMatrix mat -> ctsfloat array
void set_ctsfloat_ctsmat(ctsfloat ret[], int ret_dim, CTSMatrix mat);

// tsmatrix -> ctsmatrix
void set_ctsmatrix_tsmat(CTSMatrix ret, TSMatrix re_mat, TSMatrix im_mat);

// ctsfloat array -> CTDmatrix ret
void set_ctsmatrix_ctsfloat(CTSMatrix ret, ctsfloat array[], int array_dim);

// matrix multiplication
// ret := A * B
void mul_ctsmatrix_4m(CTSMatrix ret, CTSMatrix a, CTSMatrix b);
void mul_ctsmatrix_3m(CTSMatrix ret, CTSMatrix a, CTSMatrix b);
#ifdef USE_4M
#define mul_ctsmatrix mul_ctsmatrix_4m
#else // USE_4M
#define mul_ctsmatrix mul_ctsmatrix_3m
#endif // USE_4M

// Frobenius norm
void normf_ctsmatrix(float ret[TSSIZE], CTSMatrix mat);

// print normf
void print_normf_ctsmatrix(const char *str, CTSMatrix mat);

/*************************************************/
/* Matrix Caluculations for CTSMatrix            */
/*
void normf_ctsmatrix(float ret[TSSIZE], CTSMatrix mat)
void norm1_ctsmatrix(float ret[TSSIZE], CTSMatrix mat)
void normi_ctsmatrix(float ret[TSSIZE], CTSMatrix mat)
void add_ctsmatrix(CTSMatrix c, CTSMatrix a, CTSMatrix b);
void sub_ctsmatrix(CTSMatrix c, CTSMatrix a, CTSMatrix b);
void mul_ctsmatrix(CTSMatrix c, CTSMatrix a, CTSMatrix b);
void mul_ctsmatrix_ctsvec(CTSVector v, CTSMatrix a, CTSVector vb)
void mul_ctsmatrixt_ctsvec(CTSVector v, CTSMatrix a, CTSVector vb)
void transpose_ctsmatrix(CTSMatrix c, CTSMatrix a);
void inv_ctsmatrix(CTSMatrix a);
void subst_ctsmatrix(CTSMatrix c, CTSMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_ctsmatrix(float ret[TSSIZE], CTSMatrix mat);

/* 1 Norm of Matrix */
void norm1_ctsmatrix(float ret[TSSIZE], CTSMatrix mat);

/* c := a + b */
void add_ctsmatrix(CTSMatrix c, CTSMatrix a, CTSMatrix b);

/* c := a - b */
void sub_ctsmatrix(CTSMatrix c, CTSMatrix a, CTSMatrix b);

/* c := sc * a */
void cmul_ctsmatrix(CTSMatrix c, ctsfloat *sc, CTSMatrix a);

/* c = a^T */
void transpose_ctsmatrix(CTSMatrix c, CTSMatrix a);

/* c := conj(a)^T */
void star_ctsmatrix(CTSMatrix c, CTSMatrix a);

/* c := a */
void subst_ctsmatrix(CTSMatrix c, CTSMatrix a);

/* c := conj(a) */
void conj_ctsmatrix(CTSMatrix c, CTSMatrix a);

/* c := -a */
void neg_ctsmatrix(CTSMatrix c, CTSMatrix a);

/* c := a */
void subst_ctsmatrix(CTSMatrix c, CTSMatrix a);

/* c := I */
void setI_ctsmatrix(CTSMatrix c);

/* v := a * vb */
void mul_ctsmatrix_ctsvec_4m(CTSVector v, CTSMatrix a, CTSVector vb);
void mul_ctsmatrix_ctsvec_3m(CTSVector v, CTSMatrix a, CTSVector vb);
#ifdef USE_4M
#define mul_ctsmatrix_ctsvec mul_ctsmatrix_ctsvec_4m
#else // USE_4M
#define mul_ctsmatrix_ctsvec mul_ctsmatrix_ctsvec_3m
#endif // USE_4M

/* v := a^T * vb */
void mul_ctsmatrixt_ctsvec(CTSVector v, CTSMatrix a, CTSVector vb);

/* v := conj(a)^T * vb */
void mul_ctsmatrixs_ctsvec(CTSVector v, CTSMatrix a, CTSVector vb);

/* a = a^(-1) */
/* square matrix only */
void inv_ctsmatrix(CTSMatrix a);

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpf)a */
void subst_cmpfvector_ctsvec(CMPFVector c, CTSVector a);

/* c := (ctd)a */
void subst_ctsvector_cmpfvec(CTSVector c, CMPFVector a);

/* c := (cmpf)a */
void subst_cmpfmatrix_ctsmat(CMPFMatrix c, CTSMatrix a);

/* c := (ctd)a */
void subst_ctsmatrix_cmpfmat(CTSMatrix c, CMPFMatrix a);

/* Normwise relative error of vector */
void relerr_ctsvector_cmpfvec(float relerr[TSSIZE], CTSVector approx_vec, CMPFVector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_ctsvector_mpf(float max_relerr[TSSIZE], float min_relerr[TSSIZE], float norm_relerr[TSSIZE], CTSVector approx_vec, MPFVector true_vec, int norm_type);
#endif // USE_GMP

/* c := (ctd)a */
void subst_ctsvector_cdvec(CTSVector c, CDVector a);

/* c := (d)a */
void subst_cdvector_ctsvec(CDVector c, CTSVector a);

/* c := (ctd)a */
void subst_ctsmatrix_cdmat(CTSMatrix c, CDMatrix a);

/* c := (d)a */
void subst_cdmatrix_ctsmat(CDMatrix c, CTSMatrix a);

/* c := (td)a */
void subst_ctsvector_cdsvec(CTSVector c, CDSVector a);

/* c := (dd)a */
void subst_cdsvector_ctsvec(CDSVector c, CTSVector a);

/* c := (td)a */
void subst_ctsmatrix_cdsmat(CTSMatrix c, CDSMatrix a);

/* c := (dd)a */
void subst_cdsmatrix_ctsmat(CDSMatrix c, CTSMatrix a);


/* Normwise relative error of vector */
void relerr_ctsvector(float relerr[TSSIZE], CTSVector approx_vec, CTSVector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_ctsvector(float max_relerr[TSSIZE], float min_relerr[TSSIZE], float norm_relerr[TSSIZE], CTSVector approx_vec, CTSVector true_vec, int norm_type);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_ctsmatrix(CTSMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

// print_ctsmatrix
void print_ctsmatrix(CTSMatrix mat);

/**************************************/
/* ctdlu.c                             */
/**************************************/
int CTSLUdecomp(CTSMatrix a);
int SolveCTSLS(CTSVector answer, CTSMatrix lu, CTSVector b);
int CTSLUdecompP(CTSMatrix a, long int ch[]);
int SolveCTSLSP(CTSVector answer, CTSMatrix lu, CTSVector b, long int ch[]);
int CTSLUdecompC(CTSMatrix a, long int row_ch[], long int col_ch[]);
int SolveCTSLSC(CTSVector answer, CTSMatrix lu, CTSVector b, long int row_ch[], long int col_ch[]);
int CTSLUdecompPM(CTSMatrix a, long int ch[]);
int SolveCTSLSPM(CTSVector answer, CTSMatrix lu, CTSVector b, long int ch[]);

//--------------------------------------/
// ctdlu_strassen.c
//--------------------------------------/
int CTSLUdecomp_square(CTSMatrix a, long int start_index, long int min_dim);
int CTSLUdecomp_l21(CTSMatrix l21, CTSMatrix a, long int start_index, long int min_dim);
int CTSLUdecomp_u12(CTSMatrix u12, CTSMatrix a, long int start_index, long int min_dim);
int CTSLUdecomp_a22(CTSMatrix a, CTSMatrix d22, CTSMatrix l21, CTSMatrix u12, long int start_index, long int min_dim);
int CTSLUdecomp_strassen(CTSMatrix a, long int min_dim);
int CTSLUdecomp_strassenPM(CTSMatrix a, long int ch[], long int min_dim);

#ifdef _OPENMP
int CTSLUdecomp_square_omp(CTSMatrix a, long int start_index, long int min_dim);
int CTSLUdecomp_l21_omp(CTSMatrix l21, CTSMatrix a, long int start_index, long int min_dim);
int CTSLUdecomp_u12_omp(CTSMatrix u12, CTSMatrix a, long int start_index, long int min_dim);
int CTSLUdecomp_a22_omp(CTSMatrix a, CTSMatrix d22, CTSMatrix l21, CTSMatrix u12, long int start_index, long int min_dim);
int CTSLUdecomp_omp(CTSMatrix a);
int CTSLUdecompPM_omp(CTSMatrix a, long int ch[]);
int CTSLUdecomp_strassen_omp(CTSMatrix a, long int min_dim);
int CTSLUdecomp_strassenPM_omp(CTSMatrix a, long int ch[], long int min_dim);
#endif // _OPENMP

//--------------------------------------/
// ctdlu_oz.c
//--------------------------------------/
int CTSLUdecomp_a22_oz(CTSMatrix a, CTSMatrix d22, CTSMatrix l21, CTSMatrix u12, long int start_index, long int min_dim, int max_num_div);
int CTSLUdecomp_oz(CTSMatrix a, long int min_dim, int max_num_div);
int CTSLUdecomp_ozPM(CTSMatrix a, long int ch[], long int min_dim, int max_num_div);

#ifdef _OPENMP
int CTSLUdecomp_a22_oz_omp(CTSMatrix a, CTSMatrix d22, CTSMatrix l21, CTSMatrix u12, long int start_index, long int min_dim, int max_num_div);
int CTSLUdecomp_oz_omp(CTSMatrix a, long int min_dim, int max_num_div);
int CTSLUdecomp_ozPM_omp(CTSMatrix a, long int ch[], long int min_dim, int max_num_div);
#endif // _OPENMP
/**************************************/
/* fread_write.c                      */
/**************************************/
/* undefined
void fread_tsmatrix(FILE *fp, TSMatrix mat);
void fread_tsmatrix(FILE *fp, TSMatrix mat);
void fread_tsmatrix_fname(const char *fname, TSMatrix mat);
void fwrite_tsmatrix(FILE *fp, TSMatrix mat);
void fwrite_tsmatrix_fname(const char *fname, TSMatrix mat);
//void fread_tdpolycoef(FILE *fp, TDPoly p, long int maxdeg);
//void fread_tdpolycoef_fname(const char *fname, TDPoly p, long int maxdeg);
void fread_tsvector(FILE *fp, TSVector vec);
void fread_tsvector_fname(const char *fname, TSVector vec);
void fwrite_tsvector(FILE *fp, TSVector vec);
void fwrite_tsvector_fname(const char *fname, TSVector vec);
*/
// 2023-12-15(Fri) T.Kouya
// NOTE: removed stray mis-named declaration `read_test_linear_eq_ctd(CTSMatrix,...)`
// (copy-paste leftover — never defined or called; the real read_test_linear_eq_ctd
// takes CTDMatrix and lives in ctdlinear.h). It collided by name with the ctd
// version, breaking any TU that includes both ctdlinear.h and ctslinear.h. 2026-06-16.

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __BNC_CTSLINEAR_H__
