/********************************************************************************/
/* ctdlinear.h: Triple-double precision Complex Linear Computation Library      */
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
// define __BNC_CTDLINEAR_H__
#ifndef __BNC_CTDLINEAR_H__
  #define __BNC_CTDLINEAR_H__

#include "rdd.h"
#include "rcdd.h" // Complex TD arithmetic

#include "cdlinear.h"
#include "cddlinear.h"
#include "tdlinear.h"
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
    TDVector re; // Real part
    TDVector im; // Imaginary part
} ctdvector;

typedef ctdvector *CTDVector;

// SIMD: AVX2 and AVX-512
//#if defined(__AVX2__) || defined(__AVX512F__) // __AVX2__ or __AVX512F__
//#include "bncavx.h"
//#endif // defined(__AVX2__) || defined(__AVX512F__)

// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
#elif defined(__AVX512F__) // __AVX512F__
#else // others
#endif // __AVX2__

// get_ctdvector_i_ctdfloat
static inline ctdfloat get_ctdvector_i_ctdfloat(CTDVector vec, long int index)
{
	//static 
	ctdfloat ret;

	rtd_set(ret.val_re, get_tdvector_i(vec->re, index));
    rtd_set(ret.val_im, get_tdvector_i(vec->im, index));

	return ret;
}
// subst_ctdvector_i
// ret := get_ctdvector_i(vec, index)
static inline void subst_ctdvector_i(ctdfloat *ret, CTDVector vec, long int index)
{
	rtd_set(ret->val_re, get_tdvector_i(vec->re, index));
	rtd_set(ret->val_im, get_tdvector_i(vec->im, index));
} 
// Very dangerous!! 2024-04-18(Thu) T.Kouya
//	GET_CDDVECTOR_I(vec, index))
#define GET_CTDVECTOR_I(vec, index) ((ctdfloat *)&(get_ctdvector_i_ctdfloat((vec), (index))))
//#define get_ctdvector_i(vec, index) ((ctdfloat *)&(get_ctdvector_i_ctdfloat((vec), (index))))
static inline ctdfloat *get_ctdvector_i(CTDVector vec, long int index)
{
	//static is neccesary here! 2024-09-23 T.Kouya
	static ctdfloat ret, *ptr_ret;

	ptr_ret = &ret;

	rtd_set(ptr_ret->val_re, get_tdvector_i(vec->re, index));
	rtd_set(ptr_ret->val_im, get_tdvector_i(vec->im, index));

	return ptr_ret;
}

// set_ctdvector_i
static inline void set_ctdvector_i(CTDVector vec, long int index, ctdfloat *val) // val[TDSIZE]
{
    set_tdvector_i(vec->re, index, val->val_re);
    set_tdvector_i(vec->im, index, val->val_im);
}
#define SET_CTDVECTOR_I(vec, index, val) set_ctdvector_i((vec), (index), (val))

// set_ctdvector_i_d
static inline void set_ctdvector_i_d(CTDVector vec, long int index, double val) // val
{
    set_tdvector_i_d(vec->re, index, val);
    set0_tdvector_i(vec->im, index);
}
#define SET_CTDVECTOR_I_D(vec, index, val) set_ctdvector_i_d((vec), (index), (val))

// set_ctdvector_i_cd
static inline void set_ctdvector_i_cd(CTDVector vec, long int index, double _Complex val) // val
{
    set_tdvector_i_d(vec->re, index, __real__ val); //creal(val));
    set_tdvector_i_d(vec->im, index, __imag__ val); //cimag(val));
}

// set0_ctdvector_i
static inline void set0_ctdvector_i(CTDVector vec, long int index)
{
	set0_tdvector_i(vec->re, index);
	set0_tdvector_i(vec->im, index);
}
#define SET0_CTDVECTOR_I(vec, index) set0_ctdvector_i((vec), (index))

// initialize CTDVector
CTDVector init_ctdvector(int dimension);

// free CTDVector
void free_ctdvector(CTDVector vec);

// CTDVector vec -> ctdfloat array
void set_ctdfloat_ctdvec(ctdfloat ret[], int ret_dim, CTDVector vec);

// tdvector -> ctdvector
void set_ctdvector_tdvec(CTDVector ret, TDVector re_vec, TDVector im_vec);

// ctdfloat array -> CTDVector ret
void set_ctdvector_ctdfloat(CTDVector ret, ctdfloat array[], int array_dim);

// print tdvector
void print_ctdvector(CTDVector vec);

// set a zero vector
void set0_ctdvector(CTDVector vec);

// set_ctdvector_i_str
void set_ctdvector_i_str(CTDVector vec, long int index, const char *str);

/*************************************************/
/* Vector Calculations for CTDVector               */
/*
void add_ctdvector(CTDVector c, CTDVector a, CTDVector b)
void add2_ctdvector(CTDVector c, CTDVector a)
void sub_ctdvector(CTDVector c, CTDVector a, CTDVector b)
void sub2_ctdvector(CTDVector c, CDVector a)
void cmul_ctdvector(CTDVector c, ctdfloat *val, CTDVector a)
void cmul2_ctdvector(CTDVector c, ctdfloat *val)
void add_cmul_ctdvector(CTDVector c, CTDVector a, ctdfloat *val, CTDVector b)
double ip_ctdvector(CTDVector a, CTDVector b)
double norm1_ctdvector(CTDVector a)
double norm2_ctdvector(CTDVector a)
double normi_ctdvector(CTDVector a)
void subst_ctdvector(CTDVector c, CTDVector a)
*/
/*************************************************/
/* c = a + b */
void add_ctdvector(CTDVector c, CTDVector a, CTDVector b);

/* c += a */
void add2_ctdvector(CTDVector c, CTDVector a);

/* c = a - b */
void sub_ctdvector(CTDVector c, CTDVector a, CTDVector b);

/* c -= a */
void sub2_ctdvector(CTDVector c, CTDVector a);

/* c = val * a */
void cmul_ctdvector_4m(CTDVector c, ctdfloat *val, CTDVector a);
void cmul_ctdvector_3m(CTDVector c, ctdfloat *val, CTDVector a);
#ifdef USE_4M
#define cmul_ctdvector cmul_ctdvector_4m
#else // USE_4M
//#define cmul_ctdvector cmul_ctdvector_3m
#define cmul_ctdvector cmul_ctdvector_4m // faster than 3m on TD arithemtic
#endif // USE_4M

/* c *= val */
void cmul2_ctdvector(CTDVector c, ctdfloat *val);

/* c = a + val * b */
void add_cmul_ctdvector(CTDVector c, CTDVector a, ctdfloat *val, CTDVector b);

/* c = a - val * b */
void sub_cmul_ctdvector(CTDVector c, CTDVector a, ctdfloat *val, CTDVector b);

/* (a, b) */
void ip_ctdvector(ctdfloat *ret, CTDVector a, CTDVector b);

/* a^T * b */
void dotp_ctdvector(ctdfloat *ret, CTDVector a, CTDVector b);

/* c := a */
void subst_ctdvector(CTDVector c, CTDVector a);

/* c := -a */
void neg_ctdvector(CTDVector c, CTDVector a);

/* ||a||_1 */
void norm1_ctdvector(double ret[TDSIZE], CTDVector a);

/* ||a||_infty */
void normi_ctdvector(double ret[TDSIZE], CTDVector a);

// Euclid norm
void norm2_ctdvector(double ret[TDSIZE], CTDVector vec);

// print ctdvector
void print_ctdvector(CTDVector vec);

// CTD matrix
typedef struct{
	TDMatrix re; // Real part
	TDMatrix im; // Imaginary part
} ctdmatrix;

typedef ctdmatrix *CTDMatrix;

// get_ctdmatrix_ij
static inline ctdfloat get_ctdmatrix_ij_ctdfloat(CTDMatrix mat, long int i, long int j)
{
	long int ij_index;
	//static 
	ctdfloat ret;

	ij_index = mat->re->real_col_dim * i + j;

	rtd_set(ret.val_re, get_tdmatrix_ij(mat->re, i, j));
	rtd_set(ret.val_im, get_tdmatrix_ij(mat->im, i, j));

	return ret;
}
// subst_ctdmatrix_ij
// ret := get_ctdmatrix_ij(mat, i, j)
static inline void subst_ctdmatrix_ij(ctdfloat *ret, CTDMatrix mat, long int i, long int j)
{
	long int ij_index;

	rtd_set(ret->val_re, get_tdmatrix_ij(mat->re, i, j));
	rtd_set(ret->val_im, get_tdmatrix_ij(mat->im, i, j));

	//return ret;
} 
// Very dangerous!! 2024-04-18(Thu) T.Kouya
#define GET_CTDMATRIX_IJ(mat, i, j) (&(get_ctdmatrix_ij_ctdfloat((mat), (i), (j))))
//#define get_ctdmatrix_ij(mat, i, j) (&(get_ctdmatrix_ij_ctdfloat((mat), (i), (j))))
static inline ctdfloat *get_ctdmatrix_ij(CTDMatrix mat, long int i, long int j)
{
	long int ij_index;
	//static is neccesary here! 2024-09-23 T.Kouya
	static ctdfloat ret, *ptr_ret;

	ij_index = mat->re->real_col_dim * i + j;
	ptr_ret = &ret;

	rtd_set(ptr_ret->val_re, get_tdmatrix_ij(mat->re, i, j));
	rtd_set(ptr_ret->val_im, get_tdmatrix_ij(mat->im, i, j));

	return ptr_ret;
} 

// set_ctdmatrix_ij
static inline void set_ctdmatrix_ij(CTDMatrix mat, long int i, long int j, ctdfloat *val) // val[TDSIZE]
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_tdmatrix_ij(mat->re, i, j, val->val_re);
	set_tdmatrix_ij(mat->im, i, j, val->val_im);

	return;
} 
#define SET_CTDMATRIX_IJ(mat, i, j, val) set_ctdmatrix_ij((mat), (i), (j), (val))

// set_ctdmatrix_ij_d
static inline void set_ctdmatrix_ij_d(CTDMatrix mat, long int i, long int j, double val)
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_tdmatrix_ij_d(mat->re, i, j, val);
	set0_tdmatrix_ij(mat->im, i, j);

	return;
} 
#define SET_CTDMATRIX_IJ_D(mat, i, j, val) set_ctdmatrix_ij_d((mat), (i), (j), (val))
#define SET_CTDMATRIX_IJ_UI(mat, i, j, val) set_ctdmatrix_ij_d((mat), (i), (j), (double)(val))
#define set_ctdmatrix_ij_ui(mat, i, j, val) set_ctdmatrix_ij_d((mat), (i), (j), (double)(val))

// set_ctdmatrix_ij_td
static inline void set_ctdmatrix_ij_td(CTDMatrix mat, long int i, long int j, double val[TDSIZE])
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_tdmatrix_ij(mat->re, i, j, val);
	set0_tdmatrix_ij(mat->im, i, j);

	return;
} 

// set_ctdmatrix_ij_cd
static inline void set_ctdmatrix_ij_cd(CTDMatrix mat, long int i, long int j, double _Complex val)
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set_tdmatrix_ij_d(mat->re, i, j, __real__ val); //creal(val));
	set_tdmatrix_ij_d(mat->im, i, j, __imag__ val); //cimag(val));

	return;
} 

// set0_ctdmatrix_ij
static inline void set0_ctdmatrix_ij(CTDMatrix mat, long int i, long int j)
{
	long int ij_index;

	ij_index = mat->re->real_col_dim * i + j;

	set0_tdmatrix_ij(mat->re, i, j);
	set0_tdmatrix_ij(mat->im, i, j);

	return;
}
#define SET0_CTDMATRIX_IJ(mat, i, j) set0_ctdmatrix_ij((mat), (i), (j))

// set a zero matrix
//void set0_ctdmatrix(CTDMatrix mat);
void set0_ctdmatrix(CTDMatrix mat);

// initialize tdvector
CTDMatrix init_ctdmatrix(long int row_dim, long int col_dim);

// free tdvector
void free_ctdmatrix(CTDMatrix mat);

// print tdvector
void print_ctdmatrix(CTDMatrix mat);

// CTDMatrix mat -> ctdfloat array
void set_ctdfloat_ctdmat(ctdfloat ret[], int ret_dim, CTDMatrix mat);

// tdmatrix -> ctdmatrix
void set_ctdmatrix_tdmat(CTDMatrix ret, TDMatrix re_mat, TDMatrix im_mat);

// ctdfloat array -> CTDmatrix ret
void set_ctdmatrix_ctdfloat(CTDMatrix ret, ctdfloat array[], int array_dim);

// matrix multiplication
// ret := A * B
void mul_ctdmatrix_4m(CTDMatrix ret, CTDMatrix a, CTDMatrix b);
void mul_ctdmatrix_3m(CTDMatrix ret, CTDMatrix a, CTDMatrix b);
#ifdef USE_4M
#define mul_ctdmatrix mul_ctdmatrix_4m
#else // USE_4M
#define mul_ctdmatrix mul_ctdmatrix_3m
#endif // USE_4M

// Frobenius norm
void normf_ctdmatrix(double ret[TDSIZE], CTDMatrix mat);

// print normf
void print_normf_ctdmatrix(const char *str, CTDMatrix mat);

/*************************************************/
/* Matrix Caluculations for CTDMatrix            */
/*
void normf_ctdmatrix(double ret[TDSIZE], CTDMatrix mat)
void norm1_ctdmatrix(double ret[TDSIZE], CTDMatrix mat)
void normi_ctdmatrix(double ret[TDSIZE], CTDMatrix mat)
void add_ctdmatrix(CTDMatrix c, CTDMatrix a, CTDMatrix b);
void sub_ctdmatrix(CTDMatrix c, CTDMatrix a, CTDMatrix b);
void mul_ctdmatrix(CTDMatrix c, CTDMatrix a, CTDMatrix b);
void mul_ctdmatrix_ctdvec(CTDVector v, CTDMatrix a, CTDVector vb)
void mul_ctdmatrixt_ctdvec(CTDVector v, CTDMatrix a, CTDVector vb)
void transpose_ctdmatrix(CTDMatrix c, CTDMatrix a);
void inv_ctdmatrix(CTDMatrix a);
void subst_ctdmatrix(CTDMatrix c, CTDMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_ctdmatrix(double ret[TDSIZE], CTDMatrix mat);

/* 1 Norm of Matrix */
void norm1_ctdmatrix(double ret[TDSIZE], CTDMatrix mat);

/* c := a + b */
void add_ctdmatrix(CTDMatrix c, CTDMatrix a, CTDMatrix b);

/* c := a - b */
void sub_ctdmatrix(CTDMatrix c, CTDMatrix a, CTDMatrix b);

/* c := sc * a */
void cmul_ctdmatrix(CTDMatrix c, ctdfloat *sc, CTDMatrix a);

/* c = a^T */
void transpose_ctdmatrix(CTDMatrix c, CTDMatrix a);

/* c := conj(a)^T */
void star_ctdmatrix(CTDMatrix c, CTDMatrix a);

/* c := a */
void subst_ctdmatrix(CTDMatrix c, CTDMatrix a);

/* c := conj(a) */
void conj_ctdmatrix(CTDMatrix c, CTDMatrix a);

/* c := -a */
void neg_ctdmatrix(CTDMatrix c, CTDMatrix a);

/* c := a */
void subst_ctdmatrix(CTDMatrix c, CTDMatrix a);

/* c := I */
void setI_ctdmatrix(CTDMatrix c);

/* v := a * vb */
void mul_ctdmatrix_ctdvec_4m(CTDVector v, CTDMatrix a, CTDVector vb);
void mul_ctdmatrix_ctdvec_3m(CTDVector v, CTDMatrix a, CTDVector vb);
#ifdef USE_4M
#define mul_ctdmatrix_ctdvec mul_ctdmatrix_ctdvec_4m
#else // USE_4M
#define mul_ctdmatrix_ctdvec mul_ctdmatrix_ctdvec_3m
#endif // USE_4M

/* v := a^T * vb */
void mul_ctdmatrixt_ctdvec(CTDVector v, CTDMatrix a, CTDVector vb);

/* v := conj(a)^T * vb */
void mul_ctdmatrixs_ctdvec(CTDVector v, CTDMatrix a, CTDVector vb);

/* a = a^(-1) */
/* square matrix only */
void inv_ctdmatrix(CTDMatrix a);

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpf)a */
void subst_cmpfvector_ctdvec(CMPFVector c, CTDVector a);

/* c := (ctd)a */
void subst_ctdvector_cmpfvec(CTDVector c, CMPFVector a);

/* c := (cmpf)a */
void subst_cmpfmatrix_ctdmat(CMPFMatrix c, CTDMatrix a);

/* c := (ctd)a */
void subst_ctdmatrix_cmpfmat(CTDMatrix c, CMPFMatrix a);

/* Normwise relative error of vector */
void relerr_ctdvector_cmpfvec(double relerr[TDSIZE], CTDVector approx_vec, CMPFVector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_ctdvector_mpf(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double norm_relerr[TDSIZE], CTDVector approx_vec, MPFVector true_vec, int norm_type);
#endif // USE_GMP

/* c := (ctd)a */
void subst_ctdvector_cdvec(CTDVector c, CDVector a);

/* c := (d)a */
void subst_cdvector_ctdvec(CDVector c, CTDVector a);

/* c := (ctd)a */
void subst_ctdmatrix_cdmat(CTDMatrix c, CDMatrix a);

/* c := (d)a */
void subst_cdmatrix_ctdmat(CDMatrix c, CTDMatrix a);

/* c := (td)a */
void subst_ctdvector_cddvec(CTDVector c, CDDVector a);

/* c := (dd)a */
void subst_cddvector_ctdvec(CDDVector c, CTDVector a);

/* c := (td)a */
void subst_ctdmatrix_cddmat(CTDMatrix c, CDDMatrix a);

/* c := (dd)a */
void subst_cddmatrix_ctdmat(CDDMatrix c, CTDMatrix a);


/* Normwise relative error of vector */
void relerr_ctdvector(double relerr[TDSIZE], CTDVector approx_vec, CTDVector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_ctdvector(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double norm_relerr[TDSIZE], CTDVector approx_vec, CTDVector true_vec, int norm_type);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_ctdmatrix(CTDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

// print_ctdmatrix
void print_ctdmatrix(CTDMatrix mat);

/**************************************/
/* ctdlu.c                             */
/**************************************/
int CTDLUdecomp(CTDMatrix a);
int SolveCTDLS(CTDVector answer, CTDMatrix lu, CTDVector b);
int CTDLUdecompP(CTDMatrix a, long int ch[]);
int SolveCTDLSP(CTDVector answer, CTDMatrix lu, CTDVector b, long int ch[]);
int CTDLUdecompC(CTDMatrix a, long int row_ch[], long int col_ch[]);
int SolveCTDLSC(CTDVector answer, CTDMatrix lu, CTDVector b, long int row_ch[], long int col_ch[]);
int CTDLUdecompPM(CTDMatrix a, long int ch[]);
int SolveCTDLSPM(CTDVector answer, CTDMatrix lu, CTDVector b, long int ch[]);

//--------------------------------------/
// ctdlu_strassen.c
//--------------------------------------/
int CTDLUdecomp_square(CTDMatrix a, long int start_index, long int min_dim);
int CTDLUdecomp_l21(CTDMatrix l21, CTDMatrix a, long int start_index, long int min_dim);
int CTDLUdecomp_u12(CTDMatrix u12, CTDMatrix a, long int start_index, long int min_dim);
int CTDLUdecomp_a22(CTDMatrix a, CTDMatrix d22, CTDMatrix l21, CTDMatrix u12, long int start_index, long int min_dim);
int CTDLUdecomp_strassen(CTDMatrix a, long int min_dim);
int CTDLUdecomp_strassenPM(CTDMatrix a, long int ch[], long int min_dim);

#ifdef _OPENMP
int CTDLUdecomp_square_omp(CTDMatrix a, long int start_index, long int min_dim);
int CTDLUdecomp_l21_omp(CTDMatrix l21, CTDMatrix a, long int start_index, long int min_dim);
int CTDLUdecomp_u12_omp(CTDMatrix u12, CTDMatrix a, long int start_index, long int min_dim);
int CTDLUdecomp_a22_omp(CTDMatrix a, CTDMatrix d22, CTDMatrix l21, CTDMatrix u12, long int start_index, long int min_dim);
int CTDLUdecomp_omp(CTDMatrix a);
int CTDLUdecompPM_omp(CTDMatrix a, long int ch[]);
int CTDLUdecomp_strassen_omp(CTDMatrix a, long int min_dim);
int CTDLUdecomp_strassenPM_omp(CTDMatrix a, long int ch[], long int min_dim);
#endif // _OPENMP

//--------------------------------------/
// ctdlu_oz.c
//--------------------------------------/
int CTDLUdecomp_a22_oz(CTDMatrix a, CTDMatrix d22, CTDMatrix l21, CTDMatrix u12, long int start_index, long int min_dim, int max_num_div);
int CTDLUdecomp_oz(CTDMatrix a, long int min_dim, int max_num_div);
int CTDLUdecomp_ozPM(CTDMatrix a, long int ch[], long int min_dim, int max_num_div);

#ifdef _OPENMP
int CTDLUdecomp_a22_oz_omp(CTDMatrix a, CTDMatrix d22, CTDMatrix l21, CTDMatrix u12, long int start_index, long int min_dim, int max_num_div);
int CTDLUdecomp_oz_omp(CTDMatrix a, long int min_dim, int max_num_div);
int CTDLUdecomp_ozPM_omp(CTDMatrix a, long int ch[], long int min_dim, int max_num_div);
#endif // _OPENMP
/**************************************/
/* fread_write.c                      */
/**************************************/
/* undefined
void fread_tdmatrix(FILE *fp, TDMatrix mat);
void fread_tdmatrix(FILE *fp, TDMatrix mat);
void fread_tdmatrix_fname(const char *fname, TDMatrix mat);
void fwrite_tdmatrix(FILE *fp, TDMatrix mat);
void fwrite_tdmatrix_fname(const char *fname, TDMatrix mat);
//void fread_tdpolycoef(FILE *fp, TDPoly p, long int maxdeg);
//void fread_tdpolycoef_fname(const char *fname, TDPoly p, long int maxdeg);
void fread_tdvector(FILE *fp, TDVector vec);
void fread_tdvector_fname(const char *fname, TDVector vec);
void fwrite_tdvector(FILE *fp, TDVector vec);
void fwrite_tdvector_fname(const char *fname, TDVector vec);
*/
// 2023-12-15(Fri) T.Kouya
// read problem from file
void read_test_linear_eq_ctd(CTDMatrix A, CTDVector true_x, CTDVector b, long int dim, const char *fname_A, const char *fname_true_x, const char *fname_b);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __BNC_CTDLINEAR_H__
