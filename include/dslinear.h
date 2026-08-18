/********************************************************************************/
/* dslinear.h: float-signel precision Linear Computation Library               */
/* Copyright (C) 2021      Tomonori Kouya                                       */
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
// define __BNC_DSLINEAR_H__
#ifndef __BNC_DSLINEAR_H__
  #define __BNC_DSLINEAR_H__

#include "rds.h"

#include "flinear.h"

#ifdef USE_GMP
#include "gmp.h"
#include "mpfr.h"
#include "mpfr_dtq_sd.h"
#include "mpflinear.h"
#endif //USE_GMP//

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#if defined(__AVX2__) || defined(__AVX512F__)
#include "avx2/bncavx.h"
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2)
#include "sve2/bncsve2.h"
#include "neon/bncneon.h"   /* SVE2 build keeps NEON for the #elif fallback paths */
#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON)
#include "neon/bncneon.h"
#endif // defined(__AVX2__) || defined(__AVX512F__)

// DS vector
typedef struct
{
	long int dim; // dim <= real_dim
	long int real_dim; // multiplier of _BNC_D_WIDTH
    float *element[DSSIZE];
} dsvector;

typedef dsvector *DSVector;

// SIMD: AVX2 and AVX-512
//#if defined(__AVX2__) || defined(__AVX512F__) // __AVX2__ or __AVX512F__
//#include "bncavx.h"
//#endif // defined(__AVX2__) || defined(__AVX512F__)

// SIMD : for copy & paste
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
#elif defined(__AVX512F__) // __AVX512F__
#else // others
#endif // __AVX2__

// get_dsvector_i
static inline dsfloat get_dsvector_i_dsfloat(DSVector vec, long int index)
//static inline float *get_dsvector_i(DSVector vec, long int index)
{
//	float ret[DSSIZE];
	dsfloat ret;

	ret.val[0] = vec->element[0][index];
	ret.val[1] = vec->element[1][index];

	return ret;
}

//	GET_dsvector_I(vec, index))
#define GET_DSVECTOR_I(vec, index) ((get_dsvector_i_dsfloat((vec), (index)).val))
#define get_dsvector_i(vec, index) ((get_dsvector_i_dsfloat((vec), (index)).val))

// set_dsvector_i
static inline void set_dsvector_i(DSVector vec, long int index, float *val) // val[DSSIZE]
{
	float ret[DSSIZE];

	vec->element[0][index] = val[0];
	vec->element[1][index] = val[1];
}
#define SET_DSVECTOR_I(vec, index, val) set_dsvector_i((vec), (index), (val))

// set_dsvector_i_d
static inline void set_dsvector_i_d(DSVector vec, long int index, float val) // val
{
	float ret[DSSIZE];

	vec->element[0][index] = val;
	vec->element[1][index] = 0.0;
}
#define SET_dsvector_I_D(vec, index, val) set_dsvector_i_d((vec), (index), (val))

// set0_dsvector_i
static inline void set0_dsvector_i(DSVector vec, long int index)
{
	float ret[DSSIZE];

	vec->element[0][index] = 0.0;
	vec->element[1][index] = 0.0;
}
#define SET0_dsvector_I(vec, index) set0_dsvector_i((vec), (index))

// dsrel_diff
dsfloat dsrel_diff(dsfloat a, dsfloat b);

dsfloat dsrel_diff_array(dsfloat approx_a[], dsfloat approx_b[], int dim, int print_flag);

// Frobenius norm
dsfloat dsnormf(dsfloat array[], int dim);

// generate a text vector: vec(i) := sqrt(sqrt_seed) * (i + 1);
void set_test_dsvector(dsfloat vec[], int sqrt_seed, int dim);

// initialize DSVector
DSVector init_dsvector(int dimension);

// free DSVector
void free_dsvector(DSVector vec);

// DSVector vec -> dsfloat array
void set_dsfloat_dsvec(dsfloat ret[], int ret_dim, DSVector vec);

// dsfloat array -> DSVector ret
void set_dsvector_dsfloat(DSVector ret, dsfloat array[], int array_dim);

// print DSVector
void print_dsvector(DSVector vec);

// set a zero vector
void set0_dsvector(DSVector vec);

// set_dsvector_i_str
void set_dsvector_i_str(DSVector vec, long int index, const char *str);

/*************************************************/
/* Vector Calculations for DSVector               */
/*
void add_dsvector(DSVector c, DSVector a, DSVector b)
void add2_dsvector(DSVector c, DSVector a)
void sub_dsvector(DSVector c, DSVector a, DSVector b)
void sub2_dsvector(DSVector c, FVector a)
void cmul_dsvector(DSVector c, float val[DSSIZE], DSVector a)
void cmul2_dsvector(DSVector c, float val[DSSIZE])
void add_cmul_dsvector(DSVector c, DSVector a, float val[DSSIZE], DSVector b)
float ip_dsvector(DSVector a, DSVector b)
float norm1_dsvector(DSVector a)
float norm2_dsvector(DSVector a)
float normi_dsvector(DSVector a)
void subst_dsvector(DSVector c, DSVector a)
*/
/*************************************************/
/* c = a + b */
void add_dsvector(DSVector c, DSVector a, DSVector b);

/* c += a */
void add2_dsvector(DSVector c, DSVector a);

/* c = a - b */
void sub_dsvector(DSVector c, DSVector a, DSVector b);

/* c -= a */
void sub2_dsvector(DSVector c, DSVector a);

/* c = val * a */
void cmul_dsvector(DSVector c, float val[DSSIZE], DSVector a);

/* c *= val */
void cmul2_dsvector(DSVector c, float val[DSSIZE]);

/* c = a + val * b */
void add_cmul_dsvector(DSVector c, DSVector a, float val[DSSIZE], DSVector b);
void sub_cmul_dsvector(DSVector c, DSVector a, float val[DSSIZE], DSVector b);

/* (a, b) */
void ip_dsvector(float ret[DSSIZE], DSVector a, DSVector b);

/* c := a */
void subst_dsvector(DSVector c, DSVector a);

/* c := -a */
void neg_dsvector(DSVector c, DSVector a);

/* ||a||_1 */
void norm1_dsvector(float ret[DSSIZE], DSVector a);

/* ||a||_infty */
void normi_dsvector(float ret[DSSIZE], DSVector a);

// Euclid norm
void norm2_dsvector(float ret[DSSIZE], DSVector vec);

#if defined(__AVX2__)
/* float-float = float + float */
void _bncavx2_dsadd_f_f(dsfloat ret[], float a[], float b[], int dim);

/* float-float = float + float */
//void _bncavx2_ddvadd_d_d(DSVector *ret, float a[], float b[], int dim)
void _bncavx2_dsvadd_f_f(DSVector ret, float a[], float b[], int dim);

/* add */
void _bncavx2_dsvadd(DSVector ret, DSVector a, DSVector b, int dim);

/* add */
void _bncavx2_dsadd(dsfloat ret[], dsfloat a[], dsfloat b[], int dim);

/* float-float = float * float */
void _bncavx2_dsvmul_f_f(DSVector ret, float a[], float b[], int dim);

/* dsmul */
void _bncavx2_dsmul(dsfloat ret[], dsfloat a[], dsfloat b[], int dim);

/* dsdiv */
void _bncavx2_dsdiv(dsfloat ret[], dsfloat a[], dsfloat b[], int dim);

/* div */
void _bncavx2_dsvdiv(DSVector ret, DSVector a, DSVector b, int dim);

void _bncavx2_dsvmul(DSVector ret, DSVector a, DSVector b, int dim);
#endif // __AVX2__

// ds matrix
typedef struct{
	long int row_dim, col_dim;
	long int real_row_dim, real_col_dim; // multiplier of _BNC_D_WIDTH
	float *element[DSSIZE];
} dsmatrix;

typedef dsmatrix *DSMatrix;

// get_dsmatrix_ij
static inline dsfloat get_dsmatrix_ij_dsfloat(DSMatrix mat, long int i, long int j)
{
	long int ij_index;
//	float ret[DSSIZE];
	dsfloat ret;

	ij_index = mat->real_col_dim * i + j;

	ret.val[0] = mat->element[0][ij_index];
	ret.val[1] = mat->element[1][ij_index];

	return ret;
} 
#define GET_DSMATRIX_IJ(mat, i, j) ((get_dsmatrix_ij_dsfloat((mat), (i), (j)).val))
#define get_dsmatrix_ij(mat, i, j) ((get_dsmatrix_ij_dsfloat((mat), (i), (j)).val))

// set_dsmatrix_ij
static inline void set_dsmatrix_ij(DSMatrix mat, long int i, long int j, float *val) // val[DSSIZE]
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[0][ij_index] = val[0];
	mat->element[1][ij_index] = val[1];

	return;
} 
#define SET_DSMATRIX_IJ(mat, i, j, val) set_dsmatrix_ij((mat), (i), (j), (val))

// set_dsmatrix_ij_d
static inline void set_dsmatrix_ij_f(DSMatrix mat, long int i, long int j, float val)
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[0][ij_index] = val;
	mat->element[1][ij_index] = 0.0f;

	return;
} 
#define SET_DSMATRIX_IJ_F(mat, i, j, val) set_dsmatrix_ij_f((mat), (i), (j), (val))
#define SET_DSMATRIX_IJ_UI(mat, i, j, val) set_dsmatrix_ij_f((mat), (i), (j), (float)(val))
#define set_dsmatrix_ij_ui(mat, i, j, val) set_dsmatrix_ij_f((mat), (i), (j), (float)(val))

// set0_dsmatrix_ij
static inline void set0_dsmatrix_ij(DSMatrix mat, long int i, long int j)
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[0][ij_index] = 0.0;
	mat->element[1][ij_index] = 0.0;

	return;
}
#define SET0_DSMATRIX_IJ(mat, i, j) set0_dsmatrix_ij((mat), (i), (j))

// set a zero matrix
//void set0_dsmatrix(DSMatrix mat);
void set0_dsmatrix(DSMatrix mat);

// initialize DSVector
DSMatrix init_dsmatrix(long int row_dim, long int col_dim);

// free DSVector
void free_dsmatrix(DSMatrix mat);

// print DSVector
void print_dsmatrix(DSMatrix mat);

// DSMatrix mat -> dsfloat array
void set_dsfloat_dsmat(dsfloat ret[], int ret_dim, DSMatrix mat);

// dsfloat array -> dsmatrix ret
void set_dsmatrix_dsfloat(DSMatrix ret, dsfloat array[], int array_dim);

// matrix multiplication
// ret := A * B
void mul_dsmatrix(DSMatrix ret, DSMatrix a, DSMatrix b);

// Frobenius norm
void normf_dsmatrix(float ret[DSSIZE], DSMatrix mat);

// print normf
void print_normf_dsmatrix(const char *str, DSMatrix mat);

/*************************************************/
/* Matrix Caluculations for DSMatrix            */
/*
void normf_dsmatrix(float ret[DSSIZE], DSMatrix mat)
void norm1_dsmatrix(float ret[DSSIZE], DSMatrix mat)
void normi_dsmatrix(float ret[DSSIZE], DSMatrix mat)
void add_dsmatrix(DSMatrix c, DSMatrix a, DSMatrix b);
void sub_dsmatrix(DSMatrix c, DSMatrix a, DSMatrix b);
void mul_dsmatrix(DSMatrix c, DSMatrix a, DSMatrix b);
void mul_dsmatrix_dsvec(DSVector v, DSMatrix a, DSVector vb)
void mul_dsmatrixt_dsvec(DSVector v, DSMatrix a, DSVector vb)
void transpose_dsmatrix(DSMatrix c, DSMatrix a);
void inv_dsmatrix(DSMatrix a);
void subst_mpfmatrux(DSMatrix c, DSMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_dsmatrix(float ret[DSSIZE], DSMatrix mat);

/* 1 Norm of Matrix */
void norm1_dsmatrix(float ret[DSSIZE], DSMatrix mat);

/* c := a + b */
void add_dsmatrix(DSMatrix c, DSMatrix a, DSMatrix b);

/* c := a - b */
void sub_dsmatrix(DSMatrix c, DSMatrix a, DSMatrix b);

/* c := sc * a */
void cmul_dsmatrix(DSMatrix c, float sc[DSSIZE], DSMatrix a);

/* c = a^T */
void transpose_dsmatrix(DSMatrix c, DSMatrix a);

/* c := a */
void subst_dsmatrix(DSMatrix c, DSMatrix a);

/* c := I */
void setI_dsmatrix(DSMatrix c);

/* v := a * vb */
void mul_dsmatrix_dsvec(DSVector v, DSMatrix a, DSVector vb);

/* v := a^T * vb */
void mul_dsmatrixt_dsvec(DSVector v, DSMatrix a, DSVector vb);

/* a = a^(-1) */
/* square matrix only */
void inv_dsmatrix(DSMatrix a);

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpf)a */
void subst_mpfvector_dsvec(MPFVector c, DSVector a);

/* c := (dd)a */
void subst_dsvector_mpfvec(DSVector c, MPFVector a);

/* c := (mpf)a */
void subst_mpfmatrix_dsmat(MPFMatrix c, DSMatrix a);

/* c := (dd)a */
void subst_dsmatrix_mpfmat(DSMatrix c, MPFMatrix a);

/* Normwise relative error of vector */
void relerr_dsvector_mpfvec(float relerr[DSSIZE], DSVector approx_vec, MPFVector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_dsvector_mpf(float max_relerr[DSSIZE], float min_relerr[DSSIZE], float norm_relerr[DSSIZE], DSVector approx_vec, MPFVector true_vec, int norm_type);
#endif // USE_GMP

/* c := (dd)a */
void subst_dsvector_fvec(DSVector c, FVector a);

/* c := (d)a */
void subst_fvector_dsvec(FVector c, DSVector a);

/* c := (dd)a */
void subst_dsmatrix_fmat(DSMatrix c, FMatrix a);

/* c := (d)a */
void subst_fmatrix_dsmat(FMatrix c, DSMatrix a);

/* Normwise relative error of vector */
void relerr_dsvector(float relerr[DSSIZE], DSVector approx_vec, DSVector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_dsvector(float max_relerr[DSSIZE], float min_relerr[DSSIZE], float norm_relerr[DSSIZE], DSVector approx_vec, DSVector true_vec, int norm_type);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_dsmatrix(DSMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

/**************************************/
/* dslu.c                             */
/**************************************/
int DSLUdecomp(DSMatrix a);
int SolveDSLS(DSVector answer, DSMatrix lu, DSVector b);
int DSLUdecompP(DSMatrix a, long int ch[]);
int SolveDSLSP(DSVector answer, DSMatrix lu, DSVector b, long int ch[]);
int DSLUdecompC(DSMatrix a, long int row_ch[], long int col_ch[]);
int SolveDSLSC(DSVector answer, DSMatrix lu, DSVector b, long int row_ch[], long int col_ch[]);
int DSLUdecompPM(DSMatrix a, long int ch[]);
int SolveDSLSPM(DSVector answer, DSMatrix lu, DSVector b, long int ch[]);

/**************************************/
/* dsbmatrix.c                        */
/**************************************/

// ds

typedef struct{
	float *element; // element[2]
	float zero[DSSIZE]; // = {0.0, 0.0}; // = 0
	long int dim, upper_dim, lower_dim;
} dsbmatrix;

typedef dsbmatrix *DSBMatrix;

/* initialize band matrix */
DSBMatrix init_dsbmatrix(long int dim, long int upper_dim, long int lower_dim);

/* free mpfbmatrix */
void free_dsbmatrix(DSBMatrix mat);

/* return mat[i][j] */
float *get_dsbmatrix_ij(DSBMatrix mat, long int row_index, long int col_index);

/* set mat[i][j] = val */
void set_dsbmatrix_ij(DSBMatrix mat, long int row_index, long int col_index, float val[DSSIZE]);

/* set mat[i][j] = val */
void set_dsbmatrix_ij_f(DSBMatrix mat, long int row_index, long int col_index, float val);

/* set mat[i][j] = val */
void set_dsbmatrix_ij_ui(DSBMatrix mat, long int row_index, long int col_index, unsigned long val);

/* Multiply DSBMatrix * MPFVector */
int mul_dsbmatrix_dsvec(DSVector ret, DSBMatrix mat, DSVector vec);

/* Multiply DSBMatrix^T * MPFVector */
int mul_dsbmatrixt_dsvec(DSVector ret, DSBMatrix mat, DSVector vec);

int dsBLUdecomp(DSBMatrix a);

int SolveDSBLS(DSVector answer, DSBMatrix lu, DSVector b);

/* print band matrix */
void print_dsbmatrix(DSBMatrix mat);

/* set zero matrix */
void set0_dsbmatrix(DSBMatrix mat);

/* set identity matrix */
void setI_dsbmatrix(DSBMatrix mat);

/* Frobenius norm */
//void normf_dsbmatrix(float ret[DSSIZE], DSBMatrix mat);

/* ret := mat_a + mat_b */
//void add_dsbmatrix(DSBMatrix ret, DSBMatrix a, DSBMatrix b);

/* ret := mat_a - mat_b */
//void sub_dsbmatrix(DSBMatrix ret, DSBMatrix a, DSBMatrix b);

/* ret := mat_a - mat_b */
//void sub_dsmatrix_dsbmat_dsmat(DSMatrix ret, DSBMatrix a, DSMatrix b);

/* ret := val * a */
//void cmul_dsbmatrix(DSBMatrix ret, float val[DSSIZE], DSBMatrix a);

/* ret := a * b */
//void mul_dsmatrix_dsbmat_dsbmat(DSMatrix ret, DSBMatrix a, DSBMatrix b);

/* ret := a * b */
//void mul_dsmatrix_dsmat_dsbmat(DSMatrix ret, DSMatrix a, DSBMatrix b);

/* ret := a * b */
//void mul_dsmatrix_dsbmat_dsmat(DSMatrix ret, DSBMatrix a, DSMatrix b);

/* ret := a */
//void subst_dsbmatrix(DSBMatrix ret, DSBMatrix a);

/* ret := (DBMatrix)a */
//void subst_dbmatrix_dsbmat(DBMatrix ret, DSBMatrix a);

/* ret := (DSBMatrix)a */
//void subst_dsbmatrix_dbmat(DSBMatrix ret, DBMatrix a);

/* ret := (DSMatrix)a */
//void subst_dsmatrix_dsbmat(DSMatrix ret, DSBMatrix a);

/**************************************/
/* fread_write_dd.c                   */
/**************************************/
void fread_dsmatrix(FILE *fp, DSMatrix mat);

void fread_dsmatrix(FILE *fp, DSMatrix mat);

void fread_dsmatrix_fname(const char *fname, DSMatrix mat);

void fwrite_dsmatrix(FILE *fp, DSMatrix mat);

void fwrite_dsmatrix_fname(const char *fname, DSMatrix mat);

//void fread_ddpolycoef(FILE *fp, dsPoly p, long int maxdeg);

//void fread_ddpolycoef_fname(const char *fname, dsPoly p, long int maxdeg);

void fread_dsvector(FILE *fp, DSVector vec);

void fread_dsvector_fname(const char *fname, DSVector vec);

void fwrite_dsvector(FILE *fp, DSVector vec);

void fwrite_dsvector_fname(const char *fname, DSVector vec);


/**************************************/
/* gtestmat_dd.c                      */
/**************************************/
/* 1. Hilbert Matrix */
void hilbert_dsmatrix(DSMatrix a, long int dim);

/* 2. Lotkin Matrix */
void lotkin_dsmatrix(DSMatrix a, long int dim);

/* 3. Frank Matrix */
void frank_dsmatrix(DSMatrix a, long int dim);

/* 4. Tridiagonal Matrix */
void tridiag_dsmatrix(DSMatrix a, DSVector low_subdiag, DSVector diag, DSVector up_subdiag, long int dim);

/* 5. Integer Symmetrix Random Matrix */
void int_sym_rand_dsmatrix(DSMatrix mat, long int max, long int seed, long int dim);

/* 6. Integer Unsymmetrix Random Matrix */
void int_unsym_rand_dsmatrix(DSMatrix mat, long int max, long int seed, long int dim);

/* 7. Real Diagonal Matrix */
void diag_dsmatrix(DSMatrix mat, DSVector diag, long int dim);

/* 8. Toeplitz Matrix */
void toeplitz_dsmatrix(DSMatrix mat, float gamma_param[DSSIZE], long int dim);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __BNC_DDLINEAR_H__
