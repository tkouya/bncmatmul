/********************************************************************************/
/* ddlinear.h: Double-double and Quadruple precision Linear Computation Library */
/* Copyright (C) 2015-2020 Tomonori Kouya                                       */
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
// define __BNC_DDLINEAR_H__
#ifndef __BNC_DDLINEAR_H__
  #define __BNC_DDLINEAR_H__

#include "rdd.h"

#include "dlinear.h"
//#include "bmatrix.h"

#ifdef USE_GMP
#include "gmp.h"
#include "mpfr.h"
#include "mpfr_dtq_sd.h"
#include "mpflinear.h"
#endif //USE_GMP//

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

//#if defined(__AVX2__) || defined(__AVX512F__)
//#include "avx2/bncavx.h"
//#endif // defined(__AVX2__) || defined(__AVX512F__)

// DD vector
typedef struct
{
	long int dim; // dim <= real_dim
	long int real_dim; // multiplier of _BNC_D_WIDTH
    double *element[DDSIZE];
} ddvector;

typedef ddvector *DDVector;

// SIMD: AVX2 / AVX-512 / SVE2 / NEON
#if defined(__AVX2__) || defined(__AVX512F__) // __AVX2__ or __AVX512F__
#include "avx2/bncavx.h"
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2)
#include "sve2/bncsve2.h"
#include "neon/bncneon.h"   /* SVE2 build keeps NEON for the #elif fallback paths */
#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // __ARM_NEON
#include "neon/bncneon.h"
#endif // defined(__AVX2__) || defined(__AVX512F__)

// SIMD : for copy & paste
//#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
//#elif defined(__AVX512F__) // __AVX512F__
//#else // others
//#endif // __AVX2__

// get_ddvector_i
static inline ddfloat get_ddvector_i_ddfloat(DDVector vec, long int index)
//static inline double *get_ddvector_i(DDVector vec, long int index)
{
//	double ret[DDSIZE];
	ddfloat ret;

	ret.val[0] = vec->element[0][index];
	ret.val[1] = vec->element[1][index];

	return ret;
}

//	GET_DDVECTOR_I(vec, index))
#define GET_DDVECTOR_I(vec, index) ((get_ddvector_i_ddfloat((vec), (index)).val))
#define get_ddvector_i(vec, index) ((get_ddvector_i_ddfloat((vec), (index)).val))

// set_ddvector_i
static inline void set_ddvector_i(DDVector vec, long int index, double *val) // val[DDSIZE]
{
	double ret[DDSIZE];

	vec->element[0][index] = val[0];
	vec->element[1][index] = val[1];
}
#define SET_DDVECTOR_I(vec, index, val) set_ddvector_i((vec), (index), (val))
// 2025-02-19(Wed) T.Kouya
#define subst_ddvector_i(ret, vec, index) rdd_set((ret), get_ddvector_i((vec), (index)))

// set_ddvector_i_d
static inline void set_ddvector_i_d(DDVector vec, long int index, double val) // val
{
	double ret[DDSIZE];

	vec->element[0][index] = val;
	vec->element[1][index] = 0.0;
}
#define SET_DDVECTOR_I_D(vec, index, val) set_ddvector_i_d((vec), (index), (val))

// set0_ddvector_i
static inline void set0_ddvector_i(DDVector vec, long int index)
{
	double ret[DDSIZE];

	vec->element[0][index] = 0.0;
	vec->element[1][index] = 0.0;
}
#define SET0_DDVECTOR_I(vec, index) set0_ddvector_i((vec), (index))

// ddrel_diff
ddfloat ddrel_diff(ddfloat a, ddfloat b);

ddfloat ddrel_diff_array(ddfloat approx_a[], ddfloat approx_b[], int dim, int print_flag);

// Frobenius norm
ddfloat ddnormf(ddfloat array[], int dim);

// generate a text vector: vec(i) := sqrt(sqrt_seed) * (i + 1);
void set_test_ddvector(ddfloat vec[], int sqrt_seed, int dim);

// initialize DDVector
DDVector init_ddvector(int dimension);

// free DDVector
void free_ddvector(DDVector vec);

// DDVector vec -> ddfloat array
void set_ddfloat_ddvec(ddfloat ret[], int ret_dim, DDVector vec);

// ddfloat array -> DDVector ret
void set_ddvector_ddfloat(DDVector ret, ddfloat array[], int array_dim);

// print ddvector
void print_ddvector(DDVector vec);

// set a zero vector
void set0_ddvector(DDVector vec);

// set_ddvector_i_str
void set_ddvector_i_str(DDVector vec, long int index, const char *str);

/*************************************************/
/* Vector Calculations for DDVector               */
/*
void add_ddvector(DDVector c, DDVector a, DDVector b)
void add2_ddvector(DDVector c, DDVector a)
void sub_ddvector(DDVector c, DDVector a, DDVector b)
void sub2_ddvector(DDVector c, DVector a)
void cmul_ddvector(DDVector c, double val[DDSIZE], DDVector a)
void cmul2_ddvector(DDVector c, double val[DDSIZE])
void add_cmul_ddvector(DDVector c, DDVector a, double val[DDSIZE], DDVector b)
double ip_ddvector(DDVector a, DDVector b)
double norm1_ddvector(DDVector a)
double norm2_ddvector(DDVector a)
double normi_ddvector(DDVector a)
void subst_ddvector(DDVector c, DDVector a)
*/
/*************************************************/
/* c = a + b */
void add_ddvector(DDVector c, DDVector a, DDVector b);

/* c += a */
void add2_ddvector(DDVector c, DDVector a);

/* c = a - b */
void sub_ddvector(DDVector c, DDVector a, DDVector b);

/* c -= a */
void sub2_ddvector(DDVector c, DDVector a);

/* c = val * a */
void cmul_ddvector(DDVector c, double val[DDSIZE], DDVector a);

/* c *= val */
void cmul2_ddvector(DDVector c, double val[DDSIZE]);

/* c = a + val * b */
void add_cmul_ddvector(DDVector c, DDVector a, double val[DDSIZE], DDVector b);

/* c = a - val * b */
void sub_cmul_ddvector(DDVector c, DDVector a, double val[DDSIZE], DDVector b);

/* (a, b) */
void ip_ddvector(double ret[DDSIZE], DDVector a, DDVector b);

/* c := a */
void subst_ddvector(DDVector c, DDVector a);

/* c := -a */
void neg_ddvector(DDVector c, DDVector a);

/* ||a||_1 */
void norm1_ddvector(double ret[DDSIZE], DDVector a);

/* ||a||_infty */
void normi_ddvector(double ret[DDSIZE], DDVector a);

// Euclid norm
void norm2_ddvector(double ret[DDSIZE], DDVector vec);

#if defined(__AVX2__)
/* double-double = double + double */
void _bncavx2_ddadd_d_d(ddfloat ret[], double a[], double b[], int dim);

/* double-double = double + double */
//void _bncavx2_ddvadd_d_d(ddvector *ret, double a[], double b[], int dim)
void _bncavx2_ddvadd_d_d(DDVector ret, double a[], double b[], int dim);

/* add */
void _bncavx2_ddvadd(DDVector ret, DDVector a, DDVector b, int dim);

/* add */
void _bncavx2_ddadd(ddfloat ret[], ddfloat a[], ddfloat b[], int dim);

/* double-double = double * double */
void _bncavx2_ddvmul_d_d(DDVector ret, double a[], double b[], int dim);

/* ddmul */
void _bncavx2_ddmul(ddfloat ret[], ddfloat a[], ddfloat b[], int dim);

/* dddiv */
void _bncavx2_dddiv(ddfloat ret[], ddfloat a[], ddfloat b[], int dim);

/* div */
void _bncavx2_ddvdiv(DDVector ret, DDVector a, DDVector b, int dim);

void _bncavx2_ddvmul(DDVector ret, DDVector a, DDVector b, int dim);
#endif // __AVX2__

// ddmatmul_ddvec
void ddmatmul_ddvec(DDVector ret, DDVector mat_a, DDVector mat_b, int row_dim, int mid_dim, int col_dim);


// ddmatmul_ddvec_ur4
void ddmatmul_ddvec_ur4(DDVector ret, DDVector mat_a, DDVector mat_b, int row_dim, int mid_dim, int col_dim);

// ddmatmul_ddvec_avx2
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
void ddmatmul_ddvec_avx2(DDVector ret, DDVector mat_a, DDVector mat_b, int row_dim, int mid_dim, int col_dim);
#endif // AVX2

// DD matrix
typedef struct{
	long int row_dim, col_dim;
	long int real_row_dim, real_col_dim; // multiplier of _BNC_D_WIDTH
	double *element[DDSIZE];
} ddmatrix;

typedef ddmatrix *DDMatrix;

// get_ddmatrix_ij
static inline ddfloat get_ddmatrix_ij_ddfloat(DDMatrix mat, long int i, long int j)
{
	long int ij_index;
//	double ret[DDSIZE];
	ddfloat ret;

	ij_index = mat->real_col_dim * i + j;

	ret.val[0] = mat->element[0][ij_index];
	ret.val[1] = mat->element[1][ij_index];

	return ret;
} 
#define GET_DDMATRIX_IJ(mat, i, j) ((get_ddmatrix_ij_ddfloat((mat), (i), (j)).val))
#define get_ddmatrix_ij(mat, i, j) ((get_ddmatrix_ij_ddfloat((mat), (i), (j)).val))

// set_ddmatrix_ij
static inline void set_ddmatrix_ij(DDMatrix mat, long int i, long int j, double *val) // val[DDSIZE]
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[0][ij_index] = val[0];
	mat->element[1][ij_index] = val[1];

	return;
} 
#define SET_DDMATRIX_IJ(mat, i, j, val) set_ddmatrix_ij((mat), (i), (j), (val))

// set_ddmatrix_ij_d
static inline void set_ddmatrix_ij_d(DDMatrix mat, long int i, long int j, double val)
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[0][ij_index] = val;
	mat->element[1][ij_index] = 0.0;

	return;
} 
#define SET_DDMATRIX_IJ_D(mat, i, j, val) set_ddmatrix_ij_d((mat), (i), (j), (val))
#define SET_DDMATRIX_IJ_UI(mat, i, j, val) set_ddmatrix_ij_d((mat), (i), (j), (double)(val))
#define set_ddmatrix_ij_ui(mat, i, j, val) set_ddmatrix_ij_d((mat), (i), (j), (double)(val))

// set0_ddmatrix_ij
static inline void set0_ddmatrix_ij(DDMatrix mat, long int i, long int j)
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[0][ij_index] = 0.0;
	mat->element[1][ij_index] = 0.0;

	return;
}
#define SET0_DDMATRIX_IJ(mat, i, j) set0_ddmatrix_ij((mat), (i), (j))

// set a zero matrix
//void set0_ddmatrix(DDMatrix mat);
void set0_ddmatrix(DDMatrix mat);

// initialize ddvector
DDMatrix init_ddmatrix(long int row_dim, long int col_dim);

// free ddvector
void free_ddmatrix(DDMatrix mat);

// print ddvector
void print_ddmatrix(DDMatrix mat);

// DDMatrix mat -> ddfloat array
void set_ddfloat_ddmat(ddfloat ret[], int ret_dim, DDMatrix mat);

// ddfloat array -> DDmatrix ret
void set_ddmatrix_ddfloat(DDMatrix ret, ddfloat array[], int array_dim);

// matrix multiplication
// ret := A * B
void mul_ddmatrix(DDMatrix ret, DDMatrix a, DDMatrix b);

// Frobenius norm
void normf_ddmatrix(double ret[DDSIZE], DDMatrix mat);

// print normf
void print_normf_ddmatrix(const char *str, DDMatrix mat);

/*************************************************/
/* Matrix Caluculations for DDMatrix            */
/*
void normf_ddmatrix(double ret[DDSIZE], DDMatrix mat)
void norm1_ddmatrix(double ret[DDSIZE], DDMatrix mat)
void normi_ddmatrix(double ret[DDSIZE], DDMatrix mat)
void add_ddmatrix(DDMatrix c, DDMatrix a, DDMatrix b);
void sub_ddmatrix(DDMatrix c, DDMatrix a, DDMatrix b);
void mul_ddmatrix(DDMatrix c, DDMatrix a, DDMatrix b);
void mul_ddmatrix_ddvec(DDVector v, DDMatrix a, DDVector vb)
void mul_ddmatrixt_ddvec(DDVector v, DDMatrix a, DDVector vb)
void transpose_ddmatrix(DDMatrix c, DDMatrix a);
void inv_ddmatrix(DDMatrix a);
void subst_mpfmatrux(DDMatrix c, DDMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_ddmatrix(double ret[DDSIZE], DDMatrix mat);

/* 1 Norm of Matrix */
void norm1_ddmatrix(double ret[DDSIZE], DDMatrix mat);

/* c := a + b */
void add_ddmatrix(DDMatrix c, DDMatrix a, DDMatrix b);

/* c := a - b */
void sub_ddmatrix(DDMatrix c, DDMatrix a, DDMatrix b);

/* c := sc * a */
void cmul_ddmatrix(DDMatrix c, double sc[DDSIZE], DDMatrix a);

/* c = a^T */
void transpose_ddmatrix(DDMatrix c, DDMatrix a);

/* c := a */
void subst_ddmatrix(DDMatrix c, DDMatrix a);

/* c := -a */
void neg_ddmatrix(DDMatrix c, DDMatrix a);

/* c := I */
void setI_ddmatrix(DDMatrix c);

/* v := a * vb */
void mul_ddmatrix_ddvec(DDVector v, DDMatrix a, DDVector vb);

/* v := a^T * vb */
void mul_ddmatrixt_ddvec(DDVector v, DDMatrix a, DDVector vb);

/* a = a^(-1) */
/* square matrix only */
void inv_ddmatrix(DDMatrix a);

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpf)a */
void subst_mpfvector_ddvec(MPFVector c, DDVector a);

/* c := (dd)a */
void subst_ddvector_mpfvec(DDVector c, MPFVector a);

/* init_set_ddvector_mpfvec */
DDVector init_set_ddvector_mpfvec(MPFVector c);

/* c := (mpf)a */
void subst_mpfmatrix_ddmat(MPFMatrix c, DDMatrix a);

/* c := (dd)a */
void subst_ddmatrix_mpfmat(DDMatrix c, MPFMatrix a);

/* init_set_ddmatrix_mpfmat */
DDMatrix init_set_ddmatrix_mpfmat(MPFMatrix a);

/* Normwise relative error of vector */
void relerr_ddvector_mpfvec(double relerr[DDSIZE], DDVector approx_vec, MPFVector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_ddvector_mpf(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double norm_relerr[DDSIZE], DDVector approx_vec, MPFVector true_vec, int norm_type);
#endif // USE_GMP

/* c := (dd)a */
void subst_ddvector_dvec(DDVector c, DVector a);

/* c := (d)a */
void subst_dvector_ddvec(DVector c, DDVector a);

/* c := (dd)a */
void subst_ddmatrix_dmat(DDMatrix c, DMatrix a);

/* c := (d)a */
void subst_dmatrix_ddmat(DMatrix c, DDMatrix a);

/* Normwise relative error of vector */
void relerr_ddvector(double relerr[DDSIZE], DDVector approx_vec, DDVector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_ddvector(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double norm_relerr[DDSIZE], DDVector approx_vec, DDVector true_vec, int norm_type);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_ddmatrix(DDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

/**************************************/
/* ddlu.c                             */
/**************************************/
int DDLUdecomp(DDMatrix a);
int SolveDDLS(DDVector answer, DDMatrix lu, DDVector b);
int DDLUdecompP(DDMatrix a, long int ch[]);
int SolveDDLSP(DDVector answer, DDMatrix lu, DDVector b, long int ch[]);
int DDLUdecompC(DDMatrix a, long int row_ch[], long int col_ch[]);
int SolveDDLSC(DDVector answer, DDMatrix lu, DDVector b, long int row_ch[], long int col_ch[]);
int DDLUdecompPM(DDMatrix a, long int ch[]);
int SolveDDLSPM(DDVector answer, DDMatrix lu, DDVector b, long int ch[]);

//--------------------------------------/
// ddlu_strassen.c
//--------------------------------------/
int DDLUdecomp_square(DDMatrix a, long int start_index, long int min_dim);
int DDLUdecomp_l21(DDMatrix l21, DDMatrix a, long int start_index, long int min_dim);
int DDLUdecomp_u12(DDMatrix u12, DDMatrix a, long int start_index, long int min_dim);
int DDLUdecomp_a22(DDMatrix a, DDMatrix d22, DDMatrix l21, DDMatrix u12, long int start_index, long int min_dim);
int DDLUdecomp_strassen(DDMatrix a, long int min_dim);
int DDLUdecomp_strassenPM(DDMatrix a, long int ch[], long int min_dim);

#ifdef _OPENMP
int DDLUdecomp_square_omp(DDMatrix a, long int start_index, long int min_dim);
int DDLUdecomp_l21_omp(DDMatrix l21, DDMatrix a, long int start_index, long int min_dim);
int DDLUdecomp_u12_omp(DDMatrix u12, DDMatrix a, long int start_index, long int min_dim);
int DDLUdecomp_a22_omp(DDMatrix a, DDMatrix d22, DDMatrix l21, DDMatrix u12, long int start_index, long int min_dim);
int DDLUdecomp_omp(DDMatrix a);
int DDLUdecompPM_omp(DDMatrix a, long int ch[]);
int DDLUdecomp_strassen_omp(DDMatrix a, long int min_dim);
int DDLUdecomp_strassenPM_omp(DDMatrix a, long int ch[], long int min_dim);
#endif // _OPENMP

//--------------------------------------/
// ddlu_oz.c
//--------------------------------------/
int DDLUdecomp_a22_oz(DDMatrix a, DDMatrix d22, DDMatrix l21, DDMatrix u12, long int start_index, long int min_dim, int max_num_div);
int DDLUdecomp_oz(DDMatrix a, long int min_dim, int max_num_div);
int DDLUdecomp_ozPM(DDMatrix a, long int ch[], long int min_dim, int max_num_div);

#ifdef _OPENMP
int DDLUdecomp_a22_oz_omp(DDMatrix a, DDMatrix d22, DDMatrix l21, DDMatrix u12, long int start_index, long int min_dim, int max_num_div);
int DDLUdecomp_oz_omp(DDMatrix a, long int min_dim, int max_num_div);
int DDLUdecomp_ozPM_omp(DDMatrix a, long int ch[], long int min_dim, int max_num_div);
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
void read_test_linear_eq_dd(DDMatrix A, DDVector true_x, DDVector b, long int dim, const char *fname_A, const char *fname_true_x, const char *fname_b);

/**************************************/
/* gtestmat_dd.c                      */
/**************************************/
/* 1. Hilbert Matrix */
void hilbert_ddmatrix(DDMatrix a, long int dim);

/* 2. Lotkin Matrix */
void lotkin_ddmatrix(DDMatrix a, long int dim);

/* 3. Frank Matrix */
void frank_ddmatrix(DDMatrix a, long int dim);

/* 4. Tridiagonal Matrix */
void tridiag_ddmatrix(DDMatrix a, DDVector low_subdiag, DDVector diag, DDVector up_subdiag, long int dim);

/* 5. Integer Symmetrix Random Matrix */
void int_sym_rand_ddmatrix(DDMatrix mat, long int max, long int seed, long int dim);

/* 6. Integer Unsymmetrix Random Matrix */
void int_unsym_rand_ddmatrix(DDMatrix mat, long int max, long int seed, long int dim);

/* 7. Real Diagonal Matrix */
void diag_ddmatrix(DDMatrix mat, DDVector diag, long int dim);

/* 8. Toeplitz Matrix */
void toeplitz_ddmatrix(DDMatrix mat, double gamma_param[DDSIZE], long int dim);

// 9. Pascal Matrix
void pascal_ddmatrix(DDMatrix ret, long int dim);

// 10. I - randmatrix
void im_rand_ddmatrix(DDMatrix ret, unsigned long seed);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __BNC_DDLINEAR_H__
