/********************************************************************************/
/* float128linear.h: __float128 precision Linear Computation Library            */
/* Copyright (C) 2024 Tomonori Kouya                                            */
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
// define __BNC_FLOAT128LINEAR_H__
#ifndef __BNC_FLOAT128LINEAR_H__
  #define __BNC_FLOAT128LINEAR_H__

// float128
#include "quadmath.h" // __float128, __complex128
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
#include "bncavx.h"
//#endif // defined(__AVX2__) || defined(__AVX512F__)

// FLOAT128 vector
typedef struct
{
	long int dim; // dim <= real_dim
	long int real_dim; // multiplier of _BNC_D_WIDTH
    //double *element[DDSIZE];
    __float128 *element;
} float128vector;

typedef float128vector *FLOAT128Vector;

//	GET_FLOAT128VECTOR_I(vec, index))
#define GET_FLOAT128VECTOR_I(vec, index) (&(vec->element[i]))
#define get_float128vector_i(vec, index) (&(vec->element[i]))

// set_float128vector_i
static inline void set_float128vector_i(FLOAT128Vector vec, long int index, __float128 val)
{
    vec[index] = val;
}
#define SET_FLOAT128VECTOR_I(vec, index, val) set_float128vector_i((vec), (index), (val))

// set_float128vector_i_d
static inline void set_float128vector_i_d(FLOAT128Vector vec, long int index, double val) // val
{
	__float128 ret;

	vec->element[0][index] = val;
	vec->element[1][index] = 0.0;
}
#define SET_FLOAT128VECTOR_I_D(vec, index, val) set_float128vector_i_d((vec), (index), (val))

// set0_float128vector_i
static inline void set0_float128vector_i(FLOAT128Vector vec, long int index)
{
	vec->element[index] = (__float128)0.0;
}
#define SET0_FLOAT128VECTOR_I(vec, index) set0_float128vector_i((vec), (index))

// generate a text vector: vec(i) := sqrt(sqrt_seed) * (i + 1);
void set_test_float128vector(ddfloat vec[], int sqrt_seed, int dim);

// initialize FLOAT128Vector
FLOAT128Vector init_float128vector(int dimension);

// free FLOAT128Vector
void free_float128vector(FLOAT128Vector vec);

// print float128vector
void print_float128vector(FLOAT128Vector vec);

// set a zero vector
void set0_float128vector(FLOAT128Vector vec);

// set_float128vector_i_str
void set_float128vector_i_str(FLOAT128Vector vec, long int index, const char *str);

/*************************************************/
/* Vector Calculations for FLOAT128Vector               */
/*
void add_float128vector(FLOAT128Vector c, FLOAT128Vector a, FLOAT128Vector b)
void add2_float128vector(FLOAT128Vector c, FLOAT128Vector a)
void sub_float128vector(FLOAT128Vector c, FLOAT128Vector a, FLOAT128Vector b)
void sub2_float128vector(FLOAT128Vector c, DVector a)
void cmul_float128vector(FLOAT128Vector c, __float128 val, FLOAT128Vector a)
void cmul2_float128vector(FLOAT128Vector c, __float128 val)
void add_cmul_float128vector(FLOAT128Vector c, FLOAT128Vector a, __float128 val, FLOAT128Vector b)
double ip_float128vector(FLOAT128Vector a, FLOAT128Vector b)
double norm1_float128vector(FLOAT128Vector a)
double norm2_float128vector(FLOAT128Vector a)
double normi_float128vector(FLOAT128Vector a)
void subst_float128vector(FLOAT128Vector c, FLOAT128Vector a)
*/
/*************************************************/
/* c = a + b */
void add_float128vector(FLOAT128Vector c, FLOAT128Vector a, FLOAT128Vector b);

/* c += a */
void add2_float128vector(FLOAT128Vector c, FLOAT128Vector a);

/* c = a - b */
void sub_float128vector(FLOAT128Vector c, FLOAT128Vector a, FLOAT128Vector b);

/* c -= a */
void sub2_float128vector(FLOAT128Vector c, FLOAT128Vector a);

/* c = val * a */
void cmul_float128vector(FLOAT128Vector c, __float128 val, FLOAT128Vector a);

/* c *= val */
void cmul2_float128vector(FLOAT128Vector c, __float128 val);

/* c = a + val * b */
void add_cmul_float128vector(FLOAT128Vector c, FLOAT128Vector a, __float128 val, FLOAT128Vector b);

/* (a, b) */
void ip_float128vector(__float128 ret, FLOAT128Vector a, FLOAT128Vector b);

/* c := a */
void subst_float128vector(FLOAT128Vector c, FLOAT128Vector a);

/* c := -a */
void neg_float128vector(FLOAT128Vector c, FLOAT128Vector a);

/* ||a||_1 */
void norm1_float128vector(__float128 ret, FLOAT128Vector a);

/* ||a||_infty */
void normi_float128vector(__float128 ret, FLOAT128Vector a);

// Euclid norm
void norm2_float128vector(__float128 ret, FLOAT128Vector vec);

// FLOAT128 matrix
typedef struct{
	long int row_dim, col_dim;
	long int real_row_dim, real_col_dim; // multiplier of _BNC_D_WIDTH
	__float128 *element;
} float128matrix;

typedef float128matrix *FLOAT128Matrix;

#define GET_FLOAT128MATRIX_IJ(mat, i, j) (&(mat->element[mat->real_col_dim * i + j]))
#define get_float128matrix_ij(mat, i, j) (&(mat->element[mat->real_col_dim * i + j]))

// set_float128matrix_ij
static inline void set_float128matrix_ij(FLOAT128Matrix mat, long int i, long int j, double *val) // val[DDSIZE]
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[ij_index] = val;

	return;
} 
#define SET_FLOAT128MATRIX_IJ(mat, i, j, val) set_float128matrix_ij((mat), (i), (j), (val))

// set_float128matrix_ij_d
static inline void set_float128matrix_ij_d(FLOAT128Matrix mat, long int i, long int j, double val)
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[ij_index] = (__float128)val;

	return;
} 
#define SET_DDMATRIX_IJ_D(mat, i, j, val) set_float128matrix_ij_d((mat), (i), (j), (val))
#define SET_DDMATRIX_IJ_UI(mat, i, j, val) set_float128matrix_ij_d((mat), (i), (j), (double)(val))
#define set_float128matrix_ij_ui(mat, i, j, val) set_float128matrix_ij_d((mat), (i), (j), (double)(val))

// set0_float128matrix_ij
static inline void set0_float128matrix_ij(FLOAT128Matrix mat, long int i, long int j)
{
	long int ij_index;

	ij_index = mat->real_col_dim * i + j;

	mat->element[0][ij_index] = 0.0;
	mat->element[1][ij_index] = 0.0;

	return;
}
#define SET0_DDMATRIX_IJ(mat, i, j) set0_float128matrix_ij((mat), (i), (j))

// set a zero matrix
//void set0_float128matrix(FLOAT128Matrix mat);
void set0_float128matrix(FLOAT128Matrix mat);

// initialize float128vector
FLOAT128Matrix init_float128matrix(long int row_dim, long int col_dim);

// free float128vector
void free_float128matrix(FLOAT128Matrix mat);

// print float128vector
void print_float128matrix(FLOAT128Matrix mat);

// FLOAT128Matrix mat -> ddfloat array
void set_ddfloat_ddmat(ddfloat ret[], int ret_dim, FLOAT128Matrix mat);

// ddfloat array -> DDmatrix ret
void set_float128matrix_ddfloat(FLOAT128Matrix ret, ddfloat array[], int array_dim);

// matrix multiplication
// ret := A * B
void mul_float128matrix(FLOAT128Matrix ret, FLOAT128Matrix a, FLOAT128Matrix b);

// Frobenius norm
void normf_float128matrix(__float128 ret, FLOAT128Matrix mat);

// print normf
void print_normf_float128matrix(const char *str, FLOAT128Matrix mat);

/*************************************************/
/* Matrix Caluculations for FLOAT128Matrix            */
/*
void normf_float128matrix(__float128 ret, FLOAT128Matrix mat)
void norm1_float128matrix(__float128 ret, FLOAT128Matrix mat)
void normi_float128matrix(__float128 ret, FLOAT128Matrix mat)
void add_float128matrix(FLOAT128Matrix c, FLOAT128Matrix a, FLOAT128Matrix b);
void sub_float128matrix(FLOAT128Matrix c, FLOAT128Matrix a, FLOAT128Matrix b);
void mul_float128matrix(FLOAT128Matrix c, FLOAT128Matrix a, FLOAT128Matrix b);
void mul_float128matrix_ddvec(FLOAT128Vector v, FLOAT128Matrix a, FLOAT128Vector vb)
void mul_float128matrixt_ddvec(FLOAT128Vector v, FLOAT128Matrix a, FLOAT128Vector vb)
void transpose_float128matrix(FLOAT128Matrix c, FLOAT128Matrix a);
void inv_float128matrix(FLOAT128Matrix a);
void subst_mpfmatrux(FLOAT128Matrix c, FLOAT128Matrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
void normi_float128matrix(__float128 ret, FLOAT128Matrix mat);

/* 1 Norm of Matrix */
void norm1_float128matrix(__float128 ret, FLOAT128Matrix mat);

/* c := a + b */
void add_float128matrix(FLOAT128Matrix c, FLOAT128Matrix a, FLOAT128Matrix b);

/* c := a - b */
void sub_float128matrix(FLOAT128Matrix c, FLOAT128Matrix a, FLOAT128Matrix b);

/* c := sc * a */
void cmul_float128matrix(FLOAT128Matrix c, double sc[DDSIZE], FLOAT128Matrix a);

/* c = a^T */
void transpose_float128matrix(FLOAT128Matrix c, FLOAT128Matrix a);

/* c := a */
void subst_float128matrix(FLOAT128Matrix c, FLOAT128Matrix a);

/* c := I */
void setI_float128matrix(FLOAT128Matrix c);

/* v := a * vb */
void mul_float128matrix_float128vec(FLOAT128Vector v, FLOAT128Matrix a, FLOAT128Vector vb);

/* v := a^T * vb */
void mul_float128matrixt_float128vec(FLOAT128Vector v, FLOAT128Matrix a, FLOAT128Vector vb);

/* a = a^(-1) */
/* square matrix only */
void inv_float128matrix(FLOAT128Matrix a);

// MPFR/GMP related functions
#ifdef USE_GMP
/* c := (mpf)a */
void subst_mpfvector_float128vec(MPFVector c, FLOAT128Vector a);

/* c := (dd)a */
void subst_float128vector_mpfvec(FLOAT128Vector c, MPFVector a);

/* c := (mpf)a */
void subst_mpfmatrix_float128mat(MPFMatrix c, FLOAT128Matrix a);

/* c := (dd)a */
void subst_float128matrix_mpfmat(FLOAT128Matrix c, MPFMatrix a);

/* Normwise relative error of vector */
void relerr_float128vector_mpfvec(double relerr[DDSIZE], FLOAT128Vector approx_vec, MPFVector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_float128vector_mpf(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double norm_relerr[DDSIZE], FLOAT128Vector approx_vec, MPFVector true_vec, int norm_type);
#endif // USE_GMP

/* c := (dd)a */
void subst_float128vector_dvec(FLOAT128Vector c, DVector a);

/* c := (d)a */
void subst_dvector_ddvec(DVector c, FLOAT128Vector a);

/* c := (dd)a */
void subst_float128matrix_dmat(FLOAT128Matrix c, DMatrix a);

/* c := (d)a */
void subst_dmatrix_ddmat(DMatrix c, FLOAT128Matrix a);

/* Normwise relative error of vector */
void relerr_float128vector(double relerr[DDSIZE], FLOAT128Vector approx_vec, FLOAT128Vector true_vec, int norm_type);

/* Elementwise relative errors of vector */
void relerr_element_float128vector(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double norm_relerr[DDSIZE], FLOAT128Vector approx_vec, FLOAT128Vector true_vec, int norm_type);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_float128matrix(FLOAT128Matrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

/**************************************/
/* ddlu.c                             */
/**************************************/
int DDLUdecomp(FLOAT128Matrix a);
int SolveDDLS(FLOAT128Vector answer, FLOAT128Matrix lu, FLOAT128Vector b);
int DDLUdecompP(FLOAT128Matrix a, long int ch[]);
int SolveDDLSP(FLOAT128Vector answer, FLOAT128Matrix lu, FLOAT128Vector b, long int ch[]);
int DDLUdecompC(FLOAT128Matrix a, long int row_ch[], long int col_ch[]);
int SolveDDLSC(FLOAT128Vector answer, FLOAT128Matrix lu, FLOAT128Vector b, long int row_ch[], long int col_ch[]);
int DDLUdecompPM(FLOAT128Matrix a, long int ch[]);
int SolveDDLSPM(FLOAT128Vector answer, FLOAT128Matrix lu, FLOAT128Vector b, long int ch[]);

//--------------------------------------/
// ddlu_strassen.c
//--------------------------------------/
int DDLUdecomp_square(FLOAT128Matrix a, long int start_index, long int min_dim);
int DDLUdecomp_l21(FLOAT128Matrix l21, FLOAT128Matrix a, long int start_index, long int min_dim);
int DDLUdecomp_u12(FLOAT128Matrix u12, FLOAT128Matrix a, long int start_index, long int min_dim);
int DDLUdecomp_a22(FLOAT128Matrix a, FLOAT128Matrix d22, FLOAT128Matrix l21, FLOAT128Matrix u12, long int start_index, long int min_dim);
int DDLUdecomp_strassen(FLOAT128Matrix a, long int min_dim);
int DDLUdecomp_strassenPM(FLOAT128Matrix a, long int ch[], long int min_dim);

#ifdef _OPENMP
int DDLUdecomp_square_omp(FLOAT128Matrix a, long int start_index, long int min_dim);
int DDLUdecomp_l21_omp(FLOAT128Matrix l21, FLOAT128Matrix a, long int start_index, long int min_dim);
int DDLUdecomp_u12_omp(FLOAT128Matrix u12, FLOAT128Matrix a, long int start_index, long int min_dim);
int DDLUdecomp_a22_omp(FLOAT128Matrix a, FLOAT128Matrix d22, FLOAT128Matrix l21, FLOAT128Matrix u12, long int start_index, long int min_dim);
int DDLUdecomp_omp(FLOAT128Matrix a);
int DDLUdecompPM_omp(FLOAT128Matrix a, long int ch[]);
int DDLUdecomp_strassen_omp(FLOAT128Matrix a, long int min_dim);
int DDLUdecomp_strassenPM_omp(FLOAT128Matrix a, long int ch[], long int min_dim);
#endif // _OPENMP

//--------------------------------------/
// ddlu_oz.c
//--------------------------------------/
int DDLUdecomp_a22_oz(FLOAT128Matrix a, FLOAT128Matrix d22, FLOAT128Matrix l21, FLOAT128Matrix u12, long int start_index, long int min_dim, int max_num_div);
int DDLUdecomp_oz(FLOAT128Matrix a, long int min_dim, int max_num_div);
int DDLUdecomp_ozPM(FLOAT128Matrix a, long int ch[], long int min_dim, int max_num_div);

#ifdef _OPENMP
int DDLUdecomp_a22_oz_omp(FLOAT128Matrix a, FLOAT128Matrix d22, FLOAT128Matrix l21, FLOAT128Matrix u12, long int start_index, long int min_dim, int max_num_div);
int DDLUdecomp_oz_omp(FLOAT128Matrix a, long int min_dim, int max_num_div);
int DDLUdecomp_ozPM_omp(FLOAT128Matrix a, long int ch[], long int min_dim, int max_num_div);
#endif // _OPENMP

/**************************************/
/* fread_write.c                      */
/**************************************/
/* undefined
void fread_float128matrix(FILE *fp, FLOAT128Matrix mat);
void fread_float128matrix(FILE *fp, FLOAT128Matrix mat);
void fread_float128matrix_fname(const char *fname, FLOAT128Matrix mat);
void fwrite_float128matrix(FILE *fp, FLOAT128Matrix mat);
void fwrite_float128matrix_fname(const char *fname, FLOAT128Matrix mat);
//void fread_ddpolycoef(FILE *fp, DDPoly p, long int maxdeg);
//void fread_ddpolycoef_fname(const char *fname, DDPoly p, long int maxdeg);
void fread_float128vector(FILE *fp, FLOAT128Vector vec);
void fread_float128vector_fname(const char *fname, FLOAT128Vector vec);
void fwrite_float128vector(FILE *fp, FLOAT128Vector vec);
void fwrite_float128vector_fname(const char *fname, FLOAT128Vector vec);
*/

// defined
void read_test_linear_eq_dd(FLOAT128Matrix A, FLOAT128Vector true_x, FLOAT128Vector b, long int dim, const char *fname_A, const char *fname_true_x, const char *fname_b);

/**************************************/
/* gtestmat_dd.c                      */
/**************************************/
/* 1. Hilbert Matrix */
void hilbert_float128matrix(FLOAT128Matrix a, long int dim);

/* 2. Lotkin Matrix */
void lotkin_float128matrix(FLOAT128Matrix a, long int dim);

/* 3. Frank Matrix */
void frank_float128matrix(FLOAT128Matrix a, long int dim);

/* 4. Tridiagonal Matrix */
void tridiag_float128matrix(FLOAT128Matrix a, FLOAT128Vector low_subdiag, FLOAT128Vector diag, FLOAT128Vector up_subdiag, long int dim);

/* 5. Integer Symmetrix Random Matrix */
void int_sym_rand_float128matrix(FLOAT128Matrix mat, long int max, long int seed, long int dim);

/* 6. Integer Unsymmetrix Random Matrix */
void int_unsym_rand_float128matrix(FLOAT128Matrix mat, long int max, long int seed, long int dim);

/* 7. Real Diagonal Matrix */
void diag_float128matrix(FLOAT128Matrix mat, FLOAT128Vector diag, long int dim);

/* 8. Toeplitz Matrix */
void toeplitz_float128matrix(FLOAT128Matrix mat, double gamma_param[DDSIZE], long int dim);

// 9. Pascal Matrix
void pascal_float128matrix(FLOAT128Matrix ret, long int dim);

// 10. I - randmatrix
void im_rand_float128matrix(FLOAT128Matrix ret, unsigned long seed);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // __BNC_FLOAT128LINEAR_H__
