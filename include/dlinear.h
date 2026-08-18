//******************************************************************************
// dlinear.h : Double Precision Basic Linear Algebra 
// Copyright (C) 2020 Tomonori Kouya
// 
// This program is free software: you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by the
// Free Software Foundation, either version 3 of the License or any later
// version.
// 
// This program is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License
// for more details.
// 
// You should have received a copy of the GNU Lesser General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
// 
//******************************************************************************
#ifndef __BNC_DLINEAR_H__
#define __BNC_DLINEAR_H__

#if defined (__cplusplus)
extern "C" {
#endif // defined (__cplusplus)

#include <stdio.h>
#include <math.h>

// Common defs
#include "bnc_common.h"

// DVector 
typedef struct{
	long int dim; // dim <= real_dim
	long int real_dim; // multiplier of _BNC_D_WIDTH
	double *element;
} dvector;

typedef dvector *DVector;

// DMatrix
typedef struct{
	long int row_dim, col_dim;
	long int real_row_dim, real_col_dim; // multiplier of _BNC_D_WIDTH
	double *element;
} dmatrix;

typedef dmatrix *DMatrix;

#if defined(__AVX2__) || defined(__AVX512F__)
//#include "bncavx.h"
#include "avx2/bncavx.h"
//#endif // defined(__AVX2__) || defined(__AVX512F__)

// Arm SVE2 (Armv9-A, Grace / Neoverse V2) -- pulls in <arm_sve.h> + _bncsve2_*.
// SVE2 builds also keep NEON for any routine without a hand-written SVE2 path.
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2)
#include "sve2/bncsve2.h"
#include "neon/bncneon.h"

// Neon
//#ifdef __ARM_NEON
#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // __ARM_NEON
#include "neon/bncneon.h"
#endif // __ARM_NEON

#define get_dvector_i(vec, index) (*((vec)->element + (index)))
#define set_dvector_i(vec, index, val) (*((vec)->element + (index)) = (val))

#define gdvi get_dvector_i // used in old BNClibrary
#define sdvi set_dvector_i // used in old BNClibrary

// old
//#define get_dmatrix_ij(mat, i, j) ( *((mat)->element + (i) * (mat)->col_dim + (j)) )
//#define set_dmatrix_ij(mat, i, j, val) ( *((mat)->element + (i) * (mat)->col_dim + (j)) = (val) )

// new
#define get_dmatrix_ij(mat, i, j) ( *((mat)->element + (i) * (mat)->real_col_dim + (j)) )
#define set_dmatrix_ij(mat, i, j, val) ( *((mat)->element + (i) * (mat)->real_col_dim + (j)) = (val) )

#define gdmij get_dmatrix_ij // used in old BNClibrary
#define sdmij set_dmatrix_ij // used in old BNClibrary


// drel_diff
static double drel_diff(double a, double b)
{
    double rel_diff;

    rel_diff = fabs(a - b);
    if(a != 0.0)
        rel_diff /= fabs(a);

    return rel_diff;
}
double drel_diff_array(double approx_a[], double approx_b[], int dim, int print_flag);

/*************************************************/
/* Vector Calculations for DVector               */
/*
DVector init_dvector(long int dimension)
void free_dvector(DVector vec)
void add_dvector(DVector c, DVector a, DVector b)
void add2_dvector(DVector c, DVector a)
void sub_dvector(DVector c, DVector a, DVector b)
void sub2_dvector(DVector c, DVector a)
void cmul_dvector(DVector c, double val, DVector a)
void cmul2_dvector(DVector c, double val)
void add_cmul_dvector(DVector c, DVector a, double val, DVector b)
double ip_dvector(DVector a, DVector b)
double norm1_dvector(DVector a)
double norm2_dvector(DVector a)
double normi_dvector(DVector a)
void subst_dvector(DVector c, DVector a)
*/
/*************************************************/

DVector init_dvector(long int dimension);

void free_dvector(DVector vec);

// print_dvector
void print_dvector(DVector vec);

/* c = a + b */
void add_dvector(DVector c, DVector a, DVector b);

/* c += a */
void add2_dvector(DVector c, DVector a);

/* c = a - b */
void sub_dvector(DVector c, DVector a, DVector b);

/* c -= a */
void sub2_dvector(DVector c, DVector a);

/* c = val * a */
void cmul_dvector(DVector c, double val, DVector a);

/* c *= val */
void cmul2_dvector(DVector c, double val);

/* c = a + val * b */
void add_cmul_dvector(DVector c, DVector a, double val, DVector b);

// 2025-02-19(Wed) T.Kouya
/* c = a - val * b */
void sub_cmul_dvector(DVector c, DVector a, double val, DVector b);

/* (a, b) */
double ip_dvector(DVector a, DVector b);

/* ||a||_1 */
double norm1_dvector(DVector a);

/* ||a||_2 */
double norm2_dvector(DVector a);

/* ||a||_infty */
double normi_dvector(DVector a);

/* c := a */
void subst_dvector(DVector c, DVector a);

/* c := 0 */
void set0_dvector(DVector c);

/* append 2005.07/12 */
/*
	ret(index_start) = src(src_index_start)
	 ...
	ret(index_end  ) = src(src_index_end)
*/
void copy_dvector_ij(DVector ret, long int index_start, long int index_end, DVector src, long int src_index_start, long int src_index_end);


// 2022-11-17(Thu) T.Kouya
// absmax_dvector
double absmax_dvector(long int *max_index, DVector vec);

/*************************************************/
/* Matrix Caluculations for DMatrix              */
/*
DMatrix init_dmatrix(long int row_dimension, long int col_dimension)
void free_dmatrix(DMatrix mat)
double normf_dmatrix(DMatrix mat)
double normi_dmatrix(DMatrix mat)
double norm1_dmatrix(DMatrix mat)
void add_dmatrix(DMatrix c, DMatrix a, DMatrix b);
void sub_dmatrix(DMatrix c, DMatrix a, DMatrix b);
void mul_dmatrix(DMatrix c, DMatrix a, DMatrix b);
void transpose_dmatrix(DMatrix c, DMatrix a);
void mul_dmatrix_dvec(DVector v, DMatrix a, DVector vb)
void mul_dmatrixt_dvec(DVector v, DMatrix a, DVector vb)
void inv_dmatrix(DMatrix a);
void subst_dmatrux(DMatrix c, DMatrix a);
*/
/*************************************************/
DMatrix init_dmatrix(long int row_dimension, long int col_dimension);

void free_dmatrix(DMatrix mat);

// print_dmatrix
void print_dmatrix(DMatrix mat);

/*************************************************/
/* Matrix Caluculations for DMatrix              */
/*
double normf_dmatrix(DMatrix mat)
double normi_dmatrix(DMatrix mat)
double norm1_dmatrix(DMatrix mat)
void add_dmatrix(DMatrix c, DMatrix a, DMatrix b);
void sub_dmatrix(DMatrix c, DMatrix a, DMatrix b);
void mul_dmatrix(DMatrix c, DMatrix a, DMatrix b);
void transpose_dmatrix(DMatrix c, DMatrix a);
void mul_dmatrix_dvec(DVector v, DMatrix a, DVector vb)
void mul_dmatrixt_dvec(DVector v, DMatrix a, DVector vb)
void inv_dmatrix(DMatrix a);
void subst_dmatrux(DMatrix c, DMatrix a);
*/
/*************************************************/
/* Frobenius Norm of Matrix */
double normf_dmatrix(DMatrix mat);

/* Frobenius Norm of Matrix: array type */
double normf_dmatrix_array(double mat[], int row_dim, int col_dim);

/* Infinity Norm of Matrix */
double normi_dmatrix(DMatrix mat);

/* 1 Norm of Matrix */
double norm1_dmatrix(DMatrix mat);

/* c = a + b */
void add_dmatrix(DMatrix c, DMatrix a, DMatrix b);

/* c = a - b */
void sub_dmatrix(DMatrix c, DMatrix a, DMatrix b);

/* c = sc * a */
void cmul_dmatrix(DMatrix c, double sc, DMatrix a);

/* c = a * b */
void mul_dmatrix(DMatrix c, DMatrix a, DMatrix b);

/* c = a^T */
void transpose_dmatrix(DMatrix c, DMatrix a);

/* c := a */
void subst_dmatrix(DMatrix c, DMatrix a);

/* c := 0 */
void set0_dmatrix(DMatrix c);

/* c := I */
void setI_dmatrix(DMatrix c);

/* v = a * vb */
void mul_dmatrix_dvec(DVector v, DMatrix a, DVector vb);

/* v = a^T * vb */
//void mul_dmatrixt_dvec(DVector v, DMatrix a, DVector vb);
void mul_dmatrixt_dvec_old(DVector v, DMatrix a, DVector vb);

/* v = a^T * vb */
// Dangerous!
//void mul_dmatrixt_dvec_simd(DVector v, DMatrix a, DVector vb);
void mul_dmatrixt_dvec(DVector v, DMatrix a, DVector vb);

/* a = a^(-1) */
/* square matrix only */
void inv_dmatrix(DMatrix a);

/* Double */

/* 1. Hilbert Matrix */
//void hilbert_dmatrix(DMatrix a, long int dim);

/* 2. Lotkin Matrix */
//void lotkin_dmatrix(DMatrix a, long int dim);

/* 3. Frank Matrix */
//void frank_dmatrix(DMatrix a, long int dim);

/* 4. Tridiagonal Matrix */
//void tridiag_dmatrix(DMatrix a, DVector low_subdiag, DVector diag, DVector up_subdiag, long int dim);

/* 5. Integer Symmetrix Random Matrix */
//void int_sym_rand_dmatrix(DMatrix mat, long int max, long int seed, long int dim);

/* 6. Integer Unsymmetrix Random Matrix */
//void int_unsym_rand_dmatrix(DMatrix mat, long int max, long int seed, long int dim);

/* 7. Real Diagonal Matrix */
//void diag_dmatrix(DMatrix mat, DVector diag, long int dim);

/* 8. Toeplitz Matrix */
//void toeplitz_dmatrix(DMatrix mat, double gamma_param, long int dim);

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 1997.10.24 (Fri) Tomonori Kouya */
/*                 ver. 0.1 2000.02.24 (Thu) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int DLUdecomp(DMatrix a);
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/


/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 1997.10.24 (Fri) Tomonori Kouya */
/*                 ver. 0.1 2000.02.24 (Thu) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveDLS(DVector answer, DMatrix lu, DVector b);
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DMatrix lu: LU decomposed Matrix (given by user)   */
/*       DVector b: constant vector (given by user)         */
/*       DVector answer: Solution for linear system         */
/*       long int dim: Dimension of Matrix (given by user)  */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                 (Double Precision)       */
/*                                 (Partial Pivoting)       */
/*                                                          */
/*                 ver. 0.0 1997.10.24 (Fri) Tomonori Kouya */
/*                 ver. 0.1 2000.02.24 (Thu) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int DLUdecompP(DMatrix a, long int ch[]);
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DMatrix a: Matrix (given by user)                  */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/


/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                                 (Double Precision)       */
/*                                 (Partial Pivoting)       */
/*                                                          */
/*                 ver. 0.0 1997.10.24 (Fri) Tomonori Kouya */
/*                 ver. 0.1 2000.02.24 (Thu) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveDLSP(DVector answer, DMatrix lu, DVector b, long int ch[]);
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DMatrix lu[]: LU decomposed Matrix (given by user) */
/*       DVector b[]: constant vector (given by user)       */
/*       DVector answer[]: Solution for linear system       */
/*       long int ch: Row order (given by user)             */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                 (Double Precision)       */
/*                                 (Complete Pivoting)      */
/*                                                          */
/*                 ver. 0.0 1997.10.24 (Fri) Tomonori Kouya */
/*                 ver. 0.1 2000.02.24 (Thu) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int DLUdecompC(DMatrix a, long int row_ch[], long int col_ch[]);
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DMatrix a[]: Matrix (given by user)                */
/*       long int row_ch[]: Row order                       */
/*       long int col_ch[]: Column order                    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*  row_ch[]: Row order                                     */
/*  col_ch[]: Column order                                  */
/*                                                          */
/************************************************************/

/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                                 (Double Precision)       */
/*                                 (Complete Pivoting)      */
/*                                                          */
/*                 ver. 0.0 1997.10.24 (Fri) Tomonori Kouya */
/*                 ver. 0.1 2000.02.24 (Thu) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveDLSC(DVector answer, DMatrix lu, DVector b, long int row_ch[], long int col_ch[]);
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DMatrix lu: LU decomposed Matrix (given by user)   */
/*       DVector b: constant vector (given by user)         */
/*       DVector answer: Solution for linear system         */
/*       long int row_ch[]: Row order (given by user)       */
/*       long int col_ch[]: Column order (given by user)    */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_dmatrix(DMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

// 2022-11-17(Thu) T.Kouya
// absmax_row_dmatrix
double absmax_row_dmatrix(long int *max_j, long int row_index, DMatrix mat);

// 2022-11-17(Thu) T.Kouya
// absmax_col_dmatrix
double absmax_col_dmatrix(long int *max_i, long int col_index, DMatrix mat);

// Appended in 2024-05-09 T.Kouya
// from lanczos.c

/* c = a - sc * b */
void subcmul_dvector(DVector c, DVector a, double sc, DVector b);

/* mat := (vec[0] vec[1] ... vec[n]) */
void subst_dmatrix_dvec(DMatrix mat, DVector vec[]);

//-------------------------------/
// gtestmat.c
//-------------------------------/

/* 1. Hilbert Matrix */
void hilbert_dmatrix(DMatrix a, long int dim);

/* 2. Lotkin Matrix */
void lotkin_dmatrix(DMatrix a, long int dim);

/* 3. Frank Matrix */
void frank_dmatrix(DMatrix a, long int dim);

/* 4. Tridiagonal Matrix */
void tridiag_dmatrix(DMatrix a, DVector low_subdiag, DVector diag, DVector up_subdiag, long int dim);

/* 5. Integer Symmetrix Random Matrix */
void int_sym_rand_dmatrix(DMatrix mat, long int max, long int seed, long int dim);

/* 6. Integer Unsymmetrix Random Matrix */
void int_unsym_rand_dmatrix(DMatrix mat, long int max, long int seed, long int dim);

/* 7. Real Diagonal Matrix */
void diag_dmatrix(DMatrix mat, DVector diag, long int dim);

/* 8. Toeplitz Matrix */
void toeplitz_dmatrix(DMatrix mat, double gamma_param, long int dim);


/**************************************/
/* fread_write.c                      */
/**************************************/
void fread_dmatrix(FILE *fp, DMatrix mat);
void fread_dmatrix(FILE *fp, DMatrix mat);
void fread_dmatrix_fname(const char *fname, DMatrix mat);
void fwrite_dmatrix(FILE *fp, DMatrix mat);
void fwrite_dmatrix_fname(const char *fname, DMatrix mat);
//void fread_dpolycoef(FILE *fp, DPoly p, long int maxdeg);
//void fread_dpolycoef_fname(const char *fname, DPoly p, long int maxdeg);
void fread_dvector(FILE *fp, DVector vec);
void fread_dvector_fname(const char *fname, DVector vec);
void fwrite_dvector(FILE *fp, DVector vec);
void fwrite_dvector_fname(const char *fname, DVector vec);

#if defined (__cplusplus)
}
#endif // defined (__cplusplus)

#endif // __BNC_DLINEAR_H__