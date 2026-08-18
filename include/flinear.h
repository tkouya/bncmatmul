//******************************************************************************
// flinear.h : Single Precision Basic Linear Algebra 
// Copyright (C) 2021 Tomonori Kouya
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
#ifndef __BNC_FLINEAR_H__
#define __BNC_FLINEAR_H__

#if defined (__cplusplus)
extern "C" {
#endif // defined (__cplusplus)

#include <stdio.h>
#include <stdlib.h> // malloc and free
#include <math.h>

// Common defs
#include "bnc_common.h"

// FVector 
typedef struct{
	long int dim; // dim <= real_dim
	long int real_dim; // multiplier of _BNC_D_WIDTH
	float *element;
} fvector;

typedef fvector *FVector;

// FMatrix
typedef struct{
	long int row_dim, col_dim;
	long int real_row_dim, real_col_dim; // multiplier of _BNC_D_WIDTH
	float *element;
} fmatrix;

typedef fmatrix *FMatrix;


#if defined(__AVX2__) || defined(__AVX512F__)
#include "avx2/bncavx.h"
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2)
#include "sve2/bncsve2.h"
#include "neon/bncneon.h"   /* SVE2 build keeps NEON for the #elif fallback paths */
#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON)
#include "neon/bncneon.h"
#endif // defined(__AVX2__) || defined(__AVX512F__)

#define get_fvector_i(vec, index) (*((vec)->element + (index)))

#define set_fvector_i(vec, index, val) (*((vec)->element + (index)) = (val))

// old
//#define get_fmatrix_ij(mat, i, j) ( *((mat)->element + (i) * (mat)->col_dim + (j)) )
//#define set_fmatrix_ij(mat, i, j, val) ( *((mat)->element + (i) * (mat)->col_dim + (j)) = (val) )

// new
#define get_fmatrix_ij(mat, i, j) ( *((mat)->element + (i) * (mat)->real_col_dim + (j)) )
#define set_fmatrix_ij(mat, i, j, val) ( *((mat)->element + (i) * (mat)->real_col_dim + (j)) = (val) )

// frel_diff
static float frel_diff(float a, float b)
{
    float rel_diff;

    rel_diff = fabsf(a - b);
    if(a != 0.0)
        rel_diff /= fabsf(a);

    return rel_diff;
}
float frel_diff_array(float approx_a[], float approx_b[], int dim, int print_flag);

/*************************************************/
/* Vector Calculations for FVector               */
/*
FVector init_fvector(long int dimension)
void free_fvector(FVector vec)
void add_fvector(FVector c, FVector a, FVector b)
void add2_fvector(FVector c, FVector a)
void sub_fvector(FVector c, FVector a, FVector b)
void sub2_fvector(FVector c, FVector a)
void cmul_fvector(FVector c, float val, FVector a)
void cmul2_fvector(FVector c, float val)
void add_cmul_fvector(FVector c, FVector a, float val, FVector b)
float ip_fvector(FVector a, FVector b)
float norm1_fvector(FVector a)
float norm2_fvector(FVector a)
float normi_fvector(FVector a)
void subst_fvector(FVector c, FVector a)
*/
/*************************************************/

FVector init_fvector(long int dimension);

void free_fvector(FVector vec);

/* c = a + b */
void add_fvector(FVector c, FVector a, FVector b);

/* c += a */
void add2_fvector(FVector c, FVector a);

/* c = a - b */
void sub_fvector(FVector c, FVector a, FVector b);

/* c -= a */
void sub2_fvector(FVector c, FVector a);

/* c = val * a */
void cmul_fvector(FVector c, float val, FVector a);

/* c *= val */
void cmul2_fvector(FVector c, float val);

/* c = a + val * b */
void add_cmul_fvector(FVector c, FVector a, float val, FVector b);
void sub_cmul_fvector(FVector c, FVector a, float val, FVector b);

/* (a, b) */
float ip_fvector(FVector a, FVector b);

/* ||a||_1 */
float norm1_fvector(FVector a);

/* ||a||_2 */
float norm2_fvector(FVector a);

/* ||a||_infty */
float normi_fvector(FVector a);

/* c := a */
void subst_fvector(FVector c, FVector a);

/* c := 0 */
void set0_fvector(FVector c);

/* append 2005.07/12 */
/*
	ret(index_start) = src(src_index_start)
	 ...
	ret(index_end  ) = src(src_index_end)
*/
void copy_fvector_ij(FVector ret, long int index_start, long int index_end, FVector src, long int src_index_start, long int src_index_end);

/*************************************************/
/* Matrix Caluculations for FMatrix              */
/*
FMatrix init_fmatrix(long int row_dimension, long int col_dimension)
void free_fmatrix(FMatrix mat)
float normf_fmatrix(FMatrix mat)
float normi_fmatrix(FMatrix mat)
float norm1_fmatrix(FMatrix mat)
void add_fmatrix(FMatrix c, FMatrix a, FMatrix b);
void sub_fmatrix(FMatrix c, FMatrix a, FMatrix b);
void mul_fmatrix(FMatrix c, FMatrix a, FMatrix b);
void transpose_fmatrix(FMatrix c, FMatrix a);
void mul_fmatrix_dvec(FVector v, FMatrix a, FVector vb)
void mul_fmatrixt_dvec(FVector v, FMatrix a, FVector vb)
void inv_fmatrix(FMatrix a);
void subst_dmatrux(FMatrix c, FMatrix a);
*/
/*************************************************/
FMatrix init_fmatrix(long int row_dimension, long int col_dimension);

void free_fmatrix(FMatrix mat);

/*************************************************/
/* Matrix Caluculations for FMatrix              */
/*
float normf_fmatrix(FMatrix mat)
float normi_fmatrix(FMatrix mat)
float norm1_fmatrix(FMatrix mat)
void add_fmatrix(FMatrix c, FMatrix a, FMatrix b);
void sub_fmatrix(FMatrix c, FMatrix a, FMatrix b);
void mul_fmatrix(FMatrix c, FMatrix a, FMatrix b);
void transpose_fmatrix(FMatrix c, FMatrix a);
void mul_fmatrix_dvec(FVector v, FMatrix a, FVector vb)
void mul_fmatrixt_dvec(FVector v, FMatrix a, FVector vb)
void inv_fmatrix(FMatrix a);
void subst_dmatrux(FMatrix c, FMatrix a);
*/
/*************************************************/
/* Frobenius Norm of Matrix */
float normf_fmatrix(FMatrix mat);

/* Frobenius Norm of Matrix: array type */
float normf_fmatrix_array(float mat[], int row_dim, int col_dim);

/* Infinity Norm of Matrix */
float normi_fmatrix(FMatrix mat);

/* 1 Norm of Matrix */
float norm1_fmatrix(FMatrix mat);

/* c = a + b */
void add_fmatrix(FMatrix c, FMatrix a, FMatrix b);

/* c = a - b */
void sub_fmatrix(FMatrix c, FMatrix a, FMatrix b);

/* c = sc * a */
void cmul_fmatrix(FMatrix c, float sc, FMatrix a);

/* c = a * b */
void mul_fmatrix(FMatrix c, FMatrix a, FMatrix b);

/* c = a^T */
void transpose_fmatrix(FMatrix c, FMatrix a);

/* c := a */
void subst_fmatrix(FMatrix c, FMatrix a);

/* c := 0 */
void set0_fmatrix(FMatrix c);

/* c := I */
void setI_fmatrix(FMatrix c);

/* v = a * vb */
void mul_fmatrix_fvec(FVector v, FMatrix a, FVector vb);

/* v = a^T * vb */
//void mul_fmatrixt_dvec(FVector v, FMatrix a, FVector vb);
void mul_fmatrixt_fvec_old(FVector v, FMatrix a, FVector vb);

/* v = a^T * vb */
// Dangerous!
//void mul_fmatrixt_dvec_simd(FVector v, FMatrix a, FVector vb);
void mul_fmatrixt_fvec(FVector v, FMatrix a, FVector vb);

/* a = a^(-1) */
/* square matrix only */
void inv_fmatrix(FMatrix a);

/* Double */

/* 1. Hilbert Matrix */
void hilbert_fmatrix(FMatrix a, long int dim);

/* 2. Lotkin Matrix */
void lotkin_fmatrix(FMatrix a, long int dim);

/* 3. Frank Matrix */
void frank_fmatrix(FMatrix a, long int dim);

/* 4. Tridiagonal Matrix */
void tridiag_fmatrix(FMatrix a, FVector low_subdiag, FVector diag, FVector up_subdiag, long int dim);

/* 5. Integer Symmetrix Random Matrix */
void int_sym_rand_fmatrix(FMatrix mat, long int max, long int seed, long int dim);

/* 6. Integer Unsymmetrix Random Matrix */
void int_unsym_rand_fmatrix(FMatrix mat, long int max, long int seed, long int dim);

/* 7. Real Diagonal Matrix */
void diag_fmatrix(FMatrix mat, FVector diag, long int dim);

/* 8. Toeplitz Matrix */
void toeplitz_fmatrix(FMatrix mat, float gamma_param, long int dim);

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
int FLUdecomp(FMatrix a);
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       FMatrix a: Matrix (given by user)                 */
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
int SolveFLS(FVector answer, FMatrix lu, FVector b);
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       FMatrix lu: LU decomposed Matrix (given by user)   */
/*       FVector b: constant vector (given by user)         */
/*       FVector answer: Solution for linear system         */
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
int FLUdecompP(FMatrix a, long int ch[]);
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       FMatrix a: Matrix (given by user)                  */
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
int SolveFLSP(FVector answer, FMatrix lu, FVector b, long int ch[]);
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       FMatrix lu[]: LU decomposed Matrix (given by user) */
/*       FVector b[]: constant vector (given by user)       */
/*       FVector answer[]: Solution for linear system       */
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
int FLUdecompC(FMatrix a, long int row_ch[], long int col_ch[]);
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       FMatrix a[]: Matrix (given by user)                */
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
int SolveFLSC(FVector answer, FMatrix lu, FVector b, long int row_ch[], long int col_ch[]);
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       FMatrix lu: LU decomposed Matrix (given by user)   */
/*       FVector b: constant vector (given by user)         */
/*       FVector answer: Solution for linear system         */
/*       long int row_ch[]: Row order (given by user)       */
/*       long int col_ch[]: Column order (given by user)    */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_fmatrix(FMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

#if defined (__cplusplus)
}
#endif // defined (__cplusplus)

#endif // __BNC_DLINEAR_H__