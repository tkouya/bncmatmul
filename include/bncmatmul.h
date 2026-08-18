/********************************************************************************/
/* bncmatmul.h: Basic Numerical Compucation library                             */
/*              for fast multiple-precision MATrix MULtiplicaiton               */
/*              including MPBLAS1 and MPBLAS2 functions                         */
/* Copyright (C) 2023 Tomonori Kouya                                            */
/*                                                                              */
/* Version 0.1  : Newly built as the base of BNCmatmul Version 0.22             */
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

// define _BNCMATMUL_H
#ifndef _BNCMATMUL_H
#define _BNCMATMUL_H

#ifdef __cplusplus
#include <cstdio>
#include <cmath>
#include <ccomplex>
#else // __cplusplus
#include <stdio.h>
#include <math.h>
#include <complex.h>
#endif // __cplusplus

// --- 
// matmul_strassen.h
// ---
#include "matmul_strassen.h"

// BNC common functions
#include "bnc_common.h"

// sparse
// bncsparse.h
#include "bncsparse.h"

// band
// bmatrix.h
#include "bmatrix.h"

// Expanding IO fuctions of MMIO
// bncmm.h
#include "bncmm.h"
#include "bncmm_c.h" // for complex matrix and vector

// poly, array, dka
#include "poly.h"

// MPI functions
// mpi_bnc.h
#ifdef USE_MPI
#include "mpi_bnc.h"
#endif // USE_MPI

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

// ---------------------------
// tridiagonal.c
// ---------------------------
/* trimat:= [vec[1][0] vec[0][0  ]             0 ... 0     ] */
/*          [vec[2][0] vec[1][1  ] vec[0][1  ] 0 ... 0     ] */
/*          [        ..............................        ] */
/*          [0 ... 0   vec[2][n-3] vec[1][n-2] vec[0][n-2] ] */
/*          [0 ... 0   0           vec[2][n-2] vec[1][n-1] ] */
/*                                                           */
int init_dmatrix_tri(DVector trimat[3], long int dim);
void free_dmatrix_tri(DVector trimat[3]);
void print_dmatrix_tri(DVector trimat[3]);

/* Transform Real Symmetric Square Matric to TridiagonalForm */
int dstrimat(DVector trimat[3], DMatrix mat, DMatrix proj_mat, int flag_get_proj_mat);

//#ifdef SINGLE_USE
/* [cosine   sine] [vec0] = [ nu * vec0 / |vec0| ] */
/* [-sine  cosine] [vec1]   [ 0                  ] */
/* where nu = sqrt(vec0^2+vec1^2) */
/* Input : vec0, vec1 */
/* Output: cosine, sine, new vec0(, vec1 = 0) */
void dplane_rotation(double *cosine, double *sine, double *vec0, double *vec1);
//#endif // SINGLE_USE

//int dtriqr(DVector trimat[3], double shift, long int i1, long int i2);
void dtriqr(DVector trimat[3], double shift, long int i1, long int i2);

/* [mat11 mat12] */
/* [mat21 mat22] (mat21 = mat12) */
double dwilkinson_shift_tri(double mat11, double mat12, double mat22);

int dtriqr_iteration(DVector trimat[3], double rtol, double atol, long int maxtimes);

/* Get eigenvector of tridiagonal matrices */
void dget_eigenvector_dtri(DVector eigen_vec, DVector trimat[3], double eigenvalue, long int drop_rank);

/* Multiply diagonal matrix */
void mul_ddiagmat_dvec(DVector ret, DVector diagmat, DVector vec);

/* GMP & MPFR */
#ifdef USE_GMP
/* trimat:= [vec[1][0] vec[0][0  ]             0 ... 0     ] */
/*          [vec[2][0] vec[1][1  ] vec[0][1  ] 0 ... 0     ] */
/*          [        ..............................        ] */
/*          [0 ... 0   vec[2][n-3] vec[1][n-2] vec[0][n-2] ] */
/*          [0 ... 0   0           vec[2][n-2] vec[1][n-1] ] */
/*                                                           */
int init_mpfmatrix_tri(MPFVector trimat[3], long int dim);
int init2_mpfmatrix_tri(MPFVector trimat[3], long int dim, unsigned long prec);
void free_mpfmatrix_tri(MPFVector trimat[3]);
void print2_mpfmatrix_tri(MPFVector trimat[3], int dprec);
void print_mpfmatrix_tri(MPFVector trimat[3]);

/* Transform Real Square Matrix to Hessemberg Form */
int mpfstrimat(MPFVector trimat[3], MPFMatrix mat, MPFMatrix proj_mat, int flag_get_proj_mat);

//#ifdef SINGLE_USE
/* [cosine   sine] [vec0] = [ nu * vec0 / |vec0| ] */
/* [-sine  cosine] [vec1]   [ 0                  ] */
/* where nu = sqrt(vec0^2+vec1^2) */
/* Input : vec0, vec1 */
/* Output: cosine, sine, new vec0(, vec1 = 0) */
void mpfplane_rotation(mpf_t cosine, mpf_t sine, mpf_t vec0, mpf_t vec1);
//#endif // SINGLE_USE

//int mpftriqr(MPFVector trimat[3], mpf_t shift, long int i1, long int i2);
void mpftriqr(MPFVector trimat[3], mpf_t shift, long int i1, long int i2);

/* [mat11 mat12] */
/* [mat21 mat22] (mat21 = mat12) */
void mpfwilkinson_shift_tri(mpf_t ret, mpf_t mat11, mpf_t mat12, mpf_t mat22);

int mpftriqr_iteration(MPFVector trimat[3], mpf_t rel_tol, mpf_t abs_tol, long int maxtimes);

/* Get eigenvector of tridiagonal matrices */
void mpfget_eigenvector_mpftri_cee(MPFVector eigen_vec, MPFVector trimat[3], mpf_t eigenvalue, long int drop_rank);

/* Multiply diagonal matrix */
void mul_mpfdiagmat_mpfvec(MPFVector ret, MPFVector diagmat, MPFVector vec);;
#endif // USE_GMP

#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus

#endif // _BNCMATMUL_H
