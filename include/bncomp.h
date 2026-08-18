/********************************************************************************/
/* bncomp.h: Parallelized Mutiple Precision Linear Computation Library          */
/*                                                             with OpenMP      */
/* Copyright (C) 2013-2023 Tomonori Kouya                                       */
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
#ifndef __BNCOMP_H
#define __BNCOMP_H

#include <omp.h>
#include "dlinear.h"
#include "ddlinear.h"
#include "cddlinear.h"
#include "tdlinear.h"
#include "ctdlinear.h"
#include "qdlinear.h"
#include "cqdlinear.h"
#include "dslinear.h"
#include "tslinear.h"
#include "qslinear.h"

//#include "mpflinear.h"
//#include "clinear.h"
#include "cdlinear.h"
#include "cmpflinear.h"

#include "bmatrix.h"

// Sparse
#include "bncsparse.h"

// Poly
#include "poly.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/* global variable _bncomp */
#define BNCOMP_MAX_NUM_THREADS 128
//#define BNCOMP_MAX_NUM_THREADS 1024

#ifdef USE_OMP
	static int _bncomp_num_threads = BNCOMP_MAX_NUM_THREADS;
	//static int _bncomp_num_tmp = MAX_NUM_BNCOMP_TMP;
	static int _bncomp_num_tmp = BNCOMP_MAX_NUM_THREADS;
#else // USE_OMP
	static int _bncomp_num_threads = 1;
	//static int _bncomp_num_tmp = MAX_NUM_BNCOMP_TMP;
	static int _bncomp_num_tmp = 1;
#endif // USE_OMP

// Global working variable for double precision arithmetic
//#ifndef _BNCOMP_DEFINE_BNCOMP_GVALS
//#define _BNCOMP_DEFINE_BNCOMP_GVALS

extern double  _bncomp_g_dval[BNCOMP_MAX_NUM_THREADS];
extern DVector _bncomp_g_dvec[BNCOMP_MAX_NUM_THREADS];
extern DMatrix _bncomp_g_dmat[BNCOMP_MAX_NUM_THREADS];

// Global working variables for multiple precision arithmetic
#ifdef USE_GMP
extern mpf_t     _bncomp_g_mpfval[BNCOMP_MAX_NUM_THREADS];
extern mpf_t     _bncomp_g_mpfval2[BNCOMP_MAX_NUM_THREADS];
extern MPFVector _bncomp_g_mpfvec[BNCOMP_MAX_NUM_THREADS];
extern MPFVector _bncomp_g_mpfvec2[BNCOMP_MAX_NUM_THREADS];
extern MPFVector _bncomp_g_mpfvec3[BNCOMP_MAX_NUM_THREADS];
extern MPFVector _bncomp_g_mpfvec4[BNCOMP_MAX_NUM_THREADS];
extern MPFMatrix _bncomp_g_mpfmat[BNCOMP_MAX_NUM_THREADS];
#endif // USE_GMP

/*
double  _bncomp_g_dval[BNCOMP_MAX_NUM_THREADS];
DVector _bncomp_g_dvec[BNCOMP_MAX_NUM_THREADS];
DMatrix _bncomp_g_dmat[BNCOMP_MAX_NUM_THREADS];

// Global working variables for multiple precision arithmetic
#ifdef USE_GMP
mpf_t     _bncomp_g_mpfval[BNCOMP_MAX_NUM_THREADS];
mpf_t     _bncomp_g_mpfval2[BNCOMP_MAX_NUM_THREADS];
MPFVector _bncomp_g_mpfvec[BNCOMP_MAX_NUM_THREADS];
MPFVector _bncomp_g_mpfvec2[BNCOMP_MAX_NUM_THREADS];
MPFVector _bncomp_g_mpfvec3[BNCOMP_MAX_NUM_THREADS];
MPFVector _bncomp_g_mpfvec4[BNCOMP_MAX_NUM_THREADS];
MPFMatrix _bncomp_g_mpfmat[BNCOMP_MAX_NUM_THREADS];
#endif // USE_GMP
*/

//#endif // _BNCOMP_DEFINE_BNCOMP_GVALS

// set_bncomp_num_threads(int num_threads)
int set_bncomp_num_threads(int num_threads);

// get number of threads in BNCOMP
int get_bncomp_num_threads(void);

/* initialize */
void _bncomp_init_g_d(int vec_dim, int mat_row_dim, int mat_col_dim);

/* finalize */
void _bncomp_free_g_d(void);

//---------------------------------------
// Double
//---------------------------------------

//--------
// Vector
//--------
/* c := a */
void _bncomp_subst_dvector(DVector c, DVector a);

/* c = a + b */
void _bncomp_add_dvector(DVector c, DVector a, DVector b);

/* c = a - b */
void _bncomp_sub_dvector(DVector c, DVector a, DVector b);

/* c = val * a */
void _bncomp_cmul_dvector(DVector c, double val, DVector a);

/* (a, b) */
void _bncomp_ip_dvector(double ret, DVector a, DVector b);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void _bncomp_row_swap_dmatrix(DMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

//--------
// Matrix
//--------
/* c := a + b */
void _bncomp_add_dmatrix(DMatrix c, DMatrix a, DMatrix b);

/* c := a - b */
void _bncomp_sub_dmatrix(DMatrix c, DMatrix a, DMatrix b);

/* c := sc * a */
void _bncomp_cmul_dmatrix(DMatrix c, double sc, DMatrix a);

/* c = a * b */
void _bncomp_mul_dmatrix(DMatrix ret, DMatrix a, DMatrix b);

/* c := a */
void _bncomp_subst_dmatrix(DMatrix c, DMatrix a);

/* c := I */
void _bncomp_setI_dmatrix(DMatrix c);

// set a zero matrix
//void set0_dmatrix(DMatrix mat)
void _bncomp_set0_dmatrix(DMatrix mat);

/* v := a * vb */
void _bncomp_mul_dmatrix_dvec(DVector v, DMatrix a, DVector vb);

/* v := a^T * vb */
void _bncomp_mul_dmatrixt_dvec(DVector v, DMatrix a, DVector vb);

/* v := a * vb */
void _bncomp_mul_cdmatrix_cdvec(CDVector v, CDMatrix a, CDVector vb);

/* v := a^T * vb */
void _bncomp_mul_cdmatrixt_cdvec(CDVector v, CDMatrix a, CDVector vb);

// Matrix multiplication based on Ozaki scheme
void _bncomp_mul_dmatrix_oz(DMatrix ret, DMatrix a, int max_num_div_a, DMatrix b, int max_num_div_b);

// Fit dimension to be multiple of min_dim
void _bncomp_mul_cdmatrix_oz_3m(CDMatrix ret, CDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CDMatrix b, int max_num_div_b_real, int max_num_div_b_image);

// Fit dimension to be multiple of min_dim
void _bncomp_mul_cdmatrix_oz_4m(CDMatrix ret, CDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CDMatrix b, int max_num_div_b_real, int max_num_div_b_image);

// Simple triple-loop-way matrix multiplication (3M)
void _bncomp_mul_cdmatrix_3m(CDMatrix ret, CDMatrix a, CDMatrix b);

// Simple triple-loop-way matrix multiplication (4M)
void _bncomp_mul_cdmatrix_4m(CDMatrix ret, CDMatrix a, CDMatrix b);

//-----------------------------------------------
// Block linear computing
//-----------------------------------------------

/* substitute vector blocks */
void _bncomp_subst_dvector_blocks(DVector ret[], DVector v[], long int num_blocks);

/* inner product of vector blocks */
double _bncomp_ip_dvector_blocks(DVector va[], DVector vb[], long int num_blocks);

/* norm2 of vector blocks */
double _bncomp_norm2_dvector_blocks(DVector v[], long int num_blocks);

/* ret := va + alpha * vb */
void _bncomp_add_cmul_dvector_blocks(DVector ret[], DVector va[], double alpha, DVector vb[], long int num_blocks);

/* ret := (A \otimes I) vb[] */
void _bncomp_kmul_dmatrixI_dvector_blocks(DVector ret[], DMatrix mat, DVector vb[], long int num_blocks);

/* ret := -vb */
void _bncomp_neg_dvector_blocks(DVector vb[], long int num_blocks);

/* norm_inf of vector blocks */
double _bncomp_normi_dvector_blocks(DVector v[], long int num_blocks);

/* ret := va - vb */
void _bncomp_sub_dvector_blocks(DVector ret[], DVector va[], DVector vb[], long int num_blocks);

/* ret := va + vb */
void _bncomp_add_dvector_blocks(DVector ret[], DVector va[], DVector vb[], long int num_blocks);

/* ret := sum^n_{i,j=1} aij^2 */
double _bncomp_sumsqr_dmatrix(DMatrix mat);

/* Frobenius Norm for tridiagonal block matrix */
double _bncomp_normf_dmatrix_tridiag_blocks(DMatrix aim1[], DMatrix aii[], DMatrix aip1[], long int num_blocks);

/* [ ret[0]              ]    [ b[0]              ]   [ aii [0] aip1[0]                                                   ]   [ x[0]              ] */
/* [ ret[1]              ]    [ b[1]              ]   [ aim1[0] aii [1] aip1[1]                                           ]   [ x[1]              ] */
/* [ ................... ] := [ ................. ] - [ ................................................................. ] * [ ................. ] */
/* [ ret[num_blocks - 2] ]    [ b[num_blocks - 2] ]   [    aim1[num_blocks - 3] aii [num_blocks - 2] aip1[num_blocsk - 2] ]   [ x[num_blocks - 2] ] */
/* [ ret[num_blocks - 1] ]    [ b[num_blocks - 1] ]   [                         aim1[num_blocks - 2] aii [num_blocks - 1] ]   [ x[num_blocks - 1] ] */
void _bncomp_residual_dmat_dvec_tridiag_blocks(DVector ret[], DVector b[], DMatrix aim1[], DMatrix aii[], DMatrix aip1[], DVector x[], long int num_blocks);

/* residual for band matrix */
void _bncomp_residual_dbmat_dvec_tridiag_blocks(DVector ret[], DVector b[], DBMatrix aim1[], DBMatrix aii[], DBMatrix aip1[], DVector x[], long int num_blocks);

/* substitute vector blocks */
//void _bncomp_subst_fvector_dvec_blocks(FVector ret[], DVector v[], long int num_blocks);

/* substitute vector blocks */
//void _bncomp_subst_dvector_fvec_blocks(DVector ret[], FVector v[], long int num_blocks);

/* substitution of tridiagonal block matrix */
//void _bncomp_subst_fmatrix_dmat_tridiag_blocks(FMatrix aim1_f[], FMatrix aii_f[], FMatrix aip1_f[], DMatrix aim1[], DMatrix aii[], DMatrix aip1[], long int num_blocks);

/* ret[] := a[] * vb[] */
void _bncomp_mul_dtridiag_dvec_blocks(DVector ret[], DMatrix aim1[], DMatrix aii[], DMatrix aip1[], DVector vb[], DVector tmpv[], long int num_blocks);

/* ret[] := a_band[] * vb[] */
void _bncomp_mul_dbtridiag_dvec_blocks(DVector ret[], DBMatrix aim1[], DBMatrix aii[], DBMatrix aip1[], DVector vb[], DVector tmpv[], long int num_blocks);

/* norm_tol2_dvector */
double _bncomp_norm_tol2_dvector(DVector y_err, DVector y_new, DVector y_old, double rtol, double atol);

/* norm_tol2_dvector */
double _bncomp_norm2_diff_dvector(DVector y_new, DVector y_old, double rtol, double atol);

/* norm1 of vector blocks */
double _bncomp_norm1_dvector_blocks(DVector v[], long int num_blocks);

// normalize of vector
// kind of vector norm: 0...Infinity, 1...1norm, 2...euclid norm
// return value : ||v||
#define BNC_INFINITY_NORM	0
#define BNC_ONE_NORM		1
#define BNC_EUCLID_NORM		2
double _bncomp_norm_dvector(DVector vec, int kind_of_norm);

/* ret := val * vec */
void _bncomp_cmul_dvector_blocks(DVector ret[], double val, DVector vec[], long int num_blocks);

/* vec := val * vec */
void _bncomp_cmul2_dvector_blocks(DVector vec[], double val, long int num_blocks);

/* ||vec|| */
double _bncomp_norm_dvector_blocks(DVector vec[], long int num_blocks, int kind_of_norm);

/* ret := vec / ||vec|| */
double _bncomp_normalize_dvector_blocks(DVector ret[], DVector vec[], long int num_blocks, int kind_of_norm);

/* vec := vec / ||vec|| */
double _bncomp_normalize2_dvector_blocks(DVector vec[], long int num_blocks, int kind_of_norm);

/* BiCGSTAB for band matrix */
long int _bncomp_DBBiCGSTAB_triblock(DVector answer[], DBMatrix aim1[], DBMatrix aii[], DBMatrix aip1[], DVector b[], double reps, double aeps, long int maxtimes, long int num_blocks);

/* BiCGSTAB */
long int _bncomp_DBiCGSTAB_triblock(DVector answer[], DMatrix aim1[], DMatrix aii[], DMatrix aip1[], DVector b[], double reps, double aeps, long int maxtimes, long int num_blocks);

/* Left preconditioned BiCGSTAB for band matrix */
long int _bncomp_DBBiCGSTAB_triblock_irk(DVector answer[], DBMatrix aim1[], DBMatrix aii[], DBMatrix aip1[], DBMatrix hmat[], DVector b[], double reps, double aeps, long int maxtimes, long int num_blocks);

/* Left preconditioned BiCGSTAB */
long int _bncomp_DBiCGSTAB_triblock_irk(DVector answer[], DMatrix aim1[], DMatrix aii[], DMatrix aip1[], DMatrix hmat[], DVector b[], double reps, double aeps, long int maxtimes, long int num_blocks);

/* GMRES(m) for band matrix */
int _bncomp_DBGMRESm_triblock(DVector ret_x[], DBMatrix aim1[], DBMatrix aii[], DBMatrix aip1[], DVector vec_b[], int restart_times, int maxtimes, double rtol, double atol, long int num_blocks);

/* GMRES(m) */
int _bncomp_DGMRESm_triblock(DVector ret_x[], DMatrix aim1[], DMatrix aii[], DMatrix aip1[], DVector vec_b[], int restart_times, int maxtimes, double rtol, double atol, long int num_blocks);

//-----------------------------------------------
// Sparse
//-----------------------------------------------
/* Multiply DRSMatrix * DVector */
int _bncomp_mul_drsmatrix_dvec(DVector ret, DRSMatrix mat, DVector vec);

/* Multiply DRSMatrix^T * DVector */
int _bncomp_mul_drsmatrixt_dvec(DVector ret, DRSMatrix mat, DVector vec);

/* Multiply CDRSMatrix * CDVector */
int _bncomp_mul_cdrsmatrix_cdvec(CDVector ret, CDRSMatrix mat, CDVector vec);

/* Multiply CDRSMatrix^T * CDVector */
int _bncomp_mul_cdrsmatrixt_cdvec(CDVector ret, CDRSMatrix mat, CDVector vec);

/* Multiply conj(CDRSMatrix)^T * CDVector */
int _bncomp_mul_cdrsmatrixs_cdvec(CDVector ret, CDRSMatrix mat, CDVector vec);

/******/
/* DD */
/******/

/* c := a */
void _bncomp_subst_ddvector(DDVector c, DDVector a);

/* c = a + b */
void _bncomp_add_ddvector(DDVector c, DDVector a, DDVector b);

/* c = a - b */
void _bncomp_sub_ddvector(DDVector c, DDVector a, DDVector b);

/* c = val * a */
void _bncomp_cmul_ddvector(DDVector c, double val[DDSIZE], DDVector a);

/* (a, b) */
void _bncomp_ip_ddvector(double ret[DDSIZE], DDVector a, DDVector b);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void _bncomp_row_swap_ddmatrix(DDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

/* c := a + b */
void _bncomp_add_ddmatrix(DDMatrix c, DDMatrix a, DDMatrix b);

/* c := a - b */
void _bncomp_sub_ddmatrix(DDMatrix c, DDMatrix a, DDMatrix b);

/* c := sc * a */
void _bncomp_cmul_ddmatrix(DDMatrix c, double sc[DDSIZE], DDMatrix a);

/* c = a * b */
void _bncomp_mul_ddmatrix(DDMatrix ret, DDMatrix a, DDMatrix b);

/* OpenMP LU decomposition (partial pivoting) and its linear solver */
int _bncomp_DDLUdecompPM(DDMatrix a, long int ch[]);
int _bncomp_SolveDDLSPM(DDVector answer, DDMatrix lu, DDVector b, long int ch[]);
int _bncomp_TDLUdecompPM(TDMatrix a, long int ch[]);
int _bncomp_SolveTDLSPM(TDVector answer, TDMatrix lu, TDVector b, long int ch[]);
int _bncomp_QDLUdecompPM(QDMatrix a, long int ch[]);
int _bncomp_SolveQDLSPM(QDVector answer, QDMatrix lu, QDVector b, long int ch[]);

/* c := a */
void _bncomp_subst_ddmatrix(DDMatrix c, DDMatrix a);

/* c := I */
void _bncomp_setI_ddmatrix(DDMatrix c);

// set a zero matrix
//void set0_ddmatrix(DDMatrix mat)
void _bncomp_set0_ddmatrix(DDMatrix mat);

/* v := a * vb */
void _bncomp_mul_ddmatrix_ddvec(DDVector v, DDMatrix a, DDVector vb);

/* v := a^T * vb */
void _bncomp_mul_ddmatrixt_ddvec(DDVector v, DDMatrix a, DDVector vb);

void _bncomp_mul_ddmatrix_oz(DDMatrix ret, DDMatrix a, int max_num_div_a, DDMatrix b, int max_num_div_b);

// -------------------
// bncomp_linear_cdd.c
// -------------------
#include "_bncomp_linear_cdd.h"

//-----------------------------------------------
// Sparse
//-----------------------------------------------
/* Multiply DDRSMatrix * DDVector */
int _bncomp_mul_ddrsmatrix_ddvec(DDVector ret, DDRSMatrix mat, DDVector vec);

/* Multiply DDRSMatrix^T * DDVector */
int _bncomp_mul_ddrsmatrixt_ddvec(DDVector ret, DDRSMatrix mat, DDVector vec);

/* Multiply DRSMatrix * DDVector */
int _bncomp_mul_drsmatrix_ddvec(DDVector ret, DRSMatrix mat, DDVector vec);

/* Multiply DRSMatrix^T * DDVector */
int _bncomp_mul_drsmatrixt_ddvec(DDVector ret, DRSMatrix mat, DDVector vec);

/* Multiply CDDRSMatrix * CDDVector */
int _bncomp_mul_cddrsmatrix_cddvec(CDDVector ret, CDDRSMatrix mat, CDDVector vec);

/* Multiply CDDRSMatrix^T * CDDVector */
int _bncomp_mul_cddrsmatrixt_cddvec(CDDVector ret, CDDRSMatrix mat, CDDVector vec);

/* Multiply conj(CDDRSMatrix)^T * CDDVector */
int _bncomp_mul_cddrsmatrixs_cddvec(CDDVector ret, CDDRSMatrix mat, CDDVector vec);

/* Multiply CDRSMatrix * CDDVector */
int _bncomp_mul_cdrsmatrix_cddvec(CDDVector ret, CDRSMatrix mat, CDDVector vec);

/* Multiply CDRSMatrix^T * CDDVector */
int _bncomp_mul_cdrsmatrixt_cddvec(CDDVector ret, CDRSMatrix mat, CDDVector vec);

/* Multiply conj(CDRSMatrix)^T * CDDVector */
int _bncomp_mul_cdrsmatrixs_cddvec(CDDVector ret, CDRSMatrix mat, CDDVector vec);


/******/
/* TD */
/******/

/* c := a */
void _bncomp_subst_tdvector(TDVector c, TDVector a);

/* c = a + b */
void _bncomp_add_tdvector(TDVector c, TDVector a, TDVector b);

/* c = a - b */
void _bncomp_sub_tdvector(TDVector c, TDVector a, TDVector b);

/* c = val * a */
void _bncomp_cmul_tdvector(TDVector c, double val[TDSIZE], TDVector a);

/* (a, b) */
void _bncomp_ip_tdvector(double ret[TDSIZE], TDVector a, TDVector b);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void _bncomp_row_swap_tdmatrix(TDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

/* c := a + b */
void _bncomp_add_tdmatrix(TDMatrix c, TDMatrix a, TDMatrix b);

/* c := a - b */
void _bncomp_sub_tdmatrix(TDMatrix c, TDMatrix a, TDMatrix b);

/* c := sc * a */
void _bncomp_cmul_tdmatrix(TDMatrix c, double sc[TDSIZE], TDMatrix a);

/* c = a * b */
void _bncomp_mul_tdmatrix(TDMatrix ret, TDMatrix a, TDMatrix b);

/* c := a */
void _bncomp_subst_tdmatrix(TDMatrix c, TDMatrix a);

/* c := I */
void _bncomp_setI_tdmatrix(TDMatrix c);

// set a zero matrix
//void set0_tdmatrix(TDMatrix mat)
void _bncomp_set0_tdmatrix(TDMatrix mat);

/* v := a * vb */
void _bncomp_mul_tdmatrix_tdvec(TDVector v, TDMatrix a, TDVector vb);

/* v := a^T * vb */
void _bncomp_mul_tdmatrixt_tdvec(TDVector v, TDMatrix a, TDVector vb);

void _bncomp_mul_tdmatrix_oz(TDMatrix ret, TDMatrix a, int max_num_div_a, TDMatrix b, int max_num_div_b);

// -------------------
// bncomp_linear_ctd.c
// -------------------
#include "_bncomp_linear_ctd.h"

//-----------------------------------------------
// Sparse
//-----------------------------------------------
/* Multiply TDRSMatrix * TDVector */
int _bncomp_mul_tdrsmatrix_tdvec(TDVector ret, TDRSMatrix mat, TDVector vec);

/* Multiply TDRSMatrix^T * TDVector */
int _bncomp_mul_tdrsmatrixt_tdvec(TDVector ret, TDRSMatrix mat, TDVector vec);

/* Multiply DRSMatrix * TDVector */
int _bncomp_mul_drsmatrix_tdvec(TDVector ret, DRSMatrix mat, TDVector vec);

/* Multiply DRSMatrix^T * TDVector */
int _bncomp_mul_drsmatrixt_tdvec(TDVector ret, DRSMatrix mat, TDVector vec);

/* Multiply CTDRSMatrix * CTDVector */
int _bncomp_mul_ctdrsmatrix_ctdvec(CTDVector ret, CTDRSMatrix mat, CTDVector vec);

/* Multiply CTDRSMatrix^T * CTDVector */
int _bncomp_mul_ctdrsmatrixt_ctdvec(CTDVector ret, CTDRSMatrix mat, CTDVector vec);

/* Multiply conj(CTDRSMatrix)^T * CTDVector */
int _bncomp_mul_ctdrsmatrixs_ctdvec(CTDVector ret, CTDRSMatrix mat, CTDVector vec);

/* Multiply CDRSMatrix * CTDVector */
int _bncomp_mul_cdrsmatrix_ctdvec(CTDVector ret, CDRSMatrix mat, CTDVector vec);

/* Multiply CDRSMatrix^T * CTDVector */
int _bncomp_mul_cdrsmatrixt_ctdvec(CTDVector ret, CDRSMatrix mat, CTDVector vec);

/* Multiply conj(CDRSMatrix)^T * CTDVector */
int _bncomp_mul_cdrsmatrixs_ctdvec(CTDVector ret, CDRSMatrix mat, CTDVector vec);


/******/
/* QD */
/******/

/* c := a */
void _bncomp_subst_qdvector(QDVector c, QDVector a);

/* c = a + b */
void _bncomp_add_qdvector(QDVector c, QDVector a, QDVector b);

/* c = a - b */
void _bncomp_sub_qdvector(QDVector c, QDVector a, QDVector b);

/* c = val * a */
void _bncomp_cmul_qdvector(QDVector c, double val[QDSIZE], QDVector a);

/* (a, b) */
void _bncomp_ip_qdvector(double ret[QDSIZE], QDVector a, QDVector b);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void _bncomp_row_swap_qdmatrix(QDMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

/* c := a + b */
void _bncomp_add_qdmatrix(QDMatrix c, QDMatrix a, QDMatrix b);

/* c := a - b */
void _bncomp_sub_qdmatrix(QDMatrix c, QDMatrix a, QDMatrix b);

/* c := sc * a */
void _bncomp_cmul_qdmatrix(QDMatrix c, double sc[QDSIZE], QDMatrix a);

/* c = a * b */
void _bncomp_mul_qdmatrix(QDMatrix ret, QDMatrix a, QDMatrix b);

/* c := a */
void _bncomp_subst_qdmatrix(QDMatrix c, QDMatrix a);

/* c := I */
void _bncomp_setI_qdmatrix(QDMatrix c);

// set a zero matrix
//void set0_qdmatrix(QDMatrix mat)
void _bncomp_set0_qdmatrix(QDMatrix mat);

/* v := a * vb */
void _bncomp_mul_qdmatrix_qdvec(QDVector v, QDMatrix a, QDVector vb);

/* v := a^T * vb */
void _bncomp_mul_qdmatrixt_qdvec(QDVector v, QDMatrix a, QDVector vb);

void _bncomp_mul_qdmatrix_oz(QDMatrix ret, QDMatrix a, int max_num_div_a, QDMatrix b, int max_num_div_b);

// -------------------
// bncomp_linear_cqd.c
// -------------------
#include "_bncomp_linear_cqd.h"

//-----------------------------------------------
// Sparse
//-----------------------------------------------
/* Multiply QDRSMatrix * QDVector */
int _bncomp_mul_qdrsmatrix_qdvec(QDVector ret, QDRSMatrix mat, QDVector vec);

/* Multiply QDRSMatrix^T * QDVector */
int _bncomp_mul_qdrsmatrixt_qdvec(QDVector ret, QDRSMatrix mat, QDVector vec);

/* Multiply DRSMatrix * QDVector */
int _bncomp_mul_drsmatrix_qdvec(QDVector ret, DRSMatrix mat, QDVector vec);

/* Multiply DRSMatrix^T * QDVector */
int _bncomp_mul_drsmatrixt_qdvec(QDVector ret, DRSMatrix mat, QDVector vec);

/* Multiply CQDRSMatrix * CQDVector */
int _bncomp_mul_cqdrsmatrix_cqdvec(CQDVector ret, CQDRSMatrix mat, CQDVector vec);

/* Multiply CQDRSMatrix^T * CQDVector */
int _bncomp_mul_cqdrsmatrixt_cqdvec(CQDVector ret, CQDRSMatrix mat, CQDVector vec);

/* Multiply conj(CQDRSMatrix)^T * CQDVector */
int _bncomp_mul_cqdrsmatrixs_cqdvec(CQDVector ret, CQDRSMatrix mat, CQDVector vec);

/* Multiply CDRSMatrix * CQDVector */
int _bncomp_mul_cdrsmatrix_cqdvec(CQDVector ret, CDRSMatrix mat, CQDVector vec);

/* Multiply CDRSMatrix^T * CQDVector */
int _bncomp_mul_cdrsmatrixt_cqdvec(CQDVector ret, CDRSMatrix mat, CQDVector vec);

/* Multiply conj(CDRSMatrix)^T * CQDVector */
int _bncomp_mul_cdrsmatrixs_cqdvec(CQDVector ret, CDRSMatrix mat, CQDVector vec);


/******/
/* DS */
/******/

/* c := a */
void _bncomp_subst_dsvector(DSVector c, DSVector a);

/* c = a + b */
void _bncomp_add_dsvector(DSVector c, DSVector a, DSVector b);

/* c = a - b */
void _bncomp_sub_dsvector(DSVector c, DSVector a, DSVector b);

/* c = val * a */
void _bncomp_cmul_dsvector(DSVector c, float val[DSSIZE], DSVector a);

/* (a, b) */
void _bncomp_ip_dsvector(float ret[DSSIZE], DSVector a, DSVector b);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void _bncomp_row_swap_dsmatrix(DSMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

/* c := a + b */
void _bncomp_add_dsmatrix(DSMatrix c, DSMatrix a, DSMatrix b);

/* c := a - b */
void _bncomp_sub_dsmatrix(DSMatrix c, DSMatrix a, DSMatrix b);

/* c := sc * a */
void _bncomp_cmul_dsmatrix(DSMatrix c, float sc[DSSIZE], DSMatrix a);

/* c = a * b */
void _bncomp_mul_dsmatrix(DSMatrix ret, DSMatrix a, DSMatrix b);

/* c := a */
void _bncomp_subst_dsmatrix(DSMatrix c, DSMatrix a);

/* c := I */
void _bncomp_setI_dsmatrix(DSMatrix c);

// set a zero matrix
void _bncomp_set0_dsmatrix(DSMatrix mat);

/* v := a * vb */
void _bncomp_mul_dsmatrix_dsvec(DSVector v, DSMatrix a, DSVector vb);

/* v := a^T * vb */
void _bncomp_mul_dsmatrixt_dsvec(DSVector v, DSMatrix a, DSVector vb);

void _bncomp_mul_dsmatrix_oz(DSMatrix ret, DSMatrix a, int max_num_div_a, DSMatrix b, int max_num_div_b);


/******/
/* TS */
/******/

/* c := a */
void _bncomp_subst_tsvector(TSVector c, TSVector a);

/* c = a + b */
void _bncomp_add_tsvector(TSVector c, TSVector a, TSVector b);

/* c = a - b */
void _bncomp_sub_tsvector(TSVector c, TSVector a, TSVector b);

/* c = val * a */
void _bncomp_cmul_tsvector(TSVector c, float val[TSSIZE], TSVector a);

/* (a, b) */
void _bncomp_ip_tsvector(float ret[TSSIZE], TSVector a, TSVector b);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void _bncomp_row_swap_tsmatrix(TSMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

/* c := a + b */
void _bncomp_add_tsmatrix(TSMatrix c, TSMatrix a, TSMatrix b);

/* c := a - b */
void _bncomp_sub_tsmatrix(TSMatrix c, TSMatrix a, TSMatrix b);

/* c := sc * a */
void _bncomp_cmul_tsmatrix(TSMatrix c, float sc[TSSIZE], TSMatrix a);

/* c = a * b */
void _bncomp_mul_tsmatrix(TSMatrix ret, TSMatrix a, TSMatrix b);

/* c := a */
void _bncomp_subst_tsmatrix(TSMatrix c, TSMatrix a);

/* c := I */
void _bncomp_setI_tsmatrix(TSMatrix c);

// set a zero matrix
void _bncomp_set0_tsmatrix(TSMatrix mat);

/* v := a * vb */
void _bncomp_mul_tsmatrix_tsvec(TSVector v, TSMatrix a, TSVector vb);

/* v := a^T * vb */
void _bncomp_mul_tsmatrixt_tsvec(TSVector v, TSMatrix a, TSVector vb);

void _bncomp_mul_tsmatrix_oz(TSMatrix ret, TSMatrix a, int max_num_div_a, TSMatrix b, int max_num_div_b);


/******/
/* QS */
/******/

/* c := a */
void _bncomp_subst_qsvector(QSVector c, QSVector a);

/* c = a + b */
void _bncomp_add_qsvector(QSVector c, QSVector a, QSVector b);

/* c = a - b */
void _bncomp_sub_qsvector(QSVector c, QSVector a, QSVector b);

/* c = val * a */
void _bncomp_cmul_qsvector(QSVector c, float val[QSSIZE], QSVector a);

/* (a, b) */
void _bncomp_ip_qsvector(float ret[QSSIZE], QSVector a, QSVector b);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void _bncomp_row_swap_qsmatrix(QSMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

/* c := a + b */
void _bncomp_add_qsmatrix(QSMatrix c, QSMatrix a, QSMatrix b);

/* c := a - b */
void _bncomp_sub_qsmatrix(QSMatrix c, QSMatrix a, QSMatrix b);

/* c := sc * a */
void _bncomp_cmul_qsmatrix(QSMatrix c, float sc[QSSIZE], QSMatrix a);

/* c = a * b */
void _bncomp_mul_qsmatrix(QSMatrix ret, QSMatrix a, QSMatrix b);

/* c := a */
void _bncomp_subst_qsmatrix(QSMatrix c, QSMatrix a);

/* c := I */
void _bncomp_setI_qsmatrix(QSMatrix c);

// set a zero matrix
void _bncomp_set0_qsmatrix(QSMatrix mat);

/* v := a * vb */
void _bncomp_mul_qsmatrix_qsvec(QSVector v, QSMatrix a, QSVector vb);

/* v := a^T * vb */
void _bncomp_mul_qsmatrixt_qsvec(QSVector v, QSMatrix a, QSVector vb);

void _bncomp_mul_qsmatrix_oz(QSMatrix ret, QSMatrix a, int max_num_div_a, QSMatrix b, int max_num_div_b);


/*******/
/* MPF */
/*******/
#ifdef USE_GMP

/* c := a */
void _bncomp_subst_mpfvector(MPFVector c, MPFVector a, long int dim);

/* c = a + b */
void _bncomp_add_mpfvector(MPFVector c, MPFVector a, MPFVector b, long int dim);

/* c = a - b */
void _bncomp_sub_mpfvector(MPFVector c, MPFVector a, MPFVector b, long int dim);

/* c = val * a */
void _bncomp_cmul_mpfvector(MPFVector c, mpf_t val, MPFVector a, long int dim);

// Matrix
/* c := a + b */
void _bncomp_add_mpfmatrix(MPFMatrix c, MPFMatrix a, MPFMatrix b);

/* c := a - b */
void _bncomp_sub_mpfmatrix(MPFMatrix c, MPFMatrix a, MPFMatrix b);

/* c := sc * a */
void _bncomp_cmul_mpfmatrix(MPFMatrix c, mpf_t sc, MPFMatrix a);

/* inner product of vector blocks */
void _bncomp_ip_mpfvector(mpf_t ret, MPFVector va, MPFVector vb);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void _bncomp_row_swap_mpfmatrix(MPFMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

/* c = a * b */
void _bncomp_mul_mpfmatrix(MPFMatrix c, MPFMatrix a, MPFMatrix b);

/* c := a */
void _bncomp_subst_mpfmatrix(MPFMatrix c, MPFMatrix a);

/* c := 0 */
void _bncomp_set0_mpfmatrix(MPFMatrix c);

/* c := I */
void _bncomp_setI_mpfmatrix(MPFMatrix c);

/* v := a * vb */
void _bncomp_mul_mpfmatrix_mpfvec(MPFVector v, MPFMatrix a, MPFVector vb);

/* v := a^T * vb */
void _bncomp_mul_mpfmatrixt_mpfvec(MPFVector v, MPFMatrix a, MPFVector vb);

// Matrix multiplication based on Ozaki scheme
void _bncomp_mul_mpfmatrix_oz(MPFMatrix ret, MPFMatrix a, int max_num_div_a, MPFMatrix b, int max_num_div_b);

void _bncomp_mul_mpfmatrix_mpfvec_oz(MPFVector ret, MPFMatrix a, int max_num_div_a, MPFVector vb, int max_num_div_vb);

/*******/
/* MPC */
/*******/

//--------
// Vector
//--------
/* c := a */
void _bncomp_subst_cmpfvector(CMPFVector c, CMPFVector a, long int dim);

/* c = a - b */
void _bncomp_sub_cmpfvector(CMPFVector c, CMPFVector a, CMPFVector b, long int dim);

/* c = a + b */
void _bncomp_add_cmpfvector(CMPFVector c, CMPFVector a, CMPFVector b, long int dim);

/* c = val * a */
void _bncomp_cmul_cmpfvector(CMPFVector c, mpc_t val, CMPFVector a, long int dim);

/* inner product of vector blocks */
void _bncomp_ip_cmpfvector(mpc_t ret, CMPFVector va, CMPFVector vb);

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_endend_J)
void _bncomp_row_swap_cmpfmatrix(CMPFMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end);

//--------
// Matrix
//--------
/* c := a + b */
void _bncomp_add_cmpfmatrix(CMPFMatrix c, CMPFMatrix a, CMPFMatrix b);

/* c := a - b */
void _bncomp_sub_cmpfmatrix(CMPFMatrix c, CMPFMatrix a, CMPFMatrix b);

/* c := sc * a */
void _bncomp_cmul_cmpfmatrix(CMPFMatrix c, mpc_t sc, CMPFMatrix a);

/* c = a * b */
void _bncomp_mul_cmpfmatrix(CMPFMatrix c, CMPFMatrix a, CMPFMatrix b);

// Define _bncomp_cmpfmatrix_simple
#define _bncomp_mul_cmpfmatrix_simple _bncomp_mul_cmpfmatrix

/* c := a */
void _bncomp_subst_cmpfmatrix(CMPFMatrix c, CMPFMatrix a);

/* c := 0 */
void _bncomp_set0_cmpfmatrix(CMPFMatrix c);

/* c := I */
void _bncomp_setI_cmpfmatrix(CMPFMatrix c);

// Fully rewrite: 2022-03-29(Tue) T.Kouya
/* v := a * vb */
void _bncomp_mul_cmpfmatrix_cmpfvec(CMPFVector v, CMPFMatrix a, CMPFVector vb);

// Fully rewrite: 2022-03-29(Tue) T.Kouya
/* v := a^T * vb */
void _bncomp_mul_cmpfmatrixt_cmpfvec(CMPFVector v, CMPFMatrix a, CMPFVector vb);

/* v := conj(a)^T * vb */
void _bncomp_mul_cmpfmatrixs_cmpfvec(CMPFVector v, CMPFMatrix a, CMPFVector vb);

// Matrix multiplication based on Ozaki scheme (4M)
void _bncomp_mul_cmpfmatrix_oz_4m(CMPFMatrix ret, CMPFMatrix a, int max_num_div_a_real, int max_num_div_a_image, CMPFMatrix b, int max_num_div_b_real, int max_num_div_b_image);

// Matrix multiplication based on Ozaki scheme (3M)
void _bncomp_mul_cmpfmatrix_oz_3m(CMPFMatrix ret, CMPFMatrix a, int max_num_div_a_real, int max_num_div_a_image, CMPFMatrix b, int max_num_div_b_real, int max_num_div_b_image);

#ifdef USE_4M
#define _bncomp_mul_cmpfmatrix_oz _bncomp_mul_cmpfmatrix_oz_4m
#else // USE_4M
#define _bncomp_mul_cmpfmatrix_oz _bncomp_mul_cmpfmatrix_oz_3m
#endif // USE_4M

/* initialize */
void _bncomp_init_g_mpf(int vec_dim, int mat_row_dim, int mat_col_dim);

/* finalize */
void _bncomp_free_g_mpf(void);

/* A^T * diag(d0, ..., dn) */
void _bncomp_mul_mpfmatrixt_mpfdiagmat(MPFMatrix ret, MPFMatrix a, MPFVector d);

/* substitute vector blocks */
void _bncomp_subst_mpfvector_blocks(MPFVector ret[], MPFVector v[], long int num_blocks);

/* inner product of vector blocks */
void _bncomp_ip_mpfvector_blocks(mpf_t ret, MPFVector va[], MPFVector vb[], long int num_blocks);

/* norm2 of vector blocks */
void _bncomp_norm2_mpfvector_blocks(mpf_t ret, MPFVector v[], long int num_blocks);

/* ret := va + alpha * vb */
void _bncomp_add_cmul_mpfvector_blocks(MPFVector ret[], MPFVector va[], mpf_t alpha, MPFVector vb[], long int num_blocks);

/* ret := (A \otimes I) vb[] */
void _bncomp_kmul_mpfmatrixI_mpfvector_blocks(MPFVector ret[], MPFMatrix mat, MPFVector vb[], long int num_blocks);

/* ret := -vb */
void _bncomp_neg_mpfvector_blocks(MPFVector vb[], long int num_blocks);

/* norm_inf of vector blocks */
void _bncomp_normi_mpfvector_blocks(mpf_t ret, MPFVector v[], long int num_blocks);

/* ret := va - vb */
void _bncomp_sub_mpfvector_blocks(MPFVector ret[], MPFVector va[], MPFVector vb[], long int num_blocks);

/* ret := va + vb */
void _bncomp_add_mpfvector_blocks(MPFVector ret[], MPFVector va[], MPFVector vb[], long int num_blocks);

/* ret := sum^n_{i,j=1} aij^2 */
void _bncomp_sumsqr_mpfmatrix(mpf_t ret, MPFMatrix mat);

/* Frobenius Norm for tridiagonal block matrix */
void _bncomp_normf_mpfmatrix_tridiag_blocks(mpf_t ret, MPFMatrix aim1[], MPFMatrix aii[], MPFMatrix aip1[], long int num_blocks);

/* Frobenius Norm for tridiagonal block band matrix */
void _bncomp_normf_mpfbmatrix_tridiag_blocks(mpf_t ret, MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], long int num_blocks);

/* [ ret[0]              ]    [ b[0]              ]   [ aii [0] aip1[0]                                                   ]   [ x[0]              ] */
/* [ ret[1]              ]    [ b[1]              ]   [ aim1[0] aii [1] aip1[1]                                           ]   [ x[1]              ] */
/* [ ................... ] := [ ................. ] - [ ................................................................. ] * [ ................. ] */
/* [ ret[num_blocks - 2] ]    [ b[num_blocks - 2] ]   [    aim1[num_blocks - 3] aii [num_blocks - 2] aip1[num_blocsk - 2] ]   [ x[num_blocks - 2] ] */
/* [ ret[num_blocks - 1] ]    [ b[num_blocks - 1] ]   [                         aim1[num_blocks - 2] aii [num_blocks - 1] ]   [ x[num_blocks - 1] ] */
void _bncomp_residual_mpfmat_mpfvec_tridiag_blocks(MPFVector ret[], MPFVector b[], MPFMatrix aim1[], MPFMatrix aii[], MPFMatrix aip1[], MPFVector x[], long int num_blocks);

/* residual for band tridiagonal matrix */
void _bncomp_residual_mpfbmat_mpfvec_tridiag_blocks(MPFVector ret[], MPFVector b[], MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], MPFVector x[], long int num_blocks);

/* mul vec */
/* ret[] := a[] * x[] */
void _bncomp_mul_mpfmat_mpfvec_tridiag_blocks(MPFVector ret[], MPFMatrix aim1[], MPFMatrix aii[], MPFMatrix aip1[], MPFVector x[], long int num_blocks);

/* mul vec : band */
/* ret[] := a[] * x[] */
void _bncomp_mul_mpfbmat_mpfvec_tridiag_blocks(MPFVector ret[], MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], MPFVector x[], long int num_blocks);

/* ret[] := a[] * vb[] with working area */
void _bncomp_mul_mpftridiag_mpfvec_blocks(MPFVector ret[], MPFMatrix aim1[], MPFMatrix aii[], MPFMatrix aip1[], MPFVector vb[], MPFVector tmpv[], long int num_blocks);

/* ret[] := a_band[] * vb[] with working area */
void _bncomp_mul_mpfbtridiag_mpfvec_blocks(MPFVector ret[], MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], MPFVector vb[], MPFVector tmpv[], long int num_blocks);

/* substitution of tridiagonal block matrix */
void _bncomp_subst_mpfmatrix_tridiag_blocks(MPFMatrix aim1_f[], MPFMatrix aii_f[], MPFMatrix aip1_f[], MPFMatrix aim1[], MPFMatrix aii[], MPFMatrix aip1[], long int num_blocks);

/* substitution of tridiagonal block matrix */
void _bncomp_subst_mpfbmatrix_tridiag_blocks(MPFBMatrix aim1_f[], MPFBMatrix aii_f[], MPFBMatrix aip1_f[], MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], long int num_blocks);

/* substitute vector blocks */
void _bncomp_subst_dvector_mpfvec_blocks(DVector ret[], MPFVector v[], long int num_blocks);

/* substitute vector blocks */
void _bncomp_subst_mpfvector_dvec_blocks(MPFVector ret[], DVector v[], long int num_blocks);

/* substitution of tridiagonal block matrix */
void _bncomp_subst_dmatrix_mpfmat_tridiag_blocks(DMatrix aim1_f[], DMatrix aii_f[], DMatrix aip1_f[], MPFMatrix aim1[], MPFMatrix aii[], MPFMatrix aip1[], long int num_blocks);

/* substitution of tridiagonal block band matrix */
void _bncomp_subst_dbmatrix_mpfbmat_tridiag_blocks(DBMatrix aim1_f[], DBMatrix aii_f[], DBMatrix aip1_f[], MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], long int num_blocks);

/* norm1 of vector blocks */
void _bncomp_norm1_mpfvector_blocks(mpf_t ret, MPFVector v[], long int num_blocks);

/* ret := vec / ||vec|| */
void _bncomp_normalize_mpfvector(mpf_t norm, MPFVector ret, MPFVector vec, int kind_of_norm);

/* vec := vec / ||vec|| */
void _bncomp_normalize2_mpfvector(mpf_t norm, MPFVector vec, int kind_of_norm);

/* ret := val * vec */
void _bncomp_cmul_mpfvector_blocks(MPFVector ret[], mpf_t val, MPFVector vec[], long int num_blocks);

/* vec := val * vec */
void _bncomp_cmul2_mpfvector_blocks(MPFVector vec[], mpf_t val, long int num_blocks);

/* vector norm */
void _bncomp_norm_mpfvector_blocks(mpf_t norm, MPFVector vec[], long int num_blocks, int kind_of_norm);

/* normalization: ret[] := vec[] / ||vec[]|| */
void _bncomp_normalize_mpfvector_blocks(mpf_t norm, MPFVector ret[], MPFVector vec[], long int num_blocks, int kind_of_norm);

/* normalization: vec[] /= ||vec[]|| */
void _bncomp_normalize2_mpfvector_blocks(mpf_t norm, MPFVector vec[], long int num_blocks, int kind_of_norm);

/* BiCGSTAB for band matrix */
long int _bncomp_MPFBBiCGSTAB_triblock(MPFVector answer[], MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], MPFVector b[], mpf_t reps, mpf_t aeps, long int maxtimes, long int num_blocks);

/* BiCGSTAB */
long int _bncomp_MPFBiCGSTAB_triblock(MPFVector answer[], MPFMatrix aim1[], MPFMatrix aii[], MPFMatrix aip1[], MPFVector b[], mpf_t reps, mpf_t aeps, long int maxtimes, long int num_blocks);

/* Left preconditioned BiCGSTAB for band matrix */
long int _bncomp_MPFBBiCGSTAB_triblock_irk(MPFVector answer[], MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], MPFBMatrix hmat[], MPFVector b[], mpf_t reps, mpf_t aeps, long int maxtimes, long int num_blocks);

/* Left preconditioned BiCGSTAB */
long int _bncomp_MPFBiCGSTAB_triblock_irk(MPFVector answer[], MPFMatrix aim1[], MPFMatrix aii[], MPFMatrix aip1[], MPFMatrix hmat[], MPFVector b[], mpf_t reps, mpf_t aeps, long int maxtimes, long int num_blocks);

/* GMRES(m) for band matrix */
int _bncomp_MPFBGMRESm_triblock(MPFVector ret_x[], MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], MPFVector vec_b[], int restart_times, int maxtimes, mpf_t rtol, mpf_t atol, long int num_blocks);

/* GMRES(m) */
int _bncomp_MPFGMRESm_triblock(MPFVector ret_x[], MPFMatrix aim1[], MPFMatrix aii[], MPFMatrix aip1[], MPFVector vec_b[], int restart_times, int maxtimes, mpf_t rtol, mpf_t atol, long int num_blocks);

//-----------------------------------------------
// Sparse
//-----------------------------------------------
/* Multiply MPFRSMatrix * MPFVector */
int _bncomp_mul_mpfrsmatrix_mpfvec(MPFVector ret, MPFRSMatrix mat, MPFVector vec);

/* Multiply MPFRSMatrix^T * MPFVector */
int _bncomp_mul_mpfrsmatrixt_mpfvec(MPFVector ret, MPFRSMatrix mat, MPFVector vec);

/* Multiply DRSMatrix * MPFVector */
int _bncomp_mul_drsmatrix_mpfvec(MPFVector ret, DRSMatrix mat, MPFVector vec);

/* Multiply DRSMatrix^T * MPFVector */
int _bncomp_mul_drsmatrixt_mpfvec(MPFVector ret, DRSMatrix mat, MPFVector vec);

/* Multiply CMPFRSMatrix * CMPFVector */
int _bncomp_mul_cmpfrsmatrix_cmpfvec(CMPFVector ret, CMPFRSMatrix mat, CMPFVector vec);

/* Multiply CMPFRSMatrix^T * MPFVector */
int _bncomp_mul_cmpfrsmatrixt_cmpfvec(CMPFVector ret, CMPFRSMatrix mat, CMPFVector vec);

/* Multiply conj(CMPFRSMatrix)^T * MPFVector */
int _bncomp_mul_cmpfrsmatrixs_cmpfvec(CMPFVector ret, CMPFRSMatrix mat, CMPFVector vec);

/* Multiply CDRSMatrix * CMPFVector */
int _bncomp_mul_cdrsmatrix_cmpfvec(CMPFVector ret, CDRSMatrix mat, CMPFVector vec);

/* Multiply CDRSMatrix^T * MPFVector */
int _bncomp_mul_cdrsmatrixt_cmpfvec(CMPFVector ret, CDRSMatrix mat, CMPFVector vec);

/* Multiply conj(CDRSMatrix)^T * MPFVector */
int _bncomp_mul_cdrsmatrixs_cmpfvec(CMPFVector ret, CDRSMatrix mat, CMPFVector vec);

#endif // USE_GMP


/***********************************************/
// Conjugate-Gradient Method (Sparse & Dense Version)
/**********************************************/
//long int _bncomp_FCG(FVector answer, FMatrix a, FVector b, float reps, float aeps, long int maxtimes);
long int _bncomp_DCG_sp(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes);
long int _bncomp_bnc_DCG(DVector answer, DMatrix a, DVector b, double reps, double aeps, long int maxtimes);
// Complex
long int _bncomp_CDCOCG_sp(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes);
long int _bncomp_CDCOCG(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes);

// DD
long int _bncomp_DDCG_sp(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int _bncomp_DDCG_sp_d(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int _bncomp_DDCG(DDVector answer, DDMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
// Complex
long int _bncomp_CDDCOCG_sp(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int _bncomp_CDDCOCG_sp_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int _bncomp_CDDCOCG(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);

// TD
long int _bncomp_TDCG_sp(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int _bncomp_TDCG_sp_d(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int _bncomp_TDCG(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
// Complex
long int _bncomp_CTDCOCG_sp(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int _bncomp_CTDCOCG_sp_d(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int _bncomp_CTDCOCG(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);

// QD
long int _bncomp_QDCG_sp(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int _bncomp_QDCG_sp_d(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int _bncomp_QDCG(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
// Complex
long int _bncomp_CQDCOCG_sp(CQDVector answer, CQDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int _bncomp_CQDCOCG_sp_d(CQDVector answer, CDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int _bncomp_CQDCOCG(CQDVector answer, CQDMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);

#ifdef USE_GMP
long int _bncomp_MPFCG_sp(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int _bncomp_MPFCG_sp_d(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int _bncomp_MPFCG(MPFVector answer, MPFMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
// Complex
long int _bncomp_CMPFCOCG_sp(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int _bncomp_CMPFCOCG_sp_d(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int _bncomp_CMPFCOCG(CMPFVector answer, CMPFMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
#endif // USE_GMP

/********************************************/
/* Krylov Subspace Methods (Sparse Version) */
// krylov_omp.c, krylov_dd_omp.c, krylov_td_omp.c, krylov_qd_omp.c
/********************************************/
long int _bncomp_DBiCG_sp(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes);
long int _bncomp_DCGS_sp(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes);
long int _bncomp_DBiCGSTAB_sp(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes);
long int _bncomp_DGPBiCG_sp(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes);
long int _bncomp_DBiCG(DVector answer, DMatrix a, DVector b, double reps, double aeps, long int maxtimes);
long int _bncomp_DCGS(DVector answer, DMatrix a, DVector b, double reps, double aeps, long int maxtimes);
long int _bncomp_DBiCGSTAB(DVector answer, DMatrix a, DVector b, double reps, double aeps, long int maxtimes);
long int _bncomp_DGPBiCG(DVector answer, DMatrix a, DVector b, double reps, double aeps, long int maxtimes);

// Preconditioning
long int _bncomp_DBiCG_sp_iLU0(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int _bncomp_DCGS_sp_iLU0(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int _bncomp_DBiCGSTAB_sp_iLU0(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int _bncomp_DGPBiCG_sp_iLU0(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes, DRSMatrix ilu, double *norm2_res_history);

// DD
// Sparse
long int _bncomp_DDBiCG_sp(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int _bncomp_DDCGS_sp(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int _bncomp_DDBiCGSTAB_sp(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int _bncomp_DDGPBiCG_sp(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);

// Sparse & Preconditioning
long int _bncomp_DDBiCG_sp_iLU0(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, DDRSMatrix ilu, double *norm2_res_history);
long int _bncomp_DDCGS_sp_iLU0(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, DDRSMatrix ilu, double *norm2_res_history);
long int _bncomp_DDBiCGSTAB_sp_iLU0(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, DDRSMatrix ilu, double *norm2_res_history);
long int _bncomp_DDGPBiCG_sp_iLU0(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, DDRSMatrix ilu, double *norm2_res_history);

// Double Sparse
long int _bncomp_DDBiCG_sp_d(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int _bncomp_DDCGS_sp_d(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int _bncomp_DDBiCGSTAB_sp_d(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int _bncomp_DDGPBiCG_sp_d(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);

// Double Sparse and Preconditioning
long int _bncomp_DDBiCG_sp_d_iLU0(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int _bncomp_DDCGS_sp_d_iLU0(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int _bncomp_DDBiCGSTAB_sp_d_iLU0(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int _bncomp_DDGPBiCG_sp_d_iLU0(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);

// Dense
long int _bncomp_DDBiCG(DDVector answer, DDMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int _bncomp_DDCGS(DDVector answer, DDMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int _bncomp_DDBiCGSTAB(DDVector answer, DDMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int _bncomp_DDGPBiCG(DDVector answer, DDMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);

// TD
// Sparse
long int _bncomp_TDBiCG_sp(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int _bncomp_TDCGS_sp(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int _bncomp_TDBiCGSTAB_sp(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int _bncomp_TDGPBiCG_sp(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);

// Sparse & Preconditioning
long int _bncomp_TDBiCG_sp_iLU0(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDRSMatrix ilu, double *norm2_res_history);
long int _bncomp_TDCGS_sp_iLU0(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDRSMatrix ilu, double *norm2_res_history);
long int _bncomp_TDBiCGSTAB_sp_iLU0(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDRSMatrix ilu, double *norm2_res_history);
long int _bncomp_TDGPBiCG_sp_iLU0(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDRSMatrix ilu, double *norm2_res_history);

// Double Sparse
long int _bncomp_TDBiCG_sp_d(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int _bncomp_TDCGS_sp_d(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int _bncomp_TDBiCGSTAB_sp_d(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int _bncomp_TDGPBiCG_sp_d(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);

// Double Sparse and Preconditioning
long int _bncomp_TDBiCG_sp_d_iLU0(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int _bncomp_TDCGS_sp_d_iLU0(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int _bncomp_TDBiCGSTAB_sp_d_iLU0(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int _bncomp_TDGPBiCG_sp_d_iLU0(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);

// Dense
long int _bncomp_TDBiCG(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int _bncomp_TDCGS(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int _bncomp_TDBiCGSTAB(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int _bncomp_TDGPBiCG(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);

// QD
// Sparse
long int _bncomp_QDBiCG_sp(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int _bncomp_QDCGS_sp(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int _bncomp_QDBiCGSTAB_sp(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int _bncomp_QDGPBiCG_sp(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);

// Sparse and Preconditioning
long int _bncomp_QDBiCG_sp_iLU0(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDRSMatrix ilu, double *norm2_res_history);
long int _bncomp_QDCGS_sp_iLU0(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDRSMatrix ilu, double *norm2_res_history);
long int _bncomp_QDBiCGSTAB_sp_iLU0(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDRSMatrix ilu, double *norm2_res_history);
long int _bncomp_QDGPBiCG_sp_iLU0(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDRSMatrix ilu, double *norm2_res_history);

// Double Sparse
long int _bncomp_QDBiCG_sp_d(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int _bncomp_QDCGS_sp_d(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int _bncomp_QDBiCGSTAB_sp_d(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int _bncomp_QDGPBiCG_sp_d(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);

// Double Sparse and Preconditioning
long int _bncomp_QDBiCG_sp_d_iLU0(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int _bncomp_QDCGS_sp_d_iLU0(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int _bncomp_QDBiCGSTAB_sp_d_iLU0(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);
long int _bncomp_QDGPBiCG_sp_d_iLU0(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history);

// Dense
long int _bncomp_QDBiCG(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int _bncomp_QDCGS(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int _bncomp_QDBiCGSTAB(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int _bncomp_QDGPBiCG(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);

#ifdef USE_GMP
// SPARSE
long int _bncomp_MPFBiCG_sp(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int _bncomp_MPFCGS_sp(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int _bncomp_MPFBiCGSTAB_sp(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int _bncomp_MPFGPBiCG_sp(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);

// Double Sparse
long int _bncomp_MPFBiCG_sp_d(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int _bncomp_MPFCGS_sp_d(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int _bncomp_MPFBiCGSTAB_sp_d(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int _bncomp_MPFGPBiCG_sp_d(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes); 

// Dense
long int _bncomp_MPFBiCG(MPFVector answer, MPFMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int _bncomp_MPFCGS(MPFVector answer, MPFMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int _bncomp_MPFBiCGSTAB(MPFVector answer, MPFMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int _bncomp_MPFGPBiCG(MPFVector answer, MPFMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);

// Preconditioning
long int _bncomp_MPFBiCG_sp_iLU0(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, MPFRSMatrix ilu, MPFVector norm2_res_history);
long int _bncomp_MPFCGS_sp_iLU0(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, MPFRSMatrix ilu, MPFVector norm2_res_history);
long int _bncomp_MPFBiCGSTAB_sp_iLU0(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, MPFRSMatrix ilu, MPFVector norm2_res_history);
long int _bncomp_MPFGPBiCG_sp_iLU0(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, MPFRSMatrix ilu, MPFVector norm2_res_history);

// Preconditioning and double precision
long int _bncomp_MPFBiCG_sp_d_iLU0(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, DRSMatrix ilu, MPFVector norm2_res_history);
long int _bncomp_MPFCGS_sp_d_iLU0(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, DRSMatrix ilu, MPFVector norm2_res_history);
long int _bncomp_MPFBiCGSTAB_sp_d_iLU0(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, DRSMatrix ilu, MPFVector norm2_res_history);
long int _bncomp_MPFGPBiCG_sp_d_iLU0(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, DRSMatrix ilu, MPFVector norm2_res_history);
#endif // USE_GMP

/***********************************/
/* Complex Krylov Subspace Methods */
// krylov_c.c, krylov_cdd.c
/***********************************/
// Double precision
long int _bncomp_CDBiCG_sp(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes);
long int _bncomp_CDCGS_sp(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes);
long int _bncomp_CDBiCGSTAB_sp(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes);
long int _bncomp_CDGPBiCG_sp(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes);
// Dense
long int _bncomp_CDBiCG(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes);
long int _bncomp_CDCGS(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes);
long int _bncomp_CDBiCGSTAB(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes);
long int _bncomp_CDGPBiCG(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes);

long int _bncomp_CDDBiCG_sp    (CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int _bncomp_CDDCGS_sp     (CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int _bncomp_CDDBiCGSTAB_sp(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int _bncomp_CDDGPBiCG_sp  (CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
// Ddouble prec. sparse
long int _bncomp_CDDBiCG_sp_d    (CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int _bncomp_CDDCGS_sp_d     (CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int _bncomp_CDDBiCGSTAB_sp_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int _bncomp_CDDGPBiCG_sp_d  (CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
// Dense
long int _bncomp_CDDBiCG       (CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int _bncomp_CDDCGS        (CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int _bncomp_CDDBiCGSTAB   (CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);
long int _bncomp_CDDGPBiCG     (CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes);

long int _bncomp_CTDBiCG_sp    (CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int _bncomp_CTDCGS_sp     (CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int _bncomp_CTDBiCGSTAB_sp(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int _bncomp_CTDGPBiCG_sp  (CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
// Ddouble prec. sparse
long int _bncomp_CTDBiCG_sp_d    (CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int _bncomp_CTDCGS_sp_d     (CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int _bncomp_CTDBiCGSTAB_sp_d(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int _bncomp_CTDGPBiCG_sp_d  (CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
// Dense
long int _bncomp_CTDBiCG       (CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int _bncomp_CTDCGS        (CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int _bncomp_CTDBiCGSTAB   (CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);
long int _bncomp_CTDGPBiCG     (CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes);

long int _bncomp_CQDBiCG_sp    (CQDVector answer, CQDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int _bncomp_CQDCGS_sp     (CQDVector answer, CQDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int _bncomp_CQDBiCGSTAB_sp(CQDVector answer, CQDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int _bncomp_CQDGPBiCG_sp  (CQDVector answer, CQDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
// Ddouble prec. sparse
long int _bncomp_CQDBiCG_sp_d    (CQDVector answer, CDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int _bncomp_CQDCGS_sp_d     (CQDVector answer, CDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int _bncomp_CQDBiCGSTAB_sp_d(CQDVector answer, CDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int _bncomp_CQDGPBiCG_sp_d  (CQDVector answer, CDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
// Dense
long int _bncomp_CQDBiCG       (CQDVector answer, CQDMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int _bncomp_CQDCGS        (CQDVector answer, CQDMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int _bncomp_CQDBiCGSTAB   (CQDVector answer, CQDMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);
long int _bncomp_CQDGPBiCG     (CQDVector answer, CQDMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes);


#ifdef USE_GMP
long int _bncomp_CMPFBiCG_sp(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int _bncomp_CMPFCGS_sp(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int _bncomp_CMPFBiCGSTAB_sp(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int _bncomp_CMPFGPBiCG_sp(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
// Ddouble prec. sparse
long int _bncomp_CMPFBiCG_sp_d(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int _bncomp_CMPFCGS_sp_d(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int _bncomp_CMPFBiCGSTAB_sp_d(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int _bncomp_CMPFGPBiCG_sp_d(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
// Dense
long int _bncomp_CMPFBiCG(CMPFVector answer, CMPFMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int _bncomp_CMPFCGS(CMPFVector answer, CMPFMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int _bncomp_CMPFBiCGSTAB(CMPFVector answer, CMPFMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
long int _bncomp_CMPFGPBiCG(CMPFVector answer, CMPFMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes);
#endif //def USE_GMP

/***********************************/
/* DKA                             */
/* dka_omp.c                       */
/***********************************/
#ifdef USE_GMP

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void _bncomp_mpf_dka_init2(CMPFArray x_init, MPFPoly func, void (* get_radius)(mpf_t, MPFPoly), void (* get_center)(mpf_t, MPFPoly));

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void _bncomp_mpf_dka_init(CMPFArray x_init, MPFPoly func);

/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int _bncomp_mpf_dka(CMPFArray ans, CMPFArray x_init, MPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps);

/* Petkovic Method : 3rd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int _bncomp_mpf_petckovic(CMPFArray ans, CMPFArray x_init, MPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps);

/* Aberth Method : 3rd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int _bncomp_mpf_aberth(CMPFArray ans, CMPFArray x_init, MPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps);

#endif // USE_GMP

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif // __BNCOMP_H
