/********************************************************************************/
/* oz_scheme.h: Optimized linear computation with Ozaki scheme                  */
/* Copyright (C) 2022 Tomonori Kouya                                            */
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
#ifndef __BNC_OZ_SCHEME_H
#define __BNC_OZ_SCHEME_H

#include "rdd.h" // [dtq]dfloat and its arithmetic r[dtq]d_*
#include "matmul_strassen.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
/*------------------------------------------------------------------------------*/
/* Parallel execution layer of the Ozaki-scheme kernels                          */
/*                                                                               */
/* The kernels below spend their time in three places:                           */
/*                                                                               */
/*   1. splitting the multi-component operands into double-precision slices      */
/*      (memory bound, O(num_div * n^2) multi-component operations),             */
/*   2. the double-precision products of those slices (a DGEMM each),            */
/*   3. accumulating every slice product back in multi-component arithmetic      */
/*      (again O(num_div^2 * n^2)).                                              */
/*                                                                               */
/* Handing (2) to a threaded BLAS leaves (1) and (3) serial, and they are large  */
/* enough -- roughly 60-70% of the run time at n=1000..4000 -- to cap the        */
/* speedup badly.  So the kernels parallelize all three themselves with OpenMP   */
/* and ask the BLAS to stay single-threaded while they do it; the block loop     */
/* over the rows of C is dynamically scheduled, which also keeps heterogeneous   */
/* (big.LITTLE style) cores busy where a static split would stall on the slow    */
/* cores.                                                                        */
/*                                                                               */
/* Everything is tunable at run time, so a build never has to be repeated just   */
/* to try another setting:                                                       */
/*                                                                               */
/*   BNC_OZ_NUM_THREADS   threads used by the kernels    (default: OpenMP max)   */
/*   BNC_OZ_BLAS_THREADS  threads left to the CBLAS backend while the kernel     */
/*                        runs its own parallel loop     (default: 1)            */
/*   BNC_OZ_BLOCK_ROWS    rows of C per block            (default: automatic)    */
/*   BNC_OZ_BLOCKS_PER_THREAD  blocks per thread when BNC_OZ_BLOCK_ROWS is       */
/*                        automatic                      (default: 2)           */
/*   BNC_OZ_GEMM_MODE     "own"  = parallelize the block loop, BLAS serial       */
/*                        "blas" = serial block loop, threaded BLAS             */
/*                                                       (default: own)         */
/*                                                                               */
/* Without OpenMP every routine below degenerates to the original serial code    */
/* and the results are bit-identical either way: the row blocks are disjoint     */
/* and each element of the result still accumulates its slice products in the    */
/* same order.                                                                   */
/*------------------------------------------------------------------------------*/
#ifdef _OPENMP
#include <omp.h>
#endif // _OPENMP

#define BNC_OZ_GEMM_MODE_OWN  0
#define BNC_OZ_GEMM_MODE_BLAS 1

// threads used by the Ozaki-scheme kernels
int bnc_oz_get_num_threads(void);
void bnc_oz_set_num_threads(int num_threads);

// threads left to the CBLAS backend inside a parallel kernel
int bnc_oz_get_blas_threads(void);
void bnc_oz_set_blas_threads(int num_threads);

// rows of C per block; <= 0 means "decide automatically"
long int bnc_oz_get_block_rows(void);
void bnc_oz_set_block_rows(long int block_rows);

// BNC_OZ_GEMM_MODE_OWN or BNC_OZ_GEMM_MODE_BLAS
int bnc_oz_get_gemm_mode(void);
void bnc_oz_set_gemm_mode(int mode);

// rows per block actually used for a row_dim x (anything) product
long int bnc_oz_block_rows_for(long int row_dim, int num_threads);

// tell the CBLAS backend how many threads it may use inside a parallel kernel;
// returns the previous setting, which bnc_oz_blas_leave() puts back
int bnc_oz_blas_enter(void);
void bnc_oz_blas_leave(int prev_num_threads);

/*------------------------------------------------------------------------------*/
/* Exponent handling of the split                                                */
/*                                                                               */
/* The slices are plain double matrices, so everything the split touches has to  */
/* fit into the double exponent range -- but the operands do not.  mpf_t reaches */
/* 2^+-2^30, and even a DD matrix may legitimately hold entries near DBL_MAX.    */
/* Splitting them where they stand fails in three ways: the conversion to double */
/* returns +-Inf or 0, the threshold 2^(ceil(log2(mu)) + s) overflows to Inf and */
/* turns every entry into NaN, and once it underflows instead, (x + s) - s stops */
/* truncating the mantissa, so the slice products are quietly no longer exact.   */
/*                                                                               */
/* So each row of A and each column of B is scaled by a power of two before it   */
/* is split, and the exponent is handed back to the caller:                      */
/*                                                                               */
/*     A[i][k] = 2^sa[i] * sum_p slice_a[p][i][k]                                */
/*     B[k][j] = 2^sb[j] * sum_q slice_b[q][k][j]                                */
/*     C[i][j] = sum_{p,q} 2^(sa[p][i] + sb[q][j]) * (slice_a[p] slice_b[q])[i,j]*/
/*                                                                               */
/* The exponent is constant along k, so it factors out of the inner product and  */
/* the error-free property is untouched; scaling by a power of two is exact in   */
/* both double and mpf_t.  A fresh exponent is taken for every slice, so the     */
/* dynamic range a row may span is limited by num_div alone, not by the double   */
/* exponent range.                                                               */
/*                                                                               */
/* The shift arrays hold num_div * dim entries, indexed [slice * dim + i].       */
/* Passing NULL asks for the old unscaled split, which is what the callers that  */
/* cannot apply a scale factor still get.                                        */
/*------------------------------------------------------------------------------*/

// exponent e of x, so that |x| is in [2^(e-1), 2^e); 0 for zero and non-finite x
long int bnc_oz_exp2_d(double x);

// ldexp() with the shift saturated: a multiple-precision exponent does not fit
// in the int that ldexp() takes, and saturating flushes to 0 / +-Inf, which is
// what the unsaturated call would have returned anyway
double bnc_oz_ldexp(double x, long int shift);

/*------------------------------------------------------------------------------*/
/* Below 2^BNC_OZ_MIN_SCALED_EXP a row of a double-based matrix is left          */
/* unscaled: its slices could no longer be shifted back exactly (the low bits    */
/* would fall off the bottom of the subnormal range), and scaling cannot help    */
/* there anyway.  mpf_t has no such limit and is always scaled.                  */
/*------------------------------------------------------------------------------*/
#define BNC_OZ_MIN_SCALED_EXP (-1000L)

// ret_block := a[row_start : row_start + num_rows][*] * b   (plain double)
void bnc_oz_dgemm_block(double *ret_block, long int ld_ret_block, DMatrix a, long int row_start, long int num_rows, DMatrix b);
// ret_block := a[row_start : row_start + num_rows][*] * b   (plain double)
void bnc_oz_dgemv_block(double *ret_block, DMatrix a, long int row_start, long int num_rows, DVector b);

/***************/
/* oz_scheme.c */
/***************/
// ret_high + ret_low = org_vec
void extract_dvector(DVector ret_high_vec, DVector ret_low_vec, DVector org_vec, double num_bits);
// SplitMat_A
void split_dmatrix(DMatrix ret_high_mat, DMatrix ret_low_mat, DMatrix org_mat);
// SplitMat_B
void split_dmatrix_t(DMatrix ret_high_mat, DMatrix ret_low_mat, DMatrix org_mat);

// split vector
// ret_vec[0] + ret_vec[1] + ... ret_vec[num_div - 1] = org_vec
//void extract_dvector(DVector ret_vec[], int num_div, DVector org_vec, int num_bits)
// shift holds num_div exponents (may be NULL); see the exponent note above
int split_dvector_dvec_ex(DVector ret_vec[], long int shift[], int num_div, DVector org_vec);
int split_dvector_dvec(DVector ret_vec[], int num_div, DVector org_vec);

// SplitMat_A
// return real_num_div; row_shift holds num_div * row_dim exponents (may be NULL)
int split_dmatrix_dmat_ex(DMatrix ret_mat[], long int row_shift[], int num_div, DMatrix org_mat);
int split_dmatrix_dmat(DMatrix ret_mat[], int num_div, DMatrix org_mat);

// SplitMat_B
// return real_num_div; col_shift holds num_div * col_dim exponents (may be NULL)
int split_dmatrix_t_dmat_ex(DMatrix ret_mat[], long int col_shift[], int num_div, DMatrix org_mat);
int split_dmatrix_t_dmat(DMatrix ret_mat[], int num_div, DMatrix org_mat);

// Matrix multiplication based on Ozaki scheme
void mul_dmatrix_oz(DMatrix ret, DMatrix a, int max_num_div_a, DMatrix b, int max_num_div_b);

/******************/
/* dd_oz_scheme.c */
/******************/
// absmax_ddvector
void absmax_ddvector(double ret[DDSIZE], long int *max_index, DDVector vec);
/* c = a + (double)b */
void add_ddvector_dvec(DDVector c, DDVector a, DVector b);
/* c = a - (double)b */
void sub_ddvector_dvec(DDVector c, DDVector a, DVector b);
// split vector
// ret_vec[0] + ret_vec[1] + ... ret_vec[num_div - 1] = org_vec
// shift holds num_div exponents (may be NULL); see the exponent note above
int split_ddvector_dvec_ex(DVector ret_vec[], long int shift[], int num_div, DDVector org_vec);
int split_ddvector_dvec(DVector ret_vec[], int num_div, DDVector org_vec);
// absmax_row_ddmatrix
void absmax_row_ddmatrix(double mu[DDSIZE], long int *max_j, long int row_index, DDMatrix mat);
/* c := a + (doble)b */
void add_ddmatrix_dmat(DDMatrix c, DDMatrix a, DMatrix b);
/* c := a - (doble)b */
void sub_ddmatrix_dmat(DDMatrix c, DDMatrix a, DMatrix b);
// SplitMat_A
// return real_num_div
// row_shift holds num_div * row_dim exponents (may be NULL)
int split_ddmatrix_dmat_ex(DMatrix ret_mat[], long int row_shift[], int num_div, DDMatrix org_mat);
int split_ddmatrix_dmat(DMatrix ret_mat[], int num_div, DDMatrix org_mat);
// absmax_col_ddmatrix
void absmax_col_ddmatrix(double mu[DDSIZE], long int *max_i, long int col_index, DDMatrix mat);
// SplitMat_B
// return real_num_div
// col_shift holds num_div * col_dim exponents (may be NULL)
int split_ddmatrix_t_dmat_ex(DMatrix ret_mat[], long int col_shift[], int num_div, DDMatrix org_mat);
int split_ddmatrix_t_dmat(DMatrix ret_mat[], int num_div, DDMatrix org_mat);
// Matrix multiplication based on Ozaki scheme
void mul_ddmatrix_oz(DDMatrix ret, DDMatrix a, int max_num_div_a, DDMatrix b, int max_num_div_b);
// Matrix-Vector multiplication based on Ozaki scheme
void mul_ddmatrix_ddvec_oz(DDVector ret, DDMatrix a, int max_num_div_a, DDVector vb, int max_num_div_vb);

// Fit dimension to be multiple of min_dim
void mul_cddmatrix_oz_3m(CDDMatrix ret, CDDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CDDMatrix b, int max_num_div_b_real, int max_num_div_b_image);
// Fit dimension to be multiple of min_dim
void mul_cddmatrix_oz_4m(CDDMatrix ret, CDDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CDDMatrix b, 
int max_num_div_b_real, int max_num_div_b_image);
#ifdef USE_4M
#define mul_cddmatrix_oz mul_cddmatrix_oz_4m
#else // USE_4M
#define mul_cddmatrix_oz mul_cddmatrix_oz_3m
#endif // USE_4M


/******************/
/* td_oz_scheme.c */
/******************/
// absmax_tdvector
void absmax_tdvector(double ret[TDSIZE], long int *max_index, TDVector vec);
/* c = a + (double)b */
void add_tdvector_dvec(TDVector c, TDVector a, DVector b);
/* c = a - (double)b */
void sub_tdvector_dvec(TDVector c, TDVector a, DVector b);
// extract vector
// ret_vec[0] + ret_vec[1] + ... ret_vec[num_div - 1] = org_vec
// shift holds num_div exponents (may be NULL); see the exponent note above
int split_tdvector_dvec_ex(DVector ret_vec[], long int shift[], int num_div, TDVector org_vec);
int split_tdvector_dvec(DVector ret_vec[], int num_div, TDVector org_vec);
// absmax_row_tdmatrix
void absmax_row_tdmatrix(double mu[TDSIZE], long int *max_j, long int row_index, TDMatrix mat);
/* c := a + (doble)b */
void add_tdmatrix_dmat(TDMatrix c, TDMatrix a, DMatrix b);
/* c := a - (doble)b */
void sub_tdmatrix_dmat(TDMatrix c, TDMatrix a, DMatrix b);
// SplitMat_A
// row_shift holds num_div * row_dim exponents (may be NULL)
int split_tdmatrix_dmat_ex(DMatrix ret_mat[], long int row_shift[], int num_div, TDMatrix org_mat);
int split_tdmatrix_dmat(DMatrix ret_mat[], int num_div, TDMatrix org_mat);
// absmax_col_tdmatrix
void absmax_col_tdmatrix(double mu[TDSIZE], long int *max_i, long int col_index, TDMatrix mat);
// SplitMat_B
// col_shift holds num_div * col_dim exponents (may be NULL)
int split_tdmatrix_t_dmat_ex(DMatrix ret_mat[], long int col_shift[], int num_div, TDMatrix org_mat);
int split_tdmatrix_t_dmat(DMatrix ret_mat[], int num_div, TDMatrix org_mat);
// Matrix multiplication based on Ozaki scheme
void mul_tdmatrix_oz(TDMatrix ret, TDMatrix a, int max_num_div_a, TDMatrix b, int max_num_div_b);
// Matrix-Vector multiplication based on Ozaki scheme
void mul_tdmatrix_tdvec_oz(TDVector ret, TDMatrix a, int max_num_div_a, TDVector vb, int max_num_div_vb);

// Fit dimension to be multiple of min_dim
void mul_ctdmatrix_oz_3m(CTDMatrix ret, CTDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CTDMatrix b, int max_num_div_b_real, int max_num_div_b_image);
void mul_ctdmatrix_oz_4m(CTDMatrix ret, CTDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CTDMatrix b, int max_num_div_b_real, int max_num_div_b_image);
#ifdef USE_4M
#define mul_ctdmatrix_oz mul_ctdmatrix_oz_4m
#else // USE_4M
#define mul_ctdmatrix_oz mul_ctdmatrix_oz_3m
#endif // USE_4M


/******************/
/* qd_oz_scheme.c */
/******************/
// absmax_qdvector
void absmax_qdvector(double ret[QDSIZE], long int *max_index, QDVector vec);
/* c = a + (double)b */
void add_qdvector_dvec(QDVector c, QDVector a, DVector b);
/* c = a - (double)b */
void sub_qdvector_dvec(QDVector c, QDVector a, DVector b);
/* c := (qd)a */
void subst_dvector_qdvec(DVector c, QDVector a);
/* c := (d)a */
void subst_qdvector_dvec(QDVector c, DVector a);

// split vector
// ret_vec[0] + ret_vec[1] + ... ret_vec[num_div - 1] = org_vec
// shift holds num_div exponents (may be NULL); see the exponent note above
int split_qdvector_dvec_ex(DVector ret_vec[], long int shift[], int num_div, QDVector org_vec);
int split_qdvector_dvec(DVector ret_vec[], int num_div, QDVector org_vec);
// absmax_row_qdmatrix
void absmax_row_qdmatrix(double mu[QDSIZE], long int *max_j, long int row_index, QDMatrix mat);
/* c := a + (doble)b */
void add_qdmatrix_dmat(QDMatrix c, QDMatrix a, DMatrix b);
/* c := a - (doble)b */
void sub_qdmatrix_dmat(QDMatrix c, QDMatrix a, DMatrix b);
/* c := (d)a */
void subst_qdmatrix_dmat(QDMatrix c, DMatrix a);
/* c := (qd)a */
void subst_dmatrix_qdmat(DMatrix c, QDMatrix a);
// SplitMat_A
// row_shift holds num_div * row_dim exponents (may be NULL)
int split_qdmatrix_dmat_ex(DMatrix ret_mat[], long int row_shift[], int num_div, QDMatrix org_mat);
int split_qdmatrix_dmat(DMatrix ret_mat[], int num_div, QDMatrix org_mat);
// absmax_col_qdmatrix
void absmax_col_qdmatrix(double mu[QDSIZE], long int *max_i, long int col_index, QDMatrix mat);
// SplitMat_B
// col_shift holds num_div * col_dim exponents (may be NULL)
int split_qdmatrix_t_dmat_ex(DMatrix ret_mat[], long int col_shift[], int num_div, QDMatrix org_mat);
int split_qdmatrix_t_dmat(DMatrix ret_mat[], int num_div, QDMatrix org_mat);
// Matrix multiplication based on Ozaki scheme
void mul_qdmatrix_oz(QDMatrix ret, QDMatrix a, int max_num_div_a, QDMatrix b, int max_num_div_b);
// Matrix-Vector multiplication based on Ozaki scheme
void mul_qdmatrix_qdvec_oz(QDVector ret, QDMatrix a, int max_num_div_a, QDVector vb, int max_num_div_vb);

// Fit dimension to be multiple of min_dim
void mul_cqdmatrix_oz_3m(CQDMatrix ret, CQDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CQDMatrix b, int max_num_div_b_real, int max_num_div_b_image);
void mul_cqdmatrix_oz_4m(CQDMatrix ret, CQDMatrix a, int max_num_div_a_real, int max_num_div_a_image, CQDMatrix b, int max_num_div_b_real, int max_num_div_b_image);

#ifdef USE_4M
#define mul_cqdmatrix_oz mul_cqdmatrix_oz_4m
#else // USE_4M
#define mul_cqdmatrix_oz mul_cqdmatrix_oz_3m
#endif // USE_4M


/******************/
/* mpf_oz_scheme.c */
/******************/
//#ifdef USE_MPFLINEAR
#ifdef USE_GMP
// exponent e of x with |x| in [2^(e-1), 2^e); LONG_MIN for zero
long int bnc_oz_mpf_exp2(mpf_srcptr x);
// 2^-shift * x as a double; flushes to zero far below the shift
double bnc_oz_mpf_get_scaled_d(mpf_srcptr x, long int shift);
// ret := 2^shift * value, exactly
void bnc_oz_mpf_set_scaled_d(mpf_ptr ret, double value, long int shift);

// absmax_mpfvector
void absmax_mpfvector(mpf_t ret, long int *max_index, MPFVector vec);
/* c = a + (double)b */
void add_mpfvector_dvec(MPFVector c, MPFVector a, DVector b);
/* c = a - (double)b */
void sub_mpfvector_dvec(MPFVector c, MPFVector a, DVector b);
// extract vector
// ret_vec[0] + ret_vec[1] + ... ret_vec[num_div - 1] = org_vec
// shift holds num_div exponents (may be NULL); see the exponent note above
int split_mpfvector_dvec_ex(DVector ret_vec[], long int shift[], int num_div, MPFVector org_vec);
int split_mpfvector_dvec(DVector ret_vec[], int num_div, MPFVector org_vec);
// absmax_row_mpfmatrix
void absmax_row_mpfmatrix(mpf_t mu, long int *max_j, long int row_index, MPFMatrix mat);
/* c := a + (doble)b */
void add_mpfmatrix_dmat(MPFMatrix c, MPFMatrix a, DMatrix b);
/* c := a - (doble)b */
void sub_mpfmatrix_dmat(MPFMatrix c, MPFMatrix a, DMatrix b);
// SplitMat_A
// row_shift holds num_div * row_dim exponents (may be NULL)
int split_mpfmatrix_dmat_ex(DMatrix ret_mat[], long int row_shift[], int num_div, MPFMatrix org_mat);
int split_mpfmatrix_dmat(DMatrix ret_mat[], int num_div, MPFMatrix org_mat);
// absmax_col_mpfmatrix
void absmax_col_mpfmatrix(mpf_t mu, long int *max_i, long int col_index, MPFMatrix mat);
// SplitMat_B
// col_shift holds num_div * col_dim exponents (may be NULL)
int split_mpfmatrix_t_dmat_ex(DMatrix ret_mat[], long int col_shift[], int num_div, MPFMatrix org_mat);
int split_mpfmatrix_t_dmat(DMatrix ret_mat[], int num_div, MPFMatrix org_mat);
// Matrix multiplication based on Ozaki scheme
void mul_mpfmatrix_oz(MPFMatrix ret, MPFMatrix a, int max_num_div_a, MPFMatrix b, int max_num_div_b);
// Matrix-Vector multiplication based on Ozaki scheme
// (the definition in mpf_oz_scheme.c has always been called *_mpfvec_oz; the
//  _dvec_oz name declared here resolved to nothing)
void mul_mpfmatrix_mpfvec_oz(MPFVector ret, MPFMatrix a, int max_num_div_a, MPFVector vb, int max_num_div_vb);

// default: 3M way
//#define mul_cmpfmatrix_oz mul_cmpfmatrix_oz_4m
#define mul_cmpfmatrix_oz mul_cmpfmatrix_oz_3m

// Matrix multiplication based on Ozaki scheme (4M)
void mul_cmpfmatrix_oz_4m(CMPFMatrix ret, CMPFMatrix a, int max_num_div_a_real, int max_num_div_a_image, CMPFMatrix b, int max_num_div_b_real, int max_num_div_b_image);
// Matrix multiplication based on Ozaki scheme (3M)
void mul_cmpfmatrix_oz_3m(CMPFMatrix ret, CMPFMatrix a, int max_num_div_a_real, int max_num_div_a_image, CMPFMatrix b, int max_num_div_b_real, int max_num_div_b_image);
//#endif // USE_MPFLINEAR
#endif // USE_GMP

#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus

#endif // __BNC_OZ_SCHEME_H
