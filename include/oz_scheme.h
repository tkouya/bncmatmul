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
int split_dvector_dvec(DVector ret_vec[], int num_div, DVector org_vec);

// SplitMat_A
// return real_num_div
int split_dmatrix_dmat(DMatrix ret_mat[], int num_div, DMatrix org_mat);

// SplitMat_B
// return real_num_div
int split_dmatrix_t_dmat(DMatrix ret_mat[], int num_div, DMatrix org_mat);

/******************/
/* dd_oz_scheme.c */
/******************/
#ifdef USE_DDLINEAR
// absmax_ddvector
void absmax_ddvector(double ret[DDSIZE], long int *max_index, DDVector vec);
/* c = a + (double)b */
void add_ddvector_dvec(DDVector c, DDVector a, DVector b);
/* c = a - (double)b */
void sub_ddvector_dvec(DDVector c, DDVector a, DVector b);
// split vector
// ret_vec[0] + ret_vec[1] + ... ret_vec[num_div - 1] = org_vec
int split_ddvector_dvec(DVector ret_vec[], int num_div, DDVector org_vec);
// absmax_row_ddmatrix
void absmax_row_ddmatrix(double mu[DDSIZE], long int *max_j, long int row_index, DDMatrix mat);
/* c := a + (doble)b */
void add_ddmatrix_dmat(DDMatrix c, DDMatrix a, DMatrix b);
/* c := a - (doble)b */
void sub_ddmatrix_dmat(DDMatrix c, DDMatrix a, DMatrix b);
// SplitMat_A
// return real_num_div
int split_ddmatrix_dmat(DMatrix ret_mat[], int num_div, DDMatrix org_mat);
// absmax_col_ddmatrix
void absmax_col_ddmatrix(double mu[DDSIZE], long int *max_i, long int col_index, DDMatrix mat);
// SplitMat_B
// return real_num_div
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

#endif // USE_DDLINEAR

/******************/
/* td_oz_scheme.c */
/******************/
#ifdef USE_TDLINEAR
// absmax_tdvector
void absmax_tdvector(double ret[TDSIZE], long int *max_index, TDVector vec);
/* c = a + (double)b */
void add_tdvector_dvec(TDVector c, TDVector a, DVector b);
/* c = a - (double)b */
void sub_tdvector_dvec(TDVector c, TDVector a, DVector b);
// extract vector
// ret_vec[0] + ret_vec[1] + ... ret_vec[num_div - 1] = org_vec
int split_tdvector_dvec(DVector ret_vec[], int num_div, TDVector org_vec);
// absmax_row_tdmatrix
void absmax_row_tdmatrix(double mu[TDSIZE], long int *max_j, long int row_index, TDMatrix mat);
/* c := a + (doble)b */
void add_tdmatrix_dmat(TDMatrix c, TDMatrix a, DMatrix b);
/* c := a - (doble)b */
void sub_tdmatrix_dmat(TDMatrix c, TDMatrix a, DMatrix b);
// SplitMat_A
int split_tdmatrix_dmat(DMatrix ret_mat[], int num_div, TDMatrix org_mat);
// absmax_col_tdmatrix
void absmax_col_tdmatrix(double mu[TDSIZE], long int *max_i, long int col_index, TDMatrix mat);
// SplitMat_B
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

#endif // USE_TDLINEAR

/******************/
/* qd_oz_scheme.c */
/******************/
#ifdef USE_QDLINEAR
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
int split_qdmatrix_dmat(DMatrix ret_mat[], int num_div, QDMatrix org_mat);
// absmax_col_qdmatrix
void absmax_col_qdmatrix(double mu[QDSIZE], long int *max_i, long int col_index, QDMatrix mat);
// SplitMat_B
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

#endif // USE_QDLINEAR

/******************/
/* mpf_oz_scheme.c */
/******************/
//#ifdef USE_MPFLINEAR
#ifdef USE_GMP
// absmax_mpfvector
void absmax_mpfvector(mpf_t ret, long int *max_index, MPFVector vec);
/* c = a + (double)b */
void add_mpfvector_dvec(MPFVector c, MPFVector a, DVector b);
/* c = a - (double)b */
void sub_mpfvector_dvec(MPFVector c, MPFVector a, DVector b);
// extract vector
// ret_vec[0] + ret_vec[1] + ... ret_vec[num_div - 1] = org_vec
int split_mpfvector_dvec(DVector ret_vec[], int num_div, MPFVector org_vec);
// absmax_row_mpfmatrix
void absmax_row_mpfmatrix(mpf_t mu, long int *max_j, long int row_index, MPFMatrix mat);
/* c := a + (doble)b */
void add_mpfmatrix_dmat(MPFMatrix c, MPFMatrix a, DMatrix b);
/* c := a - (doble)b */
void sub_mpfmatrix_dmat(MPFMatrix c, MPFMatrix a, DMatrix b);
// SplitMat_A
int split_mpfmatrix_dmat(DMatrix ret_mat[], int num_div, MPFMatrix org_mat);
// absmax_col_mpfmatrix
void absmax_col_mpfmatrix(mpf_t mu, long int *max_i, long int col_index, MPFMatrix mat);
// SplitMat_B
int split_mpfmatrix_t_dmat(DMatrix ret_mat[], int num_div, MPFMatrix org_mat);
// Matrix multiplication based on Ozaki scheme
void mul_mpfmatrix_oz(MPFMatrix ret, MPFMatrix a, int max_num_div_a, MPFMatrix b, int max_num_div_b);
// Matrix-Vector multiplication based on Ozaki scheme
void mul_mpfmatrix_dvec_oz(MPFVector ret, MPFMatrix a, int max_num_div_a, MPFVector vb, int max_num_div_vb);

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
