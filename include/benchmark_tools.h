/********************************************************************************/
/* benchmark_tools.h: Various functions for benchmark                           */
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
#ifndef __BNC_BENCHMARK_TOOLS_H
#define __BNC_BENCHMARK_TOOLS_H

//#include "rdd.h" // [dtq]dfloat and its arithmetic r[dtq]d_*
//#include "matmul_strassen.h"
#include "dlinear.h"
#include "ddlinear.h"
#include "tdlinear.h"
#include "qdlinear.h"
#include "mpflinear.h"

// Sparse Matrix
#include "bncsparse.h"

// -------------------
// benchmark_tools.c
// benchmark_tools_dd.c
// -------------------
// relative errors for double precision vector
void relerr3_dvector(double *max_relerr, double *min_relerr, double *norm_relerr, DVector vec, DVector vec_true, int kind_of_norm);

// relative errors for double precision matrix
void relerr3_dmatrix(double *max_relerr, double *min_relerr, double *norm_relerr, DMatrix mat, DMatrix mat_true, int kind_of_norm);

#ifdef USE_GMP
// relative errors for double precision vector
void relerr3_dvector_mpfvec(double *max_relerr, double *min_relerr, double *norm_relerr, DVector vec, MPFVector vec_true, int kind_of_norm);

// relative errors for double precision matrix
void relerr3_dmatrix_mpfmat(double *max_relerr, double *min_relerr, double *norm_relerr, DMatrix mat, MPFMatrix mat_true, int kind_of_norm);

// relative error// relative errors
// kind_of_norm: 0-infinite norm, 1-one norm, 2 or others-Frobenius norm
void relerr3_cdmatrix_cmpfmat(double *max_abs_relerr, double *min_abs_relerr, double *max_real_relerr, double *min_real_relerr, double *max_image_relerr, double *min_image_relerr, double *norm_relerr, CDMatrix mat, CMPFMatrix mat_true, int kind_of_norm);

// kind_of_norm: 0-infinite norm, 1-one norm, 2 or others-Euclieian norm
void relerr3_cdvector_cmpfvec(double *max_abs_relerr, double *min_abs_relerr, double *max_real_relerr, double *min_real_relerr, double *max_image_relerr, double *min_image_relerr, double *norm_relerr, CDVector vec, CMPFVector vec_true, int kind_of_norm);

#endif // USE_GMP

#ifdef USE_DDLINEAR
	// ANSI C
	// relative errors for DD vector
	void relerr3_ddvector(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double norm_relerr[DDSIZE], DDVector vec, DDVector vec_true, int kind_of_norm);
	// relative errors of complex vector
	void relerr3_cddvector(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double max_real_relerr[DDSIZE], double min_real_relerr[DDSIZE], double max_image_relerr[DDSIZE], double min_image_relerr[DDSIZE], double norm_relerr[DDSIZE], CDDVector vec, CDDVector vec_true, int kind_of_norm);

	// relative errors for DD matrix
	void relerr3_ddmatrix(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double norm_relerr[DDSIZE], DDMatrix mat, DDMatrix mat_true, int kind_of_norm);
	// relative errors for CDD matrix
	void relerr3_cddmatrix(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double max_real_relerr[DDSIZE], double min_real_relerr[DDSIZE], double max_image_relerr[DDSIZE], double min_image_relerr[DDSIZE], double norm_relerr[DDSIZE], CDDMatrix mat, CDDMatrix mat_true, int kind_of_norm);

// 2024-08-02 (Fri) T.Kouya
#ifdef USE_GMP
// relative errors for double precision vector
void relerr3_ddvector_mpfvec(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double norm_relerr[DDSIZE], DDVector vec, MPFVector vec_true, int kind_of_norm);


// relative errors for double precision matrix
void relerr3_ddmatrix_mpfmat(double max_relerr[DDSIZE], double min_relerr[DDSIZE], double norm_relerr[DDSIZE], DDMatrix mat, MPFMatrix mat_true, int kind_of_norm);


// relative errors
// kind_of_norm: 0-infinite norm, 1-one norm, 2 or others-Frobenius norm
void relerr3_cddmatrix_cmpfmat(double max_abs_relerr[DDSIZE], double min_abs_relerr[DDSIZE], double max_real_relerr[DDSIZE], double min_real_relerr[DDSIZE], double max_image_relerr[DDSIZE], double min_image_relerr[DDSIZE], double norm_relerr[DDSIZE], CDDMatrix mat, CMPFMatrix mat_true, int kind_of_norm);


// relative errors
// kind_of_norm: 0-infinite norm, 1-one norm, 2 or others-Euclieian norm
void relerr3_cddvector_cmpfvec(double max_abs_relerr[DDSIZE], double min_abs_relerr[DDSIZE], double max_real_relerr[DDSIZE], double min_real_relerr[DDSIZE], double max_image_relerr[DDSIZE], double min_image_relerr[DDSIZE], double norm_relerr[DDSIZE], CDDVector vec, CMPFVector vec_true, int kind_of_norm);

#endif // USE_GMP
#endif // USE_DDLINEAR

#ifdef USE_TDLINEAR
	void relerr3_tdvector(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double norm_relerr[TDSIZE], TDVector vec, TDVector vec_true, int kind_of_norm);

	// relative errors of complex vector
	void relerr3_ctdvector(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double max_real_relerr[TDSIZE], double min_real_relerr[TDSIZE], double max_image_relerr[TDSIZE], double min_image_relerr[TDSIZE], double norm_relerr[TDSIZE], CTDVector vec, CTDVector vec_true, int kind_of_norm);

	void relerr3_tdmatrix(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double norm_relerr[TDSIZE], TDMatrix mat, TDMatrix mat_true, int kind_of_norm);

	void relerr3_ctdmatrix(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double max_real_relerr[TDSIZE], double min_real_relerr[TDSIZE], double max_image_relerr[TDSIZE], double min_image_relerr[TDSIZE], double norm_relerr[TDSIZE], CTDMatrix mat, CTDMatrix mat_true, int kind_of_norm);

#ifdef USE_GMP
// relative errors for double precision vector
void relerr3_tdvector_mpfvec(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double norm_relerr[TDSIZE], TDVector vec, MPFVector vec_true, int kind_of_norm);

// relative errors for double precision matrix
void relerr3_tdmatrix_mpfmat(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double norm_relerr[TDSIZE], TDMatrix mat, MPFMatrix mat_true, int kind_of_norm);

// kind_of_norm: 0-infinite norm, 1-one norm, 2 or others-Frobenius norm
void relerr3_ctdmatrix_cmpfmat(double max_abs_relerr[TDSIZE], double min_abs_relerr[TDSIZE], double max_real_relerr[TDSIZE], double min_real_relerr[TDSIZE], double max_image_relerr[TDSIZE], double min_image_relerr[TDSIZE], double norm_relerr[TDSIZE], CTDMatrix mat, CMPFMatrix mat_true, int kind_of_norm);

// relative errors
// kind_of_norm: 0-infinite norm, 1-one norm, 2 or others-Euclieian norm
void relerr3_ctdvector_cmpfvec(double max_abs_relerr[TDSIZE], double min_abs_relerr[TDSIZE], double max_real_relerr[TDSIZE], double min_real_relerr[TDSIZE], double max_image_relerr[TDSIZE], double min_image_relerr[TDSIZE], double norm_relerr[TDSIZE], CTDVector vec, CMPFVector vec_true, int kind_of_norm);

#endif // USE_GMP
#endif // USE_TDLINEAR

#ifdef USE_QDLINEAR
#ifdef USE_TDLINEAR
// relative errors of vector using higher precision value
void relerr3_tdvector_qdvec(double max_relerr[TDSIZE], double min_relerr[TDSIZE], double norm_relerr[TDSIZE], TDVector vec, QDVector vec_true, int kind_of_norm);
#endif // USE_TDLINEAR

	void relerr3_qdvector(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double norm_relerr[QDSIZE], QDVector vec, QDVector vec_true, int kind_of_norm);

	// relative errors of complex vector
	void relerr3_cqdvector(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double max_real_relerr[QDSIZE], double min_real_relerr[QDSIZE], double max_image_relerr[QDSIZE], double min_image_relerr[QDSIZE], double norm_relerr[QDSIZE], CQDVector vec, CQDVector vec_true, int kind_of_norm);

	void relerr3_qdmatrix(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double norm_relerr[QDSIZE], QDMatrix mat, QDMatrix mat_true, int kind_of_norm);

	void relerr3_cqdmatrix(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double max_real_relerr[QDSIZE], double min_real_relerr[QDSIZE], double max_image_relerr[QDSIZE], double min_image_relerr[QDSIZE], double norm_relerr[QDSIZE], CQDMatrix mat, CQDMatrix mat_true, int kind_of_norm);

#ifdef USE_GMP
// relative errors for double precision vector
void relerr3_qdvector_mpfvec(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double norm_relerr[QDSIZE], QDVector vec, MPFVector vec_true, int kind_of_norm);

// relative errors for double precision matrix
void relerr3_qdmatrix_mpfmat(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double norm_relerr[QDSIZE], QDMatrix mat, MPFMatrix mat_true, int kind_of_norm);

// relative errors
// kind_of_norm: 0-infinite norm, 1-one norm, 2 or others-Frobenius norm
void relerr3_cqdmatrix_cmpfmat(double max_abs_relerr[QDSIZE], double min_abs_relerr[QDSIZE], double max_real_relerr[QDSIZE], double min_real_relerr[QDSIZE], double max_image_relerr[QDSIZE], double min_image_relerr[QDSIZE], double norm_relerr[QDSIZE], CQDMatrix mat, CMPFMatrix mat_true, int kind_of_norm);

// relative errors
// kind_of_norm: 0-infinite norm, 1-one norm, 2 or others-Euclieian norm
void relerr3_cqdvector_cmpfvec(double max_abs_relerr[QDSIZE], double min_abs_relerr[QDSIZE], double max_real_relerr[QDSIZE], double min_real_relerr[QDSIZE], double max_image_relerr[QDSIZE], double min_image_relerr[QDSIZE], double norm_relerr[QDSIZE], CQDVector vec, CMPFVector vec_true, int kind_of_norm);

#endif // USE_GMP

#endif // USE_QDLINEAR


#ifdef USE_GMP
// relative errors for MPF vector
void relerr3_mpfvector(mpf_t max_relerr, mpf_t min_relerr, mpf_t norm_relerr, MPFVector vec, MPFVector vec_true, int kind_of_norm);

// relative errors for MPF matrix
void relerr3_mpfmatrix(mpf_t max_relerr, mpf_t min_relerr, mpf_t norm_relerr, MPFMatrix mat, MPFMatrix mat_true, int kind_of_norm);

// relative errors for MPC matrix
void relerr3_cmpfmatrix_old(mpf_t max_relerr, mpf_t min_relerr, mpf_t norm_relerr, CMPFMatrix mat, CMPFMatrix mat_true, int kind_of_norm);

// relative errors
// kind_of_norm: 0-infinite norm, 1-one norm, 2 or others-Frobenius norm
void relerr3_cmpfmatrix(mpf_t max_abs_relerr, mpf_t min_abs_relerr, mpf_t max_real_relerr, mpf_t min_real_relerr, mpf_t max_image_relerr, mpf_t min_image_relerr, mpf_t norm_relerr, CMPFMatrix mat, CMPFMatrix mat_true, int kind_of_norm);

// relative errors
// kind_of_norm: 0-infinite norm, 1-one norm, 2 or others-Euclieian norm
void relerr3_cmpfvector(mpf_t max_abs_relerr, mpf_t min_abs_relerr, mpf_t max_real_relerr, mpf_t min_real_relerr, mpf_t max_image_relerr, mpf_t min_image_relerr, mpf_t norm_relerr, CMPFVector vec, CMPFVector vec_true, int kind_of_norm);

#endif // USE_GMP

// 2024-08-02 (Fri) T.Kouya
#ifdef USE_GMP
#if 0
// relative errors for double precision vector
static inline void relerr3_dvector_mpfvec(double *max_relerr, double *min_relerr, double *norm_relerr, DVector vec, MPFVector vec_true, int kind_of_norm)
{
    unsigned long prec;
    mpf_t in_max_relerr, in_min_relerr, in_norm_relerr;
    MPFVector in_vec;

    // Initialize
    prec = vec_true->prec;
    mpf_init2(in_max_relerr, prec);
    mpf_init2(in_min_relerr, prec);
    mpf_init2(in_norm_relerr, prec);
    in_vec = init2_mpfvector(vec->dim, prec);

    // (mpf_t)in_vec := vec
    subst_mpfvector_dvec(in_vec, vec);
    relerr3_mpfvector(in_max_relerr, in_min_relerr, in_norm_relerr, in_vec, vec_true, kind_of_norm);

    *max_relerr = mpf_get_d(in_max_relerr);
    *min_relerr = mpf_get_d(in_min_relerr);
    *norm_relerr = mpf_get_d(in_norm_relerr);

    // Free
    mpf_clear(in_max_relerr);
    mpf_clear(in_min_relerr);
    mpf_clear(in_norm_relerr);
    free_mpfvector(in_vec);
}

// relative errors for double precision matrix
static inline void relerr3_dmatrix_mpfmat(double *max_relerr, double *min_relerr, double *norm_relerr, DMatrix mat, MPFMatrix mat_true, int kind_of_norm)
{
    unsigned long prec;
    mpf_t in_max_relerr, in_min_relerr, in_norm_relerr;
    MPFMatrix in_mat;

    // Initialize
    prec = mat_true->prec;
    mpf_init2(in_max_relerr, prec);
    mpf_init2(in_min_relerr, prec);
    mpf_init2(in_norm_relerr, prec);
    in_mat = init2_mpfmatrix(mat->row_dim, mat->col_dim, prec);

    // (mpf_t)in_vec := vec
    subst_mpfmatrix_dmat(in_mat, mat);
    relerr3_mpfmatrix(in_max_relerr, in_min_relerr, in_norm_relerr, in_mat, mat_true, kind_of_norm);

    *max_relerr = mpf_get_d(in_max_relerr);
    *min_relerr = mpf_get_d(in_min_relerr);
    *norm_relerr = mpf_get_d(in_norm_relerr);

    // Free
    mpf_clear(in_max_relerr);
    mpf_clear(in_min_relerr);
    mpf_clear(in_norm_relerr);
    free_mpfmatrix(in_mat);
}


// relative errors
// kind_of_norm: 0-infinite norm, 1-one norm, 2 or others-Frobenius norm
static inline void relerr3_cdmatrix_cmpfmat(double *max_abs_relerr, double *min_abs_relerr, double *max_real_relerr, double *min_real_relerr, double *max_image_relerr, double *min_image_relerr, double *norm_relerr, CDMatrix mat, CMPFMatrix mat_true, int kind_of_norm)
{
    unsigned long prec;
    mpf_t in_max_abs_relerr, in_min_abs_relerr, in_max_real_relerr, in_min_real_relerr, in_max_image_relerr, in_min_image_relerr, in_norm_relerr;
    CMPFMatrix in_mat;

    // Initialize
    prec = mat_true->prec;
    mpf_init2(in_max_abs_relerr, prec);
    mpf_init2(in_min_abs_relerr, prec);
    mpf_init2(in_max_real_relerr, prec);
    mpf_init2(in_min_real_relerr, prec);
    mpf_init2(in_max_image_relerr, prec);
    mpf_init2(in_min_image_relerr, prec);
    mpf_init2(in_norm_relerr, prec);
    in_mat = init2_cmpfmatrix(mat->row_dim, mat->col_dim, prec);

    // (mpf_t)in_vec := vec
    subst_cmpfmatrix_cdmat(in_mat, mat);
    relerr3_cmpfmatrix(in_max_abs_relerr, in_min_abs_relerr, in_max_real_relerr, in_min_real_relerr, in_max_image_relerr, in_min_image_relerr, in_norm_relerr, in_mat, mat_true, kind_of_norm);

    *max_abs_relerr = mpf_get_d(in_max_abs_relerr);
    *min_abs_relerr = mpf_get_d(in_min_abs_relerr);
    *max_real_relerr = mpf_get_d(in_max_real_relerr);
    *min_real_relerr = mpf_get_d(in_min_real_relerr);
    *max_image_relerr = mpf_get_d(in_max_image_relerr);
    *min_image_relerr = mpf_get_d(in_min_image_relerr);
    *norm_relerr = mpf_get_d(in_norm_relerr);

    // Free
    mpf_clear(in_max_abs_relerr);
    mpf_clear(in_min_abs_relerr);
    mpf_clear(in_max_real_relerr);
    mpf_clear(in_min_real_relerr);
    mpf_clear(in_max_image_relerr);
    mpf_clear(in_min_image_relerr);
    mpf_clear(in_norm_relerr);
    free_cmpfmatrix(in_mat);
}

// relative errors
// kind_of_norm: 0-infinite norm, 1-one norm, 2 or others-Euclieian norm
static inline void relerr3_cdvector_cmpfvec(double *max_abs_relerr, double *min_abs_relerr, double *max_real_relerr, double *min_real_relerr, double *max_image_relerr, double *min_image_relerr, double *norm_relerr, CDVector vec, CMPFVector vec_true, int kind_of_norm)
{
    unsigned long prec;
    mpf_t in_max_abs_relerr, in_min_abs_relerr, in_max_real_relerr, in_min_real_relerr, in_max_image_relerr, in_min_image_relerr, in_norm_relerr;
    CMPFVector in_vec;

    // Initialize
    prec = vec_true->prec;
    mpf_init2(in_max_abs_relerr, prec);
    mpf_init2(in_min_abs_relerr, prec);
    mpf_init2(in_max_real_relerr, prec);
    mpf_init2(in_min_real_relerr, prec);
    mpf_init2(in_max_image_relerr, prec);
    mpf_init2(in_min_image_relerr, prec);
    mpf_init2(in_norm_relerr, prec);
    in_vec = init2_cmpfvector(vec->dim, prec);

    // (mpf_t)in_vec := vec
    subst_cmpfvector_cdvec(in_vec, vec);
    relerr3_cmpfvector(in_max_abs_relerr, in_min_abs_relerr, in_max_real_relerr, in_min_real_relerr, in_max_image_relerr, in_min_image_relerr, in_norm_relerr, in_vec, vec_true, kind_of_norm);

    *max_abs_relerr = mpf_get_d(in_max_abs_relerr);
    *min_abs_relerr = mpf_get_d(in_min_abs_relerr);
    *max_real_relerr = mpf_get_d(in_max_real_relerr);
    *min_real_relerr = mpf_get_d(in_min_real_relerr);
    *max_image_relerr = mpf_get_d(in_max_image_relerr);
    *min_image_relerr = mpf_get_d(in_min_image_relerr);
    *norm_relerr = mpf_get_d(in_norm_relerr);

    // Free
    mpf_clear(in_max_abs_relerr);
    mpf_clear(in_min_abs_relerr);
    mpf_clear(in_max_real_relerr);
    mpf_clear(in_min_real_relerr);
    mpf_clear(in_max_image_relerr);
    mpf_clear(in_min_image_relerr);
    mpf_clear(in_norm_relerr);
    free_cmpfvector(in_vec);
}
#endif // 0
#endif // USE_GMP

#endif // __BNC_BENCHMARK_TOOLS_H
