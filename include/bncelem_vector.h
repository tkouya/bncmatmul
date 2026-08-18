/********************************************************************************/
/* bncelem_vector.h : SIMD-vectorized elementwise elementary functions for      */
/*                    limb-planar DD/TD/QD arrays (exp / log / sin / cos)       */
/*                                                                              */
/* The arrays are limb-planar (SoA), matching the ddvector/tdvector/qdvector    */
/* layout (double *element[DDSIZE] etc.): x[k][i] is limb k of element i.       */
/*                                                                              */
/* The implementation (src/elem_vector.c) is compiled once per SIMD variant     */
/* like every other source in src/, and walks the array in blocks of            */
/* _BNC_D_WIDTH elements: AVX-512 (W = 8), AVX2 (W = 4), NEON/SVE2 (W = 2),     */
/* W = 1 otherwise.  The per-element kernels themselves are the scalar          */
/* bnc_dd_/bnc_td_/bnc_qd_ functions of bncelem_dd.h/_td.h/_qd.h: their         */
/* argument reduction, table lookups and Newton corrections are branchy, so     */
/* unlike the +,-,*,/ EFT kernels they have no lane-uniform SIMD form.  Only    */
/* the limb gather/scatter around them vectorizes.                              */
/*                                                                              */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/* This file is part of BNCmatmul and distributed under the GNU LGPL v3.        */
/********************************************************************************/
#ifndef __BNC_ELEM_VECTOR_H
#define __BNC_ELEM_VECTOR_H

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

/* double-double: x[0..1][0..n-1] -> ret[0..1][0..n-1] */
void bnc_dd_exp_array(long n, double * const ret[2], double * const x[2]);
void bnc_dd_log_array(long n, double * const ret[2], double * const x[2]);
void bnc_dd_sin_array(long n, double * const ret[2], double * const x[2]);
void bnc_dd_cos_array(long n, double * const ret[2], double * const x[2]);

/* triple-double: x[0..2][0..n-1] -> ret[0..2][0..n-1] */
void bnc_td_exp_array(long n, double * const ret[3], double * const x[3]);
void bnc_td_log_array(long n, double * const ret[3], double * const x[3]);
void bnc_td_sin_array(long n, double * const ret[3], double * const x[3]);
void bnc_td_cos_array(long n, double * const ret[3], double * const x[3]);

/* quad-double: x[0..3][0..n-1] -> ret[0..3][0..n-1] */
void bnc_qd_exp_array(long n, double * const ret[4], double * const x[4]);
void bnc_qd_log_array(long n, double * const ret[4], double * const x[4]);
void bnc_qd_sin_array(long n, double * const ret[4], double * const x[4]);
void bnc_qd_cos_array(long n, double * const ret[4], double * const x[4]);

#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif /* __BNC_ELEM_VECTOR_H */
