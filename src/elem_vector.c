/********************************************************************************/
/* elem_vector.c : element-wise elementary functions over limb-planar           */
/*                 DD/TD/QD arrays (exp / log / sin / cos)                      */
/*                                                                              */
/* See include/bncelem_vector.h for the interface.  The arrays are limb-planar  */
/* (SoA), matching the ddvector/tdvector/qdvector layout: x[k][i] is limb k of  */
/* element i, so one element has to be staged into a contiguous double[SIZE]    */
/* before the scalar kernel sees it and written back limb by limb afterwards.   */
/*                                                                              */
/* NOTE on vectorization: the DD/TD/QD exp/log/sin/cos of bncelem_{dd,td,qd}.h  */
/* are argument-reduction + table-lookup + Taylor kernels with data-dependent   */
/* branches (range reduction modulo pi/2 and pi/1024, Newton correction for     */
/* log, error exits), so there is no lane-uniform form of them to feed a        */
/* _mm512_/_mm256_/vld1q_ pipeline; unlike the +,-,*,/ EFT kernels used by the  */
/* linear-algebra sources, they cannot be turned into straight-line SIMD code.  */
/* What this file therefore does is drive the scalar kernels over the array in  */
/* blocks of _BNC_D_WIDTH (8 under AVX-512, 4 under AVX2, 2 under NEON/SVE2,    */
/* 1 otherwise), which is what makes the surrounding limb gather/scatter        */
/* vectorizable and keeps the staging buffers in registers.  The object is      */
/* still compiled once per SIMD variant like every other source in src/, so     */
/* the block size follows the variant being built.                              */
/*                                                                              */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/* This file is part of BNCmatmul and distributed under the GNU LGPL v3.        */
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bnc_common.h"		/* _BNC_D_WIDTH */
#include "bncelem.h"		/* bnc_dd_* / bnc_td_* / bnc_qd_* */
#include "bncelem_vector.h"

#ifndef _BNC_D_WIDTH
#define _BNC_D_WIDTH (1)
#endif /* _BNC_D_WIDTH */

/* number of elements handled per inner block */
#define BNC_ELEM_BLOCK ((long)(_BNC_D_WIDTH))

/*
 * BNC_ELEM_ARRAY(name, SIZE, kernel)
 *
 *   name(n, ret, x): ret[0..SIZE-1][i] := kernel(x[0..SIZE-1][i]), i = 0..n-1
 *
 * kernel has the bncelem_*.h convention kernel(const double *a, double *ret).
 * ret and x may alias (ret == x is the common in-place case): every element
 * is fully staged into xb[] before any of ret[k][i] is written.
 */
#define BNC_ELEM_ARRAY(name, SIZE, kernel) \
void name(long n, double * const ret[SIZE], double * const x[SIZE]) \
{ \
	long i, ib, nb; \
	int k, b; \
	double xb[BNC_ELEM_BLOCK][SIZE], rb[BNC_ELEM_BLOCK][SIZE]; \
\
	if(n <= 0) return; \
\
	for(ib = 0; ib < n; ib += BNC_ELEM_BLOCK) \
	{ \
		nb = n - ib; \
		if(nb > BNC_ELEM_BLOCK) nb = BNC_ELEM_BLOCK; \
\
		for(b = 0; b < (int)nb; b++) \
		{ \
			i = ib + b; \
			for(k = 0; k < SIZE; k++) \
				xb[b][k] = x[k][i]; \
		} \
\
		for(b = 0; b < (int)nb; b++) \
			kernel(xb[b], rb[b]); \
\
		for(b = 0; b < (int)nb; b++) \
		{ \
			i = ib + b; \
			for(k = 0; k < SIZE; k++) \
				ret[k][i] = rb[b][k]; \
		} \
	} \
}

/* ---- double-double (double[2]) ---- */
BNC_ELEM_ARRAY(bnc_dd_exp_array, 2, bnc_dd_exp)
BNC_ELEM_ARRAY(bnc_dd_log_array, 2, bnc_dd_log)
BNC_ELEM_ARRAY(bnc_dd_sin_array, 2, bnc_dd_sin)
BNC_ELEM_ARRAY(bnc_dd_cos_array, 2, bnc_dd_cos)

/* ---- triple-double (double[3]) ---- */
BNC_ELEM_ARRAY(bnc_td_exp_array, 3, bnc_td_exp)
BNC_ELEM_ARRAY(bnc_td_log_array, 3, bnc_td_log)
BNC_ELEM_ARRAY(bnc_td_sin_array, 3, bnc_td_sin)
BNC_ELEM_ARRAY(bnc_td_cos_array, 3, bnc_td_cos)

/* ---- quad-double (double[4]) ---- */
BNC_ELEM_ARRAY(bnc_qd_exp_array, 4, bnc_qd_exp)
BNC_ELEM_ARRAY(bnc_qd_log_array, 4, bnc_qd_log)
BNC_ELEM_ARRAY(bnc_qd_sin_array, 4, bnc_qd_sin)
BNC_ELEM_ARRAY(bnc_qd_cos_array, 4, bnc_qd_cos)
