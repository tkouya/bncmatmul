/********************************************************************************/
/* rdtq_func.c : exported (non-inline) function versions of the DD/TD/QD and    */
/*               DS/TS/QS FMA and elementary-function interfaces                */
/*                                                                              */
/* See include/rdtq_func.h for the interface.  Everything here is a thin        */
/* wrapper that delegates to the static-inline kernels of rdd.h / rds.h and     */
/* the dtq-0.0.3 ports (bncfma_d.h / bncfma_f.h / bncelem*.h); the point of the */
/* file is to give the shared library real, callable symbols for consumers      */
/* that cannot see C headers (Python/ctypes, Fortran, other languages).         */
/*                                                                              */
/* Two conventions differ between the two worlds and are reconciled here:       */
/*   - rdtq_func.h puts the result FIRST   : rdd_exp(ret, x)                    */
/*   - the bnc_* / c_* kernels put it LAST : bnc_dd_exp(x, ret)                 */
/*                                                                              */
/* USE_R??_FUNCTIONS is defined before rdd.h / rds.h so that those headers      */
/* suppress their lowercase macro aliases (rdd_add -> RDD_ADD, ...); the        */
/* uppercase RDD_* / RDS_* macros stay available, so rdd_add() & co. below      */
/* still                                                                        */
/* follow whatever kernel the build-time flags (USE_DD_BF, USE_ACCURATE_DD_DIV, */
/* BNC_USE_FMA_DIV, ...) select -- exactly like inlined library code does.      */
/*                                                                              */
/* rdd.h / rds.h additionally define `static inline rdd_fma` (and rtd/rqd/      */
/* rds/rts/rqs).  A static inline that is never called is never emitted, so it  */
/* cannot serve as the exported symbol; the macro renames below push those      */
/* inlines aside under a *_inline suffix and leave the plain names free for the */
/* external definitions given here, which use the certified branch-free DW/TW/  */
/* QW FMA (arXiv:2607.11391) as documented in rdtq_func.h.                      */
/*                                                                              */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/* This file is part of BNCmatmul and distributed under the GNU LGPL v3.        */
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* keep the lowercase macro aliases of rdd.h / rds.h out of the way */
#define USE_RDD_FUNCTIONS
#define USE_RTD_FUNCTIONS
#define USE_RQD_FUNCTIONS
#define USE_RDS_FUNCTIONS
#define USE_RTS_FUNCTIONS
#define USE_RQS_FUNCTIONS

/* free up the rXX_fma names (see the header comment above) */
#define rdd_fma rdd_fma_inline
#define rtd_fma rtd_fma_inline
#define rqd_fma rqd_fma_inline
#define rds_fma rds_fma_inline
#define rts_fma rts_fma_inline
#define rqs_fma rqs_fma_inline

#include "rdd.h"
#include "rds.h"

#undef rdd_fma
#undef rtd_fma
#undef rqd_fma
#undef rds_fma
#undef rts_fma
#undef rqs_fma

#include "bncelem.h"		/* bnc_dd_* / bnc_td_* / bnc_qd_* */
#include "bncelem_f.h"		/* bnc_ds_* / bnc_ts_* / bnc_qs_* */
#include "bncfma_d.h"		/* bnc_dwfma / bnc_twfma / bnc_qwfma (+ _safe) */
#include "bncfma_f.h"		/* bnc_dwfmaf / bnc_twfmaf / bnc_qwfmaf (+ _safe) */

#include "rdtq_func.h"

/* ============================================================================ */
/* DD (double-double, double[2])                                                */
/* ============================================================================ */

void rdd_fma(double *ret, const double *a, const double *b, const double *c)
{
	bnc_dwfma(ret, a, b, c);
}

void rdd_fma_safe(double *ret, const double *a, double b, const double *c)
{
	bnc_dwfma_safe(ret, a, b, c);
}

void rdd_div_fma(double *ret, const double *a, const double *b)
{
	bnc_dd_div_fma(a, b, ret);
}

void rdd_exp(double *ret, const double *x)		{ bnc_dd_exp(x, ret); }
void rdd_expm1(double *ret, const double *x)		{ bnc_dd_expm1(x, ret); }
void rdd_log(double *ret, const double *x)		{ bnc_dd_log(x, ret); }
void rdd_log10(double *ret, const double *x)		{ bnc_dd_log10(x, ret); }
void rdd_sin(double *ret, const double *x)		{ bnc_dd_sin(x, ret); }
void rdd_cos(double *ret, const double *x)		{ bnc_dd_cos(x, ret); }
void rdd_tan(double *ret, const double *x)		{ c_dd_tan(x, ret); }
void rdd_nint(double *ret, const double *x)		{ bnc_dd_nint(x, ret); }

void rdd_sincos(double *sin_ret, double *cos_ret, const double *x)
{
	bnc_dd_sincos(x, sin_ret, cos_ret);
}

/* ============================================================================ */
/* TD (triple-double, double[3])                                                */
/* ============================================================================ */

void rtd_fma(double *ret, const double *a, const double *b, const double *c)
{
	bnc_twfma(ret, a, b, c);
}

void rtd_fma_safe(double *ret, const double *a, double b, const double *c)
{
	bnc_twfma_safe(ret, a, b, c);
}

void rtd_div_fma(double *ret, const double *a, const double *b)
{
	bnc_td_div_fma(a, b, ret);
}

void rtd_exp(double *ret, const double *x)		{ bnc_td_exp(x, ret); }
void rtd_expm1(double *ret, const double *x)		{ bnc_td_expm1(x, ret); }
void rtd_log(double *ret, const double *x)		{ bnc_td_log(x, ret); }
void rtd_log10(double *ret, const double *x)		{ bnc_td_log10(x, ret); }
void rtd_sin(double *ret, const double *x)		{ bnc_td_sin(x, ret); }
void rtd_cos(double *ret, const double *x)		{ bnc_td_cos(x, ret); }
void rtd_tan(double *ret, const double *x)		{ c_td_tan(x, ret); }
void rtd_nint(double *ret, const double *x)		{ bnc_td_nint(x, ret); }

void rtd_sincos(double *sin_ret, double *cos_ret, const double *x)
{
	bnc_td_sincos(x, sin_ret, cos_ret);
}

/* ============================================================================ */
/* QD (quadruple-double, double[4])                                             */
/* ============================================================================ */

void rqd_fma(double *ret, const double *a, const double *b, const double *c)
{
	bnc_qwfma(ret, a, b, c);
}

void rqd_fma_safe(double *ret, const double *a, double b, const double *c)
{
	bnc_qwfma_safe(ret, a, b, c);
}

void rqd_div_fma(double *ret, const double *a, const double *b)
{
	bnc_qd_div_fma(a, b, ret);
}

void rqd_exp(double *ret, const double *x)		{ bnc_qd_exp(x, ret); }
void rqd_expm1(double *ret, const double *x)		{ bnc_qd_expm1(x, ret); }
void rqd_log(double *ret, const double *x)		{ bnc_qd_log(x, ret); }
void rqd_log10(double *ret, const double *x)		{ bnc_qd_log10(x, ret); }
void rqd_sin(double *ret, const double *x)		{ bnc_qd_sin(x, ret); }
void rqd_cos(double *ret, const double *x)		{ bnc_qd_cos(x, ret); }
void rqd_tan(double *ret, const double *x)		{ c_qd_tan(x, ret); }
void rqd_nint(double *ret, const double *x)		{ c_qd_nint(x, ret); }

void rqd_sincos(double *sin_ret, double *cos_ret, const double *x)
{
	bnc_qd_sincos(x, sin_ret, cos_ret);
}

/* ============================================================================ */
/* DS (double-single, float[2])                                                 */
/* ============================================================================ */

void rds_fma(float *ret, const float *a, const float *b, const float *c)
{
	bnc_dwfmaf(ret, a, b, c);
}

void rds_fma_safe(float *ret, const float *a, float b, const float *c)
{
	bnc_dwfmaf_safe(ret, a, b, c);
}

void rds_div_fma(float *ret, const float *a, const float *b)
{
	bnc_ds_div_fma(a, b, ret);
}

void rds_exp(float *ret, const float *x)		{ bnc_ds_exp(x, ret); }
void rds_expm1(float *ret, const float *x)		{ bnc_ds_expm1(x, ret); }
void rds_log(float *ret, const float *x)		{ bnc_ds_log(x, ret); }
void rds_log10(float *ret, const float *x)		{ bnc_ds_log10(x, ret); }
void rds_sin(float *ret, const float *x)		{ bnc_ds_sin(x, ret); }
void rds_cos(float *ret, const float *x)		{ bnc_ds_cos(x, ret); }

void rds_sincos(float *sin_ret, float *cos_ret, const float *x)
{
	bnc_ds_sincos(x, sin_ret, cos_ret);
}

/* tan = sin / cos, evaluated from the single reduced-argument sincos */
void rds_tan(float *ret, const float *x)
{
	float s[DSSIZE], c[DSSIZE];

	bnc_ds_sincos(x, s, c);
	RDS_DIV(ret, s, c);
}

/* ============================================================================ */
/* TS (triple-single, float[3])                                                 */
/* ============================================================================ */

void rts_fma(float *ret, const float *a, const float *b, const float *c)
{
	bnc_twfmaf(ret, a, b, c);
}

void rts_fma_safe(float *ret, const float *a, float b, const float *c)
{
	bnc_twfmaf_safe(ret, a, b, c);
}

void rts_div_fma(float *ret, const float *a, const float *b)
{
	bnc_ts_div_fma(a, b, ret);
}

void rts_exp(float *ret, const float *x)		{ bnc_ts_exp(x, ret); }
void rts_expm1(float *ret, const float *x)		{ bnc_ts_expm1(x, ret); }
void rts_log(float *ret, const float *x)		{ bnc_ts_log(x, ret); }
void rts_log10(float *ret, const float *x)		{ bnc_ts_log10(x, ret); }
void rts_sin(float *ret, const float *x)		{ bnc_ts_sin(x, ret); }
void rts_cos(float *ret, const float *x)		{ bnc_ts_cos(x, ret); }

void rts_sincos(float *sin_ret, float *cos_ret, const float *x)
{
	bnc_ts_sincos(x, sin_ret, cos_ret);
}

void rts_tan(float *ret, const float *x)
{
	float s[TSSIZE], c[TSSIZE];

	bnc_ts_sincos(x, s, c);
	RTS_DIV(ret, s, c);
}

/* ============================================================================ */
/* QS (quadruple-single, float[4])                                              */
/* ============================================================================ */

void rqs_fma(float *ret, const float *a, const float *b, const float *c)
{
	bnc_qwfmaf(ret, a, b, c);
}

void rqs_fma_safe(float *ret, const float *a, float b, const float *c)
{
	bnc_qwfmaf_safe(ret, a, b, c);
}

void rqs_div_fma(float *ret, const float *a, const float *b)
{
	bnc_qs_div_fma(a, b, ret);
}

void rqs_exp(float *ret, const float *x)		{ bnc_qs_exp(x, ret); }
void rqs_expm1(float *ret, const float *x)		{ bnc_qs_expm1(x, ret); }
void rqs_log(float *ret, const float *x)		{ bnc_qs_log(x, ret); }
void rqs_log10(float *ret, const float *x)		{ bnc_qs_log10(x, ret); }
void rqs_sin(float *ret, const float *x)		{ bnc_qs_sin(x, ret); }
void rqs_cos(float *ret, const float *x)		{ bnc_qs_cos(x, ret); }
void rqs_tan(float *ret, const float *x)		{ c_qs_tan(x, ret); }

void rqs_sincos(float *sin_ret, float *cos_ret, const float *x)
{
	bnc_qs_sincos(x, sin_ret, cos_ret);
}

/* ============================================================================ */
/* Basic arithmetic -- routed through the RXX_* macros so that the build-time   */
/* kernel selection (USE_DD_BF, USE_ACCURATE_DD_ADD/DIV, BNC_USE_FMA_DIV, ...)  */
/* applies to these exported symbols exactly as it does to inlined callers.     */
/*                                                                              */
/* The c_* kernels behind those macros take plain `double *` / `float *` for    */
/* their read-only operands, so the const of the public prototypes is cast      */
/* away on the way in.  None of them writes through those arguments.            */
/* ============================================================================ */

#define D_(p) ((double *)(p))
#define F_(p) ((float *)(p))

void rdd_add(double *ret, const double *a, const double *b)	{ RDD_ADD(ret, D_(a), D_(b)); }
void rdd_sub(double *ret, const double *a, const double *b)	{ RDD_SUB(ret, D_(a), D_(b)); }
void rdd_mul(double *ret, const double *a, const double *b)	{ RDD_MUL(ret, D_(a), D_(b)); }
void rdd_div(double *ret, const double *a, const double *b)	{ RDD_DIV(ret, D_(a), D_(b)); }
void rdd_sqrt(double *ret, const double *a)			{ RDD_SQRT(ret, D_(a)); }

void rtd_add(double *ret, const double *a, const double *b)	{ RTD_ADD(ret, D_(a), D_(b)); }
void rtd_sub(double *ret, const double *a, const double *b)	{ RTD_SUB(ret, D_(a), D_(b)); }
void rtd_mul(double *ret, const double *a, const double *b)	{ RTD_MUL(ret, D_(a), D_(b)); }
void rtd_div(double *ret, const double *a, const double *b)	{ RTD_DIV(ret, D_(a), D_(b)); }
void rtd_sqrt(double *ret, const double *a)			{ RTD_SQRT(ret, D_(a)); }

void rqd_add(double *ret, const double *a, const double *b)	{ RQD_ADD(ret, D_(a), D_(b)); }
void rqd_sub(double *ret, const double *a, const double *b)	{ RQD_SUB(ret, D_(a), D_(b)); }
void rqd_mul(double *ret, const double *a, const double *b)	{ RQD_MUL(ret, D_(a), D_(b)); }
void rqd_div(double *ret, const double *a, const double *b)	{ RQD_DIV(ret, D_(a), D_(b)); }
void rqd_sqrt(double *ret, const double *a)			{ RQD_SQRT(ret, D_(a)); }

void rds_add(float *ret, const float *a, const float *b)	{ RDS_ADD(ret, F_(a), F_(b)); }
void rds_sub(float *ret, const float *a, const float *b)	{ RDS_SUB(ret, F_(a), F_(b)); }
void rds_mul(float *ret, const float *a, const float *b)	{ RDS_MUL(ret, F_(a), F_(b)); }
void rds_div(float *ret, const float *a, const float *b)	{ RDS_DIV(ret, F_(a), F_(b)); }
void rds_sqrt(float *ret, const float *a)			{ RDS_SQRT(ret, F_(a)); }

void rts_add(float *ret, const float *a, const float *b)	{ RTS_ADD(ret, F_(a), F_(b)); }
void rts_sub(float *ret, const float *a, const float *b)	{ RTS_SUB(ret, F_(a), F_(b)); }
void rts_mul(float *ret, const float *a, const float *b)	{ RTS_MUL(ret, F_(a), F_(b)); }
void rts_div(float *ret, const float *a, const float *b)	{ RTS_DIV(ret, F_(a), F_(b)); }
void rts_sqrt(float *ret, const float *a)			{ RTS_SQRT(ret, F_(a)); }

void rqs_add(float *ret, const float *a, const float *b)	{ RQS_ADD(ret, F_(a), F_(b)); }
void rqs_sub(float *ret, const float *a, const float *b)	{ RQS_SUB(ret, F_(a), F_(b)); }
void rqs_mul(float *ret, const float *a, const float *b)	{ RQS_MUL(ret, F_(a), F_(b)); }
void rqs_div(float *ret, const float *a, const float *b)	{ RQS_DIV(ret, F_(a), F_(b)); }
void rqs_sqrt(float *ret, const float *a)			{ RQS_SQRT(ret, F_(a)); }
