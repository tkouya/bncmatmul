/********************************************************************************/
/* bncelem_f.h : DS/TS/QS (float-based) elementary functions in plain C         */
/*               exp / expm1 / log / log10 / sin / cos / sincos                 */
/*                                                                              */
/* Port of the dtq-0.0.3 strategy for the float-based classes: each function    */
/* delegates to the corresponding double-based one (DS -> DD, TS -> TD,         */
/* QS -> QD).  Float limbs fit exactly in doubles, so the forward conversion    */
/* is exact; the result is split back limb-by-limb with an exact residual       */
/* subtraction (same as dtq's ds_from_dd / ts_from_td / qs_from_qd helpers).    */
/*                                                                              */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/* This file is part of BNCmatmul and distributed under the GNU LGPL v3.        */
/********************************************************************************/
#ifndef __BNC_ELEM_F_H
#define __BNC_ELEM_F_H

#include <math.h>

#include "rds.h"       /* c_ds_qs.h was absorbed into rds.h */	/* float EFTs (ftwo_sum, fquick_two_sum, ...) */
#include "bncelem.h"	/* bnc_dd_* / bnc_td_* / bnc_qd_* */
#include "bncfma_f.h"	/* bnc_dwfmaf_safe / bnc_twfmaf_safe / bnc_qwfmaf_safe */

/* ---- conversions (dtq-0.0.3 src/{ds,ts,qs}_real.cpp helpers) ---- */

/* float limbs are exactly representable in double: just place them */
static inline void bnc_dd_from_ds(const float *a, double *dd)
{
	dd[0] = (double)a[0];
	dd[1] = (double)a[1];
}

static inline void bnc_td_from_ts(const float *a, double *td)
{
	td[0] = (double)a[0];
	td[1] = (double)a[1];
	td[2] = (double)a[2];
}

static inline void bnc_qd_from_qs(const float *a, double *qd)
{
	qd[0] = (double)a[0];
	qd[1] = (double)a[1];
	qd[2] = (double)a[2];
	qd[3] = (double)a[3];
}

/* split a double-based value back into float limbs; the residual
   subtractions are exact in the double-based arithmetic */
static inline void bnc_ds_from_dd(const double *dd, float *a)
{
	double rem[2];

	a[0] = (float)dd[0];
	c_dd_sub_dd_d(dd, (double)a[0], rem);
	a[1] = (float)rem[0];
}

static inline void bnc_ts_from_td(const double *td, float *a)
{
	double rem[3];

	a[0] = (float)td[0];
	c_td_sub_td_d((double *)td, (double)a[0], rem);
	a[1] = (float)rem[0];
	c_td_sub_td_d(rem, (double)a[1], rem);
	a[2] = (float)rem[0];
}

static inline void bnc_qs_from_qd(const double *qd, float *a)
{
	double rem[4];

	a[0] = (float)qd[0];
	c_qd_sub_qd_d(qd, (double)a[0], rem);
	a[1] = (float)rem[0];
	c_qd_sub_qd_d(rem, (double)a[1], rem);
	a[2] = (float)rem[0];
	c_qd_sub_qd_d(rem, (double)a[2], rem);
	a[3] = (float)rem[0];
}

/* ---- DS ---- */
#define BNC_DS_DELEG1(name, ddfunc) \
static inline void name(const float *a, float *ret) \
{ \
	double x[2], r[2]; \
	bnc_dd_from_ds(a, x); \
	ddfunc(x, r); \
	bnc_ds_from_dd(r, ret); \
}

BNC_DS_DELEG1(bnc_ds_exp, bnc_dd_exp)
BNC_DS_DELEG1(bnc_ds_expm1, bnc_dd_expm1)
BNC_DS_DELEG1(bnc_ds_log, bnc_dd_log)
BNC_DS_DELEG1(bnc_ds_log10, bnc_dd_log10)
BNC_DS_DELEG1(bnc_ds_sin, bnc_dd_sin)
BNC_DS_DELEG1(bnc_ds_cos, bnc_dd_cos)

static inline void bnc_ds_sincos(const float *a, float *sin_a, float *cos_a)
{
	double x[2], s[2], c[2];

	bnc_dd_from_ds(a, x);
	bnc_dd_sincos(x, s, c);
	bnc_ds_from_dd(s, sin_a);
	bnc_ds_from_dd(c, cos_a);
}

/* ---- TS ---- */
#define BNC_TS_DELEG1(name, tdfunc) \
static inline void name(const float *a, float *ret) \
{ \
	double x[3], r[3]; \
	bnc_td_from_ts(a, x); \
	tdfunc(x, r); \
	bnc_ts_from_td(r, ret); \
}

BNC_TS_DELEG1(bnc_ts_exp, bnc_td_exp)
BNC_TS_DELEG1(bnc_ts_expm1, bnc_td_expm1)
BNC_TS_DELEG1(bnc_ts_log, bnc_td_log)
BNC_TS_DELEG1(bnc_ts_log10, bnc_td_log10)
BNC_TS_DELEG1(bnc_ts_sin, bnc_td_sin)
BNC_TS_DELEG1(bnc_ts_cos, bnc_td_cos)

static inline void bnc_ts_sincos(const float *a, float *sin_a, float *cos_a)
{
	double x[3], s[3], c[3];

	bnc_td_from_ts(a, x);
	bnc_td_sincos(x, s, c);
	bnc_ts_from_td(s, sin_a);
	bnc_ts_from_td(c, cos_a);
}

/* ---- QS ---- */
#define BNC_QS_DELEG1(name, qdfunc) \
static inline void name(const float *a, float *ret) \
{ \
	double x[4], r[4]; \
	bnc_qd_from_qs(a, x); \
	qdfunc(x, r); \
	bnc_qs_from_qd(r, ret); \
}

BNC_QS_DELEG1(bnc_qs_exp, bnc_qd_exp)
BNC_QS_DELEG1(bnc_qs_expm1, bnc_qd_expm1)
BNC_QS_DELEG1(bnc_qs_log, bnc_qd_log)
BNC_QS_DELEG1(bnc_qs_log10, bnc_qd_log10)
BNC_QS_DELEG1(bnc_qs_sin, bnc_qd_sin)
BNC_QS_DELEG1(bnc_qs_cos, bnc_qd_cos)

static inline void bnc_qs_sincos(const float *a, float *sin_a, float *cos_a)
{
	double x[4], s[4], c[4];

	bnc_qd_from_qs(a, x);
	bnc_qd_sincos(x, s, c);
	bnc_qs_from_qd(s, sin_a);
	bnc_qs_from_qd(c, cos_a);
}

/* ---- FMA-driven divisions (ports of dtq-0.0.3 ds/ts/qs fma_div) ---- */

/* Forward declarations: bncfma_f.h may still be mid-processing when this
   header is reached through the include cycle (see bncelem_dd.h).       */
static inline void bnc_dwfmaf_safe(float z[2], const float x[2], float y, const float c[2]);
static inline void bnc_twfmaf_safe(float z[3], const float x[3], float y, const float c[3]);
static inline void bnc_qwfmaf_safe(float z[4], const float x[4], float y, const float c[4]);

/* Renormalize three float components (ts_real.h renorm3, 3-argument). */
static inline void bnc_ts_renorm3f(float *c0, float *c1, float *c2)
{
	float s0, s1, s2 = 0.0f;

	if (isinf(*c0)) return;

	s0  = fquick_two_sum(*c1, *c2, c2);
	*c0 = fquick_two_sum(*c0, s0, c1);

	s0 = *c0;
	s1 = *c1;
	if (s1 != 0.0f) {
		s1 = fquick_two_sum(s1, *c2, &s2);
	} else {
		s0 = fquick_two_sum(s0, *c2, &s1);
	}
	*c0 = s0; *c1 = s1; *c2 = s2;
}

/* Renormalize four float components (qs_real.h renorm4, 4-argument). */
static inline void bnc_qs_renorm4f(float *c0, float *c1, float *c2, float *c3)
{
	float s0, s1, s2 = 0.0f, s3 = 0.0f;

	if (isinf(*c0)) return;

	s0  = fquick_two_sum(*c2, *c3, c3);
	s0  = fquick_two_sum(*c1, s0, c2);
	*c0 = fquick_two_sum(*c0, s0, c1);

	s0 = *c0;
	s1 = *c1;
	if (s1 != 0.0f) {
		s1 = fquick_two_sum(s1, *c2, &s2);
		if (s2 != 0.0f)
			s2 = fquick_two_sum(s2, *c3, &s3);
		else
			s1 = fquick_two_sum(s1, *c3, &s2);
	} else {
		s0 = fquick_two_sum(s0, *c2, &s1);
		if (s1 != 0.0f)
			s1 = fquick_two_sum(s1, *c3, &s2);
		else
			s0 = fquick_two_sum(s0, *c3, &s1);
	}
	*c0 = s0; *c1 = s1; *c2 = s2; *c3 = s3;
}

/* ret := a / b (DS).  Port of dtq ds_real::fma_div.
   NOTE: dtq's float fma_div calls the plain scalar-multiplier FMA; here the
   div/sqrt-safe variant is used instead, following the rationale documented
   for the double-based fma_div in dtq itself (the residuals are not
   non-overlapping, so no FastTwoSum precondition holds; the safe variant
   costs the same and is never less accurate).                            */
static inline void bnc_ds_div_fma(const float *a, const float *b, float *ret)
{
	float q1, q2, r[2];

	q1 = a[0] / b[0];
	bnc_dwfmaf_safe(r, b, -q1, a);		/* r = a - q1 * b */
	q2 = (r[0] + r[1]) / b[0];

	q1 = fquick_two_sum(q1, q2, &q2);
	ret[0] = q1; ret[1] = q2;
}

/* ret := a / b (TS).  Port of dtq ts_real::fma_div (safe-FMA variant). */
static inline void bnc_ts_div_fma(const float *a, const float *b, float *ret)
{
	float q0, q1, q2, r[3];

	q0 = a[0] / b[0];
	bnc_twfmaf_safe(r, b, -q0, a);		/* r = a - q0 * b */
	q1 = r[0] / b[0];
	bnc_twfmaf_safe(r, b, -q1, r);
	q2 = r[0] / b[0];

	bnc_ts_renorm3f(&q0, &q1, &q2);
	ret[0] = q0; ret[1] = q1; ret[2] = q2;
}

/* ret := a / b (QS).  Port of dtq qs_real::fma_div (safe-FMA variant). */
static inline void bnc_qs_div_fma(const float *a, const float *b, float *ret)
{
	float q0, q1, q2, q3, r[4];

	q0 = a[0] / b[0];
	bnc_qwfmaf_safe(r, b, -q0, a);		/* r = a - q0 * b */
	q1 = r[0] / b[0];
	bnc_qwfmaf_safe(r, b, -q1, r);
	q2 = r[0] / b[0];
	bnc_qwfmaf_safe(r, b, -q2, r);
	q3 = r[0] / b[0];

	bnc_qs_renorm4f(&q0, &q1, &q2, &q3);
	ret[0] = q0; ret[1] = q1; ret[2] = q2; ret[3] = q3;
}

#endif /* __BNC_ELEM_F_H */
