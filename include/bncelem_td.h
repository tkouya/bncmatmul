/********************************************************************************/
/* bncelem_td.h : triple-double (TD) elementary functions in plain C            */
/*                exp / expm1 / log / log10 / sin / cos / sincos                */
/*                                                                              */
/* Port of the fused-multiply-add elementary functions of dtq-0.0.3             */
/* (src/qd_real.cpp, native td transcendental section), which accumulate each   */
/* Taylor term with one certified branch-free TW-FMA (bnc_twfma, bncfma_d.h)    */
/* instead of a separate multiply and add, saving one renormalization per       */
/* term.  The convergence test uses the leading-word product, so the term       */
/* itself is never materialized.                                                */
/*                                                                              */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/* This file is part of BNCmatmul and distributed under the GNU LGPL v3.        */
/********************************************************************************/
/* IMPORTANT: compile with -ffp-contract=off (as everywhere in BNCmatmul).      */
#ifndef __BNC_ELEM_TD_H
#define __BNC_ELEM_TD_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>	/* rand(), used by c_dd_qd.h (c_dd_rand etc.) */

/* Forward declarations: c_dd_qd.h wraps these as c_td_exp, c_td_log, ...
   near its end (after including bncelem.h).  Unlike the dd/qd sets,
   which are forward-declared in c_dd_qd.h itself, the td set is declared
   here BEFORE c_dd_qd.h is included, so the wrappers also compile when
   this header is included first and c_dd_qd.h expands completely
   (bncelem.h then skips this header, whose guard is already set).       */
static inline void bnc_td_exp(const double *a, double *ret);
static inline void bnc_td_expm1(const double *a, double *ret);
static inline void bnc_td_log(const double *a, double *ret);
static inline void bnc_td_log10(const double *a, double *ret);
static inline void bnc_td_sin(const double *a, double *ret);
static inline void bnc_td_cos(const double *a, double *ret);
static inline void bnc_td_sincos(const double *a, double *sin_a, double *cos_a);

#include "c_dd_qd.h"
#include "bncfma_d.h"
#include "bncelem_tables.h"

/* Forward declaration: when this header is reached through the include
   cycle c_dd_qd.h -> bncelem.h -> bncelem_td.h while bncfma_d.h is still
   being processed (its include guard is set but its definitions come
   later in the translation unit), bnc_twfma is not yet visible.        */
static inline void bnc_twfma(double z[3], const double x[3], const double y[3], const double c[3]);
static inline void bnc_twfma_safe(double z[3], const double x[3], double y, const double c[3]);

/* ---- small helpers (exact operations) ---- */

/* ret := a * 2^p (p is a power of two: exact per-limb scaling) */
static inline void bnc_td_mul_pwr2(const double *a, double p, double *ret)
{
	ret[0] = a[0] * p;
	ret[1] = a[1] * p;
	ret[2] = a[2] * p;
}

/* ret := a * 2^n (exact per-limb scaling) */
static inline void bnc_td_ldexp(const double *a, int n, double *ret)
{
	ret[0] = ldexp(a[0], n);
	ret[1] = ldexp(a[1], n);
	ret[2] = ldexp(a[2], n);
}

#define BNC_TD_IS_ZERO(a) ((a)[0] == 0.0 && (a)[1] == 0.0 && (a)[2] == 0.0)
#define BNC_TD_IS_ONE(a)  ((a)[0] == 1.0 && (a)[1] == 0.0 && (a)[2] == 0.0)

/* nint for a plain double (round to nearest, ties away from the integer),
   as in qd/inline.h */
static inline double bnc_td_nint_d(double d)
{
	if (d == floor(d)) {
		return d;
	}
	return floor(d + 0.5);
}

/* Renormalize three components (td_inline.h renorm3, 3-argument version) */
static inline void bnc_td_renorm3(double *c0, double *c1, double *c2)
{
	double s0, s1, s2 = 0.0;

	if (isinf(*c0)) {
		return;
	}

	s0 = quick_two_sum(*c1, *c2, c2);
	*c0 = quick_two_sum(*c0, s0, c1);

	s0 = *c0;
	s1 = *c1;
	if (s1 != 0.0) {
		s1 = quick_two_sum(s1, *c2, &s2);
	} else {
		s0 = quick_two_sum(s0, *c2, &s1);
	}

	*c0 = s0;
	*c1 = s1;
	*c2 = s2;
}

/* ret := nearest integer to a (limb-wise rounding with half-way correction,
   port of td_real nint() in src/td_real.cpp) */
static inline void bnc_td_nint(const double *a, double *ret)
{
	double x0, x1, x2;

	x0 = bnc_td_nint_d(a[0]);
	x1 = x2 = 0.0;

	if (x0 == a[0]) {
		/* First double is already an integer. */
		x1 = bnc_td_nint_d(a[1]);

		if (x1 == a[1]) {
			/* Second double is already an integer. */
			x2 = bnc_td_nint_d(a[2]);
		} else {
			if (fabs(x1 - a[1]) == 0.5 && a[2] < 0.0) {
				x1 -= 1.0;
			}
		}
	} else {
		/* First double is not an integer. */
		if (fabs(x0 - a[0]) == 0.5 && a[1] < 0.0) {
			x0 -= 1.0;
		}
	}

	bnc_td_renorm3(&x0, &x1, &x2);
	ret[0] = x0; ret[1] = x1; ret[2] = x2;
}

/* ret := exp(x).  Two-level table-driven reduction (qd_real.cpp, td exp):
   a = m log 2 + j1/64 + j2/8192 + r with |r| <= 1/16384, then a short Taylor
   series whose terms are accumulated with one TW-FMA each, then two table
   multiplications and the final 2^m scaling.                                */
/* Renormalize four components into three (td_inline.h renorm3, 4-argument
   version; used by the FMA-driven division below). */
static inline void bnc_td_renorm3_4(double *c0, double *c1, double *c2, double *c3)
{
	double s0, s1, s2 = 0.0;

	if (isinf(*c0)) return;

	s0  = quick_two_sum(*c2, *c3, c3);
	s0  = quick_two_sum(*c1, s0, c2);
	*c0 = quick_two_sum(*c0, s0, c1);

	s0 = *c0;
	s1 = *c1;
	if (s1 != 0.0) {
		s1 = quick_two_sum(s1, *c2, &s2);
		if (s2 != 0.0)
			s2 += *c3;
		else
			s1 = quick_two_sum(s1, *c3, &s2);
	} else {
		s0 = quick_two_sum(s0, *c2, &s1);
		if (s1 != 0.0)
			s1 = quick_two_sum(s1, *c3, &s2);
		else
			s0 = quick_two_sum(s0, *c3, &s1);
	}

	*c0 = s0;
	*c1 = s1;
	*c2 = s2;
}

/* ret := a / b by FMA-driven long division (port of dtq-0.0.3
   td_real::fma_div).  Each residual  r <- r - q * b  is one fused TW-FMA
   in the div/sqrt-safe variant (see bncfma_d.h): the residuals are not
   guaranteed non-overlapping, so no FastTwoSum precondition holds.       */
static inline void bnc_td_div_fma(const double *a, const double *b, double *ret)
{
	double q0, q1, q2, q3, r[3];

	q0 = a[0] / b[0];
	bnc_twfma_safe(r, b, -q0, a);		/* r = a - q0 * b */

	q1 = r[0] / b[0];
	bnc_twfma_safe(r, b, -q1, r);

	q2 = r[0] / b[0];
	bnc_twfma_safe(r, b, -q2, r);

	q3 = r[0] / b[0];

	bnc_td_renorm3_4(&q0, &q1, &q2, &q3);

	ret[0] = q0; ret[1] = q1; ret[2] = q2;
}

static inline void bnc_td_exp(const double *a, double *ret)
{
	double m, r[3], s[3], p[3], t[3], u[3], thresh, tmag;
	int i, j1, j2;

	if (a[0] <= -709.0) {
		ret[0] = 0.0; ret[1] = 0.0; ret[2] = 0.0;
		return;
	}
	if (a[0] >= 709.0) {
		ret[0] = HUGE_VAL; ret[1] = HUGE_VAL; ret[2] = HUGE_VAL;
		return;
	}
	if (BNC_TD_IS_ZERO(a)) {
		ret[0] = 1.0; ret[1] = 0.0; ret[2] = 0.0;
		return;
	}
	if (BNC_TD_IS_ONE(a)) {
		ret[0] = bnc_td_e[0]; ret[1] = bnc_td_e[1]; ret[2] = bnc_td_e[2];
		return;
	}

	m = floor(a[0] / bnc_td_log2[0] + 0.5);
	c_td_mul_td_d(bnc_td_log2, m, t);
	c_td_subq(a, t, r);

	j1 = (int)floor(r[0] * 64.0 + 0.5);
	c_td_sub_td_d(r, (double)j1 * (1.0 / 64.0), r);	/* exact */
	j2 = (int)floor(r[0] * 8192.0 + 0.5);
	c_td_sub_td_d(r, (double)j2 * (1.0 / 8192.0), r);	/* exact */

	c_td_sqr(r, p);
	c_td_add_td_d(r, 1.0, t);	/* t = 1 + r */
	bnc_td_mul_pwr2(p, 0.5, u);
	c_td_add(t, u, s);		/* s = 1 + r + r^2/2 */

	thresh = bnc_td_eps;
	i = 0;
	do {
		c_td_mul(p, r, p);
		tmag = fabs(p[0] * bnc_td_inv_fact[i][0]);
		bnc_twfma(s, p, bnc_td_inv_fact[i], s);
		++i;
	} while (tmag > thresh && i < 10);

	c_td_mul(s, bnc_td_exp_table_64[j1 + 23], s);
	c_td_mul(s, bnc_td_exp_table_8192[j2 + 65], s);

	bnc_td_ldexp(s, (int)m, ret);
}

/* ret := exp(a) - 1, accurate for small |a| (no cancellation).
   For |a| <= log(2)/2 the ln2 reduction in exp() is a no-op (m = 0) and the
   repeated-squaring ladder s <- 2s + s^2 is exactly the doubling rule for
   u = exp(t) - 1, so expm1 is exp() without the final +1 (qd_real.cpp,
   td expm1 with k = 2^16).                                                  */
static inline void bnc_td_expm1(const double *a, double *ret)
{
	const double inv_k = 1.0 / 65536.0;	/* 1/2^16 */
	double r[3], s[3], p[3], t[3], u[3], thresh, tmag;
	int i, q;

	if (a[0] <= -709.0) {
		ret[0] = -1.0; ret[1] = 0.0; ret[2] = 0.0;
		return;
	}
	if (a[0] >= 709.0) {
		ret[0] = HUGE_VAL; ret[1] = HUGE_VAL; ret[2] = HUGE_VAL;
		return;
	}
	if (BNC_TD_IS_ZERO(a)) {
		ret[0] = 0.0; ret[1] = 0.0; ret[2] = 0.0;
		return;
	}
	if (fabs(a[0]) > 0.346573590279972655) {	/* log(2)/2 */
		bnc_td_exp(a, s);
		c_td_sub_td_d(s, 1.0, ret);
		return;
	}

	bnc_td_mul_pwr2(a, inv_k, r);

	c_td_sqr(r, p);
	bnc_td_mul_pwr2(p, 0.5, t);
	c_td_add(r, t, s);		/* s = r + r^2/2 */

	thresh = inv_k * bnc_td_eps;
	i = 0;
	do {
		c_td_mul(p, r, p);
		tmag = fabs(p[0] * bnc_td_inv_fact[i][0]);
		bnc_twfma(s, p, bnc_td_inv_fact[i], s);
		++i;
	} while (tmag > thresh && i < 9);

	for (q = 0; q < 16; q++) {	/* 2^16 = k */
		bnc_td_mul_pwr2(s, 2.0, t);
		c_td_sqr(s, u);
		c_td_add(t, u, s);
	}

	ret[0] = s[0]; ret[1] = s[1]; ret[2] = s[2];	/* = exp(a) - 1, no +1 */
}

/* ret := log(x) (natural logarithm) by one Newton step on f(x) = exp(x) - a:
   x' = x + a exp(-x) - 1, started from the ~32-digit double-double log of
   the top two limbs (c_dd_log), so one triple-double iteration (one td exp)
   reaches full precision (qd_real.cpp, td log).                             */
static inline void bnc_td_log(const double *a, double *ret)
{
	double x[3], mx[3], e[3], t[3], ddl[2];

	if (BNC_TD_IS_ONE(a)) {
		ret[0] = 0.0; ret[1] = 0.0; ret[2] = 0.0;
		return;
	}
	/* MPFR/IEEE semantics, silently: log(0) = -inf, log(negative) = nan. */
	if (BNC_TD_IS_ZERO(a)) {
		ret[0] = -HUGE_VAL; ret[1] = -HUGE_VAL; ret[2] = -HUGE_VAL;
		return;
	}
	if (a[0] < 0.0) {
		ret[0] = NAN; ret[1] = NAN; ret[2] = NAN;
		return;
	}

	c_dd_log(a, ddl);	/* dd log of the leading two limbs */
	x[0] = ddl[0]; x[1] = ddl[1]; x[2] = 0.0;

	c_td_neg(x, mx);
	bnc_td_exp(mx, e);
	c_td_mul(a, e, t);
	c_td_add(x, t, x);
	c_td_sub_td_d(x, 1.0, ret);
}

/* ret := log10(x) */
static inline void bnc_td_log10(const double *a, double *ret)
{
	double l[3], q[4];	/* c_td_div writes a 4th (spill) word */

	bnc_td_log(a, l);
	c_td_div(l, bnc_td_log10_tbl, q);
	ret[0] = q[0]; ret[1] = q[1]; ret[2] = q[2];
}

/* sin(a) by Taylor series; assumes |a| <= pi/2048. */
static inline void bnc_td_sin_taylor(const double *a, double *ret)
{
	double thresh, tmag, p[3], s[3], x[3];
	int i;

	if (BNC_TD_IS_ZERO(a)) {
		ret[0] = 0.0; ret[1] = 0.0; ret[2] = 0.0;
		return;
	}

	thresh = 0.5 * bnc_td_eps * fabs(a[0]);
	c_td_sqr((double *)a, x);
	x[0] = -x[0]; x[1] = -x[1]; x[2] = -x[2];
	s[0] = a[0]; s[1] = a[1]; s[2] = a[2];
	p[0] = a[0]; p[1] = a[1]; p[2] = a[2];
	i = 0;
	do {
		c_td_mul(p, x, p);
		/* s += p / i!, fused; term magnitude from the leading words */
		tmag = fabs(p[0] * bnc_td_inv_fact[i][0]);
		bnc_twfma(s, p, bnc_td_inv_fact[i], s);
		i += 2;
	} while (i < BNC_ELEM_N_INV_FACT && tmag > thresh);

	ret[0] = s[0]; ret[1] = s[1]; ret[2] = s[2];
}

/* cos(a) by Taylor series; assumes |a| <= pi/2048. */
static inline void bnc_td_cos_taylor(const double *a, double *ret)
{
	double thresh, tmag, p[3], s[3], x[3], t[3];
	int i;

	if (BNC_TD_IS_ZERO(a)) {
		ret[0] = 1.0; ret[1] = 0.0; ret[2] = 0.0;
		return;
	}

	thresh = 0.5 * bnc_td_eps;
	c_td_sqr((double *)a, x);
	x[0] = -x[0]; x[1] = -x[1]; x[2] = -x[2];
	p[0] = x[0]; p[1] = x[1]; p[2] = x[2];
	bnc_td_mul_pwr2(p, 0.5, t);
	c_td_add_td_d(t, 1.0, s);
	i = 1;
	do {
		c_td_mul(p, x, p);
		tmag = fabs(p[0] * bnc_td_inv_fact[i][0]);
		bnc_twfma(s, p, bnc_td_inv_fact[i], s);
		i += 2;
	} while (i < BNC_ELEM_N_INV_FACT && tmag > thresh);

	ret[0] = s[0]; ret[1] = s[1]; ret[2] = s[2];
}

/* sin_a := sin(a), cos_a := cos(a) via the sin Taylor series and
   cos = sqrt(1 - sin^2); assumes |a| <= pi/2048.                            */
static inline void bnc_td_sincos_taylor(const double *a, double *sin_a, double *cos_a)
{
	double s2[3], t[3];

	if (BNC_TD_IS_ZERO(a)) {
		sin_a[0] = 0.0; sin_a[1] = 0.0; sin_a[2] = 0.0;
		cos_a[0] = 1.0; cos_a[1] = 0.0; cos_a[2] = 0.0;
		return;
	}

	bnc_td_sin_taylor(a, sin_a);
	c_td_sqr(sin_a, s2);
	t[0] = 1.0; t[1] = 0.0; t[2] = 0.0;
	c_td_sub(t, s2, t);
	c_td_sqrt(t, cos_a);
}

/* Argument reduction shared by sin / cos / sincos:
   a = t + j * (pi/2) + k * (pi/1024), |t| <= pi/2048, |k| <= 256.
   Returns 0 on success, -1 if the reduction fails (|a| too large).          */
static inline int bnc_td_sincos_reduce(const double *a, double *t, int *jj, int *kk)
{
	double z[3], r[3], w[3], q, q4[4];	/* c_td_div writes a 4th (spill) word */
	int j, k, abs_k;

	/* approximately reduce modulo 2*pi */
	c_td_div(a, bnc_td_two_pi, q4);
	bnc_td_nint(q4, z);
	c_td_mul(bnc_td_two_pi, z, w);
	c_td_subq(a, w, r);

	/* approximately reduce modulo pi/2 and then modulo pi/1024 */
	q = floor(r[0] / bnc_td_pi2[0] + 0.5);
	c_td_mul_td_d(bnc_td_pi2, q, w);
	c_td_subq(r, w, t);
	j = (int)q;
	q = floor(t[0] / bnc_td_pi1024[0] + 0.5);
	c_td_mul_td_d(bnc_td_pi1024, q, w);
	c_td_subq(t, w, t);
	k = (int)q;
	abs_k = (k < 0) ? -k : k;

	if (j < -2 || j > 2) {
		fprintf(stderr, "(bnc_td_sincos): Cannot reduce modulo pi/2.\n");
		return -1;
	}
	if (abs_k > 256) {
		fprintf(stderr, "(bnc_td_sincos): Cannot reduce modulo pi/1024.\n");
		return -1;
	}

	*jj = j; *kk = k;
	return 0;
}

/* ret := sin(a) */
static inline void bnc_td_sin(const double *a, double *ret)
{
	double t[3], u[3], v[3], sin_t[3], cos_t[3], p1[3], p2[3];
	int j, k, abs_k;

	if (BNC_TD_IS_ZERO(a)) {
		ret[0] = 0.0; ret[1] = 0.0; ret[2] = 0.0;
		return;
	}

	if (bnc_td_sincos_reduce(a, t, &j, &k) != 0) {
		ret[0] = NAN; ret[1] = NAN; ret[2] = NAN;
		return;
	}
	abs_k = (k < 0) ? -k : k;

	if (k == 0) {
		switch (j) {
			case 0:
				bnc_td_sin_taylor(t, ret);
				return;
			case 1:
				bnc_td_cos_taylor(t, ret);
				return;
			case -1:
				bnc_td_cos_taylor(t, ret);
				ret[0] = -ret[0]; ret[1] = -ret[1]; ret[2] = -ret[2];
				return;
			default:
				bnc_td_sin_taylor(t, ret);
				ret[0] = -ret[0]; ret[1] = -ret[1]; ret[2] = -ret[2];
				return;
		}
	}

	u[0] = bnc_td_cos_table[abs_k - 1][0]; u[1] = bnc_td_cos_table[abs_k - 1][1]; u[2] = bnc_td_cos_table[abs_k - 1][2];
	v[0] = bnc_td_sin_table[abs_k - 1][0]; v[1] = bnc_td_sin_table[abs_k - 1][1]; v[2] = bnc_td_sin_table[abs_k - 1][2];
	bnc_td_sincos_taylor(t, sin_t, cos_t);

	c_td_mul(u, sin_t, p1);	/* p1 = u * sin_t */
	c_td_mul(v, cos_t, p2);	/* p2 = v * cos_t */
	if (j == 0) {
		if (k > 0) {
			c_td_add(p1, p2, ret);	/* u sin_t + v cos_t */
		} else {
			c_td_sub(p1, p2, ret);	/* u sin_t - v cos_t */
		}
	} else if (j == 1) {
		c_td_mul(u, cos_t, p1);
		c_td_mul(v, sin_t, p2);
		if (k > 0) {
			c_td_sub(p1, p2, ret);	/* u cos_t - v sin_t */
		} else {
			c_td_add(p1, p2, ret);	/* u cos_t + v sin_t */
		}
	} else if (j == -1) {
		c_td_mul(u, cos_t, p1);
		c_td_mul(v, sin_t, p2);
		if (k > 0) {
			c_td_sub(p2, p1, ret);	/* v sin_t - u cos_t */
		} else {
			c_td_add(p1, p2, ret);	/* -(u cos_t + v sin_t) */
			ret[0] = -ret[0]; ret[1] = -ret[1]; ret[2] = -ret[2];
		}
	} else {
		if (k > 0) {
			c_td_add(p1, p2, ret);	/* -(u sin_t + v cos_t) */
			ret[0] = -ret[0]; ret[1] = -ret[1]; ret[2] = -ret[2];
		} else {
			c_td_sub(p2, p1, ret);	/* v cos_t - u sin_t */
		}
	}
}

/* ret := cos(a) */
static inline void bnc_td_cos(const double *a, double *ret)
{
	double t[3], u[3], v[3], sin_t[3], cos_t[3], p1[3], p2[3];
	int j, k, abs_k;

	if (BNC_TD_IS_ZERO(a)) {
		ret[0] = 1.0; ret[1] = 0.0; ret[2] = 0.0;
		return;
	}

	if (bnc_td_sincos_reduce(a, t, &j, &k) != 0) {
		ret[0] = NAN; ret[1] = NAN; ret[2] = NAN;
		return;
	}
	abs_k = (k < 0) ? -k : k;

	if (k == 0) {
		switch (j) {
			case 0:
				bnc_td_cos_taylor(t, ret);
				return;
			case 1:
				bnc_td_sin_taylor(t, ret);
				ret[0] = -ret[0]; ret[1] = -ret[1]; ret[2] = -ret[2];
				return;
			case -1:
				bnc_td_sin_taylor(t, ret);
				return;
			default:
				bnc_td_cos_taylor(t, ret);
				ret[0] = -ret[0]; ret[1] = -ret[1]; ret[2] = -ret[2];
				return;
		}
	}

	bnc_td_sincos_taylor(t, sin_t, cos_t);
	u[0] = bnc_td_cos_table[abs_k - 1][0]; u[1] = bnc_td_cos_table[abs_k - 1][1]; u[2] = bnc_td_cos_table[abs_k - 1][2];
	v[0] = bnc_td_sin_table[abs_k - 1][0]; v[1] = bnc_td_sin_table[abs_k - 1][1]; v[2] = bnc_td_sin_table[abs_k - 1][2];

	if (j == 0) {
		c_td_mul(u, cos_t, p1);
		c_td_mul(v, sin_t, p2);
		if (k > 0) {
			c_td_sub(p1, p2, ret);	/* u cos_t - v sin_t */
		} else {
			c_td_add(p1, p2, ret);	/* u cos_t + v sin_t */
		}
	} else if (j == 1) {
		c_td_mul(u, sin_t, p1);
		c_td_mul(v, cos_t, p2);
		if (k > 0) {
			c_td_add(p1, p2, ret);	/* -(u sin_t + v cos_t) */
			ret[0] = -ret[0]; ret[1] = -ret[1]; ret[2] = -ret[2];
		} else {
			c_td_sub(p2, p1, ret);	/* v cos_t - u sin_t */
		}
	} else if (j == -1) {
		c_td_mul(u, sin_t, p1);
		c_td_mul(v, cos_t, p2);
		if (k > 0) {
			c_td_add(p1, p2, ret);	/* u sin_t + v cos_t */
		} else {
			c_td_sub(p1, p2, ret);	/* u sin_t - v cos_t */
		}
	} else {
		c_td_mul(v, sin_t, p1);
		c_td_mul(u, cos_t, p2);
		if (k > 0) {
			c_td_sub(p1, p2, ret);	/* v sin_t - u cos_t */
		} else {
			c_td_add(p2, p1, ret);	/* -(u cos_t + v sin_t) */
			ret[0] = -ret[0]; ret[1] = -ret[1]; ret[2] = -ret[2];
		}
	}
}

/* sin_a := sin(a), cos_a := cos(a) simultaneously.
   The j/k reconstruction follows qd_real.cpp (td sincos) case by case.      */
static inline void bnc_td_sincos(const double *a, double *sin_a, double *cos_a)
{
	double t[3], u[3], v[3], sin_t[3], cos_t[3];
	double us[3], vc[3], uc[3], vs[3];
	int j, k, abs_k;

	if (BNC_TD_IS_ZERO(a)) {
		sin_a[0] = 0.0; sin_a[1] = 0.0; sin_a[2] = 0.0;
		cos_a[0] = 1.0; cos_a[1] = 0.0; cos_a[2] = 0.0;
		return;
	}

	if (bnc_td_sincos_reduce(a, t, &j, &k) != 0) {
		sin_a[0] = NAN; sin_a[1] = NAN; sin_a[2] = NAN;
		cos_a[0] = NAN; cos_a[1] = NAN; cos_a[2] = NAN;
		return;
	}
	abs_k = (k < 0) ? -k : k;

	bnc_td_sincos_taylor(t, sin_t, cos_t);

	if (k == 0) {
		if (j == 0) {
			sin_a[0] = sin_t[0]; sin_a[1] = sin_t[1]; sin_a[2] = sin_t[2];
			cos_a[0] = cos_t[0]; cos_a[1] = cos_t[1]; cos_a[2] = cos_t[2];
		} else if (j == 1) {
			sin_a[0] = cos_t[0]; sin_a[1] = cos_t[1]; sin_a[2] = cos_t[2];
			cos_a[0] = -sin_t[0]; cos_a[1] = -sin_t[1]; cos_a[2] = -sin_t[2];
		} else if (j == -1) {
			sin_a[0] = -cos_t[0]; sin_a[1] = -cos_t[1]; sin_a[2] = -cos_t[2];
			cos_a[0] = sin_t[0]; cos_a[1] = sin_t[1]; cos_a[2] = sin_t[2];
		} else {
			sin_a[0] = -sin_t[0]; sin_a[1] = -sin_t[1]; sin_a[2] = -sin_t[2];
			cos_a[0] = -cos_t[0]; cos_a[1] = -cos_t[1]; cos_a[2] = -cos_t[2];
		}
		return;
	}

	u[0] = bnc_td_cos_table[abs_k - 1][0]; u[1] = bnc_td_cos_table[abs_k - 1][1]; u[2] = bnc_td_cos_table[abs_k - 1][2];
	v[0] = bnc_td_sin_table[abs_k - 1][0]; v[1] = bnc_td_sin_table[abs_k - 1][1]; v[2] = bnc_td_sin_table[abs_k - 1][2];

	c_td_mul(u, sin_t, us);	/* us = u * sin_t */
	c_td_mul(v, cos_t, vc);	/* vc = v * cos_t */
	c_td_mul(u, cos_t, uc);	/* uc = u * cos_t */
	c_td_mul(v, sin_t, vs);	/* vs = v * sin_t */

	if (j == 0) {
		if (k > 0) {
			c_td_add(us, vc, sin_a);	/* u sin_t + v cos_t */
			c_td_sub(uc, vs, cos_a);	/* u cos_t - v sin_t */
		} else {
			c_td_sub(us, vc, sin_a);	/* u sin_t - v cos_t */
			c_td_add(uc, vs, cos_a);	/* u cos_t + v sin_t */
		}
	} else if (j == 1) {
		if (k > 0) {
			c_td_add(us, vc, cos_a);	/* -(u sin_t + v cos_t) */
			cos_a[0] = -cos_a[0]; cos_a[1] = -cos_a[1]; cos_a[2] = -cos_a[2];
			c_td_sub(uc, vs, sin_a);	/* u cos_t - v sin_t */
		} else {
			c_td_sub(vc, us, cos_a);	/* v cos_t - u sin_t */
			c_td_add(uc, vs, sin_a);	/* u cos_t + v sin_t */
		}
	} else if (j == -1) {
		if (k > 0) {
			c_td_add(us, vc, cos_a);	/* u sin_t + v cos_t */
			c_td_sub(vs, uc, sin_a);	/* v sin_t - u cos_t */
		} else {
			c_td_sub(us, vc, cos_a);	/* u sin_t - v cos_t */
			c_td_add(uc, vs, sin_a);	/* -(u cos_t + v sin_t) */
			sin_a[0] = -sin_a[0]; sin_a[1] = -sin_a[1]; sin_a[2] = -sin_a[2];
		}
	} else {
		if (k > 0) {
			c_td_add(us, vc, sin_a);	/* -(u sin_t + v cos_t) */
			sin_a[0] = -sin_a[0]; sin_a[1] = -sin_a[1]; sin_a[2] = -sin_a[2];
			c_td_sub(vs, uc, cos_a);	/* v sin_t - u cos_t */
		} else {
			c_td_sub(vc, us, sin_a);	/* v cos_t - u sin_t */
			c_td_add(uc, vs, cos_a);	/* -(u cos_t + v sin_t) */
			cos_a[0] = -cos_a[0]; cos_a[1] = -cos_a[1]; cos_a[2] = -cos_a[2];
		}
	}
}

#endif /* __BNC_ELEM_TD_H */
