/********************************************************************************/
/* bncelem_qd.h : quad-double (QD) elementary functions in plain C              */
/*                exp / expm1 / log / log10 / sin / cos / sincos                */
/*                                                                              */
/* Port of the fused-multiply-add elementary functions of dtq-0.0.3             */
/* (src/qd_real.cpp), which accumulate each Taylor term with one certified      */
/* branch-free QW-FMA (bnc_qwfma, bncfma_d.h) instead of a separate multiply    */
/* and add, saving one renormalization per term.  The convergence test uses     */
/* the leading-word product, so the term itself is never materialized.          */
/*                                                                              */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/* This file is part of BNCmatmul and distributed under the GNU LGPL v3.        */
/********************************************************************************/
/* IMPORTANT: compile with -ffp-contract=off (as everywhere in BNCmatmul).      */
#ifndef __BNC_ELEM_QD_H
#define __BNC_ELEM_QD_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>	/* rand(), used by c_dd_qd.h (c_dd_rand etc.) */

#include "rdd.h"       /* c_dd_qd.h was absorbed into rdd.h */
#include "bncfma_d.h"
#include "bncelem_tables.h"
#include "bncelem_dd.h"	/* bnc_dd_log seeds the quad-double Newton log */

/* Forward declaration: when this header is reached through the include
   cycle c_dd_qd.h -> bncelem.h -> bncelem_qd.h while bncfma_d.h is still
   being processed (its include guard is set but its definitions come
   later in the translation unit), bnc_qwfma is not yet visible.        */
static inline void bnc_qwfma(double z[4], const double x[4], const double y[4], const double c[4]);
static inline void bnc_qwfma_safe(double z[4], const double x[4], double y, const double c[4]);

/* ---- small helpers (exact operations) ---- */

/* ret := a * 2^n (exact per-limb scaling); c_qd_mul_pwr2 covers a * 2^p. */
static inline void bnc_qd_ldexp(const double *a, int n, double *ret)
{
	ret[0] = ldexp(a[0], n);
	ret[1] = ldexp(a[1], n);
	ret[2] = ldexp(a[2], n);
	ret[3] = ldexp(a[3], n);
}

#define BNC_QD_IS_ZERO(a) ((a)[0] == 0.0 && (a)[1] == 0.0 && (a)[2] == 0.0 && (a)[3] == 0.0)
#define BNC_QD_IS_ONE(a)  ((a)[0] == 1.0 && (a)[1] == 0.0 && (a)[2] == 0.0 && (a)[3] == 0.0)

/* ret := exp(x).  Two-level table-driven reduction (same idea as the sin/cos
   tables):

       a = m log 2 + j1/64 + j2/8192 + r,   |r| <= 2^-14 + eps,

   so  exp(a) = 2^m * exp(j1/64) * exp(j2/8192) * exp(r), where the two table
   factors are precomputed and exp(r) needs only a short Taylor series whose
   terms are accumulated with one QW-FMA each.  j1/64 and j2/8192 are exact
   binary values, so removing them from r is exact.                          */
/* ret := a / b by FMA-driven long division (port of dtq-0.0.3
   qd_real::fma_div).  Each residual  r <- r - q * b  is one fused QW-FMA
   in the div/sqrt-safe variant (see bncfma_d.h): the residuals are not
   guaranteed non-overlapping, so no FastTwoSum precondition holds.
   The final renormalization uses renorm4 (c_dd_qd.h), the C port of
   QD's five-component renorm().                                          */
static inline void bnc_qd_div_fma(const double *a, const double *b, double *ret)
{
	double q0, q1, q2, q3, q4, r[4];

	q0 = a[0] / b[0];
	bnc_qwfma_safe(r, b, -q0, a);		/* r = a - q0 * b */

	q1 = r[0] / b[0];
	bnc_qwfma_safe(r, b, -q1, r);

	q2 = r[0] / b[0];
	bnc_qwfma_safe(r, b, -q2, r);

	q3 = r[0] / b[0];
	bnc_qwfma_safe(r, b, -q3, r);

	q4 = r[0] / b[0];

	renorm4(&q0, &q1, &q2, &q3, &q4);

	ret[0] = q0; ret[1] = q1; ret[2] = q2; ret[3] = q3;
}

static inline void bnc_qd_exp(const double *a, double *ret)
{
	double m, r[4], s[4], p[4], t[4], thresh, tmag;
	int i, j1, j2;

	if (a[0] <= -709.0) {
		ret[0] = 0.0; ret[1] = 0.0; ret[2] = 0.0; ret[3] = 0.0;
		return;
	}
	if (a[0] >= 709.0) {
		ret[0] = HUGE_VAL; ret[1] = HUGE_VAL; ret[2] = HUGE_VAL; ret[3] = HUGE_VAL;
		return;
	}
	if (BNC_QD_IS_ZERO(a)) {
		ret[0] = 1.0; ret[1] = 0.0; ret[2] = 0.0; ret[3] = 0.0;
		return;
	}
	if (BNC_QD_IS_ONE(a)) {
		ret[0] = bnc_qd_e[0]; ret[1] = bnc_qd_e[1];
		ret[2] = bnc_qd_e[2]; ret[3] = bnc_qd_e[3];
		return;
	}

	m = floor(a[0] / bnc_qd_log2[0] + 0.5);
	c_qd_mul_qd_d(bnc_qd_log2, m, t);
	c_qd_sub(a, t, r);

	j1 = (int)floor(r[0] * 64.0 + 0.5);
	c_qd_sub_qd_d(r, (double)j1 * (1.0 / 64.0), r);	/* exact */
	j2 = (int)floor(r[0] * 8192.0 + 0.5);
	c_qd_sub_qd_d(r, (double)j2 * (1.0 / 8192.0), r);	/* exact */

	c_qd_sqr(r, p);
	c_qd_mul_pwr2(p, 0.5, t);
	c_qd_add(r, t, s);
	c_qd_add_qd_d(s, 1.0, s);	/* s = 1 + r + r^2/2 */

	thresh = bnc_qd_eps;
	i = 0;
	do {
		c_qd_mul(p, r, p);
		/* s += p / i!, fused; term magnitude from the leading words */
		tmag = fabs(p[0] * bnc_qd_inv_fact[i][0]);
		bnc_qwfma(s, p, bnc_qd_inv_fact[i], s);
		++i;
	} while (tmag > thresh && i < 13);

	c_qd_mul(s, bnc_qd_exp_table_64[j1 + 23], s);
	c_qd_mul(s, bnc_qd_exp_table_8192[j2 + 65], s);

	bnc_qd_ldexp(s, (int)m, ret);
}

/* ret := exp(a) - 1, accurate for small |a| (no cancellation).
   For |a| <= log(2)/2 the ln2 reduction in exp() is a no-op (m = 0) and the
   repeated-squaring ladder s <- 2s + s^2 is exactly the doubling rule for
   u = exp(t) - 1, so expm1 is the |kr| <= 0.347/2^16 exp() without the
   final +1.  For larger |a|, exp(a) - 1 suffers no cancellation.            */
static inline void bnc_qd_expm1(const double *a, double *ret)
{
	const double inv_k = 1.0 / 65536.0;	/* k = 2^16 */
	double r[4], s[4], p[4], t[4], u[4], thresh, tmag;
	int i, q;

	if (a[0] <= -709.0) {
		ret[0] = -1.0; ret[1] = 0.0; ret[2] = 0.0; ret[3] = 0.0;
		return;
	}
	if (a[0] >= 709.0) {
		ret[0] = HUGE_VAL; ret[1] = HUGE_VAL; ret[2] = HUGE_VAL; ret[3] = HUGE_VAL;
		return;
	}
	if (BNC_QD_IS_ZERO(a)) {
		ret[0] = 0.0; ret[1] = 0.0; ret[2] = 0.0; ret[3] = 0.0;
		return;
	}
	if (fabs(a[0]) > 0.346573590279972655) {	/* log(2)/2 */
		bnc_qd_exp(a, s);
		c_qd_sub_qd_d(s, 1.0, ret);
		return;
	}

	c_qd_mul_pwr2(a, inv_k, r);

	c_qd_sqr(r, p);
	c_qd_mul_pwr2(p, 0.5, t);
	c_qd_add(r, t, s);	/* s = r + r^2/2, no leading 1 */

	thresh = inv_k * bnc_qd_eps;
	i = 0;
	do {
		c_qd_mul(p, r, p);
		tmag = fabs(p[0] * bnc_qd_inv_fact[i][0]);
		bnc_qwfma(s, p, bnc_qd_inv_fact[i], s);
		++i;
	} while (tmag > thresh && i < 9);

	for (q = 0; q < 16; q++) {	/* 2^16 = k */
		c_qd_mul_pwr2(s, 2.0, t);
		c_qd_sqr(s, u);
		c_qd_add(t, u, s);
	}

	ret[0] = s[0]; ret[1] = s[1]; ret[2] = s[2]; ret[3] = s[3];	/* = exp(a) - 1, no +1 */
}

/* ret := log(x) (natural logarithm) by Newton iteration on f(x) = exp(x) - a:
   x' = x + a exp(-x) - 1.  Newton doubles the number of correct digits per
   iteration, so the iteration is started from the ~32-digit double-double
   logarithm (one dd exp) instead of the 16-digit double one: a single
   quad-double iteration -- one bnc_qd_exp instead of two -- then reaches
   full precision.                                                           */
static inline void bnc_qd_log(const double *a, double *ret)
{
	double x[4], mx[4], e[4], t[4], dd_a[2], dd_l[2];

	if (BNC_QD_IS_ONE(a)) {
		ret[0] = 0.0; ret[1] = 0.0; ret[2] = 0.0; ret[3] = 0.0;
		return;
	}
	/* MPFR/IEEE semantics, silently: log(0) = -inf, log(negative) = nan. */
	if (BNC_QD_IS_ZERO(a)) {
		ret[0] = -HUGE_VAL; ret[1] = -HUGE_VAL; ret[2] = -HUGE_VAL; ret[3] = -HUGE_VAL;
		return;
	}
	if (a[0] < 0.0) {
		ret[0] = NAN; ret[1] = NAN; ret[2] = NAN; ret[3] = NAN;
		return;
	}

	dd_a[0] = a[0]; dd_a[1] = a[1];
	bnc_dd_log(dd_a, dd_l);	/* initial approximation, ~32 digits */
	x[0] = dd_l[0]; x[1] = dd_l[1]; x[2] = 0.0; x[3] = 0.0;

	mx[0] = -x[0]; mx[1] = -x[1]; mx[2] = -x[2]; mx[3] = -x[3];
	bnc_qd_exp(mx, e);
	c_qd_mul(a, e, t);
	c_qd_add(x, t, x);
	c_qd_sub_qd_d(x, 1.0, ret);
}

/* ret := log10(x) */
static inline void bnc_qd_log10(const double *a, double *ret)
{
	double l[4];

	bnc_qd_log(a, l);
	c_qd_div(l, bnc_qd_log10_tbl, ret);
}

/* sin(a) by Taylor series; assumes |a| <= pi/2048. */
static inline void bnc_qd_sin_taylor(const double *a, double *ret)
{
	double thresh, tmag, r[4], s[4], x[4];
	int i;

	if (BNC_QD_IS_ZERO(a)) {
		ret[0] = 0.0; ret[1] = 0.0; ret[2] = 0.0; ret[3] = 0.0;
		return;
	}

	thresh = 0.5 * fabs(a[0]) * bnc_qd_eps;
	c_qd_sqr(a, x);
	x[0] = -x[0]; x[1] = -x[1]; x[2] = -x[2]; x[3] = -x[3];
	s[0] = a[0]; s[1] = a[1]; s[2] = a[2]; s[3] = a[3];
	r[0] = a[0]; r[1] = a[1]; r[2] = a[2]; r[3] = a[3];
	i = 0;
	do {
		c_qd_mul(r, x, r);
		/* s += r / i!, fused; term magnitude from the leading words */
		tmag = fabs(r[0] * bnc_qd_inv_fact[i][0]);
		bnc_qwfma(s, r, bnc_qd_inv_fact[i], s);
		i += 2;
	} while (i < BNC_ELEM_N_INV_FACT && tmag > thresh);

	ret[0] = s[0]; ret[1] = s[1]; ret[2] = s[2]; ret[3] = s[3];
}

/* cos(a) by Taylor series; assumes |a| <= pi/2048. */
static inline void bnc_qd_cos_taylor(const double *a, double *ret)
{
	double thresh, tmag, r[4], s[4], x[4], t[4];
	int i;

	if (BNC_QD_IS_ZERO(a)) {
		ret[0] = 1.0; ret[1] = 0.0; ret[2] = 0.0; ret[3] = 0.0;
		return;
	}

	thresh = 0.5 * bnc_qd_eps;
	c_qd_sqr(a, x);
	x[0] = -x[0]; x[1] = -x[1]; x[2] = -x[2]; x[3] = -x[3];
	r[0] = x[0]; r[1] = x[1]; r[2] = x[2]; r[3] = x[3];
	c_qd_mul_pwr2(r, 0.5, t);
	c_qd_add_qd_d(t, 1.0, s);
	i = 1;
	do {
		c_qd_mul(r, x, r);
		tmag = fabs(r[0] * bnc_qd_inv_fact[i][0]);
		bnc_qwfma(s, r, bnc_qd_inv_fact[i], s);
		i += 2;
	} while (i < BNC_ELEM_N_INV_FACT && tmag > thresh);

	ret[0] = s[0]; ret[1] = s[1]; ret[2] = s[2]; ret[3] = s[3];
}

/* sin_a := sin(a), cos_a := cos(a); cos from sqrt(1 - sin^2), |a| <= pi/2048 */
static inline void bnc_qd_sincos_taylor(const double *a, double *sin_a, double *cos_a)
{
	double s2[4], t[4];

	if (BNC_QD_IS_ZERO(a)) {
		sin_a[0] = 0.0; sin_a[1] = 0.0; sin_a[2] = 0.0; sin_a[3] = 0.0;
		cos_a[0] = 1.0; cos_a[1] = 0.0; cos_a[2] = 0.0; cos_a[3] = 0.0;
		return;
	}

	bnc_qd_sin_taylor(a, sin_a);
	c_qd_sqr(sin_a, s2);
	t[0] = 1.0; t[1] = 0.0; t[2] = 0.0; t[3] = 0.0;
	c_qd_sub(t, s2, t);
	c_qd_sqrt(t, cos_a);
}

/* Argument reduction shared by sin / cos / sincos:
   a = t + j * (pi/2) + k * (pi/1024), |t| <= pi/2048, |k| <= 256.
   Returns 0 on success, -1 if the reduction fails (|a| too large).          */
static inline int bnc_qd_sincos_reduce(const double *a, double *t, int *jj, int *kk)
{
	double z[4], r[4], w[4], q;
	int j, k, abs_k;

	/* approximately reduce modulo 2*pi */
	c_qd_div(a, bnc_qd_two_pi, z);
	c_qd_nint(z, z);
	c_qd_mul(bnc_qd_two_pi, z, w);
	c_qd_sub(a, w, r);

	/* approximately reduce modulo pi/2 and then modulo pi/1024 */
	q = floor(r[0] / bnc_qd_pi2[0] + 0.5);
	c_qd_mul_qd_d(bnc_qd_pi2, q, w);
	c_qd_sub(r, w, t);
	j = (int)q;
	q = floor(t[0] / bnc_qd_pi1024[0] + 0.5);
	c_qd_mul_qd_d(bnc_qd_pi1024, q, w);
	c_qd_sub(t, w, t);
	k = (int)q;
	abs_k = (k < 0) ? -k : k;

	if (j < -2 || j > 2) {
		fprintf(stderr, "(bnc_qd_sincos): Cannot reduce modulo pi/2.\n");
		return -1;
	}
	if (abs_k > 256) {
		fprintf(stderr, "(bnc_qd_sincos): Cannot reduce modulo pi/1024.\n");
		return -1;
	}

	*jj = j; *kk = k;
	return 0;
}

/* ret := sin(a) */
static inline void bnc_qd_sin(const double *a, double *ret)
{
	double t[4], u[4], v[4], sin_t[4], cos_t[4], p1[4], p2[4];
	int j, k, abs_k;

	if (BNC_QD_IS_ZERO(a)) {
		ret[0] = 0.0; ret[1] = 0.0; ret[2] = 0.0; ret[3] = 0.0;
		return;
	}

	if (bnc_qd_sincos_reduce(a, t, &j, &k) != 0) {
		ret[0] = NAN; ret[1] = NAN; ret[2] = NAN; ret[3] = NAN;
		return;
	}
	abs_k = (k < 0) ? -k : k;

	if (k == 0) {
		switch (j) {
			case 0:
				bnc_qd_sin_taylor(t, ret);
				return;
			case 1:
				bnc_qd_cos_taylor(t, ret);
				return;
			case -1:
				bnc_qd_cos_taylor(t, ret);
				ret[0] = -ret[0]; ret[1] = -ret[1]; ret[2] = -ret[2]; ret[3] = -ret[3];
				return;
			default:
				bnc_qd_sin_taylor(t, ret);
				ret[0] = -ret[0]; ret[1] = -ret[1]; ret[2] = -ret[2]; ret[3] = -ret[3];
				return;
		}
	}

	u[0] = bnc_qd_cos_table[abs_k - 1][0]; u[1] = bnc_qd_cos_table[abs_k - 1][1];
	u[2] = bnc_qd_cos_table[abs_k - 1][2]; u[3] = bnc_qd_cos_table[abs_k - 1][3];
	v[0] = bnc_qd_sin_table[abs_k - 1][0]; v[1] = bnc_qd_sin_table[abs_k - 1][1];
	v[2] = bnc_qd_sin_table[abs_k - 1][2]; v[3] = bnc_qd_sin_table[abs_k - 1][3];
	bnc_qd_sincos_taylor(t, sin_t, cos_t);

	if (j == 0) {
		c_qd_mul(u, sin_t, p1);
		c_qd_mul(v, cos_t, p2);
		if (k > 0) {
			c_qd_add(p1, p2, ret);	/* u sin_t + v cos_t */
		} else {
			c_qd_sub(p1, p2, ret);	/* u sin_t - v cos_t */
		}
	} else if (j == 1) {
		c_qd_mul(u, cos_t, p1);
		c_qd_mul(v, sin_t, p2);
		if (k > 0) {
			c_qd_sub(p1, p2, ret);	/* u cos_t - v sin_t */
		} else {
			c_qd_add(p1, p2, ret);	/* u cos_t + v sin_t */
		}
	} else if (j == -1) {
		c_qd_mul(u, cos_t, p1);
		c_qd_mul(v, sin_t, p2);
		if (k > 0) {
			c_qd_sub(p2, p1, ret);	/* v sin_t - u cos_t */
		} else {
			c_qd_add(p1, p2, ret);	/* -(u cos_t + v sin_t) */
			ret[0] = -ret[0]; ret[1] = -ret[1]; ret[2] = -ret[2]; ret[3] = -ret[3];
		}
	} else {
		c_qd_mul(u, sin_t, p1);
		c_qd_mul(v, cos_t, p2);
		if (k > 0) {
			c_qd_add(p1, p2, ret);	/* -(u sin_t + v cos_t) */
			ret[0] = -ret[0]; ret[1] = -ret[1]; ret[2] = -ret[2]; ret[3] = -ret[3];
		} else {
			c_qd_sub(p2, p1, ret);	/* v cos_t - u sin_t */
		}
	}
}

/* ret := cos(a) */
static inline void bnc_qd_cos(const double *a, double *ret)
{
	double t[4], u[4], v[4], sin_t[4], cos_t[4], p1[4], p2[4];
	int j, k, abs_k;

	if (BNC_QD_IS_ZERO(a)) {
		ret[0] = 1.0; ret[1] = 0.0; ret[2] = 0.0; ret[3] = 0.0;
		return;
	}

	if (bnc_qd_sincos_reduce(a, t, &j, &k) != 0) {
		ret[0] = NAN; ret[1] = NAN; ret[2] = NAN; ret[3] = NAN;
		return;
	}
	abs_k = (k < 0) ? -k : k;

	if (k == 0) {
		switch (j) {
			case 0:
				bnc_qd_cos_taylor(t, ret);
				return;
			case 1:
				bnc_qd_sin_taylor(t, ret);
				ret[0] = -ret[0]; ret[1] = -ret[1]; ret[2] = -ret[2]; ret[3] = -ret[3];
				return;
			case -1:
				bnc_qd_sin_taylor(t, ret);
				return;
			default:
				bnc_qd_cos_taylor(t, ret);
				ret[0] = -ret[0]; ret[1] = -ret[1]; ret[2] = -ret[2]; ret[3] = -ret[3];
				return;
		}
	}

	bnc_qd_sincos_taylor(t, sin_t, cos_t);
	u[0] = bnc_qd_cos_table[abs_k - 1][0]; u[1] = bnc_qd_cos_table[abs_k - 1][1];
	u[2] = bnc_qd_cos_table[abs_k - 1][2]; u[3] = bnc_qd_cos_table[abs_k - 1][3];
	v[0] = bnc_qd_sin_table[abs_k - 1][0]; v[1] = bnc_qd_sin_table[abs_k - 1][1];
	v[2] = bnc_qd_sin_table[abs_k - 1][2]; v[3] = bnc_qd_sin_table[abs_k - 1][3];

	if (j == 0) {
		c_qd_mul(u, cos_t, p1);
		c_qd_mul(v, sin_t, p2);
		if (k > 0) {
			c_qd_sub(p1, p2, ret);	/* u cos_t - v sin_t */
		} else {
			c_qd_add(p1, p2, ret);	/* u cos_t + v sin_t */
		}
	} else if (j == 1) {
		c_qd_mul(u, sin_t, p1);
		c_qd_mul(v, cos_t, p2);
		if (k > 0) {
			c_qd_add(p1, p2, ret);	/* -(u sin_t + v cos_t) */
			ret[0] = -ret[0]; ret[1] = -ret[1]; ret[2] = -ret[2]; ret[3] = -ret[3];
		} else {
			c_qd_sub(p2, p1, ret);	/* v cos_t - u sin_t */
		}
	} else if (j == -1) {
		c_qd_mul(u, sin_t, p1);
		c_qd_mul(v, cos_t, p2);
		if (k > 0) {
			c_qd_add(p1, p2, ret);	/* u sin_t + v cos_t */
		} else {
			c_qd_sub(p1, p2, ret);	/* u sin_t - v cos_t */
		}
	} else {
		c_qd_mul(v, sin_t, p1);
		c_qd_mul(u, cos_t, p2);
		if (k > 0) {
			c_qd_sub(p1, p2, ret);	/* v sin_t - u cos_t */
		} else {
			c_qd_add(p2, p1, ret);	/* -(u cos_t + v sin_t) */
			ret[0] = -ret[0]; ret[1] = -ret[1]; ret[2] = -ret[2]; ret[3] = -ret[3];
		}
	}
}

/* sin_a := sin(a), cos_a := cos(a) simultaneously */
static inline void bnc_qd_sincos(const double *a, double *sin_a, double *cos_a)
{
	double t[4], u[4], v[4], sin_t[4], cos_t[4], p1[4], p2[4];
	int j, k, abs_k;

	if (BNC_QD_IS_ZERO(a)) {
		sin_a[0] = 0.0; sin_a[1] = 0.0; sin_a[2] = 0.0; sin_a[3] = 0.0;
		cos_a[0] = 1.0; cos_a[1] = 0.0; cos_a[2] = 0.0; cos_a[3] = 0.0;
		return;
	}

	if (bnc_qd_sincos_reduce(a, t, &j, &k) != 0) {
		sin_a[0] = NAN; sin_a[1] = NAN; sin_a[2] = NAN; sin_a[3] = NAN;
		cos_a[0] = NAN; cos_a[1] = NAN; cos_a[2] = NAN; cos_a[3] = NAN;
		return;
	}
	abs_k = (k < 0) ? -k : k;

	bnc_qd_sincos_taylor(t, sin_t, cos_t);

	if (k == 0) {
		if (j == 0) {
			sin_a[0] = sin_t[0]; sin_a[1] = sin_t[1]; sin_a[2] = sin_t[2]; sin_a[3] = sin_t[3];
			cos_a[0] = cos_t[0]; cos_a[1] = cos_t[1]; cos_a[2] = cos_t[2]; cos_a[3] = cos_t[3];
		} else if (j == 1) {
			sin_a[0] = cos_t[0]; sin_a[1] = cos_t[1]; sin_a[2] = cos_t[2]; sin_a[3] = cos_t[3];
			cos_a[0] = -sin_t[0]; cos_a[1] = -sin_t[1]; cos_a[2] = -sin_t[2]; cos_a[3] = -sin_t[3];
		} else if (j == -1) {
			sin_a[0] = -cos_t[0]; sin_a[1] = -cos_t[1]; sin_a[2] = -cos_t[2]; sin_a[3] = -cos_t[3];
			cos_a[0] = sin_t[0]; cos_a[1] = sin_t[1]; cos_a[2] = sin_t[2]; cos_a[3] = sin_t[3];
		} else {
			sin_a[0] = -sin_t[0]; sin_a[1] = -sin_t[1]; sin_a[2] = -sin_t[2]; sin_a[3] = -sin_t[3];
			cos_a[0] = -cos_t[0]; cos_a[1] = -cos_t[1]; cos_a[2] = -cos_t[2]; cos_a[3] = -cos_t[3];
		}
		return;
	}

	u[0] = bnc_qd_cos_table[abs_k - 1][0]; u[1] = bnc_qd_cos_table[abs_k - 1][1];
	u[2] = bnc_qd_cos_table[abs_k - 1][2]; u[3] = bnc_qd_cos_table[abs_k - 1][3];
	v[0] = bnc_qd_sin_table[abs_k - 1][0]; v[1] = bnc_qd_sin_table[abs_k - 1][1];
	v[2] = bnc_qd_sin_table[abs_k - 1][2]; v[3] = bnc_qd_sin_table[abs_k - 1][3];

	if (j == 0) {
		c_qd_mul(u, sin_t, p1);
		c_qd_mul(v, cos_t, p2);
		if (k > 0) {
			c_qd_add(p1, p2, sin_a);	/* u sin_t + v cos_t */
		} else {
			c_qd_sub(p1, p2, sin_a);	/* u sin_t - v cos_t */
		}
		c_qd_mul(u, cos_t, p1);
		c_qd_mul(v, sin_t, p2);
		if (k > 0) {
			c_qd_sub(p1, p2, cos_a);	/* u cos_t - v sin_t */
		} else {
			c_qd_add(p1, p2, cos_a);	/* u cos_t + v sin_t */
		}
	} else if (j == 1) {
		c_qd_mul(u, sin_t, p1);
		c_qd_mul(v, cos_t, p2);
		if (k > 0) {
			c_qd_add(p1, p2, cos_a);	/* -(u sin_t + v cos_t) */
			cos_a[0] = -cos_a[0]; cos_a[1] = -cos_a[1]; cos_a[2] = -cos_a[2]; cos_a[3] = -cos_a[3];
		} else {
			c_qd_sub(p2, p1, cos_a);	/* v cos_t - u sin_t */
		}
		c_qd_mul(u, cos_t, p1);
		c_qd_mul(v, sin_t, p2);
		if (k > 0) {
			c_qd_sub(p1, p2, sin_a);	/* u cos_t - v sin_t */
		} else {
			c_qd_add(p1, p2, sin_a);	/* u cos_t + v sin_t */
		}
	} else if (j == -1) {
		c_qd_mul(u, sin_t, p1);
		c_qd_mul(v, cos_t, p2);
		if (k > 0) {
			c_qd_add(p1, p2, cos_a);	/* u sin_t + v cos_t */
		} else {
			c_qd_sub(p1, p2, cos_a);	/* u sin_t - v cos_t */
		}
		c_qd_mul(v, sin_t, p1);
		c_qd_mul(u, cos_t, p2);
		if (k > 0) {
			c_qd_sub(p1, p2, sin_a);	/* v sin_t - u cos_t */
		} else {
			c_qd_add(p2, p1, sin_a);	/* -(u cos_t + v sin_t) */
			sin_a[0] = -sin_a[0]; sin_a[1] = -sin_a[1]; sin_a[2] = -sin_a[2]; sin_a[3] = -sin_a[3];
		}
	} else {
		if (k > 0) {
			c_qd_mul(u, sin_t, p1);
			c_qd_mul(v, cos_t, p2);
			c_qd_add(p1, p2, sin_a);	/* -(u sin_t + v cos_t) */
			sin_a[0] = -sin_a[0]; sin_a[1] = -sin_a[1]; sin_a[2] = -sin_a[2]; sin_a[3] = -sin_a[3];
			c_qd_mul(v, sin_t, p1);
			c_qd_mul(u, cos_t, p2);
			c_qd_sub(p1, p2, cos_a);	/* v sin_t - u cos_t */
		} else {
			c_qd_mul(v, cos_t, p1);
			c_qd_mul(u, sin_t, p2);
			c_qd_sub(p1, p2, sin_a);	/* v cos_t - u sin_t */
			c_qd_mul(u, cos_t, p1);
			c_qd_mul(v, sin_t, p2);
			c_qd_add(p1, p2, cos_a);	/* -(u cos_t + v sin_t) */
			cos_a[0] = -cos_a[0]; cos_a[1] = -cos_a[1]; cos_a[2] = -cos_a[2]; cos_a[3] = -cos_a[3];
		}
	}
}

#endif /* __BNC_ELEM_QD_H */
