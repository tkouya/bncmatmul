/********************************************************************************/
/* bncelem_dd.h : double-double (DD) elementary functions in plain C            */
/*                exp / expm1 / log / log10 / sin / cos / sincos                */
/*                                                                              */
/* Port of the fused-multiply-add elementary functions of dtq-0.0.3             */
/* (src/dd_real.cpp), which accumulate each Taylor term with one certified      */
/* branch-free DW-FMA (bnc_dwfma, bncfma_d.h) instead of a separate multiply    */
/* and add, saving one renormalization per term.  The convergence test uses     */
/* the leading-word product, so the term itself is never materialized.          */
/*                                                                              */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/* This file is part of BNCmatmul and distributed under the GNU LGPL v3.        */
/********************************************************************************/
/* IMPORTANT: compile with -ffp-contract=off (as everywhere in BNCmatmul).      */
#ifndef __BNC_ELEM_DD_H
#define __BNC_ELEM_DD_H

#include <math.h>
#include <stdio.h>

#include "rdd.h"       /* c_dd_qd.h was absorbed into rdd.h */
#include "bncfma_d.h"
#include "bncelem_tables.h"

/* Forward declaration: when this header is reached through the include
   cycle c_dd_qd.h -> bncelem.h -> bncelem_dd.h while bncfma_d.h is still
   being processed (its include guard is set but its definitions come
   later in the translation unit), bnc_dwfma is not yet visible.        */
static inline void bnc_dwfma(double z[2], const double x[2], const double y[2], const double c[2]);
static inline void bnc_dwfma_safe(double z[2], const double x[2], double y, const double c[2]);

/* ---- small helpers (exact operations) ---- */

/* ret := a * 2^p (p is a power of two: exact per-limb scaling) */
static inline void bnc_dd_mul_pwr2(const double *a, double p, double *ret)
{
	ret[0] = a[0] * p;
	ret[1] = a[1] * p;
}

/* ret := a * 2^n (exact per-limb scaling) */
static inline void bnc_dd_ldexp(const double *a, int n, double *ret)
{
	ret[0] = ldexp(a[0], n);
	ret[1] = ldexp(a[1], n);
}

#define BNC_DD_IS_ZERO(a) ((a)[0] == 0.0 && (a)[1] == 0.0)
#define BNC_DD_IS_ONE(a)  ((a)[0] == 1.0 && (a)[1] == 0.0)

/* nearest integer of a double (QD's qd::nint; ties round away from zero
   via floor(d + 0.5) on the non-integer branch) */
static inline double bnc_d_nint(double d)
{
	if (d == floor(d))
		return d;
	return floor(d + 0.5);
}

/* ret := nearest integer to a (port of dtq dd_real nint(); the c_dd_nint
   in c_dd_qd.h is an empty stub and must not be used) */
static inline void bnc_dd_nint(const double *a, double *ret)
{
	double hi = bnc_d_nint(a[0]);
	double lo;

	if (hi == a[0]) {
		/* High word is an integer already.  Round the low word. */
		lo = bnc_d_nint(a[1]);
		hi = quick_two_sum(hi, lo, &lo);
	} else {
		/* High word is not an integer. */
		lo = 0.0;
		if (fabs(hi - a[0]) == 0.5 && a[1] < 0.0) {
			/* High word is exactly halfway between two integers,
			   and the low word is negative: round down.          */
			hi -= 1.0;
		}
	}

	ret[0] = hi;
	ret[1] = lo;
}

/* ret := a / b by FMA-driven long division (port of dtq-0.0.3
   dd_real::fma_div).  Same correction sequence as the classic accurate
   division, but each residual  r <- r - q * b  is one fused DW-FMA instead
   of a multiply followed by a subtraction, removing one renormalization per
   step.  The FMA is the div/sqrt-safe variant: the residuals entering it
   are not guaranteed non-overlapping, so no FastTwoSum precondition can be
   asserted (the safe variant costs the same and is never less accurate).  */
static inline void bnc_dd_div_fma(const double *a, const double *b, double *ret)
{
	double q1, q2, q3, r[2], t[2];

	q1 = a[0] / b[0];		/* approximate quotient */

	bnc_dwfma_safe(r, b, -q1, a);		/* r = a - q1 * b */

	q2 = r[0] / b[0];
	bnc_dwfma_safe(r, b, -q2, r);		/* r = r - q2 * b */

	q3 = r[0] / b[0];

	q1 = quick_two_sum(q1, q2, &q2);
	t[0] = q1; t[1] = q2;
	c_dd_add_dd_d(t, q3, ret);
}

/* ret := exp(x).  Argument reduction exp(kr + m log 2) = 2^m exp(r)^k with
   k = 512, then a short Taylor series whose terms are accumulated with one
   DW-FMA each, then nine repeated squarings s <- 2s + s^2.                  */
static inline void bnc_dd_exp(const double *a, double *ret)
{
	const double inv_k = 1.0 / 512.0;
	double m, r[2], s[2], p[2], t[2], u[2], thresh, tmag;
	int i, q;

	if (a[0] <= -709.0) {
		ret[0] = 0.0; ret[1] = 0.0;
		return;
	}
	if (a[0] >= 709.0) {
		ret[0] = HUGE_VAL; ret[1] = HUGE_VAL;
		return;
	}
	if (BNC_DD_IS_ZERO(a)) {
		ret[0] = 1.0; ret[1] = 0.0;
		return;
	}
	if (BNC_DD_IS_ONE(a)) {
		ret[0] = bnc_dd_e[0]; ret[1] = bnc_dd_e[1];
		return;
	}

	m = floor(a[0] / bnc_dd_log2[0] + 0.5);
	c_dd_mul_dd_d(bnc_dd_log2, m, t);
	c_dd_sub((double *)a, t, r);
	bnc_dd_mul_pwr2(r, inv_k, r);

	c_dd_sqr(r, p);
	bnc_dd_mul_pwr2(p, 0.5, t);
	c_dd_add(r, t, s);
	c_dd_mul(p, r, p);

	thresh = inv_k * bnc_dd_eps;
	i = 0;
	do {
		tmag = fabs(p[0] * bnc_dd_inv_fact[i][0]);
		bnc_dwfma(s, p, bnc_dd_inv_fact[i], s);
		c_dd_mul(p, r, p);
		++i;
	} while (tmag > thresh && i < 6);

	for (q = 0; q < 9; q++) {	/* 2^9 = k = 512 */
		bnc_dd_mul_pwr2(s, 2.0, t);
		c_dd_sqr(s, u);
		c_dd_add(t, u, s);
	}
	c_dd_add_dd_d(s, 1.0, s);

	bnc_dd_ldexp(s, (int)m, ret);
}

/* ret := exp(a) - 1, accurate for small |a| (no cancellation).
   For |a| <= log(2)/2 the ln2 reduction in exp() is a no-op (m = 0) and the
   repeated-squaring ladder s <- 2s + s^2 is exactly the doubling rule for
   u = exp(t) - 1, so expm1 is exp() without the final +1.                   */
static inline void bnc_dd_expm1(const double *a, double *ret)
{
	const double inv_k = 1.0 / 512.0;
	double r[2], s[2], p[2], t[2], u[2], thresh, tmag;
	int i, q;

	if (a[0] <= -709.0) {
		ret[0] = -1.0; ret[1] = 0.0;
		return;
	}
	if (a[0] >= 709.0) {
		ret[0] = HUGE_VAL; ret[1] = HUGE_VAL;
		return;
	}
	if (BNC_DD_IS_ZERO(a)) {
		ret[0] = 0.0; ret[1] = 0.0;
		return;
	}
	if (fabs(a[0]) > 0.346573590279972655) {	/* log(2)/2 */
		bnc_dd_exp(a, s);
		c_dd_sub_dd_d(s, 1.0, ret);
		return;
	}

	bnc_dd_mul_pwr2((double *)a, inv_k, r);

	c_dd_sqr(r, p);
	bnc_dd_mul_pwr2(p, 0.5, t);
	c_dd_add(r, t, s);
	c_dd_mul(p, r, p);

	thresh = inv_k * bnc_dd_eps;
	i = 0;
	do {
		tmag = fabs(p[0] * bnc_dd_inv_fact[i][0]);
		bnc_dwfma(s, p, bnc_dd_inv_fact[i], s);
		c_dd_mul(p, r, p);
		++i;
	} while (tmag > thresh && i < 6);

	for (q = 0; q < 9; q++) {
		bnc_dd_mul_pwr2(s, 2.0, t);
		c_dd_sqr(s, u);
		c_dd_add(t, u, s);
	}

	ret[0] = s[0]; ret[1] = s[1];	/* = exp(a) - 1, no +1 */
}

/* ret := log(x) (natural logarithm) by one Newton step on f(x) = exp(x) - a:
   x' = x + a exp(-x) - 1, started from the double-precision log.            */
static inline void bnc_dd_log(const double *a, double *ret)
{
	double x[2], mx[2], e[2], t[2];

	if (BNC_DD_IS_ONE(a)) {
		ret[0] = 0.0; ret[1] = 0.0;
		return;
	}
	/* MPFR/IEEE semantics, silently: log(0) = -inf, log(negative) = nan. */
	if (BNC_DD_IS_ZERO(a)) {
		ret[0] = -HUGE_VAL; ret[1] = -HUGE_VAL;
		return;
	}
	if (a[0] < 0.0) {
		ret[0] = NAN; ret[1] = NAN;
		return;
	}

	x[0] = log(a[0]);	/* initial approximation */
	x[1] = 0.0;

	mx[0] = -x[0]; mx[1] = -x[1];
	bnc_dd_exp(mx, e);
	c_dd_mul((double *)a, e, t);
	c_dd_add(x, t, x);
	c_dd_sub_dd_d(x, 1.0, ret);
}

/* ret := log10(x) */
static inline void bnc_dd_log10(const double *a, double *ret)
{
	double l[2];

	bnc_dd_log(a, l);
	c_dd_div(l, (double *)bnc_dd_log10_tbl, ret);
}

/* sin(a) by Taylor series; assumes |a| <= pi/32. */
static inline void bnc_dd_sin_taylor(const double *a, double *ret)
{
	double thresh, tmag, r[2], s[2], x[2];
	int i;

	if (BNC_DD_IS_ZERO(a)) {
		ret[0] = 0.0; ret[1] = 0.0;
		return;
	}

	thresh = 0.5 * fabs(a[0]) * bnc_dd_eps;
	c_dd_sqr((double *)a, x);
	x[0] = -x[0]; x[1] = -x[1];
	s[0] = a[0]; s[1] = a[1];
	r[0] = a[0]; r[1] = a[1];
	i = 0;
	do {
		c_dd_mul(r, x, r);
		/* s += r / i!, fused; term magnitude from the leading words */
		tmag = fabs(r[0] * bnc_dd_inv_fact[i][0]);
		bnc_dwfma(s, r, bnc_dd_inv_fact[i], s);
		i += 2;
	} while (i < BNC_ELEM_N_INV_FACT && tmag > thresh);

	ret[0] = s[0]; ret[1] = s[1];
}

/* cos(a) by Taylor series; assumes |a| <= pi/32. */
static inline void bnc_dd_cos_taylor(const double *a, double *ret)
{
	double thresh, tmag, r[2], s[2], x[2], t[2];
	int i;

	if (BNC_DD_IS_ZERO(a)) {
		ret[0] = 1.0; ret[1] = 0.0;
		return;
	}

	thresh = 0.5 * bnc_dd_eps;
	c_dd_sqr((double *)a, x);
	x[0] = -x[0]; x[1] = -x[1];
	r[0] = x[0]; r[1] = x[1];
	bnc_dd_mul_pwr2(r, 0.5, t);
	c_dd_add_dd_d(t, 1.0, s);
	i = 1;
	do {
		c_dd_mul(r, x, r);
		tmag = fabs(r[0] * bnc_dd_inv_fact[i][0]);
		bnc_dwfma(s, r, bnc_dd_inv_fact[i], s);
		i += 2;
	} while (i < BNC_ELEM_N_INV_FACT && tmag > thresh);

	ret[0] = s[0]; ret[1] = s[1];
}

static inline void bnc_dd_sincos_taylor(const double *a, double *sin_a, double *cos_a)
{
	double s2[2], t[2];

	if (BNC_DD_IS_ZERO(a)) {
		sin_a[0] = 0.0; sin_a[1] = 0.0;
		cos_a[0] = 1.0; cos_a[1] = 0.0;
		return;
	}

	bnc_dd_sin_taylor(a, sin_a);
	c_dd_sqr(sin_a, s2);
	t[0] = 1.0; t[1] = 0.0;
	c_dd_sub(t, s2, t);
	c_dd_sqrt(t, cos_a);
}

/* Argument reduction shared by sin / cos / sincos:
   a = t + j * (pi/2) + k * (pi/16), |t| <= pi/32.
   Returns 0 on success, -1 if the reduction fails (|a| too large).          */
static inline int bnc_dd_sincos_reduce(const double *a, double *t, int *jj, int *kk)
{
	double z[2], r[2], w[2], q;
	int j, k;

	/* approximately reduce modulo 2*pi */
	c_dd_div((double *)a, (double *)bnc_dd_two_pi, z);
	bnc_dd_nint(z, z);
	c_dd_mul((double *)bnc_dd_two_pi, z, w);
	c_dd_sub((double *)a, w, r);

	/* approximately reduce modulo pi/2 and then modulo pi/16 */
	q = floor(r[0] / bnc_dd_pi2[0] + 0.5);
	c_dd_mul_dd_d(bnc_dd_pi2, q, w);
	c_dd_sub(r, w, t);
	j = (int)q;
	q = floor(t[0] / bnc_dd_pi16[0] + 0.5);
	c_dd_mul_dd_d(bnc_dd_pi16, q, w);
	c_dd_sub(t, w, t);
	k = (int)q;

	if (j < -2 || j > 2) {
		fprintf(stderr, "(bnc_dd_sincos): Cannot reduce modulo pi/2.\n");
		return -1;
	}
	if (k < -4 || k > 4) {
		fprintf(stderr, "(bnc_dd_sincos): Cannot reduce modulo pi/16.\n");
		return -1;
	}

	*jj = j; *kk = k;
	return 0;
}

/* ret := sin(a) */
static inline void bnc_dd_sin(const double *a, double *ret)
{
	double t[2], u[2], v[2], sin_t[2], cos_t[2], p1[2], p2[2];
	int j, k, abs_k;

	if (BNC_DD_IS_ZERO(a)) {
		ret[0] = 0.0; ret[1] = 0.0;
		return;
	}

	if (bnc_dd_sincos_reduce(a, t, &j, &k) != 0) {
		ret[0] = NAN; ret[1] = NAN;
		return;
	}
	abs_k = (k < 0) ? -k : k;

	if (k == 0) {
		switch (j) {
			case 0:
				bnc_dd_sin_taylor(t, ret);
				return;
			case 1:
				bnc_dd_cos_taylor(t, ret);
				return;
			case -1:
				bnc_dd_cos_taylor(t, ret);
				ret[0] = -ret[0]; ret[1] = -ret[1];
				return;
			default:
				bnc_dd_sin_taylor(t, ret);
				ret[0] = -ret[0]; ret[1] = -ret[1];
				return;
		}
	}

	u[0] = bnc_dd_cos_table[abs_k - 1][0]; u[1] = bnc_dd_cos_table[abs_k - 1][1];
	v[0] = bnc_dd_sin_table[abs_k - 1][0]; v[1] = bnc_dd_sin_table[abs_k - 1][1];
	bnc_dd_sincos_taylor(t, sin_t, cos_t);

	c_dd_mul(u, sin_t, p1);	/* p1 = u * sin_t */
	c_dd_mul(v, cos_t, p2);	/* p2 = v * cos_t */
	if (j == 0) {
		if (k > 0) {
			c_dd_add(p1, p2, ret);	/* u sin_t + v cos_t */
		} else {
			c_dd_sub(p1, p2, ret);	/* u sin_t - v cos_t */
		}
	} else if (j == 1) {
		c_dd_mul(u, cos_t, p1);
		c_dd_mul(v, sin_t, p2);
		if (k > 0) {
			c_dd_sub(p1, p2, ret);	/* u cos_t - v sin_t */
		} else {
			c_dd_add(p1, p2, ret);	/* u cos_t + v sin_t */
		}
	} else if (j == -1) {
		c_dd_mul(u, cos_t, p1);
		c_dd_mul(v, sin_t, p2);
		if (k > 0) {
			c_dd_sub(p2, p1, ret);	/* v sin_t - u cos_t */
		} else {
			c_dd_add(p1, p2, ret);	/* -(u cos_t + v sin_t) */
			ret[0] = -ret[0]; ret[1] = -ret[1];
		}
	} else {
		if (k > 0) {
			c_dd_add(p1, p2, ret);	/* -(u sin_t + v cos_t) */
			ret[0] = -ret[0]; ret[1] = -ret[1];
		} else {
			c_dd_sub(p2, p1, ret);	/* v cos_t - u sin_t */
		}
	}
}

/* ret := cos(a) */
static inline void bnc_dd_cos(const double *a, double *ret)
{
	double t[2], u[2], v[2], sin_t[2], cos_t[2], p1[2], p2[2];
	int j, k, abs_k;

	if (BNC_DD_IS_ZERO(a)) {
		ret[0] = 1.0; ret[1] = 0.0;
		return;
	}

	if (bnc_dd_sincos_reduce(a, t, &j, &k) != 0) {
		ret[0] = NAN; ret[1] = NAN;
		return;
	}
	abs_k = (k < 0) ? -k : k;

	if (k == 0) {
		switch (j) {
			case 0:
				bnc_dd_cos_taylor(t, ret);
				return;
			case 1:
				bnc_dd_sin_taylor(t, ret);
				ret[0] = -ret[0]; ret[1] = -ret[1];
				return;
			case -1:
				bnc_dd_sin_taylor(t, ret);
				return;
			default:
				bnc_dd_cos_taylor(t, ret);
				ret[0] = -ret[0]; ret[1] = -ret[1];
				return;
		}
	}

	bnc_dd_sincos_taylor(t, sin_t, cos_t);
	u[0] = bnc_dd_cos_table[abs_k - 1][0]; u[1] = bnc_dd_cos_table[abs_k - 1][1];
	v[0] = bnc_dd_sin_table[abs_k - 1][0]; v[1] = bnc_dd_sin_table[abs_k - 1][1];

	if (j == 0) {
		c_dd_mul(u, cos_t, p1);
		c_dd_mul(v, sin_t, p2);
		if (k > 0) {
			c_dd_sub(p1, p2, ret);	/* u cos_t - v sin_t */
		} else {
			c_dd_add(p1, p2, ret);	/* u cos_t + v sin_t */
		}
	} else if (j == 1) {
		c_dd_mul(u, sin_t, p1);
		c_dd_mul(v, cos_t, p2);
		if (k > 0) {
			c_dd_add(p1, p2, ret);	/* -(u sin_t + v cos_t) */
			ret[0] = -ret[0]; ret[1] = -ret[1];
		} else {
			c_dd_sub(p2, p1, ret);	/* v cos_t - u sin_t */
		}
	} else if (j == -1) {
		c_dd_mul(u, sin_t, p1);
		c_dd_mul(v, cos_t, p2);
		if (k > 0) {
			c_dd_add(p1, p2, ret);	/* u sin_t + v cos_t */
		} else {
			c_dd_sub(p1, p2, ret);	/* u sin_t - v cos_t */
		}
	} else {
		c_dd_mul(v, sin_t, p1);
		c_dd_mul(u, cos_t, p2);
		if (k > 0) {
			c_dd_sub(p1, p2, ret);	/* v sin_t - u cos_t */
		} else {
			c_dd_add(p2, p1, ret);	/* -(u cos_t + v sin_t) */
			ret[0] = -ret[0]; ret[1] = -ret[1];
		}
	}
}

/* sin_a := sin(a), cos_a := cos(a) simultaneously */
static inline void bnc_dd_sincos(const double *a, double *sin_a, double *cos_a)
{
	double t[2], u[2], v[2], sin_t[2], cos_t[2], s[2], c[2], p1[2], p2[2];
	int j, k, abs_j, abs_k;

	if (BNC_DD_IS_ZERO(a)) {
		sin_a[0] = 0.0; sin_a[1] = 0.0;
		cos_a[0] = 1.0; cos_a[1] = 0.0;
		return;
	}

	if (bnc_dd_sincos_reduce(a, t, &j, &k) != 0) {
		sin_a[0] = NAN; sin_a[1] = NAN;
		cos_a[0] = NAN; cos_a[1] = NAN;
		return;
	}
	abs_j = (j < 0) ? -j : j;
	abs_k = (k < 0) ? -k : k;

	bnc_dd_sincos_taylor(t, sin_t, cos_t);

	if (abs_k == 0) {
		s[0] = sin_t[0]; s[1] = sin_t[1];
		c[0] = cos_t[0]; c[1] = cos_t[1];
	} else {
		u[0] = bnc_dd_cos_table[abs_k - 1][0]; u[1] = bnc_dd_cos_table[abs_k - 1][1];
		v[0] = bnc_dd_sin_table[abs_k - 1][0]; v[1] = bnc_dd_sin_table[abs_k - 1][1];
		c_dd_mul(u, sin_t, p1);
		c_dd_mul(v, cos_t, p2);
		if (k > 0) {
			c_dd_add(p1, p2, s);	/* u sin_t + v cos_t */
		} else {
			c_dd_sub(p1, p2, s);	/* u sin_t - v cos_t */
		}
		c_dd_mul(u, cos_t, p1);
		c_dd_mul(v, sin_t, p2);
		if (k > 0) {
			c_dd_sub(p1, p2, c);	/* u cos_t - v sin_t */
		} else {
			c_dd_add(p1, p2, c);	/* u cos_t + v sin_t */
		}
	}

	if (abs_j == 0) {
		sin_a[0] = s[0]; sin_a[1] = s[1];
		cos_a[0] = c[0]; cos_a[1] = c[1];
	} else if (j == 1) {
		sin_a[0] = c[0]; sin_a[1] = c[1];
		cos_a[0] = -s[0]; cos_a[1] = -s[1];
	} else if (j == -1) {
		sin_a[0] = -c[0]; sin_a[1] = -c[1];
		cos_a[0] = s[0]; cos_a[1] = s[1];
	} else {
		sin_a[0] = -s[0]; sin_a[1] = -s[1];
		cos_a[0] = -c[0]; cos_a[1] = -c[1];
	}
}

#endif /* __BNC_ELEM_DD_H */
