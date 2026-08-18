/********************************************************************************/
/* bench/fma/test_fma_safe.c :                                                  */
/*   1. Demonstrates why the div/sqrt-safe FMA variants exist: feeds the        */
/*      standard (FastTwoSum-based) and safe (all-TwoSum) kernels operands      */
/*      whose limbs OVERLAP (i.e. not normalized expansions), as the Newton     */
/*      residuals of division do, and compares both against MPFR.               */
/*   2. Validates the FMA-driven divisions bnc_{dd,td,qd,ds,ts,qs}_div_fma      */
/*      against MPFR and times them against the classic divisions.              */
/*                                                                              */
/* Build (from repository root):                                                */
/*   gcc -O2 -ffp-contract=off -Iinclude -I/usr/local/include \                 */
/*       bench/fma/test_fma_safe.c -o bench/fma/test_fma_safe \                 */
/*       -L/usr/local/lib -lmpfr -lgmp -lm                                      */
/*                                                                              */
/* Copyright (C) 2026 Tomonori Kouya, GNU LGPL v3                               */
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <mpfr.h>

#include "c_dd_qd.h"	/* pulls in bncfma_d.h and bncelem_*.h */

#define NTRIAL 100000
#define DD_EPS 4.93038065763132e-32
#define TD_EPS 1.0947644252537636e-47
#define QD_EPS 1.21543267145725e-63

static double drand1(void) { return (double)rand() / RAND_MAX * 2.0 - 1.0; }

static double timer_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec * 1.0e9 + (double)ts.tv_nsec;
}

static void mpfr_from_limbs(mpfr_t ret, const double *x, int k)
{
	int i;

	mpfr_set_d(ret, x[0], MPFR_RNDN);
	for (i = 1; i < k; i++)
		mpfr_add_d(ret, ret, x[i], MPFR_RNDN);
}

/* |val(x) - ref| / |ref| in units of eps */
static double relerr_eps(const double *x, int k, mpfr_t ref, double eps,
                         mpfr_t t1, mpfr_t t2)
{
	mpfr_from_limbs(t1, x, k);
	mpfr_sub(t2, t1, ref, MPFR_RNDN);
	if (mpfr_zero_p(ref))
		return mpfr_zero_p(t2) ? 0.0 : HUGE_VAL;
	mpfr_div(t2, t2, ref, MPFR_RNDN);
	mpfr_abs(t2, t2, MPFR_RNDN);
	return mpfr_get_d(t2, MPFR_RNDN) / eps;
}

/* ---- 1. standard vs safe FMA on UNNORMALIZED operands ----
   The safe variants exist because for the Newton residuals of division no
   FastTwoSum precondition can be *proved*; this test probes several
   adversarial operand shapes (overlapping limbs, near-total cancellation)
   and reports both kernels' errors against MPFR.                          */
static void overlap_test(void)
{
	static const char *fam_name[3] = {
		"overlapping c (|c1| ~ |c0|)          ",
		"cancellation  c ~ -x*y, c1 overlaps  ",
		"cancellation + perturbed low limb    ",
	};
	mpfr_t ref, prod, t1, t2;
	double x[2], y, c[2], zs[2], zn[2], yv[2], p[2];
	double err_std, err_safe, e;
	int i, fam;

	mpfr_inits2(512, ref, prod, t1, t2, (mpfr_ptr)0);

	printf("[1] DW-FMA on unnormalized operands (%d trials per family)\n", NTRIAL);
	for (fam = 0; fam < 3; fam++) {
		err_std = 0.0; err_safe = 0.0;
		for (i = 0; i < NTRIAL; i++) {
			x[0] = drand1(); x[1] = drand1() * 1e-17;
			y    = drand1();
			c_dd_mul_dd_d(x, y, p);	/* p ~ x*y */
			switch (fam) {
			case 0:	/* overlapping limbs, no cancellation */
				c[0] = drand1() * 1e-16;
				c[1] = drand1() * fabs(c[0]);
				break;
			case 1:	/* near-total cancellation of the head */
				c[0] = -p[0];
				c[1] = drand1() * fabs(p[1]) * 2.0;
				break;
			default: /* cancellation of both words + overlap */
				c[0] = -p[0];
				c[1] = -p[1] * (1.0 + drand1() * 1e-3);
				break;
			}

			mpfr_from_limbs(t1, x, 2);
			mpfr_mul_d(prod, t1, y, MPFR_RNDN);
			mpfr_from_limbs(t1, c, 2);
			mpfr_add(ref, prod, t1, MPFR_RNDN);

			yv[0] = y; yv[1] = 0.0;
			bnc_dwfma(zn, x, yv, c);	/* standard kernel */
			bnc_dwfma_safe(zs, x, y, c);	/* safe kernel */

			/* absolute error scaled by |xy| + |c| (the certified bound's
			   scaling), in eps units */
			mpfr_from_limbs(t1, zn, 2);
			mpfr_sub(t2, t1, ref, MPFR_RNDN);
			e = fabs(mpfr_get_d(t2, MPFR_RNDN)) / (fabs(p[0]) + fabs(c[0]) + 1e-300) / DD_EPS;
			if (e > err_std) err_std = e;
			mpfr_from_limbs(t1, zs, 2);
			mpfr_sub(t2, t1, ref, MPFR_RNDN);
			e = fabs(mpfr_get_d(t2, MPFR_RNDN)) / (fabs(p[0]) + fabs(c[0]) + 1e-300) / DD_EPS;
			if (e > err_safe) err_safe = e;
		}
		printf("    %s std %10.4f  safe %10.4f  [eps(dd)*(|xy|+|c|)]\n",
		       fam_name[fam], err_std, err_safe);
	}
	printf("\n");
	mpfr_clears(ref, prod, t1, t2, (mpfr_ptr)0);
}

/* ---- 2. FMA-driven division: accuracy vs MPFR + speed vs classic div ---- */
typedef void (*divfn_t)(const double *, const double *, double *);

static void div_test(const char *label, int k, double eps,
                     divfn_t classic, divfn_t fmadiv)
{
	static double a[NTRIAL / 10][4], b[NTRIAL / 10][4], q[4];
	mpfr_t ref, ma, mb, t1, t2;
	double errc = 0.0, errf = 0.0, e, t0, tc, tf;
	int n = NTRIAL / 10, i, r;

	mpfr_inits2(512, ref, ma, mb, t1, t2, (mpfr_ptr)0);

	for (i = 0; i < n; i++) {
		memset(a[i], 0, sizeof(a[i]));
		memset(b[i], 0, sizeof(b[i]));
		a[i][0] = drand1() * 100.0;
		a[i][1] = a[i][0] * 1.1e-17;
		if (k >= 3) a[i][2] = a[i][0] * 1.3e-34;
		if (k >= 4) a[i][3] = a[i][0] * 1.7e-51;
		do { b[i][0] = drand1() * 100.0; } while (fabs(b[i][0]) < 1e-3);
		b[i][1] = b[i][0] * 0.9e-17;
		if (k >= 3) b[i][2] = b[i][0] * 1.2e-34;
		if (k >= 4) b[i][3] = b[i][0] * 1.5e-51;
	}

	for (i = 0; i < n; i++) {
		mpfr_from_limbs(ma, a[i], k);
		mpfr_from_limbs(mb, b[i], k);
		mpfr_div(ref, ma, mb, MPFR_RNDN);

		classic(a[i], b[i], q);
		e = relerr_eps(q, k, ref, eps, t1, t2);
		if (e > errc) errc = e;

		fmadiv(a[i], b[i], q);
		e = relerr_eps(q, k, ref, eps, t1, t2);
		if (e > errf) errf = e;
	}

	t0 = timer_ns();
	for (r = 0; r < 10; r++)
		for (i = 0; i < n; i++)
			classic(a[i], b[i], q);
	tc = (timer_ns() - t0) / (10.0 * n);

	t0 = timer_ns();
	for (r = 0; r < 10; r++)
		for (i = 0; i < n; i++)
			fmadiv(a[i], b[i], q);
	tf = (timer_ns() - t0) / (10.0 * n);

	printf("%-8s classic: max %8.3f eps %7.1f ns | fma_div: max %8.3f eps %7.1f ns | speedup %.2fx\n",
	       label, errc, tc, errf, tf, tc / tf);

	mpfr_clears(ref, ma, mb, t1, t2, (mpfr_ptr)0);
}

static void dd_div_classic(const double *a, const double *b, double *c) { c_dd_div(a, b, c); }
static void td_div_classic(const double *a, const double *b, double *c) { c_td_divq((double *)a, (double *)b, c); }
static void qd_div_classic(const double *a, const double *b, double *c) { c_qd_div(a, b, c); }

int main(void)
{
	srand(20260815);

	overlap_test();

	printf("[2] FMA-driven division vs classic (MPFR 512-bit reference)\n");
	div_test("dd div", 2, DD_EPS, dd_div_classic, bnc_dd_div_fma);
	div_test("td div", 3, TD_EPS, td_div_classic, bnc_td_div_fma);
	div_test("qd div", 4, QD_EPS, qd_div_classic, bnc_qd_div_fma);

	return 0;
}
