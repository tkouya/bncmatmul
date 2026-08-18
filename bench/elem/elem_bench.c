/********************************************************************************/
/* bench/elem/elem_bench.c :                                                    */
/*   Accuracy (vs MPFR) and performance benchmark of the dtq-0.0.3 FMA-based    */
/*   elementary functions ported to plain C (include/bncelem*.h), compared      */
/*   with the previous BNCmatmul implementations where one existed              */
/*   (c_dd_exp_orig / c_dd_log_orig; sin/cos and all TD/QD functions were       */
/*   empty stubs or missing before).                                            */
/*                                                                              */
/* Build (from repository root):                                                */
/*   gcc -O2 -ffp-contract=off -Iinclude bench/elem/elem_bench.c \              */
/*       -o bench/elem/elem_bench -lmpfr -lgmp -lm                              */
/*                                                                              */
/* Copyright (C) 2026 Tomonori Kouya, GNU LGPL v3                               */
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <mpfr.h>

#include "c_dd_qd.h"	/* pulls in bncelem.h (bnc_dd_* / bnc_td_* / bnc_qd_*) */

#define REF_PREC 512
#define NSAMPLE 4096

static double drand_range(double lo, double hi)
{
	return lo + (hi - lo) * ((double)rand() / RAND_MAX);
}

static double timer_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec * 1.0e9 + (double)ts.tv_nsec;
}

/* ret := sum of k limbs */
static void mpfr_from_limbs(mpfr_t ret, const double *x, int k)
{
	int i;

	mpfr_set_d(ret, x[0], MPFR_RNDN);
	for (i = 1; i < k; i++)
		mpfr_add_d(ret, ret, x[i], MPFR_RNDN);
}

/* relative error of the k-limb value x against ref, in units of eps */
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

typedef void (*fn1_t)(const double *, double *);
typedef int (*mpfr_fn_t)(mpfr_t, const mpfr_t, mpfr_rnd_t);

/* one accuracy+speed measurement:
   type klimb/eps, function under test, MPFR reference function, samples */
static void bench1(const char *label, int k, double eps,
                   fn1_t f, mpfr_fn_t ref_f,
                   const double *args, int nargs, int reps)
{
	static double in[NSAMPLE][4], out[NSAMPLE][4];
	mpfr_t ref, t1, t2, a;
	double maxerr = 0.0, sumerr = 0.0, err, t0, t;
	int i, r;
	volatile double sink = 0.0;

	mpfr_init2(ref, REF_PREC);
	mpfr_init2(t1, REF_PREC);
	mpfr_init2(t2, REF_PREC);
	mpfr_init2(a, REF_PREC);

	for (i = 0; i < nargs; i++) {
		memset(in[i], 0, sizeof(in[i]));
		in[i][0] = args[i];
		/* give the low limbs some mass so the test covers full precision */
		in[i][1] = args[i] * 1.1e-17;
		if (k >= 3) in[i][2] = args[i] * 1.3e-34;
		if (k >= 4) in[i][3] = args[i] * 1.7e-51;
	}

	/* accuracy */
	for (i = 0; i < nargs; i++) {
		f(in[i], out[i]);
		mpfr_from_limbs(a, in[i], k);
		ref_f(ref, a, MPFR_RNDN);
		err = relerr_eps(out[i], k, ref, eps, t1, t2);
		if (err > maxerr)
			maxerr = err;
		sumerr += err;
	}

	/* speed */
	t0 = timer_ns();
	for (r = 0; r < reps; r++)
		for (i = 0; i < nargs; i++)
			f(in[i], out[i]);
	t = timer_ns() - t0;
	for (i = 0; i < nargs; i++)
		sink += out[i][0];

	printf("%-24s  max err %10.3f eps   avg err %8.3f eps   %9.1f ns/call\n",
	       label, maxerr, sumerr / nargs, t / ((double)reps * nargs));

	(void)sink;
	mpfr_clears(ref, t1, t2, a, (mpfr_ptr)0);
}

int main(void)
{
	static double xs_exp[NSAMPLE], xs_log[NSAMPLE], xs_trig[NSAMPLE];
	int i;

	srand(20260814);
	for (i = 0; i < NSAMPLE; i++) {
		xs_exp[i] = drand_range(-20.0, 20.0);
		xs_log[i] = exp(drand_range(-7.0, 7.0));	/* log-uniform in [1e-3,1e3] */
		xs_trig[i] = drand_range(-50.0, 50.0);
	}

	printf("BNCmatmul elementary functions: accuracy vs MPFR(%d bits), %d samples\n",
	       REF_PREC, NSAMPLE);
	printf("eps(dd) = 2^-104, eps(td) = 2^-157, eps(qd) = 2^-209\n\n");

	printf("---- DD (double-double) ----\n");
	bench1("dd exp  (orig, Taylor)", 2, 4.93038065763132e-32, c_dd_exp_orig, mpfr_exp, xs_exp, NSAMPLE, 4);
	bench1("dd exp  (new, FMA)",     2, 4.93038065763132e-32, bnc_dd_exp,    mpfr_exp, xs_exp, NSAMPLE, 64);
	bench1("dd log  (orig)",         2, 4.93038065763132e-32, c_dd_log_orig, mpfr_log, xs_log, NSAMPLE, 4);
	bench1("dd log  (new, FMA)",     2, 4.93038065763132e-32, bnc_dd_log,    mpfr_log, xs_log, NSAMPLE, 64);
	bench1("dd sin  (new, FMA)",     2, 4.93038065763132e-32, bnc_dd_sin,    mpfr_sin, xs_trig, NSAMPLE, 64);
	bench1("dd cos  (new, FMA)",     2, 4.93038065763132e-32, bnc_dd_cos,    mpfr_cos, xs_trig, NSAMPLE, 64);
	printf("(dd sin/cos had no previous C implementation: empty stubs)\n\n");

	printf("---- TD (triple-double) ----\n");
	bench1("td exp  (new, FMA)", 3, 1.0947644252537636e-47, bnc_td_exp, mpfr_exp, xs_exp, NSAMPLE, 32);
	bench1("td log  (new, FMA)", 3, 1.0947644252537636e-47, bnc_td_log, mpfr_log, xs_log, NSAMPLE, 32);
	bench1("td sin  (new, FMA)", 3, 1.0947644252537636e-47, bnc_td_sin, mpfr_sin, xs_trig, NSAMPLE, 32);
	bench1("td cos  (new, FMA)", 3, 1.0947644252537636e-47, bnc_td_cos, mpfr_cos, xs_trig, NSAMPLE, 32);
	printf("(all td elementary functions were previously missing in C)\n\n");

	printf("---- QD (quad-double) ----\n");
	bench1("qd exp  (new, FMA)", 4, 1.21543267145725e-63, bnc_qd_exp, mpfr_exp, xs_exp, NSAMPLE, 16);
	bench1("qd log  (new, FMA)", 4, 1.21543267145725e-63, bnc_qd_log, mpfr_log, xs_log, NSAMPLE, 16);
	bench1("qd sin  (new, FMA)", 4, 1.21543267145725e-63, bnc_qd_sin, mpfr_sin, xs_trig, NSAMPLE, 16);
	bench1("qd cos  (new, FMA)", 4, 1.21543267145725e-63, bnc_qd_cos, mpfr_cos, xs_trig, NSAMPLE, 16);
	printf("(all qd elementary functions were previously empty stubs)\n");

	return 0;
}
