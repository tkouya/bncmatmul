/********************************************************************************/
/* bench/elem/elem_vec_bench.c :                                                */
/*   Scalar vs SIMD-vectorized elementary functions (bnc_*_array from           */
/*   src/elem_vector.c) — accuracy cross-check and ns/element timing.           */
/*                                                                              */
/* Build (from repository root), e.g. NEON:                                     */
/*   gcc -O3 -ffp-contract=off -DBNC_ENABLE_NEON -Iinclude \                    */
/*       bench/elem/elem_vec_bench.c src/elem_vector.c \                        */
/*       -o bench/elem/elem_vec_bench_neon -lm                                  */
/*                                                                              */
/* Copyright (C) 2026 Tomonori Kouya, GNU LGPL v3                               */
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include <mpfr.h>

#include "c_dd_qd.h"
#include "bncelem_vector.h"

#define N 100000L

static double timer_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec * 1.0e9 + (double)ts.tv_nsec;
}

static double drand_range(double lo, double hi)
{
	return lo + (hi - lo) * ((double)rand() / RAND_MAX);
}

typedef void (*sc_fn_t)(const double *, double *);
typedef void (*arr_fn_t)(long, double * const *, double * const *);

static void bench_one(const char *label, int k, double eps,
                      sc_fn_t sc, arr_fn_t arr, double lo, double hi, int logspace)
{
	static double xbuf[4][N], rs[4][N], rv[4][N];
	double *x[4], *r1[4], *r2[4];
	double in[4], out[4], t0, t_sc, t_vec, maxerr = 0.0, err, denom;
	long i;
	int j;

	for (j = 0; j < k; j++) {
		x[j] = xbuf[j]; r1[j] = rs[j]; r2[j] = rv[j];
	}
	for (i = 0; i < N; i++) {
		double v = drand_range(lo, hi);
		if (logspace)
			v = exp(v);
		x[0][i] = v;
		x[1][i] = v * 1.1e-17;
		if (k >= 3) x[2][i] = v * 1.3e-34;
		if (k >= 4) x[3][i] = v * 1.7e-51;
	}

	/* scalar */
	t0 = timer_ns();
	for (i = 0; i < N; i++) {
		for (j = 0; j < k; j++) in[j] = x[j][i];
		sc(in, out);
		for (j = 0; j < k; j++) r1[j][i] = out[j];
	}
	t_sc = (timer_ns() - t0) / N;

	/* vector */
	t0 = timer_ns();
	arr(N, r2, x);
	t_vec = (timer_ns() - t0) / N;

	/* accuracy: |vec - scalar| / |scalar| in eps units.  The difference is
	   accumulated in MPFR because the two results may be equal in value
	   but normalized differently (a plain double sum of the limb-wise
	   differences then loses the residual and reports a spurious error). */
	{
		mpfr_t ds, dv;

		mpfr_init2(ds, 512);
		mpfr_init2(dv, 512);
		for (i = 0; i < N; i++) {
			double m = fabs(r1[0][i]);

			mpfr_set_d(ds, r1[0][i], MPFR_RNDN);
			mpfr_set_d(dv, r2[0][i], MPFR_RNDN);
			for (j = 1; j < k; j++) {
				mpfr_add_d(ds, ds, r1[j][i], MPFR_RNDN);
				mpfr_add_d(dv, dv, r2[j][i], MPFR_RNDN);
			}
			mpfr_sub(dv, dv, ds, MPFR_RNDN);
			denom = (m > 0.0) ? m : 1.0;
			err = fabs(mpfr_get_d(dv, MPFR_RNDN)) / denom / eps;
			if (err > maxerr) maxerr = err;
		}
		mpfr_clears(ds, dv, (mpfr_ptr)0);
	}

	printf("%-14s scalar %8.1f ns/el   vector %8.1f ns/el   speedup %5.2fx   max|vec-scalar| %6.2f eps\n",
	       label, t_sc, t_vec, t_sc / t_vec, maxerr);
}

int main(void)
{
	srand(20260814);
	printf("BNCmatmul vectorized elementary functions (n=%ld)\n", N);
#if defined(__AVX512F__)
	printf("SIMD path: AVX-512 (W=8)\n\n");
#elif defined(__AVX2__)
	printf("SIMD path: AVX2 (W=4)\n\n");
#elif defined(__ARM_NEON)
	printf("SIMD path: NEON (W=2)\n\n");
#else
	printf("SIMD path: none (serial)\n\n");
#endif

	bench_one("dd exp", 2, 4.93038065763132e-32, bnc_dd_exp, bnc_dd_exp_array, -20.0, 20.0, 0);
	bench_one("dd log", 2, 4.93038065763132e-32, bnc_dd_log, bnc_dd_log_array, -7.0, 7.0, 1);
	bench_one("dd sin", 2, 4.93038065763132e-32, bnc_dd_sin, bnc_dd_sin_array, -50.0, 50.0, 0);
	bench_one("dd cos", 2, 4.93038065763132e-32, bnc_dd_cos, bnc_dd_cos_array, -50.0, 50.0, 0);
	printf("\n");
	bench_one("td exp", 3, 1.0947644252537636e-47, bnc_td_exp, bnc_td_exp_array, -20.0, 20.0, 0);
	bench_one("td log", 3, 1.0947644252537636e-47, bnc_td_log, bnc_td_log_array, -7.0, 7.0, 1);
	bench_one("td sin", 3, 1.0947644252537636e-47, bnc_td_sin, bnc_td_sin_array, -50.0, 50.0, 0);
	bench_one("td cos", 3, 1.0947644252537636e-47, bnc_td_cos, bnc_td_cos_array, -50.0, 50.0, 0);
	printf("\n");
	bench_one("qd exp", 4, 1.21543267145725e-63, bnc_qd_exp, bnc_qd_exp_array, -20.0, 20.0, 0);
	bench_one("qd log", 4, 1.21543267145725e-63, bnc_qd_log, bnc_qd_log_array, -7.0, 7.0, 1);
	bench_one("qd sin", 4, 1.21543267145725e-63, bnc_qd_sin, bnc_qd_sin_array, -50.0, 50.0, 0);
	bench_one("qd cos", 4, 1.21543267145725e-63, bnc_qd_cos, bnc_qd_cos_array, -50.0, 50.0, 0);

	return 0;
}
