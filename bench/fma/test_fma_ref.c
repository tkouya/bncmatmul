/*
 * test_fma_ref.c : correctness / certified-error-bound check of the proposed
 *                  branch-free FMA (arXiv:2607.11391v1, Algorithms 1-3)
 *                  against MPFR at 600 bits.
 *
 * Checks, for DD/TD/QD (and DS/TS/QS on request):
 *   eta = |z - (xy+c)| / (|xy| + |c|) / u^K   <=   34 / 184 / 812
 * and compares the accuracy with the existing Q (mul+add) and BF
 * (branch-free mul_bf+add_bf) variants.
 *
 * build: gcc -O3 -ffp-contract=off -Iinclude bench/fma/test_fma_ref.c \
 *            -I/usr/local/include -lmpfr -lgmp -lm -o bench/fma/test_fma_ref
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpfr.h>

#include "bncfma.h"

#define REF_PREC 600

/* ---- multi-word <-> mpfr -------------------------------------------------- */
static void mw_to_mpfr(mpfr_t r, const double *z, int K)
{
	int i;
	mpfr_set_d(r, z[0], MPFR_RNDN);
	for (i = 1; i < K; i++) mpfr_add_d(r, r, z[i], MPFR_RNDN);
}
static void mpfr_to_mw(double *z, mpfr_t v, int K)
{
	int i;
	mpfr_t t;
	mpfr_init2(t, REF_PREC);
	mpfr_set(t, v, MPFR_RNDN);
	for (i = 0; i < K; i++) {
		z[i] = mpfr_get_d(t, MPFR_RNDN);
		mpfr_sub_d(t, t, z[i], MPFR_RNDN);
	}
	mpfr_clear(t);
}
static void mwf_to_mpfr(mpfr_t r, const float *z, int K)
{
	int i;
	mpfr_set_flt(r, z[0], MPFR_RNDN);
	for (i = 1; i < K; i++) { mpfr_t t; mpfr_init2(t, REF_PREC);
		mpfr_set_flt(t, z[i], MPFR_RNDN); mpfr_add(r, r, t, MPFR_RNDN); mpfr_clear(t); }
}
static void mpfr_to_mwf(float *z, mpfr_t v, int K)
{
	int i;
	mpfr_t t;
	mpfr_init2(t, REF_PREC);
	mpfr_set(t, v, MPFR_RNDN);
	for (i = 0; i < K; i++) {
		mpfr_t s; mpfr_init2(s, REF_PREC);
		z[i] = mpfr_get_flt(t, MPFR_RNDN);
		mpfr_set_flt(s, z[i], MPFR_RNDN);
		mpfr_sub(t, t, s, MPFR_RNDN);
		mpfr_clear(s);
	}
	mpfr_clear(t);
}

/* ---- random operand ------------------------------------------------------- */
static double frand(void) { return 2.0 * ((double)rand() / (double)RAND_MAX) - 1.0; }

static void rand_mw(double *z, int K, gmp_randstate_t st, mpfr_t tmp)
{
	mpfr_urandomb(tmp, st);
	mpfr_mul_2si(tmp, tmp, (rand() % 41) - 20, MPFR_RNDN);
	if (rand() & 1) mpfr_neg(tmp, tmp, MPFR_RNDN);
	mpfr_to_mw(z, tmp, K);
}
static void rand_mwf(float *z, int K, gmp_randstate_t st, mpfr_t tmp)
{
	mpfr_urandomb(tmp, st);
	mpfr_mul_2si(tmp, tmp, (rand() % 21) - 10, MPFR_RNDN);
	if (rand() & 1) mpfr_neg(tmp, tmp, MPFR_RNDN);
	mpfr_to_mwf(z, tmp, K);
}

/* ---- existing variants for comparison ------------------------------------- */
static void dd_fma_q (double z[2], const double x[2], const double y[2], const double c[2])
{ double t[2]; c_dd_mul(x, y, t); c_dd_add_sloppy(t, c, z); }
static void dd_fma_bf(double z[2], const double x[2], const double y[2], const double c[2])
{ double t[2]; c_dd_mul_bf(x, y, t); c_dd_add_bf(t, c, z); }
static void td_fma_q (double z[3], const double x[3], const double y[3], const double c[3])
{ double t[3]; c_td_mul_sloppy((double *)x, (double *)y, t); c_td_add(t, (double *)c, z); }
static void td_fma_bf(double z[3], const double x[3], const double y[3], const double c[3])
{ double t[3]; c_td_mul_bf((double *)x, (double *)y, t); c_td_add_bf(t, (double *)c, z); }
static void qd_fma_q (double z[4], const double x[4], const double y[4], const double c[4])
{ double t[4]; c_qd_mul(x, y, t); c_qd_add(t, c, z); }
static void qd_fma_bf(double z[4], const double x[4], const double y[4], const double c[4])
{ double t[4]; c_qd_mul_bf(x, y, t); c_qd_add_bf(t, c, z); }

typedef void (*fma_fn)(double *, const double *, const double *, const double *);
typedef void (*fma_fnf)(float *, const float *, const float *, const float *);

struct stat { double max_eta, max_rel, sum_rel; long n; };
static void stat_init(struct stat *s){ s->max_eta = s->max_rel = s->sum_rel = 0.0; s->n = 0; }

static void run_double(const char *name, int K, double u_pow, double bound,
                       fma_fn fns[3], const char *fnames[3], long ntrial)
{
	int v, i;
	struct stat st[3];
	double x[4], y[4], c[4], z[4];
	mpfr_t mx, my, mc, mref, mz, mt, mden, mabs;
	gmp_randstate_t rs;

	gmp_randinit_default(rs); gmp_randseed_ui(rs, 20260728UL);
	mpfr_inits2(REF_PREC, mx, my, mc, mref, mz, mt, mden, mabs, (mpfr_ptr)0);
	for (v = 0; v < 3; v++) stat_init(&st[v]);

	for (i = 0; i < ntrial; i++) {
		rand_mw(x, K, rs, mt); rand_mw(y, K, rs, mt); rand_mw(c, K, rs, mt);
		mw_to_mpfr(mx, x, K); mw_to_mpfr(my, y, K); mw_to_mpfr(mc, c, K);
		/* reference xy + c, and denominator |xy| + |c| */
		mpfr_mul(mref, mx, my, MPFR_RNDN);
		mpfr_abs(mden, mref, MPFR_RNDN);
		mpfr_abs(mabs, mc, MPFR_RNDN);
		mpfr_add(mden, mden, mabs, MPFR_RNDN);
		mpfr_add(mref, mref, mc, MPFR_RNDN);

		for (v = 0; v < 3; v++) {
			double eta, rel;
			fns[v](z, x, y, c);
			mw_to_mpfr(mz, z, K);
			mpfr_sub(mt, mz, mref, MPFR_RNDN);
			mpfr_abs(mt, mt, MPFR_RNDN);
			/* eta: (|xy|+|c|)-relative, scaled by u^K */
			mpfr_div(mabs, mt, mden, MPFR_RNDN);
			eta = mpfr_get_d(mabs, MPFR_RNDN) / u_pow;
			/* rel: result-relative */
			mpfr_abs(mabs, mref, MPFR_RNDN);
			mpfr_div(mabs, mt, mabs, MPFR_RNDN);
			rel = mpfr_get_d(mabs, MPFR_RNDN);
			if (eta > st[v].max_eta) st[v].max_eta = eta;
			if (rel > st[v].max_rel) st[v].max_rel = rel;
			st[v].sum_rel += rel; st[v].n++;
		}
	}
	printf("%-4s  K=%d  trials=%ld   certified bound C_K = %.0f\n", name, K, ntrial, bound);
	printf("      %-8s %14s %14s %14s\n", "variant", "max eta/u^K", "max relerr", "mean relerr");
	for (v = 0; v < 3; v++)
		printf("      %-8s %14.4g %14.4e %14.4e%s\n", fnames[v], st[v].max_eta,
		       st[v].max_rel, st[v].sum_rel / (double)st[v].n,
		       (v == 2) ? ((st[v].max_eta <= bound) ? "   [bound OK]" : "   [BOUND VIOLATED]") : "");
	printf("\n");
	mpfr_clears(mx, my, mc, mref, mz, mt, mden, mabs, (mpfr_ptr)0);
	gmp_randclear(rs);
}

static void run_float(const char *name, int K, double u_pow, double bound,
                      fma_fnf fn, long ntrial)
{
	int i;
	struct stat st;
	float x[4], y[4], c[4], z[4];
	mpfr_t mx, my, mc, mref, mz, mt, mden, mabs;
	gmp_randstate_t rs;

	gmp_randinit_default(rs); gmp_randseed_ui(rs, 20260728UL);
	mpfr_inits2(REF_PREC, mx, my, mc, mref, mz, mt, mden, mabs, (mpfr_ptr)0);
	stat_init(&st);

	for (i = 0; i < ntrial; i++) {
		double eta, rel;
		rand_mwf(x, K, rs, mt); rand_mwf(y, K, rs, mt); rand_mwf(c, K, rs, mt);
		mwf_to_mpfr(mx, x, K); mwf_to_mpfr(my, y, K); mwf_to_mpfr(mc, c, K);
		mpfr_mul(mref, mx, my, MPFR_RNDN);
		mpfr_abs(mden, mref, MPFR_RNDN);
		mpfr_abs(mabs, mc, MPFR_RNDN);
		mpfr_add(mden, mden, mabs, MPFR_RNDN);
		mpfr_add(mref, mref, mc, MPFR_RNDN);

		fn(z, x, y, c);
		mwf_to_mpfr(mz, z, K);
		mpfr_sub(mt, mz, mref, MPFR_RNDN);
		mpfr_abs(mt, mt, MPFR_RNDN);
		mpfr_div(mabs, mt, mden, MPFR_RNDN);
		eta = mpfr_get_d(mabs, MPFR_RNDN) / u_pow;
		mpfr_abs(mabs, mref, MPFR_RNDN);
		mpfr_div(mabs, mt, mabs, MPFR_RNDN);
		rel = mpfr_get_d(mabs, MPFR_RNDN);
		if (eta > st.max_eta) st.max_eta = eta;
		if (rel > st.max_rel) st.max_rel = rel;
		st.sum_rel += rel; st.n++;
	}
	printf("%-4s  K=%d  trials=%ld   certified bound C_K = %.0f\n", name, K, ntrial, bound);
	printf("      %-8s %14.4g %14.4e %14.4e%s\n", "FMA", st.max_eta, st.max_rel,
	       st.sum_rel / (double)st.n, (st.max_eta <= bound) ? "   [bound OK]" : "   [BOUND VIOLATED]");
	printf("\n");
	mpfr_clears(mx, my, mc, mref, mz, mt, mden, mabs, (mpfr_ptr)0);
	gmp_randclear(rs);
}

/* ---- sanity: EFT must really be error-free (catches -ffp-contract=fast) ---- */
static int check_eft(void)
{
	double a = 1.0 + ldexp(1.0, -30), b = 1.0 - ldexp(1.0, -40), e, p;
	p = two_prod(a, b, &e);
	{
		mpfr_t m1, m2;
		int ok;
		mpfr_init2(m1, 200); mpfr_init2(m2, 200);
		mpfr_set_d(m1, a, MPFR_RNDN); mpfr_mul_d(m1, m1, b, MPFR_RNDN);
		mpfr_set_d(m2, p, MPFR_RNDN); mpfr_add_d(m2, m2, e, MPFR_RNDN);
		ok = (mpfr_cmp(m1, m2) == 0);
		mpfr_clear(m1); mpfr_clear(m2);
		return ok;
	}
}

/* ---- commutativity in x <-> y (bitwise) ------------------------------------ */
static long check_commute(int K, fma_fn fn, long ntrial)
{
	long i, bad = 0;
	double x[4], y[4], c[4], z1[4], z2[4];
	mpfr_t tmp;
	gmp_randstate_t rs;

	gmp_randinit_default(rs); gmp_randseed_ui(rs, 12345UL);
	mpfr_init2(tmp, REF_PREC);
	for (i = 0; i < ntrial; i++) {
		int k;
		rand_mw(x, K, rs, tmp); rand_mw(y, K, rs, tmp); rand_mw(c, K, rs, tmp);
		fn(z1, x, y, c);
		fn(z2, y, x, c);
		for (k = 0; k < K; k++)
			if (memcmp(&z1[k], &z2[k], sizeof(double)) != 0) { bad++; break; }
	}
	mpfr_clear(tmp);
	gmp_randclear(rs);
	return bad;
}

int main(int argc, char *argv[])
{
	long ntrial = (argc > 1) ? atol(argv[1]) : 200000L;
	double u = ldexp(1.0, -53), uf = ldexp(1.0, -24);
	fma_fn ddf[3] = { dd_fma_q, dd_fma_bf, bnc_dwfma };
	fma_fn tdf[3] = { td_fma_q, td_fma_bf, bnc_twfma };
	fma_fn qdf[3] = { qd_fma_q, qd_fma_bf, bnc_qwfma };
	const char *nm[3] = { "Q", "BF", "FMA" };

	srand(20260728);
	printf("=== proposed branch-free FMA: certified error bound check ===\n");
	printf("reference: MPFR %d bits;  eta = |z-(xy+c)| / ((|xy|+|c|) u^K)\n\n", REF_PREC);

	if (!check_eft()) {
		fprintf(stderr, "FATAL: two_prod is not error-free "
		                "(compile with -ffp-contract=off)\n");
		return 1;
	}
	printf("EFT self-check (two_prod exactness): OK\n\n");

	run_double("DD", 2, u * u,           34.0,  ddf, nm, ntrial);
	run_double("TD", 3, u * u * u,      184.0,  tdf, nm, ntrial);
	run_double("QD", 4, u * u * u * u,  812.0,  qdf, nm, ntrial);

	printf("--- commutativity  fma(x,y,c) == fma(y,x,c) bitwise ---\n");
	printf("      DD mismatches: %ld / %ld\n", check_commute(2, bnc_dwfma, ntrial), ntrial);
	printf("      TD mismatches: %ld / %ld\n", check_commute(3, bnc_twfma, ntrial), ntrial);
	printf("      QD mismatches: %ld / %ld\n\n", check_commute(4, bnc_qwfma, ntrial), ntrial);

	printf("--- single-precision base (binary32) ---\n\n");
	run_float("DS", 2, uf * uf,          34.0,  bnc_dwfmaf, ntrial);
	run_float("TS", 3, uf * uf * uf,    184.0,  bnc_twfmaf, ntrial);
	run_float("QS", 4, uf*uf*uf*uf,     812.0,  bnc_qwfmaf, ntrial);

	return 0;
}
