/*
 * poly_bench.c : polynomial benchmark of the proposed branch-free FMA
 *                (arXiv:2607.11391) as wired into src/{dd,td,qd}_poly.c.
 *
 * Kernels (all of them are chains of MACs z := x*y + c):
 *   HORNER    eval_*poly_horner   ret := ret*x + a_i          (deg+1 MACs / eval)
 *   ESTRIN    eval_*poly_estrin   pairwise a_{2i+1}*x + a_{2i}  (scalar)
 *   ESTRIN-V  _bncavx2_eval_*poly_estrin  the same, SIMD (NEON / SVE2 / AVX)
 *   HORNER-V  batch Horner: LANES evaluation points at once (SIMD over points)
 *   EVALDIFF  eval_diff_*poly     ret := ret*x + i*a_i
 *   POLYMUL   mul_*poly           c_{i+j} += a_i * b_j        (O(n^2) MACs)
 *
 * Build the SAME binary twice - once without and once with -DBNC_USE_NEW_FMA -
 * and compare: the accuracy is measured against MPFR 600-bit inside each run,
 * so the two runs can be compared directly.
 *
 * build: see build_fma.sh (poly_bench_{serial,neon,sve2}[_fma])
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <mpfr.h>

#include "bnc_common.h"
#include "rdd.h"
#include "poly.h"

#define REF_PREC 600
#define MAXK 4

/* number of evaluation points that get an MPFR reference (0 = timing only) */
static long accpts = -1;
static int  ntimed = 3;   /* timing repetitions; the best (minimum) is reported */

/* run LOOP once to warm up, then ntimed times, keeping the shortest run */
#define TIMEIT(t, LOOP) do {                                                   \
	double _best = 1e300; int _rr;                                             \
	LOOP;                                     /* warm-up, not timed */         \
	for (_rr = 0; _rr < ntimed; _rr++) {                                       \
		double _t0 = wtime();                                                  \
		LOOP;                                                                  \
		_t0 = wtime() - _t0;                                                   \
		if (_t0 < _best) _best = _t0;                                          \
	}                                                                          \
	(t) = _best;                                                               \
} while (0)

static double wtime(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static unsigned long rstate = 20260728UL;
static double urand(void)
{
	rstate ^= rstate << 13; rstate ^= rstate >> 7; rstate ^= rstate << 17;
	return (double)(rstate >> 11) / 9007199254740992.0;
}
static void rand_mw(double *z, int K)
{
	int i;
	z[0] = 2.0 * urand() - 1.0;
	for (i = 1; i < K; i++) {
		z[i] = ldexp(z[i - 1] * (0.5 + 0.5 * urand()), -53);
		if (urand() < 0.5) z[i] = -z[i];
	}
	for (i = 0; i + 1 < K; i++) z[i] = quick_two_sum(z[i], z[i + 1], &z[i + 1]);
}
static void mw_to_mpfr(mpfr_t r, const double *z, int K)
{
	int i;
	mpfr_set_d(r, z[0], MPFR_RNDN);
	for (i = 1; i < K; i++) mpfr_add_d(r, r, z[i], MPFR_RNDN);
}


/* ==========================================================================
 *  Batch (multi-point) Horner: the natural SIMD form of Horner's rule -
 *  LANES evaluation points are processed simultaneously, so the serial
 *  dependency chain of Horner is hidden across lanes.
 *  The library only ships an AVX2 version (_bncavx2_eval_*poly_horner), so the
 *  NEON / SVE2 counterparts are written here.  Q / BF / FMA are selected by the
 *  same macros as everywhere else (USE_*_BF, BNC_USE_NEW_FMA).
 * ========================================================================== */
#if defined(BNC_ENABLE_SVE2) && defined(__ARM_SVE2)
# include <arm_sve.h>
# include "sve2/bncsve2.h"
# ifndef SVE_BITS
#  define SVE_BITS 128
# endif
typedef svfloat64_t VEC __attribute__((arm_sve_vector_bits(SVE_BITS)));
# define LANES (SVE_BITS / 64)
# define PG    svptrue_b64()
# define VLOADP(d, p)  (d) = svld1_f64(PG, (p))
# define VSTOREP(p, s) svst1_f64(PG, (p), (s))
# define VSPLAT(d, s)  (d) = svdup_n_f64(s)
# define BATCH_NAME "sve2"
# ifdef BNC_USE_NEW_FMA
#  define VMAC_DD(acc, x, c) do { svfloat64_t _z0,_z1; \
     _bncsve2_dwfma(PG, &_z0,&_z1, (acc)[0],(acc)[1], (x)[0],(x)[1], (c)[0],(c)[1]); \
     (acc)[0]=_z0; (acc)[1]=_z1; } while (0)
#  define VMAC_TD(acc, x, c) do { svfloat64_t _z0,_z1,_z2; \
     _bncsve2_twfma(PG, &_z0,&_z1,&_z2, (acc)[0],(acc)[1],(acc)[2], \
                    (x)[0],(x)[1],(x)[2], (c)[0],(c)[1],(c)[2]); \
     (acc)[0]=_z0; (acc)[1]=_z1; (acc)[2]=_z2; } while (0)
#  define VMAC_QD(acc, x, c) do { svfloat64_t _z0,_z1,_z2,_z3; \
     _bncsve2_qwfma(PG, &_z0,&_z1,&_z2,&_z3, (acc)[0],(acc)[1],(acc)[2],(acc)[3], \
                    (x)[0],(x)[1],(x)[2],(x)[3], (c)[0],(c)[1],(c)[2],(c)[3]); \
     (acc)[0]=_z0; (acc)[1]=_z1; (acc)[2]=_z2; (acc)[3]=_z3; } while (0)
# else
#  define VMAC_DD(acc, x, c) do { svfloat64_t _p0,_p1,_s0,_s1; \
     _bncsve2_rdd_mul(PG, &_p0,&_p1, (acc)[0],(acc)[1], (x)[0],(x)[1]); \
     _bncsve2_rdd_add(PG, &_s0,&_s1, _p0,_p1, (c)[0],(c)[1]); \
     (acc)[0]=_s0; (acc)[1]=_s1; } while (0)
#  define VMAC_TD(acc, x, c) do { svfloat64_t _p0,_p1,_p2,_s0,_s1,_s2; \
     _bncsve2_rtd_mul(PG, &_p0,&_p1,&_p2, (acc)[0],(acc)[1],(acc)[2], \
                      (x)[0],(x)[1],(x)[2]); \
     _bncsve2_rtd_add(PG, &_s0,&_s1,&_s2, _p0,_p1,_p2, (c)[0],(c)[1],(c)[2]); \
     (acc)[0]=_s0; (acc)[1]=_s1; (acc)[2]=_s2; } while (0)
#  define VMAC_QD(acc, x, c) do { svfloat64_t _p0,_p1,_p2,_p3,_s0,_s1,_s2,_s3; \
     _bncsve2_rqd_mul(PG, &_p0,&_p1,&_p2,&_p3, (acc)[0],(acc)[1],(acc)[2],(acc)[3], \
                      (x)[0],(x)[1],(x)[2],(x)[3]); \
     _bncsve2_rqd_add(PG, &_s0,&_s1,&_s2,&_s3, _p0,_p1,_p2,_p3, \
                      (c)[0],(c)[1],(c)[2],(c)[3]); \
     (acc)[0]=_s0; (acc)[1]=_s1; (acc)[2]=_s2; (acc)[3]=_s3; } while (0)
# endif

#elif defined(BNC_ENABLE_NEON) && defined(__ARM_NEON)
# include <arm_neon.h>
# include "neon/bncneon.h"
typedef float64x2_t VEC;
# define LANES 2
# define VLOADP(d, p)  (d) = vld1q_f64(p)
# define VSTOREP(p, s) vst1q_f64((p), (s))
# define VSPLAT(d, s)  (d) = vdupq_n_f64(s)
# define BATCH_NAME "neon"
# ifdef BNC_USE_NEW_FMA
#  define VMAC_DD(acc, x, c) _bncneon_dwfma((acc), (acc), (x), (c))
#  define VMAC_TD(acc, x, c) _bncneon_twfma((acc), (acc), (x), (c))
#  define VMAC_QD(acc, x, c) _bncneon_qwfma((acc), (acc), (x), (c))
# else
#  define VMAC_DD(acc, x, c) do { VEC _t[2]; \
     _bncneon_rdd_mul(_t, (acc), (x)); _bncneon_rdd_add((acc), _t, (VEC *)(c)); } while (0)
#  define VMAC_TD(acc, x, c) do { VEC _t[3]; \
     _bncneon_rtd_mul(_t, (acc), (x)); _bncneon_rtd_add((acc), _t, (VEC *)(c)); } while (0)
#  define VMAC_QD(acc, x, c) do { VEC _t[4]; \
     _bncneon_rqd_mul(_t, (acc), (x)); _bncneon_rqd_add((acc), _t, (VEC *)(c)); } while (0)
# endif

#else   /* scalar: LANES = 1, i.e. plain Horner */
typedef double VEC;
# define LANES 1
# define VLOADP(d, p)  (d) = *(p)
# define VSTOREP(p, s) *(p) = (s)
# define VSPLAT(d, s)  (d) = (s)
# define BATCH_NAME "serial"
# define VMAC_DD(acc, x, c) rdd_fma((acc), (acc), (VEC *)(x), (VEC *)(c))
# define VMAC_TD(acc, x, c) rtd_fma((acc), (acc), (VEC *)(x), (VEC *)(c))
# define VMAC_QD(acc, x, c) rqd_fma((acc), (acc), (VEC *)(x), (VEC *)(c))
#endif

/* one batch-Horner routine per type */
#define DEF_HBATCH(TAG, KK, POLY, getpi, VMAC)                                 \
static double hbatch_##TAG(POLY a, double *const xsoa[MAXK], long npts)        \
{                                                                              \
	long p, i;                                                                 \
	int k;                                                                     \
	double sink = 0.0, o[LANES];                                               \
	VEC x[KK], acc[KK], c[KK];                                                 \
	for (p = 0; p + LANES <= npts; p += LANES) {                               \
		for (k = 0; k < KK; k++) VLOADP(x[k], &xsoa[k][p]);                     \
		for (k = 0; k < KK; k++) VSPLAT(acc[k], getpi(a, a->deg)[k]);           \
		for (i = a->deg - 1; i >= 0; i--) {                                    \
			for (k = 0; k < KK; k++) VSPLAT(c[k], getpi(a, i)[k]);              \
			VMAC(acc, x, c);                                                   \
		}                                                                      \
		VSTOREP(o, acc[0]);                                                    \
		sink += o[0];                                                          \
	}                                                                          \
	return sink;                                                               \
}

DEF_HBATCH(dd, 2, DDPoly, get_ddpoly_i, VMAC_DD)
DEF_HBATCH(td, 3, TDPoly, get_tdpoly_i, VMAC_TD)
DEF_HBATCH(qd, 4, QDPoly, get_qdpoly_i, VMAC_QD)

struct acc { double maxrel, sumrel; long n; };
static void acc_init(struct acc *a) { a->maxrel = a->sumrel = 0.0; a->n = 0; }
static void acc_add(struct acc *a, const double *z, int K, mpfr_t ref)
{
	mpfr_t t, u;
	double rel;
	mpfr_inits2(REF_PREC, t, u, (mpfr_ptr)0);
	mw_to_mpfr(t, z, K);
	mpfr_sub(t, t, ref, MPFR_RNDN); mpfr_abs(t, t, MPFR_RNDN);
	mpfr_abs(u, ref, MPFR_RNDN);
	if (!mpfr_zero_p(u)) {
		mpfr_div(t, t, u, MPFR_RNDN);
		rel = mpfr_get_d(t, MPFR_RNDN);
		if (rel > a->maxrel) a->maxrel = rel;
		a->sumrel += rel; a->n++;
	}
	mpfr_clears(t, u, (mpfr_ptr)0);
}
static void report(const char *ty, const char *op, struct acc *a, double t, long calls)
{
	printf("%-4s %-9s %12.4e %12.4e %13.4e %10ld\n", ty, op, a->maxrel,
	       a->sumrel / (double)(a->n ? a->n : 1), t / (double)calls, calls);
}

/* ==========================================================================
 *  one benchmark per type, generated by a macro
 * ========================================================================== */
#define RUN(TY, TYS, K, POLY, initp, setpi, getpi, freep,                      \
            horner, estrin, estrinv, evaldiff, mulp, hbatch)                                    \
	do {                                                                       \
		POLY a = initp(deg + 1), b = initp(deg / 2 + 1), c = initp(deg + deg / 2 + 2);   \
		double coef[MAXK], xv[MAXK], ret[MAXK];                                \
		double *xs = malloc(sizeof(double) * MAXK * (size_t)npts);             \
		double *xsoa[MAXK];                                                    \
		mpfr_t *rh, *rd, mx, mt, mc;                                           \
		long i, j, p;                                                          \
		int k;                                                                 \
		double t0, t_h, t_e, t_v, t_d, t_m, t_b;                               \
		double sink_b = 0.0;                                                   \
		struct acc ah, ae, av, ad, am;                                             \
		                                                                       \
		rstate = 20260728UL;                                                   \
		a->deg = deg;                                                          \
		for (i = 0; i <= deg; i++) { rand_mw(coef, K); setpi(a, i, coef); }     \
		b->deg = deg / 2;                                                      \
		for (i = 0; i <= deg / 2; i++) { rand_mw(coef, K); setpi(b, i, coef); } \
		for (p = 0; p < npts; p++) { rand_mw(coef, K);                          \
			for (k = 0; k < K; k++) xs[p * MAXK + k] = coef[k] * 0.5; }         \
		/* SoA copy of the evaluation points for the batch (SIMD) Horner */    \
		for (k = 0; k < MAXK; k++) xsoa[k] = malloc(sizeof(double) * (size_t)npts); \
		for (p = 0; p < npts; p++)                                             \
			for (k = 0; k < MAXK; k++) xsoa[k][p] = xs[p * MAXK + k];          \
		                                                                       \
		mpfr_inits2(REF_PREC, mx, mt, mc, (mpfr_ptr)0);                        \
		rh = malloc(sizeof(mpfr_t) * (size_t)npts);                            \
		rd = malloc(sizeof(mpfr_t) * (size_t)npts);                            \
		for (p = 0; p < accpts; p++) {                                         \
			mpfr_init2(rh[p], REF_PREC); mpfr_init2(rd[p], REF_PREC);          \
			mw_to_mpfr(mx, &xs[p * MAXK], K);                                  \
			/* Horner reference */                                             \
			mpfr_set_zero(rh[p], 1);                                           \
			for (i = deg; i >= 0; i--) {                                       \
				mpfr_mul(rh[p], rh[p], mx, MPFR_RNDN);                         \
				mw_to_mpfr(mc, getpi(a, i), K);                                \
				mpfr_add(rh[p], rh[p], mc, MPFR_RNDN);                         \
			}                                                                  \
			/* derivative reference */                                         \
			mpfr_set_zero(rd[p], 1);                                           \
			for (i = deg; i >= 1; i--) {                                       \
				mpfr_mul(rd[p], rd[p], mx, MPFR_RNDN);                         \
				mw_to_mpfr(mc, getpi(a, i), K);                                \
				mpfr_mul_ui(mc, mc, (unsigned long)i, MPFR_RNDN);              \
				mpfr_add(rd[p], rd[p], mc, MPFR_RNDN);                         \
			}                                                                  \
		}                                                                      \
		                                                                       \
		acc_init(&ah); acc_init(&ae); acc_init(&av); acc_init(&ad); acc_init(&am);            \
		TIMEIT(t_h, for (p = 0; p < npts; p++) { horner(ret, a, &xs[p * MAXK]); });  \
		for (p = 0; p < accpts; p++) {                                         \
			horner(ret, a, &xs[p * MAXK]); acc_add(&ah, ret, K, rh[p]); }      \
		                                                                       \
		TIMEIT(t_e, for (p = 0; p < npts; p++) { estrin(ret, a, &xs[p * MAXK]); });  \
		for (p = 0; p < accpts; p++) {                                         \
			estrin(ret, a, &xs[p * MAXK]); acc_add(&ae, ret, K, rh[p]); }      \
		                                                                       \
		TIMEIT(t_v, for (p = 0; p < npts; p++) { estrinv(ret, a, &xs[p * MAXK]); });  \
		for (p = 0; p < accpts; p++) {                                         \
			estrinv(ret, a, &xs[p * MAXK]); acc_add(&av, ret, K, rh[p]); }     \
		                                                                       \
		TIMEIT(t_d, for (p = 0; p < npts; p++) { evaldiff(ret, a, &xs[p * MAXK]); });  \
		for (p = 0; p < accpts; p++) {                                         \
			evaldiff(ret, a, &xs[p * MAXK]); acc_add(&ad, ret, K, rd[p]); }    \
		                                                                       \
		TIMEIT(t_b, sink_b += hbatch(a, xsoa, npts));                          \
		                                                                       \
		TIMEIT(t_m, for (i = 0; i < mulrep; i++) mulp(c, a, b));            \
		/* polynomial-product reference, coefficient by coefficient */          \
		for (i = 0; accpts > 0 && i <= deg + deg / 2; i++) {                                  \
			mpfr_set_zero(mt, 1);                                               \
			for (j = 0; j <= deg / 2; j++) {                                    \
				if (i - j < 0 || i - j > deg) continue;                         \
				mw_to_mpfr(mx, getpi(a, i - j), K);                             \
				mw_to_mpfr(mc, getpi(b, j), K);                                 \
				mpfr_mul(mx, mx, mc, MPFR_RNDN);                                \
				mpfr_add(mt, mt, mx, MPFR_RNDN);                                \
			}                                                                   \
			acc_add(&am, getpi(c, i), K, mt);                                   \
		}                                                                       \
		                                                                        \
		report(TYS, "HORNER",   &ah, t_h, npts);                                \
		report(TYS, "HORNER-V", &ah, t_b, (npts / LANES) * LANES);              \
		report(TYS, "ESTRIN",   &ae, t_e, npts);                                \
		report(TYS, "ESTRIN-V", &av, t_v, npts);                                \
		report(TYS, "EVALDIFF", &ad, t_d, npts);                                \
		report(TYS, "POLYMUL",  &am, t_m, mulrep);                              \
		                                                                        \
		for (p = 0; p < accpts; p++) { mpfr_clear(rh[p]); mpfr_clear(rd[p]); }   \
		free(rh); free(rd); free(xs);                                           \
		for (k = 0; k < MAXK; k++) free(xsoa[k]);                               \
		if (sink_b == 12345.6789) printf("");   /* keep the batch results live */ \
		mpfr_clears(mx, mt, mc, (mpfr_ptr)0);                                   \
		freep(a); freep(b); freep(c);                                           \
	} while (0)

int main(int argc, char *argv[])
{
	long deg = 1000, npts = 2000, mulrep = 20;
	int i;

	for (i = 1; i < argc; i++) {
		if      (!strcmp(argv[i], "-deg")  && i + 1 < argc) deg    = atol(argv[++i]);
		else if (!strcmp(argv[i], "-npts") && i + 1 < argc) npts   = atol(argv[++i]);
		else if (!strcmp(argv[i], "-rep")  && i + 1 < argc) mulrep = atol(argv[++i]);
		else if (!strcmp(argv[i], "-accpts") && i + 1 < argc) accpts = atol(argv[++i]);
		else if (!strcmp(argv[i], "-noacc")) accpts = 0;
		else if (!strcmp(argv[i], "-ntimed") && i + 1 < argc) ntimed = atoi(argv[++i]);
		else { fprintf(stderr, "usage: %s [-deg N] [-npts N] [-rep N] [-accpts N | -noacc]\n", argv[0]); return 1; }
	}

	if (accpts < 0 || accpts > npts) accpts = npts;

	printf("# backend=%s variant=%s\n",
#if defined(BNC_ENABLE_SVE2)
	       "sve2",
#elif defined(BNC_ENABLE_NEON)
	       "neon",
#else
	       "serial",
#endif
#ifdef BNC_USE_NEW_FMA
	       "FMA"
#elif defined(USE_DD_BF)
	       "BF"
#else
	       "Q"
#endif
	       );
	printf("# degree=%ld  evaluation points=%ld  POLYMUL: deg %ld x deg %ld, %ld reps\n",
	       deg, npts, deg, deg / 2, mulrep);
	printf("# accuracy reference: MPFR %d bits on %ld point(s); "
	       "timing = best of %d runs after a warm-up\n#\n", REF_PREC, accpts, ntimed);
	printf("%-4s %-9s %12s %12s %13s %10s\n",
	       "type", "op", "maxrelerr", "meanrelerr", "t[s/call]", "calls");

	RUN(dd, "dd", 2, DDPoly, init_ddpoly, set_ddpoly_i, get_ddpoly_i, free_ddpoly,
	    eval_ddpoly_horner, eval_ddpoly_estrin, _bncavx2_eval_ddpoly_estrin,
	    eval_diff_ddpoly, mul_ddpoly, hbatch_dd);
	RUN(td, "td", 3, TDPoly, init_tdpoly, set_tdpoly_i, get_tdpoly_i, free_tdpoly,
	    eval_tdpoly_horner, eval_tdpoly_estrin, _bncavx2_eval_tdpoly_estrin,
	    eval_diff_tdpoly, mul_tdpoly, hbatch_td);
	RUN(qd, "qd", 4, QDPoly, init_qdpoly, set_qdpoly_i, get_qdpoly_i, free_qdpoly,
	    eval_qdpoly_horner, eval_qdpoly_estrin, _bncavx2_eval_qdpoly_estrin,
	    eval_diff_qdpoly, mul_qdpoly, hbatch_qd);

	return 0;
}
