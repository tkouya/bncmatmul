/*
 * test_sve2_vs_neon.c : audit of the SVE2 single-word (DS/TS/QS) and
 *                       double-word (DD/TD/QD) add/mul routines against their
 *                       NEON counterparts.
 *
 * Both backends implement the same scalar netlists, so for identical inputs
 * they must produce bitwise identical results.  A mismatch means one of the
 * two was mis-transcribed.
 *
 * build: gcc -O2 -ffp-contract=off -mcpu=neoverse-v2 -DBNC_ENABLE_NEON \
 *            -DBNC_ENABLE_SVE2 -Iinclude bench/fma/test_sve2_vs_neon.c -lm \
 *            -o bench/fma/test_sve2_vs_neon
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "bnc_common.h"
#include "rdd.h"
#include "rds.h"
#include <arm_neon.h>
#include "neon/bncneon.h"
#include <arm_sve.h>
#include "sve2/bncsve2.h"

#define MAXK 4
#define VLMAX 16

static unsigned long rs = 424242UL;
static double u(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
                        return (double)(rs >> 11) / 9007199254740992.0; }
static void rand_d(double *z, int K)
{
	int i;
	z[0] = 2.0 * u() - 1.0;
	for (i = 1; i < K; i++) {
		z[i] = ldexp(z[i - 1] * (0.5 + 0.5 * u()), -53);
		if (u() < 0.5) z[i] = -z[i];
	}
	for (i = 0; i + 1 < K; i++) z[i] = quick_two_sum(z[i], z[i + 1], &z[i + 1]);
}
static void rand_f(float *z, int K)
{
	int i;
	z[0] = (float)(2.0 * u() - 1.0);
	for (i = 1; i < K; i++) {
		z[i] = ldexpf(z[i - 1] * (0.5f + 0.5f * (float)u()), -24);
		if (u() < 0.5) z[i] = -z[i];
	}
	for (i = 0; i + 1 < K; i++) z[i] = fquick_two_sum(z[i], z[i + 1], &z[i + 1]);
}

static long bad_total = 0;
static void report(const char *name, long bad, long n)
{
	printf("  %-28s %s  (%ld / %ld)\n", name,
	       bad ? "MISMATCH" : "identical", bad, n);
	bad_total += bad;
}

/* ---------------- float (DS / TS / QS) ---------------- */
#define AUDIT_F(NAME, K, NEONCALL, SVECALL)                                    \
	do {                                                                       \
		long bad = 0, i;                                                       \
		for (i = 0; i < n; i++) {                                              \
			float a[MAXK], b[MAXK], zn[MAXK], zv[MAXK];                        \
			float32x4_t va[MAXK], vb[MAXK], vr[MAXK];                          \
			float ba[MAXK][VLMAX], bb[MAXK][VLMAX], br[MAXK][VLMAX];           \
			svbool_t pg = svptrue_b32();                                       \
			svfloat32_t r0, r1, r2, r3;                                        \
			int k, l, vl = (int)svcntw();                                      \
			(void)r2; (void)r3;                                                \
			rand_f(a, K); rand_f(b, K);                                        \
			for (k = 0; k < K; k++) {                                          \
				float ta[4] = { a[k], a[k], a[k], a[k] };                      \
				float tb[4] = { b[k], b[k], b[k], b[k] };                      \
				va[k] = vld1q_f32(ta); vb[k] = vld1q_f32(tb);                   \
				for (l = 0; l < vl; l++) { ba[k][l] = a[k]; bb[k][l] = b[k]; }  \
			}                                                                  \
			NEONCALL;                                                          \
			for (k = 0; k < K; k++) { float o[4]; vst1q_f32(o, vr[k]); zn[k] = o[0]; } \
			SVECALL;                                                           \
			for (k = 0; k < K; k++) zv[k] = br[k][0];                          \
			if (memcmp(zn, zv, sizeof(float) * K)) bad++;                       \
		}                                                                      \
		report(NAME, bad, n);                                                  \
	} while (0)

/* ---------------- double (DD / TD / QD) ---------------- */
#define AUDIT_D(NAME, K, NEONCALL, SVECALL)                                    \
	do {                                                                       \
		long bad = 0, i;                                                       \
		for (i = 0; i < n; i++) {                                              \
			double a[MAXK], b[MAXK], zn[MAXK], zv[MAXK];                       \
			float64x2_t va[MAXK], vb[MAXK], vr[MAXK];                          \
			double ba[MAXK][VLMAX], bb[MAXK][VLMAX], br[MAXK][VLMAX];          \
			svbool_t pg = svptrue_b64();                                       \
			svfloat64_t r0, r1, r2, r3;                                        \
			int k, l, vl = (int)svcntd();                                      \
			(void)r2; (void)r3;                                                \
			rand_d(a, K); rand_d(b, K);                                        \
			for (k = 0; k < K; k++) {                                          \
				double ta[2] = { a[k], a[k] };                                 \
				double tb[2] = { b[k], b[k] };                                 \
				va[k] = vld1q_f64(ta); vb[k] = vld1q_f64(tb);                   \
				for (l = 0; l < vl; l++) { ba[k][l] = a[k]; bb[k][l] = b[k]; }  \
			}                                                                  \
			NEONCALL;                                                          \
			for (k = 0; k < K; k++) { double o[2]; vst1q_f64(o, vr[k]); zn[k] = o[0]; } \
			SVECALL;                                                           \
			for (k = 0; k < K; k++) zv[k] = br[k][0];                          \
			if (memcmp(zn, zv, sizeof(double) * K)) bad++;                      \
		}                                                                      \
		report(NAME, bad, n);                                                  \
	} while (0)

#define LD2F(x) svld1_f32(pg, x[0]), svld1_f32(pg, x[1])
#define LD3F(x) svld1_f32(pg, x[0]), svld1_f32(pg, x[1]), svld1_f32(pg, x[2])
#define LD4F(x) svld1_f32(pg, x[0]), svld1_f32(pg, x[1]), svld1_f32(pg, x[2]), svld1_f32(pg, x[3])
#define ST2F svst1_f32(pg, br[0], r0); svst1_f32(pg, br[1], r1)
#define ST3F ST2F; svst1_f32(pg, br[2], r2)
#define ST4F ST3F; svst1_f32(pg, br[3], r3)

#define LD2D(x) svld1_f64(pg, x[0]), svld1_f64(pg, x[1])
#define LD3D(x) svld1_f64(pg, x[0]), svld1_f64(pg, x[1]), svld1_f64(pg, x[2])
#define LD4D(x) svld1_f64(pg, x[0]), svld1_f64(pg, x[1]), svld1_f64(pg, x[2]), svld1_f64(pg, x[3])
#define ST2D svst1_f64(pg, br[0], r0); svst1_f64(pg, br[1], r1)
#define ST3D ST2D; svst1_f64(pg, br[2], r2)
#define ST4D ST3D; svst1_f64(pg, br[3], r3)

int main(int argc, char *argv[])
{
	long n = (argc > 1) ? atol(argv[1]) : 200000;

	printf("=== SVE2 vs NEON: bitwise agreement of the add/mul kernels ===\n");
	printf("(both implement the same scalar netlist; %ld random operand pairs each)\n\n", n);

	printf("single-word base (DS / TS / QS)\n");
	AUDIT_F("ds add_sloppy", 2,
	        _bncneon_rds_add_sloppy(vr, va, vb),
	        _bncsve2_rds_add_sloppy(pg, &r0, &r1, LD2F(ba), LD2F(bb)); ST2F);
	AUDIT_F("ds mul_sloppy", 2,
	        _bncneon_rds_mul_sloppy(vr, va, vb),
	        _bncsve2_rds_mul_sloppy(pg, &r0, &r1, LD2F(ba), LD2F(bb)); ST2F);
	AUDIT_F("ds add_bf", 2,
	        _bncneon_rds_add_bf(vr, va, vb),
	        _bncsve2_rds_add_bf(pg, &r0, &r1, LD2F(ba), LD2F(bb)); ST2F);
	AUDIT_F("ds mul_bf", 2,
	        _bncneon_rds_mul_bf(vr, va, vb),
	        _bncsve2_rds_mul_bf(pg, &r0, &r1, LD2F(ba), LD2F(bb)); ST2F);
	AUDIT_F("ts addq", 3,
	        _bncneon_rts_addq(vr, va, vb),
	        _bncsve2_rts_addq(pg, &r0, &r1, &r2, LD3F(ba), LD3F(bb)); ST3F);
	AUDIT_F("ts mulq", 3,
	        _bncneon_rts_mulq(vr, va, vb),
	        _bncsve2_rts_mulq(pg, &r0, &r1, &r2, LD3F(ba), LD3F(bb)); ST3F);
	AUDIT_F("ts add_bf", 3,
	        _bncneon_rts_add_bf(vr, va, vb),
	        _bncsve2_rts_add_bf(pg, &r0, &r1, &r2, LD3F(ba), LD3F(bb)); ST3F);
	AUDIT_F("ts mul_bf", 3,
	        _bncneon_rts_mul_bf(vr, va, vb),
	        _bncsve2_rts_mul_bf(pg, &r0, &r1, &r2, LD3F(ba), LD3F(bb)); ST3F);
	AUDIT_F("qs add_sloppy", 4,
	        _bncneon_rqs_add_sloppy(vr, va, vb),
	        _bncsve2_rqs_add_sloppy(pg, &r0, &r1, &r2, &r3, LD4F(ba), LD4F(bb)); ST4F);
	AUDIT_F("qs mul_sloppy", 4,
	        _bncneon_rqs_mul_sloppy(vr, va, vb),
	        _bncsve2_rqs_mul_sloppy(pg, &r0, &r1, &r2, &r3, LD4F(ba), LD4F(bb)); ST4F);
	AUDIT_F("qs add_bf", 4,
	        _bncneon_rqs_add_bf(vr, va, vb),
	        _bncsve2_rqs_add_bf(pg, &r0, &r1, &r2, &r3, LD4F(ba), LD4F(bb)); ST4F);
	AUDIT_F("qs mul_bf", 4,
	        _bncneon_rqs_mul_bf(vr, va, vb),
	        _bncsve2_rqs_mul_bf(pg, &r0, &r1, &r2, &r3, LD4F(ba), LD4F(bb)); ST4F);

	printf("\ndouble-word base (DD / TD / QD)\n");
	AUDIT_D("dd add_sloppy", 2,
	        _bncneon_rdd_add_sloppy(vr, va, vb),
	        _bncsve2_rdd_add_sloppy(pg, &r0, &r1, LD2D(ba), LD2D(bb)); ST2D);
	AUDIT_D("dd mul_sloppy", 2,
	        _bncneon_rdd_mul_sloppy(vr, va, vb),
	        _bncsve2_rdd_mul_sloppy(pg, &r0, &r1, LD2D(ba), LD2D(bb)); ST2D);
	AUDIT_D("dd add_bf", 2,
	        _bncneon_rdd_add_bf(vr, va, vb),
	        _bncsve2_rdd_add_bf(pg, &r0, &r1, LD2D(ba), LD2D(bb)); ST2D);
	AUDIT_D("dd mul_bf", 2,
	        _bncneon_rdd_mul_bf(vr, va, vb),
	        _bncsve2_rdd_mul_bf(pg, &r0, &r1, LD2D(ba), LD2D(bb)); ST2D);
	AUDIT_D("td addq", 3,
	        _bncneon_rtd_addq(vr, va, vb),
	        _bncsve2_rtd_addq(pg, &r0, &r1, &r2, LD3D(ba), LD3D(bb)); ST3D);
	AUDIT_D("td mulq", 3,
	        _bncneon_rtd_mulq(vr, va, vb),
	        _bncsve2_rtd_mulq(pg, &r0, &r1, &r2, LD3D(ba), LD3D(bb)); ST3D);
	AUDIT_D("td add_bf", 3,
	        _bncneon_rtd_add_bf(vr, va, vb),
	        _bncsve2_rtd_add_bf(pg, &r0, &r1, &r2, LD3D(ba), LD3D(bb)); ST3D);
	AUDIT_D("td mul_bf", 3,
	        _bncneon_rtd_mul_bf(vr, va, vb),
	        _bncsve2_rtd_mul_bf(pg, &r0, &r1, &r2, LD3D(ba), LD3D(bb)); ST3D);
	AUDIT_D("qd add_sloppy", 4,
	        _bncneon_rqd_add_sloppy(vr, va, vb),
	        _bncsve2_rqd_add_sloppy(pg, &r0, &r1, &r2, &r3, LD4D(ba), LD4D(bb)); ST4D);
	AUDIT_D("qd mul_sloppy", 4,
	        _bncneon_rqd_mul_sloppy(vr, va, vb),
	        _bncsve2_rqd_mul_sloppy(pg, &r0, &r1, &r2, &r3, LD4D(ba), LD4D(bb)); ST4D);
	AUDIT_D("qd add_bf", 4,
	        _bncneon_rqd_add_bf(vr, va, vb),
	        _bncsve2_rqd_add_bf(pg, &r0, &r1, &r2, &r3, LD4D(ba), LD4D(bb)); ST4D);
	AUDIT_D("qd mul_bf", 4,
	        _bncneon_rqd_mul_bf(vr, va, vb),
	        _bncsve2_rqd_mul_bf(pg, &r0, &r1, &r2, &r3, LD4D(ba), LD4D(bb)); ST4D);

	printf("\n%s\n", bad_total == 0 ? "SVE2 AND NEON AGREE BITWISE EVERYWHERE"
	                                : "*** SVE2/NEON MISMATCHES FOUND (see above) ***");
	return bad_total == 0 ? 0 : 1;
}
