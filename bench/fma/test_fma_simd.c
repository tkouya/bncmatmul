/*
 * test_fma_simd.c : bitwise agreement of the branch-free DW/TW/QW FMA
 *                   between the scalar reference, NEON and SVE2.
 *
 * "Vectorization does not change the rounding": every backend must reproduce
 * the scalar netlist bit for bit.
 *
 * build (both backends in one binary):
 *   gcc -O3 -ffp-contract=off -mcpu=native -DBNC_ENABLE_NEON -DBNC_ENABLE_SVE2 \
 *       -Iinclude bench/fma/test_fma_simd.c -lm -o bench/fma/test_fma_simd
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "bnc_common.h"   /* maps __ARM_FEATURE_SVE2 -> __ARM_SVE2 */
#include "bncfma.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#include "neon/_bncneon_deft.h"
#include "neon/_bncneon_feft.h"
#include "neon/_bncneon_fma.h"
#endif

#if defined(__ARM_SVE2)
#include <arm_sve.h>
#include "sve2/_bncsve2_deft.h"
#include "sve2/_bncsve2_feft.h"
#include "sve2/_bncsve2_fma.h"
#endif

#define MAXK 4
#define MAXLANE 64

static double rnd_d(void)
{
	/* random double with a wide exponent range */
	double m = (double)rand() / (double)RAND_MAX * 2.0 - 1.0;
	return ldexp(m, (rand() % 41) - 20);
}
/* build a non-overlapping K-word operand: geometrically decaying words,
   then one downward quick_two_sum sweep to renormalize */
static void rand_mw(double *z, int K)
{
	int i;
	z[0] = rnd_d();
	for (i = 1; i < K; i++) {
		double f = 0.5 + 0.5 * ((double)rand() / (double)RAND_MAX);
		z[i] = ldexp(z[i - 1] * f, -53);
		if (rand() & 1) z[i] = -z[i];
	}
	for (i = 0; i + 1 < K; i++) z[i] = quick_two_sum(z[i], z[i + 1], &z[i + 1]);
}
static void rand_mwf(float *z, int K)
{
	int i;
	z[0] = (float)rnd_d();
	for (i = 1; i < K; i++) {
		float f = 0.5f + 0.5f * ((float)rand() / (float)RAND_MAX);
		z[i] = ldexpf(z[i - 1] * f, -24);
		if (rand() & 1) z[i] = -z[i];
	}
	for (i = 0; i + 1 < K; i++) z[i] = fquick_two_sum(z[i], z[i + 1], &z[i + 1]);
}

static int bitcmp(const void *a, const void *b, size_t n) { return memcmp(a, b, n) != 0; }

int main(int argc, char *argv[])
{
	long ntrial = (argc > 1) ? atol(argv[1]) : 200000L;
	long i;
	int k, lane;
	long bad_neon[3] = {0, 0, 0}, bad_sve[3] = {0, 0, 0};
	long bad_neonf[3] = {0, 0, 0}, bad_svef[3] = {0, 0, 0};
	long nchk_neon = 0, nchk_sve = 0;

	srand(20260728);
	printf("=== branch-free FMA: backend bitwise agreement ===\n");
	printf("scalar reference vs "
#if defined(__ARM_NEON)
	       "NEON "
#endif
#if defined(__ARM_SVE2)
	       "SVE2 "
#endif
	       "\n");

#if defined(__ARM_NEON)
	printf("NEON lanes (double) = 2, (float) = 4\n");
	for (i = 0; i < ntrial; i += 2) {
		double x[2][MAXK], y[2][MAXK], c[2][MAXK], zs[2][MAXK];
		float  xf[4][MAXK], yf[4][MAXK], cf[4][MAXK], zsf[4][MAXK];
		float64x2_t vx[MAXK], vy[MAXK], vc[MAXK], vz[MAXK];
		float32x4_t sx[MAXK], sy[MAXK], sc[MAXK], sz[MAXK];
		double out[MAXK][2];
		float  outf[MAXK][4];
		int K, K3;

		for (lane = 0; lane < 2; lane++) { rand_mw(x[lane], MAXK); rand_mw(y[lane], MAXK); rand_mw(c[lane], MAXK); }
		for (lane = 0; lane < 4; lane++) { rand_mwf(xf[lane], MAXK); rand_mwf(yf[lane], MAXK); rand_mwf(cf[lane], MAXK); }

		for (K3 = 0; K3 < 3; K3++) {
			K = K3 + 2;
			/* --- scalar --- */
			for (lane = 0; lane < 2; lane++) {
				if (K == 2)      bnc_dwfma(zs[lane], x[lane], y[lane], c[lane]);
				else if (K == 3) bnc_twfma(zs[lane], x[lane], y[lane], c[lane]);
				else             bnc_qwfma(zs[lane], x[lane], y[lane], c[lane]);
			}
			for (lane = 0; lane < 4; lane++) {
				if (K == 2)      bnc_dwfmaf(zsf[lane], xf[lane], yf[lane], cf[lane]);
				else if (K == 3) bnc_twfmaf(zsf[lane], xf[lane], yf[lane], cf[lane]);
				else             bnc_qwfmaf(zsf[lane], xf[lane], yf[lane], cf[lane]);
			}
			/* --- NEON double --- */
			for (k = 0; k < K; k++) {
				double tx[2] = { x[0][k], x[1][k] }, ty[2] = { y[0][k], y[1][k] }, tc[2] = { c[0][k], c[1][k] };
				vx[k] = vld1q_f64(tx); vy[k] = vld1q_f64(ty); vc[k] = vld1q_f64(tc);
			}
			if (K == 2)      _bncneon_dwfma(vz, vx, vy, vc);
			else if (K == 3) _bncneon_twfma(vz, vx, vy, vc);
			else             _bncneon_qwfma(vz, vx, vy, vc);
			for (k = 0; k < K; k++) vst1q_f64(out[k], vz[k]);
			for (lane = 0; lane < 2; lane++)
				for (k = 0; k < K; k++)
					if (bitcmp(&out[k][lane], &zs[lane][k], sizeof(double))) { bad_neon[K3]++; k = K; }
			/* --- NEON float --- */
			for (k = 0; k < K; k++) {
				float tx[4] = { xf[0][k], xf[1][k], xf[2][k], xf[3][k] };
				float ty[4] = { yf[0][k], yf[1][k], yf[2][k], yf[3][k] };
				float tc[4] = { cf[0][k], cf[1][k], cf[2][k], cf[3][k] };
				sx[k] = vld1q_f32(tx); sy[k] = vld1q_f32(ty); sc[k] = vld1q_f32(tc);
			}
			if (K == 2)      _bncneon_dwfmaf(sz, sx, sy, sc);
			else if (K == 3) _bncneon_twfmaf(sz, sx, sy, sc);
			else             _bncneon_qwfmaf(sz, sx, sy, sc);
			for (k = 0; k < K; k++) vst1q_f32(outf[k], sz[k]);
			for (lane = 0; lane < 4; lane++)
				for (k = 0; k < K; k++)
					if (bitcmp(&outf[k][lane], &zsf[lane][k], sizeof(float))) { bad_neonf[K3]++; k = K; }
		}
		nchk_neon += 2;
	}
	printf("  NEON double  DD/TD/QD mismatches: %ld / %ld / %ld  (of %ld each)\n",
	       bad_neon[0], bad_neon[1], bad_neon[2], nchk_neon);
	printf("  NEON float   DS/TS/QS mismatches: %ld / %ld / %ld  (of %ld each)\n",
	       bad_neonf[0], bad_neonf[1], bad_neonf[2], nchk_neon * 2);
#endif

#if defined(__ARM_SVE2)
	{
		int VLd = (int)svcntd(), VLf = (int)svcntw();
		printf("SVE2 lanes (double) = %d, (float) = %d\n", VLd, VLf);
		if (VLd > MAXLANE) { printf("  (VL too large for this test)\n"); return 1; }
		for (i = 0; i < ntrial; i += VLd) {
			double x[MAXLANE][MAXK], y[MAXLANE][MAXK], c[MAXLANE][MAXK], zs[MAXLANE][MAXK];
			float  xf[MAXLANE][MAXK], yf[MAXLANE][MAXK], cf[MAXLANE][MAXK], zsf[MAXLANE][MAXK];
			double bx[MAXK][MAXLANE], by[MAXK][MAXLANE], bc[MAXK][MAXLANE], bz[MAXK][MAXLANE];
			float  fx[MAXK][MAXLANE], fy[MAXK][MAXLANE], fc[MAXK][MAXLANE], fz[MAXK][MAXLANE];
			svbool_t pgd = svptrue_b64(), pgf = svptrue_b32();
			int K, K3;

			for (lane = 0; lane < VLd; lane++) { rand_mw(x[lane], MAXK); rand_mw(y[lane], MAXK); rand_mw(c[lane], MAXK); }
			for (lane = 0; lane < VLf; lane++) { rand_mwf(xf[lane], MAXK); rand_mwf(yf[lane], MAXK); rand_mwf(cf[lane], MAXK); }
			for (k = 0; k < MAXK; k++) {
				for (lane = 0; lane < VLd; lane++) { bx[k][lane] = x[lane][k]; by[k][lane] = y[lane][k]; bc[k][lane] = c[lane][k]; }
				for (lane = 0; lane < VLf; lane++) { fx[k][lane] = xf[lane][k]; fy[k][lane] = yf[lane][k]; fc[k][lane] = cf[lane][k]; }
			}

			for (K3 = 0; K3 < 3; K3++) {
				K = K3 + 2;
				for (lane = 0; lane < VLd; lane++) {
					if (K == 2)      bnc_dwfma(zs[lane], x[lane], y[lane], c[lane]);
					else if (K == 3) bnc_twfma(zs[lane], x[lane], y[lane], c[lane]);
					else             bnc_qwfma(zs[lane], x[lane], y[lane], c[lane]);
				}
				for (lane = 0; lane < VLf; lane++) {
					if (K == 2)      bnc_dwfmaf(zsf[lane], xf[lane], yf[lane], cf[lane]);
					else if (K == 3) bnc_twfmaf(zsf[lane], xf[lane], yf[lane], cf[lane]);
					else             bnc_qwfmaf(zsf[lane], xf[lane], yf[lane], cf[lane]);
				}
				if (K == 2) {
					svfloat64_t z0, z1;
					_bncsve2_dwfma(pgd, &z0, &z1,
					        svld1_f64(pgd, bx[0]), svld1_f64(pgd, bx[1]),
					        svld1_f64(pgd, by[0]), svld1_f64(pgd, by[1]),
					        svld1_f64(pgd, bc[0]), svld1_f64(pgd, bc[1]));
					svst1_f64(pgd, bz[0], z0); svst1_f64(pgd, bz[1], z1);
				} else if (K == 3) {
					svfloat64_t z0, z1, z2;
					_bncsve2_twfma(pgd, &z0, &z1, &z2,
					        svld1_f64(pgd, bx[0]), svld1_f64(pgd, bx[1]), svld1_f64(pgd, bx[2]),
					        svld1_f64(pgd, by[0]), svld1_f64(pgd, by[1]), svld1_f64(pgd, by[2]),
					        svld1_f64(pgd, bc[0]), svld1_f64(pgd, bc[1]), svld1_f64(pgd, bc[2]));
					svst1_f64(pgd, bz[0], z0); svst1_f64(pgd, bz[1], z1); svst1_f64(pgd, bz[2], z2);
				} else {
					svfloat64_t z0, z1, z2, z3;
					_bncsve2_qwfma(pgd, &z0, &z1, &z2, &z3,
					        svld1_f64(pgd, bx[0]), svld1_f64(pgd, bx[1]), svld1_f64(pgd, bx[2]), svld1_f64(pgd, bx[3]),
					        svld1_f64(pgd, by[0]), svld1_f64(pgd, by[1]), svld1_f64(pgd, by[2]), svld1_f64(pgd, by[3]),
					        svld1_f64(pgd, bc[0]), svld1_f64(pgd, bc[1]), svld1_f64(pgd, bc[2]), svld1_f64(pgd, bc[3]));
					svst1_f64(pgd, bz[0], z0); svst1_f64(pgd, bz[1], z1);
					svst1_f64(pgd, bz[2], z2); svst1_f64(pgd, bz[3], z3);
				}
				for (lane = 0; lane < VLd; lane++)
					for (k = 0; k < K; k++)
						if (bitcmp(&bz[k][lane], &zs[lane][k], sizeof(double))) { bad_sve[K3]++; k = K; }

				if (K == 2) {
					svfloat32_t z0, z1;
					_bncsve2_dwfmaf(pgf, &z0, &z1,
					        svld1_f32(pgf, fx[0]), svld1_f32(pgf, fx[1]),
					        svld1_f32(pgf, fy[0]), svld1_f32(pgf, fy[1]),
					        svld1_f32(pgf, fc[0]), svld1_f32(pgf, fc[1]));
					svst1_f32(pgf, fz[0], z0); svst1_f32(pgf, fz[1], z1);
				} else if (K == 3) {
					svfloat32_t z0, z1, z2;
					_bncsve2_twfmaf(pgf, &z0, &z1, &z2,
					        svld1_f32(pgf, fx[0]), svld1_f32(pgf, fx[1]), svld1_f32(pgf, fx[2]),
					        svld1_f32(pgf, fy[0]), svld1_f32(pgf, fy[1]), svld1_f32(pgf, fy[2]),
					        svld1_f32(pgf, fc[0]), svld1_f32(pgf, fc[1]), svld1_f32(pgf, fc[2]));
					svst1_f32(pgf, fz[0], z0); svst1_f32(pgf, fz[1], z1); svst1_f32(pgf, fz[2], z2);
				} else {
					svfloat32_t z0, z1, z2, z3;
					_bncsve2_qwfmaf(pgf, &z0, &z1, &z2, &z3,
					        svld1_f32(pgf, fx[0]), svld1_f32(pgf, fx[1]), svld1_f32(pgf, fx[2]), svld1_f32(pgf, fx[3]),
					        svld1_f32(pgf, fy[0]), svld1_f32(pgf, fy[1]), svld1_f32(pgf, fy[2]), svld1_f32(pgf, fy[3]),
					        svld1_f32(pgf, fc[0]), svld1_f32(pgf, fc[1]), svld1_f32(pgf, fc[2]), svld1_f32(pgf, fc[3]));
					svst1_f32(pgf, fz[0], z0); svst1_f32(pgf, fz[1], z1);
					svst1_f32(pgf, fz[2], z2); svst1_f32(pgf, fz[3], z3);
				}
				for (lane = 0; lane < VLf; lane++)
					for (k = 0; k < K; k++)
						if (bitcmp(&fz[k][lane], &zsf[lane][k], sizeof(float))) { bad_svef[K3]++; k = K; }
			}
			nchk_sve += VLd;
		}
		printf("  SVE2 double  DD/TD/QD mismatches: %ld / %ld / %ld  (of %ld each)\n",
		       bad_sve[0], bad_sve[1], bad_sve[2], nchk_sve);
		printf("  SVE2 float   DS/TS/QS mismatches: %ld / %ld / %ld  (of %ld each)\n",
		       bad_svef[0], bad_svef[1], bad_svef[2], nchk_sve * (svcntw() / svcntd()));
	}
#endif

	{
		long total = 0;
		for (k = 0; k < 3; k++) total += bad_neon[k] + bad_sve[k] + bad_neonf[k] + bad_svef[k];
		printf("\n%s\n", (total == 0) ? "ALL BACKENDS BITWISE IDENTICAL TO THE SCALAR REFERENCE"
		                              : "*** MISMATCH DETECTED ***");
		return (total == 0) ? 0 : 1;
	}
}
