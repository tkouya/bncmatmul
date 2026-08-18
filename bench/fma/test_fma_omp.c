/*
 * test_fma_omp.c : OpenMP thread-safety of the branch-free DW/TW/QW FMA.
 *
 * The proposed FMA routines are pure functions of their arguments (no static
 * or global temporaries), so an OpenMP-parallel evaluation must reproduce the
 * serial evaluation bit for bit.  This test checks that for every backend
 * (scalar / NEON / SVE2), every type (DD/TD/QD, DS/TS/QS) and 1..N threads.
 *
 * build: gcc -O3 -ffp-contract=off -fopenmp -mcpu=neoverse-v2 \
 *            -DBNC_ENABLE_NEON -DBNC_ENABLE_SVE2 -Iinclude \
 *            bench/fma/test_fma_omp.c -lm -o bench/fma/test_fma_omp
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "bnc_common.h"
#include "bncfma.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#include "neon/_bncneon_fma.h"
#endif
#if defined(__ARM_SVE2)
#include <arm_sve.h>
#include "sve2/_bncsve2_fma.h"
#endif

#define N 400000
#define MAXK 4

static double *X[MAXK], *Y[MAXK], *C[MAXK], *Zs[MAXK], *Zp[MAXK];
static float  *Xf[MAXK], *Yf[MAXK], *Cf[MAXK], *Zsf[MAXK], *Zpf[MAXK];

static unsigned long rs = 20260728UL;
static double u(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
                        return (double)(rs >> 11) / 9007199254740992.0; }

static void fma_d(double *z, const double *x, const double *y, const double *c, int K)
{
	if (K == 2) bnc_dwfma(z, x, y, c);
	else if (K == 3) bnc_twfma(z, x, y, c);
	else bnc_qwfma(z, x, y, c);
}
static void fma_f(float *z, const float *x, const float *y, const float *c, int K)
{
	if (K == 2) bnc_dwfmaf(z, x, y, c);
	else if (K == 3) bnc_twfmaf(z, x, y, c);
	else bnc_qwfmaf(z, x, y, c);
}

int main(int argc, char *argv[])
{
	int K, k, nthreads = 1, bad = 0;
	long i;

	(void)argc; (void)argv;
#ifdef _OPENMP
	nthreads = omp_get_max_threads();
#endif
	for (k = 0; k < MAXK; k++) {
		X[k] = malloc(sizeof(double) * N); Y[k] = malloc(sizeof(double) * N);
		C[k] = malloc(sizeof(double) * N); Zs[k] = malloc(sizeof(double) * N);
		Zp[k] = malloc(sizeof(double) * N);
		Xf[k] = malloc(sizeof(float) * N); Yf[k] = malloc(sizeof(float) * N);
		Cf[k] = malloc(sizeof(float) * N); Zsf[k] = malloc(sizeof(float) * N);
		Zpf[k] = malloc(sizeof(float) * N);
	}
	for (i = 0; i < N; i++)
		for (k = 0; k < MAXK; k++) {
			X[k][i] = ldexp(2.0 * u() - 1.0, -53 * k);
			Y[k][i] = ldexp(2.0 * u() - 1.0, -53 * k);
			C[k][i] = ldexp(2.0 * u() - 1.0, -53 * k);
			Xf[k][i] = (float)ldexp(2.0 * u() - 1.0, -24 * k);
			Yf[k][i] = (float)ldexp(2.0 * u() - 1.0, -24 * k);
			Cf[k][i] = (float)ldexp(2.0 * u() - 1.0, -24 * k);
		}

	printf("=== OpenMP thread-safety of the branch-free FMA (%d threads, N=%d) ===\n",
	       nthreads, N);

	for (K = 2; K <= 4; K++) {
		const char *dn[5] = { "", "", "DD", "TD", "QD" };
		const char *fn[5] = { "", "", "DS", "TS", "QS" };
		/* ---- serial ---- */
		for (i = 0; i < N; i++) {
			double x[MAXK], y[MAXK], c[MAXK], z[MAXK];
			float  xf[MAXK], yf[MAXK], cf[MAXK], zf[MAXK];
			for (k = 0; k < K; k++) { x[k] = X[k][i]; y[k] = Y[k][i]; c[k] = C[k][i];
			                          xf[k] = Xf[k][i]; yf[k] = Yf[k][i]; cf[k] = Cf[k][i]; }
			fma_d(z, x, y, c, K);
			fma_f(zf, xf, yf, cf, K);
			for (k = 0; k < K; k++) { Zs[k][i] = z[k]; Zsf[k][i] = zf[k]; }
		}
		/* ---- OpenMP ---- */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 997)
#endif
		for (i = 0; i < N; i++) {
			double x[MAXK], y[MAXK], c[MAXK], z[MAXK];
			float  xf[MAXK], yf[MAXK], cf[MAXK], zf[MAXK];
			int kk;
			for (kk = 0; kk < K; kk++) { x[kk] = X[kk][i]; y[kk] = Y[kk][i]; c[kk] = C[kk][i];
			                             xf[kk] = Xf[kk][i]; yf[kk] = Yf[kk][i]; cf[kk] = Cf[kk][i]; }
			fma_d(z, x, y, c, K);
			fma_f(zf, xf, yf, cf, K);
			for (kk = 0; kk < K; kk++) { Zp[kk][i] = z[kk]; Zpf[kk][i] = zf[kk]; }
		}
		{
			int d = 0, df = 0;
			for (k = 0; k < K; k++) {
				if (memcmp(Zs[k], Zp[k], sizeof(double) * N)) d = 1;
				if (memcmp(Zsf[k], Zpf[k], sizeof(float) * N)) df = 1;
			}
			printf("  %s scalar : %s      %s scalar : %s\n",
			       dn[K], d ? "MISMATCH" : "identical",
			       fn[K], df ? "MISMATCH" : "identical");
			bad += d + df;
		}

#if defined(__ARM_NEON)
		/* ---- NEON, OpenMP ---- */
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 997)
#endif
		for (i = 0; i < N; i += 2) {
			float64x2_t vx[MAXK], vy[MAXK], vc[MAXK], vz[MAXK];
			int kk;
			for (kk = 0; kk < K; kk++) {
				double tx[2] = { X[kk][i], X[kk][i + 1] };
				double ty[2] = { Y[kk][i], Y[kk][i + 1] };
				double tc[2] = { C[kk][i], C[kk][i + 1] };
				vx[kk] = vld1q_f64(tx); vy[kk] = vld1q_f64(ty); vc[kk] = vld1q_f64(tc);
			}
			if (K == 2) _bncneon_dwfma(vz, vx, vy, vc);
			else if (K == 3) _bncneon_twfma(vz, vx, vy, vc);
			else _bncneon_qwfma(vz, vx, vy, vc);
			for (kk = 0; kk < K; kk++) { double o[2]; vst1q_f64(o, vz[kk]);
				Zp[kk][i] = o[0]; Zp[kk][i + 1] = o[1]; }
		}
		{
			int d = 0;
			for (k = 0; k < K; k++) if (memcmp(Zs[k], Zp[k], sizeof(double) * N)) d = 1;
			printf("  %s NEON   : %s (vs the serial scalar reference)\n",
			       dn[K], d ? "MISMATCH" : "identical");
			bad += d;
		}
#endif
#if defined(__ARM_SVE2)
		/* ---- SVE2, OpenMP ---- */
		{
			long vl = (long)svcntd();
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 997)
#endif
			for (i = 0; i < N; i += vl) {
				svbool_t pg = svptrue_b64();
				double bx[MAXK][8], by[MAXK][8], bc[MAXK][8], bz[MAXK][8];
				int kk;
				long l;
				for (kk = 0; kk < K; kk++)
					for (l = 0; l < vl; l++) {
						bx[kk][l] = X[kk][i + l]; by[kk][l] = Y[kk][i + l]; bc[kk][l] = C[kk][i + l];
					}
				if (K == 2) {
					svfloat64_t z0, z1;
					_bncsve2_dwfma(pg, &z0, &z1,
					        svld1_f64(pg, bx[0]), svld1_f64(pg, bx[1]),
					        svld1_f64(pg, by[0]), svld1_f64(pg, by[1]),
					        svld1_f64(pg, bc[0]), svld1_f64(pg, bc[1]));
					svst1_f64(pg, bz[0], z0); svst1_f64(pg, bz[1], z1);
				} else if (K == 3) {
					svfloat64_t z0, z1, z2;
					_bncsve2_twfma(pg, &z0, &z1, &z2,
					        svld1_f64(pg, bx[0]), svld1_f64(pg, bx[1]), svld1_f64(pg, bx[2]),
					        svld1_f64(pg, by[0]), svld1_f64(pg, by[1]), svld1_f64(pg, by[2]),
					        svld1_f64(pg, bc[0]), svld1_f64(pg, bc[1]), svld1_f64(pg, bc[2]));
					svst1_f64(pg, bz[0], z0); svst1_f64(pg, bz[1], z1); svst1_f64(pg, bz[2], z2);
				} else {
					svfloat64_t z0, z1, z2, z3;
					_bncsve2_qwfma(pg, &z0, &z1, &z2, &z3,
					        svld1_f64(pg, bx[0]), svld1_f64(pg, bx[1]), svld1_f64(pg, bx[2]), svld1_f64(pg, bx[3]),
					        svld1_f64(pg, by[0]), svld1_f64(pg, by[1]), svld1_f64(pg, by[2]), svld1_f64(pg, by[3]),
					        svld1_f64(pg, bc[0]), svld1_f64(pg, bc[1]), svld1_f64(pg, bc[2]), svld1_f64(pg, bc[3]));
					svst1_f64(pg, bz[0], z0); svst1_f64(pg, bz[1], z1);
					svst1_f64(pg, bz[2], z2); svst1_f64(pg, bz[3], z3);
				}
				for (kk = 0; kk < K; kk++)
					for (l = 0; l < vl; l++) Zp[kk][i + l] = bz[kk][l];
			}
			{
				int d = 0;
				for (k = 0; k < K; k++) if (memcmp(Zs[k], Zp[k], sizeof(double) * N)) d = 1;
				printf("  %s SVE2   : %s (vs the serial scalar reference)\n",
				       dn[K], d ? "MISMATCH" : "identical");
				bad += d;
			}
		}
#endif
	}

	printf("\n%s\n", bad == 0
	       ? "OpenMP-PARALLEL RESULTS ARE BITWISE IDENTICAL TO SERIAL (no data race)"
	       : "*** OpenMP MISMATCH ***");
	return bad == 0 ? 0 : 1;
}
