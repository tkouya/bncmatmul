/*
 * test_lib_fma.c : smoke test of the BNC_USE_NEW_FMA wiring inside the library
 *                  kernels (add_cmul_*vector = AXPY, mul_*matrix_*vec = GEMV,
 *                  mul_*matrix = GEMM) for DD / TD / QD.
 *
 * Dumps the raw results to stdout in a machine-readable form so that a build
 * with -DBNC_USE_NEW_FMA can be compared against the baseline build: the two
 * must agree to the working precision of the type (they use different
 * algorithms, so they are not bitwise equal).
 *
 * build: link src/{dd,td,qd}linear.c compiled with the same flags.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bnc_common.h"
#include "ddlinear.h"
#include "tdlinear.h"
#include "qdlinear.h"

static unsigned long rs = 12345678901UL;
static double urand(void)
{
	rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17;
	return (double)(rs >> 11) / 9007199254740992.0;
}

#define NV 1024
#define NM 64

#define RUN(TY, TYS, MT, VT, PS, initmat, initvec, setmij, setvi, getvi, \
            axpyfn, gemvfn, gemmfn, freemat, freevec)                        \
	do {                                                                     \
		MT A = initmat(NM, NM), B = initmat(NM, NM), C = initmat(NM, NM);    \
		VT x = initvec(NV), y = initvec(NV), z = initvec(NV);                \
		VT xv = initvec(NM), yv = initvec(NM);                               \
		double val[4], alpha[4];                                             \
		long i, j;                                                           \
		int k;                                                               \
		rs = 12345678901UL;                                                  \
		for (k = 0; k < PS; k++) alpha[k] = ldexp(2.0 * urand() - 1.0, -53 * k); \
		for (i = 0; i < NV; i++) {                                           \
			for (k = 0; k < PS; k++) val[k] = ldexp(2.0 * urand() - 1.0, -53 * k); \
			setvi(x, i, val);                                                \
			for (k = 0; k < PS; k++) val[k] = ldexp(2.0 * urand() - 1.0, -53 * k); \
			setvi(y, i, val);                                                \
		}                                                                    \
		for (i = 0; i < NM; i++) {                                           \
			for (k = 0; k < PS; k++) val[k] = ldexp(2.0 * urand() - 1.0, -53 * k); \
			setvi(xv, i, val);                                               \
			for (j = 0; j < NM; j++) {                                       \
				for (k = 0; k < PS; k++) val[k] = ldexp(2.0 * urand() - 1.0, -53 * k); \
				setmij(A, i, j, val);                                        \
				for (k = 0; k < PS; k++) val[k] = ldexp(2.0 * urand() - 1.0, -53 * k); \
				setmij(B, i, j, val);                                        \
			}                                                                \
		}                                                                    \
		axpyfn(z, y, alpha, x);                                              \
		for (i = 0; i < NV; i += 97)                                         \
			for (k = 0; k < PS; k++) printf(TYS " AXPY %ld_%d %.17e\n", i, k, getvi(z, i)[k]);                \
		gemvfn(yv, A, xv);                                                   \
		for (i = 0; i < NM; i += 7)                                          \
			for (k = 0; k < PS; k++) printf(TYS " GEMV %ld_%d %.17e\n", i, k, getvi(yv, i)[k]);               \
		gemmfn(C, A, B);                                                     \
		for (i = 0; i < NM; i += 7)                                          \
			for (j = 0; j < NM; j += 11)                                     \
				for (k = 0; k < PS; k++)                                     \
					printf(TYS " GEMM %ld_%ld_%d %.17e\n", i, j, k,           \
					       get_##TY##matrix_ij(C, i, j)[k]);                  \
		freemat(A); freemat(B); freemat(C);                                  \
		freevec(x); freevec(y); freevec(z); freevec(xv); freevec(yv);         \
	} while (0)

int main(void)
{
#ifdef BNC_USE_NEW_FMA
	printf("# build: BNC_USE_NEW_FMA\n");
#else
	printf("# build: baseline\n");
#endif
	RUN(dd, "dd", DDMatrix, DDVector, DDSIZE, init_ddmatrix, init_ddvector,
	    set_ddmatrix_ij, set_ddvector_i, get_ddvector_i,
	    add_cmul_ddvector, mul_ddmatrix_ddvec, mul_ddmatrix,
	    free_ddmatrix, free_ddvector);
	RUN(td, "td", TDMatrix, TDVector, TDSIZE, init_tdmatrix, init_tdvector,
	    set_tdmatrix_ij, set_tdvector_i, get_tdvector_i,
	    add_cmul_tdvector, mul_tdmatrix_tdvec, mul_tdmatrix,
	    free_tdmatrix, free_tdvector);
	RUN(qd, "qd", QDMatrix, QDVector, QDSIZE, init_qdmatrix, init_qdvector,
	    set_qdmatrix_ij, set_qdvector_i, get_qdvector_i,
	    add_cmul_qdvector, mul_qdmatrix_qdvec, mul_qdmatrix,
	    free_qdmatrix, free_qdvector);
	return 0;
}
