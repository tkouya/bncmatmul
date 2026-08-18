/*
 * fma_kernels.inc : AXPY / GEMV / GEMM kernel template.
 *
 * Included once per (type, variant) pair.  Required macros:
 *   TAG   : name suffix token, e.g. dd_q
 *   K     : number of components (2 / 3 / 4)
 *   MAC(r, a, b, c) : r[K] := a[K] * b[K] + c[K]
 * Backend macros (defined by the includer):
 *   VEC, LANES, VLOAD(dst, ptr), VSTORE(ptr, src), VSPLAT(dst, scalar), VZERO(dst)
 *
 * Storage is SoA: one BASE* per component, n padded up to a multiple of LANES.
 * GEMV uses a column-major matrix so that the row index is the unit-stride
 * (vectorized) one; GEMM uses row-major A/B/C and vectorizes over columns.
 */

#define _FMA_CAT(a, b) a##b
#define FMA_CAT(a, b) _FMA_CAT(a, b)
#define FN(name) FMA_CAT(name##_, TAG)

/* y := alpha * x + y  (n elements) */
static void FN(axpy)(long n, const VEC alpha[K], BASE *const X[K], BASE *Y[K])
{
	long i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (i = 0; i < n; i += LANES) {
		VEC x[K], y[K], r[K];
		int k;
		for (k = 0; k < K; k++) VLOAD(x[k], X[k] + i);
		for (k = 0; k < K; k++) VLOAD(y[k], Y[k] + i);
		MAC(r, alpha, x, y);
		for (k = 0; k < K; k++) VSTORE(Y[k] + i, r[k]);
	}
}

/* y := A * x   (A is n x n, column-major: A[k][j*n + i] = a_ij) */
static void FN(gemv)(long n, BASE *const A[K], BASE *const X[K], BASE *Y[K])
{
	long i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (i = 0; i < n; i += LANES) {
		VEC acc[K], a[K], xs[K], r[K];
		long j;
		int k;
		for (k = 0; k < K; k++) VZERO(acc[k]);
		for (j = 0; j < n; j++) {
			for (k = 0; k < K; k++) VSPLAT(xs[k], X[k][j]);
			for (k = 0; k < K; k++) VLOAD(a[k], A[k] + j * n + i);
			MAC(r, a, xs, acc);
			for (k = 0; k < K; k++) acc[k] = r[k];
		}
		for (k = 0; k < K; k++) VSTORE(Y[k] + i, acc[k]);
	}
}

/* C := A * B   (all n x n, row-major) */
static void FN(gemm)(long n, BASE *const A[K], BASE *const B[K], BASE *C[K])
{
	long i;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
	for (i = 0; i < n; i++) {
		long j, l;
		for (j = 0; j < n; j += LANES) {
			VEC acc[K], av[K], bv[K], r[K];
			int k;
			for (k = 0; k < K; k++) VZERO(acc[k]);
			for (l = 0; l < n; l++) {
				for (k = 0; k < K; k++) VSPLAT(av[k], A[k][i * n + l]);
				for (k = 0; k < K; k++) VLOAD(bv[k], B[k] + l * n + j);
				MAC(r, av, bv, acc);
				for (k = 0; k < K; k++) acc[k] = r[k];
			}
			for (k = 0; k < K; k++) VSTORE(C[k] + i * n + j, acc[k]);
		}
	}
}

#undef FN
#undef FMA_CAT
#undef _FMA_CAT
#undef TAG
#undef K
#undef MAC
