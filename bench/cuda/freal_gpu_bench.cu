/****************************************************************************/
/* freal_gpu_bench.cu : GPU AXPY / GEMV / GEMM with mpc_cuda's FIXED-precision */
/*   register-resident type cu_freal<PREC>.                                   */
/*                                                                            */
/*   Third arm of the EFT-vs-MPFR comparison:                                 */
/*     EFT (gdtq)   fixed, few-component (2..4 doubles), hand-written EFT     */
/*     cu_freal     fixed, PREC-bit significand held in REGISTERS             */
/*     cu_mpfr      arbitrary precision, limbs in a per-thread arena          */
/*   Same kernel structure, sizes, launch grid and timing protocol as         */
/*   mpfr_gpu_bench.cu, so the only difference is the arithmetic.             */
/*                                                                            */
/*   NOTE: cu_freal<PB> requires PB to be a positive multiple of 32           */
/*   (static_assert in cu_freal.cuh), so it cannot hit the EFT bit-widths     */
/*   53/106/159/212 exactly. Build it at the smallest multiple of 32 that is  */
/*   >= the EFT width (53->64, 106->128, 159->160, 212->224, 24->32, 48->64,  */
/*   72->96, 96->96); the emitted `bits` column records what was really used. */
/*                                                                            */
/*   Output CSV (same schema as mpfr_gpu_bench.cu):                           */
/*     op,kind,type,bits,N,gpu_sec,gpu_mflops,relerr        kind = "freal"    */
/****************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>

#include "mpc_cuda/cu_freal.cuh"

#ifndef PREC
#define PREC 128
#endif

using namespace cu_fp;
typedef cu_freal<PREC> F;

/*------------------ element routines (host + device) ------------------*/
__host__ __device__ static void
fdot_row(const double *A, const double *x, int n, double *y)
{
  F acc = F::from_double(0.0);
  for (int j = 0; j < n; ++j)
    acc = cu_fadd<PREC>(acc, cu_fmul<PREC>(F::from_double(A[j]),
                                           F::from_double(x[j])));
  *y = acc.to_double();
}

__host__ __device__ static void
fdot_ij(const double *A, const double *B, int n, int i, int j, double *c)
{
  F acc = F::from_double(0.0);
  for (int k = 0; k < n; ++k)
    acc = cu_fadd<PREC>(acc,
            cu_fmul<PREC>(F::from_double(A[(size_t) i * n + k]),
                          F::from_double(B[(size_t) k * n + j])));
  *c = acc.to_double();
}

__host__ __device__ static void
faxpy_i(const double *a, double al, const double *x, int i, double *y)
{
  F t = cu_fadd<PREC>(F::from_double(a[i]),
                      cu_fmul<PREC>(F::from_double(al), F::from_double(x[i])));
  *y = t.to_double();
}

/*------------------ device kernels ------------------*/
__global__ void kf_matvec(int n, const double *A, const double *x, double *y)
{
  int st = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += st)
    fdot_row(A + (size_t) i * n, x, n, &y[i]);
}
__global__ void kf_matmul(int n, const double *A, const double *B, double *C)
{
  int st = gridDim.x * blockDim.x; long tot = (long) n * n;
  for (long e = blockIdx.x * blockDim.x + threadIdx.x; e < tot; e += st)
    fdot_ij(A, B, n, (int) (e / n), (int) (e % n), &C[e]);
}
__global__ void kf_axpy(int n, const double *a, double al, const double *x, double *y)
{
  int st = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += st)
    faxpy_i(a, al, x, i, &y[i]);
}

/*------------------ harness ------------------*/
typedef std::chrono::high_resolution_clock clk;
static double secs(clk::time_point t0)
{ return std::chrono::duration<double>(clk::now() - t0).count(); }

static int LB = 256, LT = 128, g_reps = 3;
static double frand() { return (double) rand() / (double) RAND_MAX * 2.0 - 1.0; }
static FILE *g_out = nullptr;

static void emit(const char *op, const char *kind, const char *type, int bits,
                 long N, double tg, double relerr, double flop)
{
  double gmf = (tg > 0) ? flop / tg / 1e6 : 0;
  fprintf(g_out, "%s,%s,%s,%d,%ld,%.6e,%.4f,%.3e\n",
          op, kind, type, bits, N, tg, gmf, relerr);
  fflush(g_out);
}
template<typename Fn> static double tmin(Fn f)
{
  double b = 1e30;
  for (int r = 0; r < g_reps; ++r) {
    auto t0 = clk::now(); f(); cudaDeviceSynchronize();
    double s = secs(t0); if (s < b) b = s;
  }
  return b;
}

/*==================== benches ====================*/
static void freal_axpy(int N)
{
  size_t MV = (size_t) N * sizeof(double);
  double *a = (double *) malloc(MV), *x = (double *) malloc(MV), *yg = (double *) malloc(MV);
  srand(33); for (int i = 0; i < N; ++i) { a[i] = frand(); x[i] = frand(); }
  double al = 1.5;
  double *da, *dx, *dy;
  cudaMalloc(&da, MV); cudaMalloc(&dx, MV); cudaMalloc(&dy, MV);
  cudaMemcpy(da, a, MV, cudaMemcpyHostToDevice);
  cudaMemcpy(dx, x, MV, cudaMemcpyHostToDevice);
  int lb = (N + LT - 1) / LT; if (lb < 1) lb = 1;
  kf_axpy<<<lb, LT>>>(N, da, al, dx, dy); cudaDeviceSynchronize();
  double tg = tmin([&]{ kf_axpy<<<lb, LT>>>(N, da, al, dx, dy); });
  cudaMemcpy(yg, dy, MV, cudaMemcpyDeviceToHost);
  double mx = 0;
  for (int i = 0; i < N; ++i) {
    double ref = a[i] + al * x[i];
    double e = ref != 0 ? fabs((yg[i] - ref) / ref) : fabs(yg[i]);
    if (e > mx) mx = e;
  }
  emit("axpy", "freal", "freal", PREC, N, tg, mx, 2.0 * (double) N);
  cudaFree(da); cudaFree(dx); cudaFree(dy); free(a); free(x); free(yg);
}

static void freal_matvec(int N)
{
  size_t MA = (size_t) N * N * sizeof(double), MV = (size_t) N * sizeof(double);
  double *A = (double *) malloc(MA), *x = (double *) malloc(MV), *yg = (double *) malloc(MV);
  srand(11);
  for (int i = 0; i < N; ++i) { x[i] = frand(); for (int j = 0; j < N; ++j) A[(size_t) i * N + j] = frand(); }
  double *dA, *dx, *dy;
  cudaMalloc(&dA, MA); cudaMalloc(&dx, MV); cudaMalloc(&dy, MV);
  cudaMemcpy(dA, A, MA, cudaMemcpyHostToDevice);
  cudaMemcpy(dx, x, MV, cudaMemcpyHostToDevice);
  kf_matvec<<<LB, LT>>>(N, dA, dx, dy); cudaDeviceSynchronize();
  double tg = tmin([&]{ kf_matvec<<<LB, LT>>>(N, dA, dx, dy); });
  cudaMemcpy(yg, dy, MV, cudaMemcpyDeviceToHost);
  double mx = 0;
  for (int i = 0; i < N; ++i) {
    double ref = 0; for (int j = 0; j < N; ++j) ref += A[(size_t) i * N + j] * x[j];
    double e = ref != 0 ? fabs((yg[i] - ref) / ref) : fabs(yg[i]);
    if (e > mx) mx = e;
  }
  emit("matvec", "freal", "freal", PREC, N, tg, mx, 2.0 * (double) N * (double) N);
  cudaFree(dA); cudaFree(dx); cudaFree(dy); free(A); free(x); free(yg);
}

static void freal_matmul(int N)
{
  size_t MA = (size_t) N * N * sizeof(double);
  double *A = (double *) malloc(MA), *B = (double *) malloc(MA), *Cg = (double *) malloc(MA);
  srand(22); for (long e = 0; e < (long) N * N; ++e) { A[e] = frand(); B[e] = frand(); }
  double *dA, *dB, *dC;
  cudaMalloc(&dA, MA); cudaMalloc(&dB, MA); cudaMalloc(&dC, MA);
  cudaMemcpy(dA, A, MA, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B, MA, cudaMemcpyHostToDevice);
  kf_matmul<<<LB, LT>>>(N, dA, dB, dC); cudaDeviceSynchronize();
  double tg = tmin([&]{ kf_matmul<<<LB, LT>>>(N, dA, dB, dC); });
  cudaMemcpy(Cg, dC, MA, cudaMemcpyDeviceToHost);
  double mx = 0; int lim = N <= 128 ? N : 128;
  for (int i = 0; i < lim; ++i)
    for (int j = 0; j < lim; ++j) {
      double ref = 0;
      for (int k = 0; k < N; ++k) ref += A[(size_t) i * N + k] * B[(size_t) k * N + j];
      double e = ref != 0 ? fabs((Cg[(size_t) i * N + j] - ref) / ref)
                          : fabs(Cg[(size_t) i * N + j]);
      if (e > mx) mx = e;
    }
  emit("matmul", "freal", "freal", PREC, N, tg, mx,
       2.0 * (double) N * (double) N * (double) N);
  cudaFree(dA); cudaFree(dB); cudaFree(dC); free(A); free(B); free(Cg);
}

int main(int argc, char **argv)
{
  const char *out = nullptr; bool append = false;
  for (int i = 1; i < argc; ++i) {
    if      (!strcmp(argv[i], "--reps")    && i + 1 < argc) g_reps = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--blocks")  && i + 1 < argc) LB = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--threads") && i + 1 < argc) LT = atoi(argv[++i]);
    else if (!strcmp(argv[i], "--out")     && i + 1 < argc) out = argv[++i];
    else if (!strcmp(argv[i], "--append")) append = true;
    else { fprintf(stderr, "usage: %s [--reps N] [--blocks N] [--threads N] "
                           "[--out FILE] [--append]\n", argv[0]); return 1; }
  }
  g_out = out ? fopen(out, append ? "a" : "w") : stdout;
  if (!g_out) { fprintf(stderr, "cannot open %s\n", out ? out : "-"); return 1; }
  if (!append && out)
    fprintf(g_out, "op,kind,type,bits,N,gpu_sec,gpu_mflops,relerr\n");
  fprintf(stderr, "# freal_gpu_bench PREC=%d reps=%d grid=%dx%d\n", PREC, g_reps, LB, LT);

  const int axpyN[] = {4096, 16384, 65536};
  const int mvN[]   = {128, 256, 512};
  const int mmN[]   = {64, 128, 256};
  for (int i = 0; i < 3; ++i) { freal_axpy(axpyN[i]); fprintf(stderr, "  freal axpy N=%d done\n", axpyN[i]); }
  for (int i = 0; i < 3; ++i) { freal_matvec(mvN[i]); fprintf(stderr, "  freal matvec N=%d done\n", mvN[i]); }
  for (int i = 0; i < 3; ++i) { freal_matmul(mmN[i]); fprintf(stderr, "  freal matmul N=%d done\n", mmN[i]); }
  if (out) fclose(g_out);
  return 0;
}
