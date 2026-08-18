/* cmpflinear_cu.cu -- GPU (CUDA) multiple-precision complex linear computation
 * for BNCmatmul.  Complex matrix multiply, matrix-vector multiply and LU-based
 * linear-system solve, every element kept at a fixed compile-time precision PREC
 * bits on the device.  Mirrors the CPU cmpflinear.c (mul_cmpfmatrix etc.) and the
 * real GPU mpflinear_cu.cu.
 *
 * Numerical core: cu_fcomplex<PREC> (= a pair of register-resident cu_freal<PREC>)
 * from mpc_cuda (include/mpc_cuda/cu_fcomplex.cuh).  Complex products use the
 * fused, correctly rounded cu_fmms/cu_fmma helpers.  As with the real version the
 * value semantics let an n x n complex matrix live on the device at full
 * precision so Gaussian elimination updates it in place without double round-off.
 *
 * Complex division (needed only by LU) is built here from the real Newton
 * reciprocal: 1/b = conj(b) / |b|^2, so a/b = a*conj(b) * (1/|b|^2).  Partial
 * pivoting compares |.|^2 (= re^2 + im^2), avoiding any square root.
 *
 * Complex data crosses the C ABI as a pair of row-major double arrays (real,
 * imag).  Header-only w.r.t. mpc_cuda: compile with -I<mpc_cuda>/include, links
 * WITHOUT libmpc_cuda.a, device-links alongside the g[dtq][ds]linear_cu objects.
 *
 * Build (standalone self-test):
 *   nvcc -arch=sm_121 -O3 -fmad=false -I<mpc_cuda>/include -DCMPFLINEAR_CU_TEST \
 *        src/cmpflinear_cu.cu -lmpfr -lgmp -o cmpflinear_cu_test
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "mpc_cuda/cu_fcomplex.cuh"
using namespace cu_fp;

#ifndef PREC
#define PREC 1024            /* GPU working precision in bits (multiple of 32) */
#endif

typedef cu_freal<PREC>    F;
typedef cu_fcomplex<PREC> C;

/* ----------------------------------------------------------------------------
 * Device arithmetic helpers
 * ------------------------------------------------------------------------- */
__host__ __device__ static inline int cu_recip_iters()
{
  int iters = 1;
  for (int p = 53; p < PREC; p <<= 1) iters++;
  return iters + 1;
}
__host__ __device__ static F cu_frecip(const F &b)            /* 1/b (real) */
{
  F two = F::from_double(2.0);
  F x   = F::from_double(1.0 / b.to_double());
  int iters = cu_recip_iters();
  for (int it = 0; it < iters; ++it)
    x = cu_fmul<PREC>(x, cu_fsub<PREC>(two, cu_fmul<PREC>(b, x)));
  return x;
}

/* |z|^2 = re^2 + im^2  (real, correctly rounded fused form) */
__host__ __device__ static inline F cu_cabs2(const C &z)
{
  return cu_fmma<PREC>(z.re, z.re, z.im, z.im);
}

/* a / b  (complex):  a * conj(b) / |b|^2 */
__host__ __device__ static C cu_cdiv(const C &a, const C &b)
{
  F rden  = cu_frecip(cu_cabs2(b));
  F num_r = cu_fmma<PREC>(a.re, b.re, a.im, b.im);   /*  ar*br + ai*bi */
  F num_i = cu_fmms<PREC>(a.im, b.re, a.re, b.im);   /*  ai*br - ar*bi */
  return C(cu_fmul<PREC>(num_r, rden), cu_fmul<PREC>(num_i, rden));
}

/* ----------------------------------------------------------------------------
 * double <-> cu_fcomplex conversion kernels
 * ------------------------------------------------------------------------- */
__global__ static void cmpf_d2c_kernel(long total, const double *re, const double *im, C *c)
{
  for (long e = blockIdx.x * blockDim.x + threadIdx.x; e < total;
       e += (long) gridDim.x * blockDim.x)
    c[e] = C::from_doubles(re[e], im[e]);
}
__global__ static void cmpf_c2d_kernel(long total, const C *c, double *re, double *im)
{
  for (long e = blockIdx.x * blockDim.x + threadIdx.x; e < total;
       e += (long) gridDim.x * blockDim.x)
    { re[e] = c[e].real_d(); im[e] = c[e].imag_d(); }
}

/* ----------------------------------------------------------------------------
 * Complex matrix multiply
 * ------------------------------------------------------------------------- */
__global__ static void cmpf_matmul_kernel(int n, const double *Ar, const double *Ai,
                                          const double *Br, const double *Bi,
                                          double *Cr, double *Ci)
{
  long stride = (long) gridDim.x * blockDim.x;
  long total  = (long) n * n;
  for (long e = blockIdx.x * blockDim.x + threadIdx.x; e < total; e += stride)
    {
      int i = (int) (e / n), j = (int) (e % n);
      C acc(0.0, 0.0);
      for (int k = 0; k < n; ++k)
        {
          C a = C::from_doubles(Ar[(size_t) i*n+k], Ai[(size_t) i*n+k]);
          C bb = C::from_doubles(Br[(size_t) k*n+j], Bi[(size_t) k*n+j]);
          acc = cu_cadd<PREC>(acc, cu_cmul<PREC>(a, bb));
        }
      Cr[e] = acc.real_d(); Ci[e] = acc.imag_d();
    }
}

/* ----------------------------------------------------------------------------
 * Complex matrix-vector multiply
 * ------------------------------------------------------------------------- */
__global__ static void cmpf_matvec_kernel(int n, const double *Ar, const double *Ai,
                                          const double *xr, const double *xi,
                                          double *yr, double *yi)
{
  long stride = (long) gridDim.x * blockDim.x;
  for (long i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    {
      C acc(0.0, 0.0);
      for (int k = 0; k < n; ++k)
        {
          C a = C::from_doubles(Ar[(size_t) i*n+k], Ai[(size_t) i*n+k]);
          C xx = C::from_doubles(xr[k], xi[k]);
          acc = cu_cadd<PREC>(acc, cu_cmul<PREC>(a, xx));
        }
      yr[i] = acc.real_d(); yi[i] = acc.imag_d();
    }
}

/* ----------------------------------------------------------------------------
 * Complex LU solve (single cooperative block), partial pivoting on |.|^2.
 * ------------------------------------------------------------------------- */
__global__ static void cmpf_lu_solve_kernel(int n, C *A, C *b, C *x, int *status)
{
  int t = threadIdx.x, nt = blockDim.x;
  __shared__ int piv;
  if (t == 0) status[0] = 0;
  __syncthreads();

  for (int k = 0; k < n; ++k)
    {
      if (t == 0)
        {
          int p = k; F mx = cu_cabs2(A[(size_t) k*n+k]);
          for (int i = k + 1; i < n; ++i)
            { F v = cu_cabs2(A[(size_t) i*n+k]);
              if (cu_cmp<PREC>(v, mx) > 0) { mx = v; p = i; } }
          piv = p;
          if (A[(size_t) p*n+k].re.is_zero() && A[(size_t) p*n+k].im.is_zero())
            status[0] = k + 1;
        }
      __syncthreads();
      int p = piv;
      if (p != k)
        {
          for (int j = t; j < n; j += nt)
            { C tmp = A[(size_t) k*n+j];
              A[(size_t) k*n+j] = A[(size_t) p*n+j];
              A[(size_t) p*n+j] = tmp; }
          if (t == 0) { C tmp = b[k]; b[k] = b[p]; b[p] = tmp; }
        }
      __syncthreads();

      C pivval = A[(size_t) k*n+k];
      for (int i = k + 1 + t; i < n; i += nt)
        {
          C mik = cu_cdiv(A[(size_t) i*n+k], pivval);
          A[(size_t) i*n+k] = mik;
          for (int j = k + 1; j < n; ++j)
            A[(size_t) i*n+j] =
              cu_csub<PREC>(A[(size_t) i*n+j],
                            cu_cmul<PREC>(mik, A[(size_t) k*n+j]));
          b[i] = cu_csub<PREC>(b[i], cu_cmul<PREC>(mik, b[k]));
        }
      __syncthreads();
    }

  if (t == 0)
    for (int i = n - 1; i >= 0; --i)
      {
        C s = b[i];
        for (int j = i + 1; j < n; ++j)
          s = cu_csub<PREC>(s, cu_cmul<PREC>(A[(size_t) i*n+j], x[j]));
        x[i] = cu_cdiv(s, A[(size_t) i*n+i]);
      }
}

/* ============================================================================
 * Host entry points (C ABI)
 * ========================================================================= */
extern "C" int
cuda_mul_cmpfmatrix(double *Cr, double *Ci,
                    const double *Ar, const double *Ai,
                    const double *Br, const double *Bi,
                    int n, int lblocks, int lthreads)
{
  if (lblocks  <= 0) lblocks  = 256;
  if (lthreads <= 0) lthreads = 32;
  size_t M = (size_t) n * n * sizeof(double);
  double *dAr,*dAi,*dBr,*dBi,*dCr,*dCi;
  cudaMalloc(&dAr,M); cudaMalloc(&dAi,M); cudaMalloc(&dBr,M);
  cudaMalloc(&dBi,M); cudaMalloc(&dCr,M); cudaMalloc(&dCi,M);
  cudaMemcpy(dAr,Ar,M,cudaMemcpyHostToDevice); cudaMemcpy(dAi,Ai,M,cudaMemcpyHostToDevice);
  cudaMemcpy(dBr,Br,M,cudaMemcpyHostToDevice); cudaMemcpy(dBi,Bi,M,cudaMemcpyHostToDevice);
  cmpf_matmul_kernel<<<lblocks,lthreads>>>(n,dAr,dAi,dBr,dBi,dCr,dCi);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  cudaMemcpy(Cr,dCr,M,cudaMemcpyDeviceToHost); cudaMemcpy(Ci,dCi,M,cudaMemcpyDeviceToHost);
  cudaFree(dAr);cudaFree(dAi);cudaFree(dBr);cudaFree(dBi);cudaFree(dCr);cudaFree(dCi);
  return err ? (int) err : 0;
}

extern "C" int
cuda_mul_cmpfmatrix_vec(double *yr, double *yi,
                        const double *Ar, const double *Ai,
                        const double *xr, const double *xi,
                        int n, int lblocks, int lthreads)
{
  if (lblocks  <= 0) lblocks  = 256;
  if (lthreads <= 0) lthreads = 32;
  size_t MA = (size_t) n*n*sizeof(double), MV = (size_t) n*sizeof(double);
  double *dAr,*dAi,*dxr,*dxi,*dyr,*dyi;
  cudaMalloc(&dAr,MA); cudaMalloc(&dAi,MA);
  cudaMalloc(&dxr,MV); cudaMalloc(&dxi,MV); cudaMalloc(&dyr,MV); cudaMalloc(&dyi,MV);
  cudaMemcpy(dAr,Ar,MA,cudaMemcpyHostToDevice); cudaMemcpy(dAi,Ai,MA,cudaMemcpyHostToDevice);
  cudaMemcpy(dxr,xr,MV,cudaMemcpyHostToDevice); cudaMemcpy(dxi,xi,MV,cudaMemcpyHostToDevice);
  cmpf_matvec_kernel<<<lblocks,lthreads>>>(n,dAr,dAi,dxr,dxi,dyr,dyi);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  cudaMemcpy(yr,dyr,MV,cudaMemcpyDeviceToHost); cudaMemcpy(yi,dyi,MV,cudaMemcpyDeviceToHost);
  cudaFree(dAr);cudaFree(dAi);cudaFree(dxr);cudaFree(dxi);cudaFree(dyr);cudaFree(dyi);
  return err ? (int) err : 0;
}

extern "C" int
cuda_cmpf_lu_solve(double *xr, double *xi,
                   const double *Ar, const double *Ai,
                   const double *br, const double *bi,
                   int n, int lblocks, int lthreads)
{
  (void) lblocks;
  if (lthreads <= 0) lthreads = 128;
  cudaDeviceSetLimit(cudaLimitStackSize, (size_t) 32 * 1024);

  size_t MA = (size_t) n*n*sizeof(double), MV = (size_t) n*sizeof(double);
  double *dAr,*dAi,*dbr,*dbi; C *dA,*db,*dx; int *dstat;
  cudaMalloc(&dAr,MA); cudaMalloc(&dAi,MA); cudaMalloc(&dbr,MV); cudaMalloc(&dbi,MV);
  cudaMalloc(&dA,(size_t) n*n*sizeof(C));
  cudaMalloc(&db,(size_t) n*sizeof(C));
  cudaMalloc(&dx,(size_t) n*sizeof(C));
  cudaMalloc(&dstat,sizeof(int));
  cudaMemcpy(dAr,Ar,MA,cudaMemcpyHostToDevice); cudaMemcpy(dAi,Ai,MA,cudaMemcpyHostToDevice);
  cudaMemcpy(dbr,br,MV,cudaMemcpyHostToDevice); cudaMemcpy(dbi,bi,MV,cudaMemcpyHostToDevice);

  int blk = (int) (((long) n*n + lthreads - 1) / lthreads);
  if (blk < 1) blk = 1;
  cmpf_d2c_kernel<<<blk,lthreads>>>((long) n*n, dAr, dAi, dA);
  cmpf_d2c_kernel<<<1,lthreads>>>((long) n, dbr, dbi, db);
  cmpf_lu_solve_kernel<<<1,lthreads>>>(n, dA, db, dx, dstat);
  cmpf_c2d_kernel<<<1,lthreads>>>((long) n, dx, dbr, dbi);
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  int stat = 0;
  cudaMemcpy(&stat, dstat, sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(xr, dbr, MV, cudaMemcpyDeviceToHost);
  cudaMemcpy(xi, dbi, MV, cudaMemcpyDeviceToHost);

  cudaFree(dAr);cudaFree(dAi);cudaFree(dbr);cudaFree(dbi);
  cudaFree(dA);cudaFree(db);cudaFree(dx);cudaFree(dstat);
  return err ? (int) err : stat;
}

/* ============================================================================
 * Optional self-test: GPU vs CPU MPC.  Build with -DCMPFLINEAR_CU_TEST.
 * ========================================================================= */
#ifdef CMPFLINEAR_CU_TEST
#include <gmp.h>
#include <mpfr.h>
#include <mpc.h>

static void cpu_clu_solve(int n, const double *Ar, const double *Ai,
                          const double *br, const double *bi, mpc_t *xout)
{
  mpc_t *A = (mpc_t *) malloc((size_t) n*n*sizeof(mpc_t));
  mpc_t *b = (mpc_t *) malloc((size_t) n*sizeof(mpc_t));
  for (int i = 0; i < n*n; i++) { mpc_init2(A[i], PREC); mpc_set_d_d(A[i], Ar[i], Ai[i], MPC_RNDNN); }
  for (int i = 0; i < n; i++)   { mpc_init2(b[i], PREC); mpc_set_d_d(b[i], br[i], bi[i], MPC_RNDNN); }
  mpc_t mik, t; mpc_init2(mik, PREC); mpc_init2(t, PREC);
  mpfr_t m1, m2; mpfr_init2(m1, PREC); mpfr_init2(m2, PREC);
  for (int k = 0; k < n; k++) {
    int p = k; mpc_abs(m1, A[k*n+k], MPFR_RNDN);
    for (int i = k + 1; i < n; i++) { mpc_abs(m2, A[i*n+k], MPFR_RNDN); if (mpfr_cmp(m2, m1) > 0) { mpfr_set(m1, m2, MPFR_RNDN); p = i; } }
    if (p != k) { for (int j = 0; j < n; j++) mpc_swap(A[k*n+j], A[p*n+j]); mpc_swap(b[k], b[p]); }
    for (int i = k + 1; i < n; i++) {
      mpc_div(mik, A[i*n+k], A[k*n+k], MPC_RNDNN); mpc_set(A[i*n+k], mik, MPC_RNDNN);
      for (int j = k + 1; j < n; j++) { mpc_mul(t, mik, A[k*n+j], MPC_RNDNN); mpc_sub(A[i*n+j], A[i*n+j], t, MPC_RNDNN); }
      mpc_mul(t, mik, b[k], MPC_RNDNN); mpc_sub(b[i], b[i], t, MPC_RNDNN);
    }
  }
  for (int i = n - 1; i >= 0; i--) {
    mpc_set(xout[i], b[i], MPC_RNDNN);
    for (int j = i + 1; j < n; j++) { mpc_mul(t, A[i*n+j], xout[j], MPC_RNDNN); mpc_sub(xout[i], xout[i], t, MPC_RNDNN); }
    mpc_div(xout[i], xout[i], A[i*n+i], MPC_RNDNN);
  }
  for (int i = 0; i < n*n; i++) mpc_clear(A[i]);
  for (int i = 0; i < n; i++)   mpc_clear(b[i]);
  mpc_clear(mik); mpc_clear(t); mpfr_clear(m1); mpfr_clear(m2); free(A); free(b);
}

int main(int argc, char **argv)
{
  int n = argc > 1 ? atoi(argv[1]) : 64;
  srand(20240618);
  size_t NN = (size_t) n*n;
  double *Ar=(double*)malloc(NN*sizeof(double)), *Ai=(double*)malloc(NN*sizeof(double));
  double *Br=(double*)malloc(NN*sizeof(double)), *Bi=(double*)malloc(NN*sizeof(double));
  double *br=(double*)malloc(n*sizeof(double)),  *bi=(double*)malloc(n*sizeof(double));
  for (int i = 0; i < n; i++) {
    double rs = 0;
    for (int j = 0; j < n; j++) {
      Ar[i*n+j]=(rand()%2000-1000)/100.0; Ai[i*n+j]=(rand()%2000-1000)/100.0;
      Br[i*n+j]=(rand()%2000-1000)/100.0; Bi[i*n+j]=(rand()%2000-1000)/100.0;
      rs += fabs(Ar[i*n+j]) + fabs(Ai[i*n+j]);
    }
    Ar[i*n+i] = (Ar[i*n+i] >= 0 ? 1 : -1) * (rs + 1.0); Ai[i*n+i] *= 0.1;  /* diag dominant */
    br[i]=(rand()%2000-1000)/100.0; bi[i]=(rand()%2000-1000)/100.0;
  }

  mpc_t acc, ta, tb; mpc_init2(acc, PREC); mpc_init2(ta, PREC); mpc_init2(tb, PREC);

  /* ---- matmul ---- */
  double *Cgr=(double*)malloc(NN*sizeof(double)), *Cgi=(double*)malloc(NN*sizeof(double));
  cuda_mul_cmpfmatrix(Cgr,Cgi,Ar,Ai,Br,Bi,n,256,32);
  double mmrel = 0;
  for (int i = 0; i < n; i++) for (int j = 0; j < n; j++) {
    mpc_set_ui(acc, 0, MPC_RNDNN);
    for (int k = 0; k < n; k++) { mpc_set_d_d(ta, Ar[i*n+k], Ai[i*n+k], MPC_RNDNN);
      mpc_set_d_d(tb, Br[k*n+j], Bi[k*n+j], MPC_RNDNN); mpc_mul(ta, ta, tb, MPC_RNDNN); mpc_add(acc, acc, ta, MPC_RNDNN); }
    double cr = mpfr_get_d(mpc_realref(acc), MPFR_RNDN), ci = mpfr_get_d(mpc_imagref(acc), MPFR_RNDN);
    double er = hypot(Cgr[i*n+j]-cr, Cgi[i*n+j]-ci), mag = hypot(cr, ci);
    double r = mag > 1e-300 ? er/mag : er; if (r > mmrel) mmrel = r;
  }
  printf("c-matmul  n=%d PREC=%d  max rel err = %.3e\n", n, PREC, mmrel);

  /* ---- matvec ---- */
  double *ygr=(double*)malloc(n*sizeof(double)), *ygi=(double*)malloc(n*sizeof(double));
  cuda_mul_cmpfmatrix_vec(ygr,ygi,Ar,Ai,br,bi,n,256,32);
  double mvrel = 0;
  for (int i = 0; i < n; i++) {
    mpc_set_ui(acc, 0, MPC_RNDNN);
    for (int k = 0; k < n; k++) { mpc_set_d_d(ta, Ar[i*n+k], Ai[i*n+k], MPC_RNDNN);
      mpc_set_d_d(tb, br[k], bi[k], MPC_RNDNN); mpc_mul(ta, ta, tb, MPC_RNDNN); mpc_add(acc, acc, ta, MPC_RNDNN); }
    double cr = mpfr_get_d(mpc_realref(acc), MPFR_RNDN), ci = mpfr_get_d(mpc_imagref(acc), MPFR_RNDN);
    double er = hypot(ygr[i]-cr, ygi[i]-ci), mag = hypot(cr, ci);
    double r = mag > 1e-300 ? er/mag : er; if (r > mvrel) mvrel = r;
  }
  printf("c-matvec  n=%d PREC=%d  max rel err = %.3e\n", n, PREC, mvrel);

  /* ---- LU solve ---- */
  double *xgr=(double*)malloc(n*sizeof(double)), *xgi=(double*)malloc(n*sizeof(double));
  int st = cuda_cmpf_lu_solve(xgr,xgi,Ar,Ai,br,bi,n,0,128);
  mpc_t *xc = (mpc_t *) malloc(n*sizeof(mpc_t));
  for (int i = 0; i < n; i++) mpc_init2(xc[i], PREC);
  cpu_clu_solve(n, Ar, Ai, br, bi, xc);
  double lurel = 0;
  for (int i = 0; i < n; i++) {
    double cr = mpfr_get_d(mpc_realref(xc[i]), MPFR_RNDN), ci = mpfr_get_d(mpc_imagref(xc[i]), MPFR_RNDN);
    double er = hypot(xgr[i]-cr, xgi[i]-ci), mag = hypot(cr, ci);
    double r = mag > 1e-300 ? er/mag : er; if (r > lurel) lurel = r;
  }
  printf("c-LUsolve n=%d PREC=%d  max rel err = %.3e (status=%d)\n", n, PREC, lurel, st);

  return (mmrel < 1e-290 && mvrel < 1e-290 && lurel < 1e-12) ? 0 : 1;
}
#endif /* CMPFLINEAR_CU_TEST */
