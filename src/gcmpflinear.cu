/* gcmpflinear.cu -- CUDA multiple-precision complex (MPC) linear computation for
 * BNCmatmul.  Complex matrix multiply C = A*B with A,B,C N x N (row-major),
 * stored as separate real/imag double arrays; each complex dot product is
 * accumulated at MPC precision PREC on the GPU via the mpc_cuda device port of
 * MPC.  Mirrors the CPU mul_cmpfmatrix.  Build: same recipe as gmpflinear.cu but
 * add -Isrc/mpc_cuda (links libmpc_cuda.a).  -DGCMPFLINEAR_TEST for the self-test.
 */
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
typedef long int gmp_randstate_t[1];
#include "mpc.h"
#include "mpc_cuda/cuda_minigmp.h"
#include "mpc_cuda/cu_compat.h"

#ifndef PREC
#define PREC 1024
#endif
#ifndef GCMPF_SLAB
#define GCMPF_SLAB (16 * 1024)
#endif

#define NLIMBS ((PREC + 8 * (int) sizeof (mp_limb_t) - 1) / (8 * (int) sizeof (mp_limb_t)))
#define MPC_DECL_INIT(z)                                                 \
  mpc_t z;                                                               \
  mp_limb_t z##_rl[NLIMBS], z##_il[NLIMBS];                              \
  mpfr_custom_init_set (mpc_realref (z), MPFR_ZERO_KIND, 0, PREC, z##_rl); \
  mpfr_custom_init_set (mpc_imagref (z), MPFR_ZERO_KIND, 0, PREC, z##_il)

/* C[i,j] = sum_k A[i,k] * B[k,j]  (complex), MPC PREC accumulation */
__host__ __device__ static void
gcmpf_cdot_ij (const double *Ar, const double *Ai, const double *Br, const double *Bi,
               int n, int i, int j, double *cr, double *ci)
{
  MPC_DECL_INIT (acc); MPC_DECL_INIT (a); MPC_DECL_INIT (b); MPC_DECL_INIT (t);
  for (int k = 0; k < n; ++k)
    {
#ifdef __CUDA_ARCH__
      mpc_cuda_arena_reset ();
#endif
      size_t ia = (size_t) i * n + k, ib = (size_t) k * n + j;
      mpc_set_d_d (a, Ar[ia], Ai[ia], MPC_RNDNN);
      mpc_set_d_d (b, Br[ib], Bi[ib], MPC_RNDNN);
      mpc_mul (t, a, b, MPC_RNDNN);
      mpc_add (acc, acc, t, MPC_RNDNN);
    }
  *cr = mpfr_get_d (mpc_realref (acc), MPFR_RNDN);
  *ci = mpfr_get_d (mpc_imagref (acc), MPFR_RNDN);
}

__global__ static void
gcmpf_matmul_kernel (int n, const double *Ar, const double *Ai,
                     const double *Br, const double *Bi, double *Cr, double *Ci)
{
  int stride = gridDim.x * blockDim.x;
  long total = (long) n * n;
  for (long e = blockIdx.x * blockDim.x + threadIdx.x; e < total; e += stride)
    gcmpf_cdot_ij (Ar, Ai, Br, Bi, n, (int) (e / n), (int) (e % n), &Cr[e], &Ci[e]);
}

/* Host entry: (Cr,Ci) = (Ar,Ai) * (Br,Bi), all n x n row-major doubles. */
extern "C" int
cuda_mul_cmpfmatrix (double *Cr, double *Ci,
                     const double *Ar, const double *Ai,
                     const double *Br, const double *Bi,
                     int n, int lblocks, int lthreads)
{
  if (lblocks  <= 0) lblocks  = 256;
  if (lthreads <= 0) lthreads = 32;
  cudaDeviceSetLimit (cudaLimitMallocHeapSize, (size_t) 64 * 1024 * 1024);
  cudaDeviceSetLimit (cudaLimitStackSize,      (size_t) 160 * 1024);

  size_t M = (size_t) n * n * sizeof (double);
  double *dAr, *dAi, *dBr, *dBi, *dCr, *dCi;
  cudaMalloc (&dAr, M); cudaMalloc (&dAi, M);
  cudaMalloc (&dBr, M); cudaMalloc (&dBi, M);
  cudaMalloc (&dCr, M); cudaMalloc (&dCi, M);
  cudaMemcpy (dAr, Ar, M, cudaMemcpyHostToDevice);
  cudaMemcpy (dAi, Ai, M, cudaMemcpyHostToDevice);
  cudaMemcpy (dBr, Br, M, cudaMemcpyHostToDevice);
  cudaMemcpy (dBi, Bi, M, cudaMemcpyHostToDevice);

  size_t ntot = (size_t) lblocks * lthreads;
  char *arena; size_t *top;
  cudaMalloc (&arena, ntot * (size_t) GCMPF_SLAB);
  cudaMalloc (&top,   ntot * sizeof (size_t));
  cudaMemset (top, 0, ntot * sizeof (size_t));
  mpc_cuda_arena_base = arena; mpc_cuda_arena_slab = GCMPF_SLAB; mpc_cuda_arena_top = top;
  cudaDeviceSynchronize ();

  gcmpf_matmul_kernel<<<lblocks, lthreads>>> (n, dAr, dAi, dBr, dBi, dCr, dCi);
  cudaError_t err = cudaDeviceSynchronize ();
  if (err == cudaSuccess)
    { cudaMemcpy (Cr, dCr, M, cudaMemcpyDeviceToHost); cudaMemcpy (Ci, dCi, M, cudaMemcpyDeviceToHost); }

  cudaFree (dAr); cudaFree (dAi); cudaFree (dBr); cudaFree (dBi); cudaFree (dCr); cudaFree (dCi);
  cudaFree (arena); cudaFree (top);
  return err == cudaSuccess ? 0 : (int) err;
}

/* y_i = sum_j A[i,j] * x[j]  (complex), MPC PREC accumulation */
__host__ __device__ static void
gcmpf_cdot_row (const double *Ar, const double *Ai, const double *xr, const double *xi,
                int n, int i, double *yr, double *yi)
{
  MPC_DECL_INIT (acc); MPC_DECL_INIT (a); MPC_DECL_INIT (b); MPC_DECL_INIT (t);
  for (int j = 0; j < n; ++j)
    {
#ifdef __CUDA_ARCH__
      mpc_cuda_arena_reset ();
#endif
      size_t ia = (size_t) i * n + j;
      mpc_set_d_d (a, Ar[ia], Ai[ia], MPC_RNDNN);
      mpc_set_d_d (b, xr[j], xi[j], MPC_RNDNN);
      mpc_mul (t, a, b, MPC_RNDNN);
      mpc_add (acc, acc, t, MPC_RNDNN);
    }
  *yr = mpfr_get_d (mpc_realref (acc), MPFR_RNDN);
  *yi = mpfr_get_d (mpc_imagref (acc), MPFR_RNDN);
}

__global__ static void
gcmpf_matvec_kernel (int n, const double *Ar, const double *Ai,
                     const double *xr, const double *xi, double *yr, double *yi)
{
  int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    gcmpf_cdot_row (Ar, Ai, xr, xi, n, i, &yr[i], &yi[i]);
}

/* Host entry: (yr,yi) = (Ar,Ai) * (xr,xi) */
extern "C" int
cuda_mul_cmpfmatrix_vec (double *yr, double *yi,
                         const double *Ar, const double *Ai,
                         const double *xr, const double *xi,
                         int n, int lblocks, int lthreads)
{
  if (lblocks  <= 0) lblocks  = 256;
  if (lthreads <= 0) lthreads = 32;
  cudaDeviceSetLimit (cudaLimitMallocHeapSize, (size_t) 64 * 1024 * 1024);
  cudaDeviceSetLimit (cudaLimitStackSize,      (size_t) 160 * 1024);

  size_t M = (size_t) n * n * sizeof (double), V = (size_t) n * sizeof (double);
  double *dAr,*dAi,*dxr,*dxi,*dyr,*dyi;
  cudaMalloc(&dAr,M);cudaMalloc(&dAi,M);cudaMalloc(&dxr,V);cudaMalloc(&dxi,V);cudaMalloc(&dyr,V);cudaMalloc(&dyi,V);
  cudaMemcpy(dAr,Ar,M,cudaMemcpyHostToDevice);cudaMemcpy(dAi,Ai,M,cudaMemcpyHostToDevice);
  cudaMemcpy(dxr,xr,V,cudaMemcpyHostToDevice);cudaMemcpy(dxi,xi,V,cudaMemcpyHostToDevice);

  size_t ntot = (size_t) lblocks * lthreads;
  char *arena; size_t *top;
  cudaMalloc (&arena, ntot * (size_t) GCMPF_SLAB);
  cudaMalloc (&top,   ntot * sizeof (size_t));
  cudaMemset (top, 0, ntot * sizeof (size_t));
  mpc_cuda_arena_base = arena; mpc_cuda_arena_slab = GCMPF_SLAB; mpc_cuda_arena_top = top;
  cudaDeviceSynchronize ();

  gcmpf_matvec_kernel<<<lblocks, lthreads>>> (n, dAr, dAi, dxr, dxi, dyr, dyi);
  cudaError_t err = cudaDeviceSynchronize ();
  if (err == cudaSuccess)
    { cudaMemcpy(yr,dyr,V,cudaMemcpyDeviceToHost); cudaMemcpy(yi,dyi,V,cudaMemcpyDeviceToHost); }

  cudaFree(dAr);cudaFree(dAi);cudaFree(dxr);cudaFree(dxi);cudaFree(dyr);cudaFree(dyi);cudaFree(arena);cudaFree(top);
  return err == cudaSuccess ? 0 : (int) err;
}

#ifdef GCMPFLINEAR_TEST
int main (void)
{
  int N = 80;
  size_t M = (size_t) N * N * sizeof (double);
  double *Ar=(double*)malloc(M),*Ai=(double*)malloc(M),*Br=(double*)malloc(M),*Bi=(double*)malloc(M);
  double *Cgr=(double*)malloc(M),*Cgi=(double*)malloc(M),*Ccr=(double*)malloc(M),*Cci=(double*)malloc(M);
  for (int i=0;i<N;++i) for (int j=0;j<N;++j)
    { size_t e=(size_t)i*N+j; Ar[e]=1.0+(i+2*j)*1e-4; Ai[e]=0.3+(i+j)*1e-4; Br[e]=0.5+(2*i+j)*1e-4; Bi[e]=0.2+(i*3+j)*1e-4; }

  if (cuda_mul_cmpfmatrix (Cgr,Cgi, Ar,Ai, Br,Bi, N, 256, 32)) { fprintf(stderr,"GPU cmatmul failed\n"); return 1; }
  for (int i=0;i<N;++i) for (int j=0;j<N;++j)
    gcmpf_cdot_ij (Ar,Ai,Br,Bi, N, i,j, &Ccr[(size_t)i*N+j], &Cci[(size_t)i*N+j]);   /* CPU ref */

  double maxrel=0.0; long tot=(long)N*N;
  for (long e=0;e<tot;++e)
    { double dr=Cgr[e]-Ccr[e], di=Cgi[e]-Cci[e], mag=hypot(Ccr[e],Cci[e]);
      double r = mag!=0.0 ? hypot(dr,di)/mag : hypot(dr,di); if (r>maxrel) maxrel=r; }
  printf ("CUDA MPC matmul (N=%d, prec=%d): max relative |GPU-CPU| = %.3e  -> %s\n",
          N, PREC, maxrel, maxrel<1e-13 ? "PASS":"FAIL");

  /* matvec y = A*x (complex) */
  double *xr=(double*)malloc((size_t)N*sizeof(double)),*xi=(double*)malloc((size_t)N*sizeof(double));
  double *ygr=(double*)malloc((size_t)N*sizeof(double)),*ygi=(double*)malloc((size_t)N*sizeof(double));
  double *ycr=(double*)malloc((size_t)N*sizeof(double)),*yci=(double*)malloc((size_t)N*sizeof(double));
  for (int i=0;i<N;++i){ xr[i]=1.0+i*1e-3; xi[i]=0.2+i*5e-4; }
  cuda_mul_cmpfmatrix_vec (ygr,ygi, Ar,Ai, xr,xi, N, 256, 32);
  for (int i=0;i<N;++i) gcmpf_cdot_row (Ar,Ai, xr,xi, N, i, &ycr[i], &yci[i]);
  double mv=0.0;
  for (int i=0;i<N;++i){ double dr=ygr[i]-ycr[i],di=ygi[i]-yci[i],mag=hypot(ycr[i],yci[i]); double r=mag!=0.0?hypot(dr,di)/mag:hypot(dr,di); if(r>mv)mv=r; }
  printf ("CUDA MPC matvec (N=%d, prec=%d): max relative |GPU-CPU| = %.3e  -> %s\n",
          N, PREC, mv, mv<1e-13 ? "PASS":"FAIL");
  free(Ar);free(Ai);free(Br);free(Bi);free(Cgr);free(Cgi);free(Ccr);free(Cci);
  free(xr);free(xi);free(ygr);free(ygi);free(ycr);free(yci);
  return (maxrel<1e-13 && mv<1e-13) ? 0:1;
}
#endif
