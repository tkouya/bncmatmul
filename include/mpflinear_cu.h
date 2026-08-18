/********************************************************************************/
/* mpflinear_cu.h: GPU (CUDA) multiple-precision (MPFR-equivalent) real linear   */
/*                 computation for BNCmatmul -- C-callable host entry points.    */
/*                                                                              */
/* The GPU side keeps every element at a fixed compile-time precision PREC bits  */
/* using the register-resident cu_freal<PREC> type from mpc_cuda.  Each routine  */
/* takes/returns plain row-major double arrays (host pointers); the multiple-    */
/* precision accumulation happens entirely on the device.  These mirror the CPU  */
/* mul_mpfmatrix / mul_mpfmatrix_vec / LU solve, and coexist with the            */
/* g[dtq][ds]linear_cu GPU kernels inside libbncmm_cuda.a.                       */
/*                                                                              */
/* GPU working precision is fixed at build time (default 1024 bits, override     */
/* with -DPREC=<bits>, a multiple of 32).  This is independent of the runtime    */
/* mpf_t precision used by the CPU MPFMatrix routines.                           */
/********************************************************************************/
#ifndef _MPFLINEAR_CU_H
#define _MPFLINEAR_CU_H

#if defined(__cplusplus)
extern "C" {
#endif

/* y = alpha * x + y, length n (row-major double), PREC-bit accumulation.
 * lblocks/lthreads tune the launch grid (<=0 -> defaults).  Returns 0 on ok. */
int cuda_mpf_axpy(double *y_host, double alpha, const double *x_host,
                  long n, int lblocks, int lthreads);

/* C = A * B, all n x n row-major double, accumulated at PREC bits on the GPU.
 * lblocks/lthreads tune the launch grid (<=0 -> defaults).  Returns 0 on ok. */
int cuda_mul_mpfmatrix(double *C_host, const double *A_host, const double *B_host,
                       int n, int lblocks, int lthreads);

/* y = A * x, A n x n, x/y length n (row-major double), PREC accumulation.       */
int cuda_mul_mpfmatrix_vec(double *y_host, const double *A_host, const double *x_host,
                           int n, int lblocks, int lthreads);

/* Solve A x = b (A n x n, b/x length n, row-major double) by LU decomposition
 * with partial pivoting, every intermediate kept at PREC bits on the GPU.
 * lthreads sets the cooperative thread-block width (<=0 -> default).  The
 * factorization runs in a single block; lblocks is accepted for API symmetry.
 * Returns 0 on success, non-zero on a (near-)singular pivot or CUDA error.      */
int cuda_mpf_lu_solve(double *x_host, const double *A_host, const double *b_host,
                      int n, int lblocks, int lthreads);

#if defined(__cplusplus)
}
#endif

#endif /* _MPFLINEAR_CU_H */
