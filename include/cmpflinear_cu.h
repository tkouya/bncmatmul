/********************************************************************************/
/* cmpflinear_cu.h: GPU (CUDA) multiple-precision (MPC-equivalent) complex linear */
/*                  computation for BNCmatmul -- C-callable host entry points.   */
/*                                                                              */
/* Complex matrices/vectors are passed as a pair of row-major double arrays      */
/* (real part, imaginary part).  On the device each element is held at a fixed   */
/* compile-time precision PREC bits using the register-resident cu_fcomplex<PREC>*/
/* type from mpc_cuda.  These mirror the CPU mul_cmpfmatrix / mul_cmpfmatrix_vec  */
/* / LU solve, and coexist with the g[dtq][ds]linear_cu kernels in               */
/* libbncmm_cuda.a.                                                              */
/*                                                                              */
/* GPU working precision is fixed at build time (default 1024 bits, override     */
/* with -DPREC=<bits>, a multiple of 32).                                        */
/********************************************************************************/
#ifndef _CMPFLINEAR_CU_H
#define _CMPFLINEAR_CU_H

#if defined(__cplusplus)
extern "C" {
#endif

/* C = A * B, complex n x n; Cr/Ci, Ar/Ai, Br/Bi are the real/imag double
 * arrays (row-major).  PREC accumulation on the GPU.  Returns 0 on ok.          */
/* y = alpha * x + y (complex, split re/im arrays of length n), PREC-bit
 * accumulation.  lblocks/lthreads tune the grid (<=0 -> defaults). */
int cuda_cmpf_axpy(double *yr, double *yi, double ar, double ai,
                   const double *xr, const double *xi,
                   long n, int lblocks, int lthreads);

int cuda_mul_cmpfmatrix(double *Cr, double *Ci,
                        const double *Ar, const double *Ai,
                        const double *Br, const double *Bi,
                        int n, int lblocks, int lthreads);

/* y = A * x, complex; yr/yi (length n), Ar/Ai (n x n), xr/xi (length n).        */
int cuda_mul_cmpfmatrix_vec(double *yr, double *yi,
                            const double *Ar, const double *Ai,
                            const double *xr, const double *xi,
                            int n, int lblocks, int lthreads);

/* Solve A x = b, complex, by LU with partial pivoting (pivot on |.|^2), every
 * intermediate kept at PREC bits on the GPU.  xr/xi, Ar/Ai, br/bi are real/imag
 * double arrays.  Single-block factorization (lthreads = block width).
 * Returns 0 on success, non-zero on (near-)singular pivot or CUDA error.        */
int cuda_cmpf_lu_solve(double *xr, double *xi,
                       const double *Ar, const double *Ai,
                       const double *br, const double *bi,
                       int n, int lblocks, int lthreads);

#if defined(__cplusplus)
}
#endif

#endif /* _CMPFLINEAR_CU_H */
