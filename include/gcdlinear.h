/********************************************************************************/
/* gcdlinear.h: Native double-precision COMPLEX GPU Linear Computation (CUDA)    */
/*   complex stored as separate real/imag double arrays (SoA), like gcmpf.       */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_GCDLINEAR_H__
#define __BNC_GCDLINEAR_H__

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <complex.h>   // double _Complex (CPU CDVector / CDMatrix)

// on CPU
#include "cdlinear.h"  // CDVector / CDMatrix

// on GPU
#include "cuda_runtime.h"

#ifndef MAX_NUM_THREADS_PER_BLOCK
  #define MAX_NUM_THREADS_PER_BLOCK 128
#endif
#ifndef MAX_NUM_BLOCKS_PER_GRID
  #define MAX_NUM_BLOCKS_PER_GRID 128
#endif

// complex double GPU vector (SoA: re[], im[])
typedef struct
{
	long int dim;
	double *re, *im;
} gcdvector;
typedef gcdvector *GCDVector;

// complex double GPU matrix (row-major, stride = col_dim; SoA re[], im[])
typedef struct
{
	long int row_dim, col_dim;
	double *re, *im;
} gcdmatrix;
typedef gcdmatrix *GCDMatrix;

/*------------------------- GCDVector -------------------------*/
GCDVector init_gcdvector(long int dim);
void free_gcdvector(GCDVector vec);
GCDVector init_gcdvector_dev(long int dim);
void free_gcdvector_dev(GCDVector vec);
void subst_gcdvector_dev_cdvec(GCDVector gcdvec_dev, CDVector cdvec); // host->dev
void subst_cdvector_gcdvec_dev(CDVector cdvec, GCDVector gcdvec_dev); // dev->host
void print_gcdvector_dev(GCDVector dev_vec);

void add_gcdvector_dev(GCDVector c, GCDVector a, GCDVector b, int nbg, int ntb);
void sub_gcdvector_dev(GCDVector c, GCDVector a, GCDVector b, int nbg, int ntb);
void cmul_gcdvector_dev(GCDVector c, double val_re, double val_im, GCDVector a, int nbg, int ntb);
void subst_gcdvector_dev(GCDVector ret, GCDVector vec, int nbg, int ntb);
void set0_gcdvector_dev(GCDVector ret, int nbg, int ntb);
// inner product (a, b) = sum conj(a_i) * b_i  (Hermitian); result -> ret_re/ret_im (device)
void ip_gcdvector_dev(double *ret_re, double *ret_im, GCDVector a, GCDVector b, int nbg, int ntb);
// 2-norm (real scalar) -> ret_dev (device double)
void norm2_gcdvector_dev(double *ret_dev, GCDVector a, int nbg, int ntb);

/*------------------------- GCDMatrix -------------------------*/
GCDMatrix init_gcdmatrix(long int row_dim, long int col_dim);
void free_gcdmatrix(GCDMatrix mat);
GCDMatrix init_gcdmatrix_dev(long int row_dim, long int col_dim);
void free_gcdmatrix_dev(GCDMatrix mat);
void subst_gcdmatrix_dev_cdmat(GCDMatrix gcdmat_dev, CDMatrix cdmat); // host->dev
void subst_cdmatrix_gcdmat_dev(CDMatrix cdmat, GCDMatrix gcdmat_dev); // dev->host
void print_gcdmatrix_dev(GCDMatrix mat);

// matrix multiplication ret := A * B (complex)
void mul_gcdmatrix_dev(GCDMatrix ret, GCDMatrix a, GCDMatrix b, int nbg, int ntb);
// Frobenius norm (real scalar) -> ret_dev (device double)
void normf_gcdmatrix_dev(double *ret_dev, GCDMatrix mat, int nbg, int ntb);

void add_gcdmatrix_dev(GCDMatrix c, GCDMatrix a, GCDMatrix b, int nbg, int ntb);
void sub_gcdmatrix_dev(GCDMatrix c, GCDMatrix a, GCDMatrix b, int nbg, int ntb);
void cmul_gcdmatrix_dev(GCDMatrix c, double sc_re, double sc_im, GCDMatrix a, int nbg, int ntb);
void transpose_gcdmatrix_dev(GCDMatrix c, GCDMatrix a, int nbg, int ntb);      // plain transpose
void conjtrans_gcdmatrix_dev(GCDMatrix c, GCDMatrix a, int nbg, int ntb);      // conjugate transpose
void subst_gcdmatrix_dev(GCDMatrix c, GCDMatrix a, int nbg, int ntb);
void set0_gcdmatrix_dev(GCDMatrix c, int nbg, int ntb);
void setI_gcdmatrix_dev(GCDMatrix c, int nbg, int ntb);
void set_gcdmatrix_ij_dev(GCDMatrix mat, long int i, long int j, double val_re, double val_im);

// v := a * vb  /  v := a^T * vb  (complex)
void mul_gcdmatrix_gcdvec(GCDVector v, GCDMatrix a, GCDVector vb, int nbg, int ntb);
void mul_gcdmatrixt_gcdvec(GCDVector v, GCDMatrix a, GCDVector vb, int nbg, int ntb);

#endif // __BNC_GCDLINEAR_H__
