/********************************************************************************/
/* gcflinear.h: Native float-precision COMPLEX GPU Linear Computation (CUDA)    */
/*   complex stored as separate real/imag float arrays (SoA), like gcmpf.       */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_GCFLINEAR_H__
#define __BNC_GCFLINEAR_H__

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

// complex float GPU vector (SoA: re[], im[])
typedef struct
{
	long int dim;
	float *re, *im;
} gcfvector;
typedef gcfvector *GCFVector;

// complex float GPU matrix (row-major, stride = col_dim; SoA re[], im[])
typedef struct
{
	long int row_dim, col_dim;
	float *re, *im;
} gcfmatrix;
typedef gcfmatrix *GCFMatrix;

/*------------------------- GCFVector -------------------------*/
GCFVector init_gcfvector(long int dim);
void free_gcfvector(GCFVector vec);
GCFVector init_gcfvector_dev(long int dim);
void free_gcfvector_dev(GCFVector vec);
void subst_gcfvector_dev_cdvec(GCFVector gcfvec_dev, CDVector cdvec); // host->dev
void subst_cdvector_gcfvec_dev(CDVector cdvec, GCFVector gcfvec_dev); // dev->host
void print_gcfvector_dev(GCFVector dev_vec);

void add_gcfvector_dev(GCFVector c, GCFVector a, GCFVector b, int nbg, int ntb);
void sub_gcfvector_dev(GCFVector c, GCFVector a, GCFVector b, int nbg, int ntb);
void cmul_gcfvector_dev(GCFVector c, float val_re, float val_im, GCFVector a, int nbg, int ntb);
void subst_gcfvector_dev(GCFVector ret, GCFVector vec, int nbg, int ntb);
void set0_gcfvector_dev(GCFVector ret, int nbg, int ntb);
// inner product (a, b) = sum conj(a_i) * b_i  (Hermitian); result -> ret_re/ret_im (device)
void ip_gcfvector_dev(float *ret_re, float *ret_im, GCFVector a, GCFVector b, int nbg, int ntb);
// 2-norm (real scalar) -> ret_dev (device float)
void norm2_gcfvector_dev(float *ret_dev, GCFVector a, int nbg, int ntb);

/*------------------------- GCFMatrix -------------------------*/
GCFMatrix init_gcfmatrix(long int row_dim, long int col_dim);
void free_gcfmatrix(GCFMatrix mat);
GCFMatrix init_gcfmatrix_dev(long int row_dim, long int col_dim);
void free_gcfmatrix_dev(GCFMatrix mat);
void subst_gcfmatrix_dev_cdmat(GCFMatrix gcfmat_dev, CDMatrix cdmat); // host->dev
void subst_cdmatrix_gcfmat_dev(CDMatrix cdmat, GCFMatrix gcfmat_dev); // dev->host
void print_gcfmatrix_dev(GCFMatrix mat);

// matrix multiplication ret := A * B (complex)
void mul_gcfmatrix_dev(GCFMatrix ret, GCFMatrix a, GCFMatrix b, int nbg, int ntb);
// Frobenius norm (real scalar) -> ret_dev (device float)
void normf_gcfmatrix_dev(float *ret_dev, GCFMatrix mat, int nbg, int ntb);

void add_gcfmatrix_dev(GCFMatrix c, GCFMatrix a, GCFMatrix b, int nbg, int ntb);
void sub_gcfmatrix_dev(GCFMatrix c, GCFMatrix a, GCFMatrix b, int nbg, int ntb);
void cmul_gcfmatrix_dev(GCFMatrix c, float sc_re, float sc_im, GCFMatrix a, int nbg, int ntb);
void transpose_gcfmatrix_dev(GCFMatrix c, GCFMatrix a, int nbg, int ntb);      // plain transpose
void conjtrans_gcfmatrix_dev(GCFMatrix c, GCFMatrix a, int nbg, int ntb);      // conjugate transpose
void subst_gcfmatrix_dev(GCFMatrix c, GCFMatrix a, int nbg, int ntb);
void set0_gcfmatrix_dev(GCFMatrix c, int nbg, int ntb);
void setI_gcfmatrix_dev(GCFMatrix c, int nbg, int ntb);
void set_gcfmatrix_ij_dev(GCFMatrix mat, long int i, long int j, float val_re, float val_im);

// v := a * vb  /  v := a^T * vb  (complex)
void mul_gcfmatrix_gcfvec(GCFVector v, GCFMatrix a, GCFVector vb, int nbg, int ntb);
void mul_gcfmatrixt_gcfvec(GCFVector v, GCFMatrix a, GCFVector vb, int nbg, int ntb);

#endif // __BNC_GCFLINEAR_H__
