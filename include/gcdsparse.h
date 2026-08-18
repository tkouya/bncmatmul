/********************************************************************************/
/* gcdsparse.h: native complex double SPARSE (CSR) SpMV on GPU (CUDA)            */
/*   complex CSR with separate re/im double value arrays (SoA).                   */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_GCDSPARSE_H__
#define __BNC_GCDSPARSE_H__

#include "cuda_runtime.h"

typedef struct
{
	long int row_dim, col_dim, nnz;
	long int *row_ptr;        // length row_dim + 1
	long int *col_idx;        // length nnz
	double   *val_re, *val_im; // length nnz each
} gcdspmatrix;
typedef gcdspmatrix *GCDSPMatrix;

GCDSPMatrix init_gcdspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                 const long int *row_ptr, const long int *col_idx,
                                 const double *val_re, const double *val_im);
void free_gcdspmatrix_dev(GCDSPMatrix mat);

// (yre, yim) := A * (xre, xim)  (raw device arrays)
void mul_gcdspmatrix(double *yre, double *yim, GCDSPMatrix a, const double *xre, const double *xim, int nbg, int ntb);

#endif // __BNC_GCDSPARSE_H__
