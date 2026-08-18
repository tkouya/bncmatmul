/********************************************************************************/
/* gfsparse.h: Native float precision SPARSE (CSR) SpMV on GPU (CUDA)           */
/*   Standard CSR (row_ptr/col_idx/val); format-agnostic (no bncsparse dep).      */
/*   Reuses GFVector (gflinear.h) for the dense in/out vectors.                   */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_GFSPARSE_H__
#define __BNC_GFSPARSE_H__

#include "cuda_runtime.h"
#include "gflinear.h"   // GFVector

// CSR sparse matrix on the GPU (device pointers)
typedef struct
{
	long int row_dim, col_dim, nnz;
	long int *row_ptr;   // length row_dim + 1
	long int *col_idx;   // length nnz
	float   *val;       // length nnz
} gfspmatrix;
typedef gfspmatrix *GFSPMatrix;

// build a device CSR matrix from host CSR arrays (copied to GPU)
GFSPMatrix init_gfspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                               const long int *row_ptr, const long int *col_idx, const float *val);
void free_gfspmatrix_dev(GFSPMatrix mat);

// ret := A * x  (sparse matrix-vector product)
void mul_gfspmatrix_gfvec(GFVector ret, GFSPMatrix a, GFVector x, int nbg, int ntb);
// ret := A^T * x  (uses atomic scatter; ret zeroed internally)
void mul_gfspmatrixt_gfvec(GFVector ret, GFSPMatrix a, GFVector x, int nbg, int ntb);

#endif // __BNC_GFSPARSE_H__
