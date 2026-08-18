/********************************************************************************/
/* gdsparse.h: Native double precision SPARSE (CSR) SpMV on GPU (CUDA)           */
/*   Standard CSR (row_ptr/col_idx/val); format-agnostic (no bncsparse dep).      */
/*   Reuses GDVector (gdlinear.h) for the dense in/out vectors.                   */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_GDSPARSE_H__
#define __BNC_GDSPARSE_H__

#include "cuda_runtime.h"
#include "gdlinear.h"   // GDVector

// CSR sparse matrix on the GPU (device pointers)
typedef struct
{
	long int row_dim, col_dim, nnz;
	long int *row_ptr;   // length row_dim + 1
	long int *col_idx;   // length nnz
	double   *val;       // length nnz
} gdspmatrix;
typedef gdspmatrix *GDSPMatrix;

// build a device CSR matrix from host CSR arrays (copied to GPU)
GDSPMatrix init_gdspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                               const long int *row_ptr, const long int *col_idx, const double *val);
void free_gdspmatrix_dev(GDSPMatrix mat);

// ret := A * x  (sparse matrix-vector product)
void mul_gdspmatrix_gdvec(GDVector ret, GDSPMatrix a, GDVector x, int nbg, int ntb);
// ret := A^T * x  (uses atomic scatter; ret zeroed internally)
void mul_gdspmatrixt_gdvec(GDVector ret, GDSPMatrix a, GDVector x, int nbg, int ntb);

#endif // __BNC_GDSPARSE_H__
