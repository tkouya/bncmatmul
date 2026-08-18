/********************************************************************************/
/* gcfsparse.h: native complex float SPARSE (CSR) SpMV on GPU (CUDA)            */
/*   complex CSR with separate re/im float value arrays (SoA).                   */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_GCFSPARSE_H__
#define __BNC_GCFSPARSE_H__

#include "cuda_runtime.h"

typedef struct
{
	long int row_dim, col_dim, nnz;
	long int *row_ptr;        // length row_dim + 1
	long int *col_idx;        // length nnz
	float   *val_re, *val_im; // length nnz each
} gcfspmatrix;
typedef gcfspmatrix *GCFSPMatrix;

GCFSPMatrix init_gcfspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                 const long int *row_ptr, const long int *col_idx,
                                 const float *val_re, const float *val_im);
void free_gcfspmatrix_dev(GCFSPMatrix mat);

// (yre, yim) := A * (xre, xim)  (raw device arrays)
void mul_gcfspmatrix(float *yre, float *yim, GCFSPMatrix a, const float *xre, const float *xim, int nbg, int ntb);

#endif // __BNC_GCFSPARSE_H__
