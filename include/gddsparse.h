/********************************************************************************/
/* gddsparse.h: double-double SPARSE (CSR) SpMV on GPU (CUDA)                    */
/*   Standard CSR (row_ptr/col_idx/val) with gdd_real values.                     */
/*   Reuses GDDVector (gddlinear.h) for the dense in/out vectors.                 */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_GDDSPARSE_H__
#define __BNC_GDDSPARSE_H__

#include "cuda_runtime.h"
#include "gddlinear.h"   // gdd_real type (header-only; no gqd.cu link dependency)

typedef struct
{
	long int row_dim, col_dim, nnz;
	long int *row_ptr;   // length row_dim + 1
	long int *col_idx;   // length nnz
	gdd_real *val;       // length nnz
} gddspmatrix;
typedef gddspmatrix *GDDSPMatrix;

// build a device CSR matrix from host CSR arrays (host gdd_real val)
GDDSPMatrix init_gddspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                 const long int *row_ptr, const long int *col_idx, const gdd_real *val);
void free_gddspmatrix_dev(GDDSPMatrix mat);

// y_dev := A * x_dev  (sparse matrix-vector product, dd arithmetic; raw device gdd_real arrays)
void mul_gddspmatrix(gdd_real *y_dev, GDDSPMatrix a, const gdd_real *x_dev, int nbg, int ntb);

#endif // __BNC_GDDSPARSE_H__
