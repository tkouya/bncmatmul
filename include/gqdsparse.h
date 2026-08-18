/********************************************************************************/
/* gqdsparse.h: double-double SPARSE (CSR) SpMV on GPU (CUDA)                    */
/*   Standard CSR (row_ptr/col_idx/val) with gqd_real values.                     */
/*   Reuses GQDVector (gddlinear.h) for the dense in/out vectors.                 */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_GQDSPARSE_H__
#define __BNC_GQDSPARSE_H__

#include "cuda_runtime.h"
#include "gddlinear.h"   // gqd_real type (header-only; no gqd.cu link dependency)

typedef struct
{
	long int row_dim, col_dim, nnz;
	long int *row_ptr;   // length row_dim + 1
	long int *col_idx;   // length nnz
	gqd_real *val;       // length nnz
} gqdspmatrix;
typedef gqdspmatrix *GQDSPMatrix;

// build a device CSR matrix from host CSR arrays (host gqd_real val)
GQDSPMatrix init_gqdspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                 const long int *row_ptr, const long int *col_idx, const gqd_real *val);
void free_gqdspmatrix_dev(GQDSPMatrix mat);

// y_dev := A * x_dev  (sparse matrix-vector product, qd arithmetic; raw device gqd_real arrays)
void mul_gqdspmatrix(gqd_real *y_dev, GQDSPMatrix a, const gqd_real *x_dev, int nbg, int ntb);

#endif // __BNC_GQDSPARSE_H__
