/********************************************************************************/
/* gqssparse.h: double-double SPARSE (CSR) SpMV on GPU (CUDA)                    */
/*   Standard CSR (row_ptr/col_idx/val) with gqs_real values.                     */
/*   Reuses GQSVector (gdslinear.h) for the dense in/out vectors.                 */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_GQSSPARSE_H__
#define __BNC_GQSSPARSE_H__

#include "cuda_runtime.h"
#include "gdslinear.h"   // gqs_real type (header-only; no gqd.cu link dependency)

typedef struct
{
	long int row_dim, col_dim, nnz;
	long int *row_ptr;   // length row_dim + 1
	long int *col_idx;   // length nnz
	gqs_real *val;       // length nnz
} gqsspmatrix;
typedef gqsspmatrix *GQSSPMatrix;

// build a device CSR matrix from host CSR arrays (host gqs_real val)
GQSSPMatrix init_gqsspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                 const long int *row_ptr, const long int *col_idx, const gqs_real *val);
void free_gqsspmatrix_dev(GQSSPMatrix mat);

// y_dev := A * x_dev  (sparse matrix-vector product, dd arithmetic; raw device gqs_real arrays)
void mul_gqsspmatrix(gqs_real *y_dev, GQSSPMatrix a, const gqs_real *x_dev, int nbg, int ntb);

#endif // __BNC_GQSSPARSE_H__
