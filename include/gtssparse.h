/********************************************************************************/
/* gtssparse.h: double-double SPARSE (CSR) SpMV on GPU (CUDA)                    */
/*   Standard CSR (row_ptr/col_idx/val) with gts_real values.                     */
/*   Reuses GTSVector (gdslinear.h) for the dense in/out vectors.                 */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_GTSSPARSE_H__
#define __BNC_GTSSPARSE_H__

#include "cuda_runtime.h"
#include "gdslinear.h"   // gts_real type (header-only; no gqd.cu link dependency)

typedef struct
{
	long int row_dim, col_dim, nnz;
	long int *row_ptr;   // length row_dim + 1
	long int *col_idx;   // length nnz
	gts_real *val;       // length nnz
} gtsspmatrix;
typedef gtsspmatrix *GTSSPMatrix;

// build a device CSR matrix from host CSR arrays (host gts_real val)
GTSSPMatrix init_gtsspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                 const long int *row_ptr, const long int *col_idx, const gts_real *val);
void free_gtsspmatrix_dev(GTSSPMatrix mat);

// y_dev := A * x_dev  (sparse matrix-vector product, dd arithmetic; raw device gts_real arrays)
void mul_gtsspmatrix(gts_real *y_dev, GTSSPMatrix a, const gts_real *x_dev, int nbg, int ntb);

#endif // __BNC_GTSSPARSE_H__
