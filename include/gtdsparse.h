/********************************************************************************/
/* gtdsparse.h: double-double SPARSE (CSR) SpMV on GPU (CUDA)                    */
/*   Standard CSR (row_ptr/col_idx/val) with gtd_real values.                     */
/*   Reuses GTDVector (gddlinear.h) for the dense in/out vectors.                 */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_GTDSPARSE_H__
#define __BNC_GTDSPARSE_H__

#include "cuda_runtime.h"
#include "gddlinear.h"   // gtd_real type (header-only; no gqd.cu link dependency)

typedef struct
{
	long int row_dim, col_dim, nnz;
	long int *row_ptr;   // length row_dim + 1
	long int *col_idx;   // length nnz
	gtd_real *val;       // length nnz
} gtdspmatrix;
typedef gtdspmatrix *GTDSPMatrix;

// build a device CSR matrix from host CSR arrays (host gtd_real val)
GTDSPMatrix init_gtdspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                 const long int *row_ptr, const long int *col_idx, const gtd_real *val);
void free_gtdspmatrix_dev(GTDSPMatrix mat);

// y_dev := A * x_dev  (sparse matrix-vector product, td arithmetic; raw device gtd_real arrays)
void mul_gtdspmatrix(gtd_real *y_dev, GTDSPMatrix a, const gtd_real *x_dev, int nbg, int ntb);

#endif // __BNC_GTDSPARSE_H__
