/********************************************************************************/
/* gdssparse.h: double-double SPARSE (CSR) SpMV on GPU (CUDA)                    */
/*   Standard CSR (row_ptr/col_idx/val) with gds_real values.                     */
/*   Reuses GDSVector (gdslinear.h) for the dense in/out vectors.                 */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_GDSSPARSE_H__
#define __BNC_GDSSPARSE_H__

#include "cuda_runtime.h"
#include "gdslinear.h"   // gds_real type (header-only; no gqd.cu link dependency)

typedef struct
{
	long int row_dim, col_dim, nnz;
	long int *row_ptr;   // length row_dim + 1
	long int *col_idx;   // length nnz
	gds_real *val;       // length nnz
} gdsspmatrix;
typedef gdsspmatrix *GDSSPMatrix;

// build a device CSR matrix from host CSR arrays (host gds_real val)
GDSSPMatrix init_gdsspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                 const long int *row_ptr, const long int *col_idx, const gds_real *val);
void free_gdsspmatrix_dev(GDSSPMatrix mat);

// y_dev := A * x_dev  (sparse matrix-vector product, dd arithmetic; raw device gds_real arrays)
void mul_gdsspmatrix(gds_real *y_dev, GDSSPMatrix a, const gds_real *x_dev, int nbg, int ntb);

#endif // __BNC_GDSSPARSE_H__
