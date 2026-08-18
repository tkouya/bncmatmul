/********************************************************************************/
/* cgddsparse.h: complex double-double SPARSE (CSR) SpMV on GPU (CUDA)           */
/*   complex CSR with separate re/im gdd_real value arrays (SoA).                 */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_CGDDSPARSE_H__
#define __BNC_CGDDSPARSE_H__

#include "cuda_runtime.h"
#include "gddlinear.h"   // gdd_real

typedef struct
{
	long int row_dim, col_dim, nnz;
	long int *row_ptr;
	long int *col_idx;
	gdd_real *val_re, *val_im;
} cgddspmatrix;
typedef cgddspmatrix *CGDDSPMatrix;

CGDDSPMatrix init_cgddspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                   const long int *row_ptr, const long int *col_idx,
                                   const gdd_real *val_re, const gdd_real *val_im);
void free_cgddspmatrix_dev(CGDDSPMatrix mat);

// (yre, yim) := A * (xre, xim)  (raw device gdd_real arrays)
void mul_cgddspmatrix(gdd_real *yre, gdd_real *yim, CGDDSPMatrix a, const gdd_real *xre, const gdd_real *xim, int nbg, int ntb);

#endif // __BNC_CGDDSPARSE_H__
