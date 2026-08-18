/********************************************************************************/
/* cgqdsparse.h: complex double-double SPARSE (CSR) SpMV on GPU (CUDA)           */
/*   complex CSR with separate re/im gqd_real value arrays (SoA).                 */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_CGQDSPARSE_H__
#define __BNC_CGQDSPARSE_H__

#include "cuda_runtime.h"
#include "gddlinear.h"   // gqd_real

typedef struct
{
	long int row_dim, col_dim, nnz;
	long int *row_ptr;
	long int *col_idx;
	gqd_real *val_re, *val_im;
} cgqdspmatrix;
typedef cgqdspmatrix *CGQDSPMatrix;

CGQDSPMatrix init_cgqdspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                   const long int *row_ptr, const long int *col_idx,
                                   const gqd_real *val_re, const gqd_real *val_im);
void free_cgqdspmatrix_dev(CGQDSPMatrix mat);

// (yre, yim) := A * (xre, xim)  (raw device gqd_real arrays)
void mul_cgqdspmatrix(gqd_real *yre, gqd_real *yim, CGQDSPMatrix a, const gqd_real *xre, const gqd_real *xim, int nbg, int ntb);

#endif // __BNC_CGQDSPARSE_H__
