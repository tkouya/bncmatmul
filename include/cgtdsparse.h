/********************************************************************************/
/* cgtdsparse.h: complex double-double SPARSE (CSR) SpMV on GPU (CUDA)           */
/*   complex CSR with separate re/im gtd_real value arrays (SoA).                 */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_CGTDSPARSE_H__
#define __BNC_CGTDSPARSE_H__

#include "cuda_runtime.h"
#include "gddlinear.h"   // gtd_real

typedef struct
{
	long int row_dim, col_dim, nnz;
	long int *row_ptr;
	long int *col_idx;
	gtd_real *val_re, *val_im;
} cgtdspmatrix;
typedef cgtdspmatrix *CGTDSPMatrix;

CGTDSPMatrix init_cgtdspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                   const long int *row_ptr, const long int *col_idx,
                                   const gtd_real *val_re, const gtd_real *val_im);
void free_cgtdspmatrix_dev(CGTDSPMatrix mat);

// (yre, yim) := A * (xre, xim)  (raw device gtd_real arrays)
void mul_cgtdspmatrix(gtd_real *yre, gtd_real *yim, CGTDSPMatrix a, const gtd_real *xre, const gtd_real *xim, int nbg, int ntb);

#endif // __BNC_CGTDSPARSE_H__
