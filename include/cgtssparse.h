/********************************************************************************/
/* cgtssparse.h: complex double-double SPARSE (CSR) SpMV on GPU (CUDA)           */
/*   complex CSR with separate re/im gts_real value arrays (SoA).                 */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_CGTSSPARSE_H__
#define __BNC_CGTSSPARSE_H__

#include "cuda_runtime.h"
#include "gdslinear.h"   // gts_real

typedef struct
{
	long int row_dim, col_dim, nnz;
	long int *row_ptr;
	long int *col_idx;
	gts_real *val_re, *val_im;
} cgtsspmatrix;
typedef cgtsspmatrix *CGTSSPMatrix;

CGTSSPMatrix init_cgtsspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                   const long int *row_ptr, const long int *col_idx,
                                   const gts_real *val_re, const gts_real *val_im);
void free_cgtsspmatrix_dev(CGTSSPMatrix mat);

// (yre, yim) := A * (xre, xim)  (raw device gts_real arrays)
void mul_cgtsspmatrix(gts_real *yre, gts_real *yim, CGTSSPMatrix a, const gts_real *xre, const gts_real *xim, int nbg, int ntb);

#endif // __BNC_CGTSSPARSE_H__
