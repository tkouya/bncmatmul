/********************************************************************************/
/* cgqssparse.h: complex double-double SPARSE (CSR) SpMV on GPU (CUDA)           */
/*   complex CSR with separate re/im gqs_real value arrays (SoA).                 */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_CGQSSPARSE_H__
#define __BNC_CGQSSPARSE_H__

#include "cuda_runtime.h"
#include "gdslinear.h"   // gqs_real

typedef struct
{
	long int row_dim, col_dim, nnz;
	long int *row_ptr;
	long int *col_idx;
	gqs_real *val_re, *val_im;
} cgqsspmatrix;
typedef cgqsspmatrix *CGQSSPMatrix;

CGQSSPMatrix init_cgqsspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                   const long int *row_ptr, const long int *col_idx,
                                   const gqs_real *val_re, const gqs_real *val_im);
void free_cgqsspmatrix_dev(CGQSSPMatrix mat);

// (yre, yim) := A * (xre, xim)  (raw device gqs_real arrays)
void mul_cgqsspmatrix(gqs_real *yre, gqs_real *yim, CGQSSPMatrix a, const gqs_real *xre, const gqs_real *xim, int nbg, int ntb);

#endif // __BNC_CGQSSPARSE_H__
