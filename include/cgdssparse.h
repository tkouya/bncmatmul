/********************************************************************************/
/* cgdssparse.h: complex double-double SPARSE (CSR) SpMV on GPU (CUDA)           */
/*   complex CSR with separate re/im gds_real value arrays (SoA).                 */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_CGDSSPARSE_H__
#define __BNC_CGDSSPARSE_H__

#include "cuda_runtime.h"
#include "gdslinear.h"   // gds_real

typedef struct
{
	long int row_dim, col_dim, nnz;
	long int *row_ptr;
	long int *col_idx;
	gds_real *val_re, *val_im;
} cgdsspmatrix;
typedef cgdsspmatrix *CGDSSPMatrix;

CGDSSPMatrix init_cgdsspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                   const long int *row_ptr, const long int *col_idx,
                                   const gds_real *val_re, const gds_real *val_im);
void free_cgdsspmatrix_dev(CGDSSPMatrix mat);

// (yre, yim) := A * (xre, xim)  (raw device gds_real arrays)
void mul_cgdsspmatrix(gds_real *yre, gds_real *yim, CGDSSPMatrix a, const gds_real *xre, const gds_real *xim, int nbg, int ntb);

#endif // __BNC_CGDSSPARSE_H__
