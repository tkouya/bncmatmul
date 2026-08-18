/********************************************************************************/
/* cgtdlinear.h: Complex double-double GPU Linear Computation (CUDA)             */
/*   complex stored as separate real/imag gtd_real arrays (SoA).                 */
/*   Built on the gqd/gdtq device dd arithmetic (gddlinear.h).                    */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_CGTDLINEAR_H__
#define __BNC_CGTDLINEAR_H__

#include "gddlinear.h"   // gtd_real + inline device dd math (gqd.cuh) + GDD types
#include "ctdlinear.h"   // CTDVector / CTDMatrix (host complex dd: re/im DD parts)

#ifndef CGTD_SIZE
#define CGTD_SIZE TDSIZE
#endif

// complex dd GPU vector (SoA: re[], im[] of gtd_real)
typedef struct { long int dim; gtd_real *re, *im; } cgtdvector;
typedef cgtdvector *CGTDVector;
// complex dd GPU matrix (row-major, stride col_dim; SoA re[], im[])
typedef struct { long int row_dim, col_dim; gtd_real *re, *im; } cgtdmatrix;
typedef cgtdmatrix *CGTDMatrix;

/*--- vector ---*/
CGTDVector init_cgtdvector(long int dim);
void free_cgtdvector(CGTDVector v);
CGTDVector init_cgtdvector_dev(long int dim);
void free_cgtdvector_dev(CGTDVector v);
void subst_cgtdvector_dev_ctdvec(CGTDVector dev, CTDVector cpu); // host->dev
void subst_ctdvector_cgtdvec_dev(CTDVector cpu, CGTDVector dev); // dev->host
void add_cgtdvector_dev(CGTDVector c, CGTDVector a, CGTDVector b, int nbg, int ntb);
void sub_cgtdvector_dev(CGTDVector c, CGTDVector a, CGTDVector b, int nbg, int ntb);
void cmul_cgtdvector_dev(CGTDVector c, gtd_real vre, gtd_real vim, CGTDVector a, int nbg, int ntb);
void subst_cgtdvector_dev(CGTDVector ret, CGTDVector v, int nbg, int ntb);
void set0_cgtdvector_dev(CGTDVector ret, int nbg, int ntb);

/*--- matrix ---*/
CGTDMatrix init_cgtdmatrix(long int row_dim, long int col_dim);
void free_cgtdmatrix(CGTDMatrix m);
CGTDMatrix init_cgtdmatrix_dev(long int row_dim, long int col_dim);
void free_cgtdmatrix_dev(CGTDMatrix m);
void subst_cgtdmatrix_dev_ctdmat(CGTDMatrix dev, CTDMatrix cpu); // host->dev
void subst_ctdmatrix_cgtdmat_dev(CTDMatrix cpu, CGTDMatrix dev); // dev->host
void mul_cgtdmatrix_dev(CGTDMatrix ret, CGTDMatrix a, CGTDMatrix b, int nbg, int ntb);
void add_cgtdmatrix_dev(CGTDMatrix c, CGTDMatrix a, CGTDMatrix b, int nbg, int ntb);
void sub_cgtdmatrix_dev(CGTDMatrix c, CGTDMatrix a, CGTDMatrix b, int nbg, int ntb);
void cmul_cgtdmatrix_dev(CGTDMatrix c, gtd_real sre, gtd_real sim, CGTDMatrix a, int nbg, int ntb);
void transpose_cgtdmatrix_dev(CGTDMatrix c, CGTDMatrix a, int nbg, int ntb);
void conjtrans_cgtdmatrix_dev(CGTDMatrix c, CGTDMatrix a, int nbg, int ntb);
void subst_cgtdmatrix_dev(CGTDMatrix c, CGTDMatrix a, int nbg, int ntb);
void set0_cgtdmatrix_dev(CGTDMatrix c, int nbg, int ntb);
void setI_cgtdmatrix_dev(CGTDMatrix c, int nbg, int ntb);
void mul_cgtdmatrix_cgtdvec(CGTDVector v, CGTDMatrix a, CGTDVector vb, int nbg, int ntb);
void mul_cgtdmatrixt_cgtdvec(CGTDVector v, CGTDMatrix a, CGTDVector vb, int nbg, int ntb);

#endif // __BNC_CGTDLINEAR_H__
