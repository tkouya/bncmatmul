/********************************************************************************/
/* cgtslinear.h: Complex double-double GPU Linear Computation (CUDA)             */
/*   complex stored as separate real/imag gts_real arrays (SoA).                 */
/*   Built on the gqd/gdtq device dd arithmetic (gdslinear.h).                    */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_CGTSLINEAR_H__
#define __BNC_CGTSLINEAR_H__

#include "gdslinear.h"   // gts_real + inline device dd math (gqd.cuh) + GDD types
#include "ctslinear.h"   // CTSVector / CTSMatrix (host complex dd: re/im DD parts)

#ifndef CGTS_SIZE
#define CGTS_SIZE TSSIZE
#endif

// complex dd GPU vector (SoA: re[], im[] of gts_real)
typedef struct { long int dim; gts_real *re, *im; } cgtsvector;
typedef cgtsvector *CGTSVector;
// complex dd GPU matrix (row-major, stride col_dim; SoA re[], im[])
typedef struct { long int row_dim, col_dim; gts_real *re, *im; } cgtsmatrix;
typedef cgtsmatrix *CGTSMatrix;

/*--- vector ---*/
CGTSVector init_cgtsvector(long int dim);
void free_cgtsvector(CGTSVector v);
CGTSVector init_cgtsvector_dev(long int dim);
void free_cgtsvector_dev(CGTSVector v);
void subst_cgtsvector_dev_ctsvec(CGTSVector dev, CTSVector cpu); // host->dev
void subst_ctsvector_cgtsvec_dev(CTSVector cpu, CGTSVector dev); // dev->host
void add_cgtsvector_dev(CGTSVector c, CGTSVector a, CGTSVector b, int nbg, int ntb);
void sub_cgtsvector_dev(CGTSVector c, CGTSVector a, CGTSVector b, int nbg, int ntb);
void cmul_cgtsvector_dev(CGTSVector c, gts_real vre, gts_real vim, CGTSVector a, int nbg, int ntb);
void subst_cgtsvector_dev(CGTSVector ret, CGTSVector v, int nbg, int ntb);
void set0_cgtsvector_dev(CGTSVector ret, int nbg, int ntb);

/*--- matrix ---*/
CGTSMatrix init_cgtsmatrix(long int row_dim, long int col_dim);
void free_cgtsmatrix(CGTSMatrix m);
CGTSMatrix init_cgtsmatrix_dev(long int row_dim, long int col_dim);
void free_cgtsmatrix_dev(CGTSMatrix m);
void subst_cgtsmatrix_dev_ctsmat(CGTSMatrix dev, CTSMatrix cpu); // host->dev
void subst_ctsmatrix_cgtsmat_dev(CTSMatrix cpu, CGTSMatrix dev); // dev->host
void mul_cgtsmatrix_dev(CGTSMatrix ret, CGTSMatrix a, CGTSMatrix b, int nbg, int ntb);
void add_cgtsmatrix_dev(CGTSMatrix c, CGTSMatrix a, CGTSMatrix b, int nbg, int ntb);
void sub_cgtsmatrix_dev(CGTSMatrix c, CGTSMatrix a, CGTSMatrix b, int nbg, int ntb);
void cmul_cgtsmatrix_dev(CGTSMatrix c, gts_real sre, gts_real sim, CGTSMatrix a, int nbg, int ntb);
void transpose_cgtsmatrix_dev(CGTSMatrix c, CGTSMatrix a, int nbg, int ntb);
void conjtrans_cgtsmatrix_dev(CGTSMatrix c, CGTSMatrix a, int nbg, int ntb);
void subst_cgtsmatrix_dev(CGTSMatrix c, CGTSMatrix a, int nbg, int ntb);
void set0_cgtsmatrix_dev(CGTSMatrix c, int nbg, int ntb);
void setI_cgtsmatrix_dev(CGTSMatrix c, int nbg, int ntb);
void mul_cgtsmatrix_cgtsvec(CGTSVector v, CGTSMatrix a, CGTSVector vb, int nbg, int ntb);
void mul_cgtsmatrixt_cgtsvec(CGTSVector v, CGTSMatrix a, CGTSVector vb, int nbg, int ntb);

#endif // __BNC_CGTSLINEAR_H__
