/********************************************************************************/
/* cgddlinear.h: Complex double-double GPU Linear Computation (CUDA)             */
/*   complex stored as separate real/imag gdd_real arrays (SoA).                 */
/*   Built on the gqd/gdtq device dd arithmetic (gddlinear.h).                    */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_CGDDLINEAR_H__
#define __BNC_CGDDLINEAR_H__

#include "gddlinear.h"   // gdd_real + inline device dd math (gqd.cuh) + GDD types
#include "cddlinear.h"   // CDDVector / CDDMatrix (host complex dd: re/im DD parts)

#ifndef CGDD_SIZE
#define CGDD_SIZE DDSIZE
#endif

// complex dd GPU vector (SoA: re[], im[] of gdd_real)
typedef struct { long int dim; gdd_real *re, *im; } cgddvector;
typedef cgddvector *CGDDVector;
// complex dd GPU matrix (row-major, stride col_dim; SoA re[], im[])
typedef struct { long int row_dim, col_dim; gdd_real *re, *im; } cgddmatrix;
typedef cgddmatrix *CGDDMatrix;

/*--- vector ---*/
CGDDVector init_cgddvector(long int dim);
void free_cgddvector(CGDDVector v);
CGDDVector init_cgddvector_dev(long int dim);
void free_cgddvector_dev(CGDDVector v);
void subst_cgddvector_dev_cddvec(CGDDVector dev, CDDVector cpu); // host->dev
void subst_cddvector_cgddvec_dev(CDDVector cpu, CGDDVector dev); // dev->host
void add_cgddvector_dev(CGDDVector c, CGDDVector a, CGDDVector b, int nbg, int ntb);
void sub_cgddvector_dev(CGDDVector c, CGDDVector a, CGDDVector b, int nbg, int ntb);
void cmul_cgddvector_dev(CGDDVector c, gdd_real vre, gdd_real vim, CGDDVector a, int nbg, int ntb);
void subst_cgddvector_dev(CGDDVector ret, CGDDVector v, int nbg, int ntb);
void set0_cgddvector_dev(CGDDVector ret, int nbg, int ntb);

/*--- matrix ---*/
CGDDMatrix init_cgddmatrix(long int row_dim, long int col_dim);
void free_cgddmatrix(CGDDMatrix m);
CGDDMatrix init_cgddmatrix_dev(long int row_dim, long int col_dim);
void free_cgddmatrix_dev(CGDDMatrix m);
void subst_cgddmatrix_dev_cddmat(CGDDMatrix dev, CDDMatrix cpu); // host->dev
void subst_cddmatrix_cgddmat_dev(CDDMatrix cpu, CGDDMatrix dev); // dev->host
void mul_cgddmatrix_dev(CGDDMatrix ret, CGDDMatrix a, CGDDMatrix b, int nbg, int ntb);
void add_cgddmatrix_dev(CGDDMatrix c, CGDDMatrix a, CGDDMatrix b, int nbg, int ntb);
void sub_cgddmatrix_dev(CGDDMatrix c, CGDDMatrix a, CGDDMatrix b, int nbg, int ntb);
void cmul_cgddmatrix_dev(CGDDMatrix c, gdd_real sre, gdd_real sim, CGDDMatrix a, int nbg, int ntb);
void transpose_cgddmatrix_dev(CGDDMatrix c, CGDDMatrix a, int nbg, int ntb);
void conjtrans_cgddmatrix_dev(CGDDMatrix c, CGDDMatrix a, int nbg, int ntb);
void subst_cgddmatrix_dev(CGDDMatrix c, CGDDMatrix a, int nbg, int ntb);
void set0_cgddmatrix_dev(CGDDMatrix c, int nbg, int ntb);
void setI_cgddmatrix_dev(CGDDMatrix c, int nbg, int ntb);
void mul_cgddmatrix_cgddvec(CGDDVector v, CGDDMatrix a, CGDDVector vb, int nbg, int ntb);
void mul_cgddmatrixt_cgddvec(CGDDVector v, CGDDMatrix a, CGDDVector vb, int nbg, int ntb);

#endif // __BNC_CGDDLINEAR_H__
