/********************************************************************************/
/* cgdslinear.h: Complex double-double GPU Linear Computation (CUDA)             */
/*   complex stored as separate real/imag gds_real arrays (SoA).                 */
/*   Built on the gqd/gdtq device dd arithmetic (gdslinear.h).                    */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_CGDSLINEAR_H__
#define __BNC_CGDSLINEAR_H__

#include "gdslinear.h"   // gds_real + inline device dd math (gqd.cuh) + GDD types
#include "cdslinear.h"   // CDSVector / CDSMatrix (host complex dd: re/im DD parts)

#ifndef CGDS_SIZE
#define CGDS_SIZE DSSIZE
#endif

// complex dd GPU vector (SoA: re[], im[] of gds_real)
typedef struct { long int dim; gds_real *re, *im; } cgdsvector;
typedef cgdsvector *CGDSVector;
// complex dd GPU matrix (row-major, stride col_dim; SoA re[], im[])
typedef struct { long int row_dim, col_dim; gds_real *re, *im; } cgdsmatrix;
typedef cgdsmatrix *CGDSMatrix;

/*--- vector ---*/
CGDSVector init_cgdsvector(long int dim);
void free_cgdsvector(CGDSVector v);
CGDSVector init_cgdsvector_dev(long int dim);
void free_cgdsvector_dev(CGDSVector v);
void subst_cgdsvector_dev_cdsvec(CGDSVector dev, CDSVector cpu); // host->dev
void subst_cdsvector_cgdsvec_dev(CDSVector cpu, CGDSVector dev); // dev->host
void add_cgdsvector_dev(CGDSVector c, CGDSVector a, CGDSVector b, int nbg, int ntb);
void sub_cgdsvector_dev(CGDSVector c, CGDSVector a, CGDSVector b, int nbg, int ntb);
void cmul_cgdsvector_dev(CGDSVector c, gds_real vre, gds_real vim, CGDSVector a, int nbg, int ntb);
void subst_cgdsvector_dev(CGDSVector ret, CGDSVector v, int nbg, int ntb);
void set0_cgdsvector_dev(CGDSVector ret, int nbg, int ntb);

/*--- matrix ---*/
CGDSMatrix init_cgdsmatrix(long int row_dim, long int col_dim);
void free_cgdsmatrix(CGDSMatrix m);
CGDSMatrix init_cgdsmatrix_dev(long int row_dim, long int col_dim);
void free_cgdsmatrix_dev(CGDSMatrix m);
void subst_cgdsmatrix_dev_cdsmat(CGDSMatrix dev, CDSMatrix cpu); // host->dev
void subst_cdsmatrix_cgdsmat_dev(CDSMatrix cpu, CGDSMatrix dev); // dev->host
void mul_cgdsmatrix_dev(CGDSMatrix ret, CGDSMatrix a, CGDSMatrix b, int nbg, int ntb);
void add_cgdsmatrix_dev(CGDSMatrix c, CGDSMatrix a, CGDSMatrix b, int nbg, int ntb);
void sub_cgdsmatrix_dev(CGDSMatrix c, CGDSMatrix a, CGDSMatrix b, int nbg, int ntb);
void cmul_cgdsmatrix_dev(CGDSMatrix c, gds_real sre, gds_real sim, CGDSMatrix a, int nbg, int ntb);
void transpose_cgdsmatrix_dev(CGDSMatrix c, CGDSMatrix a, int nbg, int ntb);
void conjtrans_cgdsmatrix_dev(CGDSMatrix c, CGDSMatrix a, int nbg, int ntb);
void subst_cgdsmatrix_dev(CGDSMatrix c, CGDSMatrix a, int nbg, int ntb);
void set0_cgdsmatrix_dev(CGDSMatrix c, int nbg, int ntb);
void setI_cgdsmatrix_dev(CGDSMatrix c, int nbg, int ntb);
void mul_cgdsmatrix_cgdsvec(CGDSVector v, CGDSMatrix a, CGDSVector vb, int nbg, int ntb);
void mul_cgdsmatrixt_cgdsvec(CGDSVector v, CGDSMatrix a, CGDSVector vb, int nbg, int ntb);

#endif // __BNC_CGDSLINEAR_H__
