/********************************************************************************/
/* cgqdlinear.h: Complex double-double GPU Linear Computation (CUDA)             */
/*   complex stored as separate real/imag gqd_real arrays (SoA).                 */
/*   Built on the gqd/gdtq device dd arithmetic (gddlinear.h).                    */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_CGQDLINEAR_H__
#define __BNC_CGQDLINEAR_H__

#include "gddlinear.h"   // gqd_real + inline device dd math (gqd.cuh) + GDD types
#include "cqdlinear.h"   // CQDVector / CQDMatrix (host complex dd: re/im DD parts)

#ifndef CGQD_SIZE
#define CGQD_SIZE QDSIZE
#endif

// complex dd GPU vector (SoA: re[], im[] of gqd_real)
typedef struct { long int dim; gqd_real *re, *im; } cgqdvector;
typedef cgqdvector *CGQDVector;
// complex dd GPU matrix (row-major, stride col_dim; SoA re[], im[])
typedef struct { long int row_dim, col_dim; gqd_real *re, *im; } cgqdmatrix;
typedef cgqdmatrix *CGQDMatrix;

/*--- vector ---*/
CGQDVector init_cgqdvector(long int dim);
void free_cgqdvector(CGQDVector v);
CGQDVector init_cgqdvector_dev(long int dim);
void free_cgqdvector_dev(CGQDVector v);
void subst_cgqdvector_dev_cqdvec(CGQDVector dev, CQDVector cpu); // host->dev
void subst_cqdvector_cgqdvec_dev(CQDVector cpu, CGQDVector dev); // dev->host
void add_cgqdvector_dev(CGQDVector c, CGQDVector a, CGQDVector b, int nbg, int ntb);
void sub_cgqdvector_dev(CGQDVector c, CGQDVector a, CGQDVector b, int nbg, int ntb);
void cmul_cgqdvector_dev(CGQDVector c, gqd_real vre, gqd_real vim, CGQDVector a, int nbg, int ntb);
void subst_cgqdvector_dev(CGQDVector ret, CGQDVector v, int nbg, int ntb);
void set0_cgqdvector_dev(CGQDVector ret, int nbg, int ntb);

/*--- matrix ---*/
CGQDMatrix init_cgqdmatrix(long int row_dim, long int col_dim);
void free_cgqdmatrix(CGQDMatrix m);
CGQDMatrix init_cgqdmatrix_dev(long int row_dim, long int col_dim);
void free_cgqdmatrix_dev(CGQDMatrix m);
void subst_cgqdmatrix_dev_cqdmat(CGQDMatrix dev, CQDMatrix cpu); // host->dev
void subst_cqdmatrix_cgqdmat_dev(CQDMatrix cpu, CGQDMatrix dev); // dev->host
void mul_cgqdmatrix_dev(CGQDMatrix ret, CGQDMatrix a, CGQDMatrix b, int nbg, int ntb);
void add_cgqdmatrix_dev(CGQDMatrix c, CGQDMatrix a, CGQDMatrix b, int nbg, int ntb);
void sub_cgqdmatrix_dev(CGQDMatrix c, CGQDMatrix a, CGQDMatrix b, int nbg, int ntb);
void cmul_cgqdmatrix_dev(CGQDMatrix c, gqd_real sre, gqd_real sim, CGQDMatrix a, int nbg, int ntb);
void transpose_cgqdmatrix_dev(CGQDMatrix c, CGQDMatrix a, int nbg, int ntb);
void conjtrans_cgqdmatrix_dev(CGQDMatrix c, CGQDMatrix a, int nbg, int ntb);
void subst_cgqdmatrix_dev(CGQDMatrix c, CGQDMatrix a, int nbg, int ntb);
void set0_cgqdmatrix_dev(CGQDMatrix c, int nbg, int ntb);
void setI_cgqdmatrix_dev(CGQDMatrix c, int nbg, int ntb);
void mul_cgqdmatrix_cgqdvec(CGQDVector v, CGQDMatrix a, CGQDVector vb, int nbg, int ntb);
void mul_cgqdmatrixt_cgqdvec(CGQDVector v, CGQDMatrix a, CGQDVector vb, int nbg, int ntb);

#endif // __BNC_CGQDLINEAR_H__
