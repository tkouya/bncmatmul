/********************************************************************************/
/* cgqslinear.h: Complex double-double GPU Linear Computation (CUDA)             */
/*   complex stored as separate real/imag gqs_real arrays (SoA).                 */
/*   Built on the gqd/gdtq device dd arithmetic (gdslinear.h).                    */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#ifndef __BNC_CGQSLINEAR_H__
#define __BNC_CGQSLINEAR_H__

#include "gdslinear.h"   // gqs_real + inline device dd math (gqd.cuh) + GDD types
#include "cqslinear.h"   // CQSVector / CQSMatrix (host complex dd: re/im DD parts)

#ifndef CGQS_SIZE
#define CGQS_SIZE QSSIZE
#endif

// complex dd GPU vector (SoA: re[], im[] of gqs_real)
typedef struct { long int dim; gqs_real *re, *im; } cgqsvector;
typedef cgqsvector *CGQSVector;
// complex dd GPU matrix (row-major, stride col_dim; SoA re[], im[])
typedef struct { long int row_dim, col_dim; gqs_real *re, *im; } cgqsmatrix;
typedef cgqsmatrix *CGQSMatrix;

/*--- vector ---*/
CGQSVector init_cgqsvector(long int dim);
void free_cgqsvector(CGQSVector v);
CGQSVector init_cgqsvector_dev(long int dim);
void free_cgqsvector_dev(CGQSVector v);
void subst_cgqsvector_dev_cqsvec(CGQSVector dev, CQSVector cpu); // host->dev
void subst_cqsvector_cgqsvec_dev(CQSVector cpu, CGQSVector dev); // dev->host
void add_cgqsvector_dev(CGQSVector c, CGQSVector a, CGQSVector b, int nbg, int ntb);
void sub_cgqsvector_dev(CGQSVector c, CGQSVector a, CGQSVector b, int nbg, int ntb);
void cmul_cgqsvector_dev(CGQSVector c, gqs_real vre, gqs_real vim, CGQSVector a, int nbg, int ntb);
void subst_cgqsvector_dev(CGQSVector ret, CGQSVector v, int nbg, int ntb);
void set0_cgqsvector_dev(CGQSVector ret, int nbg, int ntb);

/*--- matrix ---*/
CGQSMatrix init_cgqsmatrix(long int row_dim, long int col_dim);
void free_cgqsmatrix(CGQSMatrix m);
CGQSMatrix init_cgqsmatrix_dev(long int row_dim, long int col_dim);
void free_cgqsmatrix_dev(CGQSMatrix m);
void subst_cgqsmatrix_dev_cqsmat(CGQSMatrix dev, CQSMatrix cpu); // host->dev
void subst_cqsmatrix_cgqsmat_dev(CQSMatrix cpu, CGQSMatrix dev); // dev->host
void mul_cgqsmatrix_dev(CGQSMatrix ret, CGQSMatrix a, CGQSMatrix b, int nbg, int ntb);
void add_cgqsmatrix_dev(CGQSMatrix c, CGQSMatrix a, CGQSMatrix b, int nbg, int ntb);
void sub_cgqsmatrix_dev(CGQSMatrix c, CGQSMatrix a, CGQSMatrix b, int nbg, int ntb);
void cmul_cgqsmatrix_dev(CGQSMatrix c, gqs_real sre, gqs_real sim, CGQSMatrix a, int nbg, int ntb);
void transpose_cgqsmatrix_dev(CGQSMatrix c, CGQSMatrix a, int nbg, int ntb);
void conjtrans_cgqsmatrix_dev(CGQSMatrix c, CGQSMatrix a, int nbg, int ntb);
void subst_cgqsmatrix_dev(CGQSMatrix c, CGQSMatrix a, int nbg, int ntb);
void set0_cgqsmatrix_dev(CGQSMatrix c, int nbg, int ntb);
void setI_cgqsmatrix_dev(CGQSMatrix c, int nbg, int ntb);
void mul_cgqsmatrix_cgqsvec(CGQSVector v, CGQSMatrix a, CGQSVector vb, int nbg, int ntb);
void mul_cgqsmatrixt_cgqsvec(CGQSVector v, CGQSMatrix a, CGQSVector vb, int nbg, int ntb);

#endif // __BNC_CGQSLINEAR_H__
