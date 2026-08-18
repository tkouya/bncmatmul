/********************************************************************************/
/* gtslinear.h: Triple-double precision Linear Computation Library (GPU + CUDA) */
/* Copyright (C) 2015-2026 Tomonori Kouya                                       */
/*                                                                              */
/* GTS (triple-double, 3 IEEE doubles, ~47 decimal digits) GPU types and        */
/* prototypes.  Mirrors the structure of gdslinear.h but contains only the      */
/* GTS-specific additions.  Shared CPU / GDS / GQS infrastructure is pulled     */
/* in via #include "gdslinear.h" so that types like GQSMatrix and helpers like  */
/* qs2gqs_dev are not duplicated.                                               */
/*                                                                              */
/* This program is free software: you can redistribute it and/or modify it      */
/* under the terms of the GNU Lesser General Public License as published by the */
/* Free Software Foundation, either version 3 of the License or any later       */
/* version.                                                                     */
/********************************************************************************/
#ifndef __BNC_GTSLINEAR_H__
  #define __BNC_GTSLINEAR_H__

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>   /* see note in gdslinear.h */
#include <string>

/* CPU side */
#include "rds.h"

/* bncmatmul's c_dd_qd.h defines `#define nint(a) (round(a))` which,
 * if it has leaked into our scope (via a prior gdslinear.h include for
 * example), would mangle qd::nint() inside qd/td_inline.h.  Undef it
 * defensively before pulling tsfloat.h. */
#ifdef nint
#  undef nint
#endif
#include "rds.h"   /* dtq-0.0.2: triple-double tsfloat type     */

#include "tslinear.h"     /* TSMatrix / TSVector (SoA host containers) */

/* GPU side: types and inline functions */
#include "cuda_runtime.h"
#include "gqd_type.h"     /* gdtq-0.0.2: single header declares all gdd/gtd/gqd + gds/gts/gqs types */

/* Pull in the rest of the GDS / GQS / CPU shared infrastructure.  Doing
 * this BEFORE the GTS definitions below ensures that GDSMatrix and
 * GQSMatrix are visible for callers that include only gtslinear.h, and
 * also avoids re-declaring all the gqs_* helpers / typedefs that were
 * already provided by gdslinear.h. */
#include "gdslinear.h"

/* ============================================================
 * GTS constants
 * ============================================================ */
static gts_real _const_gts_zero = {0.0, 0.0, 0.0};
static gts_real _const_gts_one  = {1.0, 0.0, 0.0};


/* ============================================================
 * GTS <-> tsfloat conversion helpers (host)
 * ============================================================ */

/* gts_real (GPU) -> tsfloat (host) by value. */
inline tsfloat gts_get_ts(gts_real gval)
{
    tsfloat ret;
    ret.val[0] = gval.x;
    ret.val[1] = gval.y;
    ret.val[2] = gval.z;
    return ret;
}

/* tsval := gtsval (both host pointers). */
inline void gts2ts(tsfloat *tsval, gts_real *gtsval)
{
    tsval->val[0] = gtsval->x;
    tsval->val[1] = gtsval->y;
    tsval->val[2] = gtsval->z;
}

/* gtsval := tsval (both host pointers). */
inline void ts2gts(gts_real *gtsval, tsfloat *tsval)
{
    gtsval->x = tsval->val[0];
    gtsval->y = tsval->val[1];
    gtsval->z = tsval->val[2];
}


/* ============================================================
 * GTS <-> tsfloat conversion helpers (host <-> device)
 *
 * Naming follows gdslinear.h convention: <src>2<dst>_dev means the
 * destination lives on the device (or "_dev2" prefix for the source).
 * ============================================================ */

/* host tsfloat -> device gts_real (3 small cudaMemcpy calls). */
inline void ts2gts_dev(gts_real *gtsval_dev, tsfloat *tsval)
{
    cudaMemcpy((void *)&(gtsval_dev->x), (void *)&(tsval->val[0]),
               sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)&(gtsval_dev->y), (void *)&(tsval->val[1]),
               sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)&(gtsval_dev->z), (void *)&(tsval->val[2]),
               sizeof(float), cudaMemcpyHostToDevice);
}

/* device gts_real -> host tsfloat. */
inline void gts2ts_dev(tsfloat *tsval, gts_real *gtsval_dev)
{
    cudaMemcpy((void *)&(tsval->val[0]), (void *)&(gtsval_dev->x),
               sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)&(tsval->val[1]), (void *)&(gtsval_dev->y),
               sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)&(tsval->val[2]), (void *)&(gtsval_dev->z),
               sizeof(float), cudaMemcpyDeviceToHost);
}

/* host gts_real <- device gts_real. */
inline void gts2gts_dev(gts_real *gtsval, gts_real *gtsval_dev)
{
    cudaMemcpy((void *)&(gtsval->x), (void *)&(gtsval_dev->x),
               sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)&(gtsval->y), (void *)&(gtsval_dev->y),
               sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)&(gtsval->z), (void *)&(gtsval_dev->z),
               sizeof(float), cudaMemcpyDeviceToHost);
}

/* device gts_real <- host gts_real. */
inline void gts_dev2gts(gts_real *gtsval_dev, gts_real *gtsval)
{
    cudaMemcpy((void *)&(gtsval_dev->x), (void *)&(gtsval->x),
               sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)&(gtsval_dev->y), (void *)&(gtsval->y),
               sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)&(gtsval_dev->z), (void *)&(gtsval->z),
               sizeof(float), cudaMemcpyHostToDevice);
}


/* ============================================================
 * GTS scalar / element-wise macros
 * ============================================================ */

/* element-wise zero */
#define SET0_GTS(val)  {(val).x = 0.0; (val).y = 0.0; (val).z = 0.0;}
/* element-wise one */
#define SET1_GTS(val)  {(val).x = 1.0; (val).y = 0.0; (val).z = 0.0;}
/* element-wise negate */
#define NEG_GTS(ret, val) \
    {(ret).x = -((val).x); (ret).y = -((val).y); (ret).z = -((val).z);}

#define set0_gts(val) SET0_GTS(val)
#define set1_gts(val) SET1_GTS(val)


/* Initialize a single device gts_real to zero / one. */
inline void set0_gts_dev(gts_real *gtsval_dev)
{
    SET0_GTS(_const_gts_zero);
    cudaMemcpy((void *)&(gtsval_dev->x), (void *)&(_const_gts_zero.x),
               sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)&(gtsval_dev->y), (void *)&(_const_gts_zero.y),
               sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)&(gtsval_dev->z), (void *)&(_const_gts_zero.z),
               sizeof(float), cudaMemcpyHostToDevice);
}

inline void set1_gts_dev(gts_real *gtsval_dev)
{
    SET1_GTS(_const_gts_one);
    cudaMemcpy((void *)&(gtsval_dev->x), (void *)&(_const_gts_one.x),
               sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)&(gtsval_dev->y), (void *)&(_const_gts_one.y),
               sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)&(gtsval_dev->z), (void *)&(_const_gts_one.z),
               sizeof(float), cudaMemcpyHostToDevice);
}

#define SET0_GTS_DEV(val) set0_gts_dev(val)
#define SET1_GTS_DEV(val) set1_gts_dev(val)


/* ============================================================
 * Device storage init / free for plain gts_real values
 * ============================================================ */

gts_real *_bncuda_init_gts(void);
void      _bncuda_free_gts(gts_real *dev_val);
gts_real *_bncuda_init_gts_array(int array_num);
void      _bncuda_free_gts_array(gts_real *dev_array);


/* ============================================================
 * GTSVector / gtsvector
 * ============================================================ */

#define SET_GTSVECTOR_I(vec, index, value)    { vec->element[index] = value; }
#define SET_GTSVECTOR_I_UI(vec, index, value) { vec->element[index] = (double)value; }
#define SET_GTSVECTOR_I_D(vec, index, value)  { vec->element[index] = (double)value; }
#define SET0_GTSVECTOR_I(vec, index)          { vec->element[index] = (gts_real)0.0; }
#define GET_GTSVECTOR_I(vec, index)           (vec->element[index])

#define set_gtsvector_i(vec, index, value)    SET_GTSVECTOR_I(vec, index, value)
#define set_gtsvector_i_d(vec, index, value)  SET_GTSVECTOR_I_D(vec, index, value)
#define set_gtsvector_i_ui(vec, index, value) SET_GTSVECTOR_I_UI(vec, index, value)
#define set0_gtsvector_i(vec, index)          SET_GTSVECTOR_I(vec, index)
#define get_gtsvector_i(vec, index)           GET_GTSVECTOR_I(vec, index)

typedef struct {
    long int dim;
    gts_real *element;
} gtsvector;
typedef gtsvector *GTSVector;

/* host */
GTSVector init_gtsvector(long int dim);
void      free_gtsvector(GTSVector vec);
/* device */
GTSVector init_gtsvector_dev(long int dim);
void      free_gtsvector_dev(GTSVector vec);

/* host TSVector <-> device GTSVector */
void subst_gtsvector_dev_tsvec(GTSVector gtsvec_dev, TSVector tsvec);
void subst_tsvector_gtsvec_dev(TSVector tsvec, GTSVector gtsvec_dev);

void print_gtsvector_dev(GTSVector dev_vec);

/* device-side vector ops (launchers) */
void add_gtsvector_dev (GTSVector c_dev, GTSVector a_dev, GTSVector b_dev,
                        int num_blocks_per_grid, int num_threads_per_block);
void sub_gtsvector_dev (GTSVector c_dev, GTSVector a_dev, GTSVector b_dev,
                        int num_blocks_per_grid, int num_threads_per_block);
void cmul_gtsvector_dev(GTSVector c_dev, gts_real val, GTSVector a_dev,
                        int num_blocks_per_grid, int num_threads_per_block);
void subst_gtsvector_dev(GTSVector ret_dev, GTSVector vec_dev,
                         int num_blocks_per_grid, int num_threads_per_block);
void set0_gtsvector_dev (GTSVector ret_dev,
                         int num_blocks_per_grid, int num_threads_per_block);
void ip_gtsvector_dev   (gts_real *ret_dev, GTSVector a_dev, GTSVector b_dev,
                         int num_blocks_per_grid, int num_threads_per_block);
void norm2_gtsvector_dev(gts_real *ret_dev, GTSVector a_dev,
                         int num_blocks_per_grid, int num_threads_per_block);
void norm1_gtsvector_dev(gts_real *ret_dev, GTSVector a_dev,
                         int num_blocks_per_grid, int num_threads_per_block);
void normi_gtsvector_dev(gts_real *ret_dev, GTSVector a_dev,
                         int num_blocks_per_grid, int num_threads_per_block);


/* ============================================================
 * GTSMatrix / gtsmatrix
 * ============================================================ */

#define SET_GTSMATRIX_IJ(mat, i, j, value) \
    { mat->element[i * mat->col_dim + j] = value; }
#define SET0_GTSMATRIX_IJ(mat, i, j) \
    { mat->element[i * mat->col_dim + j] = (gts_real)0.0; }
#define GET_GTSMATRIX_IJ(mat, i, j) (mat->element[i * mat->col_dim + j])

#define set_gtsmatrix_ij(mat, i, j, value)  SET_GTSMATRIX_IJ(mat, i, j, value)
#define set0_gtsmatrix_ij(mat, i, j)        SET0_GTSMATRIX_IJ(mat, i, j)
#define get_gtsmatrix_ij(mat, i, j)         GET_GTSMATRIX_IJ(mat, i, j)

typedef struct {
    long int row_dim, col_dim;
    gts_real *element;
} gtsmatrix;
typedef gtsmatrix *GTSMatrix;

/* host */
GTSMatrix init_gtsmatrix(long int row_dim, long int col_dim);
void      free_gtsmatrix(GTSMatrix mat);
/* device */
GTSMatrix init_gtsmatrix_dev(long int row_dim, long int col_dim);
void      free_gtsmatrix_dev(GTSMatrix mat);

/* host TSMatrix <-> device GTSMatrix */
void subst_gtsmatrix_dev_tsmat(GTSMatrix gtsmat_dev, TSMatrix tsmat);
void subst_tsmatrix_gtsmat_dev(TSMatrix tsmat, GTSMatrix gtsmat_dev);

void print_gtsmatrix_dev(GTSMatrix dev_mat);

/* setters / getters that bridge tsfloat <-> device gts_real elements */
void set_gtsmatrix_ij_dev   (GTSMatrix mat_dev, long int row_index,
                             long int col_index, tsfloat val);
void set_gtsmatrix_ij_d_dev (GTSMatrix mat_dev, long int row_index,
                             long int col_index, double val);
void set0_gtsmatrix_dev     (GTSMatrix ret_dev,
                             int num_blocks_per_grid, int num_threads_per_block);

/* arithmetic on the device */
void add_gtsmatrix_dev      (GTSMatrix c_dev, GTSMatrix a_dev, GTSMatrix b_dev,
                             int num_blocks_per_grid, int num_threads_per_block);
void sub_gtsmatrix_dev      (GTSMatrix c_dev, GTSMatrix a_dev, GTSMatrix b_dev,
                             int num_blocks_per_grid, int num_threads_per_block);
void cmul_gtsmatrix_dev     (GTSMatrix c_dev, gts_real sc, GTSMatrix a_dev,
                             int num_blocks_per_grid, int num_threads_per_block);
void mul_gtsmatrix_dev      (GTSMatrix c_dev, GTSMatrix a_dev, GTSMatrix b_dev,
                             int num_blocks_per_grid, int num_threads_per_block);
void subst_gtsmatrix_dev    (GTSMatrix ret_dev, GTSMatrix mat_dev,
                             int num_blocks_per_grid, int num_threads_per_block);
void normf_gtsmatrix_dev    (gts_real *ret_dev, GTSMatrix mat_dev,
                             int num_blocks_per_grid, int num_threads_per_block);
void mul_gtsmatrix_gtsvec   (GTSVector v_dev, GTSMatrix a_dev, GTSVector b_dev,
                             int num_blocks_per_grid, int num_threads_per_block);
void mul_gtsmatrixt_gtsvec  (GTSVector v_dev, GTSMatrix a_dev, GTSVector b_dev,
                             int num_blocks_per_grid, int num_threads_per_block);

#endif /* __BNC_GTSLINEAR_H__ */
