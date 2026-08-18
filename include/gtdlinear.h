/********************************************************************************/
/* gtdlinear.h: Triple-double precision Linear Computation Library (GPU + CUDA) */
/* Copyright (C) 2015-2026 Tomonori Kouya                                       */
/*                                                                              */
/* GTD (triple-double, 3 IEEE doubles, ~47 decimal digits) GPU types and        */
/* prototypes.  Mirrors the structure of gddlinear.h but contains only the      */
/* GTD-specific additions.  Shared CPU / GDD / GQD infrastructure is pulled     */
/* in via #include "gddlinear.h" so that types like GQDMatrix and helpers like  */
/* qd2gqd_dev are not duplicated.                                               */
/*                                                                              */
/* This program is free software: you can redistribute it and/or modify it      */
/* under the terms of the GNU Lesser General Public License as published by the */
/* Free Software Foundation, either version 3 of the License or any later       */
/* version.                                                                     */
/********************************************************************************/
#ifndef __BNC_GTDLINEAR_H__
  #define __BNC_GTDLINEAR_H__

#include <cmath>
#include <cstdio>
#include <cstdlib>

/* CPU side */
#include "qd/qd_real.h"

/* bncmatmul's c_dd_qd.h defines `#define nint(a) (round(a))` which,
 * if it has leaked into our scope (via a prior gddlinear.h include for
 * example), would mangle qd::nint() inside qd/td_inline.h.  Undef it
 * defensively before pulling td_real.h. */
#ifdef nint
#  undef nint
#endif
#include "qd/td_real.h"   /* dtq-0.0.2: triple-double td_real type     */

#include "tdlinear.h"     /* TDMatrix / TDVector (SoA host containers) */

/* GPU side: types and inline functions */
#include "cuda_runtime.h"
#include "gqd_type.h"     /* gtd_real (typedef of double3), gqd_real, ... */

/* Pull in the rest of the GDD / GQD / CPU shared infrastructure.  Doing
 * this BEFORE the GTD definitions below ensures that GDDMatrix and
 * GQDMatrix are visible for callers that include only gtdlinear.h, and
 * also avoids re-declaring all the gqd_* helpers / typedefs that were
 * already provided by gddlinear.h. */
#include "gddlinear.h"

/* ============================================================
 * GTD constants
 * ============================================================ */
static gtd_real _const_gtd_zero = {0.0, 0.0, 0.0};
static gtd_real _const_gtd_one  = {1.0, 0.0, 0.0};


/* ============================================================
 * GTD <-> td_real conversion helpers (host)
 * ============================================================ */

/* gtd_real (GPU) -> td_real (host) by value. */
inline td_real gtd_get_td(gtd_real gval)
{
    return td_real(gval.x, gval.y, gval.z);
}

/* tdval := gtdval (both host pointers). */
inline void gtd2td(td_real *tdval, gtd_real *gtdval)
{
    tdval->x[0] = gtdval->x;
    tdval->x[1] = gtdval->y;
    tdval->x[2] = gtdval->z;
}

/* gtdval := tdval (both host pointers). */
inline void td2gtd(gtd_real *gtdval, td_real *tdval)
{
    gtdval->x = tdval->x[0];
    gtdval->y = tdval->x[1];
    gtdval->z = tdval->x[2];
}


/* ============================================================
 * GTD <-> td_real conversion helpers (host <-> device)
 *
 * Naming follows gddlinear.h convention: <src>2<dst>_dev means the
 * destination lives on the device (or "_dev2" prefix for the source).
 * ============================================================ */

/* host td_real -> device gtd_real (3 small cudaMemcpy calls). */
inline void td2gtd_dev(gtd_real *gtdval_dev, td_real *tdval)
{
    cudaMemcpy((void *)&(gtdval_dev->x), (void *)&(tdval->x[0]),
               sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)&(gtdval_dev->y), (void *)&(tdval->x[1]),
               sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)&(gtdval_dev->z), (void *)&(tdval->x[2]),
               sizeof(double), cudaMemcpyHostToDevice);
}

/* device gtd_real -> host td_real. */
inline void gtd2td_dev(td_real *tdval, gtd_real *gtdval_dev)
{
    cudaMemcpy((void *)&(tdval->x[0]), (void *)&(gtdval_dev->x),
               sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)&(tdval->x[1]), (void *)&(gtdval_dev->y),
               sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)&(tdval->x[2]), (void *)&(gtdval_dev->z),
               sizeof(double), cudaMemcpyDeviceToHost);
}

/* host gtd_real <- device gtd_real. */
inline void gtd2gtd_dev(gtd_real *gtdval, gtd_real *gtdval_dev)
{
    cudaMemcpy((void *)&(gtdval->x), (void *)&(gtdval_dev->x),
               sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)&(gtdval->y), (void *)&(gtdval_dev->y),
               sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy((void *)&(gtdval->z), (void *)&(gtdval_dev->z),
               sizeof(double), cudaMemcpyDeviceToHost);
}

/* device gtd_real <- host gtd_real. */
inline void gtd_dev2gtd(gtd_real *gtdval_dev, gtd_real *gtdval)
{
    cudaMemcpy((void *)&(gtdval_dev->x), (void *)&(gtdval->x),
               sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)&(gtdval_dev->y), (void *)&(gtdval->y),
               sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)&(gtdval_dev->z), (void *)&(gtdval->z),
               sizeof(double), cudaMemcpyHostToDevice);
}


/* ============================================================
 * GTD scalar / element-wise macros
 * ============================================================ */

/* element-wise zero */
#define SET0_GTD(val)  {(val).x = 0.0; (val).y = 0.0; (val).z = 0.0;}
/* element-wise one */
#define SET1_GTD(val)  {(val).x = 1.0; (val).y = 0.0; (val).z = 0.0;}
/* element-wise negate */
#define NEG_GTD(ret, val) \
    {(ret).x = -((val).x); (ret).y = -((val).y); (ret).z = -((val).z);}

#define set0_gtd(val) SET0_GTD(val)
#define set1_gtd(val) SET1_GTD(val)


/* Initialize a single device gtd_real to zero / one. */
inline void set0_gtd_dev(gtd_real *gtdval_dev)
{
    SET0_GTD(_const_gtd_zero);
    cudaMemcpy((void *)&(gtdval_dev->x), (void *)&(_const_gtd_zero.x),
               sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)&(gtdval_dev->y), (void *)&(_const_gtd_zero.y),
               sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)&(gtdval_dev->z), (void *)&(_const_gtd_zero.z),
               sizeof(double), cudaMemcpyHostToDevice);
}

inline void set1_gtd_dev(gtd_real *gtdval_dev)
{
    SET1_GTD(_const_gtd_one);
    cudaMemcpy((void *)&(gtdval_dev->x), (void *)&(_const_gtd_one.x),
               sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)&(gtdval_dev->y), (void *)&(_const_gtd_one.y),
               sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy((void *)&(gtdval_dev->z), (void *)&(_const_gtd_one.z),
               sizeof(double), cudaMemcpyHostToDevice);
}

#define SET0_GTD_DEV(val) set0_gtd_dev(val)
#define SET1_GTD_DEV(val) set1_gtd_dev(val)


/* ============================================================
 * Device storage init / free for plain gtd_real values
 * ============================================================ */

gtd_real *_bncuda_init_gtd(void);
void      _bncuda_free_gtd(gtd_real *dev_val);
gtd_real *_bncuda_init_gtd_array(int array_num);
void      _bncuda_free_gtd_array(gtd_real *dev_array);


/* ============================================================
 * GTDVector / gtdvector
 * ============================================================ */

#define SET_GTDVECTOR_I(vec, index, value)    { vec->element[index] = value; }
#define SET_GTDVECTOR_I_UI(vec, index, value) { vec->element[index] = (double)value; }
#define SET_GTDVECTOR_I_D(vec, index, value)  { vec->element[index] = (double)value; }
#define SET0_GTDVECTOR_I(vec, index)          { vec->element[index] = (gtd_real)0.0; }
#define GET_GTDVECTOR_I(vec, index)           (vec->element[index])

#define set_gtdvector_i(vec, index, value)    SET_GTDVECTOR_I(vec, index, value)
#define set_gtdvector_i_d(vec, index, value)  SET_GTDVECTOR_I_D(vec, index, value)
#define set_gtdvector_i_ui(vec, index, value) SET_GTDVECTOR_I_UI(vec, index, value)
#define set0_gtdvector_i(vec, index)          SET_GTDVECTOR_I(vec, index)
#define get_gtdvector_i(vec, index)           GET_GTDVECTOR_I(vec, index)

typedef struct {
    long int dim;
    gtd_real *element;
} gtdvector;
typedef gtdvector *GTDVector;

/* host */
GTDVector init_gtdvector(long int dim);
void      free_gtdvector(GTDVector vec);
/* device */
GTDVector init_gtdvector_dev(long int dim);
void      free_gtdvector_dev(GTDVector vec);

/* host TDVector <-> device GTDVector */
void subst_gtdvector_dev_tdvec(GTDVector gtdvec_dev, TDVector tdvec);
void subst_tdvector_gtdvec_dev(TDVector tdvec, GTDVector gtdvec_dev);

void print_gtdvector_dev(GTDVector dev_vec);

/* device-side vector ops (launchers) */
void add_gtdvector_dev (GTDVector c_dev, GTDVector a_dev, GTDVector b_dev,
                        int num_blocks_per_grid, int num_threads_per_block);
void sub_gtdvector_dev (GTDVector c_dev, GTDVector a_dev, GTDVector b_dev,
                        int num_blocks_per_grid, int num_threads_per_block);
void cmul_gtdvector_dev(GTDVector c_dev, gtd_real val, GTDVector a_dev,
                        int num_blocks_per_grid, int num_threads_per_block);
void subst_gtdvector_dev(GTDVector ret_dev, GTDVector vec_dev,
                         int num_blocks_per_grid, int num_threads_per_block);
void set0_gtdvector_dev (GTDVector ret_dev,
                         int num_blocks_per_grid, int num_threads_per_block);
void ip_gtdvector_dev   (gtd_real *ret_dev, GTDVector a_dev, GTDVector b_dev,
                         int num_blocks_per_grid, int num_threads_per_block);
void norm2_gtdvector_dev(gtd_real *ret_dev, GTDVector a_dev,
                         int num_blocks_per_grid, int num_threads_per_block);
void norm1_gtdvector_dev(gtd_real *ret_dev, GTDVector a_dev,
                         int num_blocks_per_grid, int num_threads_per_block);
void normi_gtdvector_dev(gtd_real *ret_dev, GTDVector a_dev,
                         int num_blocks_per_grid, int num_threads_per_block);


/* ============================================================
 * GTDMatrix / gtdmatrix
 * ============================================================ */

#define SET_GTDMATRIX_IJ(mat, i, j, value) \
    { mat->element[i * mat->col_dim + j] = value; }
#define SET0_GTDMATRIX_IJ(mat, i, j) \
    { mat->element[i * mat->col_dim + j] = (gtd_real)0.0; }
#define GET_GTDMATRIX_IJ(mat, i, j) (mat->element[i * mat->col_dim + j])

#define set_gtdmatrix_ij(mat, i, j, value)  SET_GTDMATRIX_IJ(mat, i, j, value)
#define set0_gtdmatrix_ij(mat, i, j)        SET0_GTDMATRIX_IJ(mat, i, j)
#define get_gtdmatrix_ij(mat, i, j)         GET_GTDMATRIX_IJ(mat, i, j)

typedef struct {
    long int row_dim, col_dim;
    gtd_real *element;
} gtdmatrix;
typedef gtdmatrix *GTDMatrix;

/* host */
GTDMatrix init_gtdmatrix(long int row_dim, long int col_dim);
void      free_gtdmatrix(GTDMatrix mat);
/* device */
GTDMatrix init_gtdmatrix_dev(long int row_dim, long int col_dim);
void      free_gtdmatrix_dev(GTDMatrix mat);

/* host TDMatrix <-> device GTDMatrix */
void subst_gtdmatrix_dev_tdmat(GTDMatrix gtdmat_dev, TDMatrix tdmat);
void subst_tdmatrix_gtdmat_dev(TDMatrix tdmat, GTDMatrix gtdmat_dev);

void print_gtdmatrix_dev(GTDMatrix dev_mat);

/* setters / getters that bridge td_real <-> device gtd_real elements */
void set_gtdmatrix_ij_dev   (GTDMatrix mat_dev, long int row_index,
                             long int col_index, td_real val);
void set_gtdmatrix_ij_d_dev (GTDMatrix mat_dev, long int row_index,
                             long int col_index, double val);
void set0_gtdmatrix_dev     (GTDMatrix ret_dev,
                             int num_blocks_per_grid, int num_threads_per_block);

/* arithmetic on the device */
void add_gtdmatrix_dev      (GTDMatrix c_dev, GTDMatrix a_dev, GTDMatrix b_dev,
                             int num_blocks_per_grid, int num_threads_per_block);
void sub_gtdmatrix_dev      (GTDMatrix c_dev, GTDMatrix a_dev, GTDMatrix b_dev,
                             int num_blocks_per_grid, int num_threads_per_block);
void cmul_gtdmatrix_dev     (GTDMatrix c_dev, gtd_real sc, GTDMatrix a_dev,
                             int num_blocks_per_grid, int num_threads_per_block);
void mul_gtdmatrix_dev      (GTDMatrix c_dev, GTDMatrix a_dev, GTDMatrix b_dev,
                             int num_blocks_per_grid, int num_threads_per_block);
void subst_gtdmatrix_dev    (GTDMatrix ret_dev, GTDMatrix mat_dev,
                             int num_blocks_per_grid, int num_threads_per_block);
void normf_gtdmatrix_dev    (gtd_real *ret_dev, GTDMatrix mat_dev,
                             int num_blocks_per_grid, int num_threads_per_block);
void mul_gtdmatrix_gtdvec   (GTDVector v_dev, GTDMatrix a_dev, GTDVector b_dev,
                             int num_blocks_per_grid, int num_threads_per_block);
void mul_gtdmatrixt_gtdvec  (GTDVector v_dev, GTDMatrix a_dev, GTDVector b_dev,
                             int num_blocks_per_grid, int num_threads_per_block);

#endif /* __BNC_GTDLINEAR_H__ */
