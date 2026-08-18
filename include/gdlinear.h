/********************************************************************************/
/* gdlinear.h: Native double precision GPU Linear Computation Library with CUDA */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/*                                                                              */
/* This program is free software: you can redistribute it and/or modify it      */
/* under the terms of the GNU Lesser General Public License as published by the */
/* Free Software Foundation, either version 3 of the License or any later       */
/* version.                                                                     */
/*                                                                              */
/* This program is distributed in the hope that it will be useful, but WITHOUT  */
/* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        */
/* FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License */
/* for more details.                                                            */
/********************************************************************************/
#ifndef __BNC_GDLINEAR_H__
#define __BNC_GDLINEAR_H__

#include <cmath>
#include <cstdio>
#include <cstdlib>

// on CPU
#include "dlinear.h" // DVector / DMatrix

// on GPU
#include "cuda_runtime.h" // cudaMemcpy

// definition
#ifndef MAX_NUM_THREADS_PER_BLOCK
  #define MAX_NUM_THREADS_PER_BLOCK 128
#endif
#ifndef MAX_NUM_BLOCKS_PER_GRID
  #define MAX_NUM_BLOCKS_PER_GRID 128
#endif

// GD vector (native double, stored on CPU or GPU)
typedef struct
{
	long int dim;
	double *element;
} gdvector;

typedef gdvector *GDVector;

// GD matrix (row-major, packed: stride = col_dim)
typedef struct
{
	long int row_dim, col_dim;
	double *element;
} gdmatrix;

typedef gdmatrix *GDMatrix;

#define SET_GDVECTOR_I(vec, index, value)   { (vec)->element[index] = (value); }
#define GET_GDVECTOR_I(vec, index)          ((vec)->element[index])
#define SET_GDMATRIX_IJ(mat, i, j, value)   { (mat)->element[(i) * (mat)->col_dim + (j)] = (value); }
#define GET_GDMATRIX_IJ(mat, i, j)          ((mat)->element[(i) * (mat)->col_dim + (j)])

#define set_gdvector_i(vec, index, value)   SET_GDVECTOR_I(vec, index, value)
#define get_gdvector_i(vec, index)          GET_GDVECTOR_I(vec, index)
#define set_gdmatrix_ij(mat, i, j, value)   SET_GDMATRIX_IJ(mat, i, j, value)
#define get_gdmatrix_ij(mat, i, j)          GET_GDMATRIX_IJ(mat, i, j)

/*-------------------------------------------------------------------*/
/* GDVector                                                          */
/*-------------------------------------------------------------------*/
// initialize gdvector on CPU(host)
GDVector init_gdvector(long int dim);

// free gdvector on CPU(HOST)
void free_gdvector(GDVector vec);

// initialize gdvector on GPU
GDVector init_gdvector_dev(long int dim);

// free gdvector on GPU
void free_gdvector_dev(GDVector vec);

// gdvec on GPU := dvec (host DVector -> device)
void subst_gdvector_dev_dvec(GDVector gdvec_dev, DVector dvec);

// dvec := gdvec_dev on GPU (device -> host DVector)
void subst_dvector_gdvec_dev(DVector dvec, GDVector gdvec_dev);

// print gdvector on GPU
void print_gdvector_dev(GDVector dev_vec);

/* c = a + b */
void add_gdvector_dev(GDVector c_dev, GDVector a_dev, GDVector b_dev, int num_blocks_per_grid, int num_threads_per_block);
/* c = a - b */
void sub_gdvector_dev(GDVector c_dev, GDVector a_dev, GDVector b_dev, int num_blocks_per_grid, int num_threads_per_block);
/* c = val * a */
void cmul_gdvector_dev(GDVector c_dev, double val, GDVector a_dev, int num_blocks_per_grid, int num_threads_per_block);
/* c := vec */
void subst_gdvector_dev(GDVector ret_dev, GDVector vec_dev, int num_blocks_per_grid, int num_threads_per_block);
/* c := 0 */
void set0_gdvector_dev(GDVector ret_dev, int num_blocks_per_grid, int num_threads_per_block);
/* (a, b) */
void ip_gdvector_dev(double *ret_dev, GDVector a_dev, GDVector b_dev, int num_blocks_per_grid, int num_threads_per_block);
// norm2 / norm1 / norm_inf
void norm2_gdvector_dev(double *ret_dev, GDVector a_dev, int num_blocks_per_grid, int num_threads_per_block);
void norm1_gdvector_dev(double *ret_dev, GDVector a_dev, int num_blocks_per_grid, int num_threads_per_block);
void normi_gdvector_dev(double *ret_dev, GDVector a_dev, int num_blocks_per_grid, int num_threads_per_block);

/*-------------------------------------------------------------------*/
/* GDMatrix                                                          */
/*-------------------------------------------------------------------*/
// initialize gdmatrix on CPU(host)
GDMatrix init_gdmatrix(long int row_dim, long int col_dim);
// free gdmatrix on CPU(HOST)
void free_gdmatrix(GDMatrix mat);
// initialize gdmatrix on GPU
GDMatrix init_gdmatrix_dev(long int row_dim, long int col_dim);
// free gdmatrix on GPU
void free_gdmatrix_dev(GDMatrix mat);
// print gdmatrix on GPU
void print_gdmatrix_dev(GDMatrix mat);

// matrix multiplication ret := A * B
void mul_gdmatrix_dev(GDMatrix ret_dev, GDMatrix a_dev, GDMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block);

// gdmat on GPU := dmat (host DMatrix -> device)
void subst_gdmatrix_dev_dmat(GDMatrix gdmat_dev, DMatrix dmat);
// dmat := gdmat_dev on GPU (device -> host DMatrix)
void subst_dmatrix_gdmat_dev(DMatrix dmat, GDMatrix gdmat_dev);

// Frobenius norm
void normf_gdmatrix_dev(double *ret_dev, GDMatrix mat_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := a + b */
void add_gdmatrix_dev(GDMatrix c_dev, GDMatrix a_dev, GDMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block);
/* c := a - b */
void sub_gdmatrix_dev(GDMatrix c_dev, GDMatrix a_dev, GDMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block);
/* c := sc * a */
void cmul_gdmatrix_dev(GDMatrix c_dev, double sc, GDMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block);
/* c = a^T */
void transpose_gdmatrix_dev(GDMatrix c_dev, GDMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block);
/* c := a */
void subst_gdmatrix_dev(GDMatrix c_dev, GDMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block);
/* c := 0 */
void set0_gdmatrix_dev(GDMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block);
/* set (i, j)-element on GPU */
void set_gdmatrix_ij_dev(GDMatrix mat_dev, long int row_index, long int col_index, double val);
/* c := I */
void setI_gdmatrix_dev(GDMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block);
/* v := a * vb */
void mul_gdmatrix_gdvec(GDVector v, GDMatrix a, GDVector vb, int num_blocks_per_grid, int num_threads_per_block);
/* v := a^T * vb */
void mul_gdmatrixt_gdvec(GDVector v, GDMatrix a, GDVector vb, int num_blocks_per_grid, int num_threads_per_block);

#endif // __BNC_GDLINEAR_H__
