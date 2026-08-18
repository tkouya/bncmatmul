/********************************************************************************/
/* gflinear.h: Native float precision GPU Linear Computation Library with CUDA */
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
#ifndef __BNC_GFLINEAR_H__
#define __BNC_GFLINEAR_H__

#include <cmath>
#include <cstdio>
#include <cstdlib>

// on CPU
#include "flinear.h" // FVector / FMatrix

// on GPU
#include "cuda_runtime.h" // cudaMemcpy

// definition
#ifndef MAX_NUM_THREADS_PER_BLOCK
  #define MAX_NUM_THREADS_PER_BLOCK 128
#endif
#ifndef MAX_NUM_BLOCKS_PER_GRID
  #define MAX_NUM_BLOCKS_PER_GRID 128
#endif

// GD vector (native float, stored on CPU or GPU)
typedef struct
{
	long int dim;
	float *element;
} gfvector;

typedef gfvector *GFVector;

// GD matrix (row-major, packed: stride = col_dim)
typedef struct
{
	long int row_dim, col_dim;
	float *element;
} gfmatrix;

typedef gfmatrix *GFMatrix;

#define SET_GDVECTOR_I(vec, index, value)   { (vec)->element[index] = (value); }
#define GET_GDVECTOR_I(vec, index)          ((vec)->element[index])
#define SET_GDMATRIX_IJ(mat, i, j, value)   { (mat)->element[(i) * (mat)->col_dim + (j)] = (value); }
#define GET_GDMATRIX_IJ(mat, i, j)          ((mat)->element[(i) * (mat)->col_dim + (j)])

#define set_gfvector_i(vec, index, value)   SET_GDVECTOR_I(vec, index, value)
#define get_gfvector_i(vec, index)          GET_GDVECTOR_I(vec, index)
#define set_gfmatrix_ij(mat, i, j, value)   SET_GDMATRIX_IJ(mat, i, j, value)
#define get_gfmatrix_ij(mat, i, j)          GET_GDMATRIX_IJ(mat, i, j)

/*-------------------------------------------------------------------*/
/* GFVector                                                          */
/*-------------------------------------------------------------------*/
// initialize gfvector on CPU(host)
GFVector init_gfvector(long int dim);

// free gfvector on CPU(HOST)
void free_gfvector(GFVector vec);

// initialize gfvector on GPU
GFVector init_gfvector_dev(long int dim);

// free gfvector on GPU
void free_gfvector_dev(GFVector vec);

// gfvec on GPU := fvec (host FVector -> device)
void subst_gfvector_dev_fvec(GFVector gfvec_dev, FVector fvec);

// fvec := gfvec_dev on GPU (device -> host FVector)
void subst_fvector_gfvec_dev(FVector fvec, GFVector gfvec_dev);

// print gfvector on GPU
void print_gfvector_dev(GFVector dev_vec);

/* c = a + b */
void add_gfvector_dev(GFVector c_dev, GFVector a_dev, GFVector b_dev, int num_blocks_per_grid, int num_threads_per_block);
/* c = a - b */
void sub_gfvector_dev(GFVector c_dev, GFVector a_dev, GFVector b_dev, int num_blocks_per_grid, int num_threads_per_block);
/* c = val * a */
void cmul_gfvector_dev(GFVector c_dev, float val, GFVector a_dev, int num_blocks_per_grid, int num_threads_per_block);
/* c := vec */
void subst_gfvector_dev(GFVector ret_dev, GFVector vec_dev, int num_blocks_per_grid, int num_threads_per_block);
/* c := 0 */
void set0_gfvector_dev(GFVector ret_dev, int num_blocks_per_grid, int num_threads_per_block);
/* (a, b) */
void ip_gfvector_dev(float *ret_dev, GFVector a_dev, GFVector b_dev, int num_blocks_per_grid, int num_threads_per_block);
// norm2 / norm1 / norm_inf
void norm2_gfvector_dev(float *ret_dev, GFVector a_dev, int num_blocks_per_grid, int num_threads_per_block);
void norm1_gfvector_dev(float *ret_dev, GFVector a_dev, int num_blocks_per_grid, int num_threads_per_block);
void normi_gfvector_dev(float *ret_dev, GFVector a_dev, int num_blocks_per_grid, int num_threads_per_block);

/*-------------------------------------------------------------------*/
/* GFMatrix                                                          */
/*-------------------------------------------------------------------*/
// initialize gfmatrix on CPU(host)
GFMatrix init_gfmatrix(long int row_dim, long int col_dim);
// free gfmatrix on CPU(HOST)
void free_gfmatrix(GFMatrix mat);
// initialize gfmatrix on GPU
GFMatrix init_gfmatrix_dev(long int row_dim, long int col_dim);
// free gfmatrix on GPU
void free_gfmatrix_dev(GFMatrix mat);
// print gfmatrix on GPU
void print_gfmatrix_dev(GFMatrix mat);

// matrix multiplication ret := A * B
void mul_gfmatrix_dev(GFMatrix ret_dev, GFMatrix a_dev, GFMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block);

// gfmat on GPU := fmat (host FMatrix -> device)
void subst_gfmatrix_dev_fmat(GFMatrix gfmat_dev, FMatrix fmat);
// fmat := gfmat_dev on GPU (device -> host FMatrix)
void subst_fmatrix_gfmat_dev(FMatrix fmat, GFMatrix gfmat_dev);

// Frobenius norm
void normf_gfmatrix_dev(float *ret_dev, GFMatrix mat_dev, int num_blocks_per_grid, int num_threads_per_block);

/* c := a + b */
void add_gfmatrix_dev(GFMatrix c_dev, GFMatrix a_dev, GFMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block);
/* c := a - b */
void sub_gfmatrix_dev(GFMatrix c_dev, GFMatrix a_dev, GFMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block);
/* c := sc * a */
void cmul_gfmatrix_dev(GFMatrix c_dev, float sc, GFMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block);
/* c = a^T */
void transpose_gfmatrix_dev(GFMatrix c_dev, GFMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block);
/* c := a */
void subst_gfmatrix_dev(GFMatrix c_dev, GFMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block);
/* c := 0 */
void set0_gfmatrix_dev(GFMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block);
/* set (i, j)-element on GPU */
void set_gfmatrix_ij_dev(GFMatrix mat_dev, long int row_index, long int col_index, float val);
/* c := I */
void setI_gfmatrix_dev(GFMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block);
/* v := a * vb */
void mul_gfmatrix_gfvec(GFVector v, GFMatrix a, GFVector vb, int num_blocks_per_grid, int num_threads_per_block);
/* v := a^T * vb */
void mul_gfmatrixt_gfvec(GFVector v, GFMatrix a, GFVector vb, int num_blocks_per_grid, int num_threads_per_block);

#endif // __BNC_GFLINEAR_H__
