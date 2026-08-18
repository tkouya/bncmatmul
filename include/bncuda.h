/********************************************************************************/
/* bncuda.h: BNCpack with CUDA                                                  */
/* Copyright (c) 2015 Tomonori Kouya                                            */
/*                                                                              */
/* Version 0.1, 2015-06-24: First Preparation for bncmatmul library             */
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
/*                                                                              */
/* You should have received a copy of the GNU Lesser General Public License     */
/* along with this program.  If not, see <http://www.gnu.org/licenses/>.        */
/*                                                                              */
/********************************************************************************/
// header of bncuda_linear.c
#include "bnc.h"
#include "flinear.h"  // FMatrix, FVector used in prototypes below
#include "dlinear.h"  // DMatrix, DVector used in prototypes below

// for CUDDA & cublas
#include "cuda_runtime.h"
#include "cublas_v2.h"

// define _BNCUDA_H
#ifndef _BNCUDA_H
  #define _BNCUDA_H

/* constant cache on GPU */
#ifdef __CUDACC__
extern __device__ float *_bncuda_const_f_zero; // = 0;
extern __device__ float *_bncuda_const_f_one; // = 1
extern __device__ float *_bncuda_const_f_minus_one; // = -1
extern __device__ double *_bncuda_const_d_one; // = 1
extern __device__ double *_bncuda_const_d_zero; // = 0;
extern __device__ double *_bncuda_const_d_minus_one; // = -1

extern __device__ float _bncuda_const_f_array[3];// = {0.0f, 1.0f, -1.0f}; // on HOST
extern __device__ double _bncuda_const_d_array[3]; // = {0.0, 1.0, 1.0}; // on HOST
#endif // __CUDACC__

#if defined (__cplusplus)
extern "C" {
#endif

/* elementary functions on GPU */
int * _bncuda_init_i(void);
void _bncuda_free_i(int *dev_val);
long int * _bncuda_init_l(void);
void _bncuda_free_l(long int *dev_val);
double * _bncuda_init_d(void);
void _bncuda_free_d(double *);
double * _bncuda_init_set_d(double);
long int _bncuda_get_l(long int *dev_val);
void _bncuda_set_l(long int *dev_val, long int val);
int _bncuda_get_i(int *dev_val);
void _bncuda_set_i(int *dev_val, int val);

// init_[d,l]_array on GPU
double * _bncuda_init_d_array(int array_num);
long int * _bncuda_init_l_array(int array_num);

// free_[d,l]_array on GPU
void _bncuda_free_d_array(double *dev_array);
void _bncuda_free_l_array(long int *dev_array);


float _bncuda_get_f(float *);
double _bncuda_get_d(double *);
void _bncuda_set_f(float *, float);
void _bncuda_set_d(double *, double);

// copy a long int array from device
void _bncuda_get_l_array(long int *array, long int *dev_array, long int array_num);

// copy a long int array to device
void _bncuda_set_l_array(long int *dev_array, long int *array, long int array_num);


void _bncuda_add_d(cublasHandle_t handle, double *ret, double *a, double *b);
void _bncuda_add2_d(cublasHandle_t handle, double *ret, double *a);
void _bncuda_sub_d(cublasHandle_t handle, double *ret, double *a, double *b);
void _bncuda_mul_d(cublasHandle_t handle, double *ret, double *a, double *b);
//void _bncuda_div_d(cublasHandle_t handle, double *ret, double *a, double *b);
void _bncuda_fma_d(cublasHandle_t handle, double *ret, double *deb_a, double *b, double *c);

cublasHandle_t *_init_bncuda(cublasHandle_t *cublas_handle);
void _clear_bncuda(cublasHandle_t cublas_handle);
void _bncuda_set_dmatrix(DMatrix dev_mat, DMatrix host_mat);
void _bncuda_set_dvector(DVector dev_vec, DVector host_vec);
void _bncuda_get_dmatrix(DMatrix host_mat, DMatrix dev_mat);
void _bncuda_get_dvector(DVector host_vec, DVector dev_vec);
void _bncuda_set_fmatrix(FMatrix dev_mat, FMatrix host_mat);
void _bncuda_set_fvector(FVector dev_vec, FVector host_vec);
void _bncuda_get_fmatrix(FMatrix host_mat, FMatrix dev_mat);
void _bncuda_get_fvector(FVector host_vec, FVector dev_vec);

DVector _bncuda_init_dvector(long int dimension);
void _bncuda_free_dvector(DVector dev_vec);
DVector _bncuda_init_set_dvector(DVector host_vec);
FVector _bncuda_init_fvector(long int dimension);
void _bncuda_free_fvector(FVector dev_vec);
FVector _bncuda_init_set_fvector(FVector host_vec);
__device__ __host__ float _bncuda_get_fvector_i(FVector dev_vec, long int index);
__device__ float * _bncuda_getp_fvector_i(FVector dev_vec, long int index);
__device__ __host__ double _bncuda_get_dvector_i(DVector dev_vec, long int index);
__device__ double * _bncuda_getp_dvector_i(DVector dev_vec, long int index);
__device__ __host__ void _bncuda_set_fvector_i(FVector dev_vec, long int index, float val);
__device__ __host__ void _bncuda_set_dvector_i(DVector dev_vec, long int index, double val);
void _bncuda_print_fvector(FVector dev_fv);
void _bncuda_print_dvector(DVector dev_dv);

FMatrix _bncuda_init_fmatrix(long int row_dimension, long int col_dimension);
DMatrix _bncuda_init_dmatrix(long int row_dimension, long int col_dimension);
void _bncuda_free_fmatrix(FMatrix dev_mat);
void _bncuda_free_dmatrix(DMatrix dev_mat);
FMatrix _bncuda_init_set_fmatrix(FMatrix host_mat);
DMatrix _bncuda_init_set_dmatrix(DMatrix host_mat);
__device__ __host__ float _bncuda_get_fmatrix_ij(FMatrix dev_mat, long int row_index, long int col_index);
__device__ __host__ double _bncuda_get_dmatrix_ij(DMatrix dev_mat, long int row_index, long int col_index);
__device__ double * _bncuda_getp_dmatrix_ij(DMatrix dev_mat, long int row_index, long int col_index);
__device__ __host__ void _bncuda_set_fmatrix_ij(FMatrix dev_mat, long int row_index, long int col_index, float val);
__device__ __host__ void _bncuda_set_dmatrix_ij(DMatrix dev_mat, long int row_index, long int col_index, double val);
void _bncuda_print_fmatrix(FMatrix dev_mat);
void _bncuda_print_dmatrix(DMatrix dev_mat);

/* float */
__device__ void _bncuda_add_fvector(cublasHandle_t handle, FVector dev_c, FVector dev_a, FVector dev_b);
__device__ void _bncuda_add2_fvector(cublasHandle_t handle, FVector dev_c, FVector dev_a);
__device__ void _bncuda_sub_fvector(cublasHandle_t handle, FVector dev_c, FVector dev_a, FVector dev_b);
__device__ void _bncuda_sub2_fvector(cublasHandle_t handle, FVector dev_c, FVector dev_a);
__device__ void _bncuda_cmul_fvector(cublasHandle_t handle, FVector dev_c, float *val, FVector dev_a);
__device__ void _bncuda_cpmul_dvector(cublasHandle_t handle, DVector dev_c, double *val, DVector dev_a);
__device__ void _bncuda_cmul2_fvector(cublasHandle_t handle, FVector dev_c, float *val);
__device__ void _bncuda_cpmul2_dvector(cublasHandle_t handle, DVector dev_c, double *val);
__device__ void _bncuda_add_cmul_fvector(cublasHandle_t handle, FVector dev_c, FVector dev_a, float *val, FVector dev_b);
__device__ void _bncuda_ip_fvector(cublasHandle_t handle, float *ret, FVector dev_a, FVector dev_b);
__device__ void _bncuda_norm1_fvector(cublasHandle_t handle, float *ret, FVector dev_a);
__device__ void _bncuda_norm2_fvector(cublasHandle_t handle, float *ret, FVector dev_a);
__device__ void _bncuda_normi_fvector(cublasHandle_t handle, float *ret, FVector dev_a);
__device__ void _bncuda_subst_fvector(cublasHandle_t handle, FVector dev_c, FVector dev_a);
__device__ void _bncuda_set0_fvector(cublasHandle_t handle, FVector dev_c);

__device__ void _bncuda_add_fmatrix(cublasHandle_t handle, FMatrix dev_c, FMatrix dev_a, FMatrix dev_b);
__device__ void _bncuda_cmul_fmatrix(cublasHandle_t handle, FMatrix dev_c, float *sc, FMatrix dev_a);
__device__ void _bncuda_mul_fmatrix(cublasHandle_t handle, FMatrix dev_c, FMatrix dev_a, FMatrix dev_b);
__device__ void _bncuda_transpose_fmatrix(cublasHandle_t handle, FMatrix dev_c, FMatrix dev_a);
__device__ void _bncuda_subst_fmatrix(cublasHandle_t handle, FMatrix dev_c, FMatrix dev_a);
__device__ void _bncuda_set0_fmatrix(cublasHandle_t handle, FMatrix dev_c);
__device__ void _bncuda_setI_fmatrix(cublasHandle_t handle, FMatrix dev_c);
__device__ void _bncuda_mul_fmatrix_fvec(cublasHandle_t handle, FVector dev_v, FMatrix dev_a, FVector dev_vb);
__device__ void _bncuda_inv_fmatrix(cublasHandle_t handle, FMatrix dev_a);

/* double */
__device__ void _bncuda_add_dvector(cublasHandle_t handle, DVector dev_c, DVector dev_a, DVector dev_b);
__device__ void _bncuda_add2_dvector(cublasHandle_t handle, DVector dev_c, DVector dev_a);
__device__ void _bncuda_sub_dvector(cublasHandle_t handle, DVector dev_c, DVector dev_a, DVector dev_b);
__device__ void _bncuda_sub2_dvector(cublasHandle_t handle, DVector dev_c, DVector dev_a);
__device__ void _bncuda_cmul_dvector(cublasHandle_t handle, DVector dev_c, double *val, DVector dev_a);
__device__ void _bncuda_cmul2_dvector(cublasHandle_t handle, DVector dev_c, double *val);
__device__ void _bncuda_add_cmul_dvector(cublasHandle_t handle, DVector dev_c, DVector dev_a, double *val, DVector dev_b);
__device__ void _bncuda_add_cmul2_dvector(cublasHandle_t handle, DVector dev_c, double *val, DVector dev_a);
__device__ void _bncuda_ip_dvector(cublasHandle_t handle, double *ret, DVector dev_a, DVector dev_b);
__device__ void _bncuda_norm1_dvector(cublasHandle_t handle, double *ret, DVector dev_a);
__device__ void _bncuda_norm2_dvector(cublasHandle_t handle, double *ret, DVector dev_a);
__device__ void _bncuda_normi_dvector(cublasHandle_t handle, double *ret, DVector dev_a);
__device__ void _bncuda_subst_dvector(cublasHandle_t handle, DVector dev_c, DVector dev_a);
__device__ void _bncuda_set0_dvector(cublasHandle_t handle, DVector dev_c);
__device__ void _bncuda_copy_dvector_ij(cublasHandle_t handle, DVector ret, long int index_start, long int index_end, DVector src, long int src_index_start, long int src_index_end);

__device__ void _bncuda_normf_dmatrix(cublasHandle_t handle, double *ret, DMatrix dev_mat);
__device__ void _bncuda_add_dmatrix(cublasHandle_t handle, DMatrix dev_c, DMatrix dev_a, DMatrix dev_b);
__device__ void _bncuda_sub_dmatrix(cublasHandle_t handle, DMatrix dev_c, DMatrix dev_a, DMatrix dev_b);
__device__ void _bncuda_cmul_dmatrix(cublasHandle_t handle, DMatrix dev_c, double *sc, DMatrix dev_a);
__device__ void _bncuda_mul_dmatrix(cublasHandle_t handle, DMatrix dev_c, DMatrix dev_a, DMatrix dev_b);
__device__ void _bncuda_transpose_dmatrix(cublasHandle_t handle, DMatrix dev_c, DMatrix dev_a);
__device__ void _bncuda_subst_dmatrix(cublasHandle_t handle, DMatrix dev_c, DMatrix dev_a);
__device__ void _bncuda_set0_dmatrix(cublasHandle_t handle, DMatrix dev_c);
__device__ void _bncuda_setI_dmatrix(cublasHandle_t handle, DMatrix dev_c);
__device__ void _bncuda_mul_dmatrix_dvec(cublasHandle_t handle, DVector dev_ret, DMatrix dev_mat, DVector dev_vec);
__device__ void _bncuda_mul_dmatrixt_dvec(cublasHandle_t handle, DVector dev_ret, DMatrix dev_mat, DVector dev_vec);
__device__ void _bncuda_inv_dmatrix(cublasHandle_t handle, DMatrix a);

/* Krylov subspace method */
//__device__ void _bncuda_DBiCG(cublasHandle_t handle, long int *ret_val, DVector dev_answer, DMatrix dev_a, DVector dev_b, double *pt_reps, double *pt_aeps, long int *pt_maxtimes, DVector dev_vec[8]);
//void _bncuda_DBiCGSTAB(cublasHandle_t handle, long int *ret_val, DVector dev_answer, DMatrix dev_a, DVector dev_b, double reps, double aeps, long int maxtimes);
__global__ void _bncuda_dev_DBiCG(cublasHandle_t handle, long int *ret_val, DVector dev_answer, DMatrix dev_a, DVector dev_b, double *pt_reps, double *pt_aeps, long int *pt_maxtimes, DVector dev_vec[8]);
long int _bncuda_DBiCG(cublasHandle_t handle, DVector dev_answer, DMatrix dev_a, DVector dev_b, double reps, double aeps, long int maxtimes);

/* LU decomposition */
int _bncuda_DLU(DMatrix mat);
int _bncuda_SolveDLS(DVector x, DMatrix lu, DVector b);

/* Big matrix, Kronecker Matrix, long vector */
int _bncuda_set_bigmat_dmatrix(DMatrix ret_bigmat, DMatrix small_mat, long int bigmat_start_row_index, long int bigmat_start_col_index);
int _bncuda_kprod_dmatrix(DMatrix ret_bigmat, DMatrix left_smallmat, DMatrix right_smallmat);
int _bncuda_set_longvec_dvector(DVector ret_longvec, DVector vec, long int longvec_start_index);
int _bncuda_get_longvec_dvector(DVector ret_vec, DVector longvec, long int longvec_start_index);

#if defined (__cplusplus)
} // extern "C"
#endif

#endif // _BNCUDA_H

