/********************************************************************************/
/* bncuda_linear_cublas.c: Vector, Matrix                                       */
/* Copyright (c) 2013 Tomonori Kouya                                            */
/*                                                                              */
/* Version 0.1, 2013-01-26: first implementation with CUDA and cublas           */
/* Version 0.11,2013-03-01: Use CUBLAS_POINTER_MODE_HOST                        */
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
#ifndef USE_INCLUDE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// BNCpack
#include "bnc.h"
#endif

// BNCuda
#include "bncuda.h"

#ifndef _BNCUDA_CONST
  #define _BNCUDA_CONST
#ifdef __CUDACC__
__device__ float *_bncuda_const_f_zero; // = 0;
__device__ float *_bncuda_const_f_one; // = 1
__device__ float *_bncuda_const_f_minus_one; // = -1
__device__ double *_bncuda_const_d_one; // = 1
__device__ double *_bncuda_const_d_zero; // = 0;
__device__ double *_bncuda_const_d_minus_one; // = -1

__device__ float _bncuda_const_f_array[3];// = {0.0f, 1.0f, -1.0f}; // on HOST
__device__ double _bncuda_const_d_array[3]; // = {0.0, 1.0, 1.0}; // on HOST
#endif // __CUDACC__
#endif // _BNCUDA_CONST

// init_d on GPU
int * _bncuda_init_i(void)
{
	cudaError_t cuda_error;
	int *ret = NULL;

	cuda_error = cudaMalloc((void **)&ret, sizeof(int));
	if(cuda_error != cudaSuccess)
	{
  	
		printf("ERROR: _bncuda_init_i failed!(size = %ld)\n", sizeof(int));
		cudaFree(ret);
		return NULL;
	}

	return ret;
}

// free_d on GPU
void _bncuda_free_i(int *dev_val)
{
	cudaFree(dev_val);
}

// init_d on GPU
long int * _bncuda_init_l(void)
{
	cudaError_t cuda_error;
	long int *ret = NULL;

	cuda_error = cudaMalloc((void **)&ret, sizeof(long int));
	if(cuda_error != cudaSuccess)
	{
  	
		printf("ERROR: _bncuda_init_l failed!(size = %ld)\n", sizeof(long int));
		cudaFree(ret);
		return NULL;
	}

	return ret;
}

// free_d on GPU
void _bncuda_free_l(long int *dev_val)
{
	cudaFree(dev_val);
}

// init_d on GPU
double * _bncuda_init_d(void)
{
	cudaError_t cuda_error;
	double *ret = NULL;

	cuda_error = cudaMalloc((void **)&ret, sizeof(double));
	if(cuda_error != cudaSuccess)
	{
		printf("ERROR: _bncuda_init_d failed!(size = %ld)\n", sizeof(double));
		cudaFree(ret);
		return NULL;
	}

	return ret;
}

// free_d on GPU
void _bncuda_free_d(double *dev_val)
{
	cudaFree(dev_val);
}


// init_f on GPU
float * _bncuda_init_f(void)
{
	cudaError_t cuda_error;
	float *ret = NULL;

	cuda_error = cudaMalloc((void **)&ret, sizeof(float));
	if(cuda_error != cudaSuccess)
	{
		printf("ERROR: _bncuda_init_f failed!(size = %ld)\n", sizeof(float));
		cudaFree(ret);
		return NULL;
	}

	return ret;
}

// free_f on GPU
void _bncuda_free_f(float *dev_val)
{
	cudaFree(dev_val);
}

// init_d_array on GPU
double * _bncuda_init_d_array(int array_num)
{
	cudaError_t cuda_error;
	double *ret = NULL;

	cuda_error = cudaMalloc((void **)&ret, sizeof(double) * (int)array_num);
	if(cuda_error != cudaSuccess)
	{
		printf("ERROR: _bncuda_init_d_array failed!(size = %ld)\n", sizeof(double) * array_num);
		cudaFree(ret);
		return NULL;
	}

	return ret;
}

// free_d_array on GPU
void _bncuda_free_d_array(double *dev_array)
{
	cudaFree(dev_array);
}

// init_l_array on GPU
long int * _bncuda_init_l_array(int array_num)
{
	cudaError_t cuda_error;
	long int *ret = NULL;

	cuda_error = cudaMalloc((void **)&ret, sizeof(long int) * (int)array_num);
	if(cuda_error != cudaSuccess)
	{
		printf("ERROR: _bncuda_init_l_array failed!(size = %ld)\n", sizeof(double) * array_num);
		cudaFree(ret);
		return NULL;
	}

	return ret;
}

// free_l_array on GPU
void _bncuda_free_l_array(long int *dev_array)
{
	cudaFree(dev_array);
}

// copy a float val from device
float _bncuda_get_f(float *dev_val)
{
	float ret;

	cudaMemcpy((void *)&ret, (void *)(dev_val), sizeof(float), cudaMemcpyDeviceToHost);

	return ret;
}

// copy a double val from device
double _bncuda_get_d(double *dev_val)
{
	double ret;	

	cudaMemcpy((void *)&ret, (void *)(dev_val), sizeof(double), cudaMemcpyDeviceToHost);
//	cublasGetVector(1, sizeof(double), (void *)dev_val, 1, (void *)&ret, 1);

	return ret;
}

// copy a int val from device
int _bncuda_get_i(int *dev_val)
{
	int ret;	

	cudaMemcpy((void *)&ret, (void *)(dev_val), sizeof(int), cudaMemcpyDeviceToHost);

	return ret;
}

// copy a long val from device
long int _bncuda_get_l(long int *dev_val)
{
	long int ret;	

	cudaMemcpy((void *)&ret, (void *)(dev_val), sizeof(long int), cudaMemcpyDeviceToHost);

	return ret;
}

// copy a long int array from device
void _bncuda_get_l_array(long int *array, long int *dev_array, long int array_num)
{
	cudaMemcpy((void *)array, (void *)dev_array, sizeof(long int) * array_num, cudaMemcpyDeviceToHost);
}

// copy a float val to device
void _bncuda_set_f(float *dev_val, float val)
{
	cudaMemcpy((void *)(dev_val), (void *)&val, sizeof(float), cudaMemcpyHostToDevice);

}

// copy a double val to device
void _bncuda_set_d(double *dev_val, double val)
{
	cudaMemcpy((void *)(dev_val), (void *)&val, sizeof(double), cudaMemcpyHostToDevice);
//	cublasSetVector(1, sizeof(double), (void *)&val, 1, (void *)dev_val, 1);
}

// copy a int val to device
void _bncuda_set_i(int *dev_val, int val)
{
	cudaMemcpy((void *)(dev_val), (void *)&val, sizeof(int), cudaMemcpyHostToDevice);
}

// copy a long int val to device
void _bncuda_set_l(long int *dev_val, long int val)
{
	cudaMemcpy((void *)(dev_val), (void *)&val, sizeof(long int), cudaMemcpyHostToDevice);
}

// copy a long int array to device
void _bncuda_set_l_array(long int *dev_array, long int *array, long int array_num)
{
	cudaMemcpy((void *)(dev_array), (void *)array, sizeof(long int) * array_num, cudaMemcpyHostToDevice);
}

// Initialize & set
long int * _bncuda_init_set_l(long int val)
{
//	cudaError_t cuda_error;
	long int *ret = NULL;

	// initialize
	ret = _bncuda_init_l();
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: _bncuda_init_set_l cannot initialize!\n");
		return ret;
	}

	/* set init_val */
	_bncuda_set_l(ret, val);

	return ret;
}

// Initialize & set
double * _bncuda_init_set_d(double val)
{
//	cudaError_t cuda_error;
	double *ret = NULL;

	// initialize
	ret = _bncuda_init_d();
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: _bncuda_init_set_d cannot initialize!\n");
		return ret;
	}

	/* set init_val */
	_bncuda_set_d(ret, val);

	return ret;
}

// Initialize & set
float * _bncuda_init_set_f(float val)
{
//	cudaError_t cuda_error;
	float *ret = NULL;

	// initialize
	ret = _bncuda_init_f();
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: _bncuda_init_set_f cannot initialize!\n");
		return ret;
	}

	/* set init_val */
	_bncuda_set_f(ret, val);

	return ret;
}

// set constants on device
#ifdef __CUDACC__
__global__ void _bncuda_set_const(void)
{
	_bncuda_const_d_zero = &_bncuda_const_d_array[0]; // = 0
	_bncuda_const_d_one  = &_bncuda_const_d_array[1]; // = 1
	_bncuda_const_d_minus_one = &_bncuda_const_d_array[2]; // = -1

	_bncuda_const_f_zero = &_bncuda_const_f_array[0]; // = 0
	_bncuda_const_f_one  = &_bncuda_const_f_array[1]; // = 1
	_bncuda_const_f_minus_one = &_bncuda_const_f_array[2]; // = -1
}
#endif

#ifdef USE_CUBLAS

// initialize cublas
// return NULL if error
// return pointer_to_cublas_handle if success
cublasHandle_t *_init_bncuda(cublasHandle_t *cublas_handle)
{
	cublasStatus_t cublas_status;
	int cublas_version;

	// initialize cudablas
	cublas_status = cublasCreate(cublas_handle);
	if(cublas_status != CUBLAS_STATUS_SUCCESS)
	{
		printf("_init_bncuda: CUBLAS initialization failed!\n");
		return NULL;
	}

	// get version of cublas
	cublasGetVersion(*cublas_handle, &cublas_version);
	printf("CUBLAS Version %d\n", cublas_version);	

	/* constant cache on GPU */
#ifndef __CUDACC__
	_bncuda_const_d_array[0] = 0.0;
	_bncuda_const_d_array[1] = 1.0;
	_bncuda_const_d_array[2] = -1.0;
	_bncuda_const_f_array[0] = 0.0f;
	_bncuda_const_f_array[1] = 1.0f;
	_bncuda_const_f_array[2] = -1.0f;

#ifdef USE_POINTER_MODE_DEVICE // constants on GPU
	_bncuda_const_d_zero = _bncuda_init_set_d(0.0); // = 0
	_bncuda_const_d_one  = _bncuda_init_set_d(1.0); // = 1
	_bncuda_const_d_minus_one = _bncuda_init_set_d(-1.0); // = -1

	_bncuda_const_f_zero = _bncuda_init_set_f(0.0f); // = 0
	_bncuda_const_f_one  = _bncuda_init_set_f(1.0f); // = 1
	_bncuda_const_f_minus_one = _bncuda_init_set_f(-1.0f); // = -1
#else // USE_POINTER_MODE on HOST
	_bncuda_const_d_zero = &_bncuda_const_d_array[0]; // = 0
	_bncuda_const_d_one  = &_bncuda_const_d_array[1]; // = 1
	_bncuda_const_d_minus_one = &_bncuda_const_d_array[2]; // = -1

	_bncuda_const_f_zero = &_bncuda_const_f_array[0]; // = 0
	_bncuda_const_f_one  = &_bncuda_const_f_array[1]; // = 1
	_bncuda_const_f_minus_one = &_bncuda_const_f_array[2]; // = -1
#endif
#else // __CUDACC__
	_bncuda_set_const<<<1, 1>>>();
#endif

	// to pointer mode
#ifdef USE_POINTER_MODE_DEVICE
	cublasSetPointerMode(*cublas_handle, CUBLAS_POINTER_MODE_DEVICE);
#else // default is POINTER_MODE_HOST
	cublasSetPointerMode(*cublas_handle, CUBLAS_POINTER_MODE_HOST);
#endif

	return cublas_handle;
}

// finalize bncuda
void _clear_bncuda(cublasHandle_t cublas_handle)
{
	cublasDestroy(cublas_handle);

#ifdef USE_POINTER_MODE_DEVICE
	// clear constant cache
#ifndef __CUDACC__
	_bncuda_free_d(_bncuda_const_d_one);
	_bncuda_free_d(_bncuda_const_d_zero);
	_bncuda_free_d(_bncuda_const_d_minus_one);

	_bncuda_free_f(_bncuda_const_f_one);
	_bncuda_free_f(_bncuda_const_f_zero);
	_bncuda_free_f(_bncuda_const_f_minus_one);
#endif
#endif
}

/* double */

// copy dmatrix to device(GPU)
void _bncuda_set_dmatrix(DMatrix dev_mat, DMatrix host_mat)
{
	cublasStatus_t cublas_status;
	DMatrix tmp_mat; // transposed host_mat;

	// transpose matrix
	tmp_mat = init_dmatrix(host_mat->col_dim, host_mat->row_dim);
	transpose_dmatrix(tmp_mat, host_mat);

	// cublasSetMatrix store matrix as column-major!
	cublas_status = cublasSetMatrix((int)(host_mat->row_dim), (int)(host_mat->col_dim), (int)sizeof(double), (void *)(tmp_mat->element), (int)(host_mat->row_dim), (void *)(dev_mat->element), (int)(dev_mat->row_dim));
	if(cublas_status != CUBLAS_STATUS_SUCCESS)
	{
		printf("_bncuda_set_dmatrix: host -> dev failed!\n");
		free_dmatrix(tmp_mat);
		return;
	}

	// free
	free_dmatrix(tmp_mat);
}

// copy vector to device memory
void _bncuda_set_dvector(DVector dev_vec, DVector host_vec)
{
	cublasStatus_t cublas_status;

	cublas_status = cublasSetVector((int)(host_vec->dim), sizeof(double), (void *)(host_vec->element), 1, (void *)(dev_vec->element), 1);
	if(cublas_status != CUBLAS_STATUS_SUCCESS)
	{
		printf("_bncuda_set_dvector: host -> dev failed!\n");
		return;
	}
}

// copy matrix to ram from device memory
void _bncuda_get_dmatrix(DMatrix host_mat, DMatrix dev_mat)
{
	cublasStatus_t cublas_status;
	DMatrix tmp_mat;

	tmp_mat = init_dmatrix(dev_mat->col_dim, dev_mat->row_dim);

	// cublas store row-major!
	cublas_status = cublasGetMatrix((int)(dev_mat->row_dim), (int)(dev_mat->col_dim), (int)sizeof(double), (void *)(dev_mat->element), (int)(dev_mat->row_dim), (void *)(tmp_mat->element), (int)(host_mat->row_dim));
	if(cublas_status != CUBLAS_STATUS_SUCCESS)
	{
		printf("_bncuda_get_dmatrix: dev -> host failed!\n");
		free_dmatrix(tmp_mat);
		return;
	}

	transpose_dmatrix(host_mat, tmp_mat);
	free_dmatrix(tmp_mat);
}

// copy matrix to ram from device memory
void _bncuda_get_dvector(DVector host_vec, DVector dev_vec)
{
	cublasStatus_t cublas_status;

//	cudaMemcpy((void *)(dev_vec->element), (void *)(host_vec->element), sizeof(double) * (int)(host_vec->dim), cudaMemcpyDeviceToHost);
	cublas_status = cublasGetVector((int)(dev_vec->dim), sizeof(double), (void *)(dev_vec->element), 1, (void *)(host_vec->element), 1);
	if(cublas_status != CUBLAS_STATUS_SUCCESS)
	{
		printf("_bncuda_get_dvector: dev -> host failed!\n");
		return;
	}
}

/* Float */

// copy dmatrix to device(GPU)
__host__ void _bncuda_set_fmatrix(FMatrix dev_mat, FMatrix host_mat)
{
	cublasStatus_t cublas_status;
	FMatrix tmp_mat; // transposed host_mat;

	// transpose matrix
	tmp_mat = init_fmatrix(host_mat->col_dim, host_mat->row_dim);
	transpose_fmatrix(tmp_mat, host_mat);

	// cublasSetMatrix store matrix as column-major!
	cublas_status = cublasSetMatrix((int)(host_mat->row_dim), (int)(host_mat->col_dim), (int)sizeof(float), (void *)(tmp_mat->element), (int)(host_mat->row_dim), (void *)(dev_mat->element), (int)(dev_mat->row_dim));
	if(cublas_status != CUBLAS_STATUS_SUCCESS)
	{
		printf("_bncuda_set_fmatrix: host -> dev failed!\n");
		free_fmatrix(tmp_mat);
		return;
	}

	// free
	free_fmatrix(tmp_mat);
}

// copy vector to device memory
void _bncuda_set_fvector(FVector dev_vec, FVector host_vec)
{
	cublasStatus_t cublas_status;

	cublas_status = cublasSetVector((int)(host_vec->dim), sizeof(float), (void *)(host_vec->element), 1, (void *)(dev_vec->element), 1);
	if(cublas_status != CUBLAS_STATUS_SUCCESS)
	{
		printf("_bncuda_set_fvector: host -> dev failed!\n");
		return;
	}
}

// copy matrix to ram from device memory
__host__ void _bncuda_get_fmatrix(FMatrix host_mat, FMatrix dev_mat)
{
	cublasStatus_t cublas_status;
	FMatrix tmp_mat;

	tmp_mat = init_fmatrix(dev_mat->col_dim, dev_mat->row_dim);

	// cublas store row-major!
	cublas_status = cublasGetMatrix((int)(dev_mat->row_dim), (int)(dev_mat->col_dim), (int)sizeof(float), (void *)(dev_mat->element), (int)(dev_mat->row_dim), (void *)(tmp_mat->element), (int)(host_mat->row_dim));
	if(cublas_status != CUBLAS_STATUS_SUCCESS)
	{
		printf("_bncuda_get_fmatrix: dev -> host failed!\n");
		free_fmatrix(tmp_mat);
		return;
	}

	transpose_fmatrix(host_mat, tmp_mat);
	free_fmatrix(tmp_mat);
}

// copy matrix to ram from device memory
void _bncuda_get_fvector(FVector host_vec, FVector dev_vec)
{
	cublasStatus_t cublas_status;

	cublas_status = cublasGetVector((int)(dev_vec->dim), (int)sizeof(float), (void *)(dev_vec->element), (int)1, (void *)(host_vec->element), (int)1);
	if(cublas_status != CUBLAS_STATUS_SUCCESS)
	{
		printf("_bncuda_get_fvector: dev -> host failed!\n");
		return;
	}
}

#ifndef __CUDACC__
/* ret := dev_a + dev_b */
void _bncuda_add_d(cublasHandle_t handle, double *ret, double *dev_a, double *dev_b)
{
	// ret := dev_b
	cublasDcopy(handle, 1, dev_b, 1, ret, 1);
	// ret := dev_a + ret
	cublasDaxpy(handle, 1, _bncuda_const_d_one, dev_a, 1, ret, 1);
}

/* ret += dev_a */
void _bncuda_add2_d(cublasHandle_t handle, double *ret, double *dev_a)
{
	// ret := dev_a + ret
	cublasDaxpy(handle, 1, _bncuda_const_d_one, dev_a, 1, ret, 1);
}

/* ret := dev_a - dev_b */
void _bncuda_sub_d(cublasHandle_t handle, double *ret, double *dev_a, double *dev_b)
{
	// ret := dev_a
	cublasDcopy(handle, 1, dev_a, 1, ret, 1);
	// ret := -dev_b + ret
	cublasDaxpy(handle, 1, _bncuda_const_d_minus_one, dev_b, 1, ret, 1);
}

/* ret := dev_a * dev_b */
void _bncuda_mul_d(cublasHandle_t handle, double *ret, double *dev_a, double *dev_b)
{
	double tmp;
	// ret := dev_a * ret
	//cublasDdot(handle, 1, dev_a, 1, dev_b, 1, ret); // leak?
	tmp = _bncuda_get_d(dev_a) * _bncuda_get_d(dev_b);
	_bncuda_set_d(ret, tmp);
}
//void _bncuda_div_d(double *ret, double *dev_a, double *dev_b);

/* ret := dev_a + dev_b * dev_c */
void _bncuda_fma_d(cublasHandle_t handle, double *ret, double *dev_a, double *dev_b, double *dev_c)
{
	// ret := dev_a
	cublasDcopy(handle, 1, dev_a, 1, ret, 1);
	// ret := dev_b * dev_c + ret
	cublasDaxpy(handle, 1, dev_b, dev_c, 1, ret, 1);
}
#endif // __CUDACC__

/*************************************************/
/* Functions for Vector Types                    */
/*                                               */
/* Initialize:                                   */
/*   FVector _bncuda_init_fvector(long int dimension)    */
/*   DVector _bncuda_init_dvector(long int dimension)    */
/* Free:                                         */
/*   void _bncuda_free_fvector(FVector vec)              */
/*   void _bncuda_free_dvector(DVector vec)              */
/*   float _bncuda_get_fvector_i(FVector vec, long int index) */
/*   double _bncuda_get_dvector_i(DVector vec, long int index) */
/*   void _bncuda_set_fvector_i(FVector vec, long int index, float val) */
/*   void _bncuda_set_dvector_i(DVector vec, long int index, double val) */
/* Output:                                       */
/*   void _bncuda_print_fvector(FVector vec)             */
/*   void _bncuad_print_dvector(DVector vec)             */
/*************************************************/
// init_dvector on GPU
DVector _bncuda_init_dvector(long int dimension)
{
	cudaError_t cuda_error;
	DVector ret = NULL;
//	long int i;

	if(dimension <= 0)
	{
		fprintf(stderr, "ERROR: _bncuda_init_dvector\n");
		return ret;
	}

	ret = (DVector)malloc(sizeof(dvector));
	if(ret == NULL)
		return ret;

//	ret->element = (double *)calloc(sizeof(double), dimension);
//	if(ret->element == NULL)
//		return ret;

	cuda_error = cudaMalloc((void **)&(ret->element), (int)dimension * sizeof(double));
	if(cuda_error != cudaSuccess)
	{
		printf("device memory allocation failed!(dimension = %ld, size = %ld)\n", dimension, dimension * sizeof(double));
		free(ret);
		return NULL;
	}

	/* All 0 */
	cudaMemset((void *)(ret->element), 0, sizeof(double) * (int)dimension);
//	for(i = 0; i < dimension; i++)
//		_bncuda_set_dvector_i(ret, i, 0.0);

	ret->dim = dimension;

	return ret;
}

// free_dvector on GPU
void _bncuda_free_dvector(DVector dev_vec)
{
	cudaFree(dev_vec->element);
	free(dev_vec);
}


// initialize & set vector
DVector _bncuda_init_set_dvector(DVector host_vec)
{
	DVector ret = NULL;

	ret = _bncuda_init_dvector(host_vec->dim);
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: _bncuda_init_set_dvector cannot initialize! (dim = %ld)\n", host_vec->dim);
		return ret;
	}

	_bncuda_set_dvector(ret, host_vec);

	return ret;
}

// init_fvector on GPU
FVector _bncuda_init_fvector(long int dimension)
{
	cudaError_t cuda_error;
	FVector ret = NULL;
//	long int i;

	if(dimension <= 0)
	{
		fprintf(stderr, "ERROR: _bncuda_init_fvector\n");
		return ret;
	}

	ret = (FVector)malloc(sizeof(fvector));
	if(ret == NULL)
		return ret;

//	ret->element = (float *)calloc(sizeof(float), dimension);
//	if(ret->element == NULL)
//		return ret;

	cuda_error = cudaMalloc((void **)&(ret->element), (int)dimension * sizeof(float));
	if(cuda_error != cudaSuccess)
	{
		printf("device memory allocation failed!(dimension = %ld, size = %ld)\n", dimension, dimension * sizeof(float));
		free(ret);
		return NULL;
	}

	/* All 0 */
	cudaMemset((void *)(ret->element), 0, sizeof(float) * (int)dimension);
///	for(i = 0; i < dimension; i++)
//		*(ret->element + i) = 0.0;

	ret->dim = dimension;

	return ret;
}

// free_dvector on GPU
void _bncuda_free_fvector(FVector dev_vec)
{
	cudaFree(dev_vec->element);
	free(dev_vec);
}

// initialize & set vector
FVector _bncuda_init_set_fvector(FVector host_vec)
{
	FVector ret = NULL;

	ret = _bncuda_init_fvector(host_vec->dim);
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: _bncuda_init_set_fvector cannot initialize! (dim = %ld)\n", host_vec->dim);
		return ret;
	}

	_bncuda_set_fvector(ret, host_vec);

	return ret;
}

__device__ float _bncuda_get_fvector_i(FVector dev_vec, long int index)
{
	float ret;
//	return *(vec->element + index);
#ifndef __CUDACC__
	cudaMemcpy((void *)&ret, (void *)(dev_vec->element + index), sizeof(float) * 1, cudaMemcpyDeviceToHost);
#else
	ret = dev_vec->element[index];
#endif
	return ret;
}

__device__ float * _bncuda_getp_fvector_i(FVector dev_vec, long int index)
{
	float *ret;	

	ret = (dev_vec->element + index);
//	ret = &(dev_vec->element[index]);

	return ret;
}

__device__ double _bncuda_get_dvector_i(DVector dev_vec, long int index)
{
	double ret;	
//	return *(vec->element + index);
#ifndef __CUDACC__
	cudaMemcpy((void *)&ret, (void *)(dev_vec->element + index), sizeof(double) * 1, cudaMemcpyDeviceToHost);
#else
	ret = dev_vec->element[index];
#endif

	return ret;
}

__device__ double * _bncuda_getp_dvector_i(DVector dev_vec, long int index)
{
	double *ret;	

	ret = (dev_vec->element + index);

	return ret;
}

__device__ void _bncuda_set_fvector_i(FVector dev_vec, long int index, float val)
{
//	*(vec->element + index) = val;
#ifndef __CUDACC__
	cudaMemcpy((void *)(dev_vec->element + index), (void *)&val, sizeof(float) * 1, cudaMemcpyHostToDevice);
#else
	dev_vec->element[index] = val;
#endif
}

__device__ void _bncuda_set_dvector_i(DVector dev_vec, long int index, double val)
{
//	*(vec->element + index) = val;
#ifndef __CUDACC__
	cudaMemcpy((void *)(dev_vec->element + index), (void *)&val, sizeof(double) * 1, cudaMemcpyHostToDevice);
#else
	dev_vec->element[index] = val;
#endif
}

void _bncuda_print_fvector(FVector dev_fv)
{
	long int i;

	for(i = 0; i < dev_fv->dim; i++)
		printf("%5ld %15.7e\n", i, _bncuda_get_fvector_i(dev_fv, i));
}
void _bncuda_print_dvector(DVector dev_dv)
{
	long int i;

	for(i = 0; i < dev_dv->dim; i++)
		printf("%5ld %25.17e\n", i, _bncuda_get_dvector_i(dev_dv, i));
}

/*************************************************/
/* Function for Matrix Types                     */
/*                                               */
/* Initialize:                                   */
/*   FMatrix _bncuda_init_fmatrix(long int row_dimension, long int col_dimension)    */
/*   DMatrix _bncuda_init_dmatrix(long int row_dimension, long int col_dimension)    */
/* Free:                                         */
/*   void _bncuda_free_fmatrix(FMatrix mat)              */
/*   void _bncuda_free_dmatrix(DMatrix mat)              */
/* Get & Set:
/*   float _bncuda_get_fmatrix_ij(FMatrix mat, long int row_index, long int col_index) */
/*   float _bncuda_get_dmatrix_ij(DMatrix mat, long int row_index, long int col_index) */
/*   float _bncuda_get_mpfmatrix_ij(MPFMatrix mat, long int row_index, long int col_index) */
/*   void _bncuda_set_fmatrix_ij(FMatrix mat, long int row_index, long int col_index, float val) */
/*   void _bncuda_set_dmatrix_ij(DMatrix mat, long int row_index, long int col_index, double val) */
/* Output:                                       */
/*   void _bncuda_print_fmatrix(FMatrix mat)             */
/*   void _bncuda_print_dmatrix(DMatrix mat)             */
/*************************************************/
FMatrix _bncuda_init_fmatrix(long int row_dimension, long int col_dimension)
{
	cudaError_t cuda_error;
	FMatrix ret = NULL;
//	long int i, j;

	/* Check Dimentions */
	if(row_dimension <= 0 || col_dimension <= 0)
	{
		fprintf(stderr, "ERROR: init_fmatrix\n");
		return ret;
	}

	ret = (FMatrix)malloc(sizeof(fmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: init_fmatrix\n");
		return ret;
	}

	cuda_error = cudaMalloc((void **)&(ret->element), row_dimension * col_dimension * sizeof(float));
	if(cuda_error != cudaSuccess)
	{
		fprintf(stderr, "ERROR: _bncuda_init_fmatrix\n");
		return ret;
	}

	/* All 0 */
	cudaMemset((void *)(ret->element), 0, sizeof(float) * (int)row_dimension * (int)col_dimension);
///	for(i = 0; i < row_dimension; i++)
//		for(j = 0; j < col_dimension; j++)
//			*(ret->element + i * col_dimension + j) = (float)0.0;

	ret->row_dim = row_dimension;
	ret->col_dim = col_dimension;

	return ret;
}

// allocate dmatrix on GPU
DMatrix _bncuda_init_dmatrix(long int row_dimension, long int col_dimension)
{
	cudaError_t cuda_error;
	DMatrix ret = NULL;
//	long int i, j;

	if(row_dimension <= 0 || col_dimension <= 0)
	{
		fprintf(stderr, "ERROR: init_dmatrix\n");
		return ret;
	}

	ret = (DMatrix)malloc(sizeof(dmatrix));
	if(ret == NULL)
		return ret;

	cuda_error = cudaMalloc((void **)&(ret->element), row_dimension * col_dimension * sizeof(double));
	if(cuda_error != cudaSuccess)
	{
		printf("device memory allocation failed!(row * col = %ld, %ld, size = %ld)\n", row_dimension, col_dimension,  sizeof(double));
		return NULL;
	}
//	ret->element = (double *)calloc(sizeof(double), row_dimension * col_dimension);
//	if(ret->element == NULL)
//		return ret;

	/* All 0 */
	cudaMemset((void *)(ret->element), 0, sizeof(double) * (int)row_dimension * (int)col_dimension);
//	for(i = 0; i < row_dimension; i++)
//		for(j = 0; j < col_dimension; j++)
//			*(ret->element + i * col_dimension + j) = (double)0.0;

	ret->row_dim = row_dimension;
	ret->col_dim = col_dimension;

	return ret;
}

// free fmatrix
void _bncuda_free_fmatrix(FMatrix dev_mat)
{
	if(dev_mat == NULL)
		return;

	if(dev_mat->element != NULL)
		cudaFree(dev_mat->element);

	free(dev_mat);
}

// free dmatrix
void _bncuda_free_dmatrix(DMatrix dev_mat)
{
	if(dev_mat == NULL)
		return;

	if(dev_mat->element != NULL)
		cudaFree(dev_mat->element);

	free(dev_mat);
}

// initialize & set matrix
FMatrix _bncuda_init_set_fmatrix(FMatrix host_mat)
{
	FMatrix ret = NULL;

	ret = _bncuda_init_fmatrix(host_mat->row_dim, host_mat->col_dim);
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: _bncuda_init_set_fmatrix cannot initialize! (row_dim = %ld, col_dim = %ld)\n", host_mat->row_dim, host_mat->col_dim);
		return ret;
	}

	_bncuda_set_fmatrix(ret, host_mat);

	return ret;
}

// initialize & set matrix
DMatrix _bncuda_init_set_dmatrix(DMatrix host_mat)
{
	DMatrix ret = NULL;

	ret = _bncuda_init_dmatrix(host_mat->row_dim, host_mat->col_dim);
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: _bncuda_init_set_dmatrix cannot initialize! (row_dim = %ld, col_dim = %ld)\n", host_mat->row_dim, host_mat->col_dim);
		return ret;
	}

	_bncuda_set_dmatrix(ret, host_mat);

	return ret;
}

__device__ float _bncuda_get_fmatrix_ij(FMatrix dev_mat, long int row_index, long int col_index)
{
	float ret;
	//return (float)(*(mat->element + row_index * mat->col_dim + col_index));

	// column-major in cublas!
#ifndef __CUDACC__
	cudaMemcpy((void *)&ret, (void *)(dev_mat->element + col_index * dev_mat->row_dim + row_index), sizeof(float) * 1, cudaMemcpyDeviceToHost);
#else
	ret = dev_mat->element[col_index * dev_mat->row_dim + row_index];
#endif

	return ret;
}

__device__ double _bncuda_get_dmatrix_ij(DMatrix dev_mat, long int row_index, long int col_index)
{
	double ret;
//	return *(mat->element + row_index * mat->col_dim + col_index);

	// column-major in cublas!
#ifndef __CUDACC__
	cudaMemcpy((void *)&ret, (void *)(dev_mat->element + col_index * dev_mat->row_dim + row_index), sizeof(double) * 1, cudaMemcpyDeviceToHost);
#else
	ret = dev_mat->element[col_index * dev_mat->row_dim + row_index];
#endif

	return ret;
}

__device__ double * _bncuda_getp_dmatrix_ij(DMatrix dev_mat, long int row_index, long int col_index)
{
	double *ret;
//	return *(mat->element + row_index * mat->col_dim + col_index);

	// column-major in cublas!
	//cudaMemcpy((void *)&ret, (void *)(dev_mat->element + col_index * dev_mat->row_dim + row_index), sizeof(double) * 1, cudaMemcpyDeviceToHost);

	ret = (dev_mat->element + row_index * dev_mat->col_dim + col_index);

	return ret;
}

__device__ void _bncuda_set_fmatrix_ij(FMatrix dev_mat, long int row_index, long int col_index, float val)
{
	//*(mat->element + row_index * mat->col_dim + col_index) = (float)val;

	// column-major in cublas!
#ifndef __CUDACC__
	cudaMemcpy((void *)(dev_mat->element + col_index * dev_mat->row_dim + row_index), (void *)&val, sizeof(float) * 1, cudaMemcpyHostToDevice);
#else
	dev_mat->element[col_index * dev_mat->row_dim + row_index] = val;
#endif
}

__device__ void _bncuda_set_dmatrix_ij(DMatrix dev_mat, long int row_index, long int col_index, double val)
{
//	*(mat->element + row_index * mat->col_dim + col_index) = val;

	// column-major in cublas!
#ifndef __CUDACC__
	cudaMemcpy((void *)(dev_mat->element + col_index * dev_mat->row_dim + row_index), (void *)&val, sizeof(double) * 1, cudaMemcpyHostToDevice);
#else
	dev_mat->element[col_index * dev_mat->row_dim + row_index] = val;
#endif
}

void _bncuda_print_fmatrix(FMatrix dev_mat)
{
	long int i, j;

	for(i = 0; i < dev_mat->row_dim; i++)
	{
		printf("%5ld ", i);
		for(j = 0; j < dev_mat->col_dim; j++)
			printf("%15.7e ", _bncuda_get_fmatrix_ij(dev_mat, i, j));
		printf("\n");
	}
}

void _bncuda_print_dmatrix(DMatrix dev_mat)
{
	long int i, j;

	for(i = 0; i < dev_mat->row_dim; i++)
	{
		printf("%5ld ", i);
		for(j = 0; j < dev_mat->col_dim; j++)
			printf("%25.17e ", _bncuda_get_dmatrix_ij(dev_mat, i, j));
		printf("\n");
	}
}

/*************************************************/
/* Vector Calculations for FVector               */
/*
void add_fvector(FVector c, FVector a, FVector b)
void sub_fvector(FVector c, FVector a, FVector b)
void cmul_fvector(FVector c, float val, FVector a)
float ip_fvector(FVector a, FVector b)
float norm1_fvector(FVector a)
float norm2_fvector(FVector a)
float normi_fvector(FVector a)
void subst_fvector(FVector c, FVector a)
*/
/*************************************************/
/* c = a + b */
__device__ void _bncuda_add_fvector(cublasHandle_t handle, FVector dev_c, FVector dev_a, FVector dev_b)
{
//	long int i;

#ifndef __CUDACC__
	if((dev_a->dim != dev_b->dim) || (dev_c->dim != dev_a->dim) || (dev_c->dim != dev_b->dim))
	{
		fprintf(stderr, "ERROR: _bncuda_add_fvector\n");
		return;
	}
#endif

//	for(i = 0; i < c->dim; i++)
//		_bncuda_set_fvector_i(c, i, _bncuda_get_fvector_i(a, i) + _bncuda_get_fvector_i(b, i));

	// dev_c := dev_b;
	//cudaMemcpy(dev_c->element, dev_b->element, sizeof(float) * dev_c->dim, cudaMemcpyDeviceToDevice);
	cublasScopy(handle, (int)(dev_c->dim), dev_b->element, 1, dev_c->element, 1);
	cublasSaxpy(handle, (int)(dev_c->dim), _bncuda_const_f_one, dev_a->element, 1, dev_c->element, 1);
}

/* c += a */
__device__ void _bncuda_add2_fvector(cublasHandle_t handle, FVector dev_c, FVector dev_a)
{
//	long int i;

#ifndef __CUDACC__
	if(dev_c->dim != dev_a->dim)
	{
		fprintf(stderr, "ERROR: _bncuda_add2_fvector\n");
		return;
	}
#endif

	cublasSaxpy(handle, (int)(dev_c->dim), _bncuda_const_f_one, dev_a->element, 1, dev_c->element, 1);
}

/* c = a - b */
__device__ void _bncuda_sub_fvector(cublasHandle_t handle, FVector dev_c, FVector dev_a, FVector dev_b)
{
//	long int i;

#ifndef __CUDACC__
	if((dev_a->dim != dev_b->dim) || (dev_c->dim != dev_a->dim) || (dev_c->dim != dev_b->dim))
	{
		fprintf(stderr, "ERROR: _bncuda_sub_fvector\n");
		return;
	}
#endif

	// dev_c := dev_a;
	cublasScopy(handle, (int)(dev_c->dim), dev_a->element, 1, dev_c->element, 1);
	cublasSaxpy(handle, (int)(dev_c->dim), _bncuda_const_f_minus_one, dev_b->element, 1, dev_c->element, 1);
}

/* c -= a */
__device__ void _bncuda_sub2_fvector(cublasHandle_t handle, FVector dev_c, FVector dev_a)
{
//	long int i;

#ifndef __CUDACC__
	if(dev_c->dim != dev_a->dim)
	{
		fprintf(stderr, "ERROR: _bncuda_sub2_fvector\n");
		return;
	}
#endif

	cublasSaxpy(handle, (int)(dev_c->dim), _bncuda_const_f_minus_one, dev_a->element, 1, dev_c->element, 1);
}

/* c = val * a */
__device__ void _bncuda_cmul_fvector(cublasHandle_t handle, FVector dev_c, float *val, FVector dev_a)
{
//	long int i;

#ifndef __CUDACC__
	if(dev_c->dim != dev_a->dim)
	{
		fprintf(stderr, "ERROR: _bncuda_cmul_fvector\n");
		return;
	}
#endif

	// dev_c := dev_a;
	cublasScopy(handle, (int)(dev_c->dim), dev_a->element, 1, dev_c->element, 1);
	cublasSscal(handle, (int)(dev_c->dim), val, dev_c->element, 1);
}

/* c *= val */
__device__ void _bncuda_cmul2_fvector(cublasHandle_t handle, FVector dev_c, float *dev_val)
{
	cublasSscal(handle, (int)(dev_c->dim), dev_val, dev_c->element, 1);
}

/* c = a + val * b */
__device__ void _bncuda_add_cmul_fvector(cublasHandle_t handle, FVector dev_c, FVector dev_a, float *val, FVector dev_b)
{
#ifndef __CUDACC__
	if((dev_a->dim != dev_b->dim) || (dev_c->dim != dev_a->dim) || (dev_c->dim != dev_b->dim))
	{
		fprintf(stderr, "ERROR: _bncuda_add_cmul_fvector\n");
		return;
	}
#endif
	// dev_c := dev_a;
	cublasScopy(handle, (int)(dev_c->dim), dev_a->element, 1, dev_c->element, 1);

	// dev_c := dev_c + val * dev_b;
	cublasSaxpy(handle, (int)(dev_c->dim), val, dev_b->element, 1, dev_c->element, 1);
}

/* (a, b) */
__device__ void _bncuda_ip_fvector(cublasHandle_t handle, float *ret, FVector dev_a, FVector dev_b)
{
//	long int i;

#ifndef __CUDACC__
	if(dev_a->dim != dev_b->dim)
	{
		fprintf(stderr, "ERROR: _bncuda_ip_fvector\n");
		return;
	}
#endif

//	for(i = 0; i < a->dim; i++)
//		tmp += _bncuda_get_fvector_i(a, i) * _bncuda_get_fvector_i(b, i);
	
	cublasSdot(handle, (int)(dev_a->dim), dev_a->element, 1, dev_b->element, 1, ret); 

	return;
}

/* ||a||_1 */
__device__ void _bncuda_norm1_fvector(cublasHandle_t handle, float *ret, FVector dev_a)
{
//	float ret = 0.0;
//	long int i;

//	for(i = 0; i < a->dim; i++)
//		ret += fabs((double)_bncuda_get_fvector_i(a, i));

	cublasSasum(handle, (int)(dev_a->dim), dev_a->element, 1, ret);

	return;
}

/* ||a||_2 */
__device__ void _bncuda_norm2_fvector(cublasHandle_t handle, float *ret, FVector dev_a)
{
//	float ret = 0.0;
//	long int i;

//	for(i = 0; i < a->dim; i++)
//		ret += _bncuda_get_fvector_i(a, i) * _bncuda_get_fvector_i(a, i);

	cublasSnrm2(handle, (int)(dev_a->dim), dev_a->element, 1, ret);

	return;
}

/* ||a||_infty */
__device__ void _bncuda_normi_fvector(cublasHandle_t handle, float *ret, FVector dev_a)
{
	float tmp;
	int tmp_i, *dev_tmp_i;
#ifdef USE_POINTER_MODE_DEVICE
	dev_tmp_i = _bncuda_init_i();
#else // USE_POINTER_MODE_HOST
	dev_tmp_i = &tmp_i;
#endif

//	ret = fabs(_bncuda_get_dvector_i(a, 0));
//	for(i = 1; i < a->dim; i++)
//	{
//		tmp = fabs((double)_bncuda_get_dvector_i(a, i));
//		if(ret < tmp)
//			ret = tmp;
//	}

	cublasIsamax(handle, (int)(dev_a->dim), dev_a->element, 1, dev_tmp_i);
//	*ret = fabs(*(dev_a->element + dev_tmp_i)); // correct ?
#ifdef USE_POINTER_MODE_DEVICE
	tmp_i = _bncuda_get_i(dev_tmp_i);
#endif	

	tmp = fabs(_bncuda_get_fvector_i(dev_a, tmp_i));

#ifdef USE_POINTER_MODE_DEVICE
	_bncuda_set_f(ret, tmp);
	_bncuda_free_i(dev_tmp_i);
#else // USE_POINTER_MODE_HOST
	*ret = tmp;
#endif
	
	return;
}

/* c := a */
__device__ void _bncuda_subst_fvector(cublasHandle_t handle, FVector dev_c, FVector dev_a)
{
//	long int i;

//	for(i = 0; i < a->dim; i++)
//		_bncuda_set_fvector_i(c, i, _bncuda_get_fvector_i(a, i));
	cublasScopy(handle, (int)(dev_c->dim), dev_a->element, 1, dev_c->element, 1);
}

/* c := 0 */
__device__ void _bncuda_set0_fvector(cublasHandle_t handle, FVector dev_c)
{
	long int i;

#ifndef __CUDACC__
	cudaMemset((void *)(dev_c->element), 0, sizeof(float) * (int)(dev_c->dim));
#else
	for(i = 0; i < dev_c->dim; i++)
		_bncuda_set_fvector_i(dev_c, i, (float)0);
#endif
}

/*************************************************/
/* Vector Calculations for DVector               */
/*
void _bncuda_add_dvector(DVector c, DVector a, DVector b)
void _bncuda_add2_dvector(DVector c, DVector a)
void _bncuda_sub_dvector(DVector c, DVector a, DVector b)
void _bncuda_sub2_dvector(DVector c, DVector a)
void _bncuda_cmul_dvector(DVector c, double val, DVector a)
void _bncuda_cmul2_dvector(DVector c, double val)
void _bncuda_add_cmul_dvector(handle, DVector c, DVector a, double val, DVector b)
double _bncuda_ip_dvector(DVector a, DVector b)
double _bncuda_norm1_dvector(DVector a)
double _bncuda_norm2_dvector(DVector a)
double _bncuda_normi_dvector(DVector a)
void _bncuda_subst_dvector(DVector c, DVector a)
*/
/*************************************************/
/* c = a + b */
__device__ void _bncuda_add_dvector(cublasHandle_t handle, DVector dev_c, DVector dev_a, DVector dev_b)
{
//	long int i;

#ifndef __CUDACC__
	if((dev_a->dim != dev_b->dim) || (dev_c->dim != dev_a->dim) || (dev_c->dim != dev_b->dim))
	{
		fprintf(stderr, "ERROR: _bncuda_add_dvector\n");
		return;
	}
#endif

	// dev_c := dev_b;
	cublasDcopy(handle, (int)(dev_c->dim), dev_b->element, 1, dev_c->element, 1);
	cublasDaxpy(handle, (int)(dev_c->dim), _bncuda_const_d_one, dev_a->element, 1, dev_c->element, 1);
}

/* c += a */
__device__ void _bncuda_add2_dvector(cublasHandle_t handle, DVector dev_c, DVector dev_a)
{
//	long int i;

#ifndef __CUDACC__
	if(dev_c->dim != dev_a->dim)
	{
		fprintf(stderr, "ERROR: _bncuda_add2_dvector\n");
		return;
	}
#endif

	cublasDaxpy(handle, (int)(dev_c->dim), _bncuda_const_d_one, dev_a->element, 1, dev_c->element, 1);
}

/* c = a - b */
__device__ void _bncuda_sub_dvector(cublasHandle_t handle, DVector dev_c, DVector dev_a, DVector dev_b)
{
//	long int i;

#ifndef __CUDACC__
	if((dev_a->dim != dev_b->dim) || (dev_c->dim != dev_a->dim) || (dev_c->dim != dev_b->dim))
	{
		fprintf(stderr, "ERROR: _bncuda_sub_dvector\n");
		return;
	}
#endif

	// dev_c := dev_a;
	cublasDcopy(handle, (int)(dev_c->dim), dev_a->element, 1, dev_c->element, 1);
	cublasDaxpy(handle, (int)(dev_c->dim), _bncuda_const_d_minus_one, dev_b->element, 1, dev_c->element, 1);
}

/* c -= a */
__device__ void _bncuda_sub2_dvector(cublasHandle_t handle, DVector dev_c, DVector dev_a)
{
//	long int i;

#ifndef __CUDACC__
	if(dev_c->dim != dev_a->dim)
	{
		fprintf(stderr, "ERROR: _bncuda_sub2_dvector\n");
		return;
	}
#endif

	cublasDaxpy(handle, (int)(dev_c->dim), _bncuda_const_d_minus_one, dev_a->element, 1, dev_c->element, 1);
}

/* c = val * a */
__device__ void _bncuda_cmul_dvector(cublasHandle_t handle, DVector dev_c, double *val, DVector dev_a)
{
//	long int i;

#ifndef __CUDACC__
	if(dev_c->dim != dev_a->dim)
	{
		fprintf(stderr, "ERROR: _bncuda_cmul_dvector\n");
		return;
	}
#endif

//	for(i = 0; i < c->dim; i++)
//		_bncuda_set_dvector_i(c, i, val * _bncuda_get_dvector_i(a, i));

	// dev_c := dev_a;
	cublasDcopy(handle, (int)(dev_c->dim), dev_a->element, 1, dev_c->element, 1);
	cublasDscal(handle, (int)(dev_c->dim), val, dev_c->element, 1);
}

/* c = val * a */
__device__ void _bncuda_cpmul_dvector(cublasHandle_t handle, DVector dev_c, double *val, DVector dev_a)
{
//	long int i;

#ifndef __CUDACC__
	if(dev_c->dim != dev_a->dim)
	{
		fprintf(stderr, "ERROR: _bncuda_cpmul_dvector\n");
		return;
	}
#endif

//	for(i = 0; i < c->dim; i++)
//		_bncuda_set_dvector_i(c, i, val * _bncuda_get_dvector_i(a, i));

	// dev_c := dev_a;
	cublasDcopy(handle, (int)(dev_c->dim), dev_a->element, 1, dev_c->element, 1);
	cublasDscal(handle, (int)(dev_c->dim), val, dev_c->element, 1);

}

/* c *= val */
__device__ void _bncuda_cmul2_dvector(cublasHandle_t handle, DVector dev_c, double *val)
{
//	long int i;

//	for(i = 0; i < c->dim; i++)
//		_bncuda_set_dvector_i(c, i, val * _bncuda_get_dvector_i(c, i));

	cublasDscal(handle, (int)(dev_c->dim), val, dev_c->element, 1);
}

/* c *= val */
__device__ void _bncuda_cpmul2_dvector(cublasHandle_t handle, DVector dev_c, double *val)
{
//	long int i;

//	for(i = 0; i < c->dim; i++)
//		_bncuda_set_dvector_i(c, i, val * _bncuda_get_dvector_i(c, i));

	cublasDscal(handle, (int)(dev_c->dim), val, dev_c->element, 1);

}
/* c = a + val * b */
__device__ void _bncuda_add_cmul_dvector(cublasHandle_t handle, DVector dev_c, DVector dev_a, double *val, DVector dev_b)
{
#ifndef __CUDACC__
	if((dev_a->dim != dev_b->dim) || (dev_c->dim != dev_a->dim) || (dev_c->dim != dev_b->dim))
	{
		fprintf(stderr, "ERROR: _bncuda_add_cmul_dvector\n");
		return;
	}
#endif

//	for(i = 0; i < c->dim; i++)
//		set_dvector_i(c, i, get_dvector_i(a, i) + val * get_dvector_i(b, i));

	// dev_c := dev_a;
	cublasDcopy(handle, (int)(dev_c->dim), dev_a->element, 1, dev_c->element, 1);
	cublasDaxpy(handle, (int)(dev_c->dim), val, dev_b->element, 1, dev_c->element, 1);
}

/* c = c + val * a */
__device__ void _bncuda_add_cmul2_dvector(cublasHandle_t handle, DVector dev_c, double *val, DVector dev_a)
{
#ifndef __CUDACC__
	if(dev_c->dim != dev_a->dim)
	{
		fprintf(stderr, "ERROR: _bncuda_add_cmul2_dvector\n");
		return;
	}
#endif

//	for(i = 0; i < c->dim; i++)
//		set_dvector_i(c, i, get_dvector_i(a, i) + val * get_dvector_i(b, i));

// dev_c := dev_c + val * dev_a;
	cublasDaxpy(handle, (int)(dev_c->dim), val, dev_a->element, 1, dev_c->element, 1);
}


/* (a, b) */
__device__ void _bncuda_ip_dvector(cublasHandle_t handle, double *ret, DVector dev_a, DVector dev_b)
{
//	long int i;

#ifndef __CUDACC__
	if(dev_a->dim != dev_b->dim)
	{
		fprintf(stderr, "ERROR: _bncuda_ip_dvector\n");
		return;
	}
#endif

//	for(i = 0; i < a->dim; i++)
//		tmp += _bncuda_get_dvector_i(a, i) * _bncuda_get_dvector_i(b, i);

	//*ret = 0.0;	
	cublasDdot(handle, (int)(dev_a->dim), dev_a->element, 1, dev_b->element, 1, ret); 

	return;
}

/* ||a||_1 */
__device__ void _bncuda_norm1_dvector(cublasHandle_t handle, double *ret, DVector dev_a)
{
//	double ret = 0.0;
//	long int i;

//	for(i = 0; i < a->dim; i++)
//		ret += fabs((double)_bncuda_get_dvector_i(a, i));

	cublasDasum(handle, (int)(dev_a->dim), dev_a->element, 1, ret);

	return;
}

/* ||a||_2 */
__device__ void _bncuda_norm2_dvector(cublasHandle_t handle, double *ret, DVector dev_a)
{
//	double ret = 0.0;
//	long int i;

//	for(i = 0; i < a->dim; i++)
//		ret += _bncuda_get_dvector_i(a, i) * _bncuda_get_dvector_i(a, i);

	cublasDnrm2(handle, (int)(dev_a->dim), dev_a->element, 1, ret);

	return;
}

/* ||a||_infty */
__device__ void _bncuda_normi_dvector(cublasHandle_t handle, double *ret, DVector dev_a)
{
//	double ret, tmp;
//	long int i;
	double tmp;
	int tmp_i, *dev_tmp_i;
#ifdef USE_POINTER_MODE_DEVICE
	dev_tmp_i = _bncuda_init_i();
#else // USE_POINTER_MODE_HOST
	dev_tmp_i = &tmp_i;
#endif

//	ret = fabs(_bncuda_get_dvector_i(a, 0));
//	for(i = 1; i < a->dim; i++)
//	{
//		tmp = fabs((double)_bncuda_get_dvector_i(a, i));
//		if(ret < tmp)
//			ret = tmp;
//	}

	cublasIdamax(handle, (int)(dev_a->dim), dev_a->element, 1, dev_tmp_i);
//	*ret = fabs(*(dev_a->element + dev_tmp_i)); // correct ?
#ifdef USE_POINTER_MODE_DEVICE
	tmp_i = _bncuda_get_i(dev_tmp_i);
#endif	

	tmp = fabs(_bncuda_get_dvector_i(dev_a, tmp_i));

#ifdef USE_POINTER_MODE_DEVICE
	_bncuda_set_d(ret, tmp);
	_bncuda_free_i(dev_tmp_i);
#else // USE_POINTER_MODE_HOST
	*ret = tmp;
#endif

	return;
}

/* c := a */
__device__ void _bncuda_subst_dvector(cublasHandle_t handle, DVector dev_c, DVector dev_a)
{
//	long int i;

//	for(i = 0; i < a->dim; i++)
//		_bncuda_set_dvector_i(c, i, _bncuda_get_dvector_i(a, i));
	cublasDcopy(handle, (int)(dev_c->dim), dev_a->element, 1, dev_c->element, 1);
}

/* c := 0 */
__device__ void _bncuda_set0_dvector(cublasHandle_t handle, DVector dev_c)
{
	long int i;

	for(i = 0; i < dev_c->dim; i++)
		_bncuda_set_dvector_i(dev_c, i, (double)0);
}

/* append 2005.07/12 */
/*
	ret(index_start) = src(src_index_start)
	 ...
	ret(index_end  ) = src(src_index_end)
*/
__device__ void _bncuda_copy_dvector_ij(cublasHandle_t handle, DVector ret, long int index_start, long int index_end, DVector src, long int src_index_start, long int src_index_end)
{
	long int i;

#ifndef __CUDACC__
	if((src_index_end - src_index_start) != (index_end - index_start))
	{
		fprintf(stderr, "Invalid index!(copy_dvector_ij)\n");
		return;
	}
#endif

//	for(i = 0; i <= (index_end - index_start); i++)
//	{
//		_bncuda_set_dvector_i(ret, index_start + i, _bncuda_get_dvector_i(src, src_index_start + i));
//		printf("%d <----------------------------------> %d\n", index_start + i, src_index_start + i);
//	}
	cublasDcopy(handle, (int)(index_end - index_start), &(ret->element[index_start]), 1, &(src->element[src_index_start]), 1);
}

/*************************************************/
/* Matrix Caluculations for FMatrix              */
/*
void _bncuda_add_fmatrix(FMatrix c, FMatrix a, FMatrix b);
void _bncuda_sub_fmatrix(FMatrix c, FMatrix a, FMatrix b);
void _bncuda_mul_fmatrix(FMatrix c, FMatrix a, FMatrix b);
void _bncuda_transpose_fmatrix(FMatrix c, FMatrix a);
void _bncuda_inv_fmatrix(FMatrix a);
void _bncuda_subst_fmatrux(FMatrix c, FMatrix a);
*/
/*************************************************/
/* dev_c = dev_a + dev_c */
__device__ void _bncuda_add_fmatrix(cublasHandle_t handle, FMatrix dev_c, FMatrix dev_a, FMatrix dev_b)
{
#if __CUDA_API_VERSION < 5000
	long int i, j, row_dim, col_dim;
#endif
//	float alpha, beta;

#ifndef __CUDACC__
	/* check row_dim */
	if((dev_a->row_dim != dev_b->row_dim) || (dev_b->row_dim != dev_c->row_dim) || (dev_c->row_dim != dev_a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncuda_add_fmatrix\n");
		return;
	}

	/* check col_dim */
	if((dev_a->col_dim != dev_b->col_dim) || (dev_b->col_dim != dev_c->col_dim) || (dev_c->col_dim != dev_a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncuda_add_fmatrix\n");
		return;
	}
#endif

#if __CUDA_API_VERSION < 5000
	row_dim = dev_c->row_dim;
	col_dim = dev_c->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
			_bncuda_set_fmatrix_ij(dev_c, i, j, _bncuda_get_fmatrix_ij(dev_a, i, j) + _bncuda_get_fmatrix_ij(dev_b, i, j));
	}
#else
//	alpha = 1.0f;
//	beta = 1.0f;
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)(dev_c->row_dim), (int)(dev_c->col_dim), _bncuda_const_f_one, dev_a->element, (int)(dev_a->row_dim), _bncuda_const_f_one, dev_b->element, (int)(dev_b->row_dim), dev_c->element, (int)(dev_c->row_dim));
#endif
}

/* dev_c = dev_a - dev_b */
__device__ void _bncuda_sub_fmatrix(cublasHandle_t handle, FMatrix dev_c, FMatrix dev_a, FMatrix dev_b)
{
#if __CUDA_API_VERSION < 5000
	long int i, j, row_dim, col_dim;
#endif
//	float alpha, beta;

#ifndef __CUDACC__
	/* check row_dim */
	if((dev_a->row_dim != dev_b->row_dim) || (dev_b->row_dim != dev_c->row_dim) || (dev_c->row_dim != dev_a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncuda_sub_fmatrix\n");
		return;
	}

	/* check col_dim */
	if((dev_a->col_dim != dev_b->col_dim) || (dev_b->col_dim != dev_c->col_dim) || (dev_c->col_dim != dev_a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncuda_sub_fmatrix\n");
		return;
	}
#endif

#if __CUDA_API_VERSION < 5000
	row_dim = dev_c->row_dim;
	col_dim = dev_c->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
			_bncuda_set_fmatrix_ij(dev_c, i, j, _bncuda_get_fmatrix_ij(dev_a, i, j) - _bncuda_get_fmatrix_ij(dev_b, i, j));
	}
#else
//	alpha = 1.0f;
//	beta = -1.0f;
	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)(dev_c->row_dim), (int)(dev_c->col_dim), _bncuda_const_f_one, dev_a->element, (int)(dev_a->row_dim), _bncuda_const_f_minus_one, dev_b->element, (int)(dev_b->row_dim), dev_c->element, (int)(dev_c->row_dim));
#endif
}


/* dev_c = sc * dev_a */
__device__ void _bncuda_cmul_fmatrix(cublasHandle_t handle, FMatrix dev_c, float *sc, FMatrix dev_a)
{
#ifndef __CUDACC__
	/* check row_dim */
	if(dev_a->row_dim != dev_c->row_dim)
	{
		fprintf(stderr, "ERROR: _bncuda_cmul_fmatrix\n");
		return;
	}

	/* check col_dim */
	if(dev_a->col_dim != dev_c->col_dim)
	{
		fprintf(stderr, "ERROR: _bncuda_cmul_fmatrix\n");
		return;
	}
#endif

	// dev_c := dev_a;
	cublasScopy(handle, (int)(dev_c->row_dim * dev_c->col_dim), dev_a->element, 1, dev_c->element, 1);
	cublasSscal(handle, (int)(dev_c->row_dim * dev_c->col_dim), sc, dev_c->element, 1);
}

/* dev_c = dev_a * dev_b */
__device__ void _bncuda_mul_fmatrix(cublasHandle_t handle, FMatrix dev_c, FMatrix dev_a, FMatrix dev_b)
{
	//float alpha, beta;

#ifndef __CUDACC__
	/* dimension check */
	if((dev_c->row_dim != dev_a->row_dim) || (dev_c->col_dim != dev_b->row_dim) || (dev_a->col_dim != dev_b->row_dim))
	{
		fprintf(stderr, "ERROR: _bncuda_mul_fmatrix\n");
		return;
	}
#endif

	//alpha = 1.0f;
	//beta = 0.0f;
	//cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)(dev_a->row_dim), (int)(dev_a->col_dim), (int)(dev_b->col_dim), &alpha, dev_a->element, (int)(dev_a->row_dim), dev_b->element, (int)(dev_b->row_dim), &beta, dev_c->element, (int)(dev_c->row_dim));
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)(dev_a->row_dim), (int)(dev_a->col_dim), (int)(dev_b->col_dim), _bncuda_const_f_one, dev_a->element, (int)(dev_a->row_dim), dev_b->element, (int)(dev_b->row_dim), _bncuda_const_f_zero, dev_c->element, (int)(dev_c->row_dim));
}

/* dev_c = dev_a^T */
__device__ void _bncuda_transpose_fmatrix(cublasHandle_t handle, FMatrix dev_c, FMatrix dev_a)
{
#if __CUDA_API_VERSION < 5000
	long int i, j;
#endif
//	float alpha, beta;

#ifndef __CUDACC__
	/* Check Dimentions */
	if((dev_c->row_dim != dev_a->col_dim) || (dev_c->col_dim != dev_a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncuda_transpose_fmatrix\n");
		return;
	}
#endif
	
#if __CUDA_API_VERSION < 5000
	for(i = 0; i < dev_c->row_dim; i++)
		for(j = 0; j < dev_c->col_dim; j++)
			_bncuda_set_fmatrix_ij(dev_c, i, j, _bncuda_get_fmatrix_ij(dev_a, j, i));

#else
//	alpha = 1.0f;
//	beta = 0.0f;
//	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)(dev_c->row_dim), (int)(dev_c->col_dim), &alpha, dev_a->element, (int)(dev_a->row_dim), &beta, dev_a->element, (int)(dev_a->row_dim), dev_c->element, (int)(dev_c->row_dim));
	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)(dev_c->row_dim), (int)(dev_c->col_dim), _bncuda_const_f_one, dev_a->element, (int)(dev_a->row_dim), _bncuda_const_f_zero, dev_a->element, (int)(dev_a->row_dim), dev_c->element, (int)(dev_c->row_dim));
#endif
}

/* dev_c := dev_a */
__device__ void _bncuda_subst_fmatrix(cublasHandle_t handle, FMatrix dev_c, FMatrix dev_a)
{
//	long int i, j;
//	float tmp;

#ifndef __CUDACC__
	if((dev_c->row_dim != dev_a->row_dim) || (dev_c->col_dim != dev_a->col_dim))
	{
		fprintf(stderr, "ERROR: _subst_subst_fmatrix\n");
		return;
	}
#endif

	cublasScopy(handle, (int)(dev_c->row_dim * dev_c->col_dim), dev_a->element, 1, dev_c->element, 1);
}

/* dev_c := 0 */
__device__ void _bncuda_set0_fmatrix(cublasHandle_t handle, FMatrix dev_c)
{
#if __CUDA_API_VERSION < 5000
	long int i, j;
#endif
//	float alpha, beta;

#if __CUDA_API_VERSION < 5000
	for(i = 0; i < dev_c->row_dim; i++)
		for(j = 0; j < dev_c->col_dim; j++)
			_bncuda_set_fmatrix_ij(dev_c, i, j, 0.0f);
#else
//	alpha = 0.0f;
//	beta = 0.0f;
	cudaMemset((void *)(dev_c->element), 0, sizeof(float) * (int)dev_c->row_col * (int)dev_c->col_dim);
//	cublasSgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)(dev_c->row_dim), (int)(dev_c->col_dim), &alpha, dev_c->element, (int)(dev_c->row_dim), &beta, dev_c->element, (int)(dev_c->row_dim), dev_c->element, (int)(dev_c->row_dim));
#endif
}

/* dev_c := I */
__device__ void _bncuda_setI_fmatrix(cublasHandle_t handle, FMatrix dev_c)
{
//	long int i, j;

#ifndef __CUDACC__
	cudaMemset((void *)(dev_c->element), 0, sizeof(float) * (int)dev_c->row_dim * (int)dev_c->col_dim);
#else
	long int i;
	for(i = 0; i < dev_c->row_dim; i++)
		_bncuda_set_fmatrix_ij(dev_c, i, i, 1.0);
#endif
}

/* v = dev_a * vb */
__device__ void _bncuda_mul_fmatrix_fvec(cublasHandle_t handle, FVector dev_v, FMatrix dev_a, FVector dev_vb)
{
//	long int i, j;
//	float alpha, beta;

#ifndef __CUDACC__
	/* Check Dimension */
	if((dev_v->dim != dev_a->row_dim) || (dev_vb->dim != dev_a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncuda_mul_fmatrix_fvec\n");
		return;
	}
#endif

//	alpha = 1.0f;
//	beta = 0.0f;
//	cublasSgemv(handle, CUBLAS_OP_N, (int)(dev_a->row_dim), (int)(dev_a->col_dim), &alpha, dev_a->element, (int)(dev_a->row_dim), dev_vb->element, 1, &beta, dev_v->element, 1);
	cublasSgemv(handle, CUBLAS_OP_N, (int)(dev_a->row_dim), (int)(dev_a->col_dim), _bncuda_const_f_one, dev_a->element, (int)(dev_a->row_dim), dev_vb->element, 1, _bncuda_const_f_zero, dev_v->element, 1);
}

/* dev_a = dev_a^(-1) */
/* square matrix only */
__device__ void _bncuda_inv_fmatrix(cublasHandle_t handle, FMatrix dev_a)
{
	long int i, j, k, dim;
	float aii;

#ifndef __CUDACC__
	/* Check Dimensions */
	if(dev_a->row_dim != dev_a->col_dim)
	{
		fprintf(stderr, "ERROR: _bncuda_inv_fmatrix\n");
		return;
	}
#endif

	dim = dev_a->row_dim;

	for(i = 0; i < dim; i++)
	{
		if(_bncuda_get_fmatrix_ij(dev_a, i, i) == 0.0)
		{
#ifndef __CUDACC__
			fprintf(stderr, "ERROR: inv_fmatrix: Pivot(%ld,%ld) is zero.\n", i, i);
#endif
			return;
		}

		aii = 1.0 / _bncuda_get_fmatrix_ij(dev_a, i, i);
		_bncuda_set_fmatrix_ij(dev_a, i, i, aii);

		for(j = 0; j < i; j++)
			_bncuda_set_fmatrix_ij(dev_a, i, j, _bncuda_get_fmatrix_ij(dev_a, i, j)* aii);
		for(j = i + 1; j < dim; j++)
			_bncuda_set_fmatrix_ij(dev_a, i, j, _bncuda_get_fmatrix_ij(dev_a, i, j) * aii);

		for(j = 0; j < i; j++)
		{
			for(k = 0; k < i; k++)
				_bncuda_set_fmatrix_ij(dev_a, j, k, _bncuda_get_fmatrix_ij(dev_a, j, k) - _bncuda_get_fmatrix_ij(dev_a, j, i) * _bncuda_get_fmatrix_ij(dev_a, i, k));
			for(k = i + 1; k < dim; k++)
				_bncuda_set_fmatrix_ij(dev_a, j, k, _bncuda_get_fmatrix_ij(dev_a, j, k) - _bncuda_get_fmatrix_ij(dev_a, j, i) * _bncuda_get_fmatrix_ij(dev_a, i, k));
		}

		for(j = i + 1; j < dim; j++)
		{
			for(k = 0; k < i; k++)
				_bncuda_set_fmatrix_ij(dev_a, j, k, _bncuda_get_fmatrix_ij(dev_a, j, k) - _bncuda_get_fmatrix_ij(dev_a, j, i) * _bncuda_get_fmatrix_ij(dev_a, i, k));
			for(k = i + 1; k < dim; k++)
				_bncuda_set_fmatrix_ij(dev_a, j, k, _bncuda_get_fmatrix_ij(dev_a, j, k) - _bncuda_get_fmatrix_ij(dev_a, j, i) * _bncuda_get_fmatrix_ij(dev_a, i, k));
		}

		for(j = 0; j < i; j++)
			_bncuda_set_fmatrix_ij(dev_a, j, i, _bncuda_get_fmatrix_ij(dev_a, j, i) * -aii);
		for(j = i + 1; j < dim; j++)
			_bncuda_set_fmatrix_ij(dev_a, j, i, _bncuda_get_fmatrix_ij(dev_a, j, i) * -aii);	}
}

/*************************************************/
/* Matrix Caluculations for DMatrix              */
/*
void _bncuda_normf_dmatrix(DMatrix mat)
void _bncuda_add_dmatrix(DMatrix c, DMatrix a, DMatrix b);
void _bncuda_sub_dmatrix(DMatrix c, DMatrix a, DMatrix b);
void _bncuda_mul_dmatrix(DMatrix c, DMatrix a, DMatrix b);
void _bncuda_transpose_dmatrix(DMatrix c, DMatrix a);
void _bncuda_mul_dmatrix_dvec(DVector v, DMatrix a, DVector vb)
void _bncuda_mul_dmatrixt_dvec(DVector v, DMatrix a, DVector vb)
void _bncuda_inv_dmatrix(DMatrix a);
void _bncuda_subst_dmatrux(DMatrix c, DMatrix a);
*/
/*************************************************/
/* Frobenius Norm of Matrix */
__device__ void _bncuda_normf_dmatrix(cublasHandle_t handle, double *ret, DMatrix dev_mat)
{
	cublasDnrm2(handle, (int)(dev_mat->row_dim * dev_mat->col_dim), dev_mat->element, 1, ret);
}

/* dev_c = dev_a + dev_c */
__device__ void _bncuda_add_dmatrix(cublasHandle_t handle, DMatrix dev_c, DMatrix dev_a, DMatrix dev_b)
{
#if __CUDA_API_VERSION < 5000 
	long int i, j, row_dim, col_dim;
#endif
//	double alpha, beta;

#ifndef __CUDACC__
	/* check row_dim */
	if((dev_a->row_dim != dev_b->row_dim) || (dev_b->row_dim != dev_c->row_dim) || (dev_c->row_dim != dev_a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncuda_add_dmatrix\n");
		return;
	}

	/* check col_dim */
	if((dev_a->col_dim != dev_b->col_dim) || (dev_b->col_dim != dev_c->col_dim) || (dev_c->col_dim != dev_a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncuda_add_dmatrix\n");
		return;
	}
#endif

#if __CUDA_API_VERSION < 5000
	row_dim = dev_c->row_dim;
	col_dim = dev_c->col_dim;
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
			_bncuda_set_dmatrix_ij(dev_c, i, j, _bncuda_get_dmatrix_ij(dev_a, i, j) + _bncuda_get_dmatrix_ij(dev_b, i, j));
	}
#else
//	alpha = 1.0;
//	beta = 1.0;
	//cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)(dev_c->row_dim), (int)(dev_c->col_dim), &alpha, dev_a->element, (int)(dev_a->row_dim), &beta, dev_b->element, (int)(dev_b->row_dim), dev_c->element, (int)(dev_c->row_dim));
	cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)(dev_c->row_dim), (int)(dev_c->col_dim), _bncuda_const_d_one, dev_a->element, (int)(dev_a->row_dim), _bncuda_const_d_one, dev_b->element, (int)(dev_b->row_dim), dev_c->element, (int)(dev_c->row_dim));
#endif
}

/* dev_c = dev_a - dev_b */
__device__ void _bncuda_sub_dmatrix(cublasHandle_t handle, DMatrix dev_c, DMatrix dev_a, DMatrix dev_b)
{
#if __CUDA_API_VERSION < 5000 
	long int i, j, row_dim, col_dim;
#endif
//	double alpha, beta;

#ifndef __CUDACC__
	/* check row_dim */
	if((dev_a->row_dim != dev_b->row_dim) || (dev_b->row_dim != dev_c->row_dim) || (dev_c->row_dim != dev_a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncuda_sub_dmatrix\n");
		return;
	}

	/* check col_dim */
	if((dev_a->col_dim != dev_b->col_dim) || (dev_b->col_dim != dev_c->col_dim) || (dev_c->col_dim != dev_a->col_dim))
	{
		fprintf(stderr, "ERROR: _bncuda_sub_dmatrix\n");
		return;
	}
#endif

#if __CUDA_API_VERSION < 5000
	row_dim = dev_c->row_dim;
	col_dim = dev_c->col_dim;
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
			_bncuda_set_dmatrix_ij(dev_c, i, j, _bncuda_get_dmatrix_ij(dev_a, i, j) + _bncuda_get_dmatrix_ij(dev_b, i, j));
	}
#else
//	alpha = 1.0;
//	beta = -1.0;
//	cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)(dev_c->row_dim), (int)(dev_c->col_dim), &alpha, dev_a->element, (int)(dev_a->row_dim), &beta, dev_b->element, (int)(dev_b->row_dim), dev_c->element, (int)(dev_c->row_dim));
	cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)(dev_c->row_dim), (int)(dev_c->col_dim), _bncuda_const_d_one, dev_a->element, (int)(dev_a->row_dim), _bncuda_const_d_minus_one, dev_b->element, (int)(dev_b->row_dim), dev_c->element, (int)(dev_c->row_dim));
#endif
}


/* dev_c = sc * dev_a */
__device__ void _bncuda_cmul_dmatrix(cublasHandle_t handle, DMatrix dev_c, double *sc, DMatrix dev_a)
{
#ifndef __CUDACC__
	/* check row_dim */
	if(dev_a->row_dim != dev_c->row_dim)
	{
		fprintf(stderr, "ERROR: _bncuda_cmul_dmatrix\n");
		return;
	}

	/* check col_dim */
	if(dev_a->col_dim != dev_c->col_dim)
	{
		fprintf(stderr, "ERROR: _bncuda_cmul_dmatrix\n");
		return;
	}
#endif

	// dev_c := dev_a;
	cublasDcopy(handle, (int)(dev_c->row_dim * dev_c->col_dim), dev_a->element, 1, dev_c->element, 1);
	cublasDscal(handle, (int)(dev_c->row_dim * dev_c->col_dim), sc, dev_c->element, 1);
}

/* dev_c = dev_a * dev_b */
__device__ void _bncuda_mul_dmatrix(cublasHandle_t handle, DMatrix dev_c, DMatrix dev_a, DMatrix dev_b)
{
	//double alpha, beta;

#ifndef __CUDACC__
	/* dimension check */
	if((dev_c->row_dim != dev_a->row_dim) || (dev_c->col_dim != dev_b->row_dim) || (dev_a->col_dim != dev_b->row_dim))
	{
		fprintf(stderr, "ERROR: _bncuda_mul_dmatrix\n");
		return;
	}
#endif

//	alpha = 1.0;
//	beta = 0.0;
	//cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)(dev_a->row_dim), (int)(dev_a->col_dim), (int)(dev_b->col_dim), &alpha, dev_a->element, (int)(dev_a->row_dim), dev_b->element, (int)(dev_b->row_dim), &beta, dev_c->element, (int)(dev_c->row_dim));
	cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)(dev_a->row_dim), (int)(dev_a->col_dim), (int)(dev_b->col_dim), _bncuda_const_d_one, dev_a->element, (int)(dev_a->row_dim), dev_b->element, (int)(dev_b->row_dim), _bncuda_const_d_zero, dev_c->element, (int)(dev_c->row_dim));
}

/* dev_c = dev_a^T */
__device__ void _bncuda_transpose_dmatrix(cublasHandle_t handle, DMatrix dev_c, DMatrix dev_a)
{
#if __CUDA_API_VERSION < 5000
	long int i, j;
#endif
//	double alpha, beta;

#ifndef __CUDACC__
	/* Check Dimentions */
	if((dev_c->row_dim != dev_a->col_dim) || (dev_c->col_dim != dev_a->row_dim))
	{
		fprintf(stderr, "ERROR: _bncuda_transpose_dmatrix\n");
		return;
	}
#endif

#if __CUDA_API_VERSION < 5000
	for(i = 0; i < dev_c->row_dim; i++)
		for(j = 0; j < dev_c->col_dim; j++)
			_bncuda_set_dmatrix_ij(dev_c, i, j, _bncuda_get_dmatrix_ij(dev_a, j, i));
#else
//	alpha = 1.0;
//	beta = 0.0;
	//cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)(dev_c->row_dim), (int)(dev_c->col_dim), &alpha, dev_a->element, (int)(dev_a->row_dim), &beta, dev_a->element, (int)(dev_a->row_dim), dev_c->element, (int)(dev_c->row_dim));
	cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, (int)(dev_c->row_dim), (int)(dev_c->col_dim), _bncuda_const_d_one, dev_a->element, (int)(dev_a->row_dim), _bncuda_const_d_zero, dev_a->element, (int)(dev_a->row_dim), dev_c->element, (int)(dev_c->row_dim));
#endif
}

/* dev_c := dev_a */
__device__ void _bncuda_subst_dmatrix(cublasHandle_t handle, DMatrix dev_c, DMatrix dev_a)
{
//	long int i, j;
//	double tmp;

#ifndef __CUDACC__
	if((dev_c->row_dim != dev_a->row_dim) || (dev_c->col_dim != dev_a->col_dim))
	{
		fprintf(stderr, "ERROR: _subst_subst_dmatrix\n");
		return;
	}
#endif

	cublasDcopy(handle, (int)(dev_c->row_dim * dev_c->col_dim), dev_a->element, 1, dev_c->element, 1);
}

/* dev_c := 0 */
__device__ void _bncuda_set0_dmatrix(cublasHandle_t handle, DMatrix dev_c)
{
#if __CUDA_API_VERSION < 5000
	long int i, j;

	for(i = 0; i < dev_c->row_dim; i++)
		for(j = 0; j < dev_c->row_dim; j++)
			_bncuda_set_dmatrix_ij(dev_c, i, j, 0.0);
#else
//	double alpha, beta;

//	alpha = 0.0;
//	beta = 0.0;
#ifndef __CUDACC__
	cudaMemset((void *)(dev_c->element), 0, sizeof(double) * (int)dev_c->row_dim, (int)dev_c->col_dim);
#else
	cublasDgeam(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)(dev_c->row_dim), (int)(dev_c->col_dim), _bncuda_const_d_zero, dev_c->element, (int)(dev_c->row_dim), _bncuda_const_d_zero, dev_c->element, (int)(dev_c->row_dim), dev_c->element, (int)(dev_c->row_dim));
#endif
#endif
}

/* dev_c := I */
__device__ void _bncuda_setI_dmatrix(cublasHandle_t handle, DMatrix dev_c)
{
	long int i;

#ifndef __CUDACC__
	cudaMemset((void *)(dev_c->element), 0, sizeof(double) * (int)dev_c->row_dim * (int)dev_c->col_dim);
#else
	_bncuda_set0_dmatrix(handle, dev_c);
	for(i = 0; i < dev_c->row_dim; i++)
		_bncuda_set_dmatrix_ij(dev_c, i, i, 1.0);
#endif
}

/* v = a * vb */
// multiple mat * vec
__device__ void _bncuda_mul_dmatrix_dvec(cublasHandle_t handle, DVector ret, DMatrix dev_mat, DVector dev_vec)
{
//	double alpha, beta;

//	alpha = 1.0;
//	beta = 0.0;
	//cublasDgemv(handle, CUBLAS_OP_N, dev_mat->row_dim, dev_mat->col_dim, &alpha, dev_mat->element, dev_vec->dim, dev_vec->element, 1, &beta, ret->element, 1);
	cublasDgemv(handle, CUBLAS_OP_N, dev_mat->row_dim, dev_mat->col_dim, _bncuda_const_d_one, dev_mat->element, dev_vec->dim, dev_vec->element, 1, _bncuda_const_d_zero, ret->element, 1);
}

/* v = a^T * vb */
__device__ void _bncuda_mul_dmatrixt_dvec(cublasHandle_t handle, DVector ret, DMatrix dev_mat, DVector dev_vec)
{
//	double alpha, beta;

//	alpha = 1.0;
//	beta = 0.0;
	//cublasDgemv(handle, CUBLAS_OP_T, dev_mat->row_dim, dev_mat->col_dim, &alpha, dev_mat->element, dev_vec->dim, dev_vec->element, 1, &beta, ret->element, 1);
	cublasDgemv(handle, CUBLAS_OP_T, dev_mat->row_dim, dev_mat->col_dim, _bncuda_const_d_one, dev_mat->element, dev_vec->dim, dev_vec->element, 1, _bncuda_const_d_zero, ret->element, 1);
}

/* a = a^(-1) */
/* square matrix only */
__device__ void _bncuda_inv_dmatrix(cublasHandle_t handle, DMatrix a)
{
	long int i, j, k, dim;
	double aii;

#ifndef __CUDACC__
	/* Check Dimensions */
	if(a->row_dim != a->col_dim)
	{
		fprintf(stderr, "ERROR: inv_dmatrix\n");
		return;
	}
#endif

	dim = a->row_dim;

	for(i = 0; i < dim; i++)
	{
		if(_bncuda_get_dmatrix_ij(a, i, i) == 0.0)
		{
#ifndef __CUDACC__
			fprintf(stderr, "ERROR: inv_dmatrix: Pivot(%ld,%ld) is zero.\n", i, i);
#endif
			return;
		}

		aii = 1.0 / _bncuda_get_dmatrix_ij(a, i, i);
		_bncuda_set_dmatrix_ij(a, i, i, aii);

		for(j = 0; j < i; j++)
			_bncuda_set_dmatrix_ij(a, i, j, _bncuda_get_dmatrix_ij(a, i, j) * aii);
		for(j = i + 1; j < dim; j++)
			_bncuda_set_dmatrix_ij(a, i, j, _bncuda_get_dmatrix_ij(a, i, j) * aii);

		for(j = 0; j < i; j++)
		{
			for(k = 0; k < i; k++)
				_bncuda_set_dmatrix_ij(a, j, k, _bncuda_get_dmatrix_ij(a, j, k) - _bncuda_get_dmatrix_ij(a, j, i) * _bncuda_get_dmatrix_ij(a, i, k));
			for(k = i + 1; k < dim; k++)
				_bncuda_set_dmatrix_ij(a, j, k, _bncuda_get_dmatrix_ij(a, j, k) - _bncuda_get_dmatrix_ij(a, j, i) * _bncuda_get_dmatrix_ij(a, i, k));
		}

		for(j = i + 1; j < dim; j++)
		{
			for(k = 0; k < i; k++)
				_bncuda_set_dmatrix_ij(a, j, k, _bncuda_get_dmatrix_ij(a, j, k) - _bncuda_get_dmatrix_ij(a, j, i) * _bncuda_get_dmatrix_ij(a, i, k));
			for(k = i + 1; k < dim; k++)
				_bncuda_set_dmatrix_ij(a, j, k, _bncuda_get_dmatrix_ij(a, j, k) - _bncuda_get_dmatrix_ij(a, j, i) * _bncuda_get_dmatrix_ij(a, i, k));
		}

		for(j = 0; j < i; j++)
			_bncuda_set_dmatrix_ij(a, j, i, _bncuda_get_dmatrix_ij(a, j, i) * -aii);
		for(j = i + 1; j < dim; j++)
			_bncuda_set_dmatrix_ij(a, j, i, _bncuda_get_dmatrix_ij(a, j, i) * -aii);	}
}

//#ifdef USE_KRYLOV
/************************************************************/
/*                                                          */
/* Bi-Conjugate-Gradient Method for Real Matrix             */
/*                                 (Double Precision)       */
/*                                 (cublas)       */
/*                                                          */
/*                 ver. 0.0 2013-01-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
__global__ void _bncuda_dev_DBiCG(cublasHandle_t handle, long int *ret_val, DVector dev_answer, DMatrix dev_a, DVector dev_b, double *pt_reps, double *pt_aeps, long int *pt_maxtimes, DVector dev_vec[8])

/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       DVector answer: Solution for Ax = b                */
/*       DMatrix a: Coefficient matrix A                    */
/*                                       (given by user)    */
/*       DVector b: Constant vector b   (given by user)     */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       DVector answer: Solution for Ax = b                */
/*                                                          */
/* ERRORS                                                   */
/* Positive value ... Normal : Iterative Times              */
/*      -1 ... Rho is zero.                                 */
/*      -2 ... Denominator of Alpha is zero.                */
/*      -3 ... (Reserved)                                   */
/*      -4 ... (Reserved)                                   */
/*      -5 ... Not Converge.                                */
/*                                                          */
/************************************************************/
{
	long int times, return_val; // Fix!
	long int maxtimes;

	double alpha, alpha_den, minus_alpha;
	double beta, beta_num;
	double rho, old_rho;
	double dtmp, init_resnorm;
	double reps, aeps;

	int id_alpha = 0, id_alpha_den = 1, id_minus_alpha = 2;
	int id_beta = 3, id_beta_num = 4;
	int id_rho = 5, id_old_rho = 6;
	int id_dtmp = 7, id_init_resnorm = 8;
	int id_reps = 9, id_aeps = 10;
	double *pt_dval[11];

#ifdef USE_POINTER_MODE_HOST
//	printf("DBiCG in HOST MODE\n");
	pt_dval[id_alpha] = &alpha;
	pt_dval[id_alpha_den] = &alpha_den;
	pt_dval[id_minus_alpha] = &minus_alpha;
	pt_dval[id_beta] = &beta;
	pt_dval[id_beta_num] = &beta_num;
	pt_dval[id_rho] = &rho;
	pt_dval[id_old_rho] = &old_rho;
	pt_dval[id_dtmp] = &dtmp;
	pt_dval[id_init_resnorm] = &init_resnorm;
#else //USE_POINTER_MODE_DEVICE
	int i;//	printf("DBiCG in DEVICE MODE\n");
	for(i = 0; i < 11; i++)
		pt_dval[i] = _bncuda_init_d();
#endif

//	DVector dev_vec[8]; /* Temporary Vectors */

#ifdef USE_POINTER_MODE_HOST
	maxtimes = *pt_maxtimes;
	reps = *pt_reps;
	aeps = *pt_aeps;
#else
	maxtimes = _bncuda_get_l(pt_maxtimes);
	reps = _bncuda_get_d(pt_reps);
	aeps = _bncuda_get_d(pt_aeps);
#endif
	
//	dim = dev_answer->dim;

/* Set initial value */
//	for(i = 0; i < 8; i++)
//		dev_vec[i] = _bncuda_init_dvector(dim);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0]  == w*/
	/* vec[2] ... (b - a * vec[0])^T  == wt */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... z */
	/* vec[6] ... z^T */

	_bncuda_subst_dvector(handle, dev_vec[1], dev_b); 
	_bncuda_subst_dvector(handle, dev_vec[2], dev_b);

	//_bncuda_ip_dvector(handle, &beta_num, dev_vec[1], dev_vec[1]);
	_bncuda_ip_dvector(handle, pt_dval[id_beta_num], dev_vec[1], dev_vec[1]);
#ifdef USE_POINTER_MODE_DEVICE
	beta_num = _bncuda_get_d(pt_dval[id_beta_num]);
#endif
	init_resnorm = sqrt(beta_num);

	old_rho = 0.0;
	rho = 0.0;

	return_val = 0; // Fix!

#ifndef __CUDACC__
	printf("start _bncuda_dev_DBiCG!\n");
#endif

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
//		_bncuda_ip_dvector(handle, &rho, dev_vec[2], dev_vec[1]);
		_bncuda_ip_dvector(handle, pt_dval[id_rho], dev_vec[2], dev_vec[1]);
#ifdef USE_POINTER_MODE_DEVICE
		//printf("times = %d\n", times);
		rho = _bncuda_get_d(pt_dval[id_rho]);
#endif

		if(rho == 0.0)
		{
#ifndef __CUDACC__
			fprintf(stderr, "Rho is zero!(_bncuda_DBiCG, %ld)\n", times);
#endif
			return_val = -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* p := w, pt := wt */
			_bncuda_subst_dvector(handle, dev_vec[3], dev_vec[1]);
			_bncuda_subst_dvector(handle, dev_vec[4], dev_vec[2]);
		}
		else
		{
			beta = rho / old_rho;

			/* p := w + beta p, pt := wt + beta * pt */
#ifdef USE_POINTER_MODE_DEVICE
			_bncuda_set_d(pt_dval[id_beta], beta);
#endif
			//_bncuda_add_cmul_dvector(handle, dev_vec[7], dev_vec[1], &beta, dev_vec[3]);
			_bncuda_add_cmul_dvector(handle, dev_vec[7], dev_vec[1], pt_dval[id_beta], dev_vec[3]);
			_bncuda_subst_dvector(handle, dev_vec[3], dev_vec[7]);

//			_bncuda_add_cmul_dvector(handle, dev_vec[7], dev_vec[2], &be	printf("device .. ip\n");ta, dev_vec[4]);
			_bncuda_add_cmul_dvector(handle, dev_vec[7], dev_vec[2], pt_dval[id_beta], dev_vec[4]);
			_bncuda_subst_dvector(handle, dev_vec[4], dev_vec[7]);
		}

		/* z := Ap, zt := A^T pt */
		_bncuda_mul_dmatrix_dvec(handle, dev_vec[5], dev_a, dev_vec[3]);
		_bncuda_mul_dmatrixt_dvec(handle, dev_vec[6], dev_a, dev_vec[4]);

		//_bncuda_ip_dvector(handle, &alpha_den, dev_vec[4], dev_vec[5]);
		_bncuda_ip_dvector(handle, pt_dval[id_alpha_den], dev_vec[4], dev_vec[5]);
#ifdef USE_POINTER_MODE_DEVICE		alpha_den = _bncuda_get_d(pt_dval[id_alpha_den]);
#endif
		if(alpha_den == 0.0)
		{
#ifndef __CUDACC__
			fprintf(stderr, "Denominator of Alpha is zero!(_bncuda_DBiCG, %ld)\n", times);
#endif
			return_val = -2; // Fix!
			break; // Fix!
		}
		alpha = rho / alpha_den;
		minus_alpha = -alpha;
#ifdef USE_POINTER_MODE_DEVICE
		_bncuda_set_d(pt_dval[id_alpha], alpha);
		_bncuda_set_d(pt_dval[id_minus_alpha], minus_alpha);
#endif
		/* x = x + alpha p */
//		_bncuda_add_cmul_dvector(handle, dev_vec[0], dev_vec[0], &alpha, dev_vec[3]);
		_bncuda_add_cmul_dvector(handle, dev_vec[0], dev_vec[0], pt_dval[id_alpha], dev_vec[3]);

		/* residual */
//		_bncuda_add_cmul_dvector(handle, dev_vec[1], dev_vec[1], &minus_alpha, dev_vec[5]);
//		_bncuda_add_cmul_dvector(handle, dev_vec[2], dev_vec[2], &minus_alpha, dev_vec[6]);
		_bncuda_add_cmul_dvector(handle, dev_vec[1], dev_vec[1], pt_dval[id_minus_alpha], dev_vec[5]);
		_bncuda_add_cmul_dvector(handle, dev_vec[2], dev_vec[2], pt_dval[id_minus_alpha], dev_vec[6]);

//		_bncuda_ip_dvector(handle, &beta_num, dev_vec[1], dev_vec[1]);
		_bncuda_ip_dvector(handle, pt_dval[id_beta_num], dev_vec[1], dev_vec[1]);

#ifdef USE_POINTER_MODE_DEVICE
		beta_num = _bncuda_get_d(pt_dval[id_beta_num]);
#endif

		/* Stopping Criteria */
		dtmp = sqrt(beta_num);
		if(dtmp <= aeps + reps * init_resnorm)
		{
			//_bncuda_subst_dvector(handle, dev_answer, dev_vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		old_rho = rho;
	}
#ifdef USE_POINTER_MODE_DEVICE
	for(i = 0; i < 11; i++)
		_bncuda_free_d_array(pt_dval[i]);
#endif

	//return return_val; // Fix!
#ifdef USE_POINTER_MODE_HOST
	*ret_val = return_val;
#else // USE_POINTER_MODE_DEVICE
	_bncuda_set_l(ret_val, return_val);
#endif


	/* Not converge */
	_bncuda_subst_dvector(handle, dev_answer, dev_vec[0]);

	/* free vec[0]..[7]; */
//	for(i = 0; i < 8; i++)
//		_bncuda_free_dvector(dev_vec[i]);

	// Fix!
	if(times >= maxtimes)
	{
#ifndef __CUDACC__
		fprintf(stderr, "Not converge!(_bncuda_DBiCG, %ld)\n", times);
#endif
		return_val = -5; // Fix!
	}

#ifdef USE_POINTER_MODE_DEVICE
	for(i = 0; i < 9; i++)
		_bncuda_free_d_array(pt_dval[i]);
#endif

	//return return_val; // Fix!
#ifdef USE_POINTER_MODE_HOST
	*ret_val = return_val;
#else // USE_POINTER_MODE_DEVICE
	_bncuda_set_l(ret_val, return_val);
#endif
	return;
}


//#ifdef USE_KRYLOV
// called from host directly
long int _bncuda_DBiCG(cublasHandle_t handle, DVector dev_answer, DMatrix dev_a, DVector dev_b, double reps, double aeps, long int maxtimes)
{
	long int i, dim;
	DVector dev_vec[8]; /* Temporary Vectors */
	double *dev_reps, *dev_aeps;
	long int *dev_maxtimes, *dev_ret_val, ret_val;

	dim = dev_answer->dim;

	/* Set initial value */
	for(i = 0; i < 8; i++)
		dev_vec[i] = _bncuda_init_dvector(dim);

#ifdef USE_POINTER_MODE_DEVICE
	printf("DEVICE MODE\n");
	dev_maxtimes = _bncuda_init_set_l(maxtimes);
	dev_reps = _bncuda_init_set_d(reps);
	dev_aeps = _bncuda_init_set_d(aeps);
	dev_ret_val = _bncuda_init_l();	
#else // USE_POINTER_MODE_HOST
	printf("HOST MODE\n");
	dev_maxtimes = &maxtimes;
	dev_reps = &reps;
	dev_aeps = &aeps;
	dev_ret_val = &ret_val;
#endif

	// call as global function
#if defined(__CUDACC__)
	printf("NVCC!\n");
	_bncuda_dev_DBiCG<<<1,1>>>(handle, dev_ret_val, dev_answer, dev_a, dev_b, dev_reps, dev_aeps, dev_maxtimes, dev_vec);
#else
//	printf("HOST MODE\n");
	printf("start calling _bncuda_dev_BiCG...\n");
	_bncuda_dev_DBiCG(handle, dev_ret_val, dev_answer, dev_a, dev_b, dev_reps, dev_aeps, dev_maxtimes, dev_vec);
#endif

	/* free vec[0]..[7]; */
	for(i = 0; i < 8; i++)
		_bncuda_free_dvector(dev_vec[i]);

#ifdef USE_POINTER_MODE_DEVICE
	ret_val = _bncuda_get_l(dev_ret_val);

	_bncuda_free_l(dev_maxtimes);
	_bncuda_free_d(dev_reps);
	_bncuda_free_d(dev_aeps);
	_bncuda_free_l(dev_ret_val);
#else  // USE_POINTER_MODE_HOST
	ret_val = *dev_ret_val;
#endif

	return ret_val;
}
//} // end of extern "C"

//#ifdef USE_KRYLOV
/************************************************************/
/*                                                          */
/*               CGS Method for Real Matrix                 */
/*                                 (Double Precision)       */
/*                                 (cublas)       */
/*                                                          */
/*                 ver. 0.0 2013-01-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
__global__ void _bncuda_dev_DCGS(cublasHandle_t handle, long int *ret_val, DVector dev_answer, DMatrix dev_a, DVector dev_b, double *pt_reps, double *pt_aeps, long int *pt_maxtimes, DVector dev_vec[9])
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       DVector answer: Solution for Ax = b                */
/*       DMatrix a: Coefficient matrix A                    */
/*                                       (given by user)    */
/*       DVector b: Constant vector b   (given by user)     */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       DVector answer: Solution for Ax = b                */
/*                                                          */
/* ERRORS                                                   */
/* Positive value ... Normal : Iterative Times              */
/*      -1 ... Rho is zero.                                 */
/*      -2 ... Denominator of Alpha is zero.                */
/*      -3 ... (Reserved)                                   */
/*      -4 ... (Reserved)                                   */
/*      -5 ... Not Converge.                                */
/*                                                          */
/************************************************************/
{
	long int times, return_val, maxtimes;
	double alpha, alpha_den, minus_alpha;
	double beta, beta_num;
	double rho, old_rho;
	double dtmp, init_resnorm;
	double reps, aeps;
//	DVector dev_vec[9]; /* Temporary Vectors */

	int id_alpha = 0, id_alpha_den = 1, id_minus_alpha = 2;
	int id_beta = 3, id_beta_num = 4;
	int id_rho = 5, id_old_rho = 6;
	int id_dtmp = 7, id_init_resnorm = 8;
	int id_reps = 9, id_aeps = 10;
	double *pt_dval[11];

#ifdef USE_POINTER_MODE_HOST
//	printf("DCGS in HOST MODE\n");
	pt_dval[id_alpha] = &alpha;
	pt_dval[id_alpha_den] = &alpha_den;
	pt_dval[id_minus_alpha] = &minus_alpha;
	pt_dval[id_beta] = &beta;
	pt_dval[id_beta_num] = &beta_num;
	pt_dval[id_rho] = &rho;
	pt_dval[id_old_rho] = &old_rho;
	pt_dval[id_dtmp] = &dtmp;
	pt_dval[id_init_resnorm] = &init_resnorm;
#else //USE_POINTER_MODE_DEVICE
	int i;//	printf("DCGS in DEVICE MODE\n");
	for(i = 0; i < 11; i++)
		pt_dval[i] = _bncuda_init_d();
#endif

//	DVector dev_vec[8]; /* Temporary Vectors */

#ifdef USE_POINTER_MODE_HOST
	maxtimes = *pt_maxtimes;
	reps = *pt_reps;
	aeps = *pt_aeps;
#else
	maxtimes = _bncuda_get_l(pt_maxtimes);
	reps = _bncuda_get_d(pt_reps);
	aeps = _bncuda_get_d(pt_aeps);
#endif

/* Set initial value */
//	for(i = 0; i < 9; i++)
//		dev_vec[i] = _bncuda_init_dvector(dim);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0] */
	/* vec[2] ... (b - a * vec[0])^T */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... u */
	/* vec[6] ... u^T */
	/* vec[7] ... q */
	/* vec[8] ... v^T */

	_bncuda_subst_dvector(handle, dev_vec[1], dev_b); 
	_bncuda_subst_dvector(handle, dev_vec[2], dev_b);

	//_bncuda_ip_dvector(handle, &beta_num, dev_vec[1], dev_vec[1]);
	_bncuda_ip_dvector(handle, pt_dval[id_beta_num], dev_vec[1], dev_vec[1]);
#ifdef USE_POINTER_MODE_DEVICE
	beta_num = _bncuda_get_d(pt_dval[id_beta_num]);
#endif
	init_resnorm = sqrt(beta_num);

	old_rho = 0.0;
	rho = 0.0;
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
//		_bncuda_ip_dvector(handle, &rho, dev_vec[2], dev_vec[1]);
		_bncuda_ip_dvector(handle, pt_dval[id_rho], dev_vec[2], dev_vec[1]);
#ifdef USE_POINTER_MODE_DEVICE
		rho = _bncuda_get_d(pt_dval[id_rho]);
#endif

		if(rho == 0.0)
		{
#ifndef __CUDACC__
			fprintf(stderr, "Rho is zero!(_bncuda_DCGS, %ld)\n", times);
#endif
			return_val = -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* u := rt, p := u */
			_bncuda_subst_dvector(handle, dev_vec[5], dev_vec[1]);
			_bncuda_subst_dvector(handle, dev_vec[3], dev_vec[5]);
		}
		else
		{
			beta = rho / old_rho;
#ifdef USE_POINTER_MODE_DEVICE
			_bncuda_set_d(pt_dval[id_beta], beta);
#endif
			/* u := r + beta q, p := u + beta * (q + beta p) */
			//_bncuda_add_cmul_dvector(handle, dev_vec[5], dev_vec[1], &beta, dev_vec[7]);
			//_bncuda_add_cmul_dvector(handle, dev_vec[8], dev_vec[7], &beta, dev_vec[3]);
			//_bncuda_add_cmul_dvector(handle, dev_vec[3], dev_vec[5], &beta, dev_vec[8]);
			_bncuda_add_cmul_dvector(handle, dev_vec[5], dev_vec[1], pt_dval[id_beta], dev_vec[7]);
			_bncuda_add_cmul_dvector(handle, dev_vec[8], dev_vec[7], pt_dval[id_beta], dev_vec[3]);
			_bncuda_add_cmul_dvector(handle, dev_vec[3], dev_vec[5], pt_dval[id_beta], dev_vec[8]);
		}
		/* precondition */
		/* pt = linsolve(pc_K, -p) */
		
		/* vt := Apt */
		_bncuda_mul_dmatrix_dvec(handle, dev_vec[8], dev_a, dev_vec[3]);

		//_bncuda_ip_dvector(handle, &alpha_den, dev_vec[2], dev_vec[8]);
		_bncuda_ip_dvector(handle, pt_dval[id_alpha_den], dev_vec[2], dev_vec[8]);
#ifdef USE_POINTER_MODE_DEVICE
		alpha_den = _bncuda_get_d(pt_dval[id_alpha_den]);
#endif
		if(alpha_den == 0.0)
		{
#ifndef __CUDACC__
			fprintf(stderr, "Denominator of Alpha is zero!(_bncuda_DCGS, %ld)\n", times);
#endif
			return_val = -2; // Fix!
			break; // Fix!
		}
		alpha = rho / alpha_den;
		minus_alpha = -alpha;
#ifdef USE_POINTER_MODE_DEVICE
		_bncuda_set_d(pt_dval[id_alpha], alpha);
		_bncuda_set_d(pt_dval[id_minus_alpha], minus_alpha);
#endif

		/* q = u - alpha  vt */
		//_bncuda_add_cmul_dvector(handle, dev_vec[7], dev_vec[5], &minus_alpha, dev_vec[8]);
		_bncuda_add_cmul_dvector(handle, dev_vec[7], dev_vec[5], pt_dval[id_minus_alpha], dev_vec[8]);

		/* ut = u + q */
		_bncuda_add_dvector(handle, dev_vec[6], dev_vec[5], dev_vec[7]);

		/* x = x + alpha ut */
		//_bncuda_add_cmul_dvector(handle, dev_vec[0], dev_vec[0], &alpha, dev_vec[6]);
		_bncuda_add_cmul_dvector(handle, dev_vec[0], dev_vec[0], pt_dval[id_alpha], dev_vec[6]);

		/* residual */
		_bncuda_mul_dmatrix_dvec(handle, dev_vec[8], dev_a, dev_vec[6]);

		//_bncuda_add_cmul_dvector(handle, dev_vec[1], dev_vec[1], &minus_alpha, dev_vec[8]);
		_bncuda_add_cmul_dvector(handle, dev_vec[1], dev_vec[1], pt_dval[id_minus_alpha], dev_vec[8]);

//		_bncuda_ip_dvector(handle, &beta_num, dev_vec[1], dev_vec[1]);
		_bncuda_ip_dvector(handle, pt_dval[id_beta_num], dev_vec[1], dev_vec[1]);
#ifdef USE_POINTER_MODE_DEVICE
		beta_num = _bncuda_get_d(pt_dval[id_beta_num]);
#endif
		/* Stopping Criteria */
		dtmp = sqrt(beta_num);
		if(dtmp <= aeps + reps * init_resnorm)
		{
			_bncuda_subst_dvector(handle, dev_answer, dev_vec[0]);
			return_val = times; // Fix!
			break;
		}

		old_rho = rho;
	}

	/* Not converge */
	_bncuda_subst_dvector(handle, dev_answer, dev_vec[0]);

	/* free vec[0]..[8]; */
//	for(i = 0; i < 9; i++)
//		_bncuda_free_dvector(dev_vec[i]);

	// Fix!
	if(times >= maxtimes)
	{
#ifndef __CUDACC__
		fprintf(stderr, "Not converge!(_bncuda_DCGS, %ld)\n", times);
#endif
		return_val= -5;
	}

#ifdef USE_POINTER_MODE_DEVICE
	for(i = 0; i < 11; i++)
		_bncuda_free_d_array(pt_dval[i]);
#endif

	//return return_val; // Fix!
#ifdef USE_POINTER_MODE_HOST
	*ret_val = return_val;
#else // USE_POINTER_MODE_DEVICE
	_bncuda_set_l(ret_val, return_val);
#endif

	return;
}

// called from host directly
long int _bncuda_DCGS(cublasHandle_t handle, DVector dev_answer, DMatrix dev_a, DVector dev_b, double reps, double aeps, long int maxtimes)
{
	long int i, dim;
	DVector dev_vec[9]; /* Temporary Vectors */
	double *dev_reps, *dev_aeps;
	long int *dev_maxtimes, *dev_ret_val, ret_val;

	dim = dev_answer->dim;

	/* Set initial value */
	for(i = 0; i < 9; i++)
		dev_vec[i] = _bncuda_init_dvector(dim);

#ifdef USE_POINTER_MODE_DEVICE
	dev_maxtimes = _bncuda_init_set_l(maxtimes);
	dev_reps = _bncuda_init_set_d(reps);
	dev_aeps = _bncuda_init_set_d(aeps);
	dev_ret_val = _bncuda_init_l();	
#else // USE_POINTER_MODE_HOST
	dev_maxtimes = &maxtimes;
	dev_reps = &reps;
	dev_aeps = &aeps;
	dev_ret_val = &ret_val;
#endif

	// call as global function
#if defined(__CUDACC__)
	printf("NVCC!\n");
	_bncuda_dev_DCGS<<<1,1>>>(handle, dev_ret_val, dev_answer, dev_a, dev_b, dev_reps, dev_aeps, dev_maxtimes, dev_vec);
#else
	printf("start calling _bncuda_dev_CGS...\n");
	_bncuda_dev_DCGS(handle, dev_ret_val, dev_answer, dev_a, dev_b, dev_reps, dev_aeps, dev_maxtimes, dev_vec);
#endif

	/* free vec[0]..[8]; */
	for(i = 0; i < 9; i++)
		_bncuda_free_dvector(dev_vec[i]);

#ifdef USE_POINTER_MODE_DEVICE
	ret_val = _bncuda_get_l(dev_ret_val);

	_bncuda_free_l(dev_maxtimes);
	_bncuda_free_d(dev_reps);
	_bncuda_free_d(dev_aeps);
	_bncuda_free_l(dev_ret_val);
#else  // USE_POINTER_MODE_HOST
	ret_val = *dev_ret_val;
#endif

	return ret_val;
}

/************************************************************/
/*                                                          */
/*             BiCGSTAB Method for Real Matrix              */
/*                                 (Double Precision)       */
/*                                 (cublas)       */
/*                                                          */
/*                 ver. 0.0 2013-01-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
__global__ void _bncuda_dev_DBiCGSTAB(cublasHandle_t handle, long int *ret_val, DVector dev_answer, DMatrix dev_a, DVector dev_b, double *pt_reps, double *pt_aeps, long int *pt_maxtimes, DVector dev_vec[9])
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       DVector answer: Solution for Ax = b                */
/*       DMatrix a: Coefficient matrix A                    */
/*                                       (given by user)    */
/*       DVector b: Constant vector b   (given by user)     */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       DVector answer: Solution for Ax = b                */
/*                                                          */
/* ERRORS                                                   */
/* Positive value ... Normal : Iterative Times              */
/*      -1 ... Rho is zero.                                 */
/*      -2 ... Denominator of Alpha is zero.                */
/*      -3 ... Denominator of Omega is zero.                */
/*      -4 ... Numerator of Omega is zero.                  */
/*      -5 ... Not Converge.                                */
/*                                                          */
/************************************************************/
{
	long int times, return_val, maxtimes;
	double alpha, alpha_den, minus_alpha;
	double beta, beta_num;
	double rho, old_rho;
	double omega, omega_den, minus_omega;
	double dtmp, init_resnorm;
	double reps, aeps;
//	DVector dev_vec[9]; /* Temporary Vectors */

	int id_alpha = 0, id_alpha_den = 1, id_minus_alpha = 2;
	int id_beta = 3, id_beta_num = 4;
	int id_rho = 5, id_old_rho = 6;
	int id_omega = 7, id_omega_den = 8, id_minus_omega = 9;
	int id_dtmp = 10, id_init_resnorm = 11;
	int id_reps = 12, id_aeps = 13;
	double *pt_dval[14];

#ifdef USE_POINTER_MODE_HOST
//	printf("DBiCGSTAB in HOST MODE\n");
	pt_dval[id_alpha] = &alpha;
	pt_dval[id_alpha_den] = &alpha_den;
	pt_dval[id_minus_alpha] = &minus_alpha;
	pt_dval[id_beta] = &beta;
	pt_dval[id_beta_num] = &beta_num;
	pt_dval[id_rho] = &rho;
	pt_dval[id_old_rho] = &old_rho;
	pt_dval[id_omega] = &omega;
	pt_dval[id_omega_den] = &omega_den;
	pt_dval[id_minus_omega] = &minus_omega;
	pt_dval[id_dtmp] = &dtmp;
	pt_dval[id_init_resnorm] = &init_resnorm;
#else //USE_POINTER_MODE_DEVICE
	int i;//	printf("DBiCGSTAB in DEVICE MODE\n");
	for(i = 0; i < 14; i++)
		pt_dval[i] = _bncuda_init_d();
#endif

//	DVector dev_vec[8]; /* Temporary Vectors */

#ifdef USE_POINTER_MODE_HOST
	maxtimes = *pt_maxtimes;
	reps = *pt_reps;
	aeps = *pt_aeps;
#else
	maxtimes = _bncuda_get_l(pt_maxtimes);
	reps = _bncuda_get_d(pt_reps);
	aeps = _bncuda_get_d(pt_aeps);
#endif

/* Set initial value */
//	for(i = 0; i < 9; i++)
//		dev_vec[i] = _bncuda_init_dvector(dim);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0]  == w*/
	/* vec[2] ... (b - a * vec[0])^T  == wt */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... v */
	/* vec[6] ... s */
	/* vec[7] ... s^T */
	/* vec[8] ... t */

	_bncuda_subst_dvector(handle, dev_vec[1], dev_b); 
	_bncuda_subst_dvector(handle, dev_vec[2], dev_b);

	//_bncuda_ip_dvector(handle, &beta_num, dev_vec[1], dev_vec[1]);
	_bncuda_ip_dvector(handle, pt_dval[id_beta_num], dev_vec[1], dev_vec[1]);
#ifdef USE_POINTER_MODE_DEVICE
	beta_num = _bncuda_get_d(pt_dval[id_beta_num]);
#endif
	init_resnorm = sqrt(beta_num);

	old_rho = 0.0;
	rho = 0.0;
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		//_bncuda_ip_dvector(handle, &rho, dev_vec[2], dev_vec[1]);
		_bncuda_ip_dvector(handle, pt_dval[id_rho], dev_vec[2], dev_vec[1]);
#ifdef USE_POINTER_MODE_DEVICE
		rho = _bncuda_get_d(pt_dval[id_rho]);
#endif

		if(rho == 0.0)
		{
#ifndef __CUDACC__
			fprintf(stderr, "Rho is zero!(_bncuda_DBiCGSTAB, %ld)\n", times);
#endif
			return_val= -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* p := r */
			_bncuda_subst_dvector(handle, dev_vec[3], dev_vec[1]);
		}
		else
		{
			beta = (rho / old_rho) * (alpha / omega);
#ifdef USE_POINTER_MODE_DEVICE
			_bncuda_set_d(pt_dval[id_beta], beta);
#endif

			/* p := r + beta (p - omega v) */
			//_bncuda_add_cmul_dvector(handle, dev_vec[4], dev_vec[3], &minus_omega, dev_vec[5]);
			//_bncuda_add_cmul_dvector(handle, dev_vec[3], dev_vec[1], &beta, dev_vec[4]);
			_bncuda_add_cmul_dvector(handle, dev_vec[4], dev_vec[3], pt_dval[id_minus_omega], dev_vec[5]);
			_bncuda_add_cmul_dvector(handle, dev_vec[3], dev_vec[1], pt_dval[id_beta], dev_vec[4]);
		}
		/* precondition */
		
		/* v := Apt */
		_bncuda_mul_dmatrix_dvec(handle, dev_vec[5], dev_a, dev_vec[3]);

		//_bncuda_ip_dvector(handle, &alpha_den, dev_vec[2], dev_vec[5]);
		_bncuda_ip_dvector(handle, pt_dval[id_alpha_den], dev_vec[2], dev_vec[5]);
#ifdef USE_POINTER_MODE_DEVICE
		alpha_den = _bncuda_get_d(pt_dval[id_alpha_den]);
#endif
		if(alpha_den == 0.0)
		{
#ifndef __CUDACC__
			fprintf(stderr, "Denominator of Alpha is zero!(_bncuda_DBiCGSTAB, %ld)\n", times);
#endif
			return_val = -2; // Fix!
			break; // Fix!
		}
		alpha = rho / alpha_den;
		minus_alpha = -alpha;
#ifdef USE_POINTER_MODE_DEVICE
		_bncuda_set_d(pt_dval[id_alpha], alpha);
		_bncuda_set_d(pt_dval[id_minus_alpha], minus_alpha);
#endif

		/* s = r - alpha v */
		//_bncuda_add_cmul_dvector(handle, dev_vec[6], dev_vec[1], &minus_alpha, dev_vec[5]);
		_bncuda_add_cmul_dvector(handle, dev_vec[6], dev_vec[1], pt_dval[id_minus_alpha], dev_vec[5]);

		/* Stopping Criteria */
		//_bncuda_norm2_dvector(handle, &dtmp, dev_vec[6]);
		_bncuda_norm2_dvector(handle, pt_dval[id_dtmp], dev_vec[6]);
#ifdef USE_POINTER_MODE_DEVICE
		dtmp = _bncuda_get_d(pt_dval[id_dtmp]);
#endif
		if(dtmp <= aeps + reps * init_resnorm)
		{
			/* x = x + alpha pt */
			//_bncuda_add_cmul_dvector(handle, dev_vec[0], dev_vec[0], &alpha, dev_vec[3]);
			_bncuda_add_cmul_dvector(handle, dev_vec[0], dev_vec[0], pt_dval[id_alpha], dev_vec[3]);

			_bncuda_subst_dvector(handle, dev_answer, dev_vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		/* precondition */
		_bncuda_mul_dmatrix_dvec(handle, dev_vec[8], dev_a, dev_vec[6]);

		/* omega = (t, s) / (t, t) */
		//_bncuda_ip_dvector(handle, &omega_den, dev_vec[8], dev_vec[8]);
		_bncuda_ip_dvector(handle, pt_dval[id_omega_den], dev_vec[8], dev_vec[8]);
#ifdef USE_POINTER_MODE_DEVICE
		omega_den = _bncuda_get_d(pt_dval[id_omega_den]);
#endif
		if(omega_den == 0)
		{
#ifndef __CUDACC__
			fprintf(stderr, "Denominator of Omega is zero!(_bncuda_DBiCGSTAB, %ld)\n", times);
#endif
			return_val = -3; // Fix!
			break;
		}
		//_bncuda_ip_dvector(handle, &omega, dev_vec[8], dev_vec[6]);
		_bncuda_ip_dvector(handle, pt_dval[id_omega], dev_vec[8], dev_vec[6]);
#ifdef USE_POINTER_MODE_DEVICE
		omega = _bncuda_get_d(pt_dval[id_omega]);
#endif
		if(omega == 0)
		{
#ifndef __CUDACC__
			fprintf(stderr, "Numerator of Omega is zero!(_bncuda_DBiCGSTAB, %ld)\n", times);
#endif
			return_val = -4; // Fix!
			break; // Fix!
		}
		omega = omega / omega_den;
		minus_omega = -omega;
#ifdef USE_POINTER_MODE_DEVICE
		_bncuda_set_d(pt_dval[id_omega], omega);
		_bncuda_set_d(pt_dval[id_minus_omega], minus_omega);
#endif

		/* x = x + alpha pt + omega st */
		//_bncuda_add_cmul_dvector(handle, dev_vec[4], dev_vec[0], &alpha, dev_vec[3]);
		//_bncuda_add_cmul_dvector(handle, dev_vec[0], dev_vec[4], &omega, dev_vec[6]);
		_bncuda_add_cmul_dvector(handle, dev_vec[4], dev_vec[0], pt_dval[id_alpha], dev_vec[3]);
		_bncuda_add_cmul_dvector(handle, dev_vec[0], dev_vec[4], pt_dval[id_omega], dev_vec[6]);

		/* residual */
		//_bncuda_add_cmul_dvector(handle, dev_vec[1], dev_vec[6], &minus_omega, dev_vec[8]);
		_bncuda_add_cmul_dvector(handle, dev_vec[1], dev_vec[6], pt_dval[id_minus_omega], dev_vec[8]);

		//_bncuda_ip_dvector(handle, &beta_num, dev_vec[1], dev_vec[1]);
		_bncuda_ip_dvector(handle, pt_dval[id_beta_num], dev_vec[1], dev_vec[1]);
#ifdef USE_POINTER_MODE_DEVICE
		beta_num = _bncuda_get_d(pt_dval[id_beta_num]);
#endif

		/* Stopping Criteria */
		dtmp = sqrt(beta_num);
		if(dtmp <= aeps + reps * init_resnorm)
		{
			_bncuda_subst_dvector(handle, dev_answer, dev_vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		old_rho = rho;
	}

	/* Not converge */
	_bncuda_subst_dvector(handle, dev_answer, dev_vec[0]);

	/* free vec[0]..[3]; */
//	for(i = 0; i < 9; i++)
//		_bncuda_free_dvector(dev_vec[i]);

	// Fix!
	if(times >= maxtimes)
	{
#ifndef __CUDACC__
		fprintf(stderr, "Not converge!(_bncuda_DBiCGSTAB, %ld)\n", times);
#endif
		return_val = -5;
	}

#ifdef USE_POINTER_MODE_DEVICE
	for(i = 0; i < 14; i++)
		_bncuda_free_d_array(pt_dval[i]);
#endif

	//return return_val; // Fix!
#ifdef USE_POINTER_MODE_HOST
	*ret_val = return_val;
#else // USE_POINTER_MODE_DEVICE
	_bncuda_set_l(ret_val, return_val);
#endif
	return;
}

// called from host directly
long int _bncuda_DBiCGSTAB(cublasHandle_t handle, DVector dev_answer, DMatrix dev_a, DVector dev_b, double reps, double aeps, long int maxtimes)
{
	long int i, dim;
	DVector dev_vec[9]; /* Temporary Vectors */
	double *dev_reps, *dev_aeps;
	long int *dev_maxtimes, *dev_ret_val, ret_val;

	dim = dev_answer->dim;

	/* Set initial value */
	for(i = 0; i < 9; i++)
		dev_vec[i] = _bncuda_init_dvector(dim);

#ifdef USE_POINTER_MODE_DEVICE
	dev_maxtimes = _bncuda_init_set_l(maxtimes);
	dev_reps = _bncuda_init_set_d(reps);
	dev_aeps = _bncuda_init_set_d(aeps);
	dev_ret_val = _bncuda_init_l();	
#else // USE_POINTER_MODE_HOST
	dev_maxtimes = &maxtimes;
	dev_reps = &reps;
	dev_aeps = &aeps;
	dev_ret_val = &ret_val;
#endif

	// call as global function
#if defined(__CUDACC__)
	printf("NVCC!\n");
	_bncuda_dev_DBiCGSTAB<<<1,1>>>(handle, dev_ret_val, dev_answer, dev_a, dev_b, dev_reps, dev_aeps, dev_maxtimes, dev_vec);
#else
	printf("start calling _bncuda_dev_CGS...\n");
	_bncuda_dev_DBiCGSTAB(handle, dev_ret_val, dev_answer, dev_a, dev_b, dev_reps, dev_aeps, dev_maxtimes, dev_vec);
#endif

	/* free vec[0]..[8]; */
	for(i = 0; i < 9; i++)
		_bncuda_free_dvector(dev_vec[i]);

#ifdef USE_POINTER_MODE_DEVICE
	ret_val = _bncuda_get_l(dev_ret_val);

	_bncuda_free_l(dev_maxtimes);
	_bncuda_free_d(dev_reps);
	_bncuda_free_d(dev_aeps);
	_bncuda_free_l(dev_ret_val);
#else  // USE_POINTER_MODE_HOST
	ret_val = *dev_ret_val;
#endif

	return ret_val;
}

/************************************************************/
/*                                                          */
/* GPBiCG(Generalized Product Bi-CG) Method for Real Matrix */
/*                                 (Double Precision)       */
/*                                 (cublas)       */
/*                                                          */
/*                 ver. 0.0 2013-01-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
__global__ void _bncuda_dev_DGPBiCG(cublasHandle_t handle, long int *ret_val, DVector dev_answer, DMatrix dev_a, DVector dev_b, double *pt_reps, double *pt_aeps, long int *pt_maxtimes, DVector dev_vec[13])
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       DVector answer: Solution for Ax = b                */
/*       DMatrix a: Coefficient matrix A                    */
/*                                       (given by user)    */
/*       DVector b: Constant vector b   (given by user)     */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       DVector answer: Solution for Ax = b                */
/*                                                          */
/* ERRORS                                                   */
/* Positive value ... Normal : Iterative Times              */
/*      -1 ... Rho is zero.                                 */
/*      -2 ... Denominator of Alpha is zero.                */
/*      -3 ... (Reserved)                                   */
/*      -4 ... (Reserved)                                   */
/*      -5 ... Not Converge.                                */
/*                                                          */
/************************************************************/
{
	long int times, return_val, maxtimes;
	double alpha, alpha_den, minus_alpha;
	double beta, beta_num;
	double rho, old_rho;
	double mu[5], tau, zeta, eta, minus_zeta, minus_eta;
	double dtmp, init_resnorm;
	double reps, aeps;
//	DVector dev_vec[13]; /* Temporary Vectors */

	int i;
	int id_alpha = 0, id_alpha_den = 1, id_minus_alpha = 2;
	int id_beta = 3, id_beta_num = 4;
	int id_rho = 5, id_old_rho = 6;
	int id_mu[5] = {7, 8, 9, 10, 11}, id_tau = 12, id_zeta = 13, id_eta = 14, id_minus_zeta = 15, id_minus_eta = 16;
	int id_dtmp = 17, id_init_resnorm = 18;
	int id_reps = 19, id_aeps = 20;
	double *pt_dval[21];

#ifdef USE_POINTER_MODE_HOST
//	printf("DBiCGSTAB in HOST MODE\n");
	pt_dval[id_alpha] = &alpha;
	pt_dval[id_alpha_den] = &alpha_den;
	pt_dval[id_minus_alpha] = &minus_alpha;
	pt_dval[id_beta] = &beta;
	pt_dval[id_beta_num] = &beta_num;
	pt_dval[id_rho] = &rho;
	pt_dval[id_old_rho] = &old_rho;
	for(i = 0; i < 5; i++) pt_dval[id_mu[i]] = &mu[i];
	pt_dval[id_tau] = &tau;
	pt_dval[id_zeta] = &zeta;
	pt_dval[id_eta] = &eta;
	pt_dval[id_minus_zeta] = &minus_zeta;
	pt_dval[id_minus_eta] = &minus_eta;
	pt_dval[id_dtmp] = &dtmp;
	pt_dval[id_init_resnorm] = &init_resnorm;
#else //USE_POINTER_MODE_DEVICE
//	printf("DGPBiCG in DEVICE MODE\n");
	for(i = 0; i < 21; i++)
		pt_dval[i] = _bncuda_init_d();
#endif

//	DVector dev_vec[8]; /* Temporary Vectors */

#ifdef USE_POINTER_MODE_HOST
	maxtimes = *pt_maxtimes;
	reps = *pt_reps;
	aeps = *pt_aeps;
#else
	maxtimes = _bncuda_get_l(pt_maxtimes);
	reps = _bncuda_get_d(pt_reps);
	aeps = _bncuda_get_d(pt_aeps);
#endif

/* Set initial value */
//	for(i = 0; i < 13; i++)
//		dev_vec[i] = _bncuda_init_dvector(dim);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0] */
	/* vec[2] ... (b - a * vec[0])^T */
	/* vec[3] ... p */
	/* vec[4] ... q */
	/* vec[5] ... s */
	/* vec[6] ... t */
	/* vec[7] ... u */
	/* vec[8] ... v */
	/* vec[9] ... w */
	/* vec[10]... y */
	/* vec[11]... z */

	_bncuda_subst_dvector(handle, dev_vec[1], dev_b); 
	_bncuda_subst_dvector(handle, dev_vec[2], dev_b);

//	_bncuda_ip_dvector(handle, &beta_num, dev_vec[1], dev_vec[1]);
	_bncuda_ip_dvector(handle, pt_dval[id_beta_num], dev_vec[1], dev_vec[1]);
#ifdef USE_POINTER_MODE_DEVICE
	beta_num = _bncuda_get_d(pt_dval[id_beta_num]);
#endif
	init_resnorm = sqrt(beta_num);

	old_rho = 0.0;
	rho = 0.0;
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		//_bncuda_ip_dvector(handle, &rho, dev_vec[2], dev_vec[1]);
		_bncuda_ip_dvector(handle, pt_dval[id_rho], dev_vec[2], dev_vec[1]);
#ifdef USE_POINTER_MODE_DEVICE
		rho = _bncuda_get_d(pt_dval[id_rho]);
#endif

		if(rho == 0.0)
		{
#ifndef __CUDACC__
			fprintf(stderr, "Rho is zero!(_bncuda_DGPBiCG, %ld)\n", times);
#endif
			return_val = -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/*
				p = r;
				q = A * p;
				alpha = rho / (r0^t, q);
				t = r - alpha * q;
				v = A * t;
				y = alpha * q - r;
				mu2 = v' * t;
				mu5 = v' * v;
				zeta = mu2 / mu5;
				eta = 0;
				u = zeta * q; */
			_bncuda_subst_dvector(handle, dev_vec[3], dev_vec[1]);
			_bncuda_mul_dmatrix_dvec(handle, dev_vec[4], dev_a, dev_vec[3]);

			//_bncuda_ip_dvector(handle, &alpha_den, dev_vec[2], dev_vec[4]);
			_bncuda_ip_dvector(handle, pt_dval[id_alpha_den], dev_vec[2], dev_vec[4]);
#ifdef USE_POINTER_MODE_DEVICE
			alpha_den = _bncuda_get_d(pt_dval[id_alpha_den]);
#endif
			if(alpha_den == 0.0)
			{
#ifndef __CUDACC__
				fprintf(stderr, "Denominator of Alpha is zero!(_bncuda_DGPBiCG, %ld)\n", times);
#endif
				return_val = -2; // Fix!
				break; // Fix!
			}
			alpha = rho / alpha_den;
			minus_alpha = -alpha;
#ifdef USE_POINTER_MODE_DEVICE
			_bncuda_set_d(pt_dval[id_alpha], alpha);
			_bncuda_set_d(pt_dval[id_minus_alpha], minus_alpha);
#endif

			//_bncuda_add_cmul_dvector(handle, dev_vec[6], dev_vec[1], &minus_alpha, dev_vec[4]);
			_bncuda_add_cmul_dvector(handle, dev_vec[6], dev_vec[1], pt_dval[id_minus_alpha], dev_vec[4]);
			_bncuda_mul_dmatrix_dvec(handle, dev_vec[8], dev_a, dev_vec[6]);

			//_bncuda_cmul_dvector(handle, dev_vec[10], &alpha, dev_vec[4]);
			_bncuda_cmul_dvector(handle, dev_vec[10], pt_dval[id_alpha], dev_vec[4]);
			_bncuda_sub_dvector(handle, dev_vec[10], dev_vec[10], dev_vec[1]);
			//_bncuda_ip_dvector(handle, &mu[1], dev_vec[8], dev_vec[6]);
			//_bncuda_ip_dvector(handle, &mu[4], dev_vec[8], dev_vec[8]);
			_bncuda_ip_dvector(handle, pt_dval[id_mu[1]], dev_vec[8], dev_vec[6]);
			_bncuda_ip_dvector(handle, pt_dval[id_mu[4]], dev_vec[8], dev_vec[8]);
#ifdef USE_POINTER_MODE_DEVICE
			mu[1] = _bncuda_get_d(pt_dval[id_mu[1]]);
			mu[4] = _bncuda_get_d(pt_dval[id_mu[4]]);
#endif
			
			if(mu[4] == 0.0)
			{
#ifndef __CUDACC__
				fprintf(stderr, "Denominator of mu[4] is zero!(_bncuda_DGPBiCG, %ld)\n", times);
#endif
				return_val = -3; // Fix!
				break; // Fix!
			}
			zeta = mu[1] / mu[4];
			eta = 0.0;
			minus_zeta = -zeta;
			minus_eta = -eta;
#ifdef USE_POINTER_MODE_DEVICE
			_bncuda_set_d(pt_dval[id_zeta], zeta);
			_bncuda_set_d(pt_dval[id_eta], eta);
			_bncuda_set_d(pt_dval[id_minus_zeta], minus_zeta);
			_bncuda_set_d(pt_dval[id_minus_eta], minus_eta);
#endif
	
			//_bncuda_cmul_dvector(handle, dev_vec[7], &zeta, dev_vec[4]);
			_bncuda_cmul_dvector(handle, dev_vec[7], pt_dval[id_zeta], dev_vec[4]);
		}
		else
		{
			/*
			k_beta = (rho / old_rho) * (alpha / zeta);
			w = v + k_beta * q;
			p = ret_res + k_beta * (p - u);
			q = A * p;
			alpha = rho / (ret_res_c' * q);
			s = t - ret_res;
			t = ret_res - alpha * q;
			v = A * t;
			y = s - alpha * (w - q);
			mu1 = y' * y;
			mu2 = v' * t;
			mu3 = y' * t;
			mu4 = v' * y;
			mu5 = v' * v;
			tau = mu5 * mu1 - mu4 * mu4;
			zeta = (mu1 * mu2 - mu3 * mu4) / tau;
			eta = (mu5 * mu3 - mu4 * mu2) / tau;
			u = zeta * q + eta * (s + k_beta * u); */

			beta = (rho / old_rho) * (alpha / zeta);
#ifdef USE_POINTER_MODE_DEVICE
			_bncuda_set_d(pt_dval[id_beta], beta);
#endif

			//_bncuda_add_cmul_dvector(handle, dev_vec[9], dev_vec[8], &beta, dev_vec[4]);
			_bncuda_add_cmul_dvector(handle, dev_vec[9], dev_vec[8], pt_dval[id_beta], dev_vec[4]);
			_bncuda_sub_dvector(handle, dev_vec[3], dev_vec[3], dev_vec[7]);
			//_bncuda_add_cmul_dvector(handle, dev_vec[12], dev_vec[1], &beta, dev_vec[3]);
			_bncuda_add_cmul_dvector(handle, dev_vec[12], dev_vec[1], pt_dval[id_beta], dev_vec[3]);
			_bncuda_subst_dvector(handle, dev_vec[3], dev_vec[12]); // for GPU

			_bncuda_mul_dmatrix_dvec(handle, dev_vec[4], dev_a, dev_vec[3]);

			//_bncuda_ip_dvector(handle, &alpha_den, dev_vec[2], dev_vec[4]);
			_bncuda_ip_dvector(handle, pt_dval[id_alpha_den], dev_vec[2], dev_vec[4]);
#ifdef USE_POINTER_MODE_DEVICE
			alpha_den = _bncuda_get_d(pt_dval[id_alpha_den]);
#endif
			if(alpha_den == 0.0)
			{
#ifndef __CUDACC__
				fprintf(stderr, "Denominator of Alpha is zero!(_bncuda_DGPBiCG, %ld)\n", times);
#endif
				return_val = -2; // Fix!
				break; // Fix!
			}
			alpha = rho / alpha_den;
			minus_alpha = -alpha;
#ifdef USE_POINTER_MODE_DEVICE
			_bncuda_set_d(pt_dval[id_alpha], alpha);
			_bncuda_set_d(pt_dval[id_minus_alpha], minus_alpha);
#endif

			_bncuda_sub_dvector(handle, dev_vec[5], dev_vec[6], dev_vec[1]);
			//_bncuda_add_cmul_dvector(handle, dev_vec[6], dev_vec[1], &minus_alpha, dev_vec[4]);
			_bncuda_add_cmul_dvector(handle, dev_vec[6], dev_vec[1], pt_dval[id_minus_alpha], dev_vec[4]);

			_bncuda_mul_dmatrix_dvec(handle, dev_vec[8], dev_a, dev_vec[6]);

			_bncuda_sub_dvector(handle, dev_vec[10], dev_vec[9], dev_vec[4]);
			//_bncuda_add_cmul_dvector(handle, dev_vec[12], dev_vec[5], &minus_alpha, dev_vec[10]);
			_bncuda_add_cmul_dvector(handle, dev_vec[12], dev_vec[5], pt_dval[id_minus_alpha], dev_vec[10]);
			_bncuda_subst_dvector(handle, dev_vec[10], dev_vec[12]);// for GPU
			//_bncuda_ip_dvector(handle, &mu[0], dev_vec[10], dev_vec[10]);
			//_bncuda_ip_dvector(handle, &mu[1], dev_vec[8], dev_vec[6]);
			//_bncuda_ip_dvector(handle, &mu[2], dev_vec[10], dev_vec[6]);
			//_bncuda_ip_dvector(handle, &mu[3], dev_vec[8], dev_vec[10]);
			//_bncuda_ip_dvector(handle, &mu[4], dev_vec[8], dev_vec[8]);
			_bncuda_ip_dvector(handle, pt_dval[id_mu[0]], dev_vec[10], dev_vec[10]);
			_bncuda_ip_dvector(handle, pt_dval[id_mu[1]], dev_vec[8], dev_vec[6]);
			_bncuda_ip_dvector(handle, pt_dval[id_mu[2]], dev_vec[10], dev_vec[6]);
			_bncuda_ip_dvector(handle, pt_dval[id_mu[3]], dev_vec[8], dev_vec[10]);
			_bncuda_ip_dvector(handle, pt_dval[id_mu[4]], dev_vec[8], dev_vec[8]);
#ifdef USE_POINTER_MODE_DEVICE
			for(i = 0; i < 5; i++)
				mu[i] = _bncuda_get_d(pt_dval[id_mu[i]]);
#endif
			tau = mu[4] * mu[0] - mu[3] * mu[3];
			zeta = (mu[0] * mu[1] - mu[2] * mu[3]) / tau;
			eta = (mu[4] * mu[2] - mu[3] * mu[1]) / tau;
			minus_zeta = -zeta;
			minus_eta = -eta;
#ifdef USE_POINTER_MODE_DEVICE
			_bncuda_set_d(pt_dval[id_tau], tau);
			_bncuda_set_d(pt_dval[id_zeta], zeta);
			_bncuda_set_d(pt_dval[id_eta], eta);
			_bncuda_set_d(pt_dval[id_minus_zeta], minus_zeta);
			_bncuda_set_d(pt_dval[id_minus_eta], minus_eta);
#endif
			/* u = zeta * q + eta * (s + k_beta * u); */
			//_bncuda_add_cmul_dvector(handle, dev_vec[12], dev_vec[5], &beta, dev_vec[7]);
			_bncuda_add_cmul_dvector(handle, dev_vec[12], dev_vec[5], pt_dval[id_beta], dev_vec[7]);
			_bncuda_subst_dvector(handle, dev_vec[7], dev_vec[12]); // for GPU
			//_bncuda_cmul_dvector(handle, dev_vec[7], &eta, dev_vec[7]);
			_bncuda_cmul_dvector(handle, dev_vec[7], pt_dval[id_eta], dev_vec[7]);
			//_bncuda_add_cmul_dvector(handle, dev_vec[7], dev_vec[7], &zeta, dev_vec[4]);
			_bncuda_add_cmul_dvector(handle, dev_vec[7], dev_vec[7], pt_dval[id_zeta], dev_vec[4]);
		}
		/* z = zeta * ret_res + eta * z - alpha * u; */
		//_bncuda_cmul2_dvector(handle, dev_vec[11], &eta);
		//_bncuda_add_cmul_dvector(handle, dev_vec[11], dev_vec[11], &zeta, dev_vec[1]);
		//_bncuda_add_cmul_dvector(handle, dev_vec[11], dev_vec[11], &minus_alpha, dev_vec[7]);
		_bncuda_cmul2_dvector(handle, dev_vec[11], pt_dval[id_eta]);
		_bncuda_add_cmul_dvector(handle, dev_vec[11], dev_vec[11], pt_dval[id_zeta], dev_vec[1]);
		_bncuda_add_cmul_dvector(handle, dev_vec[11], dev_vec[11], pt_dval[id_minus_alpha], dev_vec[7]);

		/* x = x + alpha p + z */
		//_bncuda_add_cmul_dvector(handle, dev_vec[0], dev_vec[0], &alpha, dev_vec[3]);
		_bncuda_add_cmul_dvector(handle, dev_vec[0], dev_vec[0], pt_dval[id_alpha], dev_vec[3]);
		_bncuda_add2_dvector(handle, dev_vec[0], dev_vec[11]);

		/* residual */
		/* r = t - eta * y - zeta * v; */
		//_bncuda_add_cmul_dvector(handle, dev_vec[1], dev_vec[6], &minus_eta, dev_vec[10]);
		//_bncuda_add_cmul_dvector(handle, dev_vec[1], dev_vec[1], &minus_zeta, dev_vec[8]);
		_bncuda_add_cmul_dvector(handle, dev_vec[1], dev_vec[6], pt_dval[id_minus_eta], dev_vec[10]);
		_bncuda_add_cmul_dvector(handle, dev_vec[1], dev_vec[1], pt_dval[id_minus_zeta], dev_vec[8]);

		//_bncuda_ip_dvector(handle, &beta_num, dev_vec[1], dev_vec[1]);
		_bncuda_ip_dvector(handle, pt_dval[id_beta_num], dev_vec[1], dev_vec[1]);
#ifdef USE_POINTER_MODE_DEVICE
		beta_num = _bncuda_get_d(pt_dval[id_beta_num]);
#endif

		/* Stopping Criteria */
		dtmp = sqrt(beta_num);
		if(dtmp <= aeps + reps * init_resnorm)
		{
			_bncuda_subst_dvector(handle, dev_answer, dev_vec[0]);
			return_val = times; // Fix!
			break;
		}
//		printf("DGPBiCG: %5d %10.3e\n", times, dtmp / init_resnorm);
		if(zeta == 0.0)
		{
#ifndef __CUDACC__
			fprintf(stderr, "Denominator of Zeta is zero!(_bncuda_DGPBiCG, %ld)\n", times);
#endif
			return_val = -4; // Fix!
			break; // Fix!
		}

		old_rho = rho;
	}

	/* Not converge */
	_bncuda_subst_dvector(handle, dev_answer, dev_vec[0]);

	/* free vec[0]..[3]; */
//	for(i = 0; i < 13; i++)
//		_bncuda_free_dvector(dev_vec[i]);

	// Fix!
	if(times >= maxtimes)
	{
#ifndef __CUDACC__
		fprintf(stderr, "Not converge!(_bncuda_DGPBiCG, %ld)\n", times);
#endif
		return_val = -5;
	}

#ifdef USE_POINTER_MODE_DEVICE
	for(i = 0; i < 21; i++)
		_bncuda_free_d_array(pt_dval[i]);
#endif

	//return return_val; // Fix!
#ifdef USE_POINTER_MODE_HOST
	*ret_val = return_val;
#else // USE_POINTER_MODE_DEVICE
	_bncuda_set_l(ret_val, return_val);
#endif
	return;
}

// called from host directly
long int _bncuda_DGPBiCG(cublasHandle_t handle, DVector dev_answer, DMatrix dev_a, DVector dev_b, double reps, double aeps, long int maxtimes)
{
	long int i, dim;
	DVector dev_vec[13]; /* Temporary Vectors */
	double *dev_reps, *dev_aeps;
	long int *dev_maxtimes, *dev_ret_val, ret_val;

	dim = dev_answer->dim;

	/* Set initial value */
	for(i = 0; i < 13; i++)
		dev_vec[i] = _bncuda_init_dvector(dim);

#ifdef USE_POINTER_MODE_DEVICE
	dev_maxtimes = _bncuda_init_set_l(maxtimes);
	dev_reps = _bncuda_init_set_d(reps);
	dev_aeps = _bncuda_init_set_d(aeps);
	dev_ret_val = _bncuda_init_l();	
#else // USE_POINTER_MODE_HOST
	dev_maxtimes = &maxtimes;
	dev_reps = &reps;
	dev_aeps = &aeps;
	dev_ret_val = &ret_val;
#endif

	// call as global function
#if defined(__CUDACC__)
	printf("NVCC!\n");
	_bncuda_dev_DGPBiCG<<<1,1>>>(handle, dev_ret_val, dev_answer, dev_a, dev_b, dev_reps, dev_aeps, dev_maxtimes, dev_vec);
#else
	printf("start calling _bncuda_dev_DGPBiCG...\n");
	_bncuda_dev_DGPBiCG(handle, dev_ret_val, dev_answer, dev_a, dev_b, dev_reps, dev_aeps, dev_maxtimes, dev_vec);
#endif

	/* free vec[0]..[12]; */
	for(i = 0; i < 13; i++)
		_bncuda_free_dvector(dev_vec[i]);

#ifdef USE_POINTER_MODE_DEVICE
	ret_val = _bncuda_get_l(dev_ret_val);

	_bncuda_free_l(dev_maxtimes);
	_bncuda_free_d(dev_reps);
	_bncuda_free_d(dev_aeps);
	_bncuda_free_l(dev_ret_val);
#else  // USE_POINTER_MODE_HOST
	ret_val = *dev_ret_val;
#endif

	return ret_val;
}
#endif // USE_CUBLAS

//#endif

