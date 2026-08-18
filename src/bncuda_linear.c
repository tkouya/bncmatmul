/********************************************************************************/
/* bncuda_linear.c: Vector, Matrix                                              */
/* Copyright (c) 2013-2015 Tomonori Kouya                                       */
/*                                                                              */
/* Version 0.1, 2013-01-26: first implementation with CUDA and cublas           */
/* Version 0.2, 2015-06-24: append various routines to test bncmatmul library   */
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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// BNCpack
#include "bnc.h"

// for CUDA
#include "cuda.h"
#include "cuda_runtime.h"

// BNCuda
#include "bncuda.h"

// Get Device Information
// Ref: J.Sanders & E.Kandrot, CUDA by example
int _bncuda_getinfo(int *maxThreadsDim_x, int *maxGridSize_x)
{
	cudaError_t cuda_error;
	cudaDeviceProp prop;
	int device_num, i;

	device_num = 0;

	cuda_error = cudaGetDeviceCount(&device_num);
	if(device_num <= 0)
	{
		printf("No CUDA devices!\n");
		return 0;
	}

	for(i = 0; i < device_num; i++)
	{
		cuda_error = cudaGetDeviceProperties(&prop, i);
		if(cuda_error != cudaSuccess)
			printf("Warning: %d-th device has some problems\n", i);

		printf("--- General Information for %d-th device ---\n", i);
		printf("Name                    : %s\n", prop.name);
		printf("Compute capability      : %d.%d\n", prop.major, prop.minor);
#if CUDART_VERSION < 12000
		/* clockRate, deviceOverlap, kernelExecTimeoutEnabled were
		 * removed from cudaDeviceProp in CUDA 12.  On 12+ the same
		 * info is reachable via cuDeviceGetAttribute(...) but we just
		 * skip the diagnostic prints here. */
		printf("Clock rate              : %d\n", prop.clockRate);
		printf("Device copy overlap     : ");
			if(prop.deviceOverlap) printf("Enabled\n");
			else printf("Disabled\n");
		printf("Kernel execution timeout: ");
			if(prop.kernelExecTimeoutEnabled) printf("Enabled\n");
			else printf("Disabled\n");
#else
		/* CUDA 12+: asyncEngineCount replaces deviceOverlap. */
		printf("Async engine count      : %d\n", prop.asyncEngineCount);
		printf("Clock rate / KernelExecTimeout : (removed in CUDA 12)\n");
#endif

		printf("--- Memory information for %d-th device ---\n", i);
		printf("Total global memory     : %ld\n", prop.totalGlobalMem);
		printf("Total constant memory   : %ld\n", prop.totalConstMem);
		printf("Max memory pitch        : %ld\n", prop.memPitch);
		printf("Texture Alignment       : %ld\n", prop.textureAlignment);

		printf("--- Multi-processor information for %d-th device---\n", i);
		printf("Muti-processor count    : %d\n", prop.multiProcessorCount);
		printf("Shared memory per MP    : %ld\n", prop.sharedMemPerBlock);
		printf("Registers per MP        : %d\n", prop.regsPerBlock);
		printf("Threads in warp         : %d\n", prop.warpSize);
		printf("Max threads per block   : %d\n", prop.maxThreadsPerBlock);
		printf("Max thread dimensions   : (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
		printf("Max grid dimensions     : (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

		// return minimum sizes in all GPUs
		if(*maxThreadsDim_x < prop.maxThreadsDim[0])
			*maxThreadsDim_x = prop.maxThreadsDim[0];
		if(*maxGridSize_x < prop.maxGridSize[0])
			*maxGridSize_x = prop.maxGridSize[0];
	}

	// return the number of CUDA devices
	return device_num;
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
		free(ret);
		return NULL;
	}

	return ret;
}

// free_dvector on GPU
void _bncuda_free_d(double *dev_val)
{
	cudaFree(dev_val);
}


// copy a float val from device
float _bncuda_get_f(float *dev_val)
{
	static float ret;

	cudaMemcpy((void *)&ret, (void *)(dev_val), sizeof(float), cudaMemcpyDeviceToHost);

	return ret;
}

// copy a double val from device
double _bncuda_get_d(double *dev_val)
{
	static double ret;	

	cudaMemcpy((void *)&ret, (void *)(dev_val), sizeof(double), cudaMemcpyDeviceToHost);

	return ret;
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
}

// Initialize & set
double * _bncuda_init_set_d(double val)
{
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

/* double */

// copy dmatrix to device(GPU)
void _bncuda_set_dmatrix(DMatrix dev_mat, DMatrix host_mat)
{
	cudaError_t cuda_error;

	cuda_error = cudaMemcpy((void *)(dev_mat->element), (void *)(host_mat->element), (int)(host_mat->row_dim * host_mat->col_dim) * sizeof(double), cudaMemcpyHostToDevice);
	if(cuda_error != cudaSuccess)
	{
		printf("ERROR: _bncuda_set_dmatrix: host -> dev failed!\n");
		return;
	}
}

// copy vector to device memory
void _bncuda_set_dvector(DVector dev_vec, DVector host_vec)
{
	cudaError_t cuda_error;

	cuda_error = cudaMemcpy((void *)(dev_vec->element), (void *)(host_vec->element), (int)(host_vec->dim) * sizeof(double), cudaMemcpyHostToDevice);
	if(cuda_error != cudaSuccess)
	{
		printf("ERROR: _bncuda_set_dvector: host -> dev failed!\n");
		return;
	}
}


// copy matrix to ram from device memory
void _bncuda_get_dmatrix(DMatrix host_mat, DMatrix dev_mat)
{
	cudaError_t cuda_error;

	cuda_error = cudaMemcpy((void *)(host_mat->element), (void *)(dev_mat->element), (int)(host_mat->row_dim * host_mat->col_dim) * sizeof(double), cudaMemcpyDeviceToHost);
	if(cuda_error != cudaSuccess)
	{
		printf("ERROR: _bncuda_get_dmatrix: dev -> host failed!\n");
		return;
	}
}

// copy matrix to ram from device memory
void _bncuda_get_dvector(DVector host_vec, DVector dev_vec)
{
	cudaError_t cuda_error;

	cuda_error = cudaMemcpy((void *)(host_vec->element), (void *)(dev_vec->element), (int)(host_vec->dim) * sizeof(double), cudaMemcpyDeviceToHost);
	if(cuda_error != cudaSuccess)
	{
		printf("ERROR: _bncuda_get_dvector: dev -> host failed!\n");
		return;
	}
}

/* Float */

// copy matrix to device(GPU)
void _bncuda_set_fmatrix(FMatrix dev_mat, FMatrix host_mat)
{
	cudaError_t cuda_error;

	cuda_error = cudaMemcpy((void *)(dev_mat->element), (void *)(host_mat->element), (int)(host_mat->row_dim * host_mat->col_dim) * sizeof(float), cudaMemcpyHostToDevice);
	if(cuda_error != cudaSuccess)
	{
		printf("ERROR: _bncuda_set_d\fmatrix: host -> dev failed!\n");
		return;
	}
}

// copy vector to device memory
void _bncuda_set_fvector(FVector dev_vec, FVector host_vec)
{
	cudaError_t cuda_error;

	cuda_error = cudaMemcpy((void *)(dev_vec->element), (void *)(host_vec->element), (int)(host_vec->dim) * sizeof(float), cudaMemcpyHostToDevice);
	if(cuda_error != cudaSuccess)
	{
		printf("ERROR: _bncuda_set_fvector: host -> dev failed!\n");
		return;
	}
}


// copy matrix to ram from device memory
void _bncuda_get_fmatrix(FMatrix host_mat, FMatrix dev_mat)
{
	cudaError_t cuda_error;

	cuda_error = cudaMemcpy((void *)(host_mat->element), (void *)(dev_mat->element), (int)(host_mat->row_dim * host_mat->col_dim) * sizeof(float), cudaMemcpyDeviceToHost);
	if(cuda_error != cudaSuccess)
	{
		printf("ERROR: _bncuda_get_fmatrix: dev -> host failed!\n");
		return;
	}
}

// copy matrix to ram from device memory
void _bncuda_get_fvector(FVector host_vec, FVector dev_vec)
{
	cudaError_t cuda_error;

	cuda_error = cudaMemcpy((void *)(host_vec->element), (void *)(dev_vec->element), (int)(host_vec->dim) * sizeof(float), cudaMemcpyDeviceToHost);
	if(cuda_error != cudaSuccess)
	{
		printf("ERROR: _bncuda_get_fvector: dev -> host failed!\n");
		return;
	}
}

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

	if(dimension <= 0)
	{
		fprintf(stderr, "ERROR: _bncuda_init_dvector\n");
		return ret;
	}

	ret = (DVector)malloc(sizeof(dvector));
	if(ret == NULL)
		return ret;

/*	cuda_error = cudaMalloc((void **)&ret, sizeof(dvector));
	if(cuda_error != cudaSuccess)
	{
		printf("device memory allocation failed!(dimension = %ld, size = %ld)\n", dimension, sizeof(dvector));
		return NULL;

	}
*/
//	printf("success: allocation of dvector on GPU\n");

	cuda_error = cudaMalloc((void **)&(ret->element), (int)dimension * sizeof(double));
	if(cuda_error != cudaSuccess)
	{
		printf("device memory allocation failed!(dimension = %ld, size = %ld)\n", dimension, dimension * sizeof(double));
		free(ret);
		return NULL;
	}

//	printf("success: allocation of element of dvector on GPU\n");

	/* All 0 */
	cudaMemset(ret, 0, sizeof(double) * (int)dimension); 

//	cudaMemcpy((void *)&dimension, (void *)&(ret->dim), sizeof(long int), cudaMemcpyHostToDevice);
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
	cudaMemset(ret, 0, sizeof(float) * (int)dimension); 

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

float _bncuda_get_fvector_i(FVector dev_vec, long int index)
{
	static float ret;
//	return *(vec->element + index);
	cudaMemcpy((void *)&ret, (void *)(dev_vec->element + index), sizeof(float) * 1, cudaMemcpyDeviceToHost);

	return ret;
}

float * _bncuda_getp_fvector_i(FVector dev_vec, long int index)
{
	static float *ret;	

	ret = (dev_vec->element + index);

	return ret;
}

double _bncuda_get_dvector_i(DVector dev_vec, long int index)
{
	static double ret;	
//	return *(vec->element + index);
	cudaMemcpy((void *)&ret, (void *)(dev_vec->element + index), sizeof(double) * 1, cudaMemcpyDeviceToHost);

	return ret;
}

double * _bncuda_getp_dvector_i(DVector dev_vec, long int index)
{
	static double *ret;	

	ret = (dev_vec->element + index);

	return ret;
}

void _bncuda_set_fvector_i(FVector dev_vec, long int index, float val)
{
//	*(vec->element + index) = val;
	cudaMemcpy((void *)(dev_vec->element + index), (void *)&val, sizeof(float) * 1, cudaMemcpyHostToDevice);

}

void _bncuda_set_dvector_i(DVector dev_vec, long int index, double val)
{
//	*(vec->element + index) = val;
	cudaMemcpy((void *)(dev_vec->element + index), (void *)&val, sizeof(double) * 1, cudaMemcpyHostToDevice);
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
	cudaMemset(ret, 0, sizeof(float) * (int)row_dimension * (int)col_dimension); 

	ret->row_dim = row_dimension;
	ret->col_dim = col_dimension;

	return ret;
}

// allocate dmatrix on GPU
DMatrix _bncuda_init_dmatrix(long int row_dimension, long int col_dimension)
{
	cudaError_t cuda_error;
	DMatrix ret = NULL;

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
	cudaMemset(ret, 0, sizeof(double) * (int)row_dimension * (int)col_dimension); 

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

float _bncuda_get_fmatrix_ij(FMatrix dev_mat, long int row_index, long int col_index)
{
	static float ret;
	//return (float)(*(mat->element + row_index * mat->col_dim + col_index));

	// column-major in cublas!
	cudaMemcpy((void *)&ret, (void *)(dev_mat->element + col_index * dev_mat->row_dim + row_index), sizeof(float) * 1, cudaMemcpyDeviceToHost);

	return ret;
}

double _bncuda_get_dmatrix_ij(DMatrix dev_mat, long int row_index, long int col_index)
{
	static double ret;
//	return *(mat->element + row_index * mat->col_dim + col_index);

	// column-major in cublas!
	cudaMemcpy((void *)&ret, (void *)(dev_mat->element + col_index * dev_mat->row_dim + row_index), sizeof(double) * 1, cudaMemcpyDeviceToHost);

	return ret;
}
double * _bncuda_getp_dmatrix_ij(DMatrix dev_mat, long int row_index, long int col_index)
{
	static double *ret;
//	return *(mat->element + row_index * mat->col_dim + col_index);

	// column-major in cublas!
	//cudaMemcpy((void *)&ret, (void *)(dev_mat->element + col_index * dev_mat->row_dim + row_index), sizeof(double) * 1, cudaMemcpyDeviceToHost);

	ret = (dev_mat->element + row_index * dev_mat->col_dim + col_index);

	return ret;
}

void _bncuda_set_fmatrix_ij(FMatrix dev_mat, long int row_index, long int col_index, float val)
{
	//*(mat->element + row_index * mat->col_dim + col_index) = (float)val;

	// column-major in cublas!
	cudaMemcpy((void *)(dev_mat->element + col_index * dev_mat->row_dim + row_index), (void *)&val, sizeof(float) * 1, cudaMemcpyHostToDevice);
}

void _bncuda_set_dmatrix_ij(DMatrix dev_mat, long int row_index, long int col_index, double val)
{
//	*(mat->element + row_index * mat->col_dim + col_index) = val;

	// column-major in cublas!
	cudaMemcpy((void *)(dev_mat->element + col_index * dev_mat->row_dim + row_index), (void *)&val, sizeof(double) * 1, cudaMemcpyHostToDevice);
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

