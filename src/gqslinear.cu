/********************************************************************************/
/* gqslinear.cu: Double-double and Quadruple precision                          */
/*               Linear Computation Library with GQS and CUDA                   */
/* Copyright (C) 2015 Tomonori Kouya                                            */
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

#include "gdslinear.h" // G[D,Q]DVector, G[D,Q]DMatrix
	/* Pull only declarations here — gdslinear.cu owns the gqs.cu /
	 * common_s.cu device + host definitions for the whole library.
	 * Cross-TU resolution of __device__ symbols relies on -rdc=true
	 * + nvcc -dlink (see src/cuda/Makefile.in). */
	#include "common_s.cuh"
	#include "gqs.cuh"

// initialize gqsvector on CPU(host)
__host__ GQSVector init_gqsvector(long int dim)
{
	long int index;
	GQSVector ret = NULL;
	qsfloat zero = {{0.0f, 0.0f, 0.0f, 0.0f}};

	// callocation
	ret = (gqsvector *)malloc(sizeof(gqsvector));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one GQSVector\n");
		return ret;
	}

	ret->dim = dim;

	ret->element = (gqs_real *)calloc(dim, sizeof(gqs_real));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GQSVector(dim = %ld)\n", dim);
		cudaFree(ret);
		return NULL;
	}

	// set all zeros 
	for(index = 0; index < dim; index++)
		qs2gqs(&ret->element[index], &zero);
		//ret->element[index] = (gqs_real)0.0;

	return ret;
}

// free gqsvector on CPU(HOST)
__host__ void free_gqsvector(GQSVector vec)
{
	free(vec->element);
	free(vec);
}

// initialize gqsvector
__host__ GQSVector init_gqsvector_dev(long int dim)
{
	long int index;
	GQSVector ret = NULL;
	qsfloat zero = {{0.0f, 0.0f, 0.0f, 0.0f}};

	// allocation
	ret = (gqsvector *)malloc(sizeof(gqsvector));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one QSVector\n");
		return ret;
	}

	ret->element = NULL;
	ret->dim = dim;

	//ret->element = (qsfloat *)calloc(dim, sizeof(qsfloat));
	cudaMalloc((void **)&(ret->element), (size_t)(dim * sizeof(gqs_real)));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GQSVector(dim = %ld)\n", dim);
		cudaFree(ret);
		return NULL;
	}

	// set all zeros 
	for(index = 0; index < dim; index++)
		qs2gqs_dev(&(ret->element[index]), &zero);

	return ret;
}

// free qsvector
__host__ void free_gqsvector_dev(GQSVector vec)
{
	cudaFree(vec->element);
	free(vec);
}

// copy QSVector to GQSVector on GPU
// gqsvec on GPU := qsvec
__host__ void subst_gqsvector_dev_qsvec(GQSVector gqsvec_dev, QSVector qsvec)
{
	long int i, dim;

	dim = gqsvec_dev->dim;

	for(i = 0; i < dim; i++)
	{
		/* QSVector::element is SoA: double *element[QSSIZE] (4 limbs).
		 * Materialize a temporary qsfloat (AoS) for qs2gqs_dev. */
		qsfloat tmp_qs;
		tmp_qs.val[0] = qsvec->element[0][i];
		tmp_qs.val[1] = qsvec->element[1][i];
		tmp_qs.val[2] = qsvec->element[2][i];
		tmp_qs.val[3] = qsvec->element[3][i];
		qs2gqs_dev(&(gqsvec_dev->element[i]), &tmp_qs);
	}
}

// copy GQSVector on GPU to QSVector
// qsvec := gqsvec_dev on GPU
__host__ void subst_qsvector_gqsvec_dev(QSVector qsvec, GQSVector gqsvec_dev)
{
	long int i, dim;

	dim = qsvec->dim;

	for(i = 0; i < dim; i++)
	{
		/* SoA layout: see subst_gqsvector_dev_qsvec above. */
		qsfloat tmp_qs;
		gqs2qs_dev(&tmp_qs, &(gqsvec_dev->element[i]));
		qsvec->element[0][i] = tmp_qs.val[0];
		qsvec->element[1][i] = tmp_qs.val[1];
		qsvec->element[2][i] = tmp_qs.val[2];
		qsvec->element[3][i] = tmp_qs.val[3];
	}
}

// print qsvector
__host__ void print_gqsvector_dev(GQSVector dev_vec)
{
	long int index, dim;
	GQSVector host_vec;

	dim = dev_vec->dim;

	host_vec = init_gqsvector(dim);

	cudaMemcpy((void *)(host_vec->element), (void *)(dev_vec->element), size_t(sizeof(gqs_real) * dim), cudaMemcpyDeviceToHost);

	for(index = 0; index < dim; index++)
	{
		printf("%4ld: ", index);

		qsfloat _qsv = gqs_get_qs(host_vec->element[index]);
		std::cout << _qsv.val[0] << " + " << _qsv.val[1] << " + " << _qsv.val[2] << " + " << _qsv.val[3] << "\n";
	}

	free(host_vec);
}

/*************************************************/
/* Vector Calculations for QSVector               */
/*
void add_gqsvector_dev(GQSVector c_dev, GQSVector a_dev, GQSVector b_dev)
void aqd2_gqsvector_dev(GQSVector c_dev, GQSVector a_dev)
void sub_gqsvector_dev(GQSVector c_dev, GQSVector a_dev, GQSVector b_dev)
void sub2_gqsvector_dev(GQSVector c_dev, GQSVector a_dev)
void cmul_gqsvector_dev(GQSVector c_dev, gqs_real val_dev, GQSVector a_dev)
void cmul2_gqsvector_dev(GQSVector c_dev, gqs_real val_dev)
void add_cmul_gqsvector_dev(GQSVector c_dev, GQSVector a_dev, gqs_real val_dev, GQSVector b_dev)
double ip_gqsvector_dev(GQSVector a, GQSVector b_dev)
double norm1_gqsvector_dev(GQSVector a_dev)
double norm2_gqsvector_dev(GQSVector a_dev)
double normi_gqsvector_dev(GQSVector a_dev)
void subst_gqsvector_dev(GQSVector c_dev, GQSVector a_dev)
*/
/*************************************************/
/* c = a + b */
__global__ void _bncu_add_gqsvector(gqs_real *c_dev_element, gqs_real *a_dev_element, gqs_real *b_dev_element, long int dim)
{
	int index;

	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim)
	{
		c_dev_element[index] = a_dev_element[index] + b_dev_element[index];
		index += blockDim.x * gridDim.x;
	}

	__syncthreads();
}

__host__ void add_gqsvector_dev(GQSVector c_dev, GQSVector a_dev, GQSVector b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if((a_dev->dim != b_dev->dim) || (c_dev->dim != a_dev->dim) || (c_dev->dim != b_dev->dim))
	{
		fprintf(stderr, "ERROR: add_gqsvector_dev\n");
		return;
	}

/*	for(i = 0; i < c->dim; i++)
	{
		c->element[i] = a->element[i] + b->element[i];
	}
*/
	_bncu_add_gqsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, c_dev->dim);
}


/* c = a - b */
__global__ void _bncu_sub_gqsvector(gqs_real *c_dev_element, gqs_real *a_dev_element, gqs_real *b_dev_element, long int dim)
{
	int index;

	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim)
	{
		c_dev_element[index] = a_dev_element[index] - b_dev_element[index];
		index += blockDim.x * gridDim.x;
	}

	__syncthreads();
}

__host__ void sub_gqsvector_dev(GQSVector c_dev, GQSVector a_dev, GQSVector b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if((a_dev->dim != b_dev->dim) || (c_dev->dim != a_dev->dim) || (c_dev->dim != b_dev->dim))
	{
		fprintf(stderr, "ERROR: sub_gqsvector_dev\n");
		return;
	}

/*	for(i = 0; i < c->dim; i++)
	{
		c->element[i] = a->element[i] - b->element[i];
	}
*/
	_bncu_sub_gqsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, c_dev->dim);
}

// cmul vector (val, core)
// dev_ret := dev_val * dev_a on GPU
__global__ void _bncu_cmul_gqsvector(gqs_real *dev_ret_element, gqs_real val, gqs_real *dev_a_element, long int dim)
{
	int index;

	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim)
	{
		dev_ret_element[index] = val * dev_a_element[index];
		index += blockDim.x * gridDim.x;
	}

	__syncthreads();

}

/* c = val * a */
__host__ void cmul_gqsvector_dev(GQSVector c_dev, gqs_real val, GQSVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if(c_dev->dim != a_dev->dim)
	{
		fprintf(stderr, "ERROR: cmul_gqsvector_dev\n");
		return;
	}

/*	for(i = 0; i < c_dev->dim; i++)
	{
		c->element[i] = val * a->element[i];
	}
*/

	_bncu_cmul_gqsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, val, a_dev->element, c_dev->dim);

}

// subst vector (val, core)
// dev_ret := dev_vec on GPU
__global__ void _bncu_subst_gqsvector(gqs_real *dev_ret_element, gqs_real *dev_vec_element, long int dim)
{
	int index;

	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim)
	{
		dev_ret_element[index] = dev_vec_element[index];
		index += blockDim.x * gridDim.x;
	}

	__syncthreads();

}

/* c := vec */
__host__ void subst_gqsvector_dev(GQSVector ret_dev, GQSVector vec_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if(ret_dev->dim != vec_dev->dim)
	{
		fprintf(stderr, "ERROR: subst_gqsvector_dev\n");
		return;
	}

	_bncu_subst_gqsvector<<< num_blocks_per_grid, num_threads_per_block >>>(ret_dev->element, vec_dev->element, ret_dev->dim);

}

// set0 
// dev_ret := 0 on GPU
__global__ void _bncu_set0_gqsvector(gqs_real *dev_ret_element, long int dim)
{
	int index;

	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim)
	{
		SET0_GQS(dev_ret_element[index]);
		index += blockDim.x * gridDim.x;
	}

	__syncthreads();

}

/* c := 0 */
__host__ void set0_gqsvector_dev(GQSVector ret_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	_bncu_set0_gqsvector<<< num_blocks_per_grid, num_threads_per_block >>>(ret_dev->element, ret_dev->dim);
}

// ret[i] := a_vec[i] * b_vec[i]
__global__ void _bncu_mul_gqsvector(gqs_real *ret_dev_element, gqs_real *a_dev_element, gqs_real *b_dev_element, long int dim)
{
	int index;

	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim)
	{
		ret_dev_element[index] = a_dev_element[index] * b_dev_element[index];
		index += blockDim.x * gridDim.x;
	}

	__syncthreads();
}

// ret[i] := vec[i]^2
__global__ void _bncu_sqr_gqsvector(gqs_real *ret_dev_element, gqs_real *vec_dev_element, long int dim)
{
	int index;

	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim)
	{
		ret_dev_element[index] = vec_dev_element[index] * vec_dev_element[index];
		index += blockDim.x * gridDim.x;
	}

	__syncthreads();
}

// ret[i] := sqrt(vec[i])
__global__ void _bncu_sqrt_gqsvector(gqs_real *ret_dev_element, gqs_real *vec_dev_element, long int dim)
{
	int index;

	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim)
	{
		ret_dev_element[index] = (gqs_real)sqrt((gqs_real)vec_dev_element[index]);
		index += blockDim.x * gridDim.x;
	}

	__syncthreads();
}

// ret[i] := abs(vec[i])
__global__ void _bncu_abs_gqsvector(gqs_real *ret_dev_element, gqs_real *vec_dev_element, long int dim)
{
	int index;

	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim)
	{
		ret_dev_element[index] = abs(vec_dev_element[index]);
		index += blockDim.x * gridDim.x;
	}

	__syncthreads();
}

// reduction to aqd
// ret[block_index] := sum^dim_{i=1} vec[i]
__global__ void _bncu_add_reduct_gqsvector(gqs_real *ret_dev, gqs_real *vec_dev_element, long int dim)
{
	int index, cache_index;
	__shared__ gqs_real cache[MAX_NUM_THREADS_PER_BLOCK];

	// Inside Block

	index = threadIdx.x + blockIdx.x * blockDim.x;
	cache_index = threadIdx.x;

	SET0_GQS(cache[cache_index]);

	while(index < dim)
	{
		cache[cache_index] = cache[cache_index] + vec_dev_element[index];
		index += blockDim.x * gridDim.x;
	}

	__syncthreads();

	// reduction
	index = blockDim.x / 2;
	while(index > 0)
	{
		if(cache_index < index)
			cache[cache_index] = cache[cache_index] + cache[cache_index + index];

		__syncthreads();

		index /= 2;
	}

	__syncthreads();

	if(cache_index == 0)
		ret_dev[blockIdx.x] = cache[0];
}

// reduction to get maximum
// ret[block_index] := max_{i=1, max} vec[i]
__global__ void _bncu_max_reduct_gqsvector(gqs_real *ret_dev, gqs_real *vec_dev_element, long int dim)
{
	int index, cache_index;
	__shared__ gqs_real cache[MAX_NUM_THREADS_PER_BLOCK];

	// Inside Block

	index = threadIdx.x + blockIdx.x * blockDim.x;
	cache_index = threadIdx.x;

	cache[cache_index] = vec_dev_element[0];

	while(index < dim)
	{
		if(cache[cache_index] < vec_dev_element[index])
			cache[cache_index] = vec_dev_element[index];

		index += blockDim.x * gridDim.x;
	}

	__syncthreads();

	// reduction
	index = blockDim.x / 2;
	while(index > 0)
	{
		if(cache_index < index)
		{
			if(cache[cache_index] < cache[cache_index + index])
				cache[cache_index] = cache[cache_index + index];
		}

		__syncthreads();

		index /= 2;
	}

	__syncthreads();

	if(cache_index == 0)
		ret_dev[blockIdx.x] = cache[0];
}

/* (a, b) */
__host__ void ip_gqsvector_dev(gqs_real *ret_dev, GQSVector a_dev, GQSVector b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gqs_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim != b_dev->dim)
	{
		fprintf(stderr, "ERROR: ip_gqsvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gqs_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gqs_real) * num_blocks_per_grid));

	// c[i] := a[i] * b[i]
	_bncu_mul_gqsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, b_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	//block_dim = 1;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gqsvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gqs_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret_dev := block_cache_dev[0]
	cudaMemcpy((void *)ret_dev, (void *)&(block_cache_dev[0]), sizeof(gqs_real), cudaMemcpyDeviceToDevice);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// norm2
__host__ void norm2_gqsvector_dev(gqs_real *ret_dev, GQSVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gqs_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim <= 0)
	{
		fprintf(stderr, "ERROR: norm2_gqsvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gqs_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gqs_real) * num_blocks_per_grid));

	// c[i] := a[i]^2
	_bncu_sqr_gqsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gqsvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gqs_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret := sqrt(ret);
	_bncu_sqrt_gqsvector<<<1, 1>>>(ret_dev, block_cache_dev, 1);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// norm1
__host__ void norm1_gqsvector_dev(gqs_real *ret_dev, GQSVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gqs_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim <= 0)
	{
		fprintf(stderr, "ERROR: norm1_gqsvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gqs_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gqs_real) * num_blocks_per_grid));

	// c[i] := abs(a[i])
	_bncu_abs_gqsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gqsvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gqs_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret_dev := block_cache_dev[0]
	cudaMemcpy((void *)ret_dev, (void *)&(block_cache_dev[0]), sizeof(gqs_real), cudaMemcpyDeviceToDevice);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// norm_i
__host__ void normi_gqsvector_dev(gqs_real *ret_dev, GQSVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gqs_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim <= 0)
	{
		fprintf(stderr, "ERROR: normi_gqsvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gqs_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gqs_real) * num_blocks_per_grid));

	// c[i] := abs(a[i])
	_bncu_abs_gqsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_max_reduct_gqsvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gqs_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret_dev := block_cache_dev[0]
	cudaMemcpy((void *)ret_dev, (void *)&(block_cache_dev[0]), sizeof(gqs_real), cudaMemcpyDeviceToDevice);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// GQS matrix

// set a zero matrix
__host__ void set0_gqsmatrix(GQSMatrix mat, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i, j;
	qsfloat qszero = {{0.0f, 0.0f, 0.0f, 0.0f}};
	gqs_real zero;

	qs2gqs(&zero, &qszero);

	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
			mat->element[i * mat->col_dim + j] = zero;
	}
}

// initialize qsvector
GQSMatrix init_gqsmatrix(long int row_dim, long int col_dim)
{
	long int row_index, col_index;
	GQSMatrix ret = NULL;
	qsfloat qszero = {{0.0f, 0.0f, 0.0f, 0.0f}};
	gqs_real zero;

	qs2gqs(&zero, &qszero);

	// callocation
	ret = (gqsmatrix *)malloc(sizeof(gqsmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one GQSMatrix\n");
		return ret;
	}

	ret->row_dim = row_dim;
	ret->col_dim = col_dim;

	ret->element = (gqs_real *)calloc(row_dim * col_dim, sizeof(gqs_real));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GQSMatrix(row_dim, col_dim = %ld, %ld)\n", row_dim, col_dim);
		free(ret);
		return NULL;
	}

	// set all zeros 
	for(row_index = 0; row_index < row_dim; row_index++)
	{
		for(col_index = 0; col_index < col_dim; col_index++)
			ret->element[row_index * col_dim + col_index] = zero;
	}

	return ret;
}

// free qsvector
void free_gqsmatrix(GQSMatrix mat)
{
	free(mat->element);
	free(mat);
}

// initialize qsvector
GQSMatrix init_gqsmatrix_dev(long int row_dim, long int col_dim)
{
	long int row_index, col_index;
	GQSMatrix ret = NULL;
	qsfloat qszero = {{0.0f, 0.0f, 0.0f, 0.0f}};
	gqs_real zero;

	qs2gqs(&zero, &qszero);

	// callocation
	ret = (gqsmatrix *)malloc(sizeof(gqsmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one GQSMatrix\n");
		return ret;
	}

	ret->row_dim = row_dim;
	ret->col_dim = col_dim;

	ret->element = (gqs_real *)calloc(row_dim * col_dim, sizeof(gqs_real));
	cudaMalloc((void **)&(ret->element), (size_t)(row_dim * col_dim * sizeof(gqs_real)));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GQSMatrix(row_dim, col_dim = %ld, %ld)\n", row_dim, col_dim);
		free(ret);
		return NULL;
	}

	// set all zeros 
	for(row_index = 0; row_index < row_dim; row_index++)
	{
		for(col_index = 0; col_index < col_dim; col_index++)
			gqs2gqs_dev(&(ret->element[row_index * col_dim + col_index]), &zero);
	}

	return ret;
}

// free qsvector
void free_gqsmatrix_dev(GQSMatrix mat)
{
	cudaFree(mat->element);
	free(mat);
}

// print qsvector
void print_gqsmatrix_dev(GQSMatrix mat)
{
	long int row_index, col_index;
	qsfloat qsval;

	for(row_index = 0; row_index < mat->row_dim; row_index++)
	{
		for(col_index = 0; col_index < mat->col_dim; col_index++)
		{
			printf("%ld, %ld: ", row_index, col_index);
			gqs2qs_dev(&qsval, &(mat->element[row_index * mat->col_dim + col_index]));
			std::cout << qsval.val[0] << " + " << qsval.val[1] << " + " << qsval.val[2] << " + " << qsval.val[3] << "\n";
		}
	}
}


__global__ void _bncu_mul_gqsmatrix(gqs_real *ret_dev_element, long int row_dim, long int col_dim, long int mid_dim, gqs_real *a_dev_element, gqs_real *b_dev_element)
{
	long int i, j, k;
	gqs_real tmp, tmp1;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	//for(i = 0; i < row_dim; i++)
	while(i < row_dim)
	{
		for(j = 0; j < col_dim; j++)
		{
			SET0_GQS(tmp1); // = (gqs_real)0.0;
			SET0_GQS(tmp); // = (gqs_real)0.0;
			for(k = 0; k < mid_dim; k++)
			{
				tmp1 = a_dev_element[i * mid_dim + k] * b_dev_element[k * col_dim + j];
				tmp = tmp + tmp1;
			}

			ret_dev_element[i * col_dim + j] = tmp;

		}
		i += blockDim.x * gridDim.x;
	}
}

// matrix multiplication
// ret := A * B
void mul_gqsmatrix_dev(GQSMatrix ret_dev, GQSMatrix a_dev, GQSMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long row_dim, col_dim, mid_dim;

	/* dimension check */
	if((ret_dev->row_dim != a_dev->row_dim) || (ret_dev->col_dim != b_dev->col_dim) || (a_dev->col_dim != b_dev->row_dim))
	{
		fprintf(stderr, "ERROR: mul_gqsmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret_dev->row_dim, ret_dev->col_dim, a_dev->row_dim, a_dev->col_dim, b_dev->row_dim, b_dev->col_dim);
		return;
	}

	row_dim = ret_dev->row_dim;
	col_dim = ret_dev->col_dim;
	mid_dim = a_dev->col_dim;

	_bncu_mul_gqsmatrix<<<num_blocks_per_grid, num_threads_per_block>>>(ret_dev->element, row_dim, col_dim, mid_dim, a_dev->element, b_dev->element);

}

// copy QSMatrix to GQSMatrix on GPU
// gqsvec on GPU := qsvec
__host__ void subst_gqsmatrix_dev_qsmat(GQSMatrix gqsmat_dev, QSMatrix qsmat)
{
	long int i, j, row_dim, col_dim;

	row_dim = gqsmat_dev->row_dim;
	col_dim = gqsmat_dev->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			/* QSMatrix::element is SoA: double *element[QSSIZE].
			 * Materialize an AoS qsfloat for qs2gqs_dev. */
			qsfloat tmp_qs;
			long int idx = i * col_dim + j;
			tmp_qs.val[0] = qsmat->element[0][idx];
			tmp_qs.val[1] = qsmat->element[1][idx];
			tmp_qs.val[2] = qsmat->element[2][idx];
			tmp_qs.val[3] = qsmat->element[3][idx];
			qs2gqs_dev(&(gqsmat_dev->element[idx]), &tmp_qs);
		}
	}
}

// copy GQSMatrix on GPU to QSMatrix
// qsvec := gqsmat_dev on GPU
__host__ void subst_qsmatrix_gqsmat_dev(QSMatrix qsmat, GQSMatrix gqsmat_dev)
{
	long int i, j, row_dim, col_dim;

	row_dim = gqsmat_dev->row_dim;
	col_dim = gqsmat_dev->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			/* SoA layout: see subst_gqsmatrix_dev_qsmat above. */
			qsfloat tmp_qs;
			long int idx = i * col_dim + j;
			gqs2qs_dev(&tmp_qs, &(gqsmat_dev->element[idx]));
			qsmat->element[0][idx] = tmp_qs.val[0];
			qsmat->element[1][idx] = tmp_qs.val[1];
			qsmat->element[2][idx] = tmp_qs.val[2];
			qsmat->element[3][idx] = tmp_qs.val[3];
		}
	}
}

// Frobenius norm
__host__ void normf_gqsmatrix_dev(gqs_real *ret_dev, GQSMatrix mat_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, tmp_vec_dev_dim, total_dim;
//	gqs_real tmp, tmp1;
	gqs_real *block_cache_dev; // partial sum per block
	gqs_real *tmp_vec_dev; // square mul of mat

	total_dim = mat_dev->row_dim * mat_dev->col_dim; // total dimension as one vector

	// initialize block_cache_dev, tmp_vec_dev
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gqs_real) * num_blocks_per_grid));
	cudaMalloc((void **)&tmp_vec_dev, (size_t)(sizeof(gqs_real) * total_dim));

	// mat[i][j]^2
	//_bncu_mul_gqsvector<<<num_blocks_per_grid, num_threads_per_block>>>(tmp_vec_dev, mat_dev->element, mat_dev->element, total_dim);
	_bncu_sqr_gqsvector<<<num_blocks_per_grid, num_threads_per_block>>>(tmp_vec_dev, mat_dev->element, total_dim);

	// reduction to aqd
	block_dim = num_blocks_per_grid;
//	block_dim = 1;
	thread_dim = num_threads_per_block;
	tmp_vec_dev_dim = total_dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gqsvector<<<block_dim, thread_dim>>>(block_cache_dev, tmp_vec_dev, tmp_vec_dev_dim);

		if(block_dim <= 1)
			break;

		tmp_vec_dev_dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)tmp_vec_dev, (void *)block_cache_dev, (size_t)(sizeof(gqs_real) * tmp_vec_dev_dim), cudaMemcpyDeviceToDevice);

	}

	// ret := sqrt(ret);
	_bncu_sqrt_gqsvector<<<1, 1>>>(ret_dev, block_cache_dev, 1);

	// free
	cudaFree(block_cache_dev);
	cudaFree(tmp_vec_dev);

/*
	SET0_GQS(*ret); // = (gqs_real)0.0;
	SET0_GQS(tmp); // = (gqs_real)(int)0;
	SET0_GQS(tmp1); // = (gqs_real)(int)0;

	dim = mat->row_dim * mat->col_dim;

	for(i = 0; i < dim ; i++)
	{
		//tmp += mat->element[i] * mat->element[i];
		//tmp += gqs_real::mul((gqs_real)(mat->element[i]), (gqs_real)(mat->element[i]));
		tmp1 = (gqs_real)(mat->element[i]) * (gqs_real)(mat->element[i]);
		tmp = tmp + tmp1;
	}

	//*ret = sqrt(tmp);
	tmp = (gqs_real)sqrt(tmp);
	*ret = tmp;
*/
}

/*************************************************/
/* Matrix Caluculations for GQSMatrix            */
/*
void normf_gqsmatrix(double ret[DSSIZE], GQSMatrix mat)
void norm1_gqsmatrix(double ret[DSSIZE], GQSMatrix mat)
void normi_gqsmatrix(double ret[DSSIZE], GQSMatrix mat)
void add_gqsmatrix(GQSMatrix c, GQSMatrix a, GQSMatrix b);
void sub_gqsmatrix(GQSMatrix c, GQSMatrix a, GQSMatrix b);
void mul_gqsmatrix(GQSMatrix c, GQSMatrix a, GQSMatrix b);
void mul_gqsmatrix_qsvec(QSVector v, GQSMatrix a, QSVector vb)
void mul_gqsmatrixt_qsvec(QSVector v, GQSMatrix a, QSVector vb)
void transpose_gqsmatrix(GQSMatrix c, GQSMatrix a);
void inv_gqsmatrix(GQSMatrix a);
void subst_mpfmatrux(GQSMatrix c, GQSMatrix a);
*/
/*************************************************/

#if 0
/* Infinity Norm of Matrix */
void normi_gqsmatrix(gqs_real *ret, GQSMatrix mat, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i, j;
	gqs_real sum;

	SET0_GQS(*ret); // = 0.0;

	for(i = 0; i < mat->row_dim; i++)
	{
		SET0_GQS(sum); //  = 0.0;
		for(j = 0; j < mat->col_dim; j++)
			sum = sum + abs(mat->element[i * mat->row_dim + j]);

		if(*ret < sum)
			*ret = sum;

	}

	return;
}

/* 1 Norm of Matrix */
void norm1_gqsmatrix(gqs_real *ret, GQSMatrix mat, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i, j;
	gqs_real sum;

	SET0_GQS(*ret); // = 0.0;

	for(j = 0; j < mat->col_dim; j++)
	{
		SET0_GQS(sum); // = 0.0;
		for(i = 0; i < mat->row_dim; i++)
			sum = sum + abs(mat->element[i * mat->col_dim + j]);

		if(*ret < sum)
			*ret = sum;

	}

	return;
}
#endif // 0

/* c := a + b */
void add_gqsmatrix_dev(GQSMatrix c_dev, GQSMatrix a_dev, GQSMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int row_dim, col_dim; //, index;

	/* check row_dim */
	if((a_dev->row_dim != b_dev->row_dim) || (b_dev->row_dim != c_dev->row_dim) || (c_dev->row_dim != a_dev->row_dim))
	{
		fprintf(stderr, "ERROR: add_gqsmatrix_dev\n");
		return;
	}
	row_dim = c_dev->row_dim;

	/* check col_dim */
	if((a_dev->col_dim != b_dev->col_dim) || (b_dev->col_dim != c_dev->col_dim) || (c_dev->col_dim != a_dev->col_dim))
	{
		fprintf(stderr, "ERROR: add_gqsmatrix_dev\n");
		return;
	}
	col_dim = c_dev->col_dim;

	_bncu_add_gqsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, row_dim * col_dim);

/*	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			index = i * col_dim + j;
			c_dev->element[index] = a_dev->element[index] + b_dev->element[index];
		}
	}
*/
}

/* c := a - b */
void sub_gqsmatrix_dev(GQSMatrix c_dev, GQSMatrix a_dev, GQSMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int row_dim, col_dim; //, index;

	/* check row_dim */
	if((a_dev->row_dim != b_dev->row_dim) || (b_dev->row_dim != c_dev->row_dim) || (c_dev->row_dim != a_dev->row_dim))
	{
		fprintf(stderr, "ERROR: sub_gqsmatrix_dev\n");
		return;
	}
	row_dim = c_dev->row_dim;

	/* check col_dim */
	if((a_dev->col_dim != b_dev->col_dim) || (b_dev->col_dim != c_dev->col_dim) || (c_dev->col_dim != a_dev->col_dim))
	{
		fprintf(stderr, "ERROR: sub_gqsmatrix_dev\n");
		return;
	}
	col_dim = c_dev->col_dim;

	_bncu_sub_gqsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, row_dim * col_dim);

}

/* c := sc * a */
void cmul_gqsmatrix_dev(GQSMatrix c_dev, gqs_real sc, GQSMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int row_dim, col_dim; //, index;

	/* check row_dim */
	if(a_dev->row_dim != c_dev->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_gqsmatrix_dev\n");
		return;
	}
	row_dim = c_dev->row_dim;

	/* check col_dim */
	if(a_dev->col_dim != c_dev->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_gqsmatrix_dev\n");
		return;
	}
	col_dim = c_dev->col_dim;

	_bncu_cmul_gqsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, sc, a_dev->element, row_dim * col_dim);


}

/* c = a^T */
__global__ void _bncu_transpose_gqsmatrix(gqs_real *c_dev_element, gqs_real *a_dev_element, long int row_dim, long int col_dim)
{
	int i, j;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	//for(i = 0; i < c->row_dim; i++)
	while(i < row_dim)
	{
		for(j = 0; j < col_dim; j++)
			c_dev_element[i * col_dim + j] = a_dev_element[j * col_dim + i];

		__syncthreads();

		i += blockDim.x * gridDim.x;
	}
}

void transpose_gqsmatrix_dev(GQSMatrix c_dev, GQSMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	//long int i, j;

	/* Check Dimensions */
	if((c_dev->row_dim != a_dev->col_dim) || (c_dev->col_dim != a_dev->row_dim))
	{
		fprintf(stderr, "ERROR: transpose_gqsmatrix_dev\n");
		return;
	}
	
	_bncu_transpose_gqsmatrix<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, c_dev->row_dim, c_dev->col_dim);

/*	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			c->element[i * c->col_dim + j] = a->element[j * a->col_dim + i];
	}
*/
}

/* c := a */
void subst_gqsmatrix_dev(GQSMatrix c_dev, GQSMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	//long int i, j, index;

	if((c_dev->row_dim != a_dev->row_dim) || (c_dev->col_dim != a_dev->col_dim))
	{
		fprintf(stderr, "ERROR: subst_gqsmatrix_dev\n");
		return;
	}

	_bncu_subst_gqsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, c_dev->row_dim * c_dev->col_dim);

}

/* c := 0 */
void set0_gqsmatrix_dev(GQSMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block)
{
//	long int i, j;

	_bncu_set0_gqsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, c_dev->row_dim * c_dev->col_dim);

}

/* set (i, j)-element on GPU from qsfloat */
void set_gqsmatrix_ij_dev(GQSMatrix mat_dev, long int row_index, long int col_index, qsfloat val)
{
	if((row_index >= 0) && (row_index < mat_dev->row_dim) && (col_index >= 0) && (col_index < mat_dev->col_dim))
		qs2gqs_dev(&(mat_dev->element[row_index * mat_dev->col_dim + col_index]), &val);
}

/* c := I */
void setI_gqsmatrix_dev(GQSMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i;
	qsfloat one;

	one.val[0] = 1.0f; one.val[1] = 0.0f; one.val[2] = 0.0f; one.val[3] = 0.0f;

	_bncu_set0_gqsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, c_dev->row_dim * c_dev->col_dim);

	for(i = 0; i < c_dev->row_dim; i++)
		set_gqsmatrix_ij_dev(c_dev, i, i, one);
/*
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			SET0_GQS(c->element[i * c->col_dim + j]); // = 0.0;
		if(i < c->col_dim)
			SET1_GQS(c->element[i * c->col_dim + i]); // = 1.0;
	}
*/
}

/* v := a * vb */
__global__ void _bncu_mul_gqsmatrix_gqsvec(gqs_real *v_element, gqs_real *a_element, gqs_real *vb_element, long int row_dim, long int col_dim)
{
	int j, index;
	gqs_real tmp;

	index = threadIdx.x + blockIdx.x * blockDim.x;

	//for(i = 0; i < row_dim; i++)
	while(index < row_dim)
	{
		SET0_GQS(tmp); // = 0.0;
		for(j = 0; j < col_dim; j++)
			tmp = tmp + a_element[index * col_dim + j] * vb_element[j];

		v_element[index] = tmp;

		__syncthreads();

		index += blockDim.x * gridDim.x;
	}
}

void mul_gqsmatrix_gqsvec(GQSVector v, GQSMatrix a, GQSVector vb, int num_blocks_per_grid, int num_threads_per_block)
{
	/* Check Dimension */
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: mul_gqsmatrix_qsvec\n");
		return;
	}

	_bncu_mul_gqsmatrix_gqsvec<<<num_blocks_per_grid, num_threads_per_block>>>(v->element, a->element, vb->element, a->row_dim, a->col_dim);

/*	for(i = 0; i < a->row_dim; i++)
	{
		SET0_GQS(tmp); // = 0.0;
		for(j = 0; j < a->col_dim; j++)
			tmp = tmp + a->element[i * a->col_dim + j] * vb->element[j];

		v->element[i] = tmp;
	}
*/
}

/* v := a^T * vb */
__global__ void _bncu_mul_gqsmatrixt_gqsvec(gqs_real *v_element, gqs_real *a_element, gqs_real *vb_element, long int row_dim, long int col_dim)
{
	int j, index;
	gqs_real tmp;

	index = threadIdx.x + blockIdx.x * blockDim.x;

	//for(i = 0; i < col_dim; i++)
	while(index < col_dim)
	{
		SET0_GQS(tmp); // = 0.0;
		for(j = 0; j < row_dim; j++)
			tmp = tmp + a_element[j * row_dim + index] * vb_element[j];

		v_element[index] = tmp;

		__syncthreads();

		index += blockDim.x * gridDim.x;
	}
}

__host__ void mul_gqsmatrixt_gqsvec(GQSVector v, GQSMatrix a, GQSVector vb, int num_blocks_per_grid, int num_threads_per_block)
{

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_gqsmatrixt_gqsvec\n");
		return;
	}

	_bncu_mul_gqsmatrixt_gqsvec<<<num_blocks_per_grid, num_threads_per_block>>>(v->element, a->element, vb->element, a->row_dim, a->col_dim);

/*	for(i = 0; i < a->col_dim; i++)
	{
		SET0_GQS(tmp); // = 0.0;
		for(j = 0; j < a->row_dim; j++)
			tmp = tmp + a->element[j * a->row_dim + i] * vb->element[j];

		v->element[i] = tmp;
	}
*/
}

// Test main function
// NOTE: disabled — uses qd-library API not available for the POD qsfloat type.
// Use bench_gdtqs_strassen instead.
#if 0

using namespace std;

//#define ROW_DIM 10
//#define ROW_DIM 32
//#define ROW_DIM 128
#define ROW_DIM 512
#define COL_DIM ROW_DIM

int main()
{
	int i, j;
	QSVector qsvec_a, qsvec_b, qsvec_c;
	QSMatrix qsmat_a, qsmat_b, qsmat_c;
	qsfloat qsval;
	int num_blocks, num_threads;
	GQSVector gqsvec_a, gqsvec_b, gqsvec_c;
	gqs_real gqsval, *ptr_gqsval_dev;
	GQSMatrix gqsmat_a, gqsmat_b, gqsmat_c;
	

	// Initialize QD library
	fpu_fix_start(NULL);

	// DD & GQS
	qsvec_a = init_qsvector(ROW_DIM);
	qsvec_b = init_qsvector(ROW_DIM);
	qsvec_c = init_qsvector(ROW_DIM);

	for(i = 0; i < ROW_DIM; i++)
	{
		// val := sqrt((i + 1));
		qsvec_a->element[i] = (qsfloat)sqrt((qsfloat)(i + 1));
		qsvec_b->element[i] = (qsfloat)sqrt((qsfloat)(i + 1));
		//qsvec->element[i] = sqrt(qsval);
	}

	print_qsvector(qsvec_a);
	norm2_qsvector(&qsval, qsvec_a);
	printf("DD : ||vec||_2 = ");
	cout.precision(qsfloat::_ndigits);
	cout << qsval << "\n";

	// GQS start!
	GQSStart();

//	num_blocks = 1;  // #blocks per grid
//	num_blocks = 2;  // #blocks per grid
	num_blocks = 4;  // #blocks per grid
//	num_threads = 16; // #threads per block
	num_threads = 64; // #threads per block

	gqsvec_a = init_gqsvector_dev(ROW_DIM);
	gqsvec_b = init_gqsvector_dev(ROW_DIM);
	gqsvec_c = init_gqsvector_dev(ROW_DIM);
	cudaMalloc((void **)&ptr_gqsval_dev, sizeof(gqs_real));

	// gqsvec := qsvec
	subst_gqsvector_dev_qsvec(gqsvec_a, qsvec_a);
	subst_gqsvector_dev_qsvec(gqsvec_b, qsvec_b);
	//subst_qsvector_gqsvec_dev(qsvec_c, gqsvec_c);

	// ----------
	// c := a + b
	// ----------
	// DD
	printf(" DD: c := a + b\n");
	add_qsvector(qsvec_c, qsvec_a, qsvec_b);
	print_qsvector(qsvec_c);

	// GQS
	printf("GQS: c := a + b\n");
	add_gqsvector_dev(gqsvec_c, gqsvec_a, gqsvec_b, num_blocks, num_threads);
	print_gqsvector_dev(gqsvec_c);

	// ----------
	// c := a - b
	// ----------
	// DD
	printf(" DD: c := a - b\n");
	sub_qsvector(qsvec_c, qsvec_a, qsvec_b);
	print_qsvector(qsvec_c);

	// GQS
	printf("GQS: c := a - b\n");
	sub_gqsvector_dev(gqsvec_c, gqsvec_a, gqsvec_b, num_blocks, num_threads);
	print_gqsvector_dev(gqsvec_c);

	// ----------
	// c := val * a
	// ----------
	// DD
	qsval = (qsfloat)2.0;
	printf(" DD: c := val * a\n");
	cmul_qsvector(qsvec_c, qsval, qsvec_a);
	print_qsvector(qsvec_c);

	// GQS
	qs2gqs(&gqsval, &qsval);
	printf("GQS: c := val * a\n");
	cmul_gqsvector_dev(gqsvec_c, gqsval, gqsvec_a, num_blocks, num_threads);
	print_gqsvector_dev(gqsvec_c);

	// ----------
	// (a, b)
	// ----------
	// DD
	ip_qsvector(&qsval, qsvec_a, qsvec_b);
	printf(" DD : (a, b) = ");
	cout.precision(qsfloat::_ndigits);
	cout << qsval << "\n";

	// GQS
	ip_gqsvector_dev(ptr_gqsval_dev, gqsvec_a, gqsvec_b, num_blocks, num_threads);
	//ip_gqsvector_dev(ptr_gqsval_dev, gqsvec_a, gqsvec_b, 1, num_threads);
	gqs2qs_dev(&qsval, ptr_gqsval_dev);
	printf("GQS : (a, b) = ");
	cout.precision(qsfloat::_ndigits);
	cout << qsval << "\n";

	// ----------
	// ||a||_2
	// ----------
	// DD
	norm2_qsvector(&qsval, qsvec_a);
	printf(" DD : ||a||_2 = ");
	cout.precision(qsfloat::_ndigits);
	cout << qsval << "\n";

	// GQS
	norm2_gqsvector_dev(ptr_gqsval_dev, gqsvec_a, num_blocks, num_threads);
	//norm2_gqsvector_dev(ptr_gqsval_dev, gqsvec_a, 1, num_threads);
	gqs2qs_dev(&qsval, ptr_gqsval_dev);
	printf("GQS : ||a||_2 = ");
	cout.precision(qsfloat::_ndigits);
	cout << qsval << "\n";

/* Matrix */
	// DD & GQS
	qsmat_a = init_qsmatrix(ROW_DIM, COL_DIM);
	qsmat_b = init_qsmatrix(ROW_DIM, COL_DIM);
	qsmat_c = init_qsmatrix(ROW_DIM, COL_DIM);

	gqsmat_a = init_gqsmatrix_dev(ROW_DIM, COL_DIM);
	gqsmat_b = init_gqsmatrix_dev(ROW_DIM, COL_DIM);
	gqsmat_c = init_gqsmatrix_dev(ROW_DIM, COL_DIM);

	for(i = 0; i < ROW_DIM; i++)
	{
		for(j = 0; j < COL_DIM; j++)
		{
			// val := sqrt((i + 1));
			qsmat_a->element[i * COL_DIM + j] = (qsfloat)sqrt((qsfloat)(i + j + 1));
			qsmat_b->element[i * COL_DIM + j] = (qsfloat)sqrt((qsfloat)(i + j + 1));
		}
	}
	subst_gqsmatrix_dev_qsmat(gqsmat_a, qsmat_a);
	subst_gqsmatrix_dev_qsmat(gqsmat_b, qsmat_b);

	// Print gqsmatrix
	printf("qsmat_a:\n");
	normf_qsmatrix(&qsval, qsmat_a);
	cout.precision(qsfloat::_ndigits);
	cout << qsval << "\n";

	printf("gqsmat_a:\n");
	//print_gqsmatrix_dev(gqsmat_a);
	normf_gqsmatrix_dev(ptr_gqsval_dev, gqsmat_a, 1, num_threads);
	gqs2qs_dev(&qsval, ptr_gqsval_dev);
	cout.precision(qsfloat::_ndigits);
	cout << qsval << "\n";

	printf("qsmat_b:\n");
	normf_qsmatrix(&qsval, qsmat_b);
	cout.precision(qsfloat::_ndigits);
	cout << qsval << "\n";

	printf("gqsmat_b:\n");
	//print_gqsmatrix_dev(gqsmat_b);
	normf_gqsmatrix_dev(ptr_gqsval_dev, gqsmat_b, num_blocks, num_threads);
	gqs2qs_dev(&qsval, ptr_gqsval_dev);
	cout.precision(qsfloat::_ndigits);
	cout << qsval << "\n";

	// ----------
	// C := A * B
	// ----------
	// DD
	mul_qsmatrix(qsmat_c, qsmat_a, qsmat_b);
	printf("DD : || A * B ||_F:\n");
	normf_qsmatrix(&qsval, qsmat_c);
	cout.precision(qsfloat::_ndigits);
	cout << qsval << "\n";

	// GQS
	mul_gqsmatrix_dev(gqsmat_c, gqsmat_a, gqsmat_b, num_blocks, num_threads);
	printf("GQS: || A * B ||_F:\n");
	normf_gqsmatrix_dev(ptr_gqsval_dev, gqsmat_c, num_blocks, num_threads);
	gqs2qs_dev(&qsval, ptr_gqsval_dev);
	cout.precision(qsfloat::_ndigits);
	cout << qsval << "\n";

/* Free! */
	cudaFree(ptr_gqsval_dev);
	free_gqsvector_dev(gqsvec_a);
	free_gqsvector_dev(gqsvec_b);
	free_gqsvector_dev(gqsvec_c);

	free_gqsmatrix_dev(gqsmat_a);
	free_gqsmatrix_dev(gqsmat_b);
	free_gqsmatrix_dev(gqsmat_c);

	// GQS end!
	GQSEnd();

	// Free QDVectors
	free_qsvector(qsvec_a);
	free_qsvector(qsvec_b);
	free_qsvector(qsvec_c);

	free_qsmatrix(qsmat_a);
	free_qsmatrix(qsmat_b);
	free_qsmatrix(qsmat_c);

	return 0;
}
#endif // DEBUG
