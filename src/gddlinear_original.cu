/********************************************************************************/
/* gddlinear_original.cu: Double-double and Quadruple precision                 */
/*                                 Linear Computation Library with GQD and CUDA */
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

#include "gddlinear.h" // G[D,Q]DVector, G[D,Q]DMatrix
//	#include "common.cu" // G[DQ]DStart, G[DQ]DEnd
//	#include "gqd.cu"

// init_gdd on GPU
__host__ gdd_real * _bncuda_init_gdd(void)
{
	cudaError_t cuda_error;
	gdd_real *ret = NULL;

	cuda_error = cudaMalloc((void **)&ret, sizeof(gdd_real));
	if(cuda_error != cudaSuccess)
	{
		printf("ERROR: _bncuda_init_gdd_array failed!(size = %ld)\n", sizeof(gdd_real));
		cudaFree(ret);
		return NULL;
	}

	return ret;
}

// free_gdd_array on GPU
void _bncuda_free_gdd(gdd_real *dev_val)
{
	cudaFree(dev_val);
}


// init_gdd_array on GPU
__host__ gdd_real * _bncuda_init_gdd_array(int array_num)
{
	cudaError_t cuda_error;
	gdd_real *ret = NULL;

	cuda_error = cudaMalloc((void **)&ret, sizeof(gdd_real) * (int)array_num);
	if(cuda_error != cudaSuccess)
	{
		printf("ERROR: _bncuda_init_gdd_array failed!(size = %ld)\n", sizeof(gdd_real) * array_num);
		cudaFree(ret);
		return NULL;
	}

	return ret;
}

// free_gdd_array on GPU
void _bncuda_free_gdd_array(gdd_real *dev_array)
{
	cudaFree(dev_array);
}

// init_gqd on GPU
__host__ gqd_real * _bncuda_init_gqd(void)
{
	cudaError_t cuda_error;
	gqd_real *ret = NULL;

	cuda_error = cudaMalloc((void **)&ret, sizeof(gqd_real));
	if(cuda_error != cudaSuccess)
	{
		printf("ERROR: _bncuda_init_gqd_array failed!(size = %ld)\n", sizeof(gqd_real));
		cudaFree(ret);
		return NULL;
	}

	return ret;
}

// free_gqd_array on GPU
void _bncuda_free_gqd(gqd_real *dev_val)
{
	cudaFree(dev_val);
}

// init_gdd_array on GPU
__host__ gqd_real * _bncuda_init_gqd_array(int array_num)
{
	cudaError_t cuda_error;
	gqd_real *ret = NULL;

	cuda_error = cudaMalloc((void **)&ret, sizeof(gqd_real) * (int)array_num);
	if(cuda_error != cudaSuccess)
	{
		printf("ERROR: _bncuda_init_gqd_array failed!(size = %ld)\n", sizeof(gqd_real) * array_num);
		cudaFree(ret);
		return NULL;
	}

	return ret;
}

// free_gdd_array on GPU
void _bncuda_free_gqd_array(gqd_real *dev_array)
{
	cudaFree(dev_array);
}


// initialize gddvector on CPU(host)
__host__ GDDVector init_gddvector(long int dim)
{
	long int index;
	GDDVector ret = NULL;
	dd_real zero = 0.0;

	// callocation
	ret = (gddvector *)malloc(sizeof(gddvector));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one GDDVector\n");
		return ret;
	}

	ret->dim = dim;

	ret->element = (gdd_real *)calloc(dim, sizeof(gdd_real));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GDDVector(dim = %ld)\n", dim);
		cudaFree(ret);
		return NULL;
	}

	// set all zeros 
	for(index = 0; index < dim; index++)
		dd2gdd(&ret->element[index], &zero);
		//ret->element[index] = (gdd_real)0.0;

	return ret;
}

// free gddvector on CPU(HOST)
__host__ void free_gddvector(GDDVector vec)
{
	free(vec->element);
	free(vec);
}

// initialize gddvector
__host__ GDDVector init_gddvector_dev(long int dim)
{
	long int index;
	GDDVector ret = NULL;
	dd_real zero = 0.0;

	// allocation
	ret = (gddvector *)malloc(sizeof(gddvector));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one DDVector\n");
		return ret;
	}

	ret->element = NULL;
	ret->dim = dim;

	//ret->element = (dd_real *)calloc(dim, sizeof(dd_real));
	cudaMalloc((void **)&(ret->element), (size_t)(dim * sizeof(gdd_real)));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GDDVector(dim = %ld)\n", dim);
		cudaFree(ret);
		return NULL;
	}

	// set all zeros 
	for(index = 0; index < dim; index++)
		dd2gdd_dev(&(ret->element[index]), &zero);

	return ret;
}

// free ddvector
__host__ void free_gddvector_dev(GDDVector vec)
{
	cudaFree(vec->element);
	free(vec);
}

// copy DDVector to GDDVector on GPU
// gddvec on GPU := ddvec
__host__ void subst_gddvector_dev_ddvec(GDDVector gddvec_dev, DDVector ddvec)
{
	long int i, dim;

	dim = gddvec_dev->dim;

	for(i = 0; i < dim; i++)
	{
		//std::cout << ddvec->element[i] << std::endl;
		dd2gdd_dev(&(gddvec_dev->element[i]), &(ddvec->element[i]));
	}
}

// copy GDDVector on GPU to DDVector
// ddvec := gddvec_dev on GPU
__host__ void subst_ddvector_gddvec_dev(DDVector ddvec, GDDVector gddvec_dev)
{
	long int i, dim;

	dim = ddvec->dim;

	for(i = 0; i < dim; i++)
		gdd2dd_dev(&(ddvec->element[i]), &(gddvec_dev->element[i]));
}

// print ddvector
__host__ void print_gddvector_dev(GDDVector dev_vec)
{
	long int index, dim;
	GDDVector host_vec;

	dim = dev_vec->dim;

	host_vec = init_gddvector(dim);

	cudaMemcpy((void *)(host_vec->element), (void *)(dev_vec->element), size_t(sizeof(gdd_real) * dim), cudaMemcpyDeviceToHost);

	for(index = 0; index < dim; index++)
	{
		printf("%4ld: ", index);

		std::cout << gdd_get_dd(host_vec->element[index]).to_string() << "\n";
	}

	free(host_vec);
}

/*************************************************/
/* Vector Calculations for DDVector               */
/*
void add_gddvector_dev(GDDVector c_dev, GDDVector a_dev, GDDVector b_dev)
void add2_gddvector_dev(GDDVector c_dev, GDDVector a_dev)
void sub_gddvector_dev(GDDVector c_dev, GDDVector a_dev, GDDVector b_dev)
void sub2_gddvector_dev(GDDVector c_dev, GDDVector a_dev)
void cmul_gddvector_dev(GDDVector c_dev, gdd_real val_dev, GDDVector a_dev)
void cmul2_gddvector_dev(GDDVector c_dev, gdd_real val_dev)
void add_cmul_gddvector_dev(GDDVector c_dev, GDDVector a_dev, gdd_real val_dev, GDDVector b_dev)
double ip_gddvector_dev(GDDVector a, GDDVector b_dev)
double norm1_gddvector_dev(GDDVector a_dev)
double norm2_gddvector_dev(GDDVector a_dev)
double normi_gddvector_dev(GDDVector a_dev)
void subst_gddvector_dev(GDDVector c_dev, GDDVector a_dev)
*/
/*************************************************/
/* c = a + b */
__global__ void _bncu_add_gddvector(gdd_real *c_dev_element, gdd_real *a_dev_element, gdd_real *b_dev_element, long int dim)
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

__host__ void add_gddvector_dev(GDDVector c_dev, GDDVector a_dev, GDDVector b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if((a_dev->dim != b_dev->dim) || (c_dev->dim != a_dev->dim) || (c_dev->dim != b_dev->dim))
	{
		fprintf(stderr, "ERROR: add_gddvector_dev\n");
		return;
	}

/*	for(i = 0; i < c->dim; i++)
	{
		c->element[i] = a->element[i] + b->element[i];
	}
*/
	_bncu_add_gddvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, c_dev->dim);
}


/* c = a - b */
__global__ void _bncu_sub_gddvector(gdd_real *c_dev_element, gdd_real *a_dev_element, gdd_real *b_dev_element, long int dim)
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

__host__ void sub_gddvector_dev(GDDVector c_dev, GDDVector a_dev, GDDVector b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if((a_dev->dim != b_dev->dim) || (c_dev->dim != a_dev->dim) || (c_dev->dim != b_dev->dim))
	{
		fprintf(stderr, "ERROR: sub_gddvector_dev\n");
		return;
	}

/*	for(i = 0; i < c->dim; i++)
	{
		c->element[i] = a->element[i] - b->element[i];
	}
*/
	_bncu_sub_gddvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, c_dev->dim);
}

// cmul vector (val, core)
// dev_ret := dev_val * dev_a on GPU
__global__ void _bncu_cmul_gddvector(gdd_real *dev_ret_element, gdd_real val, gdd_real *dev_a_element, long int dim)
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
__host__ void cmul_gddvector_dev(GDDVector c_dev, gdd_real val, GDDVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if(c_dev->dim != a_dev->dim)
	{
		fprintf(stderr, "ERROR: cmul_gddvector_dev\n");
		return;
	}

/*	for(i = 0; i < c_dev->dim; i++)
	{
		c->element[i] = val * a->element[i];
	}
*/

	_bncu_cmul_gddvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, val, a_dev->element, c_dev->dim);

}

// subst vector (val, core)
// dev_ret := dev_vec on GPU
__global__ void _bncu_subst_gddvector(gdd_real *dev_ret_element, gdd_real *dev_vec_element, long int dim)
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
__host__ void subst_gddvector_dev(GDDVector ret_dev, GDDVector vec_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if(ret_dev->dim != vec_dev->dim)
	{
		fprintf(stderr, "ERROR: subst_gddvector_dev\n");
		return;
	}

	_bncu_subst_gddvector<<< num_blocks_per_grid, num_threads_per_block >>>(ret_dev->element, vec_dev->element, ret_dev->dim);

}

// set0 
// dev_ret := 0 on GPU
__global__ void _bncu_set0_gddvector(gdd_real *dev_ret_element, long int dim)
{
	int index;

	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim)
	{
		SET0_GDD(dev_ret_element[index]);
		index += blockDim.x * gridDim.x;
	}

	__syncthreads();

}

/* c := 0 */
__host__ void set0_gddvector_dev(GDDVector ret_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	_bncu_set0_gddvector<<< num_blocks_per_grid, num_threads_per_block >>>(ret_dev->element, ret_dev->dim);
}

// ret[i] := a_vec[i] * b_vec[i]
__global__ void _bncu_mul_gddvector(gdd_real *ret_dev_element, gdd_real *a_dev_element, gdd_real *b_dev_element, long int dim)
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
__global__ void _bncu_sqr_gddvector(gdd_real *ret_dev_element, gdd_real *vec_dev_element, long int dim)
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
__global__ void _bncu_sqrt_gddvector(gdd_real *ret_dev_element, gdd_real *vec_dev_element, long int dim)
{
	int index;

	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim)
	{
		ret_dev_element[index] = (gdd_real)sqrt((gdd_real)vec_dev_element[index]);
		index += blockDim.x * gridDim.x;
	}

	__syncthreads();
}

// ret[i] := abs(vec[i])
__global__ void _bncu_abs_gddvector(gdd_real *ret_dev_element, gdd_real *vec_dev_element, long int dim)
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

// reduction to add
// ret[block_index] := sum^dim_{i=1} vec[i]
__global__ void _bncu_add_reduct_gddvector(gdd_real *ret_dev, gdd_real *vec_dev_element, long int dim)
{
	int index, cache_index;
	__shared__ gdd_real cache[MAX_NUM_THREADS_PER_BLOCK];

	// Inside Block

	index = threadIdx.x + blockIdx.x * blockDim.x;
	cache_index = threadIdx.x;

	SET0_GDD(cache[cache_index]);

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
__global__ void _bncu_max_reduct_gddvector(gdd_real *ret_dev, gdd_real *vec_dev_element, long int dim)
{
	int index, cache_index;
	__shared__ gdd_real cache[MAX_NUM_THREADS_PER_BLOCK];

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
__host__ void ip_gddvector_dev(gdd_real *ret_dev, GDDVector a_dev, GDDVector b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gdd_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim != b_dev->dim)
	{
		fprintf(stderr, "ERROR: ip_gddvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gdd_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gdd_real) * num_blocks_per_grid));

	// c[i] := a[i] * b[i]
	_bncu_mul_gddvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, b_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	//block_dim = 1;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gddvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gdd_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret_dev := block_cache_dev[0]
	cudaMemcpy((void *)ret_dev, (void *)&(block_cache_dev[0]), sizeof(gdd_real), cudaMemcpyDeviceToDevice);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// norm2
__host__ void norm2_gddvector_dev(gdd_real *ret_dev, GDDVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gdd_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim <= 0)
	{
		fprintf(stderr, "ERROR: norm2_gddvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gdd_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gdd_real) * num_blocks_per_grid));

	// c[i] := a[i]^2
	_bncu_sqr_gddvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gddvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gdd_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret := sqrt(ret);
	_bncu_sqrt_gddvector<<<1, 1>>>(ret_dev, block_cache_dev, 1);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// norm1
__host__ void norm1_gddvector_dev(gdd_real *ret_dev, GDDVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gdd_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim <= 0)
	{
		fprintf(stderr, "ERROR: norm1_gddvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gdd_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gdd_real) * num_blocks_per_grid));

	// c[i] := abs(a[i])
	_bncu_abs_gddvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gddvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gdd_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret_dev := block_cache_dev[0]
	cudaMemcpy((void *)ret_dev, (void *)&(block_cache_dev[0]), sizeof(gdd_real), cudaMemcpyDeviceToDevice);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// norm_inf
__host__ void normi_gddvector_dev(gdd_real *ret_dev, GDDVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gdd_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim <= 0)
	{
		fprintf(stderr, "ERROR: normi_gddvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gdd_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gdd_real) * num_blocks_per_grid));

	// c[i] := abs(a[i])
	_bncu_abs_gddvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_max_reduct_gddvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gdd_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret_dev := block_cache_dev[0]
	cudaMemcpy((void *)ret_dev, (void *)&(block_cache_dev[0]), sizeof(gdd_real), cudaMemcpyDeviceToDevice);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// GDD matrix

// set a zero matrix
__host__ void set0_gddmatrix(GDDMatrix mat, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i, j;
	dd_real ddzero = 0.0;
	gdd_real zero;

	dd2gdd(&zero, &ddzero);

	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
			mat->element[i * mat->col_dim + j] = zero;
	}
}

// initialize ddvector
GDDMatrix init_gddmatrix(long int row_dim, long int col_dim)
{
	long int row_index, col_index;
	GDDMatrix ret = NULL;
	dd_real ddzero = 0.0;
	gdd_real zero;

	dd2gdd(&zero, &ddzero);

	// callocation
	ret = (gddmatrix *)malloc(sizeof(gddmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one GDDMatrix\n");
		return ret;
	}

	ret->row_dim = row_dim;
	ret->col_dim = col_dim;

	ret->element = (gdd_real *)calloc(row_dim * col_dim, sizeof(gdd_real));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GDDMatrix(row_dim, col_dim = %ld, %ld)\n", row_dim, col_dim);
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

// free ddvector
void free_gddmatrix(GDDMatrix mat)
{
	free(mat->element);
	free(mat);
}

// initialize ddvector
GDDMatrix init_gddmatrix_dev(long int row_dim, long int col_dim)
{
	long int row_index, col_index;
	GDDMatrix ret = NULL;
	dd_real ddzero = 0.0;
	gdd_real zero;

	dd2gdd(&zero, &ddzero);

	// callocation
	ret = (gddmatrix *)malloc(sizeof(gddmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one GDDMatrix\n");
		return ret;
	}

	ret->row_dim = row_dim;
	ret->col_dim = col_dim;

	ret->element = (gdd_real *)calloc(row_dim * col_dim, sizeof(gdd_real));
	cudaMalloc((void **)&(ret->element), (size_t)(row_dim * col_dim * sizeof(gdd_real)));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GDDMatrix(row_dim, col_dim = %ld, %ld)\n", row_dim, col_dim);
		free(ret);
		return NULL;
	}

	// set all zeros 
	for(row_index = 0; row_index < row_dim; row_index++)
	{
		for(col_index = 0; col_index < col_dim; col_index++)
			gdd2gdd_dev(&(ret->element[row_index * col_dim + col_index]), &zero);
	}

	return ret;
}

// free ddvector
void free_gddmatrix_dev(GDDMatrix mat)
{
	cudaFree(mat->element);
	free(mat);
}

// print ddvector
void print_gddmatrix_dev(GDDMatrix mat)
{
	long int row_index, col_index;
	dd_real ddval;

	for(row_index = 0; row_index < mat->row_dim; row_index++)
	{
		for(col_index = 0; col_index < mat->col_dim; col_index++)
		{
			printf("%ld, %ld: ", row_index, col_index);
			gdd2dd_dev(&ddval, &(mat->element[row_index * mat->col_dim + col_index]));
			std::cout << ddval.to_string() << "\n";
		}
	}
}


__global__ void _bncu_mul_gddmatrix(gdd_real *ret_dev_element, long int row_dim, long int col_dim, long int mid_dim, gdd_real *a_dev_element, gdd_real *b_dev_element)
{
	long int i, j, k;
	gdd_real tmp, tmp1;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	//for(i = 0; i < row_dim; i++)
	while(i < row_dim)
	{
		for(j = 0; j < col_dim; j++)
		{
			SET0_GDD(tmp1); // = (gdd_real)0.0;
			SET0_GDD(tmp); // = (gdd_real)0.0;
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
void mul_gddmatrix_dev(GDDMatrix ret_dev, GDDMatrix a_dev, GDDMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long row_dim, col_dim, mid_dim;

	/* dimension check */
	if((ret_dev->row_dim != a_dev->row_dim) || (ret_dev->col_dim != b_dev->col_dim) || (a_dev->col_dim != b_dev->row_dim))
	{
		fprintf(stderr, "ERROR: mul_gddmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret_dev->row_dim, ret_dev->col_dim, a_dev->row_dim, a_dev->col_dim, b_dev->row_dim, b_dev->col_dim);
		return;
	}

	row_dim = ret_dev->row_dim;
	col_dim = ret_dev->col_dim;
	mid_dim = a_dev->col_dim;

	_bncu_mul_gddmatrix<<<num_blocks_per_grid, num_threads_per_block>>>(ret_dev->element, row_dim, col_dim, mid_dim, a_dev->element, b_dev->element);

}

// copy DDMatrix to GDDMatrix on GPU
// gddvec on GPU := ddvec
__host__ void subst_gddmatrix_dev_ddmat(GDDMatrix gddmat_dev, DDMatrix ddmat)
{
	long int i, j, row_dim, col_dim;

	row_dim = gddmat_dev->row_dim;
	col_dim = gddmat_dev->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
			dd2gdd_dev(&(gddmat_dev->element[i * col_dim + j]), &(ddmat->element[i * col_dim + j]));
	}
}

// copy GDDMatrix on GPU to DDMatrix
// ddvec := gddmat_dev on GPU
__host__ void subst_ddmatrix_gddmat_dev(DDMatrix ddmat, GDDMatrix gddmat_dev)
{
	long int i, j, row_dim, col_dim;

	row_dim = gddmat_dev->row_dim;
	col_dim = gddmat_dev->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
			gdd2dd_dev(&(ddmat->element[i * col_dim + j]), &(gddmat_dev->element[i * col_dim + j]));
	}
}

// Frobenius norm
__host__ void normf_gddmatrix_dev(gdd_real *ret_dev, GDDMatrix mat_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, tmp_vec_dev_dim, total_dim;
//	gdd_real tmp, tmp1;
	gdd_real *block_cache_dev; // partial sum per block
	gdd_real *tmp_vec_dev; // square mul of mat

	total_dim = mat_dev->row_dim * mat_dev->col_dim; // total dimension as one vector

	// initialize block_cache_dev, tmp_vec_dev
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gdd_real) * num_blocks_per_grid));
	cudaMalloc((void **)&tmp_vec_dev, (size_t)(sizeof(gdd_real) * total_dim));

	// mat[i][j]^2
	//_bncu_mul_gddvector<<<num_blocks_per_grid, num_threads_per_block>>>(tmp_vec_dev, mat_dev->element, mat_dev->element, total_dim);
	_bncu_sqr_gddvector<<<num_blocks_per_grid, num_threads_per_block>>>(tmp_vec_dev, mat_dev->element, total_dim);

	// reduction to add
	block_dim = num_blocks_per_grid;
//	block_dim = 1;
	thread_dim = num_threads_per_block;
	tmp_vec_dev_dim = total_dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gddvector<<<block_dim, thread_dim>>>(block_cache_dev, tmp_vec_dev, tmp_vec_dev_dim);

		if(block_dim <= 1)
			break;

		tmp_vec_dev_dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)tmp_vec_dev, (void *)block_cache_dev, (size_t)(sizeof(gdd_real) * tmp_vec_dev_dim), cudaMemcpyDeviceToDevice);
		//printf("block_dim = %d\n", block_dim);

	}

	// ret := sqrt(ret);
	_bncu_sqrt_gddvector<<<1, 1>>>(ret_dev, block_cache_dev, 1);

	// free
	cudaFree(block_cache_dev);
	cudaFree(tmp_vec_dev);

/*
	SET0_GDD(*ret); // = (gdd_real)0.0;
	SET0_GDD(tmp); // = (gdd_real)(int)0;
	SET0_GDD(tmp1); // = (gdd_real)(int)0;

	dim = mat->row_dim * mat->col_dim;

	for(i = 0; i < dim ; i++)
	{
		//tmp += mat->element[i] * mat->element[i];
		//tmp += gdd_real::mul((gdd_real)(mat->element[i]), (gdd_real)(mat->element[i]));
		tmp1 = (gdd_real)(mat->element[i]) * (gdd_real)(mat->element[i]);
		tmp = tmp + tmp1;
	}

	//*ret = sqrt(tmp);
	tmp = (gdd_real)sqrt(tmp);
	*ret = tmp;
*/
}

/*************************************************/
/* Matrix Caluculations for GDDMatrix            */
/*
void normf_gddmatrix(double ret[DDSIZE], GDDMatrix mat)
void norm1_gddmatrix(double ret[DDSIZE], GDDMatrix mat)
void normi_gddmatrix(double ret[DDSIZE], GDDMatrix mat)
void add_gddmatrix(GDDMatrix c, GDDMatrix a, GDDMatrix b);
void sub_gddmatrix(GDDMatrix c, GDDMatrix a, GDDMatrix b);
void mul_gddmatrix(GDDMatrix c, GDDMatrix a, GDDMatrix b);
void mul_gddmatrix_ddvec(DDVector v, GDDMatrix a, DDVector vb)
void mul_gddmatrixt_ddvec(DDVector v, GDDMatrix a, DDVector vb)
void transpose_gddmatrix(GDDMatrix c, GDDMatrix a);
void inv_gddmatrix(GDDMatrix a);
void subst_mpfmatrux(GDDMatrix c, GDDMatrix a);
*/
/*************************************************/

#if 0
/* Infinity Norm of Matrix */
void normi_gddmatrix(gdd_real *ret, GDDMatrix mat, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i, j;
	gdd_real sum;

	SET0_GDD(*ret); // = 0.0;

	for(i = 0; i < mat->row_dim; i++)
	{
		SET0_GDD(sum); //  = 0.0;
		for(j = 0; j < mat->col_dim; j++)
			sum = sum + abs(mat->element[i * mat->row_dim + j]);

		if(*ret < sum)
			*ret = sum;

	}

	return;
}

/* 1 Norm of Matrix */
void norm1_gddmatrix(gdd_real *ret, GDDMatrix mat, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i, j;
	gdd_real sum;

	SET0_GDD(*ret); // = 0.0;

	for(j = 0; j < mat->col_dim; j++)
	{
		SET0_GDD(sum); // = 0.0;
		for(i = 0; i < mat->row_dim; i++)
			sum = sum + abs(mat->element[i * mat->col_dim + j]);

		if(*ret < sum)
			*ret = sum;

	}

	return;
}
#endif // 0

/* c := a + b */
void add_gddmatrix_dev(GDDMatrix c_dev, GDDMatrix a_dev, GDDMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int row_dim, col_dim; //, index;

	/* check row_dim */
	if((a_dev->row_dim != b_dev->row_dim) || (b_dev->row_dim != c_dev->row_dim) || (c_dev->row_dim != a_dev->row_dim))
	{
		fprintf(stderr, "ERROR: add_gddmatrix_dev\n");
		return;
	}
	row_dim = c_dev->row_dim;

	/* check col_dim */
	if((a_dev->col_dim != b_dev->col_dim) || (b_dev->col_dim != c_dev->col_dim) || (c_dev->col_dim != a_dev->col_dim))
	{
		fprintf(stderr, "ERROR: add_gddmatrix_dev\n");
		return;
	}
	col_dim = c_dev->col_dim;

	_bncu_add_gddvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, row_dim * col_dim);

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
void sub_gddmatrix_dev(GDDMatrix c_dev, GDDMatrix a_dev, GDDMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int row_dim, col_dim; //, index;

	/* check row_dim */
	if((a_dev->row_dim != b_dev->row_dim) || (b_dev->row_dim != c_dev->row_dim) || (c_dev->row_dim != a_dev->row_dim))
	{
		fprintf(stderr, "ERROR: sub_gddmatrix_dev\n");
		return;
	}
	row_dim = c_dev->row_dim;

	/* check col_dim */
	if((a_dev->col_dim != b_dev->col_dim) || (b_dev->col_dim != c_dev->col_dim) || (c_dev->col_dim != a_dev->col_dim))
	{
		fprintf(stderr, "ERROR: sub_gddmatrix_dev\n");
		return;
	}
	col_dim = c_dev->col_dim;

	_bncu_sub_gddvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, row_dim * col_dim);

}

/* c := sc * a */
void cmul_gddmatrix_dev(GDDMatrix c_dev, gdd_real sc, GDDMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int row_dim, col_dim; //, index;

	/* check row_dim */
	if(a_dev->row_dim != c_dev->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_gddmatrix_dev\n");
		return;
	}
	row_dim = c_dev->row_dim;

	/* check col_dim */
	if(a_dev->col_dim != c_dev->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_gddmatrix_dev\n");
		return;
	}
	col_dim = c_dev->col_dim;

	_bncu_cmul_gddvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, sc, a_dev->element, row_dim * col_dim);


}

/* c = a^T */
__global__ void _bncu_transpose_gddmatrix(gdd_real *c_dev_element, gdd_real *a_dev_element, long int row_dim, long int col_dim)
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

void transpose_gddmatrix_dev(GDDMatrix c_dev, GDDMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	//long int i, j;

	/* Check Dimensions */
	if((c_dev->row_dim != a_dev->col_dim) || (c_dev->col_dim != a_dev->row_dim))
	{
		fprintf(stderr, "ERROR: transpose_gddmatrix_dev\n");
		return;
	}
	
	_bncu_transpose_gddmatrix<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, c_dev->row_dim, c_dev->col_dim);

/*	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			c->element[i * c->col_dim + j] = a->element[j * a->col_dim + i];
	}
*/
}

/* c := a */
void subst_gddmatrix_dev(GDDMatrix c_dev, GDDMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	//long int i, j, index;

	if((c_dev->row_dim != a_dev->row_dim) || (c_dev->col_dim != a_dev->col_dim))
	{
		fprintf(stderr, "ERROR: subst_gddmatrix_dev\n");
		return;
	}

	_bncu_subst_gddvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, c_dev->row_dim * c_dev->col_dim);

}

/* c := 0 */
void set0_gddmatrix_dev(GDDMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block)
{
//	long int i, j;

	_bncu_set0_gddvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, c_dev->row_dim * c_dev->col_dim);

}

/* set (i, j)-element on GPU  */
void set_gddmatrix_ij_dev(GDDMatrix mat_dev, long int row_index, long int col_index, gdd_real val)
{
	if((row_index >= 0) && (row_index < mat_dev->row_dim) && (col_index >= 0) && (col_index < mat_dev->col_dim))
		gdd_dev2gdd(&(mat_dev->element[row_index * mat_dev->col_dim + col_index]), &val);
}

/* set (i, j)-element on GPU from dd_real */
void set_gddmatrix_ij_dd_dev(GDDMatrix mat_dev, long int row_index, long int col_index, dd_real val)
{
	if((row_index >= 0) && (row_index < mat_dev->row_dim) && (col_index >= 0) && (col_index < mat_dev->col_dim))
		dd2gdd_dev(&(mat_dev->element[row_index * mat_dev->col_dim + col_index]), &val);
}

/* c := I */
void setI_gddmatrix_dev(GDDMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i;
	dd_real one;

	one = (dd_real)1.0;

	_bncu_set0_gddvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, c_dev->row_dim * c_dev->col_dim);

	for(i = 0; i < c_dev->row_dim; i++)
		set_gddmatrix_ij_dd_dev(c_dev, i, i, one);
/*
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			SET0_GDD(c->element[i * c->col_dim + j]); // = 0.0;
		if(i < c->col_dim)
			SET1_GDD(c->element[i * c->col_dim + i]); // = 1.0;
	}
*/
}

/* v := a * vb */
__global__ void _bncu_mul_gddmatrix_gddvec(gdd_real *v_element, gdd_real *a_element, gdd_real *vb_element, long int row_dim, long int col_dim)
{
	int j, index;
	gdd_real tmp;

	index = threadIdx.x + blockIdx.x * blockDim.x;

	//for(i = 0; i < row_dim; i++)
	while(index < row_dim)
	{
		SET0_GDD(tmp); // = 0.0;
		for(j = 0; j < col_dim; j++)
			tmp = tmp + a_element[index * col_dim + j] * vb_element[j];

		v_element[index] = tmp;

		__syncthreads();

		index += blockDim.x * gridDim.x;
	}
}

void mul_gddmatrix_gddvec(GDDVector v, GDDMatrix a, GDDVector vb, int num_blocks_per_grid, int num_threads_per_block)
{
	/* Check Dimension */
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: mul_gddmatrix_ddvec\n");
		return;
	}

	_bncu_mul_gddmatrix_gddvec<<<num_blocks_per_grid, num_threads_per_block>>>(v->element, a->element, vb->element, a->row_dim, a->col_dim);

/*	for(i = 0; i < a->row_dim; i++)
	{
		SET0_GDD(tmp); // = 0.0;
		for(j = 0; j < a->col_dim; j++)
			tmp = tmp + a->element[i * a->col_dim + j] * vb->element[j];

		v->element[i] = tmp;
	}
*/
}

/* v := a^T * vb */
__global__ void _bncu_mul_gddmatrixt_gddvec(gdd_real *v_element, gdd_real *a_element, gdd_real *vb_element, long int row_dim, long int col_dim)
{
	int j, index;
	gdd_real tmp;

	index = threadIdx.x + blockIdx.x * blockDim.x;

	//for(i = 0; i < col_dim; i++)
	while(index < col_dim)
	{
		SET0_GDD(tmp); // = 0.0;
		for(j = 0; j < row_dim; j++)
			tmp = tmp + a_element[j * row_dim + index] * vb_element[j];

		v_element[index] = tmp;

		__syncthreads();

		index += blockDim.x * gridDim.x;
	}
}

__host__ void mul_gddmatrixt_gddvec(GDDVector v, GDDMatrix a, GDDVector vb, int num_blocks_per_grid, int num_threads_per_block)
{

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_gddmatrixt_gddvec\n");
		return;
	}

	_bncu_mul_gddmatrixt_gddvec<<<num_blocks_per_grid, num_threads_per_block>>>(v->element, a->element, vb->element, a->row_dim, a->col_dim);

/*	for(i = 0; i < a->col_dim; i++)
	{
		SET0_GDD(tmp); // = 0.0;
		for(j = 0; j < a->row_dim; j++)
			tmp = tmp + a->element[j * a->row_dim + i] * vb->element[j];

		v->element[i] = tmp;
	}
*/
}
//---------------------------------------%
// GQD
//---------------------------------------%

// initialize gqdvector on CPU(host)
__host__ GQDVector init_gqdvector(long int dim)
{
	long int index;
	GQDVector ret = NULL;
	qd_real zero = 0.0;

	// callocation
	ret = (gqdvector *)malloc(sizeof(gqdvector));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one GQDVector\n");
		return ret;
	}

	ret->dim = dim;

	ret->element = (gqd_real *)calloc(dim, sizeof(gqd_real));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GQDVector(dim = %ld)\n", dim);
		cudaFree(ret);
		return NULL;
	}

	// set all zeros 
	for(index = 0; index < dim; index++)
		qd2gqd(&ret->element[index], &zero);
		//ret->element[index] = (gqd_real)0.0;

	return ret;
}

// free gqdvector on CPU(HOST)
__host__ void free_gqdvector(GQDVector vec)
{
	free(vec->element);
	free(vec);
}

// initialize gqdvector
__host__ GQDVector init_gqdvector_dev(long int dim)
{
	long int index;
	GQDVector ret = NULL;
	qd_real zero = 0.0;

	// allocation
	ret = (gqdvector *)malloc(sizeof(gqdvector));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one QDVector\n");
		return ret;
	}

	ret->element = NULL;
	ret->dim = dim;

	//ret->element = (qd_real *)calloc(dim, sizeof(qd_real));
	cudaMalloc((void **)&(ret->element), (size_t)(dim * sizeof(gqd_real)));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GQDVector(dim = %ld)\n", dim);
		cudaFree(ret);
		return NULL;
	}

	// set all zeros 
	for(index = 0; index < dim; index++)
		qd2gqd_dev(&(ret->element[index]), &zero);

	return ret;
}

// free qdvector
__host__ void free_gqdvector_dev(GQDVector vec)
{
	cudaFree(vec->element);
	free(vec);
}

// copy QDVector to GQDVector on GPU
// gqdvec on GPU := qdvec
__host__ void subst_gqdvector_dev_qdvec(GQDVector gqdvec_dev, QDVector qdvec)
{
	long int i, dim;

	dim = gqdvec_dev->dim;

	for(i = 0; i < dim; i++)
	{
		//std::cout << qdvec->element[i] << std::endl;
		qd2gqd_dev(&(gqdvec_dev->element[i]), &(qdvec->element[i]));
	}
}

// copy GQDVector on GPU to QDVector
// qdvec := gqdvec_dev on GPU
__host__ void subst_qdvector_gqdvec_dev(QDVector qdvec, GQDVector gqdvec_dev)
{
	long int i, dim;

	dim = qdvec->dim;

	for(i = 0; i < dim; i++)
		gqd2qd_dev(&(qdvec->element[i]), &(gqdvec_dev->element[i]));
}

// print qdvector
__host__ void print_gqdvector_dev(GQDVector dev_vec)
{
	long int index, dim;
	GQDVector host_vec;

	dim = dev_vec->dim;

	host_vec = init_gqdvector(dim);

	cudaMemcpy((void *)(host_vec->element), (void *)(dev_vec->element), size_t(sizeof(gqd_real) * dim), cudaMemcpyDeviceToHost);

	for(index = 0; index < dim; index++)
	{
		printf("%4ld: ", index);

		std::cout << gqd_get_qd(host_vec->element[index]).to_string() << "\n";
	}

	free(host_vec);
}

/*************************************************/
/* Vector Calculations for QDVector               */
/*
void add_gqdvector_dev(GQDVector c_dev, GQDVector a_dev, GQDVector b_dev)
void aqd2_gqdvector_dev(GQDVector c_dev, GQDVector a_dev)
void sub_gqdvector_dev(GQDVector c_dev, GQDVector a_dev, GQDVector b_dev)
void sub2_gqdvector_dev(GQDVector c_dev, GQDVector a_dev)
void cmul_gqdvector_dev(GQDVector c_dev, gqd_real val_dev, GQDVector a_dev)
void cmul2_gqdvector_dev(GQDVector c_dev, gqd_real val_dev)
void add_cmul_gqdvector_dev(GQDVector c_dev, GQDVector a_dev, gqd_real val_dev, GQDVector b_dev)
double ip_gqdvector_dev(GQDVector a, GQDVector b_dev)
double norm1_gqdvector_dev(GQDVector a_dev)
double norm2_gqdvector_dev(GQDVector a_dev)
double normi_gqdvector_dev(GQDVector a_dev)
void subst_gqdvector_dev(GQDVector c_dev, GQDVector a_dev)
*/
/*************************************************/
/* c = a + b */
__global__ void _bncu_add_gqdvector(gqd_real *c_dev_element, gqd_real *a_dev_element, gqd_real *b_dev_element, long int dim)
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

__host__ void add_gqdvector_dev(GQDVector c_dev, GQDVector a_dev, GQDVector b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if((a_dev->dim != b_dev->dim) || (c_dev->dim != a_dev->dim) || (c_dev->dim != b_dev->dim))
	{
		fprintf(stderr, "ERROR: add_gqdvector_dev\n");
		return;
	}

/*	for(i = 0; i < c->dim; i++)
	{
		c->element[i] = a->element[i] + b->element[i];
	}
*/
	_bncu_add_gqdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, c_dev->dim);
}


/* c = a - b */
__global__ void _bncu_sub_gqdvector(gqd_real *c_dev_element, gqd_real *a_dev_element, gqd_real *b_dev_element, long int dim)
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

__host__ void sub_gqdvector_dev(GQDVector c_dev, GQDVector a_dev, GQDVector b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if((a_dev->dim != b_dev->dim) || (c_dev->dim != a_dev->dim) || (c_dev->dim != b_dev->dim))
	{
		fprintf(stderr, "ERROR: sub_gqdvector_dev\n");
		return;
	}

/*	for(i = 0; i < c->dim; i++)
	{
		c->element[i] = a->element[i] - b->element[i];
	}
*/
	_bncu_sub_gqdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, c_dev->dim);
}

// cmul vector (val, core)
// dev_ret := dev_val * dev_a on GPU
__global__ void _bncu_cmul_gqdvector(gqd_real *dev_ret_element, gqd_real val, gqd_real *dev_a_element, long int dim)
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
__host__ void cmul_gqdvector_dev(GQDVector c_dev, gqd_real val, GQDVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if(c_dev->dim != a_dev->dim)
	{
		fprintf(stderr, "ERROR: cmul_gqdvector_dev\n");
		return;
	}

/*	for(i = 0; i < c_dev->dim; i++)
	{
		c->element[i] = val * a->element[i];
	}
*/

	_bncu_cmul_gqdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, val, a_dev->element, c_dev->dim);

}

// subst vector (val, core)
// dev_ret := dev_vec on GPU
__global__ void _bncu_subst_gqdvector(gqd_real *dev_ret_element, gqd_real *dev_vec_element, long int dim)
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
__host__ void subst_gqdvector_dev(GQDVector ret_dev, GQDVector vec_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if(ret_dev->dim != vec_dev->dim)
	{
		fprintf(stderr, "ERROR: subst_gqdvector_dev\n");
		return;
	}

	_bncu_subst_gqdvector<<< num_blocks_per_grid, num_threads_per_block >>>(ret_dev->element, vec_dev->element, ret_dev->dim);

}

// set0 
// dev_ret := 0 on GPU
__global__ void _bncu_set0_gqdvector(gqd_real *dev_ret_element, long int dim)
{
	int index;

	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim)
	{
		SET0_GQD(dev_ret_element[index]);
		index += blockDim.x * gridDim.x;
	}

	__syncthreads();

}

/* c := 0 */
__host__ void set0_gqdvector_dev(GQDVector ret_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	_bncu_set0_gqdvector<<< num_blocks_per_grid, num_threads_per_block >>>(ret_dev->element, ret_dev->dim);
}

// ret[i] := a_vec[i] * b_vec[i]
__global__ void _bncu_mul_gqdvector(gqd_real *ret_dev_element, gqd_real *a_dev_element, gqd_real *b_dev_element, long int dim)
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
__global__ void _bncu_sqr_gqdvector(gqd_real *ret_dev_element, gqd_real *vec_dev_element, long int dim)
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
__global__ void _bncu_sqrt_gqdvector(gqd_real *ret_dev_element, gqd_real *vec_dev_element, long int dim)
{
	int index;

	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim)
	{
		ret_dev_element[index] = (gqd_real)sqrt((gqd_real)vec_dev_element[index]);
		index += blockDim.x * gridDim.x;
	}

	__syncthreads();
}

// ret[i] := abs(vec[i])
__global__ void _bncu_abs_gqdvector(gqd_real *ret_dev_element, gqd_real *vec_dev_element, long int dim)
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
__global__ void _bncu_add_reduct_gqdvector(gqd_real *ret_dev, gqd_real *vec_dev_element, long int dim)
{
	int index, cache_index;
	__shared__ gqd_real cache[MAX_NUM_THREADS_PER_BLOCK];

	// Inside Block

	index = threadIdx.x + blockIdx.x * blockDim.x;
	cache_index = threadIdx.x;

	SET0_GQD(cache[cache_index]);

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
__global__ void _bncu_max_reduct_gqdvector(gqd_real *ret_dev, gqd_real *vec_dev_element, long int dim)
{
	int index, cache_index;
	__shared__ gqd_real cache[MAX_NUM_THREADS_PER_BLOCK];

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
__host__ void ip_gqdvector_dev(gqd_real *ret_dev, GQDVector a_dev, GQDVector b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gqd_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim != b_dev->dim)
	{
		fprintf(stderr, "ERROR: ip_gqdvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gqd_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gqd_real) * num_blocks_per_grid));

	// c[i] := a[i] * b[i]
	_bncu_mul_gqdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, b_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	//block_dim = 1;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gqdvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gqd_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret_dev := block_cache_dev[0]
	cudaMemcpy((void *)ret_dev, (void *)&(block_cache_dev[0]), sizeof(gqd_real), cudaMemcpyDeviceToDevice);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// norm2
__host__ void norm2_gqdvector_dev(gqd_real *ret_dev, GQDVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gqd_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim <= 0)
	{
		fprintf(stderr, "ERROR: norm2_gqdvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gqd_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gqd_real) * num_blocks_per_grid));

	// c[i] := a[i]^2
	_bncu_sqr_gqdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gqdvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gqd_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret := sqrt(ret);
	_bncu_sqrt_gqdvector<<<1, 1>>>(ret_dev, block_cache_dev, 1);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// norm1
__host__ void norm1_gqdvector_dev(gqd_real *ret_dev, GQDVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gqd_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim <= 0)
	{
		fprintf(stderr, "ERROR: norm1_gqdvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gqd_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gqd_real) * num_blocks_per_grid));

	// c[i] := abs(a[i])
	_bncu_abs_gqdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gqdvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gqd_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret_dev := block_cache_dev[0]
	cudaMemcpy((void *)ret_dev, (void *)&(block_cache_dev[0]), sizeof(gqd_real), cudaMemcpyDeviceToDevice);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// norm_i
__host__ void normi_gqdvector_dev(gqd_real *ret_dev, GQDVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gqd_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim <= 0)
	{
		fprintf(stderr, "ERROR: normi_gqdvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gqd_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gqd_real) * num_blocks_per_grid));

	// c[i] := abs(a[i])
	_bncu_abs_gqdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_max_reduct_gqdvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gqd_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret_dev := block_cache_dev[0]
	cudaMemcpy((void *)ret_dev, (void *)&(block_cache_dev[0]), sizeof(gqd_real), cudaMemcpyDeviceToDevice);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// GQD matrix

// set a zero matrix
__host__ void set0_gqdmatrix(GQDMatrix mat, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i, j;
	qd_real qdzero = 0.0;
	gqd_real zero;

	qd2gqd(&zero, &qdzero);

	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
			mat->element[i * mat->col_dim + j] = zero;
	}
}

// initialize qdvector
GQDMatrix init_gqdmatrix(long int row_dim, long int col_dim)
{
	long int row_index, col_index;
	GQDMatrix ret = NULL;
	qd_real qdzero = 0.0;
	gqd_real zero;

	qd2gqd(&zero, &qdzero);

	// callocation
	ret = (gqdmatrix *)malloc(sizeof(gqdmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one GQDMatrix\n");
		return ret;
	}

	ret->row_dim = row_dim;
	ret->col_dim = col_dim;

	ret->element = (gqd_real *)calloc(row_dim * col_dim, sizeof(gqd_real));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GQDMatrix(row_dim, col_dim = %ld, %ld)\n", row_dim, col_dim);
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

// free qdvector
void free_gqdmatrix(GQDMatrix mat)
{
	free(mat->element);
	free(mat);
}

// initialize qdvector
GQDMatrix init_gqdmatrix_dev(long int row_dim, long int col_dim)
{
	long int row_index, col_index;
	GQDMatrix ret = NULL;
	qd_real qdzero = 0.0;
	gqd_real zero;

	qd2gqd(&zero, &qdzero);

	// callocation
	ret = (gqdmatrix *)malloc(sizeof(gqdmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one GQDMatrix\n");
		return ret;
	}

	ret->row_dim = row_dim;
	ret->col_dim = col_dim;

	ret->element = (gqd_real *)calloc(row_dim * col_dim, sizeof(gqd_real));
	cudaMalloc((void **)&(ret->element), (size_t)(row_dim * col_dim * sizeof(gqd_real)));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GQDMatrix(row_dim, col_dim = %ld, %ld)\n", row_dim, col_dim);
		free(ret);
		return NULL;
	}

	// set all zeros 
	for(row_index = 0; row_index < row_dim; row_index++)
	{
		for(col_index = 0; col_index < col_dim; col_index++)
			gqd2gqd_dev(&(ret->element[row_index * col_dim + col_index]), &zero);
	}

	return ret;
}

// free qdvector
void free_gqdmatrix_dev(GQDMatrix mat)
{
	cudaFree(mat->element);
	free(mat);
}

// print qdvector
void print_gqdmatrix_dev(GQDMatrix mat)
{
	long int row_index, col_index;
	qd_real qdval;

	for(row_index = 0; row_index < mat->row_dim; row_index++)
	{
		for(col_index = 0; col_index < mat->col_dim; col_index++)
		{
			printf("%ld, %ld: ", row_index, col_index);
			gqd2qd_dev(&qdval, &(mat->element[row_index * mat->col_dim + col_index]));
			std::cout << qdval.to_string() << "\n";
		}
	}
}


__global__ void _bncu_mul_gqdmatrix(gqd_real *ret_dev_element, long int row_dim, long int col_dim, long int mid_dim, gqd_real *a_dev_element, gqd_real *b_dev_element)
{
	long int i, j, k;
	gqd_real tmp, tmp1;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	//for(i = 0; i < row_dim; i++)
	while(i < row_dim)
	{
		for(j = 0; j < col_dim; j++)
		{
			SET0_GQD(tmp1); // = (gqd_real)0.0;
			SET0_GQD(tmp); // = (gqd_real)0.0;
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
void mul_gqdmatrix_dev(GQDMatrix ret_dev, GQDMatrix a_dev, GQDMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long row_dim, col_dim, mid_dim;

	/* dimension check */
	if((ret_dev->row_dim != a_dev->row_dim) || (ret_dev->col_dim != b_dev->col_dim) || (a_dev->col_dim != b_dev->row_dim))
	{
		fprintf(stderr, "ERROR: mul_gqdmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret_dev->row_dim, ret_dev->col_dim, a_dev->row_dim, a_dev->col_dim, b_dev->row_dim, b_dev->col_dim);
		return;
	}

	row_dim = ret_dev->row_dim;
	col_dim = ret_dev->col_dim;
	mid_dim = a_dev->col_dim;

	_bncu_mul_gqdmatrix<<<num_blocks_per_grid, num_threads_per_block>>>(ret_dev->element, row_dim, col_dim, mid_dim, a_dev->element, b_dev->element);

}

// copy QDMatrix to GQDMatrix on GPU
// gqdvec on GPU := qdvec
__host__ void subst_gqdmatrix_dev_qdmat(GQDMatrix gqdmat_dev, QDMatrix qdmat)
{
	long int i, j, row_dim, col_dim;

	row_dim = gqdmat_dev->row_dim;
	col_dim = gqdmat_dev->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
			qd2gqd_dev(&(gqdmat_dev->element[i * col_dim + j]), &(qdmat->element[i * col_dim + j]));
	}
}

// copy GQDMatrix on GPU to QDMatrix
// qdvec := gqdmat_dev on GPU
__host__ void subst_qdmatrix_gqdmat_dev(QDMatrix qdmat, GQDMatrix gqdmat_dev)
{
	long int i, j, row_dim, col_dim;

	row_dim = gqdmat_dev->row_dim;
	col_dim = gqdmat_dev->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
			gqd2qd_dev(&(qdmat->element[i * col_dim + j]), &(gqdmat_dev->element[i * col_dim + j]));
	}
}

// Frobenius norm
__host__ void normf_gqdmatrix_dev(gqd_real *ret_dev, GQDMatrix mat_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, tmp_vec_dev_dim, total_dim;
//	gqd_real tmp, tmp1;
	gqd_real *block_cache_dev; // partial sum per block
	gqd_real *tmp_vec_dev; // square mul of mat

	total_dim = mat_dev->row_dim * mat_dev->col_dim; // total dimension as one vector

	// initialize block_cache_dev, tmp_vec_dev
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gqd_real) * num_blocks_per_grid));
	cudaMalloc((void **)&tmp_vec_dev, (size_t)(sizeof(gqd_real) * total_dim));

	// mat[i][j]^2
	//_bncu_mul_gqdvector<<<num_blocks_per_grid, num_threads_per_block>>>(tmp_vec_dev, mat_dev->element, mat_dev->element, total_dim);
	_bncu_sqr_gqdvector<<<num_blocks_per_grid, num_threads_per_block>>>(tmp_vec_dev, mat_dev->element, total_dim);

	// reduction to aqd
	block_dim = num_blocks_per_grid;
//	block_dim = 1;
	thread_dim = num_threads_per_block;
	tmp_vec_dev_dim = total_dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gqdvector<<<block_dim, thread_dim>>>(block_cache_dev, tmp_vec_dev, tmp_vec_dev_dim);

		if(block_dim <= 1)
			break;

		tmp_vec_dev_dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)tmp_vec_dev, (void *)block_cache_dev, (size_t)(sizeof(gqd_real) * tmp_vec_dev_dim), cudaMemcpyDeviceToDevice);

	}

	// ret := sqrt(ret);
	_bncu_sqrt_gqdvector<<<1, 1>>>(ret_dev, block_cache_dev, 1);

	// free
	cudaFree(block_cache_dev);
	cudaFree(tmp_vec_dev);

/*
	SET0_GQD(*ret); // = (gqd_real)0.0;
	SET0_GQD(tmp); // = (gqd_real)(int)0;
	SET0_GQD(tmp1); // = (gqd_real)(int)0;

	dim = mat->row_dim * mat->col_dim;

	for(i = 0; i < dim ; i++)
	{
		//tmp += mat->element[i] * mat->element[i];
		//tmp += gqd_real::mul((gqd_real)(mat->element[i]), (gqd_real)(mat->element[i]));
		tmp1 = (gqd_real)(mat->element[i]) * (gqd_real)(mat->element[i]);
		tmp = tmp + tmp1;
	}

	//*ret = sqrt(tmp);
	tmp = (gqd_real)sqrt(tmp);
	*ret = tmp;
*/
}

/*************************************************/
/* Matrix Caluculations for GQDMatrix            */
/*
void normf_gqdmatrix(double ret[DDSIZE], GQDMatrix mat)
void norm1_gqdmatrix(double ret[DDSIZE], GQDMatrix mat)
void normi_gqdmatrix(double ret[DDSIZE], GQDMatrix mat)
void add_gqdmatrix(GQDMatrix c, GQDMatrix a, GQDMatrix b);
void sub_gqdmatrix(GQDMatrix c, GQDMatrix a, GQDMatrix b);
void mul_gqdmatrix(GQDMatrix c, GQDMatrix a, GQDMatrix b);
void mul_gqdmatrix_qdvec(QDVector v, GQDMatrix a, QDVector vb)
void mul_gqdmatrixt_qdvec(QDVector v, GQDMatrix a, QDVector vb)
void transpose_gqdmatrix(GQDMatrix c, GQDMatrix a);
void inv_gqdmatrix(GQDMatrix a);
void subst_mpfmatrux(GQDMatrix c, GQDMatrix a);
*/
/*************************************************/

#if 0
/* Infinity Norm of Matrix */
void normi_gqdmatrix(gqd_real *ret, GQDMatrix mat, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i, j;
	gqd_real sum;

	SET0_GQD(*ret); // = 0.0;

	for(i = 0; i < mat->row_dim; i++)
	{
		SET0_GQD(sum); //  = 0.0;
		for(j = 0; j < mat->col_dim; j++)
			sum = sum + abs(mat->element[i * mat->row_dim + j]);

		if(*ret < sum)
			*ret = sum;

	}

	return;
}

/* 1 Norm of Matrix */
void norm1_gqdmatrix(gqd_real *ret, GQDMatrix mat, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i, j;
	gqd_real sum;

	SET0_GQD(*ret); // = 0.0;

	for(j = 0; j < mat->col_dim; j++)
	{
		SET0_GQD(sum); // = 0.0;
		for(i = 0; i < mat->row_dim; i++)
			sum = sum + abs(mat->element[i * mat->col_dim + j]);

		if(*ret < sum)
			*ret = sum;

	}

	return;
}
#endif // 0

/* c := a + b */
void add_gqdmatrix_dev(GQDMatrix c_dev, GQDMatrix a_dev, GQDMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int row_dim, col_dim; //, index;

	/* check row_dim */
	if((a_dev->row_dim != b_dev->row_dim) || (b_dev->row_dim != c_dev->row_dim) || (c_dev->row_dim != a_dev->row_dim))
	{
		fprintf(stderr, "ERROR: add_gqdmatrix_dev\n");
		return;
	}
	row_dim = c_dev->row_dim;

	/* check col_dim */
	if((a_dev->col_dim != b_dev->col_dim) || (b_dev->col_dim != c_dev->col_dim) || (c_dev->col_dim != a_dev->col_dim))
	{
		fprintf(stderr, "ERROR: add_gqdmatrix_dev\n");
		return;
	}
	col_dim = c_dev->col_dim;

	_bncu_add_gqdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, row_dim * col_dim);

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
void sub_gqdmatrix_dev(GQDMatrix c_dev, GQDMatrix a_dev, GQDMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int row_dim, col_dim; //, index;

	/* check row_dim */
	if((a_dev->row_dim != b_dev->row_dim) || (b_dev->row_dim != c_dev->row_dim) || (c_dev->row_dim != a_dev->row_dim))
	{
		fprintf(stderr, "ERROR: sub_gqdmatrix_dev\n");
		return;
	}
	row_dim = c_dev->row_dim;

	/* check col_dim */
	if((a_dev->col_dim != b_dev->col_dim) || (b_dev->col_dim != c_dev->col_dim) || (c_dev->col_dim != a_dev->col_dim))
	{
		fprintf(stderr, "ERROR: sub_gqdmatrix_dev\n");
		return;
	}
	col_dim = c_dev->col_dim;

	_bncu_sub_gqdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, row_dim * col_dim);

}

/* c := sc * a */
void cmul_gqdmatrix_dev(GQDMatrix c_dev, gqd_real sc, GQDMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int row_dim, col_dim; //, index;

	/* check row_dim */
	if(a_dev->row_dim != c_dev->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_gqdmatrix_dev\n");
		return;
	}
	row_dim = c_dev->row_dim;

	/* check col_dim */
	if(a_dev->col_dim != c_dev->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_gqdmatrix_dev\n");
		return;
	}
	col_dim = c_dev->col_dim;

	_bncu_cmul_gqdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, sc, a_dev->element, row_dim * col_dim);


}

/* c = a^T */
__global__ void _bncu_transpose_gqdmatrix(gqd_real *c_dev_element, gqd_real *a_dev_element, long int row_dim, long int col_dim)
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

void transpose_gqdmatrix_dev(GQDMatrix c_dev, GQDMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	//long int i, j;

	/* Check Dimensions */
	if((c_dev->row_dim != a_dev->col_dim) || (c_dev->col_dim != a_dev->row_dim))
	{
		fprintf(stderr, "ERROR: transpose_gqdmatrix_dev\n");
		return;
	}
	
	_bncu_transpose_gqdmatrix<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, c_dev->row_dim, c_dev->col_dim);

/*	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			c->element[i * c->col_dim + j] = a->element[j * a->col_dim + i];
	}
*/
}

/* c := a */
void subst_gqdmatrix_dev(GQDMatrix c_dev, GQDMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	//long int i, j, index;

	if((c_dev->row_dim != a_dev->row_dim) || (c_dev->col_dim != a_dev->col_dim))
	{
		fprintf(stderr, "ERROR: subst_gqdmatrix_dev\n");
		return;
	}

	_bncu_subst_gqdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, c_dev->row_dim * c_dev->col_dim);

}

/* c := 0 */
void set0_gqdmatrix_dev(GQDMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block)
{
//	long int i, j;

	_bncu_set0_gqdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, c_dev->row_dim * c_dev->col_dim);

}

/* set (i, j)-element on GPU from gqd_real */
void set_gqdmatrix_ij_dev(GQDMatrix mat_dev, long int row_index, long int col_index, gqd_real val)
{
	if((row_index >= 0) && (row_index < mat_dev->row_dim) && (col_index >= 0) && (col_index < mat_dev->col_dim))
		gqd_dev2gqd(&(mat_dev->element[row_index * mat_dev->col_dim + col_index]), &val);
}

/* set (i, j)-element on GPU from dd_real */
void set_gqdmatrix_ij_qd_dev(GQDMatrix mat_dev, long int row_index, long int col_index, qd_real val)
{
	if((row_index >= 0) && (row_index < mat_dev->row_dim) && (col_index >= 0) && (col_index < mat_dev->col_dim))
		qd2gqd_dev(&(mat_dev->element[row_index * mat_dev->col_dim + col_index]), &val);
}


/* c := I */
void setI_gqdmatrix_dev(GQDMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i;
	qd_real one;

	one = (qd_real)1.0;

	_bncu_set0_gqdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, c_dev->row_dim * c_dev->col_dim);

	for(i = 0; i < c_dev->row_dim; i++)
		set_gqdmatrix_ij_qd_dev(c_dev, i, i, one);
/*
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			SET0_GQD(c->element[i * c->col_dim + j]); // = 0.0;
		if(i < c->col_dim)
			SET1_GQD(c->element[i * c->col_dim + i]); // = 1.0;
	}
*/
}

/* v := a * vb */
__global__ void _bncu_mul_gqdmatrix_gqdvec(gqd_real *v_element, gqd_real *a_element, gqd_real *vb_element, long int row_dim, long int col_dim)
{
	int j, index;
	gqd_real tmp;

	index = threadIdx.x + blockIdx.x * blockDim.x;

	//for(i = 0; i < row_dim; i++)
	while(index < row_dim)
	{
		SET0_GQD(tmp); // = 0.0;
		for(j = 0; j < col_dim; j++)
			tmp = tmp + a_element[index * col_dim + j] * vb_element[j];

		v_element[index] = tmp;

		__syncthreads();

		index += blockDim.x * gridDim.x;
	}
}

void mul_gqdmatrix_gqdvec(GQDVector v, GQDMatrix a, GQDVector vb, int num_blocks_per_grid, int num_threads_per_block)
{
	/* Check Dimension */
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: mul_gqdmatrix_qdvec\n");
		return;
	}

	_bncu_mul_gqdmatrix_gqdvec<<<num_blocks_per_grid, num_threads_per_block>>>(v->element, a->element, vb->element, a->row_dim, a->col_dim);

/*	for(i = 0; i < a->row_dim; i++)
	{
		SET0_GQD(tmp); // = 0.0;
		for(j = 0; j < a->col_dim; j++)
			tmp = tmp + a->element[i * a->col_dim + j] * vb->element[j];

		v->element[i] = tmp;
	}
*/
}

/* v := a^T * vb */
__global__ void _bncu_mul_gqdmatrixt_gqdvec(gqd_real *v_element, gqd_real *a_element, gqd_real *vb_element, long int row_dim, long int col_dim)
{
	int j, index;
	gqd_real tmp;

	index = threadIdx.x + blockIdx.x * blockDim.x;

	//for(i = 0; i < col_dim; i++)
	while(index < col_dim)
	{
		SET0_GQD(tmp); // = 0.0;
		for(j = 0; j < row_dim; j++)
			tmp = tmp + a_element[j * row_dim + index] * vb_element[j];

		v_element[index] = tmp;

		__syncthreads();

		index += blockDim.x * gridDim.x;
	}
}

__host__ void mul_gqdmatrixt_gqdvec(GQDVector v, GQDMatrix a, GQDVector vb, int num_blocks_per_grid, int num_threads_per_block)
{

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_gqdmatrixt_gqdvec\n");
		return;
	}

	_bncu_mul_gqdmatrixt_gqdvec<<<num_blocks_per_grid, num_threads_per_block>>>(v->element, a->element, vb->element, a->row_dim, a->col_dim);

/*	for(i = 0; i < a->col_dim; i++)
	{
		SET0_GQD(tmp); // = 0.0;
		for(j = 0; j < a->row_dim; j++)
			tmp = tmp + a->element[j * a->row_dim + i] * vb->element[j];

		v->element[i] = tmp;
	}
*/
}
