/********************************************************************************/
/* gtdlinear.cu: Triple-double and Quadruple precision                          */
/*               Linear Computation Library with GQD and CUDA                   */
/* Copyright (C) 2015-2026 Tomonori Kouya                                            */
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

#include "gtdlinear.h" // GTDVector, GTDMatrix
	/* Pull only declarations here — gddlinear.cu owns the gqd.cu /
	 * common.cu device + host definitions for the whole library.
	 * Cross-TU resolution of __device__ symbols relies on -rdc=true
	 * + nvcc -dlink (see src/cuda/Makefile.in).  Same pattern as
	 * gqdlinear.cu. */
	#include "common.cuh"
	#include "gqd.cuh"

// initialize gtdvector on CPU(host)
__host__ GTDVector init_gtdvector(long int dim)
{
	long int index;
	GTDVector ret = NULL;
	td_real zero = 0.0;

	// callocation
	ret = (gtdvector *)malloc(sizeof(gtdvector));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one GTDVector\n");
		return ret;
	}

	ret->dim = dim;

	ret->element = (gtd_real *)calloc(dim, sizeof(gtd_real));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GTDVector(dim = %ld)\n", dim);
		cudaFree(ret);
		return NULL;
	}

	// set all zeros 
	for(index = 0; index < dim; index++)
		td2gtd(&ret->element[index], &zero);
		//ret->element[index] = (gtd_real)0.0;

	return ret;
}

// free gtdvector on CPU(HOST)
__host__ void free_gtdvector(GTDVector vec)
{
	free(vec->element);
	free(vec);
}

// initialize gtdvector
__host__ GTDVector init_gtdvector_dev(long int dim)
{
	long int index;
	GTDVector ret = NULL;
	td_real zero = 0.0;

	// allocation
	ret = (gtdvector *)malloc(sizeof(gtdvector));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one TDVector\n");
		return ret;
	}

	ret->element = NULL;
	ret->dim = dim;

	//ret->element = (td_real *)calloc(dim, sizeof(td_real));
	cudaMalloc((void **)&(ret->element), (size_t)(dim * sizeof(gtd_real)));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GTDVector(dim = %ld)\n", dim);
		cudaFree(ret);
		return NULL;
	}

	// set all zeros 
	for(index = 0; index < dim; index++)
		td2gtd_dev(&(ret->element[index]), &zero);

	return ret;
}

// free ddvector
__host__ void free_gtdvector_dev(GTDVector vec)
{
	cudaFree(vec->element);
	free(vec);
}

// copy TDVector to GTDVector on GPU
// gddvec on GPU := ddvec
__host__ void subst_gtdvector_dev_tdvec(GTDVector gddvec_dev, TDVector ddvec)
{
	long int i, dim;

	dim = gddvec_dev->dim;

	for(i = 0; i < dim; i++)
	{
		/* TDVector::element is SoA: double *element[TDSIZE] with TDSIZE=3
		 * (hi=[0], mid=[1], lo=[2]).  Materialize a 3-limb td_real to feed
		 * td2gtd_dev.  BUG FIX: previously copied only x[0],x[1] which
		 * silently dropped the 3rd limb -> TD computation degenerated to
		 * DD precision. */
		td_real tmp_dd;
		tmp_dd.x[0] = ddvec->element[0][i];
		tmp_dd.x[1] = ddvec->element[1][i];
		tmp_dd.x[2] = ddvec->element[2][i];
		td2gtd_dev(&(gddvec_dev->element[i]), &tmp_dd);
	}
}

// copy GTDVector on GPU to TDVector
// ddvec := gddvec_dev on GPU
__host__ void subst_tdvector_gtdvec_dev(TDVector ddvec, GTDVector gddvec_dev)
{
	long int i, dim;

	dim = ddvec->dim;

	for(i = 0; i < dim; i++)
	{
		/* SoA layout (TDSIZE=3): copy ALL 3 limbs.  See subst_gtdvector_dev_tdvec. */
		td_real tmp_dd;
		gtd2td_dev(&tmp_dd, &(gddvec_dev->element[i]));
		ddvec->element[0][i] = tmp_dd.x[0];
		ddvec->element[1][i] = tmp_dd.x[1];
		ddvec->element[2][i] = tmp_dd.x[2];
	}
}

// print ddvector
__host__ void print_gtdvector_dev(GTDVector dev_vec)
{
	long int index, dim;
	GTDVector host_vec;

	dim = dev_vec->dim;

	host_vec = init_gtdvector(dim);

	cudaMemcpy((void *)(host_vec->element), (void *)(dev_vec->element), size_t(sizeof(gtd_real) * dim), cudaMemcpyDeviceToHost);

	for(index = 0; index < dim; index++)
	{
		printf("%4ld: ", index);

		std::cout << gtd_get_td(host_vec->element[index]).to_string() << "\n";
	}

	free(host_vec);
}

/*************************************************/
/* Vector Calculations for TDVector               */
/*
void add_gtdvector_dev(GTDVector c_dev, GTDVector a_dev, GTDVector b_dev)
void add2_gtdvector_dev(GTDVector c_dev, GTDVector a_dev)
void sub_gtdvector_dev(GTDVector c_dev, GTDVector a_dev, GTDVector b_dev)
void sub2_gtdvector_dev(GTDVector c_dev, GTDVector a_dev)
void cmul_gtdvector_dev(GTDVector c_dev, gtd_real val_dev, GTDVector a_dev)
void cmul2_gtdvector_dev(GTDVector c_dev, gtd_real val_dev)
void add_cmul_gtdvector_dev(GTDVector c_dev, GTDVector a_dev, gtd_real val_dev, GTDVector b_dev)
double ip_gtdvector_dev(GTDVector a, GTDVector b_dev)
double norm1_gtdvector_dev(GTDVector a_dev)
double norm2_gtdvector_dev(GTDVector a_dev)
double normi_gtdvector_dev(GTDVector a_dev)
void subst_gtdvector_dev(GTDVector c_dev, GTDVector a_dev)
*/
/*************************************************/
/* c = a + b */
__global__ void _bncu_add_gtdvector(gtd_real *c_dev_element, gtd_real *a_dev_element, gtd_real *b_dev_element, long int dim)
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

__host__ void add_gtdvector_dev(GTDVector c_dev, GTDVector a_dev, GTDVector b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if((a_dev->dim != b_dev->dim) || (c_dev->dim != a_dev->dim) || (c_dev->dim != b_dev->dim))
	{
		fprintf(stderr, "ERROR: add_gtdvector_dev\n");
		return;
	}

/*	for(i = 0; i < c->dim; i++)
	{
		c->element[i] = a->element[i] + b->element[i];
	}
*/
	_bncu_add_gtdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, c_dev->dim);
}


/* c = a - b */
__global__ void _bncu_sub_gtdvector(gtd_real *c_dev_element, gtd_real *a_dev_element, gtd_real *b_dev_element, long int dim)
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

__host__ void sub_gtdvector_dev(GTDVector c_dev, GTDVector a_dev, GTDVector b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if((a_dev->dim != b_dev->dim) || (c_dev->dim != a_dev->dim) || (c_dev->dim != b_dev->dim))
	{
		fprintf(stderr, "ERROR: sub_gtdvector_dev\n");
		return;
	}

/*	for(i = 0; i < c->dim; i++)
	{
		c->element[i] = a->element[i] - b->element[i];
	}
*/
	_bncu_sub_gtdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, c_dev->dim);
}

// cmul vector (val, core)
// dev_ret := dev_val * dev_a on GPU
__global__ void _bncu_cmul_gtdvector(gtd_real *dev_ret_element, gtd_real val, gtd_real *dev_a_element, long int dim)
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
__host__ void cmul_gtdvector_dev(GTDVector c_dev, gtd_real val, GTDVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if(c_dev->dim != a_dev->dim)
	{
		fprintf(stderr, "ERROR: cmul_gtdvector_dev\n");
		return;
	}

/*	for(i = 0; i < c_dev->dim; i++)
	{
		c->element[i] = val * a->element[i];
	}
*/

	_bncu_cmul_gtdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, val, a_dev->element, c_dev->dim);

}

// subst vector (val, core)
// dev_ret := dev_vec on GPU
__global__ void _bncu_subst_gtdvector(gtd_real *dev_ret_element, gtd_real *dev_vec_element, long int dim)
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
__host__ void subst_gtdvector_dev(GTDVector ret_dev, GTDVector vec_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if(ret_dev->dim != vec_dev->dim)
	{
		fprintf(stderr, "ERROR: subst_gtdvector_dev\n");
		return;
	}

	_bncu_subst_gtdvector<<< num_blocks_per_grid, num_threads_per_block >>>(ret_dev->element, vec_dev->element, ret_dev->dim);

}

// set0 
// dev_ret := 0 on GPU
__global__ void _bncu_set0_gtdvector(gtd_real *dev_ret_element, long int dim)
{
	int index;

	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim)
	{
		SET0_GTD(dev_ret_element[index]);
		index += blockDim.x * gridDim.x;
	}

	__syncthreads();

}

/* c := 0 */
__host__ void set0_gtdvector_dev(GTDVector ret_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	_bncu_set0_gtdvector<<< num_blocks_per_grid, num_threads_per_block >>>(ret_dev->element, ret_dev->dim);
}

// ret[i] := a_vec[i] * b_vec[i]
__global__ void _bncu_mul_gtdvector(gtd_real *ret_dev_element, gtd_real *a_dev_element, gtd_real *b_dev_element, long int dim)
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
__global__ void _bncu_sqr_gtdvector(gtd_real *ret_dev_element, gtd_real *vec_dev_element, long int dim)
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
__global__ void _bncu_sqrt_gtdvector(gtd_real *ret_dev_element, gtd_real *vec_dev_element, long int dim)
{
	int index;

	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim)
	{
		ret_dev_element[index] = (gtd_real)sqrt((gtd_real)vec_dev_element[index]);
		index += blockDim.x * gridDim.x;
	}

	__syncthreads();
}

// ret[i] := abs(vec[i])
__global__ void _bncu_abs_gtdvector(gtd_real *ret_dev_element, gtd_real *vec_dev_element, long int dim)
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
__global__ void _bncu_add_reduct_gtdvector(gtd_real *ret_dev, gtd_real *vec_dev_element, long int dim)
{
	int index, cache_index;
	__shared__ gtd_real cache[MAX_NUM_THREADS_PER_BLOCK];

	// Inside Block

	index = threadIdx.x + blockIdx.x * blockDim.x;
	cache_index = threadIdx.x;

	SET0_GTD(cache[cache_index]);

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
__global__ void _bncu_max_reduct_gtdvector(gtd_real *ret_dev, gtd_real *vec_dev_element, long int dim)
{
	int index, cache_index;
	__shared__ gtd_real cache[MAX_NUM_THREADS_PER_BLOCK];

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
__host__ void ip_gtdvector_dev(gtd_real *ret_dev, GTDVector a_dev, GTDVector b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gtd_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim != b_dev->dim)
	{
		fprintf(stderr, "ERROR: ip_gtdvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gtd_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gtd_real) * num_blocks_per_grid));

	// c[i] := a[i] * b[i]
	_bncu_mul_gtdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, b_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	//block_dim = 1;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gtdvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gtd_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret_dev := block_cache_dev[0]
	cudaMemcpy((void *)ret_dev, (void *)&(block_cache_dev[0]), sizeof(gtd_real), cudaMemcpyDeviceToDevice);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// norm2
__host__ void norm2_gtdvector_dev(gtd_real *ret_dev, GTDVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gtd_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim <= 0)
	{
		fprintf(stderr, "ERROR: norm2_gtdvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gtd_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gtd_real) * num_blocks_per_grid));

	// c[i] := a[i]^2
	_bncu_sqr_gtdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gtdvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gtd_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret := sqrt(ret);
	_bncu_sqrt_gtdvector<<<1, 1>>>(ret_dev, block_cache_dev, 1);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// norm1
__host__ void norm1_gtdvector_dev(gtd_real *ret_dev, GTDVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gtd_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim <= 0)
	{
		fprintf(stderr, "ERROR: norm1_gtdvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gtd_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gtd_real) * num_blocks_per_grid));

	// c[i] := abs(a[i])
	_bncu_abs_gtdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gtdvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gtd_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret_dev := block_cache_dev[0]
	cudaMemcpy((void *)ret_dev, (void *)&(block_cache_dev[0]), sizeof(gtd_real), cudaMemcpyDeviceToDevice);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// norm_inf
__host__ void normi_gtdvector_dev(gtd_real *ret_dev, GTDVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gtd_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim <= 0)
	{
		fprintf(stderr, "ERROR: normi_gtdvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gtd_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gtd_real) * num_blocks_per_grid));

	// c[i] := abs(a[i])
	_bncu_abs_gtdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_max_reduct_gtdvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gtd_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret_dev := block_cache_dev[0]
	cudaMemcpy((void *)ret_dev, (void *)&(block_cache_dev[0]), sizeof(gtd_real), cudaMemcpyDeviceToDevice);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// GDD matrix

// set a zero matrix
__host__ void set0_gtdmatrix(GTDMatrix mat, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i, j;
	td_real ddzero = 0.0;
	gtd_real zero;

	td2gtd(&zero, &ddzero);

	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
			mat->element[i * mat->col_dim + j] = zero;
	}
}

// initialize ddvector
GTDMatrix init_gtdmatrix(long int row_dim, long int col_dim)
{
	long int row_index, col_index;
	GTDMatrix ret = NULL;
	td_real ddzero = 0.0;
	gtd_real zero;

	td2gtd(&zero, &ddzero);

	// callocation
	ret = (gtdmatrix *)malloc(sizeof(gtdmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one GTDMatrix\n");
		return ret;
	}

	ret->row_dim = row_dim;
	ret->col_dim = col_dim;

	ret->element = (gtd_real *)calloc(row_dim * col_dim, sizeof(gtd_real));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GTDMatrix(row_dim, col_dim = %ld, %ld)\n", row_dim, col_dim);
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
void free_gtdmatrix(GTDMatrix mat)
{
	free(mat->element);
	free(mat);
}

// initialize ddvector
GTDMatrix init_gtdmatrix_dev(long int row_dim, long int col_dim)
{
	long int row_index, col_index;
	GTDMatrix ret = NULL;
	td_real ddzero = 0.0;
	gtd_real zero;

	td2gtd(&zero, &ddzero);

	// callocation
	ret = (gtdmatrix *)malloc(sizeof(gtdmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one GTDMatrix\n");
		return ret;
	}

	ret->row_dim = row_dim;
	ret->col_dim = col_dim;

	ret->element = (gtd_real *)calloc(row_dim * col_dim, sizeof(gtd_real));
	cudaMalloc((void **)&(ret->element), (size_t)(row_dim * col_dim * sizeof(gtd_real)));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GTDMatrix(row_dim, col_dim = %ld, %ld)\n", row_dim, col_dim);
		free(ret);
		return NULL;
	}

	// set all zeros 
	for(row_index = 0; row_index < row_dim; row_index++)
	{
		for(col_index = 0; col_index < col_dim; col_index++)
			gtd2gtd_dev(&(ret->element[row_index * col_dim + col_index]), &zero);
	}

	return ret;
}

// free ddvector
void free_gtdmatrix_dev(GTDMatrix mat)
{
	cudaFree(mat->element);
	free(mat);
}

// print ddvector
void print_gtdmatrix_dev(GTDMatrix mat)
{
	long int row_index, col_index;
	td_real ddval;

	for(row_index = 0; row_index < mat->row_dim; row_index++)
	{
		for(col_index = 0; col_index < mat->col_dim; col_index++)
		{
			printf("%ld, %ld: ", row_index, col_index);
			gtd2td_dev(&ddval, &(mat->element[row_index * mat->col_dim + col_index]));
			std::cout << ddval.to_string() << "\n";
		}
	}
}


__global__ void _bncu_mul_gtdmatrix(gtd_real *ret_dev_element, long int row_dim, long int col_dim, long int mid_dim, gtd_real *a_dev_element, gtd_real *b_dev_element)
{
	long int i, j, k;
	gtd_real tmp, tmp1;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	//for(i = 0; i < row_dim; i++)
	while(i < row_dim)
	{
		for(j = 0; j < col_dim; j++)
		{
			SET0_GTD(tmp1); // = (gtd_real)0.0;
			SET0_GTD(tmp); // = (gtd_real)0.0;
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
void mul_gtdmatrix_dev(GTDMatrix ret_dev, GTDMatrix a_dev, GTDMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long row_dim, col_dim, mid_dim;

	/* dimension check */
	if((ret_dev->row_dim != a_dev->row_dim) || (ret_dev->col_dim != b_dev->col_dim) || (a_dev->col_dim != b_dev->row_dim))
	{
		fprintf(stderr, "ERROR: mul_gtdmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret_dev->row_dim, ret_dev->col_dim, a_dev->row_dim, a_dev->col_dim, b_dev->row_dim, b_dev->col_dim);
		return;
	}

	row_dim = ret_dev->row_dim;
	col_dim = ret_dev->col_dim;
	mid_dim = a_dev->col_dim;

	_bncu_mul_gtdmatrix<<<num_blocks_per_grid, num_threads_per_block>>>(ret_dev->element, row_dim, col_dim, mid_dim, a_dev->element, b_dev->element);

}

// copy TDMatrix to GTDMatrix on GPU
// gddvec on GPU := ddvec
__host__ void subst_gtdmatrix_dev_tdmat(GTDMatrix gddmat_dev, TDMatrix ddmat)
{
	long int i, j, row_dim, col_dim;

	row_dim = gddmat_dev->row_dim;
	col_dim = gddmat_dev->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			/* TDMatrix::element is SoA: double *element[TDSIZE=3].
			 * Materialize an AoS td_real (3 limbs) for td2gtd_dev.
			 * BUG FIX: previously copied only x[0],x[1] -> 3rd limb dropped
			 * -> GPU TD matvec computed at DD precision (~1e-31 instead of ~1e-46). */
			td_real tmp_dd;
			long int idx = i * col_dim + j;
			tmp_dd.x[0] = ddmat->element[0][idx];
			tmp_dd.x[1] = ddmat->element[1][idx];
			tmp_dd.x[2] = ddmat->element[2][idx];
			td2gtd_dev(&(gddmat_dev->element[idx]), &tmp_dd);
		}
	}
}

// copy GTDMatrix on GPU to TDMatrix
// ddvec := gddmat_dev on GPU
__host__ void subst_tdmatrix_gtdmat_dev(TDMatrix ddmat, GTDMatrix gddmat_dev)
{
	long int i, j, row_dim, col_dim;

	row_dim = gddmat_dev->row_dim;
	col_dim = gddmat_dev->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			/* SoA layout (TDSIZE=3): copy ALL 3 limbs.  See subst_gtdmatrix_dev_tdmat. */
			td_real tmp_dd;
			long int idx = i * col_dim + j;
			gtd2td_dev(&tmp_dd, &(gddmat_dev->element[idx]));
			ddmat->element[0][idx] = tmp_dd.x[0];
			ddmat->element[1][idx] = tmp_dd.x[1];
			ddmat->element[2][idx] = tmp_dd.x[2];
		}
	}
}

// Frobenius norm
__host__ void normf_gtdmatrix_dev(gtd_real *ret_dev, GTDMatrix mat_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, tmp_vec_dev_dim, total_dim;
//	gtd_real tmp, tmp1;
	gtd_real *block_cache_dev; // partial sum per block
	gtd_real *tmp_vec_dev; // square mul of mat

	total_dim = mat_dev->row_dim * mat_dev->col_dim; // total dimension as one vector

	// initialize block_cache_dev, tmp_vec_dev
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gtd_real) * num_blocks_per_grid));
	cudaMalloc((void **)&tmp_vec_dev, (size_t)(sizeof(gtd_real) * total_dim));

	// mat[i][j]^2
	//_bncu_mul_gtdvector<<<num_blocks_per_grid, num_threads_per_block>>>(tmp_vec_dev, mat_dev->element, mat_dev->element, total_dim);
	_bncu_sqr_gtdvector<<<num_blocks_per_grid, num_threads_per_block>>>(tmp_vec_dev, mat_dev->element, total_dim);

	// reduction to add
	block_dim = num_blocks_per_grid;
//	block_dim = 1;
	thread_dim = num_threads_per_block;
	tmp_vec_dev_dim = total_dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gtdvector<<<block_dim, thread_dim>>>(block_cache_dev, tmp_vec_dev, tmp_vec_dev_dim);

		if(block_dim <= 1)
			break;

		tmp_vec_dev_dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)tmp_vec_dev, (void *)block_cache_dev, (size_t)(sizeof(gtd_real) * tmp_vec_dev_dim), cudaMemcpyDeviceToDevice);

	}

	// ret := sqrt(ret);
	_bncu_sqrt_gtdvector<<<1, 1>>>(ret_dev, block_cache_dev, 1);

	// free
	cudaFree(block_cache_dev);
	cudaFree(tmp_vec_dev);

/*
	SET0_GTD(*ret); // = (gtd_real)0.0;
	SET0_GTD(tmp); // = (gtd_real)(int)0;
	SET0_GTD(tmp1); // = (gtd_real)(int)0;

	dim = mat->row_dim * mat->col_dim;

	for(i = 0; i < dim ; i++)
	{
		//tmp += mat->element[i] * mat->element[i];
		//tmp += gtd_real::mul((gtd_real)(mat->element[i]), (gtd_real)(mat->element[i]));
		tmp1 = (gtd_real)(mat->element[i]) * (gtd_real)(mat->element[i]);
		tmp = tmp + tmp1;
	}

	//*ret = sqrt(tmp);
	tmp = (gtd_real)sqrt(tmp);
	*ret = tmp;
*/
}

/*************************************************/
/* Matrix Caluculations for GTDMatrix            */
/*
void normf_gtdmatrix(double ret[TDSIZE], GTDMatrix mat)
void norm1_gtdmatrix(double ret[TDSIZE], GTDMatrix mat)
void normi_gtdmatrix(double ret[TDSIZE], GTDMatrix mat)
void add_gtdmatrix(GTDMatrix c, GTDMatrix a, GTDMatrix b);
void sub_gtdmatrix(GTDMatrix c, GTDMatrix a, GTDMatrix b);
void mul_gtdmatrix(GTDMatrix c, GTDMatrix a, GTDMatrix b);
void mul_gtdmatrix_ddvec(TDVector v, GTDMatrix a, TDVector vb)
void mul_gtdmatrixt_ddvec(TDVector v, GTDMatrix a, TDVector vb)
void transpose_gtdmatrix(GTDMatrix c, GTDMatrix a);
void inv_gtdmatrix(GTDMatrix a);
void subst_mpfmatrux(GTDMatrix c, GTDMatrix a);
*/
/*************************************************/

#if 0
/* Infinity Norm of Matrix */
void normi_gtdmatrix(gtd_real *ret, GTDMatrix mat, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i, j;
	gtd_real sum;

	SET0_GTD(*ret); // = 0.0;

	for(i = 0; i < mat->row_dim; i++)
	{
		SET0_GTD(sum); //  = 0.0;
		for(j = 0; j < mat->col_dim; j++)
			sum = sum + abs(mat->element[i * mat->row_dim + j]);

		if(*ret < sum)
			*ret = sum;

	}

	return;
}

/* 1 Norm of Matrix */
void norm1_gtdmatrix(gtd_real *ret, GTDMatrix mat, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i, j;
	gtd_real sum;

	SET0_GTD(*ret); // = 0.0;

	for(j = 0; j < mat->col_dim; j++)
	{
		SET0_GTD(sum); // = 0.0;
		for(i = 0; i < mat->row_dim; i++)
			sum = sum + abs(mat->element[i * mat->col_dim + j]);

		if(*ret < sum)
			*ret = sum;

	}

	return;
}
#endif // 0

/* c := a + b */
void add_gtdmatrix_dev(GTDMatrix c_dev, GTDMatrix a_dev, GTDMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int row_dim, col_dim; //, index;

	/* check row_dim */
	if((a_dev->row_dim != b_dev->row_dim) || (b_dev->row_dim != c_dev->row_dim) || (c_dev->row_dim != a_dev->row_dim))
	{
		fprintf(stderr, "ERROR: add_gtdmatrix_dev\n");
		return;
	}
	row_dim = c_dev->row_dim;

	/* check col_dim */
	if((a_dev->col_dim != b_dev->col_dim) || (b_dev->col_dim != c_dev->col_dim) || (c_dev->col_dim != a_dev->col_dim))
	{
		fprintf(stderr, "ERROR: add_gtdmatrix_dev\n");
		return;
	}
	col_dim = c_dev->col_dim;

	_bncu_add_gtdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, row_dim * col_dim);

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
void sub_gtdmatrix_dev(GTDMatrix c_dev, GTDMatrix a_dev, GTDMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int row_dim, col_dim; //, index;

	/* check row_dim */
	if((a_dev->row_dim != b_dev->row_dim) || (b_dev->row_dim != c_dev->row_dim) || (c_dev->row_dim != a_dev->row_dim))
	{
		fprintf(stderr, "ERROR: sub_gtdmatrix_dev\n");
		return;
	}
	row_dim = c_dev->row_dim;

	/* check col_dim */
	if((a_dev->col_dim != b_dev->col_dim) || (b_dev->col_dim != c_dev->col_dim) || (c_dev->col_dim != a_dev->col_dim))
	{
		fprintf(stderr, "ERROR: sub_gtdmatrix_dev\n");
		return;
	}
	col_dim = c_dev->col_dim;

	_bncu_sub_gtdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, row_dim * col_dim);

}

/* c := sc * a */
void cmul_gtdmatrix_dev(GTDMatrix c_dev, gtd_real sc, GTDMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int row_dim, col_dim; //, index;

	/* check row_dim */
	if(a_dev->row_dim != c_dev->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_gtdmatrix_dev\n");
		return;
	}
	row_dim = c_dev->row_dim;

	/* check col_dim */
	if(a_dev->col_dim != c_dev->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_gtdmatrix_dev\n");
		return;
	}
	col_dim = c_dev->col_dim;

	_bncu_cmul_gtdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, sc, a_dev->element, row_dim * col_dim);


}

/* c = a^T */
__global__ void _bncu_transpose_gtdmatrix(gtd_real *c_dev_element, gtd_real *a_dev_element, long int row_dim, long int col_dim)
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

void transpose_gtdmatrix_dev(GTDMatrix c_dev, GTDMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	//long int i, j;

	/* Check Dimensions */
	if((c_dev->row_dim != a_dev->col_dim) || (c_dev->col_dim != a_dev->row_dim))
	{
		fprintf(stderr, "ERROR: transpose_gtdmatrix_dev\n");
		return;
	}
	
	_bncu_transpose_gtdmatrix<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, c_dev->row_dim, c_dev->col_dim);

/*	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			c->element[i * c->col_dim + j] = a->element[j * a->col_dim + i];
	}
*/
}

/* c := a */
void subst_gtdmatrix_dev(GTDMatrix c_dev, GTDMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	//long int i, j, index;

	if((c_dev->row_dim != a_dev->row_dim) || (c_dev->col_dim != a_dev->col_dim))
	{
		fprintf(stderr, "ERROR: subst_gtdmatrix_dev\n");
		return;
	}

	_bncu_subst_gtdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, c_dev->row_dim * c_dev->col_dim);

}

/* c := 0 */
void set0_gtdmatrix_dev(GTDMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block)
{
//	long int i, j;

	_bncu_set0_gtdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, c_dev->row_dim * c_dev->col_dim);

}

/* set (i, j)-element on GPU from td_real */
void set_gtdmatrix_ij_dev(GTDMatrix mat_dev, long int row_index, long int col_index, td_real val)
{
	if((row_index >= 0) && (row_index < mat_dev->row_dim) && (col_index >= 0) && (col_index < mat_dev->col_dim))
		td2gtd_dev(&(mat_dev->element[row_index * mat_dev->col_dim + col_index]), &val);
}

/* c := I */
void setI_gtdmatrix_dev(GTDMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i;
	td_real one;

	one = (td_real)1.0;

	_bncu_set0_gtdvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, c_dev->row_dim * c_dev->col_dim);

	for(i = 0; i < c_dev->row_dim; i++)
		set_gtdmatrix_ij_dev(c_dev, i, i, one);
/*
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			SET0_GTD(c->element[i * c->col_dim + j]); // = 0.0;
		if(i < c->col_dim)
			SET1_GTD(c->element[i * c->col_dim + i]); // = 1.0;
	}
*/
}

/* v := a * vb */
__global__ void _bncu_mul_gtdmatrix_gddvec(gtd_real *v_element, gtd_real *a_element, gtd_real *vb_element, long int row_dim, long int col_dim)
{
	int j, index;
	gtd_real tmp;

	index = threadIdx.x + blockIdx.x * blockDim.x;

	//for(i = 0; i < row_dim; i++)
	while(index < row_dim)
	{
		SET0_GTD(tmp); // = 0.0;
		for(j = 0; j < col_dim; j++)
			tmp = tmp + a_element[index * col_dim + j] * vb_element[j];

		v_element[index] = tmp;

		__syncthreads();

		index += blockDim.x * gridDim.x;
	}
}

void mul_gtdmatrix_gddvec(GTDVector v, GTDMatrix a, GTDVector vb, int num_blocks_per_grid, int num_threads_per_block)
{
	/* Check Dimension */
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: mul_gtdmatrix_ddvec\n");
		return;
	}

	_bncu_mul_gtdmatrix_gddvec<<<num_blocks_per_grid, num_threads_per_block>>>(v->element, a->element, vb->element, a->row_dim, a->col_dim);

/*	for(i = 0; i < a->row_dim; i++)
	{
		SET0_GTD(tmp); // = 0.0;
		for(j = 0; j < a->col_dim; j++)
			tmp = tmp + a->element[i * a->col_dim + j] * vb->element[j];

		v->element[i] = tmp;
	}
*/
}

/* v := a^T * vb */
__global__ void _bncu_mul_gtdmatrixt_gddvec(gtd_real *v_element, gtd_real *a_element, gtd_real *vb_element, long int row_dim, long int col_dim)
{
	int j, index;
	gtd_real tmp;

	index = threadIdx.x + blockIdx.x * blockDim.x;

	//for(i = 0; i < col_dim; i++)
	while(index < col_dim)
	{
		SET0_GTD(tmp); // = 0.0;
		for(j = 0; j < row_dim; j++)
			tmp = tmp + a_element[j * row_dim + index] * vb_element[j];

		v_element[index] = tmp;

		__syncthreads();

		index += blockDim.x * gridDim.x;
	}
}

__host__ void mul_gtdmatrixt_gddvec(GTDVector v, GTDMatrix a, GTDVector vb, int num_blocks_per_grid, int num_threads_per_block)
{

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_gtdmatrixt_gddvec\n");
		return;
	}

	_bncu_mul_gtdmatrixt_gddvec<<<num_blocks_per_grid, num_threads_per_block>>>(v->element, a->element, vb->element, a->row_dim, a->col_dim);

/*	for(i = 0; i < a->col_dim; i++)
	{
		SET0_GTD(tmp); // = 0.0;
		for(j = 0; j < a->row_dim; j++)
			tmp = tmp + a->element[j * a->row_dim + i] * vb->element[j];

		v->element[i] = tmp;
	}
*/
}

/* -------------------------------------------------------------------
 *  Properly-named aliases for the same-precision matvec operations.
 *  mul_gtdmatrix_gddvec / mul_gtdmatrixt_gddvec above are misnamed —
 *  despite "_gddvec" in the name they take GTDVector (gtd_real*) for
 *  ALL arguments and operate at TD precision.  gtdlinear.h declares
 *  the proper-named gtdvec variants that the bench programs expect;
 *  these wrappers satisfy the link without breaking _gddvec callers.
 * ------------------------------------------------------------------- */
__host__ void mul_gtdmatrix_gtdvec(GTDVector v, GTDMatrix a, GTDVector vb,
                                    int num_blocks_per_grid, int num_threads_per_block)
{
	mul_gtdmatrix_gddvec(v, a, vb, num_blocks_per_grid, num_threads_per_block);
}

__host__ void mul_gtdmatrixt_gtdvec(GTDVector v, GTDMatrix a, GTDVector vb,
                                     int num_blocks_per_grid, int num_threads_per_block)
{
	mul_gtdmatrixt_gddvec(v, a, vb, num_blocks_per_grid, num_threads_per_block);
}

// Test main function
#ifdef DEBUG

using namespace std;

//#define ROW_DIM 10
//#define ROW_DIM 32
//#define ROW_DIM 128
#define ROW_DIM 512
#define COL_DIM ROW_DIM

int main()
{
	int i, j;
	TDVector ddvec_a, ddvec_b, ddvec_c;
	TDMatrix ddmat_a, ddmat_b, ddmat_c;
	td_real ddval;
	int num_blocks, num_threads;
	GTDVector gddvec_a, gddvec_b, gddvec_c;
	gtd_real gddval, *ptr_gddval_dev;
	GTDMatrix gddmat_a, gddmat_b, gddmat_c;
	

	// Initialize QD library
	fpu_fix_start(NULL);

	// DD & GDD
	ddvec_a = init_ddvector(ROW_DIM);
	ddvec_b = init_ddvector(ROW_DIM);
	ddvec_c = init_ddvector(ROW_DIM);

	for(i = 0; i < ROW_DIM; i++)
	{
		// val := sqrt((i + 1));
		ddvec_a->element[i] = td_real::sqrt((int)(i + 1));
		ddvec_b->element[i] = td_real::sqrt((int)(i + 1));
		//ddvec->element[i] = sqrt(qdval);
	}

	print_ddvector(ddvec_a);
	norm2_ddvector(&ddval, ddvec_a);
	printf("DD : ||vec||_2 = ");
	cout.precision(td_real::_ndigits);
	cout << ddval << "\n";

	// GDD start!
	GTDStart();

//	num_blocks = 1;  // #blocks per grid
//	num_blocks = 2;  // #blocks per grid
	num_blocks = 4;  // #blocks per grid
//	num_threads = 16; // #threads per block
	num_threads = 64; // #threads per block

	gddvec_a = init_gtdvector_dev(ROW_DIM);
	gddvec_b = init_gtdvector_dev(ROW_DIM);
	gddvec_c = init_gtdvector_dev(ROW_DIM);
	cudaMalloc((void **)&ptr_gddval_dev, sizeof(gtd_real));

	// gddvec := ddvec
	subst_gtdvector_dev_tdvec(gddvec_a, ddvec_a);
	subst_gtdvector_dev_tdvec(gddvec_b, ddvec_b);
	//subst_tdvector_gtdvec_dev(ddvec_c, gddvec_c);

	// ----------
	// c := a + b
	// ----------
	// DD
	printf(" DD: c := a + b\n");
	add_ddvector(ddvec_c, ddvec_a, ddvec_b);
	print_ddvector(ddvec_c);

	// GDD
	printf("GDD: c := a + b\n");
	add_gtdvector_dev(gddvec_c, gddvec_a, gddvec_b, num_blocks, num_threads);
	print_gtdvector_dev(gddvec_c);

	// ----------
	// c := a - b
	// ----------
	// DD
	printf(" DD: c := a - b\n");
	sub_ddvector(ddvec_c, ddvec_a, ddvec_b);
	print_ddvector(ddvec_c);

	// GDD
	printf("GDD: c := a - b\n");
	sub_gtdvector_dev(gddvec_c, gddvec_a, gddvec_b, num_blocks, num_threads);
	print_gtdvector_dev(gddvec_c);

	// ----------
	// c := val * a
	// ----------
	// DD
	ddval = (td_real)2.0;
	printf(" DD: c := val * a\n");
	cmul_ddvector(ddvec_c, ddval, ddvec_a);
	print_ddvector(ddvec_c);

	// GDD
	td2gtd(&gddval, &ddval);
	printf("GDD: c := val * a\n");
	cmul_gtdvector_dev(gddvec_c, gddval, gddvec_a, num_blocks, num_threads);
	print_gtdvector_dev(gddvec_c);

	// ----------
	// (a, b)
	// ----------
	// DD
	ip_ddvector(&ddval, ddvec_a, ddvec_b);
	printf(" DD : (a, b) = ");
	cout.precision(td_real::_ndigits);
	cout << ddval << "\n";

	// GDD
	ip_gtdvector_dev(ptr_gddval_dev, gddvec_a, gddvec_b, num_blocks, num_threads);
	//ip_gtdvector_dev(ptr_gddval_dev, gddvec_a, gddvec_b, 1, num_threads);
	gtd2td_dev(&ddval, ptr_gddval_dev);
	printf("GDD : (a, b) = ");
	cout.precision(td_real::_ndigits);
	cout << ddval << "\n";

	// ----------
	// ||a||_2
	// ----------
	// DD
	norm2_ddvector(&ddval, ddvec_a);
	printf(" DD : ||a||_2 = ");
	cout.precision(td_real::_ndigits);
	cout << ddval << "\n";

	// GDD
	norm2_gtdvector_dev(ptr_gddval_dev, gddvec_a, num_blocks, num_threads);
	//norm2_gtdvector_dev(ptr_gddval_dev, gddvec_a, 1, num_threads);
	gtd2td_dev(&ddval, ptr_gddval_dev);
	printf("GDD : ||a||_2 = ");
	cout.precision(td_real::_ndigits);
	cout << ddval << "\n";

/* Matrix */
	// DD & GDD
	ddmat_a = init_ddmatrix(ROW_DIM, COL_DIM);
	ddmat_b = init_ddmatrix(ROW_DIM, COL_DIM);
	ddmat_c = init_ddmatrix(ROW_DIM, COL_DIM);

	gddmat_a = init_gtdmatrix_dev(ROW_DIM, COL_DIM);
	gddmat_b = init_gtdmatrix_dev(ROW_DIM, COL_DIM);
	gddmat_c = init_gtdmatrix_dev(ROW_DIM, COL_DIM);

	for(i = 0; i < ROW_DIM; i++)
	{
		for(j = 0; j < COL_DIM; j++)
		{
			// val := sqrt((i + 1));
			ddmat_a->element[i * COL_DIM + j] = td_real::sqrt((int)(i + j + 1));
			ddmat_b->element[i * COL_DIM + j] = td_real::sqrt((int)(i + j + 1));
		}
	}
	subst_gtdmatrix_dev_tdmat(gddmat_a, ddmat_a);
	subst_gtdmatrix_dev_tdmat(gddmat_b, ddmat_b);

	// Print gtdmatrix
	printf("ddmat_a:\n");
	normf_ddmatrix(&ddval, ddmat_a);
	cout.precision(td_real::_ndigits);
	cout << ddval << "\n";

	printf("gddmat_a:\n");
	//print_gtdmatrix_dev(gddmat_a);
	normf_gtdmatrix_dev(ptr_gddval_dev, gddmat_a, 1, num_threads);
	gtd2td_dev(&ddval, ptr_gddval_dev);
	cout.precision(td_real::_ndigits);
	cout << ddval << "\n";

	printf("ddmat_b:\n");
	normf_ddmatrix(&ddval, ddmat_b);
	cout.precision(td_real::_ndigits);
	cout << ddval << "\n";

	printf("gddmat_b:\n");
	//print_gtdmatrix_dev(gddmat_b);
	normf_gtdmatrix_dev(ptr_gddval_dev, gddmat_b, num_blocks, num_threads);
	gtd2td_dev(&ddval, ptr_gddval_dev);
	cout.precision(td_real::_ndigits);
	cout << ddval << "\n";

	// ----------
	// C := A * B
	// ----------
	// DD
	mul_ddmatrix(ddmat_c, ddmat_a, ddmat_b);
	printf("DD : || A * B ||_F:\n");
	normf_ddmatrix(&ddval, ddmat_c);
	cout.precision(td_real::_ndigits);
	cout << ddval << "\n";

	// GDD
	mul_gtdmatrix_dev(gddmat_c, gddmat_a, gddmat_b, num_blocks, num_threads);
	printf("GDD: || A * B ||_F:\n");
	normf_gtdmatrix_dev(ptr_gddval_dev, gddmat_c, num_blocks, num_threads);
	gtd2td_dev(&ddval, ptr_gddval_dev);
	cout.precision(td_real::_ndigits);
	cout << ddval << "\n";

/* Free! */
	cudaFree(ptr_gddval_dev);
	free_gtdvector_dev(gddvec_a);
	free_gtdvector_dev(gddvec_b);
	free_gtdvector_dev(gddvec_c);

	free_gtdmatrix_dev(gddmat_a);
	free_gtdmatrix_dev(gddmat_b);
	free_gtdmatrix_dev(gddmat_c);

	// GDD end!
	GTDEnd();

	// Free TDVectors
	free_ddvector(ddvec_a);
	free_ddvector(ddvec_b);
	free_ddvector(ddvec_c);

	free_ddmatrix(ddmat_a);
	free_ddmatrix(ddmat_b);
	free_ddmatrix(ddmat_c);

	return 0;
}
#endif // DEBUG
