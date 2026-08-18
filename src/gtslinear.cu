/********************************************************************************/
/* gtslinear.cu: Triple-double and Quadruple precision                          */
/*               Linear Computation Library with GQS and CUDA                   */
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

#include "gtslinear.h" // GTSVector, GTSMatrix
	/* Pull only declarations here — gdslinear.cu owns the gqs.cu /
	 * common_s.cu device + host definitions for the whole library.
	 * Cross-TU resolution of __device__ symbols relies on -rdc=true
	 * + nvcc -dlink (see src/cuda/Makefile.in).  Same pattern as
	 * gqslinear.cu. */
	#include "common_s.cuh"
	#include "gqs.cuh"

// initialize gtsvector on CPU(host)
__host__ GTSVector init_gtsvector(long int dim)
{
	long int index;
	GTSVector ret = NULL;
	tsfloat zero = {{0.0f, 0.0f, 0.0f}};

	// callocation
	ret = (gtsvector *)malloc(sizeof(gtsvector));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one GTSVector\n");
		return ret;
	}

	ret->dim = dim;

	ret->element = (gts_real *)calloc(dim, sizeof(gts_real));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GTSVector(dim = %ld)\n", dim);
		cudaFree(ret);
		return NULL;
	}

	// set all zeros 
	for(index = 0; index < dim; index++)
		ts2gts(&ret->element[index], &zero);
		//ret->element[index] = (gts_real)0.0;

	return ret;
}

// free gtsvector on CPU(HOST)
__host__ void free_gtsvector(GTSVector vec)
{
	free(vec->element);
	free(vec);
}

// initialize gtsvector
__host__ GTSVector init_gtsvector_dev(long int dim)
{
	long int index;
	GTSVector ret = NULL;
	tsfloat zero = {{0.0f, 0.0f, 0.0f}};

	// allocation
	ret = (gtsvector *)malloc(sizeof(gtsvector));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one TSVector\n");
		return ret;
	}

	ret->element = NULL;
	ret->dim = dim;

	//ret->element = (tsfloat *)calloc(dim, sizeof(tsfloat));
	cudaMalloc((void **)&(ret->element), (size_t)(dim * sizeof(gts_real)));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GTSVector(dim = %ld)\n", dim);
		cudaFree(ret);
		return NULL;
	}

	// set all zeros 
	for(index = 0; index < dim; index++)
		ts2gts_dev(&(ret->element[index]), &zero);

	return ret;
}

// free dsvector
__host__ void free_gtsvector_dev(GTSVector vec)
{
	cudaFree(vec->element);
	free(vec);
}

// copy TSVector to GTSVector on GPU
// gdsvec on GPU := dsvec
__host__ void subst_gtsvector_dev_tsvec(GTSVector gdsvec_dev, TSVector dsvec)
{
	long int i, dim;

	dim = gdsvec_dev->dim;

	for(i = 0; i < dim; i++)
	{
		/* TSVector::element is SoA: float *element[TSSIZE=3] (hi=[0],
		 * mid=[1], lo=[2]).  Materialize a 3-limb tsfloat to feed ts2gts_dev.
		 * BUG FIX: previously copied only val[0],val[1] -> 3rd limb dropped
		 * -> GPU TS computation degenerated to DS precision (~1e-14 instead of ~1e-21). */
		tsfloat tmp_ds;
		tmp_ds.val[0] = dsvec->element[0][i];
		tmp_ds.val[1] = dsvec->element[1][i];
		tmp_ds.val[2] = dsvec->element[2][i];
		ts2gts_dev(&(gdsvec_dev->element[i]), &tmp_ds);
	}
}

// copy GTSVector on GPU to TSVector
// dsvec := gdsvec_dev on GPU
__host__ void subst_tsvector_gtsvec_dev(TSVector dsvec, GTSVector gdsvec_dev)
{
	long int i, dim;

	dim = dsvec->dim;

	for(i = 0; i < dim; i++)
	{
		/* SoA layout (TSSIZE=3): copy ALL 3 limbs.  See subst_gtsvector_dev_tsvec. */
		tsfloat tmp_ds;
		gts2ts_dev(&tmp_ds, &(gdsvec_dev->element[i]));
		dsvec->element[0][i] = tmp_ds.val[0];
		dsvec->element[1][i] = tmp_ds.val[1];
		dsvec->element[2][i] = tmp_ds.val[2];
	}
}

// print dsvector
__host__ void print_gtsvector_dev(GTSVector dev_vec)
{
	long int index, dim;
	GTSVector host_vec;

	dim = dev_vec->dim;

	host_vec = init_gtsvector(dim);

	cudaMemcpy((void *)(host_vec->element), (void *)(dev_vec->element), size_t(sizeof(gts_real) * dim), cudaMemcpyDeviceToHost);

	for(index = 0; index < dim; index++)
	{
		printf("%4ld: ", index);

		tsfloat _tsv = gts_get_ts(host_vec->element[index]);
		std::cout << _tsv.val[0] << " + " << _tsv.val[1] << " + " << _tsv.val[2] << "\n";
	}

	free(host_vec);
}

/*************************************************/
/* Vector Calculations for TSVector               */
/*
void add_gtsvector_dev(GTSVector c_dev, GTSVector a_dev, GTSVector b_dev)
void add2_gtsvector_dev(GTSVector c_dev, GTSVector a_dev)
void sub_gtsvector_dev(GTSVector c_dev, GTSVector a_dev, GTSVector b_dev)
void sub2_gtsvector_dev(GTSVector c_dev, GTSVector a_dev)
void cmul_gtsvector_dev(GTSVector c_dev, gts_real val_dev, GTSVector a_dev)
void cmul2_gtsvector_dev(GTSVector c_dev, gts_real val_dev)
void add_cmul_gtsvector_dev(GTSVector c_dev, GTSVector a_dev, gts_real val_dev, GTSVector b_dev)
double ip_gtsvector_dev(GTSVector a, GTSVector b_dev)
double norm1_gtsvector_dev(GTSVector a_dev)
double norm2_gtsvector_dev(GTSVector a_dev)
double normi_gtsvector_dev(GTSVector a_dev)
void subst_gtsvector_dev(GTSVector c_dev, GTSVector a_dev)
*/
/*************************************************/
/* c = a + b */
__global__ void _bncu_add_gtsvector(gts_real *c_dev_element, gts_real *a_dev_element, gts_real *b_dev_element, long int dim)
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

__host__ void add_gtsvector_dev(GTSVector c_dev, GTSVector a_dev, GTSVector b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if((a_dev->dim != b_dev->dim) || (c_dev->dim != a_dev->dim) || (c_dev->dim != b_dev->dim))
	{
		fprintf(stderr, "ERROR: add_gtsvector_dev\n");
		return;
	}

/*	for(i = 0; i < c->dim; i++)
	{
		c->element[i] = a->element[i] + b->element[i];
	}
*/
	_bncu_add_gtsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, c_dev->dim);
}


/* c = a - b */
__global__ void _bncu_sub_gtsvector(gts_real *c_dev_element, gts_real *a_dev_element, gts_real *b_dev_element, long int dim)
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

__host__ void sub_gtsvector_dev(GTSVector c_dev, GTSVector a_dev, GTSVector b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if((a_dev->dim != b_dev->dim) || (c_dev->dim != a_dev->dim) || (c_dev->dim != b_dev->dim))
	{
		fprintf(stderr, "ERROR: sub_gtsvector_dev\n");
		return;
	}

/*	for(i = 0; i < c->dim; i++)
	{
		c->element[i] = a->element[i] - b->element[i];
	}
*/
	_bncu_sub_gtsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, c_dev->dim);
}

// cmul vector (val, core)
// dev_ret := dev_val * dev_a on GPU
__global__ void _bncu_cmul_gtsvector(gts_real *dev_ret_element, gts_real val, gts_real *dev_a_element, long int dim)
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
__host__ void cmul_gtsvector_dev(GTSVector c_dev, gts_real val, GTSVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if(c_dev->dim != a_dev->dim)
	{
		fprintf(stderr, "ERROR: cmul_gtsvector_dev\n");
		return;
	}

/*	for(i = 0; i < c_dev->dim; i++)
	{
		c->element[i] = val * a->element[i];
	}
*/

	_bncu_cmul_gtsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, val, a_dev->element, c_dev->dim);

}

// subst vector (val, core)
// dev_ret := dev_vec on GPU
__global__ void _bncu_subst_gtsvector(gts_real *dev_ret_element, gts_real *dev_vec_element, long int dim)
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
__host__ void subst_gtsvector_dev(GTSVector ret_dev, GTSVector vec_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if(ret_dev->dim != vec_dev->dim)
	{
		fprintf(stderr, "ERROR: subst_gtsvector_dev\n");
		return;
	}

	_bncu_subst_gtsvector<<< num_blocks_per_grid, num_threads_per_block >>>(ret_dev->element, vec_dev->element, ret_dev->dim);

}

// set0 
// dev_ret := 0 on GPU
__global__ void _bncu_set0_gtsvector(gts_real *dev_ret_element, long int dim)
{
	int index;

	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim)
	{
		SET0_GTS(dev_ret_element[index]);
		index += blockDim.x * gridDim.x;
	}

	__syncthreads();

}

/* c := 0 */
__host__ void set0_gtsvector_dev(GTSVector ret_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	_bncu_set0_gtsvector<<< num_blocks_per_grid, num_threads_per_block >>>(ret_dev->element, ret_dev->dim);
}

// ret[i] := a_vec[i] * b_vec[i]
__global__ void _bncu_mul_gtsvector(gts_real *ret_dev_element, gts_real *a_dev_element, gts_real *b_dev_element, long int dim)
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
__global__ void _bncu_sqr_gtsvector(gts_real *ret_dev_element, gts_real *vec_dev_element, long int dim)
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
__global__ void _bncu_sqrt_gtsvector(gts_real *ret_dev_element, gts_real *vec_dev_element, long int dim)
{
	int index;

	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim)
	{
		ret_dev_element[index] = (gts_real)sqrt((gts_real)vec_dev_element[index]);
		index += blockDim.x * gridDim.x;
	}

	__syncthreads();
}

// ret[i] := abs(vec[i])
__global__ void _bncu_abs_gtsvector(gts_real *ret_dev_element, gts_real *vec_dev_element, long int dim)
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
__global__ void _bncu_add_reduct_gtsvector(gts_real *ret_dev, gts_real *vec_dev_element, long int dim)
{
	int index, cache_index;
	__shared__ gts_real cache[MAX_NUM_THREADS_PER_BLOCK];

	// Inside Block

	index = threadIdx.x + blockIdx.x * blockDim.x;
	cache_index = threadIdx.x;

	SET0_GTS(cache[cache_index]);

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
__global__ void _bncu_max_reduct_gtsvector(gts_real *ret_dev, gts_real *vec_dev_element, long int dim)
{
	int index, cache_index;
	__shared__ gts_real cache[MAX_NUM_THREADS_PER_BLOCK];

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
__host__ void ip_gtsvector_dev(gts_real *ret_dev, GTSVector a_dev, GTSVector b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gts_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim != b_dev->dim)
	{
		fprintf(stderr, "ERROR: ip_gtsvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gts_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gts_real) * num_blocks_per_grid));

	// c[i] := a[i] * b[i]
	_bncu_mul_gtsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, b_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	//block_dim = 1;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gtsvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gts_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret_dev := block_cache_dev[0]
	cudaMemcpy((void *)ret_dev, (void *)&(block_cache_dev[0]), sizeof(gts_real), cudaMemcpyDeviceToDevice);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// norm2
__host__ void norm2_gtsvector_dev(gts_real *ret_dev, GTSVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gts_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim <= 0)
	{
		fprintf(stderr, "ERROR: norm2_gtsvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gts_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gts_real) * num_blocks_per_grid));

	// c[i] := a[i]^2
	_bncu_sqr_gtsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gtsvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gts_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret := sqrt(ret);
	_bncu_sqrt_gtsvector<<<1, 1>>>(ret_dev, block_cache_dev, 1);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// norm1
__host__ void norm1_gtsvector_dev(gts_real *ret_dev, GTSVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gts_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim <= 0)
	{
		fprintf(stderr, "ERROR: norm1_gtsvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gts_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gts_real) * num_blocks_per_grid));

	// c[i] := abs(a[i])
	_bncu_abs_gtsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gtsvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gts_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret_dev := block_cache_dev[0]
	cudaMemcpy((void *)ret_dev, (void *)&(block_cache_dev[0]), sizeof(gts_real), cudaMemcpyDeviceToDevice);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// norm_inf
__host__ void normi_gtsvector_dev(gts_real *ret_dev, GTSVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gts_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim <= 0)
	{
		fprintf(stderr, "ERROR: normi_gtsvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gts_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gts_real) * num_blocks_per_grid));

	// c[i] := abs(a[i])
	_bncu_abs_gtsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_max_reduct_gtsvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gts_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret_dev := block_cache_dev[0]
	cudaMemcpy((void *)ret_dev, (void *)&(block_cache_dev[0]), sizeof(gts_real), cudaMemcpyDeviceToDevice);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// GDS matrix

// set a zero matrix
__host__ void set0_gtsmatrix(GTSMatrix mat, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i, j;
	tsfloat dszero = {{0.0f, 0.0f, 0.0f}};
	gts_real zero;

	ts2gts(&zero, &dszero);

	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
			mat->element[i * mat->col_dim + j] = zero;
	}
}

// initialize dsvector
GTSMatrix init_gtsmatrix(long int row_dim, long int col_dim)
{
	long int row_index, col_index;
	GTSMatrix ret = NULL;
	tsfloat dszero = {{0.0f, 0.0f, 0.0f}};
	gts_real zero;

	ts2gts(&zero, &dszero);

	// callocation
	ret = (gtsmatrix *)malloc(sizeof(gtsmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one GTSMatrix\n");
		return ret;
	}

	ret->row_dim = row_dim;
	ret->col_dim = col_dim;

	ret->element = (gts_real *)calloc(row_dim * col_dim, sizeof(gts_real));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GTSMatrix(row_dim, col_dim = %ld, %ld)\n", row_dim, col_dim);
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

// free dsvector
void free_gtsmatrix(GTSMatrix mat)
{
	free(mat->element);
	free(mat);
}

// initialize dsvector
GTSMatrix init_gtsmatrix_dev(long int row_dim, long int col_dim)
{
	long int row_index, col_index;
	GTSMatrix ret = NULL;
	tsfloat dszero = {{0.0f, 0.0f, 0.0f}};
	gts_real zero;

	ts2gts(&zero, &dszero);

	// callocation
	ret = (gtsmatrix *)malloc(sizeof(gtsmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one GTSMatrix\n");
		return ret;
	}

	ret->row_dim = row_dim;
	ret->col_dim = col_dim;

	ret->element = (gts_real *)calloc(row_dim * col_dim, sizeof(gts_real));
	cudaMalloc((void **)&(ret->element), (size_t)(row_dim * col_dim * sizeof(gts_real)));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GTSMatrix(row_dim, col_dim = %ld, %ld)\n", row_dim, col_dim);
		free(ret);
		return NULL;
	}

	// set all zeros 
	for(row_index = 0; row_index < row_dim; row_index++)
	{
		for(col_index = 0; col_index < col_dim; col_index++)
			gts2gts_dev(&(ret->element[row_index * col_dim + col_index]), &zero);
	}

	return ret;
}

// free dsvector
void free_gtsmatrix_dev(GTSMatrix mat)
{
	cudaFree(mat->element);
	free(mat);
}

// print dsvector
void print_gtsmatrix_dev(GTSMatrix mat)
{
	long int row_index, col_index;
	tsfloat dsval;

	for(row_index = 0; row_index < mat->row_dim; row_index++)
	{
		for(col_index = 0; col_index < mat->col_dim; col_index++)
		{
			printf("%ld, %ld: ", row_index, col_index);
			gts2ts_dev(&dsval, &(mat->element[row_index * mat->col_dim + col_index]));
			std::cout << dsval.val[0] << " + " << dsval.val[1] << " + " << dsval.val[2] << "\n";
		}
	}
}


__global__ void _bncu_mul_gtsmatrix(gts_real *ret_dev_element, long int row_dim, long int col_dim, long int mid_dim, gts_real *a_dev_element, gts_real *b_dev_element)
{
	long int i, j, k;
	gts_real tmp, tmp1;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	//for(i = 0; i < row_dim; i++)
	while(i < row_dim)
	{
		for(j = 0; j < col_dim; j++)
		{
			SET0_GTS(tmp1); // = (gts_real)0.0;
			SET0_GTS(tmp); // = (gts_real)0.0;
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
void mul_gtsmatrix_dev(GTSMatrix ret_dev, GTSMatrix a_dev, GTSMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long row_dim, col_dim, mid_dim;

	/* dimension check */
	if((ret_dev->row_dim != a_dev->row_dim) || (ret_dev->col_dim != b_dev->col_dim) || (a_dev->col_dim != b_dev->row_dim))
	{
		fprintf(stderr, "ERROR: mul_gtsmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret_dev->row_dim, ret_dev->col_dim, a_dev->row_dim, a_dev->col_dim, b_dev->row_dim, b_dev->col_dim);
		return;
	}

	row_dim = ret_dev->row_dim;
	col_dim = ret_dev->col_dim;
	mid_dim = a_dev->col_dim;

	_bncu_mul_gtsmatrix<<<num_blocks_per_grid, num_threads_per_block>>>(ret_dev->element, row_dim, col_dim, mid_dim, a_dev->element, b_dev->element);

}

// copy TSMatrix to GTSMatrix on GPU
// gdsvec on GPU := dsvec
__host__ void subst_gtsmatrix_dev_tsmat(GTSMatrix gdsmat_dev, TSMatrix dsmat)
{
	long int i, j, row_dim, col_dim;

	row_dim = gdsmat_dev->row_dim;
	col_dim = gdsmat_dev->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			/* TSMatrix::element is SoA: float *element[TSSIZE=3].
			 * Materialize a 3-limb tsfloat for ts2gts_dev.
			 * BUG FIX: previously copied only val[0],val[1] -> 3rd limb dropped
			 * -> GPU TS matvec computed at DS precision (~1e-14 instead of ~1e-21). */
			tsfloat tmp_ds;
			long int idx = i * col_dim + j;
			tmp_ds.val[0] = dsmat->element[0][idx];
			tmp_ds.val[1] = dsmat->element[1][idx];
			tmp_ds.val[2] = dsmat->element[2][idx];
			ts2gts_dev(&(gdsmat_dev->element[idx]), &tmp_ds);
		}
	}
}

// copy GTSMatrix on GPU to TSMatrix
// dsvec := gdsmat_dev on GPU
__host__ void subst_tsmatrix_gtsmat_dev(TSMatrix dsmat, GTSMatrix gdsmat_dev)
{
	long int i, j, row_dim, col_dim;

	row_dim = gdsmat_dev->row_dim;
	col_dim = gdsmat_dev->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			/* SoA layout (TSSIZE=3): copy ALL 3 limbs.  See subst_gtsmatrix_dev_tsmat. */
			tsfloat tmp_ds;
			long int idx = i * col_dim + j;
			gts2ts_dev(&tmp_ds, &(gdsmat_dev->element[idx]));
			dsmat->element[0][idx] = tmp_ds.val[0];
			dsmat->element[1][idx] = tmp_ds.val[1];
			dsmat->element[2][idx] = tmp_ds.val[2];
		}
	}
}

// Frobenius norm
__host__ void normf_gtsmatrix_dev(gts_real *ret_dev, GTSMatrix mat_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, tmp_vec_dev_dim, total_dim;
//	gts_real tmp, tmp1;
	gts_real *block_cache_dev; // partial sum per block
	gts_real *tmp_vec_dev; // square mul of mat

	total_dim = mat_dev->row_dim * mat_dev->col_dim; // total dimension as one vector

	// initialize block_cache_dev, tmp_vec_dev
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gts_real) * num_blocks_per_grid));
	cudaMalloc((void **)&tmp_vec_dev, (size_t)(sizeof(gts_real) * total_dim));

	// mat[i][j]^2
	//_bncu_mul_gtsvector<<<num_blocks_per_grid, num_threads_per_block>>>(tmp_vec_dev, mat_dev->element, mat_dev->element, total_dim);
	_bncu_sqr_gtsvector<<<num_blocks_per_grid, num_threads_per_block>>>(tmp_vec_dev, mat_dev->element, total_dim);

	// reduction to add
	block_dim = num_blocks_per_grid;
//	block_dim = 1;
	thread_dim = num_threads_per_block;
	tmp_vec_dev_dim = total_dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gtsvector<<<block_dim, thread_dim>>>(block_cache_dev, tmp_vec_dev, tmp_vec_dev_dim);

		if(block_dim <= 1)
			break;

		tmp_vec_dev_dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)tmp_vec_dev, (void *)block_cache_dev, (size_t)(sizeof(gts_real) * tmp_vec_dev_dim), cudaMemcpyDeviceToDevice);

	}

	// ret := sqrt(ret);
	_bncu_sqrt_gtsvector<<<1, 1>>>(ret_dev, block_cache_dev, 1);

	// free
	cudaFree(block_cache_dev);
	cudaFree(tmp_vec_dev);

/*
	SET0_GTS(*ret); // = (gts_real)0.0;
	SET0_GTS(tmp); // = (gts_real)(int)0;
	SET0_GTS(tmp1); // = (gts_real)(int)0;

	dim = mat->row_dim * mat->col_dim;

	for(i = 0; i < dim ; i++)
	{
		//tmp += mat->element[i] * mat->element[i];
		//tmp += gts_real::mul((gts_real)(mat->element[i]), (gts_real)(mat->element[i]));
		tmp1 = (gts_real)(mat->element[i]) * (gts_real)(mat->element[i]);
		tmp = tmp + tmp1;
	}

	//*ret = sqrt(tmp);
	tmp = (gts_real)sqrt(tmp);
	*ret = tmp;
*/
}

/*************************************************/
/* Matrix Caluculations for GTSMatrix            */
/*
void normf_gtsmatrix(double ret[TSSIZE], GTSMatrix mat)
void norm1_gtsmatrix(double ret[TSSIZE], GTSMatrix mat)
void normi_gtsmatrix(double ret[TSSIZE], GTSMatrix mat)
void add_gtsmatrix(GTSMatrix c, GTSMatrix a, GTSMatrix b);
void sub_gtsmatrix(GTSMatrix c, GTSMatrix a, GTSMatrix b);
void mul_gtsmatrix(GTSMatrix c, GTSMatrix a, GTSMatrix b);
void mul_gtsmatrix_dsvec(TSVector v, GTSMatrix a, TSVector vb)
void mul_gtsmatrixt_dsvec(TSVector v, GTSMatrix a, TSVector vb)
void transpose_gtsmatrix(GTSMatrix c, GTSMatrix a);
void inv_gtsmatrix(GTSMatrix a);
void subst_mpfmatrux(GTSMatrix c, GTSMatrix a);
*/
/*************************************************/

#if 0
/* Infinity Norm of Matrix */
void normi_gtsmatrix(gts_real *ret, GTSMatrix mat, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i, j;
	gts_real sum;

	SET0_GTS(*ret); // = 0.0;

	for(i = 0; i < mat->row_dim; i++)
	{
		SET0_GTS(sum); //  = 0.0;
		for(j = 0; j < mat->col_dim; j++)
			sum = sum + abs(mat->element[i * mat->row_dim + j]);

		if(*ret < sum)
			*ret = sum;

	}

	return;
}

/* 1 Norm of Matrix */
void norm1_gtsmatrix(gts_real *ret, GTSMatrix mat, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i, j;
	gts_real sum;

	SET0_GTS(*ret); // = 0.0;

	for(j = 0; j < mat->col_dim; j++)
	{
		SET0_GTS(sum); // = 0.0;
		for(i = 0; i < mat->row_dim; i++)
			sum = sum + abs(mat->element[i * mat->col_dim + j]);

		if(*ret < sum)
			*ret = sum;

	}

	return;
}
#endif // 0

/* c := a + b */
void add_gtsmatrix_dev(GTSMatrix c_dev, GTSMatrix a_dev, GTSMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int row_dim, col_dim; //, index;

	/* check row_dim */
	if((a_dev->row_dim != b_dev->row_dim) || (b_dev->row_dim != c_dev->row_dim) || (c_dev->row_dim != a_dev->row_dim))
	{
		fprintf(stderr, "ERROR: add_gtsmatrix_dev\n");
		return;
	}
	row_dim = c_dev->row_dim;

	/* check col_dim */
	if((a_dev->col_dim != b_dev->col_dim) || (b_dev->col_dim != c_dev->col_dim) || (c_dev->col_dim != a_dev->col_dim))
	{
		fprintf(stderr, "ERROR: add_gtsmatrix_dev\n");
		return;
	}
	col_dim = c_dev->col_dim;

	_bncu_add_gtsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, row_dim * col_dim);

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
void sub_gtsmatrix_dev(GTSMatrix c_dev, GTSMatrix a_dev, GTSMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int row_dim, col_dim; //, index;

	/* check row_dim */
	if((a_dev->row_dim != b_dev->row_dim) || (b_dev->row_dim != c_dev->row_dim) || (c_dev->row_dim != a_dev->row_dim))
	{
		fprintf(stderr, "ERROR: sub_gtsmatrix_dev\n");
		return;
	}
	row_dim = c_dev->row_dim;

	/* check col_dim */
	if((a_dev->col_dim != b_dev->col_dim) || (b_dev->col_dim != c_dev->col_dim) || (c_dev->col_dim != a_dev->col_dim))
	{
		fprintf(stderr, "ERROR: sub_gtsmatrix_dev\n");
		return;
	}
	col_dim = c_dev->col_dim;

	_bncu_sub_gtsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, row_dim * col_dim);

}

/* c := sc * a */
void cmul_gtsmatrix_dev(GTSMatrix c_dev, gts_real sc, GTSMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int row_dim, col_dim; //, index;

	/* check row_dim */
	if(a_dev->row_dim != c_dev->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_gtsmatrix_dev\n");
		return;
	}
	row_dim = c_dev->row_dim;

	/* check col_dim */
	if(a_dev->col_dim != c_dev->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_gtsmatrix_dev\n");
		return;
	}
	col_dim = c_dev->col_dim;

	_bncu_cmul_gtsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, sc, a_dev->element, row_dim * col_dim);


}

/* c = a^T */
__global__ void _bncu_transpose_gtsmatrix(gts_real *c_dev_element, gts_real *a_dev_element, long int row_dim, long int col_dim)
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

void transpose_gtsmatrix_dev(GTSMatrix c_dev, GTSMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	//long int i, j;

	/* Check Dimensions */
	if((c_dev->row_dim != a_dev->col_dim) || (c_dev->col_dim != a_dev->row_dim))
	{
		fprintf(stderr, "ERROR: transpose_gtsmatrix_dev\n");
		return;
	}
	
	_bncu_transpose_gtsmatrix<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, c_dev->row_dim, c_dev->col_dim);

/*	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			c->element[i * c->col_dim + j] = a->element[j * a->col_dim + i];
	}
*/
}

/* c := a */
void subst_gtsmatrix_dev(GTSMatrix c_dev, GTSMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	//long int i, j, index;

	if((c_dev->row_dim != a_dev->row_dim) || (c_dev->col_dim != a_dev->col_dim))
	{
		fprintf(stderr, "ERROR: subst_gtsmatrix_dev\n");
		return;
	}

	_bncu_subst_gtsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, c_dev->row_dim * c_dev->col_dim);

}

/* c := 0 */
void set0_gtsmatrix_dev(GTSMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block)
{
//	long int i, j;

	_bncu_set0_gtsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, c_dev->row_dim * c_dev->col_dim);

}

/* set (i, j)-element on GPU from tsfloat */
void set_gtsmatrix_ij_dev(GTSMatrix mat_dev, long int row_index, long int col_index, tsfloat val)
{
	if((row_index >= 0) && (row_index < mat_dev->row_dim) && (col_index >= 0) && (col_index < mat_dev->col_dim))
		ts2gts_dev(&(mat_dev->element[row_index * mat_dev->col_dim + col_index]), &val);
}

/* c := I */
void setI_gtsmatrix_dev(GTSMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i;
	tsfloat one;

	one.val[0] = 1.0f; one.val[1] = 0.0f; one.val[2] = 0.0f;

	_bncu_set0_gtsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, c_dev->row_dim * c_dev->col_dim);

	for(i = 0; i < c_dev->row_dim; i++)
		set_gtsmatrix_ij_dev(c_dev, i, i, one);
/*
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			SET0_GTS(c->element[i * c->col_dim + j]); // = 0.0;
		if(i < c->col_dim)
			SET1_GTS(c->element[i * c->col_dim + i]); // = 1.0;
	}
*/
}

/* v := a * vb */
__global__ void _bncu_mul_gtsmatrix_gdsvec(gts_real *v_element, gts_real *a_element, gts_real *vb_element, long int row_dim, long int col_dim)
{
	int j, index;
	gts_real tmp;

	index = threadIdx.x + blockIdx.x * blockDim.x;

	//for(i = 0; i < row_dim; i++)
	while(index < row_dim)
	{
		SET0_GTS(tmp); // = 0.0;
		for(j = 0; j < col_dim; j++)
			tmp = tmp + a_element[index * col_dim + j] * vb_element[j];

		v_element[index] = tmp;

		__syncthreads();

		index += blockDim.x * gridDim.x;
	}
}

void mul_gtsmatrix_gdsvec(GTSVector v, GTSMatrix a, GTSVector vb, int num_blocks_per_grid, int num_threads_per_block)
{
	/* Check Dimension */
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: mul_gtsmatrix_dsvec\n");
		return;
	}

	_bncu_mul_gtsmatrix_gdsvec<<<num_blocks_per_grid, num_threads_per_block>>>(v->element, a->element, vb->element, a->row_dim, a->col_dim);

/*	for(i = 0; i < a->row_dim; i++)
	{
		SET0_GTS(tmp); // = 0.0;
		for(j = 0; j < a->col_dim; j++)
			tmp = tmp + a->element[i * a->col_dim + j] * vb->element[j];

		v->element[i] = tmp;
	}
*/
}

/* v := a^T * vb */
__global__ void _bncu_mul_gtsmatrixt_gdsvec(gts_real *v_element, gts_real *a_element, gts_real *vb_element, long int row_dim, long int col_dim)
{
	int j, index;
	gts_real tmp;

	index = threadIdx.x + blockIdx.x * blockDim.x;

	//for(i = 0; i < col_dim; i++)
	while(index < col_dim)
	{
		SET0_GTS(tmp); // = 0.0;
		for(j = 0; j < row_dim; j++)
			tmp = tmp + a_element[j * row_dim + index] * vb_element[j];

		v_element[index] = tmp;

		__syncthreads();

		index += blockDim.x * gridDim.x;
	}
}

__host__ void mul_gtsmatrixt_gdsvec(GTSVector v, GTSMatrix a, GTSVector vb, int num_blocks_per_grid, int num_threads_per_block)
{

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_gtsmatrixt_gdsvec\n");
		return;
	}

	_bncu_mul_gtsmatrixt_gdsvec<<<num_blocks_per_grid, num_threads_per_block>>>(v->element, a->element, vb->element, a->row_dim, a->col_dim);

/*	for(i = 0; i < a->col_dim; i++)
	{
		SET0_GTS(tmp); // = 0.0;
		for(j = 0; j < a->row_dim; j++)
			tmp = tmp + a->element[j * a->row_dim + i] * vb->element[j];

		v->element[i] = tmp;
	}
*/
}

/* -------------------------------------------------------------------
 *  Properly-named aliases for the same-precision matvec operations.
 *  mul_gtsmatrix_gdsvec / mul_gtsmatrixt_gdsvec above are misnamed —
 *  despite "_gdsvec" in the name they take GTSVector (gts_real*) for
 *  ALL arguments and operate at TS precision.  gtslinear.h declares
 *  the proper-named gtsvec variants that the bench programs expect;
 *  these wrappers satisfy the link without breaking _gdsvec callers.
 * ------------------------------------------------------------------- */
__host__ void mul_gtsmatrix_gtsvec(GTSVector v, GTSMatrix a, GTSVector vb,
                                    int num_blocks_per_grid, int num_threads_per_block)
{
	mul_gtsmatrix_gdsvec(v, a, vb, num_blocks_per_grid, num_threads_per_block);
}

__host__ void mul_gtsmatrixt_gtsvec(GTSVector v, GTSMatrix a, GTSVector vb,
                                     int num_blocks_per_grid, int num_threads_per_block)
{
	mul_gtsmatrixt_gdsvec(v, a, vb, num_blocks_per_grid, num_threads_per_block);
}

// Test main function
// NOTE: disabled — uses qd-library API not available for the POD tsfloat type.
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
	TSVector dsvec_a, dsvec_b, dsvec_c;
	TSMatrix dsmat_a, dsmat_b, dsmat_c;
	tsfloat dsval;
	int num_blocks, num_threads;
	GTSVector gdsvec_a, gdsvec_b, gdsvec_c;
	gts_real gdsval, *ptr_gdsval_dev;
	GTSMatrix gdsmat_a, gdsmat_b, gdsmat_c;
	

	// Initialize QD library
	fpu_fix_start(NULL);

	// DD & GDS
	dsvec_a = init_dsvector(ROW_DIM);
	dsvec_b = init_dsvector(ROW_DIM);
	dsvec_c = init_dsvector(ROW_DIM);

	for(i = 0; i < ROW_DIM; i++)
	{
		// val := sqrt((i + 1));
		dsvec_a->element[i] = tsfloat::sqrt((int)(i + 1));
		dsvec_b->element[i] = tsfloat::sqrt((int)(i + 1));
		//dsvec->element[i] = sqrt(qsval);
	}

	print_dsvector(dsvec_a);
	norm2_dsvector(&dsval, dsvec_a);
	printf("DD : ||vec||_2 = ");
	cout.precision(tsfloat::_ndigits);
	cout << dsval << "\n";

	// GDS start!
	GTSStart();

//	num_blocks = 1;  // #blocks per grid
//	num_blocks = 2;  // #blocks per grid
	num_blocks = 4;  // #blocks per grid
//	num_threads = 16; // #threads per block
	num_threads = 64; // #threads per block

	gdsvec_a = init_gtsvector_dev(ROW_DIM);
	gdsvec_b = init_gtsvector_dev(ROW_DIM);
	gdsvec_c = init_gtsvector_dev(ROW_DIM);
	cudaMalloc((void **)&ptr_gdsval_dev, sizeof(gts_real));

	// gdsvec := dsvec
	subst_gtsvector_dev_tsvec(gdsvec_a, dsvec_a);
	subst_gtsvector_dev_tsvec(gdsvec_b, dsvec_b);
	//subst_tsvector_gtsvec_dev(dsvec_c, gdsvec_c);

	// ----------
	// c := a + b
	// ----------
	// DD
	printf(" DD: c := a + b\n");
	add_dsvector(dsvec_c, dsvec_a, dsvec_b);
	print_dsvector(dsvec_c);

	// GDS
	printf("GDS: c := a + b\n");
	add_gtsvector_dev(gdsvec_c, gdsvec_a, gdsvec_b, num_blocks, num_threads);
	print_gtsvector_dev(gdsvec_c);

	// ----------
	// c := a - b
	// ----------
	// DD
	printf(" DD: c := a - b\n");
	sub_dsvector(dsvec_c, dsvec_a, dsvec_b);
	print_dsvector(dsvec_c);

	// GDS
	printf("GDS: c := a - b\n");
	sub_gtsvector_dev(gdsvec_c, gdsvec_a, gdsvec_b, num_blocks, num_threads);
	print_gtsvector_dev(gdsvec_c);

	// ----------
	// c := val * a
	// ----------
	// DD
	dsval = (tsfloat)2.0;
	printf(" DD: c := val * a\n");
	cmul_dsvector(dsvec_c, dsval, dsvec_a);
	print_dsvector(dsvec_c);

	// GDS
	ts2gts(&gdsval, &dsval);
	printf("GDS: c := val * a\n");
	cmul_gtsvector_dev(gdsvec_c, gdsval, gdsvec_a, num_blocks, num_threads);
	print_gtsvector_dev(gdsvec_c);

	// ----------
	// (a, b)
	// ----------
	// DD
	ip_dsvector(&dsval, dsvec_a, dsvec_b);
	printf(" DD : (a, b) = ");
	cout.precision(tsfloat::_ndigits);
	cout << dsval << "\n";

	// GDS
	ip_gtsvector_dev(ptr_gdsval_dev, gdsvec_a, gdsvec_b, num_blocks, num_threads);
	//ip_gtsvector_dev(ptr_gdsval_dev, gdsvec_a, gdsvec_b, 1, num_threads);
	gts2ts_dev(&dsval, ptr_gdsval_dev);
	printf("GDS : (a, b) = ");
	cout.precision(tsfloat::_ndigits);
	cout << dsval << "\n";

	// ----------
	// ||a||_2
	// ----------
	// DD
	norm2_dsvector(&dsval, dsvec_a);
	printf(" DD : ||a||_2 = ");
	cout.precision(tsfloat::_ndigits);
	cout << dsval << "\n";

	// GDS
	norm2_gtsvector_dev(ptr_gdsval_dev, gdsvec_a, num_blocks, num_threads);
	//norm2_gtsvector_dev(ptr_gdsval_dev, gdsvec_a, 1, num_threads);
	gts2ts_dev(&dsval, ptr_gdsval_dev);
	printf("GDS : ||a||_2 = ");
	cout.precision(tsfloat::_ndigits);
	cout << dsval << "\n";

/* Matrix */
	// DD & GDS
	dsmat_a = init_dsmatrix(ROW_DIM, COL_DIM);
	dsmat_b = init_dsmatrix(ROW_DIM, COL_DIM);
	dsmat_c = init_dsmatrix(ROW_DIM, COL_DIM);

	gdsmat_a = init_gtsmatrix_dev(ROW_DIM, COL_DIM);
	gdsmat_b = init_gtsmatrix_dev(ROW_DIM, COL_DIM);
	gdsmat_c = init_gtsmatrix_dev(ROW_DIM, COL_DIM);

	for(i = 0; i < ROW_DIM; i++)
	{
		for(j = 0; j < COL_DIM; j++)
		{
			// val := sqrt((i + 1));
			dsmat_a->element[i * COL_DIM + j] = tsfloat::sqrt((int)(i + j + 1));
			dsmat_b->element[i * COL_DIM + j] = tsfloat::sqrt((int)(i + j + 1));
		}
	}
	subst_gtsmatrix_dev_tsmat(gdsmat_a, dsmat_a);
	subst_gtsmatrix_dev_tsmat(gdsmat_b, dsmat_b);

	// Print gtsmatrix
	printf("dsmat_a:\n");
	normf_dsmatrix(&dsval, dsmat_a);
	cout.precision(tsfloat::_ndigits);
	cout << dsval << "\n";

	printf("gdsmat_a:\n");
	//print_gtsmatrix_dev(gdsmat_a);
	normf_gtsmatrix_dev(ptr_gdsval_dev, gdsmat_a, 1, num_threads);
	gts2ts_dev(&dsval, ptr_gdsval_dev);
	cout.precision(tsfloat::_ndigits);
	cout << dsval << "\n";

	printf("dsmat_b:\n");
	normf_dsmatrix(&dsval, dsmat_b);
	cout.precision(tsfloat::_ndigits);
	cout << dsval << "\n";

	printf("gdsmat_b:\n");
	//print_gtsmatrix_dev(gdsmat_b);
	normf_gtsmatrix_dev(ptr_gdsval_dev, gdsmat_b, num_blocks, num_threads);
	gts2ts_dev(&dsval, ptr_gdsval_dev);
	cout.precision(tsfloat::_ndigits);
	cout << dsval << "\n";

	// ----------
	// C := A * B
	// ----------
	// DD
	mul_dsmatrix(dsmat_c, dsmat_a, dsmat_b);
	printf("DD : || A * B ||_F:\n");
	normf_dsmatrix(&dsval, dsmat_c);
	cout.precision(tsfloat::_ndigits);
	cout << dsval << "\n";

	// GDS
	mul_gtsmatrix_dev(gdsmat_c, gdsmat_a, gdsmat_b, num_blocks, num_threads);
	printf("GDS: || A * B ||_F:\n");
	normf_gtsmatrix_dev(ptr_gdsval_dev, gdsmat_c, num_blocks, num_threads);
	gts2ts_dev(&dsval, ptr_gdsval_dev);
	cout.precision(tsfloat::_ndigits);
	cout << dsval << "\n";

/* Free! */
	cudaFree(ptr_gdsval_dev);
	free_gtsvector_dev(gdsvec_a);
	free_gtsvector_dev(gdsvec_b);
	free_gtsvector_dev(gdsvec_c);

	free_gtsmatrix_dev(gdsmat_a);
	free_gtsmatrix_dev(gdsmat_b);
	free_gtsmatrix_dev(gdsmat_c);

	// GDS end!
	GTSEnd();

	// Free TDVectors
	free_dsvector(dsvec_a);
	free_dsvector(dsvec_b);
	free_dsvector(dsvec_c);

	free_dsmatrix(dsmat_a);
	free_dsmatrix(dsmat_b);
	free_dsmatrix(dsmat_c);

	return 0;
}
#endif // DEBUG
