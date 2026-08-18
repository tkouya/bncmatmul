/********************************************************************************/
/* gdslinear.cu: Double-double and Quadruple precision                          */
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

#include "gdslinear.h" // G[D,Q]SVector, G[D,Q]SMatrix
	#include "common_s.cu" // GDSStart, GDSEnd (single-precision common; the double-precision common.cu is owned by gddlinear.cu)
	#include "gqs.cu"

// initialize gdsvector on CPU(host)
__host__ GDSVector init_gdsvector(long int dim)
{
	long int index;
	GDSVector ret = NULL;
	dsfloat zero = {{0.0f, 0.0f}};

	// callocation
	ret = (gdsvector *)malloc(sizeof(gdsvector));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one GDSVector\n");
		return ret;
	}

	ret->dim = dim;

	ret->element = (gds_real *)calloc(dim, sizeof(gds_real));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GDSVector(dim = %ld)\n", dim);
		cudaFree(ret);
		return NULL;
	}

	// set all zeros 
	for(index = 0; index < dim; index++)
		ds2gds(&ret->element[index], &zero);
		//ret->element[index] = (gds_real)0.0;

	return ret;
}

// free gdsvector on CPU(HOST)
__host__ void free_gdsvector(GDSVector vec)
{
	free(vec->element);
	free(vec);
}

// initialize gdsvector
__host__ GDSVector init_gdsvector_dev(long int dim)
{
	long int index;
	GDSVector ret = NULL;
	dsfloat zero = {{0.0f, 0.0f}};

	// allocation
	ret = (gdsvector *)malloc(sizeof(gdsvector));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one DSVector\n");
		return ret;
	}

	ret->element = NULL;
	ret->dim = dim;

	//ret->element = (dsfloat *)calloc(dim, sizeof(dsfloat));
	cudaMalloc((void **)&(ret->element), (size_t)(dim * sizeof(gds_real)));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GDSVector(dim = %ld)\n", dim);
		cudaFree(ret);
		return NULL;
	}

	// set all zeros 
	for(index = 0; index < dim; index++)
		ds2gds_dev(&(ret->element[index]), &zero);

	return ret;
}

// free dsvector
__host__ void free_gdsvector_dev(GDSVector vec)
{
	cudaFree(vec->element);
	free(vec);
}

// copy DSVector to GDSVector on GPU
// gdsvec on GPU := dsvec
__host__ void subst_gdsvector_dev_dsvec(GDSVector gdsvec_dev, DSVector dsvec)
{
	long int i, dim;

	dim = gdsvec_dev->dim;

	for(i = 0; i < dim; i++)
	{
		/* DSVector::element is SoA: double *element[DSSIZE] (hi=[0],
		 * lo=[1]).  Materialize a temporary dsfloat to feed ds2gds_dev,
		 * which expects an AoS dsfloat*. */
		dsfloat tmp_ds;
		tmp_ds.val[0] = dsvec->element[0][i];
		tmp_ds.val[1] = dsvec->element[1][i];
		ds2gds_dev(&(gdsvec_dev->element[i]), &tmp_ds);
	}
}

// copy GDSVector on GPU to DSVector
// dsvec := gdsvec_dev on GPU
__host__ void subst_dsvector_gdsvec_dev(DSVector dsvec, GDSVector gdsvec_dev)
{
	long int i, dim;

	dim = dsvec->dim;

	for(i = 0; i < dim; i++)
	{
		/* SoA layout: see subst_gdsvector_dev_dsvec above. */
		dsfloat tmp_ds;
		gds2ds_dev(&tmp_ds, &(gdsvec_dev->element[i]));
		dsvec->element[0][i] = tmp_ds.val[0];
		dsvec->element[1][i] = tmp_ds.val[1];
	}
}

// print dsvector
__host__ void print_gdsvector_dev(GDSVector dev_vec)
{
	long int index, dim;
	GDSVector host_vec;

	dim = dev_vec->dim;

	host_vec = init_gdsvector(dim);

	cudaMemcpy((void *)(host_vec->element), (void *)(dev_vec->element), size_t(sizeof(gds_real) * dim), cudaMemcpyDeviceToHost);

	for(index = 0; index < dim; index++)
	{
		printf("%4ld: ", index);

		dsfloat _dsv = gds_get_ds(host_vec->element[index]);
		std::cout << _dsv.val[0] << " + " << _dsv.val[1] << "\n";
	}

	free(host_vec);
}

/*************************************************/
/* Vector Calculations for DSVector               */
/*
void add_gdsvector_dev(GDSVector c_dev, GDSVector a_dev, GDSVector b_dev)
void add2_gdsvector_dev(GDSVector c_dev, GDSVector a_dev)
void sub_gdsvector_dev(GDSVector c_dev, GDSVector a_dev, GDSVector b_dev)
void sub2_gdsvector_dev(GDSVector c_dev, GDSVector a_dev)
void cmul_gdsvector_dev(GDSVector c_dev, gds_real val_dev, GDSVector a_dev)
void cmul2_gdsvector_dev(GDSVector c_dev, gds_real val_dev)
void add_cmul_gdsvector_dev(GDSVector c_dev, GDSVector a_dev, gds_real val_dev, GDSVector b_dev)
double ip_gdsvector_dev(GDSVector a, GDSVector b_dev)
double norm1_gdsvector_dev(GDSVector a_dev)
double norm2_gdsvector_dev(GDSVector a_dev)
double normi_gdsvector_dev(GDSVector a_dev)
void subst_gdsvector_dev(GDSVector c_dev, GDSVector a_dev)
*/
/*************************************************/
/* c = a + b */
__global__ void _bncu_add_gdsvector(gds_real *c_dev_element, gds_real *a_dev_element, gds_real *b_dev_element, long int dim)
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

__host__ void add_gdsvector_dev(GDSVector c_dev, GDSVector a_dev, GDSVector b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if((a_dev->dim != b_dev->dim) || (c_dev->dim != a_dev->dim) || (c_dev->dim != b_dev->dim))
	{
		fprintf(stderr, "ERROR: add_gdsvector_dev\n");
		return;
	}

/*	for(i = 0; i < c->dim; i++)
	{
		c->element[i] = a->element[i] + b->element[i];
	}
*/
	_bncu_add_gdsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, c_dev->dim);
}


/* c = a - b */
__global__ void _bncu_sub_gdsvector(gds_real *c_dev_element, gds_real *a_dev_element, gds_real *b_dev_element, long int dim)
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

__host__ void sub_gdsvector_dev(GDSVector c_dev, GDSVector a_dev, GDSVector b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if((a_dev->dim != b_dev->dim) || (c_dev->dim != a_dev->dim) || (c_dev->dim != b_dev->dim))
	{
		fprintf(stderr, "ERROR: sub_gdsvector_dev\n");
		return;
	}

/*	for(i = 0; i < c->dim; i++)
	{
		c->element[i] = a->element[i] - b->element[i];
	}
*/
	_bncu_sub_gdsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, c_dev->dim);
}

// cmul vector (val, core)
// dev_ret := dev_val * dev_a on GPU
__global__ void _bncu_cmul_gdsvector(gds_real *dev_ret_element, gds_real val, gds_real *dev_a_element, long int dim)
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
__host__ void cmul_gdsvector_dev(GDSVector c_dev, gds_real val, GDSVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if(c_dev->dim != a_dev->dim)
	{
		fprintf(stderr, "ERROR: cmul_gdsvector_dev\n");
		return;
	}

/*	for(i = 0; i < c_dev->dim; i++)
	{
		c->element[i] = val * a->element[i];
	}
*/

	_bncu_cmul_gdsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, val, a_dev->element, c_dev->dim);

}

// subst vector (val, core)
// dev_ret := dev_vec on GPU
__global__ void _bncu_subst_gdsvector(gds_real *dev_ret_element, gds_real *dev_vec_element, long int dim)
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
__host__ void subst_gdsvector_dev(GDSVector ret_dev, GDSVector vec_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	if(ret_dev->dim != vec_dev->dim)
	{
		fprintf(stderr, "ERROR: subst_gdsvector_dev\n");
		return;
	}

	_bncu_subst_gdsvector<<< num_blocks_per_grid, num_threads_per_block >>>(ret_dev->element, vec_dev->element, ret_dev->dim);

}

// set0 
// dev_ret := 0 on GPU
__global__ void _bncu_set0_gdsvector(gds_real *dev_ret_element, long int dim)
{
	int index;

	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim)
	{
		SET0_GDS(dev_ret_element[index]);
		index += blockDim.x * gridDim.x;
	}

	__syncthreads();

}

/* c := 0 */
__host__ void set0_gdsvector_dev(GDSVector ret_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	_bncu_set0_gdsvector<<< num_blocks_per_grid, num_threads_per_block >>>(ret_dev->element, ret_dev->dim);
}

// ret[i] := a_vec[i] * b_vec[i]
__global__ void _bncu_mul_gdsvector(gds_real *ret_dev_element, gds_real *a_dev_element, gds_real *b_dev_element, long int dim)
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
__global__ void _bncu_sqr_gdsvector(gds_real *ret_dev_element, gds_real *vec_dev_element, long int dim)
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
__global__ void _bncu_sqrt_gdsvector(gds_real *ret_dev_element, gds_real *vec_dev_element, long int dim)
{
	int index;

	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim)
	{
		ret_dev_element[index] = (gds_real)sqrt((gds_real)vec_dev_element[index]);
		index += blockDim.x * gridDim.x;
	}

	__syncthreads();
}

// ret[i] := abs(vec[i])
__global__ void _bncu_abs_gdsvector(gds_real *ret_dev_element, gds_real *vec_dev_element, long int dim)
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
__global__ void _bncu_add_reduct_gdsvector(gds_real *ret_dev, gds_real *vec_dev_element, long int dim)
{
	int index, cache_index;
	__shared__ gds_real cache[MAX_NUM_THREADS_PER_BLOCK];

	// Inside Block

	index = threadIdx.x + blockIdx.x * blockDim.x;
	cache_index = threadIdx.x;

	SET0_GDS(cache[cache_index]);

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
__global__ void _bncu_max_reduct_gdsvector(gds_real *ret_dev, gds_real *vec_dev_element, long int dim)
{
	int index, cache_index;
	__shared__ gds_real cache[MAX_NUM_THREADS_PER_BLOCK];

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
__host__ void ip_gdsvector_dev(gds_real *ret_dev, GDSVector a_dev, GDSVector b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gds_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim != b_dev->dim)
	{
		fprintf(stderr, "ERROR: ip_gdsvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gds_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gds_real) * num_blocks_per_grid));

	// c[i] := a[i] * b[i]
	_bncu_mul_gdsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, b_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	//block_dim = 1;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gdsvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gds_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret_dev := block_cache_dev[0]
	cudaMemcpy((void *)ret_dev, (void *)&(block_cache_dev[0]), sizeof(gds_real), cudaMemcpyDeviceToDevice);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// norm2
__host__ void norm2_gdsvector_dev(gds_real *ret_dev, GDSVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gds_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim <= 0)
	{
		fprintf(stderr, "ERROR: norm2_gdsvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gds_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gds_real) * num_blocks_per_grid));

	// c[i] := a[i]^2
	_bncu_sqr_gdsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gdsvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gds_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret := sqrt(ret);
	_bncu_sqrt_gdsvector<<<1, 1>>>(ret_dev, block_cache_dev, 1);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// norm1
__host__ void norm1_gdsvector_dev(gds_real *ret_dev, GDSVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gds_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim <= 0)
	{
		fprintf(stderr, "ERROR: norm1_gdsvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gds_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gds_real) * num_blocks_per_grid));

	// c[i] := abs(a[i])
	_bncu_abs_gdsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gdsvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gds_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret_dev := block_cache_dev[0]
	cudaMemcpy((void *)ret_dev, (void *)&(block_cache_dev[0]), sizeof(gds_real), cudaMemcpyDeviceToDevice);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// norm_inf
__host__ void normi_gdsvector_dev(gds_real *ret_dev, GDSVector a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, dim;
	gds_real *c_dev_element, *block_cache_dev;

	if(a_dev->dim <= 0)
	{
		fprintf(stderr, "ERROR: normi_gdsvector_dev\n");
		return;
	}

	// initialize
	cudaMalloc((void **)&c_dev_element, (size_t)(sizeof(gds_real) * a_dev->dim));
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gds_real) * num_blocks_per_grid));

	// c[i] := abs(a[i])
	_bncu_abs_gdsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev_element, a_dev->element, a_dev->dim);

	// reduction
	block_dim = num_blocks_per_grid;
	thread_dim = num_threads_per_block;
	dim = a_dev->dim;

	while(block_dim >= 1)
	{
		_bncu_max_reduct_gdsvector<<< block_dim, thread_dim >>>(block_cache_dev, c_dev_element, dim);

		if(block_dim <= 1)
			break;

		// c_dev_element ;= block_cache
		dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)c_dev_element, (void *)block_cache_dev, (size_t)(sizeof(gds_real) * dim), cudaMemcpyDeviceToDevice);

	}		

	// ret_dev := block_cache_dev[0]
	cudaMemcpy((void *)ret_dev, (void *)&(block_cache_dev[0]), sizeof(gds_real), cudaMemcpyDeviceToDevice);

	// clear
	cudaFree(c_dev_element);
	cudaFree(block_cache_dev);

	return;
}

// GDS matrix

// set a zero matrix
__host__ void set0_gdsmatrix(GDSMatrix mat, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i, j;
	dsfloat dszero = {{0.0f, 0.0f}};
	gds_real zero;

	ds2gds(&zero, &dszero);

	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
			mat->element[i * mat->col_dim + j] = zero;
	}
}

// initialize dsvector
GDSMatrix init_gdsmatrix(long int row_dim, long int col_dim)
{
	long int row_index, col_index;
	GDSMatrix ret = NULL;
	dsfloat dszero = {{0.0f, 0.0f}};
	gds_real zero;

	ds2gds(&zero, &dszero);

	// callocation
	ret = (gdsmatrix *)malloc(sizeof(gdsmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one GDSMatrix\n");
		return ret;
	}

	ret->row_dim = row_dim;
	ret->col_dim = col_dim;

	ret->element = (gds_real *)calloc(row_dim * col_dim, sizeof(gds_real));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GDSMatrix(row_dim, col_dim = %ld, %ld)\n", row_dim, col_dim);
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
void free_gdsmatrix(GDSMatrix mat)
{
	free(mat->element);
	free(mat);
}

// initialize dsvector
GDSMatrix init_gdsmatrix_dev(long int row_dim, long int col_dim)
{
	long int row_index, col_index;
	GDSMatrix ret = NULL;
	dsfloat dszero = {{0.0f, 0.0f}};
	gds_real zero;

	ds2gds(&zero, &dszero);

	// callocation
	ret = (gdsmatrix *)malloc(sizeof(gdsmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one GDSMatrix\n");
		return ret;
	}

	ret->row_dim = row_dim;
	ret->col_dim = col_dim;

	ret->element = (gds_real *)calloc(row_dim * col_dim, sizeof(gds_real));
	cudaMalloc((void **)&(ret->element), (size_t)(row_dim * col_dim * sizeof(gds_real)));

	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GDSMatrix(row_dim, col_dim = %ld, %ld)\n", row_dim, col_dim);
		free(ret);
		return NULL;
	}

	// set all zeros 
	for(row_index = 0; row_index < row_dim; row_index++)
	{
		for(col_index = 0; col_index < col_dim; col_index++)
			gds2gds_dev(&(ret->element[row_index * col_dim + col_index]), &zero);
	}

	return ret;
}

// free dsvector
void free_gdsmatrix_dev(GDSMatrix mat)
{
	cudaFree(mat->element);
	free(mat);
}

// print dsvector
void print_gdsmatrix_dev(GDSMatrix mat)
{
	long int row_index, col_index;
	dsfloat dsval;

	for(row_index = 0; row_index < mat->row_dim; row_index++)
	{
		for(col_index = 0; col_index < mat->col_dim; col_index++)
		{
			printf("%ld, %ld: ", row_index, col_index);
			gds2ds_dev(&dsval, &(mat->element[row_index * mat->col_dim + col_index]));
			std::cout << dsval.val[0] << " + " << dsval.val[1] << "\n";
		}
	}
}


__global__ void _bncu_mul_gdsmatrix(gds_real *ret_dev_element, long int row_dim, long int col_dim, long int mid_dim, gds_real *a_dev_element, gds_real *b_dev_element)
{
	long int i, j, k;
	gds_real tmp, tmp1;

	i = threadIdx.x + blockIdx.x * blockDim.x;

	//for(i = 0; i < row_dim; i++)
	while(i < row_dim)
	{
		for(j = 0; j < col_dim; j++)
		{
			SET0_GDS(tmp1); // = (gds_real)0.0;
			SET0_GDS(tmp); // = (gds_real)0.0;
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
void mul_gdsmatrix_dev(GDSMatrix ret_dev, GDSMatrix a_dev, GDSMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long row_dim, col_dim, mid_dim;

	/* dimension check */
	if((ret_dev->row_dim != a_dev->row_dim) || (ret_dev->col_dim != b_dev->col_dim) || (a_dev->col_dim != b_dev->row_dim))
	{
		fprintf(stderr, "ERROR: mul_gdsmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret_dev->row_dim, ret_dev->col_dim, a_dev->row_dim, a_dev->col_dim, b_dev->row_dim, b_dev->col_dim);
		return;
	}

	row_dim = ret_dev->row_dim;
	col_dim = ret_dev->col_dim;
	mid_dim = a_dev->col_dim;

	_bncu_mul_gdsmatrix<<<num_blocks_per_grid, num_threads_per_block>>>(ret_dev->element, row_dim, col_dim, mid_dim, a_dev->element, b_dev->element);

}

// copy DSMatrix to GDSMatrix on GPU
// gdsvec on GPU := dsvec
__host__ void subst_gdsmatrix_dev_dsmat(GDSMatrix gdsmat_dev, DSMatrix dsmat)
{
	long int i, j, row_dim, col_dim;

	row_dim = gdsmat_dev->row_dim;
	col_dim = gdsmat_dev->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			/* DSMatrix::element is SoA: double *element[DSSIZE].
			 * Materialize an AoS dsfloat for ds2gds_dev. */
			dsfloat tmp_ds;
			long int idx = i * col_dim + j;
			tmp_ds.val[0] = dsmat->element[0][idx];
			tmp_ds.val[1] = dsmat->element[1][idx];
			ds2gds_dev(&(gdsmat_dev->element[idx]), &tmp_ds);
		}
	}
}

// copy GDSMatrix on GPU to DSMatrix
// dsvec := gdsmat_dev on GPU
__host__ void subst_dsmatrix_gdsmat_dev(DSMatrix dsmat, GDSMatrix gdsmat_dev)
{
	long int i, j, row_dim, col_dim;

	row_dim = gdsmat_dev->row_dim;
	col_dim = gdsmat_dev->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			/* SoA layout: see subst_gdsmatrix_dev_dsmat above. */
			dsfloat tmp_ds;
			long int idx = i * col_dim + j;
			gds2ds_dev(&tmp_ds, &(gdsmat_dev->element[idx]));
			dsmat->element[0][idx] = tmp_ds.val[0];
			dsmat->element[1][idx] = tmp_ds.val[1];
		}
	}
}

// Frobenius norm
__host__ void normf_gdsmatrix_dev(gds_real *ret_dev, GDSMatrix mat_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	int block_dim, thread_dim, tmp_vec_dev_dim, total_dim;
//	gds_real tmp, tmp1;
	gds_real *block_cache_dev; // partial sum per block
	gds_real *tmp_vec_dev; // square mul of mat

	total_dim = mat_dev->row_dim * mat_dev->col_dim; // total dimension as one vector

	// initialize block_cache_dev, tmp_vec_dev
	cudaMalloc((void **)&block_cache_dev, (size_t)(sizeof(gds_real) * num_blocks_per_grid));
	cudaMalloc((void **)&tmp_vec_dev, (size_t)(sizeof(gds_real) * total_dim));

	// mat[i][j]^2
	//_bncu_mul_gdsvector<<<num_blocks_per_grid, num_threads_per_block>>>(tmp_vec_dev, mat_dev->element, mat_dev->element, total_dim);
	_bncu_sqr_gdsvector<<<num_blocks_per_grid, num_threads_per_block>>>(tmp_vec_dev, mat_dev->element, total_dim);

	// reduction to add
	block_dim = num_blocks_per_grid;
//	block_dim = 1;
	thread_dim = num_threads_per_block;
	tmp_vec_dev_dim = total_dim;

	while(block_dim >= 1)
	{
		_bncu_add_reduct_gdsvector<<<block_dim, thread_dim>>>(block_cache_dev, tmp_vec_dev, tmp_vec_dev_dim);

		if(block_dim <= 1)
			break;

		tmp_vec_dev_dim = block_dim;
		if(block_dim > thread_dim)
			block_dim = (block_dim / thread_dim) + 1;
		else
			block_dim = 1;

		cudaMemcpy((void *)tmp_vec_dev, (void *)block_cache_dev, (size_t)(sizeof(gds_real) * tmp_vec_dev_dim), cudaMemcpyDeviceToDevice);

	}

	// ret := sqrt(ret);
	_bncu_sqrt_gdsvector<<<1, 1>>>(ret_dev, block_cache_dev, 1);

	// free
	cudaFree(block_cache_dev);
	cudaFree(tmp_vec_dev);

/*
	SET0_GDS(*ret); // = (gds_real)0.0;
	SET0_GDS(tmp); // = (gds_real)(int)0;
	SET0_GDS(tmp1); // = (gds_real)(int)0;

	dim = mat->row_dim * mat->col_dim;

	for(i = 0; i < dim ; i++)
	{
		//tmp += mat->element[i] * mat->element[i];
		//tmp += gds_real::mul((gds_real)(mat->element[i]), (gds_real)(mat->element[i]));
		tmp1 = (gds_real)(mat->element[i]) * (gds_real)(mat->element[i]);
		tmp = tmp + tmp1;
	}

	//*ret = sqrt(tmp);
	tmp = (gds_real)sqrt(tmp);
	*ret = tmp;
*/
}

/*************************************************/
/* Matrix Caluculations for GDSMatrix            */
/*
void normf_gdsmatrix(double ret[DSSIZE], GDSMatrix mat)
void norm1_gdsmatrix(double ret[DSSIZE], GDSMatrix mat)
void normi_gdsmatrix(double ret[DSSIZE], GDSMatrix mat)
void add_gdsmatrix(GDSMatrix c, GDSMatrix a, GDSMatrix b);
void sub_gdsmatrix(GDSMatrix c, GDSMatrix a, GDSMatrix b);
void mul_gdsmatrix(GDSMatrix c, GDSMatrix a, GDSMatrix b);
void mul_gdsmatrix_dsvec(DSVector v, GDSMatrix a, DSVector vb)
void mul_gdsmatrixt_dsvec(DSVector v, GDSMatrix a, DSVector vb)
void transpose_gdsmatrix(GDSMatrix c, GDSMatrix a);
void inv_gdsmatrix(GDSMatrix a);
void subst_mpfmatrux(GDSMatrix c, GDSMatrix a);
*/
/*************************************************/

#if 0
/* Infinity Norm of Matrix */
void normi_gdsmatrix(gds_real *ret, GDSMatrix mat, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i, j;
	gds_real sum;

	SET0_GDS(*ret); // = 0.0;

	for(i = 0; i < mat->row_dim; i++)
	{
		SET0_GDS(sum); //  = 0.0;
		for(j = 0; j < mat->col_dim; j++)
			sum = sum + abs(mat->element[i * mat->row_dim + j]);

		if(*ret < sum)
			*ret = sum;

	}

	return;
}

/* 1 Norm of Matrix */
void norm1_gdsmatrix(gds_real *ret, GDSMatrix mat, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i, j;
	gds_real sum;

	SET0_GDS(*ret); // = 0.0;

	for(j = 0; j < mat->col_dim; j++)
	{
		SET0_GDS(sum); // = 0.0;
		for(i = 0; i < mat->row_dim; i++)
			sum = sum + abs(mat->element[i * mat->col_dim + j]);

		if(*ret < sum)
			*ret = sum;

	}

	return;
}
#endif // 0

/* c := a + b */
void add_gdsmatrix_dev(GDSMatrix c_dev, GDSMatrix a_dev, GDSMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int row_dim, col_dim; //, index;

	/* check row_dim */
	if((a_dev->row_dim != b_dev->row_dim) || (b_dev->row_dim != c_dev->row_dim) || (c_dev->row_dim != a_dev->row_dim))
	{
		fprintf(stderr, "ERROR: add_gdsmatrix_dev\n");
		return;
	}
	row_dim = c_dev->row_dim;

	/* check col_dim */
	if((a_dev->col_dim != b_dev->col_dim) || (b_dev->col_dim != c_dev->col_dim) || (c_dev->col_dim != a_dev->col_dim))
	{
		fprintf(stderr, "ERROR: add_gdsmatrix_dev\n");
		return;
	}
	col_dim = c_dev->col_dim;

	_bncu_add_gdsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, row_dim * col_dim);

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
void sub_gdsmatrix_dev(GDSMatrix c_dev, GDSMatrix a_dev, GDSMatrix b_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int row_dim, col_dim; //, index;

	/* check row_dim */
	if((a_dev->row_dim != b_dev->row_dim) || (b_dev->row_dim != c_dev->row_dim) || (c_dev->row_dim != a_dev->row_dim))
	{
		fprintf(stderr, "ERROR: sub_gdsmatrix_dev\n");
		return;
	}
	row_dim = c_dev->row_dim;

	/* check col_dim */
	if((a_dev->col_dim != b_dev->col_dim) || (b_dev->col_dim != c_dev->col_dim) || (c_dev->col_dim != a_dev->col_dim))
	{
		fprintf(stderr, "ERROR: sub_gdsmatrix_dev\n");
		return;
	}
	col_dim = c_dev->col_dim;

	_bncu_sub_gdsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, b_dev->element, row_dim * col_dim);

}

/* c := sc * a */
void cmul_gdsmatrix_dev(GDSMatrix c_dev, gds_real sc, GDSMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int row_dim, col_dim; //, index;

	/* check row_dim */
	if(a_dev->row_dim != c_dev->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_gdsmatrix_dev\n");
		return;
	}
	row_dim = c_dev->row_dim;

	/* check col_dim */
	if(a_dev->col_dim != c_dev->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_gdsmatrix_dev\n");
		return;
	}
	col_dim = c_dev->col_dim;

	_bncu_cmul_gdsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, sc, a_dev->element, row_dim * col_dim);


}

/* c = a^T */
__global__ void _bncu_transpose_gdsmatrix(gds_real *c_dev_element, gds_real *a_dev_element, long int row_dim, long int col_dim)
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

void transpose_gdsmatrix_dev(GDSMatrix c_dev, GDSMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	//long int i, j;

	/* Check Dimensions */
	if((c_dev->row_dim != a_dev->col_dim) || (c_dev->col_dim != a_dev->row_dim))
	{
		fprintf(stderr, "ERROR: transpose_gdsmatrix_dev\n");
		return;
	}
	
	_bncu_transpose_gdsmatrix<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, c_dev->row_dim, c_dev->col_dim);

/*	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			c->element[i * c->col_dim + j] = a->element[j * a->col_dim + i];
	}
*/
}

/* c := a */
void subst_gdsmatrix_dev(GDSMatrix c_dev, GDSMatrix a_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	//long int i, j, index;

	if((c_dev->row_dim != a_dev->row_dim) || (c_dev->col_dim != a_dev->col_dim))
	{
		fprintf(stderr, "ERROR: subst_gdsmatrix_dev\n");
		return;
	}

	_bncu_subst_gdsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, a_dev->element, c_dev->row_dim * c_dev->col_dim);

}

/* c := 0 */
void set0_gdsmatrix_dev(GDSMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block)
{
//	long int i, j;

	_bncu_set0_gdsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, c_dev->row_dim * c_dev->col_dim);

}

/* set (i, j)-element on GPU from dsfloat */
void set_gdsmatrix_ij_dev(GDSMatrix mat_dev, long int row_index, long int col_index, dsfloat val)
{
	if((row_index >= 0) && (row_index < mat_dev->row_dim) && (col_index >= 0) && (col_index < mat_dev->col_dim))
		ds2gds_dev(&(mat_dev->element[row_index * mat_dev->col_dim + col_index]), &val);
}

/* c := I */
void setI_gdsmatrix_dev(GDSMatrix c_dev, int num_blocks_per_grid, int num_threads_per_block)
{
	long int i;
	dsfloat one;

	one.val[0] = 1.0f; one.val[1] = 0.0f;

	_bncu_set0_gdsvector<<< num_blocks_per_grid, num_threads_per_block >>>(c_dev->element, c_dev->row_dim * c_dev->col_dim);

	for(i = 0; i < c_dev->row_dim; i++)
		set_gdsmatrix_ij_dev(c_dev, i, i, one);
/*
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			SET0_GDS(c->element[i * c->col_dim + j]); // = 0.0;
		if(i < c->col_dim)
			SET1_GDS(c->element[i * c->col_dim + i]); // = 1.0;
	}
*/
}

/* v := a * vb */
__global__ void _bncu_mul_gdsmatrix_gdsvec(gds_real *v_element, gds_real *a_element, gds_real *vb_element, long int row_dim, long int col_dim)
{
	int j, index;
	gds_real tmp;

	index = threadIdx.x + blockIdx.x * blockDim.x;

	//for(i = 0; i < row_dim; i++)
	while(index < row_dim)
	{
		SET0_GDS(tmp); // = 0.0;
		for(j = 0; j < col_dim; j++)
			tmp = tmp + a_element[index * col_dim + j] * vb_element[j];

		v_element[index] = tmp;

		__syncthreads();

		index += blockDim.x * gridDim.x;
	}
}

void mul_gdsmatrix_gdsvec(GDSVector v, GDSMatrix a, GDSVector vb, int num_blocks_per_grid, int num_threads_per_block)
{
	/* Check Dimension */
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: mul_gdsmatrix_dsvec\n");
		return;
	}

	_bncu_mul_gdsmatrix_gdsvec<<<num_blocks_per_grid, num_threads_per_block>>>(v->element, a->element, vb->element, a->row_dim, a->col_dim);

/*	for(i = 0; i < a->row_dim; i++)
	{
		SET0_GDS(tmp); // = 0.0;
		for(j = 0; j < a->col_dim; j++)
			tmp = tmp + a->element[i * a->col_dim + j] * vb->element[j];

		v->element[i] = tmp;
	}
*/
}

/* v := a^T * vb */
__global__ void _bncu_mul_gdsmatrixt_gdsvec(gds_real *v_element, gds_real *a_element, gds_real *vb_element, long int row_dim, long int col_dim)
{
	int j, index;
	gds_real tmp;

	index = threadIdx.x + blockIdx.x * blockDim.x;

	//for(i = 0; i < col_dim; i++)
	while(index < col_dim)
	{
		SET0_GDS(tmp); // = 0.0;
		for(j = 0; j < row_dim; j++)
			tmp = tmp + a_element[j * row_dim + index] * vb_element[j];

		v_element[index] = tmp;

		__syncthreads();

		index += blockDim.x * gridDim.x;
	}
}

__host__ void mul_gdsmatrixt_gdsvec(GDSVector v, GDSMatrix a, GDSVector vb, int num_blocks_per_grid, int num_threads_per_block)
{

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_gdsmatrixt_gdsvec\n");
		return;
	}

	_bncu_mul_gdsmatrixt_gdsvec<<<num_blocks_per_grid, num_threads_per_block>>>(v->element, a->element, vb->element, a->row_dim, a->col_dim);

/*	for(i = 0; i < a->col_dim; i++)
	{
		SET0_GDS(tmp); // = 0.0;
		for(j = 0; j < a->row_dim; j++)
			tmp = tmp + a->element[j * a->row_dim + i] * vb->element[j];

		v->element[i] = tmp;
	}
*/
}

// Test main function
// NOTE: disabled — this stand-alone driver uses qd-library API (dsfloat::sqrt,
// dsfloat::_ndigits, operator<<(ostream,dsfloat), fpu_fix_start, GDSStart/End)
// that does not exist for the POD dsfloat type.  Use bench_gdtqs_strassen instead.
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
	DSVector dsvec_a, dsvec_b, dsvec_c;
	DSMatrix dsmat_a, dsmat_b, dsmat_c;
	dsfloat dsval;
	int num_blocks, num_threads;
	GDSVector gdsvec_a, gdsvec_b, gdsvec_c;
	gds_real gdsval, *ptr_gdsval_dev;
	GDSMatrix gdsmat_a, gdsmat_b, gdsmat_c;
	

	// Initialize QD library
	fpu_fix_start(NULL);

	// DD & GDS
	dsvec_a = init_dsvector(ROW_DIM);
	dsvec_b = init_dsvector(ROW_DIM);
	dsvec_c = init_dsvector(ROW_DIM);

	for(i = 0; i < ROW_DIM; i++)
	{
		// val := sqrt((i + 1));
		dsvec_a->element[i] = dsfloat::sqrt((int)(i + 1));
		dsvec_b->element[i] = dsfloat::sqrt((int)(i + 1));
		//dsvec->element[i] = sqrt(qsval);
	}

	print_dsvector(dsvec_a);
	norm2_dsvector(&dsval, dsvec_a);
	printf("DD : ||vec||_2 = ");
	cout.precision(dsfloat::_ndigits);
	cout << dsval << "\n";

	// GDS start!
	GDSStart();

//	num_blocks = 1;  // #blocks per grid
//	num_blocks = 2;  // #blocks per grid
	num_blocks = 4;  // #blocks per grid
//	num_threads = 16; // #threads per block
	num_threads = 64; // #threads per block

	gdsvec_a = init_gdsvector_dev(ROW_DIM);
	gdsvec_b = init_gdsvector_dev(ROW_DIM);
	gdsvec_c = init_gdsvector_dev(ROW_DIM);
	cudaMalloc((void **)&ptr_gdsval_dev, sizeof(gds_real));

	// gdsvec := dsvec
	subst_gdsvector_dev_dsvec(gdsvec_a, dsvec_a);
	subst_gdsvector_dev_dsvec(gdsvec_b, dsvec_b);
	//subst_dsvector_gdsvec_dev(dsvec_c, gdsvec_c);

	// ----------
	// c := a + b
	// ----------
	// DD
	printf(" DD: c := a + b\n");
	add_dsvector(dsvec_c, dsvec_a, dsvec_b);
	print_dsvector(dsvec_c);

	// GDS
	printf("GDS: c := a + b\n");
	add_gdsvector_dev(gdsvec_c, gdsvec_a, gdsvec_b, num_blocks, num_threads);
	print_gdsvector_dev(gdsvec_c);

	// ----------
	// c := a - b
	// ----------
	// DD
	printf(" DD: c := a - b\n");
	sub_dsvector(dsvec_c, dsvec_a, dsvec_b);
	print_dsvector(dsvec_c);

	// GDS
	printf("GDS: c := a - b\n");
	sub_gdsvector_dev(gdsvec_c, gdsvec_a, gdsvec_b, num_blocks, num_threads);
	print_gdsvector_dev(gdsvec_c);

	// ----------
	// c := val * a
	// ----------
	// DD
	dsval = (dsfloat)2.0;
	printf(" DD: c := val * a\n");
	cmul_dsvector(dsvec_c, dsval, dsvec_a);
	print_dsvector(dsvec_c);

	// GDS
	ds2gds(&gdsval, &dsval);
	printf("GDS: c := val * a\n");
	cmul_gdsvector_dev(gdsvec_c, gdsval, gdsvec_a, num_blocks, num_threads);
	print_gdsvector_dev(gdsvec_c);

	// ----------
	// (a, b)
	// ----------
	// DD
	ip_dsvector(&dsval, dsvec_a, dsvec_b);
	printf(" DD : (a, b) = ");
	cout.precision(dsfloat::_ndigits);
	cout << dsval << "\n";

	// GDS
	ip_gdsvector_dev(ptr_gdsval_dev, gdsvec_a, gdsvec_b, num_blocks, num_threads);
	//ip_gdsvector_dev(ptr_gdsval_dev, gdsvec_a, gdsvec_b, 1, num_threads);
	gds2ds_dev(&dsval, ptr_gdsval_dev);
	printf("GDS : (a, b) = ");
	cout.precision(dsfloat::_ndigits);
	cout << dsval << "\n";

	// ----------
	// ||a||_2
	// ----------
	// DD
	norm2_dsvector(&dsval, dsvec_a);
	printf(" DD : ||a||_2 = ");
	cout.precision(dsfloat::_ndigits);
	cout << dsval << "\n";

	// GDS
	norm2_gdsvector_dev(ptr_gdsval_dev, gdsvec_a, num_blocks, num_threads);
	//norm2_gdsvector_dev(ptr_gdsval_dev, gdsvec_a, 1, num_threads);
	gds2ds_dev(&dsval, ptr_gdsval_dev);
	printf("GDS : ||a||_2 = ");
	cout.precision(dsfloat::_ndigits);
	cout << dsval << "\n";

/* Matrix */
	// DD & GDS
	dsmat_a = init_dsmatrix(ROW_DIM, COL_DIM);
	dsmat_b = init_dsmatrix(ROW_DIM, COL_DIM);
	dsmat_c = init_dsmatrix(ROW_DIM, COL_DIM);

	gdsmat_a = init_gdsmatrix_dev(ROW_DIM, COL_DIM);
	gdsmat_b = init_gdsmatrix_dev(ROW_DIM, COL_DIM);
	gdsmat_c = init_gdsmatrix_dev(ROW_DIM, COL_DIM);

	for(i = 0; i < ROW_DIM; i++)
	{
		for(j = 0; j < COL_DIM; j++)
		{
			// val := sqrt((i + 1));
			dsmat_a->element[i * COL_DIM + j] = dsfloat::sqrt((int)(i + j + 1));
			dsmat_b->element[i * COL_DIM + j] = dsfloat::sqrt((int)(i + j + 1));
		}
	}
	subst_gdsmatrix_dev_dsmat(gdsmat_a, dsmat_a);
	subst_gdsmatrix_dev_dsmat(gdsmat_b, dsmat_b);

	// Print gdsmatrix
	printf("dsmat_a:\n");
	normf_dsmatrix(&dsval, dsmat_a);
	cout.precision(dsfloat::_ndigits);
	cout << dsval << "\n";

	printf("gdsmat_a:\n");
	//print_gdsmatrix_dev(gdsmat_a);
	normf_gdsmatrix_dev(ptr_gdsval_dev, gdsmat_a, 1, num_threads);
	gds2ds_dev(&dsval, ptr_gdsval_dev);
	cout.precision(dsfloat::_ndigits);
	cout << dsval << "\n";

	printf("dsmat_b:\n");
	normf_dsmatrix(&dsval, dsmat_b);
	cout.precision(dsfloat::_ndigits);
	cout << dsval << "\n";

	printf("gdsmat_b:\n");
	//print_gdsmatrix_dev(gdsmat_b);
	normf_gdsmatrix_dev(ptr_gdsval_dev, gdsmat_b, num_blocks, num_threads);
	gds2ds_dev(&dsval, ptr_gdsval_dev);
	cout.precision(dsfloat::_ndigits);
	cout << dsval << "\n";

	// ----------
	// C := A * B
	// ----------
	// DD
	mul_dsmatrix(dsmat_c, dsmat_a, dsmat_b);
	printf("DD : || A * B ||_F:\n");
	normf_dsmatrix(&dsval, dsmat_c);
	cout.precision(dsfloat::_ndigits);
	cout << dsval << "\n";

	// GDS
	mul_gdsmatrix_dev(gdsmat_c, gdsmat_a, gdsmat_b, num_blocks, num_threads);
	printf("GDS: || A * B ||_F:\n");
	normf_gdsmatrix_dev(ptr_gdsval_dev, gdsmat_c, num_blocks, num_threads);
	gds2ds_dev(&dsval, ptr_gdsval_dev);
	cout.precision(dsfloat::_ndigits);
	cout << dsval << "\n";

/* Free! */
	cudaFree(ptr_gdsval_dev);
	free_gdsvector_dev(gdsvec_a);
	free_gdsvector_dev(gdsvec_b);
	free_gdsvector_dev(gdsvec_c);

	free_gdsmatrix_dev(gdsmat_a);
	free_gdsmatrix_dev(gdsmat_b);
	free_gdsmatrix_dev(gdsmat_c);

	// GDS end!
	GDSEnd();

	// Free DDVectors
	free_dsvector(dsvec_a);
	free_dsvector(dsvec_b);
	free_dsvector(dsvec_c);

	free_dsmatrix(dsmat_a);
	free_dsmatrix(dsmat_b);
	free_dsmatrix(dsmat_c);

	return 0;
}
#endif // DEBUG
