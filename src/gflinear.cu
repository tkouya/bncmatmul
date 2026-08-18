/********************************************************************************/
/* gflinear.cu: Native float precision GPU Linear Computation Library with CUDA*/
/* Copyright (C) 2026 Tomonori Kouya                                            */
/*                                                                              */
/* This program is free software: you can redistribute it and/or modify it      */
/* under the terms of the GNU Lesser General Public License as published by the */
/* Free Software Foundation, either version 3 of the License or any later       */
/* version.                                                                     */
/********************************************************************************/

#include "gflinear.h"

/*-------------------------------------------------------------------*/
/* GFVector                                                          */
/*-------------------------------------------------------------------*/
__host__ GFVector init_gfvector(long int dim)
{
	GFVector ret = (gfvector *)malloc(sizeof(gfvector));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one GFVector\n");
		return NULL;
	}
	ret->dim = dim;
	ret->element = (float *)calloc(dim, sizeof(float));
	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GFVector(dim = %ld)\n", dim);
		free(ret);
		return NULL;
	}
	return ret;
}

__host__ void free_gfvector(GFVector vec)
{
	free(vec->element);
	free(vec);
}

__host__ GFVector init_gfvector_dev(long int dim)
{
	GFVector ret = (gfvector *)malloc(sizeof(gfvector));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one GFVector\n");
		return NULL;
	}
	ret->element = NULL;
	ret->dim = dim;
	cudaMalloc((void **)&(ret->element), (size_t)(dim * sizeof(float)));
	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GFVector(dim = %ld) on GPU\n", dim);
		free(ret);
		return NULL;
	}
	cudaMemset((void *)(ret->element), 0, (size_t)(dim * sizeof(float)));
	return ret;
}

__host__ void free_gfvector_dev(GFVector vec)
{
	cudaFree(vec->element);
	free(vec);
}

// gfvec on GPU := fvec (host FVector is padded: real_dim stride for element, but
// element[i] is contiguous for i<dim, so a packed copy is valid)
__host__ void subst_gfvector_dev_fvec(GFVector gfvec_dev, FVector fvec)
{
	long int dim = gfvec_dev->dim;
	cudaMemcpy((void *)gfvec_dev->element, (void *)fvec->element, (size_t)(dim * sizeof(float)), cudaMemcpyHostToDevice);
}

__host__ void subst_fvector_gfvec_dev(FVector fvec, GFVector gfvec_dev)
{
	long int dim = fvec->dim;
	cudaMemcpy((void *)fvec->element, (void *)gfvec_dev->element, (size_t)(dim * sizeof(float)), cudaMemcpyDeviceToHost);
}

__host__ void print_gfvector_dev(GFVector dev_vec)
{
	long int index, dim = dev_vec->dim;
	GFVector host_vec = init_gfvector(dim);
	cudaMemcpy((void *)(host_vec->element), (void *)(dev_vec->element), (size_t)(sizeof(float) * dim), cudaMemcpyDeviceToHost);
	for(index = 0; index < dim; index++)
		printf("%4ld: %25.17e\n", index, host_vec->element[index]);
	free_gfvector(host_vec);
}

/* c = a + b */
__global__ void _bncu_add_gfvector(float *c, float *a, float *b, long int dim)
{
	long int index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim) { c[index] = a[index] + b[index]; index += blockDim.x * gridDim.x; }
}
__host__ void add_gfvector_dev(GFVector c_dev, GFVector a_dev, GFVector b_dev, int nbg, int ntb)
{
	if((a_dev->dim != b_dev->dim) || (c_dev->dim != a_dev->dim))
	{ fprintf(stderr, "ERROR: add_gfvector_dev\n"); return; }
	_bncu_add_gfvector<<<nbg, ntb>>>(c_dev->element, a_dev->element, b_dev->element, c_dev->dim);
}

/* c = a - b */
__global__ void _bncu_sub_gfvector(float *c, float *a, float *b, long int dim)
{
	long int index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim) { c[index] = a[index] - b[index]; index += blockDim.x * gridDim.x; }
}
__host__ void sub_gfvector_dev(GFVector c_dev, GFVector a_dev, GFVector b_dev, int nbg, int ntb)
{
	if((a_dev->dim != b_dev->dim) || (c_dev->dim != a_dev->dim))
	{ fprintf(stderr, "ERROR: sub_gfvector_dev\n"); return; }
	_bncu_sub_gfvector<<<nbg, ntb>>>(c_dev->element, a_dev->element, b_dev->element, c_dev->dim);
}

/* c = val * a */
__global__ void _bncu_cmul_gfvector(float *ret, float val, float *a, long int dim)
{
	long int index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim) { ret[index] = val * a[index]; index += blockDim.x * gridDim.x; }
}
__host__ void cmul_gfvector_dev(GFVector c_dev, float val, GFVector a_dev, int nbg, int ntb)
{
	if(c_dev->dim != a_dev->dim) { fprintf(stderr, "ERROR: cmul_gfvector_dev\n"); return; }
	_bncu_cmul_gfvector<<<nbg, ntb>>>(c_dev->element, val, a_dev->element, c_dev->dim);
}

/* c := vec */
__global__ void _bncu_subst_gfvector(float *ret, float *vec, long int dim)
{
	long int index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim) { ret[index] = vec[index]; index += blockDim.x * gridDim.x; }
}
__host__ void subst_gfvector_dev(GFVector ret_dev, GFVector vec_dev, int nbg, int ntb)
{
	if(ret_dev->dim != vec_dev->dim) { fprintf(stderr, "ERROR: subst_gfvector_dev\n"); return; }
	_bncu_subst_gfvector<<<nbg, ntb>>>(ret_dev->element, vec_dev->element, ret_dev->dim);
}

/* c := 0 */
__global__ void _bncu_set0_gfvector(float *ret, long int dim)
{
	long int index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim) { ret[index] = 0.0; index += blockDim.x * gridDim.x; }
}
__host__ void set0_gfvector_dev(GFVector ret_dev, int nbg, int ntb)
{
	_bncu_set0_gfvector<<<nbg, ntb>>>(ret_dev->element, ret_dev->dim);
}

// element-wise helpers
__global__ void _bncu_mul_gfvector(float *ret, float *a, float *b, long int dim)
{
	long int index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim) { ret[index] = a[index] * b[index]; index += blockDim.x * gridDim.x; }
}
__global__ void _bncu_sqr_gfvector(float *ret, float *vec, long int dim)
{
	long int index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim) { ret[index] = vec[index] * vec[index]; index += blockDim.x * gridDim.x; }
}
__global__ void _bncu_sqrt_gfvector(float *ret, float *vec, long int dim)
{
	long int index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim) { ret[index] = sqrt(vec[index]); index += blockDim.x * gridDim.x; }
}
__global__ void _bncu_abs_gfvector(float *ret, float *vec, long int dim)
{
	long int index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim) { ret[index] = fabs(vec[index]); index += blockDim.x * gridDim.x; }
}

// reduction: sum
__global__ void _bncu_add_reduct_gfvector(float *ret, float *vec, long int dim)
{
	long int index, cache_index;
	__shared__ float cache[MAX_NUM_THREADS_PER_BLOCK];
	index = threadIdx.x + blockIdx.x * blockDim.x;
	cache_index = threadIdx.x;
	cache[cache_index] = 0.0;
	while(index < dim) { cache[cache_index] += vec[index]; index += blockDim.x * gridDim.x; }
	__syncthreads();
	index = blockDim.x / 2;
	while(index > 0)
	{
		if(cache_index < index) cache[cache_index] += cache[cache_index + index];
		__syncthreads();
		index /= 2;
	}
	if(cache_index == 0) ret[blockIdx.x] = cache[0];
}

// reduction: max
__global__ void _bncu_max_reduct_gfvector(float *ret, float *vec, long int dim)
{
	long int index, cache_index;
	__shared__ float cache[MAX_NUM_THREADS_PER_BLOCK];
	index = threadIdx.x + blockIdx.x * blockDim.x;
	cache_index = threadIdx.x;
	cache[cache_index] = vec[0];
	while(index < dim) { if(cache[cache_index] < vec[index]) cache[cache_index] = vec[index]; index += blockDim.x * gridDim.x; }
	__syncthreads();
	index = blockDim.x / 2;
	while(index > 0)
	{
		if(cache_index < index) { if(cache[cache_index] < cache[cache_index + index]) cache[cache_index] = cache[cache_index + index]; }
		__syncthreads();
		index /= 2;
	}
	if(cache_index == 0) ret[blockIdx.x] = cache[0];
}

// reduce a device vector "src" of length len to a single sum/max in ret_dev (one element)
static void _gf_reduce(float *ret_dev, float *src, long int len, int nbg, int ntb, int do_max, int do_sqrt)
{
	float *c_dev, *block_cache;
	int block_dim, thread_dim;
	long int dim;
	cudaMalloc((void **)&c_dev, (size_t)(sizeof(float) * len));
	cudaMalloc((void **)&block_cache, (size_t)(sizeof(float) * nbg));
	cudaMemcpy((void *)c_dev, (void *)src, (size_t)(sizeof(float) * len), cudaMemcpyDeviceToDevice);

	block_dim = nbg; thread_dim = ntb; dim = len;
	while(block_dim >= 1)
	{
		if(do_max) _bncu_max_reduct_gfvector<<<block_dim, thread_dim>>>(block_cache, c_dev, dim);
		else       _bncu_add_reduct_gfvector<<<block_dim, thread_dim>>>(block_cache, c_dev, dim);
		if(block_dim <= 1) break;
		dim = block_dim;
		if(block_dim > thread_dim) block_dim = (block_dim / thread_dim) + 1;
		else block_dim = 1;
		cudaMemcpy((void *)c_dev, (void *)block_cache, (size_t)(sizeof(float) * dim), cudaMemcpyDeviceToDevice);
	}
	if(do_sqrt) _bncu_sqrt_gfvector<<<1, 1>>>(ret_dev, block_cache, 1);
	else        cudaMemcpy((void *)ret_dev, (void *)&(block_cache[0]), sizeof(float), cudaMemcpyDeviceToDevice);
	cudaFree(c_dev);
	cudaFree(block_cache);
}

/* (a, b) */
__host__ void ip_gfvector_dev(float *ret_dev, GFVector a_dev, GFVector b_dev, int nbg, int ntb)
{
	float *c_dev;
	if(a_dev->dim != b_dev->dim) { fprintf(stderr, "ERROR: ip_gfvector_dev\n"); return; }
	cudaMalloc((void **)&c_dev, (size_t)(sizeof(float) * a_dev->dim));
	_bncu_mul_gfvector<<<nbg, ntb>>>(c_dev, a_dev->element, b_dev->element, a_dev->dim);
	_gf_reduce(ret_dev, c_dev, a_dev->dim, nbg, ntb, 0, 0);
	cudaFree(c_dev);
}

__host__ void norm2_gfvector_dev(float *ret_dev, GFVector a_dev, int nbg, int ntb)
{
	float *c_dev;
	if(a_dev->dim <= 0) { fprintf(stderr, "ERROR: norm2_gfvector_dev\n"); return; }
	cudaMalloc((void **)&c_dev, (size_t)(sizeof(float) * a_dev->dim));
	_bncu_sqr_gfvector<<<nbg, ntb>>>(c_dev, a_dev->element, a_dev->dim);
	_gf_reduce(ret_dev, c_dev, a_dev->dim, nbg, ntb, 0, 1);
	cudaFree(c_dev);
}

__host__ void norm1_gfvector_dev(float *ret_dev, GFVector a_dev, int nbg, int ntb)
{
	float *c_dev;
	if(a_dev->dim <= 0) { fprintf(stderr, "ERROR: norm1_gfvector_dev\n"); return; }
	cudaMalloc((void **)&c_dev, (size_t)(sizeof(float) * a_dev->dim));
	_bncu_abs_gfvector<<<nbg, ntb>>>(c_dev, a_dev->element, a_dev->dim);
	_gf_reduce(ret_dev, c_dev, a_dev->dim, nbg, ntb, 0, 0);
	cudaFree(c_dev);
}

__host__ void normi_gfvector_dev(float *ret_dev, GFVector a_dev, int nbg, int ntb)
{
	float *c_dev;
	if(a_dev->dim <= 0) { fprintf(stderr, "ERROR: normi_gfvector_dev\n"); return; }
	cudaMalloc((void **)&c_dev, (size_t)(sizeof(float) * a_dev->dim));
	_bncu_abs_gfvector<<<nbg, ntb>>>(c_dev, a_dev->element, a_dev->dim);
	_gf_reduce(ret_dev, c_dev, a_dev->dim, nbg, ntb, 1, 0);
	cudaFree(c_dev);
}

/*-------------------------------------------------------------------*/
/* GFMatrix                                                          */
/*-------------------------------------------------------------------*/
__host__ GFMatrix init_gfmatrix(long int row_dim, long int col_dim)
{
	GFMatrix ret = (gfmatrix *)malloc(sizeof(gfmatrix));
	if(ret == NULL) { fprintf(stderr, "ERROR: cannot allocate one GFMatrix\n"); return NULL; }
	ret->row_dim = row_dim; ret->col_dim = col_dim;
	ret->element = (float *)calloc(row_dim * col_dim, sizeof(float));
	if(ret->element == NULL)
	{ fprintf(stderr, "ERROR: cannot allocate GFMatrix(%ld, %ld)\n", row_dim, col_dim); free(ret); return NULL; }
	return ret;
}

__host__ void free_gfmatrix(GFMatrix mat) { free(mat->element); free(mat); }

__host__ GFMatrix init_gfmatrix_dev(long int row_dim, long int col_dim)
{
	GFMatrix ret = (gfmatrix *)malloc(sizeof(gfmatrix));
	if(ret == NULL) { fprintf(stderr, "ERROR: cannot allocate one GFMatrix\n"); return NULL; }
	ret->row_dim = row_dim; ret->col_dim = col_dim; ret->element = NULL;
	cudaMalloc((void **)&(ret->element), (size_t)(row_dim * col_dim * sizeof(float)));
	if(ret->element == NULL)
	{ fprintf(stderr, "ERROR: cannot allocate GFMatrix(%ld, %ld) on GPU\n", row_dim, col_dim); free(ret); return NULL; }
	cudaMemset((void *)(ret->element), 0, (size_t)(row_dim * col_dim * sizeof(float)));
	return ret;
}

__host__ void free_gfmatrix_dev(GFMatrix mat) { cudaFree(mat->element); free(mat); }

__host__ void print_gfmatrix_dev(GFMatrix mat)
{
	long int i, j, row_dim = mat->row_dim, col_dim = mat->col_dim;
	GFMatrix h = init_gfmatrix(row_dim, col_dim);
	cudaMemcpy((void *)h->element, (void *)mat->element, (size_t)(sizeof(float) * row_dim * col_dim), cudaMemcpyDeviceToHost);
	for(i = 0; i < row_dim; i++)
		for(j = 0; j < col_dim; j++)
			printf("%ld, %ld: %25.17e\n", i, j, h->element[i * col_dim + j]);
	free_gfmatrix(h);
}

// matrix multiplication ret := A * B
__global__ void _bncu_mul_gfmatrix(float *ret, long int row_dim, long int col_dim, long int mid_dim, float *a, float *b)
{
	long int i, j, k;
	float tmp;
	i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < row_dim)
	{
		for(j = 0; j < col_dim; j++)
		{
			tmp = 0.0;
			for(k = 0; k < mid_dim; k++)
				tmp += a[i * mid_dim + k] * b[k * col_dim + j];
			ret[i * col_dim + j] = tmp;
		}
		i += blockDim.x * gridDim.x;
	}
}
__host__ void mul_gfmatrix_dev(GFMatrix ret_dev, GFMatrix a_dev, GFMatrix b_dev, int nbg, int ntb)
{
	if((ret_dev->row_dim != a_dev->row_dim) || (ret_dev->col_dim != b_dev->col_dim) || (a_dev->col_dim != b_dev->row_dim))
	{ fprintf(stderr, "ERROR: mul_gfmatrix_dev\n"); return; }
	_bncu_mul_gfmatrix<<<nbg, ntb>>>(ret_dev->element, ret_dev->row_dim, ret_dev->col_dim, a_dev->col_dim, a_dev->element, b_dev->element);
}

// gfmat on GPU := fmat (host FMatrix uses real_col_dim stride -> pack into col_dim)
__host__ void subst_gfmatrix_dev_fmat(GFMatrix gfmat_dev, FMatrix fmat)
{
	long int i, j, row_dim = gfmat_dev->row_dim, col_dim = gfmat_dev->col_dim;
	GFMatrix h = init_gfmatrix(row_dim, col_dim);
	for(i = 0; i < row_dim; i++)
		for(j = 0; j < col_dim; j++)
			h->element[i * col_dim + j] = get_fmatrix_ij(fmat, i, j);
	cudaMemcpy((void *)gfmat_dev->element, (void *)h->element, (size_t)(sizeof(float) * row_dim * col_dim), cudaMemcpyHostToDevice);
	free_gfmatrix(h);
}

__host__ void subst_fmatrix_gfmat_dev(FMatrix fmat, GFMatrix gfmat_dev)
{
	long int i, j, row_dim = gfmat_dev->row_dim, col_dim = gfmat_dev->col_dim;
	GFMatrix h = init_gfmatrix(row_dim, col_dim);
	cudaMemcpy((void *)h->element, (void *)gfmat_dev->element, (size_t)(sizeof(float) * row_dim * col_dim), cudaMemcpyDeviceToHost);
	for(i = 0; i < row_dim; i++)
		for(j = 0; j < col_dim; j++)
			set_fmatrix_ij(fmat, i, j, h->element[i * col_dim + j]);
	free_gfmatrix(h);
}

// Frobenius norm
__host__ void normf_gfmatrix_dev(float *ret_dev, GFMatrix mat_dev, int nbg, int ntb)
{
	float *tmp_vec;
	long int total = mat_dev->row_dim * mat_dev->col_dim;
	cudaMalloc((void **)&tmp_vec, (size_t)(sizeof(float) * total));
	_bncu_sqr_gfvector<<<nbg, ntb>>>(tmp_vec, mat_dev->element, total);
	_gf_reduce(ret_dev, tmp_vec, total, nbg, ntb, 0, 1);
	cudaFree(tmp_vec);
}

/* c := a + b */
__host__ void add_gfmatrix_dev(GFMatrix c_dev, GFMatrix a_dev, GFMatrix b_dev, int nbg, int ntb)
{
	if((a_dev->row_dim != c_dev->row_dim) || (a_dev->col_dim != c_dev->col_dim) ||
	   (b_dev->row_dim != c_dev->row_dim) || (b_dev->col_dim != c_dev->col_dim))
	{ fprintf(stderr, "ERROR: add_gfmatrix_dev\n"); return; }
	_bncu_add_gfvector<<<nbg, ntb>>>(c_dev->element, a_dev->element, b_dev->element, c_dev->row_dim * c_dev->col_dim);
}

/* c := a - b */
__host__ void sub_gfmatrix_dev(GFMatrix c_dev, GFMatrix a_dev, GFMatrix b_dev, int nbg, int ntb)
{
	if((a_dev->row_dim != c_dev->row_dim) || (a_dev->col_dim != c_dev->col_dim) ||
	   (b_dev->row_dim != c_dev->row_dim) || (b_dev->col_dim != c_dev->col_dim))
	{ fprintf(stderr, "ERROR: sub_gfmatrix_dev\n"); return; }
	_bncu_sub_gfvector<<<nbg, ntb>>>(c_dev->element, a_dev->element, b_dev->element, c_dev->row_dim * c_dev->col_dim);
}

/* c := sc * a */
__host__ void cmul_gfmatrix_dev(GFMatrix c_dev, float sc, GFMatrix a_dev, int nbg, int ntb)
{
	if((a_dev->row_dim != c_dev->row_dim) || (a_dev->col_dim != c_dev->col_dim))
	{ fprintf(stderr, "ERROR: cmul_gfmatrix_dev\n"); return; }
	_bncu_cmul_gfvector<<<nbg, ntb>>>(c_dev->element, sc, a_dev->element, c_dev->row_dim * c_dev->col_dim);
}

/* c = a^T */
__global__ void _bncu_transpose_gfmatrix(float *c, float *a, long int row_dim, long int col_dim)
{
	long int i, j;
	i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < row_dim)
	{
		for(j = 0; j < col_dim; j++)
			c[i * col_dim + j] = a[j * row_dim + i];
		i += blockDim.x * gridDim.x;
	}
}
__host__ void transpose_gfmatrix_dev(GFMatrix c_dev, GFMatrix a_dev, int nbg, int ntb)
{
	if((c_dev->row_dim != a_dev->col_dim) || (c_dev->col_dim != a_dev->row_dim))
	{ fprintf(stderr, "ERROR: transpose_gfmatrix_dev\n"); return; }
	_bncu_transpose_gfmatrix<<<nbg, ntb>>>(c_dev->element, a_dev->element, c_dev->row_dim, c_dev->col_dim);
}

/* c := a */
__host__ void subst_gfmatrix_dev(GFMatrix c_dev, GFMatrix a_dev, int nbg, int ntb)
{
	if((c_dev->row_dim != a_dev->row_dim) || (c_dev->col_dim != a_dev->col_dim))
	{ fprintf(stderr, "ERROR: subst_gfmatrix_dev\n"); return; }
	_bncu_subst_gfvector<<<nbg, ntb>>>(c_dev->element, a_dev->element, c_dev->row_dim * c_dev->col_dim);
}

/* c := 0 */
__host__ void set0_gfmatrix_dev(GFMatrix c_dev, int nbg, int ntb)
{
	_bncu_set0_gfvector<<<nbg, ntb>>>(c_dev->element, c_dev->row_dim * c_dev->col_dim);
}

/* set (i, j)-element on GPU */
__host__ void set_gfmatrix_ij_dev(GFMatrix mat_dev, long int row_index, long int col_index, float val)
{
	if((row_index >= 0) && (row_index < mat_dev->row_dim) && (col_index >= 0) && (col_index < mat_dev->col_dim))
		cudaMemcpy((void *)&(mat_dev->element[row_index * mat_dev->col_dim + col_index]), (void *)&val, sizeof(float), cudaMemcpyHostToDevice);
}

/* c := I */
__host__ void setI_gfmatrix_dev(GFMatrix c_dev, int nbg, int ntb)
{
	long int i;
	_bncu_set0_gfvector<<<nbg, ntb>>>(c_dev->element, c_dev->row_dim * c_dev->col_dim);
	for(i = 0; i < c_dev->row_dim; i++)
		set_gfmatrix_ij_dev(c_dev, i, i, 1.0);
}

/* v := a * vb */
__global__ void _bncu_mul_gfmatrix_gfvec(float *v, float *a, float *vb, long int row_dim, long int col_dim)
{
	long int j, index;
	float tmp;
	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < row_dim)
	{
		tmp = 0.0;
		for(j = 0; j < col_dim; j++)
			tmp += a[index * col_dim + j] * vb[j];
		v[index] = tmp;
		index += blockDim.x * gridDim.x;
	}
}
__host__ void mul_gfmatrix_gfvec(GFVector v, GFMatrix a, GFVector vb, int nbg, int ntb)
{
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{ fprintf(stderr, "ERROR: mul_gfmatrix_gfvec\n"); return; }
	_bncu_mul_gfmatrix_gfvec<<<nbg, ntb>>>(v->element, a->element, vb->element, a->row_dim, a->col_dim);
}

/* v := a^T * vb */
__global__ void _bncu_mul_gfmatrixt_gfvec(float *v, float *a, float *vb, long int row_dim, long int col_dim)
{
	long int j, index;
	float tmp;
	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < col_dim)
	{
		tmp = 0.0;
		for(j = 0; j < row_dim; j++)
			tmp += a[j * col_dim + index] * vb[j];
		v[index] = tmp;
		index += blockDim.x * gridDim.x;
	}
}
__host__ void mul_gfmatrixt_gfvec(GFVector v, GFMatrix a, GFVector vb, int nbg, int ntb)
{
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{ fprintf(stderr, "ERROR: mul_gfmatrixt_gfvec\n"); return; }
	_bncu_mul_gfmatrixt_gfvec<<<nbg, ntb>>>(v->element, a->element, vb->element, a->row_dim, a->col_dim);
}
