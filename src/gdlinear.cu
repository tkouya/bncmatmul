/********************************************************************************/
/* gdlinear.cu: Native double precision GPU Linear Computation Library with CUDA*/
/* Copyright (C) 2026 Tomonori Kouya                                            */
/*                                                                              */
/* This program is free software: you can redistribute it and/or modify it      */
/* under the terms of the GNU Lesser General Public License as published by the */
/* Free Software Foundation, either version 3 of the License or any later       */
/* version.                                                                     */
/********************************************************************************/

#include "gdlinear.h"

/*-------------------------------------------------------------------*/
/* GDVector                                                          */
/*-------------------------------------------------------------------*/
__host__ GDVector init_gdvector(long int dim)
{
	GDVector ret = (gdvector *)malloc(sizeof(gdvector));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one GDVector\n");
		return NULL;
	}
	ret->dim = dim;
	ret->element = (double *)calloc(dim, sizeof(double));
	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GDVector(dim = %ld)\n", dim);
		free(ret);
		return NULL;
	}
	return ret;
}

__host__ void free_gdvector(GDVector vec)
{
	free(vec->element);
	free(vec);
}

__host__ GDVector init_gdvector_dev(long int dim)
{
	GDVector ret = (gdvector *)malloc(sizeof(gdvector));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one GDVector\n");
		return NULL;
	}
	ret->element = NULL;
	ret->dim = dim;
	cudaMalloc((void **)&(ret->element), (size_t)(dim * sizeof(double)));
	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate GDVector(dim = %ld) on GPU\n", dim);
		free(ret);
		return NULL;
	}
	cudaMemset((void *)(ret->element), 0, (size_t)(dim * sizeof(double)));
	return ret;
}

__host__ void free_gdvector_dev(GDVector vec)
{
	cudaFree(vec->element);
	free(vec);
}

// gdvec on GPU := dvec (host DVector is padded: real_dim stride for element, but
// element[i] is contiguous for i<dim, so a packed copy is valid)
__host__ void subst_gdvector_dev_dvec(GDVector gdvec_dev, DVector dvec)
{
	long int dim = gdvec_dev->dim;
	cudaMemcpy((void *)gdvec_dev->element, (void *)dvec->element, (size_t)(dim * sizeof(double)), cudaMemcpyHostToDevice);
}

__host__ void subst_dvector_gdvec_dev(DVector dvec, GDVector gdvec_dev)
{
	long int dim = dvec->dim;
	cudaMemcpy((void *)dvec->element, (void *)gdvec_dev->element, (size_t)(dim * sizeof(double)), cudaMemcpyDeviceToHost);
}

__host__ void print_gdvector_dev(GDVector dev_vec)
{
	long int index, dim = dev_vec->dim;
	GDVector host_vec = init_gdvector(dim);
	cudaMemcpy((void *)(host_vec->element), (void *)(dev_vec->element), (size_t)(sizeof(double) * dim), cudaMemcpyDeviceToHost);
	for(index = 0; index < dim; index++)
		printf("%4ld: %25.17e\n", index, host_vec->element[index]);
	free_gdvector(host_vec);
}

/* c = a + b */
__global__ void _bncu_add_gdvector(double *c, double *a, double *b, long int dim)
{
	long int index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim) { c[index] = a[index] + b[index]; index += blockDim.x * gridDim.x; }
}
__host__ void add_gdvector_dev(GDVector c_dev, GDVector a_dev, GDVector b_dev, int nbg, int ntb)
{
	if((a_dev->dim != b_dev->dim) || (c_dev->dim != a_dev->dim))
	{ fprintf(stderr, "ERROR: add_gdvector_dev\n"); return; }
	_bncu_add_gdvector<<<nbg, ntb>>>(c_dev->element, a_dev->element, b_dev->element, c_dev->dim);
}

/* c = a - b */
__global__ void _bncu_sub_gdvector(double *c, double *a, double *b, long int dim)
{
	long int index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim) { c[index] = a[index] - b[index]; index += blockDim.x * gridDim.x; }
}
__host__ void sub_gdvector_dev(GDVector c_dev, GDVector a_dev, GDVector b_dev, int nbg, int ntb)
{
	if((a_dev->dim != b_dev->dim) || (c_dev->dim != a_dev->dim))
	{ fprintf(stderr, "ERROR: sub_gdvector_dev\n"); return; }
	_bncu_sub_gdvector<<<nbg, ntb>>>(c_dev->element, a_dev->element, b_dev->element, c_dev->dim);
}

/* c = val * a */
__global__ void _bncu_cmul_gdvector(double *ret, double val, double *a, long int dim)
{
	long int index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim) { ret[index] = val * a[index]; index += blockDim.x * gridDim.x; }
}
__host__ void cmul_gdvector_dev(GDVector c_dev, double val, GDVector a_dev, int nbg, int ntb)
{
	if(c_dev->dim != a_dev->dim) { fprintf(stderr, "ERROR: cmul_gdvector_dev\n"); return; }
	_bncu_cmul_gdvector<<<nbg, ntb>>>(c_dev->element, val, a_dev->element, c_dev->dim);
}

/* c := vec */
__global__ void _bncu_subst_gdvector(double *ret, double *vec, long int dim)
{
	long int index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim) { ret[index] = vec[index]; index += blockDim.x * gridDim.x; }
}
__host__ void subst_gdvector_dev(GDVector ret_dev, GDVector vec_dev, int nbg, int ntb)
{
	if(ret_dev->dim != vec_dev->dim) { fprintf(stderr, "ERROR: subst_gdvector_dev\n"); return; }
	_bncu_subst_gdvector<<<nbg, ntb>>>(ret_dev->element, vec_dev->element, ret_dev->dim);
}

/* c := 0 */
__global__ void _bncu_set0_gdvector(double *ret, long int dim)
{
	long int index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim) { ret[index] = 0.0; index += blockDim.x * gridDim.x; }
}
__host__ void set0_gdvector_dev(GDVector ret_dev, int nbg, int ntb)
{
	_bncu_set0_gdvector<<<nbg, ntb>>>(ret_dev->element, ret_dev->dim);
}

// element-wise helpers
__global__ void _bncu_mul_gdvector(double *ret, double *a, double *b, long int dim)
{
	long int index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim) { ret[index] = a[index] * b[index]; index += blockDim.x * gridDim.x; }
}
__global__ void _bncu_sqr_gdvector(double *ret, double *vec, long int dim)
{
	long int index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim) { ret[index] = vec[index] * vec[index]; index += blockDim.x * gridDim.x; }
}
__global__ void _bncu_sqrt_gdvector(double *ret, double *vec, long int dim)
{
	long int index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim) { ret[index] = sqrt(vec[index]); index += blockDim.x * gridDim.x; }
}
__global__ void _bncu_abs_gdvector(double *ret, double *vec, long int dim)
{
	long int index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < dim) { ret[index] = fabs(vec[index]); index += blockDim.x * gridDim.x; }
}

// reduction: sum
__global__ void _bncu_add_reduct_gdvector(double *ret, double *vec, long int dim)
{
	long int index, cache_index;
	__shared__ double cache[MAX_NUM_THREADS_PER_BLOCK];
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
__global__ void _bncu_max_reduct_gdvector(double *ret, double *vec, long int dim)
{
	long int index, cache_index;
	__shared__ double cache[MAX_NUM_THREADS_PER_BLOCK];
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
static void _gd_reduce(double *ret_dev, double *src, long int len, int nbg, int ntb, int do_max, int do_sqrt)
{
	double *c_dev, *block_cache;
	int block_dim, thread_dim;
	long int dim;
	cudaMalloc((void **)&c_dev, (size_t)(sizeof(double) * len));
	cudaMalloc((void **)&block_cache, (size_t)(sizeof(double) * nbg));
	cudaMemcpy((void *)c_dev, (void *)src, (size_t)(sizeof(double) * len), cudaMemcpyDeviceToDevice);

	block_dim = nbg; thread_dim = ntb; dim = len;
	while(block_dim >= 1)
	{
		if(do_max) _bncu_max_reduct_gdvector<<<block_dim, thread_dim>>>(block_cache, c_dev, dim);
		else       _bncu_add_reduct_gdvector<<<block_dim, thread_dim>>>(block_cache, c_dev, dim);
		if(block_dim <= 1) break;
		dim = block_dim;
		if(block_dim > thread_dim) block_dim = (block_dim / thread_dim) + 1;
		else block_dim = 1;
		cudaMemcpy((void *)c_dev, (void *)block_cache, (size_t)(sizeof(double) * dim), cudaMemcpyDeviceToDevice);
	}
	if(do_sqrt) _bncu_sqrt_gdvector<<<1, 1>>>(ret_dev, block_cache, 1);
	else        cudaMemcpy((void *)ret_dev, (void *)&(block_cache[0]), sizeof(double), cudaMemcpyDeviceToDevice);
	cudaFree(c_dev);
	cudaFree(block_cache);
}

/* (a, b) */
__host__ void ip_gdvector_dev(double *ret_dev, GDVector a_dev, GDVector b_dev, int nbg, int ntb)
{
	double *c_dev;
	if(a_dev->dim != b_dev->dim) { fprintf(stderr, "ERROR: ip_gdvector_dev\n"); return; }
	cudaMalloc((void **)&c_dev, (size_t)(sizeof(double) * a_dev->dim));
	_bncu_mul_gdvector<<<nbg, ntb>>>(c_dev, a_dev->element, b_dev->element, a_dev->dim);
	_gd_reduce(ret_dev, c_dev, a_dev->dim, nbg, ntb, 0, 0);
	cudaFree(c_dev);
}

__host__ void norm2_gdvector_dev(double *ret_dev, GDVector a_dev, int nbg, int ntb)
{
	double *c_dev;
	if(a_dev->dim <= 0) { fprintf(stderr, "ERROR: norm2_gdvector_dev\n"); return; }
	cudaMalloc((void **)&c_dev, (size_t)(sizeof(double) * a_dev->dim));
	_bncu_sqr_gdvector<<<nbg, ntb>>>(c_dev, a_dev->element, a_dev->dim);
	_gd_reduce(ret_dev, c_dev, a_dev->dim, nbg, ntb, 0, 1);
	cudaFree(c_dev);
}

__host__ void norm1_gdvector_dev(double *ret_dev, GDVector a_dev, int nbg, int ntb)
{
	double *c_dev;
	if(a_dev->dim <= 0) { fprintf(stderr, "ERROR: norm1_gdvector_dev\n"); return; }
	cudaMalloc((void **)&c_dev, (size_t)(sizeof(double) * a_dev->dim));
	_bncu_abs_gdvector<<<nbg, ntb>>>(c_dev, a_dev->element, a_dev->dim);
	_gd_reduce(ret_dev, c_dev, a_dev->dim, nbg, ntb, 0, 0);
	cudaFree(c_dev);
}

__host__ void normi_gdvector_dev(double *ret_dev, GDVector a_dev, int nbg, int ntb)
{
	double *c_dev;
	if(a_dev->dim <= 0) { fprintf(stderr, "ERROR: normi_gdvector_dev\n"); return; }
	cudaMalloc((void **)&c_dev, (size_t)(sizeof(double) * a_dev->dim));
	_bncu_abs_gdvector<<<nbg, ntb>>>(c_dev, a_dev->element, a_dev->dim);
	_gd_reduce(ret_dev, c_dev, a_dev->dim, nbg, ntb, 1, 0);
	cudaFree(c_dev);
}

/*-------------------------------------------------------------------*/
/* GDMatrix                                                          */
/*-------------------------------------------------------------------*/
__host__ GDMatrix init_gdmatrix(long int row_dim, long int col_dim)
{
	GDMatrix ret = (gdmatrix *)malloc(sizeof(gdmatrix));
	if(ret == NULL) { fprintf(stderr, "ERROR: cannot allocate one GDMatrix\n"); return NULL; }
	ret->row_dim = row_dim; ret->col_dim = col_dim;
	ret->element = (double *)calloc(row_dim * col_dim, sizeof(double));
	if(ret->element == NULL)
	{ fprintf(stderr, "ERROR: cannot allocate GDMatrix(%ld, %ld)\n", row_dim, col_dim); free(ret); return NULL; }
	return ret;
}

__host__ void free_gdmatrix(GDMatrix mat) { free(mat->element); free(mat); }

__host__ GDMatrix init_gdmatrix_dev(long int row_dim, long int col_dim)
{
	GDMatrix ret = (gdmatrix *)malloc(sizeof(gdmatrix));
	if(ret == NULL) { fprintf(stderr, "ERROR: cannot allocate one GDMatrix\n"); return NULL; }
	ret->row_dim = row_dim; ret->col_dim = col_dim; ret->element = NULL;
	cudaMalloc((void **)&(ret->element), (size_t)(row_dim * col_dim * sizeof(double)));
	if(ret->element == NULL)
	{ fprintf(stderr, "ERROR: cannot allocate GDMatrix(%ld, %ld) on GPU\n", row_dim, col_dim); free(ret); return NULL; }
	cudaMemset((void *)(ret->element), 0, (size_t)(row_dim * col_dim * sizeof(double)));
	return ret;
}

__host__ void free_gdmatrix_dev(GDMatrix mat) { cudaFree(mat->element); free(mat); }

__host__ void print_gdmatrix_dev(GDMatrix mat)
{
	long int i, j, row_dim = mat->row_dim, col_dim = mat->col_dim;
	GDMatrix h = init_gdmatrix(row_dim, col_dim);
	cudaMemcpy((void *)h->element, (void *)mat->element, (size_t)(sizeof(double) * row_dim * col_dim), cudaMemcpyDeviceToHost);
	for(i = 0; i < row_dim; i++)
		for(j = 0; j < col_dim; j++)
			printf("%ld, %ld: %25.17e\n", i, j, h->element[i * col_dim + j]);
	free_gdmatrix(h);
}

// matrix multiplication ret := A * B
__global__ void _bncu_mul_gdmatrix(double *ret, long int row_dim, long int col_dim, long int mid_dim, double *a, double *b)
{
	long int i, j, k;
	double tmp;
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
__host__ void mul_gdmatrix_dev(GDMatrix ret_dev, GDMatrix a_dev, GDMatrix b_dev, int nbg, int ntb)
{
	if((ret_dev->row_dim != a_dev->row_dim) || (ret_dev->col_dim != b_dev->col_dim) || (a_dev->col_dim != b_dev->row_dim))
	{ fprintf(stderr, "ERROR: mul_gdmatrix_dev\n"); return; }
	_bncu_mul_gdmatrix<<<nbg, ntb>>>(ret_dev->element, ret_dev->row_dim, ret_dev->col_dim, a_dev->col_dim, a_dev->element, b_dev->element);
}

// gdmat on GPU := dmat (host DMatrix uses real_col_dim stride -> pack into col_dim)
__host__ void subst_gdmatrix_dev_dmat(GDMatrix gdmat_dev, DMatrix dmat)
{
	long int i, j, row_dim = gdmat_dev->row_dim, col_dim = gdmat_dev->col_dim;
	GDMatrix h = init_gdmatrix(row_dim, col_dim);
	for(i = 0; i < row_dim; i++)
		for(j = 0; j < col_dim; j++)
			h->element[i * col_dim + j] = get_dmatrix_ij(dmat, i, j);
	cudaMemcpy((void *)gdmat_dev->element, (void *)h->element, (size_t)(sizeof(double) * row_dim * col_dim), cudaMemcpyHostToDevice);
	free_gdmatrix(h);
}

__host__ void subst_dmatrix_gdmat_dev(DMatrix dmat, GDMatrix gdmat_dev)
{
	long int i, j, row_dim = gdmat_dev->row_dim, col_dim = gdmat_dev->col_dim;
	GDMatrix h = init_gdmatrix(row_dim, col_dim);
	cudaMemcpy((void *)h->element, (void *)gdmat_dev->element, (size_t)(sizeof(double) * row_dim * col_dim), cudaMemcpyDeviceToHost);
	for(i = 0; i < row_dim; i++)
		for(j = 0; j < col_dim; j++)
			set_dmatrix_ij(dmat, i, j, h->element[i * col_dim + j]);
	free_gdmatrix(h);
}

// Frobenius norm
__host__ void normf_gdmatrix_dev(double *ret_dev, GDMatrix mat_dev, int nbg, int ntb)
{
	double *tmp_vec;
	long int total = mat_dev->row_dim * mat_dev->col_dim;
	cudaMalloc((void **)&tmp_vec, (size_t)(sizeof(double) * total));
	_bncu_sqr_gdvector<<<nbg, ntb>>>(tmp_vec, mat_dev->element, total);
	_gd_reduce(ret_dev, tmp_vec, total, nbg, ntb, 0, 1);
	cudaFree(tmp_vec);
}

/* c := a + b */
__host__ void add_gdmatrix_dev(GDMatrix c_dev, GDMatrix a_dev, GDMatrix b_dev, int nbg, int ntb)
{
	if((a_dev->row_dim != c_dev->row_dim) || (a_dev->col_dim != c_dev->col_dim) ||
	   (b_dev->row_dim != c_dev->row_dim) || (b_dev->col_dim != c_dev->col_dim))
	{ fprintf(stderr, "ERROR: add_gdmatrix_dev\n"); return; }
	_bncu_add_gdvector<<<nbg, ntb>>>(c_dev->element, a_dev->element, b_dev->element, c_dev->row_dim * c_dev->col_dim);
}

/* c := a - b */
__host__ void sub_gdmatrix_dev(GDMatrix c_dev, GDMatrix a_dev, GDMatrix b_dev, int nbg, int ntb)
{
	if((a_dev->row_dim != c_dev->row_dim) || (a_dev->col_dim != c_dev->col_dim) ||
	   (b_dev->row_dim != c_dev->row_dim) || (b_dev->col_dim != c_dev->col_dim))
	{ fprintf(stderr, "ERROR: sub_gdmatrix_dev\n"); return; }
	_bncu_sub_gdvector<<<nbg, ntb>>>(c_dev->element, a_dev->element, b_dev->element, c_dev->row_dim * c_dev->col_dim);
}

/* c := sc * a */
__host__ void cmul_gdmatrix_dev(GDMatrix c_dev, double sc, GDMatrix a_dev, int nbg, int ntb)
{
	if((a_dev->row_dim != c_dev->row_dim) || (a_dev->col_dim != c_dev->col_dim))
	{ fprintf(stderr, "ERROR: cmul_gdmatrix_dev\n"); return; }
	_bncu_cmul_gdvector<<<nbg, ntb>>>(c_dev->element, sc, a_dev->element, c_dev->row_dim * c_dev->col_dim);
}

/* c = a^T */
__global__ void _bncu_transpose_gdmatrix(double *c, double *a, long int row_dim, long int col_dim)
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
__host__ void transpose_gdmatrix_dev(GDMatrix c_dev, GDMatrix a_dev, int nbg, int ntb)
{
	if((c_dev->row_dim != a_dev->col_dim) || (c_dev->col_dim != a_dev->row_dim))
	{ fprintf(stderr, "ERROR: transpose_gdmatrix_dev\n"); return; }
	_bncu_transpose_gdmatrix<<<nbg, ntb>>>(c_dev->element, a_dev->element, c_dev->row_dim, c_dev->col_dim);
}

/* c := a */
__host__ void subst_gdmatrix_dev(GDMatrix c_dev, GDMatrix a_dev, int nbg, int ntb)
{
	if((c_dev->row_dim != a_dev->row_dim) || (c_dev->col_dim != a_dev->col_dim))
	{ fprintf(stderr, "ERROR: subst_gdmatrix_dev\n"); return; }
	_bncu_subst_gdvector<<<nbg, ntb>>>(c_dev->element, a_dev->element, c_dev->row_dim * c_dev->col_dim);
}

/* c := 0 */
__host__ void set0_gdmatrix_dev(GDMatrix c_dev, int nbg, int ntb)
{
	_bncu_set0_gdvector<<<nbg, ntb>>>(c_dev->element, c_dev->row_dim * c_dev->col_dim);
}

/* set (i, j)-element on GPU */
__host__ void set_gdmatrix_ij_dev(GDMatrix mat_dev, long int row_index, long int col_index, double val)
{
	if((row_index >= 0) && (row_index < mat_dev->row_dim) && (col_index >= 0) && (col_index < mat_dev->col_dim))
		cudaMemcpy((void *)&(mat_dev->element[row_index * mat_dev->col_dim + col_index]), (void *)&val, sizeof(double), cudaMemcpyHostToDevice);
}

/* c := I */
__host__ void setI_gdmatrix_dev(GDMatrix c_dev, int nbg, int ntb)
{
	long int i;
	_bncu_set0_gdvector<<<nbg, ntb>>>(c_dev->element, c_dev->row_dim * c_dev->col_dim);
	for(i = 0; i < c_dev->row_dim; i++)
		set_gdmatrix_ij_dev(c_dev, i, i, 1.0);
}

/* v := a * vb */
__global__ void _bncu_mul_gdmatrix_gdvec(double *v, double *a, double *vb, long int row_dim, long int col_dim)
{
	long int j, index;
	double tmp;
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
__host__ void mul_gdmatrix_gdvec(GDVector v, GDMatrix a, GDVector vb, int nbg, int ntb)
{
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{ fprintf(stderr, "ERROR: mul_gdmatrix_gdvec\n"); return; }
	_bncu_mul_gdmatrix_gdvec<<<nbg, ntb>>>(v->element, a->element, vb->element, a->row_dim, a->col_dim);
}

/* v := a^T * vb */
__global__ void _bncu_mul_gdmatrixt_gdvec(double *v, double *a, double *vb, long int row_dim, long int col_dim)
{
	long int j, index;
	double tmp;
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
__host__ void mul_gdmatrixt_gdvec(GDVector v, GDMatrix a, GDVector vb, int nbg, int ntb)
{
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{ fprintf(stderr, "ERROR: mul_gdmatrixt_gdvec\n"); return; }
	_bncu_mul_gdmatrixt_gdvec<<<nbg, ntb>>>(v->element, a->element, vb->element, a->row_dim, a->col_dim);
}
