/********************************************************************************/
/* gdsparse.cu: Native double precision SPARSE (CSR) SpMV on GPU (CUDA)          */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#include "gdsparse.h"

__host__ GDSPMatrix init_gdspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                        const long int *row_ptr, const long int *col_idx, const double *val)
{
	GDSPMatrix m = (gdspmatrix *)malloc(sizeof(gdspmatrix));
	if(m == NULL) { fprintf(stderr, "ERROR: init_gdspmatrix_dev\n"); return NULL; }
	m->row_dim = row_dim; m->col_dim = col_dim; m->nnz = nnz;
	m->row_ptr = NULL; m->col_idx = NULL; m->val = NULL;
	cudaMalloc((void **)&(m->row_ptr), (size_t)((row_dim + 1) * sizeof(long int)));
	cudaMalloc((void **)&(m->col_idx), (size_t)(nnz * sizeof(long int)));
	cudaMalloc((void **)&(m->val),     (size_t)(nnz * sizeof(double)));
	cudaMemcpy((void *)m->row_ptr, (void *)row_ptr, (size_t)((row_dim + 1) * sizeof(long int)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->col_idx, (void *)col_idx, (size_t)(nnz * sizeof(long int)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->val,     (void *)val,     (size_t)(nnz * sizeof(double)),   cudaMemcpyHostToDevice);
	return m;
}

__host__ void free_gdspmatrix_dev(GDSPMatrix mat)
{
	cudaFree(mat->row_ptr); cudaFree(mat->col_idx); cudaFree(mat->val); free(mat);
}

// y := A * x  (one thread per row, grid-stride)
__global__ void _bncu_spmv_gd(double *y, const long int *row_ptr, const long int *col_idx,
                              const double *val, const double *x, long int n)
{
	long int i = threadIdx.x + blockIdx.x * blockDim.x;
	long int stride = (long int)blockDim.x * gridDim.x;
	while(i < n)
	{
		double s = 0.0;
		long int kstart = row_ptr[i], kend = row_ptr[i + 1];
		for(long int k = kstart; k < kend; k++)
			s += val[k] * x[col_idx[k]];
		y[i] = s;
		i += stride;
	}
}
__host__ void mul_gdspmatrix_gdvec(GDVector ret, GDSPMatrix a, GDVector x, int nbg, int ntb)
{
	if((ret->dim < a->row_dim) || (x->dim != a->col_dim))
	{ fprintf(stderr, "ERROR: mul_gdspmatrix_gdvec dim\n"); return; }
	_bncu_spmv_gd<<<nbg, ntb>>>(ret->element, a->row_ptr, a->col_idx, a->val, x->element, a->row_dim);
}

// y := A^T * x  (scatter with atomicAdd; y must be pre-zeroed)
__global__ void _bncu_spmvt_gd(double *y, const long int *row_ptr, const long int *col_idx,
                               const double *val, const double *x, long int n)
{
	long int i = threadIdx.x + blockIdx.x * blockDim.x;
	long int stride = (long int)blockDim.x * gridDim.x;
	while(i < n)
	{
		double xi = x[i];
		long int kstart = row_ptr[i], kend = row_ptr[i + 1];
		for(long int k = kstart; k < kend; k++)
			atomicAdd(&y[col_idx[k]], val[k] * xi);
		i += stride;
	}
}
__host__ void mul_gdspmatrixt_gdvec(GDVector ret, GDSPMatrix a, GDVector x, int nbg, int ntb)
{
	if((ret->dim < a->col_dim) || (x->dim != a->row_dim))
	{ fprintf(stderr, "ERROR: mul_gdspmatrixt_gdvec dim\n"); return; }
	cudaMemset((void *)ret->element, 0, (size_t)(a->col_dim * sizeof(double)));
	_bncu_spmvt_gd<<<nbg, ntb>>>(ret->element, a->row_ptr, a->col_idx, a->val, x->element, a->row_dim);
}
