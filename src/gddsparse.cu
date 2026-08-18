/********************************************************************************/
/* gddsparse.cu: double-double SPARSE (CSR) SpMV on GPU (CUDA)                   */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#include "gddsparse.h"
#include "gqd.cu"   // device gdd_real arithmetic

#ifndef GDDSP_NLIMB
#define GDDSP_NLIMB DDSIZE
#endif
typedef double gddsp_scalar;
__device__ static inline gdd_real gddsp_zero() { gdd_real z; gddsp_scalar *p = (gddsp_scalar *)&z; for(int l = 0; l < GDDSP_NLIMB; l++) p[l] = 0; return z; }

__host__ GDDSPMatrix init_gddspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                          const long int *row_ptr, const long int *col_idx, const gdd_real *val)
{
	GDDSPMatrix m = (gddspmatrix *)malloc(sizeof(gddspmatrix));
	if(m == NULL) { fprintf(stderr, "ERROR: init_gddspmatrix_dev\n"); return NULL; }
	m->row_dim = row_dim; m->col_dim = col_dim; m->nnz = nnz;
	m->row_ptr = NULL; m->col_idx = NULL; m->val = NULL;
	cudaMalloc((void **)&(m->row_ptr), (size_t)((row_dim + 1) * sizeof(long int)));
	cudaMalloc((void **)&(m->col_idx), (size_t)(nnz * sizeof(long int)));
	cudaMalloc((void **)&(m->val),     (size_t)(nnz * sizeof(gdd_real)));
	cudaMemcpy((void *)m->row_ptr, (void *)row_ptr, (size_t)((row_dim + 1) * sizeof(long int)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->col_idx, (void *)col_idx, (size_t)(nnz * sizeof(long int)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->val,     (void *)val,     (size_t)(nnz * sizeof(gdd_real)),  cudaMemcpyHostToDevice);
	return m;
}

__host__ void free_gddspmatrix_dev(GDDSPMatrix mat)
{ cudaFree(mat->row_ptr); cudaFree(mat->col_idx); cudaFree(mat->val); free(mat); }

// y := A * x  (one thread per row, grid-stride, dd accumulation)
__global__ void _bncu_spmv_gdd(gdd_real *y, const long int *row_ptr, const long int *col_idx,
                               const gdd_real *val, const gdd_real *x, long int n)
{
	long int i = threadIdx.x + blockIdx.x * blockDim.x;
	long int stride = (long int)blockDim.x * gridDim.x;
	while(i < n)
	{
		gdd_real s = gddsp_zero();
		long int kstart = row_ptr[i], kend = row_ptr[i + 1];
		for(long int k = kstart; k < kend; k++)
			s = s + val[k] * x[col_idx[k]];
		y[i] = s;
		i += stride;
	}
}
__host__ void mul_gddspmatrix(gdd_real *y_dev, GDDSPMatrix a, const gdd_real *x_dev, int nbg, int ntb)
{
	_bncu_spmv_gdd<<<nbg, ntb>>>(y_dev, a->row_ptr, a->col_idx, a->val, x_dev, a->row_dim);
}
