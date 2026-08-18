/********************************************************************************/
/* gqdsparse.cu: double-double SPARSE (CSR) SpMV on GPU (CUDA)                   */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#include "gqdsparse.h"
#include "gqd.cu"   // device gqd_real arithmetic

#ifndef GQDSP_NLIMB
#define GQDSP_NLIMB QDSIZE
#endif
typedef double gqdsp_scalar;
__device__ static inline gqd_real gqdsp_zero() { gqd_real z; gqdsp_scalar *p = (gqdsp_scalar *)&z; for(int l = 0; l < GQDSP_NLIMB; l++) p[l] = 0; return z; }

__host__ GQDSPMatrix init_gqdspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                          const long int *row_ptr, const long int *col_idx, const gqd_real *val)
{
	GQDSPMatrix m = (gqdspmatrix *)malloc(sizeof(gqdspmatrix));
	if(m == NULL) { fprintf(stderr, "ERROR: init_gqdspmatrix_dev\n"); return NULL; }
	m->row_dim = row_dim; m->col_dim = col_dim; m->nnz = nnz;
	m->row_ptr = NULL; m->col_idx = NULL; m->val = NULL;
	cudaMalloc((void **)&(m->row_ptr), (size_t)((row_dim + 1) * sizeof(long int)));
	cudaMalloc((void **)&(m->col_idx), (size_t)(nnz * sizeof(long int)));
	cudaMalloc((void **)&(m->val),     (size_t)(nnz * sizeof(gqd_real)));
	cudaMemcpy((void *)m->row_ptr, (void *)row_ptr, (size_t)((row_dim + 1) * sizeof(long int)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->col_idx, (void *)col_idx, (size_t)(nnz * sizeof(long int)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->val,     (void *)val,     (size_t)(nnz * sizeof(gqd_real)),  cudaMemcpyHostToDevice);
	return m;
}

__host__ void free_gqdspmatrix_dev(GQDSPMatrix mat)
{ cudaFree(mat->row_ptr); cudaFree(mat->col_idx); cudaFree(mat->val); free(mat); }

// y := A * x  (one thread per row, grid-stride, dd accumulation)
__global__ void _bncu_spmv_gqd(gqd_real *y, const long int *row_ptr, const long int *col_idx,
                               const gqd_real *val, const gqd_real *x, long int n)
{
	long int i = threadIdx.x + blockIdx.x * blockDim.x;
	long int stride = (long int)blockDim.x * gridDim.x;
	while(i < n)
	{
		gqd_real s = gqdsp_zero();
		long int kstart = row_ptr[i], kend = row_ptr[i + 1];
		for(long int k = kstart; k < kend; k++)
			s = s + val[k] * x[col_idx[k]];
		y[i] = s;
		i += stride;
	}
}
__host__ void mul_gqdspmatrix(gqd_real *y_dev, GQDSPMatrix a, const gqd_real *x_dev, int nbg, int ntb)
{
	_bncu_spmv_gqd<<<nbg, ntb>>>(y_dev, a->row_ptr, a->col_idx, a->val, x_dev, a->row_dim);
}
