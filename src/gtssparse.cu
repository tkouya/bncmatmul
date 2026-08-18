/********************************************************************************/
/* gtssparse.cu: float-float SPARSE (CSR) SpMV on GPU (CUDA)                   */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#include "gtssparse.h"
#include "gqs.cu"   // device gts_real arithmetic

#ifndef GTSSP_NLIMB
#define GTSSP_NLIMB TSSIZE
#endif
typedef float gtssp_scalar;
__device__ static inline gts_real gtssp_zero() { gts_real z; gtssp_scalar *p = (gtssp_scalar *)&z; for(int l = 0; l < GTSSP_NLIMB; l++) p[l] = 0; return z; }

__host__ GTSSPMatrix init_gtsspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                          const long int *row_ptr, const long int *col_idx, const gts_real *val)
{
	GTSSPMatrix m = (gtsspmatrix *)malloc(sizeof(gtsspmatrix));
	if(m == NULL) { fprintf(stderr, "ERROR: init_gtsspmatrix_dev\n"); return NULL; }
	m->row_dim = row_dim; m->col_dim = col_dim; m->nnz = nnz;
	m->row_ptr = NULL; m->col_idx = NULL; m->val = NULL;
	cudaMalloc((void **)&(m->row_ptr), (size_t)((row_dim + 1) * sizeof(long int)));
	cudaMalloc((void **)&(m->col_idx), (size_t)(nnz * sizeof(long int)));
	cudaMalloc((void **)&(m->val),     (size_t)(nnz * sizeof(gts_real)));
	cudaMemcpy((void *)m->row_ptr, (void *)row_ptr, (size_t)((row_dim + 1) * sizeof(long int)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->col_idx, (void *)col_idx, (size_t)(nnz * sizeof(long int)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->val,     (void *)val,     (size_t)(nnz * sizeof(gts_real)),  cudaMemcpyHostToDevice);
	return m;
}

__host__ void free_gtsspmatrix_dev(GTSSPMatrix mat)
{ cudaFree(mat->row_ptr); cudaFree(mat->col_idx); cudaFree(mat->val); free(mat); }

// y := A * x  (one thread per row, grid-stride, dd accumulation)
__global__ void _bncu_spmv_gts(gts_real *y, const long int *row_ptr, const long int *col_idx,
                               const gts_real *val, const gts_real *x, long int n)
{
	long int i = threadIdx.x + blockIdx.x * blockDim.x;
	long int stride = (long int)blockDim.x * gridDim.x;
	while(i < n)
	{
		gts_real s = gtssp_zero();
		long int kstart = row_ptr[i], kend = row_ptr[i + 1];
		for(long int k = kstart; k < kend; k++)
			s = s + val[k] * x[col_idx[k]];
		y[i] = s;
		i += stride;
	}
}
__host__ void mul_gtsspmatrix(gts_real *y_dev, GTSSPMatrix a, const gts_real *x_dev, int nbg, int ntb)
{
	_bncu_spmv_gts<<<nbg, ntb>>>(y_dev, a->row_ptr, a->col_idx, a->val, x_dev, a->row_dim);
}
