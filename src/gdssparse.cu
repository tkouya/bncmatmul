/********************************************************************************/
/* gdssparse.cu: float-float SPARSE (CSR) SpMV on GPU (CUDA)                   */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#include "gdssparse.h"
#include "gqs.cu"   // device gds_real arithmetic

#ifndef GDSSP_NLIMB
#define GDSSP_NLIMB DSSIZE
#endif
typedef float gdssp_scalar;
__device__ static inline gds_real gdssp_zero() { gds_real z; gdssp_scalar *p = (gdssp_scalar *)&z; for(int l = 0; l < GDSSP_NLIMB; l++) p[l] = 0; return z; }

__host__ GDSSPMatrix init_gdsspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                          const long int *row_ptr, const long int *col_idx, const gds_real *val)
{
	GDSSPMatrix m = (gdsspmatrix *)malloc(sizeof(gdsspmatrix));
	if(m == NULL) { fprintf(stderr, "ERROR: init_gdsspmatrix_dev\n"); return NULL; }
	m->row_dim = row_dim; m->col_dim = col_dim; m->nnz = nnz;
	m->row_ptr = NULL; m->col_idx = NULL; m->val = NULL;
	cudaMalloc((void **)&(m->row_ptr), (size_t)((row_dim + 1) * sizeof(long int)));
	cudaMalloc((void **)&(m->col_idx), (size_t)(nnz * sizeof(long int)));
	cudaMalloc((void **)&(m->val),     (size_t)(nnz * sizeof(gds_real)));
	cudaMemcpy((void *)m->row_ptr, (void *)row_ptr, (size_t)((row_dim + 1) * sizeof(long int)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->col_idx, (void *)col_idx, (size_t)(nnz * sizeof(long int)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->val,     (void *)val,     (size_t)(nnz * sizeof(gds_real)),  cudaMemcpyHostToDevice);
	return m;
}

__host__ void free_gdsspmatrix_dev(GDSSPMatrix mat)
{ cudaFree(mat->row_ptr); cudaFree(mat->col_idx); cudaFree(mat->val); free(mat); }

// y := A * x  (one thread per row, grid-stride, dd accumulation)
__global__ void _bncu_spmv_gds(gds_real *y, const long int *row_ptr, const long int *col_idx,
                               const gds_real *val, const gds_real *x, long int n)
{
	long int i = threadIdx.x + blockIdx.x * blockDim.x;
	long int stride = (long int)blockDim.x * gridDim.x;
	while(i < n)
	{
		gds_real s = gdssp_zero();
		long int kstart = row_ptr[i], kend = row_ptr[i + 1];
		for(long int k = kstart; k < kend; k++)
			s = s + val[k] * x[col_idx[k]];
		y[i] = s;
		i += stride;
	}
}
__host__ void mul_gdsspmatrix(gds_real *y_dev, GDSSPMatrix a, const gds_real *x_dev, int nbg, int ntb)
{
	_bncu_spmv_gds<<<nbg, ntb>>>(y_dev, a->row_ptr, a->col_idx, a->val, x_dev, a->row_dim);
}
