/********************************************************************************/
/* gcfsparse.cu: native complex float SPARSE (CSR) SpMV on GPU (CUDA)           */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#include <cstdio>
#include <cstdlib>
#include "gcfsparse.h"

__host__ GCFSPMatrix init_gcfspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                          const long int *row_ptr, const long int *col_idx,
                                          const float *val_re, const float *val_im)
{
	GCFSPMatrix m = (gcfspmatrix *)malloc(sizeof(gcfspmatrix));
	if(m == NULL) { fprintf(stderr, "ERROR: init_gcfspmatrix_dev\n"); return NULL; }
	m->row_dim = row_dim; m->col_dim = col_dim; m->nnz = nnz;
	m->row_ptr = NULL; m->col_idx = NULL; m->val_re = NULL; m->val_im = NULL;
	cudaMalloc((void **)&(m->row_ptr), (size_t)((row_dim + 1) * sizeof(long int)));
	cudaMalloc((void **)&(m->col_idx), (size_t)(nnz * sizeof(long int)));
	cudaMalloc((void **)&(m->val_re),  (size_t)(nnz * sizeof(float)));
	cudaMalloc((void **)&(m->val_im),  (size_t)(nnz * sizeof(float)));
	cudaMemcpy((void *)m->row_ptr, (void *)row_ptr, (size_t)((row_dim + 1) * sizeof(long int)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->col_idx, (void *)col_idx, (size_t)(nnz * sizeof(long int)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->val_re,  (void *)val_re,  (size_t)(nnz * sizeof(float)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->val_im,  (void *)val_im,  (size_t)(nnz * sizeof(float)), cudaMemcpyHostToDevice);
	return m;
}
__host__ void free_gcfspmatrix_dev(GCFSPMatrix mat)
{ cudaFree(mat->row_ptr); cudaFree(mat->col_idx); cudaFree(mat->val_re); cudaFree(mat->val_im); free(mat); }

__global__ void _bncu_spmv_gcf(float *yre, float *yim, const long int *row_ptr, const long int *col_idx,
                               const float *vre, const float *vim, const float *xre, const float *xim, long int n)
{
	long int i = threadIdx.x + blockIdx.x * blockDim.x;
	long int stride = (long int)blockDim.x * gridDim.x;
	while(i < n)
	{
		float sre = 0.0, sim = 0.0;
		long int kstart = row_ptr[i], kend = row_ptr[i + 1];
		for(long int k = kstart; k < kend; k++)
		{
			long int c = col_idx[k];
			float vr = vre[k], vi = vim[k], xr = xre[c], xi = xim[c];
			sre += vr * xr - vi * xi;
			sim += vr * xi + vi * xr;
		}
		yre[i] = sre; yim[i] = sim;
		i += stride;
	}
}
__host__ void mul_gcfspmatrix(float *yre, float *yim, GCFSPMatrix a, const float *xre, const float *xim, int nbg, int ntb)
{
	_bncu_spmv_gcf<<<nbg, ntb>>>(yre, yim, a->row_ptr, a->col_idx, a->val_re, a->val_im, xre, xim, a->row_dim);
}
