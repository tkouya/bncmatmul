/********************************************************************************/
/* gcdsparse.cu: native complex double SPARSE (CSR) SpMV on GPU (CUDA)           */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#include <cstdio>
#include <cstdlib>
#include "gcdsparse.h"

__host__ GCDSPMatrix init_gcdspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                          const long int *row_ptr, const long int *col_idx,
                                          const double *val_re, const double *val_im)
{
	GCDSPMatrix m = (gcdspmatrix *)malloc(sizeof(gcdspmatrix));
	if(m == NULL) { fprintf(stderr, "ERROR: init_gcdspmatrix_dev\n"); return NULL; }
	m->row_dim = row_dim; m->col_dim = col_dim; m->nnz = nnz;
	m->row_ptr = NULL; m->col_idx = NULL; m->val_re = NULL; m->val_im = NULL;
	cudaMalloc((void **)&(m->row_ptr), (size_t)((row_dim + 1) * sizeof(long int)));
	cudaMalloc((void **)&(m->col_idx), (size_t)(nnz * sizeof(long int)));
	cudaMalloc((void **)&(m->val_re),  (size_t)(nnz * sizeof(double)));
	cudaMalloc((void **)&(m->val_im),  (size_t)(nnz * sizeof(double)));
	cudaMemcpy((void *)m->row_ptr, (void *)row_ptr, (size_t)((row_dim + 1) * sizeof(long int)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->col_idx, (void *)col_idx, (size_t)(nnz * sizeof(long int)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->val_re,  (void *)val_re,  (size_t)(nnz * sizeof(double)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->val_im,  (void *)val_im,  (size_t)(nnz * sizeof(double)), cudaMemcpyHostToDevice);
	return m;
}
__host__ void free_gcdspmatrix_dev(GCDSPMatrix mat)
{ cudaFree(mat->row_ptr); cudaFree(mat->col_idx); cudaFree(mat->val_re); cudaFree(mat->val_im); free(mat); }

__global__ void _bncu_spmv_gcd(double *yre, double *yim, const long int *row_ptr, const long int *col_idx,
                               const double *vre, const double *vim, const double *xre, const double *xim, long int n)
{
	long int i = threadIdx.x + blockIdx.x * blockDim.x;
	long int stride = (long int)blockDim.x * gridDim.x;
	while(i < n)
	{
		double sre = 0.0, sim = 0.0;
		long int kstart = row_ptr[i], kend = row_ptr[i + 1];
		for(long int k = kstart; k < kend; k++)
		{
			long int c = col_idx[k];
			double vr = vre[k], vi = vim[k], xr = xre[c], xi = xim[c];
			sre += vr * xr - vi * xi;
			sim += vr * xi + vi * xr;
		}
		yre[i] = sre; yim[i] = sim;
		i += stride;
	}
}
__host__ void mul_gcdspmatrix(double *yre, double *yim, GCDSPMatrix a, const double *xre, const double *xim, int nbg, int ntb)
{
	_bncu_spmv_gcd<<<nbg, ntb>>>(yre, yim, a->row_ptr, a->col_idx, a->val_re, a->val_im, xre, xim, a->row_dim);
}
