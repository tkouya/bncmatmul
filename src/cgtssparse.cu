/********************************************************************************/
/* cgtssparse.cu: complex float-float SPARSE (CSR) SpMV on GPU (CUDA)          */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#include "cgtssparse.h"
#include "gqs.cu"   // device gts_real arithmetic

#ifndef CGTSSP_NLIMB
#define CGTSSP_NLIMB TSSIZE
#endif
typedef float cgtssp_scalar;
__device__ static inline gts_real cgtssp_zero() { gts_real z; cgtssp_scalar *p = (cgtssp_scalar *)&z; for(int l = 0; l < CGTSSP_NLIMB; l++) p[l] = 0; return z; }

__host__ CGTSSPMatrix init_cgtsspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                            const long int *row_ptr, const long int *col_idx,
                                            const gts_real *val_re, const gts_real *val_im)
{
	CGTSSPMatrix m = (cgtsspmatrix *)malloc(sizeof(cgtsspmatrix));
	if(m == NULL) { fprintf(stderr, "ERROR: init_cgtsspmatrix_dev\n"); return NULL; }
	m->row_dim = row_dim; m->col_dim = col_dim; m->nnz = nnz;
	m->row_ptr = NULL; m->col_idx = NULL; m->val_re = NULL; m->val_im = NULL;
	cudaMalloc((void **)&(m->row_ptr), (size_t)((row_dim + 1) * sizeof(long int)));
	cudaMalloc((void **)&(m->col_idx), (size_t)(nnz * sizeof(long int)));
	cudaMalloc((void **)&(m->val_re),  (size_t)(nnz * sizeof(gts_real)));
	cudaMalloc((void **)&(m->val_im),  (size_t)(nnz * sizeof(gts_real)));
	cudaMemcpy((void *)m->row_ptr, (void *)row_ptr, (size_t)((row_dim + 1) * sizeof(long int)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->col_idx, (void *)col_idx, (size_t)(nnz * sizeof(long int)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->val_re,  (void *)val_re,  (size_t)(nnz * sizeof(gts_real)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->val_im,  (void *)val_im,  (size_t)(nnz * sizeof(gts_real)), cudaMemcpyHostToDevice);
	return m;
}
__host__ void free_cgtsspmatrix_dev(CGTSSPMatrix mat)
{ cudaFree(mat->row_ptr); cudaFree(mat->col_idx); cudaFree(mat->val_re); cudaFree(mat->val_im); free(mat); }

__global__ void _bncu_spmv_cgts(gts_real *yre, gts_real *yim, const long int *row_ptr, const long int *col_idx,
                                const gts_real *vre, const gts_real *vim, const gts_real *xre, const gts_real *xim, long int n)
{
	long int i = threadIdx.x + blockIdx.x * blockDim.x;
	long int stride = (long int)blockDim.x * gridDim.x;
	while(i < n)
	{
		gts_real sre = cgtssp_zero(), sim = cgtssp_zero();
		long int kstart = row_ptr[i], kend = row_ptr[i + 1];
		for(long int k = kstart; k < kend; k++)
		{
			long int c = col_idx[k];
			gts_real vr = vre[k], vi = vim[k], xr = xre[c], xi = xim[c];
			sre = sre + (vr * xr - vi * xi);
			sim = sim + (vr * xi + vi * xr);
		}
		yre[i] = sre; yim[i] = sim;
		i += stride;
	}
}
__host__ void mul_cgtsspmatrix(gts_real *yre, gts_real *yim, CGTSSPMatrix a, const gts_real *xre, const gts_real *xim, int nbg, int ntb)
{
	_bncu_spmv_cgts<<<nbg, ntb>>>(yre, yim, a->row_ptr, a->col_idx, a->val_re, a->val_im, xre, xim, a->row_dim);
}
