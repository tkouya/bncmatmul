/********************************************************************************/
/* cgqdsparse.cu: complex double-double SPARSE (CSR) SpMV on GPU (CUDA)          */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#include "cgqdsparse.h"
#include "gqd.cu"   // device gqd_real arithmetic

#ifndef CGQDSP_NLIMB
#define CGQDSP_NLIMB QDSIZE
#endif
typedef double cgqdsp_scalar;
__device__ static inline gqd_real cgqdsp_zero() { gqd_real z; cgqdsp_scalar *p = (cgqdsp_scalar *)&z; for(int l = 0; l < CGQDSP_NLIMB; l++) p[l] = 0; return z; }

__host__ CGQDSPMatrix init_cgqdspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                            const long int *row_ptr, const long int *col_idx,
                                            const gqd_real *val_re, const gqd_real *val_im)
{
	CGQDSPMatrix m = (cgqdspmatrix *)malloc(sizeof(cgqdspmatrix));
	if(m == NULL) { fprintf(stderr, "ERROR: init_cgqdspmatrix_dev\n"); return NULL; }
	m->row_dim = row_dim; m->col_dim = col_dim; m->nnz = nnz;
	m->row_ptr = NULL; m->col_idx = NULL; m->val_re = NULL; m->val_im = NULL;
	cudaMalloc((void **)&(m->row_ptr), (size_t)((row_dim + 1) * sizeof(long int)));
	cudaMalloc((void **)&(m->col_idx), (size_t)(nnz * sizeof(long int)));
	cudaMalloc((void **)&(m->val_re),  (size_t)(nnz * sizeof(gqd_real)));
	cudaMalloc((void **)&(m->val_im),  (size_t)(nnz * sizeof(gqd_real)));
	cudaMemcpy((void *)m->row_ptr, (void *)row_ptr, (size_t)((row_dim + 1) * sizeof(long int)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->col_idx, (void *)col_idx, (size_t)(nnz * sizeof(long int)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->val_re,  (void *)val_re,  (size_t)(nnz * sizeof(gqd_real)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->val_im,  (void *)val_im,  (size_t)(nnz * sizeof(gqd_real)), cudaMemcpyHostToDevice);
	return m;
}
__host__ void free_cgqdspmatrix_dev(CGQDSPMatrix mat)
{ cudaFree(mat->row_ptr); cudaFree(mat->col_idx); cudaFree(mat->val_re); cudaFree(mat->val_im); free(mat); }

__global__ void _bncu_spmv_cgqd(gqd_real *yre, gqd_real *yim, const long int *row_ptr, const long int *col_idx,
                                const gqd_real *vre, const gqd_real *vim, const gqd_real *xre, const gqd_real *xim, long int n)
{
	long int i = threadIdx.x + blockIdx.x * blockDim.x;
	long int stride = (long int)blockDim.x * gridDim.x;
	while(i < n)
	{
		gqd_real sre = cgqdsp_zero(), sim = cgqdsp_zero();
		long int kstart = row_ptr[i], kend = row_ptr[i + 1];
		for(long int k = kstart; k < kend; k++)
		{
			long int c = col_idx[k];
			gqd_real vr = vre[k], vi = vim[k], xr = xre[c], xi = xim[c];
			sre = sre + (vr * xr - vi * xi);
			sim = sim + (vr * xi + vi * xr);
		}
		yre[i] = sre; yim[i] = sim;
		i += stride;
	}
}
__host__ void mul_cgqdspmatrix(gqd_real *yre, gqd_real *yim, CGQDSPMatrix a, const gqd_real *xre, const gqd_real *xim, int nbg, int ntb)
{
	_bncu_spmv_cgqd<<<nbg, ntb>>>(yre, yim, a->row_ptr, a->col_idx, a->val_re, a->val_im, xre, xim, a->row_dim);
}
