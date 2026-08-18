/********************************************************************************/
/* cgddsparse.cu: complex double-double SPARSE (CSR) SpMV on GPU (CUDA)          */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#include "cgddsparse.h"
#include "gqd.cu"   // device gdd_real arithmetic

#ifndef CGDDSP_NLIMB
#define CGDDSP_NLIMB DDSIZE
#endif
typedef double cgddsp_scalar;
__device__ static inline gdd_real cgddsp_zero() { gdd_real z; cgddsp_scalar *p = (cgddsp_scalar *)&z; for(int l = 0; l < CGDDSP_NLIMB; l++) p[l] = 0; return z; }

__host__ CGDDSPMatrix init_cgddspmatrix_dev(long int row_dim, long int col_dim, long int nnz,
                                            const long int *row_ptr, const long int *col_idx,
                                            const gdd_real *val_re, const gdd_real *val_im)
{
	CGDDSPMatrix m = (cgddspmatrix *)malloc(sizeof(cgddspmatrix));
	if(m == NULL) { fprintf(stderr, "ERROR: init_cgddspmatrix_dev\n"); return NULL; }
	m->row_dim = row_dim; m->col_dim = col_dim; m->nnz = nnz;
	m->row_ptr = NULL; m->col_idx = NULL; m->val_re = NULL; m->val_im = NULL;
	cudaMalloc((void **)&(m->row_ptr), (size_t)((row_dim + 1) * sizeof(long int)));
	cudaMalloc((void **)&(m->col_idx), (size_t)(nnz * sizeof(long int)));
	cudaMalloc((void **)&(m->val_re),  (size_t)(nnz * sizeof(gdd_real)));
	cudaMalloc((void **)&(m->val_im),  (size_t)(nnz * sizeof(gdd_real)));
	cudaMemcpy((void *)m->row_ptr, (void *)row_ptr, (size_t)((row_dim + 1) * sizeof(long int)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->col_idx, (void *)col_idx, (size_t)(nnz * sizeof(long int)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->val_re,  (void *)val_re,  (size_t)(nnz * sizeof(gdd_real)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)m->val_im,  (void *)val_im,  (size_t)(nnz * sizeof(gdd_real)), cudaMemcpyHostToDevice);
	return m;
}
__host__ void free_cgddspmatrix_dev(CGDDSPMatrix mat)
{ cudaFree(mat->row_ptr); cudaFree(mat->col_idx); cudaFree(mat->val_re); cudaFree(mat->val_im); free(mat); }

__global__ void _bncu_spmv_cgdd(gdd_real *yre, gdd_real *yim, const long int *row_ptr, const long int *col_idx,
                                const gdd_real *vre, const gdd_real *vim, const gdd_real *xre, const gdd_real *xim, long int n)
{
	long int i = threadIdx.x + blockIdx.x * blockDim.x;
	long int stride = (long int)blockDim.x * gridDim.x;
	while(i < n)
	{
		gdd_real sre = cgddsp_zero(), sim = cgddsp_zero();
		long int kstart = row_ptr[i], kend = row_ptr[i + 1];
		for(long int k = kstart; k < kend; k++)
		{
			long int c = col_idx[k];
			gdd_real vr = vre[k], vi = vim[k], xr = xre[c], xi = xim[c];
			sre = sre + (vr * xr - vi * xi);
			sim = sim + (vr * xi + vi * xr);
		}
		yre[i] = sre; yim[i] = sim;
		i += stride;
	}
}
__host__ void mul_cgddspmatrix(gdd_real *yre, gdd_real *yim, CGDDSPMatrix a, const gdd_real *xre, const gdd_real *xim, int nbg, int ntb)
{
	_bncu_spmv_cgdd<<<nbg, ntb>>>(yre, yim, a->row_ptr, a->col_idx, a->val_re, a->val_im, xre, xim, a->row_dim);
}
