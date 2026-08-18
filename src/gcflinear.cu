/********************************************************************************/
/* gcflinear.cu: Native float-precision COMPLEX GPU Linear Computation (CUDA)   */
/*   complex as separate real/imag float arrays (SoA).                          */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#include "gcflinear.h"

// C99 <complex.h> macros (creal/cimag/_Complex_I) are unavailable in nvcc C++ mode;
// use the GCC complex extension via these helpers.
static inline double _Complex _mkc(float re, float im) { double _Complex z; __real__ z = re; __imag__ z = im; return z; }
#define _cre(z) (__real__(z))
#define _cim(z) (__imag__(z))

/*------------------------- GCFVector -------------------------*/
__host__ GCFVector init_gcfvector(long int dim)
{
	GCFVector r = (gcfvector *)malloc(sizeof(gcfvector));
	if(r == NULL) { fprintf(stderr, "ERROR: init_gcfvector\n"); return NULL; }
	r->dim = dim;
	r->re = (float *)calloc(dim, sizeof(float));
	r->im = (float *)calloc(dim, sizeof(float));
	if(r->re == NULL || r->im == NULL) { fprintf(stderr, "ERROR: init_gcfvector alloc\n"); return NULL; }
	return r;
}
__host__ void free_gcfvector(GCFVector v) { free(v->re); free(v->im); free(v); }

__host__ GCFVector init_gcfvector_dev(long int dim)
{
	GCFVector r = (gcfvector *)malloc(sizeof(gcfvector));
	if(r == NULL) { fprintf(stderr, "ERROR: init_gcfvector_dev\n"); return NULL; }
	r->dim = dim; r->re = NULL; r->im = NULL;
	cudaMalloc((void **)&(r->re), (size_t)(dim * sizeof(float)));
	cudaMalloc((void **)&(r->im), (size_t)(dim * sizeof(float)));
	cudaMemset((void *)(r->re), 0, (size_t)(dim * sizeof(float)));
	cudaMemset((void *)(r->im), 0, (size_t)(dim * sizeof(float)));
	return r;
}
__host__ void free_gcfvector_dev(GCFVector v) { cudaFree(v->re); cudaFree(v->im); free(v); }

__host__ void subst_gcfvector_dev_cdvec(GCFVector gcfvec_dev, CDVector cdvec)
{
	long int i, dim = gcfvec_dev->dim;
	GCFVector h = init_gcfvector(dim);
	for(i = 0; i < dim; i++)
	{
		double _Complex z = get_cdvector_i(cdvec, i);
		h->re[i] = _cre(z); h->im[i] = _cim(z);
	}
	cudaMemcpy((void *)gcfvec_dev->re, (void *)h->re, (size_t)(dim * sizeof(float)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)gcfvec_dev->im, (void *)h->im, (size_t)(dim * sizeof(float)), cudaMemcpyHostToDevice);
	free_gcfvector(h);
}
__host__ void subst_cdvector_gcfvec_dev(CDVector cdvec, GCFVector gcfvec_dev)
{
	long int i, dim = gcfvec_dev->dim;
	GCFVector h = init_gcfvector(dim);
	cudaMemcpy((void *)h->re, (void *)gcfvec_dev->re, (size_t)(dim * sizeof(float)), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)gcfvec_dev->im, (size_t)(dim * sizeof(float)), cudaMemcpyDeviceToHost);
	for(i = 0; i < dim; i++)
		set_cdvector_i(cdvec, i, _mkc(h->re[i], h->im[i]));
	free_gcfvector(h);
}
__host__ void print_gcfvector_dev(GCFVector dev_vec)
{
	long int i, dim = dev_vec->dim;
	GCFVector h = init_gcfvector(dim);
	cudaMemcpy((void *)h->re, (void *)dev_vec->re, (size_t)(dim * sizeof(float)), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)dev_vec->im, (size_t)(dim * sizeof(float)), cudaMemcpyDeviceToHost);
	for(i = 0; i < dim; i++) printf("%4ld: %25.17e %+25.17e i\n", i, h->re[i], h->im[i]);
	free_gcfvector(h);
}

__global__ void _bncu_add_gcfvector(float *cre, float *cim, float *are, float *aim, float *bre, float *bim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx] + bre[idx]; cim[idx] = aim[idx] + bim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void add_gcfvector_dev(GCFVector c, GCFVector a, GCFVector b, int nbg, int ntb)
{
	if((a->dim != b->dim) || (c->dim != a->dim)) { fprintf(stderr, "ERROR: add_gcfvector_dev\n"); return; }
	_bncu_add_gcfvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->dim);
}

__global__ void _bncu_sub_gcfvector(float *cre, float *cim, float *are, float *aim, float *bre, float *bim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx] - bre[idx]; cim[idx] = aim[idx] - bim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void sub_gcfvector_dev(GCFVector c, GCFVector a, GCFVector b, int nbg, int ntb)
{
	if((a->dim != b->dim) || (c->dim != a->dim)) { fprintf(stderr, "ERROR: sub_gcfvector_dev\n"); return; }
	_bncu_sub_gcfvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->dim);
}

__global__ void _bncu_cmul_gcfvector(float *cre, float *cim, float vre, float vim, float *are, float *aim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim)
	{
		float ar = are[idx], ai = aim[idx];
		cre[idx] = vre * ar - vim * ai;
		cim[idx] = vre * ai + vim * ar;
		idx += blockDim.x * gridDim.x;
	}
}
__host__ void cmul_gcfvector_dev(GCFVector c, float vre, float vim, GCFVector a, int nbg, int ntb)
{
	if(c->dim != a->dim) { fprintf(stderr, "ERROR: cmul_gcfvector_dev\n"); return; }
	_bncu_cmul_gcfvector<<<nbg, ntb>>>(c->re, c->im, vre, vim, a->re, a->im, c->dim);
}

__global__ void _bncu_subst_gcfvector(float *cre, float *cim, float *are, float *aim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx]; cim[idx] = aim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void subst_gcfvector_dev(GCFVector ret, GCFVector vec, int nbg, int ntb)
{
	if(ret->dim != vec->dim) { fprintf(stderr, "ERROR: subst_gcfvector_dev\n"); return; }
	_bncu_subst_gcfvector<<<nbg, ntb>>>(ret->re, ret->im, vec->re, vec->im, ret->dim);
}

__global__ void _bncu_set0_gcfvector(float *cre, float *cim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = 0.0; cim[idx] = 0.0; idx += blockDim.x * gridDim.x; }
}
__host__ void set0_gcfvector_dev(GCFVector ret, int nbg, int ntb)
{
	_bncu_set0_gcfvector<<<nbg, ntb>>>(ret->re, ret->im, ret->dim);
}

// element-wise conj(a)*b -> (cre,cim); and |a|^2 -> sre
__global__ void _bncu_conjmul_gcfvector(float *cre, float *cim, float *are, float *aim, float *bre, float *bim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim)
	{
		float ar = are[idx], ai = aim[idx], br = bre[idx], bi = bim[idx];
		// conj(a)*b = (ar - i ai)(br + i bi) = (ar*br + ai*bi) + i(ar*bi - ai*br)
		cre[idx] = ar * br + ai * bi;
		cim[idx] = ar * bi - ai * br;
		idx += blockDim.x * gridDim.x;
	}
}
__global__ void _bncu_absq_gcfvector(float *sre, float *are, float *aim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { sre[idx] = are[idx] * are[idx] + aim[idx] * aim[idx]; idx += blockDim.x * gridDim.x; }
}

// real-sum reduction (used for |.|^2 accumulation)
__global__ void _bncu_add_reduct_double(float *ret, float *vec, long int dim)
{
	long int index, cache_index;
	__shared__ float cache[MAX_NUM_THREADS_PER_BLOCK];
	index = threadIdx.x + blockIdx.x * blockDim.x;
	cache_index = threadIdx.x;
	cache[cache_index] = 0.0;
	while(index < dim) { cache[cache_index] += vec[index]; index += blockDim.x * gridDim.x; }
	__syncthreads();
	index = blockDim.x / 2;
	while(index > 0) { if(cache_index < index) cache[cache_index] += cache[cache_index + index]; __syncthreads(); index /= 2; }
	if(cache_index == 0) ret[blockIdx.x] = cache[0];
}
__global__ void _bncu_sqrt1(float *ret, float *src) { ret[0] = sqrt(src[0]); }

// reduce-sum a device array src(len) into ret_dev (1 elem); do_sqrt for norm
static void _gcf_reduce_sum(float *ret_dev, float *src, long int len, int nbg, int ntb, int do_sqrt)
{
	float *c_dev, *block_cache;
	int block_dim = nbg, thread_dim = ntb; long int dim = len;
	cudaMalloc((void **)&c_dev, (size_t)(sizeof(float) * len));
	cudaMalloc((void **)&block_cache, (size_t)(sizeof(float) * nbg));
	cudaMemcpy((void *)c_dev, (void *)src, (size_t)(sizeof(float) * len), cudaMemcpyDeviceToDevice);
	while(block_dim >= 1)
	{
		_bncu_add_reduct_double<<<block_dim, thread_dim>>>(block_cache, c_dev, dim);
		if(block_dim <= 1) break;
		dim = block_dim;
		if(block_dim > thread_dim) block_dim = (block_dim / thread_dim) + 1; else block_dim = 1;
		cudaMemcpy((void *)c_dev, (void *)block_cache, (size_t)(sizeof(float) * dim), cudaMemcpyDeviceToDevice);
	}
	if(do_sqrt) _bncu_sqrt1<<<1, 1>>>(ret_dev, block_cache);
	else        cudaMemcpy((void *)ret_dev, (void *)&(block_cache[0]), sizeof(float), cudaMemcpyDeviceToDevice);
	cudaFree(c_dev); cudaFree(block_cache);
}

__host__ void ip_gcfvector_dev(float *ret_re, float *ret_im, GCFVector a, GCFVector b, int nbg, int ntb)
{
	float *cre, *cim;
	if(a->dim != b->dim) { fprintf(stderr, "ERROR: ip_gcfvector_dev\n"); return; }
	cudaMalloc((void **)&cre, (size_t)(sizeof(float) * a->dim));
	cudaMalloc((void **)&cim, (size_t)(sizeof(float) * a->dim));
	_bncu_conjmul_gcfvector<<<nbg, ntb>>>(cre, cim, a->re, a->im, b->re, b->im, a->dim);
	_gcf_reduce_sum(ret_re, cre, a->dim, nbg, ntb, 0);
	_gcf_reduce_sum(ret_im, cim, a->dim, nbg, ntb, 0);
	cudaFree(cre); cudaFree(cim);
}
__host__ void norm2_gcfvector_dev(float *ret_dev, GCFVector a, int nbg, int ntb)
{
	float *sre;
	cudaMalloc((void **)&sre, (size_t)(sizeof(float) * a->dim));
	_bncu_absq_gcfvector<<<nbg, ntb>>>(sre, a->re, a->im, a->dim);
	_gcf_reduce_sum(ret_dev, sre, a->dim, nbg, ntb, 1);
	cudaFree(sre);
}

/*------------------------- GCFMatrix -------------------------*/
__host__ GCFMatrix init_gcfmatrix(long int row_dim, long int col_dim)
{
	GCFMatrix r = (gcfmatrix *)malloc(sizeof(gcfmatrix));
	if(r == NULL) { fprintf(stderr, "ERROR: init_gcfmatrix\n"); return NULL; }
	r->row_dim = row_dim; r->col_dim = col_dim;
	r->re = (float *)calloc(row_dim * col_dim, sizeof(float));
	r->im = (float *)calloc(row_dim * col_dim, sizeof(float));
	if(r->re == NULL || r->im == NULL) { fprintf(stderr, "ERROR: init_gcfmatrix alloc\n"); return NULL; }
	return r;
}
__host__ void free_gcfmatrix(GCFMatrix m) { free(m->re); free(m->im); free(m); }

__host__ GCFMatrix init_gcfmatrix_dev(long int row_dim, long int col_dim)
{
	GCFMatrix r = (gcfmatrix *)malloc(sizeof(gcfmatrix));
	if(r == NULL) { fprintf(stderr, "ERROR: init_gcfmatrix_dev\n"); return NULL; }
	r->row_dim = row_dim; r->col_dim = col_dim; r->re = NULL; r->im = NULL;
	cudaMalloc((void **)&(r->re), (size_t)(row_dim * col_dim * sizeof(float)));
	cudaMalloc((void **)&(r->im), (size_t)(row_dim * col_dim * sizeof(float)));
	cudaMemset((void *)(r->re), 0, (size_t)(row_dim * col_dim * sizeof(float)));
	cudaMemset((void *)(r->im), 0, (size_t)(row_dim * col_dim * sizeof(float)));
	return r;
}
__host__ void free_gcfmatrix_dev(GCFMatrix m) { cudaFree(m->re); cudaFree(m->im); free(m); }

__host__ void subst_gcfmatrix_dev_cdmat(GCFMatrix gcfmat_dev, CDMatrix cdmat)
{
	long int i, j, rd = gcfmat_dev->row_dim, cd = gcfmat_dev->col_dim;
	GCFMatrix h = init_gcfmatrix(rd, cd);
	for(i = 0; i < rd; i++)
		for(j = 0; j < cd; j++)
		{
			double _Complex z = get_cdmatrix_ij(cdmat, i, j);
			h->re[i * cd + j] = _cre(z); h->im[i * cd + j] = _cim(z);
		}
	cudaMemcpy((void *)gcfmat_dev->re, (void *)h->re, (size_t)(sizeof(float) * rd * cd), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)gcfmat_dev->im, (void *)h->im, (size_t)(sizeof(float) * rd * cd), cudaMemcpyHostToDevice);
	free_gcfmatrix(h);
}
__host__ void subst_cdmatrix_gcfmat_dev(CDMatrix cdmat, GCFMatrix gcfmat_dev)
{
	long int i, j, rd = gcfmat_dev->row_dim, cd = gcfmat_dev->col_dim;
	GCFMatrix h = init_gcfmatrix(rd, cd);
	cudaMemcpy((void *)h->re, (void *)gcfmat_dev->re, (size_t)(sizeof(float) * rd * cd), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)gcfmat_dev->im, (size_t)(sizeof(float) * rd * cd), cudaMemcpyDeviceToHost);
	for(i = 0; i < rd; i++)
		for(j = 0; j < cd; j++)
			set_cdmatrix_ij(cdmat, i, j, _mkc(h->re[i * cd + j], h->im[i * cd + j]));
	free_gcfmatrix(h);
}
__host__ void print_gcfmatrix_dev(GCFMatrix mat)
{
	long int i, j, rd = mat->row_dim, cd = mat->col_dim;
	GCFMatrix h = init_gcfmatrix(rd, cd);
	cudaMemcpy((void *)h->re, (void *)mat->re, (size_t)(sizeof(float) * rd * cd), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)mat->im, (size_t)(sizeof(float) * rd * cd), cudaMemcpyDeviceToHost);
	for(i = 0; i < rd; i++)
		for(j = 0; j < cd; j++)
			printf("%ld,%ld: %25.17e %+25.17e i\n", i, j, h->re[i * cd + j], h->im[i * cd + j]);
	free_gcfmatrix(h);
}

// complex matmul ret := A * B
__global__ void _bncu_mul_gcfmatrix(float *cre, float *cim, long int row_dim, long int col_dim, long int mid_dim,
                                    float *are, float *aim, float *bre, float *bim)
{
	long int i, j, k;
	i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < row_dim)
	{
		for(j = 0; j < col_dim; j++)
		{
			float sre = 0.0, sim = 0.0;
			for(k = 0; k < mid_dim; k++)
			{
				float ar = are[i * mid_dim + k], ai = aim[i * mid_dim + k];
				float br = bre[k * col_dim + j], bi = bim[k * col_dim + j];
				sre += ar * br - ai * bi;
				sim += ar * bi + ai * br;
			}
			cre[i * col_dim + j] = sre;
			cim[i * col_dim + j] = sim;
		}
		i += blockDim.x * gridDim.x;
	}
}
__host__ void mul_gcfmatrix_dev(GCFMatrix ret, GCFMatrix a, GCFMatrix b, int nbg, int ntb)
{
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{ fprintf(stderr, "ERROR: mul_gcfmatrix_dev\n"); return; }
	_bncu_mul_gcfmatrix<<<nbg, ntb>>>(ret->re, ret->im, ret->row_dim, ret->col_dim, a->col_dim, a->re, a->im, b->re, b->im);
}

__host__ void normf_gcfmatrix_dev(float *ret_dev, GCFMatrix mat, int nbg, int ntb)
{
	float *sre; long int total = mat->row_dim * mat->col_dim;
	cudaMalloc((void **)&sre, (size_t)(sizeof(float) * total));
	_bncu_absq_gcfvector<<<nbg, ntb>>>(sre, mat->re, mat->im, total);
	_gcf_reduce_sum(ret_dev, sre, total, nbg, ntb, 1);
	cudaFree(sre);
}

__host__ void add_gcfmatrix_dev(GCFMatrix c, GCFMatrix a, GCFMatrix b, int nbg, int ntb)
{
	if((a->row_dim != c->row_dim) || (a->col_dim != c->col_dim) || (b->row_dim != c->row_dim) || (b->col_dim != c->col_dim))
	{ fprintf(stderr, "ERROR: add_gcfmatrix_dev\n"); return; }
	_bncu_add_gcfvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->row_dim * c->col_dim);
}
__host__ void sub_gcfmatrix_dev(GCFMatrix c, GCFMatrix a, GCFMatrix b, int nbg, int ntb)
{
	if((a->row_dim != c->row_dim) || (a->col_dim != c->col_dim) || (b->row_dim != c->row_dim) || (b->col_dim != c->col_dim))
	{ fprintf(stderr, "ERROR: sub_gcfmatrix_dev\n"); return; }
	_bncu_sub_gcfvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->row_dim * c->col_dim);
}
__host__ void cmul_gcfmatrix_dev(GCFMatrix c, float sre, float sim, GCFMatrix a, int nbg, int ntb)
{
	if((a->row_dim != c->row_dim) || (a->col_dim != c->col_dim)) { fprintf(stderr, "ERROR: cmul_gcfmatrix_dev\n"); return; }
	_bncu_cmul_gcfvector<<<nbg, ntb>>>(c->re, c->im, sre, sim, a->re, a->im, c->row_dim * c->col_dim);
}

__global__ void _bncu_transpose_gcfmatrix(float *cre, float *cim, float *are, float *aim, long int row_dim, long int col_dim, int conj)
{
	long int i, j;
	i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < row_dim)
	{
		for(j = 0; j < col_dim; j++)
		{
			cre[i * col_dim + j] = are[j * row_dim + i];
			cim[i * col_dim + j] = conj ? -aim[j * row_dim + i] : aim[j * row_dim + i];
		}
		i += blockDim.x * gridDim.x;
	}
}
__host__ void transpose_gcfmatrix_dev(GCFMatrix c, GCFMatrix a, int nbg, int ntb)
{
	if((c->row_dim != a->col_dim) || (c->col_dim != a->row_dim)) { fprintf(stderr, "ERROR: transpose_gcfmatrix_dev\n"); return; }
	_bncu_transpose_gcfmatrix<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim, c->col_dim, 0);
}
__host__ void conjtrans_gcfmatrix_dev(GCFMatrix c, GCFMatrix a, int nbg, int ntb)
{
	if((c->row_dim != a->col_dim) || (c->col_dim != a->row_dim)) { fprintf(stderr, "ERROR: conjtrans_gcfmatrix_dev\n"); return; }
	_bncu_transpose_gcfmatrix<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim, c->col_dim, 1);
}
__host__ void subst_gcfmatrix_dev(GCFMatrix c, GCFMatrix a, int nbg, int ntb)
{
	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim)) { fprintf(stderr, "ERROR: subst_gcfmatrix_dev\n"); return; }
	_bncu_subst_gcfvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim * c->col_dim);
}
__host__ void set0_gcfmatrix_dev(GCFMatrix c, int nbg, int ntb)
{
	_bncu_set0_gcfvector<<<nbg, ntb>>>(c->re, c->im, c->row_dim * c->col_dim);
}
__host__ void set_gcfmatrix_ij_dev(GCFMatrix mat, long int i, long int j, float vre, float vim)
{
	if((i >= 0) && (i < mat->row_dim) && (j >= 0) && (j < mat->col_dim))
	{
		long int idx = i * mat->col_dim + j;
		cudaMemcpy((void *)&(mat->re[idx]), (void *)&vre, sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy((void *)&(mat->im[idx]), (void *)&vim, sizeof(float), cudaMemcpyHostToDevice);
	}
}
__host__ void setI_gcfmatrix_dev(GCFMatrix c, int nbg, int ntb)
{
	long int i;
	_bncu_set0_gcfvector<<<nbg, ntb>>>(c->re, c->im, c->row_dim * c->col_dim);
	for(i = 0; i < c->row_dim; i++) set_gcfmatrix_ij_dev(c, i, i, 1.0, 0.0);
}

// v := a * vb (complex)
__global__ void _bncu_mul_gcfmatrix_gcfvec(float *vre, float *vim, float *are, float *aim, float *bre, float *bim, long int row_dim, long int col_dim)
{
	long int j, index;
	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < row_dim)
	{
		float sre = 0.0, sim = 0.0;
		for(j = 0; j < col_dim; j++)
		{
			float ar = are[index * col_dim + j], ai = aim[index * col_dim + j];
			float br = bre[j], bi = bim[j];
			sre += ar * br - ai * bi;
			sim += ar * bi + ai * br;
		}
		vre[index] = sre; vim[index] = sim;
		index += blockDim.x * gridDim.x;
	}
}
__host__ void mul_gcfmatrix_gcfvec(GCFVector v, GCFMatrix a, GCFVector vb, int nbg, int ntb)
{
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim)) { fprintf(stderr, "ERROR: mul_gcfmatrix_gcfvec\n"); return; }
	_bncu_mul_gcfmatrix_gcfvec<<<nbg, ntb>>>(v->re, v->im, a->re, a->im, vb->re, vb->im, a->row_dim, a->col_dim);
}

// v := a^T * vb (complex, non-conjugate transpose)
__global__ void _bncu_mul_gcfmatrixt_gcfvec(float *vre, float *vim, float *are, float *aim, float *bre, float *bim, long int row_dim, long int col_dim)
{
	long int j, index;
	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < col_dim)
	{
		float sre = 0.0, sim = 0.0;
		for(j = 0; j < row_dim; j++)
		{
			float ar = are[j * col_dim + index], ai = aim[j * col_dim + index];
			float br = bre[j], bi = bim[j];
			sre += ar * br - ai * bi;
			sim += ar * bi + ai * br;
		}
		vre[index] = sre; vim[index] = sim;
		index += blockDim.x * gridDim.x;
	}
}
__host__ void mul_gcfmatrixt_gcfvec(GCFVector v, GCFMatrix a, GCFVector vb, int nbg, int ntb)
{
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim)) { fprintf(stderr, "ERROR: mul_gcfmatrixt_gcfvec\n"); return; }
	_bncu_mul_gcfmatrixt_gcfvec<<<nbg, ntb>>>(v->re, v->im, a->re, a->im, vb->re, vb->im, a->row_dim, a->col_dim);
}
