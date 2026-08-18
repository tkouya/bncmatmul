/********************************************************************************/
/* gcdlinear.cu: Native double-precision COMPLEX GPU Linear Computation (CUDA)   */
/*   complex as separate real/imag double arrays (SoA).                          */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#include "gcdlinear.h"

// C99 <complex.h> macros (creal/cimag/_Complex_I) are unavailable in nvcc C++ mode;
// use the GCC complex extension via these helpers.
static inline double _Complex _mkc(double re, double im) { double _Complex z; __real__ z = re; __imag__ z = im; return z; }
#define _cre(z) (__real__(z))
#define _cim(z) (__imag__(z))

/*------------------------- GCDVector -------------------------*/
__host__ GCDVector init_gcdvector(long int dim)
{
	GCDVector r = (gcdvector *)malloc(sizeof(gcdvector));
	if(r == NULL) { fprintf(stderr, "ERROR: init_gcdvector\n"); return NULL; }
	r->dim = dim;
	r->re = (double *)calloc(dim, sizeof(double));
	r->im = (double *)calloc(dim, sizeof(double));
	if(r->re == NULL || r->im == NULL) { fprintf(stderr, "ERROR: init_gcdvector alloc\n"); return NULL; }
	return r;
}
__host__ void free_gcdvector(GCDVector v) { free(v->re); free(v->im); free(v); }

__host__ GCDVector init_gcdvector_dev(long int dim)
{
	GCDVector r = (gcdvector *)malloc(sizeof(gcdvector));
	if(r == NULL) { fprintf(stderr, "ERROR: init_gcdvector_dev\n"); return NULL; }
	r->dim = dim; r->re = NULL; r->im = NULL;
	cudaMalloc((void **)&(r->re), (size_t)(dim * sizeof(double)));
	cudaMalloc((void **)&(r->im), (size_t)(dim * sizeof(double)));
	cudaMemset((void *)(r->re), 0, (size_t)(dim * sizeof(double)));
	cudaMemset((void *)(r->im), 0, (size_t)(dim * sizeof(double)));
	return r;
}
__host__ void free_gcdvector_dev(GCDVector v) { cudaFree(v->re); cudaFree(v->im); free(v); }

__host__ void subst_gcdvector_dev_cdvec(GCDVector gcdvec_dev, CDVector cdvec)
{
	long int i, dim = gcdvec_dev->dim;
	GCDVector h = init_gcdvector(dim);
	for(i = 0; i < dim; i++)
	{
		double _Complex z = get_cdvector_i(cdvec, i);
		h->re[i] = _cre(z); h->im[i] = _cim(z);
	}
	cudaMemcpy((void *)gcdvec_dev->re, (void *)h->re, (size_t)(dim * sizeof(double)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)gcdvec_dev->im, (void *)h->im, (size_t)(dim * sizeof(double)), cudaMemcpyHostToDevice);
	free_gcdvector(h);
}
__host__ void subst_cdvector_gcdvec_dev(CDVector cdvec, GCDVector gcdvec_dev)
{
	long int i, dim = gcdvec_dev->dim;
	GCDVector h = init_gcdvector(dim);
	cudaMemcpy((void *)h->re, (void *)gcdvec_dev->re, (size_t)(dim * sizeof(double)), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)gcdvec_dev->im, (size_t)(dim * sizeof(double)), cudaMemcpyDeviceToHost);
	for(i = 0; i < dim; i++)
		set_cdvector_i(cdvec, i, _mkc(h->re[i], h->im[i]));
	free_gcdvector(h);
}
__host__ void print_gcdvector_dev(GCDVector dev_vec)
{
	long int i, dim = dev_vec->dim;
	GCDVector h = init_gcdvector(dim);
	cudaMemcpy((void *)h->re, (void *)dev_vec->re, (size_t)(dim * sizeof(double)), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)dev_vec->im, (size_t)(dim * sizeof(double)), cudaMemcpyDeviceToHost);
	for(i = 0; i < dim; i++) printf("%4ld: %25.17e %+25.17e i\n", i, h->re[i], h->im[i]);
	free_gcdvector(h);
}

__global__ void _bncu_add_gcdvector(double *cre, double *cim, double *are, double *aim, double *bre, double *bim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx] + bre[idx]; cim[idx] = aim[idx] + bim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void add_gcdvector_dev(GCDVector c, GCDVector a, GCDVector b, int nbg, int ntb)
{
	if((a->dim != b->dim) || (c->dim != a->dim)) { fprintf(stderr, "ERROR: add_gcdvector_dev\n"); return; }
	_bncu_add_gcdvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->dim);
}

__global__ void _bncu_sub_gcdvector(double *cre, double *cim, double *are, double *aim, double *bre, double *bim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx] - bre[idx]; cim[idx] = aim[idx] - bim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void sub_gcdvector_dev(GCDVector c, GCDVector a, GCDVector b, int nbg, int ntb)
{
	if((a->dim != b->dim) || (c->dim != a->dim)) { fprintf(stderr, "ERROR: sub_gcdvector_dev\n"); return; }
	_bncu_sub_gcdvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->dim);
}

__global__ void _bncu_cmul_gcdvector(double *cre, double *cim, double vre, double vim, double *are, double *aim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim)
	{
		double ar = are[idx], ai = aim[idx];
		cre[idx] = vre * ar - vim * ai;
		cim[idx] = vre * ai + vim * ar;
		idx += blockDim.x * gridDim.x;
	}
}
__host__ void cmul_gcdvector_dev(GCDVector c, double vre, double vim, GCDVector a, int nbg, int ntb)
{
	if(c->dim != a->dim) { fprintf(stderr, "ERROR: cmul_gcdvector_dev\n"); return; }
	_bncu_cmul_gcdvector<<<nbg, ntb>>>(c->re, c->im, vre, vim, a->re, a->im, c->dim);
}

__global__ void _bncu_subst_gcdvector(double *cre, double *cim, double *are, double *aim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx]; cim[idx] = aim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void subst_gcdvector_dev(GCDVector ret, GCDVector vec, int nbg, int ntb)
{
	if(ret->dim != vec->dim) { fprintf(stderr, "ERROR: subst_gcdvector_dev\n"); return; }
	_bncu_subst_gcdvector<<<nbg, ntb>>>(ret->re, ret->im, vec->re, vec->im, ret->dim);
}

__global__ void _bncu_set0_gcdvector(double *cre, double *cim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = 0.0; cim[idx] = 0.0; idx += blockDim.x * gridDim.x; }
}
__host__ void set0_gcdvector_dev(GCDVector ret, int nbg, int ntb)
{
	_bncu_set0_gcdvector<<<nbg, ntb>>>(ret->re, ret->im, ret->dim);
}

// element-wise conj(a)*b -> (cre,cim); and |a|^2 -> sre
__global__ void _bncu_conjmul_gcdvector(double *cre, double *cim, double *are, double *aim, double *bre, double *bim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim)
	{
		double ar = are[idx], ai = aim[idx], br = bre[idx], bi = bim[idx];
		// conj(a)*b = (ar - i ai)(br + i bi) = (ar*br + ai*bi) + i(ar*bi - ai*br)
		cre[idx] = ar * br + ai * bi;
		cim[idx] = ar * bi - ai * br;
		idx += blockDim.x * gridDim.x;
	}
}
__global__ void _bncu_absq_gcdvector(double *sre, double *are, double *aim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { sre[idx] = are[idx] * are[idx] + aim[idx] * aim[idx]; idx += blockDim.x * gridDim.x; }
}

// real-sum reduction (used for |.|^2 accumulation)
__global__ void _bncu_add_reduct_double(double *ret, double *vec, long int dim)
{
	long int index, cache_index;
	__shared__ double cache[MAX_NUM_THREADS_PER_BLOCK];
	index = threadIdx.x + blockIdx.x * blockDim.x;
	cache_index = threadIdx.x;
	cache[cache_index] = 0.0;
	while(index < dim) { cache[cache_index] += vec[index]; index += blockDim.x * gridDim.x; }
	__syncthreads();
	index = blockDim.x / 2;
	while(index > 0) { if(cache_index < index) cache[cache_index] += cache[cache_index + index]; __syncthreads(); index /= 2; }
	if(cache_index == 0) ret[blockIdx.x] = cache[0];
}
__global__ void _bncu_sqrt1(double *ret, double *src) { ret[0] = sqrt(src[0]); }

// reduce-sum a device array src(len) into ret_dev (1 elem); do_sqrt for norm
static void _gcd_reduce_sum(double *ret_dev, double *src, long int len, int nbg, int ntb, int do_sqrt)
{
	double *c_dev, *block_cache;
	int block_dim = nbg, thread_dim = ntb; long int dim = len;
	cudaMalloc((void **)&c_dev, (size_t)(sizeof(double) * len));
	cudaMalloc((void **)&block_cache, (size_t)(sizeof(double) * nbg));
	cudaMemcpy((void *)c_dev, (void *)src, (size_t)(sizeof(double) * len), cudaMemcpyDeviceToDevice);
	while(block_dim >= 1)
	{
		_bncu_add_reduct_double<<<block_dim, thread_dim>>>(block_cache, c_dev, dim);
		if(block_dim <= 1) break;
		dim = block_dim;
		if(block_dim > thread_dim) block_dim = (block_dim / thread_dim) + 1; else block_dim = 1;
		cudaMemcpy((void *)c_dev, (void *)block_cache, (size_t)(sizeof(double) * dim), cudaMemcpyDeviceToDevice);
	}
	if(do_sqrt) _bncu_sqrt1<<<1, 1>>>(ret_dev, block_cache);
	else        cudaMemcpy((void *)ret_dev, (void *)&(block_cache[0]), sizeof(double), cudaMemcpyDeviceToDevice);
	cudaFree(c_dev); cudaFree(block_cache);
}

__host__ void ip_gcdvector_dev(double *ret_re, double *ret_im, GCDVector a, GCDVector b, int nbg, int ntb)
{
	double *cre, *cim;
	if(a->dim != b->dim) { fprintf(stderr, "ERROR: ip_gcdvector_dev\n"); return; }
	cudaMalloc((void **)&cre, (size_t)(sizeof(double) * a->dim));
	cudaMalloc((void **)&cim, (size_t)(sizeof(double) * a->dim));
	_bncu_conjmul_gcdvector<<<nbg, ntb>>>(cre, cim, a->re, a->im, b->re, b->im, a->dim);
	_gcd_reduce_sum(ret_re, cre, a->dim, nbg, ntb, 0);
	_gcd_reduce_sum(ret_im, cim, a->dim, nbg, ntb, 0);
	cudaFree(cre); cudaFree(cim);
}
__host__ void norm2_gcdvector_dev(double *ret_dev, GCDVector a, int nbg, int ntb)
{
	double *sre;
	cudaMalloc((void **)&sre, (size_t)(sizeof(double) * a->dim));
	_bncu_absq_gcdvector<<<nbg, ntb>>>(sre, a->re, a->im, a->dim);
	_gcd_reduce_sum(ret_dev, sre, a->dim, nbg, ntb, 1);
	cudaFree(sre);
}

/*------------------------- GCDMatrix -------------------------*/
__host__ GCDMatrix init_gcdmatrix(long int row_dim, long int col_dim)
{
	GCDMatrix r = (gcdmatrix *)malloc(sizeof(gcdmatrix));
	if(r == NULL) { fprintf(stderr, "ERROR: init_gcdmatrix\n"); return NULL; }
	r->row_dim = row_dim; r->col_dim = col_dim;
	r->re = (double *)calloc(row_dim * col_dim, sizeof(double));
	r->im = (double *)calloc(row_dim * col_dim, sizeof(double));
	if(r->re == NULL || r->im == NULL) { fprintf(stderr, "ERROR: init_gcdmatrix alloc\n"); return NULL; }
	return r;
}
__host__ void free_gcdmatrix(GCDMatrix m) { free(m->re); free(m->im); free(m); }

__host__ GCDMatrix init_gcdmatrix_dev(long int row_dim, long int col_dim)
{
	GCDMatrix r = (gcdmatrix *)malloc(sizeof(gcdmatrix));
	if(r == NULL) { fprintf(stderr, "ERROR: init_gcdmatrix_dev\n"); return NULL; }
	r->row_dim = row_dim; r->col_dim = col_dim; r->re = NULL; r->im = NULL;
	cudaMalloc((void **)&(r->re), (size_t)(row_dim * col_dim * sizeof(double)));
	cudaMalloc((void **)&(r->im), (size_t)(row_dim * col_dim * sizeof(double)));
	cudaMemset((void *)(r->re), 0, (size_t)(row_dim * col_dim * sizeof(double)));
	cudaMemset((void *)(r->im), 0, (size_t)(row_dim * col_dim * sizeof(double)));
	return r;
}
__host__ void free_gcdmatrix_dev(GCDMatrix m) { cudaFree(m->re); cudaFree(m->im); free(m); }

__host__ void subst_gcdmatrix_dev_cdmat(GCDMatrix gcdmat_dev, CDMatrix cdmat)
{
	long int i, j, rd = gcdmat_dev->row_dim, cd = gcdmat_dev->col_dim;
	GCDMatrix h = init_gcdmatrix(rd, cd);
	for(i = 0; i < rd; i++)
		for(j = 0; j < cd; j++)
		{
			double _Complex z = get_cdmatrix_ij(cdmat, i, j);
			h->re[i * cd + j] = _cre(z); h->im[i * cd + j] = _cim(z);
		}
	cudaMemcpy((void *)gcdmat_dev->re, (void *)h->re, (size_t)(sizeof(double) * rd * cd), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)gcdmat_dev->im, (void *)h->im, (size_t)(sizeof(double) * rd * cd), cudaMemcpyHostToDevice);
	free_gcdmatrix(h);
}
__host__ void subst_cdmatrix_gcdmat_dev(CDMatrix cdmat, GCDMatrix gcdmat_dev)
{
	long int i, j, rd = gcdmat_dev->row_dim, cd = gcdmat_dev->col_dim;
	GCDMatrix h = init_gcdmatrix(rd, cd);
	cudaMemcpy((void *)h->re, (void *)gcdmat_dev->re, (size_t)(sizeof(double) * rd * cd), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)gcdmat_dev->im, (size_t)(sizeof(double) * rd * cd), cudaMemcpyDeviceToHost);
	for(i = 0; i < rd; i++)
		for(j = 0; j < cd; j++)
			set_cdmatrix_ij(cdmat, i, j, _mkc(h->re[i * cd + j], h->im[i * cd + j]));
	free_gcdmatrix(h);
}
__host__ void print_gcdmatrix_dev(GCDMatrix mat)
{
	long int i, j, rd = mat->row_dim, cd = mat->col_dim;
	GCDMatrix h = init_gcdmatrix(rd, cd);
	cudaMemcpy((void *)h->re, (void *)mat->re, (size_t)(sizeof(double) * rd * cd), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)mat->im, (size_t)(sizeof(double) * rd * cd), cudaMemcpyDeviceToHost);
	for(i = 0; i < rd; i++)
		for(j = 0; j < cd; j++)
			printf("%ld,%ld: %25.17e %+25.17e i\n", i, j, h->re[i * cd + j], h->im[i * cd + j]);
	free_gcdmatrix(h);
}

// complex matmul ret := A * B
__global__ void _bncu_mul_gcdmatrix(double *cre, double *cim, long int row_dim, long int col_dim, long int mid_dim,
                                    double *are, double *aim, double *bre, double *bim)
{
	long int i, j, k;
	i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < row_dim)
	{
		for(j = 0; j < col_dim; j++)
		{
			double sre = 0.0, sim = 0.0;
			for(k = 0; k < mid_dim; k++)
			{
				double ar = are[i * mid_dim + k], ai = aim[i * mid_dim + k];
				double br = bre[k * col_dim + j], bi = bim[k * col_dim + j];
				sre += ar * br - ai * bi;
				sim += ar * bi + ai * br;
			}
			cre[i * col_dim + j] = sre;
			cim[i * col_dim + j] = sim;
		}
		i += blockDim.x * gridDim.x;
	}
}
__host__ void mul_gcdmatrix_dev(GCDMatrix ret, GCDMatrix a, GCDMatrix b, int nbg, int ntb)
{
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{ fprintf(stderr, "ERROR: mul_gcdmatrix_dev\n"); return; }
	_bncu_mul_gcdmatrix<<<nbg, ntb>>>(ret->re, ret->im, ret->row_dim, ret->col_dim, a->col_dim, a->re, a->im, b->re, b->im);
}

__host__ void normf_gcdmatrix_dev(double *ret_dev, GCDMatrix mat, int nbg, int ntb)
{
	double *sre; long int total = mat->row_dim * mat->col_dim;
	cudaMalloc((void **)&sre, (size_t)(sizeof(double) * total));
	_bncu_absq_gcdvector<<<nbg, ntb>>>(sre, mat->re, mat->im, total);
	_gcd_reduce_sum(ret_dev, sre, total, nbg, ntb, 1);
	cudaFree(sre);
}

__host__ void add_gcdmatrix_dev(GCDMatrix c, GCDMatrix a, GCDMatrix b, int nbg, int ntb)
{
	if((a->row_dim != c->row_dim) || (a->col_dim != c->col_dim) || (b->row_dim != c->row_dim) || (b->col_dim != c->col_dim))
	{ fprintf(stderr, "ERROR: add_gcdmatrix_dev\n"); return; }
	_bncu_add_gcdvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->row_dim * c->col_dim);
}
__host__ void sub_gcdmatrix_dev(GCDMatrix c, GCDMatrix a, GCDMatrix b, int nbg, int ntb)
{
	if((a->row_dim != c->row_dim) || (a->col_dim != c->col_dim) || (b->row_dim != c->row_dim) || (b->col_dim != c->col_dim))
	{ fprintf(stderr, "ERROR: sub_gcdmatrix_dev\n"); return; }
	_bncu_sub_gcdvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->row_dim * c->col_dim);
}
__host__ void cmul_gcdmatrix_dev(GCDMatrix c, double sre, double sim, GCDMatrix a, int nbg, int ntb)
{
	if((a->row_dim != c->row_dim) || (a->col_dim != c->col_dim)) { fprintf(stderr, "ERROR: cmul_gcdmatrix_dev\n"); return; }
	_bncu_cmul_gcdvector<<<nbg, ntb>>>(c->re, c->im, sre, sim, a->re, a->im, c->row_dim * c->col_dim);
}

__global__ void _bncu_transpose_gcdmatrix(double *cre, double *cim, double *are, double *aim, long int row_dim, long int col_dim, int conj)
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
__host__ void transpose_gcdmatrix_dev(GCDMatrix c, GCDMatrix a, int nbg, int ntb)
{
	if((c->row_dim != a->col_dim) || (c->col_dim != a->row_dim)) { fprintf(stderr, "ERROR: transpose_gcdmatrix_dev\n"); return; }
	_bncu_transpose_gcdmatrix<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim, c->col_dim, 0);
}
__host__ void conjtrans_gcdmatrix_dev(GCDMatrix c, GCDMatrix a, int nbg, int ntb)
{
	if((c->row_dim != a->col_dim) || (c->col_dim != a->row_dim)) { fprintf(stderr, "ERROR: conjtrans_gcdmatrix_dev\n"); return; }
	_bncu_transpose_gcdmatrix<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim, c->col_dim, 1);
}
__host__ void subst_gcdmatrix_dev(GCDMatrix c, GCDMatrix a, int nbg, int ntb)
{
	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim)) { fprintf(stderr, "ERROR: subst_gcdmatrix_dev\n"); return; }
	_bncu_subst_gcdvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim * c->col_dim);
}
__host__ void set0_gcdmatrix_dev(GCDMatrix c, int nbg, int ntb)
{
	_bncu_set0_gcdvector<<<nbg, ntb>>>(c->re, c->im, c->row_dim * c->col_dim);
}
__host__ void set_gcdmatrix_ij_dev(GCDMatrix mat, long int i, long int j, double vre, double vim)
{
	if((i >= 0) && (i < mat->row_dim) && (j >= 0) && (j < mat->col_dim))
	{
		long int idx = i * mat->col_dim + j;
		cudaMemcpy((void *)&(mat->re[idx]), (void *)&vre, sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy((void *)&(mat->im[idx]), (void *)&vim, sizeof(double), cudaMemcpyHostToDevice);
	}
}
__host__ void setI_gcdmatrix_dev(GCDMatrix c, int nbg, int ntb)
{
	long int i;
	_bncu_set0_gcdvector<<<nbg, ntb>>>(c->re, c->im, c->row_dim * c->col_dim);
	for(i = 0; i < c->row_dim; i++) set_gcdmatrix_ij_dev(c, i, i, 1.0, 0.0);
}

// v := a * vb (complex)
__global__ void _bncu_mul_gcdmatrix_gcdvec(double *vre, double *vim, double *are, double *aim, double *bre, double *bim, long int row_dim, long int col_dim)
{
	long int j, index;
	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < row_dim)
	{
		double sre = 0.0, sim = 0.0;
		for(j = 0; j < col_dim; j++)
		{
			double ar = are[index * col_dim + j], ai = aim[index * col_dim + j];
			double br = bre[j], bi = bim[j];
			sre += ar * br - ai * bi;
			sim += ar * bi + ai * br;
		}
		vre[index] = sre; vim[index] = sim;
		index += blockDim.x * gridDim.x;
	}
}
__host__ void mul_gcdmatrix_gcdvec(GCDVector v, GCDMatrix a, GCDVector vb, int nbg, int ntb)
{
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim)) { fprintf(stderr, "ERROR: mul_gcdmatrix_gcdvec\n"); return; }
	_bncu_mul_gcdmatrix_gcdvec<<<nbg, ntb>>>(v->re, v->im, a->re, a->im, vb->re, vb->im, a->row_dim, a->col_dim);
}

// v := a^T * vb (complex, non-conjugate transpose)
__global__ void _bncu_mul_gcdmatrixt_gcdvec(double *vre, double *vim, double *are, double *aim, double *bre, double *bim, long int row_dim, long int col_dim)
{
	long int j, index;
	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < col_dim)
	{
		double sre = 0.0, sim = 0.0;
		for(j = 0; j < row_dim; j++)
		{
			double ar = are[j * col_dim + index], ai = aim[j * col_dim + index];
			double br = bre[j], bi = bim[j];
			sre += ar * br - ai * bi;
			sim += ar * bi + ai * br;
		}
		vre[index] = sre; vim[index] = sim;
		index += blockDim.x * gridDim.x;
	}
}
__host__ void mul_gcdmatrixt_gcdvec(GCDVector v, GCDMatrix a, GCDVector vb, int nbg, int ntb)
{
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim)) { fprintf(stderr, "ERROR: mul_gcdmatrixt_gcdvec\n"); return; }
	_bncu_mul_gcdmatrixt_gcdvec<<<nbg, ntb>>>(v->re, v->im, a->re, a->im, vb->re, vb->im, a->row_dim, a->col_dim);
}
