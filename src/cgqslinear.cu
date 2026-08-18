/********************************************************************************/
/* cgqslinear.cu: Complex float-float GPU Linear Computation (CUDA)            */
/*   complex as separate real/imag gqs_real arrays (SoA); device dd math.        */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#include "cgqslinear.h"
#include "gqs.cu"   // device gqs_real arithmetic implementation

// per-type zero/one/neg via limb pointer-cast (component count = CG_NLIMB).
// Generic over double2/3/4 (and float2/3/4 after sed float->float).
#define CG_NLIMB CGQS_SIZE
typedef float cg_scalar;
__device__ static inline gqs_real cg_zero() { gqs_real z; cg_scalar *p = (cg_scalar *)&z; for(int l = 0; l < CG_NLIMB; l++) p[l] = 0; return z; }
__device__ static inline gqs_real cg_neg(gqs_real v) { cg_scalar *p = (cg_scalar *)&v; for(int l = 0; l < CG_NLIMB; l++) p[l] = -p[l]; return v; }
static inline gqs_real cg_host_one() { gqs_real z; cg_scalar *p = (cg_scalar *)&z; for(int l = 0; l < CG_NLIMB; l++) p[l] = 0; p[0] = 1; return z; }

/*--- host<->device element conversion helpers (limb pointer-cast) ---*/
static inline void _cgqs_pack(gqs_real *re, gqs_real *im, cqsfloat z)
{
	float *pre = (float *)re, *pim = (float *)im;
	for(int l = 0; l < CGQS_SIZE; l++) { pre[l] = z.val_re[l]; pim[l] = z.val_im[l]; }
}
static inline cqsfloat _cgqs_unpack(gqs_real re, gqs_real im)
{
	cqsfloat z;
	float *pre = (float *)&re, *pim = (float *)&im;
	for(int l = 0; l < CGQS_SIZE; l++) { z.val_re[l] = pre[l]; z.val_im[l] = pim[l]; }
	return z;
}

/*--- vector ---*/
__host__ CGQSVector init_cgqsvector(long int dim)
{
	CGQSVector r = (cgqsvector *)malloc(sizeof(cgqsvector));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgqsvector\n"); return NULL; }
	r->dim = dim;
	r->re = (gqs_real *)calloc(dim, sizeof(gqs_real));
	r->im = (gqs_real *)calloc(dim, sizeof(gqs_real));
	return r;
}
__host__ void free_cgqsvector(CGQSVector v) { free(v->re); free(v->im); free(v); }
__host__ CGQSVector init_cgqsvector_dev(long int dim)
{
	CGQSVector r = (cgqsvector *)malloc(sizeof(cgqsvector));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgqsvector_dev\n"); return NULL; }
	r->dim = dim; r->re = NULL; r->im = NULL;
	cudaMalloc((void **)&(r->re), (size_t)(dim * sizeof(gqs_real)));
	cudaMalloc((void **)&(r->im), (size_t)(dim * sizeof(gqs_real)));
	cudaMemset((void *)(r->re), 0, (size_t)(dim * sizeof(gqs_real)));
	cudaMemset((void *)(r->im), 0, (size_t)(dim * sizeof(gqs_real)));
	return r;
}
__host__ void free_cgqsvector_dev(CGQSVector v) { cudaFree(v->re); cudaFree(v->im); free(v); }

__host__ void subst_cgqsvector_dev_cqsvec(CGQSVector dev, CQSVector cpu)
{
	long int i, dim = dev->dim;
	CGQSVector h = init_cgqsvector(dim);
	for(i = 0; i < dim; i++)
		_cgqs_pack(&h->re[i], &h->im[i], get_cqsvector_i_cqsfloat(cpu, i));
	cudaMemcpy((void *)dev->re, (void *)h->re, (size_t)(dim * sizeof(gqs_real)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dev->im, (void *)h->im, (size_t)(dim * sizeof(gqs_real)), cudaMemcpyHostToDevice);
	free_cgqsvector(h);
}
__host__ void subst_cqsvector_cgqsvec_dev(CQSVector cpu, CGQSVector dev)
{
	long int i, dim = dev->dim;
	CGQSVector h = init_cgqsvector(dim);
	cudaMemcpy((void *)h->re, (void *)dev->re, (size_t)(dim * sizeof(gqs_real)), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)dev->im, (size_t)(dim * sizeof(gqs_real)), cudaMemcpyDeviceToHost);
	for(i = 0; i < dim; i++)
	{
		cqsfloat z = _cgqs_unpack(h->re[i], h->im[i]);
		set_cqsvector_i(cpu, i, &z);
	}
	free_cgqsvector(h);
}

__global__ void _bncu_add_cgqsvector(gqs_real *cre, gqs_real *cim, gqs_real *are, gqs_real *aim, gqs_real *bre, gqs_real *bim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx] + bre[idx]; cim[idx] = aim[idx] + bim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void add_cgqsvector_dev(CGQSVector c, CGQSVector a, CGQSVector b, int nbg, int ntb)
{ _bncu_add_cgqsvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->dim); }

__global__ void _bncu_sub_cgqsvector(gqs_real *cre, gqs_real *cim, gqs_real *are, gqs_real *aim, gqs_real *bre, gqs_real *bim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx] - bre[idx]; cim[idx] = aim[idx] - bim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void sub_cgqsvector_dev(CGQSVector c, CGQSVector a, CGQSVector b, int nbg, int ntb)
{ _bncu_sub_cgqsvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->dim); }

__global__ void _bncu_cmul_cgqsvector(gqs_real *cre, gqs_real *cim, gqs_real vre, gqs_real vim, gqs_real *are, gqs_real *aim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim)
	{
		gqs_real ar = are[idx], ai = aim[idx];
		cre[idx] = vre * ar - vim * ai;
		cim[idx] = vre * ai + vim * ar;
		idx += blockDim.x * gridDim.x;
	}
}
__host__ void cmul_cgqsvector_dev(CGQSVector c, gqs_real vre, gqs_real vim, CGQSVector a, int nbg, int ntb)
{ _bncu_cmul_cgqsvector<<<nbg, ntb>>>(c->re, c->im, vre, vim, a->re, a->im, c->dim); }

__global__ void _bncu_subst_cgqsvector(gqs_real *cre, gqs_real *cim, gqs_real *are, gqs_real *aim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx]; cim[idx] = aim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void subst_cgqsvector_dev(CGQSVector ret, CGQSVector v, int nbg, int ntb)
{ _bncu_subst_cgqsvector<<<nbg, ntb>>>(ret->re, ret->im, v->re, v->im, ret->dim); }

__global__ void _bncu_set0_cgqsvector(gqs_real *cre, gqs_real *cim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = cg_zero(); cim[idx] = cg_zero(); idx += blockDim.x * gridDim.x; }
}
__host__ void set0_cgqsvector_dev(CGQSVector ret, int nbg, int ntb)
{ _bncu_set0_cgqsvector<<<nbg, ntb>>>(ret->re, ret->im, ret->dim); }

/*--- matrix ---*/
__host__ CGQSMatrix init_cgqsmatrix(long int row_dim, long int col_dim)
{
	CGQSMatrix r = (cgqsmatrix *)malloc(sizeof(cgqsmatrix));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgqsmatrix\n"); return NULL; }
	r->row_dim = row_dim; r->col_dim = col_dim;
	r->re = (gqs_real *)calloc(row_dim * col_dim, sizeof(gqs_real));
	r->im = (gqs_real *)calloc(row_dim * col_dim, sizeof(gqs_real));
	return r;
}
__host__ void free_cgqsmatrix(CGQSMatrix m) { free(m->re); free(m->im); free(m); }
__host__ CGQSMatrix init_cgqsmatrix_dev(long int row_dim, long int col_dim)
{
	CGQSMatrix r = (cgqsmatrix *)malloc(sizeof(cgqsmatrix));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgqsmatrix_dev\n"); return NULL; }
	r->row_dim = row_dim; r->col_dim = col_dim; r->re = NULL; r->im = NULL;
	cudaMalloc((void **)&(r->re), (size_t)(row_dim * col_dim * sizeof(gqs_real)));
	cudaMalloc((void **)&(r->im), (size_t)(row_dim * col_dim * sizeof(gqs_real)));
	cudaMemset((void *)(r->re), 0, (size_t)(row_dim * col_dim * sizeof(gqs_real)));
	cudaMemset((void *)(r->im), 0, (size_t)(row_dim * col_dim * sizeof(gqs_real)));
	return r;
}
__host__ void free_cgqsmatrix_dev(CGQSMatrix m) { cudaFree(m->re); cudaFree(m->im); free(m); }

__host__ void subst_cgqsmatrix_dev_cqsmat(CGQSMatrix dev, CQSMatrix cpu)
{
	long int i, j, rd = dev->row_dim, cd = dev->col_dim;
	CGQSMatrix h = init_cgqsmatrix(rd, cd);
	for(i = 0; i < rd; i++)
		for(j = 0; j < cd; j++)
			_cgqs_pack(&h->re[i * cd + j], &h->im[i * cd + j], get_cqsmatrix_ij_cqsfloat(cpu, i, j));
	cudaMemcpy((void *)dev->re, (void *)h->re, (size_t)(sizeof(gqs_real) * rd * cd), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dev->im, (void *)h->im, (size_t)(sizeof(gqs_real) * rd * cd), cudaMemcpyHostToDevice);
	free_cgqsmatrix(h);
}
__host__ void subst_cqsmatrix_cgqsmat_dev(CQSMatrix cpu, CGQSMatrix dev)
{
	long int i, j, rd = dev->row_dim, cd = dev->col_dim;
	CGQSMatrix h = init_cgqsmatrix(rd, cd);
	cudaMemcpy((void *)h->re, (void *)dev->re, (size_t)(sizeof(gqs_real) * rd * cd), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)dev->im, (size_t)(sizeof(gqs_real) * rd * cd), cudaMemcpyDeviceToHost);
	for(i = 0; i < rd; i++)
		for(j = 0; j < cd; j++)
		{
			cqsfloat z = _cgqs_unpack(h->re[i * cd + j], h->im[i * cd + j]);
			set_cqsmatrix_ij(cpu, i, j, &z);
		}
	free_cgqsmatrix(h);
}

// complex matmul ret := A * B  (gqs_real arithmetic)
__global__ void _bncu_mul_cgqsmatrix(gqs_real *cre, gqs_real *cim, long int row_dim, long int col_dim, long int mid_dim,
                                     gqs_real *are, gqs_real *aim, gqs_real *bre, gqs_real *bim)
{
	long int i, j, k;
	i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < row_dim)
	{
		for(j = 0; j < col_dim; j++)
		{
			gqs_real sre = cg_zero(), sim = cg_zero();
			for(k = 0; k < mid_dim; k++)
			{
				gqs_real ar = are[i * mid_dim + k], ai = aim[i * mid_dim + k];
				gqs_real br = bre[k * col_dim + j], bi = bim[k * col_dim + j];
				sre = sre + (ar * br - ai * bi);
				sim = sim + (ar * bi + ai * br);
			}
			cre[i * col_dim + j] = sre;
			cim[i * col_dim + j] = sim;
		}
		i += blockDim.x * gridDim.x;
	}
}
__host__ void mul_cgqsmatrix_dev(CGQSMatrix ret, CGQSMatrix a, CGQSMatrix b, int nbg, int ntb)
{
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{ fprintf(stderr, "ERROR: mul_cgqsmatrix_dev\n"); return; }
	_bncu_mul_cgqsmatrix<<<nbg, ntb>>>(ret->re, ret->im, ret->row_dim, ret->col_dim, a->col_dim, a->re, a->im, b->re, b->im);
}

__host__ void add_cgqsmatrix_dev(CGQSMatrix c, CGQSMatrix a, CGQSMatrix b, int nbg, int ntb)
{ _bncu_add_cgqsvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->row_dim * c->col_dim); }
__host__ void sub_cgqsmatrix_dev(CGQSMatrix c, CGQSMatrix a, CGQSMatrix b, int nbg, int ntb)
{ _bncu_sub_cgqsvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->row_dim * c->col_dim); }
__host__ void cmul_cgqsmatrix_dev(CGQSMatrix c, gqs_real sre, gqs_real sim, CGQSMatrix a, int nbg, int ntb)
{ _bncu_cmul_cgqsvector<<<nbg, ntb>>>(c->re, c->im, sre, sim, a->re, a->im, c->row_dim * c->col_dim); }

__global__ void _bncu_transpose_cgqsmatrix(gqs_real *cre, gqs_real *cim, gqs_real *are, gqs_real *aim, long int row_dim, long int col_dim, int conj)
{
	long int i, j;
	i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < row_dim)
	{
		for(j = 0; j < col_dim; j++)
		{
			cre[i * col_dim + j] = are[j * row_dim + i];
			if(conj) cim[i * col_dim + j] = cg_neg(aim[j * row_dim + i]);
			else     cim[i * col_dim + j] = aim[j * row_dim + i];
		}
		i += blockDim.x * gridDim.x;
	}
}
__host__ void transpose_cgqsmatrix_dev(CGQSMatrix c, CGQSMatrix a, int nbg, int ntb)
{ _bncu_transpose_cgqsmatrix<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim, c->col_dim, 0); }
__host__ void conjtrans_cgqsmatrix_dev(CGQSMatrix c, CGQSMatrix a, int nbg, int ntb)
{ _bncu_transpose_cgqsmatrix<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim, c->col_dim, 1); }
__host__ void subst_cgqsmatrix_dev(CGQSMatrix c, CGQSMatrix a, int nbg, int ntb)
{ _bncu_subst_cgqsvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim * c->col_dim); }
__host__ void set0_cgqsmatrix_dev(CGQSMatrix c, int nbg, int ntb)
{ _bncu_set0_cgqsvector<<<nbg, ntb>>>(c->re, c->im, c->row_dim * c->col_dim); }

__host__ void setI_cgqsmatrix_dev(CGQSMatrix c, int nbg, int ntb)
{
	long int i;
	_bncu_set0_cgqsvector<<<nbg, ntb>>>(c->re, c->im, c->row_dim * c->col_dim);
	gqs_real one = cg_host_one();
	for(i = 0; i < c->row_dim; i++)
		cudaMemcpy((void *)&(c->re[i * c->col_dim + i]), (void *)&one, sizeof(gqs_real), cudaMemcpyHostToDevice);
}

// v := a * vb
__global__ void _bncu_mul_cgqsmatrix_cgqsvec(gqs_real *vre, gqs_real *vim, gqs_real *are, gqs_real *aim, gqs_real *bre, gqs_real *bim, long int row_dim, long int col_dim)
{
	long int j, index;
	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < row_dim)
	{
		gqs_real sre = cg_zero(), sim = cg_zero();
		for(j = 0; j < col_dim; j++)
		{
			gqs_real ar = are[index * col_dim + j], ai = aim[index * col_dim + j];
			gqs_real br = bre[j], bi = bim[j];
			sre = sre + (ar * br - ai * bi);
			sim = sim + (ar * bi + ai * br);
		}
		vre[index] = sre; vim[index] = sim;
		index += blockDim.x * gridDim.x;
	}
}
__host__ void mul_cgqsmatrix_cgqsvec(CGQSVector v, CGQSMatrix a, CGQSVector vb, int nbg, int ntb)
{
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim)) { fprintf(stderr, "ERROR: mul_cgqsmatrix_cgqsvec\n"); return; }
	_bncu_mul_cgqsmatrix_cgqsvec<<<nbg, ntb>>>(v->re, v->im, a->re, a->im, vb->re, vb->im, a->row_dim, a->col_dim);
}

// v := a^T * vb (non-conjugate)
__global__ void _bncu_mul_cgqsmatrixt_cgqsvec(gqs_real *vre, gqs_real *vim, gqs_real *are, gqs_real *aim, gqs_real *bre, gqs_real *bim, long int row_dim, long int col_dim)
{
	long int j, index;
	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < col_dim)
	{
		gqs_real sre = cg_zero(), sim = cg_zero();
		for(j = 0; j < row_dim; j++)
		{
			gqs_real ar = are[j * col_dim + index], ai = aim[j * col_dim + index];
			gqs_real br = bre[j], bi = bim[j];
			sre = sre + (ar * br - ai * bi);
			sim = sim + (ar * bi + ai * br);
		}
		vre[index] = sre; vim[index] = sim;
		index += blockDim.x * gridDim.x;
	}
}
__host__ void mul_cgqsmatrixt_cgqsvec(CGQSVector v, CGQSMatrix a, CGQSVector vb, int nbg, int ntb)
{
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim)) { fprintf(stderr, "ERROR: mul_cgqsmatrixt_cgqsvec\n"); return; }
	_bncu_mul_cgqsmatrixt_cgqsvec<<<nbg, ntb>>>(v->re, v->im, a->re, a->im, vb->re, vb->im, a->row_dim, a->col_dim);
}
