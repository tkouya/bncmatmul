/********************************************************************************/
/* cgddlinear.cu: Complex double-double GPU Linear Computation (CUDA)            */
/*   complex as separate real/imag gdd_real arrays (SoA); device dd math.        */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#include "cgddlinear.h"
#include "gqd.cu"   // device gdd_real arithmetic implementation

// per-type zero/one/neg via limb pointer-cast (component count = CG_NLIMB).
// Generic over double2/3/4 (and float2/3/4 after sed double->float).
#define CG_NLIMB CGDD_SIZE
typedef double cg_scalar;
__device__ static inline gdd_real cg_zero() { gdd_real z; cg_scalar *p = (cg_scalar *)&z; for(int l = 0; l < CG_NLIMB; l++) p[l] = 0; return z; }
__device__ static inline gdd_real cg_neg(gdd_real v) { cg_scalar *p = (cg_scalar *)&v; for(int l = 0; l < CG_NLIMB; l++) p[l] = -p[l]; return v; }
static inline gdd_real cg_host_one() { gdd_real z; cg_scalar *p = (cg_scalar *)&z; for(int l = 0; l < CG_NLIMB; l++) p[l] = 0; p[0] = 1; return z; }

/*--- host<->device element conversion helpers (limb pointer-cast) ---*/
static inline void _cgdd_pack(gdd_real *re, gdd_real *im, cddfloat z)
{
	double *pre = (double *)re, *pim = (double *)im;
	for(int l = 0; l < CGDD_SIZE; l++) { pre[l] = z.val_re[l]; pim[l] = z.val_im[l]; }
}
static inline cddfloat _cgdd_unpack(gdd_real re, gdd_real im)
{
	cddfloat z;
	double *pre = (double *)&re, *pim = (double *)&im;
	for(int l = 0; l < CGDD_SIZE; l++) { z.val_re[l] = pre[l]; z.val_im[l] = pim[l]; }
	return z;
}

/*--- vector ---*/
__host__ CGDDVector init_cgddvector(long int dim)
{
	CGDDVector r = (cgddvector *)malloc(sizeof(cgddvector));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgddvector\n"); return NULL; }
	r->dim = dim;
	r->re = (gdd_real *)calloc(dim, sizeof(gdd_real));
	r->im = (gdd_real *)calloc(dim, sizeof(gdd_real));
	return r;
}
__host__ void free_cgddvector(CGDDVector v) { free(v->re); free(v->im); free(v); }
__host__ CGDDVector init_cgddvector_dev(long int dim)
{
	CGDDVector r = (cgddvector *)malloc(sizeof(cgddvector));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgddvector_dev\n"); return NULL; }
	r->dim = dim; r->re = NULL; r->im = NULL;
	cudaMalloc((void **)&(r->re), (size_t)(dim * sizeof(gdd_real)));
	cudaMalloc((void **)&(r->im), (size_t)(dim * sizeof(gdd_real)));
	cudaMemset((void *)(r->re), 0, (size_t)(dim * sizeof(gdd_real)));
	cudaMemset((void *)(r->im), 0, (size_t)(dim * sizeof(gdd_real)));
	return r;
}
__host__ void free_cgddvector_dev(CGDDVector v) { cudaFree(v->re); cudaFree(v->im); free(v); }

__host__ void subst_cgddvector_dev_cddvec(CGDDVector dev, CDDVector cpu)
{
	long int i, dim = dev->dim;
	CGDDVector h = init_cgddvector(dim);
	for(i = 0; i < dim; i++)
		_cgdd_pack(&h->re[i], &h->im[i], get_cddvector_i_cddfloat(cpu, i));
	cudaMemcpy((void *)dev->re, (void *)h->re, (size_t)(dim * sizeof(gdd_real)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dev->im, (void *)h->im, (size_t)(dim * sizeof(gdd_real)), cudaMemcpyHostToDevice);
	free_cgddvector(h);
}
__host__ void subst_cddvector_cgddvec_dev(CDDVector cpu, CGDDVector dev)
{
	long int i, dim = dev->dim;
	CGDDVector h = init_cgddvector(dim);
	cudaMemcpy((void *)h->re, (void *)dev->re, (size_t)(dim * sizeof(gdd_real)), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)dev->im, (size_t)(dim * sizeof(gdd_real)), cudaMemcpyDeviceToHost);
	for(i = 0; i < dim; i++)
	{
		cddfloat z = _cgdd_unpack(h->re[i], h->im[i]);
		set_cddvector_i(cpu, i, &z);
	}
	free_cgddvector(h);
}

__global__ void _bncu_add_cgddvector(gdd_real *cre, gdd_real *cim, gdd_real *are, gdd_real *aim, gdd_real *bre, gdd_real *bim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx] + bre[idx]; cim[idx] = aim[idx] + bim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void add_cgddvector_dev(CGDDVector c, CGDDVector a, CGDDVector b, int nbg, int ntb)
{ _bncu_add_cgddvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->dim); }

__global__ void _bncu_sub_cgddvector(gdd_real *cre, gdd_real *cim, gdd_real *are, gdd_real *aim, gdd_real *bre, gdd_real *bim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx] - bre[idx]; cim[idx] = aim[idx] - bim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void sub_cgddvector_dev(CGDDVector c, CGDDVector a, CGDDVector b, int nbg, int ntb)
{ _bncu_sub_cgddvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->dim); }

__global__ void _bncu_cmul_cgddvector(gdd_real *cre, gdd_real *cim, gdd_real vre, gdd_real vim, gdd_real *are, gdd_real *aim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim)
	{
		gdd_real ar = are[idx], ai = aim[idx];
		cre[idx] = vre * ar - vim * ai;
		cim[idx] = vre * ai + vim * ar;
		idx += blockDim.x * gridDim.x;
	}
}
__host__ void cmul_cgddvector_dev(CGDDVector c, gdd_real vre, gdd_real vim, CGDDVector a, int nbg, int ntb)
{ _bncu_cmul_cgddvector<<<nbg, ntb>>>(c->re, c->im, vre, vim, a->re, a->im, c->dim); }

__global__ void _bncu_subst_cgddvector(gdd_real *cre, gdd_real *cim, gdd_real *are, gdd_real *aim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx]; cim[idx] = aim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void subst_cgddvector_dev(CGDDVector ret, CGDDVector v, int nbg, int ntb)
{ _bncu_subst_cgddvector<<<nbg, ntb>>>(ret->re, ret->im, v->re, v->im, ret->dim); }

__global__ void _bncu_set0_cgddvector(gdd_real *cre, gdd_real *cim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = cg_zero(); cim[idx] = cg_zero(); idx += blockDim.x * gridDim.x; }
}
__host__ void set0_cgddvector_dev(CGDDVector ret, int nbg, int ntb)
{ _bncu_set0_cgddvector<<<nbg, ntb>>>(ret->re, ret->im, ret->dim); }

/*--- matrix ---*/
__host__ CGDDMatrix init_cgddmatrix(long int row_dim, long int col_dim)
{
	CGDDMatrix r = (cgddmatrix *)malloc(sizeof(cgddmatrix));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgddmatrix\n"); return NULL; }
	r->row_dim = row_dim; r->col_dim = col_dim;
	r->re = (gdd_real *)calloc(row_dim * col_dim, sizeof(gdd_real));
	r->im = (gdd_real *)calloc(row_dim * col_dim, sizeof(gdd_real));
	return r;
}
__host__ void free_cgddmatrix(CGDDMatrix m) { free(m->re); free(m->im); free(m); }
__host__ CGDDMatrix init_cgddmatrix_dev(long int row_dim, long int col_dim)
{
	CGDDMatrix r = (cgddmatrix *)malloc(sizeof(cgddmatrix));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgddmatrix_dev\n"); return NULL; }
	r->row_dim = row_dim; r->col_dim = col_dim; r->re = NULL; r->im = NULL;
	cudaMalloc((void **)&(r->re), (size_t)(row_dim * col_dim * sizeof(gdd_real)));
	cudaMalloc((void **)&(r->im), (size_t)(row_dim * col_dim * sizeof(gdd_real)));
	cudaMemset((void *)(r->re), 0, (size_t)(row_dim * col_dim * sizeof(gdd_real)));
	cudaMemset((void *)(r->im), 0, (size_t)(row_dim * col_dim * sizeof(gdd_real)));
	return r;
}
__host__ void free_cgddmatrix_dev(CGDDMatrix m) { cudaFree(m->re); cudaFree(m->im); free(m); }

__host__ void subst_cgddmatrix_dev_cddmat(CGDDMatrix dev, CDDMatrix cpu)
{
	long int i, j, rd = dev->row_dim, cd = dev->col_dim;
	CGDDMatrix h = init_cgddmatrix(rd, cd);
	for(i = 0; i < rd; i++)
		for(j = 0; j < cd; j++)
			_cgdd_pack(&h->re[i * cd + j], &h->im[i * cd + j], get_cddmatrix_ij_cddfloat(cpu, i, j));
	cudaMemcpy((void *)dev->re, (void *)h->re, (size_t)(sizeof(gdd_real) * rd * cd), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dev->im, (void *)h->im, (size_t)(sizeof(gdd_real) * rd * cd), cudaMemcpyHostToDevice);
	free_cgddmatrix(h);
}
__host__ void subst_cddmatrix_cgddmat_dev(CDDMatrix cpu, CGDDMatrix dev)
{
	long int i, j, rd = dev->row_dim, cd = dev->col_dim;
	CGDDMatrix h = init_cgddmatrix(rd, cd);
	cudaMemcpy((void *)h->re, (void *)dev->re, (size_t)(sizeof(gdd_real) * rd * cd), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)dev->im, (size_t)(sizeof(gdd_real) * rd * cd), cudaMemcpyDeviceToHost);
	for(i = 0; i < rd; i++)
		for(j = 0; j < cd; j++)
		{
			cddfloat z = _cgdd_unpack(h->re[i * cd + j], h->im[i * cd + j]);
			set_cddmatrix_ij(cpu, i, j, &z);
		}
	free_cgddmatrix(h);
}

// complex matmul ret := A * B  (gdd_real arithmetic)
__global__ void _bncu_mul_cgddmatrix(gdd_real *cre, gdd_real *cim, long int row_dim, long int col_dim, long int mid_dim,
                                     gdd_real *are, gdd_real *aim, gdd_real *bre, gdd_real *bim)
{
	long int i, j, k;
	i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < row_dim)
	{
		for(j = 0; j < col_dim; j++)
		{
			gdd_real sre = cg_zero(), sim = cg_zero();
			for(k = 0; k < mid_dim; k++)
			{
				gdd_real ar = are[i * mid_dim + k], ai = aim[i * mid_dim + k];
				gdd_real br = bre[k * col_dim + j], bi = bim[k * col_dim + j];
				sre = sre + (ar * br - ai * bi);
				sim = sim + (ar * bi + ai * br);
			}
			cre[i * col_dim + j] = sre;
			cim[i * col_dim + j] = sim;
		}
		i += blockDim.x * gridDim.x;
	}
}
__host__ void mul_cgddmatrix_dev(CGDDMatrix ret, CGDDMatrix a, CGDDMatrix b, int nbg, int ntb)
{
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{ fprintf(stderr, "ERROR: mul_cgddmatrix_dev\n"); return; }
	_bncu_mul_cgddmatrix<<<nbg, ntb>>>(ret->re, ret->im, ret->row_dim, ret->col_dim, a->col_dim, a->re, a->im, b->re, b->im);
}

__host__ void add_cgddmatrix_dev(CGDDMatrix c, CGDDMatrix a, CGDDMatrix b, int nbg, int ntb)
{ _bncu_add_cgddvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->row_dim * c->col_dim); }
__host__ void sub_cgddmatrix_dev(CGDDMatrix c, CGDDMatrix a, CGDDMatrix b, int nbg, int ntb)
{ _bncu_sub_cgddvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->row_dim * c->col_dim); }
__host__ void cmul_cgddmatrix_dev(CGDDMatrix c, gdd_real sre, gdd_real sim, CGDDMatrix a, int nbg, int ntb)
{ _bncu_cmul_cgddvector<<<nbg, ntb>>>(c->re, c->im, sre, sim, a->re, a->im, c->row_dim * c->col_dim); }

__global__ void _bncu_transpose_cgddmatrix(gdd_real *cre, gdd_real *cim, gdd_real *are, gdd_real *aim, long int row_dim, long int col_dim, int conj)
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
__host__ void transpose_cgddmatrix_dev(CGDDMatrix c, CGDDMatrix a, int nbg, int ntb)
{ _bncu_transpose_cgddmatrix<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim, c->col_dim, 0); }
__host__ void conjtrans_cgddmatrix_dev(CGDDMatrix c, CGDDMatrix a, int nbg, int ntb)
{ _bncu_transpose_cgddmatrix<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim, c->col_dim, 1); }
__host__ void subst_cgddmatrix_dev(CGDDMatrix c, CGDDMatrix a, int nbg, int ntb)
{ _bncu_subst_cgddvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim * c->col_dim); }
__host__ void set0_cgddmatrix_dev(CGDDMatrix c, int nbg, int ntb)
{ _bncu_set0_cgddvector<<<nbg, ntb>>>(c->re, c->im, c->row_dim * c->col_dim); }

__host__ void setI_cgddmatrix_dev(CGDDMatrix c, int nbg, int ntb)
{
	long int i;
	_bncu_set0_cgddvector<<<nbg, ntb>>>(c->re, c->im, c->row_dim * c->col_dim);
	gdd_real one = cg_host_one();
	for(i = 0; i < c->row_dim; i++)
		cudaMemcpy((void *)&(c->re[i * c->col_dim + i]), (void *)&one, sizeof(gdd_real), cudaMemcpyHostToDevice);
}

// v := a * vb
__global__ void _bncu_mul_cgddmatrix_cgddvec(gdd_real *vre, gdd_real *vim, gdd_real *are, gdd_real *aim, gdd_real *bre, gdd_real *bim, long int row_dim, long int col_dim)
{
	long int j, index;
	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < row_dim)
	{
		gdd_real sre = cg_zero(), sim = cg_zero();
		for(j = 0; j < col_dim; j++)
		{
			gdd_real ar = are[index * col_dim + j], ai = aim[index * col_dim + j];
			gdd_real br = bre[j], bi = bim[j];
			sre = sre + (ar * br - ai * bi);
			sim = sim + (ar * bi + ai * br);
		}
		vre[index] = sre; vim[index] = sim;
		index += blockDim.x * gridDim.x;
	}
}
__host__ void mul_cgddmatrix_cgddvec(CGDDVector v, CGDDMatrix a, CGDDVector vb, int nbg, int ntb)
{
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim)) { fprintf(stderr, "ERROR: mul_cgddmatrix_cgddvec\n"); return; }
	_bncu_mul_cgddmatrix_cgddvec<<<nbg, ntb>>>(v->re, v->im, a->re, a->im, vb->re, vb->im, a->row_dim, a->col_dim);
}

// v := a^T * vb (non-conjugate)
__global__ void _bncu_mul_cgddmatrixt_cgddvec(gdd_real *vre, gdd_real *vim, gdd_real *are, gdd_real *aim, gdd_real *bre, gdd_real *bim, long int row_dim, long int col_dim)
{
	long int j, index;
	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < col_dim)
	{
		gdd_real sre = cg_zero(), sim = cg_zero();
		for(j = 0; j < row_dim; j++)
		{
			gdd_real ar = are[j * col_dim + index], ai = aim[j * col_dim + index];
			gdd_real br = bre[j], bi = bim[j];
			sre = sre + (ar * br - ai * bi);
			sim = sim + (ar * bi + ai * br);
		}
		vre[index] = sre; vim[index] = sim;
		index += blockDim.x * gridDim.x;
	}
}
__host__ void mul_cgddmatrixt_cgddvec(CGDDVector v, CGDDMatrix a, CGDDVector vb, int nbg, int ntb)
{
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim)) { fprintf(stderr, "ERROR: mul_cgddmatrixt_cgddvec\n"); return; }
	_bncu_mul_cgddmatrixt_cgddvec<<<nbg, ntb>>>(v->re, v->im, a->re, a->im, vb->re, vb->im, a->row_dim, a->col_dim);
}
