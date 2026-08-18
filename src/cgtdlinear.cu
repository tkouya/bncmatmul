/********************************************************************************/
/* cgtdlinear.cu: Complex double-double GPU Linear Computation (CUDA)            */
/*   complex as separate real/imag gtd_real arrays (SoA); device dd math.        */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#include "cgtdlinear.h"
#include "gqd.cu"   // device gtd_real arithmetic implementation

// per-type zero/one/neg via limb pointer-cast (component count = CG_NLIMB).
// Generic over double2/3/4 (and float2/3/4 after sed double->float).
#define CG_NLIMB CGTD_SIZE
typedef double cg_scalar;
__device__ static inline gtd_real cg_zero() { gtd_real z; cg_scalar *p = (cg_scalar *)&z; for(int l = 0; l < CG_NLIMB; l++) p[l] = 0; return z; }
__device__ static inline gtd_real cg_neg(gtd_real v) { cg_scalar *p = (cg_scalar *)&v; for(int l = 0; l < CG_NLIMB; l++) p[l] = -p[l]; return v; }
static inline gtd_real cg_host_one() { gtd_real z; cg_scalar *p = (cg_scalar *)&z; for(int l = 0; l < CG_NLIMB; l++) p[l] = 0; p[0] = 1; return z; }

/*--- host<->device element conversion helpers (limb pointer-cast) ---*/
static inline void _cgtd_pack(gtd_real *re, gtd_real *im, ctdfloat z)
{
	double *pre = (double *)re, *pim = (double *)im;
	for(int l = 0; l < CGTD_SIZE; l++) { pre[l] = z.val_re[l]; pim[l] = z.val_im[l]; }
}
static inline ctdfloat _cgtd_unpack(gtd_real re, gtd_real im)
{
	ctdfloat z;
	double *pre = (double *)&re, *pim = (double *)&im;
	for(int l = 0; l < CGTD_SIZE; l++) { z.val_re[l] = pre[l]; z.val_im[l] = pim[l]; }
	return z;
}

/*--- vector ---*/
__host__ CGTDVector init_cgtdvector(long int dim)
{
	CGTDVector r = (cgtdvector *)malloc(sizeof(cgtdvector));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgtdvector\n"); return NULL; }
	r->dim = dim;
	r->re = (gtd_real *)calloc(dim, sizeof(gtd_real));
	r->im = (gtd_real *)calloc(dim, sizeof(gtd_real));
	return r;
}
__host__ void free_cgtdvector(CGTDVector v) { free(v->re); free(v->im); free(v); }
__host__ CGTDVector init_cgtdvector_dev(long int dim)
{
	CGTDVector r = (cgtdvector *)malloc(sizeof(cgtdvector));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgtdvector_dev\n"); return NULL; }
	r->dim = dim; r->re = NULL; r->im = NULL;
	cudaMalloc((void **)&(r->re), (size_t)(dim * sizeof(gtd_real)));
	cudaMalloc((void **)&(r->im), (size_t)(dim * sizeof(gtd_real)));
	cudaMemset((void *)(r->re), 0, (size_t)(dim * sizeof(gtd_real)));
	cudaMemset((void *)(r->im), 0, (size_t)(dim * sizeof(gtd_real)));
	return r;
}
__host__ void free_cgtdvector_dev(CGTDVector v) { cudaFree(v->re); cudaFree(v->im); free(v); }

__host__ void subst_cgtdvector_dev_ctdvec(CGTDVector dev, CTDVector cpu)
{
	long int i, dim = dev->dim;
	CGTDVector h = init_cgtdvector(dim);
	for(i = 0; i < dim; i++)
		_cgtd_pack(&h->re[i], &h->im[i], get_ctdvector_i_ctdfloat(cpu, i));
	cudaMemcpy((void *)dev->re, (void *)h->re, (size_t)(dim * sizeof(gtd_real)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dev->im, (void *)h->im, (size_t)(dim * sizeof(gtd_real)), cudaMemcpyHostToDevice);
	free_cgtdvector(h);
}
__host__ void subst_ctdvector_cgtdvec_dev(CTDVector cpu, CGTDVector dev)
{
	long int i, dim = dev->dim;
	CGTDVector h = init_cgtdvector(dim);
	cudaMemcpy((void *)h->re, (void *)dev->re, (size_t)(dim * sizeof(gtd_real)), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)dev->im, (size_t)(dim * sizeof(gtd_real)), cudaMemcpyDeviceToHost);
	for(i = 0; i < dim; i++)
	{
		ctdfloat z = _cgtd_unpack(h->re[i], h->im[i]);
		set_ctdvector_i(cpu, i, &z);
	}
	free_cgtdvector(h);
}

__global__ void _bncu_add_cgtdvector(gtd_real *cre, gtd_real *cim, gtd_real *are, gtd_real *aim, gtd_real *bre, gtd_real *bim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx] + bre[idx]; cim[idx] = aim[idx] + bim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void add_cgtdvector_dev(CGTDVector c, CGTDVector a, CGTDVector b, int nbg, int ntb)
{ _bncu_add_cgtdvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->dim); }

__global__ void _bncu_sub_cgtdvector(gtd_real *cre, gtd_real *cim, gtd_real *are, gtd_real *aim, gtd_real *bre, gtd_real *bim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx] - bre[idx]; cim[idx] = aim[idx] - bim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void sub_cgtdvector_dev(CGTDVector c, CGTDVector a, CGTDVector b, int nbg, int ntb)
{ _bncu_sub_cgtdvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->dim); }

__global__ void _bncu_cmul_cgtdvector(gtd_real *cre, gtd_real *cim, gtd_real vre, gtd_real vim, gtd_real *are, gtd_real *aim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim)
	{
		gtd_real ar = are[idx], ai = aim[idx];
		cre[idx] = vre * ar - vim * ai;
		cim[idx] = vre * ai + vim * ar;
		idx += blockDim.x * gridDim.x;
	}
}
__host__ void cmul_cgtdvector_dev(CGTDVector c, gtd_real vre, gtd_real vim, CGTDVector a, int nbg, int ntb)
{ _bncu_cmul_cgtdvector<<<nbg, ntb>>>(c->re, c->im, vre, vim, a->re, a->im, c->dim); }

__global__ void _bncu_subst_cgtdvector(gtd_real *cre, gtd_real *cim, gtd_real *are, gtd_real *aim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx]; cim[idx] = aim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void subst_cgtdvector_dev(CGTDVector ret, CGTDVector v, int nbg, int ntb)
{ _bncu_subst_cgtdvector<<<nbg, ntb>>>(ret->re, ret->im, v->re, v->im, ret->dim); }

__global__ void _bncu_set0_cgtdvector(gtd_real *cre, gtd_real *cim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = cg_zero(); cim[idx] = cg_zero(); idx += blockDim.x * gridDim.x; }
}
__host__ void set0_cgtdvector_dev(CGTDVector ret, int nbg, int ntb)
{ _bncu_set0_cgtdvector<<<nbg, ntb>>>(ret->re, ret->im, ret->dim); }

/*--- matrix ---*/
__host__ CGTDMatrix init_cgtdmatrix(long int row_dim, long int col_dim)
{
	CGTDMatrix r = (cgtdmatrix *)malloc(sizeof(cgtdmatrix));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgtdmatrix\n"); return NULL; }
	r->row_dim = row_dim; r->col_dim = col_dim;
	r->re = (gtd_real *)calloc(row_dim * col_dim, sizeof(gtd_real));
	r->im = (gtd_real *)calloc(row_dim * col_dim, sizeof(gtd_real));
	return r;
}
__host__ void free_cgtdmatrix(CGTDMatrix m) { free(m->re); free(m->im); free(m); }
__host__ CGTDMatrix init_cgtdmatrix_dev(long int row_dim, long int col_dim)
{
	CGTDMatrix r = (cgtdmatrix *)malloc(sizeof(cgtdmatrix));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgtdmatrix_dev\n"); return NULL; }
	r->row_dim = row_dim; r->col_dim = col_dim; r->re = NULL; r->im = NULL;
	cudaMalloc((void **)&(r->re), (size_t)(row_dim * col_dim * sizeof(gtd_real)));
	cudaMalloc((void **)&(r->im), (size_t)(row_dim * col_dim * sizeof(gtd_real)));
	cudaMemset((void *)(r->re), 0, (size_t)(row_dim * col_dim * sizeof(gtd_real)));
	cudaMemset((void *)(r->im), 0, (size_t)(row_dim * col_dim * sizeof(gtd_real)));
	return r;
}
__host__ void free_cgtdmatrix_dev(CGTDMatrix m) { cudaFree(m->re); cudaFree(m->im); free(m); }

__host__ void subst_cgtdmatrix_dev_ctdmat(CGTDMatrix dev, CTDMatrix cpu)
{
	long int i, j, rd = dev->row_dim, cd = dev->col_dim;
	CGTDMatrix h = init_cgtdmatrix(rd, cd);
	for(i = 0; i < rd; i++)
		for(j = 0; j < cd; j++)
			_cgtd_pack(&h->re[i * cd + j], &h->im[i * cd + j], get_ctdmatrix_ij_ctdfloat(cpu, i, j));
	cudaMemcpy((void *)dev->re, (void *)h->re, (size_t)(sizeof(gtd_real) * rd * cd), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dev->im, (void *)h->im, (size_t)(sizeof(gtd_real) * rd * cd), cudaMemcpyHostToDevice);
	free_cgtdmatrix(h);
}
__host__ void subst_ctdmatrix_cgtdmat_dev(CTDMatrix cpu, CGTDMatrix dev)
{
	long int i, j, rd = dev->row_dim, cd = dev->col_dim;
	CGTDMatrix h = init_cgtdmatrix(rd, cd);
	cudaMemcpy((void *)h->re, (void *)dev->re, (size_t)(sizeof(gtd_real) * rd * cd), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)dev->im, (size_t)(sizeof(gtd_real) * rd * cd), cudaMemcpyDeviceToHost);
	for(i = 0; i < rd; i++)
		for(j = 0; j < cd; j++)
		{
			ctdfloat z = _cgtd_unpack(h->re[i * cd + j], h->im[i * cd + j]);
			set_ctdmatrix_ij(cpu, i, j, &z);
		}
	free_cgtdmatrix(h);
}

// complex matmul ret := A * B  (gtd_real arithmetic)
__global__ void _bncu_mul_cgtdmatrix(gtd_real *cre, gtd_real *cim, long int row_dim, long int col_dim, long int mid_dim,
                                     gtd_real *are, gtd_real *aim, gtd_real *bre, gtd_real *bim)
{
	long int i, j, k;
	i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < row_dim)
	{
		for(j = 0; j < col_dim; j++)
		{
			gtd_real sre = cg_zero(), sim = cg_zero();
			for(k = 0; k < mid_dim; k++)
			{
				gtd_real ar = are[i * mid_dim + k], ai = aim[i * mid_dim + k];
				gtd_real br = bre[k * col_dim + j], bi = bim[k * col_dim + j];
				sre = sre + (ar * br - ai * bi);
				sim = sim + (ar * bi + ai * br);
			}
			cre[i * col_dim + j] = sre;
			cim[i * col_dim + j] = sim;
		}
		i += blockDim.x * gridDim.x;
	}
}
__host__ void mul_cgtdmatrix_dev(CGTDMatrix ret, CGTDMatrix a, CGTDMatrix b, int nbg, int ntb)
{
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{ fprintf(stderr, "ERROR: mul_cgtdmatrix_dev\n"); return; }
	_bncu_mul_cgtdmatrix<<<nbg, ntb>>>(ret->re, ret->im, ret->row_dim, ret->col_dim, a->col_dim, a->re, a->im, b->re, b->im);
}

__host__ void add_cgtdmatrix_dev(CGTDMatrix c, CGTDMatrix a, CGTDMatrix b, int nbg, int ntb)
{ _bncu_add_cgtdvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->row_dim * c->col_dim); }
__host__ void sub_cgtdmatrix_dev(CGTDMatrix c, CGTDMatrix a, CGTDMatrix b, int nbg, int ntb)
{ _bncu_sub_cgtdvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->row_dim * c->col_dim); }
__host__ void cmul_cgtdmatrix_dev(CGTDMatrix c, gtd_real sre, gtd_real sim, CGTDMatrix a, int nbg, int ntb)
{ _bncu_cmul_cgtdvector<<<nbg, ntb>>>(c->re, c->im, sre, sim, a->re, a->im, c->row_dim * c->col_dim); }

__global__ void _bncu_transpose_cgtdmatrix(gtd_real *cre, gtd_real *cim, gtd_real *are, gtd_real *aim, long int row_dim, long int col_dim, int conj)
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
__host__ void transpose_cgtdmatrix_dev(CGTDMatrix c, CGTDMatrix a, int nbg, int ntb)
{ _bncu_transpose_cgtdmatrix<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim, c->col_dim, 0); }
__host__ void conjtrans_cgtdmatrix_dev(CGTDMatrix c, CGTDMatrix a, int nbg, int ntb)
{ _bncu_transpose_cgtdmatrix<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim, c->col_dim, 1); }
__host__ void subst_cgtdmatrix_dev(CGTDMatrix c, CGTDMatrix a, int nbg, int ntb)
{ _bncu_subst_cgtdvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim * c->col_dim); }
__host__ void set0_cgtdmatrix_dev(CGTDMatrix c, int nbg, int ntb)
{ _bncu_set0_cgtdvector<<<nbg, ntb>>>(c->re, c->im, c->row_dim * c->col_dim); }

__host__ void setI_cgtdmatrix_dev(CGTDMatrix c, int nbg, int ntb)
{
	long int i;
	_bncu_set0_cgtdvector<<<nbg, ntb>>>(c->re, c->im, c->row_dim * c->col_dim);
	gtd_real one = cg_host_one();
	for(i = 0; i < c->row_dim; i++)
		cudaMemcpy((void *)&(c->re[i * c->col_dim + i]), (void *)&one, sizeof(gtd_real), cudaMemcpyHostToDevice);
}

// v := a * vb
__global__ void _bncu_mul_cgtdmatrix_cgtdvec(gtd_real *vre, gtd_real *vim, gtd_real *are, gtd_real *aim, gtd_real *bre, gtd_real *bim, long int row_dim, long int col_dim)
{
	long int j, index;
	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < row_dim)
	{
		gtd_real sre = cg_zero(), sim = cg_zero();
		for(j = 0; j < col_dim; j++)
		{
			gtd_real ar = are[index * col_dim + j], ai = aim[index * col_dim + j];
			gtd_real br = bre[j], bi = bim[j];
			sre = sre + (ar * br - ai * bi);
			sim = sim + (ar * bi + ai * br);
		}
		vre[index] = sre; vim[index] = sim;
		index += blockDim.x * gridDim.x;
	}
}
__host__ void mul_cgtdmatrix_cgtdvec(CGTDVector v, CGTDMatrix a, CGTDVector vb, int nbg, int ntb)
{
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim)) { fprintf(stderr, "ERROR: mul_cgtdmatrix_cgtdvec\n"); return; }
	_bncu_mul_cgtdmatrix_cgtdvec<<<nbg, ntb>>>(v->re, v->im, a->re, a->im, vb->re, vb->im, a->row_dim, a->col_dim);
}

// v := a^T * vb (non-conjugate)
__global__ void _bncu_mul_cgtdmatrixt_cgtdvec(gtd_real *vre, gtd_real *vim, gtd_real *are, gtd_real *aim, gtd_real *bre, gtd_real *bim, long int row_dim, long int col_dim)
{
	long int j, index;
	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < col_dim)
	{
		gtd_real sre = cg_zero(), sim = cg_zero();
		for(j = 0; j < row_dim; j++)
		{
			gtd_real ar = are[j * col_dim + index], ai = aim[j * col_dim + index];
			gtd_real br = bre[j], bi = bim[j];
			sre = sre + (ar * br - ai * bi);
			sim = sim + (ar * bi + ai * br);
		}
		vre[index] = sre; vim[index] = sim;
		index += blockDim.x * gridDim.x;
	}
}
__host__ void mul_cgtdmatrixt_cgtdvec(CGTDVector v, CGTDMatrix a, CGTDVector vb, int nbg, int ntb)
{
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim)) { fprintf(stderr, "ERROR: mul_cgtdmatrixt_cgtdvec\n"); return; }
	_bncu_mul_cgtdmatrixt_cgtdvec<<<nbg, ntb>>>(v->re, v->im, a->re, a->im, vb->re, vb->im, a->row_dim, a->col_dim);
}
