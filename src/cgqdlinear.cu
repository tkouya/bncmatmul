/********************************************************************************/
/* cgqdlinear.cu: Complex double-double GPU Linear Computation (CUDA)            */
/*   complex as separate real/imag gqd_real arrays (SoA); device dd math.        */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#include "cgqdlinear.h"
#include "gqd.cu"   // device gqd_real arithmetic implementation

// per-type zero/one/neg via limb pointer-cast (component count = CG_NLIMB).
// Generic over double2/3/4 (and float2/3/4 after sed double->float).
#define CG_NLIMB CGQD_SIZE
typedef double cg_scalar;
__device__ static inline gqd_real cg_zero() { gqd_real z; cg_scalar *p = (cg_scalar *)&z; for(int l = 0; l < CG_NLIMB; l++) p[l] = 0; return z; }
__device__ static inline gqd_real cg_neg(gqd_real v) { cg_scalar *p = (cg_scalar *)&v; for(int l = 0; l < CG_NLIMB; l++) p[l] = -p[l]; return v; }
static inline gqd_real cg_host_one() { gqd_real z; cg_scalar *p = (cg_scalar *)&z; for(int l = 0; l < CG_NLIMB; l++) p[l] = 0; p[0] = 1; return z; }

/*--- host<->device element conversion helpers (limb pointer-cast) ---*/
static inline void _cgqd_pack(gqd_real *re, gqd_real *im, cqdfloat z)
{
	double *pre = (double *)re, *pim = (double *)im;
	for(int l = 0; l < CGQD_SIZE; l++) { pre[l] = z.val_re[l]; pim[l] = z.val_im[l]; }
}
static inline cqdfloat _cgqd_unpack(gqd_real re, gqd_real im)
{
	cqdfloat z;
	double *pre = (double *)&re, *pim = (double *)&im;
	for(int l = 0; l < CGQD_SIZE; l++) { z.val_re[l] = pre[l]; z.val_im[l] = pim[l]; }
	return z;
}

/*--- vector ---*/
__host__ CGQDVector init_cgqdvector(long int dim)
{
	CGQDVector r = (cgqdvector *)malloc(sizeof(cgqdvector));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgqdvector\n"); return NULL; }
	r->dim = dim;
	r->re = (gqd_real *)calloc(dim, sizeof(gqd_real));
	r->im = (gqd_real *)calloc(dim, sizeof(gqd_real));
	return r;
}
__host__ void free_cgqdvector(CGQDVector v) { free(v->re); free(v->im); free(v); }
__host__ CGQDVector init_cgqdvector_dev(long int dim)
{
	CGQDVector r = (cgqdvector *)malloc(sizeof(cgqdvector));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgqdvector_dev\n"); return NULL; }
	r->dim = dim; r->re = NULL; r->im = NULL;
	cudaMalloc((void **)&(r->re), (size_t)(dim * sizeof(gqd_real)));
	cudaMalloc((void **)&(r->im), (size_t)(dim * sizeof(gqd_real)));
	cudaMemset((void *)(r->re), 0, (size_t)(dim * sizeof(gqd_real)));
	cudaMemset((void *)(r->im), 0, (size_t)(dim * sizeof(gqd_real)));
	return r;
}
__host__ void free_cgqdvector_dev(CGQDVector v) { cudaFree(v->re); cudaFree(v->im); free(v); }

__host__ void subst_cgqdvector_dev_cqdvec(CGQDVector dev, CQDVector cpu)
{
	long int i, dim = dev->dim;
	CGQDVector h = init_cgqdvector(dim);
	for(i = 0; i < dim; i++)
		_cgqd_pack(&h->re[i], &h->im[i], get_cqdvector_i_cqdfloat(cpu, i));
	cudaMemcpy((void *)dev->re, (void *)h->re, (size_t)(dim * sizeof(gqd_real)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dev->im, (void *)h->im, (size_t)(dim * sizeof(gqd_real)), cudaMemcpyHostToDevice);
	free_cgqdvector(h);
}
__host__ void subst_cqdvector_cgqdvec_dev(CQDVector cpu, CGQDVector dev)
{
	long int i, dim = dev->dim;
	CGQDVector h = init_cgqdvector(dim);
	cudaMemcpy((void *)h->re, (void *)dev->re, (size_t)(dim * sizeof(gqd_real)), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)dev->im, (size_t)(dim * sizeof(gqd_real)), cudaMemcpyDeviceToHost);
	for(i = 0; i < dim; i++)
	{
		cqdfloat z = _cgqd_unpack(h->re[i], h->im[i]);
		set_cqdvector_i(cpu, i, &z);
	}
	free_cgqdvector(h);
}

__global__ void _bncu_add_cgqdvector(gqd_real *cre, gqd_real *cim, gqd_real *are, gqd_real *aim, gqd_real *bre, gqd_real *bim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx] + bre[idx]; cim[idx] = aim[idx] + bim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void add_cgqdvector_dev(CGQDVector c, CGQDVector a, CGQDVector b, int nbg, int ntb)
{ _bncu_add_cgqdvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->dim); }

__global__ void _bncu_sub_cgqdvector(gqd_real *cre, gqd_real *cim, gqd_real *are, gqd_real *aim, gqd_real *bre, gqd_real *bim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx] - bre[idx]; cim[idx] = aim[idx] - bim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void sub_cgqdvector_dev(CGQDVector c, CGQDVector a, CGQDVector b, int nbg, int ntb)
{ _bncu_sub_cgqdvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->dim); }

__global__ void _bncu_cmul_cgqdvector(gqd_real *cre, gqd_real *cim, gqd_real vre, gqd_real vim, gqd_real *are, gqd_real *aim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim)
	{
		gqd_real ar = are[idx], ai = aim[idx];
		cre[idx] = vre * ar - vim * ai;
		cim[idx] = vre * ai + vim * ar;
		idx += blockDim.x * gridDim.x;
	}
}
__host__ void cmul_cgqdvector_dev(CGQDVector c, gqd_real vre, gqd_real vim, CGQDVector a, int nbg, int ntb)
{ _bncu_cmul_cgqdvector<<<nbg, ntb>>>(c->re, c->im, vre, vim, a->re, a->im, c->dim); }

__global__ void _bncu_subst_cgqdvector(gqd_real *cre, gqd_real *cim, gqd_real *are, gqd_real *aim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx]; cim[idx] = aim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void subst_cgqdvector_dev(CGQDVector ret, CGQDVector v, int nbg, int ntb)
{ _bncu_subst_cgqdvector<<<nbg, ntb>>>(ret->re, ret->im, v->re, v->im, ret->dim); }

__global__ void _bncu_set0_cgqdvector(gqd_real *cre, gqd_real *cim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = cg_zero(); cim[idx] = cg_zero(); idx += blockDim.x * gridDim.x; }
}
__host__ void set0_cgqdvector_dev(CGQDVector ret, int nbg, int ntb)
{ _bncu_set0_cgqdvector<<<nbg, ntb>>>(ret->re, ret->im, ret->dim); }

/*--- matrix ---*/
__host__ CGQDMatrix init_cgqdmatrix(long int row_dim, long int col_dim)
{
	CGQDMatrix r = (cgqdmatrix *)malloc(sizeof(cgqdmatrix));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgqdmatrix\n"); return NULL; }
	r->row_dim = row_dim; r->col_dim = col_dim;
	r->re = (gqd_real *)calloc(row_dim * col_dim, sizeof(gqd_real));
	r->im = (gqd_real *)calloc(row_dim * col_dim, sizeof(gqd_real));
	return r;
}
__host__ void free_cgqdmatrix(CGQDMatrix m) { free(m->re); free(m->im); free(m); }
__host__ CGQDMatrix init_cgqdmatrix_dev(long int row_dim, long int col_dim)
{
	CGQDMatrix r = (cgqdmatrix *)malloc(sizeof(cgqdmatrix));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgqdmatrix_dev\n"); return NULL; }
	r->row_dim = row_dim; r->col_dim = col_dim; r->re = NULL; r->im = NULL;
	cudaMalloc((void **)&(r->re), (size_t)(row_dim * col_dim * sizeof(gqd_real)));
	cudaMalloc((void **)&(r->im), (size_t)(row_dim * col_dim * sizeof(gqd_real)));
	cudaMemset((void *)(r->re), 0, (size_t)(row_dim * col_dim * sizeof(gqd_real)));
	cudaMemset((void *)(r->im), 0, (size_t)(row_dim * col_dim * sizeof(gqd_real)));
	return r;
}
__host__ void free_cgqdmatrix_dev(CGQDMatrix m) { cudaFree(m->re); cudaFree(m->im); free(m); }

__host__ void subst_cgqdmatrix_dev_cqdmat(CGQDMatrix dev, CQDMatrix cpu)
{
	long int i, j, rd = dev->row_dim, cd = dev->col_dim;
	CGQDMatrix h = init_cgqdmatrix(rd, cd);
	for(i = 0; i < rd; i++)
		for(j = 0; j < cd; j++)
			_cgqd_pack(&h->re[i * cd + j], &h->im[i * cd + j], get_cqdmatrix_ij_cqdfloat(cpu, i, j));
	cudaMemcpy((void *)dev->re, (void *)h->re, (size_t)(sizeof(gqd_real) * rd * cd), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dev->im, (void *)h->im, (size_t)(sizeof(gqd_real) * rd * cd), cudaMemcpyHostToDevice);
	free_cgqdmatrix(h);
}
__host__ void subst_cqdmatrix_cgqdmat_dev(CQDMatrix cpu, CGQDMatrix dev)
{
	long int i, j, rd = dev->row_dim, cd = dev->col_dim;
	CGQDMatrix h = init_cgqdmatrix(rd, cd);
	cudaMemcpy((void *)h->re, (void *)dev->re, (size_t)(sizeof(gqd_real) * rd * cd), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)dev->im, (size_t)(sizeof(gqd_real) * rd * cd), cudaMemcpyDeviceToHost);
	for(i = 0; i < rd; i++)
		for(j = 0; j < cd; j++)
		{
			cqdfloat z = _cgqd_unpack(h->re[i * cd + j], h->im[i * cd + j]);
			set_cqdmatrix_ij(cpu, i, j, &z);
		}
	free_cgqdmatrix(h);
}

// complex matmul ret := A * B  (gqd_real arithmetic)
__global__ void _bncu_mul_cgqdmatrix(gqd_real *cre, gqd_real *cim, long int row_dim, long int col_dim, long int mid_dim,
                                     gqd_real *are, gqd_real *aim, gqd_real *bre, gqd_real *bim)
{
	long int i, j, k;
	i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < row_dim)
	{
		for(j = 0; j < col_dim; j++)
		{
			gqd_real sre = cg_zero(), sim = cg_zero();
			for(k = 0; k < mid_dim; k++)
			{
				gqd_real ar = are[i * mid_dim + k], ai = aim[i * mid_dim + k];
				gqd_real br = bre[k * col_dim + j], bi = bim[k * col_dim + j];
				sre = sre + (ar * br - ai * bi);
				sim = sim + (ar * bi + ai * br);
			}
			cre[i * col_dim + j] = sre;
			cim[i * col_dim + j] = sim;
		}
		i += blockDim.x * gridDim.x;
	}
}
__host__ void mul_cgqdmatrix_dev(CGQDMatrix ret, CGQDMatrix a, CGQDMatrix b, int nbg, int ntb)
{
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{ fprintf(stderr, "ERROR: mul_cgqdmatrix_dev\n"); return; }
	_bncu_mul_cgqdmatrix<<<nbg, ntb>>>(ret->re, ret->im, ret->row_dim, ret->col_dim, a->col_dim, a->re, a->im, b->re, b->im);
}

__host__ void add_cgqdmatrix_dev(CGQDMatrix c, CGQDMatrix a, CGQDMatrix b, int nbg, int ntb)
{ _bncu_add_cgqdvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->row_dim * c->col_dim); }
__host__ void sub_cgqdmatrix_dev(CGQDMatrix c, CGQDMatrix a, CGQDMatrix b, int nbg, int ntb)
{ _bncu_sub_cgqdvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->row_dim * c->col_dim); }
__host__ void cmul_cgqdmatrix_dev(CGQDMatrix c, gqd_real sre, gqd_real sim, CGQDMatrix a, int nbg, int ntb)
{ _bncu_cmul_cgqdvector<<<nbg, ntb>>>(c->re, c->im, sre, sim, a->re, a->im, c->row_dim * c->col_dim); }

__global__ void _bncu_transpose_cgqdmatrix(gqd_real *cre, gqd_real *cim, gqd_real *are, gqd_real *aim, long int row_dim, long int col_dim, int conj)
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
__host__ void transpose_cgqdmatrix_dev(CGQDMatrix c, CGQDMatrix a, int nbg, int ntb)
{ _bncu_transpose_cgqdmatrix<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim, c->col_dim, 0); }
__host__ void conjtrans_cgqdmatrix_dev(CGQDMatrix c, CGQDMatrix a, int nbg, int ntb)
{ _bncu_transpose_cgqdmatrix<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim, c->col_dim, 1); }
__host__ void subst_cgqdmatrix_dev(CGQDMatrix c, CGQDMatrix a, int nbg, int ntb)
{ _bncu_subst_cgqdvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim * c->col_dim); }
__host__ void set0_cgqdmatrix_dev(CGQDMatrix c, int nbg, int ntb)
{ _bncu_set0_cgqdvector<<<nbg, ntb>>>(c->re, c->im, c->row_dim * c->col_dim); }

__host__ void setI_cgqdmatrix_dev(CGQDMatrix c, int nbg, int ntb)
{
	long int i;
	_bncu_set0_cgqdvector<<<nbg, ntb>>>(c->re, c->im, c->row_dim * c->col_dim);
	gqd_real one = cg_host_one();
	for(i = 0; i < c->row_dim; i++)
		cudaMemcpy((void *)&(c->re[i * c->col_dim + i]), (void *)&one, sizeof(gqd_real), cudaMemcpyHostToDevice);
}

// v := a * vb
__global__ void _bncu_mul_cgqdmatrix_cgqdvec(gqd_real *vre, gqd_real *vim, gqd_real *are, gqd_real *aim, gqd_real *bre, gqd_real *bim, long int row_dim, long int col_dim)
{
	long int j, index;
	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < row_dim)
	{
		gqd_real sre = cg_zero(), sim = cg_zero();
		for(j = 0; j < col_dim; j++)
		{
			gqd_real ar = are[index * col_dim + j], ai = aim[index * col_dim + j];
			gqd_real br = bre[j], bi = bim[j];
			sre = sre + (ar * br - ai * bi);
			sim = sim + (ar * bi + ai * br);
		}
		vre[index] = sre; vim[index] = sim;
		index += blockDim.x * gridDim.x;
	}
}
__host__ void mul_cgqdmatrix_cgqdvec(CGQDVector v, CGQDMatrix a, CGQDVector vb, int nbg, int ntb)
{
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim)) { fprintf(stderr, "ERROR: mul_cgqdmatrix_cgqdvec\n"); return; }
	_bncu_mul_cgqdmatrix_cgqdvec<<<nbg, ntb>>>(v->re, v->im, a->re, a->im, vb->re, vb->im, a->row_dim, a->col_dim);
}

// v := a^T * vb (non-conjugate)
__global__ void _bncu_mul_cgqdmatrixt_cgqdvec(gqd_real *vre, gqd_real *vim, gqd_real *are, gqd_real *aim, gqd_real *bre, gqd_real *bim, long int row_dim, long int col_dim)
{
	long int j, index;
	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < col_dim)
	{
		gqd_real sre = cg_zero(), sim = cg_zero();
		for(j = 0; j < row_dim; j++)
		{
			gqd_real ar = are[j * col_dim + index], ai = aim[j * col_dim + index];
			gqd_real br = bre[j], bi = bim[j];
			sre = sre + (ar * br - ai * bi);
			sim = sim + (ar * bi + ai * br);
		}
		vre[index] = sre; vim[index] = sim;
		index += blockDim.x * gridDim.x;
	}
}
__host__ void mul_cgqdmatrixt_cgqdvec(CGQDVector v, CGQDMatrix a, CGQDVector vb, int nbg, int ntb)
{
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim)) { fprintf(stderr, "ERROR: mul_cgqdmatrixt_cgqdvec\n"); return; }
	_bncu_mul_cgqdmatrixt_cgqdvec<<<nbg, ntb>>>(v->re, v->im, a->re, a->im, vb->re, vb->im, a->row_dim, a->col_dim);
}
