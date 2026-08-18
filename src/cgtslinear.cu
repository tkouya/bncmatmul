/********************************************************************************/
/* cgtslinear.cu: Complex float-float GPU Linear Computation (CUDA)            */
/*   complex as separate real/imag gts_real arrays (SoA); device dd math.        */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#include "cgtslinear.h"
#include "gqs.cu"   // device gts_real arithmetic implementation

// per-type zero/one/neg via limb pointer-cast (component count = CG_NLIMB).
// Generic over double2/3/4 (and float2/3/4 after sed float->float).
#define CG_NLIMB CGTS_SIZE
typedef float cg_scalar;
__device__ static inline gts_real cg_zero() { gts_real z; cg_scalar *p = (cg_scalar *)&z; for(int l = 0; l < CG_NLIMB; l++) p[l] = 0; return z; }
__device__ static inline gts_real cg_neg(gts_real v) { cg_scalar *p = (cg_scalar *)&v; for(int l = 0; l < CG_NLIMB; l++) p[l] = -p[l]; return v; }
static inline gts_real cg_host_one() { gts_real z; cg_scalar *p = (cg_scalar *)&z; for(int l = 0; l < CG_NLIMB; l++) p[l] = 0; p[0] = 1; return z; }

/*--- host<->device element conversion helpers (limb pointer-cast) ---*/
static inline void _cgts_pack(gts_real *re, gts_real *im, ctsfloat z)
{
	float *pre = (float *)re, *pim = (float *)im;
	for(int l = 0; l < CGTS_SIZE; l++) { pre[l] = z.val_re[l]; pim[l] = z.val_im[l]; }
}
static inline ctsfloat _cgts_unpack(gts_real re, gts_real im)
{
	ctsfloat z;
	float *pre = (float *)&re, *pim = (float *)&im;
	for(int l = 0; l < CGTS_SIZE; l++) { z.val_re[l] = pre[l]; z.val_im[l] = pim[l]; }
	return z;
}

/*--- vector ---*/
__host__ CGTSVector init_cgtsvector(long int dim)
{
	CGTSVector r = (cgtsvector *)malloc(sizeof(cgtsvector));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgtsvector\n"); return NULL; }
	r->dim = dim;
	r->re = (gts_real *)calloc(dim, sizeof(gts_real));
	r->im = (gts_real *)calloc(dim, sizeof(gts_real));
	return r;
}
__host__ void free_cgtsvector(CGTSVector v) { free(v->re); free(v->im); free(v); }
__host__ CGTSVector init_cgtsvector_dev(long int dim)
{
	CGTSVector r = (cgtsvector *)malloc(sizeof(cgtsvector));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgtsvector_dev\n"); return NULL; }
	r->dim = dim; r->re = NULL; r->im = NULL;
	cudaMalloc((void **)&(r->re), (size_t)(dim * sizeof(gts_real)));
	cudaMalloc((void **)&(r->im), (size_t)(dim * sizeof(gts_real)));
	cudaMemset((void *)(r->re), 0, (size_t)(dim * sizeof(gts_real)));
	cudaMemset((void *)(r->im), 0, (size_t)(dim * sizeof(gts_real)));
	return r;
}
__host__ void free_cgtsvector_dev(CGTSVector v) { cudaFree(v->re); cudaFree(v->im); free(v); }

__host__ void subst_cgtsvector_dev_ctsvec(CGTSVector dev, CTSVector cpu)
{
	long int i, dim = dev->dim;
	CGTSVector h = init_cgtsvector(dim);
	for(i = 0; i < dim; i++)
		_cgts_pack(&h->re[i], &h->im[i], get_ctsvector_i_ctsfloat(cpu, i));
	cudaMemcpy((void *)dev->re, (void *)h->re, (size_t)(dim * sizeof(gts_real)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dev->im, (void *)h->im, (size_t)(dim * sizeof(gts_real)), cudaMemcpyHostToDevice);
	free_cgtsvector(h);
}
__host__ void subst_ctsvector_cgtsvec_dev(CTSVector cpu, CGTSVector dev)
{
	long int i, dim = dev->dim;
	CGTSVector h = init_cgtsvector(dim);
	cudaMemcpy((void *)h->re, (void *)dev->re, (size_t)(dim * sizeof(gts_real)), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)dev->im, (size_t)(dim * sizeof(gts_real)), cudaMemcpyDeviceToHost);
	for(i = 0; i < dim; i++)
	{
		ctsfloat z = _cgts_unpack(h->re[i], h->im[i]);
		set_ctsvector_i(cpu, i, &z);
	}
	free_cgtsvector(h);
}

__global__ void _bncu_add_cgtsvector(gts_real *cre, gts_real *cim, gts_real *are, gts_real *aim, gts_real *bre, gts_real *bim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx] + bre[idx]; cim[idx] = aim[idx] + bim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void add_cgtsvector_dev(CGTSVector c, CGTSVector a, CGTSVector b, int nbg, int ntb)
{ _bncu_add_cgtsvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->dim); }

__global__ void _bncu_sub_cgtsvector(gts_real *cre, gts_real *cim, gts_real *are, gts_real *aim, gts_real *bre, gts_real *bim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx] - bre[idx]; cim[idx] = aim[idx] - bim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void sub_cgtsvector_dev(CGTSVector c, CGTSVector a, CGTSVector b, int nbg, int ntb)
{ _bncu_sub_cgtsvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->dim); }

__global__ void _bncu_cmul_cgtsvector(gts_real *cre, gts_real *cim, gts_real vre, gts_real vim, gts_real *are, gts_real *aim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim)
	{
		gts_real ar = are[idx], ai = aim[idx];
		cre[idx] = vre * ar - vim * ai;
		cim[idx] = vre * ai + vim * ar;
		idx += blockDim.x * gridDim.x;
	}
}
__host__ void cmul_cgtsvector_dev(CGTSVector c, gts_real vre, gts_real vim, CGTSVector a, int nbg, int ntb)
{ _bncu_cmul_cgtsvector<<<nbg, ntb>>>(c->re, c->im, vre, vim, a->re, a->im, c->dim); }

__global__ void _bncu_subst_cgtsvector(gts_real *cre, gts_real *cim, gts_real *are, gts_real *aim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx]; cim[idx] = aim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void subst_cgtsvector_dev(CGTSVector ret, CGTSVector v, int nbg, int ntb)
{ _bncu_subst_cgtsvector<<<nbg, ntb>>>(ret->re, ret->im, v->re, v->im, ret->dim); }

__global__ void _bncu_set0_cgtsvector(gts_real *cre, gts_real *cim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = cg_zero(); cim[idx] = cg_zero(); idx += blockDim.x * gridDim.x; }
}
__host__ void set0_cgtsvector_dev(CGTSVector ret, int nbg, int ntb)
{ _bncu_set0_cgtsvector<<<nbg, ntb>>>(ret->re, ret->im, ret->dim); }

/*--- matrix ---*/
__host__ CGTSMatrix init_cgtsmatrix(long int row_dim, long int col_dim)
{
	CGTSMatrix r = (cgtsmatrix *)malloc(sizeof(cgtsmatrix));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgtsmatrix\n"); return NULL; }
	r->row_dim = row_dim; r->col_dim = col_dim;
	r->re = (gts_real *)calloc(row_dim * col_dim, sizeof(gts_real));
	r->im = (gts_real *)calloc(row_dim * col_dim, sizeof(gts_real));
	return r;
}
__host__ void free_cgtsmatrix(CGTSMatrix m) { free(m->re); free(m->im); free(m); }
__host__ CGTSMatrix init_cgtsmatrix_dev(long int row_dim, long int col_dim)
{
	CGTSMatrix r = (cgtsmatrix *)malloc(sizeof(cgtsmatrix));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgtsmatrix_dev\n"); return NULL; }
	r->row_dim = row_dim; r->col_dim = col_dim; r->re = NULL; r->im = NULL;
	cudaMalloc((void **)&(r->re), (size_t)(row_dim * col_dim * sizeof(gts_real)));
	cudaMalloc((void **)&(r->im), (size_t)(row_dim * col_dim * sizeof(gts_real)));
	cudaMemset((void *)(r->re), 0, (size_t)(row_dim * col_dim * sizeof(gts_real)));
	cudaMemset((void *)(r->im), 0, (size_t)(row_dim * col_dim * sizeof(gts_real)));
	return r;
}
__host__ void free_cgtsmatrix_dev(CGTSMatrix m) { cudaFree(m->re); cudaFree(m->im); free(m); }

__host__ void subst_cgtsmatrix_dev_ctsmat(CGTSMatrix dev, CTSMatrix cpu)
{
	long int i, j, rd = dev->row_dim, cd = dev->col_dim;
	CGTSMatrix h = init_cgtsmatrix(rd, cd);
	for(i = 0; i < rd; i++)
		for(j = 0; j < cd; j++)
			_cgts_pack(&h->re[i * cd + j], &h->im[i * cd + j], get_ctsmatrix_ij_ctsfloat(cpu, i, j));
	cudaMemcpy((void *)dev->re, (void *)h->re, (size_t)(sizeof(gts_real) * rd * cd), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dev->im, (void *)h->im, (size_t)(sizeof(gts_real) * rd * cd), cudaMemcpyHostToDevice);
	free_cgtsmatrix(h);
}
__host__ void subst_ctsmatrix_cgtsmat_dev(CTSMatrix cpu, CGTSMatrix dev)
{
	long int i, j, rd = dev->row_dim, cd = dev->col_dim;
	CGTSMatrix h = init_cgtsmatrix(rd, cd);
	cudaMemcpy((void *)h->re, (void *)dev->re, (size_t)(sizeof(gts_real) * rd * cd), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)dev->im, (size_t)(sizeof(gts_real) * rd * cd), cudaMemcpyDeviceToHost);
	for(i = 0; i < rd; i++)
		for(j = 0; j < cd; j++)
		{
			ctsfloat z = _cgts_unpack(h->re[i * cd + j], h->im[i * cd + j]);
			set_ctsmatrix_ij(cpu, i, j, &z);
		}
	free_cgtsmatrix(h);
}

// complex matmul ret := A * B  (gts_real arithmetic)
__global__ void _bncu_mul_cgtsmatrix(gts_real *cre, gts_real *cim, long int row_dim, long int col_dim, long int mid_dim,
                                     gts_real *are, gts_real *aim, gts_real *bre, gts_real *bim)
{
	long int i, j, k;
	i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < row_dim)
	{
		for(j = 0; j < col_dim; j++)
		{
			gts_real sre = cg_zero(), sim = cg_zero();
			for(k = 0; k < mid_dim; k++)
			{
				gts_real ar = are[i * mid_dim + k], ai = aim[i * mid_dim + k];
				gts_real br = bre[k * col_dim + j], bi = bim[k * col_dim + j];
				sre = sre + (ar * br - ai * bi);
				sim = sim + (ar * bi + ai * br);
			}
			cre[i * col_dim + j] = sre;
			cim[i * col_dim + j] = sim;
		}
		i += blockDim.x * gridDim.x;
	}
}
__host__ void mul_cgtsmatrix_dev(CGTSMatrix ret, CGTSMatrix a, CGTSMatrix b, int nbg, int ntb)
{
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{ fprintf(stderr, "ERROR: mul_cgtsmatrix_dev\n"); return; }
	_bncu_mul_cgtsmatrix<<<nbg, ntb>>>(ret->re, ret->im, ret->row_dim, ret->col_dim, a->col_dim, a->re, a->im, b->re, b->im);
}

__host__ void add_cgtsmatrix_dev(CGTSMatrix c, CGTSMatrix a, CGTSMatrix b, int nbg, int ntb)
{ _bncu_add_cgtsvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->row_dim * c->col_dim); }
__host__ void sub_cgtsmatrix_dev(CGTSMatrix c, CGTSMatrix a, CGTSMatrix b, int nbg, int ntb)
{ _bncu_sub_cgtsvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->row_dim * c->col_dim); }
__host__ void cmul_cgtsmatrix_dev(CGTSMatrix c, gts_real sre, gts_real sim, CGTSMatrix a, int nbg, int ntb)
{ _bncu_cmul_cgtsvector<<<nbg, ntb>>>(c->re, c->im, sre, sim, a->re, a->im, c->row_dim * c->col_dim); }

__global__ void _bncu_transpose_cgtsmatrix(gts_real *cre, gts_real *cim, gts_real *are, gts_real *aim, long int row_dim, long int col_dim, int conj)
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
__host__ void transpose_cgtsmatrix_dev(CGTSMatrix c, CGTSMatrix a, int nbg, int ntb)
{ _bncu_transpose_cgtsmatrix<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim, c->col_dim, 0); }
__host__ void conjtrans_cgtsmatrix_dev(CGTSMatrix c, CGTSMatrix a, int nbg, int ntb)
{ _bncu_transpose_cgtsmatrix<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim, c->col_dim, 1); }
__host__ void subst_cgtsmatrix_dev(CGTSMatrix c, CGTSMatrix a, int nbg, int ntb)
{ _bncu_subst_cgtsvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim * c->col_dim); }
__host__ void set0_cgtsmatrix_dev(CGTSMatrix c, int nbg, int ntb)
{ _bncu_set0_cgtsvector<<<nbg, ntb>>>(c->re, c->im, c->row_dim * c->col_dim); }

__host__ void setI_cgtsmatrix_dev(CGTSMatrix c, int nbg, int ntb)
{
	long int i;
	_bncu_set0_cgtsvector<<<nbg, ntb>>>(c->re, c->im, c->row_dim * c->col_dim);
	gts_real one = cg_host_one();
	for(i = 0; i < c->row_dim; i++)
		cudaMemcpy((void *)&(c->re[i * c->col_dim + i]), (void *)&one, sizeof(gts_real), cudaMemcpyHostToDevice);
}

// v := a * vb
__global__ void _bncu_mul_cgtsmatrix_cgtsvec(gts_real *vre, gts_real *vim, gts_real *are, gts_real *aim, gts_real *bre, gts_real *bim, long int row_dim, long int col_dim)
{
	long int j, index;
	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < row_dim)
	{
		gts_real sre = cg_zero(), sim = cg_zero();
		for(j = 0; j < col_dim; j++)
		{
			gts_real ar = are[index * col_dim + j], ai = aim[index * col_dim + j];
			gts_real br = bre[j], bi = bim[j];
			sre = sre + (ar * br - ai * bi);
			sim = sim + (ar * bi + ai * br);
		}
		vre[index] = sre; vim[index] = sim;
		index += blockDim.x * gridDim.x;
	}
}
__host__ void mul_cgtsmatrix_cgtsvec(CGTSVector v, CGTSMatrix a, CGTSVector vb, int nbg, int ntb)
{
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim)) { fprintf(stderr, "ERROR: mul_cgtsmatrix_cgtsvec\n"); return; }
	_bncu_mul_cgtsmatrix_cgtsvec<<<nbg, ntb>>>(v->re, v->im, a->re, a->im, vb->re, vb->im, a->row_dim, a->col_dim);
}

// v := a^T * vb (non-conjugate)
__global__ void _bncu_mul_cgtsmatrixt_cgtsvec(gts_real *vre, gts_real *vim, gts_real *are, gts_real *aim, gts_real *bre, gts_real *bim, long int row_dim, long int col_dim)
{
	long int j, index;
	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < col_dim)
	{
		gts_real sre = cg_zero(), sim = cg_zero();
		for(j = 0; j < row_dim; j++)
		{
			gts_real ar = are[j * col_dim + index], ai = aim[j * col_dim + index];
			gts_real br = bre[j], bi = bim[j];
			sre = sre + (ar * br - ai * bi);
			sim = sim + (ar * bi + ai * br);
		}
		vre[index] = sre; vim[index] = sim;
		index += blockDim.x * gridDim.x;
	}
}
__host__ void mul_cgtsmatrixt_cgtsvec(CGTSVector v, CGTSMatrix a, CGTSVector vb, int nbg, int ntb)
{
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim)) { fprintf(stderr, "ERROR: mul_cgtsmatrixt_cgtsvec\n"); return; }
	_bncu_mul_cgtsmatrixt_cgtsvec<<<nbg, ntb>>>(v->re, v->im, a->re, a->im, vb->re, vb->im, a->row_dim, a->col_dim);
}
