/********************************************************************************/
/* cgdslinear.cu: Complex float-float GPU Linear Computation (CUDA)            */
/*   complex as separate real/imag gds_real arrays (SoA); device dd math.        */
/* Copyright (C) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#include "cgdslinear.h"
#include "gqs.cu"   // device gds_real arithmetic implementation

// per-type zero/one/neg via limb pointer-cast (component count = CG_NLIMB).
// Generic over double2/3/4 (and float2/3/4 after sed float->float).
#define CG_NLIMB CGDS_SIZE
typedef float cg_scalar;
__device__ static inline gds_real cg_zero() { gds_real z; cg_scalar *p = (cg_scalar *)&z; for(int l = 0; l < CG_NLIMB; l++) p[l] = 0; return z; }
__device__ static inline gds_real cg_neg(gds_real v) { cg_scalar *p = (cg_scalar *)&v; for(int l = 0; l < CG_NLIMB; l++) p[l] = -p[l]; return v; }
static inline gds_real cg_host_one() { gds_real z; cg_scalar *p = (cg_scalar *)&z; for(int l = 0; l < CG_NLIMB; l++) p[l] = 0; p[0] = 1; return z; }

/*--- host<->device element conversion helpers (limb pointer-cast) ---*/
static inline void _cgds_pack(gds_real *re, gds_real *im, cdsfloat z)
{
	float *pre = (float *)re, *pim = (float *)im;
	for(int l = 0; l < CGDS_SIZE; l++) { pre[l] = z.val_re[l]; pim[l] = z.val_im[l]; }
}
static inline cdsfloat _cgds_unpack(gds_real re, gds_real im)
{
	cdsfloat z;
	float *pre = (float *)&re, *pim = (float *)&im;
	for(int l = 0; l < CGDS_SIZE; l++) { z.val_re[l] = pre[l]; z.val_im[l] = pim[l]; }
	return z;
}

/*--- vector ---*/
__host__ CGDSVector init_cgdsvector(long int dim)
{
	CGDSVector r = (cgdsvector *)malloc(sizeof(cgdsvector));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgdsvector\n"); return NULL; }
	r->dim = dim;
	r->re = (gds_real *)calloc(dim, sizeof(gds_real));
	r->im = (gds_real *)calloc(dim, sizeof(gds_real));
	return r;
}
__host__ void free_cgdsvector(CGDSVector v) { free(v->re); free(v->im); free(v); }
__host__ CGDSVector init_cgdsvector_dev(long int dim)
{
	CGDSVector r = (cgdsvector *)malloc(sizeof(cgdsvector));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgdsvector_dev\n"); return NULL; }
	r->dim = dim; r->re = NULL; r->im = NULL;
	cudaMalloc((void **)&(r->re), (size_t)(dim * sizeof(gds_real)));
	cudaMalloc((void **)&(r->im), (size_t)(dim * sizeof(gds_real)));
	cudaMemset((void *)(r->re), 0, (size_t)(dim * sizeof(gds_real)));
	cudaMemset((void *)(r->im), 0, (size_t)(dim * sizeof(gds_real)));
	return r;
}
__host__ void free_cgdsvector_dev(CGDSVector v) { cudaFree(v->re); cudaFree(v->im); free(v); }

__host__ void subst_cgdsvector_dev_cdsvec(CGDSVector dev, CDSVector cpu)
{
	long int i, dim = dev->dim;
	CGDSVector h = init_cgdsvector(dim);
	for(i = 0; i < dim; i++)
		_cgds_pack(&h->re[i], &h->im[i], get_cdsvector_i_cdsfloat(cpu, i));
	cudaMemcpy((void *)dev->re, (void *)h->re, (size_t)(dim * sizeof(gds_real)), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dev->im, (void *)h->im, (size_t)(dim * sizeof(gds_real)), cudaMemcpyHostToDevice);
	free_cgdsvector(h);
}
__host__ void subst_cdsvector_cgdsvec_dev(CDSVector cpu, CGDSVector dev)
{
	long int i, dim = dev->dim;
	CGDSVector h = init_cgdsvector(dim);
	cudaMemcpy((void *)h->re, (void *)dev->re, (size_t)(dim * sizeof(gds_real)), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)dev->im, (size_t)(dim * sizeof(gds_real)), cudaMemcpyDeviceToHost);
	for(i = 0; i < dim; i++)
	{
		cdsfloat z = _cgds_unpack(h->re[i], h->im[i]);
		set_cdsvector_i(cpu, i, &z);
	}
	free_cgdsvector(h);
}

__global__ void _bncu_add_cgdsvector(gds_real *cre, gds_real *cim, gds_real *are, gds_real *aim, gds_real *bre, gds_real *bim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx] + bre[idx]; cim[idx] = aim[idx] + bim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void add_cgdsvector_dev(CGDSVector c, CGDSVector a, CGDSVector b, int nbg, int ntb)
{ _bncu_add_cgdsvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->dim); }

__global__ void _bncu_sub_cgdsvector(gds_real *cre, gds_real *cim, gds_real *are, gds_real *aim, gds_real *bre, gds_real *bim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx] - bre[idx]; cim[idx] = aim[idx] - bim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void sub_cgdsvector_dev(CGDSVector c, CGDSVector a, CGDSVector b, int nbg, int ntb)
{ _bncu_sub_cgdsvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->dim); }

__global__ void _bncu_cmul_cgdsvector(gds_real *cre, gds_real *cim, gds_real vre, gds_real vim, gds_real *are, gds_real *aim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim)
	{
		gds_real ar = are[idx], ai = aim[idx];
		cre[idx] = vre * ar - vim * ai;
		cim[idx] = vre * ai + vim * ar;
		idx += blockDim.x * gridDim.x;
	}
}
__host__ void cmul_cgdsvector_dev(CGDSVector c, gds_real vre, gds_real vim, CGDSVector a, int nbg, int ntb)
{ _bncu_cmul_cgdsvector<<<nbg, ntb>>>(c->re, c->im, vre, vim, a->re, a->im, c->dim); }

__global__ void _bncu_subst_cgdsvector(gds_real *cre, gds_real *cim, gds_real *are, gds_real *aim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = are[idx]; cim[idx] = aim[idx]; idx += blockDim.x * gridDim.x; }
}
__host__ void subst_cgdsvector_dev(CGDSVector ret, CGDSVector v, int nbg, int ntb)
{ _bncu_subst_cgdsvector<<<nbg, ntb>>>(ret->re, ret->im, v->re, v->im, ret->dim); }

__global__ void _bncu_set0_cgdsvector(gds_real *cre, gds_real *cim, long int dim)
{
	long int idx = threadIdx.x + blockIdx.x * blockDim.x;
	while(idx < dim) { cre[idx] = cg_zero(); cim[idx] = cg_zero(); idx += blockDim.x * gridDim.x; }
}
__host__ void set0_cgdsvector_dev(CGDSVector ret, int nbg, int ntb)
{ _bncu_set0_cgdsvector<<<nbg, ntb>>>(ret->re, ret->im, ret->dim); }

/*--- matrix ---*/
__host__ CGDSMatrix init_cgdsmatrix(long int row_dim, long int col_dim)
{
	CGDSMatrix r = (cgdsmatrix *)malloc(sizeof(cgdsmatrix));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgdsmatrix\n"); return NULL; }
	r->row_dim = row_dim; r->col_dim = col_dim;
	r->re = (gds_real *)calloc(row_dim * col_dim, sizeof(gds_real));
	r->im = (gds_real *)calloc(row_dim * col_dim, sizeof(gds_real));
	return r;
}
__host__ void free_cgdsmatrix(CGDSMatrix m) { free(m->re); free(m->im); free(m); }
__host__ CGDSMatrix init_cgdsmatrix_dev(long int row_dim, long int col_dim)
{
	CGDSMatrix r = (cgdsmatrix *)malloc(sizeof(cgdsmatrix));
	if(r == NULL) { fprintf(stderr, "ERROR: init_cgdsmatrix_dev\n"); return NULL; }
	r->row_dim = row_dim; r->col_dim = col_dim; r->re = NULL; r->im = NULL;
	cudaMalloc((void **)&(r->re), (size_t)(row_dim * col_dim * sizeof(gds_real)));
	cudaMalloc((void **)&(r->im), (size_t)(row_dim * col_dim * sizeof(gds_real)));
	cudaMemset((void *)(r->re), 0, (size_t)(row_dim * col_dim * sizeof(gds_real)));
	cudaMemset((void *)(r->im), 0, (size_t)(row_dim * col_dim * sizeof(gds_real)));
	return r;
}
__host__ void free_cgdsmatrix_dev(CGDSMatrix m) { cudaFree(m->re); cudaFree(m->im); free(m); }

__host__ void subst_cgdsmatrix_dev_cdsmat(CGDSMatrix dev, CDSMatrix cpu)
{
	long int i, j, rd = dev->row_dim, cd = dev->col_dim;
	CGDSMatrix h = init_cgdsmatrix(rd, cd);
	for(i = 0; i < rd; i++)
		for(j = 0; j < cd; j++)
			_cgds_pack(&h->re[i * cd + j], &h->im[i * cd + j], get_cdsmatrix_ij_cdsfloat(cpu, i, j));
	cudaMemcpy((void *)dev->re, (void *)h->re, (size_t)(sizeof(gds_real) * rd * cd), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dev->im, (void *)h->im, (size_t)(sizeof(gds_real) * rd * cd), cudaMemcpyHostToDevice);
	free_cgdsmatrix(h);
}
__host__ void subst_cdsmatrix_cgdsmat_dev(CDSMatrix cpu, CGDSMatrix dev)
{
	long int i, j, rd = dev->row_dim, cd = dev->col_dim;
	CGDSMatrix h = init_cgdsmatrix(rd, cd);
	cudaMemcpy((void *)h->re, (void *)dev->re, (size_t)(sizeof(gds_real) * rd * cd), cudaMemcpyDeviceToHost);
	cudaMemcpy((void *)h->im, (void *)dev->im, (size_t)(sizeof(gds_real) * rd * cd), cudaMemcpyDeviceToHost);
	for(i = 0; i < rd; i++)
		for(j = 0; j < cd; j++)
		{
			cdsfloat z = _cgds_unpack(h->re[i * cd + j], h->im[i * cd + j]);
			set_cdsmatrix_ij(cpu, i, j, &z);
		}
	free_cgdsmatrix(h);
}

// complex matmul ret := A * B  (gds_real arithmetic)
__global__ void _bncu_mul_cgdsmatrix(gds_real *cre, gds_real *cim, long int row_dim, long int col_dim, long int mid_dim,
                                     gds_real *are, gds_real *aim, gds_real *bre, gds_real *bim)
{
	long int i, j, k;
	i = threadIdx.x + blockIdx.x * blockDim.x;
	while(i < row_dim)
	{
		for(j = 0; j < col_dim; j++)
		{
			gds_real sre = cg_zero(), sim = cg_zero();
			for(k = 0; k < mid_dim; k++)
			{
				gds_real ar = are[i * mid_dim + k], ai = aim[i * mid_dim + k];
				gds_real br = bre[k * col_dim + j], bi = bim[k * col_dim + j];
				sre = sre + (ar * br - ai * bi);
				sim = sim + (ar * bi + ai * br);
			}
			cre[i * col_dim + j] = sre;
			cim[i * col_dim + j] = sim;
		}
		i += blockDim.x * gridDim.x;
	}
}
__host__ void mul_cgdsmatrix_dev(CGDSMatrix ret, CGDSMatrix a, CGDSMatrix b, int nbg, int ntb)
{
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{ fprintf(stderr, "ERROR: mul_cgdsmatrix_dev\n"); return; }
	_bncu_mul_cgdsmatrix<<<nbg, ntb>>>(ret->re, ret->im, ret->row_dim, ret->col_dim, a->col_dim, a->re, a->im, b->re, b->im);
}

__host__ void add_cgdsmatrix_dev(CGDSMatrix c, CGDSMatrix a, CGDSMatrix b, int nbg, int ntb)
{ _bncu_add_cgdsvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->row_dim * c->col_dim); }
__host__ void sub_cgdsmatrix_dev(CGDSMatrix c, CGDSMatrix a, CGDSMatrix b, int nbg, int ntb)
{ _bncu_sub_cgdsvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, b->re, b->im, c->row_dim * c->col_dim); }
__host__ void cmul_cgdsmatrix_dev(CGDSMatrix c, gds_real sre, gds_real sim, CGDSMatrix a, int nbg, int ntb)
{ _bncu_cmul_cgdsvector<<<nbg, ntb>>>(c->re, c->im, sre, sim, a->re, a->im, c->row_dim * c->col_dim); }

__global__ void _bncu_transpose_cgdsmatrix(gds_real *cre, gds_real *cim, gds_real *are, gds_real *aim, long int row_dim, long int col_dim, int conj)
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
__host__ void transpose_cgdsmatrix_dev(CGDSMatrix c, CGDSMatrix a, int nbg, int ntb)
{ _bncu_transpose_cgdsmatrix<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim, c->col_dim, 0); }
__host__ void conjtrans_cgdsmatrix_dev(CGDSMatrix c, CGDSMatrix a, int nbg, int ntb)
{ _bncu_transpose_cgdsmatrix<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim, c->col_dim, 1); }
__host__ void subst_cgdsmatrix_dev(CGDSMatrix c, CGDSMatrix a, int nbg, int ntb)
{ _bncu_subst_cgdsvector<<<nbg, ntb>>>(c->re, c->im, a->re, a->im, c->row_dim * c->col_dim); }
__host__ void set0_cgdsmatrix_dev(CGDSMatrix c, int nbg, int ntb)
{ _bncu_set0_cgdsvector<<<nbg, ntb>>>(c->re, c->im, c->row_dim * c->col_dim); }

__host__ void setI_cgdsmatrix_dev(CGDSMatrix c, int nbg, int ntb)
{
	long int i;
	_bncu_set0_cgdsvector<<<nbg, ntb>>>(c->re, c->im, c->row_dim * c->col_dim);
	gds_real one = cg_host_one();
	for(i = 0; i < c->row_dim; i++)
		cudaMemcpy((void *)&(c->re[i * c->col_dim + i]), (void *)&one, sizeof(gds_real), cudaMemcpyHostToDevice);
}

// v := a * vb
__global__ void _bncu_mul_cgdsmatrix_cgdsvec(gds_real *vre, gds_real *vim, gds_real *are, gds_real *aim, gds_real *bre, gds_real *bim, long int row_dim, long int col_dim)
{
	long int j, index;
	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < row_dim)
	{
		gds_real sre = cg_zero(), sim = cg_zero();
		for(j = 0; j < col_dim; j++)
		{
			gds_real ar = are[index * col_dim + j], ai = aim[index * col_dim + j];
			gds_real br = bre[j], bi = bim[j];
			sre = sre + (ar * br - ai * bi);
			sim = sim + (ar * bi + ai * br);
		}
		vre[index] = sre; vim[index] = sim;
		index += blockDim.x * gridDim.x;
	}
}
__host__ void mul_cgdsmatrix_cgdsvec(CGDSVector v, CGDSMatrix a, CGDSVector vb, int nbg, int ntb)
{
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim)) { fprintf(stderr, "ERROR: mul_cgdsmatrix_cgdsvec\n"); return; }
	_bncu_mul_cgdsmatrix_cgdsvec<<<nbg, ntb>>>(v->re, v->im, a->re, a->im, vb->re, vb->im, a->row_dim, a->col_dim);
}

// v := a^T * vb (non-conjugate)
__global__ void _bncu_mul_cgdsmatrixt_cgdsvec(gds_real *vre, gds_real *vim, gds_real *are, gds_real *aim, gds_real *bre, gds_real *bim, long int row_dim, long int col_dim)
{
	long int j, index;
	index = threadIdx.x + blockIdx.x * blockDim.x;
	while(index < col_dim)
	{
		gds_real sre = cg_zero(), sim = cg_zero();
		for(j = 0; j < row_dim; j++)
		{
			gds_real ar = are[j * col_dim + index], ai = aim[j * col_dim + index];
			gds_real br = bre[j], bi = bim[j];
			sre = sre + (ar * br - ai * bi);
			sim = sim + (ar * bi + ai * br);
		}
		vre[index] = sre; vim[index] = sim;
		index += blockDim.x * gridDim.x;
	}
}
__host__ void mul_cgdsmatrixt_cgdsvec(CGDSVector v, CGDSMatrix a, CGDSVector vb, int nbg, int ntb)
{
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim)) { fprintf(stderr, "ERROR: mul_cgdsmatrixt_cgdsvec\n"); return; }
	_bncu_mul_cgdsmatrixt_cgdsvec<<<nbg, ntb>>>(v->re, v->im, a->re, a->im, vb->re, vb->im, a->row_dim, a->col_dim);
}
