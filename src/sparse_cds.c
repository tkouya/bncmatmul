/********************************************************************************/
/* sparse_cds.c : Complex double-single (cds) real-sparse matrix and SpMV.       */
/*                                                                              */
/* A complex CDS sparse matrix is a pair {re, im} of real DSRSMatrix sharing the */
/* same sparsity pattern.  The complex SpMV is computed from four real DS SpMVs  */
/* (mul_dsrsmatrix_dsvec), so it inherits the scalar/AVX2/AVX-512/NEON/SVE2      */
/* vectorisation of the real DS kernels with no extra SIMD code here.            */
/*   y = A x : Are*xre - Aim*xim + i (Are*xim + Aim*xre)                          */
/*                                                                              */
/* Copyright (c) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bncsparse.h"


#ifndef SUCCESS
#define SUCCESS (0)
#endif
#ifndef ERROR
#define ERROR (-1)
#endif

/* initialize CDSRSMatrix (re and im share the same sparsity layout) */
CDSRSMatrix init_cdsrsmatrix(long int row_dim, long int *nzero_col_dim, long int nzero_total_num)
{
	CDSRSMatrix ret = (CDSRSMatrix)malloc(sizeof(cdsrsmatrix));
	if(ret == NULL) { fprintf(stderr, "Cannot allocate CDSRSMatrix\n"); return NULL; }
	ret->re = init_dsrsmatrix(row_dim, nzero_col_dim, nzero_total_num);
	ret->im = init_dsrsmatrix(row_dim, nzero_col_dim, nzero_total_num);
	if(ret->re == NULL || ret->im == NULL) { fprintf(stderr, "Cannot allocate CDSRSMatrix re/im\n"); return NULL; }
	return ret;
}

void free_cdsrsmatrix(CDSRSMatrix mat)
{
	free_dsrsmatrix(mat->re);
	free_dsrsmatrix(mat->im);
	free(mat);
}

void set0_cdsrsmatrix(CDSRSMatrix spmat)
{
	set0_dsrsmatrix(spmat->re);
	set0_dsrsmatrix(spmat->im);
}

/* set the CDSRSMatrix ij-element (index pattern must already be present) */
void set_cdsrsmatrix_ij(CDSRSMatrix mat, long int row_index, long int col_index, cdsfloat *val)
{
	set_dsrsmatrix_ij(mat->re, row_index, col_index, val->val_re);
	set_dsrsmatrix_ij(mat->im, row_index, col_index, val->val_im);
}

/* get the CDSRSMatrix ij-element */
void get_cdsrsmatrix_ij(cdsfloat *ret, CDSRSMatrix mat, long int row_index, long int col_index)
{
	get_dsrsmatrix_ij(ret->val_re, mat->re, row_index, col_index);
	get_dsrsmatrix_ij(ret->val_im, mat->im, row_index, col_index);
}

/* initialize and set CDSRSMatrix from dense CDSMatrix */
CDSRSMatrix init_set_cdsrsmatrix_cdsmatrix(CDSMatrix org_mat)
{
	long int i, j, c;
	long int nzero_total_num, total_index, j_index;
	long int *ptr_nzero_col_dim;
	CDSRSMatrix ret;
	long int rd = org_mat->re->row_dim, cd = org_mat->re->col_dim;

	nzero_total_num = 0;
	ptr_nzero_col_dim = (long int *)malloc((size_t)(sizeof(long int) * rd));
	if(ptr_nzero_col_dim == NULL) { fprintf(stderr, "Cannot allocate ptr_nzero_col_dim\n"); return NULL; }

	/* 1st pass: count nonzeros (|re|+|im| != 0) per row */
	for(i = 0; i < rd; i++)
	{
		ptr_nzero_col_dim[i] = 0;
		for(j = 0; j < cd; j++)
		{
			cdsfloat *e = get_cdsmatrix_ij(org_mat, i, j);
			int nz = 0;
			for(c = 0; c < DSSIZE; c++) if(e->val_re[c] != 0.0f || e->val_im[c] != 0.0f) nz = 1;
			if(nz) { nzero_total_num++; ptr_nzero_col_dim[i] += 1; }
		}
	}

	ret = init_cdsrsmatrix(rd, ptr_nzero_col_dim, nzero_total_num);

	/* 2nd pass: fill index pattern + re/im values (per-row padded layout) */
	total_index = 0;
	for(i = 0; i < rd; i++)
	{
		j_index = 0;
		for(j = 0; j < cd; j++)
		{
			cdsfloat *e = get_cdsmatrix_ij(org_mat, i, j);
			int nz = 0;
			for(c = 0; c < DSSIZE; c++) if(e->val_re[c] != 0.0f || e->val_im[c] != 0.0f) nz = 1;
			if(nz)
			{
				ret->re->nzero_index[i][j_index] = j;
				ret->im->nzero_index[i][j_index] = j;
				for(c = 0; c < DSSIZE; c++)
				{
					ret->re->element[c][total_index] = e->val_re[c];
					ret->im->element[c][total_index] = e->val_im[c];
				}
				total_index += 1;
				j_index += 1;
			}
		}
		total_index += ret->re->real_nzero_col_dim[i] - ret->re->nzero_col_dim[i]; /* skip padding */
	}

	free(ptr_nzero_col_dim);
	return ret;
}

/* y := A * x  (complex) = (Are*xre - Aim*xim) + i (Are*xim + Aim*xre) */
int mul_cdsrsmatrix_cdsvec(CDSVector ret, CDSRSMatrix mat, CDSVector vec)
{
	DSVector t[4];
	long int n = mat->re->row_dim, i;

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	for(i = 0; i < 4; i++) t[i] = init_dsvector(n);

	mul_dsrsmatrix_dsvec(t[0], mat->re, vec->re);
	mul_dsrsmatrix_dsvec(t[1], mat->re, vec->im);
	mul_dsrsmatrix_dsvec(t[2], mat->im, vec->re);
	mul_dsrsmatrix_dsvec(t[3], mat->im, vec->im);

	sub_dsvector(ret->re, t[0], t[3]);
	add_dsvector(ret->im, t[1], t[2]);

	for(i = 0; i < 4; i++) free_dsvector(t[i]);
	return SUCCESS;
}

/* y := A^T * x  (complex, non-conjugated transpose) */
int mul_cdsrsmatrixt_cdsvec(CDSVector ret, CDSRSMatrix mat, CDSVector vec)
{
	DSVector t[4];
	long int n = vec->re->dim, i;

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	for(i = 0; i < 4; i++) t[i] = init_dsvector(n);

	mul_dsrsmatrixt_dsvec(t[0], mat->re, vec->re);
	mul_dsrsmatrixt_dsvec(t[1], mat->re, vec->im);
	mul_dsrsmatrixt_dsvec(t[2], mat->im, vec->re);
	mul_dsrsmatrixt_dsvec(t[3], mat->im, vec->im);

	sub_dsvector(ret->re, t[0], t[3]);
	add_dsvector(ret->im, t[1], t[2]);

	for(i = 0; i < 4; i++) free_dsvector(t[i]);
	return SUCCESS;
}

