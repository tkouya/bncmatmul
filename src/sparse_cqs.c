/********************************************************************************/
/* sparse_cqs.c : Complex quad-single (cds) real-sparse matrix and SpMV.       */
/*                                                                              */
/* A complex CDS sparse matrix is a pair {re, im} of real QSRSMatrix sharing the */
/* same sparsity pattern.  The complex SpMV is computed from four real DS SpMVs  */
/* (mul_qsrsmatrix_qsvec), so it inherits the scalar/AVX2/AVX-512/NEON/SVE2      */
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

/* initialize CQSRSMatrix (re and im share the same sparsity layout) */
CQSRSMatrix init_cqsrsmatrix(long int row_dim, long int *nzero_col_dim, long int nzero_total_num)
{
	CQSRSMatrix ret = (CQSRSMatrix)malloc(sizeof(cqsrsmatrix));
	if(ret == NULL) { fprintf(stderr, "Cannot allocate CQSRSMatrix\n"); return NULL; }
	ret->re = init_qsrsmatrix(row_dim, nzero_col_dim, nzero_total_num);
	ret->im = init_qsrsmatrix(row_dim, nzero_col_dim, nzero_total_num);
	if(ret->re == NULL || ret->im == NULL) { fprintf(stderr, "Cannot allocate CQSRSMatrix re/im\n"); return NULL; }
	return ret;
}

void free_cqsrsmatrix(CQSRSMatrix mat)
{
	free_qsrsmatrix(mat->re);
	free_qsrsmatrix(mat->im);
	free(mat);
}

void set0_cqsrsmatrix(CQSRSMatrix spmat)
{
	set0_qsrsmatrix(spmat->re);
	set0_qsrsmatrix(spmat->im);
}

/* set the CQSRSMatrix ij-element (index pattern must already be present) */
void set_cqsrsmatrix_ij(CQSRSMatrix mat, long int row_index, long int col_index, cqsfloat *val)
{
	set_qsrsmatrix_ij(mat->re, row_index, col_index, val->val_re);
	set_qsrsmatrix_ij(mat->im, row_index, col_index, val->val_im);
}

/* get the CQSRSMatrix ij-element */
void get_cqsrsmatrix_ij(cqsfloat *ret, CQSRSMatrix mat, long int row_index, long int col_index)
{
	get_qsrsmatrix_ij(ret->val_re, mat->re, row_index, col_index);
	get_qsrsmatrix_ij(ret->val_im, mat->im, row_index, col_index);
}

/* initialize and set CQSRSMatrix from dense CQSMatrix */
CQSRSMatrix init_set_cqsrsmatrix_cqsmatrix(CQSMatrix org_mat)
{
	long int i, j, c;
	long int nzero_total_num, total_index, j_index;
	long int *ptr_nzero_col_dim;
	CQSRSMatrix ret;
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
			cqsfloat *e = get_cqsmatrix_ij(org_mat, i, j);
			int nz = 0;
			for(c = 0; c < QSSIZE; c++) if(e->val_re[c] != 0.0f || e->val_im[c] != 0.0f) nz = 1;
			if(nz) { nzero_total_num++; ptr_nzero_col_dim[i] += 1; }
		}
	}

	ret = init_cqsrsmatrix(rd, ptr_nzero_col_dim, nzero_total_num);

	/* 2nd pass: fill index pattern + re/im values (per-row padded layout) */
	total_index = 0;
	for(i = 0; i < rd; i++)
	{
		j_index = 0;
		for(j = 0; j < cd; j++)
		{
			cqsfloat *e = get_cqsmatrix_ij(org_mat, i, j);
			int nz = 0;
			for(c = 0; c < QSSIZE; c++) if(e->val_re[c] != 0.0f || e->val_im[c] != 0.0f) nz = 1;
			if(nz)
			{
				ret->re->nzero_index[i][j_index] = j;
				ret->im->nzero_index[i][j_index] = j;
				for(c = 0; c < QSSIZE; c++)
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
int mul_cqsrsmatrix_cqsvec(CQSVector ret, CQSRSMatrix mat, CQSVector vec)
{
	QSVector t[4];
	long int n = mat->re->row_dim, i;

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	for(i = 0; i < 4; i++) t[i] = init_qsvector(n);

	mul_qsrsmatrix_qsvec(t[0], mat->re, vec->re);
	mul_qsrsmatrix_qsvec(t[1], mat->re, vec->im);
	mul_qsrsmatrix_qsvec(t[2], mat->im, vec->re);
	mul_qsrsmatrix_qsvec(t[3], mat->im, vec->im);

	sub_qsvector(ret->re, t[0], t[3]);
	add_qsvector(ret->im, t[1], t[2]);

	for(i = 0; i < 4; i++) free_qsvector(t[i]);
	return SUCCESS;
}

/* y := A^T * x  (complex, non-conjugated transpose) */
int mul_cqsrsmatrixt_cqsvec(CQSVector ret, CQSRSMatrix mat, CQSVector vec)
{
	QSVector t[4];
	long int n = vec->re->dim, i;

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	for(i = 0; i < 4; i++) t[i] = init_qsvector(n);

	mul_qsrsmatrixt_qsvec(t[0], mat->re, vec->re);
	mul_qsrsmatrixt_qsvec(t[1], mat->re, vec->im);
	mul_qsrsmatrixt_qsvec(t[2], mat->im, vec->re);
	mul_qsrsmatrixt_qsvec(t[3], mat->im, vec->im);

	sub_qsvector(ret->re, t[0], t[3]);
	add_qsvector(ret->im, t[1], t[2]);

	for(i = 0; i < 4; i++) free_qsvector(t[i]);
	return SUCCESS;
}

