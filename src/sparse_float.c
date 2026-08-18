/********************************************************************************/
/* sparse_float.c : Single-precision (float) real sparse matrix/SpMV            */
/*                                                                              */
/* SpMV implementations: scalar + AVX2 + AVX-512 + NEON + SVE2.                 */
/* Element layout is per-row padded to a multiple of _BNC_S_WIDTH (the float    */
/* SIMD width): each row holds real_nzero_col_dim[i] contiguous slots, the      */
/* first nzero_col_dim[i] are nonzeros and the rest are zero padding.  Every    */
/* SpMV processes the full padded row (padding contributes 0) and guards the    */
/* irregular x-gather so out-of-range lanes read 0, so total_index advances by  */
/* real_nzero_col_dim[i] per row with no residual handling.                     */
/*                                                                              */
/* Copyright (c) 2026 Tomonori Kouya                                            */
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bncsparse.h"

#ifdef USE_FLINEAR

#ifndef EMPTY
#define EMPTY (-1)
#endif
#ifndef SUCCESS
#define SUCCESS (0)
#endif
#ifndef ERROR
#define ERROR (-1)
#endif

/* initialize FRSMatrix */
FRSMatrix init_frsmatrix(long int row_dim, long int *nzero_col_dim, long int nzero_total_num)
{
	FRSMatrix ret;
	long int i, j;

	ret = (FRSMatrix)malloc(sizeof(frsmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "Cannot allocate FRSMatrix\n");
		return ret;
	}
	if((row_dim < 0) || (nzero_total_num < 0))
	{
		fprintf(stderr, "Illigal nzero values\n");
		return NULL;
	}

	ret->row_dim = row_dim;
	ret->col_dim = row_dim; // Square matrix only
	ret->nzero_total_num = nzero_total_num;

	ret->nzero_col_dim = (long int *)calloc(ret->row_dim, sizeof(long int));
	ret->nzero_row_dim = (long int *)calloc(ret->col_dim, sizeof(long int));
	ret->nzero_index = (long int **)calloc(ret->row_dim, sizeof(long int *));
	ret->real_nzero_col_dim = (long int *)calloc(ret->row_dim, sizeof(long int));
	if(ret->nzero_index == NULL)
	{
		fprintf(stderr, "Cannot allocate FRSMatrix(nzero_index!)\n");
		return NULL;
	}

	ret->real_nzero_total_num = 0;
	for(i = 0; i < row_dim; i++)
	{
		if(nzero_col_dim[i] < 0)
		{
			fprintf(stderr, "Illigal nzero values(nzero_col_dim[%ld])\n", i);
			return NULL;
		}
		ret->nzero_col_dim[i] = nzero_col_dim[i];
		// alignment for SIMD (float width)
		ret->real_nzero_col_dim[i] = (long int)ceil((double)(nzero_col_dim[i]) / (double)_BNC_S_WIDTH) * _BNC_S_WIDTH;
		ret->nzero_index[i] = (long int *)calloc(nzero_col_dim[i], sizeof(long int));
		if(ret->nzero_index[i] == NULL)
		{
			fprintf(stderr, "Cannot allocate FRSMatrix(nzero_index[%ld]!)\n", i);
			return NULL;
		}
		for(j = 0; j < nzero_col_dim[i]; j++)
			ret->nzero_index[i][j] = EMPTY;

		ret->real_nzero_total_num += ret->real_nzero_col_dim[i];
	}

	/* allocate element array (aligned, padded, zeroed) */
	ret->element = (float *)BNC_CALLOC(ret->real_nzero_total_num, sizeof(float));
	if(ret->element == NULL)
	{
		fprintf(stderr, "Cannot allocate FRSMatrix(element!)\n");
		return NULL;
	}
	for(i = 0; i < ret->real_nzero_total_num; i++)
		ret->element[i] = 0.0f;

	return ret;
}

/* Clear FRSMatrix */
void free_frsmatrix(FRSMatrix mat)
{
	long int i;

	free(mat->element);
	free(mat->nzero_col_dim);
	free(mat->nzero_row_dim);
	for(i = 0; i < mat->row_dim; i++)
		free(mat->nzero_index[i]);
	free(mat->nzero_index);
	free(mat->real_nzero_col_dim);
	free(mat);
}

/* set nzero_row_dim automatically */
void set_nzero_row_dim_f(FRSMatrix mat)
{
	long int i, j;

	for(i = 0; i < mat->col_dim; i++)
		mat->nzero_row_dim[i] = 0;
	for(i = 0; i < mat->row_dim; i++)
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
			mat->nzero_row_dim[mat->nzero_index[i][j]]++;
}

/* mat := 0 (keep sparsity pattern) */
void set0_frsmatrix(FRSMatrix spmat)
{
	long int index;

	for(index = 0; index < spmat->real_nzero_total_num; index++)
		spmat->element[index] = 0.0f;
}

/* get the FRSMatrix ij-element */
float get_frsmatrix_ij(FRSMatrix mat, long int row_index, long int col_index)
{
	long int i, j, total_index;

	if((row_index < 0) || (row_index >= mat->row_dim) || (col_index < 0) || (col_index >= mat->col_dim))
	{
		fprintf(stderr, "Warning: row_index(%ld) or col_index(%ld) is illegal!\n", row_index, col_index);
		return 0.0f;
	}

	if(mat->real_nzero_col_dim[row_index] >= 1)
	{
		total_index = 0;
		for(i = 0; i < row_index; i++)
			total_index += mat->real_nzero_col_dim[i];
		for(j = 0; j < mat->nzero_col_dim[row_index]; j++)
		{
			if(mat->nzero_index[row_index][j] == col_index)
				return mat->element[total_index];
			total_index++;
		}
	}

	return 0.0f;
}

/* set the FRSMatrix ij-element */
void set_frsmatrix_ij(FRSMatrix mat, long int row_index, long int col_index, float val)
{
	long int i, j, total_index;

	if((row_index < 0) || (row_index >= mat->row_dim) || (col_index < 0) || (col_index >= mat->col_dim))
	{
		fprintf(stderr, "Warning: row_index(%ld) or col_index(%ld) is illegal!\n", row_index, col_index);
		return;
	}

	if(mat->nzero_col_dim[row_index] >= 1)
	{
		total_index = 0;
		for(i = 0; i < row_index; i++)
			total_index += mat->real_nzero_col_dim[i];
		for(j = 0; j < mat->nzero_col_dim[row_index]; j++)
		{
			if(mat->nzero_index[row_index][j] == col_index)
			{
				mat->element[total_index] = val;
				return;
			}
			total_index++;
		}
	}
}

/* initialize and set FRSMatrix from dense FMatrix */
FRSMatrix init_set_frsmatrix_fmatrix(FMatrix org_mat)
{
	long int i, j;
	long int nzero_total_num, total_index, j_index;
	long int *ptr_nzero_col_dim;
	FRSMatrix ret;

	nzero_total_num = 0;
	ptr_nzero_col_dim = (long int *)malloc((size_t)(sizeof(long int) * (org_mat->row_dim)));
	if(ptr_nzero_col_dim == NULL)
	{
		fprintf(stderr, "Cannot allocate ptr_nzero_col_dim\n");
		return NULL;
	}

	/* 1st pass: count nonzeros per row */
	for(i = 0; i < org_mat->row_dim; i++)
	{
		ptr_nzero_col_dim[i] = 0;
		for(j = 0; j < org_mat->col_dim; j++)
		{
			if(get_fmatrix_ij(org_mat, i, j) != 0.0f)
			{
				nzero_total_num++;
				ptr_nzero_col_dim[i] += 1;
			}
		}
	}

	ret = init_frsmatrix(org_mat->row_dim, ptr_nzero_col_dim, nzero_total_num);

	/* 2nd pass: fill nonzeros (per-row padded layout) */
	total_index = 0;
	for(i = 0; i < ret->row_dim; i++)
	{
		j_index = 0;
		for(j = 0; j < ret->col_dim; j++)
		{
			float e = get_fmatrix_ij(org_mat, i, j);
			if(e != 0.0f)
			{
				ret->nzero_index[i][j_index] = j;
				ret->element[total_index] = e;
				total_index += 1;
				j_index += 1;
			}
		}
		total_index += ret->real_nzero_col_dim[i] - ret->nzero_col_dim[i]; /* skip padding */
	}

	free(ptr_nzero_col_dim);
	return ret;
}

/* Multiply FRSMatrix * FVector : ret := mat * vec */
int mul_frsmatrix_fvec(FVector ret, FRSMatrix mat, FVector vec)
{
	long int i, j, total_index;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a8, vb8, tmp8;
	long int jmax, k;
	float vset[_BNC_S_WIDTH], lane[_BNC_S_WIDTH];
	float mat_vec_i;

	for(i = 0; i < mat->row_dim; i++)
	{
		tmp8 = _mm256_setzero_ps();
		jmax = mat->real_nzero_col_dim[i];
		for(j = 0; j < jmax; j += _BNC_S_WIDTH)
		{
			a8 = _mm256_load_ps(&(mat->element[total_index]));
			for(k = 0; k < _BNC_S_WIDTH; k++)
				vset[k] = ((j + k) < mat->nzero_col_dim[i]) ? vec->element[mat->nzero_index[i][j + k]] : 0.0f;
			vb8 = _mm256_loadu_ps(vset);
			tmp8 = _mm256_fmadd_ps(a8, vb8, tmp8);
			total_index += _BNC_S_WIDTH;
		}
		_mm256_storeu_ps(lane, tmp8);
		mat_vec_i = 0.0f;
		for(k = 0; k < _BNC_S_WIDTH; k++)
			mat_vec_i += lane[k];
		set_fvector_i(ret, i, mat_vec_i);
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512 a16, vb16, tmp16;
	long int jmax, k;
	float vset[_BNC_S_WIDTH], lane[_BNC_S_WIDTH];
	float mat_vec_i;

	for(i = 0; i < mat->row_dim; i++)
	{
		tmp16 = _mm512_setzero_ps();
		jmax = mat->real_nzero_col_dim[i];
		for(j = 0; j < jmax; j += _BNC_S_WIDTH)
		{
			a16 = _mm512_load_ps(&(mat->element[total_index]));
			for(k = 0; k < _BNC_S_WIDTH; k++)
				vset[k] = ((j + k) < mat->nzero_col_dim[i]) ? vec->element[mat->nzero_index[i][j + k]] : 0.0f;
			vb16 = _mm512_loadu_ps(vset);
			tmp16 = _mm512_fmadd_ps(a16, vb16, tmp16);
			total_index += _BNC_S_WIDTH;
		}
		_mm512_storeu_ps(lane, tmp16);
		mat_vec_i = 0.0f;
		for(k = 0; k < _BNC_S_WIDTH; k++)
			mat_vec_i += lane[k];
		set_fvector_i(ret, i, mat_vec_i);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2
	{
		long _vl = (long)svcntw();
		float *gbuf, *lbuf;
		long int jmax, k;
		float mat_vec_i;
		svbool_t pg = svptrue_b32();
		gbuf = (float *)BNC_MALLOC(sizeof(float) * _vl);
		lbuf = (float *)BNC_MALLOC(sizeof(float) * _vl);

		for(i = 0; i < mat->row_dim; i++)
		{
			svfloat32_t tmp_v = svdup_n_f32(0.0f);
			jmax = mat->real_nzero_col_dim[i];
			for(j = 0; j < jmax; j += _vl)
			{
				svfloat32_t a_v = svld1_f32(pg, &(mat->element[total_index]));
				for(k = 0; k < _vl; k++)
					gbuf[k] = ((j + k) < mat->nzero_col_dim[i]) ? vec->element[mat->nzero_index[i][j + k]] : 0.0f;
				svfloat32_t b_v = svld1_f32(pg, gbuf);
				tmp_v = svmla_f32_x(pg, tmp_v, a_v, b_v);
				total_index += _vl;
			}
			svst1_f32(pg, lbuf, tmp_v);
			mat_vec_i = 0.0f;
			for(k = 0; k < _vl; k++)
				mat_vec_i += lbuf[k];
			set_fvector_i(ret, i, mat_vec_i);
		}
		free(gbuf);
		free(lbuf);
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM Neon
	float32x4_t a4, vb4, tmp4;
	long int jmax, k;
	float vset[4], lane[4];
	float mat_vec_i;

	for(i = 0; i < mat->row_dim; i++)
	{
		tmp4 = vdupq_n_f32(0.0f);
		jmax = mat->real_nzero_col_dim[i];
		for(j = 0; j < jmax; j += 4)
		{
			a4 = vld1q_f32(&(mat->element[total_index]));
			for(k = 0; k < 4; k++)
				vset[k] = ((j + k) < mat->nzero_col_dim[i]) ? vec->element[mat->nzero_index[i][j + k]] : 0.0f;
			vb4 = vld1q_f32(vset);
			tmp4 = vfmaq_f32(tmp4, a4, vb4);
			total_index += 4;
		}
		vst1q_f32(lane, tmp4);
		mat_vec_i = 0.0f;
		for(k = 0; k < 4; k++)
			mat_vec_i += lane[k];
		set_fvector_i(ret, i, mat_vec_i);
	}

#else // scalar
	float acc;
	for(i = 0; i < mat->row_dim; i++)
	{
		acc = 0.0f;
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			acc += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			total_index++;
		}
		total_index += mat->real_nzero_col_dim[i] - mat->nzero_col_dim[i]; /* skip padding */
		set_fvector_i(ret, i, acc);
	}
#endif

	return SUCCESS;
}

/* Multiply FRSMatrix^T * FVector : ret := mat^T * vec (scatter accumulation) */
int mul_frsmatrixt_fvec(FVector ret, FRSMatrix mat, FVector vec)
{
	long int i, j, total_index;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	for(i = 0; i < ret->dim; i++)
		set_fvector_i(ret, i, 0.0f);

	total_index = 0;
	for(i = 0; i < mat->row_dim; i++)
	{
		float vec_i = vec->element[i];
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			long idx = mat->nzero_index[i][j];
			ret->element[idx] += mat->element[total_index] * vec_i;
			total_index++;
		}
		total_index += mat->real_nzero_col_dim[i] - mat->nzero_col_dim[i]; /* skip padding */
	}

	return SUCCESS;
}

#endif // USE_FLINEAR
