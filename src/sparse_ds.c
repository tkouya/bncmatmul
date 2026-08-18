/********************************************************************************/
/* sparse_ds.c : Double-single (float-float, DSSIZE=2) real sparse matrix/SpMV  */
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

#ifdef USE_DSLINEAR

#ifndef EMPTY
#define EMPTY (-1)
#endif
#ifndef SUCCESS
#define SUCCESS (0)
#endif
#ifndef ERROR
#define ERROR (-1)
#endif

/* initialize DSRSMatrix */
DSRSMatrix init_dsrsmatrix(long int row_dim, long int *nzero_col_dim, long int nzero_total_num)
{
	DSRSMatrix ret;
	long int i, j, c;

	ret = (DSRSMatrix)malloc(sizeof(dsrsmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "Cannot allocate DSRSMatrix\n");
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
		fprintf(stderr, "Cannot allocate DSRSMatrix(nzero_index!)\n");
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
			fprintf(stderr, "Cannot allocate DSRSMatrix(nzero_index[%ld]!)\n", i);
			return NULL;
		}
		for(j = 0; j < nzero_col_dim[i]; j++)
			ret->nzero_index[i][j] = EMPTY;

		ret->real_nzero_total_num += ret->real_nzero_col_dim[i];
	}

	/* allocate element components (aligned, padded, zeroed) */
	if(ret->real_nzero_total_num < 0)
	{
		fprintf(stderr, "Illigal real_nzero_total_num(%ld)\n", ret->real_nzero_total_num);
		return NULL;
	}
	for(c = 0; c < DSSIZE; c++)
	{
		ret->element[c] = (float *)BNC_CALLOC(ret->real_nzero_total_num, sizeof(float));
		if(ret->element[c] == NULL)
		{
			fprintf(stderr, "Cannot allocate DSRSMatrix(element[%ld]!)\n", c);
			return NULL;
		}
		for(i = 0; i < ret->real_nzero_total_num; i++)
			ret->element[c][i] = 0.0f;
	}

	return ret;
}

/* Clear DSRSMatrix */
void free_dsrsmatrix(DSRSMatrix mat)
{
	long int i, c;

	for(c = 0; c < DSSIZE; c++)
		free(mat->element[c]);
	free(mat->nzero_col_dim);
	free(mat->nzero_row_dim);
	for(i = 0; i < mat->row_dim; i++)
		free(mat->nzero_index[i]);
	free(mat->nzero_index);
	free(mat->real_nzero_col_dim);
	free(mat);
}

/* set nzero_row_dim automatically */
void set_nzero_row_dim_ds(DSRSMatrix mat)
{
	long int i, j;

	for(i = 0; i < mat->col_dim; i++)
		mat->nzero_row_dim[i] = 0;
	for(i = 0; i < mat->row_dim; i++)
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
			mat->nzero_row_dim[mat->nzero_index[i][j]]++;
}

/* mat := 0 (keep sparsity pattern) */
void set0_dsrsmatrix(DSRSMatrix spmat)
{
	long int index, c;

	for(c = 0; c < DSSIZE; c++)
		for(index = 0; index < spmat->real_nzero_total_num; index++)
			spmat->element[c][index] = 0.0f;
}

/* get the DSRSMatrix ij-element */
void get_dsrsmatrix_ij(float ret[DSSIZE], DSRSMatrix mat, long int row_index, long int col_index)
{
	long int i, j, c, total_index;

	for(c = 0; c < DSSIZE; c++) ret[c] = 0.0f;

	if((row_index < 0) || (row_index >= mat->row_dim) || (col_index < 0) || (col_index >= mat->col_dim))
	{
		fprintf(stderr, "Warning: row_index(%ld) or col_index(%ld) is illegal!\n", row_index, col_index);
		return;
	}

	if(mat->real_nzero_col_dim[row_index] >= 1)
	{
		total_index = 0;
		for(i = 0; i < row_index; i++)
			total_index += mat->real_nzero_col_dim[i];
		for(j = 0; j < mat->nzero_col_dim[row_index]; j++)
		{
			if(mat->nzero_index[row_index][j] == col_index)
			{
				for(c = 0; c < DSSIZE; c++) ret[c] = mat->element[c][total_index];
				return;
			}
			total_index++;
		}
	}
}

/* set the DSRSMatrix ij-element */
void set_dsrsmatrix_ij(DSRSMatrix mat, long int row_index, long int col_index, float val[DSSIZE])
{
	long int i, j, c, total_index;

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
				for(c = 0; c < DSSIZE; c++) mat->element[c][total_index] = val[c];
				return;
			}
			total_index++;
		}
	}
}

/* initialize and set DSRSMatrix from dense DSMatrix */
DSRSMatrix init_set_dsrsmatrix_dsmatrix(DSMatrix org_mat)
{
	long int i, j, c;
	long int nzero_total_num, total_index, j_index;
	long int *ptr_nzero_col_dim;
	float v[DSSIZE];
	DSRSMatrix ret;

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
			float *e = get_dsmatrix_ij(org_mat, i, j);
			int nz = 0;
			for(c = 0; c < DSSIZE; c++) if(e[c] != 0.0f) nz = 1;
			if(nz)
			{
				nzero_total_num++;
				ptr_nzero_col_dim[i] += 1;
			}
		}
	}

	ret = init_dsrsmatrix(org_mat->row_dim, ptr_nzero_col_dim, nzero_total_num);

	/* 2nd pass: fill nonzeros (per-row padded layout) */
	total_index = 0;
	for(i = 0; i < ret->row_dim; i++)
	{
		j_index = 0;
		for(j = 0; j < ret->col_dim; j++)
		{
			float *e = get_dsmatrix_ij(org_mat, i, j);
			int nz = 0;
			for(c = 0; c < DSSIZE; c++) if(e[c] != 0.0f) nz = 1;
			if(nz)
			{
				ret->nzero_index[i][j_index] = j;
				for(c = 0; c < DSSIZE; c++) ret->element[c][total_index] = e[c];
				total_index += 1;
				j_index += 1;
			}
		}
		total_index += ret->real_nzero_col_dim[i] - ret->nzero_col_dim[i]; /* skip padding */
	}
	(void)v;

	free(ptr_nzero_col_dim);
	return ret;
}

/* Multiply DSRSMatrix * DSVector : ret := mat * vec */
int mul_dsrsmatrix_dsvec(DSVector ret, DSRSMatrix mat, DSVector vec)
{
	long int i, j, c, total_index;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a8[DSSIZE], vb8[DSSIZE], tmp8[DSSIZE], tmp8_mul[DSSIZE];
	long int jmax, k;
	float vset[DSSIZE][_BNC_S_WIDTH], lane[DSSIZE][_BNC_S_WIDTH];
	float mat_vec_i[DSSIZE], pair[DSSIZE];

	for(i = 0; i < mat->row_dim; i++)
	{
		_bncavx2_set0_ds(tmp8);
		jmax = mat->real_nzero_col_dim[i];
		for(j = 0; j < jmax; j += _BNC_S_WIDTH)
		{
			for(c = 0; c < DSSIZE; c++)
				a8[c] = _mm256_load_ps(&(mat->element[c][total_index]));
			for(c = 0; c < DSSIZE; c++)
			{
				for(k = 0; k < _BNC_S_WIDTH; k++)
					vset[c][k] = ((j + k) < mat->nzero_col_dim[i]) ? vec->element[c][mat->nzero_index[i][j + k]] : 0.0f;
				vb8[c] = _mm256_loadu_ps(vset[c]);
			}
			_bncavx2_rds_mul(tmp8_mul, a8, vb8);
			_bncavx2_rds_add(tmp8, tmp8, tmp8_mul);
			total_index += _BNC_S_WIDTH;
		}
		for(c = 0; c < DSSIZE; c++) { _mm256_storeu_ps(lane[c], tmp8[c]); mat_vec_i[c] = 0.0f; }
		for(k = 0; k < _BNC_S_WIDTH; k++)
		{
			for(c = 0; c < DSSIZE; c++) pair[c] = lane[c][k];
			rds_add(mat_vec_i, mat_vec_i, pair);
		}
		set_dsvector_i(ret, i, mat_vec_i);
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512 a16[DSSIZE], vb16[DSSIZE], tmp16[DSSIZE], tmp16_mul[DSSIZE];
	long int jmax, k;
	float vset[DSSIZE][_BNC_S_WIDTH], lane[DSSIZE][_BNC_S_WIDTH];
	float mat_vec_i[DSSIZE], pair[DSSIZE];

	for(i = 0; i < mat->row_dim; i++)
	{
		_bncavx512_set0_ds(tmp16);
		jmax = mat->real_nzero_col_dim[i];
		for(j = 0; j < jmax; j += _BNC_S_WIDTH)
		{
			for(c = 0; c < DSSIZE; c++)
				a16[c] = _mm512_load_ps(&(mat->element[c][total_index]));
			for(c = 0; c < DSSIZE; c++)
			{
				for(k = 0; k < _BNC_S_WIDTH; k++)
					vset[c][k] = ((j + k) < mat->nzero_col_dim[i]) ? vec->element[c][mat->nzero_index[i][j + k]] : 0.0f;
				vb16[c] = _mm512_loadu_ps(vset[c]);
			}
			_bncavx512_rds_mul(tmp16_mul, a16, vb16);
			_bncavx512_rds_add(tmp16, tmp16, tmp16_mul);
			total_index += _BNC_S_WIDTH;
		}
		for(c = 0; c < DSSIZE; c++) { _mm512_storeu_ps(lane[c], tmp16[c]); mat_vec_i[c] = 0.0f; }
		for(k = 0; k < _BNC_S_WIDTH; k++)
		{
			for(c = 0; c < DSSIZE; c++) pair[c] = lane[c][k];
			rds_add(mat_vec_i, mat_vec_i, pair);
		}
		set_dsvector_i(ret, i, mat_vec_i);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2
	{
		long _vl = (long)svcntw();
		float *gbuf[DSSIZE], *lbuf[DSSIZE];
		long int jmax, k;
		float mat_vec_i[DSSIZE], pair[DSSIZE];
		svbool_t pg = svptrue_b32();
		for(c = 0; c < DSSIZE; c++) { gbuf[c] = (float *)BNC_MALLOC(sizeof(float) * _vl); lbuf[c] = (float *)BNC_MALLOC(sizeof(float) * _vl); }

		for(i = 0; i < mat->row_dim; i++)
		{
			svfloat32_t t0, t1, a0, a1, b0, b1, m0, m1;
			_bncsve2_rds_set0(&t0, &t1);
			jmax = mat->real_nzero_col_dim[i];
			for(j = 0; j < jmax; j += _vl)
			{
				a0 = svld1_f32(pg, &(mat->element[0][total_index]));
				a1 = svld1_f32(pg, &(mat->element[1][total_index]));
				for(k = 0; k < _vl; k++)
				{
					if((j + k) < mat->nzero_col_dim[i])
					{
						long idx = mat->nzero_index[i][j + k];
						gbuf[0][k] = vec->element[0][idx];
						gbuf[1][k] = vec->element[1][idx];
					}
					else { gbuf[0][k] = 0.0f; gbuf[1][k] = 0.0f; }
				}
				b0 = svld1_f32(pg, gbuf[0]);
				b1 = svld1_f32(pg, gbuf[1]);
				_bncsve2_rds_mul(pg, &m0, &m1, a0, a1, b0, b1);
				_bncsve2_rds_add(pg, &t0, &t1, t0, t1, m0, m1);
				total_index += _vl;
			}
			svst1_f32(pg, lbuf[0], t0);
			svst1_f32(pg, lbuf[1], t1);
			mat_vec_i[0] = 0.0f; mat_vec_i[1] = 0.0f;
			for(k = 0; k < _vl; k++)
			{
				pair[0] = lbuf[0][k]; pair[1] = lbuf[1][k];
				rds_add(mat_vec_i, mat_vec_i, pair);
			}
			set_dsvector_i(ret, i, mat_vec_i);
		}
		for(c = 0; c < DSSIZE; c++) { free(gbuf[c]); free(lbuf[c]); }
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM Neon
	float32x4_t a4[DSSIZE], vb4[DSSIZE], tmp4[DSSIZE], tmp4_mul[DSSIZE];
	long int jmax, k;
	float vset[DSSIZE][4], lane[DSSIZE][4];
	float mat_vec_i[DSSIZE], pair[DSSIZE];

	for(i = 0; i < mat->row_dim; i++)
	{
		_bncneon_set0_ds(tmp4);
		jmax = mat->real_nzero_col_dim[i];
		for(j = 0; j < jmax; j += 4)
		{
			for(c = 0; c < DSSIZE; c++)
				a4[c] = vld1q_f32(&(mat->element[c][total_index]));
			for(c = 0; c < DSSIZE; c++)
			{
				for(k = 0; k < 4; k++)
					vset[c][k] = ((j + k) < mat->nzero_col_dim[i]) ? vec->element[c][mat->nzero_index[i][j + k]] : 0.0f;
				vb4[c] = vld1q_f32(vset[c]);
			}
			_bncneon_rds_mul(tmp4_mul, a4, vb4);
			_bncneon_rds_add(tmp4, tmp4, tmp4_mul);
			total_index += 4;
		}
		for(c = 0; c < DSSIZE; c++) { vst1q_f32(lane[c], tmp4[c]); mat_vec_i[c] = 0.0f; }
		for(k = 0; k < 4; k++)
		{
			for(c = 0; c < DSSIZE; c++) pair[c] = lane[c][k];
			rds_add(mat_vec_i, mat_vec_i, pair);
		}
		set_dsvector_i(ret, i, mat_vec_i);
	}

#else // scalar
	float mat_ij[DSSIZE], vec_j[DSSIZE], prod[DSSIZE], acc[DSSIZE];
	for(i = 0; i < mat->row_dim; i++)
	{
		for(c = 0; c < DSSIZE; c++) acc[c] = 0.0f;
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			for(c = 0; c < DSSIZE; c++)
			{
				mat_ij[c] = mat->element[c][total_index];
				vec_j[c]  = vec->element[c][mat->nzero_index[i][j]];
			}
			rds_mul(prod, mat_ij, vec_j);
			rds_add(acc, acc, prod);
			total_index++;
		}
		total_index += mat->real_nzero_col_dim[i] - mat->nzero_col_dim[i]; /* skip padding */
		set_dsvector_i(ret, i, acc);
	}
#endif

	return SUCCESS;
}

/* Multiply DSRSMatrix^T * DSVector : ret := mat^T * vec (scatter accumulation) */
int mul_dsrsmatrixt_dsvec(DSVector ret, DSRSMatrix mat, DSVector vec)
{
	long int i, j, c, total_index;
	float mat_ij[DSSIZE], vec_i[DSSIZE], prod[DSSIZE], acc[DSSIZE];

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	for(i = 0; i < ret->dim; i++)
	{
		float z[DSSIZE]; for(c = 0; c < DSSIZE; c++) z[c] = 0.0f;
		set_dsvector_i(ret, i, z);
	}

	total_index = 0;
	for(i = 0; i < mat->row_dim; i++)
	{
		for(c = 0; c < DSSIZE; c++) vec_i[c] = vec->element[c][i];
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			long idx = mat->nzero_index[i][j];
			for(c = 0; c < DSSIZE; c++)
			{
				mat_ij[c] = mat->element[c][total_index];
				acc[c] = ret->element[c][idx];
			}
			rds_mul(prod, mat_ij, vec_i);
			rds_add(acc, acc, prod);
			for(c = 0; c < DSSIZE; c++) ret->element[c][idx] = acc[c];
			total_index++;
		}
		total_index += mat->real_nzero_col_dim[i] - mat->nzero_col_dim[i]; /* skip padding */
	}

	return SUCCESS;
}

#endif // USE_DSLINEAR
