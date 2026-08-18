/********************************************************************************/
/* sparse_qs.c : Double-single (float-float, QSSIZE=2) real sparse matrix/SpMV  */
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

#ifdef USE_QSLINEAR

#ifndef EMPTY
#define EMPTY (-1)
#endif
#ifndef SUCCESS
#define SUCCESS (0)
#endif
#ifndef ERROR
#define ERROR (-1)
#endif

/* initialize QSRSMatrix */
QSRSMatrix init_qsrsmatrix(long int row_dim, long int *nzero_col_dim, long int nzero_total_num)
{
	QSRSMatrix ret;
	long int i, j, c;

	ret = (QSRSMatrix)malloc(sizeof(qsrsmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "Cannot allocate QSRSMatrix\n");
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
		fprintf(stderr, "Cannot allocate QSRSMatrix(nzero_index!)\n");
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
			fprintf(stderr, "Cannot allocate QSRSMatrix(nzero_index[%ld]!)\n", i);
			return NULL;
		}
		for(j = 0; j < nzero_col_dim[i]; j++)
			ret->nzero_index[i][j] = EMPTY;

		ret->real_nzero_total_num += ret->real_nzero_col_dim[i];
	}

	/* allocate element components (aligned, padded, zeroed) */
	for(c = 0; c < QSSIZE; c++)
	{
		ret->element[c] = (float *)BNC_CALLOC(ret->real_nzero_total_num, sizeof(float));
		if(ret->element[c] == NULL)
		{
			fprintf(stderr, "Cannot allocate QSRSMatrix(element[%ld]!)\n", c);
			return NULL;
		}
		for(i = 0; i < ret->real_nzero_total_num; i++)
			ret->element[c][i] = 0.0f;
	}

	return ret;
}

/* Clear QSRSMatrix */
void free_qsrsmatrix(QSRSMatrix mat)
{
	long int i, c;

	for(c = 0; c < QSSIZE; c++)
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
void set_nzero_row_dim_qs(QSRSMatrix mat)
{
	long int i, j;

	for(i = 0; i < mat->col_dim; i++)
		mat->nzero_row_dim[i] = 0;
	for(i = 0; i < mat->row_dim; i++)
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
			mat->nzero_row_dim[mat->nzero_index[i][j]]++;
}

/* mat := 0 (keep sparsity pattern) */
void set0_qsrsmatrix(QSRSMatrix spmat)
{
	long int index, c;

	for(c = 0; c < QSSIZE; c++)
		for(index = 0; index < spmat->real_nzero_total_num; index++)
			spmat->element[c][index] = 0.0f;
}

/* get the QSRSMatrix ij-element */
void get_qsrsmatrix_ij(float ret[QSSIZE], QSRSMatrix mat, long int row_index, long int col_index)
{
	long int i, j, c, total_index;

	for(c = 0; c < QSSIZE; c++) ret[c] = 0.0f;

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
				for(c = 0; c < QSSIZE; c++) ret[c] = mat->element[c][total_index];
				return;
			}
			total_index++;
		}
	}
}

/* set the QSRSMatrix ij-element */
void set_qsrsmatrix_ij(QSRSMatrix mat, long int row_index, long int col_index, float val[QSSIZE])
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
				for(c = 0; c < QSSIZE; c++) mat->element[c][total_index] = val[c];
				return;
			}
			total_index++;
		}
	}
}

/* initialize and set QSRSMatrix from dense QSMatrix */
QSRSMatrix init_set_qsrsmatrix_qsmatrix(QSMatrix org_mat)
{
	long int i, j, c;
	long int nzero_total_num, total_index, j_index;
	long int *ptr_nzero_col_dim;
	float v[QSSIZE];
	QSRSMatrix ret;

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
			float *e = get_qsmatrix_ij(org_mat, i, j);
			int nz = 0;
			for(c = 0; c < QSSIZE; c++) if(e[c] != 0.0f) nz = 1;
			if(nz)
			{
				nzero_total_num++;
				ptr_nzero_col_dim[i] += 1;
			}
		}
	}

	ret = init_qsrsmatrix(org_mat->row_dim, ptr_nzero_col_dim, nzero_total_num);

	/* 2nd pass: fill nonzeros (per-row padded layout) */
	total_index = 0;
	for(i = 0; i < ret->row_dim; i++)
	{
		j_index = 0;
		for(j = 0; j < ret->col_dim; j++)
		{
			float *e = get_qsmatrix_ij(org_mat, i, j);
			int nz = 0;
			for(c = 0; c < QSSIZE; c++) if(e[c] != 0.0f) nz = 1;
			if(nz)
			{
				ret->nzero_index[i][j_index] = j;
				for(c = 0; c < QSSIZE; c++) ret->element[c][total_index] = e[c];
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

/* Multiply QSRSMatrix * QSVector : ret := mat * vec */
int mul_qsrsmatrix_qsvec(QSVector ret, QSRSMatrix mat, QSVector vec)
{
	long int i, j, c, total_index;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256 a8[QSSIZE], vb8[QSSIZE], tmp8[QSSIZE], tmp8_mul[QSSIZE];
	long int jmax, k;
	float vset[QSSIZE][_BNC_S_WIDTH], lane[QSSIZE][_BNC_S_WIDTH];
	float mat_vec_i[QSSIZE], pair[QSSIZE];

	for(i = 0; i < mat->row_dim; i++)
	{
		_bncavx2_set0_qs(tmp8);
		jmax = mat->real_nzero_col_dim[i];
		for(j = 0; j < jmax; j += _BNC_S_WIDTH)
		{
			for(c = 0; c < QSSIZE; c++)
				a8[c] = _mm256_load_ps(&(mat->element[c][total_index]));
			for(c = 0; c < QSSIZE; c++)
			{
				for(k = 0; k < _BNC_S_WIDTH; k++)
					vset[c][k] = ((j + k) < mat->nzero_col_dim[i]) ? vec->element[c][mat->nzero_index[i][j + k]] : 0.0f;
				vb8[c] = _mm256_loadu_ps(vset[c]);
			}
			_bncavx2_rqs_mul(tmp8_mul, a8, vb8);
			_bncavx2_rqs_add(tmp8, tmp8, tmp8_mul);
			total_index += _BNC_S_WIDTH;
		}
		for(c = 0; c < QSSIZE; c++) { _mm256_storeu_ps(lane[c], tmp8[c]); mat_vec_i[c] = 0.0f; }
		for(k = 0; k < _BNC_S_WIDTH; k++)
		{
			for(c = 0; c < QSSIZE; c++) pair[c] = lane[c][k];
			rqs_add(mat_vec_i, mat_vec_i, pair);
		}
		set_qsvector_i(ret, i, mat_vec_i);
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512 a16[QSSIZE], vb16[QSSIZE], tmp16[QSSIZE], tmp16_mul[QSSIZE];
	long int jmax, k;
	float vset[QSSIZE][_BNC_S_WIDTH], lane[QSSIZE][_BNC_S_WIDTH];
	float mat_vec_i[QSSIZE], pair[QSSIZE];

	for(i = 0; i < mat->row_dim; i++)
	{
		_bncavx512_set0_qs(tmp16);
		jmax = mat->real_nzero_col_dim[i];
		for(j = 0; j < jmax; j += _BNC_S_WIDTH)
		{
			for(c = 0; c < QSSIZE; c++)
				a16[c] = _mm512_load_ps(&(mat->element[c][total_index]));
			for(c = 0; c < QSSIZE; c++)
			{
				for(k = 0; k < _BNC_S_WIDTH; k++)
					vset[c][k] = ((j + k) < mat->nzero_col_dim[i]) ? vec->element[c][mat->nzero_index[i][j + k]] : 0.0f;
				vb16[c] = _mm512_loadu_ps(vset[c]);
			}
			_bncavx512_rqs_mul(tmp16_mul, a16, vb16);
			_bncavx512_rqs_add(tmp16, tmp16, tmp16_mul);
			total_index += _BNC_S_WIDTH;
		}
		for(c = 0; c < QSSIZE; c++) { _mm512_storeu_ps(lane[c], tmp16[c]); mat_vec_i[c] = 0.0f; }
		for(k = 0; k < _BNC_S_WIDTH; k++)
		{
			for(c = 0; c < QSSIZE; c++) pair[c] = lane[c][k];
			rqs_add(mat_vec_i, mat_vec_i, pair);
		}
		set_qsvector_i(ret, i, mat_vec_i);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2
	{
		long _vl = (long)svcntw();
		float *gbuf[QSSIZE], *lbuf[QSSIZE];
		long int jmax, k;
		float mat_vec_i[QSSIZE], pair[QSSIZE];
		svbool_t pg = svptrue_b32();
		for(c = 0; c < QSSIZE; c++) { gbuf[c] = (float *)BNC_MALLOC(sizeof(float) * _vl); lbuf[c] = (float *)BNC_MALLOC(sizeof(float) * _vl); }

		for(i = 0; i < mat->row_dim; i++)
		{
			svfloat32_t t0, t1, t2, t3, a0, a1, a2, a3, b0, b1, b2, b3, m0, m1, m2, m3;
			_bncsve2_rqs_set0(&t0, &t1, &t2, &t3);
			jmax = mat->real_nzero_col_dim[i];
			for(j = 0; j < jmax; j += _vl)
			{
				a0 = svld1_f32(pg, &(mat->element[0][total_index]));
				a1 = svld1_f32(pg, &(mat->element[1][total_index]));
				a2 = svld1_f32(pg, &(mat->element[2][total_index]));
				a3 = svld1_f32(pg, &(mat->element[3][total_index]));
				for(k = 0; k < _vl; k++)
				{
					if((j + k) < mat->nzero_col_dim[i])
					{
						long idx = mat->nzero_index[i][j + k];
						gbuf[0][k] = vec->element[0][idx];
						gbuf[1][k] = vec->element[1][idx];
						gbuf[2][k] = vec->element[2][idx];
						gbuf[3][k] = vec->element[3][idx];
					}
					else { gbuf[0][k] = 0.0f; gbuf[1][k] = 0.0f; gbuf[2][k] = 0.0f; gbuf[3][k] = 0.0f; }
				}
				b0 = svld1_f32(pg, gbuf[0]);
				b1 = svld1_f32(pg, gbuf[1]);
				b2 = svld1_f32(pg, gbuf[2]);
				b3 = svld1_f32(pg, gbuf[3]);
				_bncsve2_rqs_mul(pg, &m0, &m1, &m2, &m3, a0, a1, a2, a3, b0, b1, b2, b3);
				_bncsve2_rqs_add(pg, &t0, &t1, &t2, &t3, t0, t1, t2, t3, m0, m1, m2, m3);
				total_index += _vl;
			}
			svst1_f32(pg, lbuf[0], t0);
			svst1_f32(pg, lbuf[1], t1);
			svst1_f32(pg, lbuf[2], t2);
			svst1_f32(pg, lbuf[3], t3);
			mat_vec_i[0] = 0.0f; mat_vec_i[1] = 0.0f; mat_vec_i[2] = 0.0f; mat_vec_i[3] = 0.0f;
			for(k = 0; k < _vl; k++)
			{
				pair[0] = lbuf[0][k]; pair[1] = lbuf[1][k]; pair[2] = lbuf[2][k]; pair[3] = lbuf[3][k];
				rqs_add(mat_vec_i, mat_vec_i, pair);
			}
			set_qsvector_i(ret, i, mat_vec_i);
		}
		for(c = 0; c < QSSIZE; c++) { free(gbuf[c]); free(lbuf[c]); }
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM Neon
	float32x4_t a4[QSSIZE], vb4[QSSIZE], tmp4[QSSIZE], tmp4_mul[QSSIZE];
	long int jmax, k;
	float vset[QSSIZE][4], lane[QSSIZE][4];
	float mat_vec_i[QSSIZE], pair[QSSIZE];

	for(i = 0; i < mat->row_dim; i++)
	{
		_bncneon_set0_qs(tmp4);
		jmax = mat->real_nzero_col_dim[i];
		for(j = 0; j < jmax; j += 4)
		{
			for(c = 0; c < QSSIZE; c++)
				a4[c] = vld1q_f32(&(mat->element[c][total_index]));
			for(c = 0; c < QSSIZE; c++)
			{
				for(k = 0; k < 4; k++)
					vset[c][k] = ((j + k) < mat->nzero_col_dim[i]) ? vec->element[c][mat->nzero_index[i][j + k]] : 0.0f;
				vb4[c] = vld1q_f32(vset[c]);
			}
			_bncneon_rqs_mul(tmp4_mul, a4, vb4);
			_bncneon_rqs_add(tmp4, tmp4, tmp4_mul);
			total_index += 4;
		}
		for(c = 0; c < QSSIZE; c++) { vst1q_f32(lane[c], tmp4[c]); mat_vec_i[c] = 0.0f; }
		for(k = 0; k < 4; k++)
		{
			for(c = 0; c < QSSIZE; c++) pair[c] = lane[c][k];
			rqs_add(mat_vec_i, mat_vec_i, pair);
		}
		set_qsvector_i(ret, i, mat_vec_i);
	}

#else // scalar
	float mat_ij[QSSIZE], vec_j[QSSIZE], prod[QSSIZE], acc[QSSIZE];
	for(i = 0; i < mat->row_dim; i++)
	{
		for(c = 0; c < QSSIZE; c++) acc[c] = 0.0f;
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			for(c = 0; c < QSSIZE; c++)
			{
				mat_ij[c] = mat->element[c][total_index];
				vec_j[c]  = vec->element[c][mat->nzero_index[i][j]];
			}
			rqs_mul(prod, mat_ij, vec_j);
			rqs_add(acc, acc, prod);
			total_index++;
		}
		total_index += mat->real_nzero_col_dim[i] - mat->nzero_col_dim[i]; /* skip padding */
		set_qsvector_i(ret, i, acc);
	}
#endif

	return SUCCESS;
}

/* Multiply QSRSMatrix^T * QSVector : ret := mat^T * vec (scatter accumulation) */
int mul_qsrsmatrixt_qsvec(QSVector ret, QSRSMatrix mat, QSVector vec)
{
	long int i, j, c, total_index;
	float mat_ij[QSSIZE], vec_i[QSSIZE], prod[QSSIZE], acc[QSSIZE];

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	for(i = 0; i < ret->dim; i++)
	{
		float z[QSSIZE]; for(c = 0; c < QSSIZE; c++) z[c] = 0.0f;
		set_qsvector_i(ret, i, z);
	}

	total_index = 0;
	for(i = 0; i < mat->row_dim; i++)
	{
		for(c = 0; c < QSSIZE; c++) vec_i[c] = vec->element[c][i];
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			long idx = mat->nzero_index[i][j];
			for(c = 0; c < QSSIZE; c++)
			{
				mat_ij[c] = mat->element[c][total_index];
				acc[c] = ret->element[c][idx];
			}
			rqs_mul(prod, mat_ij, vec_i);
			rqs_add(acc, acc, prod);
			for(c = 0; c < QSSIZE; c++) ret->element[c][idx] = acc[c];
			total_index++;
		}
		total_index += mat->real_nzero_col_dim[i] - mat->nzero_col_dim[i]; /* skip padding */
	}

	return SUCCESS;
}

#endif // USE_QSLINEAR
