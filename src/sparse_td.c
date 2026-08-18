/********************************************************************************/
/*                                                                              */
/* sparse_td.c : Sparse Matrix and Vector Library (TD Precision)                */
/* Copyright (c) 2024 Tomonori Kouya, All rights reserved.                      */
/*                                                                              */
/* Version 0.2 2024-08-01 : Append Ozaki scheme for SpMV                        */
/* Version 0.1 2024-04-23 : Create sparse_td.c from sparse_td.c                 */
/*                                                                              */
/* This program is free software: you can redistribute it and/or modify it      */
/* under the terms of the GNU Lesser General Public License as published by the */
/* Free Software Foundation, either version 3 of the License or any later       */
/* version.                                                                     */
/*                                                                              */
/* This program is distributed in the hope that it will be useful, but WITHOUT  */
/* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or        */
/* FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License */
/* for more details.                                                            */
/*                                                                              */
/* You should have received a copy of the GNU Lesser General Public License     */
/* along with this program.  If not, see <http://www.gnu.org/licenses/>.        */
/*                                                                              */
/********************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bncsparse.h"

//#ifdef USE_GMP

/* Sparse matrix struct */
/* Example:             */
/*      0 1 2 3 4       */
/* A = [a 0 b c 0]0     */
/*     [0 d 0 0 0]1     */
/*     [0 e f 0 0]2     */
/*     [0 0 0 g 0]3     */
/*     [0 0 h 0 i]4     */
/*                      */
/* <--> element = [a b c d e f g h i] */
/*      row_dim = 5, col_dim = 5 */
/*      nzero_index[0] = [0 2 3] */
/*      nzero_index[1] = [4]     */
/*      nzero_index[2] = [1 2]   */
/*      nzero_index[3] = [3]     */
/*      nzero_index[4] = [2 4]   */
/*      nzero_col_dim[0] = 3 */
/*      nzero_col_dim[1] = 1 */
/*      nzero_col_dim[2] = 2 */
/*      nzero_col_dim[3] = 1 */
/*      nzero_col_dim[4] = 2 */
/*      nzero_total_num = 9  */
/*	    zero_element = 0UL   */

/* initialize TDRSMatrix */
TDRSMatrix init_tdrsmatrix(long int row_dim, long int *nzero_col_dim, long int nzero_total_num)
{
	TDRSMatrix ret;
	long int i, j;

	ret = (TDRSMatrix)malloc(sizeof(tdrsmatrix));
	//ret = (TDRSMatrix)BNC_MALLOC(sizeof(tdrsmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "Cannot allocate TDRSMatrix\n");
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

	/* allocate nzero_index */
	//printf("%ld %ld %ld %ld\n", ret->row_dim, ret->col_dim, ret->nzero_total_num, sizeof(long int *) * row_dim);
	// ret->nzero_col_dim = (long int *)malloc(sizeof(long int *) * row_dim);
	// ret->nzero_row_dim = (long int *)malloc(sizeof(long int *) * ret->col_dim);
	// ret->nzero_index = (long int **)malloc(sizeof(long int *) * row_dim);
	// Fix it!: 2024-04-25(Thu) T.Kouya
	ret->nzero_col_dim = (long int *)calloc(ret->row_dim, sizeof(long int));
	ret->nzero_row_dim = (long int *)calloc(ret->col_dim, sizeof(long int));
	ret->nzero_index = (long int **)calloc(ret->row_dim, sizeof(long int *));
	ret->real_nzero_col_dim = (long int *)calloc(ret->row_dim, sizeof(long int));

	if(ret->nzero_index == NULL)
	{
		fprintf(stderr, "Cannot allocate TDRSMatrix(nzero_index!)\n");
		return NULL;
	}

	// real_nzero_total_num := real_nzero_col_dim[0] + ... + real_nzero_col_dim[nzero_]
	ret->real_nzero_total_num = 0;
	for(i = 0; i < row_dim; i++)
	{
		if(nzero_col_dim[i] < 0)
		{
			fprintf(stderr, "Illigal nzero values(nzero_col_dim[%ld])\n", i);
			return NULL;
		}
		ret->nzero_col_dim[i] = nzero_col_dim[i];
		// alignment for SIMD
		ret->real_nzero_col_dim[i] = (long int)ceil((double)(nzero_col_dim[i]) / (double)_BNC_D_WIDTH) * _BNC_D_WIDTH;
		//ret->nzero_index[i] = (long int *)malloc(sizeof(long int) * nzero_col_dim[i]);
		ret->nzero_index[i] = (long int *)calloc(nzero_col_dim[i], sizeof(long int));
		if(ret->nzero_index[i] == NULL)
		{
			fprintf(stderr, "Cannot allocate TDRSMatrix(nzero_index[%ld]!)\n", i);
			return NULL;
		}
		for(j = 0; j < nzero_col_dim[i]; j++)
			ret->nzero_index[i][j] = EMPTY;

		ret->real_nzero_total_num += ret->real_nzero_col_dim[i];
	}
	
	/* allocate element */
	//ret->element[0] = (double *)calloc(nzero_total_num, sizeof(double));
	//ret->element[1] = (double *)calloc(nzero_total_num, sizeof(double));
	//ret->element[2] = (double *)calloc(nzero_total_num, sizeof(double));
	ret->element[0] = (double *)BNC_CALLOC(ret->real_nzero_total_num, sizeof(double));
	ret->element[1] = (double *)BNC_CALLOC(ret->real_nzero_total_num, sizeof(double));
	ret->element[2] = (double *)BNC_CALLOC(ret->real_nzero_total_num, sizeof(double));
	if((ret->element[0] == NULL) || (ret->element[1] == NULL)|| (ret->element[2] == NULL))
	{
		fprintf(stderr, "Cannot allocate TDRSMatrix(element!)\n");
		return NULL;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	for(i = 0; i < ret->real_nzero_total_num; i += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&(ret->element[0][i]), zero4);
		_mm256_store_pd(&(ret->element[1][i]), zero4);
		_mm256_store_pd(&(ret->element[2][i]), zero4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(i = 0; i < ret->real_nzero_total_num; i += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&(ret->element[0][i]), zero8);
		_mm512_store_pd(&(ret->element[1][i]), zero8);
		_mm512_store_pd(&(ret->element[2][i]), zero8);
	}
#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2
	{
		long _vl = (long)svcntd();
		svfloat64_t zerov = svdup_n_f64(0.0);
		for(i = 0; i < ret->real_nzero_total_num; i += _vl)
		{
			svbool_t pg = svwhilelt_b64_s64((int64_t)i,
			                                (int64_t)ret->real_nzero_total_num);
			svst1_f64(pg, &(ret->element[0][i]), zerov);
			svst1_f64(pg, &(ret->element[1][i]), zerov);
			svst1_f64(pg, &(ret->element[2][i]), zerov);
		}
	}
#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM Neon
	float64x2_t zero2;

	zero2 = vdupq_n_f64(0.0);
	for(i = 0; i < ret->real_nzero_total_num; i += 2)
	{
		vst1q_f64(&(ret->element[0][i]), zero2);
		vst1q_f64(&(ret->element[1][i]), zero2);
		vst1q_f64(&(ret->element[2][i]), zero2);
	}


#else // __AVX2__
	// element := 0
	for(i = 0; i < nzero_total_num; i++)
	{
		ret->element[0][i] = 0.0;
		ret->element[1][i] = 0.0;
		ret->element[2][i] = 0.0;
	}
#endif // __AVX2__

	return ret;
}

/* Clear TDRSMatrix */
void free_tdrsmatrix(TDRSMatrix mat)
{
	long int i;

//	free(mat->element);
	free(mat->nzero_col_dim);
	free(mat->nzero_row_dim);
	free(mat->real_nzero_col_dim);

	for(i = 0; i < mat->row_dim; i++)
		free(mat->nzero_index[i]);

	free(mat->nzero_index);

	//free(mat);
	if(mat->element[0] != NULL)
		free(mat->element[0]);
	if(mat->element[1] != NULL)
		free(mat->element[1]);
	if(mat->element[2] != NULL)
		free(mat->element[2]);

	free(mat);
}

/* set nzero_row_dim automatically */
void set_nzero_row_dim_td(TDRSMatrix mat)
{
	long int i, j;

	/* all clear */
	for(i = 0; i < mat->col_dim; i++)
		mat->nzero_row_dim[i] = 0;

	/* search */
	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
			mat->nzero_row_dim[*(mat->nzero_index[i] + j)]++;
	}
}

/* Print TDRSMatrix */
void print_tdrsmatrix(TDRSMatrix mat)
{
	long int i, j, total_index;
	tdfloat tmp;

	if(mat == NULL)
		fprintf(stderr, "ERROR!\n");

	total_index = 0;
	for(i = 0; i < mat->row_dim; i++)
	{
		// Fix!: 2011-08-29 by T.Kouya
		if(mat->nzero_col_dim[i] >= 1)
		{
			printf("%5ld: ", i);
			for(j = 0; j < mat->real_nzero_col_dim[i]; j++)
			{
				// printf("%ld->%f, ", mat->nzero_index[i][j], mat->element[total_index++]);
				if(j < mat->nzero_col_dim[i])
				{
					printf("%ld-> ", mat->nzero_index[i][j]);
					//mpf_out_str(stdout, 10, 0, mat->element[total_index++]);
					tmp.val[0] = mat->element[0][total_index + j];
					tmp.val[1] = mat->element[1][total_index + j];
					tmp.val[2] = mat->element[2][total_index + j];
					rtd_out_str(tmp.val);
					if(j != (mat->nzero_col_dim[i] - 1)) printf(", ");
				}
				total_index++;
			}
			printf("\n");
		}
		total_index += mat->real_nzero_col_dim[i];
	}
	return;
}

/* initialize and substitute TDRSMatrix from MPFMatrix */
TDRSMatrix init_set_tdrsmatrix_tdmatrix(TDMatrix org_mat)
{
	long int i, j;
	long int nzero_total_num, total_index, j_index;
	long int *ptr_nzero_col_dim;
	TDRSMatrix ret;
	tdfloat tmp;

	/* initialize variables */
	nzero_total_num = 0;

	ptr_nzero_col_dim = (long int *)malloc((size_t)(sizeof(long int) * (org_mat->row_dim)));

	/* initialize num_col as numbers of nonzero element in each row */
	if(ptr_nzero_col_dim == NULL)
	{
		fprintf(stderr, "Cannot allocate num_col(size %ld)\n", sizeof(long int) * (org_mat->row_dim));
		return NULL;
	}

	/* Read org_mat (1st) */
	for(i = 0; i < org_mat->row_dim; i++)
	{
		*(ptr_nzero_col_dim + i) = 0;
		for(j = 0; j < org_mat->col_dim; j++)
		{
			//if(mpf_cmp_ui(get_mpfmatrix_ij(org_mat, i, j), 0UL) != 0)
			if(rtd_cmp_ui(get_tdmatrix_ij(org_mat, i, j), 0UL) != 0)
			{
				/* get nzero_row_dim, nzero_col_dim, nzero_total_num */
				nzero_total_num++;
				*(ptr_nzero_col_dim + i) += 1;
			}
		}
	}

	// initialize
	ret = init_tdrsmatrix(org_mat->row_dim, ptr_nzero_col_dim, nzero_total_num); // , org_mat->prec);

	/* Read org_mat (2nd) */
	total_index = 0;
	for(i = 0; i < ret->row_dim; i++)
	{
		j_index = 0;
		for(j = 0; j < ret->col_dim; j++)
		{
			//if(mpf_cmp_ui(get_mpfmatrix_ij(org_mat, i, j), 0UL) != 0)
			if(rtd_cmp_ui(get_tdmatrix_ij(org_mat, i, j), 0UL) != 0)
			{
				ret->nzero_index[i][j_index] =  j;
				//mpf_set(*(ret->element + total_index), get_mpfmatrix_ij(org_mat, i, j));
				rtd_set(tmp.val, get_tdmatrix_ij(org_mat, i, j));
				ret->element[0][total_index + j_index] = tmp.val[0];
				ret->element[1][total_index + j_index] = tmp.val[1];
				ret->element[2][total_index + j_index] = tmp.val[2];

				//total_index += 1;
				j_index += 1;
			}
		}
		total_index += ret->real_nzero_col_dim[i];
	}

	free(ptr_nzero_col_dim);

	return ret;
}

// 2024-08-01(Thu) T.Kouya
/* get the DRSMatrix ij-element */
void get_tdrsmatrix_ij(double ret[TDSIZE], TDRSMatrix mat, long int row_index, long int col_index)
{
	long int i, j, total_index;

	if((row_index < 0) || (row_index >= mat->row_dim) || (col_index < 0) || (col_index >= mat->col_dim))
	{
		fprintf(stderr, "Warning: row_index(%ld) or col_index(%ld) is illegal!\n", row_index, col_index);
		rtd_set0(ret);
		//return mat->zero_element;
		return;
	}

	// finding mat_ij element
	if(mat->nzero_col_dim[row_index] >= 1)
	{
		total_index = 0;
		for(i = 0; i < row_index; i++)
			total_index += mat->real_nzero_col_dim[i];
			//total_index += mat->nzero_col_dim[i];

		for(j = 0; j < mat->nzero_col_dim[row_index] ; j++)
		{
			// Find!
			if(mat->nzero_index[row_index][j] == col_index)
			{
				//return mat->element[total_index];
				ret[0] = mat->element[0][total_index];
				ret[1] = mat->element[1][total_index];
				ret[2] = mat->element[2][total_index];

				return;
			}
			
			total_index++;
		}
	}

	// Not found -> return 0
	//return mat->zero_element;
	rtd_set0(ret);
	return;
}

// 2024-08-04 (Sun) T.Kouya
/* set the TDRSMatrix ij-element */
void set_tdrsmatrix_ij(TDRSMatrix mat, long int row_index, long int col_index, double val[TDSIZE])
{
	long int i, j, total_index;

	if((row_index < 0) || (row_index >= mat->row_dim) || (col_index < 0) || (col_index >= mat->col_dim))
	{
		fprintf(stderr, "Warning: row_index(%ld) or col_index(%ld) is illegal!\n", row_index, col_index);
		//return mat->zero_element;
		return;
	}

	// finding mat_ij element
	if(mat->nzero_col_dim[row_index] >= 1)
	{
		total_index = 0;
		for(i = 0; i < row_index; i++)
			total_index += mat->real_nzero_col_dim[i];
			//total_index += mat->nzero_col_dim[i];

		for(j = 0; j < mat->nzero_col_dim[row_index] ; j++)
		{
			// Find!
			if(mat->nzero_index[row_index][j] == col_index)
			{
				//return mat->element[total_index];
				//ret[0] = mat->element[0][total_index];
				//ret[1] = mat->element[1][total_index];
				mat->element[0][total_index] = val[0];
				mat->element[1][total_index] = val[1];
				mat->element[2][total_index] = val[2];

				return;
			}
			
			total_index++;
		}
	}

	return;
}

// 2024-07-30(Tue) T.Kouya
// init and set TDRSMatrix
TDRSMatrix init_set_tdrsmatrix(TDRSMatrix org_sp)
{
	TDRSMatrix ret;
	double ddtmp[TDSIZE];
	long int org_sp_total_index, total_index, i, j;

    //printf("row_dim, nzero_total_num = %ld, %ld\n", org_sp->row_dim, org_sp->nzero_total_num);
    ret = init_tdrsmatrix(org_sp->row_dim, org_sp->nzero_col_dim, org_sp->nzero_total_num);
    //printf("init_tdrsmatrix!\n");

	 // Real total number of non-zero elements
	ret->real_nzero_total_num = org_sp->real_nzero_total_num;

    // copy nzero_index
	for(i = 0; i < org_sp->row_dim; i++)
	{
		// Real numbers of non-zero elements in i-th row
		ret->real_nzero_col_dim[i] = org_sp->real_nzero_col_dim[i];

		for(j = 0; j < org_sp->nzero_col_dim[i]; j++)
			ret->nzero_index[i][j] = org_sp->nzero_index[i][j];
	}

	// copy values
    for(total_index = 0; total_index < org_sp->real_nzero_total_num; total_index++)
	{
		ret->element[0][total_index] = org_sp->element[0][total_index];
		ret->element[1][total_index] = org_sp->element[1][total_index];
		ret->element[2][total_index] = org_sp->element[2][total_index];
	}

	return ret;
}

// 2024-08-01(Thu) T.Kouya
// spmat := 0
void set0_tdrsmatrix(TDRSMatrix spmat)
{
	long int index;

	// substitution
	for(index = 0; index < spmat->real_nzero_total_num; index++)
	{
		spmat->element[0][index] = (double)0.0;
		spmat->element[1][index] = (double)0.0;
		spmat->element[2][index] = (double)0.0;
	}

	return;
}

// 2024-12-03(Tue) T.Kouya
// ret := spmat_a
void subst_tdrsmatrix(TDRSMatrix ret, TDRSMatrix spmat_a)
{
	long int i, j, total_index;
/* <--> element = [a b c d e f g h i] */
/*      row_dim = 5, col_dim = 5 */
/*      nzero_index[0] = [0 2 3] */
/*      nzero_index[1] = [4]     */
/*      nzero_index[2] = [1 2]   */
/*      nzero_index[3] = [3]     */
/*      nzero_index[4] = [2 4]   */
/*      nzero_col_dim[0] = 3 */
/*      nzero_col_dim[1] = 1 */
/*      nzero_col_dim[2] = 2 */
/*      nzero_col_dim[3] = 1 */
/*      nzero_col_dim[4] = 2 */
/*      nzero_total_num = 9  */
/*      zero_element = 0.0   */

	// check parameters
	if(ret->nzero_total_num != spmat_a->nzero_total_num)
	{
		fprintf(stderr, "ERROR(subst_tdrsmatrix): mismatched nzero_total_nums! (%ld != %ld)\n", ret->nzero_total_num, spmat_a->nzero_total_num);
		return; // -1;
	}
	if((ret->row_dim != spmat_a->row_dim) || (ret->col_dim != spmat_a->col_dim))
	{
		fprintf(stderr, "ERROR(subst_tdrsmatrix): mismatched dimensions! ((%ld, %ld) != (%ld, %ld))\n", ret->row_dim, ret->col_dim, spmat_a->row_dim, spmat_a->col_dim);
		return; // -2;
	}

	// Real total number of non-zero elements
	ret->real_nzero_total_num = spmat_a->real_nzero_total_num;

    // copy nzero_index
	for(i = 0; i < spmat_a->row_dim; i++)
	{
		// Real numbers of non-zero elements in i-th row
		ret->real_nzero_col_dim[i] = spmat_a->real_nzero_col_dim[i];

		for(j = 0; j < spmat_a->nzero_col_dim[i]; j++)
			ret->nzero_index[i][j] = spmat_a->nzero_index[i][j];
	}

	// substitution
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	for(total_index = 0; total_index < ret->real_nzero_total_num; total_index += _BNC_D_WIDTH)
	{
		_mm256_store_pd(&(ret->element[0][total_index]), _mm256_load_pd(&(spmat_a->element[0][total_index])));
		_mm256_store_pd(&(ret->element[1][total_index]), _mm256_load_pd(&(spmat_a->element[1][total_index])));
		_mm256_store_pd(&(ret->element[2][total_index]), _mm256_load_pd(&(spmat_a->element[2][total_index])));
	}

#elif defined(__AVX512F__) // __AVX512F__
	for(total_index = 0; total_index < ret->real_nzero_total_num; total_index += _BNC_D_WIDTH)
	{
		_mm512_store_pd(&(ret->element[0][total_index]), _mm512_load_pd(&(spmat_a->element[0][total_index])));
		_mm512_store_pd(&(ret->element[1][total_index]), _mm512_load_pd(&(spmat_a->element[1][total_index])));
		_mm512_store_pd(&(ret->element[2][total_index]), _mm512_load_pd(&(spmat_a->element[2][total_index])));
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2
	{
		long _vl = (long)svcntd();
		for(total_index = 0; total_index < ret->real_nzero_total_num; total_index += _vl)
		{
			svbool_t pg = svwhilelt_b64_s64((int64_t)total_index,
			                                (int64_t)ret->real_nzero_total_num);
			svst1_f64(pg, &(ret->element[0][total_index]),
			          svld1_f64(pg, &(spmat_a->element[0][total_index])));
			svst1_f64(pg, &(ret->element[1][total_index]),
			          svld1_f64(pg, &(spmat_a->element[1][total_index])));
			svst1_f64(pg, &(ret->element[2][total_index]),
			          svld1_f64(pg, &(spmat_a->element[2][total_index])));
		}
	}
#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM Neon
	for(total_index = 0; total_index < ret->real_nzero_total_num; total_index += 2)
	{
		vst1q_f64(&(ret->element[0][total_index]), vld1q_f64(&(spmat_a->element[0][total_index])));
		vst1q_f64(&(ret->element[1][total_index]), vld1q_f64(&(spmat_a->element[1][total_index])));
		vst1q_f64(&(ret->element[2][total_index]), vld1q_f64(&(spmat_a->element[2][total_index])));
	}



#else // others
	//memcpy(ret->element, spmat_a->element, sizeof(double) * spmat_a->nzero_total_num);
	//for(total_index = 0; total_index < ret->nzero_total_num; total_index++)
	for(total_index = 0; total_index < ret->real_nzero_total_num; total_index++)
	{
		ret->element[0][total_index] = spmat_a->element[0][total_index];
		ret->element[1][total_index] = spmat_a->element[1][total_index];
		ret->element[2][total_index] = spmat_a->element[2][total_index];
	}

#endif // __AVX2__

}

// 2024-08-01 T.Kouya
// init and set DRSMatrix from TDRSMatrix
DRSMatrix init_set_drsmatrix_tdrsmat(TDRSMatrix org_sp)
{
	DRSMatrix ret;
	double ddtmp[TDSIZE];
	long int org_sp_total_index, total_index, i, j;

    //printf("row_dim, nzero_total_num = %ld, %ld\n", org_sp->row_dim, org_sp->nzero_total_num);
    ret = init_drsmatrix(org_sp->row_dim, org_sp->nzero_col_dim, org_sp->nzero_total_num);
    //printf("init_tdrsmatrix!\n");

	// Real total number of non-zero elements
	ret->real_nzero_total_num = org_sp->real_nzero_total_num;

    // copy nzero_index
	for(i = 0; i < org_sp->row_dim; i++)
	{
		// Real numbers of non-zero elements in i-th row
		ret->real_nzero_col_dim[i] = org_sp->real_nzero_col_dim[i];

		for(j = 0; j < org_sp->nzero_col_dim[i]; j++)
			ret->nzero_index[i][j] = org_sp->nzero_index[i][j];
	}

	// copy values
    for(total_index = 0; total_index < org_sp->real_nzero_total_num; total_index++)
	{
		ret->element[total_index] = org_sp->element[0][total_index];
		//ret->element[1][total_index] = org_sp->element[1][total_index];
	}

	return ret;
}

#ifdef USE_GMP
// MPFRSMatrix -> TDRSMatrix
TDRSMatrix init_set_tdrsmatrix_mpfrsmatrix(MPFRSMatrix org_sp)
{
	TDRSMatrix ret;
	double tdtmp[TDSIZE];
	long int org_sp_total_index, total_index, i, j;

    //printf("row_dim, nzero_total_num = %ld, %ld\n", mpfa_sp->row_dim, mpfa_sp->nzero_total_num);
    ret = init_tdrsmatrix(org_sp->row_dim, org_sp->nzero_col_dim, org_sp->nzero_total_num);
    //printf("init_tdrsmatrix!\n");

    // MPFR -> TD
    total_index = 0;
	org_sp_total_index = 0;
	for(i = 0; i < ret->row_dim; i++)
	{
		for(j = 0; j < ret->nzero_col_dim[i]; j++)
		{
            mpf_get_td(tdtmp, org_sp->element[org_sp_total_index + j]);
            ret->element[0][total_index + j] = tdtmp[0];
            ret->element[1][total_index + j] = tdtmp[1];
            ret->element[2][total_index + j] = tdtmp[2];

            ret->nzero_index[i][j] = org_sp->nzero_index[i][j];
			//printf("%ld %ld -> %ld\n", i, j, org_sp->nzero_index[i][j]);
			//total_index++;
		}
		total_index += ret->real_nzero_col_dim[i];
		org_sp_total_index += org_sp->nzero_col_dim[i];
	}
    //print_tdrsmatrix(ret);

	return ret;
}
#endif // USE_GMP

/* Get variables to initialize TDRSMatrix */
int get_vars_tdrsmatrix_fname(long int *ptr_row_dim, long int **ptr_nzero_col_dim, long int *ptr_nzero_total_num, const char *fname)
{
	FILE *fp;
	static char line_buf[LINE_BUF_LEN];
	long int i, j;
	long int row_index, col_index, max_row_index, max_col_index, nzero_total_num;
	long int ele_index;

	/* file open */
	fp = fopen(fname, "r");
	if(fp == NULL)
	{
		fprintf(stderr, "Cannot open %s\n", fname);
		return ERROR;
	}

	/* initialize variables */
	nzero_total_num = 0;
	max_row_index = 0;
	max_col_index = 0;

	/* Read file (1st) */
	while(fgets(line_buf, LINE_BUF_LEN - 1, fp) != NULL)
	{
		/* get row_index and col_index */
		sscanf(line_buf, "%ld %ld", &row_index, &col_index);

		/* get nzero_row_dim, nzero_col_dim, nzero_total_num */
		nzero_total_num++;
		//row_index--; col_index--;
		if(max_row_index < row_index)
			max_row_index = row_index;
		if(max_col_index < col_index)
			max_col_index = col_index;
	}

//	printf("Total URIs: %ld\n", max_row_index + 1);

	/* initialize num_col as numbers of nonzero element in each row */
	*ptr_row_dim = max_row_index + 1;
	*ptr_nzero_total_num = nzero_total_num;
	*ptr_nzero_col_dim = (long int *)malloc((size_t)(sizeof(long int) * (max_row_index + 1)));
	if(*ptr_nzero_col_dim == NULL)
	{
		fprintf(stderr, "Cannot allocate num_col(size %ld)\n", sizeof(long int) * (max_row_index + 1));
		return ERROR;
	}

	/* Read file (2nd) */
	for(i = 0; i <= max_row_index; i++)
		*(*ptr_nzero_col_dim + i) = 0;

	rewind(fp);
	while(fgets(line_buf, LINE_BUF_LEN - 1, fp) != NULL)
	{
		/* get row_index, col_index, element */
		sscanf(line_buf, "%ld %ld", &row_index, &col_index);
		//fprintf(stderr, "%ld %ld\n", row_index--, col_index--);
		*(*ptr_nzero_col_dim + row_index) += 1;
	}

	fclose(fp);

	return SUCCESS;
}

// 2025-07-17(Thu)
#ifdef USE_IMKL
// ret := (sparse_matrix_t)mat
void convert_indeces_tdrsmatrix_mkl_csrmat(MKL_INT *ret_i_csr_start, MKL_INT *ret_i_csr_end, MKL_INT *ret_j_csr, TDRSMatrix mat)
{
	long int i, j, total_index;
    //int *i_mat_csr, *j_mat_csr, row_dim = (int)mat->row_dim;
    int row_dim = (int)mat->row_dim;

	// convert our CSR to intel math kernel csr format
	//i_mat_csr = (int *)calloc(mat->row_dim + 1, sizeof(int));
	//j_mat_csr = (int *)calloc(mat->real_nzero_total_num, sizeof(int));
	ret_i_csr_start = (MKL_INT *)calloc(mat->row_dim, sizeof(MKL_INT));
	ret_i_csr_end = (MKL_INT *)calloc(mat->row_dim, sizeof(MKL_INT));
	ret_j_csr = (MKL_INT *)calloc(mat->real_nzero_total_num, sizeof(MKL_INT));

    total_index = 0;
	for(i = 0; i < row_dim; i++)
	{
		ret_i_csr_start[i] = (MKL_INT)total_index;
		//if(a->nzero_col_dim[i] >= 1)
		if(mat->real_nzero_col_dim[i] >= 1)
		{
			for(j = 0; j < mat->nzero_col_dim[i]; j++)
			{
				ret_j_csr[total_index] = mat->nzero_index[i][j];
				total_index++;
			}
    		//ret_i_csr_end[i] = (MKL_INT)total_index - 1;
			// embed gap among nzero_col_dim and real_nzero_col_dim
			for(j = mat->nzero_col_dim[i]; j < mat->real_nzero_col_dim[i]; j++)
			{
				ret_j_csr[total_index] = mat->nzero_index[i][mat->nzero_col_dim[i] - 1] + (j + 1 - mat->nzero_col_dim[i]);
				total_index++;
			}
		}
		ret_i_csr_end[i] = (MKL_INT)total_index;
        //ret_i_csr_end[i] = ret_i_csr_start[i] + mat->real_nzero_col_dim[i];
    }
}
// ret := (sparse_matrix_t)mat
sparse_status_t convert_tdrsmatrix_mkl_csrmat(sparse_matrix_t *ret[TDSIZE], MKL_INT *ret_i_csr_start, MKL_INT *ret_i_csr_end, MKL_INT *ret_j_csr, TDRSMatrix mat)
{
	//long int i, j, total_index;
    //int *i_mat_csr, *j_mat_csr, row_dim = (int)mat->row_dim;
    //int row_dim = (int)mat->row_dim;
    sparse_status_t ret_mkl[TDSIZE];

#if 0
	// convert our CSR to intel math kernel csr format
	//i_mat_csr = (int *)calloc(mat->row_dim + 1, sizeof(int));
	//j_mat_csr = (int *)calloc(mat->real_nzero_total_num, sizeof(int));
	ret_i_csr_start = (MKL_INT *)calloc(mat->row_dim, sizeof(MKL_INT));
	ret_i_csr_end = (MKL_INT *)calloc(mat->row_dim, sizeof(MKL_INT));
	ret_j_csr = (MKL_INT *)calloc(mat->real_nzero_total_num, sizeof(MKL_INT));

    total_index = 0;
	for(i = 0; i < row_dim; i++)
	{
		ret_i_csr_start[i] = (MKL_INT)total_index;
		//if(a->nzero_col_dim[i] >= 1)
		if(mat->real_nzero_col_dim[i] >= 1)
		{
			for(j = 0; j < mat->nzero_col_dim[i]; j++)
			{
				ret_j_csr[total_index] = mat->nzero_index[i][j];
				total_index++;
			}
    		//ret_i_csr_end[i] = (MKL_INT)total_index - 1;
			// embed gap among nzero_col_dim and real_nzero_col_dim
			for(j = mat->nzero_col_dim[i]; j < mat->real_nzero_col_dim[i]; j++)
			{
				ret_j_csr[total_index] = mat->nzero_index[i][mat->nzero_col_dim[i] - 1] + (j + 1 - mat->nzero_col_dim[i]);
				total_index++;
			}
		}
		ret_i_csr_end[i] = (MKL_INT)total_index;
        //ret_i_csr_end[i] = ret_i_csr_start[i] + mat->real_nzero_col_dim[i];
    }
#endif // 0
	convert_indeces_tdrsmatrix_mkl_csrmat(ret_i_csr_start, ret_i_csr_end, ret_j_csr, mat);

	//i_mat_csr_end[row_dim] = mat->real_nzero_total_num;
	//mkl_cspblas_dcsrgemv("N", &row_dim, mat->element, i_mat_csr, j_mat_csr, vec->element, ret->element);
    ret_mkl[0] = mkl_sparse_d_create_csr(
        ret[0],
        SPARSE_INDEX_BASE_ZERO,
        (MKL_INT)mat->row_dim,
        (MKL_INT)mat->col_dim,
        ret_i_csr_start,
        ret_i_csr_end,
        ret_j_csr,
        mat->element[0]
    );
    ret_mkl[1] = mkl_sparse_d_create_csr(
        ret[1],
        SPARSE_INDEX_BASE_ZERO,
        (MKL_INT)mat->row_dim,
        (MKL_INT)mat->col_dim,
        ret_i_csr_start,
        ret_i_csr_end,
        ret_j_csr,
        mat->element[1]
    );
    ret_mkl[2] = mkl_sparse_d_create_csr(
        ret[2],
        SPARSE_INDEX_BASE_ZERO,
        (MKL_INT)mat->row_dim,
        (MKL_INT)mat->col_dim,
        ret_i_csr_start,
        ret_i_csr_end,
        ret_j_csr,
        mat->element[2]
    );
	//free(i_mat_csr); free(j_mat_csr);
    return (ret_mkl[0] & ret_mkl[1] & ret_mkl[2]);
}
// ret := (sparse_matrix_t)mat
sparse_status_t subst_tdrsmatrix_mkl_csrmat(sparse_matrix_t *ret[TDSIZE], MKL_INT *i_csr_start, MKL_INT *i_csr_end, MKL_INT *j_csr, TDRSMatrix mat)
{
	//long int i, j, total_index;
    //int *i_mat_csr, *j_mat_csr, row_dim = (int)mat->row_dim;
    //int row_dim = (int)mat->row_dim;
    sparse_status_t ret_mkl[TDSIZE];

	//i_mat_csr_end[row_dim] = mat->real_nzero_total_num;
	//mkl_cspblas_dcsrgemv("N", &row_dim, mat->element, i_mat_csr, j_mat_csr, vec->element, ret->element);
    ret_mkl[0] = mkl_sparse_d_create_csr(
        ret[0],
        SPARSE_INDEX_BASE_ZERO,
        (MKL_INT)mat->row_dim,
        (MKL_INT)mat->col_dim,
        i_csr_start,
        i_csr_end,
        j_csr,
        mat->element[0]
    );
    ret_mkl[1] = mkl_sparse_d_create_csr(
        ret[1],
        SPARSE_INDEX_BASE_ZERO,
        (MKL_INT)mat->row_dim,
        (MKL_INT)mat->col_dim,
        i_csr_start,
        i_csr_end,
        j_csr,
        mat->element[1]
    );
    ret_mkl[2] = mkl_sparse_d_create_csr(
        ret[2],
        SPARSE_INDEX_BASE_ZERO,
        (MKL_INT)mat->row_dim,
        (MKL_INT)mat->col_dim,
        i_csr_start,
        i_csr_end,
        j_csr,
        mat->element[2]
    );

	//free(i_mat_csr); free(j_mat_csr);
    return (ret_mkl[0] & ret_mkl[1] & ret_mkl[2]);
}
#endif // USE_IMKL

/* Multiply TDRSMatrix * TDVector */
int mul_tdrsmatrix_tdvec(TDVector ret, TDRSMatrix mat, TDVector vec)
{
	long int i, j, total_index;
	//tdfloat tmp, mat_ij, vec_j, ret_i;
	double tmp[TDSIZE], mat_ij[TDSIZE], vec_j[TDSIZE], ret_i[TDSIZE];

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[TDSIZE], vb4[TDSIZE], tmp4[TDSIZE], tmp4_mul[TDSIZE];
	long int jmax, jres, col_dim;
	double mat_vec_i[TDSIZE], vset[_BNC_D_WIDTH];

	for(i = 0; i < mat->row_dim; i++)
	{
		//ret->element[i] = 0.0;
		//tmp4 = _mm256_setzero_pd();
		_bncavx2_set0_td(tmp4);

		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		jmax = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;
		//printf("%ld jmax, jres = %ld, %ld\n", i, jmax, jres);
		for(j = 0; j < jmax; j += _BNC_D_WIDTH)
		{
			//printf("load "); fflush(stdout);
			//ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			a4[0]  = _mm256_load_pd(&(mat->element[0][total_index]));
			a4[1]  = _mm256_load_pd(&(mat->element[1][total_index]));
			a4[2]  = _mm256_load_pd(&(mat->element[2][total_index]));

			vset[0] = 0.0; vset[1] = 0.0; vset[2] = 0.0; vset[3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[0] = vec->element[0][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[1] = vec->element[0][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[2] = vec->element[0][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[3] = vec->element[0][mat->nzero_index[i][j + 3]];
			//printf("set0 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			vb4[0] = _mm256_set_pd(
				vset[3], // vec->element[0][mat->nzero_index[i][j + 3]],
				vset[2], // vec->element[0][mat->nzero_index[i][j + 2]],
				vset[1], // vec->element[0][mat->nzero_index[i][j + 1]],
				vset[0]  // vec->element[0][mat->nzero_index[i][j    ]]
			);

			//printf("set1 ");
			vset[0] = 0.0; vset[1] = 0.0; vset[2] = 0.0; vset[3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[0] = vec->element[1][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[1] = vec->element[1][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[2] = vec->element[1][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[3] = vec->element[1][mat->nzero_index[i][j + 3]];
			//printf("set1 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			vb4[1] = _mm256_set_pd(
				vset[3], // vec->element[1][mat->nzero_index[i][j + 3]],
				vset[2], // vec->element[1][mat->nzero_index[i][j + 2]],
				vset[1], // vec->element[1][mat->nzero_index[i][j + 1]],
				vset[0]  // vec->element[1][mat->nzero_index[i][j    ]]
			);

			//printf("set2 ");
			vset[0] = 0.0; vset[1] = 0.0; vset[2] = 0.0; vset[3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[0] = vec->element[2][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[1] = vec->element[2][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[2] = vec->element[2][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[3] = vec->element[2][mat->nzero_index[i][j + 3]];
			//printf("set1 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			vb4[2] = _mm256_set_pd(
				vset[3], // vec->element[1][mat->nzero_index[i][j + 3]],
				vset[2], // vec->element[1][mat->nzero_index[i][j + 2]],
				vset[1], // vec->element[1][mat->nzero_index[i][j + 1]],
				vset[0]  // vec->element[1][mat->nzero_index[i][j    ]]
			);

			//printf("mul_add "); fflush(stdout);
			//tmp4 = _mm256_fmadd_pd(a4, vb4, tmp4);
			_bncavx2_rtd_mul(tmp4_mul, a4, vb4);
			_bncavx2_rtd_add(tmp4, tmp4, tmp4_mul);

			// total_index++;
			total_index += _BNC_D_WIDTH;
			//printf("%ld ", j);
		}
		//total_index += mat->real_nzero_col_dim[i];
		//printf("\n");
		//mat_vec_i = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3];
		_bncavx2_rtd_sum256d(mat_vec_i, tmp4);

		//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i[0], mat->nzero_col_dim[i], jmax, jres);
/*
		col_dim = mat->nzero_col_dim[i];
		switch(jres)
		{
			case 1:
				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 1]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 1]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 1]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				break; 

			case 2:
				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 2]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 2]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 2]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;	
			
				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 1]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 1]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 1]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				break; 

			case 3:
				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 3]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 3]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 3]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 3]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;	

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 2]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 2]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 2]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;	

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 1]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 1]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 1]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				break; 
		}
*/
		//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i[0]);
		set_tdvector_i(ret, i, mat_vec_i);
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[TDSIZE], vb8[TDSIZE], tmp8[TDSIZE], tmp8_mul[TDSIZE];
	long int jmax, jres, col_dim;
	double mat_vec_i[TDSIZE];

	for(i = 0; i < mat->row_dim; i++)
	{
		//ret->element[i] = 0.0;]
		//tmp8 = _mm512_setzero_pd();
		_bncavx512_set0_td(tmp8);

		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		jmax = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;
		{
		long int jfull = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		long int *_nzi = mat->nzero_index[i];
		/* full chunks: every lane valid -> branch-free direct gather */
		for(j = 0; j < jfull; j += _BNC_D_WIDTH)
		{
			a8[0] = _mm512_load_pd(&(mat->element[0][total_index]));
			a8[1] = _mm512_load_pd(&(mat->element[1][total_index]));
			a8[2] = _mm512_load_pd(&(mat->element[2][total_index]));
			vb8[0] = _mm512_set_pd(vec->element[0][_nzi[j+7]], vec->element[0][_nzi[j+6]], vec->element[0][_nzi[j+5]], vec->element[0][_nzi[j+4]], vec->element[0][_nzi[j+3]], vec->element[0][_nzi[j+2]], vec->element[0][_nzi[j+1]], vec->element[0][_nzi[j  ]]);
			vb8[1] = _mm512_set_pd(vec->element[1][_nzi[j+7]], vec->element[1][_nzi[j+6]], vec->element[1][_nzi[j+5]], vec->element[1][_nzi[j+4]], vec->element[1][_nzi[j+3]], vec->element[1][_nzi[j+2]], vec->element[1][_nzi[j+1]], vec->element[1][_nzi[j  ]]);
			vb8[2] = _mm512_set_pd(vec->element[2][_nzi[j+7]], vec->element[2][_nzi[j+6]], vec->element[2][_nzi[j+5]], vec->element[2][_nzi[j+4]], vec->element[2][_nzi[j+3]], vec->element[2][_nzi[j+2]], vec->element[2][_nzi[j+1]], vec->element[2][_nzi[j  ]]);
			_bncavx512_rtd_mul(tmp8_mul, a8, vb8);
			_bncavx512_rtd_add(tmp8, tmp8, tmp8_mul);
			total_index += _BNC_D_WIDTH;
		}
		/* tail: <=1 partial chunk, bounds-guarded (padded lanes -> 0) */
		if(jfull < jmax)
		{
			long int ncd = mat->nzero_col_dim[i]; int _qg;
			a8[0] = _mm512_load_pd(&(mat->element[0][total_index]));
			a8[1] = _mm512_load_pd(&(mat->element[1][total_index]));
			a8[2] = _mm512_load_pd(&(mat->element[2][total_index]));
			{ double _vsg[8]={0,0,0,0,0,0,0,0};
			  for(_qg=0;_qg<_BNC_D_WIDTH;_qg++) if((jfull+_qg)<ncd) _vsg[_qg]=vec->element[0][_nzi[jfull+_qg]];
			  vb8[0] = _mm512_set_pd(_vsg[7],_vsg[6],_vsg[5],_vsg[4],_vsg[3],_vsg[2],_vsg[1],_vsg[0]); }
			{ double _vsg[8]={0,0,0,0,0,0,0,0};
			  for(_qg=0;_qg<_BNC_D_WIDTH;_qg++) if((jfull+_qg)<ncd) _vsg[_qg]=vec->element[1][_nzi[jfull+_qg]];
			  vb8[1] = _mm512_set_pd(_vsg[7],_vsg[6],_vsg[5],_vsg[4],_vsg[3],_vsg[2],_vsg[1],_vsg[0]); }
			{ double _vsg[8]={0,0,0,0,0,0,0,0};
			  for(_qg=0;_qg<_BNC_D_WIDTH;_qg++) if((jfull+_qg)<ncd) _vsg[_qg]=vec->element[2][_nzi[jfull+_qg]];
			  vb8[2] = _mm512_set_pd(_vsg[7],_vsg[6],_vsg[5],_vsg[4],_vsg[3],_vsg[2],_vsg[1],_vsg[0]); }
			_bncavx512_rtd_mul(tmp8_mul, a8, vb8);
			_bncavx512_rtd_add(tmp8, tmp8, tmp8_mul);
			total_index += _BNC_D_WIDTH;
		}
		}
		//mat_vec_i = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3] + tmp4[4] + tmp4[5] + tmp4[6] + tmp4[7];
		_bncavx512_rtd_sum512d(mat_vec_i, tmp8);

/*
		col_dim = mat->nzero_col_dim[i];
		switch(jres)
		{
			case 1: 
				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_col_dim[i] - 1];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 1]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 1]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 1]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;
				break; 

			case 2:
				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 2]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 2]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 2]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 1]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 1]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 1]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;
				break; 

			case 3:
				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 3]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 3]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 3]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 3]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 2]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 2]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 2]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 1]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 1]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 1]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;
				break; 

			case 4:
				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 4]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 4]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 4]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 4]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 3]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 3]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 3]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 3]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 2]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 2]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 2]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 1]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 1]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 1]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;
				break;

			case 5:
				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 5]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 5]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 5]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 5]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 4]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 4]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 4]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 4]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 3]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 3]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 3]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 3]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 2]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 2]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 2]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 1]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 1]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 1]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;
				break;

			case 6:
				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 6]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 6]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 6]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 6]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;
				
				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 5]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 5]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 5]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 5]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 4]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 4]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 4]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 4]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 3]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 3]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 3]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 3]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 2]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 2]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 2]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 1]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 1]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 1]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;
				break;

			case 7:
				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 7]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 7]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 7]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 7]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 6]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 6]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 6]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 6]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 5]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 5]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 5]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 5]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 4]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 4]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 4]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 4]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 3]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 3]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 3]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 3]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 2]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 2]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 2]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[2] = mat->element[2][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 1]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 1]];
				vec_j[2] = vec->element[2][mat->nzero_index[i][col_dim - 1]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;
				break;
		}
*/
		set_tdvector_i(ret, i, mat_vec_i);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2
	/* SVE2 TD SpMV: hardware gather for vec->element[0..2][nzero_index[i][j+k]],
	 * triple-double FMA per VL via _bncsve2_rtd_mul/add. */
	{
		long _vl = (long)svcntd();
		long int jmax, jres;
		double mat_vec_i[TDSIZE];
		for(i = 0; i < mat->row_dim; i++)
		{
			svfloat64_t t0, t1, t2;
			_bncsve2_rtd_set0(&t0, &t1, &t2);
			long int nz = mat->nzero_col_dim[i];
			jmax = (nz / _vl) * _vl;
			jres = nz - jmax;
			for(j = 0; j < jmax; j += _vl)
			{
				svbool_t pg = svptrue_b64();
				svfloat64_t a0 = svld1_f64(pg, &(mat->element[0][total_index]));
				svfloat64_t a1 = svld1_f64(pg, &(mat->element[1][total_index]));
				svfloat64_t a2v= svld1_f64(pg, &(mat->element[2][total_index]));
				svint64_t idx  = svld1_s64(pg, (int64_t*)&(mat->nzero_index[i][j]));
				svfloat64_t b0 = svld1_gather_s64index_f64(pg, vec->element[0], idx);
				svfloat64_t b1 = svld1_gather_s64index_f64(pg, vec->element[1], idx);
				svfloat64_t b2 = svld1_gather_s64index_f64(pg, vec->element[2], idx);
				svfloat64_t m0, m1, m2;
				_bncsve2_rtd_mul(pg, &m0, &m1, &m2, a0, a1, a2v, b0, b1, b2);
				_bncsve2_rtd_add(pg, &t0, &t1, &t2, t0, t1, t2, m0, m1, m2);
				total_index += _vl;
			}
			{
				long _L, _vl = (long)svcntd();
				double _la0[64], _la1[64], _la2[64];
				svst1_f64(svptrue_b64(), _la0, t0);
				svst1_f64(svptrue_b64(), _la1, t1);
				svst1_f64(svptrue_b64(), _la2, t2);
				rtd_set_ui(mat_vec_i, 0UL);
				for(_L = 0; _L < _vl; _L++)
				{
					rtd_add_d(mat_vec_i, mat_vec_i, _la0[_L]);
					rtd_add_d(mat_vec_i, mat_vec_i, _la1[_L]);
					rtd_add_d(mat_vec_i, mat_vec_i, _la2[_L]);
				}
			}
			/* scalar tail */
			{
				double mij[TDSIZE], vj[TDSIZE], pr[TDSIZE];
				long int col_dim = mat->nzero_col_dim[i];
				for(j = 0; j < jres; j++)
				{
					long int idx_t = mat->nzero_index[i][col_dim - jres + j];
					mij[0] = mat->element[0][total_index];
					mij[1] = mat->element[1][total_index];
					mij[2] = mat->element[2][total_index];
					vj[0]  = vec->element[0][idx_t];
					vj[1]  = vec->element[1][idx_t];
					vj[2]  = vec->element[2][idx_t];
					rtd_mul(pr, mij, vj);
					rtd_add(mat_vec_i, mat_vec_i, pr);
					total_index++;
				}
			}
			total_index += mat->real_nzero_col_dim[i] - nz; /* skip per-row SIMD padding */
			set_tdvector_i(ret, i, mat_vec_i);
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM Neon
	float64x2_t a2[TDSIZE], vb2[TDSIZE], tmp2[TDSIZE], tmp4_mul[TDSIZE];
	long int jmax, jres, col_dim;
	double mat_vec_i[TDSIZE], vset[TDSIZE];

	for(i = 0; i < mat->row_dim; i++)
	{
		//ret->element[i] = 0.0;
		//tmp2 = vdupq_n_f64(0.0);
		_bncneon_set0_td(tmp2);

		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		//jmax = (mat->nzero_col_dim[i] / 2) * 2;
		//jres =  mat->nzero_col_dim[i] % 2;
		jmax = (mat->real_nzero_col_dim[i] / 2) * 2;
		jres =  mat->real_nzero_col_dim[i] % 2;
		//printf("%ld jmax, jres = %ld, %ld\n", i, jmax, jres);
		for(j = 0; j < jmax; j += 2)
		{
			//printf("load "); fflush(stdout);
			//ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			a2[0] = vld1q_f64(&(mat->element[0][total_index]));
			a2[1] = vld1q_f64(&(mat->element[1][total_index]));
			a2[2] = vld1q_f64(&(mat->element[2][total_index]));

			vset[0] = 0.0; vset[1] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[0] = vec->element[0][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[1] = vec->element[0][mat->nzero_index[i][j + 1]];
			//printf("set0 %g, %g, %g, %g ", vset[0], vset[1], vset[0], vset[1]); fflush(stdout);
    		vb2[0] = vld1q_f64(vset);
			//vb2[0] = _mm256_set_pd(
			//	vset[1], // vec->element[0][mat->nzero_index[i][j + 1]],
			//	vset[0]  // vec->element[0][mat->nzero_index[i][j    ]]
			//);

			//printf("set1 ");
			vset[0] = 0.0; vset[1] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[0] = vec->element[1][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[1] = vec->element[1][mat->nzero_index[i][j + 1]];
			//printf("set1 %g, %g, %g, %g ", vset[0], vset[1], vset[0], vset[1]); fflush(stdout);
			vb2[1] = vld1q_f64(vset);
			//vb2[1] = _mm256_set_pd(
			//	vset[1], // vec->element[1][mat->nzero_index[i][j + 1]],
			//	vset[0]  // vec->element[1][mat->nzero_index[i][j    ]]
			//);

			//printf("set2 ");
			vset[0] = 0.0; vset[1] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[0] = vec->element[2][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[1] = vec->element[2][mat->nzero_index[i][j + 1]];
			//printf("set1 %g, %g, %g, %g ", vset[0], vset[1], vset[0], vset[1]); fflush(stdout);
    		vb2[2] = vld1q_f64(vset);
			//vb2[2] = _mm256_set_pd(
			//	vset[1], // vec->element[1][mat->nzero_index[i][j + 1]],
			//	vset[0]  // vec->element[1][mat->nzero_index[i][j    ]]
			//);
//
			//printf("mul_add "); fflush(stdout);
			//tmp2 = vfmaq_f64(tmp2, a2, vb2);
			_bncneon_rtd_mul(tmp4_mul, a2, vb2);
			_bncneon_rtd_add(tmp2, tmp2, tmp4_mul);

			// total_index++;
			total_index += 2;
			//printf("%ld ", j);
		}
		//total_index += mat->real_nzero_col_dim[i];
		//printf("\n");
		//mat_vec_i = tmp2[0] + tmp2[1];
		_bncneon_rtd_sum128d(mat_vec_i, tmp2);

		//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i[0], mat->nzero_col_dim[i], jmax, jres);
/*
		col_dim = mat->nzero_col_dim[i];
		switch(jres)
		{
			case 1:
				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[0] = mat->element[0][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 1]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 1]];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 1]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				break;

			case 2:
				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[0] = mat->element[0][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 2]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 2]];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 2]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;
			
				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[0] = mat->element[0][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 1]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 1]];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 1]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				break;

			case 3:
				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 3]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[0] = mat->element[0][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 3]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 3]];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 3]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[0] = mat->element[0][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 2]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 2]];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 2]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				//mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				mat_ij[0] = mat->element[0][total_index];
				mat_ij[1] = mat->element[1][total_index];
				mat_ij[0] = mat->element[0][total_index];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 1]];
				vec_j[1] = vec->element[1][mat->nzero_index[i][col_dim - 1]];
				vec_j[0] = vec->element[0][mat->nzero_index[i][col_dim - 1]];
				rtd_mul(tmp, mat_ij, vec_j);
				rtd_add(mat_vec_i, mat_vec_i, tmp);
				total_index++;

				break;
		}
*/
		//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i[0]);
		set_tdvector_i(ret, i, mat_vec_i);
	}



#else // others

	for(i = 0; i < mat->row_dim; i++)
	{
		//get_mpfvector_i(ret, i) = 0.0;
		//mpf_set_ui(get_mpfvector_i(ret, i), 0UL);
		//rtd_set_ui(get_tdvector_i(ret, i), 0UL);
		rtd_set_ui(ret_i, 0UL);

		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			//get_mpfvector_i(ret, i) += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			//mpf_mul(tmp, mat->element[total_index], get_mpfvector_i(vec, mat->nzero_index[i][j]]));
			//mpf_mul(tmp, get_mpfvector_i(mat, total_index), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			//rtd_mul(tmp, (mpf_ptr)(mat->element[total_index]), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			mat_ij[0] = mat->element[0][total_index];
			mat_ij[1] = mat->element[1][total_index];
			mat_ij[2] = mat->element[2][total_index];

			vec_j[0] = vec->element[0][mat->nzero_index[i][j]];
			vec_j[1] = vec->element[1][mat->nzero_index[i][j]];
			vec_j[2] = vec->element[2][mat->nzero_index[i][j]];

			//printf("%d, %d: rtd_mul ", i, j);
			rtd_mul(tmp, mat_ij, vec_j);
			//mpf_add(get_mpfvector_i(ret, i), get_mpfvector_i(ret, i), tmp);
			//printf("rtd_add ");
			rtd_add(ret_i, ret_i, tmp);
			total_index++;
		}
		set_tdvector_i(ret, i, ret_i);
		//printf("set_tdvector_i %ld\n", i);
	}

#endif // __AVX2__

	return SUCCESS;
}

/* Multiply TDRSMatrix^T * TDVector */
int mul_tdrsmatrixt_tdvec(TDVector ret, TDRSMatrix mat, TDVector vec)
{
	long int i, j, total_index;
	//tdfloat tmp, mat_ji, vec_j, ret_j;
	double tmp[TDSIZE], mat_ji[TDSIZE], vec_i[TDSIZE], ret_j[TDSIZE];

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	//for(i = 0; i < mat->row_dim; i++)
	//	mpf_set_ui(get_mpfvector_i(ret, i), 0UL);

	// ret := 0
	set0_tdvector(ret);
	total_index = 0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4[TDSIZE], vb4[TDSIZE], tmp4[TDSIZE], ret4[TDSIZE];
	long int jmax, jres, col_dim;
	double vset[_BNC_D_WIDTH];

	for(i = 0; i < mat->row_dim; i++)
	{
		vb4[0] = _mm256_set1_pd(vec->element[0][i]);
		vb4[1] = _mm256_set1_pd(vec->element[1][i]);
		vb4[2] = _mm256_set1_pd(vec->element[2][i]);

		//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		jmax = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

		//jmax = (mat->row_dim / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres =  mat->row_dim % _BNC_D_WIDTH;
		for(j = 0; j < jmax; j += _BNC_D_WIDTH)
		{
			//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
			a4[0] = _mm256_load_pd(&(mat->element[0][total_index]));
			a4[1] = _mm256_load_pd(&(mat->element[1][total_index]));
			a4[2] = _mm256_load_pd(&(mat->element[2][total_index]));

			vset[0] = 0.0; vset[1] = 0.0; vset[2] = 0.0; vset[3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[0] = ret->element[0][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[1] = ret->element[0][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[2] = ret->element[0][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[3] = ret->element[0][mat->nzero_index[i][j + 3]];

			ret4[0] = _mm256_set_pd(
				vset[3], // ret->element[0][mat->nzero_index[i][j + 3]],
				vset[2], // ret->element[0][mat->nzero_index[i][j + 2]],
				vset[1], // ret->element[0][mat->nzero_index[i][j + 1]],
				vset[0]  // ret->element[0][mat->nzero_index[i][j    ]]
			);
			vset[0] = 0.0; vset[1] = 0.0; vset[2] = 0.0; vset[3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[0] = ret->element[1][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[1] = ret->element[1][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[2] = ret->element[1][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[3] = ret->element[1][mat->nzero_index[i][j + 3]];

			ret4[1] = _mm256_set_pd(
				vset[3], // ret->element[1][mat->nzero_index[i][j + 3]],
				vset[2], // ret->element[1][mat->nzero_index[i][j + 2]],
				vset[1], // ret->element[1][mat->nzero_index[i][j + 1]],
				vset[0]  // ret->element[1][mat->nzero_index[i][j    ]]
			);

			vset[0] = 0.0; vset[1] = 0.0; vset[2] = 0.0; vset[3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[0] = ret->element[2][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[1] = ret->element[2][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[2] = ret->element[2][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[3] = ret->element[2][mat->nzero_index[i][j + 3]];

			ret4[2] = _mm256_set_pd(
				vset[3], // ret->element[0][mat->nzero_index[i][j + 3]],
				vset[2], // ret->element[0][mat->nzero_index[i][j + 2]],
				vset[1], // ret->element[0][mat->nzero_index[i][j + 1]],
				vset[0]  // ret->element[0][mat->nzero_index[i][j    ]]
			);

			//ret4 = _mm256_fmadd_pd(a4, vb4, ret4);
			_bncavx2_rtd_mul(tmp4, a4, vb4);
			_bncavx2_rtd_add(ret4, ret4, tmp4);

			if((j    ) < mat->nzero_col_dim[i])
			{
				ret->element[0][mat->nzero_index[i][j    ]] = ret4[0][0];
				ret->element[1][mat->nzero_index[i][j    ]] = ret4[1][0];
				ret->element[2][mat->nzero_index[i][j    ]] = ret4[2][0];
			}
			if((j + 1) < mat->nzero_col_dim[i])
			{
				ret->element[0][mat->nzero_index[i][j + 1]] = ret4[0][1];
				ret->element[1][mat->nzero_index[i][j + 1]] = ret4[1][1];
				ret->element[2][mat->nzero_index[i][j + 1]] = ret4[2][1];
			}
			if((j + 2) < mat->nzero_col_dim[i])
			{
				ret->element[0][mat->nzero_index[i][j + 2]] = ret4[0][2];
				ret->element[1][mat->nzero_index[i][j + 2]] = ret4[1][2];
				ret->element[2][mat->nzero_index[i][j + 2]] = ret4[2][2];
			}
			if((j + 3) < mat->nzero_col_dim[i])
			{
				ret->element[0][mat->nzero_index[i][j + 3]] = ret4[0][3];
				ret->element[1][mat->nzero_index[i][j + 3]] = ret4[1][3];
				ret->element[2][mat->nzero_index[i][j + 3]] = ret4[2][3];
			}
/*
			ret->element[0][mat->nzero_index[i][j + 3]] = ret4[0][3];
			ret->element[0][mat->nzero_index[i][j + 2]] = ret4[0][2];
			ret->element[0][mat->nzero_index[i][j + 1]] = ret4[0][1];
			ret->element[0][mat->nzero_index[i][j    ]] = ret4[0][0];

			ret->element[1][mat->nzero_index[i][j + 3]] = ret4[1][3];
			ret->element[1][mat->nzero_index[i][j + 2]] = ret4[1][2];
			ret->element[1][mat->nzero_index[i][j + 1]] = ret4[1][1];
			ret->element[1][mat->nzero_index[i][j    ]] = ret4[1][0];

			ret->element[2][mat->nzero_index[i][j + 3]] = ret4[2][3];
			ret->element[2][mat->nzero_index[i][j + 2]] = ret4[2][2];
			ret->element[2][mat->nzero_index[i][j + 1]] = ret4[2][1];
			ret->element[2][mat->nzero_index[i][j    ]] = ret4[2][0];
*/
			// total_index++;
			total_index += _BNC_D_WIDTH;
		}
		//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);
/*
		col_dim = mat->nzero_col_dim[i];

		vec_i[0] = vec->element[0][i];
		vec_i[1] = vec->element[1][i];
		vec_i[2] = vec->element[2][i];

		switch(jres)
		{
			case 1: 
				//ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 1]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 1]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 1]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 1]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 1]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 1]] = ret_j[2];

				total_index++;
				break;

			case 2:
				//ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 2]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 2]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 2]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 1]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 1]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 1]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 1]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 1]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 1]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 1]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 1]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 1]] = ret_j[2];

				total_index++;
				break;

			case 3:
				//ret->element[mat->nzero_index[i][col_dim - 3]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 3]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 3]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 3]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 3]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 3]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 3]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 2]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 2]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 2]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 2]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 2]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 2]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 1]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 1]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 1]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 1]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 1]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 1]] = ret_j[2];

				total_index++;
				break; 

		}
*/
		//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8[TDSIZE], vb8[TDSIZE], tmp8[TDSIZE], ret8[TDSIZE];
	long int jmax, jres, col_dim;

	for(i = 0; i < mat->row_dim; i++)
	{
		vb8[0] = _mm512_set1_pd(vec->element[0][i]);
		vb8[1] = _mm512_set1_pd(vec->element[1][i]);
		vb8[2] = _mm512_set1_pd(vec->element[2][i]);

		//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		jmax = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

		{
		long int jfull = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		long int *_nzi = mat->nzero_index[i];
		for(j = 0; j < jfull; j += _BNC_D_WIDTH)
		{
			a8[0] = _mm512_load_pd(&(mat->element[0][total_index]));
			a8[1] = _mm512_load_pd(&(mat->element[1][total_index]));
			a8[2] = _mm512_load_pd(&(mat->element[2][total_index]));
			ret8[0] = _mm512_set_pd(ret->element[0][_nzi[j+7]], ret->element[0][_nzi[j+6]], ret->element[0][_nzi[j+5]], ret->element[0][_nzi[j+4]], ret->element[0][_nzi[j+3]], ret->element[0][_nzi[j+2]], ret->element[0][_nzi[j+1]], ret->element[0][_nzi[j  ]]);
			ret8[1] = _mm512_set_pd(ret->element[1][_nzi[j+7]], ret->element[1][_nzi[j+6]], ret->element[1][_nzi[j+5]], ret->element[1][_nzi[j+4]], ret->element[1][_nzi[j+3]], ret->element[1][_nzi[j+2]], ret->element[1][_nzi[j+1]], ret->element[1][_nzi[j  ]]);
			ret8[2] = _mm512_set_pd(ret->element[2][_nzi[j+7]], ret->element[2][_nzi[j+6]], ret->element[2][_nzi[j+5]], ret->element[2][_nzi[j+4]], ret->element[2][_nzi[j+3]], ret->element[2][_nzi[j+2]], ret->element[2][_nzi[j+1]], ret->element[2][_nzi[j  ]]);
			_bncavx512_rtd_mul(tmp8, a8, vb8);
			_bncavx512_rtd_add(ret8, ret8, tmp8);
			ret->element[0][_nzi[j+7]] = ret8[0][7];
			ret->element[0][_nzi[j+6]] = ret8[0][6];
			ret->element[0][_nzi[j+5]] = ret8[0][5];
			ret->element[0][_nzi[j+4]] = ret8[0][4];
			ret->element[0][_nzi[j+3]] = ret8[0][3];
			ret->element[0][_nzi[j+2]] = ret8[0][2];
			ret->element[0][_nzi[j+1]] = ret8[0][1];
			ret->element[0][_nzi[j  ]] = ret8[0][0];
			ret->element[1][_nzi[j+7]] = ret8[1][7];
			ret->element[1][_nzi[j+6]] = ret8[1][6];
			ret->element[1][_nzi[j+5]] = ret8[1][5];
			ret->element[1][_nzi[j+4]] = ret8[1][4];
			ret->element[1][_nzi[j+3]] = ret8[1][3];
			ret->element[1][_nzi[j+2]] = ret8[1][2];
			ret->element[1][_nzi[j+1]] = ret8[1][1];
			ret->element[1][_nzi[j  ]] = ret8[1][0];
			ret->element[2][_nzi[j+7]] = ret8[2][7];
			ret->element[2][_nzi[j+6]] = ret8[2][6];
			ret->element[2][_nzi[j+5]] = ret8[2][5];
			ret->element[2][_nzi[j+4]] = ret8[2][4];
			ret->element[2][_nzi[j+3]] = ret8[2][3];
			ret->element[2][_nzi[j+2]] = ret8[2][2];
			ret->element[2][_nzi[j+1]] = ret8[2][1];
			ret->element[2][_nzi[j  ]] = ret8[2][0];
			total_index += _BNC_D_WIDTH;
		}
		if(jfull < jmax)
		{
			long int ncd = mat->nzero_col_dim[i]; int _qg;
			a8[0] = _mm512_load_pd(&(mat->element[0][total_index]));
			a8[1] = _mm512_load_pd(&(mat->element[1][total_index]));
			a8[2] = _mm512_load_pd(&(mat->element[2][total_index]));
			{ double _vsg[8]={0,0,0,0,0,0,0,0};
			  for(_qg=0;_qg<_BNC_D_WIDTH;_qg++) if((jfull+_qg)<ncd) _vsg[_qg]=ret->element[0][_nzi[jfull+_qg]];
			  ret8[0] = _mm512_set_pd(_vsg[7],_vsg[6],_vsg[5],_vsg[4],_vsg[3],_vsg[2],_vsg[1],_vsg[0]); }
			{ double _vsg[8]={0,0,0,0,0,0,0,0};
			  for(_qg=0;_qg<_BNC_D_WIDTH;_qg++) if((jfull+_qg)<ncd) _vsg[_qg]=ret->element[1][_nzi[jfull+_qg]];
			  ret8[1] = _mm512_set_pd(_vsg[7],_vsg[6],_vsg[5],_vsg[4],_vsg[3],_vsg[2],_vsg[1],_vsg[0]); }
			{ double _vsg[8]={0,0,0,0,0,0,0,0};
			  for(_qg=0;_qg<_BNC_D_WIDTH;_qg++) if((jfull+_qg)<ncd) _vsg[_qg]=ret->element[2][_nzi[jfull+_qg]];
			  ret8[2] = _mm512_set_pd(_vsg[7],_vsg[6],_vsg[5],_vsg[4],_vsg[3],_vsg[2],_vsg[1],_vsg[0]); }
			_bncavx512_rtd_mul(tmp8, a8, vb8);
			_bncavx512_rtd_add(ret8, ret8, tmp8);
			if((jfull+7)<ncd) ret->element[0][_nzi[jfull+7]] = ret8[0][7];
			if((jfull+6)<ncd) ret->element[0][_nzi[jfull+6]] = ret8[0][6];
			if((jfull+5)<ncd) ret->element[0][_nzi[jfull+5]] = ret8[0][5];
			if((jfull+4)<ncd) ret->element[0][_nzi[jfull+4]] = ret8[0][4];
			if((jfull+3)<ncd) ret->element[0][_nzi[jfull+3]] = ret8[0][3];
			if((jfull+2)<ncd) ret->element[0][_nzi[jfull+2]] = ret8[0][2];
			if((jfull+1)<ncd) ret->element[0][_nzi[jfull+1]] = ret8[0][1];
			if((jfull+0)<ncd) ret->element[0][_nzi[jfull  ]] = ret8[0][0];
			if((jfull+7)<ncd) ret->element[1][_nzi[jfull+7]] = ret8[1][7];
			if((jfull+6)<ncd) ret->element[1][_nzi[jfull+6]] = ret8[1][6];
			if((jfull+5)<ncd) ret->element[1][_nzi[jfull+5]] = ret8[1][5];
			if((jfull+4)<ncd) ret->element[1][_nzi[jfull+4]] = ret8[1][4];
			if((jfull+3)<ncd) ret->element[1][_nzi[jfull+3]] = ret8[1][3];
			if((jfull+2)<ncd) ret->element[1][_nzi[jfull+2]] = ret8[1][2];
			if((jfull+1)<ncd) ret->element[1][_nzi[jfull+1]] = ret8[1][1];
			if((jfull+0)<ncd) ret->element[1][_nzi[jfull  ]] = ret8[1][0];
			if((jfull+7)<ncd) ret->element[2][_nzi[jfull+7]] = ret8[2][7];
			if((jfull+6)<ncd) ret->element[2][_nzi[jfull+6]] = ret8[2][6];
			if((jfull+5)<ncd) ret->element[2][_nzi[jfull+5]] = ret8[2][5];
			if((jfull+4)<ncd) ret->element[2][_nzi[jfull+4]] = ret8[2][4];
			if((jfull+3)<ncd) ret->element[2][_nzi[jfull+3]] = ret8[2][3];
			if((jfull+2)<ncd) ret->element[2][_nzi[jfull+2]] = ret8[2][2];
			if((jfull+1)<ncd) ret->element[2][_nzi[jfull+1]] = ret8[2][1];
			if((jfull+0)<ncd) ret->element[2][_nzi[jfull  ]] = ret8[2][0];
			total_index += _BNC_D_WIDTH;
		}
		}
		//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);
/*
		col_dim = mat->nzero_col_dim[i];
		switch(jres)
		{
			case 1: 
				//ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 1]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 1]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 1]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 1]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 1]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 1]] = ret_j[2];

				total_index++;

				break;

			case 2:
				//ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 2]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 2]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 2]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 2]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 2]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 2]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 1]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 1]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 1]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 1]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 1]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 1]] = ret_j[2];

				total_index++;

				break;

			case 3:
				//ret->element[mat->nzero_index[i][col_dim - 3]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 3]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 3]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 3]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 3]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 3]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 3]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 2]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 2]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 2]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 2]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 2]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 2]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 1]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 1]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 1]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 1]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 1]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 1]] = ret_j[2];

				total_index++;
				break;

			case 4:
				//ret->element[mat->nzero_index[i][col_dim - 4]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 4]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 4]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 4]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 4]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 4]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 4]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 3]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 3]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 3]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 3]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 3]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 3]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 3]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 2]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 2]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 2]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 2]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 2]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 2]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 1]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 1]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 1]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 1]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 1]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 1]] = ret_j[2];

				total_index++;

				break;

			case 5:
				//ret->element[mat->nzero_index[i][col_dim - 5]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 5]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 5]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 5]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 5]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 5]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 5]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 4]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 4]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 4]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 4]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 4]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 4]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 4]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 3]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 3]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 3]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 3]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 3]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 3]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 3]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 2]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 2]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 2]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 2]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 2]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 2]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 1]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 1]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 1]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 1]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 1]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 1]] = ret_j[2];

				total_index++;
				break;

			case 6:
				//ret->element[mat->nzero_index[i][col_dim - 6]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 6]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 6]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 6]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 6]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 6]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 6]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 5]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 5]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 5]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 5]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 5]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 5]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 5]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 4]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 4]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 4]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 4]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);
				ret->element[0][mat->nzero_index[i][col_dim - 4]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 4]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 4]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 3]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 3]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 3]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 3]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 3]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 3]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 3]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 2]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 2]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 2]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 2]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 2]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 2]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 1]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 1]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 1]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 1]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 1]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 1]] = ret_j[2];

				total_index++;
				break;

			case 7:
				//ret->element[mat->nzero_index[i][col_dim - 7]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 7]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 7]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 7]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 7]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 7]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 7]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 6]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 6]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 6]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 6]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 6]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 6]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 6]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 5]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 5]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 5]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 5]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 5]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 5]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 5]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 4]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 4]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 4]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 4]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 4]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 4]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 4]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 3]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 3]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 3]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 3]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 3]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 3]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 3]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 2]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 2]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 2]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 2]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 2]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 2]] = ret_j[2];

				total_index++;

				//ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				mat_ji[0] = mat->element[0][total_index];
				mat_ji[1] = mat->element[1][total_index];
				mat_ji[2] = mat->element[2][total_index];

				ret_j[0] = ret->element[0][mat->nzero_index[i][col_dim - 1]];
				ret_j[1] = ret->element[1][mat->nzero_index[i][col_dim - 1]];
				ret_j[2] = ret->element[2][mat->nzero_index[i][col_dim - 1]];

				rtd_mul(tmp, mat_ji, vec_i);
				rtd_add(ret_j, ret_j, tmp);

				ret->element[0][mat->nzero_index[i][col_dim - 1]] = ret_j[0];
				ret->element[1][mat->nzero_index[i][col_dim - 1]] = ret_j[1];
				ret->element[2][mat->nzero_index[i][col_dim - 1]] = ret_j[2];

				total_index++;
				break;

		}
*/
		//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2
	/* SVE2 TD A^T*x: gather TD limbs of ret[], FMA, scatter back. */
	{
		long _vl = (long)svcntd();
		long int jmax, jres;
		for(i = 0; i < mat->row_dim; i++)
		{
			svfloat64_t vb0 = svdup_n_f64(vec->element[0][i]);
			svfloat64_t vb1 = svdup_n_f64(vec->element[1][i]);
			svfloat64_t vb2v= svdup_n_f64(vec->element[2][i]);
			long int nz = mat->nzero_col_dim[i];
			jmax = (nz / _vl) * _vl;
			jres = nz - jmax;
			for(j = 0; j < jmax; j += _vl)
			{
				svbool_t pg = svptrue_b64();
				svfloat64_t a0 = svld1_f64(pg, &(mat->element[0][total_index]));
				svfloat64_t a1 = svld1_f64(pg, &(mat->element[1][total_index]));
				svfloat64_t a2v= svld1_f64(pg, &(mat->element[2][total_index]));
				svint64_t idx  = svld1_s64(pg, (int64_t*)&(mat->nzero_index[i][j]));
				svfloat64_t r0 = svld1_gather_s64index_f64(pg, ret->element[0], idx);
				svfloat64_t r1 = svld1_gather_s64index_f64(pg, ret->element[1], idx);
				svfloat64_t r2 = svld1_gather_s64index_f64(pg, ret->element[2], idx);
				svfloat64_t m0, m1, m2;
				_bncsve2_rtd_mul(pg, &m0, &m1, &m2, a0, a1, a2v, vb0, vb1, vb2v);
				_bncsve2_rtd_add(pg, &r0, &r1, &r2, r0, r1, r2, m0, m1, m2);
				svst1_scatter_s64index_f64(pg, ret->element[0], idx, r0);
				svst1_scatter_s64index_f64(pg, ret->element[1], idx, r1);
				svst1_scatter_s64index_f64(pg, ret->element[2], idx, r2);
				total_index += _vl;
			}
			/* scalar tail */
			{
				double mij[TDSIZE], vi[TDSIZE], pr[TDSIZE], cur[TDSIZE];
				long int col_dim = mat->nzero_col_dim[i];
				vi[0] = vec->element[0][i];
				vi[1] = vec->element[1][i];
				vi[2] = vec->element[2][i];
				for(j = 0; j < jres; j++)
				{
					long int idx_t = mat->nzero_index[i][col_dim - jres + j];
					mij[0] = mat->element[0][total_index];
					mij[1] = mat->element[1][total_index];
					mij[2] = mat->element[2][total_index];
					cur[0] = ret->element[0][idx_t];
					cur[1] = ret->element[1][idx_t];
					cur[2] = ret->element[2][idx_t];
					rtd_mul(pr, mij, vi);
					rtd_add(cur, cur, pr);
					ret->element[0][idx_t] = cur[0];
					ret->element[1][idx_t] = cur[1];
					ret->element[2][idx_t] = cur[2];
					total_index++;
				}
			}
			total_index += mat->real_nzero_col_dim[i] - nz; /* skip per-row SIMD padding */
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM Neon (correct scalar transpose + padding skip)

	for(i = 0; i < mat->row_dim; i++)
	{
		vec_i[0] = vec->element[0][i];
		vec_i[1] = vec->element[1][i];
		vec_i[2] = vec->element[2][i];

		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
			mat_ji[0] = mat->element[0][total_index];
			mat_ji[1] = mat->element[1][total_index];
			mat_ji[2] = mat->element[2][total_index];

			ret_j[0] = ret->element[0][mat->nzero_index[i][j]];
			ret_j[1] = ret->element[1][mat->nzero_index[i][j]];
			ret_j[2] = ret->element[2][mat->nzero_index[i][j]];

			//rtd_mul(tmp, mat->element[total_index]), get_tdvector_i(vec, i));
			rtd_mul(tmp, mat_ji, vec_i);
			//rtd_add(ret->element[mat->nzero_index[i][j]], ret->element[mat->nzero_index[i][j]], tmp);
			rtd_add(ret_j, ret_j, tmp);
			set_tdvector_i(ret, mat->nzero_index[i][j], ret_j);
			total_index++;
		}
		total_index += mat->real_nzero_col_dim[i] - mat->nzero_col_dim[i]; /* skip per-row SIMD padding */
	}

#else // __AVX2__

	for(i = 0; i < mat->row_dim; i++)
	{
		vec_i[0] = vec->element[0][i];
		vec_i[1] = vec->element[1][i];
		vec_i[2] = vec->element[2][i];

		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
			mat_ji[0] = mat->element[0][total_index];
			mat_ji[1] = mat->element[1][total_index];
			mat_ji[2] = mat->element[2][total_index];

			ret_j[0] = ret->element[0][mat->nzero_index[i][j]];
			ret_j[1] = ret->element[1][mat->nzero_index[i][j]];
			ret_j[2] = ret->element[2][mat->nzero_index[i][j]];

			//rtd_mul(tmp, mat->element[total_index]), get_tdvector_i(vec, i));
			rtd_mul(tmp, mat_ji, vec_i);
			//rtd_add(ret->element[mat->nzero_index[i][j]], ret->element[mat->nzero_index[i][j]], tmp);
			rtd_add(ret_j, ret_j, tmp);
			set_tdvector_i(ret, mat->nzero_index[i][j], ret_j);
			total_index++;
		}
	}

#endif // __AVX2__

	return SUCCESS;
}

/* Multiply DRSMatrix * TDVector */
int mul_drsmatrix_tdvec(TDVector ret, DRSMatrix mat, TDVector vec)
{
	long int i, j, total_index;
	//tdfloat tmp, mat_ij, vec_j, ret_i;
	double tmp[TDSIZE], mat_ij, vec_j[TDSIZE], ret_i[TDSIZE];

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, vb4[TDSIZE], tmp4[TDSIZE], tmp4_mul[TDSIZE];
	long int jmax, jres, col_dim;
	double mat_vec_i[TDSIZE], vset[_BNC_D_WIDTH];

	for(i = 0; i < mat->row_dim; i++)
	{
		//ret->element[i] = 0.0;
		//tmp4 = _mm256_setzero_pd();
		_bncavx2_set0_td(tmp4);

		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		jmax = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;
		//printf("%ld jmax, jres = %ld, %ld\n", i, jmax, jres);
		for(j = 0; j < jmax; j += _BNC_D_WIDTH)
		{
			//printf("load "); fflush(stdout);
			//ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			a4 = _mm256_load_pd(&(mat->element[total_index]));
			//a4[1]  = _mm256_load_pd(&(mat->element[1][total_index]));
			//a4[2]  = _mm256_load_pd(&(mat->element[2][total_index]));

			vset[0] = 0.0; vset[1] = 0.0; vset[2] = 0.0; vset[3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[0] = vec->element[0][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[1] = vec->element[0][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[2] = vec->element[0][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[3] = vec->element[0][mat->nzero_index[i][j + 3]];
			//printf("set0 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			vb4[0] = _mm256_set_pd(
				vset[3], // vec->element[0][mat->nzero_index[i][j + 3]],
				vset[2], // vec->element[0][mat->nzero_index[i][j + 2]],
				vset[1], // vec->element[0][mat->nzero_index[i][j + 1]],
				vset[0]  // vec->element[0][mat->nzero_index[i][j    ]]
			);

			//printf("set1 ");
			vset[0] = 0.0; vset[1] = 0.0; vset[2] = 0.0; vset[3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[0] = vec->element[1][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[1] = vec->element[1][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[2] = vec->element[1][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[3] = vec->element[1][mat->nzero_index[i][j + 3]];
			//printf("set1 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			vb4[1] = _mm256_set_pd(
				vset[3], // vec->element[1][mat->nzero_index[i][j + 3]],
				vset[2], // vec->element[1][mat->nzero_index[i][j + 2]],
				vset[1], // vec->element[1][mat->nzero_index[i][j + 1]],
				vset[0]  // vec->element[1][mat->nzero_index[i][j    ]]
			);

			//printf("set2 ");
			vset[0] = 0.0; vset[1] = 0.0; vset[2] = 0.0; vset[3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[0] = vec->element[2][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[1] = vec->element[2][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[2] = vec->element[2][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[3] = vec->element[2][mat->nzero_index[i][j + 3]];
			//printf("set1 %g, %g, %g, %g ", vset[0], vset[1], vset[2], vset[3]); fflush(stdout);
			vb4[2] = _mm256_set_pd(
				vset[3], // vec->element[1][mat->nzero_index[i][j + 3]],
				vset[2], // vec->element[1][mat->nzero_index[i][j + 2]],
				vset[1], // vec->element[1][mat->nzero_index[i][j + 1]],
				vset[0]  // vec->element[1][mat->nzero_index[i][j    ]]
			);

			//printf("mul_add "); fflush(stdout);
			//tmp4 = _mm256_fmadd_pd(a4, vb4, tmp4);
			//_bncavx2_rtd_mul(tmp4_mul, a4, vb4);
			_bncavx2_rtd_mulq_d(tmp4_mul, vb4, a4);//, vb4);
			_bncavx2_rtd_add(tmp4, tmp4, tmp4_mul);

			// total_index++;
			total_index += _BNC_D_WIDTH;
			//printf("%ld ", j);
		}
		//total_index += mat->real_nzero_col_dim[i];
		//printf("\n");
		//mat_vec_i = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3];
		_bncavx2_rtd_sum256d(mat_vec_i, tmp4);
		//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i[0]);
		set_tdvector_i(ret, i, mat_vec_i);
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, vb8[TDSIZE], tmp8[TDSIZE], tmp8_mul[TDSIZE];
	long int jmax, jres, col_dim;
	double mat_vec_i[TDSIZE];

	for(i = 0; i < mat->row_dim; i++)
	{
		//ret->element[i] = 0.0;]
		//tmp8 = _mm512_setzero_pd();
		_bncavx512_set0_td(tmp8);

		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		jmax = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;
		for(j = 0; j < jmax; j += _BNC_D_WIDTH)
		{
			//ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			a8 = _mm512_load_pd(&(mat->element[total_index]));
			//a8[1] = _mm512_load_pd(&(mat->element[1][total_index]));
			//a8[2] = _mm512_load_pd(&(mat->element[2][total_index]));

			{ double _vsg[8]={0,0,0,0,0,0,0,0}; long _qg;
			  for(_qg=0;_qg<_BNC_D_WIDTH;_qg++) if((j+_qg)<mat->nzero_col_dim[i]) _vsg[_qg]=vec->element[0][mat->nzero_index[i][j+_qg]];
			  vb8[0] = _mm512_set_pd(_vsg[7],_vsg[6],_vsg[5],_vsg[4],_vsg[3],_vsg[2],_vsg[1],_vsg[0]); }
			{ double _vsg[8]={0,0,0,0,0,0,0,0}; long _qg;
			  for(_qg=0;_qg<_BNC_D_WIDTH;_qg++) if((j+_qg)<mat->nzero_col_dim[i]) _vsg[_qg]=vec->element[1][mat->nzero_index[i][j+_qg]];
			  vb8[1] = _mm512_set_pd(_vsg[7],_vsg[6],_vsg[5],_vsg[4],_vsg[3],_vsg[2],_vsg[1],_vsg[0]); }
			{ double _vsg[8]={0,0,0,0,0,0,0,0}; long _qg;
			  for(_qg=0;_qg<_BNC_D_WIDTH;_qg++) if((j+_qg)<mat->nzero_col_dim[i]) _vsg[_qg]=vec->element[2][mat->nzero_index[i][j+_qg]];
			  vb8[2] = _mm512_set_pd(_vsg[7],_vsg[6],_vsg[5],_vsg[4],_vsg[3],_vsg[2],_vsg[1],_vsg[0]); }

			//tmp8 = _mm512_fmadd_pd(a4, vb4, tmp4);
			//_bncavx512_rtd_mul(tmp8_mul, a8, vb8);
			_bncavx512_rtd_mul_d(tmp8_mul, vb8, a8);
			_bncavx512_rtd_add(tmp8, tmp8, tmp8_mul);

			// total_index++;
			total_index += _BNC_D_WIDTH;
		}
		//mat_vec_i = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3] + tmp4[4] + tmp4[5] + tmp4[6] + tmp4[7];
		_bncavx512_rtd_sum512d(mat_vec_i, tmp8);
		set_tdvector_i(ret, i, mat_vec_i);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2
	/* SVE2 D*TD SpMV: mat is plain double, vec is TD; gather TD limbs of vec. */
	{
		long _vl = (long)svcntd();
		long int jmax, jres;
		double mat_vec_i[TDSIZE];
		for(i = 0; i < mat->row_dim; i++)
		{
			svfloat64_t t0, t1, t2;
			_bncsve2_rtd_set0(&t0, &t1, &t2);
			long int nz = mat->nzero_col_dim[i];
			jmax = (nz / _vl) * _vl;
			jres = nz - jmax;
			for(j = 0; j < jmax; j += _vl)
			{
				svbool_t pg = svptrue_b64();
				svfloat64_t a_v = svld1_f64(pg, &(mat->element[total_index]));
				svint64_t idx   = svld1_s64(pg, (int64_t*)&(mat->nzero_index[i][j]));
				svfloat64_t b0 = svld1_gather_s64index_f64(pg, vec->element[0], idx);
				svfloat64_t b1 = svld1_gather_s64index_f64(pg, vec->element[1], idx);
				svfloat64_t b2 = svld1_gather_s64index_f64(pg, vec->element[2], idx);
				/* Promote D to TD: (a, 0, 0) * (b0, b1, b2) */
				svfloat64_t zerov = svdup_n_f64(0.0);
				svfloat64_t m0, m1, m2;
				_bncsve2_rtd_mul(pg, &m0, &m1, &m2, a_v, zerov, zerov, b0, b1, b2);
				_bncsve2_rtd_add(pg, &t0, &t1, &t2, t0, t1, t2, m0, m1, m2);
				total_index += _vl;
			}
			{
				long _L, _vl = (long)svcntd();
				double _la0[64], _la1[64], _la2[64];
				svst1_f64(svptrue_b64(), _la0, t0);
				svst1_f64(svptrue_b64(), _la1, t1);
				svst1_f64(svptrue_b64(), _la2, t2);
				rtd_set_ui(mat_vec_i, 0UL);
				for(_L = 0; _L < _vl; _L++)
				{
					rtd_add_d(mat_vec_i, mat_vec_i, _la0[_L]);
					rtd_add_d(mat_vec_i, mat_vec_i, _la1[_L]);
					rtd_add_d(mat_vec_i, mat_vec_i, _la2[_L]);
				}
			}
			/* scalar tail */
			{
				double mij[TDSIZE], vj[TDSIZE], pr[TDSIZE];
				long int col_dim = mat->nzero_col_dim[i];
				for(j = 0; j < jres; j++)
				{
					long int idx_t = mat->nzero_index[i][col_dim - jres + j];
					mij[0] = mat->element[total_index];
					mij[1] = 0.0; mij[2] = 0.0;
					vj[0]  = vec->element[0][idx_t];
					vj[1]  = vec->element[1][idx_t];
					vj[2]  = vec->element[2][idx_t];
					rtd_mul(pr, mij, vj);
					rtd_add(mat_vec_i, mat_vec_i, pr);
					total_index++;
				}
			}
			total_index += mat->real_nzero_col_dim[i] - nz; /* skip per-row SIMD padding */
			set_tdvector_i(ret, i, mat_vec_i);
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM Neon
	float64x2_t a2, vb2[TDSIZE], tmp2[TDSIZE], tmp4_mul[TDSIZE];
	long int jmax, jres, col_dim;
	double mat_vec_i[TDSIZE], vset[2];

	for(i = 0; i < mat->row_dim; i++)
	{
		//ret->element[i] = 0.0;
		//tmp2 = vdupq_n_f64(0.0);
		_bncneon_set0_td(tmp2);

		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		//jmax = (mat->nzero_col_dim[i] / 2) * 2;
		//jres =  mat->nzero_col_dim[i] % 2;
		jmax = (mat->real_nzero_col_dim[i] / 2) * 2;
		jres =  mat->real_nzero_col_dim[i] % 2;
		//printf("%ld jmax, jres = %ld, %ld\n", i, jmax, jres);
		for(j = 0; j < jmax; j += 2)
		{
			//printf("load "); fflush(stdout);
			//ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			a2 = vld1q_f64(&(mat->element[total_index]));
			//a2[1] = vld1q_f64(&(mat->element[1][total_index]));
			//a2[0] = vld1q_f64(&(mat->element[0][total_index]));

			vset[0] = 0.0; vset[1] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[0] = vec->element[0][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[1] = vec->element[0][mat->nzero_index[i][j + 1]];
			//printf("set0 %g, %g, %g, %g ", vset[0], vset[1], vset[0], vset[1]); fflush(stdout);
    		vb2[0] = vld1q_f64(vset);
			//vb2[0] = _mm256_set_pd(
			//	vset[1], // vec->element[0][mat->nzero_index[i][j + 1]],
			//	vset[0]  // vec->element[0][mat->nzero_index[i][j    ]]
			//);

			//printf("set1 ");
			vset[0] = 0.0; vset[1] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[0] = vec->element[1][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[1] = vec->element[1][mat->nzero_index[i][j + 1]];
			//printf("set1 %g, %g, %g, %g ", vset[0], vset[1], vset[0], vset[1]); fflush(stdout);
    		vb2[1] = vld1q_f64(vset);
			// vb2[1] = _mm256_set_pd(
			// 	vset[1], // vec->element[1][mat->nzero_index[i][j + 1]],
			// 	vset[0]  // vec->element[1][mat->nzero_index[i][j    ]]
			// );

			//printf("set2 ");
			vset[0] = 0.0; vset[1] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[0] = vec->element[2][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[1] = vec->element[2][mat->nzero_index[i][j + 1]];
			//printf("set1 %g, %g, %g, %g ", vset[0], vset[1], vset[0], vset[1]); fflush(stdout);
    		vb2[2] = vld1q_f64(vset);
			//vb2[0] = _mm256_set_pd(
			//	vset[1], // vec->element[1][mat->nzero_index[i][j + 1]],
			//	vset[0]  // vec->element[1][mat->nzero_index[i][j    ]]
			//);

			//printf("mul_add "); fflush(stdout);
			//tmp2 = vfmaq_f64(tmp2, a2, vb2);
			//_bncneon_rtd_mul(tmp4_mul, a2, vb2);
			_bncneon_rtd_mulq_d(tmp4_mul, vb2, a2);//, vb2);
			_bncneon_rtd_add(tmp2, tmp2, tmp4_mul);

			// total_index++;
			total_index += 2;
			//printf("%ld ", j);
		}
		//total_index += mat->real_nzero_col_dim[i];
		//printf("\n");
		//mat_vec_i = tmp2[0] + tmp2[1];
		_bncneon_rtd_sum128d(mat_vec_i, tmp2);
		//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i[0]);
		set_tdvector_i(ret, i, mat_vec_i);
	}



#else // others

	for(i = 0; i < mat->row_dim; i++)
	{
		//get_mpfvector_i(ret, i) = 0.0;
		//mpf_set_ui(get_mpfvector_i(ret, i), 0UL);
		//rtd_set_ui(get_tdvector_i(ret, i), 0UL);
		rtd_set_ui(ret_i, 0UL);

		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			//get_mpfvector_i(ret, i) += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			//mpf_mul(tmp, mat->element[total_index], get_mpfvector_i(vec, mat->nzero_index[i][j]]));
			//mpf_mul(tmp, get_mpfvector_i(mat, total_index), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			//rtd_mul(tmp, (mpf_ptr)(mat->element[total_index]), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			mat_ij = mat->element[total_index];
			//mat_ij[1] = mat->element[1][total_index];
			//mat_ij[2] = mat->element[2][total_index];

			vec_j[0] = vec->element[0][mat->nzero_index[i][j]];
			vec_j[1] = vec->element[1][mat->nzero_index[i][j]];
			vec_j[2] = vec->element[2][mat->nzero_index[i][j]];

			//printf("%d, %d: rtd_mul ", i, j);
			//rtd_mul(tmp, mat_ij, vec_j);
			rtd_mul_d(tmp, vec_j, mat_ij);
			//mpf_add(get_mpfvector_i(ret, i), get_mpfvector_i(ret, i), tmp);
			//printf("rtd_add ");
			rtd_add(ret_i, ret_i, tmp);
			total_index++;
		}
		set_tdvector_i(ret, i, ret_i);
		//printf("set_tdvector_i %ld\n", i);
	}

#endif // __AVX2__

	return SUCCESS;
}

/* Multiply DRSMatrix^T * TDVector */
int mul_drsmatrixt_tdvec(TDVector ret, DRSMatrix mat, TDVector vec)
{
	long int i, j, total_index;
	//tdfloat tmp, mat_ji, vec_j, ret_j;
	double tmp[TDSIZE], mat_ji, vec_i[TDSIZE], ret_j[TDSIZE];

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	//for(i = 0; i < mat->row_dim; i++)
	//	mpf_set_ui(get_mpfvector_i(ret, i), 0UL);

	// ret := 0
	set0_tdvector(ret);
	total_index = 0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, vb4[TDSIZE], tmp4[TDSIZE], ret4[TDSIZE];
	long int jmax, jres, col_dim;
	double vset[_BNC_D_WIDTH];

	for(i = 0; i < mat->row_dim; i++)
	{
		vb4[0] = _mm256_set1_pd(vec->element[0][i]);
		vb4[1] = _mm256_set1_pd(vec->element[1][i]);
		vb4[2] = _mm256_set1_pd(vec->element[2][i]);

		//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		jmax = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

		//jmax = (mat->row_dim / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres =  mat->row_dim % _BNC_D_WIDTH;
		for(j = 0; j < jmax; j += _BNC_D_WIDTH)
		{
			//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
			a4 = _mm256_load_pd(&(mat->element[total_index]));
			//a4[1] = _mm256_load_pd(&(mat->element[1][total_index]));
			//a4[2] = _mm256_load_pd(&(mat->element[2][total_index]));

			vset[0] = 0.0; vset[1] = 0.0; vset[2] = 0.0; vset[3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[0] = ret->element[0][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[1] = ret->element[0][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[2] = ret->element[0][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[3] = ret->element[0][mat->nzero_index[i][j + 3]];

			ret4[0] = _mm256_set_pd(
				vset[3], // ret->element[0][mat->nzero_index[i][j + 3]],
				vset[2], // ret->element[0][mat->nzero_index[i][j + 2]],
				vset[1], // ret->element[0][mat->nzero_index[i][j + 1]],
				vset[0]  // ret->element[0][mat->nzero_index[i][j    ]]
			);
			vset[0] = 0.0; vset[1] = 0.0; vset[2] = 0.0; vset[3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[0] = ret->element[1][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[1] = ret->element[1][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[2] = ret->element[1][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[3] = ret->element[1][mat->nzero_index[i][j + 3]];

			ret4[1] = _mm256_set_pd(
				vset[3], // ret->element[1][mat->nzero_index[i][j + 3]],
				vset[2], // ret->element[1][mat->nzero_index[i][j + 2]],
				vset[1], // ret->element[1][mat->nzero_index[i][j + 1]],
				vset[0]  // ret->element[1][mat->nzero_index[i][j    ]]
			);

			vset[0] = 0.0; vset[1] = 0.0; vset[2] = 0.0; vset[3] = 0.0;
			if((j    ) < mat->nzero_col_dim[i]) vset[0] = ret->element[2][mat->nzero_index[i][j    ]];
			if((j + 1) < mat->nzero_col_dim[i]) vset[1] = ret->element[2][mat->nzero_index[i][j + 1]];
			if((j + 2) < mat->nzero_col_dim[i]) vset[2] = ret->element[2][mat->nzero_index[i][j + 2]];
			if((j + 3) < mat->nzero_col_dim[i]) vset[3] = ret->element[2][mat->nzero_index[i][j + 3]];

			ret4[2] = _mm256_set_pd(
				vset[3], // ret->element[0][mat->nzero_index[i][j + 3]],
				vset[2], // ret->element[0][mat->nzero_index[i][j + 2]],
				vset[1], // ret->element[0][mat->nzero_index[i][j + 1]],
				vset[0]  // ret->element[0][mat->nzero_index[i][j    ]]
			);

			//ret4 = _mm256_fmadd_pd(a4, vb4, ret4);
			//_bncavx2_rtd_mul(tmp4, a4, vb4);
			//_bncavx2_rtd_mul_d(tmp4, a4, vb4); // , a4);
			_bncavx2_rtd_mulq_d(tmp4, vb4, a4);
			_bncavx2_rtd_add(ret4, ret4, tmp4);

			if((j    ) < mat->nzero_col_dim[i])
			{
				ret->element[0][mat->nzero_index[i][j    ]] = ret4[0][0];
				ret->element[1][mat->nzero_index[i][j    ]] = ret4[1][0];
				ret->element[2][mat->nzero_index[i][j    ]] = ret4[2][0];
			}
			if((j + 1) < mat->nzero_col_dim[i])
			{
				ret->element[0][mat->nzero_index[i][j + 1]] = ret4[0][1];
				ret->element[1][mat->nzero_index[i][j + 1]] = ret4[1][1];
				ret->element[2][mat->nzero_index[i][j + 1]] = ret4[2][1];
			}
			if((j + 2) < mat->nzero_col_dim[i])
			{
				ret->element[0][mat->nzero_index[i][j + 2]] = ret4[0][2];
				ret->element[1][mat->nzero_index[i][j + 2]] = ret4[1][2];
				ret->element[2][mat->nzero_index[i][j + 2]] = ret4[2][2];
			}
			if((j + 3) < mat->nzero_col_dim[i])
			{
				ret->element[0][mat->nzero_index[i][j + 3]] = ret4[0][3];
				ret->element[1][mat->nzero_index[i][j + 3]] = ret4[1][3];
				ret->element[2][mat->nzero_index[i][j + 3]] = ret4[2][3];
			}
			// total_index++;
			total_index += _BNC_D_WIDTH;
		}
		//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, vb8[TDSIZE], tmp8[TDSIZE], ret8[TDSIZE];
	long int jmax, jres, col_dim;

	for(i = 0; i < mat->row_dim; i++)
	{
		vb8[0] = _mm512_set1_pd(vec->element[0][i]);
		vb8[1] = _mm512_set1_pd(vec->element[1][i]);
		vb8[2] = _mm512_set1_pd(vec->element[2][i]);

		//jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		jmax = (mat->real_nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres =  mat->real_nzero_col_dim[i] % _BNC_D_WIDTH;

		for(j = 0; j < jmax; j += _BNC_D_WIDTH)
		{
			//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
			//a8[0] = _mm512_load_pd(&(mat->element[0][total_index]));
			a8 = _mm512_load_pd(&(mat->element[total_index]));
			//a8[1] = _mm512_load_pd(&(mat->element[1][total_index]));
			//a8[2] = _mm512_load_pd(&(mat->element[2][total_index]));

			{ double _vsg[8]={0,0,0,0,0,0,0,0}; long _qg;
			  for(_qg=0;_qg<_BNC_D_WIDTH;_qg++) if((j+_qg)<mat->nzero_col_dim[i]) _vsg[_qg]=ret->element[0][mat->nzero_index[i][j+_qg]];
			  ret8[0] = _mm512_set_pd(_vsg[7],_vsg[6],_vsg[5],_vsg[4],_vsg[3],_vsg[2],_vsg[1],_vsg[0]); }
			{ double _vsg[8]={0,0,0,0,0,0,0,0}; long _qg;
			  for(_qg=0;_qg<_BNC_D_WIDTH;_qg++) if((j+_qg)<mat->nzero_col_dim[i]) _vsg[_qg]=ret->element[1][mat->nzero_index[i][j+_qg]];
			  ret8[1] = _mm512_set_pd(_vsg[7],_vsg[6],_vsg[5],_vsg[4],_vsg[3],_vsg[2],_vsg[1],_vsg[0]); }
			{ double _vsg[8]={0,0,0,0,0,0,0,0}; long _qg;
			  for(_qg=0;_qg<_BNC_D_WIDTH;_qg++) if((j+_qg)<mat->nzero_col_dim[i]) _vsg[_qg]=ret->element[2][mat->nzero_index[i][j+_qg]];
			  ret8[2] = _mm512_set_pd(_vsg[7],_vsg[6],_vsg[5],_vsg[4],_vsg[3],_vsg[2],_vsg[1],_vsg[0]); }

			//ret8 = _mm512_fmadd_pd(a8, vb8, ret8);
			//_bncavx512_rtd_mul(tmp8, a8, vb8);
			_bncavx512_rtd_mul_d(tmp8, vb8, a8);
			_bncavx512_rtd_add(ret8, ret8, tmp8);

			if((j + 7) < mat->nzero_col_dim[i]) ret->element[0][mat->nzero_index[i][j + 7]] = ret8[0][7];
			if((j + 6) < mat->nzero_col_dim[i]) ret->element[0][mat->nzero_index[i][j + 6]] = ret8[0][6];
			if((j + 5) < mat->nzero_col_dim[i]) ret->element[0][mat->nzero_index[i][j + 5]] = ret8[0][5];
			if((j + 4) < mat->nzero_col_dim[i]) ret->element[0][mat->nzero_index[i][j + 4]] = ret8[0][4];
			if((j + 3) < mat->nzero_col_dim[i]) ret->element[0][mat->nzero_index[i][j + 3]] = ret8[0][3];
			if((j + 2) < mat->nzero_col_dim[i]) ret->element[0][mat->nzero_index[i][j + 2]] = ret8[0][2];
			if((j + 1) < mat->nzero_col_dim[i]) ret->element[0][mat->nzero_index[i][j + 1]] = ret8[0][1];
			if((j + 0) < mat->nzero_col_dim[i]) ret->element[0][mat->nzero_index[i][j    ]] = ret8[0][0];

			if((j + 7) < mat->nzero_col_dim[i]) ret->element[1][mat->nzero_index[i][j + 7]] = ret8[1][7];
			if((j + 6) < mat->nzero_col_dim[i]) ret->element[1][mat->nzero_index[i][j + 6]] = ret8[1][6];
			if((j + 5) < mat->nzero_col_dim[i]) ret->element[1][mat->nzero_index[i][j + 5]] = ret8[1][5];
			if((j + 4) < mat->nzero_col_dim[i]) ret->element[1][mat->nzero_index[i][j + 4]] = ret8[1][4];
			if((j + 3) < mat->nzero_col_dim[i]) ret->element[1][mat->nzero_index[i][j + 3]] = ret8[1][3];
			if((j + 2) < mat->nzero_col_dim[i]) ret->element[1][mat->nzero_index[i][j + 2]] = ret8[1][2];
			if((j + 1) < mat->nzero_col_dim[i]) ret->element[1][mat->nzero_index[i][j + 1]] = ret8[1][1];
			if((j + 0) < mat->nzero_col_dim[i]) ret->element[1][mat->nzero_index[i][j    ]] = ret8[1][0];

			if((j + 7) < mat->nzero_col_dim[i]) ret->element[2][mat->nzero_index[i][j + 7]] = ret8[2][7];
			if((j + 6) < mat->nzero_col_dim[i]) ret->element[2][mat->nzero_index[i][j + 6]] = ret8[2][6];
			if((j + 5) < mat->nzero_col_dim[i]) ret->element[2][mat->nzero_index[i][j + 5]] = ret8[2][5];
			if((j + 4) < mat->nzero_col_dim[i]) ret->element[2][mat->nzero_index[i][j + 4]] = ret8[2][4];
			if((j + 3) < mat->nzero_col_dim[i]) ret->element[2][mat->nzero_index[i][j + 3]] = ret8[2][3];
			if((j + 2) < mat->nzero_col_dim[i]) ret->element[2][mat->nzero_index[i][j + 2]] = ret8[2][2];
			if((j + 1) < mat->nzero_col_dim[i]) ret->element[2][mat->nzero_index[i][j + 1]] = ret8[2][1];
			if((j + 0) < mat->nzero_col_dim[i]) ret->element[2][mat->nzero_index[i][j    ]] = ret8[2][0];

			// total_index++;
			total_index += _BNC_D_WIDTH;
		}
		//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2
	/* SVE2 D*TD A^T*x: mat is D, vec is TD; gather TD ret[], FMA, scatter back. */
	{
		long _vl = (long)svcntd();
		long int jmax, jres;
		for(i = 0; i < mat->row_dim; i++)
		{
			svfloat64_t vb0 = svdup_n_f64(vec->element[0][i]);
			svfloat64_t vb1 = svdup_n_f64(vec->element[1][i]);
			svfloat64_t vb2v= svdup_n_f64(vec->element[2][i]);
			long int nz = mat->nzero_col_dim[i];
			jmax = (nz / _vl) * _vl;
			jres = nz - jmax;
			for(j = 0; j < jmax; j += _vl)
			{
				svbool_t pg = svptrue_b64();
				svfloat64_t a_v = svld1_f64(pg, &(mat->element[total_index]));
				svint64_t idx   = svld1_s64(pg, (int64_t*)&(mat->nzero_index[i][j]));
				svfloat64_t r0 = svld1_gather_s64index_f64(pg, ret->element[0], idx);
				svfloat64_t r1 = svld1_gather_s64index_f64(pg, ret->element[1], idx);
				svfloat64_t r2 = svld1_gather_s64index_f64(pg, ret->element[2], idx);
				/* promote D to TD: (a, 0, 0) * (vb0, vb1, vb2) */
				svfloat64_t zerov = svdup_n_f64(0.0);
				svfloat64_t m0, m1, m2;
				_bncsve2_rtd_mul(pg, &m0, &m1, &m2, a_v, zerov, zerov, vb0, vb1, vb2v);
				_bncsve2_rtd_add(pg, &r0, &r1, &r2, r0, r1, r2, m0, m1, m2);
				svst1_scatter_s64index_f64(pg, ret->element[0], idx, r0);
				svst1_scatter_s64index_f64(pg, ret->element[1], idx, r1);
				svst1_scatter_s64index_f64(pg, ret->element[2], idx, r2);
				total_index += _vl;
			}
			/* scalar tail */
			{
				double mij[TDSIZE], vi[TDSIZE], pr[TDSIZE], cur[TDSIZE];
				long int col_dim = mat->nzero_col_dim[i];
				vi[0] = vec->element[0][i];
				vi[1] = vec->element[1][i];
				vi[2] = vec->element[2][i];
				for(j = 0; j < jres; j++)
				{
					long int idx_t = mat->nzero_index[i][col_dim - jres + j];
					mij[0] = mat->element[total_index];
					mij[1] = 0.0; mij[2] = 0.0;
					cur[0] = ret->element[0][idx_t];
					cur[1] = ret->element[1][idx_t];
					cur[2] = ret->element[2][idx_t];
					rtd_mul(pr, mij, vi);
					rtd_add(cur, cur, pr);
					ret->element[0][idx_t] = cur[0];
					ret->element[1][idx_t] = cur[1];
					ret->element[2][idx_t] = cur[2];
					total_index++;
				}
			}
			total_index += mat->real_nzero_col_dim[i] - nz; /* skip per-row SIMD padding */
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM Neon (correct scalar transpose + padding skip)

	for(i = 0; i < mat->row_dim; i++)
	{
		vec_i[0] = vec->element[0][i];
		vec_i[1] = vec->element[1][i];
		vec_i[2] = vec->element[2][i];

		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
			mat_ji = mat->element[total_index];
			//mat_ji[1] = mat->element[1][total_index];
			//mat_ji[2] = mat->element[2][total_index];

			ret_j[0] = ret->element[0][mat->nzero_index[i][j]];
			ret_j[1] = ret->element[1][mat->nzero_index[i][j]];
			ret_j[2] = ret->element[2][mat->nzero_index[i][j]];

			//rtd_mul(tmp, mat->element[total_index]), get_tdvector_i(vec, i));
			//rtd_mul(tmp, mat_ji, vec_i);
			rtd_mul_d(tmp, vec_i, mat_ji);
			//rtd_add(ret->element[mat->nzero_index[i][j]], ret->element[mat->nzero_index[i][j]], tmp);
			rtd_add(ret_j, ret_j, tmp);
			set_tdvector_i(ret, mat->nzero_index[i][j], ret_j);
			total_index++;
		}
		total_index += mat->real_nzero_col_dim[i] - mat->nzero_col_dim[i]; /* skip per-row SIMD padding */
	}

#else // __AVX2__

	for(i = 0; i < mat->row_dim; i++)
	{
		vec_i[0] = vec->element[0][i];
		vec_i[1] = vec->element[1][i];
		vec_i[2] = vec->element[2][i];

		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
			mat_ji = mat->element[total_index];
			//mat_ji[1] = mat->element[1][total_index];
			//mat_ji[2] = mat->element[2][total_index];

			ret_j[0] = ret->element[0][mat->nzero_index[i][j]];
			ret_j[1] = ret->element[1][mat->nzero_index[i][j]];
			ret_j[2] = ret->element[2][mat->nzero_index[i][j]];

			//rtd_mul(tmp, mat->element[total_index]), get_tdvector_i(vec, i));
			//rtd_mul(tmp, mat_ji, vec_i);
			rtd_mul_d(tmp, vec_i, mat_ji);
			//rtd_add(ret->element[mat->nzero_index[i][j]], ret->element[mat->nzero_index[i][j]], tmp);
			rtd_add(ret_j, ret_j, tmp);
			set_tdvector_i(ret, mat->nzero_index[i][j], ret_j);
			total_index++;
		}
	}

#endif // __AVX2__

	return SUCCESS;
}

/* Power Method for Randomly Sparse Matrices */
/* 	TDVector *evec: the eigenvector for max eigenvalue */
/* 	TDRSMatrix *drsmat: Randomly sparse matrix */
/* 	double *reps, *aeps: Relative and Absolute tolerance */
/* 	long int max_times: Maximum iterative times of Power method */
void tdpower_sp(double max_eig[TDSIZE], TDVector evec, TDRSMatrix mat, double reps[TDSIZE], double aeps[TDSIZE], long int max_times)
{
	long int i, absmax_index, times;
	tdfloat absmax_new_evec, old_max_eig, tmp, tmp2;
	TDVector new_evec;

	new_evec = init_tdvector(mat->row_dim);

	/* initialize evec */
	for(i = 0; i < evec->dim; i++)
		rtd_set_ui(get_tdvector_i(evec, i), 1UL);
		//evec->element[i] = 1.0;

	/* main loop */
	//old_max_eig = 0.0;
	rtd_set_ui(old_max_eig.val, 0UL);
	for(times = 0; times < max_times; times++)
	{
		/* w := A * x */
		mul_tdrsmatrix_tdvec(new_evec, mat, evec);
		absmax_index = absmax_index_tdvector(absmax_new_evec.val, new_evec);
//		max_eig = absmax_new_evec / evec->element[absmax_index];
		rtd_div(max_eig, absmax_new_evec.val, get_tdvector_i(evec, absmax_index));
//		smul_mpfvector(evec, 1.0 / absmax_new_evec, new_evec);
//		smul_mpfvector(evec, 1.0 / norm1_MPFVector(new_evec), new_evec); // Baba's example
		norm1_tdvector(tmp.val, new_evec);
		rtd_ui_div(tmp.val, 1UL, tmp.val);
		cmul_tdvector(evec, tmp.val, new_evec);

//		if((fabs(max_eig - old_max_eig) <= reps * fabs(old_max_eig) + aeps) && (times >= 2))
		rtd_sub(tmp.val, max_eig, old_max_eig.val);
		rtd_abs(tmp.val, tmp.val);
		
		rtd_abs(tmp2.val, old_max_eig.val);
		rtd_mul(tmp2.val, reps, tmp2.val);
		rtd_add(tmp2.val, tmp2.val, aeps);
		if((rtd_cmp(tmp.val, tmp2.val) <= 0) && (times >= 2))
		{
			fprintf(stderr, "Convergent!(Iterative Times = %ld)\n", times);
			break;
		}
		if(times % 10 == 0)
			fprintf(stderr, "%5ld %25.17e\n", times, rtd_get_d(max_eig));
		//old_max_eig = max_eig;
		rtd_set(old_max_eig.val, max_eig);
	}

	free_tdvector(new_evec);

//	return max_eig;
	return;
}

/* Select index of absolute maximum element and its value in TDVector */
long int absmax_index_tdvector(double ret[TDSIZE], TDVector vec)
{
	long int absmax_index, i;
	tdfloat abs_element;

	rtd_set_ui(ret, 0UL);
	absmax_index = 0;
	for(i = 0; i < vec->dim; i++)
	{
		rtd_abs(abs_element.val, get_tdvector_i(vec, i));
		if(rtd_cmp(ret, abs_element.val) < 0)
		{
			absmax_index = i;
			//*ret = abs_element;
		}
	}

	rtd_set(ret, get_tdvector_i(vec, absmax_index));

	return absmax_index;
}

// 2024-07-30 (Tue)
/* c := (td)a */
void subst_tdrsmatrix_drsmat(TDRSMatrix c, DRSMatrix a)
{
	long int i, j, total_index;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_tdrsmatrix_drsmat\n");
		return;
	}

	//for(total_index = 0; total_index < c->nzero_total_num; total_index++)
	for(total_index = 0; total_index < c->real_nzero_total_num; total_index++)
	{
		c->element[0][total_index] = a->element[total_index];
		c->element[1][total_index] = (double)0.0;
		c->element[2][total_index] = (double)0.0;
	}
}

// 2024-07-30 (Tue)
/* c := (d)a */
void subst_drsmatrix_tdrsmat(DRSMatrix c, TDRSMatrix a)
{
	long int i, j, ij_index;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_drsmatrix_tdrsmat\n");
		return;
	}

	//for(ij_index = 0; ij_index < c->nzero_total_num; ij_index++)
	for(ij_index = 0; ij_index < c->real_nzero_total_num; ij_index++)
	{
		c->element[ij_index] = a->element[0][ij_index];
	}
}

// 2024-07-31(Ted) T.Kouya
// absmax_row_tdrsmatrix
void absmax_row_tdrsmatrix(double mu[TDSIZE], long int *max_j, long int row_index, TDRSMatrix mat)
{
    long int j, max_index = 0;
    double abs_aij[TDSIZE], aij[TDSIZE];

	//mu = fabs(mat[i * col_dim + 0]);
	get_tdrsmatrix_ij(aij, mat, row_index, 0);
    rtd_abs(mu, aij);

	//for(j = 1; j < mat->nzero_col_dim[row_index]; j++)
	for(j = 1; j < mat->real_nzero_col_dim[row_index]; j++)
	{
		get_tdrsmatrix_ij(aij, mat, row_index, j);
		rtd_abs(abs_aij, aij);
		if(rtd_cmp(abs_aij, mu) > 0)
        {
			//mu = abs_aij;
            rtd_set(mu, abs_aij);
            max_index = j;
        }
	}

    if(max_j != NULL)
        *max_j = max_index;

    //return mu;
    return;
}

// 2024-08-04 (SUN) T.Kouya
// absmax_col_tdrsmatrix
void absmax_col_tdrsmatrix(double mu[TDSIZE], long int *max_i, long int col_index, TDRSMatrix mat)
{
    long int i, max_index = 0;
    double abs_aij[TDSIZE], aij[TDSIZE];

	//mu = fabs(mat[i * col_dim + 0]);
	get_tdrsmatrix_ij(aij, mat, 0, col_index);
    rtd_abs(mu, aij);

	//for(j = 1; j < mat->nzero_col_dim[row_index]; j++)
	for(i = 1; i < mat->row_dim; i++) //mat->real_nzero_col_dim[row_index]; j++)
	{
		get_tdrsmatrix_ij(aij, mat, i, col_index);
		rtd_abs(abs_aij, aij);
		if(rtd_cmp(abs_aij, mu) > 0)
        {
			//mu = abs_aij;
            rtd_set(mu, abs_aij);
            max_index = i;
        }
	}

    if(max_i != NULL)
        *max_i = max_index;

    //return mu;
    return;
}

// 2024-07-31(Wed) T.Kouya
/* c := a + (double)b */
void add_tdrsmatrix_drsmat(TDRSMatrix c, TDRSMatrix a, DRSMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;
    double aij[TDSIZE], bij[TDSIZE], tmp[TDSIZE];

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_tdrsmatrix_drsmat\n");
		return;
	}
	row_dim = c->row_dim;
	//real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_tdrsmatrix_drsmat\n");
		return;
	}
	col_dim = c->col_dim;
	//real_col_dim = c->real_col_dim;

	real_total_dim = c->nzero_total_num; // real_row_dim * real_col_dim;

	for(index = 0; index < c->real_nzero_total_num; index++)
	{
		//rtd_add_d(tmp, get_tdrsmatrix_ij(a, i, j), get_drsmatrix_ij(b, i, j));
        bij[0] = b->element[index];
		bij[1] = 0.0; 
		bij[2] = 0.0; 
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];
		aij[2] = a->element[2][index];
        rtd_add(tmp, aij, bij);
		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1];
		c->element[2][index] = tmp[2];
	}
}

// 2024-07-31(Wed) T.Kouya
/* c := a - (double)b */
void sub_tdrsmatrix_drsmat(TDRSMatrix c, TDRSMatrix a, DRSMatrix b)
{
	long int i, j, row_dim, col_dim, real_row_dim, real_col_dim, real_total_dim, index;
    double aij[TDSIZE], bij[TDSIZE], tmp[TDSIZE];

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: sub_tdrsmatrix_drsmat\n");
		return;
	}
	row_dim = c->row_dim;
	//real_row_dim = c->real_row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: sub_tdrsmatrix_drsmat\n");
		return;
	}
	col_dim = c->col_dim;
	//real_col_dim = c->real_col_dim;

	real_total_dim = c->nzero_total_num; // real_row_dim * real_col_dim;

	for(index = 0; index < c->real_nzero_total_num; index++)
	{
		//rtd_add_d(tmp, get_tdrsmatrix_ij(a, i, j), get_drsmatrix_ij(b, i, j));
        bij[0] = b->element[index];
		bij[1] = 0.0; 
		bij[2] = 0.0; 
		aij[0] = a->element[0][index];
		aij[1] = a->element[1][index];
		aij[2] = a->element[2][index];
        rtd_sub(tmp, aij, bij);
		c->element[0][index] = tmp[0];
		c->element[1][index] = tmp[1];
		c->element[2][index] = tmp[2];
	}
}

// 2024-07-31(Wed) T.Kouya
// SplitMat_A
// return real_num_div
int split_tdrsmatrix_drsmat(DRSMatrix ret_mat[], int num_div, TDRSMatrix org_mat)
{
	long int i, j, index, row_dim, col_dim, real_total_dim, total_index;
	long int num_digits = 53; // IEEE double prec.
    int real_num_div;
	//long int num_digits = SPLIT_NUM_DIGITS; // IEEE double prec.
	//long int num_digits = 64; // IEEE double prec.
	//double *s;
    TDRSMatrix tmp_org_mat;
    DRSMatrix s, tmp_mat[2], in_ret_mat; 
	double mu, mu_total, abs_aij, t_exp, power2;

    row_dim = org_mat->row_dim;
    //row_dim = org_mat->real_row_dim;
    col_dim = org_mat->col_dim;
    //real_total_dim = org_mat->real_row_dim * org_mat->real_col_dim;
	real_total_dim = org_mat->nzero_total_num;

	// threshold matrix 
	//s = (double *)calloc(row_dim * col_dim, sizeof(double));
    //s = init_ddmatrix(row_dim, col_dim);
    s = init_set_drsmatrix_tdrsmat(org_mat); //row_dim, org_mat->nzero_col_dim, org_mat->nzero_total_num);
	set0_drsmatrix(s);

    //tmp_mat[0] = init_ddmatrix(row_dim, col_dim);
    //tmp_mat[1] = init_ddmatrix(row_dim, col_dim);
    tmp_mat[0] = init_set_drsmatrix_tdrsmat(org_mat); //row_dim, org_mat->nzero_col_dim, org_mat->nzero_total_num);
    tmp_mat[1] = init_set_drsmatrix_tdrsmat(org_mat); //row_dim, org_mat->nzero_col_dim, org_mat->nzero_total_num);
	if(ret_mat == NULL)
		in_ret_mat = init_set_drsmatrix_tdrsmat(org_mat);
	else
		in_ret_mat = ret_mat[0];

    // tmp_org_mat := org_mat;
    tmp_org_mat = init_set_tdrsmatrix(org_mat);
   	//subst_ddmatrix(tmp_org_mat, org_mat);

    real_num_div = 0;
    for(index = 0; index < num_div; index++)
    {
        //printf("In split_ddmatrix ... index= %d\n", index);
        //set0_dmatrix(ret_mat[index]);
		if(ret_mat != NULL) // Fix! 2024-09-25 T.Kouya
			in_ret_mat = ret_mat[index];

        //subst_drsmatrix_tdrsmat(ret_mat[index], tmp_org_mat);
        subst_drsmatrix_tdrsmat(in_ret_mat, tmp_org_mat);

        // mu[i] = max_j |mat[i, j]|
        // mu_total = sum mu
        mu_total = 0.0;

		total_index = 0;
        for(i = 0; i < row_dim; i++)
        {
            //absmax_row_ddmatrix(mu, NULL, i, tmp_org_mat);
            //mu = absmax_row_drsmatrix(NULL, i, ret_mat[index]);
            mu = absmax_row_drsmatrix(NULL, i, in_ret_mat);

            mu_total += mu;

            // t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
            //t_exp[0] = ceil(DLOG2(mu[0])) + ceil(((double)num_digits + DLOG2((double)(col_dim + 1))) / 2.0);
            //t_exp[1] = 0.0;
            //printf("     num, num_digits, col_dim + 1   = %15.7e, %ld, %ld\n", mu, num_digits, col_dim + 1);
            //printf("log2(num, num_digits + col_dim + 1) = %15.7e, %15.7e\n", DLOG2(mu), ceil(((double)num_digits + DLOG2((double)(col_dim + 1))) / 2.0));
            //t_exp = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(col_dim + 1))) / 2.0);
            //t_exp = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(ret_mat[index]->nzero_col_dim[i]))) / 2.0);
            t_exp = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(in_ret_mat->nzero_col_dim[i]))) / 2.0);


            // s[i, j] = 2^t_exp
            //rtd_pow(power2, two, t_exp);
            //rtd_pow_mpfr(power2, two, t_exp);
            power2 = pow(2.0, t_exp);
            //printf("power2 = %15.7e\n", power2);
            //printf("index, i, power2 = %ld, %ld, %25.17e\n", index, i, power2[0]);
            //for(j = 0; j < ret_mat[index]->real_nzero_col_dim[i]; j++)
            for(j = 0; j < in_ret_mat->real_nzero_col_dim[i]; j++)
            {
                //s[i * col_dim + j] = pow(2.0, t_exp);
                //set_ddmatrix_ij(s, i, j, power2);
                //set_dmatrix_ij(s, i, j, power2);
				s->element[total_index++] = power2;
            }
        }

        // if ret_mat[index] == 0 -> break
        if(mu_total == 0.0) break;

        // split org_mat to ret_high_mat and ret_low_mat
#ifdef USE_IMKL
        //real_total_dim = ret_mat[index]->nzero_total_num; //ret_mat[index]->real_row_dim * ret_mat[index]->real_col_dim;
        //real_total_dim = ret_mat[index]->real_nzero_total_num; //ret_mat[index]->real_row_dim * ret_mat[index]->real_col_dim;
        real_total_dim = in_ret_mat->real_nzero_total_num; //ret_mat[index]->real_row_dim * ret_mat[index]->real_col_dim;

        // tmp_mat := mat + s
        //blas_dcopy(real_total_dim, ret_mat[index]->element, 1, tmp_mat[0]->element, 1);
        //cblas_daxpy(real_total_dim, 1.0, s->element, 1, tmp_mat[0]->element, 1);
        //cblas_daxpy(real_total_dim, 1.0, s->element, 1, ret_mat[index]->element, 1);
        cblas_daxpy(real_total_dim, 1.0, s->element, 1, in_ret_mat->element, 1);


        // high_mat := tmp_mat - s
        //cblas_daxpy(real_total_dim, -1.0, s->element, 1, ret_mat[index]->element, 1);
        cblas_daxpy(real_total_dim, -1.0, s->element, 1, in_ret_mat->element, 1);


        // low_mat := mat - high_mat
        //cblas_dcopy(real_total_dim, tmp_mat[0]->element, 1, ret_mat[index]->element, 1);
#else // USE_IMKL
        // tmp_mat := mat + s
        //add_drsmatrix(tmp_mat[0], ret_mat[index], s);
        add_drsmatrix(tmp_mat[0], in_ret_mat, s);

        // high_mat := tmp_mat - s
        sub_drsmatrix(tmp_mat[1], tmp_mat[0], s);
        //subst_drsmatrix(ret_mat[index], tmp_mat[1]);
        subst_drsmatrix(in_ret_mat, tmp_mat[1]);

#endif // USE_IMKL

        // low_mat := mat - high_mat
        //sub_ddmatrix_dmat(tmp_org_mat, tmp_org_mat, tmp_mat[1]);
        //sub_tdrsmatrix_drsmat(tmp_org_mat, tmp_org_mat, ret_mat[index]);
        sub_tdrsmatrix_drsmat(tmp_org_mat, tmp_org_mat, in_ret_mat);

        // real_num_div = index + 1;
        real_num_div = index + 1;
    }

	// free s
	//free_ddmatrix(s);
	free_drsmatrix(s);
    free_tdrsmatrix(tmp_org_mat);
    free_drsmatrix(tmp_mat[0]);
    free_drsmatrix(tmp_mat[1]);
	if(ret_mat == NULL)
		free_drsmatrix(in_ret_mat);

    return real_num_div;
}

// Matrix-Vector multiplication based on Ozaki scheme
void mul_tdrsmatrix_tdvec_oz(TDVector ret, TDRSMatrix a, int max_num_div_a, TDVector vb, int max_num_div_vb) //, int num_digits)
{
    int i, j;
    int real_num_div_a, real_num_div_vb, *i_div_a_csr, *j_div_a_csr, total_index;
    long int vec_dim = ret->dim, row_dim = a->row_dim, col_dim = a->col_dim;
    DRSMatrix *div_a;
    DVector *div_vb, div_ret;
#ifdef USE_IMKL
	int *i_div_a_csr_start, *i_div_a_csr_end;
	sparse_matrix_t *mkl_div_a;
	struct matrix_descr descr;
#endif // USE_IMKL

    div_a = (DRSMatrix *)calloc(max_num_div_a, sizeof(DRSMatrix));
    div_vb = (DVector *)calloc(max_num_div_vb, sizeof(DVector));

	// convert our CSR to intel math kernel csr format
#ifdef USE_IMKL
	#ifdef USE_IMKL_OLD
	//i_div_a_csr = (int *)calloc(a->nzero_total_num, sizeof(int));
	i_div_a_csr = (int *)calloc(a->row_dim + 1, sizeof(int));
	//j_div_a_csr = (int *)calloc(a->nzero_total_num, sizeof(int));
	j_div_a_csr = (int *)calloc(a->real_nzero_total_num, sizeof(int));
	total_index = 0;
	for(i = 0; i < row_dim; i++)
	{
		i_div_a_csr[i] = total_index;
		//if(a->nzero_col_dim[i] >= 1)
		if(a->real_nzero_col_dim[i] >= 1)
		{
			for(j = 0; j < a->nzero_col_dim[i]; j++)
			{
				j_div_a_csr[total_index] = a->nzero_index[i][j];
				total_index++;
			}
			// embed gap among nzero_col_dim and real_nzero_col_dim
			for(j = a->nzero_col_dim[i]; j < a->real_nzero_col_dim[i]; j++)
			{
				j_div_a_csr[total_index] = a->nzero_index[i][a->nzero_col_dim[i] - 1] + (j + 1 - a->nzero_col_dim[i]);
				total_index++;
			}
		}
	}
	//i_div_a_csr[row_dim] = a->nzero_total_num;
	i_div_a_csr[row_dim] = a->real_nzero_total_num;
	#else // New
	mkl_div_a = (sparse_matrix_t *)calloc(max_num_div_a, sizeof(sparse_matrix_t));
	convert_indeces_tdrsmatrix_mkl_csrmat(i_div_a_csr_start, i_div_a_csr_end, j_div_a_csr, a);
	#endif // USE_IMKL_OLD
#endif // USE_IMKL

	for(i = 0; i < max_num_div_a; i++)
		div_a[i] = init_set_drsmatrix_tdrsmat(a); // a->row_dim, a->nzero_col_dim, a->nzero_total_num);
        //div_a[i] = init_drsmatrix(row_dim, col_dim);

    for(i = 0; i < max_num_div_vb; i++)
        div_vb[i] = init_dvector(vec_dim);

    div_ret = init_dvector(vec_dim);

    real_num_div_a = split_tdrsmatrix_drsmat(div_a, max_num_div_a, a);
    real_num_div_vb = split_tdvector_dvec(div_vb, max_num_div_vb, vb);

    set0_tdvector(ret);
    for(i = 0; i < real_num_div_a; i++)
    {
#ifdef USE_IMKL
		subst_drsmatrix_mkl_csrmat(&mkl_div_a[i], i_div_a_csr_start, i_div_a_csr_end, j_div_a_csr, div_a[i]);
#endif // USE_IMKL
        for(j = 0; j < real_num_div_vb; j++)
        {

#ifdef USE_IMKL
            set0_dvector(div_ret);
	#ifdef USE_IMKL_OLD
			//mul_drsmatrix_dvec(vec[8], a, vec[6]);
			mkl_cspblas_dcsrgemv("N", &row_dim, div_a[i]->element, i_div_a_csr, j_div_a_csr, div_vb[j]->element, div_ret->element);
	#else // USE_IMKL_OLD
			descr.type = SPARSE_MATRIX_TYPE_GENERAL;
			//descr.mode = ?
			//descr.diag = ?
			mkl_sparse_d_mv(
				SPARSE_OPERATION_NON_TRANSPOSE,
				(double)1.0,
				mkl_div_a[i],
				descr,
				div_vb[j]->element,
				(double)0.0,
				div_ret->element
			);
	#endif // USE_IMKL_OLD
#else // USE_IMKL
            mul_drsmatrix_dvec(div_ret, div_a[i], div_vb[j]);
#endif // USE_IMKL
			//printf("%ld,%ld ||div_ret||_F = %10.3e\n", i, j, norm2_dvector(div_ret));
            add_tdvector_dvec(ret, ret, div_ret);
       }
    }

#ifdef USE_IMKL
	#ifdef USE_IMKL_OLD
	free(i_div_a_csr);
	#else // New!
	free(i_div_a_csr_start);
	free(i_div_a_csr_end);
	for(i = 0; i < max_num_div_a; i++)
		mkl_sparse_destroy(mkl_div_a[i]);
	free(mkl_div_a);
	#endif // USE_IMKL_OLD
	free(j_div_a_csr);
#endif // USE_IMKL

	free_dvector(div_ret);
    for(i = 0; i < max_num_div_a; i++)
        free_drsmatrix(div_a[i]);
    for(i = 0; i < max_num_div_vb; i++)
        free_dvector(div_vb[i]);

    free(div_a);
    free(div_vb);

}

// 2024-08-04 (Sun) T.Kouya
// SplitMat_B
// return real_num_div
int split_tdrsmatrix_t_drsmat(DRSMatrix ret_mat[], int num_div, TDRSMatrix org_mat)
{
	long int i, j, index, row_dim, col_dim, real_total_dim;
	int real_num_div, num_digits = 53; // IEEE double prec.
	//int flag_stop = 0; //, num_digits = SPLIT_NUM_DIGITS; // IEEE double prec.
	//int flag_stop = 0, num_digits = 64; // IEEE double prec.
	//double *s;
    TDRSMatrix tmp_org_mat;
    DRSMatrix s, tmp_mat[2], in_ret_mat;
	//double mu[TDSIZE], abs_aij[TDSIZE], t_exp[TDSIZE], power2[TDSIZE], two[TDSIZE] = {2.0, 0.0};
	double mu, abs_aij, t_exp, power2, mu_total;

    row_dim = org_mat->row_dim;
    //row_dim = org_mat->real_row_dim;
    col_dim = org_mat->col_dim;
    //real_total_dim = org_mat->real_row_dim * org_mat->real_col_dim;
	real_total_dim = org_mat->nzero_total_num;

	// threshold matrix 
	//s = (double *)calloc(row_dim * col_dim, sizeof(double));
    //s = init_ddmatrix(row_dim, col_dim);
    s = init_set_drsmatrix_tdrsmat(org_mat); //row_dim, org_mat->nzero_col_dim, org_mat->nzero_total_num);
	set0_drsmatrix(s);

    //tmp_mat[0] = init_ddmatrix(row_dim, col_dim);
    //tmp_mat[1] = init_ddmatrix(row_dim, col_dim);
    tmp_mat[0] = init_set_drsmatrix_tdrsmat(org_mat); //row_dim, org_mat->nzero_col_dim, org_mat->nzero_total_num);
    tmp_mat[1] = init_set_drsmatrix_tdrsmat(org_mat); //row_dim, org_mat->nzero_col_dim, org_mat->nzero_total_num);
	if(ret_mat == NULL)
		in_ret_mat = init_set_drsmatrix_tdrsmat(org_mat);

    // tmp_org_mat := org_mat;
    tmp_org_mat = init_set_tdrsmatrix(org_mat);
   	//subst_ddmatrix(tmp_org_mat, org_mat);

    real_num_div = 0;
    for(index = 0; index < num_div; index++)
    {
		if(ret_mat != NULL)
			in_ret_mat = ret_mat[index];

        //subst_dmatrix_ddmat(ret_mat[index], tmp_org_mat);
        subst_drsmatrix_tdrsmat(in_ret_mat, tmp_org_mat);

        // mu[j] = max_j |mat[i, j]|
        // mu_total += mu
        mu_total = 0.0;
        for(j = 0; j < col_dim; j++)
        {
            /* mu = fabs(mat[0 * col_dim + j]);
            for(i = 1; i < row_dim; i++)
            {
                abs_aij = fabs(mat[i * col_dim + j]);
                if(abs_aij > mu)
                    mu = abs_aij;
            }
            */
           	//mu = absmax_col_drsmatrix(NULL, j, ret_mat[index]);
            mu = absmax_col_drsmatrix(NULL, j, in_ret_mat);
            mu_total += mu;
            //printf("mu%d: %15.7e ", j, mu);

            // t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
            //t_exp[0] = ceil(DLOG2(mu[0])) + ceil(((double)num_digits + DLOG2((double)(row_dim + 1))) / 2.0);
            //t_exp[1] = 0.0;
            t_exp = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(row_dim + 1))) / 2.0);
            //if(isnan(t_exp))
            //    flag_stop = 1;

            // s[i, j] = 2^t_exp
            //rdd_pow(power2, two, t_exp);
            //rdd_pow_mpfr(power2, two, t_exp);
            power2 = pow(2.0, t_exp);

            //printf("index, j, power2 = %ld, %ld, %25.17e\n", index, j, power2[0]);
            for(i = 0; i < row_dim; i++)
                set_drsmatrix_ij(s, i, j, power2);
                //s[i * col_dim + j] = pow(2.0, t_exp);
        }
        //if(flag_stop == 1)
        //    break;
        if(mu_total == 0.0) break;

#ifdef USE_IMKL
        //real_total_dim = ret_mat[index]->real_row_dim * ret_mat[index]->real_col_dim;
		//real_total_dim = ret_mat[index]->real_nzero_total_num;
		real_total_dim = in_ret_mat->real_nzero_total_num;

        // tmp_mat := mat + s
        //blas_dcopy(real_total_dim, ret_mat[index]->element, 1, tmp_mat[0]->element, 1);
        //cblas_daxpy(real_total_dim, 1.0, s->element, 1, tmp_mat[0]->element, 1);
        //cblas_daxpy(real_total_dim, 1.0, s->element, 1, ret_mat[index]->element, 1);
        cblas_daxpy(real_total_dim, 1.0, s->element, 1, in_ret_mat->element, 1);

        // high_mat := tmp_mat - s
        //cblas_daxpy(real_total_dim, -1.0, s->element, 1, ret_mat[index]->element, 1);
        cblas_daxpy(real_total_dim, -1.0, s->element, 1, in_ret_mat->element, 1);

        // low_mat := mat - high_mat
        //cblas_dcopy(real_total_dim, tmp_mat[0]->element, 1, ret_mat[index]->element, 1);
#else // USE_IMKL
        // tmp_mat := mat + s
        //add_drsmatrix(tmp_mat[0], ret_mat[index], s);
        add_drsmatrix(tmp_mat[0], in_ret_mat, s);

        // high_mat := tmp_mat - s
        sub_drsmatrix(tmp_mat[1], tmp_mat[0], s);
        //subst_drsmatrix(ret_mat[index], tmp_mat[1]);
        subst_drsmatrix(in_ret_mat, tmp_mat[1]);

#endif // USE_IMKL

        // low_mat := mat - high_mat
        //sub_tdrsmatrix_drsmat(tmp_org_mat, tmp_org_mat, ret_mat[index]);
        sub_tdrsmatrix_drsmat(tmp_org_mat, tmp_org_mat, in_ret_mat);

        //sub_ddmatrix_dmat(tmp_org_mat, tmp_org_mat, tmp_mat[1]);

        real_num_div = index + 1;
    }

    // free s
	free_drsmatrix(s);
    free_drsmatrix(tmp_mat[0]);
    free_drsmatrix(tmp_mat[1]);
    free_tdrsmatrix(tmp_org_mat);
	if(ret_mat == NULL)
		free_drsmatrix(in_ret_mat);

    return real_num_div;
}

// 2024-08-02(Fri) T.Kouya
// Transposed Matrix-Vector multiplication based on Ozaki scheme
void mul_tdrsmatrixt_tdvec_oz(TDVector ret, TDRSMatrix a, int max_num_div_a, TDVector vb, int max_num_div_vb)
{
    int i, j;
    int real_num_div_a, real_num_div_vb, *i_div_a_csr, *j_div_a_csr, total_index;
    long int vec_dim = ret->dim, row_dim = a->row_dim, col_dim = a->col_dim;
    DRSMatrix *div_a;
    DVector *div_vb, div_ret;
#ifdef USE_IMKL
	int *i_div_a_csr_start, *i_div_a_csr_end;
	sparse_matrix_t *mkl_div_a;
	struct matrix_descr descr;
#endif // USE_IMKL

    div_a = (DRSMatrix *)calloc(max_num_div_a, sizeof(DRSMatrix));
    div_vb = (DVector *)calloc(max_num_div_vb, sizeof(DVector));

	// convert our CSR to intel math kernel csr format
#ifdef USE_IMKL
	#ifdef USE_IMKL_OLD
	//i_div_a_csr = (int *)calloc(a->nzero_total_num, sizeof(int));
	i_div_a_csr = (int *)calloc(a->row_dim + 1, sizeof(int));
	//j_div_a_csr = (int *)calloc(a->nzero_total_num, sizeof(int));
	j_div_a_csr = (int *)calloc(a->real_nzero_total_num, sizeof(int));
	total_index = 0;
	for(i = 0; i < row_dim; i++)
	{
		i_div_a_csr[i] = total_index;
		//if(a->nzero_col_dim[i] >= 1)
		if(a->real_nzero_col_dim[i] >= 1)
		{
			for(j = 0; j < a->nzero_col_dim[i]; j++)
			{
				j_div_a_csr[total_index] = a->nzero_index[i][j];
				total_index++;
			}
			// embed gap among nzero_col_dim and real_nzero_col_dim
			for(j = a->nzero_col_dim[i]; j < a->real_nzero_col_dim[i]; j++)
			{
				j_div_a_csr[total_index] = a->nzero_index[i][a->nzero_col_dim[i] - 1] + (j + 1 - a->nzero_col_dim[i]);
				total_index++;
			}
		}
	}
	//i_div_a_csr[row_dim] = a->nzero_total_num;
	i_div_a_csr[row_dim] = a->real_nzero_total_num;
	#else // New
	mkl_div_a = (sparse_matrix_t *)calloc(max_num_div_a, sizeof(sparse_matrix_t));
	convert_indeces_tdrsmatrix_mkl_csrmat(i_div_a_csr_start, i_div_a_csr_end, j_div_a_csr, a);
	#endif // USE_IMKL_OLD
#endif // USE_IMKL

	for(i = 0; i < max_num_div_a; i++)
		div_a[i] = init_set_drsmatrix_tdrsmat(a); // a->row_dim, a->nzero_col_dim, a->nzero_total_num);
        //div_a[i] = init_drsmatrix(row_dim, col_dim);

    for(i = 0; i < max_num_div_vb; i++)
        div_vb[i] = init_dvector(vec_dim);

    div_ret = init_dvector(vec_dim);

    //real_num_div_a = split_tdrsmatrix_drsmat(div_a, max_num_div_a, a);
    real_num_div_a = split_tdrsmatrix_t_drsmat(div_a, max_num_div_a, a);
    real_num_div_vb = split_tdvector_dvec(div_vb, max_num_div_vb, vb);

    set0_tdvector(ret);
    for(i = 0; i < real_num_div_a; i++)
    {
#ifdef USE_IMKL
		subst_drsmatrix_mkl_csrmat(&mkl_div_a[i], i_div_a_csr_start, i_div_a_csr_end, j_div_a_csr, div_a[i]);
#endif // USE_IMKL
        for(j = 0; j < real_num_div_vb; j++)
        {

#ifdef USE_IMKL
            set0_dvector(div_ret);
	#ifdef USE_IMKL_OLD
			mkl_cspblas_dcsrgemv("T", &row_dim, div_a[i]->element, i_div_a_csr, j_div_a_csr, div_vb[j]->element, div_ret->element);
	#else // new
			descr.type = SPARSE_MATRIX_TYPE_GENERAL;
			//descr.mode = ?
			//descr.diag = ?
			mkl_sparse_d_mv(
				SPARSE_OPERATION_TRANSPOSE,
				(double)1.0,
				mkl_div_a[i],
				descr,
				div_vb[j]->element,
				(double)0.0,
				div_ret->element
			);
	#endif // USE_IMKL_OLD
#else // USE_IMKL
            mul_drsmatrixt_dvec(div_ret, div_a[i], div_vb[j]);
#endif // USE_IMKL
			//printf("%ld,%ld ||div_ret||_F = %10.3e\n", i, j, norm2_dvector(div_ret));
            add_tdvector_dvec(ret, ret, div_ret);
       }
    }

#ifdef USE_IMKL
	#ifdef USE_IMKL_OLD
	free(i_div_a_csr);
	#else // New!
	free(i_div_a_csr_start);
	free(i_div_a_csr_end);
	for(i = 0; i < max_num_div_a; i++)
		mkl_sparse_destroy(mkl_div_a[i]);
	free(mkl_div_a);
	#endif // USE_IMKL_OLD
	free(j_div_a_csr);
#endif // USE_IMKL

	free_dvector(div_ret);
    for(i = 0; i < max_num_div_a; i++)
        free_drsmatrix(div_a[i]);
    for(i = 0; i < max_num_div_vb; i++)
        free_dvector(div_vb[i]);

    free(div_a);
    free(div_vb);

}

// Incomplete LU decomposition; iLU0_tdrsmatrix
void iLU0_tdrsmatrix(TDRSMatrix mat)
{
	long int i, j, k, row_dim, col_dim;
	double aii[TDSIZE], aji[TDSIZE], ajk[TDSIZE], aik[TDSIZE], ctmp[TDSIZE];
	//double dtmp[TDSIZE];

	row_dim = mat->row_dim;
	col_dim = mat->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		get_tdrsmatrix_ij(aii, mat, i, i);
		if(rtd_cmp_ui(aii, 0UL) != 0)
		{
			for(j = i + 1; j < row_dim; j++)
			{
				get_tdrsmatrix_ij(aji, mat, j, i);
				if(rtd_cmp_ui(aji, 0UL) != 0)
				{
					//aji /= aii;
					rtd_div(aji, aji, aii);
					set_tdrsmatrix_ij(mat, j, i, aji);
					for(k = i + 1; k < col_dim; k++)
					{
						get_tdrsmatrix_ij(ajk, mat, j, k);
						get_tdrsmatrix_ij(aik, mat, i, k);
						if((rtd_cmp_ui(ajk, 0UL) != 0) && (rtd_cmp_ui(aik, 0UL) != 0))
						{
							//ajk = ajk - aji * aik;
							rtd_mul(ctmp, aji, aik);
							rtd_sub(ajk, ajk, ctmp);
							set_tdrsmatrix_ij(mat, j, k, ajk);
							//printf("%ld, %ld, %ld\n", i, j, k);
						}
					}
				}
			}
		}
	}
}

// iLU0_solve: iLU * x = b
void solve_iLU0_tdrsmatrix(TDVector ret, TDRSMatrix ilu, TDVector b)
{
	long int i, j, row_dim, col_dim;
	double ret_i[TDSIZE], ret_j[TDSIZE], ilu_ii[TDSIZE], ilu_ij[TDSIZE], ctmp[TDSIZE];

	row_dim = ilu->row_dim;
	col_dim = ilu->col_dim;

	// ret := b
	subst_tdvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		get_tdrsmatrix_ij(ilu_ii, ilu, i, i);
		subst_tdvector_i(ret_i, ret, i);
		if(rtd_cmp_ui(ilu_ii, 0UL) != 0)
		{
			for(j = 0; j < i; j++)
			{
				get_tdrsmatrix_ij(ilu_ij, ilu, i, j);
				if(rtd_cmp_ui(ilu_ij, 0UL) != 0)
				{
					subst_tdvector_i(ret_j, ret, j);
					//ret_j = ret_j - ilu_ji * ret_i;
					rtd_mul(ctmp, ilu_ij, ret_j);
					rtd_sub(ret_i, ret_i, ctmp);
					//set_tdvector_i(ret, j, &ret_j);
					//printf("f: %ld, %ld\n", i, j);
				}
			}
			set_tdvector_i(ret, i, ret_i);
		}
	}

	// Backward substitution
	for(i = (row_dim - 1); i >= 0; i--)
	{
		get_tdrsmatrix_ij(ilu_ii, ilu, i, i);
		subst_tdvector_i(ret_i, ret, i);
		if(rtd_cmp_ui(ilu_ii, 0UL) != 0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				get_tdrsmatrix_ij(ilu_ij, ilu, i, j);
				if(rtd_cmp_ui(ilu_ij, 0UL) != 0)
				{
					subst_tdvector_i(ret_j, ret, j);
					//ret_i = ret_i - ilu_ij * ret_j;
					rtd_mul(ctmp, ilu_ij, ret_j);
					rtd_sub(ret_i, ret_i, ctmp);
					//set_tdvector_i(ret, i, ret_i);
					//printf("b: %ld, %ld\n", i, j);
				}
			}
			//ret_i /= ilu_ii;
			rtd_div(ret_i, ret_i, ilu_ii);
			set_tdvector_i(ret, i, ret_i);
		}
	}
}

// iLU0t_solve: x^T * iLU = b^T
void solve_iLU0t_tdrsmatrix(TDVector ret, TDRSMatrix ilu, TDVector b)
{
	long int i, j, row_dim, col_dim;
	double ret_i[TDSIZE], ret_j[TDSIZE], ctmp[TDSIZE], ilu_ii[TDSIZE], ilu_ji[TDSIZE];

	row_dim = ilu->row_dim;
	col_dim = ilu->col_dim;

	// ret := b
	subst_tdvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		get_tdrsmatrix_ij(ilu_ii, ilu, i, i);
		subst_tdvector_i(ret_i, ret, i);
		if(rtd_cmp_ui(ilu_ii, 0UL) != 0)
		{
			for(j = 0; j < i; j++)
			{
				get_tdrsmatrix_ij(ilu_ji, ilu, j, i);
				if(rtd_cmp_ui(ilu_ji, 0UL) != 0)
				{
					subst_tdvector_i(ret_j, ret, j);
					//ret_i = ret_i - ilu_ji * ret_j;
					rtd_mul(ctmp, ret_j, ilu_ji);
					rtd_sub(ret_i, ret_i, ctmp);
					//set_cdvector_i(ret, i, ret_i);
				}
			}
			//ret_i /= ilu_ii;
			rtd_div(ret_i, ret_i, ilu_ii);
			set_tdvector_i(ret, i, ret_i);
		}
	}

	// Backward substitution
	for(i = row_dim - 1; i >= 0; i--)
	{
		get_tdrsmatrix_ij(ilu_ii, ilu, i, i);
		subst_tdvector_i(ret_i, ret, i);
		if(rtd_cmp_ui(ilu_ii, 0UL) != 0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				get_tdrsmatrix_ij(ilu_ji, ilu, j, i);
				if(rtd_cmp_ui(ilu_ji, 0UL) != 0)
				{
					subst_tdvector_i(ret_j, ret, j);
					//ret_i = ret_i - ilu_ji * ret_j;
					rtd_mul(ctmp, ret_j, ilu_ji);
					rtd_sub(ret_i, ret_i, ctmp);
					//set_cdvector_i(ret, j, ret_j);
				}
			}
			set_tdvector_i(ret, i, ret_i);
		}
	}
}

// iLU0_solve: iLU * x = b
void solve_iLU0_drsmatrix_tdvec(TDVector ret, DRSMatrix ilu, TDVector b)
{
	long int i, j, row_dim, col_dim;
	double ilu_ii, ilu_ij;
	double ret_i[TDSIZE], ret_j[TDSIZE], ctmp[TDSIZE];

	row_dim = ilu->row_dim;
	col_dim = ilu->col_dim;

	// ret := b
	subst_tdvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		ilu_ii = get_drsmatrix_ij(ilu, i, i);
		subst_tdvector_i(ret_i, ret, i);
		if(fabs(ilu_ii) != 0.0)
		{
			for(j = 0; j < i; j++)
			{
				ilu_ij = get_drsmatrix_ij(ilu, i, j);
				if(fabs(ilu_ij) != 0.0)
				{
					subst_tdvector_i(ret_j, ret, j);
					//ret_j = ret_j - ilu_ji * ret_i;
					rtd_mul_d(ctmp, ret_j, ilu_ij);
					rtd_sub(ret_i, ret_i, ctmp);
					//set_tdvector_i(ret, j, ret_j);
					//printf("f: %ld, %ld\n", i, j);
				}
			}
			set_tdvector_i(ret, i, ret_i);
		}
	}

	// Backward substitution
	for(i = (row_dim - 1); i >= 0; i--)
	{
		ilu_ii = get_drsmatrix_ij(ilu, i, i);
		subst_tdvector_i(ret_i, ret, i);
		if(fabs(ilu_ii) != 0.0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				ilu_ij = get_drsmatrix_ij(ilu, i, j);
				if(fabs(ilu_ij) != 0.0)
				{
					subst_tdvector_i(ret_j, ret, j);
					//ret_i = ret_i - ilu_ij * ret_j;
					//rtd_mul(ctmp, ilu_ij, ret_j);
					rtd_mul_d(ctmp, ret_j, ilu_ij);
					rtd_sub(ret_i, ret_i, ctmp);
					//set_tdvector_i(ret, i, ret_i);
					//printf("b: %ld, %ld\n", i, j);
				}
			}
			//ret_i /= ilu_ii;
			//rtd_div(ret_i, ret_i, ilu_ii);
			rtd_div_d(ret_i, ret_i, ilu_ii);
			set_tdvector_i(ret, i, ret_i);
		}
	}
}

// iLU0t_solve: x^T * iLU = b^T
void solve_iLU0t_drsmatrix_tdvec(TDVector ret, DRSMatrix ilu, TDVector b)
{
	long int i, j, row_dim, col_dim;
	double ilu_ii, ilu_ji;
	double ret_i[TDSIZE], ret_j[TDSIZE], ctmp[TDSIZE];

	row_dim = ilu->row_dim;
	col_dim = ilu->col_dim;

	// ret := b
	subst_tdvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		ilu_ii = get_drsmatrix_ij(ilu, i, i);
		subst_tdvector_i(ret_i, ret, i);
		if(fabs(ilu_ii) != 0.0)
		{
			for(j = 0; j < i; j++)
			{
				ilu_ji = get_drsmatrix_ij(ilu, j, i);
				if(fabs(ilu_ji) != 0.0)
				{
					subst_tdvector_i(ret_j, ret, j);
					//ret_i = ret_i - ilu_ji * ret_j;
					rtd_mul_d(ctmp, ret_j, ilu_ji);
					rtd_sub(ret_i, ret_i, ctmp);
					//set_cdvector_i(ret, i, ret_i);
				}
			}
			//ret_i /= ilu_ii;
			rtd_div_d(ret_i, ret_i, ilu_ii);
			set_tdvector_i(ret, i, ret_i);
		}
	}

	// Backward substitution
	for(i = row_dim - 1; i >= 0; i--)
	{
		ilu_ii = get_drsmatrix_ij(ilu, i, i);
		subst_tdvector_i(ret_i, ret, i);
		if(fabs(ilu_ii) != 0.0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				ilu_ji = get_drsmatrix_ij(ilu, j, i);
				if(fabs(ilu_ji) != 0.0)
				{
					subst_tdvector_i(ret_j, ret, j);
					//ret_i = ret_i - ilu_ji * ret_j;
					rtd_mul_d(ctmp, ret_j, ilu_ji);
					rtd_sub(ret_i, ret_i, ctmp);
					//set_cdvector_i(ret, j, ret_j);
				}
			}
			set_tdvector_i(ret, i, ret_i);
		}
	}
}
