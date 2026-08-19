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
/*------------------------------------------------------------------------------*/
/* SplitMat_A: 2^row_shift[index][i] * (ret_mat[0] + ret_mat[1] + ...) = org_mat */
/*                                                                               */
/* One pass over the non-zeros per split instead of the sweep over all           */
/* row_dim x col_dim positions the old formulation made: the row maximum used to */
/* come from absmax_row_drsmatrix(), which calls get_drsmatrix_ij() for every    */
/* column of every row, and get_drsmatrix_ij() itself walks the rows above to    */
/* find where a row starts.  Materializing the row offsets once (they are what   */
/* made the rows look sequential) turns that into O(nnz) and lets the rows go to */
/* different threads.                                                            */
/*                                                                               */
/* Each row is normalized by a power of two before it is split, so neither the   */
/* threshold nor the slice can leave the double exponent range; row_shift may be */
/* NULL, which asks for the unscaled split.  See oz_scheme.h.                    */
/*------------------------------------------------------------------------------*/
int split_tdrsmatrix_drsmat_ex(DRSMatrix ret_mat[], long int row_shift[], int num_div, TDRSMatrix org_mat)
{
	long int i, j, index, row_dim, *row_start;
	long int num_digits = 53; // IEEE double prec.
	int real_num_div, num_threads;
	double mu_total;
	TDRSMatrix tmp_org_mat;
	DRSMatrix own_ret_mat = NULL, in_ret_mat;
	double *power2;
	long int *shift;

	row_dim = org_mat->row_dim;

	power2 = (double *)calloc((size_t)row_dim, sizeof(double));
	row_start = bnc_oz_sp_row_start(org_mat->real_nzero_col_dim, row_dim);
	if(power2 == NULL || row_start == NULL)
	{
		fprintf(stderr, "ERROR: split_tdrsmatrix_drsmat: cannot allocate\n");
		free(power2);
		free(row_start);
		return 0;
	}

	if(ret_mat == NULL)
	{
		own_ret_mat = init_set_drsmatrix_tdrsmat(org_mat);
		in_ret_mat = own_ret_mat;
	}
	else
		in_ret_mat = ret_mat[0];

	// tmp_org_mat := org_mat; it always holds the part not split off yet
	tmp_org_mat = init_set_tdrsmatrix(org_mat);

	num_threads = bnc_oz_get_num_threads();

	real_num_div = 0;
	for(index = 0; index < num_div; index++)
	{
		if(ret_mat != NULL)
			in_ret_mat = ret_mat[index];
		shift = (row_shift != NULL) ? (row_shift + (size_t)index * (size_t)row_dim) : NULL;

		// pass 1: the scaled double image of every row and its maximum
		mu_total = 0.0;

#ifdef _OPENMP
		#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 16) private(i, j) reduction(+:mu_total)
#endif // _OPENMP
		for(i = 0; i < row_dim; i++)
		{
			long int base = row_start[i], nzero_dim = tmp_org_mat->nzero_col_dim[i];
			long int real_nzero_dim = tmp_org_mat->real_nzero_col_dim[i];
			const double *org_row = tmp_org_mat->element[0] + base;
			double *ret_row = in_ret_mat->element + base;
			double mu = 0.0, abs_org_ij;
			long int sigma = 0;

			for(j = 0; j < nzero_dim; j++)
			{
				abs_org_ij = fabs(org_row[j]);
				if(abs_org_ij > mu)
					mu = abs_org_ij;
			}

			if(shift != NULL)
			{
				sigma = bnc_oz_exp2_d(mu);
				if(sigma < BNC_OZ_MIN_SCALED_EXP)
					sigma = 0; // shifting back would not be exact down there
				shift[i] = sigma;
			}

			if(sigma != 0)
			{
				for(j = 0; j < nzero_dim; j++)
					ret_row[j] = bnc_oz_ldexp(org_row[j], -sigma);

				mu = bnc_oz_ldexp(mu, -sigma);
			}
			else
			{
				for(j = 0; j < nzero_dim; j++)
					ret_row[j] = org_row[j];
			}

			// the per-row SIMD padding is never read as data, but keep it zero
			for(j = nzero_dim; j < real_nzero_dim; j++)
				ret_row[j] = 0.0;

			// s[i, j] = 2^t_exp, with the row's own number of non-zeros
			power2[i] = pow(2.0, ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(tmp_org_mat->nzero_col_dim[i]))) / 2.0));
			mu_total += mu;
		}

		// nothing left to split
		if(mu_total == 0.0) break;

		// pass 2: high := (mat + s) - s, and mat := mat - 2^shift[i] * high
#ifdef _OPENMP
		#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 16) private(i, j)
#endif // _OPENMP
		for(i = 0; i < row_dim; i++)
		{
			long int base = row_start[i], nzero_dim = tmp_org_mat->nzero_col_dim[i];
			double *ret_row = in_ret_mat->element + base;
			double *org_row[TDSIZE];
			double s = power2[i], high_ij;
			double org_ij[TDSIZE], high_td[TDSIZE], rest_ij[TDSIZE];
			long int sigma = (shift != NULL) ? shift[i] : 0;
			int comp;

			for(comp = 0; comp < TDSIZE; comp++)
			{
				org_row[comp] = tmp_org_mat->element[comp] + base;
				high_td[comp] = 0.0;
			}

			for(j = 0; j < nzero_dim; j++)
			{
				// (x + s) - s keeps the leading bits of x; valid under the IEEE
				// semantics this library is compiled with (no -ffast-math)
				high_ij = ret_row[j] + s;
				high_ij = high_ij - s;
				ret_row[j] = high_ij;

				for(comp = 0; comp < TDSIZE; comp++)
					org_ij[comp] = org_row[comp][j];
				high_td[0] = (sigma != 0) ? bnc_oz_ldexp(high_ij, sigma) : high_ij;

				rtd_sub(rest_ij, org_ij, high_td);

				for(comp = 0; comp < TDSIZE; comp++)
					org_row[comp][j] = rest_ij[comp];
			}
		}

		real_num_div = index + 1;
	}

	free(power2);
	free(row_start);
	free_tdrsmatrix(tmp_org_mat);
	if(own_ret_mat != NULL)
		free_drsmatrix(own_ret_mat);

	return real_num_div;
}

// SplitMat_A without the scaling; kept for callers that cannot apply a scale factor
int split_tdrsmatrix_drsmat(DRSMatrix ret_mat[], int num_div, TDRSMatrix org_mat)
{
	return split_tdrsmatrix_drsmat_ex(ret_mat, NULL, num_div, org_mat);
}

// Matrix-Vector multiplication based on Ozaki scheme
/*------------------------------------------------------------------------------*/
/* Matrix-Vector multiplication based on Ozaki scheme                            */
/*                                                                               */
/* ret = sum_{p,q} 2^(sa[p][i] + sv[q]) * (slice_a[p] * slice_v[q]).  The rows   */
/* of ret are cut into blocks; a thread takes a block at a time and runs every   */
/* slice pair for it, accumulating in TD on the spot, so both the sparse      */
/* products and the multi-component accumulation are parallel.  Blocks are       */
/* disjoint and each element still sums its slice products in the original       */
/* order, so the result does not depend on the number of threads.                */
/*                                                                               */
/* OpenBLAS has no sparse BLAS to hand this to -- it ships dense BLAS and        */
/* LAPACK only -- so the products use the library's own CSR kernel.              */
/*------------------------------------------------------------------------------*/
void mul_tdrsmatrix_tdvec_oz(TDVector ret, TDRSMatrix a, int max_num_div_a, TDVector vb, int max_num_div_vb)
{
    int i;
    int real_num_div_a, real_num_div_vb, num_threads;
    long int row_dim = a->row_dim, block_rows, num_blocks, *row_start;
    long int *row_shift, *vec_shift;
    DRSMatrix *div_a;
    DVector *div_vb;
    double *block_buf = NULL;

    div_a = (DRSMatrix *)calloc(max_num_div_a, sizeof(DRSMatrix));
    div_vb = (DVector *)calloc(max_num_div_vb, sizeof(DVector));
    row_shift = (long int *)calloc((size_t)max_num_div_a * (size_t)row_dim, sizeof(long int));
    vec_shift = (long int *)calloc((size_t)max_num_div_vb, sizeof(long int));
    row_start = bnc_oz_sp_row_start(a->real_nzero_col_dim, row_dim);
    if(div_a == NULL || div_vb == NULL || row_shift == NULL || vec_shift == NULL || row_start == NULL)
    {
        fprintf(stderr, "ERROR: mul_tdrsmatrix_tdvec_oz: cannot allocate\n");
        free(div_a); free(div_vb); free(row_shift); free(vec_shift); free(row_start);
        return;
    }

    for(i = 0; i < max_num_div_a; i++)
        div_a[i] = init_set_drsmatrix_tdrsmat(a);
    for(i = 0; i < max_num_div_vb; i++)
        div_vb[i] = init_dvector(vb->dim); // vb->dim, not ret->dim: they differ when a is not square

    real_num_div_a = split_tdrsmatrix_drsmat_ex(div_a, row_shift, max_num_div_a, a);
    real_num_div_vb = split_tdvector_dvec_ex(div_vb, vec_shift, max_num_div_vb, vb);

    set0_tdvector(ret);

    num_threads = bnc_oz_get_num_threads();
    block_rows = bnc_oz_block_rows_for(row_dim, num_threads);
    num_blocks = (row_dim + block_rows - 1) / block_rows;

    block_buf = (double *)malloc((size_t)num_threads * (size_t)block_rows * sizeof(double));
    if(block_buf == NULL)
    {
        num_threads = 1;
        block_rows = row_dim;
        num_blocks = 1;
        block_buf = (double *)malloc((size_t)row_dim * sizeof(double));
    }

    if(block_buf != NULL)
    {
#ifdef _OPENMP
        #pragma omp parallel num_threads(num_threads)
#endif // _OPENMP
        {
            long int blk, first_row, num_rows, ii, shift_a;
            int div_i, div_j, comp;
            double *buf;
            double ret_i[TDSIZE], add_i[TDSIZE], sum_i[TDSIZE];

            for(comp = 0; comp < TDSIZE; comp++)
                add_i[comp] = 0.0;

#ifdef _OPENMP
            buf = block_buf + (size_t)omp_get_thread_num() * (size_t)block_rows;
            #pragma omp for schedule(dynamic, 1)
#else // _OPENMP
            buf = block_buf;
#endif // _OPENMP
            for(blk = 0; blk < num_blocks; blk++)
            {
                first_row = blk * block_rows;
                num_rows = ((row_dim - first_row) < block_rows) ? (row_dim - first_row) : block_rows;

                for(div_i = 0; div_i < real_num_div_a; div_i++)
                {
                    for(div_j = 0; div_j < real_num_div_vb; div_j++)
                    {
                        bnc_oz_sp_dcsr_block(buf, div_a[div_i], row_start, first_row, num_rows, div_vb[div_j]);

                        for(ii = 0; ii < num_rows; ii++)
                        {
                            if(buf[ii] == 0.0)
                                continue;

                            shift_a = row_shift[(size_t)div_i * (size_t)row_dim + first_row + ii];

                            for(comp = 0; comp < TDSIZE; comp++)
                                ret_i[comp] = ret->element[comp][first_row + ii];
                            add_i[0] = bnc_oz_ldexp(buf[ii], shift_a + vec_shift[div_j]);

                            rtd_add(sum_i, ret_i, add_i);

                            for(comp = 0; comp < TDSIZE; comp++)
                                ret->element[comp][first_row + ii] = sum_i[comp];
                        }
                    }
                }
            }
        }

        free(block_buf);
    }

    for(i = 0; i < max_num_div_a; i++)
        free_drsmatrix(div_a[i]);
    for(i = 0; i < max_num_div_vb; i++)
        free_dvector(div_vb[i]);

    free(div_a);
    free(div_vb);
    free(row_shift);
    free(vec_shift);
    free(row_start);
}

// 2024-08-04 (Sun) T.Kouya
// SplitMat_B
// return real_num_div
/*------------------------------------------------------------------------------*/
/* SplitMat_B: 2^col_shift[index][j] * (ret_mat[0] + ...) = org_mat, the         */
/* threshold and the scale taken per column.                                     */
/*                                                                               */
/* The columns of a CRS matrix are scattered, so the maxima are collected in one */
/* pass over the non-zeros into a per-thread column vector and combined          */
/* afterwards.  The old code called absmax_col_drsmatrix() once per column --    */
/* each of which walked every row -- and then set_drsmatrix_ij() for all         */
/* row_dim x col_dim positions, which is quadratic in the dimension however few  */
/* non-zeros there are.                                                          */
/*------------------------------------------------------------------------------*/
int split_tdrsmatrix_t_drsmat_ex(DRSMatrix ret_mat[], long int col_shift[], int num_div, TDRSMatrix org_mat)
{
	long int i, j, index, row_dim, col_dim, *row_start;
	int real_num_div, num_digits = 53, num_threads, thread; // IEEE double prec.
	double mu_total, tail_exp;
	TDRSMatrix tmp_org_mat;
	DRSMatrix own_ret_mat = NULL, in_ret_mat;
	double *power2, *mu_local;
	long int *shift;

	row_dim = org_mat->row_dim;
	col_dim = org_mat->col_dim;

	num_threads = bnc_oz_get_num_threads();

	power2 = (double *)calloc((size_t)col_dim, sizeof(double));
	row_start = bnc_oz_sp_row_start(org_mat->real_nzero_col_dim, row_dim);
	mu_local = (double *)calloc((size_t)num_threads * (size_t)col_dim, sizeof(double));
	if(mu_local == NULL) // retry single-threaded rather than give up
	{
		num_threads = 1;
		mu_local = (double *)calloc((size_t)col_dim, sizeof(double));
	}
	if(power2 == NULL || row_start == NULL || mu_local == NULL)
	{
		fprintf(stderr, "ERROR: split_tdrsmatrix_t_drsmat: cannot allocate\n");
		free(power2);
		free(row_start);
		free(mu_local);
		return 0;
	}

	if(ret_mat == NULL)
	{
		own_ret_mat = init_set_drsmatrix_tdrsmat(org_mat);
		in_ret_mat = own_ret_mat;
	}
	else
		in_ret_mat = ret_mat[0];

	tmp_org_mat = init_set_tdrsmatrix(org_mat);

	// the column-independent half of t_exp
	tail_exp = ceil(((double)num_digits + DLOG2((double)(row_dim + 1))) / 2.0);

	real_num_div = 0;
	for(index = 0; index < num_div; index++)
	{
		if(ret_mat != NULL)
			in_ret_mat = ret_mat[index];
		shift = (col_shift != NULL) ? (col_shift + (size_t)index * (size_t)col_dim) : NULL;

		for(i = 0; i < (long int)num_threads * col_dim; i++)
			mu_local[i] = 0.0;

		// pass 1: the column maxima, in one sweep over the non-zeros
#ifdef _OPENMP
		#pragma omp parallel num_threads(num_threads) private(i, j)
#endif // _OPENMP
		{
			double *local_mu;
#ifdef _OPENMP
			local_mu = mu_local + (size_t)omp_get_thread_num() * (size_t)col_dim;
			#pragma omp for schedule(dynamic, 16)
#else // _OPENMP
			local_mu = mu_local;
#endif // _OPENMP
			for(i = 0; i < row_dim; i++)
			{
				const double *org_row = tmp_org_mat->element[0] + row_start[i];
				const long int *index_row = tmp_org_mat->nzero_index[i];
				long int nzero_dim = tmp_org_mat->nzero_col_dim[i];
				double abs_org_ij;

				for(j = 0; j < nzero_dim; j++)
				{
					abs_org_ij = fabs(org_row[j]);
					if(abs_org_ij > local_mu[index_row[j]])
						local_mu[index_row[j]] = abs_org_ij;
				}
			}
		}

		// combine the per-thread maxima, pick the column scale and the threshold
		mu_total = 0.0;
		for(j = 0; j < col_dim; j++)
		{
			double mu = mu_local[j];
			long int sigma = 0;

			for(thread = 1; thread < num_threads; thread++)
			{
				if(mu_local[(size_t)thread * (size_t)col_dim + j] > mu)
					mu = mu_local[(size_t)thread * (size_t)col_dim + j];
			}

			if(shift != NULL)
			{
				sigma = bnc_oz_exp2_d(mu);
				if(sigma < BNC_OZ_MIN_SCALED_EXP)
					sigma = 0;
				shift[j] = sigma;

				if(sigma != 0)
					mu = bnc_oz_ldexp(mu, -sigma);
			}

			power2[j] = pow(2.0, ceil(DLOG2(mu)) + tail_exp);
			mu_total += mu;
		}

		if(mu_total == 0.0) break;

		// pass 2: scale, high := (mat + s) - s, and mat := mat - 2^shift[j] * high
#ifdef _OPENMP
		#pragma omp parallel for num_threads(num_threads) schedule(dynamic, 16) private(i, j)
#endif // _OPENMP
		for(i = 0; i < row_dim; i++)
		{
			long int base = row_start[i], nzero_dim = tmp_org_mat->nzero_col_dim[i];
			long int real_nzero_dim = tmp_org_mat->real_nzero_col_dim[i];
			const long int *index_row = tmp_org_mat->nzero_index[i];
			double *ret_row = in_ret_mat->element + base;
			double *org_row[TDSIZE];
			double s, high_ij, scaled_ij;
			double org_ij[TDSIZE], high_td[TDSIZE], rest_ij[TDSIZE];
			long int sigma;
			int comp;

			for(comp = 0; comp < TDSIZE; comp++)
			{
				org_row[comp] = tmp_org_mat->element[comp] + base;
				high_td[comp] = 0.0;
			}

			for(j = 0; j < nzero_dim; j++)
			{
				sigma = (shift != NULL) ? shift[index_row[j]] : 0;
				s = power2[index_row[j]];

				scaled_ij = (sigma != 0) ? bnc_oz_ldexp(org_row[0][j], -sigma) : org_row[0][j];

				high_ij = scaled_ij + s;
				high_ij = high_ij - s;
				ret_row[j] = high_ij;

				for(comp = 0; comp < TDSIZE; comp++)
					org_ij[comp] = org_row[comp][j];
				high_td[0] = (sigma != 0) ? bnc_oz_ldexp(high_ij, sigma) : high_ij;

				rtd_sub(rest_ij, org_ij, high_td);

				for(comp = 0; comp < TDSIZE; comp++)
					org_row[comp][j] = rest_ij[comp];
			}

			for(j = nzero_dim; j < real_nzero_dim; j++)
				ret_row[j] = 0.0;
		}

		real_num_div = index + 1;
	}

	free(power2);
	free(row_start);
	free(mu_local);
	free_tdrsmatrix(tmp_org_mat);
	if(own_ret_mat != NULL)
		free_drsmatrix(own_ret_mat);

	return real_num_div;
}

// SplitMat_B without the scaling; kept for callers that cannot apply a scale factor
int split_tdrsmatrix_t_drsmat(DRSMatrix ret_mat[], int num_div, TDRSMatrix org_mat)
{
	return split_tdrsmatrix_t_drsmat_ex(ret_mat, NULL, num_div, org_mat);
}

// 2024-08-02(Fri) T.Kouya
// Transposed Matrix-Vector multiplication based on Ozaki scheme
/*------------------------------------------------------------------------------*/
/* Transposed Matrix-Vector multiplication based on Ozaki scheme                  */
/*                                                                               */
/* ret = sum_{p,q} 2^(sa[p][j] + sv[q]) * (slice_a[p]^T * slice_v[q]).  A^T * v  */
/* scatters into the result, so the rows are split over the threads and each     */
/* thread accumulates into its own column vector, which is then reduced.  The    */
/* partial sums are exact (the split is what makes every slice product exact in  */
/* double, whatever the summation order), so the reduction changes nothing but   */
/* the speed.                                                                    */
/*------------------------------------------------------------------------------*/
void mul_tdrsmatrixt_tdvec_oz(TDVector ret, TDRSMatrix a, int max_num_div_a, TDVector vb, int max_num_div_vb)
{
    int i, j;
    int real_num_div_a, real_num_div_vb, num_threads;
    long int row_dim = a->row_dim, col_dim = a->col_dim, block_rows, num_blocks, *row_start;
    long int *col_shift, *vec_shift, jj;
    DRSMatrix *div_a;
    DVector *div_vb;
    double *acc_buf = NULL;

    div_a = (DRSMatrix *)calloc(max_num_div_a, sizeof(DRSMatrix));
    div_vb = (DVector *)calloc(max_num_div_vb, sizeof(DVector));
    col_shift = (long int *)calloc((size_t)max_num_div_a * (size_t)col_dim, sizeof(long int));
    vec_shift = (long int *)calloc((size_t)max_num_div_vb, sizeof(long int));
    row_start = bnc_oz_sp_row_start(a->real_nzero_col_dim, row_dim);
    if(div_a == NULL || div_vb == NULL || col_shift == NULL || vec_shift == NULL || row_start == NULL)
    {
        fprintf(stderr, "ERROR: mul_tdrsmatrixt_tdvec_oz: cannot allocate\n");
        free(div_a); free(div_vb); free(col_shift); free(vec_shift); free(row_start);
        return;
    }

    for(i = 0; i < max_num_div_a; i++)
        div_a[i] = init_set_drsmatrix_tdrsmat(a);
    for(i = 0; i < max_num_div_vb; i++)
        div_vb[i] = init_dvector(vb->dim);

    real_num_div_a = split_tdrsmatrix_t_drsmat_ex(div_a, col_shift, max_num_div_a, a);
    real_num_div_vb = split_tdvector_dvec_ex(div_vb, vec_shift, max_num_div_vb, vb);

    set0_tdvector(ret);

    // one accumulator per thread; fall back to fewer threads rather than ask
    // for an unreasonable amount of memory on a very wide matrix
    num_threads = bnc_oz_get_num_threads();
    while(num_threads > 1 && (size_t)num_threads * (size_t)col_dim * sizeof(double) > (size_t)256 * 1024 * 1024)
        num_threads /= 2;

    acc_buf = (double *)calloc((size_t)num_threads * (size_t)col_dim, sizeof(double));
    if(acc_buf == NULL)
    {
        num_threads = 1;
        acc_buf = (double *)calloc((size_t)col_dim, sizeof(double));
    }

    block_rows = bnc_oz_block_rows_for(row_dim, num_threads);
    num_blocks = (row_dim + block_rows - 1) / block_rows;

    if(acc_buf != NULL)
    {
        for(i = 0; i < real_num_div_a; i++)
        {
            for(j = 0; j < real_num_div_vb; j++)
            {
                const long int *shift_a = col_shift + (size_t)i * (size_t)col_dim;

#ifdef _OPENMP
                #pragma omp parallel for num_threads(num_threads) schedule(static) private(jj)
#endif // _OPENMP
                for(jj = 0; jj < (long int)num_threads * col_dim; jj++)
                    acc_buf[jj] = 0.0;

#ifdef _OPENMP
                #pragma omp parallel num_threads(num_threads)
#endif // _OPENMP
                {
                    long int blk, first_row, num_rows;
                    double *acc;

#ifdef _OPENMP
                    acc = acc_buf + (size_t)omp_get_thread_num() * (size_t)col_dim;
                    #pragma omp for schedule(dynamic, 1)
#else // _OPENMP
                    acc = acc_buf;
#endif // _OPENMP
                    for(blk = 0; blk < num_blocks; blk++)
                    {
                        first_row = blk * block_rows;
                        num_rows = ((row_dim - first_row) < block_rows) ? (row_dim - first_row) : block_rows;

                        bnc_oz_sp_dcsrt_block(acc, div_a[i], row_start, first_row, num_rows, div_vb[j]);
                    }
                }

                // the partial sums are exact, so reducing them in a fixed
                // order costs nothing in accuracy and the columns are
                // independent of each other
#ifdef _OPENMP
                #pragma omp parallel for num_threads(num_threads) schedule(static) private(jj)
#endif // _OPENMP
                for(jj = 0; jj < col_dim; jj++)
                {
                    double total = acc_buf[jj];
                    double ret_j[TDSIZE], add_j[TDSIZE], sum_j[TDSIZE];
                    int comp, th;

                    for(th = 1; th < num_threads; th++)
                        total += acc_buf[(size_t)th * (size_t)col_dim + jj];

                    if(total == 0.0)
                        continue;

                    for(comp = 0; comp < TDSIZE; comp++)
                    {
                        ret_j[comp] = ret->element[comp][jj];
                        add_j[comp] = 0.0;
                    }
                    add_j[0] = bnc_oz_ldexp(total, shift_a[jj] + vec_shift[j]);

                    rtd_add(sum_j, ret_j, add_j);

                    for(comp = 0; comp < TDSIZE; comp++)
                        ret->element[comp][jj] = sum_j[comp];
                }
            }
        }

        free(acc_buf);
    }

    for(i = 0; i < max_num_div_a; i++)
        free_drsmatrix(div_a[i]);
    for(i = 0; i < max_num_div_vb; i++)
        free_dvector(div_vb[i]);

    free(div_a);
    free(div_vb);
    free(col_shift);
    free(vec_shift);
    free(row_start);
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

/*------------------------------------------------------------------------------*/
/* Sparse-matrix times dense-matrix multiplication based on Ozaki scheme         */
/*                                                                               */
/* ret = sum_{p,q} 2^(sa[p][i] + sb[q][j]) * (slice_a[p] * slice_b[q])[i][j],    */
/* with A split by rows (SplitMat_A on the sparse operand) and B by columns      */
/* (SplitMat_B on the dense one), exactly as in mul_tdmatrix_oz().            */
/*                                                                               */
/* The rows of ret are cut into blocks and one thread takes a block at a time,   */
/* running every slice pair for it and accumulating in TD on the spot, so    */
/* the block of ret stays hot in cache and no thread ever touches another        */
/* thread's rows.  Slice pairs with p + q >= real_num_div_b contribute below the */
/* accuracy the split was asked for and are skipped, as in the dense kernel.     */
/*                                                                               */
/* There is no sparse BLAS in OpenBLAS to hand the products to, so they use the  */
/* library's own CSR-times-dense kernel.                                         */
/*------------------------------------------------------------------------------*/
void mul_tdrsmatrix_tdmat_oz(TDMatrix ret, TDRSMatrix a, int max_num_div_a, TDMatrix b, int max_num_div_b)
{
    int i, j;
    int real_num_div_a, real_num_div_b, num_threads;
    long int row_dim = a->row_dim, col_dim = ret->col_dim, mid_dim = a->col_dim;
    long int block_rows, num_blocks, *row_start;
    long int *row_shift, *col_shift;
    DRSMatrix *div_a;
    DMatrix *div_b;
    double *block_buf = NULL;

    if(mid_dim != b->row_dim)
    {
        fprintf(stderr, "ERROR: mul_tdrsmatrix_tdmat_oz mid_dim(a, b) = (%ld, %ld)!\n", mid_dim, b->row_dim);
        return;
    }

    div_a = (DRSMatrix *)calloc(max_num_div_a, sizeof(DRSMatrix));
    div_b = (DMatrix *)calloc(max_num_div_b, sizeof(DMatrix));
    row_shift = (long int *)calloc((size_t)max_num_div_a * (size_t)row_dim, sizeof(long int));
    col_shift = (long int *)calloc((size_t)max_num_div_b * (size_t)col_dim, sizeof(long int));
    if(div_a == NULL || div_b == NULL || row_shift == NULL || col_shift == NULL)
    {
        fprintf(stderr, "ERROR: mul_tdrsmatrix_tdmat_oz: cannot allocate\n");
        free(div_a); free(div_b); free(row_shift); free(col_shift);
        return;
    }

    for(i = 0; i < max_num_div_a; i++)
        div_a[i] = init_set_drsmatrix_tdrsmat(a);
    for(i = 0; i < max_num_div_b; i++)
        div_b[i] = init_dmatrix(mid_dim, col_dim);

    row_start = bnc_oz_sp_row_start(div_a[0]->real_nzero_col_dim, row_dim);
    if(row_start == NULL)
    {
        fprintf(stderr, "ERROR: mul_tdrsmatrix_tdmat_oz: cannot allocate\n");
        return;
    }

    real_num_div_a = split_tdrsmatrix_drsmat_ex(div_a, row_shift, max_num_div_a, a);
    real_num_div_b = split_tdmatrix_t_dmat_ex(div_b, col_shift, max_num_div_b, b);

    set0_tdmatrix(ret);

    num_threads = bnc_oz_get_num_threads();
    block_rows = bnc_oz_block_rows_for(row_dim, num_threads);
    num_blocks = (row_dim + block_rows - 1) / block_rows;

    block_buf = (double *)malloc((size_t)num_threads * (size_t)block_rows * (size_t)col_dim * sizeof(double));
    if(block_buf == NULL)
    {
        num_threads = 1;
        block_rows = bnc_oz_block_rows_for(row_dim, 1);
        num_blocks = (row_dim + block_rows - 1) / block_rows;
        block_buf = (double *)malloc((size_t)block_rows * (size_t)col_dim * sizeof(double));
    }

    if(block_buf != NULL)
    {
#ifdef _OPENMP
        #pragma omp parallel num_threads(num_threads)
#endif // _OPENMP
        {
            long int blk, first_row, num_rows, ii, jj, shift_a;
            int div_i, div_j;
            double *buf;
            double *ret_row[TDSIZE];
            double ret_ij[TDSIZE], add_ij[TDSIZE], sum_ij[TDSIZE];
            int comp;

            for(comp = 0; comp < TDSIZE; comp++)
                add_ij[comp] = 0.0;

#ifdef _OPENMP
            buf = block_buf + (size_t)omp_get_thread_num() * (size_t)block_rows * (size_t)col_dim;
            #pragma omp for schedule(dynamic, 1)
#else // _OPENMP
            buf = block_buf;
#endif // _OPENMP
            for(blk = 0; blk < num_blocks; blk++)
            {
                first_row = blk * block_rows;
                num_rows = ((row_dim - first_row) < block_rows) ? (row_dim - first_row) : block_rows;

                for(div_i = 0; div_i < real_num_div_a; div_i++)
                {
                    for(div_j = 0; div_j < real_num_div_b - div_i; div_j++)
                    {
                        const long int *shift_b = col_shift + (size_t)div_j * (size_t)col_dim;

                        bnc_oz_sp_dcsr_dmat_block(buf, col_dim, div_a[div_i], row_start, first_row, num_rows, div_b[div_j]);

                        for(ii = 0; ii < num_rows; ii++)
                        {
                            const double *buf_row = buf + ii * col_dim;

                            shift_a = row_shift[(size_t)div_i * (size_t)row_dim + first_row + ii];

                            for(comp = 0; comp < TDSIZE; comp++)
                                ret_row[comp] = ret->element[comp] + (first_row + ii) * ret->real_col_dim;

                            for(jj = 0; jj < col_dim; jj++)
                            {
                                for(comp = 0; comp < TDSIZE; comp++)
                                    ret_ij[comp] = ret_row[comp][jj];
                                add_ij[0] = bnc_oz_ldexp(buf_row[jj], shift_a + shift_b[jj]);

                                rtd_add(sum_ij, ret_ij, add_ij);

                                for(comp = 0; comp < TDSIZE; comp++)
                                    ret_row[comp][jj] = sum_ij[comp];
                            }
                        }
                    }
                }
            }

        }

        free(block_buf);
    }

    for(i = 0; i < max_num_div_a; i++)
        free_drsmatrix(div_a[i]);
    for(i = 0; i < max_num_div_b; i++)
        free_dmatrix(div_b[i]);

    free(div_a);
    free(div_b);
    free(row_shift);
    free(col_shift);
    free(row_start);
}
