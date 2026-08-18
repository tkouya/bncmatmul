/********************************************************************************/
/*                                                                              */
/* sparse_double.c : Sparse Matrix and Vector Library (Double Precision)        */
/* Copyright (c) 2006-2024 Tomonori Kouya, All rights reserved.                 */
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
#ifdef WIN32
#include <windows.h>
#else // WIN32
#include <sys/times.h>
#include <time.h>
#endif // WIN32

#include "bncsparse.h"

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
/*      zero_element = 0.0   */

/* initialize DRSMatrix */
DRSMatrix init_drsmatrix(long int row_dim, long int *nzero_col_dim, long int nzero_total_num)
{
	DRSMatrix ret;
	long int i, j;

	ret = (DRSMatrix)malloc(sizeof(drsmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "Cannot allocate DRSMatrix\n");
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
	ret->zero_element = 0.0;

	/* allocate nzero_index */
	//printf("%ld %ld %ld %ld\n", ret->row_dim, ret->col_dim, ret->nzero_total_num, sizeof(long int *) * row_dim);
	//ret->nzero_col_dim = (long int *)malloc(sizeof(long int *) * row_dim);
	//ret->nzero_row_dim = (long int *)malloc(sizeof(long int *) * ret->col_dim);
	// Fix! 2024-04-25(Thu) T.Kouya
	ret->nzero_col_dim = (long int *)calloc(ret->row_dim, sizeof(long int));
	ret->nzero_row_dim = (long int *)calloc(ret->col_dim, sizeof(long int));
	//ret->nzero_index = (long int **)malloc(sizeof(long int *) * row_dim);
	ret->nzero_index = (long int **)calloc(ret->row_dim, sizeof(long int *));
	ret->real_nzero_col_dim = (long int *)calloc(ret->row_dim, sizeof(long int));
	if(ret->nzero_index == NULL)
	{
		fprintf(stderr, "Cannot allocate DRSMatrix(nzero_index!)\n");
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
			fprintf(stderr, "Cannot allocate DRSMatrix(nzero_index[%ld]!)\n", i);
			return NULL;
		}
		for(j = 0; j < nzero_col_dim[i]; j++)
			ret->nzero_index[i][j] = EMPTY;

		ret->real_nzero_total_num += ret->real_nzero_col_dim[i];
		//printf("real_col_dim[%ld] = %ld\n", i, ret->real_nzero_col_dim[i]);
	}
	//printf("real_total_num = %ld\n", ret->real_nzero_total_num);

	/* allocate element */
	//ret->element = (double *)malloc(sizeof(double) * nzero_total_num);
	//ret->element = (double *)calloc(nzero_total_num, sizeof(double));
	ret->element = (double *)BNC_CALLOC(ret->real_nzero_total_num, sizeof(double));
	if(ret->element == NULL)
	{
		fprintf(stderr, "Cannot allocate DRSMatrix(element!)\n");
		return NULL;
	}

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();
	for(i = 0; i < ret->real_nzero_total_num; i += _BNC_D_WIDTH)
		_mm256_store_pd(&(ret->element[i]), zero4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();
	for(i = 0; i < ret->real_nzero_total_num; i += _BNC_D_WIDTH)
		_mm512_store_pd(&(ret->element[i]), zero8);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2
	{
		long _vl = (long)svcntd();
		svfloat64_t zerov = svdup_n_f64(0.0);
		for(i = 0; i < ret->real_nzero_total_num; i += _vl)
		{
			svbool_t pg = svwhilelt_b64_s64((int64_t)i,
			                                (int64_t)ret->real_nzero_total_num);
			svst1_f64(pg, &(ret->element[i]), zerov);
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM Neon
	float64x2_t zero2;

	zero2 = vdupq_n_f64(0.0);
	for(i = 0; i < ret->real_nzero_total_num; i += 2)
		vst1q_f64(&(ret->element[i]), zero2);



#else // __AVX2__
	for(i = 0; i < nzero_total_num; i++)
	{
		//*(ret->element + i) = 0.0;
		ret->element[i] = 0.0;
	}
#endif // __AVX2__

	return ret;
}

/* Clear DRSMatrix */
void free_drsmatrix(DRSMatrix mat)
{
	long int i;

	free(mat->element);
	free(mat->nzero_col_dim);
	free(mat->nzero_row_dim);

	for(i = 0; i < mat->row_dim; i++)
		free(mat->nzero_index[i]);

	free(mat->nzero_index);

	free(mat);
}

/* set nzero_row_dim automatically */
void set_nzero_row_dim(DRSMatrix mat)
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

/* Print DRSMatrix */
void print_drsmatrix(DRSMatrix mat)
{
	long int i, j, total_index;

	if(mat == NULL)
		fprintf(stderr, "ERROR!\n");

	total_index = 0;
	for(i = 0; i < mat->row_dim; i++)
	{
		// Fix!: 2011-08-29 by T.Kouya
		if(mat->nzero_col_dim[i] >= 1)
		{
			printf("%5ld: ", i);
			//for(j = 0; j < mat->nzero_col_dim[i] - 1 ; j++)
			for(j = 0; j < mat->real_nzero_col_dim[i]; j++)
			{
				if(j < mat->nzero_col_dim[i])
					printf("%ld->%e, ", mat->nzero_index[i][j], mat->element[total_index]);

				total_index++;
			}
			printf("\n");
		}
	}
	return;
}

/* Dense Matrix := Sparse Matrix */
void set_dmatrix_drsmatrix(DMatrix ret, DRSMatrix spmat)
{
	long int i, j, total_index;

	if((ret == NULL) || (spmat == NULL))
	{
		fprintf(stderr, "set_dmatrix_drsmatrix: ERROR!\n");
		return;
	}

	// ret := 0
	set0_dmatrix(ret);

	total_index = 0;
	for(i = 0; i < spmat->row_dim; i++)
	{
		//for(j = 0; j < spmat->nzero_col_dim[i]; j++)
		for(j = 0; j < spmat->real_nzero_col_dim[i]; j++)
		{
			//printf("%5ld: ", i);
			//printf("%ld->%e, ", spmat->nzero_index[i][j], spmat->element[total_index++]);
			if(j < spmat->nzero_col_dim[i])
				set_dmatrix_ij(ret, i, spmat->nzero_index[i][j], spmat->element[total_index]);

			total_index++;
		}
	}
	return;
}

/* initialize and substitute DRSMatrix from DMatrix */
DRSMatrix init_set_drsmatrix_dmatrix(DMatrix org_mat)
{
	long int i, j;
	long int nzero_total_num, total_index, j_index;
	long int *ptr_nzero_col_dim;
	DRSMatrix ret;

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
			if(get_dmatrix_ij(org_mat, i, j) != 0.0)
			{
				/* get nzero_row_dim, nzero_col_dim, nzero_total_num */
				nzero_total_num++;
				*(ptr_nzero_col_dim + i) += 1;
			}
		}
	}

	// initialize
	ret = init_drsmatrix(org_mat->row_dim, ptr_nzero_col_dim, nzero_total_num);

	/* Read org_mat (2nd) */
	total_index = 0;
	for(i = 0; i < ret->row_dim; i++)
	{
		j_index = 0;
		for(j = 0; j < ret->col_dim; j++)
		{
			if(get_dmatrix_ij(org_mat, i, j) != 0.0)
			{
				ret->nzero_index[i][j_index] =  j;
				*(ret->element + total_index) = get_dmatrix_ij(org_mat, i, j);
				total_index += 1;
				j_index += 1;
			}
		}
		total_index += ret->real_nzero_col_dim[i] - ret->nzero_col_dim[i]; /* skip per-row SIMD padding (zeros) */
	}

	free(ptr_nzero_col_dim);

	return ret;
}

// 2024-07-30(Tue)
/* get the DRSMatrix ij-element */
double get_drsmatrix_ij(DRSMatrix mat, long int row_index, long int col_index)
{
	long int i, j, total_index;

	if((row_index < 0) || (row_index >= mat->row_dim) || (col_index < 0) || (col_index >= mat->col_dim))
	{
		fprintf(stderr, "Warning: row_index(%ld) or col_index(%ld) is illegal!\n", row_index, col_index);
		return mat->zero_element;
	}

	// finding mat_ij element
	//if(mat->nzero_col_dim[row_index] >= 1)
	if(mat->real_nzero_col_dim[row_index] >= 1)
	{
		total_index = 0;
		for(i = 0; i < row_index; i++)
			total_index += mat->real_nzero_col_dim[i];
			//total_index += mat->nzero_col_dim[i];

		for(j = 0; j < mat->nzero_col_dim[row_index] ; j++)
		{
			// Find!
			if(mat->nzero_index[row_index][j] == col_index)
				return mat->element[total_index];
			
			total_index++;
		}
	}

	// Not found -> return 0
	return mat->zero_element;
}

// 2024-08-04 (Sun) T.Kouya
/* set the DRSMatrix ij-element */
void set_drsmatrix_ij(DRSMatrix mat, long int row_index, long int col_index, double val)
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
				mat->element[total_index] = val;

				return;
			}
			
			total_index++;
		}
	}

	return;
}

// 2024-07-30(Tue) T.Kouya
/* initialize and set DRSMatrix */
DRSMatrix init_set_drsmatrix(DRSMatrix spmat_org)
{
	long int index, i, j;
	DRSMatrix ret;

	//DRSMatrix init_drsmatrix(long int row_dim, long int *nzero_col_dim, long int nzero_total_num)
	ret = init_drsmatrix(spmat_org->row_dim, spmat_org->nzero_col_dim, spmat_org->nzero_total_num);
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR(init_set_drsmatrix): cannot allocate DRSMatrix!\n");
		return NULL;
	}

	// Real total number of non-zero elements
	ret->real_nzero_total_num = spmat_org->real_nzero_total_num;

    // copy nzero_index
	for(i = 0; i < spmat_org->row_dim; i++)
	{
		// Real numbers of non-zero elements in i-th row
		ret->real_nzero_col_dim[i] = spmat_org->real_nzero_col_dim[i];

		for(j = 0; j < spmat_org->nzero_col_dim[i]; j++)
			ret->nzero_index[i][j] = spmat_org->nzero_index[i][j];
	}

	// substitution
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	for(index = 0; index < spmat_org->real_nzero_total_num; index += _BNC_D_WIDTH)
		_mm256_store_pd(&(ret->element[index]), _mm256_load_pd(&(spmat_org->element[index])));

#elif defined(__AVX512F__) // __AVX512F__
	for(index = 0; index < spmat_org->real_nzero_total_num; index += _BNC_D_WIDTH)
		_mm512_store_pd(&(ret->element[index]), _mm512_load_pd(&(spmat_org->element[index])));

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2
	{
		long _vl = (long)svcntd();
		for(index = 0; index < spmat_org->real_nzero_total_num; index += _vl)
		{
			svbool_t pg = svwhilelt_b64_s64((int64_t)index,
			                                (int64_t)spmat_org->real_nzero_total_num);
			svst1_f64(pg, &(ret->element[index]),
			          svld1_f64(pg, &(spmat_org->element[index])));
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM Neon
	for(index = 0; index < spmat_org->real_nzero_total_num; index += 2)
		vst1q_f64(&(ret->element[index]), vld1q_f64(&(spmat_org->element[index])));



#else // others
	//memcpy(ret->element, spmat_org->element, sizeof(double) * spmat_org->nzero_total_num);
	//for(index = 0; index < spmat_org->nzero_total_num; index++)
	for(index = 0; index < spmat_org->real_nzero_total_num; index++)
		ret->element[index] = spmat_org->element[index];

#endif // __AVX2__

	return ret;
}

// 2024-07-30(Tue) T.Kouya
// spmat := 0
void set0_drsmatrix(DRSMatrix spmat)
{
	long int index;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d zero4;

	zero4 = _mm256_setzero_pd();

	for(index = 0; index < spmat->real_nzero_total_num; index += _BNC_D_WIDTH)
		_mm256_store_pd(&(spmat->element[index]), zero4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512d zero8;

	zero8 = _mm512_setzero_pd();

	for(index = 0; index < spmat->real_nzero_total_num; index += _BNC_D_WIDTH)
		_mm512_store_pd(&(spmat->element[index]), zero8);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2
	{
		long _vl = (long)svcntd();
		svfloat64_t zerov = svdup_n_f64(0.0);
		for(index = 0; index < spmat->real_nzero_total_num; index += _vl)
		{
			svbool_t pg = svwhilelt_b64_s64((int64_t)index,
			                                (int64_t)spmat->real_nzero_total_num);
			svst1_f64(pg, &(spmat->element[index]), zerov);
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM Neon
	float64x2_t zero2;

	zero2 = vdupq_n_f64(0.0);

	for(index = 0; index < spmat->real_nzero_total_num; index += 2)
		vst1q_f64(&(spmat->element[index]), zero2);



#else // others
	// substitution
	for(index = 0; index < spmat->real_nzero_total_num; index++)
		spmat->element[index] = (double)0.0;

#endif // __AVX2__

	return;
}

/* Get variables to initialize DRSMatrix */
int get_vars_drsmatrix_fname(long int *ptr_row_dim, long int **ptr_nzero_col_dim, long int *ptr_nzero_total_num, const char *fname)
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

#ifdef USE_GMP
// MPFRSMatrix -> DRSMatrix
DRSMatrix init_set_drsmatrix_mpfrsmatrix(MPFRSMatrix org_sp)
{
	DRSMatrix ret;
	double dtmp;
	long int org_sp_total_index, total_index, i, j;

    //printf("row_dim, nzero_total_num = %ld, %ld\n", org_sp->row_dim, org_sp->nzero_total_num);
    ret = init_drsmatrix(org_sp->row_dim, org_sp->nzero_col_dim, org_sp->nzero_total_num);
    //printf("init_ddrsmatrix!\n");

    // MPFR -> DD
    total_index = 0;
	org_sp_total_index = 0;
	for(i = 0; i < ret->row_dim; i++)
	{
		//printf("%ld: ", i);
		for(j = 0; j < ret->nzero_col_dim[i]; j++)
		{
            dtmp = mpf_get_d(org_sp->element[org_sp_total_index + j]);
            ret->element[total_index + j] = dtmp;
            ret->nzero_index[i][j] = org_sp->nzero_index[i][j];
			//total_index++;
			//printf("%ld ", j);
		}
		//printf("\n");
		total_index += ret->real_nzero_col_dim[i];
		org_sp_total_index += org_sp->nzero_col_dim[i];
	}
    //print_drsmatrix(ret);

	return ret;
}
#endif // USE_GMP

/* Read URI linking data */
int fread_urilinkdat_fname(DRSMatrix ret, const char *fname)
{
	FILE *fp;
	static char line_buf[LINE_BUF_LEN];
	static long int *nzero_row_dim;
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

	nzero_row_dim = (long int *)malloc(sizeof(long int) * ret->col_dim);

	/* Read file */
	for(i = 0; i < ret->row_dim; i++)
		ret->nzero_col_dim[i] = 0;
	for(i = 0; i < ret->col_dim; i++)
		nzero_row_dim[i] = 0;
	
	while(fgets(line_buf, LINE_BUF_LEN - 1, fp) != NULL)
	//for(i = 0; i <= max_row_index; i++)
	{
		/* get row_index, col_index, element */
		sscanf(line_buf, "%ld %ld", &row_index, &col_index);
		//fscanf(fp, "%ld %ld", &row_index, &col_index);
		ret->nzero_index[row_index][ret->nzero_col_dim[row_index]] = col_index;
		//printf("%ld %ld %ld->%ld\n", row_index, col_index, ret->nzero_col_dim[row_index], ret->nzero_index[row_index][ret->nzero_col_dim[row_index]]);
		ret->nzero_col_dim[row_index]++;
		nzero_row_dim[col_index]++;
	}

	ele_index = 0;
	for(i = 0; i < ret->row_dim; i++)
	{
		//printf("ret->nzero_col_dim[%ld] = %ld\n", i, ret->nzero_col_dim[i]);
		for(j = 0; j < ret->nzero_col_dim[i]; j++)
		{
			*(ret->element + ele_index) = 1.0 / (double)nzero_row_dim[ret->nzero_index[i][j]];
			//printf("%ld %ld %ld -> %f\n", ele_index, i, j, *(ret->element + ele_index));
			ele_index++;
		}
	}

	free(nzero_row_dim);

	/* file close */
	fclose(fp);

//	print_drsmatrix(ret);

	return SUCCESS;
}

#ifdef USE_IMKL
// ret := (sparse_matrix_t)mat
void convert_indeces_drsmatrix_mkl_csrmat(MKL_INT *ret_i_csr_start, MKL_INT *ret_i_csr_end, MKL_INT *ret_j_csr, DRSMatrix mat)
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
sparse_status_t convert_drsmatrix_mkl_csrmat(sparse_matrix_t *ret, MKL_INT *ret_i_csr_start, MKL_INT *ret_i_csr_end, MKL_INT *ret_j_csr, DRSMatrix mat)
{
	//long int i, j, total_index;
    //int *i_mat_csr, *j_mat_csr, row_dim = (int)mat->row_dim;
    //int row_dim = (int)mat->row_dim;
    sparse_status_t ret_mkl;

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
	convert_indeces_drsmatrix_mkl_csrmat(ret_i_csr_start, ret_i_csr_end, ret_j_csr, mat);

	//i_mat_csr_end[row_dim] = mat->real_nzero_total_num;
	//mkl_cspblas_dcsrgemv("N", &row_dim, mat->element, i_mat_csr, j_mat_csr, vec->element, ret->element);
    ret_mkl = mkl_sparse_d_create_csr(
        ret,
        SPARSE_INDEX_BASE_ZERO,
        (MKL_INT)mat->row_dim,
        (MKL_INT)mat->col_dim,
        ret_i_csr_start,
        ret_i_csr_end,
        ret_j_csr,
        mat->element
    );

	//free(i_mat_csr); free(j_mat_csr);
    return ret_mkl;
}
// ret := (sparse_matrix_t)mat
sparse_status_t subst_drsmatrix_mkl_csrmat(sparse_matrix_t *ret, MKL_INT *i_csr_start, MKL_INT *i_csr_end, MKL_INT *j_csr, DRSMatrix mat)
{
	//long int i, j, total_index;
    //int *i_mat_csr, *j_mat_csr, row_dim = (int)mat->row_dim;
    //int row_dim = (int)mat->row_dim;
    sparse_status_t ret_mkl;

	//i_mat_csr_end[row_dim] = mat->real_nzero_total_num;
	//mkl_cspblas_dcsrgemv("N", &row_dim, mat->element, i_mat_csr, j_mat_csr, vec->element, ret->element);
    ret_mkl = mkl_sparse_d_create_csr(
        ret,
        SPARSE_INDEX_BASE_ZERO,
        (MKL_INT)mat->row_dim,
        (MKL_INT)mat->col_dim,
        i_csr_start,
        i_csr_end,
        j_csr,
        mat->element
    );

	//free(i_mat_csr); free(j_mat_csr);
    return ret_mkl;
}
#endif // USE_IMKL

/* Multiply DRSMatrix * DVector */
int mul_drsmatrix_dvec(DVector ret, DRSMatrix mat, DVector vec)
{
	long int i, j, total_index;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	set0_dvector(ret);

//#ifdef USE_IMKL
#ifdef USE_IMKL
	#ifdef USE_IMKL_OLD // 2025-07-10(Thu)
	//#if 0
		int *i_mat_csr, *j_mat_csr, row_dim = (int)mat->row_dim;

		// convert our CSR to intel math kernel csr format
		i_mat_csr = (int *)calloc(mat->row_dim + 1, sizeof(int));
		j_mat_csr = (int *)calloc(mat->real_nzero_total_num, sizeof(int));

		total_index = 0;
		for(i = 0; i < row_dim; i++)
		{
			i_mat_csr[i] = (int)total_index;
			//if(a->nzero_col_dim[i] >= 1)
			if(mat->real_nzero_col_dim[i] >= 1)
			{
				for(j = 0; j < mat->nzero_col_dim[i]; j++)
				{
					j_mat_csr[total_index] = mat->nzero_index[i][j];
					total_index++;
				}
				// embed gap among nzero_col_dim and real_nzero_col_dim
				for(j = mat->nzero_col_dim[i]; j < mat->real_nzero_col_dim[i]; j++)
				{
					j_mat_csr[total_index] = mat->nzero_index[i][mat->nzero_col_dim[i] - 1] + (j + 1 - mat->nzero_col_dim[i]);
					total_index++;
				}
			}
		}
		i_mat_csr[row_dim] = mat->real_nzero_total_num;
		mkl_cspblas_dcsrgemv("N", &row_dim, mat->element, i_mat_csr, j_mat_csr, vec->element, ret->element);

		free(i_mat_csr); free(j_mat_csr);
	//#endif // 0
	#else // USE_IMKL new
		sparse_matrix_t mkl_drsmat;
		MKL_INT *i_csr_start, *i_csr_end, *j_csr;
		struct matrix_descr descr;

		// DRSMatrix -> MKL_CSRmatrix
		convert_drsmatrix_mkl_csrmat(&mkl_drsmat, i_csr_start, i_csr_end, j_csr, mat);

		// b := A * x by IMKL
		descr.type = SPARSE_MATRIX_TYPE_GENERAL;
		//descr.mode = ?
		//descr.diag = ?
		mkl_sparse_d_mv(
			SPARSE_OPERATION_NON_TRANSPOSE,
			(double)1.0,
			mkl_drsmat,
			descr,
			vec->element,
			(double)0.0,
			vec->element
		);
	    free(i_csr_start); free(i_csr_end); free(j_csr);
		mkl_sparse_destroy(mkl_drsmat);

	#endif // IMKL
#else // USE_IMKL

	total_index = 0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, vb4, tmp4;
	long int jmax, jres, col_dim;
	double mat_vec_i;

	for(i = 0; i < mat->row_dim; i++)
	{
		//ret->element[i] = 0.0;
		tmp4 = _mm256_setzero_pd();
		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		for(j = 0; j < jmax; j += _BNC_D_WIDTH)
		{
			//ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			a4  = _mm256_load_pd(&(mat->element[total_index]));
			vb4 = _mm256_set_pd(
				vec->element[mat->nzero_index[i][j + 3]],
				vec->element[mat->nzero_index[i][j + 2]],
				vec->element[mat->nzero_index[i][j + 1]],
				vec->element[mat->nzero_index[i][j    ]]
			);
			tmp4 = _mm256_fmadd_pd(a4, vb4, tmp4);

			// total_index++;
			total_index += _BNC_D_WIDTH;
		}
		//mat_vec_i = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3];
		mat_vec_i = _bncavx2_dsum256d(tmp4); // tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3];
		//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);
		col_dim = mat->nzero_col_dim[i];
		switch(jres)
		{
			case 1: 
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				break; 
			case 2:
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				break; 
			case 3:
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 3]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				break; 
		}
		//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
		set_dvector_i(ret, i, mat_vec_i);
		total_index += mat->real_nzero_col_dim[i] - mat->nzero_col_dim[i]; /* skip per-row SIMD padding */
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, vb8, tmp8;
	long int jmax, jres, col_dim;
	double mat_vec_i;

	for(i = 0; i < mat->row_dim; i++)
	{
		//ret->element[i] = 0.0;
		tmp8 = _mm512_setzero_pd();
		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres = mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		for(j = 0; j < jmax; j += _BNC_D_WIDTH)
		{
			//ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			a8  = _mm512_load_pd(&(mat->element[total_index]));
			vb8 = _mm512_set_pd(
				vec->element[mat->nzero_index[i][j + 7]],
				vec->element[mat->nzero_index[i][j + 6]],
				vec->element[mat->nzero_index[i][j + 5]],
				vec->element[mat->nzero_index[i][j + 4]],
				vec->element[mat->nzero_index[i][j + 3]],
				vec->element[mat->nzero_index[i][j + 2]],
				vec->element[mat->nzero_index[i][j + 1]],
				vec->element[mat->nzero_index[i][j    ]]
			);
			tmp8 = _mm512_fmadd_pd(a8, vb8, tmp8);

			// total_index++;
			total_index += _BNC_D_WIDTH;
		}
		//mat_vec_i = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3] + tmp4[4] + tmp4[5] + tmp4[6] + tmp4[7];
		mat_vec_i = _bncavx2_dsum512d(tmp8);
		col_dim = mat->nzero_col_dim[i];
		switch(jres)
		{
			case 1:
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				break;
			case 2:
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				break; 
			case 3:
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 3]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				break; 
			case 4:
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 4]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 3]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				break; 
			case 5:
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 5]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 4]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 3]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				break; 
			case 6:
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 6]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 5]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 4]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 3]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				break; 
			case 7:
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 7]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 6]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 5]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 4]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 3]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				break; 
		}
		set_dvector_i(ret, i, mat_vec_i);
		total_index += mat->real_nzero_col_dim[i] - mat->nzero_col_dim[i]; /* skip per-row SIMD padding */
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2
	/* SVE2 (VL-agnostic) SpMV: uses hardware gather (svld1_gather_*) for the
	 * vec->element[index[]] reads, which NEON has to emulate scalar-wise.
	 * Vector length read at runtime from svcntd(); a tail-scalar loop handles
	 * the remainder when nzero_col_dim is not a multiple of VL. */
	{
		long _vl = (long)svcntd();
		long int jmax, jres, col_dim;
		double mat_vec_i;
		for(i = 0; i < mat->row_dim; i++)
		{
			svfloat64_t tmp_v = svdup_n_f64(0.0);
			long int nz = mat->nzero_col_dim[i];
			jmax = (nz / _vl) * _vl;
			jres = nz - jmax;
			for(j = 0; j < jmax; j += _vl)
			{
				svbool_t pg = svptrue_b64();
				svfloat64_t a_v  = svld1_f64(pg, &(mat->element[total_index]));
				svint64_t   idx  = svld1_s64(pg, (int64_t*)&(mat->nzero_index[i][j]));
				svfloat64_t vb_v = svld1_gather_s64index_f64(pg, vec->element, idx);
				tmp_v = svmla_f64_x(pg, tmp_v, a_v, vb_v);
				total_index += _vl;
			}
			mat_vec_i = svaddv_f64(svptrue_b64(), tmp_v);
			col_dim = mat->nzero_col_dim[i];
			/* scalar tail */
			for(j = 0; j < jres; j++)
			{
				mat_vec_i += mat->element[total_index] *
				             vec->element[mat->nzero_index[i][col_dim - jres + j]];
				total_index++;
			}
			total_index += mat->real_nzero_col_dim[i] - nz; /* skip per-row SIMD padding */
			set_dvector_i(ret, i, mat_vec_i);
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM Neon
	float64x2_t a2, vb2, tmp2;
	long int jmax, jres, col_dim;
	double mat_vec_i;

	for(i = 0; i < mat->row_dim; i++)
	{
		//ret->element[i] = 0.0;
		tmp2 = vdupq_n_f64(0.0);
		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		jmax = (mat->nzero_col_dim[i] / 2) * 2;
		jres =  mat->nzero_col_dim[i] % 2;
		for(j = 0; j < jmax; j += 2)
		{
			//ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			a2  = vld1q_f64(&(mat->element[total_index]));
			vb2 = (float64x2_t){
				vec->element[mat->nzero_index[i][j    ]],
				vec->element[mat->nzero_index[i][j + 1]]
			};
			tmp2 = vfmaq_f64(tmp2, a2, vb2);

			// total_index++;
			total_index += 2;
		}
		//mat_vec_i = vgetq_lane_f64(tmp2, 0) + vgetq_lane_f64(tmp2, 1);
		mat_vec_i = (vgetq_lane_f64(tmp2, 0) + vgetq_lane_f64(tmp2, 1)); // vgetq_lane_f64(tmp2, 0) + vgetq_lane_f64(tmp2, 1);
		//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);
		col_dim = mat->nzero_col_dim[i];
		switch(jres)
		{
			case 1:
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				break;
			case 2:
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				break;
			case 3:
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 3]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 2]];
				mat_vec_i += mat->element[total_index++] * vec->element[mat->nzero_index[i][col_dim - 1]];
				break;
		}
		//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
		set_dvector_i(ret, i, mat_vec_i);
		total_index += mat->real_nzero_col_dim[i] - mat->nzero_col_dim[i]; /* skip per-row SIMD padding */
	}



#else // others
	for(i = 0; i < mat->row_dim; i++)
	{
		ret->element[i] = 0.0;
		//for(j = 0; j < mat->real_nzero_col_dim[i]; j++)
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
// fix!: 2011-08-29 by T.Kouya
			ret->element[i] += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			total_index++;
		}
	}
#endif // __AVX2__
#endif // USE_IMKL

	return SUCCESS;
}

/* Multiply DRSMatrix^T * DVector */
int mul_drsmatrixt_dvec(DVector ret, DRSMatrix mat, DVector vec)
{
	long int i, j, total_index;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

//#if 0
#ifdef USE_IMKL
	#ifdef USE_IMKL_OLD // 2025-07-10(Thu)
	int *i_mat_csr, *j_mat_csr, row_dim = (int)mat->row_dim;

	// convert our CSR to intel math kernel csr format
	i_mat_csr = (int *)calloc(mat->row_dim + 1, sizeof(int));
	j_mat_csr = (int *)calloc(mat->real_nzero_total_num, sizeof(int));

	total_index = 0;
	for(i = 0; i < row_dim; i++)
	{
		i_mat_csr[i] = (int)total_index;
		//if(a->nzero_col_dim[i] >= 1)
		if(mat->real_nzero_col_dim[i] >= 1)
		{
			for(j = 0; j < mat->nzero_col_dim[i]; j++)
			{
				j_mat_csr[total_index] = mat->nzero_index[i][j];
				total_index++;
			}
			// embed gap among nzero_col_dim and real_nzero_col_dim
			for(j = mat->nzero_col_dim[i]; j < mat->real_nzero_col_dim[i]; j++)
			{
				j_mat_csr[total_index] = mat->nzero_index[i][mat->nzero_col_dim[i] - 1] + (j + 1 - mat->nzero_col_dim[i]);
				total_index++;
			}
		}
	}
	i_mat_csr[row_dim] = mat->real_nzero_total_num;
	mkl_cspblas_dcsrgemv("T", &row_dim, mat->element, i_mat_csr, j_mat_csr, vec->element, ret->element);

	free(i_mat_csr); free(j_mat_csr);
	#else // New
		sparse_matrix_t mkl_drsmat;
		MKL_INT *i_csr_start, *i_csr_end, *j_csr;
		struct matrix_descr descr;

		// DRSMatrix -> MKL_CSRmatrix
		convert_drsmatrix_mkl_csrmat(&mkl_drsmat, i_csr_start, i_csr_end, j_csr, mat);

		// b := A * x by IMKL
		descr.type = SPARSE_MATRIX_TYPE_GENERAL;
		//descr.mode = ?
		//descr.diag = ?
		mkl_sparse_d_mv(
			SPARSE_OPERATION_TRANSPOSE,
			(double)1.0,
			mkl_drsmat,
			descr,
			vec->element,
			(double)0.0,
			vec->element
		);
	    free(i_csr_start); free(i_csr_end); free(j_csr);
	#endif // USE_IMKL_OLD
#else // USE_IMKL
//#endif // 0

	total_index = 0;
	set0_dvector(ret);

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, vb4, ret4;
	long int jmax, jres, col_dim;

	for(i = 0; i < mat->row_dim; i++)
	{
		vb4 = _mm256_set1_pd(vec->element[i]);
		jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;
		//jmax = (mat->row_dim / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		//jres =  mat->row_dim % _BNC_D_WIDTH;
		for(j = 0; j < jmax; j += _BNC_D_WIDTH)
		{
			//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
			a4  = _mm256_load_pd(&(mat->element[total_index]));
			ret4 = _mm256_set_pd(
				ret->element[mat->nzero_index[i][j + 3]],
				ret->element[mat->nzero_index[i][j + 2]],
				ret->element[mat->nzero_index[i][j + 1]],
				ret->element[mat->nzero_index[i][j    ]]
			);
			ret4 = _mm256_fmadd_pd(a4, vb4, ret4);

			ret->element[mat->nzero_index[i][j + 3]] = ret4[3];
			ret->element[mat->nzero_index[i][j + 2]] = ret4[2];
			ret->element[mat->nzero_index[i][j + 1]] = ret4[1];
			ret->element[mat->nzero_index[i][j    ]] = ret4[0];

			// total_index++;
			total_index += _BNC_D_WIDTH;
		}
		//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);
		col_dim = mat->nzero_col_dim[i];
		//col_dim = mat->real_nzero_col_dim[i];
		switch(jres)
		{
			case 1: 
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break; 
			case 2:
				ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break; 
			case 3:
				ret->element[mat->nzero_index[i][col_dim - 3]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break;
		}
		total_index += mat->real_nzero_col_dim[i] - mat->nzero_col_dim[i]; /* skip per-row SIMD padding */
		//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, vb8, ret8;
	long int jmax, jres, col_dim;

	for(i = 0; i < mat->row_dim; i++)
	{
		vb8 = _mm512_set1_pd(vec->element[i]);
		jmax = (mat->nzero_col_dim[i] / _BNC_D_WIDTH) * _BNC_D_WIDTH;
		jres =  mat->nzero_col_dim[i] % _BNC_D_WIDTH;

		for(j = 0; j < jmax; j += _BNC_D_WIDTH)
		{
			//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
			a8  = _mm512_load_pd(&(mat->element[total_index]));
			ret8 = _mm512_set_pd(
				ret->element[mat->nzero_index[i][j + 7]],
				ret->element[mat->nzero_index[i][j + 6]],
				ret->element[mat->nzero_index[i][j + 5]],
				ret->element[mat->nzero_index[i][j + 4]],
				ret->element[mat->nzero_index[i][j + 3]],
				ret->element[mat->nzero_index[i][j + 2]],
				ret->element[mat->nzero_index[i][j + 1]],
				ret->element[mat->nzero_index[i][j    ]]
			);
			ret8 = _mm512_fmadd_pd(a8, vb8, ret8);

			ret->element[mat->nzero_index[i][j + 7]] = ret8[7];
			ret->element[mat->nzero_index[i][j + 6]] = ret8[6];
			ret->element[mat->nzero_index[i][j + 5]] = ret8[5];
			ret->element[mat->nzero_index[i][j + 4]] = ret8[4];
			ret->element[mat->nzero_index[i][j + 3]] = ret8[3];
			ret->element[mat->nzero_index[i][j + 2]] = ret8[2];
			ret->element[mat->nzero_index[i][j + 1]] = ret8[1];
			ret->element[mat->nzero_index[i][j    ]] = ret8[0];

			// total_index++;
			total_index += _BNC_D_WIDTH;
		}
		//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);
		col_dim = mat->nzero_col_dim[i];
		switch(jres)
		{
			case 1:
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break;
			case 2:
				ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break;
			case 3:
				ret->element[mat->nzero_index[i][col_dim - 3]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break; 
			case 4:
				ret->element[mat->nzero_index[i][col_dim - 4]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 3]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break; 
			case 5:
				ret->element[mat->nzero_index[i][col_dim - 5]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 4]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 3]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break; 
			case 6:
				ret->element[mat->nzero_index[i][col_dim - 6]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 5]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 4]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 3]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break; 
			case 7:
				ret->element[mat->nzero_index[i][col_dim - 7]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 6]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 5]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 4]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 3]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break;

		}
		total_index += mat->real_nzero_col_dim[i] - mat->nzero_col_dim[i]; /* skip per-row SIMD padding */
		//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2
	/* SVE2 A^T*x: gather current ret[index], FMA, then scatter back.
	 * SVE2 has both svld1_gather_* and svst1_scatter_*, so this is direct;
	 * the NEON path has to do two scalar loads + two scalar stores per pair. */
	{
		long _vl = (long)svcntd();
		long int jmax, jres, col_dim;
		for(i = 0; i < mat->row_dim; i++)
		{
			svfloat64_t vb_v = svdup_n_f64(vec->element[i]);
			long int nz = mat->nzero_col_dim[i];
			jmax = (nz / _vl) * _vl;
			jres = nz - jmax;
			for(j = 0; j < jmax; j += _vl)
			{
				svbool_t pg = svptrue_b64();
				svfloat64_t a_v  = svld1_f64(pg, &(mat->element[total_index]));
				svint64_t   idx  = svld1_s64(pg, (int64_t*)&(mat->nzero_index[i][j]));
				svfloat64_t cur  = svld1_gather_s64index_f64(pg, ret->element, idx);
				svfloat64_t newv = svmla_f64_x(pg, cur, a_v, vb_v);
				svst1_scatter_s64index_f64(pg, ret->element, idx, newv);
				total_index += _vl;
			}
			col_dim = mat->nzero_col_dim[i];
			/* scalar tail */
			for(j = 0; j < jres; j++)
			{
				ret->element[mat->nzero_index[i][col_dim - jres + j]]
				    += mat->element[total_index] * vec->element[i];
				total_index++;
			}
			total_index += mat->real_nzero_col_dim[i] - nz; /* skip per-row SIMD padding */
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM Neon
	float64x2_t a2, vb2, ret2;
	long int jmax, jres, col_dim;

	for(i = 0; i < mat->row_dim; i++)
	{
		vb2 = vdupq_n_f64(vec->element[i]);
		jmax = (mat->nzero_col_dim[i] / 2) * 2;
		jres =  mat->nzero_col_dim[i] % 2;
		//jmax = (mat->row_dim / 2) * 2;
		//jres =  mat->row_dim % 2;
		for(j = 0; j < jmax; j += 2)
		{
			//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
			a2  = vld1q_f64(&(mat->element[total_index]));
			ret2 = (float64x2_t){
				ret->element[mat->nzero_index[i][j    ]],
				ret->element[mat->nzero_index[i][j + 1]]
			};
			ret2 = vfmaq_f64(ret2, a2, vb2);

			ret->element[mat->nzero_index[i][j + 1]] = vgetq_lane_f64(ret2, 1);
			ret->element[mat->nzero_index[i][j    ]] = vgetq_lane_f64(ret2, 0);

			// total_index++;
			total_index += 2;
		}
		//printf("%ld mat_vec_i, jmax, jrec = %25.17e, %ld -> %ld, %ld\n", i, mat_vec_i, mat->nzero_col_dim[i], jmax, jres);
		col_dim = mat->nzero_col_dim[i];
		//col_dim = mat->real_nzero_col_dim[i];
		switch(jres)
		{
			case 1:
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break;
			case 2:
				ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break;
			case 3:
				ret->element[mat->nzero_index[i][col_dim - 3]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 2]] += mat->element[total_index++] * vec->element[i];
				ret->element[mat->nzero_index[i][col_dim - 1]] += mat->element[total_index++] * vec->element[i];
				break;
		}
		//printf("%ld mat_vec_i = %25.17e\n", i, mat_vec_i);
		total_index += mat->real_nzero_col_dim[i] - mat->nzero_col_dim[i]; /* skip per-row SIMD padding */
	}



#else // __AVX2__
	//for(i = 0; i < mat->row_dim; i++)
	//	ret->element[i] = 0.0;

	total_index = 0;
	for(i = 0; i < mat->row_dim; i++)
	{
		//for(j = 0; j < mat->real_nzero_col_dim[i]; j++)
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
			total_index++;
		}
	}
#endif // __AVX2__
#endif // USE_IMKL

	return SUCCESS;
}

/* Power Method for Randomly Sparse Matrices */
/* 	double *evec: the eigenvector for max eigenvalue */
/* 	double *drsmat: Randomly sparse matrix */
/* 	double reps, aeps: Relative and Absolute tolerance */
/* 	long int max_times: Maximum iterative times of Power method */
double dpower_rsmatrix(DVector evec, DRSMatrix mat, double reps, double aeps, long int max_times)
{
	long int i, absmax_index, times;
	double absmax_new_evec, max_eig, old_max_eig;
	DVector new_evec;

	new_evec = init_dvector(mat->row_dim);

	/* initialize evec */
	for(i = 0; i < evec->dim; i++)
		evec->element[i] = 1.0;

	/* main loop */
	old_max_eig = 0.0;
	for(times = 0; times < max_times; times++)
	{
		/* w := A * x */
		mul_drsmatrix_dvec(new_evec, mat, evec);
		absmax_index = absmax_index_dvector(&absmax_new_evec, new_evec);
		max_eig = absmax_new_evec / evec->element[absmax_index];
//		smul_dvector(evec, 1.0 / absmax_new_evec, new_evec);
		smul_dvector(evec, 1.0 / norm1_dvector(new_evec), new_evec); // Baba's example
		if((fabs(max_eig - old_max_eig) <= reps * fabs(old_max_eig) + aeps) && (times >= 2))
		{
			fprintf(stderr, "Convergent!(Iterative Times = %ld)\n", times);
			break;
		}
		if(times % 10 == 0)
			fprintf(stderr, "%5ld %25.17e\n", times, max_eig);
		old_max_eig = max_eig;
	}

	return max_eig;
}

/* Scalar multiply of DVector */
int smul_dvector(DVector ret, double scalar, DVector vec)
{
	long int i;

	if(ret->dim < vec->dim)
		return ERROR;

	for(i = 0; i < vec->dim; i++)
		ret->element[i] = scalar * vec->element[i];

	return SUCCESS;
}

/* Select index of absolute maximum element and its value in DVector */
long int absmax_index_dvector(double *ret, DVector vec)
{
	long int absmax_index, i;
	double abs_element;

	*ret = 0.0;
	absmax_index = 0;
	for(i = 0; i < vec->dim; i++)
	{
		abs_element = fabs(vec->element[i]);
		if(*ret < abs_element)
		{
			absmax_index = i;
			//*ret = abs_element;
		}
	}

	*ret = vec->element[absmax_index];

	return absmax_index;
}

// 2024-07-30(Tue) T.Kouya
// ret := spmat_a
void subst_drsmatrix(DRSMatrix ret, DRSMatrix spmat_a)
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
		fprintf(stderr, "ERROR(subst_drsmatrix): mismatched nzero_total_nums! (%ld != %ld)\n", ret->nzero_total_num, spmat_a->nzero_total_num);
		return; // -1;
	}
	if((ret->row_dim != spmat_a->row_dim) || (ret->col_dim != spmat_a->col_dim))
	{
		fprintf(stderr, "ERROR(subst_drsmatrix): mismatched dimensions! ((%ld, %ld) != (%ld, %ld))\n", ret->row_dim, ret->col_dim, spmat_a->row_dim, spmat_a->col_dim);
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
		_mm256_store_pd(&(ret->element[total_index]), _mm256_load_pd(&(spmat_a->element[total_index])));

#elif defined(__AVX512F__) // __AVX512F__
	for(total_index = 0; total_index < ret->real_nzero_total_num; total_index += _BNC_D_WIDTH)
		_mm512_store_pd(&(ret->element[total_index]), _mm512_load_pd(&(spmat_a->element[total_index])));

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2
	{
		long _vl = (long)svcntd();
		for(total_index = 0; total_index < ret->real_nzero_total_num; total_index += _vl)
		{
			svbool_t pg = svwhilelt_b64_s64((int64_t)total_index,
			                                (int64_t)ret->real_nzero_total_num);
			svst1_f64(pg, &(ret->element[total_index]),
			          svld1_f64(pg, &(spmat_a->element[total_index])));
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM Neon
	for(total_index = 0; total_index < ret->real_nzero_total_num; total_index += 2)
		vst1q_f64(&(ret->element[total_index]), vld1q_f64(&(spmat_a->element[total_index])));



#else // others
	//memcpy(ret->element, spmat_a->element, sizeof(double) * spmat_a->nzero_total_num);
	//for(total_index = 0; total_index < ret->nzero_total_num; total_index++)
	for(total_index = 0; total_index < ret->real_nzero_total_num; total_index++)
		ret->element[total_index] = spmat_a->element[total_index];

#endif // __AVX2__

}

// 2024-07-30(Tue) T.Kouya
// ret := spmat_a + spmat_b
void add_drsmatrix(DRSMatrix ret, DRSMatrix spmat_a, DRSMatrix spmat_b)
{
	long int index;

	// check parameters
	if((ret->nzero_total_num != spmat_a->nzero_total_num) || (spmat_a->nzero_total_num != spmat_b->nzero_total_num))
	{
		fprintf(stderr, "ERROR(add_drsmatrix): mismatched nzero_total_nums! (%ld != %ld) or (%ld != %ld) \n", ret->nzero_total_num, spmat_a->nzero_total_num, spmat_a->nzero_total_num, spmat_b->nzero_total_num);
		return;
	}
	if((ret->row_dim != spmat_a->row_dim) || (ret->col_dim != spmat_a->col_dim) || (spmat_a->row_dim != spmat_b->row_dim) || (spmat_a->col_dim != spmat_b->col_dim))
	{
		fprintf(stderr, "ERROR(add_drsmatrix): mismatched dimensions! ((%ld, %ld) != (%ld, %ld) != (%ld, %ld)\n", ret->row_dim, ret->col_dim, spmat_a->row_dim, spmat_a->col_dim, spmat_b->row_dim, spmat_b->col_dim);
		return;
	}

	// addition
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, b4, ret4;

	for(index = 0; index < ret->real_nzero_total_num; index += _BNC_D_WIDTH)
	{
		a4 = _mm256_load_pd(&(spmat_a->element[index]));
		b4 = _mm256_load_pd(&(spmat_b->element[index]));
		ret4 = _mm256_add_pd(a4, b4);
		_mm256_store_pd(&(ret->element[index]), ret4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, b8, ret8;

	for(index = 0; index < ret->real_nzero_total_num; index += _BNC_D_WIDTH)
	{
		a8 = _mm512_load_pd(&(spmat_a->element[index]));
		b8 = _mm512_load_pd(&(spmat_b->element[index]));
		ret8 = _mm512_add_pd(a8, b8);
		_mm512_store_pd(&(ret->element[index]), ret8);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2
	{
		long _vl = (long)svcntd();
		for(index = 0; index < ret->real_nzero_total_num; index += _vl)
		{
			svbool_t pg = svwhilelt_b64_s64((int64_t)index,
			                                (int64_t)ret->real_nzero_total_num);
			svfloat64_t a_v = svld1_f64(pg, &(spmat_a->element[index]));
			svfloat64_t b_v = svld1_f64(pg, &(spmat_b->element[index]));
			svst1_f64(pg, &(ret->element[index]), svadd_f64_x(pg, a_v, b_v));
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM Neon
	float64x2_t a2, b2, ret2;

	for(index = 0; index < ret->real_nzero_total_num; index += 2)
	{
		a2 = vld1q_f64(&(spmat_a->element[index]));
		b2 = vld1q_f64(&(spmat_b->element[index]));
		ret2 = vaddq_f64(a2, b2);
		vst1q_f64(&(ret->element[index]), ret2);
	}


#else // others
	//for(index = 0; index < spmat_a->nzero_total_num; index++)
	for(index = 0; index < spmat_a->real_nzero_total_num; index++)
		ret->element[index] = spmat_a->element[index] + spmat_b->element[index];

#endif // __AVX2__

}

// 2024-07-30(Tue) T.Kouya
// ret := spmat_a - spmat_b
void sub_drsmatrix(DRSMatrix ret, DRSMatrix spmat_a, DRSMatrix spmat_b)
{
	long int index;

	// check parameters
	if(ret->nzero_total_num != spmat_a->nzero_total_num)
	{
		fprintf(stderr, "ERROR(sub_drsmatrix): mismatched nzero_total_nums! (%ld != %ld)\n", ret->nzero_total_num, spmat_a->nzero_total_num);
		return; // -1;
	}
	if((ret->row_dim != spmat_a->row_dim) || (ret->col_dim != spmat_a->col_dim))
	{
		fprintf(stderr, "ERROR(sub_drsmatrix): mismatched dimensions! ((%ld, %ld) != (%ld, %ld))\n", ret->row_dim, ret->col_dim, spmat_a->row_dim, spmat_a->col_dim);
		return; //-2;
	}

	// subtraction
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, b4, ret4;

	for(index = 0; index < ret->real_nzero_total_num; index += _BNC_D_WIDTH)
	{
		a4 = _mm256_load_pd(&(spmat_a->element[index]));
		b4 = _mm256_load_pd(&(spmat_b->element[index]));
		ret4 = _mm256_sub_pd(a4, b4);
		_mm256_store_pd(&(ret->element[index]), ret4);
	}
#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, b8, ret8;

	for(index = 0; index < ret->real_nzero_total_num; index += _BNC_D_WIDTH)
	{
		a8 = _mm512_load_pd(&(spmat_a->element[index]));
		b8 = _mm512_load_pd(&(spmat_b->element[index]));
		ret8 = _mm512_sub_pd(a8, b8);
		_mm512_store_pd(&(ret->element[index]), ret8);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2
	{
		long _vl = (long)svcntd();
		for(index = 0; index < ret->real_nzero_total_num; index += _vl)
		{
			svbool_t pg = svwhilelt_b64_s64((int64_t)index,
			                                (int64_t)ret->real_nzero_total_num);
			svfloat64_t a_v = svld1_f64(pg, &(spmat_a->element[index]));
			svfloat64_t b_v = svld1_f64(pg, &(spmat_b->element[index]));
			svst1_f64(pg, &(ret->element[index]), svsub_f64_x(pg, a_v, b_v));
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM Neon
	float64x2_t a2, b2, ret2;

	for(index = 0; index < ret->real_nzero_total_num; index += 2)
	{
		a2 = vld1q_f64(&(spmat_a->element[index]));
		b2 = vld1q_f64(&(spmat_b->element[index]));
		ret2 = vsubq_f64(a2, b2);
		vst1q_f64(&(ret->element[index]), ret2);
	}


#else // others
	//for(index = 0; index < spmat_a->nzero_total_num; index++)
	for(index = 0; index < spmat_a->real_nzero_total_num; index++)
		ret->element[index] = spmat_a->element[index] - spmat_b->element[index];

#endif // __AVX2__

}

// 2024-07-30(Tue) T.Kouya
// ret := scaler * spmat_a
void cmul_drsmatrix(DRSMatrix ret, double scaler, DRSMatrix spmat_a)
{
	long int index;

	// check parameters
	if(ret->nzero_total_num != spmat_a->nzero_total_num)
	{
		fprintf(stderr, "ERROR(cmul_drsmatrix): mismatched nzero_total_nums! (%ld != %ld)\n", ret->nzero_total_num, spmat_a->nzero_total_num);
		return; // -1;
	}
	if((ret->row_dim != spmat_a->row_dim) || (ret->col_dim != spmat_a->col_dim))
	{
		fprintf(stderr, "ERROR(cmul_drsmatrix): mismatched dimensions! ((%ld, %ld) != (%ld, %ld))\n", ret->row_dim, ret->col_dim, spmat_a->row_dim, spmat_a->col_dim);
		return; // -2;
	}

	// scaler multiplication
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d a4, ret4, scaler4;

	scaler4 = _mm256_set1_pd(scaler);

	for(index = 0; index < ret->real_nzero_total_num; index += _BNC_D_WIDTH)
	{
		a4 = _mm256_load_pd(&(spmat_a->element[index]));
		ret4 = _mm256_mul_pd(scaler4, a4);
		_mm256_store_pd(&(ret->element[index]), ret4);
	}

#elif defined(__AVX512F__) // __AVX512F__
	__m512d a8, ret8, scaler8;

	scaler8 = _mm512_set1_pd(scaler);

	for(index = 0; index < ret->real_nzero_total_num; index += _BNC_D_WIDTH)
	{
		a8 = _mm512_load_pd(&(spmat_a->element[index]));
		ret8 = _mm512_mul_pd(scaler8, a8);
		_mm512_store_pd(&(ret->element[index]), ret8);
	}

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2
	{
		long _vl = (long)svcntd();
		svfloat64_t scalerv = svdup_n_f64(scaler);
		for(index = 0; index < ret->real_nzero_total_num; index += _vl)
		{
			svbool_t pg = svwhilelt_b64_s64((int64_t)index,
			                                (int64_t)ret->real_nzero_total_num);
			svfloat64_t a_v = svld1_f64(pg, &(spmat_a->element[index]));
			svst1_f64(pg, &(ret->element[index]),
			          svmul_f64_x(pg, scalerv, a_v));
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM Neon
	float64x2_t a2, ret2, scaler2;

	scaler2 = vdupq_n_f64(scaler);

	for(index = 0; index < ret->real_nzero_total_num; index += 2)
	{
		a2 = vld1q_f64(&(spmat_a->element[index]));
		ret2 = vmulq_f64(scaler2, a2);
		vst1q_f64(&(ret->element[index]), ret2);
	}



#else // others
	//for(index = 0; index < spmat_a->nzero_total_num; index++)
	for(index = 0; index < ret->real_nzero_total_num; index++)
		ret->element[index] = scaler * spmat_a->element[index];

#endif // __AVX2__
}

// 2024-07-30(Tue) T.Kouya
// absmax_row_drsmatrix
double absmax_row_drsmatrix(long int *max_j, long int row_index, DRSMatrix mat)
{
    long int j, max_index = 0;
    double mu, abs_aij;

	mu = fabs(get_drsmatrix_ij(mat, row_index, 0));

	for(j = 1; j < mat->col_dim; j++)
	{
		abs_aij = fabs(get_drsmatrix_ij(mat, row_index, j));
		if(abs_aij > mu)
        {
			mu = abs_aij;
            //max_index = j;
            //max_index = mat->nzero_index[row_index][j];
			max_index = j; // Fix! 2024-08-03(Sat) T.Kouya
        }
	}

    if(max_j != NULL)
        *max_j = max_index;

    return mu;
}

// 2024-07-30(Tue) T.Kouya
// absmax_col_dmatrix
double absmax_col_drsmatrix(long int *max_i, long int col_index, DRSMatrix mat)
{
    long int i, max_index = 0;
    double mu, abs_aij;

	mu = fabs(get_drsmatrix_ij(mat, 0, col_index));
	for(i = 1; i < mat->row_dim; i++)
	{
        abs_aij = fabs(get_drsmatrix_ij(mat, i, col_index));
		if(abs_aij > mu)
        {
            mu = abs_aij; // Fix! 2022-11-16(Wed)
            //max_index = i;
			max_index = mat->nzero_index[i][col_index];
        }
	}

    if(max_i != NULL)
        *max_i = max_index;

    //return;
    return mu;
}

// 2024-08-01(Thu) T.Kouya
// Frobenius norm of mat
double normf_drsmatrix(DRSMatrix mat)
{
	long int index;
	double ret, ret_sum2;

	ret_sum2 = 0.0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d mat4, tmp4;

	// tmp4 := 0
	tmp4 = _mm256_setzero_pd();

	for(index = 0; index < mat->real_nzero_total_num; index += _BNC_D_WIDTH)
	{
		mat4 = _mm256_load_pd(&(mat->element[index]));

		// tmp4 += a4 * a4
		tmp4 = _mm256_fmadd_pd(mat4, mat4, tmp4);
	}

	ret_sum2 = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3];

#elif defined(__AVX512F__) // __AVX512F__
	__m512d mat8, tmp8;

	// tmp8 := 0
	tmp8 = _mm512_setzero_pd();

	for(index = 0; index < mat->real_nzero_total_num; index += _BNC_D_WIDTH)
	{
		mat8 = _mm512_load_pd(&(mat->element[index]));

		// tmp8 += a8 * a8
		tmp8= _mm512_fmadd_pd(mat8, mat8, tmp8);
	}

	ret_sum2 = tmp8[0] + tmp8[1] + tmp8[2] + tmp8[3] + tmp8[4] + tmp8[5] + tmp8[6] + tmp8[7];

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2
	{
		long _vl = (long)svcntd();
		svfloat64_t tmp_v = svdup_n_f64(0.0);
		for(index = 0; index < mat->real_nzero_total_num; index += _vl)
		{
			svbool_t pg = svwhilelt_b64_s64((int64_t)index,
			                                (int64_t)mat->real_nzero_total_num);
			svfloat64_t a_v = svld1_f64(pg, &(mat->element[index]));
			tmp_v = svmla_f64_x(pg, tmp_v, a_v, a_v);
		}
		ret_sum2 = svaddv_f64(svptrue_b64(), tmp_v);
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM Neon
	float64x2_t mat2, tmp2;

	// tmp2 := 0
	tmp2 = vdupq_n_f64(0.0);

	for(index = 0; index < mat->real_nzero_total_num; index += 2)
	{
		mat2 = vld1q_f64(&(mat->element[index]));

		// tmp2 += a2 * a2
		tmp2 = vfmaq_f64(tmp2, mat2, mat2);
	}

	ret_sum2 = vgetq_lane_f64(tmp2, 0) + vgetq_lane_f64(tmp2, 1);



#else // others
	//for(index = 0; index < mat->nzero_total_num; index++)
	for(index = 0; index < mat->real_nzero_total_num; index++)
		ret_sum2 += mat->element[index] * mat->element[index];

#endif // __AVX2__

	ret = sqrt(ret_sum2);

	return ret;
}

// 2024-07-30(Tue) T.Kouya
// SplitMat_A
void split_drsmatrix(DRSMatrix ret_high_mat, DRSMatrix ret_low_mat, DRSMatrix org_mat)
{
	long int i, j, row_dim, col_dim, real_total_dim, total_index;
	long int num_digits = 53; // IEEE double prec.
	//double *s;
    DRSMatrix s;
	double mu, abs_aij, t_exp;

    row_dim = org_mat->row_dim;
    col_dim = org_mat->col_dim;
    //row_dim = org_mat->real_row_dim;
    //col_dim = org_mat->real_col_dim;
    //real_total_dim = org_mat->nzero_total_num; // row_dim * col_dim;
	real_total_dim = org_mat->real_nzero_total_num; // row_dim * col_dim;

	// threshold matrix 
	//s = (double *)calloc(row_dim * col_dim, sizeof(double));
    s = init_set_drsmatrix(org_mat);
    set0_drsmatrix(s);

	// mu[i] = max_j |mat[i, j]|
    total_index = 0;
	for(i = 0; i < row_dim; i++)
	{
		//mu = fabs(mat[i * col_dim + 0]);
		/*for(j = 1; j < col_dim; j++)
		{
			abs_aij = fabs(mat[i * col_dim + j]);
			if(abs_aij > mu)
				mu = abs_aij;
		}*/
        mu = absmax_row_drsmatrix(NULL, i, org_mat);

		// t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
		//t_exp = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(col_dim + 1))) / 2.0);
		t_exp = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(org_mat->nzero_col_dim[i] + 1))) / 2.0);

		// s[i, j] = 2^t_exp
		//for(j = 0; j < org_mat->nzero_col_dim[i]; j++)
		for(j = 0; j < org_mat->real_nzero_col_dim[i]; j++)
        {
			//s[i * col_dim + j] = pow(2.0, t_exp);
            //set_drsmatrix_ij(s, i, j, pow(2.0, t_exp));
            s->element[total_index++] = pow(2.0, t_exp);
        }
	}

// split org_mat to ret_high_mat and ret_low_mat
#ifdef USE_CBLAS
	// tmp_mat := mat + s
	cblas_dcopy(real_total_dim, org_mat->element, 1, ret_high_mat->element, 1);
	cblas_daxpy(real_total_dim, 1.0, s->element, 1, ret_high_mat->element, 1);

	// high_mat := tmp_mat - s
	cblas_daxpy(real_total_dim, -1.0, s->element, 1, ret_high_mat->element, 1);

	// low_mat := mat - high_mat
	cblas_dcopy(real_total_dim, mat->element, 1, ret_low_mat->element, 1);
	cblas_daxpy(real_total_dim, -1.0, ret_high_mat->element, 1, ret_low_mat->element, 1);
#else // USE_CBLAS
	// tmp_mat := mat + s
	add_drsmatrix(ret_high_mat, org_mat, s);

	// high_mat := tmp_mat - s
	sub_drsmatrix(ret_high_mat, ret_high_mat, s);

	// low_mat := mat - high_mat
    sub_drsmatrix(ret_low_mat, org_mat, ret_high_mat);
#endif // USE_CBLAS

	// free s
	free_drsmatrix(s);
}

// 2024-08-05 (Mon) T.Kouya
// SplitMat_A
int split_drsmatrix_drsmat(DRSMatrix ret_mat[], int num_div, DRSMatrix org_mat)
{
	long int i, j, row_dim, col_dim, real_total_dim, total_index;
	long int num_digits = 53; // IEEE double prec.
	int index, real_num_div;
	//double *s;
    DRSMatrix s, in_ret_mat, tmp_mat[2], tmp_org_mat;
	double mu, abs_aij, t_exp, mu_total, power2;

    row_dim = org_mat->row_dim;
    col_dim = org_mat->col_dim;
    //row_dim = org_mat->real_row_dim;
    //col_dim = org_mat->real_col_dim;
    //real_total_dim = org_mat->nzero_total_num; // row_dim * col_dim;
	real_total_dim = org_mat->real_nzero_total_num; // row_dim * col_dim;

	// threshold matrix 
	//s = (double *)calloc(row_dim * col_dim, sizeof(double));
    s = init_set_drsmatrix(org_mat);
    set0_drsmatrix(s);

	tmp_mat[0] = init_set_drsmatrix(org_mat);
	tmp_mat[1] = init_set_drsmatrix(org_mat);
	if(ret_mat == NULL)
		in_ret_mat = init_set_drsmatrix(org_mat);

	tmp_org_mat = init_set_drsmatrix(org_mat);

	real_num_div = 0;
	for(index = 0; index < num_div; index++)
	{
		if(ret_mat != NULL)
			in_ret_mat = ret_mat[index];

		subst_drsmatrix(in_ret_mat, tmp_org_mat);

		// mu[i] = max_j |mat[i, j]|
		// mu_total += mu
		mu_total = 0.0;

		total_index = 0;
		for(i = 0; i < row_dim; i++)
		{
			//mu = fabs(mat[i * col_dim + 0]);
			/*for(j = 1; j < col_dim; j++)
			{
				abs_aij = fabs(mat[i * col_dim + j]);
				if(abs_aij > mu)
					mu = abs_aij;
			}*/
			mu = absmax_row_drsmatrix(NULL, i, in_ret_mat);
			mu_total += mu;

			// t_exp = ceil(log2(mu)) + ceil(s + log2(col_dim + 1) / 2)
			//t_exp = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(col_dim + 1))) / 2.0);
			//t_exp = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(org_mat->nzero_col_dim[i] + 1))) / 2.0);
			t_exp = ceil(DLOG2(mu)) + ceil(((double)num_digits + DLOG2((double)(in_ret_mat->nzero_col_dim[i] + 1))) / 2.0);

			// s[i, j] = 2^t_exp
			//for(j = 0; j < org_mat->nzero_col_dim[i]; j++)
			power2 = pow(2.0, t_exp);
			for(j = 0; j < org_mat->real_nzero_col_dim[i]; j++)
			{
				//s[i * col_dim + j] = pow(2.0, t_exp);
				//set_drsmatrix_ij(s, i, j, pow(2.0, t_exp));
				s->element[total_index++] = power2;
			}
		}

		// if ret_mat[index] == 0 -> break
		if(mu_total == 0.0) break;

	// split org_mat to ret_high_mat and ret_low_mat
	//#ifdef USE_CBLAS
	#ifdef USE_IMKL
		real_total_dim = in_ret_mat->real_nzero_total_num;

		// tmp_mat := mat + s
		//cblas_dcopy(real_total_dim, org_mat->element, 1, ret_high_mat->element, 1);
		//cblas_daxpy(real_total_dim, 1.0, s->element, 1, ret_high_mat->element, 1);
		cblas_daxpy(real_total_dim, 1.0, s->element, 1, in_ret_mat->element, 1);

		// high_mat := tmp_mat - s
		//cblas_daxpy(real_total_dim, -1.0, s->element, 1, ret_high_mat->element, 1);
		cblas_daxpy(real_total_dim, -1.0, s->element, 1, in_ret_mat->element, 1);

		// low_mat := mat - high_mat
		//cblas_dcopy(real_total_dim, mat->element, 1, ret_low_mat->element, 1);
		//cblas_daxpy(real_total_dim, -1.0, ret_high_mat->element, 1, ret_low_mat->element, 1);
	#else // USE_CBLAS
		// tmp_mat := mat + s
		//add_drsmatrix(ret_high_mat, org_mat, s);
		add_drsmatrix(tmp_mat[0], in_ret_mat, s);

		// high_mat := tmp_mat - s
		//sub_drsmatrix(ret_high_mat, ret_high_mat, s);
		sub_drsmatrix(tmp_mat[1], tmp_mat[0], s);

		// low_mat := mat - high_mat
		//sub_drsmatrix(ret_low_mat, org_mat, ret_high_mat);
		subst_drsmatrix(in_ret_mat, tmp_mat[1]);
	#endif // USE_CBLAS

		sub_drsmatrix(tmp_org_mat, tmp_org_mat, in_ret_mat);
		real_num_div = index + 1;
	}

	// free s
	free_drsmatrix(s);
    free_drsmatrix(tmp_org_mat);
    free_drsmatrix(tmp_mat[0]);
    free_drsmatrix(tmp_mat[1]);
	if(ret_mat == NULL)
		free_drsmatrix(in_ret_mat);

    return real_num_div;
}

// 2024-08-04 (Sun) T.Kouya
// SplitMat_B
// return real_num_div
int split_drsmatrix_t_drsmat(DRSMatrix ret_mat[], int num_div, DRSMatrix org_mat)
{
	long int i, j, index, row_dim, col_dim, real_total_dim;
	int real_num_div, num_digits = 53; // IEEE double prec.
	//int flag_stop = 0; //, num_digits = SPLIT_NUM_DIGITS; // IEEE double prec.
	//int flag_stop = 0, num_digits = 64; // IEEE double prec.
	//double *s;
    DRSMatrix tmp_org_mat;
    DRSMatrix s, tmp_mat[2], in_ret_mat;
	//double mu[DDSIZE], abs_aij[DDSIZE], t_exp[DDSIZE], power2[DDSIZE], two[DDSIZE] = {2.0, 0.0};
	double mu, abs_aij, t_exp, power2, mu_total;

    row_dim = org_mat->row_dim;
    //row_dim = org_mat->real_row_dim;
    col_dim = org_mat->col_dim;
    //real_total_dim = org_mat->real_row_dim * org_mat->real_col_dim;
	real_total_dim = org_mat->nzero_total_num;

	// threshold matrix 
	//s = (double *)calloc(row_dim * col_dim, sizeof(double));
    //s = init_ddmatrix(row_dim, col_dim);
    s = init_set_drsmatrix(org_mat); //row_dim, org_mat->nzero_col_dim, org_mat->nzero_total_num);
	set0_drsmatrix(s);

    //tmp_mat[0] = init_ddmatrix(row_dim, col_dim);
    //tmp_mat[1] = init_ddmatrix(row_dim, col_dim);
    tmp_mat[0] = init_set_drsmatrix(org_mat); //row_dim, org_mat->nzero_col_dim, org_mat->nzero_total_num);
    tmp_mat[1] = init_set_drsmatrix(org_mat); //row_dim, org_mat->nzero_col_dim, org_mat->nzero_total_num);
	if(ret_mat == NULL)
		in_ret_mat = init_set_drsmatrix(org_mat);

    // tmp_org_mat := org_mat;
    tmp_org_mat = init_set_drsmatrix(org_mat);
   	//subst_ddmatrix(tmp_org_mat, org_mat);

    real_num_div = 0;
    for(index = 0; index < num_div; index++)
    {
        //subst_dmatrix_ddmat(ret_mat[index], tmp_org_mat);
		if(ret_mat != NULL)
			in_ret_mat = ret_mat[index];

	    subst_drsmatrix(in_ret_mat, tmp_org_mat);

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
        //sub_ddrsmatrix_drsmat(tmp_org_mat, tmp_org_mat, ret_mat[index]);
        sub_drsmatrix(tmp_org_mat, tmp_org_mat, in_ret_mat);
        //sub_ddmatrix_dmat(tmp_org_mat, tmp_org_mat, tmp_mat[1]);

        real_num_div = index + 1;
    }

    // free s
	free_drsmatrix(s);
    free_drsmatrix(tmp_mat[0]);
    free_drsmatrix(tmp_mat[1]);
    free_drsmatrix(tmp_org_mat);
	if(ret_mat == NULL)
		free_drsmatrix(in_ret_mat);

    return real_num_div;
}

// Matrix-Vector multiplication based on Ozaki scheme
void mul_drsmatrix_dvec_oz(DVector ret, DRSMatrix a, int max_num_div_a, DVector vb, int max_num_div_vb) //, int num_digits)
{
    int i, j;
    int real_num_div_a, real_num_div_vb, *i_div_a_csr, *j_div_a_csr, total_index; // old
    long int vec_dim = ret->dim, row_dim = a->row_dim, col_dim = a->col_dim;
    DRSMatrix *div_a;
    DVector *div_vb, div_ret;
	// New for IMKL inspector-executor SPBLAS
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
	convert_indeces_drsmatrix_mkl_csrmat(i_div_a_csr_start, i_div_a_csr_end, j_div_a_csr, a);
	#endif // USE_IMKL_OLD
#endif // USE_IMKL

	for(i = 0; i < max_num_div_a; i++)
		div_a[i] = init_set_drsmatrix(a); // a->row_dim, a->nzero_col_dim, a->nzero_total_num);
        //div_a[i] = init_drsmatrix(row_dim, col_dim);

    for(i = 0; i < max_num_div_vb; i++)
        div_vb[i] = init_dvector(vec_dim);

    div_ret = init_dvector(vec_dim);

    real_num_div_a = split_drsmatrix_drsmat(div_a, max_num_div_a, a);
    real_num_div_vb = split_dvector_dvec(div_vb, max_num_div_vb, vb);

    set0_dvector(ret);
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
	#else // new
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
            add_dvector(ret, ret, div_ret);
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

// 2024-08-02(Fri) T.Kouya
// Transposed Matrix-Vector multiplication based on Ozaki scheme
void mul_drsmatrixt_dvec_oz(DVector ret, DRSMatrix a, int max_num_div_a, DVector vb, int max_num_div_vb)
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
	convert_indeces_drsmatrix_mkl_csrmat(i_div_a_csr_start, i_div_a_csr_end, j_div_a_csr, a);
	#endif // USE_IMKL_OLD
#endif // USE_IMKL_OLD

    for(i = 0; i < max_num_div_a; i++)
		div_a[i] = init_set_drsmatrix(a); // a->row_dim, a->nzero_col_dim, a->nzero_total_num);
        //div_a[i] = init_drsmatrix(row_dim, col_dim);

    for(i = 0; i < max_num_div_vb; i++)
        div_vb[i] = init_dvector(vec_dim);

    div_ret = init_dvector(vec_dim);

    //real_num_div_a = split_ddrsmatrix_drsmat(div_a, max_num_div_a, a);
    real_num_div_a = split_drsmatrix_t_drsmat(div_a, max_num_div_a, a);
    real_num_div_vb = split_dvector_dvec(div_vb, max_num_div_vb, vb);

    set0_dvector(ret);
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
			// div_ret := div_a^T * div_vb
            mul_drsmatrixt_dvec(div_ret, div_a[i], div_vb[j]);
#endif // USE_IMKL
			//printf("%ld,%ld ||div_ret||_F = %10.3e\n", i, j, norm2_dvector(div_ret));
            add_dvector(ret, ret, div_ret);
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

// Imcomplete LU decomposition; iLU0_drsmatrix
void iLU0_drsmatrix(DRSMatrix mat)
{
	long int i, j, k, row_dim, col_dim;
	double aii, aji, ajk, aik;

	row_dim = mat->row_dim;
	col_dim = mat->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		aii = get_drsmatrix_ij(mat, i, i);
		if(fabs(aii) != 0.0)
		{
			for(j = i + 1; j < row_dim; j++)
			{
				aji = get_drsmatrix_ij(mat, j, i);
				if(fabs(aji) != 0.0)
				{
					aji /= aii;
					set_drsmatrix_ij(mat, j, i, aji);
					for(k = i + 1; k < col_dim; k++)
					{
						ajk = get_drsmatrix_ij(mat, j, k);
						aik = get_drsmatrix_ij(mat, i, k);
						if((fabs(ajk) != 0.0) && (fabs(aik) != 0.0))
						{
							ajk = ajk - aji * aik;
							set_drsmatrix_ij(mat, j, k, ajk);
						}
					}
				}
			}
		}
	}
}

// iLU0_solve: iLU *  = b
void solve_iLU0_drsmatrix(DVector ret, DRSMatrix ilu, DVector b)
{
	long int i, j, row_dim, col_dim;
	double ret_i, ret_j, ilu_ii, ilu_ji, ilu_ij;

	row_dim = ilu->row_dim;
	col_dim = ilu->col_dim;

	// ret := b
	subst_dvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		ilu_ii = get_drsmatrix_ij(ilu, i, i);
		ret_i = get_dvector_i(ret, i);
		if(fabs(ilu_ii) != 0.0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				ilu_ji = get_drsmatrix_ij(ilu, j, i);
				if(fabs(ilu_ji) != 0.0)
				{
					ret_j = get_dvector_i(ret, j);
					ret_j = ret_j - ilu_ji * ret_i;
					set_dvector_i(ret, j, ret_j);
				}
			}
		}
	}

	// Backward substitution
	for(i = (row_dim - 1); i >= 0; i--)
	{
		ilu_ii = get_drsmatrix_ij(ilu, i, i);
		ret_i = get_dvector_i(ret, i);
		if(fabs(ilu_ii) != 0.0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				ilu_ij = get_drsmatrix_ij(ilu, i, j);
				if(fabs(ilu_ij) != 0.0)
				{
					ret_j = get_dvector_i(ret, j);
					ret_i = ret_i - ilu_ij * ret_j;
					set_dvector_i(ret, i, ret_i);
				}
			}
			ret_i /= ilu_ii;
			set_dvector_i(ret, i, ret_i);
		}
	}
}

// iLU0t_solve: x^T * iLU = b^T
void solve_iLU0t_drsmatrix(DVector ret, DRSMatrix ilu, DVector b)
{
	long int i, j, row_dim, col_dim;
	double ret_i, ret_j, ilu_ii, ilu_ji;

	row_dim = ilu->row_dim;
	col_dim = ilu->col_dim;

	// ret := b
	subst_dvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		ilu_ii = get_drsmatrix_ij(ilu, i, i);
		ret_i = get_dvector_i(ret, i);
		if(fabs(ilu_ii) != 0.0)
		{
			for(j = 0; j < i; j++)
			{
				ilu_ji = get_drsmatrix_ij(ilu, j, i);
				if(fabs(ilu_ji) != 0.0)
				{
					ret_j = get_dvector_i(ret, j);
					ret_i = ret_i - ilu_ji * ret_j;
					//set_cdvector_i(ret, i, ret_i);
				}
			}
			ret_i /= ilu_ii;
			set_dvector_i(ret, i, ret_i);
		}
	}

	// Backward substitution
	for(i = row_dim - 1; i >= 0; i--)
	{
		ilu_ii = get_drsmatrix_ij(ilu, i, i);
		ret_i = get_dvector_i(ret, i);
		if(fabs(ilu_ii) != 0.0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				ilu_ji = get_drsmatrix_ij(ilu, j, i);
				if(fabs(ilu_ji) != 0.0)
				{
					ret_j = get_dvector_i(ret, j);
					ret_i = ret_i - ilu_ji * ret_j;
					//set_cdvector_i(ret, j, ret_j);
				}
			}
			set_dvector_i(ret, i, ret_i);
		}
	}
}
