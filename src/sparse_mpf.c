/********************************************************************************/
/*                                                                              */
/* sparse_mpf.c : Sparse Matrix and Vector Library (Multiple Precision)         */
/* Copyright (c) 2011 Tomonori Kouya, All rights reserved.                      */
/*                                                                              */
/* Version 0.1 2011-08-29 : Bug fix in mul_mpfrsmatrix_mpfvec                   */
/* Version 0.1 2012-03-23 : Bug fix in init2_mpfrsmatrix                        */
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

#ifdef USE_GMP

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

/* initialize MPFRSMatrix */
MPFRSMatrix init_mpfrsmatrix(long int row_dim, long int *nzero_col_dim, long int nzero_total_num)
{
	MPFRSMatrix ret;
	long int i, j;

	ret = (MPFRSMatrix)malloc(sizeof(mpfrsmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "Cannot allocate MPFRSMatrix\n");
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
	ret->prec = get_bnc_default_prec();
	//mpf_init2(ret->zero_element, ret->prec);
	mpf_init(ret->zero_element);
	mpf_set_ui(ret->zero_element, 0UL);

	/* allocate nzero_index */
	//printf("%ld %ld %ld %ld\n", ret->row_dim, ret->col_dim, ret->nzero_total_num, sizeof(long int *) * row_dim);
	// ret->nzero_col_dim = (long int *)malloc(sizeof(long int *) * row_dim);
	// ret->nzero_row_dim = (long int *)malloc(sizeof(long int *) * ret->col_dim);
	// ret->nzero_index = (long int **)malloc(sizeof(long int *) * row_dim);
	// Fix it!: 2024-04-25(Thu) T.Kouya
	ret->nzero_col_dim = (long int *)calloc(ret->row_dim, sizeof(long int));
	ret->nzero_row_dim = (long int *)calloc(ret->col_dim, sizeof(long int));
	ret->nzero_index = (long int **)calloc(ret->row_dim, sizeof(long int *));

	if(ret->nzero_index == NULL)
	{
		fprintf(stderr, "Cannot allocate MPFRSMatrix(nzero_index!)\n");
		return NULL;
	}
	for(i = 0; i < row_dim; i++)
	{
		if(nzero_col_dim[i] < 0)
		{
			fprintf(stderr, "Illigal nzero values(nzero_col_dim[%ld])\n", i);
			return NULL;
		}
		ret->nzero_col_dim[i] = nzero_col_dim[i];
		//ret->nzero_index[i] = (long int *)malloc(sizeof(long int) * nzero_col_dim[i]);
		ret->nzero_index[i] = (long int *)calloc(nzero_col_dim[i], sizeof(long int));
		if(ret->nzero_index[i] == NULL)
		{
			fprintf(stderr, "Cannot allocate MPFRSMatrix(nzero_index[%ld]!)\n", i);
			return NULL;
		}
		for(j = 0; j < nzero_col_dim[i]; j++)
			ret->nzero_index[i][j] = EMPTY;
	}
	
	/* allocate element */
	//ret->element = (mpf_ptr)malloc(sizeof(mpf_t) * nzero_total_num);
	ret->element = (mpf_t *)calloc(nzero_total_num, sizeof(mpf_t));
	if(ret->element == NULL)
	{
		fprintf(stderr, "Cannot allocate MPFRSMatrix(element!)\n");
		return NULL;
	}
//	for(i = 0; i < nzero_total_num; i++)
//		*(ret->element + i) = 0.0;
	for(i = 0; i < nzero_total_num; i++)
	{
		//mpf_init(*(ret->element + i));
		//mpf_set_ui(*(ret->element + i), 0UL);
		mpf_init(ret->element[i]);
		mpf_set_ui(ret->element[i], 0UL);
//		mpf_init_set_ui(*(ret->element + i), 0UL);
	}

	return ret;
}

/* initialize MPFRSMatrix with specific precision */
MPFRSMatrix init2_mpfrsmatrix(long int row_dim, long int *nzero_col_dim, long int nzero_total_num, unsigned long prec)
{
	MPFRSMatrix ret;
	long int i, j;

	ret = (MPFRSMatrix)malloc(sizeof(mpfrsmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "Cannot allocate MPFRSMatrix\n");
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
	ret->prec = prec; // Fix!
	mpf_init2(ret->zero_element, ret->prec);
	mpf_set_ui(ret->zero_element, 0UL);

	/* allocate nzero_index */
	//printf("%ld %ld %ld %ld\n", ret->row_dim, ret->col_dim, ret->nzero_total_num, sizeof(long int *) * row_dim);
	//ret->nzero_col_dim = (long int *)malloc(sizeof(long int *) * row_dim);
	//ret->nzero_row_dim = (long int *)malloc(sizeof(long int *) * ret->col_dim);
	//ret->nzero_index = (long int **)malloc(sizeof(long int *) * row_dim);
	// Fix it!: 2024-04-25(Thu) T.Kouya
	ret->nzero_col_dim = (long int *)calloc(ret->row_dim, sizeof(long int));
	ret->nzero_row_dim = (long int *)calloc(ret->col_dim, sizeof(long int));
	ret->nzero_index = (long int **)calloc(ret->row_dim, sizeof(long int *));

	if(ret->nzero_index == NULL)
	{
		fprintf(stderr, "Cannot allocate MPFRSMatrix(nzero_index!)\n");
		return NULL;
	}
	for(i = 0; i < row_dim; i++)
	{
		if(nzero_col_dim[i] < 0)
		{
			fprintf(stderr, "Illigal nzero values(nzero_col_dim[%ld])\n", i);
			return NULL;
		}
		ret->nzero_col_dim[i] = nzero_col_dim[i];
		//ret->nzero_index[i] = (long int *)malloc(sizeof(long int) * nzero_col_dim[i]);
		ret->nzero_index[i] = (long int *)calloc(nzero_col_dim[i], sizeof(long int));
		if(ret->nzero_index[i] == NULL)
		{
			fprintf(stderr, "Cannot allocate MPFRSMatrix(nzero_index[%ld]!)\n", i);
			return NULL;
		}
		for(j = 0; j < nzero_col_dim[i]; j++)
			ret->nzero_index[i][j] = EMPTY;
	}
	
	/* allocate element */
	//ret->element = (mpf_ptr)malloc(sizeof(mpf_t) * nzero_total_num);
	ret->element = (mpf_t *)calloc(nzero_total_num, sizeof(mpf_t));
	if(ret->element == NULL)
	{
		fprintf(stderr, "Cannot allocate MPFRSMatrix(element!)\n");
		return NULL;
	}
//	for(i = 0; i < nzero_total_num; i++)
//		*(ret->element + i) = 0.0;
	for(i = 0; i < nzero_total_num; i++)
	{
		//mpf_init2(*(ret->element + i), prec);
		//mpf_set_ui(*(ret->element + i), 0UL);
		mpf_init2(ret->element[i], prec);
		mpf_set_ui(ret->element[i], 0UL);
//		mpf_init_set_ui(*(ret->element + i), 0UL);
	}

	return ret;
}

/* Clear MPFRSMatrix */
void free_mpfrsmatrix(MPFRSMatrix mat)
{
	long int i;

//	free(mat->element);
	free(mat->nzero_col_dim);
	free(mat->nzero_row_dim);
	mpf_clear(mat->zero_element);

	for(i = 0; i < mat->row_dim; i++)
		free(mat->nzero_index[i]);

	free(mat->nzero_index);

	//free(mat);
	if(mat->element != NULL)
	{
		for(i = 0; i < mat->nzero_total_num; i++)
			mpf_clear(mat->element[i]);
			//mpf_clear(*(mat->element + i));

		free(mat->element);
	}

	free(mat);
}

/* set nzero_row_dim automatically */
void set_nzero_row_dim_mpf(MPFRSMatrix mat)
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

/* Print MPFRSMatrix */
void print_mpfrsmatrix(MPFRSMatrix mat)
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
			for(j = 0; j < mat->nzero_col_dim[i] - 1 ; j++)
			{
				// printf("%ld->%f, ", mat->nzero_index[i][j], mat->element[total_index++]);
				printf("%ld-> ", mat->nzero_index[i][j]);
				mpf_out_str(stdout, 10, 0, mat->element[total_index++]);
				printf(", ");
			}
			//printf("%ld->%f\n", mat->nzero_index[i][mat->nzero_col_dim[i] - 1], mat->element[total_index++]);
			printf("%ld->", mat->nzero_index[i][mat->nzero_col_dim[i] - 1]);
			mpf_out_str(stdout, 10, 0, mat->element[total_index++]);
			printf("\n");
		}
	}
	return;
}

// 2024-10-16(Wed) T.Kouya
/* get the MPFRSMatrix ij-element */
void get_mpfrsmatrix_ij(mpf_t ret, MPFRSMatrix mat, long int row_index, long int col_index)
{
	long int i, j, total_index;

	if((row_index < 0) || (row_index >= mat->row_dim) || (col_index < 0) || (col_index >= mat->col_dim))
	{
		fprintf(stderr, "Warning: row_index(%ld) or col_index(%ld) is illegal!\n", row_index, col_index);
		mpf_set_ui(ret, 0UL); //rdd_set0(ret);
		//return mat->zero_element;
		return;
	}

	// finding mat_ij element
	if(mat->nzero_col_dim[row_index] >= 1)
	{
		total_index = 0;
		for(i = 0; i < row_index; i++)
			total_index += mat->nzero_col_dim[i];
			//total_index += mat->real_nzero_col_dim[i];

		for(j = 0; j < mat->nzero_col_dim[row_index] ; j++)
		{
			// Find!
			if(mat->nzero_index[row_index][j] == col_index)
			{
				//return mat->element[total_index];
				//ret = mat->element[total_index];
				mpf_set(ret, mat->element[total_index]);

				return;
			}
			
			total_index++;
		}
	}

	// Not found -> return 0
	//return mat->zero_element;
	mpf_set_ui(ret, 0UL); //rdd_set0(ret);
	return;
}

// 2024-10-16 (Wed) T.Kouya
/* set the MPFRSMatrix ij-element */
void set_mpfrsmatrix_ij(MPFRSMatrix mat, long int row_index, long int col_index, mpf_t val)
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
			total_index += mat->nzero_col_dim[i];
			//total_index += mat->real_nzero_col_dim[i];

		for(j = 0; j < mat->nzero_col_dim[row_index] ; j++)
		{
			// Find!
			if(mat->nzero_index[row_index][j] == col_index)
			{
				//return mat->element[total_index];
				//ret[0] = mat->element[0][total_index];
				//ret[1] = mat->element[1][total_index];
				//mat->element[total_index] = val;
				mpf_set(mat->element[total_index], val);

				return;
			}
			
			total_index++;
		}
	}

	return;
}

// 2024-10-16(Wed) T.Kouya
// init and set MPFRSMatrix
MPFRSMatrix init_set_mpfrsmatrix(MPFRSMatrix org_sp)
{
	MPFRSMatrix ret;
	long int org_sp_total_index, total_index, i, j;

    //printf("row_dim, nzero_total_num = %ld, %ld\n", org_sp->row_dim, org_sp->nzero_total_num);
    ret = init_mpfrsmatrix(org_sp->row_dim, org_sp->nzero_col_dim, org_sp->nzero_total_num);
    //printf("init_ddrsmatrix!\n");

	 // Real total number of non-zero elements
	//ret->real_nzero_total_num = org_sp->real_nzero_total_num;

    // copy nzero_index
	for(i = 0; i < org_sp->row_dim; i++)
	{
		// Real numbers of non-zero elements in i-th row
		//ret->real_nzero_col_dim[i] = org_sp->real_nzero_col_dim[i];

		for(j = 0; j < org_sp->nzero_col_dim[i]; j++)
			ret->nzero_index[i][j] = org_sp->nzero_index[i][j];
	}

	// copy values
    //for(total_index = 0; total_index < org_sp->real_nzero_total_num; total_index++)
    for(total_index = 0; total_index < org_sp->nzero_total_num; total_index++)	
	{
		//ret->element[total_index] = org_sp->element[total_index];
		mpf_set(ret->element[total_index], org_sp->element[total_index]);
	}

	return ret;
}

// 2024-10-16(Wed) T.Kouya
// spmat := 0
void set0_mpfrsmatrix(MPFRSMatrix spmat)
{
	long int index;

	// substitution
	//for(index = 0; index < spmat->real_nzero_total_num; index++)
	for(index = 0; index < spmat->nzero_total_num; index++)
	{
		//spmat->element[index] = (double)0.0;
		mpf_set_ui(spmat->element[index], 0UL);
	}
		

	return;
}


// 2024-09-05(Thu) T.Kouya
// Frobenius norm of mat
void normf_mpfrsmatrix(mpf_t ret, MPFRSMatrix mat)
{
	long int index;
	//double ret, ret_sum2;
	mpf_t ret_sum2;

	mpf_init2(ret_sum2, mat->prec);

	//ret_sum2 = 0.0;
	mpf_set_ui(ret_sum2, 0UL);
	//for(index = 0; index < mat->nzero_total_num; index++)
	for(index = 0; index < mat->nzero_total_num; index++)
		mpfr_fma(ret_sum2, mat->element[index], mat->element[index], ret_sum2, MPFR_RNDN);

	//ret = sqrt(ret_sum2);
	mpf_sqrt(ret, ret_sum2);

	mpf_clear(ret_sum2);
}

/* Dense Matrix := Sparse Matrix */
void set_mpfmatrix_mpfrsmat(MPFMatrix ret, MPFRSMatrix spmat)
{
	long int i, j, total_index;

	if((ret == NULL) || (spmat == NULL))
	{
		fprintf(stderr, "set_mpfmatrix_mpfrsmatrix: ERROR!\n");
		return;
	}

	// ret := 0
	set0_mpfmatrix(ret);

	total_index = 0;
	for(i = 0; i < spmat->row_dim; i++)
	{
		for(j = 0; j < spmat->nzero_col_dim[i]; j++)
		{
			//printf("%5ld: ", i);
			//printf("%ld->%e, ", spmat->nzero_index[i][j], spmat->element[total_index++]);
			set_mpfmatrix_ij(ret, i, spmat->nzero_index[i][j], spmat->element[total_index++]);
		}
	}
	return;
}

/* initialize and substitute MPFRSMatrix from MPFMatrix */
MPFRSMatrix init_set_mpfrsmatrix_mpfmatrix(MPFMatrix org_mat)
{
	long int i, j;
	long int nzero_total_num, total_index, j_index;
	long int *ptr_nzero_col_dim;
	MPFRSMatrix ret;

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
			if(mpf_cmp_ui(get_mpfmatrix_ij(org_mat, i, j), 0UL) != 0)
			{
				/* get nzero_row_dim, nzero_col_dim, nzero_total_num */
				nzero_total_num++;
				*(ptr_nzero_col_dim + i) += 1;
			}
		}
	}

	// initialize
	ret = init2_mpfrsmatrix(org_mat->row_dim, ptr_nzero_col_dim, nzero_total_num, org_mat->prec);

	/* Read org_mat (2nd) */
	total_index = 0;
	for(i = 0; i < ret->row_dim; i++)
	{
		j_index = 0;
		for(j = 0; j < ret->col_dim; j++)
		{
			if(mpf_cmp_ui(get_mpfmatrix_ij(org_mat, i, j), 0UL) != 0)
			{
				ret->nzero_index[i][j_index] =  j;
				mpf_set(*(ret->element + total_index), get_mpfmatrix_ij(org_mat, i, j));
				total_index += 1;
				j_index += 1;
			}
		}
	}

	free(ptr_nzero_col_dim);

	return ret;
}

/* Get variables to initialize MPFRSMatrix */
int get_vars_mpfrsmatrix_fname(long int *ptr_row_dim, long int **ptr_nzero_col_dim, long int *ptr_nzero_total_num, const char *fname)
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

/* Read URI linking data */
int fread_urilinkdat_fname_mpf(MPFRSMatrix ret, const char *fname)
{
	FILE *fp;
	static char line_buf[LINE_BUF_LEN];
	static long int *nzero_row_dim;
	long int i, j;
	long int row_index, col_index, max_row_index, max_col_index, nzero_total_num;
	long int ele_index;
	unsigned long prec;
	mpf_t tmp;

	// init tmp
	prec = ret->prec;
	mpf_init2(tmp, prec);

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
			//*(ret->element + ele_index) = 1.0 / (mpf_t)nzero_row_dim[ret->nzero_index[i][j]];
			mpf_set_ui(tmp, (unsigned long)nzero_row_dim[ret->nzero_index[i][j]]);
			mpf_ui_div(*(ret->element + ele_index), 1UL, tmp);

			//printf("%ld %ld %ld -> %f\n", ele_index, i, j, *(ret->element + ele_index));
			ele_index++;
		}
	}

	mpf_clear(tmp);
	free(nzero_row_dim);

	/* file close */
	fclose(fp);

//	print_MPFRSMatrix(ret);

	return SUCCESS;
}


/* Multiply MPFRSMatrix * MPFVector */
int mul_mpfrsmatrix_mpfvec(MPFVector ret, MPFRSMatrix mat, MPFVector vec)
{
	long int i, j, total_index;
	unsigned long prec;
	mpf_t tmp;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	prec = ret->prec;
	mpf_init2(tmp, prec);

	total_index = 0;
	for(i = 0; i < mat->row_dim; i++)
	{
		//get_mpfvector_i(ret, i) = 0.0;
		mpf_set_ui(get_mpfvector_i(ret, i), 0UL);
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			//get_mpfvector_i(ret, i) += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			//mpf_mul(tmp, mat->element[total_index], get_mpfvector_i(vec, mat->nzero_index[i][j]]));
			//mpf_mul(tmp, get_mpfvector_i(mat, total_index), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			mpf_mul(tmp, (mpf_ptr)(mat->element[total_index]), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			mpf_add(get_mpfvector_i(ret, i), get_mpfvector_i(ret, i), tmp);
			total_index++;
		}
	}

	mpf_clear(tmp);

	return SUCCESS;
}

/* Multiply MPFRSMatrix^T * MPFVector */
int mul_mpfrsmatrixt_mpfvec(MPFVector ret, MPFRSMatrix mat, MPFVector vec)
{
	long int i, j, total_index;
	unsigned long prec;
	mpf_t tmp, vi;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	prec = ret->prec;
	mpf_init2(tmp, prec);
	mpf_init2(vi, prec);

	set0_mpfvector(ret);
	//for(i = 0; i < mat->row_dim; i++)
	//	mpf_set_ui(get_mpfvector_i(ret, i), 0UL);

	total_index = 0;
	//for(i = 0; i < mat->row_dim; i++)
	for(i = 0; i < mat->row_dim; i++)
	{
		mpf_set(vi, get_mpfvector_i(vec, i));
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
			mpf_mul(tmp, (mpf_ptr)(mat->element[total_index]), vi); // get_mpfvector_i(vec, i));
			mpf_add((mpf_ptr)(ret->element[mat->nzero_index[i][j]]), (mpf_ptr)(ret->element[mat->nzero_index[i][j]]), tmp);
			total_index++;
		}
	}

	mpf_clear(tmp);
	mpf_clear(vi);

	return SUCCESS;
}

/* Multiply DRSMatrix * MPFVector */
int mul_drsmatrix_mpfvec(MPFVector ret, DRSMatrix mat, MPFVector vec)
{
	long int i, j, total_index;
	unsigned long prec;
	mpf_t tmp;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	prec = ret->prec;
	mpf_init2(tmp, prec);

	total_index = 0;
	for(i = 0; i < mat->row_dim; i++)
	{
		//get_mpfvector_i(ret, i) = 0.0;
		mpf_set_ui(get_mpfvector_i(ret, i), 0UL);
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			//get_mpfvector_i(ret, i) += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			//mpf_mul(tmp, mat->element[total_index], get_mpfvector_i(vec, mat->nzero_index[i][j]]));
			//mpf_mul(tmp, get_mpfvector_i(mat, total_index), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			//mpf_mul(tmp, (mpf_ptr)(mat->element[total_index]), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			mpf_mul_d(tmp, get_mpfvector_i(vec, mat->nzero_index[i][j]), mat->element[total_index]);
			mpf_add(get_mpfvector_i(ret, i), get_mpfvector_i(ret, i), tmp);
			total_index++;
		}
		total_index += (mat->real_nzero_col_dim[i] - mat->nzero_col_dim[i]);
	}

	mpf_clear(tmp);

	return SUCCESS;
}

/* Multiply DRSMatrix^T * MPFVector */
int mul_drsmatrixt_mpfvec(MPFVector ret, DRSMatrix mat, MPFVector vec)
{
	long int i, j, total_index;
	unsigned long prec;
	mpf_t tmp, vi;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	prec = ret->prec;
	mpf_init2(tmp, prec);
	mpf_init2(vi, prec);

	set0_mpfvector(ret);
	//for(i = 0; i < mat->row_dim; i++)
	//	mpf_set_ui(get_mpfvector_i(ret, i), 0UL);

	total_index = 0;
	//for(i = 0; i < mat->row_dim; i++)
	for(i = 0; i < mat->row_dim; i++)
	{
		mpf_set(vi, get_mpfvector_i(vec, i));
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
			//mpf_mul(tmp, (mpf_ptr)(mat->element[total_index]), vi); // get_mpfvector_i(vec, i));
			mpf_mul_d(tmp, vi, mat->element[total_index]); // get_mpfvector_i(vec, i));
			mpf_add((mpf_ptr)(ret->element[mat->nzero_index[i][j]]), (mpf_ptr)(ret->element[mat->nzero_index[i][j]]), tmp);
			total_index++;
		}
		total_index += (mat->real_nzero_col_dim[i] - mat->nzero_col_dim[i]);
	}

	mpf_clear(tmp);
	mpf_clear(vi);

	return SUCCESS;
}

/* Multiply MPFRSMatrix^T * MPFVector */
int mul_mpfrsmatrixt_mpfvec_old(MPFVector ret, MPFRSMatrix mat, MPFVector vec)
{
	long int i, j, total_index;
	unsigned long prec;
	mpf_t tmp;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	prec = ret->prec;
	mpf_init2(tmp, prec);

	for(i = 0; i < mat->row_dim; i++)
		mpf_set_ui(get_mpfvector_i(ret, i), 0UL);

	total_index = 0;
	//for(i = 0; i < mat->row_dim; i++)
	for(i = 0; i < mat->nzero_col_dim[i]; i++)
	{
		//for(j = 0; j < mat->nzero_col_dim[i]; j++)
		for(j = 0; j < mat->row_dim; j++)
		{
			//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
			//mpf_mul(tmp, get_mpfvector_i(mat, total_index), get_mpfvector_i(vec, i));
			mpf_mul(tmp, (mpf_ptr)(mat->element[mat->nzero_index[j][i]]), get_mpfvector_i(vec, j));
			//mpf_add(ret->element[mat->nzero_index[j][j]], ret->element[mat->nzero_index[i][j]], tmp);
			mpf_add(get_mpfvector_i(ret, j), get_mpfvector_i(ret, j), tmp);
			//mpf_add(get_mpfvector_i(ret, mat->nzero_index[i][j]), get_mpfvector_i(ret, mat->nzero_index[i][j]), tmp);
			total_index++;
		}
	}

	mpf_clear(tmp);

	return SUCCESS;
}

/* Power Method for Randomly Sparse Matrices */
/* 	mpf_t *evec: the eigenvector for max eigenvalue */
/* 	mpf_t *drsmat: Randomly sparse matrix */
/* 	mpf_t reps, aeps: Relative and Absolute tolerance */
/* 	long int max_times: Maximum iterative times of Power method */
void mpfpower_rsmatrix(mpf_t max_eig, MPFVector evec, MPFRSMatrix mat, mpf_t reps, mpf_t aeps, long int max_times)
{
	long int i, absmax_index, times;
	mpf_t absmax_new_evec, old_max_eig, tmp, tmp2;
	MPFVector new_evec;
	unsigned long prec;

	prec = evec->prec;

	mpf_init2(absmax_new_evec, prec);
//	mpf_init2(max_eig, prec);
	mpf_init2(old_max_eig, prec);
	mpf_init2(tmp, prec);
	mpf_init2(tmp2, prec);

	new_evec = init2_mpfvector(mat->row_dim, prec);

	/* initialize evec */
	for(i = 0; i < evec->dim; i++)
		mpf_set_ui(get_mpfvector_i(evec, i), 1UL);
		//evec->element[i] = 1.0;

	/* main loop */
	//old_max_eig = 0.0;
	mpf_set_ui(old_max_eig, 0UL);
	for(times = 0; times < max_times; times++)
	{
		/* w := A * x */
		mul_mpfrsmatrix_mpfvec(new_evec, mat, evec);
		absmax_index = absmax_index_mpfvector(absmax_new_evec, new_evec);
//		max_eig = absmax_new_evec / evec->element[absmax_index];
		mpf_div(max_eig, absmax_new_evec, get_mpfvector_i(evec, absmax_index));
//		smul_mpfvector(evec, 1.0 / absmax_new_evec, new_evec);
//		smul_mpfvector(evec, 1.0 / norm1_MPFVector(new_evec), new_evec); // Baba's example
		norm1_mpfvector(tmp, new_evec);
		mpf_ui_div(tmp, 1UL, tmp);
		smul_mpfvector(evec, tmp, new_evec);

//		if((fabs(max_eig - old_max_eig) <= reps * fabs(old_max_eig) + aeps) && (times >= 2))
		mpf_sub(tmp, max_eig, old_max_eig);
		mpf_abs(tmp, tmp);
		
		mpf_abs(tmp2, old_max_eig);
		mpf_mul(tmp2, reps, tmp2);
		mpf_add(tmp2, tmp2, aeps);
		if((mpf_cmp(tmp, tmp2) <= 0) && (times >= 2))
		{
			fprintf(stderr, "Convergent!(Iterative Times = %ld)\n", times);
			break;
		}
		if(times % 10 == 0)
			fprintf(stderr, "%5ld %25.17e\n", times, mpf_get_d(max_eig));
		//old_max_eig = max_eig;
		mpf_set(old_max_eig, max_eig);
	}

//	mpf_clear(max_eig);
	mpf_clear(old_max_eig);
	mpf_clear(absmax_new_evec);
	mpf_clear(tmp);
	mpf_clear(tmp2);

	free_mpfvector(new_evec);

//	return max_eig;
	return;
}

/* Scalar multiply of MPFVector */
int smul_mpfvector(MPFVector ret, mpf_t scalar, MPFVector vec)
{
	long int i;

	if(ret->dim < vec->dim)
		return ERROR;

	for(i = 0; i < vec->dim; i++)
		mpf_mul(get_mpfvector_i(ret, i),  scalar, get_mpfvector_i(vec, i));

	return SUCCESS;
}

/* Select index of absolute maximum element and its value in MPFVector */
long int absmax_index_mpfvector(mpf_t ret, MPFVector vec)
{
	long int absmax_index, i;
	mpf_t abs_element;

	mpf_init2(abs_element, mpf_get_prec(ret));

	mpf_set_ui(ret, 0UL);
	absmax_index = 0;
	for(i = 0; i < vec->dim; i++)
	{
		mpf_abs(abs_element, get_mpfvector_i(vec, i));
		if(mpf_cmp(ret, abs_element) < 0)
		{
			absmax_index = i;
			//*ret = abs_element;
		}
	}

	mpf_set(ret, get_mpfvector_i(vec, absmax_index));

	mpf_clear(abs_element);

	return absmax_index;
}


// Incomplete LU decomposition; iLU0_drsmatrix
void iLU0_mpfrsmatrix(MPFRSMatrix mat)
{
	unsigned long prec;
	long int i, j, k, row_dim, col_dim;
	mpf_t aii, aji, ajk, aik, ctmp;
	mpf_t dtmp, dtmp1;
	//double dtmp[DDSIZE];

	row_dim = mat->row_dim;
	col_dim = mat->col_dim;
	prec = mat->prec;

	mpf_init2(aii, prec);
	mpf_init2(aji, prec);
	mpf_init2(ajk, prec);
	mpf_init2(aik, prec);
	mpf_init2(ctmp, prec);
	mpf_init2(dtmp, prec);
	mpf_init2(dtmp1, prec);

	for(i = 0; i < row_dim; i++)
	{
		//mpf_set(aii, get_mpfrsmatrix_ij(mat, i, i));
		get_mpfrsmatrix_ij(aii, mat, i, i);
		mpf_abs(dtmp, aii);
		if(mpf_cmp_ui(dtmp, 0UL) != 0)
		{
			for(j = i + 1; j < row_dim; j++)
			{
				//mpf_set(aji, get_mpfrsmatrix_ij(mat, j, i));
				get_mpfrsmatrix_ij(aji, mat, j, i);
				mpf_abs(dtmp, aji);
				if(mpf_cmp_ui(dtmp, 0UL) != 0)
				{
					//aji /= aii;
					mpf_div(aji, aji, aii);
					set_mpfrsmatrix_ij(mat, j, i, aji);
					for(k = i + 1; k < col_dim; k++)
					{
						//mpf_set(ajk, get_mpfrsmatrix_ij(mat, j, k));
						//mpf_set(aik, get_mpfrsmatrix_ij(mat, i, k));
						get_mpfrsmatrix_ij(ajk, mat, j, k);
						get_mpfrsmatrix_ij(aik, mat, i, k);
						mpf_abs(dtmp, ajk);
						mpf_abs(dtmp1, aik);
						if((mpf_cmp_ui(dtmp, 0UL) != 0) && (mpf_cmp_ui(dtmp1, 0UL) != 0))
						{
							//ajk = ajk - aji * aik;
							mpf_mul(ctmp, aji, aik);
							mpf_sub(ajk, ajk, ctmp);
							set_mpfrsmatrix_ij(mat, j, k, ajk);
							//printf("%ld, %ld, %ld\n", i, j, k);
						}
					}
				}
			}
		}
	}

	mpf_clear(aii);
	mpf_clear(aji);
	mpf_clear(ajk);
	mpf_clear(aik);
	mpf_clear(ctmp);
	mpf_clear(dtmp);
	mpf_clear(dtmp1);
}

// iLU0_solve: iLU * x = b
void solve_iLU0_mpfrsmatrix(MPFVector ret, MPFRSMatrix ilu, MPFVector b)
{
	unsigned long prec;
	long int i, j, row_dim, col_dim;
	mpf_t ret_i, ret_j, ilu_ii, ilu_ij, ctmp;
	mpf_t dtmp;

	prec = ret->prec;
	row_dim = ilu->row_dim;
	col_dim = ilu->col_dim;

	mpf_init2(ret_i, prec);
	mpf_init2(ret_j, prec);
	mpf_init2(ilu_ii, prec);
	mpf_init2(ilu_ij, prec);
	mpf_init2(ctmp, prec);
	mpf_init2(dtmp, prec);

	// ret := b
	subst_mpfvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		get_mpfrsmatrix_ij(ilu_ii, ilu, i, i);
		mpf_set(ret_i, get_mpfvector_i(ret, i));
		mpf_abs(dtmp, ilu_ii);
		if(mpf_cmp_ui(dtmp, 0UL) != 0)
		{
			for(j = 0; j < i; j++)
			{
				get_mpfrsmatrix_ij(ilu_ij, ilu, i, j);
				mpf_abs(dtmp, ilu_ij);
				if(mpf_cmp_ui(dtmp, 0UL) != 0)
				{
					mpf_set(ret_j, get_mpfvector_i(ret, j));
					//ret_j = ret_j - ilu_ji * ret_i;
					mpf_mul(ctmp, ilu_ij, ret_j);
					mpf_sub(ret_i, ret_i, ctmp);
					//set_cddvector_i(ret, j, &ret_j);
					//printf("f: %ld, %ld\n", i, j);
				}
			}
			set_mpfvector_i(ret, i, ret_i);
		}
	}

	// Backward substitution
	for(i = (row_dim - 1); i >= 0; i--)
	{
		get_mpfrsmatrix_ij(ilu_ii, ilu, i, i);
		mpf_set(ret_i, get_mpfvector_i(ret, i));
		mpf_abs(dtmp, ret_i);
		if(mpf_cmp_ui(dtmp, 0UL) != 0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				get_mpfrsmatrix_ij(ilu_ij, ilu, i, j);
				mpf_abs(dtmp, ilu_ij);
				if(mpf_cmp_ui(dtmp, 0UL) != 0)
				{
					mpf_set(ret_j, get_mpfvector_i(ret, j));
					//ret_i = ret_i - ilu_ij * ret_j;
					mpf_mul(ctmp, ilu_ij, ret_j);
					mpf_sub(ret_i, ret_i, ctmp);
					//set_cddvector_i(ret, i, &ret_i);
					//printf("b: %ld, %ld\n", i, j);
				}
			}
			//ret_i /= ilu_ii;
			mpf_div(ret_i, ret_i, ilu_ii);
			set_mpfvector_i(ret, i, ret_i);
		}
	}

	mpf_clear(ret_i);
	mpf_clear(ret_j);
	mpf_clear(ilu_ii);
	mpf_clear(ilu_ij);
	mpf_clear(ctmp);
	mpf_clear(dtmp);
}

// iLU0t_solve: x^t * (iLU) = b^t
void solve_iLU0t_mpfrsmatrix(MPFVector ret, MPFRSMatrix ilu, MPFVector b)
{
	unsigned long prec;
	long int i, j, row_dim, col_dim;
	mpf_t ret_i, ret_j, ilu_ii, ilu_ji, ctmp;
	mpf_t dtmp, dtmp1;
	//double dtmp[DDSIZE];

	row_dim = ilu->row_dim;
	col_dim = ilu->col_dim;
	prec = ilu->prec;

	mpf_init2(ret_i, prec);
	mpf_init2(ret_j, prec);
	mpf_init2(ilu_ii, prec);
	mpf_init2(ilu_ji, prec);
	mpf_init2(ctmp, prec);
	mpf_init2(dtmp, prec);

	// ret := b
	subst_mpfvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		get_mpfrsmatrix_ij(ilu_ii, ilu, i, i);
		mpf_set(ret_i, get_mpfvector_i(ret, i));
		mpf_abs(dtmp, ilu_ii);
		if(mpf_cmp_ui(dtmp, 0UL) != 0)
		{
			for(j = 0; j < i; j++)
			{
				get_mpfrsmatrix_ij(ilu_ji, ilu, j, i);
				mpf_abs(dtmp, ilu_ji);
				if(mpf_cmp_ui(dtmp, 0UL) != 0)
				{
					mpf_set(ret_j, get_mpfvector_i(ret, j));;
					//ret_j = ret_j - ilu_ji * ret_i;
					mpf_mul(ctmp, ilu_ji, ret_j);
					mpf_sub(ret_i, ret_i, ctmp);
					//printf("f: %ld, %ld\n", i, j);
				}
			}
			mpf_div(ret_i, ret_i, ilu_ii);
			set_mpfvector_i(ret, i, ret_i);
		}
	}

	// Backward substitution
	for(i = (row_dim - 1); i >= 0; i--)
	{
		get_mpfrsmatrix_ij(ilu_ii, ilu, i, i);
		mpf_set(ret_i, get_mpfvector_i(ret, i));
		mpf_abs(dtmp, ilu_ii);
		if(mpf_cmp_ui(dtmp, 0UL) != 0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				get_mpfrsmatrix_ij(ilu_ji, ilu, j, i);
				mpf_abs(dtmp, ilu_ji);
				if(mpf_cmp_ui(dtmp, 0UL) != 0)
				{
					mpf_set(ret_j, get_mpfvector_i(ret, j));
					//ret_i = ret_i - ilu_ij * ret_j;
					mpf_mul(ctmp, ilu_ji, ret_j);
					mpf_sub(ret_i, ret_i, ctmp);
					//set_cddvector_i(ret, i, &ret_i);
					//printf("b: %ld, %ld\n", i, j);
				}
			}
			set_mpfvector_i(ret, i, ret_i);
		}
	}

	mpf_clear(ret_i);
	mpf_clear(ret_j);
	mpf_clear(ilu_ii);
	mpf_clear(ilu_ji);
	mpf_clear(ctmp);
	mpf_clear(dtmp);
}

// iLU0_solve: iLU * x = b
void solve_iLU0_drsmatrix_mpfvec(MPFVector ret, DRSMatrix ilu, MPFVector b)
{
	unsigned long prec;
	long int i, j, row_dim, col_dim;
	mpf_t ret_i, ret_j, ctmp;
	double ilu_ii, ilu_ij;
	mpf_t dtmp, dtmp1;
	//double dtmp[DDSIZE];

	row_dim = ilu->row_dim;
	col_dim = ilu->col_dim;
	prec = ret->prec;

	mpf_init2(ret_i, prec);
	mpf_init2(ret_j, prec);
	mpf_init2(ctmp, prec);
	mpf_init2(dtmp, prec);
	mpf_init2(dtmp1, prec);

	// ret := b
	subst_mpfvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		ilu_ii = get_drsmatrix_ij(ilu, i, i);
		mpf_set(ret_i, get_mpfvector_i(ret, i));
		if(fabs(ilu_ii) != 0.0)
		{
			for(j = 0; j < i; j++)
			{
				ilu_ij = get_drsmatrix_ij(ilu, i, j);
				if(fabs(ilu_ij) != 0.0)
				{
					mpf_set(ret_j, get_mpfvector_i(ret, j));
					//ret_j = ret_j - ilu_ji * ret_i;
					mpf_mul_d(ctmp, ret_j, ilu_ij);
					mpf_sub(ret_i, ret_i, ctmp);
					//printf("f: %ld, %ld\n", i, j);
				}
			}
			set_mpfvector_i(ret, i, ret_i);
		}
	}

	// Backward substitution
	for(i = (row_dim - 1); i >= 0; i--)
	{
		ilu_ii = get_drsmatrix_ij(ilu, i, i);
		mpf_set(ret_i, get_mpfvector_i(ret, i));
		if(fabs(ilu_ii) != 0.0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				ilu_ij = get_drsmatrix_ij(ilu, i, j);
				if(fabs(ilu_ij) != 0.0)
				{
					mpf_set(ret_j, get_mpfvector_i(ret, j));
					//ret_i = ret_i - ilu_ij * ret_j;
					mpf_mul_d(ctmp, ret_j, ilu_ij);
					mpf_sub(ret_i, ret_i, ctmp);
					//set_cddvector_i(ret, i, &ret_i);
					//printf("b: %ld, %ld\n", i, j);
				}
			}
			//ret_i /= ilu_ii;
			mpf_div_d(ret_i, ret_i, ilu_ii);
			set_mpfvector_i(ret, i, ret_i);
		}
	}

	mpf_clear(ret_i);
	mpf_clear(ret_j);
	mpf_clear(ctmp);
	mpf_clear(dtmp);
	mpf_clear(dtmp1);
}

// iLU0t_solve: x^t(iLU) = b^t
void solve_iLU0t_drsmatrix_mpfvec(MPFVector ret, DRSMatrix ilu, MPFVector b)
{
	unsigned long prec;
	long int i, j, row_dim, col_dim;
	mpf_t ret_i, ret_j, ctmp;
	double ilu_ii, ilu_ji;
	mpf_t dtmp, dtmp1;
	//double dtmp[DDSIZE];

	row_dim = ilu->row_dim;
	col_dim = ilu->col_dim;
	prec = ret->prec;

	mpf_init2(ret_i, prec);
	mpf_init2(ret_j, prec);
	mpf_init2(ctmp, prec);
	mpf_init2(dtmp, prec);

	// ret := b
	subst_mpfvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		ilu_ii = get_drsmatrix_ij(ilu, i, i);
		mpf_set(ret_i, get_mpfvector_i(ret, i));
		if(fabs(ilu_ii) != 0.0)
		{
			for(j = 0; j < i; j++)
			{
				ilu_ji = get_drsmatrix_ij(ilu, j, i);
				if(fabs(ilu_ji) != 0.0)
				{
					mpf_set(ret_j, get_mpfvector_i(ret, j));
					//ret_j = ret_j - ilu_ji * ret_i;
					mpf_mul_d(ctmp, ret_j, ilu_ji);
					mpf_sub(ret_i, ret_i, ctmp);
					//printf("f: %ld, %ld\n", i, j);
				}
			}
			mpf_div_d(ret_i, ret_i, ilu_ii);
			set_mpfvector_i(ret, i, ret_i);
		}
	}

	// Backward substitution
	for(i = (row_dim - 1); i >= 0; i--)
	{
		ilu_ii = get_drsmatrix_ij(ilu, i, i);
		mpf_set(ret_i, get_mpfvector_i(ret, i));
		if(fabs(ilu_ii) != 0.0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				ilu_ji = get_drsmatrix_ij(ilu, j, i);
				if(fabs(ilu_ji) != 0.0)
				{
					mpf_set(ret_j, get_mpfvector_i(ret, j));
					//ret_i = ret_i - ilu_ij * ret_j;
					mpf_mul_d(ctmp, ret_j, ilu_ji);
					mpf_sub(ret_i, ret_i, ctmp);
					//set_cddvector_i(ret, i, &ret_i);
					//printf("b: %ld, %ld\n", i, j);
				}
			}
			set_mpfvector_i(ret, i, ret_i);
		}
	}

	mpf_clear(ret_i);
	mpf_clear(ret_j);
	mpf_clear(ctmp);
	mpf_clear(dtmp);
}
#endif // USE_GMP
