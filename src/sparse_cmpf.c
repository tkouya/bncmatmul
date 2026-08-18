/********************************************************************************/
/*                                                                              */
/* sparse_cmpf.c : Complex Sparse Matrix and Vector Library (Multiple Precision)*/
/* Copyright (c) 2024 Tomonori Kouya, All rights reserved.                      */
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
// ret := a * b
void _bnc_mpc_mul_cd(mpc_t ret, mpc_t a, double _Complex b)
{
	// 4M method
	mpf_ptr ret_re, ret_im, a_re, a_im;
	mpf_t tmp;

	//printf("_bnc_mpc_mul_cd start!\n");
	mpf_init2(tmp, mpc_get_prec(ret));

	ret_re = mpc_realref(ret);
	ret_im = mpc_imagref(ret);
	a_re = mpc_realref(a);
	a_im = mpc_imagref(a);

	// real part: ret.re := a.re * b.re - a.im * b.im
	mpf_mul_d(ret_re, a_re, creal(b));
	mpf_mul_d(tmp, a_im, cimag(b));
	mpf_sub(ret_re, ret_re, tmp);

	// imag part: ret_im := a.re * b.im + a.im * a.re
	mpf_mul_d(ret_im, a_re, cimag(b));
	mpf_mul_d(tmp, a_im, creal(b));
	mpf_add(ret_im, ret_im, tmp);

	mpf_clear(tmp);
}

// ret := a / b
void _bnc_mpc_div_cd(mpc_t ret, mpc_t a, double _Complex b)
{
	// 4M method
	mpf_ptr ret_re, ret_im, a_re, a_im;
	mpf_t tmp, abs_b2;

	//printf("_bnc_mpc_mul_cd start!\n");
	mpf_init2(tmp, mpc_get_prec(ret));
	mpf_init2(abs_b2, mpc_get_prec(ret));

	ret_re = mpc_realref(ret);
	ret_im = mpc_imagref(ret);
	a_re = mpc_realref(a);
	a_im = mpc_imagref(a);

	// ret := (a.re + a.im * I) * (b.re - b.im * I) / (b.re^2 + b.im^2)

	// real part: ret.re := a.re * b.re + a.im * b.im
	mpf_mul_d(ret_re, a_re, creal(b));
	mpf_mul_d(tmp, a_im, cimag(b));
	mpf_add(ret_re, ret_re, tmp);

	// imag part: ret.im := a.im * b.re - a.re * b.im
	mpf_mul_d(ret_im, a_re, cimag(b));
	mpf_mul_d(tmp, a_im, creal(b));
	mpf_sub(ret_im, tmp, ret_im); // , tmp);

	// b2 := b.re ^ 2 + b.im ^ 2
	mpf_set_d(abs_b2, creal(b));
	mpf_mul_d(abs_b2, abs_b2, creal(b));
	mpf_set_d(tmp, cimag(b));
	mpf_mul_d(tmp, tmp, cimag(b));
	mpf_add(abs_b2, abs_b2, tmp);

	mpf_div(ret_re, ret_re, abs_b2);
	mpf_div(ret_im, ret_im, abs_b2);

	mpf_clear(tmp);
	mpf_clear(abs_b2);
}

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

/* initialize CMPFRSMatrix */
CMPFRSMatrix init_cmpfrsmatrix(long int row_dim, long int *nzero_col_dim, long int nzero_total_num)
{
	CMPFRSMatrix ret;
	long int i, j;

	ret = (CMPFRSMatrix)malloc(sizeof(cmpfrsmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "Cannot allocate CMPFRSMatrix\n");
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
	//mpc_init2(ret->zero_element, ret->prec);
	mpc_init(ret->zero_element);
	mpc_set_si(ret->zero_element, 0L, MPC_RNDNN);

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
		fprintf(stderr, "Cannot allocate CMPFRSMatrix(nzero_index!)\n");
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
			fprintf(stderr, "Cannot allocate CMPFRSMatrix(nzero_index[%ld]!)\n", i);
			return NULL;
		}
		for(j = 0; j < nzero_col_dim[i]; j++)
			ret->nzero_index[i][j] = EMPTY;
	}
	
	/* allocate element */
	//ret->element = (mpc_ptr)malloc(sizeof(mpc_t) * nzero_total_num);
	ret->element = (mpc_t *)calloc(nzero_total_num, sizeof(mpc_t));
	if(ret->element == NULL)
	{
		fprintf(stderr, "Cannot allocate CMPFRSMatrix(element!)\n");
		return NULL;
	}
//	for(i = 0; i < nzero_total_num; i++)
//		*(ret->element + i) = 0.0;
	for(i = 0; i < nzero_total_num; i++)
	{
		//mpc_init(*(ret->element + i));
		//mpc_set_si(*(ret->element + i), 0UL);
		mpc_init(ret->element[i]);
		mpc_set_si(ret->element[i], 0L, MPC_RNDNN);
//		mpc_init_set_ui(*(ret->element + i), 0UL);
	}

	return ret;
}

/* initialize CMPFRSMatrix with specific precision */
CMPFRSMatrix init2_cmpfrsmatrix(long int row_dim, long int *nzero_col_dim, long int nzero_total_num, unsigned long prec)
{
	CMPFRSMatrix ret;
	long int i, j;

	ret = (CMPFRSMatrix)malloc(sizeof(cmpfrsmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "Cannot allocate CMPFRSMatrix\n");
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
	mpc_init2(ret->zero_element, ret->prec);
	mpc_set_si(ret->zero_element, 0L, MPC_RNDNN);

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
		fprintf(stderr, "Cannot allocate CMPFRSMatrix(nzero_index!)\n");
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
			fprintf(stderr, "Cannot allocate CMPFRSMatrix(nzero_index[%ld]!)\n", i);
			return NULL;
		}
		for(j = 0; j < nzero_col_dim[i]; j++)
			ret->nzero_index[i][j] = EMPTY;
	}
	
	/* allocate element */
	//ret->element = (mpc_ptr)malloc(sizeof(mpc_t) * nzero_total_num);
	ret->element = (mpc_t *)calloc(nzero_total_num, sizeof(mpc_t));
	if(ret->element == NULL)
	{
		fprintf(stderr, "Cannot allocate CMPFRSMatrix(element!)\n");
		return NULL;
	}
//	for(i = 0; i < nzero_total_num; i++)
//		*(ret->element + i) = 0.0;
	for(i = 0; i < nzero_total_num; i++)
	{
		//mpc_init2(*(ret->element + i), prec);
		//mpc_set_si(*(ret->element + i), 0UL);
		mpc_init2(ret->element[i], prec);
		mpc_set_si(ret->element[i], 0L, MPC_RNDNN);
//		mpc_init_set_ui(*(ret->element + i), 0UL);
	}

	return ret;
}

/* Clear CMPFRSMatrix */
void free_cmpfrsmatrix(CMPFRSMatrix mat)
{
	long int i;

//	free(mat->element);
	free(mat->nzero_col_dim);
	free(mat->nzero_row_dim);
	mpc_clear(mat->zero_element);

	for(i = 0; i < mat->row_dim; i++)
		free(mat->nzero_index[i]);

	free(mat->nzero_index);

	//free(mat);
	if(mat->element != NULL)
	{
		for(i = 0; i < mat->nzero_total_num; i++)
			mpc_clear(mat->element[i]);
			//mpc_clear(*(mat->element + i));

		free(mat->element);
	}

	free(mat);
}

/* set nzero_row_dim automatically */
void set_nzero_row_dim_cmpf(CMPFRSMatrix mat)
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

/* Print CMPFRSMatrix */
void print_cmpfrsmatrix(CMPFRSMatrix mat)
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
				mpc_out_str(stdout, 10, 0, mat->element[total_index++], MPC_RNDNN);
				printf(", ");
			}
			//printf("%ld->%f\n", mat->nzero_index[i][mat->nzero_col_dim[i] - 1], mat->element[total_index++]);
			printf("%ld->", mat->nzero_index[i][mat->nzero_col_dim[i] - 1]);
			mpc_out_str(stdout, 10, 0, mat->element[total_index++], MPC_RNDNN);
			printf("\n");
		}
	}
	return;
}

// 2024-12-04(Tue)
/* get the CMPFRSMatrix ij-element */
mpc_ptr get_cmpfrsmatrix_ij(CMPFRSMatrix mat, long int row_index, long int col_index)
{
	long int i, j, total_index;
	mpc_ptr ret = NULL; //  = 0.0 + 0.0 * I;

	if((row_index < 0) || (row_index >= mat->row_dim) || (col_index < 0) || (col_index >= mat->col_dim))
	{
		fprintf(stderr, "Warning: row_index(%ld) or col_index(%ld) is illegal!\n", row_index, col_index);
		return ret; // mat->zero_element;
	}

	// finding mat_ij element
	if(mat->nzero_col_dim[row_index] >= 1)
	{
		total_index = 0;
		for(i = 0; i < row_index; i++)
			total_index += mat->nzero_col_dim[i];

		for(j = 0; j < mat->nzero_col_dim[row_index] ; j++)
		{
			// Find!
			if(mat->nzero_index[row_index][j] == col_index)
			{
				ret = mat->element[total_index];
				return ret; // mat->element[total_index];
			}
			
			total_index++;
		}
	}

	// Not found -> return 0
	//return ret; // mat->re->zero_element;
	return mat->zero_element;
}

// 2024-12-04 (Tue) T.Kouya
/* set the CMPFRSMatrix ij-element */
void set_cmpfrsmatrix_ij(CMPFRSMatrix mat, long int row_index, long int col_index, mpc_t val)
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

		for(j = 0; j < mat->nzero_col_dim[row_index] ; j++)
		{
			// Find!
			if(mat->nzero_index[row_index][j] == col_index)
			{
				//return mat->element[total_index];
				//mat->element[total_index] = val;
				mpc_set(mat->element[total_index], val, MPC_RNDNN);
	
				return;
			}
			
			total_index++;
		}
	}

	return;
}

// 2024-12-17 (Tue) T.Kouya
// init and set CMPFRSMatrix
CMPFRSMatrix init_set_cmpfrsmatrix(CMPFRSMatrix org_sp)
{
	CMPFRSMatrix ret;
	long int org_sp_total_index, total_index, i, j;

    //printf("row_dim, nzero_total_num = %ld, %ld\n", org_sp->row_dim, org_sp->nzero_total_num);
    ret = init_cmpfrsmatrix(org_sp->row_dim, org_sp->nzero_col_dim, org_sp->nzero_total_num);
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
		mpc_set(ret->element[total_index], org_sp->element[total_index], MPC_RNDNN);
	}

	return ret;
}


// 2024-09-05(Thu) T.Kouya
// Frobenius norm of mat
void normf_cmpfrsmatrix(mpf_t ret, CMPFRSMatrix mat)
{
	long int index;
	//double ret, ret_sum2;
	mpf_t ret_sum2;

	mpf_init2(ret_sum2, mat->prec);

	//ret_sum2 = 0.0;
	mpf_set_ui(ret_sum2, 0UL);
	//for(index = 0; index < mat->nzero_total_num; index++)
	for(index = 0; index < mat->nzero_total_num; index++)
	{
		mpfr_fma(
			ret_sum2, 
			mpc_realref(mat->element[index]), mpc_realref(mat->element[index]),
			ret_sum2, MPFR_RNDN
		);
		mpfr_fma(
			ret_sum2, 
			mpc_imagref(mat->element[index]), mpc_imagref(mat->element[index]),
			ret_sum2, MPFR_RNDN
		);
	}

	//ret = sqrt(ret_sum2);
	mpf_sqrt(ret, ret_sum2);

	mpf_clear(ret_sum2);
}

/* Dense Matrix := Sparse Matrix */
void set_cmpfmatrix_cmpfrsmat(CMPFMatrix ret, CMPFRSMatrix spmat)
{
	long int i, j, total_index;

	if((ret == NULL) || (spmat == NULL))
	{
		fprintf(stderr, "set_mpfmatrix_cmpfrsmatrix: ERROR!\n");
		return;
	}

	// ret := 0
	set0_cmpfmatrix(ret);

	total_index = 0;
	for(i = 0; i < spmat->row_dim; i++)
	{
		for(j = 0; j < spmat->nzero_col_dim[i]; j++)
		{
			//printf("%5ld: ", i);
			//printf("%ld->%e, ", spmat->nzero_index[i][j], spmat->element[total_index++]);
			set_cmpfmatrix_ij(ret, i, spmat->nzero_index[i][j], spmat->element[total_index++]);
		}
	}
	return;
}

/* initialize and substitute CMPFRSMatrix from CMPFMatrix */
CMPFRSMatrix init_set_cmpfrsmatrix_cmpfmatrix(CMPFMatrix org_mat)
{
	long int i, j;
	long int nzero_total_num, total_index, j_index;
	long int *ptr_nzero_col_dim;
	CMPFRSMatrix ret;
    //mpf_t tmp;

	/* initialize variables */
	nzero_total_num = 0;
    //mpf_init2(tmp, org_mat->prec);

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
            //mpc_abs(tmp, get_cmpfmatrix_ij(org_mat, i, j));
			//if(mpc_cmp_ui(get_cmpfmatrix_ij(org_mat, i, j), 0UL) != 0)
			if(mpc_cmp_si(get_cmpfmatrix_ij(org_mat, i, j), 0L) != 0)
			{
				/* get nzero_row_dim, nzero_col_dim, nzero_total_num */
				nzero_total_num++;
				*(ptr_nzero_col_dim + i) += 1;
			}
		}
	}

	// initialize
	ret = init2_cmpfrsmatrix(org_mat->row_dim, ptr_nzero_col_dim, nzero_total_num, org_mat->prec);

	/* Read org_mat (2nd) */
	total_index = 0;
	for(i = 0; i < ret->row_dim; i++)
	{
		j_index = 0;
		for(j = 0; j < ret->col_dim; j++)
		{
			if(mpc_cmp_si(get_cmpfmatrix_ij(org_mat, i, j), 0L) != 0)
			{
				ret->nzero_index[i][j_index] =  j;
				mpc_set(*(ret->element + total_index), get_cmpfmatrix_ij(org_mat, i, j), MPC_RNDNN);
				total_index += 1;
				j_index += 1;
			}
		}
	}

    //mpf_clear(tmp);
	free(ptr_nzero_col_dim);

	return ret;
}

/* Get variables to initialize CMPFRSMatrix */
int get_vars_cmpfrsmatrix_fname(long int *ptr_row_dim, long int **ptr_nzero_col_dim, long int *ptr_nzero_total_num, const char *fname)
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

/* Multiply CMPFRSMatrix * CMPFVector */
int mul_cmpfrsmatrix_cmpfvec(CMPFVector ret, CMPFRSMatrix mat, CMPFVector vec)
{
	long int i, j, total_index;
	unsigned long prec;
	mpc_t tmp;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	prec = ret->prec;
	mpc_init2(tmp, prec);

	total_index = 0;
	for(i = 0; i < mat->row_dim; i++)
	{
		//get_mpfvector_i(ret, i) = 0.0;
		mpc_set_si(get_cmpfvector_i(ret, i), 0UL, MPC_RNDNN);
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			//get_mpfvector_i(ret, i) += mat->element[total_index] * vec->element[mat->nzero_index[i][j]];
			//mpc_mul(tmp, mat->element[total_index], get_mpfvector_i(vec, mat->nzero_index[i][j]]));
			//mpc_mul(tmp, get_mpfvector_i(mat, total_index), get_mpfvector_i(vec, mat->nzero_index[i][j]));
			mpc_mul(tmp, (mpc_ptr)(mat->element[total_index]), get_cmpfvector_i(vec, mat->nzero_index[i][j]), MPC_RNDNN);
			mpc_add(get_cmpfvector_i(ret, i), get_cmpfvector_i(ret, i), tmp, MPC_RNDNN);
			total_index++;
		}
	}

	mpc_clear(tmp);

	return SUCCESS;
}

/* Multiply CMPFRSMatrix^T * MPFVector */
int mul_cmpfrsmatrixt_cmpfvec(CMPFVector ret, CMPFRSMatrix mat, CMPFVector vec)
{
	long int i, j, total_index;
	unsigned long prec;
	mpc_t tmp;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	prec = ret->prec;
	mpc_init2(tmp, prec);

	for(i = 0; i < mat->row_dim; i++)
		mpc_set_si(get_cmpfvector_i(ret, i), 0L, MPC_RNDNN);

	total_index = 0;
	//for(i = 0; i < mat->row_dim; i++)
	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
			mpc_mul(tmp, (mpc_ptr)(mat->element[total_index]), get_cmpfvector_i(vec, i), MPC_RNDNN);
			mpc_add((mpc_ptr)(ret->element[mat->nzero_index[i][j]]), (mpc_ptr)(ret->element[mat->nzero_index[i][j]]), tmp, MPC_RNDNN);
			total_index++;
		}
	}

	mpc_clear(tmp);

	return SUCCESS;
}

/* Multiply conj(CMPFRSMatrix)^T * MPFVector */
int mul_cmpfrsmatrixs_cmpfvec(CMPFVector ret, CMPFRSMatrix mat, CMPFVector vec)
{
	long int i, j, total_index;
	unsigned long prec;
	mpc_t tmp, mat_ij_conj;

	if((ret->dim < mat->col_dim) || (vec->dim != mat->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	prec = ret->prec;
	mpc_init2(tmp, prec);
	mpc_init2(mat_ij_conj, prec);

	for(i = 0; i < mat->row_dim; i++)
		mpc_set_si(get_cmpfvector_i(ret, i), 0L, MPC_RNDNN);

	total_index = 0;
	//for(i = 0; i < mat->row_dim; i++)
	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->nzero_col_dim[i]; j++)
		{
			//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
			mpc_conj(mat_ij_conj, (mpc_ptr)(mat->element[total_index]), MPC_RNDNN);
			//mpc_mul(tmp, (mpc_ptr)(mat->element[total_index]), get_cmpfvector_i(vec, i), MPC_RNDNN);
			mpc_mul(tmp, mat_ij_conj, get_cmpfvector_i(vec, i), MPC_RNDNN);
			mpc_add((mpc_ptr)(ret->element[mat->nzero_index[i][j]]), (mpc_ptr)(ret->element[mat->nzero_index[i][j]]), tmp, MPC_RNDNN);
			total_index++;
		}
	}

	mpc_clear(tmp);
	mpc_clear(mat_ij_conj);

	return SUCCESS;
}

/* Scalar multiply of CMPFVector */
int smul_cmpfvector(CMPFVector ret, mpc_t scalar, CMPFVector vec)
{
	long int i;

	if(ret->dim < vec->dim)
		return ERROR;

	for(i = 0; i < vec->dim; i++)
		mpc_mul(get_cmpfvector_i(ret, i),  scalar, get_cmpfvector_i(vec, i), MPC_RNDNN);

	return SUCCESS;
}

/* Select index of absolute maximum element and its value in MPFVector */
long int absmax_index_cmpfvector(mpc_t ret, CMPFVector vec)
{
	long int absmax_index, i;
	mpf_t abs_element, abs_ret;

	mpf_init2(abs_element, mpc_get_prec(ret));
	mpf_init2(abs_ret, mpc_get_prec(ret));

	mpc_set_si(ret, 0UL, MPC_RNDNN);
	mpf_set_ui(abs_ret, 0UL); // ret = 0 -> abs_ret = 0
	absmax_index = 0;
	for(i = 0; i < vec->dim; i++)
	{
		mpc_abs(abs_element, get_cmpfvector_i(vec, i), MPC_RNDNN);
		if(mpf_cmp(abs_ret, abs_element) < 0)
		{
			absmax_index = i;
			//*ret = abs_element;
			mpf_set(abs_ret, abs_element);
		}
	}

	mpc_set(ret, get_cmpfvector_i(vec, absmax_index), MPC_RNDNN);

	mpf_clear(abs_element);
	mpf_clear(abs_ret);

	return absmax_index;
}

/* Multiply CDRSMatrix * CMPFVector */
int mul_cdrsmatrix_cmpfvec(CMPFVector ret, CDRSMatrix mat, CMPFVector vec)
{
	long int i, j, total_index;
	unsigned long prec;
	mpc_t tmp;
	double _Complex mat_ij;

	if((ret->dim < mat->re->col_dim) || (vec->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	prec = ret->prec;
	mpc_init2(tmp, prec);

	total_index = 0;
	for(i = 0; i < mat->re->row_dim; i++)
	{
		//get_mpfvector_i(ret, i) = 0.0;
		mpc_set_si(get_cmpfvector_i(ret, i), 0UL, MPC_RNDNN);
		for(j = 0; j < mat->re->nzero_col_dim[i]; j++)
		{
			//mpc_mul(tmp, (mpc_ptr)(mat->element[total_index]), get_cmpfvector_i(vec, mat->nzero_index[i][j]), MPC_RNDNN);
			mat_ij = mat->re->element[total_index] + mat->im->element[total_index] * I;
			_bnc_mpc_mul_cd(tmp, get_cmpfvector_i(vec, mat->re->nzero_index[i][j]), mat_ij);
			mpc_add(get_cmpfvector_i(ret, i), get_cmpfvector_i(ret, i), tmp, MPC_RNDNN);
			total_index++;
		}
		total_index += (mat->re->real_nzero_col_dim[i] - mat->re->nzero_col_dim[i]);
	}

	mpc_clear(tmp);

	return SUCCESS;
}

/* Multiply CDRSMatrix^T * MPFVector */
int mul_cdrsmatrixt_cmpfvec(CMPFVector ret, CDRSMatrix mat, CMPFVector vec)
{
	long int i, j, total_index;
	unsigned long prec;
	mpc_t tmp;
	double _Complex mat_ij;

	if((ret->dim < mat->re->col_dim) || (vec->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	prec = ret->prec;
	mpc_init2(tmp, prec);

	for(i = 0; i < mat->re->row_dim; i++)
		mpc_set_si(get_cmpfvector_i(ret, i), 0L, MPC_RNDNN);

	total_index = 0;
	//for(i = 0; i < mat->row_dim; i++)
	for(i = 0; i < mat->re->row_dim; i++)
	{
		for(j = 0; j < mat->re->nzero_col_dim[i]; j++)
		{
			//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
			mat_ij = mat->re->element[total_index] + mat->im->element[total_index] * I;
			_bnc_mpc_mul_cd(tmp, get_cmpfvector_i(vec, i), mat_ij);
			mpc_add((mpc_ptr)(ret->element[mat->re->nzero_index[i][j]]), (mpc_ptr)(ret->element[mat->re->nzero_index[i][j]]), tmp, MPC_RNDNN);
			total_index++;
		}
		total_index += (mat->re->real_nzero_col_dim[i] - mat->re->nzero_col_dim[i]);
	}

	mpc_clear(tmp);

	return SUCCESS;
}

/* Multiply conj(CDRSMatrix)^T * MPFVector */
int mul_cdrsmatrixs_cmpfvec(CMPFVector ret, CDRSMatrix mat, CMPFVector vec)
{
	long int i, j, total_index;
	unsigned long prec;
	mpc_t tmp;
	double _Complex mat_ij_conj;

	if((ret->dim < mat->re->col_dim) || (vec->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	prec = ret->prec;
	mpc_init2(tmp, prec);

	for(i = 0; i < mat->re->row_dim; i++)
		mpc_set_si(get_cmpfvector_i(ret, i), 0L, MPC_RNDNN);

	total_index = 0;
	//for(i = 0; i < mat->row_dim; i++)
	for(i = 0; i < mat->re->row_dim; i++)
	{
		for(j = 0; j < mat->re->nzero_col_dim[i]; j++)
		{
			//ret->element[mat->nzero_index[i][j]] += mat->element[total_index] * vec->element[i];
			//mat_ij = mat->re->element[total_index] + mat->im->element[total_index] * I;
			mat_ij_conj = mat->re->element[total_index] - mat->im->element[total_index] * I;
			_bnc_mpc_mul_cd(tmp, get_cmpfvector_i(vec, i), mat_ij_conj);
			mpc_add((mpc_ptr)(ret->element[mat->re->nzero_index[i][j]]), (mpc_ptr)(ret->element[mat->re->nzero_index[i][j]]), tmp, MPC_RNDNN);
			total_index++;
		}
		total_index += (mat->re->real_nzero_col_dim[i] - mat->re->nzero_col_dim[i]);
	}

	mpc_clear(tmp);

	return SUCCESS;
}

// Imcomplete LU decomposition; iLU0_drsmatrix
void iLU0_cmpfrsmatrix(CMPFRSMatrix mat)
{
	unsigned long prec;
	long int i, j, k, row_dim, col_dim;
	mpc_t aii, aji, ajk, aik, ctmp;
	mpc_rnd_t rnd = get_bnc_default_rounding_mode_c();
	mpf_t dtmp, dtmp1;
	//double dtmp[DDSIZE];

	row_dim = mat->row_dim;
	col_dim = mat->col_dim;
	prec = mat->prec;

	mpc_init2(aii, prec);
	mpc_init2(aji, prec);
	mpc_init2(ajk, prec);
	mpc_init2(aik, prec);
	mpc_init2(ctmp, prec);
	mpf_init2(dtmp, prec);
	mpf_init2(dtmp1, prec);

	for(i = 0; i < row_dim; i++)
	{
		mpc_set(aii, get_cmpfrsmatrix_ij(mat, i, i), rnd);
		mpc_abs(dtmp, aii, rnd);
		if(mpf_cmp_ui(dtmp, 0UL) != 0)
		{
			for(j = i + 1; j < row_dim; j++)
			{
				mpc_set(aji, get_cmpfrsmatrix_ij(mat, j, i), rnd);
				mpc_abs(dtmp, aji, rnd);
				if(mpf_cmp_ui(dtmp, 0UL) != 0)
				{
					//aji /= aii;
					mpc_div(aji, aji, aii, rnd);
					set_cmpfrsmatrix_ij(mat, j, i, aji);
					for(k = i + 1; k < col_dim; k++)
					{
						mpc_set(ajk, get_cmpfrsmatrix_ij(mat, j, k), rnd);
						mpc_set(aik, get_cmpfrsmatrix_ij(mat, i, k), rnd);
						mpc_abs(dtmp, ajk, rnd);
						mpc_abs(dtmp1, aik, rnd);
						if((mpf_cmp_ui(dtmp, 0UL) != 0) && (mpf_cmp_ui(dtmp1, 0UL) != 0))
						{
							//ajk = ajk - aji * aik;
							mpc_mul(ctmp, aji, aik, rnd);
							mpc_sub(ajk, ajk, ctmp, rnd);
							set_cmpfrsmatrix_ij(mat, j, k, ajk);
							//printf("%ld, %ld, %ld\n", i, j, k);
						}
					}
				}
			}
		}
	}

	mpc_clear(aii);
	mpc_clear(aji);
	mpc_clear(ajk);
	mpc_clear(aik);
	mpc_clear(ctmp);
	mpf_clear(dtmp);
	mpf_clear(dtmp1);
}

// iLU0_solve: iLU * x = b
void solve_iLU0_cmpfrsmatrix(CMPFVector ret, CMPFRSMatrix ilu, CMPFVector b)
{
	unsigned long prec;
	long int i, j, row_dim, col_dim;
	mpc_t ret_i, ret_j, ilu_ii, ilu_ij, ctmp;
	mpc_rnd_t rnd = get_bnc_default_rounding_mode_c();
	mpf_t dtmp;

	prec = ret->prec;
	row_dim = ilu->row_dim;
	col_dim = ilu->col_dim;

	mpc_init2(ret_i, prec);
	mpc_init2(ret_j, prec);
	mpc_init2(ilu_ii, prec);
	mpc_init2(ilu_ij, prec);
	mpc_init2(ctmp, prec);
	mpf_init2(dtmp, prec);

	// ret := b
	subst_cmpfvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		mpc_set(ilu_ii, get_cmpfrsmatrix_ij(ilu, i, i), rnd);
		mpc_set(ret_i, get_cmpfvector_i(ret, i), rnd);
		mpc_abs(dtmp, ilu_ii, rnd);
		if(mpf_cmp_ui(dtmp, 0UL) != 0)
		{
			for(j = 0; j < i; j++)
			{
				mpc_set(ilu_ij, get_cmpfrsmatrix_ij(ilu, i, j), rnd);
				mpc_abs(dtmp, ilu_ij, rnd);
				if(mpf_cmp_ui(dtmp, 0UL) != 0)
				{
					mpc_set(ret_j, get_cmpfvector_i(ret, j), rnd);
					//ret_j = ret_j - ilu_ji * ret_i;
					mpc_mul(ctmp, ilu_ij, ret_j, rnd);
					mpc_sub(ret_i, ret_i, ctmp, rnd);
					//set_cddvector_i(ret, j, &ret_j);
					//printf("f: %ld, %ld\n", i, j);
				}
			}
			set_cmpfvector_i(ret, i, ret_i);
		}
	}

	// Backward substitution
	for(i = (row_dim - 1); i >= 0; i--)
	{
		mpc_set(ilu_ii, get_cmpfrsmatrix_ij(ilu, i, i), rnd);
		mpc_set(ret_i, get_cmpfvector_i(ret, i), rnd);
		mpc_abs(dtmp, ret_i, MPC_RNDNN);
		if(mpf_cmp_ui(dtmp, 0UL) != 0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				mpc_set(ilu_ij, get_cmpfrsmatrix_ij(ilu, i, j), rnd);
				mpc_abs(dtmp, ilu_ij, MPC_RNDNN);
				if(mpf_cmp_ui(dtmp, 0UL) != 0)
				{
					mpc_set(ret_j, get_cmpfvector_i(ret, j), rnd);
					//ret_i = ret_i - ilu_ij * ret_j;
					mpc_mul(ctmp, ilu_ij, ret_j, MPC_RNDNN);
					mpc_sub(ret_i, ret_i, ctmp, MPC_RNDNN);
					//set_cddvector_i(ret, i, &ret_i);
					//printf("b: %ld, %ld\n", i, j);
				}
			}
			//ret_i /= ilu_ii;
			mpc_div(ret_i, ret_i, ilu_ii, MPC_RNDNN);
			set_cmpfvector_i(ret, i, ret_i);
		}
	}

	mpc_clear(ret_i);
	mpc_clear(ret_j);
	mpc_clear(ilu_ii);
	mpc_clear(ilu_ij);
	mpc_clear(ctmp);
	mpf_clear(dtmp);
}

// iLU0t_solve: x^t * (iLU) = b^t
void solve_iLU0t_cmpfrsmatrix(CMPFVector ret, CMPFRSMatrix ilu, CMPFVector b)
{
	unsigned long prec;
	long int i, j, row_dim, col_dim;
	mpc_t ret_i, ret_j, ilu_ii, ilu_ji, ctmp;
	mpc_rnd_t rnd = get_bnc_default_rounding_mode_c();
	mpf_t dtmp, dtmp1;
	//double dtmp[DDSIZE];

	row_dim = ilu->row_dim;
	col_dim = ilu->col_dim;
	prec = ilu->prec;

	mpc_init2(ret_i, prec);
	mpc_init2(ret_j, prec);
	mpc_init2(ilu_ii, prec);
	mpc_init2(ilu_ji, prec);
	mpc_init2(ctmp, prec);
	mpf_init2(dtmp, prec);

	// ret := b
	subst_cmpfvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		mpc_set(ilu_ii, get_cmpfrsmatrix_ij(ilu, i, i), rnd);
		mpc_set(ret_i, get_cmpfvector_i(ret, i), rnd);
		mpc_abs(dtmp, ilu_ii, rnd);
		if(mpf_cmp_ui(dtmp, 0UL) != 0)
		{
			for(j = 0; j < i; j++)
			{
				mpc_set(ilu_ji, get_cmpfrsmatrix_ij(ilu, j, i), rnd);
				mpc_abs(dtmp, ilu_ji, rnd);
				if(mpf_cmp_ui(dtmp, 0UL) != 0)
				{
					mpc_set(ret_j, get_cmpfvector_i(ret, j), rnd);;
					//ret_j = ret_j - ilu_ji * ret_i;
					mpc_mul(ctmp, ilu_ji, ret_j, rnd);
					mpc_sub(ret_i, ret_i, ctmp, rnd);
					//printf("f: %ld, %ld\n", i, j);
				}
			}
			mpc_div(ret_i, ret_i, ilu_ii, rnd);
			set_cmpfvector_i(ret, i, ret_i);
		}
	}

	// Backward substitution
	for(i = (row_dim - 1); i >= 0; i--)
	{
		mpc_set(ilu_ii, get_cmpfrsmatrix_ij(ilu, i, i), rnd);
		mpc_set(ret_i, get_cmpfvector_i(ret, i), rnd);
		mpc_abs(dtmp, ilu_ii, rnd);
		if(mpf_cmp_ui(dtmp, 0UL) != 0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				mpc_set(ilu_ji, get_cmpfrsmatrix_ij(ilu, j, i), rnd);
				mpc_abs(dtmp, ilu_ji, rnd);
				if(mpf_cmp_ui(dtmp, 0UL) != 0)
				{
					mpc_set(ret_j, get_cmpfvector_i(ret, j), rnd);
					//ret_i = ret_i - ilu_ij * ret_j;
					mpc_mul(ctmp, ilu_ji, ret_j, rnd);
					mpc_sub(ret_i, ret_i, ctmp, rnd);
					//set_cddvector_i(ret, i, &ret_i);
					//printf("b: %ld, %ld\n", i, j);
				}
			}
			set_cmpfvector_i(ret, i, ret_i);
		}
	}

	mpc_clear(ret_i);
	mpc_clear(ret_j);
	mpc_clear(ilu_ii);
	mpc_clear(ilu_ji);
	mpc_clear(ctmp);
	mpf_clear(dtmp);
}

// iLU0s_solve: x^t * conj(iLU) = b^t
void solve_iLU0s_cmpfrsmatrix(CMPFVector ret, CMPFRSMatrix ilu, CMPFVector b)
{
	unsigned long prec;
	long int i, j, row_dim, col_dim;
	mpc_t ret_i, ret_j, ilu_ii, ilu_ji, conj_ilu_ii, conj_ilu_ji, ctmp;
	mpc_rnd_t rnd = get_bnc_default_rounding_mode_c();
	mpf_t dtmp, dtmp1;
	//double dtmp[DDSIZE];

	row_dim = ilu->row_dim;
	col_dim = ilu->col_dim;
	prec = ilu->prec;

	mpc_init2(ret_i, prec);
	mpc_init2(ret_j, prec);
	mpc_init2(ilu_ii, prec);
	mpc_init2(ilu_ji, prec);
	mpc_init2(conj_ilu_ii, prec);
	mpc_init2(conj_ilu_ji, prec);
	mpc_init2(ctmp, prec);
	mpf_init2(dtmp, prec);

	// ret := b
	subst_cmpfvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		mpc_set(ilu_ii, get_cmpfrsmatrix_ij(ilu, i, i), rnd);
		mpc_set(ret_i, get_cmpfvector_i(ret, i), rnd);
		mpc_abs(dtmp, ilu_ii, rnd);
		if(mpf_cmp_ui(dtmp, 0UL) != 0)
		{
			mpc_conj(conj_ilu_ii, ilu_ii, rnd);
			for(j = 0; j < i; j++)
			{
				mpc_set(ilu_ji, get_cmpfrsmatrix_ij(ilu, j, i), rnd);
				mpc_abs(dtmp, ilu_ji, rnd);
				if(mpf_cmp_ui(dtmp, 0UL) != 0)
				{
					mpc_conj(conj_ilu_ji, ilu_ji, rnd);
					mpc_set(ret_j, get_cmpfvector_i(ret, j), rnd);;
					//ret_j = ret_j - conj(ilu_ji) * ret_i;
					//mpc_mul(ctmp, ilu_ji, ret_j, rnd);
					mpc_mul(ctmp, conj_ilu_ji, ret_j, rnd);
					mpc_sub(ret_i, ret_i, ctmp, rnd);
					//printf("f: %ld, %ld\n", i, j);
				}
			}
			//mpc_div(ret_i, ret_i, ilu_ii, rnd);
			mpc_div(ret_i, ret_i, conj_ilu_ii, rnd);
			set_cmpfvector_i(ret, i, ret_i);
		}
	}

	// Backward substitution
	for(i = (row_dim - 1); i >= 0; i--)
	{
		mpc_set(ilu_ii, get_cmpfrsmatrix_ij(ilu, i, i), rnd);
		mpc_set(ret_i, get_cmpfvector_i(ret, i), rnd);
		mpc_abs(dtmp, ilu_ii, rnd);
		if(mpf_cmp_ui(dtmp, 0UL) != 0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				mpc_set(ilu_ji, get_cmpfrsmatrix_ij(ilu, j, i), rnd);
				mpc_abs(dtmp, ilu_ji, rnd);
				if(mpf_cmp_ui(dtmp, 0UL) != 0)
				{
					mpc_conj(conj_ilu_ji, ilu_ji, rnd);
					mpc_set(ret_j, get_cmpfvector_i(ret, j), rnd);
					//ret_i = ret_i - conj(ilu_ij) * ret_j;
					mpc_mul(ctmp, conj_ilu_ji, ret_j, rnd);
					mpc_sub(ret_i, ret_i, ctmp, rnd);
					//set_cddvector_i(ret, i, &ret_i);
					//printf("b: %ld, %ld\n", i, j);
				}
			}
			set_cmpfvector_i(ret, i, ret_i);
		}
	}

	mpc_clear(ret_i);
	mpc_clear(ret_j);
	mpc_clear(ilu_ii);
	mpc_clear(ilu_ji);
	mpc_clear(conj_ilu_ii);
	mpc_clear(conj_ilu_ji);
	mpc_clear(ctmp);
	mpf_clear(dtmp);
}

// iLU0_solve: iLU * x = b
void solve_iLU0_cdrsmatrix_cmpfvec(CMPFVector ret, CDRSMatrix ilu, CMPFVector b)
{
	unsigned long prec;
	long int i, j, row_dim, col_dim;
	mpc_t ret_i, ret_j, ctmp;
	double _Complex ilu_ii, ilu_ij;
	mpc_rnd_t rnd = get_bnc_default_rounding_mode_c();
	mpf_t dtmp, dtmp1;
	//double dtmp[DDSIZE];

	row_dim = ilu->re->row_dim;
	col_dim = ilu->re->col_dim;
	prec = ret->prec;

	mpc_init2(ret_i, prec);
	mpc_init2(ret_j, prec);
	mpc_init2(ctmp, prec);
	mpf_init2(dtmp, prec);
	mpf_init2(dtmp1, prec);

	// ret := b
	subst_cmpfvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		mpc_set(ret_i, get_cmpfvector_i(ret, i), rnd);
		if(cabs(ilu_ii) != 0.0)
		{
			for(j = 0; j < i; j++)
			{
				ilu_ij = get_cdrsmatrix_ij(ilu, i, j);
				if(cabs(ilu_ij) != 0.0)
				{
					mpc_set(ret_j, get_cmpfvector_i(ret, j), rnd);
					//ret_j = ret_j - ilu_ji * ret_i;
					_bnc_mpc_mul_cd(ctmp, ret_j, ilu_ij);
					mpc_sub(ret_i, ret_i, ctmp, rnd);
					//printf("f: %ld, %ld\n", i, j);
				}
			}
			set_cmpfvector_i(ret, i, ret_i);
		}
	}

	// Backward substitution
	for(i = (row_dim - 1); i >= 0; i--)
	{
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		mpc_set(ret_i, get_cmpfvector_i(ret, i), rnd);
		if(cabs(ilu_ii) != 0.0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				ilu_ij = get_cdrsmatrix_ij(ilu, i, j);
				if(cabs(ilu_ij) != 0.0)
				{
					mpc_set(ret_j, get_cmpfvector_i(ret, j), rnd);
					//ret_i = ret_i - ilu_ij * ret_j;
					_bnc_mpc_mul_cd(ctmp, ret_j, ilu_ij);
					mpc_sub(ret_i, ret_i, ctmp, rnd);
					//set_cddvector_i(ret, i, &ret_i);
					//printf("b: %ld, %ld\n", i, j);
				}
			}
			//ret_i /= ilu_ii;
			_bnc_mpc_div_cd(ret_i, ret_i, ilu_ii);
			set_cmpfvector_i(ret, i, ret_i);
		}
	}

	mpc_clear(ret_i);
	mpc_clear(ret_j);
	mpc_clear(ctmp);
	mpf_clear(dtmp);
	mpf_clear(dtmp1);
}

// iLU0t_solve: x^t(iLU) = b^t
void solve_iLU0t_cdrsmatrix_cmpfvec(CMPFVector ret, CDRSMatrix ilu, CMPFVector b)
{
	unsigned long prec;
	long int i, j, row_dim, col_dim;
	mpc_t ret_i, ret_j, ctmp;
	double _Complex ilu_ii, ilu_ji;
	mpc_rnd_t rnd = get_bnc_default_rounding_mode_c();
	mpf_t dtmp, dtmp1;
	//double dtmp[DDSIZE];

	row_dim = ilu->re->row_dim;
	col_dim = ilu->re->col_dim;
	prec = ret->prec;

	mpc_init2(ret_i, prec);
	mpc_init2(ret_j, prec);
	mpc_init2(ctmp, prec);
	mpf_init2(dtmp, prec);

	// ret := b
	subst_cmpfvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		mpc_set(ret_i, get_cmpfvector_i(ret, i), rnd);
		if(cabs(ilu_ii) != 0.0)
		{
			for(j = 0; j < i; j++)
			{
				ilu_ji = get_cdrsmatrix_ij(ilu, j, i);
				if(cabs(ilu_ji) != 0.0)
				{
					mpc_set(ret_j, get_cmpfvector_i(ret, j), rnd);
					//ret_j = ret_j - ilu_ji * ret_i;
					_bnc_mpc_mul_cd(ctmp, ret_j, ilu_ji);
					mpc_sub(ret_i, ret_i, ctmp, rnd);
					//printf("f: %ld, %ld\n", i, j);
				}
			}
			_bnc_mpc_div_cd(ret_i, ret_i, ilu_ii);
			set_cmpfvector_i(ret, i, ret_i);
		}
	}

	// Backward substitution
	for(i = (row_dim - 1); i >= 0; i--)
	{
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		mpc_set(ret_i, get_cmpfvector_i(ret, i), rnd);
		if(cabs(ilu_ii) != 0.0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				ilu_ji = get_cdrsmatrix_ij(ilu, j, i);
				if(cabs(ilu_ji) != 0.0)
				{
					mpc_set(ret_j, get_cmpfvector_i(ret, j), rnd);
					//ret_i = ret_i - ilu_ij * ret_j;
					_bnc_mpc_mul_cd(ctmp, ret_j, ilu_ji);
					mpc_sub(ret_i, ret_i, ctmp, rnd);
					//set_cddvector_i(ret, i, &ret_i);
					//printf("b: %ld, %ld\n", i, j);
				}
			}
			set_cmpfvector_i(ret, i, ret_i);
		}
	}

	mpc_clear(ret_i);
	mpc_clear(ret_j);
	mpc_clear(ctmp);
	mpf_clear(dtmp);
}

// iLU0s_solve: x^t * conj(iLU) = b^t
void solve_iLU0s_cdrsmatrix_cmpfvec(CMPFVector ret, CDRSMatrix ilu, CMPFVector b)
{
	unsigned long prec;
	long int i, j, row_dim, col_dim;
	mpc_t ret_i, ret_j, ctmp;
	double _Complex ilu_ii, ilu_ji, conj_ilu_ii, conj_ilu_ji;
	mpc_rnd_t rnd = get_bnc_default_rounding_mode_c();
	mpf_t dtmp, dtmp1;
	//double dtmp[DDSIZE];

	row_dim = ilu->re->row_dim;
	col_dim = ilu->re->col_dim;
	prec = ret->prec;

	mpc_init2(ret_i, prec);
	mpc_init2(ret_j, prec);
	mpc_init2(ctmp, prec);
	mpf_init2(dtmp, prec);

	// ret := b
	subst_cmpfvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		mpc_set(ret_i, get_cmpfvector_i(ret, i), rnd);
		if(cabs(ilu_ii) != 0.0)
		{
			conj_ilu_ii = conj(ilu_ii);
			for(j = 0; j < i; j++)
			{
				ilu_ji = get_cdrsmatrix_ij(ilu, j, i);
				if(cabs(ilu_ji) != 0.0)
				{
					conj_ilu_ji = conj(ilu_ji);
					mpc_set(ret_j, get_cmpfvector_i(ret, j), rnd);
					//ret_j = ret_j - conj(ilu_ji) * ret_i;
					//_bnc_mpc_mul_cd(ctmp, ret_j, ilu_ji);
					_bnc_mpc_mul_cd(ctmp, ret_j, conj_ilu_ji);
					mpc_sub(ret_i, ret_i, ctmp, rnd);
					//printf("f: %ld, %ld\n", i, j);
				}
			}
			//_bnc_mpc_div_cd(ret_i, ret_i, ilu_ii);
			_bnc_mpc_div_cd(ret_i, ret_i, conj_ilu_ii);
			set_cmpfvector_i(ret, i, ret_i);
		}
	}

	// Backward substitution
	for(i = (row_dim - 1); i >= 0; i--)
	{
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		mpc_set(ret_i, get_cmpfvector_i(ret, i), rnd);
		if(cabs(ilu_ii) != 0.0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				ilu_ji = get_cdrsmatrix_ij(ilu, j, i);
				if(cabs(ilu_ji) != 0.0)
				{
					conj_ilu_ji = conj(ilu_ji);
					mpc_set(ret_j, get_cmpfvector_i(ret, j), rnd);
					//ret_i = ret_i - conj(ilu_ij) * ret_j;
					//_bnc_mpc_mul_cd(ctmp, ret_j, ilu_ji);
					_bnc_mpc_mul_cd(ctmp, ret_j, conj_ilu_ji);
					mpc_sub(ret_i, ret_i, ctmp, rnd);
					//set_cddvector_i(ret, i, &ret_i);
					//printf("b: %ld, %ld\n", i, j);
				}
			}
			set_cmpfvector_i(ret, i, ret_i);
		}
	}

	mpc_clear(ret_i);
	mpc_clear(ret_j);
	mpc_clear(ctmp);
	mpf_clear(dtmp);
}


#endif // USE_GMP
