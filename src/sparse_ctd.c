/********************************************************************************/
/*                                                                              */
/* sparse_ctd.c : Complex Sparse Matrix and Vector Library (TD Precision)       */
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
#ifdef WIN32
#include <windows.h>
#else // WIN32
#include <sys/times.h>
#include <time.h>
#endif // WIN32

#include "bncsparse.h"

#ifdef USE_TDLINEAR

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

/* initialize CDRSMatrix */
CTDRSMatrix init_ctdrsmatrix(long int row_dim, long int *nzero_col_dim, long int nzero_total_num)
{
	CTDRSMatrix ret;
	long int i, j;

	ret = (CTDRSMatrix)malloc(sizeof(ctdrsmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "Cannot allocate CTDRSMatrix\n");
		return ret;
	}
	
	if((row_dim < 0) || (nzero_total_num < 0))
	{
		fprintf(stderr, "Illigal nzero values\n");
		return NULL;
	}

    // re, im = init_drsmatrix
    ret->re = init_tdrsmatrix(row_dim, nzero_col_dim, nzero_total_num);
    ret->im = init_tdrsmatrix(row_dim, nzero_col_dim, nzero_total_num);

	return ret;
}

/* Clear CTDRSMatrix */
void free_ctdrsmatrix(CTDRSMatrix mat)
{

    free_tdrsmatrix(mat->re);
    free_tdrsmatrix(mat->im);

	free(mat);
}

/* Print CTDRSMatrix */
void print_ctdrsmatrix(CTDRSMatrix mat)
{
	long int i, j, total_index;
	tdfloat tmp;

	if(mat == NULL)
		fprintf(stderr, "ERROR!\n");

	total_index = 0;
	for(i = 0; i < mat->re->row_dim; i++)
	{		// Fix!: 2011-08-29 by T.Kouya
		if(mat->re->nzero_col_dim[i] >= 1)
		{
            printf("%5ld: ", i);
            for(j = 0; j < mat->re->real_nzero_col_dim[i]; j++)
            {
                // printf("%ld->%f, ", mat->nzero_index[i][j], mat->element[total_index++]);
                if(j < mat->re->nzero_col_dim[i])
                {
                    printf("%ld-> ", mat->re->nzero_index[i][j]);
                    //mpf_out_str(stdout, 10, 0, mat->element[total_index++]);
                    tmp.val[0] = mat->re->element[0][total_index + j];
                    tmp.val[1] = mat->re->element[1][total_index + j];
                    tmp.val[2] = mat->re->element[2][total_index + j];
                    rtd_out_str(tmp.val);
                    printf(" + ");
                    tmp.val[0] = mat->im->element[0][total_index + j];
                    tmp.val[1] = mat->im->element[1][total_index + j];
                    tmp.val[2] = mat->im->element[2][total_index + j];
                    rtd_out_str(tmp.val);
                    printf(" * I");
                    if(j != (mat->re->nzero_col_dim[i] - 1)) printf(", ");
                }
                total_index++;
            }
            printf("\n");
        }
		total_index += mat->re->real_nzero_col_dim[i];
	}
	return;
}

/* Dense Matrix := Sparse Matrix */
void set_ctdmatrix_ctdrsmatrix(CTDMatrix ret, CTDRSMatrix spmat)
{
	long int i, j, total_index;
    double dtmp[TDSIZE];

	if((ret == NULL) || (spmat == NULL))
	{
		fprintf(stderr, "set_ctdmatrix_ctdrsmatrix: ERROR!\n");
		return;
	}

	// ret := 0
	set0_ctdmatrix(ret);

	total_index = 0;
	for(i = 0; i < spmat->re->row_dim; i++)
	{
		for(j = 0; j < spmat->re->nzero_col_dim[i]; j++)
		{
			//printf("%5ld: ", i);
			//printf("%ld->%e, ", spmat->nzero_index[i][j], spmat->element[total_index++]);
			//ctmp = spmat->re->element[total_index] + spmat->im->element[total_index] * I;
            dtmp[0] = spmat->re->element[0][total_index];
            dtmp[1] = spmat->re->element[1][total_index];
            dtmp[2] = spmat->re->element[2][total_index];
			set_tdmatrix_ij(ret->re, i, spmat->re->nzero_index[i][j], dtmp);
            dtmp[0] = spmat->im->element[0][total_index];
            dtmp[1] = spmat->im->element[1][total_index];
            dtmp[2] = spmat->im->element[2][total_index];
            set_tdmatrix_ij(ret->im, i, spmat->im->nzero_index[i][j], dtmp);
            total_index++;
		}
	}
	return;
}

/* initialize and substitute DRSMatrix from DMatrix */
CTDRSMatrix init_set_ctdrsmatrix_ctdmatrix(CTDMatrix org_mat)
{
	long int i, j;
	long int nzero_total_num, total_index, j_index;
	long int *ptr_nzero_col_dim;
	ctdfloat ctmp, *ptr_ctmp;
    tdfloat dtmp;
	CTDRSMatrix ret;

	/* initialize variables */
	nzero_total_num = 0;

	ptr_nzero_col_dim = (long int *)malloc((size_t)(sizeof(long int) * (org_mat->re->row_dim)));

	/* initialize num_col as numbers of nonzero element in each row */
	if(ptr_nzero_col_dim == NULL)
	{
		fprintf(stderr, "Cannot allocate num_col(size %ld)\n", sizeof(long int) * (org_mat->re->row_dim));
		return NULL;
	}

	/* Read org_mat (1st) */
	for(i = 0; i < org_mat->re->row_dim; i++)
	{
		*(ptr_nzero_col_dim + i) = 0;
		for(j = 0; j < org_mat->re->col_dim; j++)
		{
            ptr_ctmp = get_ctdmatrix_ij(org_mat, i, j);
			//if(cabs(get_cdmatrix_ij(org_mat, i, j)) != 0.0)
            rctd_abs(&dtmp, ptr_ctmp);
            if(rtd_cmp_ui(dtmp.val, 0UL) != 0)
			{
				/* get nzero_row_dim, nzero_col_dim, nzero_total_num */
				nzero_total_num++;
				*(ptr_nzero_col_dim + i) += 1;
			}
		}
	}

	// initialize
	ret = init_ctdrsmatrix(org_mat->re->row_dim, ptr_nzero_col_dim, nzero_total_num);

	/* Read org_mat (2nd) */
	total_index = 0;
	for(i = 0; i < ret->re->row_dim; i++)
	{
		j_index = 0;
		for(j = 0; j < ret->re->col_dim; j++)
		{
            ptr_ctmp = get_ctdmatrix_ij(org_mat, i, j);
            rctd_abs(&dtmp, ptr_ctmp);
			//if(cabs(get_cdmatrix_ij(org_mat, i, j)) != 0.0)
            if(rtd_cmp_ui(dtmp.val, 0UL) != 0)
			{
				ret->re->nzero_index[i][j_index] =  j;
				//get_ctdmatrix_ij(&ctmp, org_mat, i, j);
                ptr_ctmp = get_ctdmatrix_ij(org_mat, i, j);
				//rtd_set(&(ret->re->element[total_index]), ptr_ctmp->val_re); //get_dmatrix_ij(org_mat, i, j);
				//rtd_set(&(ret->im->element[total_index]), ptr_ctmp->val_im); //get_dmatrix_ij(org_mat, i, j);
				ret->re->element[0][total_index] = ptr_ctmp->val_re[0];
				ret->re->element[1][total_index] = ptr_ctmp->val_re[1];
				ret->re->element[2][total_index] = ptr_ctmp->val_re[2];

				ret->im->element[0][total_index] = ptr_ctmp->val_im[0];
				ret->im->element[1][total_index] = ptr_ctmp->val_im[1];
				ret->im->element[2][total_index] = ptr_ctmp->val_im[2];

				total_index++;
				j_index++;
			}
		}
	}

	free(ptr_nzero_col_dim);

	return ret;
}

// 2024-09-04(Wed) T.Kouya
/* get the CTDRSMatrix ij-element */
void get_ctdrsmatrix_ij(ctdfloat *ret, CTDRSMatrix mat, long int row_index, long int col_index)
{
    get_tdrsmatrix_ij(ret->val_re, mat->re, row_index, col_index);
    get_tdrsmatrix_ij(ret->val_im, mat->im, row_index, col_index);
}

// 2024-09-04 (Wed) T.Kouya
/* set the CTDRSMatrix ij-element */
void set_ctdrsmatrix_ij(CTDRSMatrix mat, long int row_index, long int col_index, ctdfloat *val)
{
    set_tdrsmatrix_ij(mat->re, row_index, col_index, val->val_re);
    set_tdrsmatrix_ij(mat->im, row_index, col_index, val->val_im);
}

// 2024-12-03 (Tue) T.Kouya
CTDRSMatrix init_set_ctdrsmatrix(CTDRSMatrix org_mat)
{
	CTDRSMatrix ret;

	ret = init_ctdrsmatrix(org_mat->re->row_dim, org_mat->re->nzero_col_dim, org_mat->re->nzero_total_num);
	subst_tdrsmatrix(ret->re, org_mat->re);
	subst_tdrsmatrix(ret->im, org_mat->im);

	return ret;
}

#ifdef USE_GMP
// CMPFRSMatrix -> CTDRSMatrix
CTDRSMatrix init_set_ctdrsmatrix_cmpfrsmatrix(CMPFRSMatrix org_sp)
{
	CTDRSMatrix ret;
    ctdfloat ctdtmp;
	long int org_sp_total_index, total_index, i, j;

    //printf("row_dim, nzero_total_num = %ld, %ld\n", org_sp->row_dim, org_sp->nzero_total_num);
    ret = init_ctdrsmatrix(org_sp->row_dim, org_sp->nzero_col_dim, org_sp->nzero_total_num);
    //printf("init_tdrsmatrix!\n");

    // MPFR -> TD
    total_index = 0;
	org_sp_total_index = 0;
	for(i = 0; i < ret->re->row_dim; i++)
	{
		//printf("%ld: ", i);
		for(j = 0; j < ret->re->nzero_col_dim[i]; j++)
		{
            //mpf_get_td(tdtmp, org_sp->element[org_sp_total_index + j]);
            mpf_get_td(ctdtmp.val_re, mpc_realref(org_sp->element[org_sp_total_index + j]));
            mpf_get_td(ctdtmp.val_im, mpc_imagref(org_sp->element[org_sp_total_index + j]));

            //ret->element[0][total_index + j] = tdtmp[0];
            //ret->element[1][total_index + j] = tdtmp[1];
            ret->re->element[0][total_index + j] = ctdtmp.val_re[0];
            ret->re->element[1][total_index + j] = ctdtmp.val_re[1];
            ret->re->element[2][total_index + j] = ctdtmp.val_re[2];
            ret->im->element[0][total_index + j] = ctdtmp.val_im[0];
            ret->im->element[1][total_index + j] = ctdtmp.val_im[1];
            ret->im->element[2][total_index + j] = ctdtmp.val_im[2];

            ret->re->nzero_index[i][j] = org_sp->nzero_index[i][j];
            ret->im->nzero_index[i][j] = org_sp->nzero_index[i][j];
			//total_index++;
			//printf("%ld ", j);
		}
		//printf("\n");
		total_index += ret->re->real_nzero_col_dim[i];
		org_sp_total_index += org_sp->nzero_col_dim[i];
	}
    //print_tdrsmatrix(ret);

	return ret;
}
#endif // USE_GMP

/* Get variables to initialize DRSMatrix */
int get_vars_ctdrsmatrix_fname(long int *ptr_row_dim, long int **ptr_nzero_col_dim, long int *ptr_nzero_total_num, const char *fname)
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

/* Multiply CTDRSMatrix * CTDVector */
int mul_ctdrsmatrix_ctdvec(CTDVector ret, CTDRSMatrix mat, CTDVector vec)
{
	long int i, j, total_index;
	TDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_tdvector(mat->re->row_dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	mul_tdrsmatrix_tdvec(tmp_vec[0], mat->re, vec->re);
	mul_tdrsmatrix_tdvec(tmp_vec[1], mat->re, vec->im);
	mul_tdrsmatrix_tdvec(tmp_vec[2], mat->im, vec->re);
	mul_tdrsmatrix_tdvec(tmp_vec[3], mat->im, vec->im);

	sub_tdvector(ret->re, tmp_vec[0], tmp_vec[3]);
	add_tdvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_tdvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply CTDRSMatrix^T * CTDVector */
int mul_ctdrsmatrixt_ctdvec(CTDVector ret, CTDRSMatrix mat, CTDVector vec)
{
	long int i, j, total_index;
	TDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_tdvector(vec->re->dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	mul_tdrsmatrixt_tdvec(tmp_vec[0], mat->re, vec->re);
	mul_tdrsmatrixt_tdvec(tmp_vec[1], mat->re, vec->im);
	mul_tdrsmatrixt_tdvec(tmp_vec[2], mat->im, vec->re);
	mul_tdrsmatrixt_tdvec(tmp_vec[3], mat->im, vec->im);

	sub_tdvector(ret->re, tmp_vec[0], tmp_vec[3]);
	add_tdvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_tdvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply conj(CTDRSMatrix)^T * CTDVector */
int mul_ctdrsmatrixs_ctdvec(CTDVector ret, CTDRSMatrix mat, CTDVector vec)
{
	long int i, j, total_index;
	TDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_tdvector(vec->re->dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	mul_tdrsmatrixt_tdvec(tmp_vec[0], mat->re, vec->re);
	mul_tdrsmatrixt_tdvec(tmp_vec[1], mat->re, vec->im);
	mul_tdrsmatrixt_tdvec(tmp_vec[2], mat->im, vec->re);
	mul_tdrsmatrixt_tdvec(tmp_vec[3], mat->im, vec->im);

	// mat_re * vec_re + mat_im * vec_im
	add_tdvector(ret->re, tmp_vec[0], tmp_vec[3]);
	// mat_re * vec_im - mat_im * vec_re
	sub_tdvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_tdvector(tmp_vec[i]);

	return SUCCESS;
}


/* Multiply CDRSMatrix * CTDVector */
int mul_cdrsmatrix_ctdvec(CTDVector ret, CDRSMatrix mat, CTDVector vec)
{
	long int i, j, total_index;
	TDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_tdvector(mat->re->row_dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	mul_drsmatrix_tdvec(tmp_vec[0], mat->re, vec->re);
	mul_drsmatrix_tdvec(tmp_vec[1], mat->re, vec->im);
	mul_drsmatrix_tdvec(tmp_vec[2], mat->im, vec->re);
	mul_drsmatrix_tdvec(tmp_vec[3], mat->im, vec->im);

	sub_tdvector(ret->re, tmp_vec[0], tmp_vec[3]);
	add_tdvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_tdvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply CDRSMatrix^T * CTDVector */
int mul_cdrsmatrixt_ctdvec(CTDVector ret, CDRSMatrix mat, CTDVector vec)
{
	long int i, j, total_index;
	TDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_tdvector(vec->re->dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	mul_drsmatrixt_tdvec(tmp_vec[0], mat->re, vec->re);
	mul_drsmatrixt_tdvec(tmp_vec[1], mat->re, vec->im);
	mul_drsmatrixt_tdvec(tmp_vec[2], mat->im, vec->re);
	mul_drsmatrixt_tdvec(tmp_vec[3], mat->im, vec->im);

	sub_tdvector(ret->re, tmp_vec[0], tmp_vec[3]);
	add_tdvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_tdvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply conj(CDRSMatrix)^T * CTDVector */
int mul_cdrsmatrixs_ctdvec(CTDVector ret, CDRSMatrix mat, CTDVector vec)
{
	long int i, j, total_index;
	TDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_tdvector(vec->re->dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	mul_drsmatrixt_tdvec(tmp_vec[0], mat->re, vec->re);
	mul_drsmatrixt_tdvec(tmp_vec[1], mat->re, vec->im);
	mul_drsmatrixt_tdvec(tmp_vec[2], mat->im, vec->re);
	mul_drsmatrixt_tdvec(tmp_vec[3], mat->im, vec->im);

	// mat_re * vec_re + mat_im * vec_im
	add_tdvector(ret->re, tmp_vec[0], tmp_vec[3]);
	// mat_re * vec_im - mat_im * vec_re
	sub_tdvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_tdvector(tmp_vec[i]);

	return SUCCESS;
}

// Imcomplete LU decomposition; iLU0_drsmatrix
void iLU0_ctdrsmatrix(CTDRSMatrix mat)
{
	long int i, j, k, row_dim, col_dim;
	ctdfloat aii, aji, ajk, aik, ctmp;
	//double dtmp[DDSIZE];

	row_dim = mat->re->row_dim;
	col_dim = mat->re->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		get_ctdrsmatrix_ij(&aii, mat, i, i);
		if(rctd_cmp_abs_ui(&aii, 0UL) != 0)
		{
			for(j = i + 1; j < row_dim; j++)
			{
				get_ctdrsmatrix_ij(&aji, mat, j, i);
				if(rctd_cmp_abs_ui(&aji, 0UL) != 0)
				{
					//aji /= aii;
					rctd_div(&aji, &aji, &aii);
					set_ctdrsmatrix_ij(mat, j, i, &aji);
					for(k = i + 1; k < col_dim; k++)
					{
						get_ctdrsmatrix_ij(&ajk, mat, j, k);
						get_ctdrsmatrix_ij(&aik, mat, i, k);
						if((rctd_cmp_abs_ui(&ajk, 0UL) != 0) && (rctd_cmp_abs_ui(&aik, 0UL) != 0))
						{
							//ajk = ajk - aji * aik;
							rctd_mul(&ctmp, &aji, &aik);
							rctd_sub(&ajk, &ajk, &ctmp);
							set_ctdrsmatrix_ij(mat, j, k, &ajk);
							//printf("%ld, %ld, %ld\n", i, j, k);
						}
					}
				}
			}
		}
	}
}

// iLU0_solve: iLU * x = b
void solve_iLU0_ctdrsmatrix(CTDVector ret, CTDRSMatrix ilu, CTDVector b)
{
	long int i, j, row_dim, col_dim;
	ctdfloat ret_i, ret_j, ilu_ii, ilu_ij, ctmp;

	row_dim = ilu->re->row_dim;
	col_dim = ilu->re->col_dim;

	// ret := b
	subst_ctdvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		get_ctdrsmatrix_ij(&ilu_ii, ilu, i, i);
		subst_ctdvector_i(&ret_i, ret, i);
		if(rctd_cmp_abs_ui(&ilu_ii, 0UL) != 0)
		{
			for(j = 0; j < i; j++)
			{
				get_ctdrsmatrix_ij(&ilu_ij, ilu, i, j);
				if(rctd_cmp_abs_ui(&ilu_ij, 0UL) != 0)
				{
					subst_ctdvector_i(&ret_j, ret, j);
					//ret_j = ret_j - ilu_ji * ret_i;
					rctd_mul(&ctmp, &ilu_ij, &ret_j);
					rctd_sub(&ret_i, &ret_i, &ctmp);
					//set_ctdvector_i(ret, j, &ret_j);
					//printf("f: %ld, %ld\n", i, j);
				}
			}
			set_ctdvector_i(ret, i, &ret_i);
		}
	}

	// Backward substitution
	for(i = (row_dim - 1); i >= 0; i--)
	{
		get_ctdrsmatrix_ij(&ilu_ii, ilu, i, i);
		subst_ctdvector_i(&ret_i, ret, i);
		if(rctd_cmp_abs_ui(&ilu_ii, 0UL) != 0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				get_ctdrsmatrix_ij(&ilu_ij, ilu, i, j);
				if(rctd_cmp_abs_ui(&ilu_ij, 0UL) != 0)
				{
					subst_ctdvector_i(&ret_j, ret, j);
					//ret_i = ret_i - ilu_ij * ret_j;
					rctd_mul(&ctmp, &ilu_ij, &ret_j);
					rctd_sub(&ret_i, &ret_i, &ctmp);
					//set_ctdvector_i(ret, i, &ret_i);
					//printf("b: %ld, %ld\n", i, j);
				}
			}
			//ret_i /= ilu_ii;
			rctd_div(&ret_i, &ret_i, &ilu_ii);
			set_ctdvector_i(ret, i, &ret_i);
		}
	}
}

// iLU0t_solve: x^t * iLU = b^t
void solve_iLU0t_ctdrsmatrix(CTDVector ret, CTDRSMatrix ilu, CTDVector b)
{
	long int i, j, row_dim, col_dim;
	ctdfloat ret_i, ret_j, ilu_ii, ilu_ji, ctmp;

	row_dim = ilu->re->row_dim;
	col_dim = ilu->re->col_dim;

	// ret := b
	subst_ctdvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		get_ctdrsmatrix_ij(&ilu_ii, ilu, i, i);
		subst_ctdvector_i(&ret_i, ret, i);
		if(rctd_cmp_abs_ui(&ilu_ii, 0UL) != 0)
		{
			for(j = 0; j < i; j++)
			{
				get_ctdrsmatrix_ij(&ilu_ji, ilu, j, i);
				if(rctd_cmp_abs_ui(&ilu_ji, 0UL) != 0)
				{
					subst_ctdvector_i(&ret_j, ret, j);
					//ret_j = ret_j - ilu_ji * ret_i;
					rctd_mul(&ctmp, &ilu_ji, &ret_j);
					rctd_sub(&ret_i, &ret_i, &ctmp);
					//printf("f: %ld, %ld\n", i, j);
				}
			}
			//ret_i /= ilu_ii;
			rctd_div(&ret_i, &ret_i, &ilu_ii);
			set_ctdvector_i(ret, i, &ret_i);
		}
	}

	// Backward substitution
	for(i = (row_dim - 1); i >= 0; i--)
	{
		get_ctdrsmatrix_ij(&ilu_ii, ilu, i, i);
		subst_ctdvector_i(&ret_i, ret, i);
		if(rctd_cmp_abs_ui(&ilu_ii, 0UL) != 0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				get_ctdrsmatrix_ij(&ilu_ji, ilu, j, i);
				if(rctd_cmp_abs_ui(&ilu_ji, 0UL) != 0)
				{
					subst_ctdvector_i(&ret_j, ret, j);
					//ret_i = ret_i - ilu_ij * ret_j;
					rctd_mul(&ctmp, &ilu_ji, &ret_j);
					rctd_sub(&ret_i, &ret_i, &ctmp);
					//printf("b: %ld, %ld\n", i, j);
				}
			}
			set_ctdvector_i(ret, i, &ret_i);
		}
	}
}

// iLU0s_solve: x^t * conj(iLU) = b^t
void solve_iLU0s_ctdrsmatrix(CTDVector ret, CTDRSMatrix ilu, CTDVector b)
{
	long int i, j, row_dim, col_dim;
	ctdfloat ret_i, ret_j, ilu_ii, ilu_ji, ctmp, conj_ilu_ii, conj_ilu_ji;

	row_dim = ilu->re->row_dim;
	col_dim = ilu->re->col_dim;

	// ret := b
	subst_ctdvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		get_ctdrsmatrix_ij(&ilu_ii, ilu, i, i);
		subst_ctdvector_i(&ret_i, ret, i);
		if(rctd_cmp_abs_ui(&ilu_ii, 0UL) != 0)
		{
			rctd_conj(&conj_ilu_ii, &ilu_ii);
			for(j = 0; j < i; j++)
			{
				get_ctdrsmatrix_ij(&ilu_ji, ilu, j, i);
				if(rctd_cmp_abs_ui(&ilu_ji, 0UL) != 0)
				{
					rctd_conj(&conj_ilu_ji, &ilu_ji);
					subst_ctdvector_i(&ret_j, ret, j);
					//ret_j = ret_j - ilu_ji * ret_i;
					//rctd_mul(&ctmp, &ilu_ji, &ret_j);
					rctd_mul(&ctmp, &conj_ilu_ji, &ret_j);
					rctd_sub(&ret_i, &ret_i, &ctmp);
					//printf("f: %ld, %ld\n", i, j);
				}
			}
			//ret_i /= ilu_ii;
			//rctd_div(&ret_i, &ret_i, &ilu_ii);
			rctd_div(&ret_i, &ret_i, &conj_ilu_ii);
			set_ctdvector_i(ret, i, &ret_i);
		}
	}

	// Backward substitution
	for(i = (row_dim - 1); i >= 0; i--)
	{
		get_ctdrsmatrix_ij(&ilu_ii, ilu, i, i);
		subst_ctdvector_i(&ret_i, ret, i);
		if(rctd_cmp_abs_ui(&ilu_ii, 0UL) != 0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				get_ctdrsmatrix_ij(&ilu_ji, ilu, j, i);
				if(rctd_cmp_abs_ui(&ilu_ji, 0UL) != 0)
				{
					rctd_conj(&conj_ilu_ji, &ilu_ji);
					subst_ctdvector_i(&ret_j, ret, j);
					//ret_i = ret_i - ilu_ij * ret_j;
					//rctd_mul(&ctmp, &ilu_ji, &ret_j);
					rctd_mul(&ctmp, &conj_ilu_ji, &ret_j);
					rctd_sub(&ret_i, &ret_i, &ctmp);
					//printf("b: %ld, %ld\n", i, j);
				}
			}
			set_ctdvector_i(ret, i, &ret_i);
		}
	}
}


// iLU0_solve: iLU * x = b
void solve_iLU0_cdrsmatrix_ctdvec(CTDVector ret, CDRSMatrix ilu, CTDVector b)
{
	long int i, j, row_dim, col_dim;
	double _Complex ilu_ii, ilu_ij;
	ctdfloat ret_i, ret_j, ctmp;

	row_dim = ilu->re->row_dim;
	col_dim = ilu->re->col_dim;

	// ret := b
	subst_ctdvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		subst_ctdvector_i(&ret_i, ret, i);
		if(cabs(ilu_ii) != 0.0)
		{
			for(j = 0; j < i; j++)
			{
				ilu_ij = get_cdrsmatrix_ij(ilu, i, j);
				if(cabs(ilu_ij) != 0.0)
				{
					subst_ctdvector_i(&ret_j, ret, j);
					//ret_j = ret_j - ilu_ji * ret_i;
					rctd_mul_cd(&ctmp, &ret_j, ilu_ij);
					rctd_sub(&ret_i, &ret_i, &ctmp);
					//set_ctdvector_i(ret, j, &ret_j);
					//printf("f: %ld, %ld\n", i, j);
				}
			}
			set_ctdvector_i(ret, i, &ret_i);
		}
	}

	// Backward substitution
	for(i = (row_dim - 1); i >= 0; i--)
	{
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		subst_ctdvector_i(&ret_i, ret, i);
		if(cabs(ilu_ii) != 0.0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				ilu_ij = get_cdrsmatrix_ij(ilu, i, j);
				if(cabs(ilu_ij) != 0.0)
				{
					subst_ctdvector_i(&ret_j, ret, j);
					//ret_i = ret_i - ilu_ij * ret_j;
					//rctd_mul(&ctmp, &ilu_ij, &ret_j);
					rctd_mul_cd(&ctmp, &ret_j, ilu_ij);
					rctd_sub(&ret_i, &ret_i, &ctmp);
					//set_ctdvector_i(ret, i, &ret_i);
					//printf("b: %ld, %ld\n", i, j);
				}
			}
			//ret_i /= ilu_ii;
			//rctd_div(&ret_i, &ret_i, &ilu_ii);
			rctd_div_cd(&ret_i, &ret_i, ilu_ii);
			set_ctdvector_i(ret, i, &ret_i);
		}
	}
}

// iLU0_solve: x^t * iLU = b^t
void solve_iLU0t_cdrsmatrix_ctdvec(CTDVector ret, CDRSMatrix ilu, CTDVector b)
{
	long int i, j, row_dim, col_dim;
	ctdfloat ret_i, ret_j, ctmp;
	double _Complex ilu_ii, ilu_ji;

	row_dim = ilu->re->row_dim;
	col_dim = ilu->re->col_dim;

	// ret := b
	subst_ctdvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		//get_ctdrsmatrix_ij(&ilu_ii, ilu, i, i);
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		subst_ctdvector_i(&ret_i, ret, i);
		if(cabs(ilu_ii) != 0.0)
		{
			for(j = 0; j < i; j++)
			{
				//get_ctdrsmatrix_ij(&ilu_ji, ilu, j, i);
				ilu_ji = get_cdrsmatrix_ij(ilu, j, i);
				if(cabs(ilu_ji) != 0.0)
				{
					subst_ctdvector_i(&ret_j, ret, j);
					//ret_j = ret_j - ilu_ji * ret_i;
					//rctd_mul(&ctmp, &ilu_ji, &ret_i);
					rctd_mul_cd(&ctmp, &ret_j, ilu_ji);
					rctd_sub(&ret_i, &ret_i, &ctmp);
					//printf("f: %ld, %ld\n", i, j);
				}
			}
			//ret_i /= ilu_ii;
			rctd_div_cd(&ret_i, &ret_i, ilu_ii);
			set_ctdvector_i(ret, i, &ret_i);
		}
	}

	// Backward substitution
	for(i = (row_dim - 1); i >= 0; i--)
	{
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		subst_ctdvector_i(&ret_i, ret, i);
		if(cabs(ilu_ii) != 0.0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				ilu_ji = get_cdrsmatrix_ij(ilu, j, i);
				if(cabs(ilu_ji) != 0.0)
				{
					subst_ctdvector_i(&ret_j, ret, j);
					//ret_i = ret_i - ilu_ij * ret_j;
					//rctd_mul(&ctmp, &ilu_ij, &ret_j);
					rctd_mul_cd(&ctmp, &ret_j, ilu_ji);
					rctd_sub(&ret_i, &ret_i, &ctmp);
					//printf("b: %ld, %ld\n", i, j);
				}
			}
			set_ctdvector_i(ret, i, &ret_i);
		}
	}
}

// iLUs0_solve: x^t * conj(iLU) = b^t
void solve_iLU0s_cdrsmatrix_ctdvec(CTDVector ret, CDRSMatrix ilu, CTDVector b)
{
	long int i, j, row_dim, col_dim;
	ctdfloat ret_i, ret_j, ctmp;
	double _Complex ilu_ii, ilu_ji, conj_ilu_ii, conj_ilu_ji;

	row_dim = ilu->re->row_dim;
	col_dim = ilu->re->col_dim;

	// ret := b
	subst_ctdvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		//get_ctdrsmatrix_ij(&ilu_ii, ilu, i, i);
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		subst_ctdvector_i(&ret_i, ret, i);
		if(cabs(ilu_ii) != 0.0)
		{
			conj_ilu_ii = conj(ilu_ii);
			for(j = 0; j < i; j++)
			{
				//get_ctdrsmatrix_ij(&ilu_ji, ilu, j, i);
				ilu_ji = get_cdrsmatrix_ij(ilu, j, i);
				if(cabs(ilu_ji) != 0.0)
				{
					conj_ilu_ji = conj(ilu_ji);
					subst_ctdvector_i(&ret_j, ret, j);
					//ret_j = ret_j - ilu_ji * ret_i;
					//rctd_mul(&ctmp, &ilu_ji, &ret_i);
					//rctd_mul_cd(&ctmp, &ret_j, ilu_ji);
					rctd_mul_cd(&ctmp, &ret_j, conj_ilu_ji);
					rctd_sub(&ret_i, &ret_i, &ctmp);
					//printf("f: %ld, %ld\n", i, j);
				}
			}
			//ret_i /= ilu_ii;
			//rctd_div_cd(&ret_i, &ret_i, ilu_ii);
			rctd_div_cd(&ret_i, &ret_i, conj_ilu_ii);
			set_ctdvector_i(ret, i, &ret_i);
		}
	}

	// Backward substitution
	for(i = (row_dim - 1); i >= 0; i--)
	{
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		subst_ctdvector_i(&ret_i, ret, i);
		if(cabs(ilu_ii) != 0.0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				ilu_ji = get_cdrsmatrix_ij(ilu, j, i);
				if(cabs(ilu_ji) != 0.0)
				{
					conj_ilu_ji = conj(ilu_ji); 
					subst_ctdvector_i(&ret_j, ret, j);
					//ret_i = ret_i - ilu_ij * ret_j;
					//rctd_mul(&ctmp, &ilu_ij, &ret_j);
					//rctd_mul_cd(&ctmp, &ret_j, ilu_ji);
					rctd_mul_cd(&ctmp, &ret_j, conj_ilu_ji);
					rctd_sub(&ret_i, &ret_i, &ctmp);
					//printf("b: %ld, %ld\n", i, j);
				}
			}
			set_ctdvector_i(ret, i, &ret_i);
		}
	}
}

#endif // ifdef USE_TDLINEAR
