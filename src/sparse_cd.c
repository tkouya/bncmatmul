/********************************************************************************/
/*                                                                              */
/* sparse_cd.c : Complex Sparse Matrix and Vector Library (Double Precision)    */
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
CDRSMatrix init_cdrsmatrix(long int row_dim, long int *nzero_col_dim, long int nzero_total_num)
{
	CDRSMatrix ret;
	long int i, j;

	ret = (CDRSMatrix)malloc(sizeof(cdrsmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "Cannot allocate CDRSMatrix\n");
		return ret;
	}
	
	if((row_dim < 0) || (nzero_total_num < 0))
	{
		fprintf(stderr, "Illigal nzero values\n");
		return NULL;
	}

    // re, im = init_drsmatrix
    ret->re = init_drsmatrix(row_dim, nzero_col_dim, nzero_total_num);
    ret->im = init_drsmatrix(row_dim, nzero_col_dim, nzero_total_num);

	return ret;
}

/* Clear CDRSMatrix */
void free_cdrsmatrix(CDRSMatrix mat)
{

    free_drsmatrix(mat->re);
    free_drsmatrix(mat->im);

	free(mat);
}

/* Print CDRSMatrix */
void print_cdrsmatrix(CDRSMatrix mat)
{
	long int i, j, total_index;

	if(mat == NULL)
		fprintf(stderr, "ERROR!\n");

	total_index = 0;
	for(i = 0; i < mat->re->row_dim; i++)
	{
		// Fix!: 2011-08-29 by T.Kouya
		if(mat->re->nzero_col_dim[i] >= 1)
		{
			printf("%5ld: ", i);
			//for(j = 0; j < mat->re->nzero_col_dim[i] - 1 ; j++)
			for(j = 0; j < mat->re->real_nzero_col_dim[i]; j++)
			{
				if(j < (mat->re->nzero_col_dim[i] - 1))
					printf("%ld->%e + %e * I, ", mat->re->nzero_index[i][j], mat->re->element[total_index], mat->im->element[total_index]);
				else if(j == (mat->re->nzero_col_dim[i] - 1))
					printf("%ld->%e + %e * I", mat->re->nzero_index[i][j], mat->re->element[total_index], mat->im->element[total_index]);

				total_index++;
			}
			printf("\n");
			//printf("%ld->%e + %e * I\n", mat->re->nzero_index[i][mat->re->nzero_col_dim[i] - 1], mat->re->element[total_index], mat->im->element[total_index]);
			//total_index++;
		}
	}
	return;
}

/* Dense Matrix := Sparse Matrix */
void set_cdmatrix_cdrsmatrix(CDMatrix ret, CDRSMatrix spmat)
{
	long int i, j, total_index;
	double _Complex ctmp;

	if((ret == NULL) || (spmat == NULL))
	{
		fprintf(stderr, "set_cdmatrix_cdrsmatrix: ERROR!\n");
		return;
	}

	// ret := 0
	set0_cdmatrix(ret);

	total_index = 0;
	for(i = 0; i < spmat->re->row_dim; i++)
	{
		//for(j = 0; j < spmat->re->nzero_col_dim[i]; j++)
		for(j = 0; j < spmat->re->real_nzero_col_dim[i]; j++)
		{
			if(j < spmat->re->nzero_col_dim[i])
			{
				//printf("%5ld: ", i);
				//printf("%ld->%e, ", spmat->nzero_index[i][j], spmat->element[total_index++]);
				ctmp = spmat->re->element[total_index] + spmat->im->element[total_index] * I;
				set_cdmatrix_ij(ret, i, spmat->re->nzero_index[i][j], ctmp); // spmat->element[total_index++]);	
			}
			total_index++; // Fix! 2024-09-26(Thu) T.Kouya
		}
	}
	return;
}

// 2024-09-05(Thu) T.Kouya
// Frobenius norm of mat
double normf_cdrsmatrix(CDRSMatrix mat)
{
	long int index;
	double ret, ret_sum2;

	ret_sum2 = 0.0;

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d mat4_re, mat4_im, tmp4_re, tmp4_im;

	// tmp4 := 0
	tmp4_re = _mm256_setzero_pd();
	tmp4_im = _mm256_setzero_pd();

	for(index = 0; index < mat->re->real_nzero_total_num; index += _BNC_D_WIDTH)
	{
		mat4_re = _mm256_load_pd(&(mat->re->element[index]));
		mat4_im = _mm256_load_pd(&(mat->im->element[index]));

		// tmp4_re += a4_re * a4_re
		// tmp4_im += a4_im * a4_im
		tmp4_re = _mm256_fmadd_pd(mat4_re, mat4_re, tmp4_re);
		tmp4_im = _mm256_fmadd_pd(mat4_im, mat4_im, tmp4_im);
	}

	ret_sum2  = tmp4_re[0] + tmp4_re[1] + tmp4_re[2] + tmp4_re[3];
	ret_sum2 += tmp4_im[0] + tmp4_im[1] + tmp4_im[2] + tmp4_im[3];

#elif defined(__AVX512F__) // __AVX512F__
	__m512d mat8_re, mat8_im, tmp8_re, tmp8_im;

	// tmp8 := 0
	tmp8_re = _mm512_setzero_pd();
	tmp8_im = _mm512_setzero_pd();

	for(index = 0; index < mat->re->real_nzero_total_num; index += _BNC_D_WIDTH)
	{
		mat8_re = _mm512_load_pd(&(mat->re->element[index]));
		mat8_im = _mm512_load_pd(&(mat->im->element[index]));

		// tmp8 += a8 * a8
		tmp8_re = _mm512_fmadd_pd(mat8_re, mat8_re, tmp8_re);
		tmp8_im = _mm512_fmadd_pd(mat8_im, mat8_im, tmp8_im);
	}

	ret_sum2  = tmp8_re[0] + tmp8_re[1] + tmp8_re[2] + tmp8_re[3] + tmp8_re[4] + tmp8_re[5] + tmp8_re[6] + tmp8_re[7];
	ret_sum2 += tmp8_im[0] + tmp8_im[1] + tmp8_im[2] + tmp8_im[3] + tmp8_im[4] + tmp8_im[5] + tmp8_im[6] + tmp8_im[7];

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2
	{
		long vl = (long)svcntd();
		svfloat64_t tmp_re = svdup_n_f64(0.0);
		svfloat64_t tmp_im = svdup_n_f64(0.0);
		for(index = 0; index < mat->re->real_nzero_total_num; index += vl)
		{
			svbool_t pg = svwhilelt_b64_s64((int64_t)index, (int64_t)mat->re->real_nzero_total_num);
			svfloat64_t a_re = svld1_f64(pg, &(mat->re->element[index]));
			svfloat64_t a_im = svld1_f64(pg, &(mat->im->element[index]));
			tmp_re = svmla_f64_x(pg, tmp_re, a_re, a_re);
			tmp_im = svmla_f64_x(pg, tmp_im, a_im, a_im);
		}
		ret_sum2  = svaddv_f64(svptrue_b64(), tmp_re);
		ret_sum2 += svaddv_f64(svptrue_b64(), tmp_im);
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM Neon
	{
		float64x2_t tmp2_re = vdupq_n_f64(0.0);
		float64x2_t tmp2_im = vdupq_n_f64(0.0);
		long int n = mat->re->real_nzero_total_num;
		long int n2 = (n / 2) * 2;
		for(index = 0; index < n2; index += 2)
		{
			float64x2_t a2_re = vld1q_f64(&(mat->re->element[index]));
			float64x2_t a2_im = vld1q_f64(&(mat->im->element[index]));
			tmp2_re = vfmaq_f64(tmp2_re, a2_re, a2_re);
			tmp2_im = vfmaq_f64(tmp2_im, a2_im, a2_im);
		}
		ret_sum2  = vgetq_lane_f64(tmp2_re, 0) + vgetq_lane_f64(tmp2_re, 1);
		ret_sum2 += vgetq_lane_f64(tmp2_im, 0) + vgetq_lane_f64(tmp2_im, 1);
		// tail
		for(index = n2; index < n; index++)
		{
			ret_sum2 += mat->re->element[index] * mat->re->element[index];
			ret_sum2 += mat->im->element[index] * mat->im->element[index];
		}
	}

#else // others
	//for(index = 0; index < mat->nzero_total_num; index++)
	for(index = 0; index < mat->re->real_nzero_total_num; index++)
	{
		ret_sum2 += mat->re->element[index] * mat->re->element[index];
		ret_sum2 += mat->im->element[index] * mat->im->element[index];
	}
#endif // __AVX2__

	ret = sqrt(ret_sum2);

	return ret;
}

/* initialize and substitute CDRSMatrix from CDMatrix */
CDRSMatrix init_set_cdrsmatrix_cdmatrix(CDMatrix org_mat)
{
	long int i, j;
	long int nzero_total_num, total_index, j_index;
	long int *ptr_nzero_col_dim;
	double _Complex ctmp;
	CDRSMatrix ret;

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
			if(cabs(get_cdmatrix_ij(org_mat, i, j)) != 0.0)
			{
				/* get nzero_row_dim, nzero_col_dim, nzero_total_num */
				nzero_total_num++;
				*(ptr_nzero_col_dim + i) += 1;
			}
		}
	}

	// initialize
	ret = init_cdrsmatrix(org_mat->row_dim, ptr_nzero_col_dim, nzero_total_num);

	/* Read org_mat (2nd) */
	total_index = 0;
	for(i = 0; i < ret->re->row_dim; i++)
	{
		j_index = 0;
		for(j = 0; j < ret->re->col_dim; j++)
		{
			if(cabs(get_cdmatrix_ij(org_mat, i, j)) != 0.0)
			{
				ret->re->nzero_index[i][j_index] =  j;
				ctmp = get_cdmatrix_ij(org_mat, i, j);
				ret->re->element[total_index] = creal(ctmp); //get_dmatrix_ij(org_mat, i, j);
				ret->im->element[total_index] = cimag(ctmp); //get_dmatrix_ij(org_mat, i, j);
				total_index++;
				j_index++;
			}
		}
	}

	free(ptr_nzero_col_dim);

	return ret;
}

/* Initialize and substitute CDRSMatrix from CDRSMatrix */
CDRSMatrix init_set_cdrsmatrix(CDRSMatrix org_mat)
{
	CDRSMatrix ret;

	// initialize
	ret = init_cdrsmatrix(org_mat->re->row_dim, org_mat->re->nzero_col_dim, org_mat->re->nzero_total_num);

	if(ret != NULL)
	{
		subst_drsmatrix(ret->re, org_mat->re);
		subst_drsmatrix(ret->im, org_mat->im);
	}

	return ret;
}

#ifdef USE_GMP
// CMPFRSMatrix -> CDRSMatrix
CDRSMatrix init_set_cdrsmatrix_cmpfrsmatrix(CMPFRSMatrix org_sp)
{
	CDRSMatrix ret;
    double _Complex cdtmp;
	long int org_sp_total_index, total_index, i, j;

    //printf("row_dim, nzero_total_num = %ld, %ld\n", org_sp->row_dim, org_sp->nzero_total_num);
    ret = init_cdrsmatrix(org_sp->row_dim, org_sp->nzero_col_dim, org_sp->nzero_total_num);
    //printf("init_ddrsmatrix!\n");

    // MPFR -> DD
    total_index = 0;
	org_sp_total_index = 0;
	for(i = 0; i < ret->re->row_dim; i++)
	{
		//printf("%ld: ", i);
		for(j = 0; j < ret->re->nzero_col_dim[i]; j++)
		{
            //mpf_get_dd(ddtmp, org_sp->element[org_sp_total_index + j]);
            ret->re->element[total_index + j] = mpf_get_d(mpc_realref(org_sp->element[org_sp_total_index + j]));
            ret->im->element[total_index + j] = mpf_get_d(mpc_imagref(org_sp->element[org_sp_total_index + j]));

            //ret->element[0][total_index + j] = ddtmp[0];
            //ret->element[1][total_index + j] = ddtmp[1];

            ret->re->nzero_index[i][j] = org_sp->nzero_index[i][j];
			ret->im->nzero_index[i][j] = org_sp->nzero_index[i][j];
			//total_index++;
			//printf("%ld ", j);
		}
		//printf("\n");
		total_index += ret->re->real_nzero_col_dim[i];
		org_sp_total_index += org_sp->nzero_col_dim[i];
	}
    //print_ddrsmatrix(ret);

	return ret;
}
#endif // USE_GMP

// 2024-11-07(Tue)
/* get the CDRSMatrix ij-element */
double _Complex get_cdrsmatrix_ij(CDRSMatrix mat, long int row_index, long int col_index)
{
	long int i, j, total_index;
	double _Complex ret = 0.0 + 0.0 * I;

	if((row_index < 0) || (row_index >= mat->re->row_dim) || (col_index < 0) || (col_index >= mat->re->col_dim))
	{
		fprintf(stderr, "Warning: row_index(%ld) or col_index(%ld) is illegal!\n", row_index, col_index);
		return ret; // mat->zero_element;
	}

	// finding mat_ij element
	//if(mat->nzero_col_dim[row_index] >= 1)
	if(mat->re->real_nzero_col_dim[row_index] >= 1)
	{
		total_index = 0;
		for(i = 0; i < row_index; i++)
			total_index += mat->re->real_nzero_col_dim[i];
			//total_index += mat->nzero_col_dim[i];

		for(j = 0; j < mat->re->nzero_col_dim[row_index] ; j++)
		{
			// Find!
			if(mat->re->nzero_index[row_index][j] == col_index)
			{
				ret = mat->re->element[total_index] + mat->im->element[total_index] * I;
				return ret; // mat->element[total_index];
			}
			
			total_index++;
		}
	}

	// Not found -> return 0
	return ret; // mat->re->zero_element;
}

// 2024-08-04 (Sun) T.Kouya
/* set the CDRSMatrix ij-element */
void set_cdrsmatrix_ij(CDRSMatrix mat, long int row_index, long int col_index, double _Complex val)
{
	long int i, j, total_index;

	if((row_index < 0) || (row_index >= mat->re->row_dim) || (col_index < 0) || (col_index >= mat->re->col_dim))
	{
		fprintf(stderr, "Warning: row_index(%ld) or col_index(%ld) is illegal!\n", row_index, col_index);
		//return mat->zero_element;
		return;
	}

	// finding mat_ij element
	if(mat->re->nzero_col_dim[row_index] >= 1)
	{
		total_index = 0;
		for(i = 0; i < row_index; i++)
			total_index += mat->re->real_nzero_col_dim[i];
			//total_index += mat->nzero_col_dim[i];

		for(j = 0; j < mat->re->nzero_col_dim[row_index] ; j++)
		{
			// Find!
			if(mat->re->nzero_index[row_index][j] == col_index)
			{
				//return mat->element[total_index];
				//mat->element[total_index] = val;
				mat->re->element[total_index] = creal(val);
				mat->im->element[total_index] = cimag(val);

				return;
			}
			
			total_index++;
		}
	}

	return;
}

/* Get variables to initialize DRSMatrix */
int get_vars_cdrsmatrix_fname(long int *ptr_row_dim, long int **ptr_nzero_col_dim, long int *ptr_nzero_total_num, const char *fname)
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

/* Multiply CDRSMatrix * CDVector */
int mul_cdrsmatrix_cdvec(CDVector ret, CDRSMatrix mat, CDVector vec)
{
	long int i, j, total_index;
	DVector in_vec_re, in_vec_im, in_ret_re, in_ret_im;
	DVector tmp_vec[4];

	if((ret->dim < mat->re->col_dim) || (vec->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	// vec_re * vec_im * i = vec
	in_vec_re = init_dvector(vec->dim);
	in_vec_im = init_dvector(vec->dim);
	separate_cdvector(in_vec_re, in_vec_im, vec);

	in_ret_re = init_dvector(mat->re->row_dim);
	in_ret_im = init_dvector(mat->re->row_dim);
	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_dvector(mat->re->row_dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	mul_drsmatrix_dvec(tmp_vec[0], mat->re, in_vec_re);
	mul_drsmatrix_dvec(tmp_vec[1], mat->re, in_vec_im);
	mul_drsmatrix_dvec(tmp_vec[2], mat->im, in_vec_re);
	mul_drsmatrix_dvec(tmp_vec[3], mat->im, in_vec_im);

	sub_dvector(in_ret_re, tmp_vec[0], tmp_vec[3]);
	add_dvector(in_ret_im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	free_dvector(in_vec_re);
	free_dvector(in_vec_im);
	free_dvector(in_ret_re);
	free_dvector(in_ret_im);
	for(i = 0; i < 4; i++)
		free_dvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply CDRSMatrix^T * CDVector */
int mul_cdrsmatrixt_cdvec(CDVector ret, CDRSMatrix mat, CDVector vec)
{
	long int i, j, total_index;
	DVector in_vec_re, in_vec_im, in_ret_re, in_ret_im;
	DVector tmp_vec[4];

	if((ret->dim < mat->re->col_dim) || (vec->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	// vec_re * vec_im * i = vec
	in_vec_re = init_dvector(mat->re->row_dim);
	in_vec_im = init_dvector(mat->re->row_dim);
	separate_cdvector(in_vec_re, in_vec_im, vec);

	in_ret_re = init_dvector(vec->dim);
	in_ret_im = init_dvector(vec->dim);
	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_dvector(vec->dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	mul_drsmatrixt_dvec(tmp_vec[0], mat->re, in_vec_re);
	mul_drsmatrixt_dvec(tmp_vec[1], mat->re, in_vec_im);
	mul_drsmatrixt_dvec(tmp_vec[2], mat->im, in_vec_re);
	mul_drsmatrixt_dvec(tmp_vec[3], mat->im, in_vec_im);

	sub_dvector(in_ret_re, tmp_vec[0], tmp_vec[3]);
	add_dvector(in_ret_im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	free_dvector(in_vec_re);
	free_dvector(in_vec_im);
	free_dvector(in_ret_re);
	free_dvector(in_ret_im);

	// unknown error
	// 2024-09-24 T.Kouya
	for(i = 0; i < 4; i++)
		free_dvector(tmp_vec[i]);
	//for(i = 0; i < 4; i++)
	//	free(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply conj(CDRSMatrix^T) * CDVector */
int mul_cdrsmatrixs_cdvec(CDVector ret, CDRSMatrix mat, CDVector vec)
{
	long int i, j, total_index;
	DVector in_vec_re, in_vec_im, in_ret_re, in_ret_im;
	DVector tmp_vec[4];

	if((ret->dim < mat->re->col_dim) || (vec->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	// vec_re * vec_im * i = vec
	in_vec_re = init_dvector(mat->re->row_dim);
	in_vec_im = init_dvector(mat->re->row_dim);
	separate_cdvector(in_vec_re, in_vec_im, vec);

	in_ret_re = init_dvector(vec->dim);
	in_ret_im = init_dvector(vec->dim);
	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_dvector(vec->dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	mul_drsmatrixt_dvec(tmp_vec[0], mat->re, in_vec_re);
	mul_drsmatrixt_dvec(tmp_vec[1], mat->re, in_vec_im);
	mul_drsmatrixt_dvec(tmp_vec[2], mat->im, in_vec_re);
	mul_drsmatrixt_dvec(tmp_vec[3], mat->im, in_vec_im);

	// mat_re * vec_re + mat_im * vec_im
	add_dvector(in_ret_re, tmp_vec[0], tmp_vec[3]);
	// mat_re * vec_im - mat_im * vec_re
	sub_dvector(in_ret_im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	free_dvector(in_vec_re);
	free_dvector(in_vec_im);
	free_dvector(in_ret_re);
	free_dvector(in_ret_im);

	// unknown error
	// 2024-09-24 T.Kouya
	for(i = 0; i < 4; i++)
		free_dvector(tmp_vec[i]);
	//for(i = 0; i < 4; i++)
	//	free(tmp_vec[i]);

	return SUCCESS;
}

// Imcomplete LU decomposition; iLU0_drsmatrix
void iLU0_cdrsmatrix(CDRSMatrix mat)
{
	long int i, j, k, row_dim, col_dim;
	double _Complex aii, aji, ajk, aik;

	row_dim = mat->re->row_dim;
	col_dim = mat->re->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		aii = get_cdrsmatrix_ij(mat, i, i);
		if(cabs(aii) != 0.0)
		{
			//printf("%ld: ", i);
			for(j = i + 1; j < row_dim; j++)
			{
				aji = get_cdrsmatrix_ij(mat, j, i);
				if(cabs(aji) != 0.0)
				{
					aji /= aii;
					set_cdrsmatrix_ij(mat, j, i, aji);
					for(k = i + 1; k < col_dim; k++)
					{
						ajk = get_cdrsmatrix_ij(mat, j, k);
						aik = get_cdrsmatrix_ij(mat, i, k);
						if((cabs(ajk) != 0.0) && (cabs(aik) != 0.0))
						{
							ajk = ajk - aji * aik;
							set_cdrsmatrix_ij(mat, j, k, ajk);
							//printf("%ld, %ld, %ld\n", i, j, k);
						}
					}
				}
			}
		}
	}
}

// iLU0_solve: iLU * x = b
void solve_iLU0_cdrsmatrix(CDVector ret, CDRSMatrix ilu, CDVector b)
{
	long int i, j, row_dim, col_dim;
	double _Complex ret_i, ret_j, ilu_ii, ilu_ij;

	row_dim = ilu->re->row_dim;
	col_dim = ilu->re->col_dim;

	// ret := b
	subst_cdvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		ret_i = get_cdvector_i(ret, i);
		if(cabs(ilu_ii) != 0.0)
		{
			//for(j = (i + 1); j < col_dim; j++)
			for(j = 0; j < i; j++)
			{
				ilu_ij = get_cdrsmatrix_ij(ilu, i, j);
				if(cabs(ilu_ij) != 0.0)
				{
					ret_j = get_cdvector_i(ret, j);
					ret_i = ret_i - ilu_ij * ret_j;
					//set_cdvector_i(ret, j, ret_j);
				}
			}
			set_cdvector_i(ret, i, ret_i);
		}
	}

	// Backward substitution
	for(i = (row_dim - 1); i >= 0; i--)
	{
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		ret_i = get_cdvector_i(ret, i);
		if(cabs(ilu_ii) != 0.0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				ilu_ij = get_cdrsmatrix_ij(ilu, i, j);
				if(cabs(ilu_ij) != 0.0)
				{
					ret_j = get_cdvector_i(ret, j);
					ret_i = ret_i - ilu_ij * ret_j;
					//set_cdvector_i(ret, i, ret_i);
				}
			}
			ret_i /= ilu_ii;
			set_cdvector_i(ret, i, ret_i);
		}
	}
}

// iLU0t_solve: x^T * iLU = b^T
void solve_iLU0t_cdrsmatrix(CDVector ret, CDRSMatrix ilu, CDVector b)
{
	long int i, j, row_dim, col_dim;
	double _Complex ret_i, ret_j, ilu_ii, ilu_ji;

	row_dim = ilu->re->row_dim;
	col_dim = ilu->re->col_dim;

	// ret := b
	subst_cdvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		ret_i = get_cdvector_i(ret, i);
		if(cabs(ilu_ii) != 0.0)
		{
			for(j = 0; j < i; j++)
			{
				ilu_ji = get_cdrsmatrix_ij(ilu, j, i);
				if(cabs(ilu_ji) != 0.0)
				{
					ret_j = get_cdvector_i(ret, j);
					ret_i = ret_i - ilu_ji * ret_j;
					//set_cdvector_i(ret, i, ret_i);
				}
			}
			ret_i /= ilu_ii;
			set_cdvector_i(ret, i, ret_i);
		}
	}

	// Backward substitution
	for(i = row_dim - 1; i >= 0; i--)
	{
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		ret_i = get_cdvector_i(ret, i);
		if(cabs(ilu_ii) != 0.0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				ilu_ji = get_cdrsmatrix_ij(ilu, j, i);
				if(cabs(ilu_ji) != 0.0)
				{
					ret_j = get_cdvector_i(ret, j);
					ret_i = ret_i - ilu_ji * ret_j;
					//set_cdvector_i(ret, j, ret_j);
				}
			}
			set_cdvector_i(ret, i, ret_i);
		}
	}
}

// iLU0s_solve: x^T * conj(iLU) = b^T
void solve_iLU0s_cdrsmatrix(CDVector ret, CDRSMatrix ilu, CDVector b)
{
	long int i, j, row_dim, col_dim;
	double _Complex ret_i, ret_j, ilu_ii, ilu_ji, conj_ilu_ii, conj_ilu_ji;

	row_dim = ilu->re->row_dim;
	col_dim = ilu->re->col_dim;

	// ret := b
	subst_cdvector(ret, b);

	// Forward substitution: solve conj(U)^T * y = b
	for(i = 0; i < row_dim; i++)
	{
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		ret_i = get_cdvector_i(ret, i);
		if(cabs(ilu_ii) != 0.0)
		{
			conj_ilu_ii = conj(ilu_ii);
			for(j = 0; j < i; j++)
			{
				ilu_ji = get_cdrsmatrix_ij(ilu, j, i);
				if(cabs(ilu_ji) != 0.0)
				{
					conj_ilu_ji = conj(ilu_ji);
					ret_j = get_cdvector_i(ret, j);
					//ret_i = ret_i - ilu_ji * ret_j;
					ret_i = ret_i - conj_ilu_ji * ret_j;
					//set_cdvector_i(ret, i, ret_i);
				}
			}
			//ret_i /= ilu_ii;
			ret_i /= conj_ilu_ii;
			set_cdvector_i(ret, i, ret_i);
		}
	}

	// Backward substitution
	for(i = row_dim - 1; i >= 0; i--)
	{
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		ret_i = get_cdvector_i(ret, i);
		if(cabs(ilu_ii) != 0.0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				ilu_ji = get_cdrsmatrix_ij(ilu, j, i);
				if(cabs(ilu_ji) != 0.0)
				{
					conj_ilu_ji = conj(ilu_ji);
					ret_j = get_cdvector_i(ret, j);
					//ret_i = ret_i - ilu_ji * ret_j;
					ret_i = ret_i - conj_ilu_ji * ret_j;
					//set_cdvector_i(ret, j, ret_j);
				}
			}
			set_cdvector_i(ret, i, ret_i);
		}
	}
}
