/********************************************************************************/
/*                                                                              */
/* sparse_cdd.c : Complex Sparse Matrix and Vector Library (DD Precision)       */
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

/* initialize CDDRSMatrix */
CDDRSMatrix init_cddrsmatrix(long int row_dim, long int *nzero_col_dim, long int nzero_total_num)
{
	CDDRSMatrix ret;
	long int i, j;

	ret = (CDDRSMatrix)malloc(sizeof(cddrsmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "Cannot allocate CDDRSMatrix\n");
		return ret;
	}
	
	if((row_dim < 0) || (nzero_total_num < 0))
	{
		fprintf(stderr, "Illigal nzero values\n");
		return NULL;
	}

    // re, im = init_drsmatrix
    ret->re = init_ddrsmatrix(row_dim, nzero_col_dim, nzero_total_num);
    ret->im = init_ddrsmatrix(row_dim, nzero_col_dim, nzero_total_num);

	return ret;
}

/* Clear CDDRSMatrix */
void free_cddrsmatrix(CDDRSMatrix mat)
{

    free_ddrsmatrix(mat->re);
    free_ddrsmatrix(mat->im);

	free(mat);
}

/* Print CDDRSMatrix */
void print_cddrsmatrix(CDDRSMatrix mat)
{
	long int i, j, total_index;
	ddfloat tmp;

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
                    rdd_out_str(tmp.val);
                    printf(" + ");
                    tmp.val[0] = mat->im->element[0][total_index + j];
                    tmp.val[1] = mat->im->element[1][total_index + j];
                    rdd_out_str(tmp.val);
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

// 2024-09-05(Thu) T.Kouya
// Frobenius norm of mat
void normf_cddrsmatrix(double ret[DDSIZE], CDDRSMatrix mat)
{
	long int index;
	double tmp[DDSIZE], ret_sum2[DDSIZE];

	//ret_sum2 = 0.0;
	rdd_set0(ret_sum2);

#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
	__m256d mat4_re[DDSIZE], mat4_im[DDSIZE], tmp4[DDSIZE], mul4_re[DDSIZE], mul4_im[DDSIZE];

	// tmp4 := 0
	_bncavx2_rdd_set0(tmp4); // = _mm256_setzero_pd();

	for(index = 0; index < mat->re->real_nzero_total_num; index += _BNC_D_WIDTH)
	{
		mat4_re[0] = _mm256_load_pd(&(mat->re->element[0][index]));
		mat4_re[1] = _mm256_load_pd(&(mat->re->element[1][index]));
		mat4_im[0] = _mm256_load_pd(&(mat->im->element[0][index]));
		mat4_im[1] = _mm256_load_pd(&(mat->im->element[1][index]));

		// tmp4 += a4 * a4
		_bncavx2_rdd_mul(mul4_re, mat4_re, mat4_re);
		_bncavx2_rdd_mul(mul4_im, mat4_im, mat4_im);

		//tmp4 = _mm256_fmadd_pd(mat4, mat4, tmp4);
		_bncavx2_rdd_add(tmp4, tmp4, mul4_re);
		_bncavx2_rdd_add(tmp4, tmp4, mul4_im);
	}

	//ret_sum2 = tmp4[0] + tmp4[1] + tmp4[2] + tmp4[3];
	_bncavx2_rdd_sum256d(ret_sum2, tmp4);

#elif defined(__AVX512F__) // __AVX512F__
	__m512d mat8_re[DDSIZE], mat8_im[DDSIZE], tmp8[DDSIZE], mul8_re[DDSIZE], mul8_im[DDSIZE];

	// tmp8 := 0
	//tmp8 = _mm512_setzero_pd();
	_bncavx512_rdd_set0(tmp8);

	for(index = 0; index < mat->re->real_nzero_total_num; index += _BNC_D_WIDTH)
	{
		mat8_re[0] = _mm512_load_pd(&(mat->re->element[0][index]));
		mat8_re[1] = _mm512_load_pd(&(mat->re->element[1][index]));
		mat8_im[0] = _mm512_load_pd(&(mat->im->element[0][index]));
		mat8_im[1] = _mm512_load_pd(&(mat->im->element[1][index]));

		// tmp8 += a8 * a8
		_bncavx512_rdd_mul(mul8_re, mat8_re, mat8_re);
		_bncavx512_rdd_mul(mul8_im, mat8_im, mat8_im);

		//tmp8= _mm512_fmadd_pd(mat8, mat8, tmp8);
		_bncavx512_rdd_add(tmp8, tmp8, mul8_re);
		_bncavx512_rdd_add(tmp8, tmp8, mul8_im);
	}

	//ret_sum2 = tmp8[0] + tmp8[1] + tmp8[2] + tmp8[3] + tmp8[4] + tmp8[5] + tmp8[6] + tmp8[7];
	_bncavx512_rdd_sum512d(ret_sum2, tmp8);

#elif defined(__ARM_SVE2) && defined(BNC_ENABLE_SVE2) // Arm SVE2
	{
		long _vl = (long)svcntd();
		svfloat64_t t0, t1;
		_bncsve2_rdd_set0(&t0, &t1);
		for(index = 0; index < mat->re->real_nzero_total_num; index += _vl)
		{
			svbool_t pg = svwhilelt_b64_s64((int64_t)index,
			                                (int64_t)mat->re->real_nzero_total_num);
			svfloat64_t r0 = svld1_f64(pg, &(mat->re->element[0][index]));
			svfloat64_t r1 = svld1_f64(pg, &(mat->re->element[1][index]));
			svfloat64_t i0 = svld1_f64(pg, &(mat->im->element[0][index]));
			svfloat64_t i1 = svld1_f64(pg, &(mat->im->element[1][index]));
			svfloat64_t m0, m1;
			/* tmp += |re|^2 + |im|^2 */
			_bncsve2_rdd_mul(pg, &m0, &m1, r0, r1, r0, r1);
			_bncsve2_rdd_add(pg, &t0, &t1, t0, t1, m0, m1);
			_bncsve2_rdd_mul(pg, &m0, &m1, i0, i1, i0, i1);
			_bncsve2_rdd_add(pg, &t0, &t1, t0, t1, m0, m1);
		}
		{
			long _L, _vl = (long)svcntd();
			double _la0[64], _la1[64];
			svst1_f64(svptrue_b64(), _la0, t0);
			svst1_f64(svptrue_b64(), _la1, t1);
			rdd_set_ui(ret_sum2, 0UL);
			for(_L = 0; _L < _vl; _L++)
			{
				rdd_add_d(ret_sum2, ret_sum2, _la0[_L]);
				rdd_add_d(ret_sum2, ret_sum2, _la1[_L]);
			}
		}
	}

#elif defined(__ARM_NEON) && defined(BNC_ENABLE_NEON) // ARM Neon
	float64x2_t mat4_re[DDSIZE], mat4_im[DDSIZE], tmp2[DDSIZE], mul4_re[DDSIZE], mul4_im[DDSIZE];

	// tmp2 := 0
	_bncneon_rdd_set0(tmp2); // = vdupq_n_f64(0.0);

	for(index = 0; index < mat->re->real_nzero_total_num; index += 2)
	{
		mat4_re[0] = vld1q_f64(&(mat->re->element[0][index]));
		mat4_re[1] = vld1q_f64(&(mat->re->element[1][index]));
		mat4_im[0] = vld1q_f64(&(mat->im->element[0][index]));
		mat4_im[1] = vld1q_f64(&(mat->im->element[1][index]));

		// tmp2 += a2 * a2
		_bncneon_rdd_mul(mul4_re, mat4_re, mat4_re);
		_bncneon_rdd_mul(mul4_im, mat4_im, mat4_im);

		//tmp2 = vfmaq_f64(tmp2, mat2, mat2);
		_bncneon_rdd_add(tmp2, tmp2, mul4_re);
		_bncneon_rdd_add(tmp2, tmp2, mul4_im);
	}

	//ret_sum2 = tmp2[0] + tmp2[1];
	_bncneon_rdd_sum128d(ret_sum2, tmp2);



#else // others
	double mat_element[DDSIZE], mul[DDSIZE];

	//for(index = 0; index < mat->nzero_total_num; index++)
	for(index = 0; index < mat->re->real_nzero_total_num; index++)
	{
		//ret_sum2 += mat->element[index] * mat->element[index];
		mat_element[0] = mat->re->element[0][index];
		mat_element[1] = mat->re->element[1][index];
		rdd_mul(mul, mat_element, mat_element);
		rdd_add(ret_sum2, ret_sum2, mul);

		mat_element[0] = mat->im->element[0][index];
		mat_element[1] = mat->im->element[1][index];
		rdd_mul(mul, mat_element, mat_element);
		rdd_add(ret_sum2, ret_sum2, mul);
	}
#endif // __AVX2__

	//ret = sqrt(ret_sum2);
	rdd_sqrt(ret, ret_sum2);

	//return ret;
}

/* Dense Matrix := Sparse Matrix */
void set_cddmatrix_cddrsmatrix(CDDMatrix ret, CDDRSMatrix spmat)
{
	long int i, j, total_index;
    double dtmp[DDSIZE];

	if((ret == NULL) || (spmat == NULL))
	{
		fprintf(stderr, "set_cddmatrix_cddrsmatrix: ERROR!\n");
		return;
	}

	// ret := 0
	set0_cddmatrix(ret);

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
			set_ddmatrix_ij(ret->re, i, spmat->re->nzero_index[i][j], dtmp);
            dtmp[0] = spmat->im->element[0][total_index];
            dtmp[1] = spmat->im->element[1][total_index];
            set_ddmatrix_ij(ret->im, i, spmat->im->nzero_index[i][j], dtmp);
            total_index++;
		}
	}
	return;
}

/* initialize and substitute DRSMatrix from DMatrix */
CDDRSMatrix init_set_cddrsmatrix_cddmatrix(CDDMatrix org_mat)
{
	long int i, j;
	long int nzero_total_num, total_index, j_index;
	long int *ptr_nzero_col_dim;
	cddfloat ctmp, *ptr_ctmp;
    ddfloat dtmp;
	CDDRSMatrix ret;

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
            ptr_ctmp = get_cddmatrix_ij(org_mat, i, j);
			//if(cabs(get_cdmatrix_ij(org_mat, i, j)) != 0.0)
            rcdd_abs(&dtmp, ptr_ctmp);
            if(rdd_cmp_ui(dtmp.val, 0UL) != 0)
			{
				/* get nzero_row_dim, nzero_col_dim, nzero_total_num */
				nzero_total_num++;
				*(ptr_nzero_col_dim + i) += 1;
			}
		}
	}

	// initialize
	ret = init_cddrsmatrix(org_mat->re->row_dim, ptr_nzero_col_dim, nzero_total_num);

	/* Read org_mat (2nd) */
	total_index = 0;
	for(i = 0; i < ret->re->row_dim; i++)
	{
		j_index = 0;
		for(j = 0; j < ret->re->col_dim; j++)
		{
            ptr_ctmp = get_cddmatrix_ij(org_mat, i, j);
            rcdd_abs(&dtmp, ptr_ctmp);
			//if(cabs(get_cdmatrix_ij(org_mat, i, j)) != 0.0)
            if(rdd_cmp_ui(dtmp.val, 0UL) != 0)
			{
				ret->re->nzero_index[i][j_index] =  j;
				//get_cddmatrix_ij(&ctmp, org_mat, i, j);
                ptr_ctmp = get_cddmatrix_ij(org_mat, i, j);
				//rdd_set(&(ret->re->element[total_index]), ptr_ctmp->val_re); //get_dmatrix_ij(org_mat, i, j);
				//rdd_set(&(ret->im->element[total_index]), ptr_ctmp->val_im); //get_dmatrix_ij(org_mat, i, j);
				ret->re->element[0][total_index] = ptr_ctmp->val_re[0];
				ret->re->element[1][total_index] = ptr_ctmp->val_re[1];
				ret->im->element[0][total_index] = ptr_ctmp->val_im[0];
				ret->im->element[1][total_index] = ptr_ctmp->val_im[1];

				total_index++;
				j_index++;
			}
		}
	}

	free(ptr_nzero_col_dim);

	return ret;
}

// 2024-09-04(Wed) T.Kouya
/* get the CDDRSMatrix ij-element */
void get_cddrsmatrix_ij(cddfloat *ret, CDDRSMatrix mat, long int row_index, long int col_index)
{
    get_ddrsmatrix_ij(ret->val_re, mat->re, row_index, col_index);
    get_ddrsmatrix_ij(ret->val_im, mat->im, row_index, col_index);
}

// 2024-09-04 (Wed) T.Kouya
/* set the CDDRSMatrix ij-element */
void set_cddrsmatrix_ij(CDDRSMatrix mat, long int row_index, long int col_index, cddfloat *val)
{
    set_ddrsmatrix_ij(mat->re, row_index, col_index, val->val_re);
    set_ddrsmatrix_ij(mat->im, row_index, col_index, val->val_im);
}

// 2024-12-03 (Tue) T.Kouya
CDDRSMatrix init_set_cddrsmatrix(CDDRSMatrix org_mat)
{
	CDDRSMatrix ret;

	ret = init_cddrsmatrix(org_mat->re->row_dim, org_mat->re->nzero_col_dim, org_mat->re->nzero_total_num);
	subst_ddrsmatrix(ret->re, org_mat->re);
	subst_ddrsmatrix(ret->im, org_mat->im);

	return ret;
}

#ifdef USE_GMP
// CMPFRSMatrix -> CDDRSMatrix
CDDRSMatrix init_set_cddrsmatrix_cmpfrsmatrix(CMPFRSMatrix org_sp)
{
	CDDRSMatrix ret;
    cddfloat cddtmp;
	long int org_sp_total_index, total_index, i, j;

    //printf("row_dim, nzero_total_num = %ld, %ld\n", org_sp->row_dim, org_sp->nzero_total_num);
    ret = init_cddrsmatrix(org_sp->row_dim, org_sp->nzero_col_dim, org_sp->nzero_total_num);
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
            mpf_get_dd(cddtmp.val_re, mpc_realref(org_sp->element[org_sp_total_index + j]));
            mpf_get_dd(cddtmp.val_im, mpc_imagref(org_sp->element[org_sp_total_index + j]));

            //ret->element[0][total_index + j] = ddtmp[0];
            //ret->element[1][total_index + j] = ddtmp[1];
            ret->re->element[0][total_index + j] = cddtmp.val_re[0];
            ret->re->element[1][total_index + j] = cddtmp.val_re[1];
            ret->im->element[0][total_index + j] = cddtmp.val_im[0];
            ret->im->element[1][total_index + j] = cddtmp.val_im[1];

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

/* Get variables to initialize DRSMatrix */
int get_vars_cddrsmatrix_fname(long int *ptr_row_dim, long int **ptr_nzero_col_dim, long int *ptr_nzero_total_num, const char *fname)
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

/* Multiply CDDRSMatrix * CDDVector */
int mul_cddrsmatrix_cddvec(CDDVector ret, CDDRSMatrix mat, CDDVector vec)
{
	long int i, j, total_index;
	DDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_ddvector(mat->re->row_dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	mul_ddrsmatrix_ddvec(tmp_vec[0], mat->re, vec->re);
	mul_ddrsmatrix_ddvec(tmp_vec[1], mat->re, vec->im);
	mul_ddrsmatrix_ddvec(tmp_vec[2], mat->im, vec->re);
	mul_ddrsmatrix_ddvec(tmp_vec[3], mat->im, vec->im);

	sub_ddvector(ret->re, tmp_vec[0], tmp_vec[3]);
	add_ddvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_ddvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply CDDRSMatrix^T * CDDVector */
int mul_cddrsmatrixt_cddvec(CDDVector ret, CDDRSMatrix mat, CDDVector vec)
{
	long int i, j, total_index;
	DDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_ddvector(vec->re->dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	mul_ddrsmatrixt_ddvec(tmp_vec[0], mat->re, vec->re);
	mul_ddrsmatrixt_ddvec(tmp_vec[1], mat->re, vec->im);
	mul_ddrsmatrixt_ddvec(tmp_vec[2], mat->im, vec->re);
	mul_ddrsmatrixt_ddvec(tmp_vec[3], mat->im, vec->im);

	sub_ddvector(ret->re, tmp_vec[0], tmp_vec[3]);
	add_ddvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_ddvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply conj(CDDRSMatrix)^T * CDDVector */
int mul_cddrsmatrixs_cddvec(CDDVector ret, CDDRSMatrix mat, CDDVector vec)
{
	long int i, j, total_index;
	DDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_ddvector(vec->re->dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	mul_ddrsmatrixt_ddvec(tmp_vec[0], mat->re, vec->re);
	mul_ddrsmatrixt_ddvec(tmp_vec[1], mat->re, vec->im);
	mul_ddrsmatrixt_ddvec(tmp_vec[2], mat->im, vec->re);
	mul_ddrsmatrixt_ddvec(tmp_vec[3], mat->im, vec->im);

	// mat_re * vec_re + mat_im * vec_im
	add_ddvector(ret->re, tmp_vec[0], tmp_vec[3]);
	// mat_re * vec_im - mat_im * vec_re
	sub_ddvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_ddvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply CDRSMatrix * CDDVector */
int mul_cdrsmatrix_cddvec(CDDVector ret, CDRSMatrix mat, CDDVector vec)
{
	long int i, j, total_index;
	DDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_ddvector(mat->re->row_dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	mul_drsmatrix_ddvec(tmp_vec[0], mat->re, vec->re);
	mul_drsmatrix_ddvec(tmp_vec[1], mat->re, vec->im);
	mul_drsmatrix_ddvec(tmp_vec[2], mat->im, vec->re);
	mul_drsmatrix_ddvec(tmp_vec[3], mat->im, vec->im);

	sub_ddvector(ret->re, tmp_vec[0], tmp_vec[3]);
	add_ddvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_ddvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply CDRSMatrix^T * CDDVector */
int mul_cdrsmatrixt_cddvec(CDDVector ret, CDRSMatrix mat, CDDVector vec)
{
	long int i, j, total_index;
	DDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_ddvector(mat->re->row_dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	mul_drsmatrixt_ddvec(tmp_vec[0], mat->re, vec->re);
	mul_drsmatrixt_ddvec(tmp_vec[1], mat->re, vec->im);
	mul_drsmatrixt_ddvec(tmp_vec[2], mat->im, vec->re);
	mul_drsmatrixt_ddvec(tmp_vec[3], mat->im, vec->im);

	sub_ddvector(ret->re, tmp_vec[0], tmp_vec[3]);
	add_ddvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_ddvector(tmp_vec[i]);

	return SUCCESS;
}

/* Multiply conj(CDRSMatrix)^T * CDDVector */
int mul_cdrsmatrixs_cddvec(CDDVector ret, CDRSMatrix mat, CDDVector vec)
{
	long int i, j, total_index;
	DDVector tmp_vec[4];

	if((ret->re->dim < mat->re->col_dim) || (vec->re->dim != mat->re->col_dim))
	{
		fprintf(stderr, "Illegal dimension!\n");
		return ERROR;
	}

	total_index = 0;

	for(i = 0; i < 4; i++)
		tmp_vec[i] = init_ddvector(mat->re->row_dim);

	// mat_re * vec_re, mat_re * vec_im, mat_im * vec_re, mat_im * vec_im
	mul_drsmatrixt_ddvec(tmp_vec[0], mat->re, vec->re);
	mul_drsmatrixt_ddvec(tmp_vec[1], mat->re, vec->im);
	mul_drsmatrixt_ddvec(tmp_vec[2], mat->im, vec->re);
	mul_drsmatrixt_ddvec(tmp_vec[3], mat->im, vec->im);

	// mat_re * vec_re + mat_im * vec_im
	add_ddvector(ret->re, tmp_vec[0], tmp_vec[3]);
	// mat_re * vec_im - mat_im * vec_re
	sub_ddvector(ret->im, tmp_vec[1], tmp_vec[2]);

	// ret := in_ret_re + in_ret_im * I
	//merge_cdvector(ret, in_ret_re, in_ret_im);

	// free
	for(i = 0; i < 4; i++)
		free_ddvector(tmp_vec[i]);

	return SUCCESS;
}

// Incomplete LU decomposition; iLU0_drsmatrix
void iLU0_cddrsmatrix(CDDRSMatrix mat)
{
	long int i, j, k, row_dim, col_dim;
	cddfloat aii, aji, ajk, aik, ctmp;
	//double dtmp[DDSIZE];

	row_dim = mat->re->row_dim;
	col_dim = mat->re->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		get_cddrsmatrix_ij(&aii, mat, i, i);
		if(rcdd_cmp_abs_ui(&aii, 0UL) != 0)
		{
			for(j = i + 1; j < row_dim; j++)
			{
				get_cddrsmatrix_ij(&aji, mat, j, i);
				if(rcdd_cmp_abs_ui(&aji, 0UL) != 0)
				{
					//aji /= aii;
					rcdd_div(&aji, &aji, &aii);
					set_cddrsmatrix_ij(mat, j, i, &aji);
					for(k = i + 1; k < col_dim; k++)
					{
						get_cddrsmatrix_ij(&ajk, mat, j, k);
						get_cddrsmatrix_ij(&aik, mat, i, k);
						if((rcdd_cmp_abs_ui(&ajk, 0UL) != 0) && (rcdd_cmp_abs_ui(&aik, 0UL) != 0))
						{
							//ajk = ajk - aji * aik;
							rcdd_mul(&ctmp, &aji, &aik);
							rcdd_sub(&ajk, &ajk, &ctmp);
							set_cddrsmatrix_ij(mat, j, k, &ajk);
							//printf("%ld, %ld, %ld\n", i, j, k);
						}
					}
				}
			}
		}
	}
}

// iLU0_solve: iLU * x = b
void solve_iLU0_cddrsmatrix(CDDVector ret, CDDRSMatrix ilu, CDDVector b)
{
	long int i, j, row_dim, col_dim;
	cddfloat ret_i, ret_j, ilu_ii, ilu_ij, ctmp;

	row_dim = ilu->re->row_dim;
	col_dim = ilu->re->col_dim;

	// ret := b
	subst_cddvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		get_cddrsmatrix_ij(&ilu_ii, ilu, i, i);
		subst_cddvector_i(&ret_i, ret, i);
		if(rcdd_cmp_abs_ui(&ilu_ii, 0UL) != 0)
		{
			for(j = 0; j < i; j++)
			{
				get_cddrsmatrix_ij(&ilu_ij, ilu, i, j);
				if(rcdd_cmp_abs_ui(&ilu_ij, 0UL) != 0)
				{
					subst_cddvector_i(&ret_j, ret, j);
					//ret_j = ret_j - ilu_ji * ret_i;
					rcdd_mul(&ctmp, &ilu_ij, &ret_j);
					rcdd_sub(&ret_i, &ret_i, &ctmp);
					//set_cddvector_i(ret, j, &ret_j);
					//printf("f: %ld, %ld\n", i, j);
				}
			}
			set_cddvector_i(ret, i, &ret_i);
		}
	}

	// Backward substitution
	for(i = (row_dim - 1); i >= 0; i--)
	{
		get_cddrsmatrix_ij(&ilu_ii, ilu, i, i);
		subst_cddvector_i(&ret_i, ret, i);
		if(rcdd_cmp_abs_ui(&ilu_ii, 0UL) != 0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				get_cddrsmatrix_ij(&ilu_ij, ilu, i, j);
				if(rcdd_cmp_abs_ui(&ilu_ij, 0UL) != 0)
				{
					subst_cddvector_i(&ret_j, ret, j);
					//ret_i = ret_i - ilu_ij * ret_j;
					rcdd_mul(&ctmp, &ilu_ij, &ret_j);
					rcdd_sub(&ret_i, &ret_i, &ctmp);
					//set_cddvector_i(ret, i, &ret_i);
					//printf("b: %ld, %ld\n", i, j);
				}
			}
			//ret_i /= ilu_ii;
			rcdd_div(&ret_i, &ret_i, &ilu_ii);
			set_cddvector_i(ret, i, &ret_i);
		}
	}
}

// iLU0t_solve: x^T * iLU = b^T
void solve_iLU0t_cddrsmatrix(CDDVector ret, CDDRSMatrix ilu, CDDVector b)
{
	long int i, j, row_dim, col_dim;
	cddfloat ret_i, ret_j, ctmp, ilu_ii, ilu_ji;

	row_dim = ilu->re->row_dim;
	col_dim = ilu->re->col_dim;

	// ret := b
	subst_cddvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		get_cddrsmatrix_ij(&ilu_ii, ilu, i, i);
		subst_cddvector_i(&ret_i, ret, i);
		if(rcdd_cmp_abs_ui(&ilu_ii, 0UL) != 0)
		{
			for(j = 0; j < i; j++)
			{
				get_cddrsmatrix_ij(&ilu_ji, ilu, j, i);
				if(rcdd_cmp_abs_ui(&ilu_ji, 0UL) != 0)
				{
					subst_cddvector_i(&ret_j, ret, j);
					//ret_i = ret_i - ilu_ji * ret_j;
					rcdd_mul(&ctmp, &ret_j, &ilu_ji);
					rcdd_sub(&ret_i, &ret_i, &ctmp);
					//set_cdvector_i(ret, i, ret_i);
				}
			}
			//ret_i /= ilu_ii;
			rcdd_div(&ret_i, &ret_i, &ilu_ii);
			set_cddvector_i(ret, i, &ret_i);
		}
	}

	// Backward substitution
	for(i = row_dim - 1; i >= 0; i--)
	{
		get_cddrsmatrix_ij(&ilu_ii, ilu, i, i);
		subst_cddvector_i(&ret_i, ret, i);
		if(rcdd_cmp_abs_ui(&ilu_ii, 0UL) != 0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				get_cddrsmatrix_ij(&ilu_ji, ilu, j, i);
				if(rcdd_cmp_abs_ui(&ilu_ji, 0UL) != 0)
				{
					subst_cddvector_i(&ret_j, ret, j);
					//ret_i = ret_i - ilu_ji * ret_j;
					rcdd_mul(&ctmp, &ret_j, &ilu_ji);
					rcdd_sub(&ret_i, &ret_i, &ctmp);
					//set_cdvector_i(ret, j, ret_j);
				}
			}
			set_cddvector_i(ret, i, &ret_i);
		}
	}
}

// iLU0s_solve: x^T * conj(iLU) = b^T
void solve_iLU0s_cddrsmatrix(CDDVector ret, CDDRSMatrix ilu, CDDVector b)
{
	long int i, j, row_dim, col_dim;
	cddfloat ret_i, ret_j, ctmp, ilu_ii, ilu_ji, conj_ilu_ii, conj_ilu_ji;

	row_dim = ilu->re->row_dim;
	col_dim = ilu->re->col_dim;

	// ret := b
	subst_cddvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		get_cddrsmatrix_ij(&ilu_ii, ilu, i, i);
		subst_cddvector_i(&ret_i, ret, i);
		if(rcdd_cmp_abs_ui(&ilu_ii, 0UL) != 0)
		{
			rcdd_conj(&conj_ilu_ii, &ilu_ii);
			for(j = 0; j < i; j++)
			{
				get_cddrsmatrix_ij(&ilu_ji, ilu, j, i);
				if(rcdd_cmp_abs_ui(&ilu_ji, 0UL) != 0)
				{
					rcdd_conj(&conj_ilu_ji, &ilu_ji);
					subst_cddvector_i(&ret_j, ret, j);
					//ret_i = ret_i - ilu_ji * ret_j;
					//rcdd_mul(&ctmp, &ret_j, &ilu_ji);
					rcdd_mul(&ctmp, &ret_j, &conj_ilu_ji);
					rcdd_sub(&ret_i, &ret_i, &ctmp);
					//set_cdvector_i(ret, i, ret_i);
				}
			}
			//ret_i /= ilu_ii;
			//rcdd_div(&ret_i, &ret_i, &ilu_ii);
			rcdd_div(&ret_i, &ret_i, &conj_ilu_ii);
			set_cddvector_i(ret, i, &ret_i);
		}
	}

	// Backward substitution
	for(i = row_dim - 1; i >= 0; i--)
	{
		get_cddrsmatrix_ij(&ilu_ii, ilu, i, i);
		subst_cddvector_i(&ret_i, ret, i);
		if(rcdd_cmp_abs_ui(&ilu_ii, 0UL) != 0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				get_cddrsmatrix_ij(&ilu_ji, ilu, j, i);
				if(rcdd_cmp_abs_ui(&ilu_ji, 0UL) != 0)
				{
					rcdd_conj(&conj_ilu_ji, &ilu_ji);
					subst_cddvector_i(&ret_j, ret, j);
					//ret_i = ret_i - ilu_ji * ret_j;
					//rcdd_mul(&ctmp, &ret_j, &ilu_ji);
					rcdd_mul(&ctmp, &ret_j, &conj_ilu_ji);
					rcdd_sub(&ret_i, &ret_i, &ctmp);
					//set_cdvector_i(ret, j, ret_j);
				}
			}
			set_cddvector_i(ret, i, &ret_i);
		}
	}
}

// iLU0_solve: iLU * x = b
void solve_iLU0_cdrsmatrix_cddvec(CDDVector ret, CDRSMatrix ilu, CDDVector b)
{
	long int i, j, row_dim, col_dim;
	double _Complex ilu_ii, ilu_ij;
	cddfloat ret_i, ret_j, ctmp;

	row_dim = ilu->re->row_dim;
	col_dim = ilu->re->col_dim;

	// ret := b
	subst_cddvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		subst_cddvector_i(&ret_i, ret, i);
		if(cabs(ilu_ii) != 0.0)
		{
			for(j = 0; j < i; j++)
			{
				ilu_ij = get_cdrsmatrix_ij(ilu, i, j);
				if(cabs(ilu_ij) != 0.0)
				{
					subst_cddvector_i(&ret_j, ret, j);
					//ret_j = ret_j - ilu_ji * ret_i;
					rcdd_mul_cd(&ctmp, &ret_j, ilu_ij);
					rcdd_sub(&ret_i, &ret_i, &ctmp);
					//set_cddvector_i(ret, j, &ret_j);
					//printf("f: %ld, %ld\n", i, j);
				}
			}
			set_cddvector_i(ret, i, &ret_i);
		}
	}

	// Backward substitution
	for(i = (row_dim - 1); i >= 0; i--)
	{
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		subst_cddvector_i(&ret_i, ret, i);
		if(cabs(ilu_ii) != 0.0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				ilu_ij = get_cdrsmatrix_ij(ilu, i, j);
				if(cabs(ilu_ij) != 0.0)
				{
					subst_cddvector_i(&ret_j, ret, j);
					//ret_i = ret_i - ilu_ij * ret_j;
					//rcdd_mul(&ctmp, &ilu_ij, &ret_j);
					rcdd_mul_cd(&ctmp, &ret_j, ilu_ij);
					rcdd_sub(&ret_i, &ret_i, &ctmp);
					//set_cddvector_i(ret, i, &ret_i);
					//printf("b: %ld, %ld\n", i, j);
				}
			}
			//ret_i /= ilu_ii;
			//rcdd_div(&ret_i, &ret_i, &ilu_ii);
			rcdd_div_cd(&ret_i, &ret_i, ilu_ii);
			set_cddvector_i(ret, i, &ret_i);
		}
	}
}

// iLU0t_solve: x^T * iLU = b^T
void solve_iLU0t_cdrsmatrix_cddvec(CDDVector ret, CDRSMatrix ilu, CDDVector b)
{
	long int i, j, row_dim, col_dim;
	double _Complex ilu_ii, ilu_ji;
	cddfloat ret_i, ret_j, ctmp;

	row_dim = ilu->re->row_dim;
	col_dim = ilu->re->col_dim;

	// ret := b
	subst_cddvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		subst_cddvector_i(&ret_i, ret, i);
		if(cabs(ilu_ii) != 0.0)
		{
			for(j = 0; j < i; j++)
			{
				ilu_ji = get_cdrsmatrix_ij(ilu, j, i);
				if(cabs(ilu_ji) != 0.0)
				{
					subst_cddvector_i(&ret_j, ret, j);
					//ret_i = ret_i - ilu_ji * ret_j;
					rcdd_mul_cd(&ctmp, &ret_j, ilu_ji);
					rcdd_sub(&ret_i, &ret_i, &ctmp);
					//set_cdvector_i(ret, i, ret_i);
				}
			}
			//ret_i /= ilu_ii;
			rcdd_div_cd(&ret_i, &ret_i, ilu_ii);
			set_cddvector_i(ret, i, &ret_i);
		}
	}

	// Backward substitution
	for(i = row_dim - 1; i >= 0; i--)
	{
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		subst_cddvector_i(&ret_i, ret, i);
		if(cabs(ilu_ii) != 0.0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				ilu_ji = get_cdrsmatrix_ij(ilu, j, i);
				if(cabs(ilu_ji) != 0.0)
				{
					subst_cddvector_i(&ret_j, ret, j);
					//ret_i = ret_i - ilu_ji * ret_j;
					rcdd_mul_cd(&ctmp, &ret_j, ilu_ji);
					rcdd_sub(&ret_i, &ret_i, &ctmp);
					//set_cdvector_i(ret, j, ret_j);
				}
			}
			set_cddvector_i(ret, i, &ret_i);
		}
	}
}

// iLU0s_solve: x^T * conj(iLU) = b^T
void solve_iLU0s_cdrsmatrix_cddvec(CDDVector ret, CDRSMatrix ilu, CDDVector b)
{
	long int i, j, row_dim, col_dim;
	double _Complex ilu_ii, ilu_ji, conj_ilu_ii, conj_ilu_ji;
	cddfloat ret_i, ret_j, ctmp;

	row_dim = ilu->re->row_dim;
	col_dim = ilu->re->col_dim;

	// ret := b
	subst_cddvector(ret, b);

	// Forward substitution
	for(i = 0; i < row_dim; i++)
	{
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		subst_cddvector_i(&ret_i, ret, i);
		if(cabs(ilu_ii) != 0.0)
		{
 			//get_cdrsmatrix_ij(ilu, i, i);
 			conj_ilu_ii = conj(ilu_ii);
			for(j = 0; j < i; j++)
			{
				ilu_ji = get_cdrsmatrix_ij(ilu, j, i);
				if(cabs(ilu_ji) != 0.0)
				{
		 			conj_ilu_ji = conj(ilu_ji);
					subst_cddvector_i(&ret_j, ret, j);
					//ret_i = ret_i - ilu_ji * ret_j;
					//rcdd_mul_cd(&ctmp, &ret_j, ilu_ji);
					rcdd_mul_cd(&ctmp, &ret_j, conj_ilu_ji);
					rcdd_sub(&ret_i, &ret_i, &ctmp);
					//set_cdvector_i(ret, i, ret_i);
				}
			}
			//ret_i /= ilu_ii;
			//rcdd_div_cd(&ret_i, &ret_i, ilu_ii);
			rcdd_div_cd(&ret_i, &ret_i, conj_ilu_ii);
			set_cddvector_i(ret, i, &ret_i);
		}
	}

	// Backward substitution
	for(i = row_dim - 1; i >= 0; i--)
	{
		ilu_ii = get_cdrsmatrix_ij(ilu, i, i);
		subst_cddvector_i(&ret_i, ret, i);
		if(cabs(ilu_ii) != 0.0)
		{
			for(j = (i + 1); j < col_dim; j++)
			{
				ilu_ji = get_cdrsmatrix_ij(ilu, j, i);
				if(cabs(ilu_ji) != 0.0)
				{
					conj_ilu_ji = conj(ilu_ji);
					subst_cddvector_i(&ret_j, ret, j);
					//ret_i = ret_i - ilu_ji * ret_j;
					//rcdd_mul_cd(&ctmp, &ret_j, ilu_ji);
					rcdd_mul_cd(&ctmp, &ret_j, conj_ilu_ji);
					rcdd_sub(&ret_i, &ret_i, &ctmp);
					//set_cdvector_i(ret, j, ret_j);
				}
			}
			set_cddvector_i(ret, i, &ret_i);
		}
	}
}

