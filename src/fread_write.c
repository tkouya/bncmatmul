/********************************************************************************/
/* fread_write.c: Read and Write files to store *matrix, *vector etc.           */
/*                                                                              */
/* Copyright (c) 2005-2023 Tomonori Kouya                                       */
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
#include <string.h>

//#include "bnc.h"
#include "dlinear.h"
#include "cddlinear.h"
#include "ctdlinear.h"
#include "cqdlinear.h"
#include "mpflinear.h"
//#include "clinear.h"
#include "cdlinear.h"
#include "cmpflinear.h"


#define BNC_MAX_READ_DDIGITS 8291
#define BNC_MAX_BUF_SIZE 65536
#define BNC_MAX_DATUM_PER_LINE 10

// ref) https://programming-place.net/ppp/contents/c/appendix/reference/fgets.html
// read one line and divide it to string array
// Input : "23, 55, 3.0e-10, -5.99e+1"
// Output: string[] = ["23", "55, "3.0e-10", "-5.99e+1"], ret = 4
int bnc_readline(char *ret_string_array[BNC_MAX_DATUM_PER_LINE], int max_num_ret_string_array, const char *fname)
{
	int num_strings = 0;
	char buf[BNC_MAX_BUF_SIZE]; // one line
	FILE *fp;
	char *line_ptr;
	int i, j, max_i, end_flag = 0;

	fp = fopen(fname, "r+t");
	if(fp == NULL)
	{
		fprintf(stderr, "bnc_readline: cannot open %s\n", fname);
		return 0;
	}

	do{
		if(fgets(buf, sizeof(buf), fp) != NULL)
		{
			// replace \n to \0
			line_ptr = strchr(buf, '\n');
			if(line_ptr != NULL)
				*line_ptr = '\0';
		}
		else
			break;

		// search white space
		end_flag = 0;
		max_i = (BNC_MAX_DATUM_PER_LINE <  max_num_ret_string_array) ?  BNC_MAX_DATUM_PER_LINE : max_num_ret_string_array;
		for(i = 0; i < max_i; i++)
		{
			line_ptr = &buf[0];

			// skip spaces
			while(*line_ptr++ != ' ')
			{
				if(*line_ptr == '\0')
				{
					end_flag = 1;
					break;
				}
			}
			if(end_flag == 1)
				break;

			// read strings
			for(j = 0; j < BNC_MAX_READ_DDIGITS; j++)
			{
				ret_string_array[i][j] = *line_ptr++;
				if((*line_ptr == ',') || (*line_ptr == '\t') || (*line_ptr == ' ') || (*line_ptr == ':') || (*line_ptr == ';'))
				{
					ret_string_array[i][j] = '\0';
					break;
				}

				if(*line_ptr == '\0')
				{
					ret_string_array[i][j] = '\0';
					end_flag = 1;
					break;
				}
			}
			if(end_flag == 1)
				break;
		}

	} while(!feof(fp));

	return end_flag;
}

void fread_dmatrix(FILE *fp, DMatrix mat)
{
	long int i, row_index, col_index;
	double tmp;
	int ret;

	for(i = 0; i < mat->row_dim * mat->col_dim; i++)
	{
		//fscanf(fp, "%ld, %ld, %lf", &row_index, &col_index, &tmp);
		ret = fscanf(fp, "%ld, %ld, %lf", &row_index, &col_index, &tmp);
		set_dmatrix_ij(mat, row_index, col_index, tmp);
		if(feof(fp) != 0)
			break;
	}
}

void fread_dmatrix_fname(const char *fname, DMatrix mat)
{
	FILE *fp;

	if(fname == NULL)
	{
		fprintf(stderr, "file name is undefined!\n");
		return;
	}

	fp = fopen(fname, "r");
	if(fp == NULL)
	{
		fprintf(stderr, "%s cannot be opened!\n", fname);
		return;
	}

	fread_dmatrix(fp, mat);

	fclose(fp);
}

void fwrite_dmatrix(FILE *fp, DMatrix mat)
{
	long int i, j;

	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
			fprintf(fp, "%ld, %ld, %lf\n", i, j, get_dmatrix_ij(mat, i, j));
	}
}

void fwrite_dmatrix_fname(const char *fname, DMatrix mat)
{
	FILE *fp;

	if(fname == NULL)
	{
		fprintf(stderr, "file name is undefined!\n");
		return;
	}

	fp = fopen(fname, "w");
	if(fp == NULL)
	{
		fprintf(stderr, "%s cannot be opened!\n", fname);
		return;
	}

	fwrite_dmatrix(fp, mat);

	fclose(fp);
}

/*
void fread_dpolycoef(FILE *fp, DPoly p, long int maxdeg)
{
	long int i, index;
	double tmp;

	for(i = 0; i <= maxdeg; i++)
	{
		fscanf(fp, "%ld, %lf", &index, &tmp);
		set_dpoly_i(p, index, tmp);
	}
}

void fread_dpolycoef_fname(const char *fname, DPoly p, long int maxdeg)
{
	FILE *fp;

	if(fname == NULL)
	{
		fprintf(stderr, "file name is undefined!\n");
		return;
	}

	fp = fopen(fname, "r");
	if(fp == NULL)
	{
		fprintf(stderr, "%s cannot be opened!\n", fname);
		return;
	}

	fread_dpolycoef(fp, p, maxdeg);

	fclose(fp);
}
*/

void fread_dvector(FILE *fp, DVector vec)
{
	long int i, index;
	double tmp;
	int ret;

	for(i = 0; i < vec->dim; i++)
	{
		ret = fscanf(fp, "%ld, %lf", &index, &tmp);
		set_dvector_i(vec, index, tmp);
	}
}

void fread_dvector_fname(const char *fname, DVector vec)
{
	FILE *fp;

	if(fname == NULL)
	{
		fprintf(stderr, "file name is undefined!\n");
		return;
	}

	fp = fopen(fname, "r");
	if(fp == NULL)
	{
		fprintf(stderr, "%s cannot be opened!\n", fname);
		return;
	}

	fread_dvector(fp, vec);

	fclose(fp);
}

void fwrite_dvector(FILE *fp, DVector vec)
{
	long int i;

	for(i = 0; i < vec->dim; i++)
		fprintf(fp, "%ld, %25.17e\n", i, get_dvector_i(vec, i));
}

void fwrite_dvector_fname(const char *fname, DVector vec)
{
	FILE *fp;

	if(fname == NULL)
	{
		fprintf(stderr, "file name is undefined!\n");
		return;
	}

	fp = fopen(fname, "w");
	if(fp == NULL)
	{
		fprintf(stderr, "%s cannot be opened!\n", fname);
		return;
	}

	fwrite_dvector(fp, vec);

	fclose(fp);
}

// read and write MPFVector, MPFMatrix
#ifdef USE_GMP
void fread_mpfmatrix(FILE *fp, MPFMatrix mat)
{
	long int i, row_index, col_index;
	char tmpsrc[65536]; /* maximum degits */
	mpf_t tmp;
	int ret;

	mpf_init2(tmp, mat->prec);

	for(i = 0; i < mat->row_dim * mat->col_dim; i++)
	{
		ret = fscanf(fp, "%ld, %ld, %s", &row_index, &col_index, tmpsrc);
		mpf_set_str(tmp, tmpsrc, 10);
		set_mpfmatrix_ij(mat, row_index, col_index, tmp);
		if(feof(fp) != 0)
			break;
	}

	mpf_clear(tmp);
}

void fread_mpfmatrix_fname(const char *fname, MPFMatrix mat)
{
	FILE *fp;

	if(fname == NULL)
	{
		fprintf(stderr, "file name is undefined!\n");
		return;
	}

	fp = fopen(fname, "r");
	if(fp == NULL)
	{
		fprintf(stderr, "%s cannot be opened!\n", fname);
		return;
	}

	fread_mpfmatrix(fp, mat);

	fclose(fp);
}

void fwrite_mpfmatrix(FILE *fp, MPFMatrix mat)
{
	long int i, j;
	char tmpsrc[65536]; /* maximum digits */
	mpf_t tmp;

	mpf_init2(tmp, mat->prec);

	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
		{
			if(mpf_cmp_ui(get_mpfmatrix_ij(mat, i, j), 0UL) != 0)
			{
				fprintf(fp, "%ld, %ld, ", i, j);
				mpf_out_str(fp, 10, 0, get_mpfmatrix_ij(mat, i, j));
				fprintf(fp, "\n");
			}
		}
	}

	mpf_clear(tmp);
}

void fwrite_mpfmatrix_fname(const char *fname, MPFMatrix mat)
{
	FILE *fp;

	if(fname == NULL)
	{
		fprintf(stderr, "file name is undefined!\n");
		return;
	}

	fp = fopen(fname, "w");
	if(fp == NULL)
	{
		fprintf(stderr, "%s cannot be opened!\n", fname);
		return;
	}

	fwrite_mpfmatrix(fp, mat);

	fclose(fp);
}
#if 0
void fread_mpfpolycoef(FILE *fp, MPFPoly p, long int maxdeg)
{
	long int i, index;
	char tmpsrc[65536]; /* maximum degits */
	mpf_t tmp;

	mpf_init2(tmp, p->prec);

	for(i = 0; i <= maxdeg; i++)
	{
		fscanf(fp, "%ld, %s", &index, tmpsrc);
//		printf("%d: %s\n", index, tmpsrc);
		mpf_set_str(tmp, tmpsrc, 10);
		set_mpfpoly_i(p, index, tmp);
	}

	mpf_clear(tmp);
}

void fread_mpfpolycoef_fname(const char *fname, MPFPoly p, long int maxdeg)
{
	FILE *fp;

	if(fname == NULL)
	{
		fprintf(stderr, "file name is undefined!\n");
		return;
	}

	fp = fopen(fname, "r");
	if(fp == NULL)
	{
		fprintf(stderr, "%s cannot be opened!\n", fname);
		return;
	}

	fread_mpfpolycoef(fp, p, maxdeg);

	fclose(fp);
}
#endif // 0

void fread_mpfvector(FILE *fp, MPFVector vec)
{
	long int i, index;
	char tmpsrc[65536]; /* maximum digits */
	mpf_t tmp;
	int ret;

	mpf_init2(tmp, vec->prec);

	for(i = 0; i < vec->dim; i++)
	{
		ret = fscanf(fp, "%ld, %s", &index, tmpsrc);
		mpf_set_str(tmp, tmpsrc, 10);
		set_mpfvector_i(vec, index, tmp);
	}

	mpf_clear(tmp);
}

void fread_mpfvector_fname(const char *fname, MPFVector vec)
{
	FILE *fp;

	if(fname == NULL)
	{
		fprintf(stderr, "file name is undefined!\n");
		return;
	}

	fp = fopen(fname, "r");
	if(fp == NULL)
	{
		fprintf(stderr, "%s cannot be opened!\n", fname);
		return;
	}

	fread_mpfvector(fp, vec);

	fclose(fp);
}

void fwrite_mpfvector(FILE *fp, MPFVector vec)
{
	long int i;

	for(i = 0; i < vec->dim; i++)
	{
		fprintf(fp, "%ld, ", i);
		mpf_out_str(fp, 10, 0, get_mpfvector_i(vec, i));
		fprintf(fp, "\n");
	}
}

void fwrite_mpfvector_fname(const char *fname, MPFVector vec)
{
	FILE *fp;

	if(fname == NULL)
	{
		fprintf(stderr, "file name is undefined!\n");
		return;
	}

	fp = fopen(fname, "w");
	if(fp == NULL)
	{
		fprintf(stderr, "%s cannot be opened!\n", fname);
		return;
	}

	fwrite_mpfvector(fp, vec);

	fclose(fp);
}
#endif // USE_GMP

// read and write CMPFVector, CMPFMatrix
#ifdef USE_GMP
void fread_cmpfmatrix(FILE *fp, CMPFMatrix mat)
{
	long int i, row_index, col_index;
	char tmpsrc[2][65536]; /* maximum digits */
	mpf_t real, imag;
	mpc_t ctmp;
	int ret;

	mpf_init2(real, mat->prec);
	mpf_init2(imag, mat->prec);
	mpc_init2(ctmp, mat->prec);

	// i, j, real, imag
	for(i = 0; i < mat->row_dim * mat->col_dim; i++)
	{
		ret = fscanf(fp, "%ld, %ld, %s, %s", &row_index, &col_index, tmpsrc[0], tmpsrc[1]);
		mpf_set_str(real, tmpsrc[0], 10);
		mpf_set_str(imag, tmpsrc[1], 10);
		mpc_set_fr_fr(ctmp, real, imag, get_bnc_default_rounding_mode());
		set_cmpfmatrix_ij(mat, row_index, col_index, ctmp);
		if(feof(fp) != 0)
			break;
	}

	mpf_clear(real);
	mpf_clear(imag);
	mpc_clear(ctmp);
}

void fread_cmpfmatrix_fname(const char *fname, CMPFMatrix mat)
{
	FILE *fp;

	if(fname == NULL)
	{
		fprintf(stderr, "file name is undefined!\n");
		return;
	}

	fp = fopen(fname, "r");
	if(fp == NULL)
	{
		fprintf(stderr, "%s cannot be opened!\n", fname);
		return;
	}

	fread_cmpfmatrix(fp, mat);

	fclose(fp);
}

void fwrite_cmpfmatrix(FILE *fp, CMPFMatrix mat)
{
	long int i, j;

	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
		{
			if(mpc_cmp_si_si(get_cmpfmatrix_ij(mat, i, j), 0L, 0L) != 0)
			{
				fprintf(fp, "%ld, %ld, ", i, j);
				mpf_out_str(fp, 10, 0, mpc_realref(get_cmpfmatrix_ij(mat, i, j)));
				fprintf(fp, ", ");
				mpf_out_str(fp, 10, 0, mpc_imagref(get_cmpfmatrix_ij(mat, i, j)));
				fprintf(fp, "\n");
			}
		}
	}
}

void fwrite_cmpfmatrix_fname(const char *fname, CMPFMatrix mat)
{
	FILE *fp;

	if(fname == NULL)
	{
		fprintf(stderr, "file name is undefined!\n");
		return;
	}

	fp = fopen(fname, "w");
	if(fp == NULL)
	{
		fprintf(stderr, "%s cannot be opened!\n", fname);
		return;
	}

	fwrite_cmpfmatrix(fp, mat);

	fclose(fp);
}

void fread_cmpfvector(FILE *fp, CMPFVector vec)
{
	long int i, index;
	char tmpsrc[2][65536]; /* maximum digits */
	mpf_t real, imag;
	mpc_t ctmp;
	int ret;

	mpf_init2(real, vec->prec);
	mpf_init2(imag, vec->prec);
	mpc_init2(ctmp, vec->prec);

	for(i = 0; i < vec->dim; i++)
	{
		ret = fscanf(fp, "%ld, %s, %s", &index, tmpsrc[0], tmpsrc[1]);
		mpf_set_str(real, tmpsrc[0], 10);
		mpf_set_str(imag, tmpsrc[1], 10);
		mpc_set_fr_fr(ctmp, real, imag, get_bnc_default_rounding_mode());
		set_cmpfvector_i(vec, index, ctmp);
	}

	mpf_clear(real);
	mpf_clear(imag);
	mpc_clear(ctmp);
}

void fread_cmpfvector_fname(const char *fname, CMPFVector vec)
{
	FILE *fp;

	if(fname == NULL)
	{
		fprintf(stderr, "file name is undefined!\n");
		return;
	}

	fp = fopen(fname, "r");
	if(fp == NULL)
	{
		fprintf(stderr, "%s cannot be opened!\n", fname);
		return;
	}

	fread_cmpfvector(fp, vec);

	fclose(fp);
}

void fwrite_cmpfvector(FILE *fp, CMPFVector vec)
{
	long int i;

	for(i = 0; i < vec->dim; i++)
	{
		fprintf(fp, "%ld, ", i);
		mpf_out_str(fp, 10, 0, mpc_realref(get_cmpfvector_i(vec, i)));
		fprintf(fp, ", ");
		mpf_out_str(fp, 10, 0, mpc_imagref(get_cmpfvector_i(vec, i)));
		fprintf(fp, "\n");
	}
}

void fwrite_cmpfvector_fname(const char *fname, CMPFVector vec)
{
	FILE *fp;

	if(fname == NULL)
	{
		fprintf(stderr, "file name is undefined!\n");
		return;
	}

	fp = fopen(fname, "w");
	if(fp == NULL)
	{
		fprintf(stderr, "%s cannot be opened!\n", fname);
		return;
	}

	fwrite_cmpfvector(fp, vec);

	fclose(fp);
}
#endif // USE_GMP

// DD
// 2021-07-15(Wed) T.Kouya
// read problem from file
void read_test_linear_eq_dd(DDMatrix A, DDVector true_x, DDVector b, long int dim, const char *fname_A, const char *fname_true_x, const char *fname_b)
{
	FILE *fp;
	int i, j, ret_scan;
	char str_num[8192]; // up to 8192 dec.digits
	double tmp[DDSIZE];

	// A
	fp = fopen(fname_A, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open \"%s\" to read matrix A!\n", fname_A);
	}
	else{
		//printf("%s was successfully read...\n", fname_A);
		do {
			ret_scan = fscanf(fp, "%d, %d, %s", &i, &j, str_num);
			//A[i * dim + j] = (T)str_num;
			rdd_set_str(tmp, str_num);
			set_ddmatrix_ij(A, i, j, tmp);
			//printf("%s -- \n", str_num); rdd_out_str(tmp); printf("\n");
		} while(!feof(fp));

		fclose(fp);
	}

	// true_x
	fp = fopen(fname_true_x, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open \"%s\" to read true x!\n", fname_true_x);
	}
	else{
		//printf("%s was successfully read...\n", fname_true_x);
		do {
			ret_scan = fscanf(fp, "%d, %s", &i, str_num);
			//true_x[i] = (T)str_num;
			rdd_set_str(tmp, str_num);
			set_ddvector_i(true_x, i, tmp);
		} while(!feof(fp));

		fclose(fp);
	}

	// b
	fp = fopen(fname_b, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open \"%s\" to read true x!\n", fname_b);
	}
	else{
		//printf("%s was successfully read...\n", fname_b);
		do {
			ret_scan = fscanf(fp, "%d, %s", &i, str_num);
			//b[i] = (T)str_num;
			rdd_set_str(tmp, str_num);
			set_ddvector_i(b, i, tmp);
		} while(!feof(fp));

		fclose(fp);
	}
}

// 2021-07-15(Wed) T.Kouya
// read problem from file
void read_test_linear_eq_td(TDMatrix A, TDVector true_x, TDVector b, long int dim, const char *fname_A, const char *fname_true_x, const char *fname_b)
{
	FILE *fp;
	int i, j, ret_scan;
	char str_num[8192]; // up to 8192 dec.digits
	double tmp[TDSIZE];

	// A
	fp = fopen(fname_A, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open \"%s\" to read matrix A!\n", fname_A);
	}
	else{
		//printf("%s was successfully read...\n", fname_A);
		do {
			ret_scan = fscanf(fp, "%d, %d, %s", &i, &j, str_num);
			//A[i * dim + j] = (T)str_num;
			rtd_set_str(tmp, str_num);
			set_tdmatrix_ij(A, i, j, tmp);
			//printf("%s -- \n", str_num); rdd_out_str(tmp); printf("\n");
		} while(!feof(fp));

		fclose(fp);
	}

	// true_x
	fp = fopen(fname_true_x, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open \"%s\" to read true x!\n", fname_true_x);
	}
	else{
		//printf("%s was successfully read...\n", fname_true_x);
		do {
			ret_scan = fscanf(fp, "%d, %s", &i, str_num);
			//true_x[i] = (T)str_num;
			rtd_set_str(tmp, str_num);
			set_tdvector_i(true_x, i, tmp);
		} while(!feof(fp));

		fclose(fp);
	}

	// b
	fp = fopen(fname_b, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open \"%s\" to read true x!\n", fname_b);
	}
	else{
		//printf("%s was successfully read...\n", fname_b);
		do {
			ret_scan = fscanf(fp, "%d, %s", &i, str_num);
			//b[i] = (T)str_num;
			rtd_set_str(tmp, str_num);
			set_tdvector_i(b, i, tmp);
		} while(!feof(fp));

		fclose(fp);
	}
}

// 2021-07-15(Wed) T.Kouya
// read problem from file
void read_test_linear_eq_qd(QDMatrix A, QDVector true_x, QDVector b, long int dim, const char *fname_A, const char *fname_true_x, const char *fname_b)
{
	FILE *fp;
	int i, j, ret_scan;
	char str_num[8192]; // up to 8192 dec.digits
	double tmp[QDSIZE];

	// A
	fp = fopen(fname_A, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open \"%s\" to read matrix A!\n", fname_A);
	}
	else{
		//printf("%s was successfully read...\n", fname_A);
		do {
			ret_scan = fscanf(fp, "%d, %d, %s", &i, &j, str_num);
			//A[i * dim + j] = (T)str_num;
			rqd_set_str(tmp, str_num);
			set_qdmatrix_ij(A, i, j, tmp);
			//printf("%s -- \n", str_num); rdd_out_str(tmp); printf("\n");
		} while(!feof(fp));

		fclose(fp);
	}

	// true_x
	fp = fopen(fname_true_x, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open \"%s\" to read true x!\n", fname_true_x);
	}
	else{
		//printf("%s was successfully read...\n", fname_true_x);
		do {
			ret_scan = fscanf(fp, "%d, %s", &i, str_num);
			//true_x[i] = (T)str_num;
			//rqd_set_str(tmp, str_num);
			set_qdvector_i_str(true_x, i, str_num);
		} while(!feof(fp));

		fclose(fp);
	}

	// b
	fp = fopen(fname_b, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open \"%s\" to read true x!\n", fname_b);
	}
	else{
		//printf("%s was successfully read...\n", fname_b);
		do {
			ret_scan = fscanf(fp, "%d, %s", &i, str_num);
			//b[i] = (T)str_num;
			//rqd_set_str(tmp, str_num);
			set_qdvector_i_str(b, i, str_num);
		} while(!feof(fp));

		fclose(fp);
	}
}

#ifdef USE_GMP
// 2021-07-15(Wed) T.Kouya
// read problem from file
void read_test_linear_eq(MPFMatrix A, MPFVector true_x, MPFVector b, long int dim, const char *fname_A, const char *fname_true_x, const char *fname_b)
{
	FILE *fp;
	int i, j, ret_scan;
	char str_num[8192]; // up to 8192 dec.digits

	// A
	fp = fopen(fname_A, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open %s to read matrix A!\n", fname_A);
	}
	else{
		//printf("%s was successfully read...\n", fname_A);
		do {
			ret_scan = fscanf(fp, "%d, %d, %s", &i, &j, str_num);
			//A[i * dim + j] = (T)str_num;
			set_mpfmatrix_ij_str(A, i, j, str_num, 10);
		} while(!feof(fp));

		fclose(fp);
	}

	// true_x
	fp = fopen(fname_true_x, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open %s to read true x!\n", fname_true_x);
	}
	else{
		//printf("%s was successfully read...\n", fname_true_x);
		do {
			ret_scan = fscanf(fp, "%d, %s", &i, str_num);
			//true_x[i] = (T)str_num;
			set_mpfvector_i_str(true_x, i, str_num, 10);
		} while(!feof(fp));

		fclose(fp);
	}

	// b
	fp = fopen(fname_b, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open %s to read b!\n", fname_b);
	}
	else{
		//printf("%s was successfully read...\n", fname_b);
		do {
			ret_scan = fscanf(fp, "%d, %s", &i, str_num);
			//b[i] = (T)str_num;
			set_mpfvector_i_str(b, i, str_num, 10);
		} while(!feof(fp));

		fclose(fp);
	}
}

// 2023-04-07(Fri) T.Kouya
// write problem into file
void write_test_linear_eq(MPFMatrix A, MPFVector true_x, MPFVector b, long int dim, const char *fname_A, const char *fname_true_x, const char *fname_b)
{
	FILE *fp;
	int i, j, ret_scan;
	char str_num[8192]; // up to 8192 dec.digits

	// A
	fp = fopen(fname_A, "w+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open %s to write matrix A!\n", fname_A);
	}
	else{
		//printf("%s was successfully read...\n", fname_A);
		fwrite_mpfmatrix(fp, A);

		fclose(fp);
	}

	// true_x
	fp = fopen(fname_true_x, "w+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open %s to write true x!\n", fname_true_x);
	}
	else{
		//printf("%s was successfully read...\n", fname_true_x);
		fwrite_mpfvector(fp, true_x);

		fclose(fp);
	}

	// b
	fp = fopen(fname_b, "w+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open %s to writeb!\n", fname_b);
	}
	else{
		//printf("%s was successfully read...\n", fname_b);
		fwrite_mpfvector(fp, b);

		fclose(fp);
	}
}

// 2023-04-07(Fri) T.Kouya
// read problem from file
void read_test_linear_eq_c(CMPFMatrix A, CMPFVector true_x, CMPFVector b, long int dim, const char *fname_A, const char *fname_true_x, const char *fname_b)
{
	FILE *fp;
	int i, j, ret_scan;
	char buf[BNC_MAX_BUF_SIZE], str_num0[BNC_MAX_READ_DDIGITS], str_num1[BNC_MAX_READ_DDIGITS]; // up to 8192 dec.digits
	mpf_t real, imag;
	mpc_t ctmp;
	char *ret;

	// A
	mpf_init2(real, A->prec);mpf_init2(imag, A->prec);mpc_init2(ctmp, A->prec);

	fp = fopen(fname_A, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open %s to read matrix A!\n", fname_A);
	}
	else{
		#ifdef DEBUG
		printf("DEBUG\n");
		#endif // DEBUG
		//printf("%s was successfully read...\n", fname_A);
		do {
			ret = fgets(buf, BNC_MAX_BUF_SIZE, fp);
			//ret_scan = fscanf(fp, "%d, %d, %s, %s", &i, &j, str_num0, str_num1);
			//ret_scan = sscanf(buf, "%d, %d, %s, %s", &i, &j, str_num0, str_num1);
			ret_scan = sscanf(buf, "%d, %d, %[^,], %[^,]", &i, &j, str_num0, str_num1);
			//printf("buf = %s", buf);
			//printf("i, j, str_num[0], str_num[1] = (%d, %d) %s, %s\n", i, j, str_num0, str_num1);
			//A[i * dim + j] = (T)str_num;
			mpf_set_str(real, str_num0, 10);
			mpf_set_str(imag, str_num1, 10);
			mpc_set_fr_fr(ctmp, real, imag, get_bnc_default_rounding_mode());
			set_cmpfmatrix_ij(A, i, j, ctmp);
		} while(!feof(fp));

		fclose(fp);
	}

	mpf_clear(real);mpf_clear(imag);mpc_clear(ctmp);

	// true_x
	mpf_init2(real, true_x->prec);mpf_init2(imag, true_x->prec);mpc_init2(ctmp, true_x->prec);

	fp = fopen(fname_true_x, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open %s to read true x!\n", fname_true_x);
	}
	else{
		//printf("%s was successfully read...\n", fname_true_x);
		do {
			//ret_scan = fscanf(fp, "%d, %s, %s", &i, str_num0, str_num1);
			ret = fgets(buf, BNC_MAX_BUF_SIZE, fp);
			ret_scan = sscanf(buf, "%d, %[^,], %[^,]", &i, str_num0, str_num1);
			//true_x[i] = (T)str_num;
			mpf_set_str(real, str_num0, 10);
			mpf_set_str(imag, str_num1, 10);
			mpc_set_fr_fr(ctmp, real, imag, get_bnc_default_rounding_mode());
			set_cmpfvector_i(true_x, i, ctmp);
		} while(!feof(fp));

		fclose(fp);
	}
	mpf_clear(real);mpf_clear(imag);mpc_clear(ctmp);

	// b
	mpf_init2(real, b->prec);mpf_init2(imag, b->prec);mpc_init2(ctmp, b->prec);

	fp = fopen(fname_b, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open %s to read b!\n", fname_b);
	}
	else{
		//printf("%s was successfully read...\n", fname_b);
		do {
			//ret_scan = fscanf(fp, "%d, %s, %s", &i, str_num0, str_num1);
			ret = fgets(buf, BNC_MAX_BUF_SIZE, fp);
			ret_scan = sscanf(buf, "%d, %[^,], %[^,]", &i, str_num0, str_num1);
			//b[i] = (T)str_num;
			mpf_set_str(real, str_num0, 10);
			mpf_set_str(imag, str_num1, 10);
			mpc_set_fr_fr(ctmp, real, imag, get_bnc_default_rounding_mode());
			set_cmpfvector_i(b, i, ctmp);
		} while(!feof(fp));

		fclose(fp);
	}
	mpf_clear(real);mpf_clear(imag);mpc_clear(ctmp);
}

// 2023-04-07(Fri) T.Kouya
// write problem into file
void write_test_linear_eq_c(CMPFMatrix A, CMPFVector true_x, CMPFVector b, long int dim, const char *fname_A, const char *fname_true_x, const char *fname_b)
{
	FILE *fp;
	int i, j, ret_scan;
	char str_num[BNC_MAX_READ_DDIGITS]; // up to 8192 dec.digits

	// A
	fp = fopen(fname_A, "w+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open %s to write matrix A!\n", fname_A);
	}
	else{
		//printf("%s was successfully read...\n", fname_A);
		fwrite_cmpfmatrix(fp, A);

		fclose(fp);
	}

	// true_x
	fp = fopen(fname_true_x, "w+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open %s to write true x!\n", fname_true_x);
	}
	else{
		//printf("%s was successfully read...\n", fname_true_x);
		fwrite_cmpfvector(fp, true_x);

		fclose(fp);
	}

	// b
	fp = fopen(fname_b, "w+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open %s to writeb!\n", fname_b);
	}
	else{
		//printf("%s was successfully read...\n", fname_b);
		fwrite_cmpfvector(fp, b);

		fclose(fp);
	}
}

#endif // USE_GMP

// 2023-12-14(Thu) T.Kouya
// read problem from file
void read_test_linear_eq_cdd(CDDMatrix A, CDDVector true_x, CDDVector b, long int dim, const char *fname_A, const char *fname_true_x, const char *fname_b)
{
	FILE *fp;
	int i, j, ret_scan;
	char buf[BNC_MAX_BUF_SIZE], str_num0[BNC_MAX_READ_DDIGITS], str_num1[BNC_MAX_READ_DDIGITS]; // up to 8192 dec.digits
	//mpf_t real, imag;
	//mpc_t ctmp;
	double real[DDSIZE], imag[DDSIZE];
	cddfloat ctmp;
	char *ret;

	// A
	//mpf_init2(real, A->prec);mpf_init2(imag, A->prec);mpc_init2(ctmp, A->prec);

	fp = fopen(fname_A, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open %s to read matrix A!\n", fname_A);
	}
	else{
		#ifdef DEBUG
		printf("DEBUG\n");
		#endif // DEBUG
		//printf("%s was successfully read...\n", fname_A);
		do {
			ret = fgets(buf, BNC_MAX_BUF_SIZE, fp);
			//ret_scan = fscanf(fp, "%d, %d, %s, %s", &i, &j, str_num0, str_num1);
			//ret_scan = sscanf(buf, "%d, %d, %s, %s", &i, &j, str_num0, str_num1);
			ret_scan = sscanf(buf, "%d, %d, %[^,], %[^,]", &i, &j, str_num0, str_num1);
			//printf("buf = %s", buf);
			//printf("i, j, str_num[0], str_num[1] = (%d, %d) %s, %s\n", i, j, str_num0, str_num1);
			//A[i * dim + j] = (T)str_num;
			//mpf_set_str(real, str_num0, 10);
			rdd_set_str(real, str_num0);
			//mpf_set_str(imag, str_num1, 10);
			rdd_set_str(imag, str_num1);
			//mpc_set_fr_fr(ctmp, real, imag, get_bnc_default_rounding_mode());
			rcdd_set_dd_dd(&ctmp, real, imag);
			//set_cmpfmatrix_ij(A, i, j, ctmp);
			set_cddmatrix_ij(A, i, j, &ctmp);
		} while(!feof(fp));

		fclose(fp);
	}

	//mpf_clear(real);mpf_clear(imag);mpc_clear(ctmp);

	// true_x
	//mpf_init2(real, true_x->prec);mpf_init2(imag, true_x->prec);mpc_init2(ctmp, true_x->prec);

	fp = fopen(fname_true_x, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open %s to read true x!\n", fname_true_x);
	}
	else{
		//printf("%s was successfully read...\n", fname_true_x);
		do {
			//ret_scan = fscanf(fp, "%d, %s, %s", &i, str_num0, str_num1);
			ret = fgets(buf, BNC_MAX_BUF_SIZE, fp);
			ret_scan = sscanf(buf, "%d, %[^,], %[^,]", &i, str_num0, str_num1);
			//true_x[i] = (T)str_num;
			//mpf_set_str(real, str_num0, 10);
			rdd_set_str(real, str_num0);
			//mpf_set_str(imag, str_num1, 10);
			rdd_set_str(imag, str_num1);
			//mpc_set_fr_fr(ctmp, real, imag, get_bnc_default_rounding_mode());
			rcdd_set_dd_dd(&ctmp, real, imag);
			//set_cmpfvector_i(true_x, i, ctmp);
			set_cddvector_i(true_x, i, &ctmp);
		} while(!feof(fp));

		fclose(fp);
	}
	//mpf_clear(real);mpf_clear(imag);mpc_clear(ctmp);

	// b
	//mpf_init2(real, b->prec);mpf_init2(imag, b->prec);mpc_init2(ctmp, b->prec);

	fp = fopen(fname_b, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open %s to read b!\n", fname_b);
	}
	else{
		//printf("%s was successfully read...\n", fname_b);
		do {
			//ret_scan = fscanf(fp, "%d, %s, %s", &i, str_num0, str_num1);
			ret = fgets(buf, BNC_MAX_BUF_SIZE, fp);
			ret_scan = sscanf(buf, "%d, %[^,], %[^,]", &i, str_num0, str_num1);
			//b[i] = (T)str_num;
			//mpf_set_str(real, str_num0, 10);
			rdd_set_str(real, str_num0);
			//mpf_set_str(imag, str_num1, 10);
			rdd_set_str(imag, str_num1);
			//mpc_set_fr_fr(ctmp, real, imag, get_bnc_default_rounding_mode());
			rcdd_set_dd_dd(&ctmp, real, imag);
			//set_cmpfvector_i(b, i, ctmp);
			set_cddvector_i(b, i, &ctmp);
		} while(!feof(fp));

		fclose(fp);
	}
	//mpf_clear(real);mpf_clear(imag);mpc_clear(ctmp);
}

// 2023-12-15(Fri) T.Kouya
// read problem from file
void read_test_linear_eq_ctd(CTDMatrix A, CTDVector true_x, CTDVector b, long int dim, const char *fname_A, const char *fname_true_x, const char *fname_b)
{
	FILE *fp;
	int i, j, ret_scan;
	char buf[BNC_MAX_BUF_SIZE], str_num0[BNC_MAX_READ_DDIGITS], str_num1[BNC_MAX_READ_DDIGITS]; // up to 8192 dec.digits
	//mpf_t real, imag;
	//mpc_t ctmp;
	double real[TDSIZE], imag[TDSIZE];
	ctdfloat ctmp;
	char *ret;

	// A
	//mpf_init2(real, A->prec);mpf_init2(imag, A->prec);mpc_init2(ctmp, A->prec);

	fp = fopen(fname_A, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open %s to read matrix A!\n", fname_A);
	}
	else{
		#ifdef DEBUG
		printf("DEBUG\n");
		#endif // DEBUG
		//printf("%s was successfully read...\n", fname_A);
		do {
			ret = fgets(buf, BNC_MAX_BUF_SIZE, fp);
			//ret_scan = fscanf(fp, "%d, %d, %s, %s", &i, &j, str_num0, str_num1);
			//ret_scan = sscanf(buf, "%d, %d, %s, %s", &i, &j, str_num0, str_num1);
			ret_scan = sscanf(buf, "%d, %d, %[^,], %[^,]", &i, &j, str_num0, str_num1);
			//printf("buf = %s", buf);
			//printf("i, j, str_num[0], str_num[1] = (%d, %d) %s, %s\n", i, j, str_num0, str_num1);
			//A[i * dim + j] = (T)str_num;
			//mpf_set_str(real, str_num0, 10);
			rtd_set_str(real, str_num0);
			//mpf_set_str(imag, str_num1, 10);
			rtd_set_str(imag, str_num1);
			//mpc_set_fr_fr(ctmp, real, imag, get_bnc_default_rounding_mode());
			rctd_set_td_td(&ctmp, real, imag);
			//set_cmpfmatrix_ij(A, i, j, ctmp);
			set_ctdmatrix_ij(A, i, j, &ctmp);
		} while(!feof(fp));

		fclose(fp);
	}

	//mpf_clear(real);mpf_clear(imag);mpc_clear(ctmp);

	// true_x
	//mpf_init2(real, true_x->prec);mpf_init2(imag, true_x->prec);mpc_init2(ctmp, true_x->prec);

	fp = fopen(fname_true_x, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open %s to read true x!\n", fname_true_x);
	}
	else{
		//printf("%s was successfully read...\n", fname_true_x);
		do {
			//ret_scan = fscanf(fp, "%d, %s, %s", &i, str_num0, str_num1);
			ret = fgets(buf, BNC_MAX_BUF_SIZE, fp);
			ret_scan = sscanf(buf, "%d, %[^,], %[^,]", &i, str_num0, str_num1);
			//true_x[i] = (T)str_num;
			//mpf_set_str(real, str_num0, 10);
			rtd_set_str(real, str_num0);
			//mpf_set_str(imag, str_num1, 10);
			rtd_set_str(imag, str_num1);
			//mpc_set_fr_fr(ctmp, real, imag, get_bnc_default_rounding_mode());
			rctd_set_td_td(&ctmp, real, imag);
			//set_cmpfvector_i(true_x, i, ctmp);
			set_ctdvector_i(true_x, i, &ctmp);
		} while(!feof(fp));

		fclose(fp);
	}
	//mpf_clear(real);mpf_clear(imag);mpc_clear(ctmp);

	// b
	//mpf_init2(real, b->prec);mpf_init2(imag, b->prec);mpc_init2(ctmp, b->prec);

	fp = fopen(fname_b, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open %s to read b!\n", fname_b);
	}
	else{
		//printf("%s was successfully read...\n", fname_b);
		do {
			//ret_scan = fscanf(fp, "%d, %s, %s", &i, str_num0, str_num1);
			ret = fgets(buf, BNC_MAX_BUF_SIZE, fp);
			ret_scan = sscanf(buf, "%d, %[^,], %[^,]", &i, str_num0, str_num1);
			//b[i] = (T)str_num;
			//mpf_set_str(real, str_num0, 10);
			rtd_set_str(real, str_num0);
			//mpf_set_str(imag, str_num1, 10);
			rtd_set_str(imag, str_num1);
			//mpc_set_fr_fr(ctmp, real, imag, get_bnc_default_rounding_mode());
			rctd_set_td_td(&ctmp, real, imag);
			//set_cmpfvector_i(b, i, ctmp);
			set_ctdvector_i(b, i, &ctmp);
		} while(!feof(fp));

		fclose(fp);
	}
	//mpf_clear(real);mpf_clear(imag);mpc_clear(ctmp);
}

// 2023-12-15(Fri) T.Kouya
// read problem from file
void read_test_linear_eq_cqd(CQDMatrix A, CQDVector true_x, CQDVector b, long int dim, const char *fname_A, const char *fname_true_x, const char *fname_b)
{
	FILE *fp;
	int i, j, ret_scan;
	char buf[BNC_MAX_BUF_SIZE], str_num0[BNC_MAX_READ_DDIGITS], str_num1[BNC_MAX_READ_DDIGITS]; // up to 8192 dec.digits
	//mpf_t real, imag;
	//mpc_t ctmp;
	double real[QDSIZE], imag[QDSIZE];
	cqdfloat ctmp;
	char *ret;

	// A
	//mpf_init2(real, A->prec);mpf_init2(imag, A->prec);mpc_init2(ctmp, A->prec);

	fp = fopen(fname_A, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open %s to read matrix A!\n", fname_A);
	}
	else{
		#ifdef DEBUG
		printf("DEBUG\n");
		#endif // DEBUG
		//printf("%s was successfully read...\n", fname_A);
		do {
			ret = fgets(buf, BNC_MAX_BUF_SIZE, fp);
			//ret_scan = fscanf(fp, "%d, %d, %s, %s", &i, &j, str_num0, str_num1);
			//ret_scan = sscanf(buf, "%d, %d, %s, %s", &i, &j, str_num0, str_num1);
			ret_scan = sscanf(buf, "%d, %d, %[^,], %[^,]", &i, &j, str_num0, str_num1);
			//printf("buf = %s", buf);
			//printf("i, j, str_num[0], str_num[1] = (%d, %d) %s, %s\n", i, j, str_num0, str_num1);
			//A[i * dim + j] = (T)str_num;
			//mpf_set_str(real, str_num0, 10);
			rqd_set_str(real, str_num0);
			//mpf_set_str(imag, str_num1, 10);
			rqd_set_str(imag, str_num1);
			//mpc_set_fr_fr(ctmp, real, imag, get_bnc_default_rounding_mode());
			rcqd_set_qd_qd(&ctmp, real, imag);
			//set_cmpfmatrix_ij(A, i, j, ctmp);
			set_cqdmatrix_ij(A, i, j, &ctmp);
		} while(!feof(fp));

		fclose(fp);
	}

	//mpf_clear(real);mpf_clear(imag);mpc_clear(ctmp);

	// true_x
	//mpf_init2(real, true_x->prec);mpf_init2(imag, true_x->prec);mpc_init2(ctmp, true_x->prec);

	fp = fopen(fname_true_x, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open %s to read true x!\n", fname_true_x);
	}
	else{
		//printf("%s was successfully read...\n", fname_true_x);
		do {
			//ret_scan = fscanf(fp, "%d, %s, %s", &i, str_num0, str_num1);
			ret = fgets(buf, BNC_MAX_BUF_SIZE, fp);
			ret_scan = sscanf(buf, "%d, %[^,], %[^,]", &i, str_num0, str_num1);
			//true_x[i] = (T)str_num;
			//mpf_set_str(real, str_num0, 10);
			rqd_set_str(real, str_num0);
			//mpf_set_str(imag, str_num1, 10);
			rqd_set_str(imag, str_num1);
			//mpc_set_fr_fr(ctmp, real, imag, get_bnc_default_rounding_mode());
			rcqd_set_qd_qd(&ctmp, real, imag);
			//set_cmpfvector_i(true_x, i, ctmp);
			set_cqdvector_i(true_x, i, &ctmp);
		} while(!feof(fp));

		fclose(fp);
	}
	//mpf_clear(real);mpf_clear(imag);mpc_clear(ctmp);

	// b
	//mpf_init2(real, b->prec);mpf_init2(imag, b->prec);mpc_init2(ctmp, b->prec);

	fp = fopen(fname_b, "r+t");
	if(!fp)
	{
		fprintf(stderr, "cannot open %s to read b!\n", fname_b);
	}
	else{
		//printf("%s was successfully read...\n", fname_b);
		do {
			//ret_scan = fscanf(fp, "%d, %s, %s", &i, str_num0, str_num1);
			ret = fgets(buf, BNC_MAX_BUF_SIZE, fp);
			ret_scan = sscanf(buf, "%d, %[^,], %[^,]", &i, str_num0, str_num1);
			//b[i] = (T)str_num;
			//mpf_set_str(real, str_num0, 10);
			rqd_set_str(real, str_num0);
			//mpf_set_str(imag, str_num1, 10);
			rqd_set_str(imag, str_num1);
			//mpc_set_fr_fr(ctmp, real, imag, get_bnc_default_rounding_mode());
			rcqd_set_qd_qd(&ctmp, real, imag);
			//set_cmpfvector_i(b, i, ctmp);
			set_cqdvector_i(b, i, &ctmp);
		} while(!feof(fp));

		fclose(fp);
	}
	//mpf_clear(real);mpf_clear(imag);mpc_clear(ctmp);
}


#ifdef DEBUG
int main(int argc, char *argv[])
{
    long int dim;
    unsigned long prec;
	char fname_A[256], fname_true_x[256], fname_vec_b[256];
    CMPFMatrix A, diag;
    CMPFVector true_x, vec_b;

//	dim = 128;
	if(argc <= 1)
	{
		fprintf(stderr, "Usage: %s [dim] [prec_b]\n", argv[0]);
		return 0;
	}
	dim = atol(argv[1]);
	if(dim <= 0)
		return 0;

	prec = 128;
	if(argc >= 3)
	{
		prec = (unsigned long)atol(argv[2]);
	//	if(prec < 128)
	//		prec = 128;
	}

    set_bnc_default_prec(prec);

    A = init2_cmpfmatrix(dim, dim, prec);
    diag = init2_cmpfmatrix(dim, dim, prec);
    true_x = init2_cmpfvector(dim, prec);
    vec_b = init2_cmpfvector(dim, prec);

    //get_mptest_linear_eq_c(A, vec_b, true_x, diag, prec, dim, 1, 25, 20230407);

   	sprintf(fname_A, "../python/cmat_a_%d_%d_b%d_c.txt", dim, dim, prec);
	sprintf(fname_true_x, "../python/cvec_true_x_%d_b%d_c.txt", dim, prec);
	sprintf(fname_vec_b, "../python/cvec_b_%d_b%d_c.txt", dim, prec);	

    //write_test_linear_eq_c(A, true_x, vec_b, dim, fname_A, fname_true_x, fname_vec_b);

	read_test_linear_eq_c(A, true_x, vec_b, dim, fname_A, fname_true_x, fname_vec_b);

    free_cmpfmatrix(A);
    free_cmpfmatrix(diag);
    free_cmpfvector(true_x);
    free_cmpfvector(vec_b);

    return 0;
}
#endif // DEBUG
