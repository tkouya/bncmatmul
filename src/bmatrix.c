/********************************************************************************/
/* bmatrix.c: Vector, Matrix                                                    */
/* Copyright (c) 2012 Tomonori Kouya                                            */
/*                                                                              */
/* Version 0.1, 2012-07-01: create bmatrix.c                                    */
/* Version 0.2, 2012-07-18: Bug fix: set[0I]_mpfbmatrix                         */
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

//#include "bnc.h"
//#include "flinear.h"
//#include "dlinear.h"
//#include "mpflinear.h"
#include "bmatrix.h"

/* float */

/* initialize band matrix */
FBMatrix init_fbmatrix(long int dim, long int upper_dim, long int lower_dim)
{
	FBMatrix ret = NULL;
	long int i, j;

	if((dim <= 0) || ((upper_dim + lower_dim + 1) > dim) || (upper_dim < 0) || (lower_dim < 0))
	{
		fprintf(stderr, "ERROR: init_fbmatrix(dim = %ld, upper_dim = %ld, lower_dim = %ld)\n", dim, upper_dim, lower_dim);
		return ret;
	}

	ret = (FBMatrix)malloc(sizeof(fbmatrix));
	if(ret == NULL)
		return ret;

	ret->element = (float *)calloc(sizeof(float), (upper_dim + lower_dim + 1) * dim);
	if(ret->element == NULL)
		return ret;

	/* All 0 */
	for(i = 0; i < (upper_dim + lower_dim + 1); i++)
		for(j = 0; j < dim; j++)
			*(ret->element + i * dim + j) = (float)0.0;

	ret->dim = dim;
	ret->upper_dim = upper_dim;
	ret->lower_dim = lower_dim;

	return ret;
}	

/* free dbmatrix */
void free_fbmatrix(FBMatrix mat)
{
	if(mat == NULL)
		return;

	if(mat->element != NULL)
		free(mat->element);

	free(mat);
}

/* return mat[i][j] */
float get_fbmatrix_ij(FBMatrix mat, long int row_index, long int col_index)
{
	long int i, j;
	float retval = 0.0f;

	i = mat->upper_dim + (row_index - col_index);
	j = col_index;

	if((row_index < 0) || (row_index >= mat->dim) || (col_index < 0) || (col_index >= mat->dim))
	{
		fprintf(stderr, "Warning: get_fbmatrix_ij: Illegal index! (row_index(%ld) -> i(%ld)), (col_index(%ld) -> j(%ld))\n", row_index, i, col_index, j);
		retval = 0.0f;
	}
	if((i < 0) || (i >= (mat->upper_dim + mat->lower_dim + 1)) || (j < 0) || (j >= mat->dim))
	{
		retval = 0.0f;
	}
	else
	{
		retval = *(mat->element + i * mat->dim + j);
	}

	return retval;
}

/* set mat[i][j] = val */
void set_fbmatrix_ij(FBMatrix mat, long int row_index, long int col_index, float val)
{
	long int i, j;

	i = mat->upper_dim + (row_index - col_index);
	j = col_index;

	if((row_index < 0) || (row_index >= mat->dim) || (col_index < 0) || (col_index >= mat->dim))
	{
		fprintf(stderr, "Warning: set_fbmatrix_ij: Illegal index! (row_index(%ld) -> i(%ld)), (col_index(%ld) -> j(%ld))\n", row_index, i, col_index, j);
	}

	if((i >= 0) && (i < (mat->upper_dim + mat->lower_dim + 1)) && (j >= 0) && (j < mat->dim))
		*(mat->element + i * mat->dim + j) = val;
}

/* Multiply DBMatrix * DVector */
int mul_fbmatrix_fvec(FVector ret, FBMatrix mat, FVector vec)
{
	long int i, j, max_j, min_j, total_index;

	if((ret->dim < mat->dim) || (vec->dim != mat->dim))
	{
		fprintf(stderr, "mul_fbmatrix_dvec: Illegal dimension!\n");
		return BNC_ERROR;
	}

	for(i = 0; i < mat->dim; i++)
	{
		// Diagonal element
		ret->element[i] = get_fbmatrix_ij(mat, i, i) * get_fvector_i(vec, i);

		// Upper triangular element
		max_j = i + mat->upper_dim + 1;
		if(max_j > mat->dim)
			max_j = mat->dim;
		for(j = i + 1; j < max_j; j++)
			ret->element[i] += get_fbmatrix_ij(mat, i, j) * get_fvector_i(vec, j);

		// Lower triangular element
		min_j = i - mat->lower_dim;
		if(min_j < 0)
			min_j = 0;
		for(j = min_j; j < i; j++)
			ret->element[i] += get_fbmatrix_ij(mat, i, j) * get_fvector_i(vec, j);
	}

	return BNC_SUCCESS;
}


/* double */

/* initialize band matrix */
DBMatrix init_dbmatrix(long int dim, long int upper_dim, long int lower_dim)
{
	DBMatrix ret = NULL;
	long int i, j;

	if((dim <= 0) || ((upper_dim + lower_dim + 1) > dim) || (upper_dim < 0) || (lower_dim < 0))
	{
		fprintf(stderr, "ERROR: init_dbmatrix(dim = %ld, upper_dim = %ld, lower_dim = %ld)\n", dim, upper_dim, lower_dim);
		return ret;
	}

	ret = (DBMatrix)malloc(sizeof(dbmatrix));
	if(ret == NULL)
		return ret;

	ret->element = (double *)calloc(sizeof(double), (upper_dim + lower_dim + 1) * dim);
	if(ret->element == NULL)
		return ret;

	/* All 0 */
	for(i = 0; i < (upper_dim + lower_dim + 1); i++)
		for(j = 0; j < dim; j++)
			*(ret->element + i * dim + j) = (double)0.0;

	ret->dim = dim;
	ret->upper_dim = upper_dim;
	ret->lower_dim = lower_dim;

	return ret;
}	

/* free dbmatrix */
void free_dbmatrix(DBMatrix mat)
{
	if(mat == NULL)
		return;

	if(mat->element != NULL)
		free(mat->element);

	free(mat);
}

/* return mat[i][j] */
double get_dbmatrix_ij(DBMatrix mat, long int row_index, long int col_index)
{
	long int i, j;
	double retval = 0.0;

	i = mat->upper_dim + (row_index - col_index);
	j = col_index;

	if((row_index < 0) || (row_index >= mat->dim) || (col_index < 0) || (col_index >= mat->dim))
	{
		fprintf(stderr, "Warning: get_dbmatrix_ij: Illegal index! (row_index(%ld) -> i(%ld)), (col_index(%ld) -> j(%ld))\n", row_index, i, col_index, j);
		retval = 0.0;
	}
	if((i < 0) || (i >= (mat->upper_dim + mat->lower_dim + 1)) || (j < 0) || (j >= mat->dim))
	{
		retval = 0.0;
	}
	else
	{
		retval = *(mat->element + i * mat->dim + j);
	}

	return retval;
}

/* set mat[i][j] = val */
void set_dbmatrix_ij(DBMatrix mat, long int row_index, long int col_index, double val)
{
	long int i, j;

	i = mat->upper_dim + (row_index - col_index);
	j = col_index;

	if((row_index < 0) || (row_index >= mat->dim) || (col_index < 0) || (col_index >= mat->dim))
	{
		fprintf(stderr, "Warning: set_dbmatrix_ij: Illegal index! (row_index(%ld) -> i(%ld)), (col_index(%ld) -> j(%ld))\n", row_index, i, col_index, j);
	}

	if((i >= 0) && (i < (mat->upper_dim + mat->lower_dim + 1)) && (j >= 0) && (j < mat->dim))
		*(mat->element + i * mat->dim + j) = val;
}

/* Multiply DBMatrix * DVector */
int mul_dbmatrix_dvec(DVector ret, DBMatrix mat, DVector vec)
{
	long int i, j, max_j, min_j, total_index;

	if((ret->dim < mat->dim) || (vec->dim != mat->dim))
	{
		fprintf(stderr, "mul_dbmatrix_dvec: Illegal dimension!\n");
		return BNC_ERROR;
	}

	for(i = 0; i < mat->dim; i++)
	{
		// Diagonal element
		ret->element[i] = get_dbmatrix_ij(mat, i, i) * get_dvector_i(vec, i);

		// Upper triangular element
		max_j = i + mat->upper_dim + 1;
		if(max_j > mat->dim)
			max_j = mat->dim;
		for(j = i + 1; j < max_j; j++)
			ret->element[i] += get_dbmatrix_ij(mat, i, j) * get_dvector_i(vec, j);

		// Lower triangular element
		min_j = i - mat->lower_dim;
		if(min_j < 0)
			min_j = 0;
		for(j = min_j; j < i; j++)
			ret->element[i] += get_dbmatrix_ij(mat, i, j) * get_dvector_i(vec, j);
	}

	return BNC_SUCCESS;
}

/* Multiply DBMatrix^T * DVector */
int mul_dbmatrixt_dvec(DVector ret, DBMatrix mat, DVector vec)
{
	long int i, j, max_j, min_j, total_index;

	if((ret->dim < mat->dim) || (vec->dim != mat->dim))
	{
		fprintf(stderr, "mul_dbmatrix_dvec: Illegal dimension!\n");
		return BNC_ERROR;
	}

	for(i = 0; i < mat->dim; i++)
	{
		// Diagonal element
		ret->element[i] = get_dbmatrix_ij(mat, i, i) * get_dvector_i(vec, i);

		// Upper triangular element
		//max_j = i + mat->upper_dim + 1;
		max_j = i + mat->lower_dim + 1;
		if(max_j > mat->dim)
			max_j = mat->dim;
		for(j = i + 1; j < max_j; j++)
			ret->element[i] += get_dbmatrix_ij(mat, j, i) * get_dvector_i(vec, j);

		// Lower triangular element
		//min_j = i - max->lower_dim;
		min_j = i - mat->upper_dim;
		if(min_j < 0)
			min_j = 0;
		for(j = min_j; j < i; j++)
			ret->element[i] += get_dbmatrix_ij(mat, j, i) * get_dvector_i(vec, j);
	}

	return BNC_SUCCESS;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Band Matrix        */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 2012-07-01 (Sun) Tomonori Kouya */
/*                                                          */
/************************************************************/
int DBLUdecomp(DBMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DBMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, kmax, jmax, dim;
	double dmaxii;

	dim = a->dim;

	for(i = 0; i < dim; i++)
	{
		dmaxii = fabs(get_dbmatrix_ij(a, i, i));
		if(dmaxii == 0.0)
		{
			fprintf(stderr, "%ld : Error! (DBLUdecomp)!\n", i);
			return -1;
		}

		jmax = i + a->lower_dim + 1;
		if(jmax > dim)
			jmax = dim;

		for(j = (i + 1); j < jmax; j++)
			set_dbmatrix_ij(a, j, i, get_dbmatrix_ij(a, j, i) / get_dbmatrix_ij(a, i, i));

		for(j = (i + 1); j < jmax; j++)
		{
			kmax = j + a->upper_dim + 1;
			if(kmax > dim)
				kmax = dim;

			for(k = (i + 1); k < kmax; k++)
				set_dbmatrix_ij(a, j, k, get_dbmatrix_ij(a, j, k) - get_dbmatrix_ij(a, j, i) * get_dbmatrix_ij(a, i, k));
		}
	}

	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                       (LU Decomposed Square Band Matrix) */
/*                                 (Double Precision)       */
/*                                                          */
/*                       ver. 0.0 2012-07-01 Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveDBLS(DVector answer, DBMatrix lu, DVector b)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       DBMatrix lu: LU decomposed Matrix (given by user)  */
/*       DVector b: constant vector (given by user)         */
/*       DVector answer: Solution for linear system         */
/*       long int dim: Dimension of Matrix (given by user)  */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, jmax, dim;
	double dtmp;

	dim = answer->dim;

	subst_dvector(answer, b);

/* Forward */
	for(i = 0; i < dim; i++)
	{
		if(get_dbmatrix_ij(lu, i, i) == 0.0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveDBLS, %ld)\n", i);
			return -1;
		}

		jmax = i + lu->lower_dim + 1;
		if(jmax > dim)
			jmax = dim;

		for(j = (i + 1); j < jmax; j++)
			set_dvector_i(answer, j, get_dvector_i(answer, j) - get_dbmatrix_ij(lu, j, i) * get_dvector_i(answer, i));
	}

/* Backword */
	for(i = (dim - 1); i >= 0; i--)
	{
		jmax = i + lu->upper_dim + 1;
		if(jmax > dim)
			jmax = dim;

		for(j = (i + 1); j < jmax; j++)
			set_dvector_i(answer, i, get_dvector_i(answer, i) - get_dbmatrix_ij(lu, i, j) * get_dvector_i(answer, j));

		set_dvector_i(answer, i, get_dvector_i(answer, i) / get_dbmatrix_ij(lu, i, i));
	}

	return 0;
}

/* print band matrix */
void print_dbmatrix(DBMatrix mat)
{
	long int i, j;

	for(i = 0; i < mat->dim; i++)
	{
		printf("%5ld: ", i);
		for(j = 0; j < mat->dim; j++)
			printf("%10.3e ", get_dbmatrix_ij(mat, i, j));
		printf("\n");
	}
}

/* set zero matrix */
void set0_dbmatrix(DBMatrix mat)
{
	long int i, j;

	/* All 0 */
	for(i = 0; i < (mat->upper_dim + mat->lower_dim + 1); i++)
		for(j = 0; j < mat->dim; j++)
			*(mat->element + i * mat->dim + j) = (double)0.0;

}

/* set identity matrix */
void setI_dbmatrix(DBMatrix mat)
{
	long int i, j;

	/* All 0 */
	for(i = 0; i < (mat->upper_dim + mat->lower_dim + 1); i++)
		for(j = 0; j < mat->dim; j++)
			*(mat->element + i * mat->dim + j) = (double)0.0;

	/* mat[i][i] = 1 */
	for(i = 0; i < mat->dim; i++)
		*(mat->element + mat->upper_dim * mat->dim + i) = (double)1.0;

}

/* ret := mat_a + mat_b */
void add_dbmatrix(DBMatrix ret, DBMatrix a, DBMatrix b)
{
	long int dim, upper_dim, lower_dim;
	long int i, j, jmin, jmax;

	if((ret->dim != a->dim) || (a->dim != b->dim) || (ret->upper_dim != a->upper_dim) || (ret->lower_dim != a->lower_dim) || (a->upper_dim != b->upper_dim) || (a->lower_dim != b->lower_dim))
	{
		fprintf(stderr, "ERROR: add_dbmatrix: Mismatch dimension(s): (ret, a, b)->dim = (%ld, %ld, %ld), (ret, a, b)->upper_dim = (%ld, %ld, %ld), (ret, a, b)->lower_dim = (%ld, %ld, %ld)\n", ret->dim, a->dim, b->dim, ret->upper_dim, a->upper_dim, b->upper_dim, ret->lower_dim, a->lower_dim, b->lower_dim);
		return;
	}

	dim = ret->dim;
	upper_dim = ret->upper_dim;
	lower_dim = ret->lower_dim;

	for(i = 0; i < dim; i++)
	{
		jmin = i - lower_dim;
		if(jmin < 0)
			jmin = 0;
		jmax = i + upper_dim + 1;
		if(jmax > dim)
			jmax = dim;

		for(j = jmin; j < jmax; j++)
			set_dbmatrix_ij(ret, i, j, get_dbmatrix_ij(a, i, j) + get_dbmatrix_ij(b, i, j));
	}
}

/* ret := mat_a - mat_b */
void sub_dbmatrix(DBMatrix ret, DBMatrix a, DBMatrix b)
{
	long int dim, upper_dim, lower_dim;
	long int i, j, jmin, jmax;

	if((ret->dim != a->dim) || (a->dim != b->dim) || (ret->upper_dim != a->upper_dim) || (ret->lower_dim != a->lower_dim) || (a->upper_dim != b->upper_dim) || (a->lower_dim != b->lower_dim))
	{
		fprintf(stderr, "ERROR: sub_dbmatrix: Mismatch dimension(s): (ret, a, b)->dim = (%ld, %ld, %ld), (ret, a, b)->upper_dim = (%ld, %ld, %ld), (ret, a, b)->lower_dim = (%ld, %ld, %ld)\n", ret->dim, a->dim, b->dim, ret->upper_dim, a->upper_dim, b->upper_dim, ret->lower_dim, a->lower_dim, b->lower_dim);
		return;
	}

	dim = ret->dim;
	upper_dim = ret->upper_dim;
	lower_dim = ret->lower_dim;

	for(i = 0; i < dim; i++)
	{
		jmin = i - lower_dim;
		if(jmin < 0)
			jmin = 0;
		jmax = i + upper_dim + 1;
		if(jmax > dim)
			jmax = dim;

		for(j = jmin; j < jmax; j++)
			set_dbmatrix_ij(ret, i, j, get_dbmatrix_ij(a, i, j) - get_dbmatrix_ij(b, i, j));
	}
}

/* ret := mat_a - mat_b */
void sub_dmatrix_dbmat_dmat(DMatrix ret, DBMatrix a, DMatrix b)
{
	long int i, j;

	if((ret->row_dim != a->dim) || (ret->row_dim != b->row_dim) || (ret->col_dim != a->dim) || (ret->col_dim != b->col_dim))
	{
		fprintf(stderr, "ERROR: sub_dmatrix_dbmat_dmat: Mismatch dimension(s): (ret, a, b)->row_dim = (%ld, %ld, %ld), (ret, a, b)->col_dim = (%ld, %ld, %ld)\n", ret->row_dim, a->dim, b->row_dim, ret->col_dim, a->dim, b->col_dim);
		return;
	}

	for(i = 0; i < ret->row_dim; i++)
	{
		for(j = 0; j < ret->col_dim; j++)
			set_dmatrix_ij(ret, i, j, get_dbmatrix_ij(a, i, j) - get_dmatrix_ij(b, i, j));
	}
}

/* ret := val * a */
void cmul_dbmatrix(DBMatrix ret, double val, DBMatrix a)
{
	long int dim, upper_dim, lower_dim;
	long int i, j, jmin, jmax;

	if((ret->dim != a->dim) || (ret->upper_dim != a->upper_dim) || (ret->lower_dim != a->lower_dim))
	{
		fprintf(stderr, "ERROR: cmul_dbmatrix: Mismatch dimension(s): (ret, a)->dim = (%ld, %ld), (ret, a)->upper_dim = (%ld, %ld), (ret, a)->lower_dim = (%ld, %ld)\n", ret->dim, a->dim, ret->upper_dim, a->upper_dim, ret->lower_dim, a->lower_dim);
		return;
	}

	dim = ret->dim;
	upper_dim = ret->upper_dim;
	lower_dim = ret->lower_dim;

	for(i = 0; i < dim; i++)
	{
		jmin = i - lower_dim;
		if(jmin < 0)
			jmin = 0;
		jmax = i + upper_dim + 1;
		if(jmax > dim)
			jmax = dim;

		for(j = jmin; j < jmax; j++)
			set_dbmatrix_ij(ret, i, j, val * get_dbmatrix_ij(a, i, j));
	}
}

/* ret := a * b */

/* ex1) dim = 5, upper_dim = 2, lower_dim = 1     */
/*            -> upper_dim = 4, lower_dim = 2     */
/* [ r r r r r ]    [ a a a 0 0 ]   [ b b b 0 0 ] */
/* [ r r r r r ]    [ a a a a 0 ]   [ b b b b 0 ] */
/* [ r r r r r ] := [ 0 a a a a ] * [ 0 b b b b ] */
/* [ 0 r r r r ]    [ 0 0 a a a ]   [ 0 0 b b b ] */
/* [ 0 0 r r r ]    [ 0 0 0 a a ]   [ 0 0 0 b b ] */

/* ex2) dim = 5, upper_dim = 1, lower_dim = 1     */
/*            -> upper_dim = 2, lower_dim = 2     */
/* [ r r r 0 0 ]    [ a a 0 0 0 ]   [ b b 0 0 0 ] */
/* [ r r r r 0 ]    [ a a a 0 0 ]   [ b b b 0 0 ] */
/* [ r r r r r ] := [ 0 a a a 0 ] * [ 0 b b b 0 ] */
/* [ 0 r r r r ]    [ 0 0 a a a ]   [ 0 0 b b b ] */
/* [ 0 0 r r r ]    [ 0 0 0 a a ]   [ 0 0 0 b b ] */

/* ex3) dim = 5, upper_dim = 0, lower_dim = 1     */
/*            -> upper_dim = 0, lower_dim = 2     */
/* [ r 0 0 0 0 ]    [ a 0 0 0 0 ]   [ b 0 0 0 0 ] */
/* [ r r 0 0 0 ]    [ a a 0 0 0 ]   [ b b 0 0 0 ] */
/* [ r r r 0 0 ] := [ 0 a a 0 0 ] * [ 0 b b 0 0 ] */
/* [ 0 r r r 0 ]    [ 0 0 a a 0 ]   [ 0 0 b b 0 ] */
/* [ 0 0 r r r ]    [ 0 0 0 a a ]   [ 0 0 0 b b ] */

/* ex4) dim = 5, upper_dim = 1, lower_dim = 0     */
/*            -> upper_dim = 2, lower_dim = 0     */
/* [ r r r 0 0 ]    [ a a 0 0 0 ]   [ b b 0 0 0 ] */
/* [ 0 r r r 0 ]    [ 0 a a 0 0 ]   [ 0 b b 0 0 ] */
/* [ 0 0 r r r ] := [ 0 0 a a 0 ] * [ 0 0 b b 0 ] */
/* [ 0 0 0 r r ]    [ 0 0 0 a a ]   [ 0 0 0 b b ] */
/* [ 0 0 0 0 r ]    [ 0 0 0 0 a ]   [ 0 0 0 0 b ] */

/* ret := a * b */
void mul_dmatrix_dbmat_dbmat(DMatrix ret, DBMatrix a, DBMatrix b)
{
	long int i, j, k;
	double tmp;

	if((ret->row_dim != a->dim) || (ret->row_dim != b->dim) || (ret->col_dim != a->dim) || (ret->col_dim != b->dim))
	{
		fprintf(stderr, "ERROR: mul_dmatrix_dbmat_dbmat: Mismatch dimension(s): (ret, a, b)->row_dim = (%ld, %ld, %ld), (ret, a, b)->col_dim = (%ld, %ld, %ld)\n", ret->row_dim, a->dim, b->dim, ret->col_dim, a->dim, b->dim);
		return;
	}

	for(i = 0; i < ret->row_dim; i++)
	{
		for(j = 0; j < ret->col_dim; j++)
		{
			tmp = 0.0;
			for(k = 0; k < a->dim; k++)
				tmp += get_dbmatrix_ij(a, i, k) * get_dbmatrix_ij(b, k, j);

			set_dmatrix_ij(ret, i, j, tmp);
		}
	}
}

/* ret := a * b */
void mul_dmatrix_dmat_dbmat(DMatrix ret, DMatrix a, DBMatrix b)
{
	long int i, j, k;
	double tmp;

	if((ret->row_dim != a->row_dim) || (ret->row_dim != b->dim) || (ret->col_dim != a->col_dim) || (ret->col_dim != b->dim))
	{
		fprintf(stderr, "ERROR: mul_dmatrix_dmat_dbmat: Mismatch dimension(s): (ret, a, b)->row_dim = (%ld, %ld, %ld), (ret, a, b)->col_dim = (%ld, %ld, %ld)\n", ret->row_dim, a->row_dim, b->dim, ret->col_dim, a->col_dim, b->dim);
		return;
	}

	for(i = 0; i < ret->row_dim; i++)
	{
		for(j = 0; j < ret->col_dim; j++)
		{
			tmp = 0.0;
			for(k = 0; k < a->col_dim; k++)
				tmp += get_dmatrix_ij(a, i, k) * get_dbmatrix_ij(b, k, j);

			set_dmatrix_ij(ret, i, j, tmp);
		}
	}
}

/* ret := a * b */
void mul_dmatrix_dbmat_dmat(DMatrix ret, DBMatrix a, DMatrix b)
{
	long int i, j, k;
	double tmp;

	if((ret->row_dim != a->dim) || (ret->row_dim != b->row_dim) || (ret->col_dim != a->dim) || (ret->col_dim != b->col_dim))
	{
		fprintf(stderr, "ERROR: mul_dmatrix_dbmat_dmat: Mismatch dimension(s): (ret, a, b)->row_dim = (%ld, %ld, %ld), (ret, a, b)->col_dim = (%ld, %ld, %ld)\n", ret->row_dim, a->dim, b->row_dim, ret->col_dim, a->dim, b->col_dim);
		return;
	}

	for(i = 0; i < ret->row_dim; i++)
	{
		for(j = 0; j < ret->col_dim; j++)
		{
			tmp = 0.0;
			for(k = 0; k < a->dim; k++)
				tmp += get_dbmatrix_ij(a, i, k) * get_dmatrix_ij(b, k, j);

			set_dmatrix_ij(ret, i, j, tmp);
		}
	}
}

/* ret := a */
void subst_dbmatrix(DBMatrix ret, DBMatrix a)
{
	long int dim, upper_dim, lower_dim;
	long int i, j, jmin, jmax;

	if((ret->dim != a->dim) || (ret->upper_dim != a->upper_dim) || (ret->lower_dim != a->lower_dim))
	{
		fprintf(stderr, "ERROR: subst_dbmatrix: Mismatch dimension(s): (ret, a)->dim = (%ld, %ld), (ret, a)->upper_dim = (%ld, %ld), (ret, a)->lower_dim = (%ld, %ld)\n", ret->dim, a->dim, ret->upper_dim, a->upper_dim, ret->lower_dim, a->lower_dim);
		return;
	}

	dim = ret->dim;
	upper_dim = ret->upper_dim;
	lower_dim = ret->lower_dim;

	for(i = 0; i < dim; i++)
	{
		jmin = i - lower_dim;
		if(jmin < 0)
			jmin = 0;
		jmax = i + upper_dim + 1;
		if(jmax > dim)
			jmax = dim;

		for(j = jmin; j < jmax; j++)
			set_dbmatrix_ij(ret, i, j, get_dbmatrix_ij(a, i, j));
	}
}

/* ret := (DBMatrix)a */
void subst_fbmatrix_dbmat(FBMatrix ret, DBMatrix a)
{
	long int dim, upper_dim, lower_dim;
	long int i, j, jmin, jmax;

	if((ret->dim != a->dim) || (ret->upper_dim != a->upper_dim) || (ret->lower_dim != a->lower_dim))
	{
		fprintf(stderr, "ERROR: subst_fbmatrix_dbmat: Mismatch dimension(s): (ret, a)->dim = (%ld, %ld), (ret, a)->upper_dim = (%ld, %ld), (ret, a)->lower_dim = (%ld, %ld)\n", ret->dim, a->dim, ret->upper_dim, a->upper_dim, ret->lower_dim, a->lower_dim);
		return;
	}

	dim = ret->dim;
	upper_dim = ret->upper_dim;
	lower_dim = ret->lower_dim;

	for(i = 0; i < dim; i++)
	{
		jmin = i - lower_dim;
		if(jmin < 0)
			jmin = 0;
		jmax = i + upper_dim + 1;
		if(jmax > dim)
			jmax = dim;

		for(j = jmin; j < jmax; j++)
			set_fbmatrix_ij(ret, i, j, (float)get_dbmatrix_ij(a, i, j));
	}
}

/* ret := (FBMatrix)a */
void subst_dbmatrix_fbmat(DBMatrix ret, FBMatrix a)
{
	long int dim, upper_dim, lower_dim;
	long int i, j, jmin, jmax;

	if((ret->dim != a->dim) || (ret->upper_dim != a->upper_dim) || (ret->lower_dim != a->lower_dim))
	{
		fprintf(stderr, "ERROR: subst_dbmatrix_fbmat: Mismatch dimension(s): (ret, a)->dim = (%ld, %ld), (ret, a)->upper_dim = (%ld, %ld), (ret, a)->lower_dim = (%ld, %ld)\n", ret->dim, a->dim, ret->upper_dim, a->upper_dim, ret->lower_dim, a->lower_dim);
		return;
	}

	dim = ret->dim;
	upper_dim = ret->upper_dim;
	lower_dim = ret->lower_dim;

	for(i = 0; i < dim; i++)
	{
		jmin = i - lower_dim;
		if(jmin < 0)
			jmin = 0;
		jmax = i + upper_dim + 1;
		if(jmax > dim)
			jmax = dim;

		for(j = jmin; j < jmax; j++)
			set_dbmatrix_ij(ret, i, j, (double)get_fbmatrix_ij(a, i, j));
	}
}

/* ret := (DMatrix)a */
void subst_dmatrix_dbmat(DMatrix ret, DBMatrix a)
{
	long int dim, upper_dim, lower_dim;
	long int i, j, jmin, jmax;

	if((ret->row_dim != a->dim) || (ret->col_dim != a->dim))
	{
		fprintf(stderr, "ERROR: subst_dmatrix_dbmat: Mismatch dimension(s): (ret, a)->row_dim = (%ld, %ld), (ret, a)->col_dim = (%ld, %ld)\n", ret->row_dim, a->dim, ret->col_dim, a->dim);
		return;
	}

	// ret := 0
	set0_dmatrix(ret);

	dim = ret->row_dim;
	upper_dim = a->upper_dim;
	lower_dim = a->lower_dim;

	for(i = 0; i < dim; i++)
	{
		jmin = i - lower_dim;
		if(jmin < 0)
			jmin = 0;
		jmax = i + upper_dim + 1;
		if(jmax > dim)
			jmax = dim;

		for(j = jmin; j < jmax; j++)
			set_dmatrix_ij(ret, i, j, (double)get_dbmatrix_ij(a, i, j));
	}
}


/* MPF */

#ifdef USE_GMP

/* initialize band matrix */
MPFBMatrix init_mpfbmatrix(long int dim, long int upper_dim, long int lower_dim)
{
	MPFBMatrix ret = NULL;
	long int i, j;

	if((dim <= 0) || ((upper_dim + lower_dim + 1) > dim) || (upper_dim < 0) || (lower_dim < 0))
	{
		fprintf(stderr, "ERROR: init_mpfbmatrix(dim = %ld, upper_dim = %ld, lower_dim = %ld)\n", dim, upper_dim, lower_dim);
		return ret;
	}

	ret = (MPFBMatrix)malloc(sizeof(mpfbmatrix));
	if(ret == NULL)
		return ret;

	ret->element = (mpf_t *)calloc(sizeof(mpf_t), (upper_dim + lower_dim + 1) * dim);
	if(ret->element == NULL)
		return ret;

	/* All 0 */
	for(i = 0; i < (upper_dim + lower_dim + 1); i++)
	{
		for(j = 0; j < dim; j++)
		{
			mpf_init((mpf_ptr)(ret->element + i * dim + j));
			mpf_set_ui((mpf_ptr)(ret->element + i * dim + j), 0UL);
		}
	}

	/* zero = 0 */
	mpf_init_set_ui(ret->zero, 0UL);

	ret->prec = get_bnc_default_prec();
	ret->dim = dim;
	ret->upper_dim = upper_dim;
	ret->lower_dim = lower_dim;

	return ret;
}	

/* initialize band matrix */
MPFBMatrix init2_mpfbmatrix(long int dim, long int upper_dim, long int lower_dim, unsigned long prec)
{
	MPFBMatrix ret = NULL;
	long int i, j;

	if((dim <= 0) || ((upper_dim + lower_dim + 1) > dim) || (upper_dim < 0) || (lower_dim < 0))
	{
		fprintf(stderr, "ERROR: init2_mpfbmatrix(dim = %ld, upper_dim = %ld, lower_dim = %ld, prec = %ld)\n", dim, upper_dim, lower_dim, prec);
		return ret;
	}

	ret = (MPFBMatrix)malloc(sizeof(mpfbmatrix));
	if(ret == NULL)
		return ret;

	ret->element = (mpf_t *)calloc(sizeof(mpf_t), (upper_dim + lower_dim + 1) * dim);
	if(ret->element == NULL)
		return ret;

	/* All 0 */
	for(i = 0; i < (upper_dim + lower_dim + 1); i++)
	{
		for(j = 0; j < dim; j++)
		{
			mpf_init2((mpf_ptr)(ret->element + i * dim + j), prec);
			mpf_set_ui((mpf_ptr)(ret->element + i * dim + j), 0UL);
		}
	}

	/* zero = 0 */
	mpf_init2(ret->zero, prec);
	mpf_set_ui(ret->zero, 0UL);

	ret->prec = prec;
	ret->dim = dim;
	ret->upper_dim = upper_dim;
	ret->lower_dim = lower_dim;

	return ret;
}	

/* free mpfbmatrix */
void free_mpfbmatrix(MPFBMatrix mat)
{
	long int i, j;

	if(mat == NULL)
		return;

	if(mat->element != NULL)
	{
		for(i = 0; i < (mat->upper_dim + mat->lower_dim + 1); i++)
		{
			for(j = 0; j < mat->dim; j++)
				mpf_clear((mpf_ptr)(mat->element + i * mat->dim + j));
		}
		free(mat->element);
		mpf_clear(mat->zero);
	}
	free(mat);
}

/* return mat[i][j] */
mpf_ptr get_mpfbmatrix_ij(MPFBMatrix mat, long int row_index, long int col_index)
{
	long int i, j;

	i = mat->upper_dim + (row_index - col_index);
	j = col_index;

	if((row_index < 0) || (row_index >= mat->dim) || (col_index < 0) || (col_index >= mat->dim))
	{
		fprintf(stderr, "Warning: get_mpfbmatrix_ij: Illegal index! (row_index(%ld) -> i(%ld)), (col_index(%ld) -> j(%ld))\n", row_index, i, col_index, j);
		return mat->zero;
	}
	if((i < 0) || (i >= (mat->upper_dim + mat->lower_dim + 1)) || (j < 0) || (j >= mat->dim))
	{
		return mat->zero;
	}
	else
	{
		return *(mat->element + i * mat->dim + j);
	}
}

/* set mat[i][j] = val */
void set_mpfbmatrix_ij(MPFBMatrix mat, long int row_index, long int col_index, mpf_t val)
{
	long int i, j;

	i = mat->upper_dim + (row_index - col_index);
	j = col_index;

	if((row_index < 0) || (row_index >= mat->dim) || (col_index < 0) || (col_index >= mat->dim))
	{
		fprintf(stderr, "Warning: set_mpfbmatrix_ij: Illegal index! (row_index(%ld) -> i(%ld)), (col_index(%ld) -> j(%ld))\n", row_index, i, col_index, j);
	}

	if((i >= 0) && (i < (mat->upper_dim + mat->lower_dim + 1)) && (j >= 0) && (j < mat->dim))
		mpf_set(*(mat->element + i * mat->dim + j), val);
}

/* set mat[i][j] = val */
void set_mpfbmatrix_ij_d(MPFBMatrix mat, long int row_index, long int col_index, double val)
{
	long int i, j;

	i = mat->upper_dim + (row_index - col_index);
	j = col_index;

	if((row_index < 0) || (row_index >= mat->dim) || (col_index < 0) || (col_index >= mat->dim))
	{
		fprintf(stderr, "Warning: set_mpfbmatrix_ij_d: Illegal index! (row_index(%ld) -> i(%ld)), (col_index(%ld) -> j(%ld))\n", row_index, i, col_index, j);
	}

	if((i >= 0) && (i < (mat->upper_dim + mat->lower_dim + 1)) && (j >= 0) && (j < mat->dim))
		mpf_set_d(*(mat->element + i * mat->dim + j), val);
}

/* set mat[i][j] = val */
void set_mpfbmatrix_ij_ui(MPFBMatrix mat, long int row_index, long int col_index, unsigned long val)
{
	long int i, j;

	i = mat->upper_dim + (row_index - col_index);
	j = col_index;

	if((row_index < 0) || (row_index >= mat->dim) || (col_index < 0) || (col_index >= mat->dim))
	{
		fprintf(stderr, "Warning: set_mpfbmatrix_ij_ui: Illegal index! (row_index(%ld) -> i(%ld)), (col_index(%ld) -> j(%ld))\n", row_index, i, col_index, j);
	}

	if((i >= 0) && (i < (mat->upper_dim + mat->lower_dim + 1)) && (j >= 0) && (j < mat->dim))
		mpf_set_ui(*(mat->element + i * mat->dim + j), val);
}

/* Multiply MPFBMatrix * MPFVector */
int mul_mpfbmatrix_mpfvec(MPFVector ret, MPFBMatrix mat, MPFVector vec)
{
	long int i, j, max_j, min_j, total_index;

	if((ret->dim < mat->dim) || (vec->dim != mat->dim))
	{
		fprintf(stderr, "mul_mpfbmatrix_dvec: Illegal dimension!\n");
		return BNC_ERROR;
	}

	for(i = 0; i < mat->dim; i++)
	{
		// Diagonal element
		//ret->element[i] = get_mpfbmatrix_ij(mat, i, i) * get_dvector_i(vec, i);
		mpf_mul(ret->element[i], get_mpfbmatrix_ij(mat, i, i), get_mpfvector_i(vec, i));

		// Upper triangular element
		max_j = i + mat->upper_dim + 1;
		if(max_j > mat->dim)
			max_j = mat->dim;
		for(j = i + 1; j < max_j; j++)
		{
			//ret->element[i] += get_mpfbmatrix_ij(mat, i, j) * get_dvector_i(vec, j);
			mpf_fma(ret->element[i], get_mpfbmatrix_ij(mat, i, j), get_mpfvector_i(vec, j), ret->element[i]);
		}

		// Lower triangular element
		min_j = i - mat->lower_dim;
		if(min_j < 0)
			min_j = 0;
		for(j = min_j; j < i; j++)
		{
			//ret->element[i] += get_mpfbmatrix_ij(mat, i, j) * get_dvector_i(vec, j);
			mpf_fma(ret->element[i], get_mpfbmatrix_ij(mat, i, j), get_mpfvector_i(vec, j), ret->element[i]);
		}
	}

	return BNC_SUCCESS;
}

/* Multiply MPFBMatrix^T * MPFVector */
int mul_mpfbmatrixt_mpfvec(MPFVector ret, MPFBMatrix mat, MPFVector vec)
{
	long int i, j, max_j, min_j, total_index;

	if((ret->dim < mat->dim) || (vec->dim != mat->dim))
	{
		fprintf(stderr, "mul_mpfbmatrix_dvec: Illegal dimension!\n");
		return BNC_ERROR;
	}

	for(i = 0; i < mat->dim; i++)
	{
		// Diagonal element
		//ret->element[i] = get_mpfbmatrix_ij(mat, i, i) * get_dvector_i(vec, i);
		mpf_mul(ret->element[i], get_mpfbmatrix_ij(mat, i, i), get_mpfvector_i(vec, i));

		// Upper triangular element
		//max_j = i + mat->upper_dim + 1;
		max_j = i + mat->lower_dim + 1;
		if(max_j > mat->dim)
			max_j = mat->dim;
		for(j = i + 1; j < max_j; j++)
		{
			//ret->element[i] += get_mpfbmatrix_ij(mat, j, i) * get_dvector_i(vec, j);
			mpf_fma(ret->element[i], get_mpfbmatrix_ij(mat, j, i), get_mpfvector_i(vec, j), ret->element[i]);
		}

		// Lower triangular element
		//min_j = i - max->lower_dim;
		min_j = i - mat->upper_dim;
		if(min_j < 0)
			min_j = 0;
		for(j = min_j; j < i; j++)
		{
			//ret->element[i] += get_mpfbmatrix_ij(mat, j, i) * get_dvector_i(vec, j);
			mpf_fma(ret->element[i], get_mpfbmatrix_ij(mat, j, i), get_mpfvector_i(vec, j), ret->element[i]);
		}
	}

	return BNC_SUCCESS;
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Band Matrix        */
/*                                 (Multiple Precision)     */
/*                                                          */
/*                 ver. 0.0 2012-07-03 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
int MPFBLUdecomp(MPFBMatrix a)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       MPFBMatrix a: Matrix (given by user)                 */
/*                   a[i * dim + j] ... (i,j)-entry of A    */
/*                                                          */
/* returns                                                  */
/*       a[]: LU decomposed matrix                          */
/*                                                          */
/************************************************************/
{
	long int i, j, k, kmax, jmax, dim;
	mpf_t dmaxii, tmp;

	mpf_init2(dmaxii, a->prec);
	mpf_init2(tmp, a->prec);

	dim = a->dim;

	for(i = 0; i < dim; i++)
	{
		mpf_abs(dmaxii, get_mpfbmatrix_ij(a, i, i));
		if(mpf_cmp_ui(dmaxii, 0UL) == 0)
		{
			fprintf(stderr, "%ld : Error! (MPFBLUdecomp)!\n", i);
			return -1;
		}

		jmax = i + a->lower_dim + 1;
		if(jmax > dim)
			jmax = dim;

		for(j = (i + 1); j < jmax; j++)
		{
			//set_mpfbmatrix_ij(a, j, i, get_mpfbmatrix_ij(a, j, i) / get_mpfbmatrix_ij(a, i, i));
			mpf_div(dmaxii, get_mpfbmatrix_ij(a, j, i), get_mpfbmatrix_ij(a, i, i));
			set_mpfbmatrix_ij(a, j, i, dmaxii);
		}

		for(j = (i + 1); j < jmax; j++)
		{
			kmax = j + a->upper_dim + 1;
			if(kmax > dim)
				kmax = dim;

			for(k = (i + 1); k < kmax; k++)
			{
				mpf_mul(dmaxii, get_mpfbmatrix_ij(a, j, i), get_mpfbmatrix_ij(a, i, k));
				mpf_sub(tmp, get_mpfbmatrix_ij(a, j, k), dmaxii);
				set_mpfbmatrix_ij(a, j, k, tmp);
			}
		}
	}

	mpf_clear(dmaxii);
	mpf_clear(tmp);

	return 0;
}

/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                       (LU Decomposed Square Band Matrix) */
/*                                 (Multiple Precision)     */
/*                                                          */
/*                       ver. 0.0 2012-07-01 Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveMPFBLS(MPFVector answer, MPFBMatrix lu, MPFVector b)
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       MPFBMatrix lu: LU decomposed Matrix (given by user)*/
/*       MPFVector b: constant vector (given by user)       */
/*       MPFVector answer: Solution for linear system       */
/*       long int dim: Dimension of Matrix (given by user)  */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, jmax, dim;
	mpf_t tmp, tmp1;

	dim = answer->dim;
	mpf_init2(tmp, answer->prec);
	mpf_init2(tmp1, answer->prec);

	subst_mpfvector(answer, b);

/* Forward */
	for(i = 0; i < dim; i++)
	{
		mpf_abs(tmp, get_mpfbmatrix_ij(lu, i, i));
		if(mpf_cmp_ui(tmp, 0UL) == 0)
		{
			fprintf(stderr, "Unable to solve the linear system!(SolveMPFBLS, %ld)\n", i);
			return -1;
		}

		jmax = i + lu->lower_dim + 1;
		if(jmax > dim)
			jmax = dim;

		for(j = (i + 1); j < jmax; j++)
		{
			mpf_mul(tmp, get_mpfbmatrix_ij(lu, j, i), get_mpfvector_i(answer, i));
			mpf_sub(tmp1, get_mpfvector_i(answer, j), tmp);
			set_mpfvector_i(answer, j, tmp1);
		}
	}

/* Backward */
	for(i = (dim - 1); i >= 0; i--)
	{
		jmax = i + lu->upper_dim + 1;
		if(jmax > dim)
			jmax = dim;

		for(j = (i + 1); j < jmax; j++)
		{
			mpf_mul(tmp, get_mpfbmatrix_ij(lu, i, j), get_mpfvector_i(answer, j));
			mpf_sub(tmp1, get_mpfvector_i(answer, i),  tmp);
			set_mpfvector_i(answer, i, tmp1);
		}

		mpf_div(tmp, get_mpfvector_i(answer, i), get_mpfbmatrix_ij(lu, i, i));
		set_mpfvector_i(answer, i, tmp);
	}

	mpf_clear(tmp);
	mpf_clear(tmp1);

	return 0;
}

/* print band matrix */
void print_mpfbmatrix(MPFBMatrix mat)
{
	long int i, j;

	for(i = 0; i < mat->dim; i++)
	{
		printf("%5ld: ", i);
		for(j = 0; j < mat->dim; j++)
			printf("%10.3e ", mpf_get_d(get_mpfbmatrix_ij(mat, i, j)));
			//printf("%10.3e ", mpf2double(get_mpfbmatrix_ij(mat, i, j)));
		printf("\n");
	}
}

/* set zero matrix */
void set0_mpfbmatrix(MPFBMatrix mat)
{
	long int i, j;

	/* All 0 */
	for(i = 0; i < (mat->upper_dim + mat->lower_dim + 1); i++)
	{
		for(j = 0; j < mat->dim; j++)
			mpf_set_ui((mpf_ptr)(mat->element + i * mat->dim + j), 0UL);
	}
}

/* set identity matrix */
void setI_mpfbmatrix(MPFBMatrix mat)
{
	long int i, j;

	/* All 0 */
	for(i = 0; i < (mat->upper_dim + mat->lower_dim + 1); i++)
	{
		for(j = 0; j < mat->dim; j++)
			mpf_set_ui((mpf_ptr)(mat->element + i * mat->dim + j), 0UL);
	}

	/* mat[i][i] = 1 */
	for(i = 0; i < mat->dim; i++)
		mpf_set_ui((mpf_ptr)(mat->element + mat->upper_dim * mat->dim + i), 1UL);

}

/* ret := mat_a + mat_b */
void add_mpfbmatrix(MPFBMatrix ret, MPFBMatrix a, MPFBMatrix b)
{
	long int dim, upper_dim, lower_dim;
	long int i, j, jmin, jmax;
	mpf_t tmp;

	if((ret->dim != a->dim) || (a->dim != b->dim) || (ret->upper_dim != a->upper_dim) || (ret->lower_dim != a->lower_dim) || (a->upper_dim != b->upper_dim) || (a->lower_dim != b->lower_dim))
	{
		fprintf(stderr, "ERROR: add_mpfbmatrix: Mismatch dimension(s): (ret, a, b)->dim = (%ld, %ld, %ld), (ret, a, b)->upper_dim = (%ld, %ld, %ld), (ret, a, b)->lower_dim = (%ld, %ld, %ld)\n", ret->dim, a->dim, b->dim, ret->upper_dim, a->upper_dim, b->upper_dim, ret->lower_dim, a->lower_dim, b->lower_dim);
		return;
	}

	dim = ret->dim;
	upper_dim = ret->upper_dim;
	lower_dim = ret->lower_dim;

	mpf_init2(tmp, ret->prec);

	for(i = 0; i < dim; i++)
	{
		jmin = i - lower_dim;
		if(jmin < 0)
			jmin = 0;
		jmax = i + upper_dim + 1;
		if(jmax > dim)
			jmax = dim;

		for(j = jmin; j < jmax; j++)
		{
			mpf_add(tmp, get_mpfbmatrix_ij(a, i, j), get_mpfbmatrix_ij(b, i, j));
			set_mpfbmatrix_ij(ret, i, j, tmp);
		}
	}

	mpf_clear(tmp);
}

/* ret := mat_a - mat_b */
void sub_mpfbmatrix(MPFBMatrix ret, MPFBMatrix a, MPFBMatrix b)
{
	long int dim, upper_dim, lower_dim;
	long int i, j, jmin, jmax;
	mpf_t tmp;

	if((ret->dim != a->dim) || (a->dim != b->dim) || (ret->upper_dim != a->upper_dim) || (ret->lower_dim != a->lower_dim) || (a->upper_dim != b->upper_dim) || (a->lower_dim != b->lower_dim))
	{
		fprintf(stderr, "ERROR: sub_mpfbmatrix: Mismatch dimension(s): (ret, a, b)->dim = (%ld, %ld, %ld), (ret, a, b)->upper_dim = (%ld, %ld, %ld), (ret, a, b)->lower_dim = (%ld, %ld, %ld)\n", ret->dim, a->dim, b->dim, ret->upper_dim, a->upper_dim, b->upper_dim, ret->lower_dim, a->lower_dim, b->lower_dim);
		return;
	}

	dim = ret->dim;
	upper_dim = ret->upper_dim;
	lower_dim = ret->lower_dim;

	mpf_init2(tmp, ret->prec);

	for(i = 0; i < dim; i++)
	{
		jmin = i - lower_dim;
		if(jmin < 0)
			jmin = 0;
		jmax = i + upper_dim + 1;
		if(jmax > dim)
			jmax = dim;

		for(j = jmin; j < jmax; j++)
		{
			mpf_sub(tmp, get_mpfbmatrix_ij(a, i, j), get_mpfbmatrix_ij(b, i, j));
			set_mpfbmatrix_ij(ret, i, j, tmp);
		}
	}

	mpf_clear(tmp);

}

/* ret := mat_a - mat_b */
void sub_mpfmatrix_mpfbmat_mpfmat(MPFMatrix ret, MPFBMatrix a, MPFMatrix b)
{
	long int i, j;
	mpf_t tmp;

	if((ret->row_dim != a->dim) || (ret->row_dim != b->row_dim) || (ret->col_dim != a->dim) || (ret->col_dim != b->col_dim))
	{
		fprintf(stderr, "ERROR: sub_mpfmatrix_mpfbmat_mpfmat: Mismatch dimension(s): (ret, a, b)->row_dim = (%ld, %ld, %ld), (ret, a, b)->col_dim = (%ld, %ld, %ld)\n", ret->row_dim, a->dim, b->row_dim, ret->col_dim, a->dim, b->col_dim);
		return;
	}

	mpf_init2(tmp, ret->prec);

	for(i = 0; i < ret->row_dim; i++)
	{
		for(j = 0; j < ret->col_dim; j++)
		{
			mpf_sub(tmp, get_mpfbmatrix_ij(a, i, j), get_mpfmatrix_ij(b, i, j));
			set_mpfmatrix_ij(ret, i, j, tmp);
		}
	}

	mpf_clear(tmp);

}


/* ret := val * a */
void cmul_mpfbmatrix(MPFBMatrix ret, mpf_t val, MPFBMatrix a)
{
	long int dim, upper_dim, lower_dim;
	long int i, j, jmin, jmax;
	mpf_t tmp;

	if((ret->dim != a->dim) || (ret->upper_dim != a->upper_dim) || (ret->lower_dim != a->lower_dim))
	{
		fprintf(stderr, "ERROR: cmul_mpfbmatrix: Mismatch dimension(s): (ret, a)->dim = (%ld, %ld), (ret, a)->upper_dim = (%ld, %ld), (ret, a)->lower_dim = (%ld, %ld)\n", ret->dim, a->dim, ret->upper_dim, a->upper_dim, ret->lower_dim, a->lower_dim);
		return;
	}

	dim = ret->dim;
	upper_dim = ret->upper_dim;
	lower_dim = ret->lower_dim;

	mpf_init2(tmp, ret->prec);

	for(i = 0; i < dim; i++)
	{
		jmin = i - lower_dim;
		if(jmin < 0)
			jmin = 0;
		jmax = i + upper_dim + 1;
		if(jmax > dim)
			jmax = dim;

		for(j = jmin; j < jmax; j++)
		{
			mpf_mul(tmp, val, get_mpfbmatrix_ij(a, i, j));
			set_mpfbmatrix_ij(ret, i, j, tmp);
		}
	}

	mpf_clear(tmp);
}

/* ret := a * b */

/* ex1) dim = 5, upper_dim = 2, lower_dim = 1     */
/*            -> upper_dim = 4, lower_dim = 2     */
/* [ r r r r r ]    [ a a a 0 0 ]   [ b b b 0 0 ] */
/* [ r r r r r ]    [ a a a a 0 ]   [ b b b b 0 ] */
/* [ r r r r r ] := [ 0 a a a a ] * [ 0 b b b b ] */
/* [ 0 r r r r ]    [ 0 0 a a a ]   [ 0 0 b b b ] */
/* [ 0 0 r r r ]    [ 0 0 0 a a ]   [ 0 0 0 b b ] */

/* ex2) dim = 5, upper_dim = 1, lower_dim = 1     */
/*            -> upper_dim = 2, lower_dim = 2     */
/* [ r r r 0 0 ]    [ a a 0 0 0 ]   [ b b 0 0 0 ] */
/* [ r r r r 0 ]    [ a a a 0 0 ]   [ b b b 0 0 ] */
/* [ r r r r r ] := [ 0 a a a 0 ] * [ 0 b b b 0 ] */
/* [ 0 r r r r ]    [ 0 0 a a a ]   [ 0 0 b b b ] */
/* [ 0 0 r r r ]    [ 0 0 0 a a ]   [ 0 0 0 b b ] */

/* ex3) dim = 5, upper_dim = 0, lower_dim = 1     */
/*            -> upper_dim = 0, lower_dim = 2     */
/* [ r 0 0 0 0 ]    [ a 0 0 0 0 ]   [ b 0 0 0 0 ] */
/* [ r r 0 0 0 ]    [ a a 0 0 0 ]   [ b b 0 0 0 ] */
/* [ r r r 0 0 ] := [ 0 a a 0 0 ] * [ 0 b b 0 0 ] */
/* [ 0 r r r 0 ]    [ 0 0 a a 0 ]   [ 0 0 b b 0 ] */
/* [ 0 0 r r r ]    [ 0 0 0 a a ]   [ 0 0 0 b b ] */

/* ex4) dim = 5, upper_dim = 1, lower_dim = 0     */
/*            -> upper_dim = 2, lower_dim = 0     */
/* [ r r r 0 0 ]    [ a a 0 0 0 ]   [ b b 0 0 0 ] */
/* [ 0 r r r 0 ]    [ 0 a a 0 0 ]   [ 0 b b 0 0 ] */
/* [ 0 0 r r r ] := [ 0 0 a a 0 ] * [ 0 0 b b 0 ] */
/* [ 0 0 0 r r ]    [ 0 0 0 a a ]   [ 0 0 0 b b ] */
/* [ 0 0 0 0 r ]    [ 0 0 0 0 a ]   [ 0 0 0 0 b ] */

/* ret := a * b */
void mul_mpfmatrix_mpfbmat_mpfbmat(MPFMatrix ret, MPFBMatrix a, MPFBMatrix b)
{
	long int i, j, k;
	mpf_t tmp, tmp1;

	if((ret->row_dim != a->dim) || (ret->row_dim != b->dim) || (ret->col_dim != a->dim) || (ret->col_dim != b->dim))
	{
		fprintf(stderr, "ERROR: mul_mpfmatrix_mpfbmat_mpfbmat: Mismatch dimension(s): (ret, a, b)->row_dim = (%ld, %ld, %ld), (ret, a, b)->col_dim = (%ld, %ld, %ld)\n", ret->row_dim, a->dim, b->dim, ret->col_dim, a->dim, b->dim);
		return;
	}

	// Initialize
	mpf_init2(tmp, ret->prec);
	mpf_init2(tmp1, ret->prec);

	for(i = 0; i < ret->row_dim; i++)
	{
		for(j = 0; j < ret->col_dim; j++)
		{
			mpf_set_ui(tmp, 0UL);
			for(k = 0; k < a->dim; k++)
			{
				//tmp += get_mpfbmatrix_ij(a, i, k) * get_mpfbmatrix_ij(b, k, j);
				mpf_mul(tmp1, get_mpfbmatrix_ij(a, i, k), get_mpfbmatrix_ij(b, k, j));
				mpf_add(tmp, tmp, tmp1);
			}

			set_mpfmatrix_ij(ret, i, j, tmp);
		}
	}

	// clear
	mpf_clear(tmp);
	mpf_clear(tmp1);
}

/* ret := a * b */
void mul_mpfmatrix_mpfmat_mpfbmat(MPFMatrix ret, MPFMatrix a, MPFBMatrix b)
{
	long int i, j, k;
	mpf_t tmp, tmp1;

	if((ret->row_dim != a->row_dim) || (ret->row_dim != b->dim) || (ret->col_dim != a->col_dim) || (ret->col_dim != b->dim))
	{
		fprintf(stderr, "ERROR: mul_mpfmatrix_mpfmat_mpfbmat: Mismatch dimension(s): (ret, a, b)->row_dim = (%ld, %ld, %ld), (ret, a, b)->col_dim = (%ld, %ld, %ld)\n", ret->row_dim, a->row_dim, b->dim, ret->col_dim, a->col_dim, b->dim);
		return;
	}

	// Initialize
	mpf_init2(tmp, ret->prec);
	mpf_init2(tmp1, ret->prec);

	for(i = 0; i < ret->row_dim; i++)
	{
		for(j = 0; j < ret->col_dim; j++)
		{
			mpf_set_ui(tmp, 0UL);
			for(k = 0; k < a->col_dim; k++)
			{
				//tmp += get_mpfmatrix_ij(a, i, k) * get_mpfbmatrix_ij(b, k, j);
				mpf_mul(tmp1, get_mpfmatrix_ij(a, i, k), get_mpfbmatrix_ij(b, k, j));
				mpf_add(tmp, tmp, tmp1);
			}

			set_mpfmatrix_ij(ret, i, j, tmp);
		}
	}

	// clear
	mpf_clear(tmp);
	mpf_clear(tmp1);
}

/* ret := a * b */
void mul_mpfmatrix_mpfbmat_mpfmat(MPFMatrix ret, MPFBMatrix a, MPFMatrix b)
{
	long int i, j, k;
	mpf_t tmp, tmp1;

	if((ret->row_dim != a->dim) || (ret->row_dim != b->row_dim) || (ret->col_dim != a->dim) || (ret->col_dim != b->col_dim))
	{
		fprintf(stderr, "ERROR: mul_mpfmatrix_mpfbmat_mpfmat: Mismatch dimension(s): (ret, a, b)->row_dim = (%ld, %ld, %ld), (ret, a, b)->col_dim = (%ld, %ld, %ld)\n", ret->row_dim, a->dim, b->row_dim, ret->col_dim, a->dim, b->col_dim);
		return;
	}

	// Initialize
	mpf_init2(tmp, ret->prec);
	mpf_init2(tmp1, ret->prec);

	for(i = 0; i < ret->row_dim; i++)
	{
		for(j = 0; j < ret->col_dim; j++)
		{
			mpf_set_ui(tmp, 0UL);
			for(k = 0; k < a->dim; k++)
			{
				//tmp += get_mpfbmatrix_ij(a, i, k) * get_mpfmatrix_ij(b, k, j);
				mpf_mul(tmp1, get_mpfbmatrix_ij(a, i, k), get_mpfmatrix_ij(b, k, j));
				mpf_add(tmp, tmp, tmp1);
			}

			set_mpfmatrix_ij(ret, i, j, tmp);
		}
	}

	// clear
	mpf_clear(tmp);
	mpf_clear(tmp1);

}

/* ret := a */
void subst_mpfbmatrix(MPFBMatrix ret, MPFBMatrix a)
{
	long int dim, upper_dim, lower_dim;
	long int i, j, jmin, jmax;

	if((ret->dim != a->dim) || (ret->upper_dim != a->upper_dim) || (ret->lower_dim != a->lower_dim))
	{
		fprintf(stderr, "ERROR: subst_mpfbmatrix: Mismatch dimension(s): (ret, a)->dim = (%ld, %ld), (ret, a)->upper_dim = (%ld, %ld), (ret, a)->lower_dim = (%ld, %ld)\n", ret->dim, a->dim, ret->upper_dim, a->upper_dim, ret->lower_dim, a->lower_dim);
		return;
	}

	dim = ret->dim;
	upper_dim = ret->upper_dim;
	lower_dim = ret->lower_dim;

	for(i = 0; i < dim; i++)
	{
		jmin = i - lower_dim;
		if(jmin < 0)
			jmin = 0;
		jmax = i + upper_dim + 1;
		if(jmax > dim)
			jmax = dim;

		for(j = jmin; j < jmax; j++)
			set_mpfbmatrix_ij(ret, i, j, get_mpfbmatrix_ij(a, i, j));
	}
}

/* ret := (DBMatrix)a */
void subst_dbmatrix_mpfbmat(DBMatrix ret, MPFBMatrix a)
{
	long int dim, upper_dim, lower_dim;
	long int i, j, jmin, jmax;

	if((ret->dim != a->dim) || (ret->upper_dim != a->upper_dim) || (ret->lower_dim != a->lower_dim))
	{
		fprintf(stderr, "ERROR: subst_dbmatrix_mpfbmat: Mismatch dimension(s): (ret, a)->dim = (%ld, %ld), (ret, a)->upper_dim = (%ld, %ld), (ret, a)->lower_dim = (%ld, %ld)\n", ret->dim, a->dim, ret->upper_dim, a->upper_dim, ret->lower_dim, a->lower_dim);
		return;
	}

	dim = ret->dim;
	upper_dim = ret->upper_dim;
	lower_dim = ret->lower_dim;

	for(i = 0; i < dim; i++)
	{
		jmin = i - lower_dim;
		if(jmin < 0)
			jmin = 0;
		jmax = i + upper_dim + 1;
		if(jmax > dim)
			jmax = dim;

		for(j = jmin; j < jmax; j++)
			set_dbmatrix_ij(ret, i, j, (double)mpf_get_d(get_mpfbmatrix_ij(a, i, j)));
	}
}

/* ret := (MPFBMatrix)a */
void subst_mpfbmatrix_dbmat(MPFBMatrix ret, DBMatrix a)
{
	long int dim, upper_dim, lower_dim;
	long int i, j, jmin, jmax;

	if((ret->dim != a->dim) || (ret->upper_dim != a->upper_dim) || (ret->lower_dim != a->lower_dim))
	{
		fprintf(stderr, "ERROR: subst_mpfbmatrix_dbmat: Mismatch dimension(s): (ret, a)->dim = (%ld, %ld), (ret, a)->upper_dim = (%ld, %ld), (ret, a)->lower_dim = (%ld, %ld)\n", ret->dim, a->dim, ret->upper_dim, a->upper_dim, ret->lower_dim, a->lower_dim);
		return;
	}

	dim = ret->dim;
	upper_dim = ret->upper_dim;
	lower_dim = ret->lower_dim;

	for(i = 0; i < dim; i++)
	{
		jmin = i - lower_dim;
		if(jmin < 0)
			jmin = 0;
		jmax = i + upper_dim + 1;
		if(jmax > dim)
			jmax = dim;

		for(j = jmin; j < jmax; j++)
			set_mpfbmatrix_ij_d(ret, i, j, (double)get_dbmatrix_ij(a, i, j));
	}
}

/* ret := (MPFMatrix)a */
void subst_mpfmatrix_mpfbmat(MPFMatrix ret, MPFBMatrix a)
{
	long int dim, upper_dim, lower_dim;
	long int i, j, jmin, jmax;

	if((ret->row_dim != a->dim) || (ret->col_dim != a->dim))
	{
		fprintf(stderr, "ERROR: subst_mpfmatrix_mpfbmat: Mismatch dimension(s): (ret, a)->row_dim = (%ld, %ld), (ret, a)->col_dim = (%ld, %ld)\n", ret->row_dim, a->dim, ret->col_dim, a->dim);
		return;
	}

	// ret := 0
	set0_mpfmatrix(ret);

	dim = ret->row_dim;
	upper_dim = a->upper_dim;
	lower_dim = a->lower_dim;

	for(i = 0; i < dim; i++)
	{
		jmin = i - lower_dim;
		if(jmin < 0)
			jmin = 0;
		jmax = i + upper_dim + 1;
		if(jmax > dim)
			jmax = dim;

		for(j = jmin; j < jmax; j++)
			set_mpfmatrix_ij(ret, i, j, get_mpfbmatrix_ij(a, i, j));
	}
}
#endif
