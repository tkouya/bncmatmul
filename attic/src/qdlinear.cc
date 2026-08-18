/********************************************************************************/
/* qdlinear.cc:Double-double and Quadruple precision Linear Computation Library */
/* Copyright (C) 2015 Tomonori Kouya                                            */
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
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "qd/qd_real.h"

#include "ddlinear.h"

#ifdef __cplusplus
//extern "C" {
#endif // __cplusplus

// QD vector

// Use set_qdvector_i as functions
#ifdef USE_QDLINEAR_FUNCTIONS
#ifndef __cplusplus
// set element
//void set_qdvector_i(DDVector vec, long int index, dd_real value)
void set_qdvector_i(QDVector vec, long int index, double value[QDSIZE])
{
	if((index < 0) || (index >= vec->dim))
	{
		fprintf(stderr, "ERROR: illegal element index! (index = %ld)\n", index);
		return;
	}

//	*(vec->element + index * QDSIZE)     = value[0];
//	*(vec->element + index * QDSIZE + 1) = value[1];
//	*(vec->element + index * QDSIZE + 2) = value[2];
//	*(vec->element + index * QDSIZE + 3) = value[3];
	SET_QDVECTOR_I(vec, index, value);
}

// set a zero vector
//void set0_qdvector(QDVector vec)
void set0_qdvector(QDVector vec)
{
	long int index;

	for(index = 0; index < vec->dim; index++)
		SET0_QDVECTOR_I(vec, index);
}
#endif // __cplusplus
#endif // USE_QDLINEAR_FUNCTIONS

// initialize qdvector
QDVector init_qdvector(long int dim)
{
	long int index;
	QDVector ret = NULL;

	// callocation
	ret = (qdvector *)malloc(sizeof(qdvector));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one QDVector\n");
		return ret;
	}

	ret->dim = dim;
#ifdef __cplusplus
	ret->element = (qd_real *)calloc(dim, sizeof(qd_real));
//	ret->element = new qd_real[dim];
#else // __cplusplus
	ret->element = (double *)calloc(dim, sizeof(double) * QDSIZE);
#endif // __cplusplus
	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate QDVector(dim = %ld)\n", dim);
		free(ret);
		return NULL;
	}

	// set all zeros 
	for(index = 0; index < dim; index++)
	{
#ifdef __cplusplus
		ret->element[index] = (qd_real)0.0;
		//ret->element[index] = (qd_real)((int)(index + 1));
		//ret->element[index] = sqrt(ret->element[index]);
#else // __cplusplus
		*(ret->element + index * QDSIZE)     = 0.0;
		*(ret->element + index * QDSIZE + 1) = 0.0;
		*(ret->element + index * QDSIZE + 2) = 0.0;
		*(ret->element + index * QDSIZE + 3) = 0.0;
#endif // __cplusplus
	}

	return ret;
}

// free qdvector
void free_qdvector(QDVector vec)
{
	free(vec->element);
	free(vec);
}

// print ddvector
void print_qdvector(QDVector vec)
{
	long int index;

	for(index = 0; index < vec->dim; index++)
	{
		printf("%4ld: ", index);
#ifdef __cplusplus
		std::cout << (vec->element[index]).to_string(62) << "\n";
#else // __cplusplus
		//c_qd_write((vec->element + index * QDSIZE));
		c_qd_write(GET_QDVECTOR_I(vec, index));
#endif // __cplusplus
	}
}

/*************************************************/
/* Vector Calculations for QDVector               */
/*
void add_qdvector(QDVector c, QDVector a, QDVector b)
void add2_qdvector(QDVector c, QDVector a)
void sub_qdvector(QDVector c, QDVector a, QDVector b)
void sub2_qdvector(QDVector c, DVector a)
void cmul_qdvector(QDVector c, double val[QDSIZE], QDVector a)
void cmul2_qdvector(QDVector c, double val[QDSIZE])
void add_cmul_qdvector(QDVector c, QDVector a, double val[QDSIZE], QDVector b)
double ip_qdvector(QDVector a, QDVector b)
double norm1_qdvector(QDVector a)
double norm2_qdvector(QDVector a)
double normi_qdvector(QDVector a)
void subst_qdvector(QDVector c, QDVector a)
*/
/*************************************************/
/* c = a + b */
void add_qdvector(QDVector c, QDVector a, QDVector b)
{
	long int i;
	static double tmp[QDSIZE];

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_qdvector\n");
		return;
	}

	for(i = 0; i < c->dim; i++)
	{
#ifdef __cplusplus
		c->element[i] = a->element[i] + b->element[i];
#else // __cplusplus
		rqd_add(tmp, get_qdvector_i(a, i),  get_qdvector_i(b, i));
		set_qdvector_i(c, i, tmp);
#endif // __cplusplus
	}
}

/* c += a */
void add2_qdvector(QDVector c, QDVector a)
{
	long int i;
	static double tmp[QDSIZE];

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: add2_qdvector\n");
		return;
	}

	for(i = 0; i < c->dim; i++)
	{
#ifdef __cplusplus
		c->element[i] += a->element[i];
#else // __cplusplus
		rqd_add(tmp, get_qdvector_i(c, i), get_qdvector_i(a, i));
		set_qdvector_i(c, i, tmp);
#endif // __cplusplus
	}

}

/* c = a - b */
void sub_qdvector(QDVector c, QDVector a, QDVector b)
{
	long int i;
	static double tmp[QDSIZE];

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_qdvector\n");
		return;
	}

	for(i = 0; i < c->dim; i++)
	{
#ifdef __cplusplus
		c->element[i] = a->element[i] - b->element[i];
#else // __cplusplus
		rqd_sub(tmp, get_qdvector_i(a, i), get_qdvector_i(b, i));
		set_qdvector_i(c, i, tmp);
#endif // __cplusplus
	}

}

/* c -= a */
void sub2_qdvector(QDVector c, QDVector a)
{
	long int i;
	static double tmp[QDSIZE];

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: sub2_qdvector\n");
		return;
	}

	for(i = 0; i < c->dim; i++)
	{
#ifdef __cplusplus
		c->element[i] -= a->element[i];
#else // __cplusplus
		rqd_sub(tmp, get_qdvector_i(c, i), get_qdvector_i(a, i));
		set_qdvector_i(c, i, tmp);
#endif // __cplusplus
	}

}

/* c = val * a */
#ifdef __cplusplus
void cmul_qdvector(QDVector c, qd_real val, QDVector a)
#else // __cplusplus
void cmul_qdvector(QDVector c, double val[QDSIZE], QDVector a)
#endif // __cplusplus
{
	long int i;
	static double tmp[QDSIZE];

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: cmul_qdvector\n");
		return;
	}

	for(i = 0; i < c->dim; i++)
	{
#ifdef __cplusplus
		c->element[i] = val * a->element[i];
#else // __cplusplus
		rqd_mul(tmp, val, get_qdvector_i(a, i));
		set_qdvector_i(c, i, tmp);
#endif // __cplusplus
	}

}

/* c *= val */
#ifdef __cplusplus
void cmul2_qdvector(QDVector c, qd_real val)
#else // __cplusplus
void cmul2_qdvector(QDVector c, double val[QDSIZE])
#endif // __cplusplus
{
	long int i;
	static double tmp[QDSIZE];

	for(i = 0; i < c->dim; i++)
	{
#ifdef __cplusplus
		c->element[i] *= val;
#else // __cplusplus
		rqd_mul(tmp, val, get_qdvector_i(c, i));
		set_qdvector_i(c, i, tmp);
#endif // __cplusplus
	}

}

/* c = a + val * b */
#ifdef __cplusplus
void add_cmul_qdvector(QDVector c, QDVector a, qd_real val, QDVector b)
#else // __cplusplus
void add_cmul_qdvector(QDVector c, QDVector a, double val[QDSIZE], QDVector b)
#endif // __cplusplus
{
	long int i;
	static double tmp[QDSIZE];

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_cmul_qdvector\n");
		return;
	}

	for(i = 0; i < c->dim; i++)
	{
#ifdef __cplusplus
		c->element[i] = a->element[i] + val * b->element[i];
#else // __cplusplus
		rqd_mul(tmp, val, get_qdvector_i(b, i));
		rqd_add(tmp, tmp, get_qdvector_i(a, i));
		set_qdvector_i(c, i, tmp);
#endif // __cplusplus
	}
}

/* (a, b) */
#ifdef __cplusplus
void ip_qdvector(qd_real &ret, QDVector a, QDVector b)
#else // __cplusplus
void ip_qdvector(double ret[QDSIZE], QDVector a, QDVector b)
#endif // __cplusplus
{
	static double tmp[QDSIZE];
	long int i;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: ip_qdvector\n");
		return;
	}

#ifdef __cplusplus
	ret = 0.0;
#else // __cplusplus
	set0_dd(ret);
#endif // __cplusplus
	for(i = 0; i < a->dim; i++)
	{
#ifdef __cplusplus
		ret += a->element[i] * b->element[i];
#else // __cplusplus
		rqd_mul(tmp, get_qdvector_i(a, i), get_qdvector_i(b, i));
		rqd_add(ret, ret, tmp);
#endif // __cplusplus
	}

	return;
}

/* c := a */
void subst_qdvector(QDVector c, QDVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
#ifdef __cplusplus
		c->element[i] = a->element[i];
#else // __cplusplus
		set_qdvector_i(c, i, get_qdvector_i(a, i));
#endif // __cplusplus
}


/* ||a||_1 */
#ifdef __cplusplus
//void norm1_qdvector(qd_real &ret, QDVector a)
void norm1_qdvector(qd_real *ret, QDVector a)
#else // __cplusplus
void norm1_qdvector(double ret[QDSIZE], QDVector a)
#endif // __cplusplus
{
	long int i;
#ifdef __cplusplus
	*ret = 0.0;
#else // __cplusplus
	static double tmp[QDSIZE];

	set0_dd(ret);
#endif // __cplusplus
	for(i = 0; i < a->dim; i++)
	{
#ifdef __cplusplus
		*ret += abs(a->element[i]);
#else // __cplusplus
		rqd_abs(tmp, get_qdvector_i(a, i));
		rqd_add(ret, ret, tmp);
#endif // __cplusplus
	}

	return;
}


/* ||a||_infty */
#ifdef __cplusplus
//void normi_qdvector(qd_real &ret, QDVector a)
void normi_qdvector(qd_real *ret, QDVector a)
#else // __cplusplus
void normi_qdvector(double ret[QDSIZE], QDVector a)
#endif // __cplusplus
{
	long int i;
#ifdef __cplusplus
	qd_real tmp;

	*ret = 0.0;
	tmp = 0.0;
#else // __cplusplus
	static double tmp[QDSIZE];

	rqd_abs(ret, get_qdvector_i(a, 0));
#endif // __cplusplus
	for(i = 1; i < a->dim; i++)
	{
#ifdef __cplusplus
		tmp = abs(a->element[i]);
		if(*ret < tmp)
			*ret = tmp;
#else // __cplusplus
		rqd_abs(tmp, get_qdvector_i(a, i));
		if(rqd_cmp(ret, tmp) < 0)
			rqd_set(ret, tmp);
#endif // __cplusplus
	}

	return;
}

// Euclid norm
#ifdef __cplusplus
//void norm2_qdvector(qd_real &ret, QDVector vec)
void norm2_qdvector(qd_real *ret, QDVector vec)
#else // __cplusplus
void norm2_qdvector(double ret[QDSIZE], QDVector vec)
#endif // __cplusplus
{
	long int i, dim;
#ifdef __cplusplus
	qd_real tmp;

	*ret = (qd_real)0.0;
	tmp = (qd_real)0.0;
#else // __cplusplus
	static double tmp[QDSIZE];

	c_qd_copy_d((double)0.0, tmp);
	c_qd_copy_d((double)0.0, ret);
#endif // __cplusplus

	dim = vec->dim;

	for(i = 0; i < dim ; i++)
	{
#ifdef __cplusplus
		//tmp += vec->element[i] * vec->element[i];
		tmp += qd_real::accurate_mul(vec->element[i], vec->element[i]);
#else // __cplusplus
		c_qd_sqr(GET_QDVECTOR_I(vec, i), tmp);
		c_qd_add(tmp, ret, ret);
#endif // __cplusplus
	}

#ifdef __cplusplus
	//*ret = (qd_real)sqrt(tmp);
	tmp = (qd_real)sqrt(tmp);
	*ret = tmp;
#else // __cplusplus
	c_qd_sqrt(ret, tmp);
	c_qd_copy(tmp, ret);
#endif // __cplusplus
}

// QD matrix

#ifdef USE_QDLINEAR_FUNCTIONS
// set element
//void set_ddvector_i(DDVector vec, long int index, dd_real value)
void set_qdmatrix_ij(QDMatrix mat, long int row_index, long int col_index, double value[QDSIZE])
{
	if((row_index < 0) || (col_index < 0) || (row_index >= mat->row_dim) || (col_index >= mat->col_dim))
	{
		fprintf(stderr, "ERROR: illegal element index! (row_index, col_index = %ld, %ld)\n", row_index, col_index);
		return;
	}

#ifdef __cplusplus
#else // __cplusplus
//	*(vec->element + index * DDSIZE)     = value[0];
//	*(vec->element + index * DDSIZE + 1) = value[1];
	SET_QDMATRIX_IJ(mat, row_index, col_index, value);
#endif // __cplusplus
}
#endif // USE_QDLINEAR_FUNCTIONS

// set a zero matrix
//void set0_qdmatrix(QDMatrix mat)
void set0_qdmatrix(QDMatrix mat)
{
	long int i, j;

	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
		{
#ifdef __cplusplus
			mat->element[i * mat->col_dim + j] = 0.0;
#else // __cplusplus
			SET0_QDMATRIX_IJ(mat, i, j);
#endif // __cplusplus
		}
	}
}

// initialize qdmatrix
QDMatrix init_qdmatrix(long int row_dim, long int col_dim)
{
	long int row_index, col_index;
	QDMatrix ret = NULL;

	// callocation
	ret = (qdmatrix *)malloc(sizeof(qdmatrix));
	if(ret == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate one qdmatrix\n");
		return ret;
	}

	ret->row_dim = row_dim;
	ret->col_dim = col_dim;
#ifdef __cplusplus
	ret->element = (qd_real *)calloc(row_dim * col_dim, sizeof(qd_real));
#else // __cplusplus
	ret->element = (double *)calloc(row_dim * col_dim, sizeof(double) * QDSIZE);
#endif // __cplusplus
	if(ret->element == NULL)
	{
		fprintf(stderr, "ERROR: cannot allocate qdmatrix(row_dim, col_dim = %ld, %ld)\n", row_dim, col_dim);
		free(ret);
		return NULL;
	}

	// set all zeros 
	for(row_index = 0; row_index < row_dim; row_index++)
	{
		for(col_index = 0; col_index < col_dim; col_index++)
		{
#ifdef __cplusplus
			ret->element[row_index * col_dim + col_index] = 0.0;
#else // __cplusplus
			//*(ret->element + index * DDSIZE)     = 0.0;
			//*(ret->element + index * DDSIZE + 1) = 0.0;
			SET0_QDMATRIX_IJ(ret, row_index, col_index);
#endif // __cplusplus
		}
	}

	return ret;
}

// free qdmatrix
void free_qdmatrix(QDMatrix mat)
{
	free(mat->element);
	free(mat);
}

// print qdmatrix
void print_qdmatrix(QDMatrix mat)
{
	long int row_index, col_index;

	for(row_index = 0; row_index < mat->row_dim; row_index++)
	{
		for(col_index = 0; col_index < mat->col_dim; col_index++)
		{
			printf("%ld, %ld: ", row_index, col_index);
#ifdef __cplusplus
			std::cout << (mat->element[row_index * mat->col_dim + col_index]).to_string() << "\n";
#else // __cplusplus
	//		c_qd_write((vec->element + index * QDSIZE));
			c_qd_write(GET_QDMATRIX_IJ(mat, row_index, col_index));
#endif // __cplusplus
		}
	}
}

/*************************************************/
/* Matrix Caluculations for QDMatrix            */
/*
void normf_qdmatrix(double ret[QDSIZE], QDMatrix mat)
void norm1_qdmatrix(double ret[QDSIZE], QDMatrix mat)
void normi_qdmatrix(double ret[QDSIZE], QDMatrix mat)
void add_qdmatrix(QDMatrix c, QDMatrix a, QDMatrix b);
void sub_qdmatrix(QDMatrix c, QDMatrix a, QDMatrix b);
void mul_qdmatrix(QDMatrix c, QDMatrix a, QDMatrix b);
void mul_qdmatrix_qdvec(QDVector v, QDMatrix a, QDVector vb)
void mul_qdmatrixt_qdvec(QDVector v, QDMatrix a, QDVector vb)
void transpose_qdmatrix(QDMatrix c, QDMatrix a);
void inv_qdmatrix(QDMatrix a);
void subst_mpfmatrux(QDMatrix c, QDMatrix a);
*/
/*************************************************/

/* Infinity Norm of Matrix */
#ifdef __cplusplus
void normi_qdmatrix(qd_real &ret, QDMatrix mat)
#else // __cplusplus
void normi_qdmatrix(double ret[QDSIZE], QDMatrix mat)
#endif // __cplusplus
{
	long int i, j;
#ifdef __cplusplus
	qd_real sum;
#else // __cplusplus
	static double tmp[QDSIZE], sum[QDSIZE];

	set0_qd(ret);
#endif // __cplusplus
	for(i = 0; i < mat->row_dim; i++)
	{
#ifdef __cplusplus
		sum = 0.0;
		for(j = 0; j < mat->col_dim; j++)
			sum += abs(mat->element[i * mat->col_dim + j]);

		if(ret < sum)
			ret = sum;

#else // __cplusplus
		set0_qd(sum);
		for(j = 0; j < mat->col_dim; j++)
		{
			rqd_abs(tmp, get_qdmatrix_ij(mat, i, j));
			rqd_add(sum, sum, tmp);
		}
		if(rqd_cmp(ret, sum) < 0)
			rqd_set(ret, sum);
#endif // __cplusplus
	}

	return;
}

/* 1 Norm of Matrix */
#ifdef __cplusplus
void norm1_qdmatrix(qd_real &ret, QDMatrix mat)
#else // __cplusplus
void norm1_qdmatrix(double ret[QDSIZE], QDMatrix mat)
#endif // __cplusplus
{
	long int i, j;
#ifdef __cplusplus
	qd_real sum;
#else // __cplusplus
	static double tmp[QDSIZE], sum[QDSIZE];

	set0_qd(ret);
#endif // __cplusplus
	for(j = 0; j < mat->col_dim; j++)
	{
#ifdef __cplusplus
		sum = 0.0;
		for(i = 0; i < mat->row_dim; i++)
			sum += abs(mat->element[i * mat->col_dim + j]);

		if(ret < sum)
			ret = sum;

#else // __cplusplus
		set0_qd(sum);
		for(i = 0; i < mat->row_dim; i++)
		{
			rqd_abs(tmp, get_qdmatrix_ij(mat, i, j));
			rqd_add(sum, sum, tmp);
		}
		if(rqd_cmp(ret, sum) < 0)
			rqd_set(ret, sum);
#endif // __cplusplus
	}

	return;
}

/* c := a + b */
void add_qdmatrix(QDMatrix c, QDMatrix a, QDMatrix b)
{
	long int i, j, row_dim, col_dim, index;
	static double tmp[QDSIZE];

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_qdmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_qdmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
#ifdef __cplusplus
			index = i * row_dim + j;
			c->element[index] = a->element[index] + b->element[index];
#else // __cplusplus
			rqd_add(tmp, get_qdmatrix_ij(a, i, j), get_qdmatrix_ij(b, i, j));
			set_qdmatrix_ij(c, i, j, tmp);
#endif // __cplusplus
		}
	}
}

/* c := a - b */
void sub_qdmatrix(QDMatrix c, QDMatrix a, QDMatrix b)
{
	long int i, j, row_dim, col_dim, index;
	static double tmp[QDSIZE];

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_qdmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_qdmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
#ifdef __cplusplus
			index = i * row_dim + j;
			c->element[index] = a->element[index] - b->element[index];
#else // __cplusplus
			rqd_sub(tmp, get_qdmatrix_ij(a, i, j), get_qdmatrix_ij(b, i, j));
			set_qdmatrix_ij(c, i, j, tmp);
#endif // __cplusplus
		}
	}
}

/* c := sc * a */
#ifdef __cplusplus
void cmul_qdmatrix(QDMatrix c, qd_real sc, QDMatrix a)
#else // __cplusplus
void cmul_qdmatrix(QDMatrix c, double sc[QDSIZE], QDMatrix a)
#endif // __cplusplus
{
	long int i, j, row_dim, col_dim;
#ifdef __cplusplus
#else // __cplusplus
	static double tmp[QDSIZE];
#endif // __cplusplus

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_qdmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_qdmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
#ifdef __cplusplus
			c->element[i * c->row_dim + j] = sc * a->element[i * a->col_dim + j];
#else // __cplusplus
			rqd_mul(tmp, sc, get_qdmatrix_ij(a, i, j));
			set_qdmatrix_ij(c, i, j, tmp);
#endif // __cplusplus
		}
	}
}

/* c = a^T */
void transpose_qdmatrix(QDMatrix c, QDMatrix a)
{
	long int i, j;

	/* Check Dimensions */
	if((c->row_dim != a->col_dim) || (c->col_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: transpose_qdmatrix\n");
		return;
	}
	
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
		{
#ifdef __cplusplus
			c->element[i * c->col_dim + j] = a->element[j * a->row_dim + j];
#else // __cplusplus
			set_qdmatrix_ij(c, i, j, get_qdmatrix_ij(a, j, i));
#endif // __cplusplus
		}
	}
}

/* c := a */
void subst_qdmatrix(QDMatrix c, QDMatrix a)
{
	long int i, j;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_qdmatrix\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
#ifdef __cplusplus
			c->element[i * c->col_dim + j] = a->element[i * a->col_dim + j];
#else // __cplusplus
			set_qdmatrix_ij(c, i, j, get_qdmatrix_ij(a, i, j));
#endif // __cplusplus
		}
	}
}

/* c := I */
void setI_qdmatrix(QDMatrix c)
{
	long int i, j;
#ifdef __cplusplus
#else // __cplusplus
	static double tmp0[QDSIZE], tmp1[QDSIZE];
	rqd_set_ui(tmp0, 0UL);
	rqd_set_ui(tmp1, 1UL);
#endif // __cplusplus

	for(i = 0; i < c->row_dim; i++)
	{
#ifdef __cplusplus
		for(j = 0; j < c->col_dim; j++)
			c->element[i * c->col_dim + j] = 0.0;
		if(i < c->col_dim)
			c->element[i * c->col_dim + i] = 1.0;
#else // __cplusplus
		for(j = 0; j < c->col_dim; j++)
			set_qdmatrix_ij(c, i, j, tmp0);
		if(i < c->col_dim)
			set_qdmatrix_ij(c, i, i, tmp1);
#endif // __cplusplus
	}
}

/* v := a * vb */
void mul_qdmatrix_qdvec(QDVector v, QDMatrix a, QDVector vb)
{
	long int i, j;
#ifdef __cplusplus
	qd_real tmp;
#else // __cplusplus
	static double tmp[QDSIZE], tmp1[QDSIZE];
#endif // __cplusplus

	/* Check Dimension */
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: mul_qdmatrix_qdvec\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
#ifdef __cplusplus
		tmp = 0.0;
		for(j = 0; j < a->col_dim; j++)
			tmp += a->element[i * a->col_dim + j] * vb->element[j];

		v->element[i] = tmp;
#else // __cplusplus
		set0_qd(tmp);
		for(j = 0; j < a->col_dim; j++)
		{
			rqd_mul(tmp1, get_qdmatrix_ij(a, i, j), get_qdvector_i(vb, j));
			rqd_add(tmp, tmp, tmp1);
		}
		set_qdvector_i(v, i, tmp);
#endif // __cplusplus
	}
}

/* v := a^T * vb */
void mul_qdmatrixt_qdvec(QDVector v, QDMatrix a, QDVector vb)
{
	long int i, j;
#ifdef __cplusplus
	qd_real tmp;
#else // __cplusplus
	static double tmp[QDSIZE], tmp1[QDSIZE];
#endif // __cplusplus

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_qdmatrixt_qdvec\n");
		return;
	}

	for(i = 0; i < a->col_dim; i++)
	{
#ifdef __cplusplus
		tmp = 0.0;
		for(j = 0; j < a->row_dim; j++)
			tmp += a->element[j * a->row_dim + i] * vb->element[j];

		v->element[i] = tmp;
#else // __cplusplus
		set0_qd(tmp);
		for(j = 0; j < a->row_dim; j++)
		{
			rqd_mul(tmp1, get_qdmatrix_ij(a, j, i), get_qdvector_i(vb, j));
			rqd_add(tmp, tmp, tmp1);
		}
		set_qdvector_i(v, i, tmp);
#endif // __cplusplus
	}
}

/* a = a^(-1) */
/* square matrix only */
void inv_qdmatrix(QDMatrix a)
{
	long int i, j, k, dim;
#ifdef __cplusplus
	qd_real tmp, aii;
#else // __cplusplus
	static double tmp[QDSIZE], aii[QDSIZE];
#endif // __cplusplus

	/* Check Dimensions */
	if(a->row_dim != a->col_dim)
	{
		fprintf(stderr, "ERROR: inv_qdmatrix\n");
		return;
	}

	dim = a->row_dim;

	for(i = 0; i < dim; i++)
	{
#ifdef __cplusplus
		if(a->element[i * dim + i] == 0.0)
#else // __cplusplus
		if(rqd_cmp_ui(get_qdmatrix_ij(a, i, i), 0UL) == 0) 
#endif // __cplusplus
		{
			fprintf(stderr, "ERROR: inv_qdmatrix: Pivot(%ld,%ld) is zero.\n", i, i);
			return;
		}

#ifdef __cplusplus
		aii = 1.0 / a->element[i * dim + i];
		a->element[i * dim + i] = aii;
#else // __cplusplus
		rqd_ui_div(aii, 1UL, get_qdmatrix_ij(a, i, i));
		set_qdmatrix_ij(a, i, i, aii);
#endif // __cplusplus

		for(j = 0; j < i; j++)
		{
#ifdef __cplusplus
			//a->element[i * dim + j] *= aii;
			a->element[i * dim + j] = qd_real::accurate_mul(aii, a->element[i * dim + j]);
#else // __cplusplus
			rqd_mul(tmp, get_qdmatrix_ij(a, i, j), aii);
			set_qdmatrix_ij(a, i, j, tmp);
#endif // __cplusplus
		}
		for(j = i + 1; j < dim; j++)
		{
#ifdef __cplusplus
			//a->element[i * dim + j] *= aii;
			a->element[i * dim + j] = qd_real::accurate_mul(aii, a->element[i * dim + j]);
#else // __cplusplus
			rqd_mul(tmp, get_qdmatrix_ij(a, i, j), aii);
			set_qdmatrix_ij(a, i, j, tmp);
#endif // __cplusplus
		}

		for(j = 0; j < i; j++)
		{
			for(k = 0; k < i; k++)
			{
#ifdef __cplusplus
				//a->element[j * dim + k] -= a->element[j * dim + i] * a->element[i * dim + k];
				a->element[j * dim + k] -= qd_real::accurate_mul(a->element[j * dim + i], a->element[i * dim + k]);
#else // __cplusplus
				rqd_mul(tmp, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, k));
				rqd_sub(tmp, get_qdmatrix_ij(a, j, k), tmp);
				set_qdmatrix_ij(a, j, k, tmp);
#endif // __cplusplus
			}
			for(k = i + 1; k < dim; k++)
			{
#ifdef __cplusplus
				//a->element[j * dim + k] -= a->element[j * dim + i] * a->element[i * dim + k];
				a->element[j * dim + k] -= qd_real::accurate_mul(a->element[j * dim + i], a->element[i * dim + k]);
#else // __cplusplus
				rqd_mul(tmp, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, k));
				rqd_sub(tmp, get_qdmatrix_ij(a, j, k), tmp);
				set_qdmatrix_ij(a, j, k, tmp);
#endif // __cplusplus
			}
		}

		for(j = i + 1; j < dim; j++)
		{
			for(k = 0; k < i; k++)
			{
#ifdef __cplusplus
				//a->element[j * dim + k] -= a->element[j * dim + i] * a->element[i * dim + k];
				a->element[j * dim + k] -= qd_real::accurate_mul(a->element[j * dim + i], a->element[i * dim + k]);
#else // __cplusplus
				rqd_mul(tmp, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, k));
				rqd_sub(tmp, get_qdmatrix_ij(a, j, k), tmp);
				set_qdmatrix_ij(a, j, k, tmp);
#endif // __cplusplus
			}
			for(k = i + 1; k < dim; k++)
			{
#ifdef __cplusplus
				//a->element[j * dim + k] -= a->element[j * dim + i] * a->element[i * dim + k];
				a->element[j * dim + k] -= qd_real::accurate_mul(a->element[j * dim + i], a->element[i * dim + k]);
#else // __cplusplus
				rqd_mul(tmp, get_qdmatrix_ij(a, j, i), get_qdmatrix_ij(a, i, k));
				rqd_sub(tmp, get_qdmatrix_ij(a, j, k), tmp);
				set_qdmatrix_ij(a, j, k, tmp);
#endif // __cplusplus
			}
		}

		for(j = 0; j < i; j++)
		{
#ifdef __cplusplus
			//a->element[j * dim + i] = (-aii) * a->element[j * dim + i];
			a->element[j * dim + i] = qd_real::accurate_mul(-aii, a->element[j * dim + i]);
#else // __cplusplus
			rqd_neg(tmp, aii); /* tmp := -aii */
			rqd_mul(tmp, tmp, get_qdmatrix_ij(a, j, i));
			set_qdmatrix_ij(a, j, i, tmp);
#endif // __cplusplus
		}
		for(j = i + 1; j < dim; j++)
		{
#ifdef __cplusplus
			//a->element[j * dim + i] = (-aii) * a->element[j * dim + i];
			a->element[j * dim + i] = qd_real::accurate_mul(-aii, a->element[j * dim + i]);
#else // __cplusplus
			rqd_neg(tmp, aii); /* tmp := -aii */
			rqd_mul(tmp, tmp, get_qdmatrix_ij(a, j, i));
			set_qdmatrix_ij(a, j, i, tmp);
#endif // __cplusplus
		}
	}
}

// matrix multiplication
// ret := A * B
void mul_qdmatrix(QDMatrix ret, QDMatrix a, QDMatrix b)
{
	long int i, j, k;
	long int row_dim, col_dim, mid_dim;
#ifdef __cplusplus
	qd_real tmp, tmp1;
#else // __cplusplus
	static double tmp[QDSIZE];
#endif // __cplusplus

	/* dimension check */
	if((ret->row_dim != a->row_dim) || (ret->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: mul_qdmatrix: ret(%ld, %ld), a(%ld, %ld), b(%ld, %ld)\n", ret->row_dim, ret->col_dim, a->row_dim, a->col_dim, b->row_dim, b->col_dim);
		return;
	}

	row_dim = ret->row_dim;
	col_dim = ret->col_dim;
	mid_dim = a->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
#ifdef __cplusplus
			tmp = (qd_real)0.0;
			for(k = 0; k < mid_dim; k++)
			{
				//tmp = a->element[i * a->col_dim + k];
				//tmp = tmp * (b->element[k * b->col_dim + j]);
				//tmp1 = (a->element[i * a->col_dim + k]) * (b->element[k * b->col_dim + j]);
				tmp1 = qd_real::accurate_mul((qd_real &)(a->element[i * a->col_dim + k]), (qd_real &)(b->element[k * b->col_dim + j]));
				//ret->element[i * col_dim + j] += tmp;
				tmp = qd_real::ieee_add(tmp, tmp1);
			}

			ret->element[i * col_dim + j] = (qd_real)tmp;
#else // __cplusplus
			set0_qd(GET_QDMATRIX_IJ(ret, i, j));
			for(k = 0; k < mid_dim; k++)
			{
				c_qd_mul(GET_QDMATRIX_IJ(a, i, k), GET_QDMATRIX_IJ(b, k, j), tmp);
				c_qd_add(tmp, GET_QDMATRIX_IJ(ret, i, j), GET_QDMATRIX_IJ(ret, i, j));
			}
#endif // __cplusplus
		}
	}
}

// Frobenius norm
#ifdef __cplusplus
//void normf_qdmatrix(qd_real &ret, QDMatrix mat)
void normf_qdmatrix(qd_real *ret, QDMatrix mat)
#else // __cplusplus
void normf_qdmatrix(double ret[QDSIZE], QDMatrix mat)
#endif // __cplusplus
{
	long int i, dim;
#ifdef __cplusplus
	qd_real tmp, tmp1;

	*ret = 0.0;
	tmp  = (qd_real)0.0;
	tmp1 = (qd_real)0.0;
#else // __cplusplus
	static double tmp[QDSIZE];

	set0_qd(tmp);
	set0_qd(ret);
#endif // __cplusplus

	dim = mat->row_dim * mat->col_dim;

	for(i = 0; i < dim ; i++)
	{
#ifdef __cplusplus
		tmp1 = qd_real::accurate_mul(mat->element[i], mat->element[i]);
		tmp = qd_real::ieee_add(tmp,  tmp1);
#else // __cplusplus
		c_qd_sqr(GET_QDVECTOR_I(mat, i), tmp);
		c_qd_add(tmp, ret, ret);
#endif // __cplusplus
	}

#ifdef __cplusplus
	tmp = sqrt(tmp);
	*ret = tmp;
#else // __cplusplus
	c_qd_sqrt(ret, tmp);
	c_qd_copy(tmp, ret);
#endif // __cplusplus
}

#include "bnc.h"

#ifdef USE_GMP

// mpf to qd
#ifdef __cplusplus
//void mpf_get_qd(qd_real &ret, mpf_t val)
void mpf_get_qd(qd_real *ret, mpf_t val)
#else // __cplusplus
void mpf_get_qd(double ret[QDSIZE], mpf_t val)
#endif // __cplusplus
{
	static char str_qd[256];
	static mp_exp_t exp;

	mpf_get_str(str_qd, &exp, 10, 70, val);
	sprintf(str_qd, "%se%+-ld\n", str_qd, exp);
//	mpf_out_str(stdout, 10, 0, val); printf("\n");

#ifdef __cplusplus
	qd_real ret_org = *ret;
	ret_org.read(str_qd, ret_org);
#else // __cplusplus
	rqd_get_str(ret, str_qd);
#endif // __cplusplus
}

// qd to mpf
#ifdef __cplusplus
void mpf_set_qd(mpf_t ret, qd_real val)
#else // __cplusplus
void mpf_set_qd(mpf_t ret, double val[QDSIZE])
#endif // __cplusplus
{
	static char str_qd[256];

#ifdef __cplusplus
	val.write(str_qd, 10);
#else // __cplusplus
	rqd_set_str(str_qd, val);
	//printf("%s \n", str_qd);
#endif // __cplusplus

	mpf_set_str(ret, str_qd, 10);
}

/* Normwise relative error of vector */
#ifdef __cplusplus
void relerr_qdvector(qd_real &relerr, QDVector approx_vec, QDVector true_vec, int norm_type)
{
	qd_real norm_true_vec, norm_diff_vec;
#else // __cplusplus
void relerr_qdvector(double relerr[QDSIZE], QDVector approx_vec, QDVector true_vec, int norm_type)
{
	double norm_true_vec[QDSIZE], norm_diff_vec[QDSIZE];
#endif // __cplusplus
	QDVector diff_vec;

	diff_vec = init_qdvector(approx_vec->dim);

	// diff_vec := approx_vec - true_vec
	sub_qdvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
#ifdef __cplusplus
		// inifinity norm 
		case 0:
			normi_qdvector(&norm_diff_vec, diff_vec);
			normi_qdvector(&norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_qdvector(&norm_diff_vec, diff_vec);
			norm1_qdvector(&norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_qdvector(&norm_diff_vec, diff_vec);
			norm2_qdvector(&norm_true_vec, true_vec);
			break;

#else // __cplusplus

		// inifinity norm 
		case 0:
			normi_qdvector(norm_diff_vec, diff_vec);
			normi_qdvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_qdvector(norm_diff_vec, diff_vec);
			norm1_qdvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_qdvector(norm_diff_vec, diff_vec);
			norm2_qdvector(norm_true_vec, true_vec);
			break;
#endif // __cplusplus
	}

#ifdef __cplusplus
	if(norm_true_vec != 0.0)
		relerr = norm_diff_vec / norm_true_vec;
#else // __cplusplus
	if(rqd_cmp_ui(norm_true_vec, 0UL) != 0)
		rqd_div(relerr, norm_diff_vec, norm_true_vec);
#endif // __cplusplus

	free_qdvector(diff_vec);

	return;
}

/* Elementwise relative errors of vector */
#ifdef __cplusplus
void relerr_element_qdvector(qd_real &max_relerr, qd_real &min_relerr, qd_real &norm_relerr, QDVector approx_vec, QDVector true_vec, int norm_type)
{
	qd_real abs_true_vec, abs_diff_vec, norm_diff_vec, norm_true_vec;
#else // __cplusplus
void relerr_element_qdvector(double max_relerr[QDSIZE], double min_relerr[QDSIZE], double norm_relerr[QDSIZE], QDVector approx_vec, QDVector true_vec, int norm_type)
{
	double abs_true_vec[QDSIZE], abs_diff_vec[QDSIZE], norm_diff_vec[QDSIZE], norm_true_vec[QDSIZE];
#endif // __cplusplus
	long int i;
	QDVector diff_vec;

	diff_vec = init_qdvector(approx_vec->dim);

	// diff_vec := approx_vec - true_vec
	sub_qdvector(diff_vec, approx_vec, true_vec);

#ifdef __cplusplus
	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_qdvector(&norm_diff_vec, diff_vec);
			normi_qdvector(&norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_qdvector(&norm_diff_vec, diff_vec);
			norm1_qdvector(&norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_qdvector(&norm_diff_vec, diff_vec);
			norm2_qdvector(&norm_true_vec, true_vec);
			break;
	}

	norm_relerr = norm_diff_vec;
	if(norm_true_vec != 0.0)
		norm_relerr = norm_diff_vec / norm_true_vec;

	// relative errors of each elements
	max_relerr = 0.0;
	normi_qdvector(&min_relerr, diff_vec);
	for(i = 0; i < approx_vec->dim; i++)
	{
		abs_diff_vec = abs(diff_vec->element[i]);
		abs_true_vec = abs(true_vec->element[i]);
		if(abs_true_vec != 0.0)
			abs_diff_vec = abs_diff_vec / abs_true_vec;
		
		if(max_relerr < abs_diff_vec)
			max_relerr = abs_diff_vec;
		if(min_relerr > abs_diff_vec)
			min_relerr = abs_diff_vec;
	}

#else // __cplusplus
	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_qdvector(norm_diff_vec, diff_vec);
			normi_qdvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_qdvector(norm_diff_vec, diff_vec);
			norm1_qdvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_qdvector(norm_diff_vec, diff_vec);
			norm2_qdvector(norm_true_vec, true_vec);
			break;
	}

	rqd_set(norm_relerr, norm_diff_vec);
	if(rqd_cmp_ui(norm_true_vec, 0UL) != 0)
		rqd_div(norm_relerr, norm_diff_vec, norm_true_vec);

	// relative errors of each elements
	rqd_set_ui(max_relerr, 0UL);
	normi_qdvector(min_relerr, diff_vec);
	for(i = 0; i < approx_vec->dim; i++)
	{
		rqd_abs(abs_diff_vec, get_qdvector_i(diff_vec, i));
		rqd_abs(abs_true_vec, get_qdvector_i(true_vec, i));
		if(rqd_cmp_ui(abs_true_vec, 0UL) != 0)
			rqd_div(abs_diff_vec, abs_diff_vec, abs_true_vec);
		
		if(rqd_cmp(max_relerr, abs_diff_vec) < 0)
			rqd_set(max_relerr, abs_diff_vec);
		if(rqd_cmp(min_relerr, abs_diff_vec) > 0)
			rqd_set(min_relerr, abs_diff_vec);
	}
#endif // __cplusplus

	free_qdvector(diff_vec);// Fix! 2012-06-03 by T.Kouya

	return;
}


#endif // USE_GMP


#ifdef __cplusplus
//}
#endif

// Test main function
#ifdef DEBUG

/*
[tkouya@tkhome-centos7 bncmatmul-0.2]$ ./ddlinear_test
DD : ||vec||_2 = 4.5607017003965519165441961022668e+01
QD : ||vec||_2 = 4.56070170039655191654419610226701791630402124366564150118987767e+01
-------------------------------------------------------------------------------
BNC Default Precision    : 128 bits(38.5 decimal digits)
BNC Default Rounding Mode: Round to Nearest
-------------------------------------------------------------------------------
MPF: ||C||_F = 2.520596589389695218960806063512856930363775466732230984833583157335282071005800e5
MPF: ||C||_F = 2.520596589389695218960806063512856930373e5
 DD: ||C||_F = 2.5205965893896952189608060635128e+05
 QD: ||C||_F = 2.52059658938969521896080606351285693036377546673223098483358316e+05
*/

//#define ROW_DIM 2
//#define COL_DIM 2
//#define ROW_DIM 10
//#define COL_DIM 10
//#define ROW_DIM 64
//#define COL_DIM 64
//#define ROW_DIM 128
//#define COL_DIM 128
#define ROW_DIM 256
#define COL_DIM 256
//#define ROW_DIM 512
//#define COL_DIM 512
//#define ROW_DIM 1024
//#define COL_DIM 1024

using namespace std;

int main()
{
	int i, j;
	double stime, etime[3];
	qd_real qdval, qdtmp;
	QDVector qdvec;
	QDMatrix qdmat_c, qdmat_a, qdmat_b;
	mpf_t mpftmp, mpfval;
	MPFMatrix mpfmat_c, mpfmat_a, mpfmat_b;

	// Initialize QD library
	fpu_fix_start(NULL);

	// QD
	qdvec = init_qdvector(ROW_DIM);

	//return 0;
	for(i = 0; i < ROW_DIM; i++)
	{
		// val := sqrt((i + 1));
		//qdval = (int)(i + 1);
		//qdval = (qd_real)(i + 1);
		qdvec->element[i] = (qd_real)(i + 1);
		//qdvec->element[i] = (qd_real)sqrt((int)(i + 1));
		qdvec->element[i] = (qd_real)sqrt(qdvec->element[i]);
		//qdvec->element[i] = sqrt(qdval);
	}

//	print_qdvector(qdvec);

	norm2_qdvector(&qdval, qdvec);
	printf("QD : ||vec||_2 = ");
	cout.precision(qd_real::_ndigits);
	cout << qdval << "\n";

	free_qdvector(qdvec);

	// MPFMatrix

	set_bnc_default_prec(128);

	mpf_init(mpftmp);
	mpf_init(mpfval);

	mpfmat_a = init_mpfmatrix(ROW_DIM, COL_DIM);
	mpfmat_b = init_mpfmatrix(ROW_DIM, COL_DIM);
	mpfmat_c = init_mpfmatrix(ROW_DIM, COL_DIM);

	for(i = 0; i < ROW_DIM; i++)
	{
		for(j = 0; j < COL_DIM; j++)
		{
			mpf_set_ui(mpfval, (unsigned long)(i + j + 1));
			mpf_sqrt(mpfval, mpfval);
			set_mpfmatrix_ij(mpfmat_a, i, j, mpfval);

			mpf_set_ui(mpfval, (unsigned long)((ROW_DIM - i) + (COL_DIM - j)));
			mpf_sqrt(mpfval, mpfval);
			set_mpfmatrix_ij(mpfmat_b, i, j, mpfval);
		}
	}
//	print_mpfmatrix(mpfmat_a);
//	print_mpfmatrix(mpfmat_b);

	stime = get_secv();
	mul_mpfmatrix(mpfmat_c, mpfmat_a, mpfmat_b);
	etime[0] = get_secv() - stime;

//	print_mpfmatrix(mpfmat_c);
	normf_mpfmatrix(mpfval, mpfmat_c);
	printf("MPF: ||C||_F = ");
	mpf_out_str(stdout, 10, 0, mpfval);
	printf("\n");

	free_mpfmatrix(mpfmat_a);
	free_mpfmatrix(mpfmat_b);
	free_mpfmatrix(mpfmat_c);

	// QDmatrix
	qdmat_a = init_qdmatrix(ROW_DIM, COL_DIM);
	qdmat_b = init_qdmatrix(ROW_DIM, COL_DIM);
	qdmat_c = init_qdmatrix(ROW_DIM, COL_DIM);

	//return 0;

	for(i = 0; i < ROW_DIM; i++)
	{
		for(j = 0; j < COL_DIM; j++)
		{
			qdmat_a->element[i * COL_DIM + j] = (qd_real)(i + j + 1);
			qdmat_a->element[i * COL_DIM + j] = (qd_real)sqrt(qdmat_a->element[i * COL_DIM + j]);
			qdmat_b->element[i * COL_DIM + j] = (qd_real)((ROW_DIM - i) + (COL_DIM - j));
			qdmat_b->element[i * COL_DIM + j] = (qd_real)sqrt(qdmat_b->element[i * COL_DIM + j]);
		}
	}
	printf("qdmat_a:\n");
//	print_qdmatrix(qdmat_a);
	printf("qdmat_b:\n");
//	print_qdmatrix(qdmat_b);

	stime = get_secv();
	mul_qdmatrix(qdmat_c, qdmat_a, qdmat_b);
	etime[2] = get_secv() - stime;

	printf("qdmat_c:\n");
//	print_qdmatrix(qdmat_c);
//	normf_qdmatrix(&qdval, qdmat_c);
	printf(" QD: ||C||_F = ");
	cout << qdval << "\n";

	mpf_get_qd(&qdval, mpfval);
	printf("qd val = "); cout << qdval << "\n";
	mpf_set_qd(mpfval, qdval);
	printf("qd val = "); mpf_out_str(stdout, 10, 0, mpfval); printf("\n");

	free_qdmatrix(qdmat_a);
	free_qdmatrix(qdmat_b);
	free_qdmatrix(qdmat_c);

	printf("%d x %d matmul\n", ROW_DIM, COL_DIM);
	printf("mpfr(%4ld): %f s\n", mpfr_get_default_prec(), etime[0]);
	printf("dd        : %f s\n", etime[1]);
	printf("qd        : %f s\n", etime[1]);

	return 0;
}

#endif // DEBUG
