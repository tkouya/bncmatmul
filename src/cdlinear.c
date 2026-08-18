/********************************************************************************/
/* cdlinear.c: Double precision complex Vector, Matrix                          */
/* Copyright (c) 2024 Tomonori Kouya                                            */
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
#include <complex.h> // double _Complex

#include "cdlinear.h"

/*************************************************/
/* Functions for Vector Types                    */
/*                                               */
/* Initialize:                                   */
/*   CDVector init_cdvector(long int dimension)    */
/*   CMPFVector init_cmpfvector(long int dimension) */
/*   CMPFVector init2_cmpfvector(long int dimension, unsigned long mbits) */
/* Free:                                         */
/*   void free_cdvector(CDVector vec)              */
/*   void free_cmpfvector(CMPFVector vec)          */
/* Get & Set Values:                             */
/*   double _Complex get_cdvector_i(CDVector vec, long int index) */
/*   mpc_t *get_cmpfvector_i(CMPFVector vec, long int index) */
/*   void set_cdvector_i(CDVector vec, long int index, double _Complex val) */
/*   void set_cmpfvector_i(CMPFVector vec, long int index, mpc_t val) */
/*   void set_cmpfvector_i_d(CMPFVector vec, long int index, double _Complex val) */
/* Output:                                       */
/*   void print_cdvector(CDVector vec)             */
/*   void print_cmpfvector(CMPFVector vec)         */
/*************************************************/
CDVector init_cdvector(long int dimension)
{
	CDVector ret = NULL;
	long int i;

	if(dimension <= 0)
	{
		fprintf(stderr, "ERROR: init_cdvector\n");
		return ret;
	}

	ret = (CDVector)malloc(sizeof(cdvector));
	if(ret == NULL)
		return ret;

	ret->element = (double _Complex *)calloc(dimension, sizeof(double _Complex));
	if(ret->element == NULL)
		return ret;

	/* All 0 */
	for(i = 0; i < dimension; i++)
		*(ret->element + i) = 0.0 + 0.0 * I;

	ret->dim = dimension;

	return ret;
}

void free_cdvector(CDVector vec)
{
	if(vec == NULL)
		return;

	if(vec->element != NULL)
		free(vec->element);

	free(vec);
}

double _Complex get_cdvector_i(CDVector vec, long int index)
{
	return *(vec->element + index);
}

void set_cdvector_i(CDVector vec, long int index, double _Complex val)
{
	*(vec->element + index) = val;
}

void print_cdvector(CDVector dv)
{
	long int i, dim;

	for(i = 0; i < dv->dim; i++)
		printf("%5ld %25.17e + %25.17e * I\n", i, creal(get_cdvector_i(dv, i)), cimag(get_cdvector_i(dv, i)));
}

/*************************************************/
/* Function for Matrix Types                     */
/*                                               */
/* Initialize:                                   */
/*   CDMatrix init_cdmatrix(long int row_dimension, long int col_dimension)    */
/*   CMPFMatrix init_cmpfmatrix(long int row_dimension, long int col_dimension)*/
/*   CMPFMatrix init2_cmpfmatrix(long int row_dimension, long int col_dimension, unsigned long mbits) */
/* Free:                                         */
/*   void free_cdmatrix(CDMatrix mat)              */
/*   void free_cmpfmatrix(CMPFMatrix mat)          */
/* Get & Set:                                      */
/*   double get_cdmatrix_ij(CDMatrix mat, long int row_index, long int col_index) */
/*   double get_cmpfmatrix_ij(CMPFMatrix mat, long int row_index, long int col_index) */
/*   void set_cdmatrix_ij(CDMatrix mat, long int row_index, long int col_index, double _Complex val) */
/*   void set_cmpfmatrix_ij(CMPFMatrix mat, long int row_index, long int col_index, mpc_t val) */
/*   void set_cmpfmatrix_ij_d(CMPFMatrix mat, long int row_index, long int col_index, double _Complexval) */
/* Output:                                       */
/*   void print_cdmatrix(CDMatrix mat)             */
/*   void print_cmpfmatrix(CMPFMatrix mat)         */
/*************************************************/
CDMatrix init_cdmatrix(long int row_dimension, long int col_dimension)
{
	CDMatrix ret = NULL;
	long int i, j;

	if(row_dimension <= 0 || col_dimension <= 0)
	{
		fprintf(stderr, "ERROR: init_cdmatrix\n");
		return ret;
	}

	ret = (CDMatrix)malloc(sizeof(cdmatrix));
	if(ret == NULL)
		return ret;

	ret->element = (double _Complex *)calloc(row_dimension * col_dimension, sizeof(double _Complex));
	if(ret->element == NULL)
		return ret;

	/* All 0 */
	for(i = 0; i < row_dimension; i++)
		for(j = 0; j < col_dimension; j++)
			*(ret->element + i * col_dimension + j) = (double)0.0 + (double)0.0 * I;

	ret->row_dim = row_dimension;
	ret->col_dim = col_dimension;

	return ret;
}

// free_cdmatrix
void free_cdmatrix(CDMatrix mat)
{
	if(mat == NULL)
		return;

	if(mat->element != NULL)
		free(mat->element);

	free(mat);
}

double _Complex get_cdmatrix_ij(CDMatrix mat, long int row_index, long int col_index)
{
	return *(mat->element + row_index * mat->col_dim + col_index);
}

void set_cdmatrix_ij(CDMatrix mat, long int row_index, long int col_index, double _Complex val)
{
	*(mat->element + row_index * mat->col_dim + col_index) = val;
}

void print_cdmatrix(CDMatrix mat)
{
	long int i, j;

	for(i = 0; i < mat->row_dim; i++)
	{
		printf("%5ld ", i);
		for(j = 0; j < mat->col_dim; j++)
			printf("%25.17e+%25.17e*I ", creal(get_cdmatrix_ij(mat, i, j)), cimag(get_cdmatrix_ij(mat, i, j)));
		printf("\n");
	}
}


/*************************************************/
/* Vector Calculations for CDVector               */
/*
void add_cdvector(CDVector c, CDVector a, CDVector b)
void add2_cdvector(CDVector c, CDVector a)
void sub_cdvector(CDVector c, CDVector a, CDVector b)
void sub2_cdvector(CDVector c, CDVector a)
void cmul_cdvector(CDVector c, double val, CDVector a)
void cmul2_cdvector(CDVector c, double val)
void add_cmul_cdvector(CDVector c, CDVector a, double val, CDVector b)
double ip_cdvector(CDVector a, CDVector b)
double norm1_cdvector(CDVector a)
double norm2_cdvector(CDVector a)
double normi_cdvector(CDVector a)
void subst_cdvector(CDVector c, CDVector a)
*/
/*************************************************/
/* c = a + b */
void add_cdvector(CDVector c, CDVector a, CDVector b)
{
	long int i;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_cdvector\n");
		return;
	}

	for(i = 0; i < c->dim; i++)
		set_cdvector_i(c, i, get_cdvector_i(a, i) + get_cdvector_i(b, i));

}

/* c += a */
void add2_cdvector(CDVector c, CDVector a)
{
	long int i;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: add2_cdvector\n");
		return;
	}

	for(i = 0; i < c->dim; i++)
		set_cdvector_i(c, i, get_cdvector_i(c, i) + get_cdvector_i(a, i));

}

/* c = a - b */
void sub_cdvector(CDVector c, CDVector a, CDVector b)
{
	long int i;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_cdvector\n");
		return;
	}

	for(i = 0; i < c->dim; i++)
		set_cdvector_i(c, i, get_cdvector_i(a, i) - get_cdvector_i(b, i));

}

/* c -= a */
void sub2_cdvector(CDVector c, CDVector a)
{
	long int i;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: sub2_cdvector\n");
		return;
	}

	for(i = 0; i < c->dim; i++)
		set_cdvector_i(c, i, get_cdvector_i(c, i) - get_cdvector_i(a, i));

}

/* c = val * a */
void cmul_cdvector(CDVector c, double _Complex val, CDVector a)
{
	long int i;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: cmul_cdvector\n");
		return;
	}

	for(i = 0; i < c->dim; i++)
		set_cdvector_i(c, i, val * get_cdvector_i(a, i));

}

/* c *= val */
void cmul2_cdvector(CDVector c, double _Complex val)
{
	long int i;

	for(i = 0; i < c->dim; i++)
		set_cdvector_i(c, i, val * get_cdvector_i(c, i));

}

/* c = a + val * b */
void add_cmul_cdvector(CDVector c, CDVector a, double _Complex val, CDVector b)
{
	long int i;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_cmul_cdvector\n");
		return;
	}

	for(i = 0; i < c->dim; i++)
		set_cdvector_i(c, i, get_cdvector_i(a, i) + val * get_cdvector_i(b, i));

}

/* c = a - val * b */
void sub_cmul_cdvector(CDVector c, CDVector a, double _Complex val, CDVector b)
{
	long int i;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_cmul_cdvector\n");
		return;
	}

	for(i = 0; i < c->dim; i++)
		set_cdvector_i(c, i, get_cdvector_i(a, i) - val * get_cdvector_i(b, i));

}


/* (a, b) */
// fix!: 2024-11-06(Wed) T.Kouya
double _Complex ip_cdvector(CDVector a, CDVector b)
{
	double _Complex tmp = 0.0;
	long int i;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: ip_cdvector\n");
		return 0;
	}

	for(i = 0; i < a->dim; i++)
		tmp += conj(get_cdvector_i(a, i)) * get_cdvector_i(b, i);

	return tmp;\
}

/* Sum a[i] * b[i] */
double _Complex dotp_cdvector(CDVector a, CDVector b)
{
	double _Complex tmp = 0.0;
	long int i;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: dotp_cdvector\n");
		return 0;
	}

	for(i = 0; i < a->dim; i++)
		tmp += get_cdvector_i(a, i) * get_cdvector_i(b, i);

	return tmp;
}


/* ||a||_1 */
double norm1_cdvector(CDVector a)
{
	double ret = 0.0;
	long int i;

	for(i = 0; i < a->dim; i++)
		ret += cabs(get_cdvector_i(a, i));

	return ret;
}

/* ||a||_2 */
double norm2_cdvector(CDVector a)
{
	double ret = 0.0;
	long int i;

	// Fix! 2024-09-26 T.Kouya
	for(i = 0; i < a->dim; i++)
	{
		//ret += get_cdvector_i(a, i) * get_cdvector_i(a, i);
		ret += creal(get_cdvector_i(a, i)) * creal(get_cdvector_i(a, i));
		ret += cimag(get_cdvector_i(a, i)) * cimag(get_cdvector_i(a, i));
	}

	return sqrt(ret);
}

/* ||a||_infty */
double normi_cdvector(CDVector a)
{
	double ret, tmp;
	long int i;

	ret = cabs(get_cdvector_i(a, 0));
	for(i = 1; i < a->dim; i++)
	{
		tmp = cabs(get_cdvector_i(a, i));
		if(ret < tmp)
			ret = tmp;
	}

	return ret;
}

/* c := a */
void subst_cdvector(CDVector c, CDVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
		set_cdvector_i(c, i, get_cdvector_i(a, i));
}

/* c := conj(a) */
void conj_cdvector(CDVector c, CDVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
		set_cdvector_i(c, i, conj(get_cdvector_i(a, i)));
}

/* c := -a */
void neg_cdvector(CDVector c, CDVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
		set_cdvector_i(c, i, -get_cdvector_i(a, i));
}


/* c := real_a */
void subst_cdvector_dvec(CDVector c, DVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
		set_cdvector_i(c, i, get_dvector_i(a, i) + 0.0 * I);
}

/* real_c := real(a) */
void subst_dvector_real_cdvec(DVector c, CDVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
		set_dvector_i(c, i, creal(get_cdvector_i(a, i)));
}

/* real_c := image(a) */
void subst_dvector_image_cdvec(DVector c, CDVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
		set_dvector_i(c, i, cimag(get_cdvector_i(a, i)));
}

/* c := 0 */
void set0_cdvector(CDVector c)
{
	long int i;

	for(i = 0; i < c->dim; i++)
		set_cdvector_i(c, i, (double)0 + (double)0 * I);
}

/* append 2005.07/12 */
/*
	ret(index_start) = src(src_index_start)
	 ...
	ret(index_end  ) = src(src_index_end)
*/
void copy_cdvector_ij(CDVector ret, long int index_start, long int index_end, CDVector src, long int src_index_start, long int src_index_end)
{
	long int i, itmp;

	if((src_index_end - src_index_start) != (index_end - index_start))
	{
		fprintf(stderr, "Invalid index!(copy_cdvector_ij)\n");
		return;
	}

	for(i = 0; i <= (index_end - index_start); i++)
	{
		set_cdvector_i(ret, index_start + i, get_cdvector_i(src, src_index_start + i));
//		printf("%ld <----------------------------------> %ld\n", index_start + i, src_index_start + i);
	}
}

/* append 2023-02-26 */
// ret_real + ret_image * I := src
void separate_cdvector(DVector ret_real, DVector ret_image, CDVector src)
{
	long int i;
    double _Complex tmp;

	for(i = 0; i < src->dim ; i++)
	{
        tmp = get_cdvector_i(src, i);

        set_dvector_i(ret_real , i, creal(tmp));
        set_dvector_i(ret_image, i, cimag(tmp));
	}
}

/* append 2023-02-27 */
// ret := src_real + src_image * I
void merge_cdvector(CDVector ret, DVector src_real, DVector src_image)
{
	long int i;
    double _Complex tmp;

	for(i = 0; i < ret->dim ; i++)
	{
        tmp = get_dvector_i(src_real, i) + get_dvector_i(src_image, i) * I;

        set_cdvector_i(ret, i, tmp);
	}
}

/*************************************************/
/* Matrix Caluculations for CDMatrix              */
/*
double normf_cdmatrix(CDMatrix mat)
double normi_cdmatrix(CDMatrix mat)
double norm1_cdmatrix(CDMatrix mat)
void add_cdmatrix(CDMatrix c, CDMatrix a, CDMatrix b);
void sub_cdmatrix(CDMatrix c, CDMatrix a, CDMatrix b);
void mul_cdmatrix(CDMatrix c, CDMatrix a, CDMatrix b);
void transpose_cdmatrix(CDMatrix c, CDMatrix a);
void mul_cdmatrix_cdvec(CDVector v, CDMatrix a, CDVector vb)
void mul_cdmatrixt_cdvec(CDVector v, CDMatrix a, CDVector vb)
void inv_cdmatrix(CDMatrix a);
void subst_cdmatrix(CDMatrix c, CDMatrix a);
*/
/*************************************************/
/* Frobenius Norm of Matrix */
double normf_cdmatrix(CDMatrix mat)
{
	long int i, j;
	double ret;

	ret = 0.0;
	for(i = 0; i < mat->row_dim; i++)
		for(j = 0; j < mat->col_dim; j++)
			ret += (get_cdmatrix_ij(mat, i, j) * conj(get_cdmatrix_ij(mat, i, j)));

	ret = sqrt(ret);

	return ret;
}

/* Infinity Norm of Matrix */
double normi_cdmatrix(CDMatrix mat)
{
	long int i, j;
	double ret, sum;

	ret = 0.0;
	for(i = 0; i < mat->row_dim; i++)
	{
		sum = 0.0;
		for(j = 0; j < mat->col_dim; j++)
			sum += cabs(get_cdmatrix_ij(mat, i, j));
		if(ret < sum)
			ret = sum;
	}

	return ret;
}

/* 1 Norm of Matrix */
double norm1_cdmatrix(CDMatrix mat)
{
	long int i, j;
	double ret, sum;

	ret = 0.0;
	for(j = 0; j < mat->col_dim; j++)
	{
		sum = 0.0;
		for(i = 0; i < mat->row_dim; i++)
			sum += cabs(get_cdmatrix_ij(mat, i, j));
		if(ret < sum)
			ret = sum;
	}

	return ret;
}

/* c = a + b */
void add_cdmatrix(CDMatrix c, CDMatrix a, CDMatrix b)
{
	long int i, j, row_dim, col_dim;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_cdmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_cdmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
			set_cdmatrix_ij(c, i, j, get_cdmatrix_ij(a, i, j) + get_cdmatrix_ij(b, i, j));
	}
}

/* c = a - b */
void sub_cdmatrix(CDMatrix c, CDMatrix a, CDMatrix b)
{
	long int i, j, row_dim, col_dim;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: sub_cdmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: sub_cdmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
			set_cdmatrix_ij(c, i, j, get_cdmatrix_ij(a, i, j) - get_cdmatrix_ij(b, i, j));
	}
}

/* c = sc * a */
void cmul_cdmatrix(CDMatrix c, double _Complex sc, CDMatrix a)
{
	long int i, j, row_dim, col_dim;

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_cdmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_cdmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
			set_cdmatrix_ij(c, i, j, sc * get_cdmatrix_ij(a, i, j));
	}
}

/* c = a * b */
void mul_cdmatrix(CDMatrix c, CDMatrix a, CDMatrix b)
{
	long int i, j, k;
	double _Complex tmp;

	/* dimension check */
	if((c->row_dim != a->row_dim) || (c->col_dim != b->row_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: mul_cdmatrix\n");
		return;
	}

	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
		{
			tmp = 0.0;
			for(k = 0; k < a->col_dim; k++)
				tmp += get_cdmatrix_ij(a, i, k) * get_cdmatrix_ij(b, k, j);
			set_cdmatrix_ij(c, i, j, tmp);
		}
	}
}

/* c = a^T */
void transpose_cdmatrix(CDMatrix c, CDMatrix a)
{
	long int i, j;

	/* Check Dimentions */
	if((c->row_dim != a->col_dim) || (c->col_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: transpose_cdmatrix\n");
		return;
	}
	
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_cdmatrix_ij(c, i, j, get_cdmatrix_ij(a, j, i));
	}
}

/* c = conj(a)^T */
void star_cdmatrix(CDMatrix c, CDMatrix a)
{
	long int i, j;

	/* Check Dimentions */
	if((c->row_dim != a->col_dim) || (c->col_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: star_cdmatrix\n");
		return;
	}
	
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_cdmatrix_ij(c, i, j, conj(get_cdmatrix_ij(a, j, i)));
	}
}


/* c := a */
void subst_cdmatrix(CDMatrix c, CDMatrix a)
{
	long int i, j;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_cdmatrix\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_cdmatrix_ij(c, i, j, get_cdmatrix_ij(a, i, j));
		}
	}
}

/* c := conj(a) */
void conj_cdmatrix(CDMatrix c, CDMatrix a)
{
	long int i, j;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: conj_cdmatrix\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_cdmatrix_ij(c, i, j, conj(get_cdmatrix_ij(a, i, j)));
		}
	}
}

/* c := -a */
void neg_cdmatrix(CDMatrix c, CDMatrix a)
{
	long int i, j;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: neg_cdmatrix\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_cdmatrix_ij(c, i, j, -get_cdmatrix_ij(a, i, j));
		}
	}
}

/* c := real_a */
void subst_cdmatrix_dmat(CDMatrix c, DMatrix a)
{
	long int i, j;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_cdmatrix_dmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_cdmatrix_ij(c, i, j, get_dmatrix_ij(a, i, j) + 0.0 * I);
		}
	}
}

/* c := 0 */
void set0_cdmatrix(CDMatrix c)
{
	long int i, j;

	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_cdmatrix_ij(c, i, j, 0.0 + 0.0 * I);
	}
}

/* c := I */
void setI_cdmatrix(CDMatrix c)
{
	long int i, j;

	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_cdmatrix_ij(c, i, j, 0.0 + 0.0 * I);
		if(i < c->col_dim)
			set_cdmatrix_ij(c, i, i, 1.0 + 0.0 * I);
	}
}

/* v = a * vb */
void mul_cdmatrix_cdvec(CDVector v, CDMatrix a, CDVector vb)
{
	long int i, j;
	double _Complex tmp;

	/* Check Dimension */
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: mul_cdmatrix_dvec\n");
		return;
	}
	
	for(i = 0; i < a->row_dim; i++)
	{
		tmp = 0.0;
		for(j = 0; j < a->col_dim; j++)
			tmp += get_cdmatrix_ij(a, i, j) * get_cdvector_i(vb, j);
		set_cdvector_i(v, i, tmp);
	}
}

/* v = a^T * vb */
void mul_cdmatrixt_cdvec(CDVector v, CDMatrix a, CDVector vb)
{
	long int i, j;
	double _Complex tmp;

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_cdmatrixt_dvec\n");
		return;
	}
	
	for(i = 0; i < a->col_dim; i++)
	{
		tmp = 0.0;
		for(j = 0; j < a->row_dim; j++)
			tmp += get_cdmatrix_ij(a, j, i) * get_cdvector_i(vb, j);
		set_cdvector_i(v, i, tmp);
	}
}


/* v = conj(a)^T * vb */
void mul_cdmatrixs_cdvec(CDVector v, CDMatrix a, CDVector vb)
{
	long int i, j;
	double _Complex tmp;

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_cdmatrixs_dvec\n");
		return;
	}
	
	for(i = 0; i < a->col_dim; i++)
	{
		tmp = 0.0;
		for(j = 0; j < a->row_dim; j++)
			tmp += conj(get_cdmatrix_ij(a, j, i)) * get_cdvector_i(vb, j);
		set_cdvector_i(v, i, tmp);
	}
}

/* a = a^(-1) */
/* square matrix only */
void inv_cdmatrix(CDMatrix a)
{
	long int i, j, k, dim;
	double _Complex tmp, aii;

	/* Check Dimensions */
	if(a->row_dim != a->col_dim)
	{
		fprintf(stderr, "ERROR: inv_cdmatrix\n");
		return;
	}

	dim = a->row_dim;

	for(i = 0; i < dim; i++)
	{
		if(cabs(get_cdmatrix_ij(a, i, i)) == 0.0)
		{
			fprintf(stderr, "ERROR: inv_cdmatrix: Pivot(%ld,%ld) is zero.\n", i, i);
			return;
		}

		aii = (1.0 + 0.0 * I) / get_cdmatrix_ij(a, i, i);
		set_cdmatrix_ij(a, i, i, aii);

		for(j = 0; j < i; j++)
			set_cdmatrix_ij(a, i, j, get_cdmatrix_ij(a, i, j) * aii);
		for(j = i + 1; j < dim; j++)
			set_cdmatrix_ij(a, i, j, get_cdmatrix_ij(a, i, j) * aii);

		for(j = 0; j < i; j++)
		{
			for(k = 0; k < i; k++)
				set_cdmatrix_ij(a, j, k, get_cdmatrix_ij(a, j, k) - get_cdmatrix_ij(a, j, i) * get_cdmatrix_ij(a, i, k));
			for(k = i + 1; k < dim; k++)
				set_cdmatrix_ij(a, j, k, get_cdmatrix_ij(a, j, k) - get_cdmatrix_ij(a, j, i) * get_cdmatrix_ij(a, i, k));
		}

		for(j = i + 1; j < dim; j++)
		{
			for(k = 0; k < i; k++)
				set_cdmatrix_ij(a, j, k, get_cdmatrix_ij(a, j, k) - get_cdmatrix_ij(a, j, i) * get_cdmatrix_ij(a, i, k));
			for(k = i + 1; k < dim; k++)
				set_cdmatrix_ij(a, j, k, get_cdmatrix_ij(a, j, k) - get_cdmatrix_ij(a, j, i) * get_cdmatrix_ij(a, i, k));
		}

		for(j = 0; j < i; j++)
			set_cdmatrix_ij(a, j, i, get_cdmatrix_ij(a, j, i) * -aii);
		for(j = i + 1; j < dim; j++)
			set_cdmatrix_ij(a, j, i, get_cdmatrix_ij(a, j, i) * -aii);
	}
}

/* append 2023-02-26 */
// ret_real + ret_image * I := src
void separate_cdmatrix(DMatrix ret_real, DMatrix ret_image, CDMatrix src)
{
	long int i, j;

	for(i = 0; i < src->row_dim ; i++)
	{
        for(j = 0; j < src->col_dim; j++)
        {
            set_dmatrix_ij(ret_real , i, j, creal(get_cdmatrix_ij(src, i, j)));
            set_dmatrix_ij(ret_image, i, j, cimag(get_cdmatrix_ij(src, i, j)));
        }
	}
}

/* append 2023-02-27 */
// ret := src_real + src_image * I
void merge_cdmatrix(CDMatrix ret, DMatrix src_real, DMatrix src_image)
{
	long int i, j;
    double _Complex tmp;

	for(i = 0; i < ret->row_dim ; i++)
	{
        for(j = 0; j < ret->col_dim; j++)
        {
            tmp = get_dmatrix_ij(src_real, i, j) + get_dmatrix_ij(src_image, i, j) * I;

            set_cdmatrix_ij(ret, i, j, tmp);
        }
	}
}
