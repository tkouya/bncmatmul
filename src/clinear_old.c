/********************************************************************************/
/* clinear.c: Vector, Matrix                                                    */
/* Copyright (c) 2011-2012 Tomonori Kouya                                       */
/*                                                                              */
/* Version 0.0, 2011-11-24: new produce based on linear.c in BNCpack            */
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

//include "bnc.h"
#include "clinear.h"

/*************************************************/
/* Functions for Vector Types                    */
/*                                               */
/* Initialize:                                   */
/*   CMPFVector init_cmpfvector(long int dimension) */
/*   CMPFVector init2_cmpfvector(long int dimension, unsigned long mbits) */
/* Free:                                         */
/*   void free_cmpfvector(CMPFVector vec)          */
/* Get & Set Values:                             */
/*   void set_cmpfvector_i(CMPFVector vec, long int index, mpc_t val) */
/*   void set_cmpfvector_i_d(CMPFVector vec, long int index, double _Complex val) */
/* Output:                                       */
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

#ifdef USE_GMP
CMPFVector init_cmpfvector(long int dimension)
{
	CMPFVector ret = NULL;
	long int i;

	if(dimension <= 0)
	{
		fprintf(stderr, "ERROR: init_cmpfvector\n");
		return ret;
	}

	ret = (CMPFVector)malloc(sizeof(cmpfvector));
	if(ret == NULL)
		return ret;

	ret->element = (mpc_t *)calloc(dimension, sizeof(mpc_t));
//	ret->element = (mpf_t *)malloc(sizeof(mpf_t) * dimension);
	if(ret->element == NULL)
		return NULL;

	/* All 0 */
	for(i = 0; i < dimension; i++)
	{
		//(mpc_t *)(ret->element + i) = init_mpc_t();
		//mpf_init((ret->element + i)->re);
		//mpf_init((ret->element + i)->im);
		//(ret->element + i)->prec = get_bnc_default_prec();

		mpc_init((mpc_ptr)(ret->element + i)); //, get_bnc_default_prec());

		if((ret->element + i) == NULL)
			return NULL;

		//set0_mpc_t((ret->element + i));
		mpc_set_ui_ui((mpc_ptr)(ret->element + i), 0UL, 0UL, get_bnc_default_rounding_mode());
	}

	ret->dim = dimension;

	ret->prec = get_bnc_default_prec();

	return ret;
}

/* mbits ... A number of at least bits of mantissa */
CMPFVector init2_cmpfvector(long int dimension, unsigned long int mbits)
{
	CMPFVector ret = NULL;
	long int i;

	if(dimension <= 0)
	{
		fprintf(stderr, "ERROR: init2_cmpfvector\n");
		return ret;
	}

	ret = (CMPFVector)malloc(sizeof(cmpfvector));
	if(ret == NULL)
		return ret;

	ret->element = (mpc_t *)calloc(dimension, sizeof(mpc_t));
//	ret->element = (mpf_t *)calloc(sizeof(mpf_t), dimension);
//	ret->element = (mpf_t *)malloc(sizeof(mpf_t) * dimension);
	if(ret->element == NULL)
		return NULL;

	/* All 0 */
	for(i = 0; i < dimension; i++)
	{
		//*(ret->element + i) = init2_mpc_t(mbits);
		//mpf_init2((ret->element + i)->re, mbits);
		//mpf_init2((ret->element + i)->im, mbits);
		//(ret->element + i)->prec = mbits;
		mpc_init2((mpc_ptr)(ret->element + i), mbits);

		if((ret->element + i) == NULL)
			return NULL;

		//set0_mpc_t((ret->element + i));
		mpc_set_ui_ui((mpc_ptr)(ret->element + i), 0UL, 0UL, get_bnc_default_rounding_mode());
	}

	ret->dim = dimension;

	ret->prec = mbits;

	return ret;
}

void free_cmpfvector(CMPFVector vec)
{
	long int i;

	if(vec == NULL)
		return;

	if(vec->element != NULL)
	{
		for(i = 0; i < vec->dim; i++)
		{
			mpc_clear((mpc_ptr)(vec->element + i));
			//mpf_clear((vec->element + i)->re);
			//mpf_clear((vec->element + i)->im);
		}
		free(vec->element);
	}

//	free(&(vec->dim));
//	free(&(vec->prec));
	free(vec);
}
#endif // USE_GMP

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

#ifdef USE_GMP
mpc_ptr get_cmpfvector_i(CMPFVector vec, long int index)
{
	return (mpc_ptr)(vec->element + index);
}

void set_cmpfvector_i(CMPFVector vec, long int index, mpc_t val)
{
	mpc_set((mpc_ptr)(vec->element + index), val, get_bnc_default_rounding_mode());
	//subst_mpc_t((vec->element + index), val);
}

// vec[i] := val + 0 * I
void set_cmpfvector_i_real(CMPFVector vec, long int index, mpf_t val)
{
	mpc_set_fr((mpc_ptr)(vec->element + index), val, get_bnc_default_rounding_mode());
	//subst_mpc_t((vec->element + index), val);
    //set0_mpc_t((vec->element + index));
	//set_real_mpc_t((vec->element + index), val);
}

// vec[i] := 0 + val * I
void set_cmpfvector_i_image(CMPFVector vec, long int index, mpf_t val)
{
	mpf_set_str(mpc_realref((mpc_ptr)(vec->element + index)), "0", 10);
	mpf_set(mpc_imagref((mpc_ptr)(vec->element + index)), val);	
}

void set_cmpfvector_i_d(CMPFVector vec, long int index, double _Complex val)
{
	mpc_set_dc(*(vec->element + index), val, get_bnc_default_rounding_mode());
	//set_mpc_t_d((vec->element + index), val);
}

void set_cmpfvector_i_str(CMPFVector vec, long int index, const char *re_str, const char *im_str, int base)
{
	mpf_set_str(mpc_realref((mpc_ptr)(vec->element + index)), re_str, base);
	mpf_set_str(mpc_imagref((mpc_ptr)(vec->element + index)), im_str, base);
}

void set_cmpfvector_i_ui(CMPFVector vec, long int index, unsigned long val)
{
	mpc_set_ui(*(vec->element + index), val, get_bnc_default_rounding_mode());
}

/* get precision of CMPFVector */
unsigned long int prec_cmpfvector(CMPFVector vec)
{
	return vec->prec;
}
/* get precision of CMPFVector */
unsigned long int get_prec_cmpfvector(CMPFVector vec)
{
	return vec->prec;
}

/* search minimam precision in CMPFVector */
unsigned long int minprec_cmpfvector(CMPFVector vec)
{
	unsigned long int prec, tmp;
	long int i;

	prec = mpc_get_prec(get_cmpfvector_i(vec, 0));
//	prec = get_prec_mpc_t(get_cmpfvector_i(vec, 0));
	for(i = 1; i < vec->dim; i++)
	{
		tmp = mpc_get_prec(get_cmpfvector_i(vec, i));
		//tmp = get_prec_mpc_t(get_cmpfvector_i(vec, i));
		if(prec > tmp)
			prec = tmp;
	}

	return prec;
}

/* search maximam precision in CMPFVector */
unsigned long int maxprec_cmpfvector(CMPFVector vec)
{
	unsigned long int prec, tmp;
	long int i;

	prec = mpc_get_prec(get_cmpfvector_i(vec, 0));
//	prec = get_prec_mpc_t(get_cmpfvector_i(vec, 0));
	for(i = 1; i < vec->dim; i++)
	{
		tmp = mpc_get_prec(get_cmpfvector_i(vec, i));
		//tmp = get_prec_mpc_t(get_cmpfvector_i(vec, i));
		if(prec < tmp)
			prec = tmp;
	}

	return prec;
}

void print_cmpfvector(CMPFVector vec)
{
	long int i;

	for(i = 0; i < vec->dim; i++)
	{
		printf("%5ld ", i);
		mpf_out_str(stdout, 10, 0, mpc_realref(get_cmpfvector_i(vec, i)));
		printf(" ");
		mpf_out_str(stdout, 10, 0, mpc_imagref(get_cmpfvector_i(vec, i)));
		//print_mpc_t(get_cmpfvector_i(vec, i));
		printf("\n");
	}
}
#endif // USE_MPF

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

#ifdef USE_GMP
CMPFMatrix init_cmpfmatrix(long int row_dimension, long int col_dimension)
{
	CMPFMatrix ret = NULL;
	long int i, j;

	if(row_dimension <= 0 || col_dimension <= 0)
	{
		fprintf(stderr, "ERROR: init_cmpfmatrix\n");
		return ret;
	}

	ret = (CMPFMatrix)malloc(sizeof(cmpfmatrix));
	if(ret == NULL)
		return ret;

//	ret->element = (mpf_t *)calloc(sizeof(mpf_t), row_dimension * col_dimension);
	ret->element = (mpc_t *)calloc(row_dimension * col_dimension, sizeof(mpc_t));
	if(ret->element == NULL)
		return ret;

	/* All 0 */
	for(i = 0; i < row_dimension; i++)
		for(j = 0; j < col_dimension; j++)
		{
			mpc_init(*(ret->element + i * col_dimension + j));
			//mpf_init((ret->element + i * col_dimension + j)->re);
			//mpf_init((ret->element + i * col_dimension + j)->im);
			//(ret->element + i * col_dimension + j)->prec = get_bnc_default_prec();
			mpc_set_ui_ui((mpc_ptr)(ret->element + i * col_dimension + j), 0UL, 0UL, get_bnc_default_rounding_mode());
			//set0_mpc_t((ret->element + i * col_dimension + j));
		}

	ret->row_dim = row_dimension;
	ret->col_dim = col_dimension;

	ret->prec = get_bnc_default_prec();

	return ret;
}

/* mbits ... A number of at least bits of mantissa */
CMPFMatrix init2_cmpfmatrix(long int row_dimension, long int col_dimension, unsigned long mbits)
{
	CMPFMatrix ret = NULL;
	long int i, j;

	if(row_dimension <= 0 || col_dimension <= 0)
	{
		fprintf(stderr, "ERROR: init2_cmpfmatrix\n");
		return ret;
	}

	ret = (CMPFMatrix)malloc(sizeof(cmpfmatrix));
	if(ret == NULL)
		return ret;

//	ret->element = (mpf_t *)calloc(sizeof(mpf_t), row_dimension * col_dimension);
	ret->element = (mpc_t *)calloc(row_dimension * col_dimension, sizeof(mpc_t));
	if(ret->element == NULL)
		return ret;

	/* All 0 */
	for(i = 0; i < row_dimension; i++)
		for(j = 0; j < col_dimension; j++)
		{
			mpc_init2(*(ret->element + i * col_dimension + j), mbits);
			//mpf_init2((ret->element + i * col_dimension + j)->re, mbits);
			//mpf_init2((ret->element + i * col_dimension + j)->im, mbits);
			//(ret->element + i * col_dimension + j)->prec = mbits;
			mpc_set_ui_ui((mpc_ptr)(ret->element + i * col_dimension + j), 0UL, 0UL, get_bnc_default_rounding_mode());
			//set0_mpc_t((ret->element + i * col_dimension + j));
		}

	ret->row_dim = row_dimension;
	ret->col_dim = col_dimension;

	ret->prec = mbits;

	return ret;
}
#endif // USE_GMP

void free_cdmatrix(CDMatrix mat)
{
	if(mat == NULL)
		return;

	if(mat->element != NULL)
		free(mat->element);

	free(mat);
}

#ifdef USE_GMP
void free_cmpfmatrix(CMPFMatrix mat)
{
	long int i, j;

	if(mat == NULL)
		return;

	if(mat->element != NULL)
	{
		for(i = 0; i < mat->row_dim; i++)
		{
			for(j = 0; j < mat->col_dim; j++)
				mpc_clear((mpc_ptr)(mat->element + i * mat->col_dim + j));
		}

		free(mat->element);
	}
	free(mat);
}
#endif // USE_GMP

double _Complex get_cdmatrix_ij(CDMatrix mat, long int row_index, long int col_index)
{
	return *(mat->element + row_index * mat->col_dim + col_index);
}

#ifdef USE_GMP
mpc_ptr get_cmpfmatrix_ij(CMPFMatrix mat, long int row_index, long int col_index)
{
	return (mpc_ptr)(mat->element + row_index * mat->col_dim + col_index);
}
#endif // USE_GMP

void set_cdmatrix_ij(CDMatrix mat, long int row_index, long int col_index, double _Complex val)
{
	*(mat->element + row_index * mat->col_dim + col_index) = val;
}

#ifdef USE_GMP
void set_cmpfmatrix_ij(CMPFMatrix mat, long int row_index, long int col_index, mpc_t val)
{
	mpc_set((mpc_ptr)(mat->element + row_index * mat->col_dim + col_index), val, get_bnc_default_rounding_mode());
}

void set_cmpfmatrix_ij_d(CMPFMatrix mat, long int row_index, long int col_index, double _Complex val)
{
	mpc_set_d_d((mpc_ptr)(mat->element + row_index * mat->col_dim + col_index), val, (double)0.0, get_bnc_default_rounding_mode());
	//set_mpc_t_d((mat->element + row_index * mat->col_dim + col_index), val);
}

void set_cmpfmatrix_ij_str(CMPFMatrix mat, long int row_index, long int col_index, const char *re_str, const char *im_str, int base)
{
	mpf_set_str(mpc_realref((mpc_ptr)(mat->element + row_index * mat->col_dim + col_index)), re_str, base);
	mpf_set_str(mpc_imagref((mpc_ptr)(mat->element + row_index * mat->col_dim + col_index)), im_str, base);
	//set_mpc_t_str_str((mat->element + row_index * mat->col_dim + col_index), re_str, base, im_str, base);
}

void set_cmpfmatrix_ij_ui(CMPFMatrix mat, long int row_index, long int col_index, unsigned long val)
{
	mpc_set_ui_ui((mpc_ptr)(mat->element + row_index * mat->col_dim + col_index), val, 0UL, get_bnc_default_prec());
}

/* get precision of CMPFMatrix */
unsigned long int get_prec_cmpfmatrix(CMPFMatrix mat)
{
	return mat->prec;
}
unsigned long int prec_cmpfmatrix(CMPFMatrix mat)
{
	return mat->prec;
}

/* search minimam precision in CMPFMatrix */
unsigned long int minprec_cmpfmatrix(CMPFMatrix mat)
{
	unsigned long int prec, tmp;
	long int i, j;

	prec = mpc_get_prec(get_cmpfmatrix_ij(mat, 0, 0));
	//prec = get_prec_mpc_t(get_cmpfmatrix_ij(mat, 0, 0));
	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
		{
			tmp = mpc_get_prec(get_cmpfmatrix_ij(mat, i, j));
			//tmp = get_prec_mpc_t(get_cmpfmatrix_ij(mat, i, j));
			if(prec > tmp)
				prec = tmp;
		}
	}

	return prec;
}

/* search maximam precision in CMPFMatrix */
unsigned long int maxprec_cmpfmatrix(CMPFMatrix mat)
{
	unsigned long int prec, tmp;
	long int i, j;

	prec = mpc_get_prec(get_cmpfmatrix_ij(mat, 0, 0));
	//prec = get_prec_mpc_t(get_cmpfmatrix_ij(mat, 0, 0));
	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
		{
			tmp = mpc_get_prec(get_cmpfmatrix_ij(mat, i, j));
			//tmp = get_prec_mpc_t(get_cmpfmatrix_ij(mat, i, j));
			if(prec < tmp)
				prec = tmp;
		}
	}

	return prec;
}
#endif // USE_GMP

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

#ifdef USE_GMP
void print_cmpfmatrix(CMPFMatrix mat)
{
	long int i, j;

	for(i = 0; i < mat->row_dim; i++)
	{
		printf("%5ld ", i);
		for(j = 0; j < mat->col_dim; j++)
		{
			//mpf_out_str(stdout, 10, 0, get_cmpfmatrix_ij(mat, i, j));
			
			// print real part
			mpf_out_str(stdout, 10, 0, mpc_realref(get_cmpfmatrix_ij(mat, i, j)));
			printf(" + ");
			mpf_out_str(stdout, 10, 0, mpc_imagref(get_cmpfmatrix_ij(mat, i, j)));
			printf(" * I ");
		}
		printf("\n");
	}
}
#endif // USE_GMP

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

/* (a, b) */
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

	for(i = 0; i < a->dim; i++)
		ret += get_cdvector_i(a, i) * get_cdvector_i(a, i);

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
/* Vector Calculations for CMPFVector             */
/*
void add_cmpfvector(CMPFVector c, CMPFVector a, CMPFVector b)
void add2_cmpfvector(CMPFVector c, CMPFVector a)
void sub_cmpfvector(CMPFVector c, CMPFVector a, CMPFVector b)
void sub2_cmpfvector(CMPFVector c, CMPFVector a)
void cmul_cmpfvector(CMPFVector c, mpc_t val, CMPFVector a)
void cmul2_cmpfvector(CMPFVector c, mpc_t val)
void add_cmul_cmpfvector(CMPFVector c, CMPFVector a, mpc_t val, CMPFVector b)
void ip_cmpfvector(mpc_t ret, CMPFVector a, CMPFVector b)
void norm1_cmpfvector(mpf_t ret, CMPFVector a)
void norm2_cmpfvector(mpf_t ret, CMPFVector a)
void normi_cmpfvector(mpf_t ret, CMPFVector a)
void subst_cmpfvector(CMPFVector c, CMPFVector a)
*/
/*************************************************/
#ifdef USE_GMP
/* c = a + b */
void add_cmpfvector(CMPFVector c, CMPFVector a, CMPFVector b)
{
	long int i;
//	mpf_t tmp;
	mpc_t tmp;
	unsigned long int prec;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_cmpfvector\n");
		return;
	}

	prec = get_prec_cmpfvector(c);

	mpc_init2(tmp, prec);
	//tmp = init2_mpc_t(prec);

	for(i = 0; i < c->dim; i++)
	{
	//	mpf_add(tmp, get_cmpfvector_i(a, i), get_cmpfvector_i(b, i));
		mpc_add(tmp, get_cmpfvector_i(a, i), get_cmpfvector_i(b, i), get_bnc_default_rounding_mode());
		set_cmpfvector_i(c, i, tmp);
	}

	//mpf_clear(tmp);
	mpc_clear(tmp);

}

/* c += a */
void add2_cmpfvector(CMPFVector c, CMPFVector a)
{
	long int i;
//	mpf_t tmp;
	mpc_t tmp;
	unsigned long int prec;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: add2_cmpfvector\n");
		return;
	}

	prec = get_prec_cmpfvector(c);

	mpc_init2(tmp, prec);
	//tmp = init2_mpc_t(prec);

	for(i = 0; i < c->dim; i++)
	{
//		mpf_add(tmp, get_cmpfvector_i(c, i), get_cmpfvector_i(a, i));
		mpc_add(tmp, get_cmpfvector_i(c, i), get_cmpfvector_i(a, i), get_bnc_default_rounding_mode());
		set_cmpfvector_i(c, i, tmp);
	}

	mpc_clear(tmp);
//	mpf_clear(tmp);
}

/* c = a - b */
void sub_cmpfvector(CMPFVector c, CMPFVector a, CMPFVector b)
{
	long int i;
	//mpf_t tmp, tmpa, tmpb;
	mpc_t tmp;
	unsigned long int prec;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_cmpfvector\n");
		return;
	}

	prec = get_prec_cmpfvector(c);

	mpc_init2(tmp, prec);
	//tmp = init2_mpc_t(prec);

	for(i = 0; i < c->dim; i++)
	{
		//mpf_sub(tmp, get_cmpfvector_i(a, i), get_cmpfvector_i(b, i));
		mpc_sub(tmp, get_cmpfvector_i(a, i), get_cmpfvector_i(b, i), get_bnc_default_rounding_mode());
		set_cmpfvector_i(c, i, tmp);
	}

	//mpf_clear(tmp);
	mpc_clear(tmp);
}

/* c -= a */
void sub2_cmpfvector(CMPFVector c, CMPFVector a)
{
	long int i;
	//mpf_t tmp;
	mpc_t tmp;
	unsigned long int prec;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: sub2_cmpfvector\n");
		return;
	}

	prec = get_prec_cmpfvector(c);

	mpc_init2(tmp, prec);
	//tmp = init2_mpc_t(prec);

	for(i = 0; i < c->dim; i++)
	{
		//mpf_sub(tmp, get_cmpfvector_i(c, i), get_cmpfvector_i(a, i));
		mpc_sub(tmp, get_cmpfvector_i(c, i), get_cmpfvector_i(a, i), get_bnc_default_rounding_mode());
		set_cmpfvector_i(c, i, tmp);
	}

	//mpf_clear(tmp);
	mpc_clear(tmp);
}

/* c = val * a */
void cmul_cmpfvector(CMPFVector c, mpc_t val, CMPFVector a)
{
	long int i;
	//mpf_t tmp, tmpa;
	mpc_t tmp, tmpa;
	unsigned long int prec;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: cmul_cmpfvector\n");
		return;
	}

	prec = get_prec_cmpfvector(c);

	mpc_init2(tmp, prec);
	//tmp = init2_mpc_t(prec);

	for(i = 0; i < c->dim; i++)
	{
		//mpf_mul(tmp, val, get_cmpfvector_i(a, i));
		mpc_mul(tmp, val, get_cmpfvector_i(a, i), get_bnc_default_rounding_mode());
		set_cmpfvector_i(c, i, tmp);
	}

	//mpf_clear(tmp);
	mpc_clear(tmp);
}

/* c = val * a */
void cmul_cmpfvector_4m(CMPFVector c, mpc_t val, CMPFVector a)
{
	long int i;
	unsigned long int prec;
	MPFVector t1, t2, t3, a_re, a_im, c_re, c_im;
	//mpf_t tmp;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: cmul_cmpfvector_4m\n");
		return;
	}

	prec = get_prec_cmpfvector(c);
	//mpf_init2(tmp, prec)

	c_re = init2_mpfvector(c->dim, prec);
	c_im = init2_mpfvector(c->dim, prec);
	a_re = init2_mpfvector(a->dim, a->prec);
	a_im = init2_mpfvector(a->dim, a->prec);

    t1 = init2_mpfvector(c->dim, prec);
    t2 = init2_mpfvector(c->dim, prec);
	separate_cmpfvector(a_re, a_im, a);

    cmul_mpfvector(t1, mpc_realref(val), a_re);
    cmul_mpfvector(t2, mpc_imagref(val), a_im);
    sub_mpfvector(c_re, t1, t2);

    //#ifdef USE_4M
        // 4M
        cmul_mpfvector(t1, mpc_imagref(val), a_re);
        cmul_mpfvector(t2, mpc_realref(val), a_im);
        add_mpfvector(c_im, t1, t2);
    //#else // USE_4M
        // 3M
    /*
        mpf_add(tmp.val, val->val_re, val->val_im);
        t3 = init2_mpfvector(c->re->dim, prec);
        add_mpfvector(t3, a->re, a->im);
        cmul_mpfvector(c->im, tmp.val, t3);
        sub_mpfvector(c->im, c->im, t1);
        sub_mpfvector(c->im, c->im, t2);
        free_mpfvector(t3);
    */
    //#endif // USE_4M

	merge_cmpfvector(c, c_re, c_im);

	free_mpfvector(c_re);
	free_mpfvector(c_im);
	free_mpfvector(a_re);
	free_mpfvector(a_im);
    free_mpfvector(t1);
    free_mpfvector(t2);
}
/* c = val * real_a */
void cmul_cmpfvector_mpfvec(CMPFVector c, mpc_t val, MPFVector a)
{
	long int i;
	//mpf_t tmp, tmpa;
	mpc_t tmp, tmpa;
	unsigned long int prec;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: cmul_cmpfvector_mpfvec\n");
		return;
	}

	prec = get_prec_cmpfvector(c);

	mpc_init2(tmp, prec);
	//tmp = init2_mpc_t(prec);

	for(i = 0; i < c->dim; i++)
	{
		//mpf_mul(tmp, val, get_cmpfvector_i(a, i));
		mpc_mul_fr(tmp, val, get_mpfvector_i(a, i), get_bnc_default_rounding_mode());
		set_cmpfvector_i(c, i, tmp);
	}

	//mpf_clear(tmp);
	mpc_clear(tmp);
}

/* c *= val */
void cmul2_cmpfvector(CMPFVector c, mpc_t val)
{
	long int i;
	mpc_t tmp;
	unsigned long int prec;

	prec = get_prec_cmpfvector(c);

	mpc_init2(tmp, prec);

	for(i = 0; i < c->dim; i++)
	{
		//mpf_mul(tmp, val, get_cmpfvector_i(c, i));
		mpc_mul(tmp, val, get_cmpfvector_i(c, i), get_bnc_default_rounding_mode());
		set_cmpfvector_i(c, i, tmp);
	}

	//mpf_clear(tmp);
	mpc_clear(tmp);
}

/* c = a + val * b */
void add_cmul_cmpfvector(CMPFVector c, CMPFVector a, mpc_t val, CMPFVector b)
{
	long int i;
	//mpf_t tmp, tmp2;
	mpc_t tmp, tmp2;
	unsigned long int prec;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_cmpfvector\n");
		return;
	}

	prec = get_prec_cmpfvector(c);

	//mpf_init2(tmp, prec);
	//mpf_init2(tmp2, prec);
	mpc_init2(tmp, prec);
	mpc_init2(tmp2, prec);

	for(i = 0; i < c->dim; i++)
	{
		//mpf_mul(tmp2, val, get_cmpfvector_i(b, i));
		mpc_mul(tmp2, val, get_cmpfvector_i(b, i), get_bnc_default_rounding_mode());
		//mpf_add(tmp, get_cmpfvector_i(a, i), tmp2);
		mpc_add(tmp, get_cmpfvector_i(a, i), tmp2, get_bnc_default_rounding_mode());
		set_cmpfvector_i(c, i, tmp);
	}

	//mpf_clear(tmp);
	//mpf_clear(tmp2);
	mpc_clear(tmp);
	mpc_clear(tmp2);
}

/* ret = (a, b) */
void ip_cmpfvector(mpc_t ret, CMPFVector a, CMPFVector b)
{
	long int i;
	//mpf_t tmp;
	mpc_t tmp;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: ip_cmpfvector\n");
		return;
	}

	mpc_init2(tmp, mpc_get_prec(ret));
	//tmp = init2_mpc_t(get_prec_mpc_t(ret));

	mpc_set_ui_ui(ret, 0UL, 0UL, get_bnc_default_rounding_mode()); /* ret := 0 */
	//set0_mpc_t(ret);
	for(i = 0; i < a->dim; i++)
	{
		//mpf_mul(tmp, get_cmpfvector_i(a, i), get_cmpfvector_i(b, i));
		mpc_mul(tmp, get_cmpfvector_i(a, i), get_cmpfvector_i(b, i), get_bnc_default_rounding_mode());
		//mpf_add(ret, ret, tmp);
		mpc_add(ret, ret, tmp, get_bnc_default_rounding_mode());
	}

	//mpf_clear(tmp);
	mpc_clear(tmp);
}

/* ret = ||a||_1 */
void norm1_cmpfvector(mpf_t ret, CMPFVector a)
{

	long int i;
	mpf_t tmp;

	mpf_init2(tmp, mpf_get_prec(ret));

	mpf_set_ui(ret, 0UL); /* ret := 0 */
	for(i = 0; i < a->dim; i++)
	{
		mpc_abs(tmp, get_cmpfvector_i(a, i), get_bnc_default_rounding_mode());
		//abs_mpc_t(tmp, get_cmpfvector_i(a, i));
		mpf_add(ret, ret, tmp);
	}

	mpf_clear(tmp);
}

/* ret := ||a||_2 */
void norm2_cmpfvector(mpf_t ret, CMPFVector a)
{
	long int i;
	mpf_t re, im;

	mpf_init2(re, mpf_get_prec(ret));
	mpf_init2(im, mpf_get_prec(ret));

	mpf_set_ui(ret, 0UL); /* ret := 0 */
	for(i = 0; i < a->dim; i++)
	{
		//mpf_mul(tmp, get_cmpfvector_i(a, i), get_cmpfvector_i(a, i));
		mpf_set(re, mpc_realref(get_cmpfvector_i(a, i)));
		mpf_set(im, mpc_imagref(get_cmpfvector_i(a, i)));
		mpf_mul(re, re, re);
		mpf_mul(im, im, im);
		mpf_add(ret, ret, re);
		mpf_add(ret, ret, im);
	}
	mpf_sqrt(ret, ret);

	mpf_clear(re);
	mpf_clear(im);
}

/* ||a||_infty */
void normi_cmpfvector(mpf_t ret, CMPFVector a)
{
	long int i;
	mpf_t tmp;

	mpf_init2(tmp, mpf_get_prec(ret));

	mpc_abs(ret, get_cmpfvector_i(a, 0), get_bnc_default_rounding_mode());
	//abs_mpc_t(ret, get_cmpfvector_i(a, 0));
	for(i = 1; i < a->dim; i++)
	{
		mpc_abs(tmp, get_cmpfvector_i(a, i), get_bnc_default_rounding_mode());
		//abs_mpc_t(tmp, get_cmpfvector_i(a, i));
		if(mpf_cmp(ret, tmp) < 0) /* ret < tmp */
			mpf_set(ret, tmp);
	}

	mpf_clear(tmp);
}

/* c := a */
void subst_cmpfvector(CMPFVector c, CMPFVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
		set_cmpfvector_i(c, i, get_cmpfvector_i(a, i));
}

/* c := real_a */
void subst_cmpfvector_mpfvec(CMPFVector c, MPFVector a)
{
	long int i;
	mpc_t tmp;

	mpc_init2(tmp, c->prec);

	for(i = 0; i < a->dim; i++)
	{
		mpc_set_ui_ui(tmp, 0UL, 0UL, get_bnc_default_rounding_mode());
		mpc_set_fr(tmp, get_mpfvector_i(a, i), get_bnc_default_rounding_mode());

		set_cmpfvector_i(c, i, tmp);
	}

	mpc_clear(tmp);
}

/* real_c := real(a) */
void subst_mpfvector_real_cmpfvec(MPFVector c, CMPFVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
	{
		set_mpfvector_i(c, i, mpc_realref(get_cmpfvector_i(a, i)));
	}
}

/* real_c := image(a) */
void subst_mpfvector_image_cmpfvec(MPFVector c, CMPFVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
	{
		set_mpfvector_i(c, i, mpc_imagref(get_cmpfvector_i(a, i)));
	}
}

/* c := 0 */
void set0_cmpfvector(CMPFVector c)
{
	unsigned long int prec;
	long int i;
	mpc_t tmp;

	prec = get_prec_cmpfvector(c);

	mpc_init2(tmp, prec);
	//tmp = init2_mpc_t(prec);
	mpc_set_ui_ui(tmp, 0UL, 0UL, get_bnc_default_rounding_mode());
	//set0_mpc_t(tmp);

	for(i = 0; i < c->dim; i++)
		set_cmpfvector_i(c, i, tmp);

	//mpf_clear(tmp);
	mpc_clear(tmp);
}

/* append 2005.07/12 */
/*
	ret(index_start) = src(src_index_start)
	 ...
	ret(index_end  ) = src(src_index_end)
*/
void copy_cmpfvector_ij(CMPFVector ret, long int index_start, long int index_end, CMPFVector src, long int src_index_start, long int src_index_end)
{
	long int i, itmp;

	if((src_index_end - src_index_start) != (index_end - index_start))
	{
		fprintf(stderr, "Invalid index!(copy_cmpfvector_ij)\n");
		return;
	}

	for(i = 0; i <= (index_end - index_start); i++)
	{
		set_cmpfvector_i(ret, index_start + i, get_cmpfvector_i(src, src_index_start + i));
//		printf("%ld <----------------------------------> %ld\n", index_start + i, src_index_start + i);
	}
}

/* append 2023-02-26 */
// ret_real + ret_image * I := src
void separate_cmpfvector(MPFVector ret_real, MPFVector ret_image, CMPFVector src)
{
	long int i;

	for(i = 0; i < src->dim ; i++)
	{
        set_mpfvector_i(ret_real , i, mpc_realref(get_cmpfvector_i(src, i)));
        set_mpfvector_i(ret_image, i, mpc_imagref(get_cmpfvector_i(src, i)));
	}
}

/* append 2023-02-27 */
// ret := src_real + src_image * I
void merge_cmpfvector(CMPFVector ret, MPFVector src_real, MPFVector src_image)
{
	long int i, j;
    mpc_t tmp;

    mpc_init2(tmp, ret->prec);

	for(i = 0; i < ret->dim ; i++)
	{
        mpc_set_fr_fr(
			tmp,
			get_mpfvector_i(src_real, i),
			get_mpfvector_i(src_image, i),
			get_bnc_default_rounding_mode()
		);

        set_cmpfvector_i(ret, i, tmp);
	}

    mpc_clear(tmp);
}
#endif // USE_GMP


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

/*************************************************/
/* Matrix Caluculations for CMPFMatrix            */
/*
void normf_cmpfmatrix(mpf_t ret, CMPFMatrix mat)
void norm1_cmpfmatrix(mpf_t ret, CMPFMatrix mat)
void normi_cmpfmatrix(mpf_t ret, CMPFMatrix mat)
void add_cmpfmatrix(CMPFMatrix c, CMPFMatrix a, CMPFMatrix b);
void sub_cmpfmatrix(CMPFMatrix c, CMPFMatrix a, CMPFMatrix b);
void mul_cmpfmatrix(CMPFMatrix c, CMPFMatrix a, CMPFMatrix b);
void mul_cmpfmatrix_mpfvec(CMPFVector v, CMPFMatrix a, CMPFVector vb)
void mul_cmpfmatrixt_mpfvec(CMPFVector v, CMPFMatrix a, CMPFVector vb)
void transpose_cmpfmatrix(CMPFMatrix c, CMPFMatrix a);
void inv_cmpfmatrix(CMPFMatrix a);
void subst_mpfmatrux(CMPFMatrix c, CMPFMatrix a);
*/
/*************************************************/
#ifdef USE_GMP
/* Frobenius Norm of Matrix */
void normf_cmpfmatrix(mpf_t ret, CMPFMatrix mat)
{
	long int i, j;
	mpf_t re, im;

	mpf_init2(re, mpf_get_prec(ret));
	mpf_init2(im, mpf_get_prec(ret));
	mpf_set_ui(ret, 0UL);
	for(i = 0; i < mat->row_dim; i++)
		for(j = 0; j < mat->col_dim; j++)
		{
			//mpf_mul(tmp, get_cmpfmatrix_ij(mat, i, j), get_cmpfmatrix_ij(mat, i, j));
			mpf_set(re, mpc_realref(get_cmpfmatrix_ij(mat, i, j)));
			mpf_set(im, mpc_imagref(get_cmpfmatrix_ij(mat, i, j)));
			mpf_mul(re, re, re);
			mpf_mul(im, im, im);
			mpf_add(ret, ret, re); // Fix! 2011-12-08
			mpf_add(ret, ret, im);
		}

	mpf_sqrt(ret, ret);

	mpf_clear(im);
	mpf_clear(re);

	return;
}

/* Infinity Norm of Matrix */
void normi_cmpfmatrix(mpf_t ret, CMPFMatrix mat)
{
	long int i, j;
	mpf_t tmp, sum;

	mpf_init2(tmp, mat->prec);
	mpf_init2(sum, mat->prec);

	mpf_set_ui(ret, 0UL);
	for(i = 0; i < mat->row_dim; i++)
	{
		mpf_set_ui(sum, 0UL);
		for(j = 0; j < mat->col_dim; j++)
		{
			mpc_abs(tmp, get_cmpfmatrix_ij(mat, i, j), get_bnc_default_rounding_mode());
			//abs_mpc_t(tmp, get_cmpfmatrix_ij(mat, i, j));
			mpf_add(sum, sum, tmp);
		}
		if(mpf_cmp(ret, sum) < 0)
			mpf_set(ret, sum);
	}

	mpf_clear(tmp);
	mpf_clear(sum);

	return;
}

/* 1 Norm of Matrix */
void norm1_cmpfmatrix(mpf_t ret, CMPFMatrix mat)
{
	long int i, j;
	mpf_t tmp, sum;

	mpf_init2(tmp, mat->prec);
	mpf_init2(sum, mat->prec);

	mpf_set_ui(ret, 0UL);
	for(j = 0; j < mat->col_dim; j++)
	{
		mpf_set_ui(sum, 0UL);
		for(i = 0; i < mat->row_dim; i++)
		{
			mpc_abs(tmp, get_cmpfmatrix_ij(mat, i, j), get_bnc_default_rounding_mode());
			//abs_mpc_t(tmp, get_cmpfmatrix_ij(mat, i, j));
			mpf_add(sum, sum, tmp);
		}
		if(mpf_cmp(ret, sum) < 0)
			mpf_set(ret, sum);
	}

	mpf_clear(tmp);
	mpf_clear(sum);

	return;
}

/* c := a + b */
void add_cmpfmatrix(CMPFMatrix c, CMPFMatrix a, CMPFMatrix b)
{
	long int i, j, row_dim, col_dim;
	//mpf_t tmp;
	mpc_t tmp;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_cmpfmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_cmpfmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	mpc_init2(tmp, prec_cmpfmatrix(c));
	//tmp = init2_mpc_t(get_prec_cmpfmatrix(c));

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			//mpf_add(tmp, get_cmpfmatrix_ij(a, i, j), get_cmpfmatrix_ij(b, i, j));
			mpc_add(tmp, get_cmpfmatrix_ij(a, i, j), get_cmpfmatrix_ij(b, i, j), get_bnc_default_rounding_mode());
			set_cmpfmatrix_ij(c, i, j, tmp);
		}
	}

	//mpf_clear(tmp);
	mpc_clear(tmp);
}

/* c := a - b */
void sub_cmpfmatrix(CMPFMatrix c, CMPFMatrix a, CMPFMatrix b)
{
	long int i, j, row_dim, col_dim;
	//mpf_t tmp;
	mpc_t tmp;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_cmpfmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_cmpfmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	mpc_init2(tmp, prec_cmpfmatrix(c));
	//tmp = init2_mpc_t(get_prec_cmpfmatrix(c));
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			//mpf_sub(tmp, get_cmpfmatrix_ij(a, i, j), get_cmpfmatrix_ij(b, i, j));
			mpc_sub(tmp, get_cmpfmatrix_ij(a, i, j), get_cmpfmatrix_ij(b, i, j), get_bnc_default_rounding_mode());
			set_cmpfmatrix_ij(c, i, j, tmp);
		}
	}
	//mpf_clear(tmp);
	mpc_clear(tmp);
}

/* c := sc * a */
void cmul_cmpfmatrix(CMPFMatrix c, mpc_t sc, CMPFMatrix a)
{
	long int i, j, row_dim, col_dim;
	//mpf_t tmp;
	mpc_t tmp;

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_cmpfmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_cmpfmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	mpc_init2(tmp, prec_cmpfmatrix(c));
	//tmp = init2_mpc_t(get_prec_cmpfmatrix(c));
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			//mpf_mul(tmp, sc, get_cmpfmatrix_ij(a, i, j));
			mpc_mul(tmp, sc, get_cmpfmatrix_ij(a, i, j), get_bnc_default_rounding_mode());
			set_cmpfmatrix_ij(c, i, j, tmp);
		}
	}
	//mpf_clear(tmp);
	mpc_clear(tmp);
}


/* c = a * b */
void mul_cmpfmatrix(CMPFMatrix c, CMPFMatrix a, CMPFMatrix b)
{
	long int i, j, k;
	//mpf_t tmp, tmp1;
	mpc_t tmp, tmp1;

	/* dimension check */
	if((c->row_dim != a->row_dim) || (c->col_dim != b->row_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: mul_cmpfmatrix\n");
		return;
	}

	mpc_init2(tmp, prec_cmpfmatrix(c));
	mpc_init2(tmp1, prec_cmpfmatrix(c));
	//tmp  = init2_mpc_t(get_prec_cmpfmatrix(c));
	//tmp1 = init2_mpc_t(get_prec_cmpfmatrix(c));
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
		{
			mpc_set_ui_ui(tmp, 0UL, 0UL, get_bnc_default_rounding_mode());
			//set0_mpc_t(tmp);
			for(k = 0; k < a->col_dim; k++)
			{
				//mpf_mul(tmp1, get_cmpfmatrix_ij(a, i, k), get_cmpfmatrix_ij(b, k, j));
				//mpf_add(tmp, tmp, tmp1);
				mpc_mul(tmp1, get_cmpfmatrix_ij(a, i, k), get_cmpfmatrix_ij(b, k, j), get_bnc_default_rounding_mode());
				mpc_add(tmp, tmp, tmp1, get_bnc_default_rounding_mode());
			}
			set_cmpfmatrix_ij(c, i, j, tmp);
		}
	}
	//mpf_clear(tmp);
	//mpf_clear(tmp1);
	mpc_clear(tmp);
	mpc_clear(tmp1);
}

/* c = a^T */
void transpose_cmpfmatrix(CMPFMatrix c, CMPFMatrix a)
{
	long int i, j;

	/* Check Dimensions */
	if((c->row_dim != a->col_dim) || (c->col_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: transpose_cmpfmatrix\n");
		return;
	}
	
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_cmpfmatrix_ij(c, i, j, get_cmpfmatrix_ij(a, j, i));
	}
}

/* c := a */
void subst_cmpfmatrix(CMPFMatrix c, CMPFMatrix a)
{
	long int i, j;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_cmpfmatrix\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
			set_cmpfmatrix_ij(c, i, j, get_cmpfmatrix_ij(a, i, j));
	}
}

/* c := real_a */
void subst_cmpfmatrix_mpfmat(CMPFMatrix c, MPFMatrix a)
{
	long int i, j;
	mpc_t tmp;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_cmpfmatrix_mpfmat\n");
		return;
	}

	mpc_init2(tmp, c->prec);

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			mpc_set_ui_ui(tmp, 0UL, 0UL, get_bnc_default_rounding_mode());
			mpc_set_fr(tmp, get_mpfmatrix_ij(a, i, j), get_bnc_default_rounding_mode());

			set_cmpfmatrix_ij(c, i, j, tmp);
		}
	}

	mpc_clear(tmp);
}

/* c := 0 */
void set0_cmpfmatrix(CMPFMatrix c)
{
	long int i, j;
	//mpf_t tmp;
	mpc_t tmp;

	mpc_init2(tmp, c->prec);
	//tmp = init2_mpc_t(c->prec);
	//mpf_set_ui(tmp, 0UL);
	mpc_set_ui_ui(tmp, 0UL, 0UL, get_bnc_default_rounding_mode());

	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_cmpfmatrix_ij(c, i, j, tmp);
	}

	//mpf_clear(tmp);
	mpc_clear(tmp);
}

/* c := I */
void setI_cmpfmatrix(CMPFMatrix c)
{
	long int i, j;
	//mpf_t tmp0, tmp1;
	mpc_t tmp0, tmp1;

	mpc_init2(tmp0, c->prec);
	mpc_init2(tmp1, c->prec);
	//mpf_set_ui(tmp0, 0UL);
	//mpf_set_ui(tmp1, 1UL);
	mpc_set_ui_ui(tmp0, 0UL, 0UL, get_bnc_default_rounding_mode());
	mpc_set_ui_ui(tmp1, 1UL, 0UL, get_bnc_default_rounding_mode());

	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_cmpfmatrix_ij(c, i, j, tmp0);
		if(i < c->col_dim)
			set_cmpfmatrix_ij(c, i, i, tmp1);
	}

	//mpf_clear(tmp0);
	//mpf_clear(tmp1);
	mpc_clear(tmp0);
	mpc_clear(tmp1);
}

/* v := a * vb */
void mul_cmpfmatrix_cmpfvec(CMPFVector v, CMPFMatrix a, CMPFVector vb)
{
	long int i, j;
	//mpf_t tmp, tmp1;
	mpc_t tmp, tmp1;	

	/* Check Dimension */
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: mul_cmpfmatrix_dvec\n");
		return;
	}

	mpc_init2(tmp, prec_cmpfvector(v));
	mpc_init2(tmp1, prec_cmpfvector(v));
	//tmp  = init2_mpc_t(get_prec_cmpfvector(v));
	//tmp1 = init2_mpc_t(get_prec_cmpfvector(v));
	for(i = 0; i < a->row_dim; i++)
	{
		mpc_set_ui_ui(tmp, 0UL, 0UL, get_bnc_default_rounding_mode());
		//set0_mpc_t(tmp);
		for(j = 0; j < a->col_dim; j++)
		{
			//mpf_mul(tmp1, get_cmpfmatrix_ij(a, i, j), get_cmpfvector_i(vb, j));
			//mpf_add(tmp, tmp, tmp1);
			mpc_mul(tmp1, get_cmpfmatrix_ij(a, i, j), get_cmpfvector_i(vb, j), get_bnc_default_rounding_mode());
			mpc_add(tmp, tmp, tmp1, get_bnc_default_rounding_mode());
		}
		set_cmpfvector_i(v, i, tmp);
	}
	//mpf_clear(tmp);
	//mpf_clear(tmp1);
	mpc_clear(tmp);
	mpc_clear(tmp1);
}

/* v := a * vb */
void mul_cmpfmatrix_cmpfvec_4m(CMPFVector v, CMPFMatrix a, CMPFVector vb)
{
	long int i, j;
	unsigned long prec;
    MPFVector t1, t2, t3, t4, v_re, v_im, vb_re, vb_im;
    MPFMatrix a_re, a_im;

	/* Check Dimension */
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: mul_cmpfmatrix_dvec_4m\n");
		return;
	}
	prec = v->prec;

	v_re = init2_mpfvector(v->dim, v->prec);
	v_im = init2_mpfvector(v->dim, v->prec);
	vb_re = init2_mpfvector(vb->dim, vb->prec);
	vb_im = init2_mpfvector(vb->dim, vb->prec);
	separate_cmpfvector(vb_re, vb_im, vb);

	a_re = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);
	a_im = init2_mpfmatrix(a->row_dim, a->col_dim, a->prec);
	separate_cmpfmatrix(a_re, a_im, a);

    t1 = init2_mpfvector(v->dim, v->prec);
    t2 = init2_mpfvector(v->dim, v->prec);
    //t3 = init2_mpfvector(v->dim, v->prec);
    //t4 = init2_mpfvector(v->dim, v->prec);

    mul_mpfmatrix_mpfvec(t1, a_re, vb_re);
    mul_mpfmatrix_mpfvec(t2, a_im, vb_im);
    sub_mpfvector(v_re, t1, t2);

    //#ifdef USE_4M
        // 4M
        mul_mpfmatrix_mpfvec(t1, a_im, vb_re);
        mul_mpfmatrix_mpfvec(t2, a_re, vb_im);
        add_mpfvector(v_im, t1, t2);
    //#else // USE_4M
    /*
        // 3M
        tmp_mat = init_qdmatrix(a->re->row_dim, a->re->col_dim);
        add_qdmatrix(tmp_mat, a->re, a->im);
        add_qdvector(t3, vb->re, vb->im);
        mul_qdmatrix_qdvec(t4, tmp_mat, t3);
        sub_qdvector(v->im, t4, t1);
        sub_qdvector(v->im, v->im, t2);
        free_qdmatrix(tmp_mat);
    */
    //#endif // USE_4M

	merge_cmpfvector(v, v_re, v_im);

    free_mpfvector(vb_re);
    free_mpfvector(vb_im);
    free_mpfvector(v_re);
    free_mpfvector(v_im);
    free_mpfmatrix(a_re);
    free_mpfmatrix(a_im);

    free_mpfvector(t1);
    free_mpfvector(t2);
    //free_mpfvector(t3);
    //free_mpfvector(t4);
}


/* v := a^T * vb */
void mul_cmpfmatrixt_cmpfvec(CMPFVector v, CMPFMatrix a, CMPFVector vb)
{
	long int i, j;
	//mpf_t tmp, tmp1;
	mpc_t tmp, tmp1;

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_cmpfmatrixt_mpfvec\n");
		return;
	}

	mpc_init2(tmp, get_prec_cmpfvector(v));
	mpc_init2(tmp1, get_prec_cmpfvector(v));
	for(i = 0; i < a->col_dim; i++)
	{
		mpc_set_ui_ui(tmp, 0UL, 0UL, get_bnc_default_rounding_mode());
		//set0_mpc_t(tmp);
		for(j = 0; j < a->row_dim; j++)
		{
			//mpf_mul(tmp1, get_cmpfmatrix_ij(a, j, i), get_cmpfvector_i(vb, j));
			//mpf_add(tmp, tmp, tmp1);
			mpc_mul(tmp1, get_cmpfmatrix_ij(a, j, i), get_cmpfvector_i(vb, j), get_bnc_default_rounding_mode());
			mpc_add(tmp, tmp, tmp1, get_bnc_default_rounding_mode());
		}
		set_cmpfvector_i(v, i, tmp);
	}
	//mpf_clear(tmp);
	//mpf_clear(tmp1);
	mpc_clear(tmp);
	mpc_clear(tmp1);
}

/* a = a^(-1) */
/* square matrix only */
void inv_cmpfmatrix(CMPFMatrix a)
{
	long int i, j, k, dim;
	mpc_t ctmp, aii;
	mpf_t tmp;

	/* Check Dimensions */
	if(a->row_dim != a->col_dim)
	{
		fprintf(stderr, "ERROR: inv_cmpfmatrix\n");
		return;
	}

	mpc_init2(ctmp, get_prec_cmpfmatrix(a));
	mpc_init2(aii, get_prec_cmpfmatrix(a));
	mpf_init2(tmp, get_prec_cmpfmatrix(a));
	dim = a->row_dim;

	for(i = 0; i < dim; i++)
	{
		mpc_abs(tmp, get_cmpfmatrix_ij(a, i, i), get_bnc_default_rounding_mode());
		if(mpf_cmp_ui(tmp, 0UL) == 0) 
		{
			fprintf(stderr, "ERROR: inv_cmpfmatrix: Pivot(%ld,%ld) is zero.\n", i, i);
			return;
		}

		mpc_ui_div(aii, 1UL, get_cmpfmatrix_ij(a, i, i), get_bnc_default_rounding_mode());
		//inv_mpc_t(aii, get_cmpfmatrix_ij(a, i, i));
		set_cmpfmatrix_ij(a, i, i, aii);

		for(j = 0; j < i; j++)
		{
			//mpf_mul(tmp, get_cmpfmatrix_ij(a, i, j), aii);
			mpc_mul(ctmp, get_cmpfmatrix_ij(a, i, j), aii, get_bnc_default_rounding_mode());
			set_cmpfmatrix_ij(a, i, j, ctmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			//mpf_mul(tmp, get_cmpfmatrix_ij(a, i, j), aii);
			mpc_mul(ctmp, get_cmpfmatrix_ij(a, i, j), aii, get_bnc_default_rounding_mode());
			set_cmpfmatrix_ij(a, i, j, ctmp);
		}

		for(j = 0; j < i; j++)
		{
			for(k = 0; k < i; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
				mpc_mul(ctmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k), get_bnc_default_rounding_mode());
				mpc_sub(ctmp, get_cmpfmatrix_ij(a, j, k), ctmp, get_bnc_default_rounding_mode());
				set_cmpfmatrix_ij(a, j, k, ctmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
				mpc_mul(ctmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k), get_bnc_default_rounding_mode());
				mpc_sub(ctmp, get_cmpfmatrix_ij(a, j, k), ctmp, get_bnc_default_rounding_mode());
				set_cmpfmatrix_ij(a, j, k, ctmp);
			}
		}

		for(j = i + 1; j < dim; j++)
		{
			for(k = 0; k < i; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
				mpc_mul(ctmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k), get_bnc_default_rounding_mode());
				mpc_sub(ctmp, get_cmpfmatrix_ij(a, j, k), ctmp, get_bnc_default_rounding_mode());
				set_cmpfmatrix_ij(a, j, k, ctmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
				mpc_mul(ctmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k), get_bnc_default_rounding_mode());
				mpc_sub(ctmp, get_cmpfmatrix_ij(a, j, k), ctmp, get_bnc_default_rounding_mode());
				set_cmpfmatrix_ij(a, j, k, ctmp);
			}
		}

		for(j = 0; j < i; j++)
		{
			//mpf_neg(tmp, aii); /* tmp := -aii */
			mpc_neg(ctmp, aii, get_bnc_default_rounding_mode());
			//mpf_mul(tmp, tmp, get_cmpfmatrix_ij(a, j, i));
			mpc_mul(ctmp, ctmp, get_cmpfmatrix_ij(a, j, i), get_bnc_default_rounding_mode());
			set_cmpfmatrix_ij(a, j, i, ctmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			//mpf_neg(tmp, aii); /* tmp := -aii */
			mpc_neg(ctmp, aii, get_bnc_default_rounding_mode());
			//mpf_mul(tmp, tmp, get_cmpfmatrix_ij(a, j, i));
			mpc_mul(ctmp, ctmp, get_cmpfmatrix_ij(a, j, i), get_bnc_default_rounding_mode());
			set_cmpfmatrix_ij(a, j, i, ctmp);
		}
	}

	mpc_clear(aii);
	mpc_clear(ctmp);
	mpf_clear(tmp);
}

/* append 2023-02-26 */
// ret_real + ret_image * I := src
void separate_cmpfmatrix(MPFMatrix ret_real, MPFMatrix ret_image, CMPFMatrix src)
{
	long int i, j;

	for(i = 0; i < src->row_dim ; i++)
	{
        for(j = 0; j < src->col_dim; j++)
        {
            set_mpfmatrix_ij(ret_real , i, j, mpc_realref(get_cmpfmatrix_ij(src, i, j)));
            set_mpfmatrix_ij(ret_image, i, j, mpc_imagref(get_cmpfmatrix_ij(src, i, j)));
        }
	}
}

/* append 2023-02-27 */
// ret := src_real + src_image * I
void merge_cmpfmatrix(CMPFMatrix ret, MPFMatrix src_real, MPFMatrix src_image)
{
	long int i, j;
    mpc_t tmp;

    mpc_init2(tmp, ret->prec);

	for(i = 0; i < ret->row_dim ; i++)
	{
        for(j = 0; j < ret->col_dim; j++)
        {

            mpc_set_fr_fr(
				tmp,
				get_mpfmatrix_ij(src_real, i, j),
				get_mpfmatrix_ij(src_image, i, j),
				get_bnc_default_rounding_mode()
			);

            set_cmpfmatrix_ij(ret, i, j, tmp);
        }
	}

    mpc_clear(tmp);
}

/* vec := -vec */
void neg_cmpfvector(CMPFVector vec)
{
	long int i;
	mpc_t tmp;

	mpc_init2(tmp, vec->prec);

	for(i = 0; i < vec->dim; i++)
	{
		mpc_neg(tmp, get_cmpfvector_i(vec, i), get_bnc_default_rounding_mode());
		set_cmpfvector_i(vec, i, tmp);
	}

	mpc_clear(tmp);
}

/* Normwise relative error of vector */
void relerr_cmpfvector(mpf_t relerr, CMPFVector approx_vec, CMPFVector true_vec, int norm_type)
{
	unsigned long prec;
	mpf_t norm_true_vec, norm_diff_vec;
	CMPFVector diff_vec;

	prec = approx_vec->prec;

	diff_vec = init2_cmpfvector(approx_vec->dim, prec);
	mpf_init2(norm_true_vec, prec);
	mpf_init2(norm_diff_vec, prec);

	// diff_vec := approx_vec - true_vec
	sub_cmpfvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_cmpfvector(norm_diff_vec, diff_vec);
			normi_cmpfvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_cmpfvector(norm_diff_vec, diff_vec);
			norm1_cmpfvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_cmpfvector(norm_diff_vec, diff_vec);
			norm2_cmpfvector(norm_true_vec, true_vec);
			break;
	}

	if(mpf_cmp_ui(norm_true_vec, 0UL) != 0)
		mpf_div(relerr, norm_diff_vec, norm_true_vec);

	free_cmpfvector(diff_vec);
	mpf_clear(norm_diff_vec);
	mpf_clear(norm_true_vec);

	return;
}

/* Elementwise relative errors of vector */
void relerr_element_cmpfvector(mpf_t max_relerr, mpf_t min_relerr, mpf_t norm_relerr, 
CMPFVector approx_vec, CMPFVector true_vec, int norm_type)
{
	unsigned long prec;
	long int i;
	mpf_t abs_true_vec, abs_diff_vec, norm_diff_vec, norm_true_vec;
	CMPFVector diff_vec;

	prec = approx_vec->prec;

	diff_vec = init2_cmpfvector(approx_vec->dim, prec);
	mpf_init2(abs_true_vec, prec);
	mpf_init2(abs_diff_vec, prec);
	mpf_init2(norm_diff_vec, prec);
	mpf_init2(norm_true_vec, prec);

	// diff_vec := approx_vec - true_vec
	sub_cmpfvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_cmpfvector(norm_diff_vec, diff_vec);
			normi_cmpfvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_cmpfvector(norm_diff_vec, diff_vec);
			norm1_cmpfvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_cmpfvector(norm_diff_vec, diff_vec);
			norm2_cmpfvector(norm_true_vec, true_vec);
			break;
	}

	mpf_set(norm_relerr, norm_diff_vec);
	if(mpf_cmp_ui(norm_true_vec, 0UL) != 0)
		mpf_div(norm_relerr, norm_diff_vec, norm_true_vec);

	// relative errors of each elements
	mpf_set_ui(max_relerr, 0UL);
	normi_cmpfvector(min_relerr, diff_vec);
	for(i = 0; i < approx_vec->dim; i++)
	{
		mpc_abs(abs_diff_vec, get_cmpfvector_i(diff_vec, i), get_bnc_default_rounding_mode());
		mpc_abs(abs_true_vec, get_cmpfvector_i(true_vec, i), get_bnc_default_rounding_mode());
		if(mpf_cmp_ui(abs_true_vec, 0UL) != 0)
			mpf_div(abs_diff_vec, abs_diff_vec, abs_true_vec);
		
		if(mpf_cmp(max_relerr, abs_diff_vec) < 0)
			mpf_set(max_relerr, abs_diff_vec);
		if(mpf_cmp(min_relerr, abs_diff_vec) > 0)
			mpf_set(min_relerr, abs_diff_vec);
	}

	free_cmpfvector(diff_vec);// Fix! 2012-06-03 by T.Kouya
	mpf_clear(abs_true_vec);
	mpf_clear(abs_diff_vec);
	mpf_clear(norm_true_vec);
	mpf_clear(norm_diff_vec);

	return;
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_cmpfmatrix(CMPFMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
	long int i, j, true_end;
	mpc_t tmp;

	mpc_init2(tmp, mat->prec);

	true_end = (col_end > mat->col_dim) ? mat->col_dim : col_end;

	for(i = col_start; i < true_end; i++)
	{
		mpc_set(tmp, get_cmpfmatrix_ij(mat, row_index0, i), get_bnc_default_rounding_mode());
		set_cmpfmatrix_ij(mat, row_index0, i, get_cmpfmatrix_ij(mat, row_index1, i));
		set_cmpfmatrix_ij(mat, row_index1, i, tmp);
	}

	mpc_clear(tmp);
}

// 2024-08-02 (Fri) T.Kouya
// ret := (cmpf)a
void subst_cmpfvector_cdvec(CMPFVector ret, CDVector a)
{
	long int i;
	mpc_t tmp;

	mpc_init2(tmp, ret->prec);
	for(i = 0; i < a->dim; i++)
		mpc_set_dc(tmp, get_cdvector_i(a, i), MPFR_RNDN);

	mpc_clear(tmp);
	
}

// 2024-08-02 (Fri) T.Kouya
// ret := (cmpf)a
void subst_cmpfmatrix_cdmat(CMPFMatrix ret, CDMatrix a)
{
	long int i, j;
	mpc_t tmp;

	mpc_init2(tmp, ret->prec);
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
			mpc_set_dc(tmp, get_cdmatrix_ij(a, i, j), MPFR_RNDN);
	}

	mpc_clear(tmp);
	
}
#endif // USE_GMP
