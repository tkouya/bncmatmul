/********************************************************************************/
/* cmpflinear.c: MPC-based Vector, Matrix                                       */
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

//include "bnc.h"
#include "cmpflinear.h"

#ifdef USE_GMP

#if 0
// ret := a * b
void _bnc_mpc_mul_cd(mpc_t ret, mpc_t a, double _Complex b)
{
	// 4M method
	mpf_ptr ret_re, ret_im, a_re, a_im;
	mpf_t tmp;

	mpf_init2(tmp, mpc_get_prec(ret));

	ret_re = mpc_realref(ret);
	ret_im = mpc_imagref(ret);
	a_re = mpc_realref(a);
	a_im = mpc_imagref(a);

	// real part
	mpf_mul_d(ret_re, a_re, creal(b));
	mpf_mul_d(tmp, a_im, cimag(b));
	mpf_sub(ret_re, ret_re, tmp);

	// imag part
	mpf_mul_d(ret_im, a_re, cimag(b));
	mpf_mul_d(tmp, a_im, creal(b));
	mpf_add(ret_im, ret_im, tmp);

	mpf_clear(tmp);
}
#endif // 0

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
		mpc_set_ui_ui((mpc_ptr)(ret->element + i), 0UL, 0UL, get_bnc_default_rounding_mode_c());
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
		mpc_set_ui_ui((mpc_ptr)(ret->element + i), 0UL, 0UL, get_bnc_default_rounding_mode_c());
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

mpc_ptr get_cmpfvector_i(CMPFVector vec, long int index)
{
	return (mpc_ptr)(vec->element + index);
}

void set_cmpfvector_i(CMPFVector vec, long int index, mpc_t val)
{
	mpc_set((mpc_ptr)(vec->element + index), val, get_bnc_default_rounding_mode_c());
	//subst_mpc_t((vec->element + index), val);
}

// vec[i] := val + 0 * I
void set_cmpfvector_i_real(CMPFVector vec, long int index, mpf_t val)
{
	mpc_set_fr((mpc_ptr)(vec->element + index), val, get_bnc_default_rounding_mode_c());
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
	mpc_set_dc(*(vec->element + index), val, get_bnc_default_rounding_mode_c());
	//set_mpc_t_d((vec->element + index), val);
}

void set_cmpfvector_i_str(CMPFVector vec, long int index, const char *re_str, const char *im_str, int base)
{
	mpf_set_str(mpc_realref((mpc_ptr)(vec->element + index)), re_str, base);
	mpf_set_str(mpc_imagref((mpc_ptr)(vec->element + index)), im_str, base);
}

void set_cmpfvector_i_ui(CMPFVector vec, long int index, unsigned long val)
{
	mpc_set_ui(*(vec->element + index), val, get_bnc_default_rounding_mode_c());
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

/*************************************************/
/* Function for Matrix Types                     */
/*                                               */
/* Initialize:                                   */
/*   CMPFMatrix init_cmpfmatrix(long int row_dimension, long int col_dimension)*/
/*   CMPFMatrix init2_cmpfmatrix(long int row_dimension, long int col_dimension, unsigned long mbits) */
/* Free:                                         */
/*   void free_cmpfmatrix(CMPFMatrix mat)          */
/* Get & Set:                                      */
/*   double get_cmpfmatrix_ij(CMPFMatrix mat, long int row_index, long int col_index) */
/*   void set_cmpfmatrix_ij(CMPFMatrix mat, long int row_index, long int col_index, mpc_t val) */
/*   void set_cmpfmatrix_ij_d(CMPFMatrix mat, long int row_index, long int col_index, double _Complexval) */
/* Output:                                       */
/*   void print_cmpfmatrix(CMPFMatrix mat)         */
/*************************************************/
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
			mpc_set_ui_ui((mpc_ptr)(ret->element + i * col_dimension + j), 0UL, 0UL, get_bnc_default_rounding_mode_c()); // fix! 2026-02-01 by T.Kouya
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
			mpc_set_ui_ui((mpc_ptr)(ret->element + i * col_dimension + j), 0UL, 0UL, get_bnc_default_rounding_mode_c()); // fix! 2026-02-01 by T.Kouya
			//set0_mpc_t((ret->element + i * col_dimension + j));
		}

	ret->row_dim = row_dimension;
	ret->col_dim = col_dimension;

	ret->prec = mbits;

	return ret;
}

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

mpc_ptr get_cmpfmatrix_ij(CMPFMatrix mat, long int row_index, long int col_index)
{
	return (mpc_ptr)(mat->element + row_index * mat->col_dim + col_index);
}

void set_cmpfmatrix_ij(CMPFMatrix mat, long int row_index, long int col_index, mpc_t val)
{
	mpc_set((mpc_ptr)(mat->element + row_index * mat->col_dim + col_index), val, get_bnc_default_rounding_mode_c());
}

void set_cmpfmatrix_ij_d(CMPFMatrix mat, long int row_index, long int col_index, double _Complex val)
{
	mpc_set_d_d((mpc_ptr)(mat->element + row_index * mat->col_dim + col_index), val, (double)0.0, get_bnc_default_rounding_mode_c());
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
		mpc_add(tmp, get_cmpfvector_i(a, i), get_cmpfvector_i(b, i), get_bnc_default_rounding_mode_c());
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
		mpc_add(tmp, get_cmpfvector_i(c, i), get_cmpfvector_i(a, i), get_bnc_default_rounding_mode_c());
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
		mpc_sub(tmp, get_cmpfvector_i(a, i), get_cmpfvector_i(b, i), get_bnc_default_rounding_mode_c());
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
		mpc_sub(tmp, get_cmpfvector_i(c, i), get_cmpfvector_i(a, i), get_bnc_default_rounding_mode_c());
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
		mpc_mul(tmp, val, get_cmpfvector_i(a, i), get_bnc_default_rounding_mode_c());
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
		mpc_mul_fr(tmp, val, get_mpfvector_i(a, i), get_bnc_default_rounding_mode_c());
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
		mpc_mul(tmp, val, get_cmpfvector_i(c, i), get_bnc_default_rounding_mode_c());
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
		mpc_mul(tmp2, val, get_cmpfvector_i(b, i), get_bnc_default_rounding_mode_c());
		//mpf_add(tmp, get_cmpfvector_i(a, i), tmp2);
		mpc_add(tmp, get_cmpfvector_i(a, i), tmp2, get_bnc_default_rounding_mode_c());
		set_cmpfvector_i(c, i, tmp);
	}

	//mpf_clear(tmp);
	//mpf_clear(tmp2);
	mpc_clear(tmp);
	mpc_clear(tmp2);
}

/* c = a - val * b */
void sub_cmul_cmpfvector(CMPFVector c, CMPFVector a, mpc_t val, CMPFVector b)
{
	long int i;
	//mpf_t tmp, tmp2;
	mpc_t tmp, tmp2;
	unsigned long int prec;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_cmpfvector\n");
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
		mpc_mul(tmp2, val, get_cmpfvector_i(b, i), get_bnc_default_rounding_mode_c());
		//mpf_add(tmp, get_cmpfvector_i(a, i), tmp2);
		mpc_sub(tmp, get_cmpfvector_i(a, i), tmp2, get_bnc_default_rounding_mode_c());
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
	mpc_t tmp, a_conj;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: ip_cmpfvector\n");
		return;
	}

	mpc_init2(tmp, mpc_get_prec(ret));
	mpc_init2(a_conj, mpc_get_prec(ret));

	//tmp = init2_mpc_t(get_prec_mpc_t(ret));

	mpc_set_ui_ui(ret, 0UL, 0UL, get_bnc_default_rounding_mode_c()); /* ret := 0 */
	//set0_mpc_t(ret);
	for(i = 0; i < a->dim; i++)
	{
		//mpf_mul(tmp, get_cmpfvector_i(a, i), get_cmpfvector_i(b, i));
		mpc_conj(a_conj, get_cmpfvector_i(a, i), get_bnc_default_rounding_mode_c());
		mpc_mul(tmp, a_conj, get_cmpfvector_i(b, i), get_bnc_default_rounding_mode_c());
		//mpf_add(ret, ret, tmp);
		mpc_add(ret, ret, tmp, get_bnc_default_rounding_mode_c());
	}

	//mpf_clear(tmp);
	mpc_clear(tmp);
	mpc_clear(a_conj);
}

/* ret = Sum a[i] * b[i] */
void dotp_cmpfvector(mpc_t ret, CMPFVector a, CMPFVector b)
{
	long int i;
	//mpf_t tmp;
	mpc_t tmp;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: dotp_cmpfvector\n");
		return;
	}

	mpc_init2(tmp, mpc_get_prec(ret));
	//tmp = init2_mpc_t(get_prec_mpc_t(ret));

	mpc_set_ui_ui(ret, 0UL, 0UL, get_bnc_default_rounding_mode_c()); /* ret := 0 */
	//set0_mpc_t(ret);
	for(i = 0; i < a->dim; i++)
	{
		//mpf_mul(tmp, get_cmpfvector_i(a, i), get_cmpfvector_i(b, i));
		mpc_mul(tmp, get_cmpfvector_i(a, i), get_cmpfvector_i(b, i), get_bnc_default_rounding_mode_c());
		//mpf_add(ret, ret, tmp);
		mpc_add(ret, ret, tmp, get_bnc_default_rounding_mode_c());
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
		mpc_abs(tmp, get_cmpfvector_i(a, i), get_bnc_default_rounding_mode_c());
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

	mpc_abs(ret, get_cmpfvector_i(a, 0), get_bnc_default_rounding_mode_c());
	//abs_mpc_t(ret, get_cmpfvector_i(a, 0));
	for(i = 1; i < a->dim; i++)
	{
		mpc_abs(tmp, get_cmpfvector_i(a, i), get_bnc_default_rounding_mode_c());
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

/* c := conj(a) */
void conj_cmpfvector(CMPFVector c, CMPFVector a)
{
	long int i;
	mpc_t a_conj;

	mpc_init2(a_conj, a->prec);

	for(i = 0; i < a->dim; i++)
	{
		mpc_conj(a_conj, get_cmpfvector_i(a, i), get_bnc_default_rounding_mode_c());
		set_cmpfvector_i(c, i, a_conj); // get_cmpfvector_i(a, i));
	}

	mpc_clear(a_conj);
}

/* c := real_a */
void subst_cmpfvector_mpfvec(CMPFVector c, MPFVector a)
{
	long int i;
	mpc_t tmp;

	mpc_init2(tmp, c->prec);

	for(i = 0; i < a->dim; i++)
	{
		mpc_set_ui_ui(tmp, 0UL, 0UL, get_bnc_default_rounding_mode_c());
		mpc_set_fr(tmp, get_mpfvector_i(a, i), get_bnc_default_rounding_mode_c());

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
	mpc_set_ui_ui(tmp, 0UL, 0UL, get_bnc_default_rounding_mode_c());
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
			get_bnc_default_rounding_mode_c()
		);

        set_cmpfvector_i(ret, i, tmp);
	}

    mpc_clear(tmp);
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
			mpc_abs(tmp, get_cmpfmatrix_ij(mat, i, j), get_bnc_default_rounding_mode_c());
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
			mpc_abs(tmp, get_cmpfmatrix_ij(mat, i, j), get_bnc_default_rounding_mode_c());
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
			mpc_add(tmp, get_cmpfmatrix_ij(a, i, j), get_cmpfmatrix_ij(b, i, j), get_bnc_default_rounding_mode_c());
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
			mpc_sub(tmp, get_cmpfmatrix_ij(a, i, j), get_cmpfmatrix_ij(b, i, j), get_bnc_default_rounding_mode_c());
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
			mpc_mul(tmp, sc, get_cmpfmatrix_ij(a, i, j), get_bnc_default_rounding_mode_c());
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
	if((c->row_dim != a->row_dim) || (c->col_dim != b->col_dim) || (a->col_dim != b->row_dim))
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
			mpc_set_ui_ui(tmp, 0UL, 0UL, get_bnc_default_rounding_mode_c());
			//set0_mpc_t(tmp);
			for(k = 0; k < a->col_dim; k++)
			{
				//mpf_mul(tmp1, get_cmpfmatrix_ij(a, i, k), get_cmpfmatrix_ij(b, k, j));
				//mpf_add(tmp, tmp, tmp1);
				mpc_mul(tmp1, get_cmpfmatrix_ij(a, i, k), get_cmpfmatrix_ij(b, k, j), get_bnc_default_rounding_mode_c());
				mpc_add(tmp, tmp, tmp1, get_bnc_default_rounding_mode_c());
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

/* c = conj(a)^T */
void star_cmpfmatrix(CMPFMatrix c, CMPFMatrix a)
{
	long int i, j;
	mpc_t a_conj;

	/* Check Dimensions */
	if((c->row_dim != a->col_dim) || (c->col_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: star_cmpfmatrix\n");
		return;
	}

	mpc_init2(a_conj, a->prec);
	
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
		{
			mpc_conj(a_conj,  get_cmpfmatrix_ij(a, j, i), get_bnc_default_rounding_mode_c());
			set_cmpfmatrix_ij(c, i, j, a_conj); //get_cmpfmatrix_ij(a, j, i));
		}
	}

	mpc_clear(a_conj);
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

/* c := -a */
void neg_cmpfmatrix(CMPFMatrix c, CMPFMatrix a)
{
	long int i, j;
	mpc_t a_conj;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: cneg_cmpfmatrix\n");
		return;
	}

	mpc_init2(a_conj, a->prec);

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			mpc_neg(a_conj, get_cmpfmatrix_ij(a, i, j), get_bnc_default_rounding_mode_c());
			set_cmpfmatrix_ij(c, i, j, a_conj); //get_cmpfmatrix_ij(a, i, j));
		}
	}

	mpc_clear(a_conj);

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
			mpc_set_ui_ui(tmp, 0UL, 0UL, get_bnc_default_rounding_mode_c());
			mpc_set_fr(tmp, get_mpfmatrix_ij(a, i, j), get_bnc_default_rounding_mode_c());

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
	mpc_set_ui_ui(tmp, 0UL, 0UL, get_bnc_default_rounding_mode_c());

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
	mpc_set_ui_ui(tmp0, 0UL, 0UL, get_bnc_default_rounding_mode_c());
	mpc_set_ui_ui(tmp1, 1UL, 0UL, get_bnc_default_rounding_mode_c());

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
		mpc_set_ui_ui(tmp, 0UL, 0UL, get_bnc_default_rounding_mode_c());
		//set0_mpc_t(tmp);
		for(j = 0; j < a->col_dim; j++)
		{
			//mpf_mul(tmp1, get_cmpfmatrix_ij(a, i, j), get_cmpfvector_i(vb, j));
			//mpf_add(tmp, tmp, tmp1);
			mpc_mul(tmp1, get_cmpfmatrix_ij(a, i, j), get_cmpfvector_i(vb, j), get_bnc_default_rounding_mode_c());
			mpc_add(tmp, tmp, tmp1, get_bnc_default_rounding_mode_c());
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
		mpc_set_ui_ui(tmp, 0UL, 0UL, get_bnc_default_rounding_mode_c());
		//set0_mpc_t(tmp);
		for(j = 0; j < a->row_dim; j++)
		{
			//mpf_mul(tmp1, get_cmpfmatrix_ij(a, j, i), get_cmpfvector_i(vb, j));
			//mpf_add(tmp, tmp, tmp1);
			mpc_mul(tmp1, get_cmpfmatrix_ij(a, j, i), get_cmpfvector_i(vb, j), get_bnc_default_rounding_mode_c());
			mpc_add(tmp, tmp, tmp1, get_bnc_default_rounding_mode_c());
		}
		set_cmpfvector_i(v, i, tmp);
	}
	//mpf_clear(tmp);
	//mpf_clear(tmp1);
	mpc_clear(tmp);
	mpc_clear(tmp1);
}

/* v := conj(a)^T * vb */
void mul_cmpfmatrixs_cmpfvec(CMPFVector v, CMPFMatrix a, CMPFVector vb)
{
	long int i, j;
	//mpf_t tmp, tmp1;
	mpc_t tmp, tmp1, a_conj;

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_cmpfmatrixs_mpfvec\n");
		return;
	}

	mpc_init2(tmp, get_prec_cmpfvector(v));
	mpc_init2(tmp1, get_prec_cmpfvector(v));
	mpc_init2(a_conj, get_prec_cmpfvector(v));

	for(i = 0; i < a->col_dim; i++)
	{
		mpc_set_ui_ui(tmp, 0UL, 0UL, get_bnc_default_rounding_mode_c());
		//set0_mpc_t(tmp);
		for(j = 0; j < a->row_dim; j++)
		{
			//mpf_mul(tmp1, get_cmpfmatrix_ij(a, j, i), get_cmpfvector_i(vb, j));
			//mpf_add(tmp, tmp, tmp1);
			mpc_conj(a_conj, get_cmpfmatrix_ij(a, j, i), get_bnc_default_rounding_mode_c());
			//mpc_mul(tmp1, get_cmpfmatrix_ij(a, j, i), get_cmpfvector_i(vb, j), get_bnc_default_rounding_mode_c());
			mpc_mul(tmp1, a_conj, get_cmpfvector_i(vb, j), get_bnc_default_rounding_mode_c());
			mpc_add(tmp, tmp, tmp1, get_bnc_default_rounding_mode_c());
		}
		set_cmpfvector_i(v, i, tmp);
	}
	//mpf_clear(tmp);
	//mpf_clear(tmp1);
	mpc_clear(tmp);
	mpc_clear(tmp1);
	mpc_clear(a_conj);
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
		mpc_abs(tmp, get_cmpfmatrix_ij(a, i, i), get_bnc_default_rounding_mode_c());
		if(mpf_cmp_ui(tmp, 0UL) == 0) 
		{
			fprintf(stderr, "ERROR: inv_cmpfmatrix: Pivot(%ld,%ld) is zero.\n", i, i);
			return;
		}

		mpc_ui_div(aii, 1UL, get_cmpfmatrix_ij(a, i, i), get_bnc_default_rounding_mode_c());
		//inv_mpc_t(aii, get_cmpfmatrix_ij(a, i, i));
		set_cmpfmatrix_ij(a, i, i, aii);

		for(j = 0; j < i; j++)
		{
			//mpf_mul(tmp, get_cmpfmatrix_ij(a, i, j), aii);
			mpc_mul(ctmp, get_cmpfmatrix_ij(a, i, j), aii, get_bnc_default_rounding_mode_c());
			set_cmpfmatrix_ij(a, i, j, ctmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			//mpf_mul(tmp, get_cmpfmatrix_ij(a, i, j), aii);
			mpc_mul(ctmp, get_cmpfmatrix_ij(a, i, j), aii, get_bnc_default_rounding_mode_c());
			set_cmpfmatrix_ij(a, i, j, ctmp);
		}

		for(j = 0; j < i; j++)
		{
			for(k = 0; k < i; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
				mpc_mul(ctmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k), get_bnc_default_rounding_mode_c());
				mpc_sub(ctmp, get_cmpfmatrix_ij(a, j, k), ctmp, get_bnc_default_rounding_mode_c());
				set_cmpfmatrix_ij(a, j, k, ctmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
				mpc_mul(ctmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k), get_bnc_default_rounding_mode_c());
				mpc_sub(ctmp, get_cmpfmatrix_ij(a, j, k), ctmp, get_bnc_default_rounding_mode_c());
				set_cmpfmatrix_ij(a, j, k, ctmp);
			}
		}

		for(j = i + 1; j < dim; j++)
		{
			for(k = 0; k < i; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
				mpc_mul(ctmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k), get_bnc_default_rounding_mode_c());
				mpc_sub(ctmp, get_cmpfmatrix_ij(a, j, k), ctmp, get_bnc_default_rounding_mode_c());
				set_cmpfmatrix_ij(a, j, k, ctmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				//mpf_mul(tmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k));
				//mpf_sub(tmp, get_cmpfmatrix_ij(a, j, k), tmp);
				mpc_mul(ctmp, get_cmpfmatrix_ij(a, j, i), get_cmpfmatrix_ij(a, i, k), get_bnc_default_rounding_mode_c());
				mpc_sub(ctmp, get_cmpfmatrix_ij(a, j, k), ctmp, get_bnc_default_rounding_mode_c());
				set_cmpfmatrix_ij(a, j, k, ctmp);
			}
		}

		for(j = 0; j < i; j++)
		{
			//mpf_neg(tmp, aii); /* tmp := -aii */
			mpc_neg(ctmp, aii, get_bnc_default_rounding_mode_c());
			//mpf_mul(tmp, tmp, get_cmpfmatrix_ij(a, j, i));
			mpc_mul(ctmp, ctmp, get_cmpfmatrix_ij(a, j, i), get_bnc_default_rounding_mode_c());
			set_cmpfmatrix_ij(a, j, i, ctmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			//mpf_neg(tmp, aii); /* tmp := -aii */
			mpc_neg(ctmp, aii, get_bnc_default_rounding_mode_c());
			//mpf_mul(tmp, tmp, get_cmpfmatrix_ij(a, j, i));
			mpc_mul(ctmp, ctmp, get_cmpfmatrix_ij(a, j, i), get_bnc_default_rounding_mode_c());
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
				get_bnc_default_rounding_mode_c()
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
		mpc_neg(tmp, get_cmpfvector_i(vec, i), get_bnc_default_rounding_mode_c());
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
		mpc_abs(abs_diff_vec, get_cmpfvector_i(diff_vec, i), get_bnc_default_rounding_mode_c());
		mpc_abs(abs_true_vec, get_cmpfvector_i(true_vec, i), get_bnc_default_rounding_mode_c());
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
		mpc_set(tmp, get_cmpfmatrix_ij(mat, row_index0, i), get_bnc_default_rounding_mode_c());
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
	{
		mpc_set_dc(tmp, get_cdvector_i(a, i), MPFR_RNDN);
		set_cmpfvector_i(ret, i, tmp); // Fix! 2024-10-25(Fri) T.Kouya
	}

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
		{
			mpc_set_dc(tmp, get_cdmatrix_ij(a, i, j), MPFR_RNDN);
			set_cmpfmatrix_ij(ret, i, j, tmp); // Fix! 2024-10-25(Fri) T.Kouya
		}
	}

	mpc_clear(tmp);
	
}

// 2026-02-03(Tue) T.Kouya
// absmax_cmpfvector
void absmax_cmpfvector(mpf_t ret, long int *max_index, CMPFVector vec)
{
    long int i, max_i, dim = vec->dim;
    mpf_t abs_val;
	mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

	mpf_init2(abs_val, mpf_get_prec(ret));

	max_i = 0;
    //ret = 0.0;
	mpf_set_ui(abs_val, 0UL);

	for(i = 0; i < dim; i++)
    {
        mpc_abs(abs_val, get_cmpfvector_i(vec, i), rndc);
        if(mpf_cmp(ret, abs_val) < 0)
        {
            //ret = abs_val;f
			mpf_set(ret, abs_val);
            max_i = i;
        }
    }

    if(max_index != NULL)
        *max_index = max_i;

	mpf_clear(abs_val);

	return;
}

// 2026-02-03(Tue) T.Kouya
// absmax_row_cmpfmatrix
void absmax_row_cmpfmatrix(mpf_t ret, long int *max_j, long int row_index, CMPFMatrix mat)
{
    long int j, max_index = 0;
    mpf_t abs_aij;
	mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

	mpf_init2(abs_aij, mpf_get_prec(ret));

	mpc_abs(ret, get_cmpfmatrix_ij(mat, row_index, 0), rndc);

	for(j = 1; j < mat->col_dim; j++)
	{
		mpc_abs(abs_aij, get_cmpfmatrix_ij(mat, row_index, j), rndc);
		if(mpf_cmp(abs_aij, ret) > 0)
        {
			mpf_set(ret, abs_aij);
            max_index = j;
        }
	}

    if(max_j != NULL)
        *max_j = max_index;

	mpf_clear(abs_aij);

    return;
}

// 2022-11-17(Thu) T.Kouya
// absmax_col_cmpfmatrix
void absmax_col_cpfmatrix(mpf_t ret, long int *max_i, long int col_index, CMPFMatrix mat)
{
    long int i, max_index = 0;
    mpf_t abs_aij;
	mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

	mpf_init2(abs_aij, mpf_get_prec(ret));

	mpc_abs(ret, get_cmpfmatrix_ij(mat, col_index, 0), rndc);

	for(i = 1; i < mat->row_dim; i++)
	{
		mpc_abs(abs_aij, get_cmpfmatrix_ij(mat, i, col_index), rndc);
		if(mpf_cmp(abs_aij, ret) > 0)
        {
			mpf_set(ret, abs_aij);
            max_index = i;
        }
	}

    if(max_i != NULL)
        *max_i = max_index;

    return;
}

// 2026-02-03(Tue) T.Kouya
// absmax_mpfmatrix
void absmax_cmpfmatrix(mpf_t ret, long int *max_i, long int *max_j, CMPFMatrix mat)
{
    long int i, j, max_row_index = 0, max_col_index = 0;
    mpf_t abs_aij;
	mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

	mpf_init2(abs_aij, mpf_get_prec(ret));

	mpc_abs(ret, get_cmpfmatrix_ij(mat, 0, 0), rndc);
	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
		{
			mpc_abs(abs_aij, get_cmpfmatrix_ij(mat, i, j), rndc);
			if(mpf_cmp(abs_aij, ret) > 0)
			{
				mpf_set(ret, abs_aij);
				max_row_index = i;
				max_col_index = j;
			}
		}
	}

    if(max_i != NULL)
        *max_i = max_row_index;
	if(max_j != NULL)
		*max_j = max_col_index;

	mpf_clear(abs_aij);

	return;
}

#endif // USE_GMP
