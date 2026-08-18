/********************************************************************************/
/* linear.c: Vector, Matrix                                                     */
/* Copyright (c) 2000-2011 Tomonori Kouya                                       */
/*                                                                              */
/* Version 0.1, 2005.07/11: append copy_*vector_ij                              */
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
#include "dlinear.h"
#include "mpflinear.h"

/*************************************************/
/* Functions for Vector Types                    */
/*                                               */
/* Initialize:                                   */
/*   MPFVector init_mpfvector(long int dimension)*/
/*   MPFVector init2_mpfvector(long int dimension, unsigned long mbits)*/
/* Free:                                         */
/*   void free_mpfvector(MPFVector vec)          */
/* Get & Set Values:                             */
/*   mpf_t *get_mpfvector_i(MPFVector vec, long int index) */
/*   void set_mpfvector_i(MPFVector vec, long int index, mpf_t val) */
/*   void set_mpfvector_i_d(MPFVector vec, long int index, double val) */
/* Output:                                       */
/*   void print_mpfvector(MPFVector vec)         */
/*   void print_fdmpfvector(FVector fv, DVector dv, MPFVector mpfv) */
/*************************************************/
#ifdef USE_GMP

MPFVector init_mpfvector(long int dimension)
{
	MPFVector ret = NULL;
	long int i;

	if(dimension <= 0)
	{
		fprintf(stderr, "ERROR: init_mpfvector\n");
		return ret;
	}

	ret = (MPFVector)malloc(sizeof(mpfvector));
	if(ret == NULL)
		return ret;

	ret->element = (mpf_t *)calloc(sizeof(mpf_t), dimension);
//	ret->element = (mpf_t *)malloc(sizeof(mpf_t) * dimension);
	if(ret->element == NULL)
		return NULL;

	/* All 0 */
	for(i = 0; i < dimension; i++)
	{
		mpf_init((mpf_ptr)(ret->element + i));
		mpf_set_ui((mpf_ptr)(ret->element + i), 0UL);
		if((ret->element+i) == NULL)
			return NULL;
	}

	ret->dim = dimension;
	ret->real_dim = dimension; // 2022-11-18(Fri) T.Kouya

	ret->prec = get_bnc_default_prec();

	return ret;
}

/* mbits ... A number of at least bits of mantissa */
MPFVector init2_mpfvector(long int dimension, unsigned long int mbits)
{
	MPFVector ret = NULL;
	long int i;

	if(dimension <= 0)
	{
		fprintf(stderr, "ERROR: init2_mpfvector\n");
		return ret;
	}

	ret = (MPFVector)malloc(sizeof(mpfvector));
	if(ret == NULL)
		return ret;

	ret->element = (mpf_t *)calloc(sizeof(mpf_t), dimension);
//	ret->element = (mpf_t *)malloc(sizeof(mpf_t) * dimension);
	if(ret->element == NULL)
		return NULL;

	/* All 0 */
	for(i = 0; i < dimension; i++)
	{
		mpf_init2((mpf_ptr)(ret->element + i), mbits);
		mpf_set_ui((mpf_ptr)(ret->element + i), 0UL);
		if((ret->element+i) == NULL)
			return NULL;
	}

	ret->dim = dimension;
	ret->real_dim = dimension; // 2022-11-18(Fri) T.Kouya

	ret->prec = mbits;

	return ret;
}
#endif // USE_GMP

#ifdef USE_GMP
void free_mpfvector(MPFVector vec)
{
	long int i;

	if(vec == NULL)
		return;

	if(vec->element != NULL)
	{
		for(i = 0; i < vec->dim; i++)
			mpf_clear((mpf_ptr)(vec->element + i));
		free(vec->element); // Fix! 2012-06-03 by T.Kouya
	}

//	free(&(vec->dim));
//	free(&(vec->prec));
	free(vec);

}
#endif

#ifdef USE_GMP
#ifndef GET_VECTOR_I
mpf_ptr get_mpfvector_i(MPFVector vec, long int index)
{
	return *(vec->element + index);
}
#endif // GET_VECTOR_I
#endif

#ifdef USE_GMP
void set_mpfvector_i(MPFVector vec, long int index, mpf_t val)
{
	mpf_set((mpf_ptr)(vec->element + index), val);
}

void set_mpfvector_i_d(MPFVector vec, long int index, double val)
{
	mpf_set_d(*(vec->element + index), val);
}

void set_mpfvector_i_str(MPFVector vec, long int index, const char *str, int base)
{
	mpf_set_str(*(vec->element + index), str, base);
}

void set_mpfvector_i_ui(MPFVector vec, long int index, unsigned long val)
{
	mpf_set_ui(*(vec->element + index), val);
}

/* get precision of MPFVector */
unsigned long int prec_mpfvector(MPFVector vec)
{
	return vec->prec;
}

/* search minimam precision in MPFVector */
unsigned long int minprec_mpfvector(MPFVector vec)
{
	unsigned long int prec, tmp;
	long int i;

	prec = mpf_get_prec(get_mpfvector_i(vec, 0));
	for(i = 1; i < vec->dim; i++)
	{
		tmp = mpf_get_prec(get_mpfvector_i(vec, i));
		if(prec > tmp)
			prec = tmp;
	}

	return prec;
}

/* search maximam precision in MPFVector */
unsigned long int maxprec_mpfvector(MPFVector vec)
{
	unsigned long int prec, tmp;
	long int i;

	prec = mpf_get_prec(get_mpfvector_i(vec, 0));
	for(i = 1; i < vec->dim; i++)
	{
		tmp = mpf_get_prec(get_mpfvector_i(vec, i));
		if(prec < tmp)
			prec = tmp;
	}

	return prec;
}
#endif // USE_GMP

#ifdef USE_GMP
void print_mpfvector(MPFVector vec)
{
	long int i;

	for(i = 0; i < vec->dim; i++)
	{
		printf("%5ld ", i);
		mpf_out_str(stdout, 10, 0, get_mpfvector_i(vec, i));
		printf("\n");
	}
}
#endif // USE_GMP

/*************************************************/
/* Function for Matrix Types                     */
/*                                               */
/* Initialize:                                   */
/*   MPFMatrix init_mpfmatrix(long int row_dimension, long int col_dimension) */
/*   MPFMatrix init2_mpfmatrix(long int row_dimension, long int col_dimension, unsigned long mbits) */
/* Free:                                         */
/*   void free_mpfmatrix(MPFMatrix mat)          */
/* Get & Set:                                    */
/*   float get_mpfmatrix_ij(MPFMatrix mat, long int row_index, long int col_index) */
/*   void set_mpfmatrix_ij(MPFMatrix mat, long int row_index, long int col_index, mpf_t val) */
/*   void set_mpfmatrix_ij_d(MPFMatrix mat, long int row_index, long int col_index, double val) */
/* Output:                                       */
/*   void print_mpfmatrix(MPFMatrix mat)         */
/*************************************************/

#ifdef USE_GMP

MPFMatrix init_mpfmatrix(long int row_dimension, long int col_dimension)
{
	MPFMatrix ret = NULL;
	long int i, j;

	if(row_dimension <= 0 || col_dimension <= 0)
	{
		fprintf(stderr, "ERROR: init_mpfmatrix\n");
		return ret;
	}

	ret = (MPFMatrix)malloc(sizeof(mpfmatrix));
	if(ret == NULL)
		return ret;

	ret->element = (mpf_t *)calloc(sizeof(mpf_t), row_dimension * col_dimension);
	if(ret->element == NULL)
		return ret;

	/* All 0 */
	for(i = 0; i < row_dimension; i++)
		for(j = 0; j < col_dimension; j++)
		{
			mpf_init(*(ret->element + i * col_dimension + j));
			mpf_set_ui(*(ret->element + i * col_dimension + j), 0UL);
		}

	ret->row_dim = row_dimension;
	ret->col_dim = col_dimension;

	ret->real_row_dim = row_dimension; // 2022-11-18(Fri) T.Kouya
	ret->real_col_dim = col_dimension; // 2022-11-18(Fri) T.Kouya

	ret->prec = get_bnc_default_prec();

	return ret;
}
/* mbits ... A number of at least bits of mantissa */
MPFMatrix init2_mpfmatrix(long int row_dimension, long int col_dimension, unsigned long mbits)
{
	MPFMatrix ret = NULL;
	long int i, j;

	if(row_dimension <= 0 || col_dimension <= 0)
	{
		fprintf(stderr, "ERROR: init_mpfmatrix\n");
		return ret;
	}

	ret = (MPFMatrix)malloc(sizeof(mpfmatrix));
	if(ret == NULL)
		return ret;

	ret->element = (mpf_t *)calloc(sizeof(mpf_t), row_dimension * col_dimension);
	if(ret->element == NULL)
		return ret;

	/* All 0 */
	for(i = 0; i < row_dimension; i++)
		for(j = 0; j < col_dimension; j++)
		{
			mpf_init2(*(ret->element + i * col_dimension + j), mbits);
			mpf_set_ui(*(ret->element + i * col_dimension + j), 0UL);
		}

	ret->row_dim = row_dimension;
	ret->col_dim = col_dimension;

	ret->real_row_dim = row_dimension; // 2022-11-18(Fri) T.Kouya
	ret->real_col_dim = col_dimension; // 2022-11-18(Fri) T.Kouya

	ret->prec = mbits;

	return ret;
}
#endif // USE_GMP

#ifdef USE_GMP
void free_mpfmatrix(MPFMatrix mat)
{
	long int i, j;

	if(mat == NULL)
		return;

	if(mat->element != NULL)
	{
		for(i = 0; i < mat->row_dim; i++)
			for(j = 0; j < mat->col_dim; j++)
				mpf_clear(*(mat->element + i * mat->col_dim + j));

		free(mat->element); // Fix! 2012-06-03 by T.Kouya
	}

	free(mat);
}
#endif // USE_GMP


#ifdef USE_GMP
mpf_ptr get_mpfmatrix_ij(MPFMatrix mat, long int row_index, long int col_index)
{
	return *(mat->element + row_index * mat->col_dim + col_index);
}

void set_mpfmatrix_ij(MPFMatrix mat, long int row_index, long int col_index, mpf_t val)
{
	mpf_set(*(mat->element + row_index * mat->col_dim + col_index), val);
}

void set_mpfmatrix_ij_d(MPFMatrix mat, long int row_index, long int col_index, double val)
{
	mpf_set_d((mpf_ptr)(mat->element + row_index * mat->col_dim + col_index), val);
}

void set_mpfmatrix_ij_str(MPFMatrix mat, long int row_index, long int col_index, const char *str, int base)
{
	mpf_set_str((mpf_ptr)(mat->element + row_index * mat->col_dim + col_index), str, base);
}

void set_mpfmatrix_ij_ui(MPFMatrix mat, long int row_index, long int col_index, unsigned long val)
{
	mpf_set_d((mpf_ptr)(mat->element + row_index * mat->col_dim + col_index), val);
}

/* get precision of MPFMatrix */
unsigned long int prec_mpfmatrix(MPFMatrix mat)
{
	return mat->prec;
}

/* search minimam precision in MPFMatrix */
unsigned long int minprec_mpfmatrix(MPFMatrix mat)
{
	unsigned long int prec, tmp;
	long int i, j;

	prec = mpf_get_prec(get_mpfmatrix_ij(mat, 0, 0));
	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
		{
			tmp = mpf_get_prec(get_mpfmatrix_ij(mat, i, j));
			if(prec > tmp)
				prec = tmp;
		}
	}

	return prec;
}

/* search maximam precision in MPFMatrix */
unsigned long int maxprec_mpfmatrix(MPFMatrix mat)
{
	unsigned long int prec, tmp;
	long int i, j;

	prec = mpf_get_prec(get_mpfmatrix_ij(mat, 0, 0));
	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
		{
			tmp = mpf_get_prec(get_mpfmatrix_ij(mat, i, j));
			if(prec < tmp)
				prec = tmp;
		}
	}

	return prec;
}

void print_mpfmatrix(MPFMatrix mat)
{
	long int i, j;

	for(i = 0; i < mat->row_dim; i++)
	{
		printf("%5ld ", i);
		for(j = 0; j < mat->col_dim; j++)
		{
			mpf_out_str(stdout, 10, 0, get_mpfmatrix_ij(mat, i, j));
			printf(" ");
		}
		printf("\n");
	}
}
#endif

/*************************************************/
/* Vector Calculations for MPFVector             */
/*
void add_mpfvector(MPFVector c, MPFVector a, MPFVector b)
void add2_mpfvector(MPFVector c, MPFVector a)
void sub_mpfvector(MPFVector c, MPFVector a, MPFVector b)
void sub2_mpfvector(MPFVector c, MPFVector a)
void cmul_mpfvector(MPFVector c, mpf_t val, MPFVector a)
void cmul2_mpfvector(MPFVector c, mpf_t val)
void add_cmul_mpfvector(MPFVector c, MPFVector a, mpf_t val, MPFVector b)
void ip_mpfvector(mpf_t ret, MPFVector a, MPFVector b)
void norm1_mpfvector(mpf_t ret, MPFVector a)
void norm2_mpfvector(mpf_t ret, MPFVector a)
void normi_mpfvector(mpf_t ret, MPFVector a)
void subst_mpfvector(MPFVector c, MPFVector a)
*/
/*************************************************/
#ifdef USE_GMP
/* c = a + b */
void add_mpfvector(MPFVector c, MPFVector a, MPFVector b)
{
	long int i;
	mpf_t tmp;
	unsigned long int prec;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_mpfvector\n");
		return;
	}

	prec = prec_mpfvector(c);

	mpf_init2(tmp, prec);

	for(i = 0; i < c->dim; i++)
	{
		mpf_add(tmp, get_mpfvector_i(a, i), get_mpfvector_i(b, i));
		set_mpfvector_i(c, i, tmp);
	}

	mpf_clear(tmp);

}

/* c += a */
void add2_mpfvector(MPFVector c, MPFVector a)
{
	long int i;
	mpf_t tmp;
	unsigned long int prec;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: add2_mpfvector\n");
		return;
	}

	prec = prec_mpfvector(c);

	mpf_init2(tmp, prec);

	for(i = 0; i < c->dim; i++)
	{
		mpf_add(tmp, get_mpfvector_i(c, i), get_mpfvector_i(a, i));
		set_mpfvector_i(c, i, tmp);
	}

	mpf_clear(tmp);
}

/* c = a - b */
void sub_mpfvector(MPFVector c, MPFVector a, MPFVector b)
{
	long int i;
	mpf_t tmp, tmpa, tmpb;
	unsigned long int prec;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: sub_mpfvector\n");
		return;
	}

	prec = prec_mpfvector(c);

	mpf_init2(tmp, prec);

	for(i = 0; i < c->dim; i++)
	{
		mpf_sub(tmp, get_mpfvector_i(a, i), get_mpfvector_i(b, i));
		set_mpfvector_i(c, i, tmp);
	}

	mpf_clear(tmp);
}

/* c -= a */
void sub2_mpfvector(MPFVector c, MPFVector a)
{
	long int i;
	mpf_t tmp;
	unsigned long int prec;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: sub2_mpfvector\n");
		return;
	}

	prec = prec_mpfvector(c);

	mpf_init2(tmp, prec);

	for(i = 0; i < c->dim; i++)
	{
		mpf_sub(tmp, get_mpfvector_i(c, i), get_mpfvector_i(a, i));
		set_mpfvector_i(c, i, tmp);
	}

	mpf_clear(tmp);

}

/* c = val * a */
void cmul_mpfvector(MPFVector c, mpf_t val, MPFVector a)
{
	long int i;
	mpf_t tmp, tmpa;
	unsigned long int prec;

	if(c->dim != a->dim)
	{
		fprintf(stderr, "ERROR: cmul_mpfvector\n");
		return;
	}

	prec = prec_mpfvector(c);

	mpf_init2(tmp, prec);

	for(i = 0; i < c->dim; i++)
	{
		mpf_mul(tmp, val, get_mpfvector_i(a, i));
		set_mpfvector_i(c, i, tmp);
	}

	mpf_clear(tmp);
}

/* c *= val */
void cmul2_mpfvector(MPFVector c, mpf_t val)
{
	long int i;
	mpf_t tmp;
	unsigned long int prec;

	prec = prec_mpfvector(c);

	mpf_init2(tmp, prec);

	for(i = 0; i < c->dim; i++)
	{
		mpf_mul(tmp, val, get_mpfvector_i(c, i));
		set_mpfvector_i(c, i, tmp);
	}

	mpf_clear(tmp);

}

/* c = a + val * b */
void add_cmul_mpfvector(MPFVector c, MPFVector a, mpf_t val, MPFVector b)
{
	long int i;
	mpf_t tmp, tmp2;
	unsigned long int prec;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: add_mpfvector\n");
		return;
	}

	prec = prec_mpfvector(c);

	mpf_init2(tmp, prec);
	mpf_init2(tmp2, prec);

	for(i = 0; i < c->dim; i++)
	{
		mpf_mul(tmp2, val, get_mpfvector_i(b, i));
		mpf_add(tmp, get_mpfvector_i(a, i), tmp2);
		set_mpfvector_i(c, i, tmp);
	}

	mpf_clear(tmp);
	mpf_clear(tmp2);

}

/* ret = (a, b) */
void ip_mpfvector(mpf_t ret, MPFVector a, MPFVector b)
{
	long int i;
	mpf_t tmp;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: ip_mpfvector\n");
		return;
	}

	mpf_init2(tmp, mpf_get_prec(ret));

	mpf_set_ui(ret, 0UL); /* ret := 0 */
	for(i = 0; i < a->dim; i++)
	{
#ifndef USE_MPFR
		mpf_mul(tmp, get_mpfvector_i(a, i), get_mpfvector_i(b, i));
		mpf_add(ret, ret, tmp);
#else
		mpfr_fma(ret, get_mpfvector_i(a, i), get_mpfvector_i(b, i), ret, bnc_default_rounding_mode);
#endif
	}

	mpf_clear(tmp);
}

/* ret = ||a||_1 */
void norm1_mpfvector(mpf_t ret, MPFVector a)
{

	long int i;
	mpf_t tmp;

	mpf_init2(tmp, mpf_get_prec(ret));

	mpf_set_ui(ret, 0UL); /* ret := 0 */
	for(i = 0; i < a->dim; i++)
	{
		mpf_abs(tmp, get_mpfvector_i(a, i));
		mpf_add(ret, ret, tmp);
	}

	mpf_clear(tmp);
}

/* ret := ||a||_2 */
void norm2_mpfvector(mpf_t ret, MPFVector a)
{
	long int i;
	mpf_t tmp;

	mpf_init2(tmp, mpf_get_prec(ret));

	mpf_set_ui(ret, 0UL); /* ret := 0 */
	for(i = 0; i < a->dim; i++)
	{
#ifndef USE_MPFR
		mpf_mul(tmp, get_mpfvector_i(a, i), get_mpfvector_i(a, i));
		mpf_add(ret, ret, tmp);
#else
		mpfr_fma(ret, get_mpfvector_i(a, i), get_mpfvector_i(a, i), ret, bnc_default_rounding_mode);
#endif
	}
	mpf_sqrt(ret, ret);

	mpf_clear(tmp);
}

/* ||a||_infty */
void normi_mpfvector(mpf_t ret, MPFVector a)
{
	long int i;
	mpf_t tmp;

	mpf_init2(tmp, mpf_get_prec(ret));

	mpf_abs(ret, get_mpfvector_i(a, 0));
	for(i = 1; i < a->dim; i++)
	{
		mpf_abs(tmp, get_mpfvector_i(a, i));
		if(mpf_cmp(ret, tmp) < 0) /* ret < tmp */
			mpf_set(ret, tmp);
	}

	mpf_clear(tmp);
}

/* c := a */
void subst_mpfvector(MPFVector c, MPFVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
		set_mpfvector_i(c, i, get_mpfvector_i(a, i));
}

/* c := 0 */
void set0_mpfvector(MPFVector c)
{
	unsigned long int prec;
	long int i;
	mpf_t tmp;

	prec = prec_mpfvector(c);

	mpf_init2(tmp, prec);
	mpf_set_ui(tmp, 0UL);

	for(i = 0; i < c->dim; i++)
		set_mpfvector_i(c, i, tmp);

	mpf_clear(tmp);
}

/* append 2005.07/12 */
/*
	ret(index_start) = src(src_index_start)
	 ...
	ret(index_end  ) = src(src_index_end)
*/
void copy_mpfvector_ij(MPFVector ret, long int index_start, long int index_end, MPFVector src, long int src_index_start, long int src_index_end)
{
	long int i, itmp;

	if((src_index_end - src_index_start) != (index_end - index_start))
	{
		fprintf(stderr, "Invalid index!(copy_mpfvector_ij)\n");
		return;
	}

	for(i = 0; i <= (index_end - index_start); i++)
	{
		set_mpfvector_i(ret, index_start + i, get_mpfvector_i(src, src_index_start + i));
//		printf("%ld <----------------------------------> %ld\n", index_start + i, src_index_start + i);
	}
}

#endif


/*************************************************/
/* Matrix Caluculations for MPFMatrix            */
/*
void normf_mpfmatrix(mpf_t ret, MPFMatrix mat)
void norm1_mpfmatrix(mpf_t ret, MPFMatrix mat)
void normi_mpfmatrix(mpf_t ret, MPFMatrix mat)
void add_mpfmatrix(MPFMatrix c, MPFMatrix a, MPFMatrix b);
void sub_mpfmatrix(MPFMatrix c, MPFMatrix a, MPFMatrix b);
void mul_mpfmatrix(MPFMatrix c, MPFMatrix a, MPFMatrix b);
void mul_mpfmatrix_mpfvec(MPFVector v, MPFMatrix a, MPFVector vb)
void mul_mpfmatrixt_mpfvec(MPFVector v, MPFMatrix a, MPFVector vb)
void transpose_mpfmatrix(MPFMatrix c, MPFMatrix a);
void inv_mpfmatrix(MPFMatrix a);
void subst_mpfmatrux(MPFMatrix c, MPFMatrix a);
*/
/*************************************************/
#ifdef USE_GMP
/* Frobenius Norm of Matrix */
void normf_mpfmatrix(mpf_t ret, MPFMatrix mat)
{
	long int i, j;
	mpf_t tmp;

	mpf_init2(tmp, mat->prec);
	mpf_set_ui(ret, 0UL);
	for(i = 0; i < mat->row_dim; i++)
		for(j = 0; j < mat->col_dim; j++)
		{
			mpf_mul(tmp, get_mpfmatrix_ij(mat, i, j), get_mpfmatrix_ij(mat, i, j));
			mpf_add(ret, ret, tmp);
		}

	mpf_sqrt(ret, ret);

	mpf_clear(tmp);

	return;
}

/* Infinity Norm of Matrix */
void normi_mpfmatrix(mpf_t ret, MPFMatrix mat)
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
			mpf_abs(tmp, get_mpfmatrix_ij(mat, i, j));
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
void norm1_mpfmatrix(mpf_t ret, MPFMatrix mat)
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
			mpf_abs(tmp, get_mpfmatrix_ij(mat, i, j));
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
void add_mpfmatrix(MPFMatrix c, MPFMatrix a, MPFMatrix b)
{
	long int i, j, row_dim, col_dim;
	mpf_t tmp;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_mpfmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_mpfmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	mpf_init2(tmp, prec_mpfmatrix(c));

	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			mpf_add(tmp, get_mpfmatrix_ij(a, i, j), get_mpfmatrix_ij(b, i, j));
			set_mpfmatrix_ij(c, i, j, tmp);
		}
	}

	mpf_clear(tmp);
}

/* c := a - b */
void sub_mpfmatrix(MPFMatrix c, MPFMatrix a, MPFMatrix b)
{
	long int i, j, row_dim, col_dim;
	mpf_t tmp;

	/* check row_dim */
	if((a->row_dim != b->row_dim) || (b->row_dim != c->row_dim) || (c->row_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: add_mpfmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if((a->col_dim != b->col_dim) || (b->col_dim != c->col_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: add_mpfmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	mpf_init2(tmp, prec_mpfmatrix(c));
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			mpf_sub(tmp, get_mpfmatrix_ij(a, i, j), get_mpfmatrix_ij(b, i, j));
			set_mpfmatrix_ij(c, i, j, tmp);
		}
	}
	mpf_clear(tmp);
}

/* c := sc * a */
void cmul_mpfmatrix(MPFMatrix c, mpf_t sc, MPFMatrix a)
{
	long int i, j, row_dim, col_dim;
	mpf_t tmp;

	/* check row_dim */
	if(a->row_dim != c->row_dim)
	{
		fprintf(stderr, "ERROR: cmul_mpfmatrix\n");
		return;
	}
	row_dim = c->row_dim;

	/* check col_dim */
	if(a->col_dim != c->col_dim)
	{
		fprintf(stderr, "ERROR: cmul_mpfmatrix\n");
		return;
	}
	col_dim = c->col_dim;

	mpf_init2(tmp, prec_mpfmatrix(c));
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			mpf_mul(tmp, sc, get_mpfmatrix_ij(a, i, j));
			set_mpfmatrix_ij(c, i, j, tmp);
		}
	}
	mpf_clear(tmp);
}


/* c = a * b */
void mul_mpfmatrix(MPFMatrix c, MPFMatrix a, MPFMatrix b)
{
	long int i, j, k;
	mpf_t tmp, tmp1;

	/* dimension check */
	if((c->row_dim != a->row_dim) || (c->col_dim != b->row_dim) || (a->col_dim != b->row_dim))
	{
		fprintf(stderr, "ERROR: mul_mpfmatrix\n");
		return;
	}

	mpf_init2(tmp, prec_mpfmatrix(c));
	mpf_init2(tmp1, prec_mpfmatrix(c));
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
		{
			mpf_set_ui(tmp, 0UL);
			for(k = 0; k < a->col_dim; k++)
			{
#ifndef USE_MPFR
				mpf_mul(tmp1, get_mpfmatrix_ij(a, i, k), get_mpfmatrix_ij(b, k, j));
				mpf_add(tmp, tmp, tmp1);
#else
				mpfr_fma(tmp, get_mpfmatrix_ij(a, i, k), get_mpfmatrix_ij(b, k, j), tmp, bnc_default_rounding_mode);
#endif
			}
			set_mpfmatrix_ij(c, i, j, tmp);
		}
	}
	mpf_clear(tmp);
	mpf_clear(tmp1);
}

/* c = a^T */
void transpose_mpfmatrix(MPFMatrix c, MPFMatrix a)
{
	long int i, j;

	/* Check Dimensions */
	if((c->row_dim != a->col_dim) || (c->col_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: transpose_mpfmatrix\n");
		return;
	}
	
	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_mpfmatrix_ij(c, i, j, get_mpfmatrix_ij(a, j, i));
	}
}

/* c := a */
void subst_mpfmatrix(MPFMatrix c, MPFMatrix a)
{
	long int i, j;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_mpfmatrix\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
			set_mpfmatrix_ij(c, i, j, get_mpfmatrix_ij(a, i, j));
	}
}

/* c := 0 */
void set0_mpfmatrix(MPFMatrix c)
{
	long int i, j;
	mpf_t tmp;

	mpf_init2(tmp, c->prec);
	mpf_set_ui(tmp, 0UL);

	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_mpfmatrix_ij(c, i, j, tmp);
	}

	mpf_clear(tmp);
}

/* c := I */
void setI_mpfmatrix(MPFMatrix c)
{
	long int i, j;
	mpf_t tmp0, tmp1;

	mpf_init2(tmp0, c->prec);
	mpf_init2(tmp1, c->prec);
	mpf_set_ui(tmp0, 0UL);
	mpf_set_ui(tmp1, 1UL);

	for(i = 0; i < c->row_dim; i++)
	{
		for(j = 0; j < c->col_dim; j++)
			set_mpfmatrix_ij(c, i, j, tmp0);
		if(i < c->col_dim)
			set_mpfmatrix_ij(c, i, i, tmp1);
	}

	mpf_clear(tmp0);
	mpf_clear(tmp1);
}

/* v := a * vb */
void mul_mpfmatrix_mpfvec(MPFVector v, MPFMatrix a, MPFVector vb)
{
	long int i, j;
	mpf_t tmp, tmp1;

	/* Check Dimension */
	if((v->dim != a->row_dim) || (vb->dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: mul_mpfmatrix_dvec\n");
		return;
	}

	mpf_init2(tmp, prec_mpfvector(v));
	mpf_init2(tmp1, prec_mpfvector(v));
	for(i = 0; i < a->row_dim; i++)
	{
		mpf_set_ui(tmp, 0UL);
		for(j = 0; j < a->col_dim; j++)
		{
#ifndef USE_MPFR
			mpf_mul(tmp1, get_mpfmatrix_ij(a, i, j), get_mpfvector_i(vb, j));
			mpf_add(tmp, tmp, tmp1);
#else
			mpfr_fma(tmp, get_mpfmatrix_ij(a, i, j), get_mpfvector_i(vb, j), tmp, bnc_default_rounding_mode);
#endif
		}
		set_mpfvector_i(v, i, tmp);
	}
	mpf_clear(tmp);
	mpf_clear(tmp1);
}

/* v := a^T * vb */
void mul_mpfmatrixt_mpfvec(MPFVector v, MPFMatrix a, MPFVector vb)
{
	long int i, j;
	mpf_t tmp, tmp1;

	/* Check Dimension */
	if((v->dim != a->col_dim) || (vb->dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: mul_mpfmatrixt_mpfvec\n");
		return;
	}

	mpf_init2(tmp, prec_mpfvector(v));
	mpf_init2(tmp1, prec_mpfvector(v));
	for(i = 0; i < a->col_dim; i++)
	{
		mpf_set_ui(tmp, 0UL);
		for(j = 0; j < a->row_dim; j++)
		{
#ifndef USE_MPFR
			mpf_mul(tmp1, get_mpfmatrix_ij(a, j, i), get_mpfvector_i(vb, j));
			mpf_add(tmp, tmp, tmp1);
#else
			mpfr_fma(tmp, get_mpfmatrix_ij(a, j, i), get_mpfvector_i(vb, j), tmp, bnc_default_rounding_mode);
#endif
		}
		set_mpfvector_i(v, i, tmp);
	}
	mpf_clear(tmp);
	mpf_clear(tmp1);
}

/* a = a^(-1) */
/* square matrix only */
void inv_mpfmatrix(MPFMatrix a)
{
	long int i, j, k, dim;
	mpf_t tmp, aii;

	/* Check Dimensions */
	if(a->row_dim != a->col_dim)
	{
		fprintf(stderr, "ERROR: inv_mpfmatrix\n");
		return;
	}

	mpf_init2(tmp, prec_mpfmatrix(a));
	mpf_init2(aii, prec_mpfmatrix(a));
	dim = a->row_dim;

	for(i = 0; i < dim; i++)
	{
		if(mpf_cmp_ui(get_mpfmatrix_ij(a, i, i), 0UL) == 0) 
		{
			fprintf(stderr, "ERROR: inv_mpfmatrix: Pivot(%ld,%ld) is zero.\n", i, i);
			return;
		}

		mpf_ui_div(aii, 1UL, get_mpfmatrix_ij(a, i, i));
		set_mpfmatrix_ij(a, i, i, aii);

		for(j = 0; j < i; j++)
		{
			mpf_mul(tmp, get_mpfmatrix_ij(a, i, j), aii);
			set_mpfmatrix_ij(a, i, j, tmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			mpf_mul(tmp, get_mpfmatrix_ij(a, i, j), aii);
			set_mpfmatrix_ij(a, i, j, tmp);
		}

		for(j = 0; j < i; j++)
		{
			for(k = 0; k < i; k++)
			{
				mpf_mul(tmp, get_mpfmatrix_ij(a, j, i), get_mpfmatrix_ij(a, i, k));
				mpf_sub(tmp, get_mpfmatrix_ij(a, j, k), tmp);
				set_mpfmatrix_ij(a, j, k, tmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				mpf_mul(tmp, get_mpfmatrix_ij(a, j, i), get_mpfmatrix_ij(a, i, k));
				mpf_sub(tmp, get_mpfmatrix_ij(a, j, k), tmp);
				set_mpfmatrix_ij(a, j, k, tmp);
			}
		}

		for(j = i + 1; j < dim; j++)
		{
			for(k = 0; k < i; k++)
			{
				mpf_mul(tmp, get_mpfmatrix_ij(a, j, i), get_mpfmatrix_ij(a, i, k));
				mpf_sub(tmp, get_mpfmatrix_ij(a, j, k), tmp);
				set_mpfmatrix_ij(a, j, k, tmp);
			}
			for(k = i + 1; k < dim; k++)
			{
				mpf_mul(tmp, get_mpfmatrix_ij(a, j, i), get_mpfmatrix_ij(a, i, k));
				mpf_sub(tmp, get_mpfmatrix_ij(a, j, k), tmp);
				set_mpfmatrix_ij(a, j, k, tmp);
			}
		}

		for(j = 0; j < i; j++)
		{
			mpf_neg(tmp, aii); /* tmp := -aii */
			mpf_mul(tmp, tmp, get_mpfmatrix_ij(a, j, i));
			set_mpfmatrix_ij(a, j, i, tmp);
		}
		for(j = i + 1; j < dim; j++)
		{
			mpf_neg(tmp, aii); /* tmp := -aii */
			mpf_mul(tmp, tmp, get_mpfmatrix_ij(a, j, i));
			set_mpfmatrix_ij(a, j, i, tmp);
		}
	}

	mpf_clear(aii);
	mpf_clear(tmp);
}

// ----- linear_append.c ------

#ifdef USE_FLINEAR

/* c := (double)a */
void subst_fmatrix_dmat(FMatrix c, DMatrix a)
{
	long int i, j;
	float tmp;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_fmatrix_dmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_fmatrix_ij(c, i, j, (float)get_dmatrix_ij(a, i, j));
		}
	}
}

/* c := (double)a^T */
void subst_fmatrix_dmat_trans(FMatrix c, DMatrix a)
{
	long int i, j;
	float tmp;

	if((c->row_dim != a->col_dim) || (c->col_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: subst_fmatrix_dmat_trans\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			set_fmatrix_ij(c, j, i, (float)get_dmatrix_ij(a, i, j));
		}
	}
}

/* c := (double)a */
void subst_fvector_dvec(FVector c, DVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
		set_fvector_i(c, i, (float)get_dvector_i(a, i));
}

/* (double)c := (float)a */
void subst_dvector_fvec(DVector c, FVector a)
{
	long int i;

	for(i = 0; i < a->dim; i++)
		set_dvector_i(c, i, (double)get_fvector_i(a, i));
}
#endif // ifdef USE_FLINEAR



//#ifdef USE_GMP


/* get residual in mpf_t precision */
/* r := b - A * x */
void residual_mpfmat_mpfvec(MPFVector r, MPFVector b, MPFMatrix mat, MPFVector x)
{
	mul_mpfmatrix_mpfvec(r, mat, x);
	sub_mpfvector(r, b, r);
}

//#if USE_DLINEAR
/* get residual in double precision */
/* r := b - A * x */
void residual_dmat_dvec(DVector r, DVector b, DMatrix mat, DVector x)
{
	mul_dmatrix_dvec(r, mat, x);
	sub_dvector(r, b, r);
}

/* c := (mpf_t)a */
void subst_dmatrix_mpfmat(DMatrix c, MPFMatrix a)
{
	long int i, j;
	double tmp;

	if((c->row_dim != a->row_dim) || (c->col_dim != a->col_dim))
	{
		fprintf(stderr, "ERROR: subst_dmatrix_mpfmat\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			tmp = mpf_get_d(get_mpfmatrix_ij(a, i, j));
			set_dmatrix_ij(c, i, j, tmp);
		}
	}
}

/* c := (mpf_t)a^T */
void subst_dmatrix_mpfmat_trans(DMatrix c, MPFMatrix a)
{
	long int i, j;
	double tmp;

	if((c->row_dim != a->col_dim) || (c->col_dim != a->row_dim))
	{
		fprintf(stderr, "ERROR: subst_dmatrix_mpfmat_trans\n");
		return;
	}

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			tmp = mpf_get_d(get_mpfmatrix_ij(a, i, j));
			set_dmatrix_ij(c, j, i, tmp);
		}
	}
}

/* c := (double)a */
void subst_dvector_mpfvec(DVector c, MPFVector a)
{
	long int i;
	double tmp;

	for(i = 0; i < a->dim; i++)
	{
		tmp = mpf_get_d(get_mpfvector_i(a, i));
		set_dvector_i(c, i, tmp);
	}
}

/* (mpf_t)c := (double)a */
void subst_mpfvector_dvec(MPFVector c, DVector a)
{
	long int i;
	mpf_t tmp;

	mpf_init2(tmp, c->prec);

	for(i = 0; i < a->dim; i++)
	{
		mpf_set_d(tmp, get_dvector_i(a, i));
		set_mpfvector_i(c, i, tmp);
	}

	mpf_clear(tmp);
}
//endif // if USE_DLINEAR

/* vec := -vec */
void neg_mpfvector(MPFVector vec)
{
	long int i;
	mpf_t tmp;

	mpf_init2(tmp, vec->prec);

	for(i = 0; i < vec->dim; i++)
	{
		mpf_neg(tmp, get_mpfvector_i(vec, i));
		set_mpfvector_i(vec, i, tmp);
	}

	mpf_clear(tmp);
}

/* Normwise relative error of vector */
void relerr_mpfvector(mpf_t relerr, MPFVector approx_vec, MPFVector true_vec, int norm_type)
{
	unsigned long prec;
	mpf_t norm_true_vec, norm_diff_vec;
	MPFVector diff_vec;

	prec = approx_vec->prec;

	diff_vec = init2_mpfvector(approx_vec->dim, prec);
	mpf_init2(norm_true_vec, prec);
	mpf_init2(norm_diff_vec, prec);

	// diff_vec := approx_vec - true_vec
	sub_mpfvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_mpfvector(norm_diff_vec, diff_vec);
			normi_mpfvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_mpfvector(norm_diff_vec, diff_vec);
			norm1_mpfvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_mpfvector(norm_diff_vec, diff_vec);
			norm2_mpfvector(norm_true_vec, true_vec);
			break;
	}

	if(mpf_cmp_ui(norm_true_vec, 0UL) != 0)
		mpf_div(relerr, norm_diff_vec, norm_true_vec);

	free_mpfvector(diff_vec);
	mpf_clear(norm_diff_vec);
	mpf_clear(norm_true_vec);

	return;
}

/* Elementwise relative errors of vector */
void relerr_element_mpfvector(mpf_t max_relerr, mpf_t min_relerr, mpf_t norm_relerr, MPFVector approx_vec, MPFVector true_vec, int norm_type)
{
	unsigned long prec;
	long int i;
	mpf_t abs_true_vec, abs_diff_vec, norm_diff_vec, norm_true_vec;
	MPFVector diff_vec;

	prec = approx_vec->prec;

	diff_vec = init2_mpfvector(approx_vec->dim, prec);
	mpf_init2(abs_true_vec, prec);
	mpf_init2(abs_diff_vec, prec);
	mpf_init2(norm_diff_vec, prec);
	mpf_init2(norm_true_vec, prec);

	// diff_vec := approx_vec - true_vec
	sub_mpfvector(diff_vec, approx_vec, true_vec);

	switch(norm_type)
	{
		// inifinity norm 
		case 0:
			normi_mpfvector(norm_diff_vec, diff_vec);
			normi_mpfvector(norm_true_vec, true_vec);
			break;

		// 1-norm
		case 1:
			norm1_mpfvector(norm_diff_vec, diff_vec);
			norm1_mpfvector(norm_true_vec, true_vec);
			break;

		// euclid norm: default
		case 2:
		default:
			norm2_mpfvector(norm_diff_vec, diff_vec);
			norm2_mpfvector(norm_true_vec, true_vec);
			break;
	}

	mpf_set(norm_relerr, norm_diff_vec);
	if(mpf_cmp_ui(norm_true_vec, 0UL) != 0)
		mpf_div(norm_relerr, norm_diff_vec, norm_true_vec);

	// relative errors of each elements
	mpf_set_ui(max_relerr, 0UL);
	normi_mpfvector(min_relerr, diff_vec);
	for(i = 0; i < approx_vec->dim; i++)
	{
		mpf_abs(abs_diff_vec, get_mpfvector_i(diff_vec, i));
		mpf_abs(abs_true_vec, get_mpfvector_i(true_vec, i));
		if(mpf_cmp_ui(abs_true_vec, 0UL) != 0)
			mpf_div(abs_diff_vec, abs_diff_vec, abs_true_vec);
		
		if(mpf_cmp(max_relerr, abs_diff_vec) < 0)
			mpf_set(max_relerr, abs_diff_vec);
		if(mpf_cmp(min_relerr, abs_diff_vec) > 0)
			mpf_set(min_relerr, abs_diff_vec);
	}

	free_mpfvector(diff_vec);// Fix! 2012-06-03 by T.Kouya
	mpf_clear(abs_true_vec);
	mpf_clear(abs_diff_vec);
	mpf_clear(norm_true_vec);
	mpf_clear(norm_diff_vec);

	return;
}

// Exchange a(row_index0, col_start:col_end) to a(row_index1, col_start:col_end)
void row_swap_mpfmatrix(MPFMatrix mat, long int row_index0, long int row_index1, long int col_start, long int col_end)
{
	long int i, j, true_end;
	mpf_t tmp;

	mpf_init2(tmp, mat->prec);

	true_end = (col_end > mat->col_dim) ? mat->col_dim : col_end;

	for(i = col_start; i < true_end; i++)
	{
		mpf_set(tmp, get_mpfmatrix_ij(mat, row_index0, i));
		set_mpfmatrix_ij(mat, row_index0, i, get_mpfmatrix_ij(mat, row_index1, i));
		set_mpfmatrix_ij(mat, row_index1, i, tmp);
	}

	mpf_clear(tmp);
}

/************************************************************/
/*                                                          */
/*           LU Decomposition for Square Dense Matrix       */
/*                                 (Multi-Precision)        */
/*                                 (Partial Pivoting)       */
/*                                                          */
/*                 ver. 0.0 2000.02.28 (Thu) Tomonori Kouya */
/*                 ver. 0.1 2000.07.05 (Wed) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
int MPFLUdecompPM(MPFMatrix a, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       MPFMatrix a: Matrix (given by user)                */
/*       long int ch: Row order                             */
/*                                                          */
/* returns                                                  */
/*       a: LU decomposed matrix                            */
/*      ch: Row order                                       */
/*                                                          */
/************************************************************/
{
	long int i, j, k, imax, itmp, dim;
	mpf_t tmp, axii;

	dim = a->col_dim;

	mpf_init2(tmp, a->prec);
	mpf_init2(axii, a->prec);
	for(i = 0; i < dim; i++)
		ch[i] = i;

	for(i = 0; i < dim; i++)
	{
		mpf_abs(axii, get_mpfmatrix_ij(a, i, i));
		imax = i;
		for(j = (i + 1); j < dim; j++)
		{
			mpf_abs(tmp, get_mpfmatrix_ij(a, j, i));
			if(mpf_cmp(tmp, axii) > 0)
			{
				imax = j;
				mpf_set(axii, tmp);
			}
		}

		if(mpf_cmp_ui(axii, 0UL) == 0)
		{
			mpf_clear(tmp);
			mpf_clear(axii);
			fprintf(stderr, "%ld : Error! MPFLUdecompP!\n", i);
			return -1;
		}

		if(imax != i)
		{
			itmp = ch[imax];
			ch[imax] = ch[i];
			ch[i] = itmp;

			row_swap_mpfmatrix(a, i, imax, 0, a->col_dim);
		}

		for(j = (i + 1); j < dim; j++)
		{
			mpf_div(tmp, get_mpfmatrix_ij(a, j, i), get_mpfmatrix_ij(a, i, i));
			set_mpfmatrix_ij(a, j, i, tmp);
		}
		for(j = (i + 1); j < dim; j++)
		{
			for(k = (i + 1); k < dim; k++)
			{
				mpf_mul(tmp, get_mpfmatrix_ij(a, j, i), get_mpfmatrix_ij(a, i, k));
				mpf_sub(tmp, get_mpfmatrix_ij(a, j, k), tmp);
				set_mpfmatrix_ij(a, j, k, tmp);
			}
		}
	}

	mpf_clear(tmp);
	mpf_clear(axii);
	return 0;
}



/************************************************************/
/*                                                          */
/*                 Solver for Linear System                 */
/*                      (LU Decomposed Square Dense Matrix) */
/*                                 (Multi-Precision)        */
/*                                 (Partial Pivoting)       */
/*                                                          */
/*                 ver. 0.0 2000.02.28 (Mon) Tomonori Kouya */
/*                 ver. 0.1 2000.07.05 (Wed) Tomonori Kouya */
/*                 ver. 0.2 2006.01.13 (Fri) Tomonori Kouya */
/*                 ver. 0.3 2012-07-18 (Web) Tomonori Kouya */
/*                                                          */
/************************************************************/
int SolveMPFLSPM(MPFVector answer, MPFMatrix lu, MPFVector b, long int ch[])
/************************************************************/
/*                                                          */
/* entries                                                  */
/*       MPFMatrix lu[]: LU decomposed Matrix(given by user)*/
/*       MPFVector b[]: constant vector (given by user)     */
/*       MPFVector answer[]: Solution for linear system     */
/*       long int ch: Row order (given by user)             */
/*                                                          */
/* returns                                                  */
/*       answer[]: Solution for linear system               */
/*                                                          */
/************************************************************/
{
	long int i, j, dim;
	mpf_t tmp;

	mpf_init2(tmp, answer->prec);
	dim = answer->dim;

	for(i = 0; i < dim; i++)
		set_mpfvector_i(answer, i, get_mpfvector_i(b, ch[i]));

	
/* Forward */
	for(i = 0; i < dim; i++)
	{
		if(mpf_cmp_ui(get_mpfmatrix_ij(lu, i, i), 0UL) == 0)
		{
			mpf_clear(tmp);
			fprintf(stderr, "Unable to solve the linear system!(SolveMPFLSPM, %ld)\n", i);
			return -1;
		}

		for(j = (i + 1); j < dim; j++)
		{
			mpf_mul(tmp, get_mpfmatrix_ij(lu, j, i), get_mpfvector_i(answer, i));
			mpf_sub(tmp, get_mpfvector_i(answer, j), tmp);
			set_mpfvector_i(answer, j, tmp);
		}
	}

/* Backward */
	for(i = (dim - 1); i >= 0; i--)
	{
		for(j = (i + 1); j < dim; j++)
		{
			
			mpf_mul(tmp, get_mpfmatrix_ij(lu, i, j), get_mpfvector_i(answer, j));
			mpf_sub(tmp, get_mpfvector_i(answer, i), tmp);
			set_mpfvector_i(answer, i, tmp);
		}
		mpf_div(tmp, get_mpfvector_i(answer, i), get_mpfmatrix_ij(lu, i, i));
		set_mpfvector_i(answer, i, tmp);
	}

	mpf_clear(tmp); // Fix! 2012-07-18 by T.Kouya

	return 0;
}


// Appended in 2024-05-09 T.Kouya
// from lanczos.c

/* c = a - sc * b */
void subcmul_mpfvector(MPFVector c, MPFVector a, mpf_t sc, MPFVector b)
{
	long int i;
	mpf_t tmp, tmp1;

	if((a->dim != b->dim) || (c->dim != a->dim) || (c->dim != b->dim))
	{
		fprintf(stderr, "ERROR: subcmul_mpfvector\n");
		return;
	}

	//cmul_mpfvector(c, sc, b);
	//sub_mpfvector(c, a, c);
	mpf_init2(tmp, c->prec);
	mpf_init2(tmp1, c->prec);
	for(i = 0; i < c->dim; i++)
	{
		mpf_set(tmp1, get_mpfvector_i(a, i));
		mpf_mul(tmp, sc, get_mpfvector_i(b, i));
		mpf_sub(tmp, tmp1, tmp);
		set_mpfvector_i(c, i, tmp);
	}
	mpf_clear(tmp);
	mpf_clear(tmp1);
}

/* mat := (vec[0] vec[1] ... vec[n]) */
void subst_mpfmatrix_mpfvec(MPFMatrix mat, MPFVector vec[])
{
	long int i, j;

	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
			set_mpfmatrix_ij(mat, i, j, get_mpfvector_i(vec[j], i));
	}
}

/* (mpf_t)c := (double)a */
void subst_mpfmatrix_dmat(MPFMatrix c, DMatrix a)
{
	long int i, j;
	mpf_t tmp;

	mpf_init2(tmp, c->prec);

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			mpf_set_d(tmp, get_dmatrix_ij(a, i, j));
			set_mpfmatrix_ij(c, i, j, tmp);
		}
	}

	mpf_clear(tmp);
}

#if 0

// 2026-02-03(Tue) T.Kouya
// absmax_mpfvector
void absmax_mpfvector(mpf_t ret, long int *max_index, MPFVector vec)
{
    long int i, max_i, dim = vec->dim;
    mpf_t abs_val;

	mpf_init2(abs_val, mpf_get_prec(ret));

	max_i = 0;
    //ret = 0.0;
	mpf_set_ui(abs_val, 0UL);

	for(i = 0; i < dim; i++)
    {
        mpf_abs(abs_val, get_mpfvector_i(vec, i));
        if(mpf_cmp(ret, abs_val) < 0)
        {
            //ret = abs_val;
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
// absmax_row_mpfmatrix
void absmax_row_mpfmatrix(mpf_t ret, long int *max_j, long int row_index, MPFMatrix mat)
{
    long int j, max_index = 0;
    mpf_t abs_aij;

	mpf_init2(abs_aij, mpf_get_prec(ret));

	mpf_abs(ret, get_mpfmatrix_ij(mat, row_index, 0));

	for(j = 1; j < mat->col_dim; j++)
	{
		mpf_abs(abs_aij, get_mpfmatrix_ij(mat, row_index, j));
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
// absmax_col_dmatrix
void absmax_col_mpfmatrix(mpf_t ret, long int *max_i, long int col_index, MPFMatrix mat)
{
    long int i, max_index = 0;
    mpf_t abs_aij;

	mpf_init2(abs_aij, mpf_get_prec(ret));

	mpf_abs(ret, get_mpfmatrix_ij(mat, col_index, 0));

	for(i = 1; i < mat->row_dim; i++)
	{
		mpf_abs(abs_aij, get_mpfmatrix_ij(mat, i, col_index));
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
#endif // 0

// 2026-02-03(Tue) T.Kouya
// absmax_mpfmatrix
void absmax_mpfmatrix(mpf_t ret, long int *max_i, long int *max_j, MPFMatrix mat)
{
    long int i, j, max_row_index = 0, max_col_index = 0;
    mpf_t abs_aij;

	mpf_init2(abs_aij, mpf_get_prec(ret));

	mpf_abs(ret, get_mpfmatrix_ij(mat, 0, 0));
	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
		{
			mpf_abs(abs_aij, get_mpfmatrix_ij(mat, i, j));
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

#if 0 //USE_DDLINEAR
/* (mpf_t)c := (dd)a */
void subst_mpfmatrix_ddmat(MPFMatrix c, DDMatrix a)
{
	long int i, j;
	mpf_t tmp;

	mpf_init2(tmp, c->prec);

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			mpf_set_dd(tmp, get_ddmatrix_ij(a, i, j));
			set_mpfmatrix_ij(c, i, j, tmp);
		}
	}

	mpf_clear(tmp);
}

/* (mpf_t)c := (td)a */
void subst_mpfmatrix_tdmat(MPFMatrix c, TDMatrix a)
{
	long int i, j;
	mpf_t tmp;

	mpf_init2(tmp, c->prec);

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			mpf_set_td(tmp, get_tdmatrix_ij(a, i, j));
			set_mpfmatrix_ij(c, i, j, tmp);
		}
	}

	mpf_clear(tmp);
}

/* (mpf_t)c := (qd)a */
void subst_mpfmatrix_qdmat(MPFMatrix c, QDMatrix a)
{
	long int i, j;
	mpf_t tmp;

	mpf_init2(tmp, c->prec);

	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			mpf_set_qd(tmp, get_qdmatrix_ij(a, i, j));
			set_mpfmatrix_ij(c, i, j, tmp);
		}
	}

	mpf_clear(tmp);
}
#endif // 0

#endif // USE_GMP