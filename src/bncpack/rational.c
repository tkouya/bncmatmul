/********************************************************************************/
/* rational.c: Rational Vector and Matrix                                       */
/* Copyright (c) 2004-2011 Tomonori Kouya                                       */
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

#include "bnc.h"

#ifdef USE_GMP

void mpq_get_f(mpf_t ret, mpq_t src)
{
#ifdef USE_MPFR
	mpfr_set_q(ret, src, bnc_default_rounding_mode);
#else
	char *cnum, *cden;
	mpf_t fnum, fden;

	mpf_init2(fnum, mpf_get_prec(ret));
	mpf_init2(fden, mpf_get_prec(ret));

	cnum = (char *)malloc(mpz_sizeinbase(mpq_numref(src), 10) + 2);
	cden = (char *)malloc(mpz_sizeinbase(mpq_denref(src), 10) + 2);
	mpz_get_str(cnum, 10, mpq_numref(src));
	mpz_get_str(cden, 10, mpq_denref(src));

	mpf_set_str(fnum, cnum, 10);
	mpf_set_str(fden, cden, 10);
	mpf_div(ret, fnum, fden);

	mpf_clear(fnum);
	mpf_clear(fden);
	free(cnum);
	free(cden);
#endif
}

MPQVector init_mpqvector(long int dimension)
{
	MPQVector ret = NULL;
	long int i;

	if(dimension <= 0)
	{
		fprintf(stderr, "ERROR: init_mpqvector\n");
		return ret;
	}

	ret = (MPQVector)malloc(sizeof(mpqvector));
	if(ret == NULL)
		return ret;

	ret->element = (mpq_t *)calloc(sizeof(mpq_t), dimension);
	if(ret->element == NULL)
		return NULL;

	/* All 0 */
	for(i = 0; i < dimension; i++)
	{
		mpq_init((mpq_ptr)(ret->element + i));
		mpq_set_ui((mpq_ptr)(ret->element + i), 0UL, 0UL);
		if((ret->element + i) == NULL)
			return NULL;
	}

	ret->dim = dimension;

	return ret;
}

void free_mpqvector(MPQVector vec)
{
	long int i;

	if(vec == NULL)
		return;

	if(vec->element != NULL)
	{
		for(i = 0; i < vec->dim; i++)
			mpq_clear((mpq_ptr)(vec->element + i));
	}

	free(vec);

}

mpq_ptr get_mpqvector_i(MPQVector vec, long int index)
{
	return *(vec->element + index);
}

void set_mpqvector_i(MPQVector vec, long int index, mpq_t val)
{
	mpq_set((mpq_ptr)(vec->element + index), val);
}

void set_mpqvector_i_z(MPQVector vec, long int index, mpz_t val)
{
	mpq_set_z(*(vec->element + index), val);
}

void set_mpqvector_i_str_b(MPQVector vec, long int index, const char *str, int base)
{
	mpq_set_str(*(vec->element + index), str, base);
}

void set_mpqvector_i_str(MPQVector vec, long int index, const char *str)
{
	mpq_set_str(*(vec->element + index), str, 10);
}

void set_mpqvector_i_ui(MPQVector vec, long int index, unsigned long val_num, unsigned long val_den)
{
	mpq_set_ui(*(vec->element + index), val_num, val_den);
}

void set_mpqvector_i_si(MPQVector vec, long int index, long val_num, unsigned long val_den)
{
	mpq_set_si(*(vec->element + index), val_num, val_den);
}

void print_mpqvector(MPQVector vec)
{
	long int i;

	for(i = 0; i < vec->dim; i++)
	{
		printf("%5ld ", i);
		mpq_out_str(stdout, 10, get_mpqvector_i(vec, i));
		printf("\n");
	}
}

/* ret = (a, b) */
void ip_mpqvector(mpq_t ret, MPQVector a, MPQVector b)
{
	long int i;
	mpq_t tmp;

	if(a->dim != b->dim)
	{
		fprintf(stderr, "ERROR: ip_mpqvector\n");
		return;
	}

	mpq_init(tmp);

	mpq_set_str(ret, "0", 10); /* ret := 0 */
	for(i = 0; i < a->dim; i++)
	{
		mpq_mul(tmp, get_mpqvector_i(a, i), get_mpqvector_i(b, i));
		mpq_add(ret, ret, tmp);
	}

	mpq_clear(tmp);
}

#endif
