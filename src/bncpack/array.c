/********************************************************************************/
/* array.c:                                                                     */
/* Copyright (C) 2003-2011 Tomonori Kouya                                       */
/*                                                                              */
/* Ver. 0.1 2008-06-04: Fix init_c[fd]array                                     */
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
#include "bnc.h"

/*******************************/
/* For IEEE 754 float          */
/* init_farray                 */
/* free_farray                 */
/* get_farray_i                */
/* set_farray_i                */
/*******************************/
FArray init_farray(long int array_size)
{
	FArray st;
	long int i;

	st = (FArray)malloc(sizeof(FArray));
	if(st == NULL)
		return NULL;

	st->array = (float *)calloc(sizeof(float), array_size);
	if(st->array == NULL)
	{
		free(st);
		return NULL;
	}

	st->size = array_size;

	for(i = 0; i < array_size; i++)
		*(st->array + i) = 0.0;

	return st;
}

void free_farray(FArray st)
{
	if(st == NULL)
		return;

	if(st->array != NULL)
		free(st->array);

	free(st);
}

float get_farray_i(FArray array, long int index)
{
	return *(array->array + index);
}

void set_farray_i(FArray array, long int index, float val)
{
	*(array->array + index) = val;
}

void subst_farray(FArray c, FArray a)
{
	long int i, size;

	size = a->size;
	if(c->size < a->size)
	{
		fprintf(stderr, "Warning: Not enough array size!\n");
		size = c->size;
	}

	for(i = 0; i < size; i++)
		set_farray_i(c, i, get_farray_i(a, i));
}

void print_farray(FArray array)
{
	long int i;

	for(i = 0; i < array->size; i++)
		printf("%5ld: %15.7e\n", i, get_farray_i(array, i));
}

/*******************************/
/* For IEEE 754 complex float  */
/* init_cfarray                */
/* free_cfarray                */
/* get_cfarray_i               */
/* set_cfarray_i               */
/*******************************/
CFArray init_cfarray(long int array_size)
{
	CFArray st;
	long int i;

	st = (CFArray)malloc(sizeof(cfarray)); // Fix!!: 2008-06-04
	if(st == NULL)
		return NULL;

	st->array = (fcmplx *)calloc(sizeof(fcmplx), array_size);
	if(st->array == NULL)
	{
		free(st);
		return NULL;
	}

	st->size = array_size;

	for(i = 0; i < array_size; i++)
		set0_fcmplx(st->array + i);

	return st;
}

void free_cfarray(CFArray st)
{
	if(st == NULL)
		return;

	if(st->array != NULL)
		free(st->array);

	free(st);
}

FCmplx get_cfarray_i(CFArray array, long int index)
{
	return (array->array + index);
}

void set_cfarray_i(CFArray array, long int index, FCmplx val)
{
	subst_fcmplx((array->array + index), val);
}

void subst_cfarray(CFArray c, CFArray a)
{
	long int i, size;

	size = a->size;
	if(c->size < a->size)
	{
		fprintf(stderr, "Warning: Not enough array size!\n");
		size = c->size;
	}

	for(i = 0; i < size; i++)
		set_cfarray_i(c, i, get_cfarray_i(a, i));
}

void print_cfarray(CFArray array)
{
	long int i;

	for(i = 0; i < array->size; i++)
	{
		printf("%5ld ", i);
		print_fcmplx(get_cfarray_i(array, i));
	}
}

/*******************************/
/* For IEEE 754 double         */
/* init_darray                 */
/* free_darray                 */
/* push_darray                 */
/* pop_darray                  */
/*******************************/
DArray init_darray(long int array_size)
{
	DArray st;
	long int i;

	st = (DArray)malloc(sizeof(DArray));
	if(st == NULL)
		return NULL;

	st->array = (double *)calloc(sizeof(double), array_size);
	if(st->array == NULL)
	{
		free(st);
		return NULL;
	}

	st->size = array_size;

	for(i = 0; i < array_size; i++)
		*(st->array + i) = 0.0;

	return st;
}

void free_darray(DArray st)
{
	if(st == NULL)
		return;

	if(st->array != NULL)
		free(st->array);

	free(st);
}

double get_darray_i(DArray array, long int index)
{
	return *(array->array + index);
}


void set_darray_i(DArray array, long int index, double val)
{
	*(array->array + index) = val;
}

void subst_darray(DArray c, DArray a)
{
	long int i, size;

	size = a->size;
	if(c->size < a->size)
	{
		fprintf(stderr, "Warning: Not enough array size!\n");
		size = c->size;
	}

	for(i = 0; i < size; i++)
		set_darray_i(c, i, get_darray_i(a, i));
}

void print_darray(DArray array)
{
	long int i;

	for(i = 0; i < array->size; i++)
	{
		printf("%5ld %25.17e\n", i, get_darray_i(array, i));
	}
}

/*******************************/
/* For IEEE 754 double         */
/* init_darray                 */
/* free_darray                 */
/* push_darray                 */
/* pop_darray                  */
/*******************************/
CDArray init_cdarray(long int array_size)
{
	CDArray st;
	long int i;

	st = (CDArray)malloc(sizeof(cdarray)); // Fix!!: 2008-06-04
	if(st == NULL)
		return NULL;

	st->array = (dcmplx *)calloc(sizeof(dcmplx), array_size);
	if(st->array == NULL)
	{
		free(st);
		return NULL;
	}

	st->size = array_size;

	for(i = 0; i < array_size; i++)
		set0_dcmplx((st->array + i));

	return st;
}

void free_cdarray(CDArray st)
{
	if(st == NULL)
		return;

	if(st->array != NULL)
		free(st->array);

	free(st);
}

DCmplx get_cdarray_i(CDArray array, long int index)
{
	return (array->array + index);
}


void set_cdarray_i(CDArray array, long int index, DCmplx val)
{
	subst_dcmplx((array->array + index), val);
}

void subst_cdarray(CDArray c, CDArray a)
{
	long int i, size;

	size = a->size;
	if(c->size < a->size)
	{
		fprintf(stderr, "Warning: Not enough array size!\n");
		size = c->size;
	}

	for(i = 0; i < size; i++)
		set_cdarray_i(c, i, get_cdarray_i(a, i));
}

void print_cdarray(CDArray array)
{
	long int i;

	for(i = 0; i < array->size; i++)
	{
		printf("%5ld ", i);
		print_dcmplx(get_cdarray_i(array, i));
	}
}

/*******************************/
/* For GMP                     */
/* init_mpfarray               */
/* free_mpfarray               */
/* get_mpfarray_i              */
/* set_mpfarray_i              */
/*******************************/
#ifdef USE_GMP
MPFArray init_mpfarray(long int array_size)
{
	MPFArray st;
	long int i;

	st = (MPFArray)malloc(sizeof(mpfarray));
	if(st == NULL)
		return NULL;

	st->array = (mpf_t *)calloc(sizeof(mpf_t), array_size);
	if(st->array == NULL)
	{
		free(st);
		return NULL;
	}

	st->size = array_size;

	for(i = 0; i < array_size; i++)
		mpf_init((mpf_ptr)st->array + i);

	/* set precision */
	st -> prec = get_bnc_default_prec();

	return st;
}

MPFArray init2_mpfarray(long int array_size, unsigned long prec)
{
	MPFArray st;
	long int i;

	st = (MPFArray)malloc(sizeof(mpfarray));
	if(st == NULL)
		return NULL;

	st->array = (mpf_t *)calloc(sizeof(mpf_t), array_size);
	if(st->array == NULL)
	{
		free(st);
		return NULL;
	}

	st->size = array_size;

	for(i = 0; i < array_size; i++)
		mpf_init2((mpf_ptr)st->array + i, prec);

	/* set precision */
	st -> prec = prec;

	return st;
}

void free_mpfarray(MPFArray st)
{
	long int i;

	if(st == NULL)
		return;

	if(st->array != NULL)
	{
		for(i = 0; i < st->size; i++)
			mpf_clear((mpf_ptr)st->array + i);
		free(st->array);
	}
	free(st);
}

mpf_ptr get_mpfarray_i(MPFArray array, long int index)
{
	return *(array->array + index);
}

void set_mpfarray_i(MPFArray array, long int index, mpf_t val)
{
	mpf_set((mpf_ptr)(array->array + index), val);
}

void set_mpfarray_i_d(MPFArray array, long int index, double val)
{
	mpf_set_d(*(array->array + index), val);
}

void set_mpfarray_i_str(MPFArray array, long int index, const char *str, int base)
{
	mpf_set_str(*(array->array + index), str, base);
}

/* get precision of MPFArray */
unsigned long int prec_mpfarray(MPFArray array)
{
	return array->prec;
}

void subst_mpfarray(MPFArray c, MPFArray a)
{
	long int i, size;

	size = a->size;
	if(c->size < a->size)
	{
		fprintf(stderr, "Warning: Not enough array size!\n");
		size = c->size;
	}

	for(i = 0; i < size; i++)
		set_mpfarray_i(c, i, get_mpfarray_i(a, i));
}

void print_mpfarray(MPFArray array)
{
	long int i;

	for(i = 0; i < array->size; i++)
	{
		printf("%5ld ", i);
		mpf_out_str(stdout, 10, 0, get_mpfarray_i(array, i));
		printf("\n");
	}
}

/*******************************/
/* For GMP                     */
/* init_cmpfarray              */
/* free_cmpfarray              */
/* set_cmpfarray_i             */
/* get_cmpfarray_i             */
/*******************************/
CMPFArray init_cmpfarray(long int array_size)
{
	CMPFArray st;
	long int i;
	mpfcmplx *array;

	st = (CMPFArray)malloc(sizeof(cmpfarray));
	if(st == NULL)
		return NULL;

	st->array = (mpfcmplx *)calloc(sizeof(mpfcmplx), array_size);
	if(st->array == NULL)
	{
		free(st);
		return NULL;
	}

	st->size = array_size;

	/* set precision */
	st -> prec = get_bnc_default_prec();

	for(i = 0; i < array_size; i++)
	{
		mpf_init_set_ui((st->array + i)->re, 0UL);
		mpf_init_set_ui((st->array + i)->im, 0UL);
		(st->array + i)->prec = st->prec;
	}

	return st;
}

CMPFArray init2_cmpfarray(long int array_size, unsigned long prec)
{
	CMPFArray st;
	long int i;

	st = (CMPFArray)malloc(sizeof(cmpfarray));
	if(st == NULL)
		return NULL;

	st->array = (mpfcmplx *)calloc(sizeof(mpfcmplx), array_size);
	if(st->array == NULL)
	{
		free(st);
		return NULL;
	}

	st->size = array_size;

	/* set precision */
	st -> prec = prec;

	for(i = 0; i < array_size; i++)
	{
		mpf_init2((st->array + i)->re, prec);
		mpf_set_ui((st->array + i)->re, 0UL);
		mpf_init2((st->array + i)->im, prec);
		mpf_set_ui((st->array + i)->im, 0UL);
		(st->array + i)->prec = st->prec;
	}


	return st;
}

void free_cmpfarray(CMPFArray st)
{
	long int i;

	if(st == NULL)
		return;

	if(st->array != NULL)
	{
		for(i = 0; i < st->size; i++)
		{
			mpf_clear((st->array + i)->re);
			mpf_clear((st->array + i)->im);
		}
		free(st->array);
	}
	free(st);
}

MPFCmplx get_cmpfarray_i(CMPFArray array, long int index)
{
	return (array->array + index);
}

void set_cmpfarray_i(CMPFArray array, long int index, MPFCmplx val)
{
	subst_mpfcmplx((array->array + index), val);
}

/* get precision of MPFArray */
unsigned long int prec_cmpfarray(CMPFArray array)
{
	return array->prec;
}

void subst_cmpfarray(CMPFArray c, CMPFArray a)
{
	long int i, size;

	size = a->size;
	if(c->size < a->size)
	{
		fprintf(stderr, "Warning: Not enough array size!\n");
		size = c->size;
	}

	for(i = 0; i < size; i++)
		set_cmpfarray_i(c, i, get_cmpfarray_i(a, i));
}

void print_cmpfarray(CMPFArray array)
{
	long int i;

	for(i = 0; i < array->size; i++)
	{
		printf("%5ld ", i);
		print_mpfcmplx(get_cmpfarray_i(array, i));
	}
}
#endif
