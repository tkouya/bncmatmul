/********************************************************************************/
/* array.c:                                                                     */
/* Copyright (C) 2003-2025 Tomonori Kouya                                       */
/*                                                                              */
/* Ver. 0.2 2025-01-25: Modify with MPC                                         */
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
//#include "bnc.h"
#include "poly.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

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
