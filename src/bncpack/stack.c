/********************************************************************************/
/* stack.c:                                                                     */
/* Copyright (C) 2003-2011 Tomonori Kouya                                       */
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
/* init_fstack                 */
/* free_fstack                 */
/* push_fstack                 */
/* pop_fstack                  */
/*******************************/
FStack init_fstack(long int stack_size)
{
	FStack st;
	long int i;

	st = (FStack)malloc(sizeof(fstack));
	if(st == NULL)
		return NULL;

	st->array = (float *)calloc(sizeof(float), stack_size);
	if(st->array == NULL)
	{
		free(st);
		return NULL;
	}

	st->index = -1; /* top of stack : 0 .. (size - 1) */
	st->size = stack_size;

	for(i = 0; i < stack_size; i++)
		*(st->array + i) = 0.0;

	return st;
}

void free_fstack(FStack st)
{
	if(st == NULL)
		return;

	if(st->array != NULL)
		free(st->array);

	free(st);
}

void push_fstack(FStack st, float val)
{
	if(st->index >= st->size)
	{
		fprintf(stderr, "FStack Overflow!(%ld)\n", st->index);
		return;
	}
	st->index++;
	*(st->array + st->index) = val;
}

float pop_fstack(FStack st)
{
	float rval = 0.0;

	if(st->index < 0)
	{
		fprintf(stderr, "FStack Underflow!(%ld)\n", st->index);
		return rval;
	}
	rval =  *(st->array + st->index);
	st->index--;

	return rval;
}

/*******************************/
/* For IEEE 754 double         */
/* init_dstack                 */
/* free_dstack                 */
/* push_dstack                 */
/* pop_dstack                  */
/*******************************/
DStack init_dstack(long int stack_size)
{
	DStack st;
	long int i;

	st = (DStack)malloc(sizeof(dstack));
	if(st == NULL)
		return NULL;

	st->array = (double *)calloc(sizeof(double), stack_size);
	if(st->array == NULL)
	{
		free(st);
		return NULL;
	}

	st->index = -1; /* top of stack : 0 .. (size - 1) */
	st->size = stack_size;

	for(i = 0; i < stack_size; i++)
		*(st->array + i) = 0.0;

	return st;
}

void free_dstack(DStack st)
{
	if(st == NULL)
		return;

	if(st->array != NULL)
		free(st->array);

	free(st);
}

void push_dstack(DStack st, double val)
{
	if(st->index >= st->size)
	{
		fprintf(stderr, "DStack Overflow!(%ld)\n", st->index);
		return;
	}
	st->index++;
	*(st->array + st->index) = val;
}

double pop_dstack(DStack st)
{
	double rval = 0.0;

	if(st->index < 0)
	{
		fprintf(stderr, "DStack Underflow!(%ld)\n", st->index);
		return rval;
	}
	rval =  *(st->array + st->index);
	st->index--;

	return rval;
}

/*******************************/
/* For GMP                     */
/* init_mpfstack               */
/* free_mpfstack               */
/* push_mpfstack               */
/* pop_mpfstack                */
/*******************************/
#ifdef USE_GMP
MPFStack init_mpfstack(long int stack_size)
{
	MPFStack st;
	long int i;

	st = (MPFStack)malloc(sizeof(mpfstack));
	if(st == NULL)
		return NULL;

	st->array = (mpf_t *)calloc(sizeof(mpf_t), stack_size);
	if(st->array == NULL)
	{
		free(st);
		return NULL;
	}

	st->index = -1; /* top of stack : 0 .. (size - 1) */
	st->size = stack_size;

	for(i = 0; i < stack_size; i++)
		mpf_init((mpf_ptr)st->array + i);

	/* set precision */
	st -> prec = get_bnc_default_prec();

	return st;
}

MPFStack init2_mpfstack(long int stack_size, unsigned long prec)
{
	MPFStack st;
	long int i;

	st = (MPFStack)malloc(sizeof(mpfstack));
	if(st == NULL)
		return NULL;

	st->array = (mpf_t *)calloc(sizeof(mpf_t), stack_size);
	if(st->array == NULL)
	{
		free(st);
		return NULL;
	}

	st->index = -1; /* top of stack : 0 .. (size - 1) */
	st->size = stack_size;

	for(i = 0; i < stack_size; i++)
		mpf_init2((mpf_ptr)st->array + i, prec);

	/* set precision */
	st -> prec = prec;

	return st;
}

void free_mpfstack(MPFStack st)
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

void push_mpfstack(MPFStack st, mpf_t val)
{
	if(st->index >= st->size)
	{
		fprintf(stderr, "MPFStack Overflow!(%ld)\n", st->index);
		return;
	}
	st->index++;
	mpf_set((mpf_ptr)st->array + st->index, val);
}

void pop_mpfstack(mpf_t rval, MPFStack st)
{
	if(st->index < 0)
	{
		fprintf(stderr, "MPFStack Underflow!(%ld)\n", st->index);
		return;
	}
	mpf_set(rval, (mpf_ptr)st->array + st->index);
	st->index--;
}
#endif
