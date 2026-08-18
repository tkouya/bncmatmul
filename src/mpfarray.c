/********************************************************************************/
/* mpfarray.c:                                                                  */
/* Copyright (C) 2025 Tomonori Kouya                                            */
/*                                                                              */
/* Ver. 0.1 2025-01-15: Modify with MPC                                         */
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

#ifdef USE_GMP
/*******************************/
/* For GMP                     */
/* init_mpfarray               */
/* free_mpfarray               */
/* get_mpfarray_i              */
/* set_mpfarray_i              */
/*******************************/
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

// 202-01-21 T.Kouya
void set_mpfarray_i_ui(MPFArray array, long int index, unsigned long val)
{
	mpf_set_ui(*(array->array + index), val);
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
/* For MPC (New)               */
/* init_cmpfarray              */
/* free_cmpfarray              */
/* set_cmpfarray_i             */
/* get_cmpfarray_i             */
/*******************************/
CMPFArray init_cmpfarray(long int array_size)
{
	CMPFArray st;
	long int i;
	//mpfcmplx *array;
    mpc_t *array;

	st = (CMPFArray)malloc(sizeof(cmpfarray));
	if(st == NULL)
		return NULL;

	//st->array = (mpfcmplx *)calloc(sizeof(mpfcmplx), array_size);
    st->array = (mpc_t *)calloc(array_size, sizeof(mpc_t));
	if(st->array == NULL)
	{
		free(st);
		return NULL;
	}

	st->size = array_size;

	/* set precision */
	st->prec = get_bnc_default_prec();

	for(i = 0; i < array_size; i++)
	{
		//mpf_init_set_ui((st->array + i)->re, 0UL);
		//mpf_init_set_ui((st->array + i)->im, 0UL);
        mpc_init2(st->array[i], st->prec);
        mpc_set_ui(st->array[i], 0UL, get_bnc_default_rounding_mode_c());
		//(st->array + i)->prec = st->prec;
        //st->array[i]->prec = st->prec;
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

	//st->array = (mpfcmplx *)calloc(sizeof(mpfcmplx), array_size);
	st->array = (mpc_t *)calloc(array_size, sizeof(mpc_t));
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
		//mpf_init2((st->array + i)->re, prec);
		//mpf_set_ui((st->array + i)->re, 0UL);
		//mpf_init2((st->array + i)->im, prec);
		//mpf_set_ui((st->array + i)->im, 0UL);
        mpc_init2(st->array[i], prec);
        mpc_set_ui(st->array[i], 0UL, get_bnc_default_rounding_mode_c());
		//(st->array + i)->prec = st->prec;
        //st->array[i]->prec = st->prec;
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
			//mpf_clear((st->array + i)->re);
			//mpf_clear((st->array + i)->im);
            mpc_clear(st->array[i]);
		}
		free(st->array);
	}
	free(st);
}

// 2025-06-26(Thu) T.Kouya
//CMPFArray init_set_cmpfarray_mpc_array(mpc_t array[], long int num_array)
//{}

//MPFCmplx get_cmpfarray_i(CMPFArray array, long int index)
mpc_ptr get_cmpfarray_i(CMPFArray array, long int index)
{
	return (mpc_ptr)(array->array + index);
}

//void set_cmpfarray_i(CMPFArray array, long int index, MPFCmplx val)
void set_cmpfarray_i(CMPFArray array, long int index, mpc_t val)
{
	//subst_mpfcmplx((array->array + index), val);
    mpc_set(array->array[index], val, get_bnc_default_rounding_mode_c());
}

void set_cmpfarray_i_ui(CMPFArray array, long int index, unsigned long val)
{
	//subst_mpfcmplx((array->array + index), val);
    mpc_set_ui(array->array[index], val, get_bnc_default_rounding_mode_c());
}

void set_cmpfarray_i_d(CMPFArray array, long int index, double val)
{
	//subst_mpfcmplx((array->array + index), val);
    mpc_set_d(array->array[index], val, get_bnc_default_rounding_mode_c());
}

void set_cmpfarray_i_real(CMPFArray array, long int index, mpf_t val)
{
	//subst_mpfcmplx((array->array + index), val);
    mpc_set_fr(array->array[index], val, get_bnc_default_rounding_mode_c());
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
		//printf("%5ld ", i);
		//print_mpfcmplx(get_cmpfarray_i(array, i));
        mpfr_printf("%5ld (%RNe, %RNe)\n", i, mpc_realref(array->array[i]), mpc_imagref(array->array[i]));
	}
}

/*******************************/
/* For MPCmplx (Old)           */
/* _bncold_init_cmpfarray      */
/* _bncold_free_cmpfarray      */
/* _bncold_set_cmpfarray_i     */
/* _bncold_get_cmpfarray_i     */
/*******************************/
_bncold_CMPFArray _bncold_init_cmpfarray(long int array_size)
{
	_bncold_CMPFArray st;
	long int i;
	mpfcmplx *array;

	st = (_bncold_CMPFArray)malloc(sizeof(_bncold_cmpfarray));
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

_bncold_CMPFArray _bncold_init2_cmpfarray(long int array_size, unsigned long prec)
{
	_bncold_CMPFArray st;
	long int i;

	st = (_bncold_CMPFArray)malloc(sizeof(_bncold_cmpfarray));
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

void _bncold_free_cmpfarray(_bncold_CMPFArray st)
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

MPFCmplx _bncold_get_cmpfarray_i(_bncold_CMPFArray array, long int index)
{
	return (array->array + index);
}

void _bncold_set_cmpfarray_i(_bncold_CMPFArray array, long int index, MPFCmplx val)
{
    subst_mpfcmplx((array->array + index), val);
}

/* get precision of MPFArray */
unsigned long int _bncold_prec_cmpfarray(_bncold_CMPFArray array)
{
	return array->prec;
}

void _bncold_subst_cmpfarray(_bncold_CMPFArray c, _bncold_CMPFArray a)
{
	long int i, size;

	size = a->size;
	if(c->size < a->size)
	{
		fprintf(stderr, "Warning: Not enough array size!\n");
		size = c->size;
	}

	for(i = 0; i < size; i++)
		_bncold_set_cmpfarray_i(c, i, _bncold_get_cmpfarray_i(a, i));
}

void _bncold_print_cmpfarray(_bncold_CMPFArray array)
{
	long int i;

	for(i = 0; i < array->size; i++)
	{
		printf("%5ld ", i);
		print_mpfcmplx(_bncold_get_cmpfarray_i(array, i));
	}
}
#endif // USE_GMP

#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus
