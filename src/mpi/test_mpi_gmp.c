/********************************************************************************/
/* test_mpi_gmp.c: Test Program for printing information on MPI env        */
/* Copyright (C) 2011 Tomonori Kouya                                            */
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
#include <string.h>
#include <math.h>

#include "mpi.h"
#include "mpi_gmp.h"

#define ARRAY_DIM 10
//#define PREC 128
#define PREC 256

main()
{
	long int i;
	mpf_t array[ARRAY_DIM], other_array[ARRAY_DIM];
	void *array_buf;
	void *other_array_buf;

	mpf_set_default_prec(PREC);

	for(i = 0; i < ARRAY_DIM; i++)
	{	
		mpf_init(array[i]);
		mpf_init(other_array[i]);
	}

	mpf_sqrt_ui(array[0], 2UL);
	if(ARRAY_DIM >= 2)
		mpf_sqrt_ui(array[1], 5UL);
	for(i = 2; i < ARRAY_DIM; i++)
	{
		mpf_mul(array[i], array[i-1], array[i-2]);
		mpf_mul_ui(array[i], array[i], (i + 1));
		printf("%5d ", i); mpf_out_str(stdout, 10, 0, array[i]); printf("\n");
//		printf("    %x->(%d, %d, %d, %d)\n", (unsigned long)array[i], array[i]->_mpfr_sign, array[i]->_mpfr_prec, array[i]->_mpfr_exp, array[i]->_mpfr_d);
	}

	/* alloc */
	array_buf = allocbuf_mpf(PREC, ARRAY_DIM);
	other_array_buf = allocbuf_mpf(PREC, ARRAY_DIM);

	/* pack */
	pack_mpf(array[0], ARRAY_DIM, array_buf);

	/* memcpy */
	memcpy(other_array_buf, array_buf, get_bufsize_mpf(array[0], ARRAY_DIM));

	/* unpack */
	//unpack_mpf(other_array_buf, other_array[0], ARRAY_DIM);
	unpack_mpf(array_buf, other_array[0], ARRAY_DIM);

	/* free */
	free(array_buf);
	free(other_array_buf);

	for(i = 0; i < ARRAY_DIM; i++)
	{
		mpf_sub(other_array[i], other_array[i], array[i]);
		printf("%5d ", i); mpf_out_str(stdout, 10, 0, other_array[i]); printf("\n");
		//printf("    %x->(%d, %d, %d, %d)\n", (unsigned long)other_array[i], other_array[i]->_mpfr_sign, other_array[i]->_mpfr_prec, other_array[i]->_mpfr_exp, other_array[i]->_mpfr_d);

		mpf_clear(array[i]);
		mpf_clear(other_array[i]);
	}
}
