/********************************************************************************/
/* mpi_complex.c:                                                               */
/* Copyright (C) 2003-2011 Tomonori Kouya, All rights reserved.                 */
/*                                                                              */
/* Version 0.0: 2003.08/21                                                      */
/* Version 0.1: 2003.10/04                                                      */
/* Version 0.2: 2005.05/28 suport MPICH2                                        */
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
#include <math.h>

#include "mpi.h"
#include "bnc.h"
#include "mpi_gmp.h"
#include "mpi_bnc.h"

/***************************************/
/* Complex Number                      */
/***************************************/

/* double complex */

void *allocbuf_dcmplx(int incount)
{
	size_t bufsize;
	void *ret;

	bufsize = sizeof(double) * 2 * incount;

	ret = (double *)malloc(bufsize);

	return ret;
}

int pack_cdarray(CDArray array, void *buf)
{
	long int i;
	unsigned char *tmp_buf;
	CDArray tmp_array;
	dcmplx tmp_cmplx;

	tmp_buf = (unsigned char *)buf;
	tmp_array = array;
	for(i = 0; i < array->size; i++)
	{
//		printf("pack(%f, %f)\n", tmp_array->array->re, tmp_array->array->im);
		tmp_cmplx.re = get_real_dcmplx(get_cdarray_i(tmp_array, i));
		tmp_cmplx.im = get_image_dcmplx(get_cdarray_i(tmp_array, i));
		memcpy(tmp_buf, &(tmp_cmplx.re), sizeof(double));
		tmp_buf += sizeof(double);
		memcpy(tmp_buf, &(tmp_cmplx.im), sizeof(double));
		tmp_buf += sizeof(double);
	}

	return (int)(2 * array->size);
}

void unpack_cdarray(void *buf, CDArray array, long int size)
{
	long int i;
	unsigned char *tmp_buf;
	CDArray tmp_array;
	dcmplx tmp_cmplx;

	tmp_buf = (unsigned char *)buf;
	tmp_array = array;
	for(i = 0; i < size; i++)
	{
		memcpy(&(tmp_cmplx.re), tmp_buf, sizeof(double));
		tmp_buf += sizeof(double);
		memcpy(&(tmp_cmplx.im), tmp_buf, sizeof(double));
		tmp_buf += sizeof(double);
//		printf("%d %d unpack(%f, %f)\n", size, i, tmp_cmplx.re, tmp_cmplx.im);
//		tmp_array->array++;
		set_cdarray_i(tmp_array, i, &tmp_cmplx);
	}

	array->size = size;
}
void unpack_cdarray_i(void *buf, CDArray array, long int index, long int size)
{
	long int i;
	unsigned char *tmp_buf;
	CDArray tmp_array;
	dcmplx tmp_cmplx;

	tmp_buf = (unsigned char *)buf;
	tmp_array = array;
	for(i = 0; i < size; i++)
	{
		memcpy(&(tmp_cmplx.re), tmp_buf, sizeof(double));
		tmp_buf += sizeof(double);
		memcpy(&(tmp_cmplx.im), tmp_buf, sizeof(double));
		tmp_buf += sizeof(double);
//		printf("%d %d unpack(%f, %f)\n", size, i, tmp_cmplx.re, tmp_cmplx.im);
//		tmp_array->array++;
		set_cdarray_i(tmp_array, i + index, &tmp_cmplx);
	}

//	array->size = size;
}

/* Collect vectors on P0 */
void _mpi_collect_cdarray(CDArray src_array, long int d_dim[], CDArray d_array, MPI_Comm comm)
{
	long int local_dim, index, total_index, i, dim;
	int myrank, procs, num_procs;
	MPI_Status st;
	void *sbuf, *rbuf;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &procs);

	if(procs == 0)
		dim = src_array->size;

	MPI_Bcast(&dim, 1, MPI_LONG, 0, comm);
//	printf("rank: %d, dim = %d\n", procs, dim);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);

	/* Pi -> P0 */

#ifndef MPICH2
	sbuf = allocbuf_dcmplx((int)local_dim);
	pack_cdarray(d_array, sbuf);
	MPI_Send(
		sbuf,
		d_dim[procs] * 2,
		MPI_DOUBLE,
		0, 0, comm);
	free(sbuf);

	if(procs == 0)
	{
//		print_dvector(d_vec);
		total_index = 0;
		rbuf = allocbuf_dcmplx((int)local_dim);
		for(i = 0; i < num_procs; i++)
		{
			MPI_Recv(
				rbuf,
				d_dim[i] * 2,
				MPI_DOUBLE,
				i,
				0,
				comm, &st);
			unpack_cdarray_i(rbuf, src_array, total_index, d_dim[i]);
			total_index += d_dim[i];
		}
		free(rbuf);
	}
#else
	sbuf = allocbuf_dcmplx((int)local_dim);
	pack_cdarray(d_array, sbuf);
	if(procs == 0)
		rbuf = allocbuf_dcmplx((int)(local_dim * num_procs));

	MPI_Gather(
		(void *)sbuf,
		local_dim * 2,
		MPI_DOUBLE,
		(void *)rbuf,
		local_dim * 2,
		MPI_DOUBLE,
		0,
		comm);

	if(procs == 0)
	{
		unpack_cdarray_i(rbuf, src_array, 0, local_dim * num_procs);
		free(rbuf);
	}

	free(sbuf);
#endif

}

#ifdef USE_GMP

/* mpf_t */

MPI_Datatype _bnc_mpfcmplx;
MPI_Datatype _tmp_bnc_mpfcmplx;

#define MPI_BNC_MPFCMPLX _bnc_mpfcmplx

/* typedef and commit to mpich */
void commit_mpi_mpfcmplx(MPI_Datatype *mpfcmplx_t, unsigned long prec, MPI_Comm comm)
{
	int pos;

	/* commit mpf_t */
	commit_mpf(&_tmp_bnc_mpfcmplx, prec, comm);

	MPI_Type_contiguous(2, _tmp_bnc_mpfcmplx, mpfcmplx_t);
	MPI_Type_commit(mpfcmplx_t);

	/* confirm */
	MPI_Type_size(_tmp_bnc_mpfcmplx, &pos);
//	printf("_tmp_bnc_mpfcmplx -> %d\n", pos);
	MPI_Type_size(*mpfcmplx_t, &pos);
//	printf("       mpfcmplx_t -> %d\n", pos);
}

/* clear type */
void free_mpi_mpfcmplx(MPI_Datatype *mpfcmplx_t)
{
	free_mpf(&_tmp_bnc_mpfcmplx);
	MPI_Type_free(mpfcmplx_t);
}

size_t get_bufsize_mpfcmplx(MPFCmplx a, int incount)
{
	size_t bufsize;

	bufsize = (size_t)(incount * (get_bufsize_mpf(a->re, 1) + get_bufsize_mpf(a->im, 1)));

	return bufsize;
}

void *allocbuf_mpfcmplx(unsigned long prec, int incount)
{
	void *buf;
	MPFCmplx a;

	a = init2_mpfcmplx(prec);

	buf = (void *)malloc(get_bufsize_mpfcmplx(a, incount));

	free_mpfcmplx(a);

	return buf;
}

int pack_cmpfarray(CMPFArray array, void *buf)
{
	long int i;
	unsigned char *tmp_buf;
	CMPFArray tmp_array;
	MPFCmplx tmp_cmplx;

	tmp_buf = (unsigned char *)buf;
	tmp_array = array;
	tmp_cmplx = init2_mpfcmplx(array->prec);

	if(buf == NULL)
		buf = allocbuf_mpfcmplx(array->prec, array->size);

	for(i = 0; i < array->size; i++)
	{
		subst_mpfcmplx(tmp_cmplx, get_cmpfarray_i(tmp_array, i));

		pack_mpf(tmp_cmplx->re, 1, tmp_buf);
		tmp_buf += get_bufsize_mpf(tmp_cmplx->re, 1);

		pack_mpf(tmp_cmplx->im, 1, tmp_buf);
		tmp_buf += get_bufsize_mpf(tmp_cmplx->im, 1);

	}

	free_mpfcmplx(tmp_cmplx);

	return (int)(array->size);
}

int pack_cmpfarray_size(CMPFArray array, void *buf, long int index, long int size)
{
	long int i;
	unsigned char *tmp_buf;
	CMPFArray tmp_array;
	MPFCmplx tmp_cmplx;

	tmp_buf = (unsigned char *)buf;
	tmp_array = array;
	tmp_cmplx = init2_mpfcmplx(array->prec);

	if(buf == NULL)
		buf = allocbuf_mpfcmplx(array->prec, size);

	for(i = 0; i < size; i++)
	{
		subst_mpfcmplx(tmp_cmplx, get_cmpfarray_i(tmp_array, index+i));

		pack_mpf(tmp_cmplx->re, 1, tmp_buf);
		tmp_buf += get_bufsize_mpf(tmp_cmplx->re, 1);

		pack_mpf(tmp_cmplx->im, 1, tmp_buf);
		tmp_buf += get_bufsize_mpf(tmp_cmplx->im, 1);

	}

	free_mpfcmplx(tmp_cmplx);

	return (int)(array->size);
}

void unpack_cmpfarray(void *buf, CMPFArray array, long int size)
{
	long int i;
	unsigned char *tmp_buf;
	CMPFArray tmp_array;
	MPFCmplx tmp_cmplx;

	tmp_buf = (unsigned char *)buf;
	tmp_array = array;
	tmp_cmplx = init2_mpfcmplx(array->prec);

	for(i = 0; i < size; i++)
	{
		unpack_mpf(tmp_buf, tmp_cmplx->re, 1);
		tmp_buf += get_bufsize_mpf(tmp_cmplx->re, 1);

		unpack_mpf(tmp_buf, tmp_cmplx->im, 1);
		tmp_buf += get_bufsize_mpf(tmp_cmplx->im, 1);

		set_cmpfarray_i(tmp_array, i, tmp_cmplx);

	}

	free_mpfcmplx(tmp_cmplx);

	array->size = size;
}

void unpack_cmpfarray_i(void *buf, CMPFArray array, long int index, long int size)
{
	long int i;
	unsigned char *tmp_buf;
	CMPFArray tmp_array;
	MPFCmplx tmp_cmplx;

	tmp_buf = (unsigned char *)buf;
	tmp_array = array;
	tmp_cmplx = init2_mpfcmplx(array->prec);

	for(i = 0; i < size; i++)
	{
		unpack_mpf(tmp_buf, tmp_cmplx->re, 1);
		tmp_buf += get_bufsize_mpf(tmp_cmplx->re, 1);

		unpack_mpf(tmp_buf, tmp_cmplx->im, 1);
		tmp_buf += get_bufsize_mpf(tmp_cmplx->im, 1);

		set_cmpfarray_i(tmp_array, i + index, tmp_cmplx);

	}

	free_mpfcmplx(tmp_cmplx);

//	array->size = size;
}

/* Divide original array on P0 */
void _mpi_divide_cmpfarray(CMPFArray d_vec, long int d_dim[], CMPFArray src_vec, MPI_Comm comm)
{
	long int local_dim, index, total_index, i, dim;
	int myrank, procs, num_procs;
	MPI_Status st;
	void *sbuf, *rbuf;

	MPI_Comm_size(comm, &num_procs);
//	MPI_Comm_rank(comm, &myrank);
	MPI_Comm_rank(comm, &procs);

	if(procs == 0)
		dim = src_vec->size;

	MPI_Bcast(&dim, 1, MPI_LONG, 0, comm);
//	printf("rank: %d, dim = %d\n", procs, dim);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);

//	for(i = 0; i < procs - 1; i++)
//	printf("rank: %d, dim[%d] = %d, total_index = %d\n", procs, procs, d_dim[procs], total_index);

	/* P0 -> Pi */
	if(procs == 0)
	{
		total_index = 0;
		sbuf = allocbuf_mpfcmplx(src_vec->prec, local_dim);
		for(i = 0; i < num_procs; i++)
		{
			MPI_Send(
				&src_vec->prec,
				1,
				MPI_UNSIGNED_LONG,
				i,
				0,
				comm);
			if(d_dim[i] <= 0)
				break;
			pack_cmpfarray_size(src_vec, sbuf, total_index, d_dim[i]);
			MPI_Send(
				sbuf,
				d_dim[i],
				MPI_BNC_MPFCMPLX,
				i,
				0,
				comm);
			total_index += d_dim[i];
		}
		free(sbuf);
	}
//	else
	{
		MPI_Recv(
			&d_vec->prec,
			1,
			MPI_UNSIGNED_LONG,
			0, 0, comm, &st);

//		printf("(rank: %d, dim: %d)", procs, d_dim[procs]);print_mpfvector(d_vec);
		if(d_dim[procs] <= 0)
			return;

		rbuf = allocbuf_mpfcmplx(d_vec->prec, d_dim[procs]);
//		printf("(rank: %d, prec->%d)\n", procs, d_vec->prec);
		MPI_Recv(
			rbuf,
			d_dim[procs],
			MPI_BNC_MPFCMPLX,
			0, 0, comm, &st);
		unpack_cmpfarray(rbuf, d_vec, d_dim[procs]);
		free(rbuf);
	}
}

/* Collect array on P0 */
void _mpi_collect_cmpfarray(CMPFArray src_array, long int d_dim[], CMPFArray d_array, MPI_Comm comm)
{
	long int local_dim, index, total_index, i, dim;
	int myrank, procs, num_procs;
	MPI_Status st;
	void *sbuf, *rbuf;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &procs);

	if(procs == 0)
		dim = src_array->size;

	MPI_Bcast(&dim, 1, MPI_LONG, 0, comm);
	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);
//	printf("rank: %d, dim = %d, local_dim(d_dim[%d]): %d(%d)\n", procs, dim, procs, local_dim, d_dim[procs]);


	/* Pi -> P0 */

#ifndef MPICH2
	MPI_Send(
		&d_array->prec,
		1,
		MPI_UNSIGNED_LONG,
		0, 0, comm);

//	printf("(rank: %d, d_dim[%d]: %d, prec: %d)", procs, procs, d_dim[procs], d_array->size);
	sbuf = allocbuf_mpfcmplx(d_array->prec, local_dim);
//	pack_mpf(gmpfvi(d_vec, 0), d_dim[procs], sbuf);
	pack_cmpfarray(d_array, sbuf);
	MPI_Send(
		sbuf,
		d_dim[procs],
		MPI_BNC_MPFCMPLX,
		0, 0, comm);
	free(sbuf);

	if(procs == 0)
	{
//		print_cmpfarray(d_array);
		total_index = 0;
		rbuf = allocbuf_mpfcmplx(src_array->prec, local_dim);
		for(i = 0; i < num_procs; i++)
		{
			MPI_Recv(
				&src_array->prec,
				1,
				MPI_UNSIGNED_LONG,
				i,
				0,
				comm, &st);
			MPI_Recv(
				rbuf,
				d_dim[i],
				MPI_BNC_MPFCMPLX,
				i,
				0,
				comm, &st);
			unpack_cmpfarray_i(rbuf, src_array, total_index, d_dim[i]);
			total_index += d_dim[i];
		}
		free(rbuf);
	}
#else
	sbuf = allocbuf_mpfcmplx(d_array->prec, local_dim);
	pack_cmpfarray(d_array, sbuf);
	if(procs == 0)
		rbuf = allocbuf_mpfcmplx(src_array->prec, local_dim * num_procs);

	MPI_Gather(
		(void *)sbuf,
		local_dim,
		MPI_BNC_MPFCMPLX,
		(void *)rbuf,
		local_dim,
		MPI_BNC_MPFCMPLX,
		0,
		comm);

	if(procs == 0)
	{
		unpack_cmpfarray_i(rbuf, src_array, 0, local_dim * num_procs);
		free(rbuf);
	}

	free(sbuf);
#endif

}
#endif
