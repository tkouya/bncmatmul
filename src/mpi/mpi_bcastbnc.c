/********************************************************************************/
/* mpi_bcastbnc.c: Broadcast BNC defined data                                   */
/* Copyright (C) 2003-2011 Tomonori Kouya, All rights reserved.                 */
/*                                                                              */
/* Version 0.0: 2003.08/23                                                      */
/* Version 0.1: 2005.05/28 support MPICH2                                       */
/* Version 0.2: 2005.07/12 support Allgather                                    */
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

/* Bcast DPoly from P0 to other processes */
void _mpi_bcast_dpoly(DPoly poly, MPI_Comm comm)
{
	long int i, deg, max_len;
	int procs, num_procs;
	MPI_Status st;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &procs);

	if(procs == 0)
	{
		deg = poly->deg;
		max_len = poly->max_len;
	}

	MPI_Bcast(&deg, 1, MPI_LONG, 0, comm);
	MPI_Bcast(&max_len, 1, MPI_LONG, 0, comm);
	poly->deg = deg;
	poly->max_len = max_len;

	/* P0 -> Pi */

	if(max_len <= 0)
		return;

#ifndef MPICH2
	if(procs == 0)
	{
		for(i = 1; i < num_procs; i++)
		{
			MPI_Send(
				poly->coef,
				max_len,
				MPI_DOUBLE,
				i,
				0,
				comm);
		}
	}
	else
	{
		MPI_Recv(
			poly->coef,
			max_len,
			MPI_DOUBLE,
			0, 0, comm, &st);
	}
#else
	MPI_Bcast(poly->coef, max_len, MPI_DOUBLE, 0, comm);
#endif
}

/* Bcast DVector from P0 to other processes */
void _mpi_bcast_dvector(DVector vec, MPI_Comm comm)
{
	long int i, dim;
	int procs, num_procs;
	MPI_Status st;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &procs);

	if(procs == 0)
	{
		dim = vec->dim;
	}

	MPI_Bcast(&dim, 1, MPI_LONG, 0, comm);
	vec->dim = dim;

	/* P0 -> Pi */

	if(dim <= 0)
		return;

#ifndef MPICH2
	if(procs == 0)
	{
		for(i = 1; i < num_procs; i++)
		{
			MPI_Send(
				vec->element,
				dim,
				MPI_DOUBLE,
				i,
				0,
				comm);
		}
	}
	else
	{
		MPI_Recv(
			vec->element,
			dim,
			MPI_DOUBLE,
			0, 0, comm, &st);
	}
#else
	MPI_Bcast(vec->element, dim, MPI_DOUBLE, 0, comm);
#endif
}

/* Allgather DVector especially for ODEs */
void _mpi_allgather_dvector(DVector full_vec, DVector local_vec, MPI_Comm comm)
{
	long int i, local_dim;
	int procs, num_procs;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &procs);

	/* check dim sizes */
	if(full_vec->dim < (local_vec->dim * num_procs))
	{
		fprintf(stderr, "full_vec's dim size(%d) is too small!!(_mpi_allgather_dvector)\n", full_vec->dim); 
		return;
	}

	if((local_vec->dim <= 0) || (full_vec->dim <= 0))
		return;

	local_dim = local_vec->dim;
	MPI_Allgather(local_vec->element, local_dim, MPI_DOUBLE, full_vec->element, local_dim, MPI_DOUBLE, comm);
}

#ifdef USE_GMP

/* Bcast MPFPoly from P0 to other processes */
void _mpi_bcast_mpfpoly(MPFPoly poly, MPI_Comm comm)
{
	long int i, deg, max_len;
	unsigned long prec;
	void *buf;
	int procs, num_procs;
	MPI_Status st;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &procs);

	if(procs == 0)
	{
		deg = poly->deg;
		max_len = poly->max_len;
		prec = poly->prec;
	}

	MPI_Bcast(&deg, 1, MPI_LONG, 0, comm);
	MPI_Bcast(&max_len, 1, MPI_LONG, 0, comm);
	MPI_Bcast(&prec, 1, MPI_UNSIGNED_LONG, 0, comm);
	poly->deg = deg;
	poly->max_len = max_len;
	poly->prec = prec;

	/* P0 -> Pi */

	if(max_len <= 0)
		return;

	buf = allocbuf_mpf(prec, max_len);

#ifndef MPICH2
	if(procs == 0)
	{
		for(i = 1; i < num_procs; i++)
		{
			pack_mpf(get_mpfpoly_i(poly, 0), max_len, buf);
			MPI_Send(
				buf,
				max_len,
				MPI_MPF,
				i,
				0,
				comm);
		}
	}
	else
	{
		MPI_Recv(
			buf,
			max_len,
			MPI_MPF,
			0, 0, comm, &st);
		unpack_mpf(buf, get_mpfpoly_i(poly, 0), max_len);
	}
#else
	if(procs == 0)
		pack_mpf(get_mpfpoly_i(poly, 0), max_len, buf);

	MPI_Bcast(buf, max_len, MPI_MPF, 0, comm);
	unpack_mpf(buf, get_mpfpoly_i(poly, 0), max_len);
#endif

	free(buf);
}

/* Bcast MPFVector from P0 to other processes */
void _mpi_bcast_mpfvector(MPFVector vec, MPI_Comm comm)
{
	long int i, dim;
	unsigned long prec;
	void *buf;
	int procs, num_procs;
	MPI_Status st;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &procs);

	if(procs == 0)
	{
		dim = vec->dim;
		prec = vec->prec;
	}

	MPI_Bcast(&dim, 1, MPI_LONG, 0, comm);
	MPI_Bcast(&prec, 1, MPI_UNSIGNED_LONG, 0, comm);
	vec->dim = dim;
	vec->prec = prec;

	/* P0 -> Pi */

	if(dim <= 0)
		return;

	buf = allocbuf_mpf(prec, dim);

#ifndef MPICH2
	if(procs == 0)
	{
		for(i = 1; i < num_procs; i++)
		{
			pack_mpf(get_mpfvector_i(vec, 0), dim, buf);
			MPI_Send(
				buf,
				dim,
				MPI_MPF,
				i,
				0,
				comm);
		}
	}
	else
	{
		MPI_Recv(
			buf,
			dim,
			MPI_MPF,
			0, 0, comm, &st);
		unpack_mpf(buf, get_mpfvector_i(vec, 0), dim);
	}
#else
	if(procs == 0)
		pack_mpf(get_mpfvector_i(vec, 0), dim, buf);

	MPI_Bcast(buf, dim, MPI_MPF, 0, comm);
	unpack_mpf(buf, get_mpfvector_i(vec, 0), dim);
#endif

	free(buf);
}

/* Allgather DVector especially for ODEs */
void _mpi_allgather_mpfvector(MPFVector full_vec, MPFVector local_vec, MPI_Comm comm)
{
	long int i, local_dim, dim;
	int procs, num_procs;
	void *local_buf, *buf;
	unsigned long prec;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &procs);

	/* check dim sizes */
	if(full_vec->dim < (local_vec->dim * num_procs))
	{
		fprintf(stderr, "full_vec's dim size(%d) is too small!!(_mpi_allgather_mpfvector)\n", full_vec->dim); 
		return;
	}

	if((local_vec->dim <= 0) || (full_vec->dim <= 0))
		return;

	local_dim = local_vec->dim;
	dim = full_vec->dim;
	prec = local_vec->prec;
	local_buf = allocbuf_mpf(prec, local_dim);
	buf = allocbuf_mpf(prec, dim);

	pack_mpf(get_mpfvector_i(local_vec, 0), local_dim, local_buf);
	MPI_Allgather(local_buf, local_dim, MPI_MPF, buf, local_dim, MPI_MPF, comm);
	unpack_mpf(buf, get_mpfvector_i(full_vec, 0), dim);

	free(local_buf);
	free(buf);
}
#endif
