/********************************************************************************/
/* mpi_aux.c:                                                                   */
/* Copyright (C) 2003-2011 Tomonori Kouya, All rights reserved.                 */
/*                                                                              */
/* Version 0.0: 2003.10/04                                                      */
/* Version 0.1: 2005.05/28 support MPICH2                                       */
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

#ifdef USE_GMP
#include "mpi_gmp.h"
#endif

#include "mpi_bnc.h"

/* set precition and default rounding mode */
#ifdef USE_GMP
void _mpi_set_bnc_default_prec(unsigned long prec, MPI_Comm comm)
{
	int myrank, num_procs;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	_set_bnc_default_prec(prec);
	if(myrank == 0)
	{
		set_bnc_default_prec(prec);
#ifndef LAM_MPI
#ifdef OMPI_MPI_H
#else
		printf("MPICH%d, MPI Version %d.%d\n", MPICH_NAME, MPI_VERSION, MPI_SUBVERSION);
#endif
#else
		printf("LAM_MPI, MPI Version %d.%d\n", MPI_VERSION, MPI_SUBVERSION);
#endif
		printf("-------------------------------------------------------------------------------\n");
	}
}
void _mpi_set_bnc_default_prec_decimal(unsigned long prec, MPI_Comm comm)
{
	int myrank, num_procs;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	_set_bnc_default_prec_decimal(prec);
	if(myrank == 0)
	{
		set_bnc_default_prec_decimal(prec);
#ifndef LAM_MPI
#ifdef OMPI_MPI_H
#else
		printf("MPICH%d, MPI Version %d.%d\n", MPICH_NAME, MPI_VERSION, MPI_SUBVERSION);
#endif
#else
		printf("LAM, MPI Version %d.%d\n", MPI_VERSION, MPI_SUBVERSION);
#endif
		printf("-------------------------------------------------------------------------------\n");
	}
}
#ifdef USE_MPFR
void _mpi_set_bnc_rounding_mode(mp_rnd_t rmode, MPI_Comm comm)
{
	int myrank, num_procs;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	_set_bnc_rounding_mode(rmode);
	if(myrank == 0) set_bnc_rounding_mode(rmode);
}
#endif
#endif
