/********************************************************************************/
/* print_process.c:                                                             */
/* Copyright (C) 2003-2011 Tomonori Kouya                                       */
/*                                                                              */
/* Version 0.0: 2003.08/21                                                      */
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
#include "mpi.h"
#include <stdio.h>
#include <math.h>

#ifdef USE_GMP
#include "mpi_gmp.h"
#endif

int main(int argc,char *argv[])
{
    double startwtime = 0.0, endwtime;
    int  namelen, numprocs, myid;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Datatype gmp_mpf;
    MPI_Op gmp_mpf_add;

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Get_processor_name(processor_name,&namelen);

    fprintf(stdout,"Process %d of %d on %s\n",
	    myid, numprocs, processor_name);

#ifdef USE_GMP
#define MPF_PREC 8192

    mpf_set_default_prec(MPF_PREC);
    commit_mpf(&gmp_mpf, MPF_PREC, MPI_COMM_WORLD);
    create_mpf_op(&gmp_mpf_add, _mpi_mpf_add, MPI_COMM_WORLD);

    free_mpf(&gmp_mpf);
    free_mpf_op(&gmp_mpf_add);
#endif

    MPI_Finalize();
    return 0;
}
