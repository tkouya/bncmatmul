/********************************************************************************/
/* test_mpiint.c:                                                               */
/* Copyright (C) 2003 Tomonori Kouya                                            */
/*                                                                              */
/* Version 0.0: 2003.08/21                                                      */
/* Version 0.1: 2003.08/21                                                      */
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

#include "mpi_bnc.h"

double f(double a)
{
	return (4.0 / (1.0 + a*a));
}

#ifdef USE_GMP
void mpf_f(mpf_t ret, mpf_t a)
{
	mpf_t tmp;

	mpf_init2(tmp, mpf_get_prec(ret));

	mpf_mul(tmp, a, a);
	mpf_add_ui(tmp, tmp, 1UL);
	mpf_ui_div(ret, 4UL, tmp);

	mpf_clear(tmp);

	return;
}
#endif

int main(int argc,char *argv[])
{
	int n, myid, numprocs, i;
	double pi, start_x, end_x;
	double startwtime = 0.0, endwtime;
	int  namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

#ifdef USE_GMP
	mpf_t mpf_pi, mpf_h, mpf_x;
	int mpf_size, pos;
#endif

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	MPI_Get_processor_name(processor_name,&namelen);

	fprintf(stdout,"Process %d of %d on %s\n", myid, numprocs, processor_name);

#ifdef USE_GMP
#define MPF_PREC 256

	mpf_set_default_prec(MPF_PREC);
	commit_mpf(&(MPI_MPF), MPF_PREC, MPI_COMM_WORLD);
	create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, MPI_COMM_WORLD);
#endif
	n=14 * 16384;


#ifdef USE_GMP
	mpf_init(mpf_h); mpf_init(mpf_x); mpf_init(mpf_pi);
#endif

	if(myid == 0) startwtime = MPI_Wtime();
	start_x = 0.0; end_x = 1.0;
	_mpi_dtrapezoidal_fs(&pi, start_x, end_x, f, n, MPI_COMM_WORLD);
	if(myid == 0)
	{
		endwtime = MPI_Wtime() - startwtime;
		printf("BNC: _mpi_dtrapezoidal_fs = %e\n", pi);
		printf("Time: %f\n", endwtime);
	} 

	if(myid == 0) startwtime = MPI_Wtime();
	start_x = 0.0; end_x = 1.0;
	_mpi_dtrapezoidal_fs_all(&pi, start_x, end_x, f, n, MPI_COMM_WORLD);
	if(myid == 0)
	{
		endwtime = MPI_Wtime() - startwtime;
		printf("BNC: _mpi_dtrapezoidal_fs_all = %e\n", pi);
		printf("Time: %f\n", endwtime);
	} 

	if(myid == 0) startwtime = MPI_Wtime();
	start_x = 0.0; end_x = 1.0;
	_mpi_dmtrapezoidal_fs_all(&pi, start_x, end_x, f, n, MPI_COMM_WORLD);
	if(myid == 0)
	{
		endwtime = MPI_Wtime() - startwtime;
		printf("BNC: _mpi_dtrapezoidal_fs_all = %e\n", pi);
		printf("Time: %f\n", endwtime);
	} 


#ifdef USE_GMP
	mpf_set_ui(mpf_x, 0UL); mpf_set_ui(mpf_h, 1UL);
	if(myid == 0) startwtime = MPI_Wtime();
	_mpi_mpf_trapezoidal_fs(mpf_pi, mpf_x, mpf_h, mpf_f, n, MPI_COMM_WORLD, MPI_MPF);
	if(myid == 0)
	{
		endwtime = MPI_Wtime() - startwtime;
		printf("BNC: _mpi_mpf_trapezoidal_fs = \n");
		mpf_out_str(stdout, 10, 0, mpf_pi);printf("\n");
		printf("Time: %f\n", endwtime);
	}

	mpf_set_ui(mpf_x, 0UL); mpf_set_ui(mpf_h, 1UL);
	if(myid == 0) startwtime = MPI_Wtime();
	_mpi_mpf_trapezoidal_fs_all(mpf_pi, mpf_x, mpf_h, mpf_f, n, MPI_COMM_WORLD);
	if(myid == 0)
	{
		endwtime = MPI_Wtime() - startwtime;
		printf("BNC: _mpi_mpf_trapezoidal_fs_all = \n");
		mpf_out_str(stdout, 10, 0, mpf_pi);printf("\n");
		printf("Time: %f\n", endwtime);
	}

	mpf_set_ui(mpf_x, 0UL); mpf_set_ui(mpf_h, 1UL);
	if(myid == 0) startwtime = MPI_Wtime();
	_mpi_mpf_mtrapezoidal_fs_all(mpf_pi, mpf_x, mpf_h, mpf_f, n, MPI_COMM_WORLD);
	if(myid == 0)
	{
		endwtime = MPI_Wtime() - startwtime;
		printf("BNC: _mpi_mpf_mtrapezoidal_fs_all = \n");
		mpf_out_str(stdout, 10, 0, mpf_pi);printf("\n");
		printf("Time: %f\n", endwtime);
	}

	mpf_clear(mpf_x); mpf_clear(mpf_h); mpf_clear(mpf_pi);

	free_mpf(&(MPI_MPF));
	free_mpf_op(&(MPI_MPF_SUM));
#endif

	MPI_Finalize();

}
