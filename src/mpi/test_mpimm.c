/********************************************************************************/
/* test_mpimm.c:                                                                */
/* Copyright (C) 2003-2011 Tomonori Kouya                                       */
/*                                                                              */
/* Version 0.0: 2003.10/29                                                      */
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

//#define DIM 5
//#define DIM 64
#define DIM 512
//#define DIM 1024
main(int argc, char *argv[])
{
	DMatrix my_dmat_ans[MPI_GMP_MAXPROCS], my_dmat1[MPI_GMP_MAXPROCS], my_dmat2[MPI_GMP_MAXPROCS];
	DMatrix dmat_ans, dmat1, dmat2;

#ifdef USE_GMP
	MPFMatrix my_mpfmat_ans[MPI_GMP_MAXPROCS], my_mpfmat1[MPI_GMP_MAXPROCS], my_mpfmat2[MPI_GMP_MAXPROCS];
	MPFMatrix mpfmat_ans, mpfmat1, mpfmat2;
	mpf_t tmp;
#endif
	long int d_ddim[MPI_GMP_MAXPROCS];
	long int i, j, local_dim, total_dim;
	int myrank, num_procs;
	double start_wtime, end_wtime;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

/* double */

	/* init */
	local_dim = _mpi_divide_dim(d_ddim, DIM, num_procs);
	total_dim = local_dim * num_procs;

	/* my_dmat[0] ... my_dmat[num_procs - 1] */
	_mpi_init_dmatrix(my_dmat_ans, d_ddim, total_dim, MPI_COMM_WORLD);
	_mpi_init_dmatrix(my_dmat1, d_ddim, total_dim, MPI_COMM_WORLD);
	_mpi_init_dmatrix(my_dmat2, d_ddim, total_dim, MPI_COMM_WORLD);

	if(myrank == 0)
	{
		dmat_ans = init_dmatrix(total_dim, total_dim);
		dmat1 = init_dmatrix(total_dim, total_dim);
		dmat2 = init_dmatrix(total_dim, total_dim);
		for(i = 0; i < DIM; i++)
			for(j = 0; j < DIM; j++)
			{
				sdmij(dmat1, i, j, (double)(i*DIM + j + 1));
				sdmij(dmat2, i, j, (double)(DIM * DIM - (i*DIM + j)));
			}
	}

	_mpi_divide_dmatrix(my_dmat1, d_ddim, dmat1, MPI_COMM_WORLD);
	_mpi_divide_dmatrix(my_dmat2, d_ddim, dmat2, MPI_COMM_WORLD);

	if(myrank == 0)
		start_wtime = MPI_Wtime();

	_mpi_mul_dmatrix(my_dmat_ans, my_dmat1, my_dmat2, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	if(myrank == 0)
		end_wtime = MPI_Wtime();

	_mpi_collect_dmatrix(dmat_ans, d_ddim, my_dmat_ans, MPI_COMM_WORLD);

	/* free */
	_mpi_free_dmatrix(my_dmat_ans, MPI_COMM_WORLD);
	_mpi_free_dmatrix(my_dmat1, MPI_COMM_WORLD);
	_mpi_free_dmatrix(my_dmat2, MPI_COMM_WORLD);

	if(myrank == 0)
	{
//		printf("MPIBNC:\n"); print_dmatrix(dmat_ans);
		printf("MPI_MUL_TIME(Dim:%dx%d, Procs:%d): %f\n", DIM, DIM, num_procs, end_wtime - start_wtime);

//		start_wtime = get_secv();
//		mul_dmatrix(dmat_ans, dmat1, dmat2);
//		end_wtime = get_secv();

//		printf("BNC:\n"); print_dmatrix(dmat_ans);
//		printf("BNC_MUL_TIME(Dim:%dx%d): %f\n", DIM, DIM, end_wtime - start_wtime);

		free_dmatrix(dmat_ans);
		free_dmatrix(dmat1);
		free_dmatrix(dmat2);
	}


/* mpf_t */

#ifdef USE_GMP
#define PREC 128
//#define PREC 256
//#define PREC 512

	_mpi_set_bnc_default_prec(PREC, MPI_COMM_WORLD);
	commit_mpf(&(MPI_MPF), PREC, MPI_COMM_WORLD);
	create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, MPI_COMM_WORLD);

//	goto end;

	/* init */
	local_dim = _mpi_divide_dim(d_ddim, DIM, num_procs);
	total_dim = local_dim * num_procs;

	/* my_mpfmat[0] ... my_mpfmat[num_procs - 1] */
	_mpi_init_mpfmatrix(my_mpfmat_ans, d_ddim, total_dim, MPI_COMM_WORLD);
	_mpi_init_mpfmatrix(my_mpfmat1, d_ddim, total_dim, MPI_COMM_WORLD);
	_mpi_init_mpfmatrix(my_mpfmat2, d_ddim, total_dim, MPI_COMM_WORLD);

	if(myrank == 0)
	{
		mpfmat_ans = init_mpfmatrix(total_dim, total_dim);
		mpfmat1 = init_mpfmatrix(total_dim, total_dim);
		mpfmat2 = init_mpfmatrix(total_dim, total_dim);
		for(i = 0; i < DIM; i++)
			for(j = 0; j < DIM; j++)

			{
				set_mpfmatrix_ij_ui(mpfmat1, i, j, (unsigned long)(i*DIM + j + 1));
				set_mpfmatrix_ij_ui(mpfmat2, i, j, (unsigned long)(DIM * DIM - (i*DIM + j)));
			}
	}

	_mpi_divide_mpfmatrix(my_mpfmat1, d_ddim, mpfmat1, MPI_COMM_WORLD);
	_mpi_divide_mpfmatrix(my_mpfmat2, d_ddim, mpfmat2, MPI_COMM_WORLD);

	if(myrank == 0)
		start_wtime = MPI_Wtime();

	_mpi_mul_mpfmatrix(my_mpfmat_ans, my_mpfmat1, my_mpfmat2, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	if(myrank == 0)
		end_wtime = MPI_Wtime();

	_mpi_collect_mpfmatrix(mpfmat_ans, d_ddim, my_mpfmat_ans, MPI_COMM_WORLD);

	/* free */
	_mpi_free_mpfmatrix(my_mpfmat_ans, MPI_COMM_WORLD);
	_mpi_free_mpfmatrix(my_mpfmat1, MPI_COMM_WORLD);
	_mpi_free_mpfmatrix(my_mpfmat2, MPI_COMM_WORLD);

	if(myrank == 0)
	{
//		printf("MPIBNC:\n"); print_mpfmatrix(mpfmat_ans);
		printf("MPI_MUL_TIME(Dim:%dx%d, Procs:%d): %f\n", DIM, DIM, num_procs, end_wtime - start_wtime);

//		start_wtime = get_secv();
//		mul_mpfmatrix(mpfmat_ans, mpfmat1, mpfmat2);
//		end_wtime = get_secv();

//		printf("BNC:\n"); print_mpfmatrix(mpfmat_ans);
//		printf("BNC_MUL_TIME(Dim:%dx%d): %f\n", DIM, DIM, end_wtime - start_wtime);


		free_mpfmatrix(mpfmat_ans);
		free_mpfmatrix(mpfmat1);
		free_mpfmatrix(mpfmat2);
	}

end:
	free_mpf(&(MPI_MPF));
#endif
	MPI_Finalize();

}
