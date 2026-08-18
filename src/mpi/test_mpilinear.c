/********************************************************************************/
/* test_mpilinear.c:                                                            */
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
#include <stdio.h>
#include <math.h>

#include "mpi.h"
#include "bnc.h"

#ifdef USE_GMP
#include "mpi_gmp.h"
#endif

#include "mpi_bnc.h"

#define DIM 8

main(int argc, char *argv[])
{
	DVector my_dvec, my_dvec1, my_dvec2, main_dvec, main_dvec1, main_dvec_tmp;
	DMatrix my_dmat[MPI_GMP_MAXPROCS], main_dmat, main_dmat1;

#ifdef USE_GMP
	MPFVector my_mpfvec, my_mpfvec1, my_mpfvec2, main_mpfvec, main_mpfvec1, main_mpfvec_tmp;
	MPFMatrix my_mpfmat[MPI_GMP_MAXPROCS], main_mpfmat, main_mpfmat1;
	mpf_t tmp;
#endif
	long int d_ddim[MPI_GMP_MAXPROCS];
	long int i, j, local_dim;
	int myrank, num_procs;
	double start_wtime, end_wtime;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

/* double */

	goto MPF;

	/* my_dvec on Pi */
	my_dvec = _mpi_init_dvector(d_ddim, DIM, MPI_COMM_WORLD);
	my_dvec1 = _mpi_init_dvector(d_ddim, DIM, MPI_COMM_WORLD);
	my_dvec2 = _mpi_init_dvector(d_ddim, DIM, MPI_COMM_WORLD);

	/* my_dmat[0] ... my_dmat[num_procs - 1] */
	_mpi_init_dmatrix(my_dmat, d_ddim, DIM, MPI_COMM_WORLD);

	/* init */
	local_dim = _mpi_divide_dim(d_ddim, DIM, num_procs);
	if(myrank == 0)
	{
		main_dvec = init_dvector(local_dim * num_procs);
		main_dvec1 = init_dvector(local_dim * num_procs);
		for(i = 0; i < DIM; i++)
		{
			sdvi(main_dvec, i, (double)(i + 1));
			sdvi(main_dvec1, i, (double)(DIM - i));
		}
		print_dvector(main_dvec);
		printf("--\n");
		main_dmat = init_dmatrix(local_dim * num_procs, local_dim * num_procs);
		main_dmat1 = init_dmatrix(local_dim * num_procs, local_dim * num_procs);
		for(i = 0; i < DIM; i++)
			for(j = 0; j < DIM; j++)
				sdmij(main_dmat, i, j, (double)(i + j - 1));
//		print_dmatrix(main_dmat);
		printf("--\n");
	}
	main_dvec_tmp = init_dvector(local_dim * num_procs);

//	printf("rank %d:\n", myrank); print_dvector(my_dvec);

	_mpi_divide_dvector(my_dvec, d_ddim, main_dvec, MPI_COMM_WORLD);
	_mpi_divide_dvector(my_dvec1, d_ddim, main_dvec1, MPI_COMM_WORLD);

	_mpi_divide_dmatrix(my_dmat, d_ddim, main_dmat, MPI_COMM_WORLD);

	printf("(vec)rank %d:\n", myrank); print_dvector(my_dvec);
	printf("(mat)rank %d:\n", myrank);
	for(i = 0; i < num_procs; i++)
	{
//		printf("%d:", i); print_dmatrix(my_dmat[i]);
	}

	_mpi_collect_dvector(main_dvec1, d_ddim, my_dvec, MPI_COMM_WORLD);
	_mpi_collect_dmatrix(main_dmat1, d_ddim, my_dmat, MPI_COMM_WORLD);

//	printf("(rank %d) ip = %e\n", myrank, _mpi_ip_dvector(my_dvec, my_dvec1, MPI_COMM_WORLD));

	MPI_Barrier(MPI_COMM_WORLD);
	if(myrank == 0)
		start_wtime = MPI_Wtime();

	_mpi_mul_dmatrix_dvec(my_dvec2, my_dmat, my_dvec, main_dvec_tmp, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	if(myrank == 0)
		end_wtime = MPI_Wtime();

//	printf("(rank %d) ", myrank); print_dvector(my_dvec2);
	_mpi_collect_dvector(main_dvec_tmp, d_ddim, my_dvec2, MPI_COMM_WORLD);

	_mpi_free_dvector(my_dvec);
	_mpi_free_dvector(my_dvec1);
	_mpi_free_dvector(my_dvec2);

	_mpi_free_dmatrix(my_dmat, MPI_COMM_WORLD);

	/* free */
	if(myrank == 0)
	{
		printf("main_dvec: "); print_dvector(main_dvec1);
//		printf("main_dmat: "); print_dmatrix(main_dmat1);
		printf("main_dvec_tmp: "); print_dvector(main_dvec_tmp);

		printf("MPI_MUL_TIME: %f\n", end_wtime - start_wtime);
		start_wtime = get_secv();
		mul_dmatrix_dvec(main_dvec_tmp, main_dmat, main_dvec);
		end_wtime = get_secv();
		printf("BNC_MUL_TIME: %f\n", end_wtime - start_wtime);
//		printf("BNC:"); print_dvector(main_dvec_tmp);


		free_dvector(main_dvec);
		free_dvector(main_dvec1);
		free_dvector(main_dvec_tmp);
		free_dmatrix(main_dmat);
		free_dmatrix(main_dmat1);
	}

	goto end;

/* mpf_t */
MPF:

#ifdef USE_GMP
#define PREC 128

//	set_bnc_default_prec(PREC);
	_mpi_set_bnc_default_prec(PREC, MPI_COMM_WORLD);
	commit_mpf(&(MPI_MPF), PREC, MPI_COMM_WORLD);
	create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, MPI_COMM_WORLD);
//	goto end;

	/* my_dvec on Pi */
	my_mpfvec = _mpi_init_mpfvector(d_ddim, DIM, MPI_COMM_WORLD);
	my_mpfvec1 = _mpi_init_mpfvector(d_ddim, DIM, MPI_COMM_WORLD);
	my_mpfvec2 = _mpi_init_mpfvector(d_ddim, DIM, MPI_COMM_WORLD);

	/* my_dmat[0] ... my_dmat[num_procs - 1] */
	_mpi_init_mpfmatrix(my_mpfmat, d_ddim, DIM, MPI_COMM_WORLD);

	/* init */
	local_dim = _mpi_divide_dim(d_ddim, DIM, num_procs);
	if(myrank == 0)
	{
		main_mpfvec = init_mpfvector(local_dim * num_procs);
		main_mpfvec1 = init_mpfvector(local_dim * num_procs);
		for(i = 0; i < DIM; i++)
		{
			smpfviui(main_mpfvec, i, (unsigned long)(i + 1));
			smpfviui(main_mpfvec1, i, (unsigned long)(DIM - i));
		}
		print_mpfvector(main_mpfvec);
		printf("--\n");
		main_mpfmat = init_mpfmatrix(local_dim * num_procs, local_dim * num_procs);
		main_mpfmat1 = init_mpfmatrix(local_dim * num_procs, local_dim * num_procs);
		for(i = 0; i < DIM; i++)
			for(j = 0; j < DIM; j++)
				smpfmijui(main_mpfmat, i, j, (unsigned long)(i + j - 1));
//		print_mpfmatrix(main_mpfmat);
		printf("--\n");
	}
	main_mpfvec_tmp = init_mpfvector(local_dim * num_procs);

//	printf("rank %d:\n", myrank); print_dvector(my_dvec);

	_mpi_divide_mpfvector(my_mpfvec, d_ddim, main_mpfvec, MPI_COMM_WORLD);
	_mpi_divide_mpfvector(my_mpfvec1, d_ddim, main_mpfvec1, MPI_COMM_WORLD);
	_mpi_divide_mpfmatrix(my_mpfmat, d_ddim, main_mpfmat, MPI_COMM_WORLD);


	printf("(vec)rank %d:\n", myrank); print_mpfvector(my_mpfvec);
	printf("(mat)rank %d:\n", myrank);

	for(i = 0; i < num_procs; i++)
	{
//		printf("%d:", i); print_mpfmatrix(my_mpfmat[i]);
	}

	_mpi_collect_mpfvector(main_mpfvec1, d_ddim, my_mpfvec, MPI_COMM_WORLD);
	_mpi_collect_mpfmatrix(main_mpfmat1, d_ddim, my_mpfmat, MPI_COMM_WORLD);
	mpf_init(tmp);

	_mpi_ip_mpfvector(tmp, my_mpfvec, my_mpfvec1, MPI_COMM_WORLD);
	printf("(rank %d) ip = %e\n", myrank, mpf_get_d(tmp));

	MPI_Barrier(MPI_COMM_WORLD);
	if(myrank == 0)
		start_wtime = MPI_Wtime();

	_mpi_mul_mpfmatrix_mpfvec(my_mpfvec2, my_mpfmat, my_mpfvec, main_mpfvec_tmp, MPI_COMM_WORLD);
	MPI_Barrier(MPI_COMM_WORLD);
	if(myrank == 0)
		end_wtime = MPI_Wtime();

	printf("(rank %d) ", myrank); print_mpfvector(my_mpfvec2);
//	goto end;
	_mpi_collect_mpfvector(main_mpfvec_tmp, d_ddim, my_mpfvec2, MPI_COMM_WORLD);

/*	_mpi_free_mpfvector(my_mpfvec);
	_mpi_free_mpfvector(my_mpfvec1);
	_mpi_free_mpfvector(my_mpfvec2);

	_mpi_free_mpfmatrix(my_mpfmat, MPI_COMM_WORLD);
*/
	/* free */
	if(myrank == 0)
	{
		printf("main_mpfvec: "); print_mpfvector(main_mpfvec1);
//		printf("main_mpfmat: "); print_mpfmatrix(main_mpfmat1);
		printf("main_mpfvec_tmp: "); print_mpfvector(main_mpfvec_tmp);

		printf("MPI_MUL_TIME: %f\n", end_wtime - start_wtime);
		start_wtime = get_secv();
		mul_mpfmatrix_mpfvec(main_mpfvec_tmp, main_mpfmat, main_mpfvec);
		end_wtime = get_secv();
		printf("BNC_MUL_TIME: %f\n", end_wtime - start_wtime);
//		printf("BNC:"); print_mpfvector(main_mpfvec_tmp);


		free_mpfvector(main_mpfvec);
		free_mpfvector(main_mpfvec1);
		free_mpfvector(main_mpfvec_tmp);
		free_mpfmatrix(main_mpfmat);
		free_mpfmatrix(main_mpfmat1);
	}


	free_mpf(&(MPI_MPF));
#endif
end:
	MPI_Finalize();

}
