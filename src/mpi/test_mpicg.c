/********************************************************************************/
/* test_mpicg.c: Test Program for DCG, MPFCG                                    */
/* Copyright (C) 2003-2011 Tomonori Kouya                                       */
/*                                                                              */
/* Version 0.0: 2003.08/21                                                      */
/* Version 0.1: 2003.10/04                                                      */
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
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/times.h>

#include "bnc.h"

#ifdef USE_GMP
#include "mpi_gmp.h"
#endif

#include "mpi_bnc.h"

#define DIM 100
//#define DIM 1024
//#define DIM 512

void get_dproblem(DMatrix a, DVector b, DVector ans, long dim)
{
	long int i, j, k;
	double tmp;

	/* Frank Matrix */
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			if(i < j)
				set_dmatrix_ij(a, i, j, (double)(dim - j));
			else
				set_dmatrix_ij(a, i, j, (double)(dim - i));
		}
	}

	/* Answer */
	for(i = 0; i < dim; i++)
		set_dvector_i(ans, i, (double)i);

	/* Make constant vector */
	mul_dmatrix_dvec(b, a, ans);
}

#ifdef USE_GMP
void get_mpfproblem(MPFMatrix a, MPFVector b, MPFVector ans, long dim)
{
	long int i, j, k;
	mpf_t tmp;

	mpf_init2(tmp, prec_mpfvector(ans));

	/* Frank Matrix */
	for(i = 0; i < dim; i++)
	{
		for(j = 0; j < dim; j++)
		{
			if(i < j)
			{
				mpf_set_si(tmp, dim - j);
				set_mpfmatrix_ij(a, i, j, tmp);
			}
			else
			{
				mpf_set_si(tmp, dim - i);
				set_mpfmatrix_ij(a, i, j, tmp);
			}
		}
	}

	/* Answer */
	for(i = 0; i < dim; i++)
	{
		mpf_set_si(tmp, i);
		set_mpfvector_i(ans, i, tmp);
	}

	/* Make constant vector */
	mul_mpfmatrix_mpfvec(b, a, ans);
}
#endif

main(int argc, char *argv[])
{
	int myid, numprocs;
	int  namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	long int d_ddim[MPI_GMP_MAXPROCS], local_dim;
	DMatrix da, my_da[MPI_GMP_MAXPROCS];
	DVector db, dx, dans, my_db, my_dx, my_dans;
	double start, ftime, dtime, startwtime[2], endwtime[2];

#ifdef USE_GMP
	MPFMatrix mpfa, my_mpfa[MPI_GMP_MAXPROCS];
	MPFVector mpfb, mpfx, mpfans;
	MPFVector my_mpfb, my_mpfx, my_mpfans;
	mpf_t reps, aeps;
	MPFMatrix mpfa2;
	MPFVector mpfb2, mpfx2, mpfans2;
	mpf_t reps2, aeps2;
	MPFMatrix mpfa3;
	MPFVector mpfb3, mpfx3, mpfans3;
	mpf_t reps3, aeps3;
	long int itimes_mpf, itimes_mpf2, itimes_mpf3, itimes_mpfm;
	double mpftime[3];
#endif
	long int itimes_f, itimes_d, itimes_dm;
	long int i, j;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	MPI_Get_processor_name(processor_name,&namelen);

	fprintf(stdout,"Process %d of %d on %s\n",
		myid, numprocs, processor_name);

	goto MPF;
/* Double */
	/* divide problem */
	local_dim = _mpi_divide_dim(d_ddim, DIM, numprocs);
	if(myid == 0)
	{
		/* initialize */
		da = init_dmatrix(DIM, DIM);
		db = init_dvector(DIM);
		dx = init_dvector(DIM);
		dans = init_dvector(DIM);

		/* get problem */
		get_dproblem(da, db, dans, DIM);
	
//		print_dmatrix(da);
	}


	my_db = _mpi_init_dvector(d_ddim, DIM, MPI_COMM_WORLD);
	my_dx = _mpi_init_dvector(d_ddim, DIM, MPI_COMM_WORLD);
	_mpi_init_dmatrix(my_da, d_ddim, DIM, MPI_COMM_WORLD);

	_mpi_divide_dvector(my_db, d_ddim, db, MPI_COMM_WORLD);
	_mpi_divide_dmatrix(my_da, d_ddim, da, MPI_COMM_WORLD);

	if(myid == 0) startwtime[0] = MPI_Wtime();
	itimes_dm = _mpi_DCG(my_dx, my_da, my_db, 1.0e-13, 1.0e-99, DIM * 5, DIM, MPI_COMM_WORLD);
	if(myid == 0) endwtime[0] = MPI_Wtime() - startwtime[0];
//	for(i = 0; i < local_dim; i++)
//		printf("%5ld %25.17e\n", i, get_dvector_i(my_dx, i));
	_mpi_collect_dvector(dx, d_ddim, my_dx, MPI_COMM_WORLD);
	if(myid == 0) print_dvector(dx);

	if(myid == 0)
	{

		/* run DCG */
		start = get_secv();
		itimes_d = DCG(dx, da, db, 1.0e-13, 1.0e-99, DIM * 5);
		dtime = get_secv() - start;

		/* print */
		for(i = 0; i < DIM; i++)
			printf("%5ld %25.17e %25.17e\n", i, get_dvector_i(dx, i), get_dvector_i(dans, i));
		
		/* end */
		free_dmatrix(da);
		free_dvector(db);
		free_dvector(dx);
		free_dvector(dans);
	}

//	goto end;

MPF:
#ifdef USE_GMP
/* MPF */
#define MPF_PREC 128
//#define MPF_PREC 1024
//#define MPF_PREC 512
//#define MPF_PREC 256
//	set_bnc_default_prec(MPF_PREC);
//	mpf_set_default_prec(MPF_PREC);
	_mpi_set_bnc_default_prec(MPF_PREC, MPI_COMM_WORLD);
	commit_mpf(&(MPI_MPF), MPF_PREC, MPI_COMM_WORLD);
	create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, MPI_COMM_WORLD);

	/* initialize */
	mpf_init(reps);
	mpf_init2(reps2, 256);
	mpf_init2(reps3, 512);
	mpf_init(aeps);
	mpf_init2(aeps2, 256);
	mpf_init2(aeps3, 512);

	/* divide problem */
	local_dim = _mpi_divide_dim(d_ddim, DIM, numprocs);
	if(myid == 0)
	{
		mpfa  = init_mpfmatrix(DIM, DIM);
		mpfa2 = init2_mpfmatrix(DIM, DIM, 256);
		mpfa3 = init2_mpfmatrix(DIM, DIM, 512);
		mpfb  = init_mpfvector(DIM);
		mpfb2 = init2_mpfvector(DIM, 256);
		mpfb3 = init2_mpfvector(DIM, 512);
		mpfx  = init_mpfvector(DIM);
		mpfx2 = init2_mpfvector(DIM, 256);
		mpfx3 = init2_mpfvector(DIM, 512);
		mpfans  = init_mpfvector(DIM);
		mpfans2 = init2_mpfvector(DIM, 256);
		mpfans3 = init2_mpfvector(DIM, 512);

		/* get problem */
		get_mpfproblem(mpfa, mpfb, mpfans, DIM);
		get_mpfproblem(mpfa2, mpfb2, mpfans2, DIM);
		get_mpfproblem(mpfa3, mpfb3, mpfans3, DIM);

		// print_mpfmatrix(mpfa);
		// print_mpfmatrix(mpfa2);
		// print_mpfmatrix(mpfa3);
	}

	/* run MPFFCG */
	mpf_set_d(reps, 1.0e-20);
	mpf_set_d(reps2, 1.0e-20);
	mpf_set_d(reps3, 1.0e-20);
	mpf_set_d(aeps, 1.0e-50);
	mpf_set_d(aeps2, 1.0e-50);
	mpf_set_d(aeps3, 1.0e-50);

	my_mpfb = _mpi_init_mpfvector(d_ddim, DIM, MPI_COMM_WORLD);
	my_mpfx = _mpi_init_mpfvector(d_ddim, DIM, MPI_COMM_WORLD);
	_mpi_init_mpfmatrix(my_mpfa, d_ddim, DIM, MPI_COMM_WORLD);

	_mpi_divide_mpfvector(my_mpfb, d_ddim, mpfb, MPI_COMM_WORLD);
	_mpi_divide_mpfmatrix(my_mpfa, d_ddim, mpfa, MPI_COMM_WORLD);

	if(myid == 0) startwtime[1] = MPI_Wtime();
	itimes_mpfm = _mpi_MPFCG(my_mpfx, my_mpfa, my_mpfb, reps, aeps, DIM * 5, DIM, MPI_COMM_WORLD);
	if(myid == 0) endwtime[1] = MPI_Wtime() - startwtime[1];

/*	for(i = 0; i < local_dim; i++)
	{
		printf("%5ld ", i);
		mpf_out_str(stdout, 10, 0, get_mpfvector_i(my_mpfx, i));
		printf("\n");
	}
*/
	_mpi_collect_mpfvector(mpfx, d_ddim, my_mpfx, MPI_COMM_WORLD);
	if(myid == 0) print_mpfvector(mpfx);

	if(myid == 0)
	{
		start = get_secv();
		itimes_mpf  = MPFCG(mpfx, mpfa, mpfb, reps, aeps, DIM * 5);
		mpftime[0] = get_secv() - start;
/*
		start = get_secv();
		itimes_mpf2 = MPFCG(mpfx2, mpfa2, mpfb2, reps2, aeps2, DIM * 5);
		mpftime[1] = get_secv() - start;

		start = get_secv();
		itimes_mpf3 = MPFCG(mpfx3, mpfa3, mpfb3, reps3, aeps3, DIM * 5);
		mpftime[2] = get_secv() - start;
*/
		/* print */
		for(i = 0; i < DIM; i++)
		{
			printf("%5ld ", i);
			mpf_out_str(stdout, 10, 0, get_mpfvector_i(mpfx, i));
			printf(" ");
			mpf_out_str(stdout, 10, 0, get_mpfvector_i(mpfx2, i));
			printf(" ");
			mpf_out_str(stdout, 10, 0, get_mpfvector_i(mpfx3, i));
			printf(" ");
			mpf_out_str(stdout, 10, 0, get_mpfvector_i(mpfans, i));
			printf("\n");
		}
		free_mpfmatrix(mpfa);
		free_mpfmatrix(mpfa2);
		free_mpfmatrix(mpfa3);
		free_mpfvector(mpfb);
		free_mpfvector(mpfb2);
		free_mpfvector(mpfb3);
		free_mpfvector(mpfx);
		free_mpfvector(mpfx2);
		free_mpfvector(mpfx3);
		free_mpfvector(mpfans);
		free_mpfvector(mpfans2);
		free_mpfvector(mpfans3);
	}

	/* end */
	mpf_clear(reps); mpf_clear(aeps);
	mpf_clear(reps2); mpf_clear(aeps2);
	mpf_clear(reps3); mpf_clear(aeps3);

	free_mpf(&(MPI_MPF));
	free_mpf_op(&(MPI_MPF_SUM));
#endif

end:
	MPI_Finalize();

	if(myid == 0){
	/* print itimes */
	printf("Iterative Times\n");
	printf("double(MPI)    : %ld(%f)\n", itimes_dm, endwtime[0]);
	printf("double         : %ld(%f)\n", itimes_d, dtime);
#ifdef USE_GMP
	printf("mpf_t(MPI, %d) : %ld(%f)\n", MPF_PREC, itimes_mpfm, endwtime[1]);
	printf("mpf_t(128)     : %ld(%f)\n", itimes_mpf, mpftime[0]);
	printf("mpf_t(256)     : %ld(%f)\n", itimes_mpf2, mpftime[1]);
	printf("mpf_t(512)     : %ld(%f)\n", itimes_mpf3, mpftime[2]);
#endif
	}
}

