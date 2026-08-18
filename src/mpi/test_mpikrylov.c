/********************************************************************************/
/* test_mpicg.c: Test Program for DCG, MPFCG                                    */
/* Copyright (C) 2003-2011 Tomonori Kouya                                       */
/*                                                                              */
/* Version 0.0: 2004.11/11(Thu)                                                 */
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
//#include <sys/times.h>

#include "bnc.h"

#ifdef USE_GMP
#include "mpi_gmp.h"
#endif

#include "mpi_bnc.h"

void get_dproblem(DMatrix a, DVector b, DVector ans, long int dim)
{
	long int i, j, k;
	double tmp;

	/* Lotkin Matrix */
	//lotkin_dmatrix(a);

	/* Toeplitz Matrix */
	toeplitz_dmatrix(a, 1.7, dim);

	/* Answer */
	for(i = 0; i < ans->dim; i++)
		set_dvector_i(ans, i, (double)i);

	/* Make constant vector */
	mul_dmatrix_dvec(b, a, ans);
}

#ifdef USE_GMP
void get_mpfproblem(MPFMatrix a, MPFVector b, MPFVector ans, long int dim)
{
	long int i, j, k;
	mpf_t tmp;

	mpf_init2(tmp, prec_mpfvector(ans));

	/* Lotkin Matrix */
	//lotkin_mpfmatrix(a);

	/* Toeplitz Matrix */
	mpf_set_str(tmp, "1.7", 10);
	toeplitz_mpfmatrix(a, tmp, dim);

	/* Answer */
	for(i = 0; i < ans->dim; i++)
	{
		mpf_set_si(tmp, i);
		set_mpfvector_i(ans, i, tmp);
	}

	/* Make constant vector */
	mul_mpfmatrix_mpfvec(b, a, ans);
}
#endif

//#define DIM 200
//#define DIM 1100
//#define DIM 100
//#define DIM 10
#define DIM 1024
//#define DIM 512

main(int argc, char *argv[])
{
	int myid, numprocs;
	int  namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	long int d_ddim[MPI_GMP_MAXPROCS], local_dim, total_dim;
	DMatrix da, my_da[MPI_GMP_MAXPROCS], dat, my_dat[MPI_GMP_MAXPROCS];
	DVector db, dx, dans, my_db, my_dx, my_dans;
	double start, ftime, dtime, startwtime[2], endwtime[2];

#ifdef USE_GMP
	MPFMatrix mpfa, my_mpfa[MPI_GMP_MAXPROCS], mpfat, my_mpfat[MPI_GMP_MAXPROCS];
	MPFVector mpfb, mpfx, mpfans;
	MPFVector my_mpfb, my_mpfx, my_mpfans;
	mpf_t reps, aeps;
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

	goto mpf;
/* Double */
	/* divide problem */
	local_dim = _mpi_divide_dim(d_ddim, DIM, numprocs);
	total_dim = local_dim * numprocs;
	if(myid == 0)
	{
		/* initialize */
		da = init_dmatrix(total_dim, total_dim);
		dat= init_dmatrix(total_dim, total_dim);
		db = init_dvector(total_dim);
		dx = init_dvector(total_dim);
		dans = init_dvector(total_dim);

		/* get problem */
		get_dproblem(da, db, dans, DIM);
		transpose_dmatrix(dat, da);
	
//		print_dmatrix(da);
	}

	my_db = _mpi_init_dvector(d_ddim, total_dim, MPI_COMM_WORLD);
	my_dx = _mpi_init_dvector(d_ddim, total_dim, MPI_COMM_WORLD);
	_mpi_init_dmatrix(my_da, d_ddim, total_dim, MPI_COMM_WORLD);
	_mpi_init_dmatrix(my_dat, d_ddim, total_dim, MPI_COMM_WORLD);

	_mpi_divide_dvector(my_db, d_ddim, db, MPI_COMM_WORLD);
	_mpi_divide_dmatrix(my_da, d_ddim, da, MPI_COMM_WORLD);
	_mpi_divide_dmatrix(my_dat, d_ddim, dat, MPI_COMM_WORLD);

	if(myid == 0) startwtime[0] = MPI_Wtime();
//	itimes_dm = _mpi_DBiCG(my_dx, my_da, my_dat, my_db, 1.0e-13, 1.0e-99, DIM * 5, DIM, MPI_COMM_WORLD);
	itimes_dm = _mpi_DCGS(my_dx, my_da, my_db, 1.0e-13, 1.0e-99, DIM * 5, DIM, MPI_COMM_WORLD);
//	itimes_dm = _mpi_DBiCGSTAB(my_dx, my_da, my_db, 1.0e-13, 1.0e-99, DIM * 5, DIM, MPI_COMM_WORLD);
//	itimes_dm = _mpi_DGPBiCG(my_dx, my_da, my_db, 1.0e-13, 1.0e-99, DIM * 5, DIM, MPI_COMM_WORLD);
	if(myid == 0) endwtime[0] = MPI_Wtime() - startwtime[0];
//	for(i = 0; i < local_dim; i++)
//		printf("%5ld %25.17e\n", i, get_dvector_i(my_dx, i));
	_mpi_collect_dvector(dx, d_ddim, my_dx, MPI_COMM_WORLD);
	if(myid == 0) print_dvector(dx);

	if(myid == 0)
	{

		/* run DCG */
		start = get_secv();
//		itimes_d = DBiCG(dx, da, db, 1.0e-13, 1.0e-99, DIM * 5);
		itimes_d = DCGS(dx, da, db, 1.0e-13, 1.0e-99, DIM * 5);
//		itimes_d = DBiCGSTAB(dx, da, db, 1.0e-13, 1.0e-99, DIM * 5);
//		itimes_d = DGPBiCG(dx, da, db, 1.0e-13, 1.0e-99, DIM * 5);
		dtime = get_secv() - start;

		/* print */
		for(i = 0; i < DIM; i++)
			printf("%5ld %25.17e %25.17e\n", i, get_dvector_i(dx, i), get_dvector_i(dans, i));
		
		/* end */
		free_dmatrix(da);
		free_dmatrix(dat);
		free_dvector(db);
		free_dvector(dx);
		free_dvector(dans);
	}

//	goto end;

mpf:

#ifdef USE_GMP
/* MPF */
//#define MPF_PREC 8192
//#define MPF_PREC 4096
#define MPF_PREC 2048
//#define MPF_PREC 1024
//#define MPF_PREC 512
//#define MPF_PREC 256
//#define MPF_PREC 128
//#define MPF_PREC_DEC 50
//#define MPF_PREC_DEC 100
//#define MPF_PREC_DEC 500
//#define MPF_PREC_DEC 1000
//#define MPF_PREC_DEC 10000
//#define MPF_PREC ((unsigned long)ceil(MPF_PREC_DEC/log10(2.0)))
_mpi_set_bnc_default_prec(MPF_PREC, MPI_COMM_WORLD);
commit_mpf(&(MPI_MPF), MPF_PREC, MPI_COMM_WORLD);
//	_mpi_set_bnc_default_prec(MPF_PREC-1, MPI_COMM_WORLD);
//	commit_mpf(&(MPI_MPF), MPF_PREC-1, MPI_COMM_WORLD);
	create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, MPI_COMM_WORLD);

	/* initialize */
	mpf_init(reps);
	mpf_init(aeps);

	/* divide problem */
	local_dim = _mpi_divide_dim(d_ddim, DIM, numprocs);
	total_dim = local_dim * numprocs;
	if(myid == 0)
	{
		mpfa  = init_mpfmatrix(total_dim, total_dim);
		mpfat = init_mpfmatrix(total_dim, total_dim);
		mpfb  = init_mpfvector(total_dim);
		mpfx  = init_mpfvector(total_dim);
		mpfans  = init_mpfvector(total_dim);

		/* get problem */
		get_mpfproblem(mpfa, mpfb, mpfans, DIM);

		transpose_mpfmatrix(mpfat, mpfa);

		// print_mpfmatrix(mpfa);
		// print_mpfmatrix(mpfa2);
		// print_mpfmatrix(mpfa3);
	}

	/* run MPFFCG */
	mpf_set_d(reps, 1.0e-20);
	mpf_set_d(aeps, 1.0e-50);

	my_mpfb = _mpi_init_mpfvector(d_ddim, total_dim, MPI_COMM_WORLD);
	my_mpfx = _mpi_init_mpfvector(d_ddim, total_dim, MPI_COMM_WORLD);
	_mpi_init_mpfmatrix(my_mpfa, d_ddim, total_dim, MPI_COMM_WORLD);
	_mpi_init_mpfmatrix(my_mpfat, d_ddim, total_dim, MPI_COMM_WORLD);

	_mpi_divide_mpfvector(my_mpfb, d_ddim, mpfb, MPI_COMM_WORLD);
	_mpi_divide_mpfmatrix(my_mpfa, d_ddim, mpfa, MPI_COMM_WORLD);
	_mpi_divide_mpfmatrix(my_mpfat, d_ddim, mpfat, MPI_COMM_WORLD);

	if(myid == 0) startwtime[1] = MPI_Wtime();
	//itimes_mpfm = _mpi_MPFBiCG(my_mpfx, my_mpfa, my_mpfat, my_mpfb, reps, aeps, DIM * 5, DIM, MPI_COMM_WORLD);
	//itimes_mpfm = _mpi_MPFCGS(my_mpfx, my_mpfa, my_mpfb, reps, aeps, DIM * 5, DIM, MPI_COMM_WORLD);
	//itimes_mpfm = _mpi_MPFBiCGSTAB(my_mpfx, my_mpfa, my_mpfb, reps, aeps, DIM * 5, DIM, MPI_COMM_WORLD);
	itimes_mpfm = _mpi_MPFGPBiCG(my_mpfx, my_mpfa, my_mpfb, reps, aeps, DIM * 5, DIM, MPI_COMM_WORLD);
	if(myid == 0) endwtime[1] = MPI_Wtime() - startwtime[1];

/*	for(i = 0; i < local_dim; i++)
	{
		printf("%5ld ", i);
		mpf_out_str(stdout, 10, 0, get_mpfvector_i(my_mpfx, i));
		printf("\n");
	}
*/
	_mpi_collect_mpfvector(mpfx, d_ddim, my_mpfx, MPI_COMM_WORLD);
	if(myid == 0)
	{	//print_mpfvector(mpfx);
		printf("mpfxi[0]    : "); mpf_out_str(stdout, 10, 0, gmpfvi(mpfx, 0)); printf("\n");
		printf("mpfxi[DIM-1]: "); mpf_out_str(stdout, 10, 0, gmpfvi(mpfx, DIM - 1)); printf("\n");
	}


	if(myid == 0)
	{
		start = get_secv();
		//itimes_mpf  = MPFBiCG(mpfx, mpfa, mpfb, reps, aeps, DIM * 5);
		//itimes_mpf  = MPFCGS(mpfx, mpfa, mpfb, reps, aeps, DIM * 5);
		//itimes_mpf  = MPFBiCGSTAB(mpfx, mpfa, mpfb, reps, aeps, DIM * 5);
		//itimes_mpf  = MPFGPBiCG(mpfx, mpfa, mpfb, reps, aeps, DIM * 5);
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
/*		for(i = 0; i < DIM; i++)
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
*/		free_mpfmatrix(mpfa);
		free_mpfmatrix(mpfat);
		free_mpfvector(mpfb);
		free_mpfvector(mpfx);
		free_mpfvector(mpfans);
	}

	/* end */
	mpf_clear(reps); mpf_clear(aeps);

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
//	printf("mpf_t(%d)      : %ld(%f)\n", MPF_PREC, itimes_mpf, mpftime[0]);
#endif
	exit(0);
	}
}
