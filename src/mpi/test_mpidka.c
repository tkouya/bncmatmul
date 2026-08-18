/********************************************************************************/
/* test_mpidka.c: Durand-Kerner-Aberth Methods                                  */
/* Copyright (C) 2003 Tomonori Kouya                                            */
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
#include <math.h>

#include "mpi.h"
#include "bnc.h"

#ifdef USE_GMP
#include "mpi_gmp.h"
#endif

#include "mpi_bnc.h"

#define MAX_LENGTH 1024

main(int argc, char *argv[])
{
	long int i, dtimes, mpftimes;
	FILE * dcoef, * mpfcoef;

	long int local_dim, dd_dim[MPI_GMP_MAXPROCS];
	CDArray cdans, cdinit, local_cdans, local_cdinit;
	DPoly df;
	double dabs_eps, drel_eps;
	double startwtime[2], endwtime[2];
#ifdef USE_GMP
	CMPFArray cmpfans, cmpfinit, local_cmpfans, local_cmpfinit;
	MPFPoly mpff;
	mpf_t mpfabs_eps, mpfrel_eps;
#endif
	int myrank, num_procs;

/* for MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

//	goto mpf;

/* double */

	/* init */
	dabs_eps = 1.0e-100;
	drel_eps = 1.0e-7;
	df = init_dpoly(MAX_LENGTH);

#define DEG 20

	local_dim = _mpi_divide_dim(dd_dim, DEG, num_procs);
	if(myrank == 0)
	{
//		for(i = 0; i < DEG; i++)
//			printf("%d d_dim[%d]: %d\n", local_dim, i, dd_dim[i]);
	}
	local_cdans = init_cdarray(local_dim);
	local_cdinit = init_cdarray(local_dim);

	cdans = init_cdarray(DEG);
	cdinit = init_cdarray(DEG);
	if(myrank == 0)
	{
		dcoef = fopen("test_mpidkacoef.dat", "r");
		fread_dpolycoef(dcoef, df, DEG);
		fclose(dcoef);
	}
//	exit(0);

/*
// ff = (x-1)(x-2)(x-3)(x-4)(x-5) 
	set_dpoly_i(df, 0, (double)-120);
	set_dpoly_i(df, 1, (double)274);
	set_dpoly_i(df, 2, (double)-225);
	set_dpoly_i(df, 3, (double)85);
	set_dpoly_i(df, 4, (double)-15);
	set_dpoly_i(df, 5, (double)1);


	// ff = (x-1)(x-2)...(x-10)
	set_dpoly_i(df, 0, (double)3628800);
	set_dpoly_i(df, 1, (double)-10628640);
	set_dpoly_i(df, 2, (double)12753576);
	set_dpoly_i(df, 3, (double)-8409500);
	set_dpoly_i(df, 4, (double)3416930);
	set_dpoly_i(df, 5, (double)-902055);
	set_dpoly_i(df, 6, (double)157773);
	set_dpoly_i(df, 7, (double)-18150);
	set_dpoly_i(df, 8, (double)1320);
	set_dpoly_i(df, 9, (double)-55);
	set_dpoly_i(df,10, (double)1);
*/

	/* Bcast dpoly */
	_mpi_bcast_dpoly(df, MPI_COMM_WORLD);
	if(myrank == 0)
	{
		printf("rank: %d\n", myrank);
		print_dpoly(df);
	}

	/* set Aberth's initial value */
	_mpi_ddka_init(local_cdinit, df, MPI_COMM_WORLD);
//	print_cdarray(local_cdinit);

	/* DKA method */
	if(myrank == 0) startwtime[0] = MPI_Wtime();
	_mpi_ddka(&dtimes, cdans, local_cdans, cdinit, local_cdinit, df, 1000, dabs_eps, drel_eps, MPI_COMM_WORLD);
	if(myrank == 0) endwtime[0] = MPI_Wtime();

	/* print answer */
	if(myrank == 0) printf("Iterative times: %d\n", dtimes);
	_mpi_collect_cdarray(cdans, dd_dim, local_cdans, MPI_COMM_WORLD);
	if(myrank == 0) print_cdarray(cdans);

	/* clear */
	free_dpoly(df);
	free_cdarray(cdans);
	free_cdarray(cdinit);

//	goto end;

#ifdef USE_GMP
/* mpf_t */
mpf:
#define PREC 128
//#define PREC 256
//#define PREC 512
//#define PREC 1024
#define MPFDEG 20
//	set_bnc_default_prec(PREC);
	_mpi_set_bnc_default_prec(PREC, MPI_COMM_WORLD);
	commit_mpf(&(MPI_MPF), PREC, MPI_COMM_WORLD);
	commit_mpi_mpfcmplx(&(MPI_BNC_MPFCMPLX), PREC, MPI_COMM_WORLD);

	/* init */
	mpf_init_set_d(mpfabs_eps, 1.0e-100);
	mpf_init_set_d(mpfrel_eps, 1.0e-7);

	mpff = init_mpfpoly(MAX_LENGTH);
	cmpfans = init_cmpfarray(MPFDEG);
	cmpfinit = init_cmpfarray(MPFDEG);

	local_dim = _mpi_divide_dim(dd_dim, MPFDEG, num_procs);
	if(myrank == 0)
	{
//		for(i = 0; i < num_procs; i++)
//			printf("%d d_dim[%d]: %d\n", local_dim, i, dd_dim[i]);
	}
	local_cmpfans = init_cmpfarray(local_dim);
	local_cmpfinit = init_cmpfarray(local_dim);

	cmpfans = init_cmpfarray(MPFDEG);
	cmpfinit = init_cmpfarray(MPFDEG);
	if(myrank == 0)
	{
		mpfcoef = fopen("test_mpidkacoef2.dat", "r");
		fread_mpfpolycoef(mpfcoef, mpff, DEG);
		print_mpfpoly(mpff);fflush(stdout);
		fclose(mpfcoef);
	}
//	goto end;

/*
	//  ff = (x-1)(x-2)...(x-10) 
	set_mpfpoly_i_str(mpff, 0, "3628800", 10);
	set_mpfpoly_i_str(mpff, 1, "-10628640", 10);
	set_mpfpoly_i_str(mpff, 2, "12753576", 10);
	set_mpfpoly_i_str(mpff, 3, "-8409500", 10);
	set_mpfpoly_i_str(mpff, 4, "3416930", 10);
	set_mpfpoly_i_str(mpff, 5, "-902055", 10);
	set_mpfpoly_i_str(mpff, 6, "157773", 10);
	set_mpfpoly_i_str(mpff, 7, "-18150", 10);
	set_mpfpoly_i_str(mpff, 8, "1320", 10);
	set_mpfpoly_i_str(mpff, 9, "-55", 10);
	set_mpfpoly_i_str(mpff,10, "1", 10);
*/

	/* Bcast mpfpoly */
	_mpi_bcast_mpfpoly(mpff, MPI_COMM_WORLD);
	if(myrank == 0)
	{
		printf("rank: %d\n", myrank);
		print_mpfpoly(mpff);fflush(stdout);
	}

	/* set Aberth's initial value */
	_mpi_mpf_dka_init(local_cmpfinit, mpff, MPI_COMM_WORLD);
//	print_cmpfarray(local_cmpfinit);

	/* DKA method */
	if(myrank == 0) startwtime[1] = MPI_Wtime();
	_mpi_mpf_dka(&mpftimes, cmpfans, local_cmpfans, cmpfinit, local_cmpfinit, mpff, 1000, mpfabs_eps, mpfrel_eps, MPI_COMM_WORLD);
	if(myrank == 0) endwtime[1] = MPI_Wtime();

	/* print answer */
	if(myrank == 0) printf("Iterative times: %d\n", mpftimes);
	_mpi_collect_cmpfarray(cmpfans, dd_dim, local_cmpfans, MPI_COMM_WORLD);
//	print_cmpfarray(local_cmpfans);
	if(myrank == 0) print_cmpfarray(cmpfans);

	/* clear */
	free_mpfpoly(mpff);
	free_cmpfarray(local_cmpfans);
	free_cmpfarray(local_cmpfinit);
	free_cmpfarray(cmpfans);
	free_cmpfarray(cmpfinit);

	free_mpi_mpfcmplx(&(MPI_BNC_MPFCMPLX));
	free_mpf(&(MPI_MPF));

#endif
end:
	MPI_Finalize();
	if(myrank == 0)
	{
		printf("double_DKA     : %f sec\n", endwtime[0] - startwtime[0]);
#ifdef USE_GMP
		printf("mpf_DKA(%dbits): %f sec\n", PREC, endwtime[1] - startwtime[1]);
#endif
	}
	exit(0);

}

