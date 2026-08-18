/********************************************************************************/
/* Test program for ex_*.c                                                      */
/* Copyright (C) 2004-2011 Tomonori Kouya                                       */
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

#define DALPHA 0.99999999
//#define MPFALPHA_U 99999999
//#define MPFALPHA_D 100000000
#define MPFALPHA_U 99999999
#define MPFALPHA_D 100000000
//#define DALPHA 0.9

/* double */
void ddf(DVector local_y, double x0, DVector y0, MPI_Comm comm)
{
	int myrank;

	MPI_Comm_rank(comm, &myrank);

	/* y(0)' = y(1) */
	/* y(1)' = -alpha * y(0)^2 * sin(x0) + 2 * alpha * y(0) * y(1) * cos(x0) */
	if(myrank == 0)
		set_dvector_i(local_y, 0, gdvi(y0, 1));
	if(myrank == 1)
		set_dvector_i(local_y, 0, -DALPHA * gdvi(y0, 0) *gdvi(y0, 0) * sin(x0) + 2 * DALPHA * gdvi(y0, 0) * gdvi(y0, 1) * cos(x0));

	return;
}

/* y(0) = 1/(1-alpha * sin(x)) */
/* y(1) = alpha * cos(x) / (1-alpha * sin(x))^2 */
void dans(DVector y, double x)
{
	long int i;
	double base;

	base = 1.0 - DALPHA * sin(x);

	set_dvector_i(y, 0, 1.0 / base);
	set_dvector_i(y, 1, DALPHA * cos(x) / (base * base));

	return;
}

void initial_dvalue(double *lf_initx, double *lf_maxx, DVector lf_inity)
{
	long int i;

	/* x0 := 0 */
	/* y := [1, ..., 1]^T */
	*lf_initx = 0.0;
	*lf_maxx = 37.0;

	set_dvector_i(lf_inity, 0, 1.0);
	set_dvector_i(lf_inity, 1, DALPHA);
}

#ifdef USE_GMP
/* mpf_t */
void df(MPFVector local_y, mpf_t x0, MPFVector y0, MPI_Comm comm)
{
	static int init_flag = 1;
	mpf_t tmp, tmp1;
	int myrank;

	MPI_Comm_rank(comm, &myrank);

//	if(init_flag == 1)
	{
		mpf_init2(tmp, local_y->prec);
		mpf_init2(tmp1, local_y->prec);
		init_flag = 0;
	}
	/* y' = y */
	/* y(0)' = y(1) */
	/* y(1)' = -alpha * y(0)^2 * sin(x0) + 2 * alpha * y(0) * y(1) * cos(x0) */
	if(myrank == 0)
		set_mpfvector_i(local_y, 0, gmpfvi(y0, 1));

	if(myrank == 1)
	{
		mpf_sin(tmp, x0);
		mpf_mul_ui(tmp, tmp, MPFALPHA_U);
		mpf_div_ui(tmp, tmp, MPFALPHA_D);
		mpf_mul(tmp, tmp, gmpfvi(y0, 0));
		mpf_mul(tmp, tmp, gmpfvi(y0, 0));
		mpf_neg(tmp, tmp);

		mpf_cos(tmp1, x0);
		mpf_mul_ui(tmp1, tmp1, 2UL);
		mpf_mul_ui(tmp1, tmp1, MPFALPHA_U);
		mpf_div_ui(tmp1, tmp1, MPFALPHA_D);
		mpf_mul(tmp1, tmp1, gmpfvi(y0, 0));
		mpf_mul(tmp1, tmp1, gmpfvi(y0, 1));

		mpf_add(tmp, tmp, tmp1);
		set_mpfvector_i(local_y, 0, tmp);
	}

	mpf_clear(tmp);
	mpf_clear(tmp1);

	return;
}

/* y(0) = 1/(1-alpha * sin(x)) */
/* y(1) = alpha * cos(x) / (1-alpha * sin(x))^2 */
void ans(MPFVector y, mpf_t x)
{
	long int i;
	mpf_t tmp, tmp1;

	mpf_init2(tmp, y->prec);
	mpf_init2(tmp1, y->prec);

	mpf_sin(tmp, x);
	mpf_mul_ui(tmp, tmp, MPFALPHA_U);
	mpf_div_ui(tmp, tmp, MPFALPHA_D);
	mpf_ui_sub(tmp, 1UL, tmp);
	mpf_ui_div(tmp, 1UL, tmp);

	set_mpfvector_i(y, 0, tmp);

	mpf_mul(tmp, tmp, tmp);
	mpf_mul_ui(tmp, tmp, MPFALPHA_U);
	mpf_div_ui(tmp, tmp, MPFALPHA_D);
	mpf_cos(tmp1, x);
	mpf_mul(tmp, tmp, tmp1);
	
	set_mpfvector_i(y, 1, tmp);

	mpf_clear(tmp);
	mpf_clear(tmp1);
	return;
}


void initial_value(mpf_t lf_initx, mpf_t lf_maxx, MPFVector lf_inity)
{
	long int i;
	mpf_t tmp;

	mpf_init2(tmp, lf_inity->prec);

	/* x0 := 0 */
	/* y := [1, ..., 1]^T */
	mpf_set_ui(lf_initx, 0UL);
	mpf_set_ui(lf_maxx, 37UL);

	mpf_set_ui(tmp, 1UL);
	set_mpfvector_i(lf_inity, 0, tmp);
	mpf_set_ui(tmp, MPFALPHA_U);
	mpf_div_ui(tmp, tmp, MPFALPHA_D);
	set_mpfvector_i(lf_inity, 1, tmp);

	mpf_clear(tmp);
}
#endif

#define DPREC 50
#define DIM 2

main(int argc, char *argv[])
{
	DVector dy, init_dy, lf_dtmp;
	long int div_num;
	double dx, init_dx, dstepsize, dabs_tol, drel_tol;
	double stime, etime;
#ifdef USE_GMP
	MPFVector y, init_y, lf_tmp;
	mpf_t x, init_x, stepsize, abs_tol, rel_tol;
#endif
	int myrank, num_procs;

/* for MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

#ifdef USE_GMP
//	set_bnc_default_prec(128);
	_mpi_set_bnc_default_prec_decimal(DPREC, MPI_COMM_WORLD);
	commit_mpf(&(MPI_MPF), ceil(DPREC/log10(2.0)), MPI_COMM_WORLD);
	create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, MPI_COMM_WORLD);

	/* x := 1 */
	mpf_init(init_x);
	mpf_init(x);

	y = init_mpfvector(DIM);
	lf_tmp = init_mpfvector(DIM);
	init_y = init_mpfvector(DIM);
	initial_value(init_x, x, init_y);
#endif

	dy = init_dvector(DIM);
	lf_dtmp = init_dvector(DIM);
	init_dy = init_dvector(DIM);
	initial_dvalue(&init_dx, &dx, init_dy);

	dstepsize = 1.0/2;
	dabs_tol = 0.0;
//	drel_tol = 1.0e-15;
//	drel_tol = 1.0e-30;
	drel_tol = 0.0;
#ifdef USE_GMP
	mpf_init_set_d(stepsize, dstepsize);
	mpf_init_set_d(abs_tol, dabs_tol);
	mpf_init_set_d(rel_tol, drel_tol);
//	mpf_init_set_d(rel_tol, 1.0e-20);
#endif

//	goto HARMONIC;

	/* Extrapolation */
	printf("-- ex_nim -- \n");
	initial_dvalue(&init_dx, &dx, init_dy);
//	_mpi_dex_nim(stdout, dx, dy, init_dx, init_dy, dstepsize, ddf, drel_tol, dabs_tol, 4, dans, MPI_COMM_WORLD);
	_mpi_dex_nim(stdout, dx, dy, init_dx, init_dy, dstepsize, ddf, drel_tol, dabs_tol, 8, dans, MPI_COMM_WORLD);
//	_mpi_dex_nim(stdout, dx, dy, init_dx, init_dy, dstepsize, ddf, drel_tol, dabs_tol, 16, dans, MPI_COMM_WORLD);
#ifdef USE_GMP
/* 4stages, 128bits */
/*
-- ex_nim ended --

real    2m57.379s
user    2m53.577s
sys     0m0.249s
*/
	initial_value(init_x, x, init_y);
//	_mpi_mpf_ex_nim(stdout, x, y, init_x, init_y, stepsize, df, rel_tol, abs_tol, 4, ans, 100, MPI_COMM_WORLD);
	_mpi_mpf_ex_nim(stdout, x, y, init_x, init_y, stepsize, df, rel_tol, abs_tol, 8, ans, 1, MPI_COMM_WORLD);
//	_mpi_mpf_ex_nim(stdout, x, y, init_x, init_y, stepsize, df, rel_tol, abs_tol, 16, ans, 1, MPI_COMM_WORLD);
#endif
	printf("-- ex_nim ended -- \n");

//	goto END;
HARMONIC:

	/* Extrapolation */
	printf("-- ex_harmonic -- \n");
	initial_dvalue(&init_dx, &dx, init_dy);
//	_mpi_dex_harmonic(stdout, dx, dy, init_dx, init_dy, dstepsize, ddf, drel_tol, dabs_tol, 4, dans, MPI_COMM_WORLD);
	_mpi_dex_harmonic(stdout, dx, dy, init_dx, init_dy, dstepsize, ddf, drel_tol, dabs_tol, 8, dans, MPI_COMM_WORLD);
//	_mpi_dex_harmonic(stdout, dx, dy, init_dx, init_dy, dstepsize, ddf, drel_tol, dabs_tol, 16, dans, MPI_COMM_WORLD);
#ifdef USE_GMP
/* 4stages, 128bits */
/*-- ex_harmonic ended --

real    2m35.402s
user    2m29.843s
sys     0m0.531s
*/
	initial_value(init_x, x, init_y);
//	_mpi_mpf_ex_harmonic(stdout, x, y, init_x, init_y, stepsize, df, rel_tol, abs_tol, 4, ans, 1, MPI_COMM_WORLD);
	_mpi_mpf_ex_harmonic(stdout, x, y, init_x, init_y, stepsize, df, rel_tol, abs_tol, 8, ans, 1, MPI_COMM_WORLD);
//	_mpi_mpf_ex_harmonic(stdout, x, y, init_x, init_y, stepsize, df, rel_tol, abs_tol, 16, ans, 1, MPI_COMM_WORLD);
#endif
	printf("-- ex_harmonic ended -- \n");

END:

	MPI_Finalize();
}
