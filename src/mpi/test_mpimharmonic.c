/********************************************************************************/
/* Test program for mpi_mharmonic.c                                             */
/* Copyright 2004-2011, Tomonori Kouya                                          */
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

/* Integration Interval */
void dinterval(double *a, double *b)
{
	*a = 0.0;
	*b = 1.0;
}

#ifdef USE_GMP
/* Integration Interval */
void mpf_interval(mpf_t a, mpf_t b)
{
	mpf_set_ui(a, 0UL);
	mpf_set_ui(b, 1UL);
}
#endif

/* test function : f(x) = exp(x) */
extern double df(double x)
{
	return exp(x);
}

#ifdef USE_GMP
/* test function : f(x) = exp(x) */
void mpf_f(mpf_t ret, mpf_t x)
{
	mpf_exp(ret, x);
}
#endif

main(int argc, char *argv[])
{
	long int num_div;
	double da, db, dans;
	double stime, etime;
#ifdef USE_GMP
	mpf_t mpf_a, mpf_b, mpf_ans;
#endif
	int myrank, num_procs;

/* for MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

#ifdef USE_GMP
#define DPREC 50
//	set_bnc_default_prec(128);
	_mpi_set_bnc_default_prec_decimal(DPREC, MPI_COMM_WORLD);
	commit_mpf(&(MPI_MPF), ceil(DPREC/log10(2.0)), MPI_COMM_WORLD);
	create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, MPI_COMM_WORLD);
#endif

/* double */
	dinterval(&da, &db);

	if(myrank == 0)
	{
	printf("IEEE double\n");

	printf("Romberg Integral[%25.17e, %25.17e]\n", da, db);
	}

	for(num_div = 8; num_div <= 128; num_div *= 2)
	{
//		printf(" %10d , %25.17e , ", num_div, dromberg_integral(da, db, df, num_div));
		stime = get_secv();
		//_mpi_dmromberg_integral(&dans, da, db, df, num_div, MPI_COMM_WORLD);
		_mpi_dmharmonic_integral(&dans, da, db, df, num_div, MPI_COMM_WORLD);
		etime = get_secv();
		if(myrank == 0)
		{
		printf(" %10d , %25.17e, %10.3e", num_div, dans, etime - stime);
//		printf("Y(DOUBLE):\n"); print_dmatrix(dymat);
//		printf("YDIFF    :\n"); print_dmatrix(dydiffmat);
		printf("\n");
		}
	}

/* mpf */
#ifdef USE_GMP


	mpf_init(mpf_ans);
	mpf_init(mpf_a); mpf_init(mpf_b);
	mpf_interval(mpf_a, mpf_b);

	if(myrank == 0)
	{
	printf("MPF(%d bits)\n", get_bnc_default_prec());
	printf("Romberg Integral[");

	mpf_out_str(stdout, 10, 0, mpf_a); printf(", ");
	mpf_out_str(stdout, 10, 0, mpf_b); printf("]\n");
	}
	for(num_div = 8; num_div <= 128; num_div *= 2)
	{
//		mpf_trapezoidal_fs(mpf_ans, mpf_a, mpf_b, mpf_f, num_div);
//		mpf_romberg_integral(mpf_ans, mpf_a, mpf_b, mpf_f, num_div);
		stime = MPI_Wtime();
//		mpf_mromberg_integral(mpf_ans, mpf_a, mpf_b, mpf_f, num_div);
		_mpi_mpf_mharmonic_integral(mpf_ans, mpf_a, mpf_b, mpf_f, num_div, MPI_COMM_WORLD);
//		mpf_mharmonic1_integral(mpf_ans, mpf_a, mpf_b, mpf_f, num_div);
//		mpf_mharmonic2_integral(mpf_ans, mpf_a, mpf_b, mpf_f, num_div);
//		mpf_mharmonic3_integral(mpf_ans, mpf_a, mpf_b, mpf_f, num_div);
		etime = MPI_Wtime();
		if(myrank == 0)
		{
//		printf("Y(MPF%d):\n", get_bnc_default_prec()); print_mpfmatrix(mpfymat);
//		printf("YDIFF    :\n"); print_mpfmatrix(mpfydiffmat);
		printf("%5d, ",num_div); mpf_out_str(stdout, 10, 0, mpf_ans);
		printf(", %10.3e", etime - stime);

		printf("\n");
		}
	}

	mpf_clear(mpf_ans);
	mpf_clear(mpf_a);
	mpf_clear(mpf_b);
#endif

	MPI_Finalize();

}
