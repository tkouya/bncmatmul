/************************************************/
/* Test program for ex_*.c                      */
/* Copyleft 2004, Tomonori Kouya                */
/************************************************/
#include <stdio.h>
#include <math.h>

#include "mpi.h"
#include "bnc.h"

#ifdef USE_GMP
#include "mpi_gmp.h"
#endif

#include "mpi_bnc.h"

/* double */
void ddf(DVector local_y, double x0, DVector y0, int myrank)
{
	/* y(0)' = y(1) */
	/* y(1)' = -alpha * y(0)^2 * sin(x0) + 2 * alpha * y(0) * y(1) * cos(x0) */
	if(myrank == 0)
	{
		set_dvector_i(local_y, 0, sinh(x0));
		set_dvector_i(local_y, 1, cosh(x0));
	}
	if(myrank == 1)
	{
		set_dvector_i(local_y, 0, sinh(x0));
		set_dvector_i(local_y, 1, cosh(x0));
	}
	if(myrank == 2)
	{
		set_dvector_i(local_y, 0, sinh(x0));
		set_dvector_i(local_y, 1, cosh(x0));
	}
	if(myrank == 3)
	{
		set_dvector_i(local_y, 0, sinh(x0));
		set_dvector_i(local_y, 1, cosh(x0));
	}
	if(myrank == 4)
	{
		set_dvector_i(local_y, 0, sinh(x0));
		set_dvector_i(local_y, 1, cosh(x0));
	}
	if(myrank == 5)
	{
		set_dvector_i(local_y, 0, sinh(x0));
		set_dvector_i(local_y, 1, cosh(x0));
	}
	if(myrank == 6)
	{
		set_dvector_i(local_y, 0, sinh(x0));
		set_dvector_i(local_y, 1, cosh(x0));
	}
	if(myrank == 7)
	{
		set_dvector_i(local_y, 0, sinh(x0));
		set_dvector_i(local_y, 1, cosh(x0));
	}

	return;
}

/* y(0) = 1/(1-alpha * sin(x)) */
/* y(1) = alpha * cos(x) / (1-alpha * sin(x))^2 */
void dans(DVector y, double x)
{
	long int i;

	set_dvector_i(y, 0, cosh(x));
	set_dvector_i(y, 1, sinh(x));
	set_dvector_i(y, 2, cosh(x));
	set_dvector_i(y, 3, sinh(x));
	set_dvector_i(y, 4, cosh(x));
	set_dvector_i(y, 5, sinh(x));
	set_dvector_i(y, 6, cosh(x));
	set_dvector_i(y, 7, sinh(x));
	set_dvector_i(y, 8, cosh(x));
	set_dvector_i(y, 9, sinh(x));
	set_dvector_i(y, 10, cosh(x));
	set_dvector_i(y, 11, sinh(x));
	set_dvector_i(y, 12, cosh(x));
	set_dvector_i(y, 13, sinh(x));
	set_dvector_i(y, 14, cosh(x));
	set_dvector_i(y, 15, sinh(x));

	return;
}

void initial_dvalue(double *lf_initx, double *lf_maxx, DVector lf_inity)
{
	long int i;

	/* x0 := 0 */
	/* y := [1, ..., 1]^T */
	*lf_initx = 0.0;
	*lf_maxx = 37.0;

	set_dvector_i(lf_inity, 0, cosh(0.0));
	set_dvector_i(lf_inity, 1, sinh(0.0));
	set_dvector_i(lf_inity, 2, cosh(0.0));
	set_dvector_i(lf_inity, 3, sinh(0.0));
	set_dvector_i(lf_inity, 4, cosh(0.0));
	set_dvector_i(lf_inity, 5, sinh(0.0));
	set_dvector_i(lf_inity, 6, cosh(0.0));
	set_dvector_i(lf_inity, 7, sinh(0.0));
	set_dvector_i(lf_inity, 8, cosh(0.0));
	set_dvector_i(lf_inity, 9, sinh(0.0));
	set_dvector_i(lf_inity, 10, cosh(0.0));
	set_dvector_i(lf_inity, 11, sinh(0.0));
	set_dvector_i(lf_inity, 12, cosh(0.0));
	set_dvector_i(lf_inity, 13, sinh(0.0));
	set_dvector_i(lf_inity, 14, cosh(0.0));
	set_dvector_i(lf_inity, 15, sinh(0.0));

}

#ifdef USE_GMP
/* mpf_t */
void df(MPFVector local_y, mpf_t x0, MPFVector y0, int myrank)
{
	static int init_flag = 1;
	mpf_t tmp, tmp1;

//	if(init_flag == 1)
	{
		mpf_init2(tmp, local_y->prec);
		init_flag = 0;
	}
	/* y' = y */
	/* y(0)' = y(1) */
	/* y(1)' = -alpha * y(0)^2 * sin(x0) + 2 * alpha * y(0) * y(1) * cos(x0) */
	if(myrank == 0)
	{
		mpfr_sinh(tmp, x0, get_bnc_default_rounding_mode());
		set_mpfvector_i(local_y, 0, tmp);
		mpfr_cosh(tmp, x0, get_bnc_default_rounding_mode());
		set_mpfvector_i(local_y, 1, tmp);
	}
	if(myrank == 1)
	{
		mpfr_sinh(tmp, x0, get_bnc_default_rounding_mode());
		set_mpfvector_i(local_y, 0, tmp);
		mpfr_cosh(tmp, x0, get_bnc_default_rounding_mode());
		set_mpfvector_i(local_y, 1, tmp);
	}
	if(myrank == 2)
	{
		mpfr_sinh(tmp, x0, get_bnc_default_rounding_mode());
		set_mpfvector_i(local_y, 0, tmp);
		mpfr_cosh(tmp, x0, get_bnc_default_rounding_mode());
		set_mpfvector_i(local_y, 1, tmp);
	}
	if(myrank == 3)
	{
		mpfr_sinh(tmp, x0, get_bnc_default_rounding_mode());
		set_mpfvector_i(local_y, 0, tmp);
		mpfr_cosh(tmp, x0, get_bnc_default_rounding_mode());
		set_mpfvector_i(local_y, 1, tmp);
	}
	if(myrank == 4)
	{
		mpfr_sinh(tmp, x0, get_bnc_default_rounding_mode());
		set_mpfvector_i(local_y, 0, tmp);
		mpfr_cosh(tmp, x0, get_bnc_default_rounding_mode());
		set_mpfvector_i(local_y, 1, tmp);
	}
	if(myrank == 5)
	{
		mpfr_sinh(tmp, x0, get_bnc_default_rounding_mode());
		set_mpfvector_i(local_y, 0, tmp);
		mpfr_cosh(tmp, x0, get_bnc_default_rounding_mode());
		set_mpfvector_i(local_y, 1, tmp);
	}
	if(myrank == 6)
	{
		mpfr_sinh(tmp, x0, get_bnc_default_rounding_mode());
		set_mpfvector_i(local_y, 0, tmp);
		mpfr_cosh(tmp, x0, get_bnc_default_rounding_mode());
		set_mpfvector_i(local_y, 1, tmp);
	}
	if(myrank == 7)
	{
		mpfr_sinh(tmp, x0, get_bnc_default_rounding_mode());
		set_mpfvector_i(local_y, 0, tmp);
		mpfr_cosh(tmp, x0, get_bnc_default_rounding_mode());
		set_mpfvector_i(local_y, 0, tmp);
	}

	mpf_clear(tmp);

	return;
}

/* y(0) = 1/(1-alpha * sin(x)) */
/* y(1) = alpha * cos(x) / (1-alpha * sin(x))^2 */
void ans(MPFVector y, mpf_t x)
{
	long int i;
	mpf_t tmp, tmp1;

	mpf_init2(tmp, y->prec);

	mpfr_cosh(tmp, x, get_bnc_default_rounding_mode());
	set_mpfvector_i(y, 0, tmp);

	mpfr_sinh(tmp, x, get_bnc_default_rounding_mode());
	set_mpfvector_i(y, 1, tmp);

	mpfr_cosh(tmp, x, get_bnc_default_rounding_mode());
	set_mpfvector_i(y, 2, tmp);

	mpfr_sinh(tmp, x, get_bnc_default_rounding_mode());
	set_mpfvector_i(y, 3, tmp);

	mpfr_cosh(tmp, x, get_bnc_default_rounding_mode());
	set_mpfvector_i(y, 4, tmp);

	mpfr_sinh(tmp, x, get_bnc_default_rounding_mode());
	set_mpfvector_i(y, 5, tmp);

	mpfr_cosh(tmp, x, get_bnc_default_rounding_mode());
	set_mpfvector_i(y, 6, tmp);

	mpfr_sinh(tmp, x, get_bnc_default_rounding_mode());
	set_mpfvector_i(y, 7, tmp);

	mpfr_cosh(tmp, x, get_bnc_default_rounding_mode());
	set_mpfvector_i(y, 8, tmp);

	mpfr_sinh(tmp, x, get_bnc_default_rounding_mode());
	set_mpfvector_i(y, 9, tmp);

	mpfr_cosh(tmp, x, get_bnc_default_rounding_mode());
	set_mpfvector_i(y, 10, tmp);

	mpfr_sinh(tmp, x, get_bnc_default_rounding_mode());
	set_mpfvector_i(y, 11, tmp);

	mpfr_cosh(tmp, x, get_bnc_default_rounding_mode());
	set_mpfvector_i(y, 12, tmp);

	mpfr_sinh(tmp, x, get_bnc_default_rounding_mode());
	set_mpfvector_i(y, 13, tmp);

	mpfr_cosh(tmp, x, get_bnc_default_rounding_mode());
	set_mpfvector_i(y, 14, tmp);

	mpfr_sinh(tmp, x, get_bnc_default_rounding_mode());
	set_mpfvector_i(y, 15, tmp);

	mpf_clear(tmp);

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

	mpfr_cosh(tmp, lf_initx, get_bnc_default_rounding_mode());
	set_mpfvector_i(lf_inity, 0, tmp);

	mpfr_sinh(tmp, lf_initx, get_bnc_default_rounding_mode());
	set_mpfvector_i(lf_inity, 1, tmp);

	mpfr_cosh(tmp, lf_initx, get_bnc_default_rounding_mode());
	set_mpfvector_i(lf_inity, 2, tmp);

	mpfr_sinh(tmp, lf_initx, get_bnc_default_rounding_mode());
	set_mpfvector_i(lf_inity, 3, tmp);

	mpfr_cosh(tmp, lf_initx, get_bnc_default_rounding_mode());
	set_mpfvector_i(lf_inity, 4, tmp);

	mpfr_sinh(tmp, lf_initx, get_bnc_default_rounding_mode());
	set_mpfvector_i(lf_inity, 5, tmp);

	mpfr_cosh(tmp, lf_initx, get_bnc_default_rounding_mode());
	set_mpfvector_i(lf_inity, 6, tmp);

	mpfr_sinh(tmp, lf_initx, get_bnc_default_rounding_mode());
	set_mpfvector_i(lf_inity, 7, tmp);

	mpfr_cosh(tmp, lf_initx, get_bnc_default_rounding_mode());
	set_mpfvector_i(lf_inity, 8, tmp);

	mpfr_sinh(tmp, lf_initx, get_bnc_default_rounding_mode());
	set_mpfvector_i(lf_inity, 9, tmp);

	mpfr_cosh(tmp, lf_initx, get_bnc_default_rounding_mode());
	set_mpfvector_i(lf_inity, 10, tmp);

	mpfr_sinh(tmp, lf_initx, get_bnc_default_rounding_mode());
	set_mpfvector_i(lf_inity, 11, tmp);

	mpfr_cosh(tmp, lf_initx, get_bnc_default_rounding_mode());
	set_mpfvector_i(lf_inity, 12, tmp);

	mpfr_sinh(tmp, lf_initx, get_bnc_default_rounding_mode());
	set_mpfvector_i(lf_inity, 13, tmp);

	mpfr_cosh(tmp, lf_initx, get_bnc_default_rounding_mode());
	set_mpfvector_i(lf_inity, 14, tmp);

	mpfr_sinh(tmp, lf_initx, get_bnc_default_rounding_mode());
	set_mpfvector_i(lf_inity, 15, tmp);

	mpf_clear(tmp);
}
#endif

#define DPREC 100
#define DIM 16

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
//	_mpi_dex_nim(stdout, dx, dy, init_dx, init_dy, dstepsize, ddf, drel_tol, dabs_tol, 8, dans, MPI_COMM_WORLD);
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
