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


#define PROCS 8

/* double */
void ddg(DVector local_y, double x0, int myrank)
{
	long int i, index, base_index;

	for(i = 0; i < local_y->dim; i++)
	{
		set_dvector_i(local_y, i, 0.0);
	}

	return;
}

/* y(0) = 1/(1-alpha * sin(x)) */
/* y(1) = alpha * cos(x) / (1-alpha * sin(x))^2 */
void dans(DVector y, double x)
{
	long int i;

	for(i = 0; i < y->dim; i++)
		set_dvector_i(y, i, exp((i + 1) * x));

	return;
}

void initial_dvalue(double *lf_initx, double *lf_maxx, DVector lf_inity)
{
	long int i;

	/* x0 := 0 */
	/* y := [1, ..., 1]^T */
	*lf_initx = 0.0;
	*lf_maxx = 0.001;
	for(i = 0; i < lf_inity->dim; i++)
		set_dvector_i(lf_inity, i, 1.0);
}

#ifdef USE_GMP
/* mpf_t */
void dg(MPFVector local_y, mpf_t x0, int myrank)
{
	long int i, index, base_index;
	mpf_t tmp;

	mpf_init2(tmp, local_y->prec);

	/* y' = y */

	for(i = 0; i < local_y->dim; i++)
	{
		mpf_set_ui(tmp, 0.0);
		set_mpfvector_i(local_y, i, tmp);
	}

	mpf_clear(tmp);

	return;
}

/* y = [exp(x), ...., exp(x)]^T */
void ans(MPFVector y, mpf_t x)
{
	long int i;
	mpf_t tmp;

	mpf_init2(tmp, y->prec);
	for(i = 0; i < y->dim; i++)
	{
		mpf_mul_ui(tmp, x, i + 1);
		mpf_exp(tmp, tmp);
		set_mpfvector_i(y, i, tmp);
	}
	mpf_clear(tmp);
	return;
}


void initial_value(mpf_t lf_initx, mpf_t lf_maxx, MPFVector lf_inity)
{
	long int i;

	/* x0 := 0 */
	/* y := [1, ..., 1]^T */
	mpf_init_set_str(lf_initx, "0.0", 10);
	mpf_init_set_str(lf_maxx, "0.001", 10);
	for(i = 0; i < lf_inity->dim; i++)
		set_mpfvector_i_str(lf_inity, i, "1.0", 10);
}
#endif

//#define DPREC 50
#define PREC 128
#define DIM 256

main(int argc, char *argv[])
{
	DVector dy, init_dy, lf_dtmp;
	DMatrix dmat;
	long int div_num, i;
	double dx, init_dx, dstepsize, dabs_tol, drel_tol;
	double stime, etime;
#ifdef USE_GMP
	MPFVector y, init_y, lf_tmp;
	MPFMatrix mat;
	mpf_t x, init_x, stepsize, abs_tol, rel_tol, tmp;
#endif
	int myrank, num_procs;

/* for MPI */
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

#ifdef USE_GMP
//	set_bnc_default_prec(128);
//	_mpi_set_bnc_default_prec_decimal(DPREC, MPI_COMM_WORLD);
//	commit_mpf(&(MPI_MPF), ceil(DPREC/log10(2.0)), MPI_COMM_WORLD);
	_mpi_set_bnc_default_prec(PREC, MPI_COMM_WORLD);
	commit_mpf(&(MPI_MPF), PREC, MPI_COMM_WORLD);
	create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, MPI_COMM_WORLD);

	/* x := 1 */
	mpf_init(init_x);
	mpf_init(x);

	y = init_mpfvector(DIM);
	lf_tmp = init_mpfvector(DIM);
	init_y = init_mpfvector(DIM);
	initial_value(init_x, x, init_y);

	if(myrank == 0)
	{
		mat = init_mpfmatrix(DIM, DIM);
		mpf_init(tmp);
/*		for(i = 0; i < DIM; i++)
		{
			mpf_set_ui(tmp, (unsigned long)(1));
			set_mpfmatrix_ij(mat, i, i, tmp);
		}
*/
		frank_mpfmatrix(mat);
	}
#endif

	dy = init_dvector(DIM);
	lf_dtmp = init_dvector(DIM);
	init_dy = init_dvector(DIM);
	initial_dvalue(&init_dx, &dx, init_dy);

	if(myrank == 0)
	{
		dmat = init_dmatrix(DIM, DIM);
/*		for(i = 0; i < DIM; i++)
			set_dmatrix_ij(dmat, i, i, (double)(1));
*/
		frank_dmatrix(dmat);
	}

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

	for(div_num = 8; div_num <= 128; div_num *= 4)
	{
		initial_dvalue(&init_dx, &dx, init_dy);
		stime = MPI_Wtime();
		_mpi_dex_nim_fs_lo(NULL, dx, dy, init_dx, init_dy, div_num, dmat, ddg, drel_tol, dabs_tol, 8, MPI_COMM_WORLD);
		etime = MPI_Wtime();
		if(myrank == 0)
		{
			printf("time(sec): %f\n\n", etime - stime);
//			dans(init_dy, dx);
//			dx = normi_dvector(init_dy);
//			sub_dvector(init_dy, init_dy, dy);
//			init_dx = normi_dvector(init_dy);
//			printf("error, rerror(infty norm): %10.3e, %10.3e\n\n", init_dx, init_dx/dx);
		}
#ifdef USE_GMP
		initial_value(init_x, x, init_y);
		stime = MPI_Wtime();
		_mpi_mpf_ex_nim_fs_lo(NULL, x, y, init_x, init_y, div_num, mat, dg, rel_tol, abs_tol, 8, MPI_COMM_WORLD);
		etime = MPI_Wtime();
		if(myrank == 0)
		{
			printf("time(sec): %f\n\n", etime - stime);
//			ans(init_y, x);
//			normi_mpfvector(x, init_y);
//			sub_mpfvector(init_y, init_y, y);
//			normi_mpfvector(init_x, init_y);
//			printf("error, rerror(infty norm): %10.3e, %10.3e\n\n", mpf_get_d(init_x), mpf_get_d(init_x)/mpf_get_d(x));
		}
#endif
	}
//	initial_dvalue(&init_dx, &dx, init_dy);
//	_mpi_dex_nim_lo(stdout, dx, dy, init_dx, init_dy, dstepsize, dmat, ddg, drel_tol, dabs_tol, 2, MPI_COMM_WORLD);
//	_mpi_dex_nim_lo(stdout, dx, dy, init_dx, init_dy, dstepsize, dmat, ddg, drel_tol, dabs_tol, 4, MPI_COMM_WORLD);
//	_mpi_dex_nim_lo(stdout, dx, dy, init_dx, init_dy, dstepsize, dmat, ddg, drel_tol, dabs_tol, 8, MPI_COMM_WORLD);
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
//	_mpi_mpf_ex_nim_lo(stdout, x, y, init_x, init_y, stepsize, mat, dg, rel_tol, abs_tol, 8, MPI_COMM_WORLD);
//	_mpi_mpf_ex_nim(stdout, x, y, init_x, init_y, stepsize, df, rel_tol, abs_tol, 16, ans, 1, MPI_COMM_WORLD);
#endif
	printf("-- ex_nim ended -- \n");

	goto END;

HARMONIC:

	/* Extrapolation */
	printf("-- ex_harmonic -- \n");
	for(div_num = 2; div_num <= 128; div_num *= 2)
	{
		initial_dvalue(&init_dx, &dx, init_dy);
		stime = MPI_Wtime();
		_mpi_dex_harmonic_fs_lo(NULL, dx, dy, init_dx, init_dy, div_num, dmat, ddg, drel_tol, dabs_tol, 8, MPI_COMM_WORLD);
		etime = MPI_Wtime();
		if(myrank == 0)
		{
			printf("time(sec): %f\n", etime - stime);
//			dans(init_dy, dx);
//			dx = normi_dvector(init_dy);
//			sub_dvector(init_dy, init_dy, dy);
//			init_dx = normi_dvector(init_dy);
//			printf("error, rerror(infty norm): %10.3e, %10.3e\n\n", init_dx, init_dx/dx);
		}
#ifdef USE_GMP
		initial_value(init_x, x, init_y);
		stime = MPI_Wtime();
		_mpi_mpf_ex_harmonic_fs_lo(NULL, x, y, init_x, init_y, div_num, mat, dg, rel_tol, abs_tol, 8, MPI_COMM_WORLD);
		etime = MPI_Wtime();
		if(myrank == 0)
		{
			printf("time(sec): %f\n", etime - stime);
//			ans(init_y, x);
//			normi_mpfvector(x, init_y);
//			sub_mpfvector(init_y, init_y, y);
//			normi_mpfvector(init_x, init_y);
//			printf("error, rerror(infty norm): %10.3e, %10.3e\n\n", mpf_get_d(init_x), mpf_get_d(init_x)/mpf_get_d(x));
		}
#endif
	}
	initial_dvalue(&init_dx, &dx, init_dy);
//	_mpi_dex_harmonic(stdout, dx, dy, init_dx, init_dy, dstepsize, ddf, drel_tol, dabs_tol, 4, dans, MPI_COMM_WORLD);
//	_mpi_dex_harmonic_lo(stdout, dx, dy, init_dx, init_dy, dstepsize, dmat, ddg, drel_tol, dabs_tol, 8, MPI_COMM_WORLD);
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
//	_mpi_mpf_ex_harmonic_lo(stdout, x, y, init_x, init_y, stepsize, mat, dg, rel_tol, abs_tol, 8, MPI_COMM_WORLD);
//	_mpi_mpf_ex_harmonic(stdout, x, y, init_x, init_y, stepsize, df, rel_tol, abs_tol, 16, ans, 1, MPI_COMM_WORLD);
#endif
	printf("-- ex_harmonic ended -- \n");

END:

	MPI_Finalize();
}
