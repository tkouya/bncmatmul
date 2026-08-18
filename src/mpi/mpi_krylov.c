/********************************************************************************/
/* mpi_krylov.c based on krylov.c in BNCpack                                    */
/*                                                                              */
/* Copyright (C) 2004-2011 Tomonori Kouya                                       */
/*                                                                              */
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

/************************************************************/
/*                                                          */
/* Bi-Conjugate-Gradient Method for Real Matrix             */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 2004.11.11 (Thu) Tomonori Kouya */
/*                                                          */
/************************************************************/
long int _mpi_DBiCG(DVector local_answer, DMatrix local_a[], DMatrix local_at[], DVector local_b, double reps, double aeps, long int maxtimes, long int dim, MPI_Comm comm)
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       DVector answer: Solution for Ax = b                */
/*       DMatrix a: Coefficient matrix A                    */
/*                                       (given by user)    */
/*       DVector b: Constant vector b   (given by user)     */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       DVector answer: Solution for Ax = b                */
/*                                                          */
/* ERRORS                                                   */
/* Positive value ... Normal : Iterative Times              */
/*      -1 ... Rho is zero.                                 */
/*      -2 ... Denominator of Alpha is zero.                */
/*      -3 ... (Reserved)                                   */
/*      -4 ... (Reserved)                                   */
/*      -5 ... Not Converge.                                */
/*                                                          */
/************************************************************/
{
	long int i, j, times, local_dim, d_dim[MPI_GMP_MAXPROCS];
	double alpha, alpha_num, alpha_den;
	double beta, beta_num;
	double rho, old_rho;
	double dtmp, init_resnorm;
	DVector local_vec[7], big_vec; /* Temporary Vectors */

	int myrank, num_procs;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);

/* Set initial value */
	for(i = 0; i < 7; i++)
		local_vec[i] = init_dvector(local_dim);
	big_vec = init_dvector(local_dim * num_procs);

	/* local_vec[0] ... approximation of solution */
	/* local_vec[1] ... residual : b - a * local_vec[0]  == w*/
	/* local_vec[2] ... (b - a * local_vec[0])^T  == wt */
	/* local_vec[3] ... p */
	/* local_vec[4] ... p^T */
	/* local_vec[5] ... z */
	/* local_vec[6] ... z^T */

	subst_dvector(local_vec[1], local_b); 
	subst_dvector(local_vec[2], local_b);

	beta_num = _mpi_ip_dvector(local_vec[1], local_vec[1], comm);
	init_resnorm = sqrt(beta_num);

	old_rho = 0.0;
	rho = 0.0;

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		rho = _mpi_ip_dvector(local_vec[2], local_vec[1], comm);

		if(rho == 0.0)
		{
			fprintf(stderr, "Rho is zero!(_mpi_DBiCG, %d)\n", times);
			return -1;
		}

		if(times == 0)
		{
			/* p := w, pt := wt */
			subst_dvector(local_vec[3], local_vec[1]);
			subst_dvector(local_vec[4], local_vec[2]);
		}
		else
		{
			beta = rho / old_rho;

			/* p := w + beta p, pt := wt + beta * pt */
			add_cmul_dvector(local_vec[3], local_vec[1], beta, local_vec[3]);
			add_cmul_dvector(local_vec[4], local_vec[2], beta, local_vec[4]);
		}
		/* z := Ap, zt := A^T pt */
		_mpi_mul_dmatrix_dvec(local_vec[5], local_a, local_vec[3], big_vec, comm);
		_mpi_mul_dmatrix_dvec(local_vec[6], local_at, local_vec[4], big_vec, comm);

		alpha_den = _mpi_ip_dvector(local_vec[4], local_vec[5], comm);
		if(alpha_den == 0.0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(DBiCG, %d)\n", times);
			return -2;
		}
		alpha = rho / alpha_den;

		/* x = x + alpha p */
		add_cmul_dvector(local_vec[0], local_vec[0], alpha, local_vec[3]);

		/* residual */
		add_cmul_dvector(local_vec[1], local_vec[1], -alpha, local_vec[5]);
		add_cmul_dvector(local_vec[2], local_vec[2], -alpha, local_vec[6]);

		beta_num = _mpi_ip_dvector(local_vec[1], local_vec[1], comm);

		/* Stopping Criteria */
		dtmp = sqrt(beta_num);
		if(dtmp <= aeps + reps * init_resnorm)
		{
			subst_dvector(local_answer, local_vec[0]);
			return times;
		}

		old_rho = rho;
	}

	/* Not converge */
	subst_dvector(local_answer, local_vec[0]);

	/* free local_vec[0]..[3]; */
	for(i = 0; i < 7; i++)
		free_dvector(local_vec[i]);
	free_dvector(big_vec);

	fprintf(stderr, "Not converge!(DBiCG, %d)\n", times);
	return -5;

}

/************************************************************/
/*                                                          */
/*               CGS Method for Real Matrix                 */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 2004.11.06 (Sat) Tomonori Kouya */
/*                                                          */
/************************************************************/
long int _mpi_DCGS(DVector local_answer, DMatrix local_a[], DVector local_b, double reps, double aeps, long int maxtimes, long int dim, MPI_Comm comm)
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       DVector answer: Solution for Ax = b                */
/*       DMatrix a: Coefficient matrix A                    */
/*                                       (given by user)    */
/*       DVector b: Constant vector b   (given by user)     */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       DVector answer: Solution for Ax = b                */
/*                                                          */
/* ERRORS                                                   */
/* Positive value ... Normal : Iterative Times              */
/*      -1 ... Rho is zero.                                 */
/*      -2 ... Denominator of Alpha is zero.                */
/*      -3 ... (Reserved)                                   */
/*      -4 ... (Reserved)                                   */
/*      -5 ... Not Converge.                                */
/*                                                          */
/************************************************************/
{
	long int i, j, times, local_dim, d_dim[MPI_GMP_MAXPROCS];
	double alpha, alpha_num, alpha_den;
	double beta, beta_num;
	double rho, old_rho;
	double dtmp, init_resnorm;
	DVector local_vec[9], big_vec; /* Temporary Vectors */

	int myrank, num_procs;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);

/* Set initial value */
	for(i = 0; i < 9; i++)
		local_vec[i] = init_dvector(local_dim);
	big_vec = init_dvector(local_dim * num_procs);

	/* local_vec[0] ... approximation of solution */
	/* local_vec[1] ... residual : b - a * local_vec[0] */
	/* local_vec[2] ... (b - a * local_vec[0])^T */
	/* local_vec[3] ... p */
	/* local_vec[4] ... p^T */
	/* local_vec[5] ... u */
	/* local_vec[6] ... u^T */
	/* local_vec[7] ... q */
	/* local_vec[8] ... v^T */

	subst_dvector(local_vec[1], local_b); 
	subst_dvector(local_vec[2], local_b);

	beta_num = _mpi_ip_dvector(local_vec[1], local_vec[1], comm);
	init_resnorm = sqrt(beta_num);

	old_rho = 0.0;
	rho = 0.0;

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		rho = _mpi_ip_dvector(local_vec[2], local_vec[1], comm);

		if(rho == 0.0)
		{
			fprintf(stderr, "Rho is zero!(DCGS, %d)\n", times);
			return -1;
		}

		if(times == 0)
		{
			/* u := rt, p := u */
			subst_dvector(local_vec[5], local_vec[1]);
			subst_dvector(local_vec[3], local_vec[5]);
		}
		else
		{
			beta = rho / old_rho;

			/* u := r + beta q, p := u + beta * (q + beta p) */
			add_cmul_dvector(local_vec[5], local_vec[1], beta, local_vec[7]);
			add_cmul_dvector(local_vec[8], local_vec[7], beta, local_vec[3]);
			add_cmul_dvector(local_vec[3], local_vec[5], beta, local_vec[8]);
		}
		/* precondition */
		/* pt = linsolve(pc_K, -p) */
		
		/* vt := Apt */
		_mpi_mul_dmatrix_dvec(local_vec[8], local_a,  local_vec[3], big_vec, comm);

		alpha_den = _mpi_ip_dvector(local_vec[2], local_vec[8], comm);
		if(alpha_den == 0.0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(DCGS, %d)\n", times);
			return -2;
		}
		alpha = rho / alpha_den;

		/* q = u - alpha  vt */
		add_cmul_dvector(local_vec[7], local_vec[5], -alpha, local_vec[8]);

		/* ut = u + q */
		add_dvector(local_vec[6], local_vec[5], local_vec[7]);

		/* x = x + alpha ut */
		add_cmul_dvector(local_vec[0], local_vec[0], alpha, local_vec[6]);

		/* residual */
		_mpi_mul_dmatrix_dvec(local_vec[8], local_a, local_vec[6], big_vec, comm);
		add_cmul_dvector(local_vec[1], local_vec[1], -alpha, local_vec[8]);

		beta_num = _mpi_ip_dvector(local_vec[1], local_vec[1], comm);

		/* Stopping Criteria */
		dtmp = sqrt(beta_num);
		if(dtmp <= aeps + reps * init_resnorm)
		{
			subst_dvector(local_answer, local_vec[0]);
			return times;
		}

		old_rho = rho;
	}

	/* Not converge */
	subst_dvector(local_answer, local_vec[0]);

	/* free local_vec[0]..[8]; */
	for(i = 0; i < 9; i++)
		free_dvector(local_vec[i]);
	free_dvector(big_vec);

	fprintf(stderr, "Not converge!(DCGS, %d)\n", times);
	return -5;

}

/************************************************************/
/*                                                          */
/*             BiCGSTAB Method for Real Matrix              */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 2004.11.06 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
long int _mpi_DBiCGSTAB(DVector local_answer, DMatrix local_a[], DVector local_b, double reps, double aeps, long int maxtimes, long int dim, MPI_Comm comm)
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       DVector answer: Solution for Ax = b                */
/*       DMatrix a: Coefficient matrix A                    */
/*                                       (given by user)    */
/*       DVector b: Constant vector b   (given by user)     */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       DVector answer: Solution for Ax = b                */
/*                                                          */
/* ERRORS                                                   */
/* Positive value ... Normal : Iterative Times              */
/*      -1 ... Rho is zero.                                 */
/*      -2 ... Denominator of Alpha is zero.                */
/*      -3 ... Denominator of Omega is zero.                */
/*      -4 ... Numerator of Omega is zero.                  */
/*      -5 ... Not Converge.                                */
/*                                                          */
/************************************************************/
{
	long int i, j, times, local_dim, d_dim[MPI_GMP_MAXPROCS];
	double alpha, alpha_num, alpha_den;
	double beta, beta_num;
	double rho, old_rho;
	double omega, omega_den;
	double dtmp, init_resnorm;
	DVector local_vec[9], big_vec; /* Temporary Vectors */

	int myrank, num_procs;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);

/* Set initial value */
	for(i = 0; i < 9; i++)
		local_vec[i] = init_dvector(local_dim);
	big_vec = init_dvector(local_dim * num_procs);

	/* local_vec[0] ... approximation of solution */
	/* local_vec[1] ... residual : b - a * local_vec[0]  == w*/
	/* local_vec[2] ... (b - a * local_vec[0])^T  == wt */
	/* local_vec[3] ... p */
	/* local_vec[4] ... p^T */
	/* local_vec[5] ... v */
	/* local_vec[6] ... s */
	/* local_vec[7] ... s^T */
	/* local_vec[8] ... t */

	subst_dvector(local_vec[1], local_b); 
	subst_dvector(local_vec[2], local_b);

	beta_num = _mpi_ip_dvector(local_vec[1], local_vec[1], comm);
	init_resnorm = sqrt(beta_num);

	old_rho = 0.0;
	rho = 0.0;

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		rho = _mpi_ip_dvector(local_vec[2], local_vec[1], comm);

		if(rho == 0.0)
		{
			fprintf(stderr, "Rho is zero!(DBiCG, %d)\n", times);
			return -1;
		}

		if(times == 0)
		{
			/* p := r */
			subst_dvector(local_vec[3], local_vec[1]);
		}
		else
		{
			beta = (rho / old_rho) * (alpha / omega);

			/* p := r + beta (p - omega v) */
			add_cmul_dvector(local_vec[4], local_vec[3], -omega, local_vec[5]);
			add_cmul_dvector(local_vec[3], local_vec[1], beta, local_vec[4]);
		}
		/* precondition */
		
		/* v := Apt */
		_mpi_mul_dmatrix_dvec(local_vec[5], local_a, local_vec[3], big_vec, comm);

		alpha_den = _mpi_ip_dvector(local_vec[2], local_vec[5], comm);
		if(alpha_den == 0.0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(DBiCGSTAB, %d)\n", times);
			return -2;
		}
		alpha = rho / alpha_den;

		/* s = r - alpha v */
		add_cmul_dvector(local_vec[6], local_vec[1], -alpha, local_vec[5]);

		/* Stopping Criteria */
		// dtmp = norm2_dvector(local_vec[6]);
		dtmp = _mpi_ip_dvector(local_vec[6], local_vec[6], comm);
		dtmp = sqrt(dtmp);

		if(dtmp <= aeps + reps * init_resnorm)
		{
			/* x = x + alpha pt */
			add_cmul_dvector(local_vec[0], local_vec[0], alpha, local_vec[3]);

			subst_dvector(local_answer, local_vec[0]);
			return times;
		}

		/* precondition */

		_mpi_mul_dmatrix_dvec(local_vec[8], local_a, local_vec[6], big_vec, comm);

		/* omega = (t, s) / (t, t) */
		omega_den = _mpi_ip_dvector(local_vec[8], local_vec[8], comm);
		if(omega_den == 0)
		{
			fprintf(stderr, "Denominator of Omega is zero!(DBiCGSTAB, %d)\n", times);
			return -3;
		}
		omega = _mpi_ip_dvector(local_vec[8], local_vec[6], comm);
		if(omega == 0)
		{
			fprintf(stderr, "Numerator of Omega is zero!(DBiCGSTAB, %d)\n", times);
			return -4;
		}
		omega = omega / omega_den;

		/* x = x + alpha pt + omega st */
		add_cmul_dvector(local_vec[4], local_vec[0], alpha, local_vec[3]);
		add_cmul_dvector(local_vec[0], local_vec[4], omega, local_vec[6]);

		/* residual */
		add_cmul_dvector(local_vec[1], local_vec[6], -omega, local_vec[8]);

		beta_num = _mpi_ip_dvector(local_vec[1], local_vec[1], comm);

		/* Stopping Criteria */
		dtmp = sqrt(beta_num);
		if(dtmp <= aeps + reps * init_resnorm)
		{
			subst_dvector(local_answer, local_vec[0]);
			return times;
		}

		old_rho = rho;
	}

	/* Not converge */
	subst_dvector(local_answer, local_vec[0]);

	/* free local_vec[0]..[3]; */
	for(i = 0; i < 9; i++)
		free_dvector(local_vec[i]);
	free_dvector(big_vec);

	fprintf(stderr, "Not converge!(DBiCGSTAB, %d)\n", times);
	return -5;

}

/************************************************************/
/*                                                          */
/* GPBiCG(Generalized Product Bi-CG) Method for Real Matrix */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 2004.11.06 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
long int _mpi_DGPBiCG(DVector local_answer, DMatrix local_a[], DVector local_b, double reps, double aeps, long int maxtimes, long int dim, MPI_Comm comm)
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       DVector answer: Solution for Ax = b                */
/*       DMatrix a: Coefficient matrix A                    */
/*                                       (given by user)    */
/*       DVector b: Constant vector b   (given by user)     */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       DVector answer: Solution for Ax = b                */
/*                                                          */
/* ERRORS                                                   */
/* Positive value ... Normal : Iterative Times              */
/*      -1 ... Rho is zero.                                 */
/*      -2 ... Denominator of Alpha is zero.                */
/*      -3 ... (Reserved)                                   */
/*      -4 ... (Reserved)                                   */
/*      -5 ... Not Converge.                                */
/*                                                          */
/************************************************************/
{
	long int i, j, times, local_dim, d_dim[MPI_GMP_MAXPROCS];
	double alpha, alpha_num, alpha_den;
	double beta, beta_num;
	double rho, old_rho;
	double mu[5], tau, zeta, eta;
	double dtmp, init_resnorm;
	DVector local_vec[12], big_vec; /* Temporary Vectors */

	int myrank, num_procs;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);

/* Set initial value */
	for(i = 0; i < 12; i++)
		local_vec[i] = init_dvector(local_dim);
	big_vec = init_dvector(local_dim * num_procs);

	/* local_vec[0] ... approximation of solution */
	/* local_vec[1] ... residual : b - a * local_vec[0] */
	/* local_vec[2] ... (b - a * local_vec[0])^T */
	/* local_vec[3] ... p */
	/* local_vec[4] ... q */
	/* local_vec[5] ... s */
	/* local_vec[6] ... t */
	/* local_vec[7] ... u */
	/* local_vec[8] ... v */
	/* local_vec[9] ... w */
	/* local_vec[10]... y */
	/* local_vec[11]... z */

	subst_dvector(local_vec[1], local_b); 
	subst_dvector(local_vec[2], local_b);

	beta_num = _mpi_ip_dvector(local_vec[1], local_vec[1], comm);
	init_resnorm = sqrt(beta_num);

	old_rho = 0.0;
	rho = 0.0;

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		rho = _mpi_ip_dvector(local_vec[2], local_vec[1], comm);

		if(rho == 0.0)
		{
			fprintf(stderr, "Rho is zero!(DGPBiCG, %d)\n", times);
			return -1;
		}

		if(times == 0)
		{
			/*
				p = r;
				q = A * p;
				alpha = rho / (r0^t, q);
				t = r - alpha * q;
				v = A * t;
				y = alpha * q - r;
				mu2 = v' * t;
				mu5 = v' * v;
				zeta = mu2 / mu5;
				eta = 0;
				u = zeta * q; */
			subst_dvector(local_vec[3], local_vec[1]);
			_mpi_mul_dmatrix_dvec(local_vec[4], local_a,  local_vec[3], big_vec, comm);
			alpha_den = _mpi_ip_dvector(local_vec[2], local_vec[4], comm);
			if(alpha_den == 0.0)
			{
				fprintf(stderr, "Denominator of Alpha is zero!(DGPBiCG, %d)\n", times);
				return -2;
			}
			alpha = rho / alpha_den;
			add_cmul_dvector(local_vec[6], local_vec[1], -alpha, local_vec[4]);
			_mpi_mul_dmatrix_dvec(local_vec[8], local_a,  local_vec[6], big_vec, comm);
			cmul_dvector(local_vec[10], alpha, local_vec[4]);
			sub_dvector(local_vec[10], local_vec[10], local_vec[1]);
			mu[1] = _mpi_ip_dvector(local_vec[8], local_vec[6], comm);
			mu[4] = _mpi_ip_dvector(local_vec[8], local_vec[8], comm);
			if(mu[4] == 0.0)
			{
				fprintf(stderr, "Denominator of mu[4] is zero!(DGPBiCG, %d)\n", times);
				return -3;
			}
			zeta = mu[1] / mu[4];
			eta = 0.0;
			cmul_dvector(local_vec[7], zeta, local_vec[4]);
		}
		else
		{
			/*
			k_beta = (rho / old_rho) * (alpha / zeta);
			w = v + k_beta * q;
			p = ret_res + k_beta * (p - u);
			q = A * p;
			alpha = rho / (ret_res_c' * q);
			s = t - ret_res;
			t = ret_res - alpha * q;
			v = A * t;
			y = s - alpha * (w - q);
			mu1 = y' * y;
			mu2 = v' * t;
			mu3 = y' * t;
			mu4 = v' * y;
			mu5 = v' * v;
			tau = mu5 * mu1 - mu4 * mu4;
			zeta = (mu1 * mu2 - mu3 * mu4) / tau;
			eta = (mu5 * mu3 - mu4 * mu2) / tau;
			u = zeta * q + eta * (s + k_beta * u); */

			beta = (rho / old_rho) * (alpha / zeta);

			add_cmul_dvector(local_vec[9], local_vec[8], beta, local_vec[4]);
			sub_dvector(local_vec[3], local_vec[3], local_vec[7]);
			add_cmul_dvector(local_vec[3], local_vec[1], beta, local_vec[3]);
			_mpi_mul_dmatrix_dvec(local_vec[4], local_a,  local_vec[3], big_vec, comm);
			alpha_den = _mpi_ip_dvector(local_vec[2], local_vec[4], comm);
			if(alpha_den == 0.0)
			{
				fprintf(stderr, "Denominator of Alpha is zero!(DGPBiCG, %d)\n", times);
				return -2;
			}
			alpha = rho / alpha_den;
			sub_dvector(local_vec[5], local_vec[6], local_vec[1]);
			add_cmul_dvector(local_vec[6], local_vec[1], -alpha, local_vec[4]);
			_mpi_mul_dmatrix_dvec(local_vec[8], local_a,  local_vec[6], big_vec, comm);
			sub_dvector(local_vec[10], local_vec[9], local_vec[4]);
			add_cmul_dvector(local_vec[10], local_vec[5], -alpha, local_vec[10]);
			mu[0] = _mpi_ip_dvector(local_vec[10], local_vec[10], comm);
			mu[1] = _mpi_ip_dvector(local_vec[8], local_vec[6], comm);
			mu[2] = _mpi_ip_dvector(local_vec[10], local_vec[6], comm);
			mu[3] = _mpi_ip_dvector(local_vec[8], local_vec[10], comm);
			mu[4] = _mpi_ip_dvector(local_vec[8], local_vec[8], comm);
			tau = mu[4] * mu[0] - mu[3] * mu[3];
			zeta = (mu[0] * mu[1] - mu[2] * mu[3]) / tau;
			eta = (mu[4] * mu[2] - mu[3] * mu[1]) / tau;

			/* u = zeta * q + eta * (s + k_beta * u); */
			add_cmul_dvector(local_vec[7], local_vec[5], beta, local_vec[7]);
			cmul_dvector(local_vec[7], eta, local_vec[7]);
			add_cmul_dvector(local_vec[7], local_vec[7], zeta, local_vec[4]);
		}
		/* z = zeta * ret_res + eta * z - alpha * u; */
		cmul2_dvector(local_vec[11], eta);
		add_cmul_dvector(local_vec[11], local_vec[11], zeta, local_vec[1]);
		add_cmul_dvector(local_vec[11], local_vec[11], -alpha, local_vec[7]);

		/* x = x + alpha p + z */
		add_cmul_dvector(local_vec[0], local_vec[0], alpha, local_vec[3]);
		add2_dvector(local_vec[0], local_vec[11]);

		/* residual */
		/* r = t - eta * y - zeta * v; */
		add_cmul_dvector(local_vec[1], local_vec[6], -eta, local_vec[10]);
		add_cmul_dvector(local_vec[1], local_vec[1], -zeta, local_vec[8]);

		beta_num = _mpi_ip_dvector(local_vec[1], local_vec[1], comm);

		/* Stopping Criteria */
		dtmp = sqrt(beta_num);
		if(dtmp <= aeps + reps * init_resnorm)
		{
			subst_dvector(local_answer, local_vec[0]);
			return times;
		}
//		printf("DGPBiCG: %5d %10.3e\n", times, dtmp / init_resnorm);
		if(zeta == 0.0)
		{
			fprintf(stderr, "Denominator of Zeta is zero!(DGPBiCG, %d)\n", times);
			return -4;
		}

		old_rho = rho;
	}

	/* Not converge */
	subst_dvector(local_answer, local_vec[0]);

	/* free local_vec[0]..[3]; */
	for(i = 0; i < 12; i++)
		free_dvector(local_vec[i]);
	free_dvector(big_vec);

	fprintf(stderr, "Not converge!(DGPBiCG, %d)\n", times);
	return -5;

}


#ifdef USE_GMP

/************************************************************/
/*                                                          */
/*                Bi-Conjugate-Gradient Method              */
/*                                 (Multi-Precision)        */
/*                                                          */
/*                 ver. 0.0 2004.11.05 (Fri) Tomonori Kouya */
/*                                                          */
/************************************************************/
long int _mpi_MPFBiCG(MPFVector local_answer, MPFMatrix local_a[], MPFMatrix local_at[], MPFVector local_b, mpf_t reps, mpf_t aeps, long int maxtimes, long int dim, MPI_Comm comm)
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       MPFVector answer: Solution for Ax = b              */
/*       MPFMatrix a: Coefficient matrix A                  */
/*                                       (given by user)    */
/*       MPFVector b: Constant vector b   (given by user)   */
/*       mpf_t reps: Relative tolerance (given by user)     */
/*       mpf_t aeps: Absolute tolerance (given by user)     */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       MPFVector answer: Solution for Ax = b              */
/*                                                          */
/* ERRORS                                                   */
/* Positive value ... Normal : Iterative Times              */
/*      -1 ... Deminator of Alpha is zero.                  */
/*      -2 ... Numerator of Alpha is zero.                  */
/*      -3 ... (Reserved)                                   */
/*      -4 ... (Reserved)                                   */
/*      -5 ... Not Converge.                                */
/*                                                          */
/************************************************************/
{
	unsigned long int prec;
	long int i, j, times, local_dim, d_dim[MPI_GMP_MAXPROCS];
	mpf_t alpha, alpha_num, alpha_den;
	mpf_t rho, old_rho;
	mpf_t beta, beta_num;
	mpf_t dtmp, dtmp1, init_resnorm;
	MPFVector local_vec[7], big_vec; /* Temporary Vectors */

	int myrank, num_procs;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);

	prec = local_answer->prec;

/* Set initial value */
	mpf_init2(alpha, prec);
	mpf_init2(alpha_num, prec);
	mpf_init2(alpha_den, prec);
	mpf_init2(rho, prec);
	mpf_init2(old_rho, prec);
	mpf_init2(beta, prec);
	mpf_init2(beta_num, prec);
	mpf_init2(dtmp, prec);
	mpf_init2(dtmp1, prec);
	mpf_init2(init_resnorm, prec);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);
	for(i = 0; i < 7; i++)
		local_vec[i] = init2_mpfvector(local_dim, prec);
	big_vec = init2_mpfvector(num_procs * local_dim, prec);

	/* local_vec[0] ... approximation of solution */
	/* local_vec[1] ... residual : b - a * local_vec[0]  == w*/
	/* local_vec[2] ... (b - a * local_vec[0])^T  == wt */
	/* local_vec[3] ... p */
	/* local_vec[4] ... p^T */
	/* local_vec[5] ... z */
	/* local_vec[6] ... z^T */

	subst_mpfvector(local_vec[1], local_b); 
	subst_mpfvector(local_vec[2], local_b);

	_mpi_ip_mpfvector(beta_num, local_vec[1], local_vec[1], comm);
	mpf_sqrt(init_resnorm, beta_num);

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		_mpi_ip_mpfvector(rho, local_vec[2], local_vec[1], comm);

		if(mpf_cmp_ui(rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(MPFBiCG, %d)\n", times);
			return -1;
		}

		if(times == 0)
		{
			/* p := w, pt := wt */
			subst_mpfvector(local_vec[3], local_vec[1]);
			subst_mpfvector(local_vec[4], local_vec[2]);
		}
		else
		{
			mpf_div(beta, rho, old_rho);

			/* p := w + beta p, pt := wt + beta * pt */
			add_cmul_mpfvector(local_vec[3], local_vec[1], beta, local_vec[3]);
			add_cmul_mpfvector(local_vec[4], local_vec[2], beta, local_vec[4]);
		}
		/* z := Ap, zt := A^T pt */
		_mpi_mul_mpfmatrix_mpfvec(local_vec[5], local_a,  local_vec[3], big_vec, comm);
		_mpi_mul_mpfmatrix_mpfvec(local_vec[6], local_at, local_vec[4], big_vec, comm);

		_mpi_ip_mpfvector(alpha_den, local_vec[4], local_vec[5], comm);
		if(mpf_cmp_ui(alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(MPFBiCG, %d)\n", times);
			return -2;
		}
		mpf_div(alpha, rho, alpha_den);

		/* x = x + alpha p */
		add_cmul_mpfvector(local_vec[0], local_vec[0], alpha, local_vec[3]);

		/* residual */
		mpf_neg(dtmp, alpha);
		add_cmul_mpfvector(local_vec[1], local_vec[1], dtmp, local_vec[5]);
		add_cmul_mpfvector(local_vec[2], local_vec[2], dtmp, local_vec[6]);

		_mpi_ip_mpfvector(beta_num, local_vec[1], local_vec[1], comm);

		/* Stopping Criteria */
		mpf_sqrt(dtmp, beta_num);
		mpf_mul(dtmp1, reps, init_resnorm);
		mpf_add(dtmp1, dtmp1, aeps);
		if(mpf_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_mpfvector(local_answer, local_vec[0]);
			return times;
		}

		mpf_set(old_rho, rho);
	}

	/* Not converge */
	subst_mpfvector(local_answer, local_vec[0]);

	/* free local_vec[0]..[3]; */
	for(i = 0; i < 7; i++)
		free_mpfvector(local_vec[i]);
	free_mpfvector(big_vec);

	mpf_clear(alpha); mpf_clear(alpha_num); mpf_clear(alpha_den);
	mpf_clear(rho); mpf_clear(old_rho);
	mpf_clear(beta); mpf_clear(beta_num);
	mpf_clear(dtmp); mpf_clear(dtmp1); mpf_clear(init_resnorm);

	fprintf(stderr, "Not converge!(MPFBiCG, %d)\n", times);
	return -5;

}
/************************************************************/
/*                                                          */
/*               CGS Method for Real Matrix                 */
/*                                 (Multi-Precision)        */
/*                                                          */
/*                 ver. 0.0 2004.11.06 (Sat) Tomonori Kouya */
/*                                                          */
/************************************************************/
long int _mpi_MPFCGS(MPFVector local_answer, MPFMatrix local_a[], MPFVector local_b, mpf_t reps, mpf_t aeps, long int maxtimes, long int dim, MPI_Comm comm)
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       MPFVector answer: Solution for Ax = b              */
/*       MPFMatrix a: Coefficient matrix A                  */
/*                                       (given by user)    */
/*       MPFVector b: Constant vector b  (given by user)    */
/*       mpf_t reps: Relative tolerance (given by user)     */
/*       mpf_t aeps: Absolute tolerance (given by user)     */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       MPFVector answer: Solution for Ax = b              */
/*                                                          */
/* ERRORS                                                   */
/* Positive value ... Normal : Iterative Times              */
/*      -1 ... Rho is zero.                                 */
/*      -2 ... Denominator of Alpha is zero.                */
/*      -3 ... (Reserved)                                   */
/*      -4 ... (Reserved)                                   */
/*      -5 ... Not Converge.                                */
/*                                                          */
/************************************************************/
{
	long int i, j, times, local_dim, d_dim[MPI_GMP_MAXPROCS];
	unsigned long prec;
	mpf_t dtmp, dtmp1;
	mpf_t alpha, alpha_num, alpha_den;
	mpf_t beta, beta_num;
	mpf_t rho, old_rho;
	mpf_t init_resnorm;
	MPFVector local_vec[9], big_vec; /* Temporary Vectors */

	int myrank, num_procs;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);

	prec = local_answer->prec;

/* Set initial value */
	mpf_init2(alpha, prec);
	mpf_init2(alpha_num, prec);
	mpf_init2(alpha_den, prec);
	mpf_init2(rho, prec);
	mpf_init2(old_rho, prec);
	mpf_init2(beta, prec);
	mpf_init2(beta_num, prec);
	mpf_init2(dtmp, prec);
	mpf_init2(dtmp1, prec);
	mpf_init2(init_resnorm, prec);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);
	for(i = 0; i < 9; i++)
		local_vec[i] = init2_mpfvector(local_dim, prec);
	big_vec = init2_mpfvector(num_procs * local_dim, prec);

	/* local_vec[0] ... approximation of solution */
	/* local_vec[1] ... residual : b - a * local_vec[0] */
	/* local_vec[2] ... (b - a * local_vec[0])^T */
	/* local_vec[3] ... p */
	/* local_vec[4] ... p^T */
	/* local_vec[5] ... u */
	/* local_vec[6] ... u^T */
	/* local_vec[7] ... q */
	/* local_vec[8] ... v^T */

	subst_mpfvector(local_vec[1], local_b); 
	subst_mpfvector(local_vec[2], local_b);

	_mpi_ip_mpfvector(beta_num, local_vec[1], local_vec[1], comm);
	mpf_sqrt(init_resnorm, beta_num);

	mpf_set_ui(old_rho, 0UL);
	mpf_set_ui(rho, 0UL);

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		_mpi_ip_mpfvector(rho, local_vec[2], local_vec[1], comm);

		if(mpf_cmp_ui(rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(MPFCGS, %d)\n", times);
			return -1;
		}

		if(times == 0)
		{
			/* u := rt, p := u */
			subst_mpfvector(local_vec[5], local_vec[1]);
			subst_mpfvector(local_vec[3], local_vec[5]);
		}
		else
		{
			mpf_div(beta, rho, old_rho);

			/* u := r + beta q, p := u + beta * (q + beta p) */
			add_cmul_mpfvector(local_vec[5], local_vec[1], beta, local_vec[7]);
			add_cmul_mpfvector(local_vec[8], local_vec[7], beta, local_vec[3]);
			add_cmul_mpfvector(local_vec[3], local_vec[5], beta, local_vec[8]);
		}
		/* precondition */
		/* pt = linsolve(pc_K, -p) */
		
		/* vt := Apt */
		_mpi_mul_mpfmatrix_mpfvec(local_vec[8], local_a,  local_vec[3], big_vec, comm);

		_mpi_ip_mpfvector(alpha_den, local_vec[2], local_vec[8], comm);
		if(mpf_cmp_ui(alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(MPFCGS, %d)\n", times);
			return -2;
		}
		mpf_div(alpha, rho, alpha_den);

		/* q = u - alpha  vt */
		mpf_neg(dtmp, alpha);
		add_cmul_mpfvector(local_vec[7], local_vec[5], dtmp, local_vec[8]);

		/* ut = u + q */
		add_mpfvector(local_vec[6], local_vec[5], local_vec[7]);

		/* x = x + alpha ut */
		add_cmul_mpfvector(local_vec[0], local_vec[0], alpha, local_vec[6]);

		/* residual */
		_mpi_mul_mpfmatrix_mpfvec(local_vec[8], local_a,  local_vec[6], big_vec, comm);
		mpf_neg(dtmp, alpha);
		add_cmul_mpfvector(local_vec[1], local_vec[1], dtmp, local_vec[8]);

		_mpi_ip_mpfvector(beta_num, local_vec[1], local_vec[1], comm);

		/* Stopping Criteria */
		mpf_sqrt(dtmp, beta_num);
		mpf_mul(dtmp1, reps, init_resnorm);
		mpf_add(dtmp1, dtmp1, aeps);
		if(mpf_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_mpfvector(local_answer, local_vec[0]);
			return times;
		}

		mpf_set(old_rho, rho);
	}

	/* Not converge */
	subst_mpfvector(local_answer, local_vec[0]);

	/* free local_vec[0]..[8]; */
	for(i = 0; i < 9; i++)
		free_mpfvector(local_vec[i]);
	free_mpfvector(big_vec);

	mpf_clear(alpha); mpf_clear(alpha_num); mpf_clear(alpha_den);
	mpf_clear(rho); mpf_clear(old_rho);
	mpf_clear(beta); mpf_clear(beta_num);
	mpf_clear(dtmp); mpf_clear(dtmp1); mpf_clear(init_resnorm);

	fprintf(stderr, "Not converge!(MPFCGS, %d)\n", times);
	return -5;

}

/************************************************************/
/*                                                          */
/*             BiCGSTAB Method for Real Matrix              */
/*                                 (Multi-Precision)        */
/*                                                          */
/*                 ver. 0.0 2004.11.06 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
long int _mpi_MPFBiCGSTAB(MPFVector local_answer, MPFMatrix local_a[], MPFVector local_b, mpf_t reps, mpf_t aeps, long int maxtimes, long int dim, MPI_Comm comm)
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       MPFVector answer: Solution for Ax = b              */
/*       MPFMatrix a: Coefficient matrix A                  */
/*                                       (given by user)    */
/*       MPFVector b: Constant vector b   (given by user)   */
/*       mpf_t reps: Relative tolerance (given by user)     */
/*       mpf_t aeps: Absolute tolerance (given by user)     */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       MPFVector answer: Solution for Ax = b              */
/*                                                          */
/* ERRORS                                                   */
/* Positive value ... Normal : Iterative Times              */
/*      -1 ... Rho is zero.                                 */
/*      -2 ... Denominator of Alpha is zero.                */
/*      -3 ... Denominator of Omega is zero.                */
/*      -4 ... Numerator of Omega is zero.                  */
/*      -5 ... Not Converge.                                */
/*                                                          */
/************************************************************/
{
	long int i, j, times, local_dim, d_dim[MPI_GMP_MAXPROCS];
	unsigned long prec;
	mpf_t dtmp, dtmp1;
	mpf_t alpha, alpha_num, alpha_den;
	mpf_t beta, beta_num;
	mpf_t rho, old_rho;
	mpf_t omega, omega_den;
	mpf_t init_resnorm;
	MPFVector local_vec[9], big_vec; /* Temporary Vectors */

	int myrank, num_procs;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);

	prec = local_answer->prec;

/* Set initial value */
	mpf_init2(alpha, prec);
	mpf_init2(alpha_num, prec);
	mpf_init2(alpha_den, prec);
	mpf_init2(rho, prec);
	mpf_init2(old_rho, prec);
	mpf_init2(beta, prec);
	mpf_init2(beta_num, prec);
	mpf_init2(omega, prec);
	mpf_init2(omega_den, prec);
	mpf_init2(dtmp, prec);
	mpf_init2(dtmp1, prec);
	mpf_init2(init_resnorm, prec);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);
	for(i = 0; i < 9; i++)
		local_vec[i] = init2_mpfvector(local_dim, prec);
	big_vec = init2_mpfvector(num_procs * local_dim, prec);

	/* local_vec[0] ... approximation of solution */
	/* local_vec[1] ... residual : b - a * local_vec[0]  == w*/
	/* local_vec[2] ... (b - a * local_vec[0])^T  == wt */
	/* local_vec[3] ... p */
	/* local_vec[4] ... p^T */
	/* local_vec[5] ... v */
	/* local_vec[6] ... s */
	/* local_vec[7] ... s^T */
	/* local_vec[8] ... t */

	subst_mpfvector(local_vec[1], local_b); 
	subst_mpfvector(local_vec[2], local_b);

	_mpi_ip_mpfvector(beta_num, local_vec[1], local_vec[1], comm);
	mpf_sqrt(init_resnorm, beta_num);

	mpf_set_ui(old_rho, 0UL);
	mpf_set_ui(rho, 0UL);

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		_mpi_ip_mpfvector(rho, local_vec[2], local_vec[1], comm);

		if(mpf_cmp_ui(rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(MPFBiCGSTAB, %d)\n", times);
			return -1;
		}

		if(times == 0)
		{
			/* p := r */
			subst_mpfvector(local_vec[3], local_vec[1]);
		}
		else
		{
			mpf_div(beta, rho, old_rho);
			mpf_div(dtmp, alpha, omega);
			mpf_mul(beta, beta, dtmp);

			/* p := r + beta (p - omega v) */
			mpf_neg(dtmp, omega);
			add_cmul_mpfvector(local_vec[4], local_vec[3], dtmp, local_vec[5]);
			add_cmul_mpfvector(local_vec[3], local_vec[1], beta, local_vec[4]);
		}
		/* precondition */
		
		/* v := Apt */
		_mpi_mul_mpfmatrix_mpfvec(local_vec[5], local_a,  local_vec[3], big_vec, comm);

		_mpi_ip_mpfvector(alpha_den, local_vec[2], local_vec[5], comm);
		if(mpf_cmp_ui(alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(MPFBiCGSTAB, %d)\n", times);
			return -2;
		}
		mpf_div(alpha, rho, alpha_den);

		/* s = r - alpha v */
		mpf_neg(dtmp, alpha);
		add_cmul_mpfvector(local_vec[6], local_vec[1], dtmp, local_vec[5]);

		/* Stopping Criteria */
		// norm2_mpfvector(dtmp, local_vec[6]);
		_mpi_ip_mpfvector(dtmp, local_vec[6], local_vec[6], comm);
		mpf_sqrt(dtmp, dtmp);

		mpf_mul(dtmp1, reps, init_resnorm);
		mpf_add(dtmp1, dtmp1, aeps);
		if(mpf_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			/* x = x + alpha pt */
			add_cmul_mpfvector(local_vec[0], local_vec[0], alpha, local_vec[3]);

			subst_mpfvector(local_answer, local_vec[0]);
			return times;
		}

		/* precondition */

		_mpi_mul_mpfmatrix_mpfvec(local_vec[8], local_a,  local_vec[6], big_vec, comm);

		/* omega = (t, s) / (t, t) */
		_mpi_ip_mpfvector(omega_den, local_vec[8], local_vec[8], comm);
		if(mpf_cmp_ui(omega_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Omega is zero!(MPFBiCGSTAB, %d)\n", times);
			return -3;
		}
		_mpi_ip_mpfvector(omega, local_vec[8], local_vec[6], comm);
		if(mpf_cmp_ui(omega, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Omega is zero!(MPFBiCGSTAB, %d)\n", times);
			return -4;
		}
		mpf_div(omega, omega, omega_den);

		/* x = x + alpha pt + omega st */
		add_cmul_mpfvector(local_vec[4], local_vec[0], alpha, local_vec[3]);
		add_cmul_mpfvector(local_vec[0], local_vec[4], omega, local_vec[6]);

		/* residual */
		mpf_neg(dtmp, omega);
		add_cmul_mpfvector(local_vec[1], local_vec[6], dtmp, local_vec[8]);

		_mpi_ip_mpfvector(beta_num, local_vec[1], local_vec[1], comm);

		/* Stopping Criteria */
		mpf_sqrt(dtmp, beta_num);
		mpf_mul(dtmp1, reps, init_resnorm);
		mpf_add(dtmp1, dtmp1, aeps);
		if(mpf_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_mpfvector(local_answer, local_vec[0]);
			return times;
		}

		mpf_set(old_rho, rho);
	}

	/* Not converge */
	subst_mpfvector(local_answer, local_vec[0]);

	/* free local_vec[0]..[3]; */
	for(i = 0; i < 9; i++)
		free_mpfvector(local_vec[i]);
	free_mpfvector(big_vec);

	mpf_clear(alpha); mpf_clear(alpha_num); mpf_clear(alpha_den);
	mpf_clear(rho); mpf_clear(old_rho);
	mpf_clear(beta); mpf_clear(beta_num);
	mpf_clear(omega); mpf_clear(omega_den);
	mpf_clear(dtmp); mpf_clear(dtmp1); mpf_clear(init_resnorm);

	fprintf(stderr, "Not converge!(MPFBiCGSTAB, %d)\n", times);
	return -5;

}

/************************************************************/
/*                                                          */
/* GPBiCG(Generalized Product Bi-CG) Method for Real Matrix */
/*                                 (Multi-Precision)        */
/*                                                          */
/*                 ver. 0.0 2004.11.06 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
long int _mpi_MPFGPBiCG(MPFVector local_answer, MPFMatrix local_a[], MPFVector local_b, mpf_t reps, mpf_t aeps, long int maxtimes, long int dim, MPI_Comm comm)
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       MPFVector answer: Solution for Ax = b              */
/*       MPFMatrix a: Coefficient matrix A                  */
/*                                       (given by user)    */
/*       MPFVector b: Constant vector b   (given by user)   */
/*       mpf_t reps: Relative tolerance (given by user)     */
/*       mpf_t aeps: Absolute tolerance (given by user)     */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       MPFVector answer: Solution for Ax = b              */
/*                                                          */
/* ERRORS                                                   */
/* Positive value ... Normal : Iterative Times              */
/*      -1 ... Rho is zero.                                 */
/*      -2 ... Denominator of Alpha is zero.                */
/*      -3 ... (Reserved)                                   */
/*      -4 ... (Reserved)                                   */
/*      -5 ... Not Converge.                                */
/*                                                          */
/************************************************************/
{
	long int i, j, times, local_dim, d_dim[MPI_GMP_MAXPROCS];
	unsigned long prec;
	mpf_t dtmp, dtmp1;
	mpf_t alpha, alpha_num, alpha_den;
	mpf_t beta, beta_num;
	mpf_t rho, old_rho;
	mpf_t mu[5], tau, zeta, eta;
	mpf_t init_resnorm;
	MPFVector local_vec[12], big_vec; /* Temporary Vectors */

	int myrank, num_procs;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);

	prec = local_answer->prec;

/* Set initial value */
	mpf_init2(alpha, prec);
	mpf_init2(alpha_num, prec);
	mpf_init2(alpha_den, prec);
	mpf_init2(rho, prec);
	mpf_init2(old_rho, prec);
	mpf_init2(beta, prec);
	mpf_init2(beta_num, prec);
	mpf_init2(dtmp, prec);
	mpf_init2(dtmp1, prec);
	mpf_init2(tau, prec); mpf_init2(zeta, prec); mpf_init2(eta, prec);
	for(i = 0; i < 5; i++)
		mpf_init2(mu[i], prec);
	mpf_init2(init_resnorm, prec);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);
	for(i = 0; i < 12; i++)
		local_vec[i] = init2_mpfvector(local_dim, prec);
	big_vec = init2_mpfvector(num_procs * local_dim, prec);

	/* local_vec[0] ... approximation of solution */
	/* local_vec[1] ... residual : b - a * local_vec[0] */
	/* local_vec[2] ... (b - a * local_vec[0])^T */
	/* local_vec[3] ... p */
	/* local_vec[4] ... q */
	/* local_vec[5] ... s */
	/* local_vec[6] ... t */
	/* local_vec[7] ... u */
	/* local_vec[8] ... v */
	/* local_vec[9] ... w */
	/* local_vec[10]... y */
	/* local_vec[11]... z */

	subst_mpfvector(local_vec[1], local_b); 
	subst_mpfvector(local_vec[2], local_b);

	_mpi_ip_mpfvector(beta_num, local_vec[1], local_vec[1], comm);
	mpf_sqrt(init_resnorm, beta_num);

	mpf_set_ui(old_rho, 0UL);
	mpf_set_ui(rho, 0UL);

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		_mpi_ip_mpfvector(rho, local_vec[2], local_vec[1], comm);

		if(mpf_cmp_ui(rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(MPFGPBiCG, %d)\n", times);
			return -1;
		}

		if(times == 0)
		{
			/*
				p = r;
				q = A * p;
				alpha = rho / (r0^t, q);
				t = r - alpha * q;
				v = A * t;
				y = alpha * q - r;
				mu2 = v' * t;
				mu5 = v' * v;
				zeta = mu2 / mu5;
				eta = 0;
				u = zeta * q; */
			subst_mpfvector(local_vec[3], local_vec[1]);
			_mpi_mul_mpfmatrix_mpfvec(local_vec[4], local_a,  local_vec[3], big_vec, comm);
			_mpi_ip_mpfvector(alpha_den, local_vec[2], local_vec[4], comm);
			if(mpf_cmp_ui(alpha_den, 0UL) == 0)
			{
				fprintf(stderr, "Denominator of Alpha is zero!(MPFGPBiCG, %d)\n", times);
				return -2;
			}
			mpf_div(alpha, rho, alpha_den);

			mpf_neg(dtmp, alpha);
			add_cmul_mpfvector(local_vec[6], local_vec[1], dtmp, local_vec[4]);
			_mpi_mul_mpfmatrix_mpfvec(local_vec[8], local_a,  local_vec[6], big_vec, comm);
			cmul_mpfvector(local_vec[10], alpha, local_vec[4]);
			sub_mpfvector(local_vec[10], local_vec[10], local_vec[1]);
			_mpi_ip_mpfvector(mu[1], local_vec[8], local_vec[6], comm);
			_mpi_ip_mpfvector(mu[4], local_vec[8], local_vec[8], comm);
			if(mpf_cmp_ui(mu[4], 0UL) == 0)
			{
				fprintf(stderr, "Denominator of mu[4] is zero!(MPFGPBiCG, %d)\n", times);
				return -3;
			}
			mpf_div(zeta, mu[1], mu[4]);
			mpf_set_ui(eta, 0UL);
			cmul_mpfvector(local_vec[7], zeta, local_vec[4]);
		}
		else
		{
			/*
			k_beta = (rho / old_rho) * (alpha / zeta);
			w = v + k_beta * q;
			p = ret_res + k_beta * (p - u);
			q = A * p;
			alpha = rho / (ret_res_c' * q);
			s = t - ret_res;
			t = ret_res - alpha * q;
			v = A * t;
			y = s - alpha * (w - q);
			mu1 = y' * y;
			mu2 = v' * t;
			mu3 = y' * t;
			mu4 = v' * y;
			mu5 = v' * v;
			tau = mu5 * mu1 - mu4 * mu4;
			zeta = (mu1 * mu2 - mu3 * mu4) / tau;
			eta = (mu5 * mu3 - mu4 * mu2) / tau;
			u = zeta * q + eta * (s + k_beta * u); */

			mpf_div(beta, rho, old_rho);
			mpf_div(dtmp, alpha, zeta);
			mpf_mul(beta, beta, dtmp);

			add_cmul_mpfvector(local_vec[9], local_vec[8], beta, local_vec[4]);
			sub_mpfvector(local_vec[3], local_vec[3], local_vec[7]);
			add_cmul_mpfvector(local_vec[3], local_vec[1], beta, local_vec[3]);
			_mpi_mul_mpfmatrix_mpfvec(local_vec[4], local_a,  local_vec[3], big_vec, comm);
			_mpi_ip_mpfvector(alpha_den, local_vec[2], local_vec[4], comm);
			if(mpf_cmp_ui(alpha_den, 0UL) == 0)
			{
				fprintf(stderr, "Denominator of Alpha is zero!(MPFBiCG, %d)\n", times);
				return -2;
			}
			mpf_div(alpha, rho, alpha_den);
			sub_mpfvector(local_vec[5], local_vec[6], local_vec[1]);
			mpf_neg(dtmp, alpha);
			add_cmul_mpfvector(local_vec[6], local_vec[1], dtmp, local_vec[4]);
			_mpi_mul_mpfmatrix_mpfvec(local_vec[8], local_a,  local_vec[6], big_vec, comm);
			sub_mpfvector(local_vec[10], local_vec[9], local_vec[4]);

			mpf_neg(dtmp, alpha);
			add_cmul_mpfvector(local_vec[10], local_vec[5], dtmp, local_vec[10]);
			_mpi_ip_mpfvector(mu[0], local_vec[10], local_vec[10], comm);
			_mpi_ip_mpfvector(mu[1], local_vec[8], local_vec[6], comm);
			_mpi_ip_mpfvector(mu[2], local_vec[10], local_vec[6], comm);
			_mpi_ip_mpfvector(mu[3], local_vec[8], local_vec[10], comm);
			_mpi_ip_mpfvector(mu[4], local_vec[8], local_vec[8], comm);

			/* tau = mu[4] * mu[0] - mu[3] * mu[3]; */
			mpf_mul(dtmp, mu[4], mu[0]);
			mpf_mul(dtmp1, mu[3], mu[3]);
			mpf_sub(tau, dtmp, dtmp1);

			if(mpf_cmp_ui(tau, 0UL) == 0)
			{
				fprintf(stderr, "Denominator of Tau is zero!(MPFGPBiCG, %d)\n", times);
				return -3;
			}

			/* zeta = (mu[0] * mu[1] - mu[2] * mu[3]) / tau; */
			mpf_mul(dtmp, mu[0], mu[1]);
			mpf_mul(dtmp1, mu[2], mu[3]);
			mpf_sub(zeta, dtmp, dtmp1);
			mpf_div(zeta, zeta, tau);

			/* eta = (mu[4] * mu[2] - mu[3] * mu[1]) / tau; */
			mpf_mul(dtmp, mu[4], mu[2]);
			mpf_mul(dtmp1, mu[3], mu[1]);
			mpf_sub(eta, dtmp, dtmp1);
			mpf_div(eta, eta, tau);

			/* u = zeta * q + eta * (s + k_beta * u); */
			add_cmul_mpfvector(local_vec[7], local_vec[5], beta, local_vec[7]);
			cmul_mpfvector(local_vec[7], eta, local_vec[7]);
			add_cmul_mpfvector(local_vec[7], local_vec[7], zeta, local_vec[4]);
		}
		/* z = zeta * ret_res + eta * z - alpha * u; */
		cmul2_mpfvector(local_vec[11], eta);
		add_cmul_mpfvector(local_vec[11], local_vec[11], zeta, local_vec[1]);
		mpf_neg(dtmp, alpha);
		add_cmul_mpfvector(local_vec[11], local_vec[11], dtmp, local_vec[7]);

		/* x = x + alpha p + z */
		add_cmul_mpfvector(local_vec[0], local_vec[0], alpha, local_vec[3]);
		add2_mpfvector(local_vec[0], local_vec[11]);

		/* residual */
		/* r = t - eta * y - zeta * v; */
		mpf_neg(dtmp, eta);
		add_cmul_mpfvector(local_vec[1], local_vec[6], dtmp, local_vec[10]);
		mpf_neg(dtmp, zeta);
		add_cmul_mpfvector(local_vec[1], local_vec[1], dtmp, local_vec[8]);

		_mpi_ip_mpfvector(beta_num, local_vec[1], local_vec[1], comm);

		/* Stopping Criteria */
		mpf_sqrt(dtmp, beta_num);
		mpf_mul(dtmp1, reps, init_resnorm);
		mpf_add(dtmp1, dtmp1, aeps);
		if(mpf_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_mpfvector(local_answer, local_vec[0]);
			return times;
		}
		if(myrank == 0)
		{
			fprintf(stderr, "%d ");
			// dtmp = ||rk||/||r0||2
			mpf_div(dtmp, dtmp, init_resnorm);
			mpf_out_str(stderr, 10, 5, dtmp);
		}

		if(mpf_cmp_ui(zeta, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Zeta is zero!(MPFGPBiCG, %d)\n", times);
			return -4;
		}

		mpf_set(old_rho, rho);
	}

	/* Not converge */
	subst_mpfvector(local_answer, local_vec[0]);

	/* free local_vec[0]..[3]; */
	for(i = 0; i < 12; i++)
		free_mpfvector(local_vec[i]);
	free_mpfvector(big_vec);

	mpf_clear(alpha); mpf_clear(alpha_num); mpf_clear(alpha_den);
	mpf_clear(rho); mpf_clear(old_rho);
	mpf_clear(beta); mpf_clear(beta_num);
	mpf_clear(tau); mpf_clear(zeta); mpf_clear(eta);
	for(i = 0; i < 5; i++)
		mpf_clear(mu[i]);
	mpf_clear(dtmp); mpf_clear(dtmp1); mpf_clear(init_resnorm);

	fprintf(stderr, "Not converge!(MPFGPBiCG, %d)\n", times);
	return -5;

}

#endif
