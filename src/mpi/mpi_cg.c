/********************************************************************************/
/* mpi_cg.c:                                                                    */
/* Copyright (C) 2003-2011 Tomonori Kouya, All rights reserved.                 */
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
/*************************************************/
/* DCG, MPFCG                                    */
/*************************************************/
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

long int _mpi_DCG(DVector local_answer, DMatrix local_a[], DVector local_b, double reps, double aeps, long int maxtimes, long int dim, MPI_Comm comm)
{
	long int i, j, times, local_dim, d_dim[MPI_GMP_MAXPROCS];
	double alpha, alpha_num, alpha_den;
	double beta, beta_num, beta_den;
	double dtmp, init_resnorm;
	DVector local_vec[4], big_vec; /* Temporary Vectors */

	int myrank, num_procs;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);

/* Set initial value */
	for(i = 0; i < 4; i++)
		local_vec[i] = init_dvector(local_dim);
	big_vec = init_dvector(local_dim * num_procs);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... p */
	/* vec[2] ... residual : b - a * vec[0] */
	/* vec[3] ... a * p */
	subst_dvector(local_vec[1], local_b); 
	subst_dvector(local_vec[2], local_b);

	beta_num = _mpi_ip_dvector(local_vec[2], local_vec[2], comm);
	init_resnorm = sqrt(beta_num);

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* Ap */
		_mpi_mul_dmatrix_dvec(local_vec[3], local_a, local_vec[1], big_vec, comm);

		/* alpha = alpha_num / alpha_den */
		alpha_den = _mpi_ip_dvector(local_vec[1], local_vec[3], comm);
		alpha_num = _mpi_ip_dvector(local_vec[1], local_vec[2], comm);
		if(alpha_den == 0.0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(DCG, %d)\n", times);
			return -1;
		}
		if(alpha_num == 0.0)
		{
			fprintf(stderr, "Numerator of Alpha is zero!(DCG, %d)\n", times);
			return -2;
		}

		alpha = alpha_num / alpha_den;

		/* x = x + alpha p */
		//for(i = 0; i < local_dim; i++)
		for(i = 0; i < d_dim[myrank]; i++)
		{
			dtmp = get_dvector_i(local_vec[0], i) + alpha * get_dvector_i(local_vec[1], i);
			set_dvector_i(local_vec[0], i, dtmp);
		}

		/* residual */
		beta_den = beta_num;
		//for(i = 0; i < local_dim; i++)
		for(i = 0; i < d_dim[myrank]; i++)
		{
			dtmp = get_dvector_i(local_vec[2], i) - alpha * get_dvector_i(local_vec[3], i);
			set_dvector_i(local_vec[2], i, dtmp);
		}
		beta_num = _mpi_ip_dvector(local_vec[2], local_vec[2], comm);

		/* Stopping Criteria */
		dtmp = sqrt(beta_num);
		if(dtmp < aeps + reps * init_resnorm)
		{
			subst_dvector(local_answer, local_vec[0]);
			return times;
		}

		if(beta_den == 0.0)
		{
			fprintf(stderr, "Denominator of Beta is zero!(DCG, %d)\n", times);
			return -3;
		}
		if(beta_num == 0.0)
		{
			fprintf(stderr, "Numerator of Beta is zero!(DCG, %d)\n", times);
			return -4;
		}

		/* beta */
		beta = beta_num / beta_den;

		/* p */
		//for(i = 0; i < local_dim; i++)
		for(i = 0; i < d_dim[myrank]; i++)
		{
			dtmp = get_dvector_i(local_vec[2], i) + beta * get_dvector_i(local_vec[1], i);
			set_dvector_i(local_vec[1], i, dtmp);
		}
	}

	/* Not converge */
	subst_dvector(local_answer, local_vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 4; i++)
		free_dvector(local_vec[i]);
	free_dvector(big_vec);

	fprintf(stderr, "Not converge!(DCG, %d)\n", times);
	return -5;

}

#ifdef USE_GMP
long int _mpi_MPFCG(MPFVector local_answer, MPFMatrix local_a[], MPFVector local_b, mpf_t reps, mpf_t aeps, long int maxtimes, long int dim, MPI_Comm comm)
{
	unsigned long int prec;
	long int i, j, times, local_dim, d_dim[MPI_GMP_MAXPROCS];
	mpf_t alpha, alpha_num, alpha_den;
	mpf_t beta, beta_num, beta_den;
	mpf_t dtmp, dtmp1, init_resnorm;
	MPFVector local_vec[4], big_vec; /* Temporary Vectors */

	int myrank, num_procs;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);

	prec = local_answer->prec;

/* Set initial value */
	mpf_init2(alpha, prec);
	mpf_init2(alpha_num, prec);
	mpf_init2(alpha_den, prec);
	mpf_init2(beta, prec);
	mpf_init2(beta_num, prec);
	mpf_init2(beta_den, prec);
	mpf_init2(dtmp, prec);
	mpf_init2(dtmp1, prec);
	mpf_init2(init_resnorm, prec);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);
	for(i = 0; i < 4; i++)
		local_vec[i] = init2_mpfvector(local_dim, prec);
	big_vec = init2_mpfvector(num_procs * local_dim, prec);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... p */
	/* vec[2] ... residual : b - a * vec[0] */
	/* vec[3] ... a * p */
	subst_mpfvector(local_vec[1], local_b); 
	subst_mpfvector(local_vec[2], local_b);

	_mpi_ip_mpfvector(beta_num, local_vec[2], local_vec[2], comm);
	mpf_sqrt(init_resnorm, beta_num);

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* Ap */
		_mpi_mul_mpfmatrix_mpfvec(local_vec[3], local_a, local_vec[1], big_vec, comm);


		/* alpha = alpha_num / alpha_den */
		_mpi_ip_mpfvector(alpha_den, local_vec[1], local_vec[3], comm);
		_mpi_ip_mpfvector(alpha_num, local_vec[1], local_vec[2], comm);

		if(mpf_cmp_ui(alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(MPFCG, %d)\n", times);
			return -1;
		}
		if(mpf_cmp_ui(alpha_num, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Alpha is zero!(MPFCG, %d)\n", times);
			return -2;
		}

		mpf_div(alpha, alpha_num, alpha_den);

		/* x = x + alpha p */
		//for(i = 0; i < local_dim; i++)
		for(i = 0; i < d_dim[myrank]; i++)
		{
			mpf_mul(dtmp, alpha, get_mpfvector_i(local_vec[1], i));
			mpf_add(dtmp, get_mpfvector_i(local_vec[0], i), dtmp);
			set_mpfvector_i(local_vec[0], i, dtmp);
		}

		/* residual */
		mpf_set(beta_den, beta_num);
		//for(i = 0; i < local_dim; i++)
		for(i = 0; i < d_dim[myrank]; i++)
		{
			mpf_mul(dtmp, alpha, get_mpfvector_i(local_vec[3], i));
			mpf_sub(dtmp, get_mpfvector_i(local_vec[2], i), dtmp);
			set_mpfvector_i(local_vec[2], i, dtmp);
		}
		_mpi_ip_mpfvector(beta_num, local_vec[2], local_vec[2], comm);

		/* Stopping Criteria */
		mpf_sqrt(dtmp, beta_num);
		mpf_mul(dtmp1, reps, init_resnorm);
		mpf_add(dtmp1, dtmp1, aeps);
		if(mpf_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_mpfvector(local_answer, local_vec[0]);
			return times;
		}

		if(mpf_cmp_ui(beta_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Beta is zero!(MPFCG, %d)\n", times);
			return -3;
		}
		if(mpf_cmp_ui(beta_num, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Beta is zero!(MPFCG, %d)\n", times);
			return -4;
		}

		/* beta */
		mpf_div(beta, beta_num, beta_den);

		/* p */
		//for(i = 0; i < local_dim; i++)
		for(i = 0; i < d_dim[myrank]; i++)
		{
			mpf_mul(dtmp, beta, get_mpfvector_i(local_vec[1], i));
			mpf_add(dtmp, get_mpfvector_i(local_vec[2], i), dtmp);
			set_mpfvector_i(local_vec[1], i, dtmp);
		}
	}

	/* Not converge */
	subst_mpfvector(local_answer, local_vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 4; i++)
		free_mpfvector(local_vec[i]);
	free_mpfvector(big_vec);

	mpf_clear(alpha); mpf_clear(alpha_num); mpf_clear(alpha_den);
	mpf_clear(beta); mpf_clear(beta_num); mpf_clear(beta_den);
	mpf_clear(dtmp); mpf_clear(dtmp1); mpf_clear(init_resnorm);

	fprintf(stderr, "Not converge!(MPFCG, %d)\n", times);
	return -5;

}
#endif
