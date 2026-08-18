/********************************************************************************/
/* krylov_omp.c:                                                                */
/* Copyright (C) 2024 Tomonori Kouya                                            */
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
/* stdio.h */
#ifndef _STDIO_H
#include <stdio.h>
#endif

/* math.h */
#ifndef _MATH_H
#include <math.h>
#endif

//#include "bnc.h"
#include "bncmatmul.h"
#include "bncomp.h"

//#ifdef USE_SPARSE_VERSION
/* Sparse Matrix */
//  #include "bncsparse.h"
//#endif

// Copy & Paste
#ifdef USE_OMP_VERSION
#else // USE_OMP_VERSION
#endif // USE_OMP_VERSION

#ifndef USE_SPARSE_D_VERSION

/************************************************************/
/*                                                          */
/* Bi-Conjugate-Gradient Method for Real Matrix             */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 2024-10-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
  long int _bncomp_DBiCG_sp(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes)
#else // USE_SPARSE_VERSION
  long int _bncomp_DBiCG(DVector answer, DMatrix a, DVector b, double reps, double aeps, long int maxtimes)
#endif // USE_SPARSE_VERSION
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
	long int i, j, times, dim, return_val; // Fix!
	double alpha, alpha_num, alpha_den;
	double beta, beta_num;
	double rho, old_rho;
	double dtmp, init_resnorm;
	DVector vec[7]; /* Temporary Vectors */

	dim = answer->dim;

/* Set initial value */
	for(i = 0; i < 7; i++)
		vec[i] = init_dvector(dim);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0]  == w*/
	/* vec[2] ... (b - a * vec[0])^T  == wt */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... z */
	/* vec[6] ... z^T */

	subst_dvector(vec[1], b); 
	subst_dvector(vec[2], b);

	beta_num = ip_dvector(vec[1], vec[1]);
	init_resnorm = sqrt(beta_num);

	old_rho = 0.0;
	rho = 0.0;
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		rho = ip_dvector(vec[2], vec[1]);

		if(rho == 0.0)
		{
			fprintf(stderr, "Rho is zero!(DBiCG, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* p := w, pt := wt */
			subst_dvector(vec[3], vec[1]);
			subst_dvector(vec[4], vec[2]);
		}
		else
		{
			beta = rho / old_rho;

			/* p := w + beta p, pt := wt + beta * pt */
			add_cmul_dvector(vec[3], vec[1], beta, vec[3]);
			add_cmul_dvector(vec[4], vec[2], beta, vec[4]);
		}

		/* z := Ap, zt := A^T pt */
#ifdef USE_SPARSE_VERSION
		_bncomp_mul_drsmatrix_dvec(vec[5], a, vec[3]);
		_bncomp_mul_drsmatrixt_dvec(vec[6], a, vec[4]);
#else // USE_SPARSE_VERSION
		_bncomp_mul_dmatrix_dvec(vec[5], a, vec[3]);
		_bncomp_mul_dmatrixt_dvec(vec[6], a, vec[4]);
#endif // USE_SPARSE_VERSION

		alpha_den = ip_dvector(vec[4], vec[5]);
		if(alpha_den == 0.0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(DBiCG, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		alpha = rho / alpha_den;

		/* x = x + alpha p */
		add_cmul_dvector(vec[0], vec[0], alpha, vec[3]);

		/* residual */
		add_cmul_dvector(vec[1], vec[1], -alpha, vec[5]);
		add_cmul_dvector(vec[2], vec[2], -alpha, vec[6]);

		beta_num = ip_dvector(vec[1], vec[1]);

		/* Stopping Criteria */
		dtmp = sqrt(beta_num);
		if(dtmp <= aeps + reps * init_resnorm)
		{
			subst_dvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		old_rho = rho;
	}

	/* Not converge */
	subst_dvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 7; i++)
		free_dvector(vec[i]);

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(DBiCG, %ld)\n", times);
		return_val = -5; // Fix!
	}

	return return_val; // Fix!

}

/************************************************************/
/*                                                          */
/*               CGS Method for Real Matrix                 */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 2024-10-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
  long int _bncomp_DCGS_sp(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes)
#else // USE_SPARSE_VERSION
  long int _bncomp_DCGS(DVector answer, DMatrix a, DVector b, double reps, double aeps, long int maxtimes) // avoid name confict with DCG of IMKL
#endif // USE_SPARSE_VERSION
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
	long int i, j, times, dim, return_val;
	double alpha, alpha_num, alpha_den;
	double beta, beta_num;
	double rho, old_rho;
	double dtmp, init_resnorm;
	DVector vec[9]; /* Temporary Vectors */

	dim = answer->dim;

/* Set initial value */
	for(i = 0; i < 9; i++)
		vec[i] = init_dvector(dim);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0] */
	/* vec[2] ... (b - a * vec[0])^T */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... u */
	/* vec[6] ... u^T */
	/* vec[7] ... q */
	/* vec[8] ... v^T */

	subst_dvector(vec[1], b); 
	subst_dvector(vec[2], b);

	beta_num = ip_dvector(vec[1], vec[1]);
	init_resnorm = sqrt(beta_num);

	old_rho = 0.0;
	rho = 0.0;
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		rho = ip_dvector(vec[2], vec[1]);

		if(rho == 0.0)
		{
			fprintf(stderr, "Rho is zero!(DCGS, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* u := rt, p := u */
			subst_dvector(vec[5], vec[1]);
			subst_dvector(vec[3], vec[5]);
		}
		else
		{
			beta = rho / old_rho;

			/* u := r + beta q, p := u + beta * (q + beta p) */
			add_cmul_dvector(vec[5], vec[1], beta, vec[7]);
			add_cmul_dvector(vec[8], vec[7], beta, vec[3]);
			add_cmul_dvector(vec[3], vec[5], beta, vec[8]);
		}
		/* precondition */
		/* pt = linsolve(pc_K, -p) */
		
		/* vt := Apt */
#ifdef USE_SPARSE_VERSION
		_bncomp_mul_drsmatrix_dvec(vec[8], a, vec[3]);
#else // USE_SPARSE_VERSION
		_bncomp_mul_dmatrix_dvec(vec[8], a, vec[3]);
#endif // USE_SPARSE_VERSION

		alpha_den = ip_dvector(vec[2], vec[8]);
		if(alpha_den == 0.0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(DCGS, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		alpha = rho / alpha_den;

		/* q = u - alpha  vt */
		add_cmul_dvector(vec[7], vec[5], -alpha, vec[8]);

		/* ut = u + q */
		add_dvector(vec[6], vec[5], vec[7]);

		/* x = x + alpha ut */
		add_cmul_dvector(vec[0], vec[0], alpha, vec[6]);

		/* residual */
#ifdef USE_SPARSE_VERSION
		_bncomp_mul_drsmatrix_dvec(vec[8], a, vec[6]);
#else // USE_SPARSE_VERSION
		_bncomp_mul_dmatrix_dvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION
		add_cmul_dvector(vec[1], vec[1], -alpha, vec[8]);

		beta_num = ip_dvector(vec[1], vec[1]);

		/* Stopping Criteria */
		dtmp = sqrt(beta_num);
		if(dtmp <= aeps + reps * init_resnorm)
		{
			subst_dvector(answer, vec[0]);
			return_val = times; // Fix!
			break;
		}

		old_rho = rho;
	}

	/* Not converge */
	subst_dvector(answer, vec[0]);

	/* free vec[0]..[8]; */
	for(i = 0; i < 9; i++)
		free_dvector(vec[i]);

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(DCGS, %ld)\n", times);
		return_val= -5;
	}

	return return_val;

}

/************************************************************/
/*                                                          */
/*             BiCGSTAB Method for Real Matrix              */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 2024-10-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
  long int _bncomp_DBiCGSTAB_sp(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes)
#else // USE_SPARSE_VERSION
  long int _bncomp_DBiCGSTAB(DVector answer, DMatrix a, DVector b, double reps, double aeps, long int maxtimes)
#endif // USE_SPARSE_VERSION
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
	long int i, j, times, dim, return_val;
	double alpha, alpha_num, alpha_den;
	double beta, beta_num;
	double rho, old_rho;
	double omega, omega_den;
	double dtmp, init_resnorm;
	DVector vec[9]; /* Temporary Vectors */

	dim = answer->dim;

/* Set initial value */
	for(i = 0; i < 9; i++)
		vec[i] = init_dvector(dim);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0]  == w*/
	/* vec[2] ... (b - a * vec[0])^T  == wt */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... v */
	/* vec[6] ... s */
	/* vec[7] ... s^T */
	/* vec[8] ... t */

	subst_dvector(vec[1], b); 
	subst_dvector(vec[2], b);

	beta_num = ip_dvector(vec[1], vec[1]);
	init_resnorm = sqrt(beta_num);

	old_rho = 0.0;
	rho = 0.0;
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		rho = ip_dvector(vec[2], vec[1]);

		if(rho == 0.0)
		{
			fprintf(stderr, "Rho is zero!(DBiCG, %ld)\n", times);
			return_val= -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* p := r */
			subst_dvector(vec[3], vec[1]);
		}
		else
		{
			beta = (rho / old_rho) * (alpha / omega);

			/* p := r + beta (p - omega v) */
			add_cmul_dvector(vec[4], vec[3], -omega, vec[5]);
			add_cmul_dvector(vec[3], vec[1], beta, vec[4]);
		}
		/* precondition */
		
		/* v := Apt */
#ifdef USE_SPARSE_VERSION
		_bncomp_mul_drsmatrix_dvec(vec[5], a, vec[3]);
#else // USE_SPARSE_VERSION
		_bncomp_mul_dmatrix_dvec(vec[5], a, vec[3]);
#endif // USE_SPARSE_VERSION

		alpha_den = ip_dvector(vec[2], vec[5]);
		if(alpha_den == 0.0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(DBiCGSTAB, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		alpha = rho / alpha_den;

		/* s = r - alpha v */
		add_cmul_dvector(vec[6], vec[1], -alpha, vec[5]);

		/* Stopping Criteria */
		dtmp = norm2_dvector(vec[6]);
		if(dtmp <= aeps + reps * init_resnorm)
		{
			/* x = x + alpha pt */
			add_cmul_dvector(vec[0], vec[0], alpha, vec[3]);

			subst_dvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		/* precondition */
#ifdef USE_SPARSE_VERSION
		_bncomp_mul_drsmatrix_dvec(vec[8], a, vec[6]);
#else // USE_SPARSE_VERSION
		_bncomp_mul_dmatrix_dvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION
		/* omega = (t, s) / (t, t) */
		omega_den = ip_dvector(vec[8], vec[8]);
		if(omega_den == 0)
		{
			fprintf(stderr, "Denominator of Omega is zero!(DBiCGSTAB, %ld)\n", times);
			return_val = -3; // Fix!
			break;
		}
		omega = ip_dvector(vec[8], vec[6]);
		if(omega == 0)
		{
			fprintf(stderr, "Numerator of Omega is zero!(DBiCGSTAB, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}
		omega = omega / omega_den;

		/* x = x + alpha pt + omega st */
		add_cmul_dvector(vec[4], vec[0], alpha, vec[3]);
		add_cmul_dvector(vec[0], vec[4], omega, vec[6]);

		/* residual */
		add_cmul_dvector(vec[1], vec[6], -omega, vec[8]);

		beta_num = ip_dvector(vec[1], vec[1]);

		/* Stopping Criteria */
		dtmp = sqrt(beta_num);
		if(dtmp <= aeps + reps * init_resnorm)
		{
			subst_dvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		old_rho = rho;
	}

	/* Not converge */
	subst_dvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 9; i++)
		free_dvector(vec[i]);

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(DBiCGSTAB, %ld)\n", times);
		return_val = -5;
	}

	return return_val; // Fix!

}

/************************************************************/
/*                                                          */
/* GPBiCG(Generalized Product Bi-CG) Method for Real Matrix */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 2024-10-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
  long int _bncomp_DGPBiCG_sp(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes)
#else // USE_SPARSE_VERSION
  long int _bncomp_DGPBiCG(DVector answer, DMatrix a, DVector b, double reps, double aeps, long int maxtimes)
#endif // USE_SPARSE_VERSION
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
	long int i, j, times, dim, return_val;
	double alpha, alpha_num, alpha_den;
	double beta, beta_num;
	double rho, old_rho;
	double mu[5], tau, zeta, eta;
	double dtmp, init_resnorm;
	DVector vec[12]; /* Temporary Vectors */

	dim = answer->dim;

/* Set initial value */
	for(i = 0; i < 12; i++)
		vec[i] = init_dvector(dim);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0] */
	/* vec[2] ... (b - a * vec[0])^T */
	/* vec[3] ... p */
	/* vec[4] ... q */
	/* vec[5] ... s */
	/* vec[6] ... t */
	/* vec[7] ... u */
	/* vec[8] ... v */
	/* vec[9] ... w */
	/* vec[10]... y */
	/* vec[11]... z */

	subst_dvector(vec[1], b); 
	subst_dvector(vec[2], b);

	beta_num = ip_dvector(vec[1], vec[1]);
	init_resnorm = sqrt(beta_num);

	old_rho = 0.0;
	rho = 0.0;
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		rho = ip_dvector(vec[2], vec[1]);

		if(rho == 0.0)
		{
			fprintf(stderr, "Rho is zero!(DGPBiCG, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
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
			subst_dvector(vec[3], vec[1]);
#ifdef USE_SPARSE_VERSION
			_bncomp_mul_drsmatrix_dvec(vec[4], a, vec[3]);
#else // USE_SPARSE_VERSION
			_bncomp_mul_dmatrix_dvec(vec[4], a, vec[3]);
#endif // USE_SPARSE_VERSION
			alpha_den = ip_dvector(vec[2], vec[4]);
			if(alpha_den == 0.0)
			{
				fprintf(stderr, "Denominator of Alpha is zero!(DGPBiCG, %ld)\n", times);
				return_val = -2; // Fix!
				break; // Fix!
			}
			alpha = rho / alpha_den;
			add_cmul_dvector(vec[6], vec[1], -alpha, vec[4]);
#ifdef USE_SPARSE_VERSION
			_bncomp_mul_drsmatrix_dvec(vec[8], a, vec[6]);
#else // USE_SPARSE_VERSION
			_bncomp_mul_dmatrix_dvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION
			cmul_dvector(vec[10], alpha, vec[4]);
			sub_dvector(vec[10], vec[10], vec[1]);
			mu[1] = ip_dvector(vec[8], vec[6]);
			mu[4] = ip_dvector(vec[8], vec[8]);
			if(mu[4] == 0.0)
			{
				fprintf(stderr, "Denominator of mu[4] is zero!(DGPBiCG, %ld)\n", times);
				return_val = -3; // Fix!
				break; // Fix!
			}
			zeta = mu[1] / mu[4];
			eta = 0.0;
			cmul_dvector(vec[7], zeta, vec[4]);
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

			add_cmul_dvector(vec[9], vec[8], beta, vec[4]);
			sub_dvector(vec[3], vec[3], vec[7]);
			add_cmul_dvector(vec[3], vec[1], beta, vec[3]);
#ifdef USE_SPARSE_VERSION
			_bncomp_mul_drsmatrix_dvec(vec[4], a, vec[3]);
#else // USE_SPARSE_VERSION
			_bncomp_mul_dmatrix_dvec(vec[4], a, vec[3]);
#endif // USE_SPARSE_VERSION
			alpha_den = ip_dvector(vec[2], vec[4]);
			if(alpha_den == 0.0)
			{
				fprintf(stderr, "Denominator of Alpha is zero!(DGPBiCG, %ld)\n", times);
				return_val = -2; // Fix!
				break; // Fix!
			}
			alpha = rho / alpha_den;
			sub_dvector(vec[5], vec[6], vec[1]);
			add_cmul_dvector(vec[6], vec[1], -alpha, vec[4]);
#ifdef USE_SPARSE_VERSION
			_bncomp_mul_drsmatrix_dvec(vec[8], a, vec[6]);
#else // USE_SPARSE_VERSION
			_bncomp_mul_dmatrix_dvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION
			sub_dvector(vec[10], vec[9], vec[4]);
			add_cmul_dvector(vec[10], vec[5], -alpha, vec[10]);
			mu[0] = ip_dvector(vec[10], vec[10]);
			mu[1] = ip_dvector(vec[8], vec[6]);
			mu[2] = ip_dvector(vec[10], vec[6]);
			mu[3] = ip_dvector(vec[8], vec[10]);
			mu[4] = ip_dvector(vec[8], vec[8]);
			tau = mu[4] * mu[0] - mu[3] * mu[3];
			zeta = (mu[0] * mu[1] - mu[2] * mu[3]) / tau;
			eta = (mu[4] * mu[2] - mu[3] * mu[1]) / tau;

			/* u = zeta * q + eta * (s + k_beta * u); */
			add_cmul_dvector(vec[7], vec[5], beta, vec[7]);
			cmul_dvector(vec[7], eta, vec[7]);
			add_cmul_dvector(vec[7], vec[7], zeta, vec[4]);
		}
		/* z = zeta * ret_res + eta * z - alpha * u; */
		cmul2_dvector(vec[11], eta);
		add_cmul_dvector(vec[11], vec[11], zeta, vec[1]);
		add_cmul_dvector(vec[11], vec[11], -alpha, vec[7]);

		/* x = x + alpha p + z */
		add_cmul_dvector(vec[0], vec[0], alpha, vec[3]);
		add2_dvector(vec[0], vec[11]);

		/* residual */
		/* r = t - eta * y - zeta * v; */
		add_cmul_dvector(vec[1], vec[6], -eta, vec[10]);
		add_cmul_dvector(vec[1], vec[1], -zeta, vec[8]);

		beta_num = ip_dvector(vec[1], vec[1]);

		/* Stopping Criteria */
		dtmp = sqrt(beta_num);
		if(dtmp <= aeps + reps * init_resnorm)
		{
			subst_dvector(answer, vec[0]);
			return_val = times; // Fix!
			break;
		}
//		printf("DGPBiCG: %5d %10.3e\n", times, dtmp / init_resnorm);
		if(zeta == 0.0)
		{
			fprintf(stderr, "Denominator of Zeta is zero!(DGPBiCG, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}

		old_rho = rho;
	}

	/* Not converge */
	subst_dvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 12; i++)
		free_dvector(vec[i]);

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(DGPBiCG, %ld)\n", times);
		return_val = -5;
	}

	return return_val;

}

#endif // ifndef USE_SPARSE_D_VERSION


#ifdef USE_GMP

/************************************************************/
/*                                                          */
/*                Bi-Conjugate-Gradient Method              */
/*                                 (Multi-Precision)        */
/*                                                          */
/*                 ver. 0.0 2024-10-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
  long int _bncomp_MPFBiCG_sp(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
#elif USE_SPARSE_D_VERSION
  long int _bncomp_MPFBiCG_sp_d(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
#else // USE_SPARSE_VERSION
  long int _bncomp_MPFBiCG(MPFVector answer, MPFMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
#endif // USE_SPARSE_VERSION
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
	long int i, j, times, dim, return_val;
	mpf_t alpha, alpha_num, alpha_den;
	mpf_t rho, old_rho;
	mpf_t beta, beta_num;
	mpf_t dtmp, dtmp1, init_resnorm;
	MPFVector vec[7]; /* Temporary Vectors */

	dim = answer->dim;
	prec = answer->prec;

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

	for(i = 0; i < 7; i++)
		vec[i] = init2_mpfvector(dim, prec);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0]  == w*/
	/* vec[2] ... (b - a * vec[0])^T  == wt */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... z */
	/* vec[6] ... z^T */

	subst_mpfvector(vec[1], b); 
	subst_mpfvector(vec[2], b);

	ip_mpfvector(beta_num, vec[1], vec[1]);
	mpf_sqrt(init_resnorm, beta_num);
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_mpfvector(rho, vec[2], vec[1]);

		if(mpf_cmp_ui(rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(MPFBiCG, %ld)\n", times);
			return_val = -1; // Fix!
			break;
		}

		if(times == 0)
		{
			/* p := w, pt := wt */
			subst_mpfvector(vec[3], vec[1]);
			subst_mpfvector(vec[4], vec[2]);
		}
		else
		{
			mpf_div(beta, rho, old_rho);

			/* p := w + beta p, pt := wt + beta * pt */
			add_cmul_mpfvector(vec[3], vec[1], beta, vec[3]);
			add_cmul_mpfvector(vec[4], vec[2], beta, vec[4]);
		}
		/* z := Ap, zt := A^T pt */
#ifdef USE_SPARSE_VERSION
		_bncomp_mul_mpfrsmatrix_mpfvec(vec[5], a, vec[3]);
		_bncomp_mul_mpfrsmatrixt_mpfvec(vec[6], a, vec[4]);
#elif USE_SPARSE_D_VERSION
		_bncomp_mul_drsmatrix_mpfvec(vec[5], a, vec[3]);
		_bncomp_mul_drsmatrixt_mpfvec(vec[6], a, vec[4]);
#else // USE_SPARSE_VERSION
		_bncomp_mul_mpfmatrix_mpfvec(vec[5], a, vec[3]);
		_bncomp_mul_mpfmatrixt_mpfvec(vec[6], a, vec[4]);
#endif // USE_SPARSE_VERSION
		ip_mpfvector(alpha_den, vec[4], vec[5]);
		if(mpf_cmp_ui(alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(MPFBiCG, %ld)\n", times);
			return_val = -2; // Fix!
			break;
		}
		mpf_div(alpha, rho, alpha_den);

		/* x = x + alpha p */
		add_cmul_mpfvector(vec[0], vec[0], alpha, vec[3]);

		/* residual */
		mpf_neg(dtmp, alpha);
		add_cmul_mpfvector(vec[1], vec[1], dtmp, vec[5]);
		add_cmul_mpfvector(vec[2], vec[2], dtmp, vec[6]);

		ip_mpfvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		mpf_sqrt(dtmp, beta_num);
		mpf_mul(dtmp1, reps, init_resnorm);
		mpf_add(dtmp1, dtmp1, aeps);
		if(mpf_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_mpfvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		mpf_set(old_rho, rho);
	}

	/* Not converge */
	subst_mpfvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 7; i++)
		free_mpfvector(vec[i]);

	mpf_clear(alpha); mpf_clear(alpha_num); mpf_clear(alpha_den);
	mpf_clear(rho); mpf_clear(old_rho);
	mpf_clear(beta); mpf_clear(beta_num);
	mpf_clear(dtmp); mpf_clear(dtmp1); mpf_clear(init_resnorm);

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(MPFBiCG, %ld)\n", times);
		return_val = -5;
	}

	return return_val; // Fix!

}
/************************************************************/
/*                                                          */
/*               CGS Method for Real Matrix                 */
/*                                 (Multi-Precision)        */
/*                                                          */
/*                 ver. 0.0 2024-10-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
  long int _bncomp_MPFCGS_sp(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
#elif USE_SPARSE_D_VERSION
  long int _bncomp_MPFCGS_sp_d(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
#else // USE_SPARSE_VERSION
  long int _bncomp_MPFCGS(MPFVector answer, MPFMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
#endif // USE_SPARSE_VERSION
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
	long int i, j, times, dim, return_val;
	unsigned long prec;
	mpf_t dtmp, dtmp1;
	mpf_t alpha, alpha_num, alpha_den;
	mpf_t beta, beta_num;
	mpf_t rho, old_rho;
	mpf_t init_resnorm;
	MPFVector vec[9]; /* Temporary Vectors */

	dim = answer->dim;
	prec = answer->prec;

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

	for(i = 0; i < 9; i++)
		vec[i] = init2_mpfvector(dim, prec);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0] */
	/* vec[2] ... (b - a * vec[0])^T */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... u */
	/* vec[6] ... u^T */
	/* vec[7] ... q */
	/* vec[8] ... v^T */

	subst_mpfvector(vec[1], b); 
	subst_mpfvector(vec[2], b);

	ip_mpfvector(beta_num, vec[1], vec[1]);
	mpf_sqrt(init_resnorm, beta_num);

	mpf_set_ui(old_rho, 0UL);
	mpf_set_ui(rho, 0UL);
	return_val = 0;

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_mpfvector(rho, vec[2], vec[1]);

		if(mpf_cmp_ui(rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(MPFCGS, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* u := rt, p := u */
			subst_mpfvector(vec[5], vec[1]);
			subst_mpfvector(vec[3], vec[5]);
		}
		else
		{
			mpf_div(beta, rho, old_rho);

			/* u := r + beta q, p := u + beta * (q + beta p) */
			add_cmul_mpfvector(vec[5], vec[1], beta, vec[7]);
			add_cmul_mpfvector(vec[8], vec[7], beta, vec[3]);
			add_cmul_mpfvector(vec[3], vec[5], beta, vec[8]);
		}
		/* precondition */
		/* pt = linsolve(pc_K, -p) */
		
		/* vt := Apt */
#ifdef USE_SPARSE_VERSION
		_bncomp_mul_mpfrsmatrix_mpfvec(vec[8], a, vec[3]);
#elif USE_SPARSE_D_VERSION
		_bncomp_mul_drsmatrix_mpfvec(vec[8], a, vec[3]);
#else // USE_SPARSE_VERSION
		_bncomp_mul_mpfmatrix_mpfvec(vec[8], a, vec[3]);
#endif // USE_SPARSE_VERSION

		ip_mpfvector(alpha_den, vec[2], vec[8]);
		if(mpf_cmp_ui(alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(MPFCGS, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		mpf_div(alpha, rho, alpha_den);

		/* q = u - alpha  vt */
		mpf_neg(dtmp, alpha);
		add_cmul_mpfvector(vec[7], vec[5], dtmp, vec[8]);

		/* ut = u + q */
		add_mpfvector(vec[6], vec[5], vec[7]);

		/* x = x + alpha ut */
		add_cmul_mpfvector(vec[0], vec[0], alpha, vec[6]);

		/* residual */
#ifdef USE_SPARSE_VERSION
		_bncomp_mul_mpfrsmatrix_mpfvec(vec[8], a, vec[6]);
#elif USE_SPARSE_D_VERSION
		_bncomp_mul_drsmatrix_mpfvec(vec[8], a, vec[6]);
#else // USE_SPARSE_VERSION
		_bncomp_mul_mpfmatrix_mpfvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION
		mpf_neg(dtmp, alpha);
		add_cmul_mpfvector(vec[1], vec[1], dtmp, vec[8]);

		ip_mpfvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		mpf_sqrt(dtmp, beta_num);
		mpf_mul(dtmp1, reps, init_resnorm);
		mpf_add(dtmp1, dtmp1, aeps);
		if(mpf_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_mpfvector(answer, vec[0]);
			return_val = times;
			break;
		}

		mpf_set(old_rho, rho);
	}

	/* Not converge */
	subst_mpfvector(answer, vec[0]);

	/* free vec[0]..[8]; */
	for(i = 0; i < 9; i++)
		free_mpfvector(vec[i]);

	mpf_clear(alpha); mpf_clear(alpha_num); mpf_clear(alpha_den);
	mpf_clear(rho); mpf_clear(old_rho);
	mpf_clear(beta); mpf_clear(beta_num);
	mpf_clear(dtmp); mpf_clear(dtmp1); mpf_clear(init_resnorm);

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(MPFCGS, %ld)\n", times);
		return_val = -5;
	}

	// Fix!
	return return_val;
}

/************************************************************/
/*                                                          */
/*             BiCGSTAB Method for Real Matrix              */
/*                                 (Multi-Precision)        */
/*                                                          */
/*                 ver. 0.0 2024-10-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
  long int _bncomp_MPFBiCGSTAB_sp(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
#elif USE_SPARSE_D_VERSION
  long int _bncomp_MPFBiCGSTAB_sp_d(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
#else // USE_SPARSE_VERSION
  long int _bncomp_MPFBiCGSTAB(MPFVector answer, MPFMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
#endif // USE_SPARSE_VERSION
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
	long int i, j, times, dim, return_val;
	unsigned long prec;
	mpf_t dtmp, dtmp1;
	mpf_t alpha, alpha_num, alpha_den;
	mpf_t beta, beta_num;
	mpf_t rho, old_rho;
	mpf_t omega, omega_den;
	mpf_t init_resnorm;
	MPFVector vec[9]; /* Temporary Vectors */

	dim = answer->dim;
	prec = answer->prec;

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

	for(i = 0; i < 9; i++)
		vec[i] = init2_mpfvector(dim, prec);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0]  == w*/
	/* vec[2] ... (b - a * vec[0])^T  == wt */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... v */
	/* vec[6] ... s */
	/* vec[7] ... s^T */
	/* vec[8] ... t */

	subst_mpfvector(vec[1], b); 
	subst_mpfvector(vec[2], b);

	ip_mpfvector(beta_num, vec[1], vec[1]);
	mpf_sqrt(init_resnorm, beta_num);

	mpf_set_ui(old_rho, 0UL);
	mpf_set_ui(rho, 0UL);
	return_val = 0;

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_mpfvector(rho, vec[2], vec[1]);

		if(mpf_cmp_ui(rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(MPFBiCGSTAB, %ld)\n", times);
			return_val = -1; // Fix!
			break;
		}

		if(times == 0)
		{
			/* p := r */
			subst_mpfvector(vec[3], vec[1]);
		}
		else
		{
			mpf_div(beta, rho, old_rho);
			mpf_div(dtmp, alpha, omega);
			mpf_mul(beta, beta, dtmp);

			/* p := r + beta (p - omega v) */
			mpf_neg(dtmp, omega);
			add_cmul_mpfvector(vec[4], vec[3], dtmp, vec[5]);
			add_cmul_mpfvector(vec[3], vec[1], beta, vec[4]);
		}
		/* precondition */
		
		/* v := Apt */
#ifdef USE_SPARSE_VERSION
		_bncomp_mul_mpfrsmatrix_mpfvec(vec[5], a, vec[3]);
#elif USE_SPARSE_D_VERSION
		_bncomp_mul_drsmatrix_mpfvec(vec[5], a, vec[3]);
#else // USE_SPARSE_VERSION
		_bncomp_mul_mpfmatrix_mpfvec(vec[5], a, vec[3]);
#endif // USE_SPARSE_VERSION

		ip_mpfvector(alpha_den, vec[2], vec[5]);
		if(mpf_cmp_ui(alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(MPFBiCGSTAB, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		mpf_div(alpha, rho, alpha_den);

		/* s = r - alpha v */
		mpf_neg(dtmp, alpha);
		add_cmul_mpfvector(vec[6], vec[1], dtmp, vec[5]);

		/* Stopping Criteria */
		norm2_mpfvector(dtmp, vec[6]);
		mpf_mul(dtmp1, reps, init_resnorm);
		mpf_add(dtmp1, dtmp1, aeps);
		if(mpf_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			/* x = x + alpha pt */
			add_cmul_mpfvector(vec[0], vec[0], alpha, vec[3]);

			subst_mpfvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		/* precondition */

#ifdef USE_SPARSE_VERSION
		_bncomp_mul_mpfrsmatrix_mpfvec(vec[8], a, vec[6]);
#elif USE_SPARSE_D_VERSION
		_bncomp_mul_drsmatrix_mpfvec(vec[8], a, vec[6]);	
#else // USE_SPARSE_VERSION
		_bncomp_mul_mpfmatrix_mpfvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION

		/* omega = (t, s) / (t, t) */
		ip_mpfvector(omega_den, vec[8], vec[8]);
		if(mpf_cmp_ui(omega_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Omega is zero!(MPFBiCGSTAB, %ld)\n", times);
			return_val = -3; // Fix!
			break;
		}
		ip_mpfvector(omega, vec[8], vec[6]);
		if(mpf_cmp_ui(omega, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Omega is zero!(MPFBiCGSTAB, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}
		mpf_div(omega, omega, omega_den);

		/* x = x + alpha pt + omega st */
		add_cmul_mpfvector(vec[4], vec[0], alpha, vec[3]);
		add_cmul_mpfvector(vec[0], vec[4], omega, vec[6]);

		/* residual */
		mpf_neg(dtmp, omega);
		add_cmul_mpfvector(vec[1], vec[6], dtmp, vec[8]);

		ip_mpfvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		mpf_sqrt(dtmp, beta_num);
		mpf_mul(dtmp1, reps, init_resnorm);
		mpf_add(dtmp1, dtmp1, aeps);
		if(mpf_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_mpfvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		mpf_set(old_rho, rho);
	}

	/* Not converge */
	subst_mpfvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 9; i++)
		free_mpfvector(vec[i]);

	mpf_clear(alpha); mpf_clear(alpha_num); mpf_clear(alpha_den);
	mpf_clear(rho); mpf_clear(old_rho);
	mpf_clear(beta); mpf_clear(beta_num);
	mpf_clear(omega); mpf_clear(omega_den);
	mpf_clear(dtmp); mpf_clear(dtmp1); mpf_clear(init_resnorm);

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(MPFBiCGSTAB, %ld)\n", times);
		return_val = -5;
	}

	// Fix!
	return return_val;
}

/************************************************************/
/*                                                          */
/* GPBiCG(Generalized Product Bi-CG) Method for Real Matrix */
/*                                 (Multi-Precision)        */
/*                                                          */
/*                 ver. 0.0 2024-10-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
  long int _bncomp_MPFGPBiCG_sp(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
#elif USE_SPARSE_D_VERSION
  long int _bncomp_MPFGPBiCG_sp_d(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
#else // USE_SPARSE_VERSION
  long int _bncomp_MPFGPBiCG(MPFVector answer, MPFMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
#endif // USE_SPARSE_VERSION
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
	long int i, j, times, dim, return_val; // Fix!
	unsigned long prec;
	mpf_t dtmp, dtmp1;
	mpf_t alpha, alpha_num, alpha_den;
	mpf_t beta, beta_num;
	mpf_t rho, old_rho;
	mpf_t mu[5], tau, zeta, eta;
	mpf_t init_resnorm;
	MPFVector vec[12]; /* Temporary Vectors */

	dim = answer->dim;
	prec = answer->prec;

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

	for(i = 0; i < 12; i++)
		vec[i] = init2_mpfvector(dim, prec);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0] */
	/* vec[2] ... (b - a * vec[0])^T */
	/* vec[3] ... p */
	/* vec[4] ... q */
	/* vec[5] ... s */
	/* vec[6] ... t */
	/* vec[7] ... u */
	/* vec[8] ... v */
	/* vec[9] ... w */
	/* vec[10]... y */
	/* vec[11]... z */

	subst_mpfvector(vec[1], b); 
	subst_mpfvector(vec[2], b);

	ip_mpfvector(beta_num, vec[1], vec[1]);
	mpf_sqrt(init_resnorm, beta_num);

	mpf_set_ui(old_rho, 0UL);
	mpf_set_ui(rho, 0UL);
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_mpfvector(rho, vec[2], vec[1]);

		if(mpf_cmp_ui(rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(MPFGPBiCG, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
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
			subst_mpfvector(vec[3], vec[1]);
#ifdef USE_SPARSE_VERSION
			_bncomp_mul_mpfrsmatrix_mpfvec(vec[4], a, vec[3]);
#elif USE_SPARSE_D_VERSION
			_bncomp_mul_drsmatrix_mpfvec(vec[4], a, vec[3]);
#else // USE_SPARSE_VERSION
			_bncomp_mul_mpfmatrix_mpfvec(vec[4], a, vec[3]);
#endif // USE_SPARSE_VERSION
			ip_mpfvector(alpha_den, vec[2], vec[4]);
			if(mpf_cmp_ui(alpha_den, 0UL) == 0)
			{
				fprintf(stderr, "Denominator of Alpha is zero!(MPFGPBiCG, %ld)\n", times);
				return_val = -2; // Fix!
				break; // Fix!
			}
			mpf_div(alpha, rho, alpha_den);

			mpf_neg(dtmp, alpha);
			add_cmul_mpfvector(vec[6], vec[1], dtmp, vec[4]);
#ifdef USE_SPARSE_VERSION
			_bncomp_mul_mpfrsmatrix_mpfvec(vec[8], a, vec[6]);
#elif USE_SPARSE_D_VERSION
			_bncomp_mul_drsmatrix_mpfvec(vec[8], a, vec[6]);
#else // USE_SPARSE_VERSION
			_bncomp_mul_mpfmatrix_mpfvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION
			cmul_mpfvector(vec[10], alpha, vec[4]);
			sub_mpfvector(vec[10], vec[10], vec[1]);
			ip_mpfvector(mu[1], vec[8], vec[6]);
			ip_mpfvector(mu[4], vec[8], vec[8]);
			if(mpf_cmp_ui(mu[4], 0UL) == 0)
			{
				fprintf(stderr, "Denominator of mu[4] is zero!(MPFGPBiCG, %ld)\n", times);
				return_val = -3; // Fix!
				break;
			}
			mpf_div(zeta, mu[1], mu[4]);
			mpf_set_ui(eta, 0UL);
			cmul_mpfvector(vec[7], zeta, vec[4]);
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

			add_cmul_mpfvector(vec[9], vec[8], beta, vec[4]);
			sub_mpfvector(vec[3], vec[3], vec[7]);
			add_cmul_mpfvector(vec[3], vec[1], beta, vec[3]);
#ifdef USE_SPARSE_VERSION
			_bncomp_mul_mpfrsmatrix_mpfvec(vec[4], a, vec[3]);
#elif USE_SPARSE_D_VERSION
			_bncomp_mul_drsmatrix_mpfvec(vec[4], a, vec[3]);
#else // USE_SPARSE_VERSION
			_bncomp_mul_mpfmatrix_mpfvec(vec[4], a, vec[3]);
#endif // USE_SPARSE_VERSION
			ip_mpfvector(alpha_den, vec[2], vec[4]);
			if(mpf_cmp_ui(alpha_den, 0UL) == 0)
			{
				fprintf(stderr, "Denominator of Alpha is zero!(MPFBiCG, %ld)\n", times);
				return_val = -2; // Fix!
				break; // Fix!
			}
			mpf_div(alpha, rho, alpha_den);
			sub_mpfvector(vec[5], vec[6], vec[1]);
			mpf_neg(dtmp, alpha);
			add_cmul_mpfvector(vec[6], vec[1], dtmp, vec[4]);
#ifdef USE_SPARSE_VERSION
			_bncomp_mul_mpfrsmatrix_mpfvec(vec[4], a, vec[3]);
			_bncomp_mul_mpfrsmatrix_mpfvec(vec[8], a, vec[6]);
#elif USE_SPARSE_D_VERSION
			_bncomp_mul_drsmatrix_mpfvec(vec[4], a, vec[3]);
			_bncomp_mul_drsmatrix_mpfvec(vec[8], a, vec[6]);
#else // USE_SPARSE_VERSION
			_bncomp_mul_mpfmatrix_mpfvec(vec[4], a, vec[3]);
			_bncomp_mul_mpfmatrix_mpfvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION
			sub_mpfvector(vec[10], vec[9], vec[4]);

			mpf_neg(dtmp, alpha);
			add_cmul_mpfvector(vec[10], vec[5], dtmp, vec[10]);
			ip_mpfvector(mu[0], vec[10], vec[10]);
			ip_mpfvector(mu[1], vec[8], vec[6]);
			ip_mpfvector(mu[2], vec[10], vec[6]);
			ip_mpfvector(mu[3], vec[8], vec[10]);
			ip_mpfvector(mu[4], vec[8], vec[8]);

			/* tau = mu[4] * mu[0] - mu[3] * mu[3]; */
			mpf_mul(dtmp, mu[4], mu[0]);
			mpf_mul(dtmp1, mu[3], mu[3]);
			mpf_sub(tau, dtmp, dtmp1);

			if(mpf_cmp_ui(tau, 0UL) == 0)
			{
				fprintf(stderr, "Denominator of Tau is zero!(MPFGPBiCG, %ld)\n", times);
				return_val = -3; // Fix!
				break;
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
			add_cmul_mpfvector(vec[7], vec[5], beta, vec[7]);
			cmul_mpfvector(vec[7], eta, vec[7]);
			add_cmul_mpfvector(vec[7], vec[7], zeta, vec[4]);
		}
		/* z = zeta * ret_res + eta * z - alpha * u; */
		cmul2_mpfvector(vec[11], eta);
		add_cmul_mpfvector(vec[11], vec[11], zeta, vec[1]);
		mpf_neg(dtmp, alpha);
		add_cmul_mpfvector(vec[11], vec[11], dtmp, vec[7]);

		/* x = x + alpha p + z */
		add_cmul_mpfvector(vec[0], vec[0], alpha, vec[3]);
		add2_mpfvector(vec[0], vec[11]);

		/* residual */
		/* r = t - eta * y - zeta * v; */
		mpf_neg(dtmp, eta);
		add_cmul_mpfvector(vec[1], vec[6], dtmp, vec[10]);
		mpf_neg(dtmp, zeta);
		add_cmul_mpfvector(vec[1], vec[1], dtmp, vec[8]);

		ip_mpfvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		mpf_sqrt(dtmp, beta_num);
		mpf_mul(dtmp1, reps, init_resnorm);
		mpf_add(dtmp1, dtmp1, aeps);
		if(mpf_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_mpfvector(answer, vec[0]);
			return_val = times; // Fix!
			break;
		}

		if(mpf_cmp_ui(zeta, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Zeta is zero!(MPFGPBiCG, %ld)\n", times);
			return_val = -4; // Fix!
			break;
		}

		mpf_set(old_rho, rho);
	}

	/* Not converge */
	subst_mpfvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 12; i++)
		free_mpfvector(vec[i]);

	mpf_clear(alpha); mpf_clear(alpha_num); mpf_clear(alpha_den);
	mpf_clear(rho); mpf_clear(old_rho);
	mpf_clear(beta); mpf_clear(beta_num);
	mpf_clear(tau); mpf_clear(zeta); mpf_clear(eta);
	for(i = 0; i < 5; i++)
		mpf_clear(mu[i]);
	mpf_clear(dtmp); mpf_clear(dtmp1); mpf_clear(init_resnorm);

	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(MPFGPBiCG, %ld)\n", times);
		return_val = -5; // Fix!
	}

	// Fix!
	return return_val;
}

#endif // USE_GMP
