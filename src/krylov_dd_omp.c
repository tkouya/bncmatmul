/********************************************************************************/
/* krylov_dd_omp.c:                                                             */
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

#ifdef USE_DDLINEAR

/************************************************************/
/*                                                          */
/* Bi-Conjugate-Gradient Method for Real Matrix             */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 2024-10-28 (Mon) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
  long int _bncomp_DDBiCG_sp(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
#elif USE_SPARSE_D_VERSION
  long int _bncomp_DDBiCG_sp_d(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
#else // USE_SPARSE_VERSION
  long int _bncomp_DDBiCG(DDVector answer, DDMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       DDVector answer: Solution for Ax = b                */
/*       DDMatrix a: Coefficient matrix A                    */
/*                                       (given by user)    */
/*       DDVector b: Constant vector b   (given by user)     */
/*       double reps[DDSIZE]: Relative tolerance (given by user)    */
/*       double aeps[DDSIZE]: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       DDVector answer: Solution for Ax = b                */
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
	double alpha[DDSIZE], alpha_num[DDSIZE], alpha_den[DDSIZE], minus_alpha[DDSIZE];
	double beta[DDSIZE], beta_num[DDSIZE];
	double rho[DDSIZE], old_rho[DDSIZE];
	double dtmp[DDSIZE], dtmp1[DDSIZE], init_resnorm[DDSIZE];
	DDVector vec[7]; /* Temporary Vectors */

	dim = answer->dim;

/* Set initial value */
	for(i = 0; i < 7; i++)
		vec[i] = init_ddvector(dim);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0]  == w*/
	/* vec[2] ... (b - a * vec[0])^T  == wt */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... z */
	/* vec[6] ... z^T */

	subst_ddvector(vec[1], b); 
	subst_ddvector(vec[2], b);

	//ip_ddvector(beta_num, vec[1], vec[1]);
    ip_ddvector(beta_num, vec[1], vec[1]);
	//init_resnorm = sqrt(beta_num);
    rdd_sqrt(init_resnorm, beta_num);

	//old_rho = 0.0;
	//rho = 0.0;
    rdd_set0(old_rho);
    rdd_set0(rho);
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		//ip_ddvector(rho, vec[2], vec[1]);
        ip_ddvector(rho, vec[2], vec[1]);

		//if(rdd_cmp_ui(rho, 0UL) == 0)
        if(rdd_cmp_ui(rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(DDBiCG, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* p := w, pt := wt */
			subst_ddvector(vec[3], vec[1]);
			subst_ddvector(vec[4], vec[2]);
		}
		else
		{
			//beta = rho / old_rho;
            rdd_div(beta, rho, old_rho);

			/* p := w + beta p, pt := wt + beta * pt */
			add_cmul_ddvector(vec[3], vec[1], beta, vec[3]);
			add_cmul_ddvector(vec[4], vec[2], beta, vec[4]);
		}

		/* z := Ap, zt := A^T pt */
#ifdef USE_SPARSE_VERSION
		_bncomp_mul_ddrsmatrix_ddvec(vec[5], a, vec[3]);
		_bncomp_mul_ddrsmatrixt_ddvec(vec[6], a, vec[4]);
#elif USE_SPARSE_D_VERSION
		_bncomp_mul_drsmatrix_ddvec(vec[5], a, vec[3]);
		_bncomp_mul_drsmatrixt_ddvec(vec[6], a, vec[4]);
#else // USE_SPARSE_VERSION
		_bncomp_mul_ddmatrix_ddvec(vec[5], a, vec[3]);
		_bncomp_mul_ddmatrixt_ddvec(vec[6], a, vec[4]);
#endif // USE_SPARSE_VERSION

		//ip_ddvector(alpha_den, vec[4], vec[5]);
        ip_ddvector(alpha_den, vec[4], vec[5]);
		//if(rdd_cmp_ui(alpha_den, 0UL) == 0)
        if(rdd_cmp_ui(alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(DDBiCG, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		//alpha = rho / alpha_den;
        rdd_div(alpha, rho, alpha_den);

		/* x = x + alpha p */
		add_cmul_ddvector(vec[0], vec[0], alpha, vec[3]);

		/* residual */
        rdd_neg(minus_alpha, alpha);
		add_cmul_ddvector(vec[1], vec[1], minus_alpha, vec[5]);
		add_cmul_ddvector(vec[2], vec[2], minus_alpha, vec[6]);

		//ip_ddvector(beta_num, vec[1], vec[1]);
        ip_ddvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		//rdd_sqrt(dtmp, beta_num);
        rdd_sqrt(dtmp, beta_num);
		// dtmp1 = aeps + reps * init_resnorm)
        rdd_mul(dtmp1, reps, init_resnorm);
        rdd_add(dtmp1, dtmp1, aeps);
		//if(dtmp <= aeps + reps * init_resnorm)
        if(rdd_cmp(dtmp, dtmp1) <= 0)
		{
			subst_ddvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		//rdd_set(old_rho, rho);;
        rdd_set(old_rho, rho);
	}

	/* Not converge */
	subst_ddvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 7; i++)
		free_ddvector(vec[i]);

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(DDBiCG, %ld)\n", times);
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
  long int _bncomp_DDCGS_sp(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
#elif USE_SPARSE_D_VERSION
  long int _bncomp_DDCGS_sp_d(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
#else // USE_SPARSE_VERSION
  long int _bncomp_DDCGS(DDVector answer, DDMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes) // avoid name confict with DCG of IMKL
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       DDVector answer: Solution for Ax = b                */
/*       DDMatrix a: Coefficient matrix A                    */
/*                                       (given by user)    */
/*       DDVector b: Constant vector b   (given by user)     */
/*       double reps[DDSIZE]: Relative tolerance (given by user)    */
/*       double aeps[DDSIZE]: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       DDVector answer: Solution for Ax = b                */
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
	double alpha[DDSIZE], minus_alpha[DDSIZE], alpha_num[DDSIZE], alpha_den[DDSIZE];
	double beta[DDSIZE], beta_num[DDSIZE];
	double rho[DDSIZE], old_rho[DDSIZE];
	double dtmp[DDSIZE], dtmp1[DDSIZE], init_resnorm[DDSIZE];
	DDVector vec[9]; /* Temporary Vectors */

	dim = answer->dim;

/* Set initial value */
	for(i = 0; i < 9; i++)
		vec[i] = init_ddvector(dim);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0] */
	/* vec[2] ... (b - a * vec[0])^T */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... u */
	/* vec[6] ... u^T */
	/* vec[7] ... q */
	/* vec[8] ... v^T */

	subst_ddvector(vec[1], b); 
	subst_ddvector(vec[2], b);

	ip_ddvector(beta_num, vec[1], vec[1]);
	rdd_sqrt(init_resnorm, beta_num);

	rdd_set0(old_rho); // = 0.0;
	rdd_set0(rho); // = 0.0;
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_ddvector(rho, vec[2], vec[1]);

		if(rdd_cmp_ui(rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(DCGS, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* u := rt, p := u */
			subst_ddvector(vec[5], vec[1]);
			subst_ddvector(vec[3], vec[5]);
		}
		else
		{
			//beta = rho / old_rho;
            rdd_div(beta, rho, old_rho);

			/* u := r + beta q, p := u + beta * (q + beta p) */
			add_cmul_ddvector(vec[5], vec[1], beta, vec[7]);
			add_cmul_ddvector(vec[8], vec[7], beta, vec[3]);
			add_cmul_ddvector(vec[3], vec[5], beta, vec[8]);
		}
		/* precondition */
		/* pt = linsolve(pc_K, -p) */
		
		/* vt := Apt */
#ifdef USE_SPARSE_VERSION
		_bncomp_mul_ddrsmatrix_ddvec(vec[8], a, vec[3]);
#elif USE_SPARSE_D_VERSION
		_bncomp_mul_drsmatrix_ddvec(vec[8], a, vec[3]);
#else // USE_SPARSE_VERSION
		_bncomp_mul_ddmatrix_ddvec(vec[8], a, vec[3]);
#endif // USE_SPARSE_VERSION

		ip_ddvector(alpha_den, vec[2], vec[8]);
		if(rdd_cmp_ui(alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(DDCGS, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		//alpha = rho / alpha_den;
        rdd_div(alpha, rho, alpha_den);

		/* q = u - alpha  vt */
        rdd_neg(minus_alpha, alpha);
		add_cmul_ddvector(vec[7], vec[5], minus_alpha, vec[8]);

		/* ut = u + q */
		add_ddvector(vec[6], vec[5], vec[7]);

		/* x = x + alpha ut */
		add_cmul_ddvector(vec[0], vec[0], alpha, vec[6]);

		/* residual */
#ifdef USE_SPARSE_VERSION
		_bncomp_mul_ddrsmatrix_ddvec(vec[8], a, vec[6]);
#elif USE_SPARSE_D_VERSION
		_bncomp_mul_drsmatrix_ddvec(vec[8], a, vec[6]);
#else // USE_SPARSE_VERSION
		_bncomp_mul_ddmatrix_ddvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION
		add_cmul_ddvector(vec[1], vec[1], minus_alpha, vec[8]);

		ip_ddvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		rdd_sqrt(dtmp, beta_num);
		// dtmp1 = aeps + reps * init_resnorm)
        rdd_mul(dtmp1, reps, init_resnorm);
        rdd_add(dtmp1, dtmp1, aeps);
		//if(dtmp <= aeps + reps * init_resnorm)
        if(rdd_cmp(dtmp, dtmp1) <= 0)
		{
			subst_ddvector(answer, vec[0]);
			return_val = times; // Fix!
			break;
		}

		rdd_set(old_rho, rho);
	}

	/* Not converge */
	subst_ddvector(answer, vec[0]);

	/* free vec[0]..[8]; */
	for(i = 0; i < 9; i++)
		free_ddvector(vec[i]);

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
  long int _bncomp_DDBiCGSTAB_sp(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
#elif USE_SPARSE_D_VERSION
  long int _bncomp_DDBiCGSTAB_sp_d(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
#else // USE_SPARSE_VERSION
  long int _bncomp_DDBiCGSTAB(DDVector answer, DDMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       DDVector answer: Solution for Ax = b                */
/*       DDMatrix a: Coefficient matrix A                    */
/*                                       (given by user)    */
/*       DDVector b: Constant vector b   (given by user)     */
/*       double reps[DDSIZE]: Relative tolerance (given by user)    */
/*       double aeps[DDSIZE]: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       DDVector answer: Solution for Ax = b                */
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
	double alpha[DDSIZE], minus_alpha[DDSIZE], alpha_num[DDSIZE], alpha_den[DDSIZE];
	double beta[DDSIZE], beta_num[DDSIZE];
	double rho[DDSIZE], old_rho[DDSIZE];
	double omega[DDSIZE], minus_omega[DDSIZE], omega_den[DDSIZE];
	double dtmp[DDSIZE], dtmp1[DDSIZE], init_resnorm[DDSIZE];
	DDVector vec[9]; /* Temporary Vectors */

	dim = answer->dim;

/* Set initial value */
	for(i = 0; i < 9; i++)
		vec[i] = init_ddvector(dim);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0]  == w*/
	/* vec[2] ... (b - a * vec[0])^T  == wt */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... v */
	/* vec[6] ... s */
	/* vec[7] ... s^T */
	/* vec[8] ... t */

	subst_ddvector(vec[1], b); 
	subst_ddvector(vec[2], b);

	ip_ddvector(beta_num, vec[1], vec[1]);
	rdd_sqrt(init_resnorm, beta_num);

	rdd_set0(old_rho); // = 0.0;
	rdd_set0(rho); // = 0.0;
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_ddvector(rho, vec[2], vec[1]);

		if(rdd_cmp_ui(rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(DDBiCG, %ld)\n", times);
			return_val= -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* p := r */
			subst_ddvector(vec[3], vec[1]);
		}
		else
		{
			//beta = (rho / old_rho) * (alpha / omega);
            rdd_div(dtmp, rho, old_rho);
            rdd_div(dtmp1, alpha, omega);
            rdd_mul(beta, dtmp, dtmp1);

			/* p := r + beta (p - omega v) */
			rdd_neg(minus_omega, omega);
			add_cmul_ddvector(vec[4], vec[3], minus_omega, vec[5]);
			add_cmul_ddvector(vec[3], vec[1], beta, vec[4]);
		}
		/* precondition */
		
		/* v := Apt */
#ifdef USE_SPARSE_VERSION
		_bncomp_mul_ddrsmatrix_ddvec(vec[5], a, vec[3]);
#elif USE_SPARSE_D_VERSION
		_bncomp_mul_drsmatrix_ddvec(vec[5], a, vec[3]);
#else // USE_SPARSE_VERSION
		_bncomp_mul_ddmatrix_ddvec(vec[5], a, vec[3]);
#endif // USE_SPARSE_VERSION

		ip_ddvector(alpha_den, vec[2], vec[5]);
		if(rdd_cmp_ui(alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(DDBiCGSTAB, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		//alpha = rho / alpha_den;
        rdd_div(alpha, rho, alpha_den);

		/* s = r - alpha v */
        rdd_neg(minus_alpha, alpha);
		add_cmul_ddvector(vec[6], vec[1], minus_alpha, vec[5]);

		/* Stopping Criteria */
		norm2_ddvector(dtmp, vec[6]);
		// aeps + reps * init_resnorm
		rdd_mul(dtmp1, reps, init_resnorm);
		rdd_add(dtmp1, dtmp1, aeps);
		if(rdd_cmp(dtmp, dtmp1) <= 0) // aeps + reps * init_resnorm)
		{
			/* x = x + alpha pt */
			add_cmul_ddvector(vec[0], vec[0], alpha, vec[3]);

			subst_ddvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		/* precondition */
#ifdef USE_SPARSE_VERSION
		_bncomp_mul_ddrsmatrix_ddvec(vec[8], a, vec[6]);
#elif USE_SPARSE_D_VERSION
		_bncomp_mul_drsmatrix_ddvec(vec[8], a, vec[6]);
#else // USE_SPARSE_VERSION
		_bncomp_mul_ddmatrix_ddvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION
		/* omega = (t, s) / (t, t) */
		ip_ddvector(omega_den, vec[8], vec[8]);
		if(rdd_cmp_ui(omega_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Omega is zero!(DDBiCGSTAB, %ld)\n", times);
			return_val = -3; // Fix!
			break;
		}
		ip_ddvector(omega, vec[8], vec[6]);
		if(rdd_cmp_ui(omega, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Omega is zero!(DDBiCGSTAB, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}
		//omega = omega / omega_den;
        rdd_div(omega, omega, omega_den);

		/* x = x + alpha pt + omega st */
		add_cmul_ddvector(vec[4], vec[0], alpha, vec[3]);
		add_cmul_ddvector(vec[0], vec[4], omega, vec[6]);

		/* residual */
        rdd_neg(dtmp, omega);
		add_cmul_ddvector(vec[1], vec[6], dtmp, vec[8]);

		ip_ddvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		rdd_sqrt(dtmp, beta_num);
		// dtmp1 = aeps + reps * init_resnorm)
        rdd_mul(dtmp1, reps, init_resnorm);
        rdd_add(dtmp1, dtmp1, aeps);
		//if(dtmp <= aeps + reps * init_resnorm)
        if(rdd_cmp(dtmp, dtmp1) <= 0)
		{
			subst_ddvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		rdd_set(old_rho, rho);
	}

	/* Not converge */
	subst_ddvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 9; i++)
		free_ddvector(vec[i]);

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(DDBiCGSTAB, %ld)\n", times);
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
  long int _bncomp_DDGPBiCG_sp(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
#elif USE_SPARSE_D_VERSION
  long int _bncomp_DDGPBiCG_sp_d(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
#else // USE_SPARSE_VERSION
  long int _bncomp_DDGPBiCG(DDVector answer, DDMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       DDVector answer: Solution for Ax = b                */
/*       DDMatrix a: Coefficient matrix A                    */
/*                                       (given by user)    */
/*       DDVector b: Constant vector b   (given by user)     */
/*       double reps[DDSIZE]: Relative tolerance (given by user)    */
/*       double aeps[DDSIZE]: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       DDVector answer: Solution for Ax = b                */
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
	double alpha[DDSIZE], minus_alpha[DDSIZE], alpha_num[DDSIZE], alpha_den[DDSIZE];
	double beta[DDSIZE], beta_num[DDSIZE];
	double rho[DDSIZE], old_rho[DDSIZE];
	double mu[5][DDSIZE], tau[DDSIZE], zeta[DDSIZE], minus_zeta[DDSIZE], eta[DDSIZE], minus_eta[DDSIZE];
	double dtmp[DDSIZE], dtmp1[DDSIZE], init_resnorm[DDSIZE];
	DDVector vec[12]; /* Temporary Vectors */

	dim = answer->dim;

/* Set initial value */
	for(i = 0; i < 12; i++)
		vec[i] = init_ddvector(dim);

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

	subst_ddvector(vec[1], b); 
	subst_ddvector(vec[2], b);

	ip_ddvector(beta_num, vec[1], vec[1]);
	rdd_sqrt(init_resnorm, beta_num);

	rdd_set0(old_rho); // = 0.0;
	rdd_set0(rho); // = 0.0;
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_ddvector(rho, vec[2], vec[1]);

		if(rdd_cmp_ui(rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(DDGPBiCG, %ld)\n", times);
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
			subst_ddvector(vec[3], vec[1]);
#ifdef USE_SPARSE_VERSION
			_bncomp_mul_ddrsmatrix_ddvec(vec[4], a, vec[3]);
#elif USE_SPARSE_D_VERSION
			_bncomp_mul_drsmatrix_ddvec(vec[4], a, vec[3]);
#else // USE_SPARSE_VERSION
			_bncomp_mul_ddmatrix_ddvec(vec[4], a, vec[3]);
#endif // USE_SPARSE_VERSION
			ip_ddvector(alpha_den, vec[2], vec[4]);
			if(rdd_cmp_ui(alpha_den, 0UL) == 0)
			{
				fprintf(stderr, "Denominator of Alpha is zero!(DDGPBiCG, %ld)\n", times);
				return_val = -2; // Fix!
				break; // Fix!
			}
			//alpha = rho / alpha_den;
            rdd_div(alpha, rho, alpha_den);
            rdd_neg(minus_alpha, alpha);
			add_cmul_ddvector(vec[6], vec[1], minus_alpha, vec[4]);
#ifdef USE_SPARSE_VERSION
			_bncomp_mul_ddrsmatrix_ddvec(vec[8], a, vec[6]);
#elif USE_SPARSE_D_VERSION
			_bncomp_mul_drsmatrix_ddvec(vec[8], a, vec[6]);
#else // USE_SPARSE_VERSION
			_bncomp_mul_ddmatrix_ddvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION
			cmul_ddvector(vec[10], alpha, vec[4]);
			sub_ddvector(vec[10], vec[10], vec[1]);
			ip_ddvector(mu[1], vec[8], vec[6]);
			ip_ddvector(mu[4], vec[8], vec[8]);
			//if(mu[4] == 0.0)
            if(rdd_cmp_ui(mu[4], 0UL) == 0)
			{
				fprintf(stderr, "Denominator of mu[4] is zero!(DDGPBiCG, %ld)\n", times);
				return_val = -3; // Fix!
				break; // Fix!
			}
			//zeta = mu[1] / mu[4];
            rdd_div(zeta, mu[1], mu[4]);
			//eta = 0.0;
            rdd_set0(eta);
			cmul_ddvector(vec[7], zeta, vec[4]);
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

			//beta = (rho / old_rho) * (alpha / zeta);
            rdd_div(dtmp, rho, old_rho);
            rdd_div(dtmp1, alpha, zeta);
            rdd_mul(beta, dtmp, dtmp1);

			add_cmul_ddvector(vec[9], vec[8], beta, vec[4]);
			sub_ddvector(vec[3], vec[3], vec[7]);
			add_cmul_ddvector(vec[3], vec[1], beta, vec[3]);
#ifdef USE_SPARSE_VERSION
			_bncomp_mul_ddrsmatrix_ddvec(vec[4], a, vec[3]);
#elif USE_SPARSE_D_VERSION
			_bncomp_mul_drsmatrix_ddvec(vec[4], a, vec[3]);
#else // USE_SPARSE_VERSION
			_bncomp_mul_ddmatrix_ddvec(vec[4], a, vec[3]);
#endif // USE_SPARSE_VERSION
			ip_ddvector(alpha_den, vec[2], vec[4]);
			if(rdd_cmp_ui(alpha_den, 0UL) == 0)
			{
				fprintf(stderr, "Denominator of Alpha is zero!(DDGPBiCG, %ld)\n", times);
				return_val = -2; // Fix!
				break; // Fix!
			}
			//alpha = rho / alpha_den;
            rdd_div(alpha, rho, alpha_den);
			sub_ddvector(vec[5], vec[6], vec[1]);
            rdd_neg(minus_alpha, alpha);
			add_cmul_ddvector(vec[6], vec[1], minus_alpha, vec[4]);
#ifdef USE_SPARSE_VERSION
			_bncomp_mul_ddrsmatrix_ddvec(vec[8], a, vec[6]);
#elif USE_SPARSE_D_VERSION
			_bncomp_mul_drsmatrix_ddvec(vec[8], a, vec[6]);
#else // USE_SPARSE_VERSION
			_bncomp_mul_ddmatrix_ddvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION
			sub_ddvector(vec[10], vec[9], vec[4]);
			add_cmul_ddvector(vec[10], vec[5], minus_alpha, vec[10]);
			ip_ddvector(mu[0], vec[10], vec[10]);
			ip_ddvector(mu[1], vec[8], vec[6]);
			ip_ddvector(mu[2], vec[10], vec[6]);
			ip_ddvector(mu[3], vec[8], vec[10]);
			ip_ddvector(mu[4], vec[8], vec[8]);
			//tau = mu[4] * mu[0] - mu[3] * mu[3];
            rdd_mul(tau, mu[4], mu[0]);
            rdd_mul(dtmp, mu[3], mu[3]);
            rdd_sub(tau, tau, dtmp);

			//zeta = (mu[0] * mu[1] - mu[2] * mu[3]) / tau;
            rdd_mul(zeta, mu[0], mu[1]);
            rdd_mul(dtmp, mu[2], mu[3]);
            rdd_sub(zeta, zeta, dtmp);
            rdd_div(zeta, zeta, tau);

			//eta = (mu[4] * mu[2] - mu[3] * mu[1]) / tau;
            rdd_mul(eta, mu[4], mu[2]);
            rdd_mul(dtmp, mu[3], mu[1]);
            rdd_sub(eta, eta, dtmp);
            rdd_div(eta, eta, tau);

			/* u = zeta * q + eta * (s + k_beta * u); */
			add_cmul_ddvector(vec[7], vec[5], beta, vec[7]);
			cmul_ddvector(vec[7], eta, vec[7]);
			add_cmul_ddvector(vec[7], vec[7], zeta, vec[4]);
		}
		/* z = zeta * ret_res + eta * z - alpha * u; */
		cmul2_ddvector(vec[11], eta);
		add_cmul_ddvector(vec[11], vec[11], zeta, vec[1]);
        rdd_neg(minus_alpha, alpha);
		add_cmul_ddvector(vec[11], vec[11], minus_alpha, vec[7]);

		/* x = x + alpha p + z */
		add_cmul_ddvector(vec[0], vec[0], alpha, vec[3]);
		add2_ddvector(vec[0], vec[11]);

		/* residual */
		/* r = t - eta * y - zeta * v; */
		rdd_neg(minus_eta, eta);
		rdd_neg(minus_zeta, zeta);
		add_cmul_ddvector(vec[1], vec[6], minus_eta, vec[10]);
		add_cmul_ddvector(vec[1], vec[1], minus_zeta, vec[8]);

		ip_ddvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		rdd_sqrt(dtmp, beta_num);
		// dtmp1 = aeps + reps * init_resnorm)
        rdd_mul(dtmp1, reps, init_resnorm);
        rdd_add(dtmp1, dtmp1, aeps);
		//if(dtmp <= aeps + reps * init_resnorm)
        if(rdd_cmp(dtmp, dtmp1) <= 0)
		{
			subst_ddvector(answer, vec[0]);
			return_val = times; // Fix!
			break;
		}
//		printf("DGPBiCG: %5d %10.3e\n", times, dtmp / init_resnorm);
		//if(zeta == 0.0)
        if(rdd_cmp_ui(zeta, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Zeta is zero!(DDGPBiCG, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}

		rdd_set(old_rho, rho);;
	}

	/* Not converge */
	subst_ddvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 12; i++)
		free_ddvector(vec[i]);

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(DDGPBiCG, %ld)\n", times);
		return_val = -5;
	}

	return return_val;

}

#endif // USE_DDLINEAR

