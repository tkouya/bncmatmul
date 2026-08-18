/********************************************************************************/
/* krylov_c.c:                                                                  */
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
/* Bi-Conjugate-Gradient Method for Complex Matrix          */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 2024-11-20 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDBiCG_sp(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CDBiCG_sp_iLU0(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDBiCG_sp(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes)
		#else // USE_PRECOND
		long int CDBiCG_sp_iLU0(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDBiCG(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CDBiCG_iLU0(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes, CDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDBiCG(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes)
		#else // USE_PRECOND
		long int CDBiCG_iLU0(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes, CDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       CDVector answer: Solution for Ax = b               */
/*       CDMatrix a: Coefficient matrix A                   */
/*                                       (given by user)    */
/*       CDVector b: Constant vector b   (given by user)    */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       CDVector answer: Solution for Ax = b               */
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
	double _Complex alpha, alpha_num, alpha_den;
	double _Complex beta, beta_num;
	double _Complex rho, old_rho;
	double dtmp, init_resnorm;
	CDVector vec[9]; /* Temporary Vectors */
	CDVector dvec[2]; // for Preconditioning

	dim = answer->dim;

/* Set initial value */
	for(i = 0; i < 9; i++)
		vec[i] = init_cdvector(dim);

#ifdef USE_PRECOND
	dvec[0] = init_cdvector(dim);
	dvec[1] = init_cdvector(dim);
#endif // USE_PRECOND

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0]  == r */
	/* vec[2] ... (b - a * vec[0])^*  == rh */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... z */
	/* vec[6] ... z^T */
	// vec[7] ... w  == M^(-1) * r
	// vec[8] ... wt == M^(-H) * rh

	subst_cdvector(vec[1], b);
	subst_cdvector(vec[2], b);
	subst_cdvector(vec[7], vec[1]);
	subst_cdvector(vec[8], vec[2]);
#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_cdvector(dvec[0], vec[7]);
		solve_iLU0_cdrsmatrix(vec[7], ilu, dvec[0]);
		subst_cdvector(dvec[1], vec[8]);
		//solve_iLU0t_cdrsmatrix(vec[2], ilu, dvec[1]);
		solve_iLU0s_cdrsmatrix(vec[8], ilu, dvec[1]);
}
#endif // USE_PRECOND
	//subst_cdvector(vec[2], b);
	//subst_cdvector(vec[2], vec[1]);

	//beta_num = ip_cdvector(vec[2], vec[1]);
	//init_resnorm = sqrt(beta_num);
    init_resnorm = norm2_cdvector(vec[1]);
	#ifdef USE_PRECOND
	if(norm2_res_history != NULL) norm2_res_history[0] = init_resnorm;
	#endif // USE_PRECOND

	old_rho = 0.0;
	rho = 0.0;
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		//rho = ip_cdvector(vec[2], vec[1]);
		rho = ip_cdvector(vec[2], vec[7]);

		if(cabs(rho) == 0.0)
		{
			fprintf(stderr, "Rho is zero!(CDBiCG, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* p := w, pt := wt */
			//subst_cdvector(vec[3], vec[1]);
			//subst_cdvector(vec[4], vec[2]);
			subst_cdvector(vec[3], vec[7]);
			subst_cdvector(vec[4], vec[8]);
		}
		else
		{
			beta = rho / old_rho;

			/* p := w + beta p, pt := wt + beta * pt */
			//add_cmul_cdvector(vec[3], vec[1], beta, vec[3]);
			add_cmul_cdvector(vec[3], vec[7], beta, vec[3]);
			//add_cmul_cdvector(vec[4], vec[2], beta, vec[4]);
			//add_cmul_cdvector(vec[4], vec[2], conj(beta), vec[4]);
			add_cmul_cdvector(vec[4], vec[8], conj(beta), vec[4]);
		}

#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			//subst_cdvector(dvec[0], vec[5]);
			//solve_iLU0_cdrsmatrix(vec[5], ilu, dvec[0]);
			//subst_cdvector(dvec[1], vec[4]); //vec[6]);
			//solve_iLU0t_cdrsmatrix(vec[6], ilu, dvec[1]);
			//solve_iLU0s_cdrsmatrix(vec[4], ilu, dvec[1]);
		}
#endif // USE_PRECOND

/* z := Ap, zt := A^H pt */
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cdrsmatrix_cdvec(vec[5], a, vec[3]);
		_bncomp_mul_cdrsmatrixs_cdvec(vec[6], a, vec[4]);
	#else // USE_OMP_VERSION
		mul_cdrsmatrix_cdvec(vec[5], a, vec[3]);
		mul_cdrsmatrixs_cdvec(vec[6], a, vec[4]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_cdmatrix_cdvec(vec[5], a, vec[3]);
		mul_cdmatrixs_cdvec(vec[6], a, vec[4]);
#endif // USE_SPARSE_VERSION

		alpha_den = ip_cdvector(vec[4], vec[5]);
		if(cabs(alpha_den) == 0.0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(CDBiCG, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		alpha = rho / alpha_den;

		/* x = x + alpha p */
		add_cmul_cdvector(vec[0], vec[0], alpha, vec[3]);

		/* residual */
		//add_cmul_cdvector(vec[1], vec[1], -alpha, vec[5]);
		//add_cmul_cdvector(vec[2], vec[2], -alpha, vec[6]);
		sub_cmul_cdvector(vec[1], vec[1], alpha, vec[5]);
		//sub_cmul_cdvector(vec[1], vec[1], alpha, vec[7]);
		sub_cmul_cdvector(vec[2], vec[2], alpha, vec[6]);
		//sub_cmul_cdvector(vec[2], vec[2], conj(alpha), vec[8]);

		//beta_num = ip_cdvector(vec[1], vec[1]);
        //beta_num = ip_cdvector(vec[2], vec[1]);

		/* Stopping Criteria */
		//dtmp = sqrt(beta_num);
        dtmp = norm2_cdvector(vec[1]);
		#ifdef USE_PRECOND
		if(norm2_res_history != NULL) norm2_res_history[times + 1] = dtmp;
		#endif // USE_PRECOND
		if(dtmp <= aeps + reps * init_resnorm)
		{
			subst_cdvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}
		subst_cdvector(vec[7], vec[1]);
		subst_cdvector(vec[8], vec[2]);
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_cdvector(dvec[0], vec[7]);
			//solve_iLU0_cdrsmatrix(vec[5], ilu, dvec[0]);
			solve_iLU0_cdrsmatrix(vec[7], ilu, dvec[0]);
			subst_cdvector(dvec[1], vec[8]);
			//solve_iLU0t_cdrsmatrix(vec[6], ilu, dvec[1]);
			//solve_iLU0s_cdrsmatrix(vec[6], ilu, dvec[1]);
			solve_iLU0s_cdrsmatrix(vec[8], ilu, dvec[1]);
		}
#endif // USE_PRECOND

		old_rho = rho;
	}

	/* Not converge */
	subst_cdvector(answer, vec[0]);

	/* free vec[0]..[8]; */
	for(i = 0; i < 9; i++)
		free_cdvector(vec[i]);

#ifdef USE_PRECOND
	free_cdvector(dvec[0]);
	free_cdvector(dvec[1]);
#endif // USE_PRECOND

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(CDBiCG, %ld)\n", times);
		return_val = -5; // Fix!
	}

	return return_val; // Fix!

}

/************************************************************/
/*                                                          */
/*               CGS Method for Complex Matrix              */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 2024-11-20 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDCGS_sp(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CDCGS_sp_iLU0(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDCGS_sp(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes)
		#else // USE_PRECOND
		long int CDCGS_sp_iLU0(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDCGS(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CDCGS_iLU0(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes, CDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDCGS(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes)
		#else // USE_PRECOND
		long int CDCGS_iLU0(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes, CDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       CDVector answer: Solution for Ax = b               */
/*       CDMatrix a: Coefficient matrix A                   */
/*                                       (given by user)    */
/*       CDVector b: Constant vector b   (given by user)    */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       CDVector answer: Solution for Ax = b               */
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
	double _Complex alpha, alpha_num, alpha_den;
	double _Complex beta, beta_num;
	double _Complex rho, old_rho;
	double dtmp, init_resnorm;
	CDVector vec[9]; /* Temporary Vectors */
	CDVector dvec[2]; // for preconditioning

	dim = answer->dim;

/* Set initial value */
	for(i = 0; i < 9; i++)
		vec[i] = init_cdvector(dim);

#ifdef USE_PRECOND
	dvec[0] = init_cdvector(dim);
	dvec[1] = init_cdvector(dim);
#endif // USE_PRECOND

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0] */
	/* vec[2] ... (b - a * vec[0])^T */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... u */
	/* vec[6] ... u^T */
	/* vec[7] ... q */
	/* vec[8] ... v^T */

	subst_cdvector(vec[1], b);
#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_cdvector(dvec[0], vec[1]);
		solve_iLU0_cdrsmatrix(vec[1], ilu, dvec[0]);
	}
#else // USE_PRECOND
	subst_cdvector(vec[1], b);
#endif // USE_PRECOND
	//subst_cdvector(vec[2], b);
	subst_cdvector(vec[2], vec[1]);

	//beta_num = ip_dvector(vec[1], vec[1]);
	//init_resnorm = sqrt(beta_num);
    init_resnorm = norm2_cdvector(vec[1]);
	#ifdef USE_PRECOND
	if(norm2_res_history != NULL) norm2_res_history[0] = init_resnorm;
	#endif // USE_PRECOND

	old_rho = 0.0;
	rho = 0.0;
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		rho = ip_cdvector(vec[2], vec[1]);

		if(cabs(rho) == 0.0)
		{
			fprintf(stderr, "Rho is zero!(CDCGS, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* u := rt, p := u */
			subst_cdvector(vec[5], vec[1]);
			subst_cdvector(vec[3], vec[5]);
		}
		else
		{
			beta = rho / old_rho;

			/* u := r + beta q, p := u + beta * (q + beta p) */
			add_cmul_cdvector(vec[5], vec[1], beta, vec[7]);
			add_cmul_cdvector(vec[8], vec[7], beta, vec[3]);
			add_cmul_cdvector(vec[3], vec[5], beta, vec[8]);
		}
		
		/* vt := Apt */
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cdrsmatrix_cdvec(vec[8], a, vec[3]);
	#else // USE_OMP_VERSION
		mul_cdrsmatrix_cdvec(vec[8], a, vec[3]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_cdmatrix_cdvec(vec[8], a, vec[3]);
#endif // USE_SPARSE_VERSION

		/* precondition */
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_cdvector(dvec[0], vec[8]);
			solve_iLU0_cdrsmatrix(vec[8], ilu, dvec[0]);
		}
#endif // USE_PRECOND

		alpha_den = ip_cdvector(vec[2], vec[8]);
		if(alpha_den == 0.0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(CDCGS, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		alpha = rho / alpha_den;

		/* q = u - alpha  vt */
		add_cmul_cdvector(vec[7], vec[5], -alpha, vec[8]);

		/* ut = u + q */
		add_cdvector(vec[6], vec[5], vec[7]);

		/* x = x + alpha ut */
		add_cmul_cdvector(vec[0], vec[0], alpha, vec[6]);

		/* residual */
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cdrsmatrix_cdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
		mul_cdrsmatrix_cdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_cdmatrix_cdvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION

		/* precondition */
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_cdvector(dvec[0], vec[8]);
			solve_iLU0_cdrsmatrix(vec[8], ilu, dvec[0]);
		}
#endif // USE_PRECOND

		add_cmul_cdvector(vec[1], vec[1], -alpha, vec[8]);

		//beta_num = ip_dvector(vec[1], vec[1]);

		/* Stopping Criteria */
		//dtmp = sqrt(beta_num);
        dtmp = norm2_cdvector(vec[1]);
		#ifdef USE_PRECOND
		if(norm2_res_history != NULL) norm2_res_history[times + 1] = dtmp;
		#endif // USE_PRECOND

		if(dtmp <= aeps + reps * init_resnorm)
		{
			subst_cdvector(answer, vec[0]);
			return_val = times; // Fix!
			break;
		}

		old_rho = rho;
	}

	/* Not converge */
	subst_cdvector(answer, vec[0]);

	/* free vec[0]..[8]; */
	for(i = 0; i < 9; i++)
		free_cdvector(vec[i]);

#ifdef USE_PRECOND
	free_cdvector(dvec[0]);
	free_cdvector(dvec[1]);
#endif // USE_PRECOND

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(CDCGS, %ld)\n", times);
		return_val= -5;
	}

	return return_val;

}

/************************************************************/
/*                                                          */
/*             BiCGSTAB Method for Complex Matrix           */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 2024-11-20 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDBiCGSTAB_sp(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CDBiCGSTAB_sp_iLU0(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDBiCGSTAB_sp(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes)
		#else // USE_PRECOND
		long int CDBiCGSTAB_sp_iLU0(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDBiCGSTAB(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CDBiCGSTAB_iLU0(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes, CDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDBiCGSTAB(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes)
		#else // USE_PRECOND
		long int CDBiCGSTAB_iLU0(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes, CDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       CDVector answer: Solution for Ax = b               */
/*       CDMatrix a: Coefficient matrix A                   */
/*                                       (given by user)    */
/*       CDVector b: Constant vector b   (given by user)    */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       CDVector answer: Solution for Ax = b               */
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
	double _Complex alpha, alpha_num, alpha_den;
	double _Complex beta, beta_num;
	double _Complex rho, old_rho;
	double _Complex omega, omega_den;
	double dtmp, init_resnorm;
	//CDVector vec[9]; /* Temporary Vectors */
	CDVector vec[10]; /* Temporary Vectors */
	CDVector dvec[2]; // for Preconditioning

	dim = answer->dim;

/* Set initial value */
	//for(i = 0; i < 9; i++)
	for(i = 0; i < 10; i++)
		vec[i] = init_cdvector(dim);

#ifdef USE_PRECOND
	dvec[0] = init_cdvector(dim);
	dvec[1] = init_cdvector(dim);
#endif // USE_PRECOND

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0]  == w*/
	/* vec[2] ... (b - a * vec[0])^*  == wt */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... v */
	/* vec[6] ... s */
	/* vec[7] ... s^T */
	/* vec[8] ... t */

	subst_cdvector(vec[1], b);
#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_cdvector(dvec[0], vec[1]);
		solve_iLU0_cdrsmatrix(vec[1], ilu, dvec[0]);
	}
#else // USE_PRECOND
	subst_cdvector(vec[1], b);
#endif // USE_PRECOND
	//subst_cdvector(vec[2], b);
	subst_cdvector(vec[2], vec[1]);

	//beta_num = ip_cdvector(vec[1], vec[1]);
	//init_resnorm = sqrt(beta_num);
    init_resnorm = norm2_cdvector(vec[1]);
	#ifdef USE_PRECOND
	if(norm2_res_history != NULL) norm2_res_history[0] = init_resnorm;
	#endif // USE_PRECOND

	old_rho = 0.0;
	rho = 0.0;
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		rho = ip_cdvector(vec[2], vec[1]);

		if(cabs(rho) == 0.0)
		{
			fprintf(stderr, "Rho is zero!(CDBiCGSTAB, %ld)\n", times);
			return_val= -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* p := r */
			subst_cdvector(vec[3], vec[1]);
		}
		else
		{
			beta = (rho / old_rho) * (alpha / omega);

			/* p := r + beta (p - omega v) */
			//add_cmul_cdvector(vec[4], vec[3], -omega, vec[5]);
			sub_cmul_cdvector(vec[4], vec[3], omega, vec[5]);
			add_cmul_cdvector(vec[3], vec[1], beta, vec[4]);
		}		
		/* v := Apt */
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cdrsmatrix_cdvec(vec[5], a, vec[3]);
	#else // USE_OMP_VERSION
		mul_cdrsmatrix_cdvec(vec[5], a, vec[3]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_cdmatrix_cdvec(vec[5], a, vec[3]);
#endif // USE_SPARSE_VERSION

		/* precondition */
		// v := M^(-1)Apt -> Solve Mv = Apt for v
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_cdvector(vec[9], vec[5]);
			solve_iLU0_cdrsmatrix(vec[5], ilu, vec[9]);
		}
#endif // USE_PRECOND

		alpha_den = ip_cdvector(vec[2], vec[5]);
		if(cabs(alpha_den) == 0.0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(CDBiCGSTAB, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		alpha = rho / alpha_den;

		/* s = r - alpha v */
		//add_cmul_cdvector(vec[6], vec[1], -alpha, vec[5]);
		sub_cmul_cdvector(vec[6], vec[1], alpha, vec[5]);

		/* Stopping Criteria */
		dtmp = norm2_cdvector(vec[6]);
		if(dtmp <= aeps + reps * init_resnorm)
		{
			/* x = x + alpha pt */
			add_cmul_cdvector(vec[0], vec[0], alpha, vec[3]);

			subst_cdvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		/* precondition */
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cdrsmatrix_cdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
		mul_cdrsmatrix_cdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_cdmatrix_cdvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION

		/* precondition */
		// v := M^(-1)Apt -> Solve Mv = Apt for v
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_cdvector(vec[9], vec[8]);
			solve_iLU0_cdrsmatrix(vec[8], ilu, vec[9]);
		}
#endif // USE_PRECOND

		/* omega = (t, s) / (t, t) */
		omega_den = ip_cdvector(vec[8], vec[8]);
		if(cabs(omega_den) == 0)
		{
			fprintf(stderr, "Denominator of Omega is zero!(CDBiCGSTAB, %ld)\n", times);
			return_val = -3; // Fix!
			break;
		}
		omega = ip_cdvector(vec[8], vec[6]);
		if(cabs(omega) == 0)
		{
			fprintf(stderr, "Numerator of Omega is zero!(CDBiCGSTAB, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}
		omega = omega / omega_den;

		/* x = x + alpha pt + omega st */
		add_cmul_cdvector(vec[4], vec[0], alpha, vec[3]);
		add_cmul_cdvector(vec[0], vec[4], omega, vec[6]);

		/* residual */
		//add_cmul_cdvector(vec[1], vec[6], -omega, vec[8]);
		sub_cmul_cdvector(vec[1], vec[6], omega, vec[8]);

		//beta_num = ip_cdvector(vec[1], vec[1]);

		/* Stopping Criteria */
		//dtmp = sqrt(beta_num);
        dtmp = norm2_cdvector(vec[1]);
		#ifdef USE_PRECOND
		if(norm2_res_history != NULL) norm2_res_history[times + 1] = dtmp;
		#endif // USE_PRECOND
		if(dtmp <= aeps + reps * init_resnorm)
		{
			subst_cdvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		old_rho = rho;
	}

	/* Not converge */
	subst_cdvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	//for(i = 0; i < 9; i++)
	for(i = 0; i < 10; i++)
		free_cdvector(vec[i]);

#ifdef USE_PRECOND
	free_cdvector(dvec[0]);
	free_cdvector(dvec[1]);
#endif // USE_PRECOND

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(CDBiCGSTAB, %ld)\n", times);
		return_val = -5;
	}

	return return_val; // Fix!

}

/************************************************************/
/*                                                          */
/* GPBiCG(Generalized Product Bi-CG) Method                 */
/*                                       for Complex Matrix */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 2024-11-20 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDGPBiCG_sp(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CDGPBiCG_sp_iLU0(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDGPBiCG_sp(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes)
		#else // USE_PRECOND
		long int CDGPBiCG_sp_iLU0(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDGPBiCG(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CDGPBiCG_iLU0(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes, CDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDGPBiCG(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes)
		#else // USE_PRECOND
		long int CDGPBiCG_iLU0(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes, CDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       CDVector answer: Solution for Ax = b               */
/*       CDMatrix a: Coefficient matrix A                   */
/*                                       (given by user)    */
/*       CDVector b: Constant vector b   (given by user)    */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       CDVector answer: Solution for Ax = b               */
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
	double _Complex alpha, alpha_num, alpha_den;
	double _Complex beta, beta_num;
	double _Complex rho, old_rho;
	double _Complex mu[5], tau, zeta, eta;
	double dtmp, init_resnorm;
	CDVector vec[12]; /* Temporary Vectors */
	CDVector dvec[2]; // Preconditioning

	dim = answer->dim;

/* Set initial value */
	for(i = 0; i < 12; i++)
		vec[i] = init_cdvector(dim);

#ifdef USE_PRECOND
	dvec[0] = init_cdvector(dim);
	dvec[1] = init_cdvector(dim);
#endif // USE_PRECOND

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

	subst_cdvector(vec[1], b);
#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_cdvector(dvec[0], vec[1]);
		solve_iLU0_cdrsmatrix(vec[1], ilu, dvec[0]);
	}
#else // USE_PRECOND
	subst_cdvector(vec[1], b);
#endif // USE_PRECOND
	//subst_cdvector(vec[2], b);
	subst_cdvector(vec[2], vec[1]);

	//beta_num = ip_cdvector(vec[1], vec[1]);
	//init_resnorm = sqrt(beta_num);
    init_resnorm = norm2_cdvector(vec[1]);
#ifdef USE_PRECOND
	if(norm2_res_history != NULL) norm2_res_history[0] = init_resnorm;
#endif// USE_PRECOND

	old_rho = 0.0;
	rho = 0.0;
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		rho = ip_cdvector(vec[2], vec[1]);

		if(cabs(rho) == 0.0)
		{
			fprintf(stderr, "Rho is zero!(CDGPBiCG, %ld)\n", times);
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
			subst_cdvector(vec[3], vec[1]);
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_cdrsmatrix_cdvec(vec[4], a, vec[3]);
	#else // USE_OMP_VERSION
			mul_cdrsmatrix_cdvec(vec[4], a, vec[3]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
			mul_cdmatrix_cdvec(vec[4], a, vec[3]);
#endif // USE_SPARSE_VERSION
#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_cdvector(dvec[0], vec[4]);
				solve_iLU0_cdrsmatrix(vec[4], ilu, dvec[0]);
			}
#endif // USE_PRECOND

			alpha_den = ip_cdvector(vec[2], vec[4]);
			if(cabs(alpha_den) == 0.0)
			{
				fprintf(stderr, "Denominator of Alpha is zero!(CDGPBiCG, %ld)\n", times);
				return_val = -2; // Fix!
				break; // Fix!
			}
			alpha = rho / alpha_den;
			//add_cmul_cdvector(vec[6], vec[1], -alpha, vec[4]);
			sub_cmul_cdvector(vec[6], vec[1], alpha, vec[4]);
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_cdrsmatrix_cdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
			mul_cdrsmatrix_cdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
			mul_cdmatrix_cdvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION
#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_cdvector(dvec[0], vec[8]);
				solve_iLU0_cdrsmatrix(vec[8], ilu, dvec[0]);
			}
#endif // USE_PRECOND
			cmul_cdvector(vec[10], alpha, vec[4]);
			sub_cdvector(vec[10], vec[10], vec[1]);
			mu[1] = ip_cdvector(vec[8], vec[6]);
			mu[4] = ip_cdvector(vec[8], vec[8]);
			if(cabs(mu[4]) == 0.0)
			{
				fprintf(stderr, "Denominator of mu[4] is zero!(CDGPBiCG, %ld)\n", times);
				return_val = -3; // Fix!
				break; // Fix!
			}
			zeta = mu[1] / mu[4];
			eta = 0.0;
			cmul_cdvector(vec[7], zeta, vec[4]);
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

			add_cmul_cdvector(vec[9], vec[8], beta, vec[4]);
			sub_cdvector(vec[3], vec[3], vec[7]);
			add_cmul_cdvector(vec[3], vec[1], beta, vec[3]);
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_cdrsmatrix_cdvec(vec[4], a, vec[3]);
	#else // USE_OMP_VERSION
			mul_cdrsmatrix_cdvec(vec[4], a, vec[3]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
			mul_cdmatrix_cdvec(vec[4], a, vec[3]);
#endif // USE_SPARSE_VERSION
#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_cdvector(dvec[0], vec[4]);
				solve_iLU0_cdrsmatrix(vec[4], ilu, dvec[0]);
			}
#endif // USE_PRECOND
			alpha_den = ip_cdvector(vec[2], vec[4]);
			if(cabs(alpha_den) == 0.0)
			{
				fprintf(stderr, "Denominator of Alpha is zero!(CDGPBiCG, %ld)\n", times);
				return_val = -2; // Fix!
				break; // Fix!
			}
			alpha = rho / alpha_den;
			sub_cdvector(vec[5], vec[6], vec[1]);
			add_cmul_cdvector(vec[6], vec[1], -alpha, vec[4]);
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_cdrsmatrix_cdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
			mul_cdrsmatrix_cdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
			mul_cdmatrix_cdvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION
#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_cdvector(dvec[0], vec[8]);
				solve_iLU0_cdrsmatrix(vec[8], ilu, dvec[0]);
			}
#endif // USE_PRECOND
			sub_cdvector(vec[10], vec[9], vec[4]);
			add_cmul_cdvector(vec[10], vec[5], -alpha, vec[10]);
			mu[0] = ip_cdvector(vec[10], vec[10]);
			mu[1] = ip_cdvector(vec[8], vec[6]);
			mu[2] = ip_cdvector(vec[10], vec[6]);
			mu[3] = ip_cdvector(vec[8], vec[10]);
			mu[4] = ip_cdvector(vec[8], vec[8]);
			tau = mu[4] * mu[0] - mu[3] * mu[3];
			if(cabs(tau) == 0.0)
			{
				fprintf(stderr, "Denominator of Tau is zero!(CDGPBiCG, %ld)\n", times);
				return_val = -3; // Fix!
				break;
			}

			zeta = (mu[0] * mu[1] - mu[2] * mu[3]) / tau;
			eta = (mu[4] * mu[2] - mu[3] * mu[1]) / tau;

			/* u = zeta * q + eta * (s + k_beta * u); */
			add_cmul_cdvector(vec[7], vec[5], beta, vec[7]);
			cmul_cdvector(vec[7], eta, vec[7]);
			add_cmul_cdvector(vec[7], vec[7], zeta, vec[4]);
		}
		/* z = zeta * ret_res + eta * z - alpha * u; */
		cmul2_cdvector(vec[11], eta);
		add_cmul_cdvector(vec[11], vec[11], zeta, vec[1]);
		add_cmul_cdvector(vec[11], vec[11], -alpha, vec[7]);

		/* x = x + alpha p + z */
		add_cmul_cdvector(vec[0], vec[0], alpha, vec[3]);
		add2_cdvector(vec[0], vec[11]);

		/* residual */
		/* r = t - eta * y - zeta * v; */
		add_cmul_cdvector(vec[1], vec[6], -eta, vec[10]);
		add_cmul_cdvector(vec[1], vec[1], -zeta, vec[8]);

		//beta_num = ip_cdvector(vec[1], vec[1]);

		/* Stopping Criteria */
		//dtmp = sqrt(beta_num);
        dtmp = norm2_cdvector(vec[1]);
#ifdef USE_PRECOND
		if(norm2_res_history != NULL) norm2_res_history[times + 1] = dtmp;
#endif// USE_PRECOND
		if(dtmp <= aeps + reps * init_resnorm)
		{
			subst_cdvector(answer, vec[0]);
			return_val = times; // Fix!
			break;
		}
//		printf("CDGPBiCG: %5d %10.3e\n", times, dtmp / init_resnorm);
		if(cabs(zeta) == 0.0)
		{
			fprintf(stderr, "Denominator of Zeta is zero!(CDGPBiCG, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}

		old_rho = rho;
	}

	/* Not converge */
	subst_cdvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 12; i++)
		free_cdvector(vec[i]);

#ifdef USE_PRECOND
	free_cdvector(dvec[0]); // = init_cdvector(dim);
	free_cdvector(dvec[1]); // = init_cdvector(dim);
#endif // USE_PRECOND

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(CDGPBiCG, %ld)\n", times);
		return_val = -5;
	}

	return return_val;

}

#endif // ifndef USE_SPARSE_D_VERSION

// ------------------------------------
// MPC (MPF)
// ------------------------------------
#ifdef USE_GMP

/************************************************************/
/*                                                          */
/*                Bi-Conjugate-Gradient Method              */
/*                                 (Multi-Precision)        */
/*                                                          */
/*                 ver. 0.0 2024-11-20 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CMPFBiCG_sp(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CMPFBiCG_sp_iLU0(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CMPFRSMatrix ilu, MPFVector norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CMPFBiCG_sp(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
		#else // USE_PRECOND
		long int CMPFBiCG_sp_iLU0(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CMPFRSMatrix ilu, MPFVector norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CMPFBiCG_sp_d(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CMPFBiCG_sp_d_iLU0(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CDRSMatrix ilu, MPFVector norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CMPFBiCG_sp_d(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
		#else // USE_PRECOND
		long int CMPFBiCG_sp_d_iLU0(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CDRSMatrix ilu, MPFVector norm2_res_history)
		#endif //USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifndef USE_PRECOND
	long int CMPFBiCG(CMPFVector answer, CMPFMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
	#else // USE_PRECOND
	long int CMPFBiCG_iLU0(CMPFVector answer, CMPFMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CMPFMatrix ilu, MPFVector norm2_res_history)
	#endif // USE_PRECOND
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       CMPFVector answer: Solution for Ax = b             */
/*       CMPFMatrix a: Coefficient matrix A                 */
/*                                       (given by user)    */
/*       CMPFVector b: Constant vector b   (given by user)  */
/*       mpf_t reps: Relative tolerance (given by user)     */
/*       mpf_t aeps: Absolute tolerance (given by user)     */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       CMPFVector answer: Solution for Ax = b             */
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
	mpc_t alpha, alpha_num, alpha_den, conj_alpha;
	mpc_t rho, old_rho;
	mpc_t beta, beta_num, conj_beta;
	mpf_t dtmp, dtmp1, init_resnorm;
	CMPFVector vec[9]; /* Temporary Vectors */
	CMPFVector dvec[2]; // Preconditioning

	dim = answer->dim;
	prec = answer->prec;

/* Set initial value */
	mpc_init2(alpha, prec);
	mpc_init2(alpha_num, prec);
	mpc_init2(alpha_den, prec);
	mpc_init2(rho, prec);
	mpc_init2(old_rho, prec);
	mpc_init2(beta, prec);
	mpc_init2(beta_num, prec);
	mpc_init2(conj_alpha, prec);
	mpc_init2(conj_beta, prec);
	mpf_init2(dtmp, prec);
	mpf_init2(dtmp1, prec);
	mpf_init2(init_resnorm, prec);

	for(i = 0; i < 9; i++)
		vec[i] = init2_cmpfvector(dim, prec);

#ifdef USE_PRECOND
	dvec[0] = init2_cmpfvector(dim, prec);
	dvec[1] = init2_cmpfvector(dim, prec);
#endif // USE_PRECOND

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0]  == w*/
	/* vec[2] ... (b - a * vec[0])^T  == wt */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... z */
	/* vec[6] ... z^T */

	subst_cmpfvector(vec[1], b); 
	subst_cmpfvector(vec[2], b);
	subst_cmpfvector(vec[7], vec[1]);
	subst_cmpfvector(vec[8], vec[2]);
#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_cmpfvector(dvec[0], vec[1]);
		subst_cmpfvector(dvec[1], vec[2]);
#ifdef USE_SPARSE_D_VERSION
		//solve_iLU0_cdrsmatrix_cmpfvec(vec[1], ilu, dvec[0]);
		//solve_iLU0s_cdrsmatrix_cmpfvec(vec[2], ilu, dvec[1]);
		solve_iLU0_cdrsmatrix_cmpfvec(vec[7], ilu, dvec[0]);
		solve_iLU0s_cdrsmatrix_cmpfvec(vec[8], ilu, dvec[1]);
#else // USE_SPARSE_D_VERSION
		//solve_iLU0_cmpfrsmatrix(vec[1], ilu, dvec[0]);
		//solve_iLU0s_cmpfrsmatrix(vec[2], ilu, dvec[1]);
		solve_iLU0_cmpfrsmatrix(vec[7], ilu, dvec[0]);
		solve_iLU0s_cmpfrsmatrix(vec[8], ilu, dvec[1]);
#endif // 
	}
#endif // USE_PRECOND
	//conj_cmpfvector(vec[2], b);

	//ip_cmpfvector(beta_num, vec[1], vec[1]);
	//mpf_sqrt(init_resnorm, beta_num);
    norm2_cmpfvector(init_resnorm, vec[1]);
#ifdef USE_PRECOND
	if(norm2_res_history != NULL) set_mpfvector_i(norm2_res_history, 0, init_resnorm);
#endif// USE_PRECOND
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_cmpfvector(rho, vec[2], vec[7]);

		if(mpc_cmp_si(rho, 0L) == 0)
		{
			fprintf(stderr, "Rho is zero!(CMPFBiCG, %ld)\n", times);
			return_val = -1; // Fix!
			break;
		}

		if(times == 0)
		{
			/* p := w, pt := wt */
			subst_cmpfvector(vec[3], vec[7]);
			subst_cmpfvector(vec[4], vec[8]);
		}
		else
		{
			mpc_div(beta, rho, old_rho, MPC_RNDNN);

			/* p := w + beta p, pt := wt + beta * pt */
			add_cmul_cmpfvector(vec[3], vec[7], beta, vec[3]);
			//add_cmul_cmpfvector(vec[4], vec[2], beta, vec[4]);
			mpc_conj(conj_beta, beta, MPC_RNDNN);
			add_cmul_cmpfvector(vec[4], vec[8], conj_beta, vec[4]);
		}
		/* z := Ap, zt := A^T pt */
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cmpfrsmatrix_cmpfvec(vec[5], a, vec[3]);
		_bncomp_mul_cmpfrsmatrixs_cmpfvec(vec[6], a, vec[4]);
	#else // USE_OMP_VERSION
		mul_cmpfrsmatrix_cmpfvec(vec[5], a, vec[3]);
		mul_cmpfrsmatrixs_cmpfvec(vec[6], a, vec[4]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cdrsmatrix_cmpfvec(vec[5], a, vec[3]);
		_bncomp_mul_cdrsmatrixs_cmpfvec(vec[6], a, vec[4]);
	#else // USE_OMP_VERSION
		mul_cdrsmatrix_cmpfvec(vec[5], a, vec[3]);
		mul_cdrsmatrixs_cmpfvec(vec[6], a, vec[4]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_cmpfmatrix_cmpfvec(vec[5], a, vec[3]);
		mul_cmpfmatrixs_cmpfvec(vec[6], a, vec[4]);
#endif // USE_SPARSE_VERSION
		ip_cmpfvector(alpha_den, vec[4], vec[5]);
		if(mpc_cmp_si(alpha_den, 0L) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(CMPFBiCG, %ld)\n", times);
			return_val = -2; // Fix!
			break;
		}
		mpc_div(alpha, rho, alpha_den, MPC_RNDNN);

		/* x = x + alpha p */
		add_cmul_cmpfvector(vec[0], vec[0], alpha, vec[3]);

		/* residual */
		//mpc_neg(dtmp, alpha);
		sub_cmul_cmpfvector(vec[1], vec[1], alpha, vec[5]);
		//sub_cmul_cmpfvector(vec[2], vec[2], alpha, vec[6]);
		mpc_conj(conj_alpha, alpha, MPC_RNDNN);
		sub_cmul_cmpfvector(vec[2], vec[2], conj_alpha, vec[6]);

		//ip_cmpfvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		//mpf_sqrt(dtmp, beta_num);
        norm2_cmpfvector(dtmp, vec[1]);
#ifdef USE_PRECOND
		if(norm2_res_history != NULL) set_mpfvector_i(norm2_res_history, times + 1, dtmp);
#endif// USE_PRECOND
		mpf_mul(dtmp1, reps, init_resnorm);
		mpf_add(dtmp1, dtmp1, aeps);
		if(mpf_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_cmpfvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		subst_cmpfvector(vec[7], vec[1]);
		subst_cmpfvector(vec[8], vec[2]);
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_cmpfvector(dvec[0], vec[7]);
			subst_cmpfvector(dvec[1], vec[8]);
#ifdef USE_SPARSE_D_VERSION
			solve_iLU0_cdrsmatrix_cmpfvec(vec[7], ilu, dvec[0]);
			//solve_iLU0t_cdrsmatrix_cmpfvec(vec[6], ilu, dvec[1]);
			solve_iLU0s_cdrsmatrix_cmpfvec(vec[8], ilu, dvec[1]);
#else // USE_SPARSE_D_VERSION
			solve_iLU0_cmpfrsmatrix(vec[7], ilu, dvec[0]);
			//solve_iLU0t_cmpfrsmatrix(vec[6], ilu, dvec[1]);
			solve_iLU0s_cmpfrsmatrix(vec[8], ilu, dvec[1]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

		mpc_set(old_rho, rho, MPC_RNDNN);
	}

	/* Not converge */
	subst_cmpfvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 9; i++)
		free_cmpfvector(vec[i]);

#ifdef USE_PRECOND
	free_cmpfvector(dvec[0]);
	free_cmpfvector(dvec[1]);
#endif // USE_PRECOND

	mpc_clear(alpha); mpc_clear(alpha_num); mpc_clear(alpha_den);
	mpc_clear(rho); mpc_clear(old_rho);
	mpc_clear(beta); mpc_clear(beta_num);
	mpf_clear(dtmp); mpf_clear(dtmp1); mpf_clear(init_resnorm);
	mpc_clear(conj_alpha); mpc_clear(conj_beta);

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(CMPFBiCG, %ld)\n", times);
		return_val = -5;
	}

	return return_val; // Fix!

}

/************************************************************/
/*                                                          */
/*               CGS Method for Complex Matrix              */
/*                                 (Multi-Precision)        */
/*                                                          */
/*                 ver. 0.0 2024-11-20 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CMPFCGS_sp(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CMPFCGS_sp_iLU0(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CMPFRSMatrix ilu, MPFVector norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CMPFCGS_sp(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
		#else // USE_PRECOND
		long int CMPFCGS_sp_iLU0(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CMPFRSMatrix ilu, MPFVector norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CMPFCGS_sp_d(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CMPFCGS_sp_d_iLU0(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CDRSMatrix ilu, MPFVector norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CMPFCGS_sp_d(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
		#else // USE_PRECOND
		long int CMPFCGS_sp_d_iLU0(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CDRSMatrix ilu, MPFVector norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifndef USE_PRECOND
	long int CMPFCGS(CMPFVector answer, CMPFMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
	#else // USE_PRECOND
	long int CMPFCGS_iLU0(CMPFVector answer, CMPFMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CMPFMatrix ilu, MPFVector norm2_res_history)
	#endif // USE_PRECOND
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       CMPFVector answer: Solution for Ax = b             */
/*       MPFMatrix a: Coefficient matrix A                  */
/*                                       (given by user)    */
/*       CMPFVector b: Constant vector b  (given by user)   */
/*       mpf_t reps: Relative tolerance (given by user)     */
/*       mpf_t aeps: Absolute tolerance (given by user)     */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       CMPFVector answer: Solution for Ax = b              */
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
	mpc_t ctmp, ctmp1;
	mpc_t alpha, alpha_num, alpha_den;
	mpc_t beta, beta_num;
	mpc_t rho, old_rho;
	mpf_t dtmp, dtmp1, init_resnorm;
	CMPFVector vec[9]; /* Temporary Vectors */
	CMPFVector dvec[2]; // Preconditioning
	CDVector in_dvec[2];

	dim = answer->dim;
	prec = answer->prec;

/* Set initial value */
	mpc_init2(alpha, prec);
	mpc_init2(alpha_num, prec);
	mpc_init2(alpha_den, prec);
	mpc_init2(rho, prec);
	mpc_init2(old_rho, prec);
	mpc_init2(beta, prec);
	mpc_init2(beta_num, prec);
    mpc_init2(ctmp, prec);
    mpc_init2(ctmp1, prec);
	mpf_init2(dtmp, prec);
	mpf_init2(dtmp1, prec);
	mpf_init2(init_resnorm, prec);

	for(i = 0; i < 9; i++)
		vec[i] = init2_cmpfvector(dim, prec);

#ifdef USE_PRECOND
	dvec[0] = init2_cmpfvector(dim, prec);
	dvec[1] = init2_cmpfvector(dim, prec);
	#if defined(USE_SPARSE_D_VERSION) && defined(USE_VEC_D)
	in_dvec[0] = init2_cdvector(dim);
	in_dvec[1] = init2_cdvector(dim);
	#endif // defined(USE_SPARSE_D_VERSION) && defined(USE_VEC_D)
#endif // USE_PRECOND

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0] */
	/* vec[2] ... (b - a * vec[0])^T */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... u */
	/* vec[6] ... u^T */
	/* vec[7] ... q */
	/* vec[8] ... v^T */

	subst_cmpfvector(vec[1], b); 
#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_cmpfvector(dvec[0], vec[1]);
#ifdef USE_SPARSE_D_VERSION
		// normalize
		//normi_cmpfvector(dtmp, dvec[0]);
		//mpc_set_f(ctmp, dtmp, MPC_RNDNN);
		//mpf_ui_div(dtmp1, 1UL, dtmp); mpc_set_f(ctmp1, dtmp1, MPC_RNDNN);
		//cmul_cmpfvector(dvec[1], ctmp1, dvec[0]);
		//solve_iLU0_cdrsmatrix_cmpfvec(dvec[0], ilu, dvec[1]);
		solve_iLU0_cdrsmatrix_cmpfvec(vec[1], ilu, dvec[0]);
		// denormalize
		//cmul_cmpfvector(vec[1], ctmp, dvec[0]);
#else // USE_SPARSE_D_VERSION
		solve_iLU0_cmpfrsmatrix(vec[1], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
	}
#endif // USE_PRECOND
	//subst_cmpfvector(vec[2], b);
	subst_cmpfvector(vec[2], vec[1]);
	//subst_cmpfvector(vec[2], b);

	//ip_cmpfvector(beta_num, vec[1], vec[1]);
	//mpf_sqrt(init_resnorm, beta_num);
    norm2_cmpfvector(init_resnorm, vec[1]);
	#ifdef USE_PRECOND
	if(norm2_res_history != NULL) set_mpfvector_i(norm2_res_history, 0, init_resnorm);
	#endif // USE_PRECOND

	mpc_set_ui(old_rho, 0UL, MPC_RNDNN);
	mpc_set_ui(rho, 0UL, MPC_RNDNN);
	return_val = 0;

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_cmpfvector(rho, vec[2], vec[1]);

		if(mpc_cmp_si(rho, 0L) == 0)
		{
			fprintf(stderr, "Rho is zero!(CMPFCGS, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* u := rt, p := u */
			subst_cmpfvector(vec[5], vec[1]);
			subst_cmpfvector(vec[3], vec[5]);
		}
		else
		{
			mpc_div(beta, rho, old_rho, MPC_RNDNN);

			/* u := r + beta q, p := u + beta * (q + beta p) */
			add_cmul_cmpfvector(vec[5], vec[1], beta, vec[7]);
			add_cmul_cmpfvector(vec[8], vec[7], beta, vec[3]);
			add_cmul_cmpfvector(vec[3], vec[5], beta, vec[8]);
		}
		
		/* vt := Apt */
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cmpfrsmatrix_cmpfvec(vec[8], a, vec[3]);
	#else // USE_OMP_VERSION
		mul_cmpfrsmatrix_cmpfvec(vec[8], a, vec[3]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cdrsmatrix_cmpfvec(vec[8], a, vec[3]);
	#else // USE_OMP_VERSION
		mul_cdrsmatrix_cmpfvec(vec[8], a, vec[3]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_cmpfmatrix_cmpfvec(vec[8], a, vec[3]);
#endif // USE_SPARSE_VERSION

		/* precondition */
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_cmpfvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
			// normalize
			//normi_cmpfvector(dtmp, dvec[0]);
			//mpc_set_f(ctmp, dtmp, MPC_RNDNN);
			//mpf_ui_div(dtmp1, 1UL, dtmp); mpc_set_f(ctmp1, dtmp1, MPC_RNDNN);
			//cmul_cmpfvector(dvec[1], ctmp1, dvec[0]);
			//solve_iLU0_cdrsmatrix_cmpfvec(dvec[0], ilu, dvec[1]);
			solve_iLU0_cdrsmatrix_cmpfvec(vec[8], ilu, dvec[0]);
			// denormalize
			//cmul_cmpfvector(vec[8], ctmp, dvec[0]);
#else // USE_SPARSE_D_VERSION
			solve_iLU0_cmpfrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

		ip_cmpfvector(alpha_den, vec[2], vec[8]);
		if(mpc_cmp_si(alpha_den, 0L) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(MPFCGS, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		mpc_div(alpha, rho, alpha_den, MPC_RNDNN);

		/* q = u - alpha  vt */
		//mpf_neg(dtmp, alpha);
		sub_cmul_cmpfvector(vec[7], vec[5], alpha, vec[8]);

		/* ut = u + q */
		add_cmpfvector(vec[6], vec[5], vec[7]);

		/* x = x + alpha ut */
		add_cmul_cmpfvector(vec[0], vec[0], alpha, vec[6]);

		/* residual */
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cmpfrsmatrix_cmpfvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
		mul_cmpfrsmatrix_cmpfvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cdrsmatrix_cmpfvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
		mul_cdrsmatrix_cmpfvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_cmpfmatrix_cmpfvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION

		/* precondition */
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_cmpfvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
			// normalize
			//normi_cmpfvector(dtmp, dvec[0]);
			//mpc_set_f(ctmp, dtmp, MPC_RNDNN);
			//mpf_ui_div(dtmp1, 1UL, dtmp); mpc_set_f(ctmp1, dtmp1, MPC_RNDNN);
			//cmul_cmpfvector(dvec[1], ctmp1, dvec[0]);
			//solve_iLU0_cdrsmatrix_cmpfvec(dvec[0], ilu, dvec[1]);
			solve_iLU0_cdrsmatrix_cmpfvec(vec[8], ilu, dvec[0]);
			// denormalize
			//cmul_cmpfvector(vec[8], ctmp, dvec[0]);
#else // USE_SPARSE_D_VERSION
			solve_iLU0_cmpfrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

		//mpf_neg(dtmp, alpha);
		sub_cmul_cmpfvector(vec[1], vec[1], alpha, vec[8]);

		//ip_cmpfvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		//mpf_sqrt(dtmp, beta_num);
        norm2_cmpfvector(dtmp, vec[1]);
#ifdef USE_PRECOND
		if(norm2_res_history != NULL) set_mpfvector_i(norm2_res_history, times + 1, dtmp);
#endif // USE_PRECOND

		mpf_mul(dtmp1, reps, init_resnorm);
		mpf_add(dtmp1, dtmp1, aeps);
		if(mpf_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_cmpfvector(answer, vec[0]);
			return_val = times;
			break;
		}

		mpc_set(old_rho, rho, MPC_RNDNN);
	}

	/* Not converge */
	subst_cmpfvector(answer, vec[0]);

	/* free vec[0]..[8]; */
	for(i = 0; i < 9; i++)
		free_cmpfvector(vec[i]);

#ifdef USE_PRECOND
	free_cmpfvector(dvec[0]); // = init2_cmpfvector(dim, prec);
	free_cmpfvector(dvec[1]); // = init2_cmpfvector(dim, prec);
	#if defined(USE_SPARSE_D_VERSION) && defined(USE_VEC_D)
	free_cdvector(in_dvec[0]);
	free_cdvector(in_dvec[1]);
	#endif // defined(USE_SPARSE_D_VERSION) && defined(USE_VEC_D)
#endif // USE_PRECOND

	mpc_clear(alpha); mpc_clear(alpha_num); mpc_clear(alpha_den);
	mpc_clear(rho); mpc_clear(old_rho);
	mpc_clear(beta); mpc_clear(beta_num);
	mpc_clear(ctmp); mpc_clear(ctmp1);
	mpf_clear(dtmp); mpf_clear(dtmp1); mpf_clear(init_resnorm);

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(CMPFCGS, %ld)\n", times);
		return_val = -5;
	}

	// Fix!
	return return_val;
}

/************************************************************/
/*                                                          */
/*             BiCGSTAB Method for Complex Matrix           */
/*                                 (Multi-Precision)        */
/*                                                          */
/*                 ver. 0.0 2024-11-20 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CMPFBiCGSTAB_sp(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CMPFBiCGSTAB_sp_iLU0(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CMPFRSMatrix ilu, MPFVector norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CMPFBiCGSTAB_sp(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
		#else // USE_PRECOND
		long int CMPFBiCGSTAB_sp_iLU0(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CMPFRSMatrix ilu, MPFVector norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CMPFBiCGSTAB_sp_d(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CMPFBiCGSTAB_sp_d_iLU0(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CDRSMatrix ilu, MPFVector norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CMPFBiCGSTAB_sp_d(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
		#else // USE_PRECOND
		long int CMPFBiCGSTAB_sp_d_iLU0(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CDRSMatrix ilu, MPFVector norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifndef USE_PRECOND
	long int CMPFBiCGSTAB(CMPFVector answer, CMPFMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
	#else // USE_PRECOND
	long int CMPFBiCGSTAB_iLU0(CMPFVector answer, CMPFMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CMPFMatrix ilu, MPFVector norm2_res_history)
	#endif // USE_PRECOND
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       CMPFVector answer: Solution for Ax = b             */
/*       CMPFMatrix a: Coefficient matrix A                 */
/*                                       (given by user)    */
/*       CMPFVector b: Constant vector b   (given by user)  */
/*       mpf_t reps: Relative tolerance (given by user)     */
/*       mpf_t aeps: Absolute tolerance (given by user)     */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       CMPFVector answer: Solution for Ax = b             */
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
    mpc_t ctmp;
	mpc_t alpha, alpha_num, alpha_den;
	mpc_t beta, beta_num;
	mpc_t rho, old_rho;
	mpc_t omega, omega_den;
	mpf_t init_resnorm;
	CMPFVector vec[9]; /* Temporary Vectors */
	CMPFVector dvec[2]; // for Preconditioning

	dim = answer->dim;
	prec = answer->prec;

/* Set initial value */
    mpc_init2(ctmp, prec);
	mpc_init2(alpha, prec); mpc_init2(alpha_num, prec); mpc_init2(alpha_den, prec);
	mpc_init2(rho, prec); mpc_init2(old_rho, prec);
	mpc_init2(beta, prec); mpc_init2(beta_num, prec);
	mpc_init2(omega, prec); mpc_init2(omega_den, prec);

	mpf_init2(dtmp, prec); mpf_init2(dtmp1, prec); mpf_init2(init_resnorm, prec);

	for(i = 0; i < 9; i++)
		vec[i] = init2_cmpfvector(dim, prec);

#ifdef USE_PRECOND
	dvec[0] = init2_cmpfvector(dim, prec);
	dvec[1] = init2_cmpfvector(dim, prec);
#endif // USE_PRECOND

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0]  == w*/
	/* vec[2] ... (b - a * vec[0])^T  == wt */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... v */
	/* vec[6] ... s */
	/* vec[7] ... s^T */
	/* vec[8] ... t */

	subst_cmpfvector(vec[1], b);
#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_cmpfvector(dvec[0], vec[1]);
#ifdef USE_SPARSE_D_VERSION
		solve_iLU0_cdrsmatrix_cmpfvec(vec[1], ilu, dvec[0]);
#else // USE_SPRARSE_D_VERSION
		solve_iLU0_cmpfrsmatrix(vec[1], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
	}
#endif // USE_PRECOND
	//subst_cmpfvector(vec[2], b);
	subst_cmpfvector(vec[2], vec[1]);

	//ip_cmpfvector(beta_num, vec[1], vec[1]);
	//mpf_sqrt(init_resnorm, beta_num);
    norm2_cmpfvector(init_resnorm, vec[1]);
	#ifdef USE_PRECOND
	if(norm2_res_history != NULL) set_mpfvector_i(norm2_res_history, 0, init_resnorm);
	#endif // USE_PRECOND

	mpc_set_ui(old_rho, 0UL, MPC_RNDNN);
	mpc_set_ui(rho, 0UL, MPC_RNDNN);
	return_val = 0;

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_cmpfvector(rho, vec[2], vec[1]);

		if(mpc_cmp_si(rho, 0L) == 0)
		{
			fprintf(stderr, "Rho is zero!(CMPFBiCGSTAB, %ld)\n", times);
			return_val = -1; // Fix!
			break;
		}

		if(times == 0)
		{
			/* p := r */
			subst_cmpfvector(vec[3], vec[1]);
		}
		else
		{
			mpc_div(beta, rho, old_rho, MPC_RNDNN);
			mpc_div(ctmp, alpha, omega, MPC_RNDNN);
			mpc_mul(beta, beta, ctmp, MPC_RNDNN);

			/* p := r + beta (p - omega v) */
			//mpf_neg(dtmp, omega);
			sub_cmul_cmpfvector(vec[4], vec[3], omega, vec[5]);
			add_cmul_cmpfvector(vec[3], vec[1], beta, vec[4]);
		}
		
		/* v := Apt */
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cmpfrsmatrix_cmpfvec(vec[5], a, vec[3]);
	#else // USE_OMP_VERSION
		mul_cmpfrsmatrix_cmpfvec(vec[5], a, vec[3]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cdrsmatrix_cmpfvec(vec[5], a, vec[3]);
	#else // USE_OMP_VERSION
		mul_cdrsmatrix_cmpfvec(vec[5], a, vec[3]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_cmpfmatrix_cmpfvec(vec[5], a, vec[3]);
#endif // USE_SPARSE_VERSION

		/* precondition */
		// v := M^(-1)Apt -> Solve Mv = Apt for v
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_cmpfvector(dvec[0], vec[5]);
#ifdef USE_SPARSE_D_VERSION
			solve_iLU0_cdrsmatrix_cmpfvec(vec[5], ilu, dvec[0]);
#else // USE_SPRARSE_D_VERSION
			solve_iLU0_cmpfrsmatrix(vec[5], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

		ip_cmpfvector(alpha_den, vec[2], vec[5]);
		if(mpc_cmp_si(alpha_den, 0L) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(CMPFBiCGSTAB, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		mpc_div(alpha, rho, alpha_den, MPC_RNDNN);

		/* s = r - alpha v */
		//mpf_neg(dtmp, alpha);
		sub_cmul_cmpfvector(vec[6], vec[1], alpha, vec[5]);

		/* Stopping Criteria */
		norm2_cmpfvector(dtmp, vec[6]);
		mpf_mul(dtmp1, reps, init_resnorm);
		mpf_add(dtmp1, dtmp1, aeps);
		if(mpf_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			/* x = x + alpha pt */
			add_cmul_cmpfvector(vec[0], vec[0], alpha, vec[3]);

			subst_cmpfvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		/* precondition */

#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cmpfrsmatrix_cmpfvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
		mul_cmpfrsmatrix_cmpfvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cdrsmatrix_cmpfvec(vec[8], a, vec[6]);	
	#else // USE_OMP_VERSION
		mul_cdrsmatrix_cmpfvec(vec[8], a, vec[6]);	
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_cmpfmatrix_cmpfvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION

		/* precondition */
		// v := M^(-1)Apt -> Solve Mv = Apt for v
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_cmpfvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
			solve_iLU0_cdrsmatrix_cmpfvec(vec[8], ilu, dvec[0]);
#else // USE_SPRARSE_D_VERSION
			solve_iLU0_cmpfrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

		/* omega = (t, s) / (t, t) */
		ip_cmpfvector(omega_den, vec[8], vec[8]);
		if(mpc_cmp_si(omega_den, 0L) == 0)
		{
			fprintf(stderr, "Denominator of Omega is zero!(MPFBiCGSTAB, %ld)\n", times);
			return_val = -3; // Fix!
			break;
		}
		ip_cmpfvector(omega, vec[8], vec[6]);
		if(mpc_cmp_si(omega, 0L) == 0)
		{
			fprintf(stderr, "Numerator of Omega is zero!(MPFBiCGSTAB, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}
		mpc_div(omega, omega, omega_den, MPC_RNDNN);

		/* x = x + alpha pt + omega st */
		add_cmul_cmpfvector(vec[4], vec[0], alpha, vec[3]);
		add_cmul_cmpfvector(vec[0], vec[4], omega, vec[6]);

		/* residual */
		//mpf_neg(dtmp, omega);
		sub_cmul_cmpfvector(vec[1], vec[6], omega, vec[8]);

		//ip_cmpfvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		//mpf_sqrt(dtmp, beta_num);
        norm2_cmpfvector(dtmp, vec[1]);
#ifdef USE_PRECOND
		if(norm2_res_history != NULL) set_mpfvector_i(norm2_res_history, times + 1, dtmp);
#endif // USE_PRECOND

		mpf_mul(dtmp1, reps, init_resnorm);
		mpf_add(dtmp1, dtmp1, aeps);
		if(mpf_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_cmpfvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		mpc_set(old_rho, rho, MPC_RNDNN);
	}

	/* Not converge */
	subst_cmpfvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 9; i++)
		free_cmpfvector(vec[i]);

#ifdef USE_PRECOND
	free_cmpfvector(dvec[0]);
	free_cmpfvector(dvec[1]);
#endif // USE_PRECOND

    mpc_clear(ctmp);
	mpc_clear(alpha); mpc_clear(alpha_num); mpc_clear(alpha_den);
	mpc_clear(rho); mpc_clear(old_rho);
	mpc_clear(beta); mpc_clear(beta_num);
	mpc_clear(omega); mpc_clear(omega_den);
	mpf_clear(dtmp); mpf_clear(dtmp1); mpf_clear(init_resnorm);

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(CMPFBiCGSTAB, %ld)\n", times);
		return_val = -5;
	}

	// Fix!
	return return_val;
}

/************************************************************/
/*                                                          */
/* GPBiCG(Generalized Product Bi-CG) Method                 */
/*                                      for Complex Matrix  */
/*                                 (Multi-Precision)        */
/*                                                          */
/*                 ver. 0.0 2024-11-20 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CMPFGPBiCG_sp(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CMPFGPBiCG_sp_iLU0(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CMPFRSMatrix ilu, MPFVector norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CMPFGPBiCG_sp(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
		#else // USE_PRECOND
		long int CMPFGPBiCG_sp_iLU0(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CMPFRSMatrix ilu, MPFVector norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CMPFGPBiCG_sp_d(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CMPFGPBiCG_sp_d_iLU0(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CDRSMatrix ilu, MPFVector norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CMPFGPBiCG_sp_d(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
		#else // USE_PRECOND
		long int CMPFGPBiCG_sp_d_iLU0(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CDRSMatrix ilu, MPFVector norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifndef USE_PRECOND
	long int CMPFGPBiCG(CMPFVector answer, CMPFMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
	#else // USE_PRECOND
	long int CMPFGPBiCG_iLU0(CMPFVector answer, CMPFMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes, CMPFMatrix ilu, MPFVector norm2_res_history)
	#endif // USE_PRECOND
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       CMPFVector answer: Solution for Ax = b             */
/*       CMPFMatrix a: Coefficient matrix A                 */
/*                                       (given by user)    */
/*       CMPFVector b: Constant vector b   (given by user)  */
/*       mpf_t reps: Relative tolerance (given by user)     */
/*       mpf_t aeps: Absolute tolerance (given by user)     */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       CMPFVector answer: Solution for Ax = b             */
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
    mpc_t ctmp, ctmp1;
	mpc_t alpha, alpha_num, alpha_den;
	mpc_t beta, beta_num;
	mpc_t rho, old_rho;
	mpc_t mu[5], tau, zeta, eta;
	mpf_t init_resnorm;
	CMPFVector vec[12]; /* Temporary Vectors */
	CMPFVector dvec[2]; // Preconditioning

	dim = answer->dim;
	prec = answer->prec;

/* Set initial value */
	mpc_init2(alpha, prec);
	mpc_init2(alpha_num, prec);
	mpc_init2(alpha_den, prec);
	mpc_init2(rho, prec);
	mpc_init2(old_rho, prec);
	mpc_init2(beta, prec);
	mpc_init2(beta_num, prec);
	mpc_init2(tau, prec); mpc_init2(zeta, prec); mpc_init2(eta, prec);
    mpc_init2(ctmp, prec); mpc_init2(ctmp1, prec);
	mpf_init2(dtmp, prec);
	mpf_init2(dtmp1, prec);
	for(i = 0; i < 5; i++)
		mpc_init2(mu[i], prec);
	mpf_init2(init_resnorm, prec);

	for(i = 0; i < 12; i++)
		vec[i] = init2_cmpfvector(dim, prec);

#ifdef USE_PRECOND
	dvec[0] = init2_cmpfvector(dim, prec);
	dvec[1] = init2_cmpfvector(dim, prec);
#endif // USE_PRECOND

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

	subst_cmpfvector(vec[1], b);
#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_cmpfvector(dvec[0], vec[1]);
#ifdef USE_SPARSE_D_VERSION
		solve_iLU0_cdrsmatrix_cmpfvec(vec[1], ilu, dvec[0]);
#else // USE_SPRARSE_D_VERSION
		solve_iLU0_cmpfrsmatrix(vec[1], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
	}
#endif // USE_PRECOND
	//subst_cmpfvector(vec[2], b);
	subst_cmpfvector(vec[2], vec[1]);

	//ip_cmpfvector(beta_num, vec[1], vec[1]);
	//mpf_sqrt(init_resnorm, beta_num);
    norm2_cmpfvector(init_resnorm, vec[1]);
#ifdef USE_PRECOND
	if(norm2_res_history != NULL) set_mpfvector_i(norm2_res_history, 0, init_resnorm);
#endif// USE_PRECOND

	mpc_set_ui(old_rho, 0UL, MPC_RNDNN);
	mpc_set_ui(rho, 0UL, MPC_RNDNN);
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_cmpfvector(rho, vec[2], vec[1]);

		if(mpc_cmp_si(rho, 0L) == 0)
		{
			fprintf(stderr, "Rho is zero!(CMPFGPBiCG, %ld)\n", times);
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
			subst_cmpfvector(vec[3], vec[1]);
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_cmpfrsmatrix_cmpfvec(vec[4], a, vec[3]);
	#else // USE_OMP_VERSION
			mul_cmpfrsmatrix_cmpfvec(vec[4], a, vec[3]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_cdrsmatrix_cmpfvec(vec[4], a, vec[3]);
	#else // USE_OMP_VERSION
			mul_cdrsmatrix_cmpfvec(vec[4], a, vec[3]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
			mul_cmpfmatrix_cmpfvec(vec[4], a, vec[3]);
#endif // USE_SPARSE_VERSION

#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_cmpfvector(dvec[0], vec[4]);
#ifdef USE_SPARSE_D_VERSION
				solve_iLU0_cdrsmatrix_cmpfvec(vec[4], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
				solve_iLU0_cmpfrsmatrix(vec[4], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
			}
#endif // USE_PRECOND

			ip_cmpfvector(alpha_den, vec[2], vec[4]);
			if(mpc_cmp_si(alpha_den, 0L) == 0)
			{
				fprintf(stderr, "Denominator of Alpha is zero!(CMPFGPBiCG, %ld)\n", times);
				return_val = -2; // Fix!
				break; // Fix!
			}
			mpc_div(alpha, rho, alpha_den, MPC_RNDNN);

			//mpf_neg(dtmp, alpha);
			sub_cmul_cmpfvector(vec[6], vec[1], alpha, vec[4]);
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_cmpfrsmatrix_cmpfvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
			mul_cmpfrsmatrix_cmpfvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_cdrsmatrix_cmpfvec(vec[8], a, vec[6]);
	#else // ifdef USE_OMP_VERSION
			mul_cdrsmatrix_cmpfvec(vec[8], a, vec[6]);
	#endif //def USE_OMP_VERSION
#else // USE_SPARSE_VERSION
			mul_cmpfmatrix_cmpfvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION

#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_cmpfvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
				solve_iLU0_cdrsmatrix_cmpfvec(vec[8], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
				solve_iLU0_cmpfrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
			}
#endif // USE_PRECOND

			cmul_cmpfvector(vec[10], alpha, vec[4]);
			sub_cmpfvector(vec[10], vec[10], vec[1]);
			ip_cmpfvector(mu[1], vec[8], vec[6]);
			ip_cmpfvector(mu[4], vec[8], vec[8]);
			if(mpc_cmp_si(mu[4], 0L) == 0)
			{
				fprintf(stderr, "Denominator of mu[4] is zero!(CMPFGPBiCG, %ld)\n", times);
				return_val = -3; // Fix!
				break;
			}
			mpc_div(zeta, mu[1], mu[4], MPC_RNDNN);
			mpc_set_ui(eta, 0UL, MPC_RNDNN);
			cmul_cmpfvector(vec[7], zeta, vec[4]);
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

			mpc_div(beta, rho, old_rho, MPC_RNDNN);
			mpc_div(ctmp, alpha, zeta, MPC_RNDNN);
			mpc_mul(beta, beta, ctmp, MPC_RNDNN);

			add_cmul_cmpfvector(vec[9], vec[8], beta, vec[4]);
			sub_cmpfvector(vec[3], vec[3], vec[7]);
			add_cmul_cmpfvector(vec[3], vec[1], beta, vec[3]);
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_cmpfrsmatrix_cmpfvec(vec[4], a, vec[3]);
	#else // ifdef USE_OMP_VERSION
			mul_cmpfrsmatrix_cmpfvec(vec[4], a, vec[3]);
	#endif //def USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_cdrsmatrix_cmpfvec(vec[4], a, vec[3]);
	#else // ifdef USE_OMP_VERSION
			mul_cdrsmatrix_cmpfvec(vec[4], a, vec[3]);
	#endif //def USE_OMP_VERSION
#else // USE_SPARSE_VERSION
			mul_cmpfmatrix_cmpfvec(vec[4], a, vec[3]);
#endif // USE_SPARSE_VERSION

#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_cmpfvector(dvec[0], vec[4]);
#ifdef USE_SPARSE_D_VERSION
				solve_iLU0_cdrsmatrix_cmpfvec(vec[4], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
				solve_iLU0_cmpfrsmatrix(vec[4], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
			}
#endif // USE_PRECOND

			ip_cmpfvector(alpha_den, vec[2], vec[4]);
			if(mpc_cmp_si(alpha_den, 0L) == 0)
			{
				fprintf(stderr, "Denominator of Alpha is zero!(CMPFGPBiCG, %ld)\n", times);
				return_val = -2; // Fix!
				break; // Fix!
			}
			mpc_div(alpha, rho, alpha_den, MPC_RNDNN);
			sub_cmpfvector(vec[5], vec[6], vec[1]);
			//mpf_neg(dtmp, alpha);
			sub_cmul_cmpfvector(vec[6], vec[1], alpha, vec[4]);
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_cmpfrsmatrix_cmpfvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
			//mul_cmpfrsmatrix_cmpfvec(vec[4], a, vec[3]);
			mul_cmpfrsmatrix_cmpfvec(vec[8], a, vec[6]);
	#endif //def USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_cdrsmatrix_cmpfvec(vec[8], a, vec[6]);
	#else // ifdef USE_OMP_VERSION
			//mul_cdrsmatrix_cmpfvec(vec[4], a, vec[3]);
			mul_cdrsmatrix_cmpfvec(vec[8], a, vec[6]);
	#endif // def USE_OMP_VERSION
#else // USE_SPARSE_VERSION
			//mul_cmpfmatrix_cmpfvec(vec[4], a, vec[3]);
			mul_cmpfmatrix_cmpfvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION

#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_cmpfvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
				solve_iLU0_cdrsmatrix_cmpfvec(vec[8], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
				solve_iLU0_cmpfrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
			}
#endif // USE_PRECOND

			sub_cmpfvector(vec[10], vec[9], vec[4]);

			//mpf_neg(dtmp, alpha);
			sub_cmul_cmpfvector(vec[10], vec[5], alpha, vec[10]);
			ip_cmpfvector(mu[0], vec[10], vec[10]);
			ip_cmpfvector(mu[1], vec[8], vec[6]);
			ip_cmpfvector(mu[2], vec[10], vec[6]);
			ip_cmpfvector(mu[3], vec[8], vec[10]);
			ip_cmpfvector(mu[4], vec[8], vec[8]);

			/* tau = mu[4] * mu[0] - mu[3] * mu[3]; */
			mpc_mul(ctmp, mu[4], mu[0], MPC_RNDNN);
			mpc_mul(ctmp1, mu[3], mu[3], MPC_RNDNN);
			mpc_sub(tau, ctmp, ctmp1, MPC_RNDNN);

			if(mpc_cmp_si(tau, 0L) == 0)
			{
				fprintf(stderr, "Denominator of Tau is zero!(CMPFGPBiCG, %ld)\n", times);
				return_val = -3; // Fix!
				break;
			}

			/* zeta = (mu[0] * mu[1] - mu[2] * mu[3]) / tau; */
			mpc_mul(ctmp, mu[0], mu[1], MPC_RNDNN);
			mpc_mul(ctmp1, mu[2], mu[3], MPC_RNDNN);
			mpc_sub(zeta, ctmp, ctmp1, MPC_RNDNN);
			mpc_div(zeta, zeta, tau, MPC_RNDNN);

			/* eta = (mu[4] * mu[2] - mu[3] * mu[1]) / tau; */
			mpc_mul(ctmp, mu[4], mu[2], MPC_RNDNN);
			mpc_mul(ctmp1, mu[3], mu[1], MPC_RNDNN);
			mpc_sub(eta, ctmp, ctmp1, MPC_RNDNN);
			mpc_div(eta, eta, tau, MPC_RNDNN);

			/* u = zeta * q + eta * (s + k_beta * u); */
			add_cmul_cmpfvector(vec[7], vec[5], beta, vec[7]);
			cmul_cmpfvector(vec[7], eta, vec[7]);
			add_cmul_cmpfvector(vec[7], vec[7], zeta, vec[4]);
		}
		/* z = zeta * ret_res + eta * z - alpha * u; */
		cmul2_cmpfvector(vec[11], eta);
		add_cmul_cmpfvector(vec[11], vec[11], zeta, vec[1]);
		//mpf_neg(dtmp, alpha);
		sub_cmul_cmpfvector(vec[11], vec[11], alpha, vec[7]);

		/* x = x + alpha p + z */
		add_cmul_cmpfvector(vec[0], vec[0], alpha, vec[3]);
		add2_cmpfvector(vec[0], vec[11]);

		/* residual */
		/* r = t - eta * y - zeta * v; */
		//mpf_neg(dtmp, eta);
		sub_cmul_cmpfvector(vec[1], vec[6], eta, vec[10]);
		//mpf_neg(dtmp, zeta);
		sub_cmul_cmpfvector(vec[1], vec[1], zeta, vec[8]);

		//ip_cmpfvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		//mpf_sqrt(dtmp, beta_num);
        norm2_cmpfvector(dtmp, vec[1]);
#ifdef USE_PRECOND
		if(norm2_res_history != NULL) set_mpfvector_i(norm2_res_history, times + 1, dtmp);
#endif// USE_PRECOND
		mpf_mul(dtmp1, reps, init_resnorm);
		mpf_add(dtmp1, dtmp1, aeps);
		if(mpf_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_cmpfvector(answer, vec[0]);
			return_val = times; // Fix!
			break;
		}

		if(mpc_cmp_si(zeta, 0L) == 0)
		{
			fprintf(stderr, "Denominator of Zeta is zero!(CMPFGPBiCG, %ld)\n", times);
			return_val = -4; // Fix!
			break;
		}

		mpc_set(old_rho, rho, MPC_RNDNN);
	}

	/* Not converge */
	subst_cmpfvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 12; i++)
		free_cmpfvector(vec[i]);

#ifdef USE_PRECOND
	free_cmpfvector(dvec[0]);
	free_cmpfvector(dvec[1]);
#endif // USE_PRECOND

	mpc_clear(alpha); mpc_clear(alpha_num); mpc_clear(alpha_den);
	mpc_clear(rho); mpc_clear(old_rho);
	mpc_clear(beta); mpc_clear(beta_num);
	mpc_clear(tau); mpc_clear(zeta); mpc_clear(eta);
    mpc_clear(ctmp); mpc_clear(ctmp1);
	for(i = 0; i < 5; i++)
		mpc_clear(mu[i]);
	mpf_clear(dtmp); mpf_clear(dtmp1); mpf_clear(init_resnorm);

	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(CMPFGPBiCG, %ld)\n", times);
		return_val = -5; // Fix!
	}

	// Fix!
	return return_val;
}

#endif // USE_GMP
