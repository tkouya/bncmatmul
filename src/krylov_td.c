/********************************************************************************/
/* krylov_td.c:                                                                 */
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

#ifdef USE_TDLINEAR

/************************************************************/
/*                                                          */
/* Bi-Conjugate-Gradient Method for Real Matrix             */
/*                                     (TD Precision)       */
/*                                                          */
/*                 ver. 0.0 2024-04-23 (Tue) Tomonori Kouya */
/*                 ver. 0.1 2024-10-21 (Mon) Tomonori Kouya */
/*                 ver. 0.2 2025-02-20 (Thu) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifdef USE_PRECOND
		long int _bncomp_TDBiCG_sp_iLU0(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDRSMatrix ilu, double *norm2_res_history)
		#else // USE_PRECOND
		long int _bncomp_TDBiCG_sp(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifdef USE_PRECOND
		long int TDBiCG_sp_iLU0(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDRSMatrix ilu, double *norm2_res_history)
		#else // USE_PRECOND
		long int TDBiCG_sp(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		#ifdef USE_PRECOND
		long int _bncomp_TDBiCG_sp_d_iLU0(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history)
		#else // USE_PRECOND
		long int _bncomp_TDBiCG_sp_d(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifdef USE_PRECOND
		long int TDBiCG_sp_d_iLU0(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history)
		#else // USE_PRECOND
		long int TDBiCG_sp_d(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifdef USE_PRECOND
		long int _bncomp_TDBiCG_iLU0(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDMatrix ilu, double *norm2_res_history)
		#else // USE_PRECOND
		long int _bncomp_TDBiCG(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifdef USE_PRECOND
		long int TDBiCG_iLU0(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDMatrix ilu, double *norm2_res_history)
		#else // USE_PRECOND
		long int TDBiCG(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       TDVector answer: Solution for Ax = b                */
/*       TDMatrix a: Coefficient matrix A                    */
/*                                       (given by user)    */
/*       TDVector b: Constant vector b   (given by user)     */
/*       double reps[TDSIZE]: Relative tolerance (given by user)    */
/*       double aeps[TDSIZE]: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       TDVector answer: Solution for Ax = b                */
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
	double alpha[TDSIZE], alpha_num[TDSIZE], alpha_den[TDSIZE], minus_alpha[TDSIZE];
	double beta[TDSIZE], beta_num[TDSIZE];
	double rho[TDSIZE], old_rho[TDSIZE];
	double dtmp[TDSIZE], dtmp1[TDSIZE], init_resnorm[TDSIZE];
	TDVector vec[9]; /* Temporary Vectors */
	TDVector dvec[2]; // for Preconditioning

	dim = answer->dim;

/* Set initial value */
	for(i = 0; i < 9; i++)
		vec[i] = init_tdvector(dim);

#ifdef USE_PRECOND
	dvec[0] = init_tdvector(dim);
	dvec[1] = init_tdvector(dim);
#endif // USE_PRECOND

/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0]  == w*/
	/* vec[2] ... (b - a * vec[0])^T  == wt */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... z */
	/* vec[6] ... z^T */

	subst_tdvector(vec[1], b); 
	subst_tdvector(vec[2], b);
	subst_tdvector(vec[7], vec[1]);
	subst_tdvector(vec[8], vec[2]);
#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_tdvector(dvec[0], vec[1]);
		subst_tdvector(dvec[1], vec[2]);
#ifdef USE_SPARSE_D_VERSION
		solve_iLU0_drsmatrix_tdvec(vec[7], ilu, dvec[0]);
		solve_iLU0t_drsmatrix_tdvec(vec[8], ilu, dvec[1]);
#else // USE_SPARSE_D_VERSION
		solve_iLU0_tdrsmatrix(vec[7], ilu, dvec[0]);
		solve_iLU0t_tdrsmatrix(vec[8], ilu, dvec[1]);
#endif // 
	}
#endif // USE_PRECOND

	//ip_tdvector(beta_num, vec[1], vec[1]);
    //ip_tdvector(beta_num, vec[1], vec[1]);
	//init_resnorm = sqrt(beta_num);
    //rtd_sqrt(init_resnorm, beta_num);
    norm2_tdvector(init_resnorm, vec[1]);
	#ifdef USE_PRECOND
	if(norm2_res_history != NULL) norm2_res_history[0] = init_resnorm[0];
	#endif // USE_PRECOND

	//old_rho = 0.0;
	//rho = 0.0;
    rtd_set0(old_rho);
    rtd_set0(rho);
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		//ip_tdvector(rho, vec[2], vec[1]);
        //ip_tdvector(rho, vec[2], vec[1]);
		ip_tdvector(rho, vec[2], vec[7]);

		//if(rtd_cmp_ui(rho, 0UL) == 0)
        if(rtd_cmp_ui(rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(TDBiCG, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* p := w, pt := wt */
			//subst_tdvector(vec[3], vec[1]);
			//subst_tdvector(vec[4], vec[2]);
			subst_tdvector(vec[3], vec[7]);
			subst_tdvector(vec[4], vec[8]);
		}
		else
		{
			//beta = rho / old_rho;
            rtd_div(beta, rho, old_rho);

			/* p := w + beta p, pt := wt + beta * pt */
			//add_cmul_tdvector(vec[3], vec[1], beta, vec[3]);
			//add_cmul_tdvector(vec[4], vec[2], beta, vec[4]);
			add_cmul_tdvector(vec[3], vec[7], beta, vec[3]);
			add_cmul_tdvector(vec[4], vec[8], beta, vec[4]);
		}

		/* z := Ap, zt := A^T pt */
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_tdrsmatrix_tdvec(vec[5], a, vec[3]);
		_bncomp_mul_tdrsmatrixt_tdvec(vec[6], a, vec[4]);
	#else // USE_OMP_VERSION
		mul_tdrsmatrix_tdvec(vec[5], a, vec[3]);
		mul_tdrsmatrixt_tdvec(vec[6], a, vec[4]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_drsmatrix_tdvec(vec[5], a, vec[3]);
		_bncomp_mul_drsmatrixt_tdvec(vec[6], a, vec[4]);
	#else // USE_OMP_VERSION
		mul_drsmatrix_tdvec(vec[5], a, vec[3]);
		mul_drsmatrixt_tdvec(vec[6], a, vec[4]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_tdmatrix_tdvec(vec[5], a, vec[3]);
		mul_tdmatrixt_tdvec(vec[6], a, vec[4]);
#endif // USE_SPARSE_VERSION

		//ip_tdvector(alpha_den, vec[4], vec[5]);
        ip_tdvector(alpha_den, vec[4], vec[5]);
		//if(rtd_cmp_ui(alpha_den, 0UL) == 0)
        if(rtd_cmp_ui(alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(TDBiCG, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		//alpha = rho / alpha_den;
        rtd_div(alpha, rho, alpha_den);

		/* x = x + alpha p */
		add_cmul_tdvector(vec[0], vec[0], alpha, vec[3]);

		/* residual */
        rtd_neg(minus_alpha, alpha);
		add_cmul_tdvector(vec[1], vec[1], minus_alpha, vec[5]);
		add_cmul_tdvector(vec[2], vec[2], minus_alpha, vec[6]);

		//ip_tdvector(beta_num, vec[1], vec[1]);
        //ip_tdvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		//rtd_sqrt(dtmp, beta_num);
        //rtd_sqrt(dtmp, beta_num);
        norm2_tdvector(dtmp, vec[1]);
		#ifdef USE_PRECOND
		if(norm2_res_history != NULL) norm2_res_history[times + 1] = dtmp[0];
		#endif // USE_PRECOND
		// dtmp1 = aeps + reps * init_resnorm)
        rtd_mul(dtmp1, reps, init_resnorm);
        rtd_add(dtmp1, dtmp1, aeps);
		//if(dtmp <= aeps + reps * init_resnorm)
        if(rtd_cmp(dtmp, dtmp1) <= 0)
		{
			subst_tdvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		subst_tdvector(vec[7], vec[1]);
		subst_tdvector(vec[8], vec[2]);
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_tdvector(dvec[0], vec[7]);
			subst_tdvector(dvec[1], vec[8]);
#ifdef USE_SPARSE_D_VERSION
			solve_iLU0_drsmatrix_tdvec(vec[7], ilu, dvec[0]);
			//solve_iLU0t_cdrsmatrix_cddvec(vec[6], ilu, dvec[1]);
			solve_iLU0t_drsmatrix_tdvec(vec[8], ilu, dvec[1]);
#else // USE_SPARSE_D_VERSION
			solve_iLU0_tdrsmatrix(vec[7], ilu, dvec[0]);
			//solve_iLU0t_cddrsmatrix(vec[6], ilu, dvec[1]);
			solve_iLU0t_tdrsmatrix(vec[8], ilu, dvec[1]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

//rtd_set(old_rho, rho);;
        rtd_set(old_rho, rho);
	}

	/* Not converge */
	subst_tdvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 9; i++)
		free_tdvector(vec[i]);

#ifdef USE_PRECOND
	free_tdvector(dvec[0]);
	free_tdvector(dvec[1]);
#endif // USE_PRECOND

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(TDBiCG, %ld)\n", times);
		return_val = -5; // Fix!
	}

	return return_val; // Fix!

}

/************************************************************/
/*                                                          */
/*               CGS Method for Real Matrix                 */
/*                                     (TD Precision)       */
/*                                                          */
/*                 ver. 0.0 2004.11.06 (Sat) Tomonori Kouya */
/*                 ver. 0.1 2012-03-17 (Sat) Tomonori Kouya */
/*                 ver. 0.2 2024-10-24 (Mon) Tomonori Kouya */
/*                 ver. 0.3 2025-02-20 (Thu) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_TDCGS_sp(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_TDCGS_sp_iLU0(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int TDCGS_sp(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int TDCGS_sp_iLU0(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_TDCGS_sp_d(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_TDCGS_sp_d_iLU0(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int TDCGS_sp_d(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int TDCGS_sp_d_iLU0(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_TDCGS(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_TDCGS_iLU0(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int TDCGS(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int TDCGS_iLU0(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       TDVector answer: Solution for Ax = b                */
/*       TDMatrix a: Coefficient matrix A                    */
/*                                       (given by user)    */
/*       TDVector b: Constant vector b   (given by user)     */
/*       double reps[TDSIZE]: Relative tolerance (given by user)    */
/*       double aeps[TDSIZE]: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       TDVector answer: Solution for Ax = b                */
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
	double alpha[TDSIZE], minus_alpha[TDSIZE], alpha_num[TDSIZE], alpha_den[TDSIZE];
	double beta[TDSIZE], beta_num[TDSIZE];
	double rho[TDSIZE], old_rho[TDSIZE];
	double dtmp[TDSIZE], dtmp1[TDSIZE], init_resnorm[TDSIZE];
	TDVector vec[9]; /* Temporary Vectors */
	TDVector dvec[2]; // for preconditioning

	dim = answer->dim;

/* Set initial value */
	for(i = 0; i < 9; i++)
		vec[i] = init_tdvector(dim);

#ifdef USE_PRECOND
		dvec[0] = init_tdvector(dim);
		dvec[1] = init_tdvector(dim);
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

	subst_tdvector(vec[1], b); 
	#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_tdvector(dvec[0], vec[1]);
#ifdef USE_SPARSE_D_VERSION
		solve_iLU0_drsmatrix_tdvec(vec[1], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
		solve_iLU0_tdrsmatrix(vec[1], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
	}
#endif // USE_PRECOND
	//subst_tdvector(vec[2], b);
	subst_tdvector(vec[2], vec[1]);

	//ip_tdvector(beta_num, vec[1], vec[1]);
	//rtd_sqrt(init_resnorm, beta_num);
    norm2_tdvector(init_resnorm, vec[1]);
	#ifdef USE_PRECOND
	if(norm2_res_history != NULL) norm2_res_history[0] = init_resnorm[0];
	#endif // USE_PRECOND

	rtd_set0(old_rho); // = 0.0;
	rtd_set0(rho); // = 0.0;
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_tdvector(rho, vec[2], vec[1]);

		if(rtd_cmp_ui(rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(DCGS, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* u := rt, p := u */
			subst_tdvector(vec[5], vec[1]);
			subst_tdvector(vec[3], vec[5]);
		}
		else
		{
			//beta = rho / old_rho;
            rtd_div(beta, rho, old_rho);

			/* u := r + beta q, p := u + beta * (q + beta p) */
			add_cmul_tdvector(vec[5], vec[1], beta, vec[7]);
			add_cmul_tdvector(vec[8], vec[7], beta, vec[3]);
			add_cmul_tdvector(vec[3], vec[5], beta, vec[8]);
		}

		/* vt := Apt */
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_tdrsmatrix_tdvec(vec[8], a, vec[3]);
	#else // USE_OMP_VERSION
		mul_tdrsmatrix_tdvec(vec[8], a, vec[3]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_drsmatrix_tdvec(vec[8], a, vec[3]);
	#else // USE_OMP_VERSION
		mul_drsmatrix_tdvec(vec[8], a, vec[3]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_tdmatrix_tdvec(vec[8], a, vec[3]);
#endif // USE_SPARSE_VERSION

		/* precondition */
		#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_tdvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
			solve_iLU0_drsmatrix_tdvec(vec[8], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
			solve_iLU0_tdrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

		ip_tdvector(alpha_den, vec[2], vec[8]);
		if(rtd_cmp_ui(alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(TDCGS, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		//alpha = rho / alpha_den;
        rtd_div(alpha, rho, alpha_den);

		/* q = u - alpha  vt */
        rtd_neg(minus_alpha, alpha);
		add_cmul_tdvector(vec[7], vec[5], minus_alpha, vec[8]);

		/* ut = u + q */
		add_tdvector(vec[6], vec[5], vec[7]);

		/* x = x + alpha ut */
		add_cmul_tdvector(vec[0], vec[0], alpha, vec[6]);

		/* residual */
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_tdrsmatrix_tdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
		mul_tdrsmatrix_tdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_drsmatrix_tdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
		mul_drsmatrix_tdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_tdmatrix_tdvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION

		/* precondition */
		#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_tdvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
			solve_iLU0_drsmatrix_tdvec(vec[8], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
			solve_iLU0_tdrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND
		add_cmul_tdvector(vec[1], vec[1], minus_alpha, vec[8]);

		//ip_tdvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		//rtd_sqrt(dtmp, beta_num);
        norm2_tdvector(dtmp, vec[1]);
		#ifdef USE_PRECOND
		if(norm2_res_history != NULL) norm2_res_history[times + 1] = dtmp[0];
		#endif // USE_PRECOND

		// dtmp1 = aeps + reps * init_resnorm)
        rtd_mul(dtmp1, reps, init_resnorm);
        rtd_add(dtmp1, dtmp1, aeps);
		//if(dtmp <= aeps + reps * init_resnorm)
        if(rtd_cmp(dtmp, dtmp1) <= 0)
		{
			subst_tdvector(answer, vec[0]);
			return_val = times; // Fix!
			break;
		}

		rtd_set(old_rho, rho);
	}

	/* Not converge */
	subst_tdvector(answer, vec[0]);

	/* free vec[0]..[8]; */
	for(i = 0; i < 9; i++)
		free_tdvector(vec[i]);

#ifdef USE_PRECOND
	free_tdvector(dvec[0]);
	free_tdvector(dvec[1]);
#endif // USE_PRECOND

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
/*                                     (TD Precision)       */
/*                                                          */
/*                 ver. 0.0 2004.11.06 (Wed) Tomonori Kouya */
/*                 ver. 0.1 2012-03-17 (Sat) Tomonori Kouya */
/*                 ver. 0.2 2024-10-21 (Mon) Tomonori Kouya */
/*                 ver. 0.3 2025-02-20 (Thu) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_TDBiCGSTAB_sp(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_TDBiCGSTAB_sp_iLU0(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int TDBiCGSTAB_sp(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int TDBiCGSTAB_sp_iLU0(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_TDBiCGSTAB_sp_d(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_TDBiCGSTAB_sp_d_iLU0(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int TDBiCGSTAB_sp_d(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int TDBiCGSTAB_sp_d_iLU0(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_TDBiCGSTAB(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_TDBiCGSTAB_iLU0(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int TDBiCGSTAB(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int TDBiCGSTAB_iLU0(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       TDVector answer: Solution for Ax = b                */
/*       TDMatrix a: Coefficient matrix A                    */
/*                                       (given by user)    */
/*       TDVector b: Constant vector b   (given by user)     */
/*       double reps[TDSIZE]: Relative tolerance (given by user)    */
/*       double aeps[TDSIZE]: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       TDVector answer: Solution for Ax = b                */
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
	double alpha[TDSIZE], minus_alpha[TDSIZE], alpha_num[TDSIZE], alpha_den[TDSIZE];
	double beta[TDSIZE], beta_num[TDSIZE];
	double rho[TDSIZE], old_rho[TDSIZE];
	double omega[TDSIZE], minus_omega[TDSIZE], omega_den[TDSIZE];
	double dtmp[TDSIZE], dtmp1[TDSIZE], init_resnorm[TDSIZE];
	TDVector vec[9]; /* Temporary Vectors */
	TDVector dvec[2]; // for Preconditioning

	dim = answer->dim;

/* Set initial value */
	for(i = 0; i < 9; i++)
		vec[i] = init_tdvector(dim);

#ifdef USE_PRECOND
	dvec[0] = init_tdvector(dim);
	dvec[1] = init_tdvector(dim);
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

	subst_tdvector(vec[1], b); 
	#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_tdvector(dvec[0], vec[1]);
#ifdef USE_SPARSE_D_VERSION
		solve_iLU0_drsmatrix_tdvec(vec[1], ilu, dvec[0]);
#else // USE_SPRARSE_D_VERSION
		solve_iLU0_tdrsmatrix(vec[1], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
	}
#endif // USE_PRECOND
	//subst_tdvector(vec[2], b);
	subst_tdvector(vec[2], vec[1]);

	//ip_tdvector(beta_num, vec[1], vec[1]);
	//rtd_sqrt(init_resnorm, beta_num);
    norm2_tdvector(init_resnorm, vec[1]);
	#ifdef USE_PRECOND
	if(norm2_res_history != NULL) norm2_res_history[0] = init_resnorm[0];
	#endif // USE_PRECOND

	rtd_set0(old_rho); // = 0.0;
	rtd_set0(rho); // = 0.0;
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_tdvector(rho, vec[2], vec[1]);

		if(rtd_cmp_ui(rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(TDBiCG, %ld)\n", times);
			return_val= -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* p := r */
			subst_tdvector(vec[3], vec[1]);
		}
		else
		{
			//beta = (rho / old_rho) * (alpha / omega);
            rtd_div(dtmp, rho, old_rho);
            rtd_div(dtmp1, alpha, omega);
            rtd_mul(beta, dtmp, dtmp1);

			/* p := r + beta (p - omega v) */
			rtd_neg(minus_omega, omega);
			add_cmul_tdvector(vec[4], vec[3], minus_omega, vec[5]);
			add_cmul_tdvector(vec[3], vec[1], beta, vec[4]);
		}
		/* precondition */
		
		/* v := Apt */
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_tdrsmatrix_tdvec(vec[5], a, vec[3]);
	#else // USE_OMP_VERSION
		mul_tdrsmatrix_tdvec(vec[5], a, vec[3]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_drsmatrix_tdvec(vec[5], a, vec[3]);
	#else // USE_OMP_VERSION
		mul_drsmatrix_tdvec(vec[5], a, vec[3]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_tdmatrix_tdvec(vec[5], a, vec[3]);
#endif // USE_SPARSE_VERSION

		/* precondition */
		// v := M^(-1)Apt -> Solve Mv = Apt for v
		#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_tdvector(dvec[0], vec[5]);
#ifdef USE_SPARSE_D_VERSION
			solve_iLU0_drsmatrix_tdvec(vec[5], ilu, dvec[0]);
#else // USE_SPRARSE_D_VERSION
			solve_iLU0_tdrsmatrix(vec[5], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

		ip_tdvector(alpha_den, vec[2], vec[5]);
		if(rtd_cmp_ui(alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(TDBiCGSTAB, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		//alpha = rho / alpha_den;
        rtd_div(alpha, rho, alpha_den);

		/* s = r - alpha v */
        rtd_neg(minus_alpha, alpha);
		add_cmul_tdvector(vec[6], vec[1], minus_alpha, vec[5]);

		/* Stopping Criteria */
		norm2_tdvector(dtmp, vec[6]);
		// aeps + reps * init_resnorm
		rtd_mul(dtmp1, reps, init_resnorm);
		rtd_add(dtmp1, dtmp1, aeps);
		if(rtd_cmp(dtmp, dtmp1) <= 0) // aeps + reps * init_resnorm)
		{
			/* x = x + alpha pt */
			add_cmul_tdvector(vec[0], vec[0], alpha, vec[3]);

			subst_tdvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		/* precondition */
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_tdrsmatrix_tdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
		mul_tdrsmatrix_tdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_drsmatrix_tdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
		mul_drsmatrix_tdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_tdmatrix_tdvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION

		/* precondition */
		// v := M^(-1)Apt -> Solve Mv = Apt for v
		#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_tdvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
			solve_iLU0_drsmatrix_tdvec(vec[8], ilu, dvec[0]);
#else // USE_SPRARSE_D_VERSION
			solve_iLU0_tdrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

		/* omega = (t, s) / (t, t) */
		ip_tdvector(omega_den, vec[8], vec[8]);
		if(rtd_cmp_ui(omega_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Omega is zero!(TDBiCGSTAB, %ld)\n", times);
			return_val = -3; // Fix!
			break;
		}
		ip_tdvector(omega, vec[8], vec[6]);
		if(rtd_cmp_ui(omega, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Omega is zero!(TDBiCGSTAB, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}
		//omega = omega / omega_den;
        rtd_div(omega, omega, omega_den);

		/* x = x + alpha pt + omega st */
		add_cmul_tdvector(vec[4], vec[0], alpha, vec[3]);
		add_cmul_tdvector(vec[0], vec[4], omega, vec[6]);

		/* residual */
        rtd_neg(dtmp, omega);
		add_cmul_tdvector(vec[1], vec[6], dtmp, vec[8]);

		//ip_tdvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		//rtd_sqrt(dtmp, beta_num);
        norm2_tdvector(dtmp, vec[1]);
		#ifdef USE_PRECOND
		if(norm2_res_history != NULL) norm2_res_history[times + 1] = dtmp[0];
		#endif // USE_PRECOND

		// dtmp1 = aeps + reps * init_resnorm)
        rtd_mul(dtmp1, reps, init_resnorm);
        rtd_add(dtmp1, dtmp1, aeps);
		//if(dtmp <= aeps + reps * init_resnorm)
        if(rtd_cmp(dtmp, dtmp1) <= 0)
		{
			subst_tdvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		rtd_set(old_rho, rho);
	}

	/* Not converge */
	subst_tdvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 9; i++)
		free_tdvector(vec[i]);

#ifdef USE_PRECOND
	free_tdvector(dvec[0]);
	free_tdvector(dvec[1]);
#endif // USE_PRECOND
	
	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(TDBiCGSTAB, %ld)\n", times);
		return_val = -5;
	}

	return return_val; // Fix!

}

/************************************************************/
/*                                                          */
/* GPBiCG(Generalized Product Bi-CG) Method for Real Matrix */
/*                                     (TD Precision)       */
/*                                                          */
/*                 ver. 0.0 2004.11.06 (Wed) Tomonori Kouya */
/*                 ver. 0.1 2012-03-17 (Sat) Tomonori Kouya */
/*                 ver. 0.2 2024-10-21 (Mon) Tomonori Kouya */
/*                 ver. 0.3 2025-02-20 (Thu) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_TDGPBiCG_sp(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_TDGPBiCG_sp_iLU0(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int TDGPBiCG_sp(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int TDGPBiCG_sp_iLU0(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_TDGPBiCG_sp_d(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_TDGPBiCG_sp_d_iLU0(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int TDGPBiCG_sp_d(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int TDGPBiCG_sp_d_iLU0(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_TDGPBiCG(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_TDGPBiCG_iLU0(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int TDGPBiCG(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int TDGPBiCG_iLU0(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, TDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       TDVector answer: Solution for Ax = b                */
/*       TDMatrix a: Coefficient matrix A                    */
/*                                       (given by user)    */
/*       TDVector b: Constant vector b   (given by user)     */
/*       double reps[TDSIZE]: Relative tolerance (given by user)    */
/*       double aeps[TDSIZE]: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       TDVector answer: Solution for Ax = b                */
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
	double alpha[TDSIZE], minus_alpha[TDSIZE], alpha_num[TDSIZE], alpha_den[TDSIZE];
	double beta[TDSIZE], beta_num[TDSIZE];
	double rho[TDSIZE], old_rho[TDSIZE];
	double mu[5][TDSIZE], tau[TDSIZE], zeta[TDSIZE], minus_zeta[TDSIZE], eta[TDSIZE], minus_eta[TDSIZE];
	double dtmp[TDSIZE], dtmp1[TDSIZE], init_resnorm[TDSIZE];
	TDVector vec[12]; /* Temporary Vectors */
	TDVector dvec[2]; // Preconditioning

	dim = answer->dim;

/* Set initial value */
	for(i = 0; i < 12; i++)
		vec[i] = init_tdvector(dim);

#ifdef USE_PRECOND
	dvec[0] = init_tdvector(dim);
	dvec[1] = init_tdvector(dim);
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

	subst_tdvector(vec[1], b); 
	#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_tdvector(dvec[0], vec[1]);
#ifdef USE_SPARSE_D_VERSION
		solve_iLU0_drsmatrix_tdvec(vec[1], ilu, dvec[0]);
#else // USE_SPRARSE_D_VERSION
		solve_iLU0_tdrsmatrix(vec[1], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
	}
#endif // USE_PRECOND
	//subst_tdvector(vec[2], b);
	subst_tdvector(vec[2], vec[1]);

	//ip_tdvector(beta_num, vec[1], vec[1]);
	//rtd_sqrt(init_resnorm, beta_num);
    norm2_tdvector(init_resnorm, vec[1]);
#ifdef USE_PRECOND
	if(norm2_res_history != NULL) norm2_res_history[0] = init_resnorm[0];
#endif// USE_PRECOND

	rtd_set0(old_rho); // = 0.0;
	rtd_set0(rho); // = 0.0;
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_tdvector(rho, vec[2], vec[1]);

		if(rtd_cmp_ui(rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(TDGPBiCG, %ld)\n", times);
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
			subst_tdvector(vec[3], vec[1]);
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_tdrsmatrix_tdvec(vec[4], a, vec[3]);
	#else // USE_OMP_VERSION
			mul_tdrsmatrix_tdvec(vec[4], a, vec[3]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_drsmatrix_tdvec(vec[4], a, vec[3]);
	#else // USE_OMP_VERSION
			mul_drsmatrix_tdvec(vec[4], a, vec[3]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
			mul_tdmatrix_tdvec(vec[4], a, vec[3]);
#endif // USE_SPARSE_VERSION

#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_tdvector(dvec[0], vec[4]);
#ifdef USE_SPARSE_D_VERSION
				solve_iLU0_drsmatrix_tdvec(vec[4], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
				solve_iLU0_tdrsmatrix(vec[4], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
			}
#endif // USE_PRECOND

			ip_tdvector(alpha_den, vec[2], vec[4]);
			if(rtd_cmp_ui(alpha_den, 0UL) == 0)
			{
				fprintf(stderr, "Denominator of Alpha is zero!(TDGPBiCG, %ld)\n", times);
				return_val = -2; // Fix!
				break; // Fix!
			}
			//alpha = rho / alpha_den;
            rtd_div(alpha, rho, alpha_den);
            rtd_neg(minus_alpha, alpha);
			add_cmul_tdvector(vec[6], vec[1], minus_alpha, vec[4]);
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_tdrsmatrix_tdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
			mul_tdrsmatrix_tdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_drsmatrix_tdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
			mul_drsmatrix_tdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
			mul_tdmatrix_tdvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION

#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_tdvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
				solve_iLU0_drsmatrix_tdvec(vec[8], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
				solve_iLU0_tdrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
			}
#endif // USE_PRECOND

			cmul_tdvector(vec[10], alpha, vec[4]);
			sub_tdvector(vec[10], vec[10], vec[1]);
			ip_tdvector(mu[1], vec[8], vec[6]);
			ip_tdvector(mu[4], vec[8], vec[8]);
			//if(mu[4] == 0.0)
            if(rtd_cmp_ui(mu[4], 0UL) == 0)
			{
				fprintf(stderr, "Denominator of mu[4] is zero!(TDGPBiCG, %ld)\n", times);
				return_val = -3; // Fix!
				break; // Fix!
			}
			//zeta = mu[1] / mu[4];
            rtd_div(zeta, mu[1], mu[4]);
			//eta = 0.0;
            rtd_set0(eta);
			cmul_tdvector(vec[7], zeta, vec[4]);
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
            rtd_div(dtmp, rho, old_rho);
            rtd_div(dtmp1, alpha, zeta);
            rtd_mul(beta, dtmp, dtmp1);

			add_cmul_tdvector(vec[9], vec[8], beta, vec[4]);
			sub_tdvector(vec[3], vec[3], vec[7]);
			add_cmul_tdvector(vec[3], vec[1], beta, vec[3]);
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_tdrsmatrix_tdvec(vec[4], a, vec[3]);
	#else // USE_OMP_VERSION
			mul_tdrsmatrix_tdvec(vec[4], a, vec[3]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_drsmatrix_tdvec(vec[4], a, vec[3]);
	#else // USE_OMP_VERSION
			mul_drsmatrix_tdvec(vec[4], a, vec[3]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
			mul_tdmatrix_tdvec(vec[4], a, vec[3]);
#endif // USE_SPARSE_VERSION

#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_tdvector(dvec[0], vec[4]);
#ifdef USE_SPARSE_D_VERSION
				solve_iLU0_drsmatrix_tdvec(vec[4], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
				solve_iLU0_tdrsmatrix(vec[4], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
			}
#endif // USE_PRECOND

			ip_tdvector(alpha_den, vec[2], vec[4]);
			if(rtd_cmp_ui(alpha_den, 0UL) == 0)
			{
				fprintf(stderr, "Denominator of Alpha is zero!(TDGPBiCG, %ld)\n", times);
				return_val = -2; // Fix!
				break; // Fix!
			}
			//alpha = rho / alpha_den;
            rtd_div(alpha, rho, alpha_den);
			sub_tdvector(vec[5], vec[6], vec[1]);
            rtd_neg(minus_alpha, alpha);
			add_cmul_tdvector(vec[6], vec[1], minus_alpha, vec[4]);
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_tdrsmatrix_tdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
			mul_tdrsmatrix_tdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_drsmatrix_tdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
			mul_drsmatrix_tdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
			mul_tdmatrix_tdvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION

#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_tdvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
				solve_iLU0_drsmatrix_tdvec(vec[8], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
				solve_iLU0_tdrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
			}
#endif // USE_PRECOND

			sub_tdvector(vec[10], vec[9], vec[4]);
			add_cmul_tdvector(vec[10], vec[5], minus_alpha, vec[10]);
			ip_tdvector(mu[0], vec[10], vec[10]);
			ip_tdvector(mu[1], vec[8], vec[6]);
			ip_tdvector(mu[2], vec[10], vec[6]);
			ip_tdvector(mu[3], vec[8], vec[10]);
			ip_tdvector(mu[4], vec[8], vec[8]);
			//tau = mu[4] * mu[0] - mu[3] * mu[3];
            rtd_mul(tau, mu[4], mu[0]);
            rtd_mul(dtmp, mu[3], mu[3]);
            rtd_sub(tau, tau, dtmp);

			//zeta = (mu[0] * mu[1] - mu[2] * mu[3]) / tau;
            rtd_mul(zeta, mu[0], mu[1]);
            rtd_mul(dtmp, mu[2], mu[3]);
            rtd_sub(zeta, zeta, dtmp);
            rtd_div(zeta, zeta, tau);

			//eta = (mu[4] * mu[2] - mu[3] * mu[1]) / tau;
            rtd_mul(eta, mu[4], mu[2]);
            rtd_mul(dtmp, mu[3], mu[1]);
            rtd_sub(eta, eta, dtmp);
            rtd_div(eta, eta, tau);

			/* u = zeta * q + eta * (s + k_beta * u); */
			add_cmul_tdvector(vec[7], vec[5], beta, vec[7]);
			cmul_tdvector(vec[7], eta, vec[7]);
			add_cmul_tdvector(vec[7], vec[7], zeta, vec[4]);
		}
		/* z = zeta * ret_res + eta * z - alpha * u; */
		cmul2_tdvector(vec[11], eta);
		add_cmul_tdvector(vec[11], vec[11], zeta, vec[1]);
        rtd_neg(minus_alpha, alpha);
		add_cmul_tdvector(vec[11], vec[11], minus_alpha, vec[7]);

		/* x = x + alpha p + z */
		add_cmul_tdvector(vec[0], vec[0], alpha, vec[3]);
		add2_tdvector(vec[0], vec[11]);

		/* residual */
		/* r = t - eta * y - zeta * v; */
		rtd_neg(minus_eta, eta);
		rtd_neg(minus_zeta, zeta);
		add_cmul_tdvector(vec[1], vec[6], minus_eta, vec[10]);
		add_cmul_tdvector(vec[1], vec[1], minus_zeta, vec[8]);

		//ip_tdvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		//rtd_sqrt(dtmp, beta_num);
        norm2_tdvector(dtmp, vec[1]);
#ifdef USE_PRECOND
		if(norm2_res_history != NULL) norm2_res_history[times + 1] = dtmp[0];
#endif// USE_PRECOND

		// dtmp1 = aeps + reps * init_resnorm)
        rtd_mul(dtmp1, reps, init_resnorm);
        rtd_add(dtmp1, dtmp1, aeps);
		//if(dtmp <= aeps + reps * init_resnorm)
        if(rtd_cmp(dtmp, dtmp1) <= 0)
		{
			subst_tdvector(answer, vec[0]);
			return_val = times; // Fix!
			break;
		}
//		printf("DGPBiCG: %5d %10.3e\n", times, dtmp / init_resnorm);
		//if(zeta == 0.0)
        if(rtd_cmp_ui(zeta, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Zeta is zero!(TDGPBiCG, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}

		rtd_set(old_rho, rho);;
	}

	/* Not converge */
	subst_tdvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 12; i++)
		free_tdvector(vec[i]);

#ifdef USE_PRECOND
	free_tdvector(dvec[0]);
	free_tdvector(dvec[1]);
#endif // USE_PRECOND

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(TDGPBiCG, %ld)\n", times);
		return_val = -5;
	}

	return return_val;

}

#endif // USE_TDLINEAR
