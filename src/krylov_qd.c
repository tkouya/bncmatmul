/********************************************************************************/
/* krylov_qd.c:                                                                 */
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

#ifdef USE_QDLINEAR

/************************************************************/
/*                                                          */
/* Bi-Conjugate-Gradient Method for Real Matrix             */
/*                                     (QD Precision)       */
/*                                                          */
/*                 ver. 0.0 2024-04-23 (Tue) Tomonori Kouya */
/*                 ver. 0.1 2024-10-21 (Mon) Tomonori Kouya */
/*                 ver. 0.2 2025-02-20 (Thu) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifdef USE_PRECOND
		long int _bncomp_QDBiCG_sp_iLU0(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDRSMatrix ilu, double *norm2_res_history)
		#else // USE_PRECOND
		long int _bncomp_QDBiCG_sp(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifdef USE_PRECOND
		long int QDBiCG_sp_iLU0(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDRSMatrix ilu, double *norm2_res_history)
		#else // USE_PRECOND
		long int QDBiCG_sp(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		#ifdef USE_PRECOND
		long int _bncomp_QDBiCG_sp_d_iLU0(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history)
		#else // USE_PRECOND
		long int _bncomp_QDBiCG_sp_d(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifdef USE_PRECOND
		long int QDBiCG_sp_d_iLU0(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history)
		#else // USE_PRECOND
		long int QDBiCG_sp_d(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifdef USE_PRECOND
		long int _bncomp_QDBiCG_iLU0(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDMatrix ilu, double *norm2_res_history)
		#else // USE_PRECOND
		long int _bncomp_QDBiCG(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifdef USE_PRECOND
		long int QDBiCG_iLU0(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDMatrix ilu, double *norm2_res_history)
		#else // USE_PRECOND
		long int QDBiCG(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       QDVector answer: Solution for Ax = b                */
/*       QDMatrix a: Coefficient matrix A                    */
/*                                       (given by user)    */
/*       QDVector b: Constant vector b   (given by user)     */
/*       double reps[QDSIZE]: Relative tolerance (given by user)    */
/*       double aeps[QDSIZE]: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       QDVector answer: Solution for Ax = b                */
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
	double alpha[QDSIZE], alpha_num[QDSIZE], alpha_den[QDSIZE], minus_alpha[QDSIZE];
	double beta[QDSIZE], beta_num[QDSIZE];
	double rho[QDSIZE], old_rho[QDSIZE];
	double dtmp[QDSIZE], dtmp1[QDSIZE], init_resnorm[QDSIZE];
	QDVector vec[9]; /* Temporary Vectors */
	QDVector dvec[2]; // for Preconditioning

	dim = answer->dim;

/* Set initial value */
	for(i = 0; i < 9; i++)
		vec[i] = init_qdvector(dim);

#ifdef USE_PRECOND
	dvec[0] = init_qdvector(dim);
	dvec[1] = init_qdvector(dim);
#endif // USE_PRECOND

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0]  == w*/
	/* vec[2] ... (b - a * vec[0])^T  == wt */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... z */
	/* vec[6] ... z^T */

	subst_qdvector(vec[1], b); 
	subst_qdvector(vec[2], b);
	subst_qdvector(vec[7], vec[1]);
	subst_qdvector(vec[8], vec[2]);
#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_qdvector(dvec[0], vec[1]);
		subst_qdvector(dvec[1], vec[2]);
#ifdef USE_SPARSE_D_VERSION
		solve_iLU0_drsmatrix_qdvec(vec[7], ilu, dvec[0]);
		solve_iLU0t_drsmatrix_qdvec(vec[8], ilu, dvec[1]);
#else // USE_SPARSE_D_VERSION
		solve_iLU0_qdrsmatrix(vec[7], ilu, dvec[0]);
		solve_iLU0t_qdrsmatrix(vec[8], ilu, dvec[1]);
#endif // USE_SPARSE_D_VERSION
	}
#endif // USE_PRECOND

	//ip_qdvector(beta_num, vec[1], vec[1]);
    //ip_qdvector(beta_num, vec[1], vec[1]);
	//init_resnorm = sqrt(beta_num);
    //rqd_sqrt(init_resnorm, beta_num);
    norm2_qdvector(init_resnorm, vec[1]);
	#ifdef USE_PRECOND
	if(norm2_res_history != NULL) norm2_res_history[0] = init_resnorm[0];
	#endif // USE_PRECOND

	//old_rho = 0.0;
	//rho = 0.0;
    rqd_set0(old_rho);
    rqd_set0(rho);
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		//ip_qdvector(rho, vec[2], vec[1]);
        ip_qdvector(rho, vec[2], vec[7]); // Fix! 2025-03-06(Thu) T.Kouya

		//if(rqd_cmp_ui(rho, 0UL) == 0)
        if(rqd_cmp_ui(rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(QDBiCG, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* p := w, pt := wt */
			subst_qdvector(vec[3], vec[7]);
			subst_qdvector(vec[4], vec[8]);
		}
		else
		{
			//beta = rho / old_rho;
            rqd_div(beta, rho, old_rho);

			/* p := w + beta p, pt := wt + beta * pt */
			add_cmul_qdvector(vec[3], vec[7], beta, vec[3]);
			add_cmul_qdvector(vec[4], vec[8], beta, vec[4]);
		}

		/* z := Ap, zt := A^T pt */
#ifdef USE_SPARSE_VERSION
	#ifdef  USE_OMP_VERSION
		_bncomp_mul_qdrsmatrix_qdvec(vec[5], a, vec[3]);
		_bncomp_mul_qdrsmatrixt_qdvec(vec[6], a, vec[4]);
	#else // USE_OMP_VERSION
		mul_qdrsmatrix_qdvec(vec[5], a, vec[3]);
		mul_qdrsmatrixt_qdvec(vec[6], a, vec[4]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef  USE_OMP_VERSION
		_bncomp_mul_drsmatrix_qdvec(vec[5], a, vec[3]);
		_bncomp_mul_drsmatrixt_qdvec(vec[6], a, vec[4]);
	#else // USE_OMP_VERSION
		mul_drsmatrix_qdvec(vec[5], a, vec[3]);
		mul_drsmatrixt_qdvec(vec[6], a, vec[4]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef  USE_OMP_VERSION
		_bncomp_mul_qdmatrix_qdvec(vec[5], a, vec[3]);
		_bncomp_mul_qdmatrixt_qdvec(vec[6], a, vec[4]);
	#else // USE_OMP_VERSION
		mul_qdmatrix_qdvec(vec[5], a, vec[3]);
		mul_qdmatrixt_qdvec(vec[6], a, vec[4]);
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION

		//ip_qdvector(alpha_den, vec[4], vec[5]);
        ip_qdvector(alpha_den, vec[4], vec[5]);
		//if(rqd_cmp_ui(alpha_den, 0UL) == 0)
        if(rqd_cmp_ui(alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(QDBiCG, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		//alpha = rho / alpha_den;
        rqd_div(alpha, rho, alpha_den);

		/* x = x + alpha p */
		add_cmul_qdvector(vec[0], vec[0], alpha, vec[3]);

		/* residual */
        rqd_neg(minus_alpha, alpha);
		add_cmul_qdvector(vec[1], vec[1], minus_alpha, vec[5]);
		add_cmul_qdvector(vec[2], vec[2], minus_alpha, vec[6]);

		//ip_qdvector(beta_num, vec[1], vec[1]);
        //ip_qdvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		//rqd_sqrt(dtmp, beta_num);
        //rqd_sqrt(dtmp, beta_num);
		norm2_qdvector(dtmp, vec[1]);
		#ifdef USE_PRECOND
		if(norm2_res_history != NULL) norm2_res_history[times + 1] = dtmp[0];
		#endif // USE_PRECOND
		// dtmp1 = aeps + reps * init_resnorm)
        rqd_mul(dtmp1, reps, init_resnorm);
        rqd_add(dtmp1, dtmp1, aeps);
		//if(dtmp <= aeps + reps * init_resnorm)
        if(rqd_cmp(dtmp, dtmp1) <= 0)
		{
			subst_qdvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		subst_qdvector(vec[7], vec[1]);
		subst_qdvector(vec[8], vec[2]);
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_qdvector(dvec[0], vec[7]);
			subst_qdvector(dvec[1], vec[8]);
#ifdef USE_SPARSE_D_VERSION
			solve_iLU0_drsmatrix_qdvec(vec[7], ilu, dvec[0]);
			solve_iLU0t_drsmatrix_qdvec(vec[8], ilu, dvec[1]);
#else // USE_SPARSE_D_VERSION
			solve_iLU0_qdrsmatrix(vec[7], ilu, dvec[0]);
			solve_iLU0t_qdrsmatrix(vec[8], ilu, dvec[1]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

		//rqd_set(old_rho, rho);;
        rqd_set(old_rho, rho);
	}

	/* Not converge */
	subst_qdvector(answer, vec[0]);

	/* free vec[0]..[9]; */
	for(i = 0; i < 9; i++)
		free_qdvector(vec[i]);

#ifdef USE_PRECOND
	free_qdvector(dvec[0]);
	free_qdvector(dvec[1]);
#endif // USE_PRECOND

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(QDBiCG, %ld)\n", times);
		return_val = -5; // Fix!
	}

	return return_val; // Fix!

}

/************************************************************/
/*                                                          */
/*               CGS Method for Real Matrix                 */
/*                                     (QD Precision)       */
/*                                                          */
/*                 ver. 0.0 2004.11.06 (Sat) Tomonori Kouya */
/*                 ver. 0.1 2012-03-17 (Sat) Tomonori Kouya */
/*                 ver. 0.2 2024-10-21 (Mon) Tomonori Kouya */
/*                 ver. 0.3 2025-02-20 (Thu) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_QDCGS_sp(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_QDCGS_sp_iLU0(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int QDCGS_sp(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int QDCGS_sp_iLU0(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_QDCGS_sp_d(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_QDCGS_sp_d_iLU0(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int QDCGS_sp_d(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int QDCGS_sp_d_iLU0(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_QDCGS(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_QDCGS_iLU0(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int QDCGS(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int QDCGS_iLU0(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       QDVector answer: Solution for Ax = b                */
/*       QDMatrix a: Coefficient matrix A                    */
/*                                       (given by user)    */
/*       QDVector b: Constant vector b   (given by user)     */
/*       double reps[QDSIZE]: Relative tolerance (given by user)    */
/*       double aeps[QDSIZE]: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       QDVector answer: Solution for Ax = b                */
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
	double alpha[QDSIZE], minus_alpha[QDSIZE], alpha_num[QDSIZE], alpha_den[QDSIZE];
	double beta[QDSIZE], beta_num[QDSIZE];
	double rho[QDSIZE], old_rho[QDSIZE];
	double dtmp[QDSIZE], dtmp1[QDSIZE], init_resnorm[QDSIZE];
	QDVector vec[9]; /* Temporary Vectors */
	QDVector dvec[2]; // for preconditioning

	dim = answer->dim;

/* Set initial value */
	for(i = 0; i < 9; i++)
		vec[i] = init_qdvector(dim);

#ifdef USE_PRECOND
	dvec[0] = init_qdvector(dim);
	dvec[1] = init_qdvector(dim);
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

	subst_qdvector(vec[1], b); 
	#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_qdvector(dvec[0], vec[1]);
#ifdef USE_SPARSE_D_VERSION
		solve_iLU0_drsmatrix_qdvec(vec[1], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
		solve_iLU0_qdrsmatrix(vec[1], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
	}
#endif // USE_PRECOND
	//subst_qdvector(vec[2], b);
	subst_qdvector(vec[2], vec[1]);

	//ip_qdvector(beta_num, vec[1], vec[1]);
	//rqd_sqrt(init_resnorm, beta_num);
    norm2_qdvector(init_resnorm, vec[1]);
#ifdef USE_PRECOND
	if(norm2_res_history != NULL) norm2_res_history[0] = init_resnorm[0];
#endif // USE_PRECOND

	rqd_set0(old_rho); // = 0.0;
	rqd_set0(rho); // = 0.0;
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_qdvector(rho, vec[2], vec[1]);

		if(rqd_cmp_ui(rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(QDCGS, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* u := rt, p := u */
			subst_qdvector(vec[5], vec[1]);
			subst_qdvector(vec[3], vec[5]);
		}
		else
		{
			//beta = rho / old_rho;
            rqd_div(beta, rho, old_rho);

			/* u := r + beta q, p := u + beta * (q + beta p) */
			add_cmul_qdvector(vec[5], vec[1], beta, vec[7]);
			add_cmul_qdvector(vec[8], vec[7], beta, vec[3]);
			add_cmul_qdvector(vec[3], vec[5], beta, vec[8]);
		}
		
		/* vt := Apt */
#ifdef USE_SPARSE_VERSION
	#ifdef  USE_OMP_VERSION
		_bncomp_mul_qdrsmatrix_qdvec(vec[8], a, vec[3]);
	#else // USE_OMP_VERSION
		mul_qdrsmatrix_qdvec(vec[8], a, vec[3]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef  USE_OMP_VERSION
		_bncomp_mul_drsmatrix_qdvec(vec[8], a, vec[3]);
	#else // USE_OMP_VERSION
		mul_drsmatrix_qdvec(vec[8], a, vec[3]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef  USE_OMP_VERSION
		_bncomp_mul_qdmatrix_qdvec(vec[8], a, vec[3]);
	#else // USE_OMP_VERSION
		mul_qdmatrix_qdvec(vec[8], a, vec[3]);
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION

		/* precondition */
		#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_qdvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
			solve_iLU0_drsmatrix_qdvec(vec[8], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
			solve_iLU0_qdrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

		ip_qdvector(alpha_den, vec[2], vec[8]);
		if(rqd_cmp_ui(alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(QDCGS, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		//alpha = rho / alpha_den;
        rqd_div(alpha, rho, alpha_den);

		/* q = u - alpha  vt */
        rqd_neg(minus_alpha, alpha);
		add_cmul_qdvector(vec[7], vec[5], minus_alpha, vec[8]);

		/* ut = u + q */
		add_qdvector(vec[6], vec[5], vec[7]);

		/* x = x + alpha ut */
		add_cmul_qdvector(vec[0], vec[0], alpha, vec[6]);

		/* residual */
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_qdrsmatrix_qdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
		mul_qdrsmatrix_qdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_drsmatrix_qdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
		mul_drsmatrix_qdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef  USE_OMP_VERSION
		_bncomp_mul_qdmatrix_qdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
		mul_qdmatrix_qdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION

		/* precondition */
		#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_qdvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
			solve_iLU0_drsmatrix_qdvec(vec[8], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
			solve_iLU0_qdrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

		add_cmul_qdvector(vec[1], vec[1], minus_alpha, vec[8]);

		//ip_qdvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		//rqd_sqrt(dtmp, beta_num);
        norm2_qdvector(dtmp, vec[1]);
		#ifdef USE_PRECOND
		if(norm2_res_history != NULL) norm2_res_history[times + 1] = dtmp[0];
		#endif // USE_PRECOND
		// dtmp1 = aeps + reps * init_resnorm)
        rqd_mul(dtmp1, reps, init_resnorm);
        rqd_add(dtmp1, dtmp1, aeps);
		//if(dtmp <= aeps + reps * init_resnorm)
        if(rqd_cmp(dtmp, dtmp1) <= 0)
		{
			subst_qdvector(answer, vec[0]);
			return_val = times; // Fix!
			break;
		}

		rqd_set(old_rho, rho);
	}

	/* Not converge */
	subst_qdvector(answer, vec[0]);

	/* free vec[0]..[8]; */
	for(i = 0; i < 9; i++)
		free_qdvector(vec[i]);

#ifdef USE_PRECOND
	free_qdvector(dvec[0]);
	free_qdvector(dvec[1]);
#endif // USE_PRECOND

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(QDCGS, %ld)\n", times);
		return_val= -5;
	}

	return return_val;

}

/************************************************************/
/*                                                          */
/*             BiCGSTAB Method for Real Matrix              */
/*                                     (QD Precision)       */
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
		long int _bncomp_QDBiCGSTAB_sp(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_QDBiCGSTAB_sp_iLU0(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int QDBiCGSTAB_sp(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int QDBiCGSTAB_sp_iLU0(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_QDBiCGSTAB_sp_d(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_QDBiCGSTAB_sp_d_iLU0(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int QDBiCGSTAB_sp_d(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int QDBiCGSTAB_sp_d_iLU0(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_QDBiCGSTAB(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_QDBiCGSTAB_iLU0(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int QDBiCGSTAB(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int QDBiCGSTAB_iLU0(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       QDVector answer: Solution for Ax = b                */
/*       QDMatrix a: Coefficient matrix A                    */
/*                                       (given by user)    */
/*       QDVector b: Constant vector b   (given by user)     */
/*       double reps[QDSIZE]: Relative tolerance (given by user)    */
/*       double aeps[QDSIZE]: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       QDVector answer: Solution for Ax = b                */
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
	double alpha[QDSIZE], minus_alpha[QDSIZE], alpha_num[QDSIZE], alpha_den[QDSIZE];
	double beta[QDSIZE], beta_num[QDSIZE];
	double rho[QDSIZE], old_rho[QDSIZE];
	double omega[QDSIZE], minus_omega[QDSIZE], omega_den[QDSIZE];
	double dtmp[QDSIZE], dtmp1[QDSIZE], init_resnorm[QDSIZE];
	QDVector vec[9]; /* Temporary Vectors */
	QDVector dvec[2]; // for Preconditioning

	dim = answer->dim;

/* Set initial value */
	for(i = 0; i < 9; i++)
		vec[i] = init_qdvector(dim);

#ifdef USE_PRECOND
	dvec[0] = init_qdvector(dim);
	dvec[1] = init_qdvector(dim);
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

	subst_qdvector(vec[1], b); 
	#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_qdvector(dvec[0], vec[1]);
#ifdef USE_SPARSE_D_VERSION
		solve_iLU0_drsmatrix_qdvec(vec[1], ilu, dvec[0]);
#else // USE_SPRARSE_D_VERSION
		solve_iLU0_qdrsmatrix(vec[1], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
	}
#endif // USE_PRECOND
	//subst_qdvector(vec[2], b);
	subst_qdvector(vec[2], vec[1]);

	//ip_qdvector(beta_num, vec[1], vec[1]);
	//rqd_sqrt(init_resnorm, beta_num);
    norm2_qdvector(init_resnorm, vec[1]);
	#ifdef USE_PRECOND
	if(norm2_res_history != NULL) norm2_res_history[0] = init_resnorm[0];
	#endif // USE_PRECOND

	rqd_set0(old_rho); // = 0.0;
	rqd_set0(rho); // = 0.0;
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_qdvector(rho, vec[2], vec[1]);

		if(rqd_cmp_ui(rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(QDBiCG, %ld)\n", times);
			return_val= -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* p := r */
			subst_qdvector(vec[3], vec[1]);
		}
		else
		{
			//beta = (rho / old_rho) * (alpha / omega);
            rqd_div(dtmp, rho, old_rho);
            rqd_div(dtmp1, alpha, omega);
            rqd_mul(beta, dtmp, dtmp1);

			/* p := r + beta (p - omega v) */
			rqd_neg(minus_omega, omega);
			add_cmul_qdvector(vec[4], vec[3], minus_omega, vec[5]);
			add_cmul_qdvector(vec[3], vec[1], beta, vec[4]);
		}
		/* precondition */
		
		/* v := Apt */
#ifdef USE_SPARSE_VERSION
	#ifdef  USE_OMP_VERSION
		_bncomp_mul_qdrsmatrix_qdvec(vec[5], a, vec[3]);
	#else // USE_OMP_VERSION
		mul_qdrsmatrix_qdvec(vec[5], a, vec[3]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef  USE_OMP_VERSION
		_bncomp_mul_drsmatrix_qdvec(vec[5], a, vec[3]);
	#else // USE_OMP_VERSION
		mul_drsmatrix_qdvec(vec[5], a, vec[3]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef  USE_OMP_VERSION
		_bncomp_mul_qdmatrix_qdvec(vec[5], a, vec[3]);
	#else // USE_OMP_VERSION
		mul_qdmatrix_qdvec(vec[5], a, vec[3]);
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION

		/* precondition */
		// v := M^(-1)Apt -> Solve Mv = Apt for v
		#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_qdvector(dvec[0], vec[5]);
#ifdef USE_SPARSE_D_VERSION
			solve_iLU0_drsmatrix_qdvec(vec[5], ilu, dvec[0]);
#else // USE_SPRARSE_D_VERSION
			solve_iLU0_qdrsmatrix(vec[5], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

		ip_qdvector(alpha_den, vec[2], vec[5]);
		if(rqd_cmp_ui(alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(QDBiCGSTAB, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		//alpha = rho / alpha_den;
        rqd_div(alpha, rho, alpha_den);

		/* s = r - alpha v */
        rqd_neg(minus_alpha, alpha);
		add_cmul_qdvector(vec[6], vec[1], minus_alpha, vec[5]);

		/* Stopping Criteria */
		norm2_qdvector(dtmp, vec[6]);
		// aeps + reps * init_resnorm
		rqd_mul(dtmp1, reps, init_resnorm);
		rqd_add(dtmp1, dtmp1, aeps);
		if(rqd_cmp(dtmp, dtmp1) <= 0) // aeps + reps * init_resnorm)
		{
			/* x = x + alpha pt */
			add_cmul_qdvector(vec[0], vec[0], alpha, vec[3]);

			subst_qdvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		/* precondition */
#ifdef USE_SPARSE_VERSION
	#ifdef  USE_OMP_VERSION
		_bncomp_mul_qdrsmatrix_qdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
		mul_qdrsmatrix_qdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef  USE_OMP_VERSION
		_bncomp_mul_drsmatrix_qdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
		mul_drsmatrix_qdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef  USE_OMP_VERSION
		_bncomp_mul_qdmatrix_qdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
		mul_qdmatrix_qdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION

		/* precondition */
		// v := M^(-1)Apt -> Solve Mv = Apt for v
		#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_qdvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
			solve_iLU0_drsmatrix_qdvec(vec[8], ilu, dvec[0]);
#else // USE_SPRARSE_D_VERSION
			solve_iLU0_qdrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

		/* omega = (t, s) / (t, t) */
		ip_qdvector(omega_den, vec[8], vec[8]);
		if(rqd_cmp_ui(omega_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Omega is zero!(QDBiCGSTAB, %ld)\n", times);
			return_val = -3; // Fix!
			break;
		}
		ip_qdvector(omega, vec[8], vec[6]);
		if(rqd_cmp_ui(omega, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Omega is zero!(QDBiCGSTAB, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}
		//omega = omega / omega_den;
        rqd_div(omega, omega, omega_den);

		/* x = x + alpha pt + omega st */
		add_cmul_qdvector(vec[4], vec[0], alpha, vec[3]);
		add_cmul_qdvector(vec[0], vec[4], omega, vec[6]);

		/* residual */
        rqd_neg(dtmp, omega);
		add_cmul_qdvector(vec[1], vec[6], dtmp, vec[8]);

		//ip_qdvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		//rqd_sqrt(dtmp, beta_num);
        norm2_qdvector(dtmp, vec[1]);
		#ifdef USE_PRECOND
		if(norm2_res_history != NULL) norm2_res_history[times + 1] = dtmp[0];
		#endif // USE_PRECOND
		// dtmp1 = aeps + reps * init_resnorm)
        rqd_mul(dtmp1, reps, init_resnorm);
        rqd_add(dtmp1, dtmp1, aeps);
		//if(dtmp <= aeps + reps * init_resnorm)
        if(rqd_cmp(dtmp, dtmp1) <= 0)
		{
			subst_qdvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		rqd_set(old_rho, rho);
	}

	/* Not converge */
	subst_qdvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 9; i++)
		free_qdvector(vec[i]);

#ifdef USE_PRECOND
	free_qdvector(dvec[0]);
	free_qdvector(dvec[1]);
#endif // USE_PRECOND
	
	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(QDBiCGSTAB, %ld)\n", times);
		return_val = -5;
	}

	return return_val; // Fix!

}

/************************************************************/
/*                                                          */
/* GPBiCG(Generalized Product Bi-CG) Method for Real Matrix */
/*                                     (QD Precision)       */
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
		long int _bncomp_QDGPBiCG_sp(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_QDGPBiCG_sp_iLU0(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int QDGPBiCG_sp(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int QDGPBiCG_sp_iLU0(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_QDGPBiCG_sp_d(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_QDGPBiCG_sp_d_iLU0(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int QDGPBiCG_sp_d(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int QDGPBiCG_sp_d_iLU0(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, DRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_QDGPBiCG(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_QDGPBiCG_iLU0(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int QDGPBiCG(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int QDGPBiCG_iLU0(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes, QDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       QDVector answer: Solution for Ax = b                */
/*       QDMatrix a: Coefficient matrix A                    */
/*                                       (given by user)    */
/*       QDVector b: Constant vector b   (given by user)     */
/*       double reps[QDSIZE]: Relative tolerance (given by user)    */
/*       double aeps[QDSIZE]: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       QDVector answer: Solution for Ax = b                */
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
	double alpha[QDSIZE], minus_alpha[QDSIZE], alpha_num[QDSIZE], alpha_den[QDSIZE];
	double beta[QDSIZE], beta_num[QDSIZE];
	double rho[QDSIZE], old_rho[QDSIZE];
	double mu[5][QDSIZE], tau[QDSIZE], zeta[QDSIZE], minus_zeta[QDSIZE], eta[QDSIZE], minus_eta[QDSIZE];
	double dtmp[QDSIZE], dtmp1[QDSIZE], init_resnorm[QDSIZE];
	QDVector vec[12]; /* Temporary Vectors */
	QDVector dvec[2]; // Preconditioning

	dim = answer->dim;

/* Set initial value */
	for(i = 0; i < 12; i++)
		vec[i] = init_qdvector(dim);

#ifdef USE_PRECOND
	dvec[0] = init_qdvector(dim);
	dvec[1] = init_qdvector(dim);
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

	subst_qdvector(vec[1], b); 
	#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_qdvector(dvec[0], vec[1]);
#ifdef USE_SPARSE_D_VERSION
		solve_iLU0_drsmatrix_qdvec(vec[1], ilu, dvec[0]);
#else // USE_SPRARSE_D_VERSION
		solve_iLU0_qdrsmatrix(vec[1], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
	}
#endif // USE_PRECOND
	//subst_qdvector(vec[2], b);
	subst_qdvector(vec[2], vec[1]);
	
	//ip_qdvector(beta_num, vec[1], vec[1]);
	//rqd_sqrt(init_resnorm, beta_num);
    norm2_qdvector(init_resnorm, vec[1]);
#ifdef USE_PRECOND
	if(norm2_res_history != NULL) norm2_res_history[0] = init_resnorm[0];
#endif// USE_PRECOND

	rqd_set0(old_rho); // = 0.0;
	rqd_set0(rho); // = 0.0;
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_qdvector(rho, vec[2], vec[1]);

		if(rqd_cmp_ui(rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(QDGPBiCG, %ld)\n", times);
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
			subst_qdvector(vec[3], vec[1]);
#ifdef USE_SPARSE_VERSION
	#ifdef  USE_OMP_VERSION
			_bncomp_mul_qdrsmatrix_qdvec(vec[4], a, vec[3]);
	#else // USE_OMP_VERSION
			mul_qdrsmatrix_qdvec(vec[4], a, vec[3]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef  USE_OMP_VERSION
			_bncomp_mul_drsmatrix_qdvec(vec[4], a, vec[3]);
	#else // USE_OMP_VERSION
			mul_drsmatrix_qdvec(vec[4], a, vec[3]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef  USE_OMP_VERSION
			_bncomp_mul_qdmatrix_qdvec(vec[4], a, vec[3]);
	#else // USE_OMP_VERSION
			mul_qdmatrix_qdvec(vec[4], a, vec[3]);
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION

#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_qdvector(dvec[0], vec[4]);
#ifdef USE_SPARSE_D_VERSION
				solve_iLU0_drsmatrix_qdvec(vec[4], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
				solve_iLU0_qdrsmatrix(vec[4], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
			}
#endif // USE_PRECOND

			ip_qdvector(alpha_den, vec[2], vec[4]);
			if(rqd_cmp_ui(alpha_den, 0UL) == 0)
			{
				fprintf(stderr, "Denominator of Alpha is zero!(QDGPBiCG, %ld)\n", times);
				return_val = -2; // Fix!
				break; // Fix!
			}
			//alpha = rho / alpha_den;
            rqd_div(alpha, rho, alpha_den);
            rqd_neg(minus_alpha, alpha);
			add_cmul_qdvector(vec[6], vec[1], minus_alpha, vec[4]);
#ifdef USE_SPARSE_VERSION
	#ifdef  USE_OMP_VERSION
			_bncomp_mul_qdrsmatrix_qdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
			mul_qdrsmatrix_qdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef  USE_OMP_VERSION
			_bncomp_mul_drsmatrix_qdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
			mul_drsmatrix_qdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef  USE_OMP_VERSION
			_bncomp_mul_qdmatrix_qdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
			mul_qdmatrix_qdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION

#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_qdvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
				solve_iLU0_drsmatrix_qdvec(vec[8], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
				solve_iLU0_qdrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
			}
#endif // USE_PRECOND

			cmul_qdvector(vec[10], alpha, vec[4]);
			sub_qdvector(vec[10], vec[10], vec[1]);
			ip_qdvector(mu[1], vec[8], vec[6]);
			ip_qdvector(mu[4], vec[8], vec[8]);
			//if(mu[4] == 0.0)
            if(rqd_cmp_ui(mu[4], 0UL) == 0)
			{
				fprintf(stderr, "Denominator of mu[4] is zero!(QDGPBiCG, %ld)\n", times);
				return_val = -3; // Fix!
				break; // Fix!
			}
			//zeta = mu[1] / mu[4];
            rqd_div(zeta, mu[1], mu[4]);
			//eta = 0.0;
            rqd_set0(eta);
			cmul_qdvector(vec[7], zeta, vec[4]);
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
            rqd_div(dtmp, rho, old_rho);
            rqd_div(dtmp1, alpha, zeta);
            rqd_mul(beta, dtmp, dtmp1);

			add_cmul_qdvector(vec[9], vec[8], beta, vec[4]);
			sub_qdvector(vec[3], vec[3], vec[7]);
			add_cmul_qdvector(vec[3], vec[1], beta, vec[3]);

#ifdef USE_SPARSE_VERSION
	#ifdef  USE_OMP_VERSION
			_bncomp_mul_qdrsmatrix_qdvec(vec[4], a, vec[3]);
	#else // USE_OMP_VERSION
			mul_qdrsmatrix_qdvec(vec[4], a, vec[3]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef  USE_OMP_VERSION
			_bncomp_mul_drsmatrix_qdvec(vec[4], a, vec[3]);
	#else // USE_OMP_VERSION
			mul_drsmatrix_qdvec(vec[4], a, vec[3]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef  USE_OMP_VERSION
			_bncomp_mul_qdmatrix_qdvec(vec[4], a, vec[3]);
	#else // USE_OMP_VERSION
			mul_qdmatrix_qdvec(vec[4], a, vec[3]);
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION

#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_qdvector(dvec[0], vec[4]);
#ifdef USE_SPARSE_D_VERSION
				solve_iLU0_drsmatrix_qdvec(vec[4], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
				solve_iLU0_qdrsmatrix(vec[4], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
			}
#endif // USE_PRECOND

			ip_qdvector(alpha_den, vec[2], vec[4]);
			if(rqd_cmp_ui(alpha_den, 0UL) == 0)
			{
				fprintf(stderr, "Denominator of Alpha is zero!(QDGPBiCG, %ld)\n", times);
				return_val = -2; // Fix!
				break; // Fix!
			}
			//alpha = rho / alpha_den;
            rqd_div(alpha, rho, alpha_den);
			sub_qdvector(vec[5], vec[6], vec[1]);
            rqd_neg(minus_alpha, alpha);
			add_cmul_qdvector(vec[6], vec[1], minus_alpha, vec[4]);
#ifdef USE_SPARSE_VERSION
	#ifdef  USE_OMP_VERSION
			_bncomp_mul_qdrsmatrix_qdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
			mul_qdrsmatrix_qdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef  USE_OMP_VERSION
			_bncomp_mul_drsmatrix_qdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
			mul_drsmatrix_qdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef  USE_OMP_VERSION
			_bncomp_mul_qdmatrix_qdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
			mul_qdmatrix_qdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION

#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_qdvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
				solve_iLU0_drsmatrix_qdvec(vec[8], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
				solve_iLU0_qdrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
			}
#endif // USE_PRECOND

			sub_qdvector(vec[10], vec[9], vec[4]);
			add_cmul_qdvector(vec[10], vec[5], minus_alpha, vec[10]);
			ip_qdvector(mu[0], vec[10], vec[10]);
			ip_qdvector(mu[1], vec[8], vec[6]);
			ip_qdvector(mu[2], vec[10], vec[6]);
			ip_qdvector(mu[3], vec[8], vec[10]);
			ip_qdvector(mu[4], vec[8], vec[8]);
			//tau = mu[4] * mu[0] - mu[3] * mu[3];
            rqd_mul(tau, mu[4], mu[0]);
            rqd_mul(dtmp, mu[3], mu[3]);
            rqd_sub(tau, tau, dtmp);

			//zeta = (mu[0] * mu[1] - mu[2] * mu[3]) / tau;
            rqd_mul(zeta, mu[0], mu[1]);
            rqd_mul(dtmp, mu[2], mu[3]);
            rqd_sub(zeta, zeta, dtmp);
            rqd_div(zeta, zeta, tau);

			//eta = (mu[4] * mu[2] - mu[3] * mu[1]) / tau;
            rqd_mul(eta, mu[4], mu[2]);
            rqd_mul(dtmp, mu[3], mu[1]);
            rqd_sub(eta, eta, dtmp);
            rqd_div(eta, eta, tau);

			/* u = zeta * q + eta * (s + k_beta * u); */
			add_cmul_qdvector(vec[7], vec[5], beta, vec[7]);
			cmul_qdvector(vec[7], eta, vec[7]);
			add_cmul_qdvector(vec[7], vec[7], zeta, vec[4]);
		}
		/* z = zeta * ret_res + eta * z - alpha * u; */
		cmul2_qdvector(vec[11], eta);
		add_cmul_qdvector(vec[11], vec[11], zeta, vec[1]);
        rqd_neg(minus_alpha, alpha);
		add_cmul_qdvector(vec[11], vec[11], minus_alpha, vec[7]);

		/* x = x + alpha p + z */
		add_cmul_qdvector(vec[0], vec[0], alpha, vec[3]);
		add2_qdvector(vec[0], vec[11]);

		/* residual */
		/* r = t - eta * y - zeta * v; */
		rqd_neg(minus_eta, eta);
		rqd_neg(minus_zeta, zeta);
		add_cmul_qdvector(vec[1], vec[6], minus_eta, vec[10]);
		add_cmul_qdvector(vec[1], vec[1], minus_zeta, vec[8]);

		//ip_qdvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		//rqd_sqrt(dtmp, beta_num);
        norm2_qdvector(dtmp, vec[1]);
#ifdef USE_PRECOND
		if(norm2_res_history != NULL) norm2_res_history[times + 1] = dtmp[0];
#endif // USE_PRECOND
		// dtmp1 = aeps + reps * init_resnorm)
        rqd_mul(dtmp1, reps, init_resnorm);
        rqd_add(dtmp1, dtmp1, aeps);
		//if(dtmp <= aeps + reps * init_resnorm)
        if(rqd_cmp(dtmp, dtmp1) <= 0)
		{
			subst_qdvector(answer, vec[0]);
			return_val = times; // Fix!
			break;
		}
//		printf("DGPBiCG: %5d %10.3e\n", times, dtmp / init_resnorm);
		//if(zeta == 0.0)
        if(rqd_cmp_ui(zeta, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Zeta is zero!(QDGPBiCG, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}

		rqd_set(old_rho, rho);;
	}

	/* Not converge */
	subst_qdvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 12; i++)
		free_qdvector(vec[i]);

#ifdef USE_PRECOND
	free_qdvector(dvec[0]);
	free_qdvector(dvec[1]);
#endif // USE_PRECOND

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(QDGPBiCG, %ld)\n", times);
		return_val = -5;
	}

	return return_val;

}

#endif // USE_QDLINEAR
