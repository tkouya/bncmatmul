/********************************************************************************/
/* krylov_ctd.c:                                                                */
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


#ifdef USE_DDLINEAR

/************************************************************/
/*                                                          */
/*                Bi-Conjugate-Gradient Method              */
/*                                 (TD Precision)           */
/*                                                          */
/*                 ver. 0.0 2024-11-25 (Sat) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifdef USE_PRECOND
		long int _bncomp_CTDBiCG_sp_iLU0(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CTDRSMatrix ilu, double *norm2_res_history)
		#else // USE_PRECOND
		long int _bncomp_CTDBiCG_sp(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifdef USE_PRECOND
		long int CTDBiCG_sp_iLU0(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CTDRSMatrix ilu, double *norm2_res_history)
		#else // USE_PRECOND
		long int CTDBiCG_sp(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		#ifdef USE_PRECOND
		long int _bncomp_CTDBiCG_sp_d_iLU0(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
		#else // USE_PRECOND
		long int _bncomp_CTDBiCG_sp_d(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifdef USE_PRECOND
		long int CTDBiCG_sp_d_iLU0(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
		#else // USE_PRECOND
		long int CTDBiCG_sp_d(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifdef USE_PRECOND
		long int _bncomp_CTDBiCG_iLU0(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CTDMatrix ilu, double *norm2_res_history)
		#else // USE_PRECOND
		long int _bncomp_CTDBiCG(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifdef USE_PRECOND
		long int CTDBiCG_iLU0(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CTDMatrix ilu, double *norm2_res_history)
		#else // USE_PRECOND
		long int CTDBiCG(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       CTDVector answer: Solution for Ax = b              */
/*       CTDMatrix a: Coefficient matrix A                  */
/*                                       (given by user)    */
/*       CTDVector b: Constant vector b   (given by user)   */
/*       double *reps: Relative tolerance (given by user)   */
/*       double *aeps: Absolute tolerance (given by user)   */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       CTDVector answer: Solution for Ax = b              */
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
	ctdfloat alpha, alpha_num, alpha_den;
	ctdfloat rho, old_rho;
	ctdfloat beta, beta_num;
	ctdfloat conj_alpha, conj_beta;
	double dtmp[TDSIZE], dtmp1[TDSIZE], init_resnorm[TDSIZE];
	CTDVector vec[9]; /* Temporary Vectors */
	CTDVector dvec[2]; // for Preconditioning

	dim = answer->re->dim;;

/* Set initial value */
	for(i = 0; i < 9; i++)
		vec[i] = init_ctdvector(dim);

#ifdef USE_PRECOND
	dvec[0] = init_ctdvector(dim);
	dvec[1] = init_ctdvector(dim);
#endif // USE_PRECOND

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0]  == w*/
	/* vec[2] ... (b - a * vec[0])^T  == wt */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... z */
	/* vec[6] ... z^T */

	subst_ctdvector(vec[1], b); 
	subst_ctdvector(vec[2], b);
	subst_ctdvector(vec[7], vec[1]);
	subst_ctdvector(vec[8], vec[2]);
#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_ctdvector(dvec[0], vec[1]);
		subst_ctdvector(dvec[1], vec[2]);
#ifdef USE_SPARSE_D_VERSION
		solve_iLU0_cdrsmatrix_ctdvec(vec[7], ilu, dvec[0]);
		solve_iLU0s_cdrsmatrix_ctdvec(vec[8], ilu, dvec[1]);
#else // USE_SPARSE_D_VERSION
		solve_iLU0_ctdrsmatrix(vec[7], ilu, dvec[0]);
		solve_iLU0s_ctdrsmatrix(vec[8], ilu, dvec[1]);
#endif // 
	}
#endif // USE_PRECOND

	//ip_ctdvector(beta_num, vec[1], vec[1]);
	//rtd_sqrt(init_resnorm, beta_num);
    norm2_ctdvector(init_resnorm, vec[1]);
	#ifdef USE_PRECOND
	if(norm2_res_history != NULL) norm2_res_history[0] = init_resnorm[0];
	#endif // USE_PRECOND
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		//ip_ctdvector(&rho, vec[2], vec[1]);
		ip_ctdvector(&rho, vec[2], vec[7]);

        rctd_abs_td(dtmp, &rho);
		//if(rctd_cmp_si(&rho, 0UL) == 0)
        if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(CTDBiCG, %ld)\n", times);
			return_val = -1; // Fix!
			break;
		}

		if(times == 0)
		{
			/* p := w, pt := wt */
			//subst_ctdvector(vec[3], vec[1]);
			//subst_ctdvector(vec[4], vec[2]);
			subst_ctdvector(vec[3], vec[7]);
			subst_ctdvector(vec[4], vec[8]);
		}
		else
		{
			rctd_div(&beta, &rho, &old_rho);

			/* p := w + beta p, pt := wt + beta * pt */
			add_cmul_ctdvector(vec[3], vec[7], &beta, vec[3]);
			//add_cmul_ctdvector(vec[4], vec[2], &beta, vec[4]);
			rctd_conj(&conj_beta, &beta);
			add_cmul_ctdvector(vec[4], vec[8], &conj_beta, vec[4]);
		}

		/* z := Ap, zt := A^T pt */
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_ctdrsmatrix_ctdvec(vec[5], a, vec[3]);
		_bncomp_mul_ctdrsmatrixs_ctdvec(vec[6], a, vec[4]);
	#else // USE_OMP_VERSION
		mul_ctdrsmatrix_ctdvec(vec[5], a, vec[3]);
		mul_ctdrsmatrixs_ctdvec(vec[6], a, vec[4]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cdrsmatrix_ctdvec(vec[5], a, vec[3]);
		_bncomp_mul_cdrsmatrixs_ctdvec(vec[6], a, vec[4]);
	#else // USE_OMP_VERSION
		mul_cdrsmatrix_ctdvec(vec[5], a, vec[3]);
		mul_cdrsmatrixs_ctdvec(vec[6], a, vec[4]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_ctdmatrix_ctdvec(vec[5], a, vec[3]);
		mul_ctdmatrixs_ctdvec(vec[6], a, vec[4]);
#endif // USE_SPARSE_VERSION

		ip_ctdvector(&alpha_den, vec[4], vec[5]);

        rctd_abs_td(dtmp, &alpha_den);
		//if(rctd_cmp_si(alpha_den, 0UL) == 0)
		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(CTDBiCG, %ld)\n", times);
			return_val = -2; // Fix!
			break;
		}
		rctd_div(&alpha, &rho, &alpha_den);

		/* x = x + alpha p */
		add_cmul_ctdvector(vec[0], vec[0], &alpha, vec[3]);

		/* residual */
		//rctd_neg(dtmp, alpha);
		sub_cmul_ctdvector(vec[1], vec[1], &alpha, vec[5]);
		//sub_cmul_ctdvector(vec[2], vec[2], &alpha, vec[6]);
		rctd_conj(&conj_alpha, &alpha);
		sub_cmul_ctdvector(vec[2], vec[2], &conj_alpha, vec[6]);

		//ip_ctdvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		//rtd_sqrt(dtmp, beta_num);
        norm2_ctdvector(dtmp, vec[1]);
		#ifdef USE_PRECOND
		if(norm2_res_history != NULL) norm2_res_history[times + 1] = dtmp[0];
		#endif // USE_PRECOND
		rtd_mul(dtmp1, reps, init_resnorm);
		rtd_add(dtmp1, dtmp1, aeps);
		if(rtd_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_ctdvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		subst_ctdvector(vec[7], vec[1]);
		subst_ctdvector(vec[8], vec[2]);
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_ctdvector(dvec[0], vec[7]);
			subst_ctdvector(dvec[1], vec[8]);
#ifdef USE_SPARSE_D_VERSION
			solve_iLU0_cdrsmatrix_ctdvec(vec[7], ilu, dvec[0]);
			//solve_iLU0t_cdrsmatrix_cddvec(vec[6], ilu, dvec[1]);
			solve_iLU0s_cdrsmatrix_ctdvec(vec[8], ilu, dvec[1]);
#else // USE_SPARSE_D_VERSION
			solve_iLU0_ctdrsmatrix(vec[7], ilu, dvec[0]);
			//solve_iLU0t_cddrsmatrix(vec[6], ilu, dvec[1]);
			solve_iLU0s_ctdrsmatrix(vec[8], ilu, dvec[1]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

		rctd_set(&old_rho, &rho);
	}

	/* Not converge */
	subst_ctdvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 9; i++)
		free_ctdvector(vec[i]);

#ifdef USE_PRECOND
	free_ctdvector(dvec[0]);
	free_ctdvector(dvec[1]);
#endif // USE_PRECOND

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(CTDBiCG, %ld)\n", times);
		return_val = -5;
	}

	return return_val; // Fix!

}
/************************************************************/
/*                                                          */
/*               CGS Method for Complex Matrix              */
/*                                 (TD Precision)           */
/*                                                          */
/*                 ver. 0.0 2024-11-25 (Sat) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CTDCGS_sp(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CTDCGS_sp_iLU0(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CTDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CTDCGS_sp(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int CTDCGS_sp_iLU0(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CTDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CTDCGS_sp_d(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CTDCGS_sp_d_iLU0(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CTDCGS_sp_d(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int CTDCGS_sp_d_iLU0(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CTDCGS(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CTDCGS_iLU0(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CTDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CTDCGS(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int CTDCGS_iLU0(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CTDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       CTDVector answer: Solution for Ax = b              */
/*       CTDMatrix a: Coefficient matrix A                  */
/*                                       (given by user)    */
/*       CTDVector b: Constant vector b  (given by user)    */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       CTDVector answer: Solution for Ax = b              */
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
	ctdfloat ctmp, ctmp1;
	ctdfloat alpha, alpha_num, alpha_den;
	ctdfloat beta, beta_num;
	ctdfloat rho, old_rho;
	double dtmp[TDSIZE], dtmp1[TDSIZE], init_resnorm[TDSIZE];
	CTDVector vec[9]; /* Temporary Vectors */
	CTDVector dvec[2]; // for preconditioning

	dim = answer->re->dim;;

/* Set initial value */
	for(i = 0; i < 9; i++)
		vec[i] = init_ctdvector(dim);

#ifdef USE_PRECOND
	dvec[0] = init_ctdvector(dim);
	dvec[1] = init_ctdvector(dim);
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

	subst_ctdvector(vec[1], b);
#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_ctdvector(dvec[0], vec[1]);
#ifdef USE_SPARSE_D_VERSION
		solve_iLU0_cdrsmatrix_ctdvec(vec[1], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
		solve_iLU0_ctdrsmatrix(vec[1], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
	}
#endif // USE_PRECOND
	//subst_ctdvector(vec[2], b);
	subst_ctdvector(vec[2], vec[1]);

	//ip_ctdvector(beta_num, vec[1], vec[1]);
	//rtd_sqrt(init_resnorm, beta_num);
    norm2_ctdvector(init_resnorm, vec[1]);
	#ifdef USE_PRECOND
	if(norm2_res_history != NULL) norm2_res_history[0] = init_resnorm[0];
	#endif // USE_PRECOND

	rctd_set_ui(&old_rho, 0UL);
	rctd_set_ui(&rho, 0UL);
	return_val = 0;

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_ctdvector(&rho, vec[2], vec[1]);

        rctd_abs_td(dtmp, &rho);
		//if(rctd_cmp_si(rho, 0UL) == 0)
		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(CTDCGS, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* u := rt, p := u */
			subst_ctdvector(vec[5], vec[1]);
			subst_ctdvector(vec[3], vec[5]);
		}
		else
		{
			rctd_div(&beta, &rho, &old_rho);

			/* u := r + beta q, p := u + beta * (q + beta p) */
			add_cmul_ctdvector(vec[5], vec[1], &beta, vec[7]);
			add_cmul_ctdvector(vec[8], vec[7], &beta, vec[3]);
			add_cmul_ctdvector(vec[3], vec[5], &beta, vec[8]);
		}
		
		/* vt := Apt */
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_ctdrsmatrix_ctdvec(vec[8], a, vec[3]);
	#else // USE_OMP_VERSION
		mul_ctdrsmatrix_ctdvec(vec[8], a, vec[3]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cdrsmatrix_ctdvec(vec[8], a, vec[3]);
	#else // USE_OMP_VERSION
		mul_cdrsmatrix_ctdvec(vec[8], a, vec[3]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_ctdmatrix_ctdvec(vec[8], a, vec[3]);
#endif // USE_SPARSE_VERSION

		/* precondition */
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_ctdvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
			solve_iLU0_cdrsmatrix_ctdvec(vec[8], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
			solve_iLU0_ctdrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

		ip_ctdvector(&alpha_den, vec[2], vec[8]);
        rctd_abs_td(dtmp, &alpha_den);
		//if(rctd_cmp_si(alpha_den, 0UL) == 0)
		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(CTDCGS, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		rctd_div(&alpha, &rho, &alpha_den);

		/* q = u - alpha  vt */
		//rtd_neg(dtmp, alpha);
		sub_cmul_ctdvector(vec[7], vec[5], &alpha, vec[8]);

		/* ut = u + q */
		add_ctdvector(vec[6], vec[5], vec[7]);

		/* x = x + alpha ut */
		add_cmul_ctdvector(vec[0], vec[0], &alpha, vec[6]);

		/* residual */
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_ctdrsmatrix_ctdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
		mul_ctdrsmatrix_ctdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cdrsmatrix_ctdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
		mul_cdrsmatrix_ctdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_ctdmatrix_ctdvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION

		/* precondition */
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_ctdvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
			solve_iLU0_cdrsmatrix_ctdvec(vec[8], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
			solve_iLU0_ctdrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

		sub_cmul_ctdvector(vec[1], vec[1], &alpha, vec[8]);

		//ip_ctdvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		//rtd_sqrt(dtmp, beta_num);
        norm2_ctdvector(dtmp, vec[1]);
		#ifdef USE_PRECOND
		if(norm2_res_history != NULL) norm2_res_history[times + 1] = dtmp[0];
		#endif // USE_PRECOND
		rtd_mul(dtmp1, reps, init_resnorm);
		rtd_add(dtmp1, dtmp1, aeps);
		if(rtd_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_ctdvector(answer, vec[0]);
			return_val = times;
			break;
		}

		rctd_set(&old_rho, &rho);
	}

	/* Not converge */
	subst_ctdvector(answer, vec[0]);

	/* free vec[0]..[8]; */
	for(i = 0; i < 9; i++)
		free_ctdvector(vec[i]);

#ifdef USE_PRECOND
	free_ctdvector(dvec[0]);
	free_ctdvector(dvec[1]);
#endif // USE_PRECOND

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(CTDCGS, %ld)\n", times);
		return_val = -5;
	}

	// Fix!
	return return_val;
}

/************************************************************/
/*                                                          */
/*             BiCGSTAB Method for Complex Matrix           */
/*                                 (TD Precision)           */
/*                                                          */
/*                 ver. 0.0 2024-11-25 (Sat) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CTDBiCGSTAB_sp(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CTDBiCGSTAB_sp_iLU0(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CTDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CTDBiCGSTAB_sp(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int CTDBiCGSTAB_sp_iLU0(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CTDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CTDBiCGSTAB_sp_d(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CTDBiCGSTAB_sp_d_iLU0(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CTDBiCGSTAB_sp_d(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int CTDBiCGSTAB_sp_d_iLU0(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CTDBiCGSTAB(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CTDBiCGSTAB_iLU0(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CTDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CTDBiCGSTAB(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int CTDBiCGSTAB_iLU0(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CTDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       CTDVector answer: Solution for Ax = b              */
/*       CTDMatrix a: Coefficient matrix A                  */
/*                                       (given by user)    */
/*       CTDVector b: Constant vector b   (given by user)   */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       CTDVector answer: Solution for Ax = b              */
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
	double dtmp[TDSIZE], dtmp1[TDSIZE];
    ctdfloat ctmp;
	ctdfloat alpha, alpha_num, alpha_den;
	ctdfloat beta, beta_num;
	ctdfloat rho, old_rho;
	ctdfloat omega, omega_den;
	double init_resnorm[TDSIZE];
	CTDVector vec[9]; /* Temporary Vectors */
	CTDVector dvec[2]; // for Preconditioning

	dim = answer->re->dim;;

/* Set initial value */
	for(i = 0; i < 9; i++)
		vec[i] = init_ctdvector(dim);

#ifdef USE_PRECOND
	dvec[0] = init_ctdvector(dim);
	dvec[1] = init_ctdvector(dim);
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

	subst_ctdvector(vec[1], b);
#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_ctdvector(dvec[0], vec[1]);
#ifdef USE_SPARSE_D_VERSION
		solve_iLU0_cdrsmatrix_ctdvec(vec[1], ilu, dvec[0]);
#else // USE_SPRARSE_D_VERSION
		solve_iLU0_ctdrsmatrix(vec[1], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
	}
#endif // USE_PRECOND
	//subst_ctdvector(vec[2], b);
	subst_ctdvector(vec[2], vec[1]);

	//ip_ctdvector(beta_num, vec[1], vec[1]);
	//rtd_sqrt(init_resnorm, beta_num);
    norm2_ctdvector(init_resnorm, vec[1]);
	#ifdef USE_PRECOND
	if(norm2_res_history != NULL) norm2_res_history[0] = init_resnorm[0];
	#endif // USE_PRECOND

	rctd_set_ui(&old_rho, 0UL);
	rctd_set_ui(&rho, 0UL);
	return_val = 0;

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_ctdvector(&rho, vec[2], vec[1]);

        rctd_abs_td(dtmp, &rho);
		//if(rctd_cmp_si(rho, 0UL) == 0)
		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(CTDBiCGSTAB, %ld)\n", times);
			return_val = -1; // Fix!
			break;
		}

		if(times == 0)
		{
			/* p := r */
			subst_ctdvector(vec[3], vec[1]);
		}
		else
		{
			rctd_div(&beta, &rho, &old_rho);
			rctd_div(&ctmp, &alpha, &omega);
			rctd_mul(&beta, &beta, &ctmp);

			/* p := r + beta (p - omega v) */
			//rtd_neg(dtmp, omega);
			sub_cmul_ctdvector(vec[4], vec[3], &omega, vec[5]);
			add_cmul_ctdvector(vec[3], vec[1], &beta, vec[4]);
		}

		/* v := Apt */
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_ctdrsmatrix_ctdvec(vec[5], a, vec[3]);
	#else // USE_OMP_VERSION
		mul_ctdrsmatrix_ctdvec(vec[5], a, vec[3]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cdrsmatrix_ctdvec(vec[5], a, vec[3]);
	#else // USE_OMP_VERSION
		mul_cdrsmatrix_ctdvec(vec[5], a, vec[3]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_ctdmatrix_ctdvec(vec[5], a, vec[3]);
#endif // USE_SPARSE_VERSION

		/* precondition */
		// v := M^(-1)Apt -> Solve Mv = Apt for v
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_ctdvector(dvec[0], vec[5]);
#ifdef USE_SPARSE_D_VERSION
			solve_iLU0_cdrsmatrix_ctdvec(vec[5], ilu, dvec[0]);
#else // USE_SPRARSE_D_VERSION
			solve_iLU0_ctdrsmatrix(vec[5], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

		ip_ctdvector(&alpha_den, vec[2], vec[5]);
        rctd_abs_td(dtmp, &alpha_den);

		//if(rctd_cmp_si(alpha_den, 0UL) == 0)
		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(CTDBiCGSTAB, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		rctd_div(&alpha, &rho, &alpha_den);

		/* s = r - alpha v */
		//rtd_neg(dtmp, alpha);
		sub_cmul_ctdvector(vec[6], vec[1], &alpha, vec[5]);

		/* Stopping Criteria */
		norm2_ctdvector(dtmp, vec[6]);
		rtd_mul(dtmp1, reps, init_resnorm);
		rtd_add(dtmp1, dtmp1, aeps);
		if(rtd_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			/* x = x + alpha pt */
			add_cmul_ctdvector(vec[0], vec[0], &alpha, vec[3]);

			subst_ctdvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_ctdrsmatrix_ctdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
		mul_ctdrsmatrix_ctdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cdrsmatrix_ctdvec(vec[8], a, vec[6]);	
	#else // USE_OMP_VERSION
		mul_cdrsmatrix_ctdvec(vec[8], a, vec[6]);	
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_ctdmatrix_ctdvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION

		/* precondition */
		// v := M^(-1)Apt -> Solve Mv = Apt for v
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_ctdvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
			solve_iLU0_cdrsmatrix_ctdvec(vec[8], ilu, dvec[0]);
#else // USE_SPRARSE_D_VERSION
			solve_iLU0_ctdrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

		/* omega = (t, s) / (t, t) */
		ip_ctdvector(&omega_den, vec[8], vec[8]);
        rctd_abs_td(dtmp, &omega_den);
		//if(rctd_cmp_si(omega_den, 0UL) == 0)
		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Omega is zero!(CTDBiCGSTAB, %ld)\n", times);
			return_val = -3; // Fix!
			break;
		}

		ip_ctdvector(&omega, vec[8], vec[6]);
        rctd_abs_td(dtmp, &omega);
		//if(rctd_cmp_si(omega, 0UL) == 0)
		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Omega is zero!(CTDBiCGSTAB, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}
		rctd_div(&omega, &omega, &omega_den);

		/* x = x + alpha pt + omega st */
		add_cmul_ctdvector(vec[4], vec[0], &alpha, vec[3]);
		add_cmul_ctdvector(vec[0], vec[4], &omega, vec[6]);

		/* residual */
		//rtd_neg(dtmp, omega);
		sub_cmul_ctdvector(vec[1], vec[6], &omega, vec[8]);

		//ip_ctdvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		//rtd_sqrt(dtmp, beta_num);
        norm2_ctdvector(dtmp, vec[1]);
		#ifdef USE_PRECOND
		if(norm2_res_history != NULL) norm2_res_history[times + 1] = dtmp[0];
		#endif // USE_PRECOND
		rtd_mul(dtmp1, reps, init_resnorm);
		rtd_add(dtmp1, dtmp1, aeps);
		if(rtd_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_ctdvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		rctd_set(&old_rho, &rho);
	}

	/* Not converge */
	subst_ctdvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 9; i++)
		free_ctdvector(vec[i]);

#ifdef USE_PRECOND
	free_ctdvector(dvec[0]);
	free_ctdvector(dvec[1]);
#endif // USE_PRECOND

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(CTDBiCGSTAB, %ld)\n", times);
		return_val = -5;
	}

	// Fix!
	return return_val;
}

/************************************************************/
/*                                                          */
/* GPBiCG(Generalized Product Bi-CG) Method                 */
/*                                      for Complex Matrix  */
/*                                 (TD Precision)           */
/*                                                          */
/*                 ver. 0.0 2024-11-25 (Sat) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CTDGPBiCG_sp(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CTDGPBiCG_sp_iLU0(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CTDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CTDGPBiCG_sp(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int CTDGPBiCG_sp_iLU0(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CTDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CTDGPBiCG_sp_d(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CTDGPBiCG_sp_d_iLU0(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CTDGPBiCG_sp_d(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int CTDGPBiCG_sp_d_iLU0(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CTDGPBiCG(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CTDGPBiCG_iLU0(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CTDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CTDGPBiCG(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int CTDGPBiCG_iLU0(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes, CTDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       CTDVector answer: Solution for Ax = b              */
/*       CTDMatrix a: Coefficient matrix A                  */
/*                                       (given by user)    */
/*       CTDVector b: Constant vector b   (given by user)   */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       CTDVector answer: Solution for Ax = b              */
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
	double dtmp[TDSIZE], dtmp1[TDSIZE];
    ctdfloat ctmp, ctmp1;
	ctdfloat alpha, alpha_num, alpha_den;
	ctdfloat beta, beta_num;
	ctdfloat rho, old_rho;
	ctdfloat mu[5], tau, zeta, eta;
	double init_resnorm[TDSIZE];
	CTDVector vec[12]; /* Temporary Vectors */
	CTDVector dvec[2]; // Preconditioning

	dim = answer->re->dim;;

/* Set initial value */
	for(i = 0; i < 12; i++)
		vec[i] = init_ctdvector(dim);

#ifdef USE_PRECOND
	dvec[0] = init_ctdvector(dim);
	dvec[1] = init_ctdvector(dim);
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

	subst_ctdvector(vec[1], b);
#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_ctdvector(dvec[0], vec[1]);
#ifdef USE_SPARSE_D_VERSION
		solve_iLU0_cdrsmatrix_ctdvec(vec[1], ilu, dvec[0]);
#else // USE_SPRARSE_D_VERSION
		solve_iLU0_ctdrsmatrix(vec[1], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
	}
#endif // USE_PRECOND
	//subst_ctdvector(vec[2], b);
	subst_ctdvector(vec[2], vec[1]);

	//ip_ctdvector(beta_num, vec[1], vec[1]);
	//rtd_sqrt(init_resnorm, beta_num);
    norm2_ctdvector(init_resnorm, vec[1]);
#ifdef USE_PRECOND
	if(norm2_res_history != NULL) norm2_res_history[0] = init_resnorm[0];
#endif// USE_PRECOND

	rctd_set_ui(&old_rho, 0UL);
	rctd_set_ui(&rho, 0UL);
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_ctdvector(&rho, vec[2], vec[1]);

        rctd_abs_td(dtmp, &rho);
		//if(rctd_cmp_si(rho, 0UL) == 0)
		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(CTDGPBiCG, %ld)\n", times);
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
			subst_ctdvector(vec[3], vec[1]);
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_ctdrsmatrix_ctdvec(vec[4], a, vec[3]);
	#else // USE_OMP_VERSION
			mul_ctdrsmatrix_ctdvec(vec[4], a, vec[3]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_cdrsmatrix_ctdvec(vec[4], a, vec[3]);
	#else // USE_OMP_VERSION
			mul_cdrsmatrix_ctdvec(vec[4], a, vec[3]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
			mul_ctdmatrix_ctdvec(vec[4], a, vec[3]);
#endif // USE_SPARSE_VERSION

#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_ctdvector(dvec[0], vec[4]);
#ifdef USE_SPARSE_D_VERSION
				solve_iLU0_cdrsmatrix_ctdvec(vec[4], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
				solve_iLU0_ctdrsmatrix(vec[4], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
			}
#endif // USE_PRECOND

			ip_ctdvector(&alpha_den, vec[2], vec[4]);
            rctd_abs_td(dtmp, &alpha_den);
			//if(rctd_cmp_si(alpha_den, 0UL) == 0)
			if(rtd_cmp_ui(dtmp, 0UL) == 0)
			{
				fprintf(stderr, "Denominator of Alpha is zero!(CTDGPBiCG, %ld)\n", times);
				return_val = -2; // Fix!
				break; // Fix!
			}
			rctd_div(&alpha, &rho, &alpha_den);

			//rtd_neg(dtmp, alpha);
			sub_cmul_ctdvector(vec[6], vec[1], &alpha, vec[4]);
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_ctdrsmatrix_ctdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
			mul_ctdrsmatrix_ctdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_cdrsmatrix_ctdvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
			mul_cdrsmatrix_ctdvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
			mul_ctdmatrix_ctdvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION

#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_ctdvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
				solve_iLU0_cdrsmatrix_ctdvec(vec[8], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
				solve_iLU0_ctdrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
			}
#endif // USE_PRECOND

			cmul_ctdvector(vec[10], &alpha, vec[4]);
			sub_ctdvector(vec[10], vec[10], vec[1]);
			ip_ctdvector(&mu[1], vec[8], vec[6]);
			ip_ctdvector(&mu[4], vec[8], vec[8]);
            rctd_abs_td(dtmp, &mu[4]);
			//if(rctd_cmp_si(&mu[4], 0UL) == 0)
			if(rtd_cmp_ui(dtmp, 0UL) == 0)
			{
				fprintf(stderr, "Denominator of mu[4] is zero!(CTDGPBiCG, %ld)\n", times);
				return_val = -3; // Fix!
				break;
			}
			rctd_div(&zeta, &mu[1], &mu[4]);
			rctd_set_ui(&eta, 0UL);
			cmul_ctdvector(vec[7], &zeta, vec[4]);
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

			rctd_div(&beta, &rho, &old_rho);
			rctd_div(&ctmp, &alpha, &zeta);
			rctd_mul(&beta, &beta, &ctmp);

			add_cmul_ctdvector(vec[9], vec[8], &beta, vec[4]);
			sub_ctdvector(vec[3], vec[3], vec[7]);
			add_cmul_ctdvector(vec[3], vec[1], &beta, vec[3]);

#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_ctdrsmatrix_ctdvec(vec[4], a, vec[3]);
	#else // USE_OMP_VERSION
			mul_ctdrsmatrix_ctdvec(vec[4], a, vec[3]);
	#endif // USE_OMP_VERSION 
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_cdrsmatrix_ctdvec(vec[4], a, vec[3]);
	#else //ifdef USE_OMP_VERSION
			mul_cdrsmatrix_ctdvec(vec[4], a, vec[3]);
	#endif // def USE_OMP_VERSION
#else // USE_SPARSE_VERSION
			mul_ctdmatrix_ctdvec(vec[4], a, vec[3]);
#endif // USE_SPARSE_VERSION

#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_ctdvector(dvec[0], vec[4]);
#ifdef USE_SPARSE_D_VERSION
				solve_iLU0_cdrsmatrix_ctdvec(vec[4], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
				solve_iLU0_ctdrsmatrix(vec[4], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
			}
#endif // USE_PRECOND

			ip_ctdvector(&alpha_den, vec[2], vec[4]);
            rctd_abs_td(dtmp, &alpha_den);
			//if(rctd_cmp_si(alpha_den, 0UL) == 0)
			if(rtd_cmp_ui(dtmp, 0UL) == 0)
			{
				fprintf(stderr, "Denominator of Alpha is zero!(CTDGPBiCG, %ld)\n", times);
				return_val = -2; // Fix!
				break; // Fix!
			}
			rctd_div(&alpha, &rho, &alpha_den);
			sub_ctdvector(vec[5], vec[6], vec[1]);
			//rtd_neg(dtmp, alpha);
			sub_cmul_ctdvector(vec[6], vec[1], &alpha, vec[4]);

#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_ctdrsmatrix_ctdvec(vec[8], a, vec[6]);
	#else // ifdef USE_OMP_VERSION
			//mul_ctdrsmatrix_ctdvec(vec[4], a, vec[3]);
			mul_ctdrsmatrix_ctdvec(vec[8], a, vec[6]);
	#endif //def USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
			_bncomp_mul_cdrsmatrix_ctdvec(vec[8], a, vec[6]);
	#else // ifdef USE_OMP_VERSION
			//mul_cdrsmatrix_ctdvec(vec[4], a, vec[3]);
			mul_cdrsmatrix_ctdvec(vec[8], a, vec[6]);
	#endif //ifdef USE_OMP_VERSION
#else // USE_SPARSE_VERSION
			//mul_ctdmatrix_ctdvec(vec[4], a, vec[3]);
			mul_ctdmatrix_ctdvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION

#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_ctdvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
				solve_iLU0_cdrsmatrix_ctdvec(vec[8], ilu, dvec[0]);
#else // USE_SPARSE_D_VERSION
				solve_iLU0_ctdrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
			}
#endif // USE_PRECOND

			sub_ctdvector(vec[10], vec[9], vec[4]);

			//rtd_neg(dtmp, alpha);
			sub_cmul_ctdvector(vec[10], vec[5], &alpha, vec[10]);
			ip_ctdvector(&mu[0], vec[10], vec[10]);
			ip_ctdvector(&mu[1], vec[8], vec[6]);
			ip_ctdvector(&mu[2], vec[10], vec[6]);
			ip_ctdvector(&mu[3], vec[8], vec[10]);
			ip_ctdvector(&mu[4], vec[8], vec[8]);

			/* tau = mu[4] * mu[0] - mu[3] * mu[3]; */
			rctd_mul(&ctmp, &mu[4], &mu[0]);
			rctd_mul(&ctmp1, &mu[3], &mu[3]);
			rctd_sub(&tau, &ctmp, &ctmp1);

            rctd_abs_td(dtmp, &tau);
			//if(rctd_cmp_si(tau, 0UL) == 0)
			if(rtd_cmp_ui(dtmp, 0UL) == 0)
			{
				fprintf(stderr, "Denominator of Tau is zero!(CTDGPBiCG, %ld)\n", times);
				return_val = -3; // Fix!
				break;
			}

			/* zeta = (mu[0] * mu[1] - mu[2] * mu[3]) / tau; */
			rctd_mul(&ctmp, &mu[0], &mu[1]);
			rctd_mul(&ctmp1, &mu[2], &mu[3]);
			rctd_sub(&zeta, &ctmp, &ctmp1);
			rctd_div(&zeta, &zeta, &tau);

			/* eta = (mu[4] * mu[2] - mu[3] * mu[1]) / tau; */
			rctd_mul(&ctmp, &mu[4], &mu[2]);
			rctd_mul(&ctmp1, &mu[3], &mu[1]);
			rctd_sub(&eta, &ctmp, &ctmp1);
			rctd_div(&eta, &eta, &tau);

			/* u = zeta * q + eta * (s + k_beta * u); */
			add_cmul_ctdvector(vec[7], vec[5], &beta, vec[7]);
			cmul_ctdvector(vec[7], &eta, vec[7]);
			add_cmul_ctdvector(vec[7], vec[7], &zeta, vec[4]);
		}
		/* z = zeta * ret_res + eta * z - alpha * u; */
		cmul2_ctdvector(vec[11], &eta);
		add_cmul_ctdvector(vec[11], vec[11], &zeta, vec[1]);
		//rtd_neg(dtmp, alpha);
		sub_cmul_ctdvector(vec[11], vec[11], &alpha, vec[7]);

		/* x = x + alpha p + z */
		add_cmul_ctdvector(vec[0], vec[0], &alpha, vec[3]);
		add2_ctdvector(vec[0], vec[11]);

		/* residual */
		/* r = t - eta * y - zeta * v; */
		//rtd_neg(dtmp, eta);
		sub_cmul_ctdvector(vec[1], vec[6], &eta, vec[10]);
		//rtd_neg(dtmp, zeta);
		sub_cmul_ctdvector(vec[1], vec[1], &zeta, vec[8]);

		//ip_ctdvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		//rtd_sqrt(dtmp, beta_num);
        norm2_ctdvector(dtmp, vec[1]);
#ifdef USE_PRECOND
		if(norm2_res_history != NULL) norm2_res_history[times + 1] = dtmp[0];
#endif// USE_PRECOND
		rtd_mul(dtmp1, reps, init_resnorm);
		rtd_add(dtmp1, dtmp1, aeps);
		if(rtd_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_ctdvector(answer, vec[0]);
			return_val = times; // Fix!
			break;
		}

        rctd_abs_td(dtmp, &zeta);
		//if(rctd_cmp_si(zeta, 0UL) == 0)
		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Zeta is zero!(CTDGPBiCG, %ld)\n", times);
			return_val = -4; // Fix!
			break;
		}

		rctd_set(&old_rho, &rho);
	}

	/* Not converge */
	subst_ctdvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 12; i++)
		free_ctdvector(vec[i]);

#ifdef USE_PRECOND
	free_ctdvector(dvec[0]);
	free_ctdvector(dvec[1]);
#endif // USE_PRECOND

	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(CTDGPBiCG, %ld)\n", times);
		return_val = -5; // Fix!
	}

	// Fix!
	return return_val;
}

#endif // USE_DDLINEAR
