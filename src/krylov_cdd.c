/********************************************************************************/
/* krylov_cdd.c:                                                                */
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



//#if 0
/************************************************************/
/*                                                          */
/*                Bi-Conjugate-Gradient Method              */
/*                                 (DD Precision)           */
/*                                                          */
/*                 ver. 0.0 2024-11-23 (Sat) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDDBiCG_sp(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CDDBiCG_sp_iLU0(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDDBiCG_sp(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int CDDBiCG_sp_iLU0(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDDBiCG_sp_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
			#ifdef USE_VEC_D
			long int _bncomp_CDDBiCG_sp_d_iLU0_vec_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
			#else // USE_VEC_D
			long int _bncomp_CDDBiCG_sp_d_iLU0(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
			#endif // USE_VEC_D
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDDBiCG_sp_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
			#ifdef USE_VEC_D
			long int CDDBiCG_sp_d_iLU0_vec_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
			#else // USE_VEC_D
			long int CDDBiCG_sp_d_iLU0(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
			#endif // USE_VEC_D
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDDBiCG(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CDDBiCG_iLU0(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDDBiCG(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int CDDBiCG_iLU0(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       CDDVector answer: Solution for Ax = b              */
/*       CDDMatrix a: Coefficient matrix A                  */
/*                                       (given by user)    */
/*       CDDVector b: Constant vector b   (given by user)   */
/*       double *reps: Relative tolerance (given by user)   */
/*       double *aeps: Absolute tolerance (given by user)   */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       CDDVector answer: Solution for Ax = b              */
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
	cddfloat alpha, alpha_num, alpha_den;
	cddfloat rho, old_rho;
	cddfloat beta, beta_num;
	cddfloat conj_alpha, conj_beta;
	double dtmp[DDSIZE], dtmp1[DDSIZE], init_resnorm[DDSIZE];
	CDDVector vec[9]; /* Temporary Vectors */
	CDDVector dvec[2]; // for Preconditioning

	dim = answer->re->dim;;

/* Set initial value */
	for(i = 0; i < 9; i++)
		vec[i] = init_cddvector(dim);

#ifdef USE_PRECOND
	dvec[0] = init_cddvector(dim);
	dvec[1] = init_cddvector(dim);
#endif // USE_PRECOND

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0]  == w*/
	/* vec[2] ... (b - a * vec[0])^T  == wt */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... z */
	/* vec[6] ... z^T */

	subst_cddvector(vec[1], b);
	subst_cddvector(vec[2], b);
	subst_cddvector(vec[7], vec[1]);
	subst_cddvector(vec[8], vec[2]);
#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_cddvector(dvec[0], vec[1]);
		subst_cddvector(dvec[1], vec[2]);
#ifdef USE_SPARSE_D_VERSION
		//solve_iLU0_cdrsmatrix_cddvec(vec[1], ilu, dvec[0]);
		//solve_iLU0s_cdrsmatrix_cddvec(vec[2], ilu, dvec[1]);
		solve_iLU0_cdrsmatrix_cddvec(vec[7], ilu, dvec[0]);
		solve_iLU0s_cdrsmatrix_cddvec(vec[8], ilu, dvec[1]);
#else // USE_SPARSE_D_VERSION
		//solve_iLU0_cddrsmatrix(vec[1], ilu, dvec[0]);
		//solve_iLU0s_cddrsmatrix(vec[2], ilu, dvec[1]);
		solve_iLU0_cddrsmatrix(vec[7], ilu, dvec[0]);
		solve_iLU0s_cddrsmatrix(vec[8], ilu, dvec[1]);
#endif // 
	}
#endif // USE_PRECOND
	//subst_cddvector(vec[2], b);
	//subst_cddvector(vec[2], vec[1]);

	//ip_cddvector(beta_num, vec[1], vec[1]);
	//rdd_sqrt(init_resnorm, beta_num);
    norm2_cddvector(init_resnorm, vec[1]);
	#ifdef USE_PRECOND
	if(norm2_res_history != NULL) norm2_res_history[0] = init_resnorm[0];
	#endif // USE_PRECOND
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		//ip_cddvector(&rho, vec[2], vec[1]);
		ip_cddvector(&rho, vec[2], vec[7]);

        rcdd_abs_dd(dtmp, &rho);
		//if(rcdd_cmp_si(&rho, 0UL) == 0)
        if(rdd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(CDDBiCG, %ld)\n", times);
			return_val = -1; // Fix!
			break;
		}

		if(times == 0)
		{
			/* p := w, pt := wt */
			//subst_cddvector(vec[3], vec[1]);
			//subst_cddvector(vec[4], vec[2]);
			subst_cddvector(vec[3], vec[7]);
			subst_cddvector(vec[4], vec[8]);
		}
		else
		{
			rcdd_div(&beta, &rho, &old_rho);

			/* p := w + beta p, pt := wt + beta * pt */
			//add_cmul_cddvector(vec[3], vec[1], &beta, vec[3]);
			add_cmul_cddvector(vec[3], vec[7], &beta, vec[3]);
			//add_cmul_cddvector(vec[4], vec[2], &beta, vec[4]);
			rcdd_conj(&conj_beta, &beta);
			//add_cmul_cddvector(vec[4], vec[2], &conj_beta, vec[4]);
			add_cmul_cddvector(vec[4], vec[8], &conj_beta, vec[4]);
		}
		/* z := Ap, zt := A^T pt */
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			//subst_cddvector(dvec[1], vec[4]);
#ifdef USE_SPARSE_D_VERSION
			//solve_iLU0t_cdrsmatrix_cddvec(vec[6], ilu, dvec[1]);
			//solve_iLU0s_cdrsmatrix_cddvec(vec[4], ilu, dvec[1]);
#else // USE_SPARSE_D_VERSION
			//solve_iLU0t_cddrsmatrix(vec[6], ilu, dvec[1]);
			//solve_iLU0s_cddrsmatrix(vec[4], ilu, dvec[1]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

#ifdef USE_SPARSE_VERSION
#ifdef USE_OMP_VERSION
		_bncomp_mul_cddrsmatrix_cddvec(vec[5], a, vec[3]);
		_bncomp_mul_cddrsmatrixs_cddvec(vec[6], a, vec[4]);
#else // USE_OMP_VERSION
		mul_cddrsmatrix_cddvec(vec[5], a, vec[3]);
		mul_cddrsmatrixs_cddvec(vec[6], a, vec[4]);
#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
#ifdef USE_OMP_VERSION
		_bncomp_mul_cdrsmatrix_cddvec(vec[5], a, vec[3]);
		_bncomp_mul_cdrsmatrixs_cddvec(vec[6], a, vec[4]);
#else // USE_OMP_VERSION
		mul_cdrsmatrix_cddvec(vec[5], a, vec[3]);
		mul_cdrsmatrixs_cddvec(vec[6], a, vec[4]);
#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_cddmatrix_cddvec(vec[5], a, vec[3]);
		mul_cddmatrixs_cddvec(vec[6], a, vec[4]);
#endif // USE_SPARSE_VERSION

		ip_cddvector(&alpha_den, vec[4], vec[5]);

        rcdd_abs_dd(dtmp, &alpha_den);
		//if(rcdd_cmp_si(alpha_den, 0UL) == 0)
		if(rdd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(CDDBiCG, %ld)\n", times);
			return_val = -2; // Fix!
			break;
		}
		rcdd_div(&alpha, &rho, &alpha_den);

		/* x = x + alpha p */
		add_cmul_cddvector(vec[0], vec[0], &alpha, vec[3]);

		/* residual */
		//rcdd_neg(dtmp, alpha);
		sub_cmul_cddvector(vec[1], vec[1], &alpha, vec[5]);
		//sub_cmul_cddvector(vec[2], vec[2], &alpha, vec[6]);
		rcdd_conj(&conj_alpha, &alpha);
		sub_cmul_cddvector(vec[2], vec[2], &conj_alpha, vec[6]);

		//ip_cddvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		//rdd_sqrt(dtmp, beta_num);
        norm2_cddvector(dtmp, vec[1]);
		#ifdef USE_PRECOND
		if(norm2_res_history != NULL) norm2_res_history[times + 1] = dtmp[0];
		#endif // USE_PRECOND
		rdd_mul(dtmp1, reps, init_resnorm);
		rdd_add(dtmp1, dtmp1, aeps);
		if(rdd_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_cddvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		subst_cddvector(vec[7], vec[1]);
		subst_cddvector(vec[8], vec[2]);
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_cddvector(dvec[0], vec[7]);
			subst_cddvector(dvec[1], vec[8]);
#ifdef USE_SPARSE_D_VERSION
			solve_iLU0_cdrsmatrix_cddvec(vec[7], ilu, dvec[0]);
			//solve_iLU0t_cdrsmatrix_cddvec(vec[6], ilu, dvec[1]);
			solve_iLU0s_cdrsmatrix_cddvec(vec[8], ilu, dvec[1]);
#else // USE_SPARSE_D_VERSION
			solve_iLU0_cddrsmatrix(vec[7], ilu, dvec[0]);
			//solve_iLU0t_cddrsmatrix(vec[6], ilu, dvec[1]);
			solve_iLU0s_cddrsmatrix(vec[8], ilu, dvec[1]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND


		rcdd_set(&old_rho, &rho);
	}

	/* Not converge */
	subst_cddvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 9; i++)
		free_cddvector(vec[i]);

#ifdef USE_PRECOND
	free_cddvector(dvec[0]);
	free_cddvector(dvec[1]);
#endif // USE_PRECOND

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(CDDBiCG, %ld)\n", times);
		return_val = -5;
	}

	return return_val; // Fix!

}
//#endif // 0

#if 0
/************************************************************/
/*                                                          */
/* Bi-Conjugate-Gradient Method for Complex Matrix          */
/*                                 (DD Precision)           */
/*                                                          */
/*                 ver. 0.0 2024-11-20 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDDBiCG_sp(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CDDBiCG_sp_iLU0(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDDBiCG_sp(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int CDDBiCG_sp_iLU0(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDDBiCG_sp_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
			#ifdef USE_VEC_D
			long int _bncomp_CDDBiCG_sp_d_iLU0_vec_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
			#else // USE_VEC_D
			long int _bncomp_CDDBiCG_sp_d_iLU0(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
			#endif // USE_VEC_D
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDDBiCG_sp_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
			#ifdef USE_VEC_D
			long int CDDBiCG_sp_d_iLU0_vec_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
			#else // USE_VEC_D
			long int CDDBiCG_sp_d_iLU0(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
			#endif // USE_VEC_D
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDDBiCG(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CDDBiCG_iLU0(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDDBiCG(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int CDDBiCG_iLU0(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       CDDVector answer: Solution for Ax = b              */
/*       CDDMatrix a: Coefficient matrix A                  */
/*                                       (given by user)    */
/*       CDDVector b: Constant vector b   (given by user)   */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       CDDVector answer: Solution for Ax = b              */
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
	cddfloat alpha, alpha_num, alpha_den;
	cddfloat rho, old_rho;
	cddfloat beta, beta_num;
	cddfloat conj_alpha, conj_beta;
	double dtmp[DDSIZE], dtmp1[DDSIZE], init_resnorm[DDSIZE];
	CDDVector vec[9]; /* Temporary Vectors */
	CDDVector dvec[2]; // for Preconditioning

	dim = answer->re->dim;

/* Set initial value */
	for(i = 0; i < 9; i++)
		vec[i] = init_cddvector(dim);

#ifdef USE_PRECOND
	dvec[0] = init_cddvector(dim);
	dvec[1] = init_cddvector(dim);
#endif // USE_PRECOND

	/* vec[0] ... approximation of solution */
	/* vec[1] ... residual : b - a * vec[0]  == w */
	/* vec[2] ... (b - a * vec[0])^*  == wt */
	/* vec[3] ... p */
	/* vec[4] ... p^T */
	/* vec[5] ... z */
	/* vec[6] ... z^T */

	subst_cddvector(vec[1], b);
	conj_cddvector(vec[2], b);

	//beta_num = ip_cddvector(vec[2], vec[1]);
	//init_resnorm = sqrt(beta_num);
    norm2_cddvector(init_resnorm, vec[1]);
	#ifdef USE_PRECOND
	if(norm2_res_history != NULL) norm2_res_history[0] = init_resnorm[0];
	#endif // USE_PRECOND

	return_val = 0; // Fix!

	/* p := w, pt := wt */
	subst_cddvector(vec[3], vec[1]);
	subst_cddvector(vec[4], vec[2]);
#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_cddvector(dvec[0], vec[1]);
		subst_cddvector(dvec[1], vec[2]);
#ifdef USE_SPARSE_D_VERSION
		solve_iLU0_cdrsmatrix_cddvec(vec[3], ilu, dvec[0]);
		solve_iLU0t_cdrsmatrix_cddvec(vec[4], ilu, dvec[1]);
#else // USE_SPARSE_D_VERSION
		solve_iLU0_cddrsmatrix(vec[3], ilu, dvec[0]);
		solve_iLU0t_cddrsmatrix(vec[4], ilu, dvec[1]);
#endif // USE_SPARSE_D_VERSION
	}
#endif // USE_PRECOND
	ip_cddvector(&rho, vec[2], vec[3]);

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* z := Ap, zt := A^H pt */
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cddrsmatrix_cddvec(vec[5], a, vec[3]);
		_bncomp_mul_cddrsmatrixs_cddvec(vec[6], a, vec[4]);
	#else // USE_OMP_VERSION
		mul_cddrsmatrix_cddvec(vec[5], a, vec[3]);
		mul_cddrsmatrixs_cddvec(vec[6], a, vec[4]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION // USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cdrsmatrix_cddvec(vec[5], a, vec[3]);
		_bncomp_mul_cdrsmatrixs_cddvec(vec[6], a, vec[4]);
	#else // USE_OMP_VERSION
		mul_cdrsmatrix_cddvec(vec[5], a, vec[3]);
		mul_cdrsmatrixs_cddvec(vec[6], a, vec[4]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_cddmatrix_cddvec(vec[5], a, vec[3]);
		mul_cddmatrixs_cddvec(vec[6], a, vec[4]);
#endif // USE_SPARSE_VERSION
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_cddvector(dvec[0], vec[5]);
			subst_cddvector(dvec[1], vec[6]);
#ifndef USE_SPARSE_D_VERSION
			solve_iLU0_cddrsmatrix(vec[5], ilu, dvec[0]);
			solve_iLU0t_cddrsmatrix(vec[6], ilu, dvec[1]);
#else // USE_SPARSE_D_VERSION
			solve_iLU0_cdrsmatrix_cddvec(vec[5], ilu, dvec[0]);
			solve_iLU0t_cdrsmatrix_cddvec(vec[6], ilu, dvec[1]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND


		ip_cddvector(&alpha_den, vec[4], vec[5]);
		if(rcdd_cmp_abs_ui(&alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(CDDBiCG, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		rcdd_div(&alpha, &rho, &alpha_den);

		/* x = x + alpha p */
		add_cmul_cddvector(vec[0], vec[0], &alpha, vec[3]);

		/* residual */
		//add_cmul_cddvector(vec[1], vec[1], -alpha, vec[5]);
		//add_cmul_cddvector(vec[2], vec[2], -alpha, vec[6]);
		sub_cmul_cddvector(vec[1], vec[1], &alpha, vec[5]);
		//sub_cmul_cddvector(vec[2], vec[2], alpha, vec[6]);
		rcdd_conj(&conj_alpha, &alpha);
		sub_cmul_cddvector(vec[2], vec[2], &conj_alpha, vec[6]);

		//beta_num = ip_cddvector(vec[1], vec[1]);
        //beta_num = ip_cddvector(vec[2], vec[1]);

		/* Stopping Criteria */
		//dtmp = sqrt(beta_num);
        norm2_cddvector(dtmp, vec[1]);
		#ifdef USE_PRECOND
		if(norm2_res_history != NULL) norm2_res_history[times + 1] = dtmp[0];
		#endif // USE_PRECOND
		//printf("%5ld %10.3e\n", times, dtmp);
		rdd_mul(dtmp1, reps, init_resnorm);
		rdd_add(dtmp1, dtmp1, aeps);
		if(rdd_cmp(dtmp, dtmp1) <= 0)
		{
			subst_cddvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		/* rho */
		subst_cddvector(vec[7], vec[1]);
		subst_cddvector(vec[8], vec[2]);

		//old_rho = rho;
		rcdd_set(&old_rho, &rho);

		ip_cddvector(&rho, vec[2], vec[7]);
		if(rcdd_cmp_abs_ui(&rho, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(CDDBiCG, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}

		//beta = rho / old_rho;
		rcdd_div(&beta, &rho, &old_rho);

		/* p := w + beta p, pt := wt + beta * pt */
		add_cmul_cddvector(vec[3], vec[7], &beta, vec[3]);
		//add_cmul_cddvector(vec[4], vec[2], beta, vec[4]);
		rcdd_conj(&conj_beta, &beta);
		add_cmul_cddvector(vec[4], vec[8], &conj_beta, vec[4]);
	}

	/* Not converge */
	subst_cddvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 9; i++)
		free_cddvector(vec[i]);

#ifdef USE_PRECOND
	free_cddvector(dvec[0]);
	free_cddvector(dvec[1]);
#endif // USE_PRECOND

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(CDDBiCG, %ld)\n", times);
		return_val = -5; // Fix!
	}

	return return_val; // Fix!

}
#endif // 0

/************************************************************/
/*                                                          */
/*               CGS Method for Complex Matrix              */
/*                                 (DD Precision)           */
/*                                                          */
/*                 ver. 0.0 2024-11-23 (Sat) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDDCGS_sp(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CDDCGS_sp_iLU0(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDDCGS_sp(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int CDDCGS_sp_iLU0(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDDCGS_sp_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
			#ifdef USE_VEC_D
			long int _bncomp_CDDCGS_sp_d_iLU0_vec_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
			#else // USE_VEC_D
			long int _bncomp_CDDCGS_sp_d_iLU0(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
			#endif // USE_VEC_D
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDDCGS_sp_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
			#ifdef USE_VEC_D
			long int CDDCGS_sp_d_iLU0_vec_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
			#else // USE_VEC_D
			long int CDDCGS_sp_d_iLU0(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
			#endif // USE_VEC_D
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDDCGS(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CDDCGS_iLU0(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDDCGS(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int CDDCGS_iLU0(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       CDDVector answer: Solution for Ax = b             */
/*       CDDMatrix a: Coefficient matrix A                  */
/*                                       (given by user)    */
/*       CDDVector b: Constant vector b  (given by user)   */
/*       double reps: Relative tolerance (given by user)     */
/*       double aeps: Absolute tolerance (given by user)     */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       CDDVector answer: Solution for Ax = b              */
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
	cddfloat ctmp, ctmp1;
	cddfloat alpha, alpha_num, alpha_den;
	cddfloat beta, beta_num;
	cddfloat rho, old_rho;
	double dtmp[DDSIZE], dtmp1[DDSIZE], init_resnorm[DDSIZE];
	CDDVector vec[9]; /* Temporary Vectors */
	CDDVector dvec[2]; // for preconditioning
	CDVector in_dvec[2]; // with inner dvecs

	dim = answer->re->dim;;

/* Set initial value */
	for(i = 0; i < 9; i++)
		vec[i] = init_cddvector(dim);

#ifdef USE_PRECOND
	dvec[0] = init_cddvector(dim);
	dvec[1] = init_cddvector(dim);
	#if defined(USE_SPARSE_D_VERSION) && defined(USE_VEC_D)
	in_dvec[0] = init_cdvector(dim);
	in_dvec[1] = init_cdvector(dim);
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

	subst_cddvector(vec[1], b);
#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_cddvector(dvec[0], vec[1]);
#ifdef USE_SPARSE_D_VERSION
	#ifdef USE_VEC_D
		subst_cdvector_cddvec(in_vec[0], dvec[0]);
		solve_iLU0_cdrsmatrix(in_dvec[1], ilu, in_dvec[0]);
		subst_cddvector_cdvec(vec[1], in_vec[1]);
	#else // USE_VEC_D
		solve_iLU0_cdrsmatrix_cddvec(vec[1], ilu, dvec[0]);
	#endif // USE_VEC_D
#else // USE_SPARSE_D_VERSION
		solve_iLU0_cddrsmatrix(vec[1], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
	}
#endif // USE_PRECOND
	//subst_cddvector(vec[2], b);
	subst_cddvector(vec[2], vec[1]);

	//ip_cddvector(beta_num, vec[1], vec[1]);
	//rdd_sqrt(init_resnorm, beta_num);
    norm2_cddvector(init_resnorm, vec[1]);
	#ifdef USE_PRECOND
	if(norm2_res_history != NULL) norm2_res_history[0] = init_resnorm[0];
	#endif // USE_PRECOND

	rcdd_set_ui(&old_rho, 0UL);
	rcdd_set_ui(&rho, 0UL);
	return_val = 0;

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_cddvector(&rho, vec[2], vec[1]);

        rcdd_abs_dd(dtmp, &rho);
		//if(rcdd_cmp_si(rho, 0UL) == 0)
		if(rdd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(CDDCGS, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}

		if(times == 0)
		{
			/* u := rt, p := u */
			subst_cddvector(vec[5], vec[1]);
			subst_cddvector(vec[3], vec[5]);
		}
		else
		{
			rcdd_div(&beta, &rho, &old_rho);

			/* u := r + beta q, p := u + beta * (q + beta p) */
			add_cmul_cddvector(vec[5], vec[1], &beta, vec[7]);
			add_cmul_cddvector(vec[8], vec[7], &beta, vec[3]);
			add_cmul_cddvector(vec[3], vec[5], &beta, vec[8]);
		}
		/* vt := Apt */
#ifdef USE_SPARSE_VERSION
#ifdef USE_OMP_VERSION
		_bncomp_mul_cddrsmatrix_cddvec(vec[8], a, vec[3]);
#else // USE_OMP_VERSION
		mul_cddrsmatrix_cddvec(vec[8], a, vec[3]);
#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
#ifdef USE_OMP_VERSION
		_bncomp_mul_cdrsmatrix_cddvec(vec[8], a, vec[3]);
#else // USE_OMP_VERSION
		mul_cdrsmatrix_cddvec(vec[8], a, vec[3]);
#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_cddmatrix_cddvec(vec[8], a, vec[3]);
#endif // USE_SPARSE_VERSION

		/* precondition */
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_cddvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
	#ifdef USE_VEC_D
			subst_cdvector_cddvec(in_vec[0], dvec[0]);
			solve_iLU0_cdrsmatrix(in_dvec[1], ilu, in_dvec[0]);
			subst_cddvector_cdvec(vec[8], in_vec[1]);
	#else // USE_VEC_D
			solve_iLU0_cdrsmatrix_cddvec(vec[8], ilu, dvec[0]);
	#endif // USE_VEC_D
#else // USE_SPARSE_D_VERSION
			solve_iLU0_cddrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

		ip_cddvector(&alpha_den, vec[2], vec[8]);
        rcdd_abs_dd(dtmp, &alpha_den);
		//if(rcdd_cmp_si(alpha_den, 0UL) == 0)
		if(rdd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(CDDCGS, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		rcdd_div(&alpha, &rho, &alpha_den);

		/* q = u - alpha  vt */
		//rdd_neg(dtmp, alpha);
		sub_cmul_cddvector(vec[7], vec[5], &alpha, vec[8]);

		/* ut = u + q */
		add_cddvector(vec[6], vec[5], vec[7]);

		/* x = x + alpha ut */
		add_cmul_cddvector(vec[0], vec[0], &alpha, vec[6]);

		/* residual */
#ifdef USE_SPARSE_VERSION
#ifdef USE_OMP_VERSION
		_bncomp_mul_cddrsmatrix_cddvec(vec[8], a, vec[6]);
#else // USE_OMP_VERSION
		mul_cddrsmatrix_cddvec(vec[8], a, vec[6]);
#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
#ifdef USE_OMP_VERSION
		_bncomp_mul_cdrsmatrix_cddvec(vec[8], a, vec[6]);
#else // USE_OMP_VERSION
		mul_cdrsmatrix_cddvec(vec[8], a, vec[6]);
#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_cddmatrix_cddvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION
		//rdd_neg(dtmp, alpha);

		/* precondition */
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_cddvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
	#ifdef USE_VEC_D
			subst_cdvector_cddvec(in_vec[0], dvec[0]);
			solve_iLU0_cdrsmatrix(in_dvec[1], ilu, in_dvec[0]);
			subst_cddvector_cdvec(vec[8], in_vec[1]);
	#else // USE_VEC_D
			solve_iLU0_cdrsmatrix_cddvec(vec[8], ilu, dvec[0]);
	#endif // USE_VEC_D
#else // USE_SPARSE_D_VERSION
			solve_iLU0_cddrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

		sub_cmul_cddvector(vec[1], vec[1], &alpha, vec[8]);

		//ip_cddvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		//rdd_sqrt(dtmp, beta_num);
        norm2_cddvector(dtmp, vec[1]);
		#ifdef USE_PRECOND
		if(norm2_res_history != NULL) norm2_res_history[times + 1] = dtmp[0];
		#endif // USE_PRECOND
		rdd_mul(dtmp1, reps, init_resnorm);
		rdd_add(dtmp1, dtmp1, aeps);
		if(rdd_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_cddvector(answer, vec[0]);
			return_val = times;
			break;
		}

		rcdd_set(&old_rho, &rho);
	}

	/* Not converge */
	subst_cddvector(answer, vec[0]);

	/* free vec[0]..[8]; */
	for(i = 0; i < 9; i++)
		free_cddvector(vec[i]);

#ifdef USE_PRECOND
	free_cddvector(dvec[0]);
	free_cddvector(dvec[1]);
	#if defined(USE_SPARSE_D_VERSION) && defined(USE_VEC_D)
	free_cdvector(in_dvec[0]);
	free_cdvector(in_dvec[1]);
	#endif // defined(USE_SPARSE_D_VERSION) && defined(USE_VEC_D)
#endif // USE_PRECOND

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(CDDCGS, %ld)\n", times);
		return_val = -5;
	}

	// Fix!
	return return_val;
}

/************************************************************/
/*                                                          */
/*             BiCGSTAB Method for Complex Matrix           */
/*                                 (DD Precision)           */
/*                                                          */
/*                 ver. 0.0 2024-11-23 (Sat) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDDBiCGSTAB_sp(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CDDBiCGSTAB_sp_iLU0(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDDBiCGSTAB_sp(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int CDDBiCGSTAB_sp_iLU0(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDDBiCGSTAB_sp_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
			#ifdef USE_VEC_D
			long int _bncomp_CDDBiCGSTAB_sp_d_iLU0_vec_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
			#else // USE_VEC_D
			long int _bncomp_CDDBiCGSTAB_sp_d_iLU0(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
			#endif // USE_VEC_D
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDDBiCGSTAB_sp_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
			#ifdef USE_VEC_D
			long int CDDBiCGSTAB_sp_d_iLU0_vec_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
			#else // USE_VEC_D
			long int CDDBiCGSTAB_sp_d_iLU0(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
			#endif // USE_VEC_D
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDDBiCGSTAB(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CDDBiCGSTAB_iLU0(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDDBiCGSTAB(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int CDDBiCGSTAB_iLU0(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       CDDVector answer: Solution for Ax = b              */
/*       CDDMatrix a: Coefficient matrix A                  */
/*                                       (given by user)    */
/*       CDDVector b: Constant vector b   (given by user)   */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       CDDVector answer: Solution for Ax = b              */
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
	double dtmp[DDSIZE], dtmp1[DDSIZE];
    cddfloat ctmp, ctmp1;
	cddfloat alpha, alpha_num, alpha_den;
	cddfloat beta, beta_num;
	cddfloat rho, old_rho;
	cddfloat omega, omega_den;
	double init_resnorm[DDSIZE];
	CDDVector vec[9]; /* Temporary Vectors */
	CDDVector dvec[2]; // for Preconditioning
	CDVector in_dvec[2]; // with inner dvecs

	dim = answer->re->dim;;

/* Set initial value */
	for(i = 0; i < 9; i++)
		vec[i] = init_cddvector(dim);

#ifdef USE_PRECOND
	dvec[0] = init_cddvector(dim);
	dvec[1] = init_cddvector(dim);
	#if defined(USE_SPARSE_D_VERSION) && defined(USE_VEC_D)
	in_dvec[0] = init_cdvector(dim);
	in_dvec[1] = init_cdvector(dim);
	#endif // defined(USE_SPARSE_D_VERSION) && defined(USE_VEC_D)
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

	subst_cddvector(vec[1], b);
#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_cddvector(dvec[0], vec[1]);
#ifdef USE_SPARSE_D_VERSION
	#ifdef USE_VEC_D
		subst_cdvector_cddvec(in_dvec[0], dvec[0]);
		solve_iLU0_cdrsmatrix(in_dvec[1], ilu, in_dvec[0]);
		subst_cddvector_cdvec(vec[1], in_dvec[1]);
	#else // USE_VEC_D
		solve_iLU0_cdrsmatrix_cddvec(vec[1], ilu, dvec[0]);
	#endif // USE_VEC_D
#else // USE_SPRARSE_D_VERSION
		solve_iLU0_cddrsmatrix(vec[1], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
	}
#endif // USE_PRECOND
	//subst_cddvector(vec[2], b);
	subst_cddvector(vec[2], vec[1]);

	//ip_cddvector(beta_num, vec[1], vec[1]);
	//rdd_sqrt(init_resnorm, beta_num);
    norm2_cddvector(init_resnorm, vec[1]);
	#ifdef USE_PRECOND
	if(norm2_res_history != NULL) norm2_res_history[0] = init_resnorm[0];
	#endif // USE_PRECOND

	rcdd_set_ui(&old_rho, 0UL);
	rcdd_set_ui(&rho, 0UL);
	return_val = 0;

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_cddvector(&rho, vec[2], vec[1]);

        rcdd_abs_dd(dtmp, &rho);
		//if(rcdd_cmp_si(rho, 0UL) == 0)
		if(rdd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(CDDBiCGSTAB, %ld)\n", times);
			return_val = -1; // Fix!
			break;
		}

		if(times == 0)
		{
			/* p := r */
			subst_cddvector(vec[3], vec[1]);
		}
		else
		{
			//rcdd_div(&beta, &rho, &old_rho);
			rcdd_div(&ctmp1, &rho, &old_rho);
			rcdd_div(&ctmp, &alpha, &omega);
			//rcdd_mul(&beta, &beta, &ctmp);
			rcdd_mul(&beta, &ctmp1, &ctmp);

			/* p := r + beta (p - omega v) */
			//rdd_neg(dtmp, omega);
			sub_cmul_cddvector(vec[4], vec[3], &omega, vec[5]);
			add_cmul_cddvector(vec[3], vec[1], &beta, vec[4]);
		}
		
		/* v := Apt */
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cddrsmatrix_cddvec(vec[5], a, vec[3]);
	#else // USE_OMP_VERSION
		mul_cddrsmatrix_cddvec(vec[5], a, vec[3]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cdrsmatrix_cddvec(vec[5], a, vec[3]);
	#else // USE_OMP_VERSION
		mul_cdrsmatrix_cddvec(vec[5], a, vec[3]);
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_cddmatrix_cddvec(vec[5], a, vec[3]);
#endif // USE_SPARSE_VERSION

		/* precondition */
		// v := M^(-1)Apt -> Solve Mv = Apt for v
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_cddvector(dvec[0], vec[5]);
#ifdef USE_SPARSE_D_VERSION
	#ifdef USE_VEC_D
			subst_cdvector_cddvec(in_dvec[0], dvec[0]);
			solve_iLU0_cdrsmatrix(in_dvec[1], ilu, in_dvec[0]);
			subst_cddvector_cdvec(vec[5], in_dvec[1]);
	#else // USE_VEC_D
			solve_iLU0_cdrsmatrix_cddvec(vec[5], ilu, dvec[0]);
	#endif // USE_VEC_D
#else // USE_SPRARSE_D_VERSION
			solve_iLU0_cddrsmatrix(vec[5], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

		ip_cddvector(&alpha_den, vec[2], vec[5]);
        rcdd_abs_dd(dtmp, &alpha_den);

		//if(rcdd_cmp_si(alpha_den, 0UL) == 0)
		if(rdd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(CDDBiCGSTAB, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}
		rcdd_div(&alpha, &rho, &alpha_den);

		/* s = r - alpha v */
		//rdd_neg(dtmp, alpha);
		sub_cmul_cddvector(vec[6], vec[1], &alpha, vec[5]);

		/* Stopping Criteria */
		norm2_cddvector(dtmp, vec[6]);
		rdd_mul(dtmp1, reps, init_resnorm);
		rdd_add(dtmp1, dtmp1, aeps);
		if(rdd_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			/* x = x + alpha pt */
			add_cmul_cddvector(vec[0], vec[0], &alpha, vec[3]);

			subst_cddvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cddrsmatrix_cddvec(vec[8], a, vec[6]);
	#else // USE_OMP_VERSION
		mul_cddrsmatrix_cddvec(vec[8], a, vec[6]);
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		_bncomp_mul_cdrsmatrix_cddvec(vec[8], a, vec[6]);	
	#else // USE_OMP_VERSION
		mul_cdrsmatrix_cddvec(vec[8], a, vec[6]);	
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
		mul_cddmatrix_cddvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION

		/* precondition */
		// v := M^(-1)Apt -> Solve Mv = Apt for v
#ifdef USE_PRECOND
		if(ilu != NULL)
		{
			subst_cddvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
	#ifdef USE_VEC_D
			subst_cdvector_cddvec(in_dvec[0], dvec[0]);
			solve_iLU0_cdrsmatrix(in_dvec[1], ilu, in_dvec[0]);
			subst_cddvector_cdvec(vec[8], in_dvec[1]);
	#else // USE_VEC_D
			solve_iLU0_cdrsmatrix_cddvec(vec[8], ilu, dvec[0]);
	#endif // USE_VEC_D
#else // USE_SPRARSE_D_VERSION
			solve_iLU0_cddrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
		}
#endif // USE_PRECOND

		/* omega = (t, s) / (t, t) */
		ip_cddvector(&omega_den, vec[8], vec[8]);
        rcdd_abs_dd(dtmp, &omega_den);
		//if(rcdd_cmp_si(omega_den, 0UL) == 0)
		if(rdd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Omega is zero!(CDDBiCGSTAB, %ld)\n", times);
			return_val = -3; // Fix!
			break;
		}

		ip_cddvector(&omega, vec[8], vec[6]);
        rcdd_abs_dd(dtmp, &omega);
		//if(rcdd_cmp_si(omega, 0UL) == 0)
		if(rdd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Omega is zero!(CDDBiCGSTAB, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}
		rcdd_div(&omega, &omega, &omega_den);

		/* x = x + alpha pt + omega st */
		add_cmul_cddvector(vec[4], vec[0], &alpha, vec[3]);
		add_cmul_cddvector(vec[0], vec[4], &omega, vec[6]);

		/* residual */
		//rdd_neg(dtmp, omega);
		sub_cmul_cddvector(vec[1], vec[6], &omega, vec[8]);

		//ip_cddvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		//rdd_sqrt(dtmp, beta_num);
        norm2_cddvector(dtmp, vec[1]);
		#ifdef USE_PRECOND
		if(norm2_res_history != NULL) norm2_res_history[times + 1] = dtmp[0];
		#endif // USE_PRECOND
		rdd_mul(dtmp1, reps, init_resnorm);
		rdd_add(dtmp1, dtmp1, aeps);
		if(rdd_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_cddvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		rcdd_set(&old_rho, &rho);
	}

	/* Not converge */
	subst_cddvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 9; i++)
		free_cddvector(vec[i]);

#ifdef USE_PRECOND
	free_cddvector(dvec[0]);
	free_cddvector(dvec[1]);
	#if defined(USE_SPARSE_D_VERSION) && defined(USE_VEC_D)
	free_cdvector(in_dvec[0]);
	free_cdvector(in_dvec[1]);
	#endif // defined(USE_SPARSE_D_VERSION) && defined(USE_VEC_D)
#endif // USE_PRECOND

	// Fix!
	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(CDDBiCGSTAB, %ld)\n", times);
		return_val = -5;
	}

	// Fix!
	return return_val;
}

/************************************************************/
/*                                                          */
/* GPBiCG(Generalized Product Bi-CG) Method                 */
/*                                      for Complex Matrix  */
/*                                 (DD Precision)           */
/*                                                          */
/*                 ver. 0.0 2024-11-23 (Sat) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDDGPBiCG_sp(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CDDGPBiCG_sp_iLU0(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDDGPBiCG_sp(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int CDDGPBiCG_sp_iLU0(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDRSMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDDGPBiCG_sp_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
			#ifdef USE_VEC_D
			long int _bncomp_CDDGPBiCG_sp_d_iLU0_vec_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
			#else // USE_VEC_D
			long int _bncomp_CDDGPBiCG_sp_d_iLU0(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
			#endif // USE_VEC_D
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDDGPBiCG_sp_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
			#ifdef USE_VEC_D
			long int CDDGPBiCG_sp_d_iLU0_vec_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
			#else // USE_VEC_D
			long int CDDGPBiCG_sp_d_iLU0(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDRSMatrix ilu, double *norm2_res_history)
			#endif // USE_VEC_D
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
	#ifdef USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int _bncomp_CDDGPBiCG(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int _bncomp_CDDGPBiCG_iLU0(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#else // USE_OMP_VERSION
		#ifndef USE_PRECOND
		long int CDDGPBiCG(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
		#else // USE_PRECOND
		long int CDDGPBiCG_iLU0(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes, CDDMatrix ilu, double *norm2_res_history)
		#endif // USE_PRECOND
	#endif // USE_OMP_VERSION
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       CDDVector answer: Solution for Ax = b              */
/*       CDDMatrix a: Coefficient matrix A                  */
/*                                       (given by user)    */
/*       CDDVector b: Constant vector b   (given by user)   */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       CDDVector answer: Solution for Ax = b              */
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
	double dtmp[DDSIZE], dtmp1[DDSIZE];
    cddfloat ctmp, ctmp1;
	cddfloat alpha, alpha_num, alpha_den;
	cddfloat beta, beta_num;
	cddfloat rho, old_rho;
	cddfloat mu[5], tau, zeta, eta;
	double init_resnorm[DDSIZE];
	CDDVector vec[12]; /* Temporary Vectors */
	CDDVector dvec[2]; // Preconditioning
	CDVector in_dvec[2]; // Preconditioning with inner dvecs

	dim = answer->re->dim;;

/* Set initial value */
	for(i = 0; i < 12; i++)
		vec[i] = init_cddvector(dim);

#ifdef USE_PRECOND
	dvec[0] = init_cddvector(dim);
	dvec[1] = init_cddvector(dim);
	#if defined(USE_SPARSE_D_VERSION) && defined(USE_VEC_D)
	in_dvec[0] = init_cdvector(dim);
	in_dvec[1] = init_cdvector(dim);
	#endif // defined(USE_SPARSE_D_VERSION) && defined(USE_VEC_D)
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

	subst_cddvector(vec[1], b); 
#ifdef USE_PRECOND
	if(ilu != NULL)
	{
		subst_cddvector(dvec[0], vec[1]);
#ifdef USE_SPARSE_D_VERSION
		#ifdef USE_VEC_D
			subst_cdvector_cddvec(in_dvec[0], dvec[0]);
			solve_iLU0_cdrsmatrix(in_dvec[1], ilu, in_dvec[0]);
			subst_cddvector_cdvec(vec[1], in_dvec[1]);
		#else // USE_VEC_D
			solve_iLU0_cdrsmatrix_cddvec(vec[1], ilu, dvec[0]);
		#endif // USE_VEC_D
#else // USE_SPRARSE_D_VERSION
		solve_iLU0_cddrsmatrix(vec[1], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
	}
#endif // USE_PRECOND
	//subst_cddvector(vec[2], b);
	subst_cddvector(vec[2], vec[1]);

	//ip_cddvector(beta_num, vec[1], vec[1]);
	//rdd_sqrt(init_resnorm, beta_num);
    norm2_cddvector(init_resnorm, vec[1]);
#ifdef USE_PRECOND
	if(norm2_res_history != NULL) norm2_res_history[0] = init_resnorm[0];
#endif// USE_PRECOND

	rcdd_set_ui(&old_rho, 0UL);
	rcdd_set_ui(&rho, 0UL);
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* rho */
		ip_cddvector(&rho, vec[2], vec[1]);

        rcdd_abs_dd(dtmp, &rho);
		//if(rcdd_cmp_si(rho, 0UL) == 0)
		if(rdd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Rho is zero!(CDDGPBiCG, %ld)\n", times);
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
			subst_cddvector(vec[3], vec[1]);
#ifdef USE_SPARSE_VERSION
#ifdef USE_OMP_VERSION
			_bncomp_mul_cddrsmatrix_cddvec(vec[4], a, vec[3]);
#else // USE_OMP_VERSION
			mul_cddrsmatrix_cddvec(vec[4], a, vec[3]);
#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
#ifdef USE_OMP_VERSION
			_bncomp_mul_cdrsmatrix_cddvec(vec[4], a, vec[3]);
#else // USE_OMP_VERSION
			mul_cdrsmatrix_cddvec(vec[4], a, vec[3]);
#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
			mul_cddmatrix_cddvec(vec[4], a, vec[3]);
#endif // USE_SPARSE_VERSION

#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_cddvector(dvec[0], vec[4]);
#ifdef USE_SPARSE_D_VERSION
		#ifdef USE_VEC_D
				subst_cdvector_cddvec(in_dvec[0], dvec[0]);
				solve_iLU0_cdrsmatrix(in_dvec[1], ilu, in_dvec[0]);
				subst_cddvector_cdvec(vec[4], in_dvec[1]);
		#else // USE_VEC_D
				solve_iLU0_cdrsmatrix_cddvec(vec[4], ilu, dvec[0]);
		#endif // USE_VEC_D
#else // USE_SPARSE_D_VERSION
				solve_iLU0_cddrsmatrix(vec[4], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
			}
#endif // USE_PRECOND

			ip_cddvector(&alpha_den, vec[2], vec[4]);
            rcdd_abs_dd(dtmp, &alpha_den);
			//if(rcdd_cmp_si(alpha_den, 0UL) == 0)
			if(rdd_cmp_ui(dtmp, 0UL) == 0)
			{
				fprintf(stderr, "Denominator of Alpha is zero!(CDDGPBiCG, %ld)\n", times);
				return_val = -2; // Fix!
				break; // Fix!
			}
			rcdd_div(&alpha, &rho, &alpha_den);

			//rdd_neg(dtmp, alpha);
			sub_cmul_cddvector(vec[6], vec[1], &alpha, vec[4]);
#ifdef USE_SPARSE_VERSION
#ifdef USE_OMP_VERSION
			_bncomp_mul_cddrsmatrix_cddvec(vec[8], a, vec[6]);
#else // USE_OMP_VERSION
			mul_cddrsmatrix_cddvec(vec[8], a, vec[6]);
#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
#ifdef USE_OMP_VERSION
			_bncomp_mul_cdrsmatrix_cddvec(vec[8], a, vec[6]);
#else // USE_OMP_VERSION
			mul_cdrsmatrix_cddvec(vec[8], a, vec[6]);
#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
			mul_cddmatrix_cddvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION

#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_cddvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
		#ifdef USE_VEC_D
				subst_cdvector_cddvec(in_dvec[0], dvec[0]);
				solve_iLU0_cdrsmatrix(in_dvec[1], ilu, in_dvec[0]);
				subst_cddvector_cdvec(vec[8], in_dvec[1]);
		#else // USE_VEC_D
				solve_iLU0_cdrsmatrix_cddvec(vec[8], ilu, dvec[0]);
		#endif // USE_VEC_D
#else // USE_SPARSE_D_VERSION
				solve_iLU0_cddrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
			}
#endif // USE_PRECOND

			cmul_cddvector(vec[10], &alpha, vec[4]);
			sub_cddvector(vec[10], vec[10], vec[1]);
			ip_cddvector(&mu[1], vec[8], vec[6]);
			ip_cddvector(&mu[4], vec[8], vec[8]);
            rcdd_abs_dd(dtmp, &mu[4]);
			//if(rcdd_cmp_si(&mu[4], 0UL) == 0)
			if(rdd_cmp_ui(dtmp, 0UL) == 0)
			{
				fprintf(stderr, "Denominator of mu[4] is zero!(CDDGPBiCG, %ld)\n", times);
				return_val = -3; // Fix!
				break;
			}
			rcdd_div(&zeta, &mu[1], &mu[4]);
			rcdd_set_ui(&eta, 0UL);
			cmul_cddvector(vec[7], &zeta, vec[4]);
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

			rcdd_div(&beta, &rho, &old_rho);
			rcdd_div(&ctmp, &alpha, &zeta);
			rcdd_mul(&beta, &beta, &ctmp);

			add_cmul_cddvector(vec[9], vec[8], &beta, vec[4]);
			sub_cddvector(vec[3], vec[3], vec[7]);
			add_cmul_cddvector(vec[3], vec[1], &beta, vec[3]);
#ifdef USE_SPARSE_VERSION
#ifdef USE_OMP_VERSION
			_bncomp_mul_cddrsmatrix_cddvec(vec[4], a, vec[3]);
#else // USE_OMP_VERSION
			mul_cddrsmatrix_cddvec(vec[4], a, vec[3]);
#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
#ifdef USE_OMP_VERSION
			_bncomp_mul_cdrsmatrix_cddvec(vec[4], a, vec[3]);
#else // USE_OMP_VERSION
			mul_cdrsmatrix_cddvec(vec[4], a, vec[3]);
#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
			mul_cddmatrix_cddvec(vec[4], a, vec[3]);
#endif // USE_SPARSE_VERSION

#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_cddvector(dvec[0], vec[4]);
#ifdef USE_SPARSE_D_VERSION
		#ifdef USE_VEC_D
				subst_cdvector_cddvec(in_dvec[0], dvec[0]);
				solve_iLU0_cdrsmatrix(in_dvec[1], ilu, in_dvec[0]);
				subst_cddvector_cdvec(vec[4], in_dvec[1]);
		#else // USE_VEC_D
				solve_iLU0_cdrsmatrix_cddvec(vec[4], ilu, dvec[0]);
		#endif // USE_VEC_D
#else // USE_SPARSE_D_VERSION
				solve_iLU0_cddrsmatrix(vec[4], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
			}
#endif // USE_PRECOND

			ip_cddvector(&alpha_den, vec[2], vec[4]);
            rcdd_abs_dd(dtmp, &alpha_den);
			//if(rcdd_cmp_si(alpha_den, 0UL) == 0)
			if(rdd_cmp_ui(dtmp, 0UL) == 0)
			{
				fprintf(stderr, "Denominator of Alpha is zero!(CDDGPBiCG, %ld)\n", times);
				return_val = -2; // Fix!
				break; // Fix!
			}
			rcdd_div(&alpha, &rho, &alpha_den);
			sub_cddvector(vec[5], vec[6], vec[1]);
			//rdd_neg(dtmp, alpha);
			sub_cmul_cddvector(vec[6], vec[1], &alpha, vec[4]);
#ifdef USE_SPARSE_VERSION
#ifdef USE_OMP_VERSION
			_bncomp_mul_cddrsmatrix_cddvec(vec[8], a, vec[6]);
#else // USE_OMP_VERSION
			//mul_cddrsmatrix_cddvec(vec[4], a, vec[3]);
			mul_cddrsmatrix_cddvec(vec[8], a, vec[6]);
#endif // USE_OMP_VERSION
#elif USE_SPARSE_D_VERSION
#ifdef USE_OMP_VERSION
			_bncomp_mul_cdrsmatrix_cddvec(vec[8], a, vec[6]);
#else // USE_OMP_VERSION
			//mul_cdrsmatrix_cddvec(vec[4], a, vec[3]);
			mul_cdrsmatrix_cddvec(vec[8], a, vec[6]);
#endif // USE_OMP_VERSION
#else // USE_SPARSE_VERSION
			//mul_cddmatrix_cddvec(vec[4], a, vec[3]);
			mul_cddmatrix_cddvec(vec[8], a, vec[6]);
#endif // USE_SPARSE_VERSION

#ifdef USE_PRECOND
			if(ilu != NULL)
			{
				subst_cddvector(dvec[0], vec[8]);
#ifdef USE_SPARSE_D_VERSION
		#ifdef USE_VEC_D
				subst_cdvector_cddvec(in_dvec[0], dvec[0]);
				solve_iLU0_cdrsmatrix(in_dvec[1], ilu, in_dvec[0]);
				subst_cddvector_cdvec(vec[8], in_dvec[1]);
		#else // USE_VEC_D
				solve_iLU0_cdrsmatrix_cddvec(vec[8], ilu, dvec[0]);
		#endif // USE_VEC_D
#else // USE_SPARSE_D_VERSION
				solve_iLU0_cddrsmatrix(vec[8], ilu, dvec[0]);
#endif // USE_SPARSE_D_VERSION
			}
#endif // USE_PRECOND

			sub_cddvector(vec[10], vec[9], vec[4]);

			//rdd_neg(dtmp, alpha);
			sub_cmul_cddvector(vec[10], vec[5], &alpha, vec[10]);
			ip_cddvector(&mu[0], vec[10], vec[10]);
			ip_cddvector(&mu[1], vec[8], vec[6]);
			ip_cddvector(&mu[2], vec[10], vec[6]);
			ip_cddvector(&mu[3], vec[8], vec[10]);
			ip_cddvector(&mu[4], vec[8], vec[8]);

			/* tau = mu[4] * mu[0] - mu[3] * mu[3]; */
			rcdd_mul(&ctmp, &mu[4], &mu[0]);
			rcdd_mul(&ctmp1, &mu[3], &mu[3]);
			rcdd_sub(&tau, &ctmp, &ctmp1);

            rcdd_abs_dd(dtmp, &tau);
			//if(rcdd_cmp_si(tau, 0UL) == 0)
			if(rdd_cmp_ui(dtmp, 0UL) == 0)
			{
				fprintf(stderr, "Denominator of Tau is zero!(CDDGPBiCG, %ld)\n", times);
				return_val = -3; // Fix!
				break;
			}

			/* zeta = (mu[0] * mu[1] - mu[2] * mu[3]) / tau; */
			rcdd_mul(&ctmp, &mu[0], &mu[1]);
			rcdd_mul(&ctmp1, &mu[2], &mu[3]);
			rcdd_sub(&zeta, &ctmp, &ctmp1);
			rcdd_div(&zeta, &zeta, &tau);

			/* eta = (mu[4] * mu[2] - mu[3] * mu[1]) / tau; */
			rcdd_mul(&ctmp, &mu[4], &mu[2]);
			rcdd_mul(&ctmp1, &mu[3], &mu[1]);
			rcdd_sub(&eta, &ctmp, &ctmp1);
			rcdd_div(&eta, &eta, &tau);

			/* u = zeta * q + eta * (s + k_beta * u); */
			add_cmul_cddvector(vec[7], vec[5], &beta, vec[7]);
			cmul_cddvector(vec[7], &eta, vec[7]);
			add_cmul_cddvector(vec[7], vec[7], &zeta, vec[4]);
		}
		/* z = zeta * ret_res + eta * z - alpha * u; */
		cmul2_cddvector(vec[11], &eta);
		add_cmul_cddvector(vec[11], vec[11], &zeta, vec[1]);
		//rdd_neg(dtmp, alpha);
		sub_cmul_cddvector(vec[11], vec[11], &alpha, vec[7]);

		/* x = x + alpha p + z */
		add_cmul_cddvector(vec[0], vec[0], &alpha, vec[3]);
		add2_cddvector(vec[0], vec[11]);

		/* residual */
		/* r = t - eta * y - zeta * v; */
		//rdd_neg(dtmp, eta);
		sub_cmul_cddvector(vec[1], vec[6], &eta, vec[10]);
		//rdd_neg(dtmp, zeta);
		sub_cmul_cddvector(vec[1], vec[1], &zeta, vec[8]);

		//ip_cddvector(beta_num, vec[1], vec[1]);

		/* Stopping Criteria */
		//rdd_sqrt(dtmp, beta_num);
        norm2_cddvector(dtmp, vec[1]);
#ifdef USE_PRECOND
		if(norm2_res_history != NULL) norm2_res_history[times + 1] = dtmp[0];
#endif// USE_PRECOND
		rdd_mul(dtmp1, reps, init_resnorm);
		rdd_add(dtmp1, dtmp1, aeps);
		if(rdd_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_cddvector(answer, vec[0]);
			return_val = times; // Fix!
			break;
		}

        rcdd_abs_dd(dtmp, &zeta);
		//if(rcdd_cmp_si(zeta, 0UL) == 0)
		if(rdd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Zeta is zero!(CDDGPBiCG, %ld)\n", times);
			return_val = -4; // Fix!
			break;
		}

		rcdd_set(&old_rho, &rho);
	}

	/* Not converge */
	subst_cddvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 12; i++)
		free_cddvector(vec[i]);

#ifdef USE_PRECOND
	free_cddvector(dvec[0]);
	free_cddvector(dvec[1]);
	#if defined(USE_SPARSE_D_VERSION) && defined(USE_VEC_D)
	free_cdvector(in_dvec[0]); // = init_cdvector(dim);
	free_cdvector(in_dvec[1]); // = init_cdvector(dim);
	#endif // defined(USE_SPARSE_D_VERSION) && defined(USE_VEC_D)
#endif // USE_PRECOND

	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(CDDGPBiCG, %ld)\n", times);
		return_val = -5; // Fix!
	}

	// Fix!
	return return_val;
}

