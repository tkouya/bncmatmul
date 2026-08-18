/********************************************************************************/
/* cg.c:                                                                        */
/* Copyright (C) 2003- Tomonori Kouya                                           */
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
#endif // _STDIO_H

/* math.h */
#ifndef _MATH_H
#include <math.h>
#endif // _MATH_H

//#include "bnc.h"

//#ifdef USE_SPARSE_VERSION
/* Sparse Matrix */
#include "bncsparse.h"
//#endif

/************************************************************/
/*                                                          */
/*                 Conjugate-Gradient Method                */
/*       for Real Symmetric Positive Definite Matrix        */
/*                                 (Single Precision)       */
/*                                                          */
/*                 ver. 0.0 1997.10.25 (Sat) Tomonori Kouya */
/*                 ver. 0.1 2000.02.24 (Thu) Tomonori Kouya */
/*                 ver. 0.2 2012-03-17 (Sat) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifndef USE_SPARSE_VERSION
#if 0
long int FCG(FVector answer, FMatrix a, FVector b, float reps, float aeps, long int maxtimes)
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       FVector answer: Solution for Ax = b                */
/*       FMatrix a: Coefficient matrix A                    */
/*                                       (given by user)    */
/*       FVector b: Constant vector b   (given by user)     */
/*       float reps: Relative tolerance (given by user)     */
/*       float aeps: Absolute tolerance (given by user)     */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       FVector answer: Solution for Ax = b                */
/*                                                          */
/* ERRORS                                                   */
/* Positive value ... Normal : Iterative Times              */
/*      -1 ... Deminator of Alpha is zero.                  */
/*      -2 ... Numerator of Alpha is zero.                  */
/*      -3 ... Deminator of Beta is zero.                   */
/*      -4 ... Numerator of Beta is zero.                   */
/*      -5 ... Not Converge.                                */
/*                                                          */
/************************************************************/
{
	long int i, j, times, dim, return_val; // Fix!
	float alpha, alpha_num, alpha_den;
	float beta, beta_num, beta_den;
	float dtmp, init_resnorm;
	FVector vec[4]; /* Temporary Vectors */

	dim = answer->dim;

/* Set initial value */
	for(i = 0; i < 4; i++)
		vec[i] = init_fvector(dim);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... p */
	/* vec[2] ... residual : b - a * vec[0] */
	/* vec[3] ... a * p */
	subst_fvector(vec[1], b); 
	subst_fvector(vec[2], b);

	beta_num = ip_fvector(vec[2], vec[2]);
	init_resnorm = sqrt(beta_num);
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* Ap */
		mul_fmatrix_fvec(vec[3], a, vec[1]);

		/* alpha = alpha_num / alpha_den */
		alpha_den = ip_fvector(vec[1], vec[3]);
		alpha_num = ip_fvector(vec[1], vec[2]);
		if(alpha_den == 0.0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(FCG, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}
		if(alpha_num == 0.0)
		{
			fprintf(stderr, "Numerator of Alpha is zero!(FCG, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}

		alpha = alpha_num / alpha_den;

		/* x = x + alpha p */
		for(i = 0; i < dim; i++)
		{
			dtmp = get_fvector_i(vec[0], i) + alpha * get_fvector_i(vec[1], i);
			set_fvector_i(vec[0], i, dtmp);
		}

		/* residual */
		beta_den = beta_num;
		for(i = 0; i < dim; i++)
		{
			dtmp = get_fvector_i(vec[2], i) - alpha * get_fvector_i(vec[3], i);
			set_fvector_i(vec[2], i, dtmp);
		}
		beta_num = ip_fvector(vec[2], vec[2]);

		/* Stopping Criteria */
		dtmp = sqrt(beta_num);
		if(dtmp < aeps + reps * init_resnorm)
		{
			subst_fvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		if(beta_den == 0.0)
		{
			fprintf(stderr, "Denominator of Beta is zero!(FCG, %ld)\n", times);
			return_val = -3; // Fix!
			break; // Fix!
		}
		if(beta_num == 0.0)
		{
			fprintf(stderr, "Numerator of Beta is zero!(FCG, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}

		/* beta */
		beta = beta_num / beta_den;

		/* p */
		for(i = 0; i < dim; i++)
		{
			dtmp = get_fvector_i(vec[2], i) + beta * get_fvector_i(vec[1], i);
			set_fvector_i(vec[1], i, dtmp);
		}
	}

	/* Not converge */
	subst_fvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 4; i++)
		free_fvector(vec[i]);

	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(FCG, %ld)\n", times);
		return_val = -5; // Fix!
	}

}
#endif // 0
#endif // USE_SPARSE_VERSION

#ifndef USE_SPARSE_D_VERSION
 
/************************************************************/
/*                                                          */
/*                 Conjugate-Gradient Method                */
/*       for Real Symmetric Positive Definite Matrix        */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 1997.10.25 (Sat) Tomonori Kouya */
/*                 ver. 0.1 2000.02.24 (Thu) Tomonori Kouya */
/*                 ver. 0.2 2012-03-17 (Sat) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
  long int DCG_sp(DVector answer, DRSMatrix a, DVector b, double reps, double aeps, long int maxtimes)
#else
 // long int DCG(DVector answer, DMatrix a, DVector b, double reps, double aeps, long int maxtimes)
  long int bnc_DCG(DVector answer, DMatrix a, DVector b, double reps, double aeps, long int maxtimes)
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
/*      -1 ... Deminator of Alpha is zero.                  */
/*      -2 ... Numerator of Alpha is zero.                  */
/*      -3 ... Deminator of Beta is zero.                   */
/*      -4 ... Numerator of Beta is zero.                   */
/*      -5 ... Not Converge.                                */
/*                                                          */
/************************************************************/
{
	long int i, j, times, dim, return_val; // Fix!
	double alpha, alpha_num, alpha_den;
	double beta, beta_num, beta_den;
	double dtmp, init_resnorm;
	DVector vec[4]; /* Temporary Vectors */

	dim = answer->dim;

/* Set initial value */
	for(i = 0; i < 4; i++)
		vec[i] = init_dvector(dim);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... p */
	/* vec[2] ... residual : b - a * vec[0] */
	/* vec[3] ... a * p */
	subst_dvector(vec[1], b); 
	subst_dvector(vec[2], b);

	beta_num = ip_dvector(vec[2], vec[2]);
	init_resnorm = sqrt(beta_num);
	return_val = 0;

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* Ap */
#ifdef USE_SPARSE_VERSION
		mul_drsmatrix_dvec(vec[3], a, vec[1]);
#else
		mul_dmatrix_dvec(vec[3], a, vec[1]);
#endif

		/* alpha = alpha_num / alpha_den */
		alpha_den = ip_dvector(vec[1], vec[3]);
		alpha_num = ip_dvector(vec[1], vec[2]);
		if(alpha_den == 0.0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(DCG, %ld)\n", times);
			return_val = -1; // Fix!
			break;
		}
		if(alpha_num == 0.0)
		{
			fprintf(stderr, "Numerator of Alpha is zero!(DCG, %ld)\n", times);
			return_val = -2; // Fix!
			break;
		}

		alpha = alpha_num / alpha_den;

		/* x = x + alpha p */
		for(i = 0; i < dim; i++)
		{
			dtmp = get_dvector_i(vec[0], i) + alpha * get_dvector_i(vec[1], i);
			set_dvector_i(vec[0], i, dtmp);
		}

		/* residual */
		beta_den = beta_num;
		for(i = 0; i < dim; i++)
		{
			dtmp = get_dvector_i(vec[2], i) - alpha * get_dvector_i(vec[3], i);
			set_dvector_i(vec[2], i, dtmp);
		}
		beta_num = ip_dvector(vec[2], vec[2]);

		/* Stopping Criteria */
		dtmp = sqrt(beta_num);
		if(dtmp < aeps + reps * init_resnorm)
		{
			subst_dvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		if(beta_den == 0.0)
		{
			fprintf(stderr, "Denominator of Beta is zero!(DCG, %ld)\n", times);
			return_val = -3; // Fix!
			break; // Fix!
		}
		if(beta_num == 0.0)
		{
			fprintf(stderr, "Numerator of Beta is zero!(DCG, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}

		/* beta */
		beta = beta_num / beta_den;

		/* p */
		for(i = 0; i < dim; i++)
		{
			dtmp = get_dvector_i(vec[2], i) + beta * get_dvector_i(vec[1], i);
			set_dvector_i(vec[1], i, dtmp);
		}
	}

	/* Not converge */
	subst_dvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 4; i++)
		free_dvector(vec[i]);

	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(DCG, %ld)\n", times);
		return_val = -5;
	}

	return return_val;

}
#endif // ifndef USE_SPARSE_D_VERSION

// DD CG
/************************************************************/
/*                                                          */
/*                 Conjugate-Gradient Method                */
/*       for Real Symmetric Positive Definite Matrix        */
/*                                    (DD precision)        */
/*                                                          */
/*               ver. 0.0 202401-10-29 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
  long int DDCG_sp(DDVector answer, DDRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
#elif USE_SPARSE_D_VERSION
  long int DDCG_sp_d(DDVector answer, DRSMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
#else // USE_SPARSE_VERSION
  long int DDCG(DDVector answer, DDMatrix a, DDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       DDVector answer: Solution for Ax = b               */
/*       DDMatrix a: Coefficient matrix A                   */
/*                                       (given by user)    */
/*       DDVector b: Constant vector b   (given by user)    */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       DDVector answer: Solution for Ax = b               */
/*                                                          */
/* ERRORS                                                   */
/* Positive value ... Normal : Iterative Times              */
/*      -1 ... Deminator of Alpha is zero.                  */
/*      -2 ... Numerator of Alpha is zero.                  */
/*      -3 ... Deminator of Beta is zero.                   */
/*      -4 ... Numerator of Beta is zero.                   */
/*      -5 ... Not Converge.                                */
/*                                                          */
/************************************************************/
{
	long int i, j, times, dim, return_val; // Fix!
	double alpha[DDSIZE], alpha_num[DDSIZE], alpha_den[DDSIZE];
	double beta[DDSIZE], beta_num[DDSIZE], beta_den[DDSIZE];
	double dtmp[DDSIZE], dtmp1[DDSIZE], init_resnorm[DDSIZE];
	DDVector vec[4]; /* Temporary Vectors */

	dim = answer->dim;

	/* Set initial value */
	for(i = 0; i < 4; i++)
		vec[i] = init_ddvector(dim);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... p */
	/* vec[2] ... residual : b - a * vec[0] */
	/* vec[3] ... a * p */
	subst_ddvector(vec[1], b); 
	subst_ddvector(vec[2], b);

	ip_ddvector(beta_num, vec[2], vec[2]);
	rdd_sqrt(init_resnorm, beta_num);
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* Ap */
#ifdef USE_SPARSE_VERSION
		mul_ddrsmatrix_ddvec(vec[3], a, vec[1]);
#elif USE_SPARSE_D_VERSION
		mul_drsmatrix_ddvec(vec[3], a, vec[1]);
#else // USE_SPARSE_VERSION
		mul_ddmatrix_ddvec(vec[3], a, vec[1]);
#endif // USE_SPARSE_VERSION
		/* alpha = alpha_num / alpha_den */
		ip_ddvector(alpha_den, vec[1], vec[3]);
		ip_ddvector(alpha_num, vec[1], vec[2]);
		if(rdd_cmp_ui(alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(DDCG, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}
		if(rdd_cmp_ui(alpha_num, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Alpha is zero!(DDCG, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}

		rdd_div(alpha, alpha_num, alpha_den);

		/* x = x + alpha p */
		for(i = 0; i < dim; i++)
		{
			rdd_mul(dtmp, alpha, get_ddvector_i(vec[1], i));
			rdd_add(dtmp, get_ddvector_i(vec[0], i), dtmp);
			set_ddvector_i(vec[0], i, dtmp);
		}

		/* residual */
		rdd_set(beta_den, beta_num);
		for(i = 0; i < dim; i++)
		{
			rdd_mul(dtmp, alpha, get_ddvector_i(vec[3], i));
			rdd_sub(dtmp, get_ddvector_i(vec[2], i), dtmp);
			set_ddvector_i(vec[2], i, dtmp);
		}
		ip_ddvector(beta_num, vec[2], vec[2]);

		/* Stopping Criteria */
		rdd_sqrt(dtmp, beta_num);
		rdd_mul(dtmp1, reps, init_resnorm);
		rdd_add(dtmp1, dtmp1, aeps);
		if(rdd_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_ddvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		if(rdd_cmp_ui(beta_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Beta is zero!(DDCG, %ld)\n", times);
			return_val = -3; // Fix!
			break; // Fix!
		}
		if(rdd_cmp_ui(beta_num, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Beta is zero!(DDCG, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}

		/* beta */
		rdd_div(beta, beta_num, beta_den);

		/* p */
		for(i = 0; i < dim; i++)
		{
			rdd_mul(dtmp, beta, get_ddvector_i(vec[1], i));
			rdd_add(dtmp, get_ddvector_i(vec[2], i), dtmp);
			set_ddvector_i(vec[1], i, dtmp);
		}
	}

	/* Not converge */
	subst_ddvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 4; i++)
		free_ddvector(vec[i]);

	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(DDCG, %ld)\n", times);
		return_val = -5;
	}

	return return_val;
}

// TD CG
/************************************************************/
/*                                                          */
/*                 Conjugate-Gradient Method                */
/*       for Real Symmetric Positive Definite Matrix        */
/*                                    (TD precision)        */
/*                                                          */
/*               ver. 0.0 202401-10-29 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
  long int TDCG_sp(TDVector answer, TDRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
#elif USE_SPARSE_D_VERSION
  long int TDCG_sp_d(TDVector answer, DRSMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
#else // USE_SPARSE_VERSION
  long int TDCG(TDVector answer, TDMatrix a, TDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       TDVector answer: Solution for Ax = b               */
/*       TDMatrix a: Coefficient matrix A                   */
/*                                       (given by user)    */
/*       TDVector b: Constant vector b   (given by user)    */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       TDVector answer: Solution for Ax = b               */
/*                                                          */
/* ERRORS                                                   */
/* Positive value ... Normal : Iterative Times              */
/*      -1 ... Deminator of Alpha is zero.                  */
/*      -2 ... Numerator of Alpha is zero.                  */
/*      -3 ... Deminator of Beta is zero.                   */
/*      -4 ... Numerator of Beta is zero.                   */
/*      -5 ... Not Converge.                                */
/*                                                          */
/************************************************************/
{
	long int i, j, times, dim, return_val; // Fix!
	double alpha[TDSIZE], alpha_num[TDSIZE], alpha_den[TDSIZE];
	double beta[TDSIZE], beta_num[TDSIZE], beta_den[TDSIZE];
	double dtmp[TDSIZE], dtmp1[TDSIZE], init_resnorm[TDSIZE];
	TDVector vec[4]; /* Temporary Vectors */

	dim = answer->dim;

	/* Set initial value */
	for(i = 0; i < 4; i++)
		vec[i] = init_tdvector(dim);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... p */
	/* vec[2] ... residual : b - a * vec[0] */
	/* vec[3] ... a * p */
	subst_tdvector(vec[1], b); 
	subst_tdvector(vec[2], b);

	ip_tdvector(beta_num, vec[2], vec[2]);
	rtd_sqrt(init_resnorm, beta_num);
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* Ap */
#ifdef USE_SPARSE_VERSION
		mul_tdrsmatrix_tdvec(vec[3], a, vec[1]);
#elif USE_SPARSE_D_VERSION
		mul_drsmatrix_tdvec(vec[3], a, vec[1]);
#else // USE_SPARSE_VERSION
		mul_tdmatrix_tdvec(vec[3], a, vec[1]);
#endif // USE_SPARSE_VERSION
		/* alpha = alpha_num / alpha_den */
		ip_tdvector(alpha_den, vec[1], vec[3]);
		ip_tdvector(alpha_num, vec[1], vec[2]);
		if(rtd_cmp_ui(alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(TDCG, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}
		if(rtd_cmp_ui(alpha_num, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Alpha is zero!(TDCG, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}

		rtd_div(alpha, alpha_num, alpha_den);

		/* x = x + alpha p */
		for(i = 0; i < dim; i++)
		{
			rtd_mul(dtmp, alpha, get_tdvector_i(vec[1], i));
			rtd_add(dtmp, get_tdvector_i(vec[0], i), dtmp);
			set_tdvector_i(vec[0], i, dtmp);
		}

		/* residual */
		rtd_set(beta_den, beta_num);
		for(i = 0; i < dim; i++)
		{
			rtd_mul(dtmp, alpha, get_tdvector_i(vec[3], i));
			rtd_sub(dtmp, get_tdvector_i(vec[2], i), dtmp);
			set_tdvector_i(vec[2], i, dtmp);
		}
		ip_tdvector(beta_num, vec[2], vec[2]);

		/* Stopping Criteria */
		rtd_sqrt(dtmp, beta_num);
		rtd_mul(dtmp1, reps, init_resnorm);
		rtd_add(dtmp1, dtmp1, aeps);
		if(rtd_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_tdvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		if(rtd_cmp_ui(beta_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Beta is zero!(TDCG, %ld)\n", times);
			return_val = -3; // Fix!
			break; // Fix!
		}
		if(rtd_cmp_ui(beta_num, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Beta is zero!(TDCG, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}

		/* beta */
		rtd_div(beta, beta_num, beta_den);

		/* p */
		for(i = 0; i < dim; i++)
		{
			rtd_mul(dtmp, beta, get_tdvector_i(vec[1], i));
			rtd_add(dtmp, get_tdvector_i(vec[2], i), dtmp);
			set_tdvector_i(vec[1], i, dtmp);
		}
	}

	/* Not converge */
	subst_tdvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 4; i++)
		free_tdvector(vec[i]);

	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(TDCG, %ld)\n", times);
		return_val = -5;
	}

	return return_val;
}

// QD CG
/************************************************************/
/*                                                          */
/*                 Conjugate-Gradient Method                */
/*       for Real Symmetric Positive Definite Matrix        */
/*                                    (QD precision)        */
/*                                                          */
/*               ver. 0.0 202401-10-29 (Tue) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
  long int QDCG_sp(QDVector answer, QDRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
#elif USE_SPARSE_D_VERSION
  long int QDCG_sp_d(QDVector answer, DRSMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
#else // USE_SPARSE_VERSION
  long int QDCG(QDVector answer, QDMatrix a, QDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       QDVector answer: Solution for Ax = b               */
/*       QDMatrix a: Coefficient matrix A                   */
/*                                       (given by user)    */
/*       QDVector b: Constant vector b   (given by user)    */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       QDVector answer: Solution for Ax = b               */
/*                                                          */
/* ERRORS                                                   */
/* Positive value ... Normal : Iterative Times              */
/*      -1 ... Deminator of Alpha is zero.                  */
/*      -2 ... Numerator of Alpha is zero.                  */
/*      -3 ... Deminator of Beta is zero.                   */
/*      -4 ... Numerator of Beta is zero.                   */
/*      -5 ... Not Converge.                                */
/*                                                          */
/************************************************************/
{
	long int i, j, times, dim, return_val; // Fix!
	double alpha[QDSIZE], alpha_num[QDSIZE], alpha_den[QDSIZE];
	double beta[QDSIZE], beta_num[QDSIZE], beta_den[QDSIZE];
	double dtmp[QDSIZE], dtmp1[QDSIZE], init_resnorm[QDSIZE];
	QDVector vec[4]; /* Temporary Vectors */

	dim = answer->dim;

	/* Set initial value */
	for(i = 0; i < 4; i++)
		vec[i] = init_qdvector(dim);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... p */
	/* vec[2] ... residual : b - a * vec[0] */
	/* vec[3] ... a * p */
	subst_qdvector(vec[1], b); 
	subst_qdvector(vec[2], b);

	ip_qdvector(beta_num, vec[2], vec[2]);
	rqd_sqrt(init_resnorm, beta_num);
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* Ap */
#ifdef USE_SPARSE_VERSION
		mul_qdrsmatrix_qdvec(vec[3], a, vec[1]);
#elif USE_SPARSE_D_VERSION
		mul_drsmatrix_qdvec(vec[3], a, vec[1]);
#else // USE_SPARSE_VERSION
		mul_qdmatrix_qdvec(vec[3], a, vec[1]);
#endif // USE_SPARSE_VERSION
		/* alpha = alpha_num / alpha_den */
		ip_qdvector(alpha_den, vec[1], vec[3]);
		ip_qdvector(alpha_num, vec[1], vec[2]);
		if(rqd_cmp_ui(alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(QDCG, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}
		if(rqd_cmp_ui(alpha_num, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Alpha is zero!(QDCG, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}

		rqd_div(alpha, alpha_num, alpha_den);

		/* x = x + alpha p */
		for(i = 0; i < dim; i++)
		{
			rqd_mul(dtmp, alpha, get_qdvector_i(vec[1], i));
			rqd_add(dtmp, get_qdvector_i(vec[0], i), dtmp);
			set_qdvector_i(vec[0], i, dtmp);
		}

		/* residual */
		rqd_set(beta_den, beta_num);
		for(i = 0; i < dim; i++)
		{
			rqd_mul(dtmp, alpha, get_qdvector_i(vec[3], i));
			rqd_sub(dtmp, get_qdvector_i(vec[2], i), dtmp);
			set_qdvector_i(vec[2], i, dtmp);
		}
		ip_qdvector(beta_num, vec[2], vec[2]);

		/* Stopping Criteria */
		rqd_sqrt(dtmp, beta_num);
		rqd_mul(dtmp1, reps, init_resnorm);
		rqd_add(dtmp1, dtmp1, aeps);
		if(rqd_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_qdvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		if(rqd_cmp_ui(beta_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Beta is zero!(QDCG, %ld)\n", times);
			return_val = -3; // Fix!
			break; // Fix!
		}
		if(rqd_cmp_ui(beta_num, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Beta is zero!(QDCG, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}

		/* beta */
		rqd_div(beta, beta_num, beta_den);

		/* p */
		for(i = 0; i < dim; i++)
		{
			rqd_mul(dtmp, beta, get_qdvector_i(vec[1], i));
			rqd_add(dtmp, get_qdvector_i(vec[2], i), dtmp);
			set_qdvector_i(vec[1], i, dtmp);
		}
	}

	/* Not converge */
	subst_qdvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 4; i++)
		free_qdvector(vec[i]);

	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(QDCG, %ld)\n", times);
		return_val = -5;
	}

	return return_val;
}


#ifdef USE_GMP
/************************************************************/
/*                                                          */
/*                 Conjugate-Gradient Method                */
/*       for Real Symmetric Positive Definite Matrix        */
/*                                 (Multi-Precision)        */
/*                                                          */
/*                 ver. 0.0 1997.10.25 (Sat) Tomonori Kouya */
/*                 ver. 0.1 2000.02.24 (Thu) Tomonori Kouya */
/*                 ver. 0.2 2001.08.02 (Thu) Tomonori Kouya */
/*                 ver. 0.3 2012-03-17 (Sat) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
  long int MPFCG_sp(MPFVector answer, MPFRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
#elif USE_SPARSE_D_VERSION
  long int MPFCG_sp_d(MPFVector answer, DRSMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
#else // USE_SPARSE_VERSION
  long int MPFCG(MPFVector answer, MPFMatrix a, MPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
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
/*      -3 ... Deminator of Beta is zero.                   */
/*      -4 ... Numerator of Beta is zero.                   */
/*      -5 ... Not Converge.                                */
/*                                                          */
/************************************************************/
{
	unsigned long int prec;
	long int i, j, times, dim, return_val; // Fix!
	mpf_t alpha, alpha_num, alpha_den;
	mpf_t beta, beta_num, beta_den;
	mpf_t dtmp, dtmp1, init_resnorm;
	MPFVector vec[4]; /* Temporary Vectors */

	dim = answer->dim;
	prec = answer->prec;

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

	for(i = 0; i < 4; i++)
		vec[i] = init2_mpfvector(dim, prec);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... p */
	/* vec[2] ... residual : b - a * vec[0] */
	/* vec[3] ... a * p */
	subst_mpfvector(vec[1], b); 
	subst_mpfvector(vec[2], b);

	ip_mpfvector(beta_num, vec[2], vec[2]);
	mpf_sqrt(init_resnorm, beta_num);
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* Ap */
#ifdef USE_SPARSE_VERSION
		mul_mpfrsmatrix_mpfvec(vec[3], a, vec[1]);
#elif USE_SPARSE_D_VERSION
		mul_drsmatrix_mpfvec(vec[3], a, vec[1]);
#else // USE_SPARSE_VERSION
		mul_mpfmatrix_mpfvec(vec[3], a, vec[1]);
#endif // USE_SPARSE_VERSION
		/* alpha = alpha_num / alpha_den */
		ip_mpfvector(alpha_den, vec[1], vec[3]);
		ip_mpfvector(alpha_num, vec[1], vec[2]);
		if(mpf_cmp_ui(alpha_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(MPFCG, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}
		if(mpf_cmp_ui(alpha_num, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Alpha is zero!(MPFCG, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}

		mpf_div(alpha, alpha_num, alpha_den);

		/* x = x + alpha p */
		for(i = 0; i < dim; i++)
		{
			mpf_mul(dtmp, alpha, get_mpfvector_i(vec[1], i));
			mpf_add(dtmp, get_mpfvector_i(vec[0], i), dtmp);
			set_mpfvector_i(vec[0], i, dtmp);
		}

		/* residual */
		mpf_set(beta_den, beta_num);
		for(i = 0; i < dim; i++)
		{
			mpf_mul(dtmp, alpha, get_mpfvector_i(vec[3], i));
			mpf_sub(dtmp, get_mpfvector_i(vec[2], i), dtmp);
			set_mpfvector_i(vec[2], i, dtmp);
		}
		ip_mpfvector(beta_num, vec[2], vec[2]);

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

		if(mpf_cmp_ui(beta_den, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Beta is zero!(MPFCG, %ld)\n", times);
			return_val = -3; // Fix!
			break; // Fix!
		}
		if(mpf_cmp_ui(beta_num, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Beta is zero!(MPFCG, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}

		/* beta */
		mpf_div(beta, beta_num, beta_den);

		/* p */
		for(i = 0; i < dim; i++)
		{
			mpf_mul(dtmp, beta, get_mpfvector_i(vec[1], i));
			mpf_add(dtmp, get_mpfvector_i(vec[2], i), dtmp);
			set_mpfvector_i(vec[1], i, dtmp);
		}
	}

	/* Not converge */
	subst_mpfvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 4; i++)
		free_mpfvector(vec[i]);
	mpf_clear(alpha); mpf_clear(alpha_num); mpf_clear(alpha_den);
	mpf_clear(beta); mpf_clear(beta_num); mpf_clear(beta_den);
	mpf_clear(dtmp); mpf_clear(dtmp1); mpf_clear(init_resnorm);

	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(MPFCG, %ld)\n", times);
		return_val = -5;
	}

	return return_val;
}
#endif // USE_GMP
