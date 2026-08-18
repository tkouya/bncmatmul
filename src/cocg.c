/********************************************************************************/
/* cocg.c:                                                                      */
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
#endif // _STDIO_H

/* math.h */
#ifndef _MATH_H
#include <math.h>
#endif // _MATH_H

//#include "bnc.h"

//#ifdef USE_SPARSE_VERSION
/* Sparse Matrix */
#include "bncsparse.h"
//#include "bncomp.h"
//#endif

#ifndef USE_SPARSE_D_VERSION 
/************************************************************/
/*                                                          */
/*      Conjugate-Orthogonal Conjugate-Gradient Method      */
/*                        for Complex Symmetric Matrix      */
/*                                 (Double Precision)       */
/*                                                          */
/*                 ver. 0.0 2024-11-06 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
  long int CDCOCG_sp(CDVector answer, CDRSMatrix a, CDVector b, double reps, double aeps, long int maxtimes)
#else
  long int CDCOCG(CDVector answer, CDMatrix a, CDVector b, double reps, double aeps, long int maxtimes)
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
/*      -1 ... Deminator of Alpha is zero.                  */
/*      -2 ... Numerator of Alpha is zero.                  */
/*      -3 ... Deminator of Beta is zero.                   */
/*      -4 ... Numerator of Beta is zero.                   */
/*      -5 ... Not Converge.                                */
/*                                                          */
/************************************************************/
{
	long int i, j, times, dim, return_val; // Fix!
	double _Complex alpha, alpha_num, alpha_den;
	double _Complex beta, beta_num, beta_den;
	double dtmp, init_resnorm;
	CDVector vec[4]; /* Temporary Vectors */

	dim = answer->dim;

/* Set initial value */
	for(i = 0; i < 4; i++)
		vec[i] = init_cdvector(dim);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... p */
	/* vec[2] ... residual : b - a * vec[0] */
	/* vec[3] ... a * p */
	subst_cdvector(vec[1], b); 
	subst_cdvector(vec[2], b);

	beta_num = dotp_cdvector(vec[2], vec[2]);
	//init_resnorm = sqrt(beta_num);
    init_resnorm = norm2_cdvector(vec[2]);
	return_val = 0;

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* Ap */
#ifdef USE_SPARSE_VERSION
		mul_cdrsmatrix_cdvec(vec[3], a, vec[1]);
#else // USE_SPARSE_VERSION
		mul_cdmatrix_cdvec(vec[3], a, vec[1]);
#endif // USE_SPARSE_VERSION

		/* alpha = alpha_num / alpha_den */
		alpha_den = dotp_cdvector(vec[1], vec[3]);
		alpha_num = dotp_cdvector(vec[2], vec[2]);
		if(cabs(alpha_den) == 0.0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(CDCOCG, %ld)\n", times);
			return_val = -1; // Fix!
			break;
		}
		if(cabs(alpha_num) == 0.0)
		{
			fprintf(stderr, "Numerator of Alpha is zero!(CDCOCG, %ld)\n", times);
			return_val = -2; // Fix!
			break;
		}

		alpha = alpha_num / alpha_den;

		/* x = x + alpha p */
		/*
        for(i = 0; i < dim; i++)
		{
			dtmp = get_dvector_i(vec[0], i) + alpha * get_dvector_i(vec[1], i);
			set_dvector_i(vec[0], i, dtmp);
		}
        */
        add_cmul_cdvector(vec[0], vec[0], alpha, vec[1]);

		/* residual */
		beta_den = beta_num;
		/*
        for(i = 0; i < dim; i++)
		{
			dtmp = get_dvector_i(vec[2], i) - alpha * get_dvector_i(vec[3], i);
			set_dvector_i(vec[2], i, dtmp);
		}
        */
        sub_cmul_cdvector(vec[2], vec[2], alpha, vec[3]);
		beta_num = dotp_cdvector(vec[2], vec[2]);

		/* Stopping Criteria */
		//dtmp = sqrt(beta_num);
        dtmp = norm2_cdvector(vec[2]);
		if(dtmp < aeps + reps * init_resnorm)
		{
			subst_cdvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

		if(cabs(beta_den) == 0.0)
		{
			fprintf(stderr, "Denominator of Beta is zero!(CDCOCG, %ld)\n", times);
			return_val = -3; // Fix!
			break; // Fix!
		}
		if(cabs(beta_num) == 0.0)
		{
			fprintf(stderr, "Numerator of Beta is zero!(CDCOCG, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}

		/* beta */
		beta = beta_num / beta_den;

		/* p */
        /*
		for(i = 0; i < dim; i++)
		{
			dtmp = get_dvector_i(vec[2], i) + beta * get_dvector_i(vec[1], i);
			set_dvector_i(vec[1], i, dtmp);
		}
        */
        add_cmul_cdvector(vec[1], vec[2], beta, vec[1]);
	}

	/* Not converge */
	subst_cdvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 4; i++)
		free_cdvector(vec[i]);

	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(CDCOCG, %ld)\n", times);
		return_val = -5;
	}

	return return_val;

}
#endif // ifndef USE_SPARSE_D_VERSION

// DD CG
/************************************************************/
/*                                                          */
/*      Conjugate-Orthogonal Conjugate-Gradient Method      */
/*                        for Complex Symmetric Matrix      */
/*                                      (DD Precision)      */
/*                                                          */
/*                 ver. 0.0 2024-11-06 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
  long int CDDCOCG_sp(CDDVector answer, CDDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
#elif USE_SPARSE_D_VERSION
  long int CDDCOCG_sp_d(CDDVector answer, CDRSMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
#else // USE_SPARSE_VERSION
  long int CDDCOCG(CDDVector answer, CDDMatrix a, CDDVector b, double reps[DDSIZE], double aeps[DDSIZE], long int maxtimes)
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
/*      -1 ... Deminator of Alpha is zero.                  */
/*      -2 ... Numerator of Alpha is zero.                  */
/*      -3 ... Deminator of Beta is zero.                   */
/*      -4 ... Numerator of Beta is zero.                   */
/*      -5 ... Not Converge.                                */
/*                                                          */
/************************************************************/
{
	long int i, j, times, dim, return_val; // Fix!
	cddfloat alpha, alpha_num, alpha_den;
	cddfloat beta, beta_num, beta_den;
	double dtmp[DDSIZE], dtmp1[DDSIZE], init_resnorm[DDSIZE];
	CDDVector vec[4]; /* Temporary Vectors */

	dim = answer->re->dim;

	/* Set initial value */
	for(i = 0; i < 4; i++)
		vec[i] = init_cddvector(dim);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... p */
	/* vec[2] ... residual : b - a * vec[0] */
	/* vec[3] ... a * p */
	subst_cddvector(vec[1], b); 
	subst_cddvector(vec[2], b);

	dotp_cddvector(&beta_num, vec[2], vec[2]);
	//rdd_sqrt(init_resnorm, beta_num);
    norm2_cddvector(init_resnorm, vec[2]);
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* Ap */
#ifdef USE_SPARSE_VERSION
		mul_cddrsmatrix_cddvec(vec[3], a, vec[1]);
#elif USE_SPARSE_D_VERSION
		mul_cdrsmatrix_cddvec(vec[3], a, vec[1]);
#else // USE_SPARSE_VERSION
		mul_cddmatrix_cddvec(vec[3], a, vec[1]);
#endif // USE_SPARSE_VERSION
		/* alpha = alpha_num / alpha_den */
		dotp_cddvector(&alpha_den, vec[1], vec[3]);
		dotp_cddvector(&alpha_num, vec[2], vec[2]);

        rcdd_abs_dd(dtmp, &alpha_den);
        rcdd_abs_dd(dtmp1, &alpha_num);

		if(rdd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(CDDCOCG, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}
		if(rdd_cmp_ui(dtmp1, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Alpha is zero!(CDDCOCG, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}

		rcdd_div(&alpha, &alpha_num, &alpha_den);

		/* x = x + alpha p */
        /*
		for(i = 0; i < dim; i++)
		{
			rdd_mul(dtmp, alpha, get_cddvector_i(vec[1], i));
			rdd_add(dtmp, get_cddvector_i(vec[0], i), dtmp);
			set_cddvector_i(vec[0], i, dtmp);
		}
        */
        add_cmul_cddvector(vec[0], vec[0], &alpha, vec[1]);

		/* residual */
		rcdd_set(&beta_den, &beta_num);
        /*
		for(i = 0; i < dim; i++)
		{
			rdd_mul(dtmp, alpha, get_cddvector_i(vec[3], i));
			rdd_sub(dtmp, get_cddvector_i(vec[2], i), dtmp);
			set_cddvector_i(vec[2], i, dtmp);
		}
        */
        sub_cmul_cddvector(vec[2], vec[2], &alpha, vec[3]);

		dotp_cddvector(&beta_num, vec[2], vec[2]);

		/* Stopping Criteria */
		//rdd_sqrt(dtmp, beta_num);
        norm2_cddvector(dtmp, vec[2]);
		rdd_mul(dtmp1, reps, init_resnorm);
		rdd_add(dtmp1, dtmp1, aeps);
		if(rdd_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_cddvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

        rcdd_abs_dd(dtmp, &beta_den);
		if(rdd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Beta is zero!(CDDCOCG, %ld)\n", times);
			return_val = -3; // Fix!
			break; // Fix!
		}
        rcdd_abs_dd(dtmp1, &beta_num);
		if(rdd_cmp_ui(dtmp1, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Beta is zero!(CDDCOCG, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}

		/* beta */
		rcdd_div(&beta, &beta_num, &beta_den);

		/* p */
        /*
		for(i = 0; i < dim; i++)
		{
			rdd_mul(dtmp, beta, get_cddvector_i(vec[1], i));
			rdd_add(dtmp, get_cddvector_i(vec[2], i), dtmp);
			set_cddvector_i(vec[1], i, dtmp);
		}
        */
        add_cmul_cddvector(vec[1], vec[2], &beta, vec[1]);

	}

	/* Not converge */
	subst_cddvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 4; i++)
		free_cddvector(vec[i]);

	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(CDDCOCG, %ld)\n", times);
		return_val = -5;
	}

	return return_val;
}

// TD CG
/************************************************************/
/*                                                          */
/*      Conjugate-Orthogonal Conjugate-Gradient Method      */
/*                        for Complex Symmetric Matrix      */
/*                                     (TD Precision)       */
/*                                                          */
/*                 ver. 0.0 2024-11-06 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
  long int CTDCOCG_sp(CTDVector answer, CTDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
#elif USE_SPARSE_D_VERSION
  long int CTDCOCG_sp_d(CTDVector answer, CDRSMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
#else // USE_SPARSE_VERSION
  long int CTDCOCG(CTDVector answer, CTDMatrix a, CTDVector b, double reps[TDSIZE], double aeps[TDSIZE], long int maxtimes)
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       CTDVector answer: Solution for Ax = b               */
/*       TDMatrix a: Coefficient matrix A                   */
/*                                       (given by user)    */
/*       CTDVector b: Constant vector b   (given by user)    */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       CTDVector answer: Solution for Ax = b               */
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
	ctdfloat alpha, alpha_num, alpha_den;
	ctdfloat beta, beta_num, beta_den;
	double dtmp[TDSIZE], dtmp1[TDSIZE], init_resnorm[TDSIZE];
	CTDVector vec[4]; /* Temporary Vectors */

	dim = answer->re->dim;

	/* Set initial value */
	for(i = 0; i < 4; i++)
		vec[i] = init_ctdvector(dim);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... p */
	/* vec[2] ... residual : b - a * vec[0] */
	/* vec[3] ... a * p */
	subst_ctdvector(vec[1], b); 
	subst_ctdvector(vec[2], b);

	dotp_ctdvector(&beta_num, vec[2], vec[2]);
	//rtd_sqrt(init_resnorm, beta_num);
    norm2_ctdvector(init_resnorm, vec[2]);
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* Ap */
#ifdef USE_SPARSE_VERSION
		mul_ctdrsmatrix_ctdvec(vec[3], a, vec[1]);
#elif USE_SPARSE_D_VERSION
		mul_cdrsmatrix_ctdvec(vec[3], a, vec[1]);
#else // USE_SPARSE_VERSION
		mul_ctdmatrix_ctdvec(vec[3], a, vec[1]);
#endif // USE_SPARSE_VERSION
		/* alpha = alpha_num / alpha_den */
		dotp_ctdvector(&alpha_den, vec[1], vec[3]);
		dotp_ctdvector(&alpha_num, vec[2], vec[2]);

		rctd_abs_td(dtmp, &alpha_den);
		rctd_abs_td(dtmp1, &alpha_num);

		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(CTDCOCG, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}
		if(rtd_cmp_ui(dtmp1, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Alpha is zero!(CTDCOCG, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}

		rctd_div(&alpha, &alpha_num, &alpha_den);

		/* x = x + alpha p */
		/*
        for(i = 0; i < dim; i++)
		{
			rtd_mul(dtmp, alpha, get_ctdvector_i(vec[1], i));
			rtd_add(dtmp, get_ctdvector_i(vec[0], i), dtmp);
			set_ctdvector_i(vec[0], i, dtmp);
		}
        */
        add_cmul_ctdvector(vec[0], vec[0], &alpha, vec[1]);


		/* residual */
		rctd_set(&beta_den, &beta_num);
		/*
        for(i = 0; i < dim; i++)
		{
			rtd_mul(dtmp, alpha, get_ctdvector_i(vec[3], i));
			rtd_sub(dtmp, get_ctdvector_i(vec[2], i), dtmp);
			set_ctdvector_i(vec[2], i, dtmp);
		}
        */
        sub_cmul_ctdvector(vec[2], vec[2], &alpha, vec[3]);
		dotp_ctdvector(&beta_num, vec[2], vec[2]);

		/* Stopping Criteria */
		//rtd_sqrt(dtmp, beta_num);
        norm2_ctdvector(dtmp, vec[2]);
		rtd_mul(dtmp1, reps, init_resnorm);
		rtd_add(dtmp1, dtmp1, aeps);
		if(rtd_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_ctdvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

        rctd_abs_td(dtmp, &beta_den);
        rctd_abs_td(dtmp1, &beta_num);

		if(rtd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Beta is zero!(CTDCOCG, %ld)\n", times);
			return_val = -3; // Fix!
			break; // Fix!
		}
		if(rtd_cmp_ui(dtmp1, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Beta is zero!(CTDCOCG, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}

		/* beta */
		rctd_div(&beta, &beta_num, &beta_den);

		/* p */
		/*
        for(i = 0; i < dim; i++)
		{
			rtd_mul(dtmp, beta, get_ctdvector_i(vec[1], i));
			rtd_add(dtmp, get_ctdvector_i(vec[2], i), dtmp);
			set_ctdvector_i(vec[1], i, dtmp);
		}
        */
        add_cmul_ctdvector(vec[1], vec[2], &beta, vec[1]);
	}

	/* Not converge */
	subst_ctdvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 4; i++)
		free_ctdvector(vec[i]);

	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(CTDCOCG, %ld)\n", times);
		return_val = -5;
	}

	return return_val;
}

// QD CG
/************************************************************/
/*                                                          */
/*      Conjugate-Orthogonal Conjugate-Gradient Method      */
/*                        for Complex Symmetric Matrix      */
/*                                      (QD Precision)      */
/*                                                          */
/*                 ver. 0.0 2024-11-06 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
  long int CQDCOCG_sp(CQDVector answer, CQDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
#elif USE_SPARSE_D_VERSION
  long int CQDCOCG_sp_d(CQDVector answer, CDRSMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
#else // USE_SPARSE_VERSION
  long int CQDCOCG(CQDVector answer, CQDMatrix a, CQDVector b, double reps[QDSIZE], double aeps[QDSIZE], long int maxtimes)
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       CQDVector answer: Solution for Ax = b               */
/*       QDMatrix a: Coefficient matrix A                   */
/*                                       (given by user)    */
/*       CQDVector b: Constant vector b   (given by user)    */
/*       double reps: Relative tolerance (given by user)    */
/*       double aeps: Absolute tolerance (given by user)    */
/*       long int maxtimes: Maximum iterative times         */
/*                                       (given by user)    */
/*                                                          */
/* RETURNS                                                  */
/*       CQDVector answer: Solution for Ax = b               */
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
	cqdfloat alpha, alpha_num, alpha_den;
	cqdfloat beta, beta_num, beta_den;
	double dtmp[QDSIZE], dtmp1[QDSIZE], init_resnorm[QDSIZE];
	CQDVector vec[4]; /* Temporary Vectors */

	dim = answer->re->dim;

	/* Set initial value */
	for(i = 0; i < 4; i++)
		vec[i] = init_cqdvector(dim);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... p */
	/* vec[2] ... residual : b - a * vec[0] */
	/* vec[3] ... a * p */
	subst_cqdvector(vec[1], b); 
	subst_cqdvector(vec[2], b);

	dotp_cqdvector(&beta_num, vec[2], vec[2]);
    norm2_cqdvector(init_resnorm, vec[2]);
	//rqd_sqrt(init_resnorm, beta_num);
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* Ap */
#ifdef USE_SPARSE_VERSION
		mul_cqdrsmatrix_cqdvec(vec[3], a, vec[1]);
#elif USE_SPARSE_D_VERSION
		mul_cdrsmatrix_cqdvec(vec[3], a, vec[1]);
#else // USE_SPARSE_VERSION
		mul_cqdmatrix_cqdvec(vec[3], a, vec[1]);
#endif // USE_SPARSE_VERSION
		/* alpha = alpha_num / alpha_den */
		dotp_cqdvector(&alpha_den, vec[1], vec[3]);
		dotp_cqdvector(&alpha_num, vec[2], vec[2]);

        rcqd_abs_qd(dtmp, &alpha_den);
        rcqd_abs_qd(dtmp1, &alpha_num);

		if(rqd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(CQDCOCG, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}
		if(rqd_cmp_ui(dtmp1, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Alpha is zero!(CQDCOCG, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}

		rcqd_div(&alpha, &alpha_num, &alpha_den);

		/* x = x + alpha p */
        /*
		for(i = 0; i < dim; i++)
		{
			rqd_mul(dtmp, alpha, get_cqdvector_i(vec[1], i));
			rqd_add(dtmp, get_cqdvector_i(vec[0], i), dtmp);
			set_cqdvector_i(vec[0], i, dtmp);
		}
        */
        add_cmul_cqdvector(vec[0], vec[0], &alpha, vec[1]);

		/* residual */
		rcqd_set(&beta_den, &beta_num);
        /*
		for(i = 0; i < dim; i++)
		{
			rqd_mul(dtmp, alpha, get_cqdvector_i(vec[3], i));
			rqd_sub(dtmp, get_cqdvector_i(vec[2], i), dtmp);
			set_cqdvector_i(vec[2], i, dtmp);
		}
        */
        sub_cmul_cqdvector(vec[2], vec[2], &alpha, vec[3]);
		dotp_cqdvector(&beta_num, vec[2], vec[2]);

		/* Stopping Criteria */
		//rqd_sqrt(dtmp, beta_num);
        norm2_cqdvector(dtmp, vec[2]);
		rqd_mul(dtmp1, reps, init_resnorm);
		rqd_add(dtmp1, dtmp1, aeps);
		if(rqd_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_cqdvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

        rcqd_abs_qd(dtmp, &beta_den);
        rcqd_abs_qd(dtmp1, &beta_num);

		if(rqd_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Beta is zero!(CQDCOCG, %ld)\n", times);
			return_val = -3; // Fix!
			break; // Fix!
		}
		if(rqd_cmp_ui(dtmp1, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Beta is zero!(CQDCOCG, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}

		/* beta */
		rcqd_div(&beta, &beta_num, &beta_den);

		/* p */
        /*
		for(i = 0; i < dim; i++)
		{
			rqd_mul(dtmp, beta, get_cqdvector_i(vec[1], i));
			rqd_add(dtmp, get_cqdvector_i(vec[2], i), dtmp);
			set_cqdvector_i(vec[1], i, dtmp);
		}
        */
        add_cmul_cqdvector(vec[1], vec[2], &beta, vec[1]);
	}

	/* Not converge */
	subst_cqdvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 4; i++)
		free_cqdvector(vec[i]);

	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(CQDCOCG, %ld)\n", times);
		return_val = -5;
	}

	return return_val;
}


#ifdef USE_GMP
/************************************************************/
/*                                                          */
/*      Conjugate-Orthogonal Conjugate-Gradient Method      */
/*                        for Complex Symmetric Matrix      */
/*                               (Arbitrary Precision)      */
/*                                                          */
/*                 ver. 0.0 2024-11-06 (Wed) Tomonori Kouya */
/*                                                          */
/************************************************************/
#ifdef USE_SPARSE_VERSION
  long int CMPFCOCG_sp(CMPFVector answer, CMPFRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
#elif USE_SPARSE_D_VERSION
  long int CMPFCOCG_sp_d(CMPFVector answer, CDRSMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
#else // USE_SPARSE_VERSION
  long int CMPFCOCG(CMPFVector answer, CMPFMatrix a, CMPFVector b, mpf_t reps, mpf_t aeps, long int maxtimes)
#endif // USE_SPARSE_VERSION
/************************************************************/
/*                                                          */
/* ENTRIES                                                  */
/*       CMPFVector answer: Solution for Ax = b             */
/*       CMPFMatrix a: Coefficient matrix A                 */
/*                                       (given by user)    */
/*       CMPFVector b: Constant vector b  (given by user)   */
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
/*      -3 ... Deminator of Beta is zero.                   */
/*      -4 ... Numerator of Beta is zero.                   */
/*      -5 ... Not Converge.                                */
/*                                                          */
/************************************************************/
{
	unsigned long int prec;
	long int i, j, times, dim, return_val; // Fix!
	mpc_t alpha, alpha_num, alpha_den;
	mpc_t beta, beta_num, beta_den;
	mpf_t dtmp, dtmp1, init_resnorm;
	CMPFVector vec[4]; /* Temporary Vectors */

	dim = answer->dim;
	prec = answer->prec;

/* Set initial value */
	mpc_init2(alpha, prec);
	mpc_init2(alpha_num, prec);
	mpc_init2(alpha_den, prec);
	mpc_init2(beta, prec);
	mpc_init2(beta_num, prec);
	mpc_init2(beta_den, prec);
	mpf_init2(dtmp, prec);
	mpf_init2(dtmp1, prec);
	mpf_init2(init_resnorm, prec);

	for(i = 0; i < 4; i++)
		vec[i] = init2_cmpfvector(dim, prec);

	/* vec[0] ... approximation of solution */
	/* vec[1] ... p */
	/* vec[2] ... residual : b - a * vec[0] */
	/* vec[3] ... a * p */
	subst_cmpfvector(vec[1], b); 
	subst_cmpfvector(vec[2], b);

	dotp_cmpfvector(beta_num, vec[2], vec[2]);
	//mpf_sqrt(init_resnorm, beta_num);
    norm2_cmpfvector(init_resnorm, vec[2]);
	return_val = 0; // Fix!

/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* Ap */
#ifdef USE_SPARSE_VERSION
		mul_cmpfrsmatrix_cmpfvec(vec[3], a, vec[1]);
#elif USE_SPARSE_D_VERSION
		mul_cdrsmatrix_cmpfvec(vec[3], a, vec[1]);
#else // USE_SPARSE_VERSION
		mul_cmpfmatrix_cmpfvec(vec[3], a, vec[1]);
#endif // USE_SPARSE_VERSION
		/* alpha = alpha_num / alpha_den */
		dotp_cmpfvector(alpha_den, vec[1], vec[3]);
		dotp_cmpfvector(alpha_num, vec[2], vec[2]);

        mpc_abs(dtmp, alpha_den, MPC_RNDNN);
		if(mpf_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Alpha is zero!(CMPFCOCG, %ld)\n", times);
			return_val = -1; // Fix!
			break; // Fix!
		}

        mpc_abs(dtmp1, alpha_num, MPC_RNDNN);
		if(mpf_cmp_ui(dtmp1, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Alpha is zero!(CMPFCOCG, %ld)\n", times);
			return_val = -2; // Fix!
			break; // Fix!
		}

		mpc_div(alpha, alpha_num, alpha_den, MPC_RNDNN);

		/* x = x + alpha p */
        /*
		for(i = 0; i < dim; i++)
		{
			mpf_mul(dtmp, alpha, get_mpfvector_i(vec[1], i));
			mpf_add(dtmp, get_mpfvector_i(vec[0], i), dtmp);
			set_mpfvector_i(vec[0], i, dtmp);
		}
        */
        add_cmul_cmpfvector(vec[0], vec[0], alpha, vec[1]);

		/* residual */
		mpc_set(beta_den, beta_num, MPC_RNDNN);
		/*
        for(i = 0; i < dim; i++)
		{
			mpf_mul(dtmp, alpha, get_mpfvector_i(vec[3], i));
			mpf_sub(dtmp, get_mpfvector_i(vec[2], i), dtmp);
			set_mpfvector_i(vec[2], i, dtmp);
		}
        */
        sub_cmul_cmpfvector(vec[2], vec[2], alpha, vec[3]);
		dotp_cmpfvector(beta_num, vec[2], vec[2]);

		/* Stopping Criteria */
		//mpf_sqrt(dtmp, beta_num);
        norm2_cmpfvector(dtmp, vec[2]);
		mpf_mul(dtmp1, reps, init_resnorm);
		mpf_add(dtmp1, dtmp1, aeps);
		if(mpf_cmp(dtmp, dtmp1) < 0) /* dtmp < dtmp1 */
		{
			subst_cmpfvector(answer, vec[0]);
			return_val = times; // Fix!
			break; // Fix!
		}

        mpc_abs(dtmp, beta_den, MPC_RNDNN);
		if(mpf_cmp_ui(dtmp, 0UL) == 0)
		{
			fprintf(stderr, "Denominator of Beta is zero!(CMPFCOCG, %ld)\n", times);
			return_val = -3; // Fix!
			break; // Fix!
		}

        mpc_abs(dtmp1, beta_num, MPC_RNDNN);
		if(mpf_cmp_ui(dtmp1, 0UL) == 0)
		{
			fprintf(stderr, "Numerator of Beta is zero!(CMPFCOCG, %ld)\n", times);
			return_val = -4; // Fix!
			break; // Fix!
		}

		/* beta */
		mpc_div(beta, beta_num, beta_den, MPC_RNDNN);

		/* p */
        /*
		for(i = 0; i < dim; i++)
		{
			mpf_mul(dtmp, beta, get_mpfvector_i(vec[1], i));
			mpf_add(dtmp, get_mpfvector_i(vec[2], i), dtmp);
			set_mpfvector_i(vec[1], i, dtmp);
		}
        */
        add_cmul_cmpfvector(vec[1], vec[2], beta, vec[1]);
	}

	/* Not converge */
	subst_cmpfvector(answer, vec[0]);

	/* free vec[0]..[3]; */
	for(i = 0; i < 4; i++)
		free_cmpfvector(vec[i]);
	mpc_clear(alpha); mpc_clear(alpha_num); mpc_clear(alpha_den);
	mpc_clear(beta); mpc_clear(beta_num); mpc_clear(beta_den);
	mpf_clear(dtmp); mpf_clear(dtmp1); mpf_clear(init_resnorm);

	if(times >= maxtimes)
	{
		fprintf(stderr, "Not converge!(CMPFCOCG, %ld)\n", times);
		return_val = -5;
	}

	return return_val;
}
#endif // USE_GMP
