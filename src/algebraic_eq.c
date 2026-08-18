/********************************************************************************/
/* Algebraic Solvers for 2nd Degree Algebraic Equations                         */
/*                                                                              */
/* Copyright (C) 2007-2011 Tomonori Kouya                                       */
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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//#include "bnc.h"
#include "bncmatmul.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/***********************************************/
/* Solver of Quadratic Equation                */
/***********************************************/

/* double */

/* coef[2] x^2 + coef[1] x + coef[0] = 0 */
/* Output: ans_re[0] + sqrt(-1)*ans_im[0], ans_re[1] + sqrt(-1) * ans_im[1] */
int dquadratic_eq(double ans_re[2], double ans_im[2], double coef[3])
{
	double d, den, tmp;

//	printf("coef: %10.3e x^2 + %10.3e x + %10.3e = 0\n", coef[2], coef[1], coef[0]);

	ans_re[0] = 0.0;
	ans_re[1] = 0.0;
	ans_im[0] = 0.0;
	ans_im[1] = 0.0;

	if(coef[2] == 0.0)
	{
		if(coef[1] == 0.0)
		{
			fprintf(stderr, "ERROR: No answers (dquadraric_eq)\n");
			return -1;
		}
		ans_re[0] = -coef[0] / coef[1];
		return 1; // number of answers
	}

	den = 2.0 * coef[2];
	d = coef[1] * coef[1] - 4.0 * coef[2] * coef[0];
	ans_re[0] = -coef[1] / den;

	/* anss are real numbers */
	if(d == 0.0)
	{
		ans_re[1] = ans_re[0];
	}
	else if(d > 0)
	{
		tmp = sqrt(d);
		if(coef[1] > 0.0)
			tmp = -tmp / den;
		else
			tmp = tmp / den;
		ans_re[0] -= tmp;
		ans_re[1] = coef[0] / (coef[2] * ans_re[0]);
	}
	/* complex numbers */
	else
	{
		tmp = sqrt(-d) / den;
		ans_re[1] = ans_re[0];
		ans_im[0] = tmp;
		ans_im[1] = -tmp;
	}
	return 2;
}

/***********************************************/
/* Cubic Equations                             */
/***********************************************/
/* Input : coef[3] * x^3 + coef[2] * x^2 + coef[1] * x + coef[0] = 0 */
/* Output: ans_re[i] + sqrt(-1)*ans_im[i] (i = 0, 1, 2) */
int dcubic_eq(double ans_re[3], double ans_im[3], double coef[4])
{
	double e, p, q, u, v, o;
	double quad_coef[3], quad_ans_re[2], quad_ans_im[2];
	double coef32, coef22, p3;

	/* check coefficients */
	if(coef[3] == 0.0)
	{
		fprintf(stderr, "Warning: not cubic equation!(dcubic_eq)\n");

		/* solve coef[2] * x^2 + coef[1] * x + coef[0] = 0 */
		quad_coef[2] = coef[2];
		quad_coef[1] = coef[1];
		quad_coef[0] = coef[0];
		ans_re[2] = 0.0;
		ans_im[2] = 0.0;
		dquadratic_eq(ans_re, ans_im, quad_coef);

		return 2;
	}

	/* set p, q */
	coef32 = coef[3] * coef[3];
	coef22 = coef[2] * coef[2];
	p = -(coef22) / (9.0 * coef32) + coef[1] / (3.0 * coef[3]);
	q = (2.0 * coef[2] * coef22) / (27.0 * coef[3] * coef32) - (coef[2] * coef[1]) / (3 * coef32) + coef[0] / coef[3];

	/* solve x^2 + q *x - p^3 = 0 */
	quad_coef[2] = 1.0;
	quad_coef[1] = q;
	quad_coef[0] = -(p * p * p);

	dquadratic_eq(quad_ans_re, quad_ans_im, quad_coef);

	/* set e */
	e = (q * q) + 4 * (p * p * p);

	/* discriminant */
	if(e >= 0.0)
	{
		u = cbrt(quad_ans_re[0]);
		v = cbrt(quad_ans_re[1]);

		ans_re[0] = u + v - coef[2] / (3.0 * coef[3]);
		ans_im[0] = 0.0;
		ans_re[1] = -0.5 * (u + v) - coef[2] / (3.0 * coef[3]);
		ans_im[1] = -sqrt(3.0) / 2.0 * (u - v);
		ans_re[2] = ans_re[1];
		ans_im[2] = -ans_im[1];
	}
	else
	{
		o = atan(sqrt(-e) / q);
		ans_re[0] = 2.0 * sqrt(-p) * cos(o / 3.0) - coef[2] / (3.0 * coef[3]);
		ans_im[0] = 0.0;
		ans_re[1] = 2.0 * sqrt(-p) * cos((2.0 * M_PI + o) / 3.0) - coef[2] / (3.0 * coef[3]);
		ans_im[1] = 0.0;
		ans_re[2] = 2.0 * sqrt(-p) * cos((4.0 * M_PI + o) / 3.0) - coef[2] / (3.0 * coef[3]);
		ans_im[2] = 0.0;
	}

	return 3;
}

/* MPF */

#ifdef USE_GMP
/* coef[2] x^2 + coef[1] x + coef[0] = 0 */
/* Output: ans_re[0] + sqrt(-1)*ans_im[0], ans_re[1] + sqrt(-1) * ans_im[1] */
int mpf_quadratic_eq(mpf_t ans_re[2], mpf_t ans_im[2], mpf_t coef[3])
{
	unsigned prec;
	mpf_t d, den, tmp;

//	printf("coef: %10.3e x^2 + %10.3e x + %10.3e = 0\n", coef[2], coef[1], coef[0]);

	prec = (mpf_get_prec(ans_re[0]) > mpf_get_prec(ans_re[1])) ? mpf_get_prec(ans_re[0]) : mpf_get_prec(ans_re[1]);

	mpf_set_ui(ans_re[0], 0UL);
	mpf_set_ui(ans_re[1], 0UL);
	mpf_set_ui(ans_im[0], 0UL);
	mpf_set_ui(ans_im[1], 0UL);

	if(mpf_cmp_ui(coef[2], 0UL) == 0)
	{
		if(mpf_cmp_ui(coef[1], 0UL) == 0)
		{
			fprintf(stderr, "ERROR: No answers (mpfquadraric_eq)\n");
			return -1;
		}
		mpf_neg(ans_re[0], coef[0]);
		mpf_div(ans_re[0], ans_re[0], coef[1]);
		return 1; // number of answers
	}

	/* Initialize */
	mpf_init2(d, prec);
	mpf_init2(den, prec);
	mpf_init2(tmp, prec);

	mpf_mul_ui(den, coef[2], 2UL);
	/* d = coef[1] * coef[1] - 4.0 * coef[2] * coef[0]; */
	mpf_mul(d, coef[1], coef[1]);
	mpf_mul(tmp, coef[2], coef[0]);
	mpf_mul_ui(tmp, tmp, 4UL);
	mpf_sub(d, d, tmp);

	/* ans_re[0] = -coef[1] / den; */
	mpf_neg(ans_re[0], coef[1]);
	mpf_div(ans_re[0], ans_re[0], den);

	/* anss are real numbers */
	if(mpf_cmp_ui(d, 0UL) == 0)
	{
		mpf_set(ans_re[1], ans_re[0]);
	}
	else if(mpf_cmp_ui(d, 0UL) > 0)
	{
		mpf_sqrt(tmp, d);
		if(mpf_cmp_ui(coef[1], 0UL) > 0)
		{
			mpf_neg(tmp, tmp);
			mpf_div(tmp, tmp, den);
		}
		else
			mpf_div(tmp, tmp, den);
		mpf_sub(ans_re[0], ans_re[0], tmp);
		mpf_mul(tmp, coef[2], ans_re[0]);
		mpf_div(ans_re[1], coef[0], tmp);
	}
	/* complex numbers */
	else
	{
		/* tmp = sqrt(-d) / den; */
		mpf_neg(tmp, d);
		mpf_sqrt(tmp, tmp);
		mpf_div(tmp, tmp, den);

		mpf_set(ans_re[1], ans_re[0]);
		mpf_set(ans_im[0], tmp);
		mpf_neg(ans_im[1], tmp);
	}

	/* free */
	mpf_clear(d);
	mpf_clear(den);
	mpf_clear(tmp);

	return 2;
}

/* coef[2] x^2 + coef[1] x + coef[0] = 0 */
/* Output: ans[0], ans[1] */
int mpc_quadratic_eq(mpc_t ans[2], mpc_t coef[3])
{
	unsigned long prec;
	mpc_t d, den, tmp;
    mpc_rnd_t rndc = get_bnc_default_rounding_mode_c();

//	printf("coef: %10.3e x^2 + %10.3e x + %10.3e = 0\n", coef[2], coef[1], coef[0]);

	prec = (mpf_get_prec(mpc_realref(ans[0])) > mpf_get_prec(mpc_realref(ans[1]))) ? mpf_get_prec(mpc_imagref(ans[0])) : mpf_get_prec(mpc_imagref(ans[1]));

	mpc_set_ui_ui(ans[0], 0UL, 0UL, rndc);
	mpc_set_ui_ui(ans[1], 0UL, 0UL, rndc);

	if(mpc_abscmp_ui(coef[2], 0UL) == 0)
	{
		if(mpc_abscmp_ui(coef[1], 0UL) == 0)
		{
			fprintf(stderr, "ERROR: No answers (mpc_quadraric_eq)\n");
			return -1;
		}
		mpc_neg(ans[0], coef[0], rndc);
		mpc_div(ans[0], ans[0], coef[1], rndc);
		return 1; // number of answers
	}

	/* Initialize */
	mpc_init2(d, prec);
	mpc_init2(den, prec);
	mpc_init2(tmp, prec);

    // den = 2 * a
	mpc_mul_ui(den, coef[2], 2UL, rndc);

    /* d = coef[1] * coef[1] - 4.0 * coef[2] * coef[0]; */
	mpc_mul(d, coef[1], coef[1], rndc);
	mpc_mul(tmp, coef[2], coef[0], rndc);
	mpc_mul_ui(tmp, tmp, 4UL, rndc);
    mpc_sub(d, d, tmp, rndc);
    // d = sqrt(d)
    mpc_sqrt(d, d, rndc);

    // ans[0] = (-coef[1] + sqrt(d)) / (2 * coef[2])
    // ans[1] = (-coef[1] - sqrt(d)) / (2 * coef[2])
    mpc_neg(ans[0], coef[1], rndc);
    mpc_neg(ans[1], coef[1], rndc);
    mpc_add(ans[0], ans[0], d, rndc);
    mpc_sub(ans[1], ans[1], d, rndc);
    //mpc_mul_ui(den, coef[2], 2UL, rndc);
    mpc_div(ans[0], ans[0], den, rndc);
    mpc_div(ans[1], ans[1], den, rndc); 

	/* free */
	mpc_clear(d);
	mpc_clear(den);
	mpc_clear(tmp);

	return 2;
}

#endif // USE_GMP


#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus
/***********************************************/
/* End of algebraic_eq.c                       */
/***********************************************/

