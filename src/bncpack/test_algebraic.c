/********************************************************************************/
/* Test Program of Algebraic solvers                                            */
/* Copyright (C) 2011 Tomonori Kouya                                            */
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

//#define USE_GMP
//#define USE_MPFR
#include "bnc.h"

main()
{
	long int i;
	double dcoef[3]; // dcoef[2] * x^2 + dcoef[1] * x + dcoef[0] = 0.0;
	double dans_re[2], dans_im[2]; // ans = dans_re[0 or 1] + dans_im[0 or 1] * sqrt(-1)
	double dcoef3[4];
	double dans3_re[3], dans3_im[3];
	DCmplx deval, dans;
	DPoly dpoly_eq;
#ifdef USE_GMP
	mpf_t mpf_tmp;
	mpf_t mpf_coef[3]; // mpf_coef[2] * x^2 + mpf_coef[1] * x + mpf_coef[0] = 0.0;
	mpf_t mpf_ans_re[2], mpf_ans_im[2]; // ans = mpf_ans_re[0 or 1] + mpf_ans_im[0 or 1] * sqrt(-1)
	MPFCmplx mpf_eval, mpf_ans;
	MPFPoly mpfpoly_eq;
#endif

	/* quadratic equation */
	dcoef[2] = exp(1.0);
	dcoef[1] = -M_PI;
	dcoef[0] = 4.0 * log10(2.0);

	// solve quadratic equation
	dquadratic_eq(dans_re, dans_im, dcoef);

	/* evaluate p(ans[i]) */
	dpoly_eq = init_dpoly(3);
	dans = init_dcmplx();
	deval = init_dcmplx();
	set_dpoly_i(dpoly_eq, 2, dcoef[2]);
	set_dpoly_i(dpoly_eq, 1, dcoef[1]);
	set_dpoly_i(dpoly_eq, 0, dcoef[0]);

	printf("Quadracic eq: \n");
	for(i = 0; i < 2; i++)
	{
		set_real_dcmplx(dans, dans_re[i]); set_image_dcmplx(dans, dans_im[i]);
		ceval_dpoly(deval, dpoly_eq, dans);
		printf("ans[%d] = %25.17e, %25.17e\np(ans[%d]) = (%8.1e,%8.1e)\n", i, dans_re[i], dans_im[i], i, get_real_dcmplx(deval), get_image_dcmplx(deval));
	}

	free_dpoly(dpoly_eq);
	free_dcmplx(dans);
	free_dcmplx(deval);

	/* cubic equation */
	dcoef3[3] = sqrt(2.0);
	dcoef3[2] = exp(1.0);
	dcoef3[1] = -M_PI;
	dcoef3[0] = 4.0 * log10(2.0);

	// solve quadratic equation
	dcubic_eq(dans3_re, dans3_im, dcoef3);

	/* evaluate p(ans[i]) */
	dpoly_eq = init_dpoly(4);
	dans = init_dcmplx();
	deval = init_dcmplx();
	set_dpoly_i(dpoly_eq, 3, dcoef3[3]);
	set_dpoly_i(dpoly_eq, 2, dcoef3[2]);
	set_dpoly_i(dpoly_eq, 1, dcoef3[1]);
	set_dpoly_i(dpoly_eq, 0, dcoef3[0]);

	printf("\nCubic eq: \n");
	for(i = 0; i < 3; i++)
	{
		set_real_dcmplx(dans, dans3_re[i]); set_image_dcmplx(dans, dans3_im[i]);
		ceval_dpoly(deval, dpoly_eq, dans);
		printf("ans[%d] = %25.17e, %25.17e\np(ans[%d]) = (%8.1e,%8.1e)\n", i, dans3_re[i], dans3_im[i], i, get_real_dcmplx(deval), get_image_dcmplx(deval));
	}

	free_dpoly(dpoly_eq);
	free_dcmplx(dans);
	free_dcmplx(deval);

#ifdef USE_GMP
#define DPREC 50

	set_bnc_default_prec_decimal(DPREC);

	mpf_init(mpf_tmp);
	mpf_init(mpf_coef[2]);
	mpf_init(mpf_coef[1]);
	mpf_init(mpf_coef[0]);
	mpf_init(mpf_ans_re[0]); mpf_init(mpf_ans_im[0]);
	mpf_init(mpf_ans_re[1]); mpf_init(mpf_ans_im[1]);

	// dcoef[2] = exp(1.0);
	mpf_set_ui(mpf_tmp, 1UL);mpf_exp(mpf_coef[2], mpf_tmp);
	//dcoef[1] = -M_PI;
	mpf_pi(mpf_coef[1]); mpf_neg(mpf_coef[1], mpf_coef[1]);
	//dcoef[0] = 4.0 * log10(2.0);
	mpf_set_ui(mpf_tmp, 2UL);
	mpf_log10(mpf_coef[0], mpf_tmp); mpf_mul_ui(mpf_coef[0], mpf_coef[0], 4UL);

	// solve quadratic equation
	mpfquadratic_eq(mpf_ans_re, mpf_ans_im, mpf_coef);

	/* evaluate p(ans[i]) */
	mpf_ans = init_mpfcmplx();
	mpf_eval = init_mpfcmplx();
	mpfpoly_eq = init_mpfpoly(3);
	set_mpfpoly_i(mpfpoly_eq, 2, mpf_coef[2]);
	set_mpfpoly_i(mpfpoly_eq, 1, mpf_coef[1]);
	set_mpfpoly_i(mpfpoly_eq, 0, mpf_coef[0]);

	printf("Quadracic eq: \n");
	for(i = 0; i < 2; i++)
	{
		set_real_mpfcmplx(mpf_ans, mpf_ans_re[i]);
		set_image_mpfcmplx(mpf_ans, mpf_ans_im[i]);
		ceval_mpfpoly(mpf_eval, mpfpoly_eq, mpf_ans);
		printf("ans[%d] =", i); mpf_out_str(stdout, 10, 0, mpf_ans_re[i]);
		printf(", "); mpf_out_str(stdout, 10, 0, mpf_ans_im[i]); printf("\n");
		printf("p(ans[%d]) = (", i); get_real_mpfcmplx(mpf_tmp, mpf_eval); mpf_out_str(stdout, 10, 1, mpf_tmp);
		printf(", "); get_image_mpfcmplx(mpf_tmp, mpf_eval); mpf_out_str(stdout, 10, 1, mpf_tmp); printf(")\n");
	}

	free(mpf_ans);
	free(mpf_eval);
	free(mpfpoly_eq);

	mpf_clear(mpf_tmp);
	mpf_clear(mpf_coef[2]);
	mpf_clear(mpf_coef[1]);
	mpf_clear(mpf_coef[0]);
	mpf_clear(mpf_ans_re[0]); mpf_clear(mpf_ans_im[0]);
	mpf_clear(mpf_ans_re[1]); mpf_clear(mpf_ans_im[1]);
#endif
}
