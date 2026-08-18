/********************************************************************************/
/* chebychev.c:                                                                 */
/* Copyright (C) 2003 Tomonori Kouya                                            */
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
#include <math.h>
#include "bnc.h"

/* deg <= 12 */
/* copied from Abramowitz & Stegun */
void _dget_chev_poly(DPoly poly, long int deg)
{
/* T0(x) = 1 */
/* T1(x) = x */
/* T2(x) = 2x^2 - 1 */
/* T3(x) = 4x^3 - 3x^2 */
/* T4(x) = 8x^4 - 8x^2 + 1 */

	switch(deg)
	{
		case 0:
			set_dpoly_i(poly, 0, 1.0);
			break;
		case 1:
			set_dpoly_i(poly, 0, 0.0);
			set_dpoly_i(poly, 1, 1.0);
			break;
		case 2:
			set_dpoly_i(poly, 0, -1.0);
			set_dpoly_i(poly, 1, 0.0);
			set_dpoly_i(poly, 2, 2.0);
			break;
		case 3:
			set_dpoly_i(poly, 0, 0.0);
			set_dpoly_i(poly, 1, -3.0);
			set_dpoly_i(poly, 2, 0.0);
			set_dpoly_i(poly, 3, 4.0);
			break;
		case 4:
			set_dpoly_i(poly, 0, 1.0);
			set_dpoly_i(poly, 1, 0.0);
			set_dpoly_i(poly, 2, -8.0);
			set_dpoly_i(poly, 3, 0.0);
			set_dpoly_i(poly, 4, 8.0);
			break;
		case 5:
			set_dpoly_i(poly, 0, 0.0);
			set_dpoly_i(poly, 1, 5.0);
			set_dpoly_i(poly, 2, 0.0);
			set_dpoly_i(poly, 3, -20.0);
			set_dpoly_i(poly, 4, 0.0);
			set_dpoly_i(poly, 5, 16.0);
			break;
		case 6:
			set_dpoly_i(poly, 0, -1.0);
			set_dpoly_i(poly, 1, 0.0);
			set_dpoly_i(poly, 2, 18.0);
			set_dpoly_i(poly, 3, 0.0);
			set_dpoly_i(poly, 4, -48.0);
			set_dpoly_i(poly, 5, 0.0);
			set_dpoly_i(poly, 6, 32.0);
			break;
		case 7:
			set_dpoly_i(poly, 0, 0.0);
			set_dpoly_i(poly, 1, -7.0);
			set_dpoly_i(poly, 2, 0.0);
			set_dpoly_i(poly, 3, 56.0);
			set_dpoly_i(poly, 4, 0.0);
			set_dpoly_i(poly, 5, -112.0);
			set_dpoly_i(poly, 6, 0.0);
			set_dpoly_i(poly, 7, 64.0);
			break;
		case 8:
			set_dpoly_i(poly, 0, 1.0);
			set_dpoly_i(poly, 1, 0.0);
			set_dpoly_i(poly, 2, -32.0);
			set_dpoly_i(poly, 3, 0.0);
			set_dpoly_i(poly, 4, 160.0);
			set_dpoly_i(poly, 5, 0.0);
			set_dpoly_i(poly, 6, -256.0);
			set_dpoly_i(poly, 7, 0.0);
			set_dpoly_i(poly, 8, 128.0);
			break;
		case 9:
			set_dpoly_i(poly, 0, 0.0);
			set_dpoly_i(poly, 1, 9.0);
			set_dpoly_i(poly, 2, 0.0);
			set_dpoly_i(poly, 3, -120.0);
			set_dpoly_i(poly, 4, 0.0);
			set_dpoly_i(poly, 5, 432.0);
			set_dpoly_i(poly, 6, 0.0);
			set_dpoly_i(poly, 7, -576.0);
			set_dpoly_i(poly, 8, 0.0);
			set_dpoly_i(poly, 9, 256.0);
			break;
		case 10:
			set_dpoly_i(poly, 0, -1.0);
			set_dpoly_i(poly, 1, 0.0);
			set_dpoly_i(poly, 2, 50.0);
			set_dpoly_i(poly, 3, 0.0);
			set_dpoly_i(poly, 4, -400.0);
			set_dpoly_i(poly, 5, 0.0);
			set_dpoly_i(poly, 6, 1120.0);
			set_dpoly_i(poly, 7, 0.0);
			set_dpoly_i(poly, 8, -1280.0);
			set_dpoly_i(poly, 9, 0.0);
			set_dpoly_i(poly, 10, 512.0);
			break;
		case 11:
			set_dpoly_i(poly, 0, 0.0);
			set_dpoly_i(poly, 1, -11.0);
			set_dpoly_i(poly, 2, 0.0);
			set_dpoly_i(poly, 3, 220.0);
			set_dpoly_i(poly, 4, 0.0);
			set_dpoly_i(poly, 5, -1232.0);
			set_dpoly_i(poly, 6, 0.0);
			set_dpoly_i(poly, 7, 2816.0);
			set_dpoly_i(poly, 8, 0.0);
			set_dpoly_i(poly, 9, -2816.0);
			set_dpoly_i(poly, 10, 0.0);
			set_dpoly_i(poly, 11, 1024.0);
			break;
		case 12:
			set_dpoly_i(poly, 0, 1.0);
			set_dpoly_i(poly, 1, 0.0);
			set_dpoly_i(poly, 2, -72.0);
			set_dpoly_i(poly, 3, 0.0);
			set_dpoly_i(poly, 4, 840.0);
			set_dpoly_i(poly, 5, 0.0);
			set_dpoly_i(poly, 6, -3584.0);
			set_dpoly_i(poly, 7, 0.0);
			set_dpoly_i(poly, 8, 6912.0);
			set_dpoly_i(poly, 9, 0.0);
			set_dpoly_i(poly, 10, -6144.0);
			set_dpoly_i(poly, 11, 0.0);
			set_dpoly_i(poly, 12, 2048.0);
			break;
	}
	
	return;
}

void dget_chev_poly(DPoly poly, long int deg)
{
	long int i, j;
	DPoly tmp_poly1, tmp_poly2;
	double tmp;

	if(deg <= 12)
	{
		_dget_chev_poly(poly, deg);
		return;
	}
	if(deg >= 512)
	{
		fprintf(stderr, "ERROR: Too big degree!(dget_chev_poly)\n");
		return;
	}

	/* deg > 12 */
	tmp_poly1 = init_dpoly(deg + 1);
	tmp_poly2 = init_dpoly(deg + 1);
	_dget_chev_poly(tmp_poly2, 12);
	_dget_chev_poly(tmp_poly1, 11);

	/* Tn+1(x) = 2x * Tn(x) - Tn-1(x) */
	for(i = 13; i <= deg; i++)
	{
		set_dpoly_i(poly, 0, 0.0);
		for(j = 0; j < i; j++)
			set_dpoly_i(poly, j + 1, 2.0 * get_dpoly_i(tmp_poly2, j));
		for(j = 0; j < i - 1; j++)
		{
			tmp = get_dpoly_i(poly, j);
			tmp -= get_dpoly_i(tmp_poly1, j);
			set_dpoly_i(poly, j, tmp);
		}
		subst_dpoly(tmp_poly1, tmp_poly2); tmp_poly1->deg = tmp_poly2->deg;
		subst_dpoly(tmp_poly2, poly); tmp_poly2->deg = poly->deg;

	}

	free_dpoly(tmp_poly1);
	free_dpoly(tmp_poly2);
}

void dget_chev_zeros(DVector zeros, long int deg)
{
	long int i;

	for(i = 0; i < deg; i++)
		/* zeros[i] = cos((2 * i + 1) * M_PI / (2 * deg)); */
		sdvi(zeros, i, cos((2 * (deg - 1 - i) + 1) * M_PI / (2 * deg)));

}

void dget_chev_zeros_g(DVector zeros, double interval_left, double interval_right, long int deg)
{
	long int i;
	double c1, c2;

	dget_chev_zeros(zeros, deg);
	c1 = (interval_left + interval_right) / 2.0;
	c2 = (interval_left - interval_right) / 2.0;
	for(i = 0; i < deg; i++)
		sdvi(zeros, i, c1 - c2 * gdvi(zeros, i));
}


void dget_chev_approx_g(DPoly chev_poly, double interval_left, double interval_right, double (*f)(double))
{
	long int deg, i, j;
	DVector vec_tmp, zeros;
	DMatrix mat_tmp;

	deg = chev_poly->max_len;

	vec_tmp = init_dvector(deg);
	zeros = init_dvector(deg);
	mat_tmp = init_dmatrix(deg, deg);

	dget_chev_zeros_g(zeros, interval_left, interval_right,  deg);
	for(i = 0; i < deg; i++)
	{
		for(j = 0; j < deg; j++)
			sdmij(mat_tmp, i, j, dpower(gdvi(zeros, i), j));
		sdvi(vec_tmp, i, f(gdvi(zeros, i)));
	}
	DLUdecomp(mat_tmp);
	SolveDLS(zeros, mat_tmp, vec_tmp);

	for(i = 0; i < deg; i++)
		set_dpoly_i(chev_poly, i, gdvi(zeros, i));

	free_dvector(vec_tmp);
	free_dvector(zeros);
	free_dmatrix(mat_tmp);
}

void dget_best_approx_g(DPoly poly, double interval_left, double interval_right, double (*f)(double), double (*df)(double), long int deg)
{
	long int i, j, times, main_times;
	double tmp, tmp1, e1, e2;
	DMatrix mat_tmp;
	DVector vec_tmp, coef_tmp, dev;

	/* init */
	mat_tmp = init_dmatrix(deg + 2, deg + 2);
	vec_tmp = init_dvector(deg + 2);
	dev = init_dvector(deg + 2);
	coef_tmp = init_dvector(deg + 2);
	dget_chev_zeros_g(dev, interval_left, interval_right, deg);
	
	tmp1 = gdvi(dev, 0);
	for(i = 1; i <= deg + 1; i++)
	{
		tmp = gdvi(dev, i);
		sdvi(dev, i, tmp1);
		tmp1 = tmp;
	}
	sdvi(dev, 0, interval_left);
	sdvi(dev, deg + 1, interval_right);

	/* Main loop */
	for(main_times = 0; main_times < 5; main_times++)
	{

	/* linear system */
	for(i = 0; i <= deg + 1; i++)
	{
		tmp = gdvi(dev, i);
		sdmij(mat_tmp, i, 0, 1.0);
		for(j = 1; j <= deg; j++)
			sdmij(mat_tmp, i, j, dpower(tmp, j));
		sdmij(mat_tmp, i, deg + 1, dpower(-1.0, i));
		sdvi(vec_tmp, i, f(tmp));
	}

	DLUdecomp(mat_tmp);
	SolveDLS(coef_tmp, mat_tmp, vec_tmp);

	print_dvector(coef_tmp);
	printf("DEVIATION:\n"); print_dvector(dev);

	for(i = 0; i <= deg; i++)
		set_dpoly_i(poly, i, gdvi(coef_tmp, i));
	printf("Best Approx Poly:\n"); print_dpoly(poly);

	/* with Regula-Falsi Method */
	for(i = 1; i <= deg; i++)
	{
		tmp = gdvi(dev, i);
		tmp1 = tmp * (1.0 + 1.0e-5); /* no reason(^^;) */
		for(times = 0; times < 2; times++)
		{
			e1 = df(tmp) - eval_diff_dpoly(poly, tmp);
			e2 = e1 - (df(tmp1) - eval_diff_dpoly(poly, tmp1));
			e1 *= (tmp - tmp1);

			tmp1 = tmp;
			if(fabs(e1 / e2)  <= fabs(tmp) * 1.0e-13)
				break;
			tmp -= e1 / e2;
		}
		sdvi(dev, i, tmp);
	}
//	sdvi(dev, 0, interval_left);
//	sdvi(dev, deg + 1, interval_right);

	printf("DEVIATION:\n"); print_dvector(dev);

	}
	/* clear */
	free_dmatrix(mat_tmp);
	free_dvector(vec_tmp);
	free_dvector(dev);
	free_dvector(coef_tmp);
}

#if USE_GMP

/* arctan(x) for GMP */
void mpf_atan(mpf_t ans, mpf_t x)
{
        mpf_t old_ans, tmp, xn, tmp_x, coef, pi2;
        unsigned long times;
        unsigned long prec;
        long int minus_flag = 0, large_flag = 0;

	/* get prec */
	prec = mpf_get_prec(ans);

	/* initialize temporary variables */
	mpf_init2(old_ans, prec);
	mpf_init2(coef, prec);
	mpf_init2(tmp_x, prec);
	mpf_init2(xn, prec);
	mpf_init2(tmp, prec);

	mpf_set(tmp_x, x);

	if(mpf_sgn(tmp_x) == 0)
	{
		mpf_set_ui(ans, 0UL);
		return;
	}
	if(mpf_sgn(tmp_x) < 0)
	{
		minus_flag = 1;
		mpf_neg(tmp_x, tmp_x);
	}
	if(mpf_cmp_ui(tmp_x, 1UL) == 0)
	{
		mpf_pi(ans);
		mpf_div_ui(ans, ans, 4UL);
		return;
	}
	if(mpf_cmp_ui(tmp_x, 1UL) > 0)
	{
		large_flag = 1;
		mpf_init2(pi2, prec);
		mpf_pi(pi2);
		mpf_div_ui(pi2, pi2, 2UL); /* pi2 = PI/2 */
		mpf_ui_div(tmp_x, 1UL, tmp_x);
	}

        /* ans := x */
        /* xn := x */
        /* old_ans := ans */
        /* coef  := 1   */
        /* tmp_x = x^2 */
        /* times := 0 */
        mpf_set(ans, tmp_x);
        mpf_set(xn, tmp_x);
        mpf_set(old_ans, ans);
        mpf_mul(tmp_x, tmp_x, tmp_x);
        mpf_set_ui(coef, 1UL);

	times = 0;

        do
        {
		/* coef := coef + 2 */
		mpf_add_ui(coef, coef, 2UL);
                mpf_mul(xn, xn, tmp_x);
                mpf_neg(xn, xn);
                mpf_div(tmp, xn, coef);

                /* ans += ans + (-1)^n * x^(2n+1)/(2n+1) */
                mpf_add(ans, ans, tmp);

/*		mpf_out_str(stdout, 10, 0, ans); printf(" ");
		mpf_out_str(stdout, 10, 0, old_ans); printf("\n");
*/

                /* ond_ans == ans */
		if(mpf_cmp(ans, old_ans) == 0)
			break;

		/* old_ans := ans */
		mpf_set(old_ans, ans);
//		times++;
        }while(1);

	/* PI/2 - atan(1/x) */
	if(large_flag == 1)
	{
		mpf_sub(ans, pi2, ans);
		mpf_clear(pi2);
	}
	/* atan(-x) = -atan(x) */
	if(minus_flag == 1)
		mpf_neg(ans, ans);

	/* clear */
	mpf_clear(old_ans);
	mpf_clear(coef);
	mpf_clear(tmp_x);
	mpf_clear(tmp);
	mpf_clear(xn);
}

/* deg <= 12 */
/* copied from Abramowitz & Stegun */
void _mpf_get_chev_poly(MPFPoly poly, long int deg)
{
/* T0(x) = 1 */
/* T1(x) = x */
/* T2(x) = 2x^2 - 1 */
/* T3(x) = 4x^3 - 3x^2 */
/* T4(x) = 8x^4 - 8x^2 + 1 */

	switch(deg)
	{
		case 0:
			set_mpfpoly_i_str(poly, 0, "1", 10);
			break;
		case 1:
			set_mpfpoly_i_str(poly, 0, "0", 10);
			set_mpfpoly_i_str(poly, 1, "1", 10);
			break;
		case 2:
			set_mpfpoly_i_str(poly, 0, "-1", 10);
			set_mpfpoly_i_str(poly, 1, "0", 10);
			set_mpfpoly_i_str(poly, 2, "2", 10);
			break;
		case 3:
			set_mpfpoly_i_str(poly, 0, "0", 10);
			set_mpfpoly_i_str(poly, 1, "-3", 10);
			set_mpfpoly_i_str(poly, 2, "0", 10);
			set_mpfpoly_i_str(poly, 3, "4", 10);
			break;
		case 4:
			set_mpfpoly_i_str(poly, 0, "1", 10);
			set_mpfpoly_i_str(poly, 1, "0", 10);
			set_mpfpoly_i_str(poly, 2, "-8", 10);
			set_mpfpoly_i_str(poly, 3, "0", 10);
			set_mpfpoly_i_str(poly, 4, "8", 10);
			break;
		case 5:
			set_mpfpoly_i_str(poly, 0, "0", 10);
			set_mpfpoly_i_str(poly, 1, "5.0", 10);
			set_mpfpoly_i_str(poly, 2, "0.0", 10);
			set_mpfpoly_i_str(poly, 3, "-20.0", 10);
			set_mpfpoly_i_str(poly, 4, "0.0", 10);
			set_mpfpoly_i_str(poly, 5, "16.0", 10);
			break;
		case 6:
			set_mpfpoly_i_str(poly, 0, "-1.0", 10);
			set_mpfpoly_i_str(poly, 1, "0.0", 10);
			set_mpfpoly_i_str(poly, 2, "18.0", 10);
			set_mpfpoly_i_str(poly, 3, "0.0", 10);
			set_mpfpoly_i_str(poly, 4, "-48.0", 10);
			set_mpfpoly_i_str(poly, 5, "0.0", 10);
			set_mpfpoly_i_str(poly, 6, "32.0", 10);
			break;
		case 7:
			set_mpfpoly_i_str(poly, 0, "0.0", 10);
			set_mpfpoly_i_str(poly, 1, "-7.0", 10);
			set_mpfpoly_i_str(poly, 2, "0.0", 10);
			set_mpfpoly_i_str(poly, 3, "56.0", 10);
			set_mpfpoly_i_str(poly, 4, "0.0", 10);
			set_mpfpoly_i_str(poly, 5, "-112.0", 10);
			set_mpfpoly_i_str(poly, 6, "0.0", 10);
			set_mpfpoly_i_str(poly, 7, "64.0", 10);
			break;
		case 8:
			set_mpfpoly_i_str(poly, 0, "1.0", 10);
			set_mpfpoly_i_str(poly, 1, "0.0", 10);
			set_mpfpoly_i_str(poly, 2, "-32.0", 10);
			set_mpfpoly_i_str(poly, 3, "0.0", 10);
			set_mpfpoly_i_str(poly, 4, "160.0", 10);
			set_mpfpoly_i_str(poly, 5, "0.0", 10);
			set_mpfpoly_i_str(poly, 6, "-256.0", 10);
			set_mpfpoly_i_str(poly, 7, "0.0", 10);
			set_mpfpoly_i_str(poly, 8, "128.0", 10);
			break;
		case 9:
			set_mpfpoly_i_str(poly, 0, "0.0", 10);
			set_mpfpoly_i_str(poly, 1, "9.0", 10);
			set_mpfpoly_i_str(poly, 2, "0.0", 10);
			set_mpfpoly_i_str(poly, 3, "-120.0", 10);
			set_mpfpoly_i_str(poly, 4, "0.0", 10);
			set_mpfpoly_i_str(poly, 5, "432.0", 10);
			set_mpfpoly_i_str(poly, 6, "0.0", 10);
			set_mpfpoly_i_str(poly, 7, "-576.0", 10);
			set_mpfpoly_i_str(poly, 8, "0.0", 10);
			set_mpfpoly_i_str(poly, 9, "256.0", 10);
			break;
		case 10:
			set_mpfpoly_i_str(poly, 0, "-1.0", 10);
			set_mpfpoly_i_str(poly, 1, "0.0", 10);
			set_mpfpoly_i_str(poly, 2, "50.0", 10);
			set_mpfpoly_i_str(poly, 3, "0.0", 10);
			set_mpfpoly_i_str(poly, 4, "-400.0", 10);
			set_mpfpoly_i_str(poly, 5, "0.0", 10);
			set_mpfpoly_i_str(poly, 6, "1120.0", 10);
			set_mpfpoly_i_str(poly, 7, "0.0", 10);
			set_mpfpoly_i_str(poly, 8, "-1280.0", 10);
			set_mpfpoly_i_str(poly, 9, "0.0", 10);
			set_mpfpoly_i_str(poly, 10, "512.0", 10);
			break;
		case 11:
			set_mpfpoly_i_str(poly, 0, "0.0", 10);
			set_mpfpoly_i_str(poly, 1, "-11.0", 10);
			set_mpfpoly_i_str(poly, 2, "0.0", 10);
			set_mpfpoly_i_str(poly, 3, "220.0", 10);
			set_mpfpoly_i_str(poly, 4, "0.0", 10);
			set_mpfpoly_i_str(poly, 5, "-1232.0", 10);
			set_mpfpoly_i_str(poly, 6, "0.0", 10);
			set_mpfpoly_i_str(poly, 7, "2816.0", 10);
			set_mpfpoly_i_str(poly, 8, "0.0", 10);
			set_mpfpoly_i_str(poly, 9, "-2816.0", 10);
			set_mpfpoly_i_str(poly, 10, "0.0", 10);
			set_mpfpoly_i_str(poly, 11, "1024.0", 10);
			break;
		case 12:
			set_mpfpoly_i_str(poly, 0, "1.0", 10);
			set_mpfpoly_i_str(poly, 1, "0.0", 10);
			set_mpfpoly_i_str(poly, 2, "-72.0", 10);
			set_mpfpoly_i_str(poly, 3, "0.0", 10);
			set_mpfpoly_i_str(poly, 4, "840.0", 10);
			set_mpfpoly_i_str(poly, 5, "0.0", 10);
			set_mpfpoly_i_str(poly, 6, "-3584.0", 10);
			set_mpfpoly_i_str(poly, 7, "0.0", 10);
			set_mpfpoly_i_str(poly, 8, "6912.0", 10);
			set_mpfpoly_i_str(poly, 9, "0.0", 10);
			set_mpfpoly_i_str(poly, 10, "-6144.0", 10);
			set_mpfpoly_i_str(poly, 11, "0.0", 10);
			set_mpfpoly_i_str(poly, 12, "2048.0", 10);
			break;
	}
	
	return;
}

void mpf_get_chev_poly(MPFPoly poly, long int deg)
{
	long int i, j;
	MPFPoly tmp_poly1, tmp_poly2;
	mpf_t tmp;

	if(deg <= 12)
	{
		_mpf_get_chev_poly(poly, deg);
		return;
	}
/*	if(deg >= ?)
	{
		fprintf(stderr, "ERROR: Too big degree!(mpf_get_chev_poly)\n");
		return;
	}
*/
	/* deg > 12 */
	mpf_init2(tmp, poly->prec);
	tmp_poly1 = init_mpfpoly(deg + 1);
	tmp_poly2 = init_mpfpoly(deg + 1);
	_mpf_get_chev_poly(tmp_poly2, 12);
	_mpf_get_chev_poly(tmp_poly1, 11);

	/* Tn+1(x) = 2x * Tn(x) - Tn-1(x) */
	for(i = 13; i <= deg; i++)
	{
		set_mpfpoly_i_str(poly, 0, "0.0", 10);
		for(j = 0; j < i; j++)
		{
			mpf_mul_ui(tmp, get_mpfpoly_i(tmp_poly2, j), 2UL);
			set_mpfpoly_i(poly, j + 1, tmp);
		}
		for(j = 0; j < i - 1; j++)
		{
			mpf_set(tmp, get_mpfpoly_i(poly, j));
			mpf_sub(tmp, tmp, get_mpfpoly_i(tmp_poly1, j));
			set_mpfpoly_i(poly, j, tmp);
		}
		subst_mpfpoly(tmp_poly1, tmp_poly2); tmp_poly1->deg = tmp_poly2->deg;
		subst_mpfpoly(tmp_poly2, poly); tmp_poly2->deg = poly->deg;

	}

	free_mpfpoly(tmp_poly1);
	free_mpfpoly(tmp_poly2);
}

void mpf_get_chev_zeros(MPFVector zeros, long int deg)
{
	long int i;
	mpf_t pi, tmp, tmp1;

	mpf_init2(pi, zeros->prec);
	mpf_init2(tmp, zeros->prec);
	mpf_init2(tmp1, zeros->prec);
	mpf_pi(pi);
	for(i = 0; i < deg; i++)
	{
		mpf_set_ui(tmp, (unsigned long)(deg - 1 - i));
		mpf_mul_ui(tmp, tmp, 2UL);
		mpf_add_ui(tmp, tmp, 1UL);
		mpf_mul(tmp, tmp, pi);
		mpf_set_ui(tmp1, (unsigned long)(2 * deg));
		mpf_div(tmp, tmp, tmp1);
		mpf_cos(tmp1, tmp);
		smpfvi(zeros, i, tmp1);
	}
	mpf_clear(pi);
	mpf_clear(tmp);
	mpf_clear(tmp1);
}

void mpf_get_chev_zeros_g(MPFVector zeros, mpf_t interval_left, mpf_t interval_right, long int deg)
{
	long int i;
	mpf_t tmp, c1, c2;

	mpf_get_chev_zeros(zeros, deg);
	mpf_init2(c1, zeros->prec);
	mpf_init2(c2, zeros->prec);
	mpf_init2(tmp, zeros->prec);

	mpf_add(c1, interval_left, interval_right);
	mpf_div_ui(c1, c1, 2UL);
	mpf_sub(c2, interval_left, interval_right);
	mpf_div_ui(c2, c2, 2UL);
	for(i = 0; i < deg; i++)
	{
		mpf_mul(tmp, c2, gmpfvi(zeros, i));
		mpf_sub(tmp, c1, tmp);
		smpfvi(zeros, i, tmp);
	}

	mpf_clear(c1);
	mpf_clear(c2);
	mpf_clear(tmp);
}

void mpf_get_chev_approx_g(MPFPoly chev_poly, mpf_t interval_left, mpf_t interval_right, void (*f)(mpf_t, mpf_t))
{
	long int deg, i, j;
	MPFVector vec_tmp, zeros;
	MPFMatrix mat_tmp;
	mpf_t tmp;

	deg = chev_poly->max_len;

	vec_tmp = init_mpfvector(deg);
	zeros = init_mpfvector(deg);
	mat_tmp = init_mpfmatrix(deg, deg);
	mpf_init2(tmp, chev_poly->prec);

	mpf_get_chev_zeros_g(zeros, interval_left, interval_right,  deg);
	for(i = 0; i < deg; i++)
	{
		for(j = 0; j < deg; j++)
		{
			mpf_power(tmp, gmpfvi(zeros, i), j);
			smpfmij(mat_tmp, i, j, tmp);
		}
		f(tmp, gmpfvi(zeros, i));
		smpfvi(vec_tmp, i, tmp);
	}
	MPFLUdecomp(mat_tmp);
	SolveMPFLS(zeros, mat_tmp, vec_tmp);

	for(i = 0; i < deg; i++)
		set_mpfpoly_i(chev_poly, i, gmpfvi(zeros, i));

	free_mpfvector(vec_tmp);
	free_mpfvector(zeros);
	free_mpfmatrix(mat_tmp);
	mpf_clear(tmp);
}

void mpf_get_best_approx_g(MPFPoly poly, mpf_t interval_left, mpf_t interval_right, void (*f)(mpf_t, mpf_t), void (*df)(mpf_t, mpf_t), long int deg)
{
	unsigned long prec, org_prec;
	long int i, j, times, main_times;
	mpf_t tmp, tmp1, tmp2, tmp3, e1, e2;
	MPFMatrix mat_tmp;
	MPFVector vec_tmp, coef_tmp, dev;
	MPFPoly poly_tmp;

	/* init */
	org_prec = poly->prec;
	prec = org_prec * 2;
//	prec = org_prec;
	mat_tmp = init2_mpfmatrix(deg + 2, deg + 2, prec);
	vec_tmp = init2_mpfvector(deg + 2, prec);
	dev = init2_mpfvector(deg + 2, prec);
	coef_tmp = init2_mpfvector(deg + 2, prec);
	poly_tmp = init2_mpfpoly(deg + 1, prec);

	mpf_init2(tmp, prec);
	mpf_init2(tmp1, prec);
	mpf_init2(tmp2, prec);
	mpf_init2(tmp3, prec);
	mpf_init2(e1, prec);
	mpf_init2(e2, prec);

	mpf_get_chev_zeros_g(dev, interval_left, interval_right, deg);

	mpf_set(tmp1, gmpfvi(dev, 0));
	for(i = 1; i <= deg + 1; i++)
	{
		mpf_set(tmp, gmpfvi(dev, i));
		smpfvi(dev, i, tmp1);
		mpf_set(tmp1, tmp);
	}
	smpfvi(dev, 0, interval_left);
	smpfvi(dev, deg + 1, interval_right);
//	print_mpfvector(dev);

	/* Main loop */
	for(main_times = 0; main_times < 10; main_times++)
	{

	/* linear system */
	for(i = 0; i <= deg + 1; i++)
	{
		mpf_set(tmp, gmpfvi(dev, i));
		mpf_set_ui(tmp2, 1UL);
		smpfmij(mat_tmp, i, 0, tmp2);
		for(j = 1; j <= deg; j++)
		{
			mpf_pow_ui(tmp1, tmp, j);
			smpfmij(mat_tmp, i, j, tmp1);
		}
		mpf_set_str(e1, "-1", 10);
		mpf_pow_ui(tmp1, e1, i);
		smpfmij(mat_tmp, i, deg + 1, tmp1);
		f(tmp1, tmp);
		smpfvi(vec_tmp, i, tmp1);
	}
//	print_mpfmatrix(mat_tmp);
//	print_mpfvector(vec_tmp);

	MPFLUdecomp(mat_tmp);
	SolveMPFLS(coef_tmp, mat_tmp, vec_tmp);

//	print_mpfvector(coef_tmp);
	printf("DEVIATION:\n"); print_mpfvector(dev);

	for(i = 0; i <= deg; i++)
		set_mpfpoly_i(poly_tmp, i, gmpfvi(coef_tmp, i));
	printf("Best Approx Poly:\n"); print_mpfpoly(poly_tmp);

	/* with Regula-Falsi Method */
	for(i = 1; i <= deg; i++)
	{
		mpf_set(tmp, gmpfvi(dev, i));
		mpf_set_ui(tmp2, 1UL);
		mpf_div_2exp(tmp2, tmp2, prec / 3);
		mpf_add_ui(tmp2, tmp2, 1UL);
		mpf_mul(tmp1, tmp, tmp2); /* no reason(^^;) */
		printf("tmp1:");mpf_out_str(stdout, 10, 0, tmp1); printf("\n");
		printf("tmp :");mpf_out_str(stdout, 10, 0, tmp); printf("\n");

		/* e1 = df(tmp) - eval_diff_dpoly(poly, tmp); */
		/* e2 = e1 - (df(tmp1) - eval_diff_dpoly(poly, tmp1)); */
		/* e1 *= (tmp - tmp1); */
		for(times = 0; times < 5; times++)
		{
			df(e1, tmp);
			eval_diff_mpfpoly(tmp2, poly_tmp, tmp);
			mpf_sub(e1, e1, tmp2);

			df(tmp2, tmp1);
			mpf_sub(e2, e1, tmp2);
			eval_diff_mpfpoly(tmp2, poly_tmp, tmp1);
			mpf_add(e2, e2, tmp2);

			mpf_sub(tmp2, tmp, tmp1);
			mpf_mul(e1, e1, tmp2);

			/* tmp1 := tmp */
			/* tmp := tmp - e1 / e2 */
			mpf_set(tmp1, tmp);
			mpf_div(tmp2, e1, e2);
			mpf_sub(tmp, tmp, tmp2);

			mpf_abs(tmp2, tmp2);
			mpf_div_2exp(tmp3, tmp1, prec / 2);
			mpf_abs(tmp3, tmp3);
//			mpf_out_str(stdout, 10, 0, e1);
//			printf("%5d %5d ", i, times);mpf_out_str(stdout, 10, 0, tmp);printf("\n");
			if(mpf_cmp(tmp2, tmp3) <= 0)
				break;
		}
		smpfvi(dev, i, tmp);
	}
//	sdvi(dev, 0, interval_left);
//	sdvi(dev, deg + 1, interval_right);

	printf("DEVIATION:\n"); print_mpfvector(dev);

	}

//	subst_mpfpoly(poly, poly_tmp);
	for(i = 0; i <= poly_tmp->deg; i++)
		set_mpfpoly_i(poly, i, get_mpfpoly_i(poly_tmp, i));
	poly->prec = org_prec;
	poly->deg = poly_tmp->deg;

	/* clear */
	mpf_clear(tmp);
	mpf_clear(tmp1);
	mpf_clear(tmp2);
	mpf_clear(tmp3);
	mpf_clear(e1);
	mpf_clear(e2);

	free_mpfmatrix(mat_tmp);
	free_mpfvector(vec_tmp);
	free_mpfvector(dev);
	free_mpfvector(coef_tmp);
	free_mpfpoly(poly_tmp);
}

#endif

#define DEG 16

/* (atan x)' = 1/(1+x^2) */
double datan(double x)
{
	return 1.0 / (1 + x * x);
}

#ifdef USE_GMP
/* (atan x)' = 1/(1+x^2) */
void mpf_datan(mpf_t ret, mpf_t x)
{
	mpf_mul(ret, x, x);
	mpf_add_ui(ret, ret, 1UL);
	mpf_ui_div(ret, 1UL, ret);

	return;
}
#endif

main()
{
	long int i;
	DPoly dchev;
	DVector dzeros;
	double dx;
#ifdef USE_GMP
	MPFPoly mpfchev, mpfbest;
	MPFVector mpfzeros;
	mpf_t tmp, tmp2, tmp3, error, mpfx, mpf_a, mpf_b;
#endif
	dzeros = init_dvector(DEG);

	dget_chev_zeros(dzeros, DEG);
	for(i = 0; i < DEG; i++)
		printf("%5d %25.17e\n", i, gdvi(dzeros, i));
	dget_chev_zeros_g(dzeros, 0.0, 1.0, DEG);
	for(i = 0; i < DEG; i++)
		printf("%5d %25.17e\n", i, gdvi(dzeros, i));

	dchev = init_dpoly(DEG+1);

	dget_chev_approx_g(dchev, 0.0, 1.0, atan);
	print_dpoly(dchev);
	for(dx = 0.0; dx <= 1.0; dx += 0.01)
		printf("%10.3e %25.17e %25.17e %10.3e\n", dx, eval_dpoly(dchev, dx), atan(dx), atan(dx) - eval_dpoly(dchev, dx));

	dget_best_approx_g(dchev, 0.0, 1.0, atan, datan, DEG);

	for(dx = 0.0; dx <= 1.0; dx += 0.01)
		printf("%10.3e %25.17e %25.17e %10.3e\n", dx, eval_dpoly(dchev, dx), atan(dx), atan(dx) - eval_dpoly(dchev, dx));

/*
	for(i = 0; i <= 2; i++)
	{
		dget_chev_poly(dchev, i);
		print_dpoly(dchev);
	}
*/
	free_dpoly(dchev);

/* GMP & MPFR */
#ifdef USE_GMP
	set_bnc_default_prec(128);

	mpfzeros = init_mpfvector(DEG*2);
	mpf_init_set_ui(mpf_a, 0UL);
	mpf_init_set_ui(mpf_b, 1UL);

	mpf_get_chev_zeros(mpfzeros, DEG);
	for(i = 0; i < DEG; i++)
		printf("%5d %25.17e\n", i, mpf_get_d(gmpfvi(mpfzeros, i)));
	mpf_get_chev_zeros_g(mpfzeros, mpf_a, mpf_b, DEG);
	for(i = 0; i < DEG; i++)
		printf("%5d %25.17e\n", i, mpf_get_d(gmpfvi(mpfzeros, i)));

	mpfchev = init_mpfpoly((DEG+1)*2);
//	mpfbest = init2_mpfpoly((DEG+1)*2, 256);
	mpfbest = init_mpfpoly((DEG+1)*2);
	
	mpf_init(mpfx);
	mpf_init(tmp);
	mpf_init(tmp2);
	mpf_init(tmp3);
	mpf_init(error);
	mpf_sub(tmp2, mpf_b, mpf_a);
	mpf_div_ui(tmp2, tmp2, 100UL);
	for(mpf_set(mpfx, mpf_a); mpf_cmp(mpfx, mpf_b) <= 0; mpf_add(mpfx, mpfx, tmp2))
	{
		mpf_atan(tmp, mpfx);
		printf("%10.3e %25.17e %25.17e %10.3e\n", mpf_get_d(mpfx), mpf_get_d(tmp), atan(mpf_get_d(mpfx)), mpf_get_d(tmp)-atan(mpf_get_d(mpfx)));
	}

	mpf_get_chev_approx_g(mpfchev, mpf_a, mpf_b, mpf_atan);
	print_mpfpoly(mpfchev);
	for(mpf_set(mpfx, mpf_a); mpf_cmp(mpfx, mpf_b) <= 0; mpf_add(mpfx, mpfx, tmp2))
	{
		eval_mpfpoly(tmp, mpfchev, mpfx);
		mpf_atan(tmp3, mpfx);
		mpf_sub(error, tmp3, tmp);
		printf("%10.3e %25.17e %25.17e %10.3e\n", mpf_get_d(mpfx), mpf_get_d(tmp), mpf_get_d(tmp3), mpf_get_d(error));
	}

	mpf_get_best_approx_g(mpfbest, mpf_a, mpf_b, mpf_atan, mpf_datan, DEG*2);
	print_mpfpoly(mpfchev);

	for(mpf_set(mpfx, mpf_a); mpf_cmp(mpfx, mpf_b) <= 0; mpf_add(mpfx, mpfx, tmp2))
	{
		eval_mpfpoly(tmp, mpfchev, mpfx);
		mpf_atan(tmp3, mpfx);
		mpf_sub(error, tmp, tmp3);
		printf("%10.3e %25.17e %25.17e %10.3e\n", mpf_get_d(mpfx), mpf_get_d(tmp), mpf_get_d(tmp3), mpf_get_d(error));
	}

	free_mpfpoly(mpfchev);
}
#endif
