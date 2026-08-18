/********************************************************************************/
/* integral.c:                                                                  */
/* Copyright (C) 2003-2011 Tomonori Kouya                                       */
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
#include "bnc.h"

/* float: Trapezoidal rule */
float ftrapezoidal_fs(float x_start, float x_end, float (*func)(float x), long int num_div)
{
	float x, h, ret;
	long int i;

	/* set stepsize */
	h = (x_end - x_start) / num_div;

	/* ret := (f(a) + f(b))/2 */
	ret = (func(x_start) + func(x_end)) / 2;

	/* ret += sum^{n-1}_{i = 1} f(x_i) */
	for(i = 1; i < num_div; i++)
	{
		x = x_start + h * i;
		ret += func(x);
	}
	ret *= h;

	return ret;
}

/* double: Trapezoidal rule */
double dtrapezoidal_fs(double x_start, double x_end, double (*func)(double x), long int num_div)
{
	double x, h, ret;
	long int i;

	/* set stepsize */
	h = (x_end - x_start) / num_div;

	/* ret := (f(a) + f(b))/2 */
	ret = (func(x_start) + func(x_end)) / 2;

	/* ret += sum^{n-1}_{i = 1} f(x_i) */
	for(i = 1; i < num_div; i++)
	{
		x = x_start + h * i;
		ret += func(x);
	}
	ret *= h;

	return ret;
}

/* double: Trapezoidal rule with end-point checks */
double dmtrapezoidal_fs(double x_start, double x_end, double (*func)(double x), long int num_div)
{
	double x, old_x_start, old_x_end, h, ret;
	long int i;

	/* check start point */
	old_x_start = x_start;
	while(isnan(func(x_start)) || isinf(func(x_start)))
	{
//		printf("%25.17e + %25.17e\n", x_start, pow(2.0, -52.0));
		x_start += pow(2.0, -50.0);
	}
//	printf("new_x_start: %25.17e\n", x_start);

	/* check end point */
	old_x_end = x_end;
	while(isnan(func(x_end)) || isinf(func(x_end)))
	{
//		printf("%25.17e - %25.17e\n", x_end, pow(2.0, -52.0));
		x_end -= pow(2.0, -50.0);
	}
//	printf("new_x_start: %25.17e\n", x_start);

	/* set stepsize */
	h = (x_end - x_start) / num_div;

	/* ret := (f(a) + f(b))/2 */
	ret = (func(x_start) + func(x_end)) / 2;

	/* ret += sum^{n-1}_{i = 1} f(x_i) */
	for(i = 1; i < num_div; i++)
	{
		x = x_start + h * i;
		ret += func(x);
	}
	ret *= h;

	return ret;
}

#ifdef USE_GMP
/* Trapezoidal rule */
void mpf_trapezoidal_fs(mpf_t ret, mpf_t x_start, mpf_t x_end, void (*func)(mpf_t, mpf_t), long int num_div)
{
	mpf_t x, h, tmp;
	long int i;

	mpf_init2(x, mpf_get_prec(x_start));
	mpf_init2(h, mpf_get_prec(x_start));
	mpf_init2(tmp, mpf_get_prec(ret));

	/* set Stepsize */
	mpf_sub(h, x_end, x_start); mpf_div_ui(h, h, (unsigned long)num_div);

	/* ret := (f(a) + f(b)) / 2 */
	func(ret, x_start);
	func(tmp, x_end);
	mpf_add(ret, ret, tmp); mpf_div_ui(ret, ret, 2UL);

	/* set starting and ending values */
	mpf_add(x, x_start, h);

	/* ret += sum^{n-1}_{i=1} f(x_i) */
	for(i = 0; i < num_div - 1; i++)
	{
		func(tmp, x);
		mpf_add(ret, ret, tmp);
		mpf_add(x, x, h);
	}

	/* ret := h * { (f(a) + f(b)) / 2 + sum^{n-1}_{i=1} f(x_i) } */
	mpf_mul(ret, ret, h);

	mpf_clear(x);
	mpf_clear(h);
	mpf_clear(tmp);
}

/* Trapezoidal rule with end-point checks */
void mpf_mtrapezoidal_fs(mpf_t ret, mpf_t x_start, mpf_t x_end, void (*func)(mpf_t, mpf_t), long int num_div)
{
	mpf_t x, h, tmp, old_x_start, old_x_end, eps;
	long int i;
	unsigned long prec;

	prec = mpf_get_prec(ret);

	mpf_init2(x, prec);
	mpf_init2(h, prec);
	mpf_init2(tmp, prec);
	mpf_init2(eps, prec);
	mpf_init2(old_x_start, prec);
	mpf_init2(old_x_end, prec);

	/* check start point */
	mpf_set(old_x_start, x_start);

	mpf_set_ui(tmp, 2UL);
	mpf_ui_div(tmp, 1UL, tmp);
	mpf_power(eps, tmp, prec-1);
//	mpf_out_str(stdout, 10, 0, eps); printf("\n");
	func(tmp, x_start);
	while(mpfr_nan_p(tmp) || mpfr_inf_p(tmp))
	{
//		mpf_out_str(stdout, 10, 0, x_start); printf("\n");
		mpf_add(x_start, x_start, eps);
		func(tmp, x_start);
	}
//	printf("new_x_start: "); mpf_out_str(stdout, 10, 0, x_start); printf("\n");

	/* check end point */
	mpf_set(old_x_end, x_end);

	mpf_set_ui(tmp, 2UL);
	mpf_ui_div(tmp, 1UL, tmp);
	mpf_power(eps, tmp, prec-1);
//	mpf_out_str(stdout, 10, 0, eps); printf("\n");
	func(tmp, x_end);
	while(mpfr_nan_p(tmp) || mpfr_inf_p(tmp))
	{
//		mpf_out_str(stdout, 10, 0, x_end); printf("\n");
		mpf_sub(x_end, x_end, eps);
		func(tmp, x_end);
	}
//	printf("new_x_end: "); mpf_out_str(stdout, 10, 0, x_end); printf("\n");

	/* set Stepsize */
	mpf_sub(h, x_end, x_start); mpf_div_ui(h, h, (unsigned long)num_div);

	/* ret := (f(a) + f(b)) / 2 */
	func(ret, x_start);
	func(tmp, x_end);
	mpf_add(ret, ret, tmp); mpf_div_ui(ret, ret, 2UL);

	/* set starting and ending values */
	mpf_add(x, x_start, h);

	/* ret += sum^{n-1}_{i=1} f(x_i) */
	for(i = 0; i < num_div - 1; i++)
	{
		func(tmp, x);
		mpf_add(ret, ret, tmp);
		mpf_add(x, x, h);
	}

	/* ret := h * { (f(a) + f(b)) / 2 + sum^{n-1}_{i=1} f(x_i) } */
	mpf_mul(ret, ret, h);

	/* return */
	mpf_set(x_start, old_x_start);
	mpf_set(x_end, old_x_end);

	mpf_clear(x);
	mpf_clear(eps);
	mpf_clear(old_x_start);
	mpf_clear(old_x_end);
	mpf_clear(h);
	mpf_clear(tmp);
}
#endif
