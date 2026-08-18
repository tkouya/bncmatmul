/********************************************************************************/
/* test_newton.c:                                                               */
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
/*************************************************/
/* test_newton.c                                 */
/*************************************************/
#include <stdio.h>
#include "bnc.h"

float ff(float x)
{
	/* x^2 - 2 = 0 */
	return x * x - 2.0;
}

float dff(float x)
{
	/* 2*x */
	return 2.0 * x;
}

double fd(double x)
{
	/* x^2 - 2 = 0 */
	return x * x - 2.0;
}

double dfd(double x)
{
	/* 2*x */
	return 2.0 * x;
}

#ifdef USE_GMP
void f(mpf_t ret, mpf_t x)
{
	/* x^2 - 2 = 0 */
	mpf_mul(ret, x, x);
	mpf_sub_ui(ret, ret, 2UL);
}

void df(mpf_t ret, mpf_t x)
{
	/* 2*x */
	mpf_mul_ui(ret, x, 2UL);
}
#endif

main()
{
	float fans;
	double dans;
	long int times;
#ifdef USE_GMP
	mpf_t ans, x0, aeps, reps;
#endif

	/* float */
	times = fnewton_1(&fans, 1.0, ff, dff, 100, 1.0e-50, 1.0e-5);
	printf(" times : %ld\n", times);
	printf("fnewton_1: %10.7e\n", fans);
	times = fsnewton_1(&fans, 1.0, ff, dff, 100, 1.0e-50, 1.0e-5);
	printf(" times : %ld\n", times);
	printf("fsnewton_1: %10.7e\n", fans);

	/* double */
	times = dnewton_1(&dans, 1.0, fd, dfd, 100, 1.0e-50, 1.0e-10);
	printf(" times : %ld\n", times);
	printf("dnewton_1: %25.17e\n", dans);
	times = dsnewton_1(&dans, 1.0, fd, dfd, 100, 1.0e-50, 1.0e-10);
	printf(" times : %ld\n", times);
	printf("dsnewton_1: %25.17e\n", dans);

#ifdef USE_GMP
	/* mpf_t */
	set_bnc_default_prec(1024);
	mpf_init(ans);
	mpf_init(x0);
	mpf_init_set_d(aeps, 1.0e-100);
	mpf_init_set_d(reps, 1.0e-50);

	/* newton_1 */
	mpf_set_ui(x0, 1UL);
	times = mpf_newton_1(ans, x0, f, df, 100, aeps, reps);
	printf(" times : %ld\n", times);
	printf("mpf_newton_1:   "); mpf_out_str(stdout, 10, 0, ans); printf("\n");

	/* snewton_1 */
	times = mpf_snewton_1(ans, x0, f, df, 100, aeps, reps);
	printf(" times : %ld\n", times);
	printf("mpf_snewton_1:   "); mpf_out_str(stdout, 10, 0, ans); printf("\n");

	mpf_sqrt_ui(ans, 2UL);
	printf("mpf_sqrt: "); mpf_out_str(stdout, 10, 0, ans); printf("\n");
#endif
}

