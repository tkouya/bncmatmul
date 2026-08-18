/********************************************************************************/
/* test_integral.c:                                                             */
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
#include <math.h>
#include "bnc.h"

/* test function : f(x) = sqrt(1 - k^2 * x^2) / sqrt(1 - x^2) */
float ff(float x)
{
	return 5 * (float)sqrt((double)(1.0 - 0.8 * 0.8 * x * x)) / (float)sqrt((double)(1.0 - x * x));
}

/* test function : f(x) = sqrt(1 - k^2 * x^2) / sqrt(1 - x^2) */
double df(double x)
{
	return 5.0 * sqrt(1.0 - 0.8 * 0.8 * x * x) / sqrt(1.0 - x * x);
}

#ifdef USE_GMP
/* test function : f(x) = sqrt(1 - k^2 * x^2) / sqrt(1 - x^2) */
void mpf_f(mpf_t ret, mpf_t x)
{
	mpf_t tmp, num, den;

	mpf_init(tmp);
	mpf_init(num);
	mpf_init(den);

	mpf_set_ui(num, 1UL);
	mpf_set_ui(tmp, 4UL); mpf_div_ui(tmp, tmp, 5UL); mpf_mul(tmp, tmp, x);
	mpf_mul(tmp, tmp, tmp);
	mpf_sub(num, num, tmp);
	mpf_sqrt(num, num);

	mpf_set_ui(den, 1UL);
	mpf_mul(tmp, x, x);
	mpf_sub(den, den, tmp);
	mpf_sqrt(den, den);

	mpf_div(ret, num, den);
	mpf_mul_ui(ret, ret, 5UL);

	mpf_clear(tmp);
	mpf_clear(num);
	mpf_clear(den);
}
#endif

main()
{
	long int num_div;
	float fa, fb;
	double da, db;
#ifdef USE_GMP
	mpf_t mpf_a, mpf_b, mpf_ans;

	set_bnc_default_prec(128);
#endif

	fa = 0.0; fb = 0.8;
	da = 0.0; db = 0.8;

#ifdef USE_GMP
	mpf_init(mpf_ans);
	mpf_init_set_ui(mpf_a, 0UL);
	mpf_init_set_ui(mpf_b, 4UL); mpf_div_ui(mpf_b, mpf_b, 5UL);
#endif

	printf("Number of Division, Integral[");

#ifdef USE_GMP
	mpf_out_str(stdout, 10, 0, mpf_a); printf(", ");
	mpf_out_str(stdout, 10, 0, mpf_b); printf("]\n");
#endif
	for(num_div = 2; num_div <= 1048576; num_div *= 4)
	{
		printf(" %10d , %15.7e, %25.17e, %25.17e, ", num_div, ftrapezoidal_fs(fa, fb, ff, num_div), dtrapezoidal_fs(da, db, df, num_div), dmtrapezoidal_fs(da, db, df, num_div));
#ifdef USE_GMP
		mpf_trapezoidal_fs(mpf_ans, mpf_a, mpf_b, mpf_f, num_div);
		mpf_out_str(stdout, 10, 0, mpf_ans);
		mpf_mtrapezoidal_fs(mpf_ans, mpf_a, mpf_b, mpf_f, num_div);
		mpf_out_str(stdout, 10, 0, mpf_ans);
#endif
		printf("\n");
	}
#ifdef USE_GMP
	mpf_clear(mpf_ans);
	mpf_clear(mpf_a);
	mpf_clear(mpf_b);
#endif
}
