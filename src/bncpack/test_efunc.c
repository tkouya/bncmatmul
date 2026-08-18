/********************************************************************************/
/* test_efunc.c:                                                                */
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

main()
{
#ifndef USE_GMP
	printf("This program are not available without GMP.\n");
#else
	long int i, max_times;
	double x_min, x_max, x, h;
	mpf_t mp_x_min, mp_x_max, mp_x, mp_h, mp_sin, mp_cos, mp_exp, mp_pi, mp_e, mp_ln2, mp_ln, mp_log10;

	set_bnc_default_prec(128);
	max_times = 128;
	x_min = -30.0;
	x_max = 30.0;
	mpf_init_set_si(mp_x_min, -30);
	mpf_init_set_si(mp_x_max, 30);

	h = (x_max - x_min) / max_times;
	mpf_init(mp_h);
	mpf_init(mp_sin);
	mpf_init(mp_cos);
	mpf_init(mp_exp);
	mpf_init(mp_pi);
	mpf_init(mp_e);
	mpf_init(mp_ln2);
	mpf_init(mp_ln);
	mpf_init(mp_log10);
	mpf_sub(mp_h, mp_x_max, mp_x_min);
	mpf_div_ui(mp_h, mp_h, (unsigned long)max_times);

	mpf_init_set(mp_x, mp_x_min);
	x = x_min;

	printf("        x             sin(x)                mpf_sin(x)\n");
	for(i = 0; i < max_times; i++)
	{
		printf("%25.17e %25.17e ", x, sin(x));
		mpf_sin(mp_sin, mp_x);
//		mpf_out_str(stdout, 10, 0, mp_sin);
		printf("%25.17e %25.17e", mpf2double(mp_sin), cos(x));
		mpf_cos(mp_cos, mp_x);
//		mpf_out_str(stdout, 10, 0, mp_cos);
		printf("%25.17e %25.17e", mpf2double(mp_cos), exp(x));
		mpf_exp(mp_exp, mp_x);
//		mpf_out_str(stdout, 10, 0, mp_exp);
		printf("%25.17e %25.17e", mpf2double(mp_exp), log(x));
		mpf_ln(mp_ln, mp_x);
//		mpf_out_str(stdout, 10, 0, mp_ln);
		printf("%25.17e %25.17e", mpf2double(mp_ln), log(x)/log(10.0));
		mpf_log10(mp_log10, mp_x);
//		mpf_out_str(stdout, 10, 0, mp_log10);
		printf("%25.17e\n", mpf2double(mp_log10));

		x += h;
		mpf_add(mp_x, mp_x, mp_h);
	}
	mpf_set_ui(mp_x, 100UL);
	printf("sin(%25.17e): ", mpf2double(mp_x));
	mpf_sin(mp_sin, mp_x);
	mpf_out_str(stdout, 10, 0, mp_sin);
	printf(" %25.17e", sin(mpf2double(mp_x)));
	printf("\n");
	printf("cos(%25.17e): ", mpf2double(mp_x));
	mpf_cos(mp_cos, mp_x);
	mpf_out_str(stdout, 10, 0, mp_cos);
	printf(" %25.17e", cos(mpf2double(mp_x)));
	printf("\n");
	printf("exp(%25.17e): ", mpf2double(mp_x));
	mpf_exp(mp_exp, mp_x);
	mpf_out_str(stdout, 10, 0, mp_exp);
	printf(" %25.17e", exp(mpf2double(mp_x)));
	printf("\n");
	printf("ln (%25.17e): ", mpf2double(mp_x));
	mpf_ln(mp_ln, mp_x);
	mpf_out_str(stdout, 10, 0, mp_ln);
	printf(" %25.17e", log(mpf2double(mp_x)));
	printf("\n");
	printf("log10(%25.17e): ", mpf2double(mp_x));
	mpf_log10(mp_log10, mp_x);
	mpf_out_str(stdout, 10, 0, mp_log10);
	printf(" %25.17e", log(mpf2double(mp_x))/log(10.0));
	printf("\n");

	printf("PI: ");
	mpf_pi(mp_pi);
	mpf_out_str(stdout, 10, 0, mp_pi);
	printf(" -> %25.17e", mpf2double(mp_pi));
	mpf_floor(mp_pi, mp_pi);
	printf(" floor(PI), floor(PI*10000):");
	mpf_out_str(stdout, 10, 0, mp_pi);
	mpf_pi(mp_pi); mpf_mul_ui(mp_pi, mp_pi, 10000UL); mpf_floor(mp_pi, mp_pi);
	printf(", "); mpf_out_str(stdout, 10, 0, mp_pi);
	fflush(stdout);
	printf("\n");
	printf("E : ");
	mpf_e(mp_e);
	mpf_out_str(stdout, 10, 0, mp_e);
	printf(" -> %25.17e", mpf2double(mp_e));
	mpf_floor(mp_e, mp_e);
	printf(" floor(E):");
	mpf_out_str(stdout, 10, 0, mp_e);
	printf("\n");
	printf("log 2 : ");
	mpf_ln_2(mp_ln2);
	mpf_out_str(stdout, 10, 0, mp_ln2);
	printf(" -> %25.17e", mpf2double(mp_ln2));
	mpf_floor(mp_ln2, mp_ln2);
	printf(" floor(ln2):");
	mpf_out_str(stdout, 10, 0, mp_ln2);
	printf("\n");

	mpf_clear(mp_h);
	mpf_clear(mp_sin);
	mpf_clear(mp_cos);
	mpf_clear(mp_exp);
	mpf_clear(mp_pi);
	mpf_clear(mp_e);
	mpf_clear(mp_ln2);
	mpf_clear(mp_ln);
	mpf_clear(mp_log10);
#endif
}
