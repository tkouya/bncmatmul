/********************************************************************************/
/* test_eps.c:                                                                  */
/* Copyright (C) 2003 Tomonori Kouya                                            */
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
#include <gmp.h>

double get_mepsilon(void)
{
	double ret, one;

	one = (double)1;
	ret = one;

	/* if(one + ret == one) ret *= 2; */
	while((one + ret) > one)
		ret /= 2.0;

	ret *= 2.0;

	return ret;
}

#ifdef USE_GMP
void get_mepsilon_mpf(mpf_t ret)
{
	mpf_t one, tmp;

	mpf_init2(one, mpf_get_prec(ret));
	mpf_init2(tmp, mpf_get_prec(ret));

	mpf_set_ui(one, 1UL);
	mpf_set(ret, one);

	/* if(one + ret == one) ret *= 2 */
	do{
		mpf_div_ui(ret, ret, 2UL);
		mpf_add(tmp, one, ret);
	}
	while(mpf_cmp(tmp, one) > 0);

	mpf_mul_ui(ret, ret, 2UL);

	mpf_clear(tmp);
	mpf_clear(one);
}
#endif

#define PREC 128
main()
{
	double deps;
#ifdef USE_GMP
	mpf_t mpf_eps;
#endif
	unsigned long prec;

	deps = get_mepsilon();

	printf("IEEE754 double(1 + eps): %25.17e(%25.17e)\n", deps, deps + 1);
#ifdef USE_GMP
	for(prec = PREC; prec <= 16384; prec *= 2)
	{
	mpf_init2(mpf_eps, prec); get_mepsilon_mpf(mpf_eps);
	printf("mpf_t   %6d(1 + eps):", prec);
	mpf_out_str(stdout, 10, 0, mpf_eps);
	printf("(");
	mpf_add_ui(mpf_eps, mpf_eps, 1UL);
	mpf_out_str(stdout, 10, 0, mpf_eps);
	printf(")\n");
	mpf_clear(mpf_eps);
	}
#endif
}

