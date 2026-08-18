/********************************************************************************/
/* Test Program for Numerical Differential based on Extrapolation               */
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

double dfunc1(double x)
{
	return sin(x);
}

double dfunc1_ans(double x)
{
	return cos(x);
}

#ifdef USE_GMP
void mpffunc1(mpf_t ret, mpf_t x)
{
	mpf_sin(ret, x);

	return;
}

void mpffunc1_ans(mpf_t ret, mpf_t x)
{
	mpf_cos(ret, x);

	return;
}
#endif

#define FNM_DIM 10

main()
{
	long int i, num_stage;
	double drel_tol, dabs_tol;
	double dx, dret_value;
#ifdef USE_GMP
	mpf_t mpfrel_tol, mpfabs_tol;
	mpf_t mpfx, mpfret_value;
	mpf_t mpf_tmp, mpf_rerr;
#endif

	dabs_tol = 0.0;
	drel_tol = 0.0;

	for(i = -10; i <= 10; i++)
	{
		dx = M_PI / (double)i;
	
		dret_value = dfnmdiff(dx, dfunc1, 1.0, drel_tol, dabs_tol, 20, &num_stage);
		printf("%25.17e %25.17e (%10.3e)\n", dx, dret_value, drelerr(dret_value, dfunc1_ans(dx)) );
	}

#ifdef USE_GMP
	set_bnc_default_prec_decimal(50);

	mpf_init(mpfabs_tol);
	mpf_init(mpfrel_tol);
	mpf_init(mpfx);
	mpf_init(mpfret_value);
	mpf_init(mpf_tmp);
	mpf_init(mpf_rerr);

	mpf_set_d(mpfabs_tol, 0.0);
	mpf_set_d(mpfrel_tol, 0.0);

	for(i = -10; i <= 10; i++)
	{
		mpf_pi(mpfx);
		if(i == 0)
			continue;

		if(i < 0)
		{
			mpf_div_ui(mpfx, mpfx, -i);
			mpf_neg(mpfx, mpfx);
		}
		else
			mpf_div_ui(mpfx, mpfx, i);

		mpf_set_ui(mpf_tmp, 1UL); // Initial Stepsize
		mpffnmdiff(mpfret_value, mpfx, mpffunc1, mpf_tmp, mpfrel_tol, mpfabs_tol, 20, &num_stage);
		mpffunc1_ans(mpf_tmp, mpfx);
		mpfrelerr(mpf_rerr, mpfret_value, mpf_tmp);
		mpf_out_str(stdout, 10, 17, mpfx);
		printf(" ");
		mpf_out_str(stdout, 10, 17, mpfret_value);
		printf(" (");
		mpf_out_str(stdout, 10, 4, mpf_rerr);
		printf(", ");
		mpf_out_str(stdout, 10, 4, mpf_tmp);
		printf(")\n");
	}

	mpf_clear(mpfabs_tol);
	mpf_clear(mpfrel_tol);
	mpf_clear(mpfx);
	mpf_clear(mpfret_value);
	mpf_clear(mpf_tmp);
	mpf_clear(mpf_rerr);
#endif
}
