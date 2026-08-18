/********************************************************************************/
/* test_interpol.c:                                                             */
/* Copyright (C) 2006-2011 Tomonori Kouya                                       */
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
/* Test Program for poly.c */

#include <stdio.h>
#include "bnc.h"

#define MAX_NEWTON_INTERPOL_DEG 1024

/**********************************************/
/* Newton Interpolaiton by Neville's Algorithm*/
/**********************************************/
void newton_interpol_dpoly(DPoly ret, double x[], double y[], long int max_num)
{
	DPoly f_ij[MAX_NEWTON_INTERPOL_DEG], tmp_f_ij[3], x_i[MAX_NEWTON_INTERPOL_DEG];
	long int i, j;

	if(max_num <= 0)
		return;

	/* Initialize */
	tmp_f_ij[0] = init_dpoly(max_num);
	tmp_f_ij[1] = init_dpoly(max_num);
	tmp_f_ij[2] = init_dpoly(max_num);
	for(i = 0; i < max_num; i++)
	{
		f_ij[i] = init_dpoly(max_num);
		x_i[i] = init_dpoly(max_num);
	}

	/* set f_0i := y[i] */
	for(i = 0; i < max_num; i++)
		set_dpoly_i(f_ij[i], 0, y[i]);

	/* set x_i := x - x[i] */
	for(i = 0; i < max_num; i++)
	{
		set_dpoly_i(x_i[i], 1, 1.0);
		set_dpoly_i(x_i[i], 0, -x[i]);
	}

	/* main loop */
	for(i = 1; i < max_num; i++)
	{
		for(j = 0; j < max_num - i; j++)
		{
			/* ((x - x[j]) * f_ij[j + 1] - (x - x[j+i]) * f_ij[j]) / (x[i + j] - x[j]) */
			mul_dpoly(tmp_f_ij[0], x_i[j], f_ij[j + 1]);
			mul_dpoly(tmp_f_ij[1], x_i[j + i], f_ij[j]);
			sub_dpoly(tmp_f_ij[2], tmp_f_ij[0], tmp_f_ij[1]);
			cmul2_dpoly(tmp_f_ij[2], 1.0 / (x[i+j] - x[j]));
			subst_dpoly(f_ij[j], tmp_f_ij[2]);

			printf("%d, %d: \n", i, j); print_dpoly(f_ij[j]);

			/* clean */
			set0_dpoly(tmp_f_ij[0]);
			set0_dpoly(tmp_f_ij[1]);
			set0_dpoly(tmp_f_ij[2]);
		}
	}

	/* p_{n-1}(x) = f_ij[0] */
	subst_dpoly(ret, f_ij[0]);

	/* clean */
	free_dpoly(tmp_f_ij[0]);
	free_dpoly(tmp_f_ij[1]);
	free_dpoly(tmp_f_ij[2]);
	for(i = 0; i < max_num; i++)
	{
		free_dpoly(f_ij[i]);
		free_dpoly(x_i[i]);
	}
}

#ifdef USE_GMP
/**********************************************/
/* Newton Interpolaiton by Neville's Algorithm*/
/**********************************************/
void newton_interpol_mpfpoly(MPFPoly ret, mpf_t x[], mpf_t y[], long int max_num)
{
	MPFPoly f_ij[MAX_NEWTON_INTERPOL_DEG], tmp_f_ij[3], x_i[MAX_NEWTON_INTERPOL_DEG];
	long int i, j;
	unsigned long prec;
	mpf_t tmp;

	if(max_num <= 0)
		return;

	prec = ret->prec;

	/* Initialize */
	mpf_init2(tmp, prec);
	tmp_f_ij[0] = init2_mpfpoly(max_num, prec);
	tmp_f_ij[1] = init2_mpfpoly(max_num, prec);
	tmp_f_ij[2] = init2_mpfpoly(max_num, prec);
	for(i = 0; i < max_num; i++)
	{
		f_ij[i] = init2_mpfpoly(max_num, prec);
		x_i[i] = init2_mpfpoly(max_num, prec);
	}

	/* set f_0i := y[i] */
	for(i = 0; i < max_num; i++)
		set_mpfpoly_i(f_ij[i], 0, y[i]);

	/* set x_i := x - x[i] */
	for(i = 0; i < max_num; i++)
	{
		set_mpfpoly_i_ui(x_i[i], 1, 1UL);
		mpf_neg(tmp, x[i]);
		set_mpfpoly_i(x_i[i], 0, tmp);
	}

	/* main loop */
	for(i = 1; i < max_num; i++)
	{
		for(j = 0; j < max_num - i; j++)
		{
			/* ((x - x[j]) * f_ij[j + 1] - (x - x[j+i]) * f_ij[j]) / (x[i + j] - x[j]) */
			mul_mpfpoly(tmp_f_ij[0], x_i[j], f_ij[j + 1]);
			mul_mpfpoly(tmp_f_ij[1], x_i[j + i], f_ij[j]);
			sub_mpfpoly(tmp_f_ij[2], tmp_f_ij[0], tmp_f_ij[1]);
			mpf_sub(tmp, x[i+j], x[j]);
			mpf_ui_div(tmp, 1UL, tmp);
			cmul2_mpfpoly(tmp_f_ij[2], tmp);
			subst_mpfpoly(f_ij[j], tmp_f_ij[2]);

			printf("%d, %d: \n", i, j); print_mpfpoly(f_ij[j]);

			/* clean */
			set0_mpfpoly(tmp_f_ij[0]);
			set0_mpfpoly(tmp_f_ij[1]);
			set0_mpfpoly(tmp_f_ij[2]);
		}
	}

	/* p_{n-1}(x) = f_ij[0] */
	subst_mpfpoly(ret, f_ij[0]);

	/* clean */
	mpf_clear(tmp);
	free_mpfpoly(tmp_f_ij[0]);
	free_mpfpoly(tmp_f_ij[1]);
	free_mpfpoly(tmp_f_ij[2]);
	for(i = 0; i < max_num; i++)
	{
		free_mpfpoly(f_ij[i]);
		free_mpfpoly(x_i[i]);
	}
}
#endif

#define MAX_POLY_LEN 4096
#define MAX_DEGREE 1024

int main()
{
	long int i, num_data;

	DPoly dinterpol;
	double dx[MAX_DEGREE], dy[MAX_DEGREE];
#ifdef USE_GMP
	MPFPoly mpf_interpol;
	mpf_t mpfx[MAX_DEGREE], mpfy[MAX_DEGREE];
	mpf_t mpf_tmp;

	/* switch double or mpf */
	printf("Which do you exec me in IEEE754 double or mpf (1: double, 2: mpf) ?\n");
	scanf("%ld", &num_data);
	if(num_data == 2)
		goto MPF;
#endif

/* double */
	/* init */
	dinterpol = init_dpoly(MAX_POLY_LEN);

	/* Input */
	printf("Number of Points: "); scanf("%ld", &num_data);
	if(num_data <= 0)
		return 0;
	printf("\n Please input the coodinates of points:\n");
	for(i = 0; i < num_data; i++)
	{
		printf("x[%2d], y[%2d] = ", i, i);
		scanf("%lf,%lf", &dx[i], &dy[i]);
	}
	for(i = 0; i < num_data; i++)
		printf("(x[%2d], y[%2d]) = (%g, %g)\n", i, i, dx[i], dy[i]);

	/* interpolation */
	newton_interpol_dpoly(dinterpol, dx, dy, num_data);

	/* print */
	printf("Interpolation Polynomial: \n");
	print_dpoly(dinterpol);

	/* check */
	printf("    No.       x[i]                  y[i]                   p_n(x[i])\n");
	for(i = 0; i < num_data; i++)
		printf("%5d: %15.7e %25.17e %25.17e\n", i, dx[i], dy[i], eval_dpoly(dinterpol, dx[i]));
	/* clear */
	free_dpoly(dinterpol);

	goto END;

/* mpf */
#ifdef USE_GMP
#define DPREC 50
MPF:

	set_bnc_default_prec_decimal(50);

	/* init */
	mpf_init(mpf_tmp);
	mpf_interpol = init_mpfpoly(MAX_POLY_LEN);

	/* Input */
	printf("Number of Points: "); scanf("%ld", &num_data);
	if(num_data <= 0)
		return 0;
	printf("\n Please input the coodinates of points:\n");
	for(i = 0; i < num_data; i++)
	{
		mpf_init(mpfx[i]); mpf_init(mpfy[i]);

		printf("x[%2d] = ", i);
		mpf_inp_str(mpfx[i], stdin, 10);
		printf("y[%2d] = ", i);
		mpf_inp_str(mpfy[i], stdin, 10);
	}
	for(i = 0; i < num_data; i++)
	{
		printf("(x[%2d], y[%2d]) = (", i, i);
		mpf_out_str(stdout, 10, 0, mpfx[i]);
		printf(", ");
		mpf_out_str(stdout, 10, 0, mpfy[i]);
		printf(")\n");
	}

	/* interpolation */
	newton_interpol_mpfpoly(mpf_interpol, mpfx, mpfy, num_data);

	/* print */
	printf("Interpolation Polynomial: \n");
	print_mpfpoly(mpf_interpol);

	/* check */
	printf("    No.       x[i]                  y[i]                   p_n(x[i])\n");
	for(i = 0; i < num_data; i++)
	{
		eval_mpfpoly(mpf_tmp, mpf_interpol, mpfx[i]);
		printf("%5d: %15.7e %25.17e %25.17e\n", i, mpf2double(mpfx[i]), mpf2double(mpfy[i]), mpf2double(mpf_tmp));
	}

	/* clear */
	mpf_clear(mpf_tmp);
	for(i = 0; i < num_data; i++)
	{
		mpf_clear(mpfx[i]);
		mpf_clear(mpfy[i]);
	}
	free_mpfpoly(mpf_interpol);

#endif

END:
	return 0;

}
