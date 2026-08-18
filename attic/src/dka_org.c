/********************************************************************************/
/* dka.c: Durand-Kerner-Aberth Methods                                          */
/* Copyright (C) 2003-2025 Tomonori Kouya                                       */
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
//#include "bnc.h"
#include <stdio.h>
#include <math.h>

#include "poly.h" // Polynomial, array and DKA

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/* float */

/* center of gravity */
/* center = -a_{n-1} / (n * a_n) */
float fdka_center(FPoly func)
{
	float ret;

	ret = get_fpoly_i(func, func->deg - 1);
	ret /= get_fpoly_i(func, func->deg);
	ret /= (float)func->deg;
	ret = -ret;

	return ret;

}
/* radius = max (m * |a_i / a_deg|)^(1/(deg-i)) */
float fdka_radius(FPoly func)
{
	long int i;
	float ret, tmp, num_nonzero, an;

	num_nonzero = num_nonzero_fpoly(func);
	an = get_fpoly_i(func, func->deg);
	ret = num_nonzero;
	for(i = func->deg - 1; i >= 0; i--)
	{
		tmp = get_fpoly_i(func, i) / an;
		tmp *= num_nonzero;
		tmp = (float)fabs((double)tmp);
		tmp = (float)pow((double)tmp, 1.0/(double)(func->deg - i));
		if(ret < tmp)
			ret = tmp;
	}

	return ret;
}

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void fdka_init(CFArray x_init, FPoly func)
{
	long int i, itmp;
	float rad, cen, an, tmp, re_cinit, im_cinit;
	FCmplx cinit;

	rad = fdka_radius(func);
	cen = fdka_center(func);

//	printf("%f, %f\n", rad, cen);

	cinit = init_fcmplx();
	for(i = 0; i < func->deg; i++)
	{
		set0_fcmplx(cinit);
		tmp = (float)(2.0 * M_PI * i / func->deg + 3.0 / (2.0 * func->deg));
		iexp_fcmplx(cinit, tmp);
		re_cinit = get_real_fcmplx(cinit);
		im_cinit = get_image_fcmplx(cinit);

		re_cinit = cen + rad * re_cinit;
		im_cinit = rad * im_cinit;

		set_real_fcmplx(cinit, re_cinit);
		set_image_fcmplx(cinit, im_cinit);

//		printf("%5d(%f) ", i, abs_fcmplx(cinit)); print_fcmplx(cinit);

		set_cfarray_i(x_init, i, cinit);
	}
}

/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int fdka(CFArray ans, CFArray x_init, FPoly func, long int maxtimes, float abs_eps, float rel_eps)
{
	long int times, i, j, deg, flag;
	float absmodval, abs_x, abs_newx;
	FCmplx modval, up_modval, low_modval, tmp;

//	printf("\n");
//	print_cfarray(x_init);
//	printf("%f, %f\n", abs_eps, rel_eps);

	deg = ans->size;

	modval = init_fcmplx();
	low_modval = init_fcmplx();
	up_modval = init_fcmplx();
	tmp = init_fcmplx();
	for(times = 0; times <= maxtimes; times++)
	{
//		printf("%5d: ", times);

		flag = 0;
		for(i = 0; i < deg; i++)
		{
			set_real_fcmplx(low_modval, 1.0);
			set_image_fcmplx(low_modval, 0.0);
			for(j = 0; j < i; j++)
			{
				set0_fcmplx(tmp);
				sub_fcmplx(
					tmp,
					get_cfarray_i(x_init, i),
					get_cfarray_i(x_init, j)
				);
				mul2_fcmplx(low_modval, tmp);
			}
			for(j = i + 1; j < deg; j++)
			{
				set0_fcmplx(tmp);
				sub_fcmplx(
					tmp,
					get_cfarray_i(x_init, i),
					get_cfarray_i(x_init, j)
				);
				mul2_fcmplx(low_modval, tmp);
			}
			mul_fcmplx_f(low_modval, low_modval, get_fpoly_i(func, func->deg));
			ceval_fpoly(up_modval, func, get_cfarray_i(x_init, i));

			div_fcmplx(modval, up_modval, low_modval);
			sub_fcmplx(tmp, get_cfarray_i(x_init, i), modval);
			set_cfarray_i(ans, i, tmp);
//			printf("%15.7e, ", get_real_fcmplx(get_cfarray_i(ans, i)));

			/* check convergence */
			absmodval = fabs((double)abs_fcmplx(modval));
			abs_x = fabs((double)abs_fcmplx(get_cfarray_i(x_init, i)));
			abs_newx = fabs((double)abs_fcmplx(get_cfarray_i(ans, i)));
			if( absmodval > (abs_x + abs_newx) * rel_eps + abs_eps )
				flag = 1;

		}
//		printf("\n");

		/* check convergence */
		if(flag == 0)
			break;

		subst_cfarray(x_init, ans);

	}

	return times;
}

/* double */
/* center of gravity */
/* center = -a_{n-1} / (n * a_n) */
double ddka_center(DPoly func)
{
	double ret;

	ret = get_dpoly_i(func, func->deg - 1);
	ret /= get_dpoly_i(func, func->deg);
	ret /= func->deg;
	ret = -ret;

	return ret;

}
/* radius = max (m * |a_i / a_deg|)^(1/(deg-i)) */
double ddka_radius(DPoly func)
{
	long int i;
	double ret, tmp, num_nonzero, an;

	num_nonzero = num_nonzero_dpoly(func);
	an = get_dpoly_i(func, func->deg);
	ret = num_nonzero;
	for(i = func->deg - 1; i >= 0; i--)
	{
		tmp = get_dpoly_i(func, i) / an;
		tmp *= num_nonzero;
		tmp = fabs(tmp);
		tmp = pow(tmp, 1.0/(func->deg - i));
		if(ret < tmp)
			ret = tmp;
	}

	return ret;
}

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void ddka_init(CDArray x_init, DPoly func)
{
	long int i, itmp;
	double rad, cen, an, tmp, re_cinit, im_cinit;
	DCmplx cinit;

	rad = ddka_radius(func);
	cen = ddka_center(func);

//	printf("%f, %f\n", rad, cen);

	cinit = init_dcmplx();
	for(i = 0; i < func->deg; i++)
	{
		set0_dcmplx(cinit);
		tmp = (double)(2.0 * M_PI * i / func->deg + 3.0 / (2.0 * func->deg));
		iexp_dcmplx(cinit, tmp);
		re_cinit = get_real_dcmplx(cinit);
		im_cinit = get_image_dcmplx(cinit);

		re_cinit = cen + rad * re_cinit;
		im_cinit = rad * im_cinit;

		set_real_dcmplx(cinit, re_cinit);
		set_image_dcmplx(cinit, im_cinit);

//		printf("%5d(%f) ", i, abs_dcmplx(cinit)); print_dcmplx(cinit);

		set_cdarray_i(x_init, i, cinit);
	}
}

/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int ddka(CDArray ans, CDArray x_init, DPoly func, long int maxtimes, double abs_eps, double rel_eps)
{
	long int times, i, j, deg, flag;
	double absmodval, abs_x, abs_newx;
	DCmplx modval, up_modval, low_modval, tmp;

//	printf("\n");
//	print_cfarray(x_init);
//	printf("%f, %f\n", abs_eps, rel_eps);

	deg = ans->size;

	modval = init_dcmplx();
	low_modval = init_dcmplx();
	up_modval = init_dcmplx();
	tmp = init_dcmplx();
	for(times = 0; times <= maxtimes; times++)
	{
//		printf("%5d: ", times);

		flag = 0;
		for(i = 0; i < deg; i++)
		{
			set_real_dcmplx(low_modval, 1.0);
			set_image_dcmplx(low_modval, 0.0);
			for(j = 0; j < i; j++)
			{
				set0_dcmplx(tmp);
				sub_dcmplx(
					tmp,
					get_cdarray_i(x_init, i),
					get_cdarray_i(x_init, j)
				);
				mul2_dcmplx(low_modval, tmp);
			}
			for(j = i + 1; j < deg; j++)
			{
				set0_dcmplx(tmp);
				sub_dcmplx(
					tmp,
					get_cdarray_i(x_init, i),
					get_cdarray_i(x_init, j)
				);
				mul2_dcmplx(low_modval, tmp);
			}
			mul_dcmplx_d(low_modval, low_modval, get_dpoly_i(func, func->deg));
			ceval_dpoly(up_modval, func, get_cdarray_i(x_init, i));

			div_dcmplx(modval, up_modval, low_modval);
			sub_dcmplx(tmp, get_cdarray_i(x_init, i), modval);
			set_cdarray_i(ans, i, tmp);
//			printf("%25.17e, ", get_real_dcmplx(get_cfarray_i(ans, i)));

			/* check convergence */
			absmodval = abs_dcmplx(modval);
			abs_x = abs_dcmplx(get_cdarray_i(x_init, i));
			abs_newx = abs_dcmplx(get_cdarray_i(ans, i));
			if( absmodval > (abs_x + abs_newx) * rel_eps + abs_eps )
				flag = 1;

		}
//		printf("\n");

		/* check convergence */
		if(flag == 0)
			break;

		subst_cdarray(x_init, ans);
//		printf("%ld: ", times); print_cdarray(x_init);

	}

	return times;
}

#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus
