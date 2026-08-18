/********************************************************************************/
/* dka.c: Durand-Kerner-Aberth Methods                                          */
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
#include "bnc.h"
#include <stdio.h>
#include <math.h>

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

#ifdef USE_GMP

/* mpf_t */

/* center of gravity */
/* center = -a_{n-1} / (n * a_n) */
void mpf_dka_center(mpf_t ret, MPFPoly func)
{
	mpf_set(ret, get_mpfpoly_i(func, func->deg - 1));

	mpf_div(ret, ret, get_mpfpoly_i(func, func->deg));
	mpf_div_ui(ret, ret, (unsigned long)func->deg);
	mpf_neg(ret, ret);
}

/* radius = max (m * |a_i / a_deg|)^(1/(deg-i)) */
void mpf_dka_radius(mpf_t ret, MPFPoly func)
{
	long int i;
	mpf_t tmp, num_nonzero, an;
	double dtmp;

	mpf_init2(tmp, mpf_get_prec(ret));
	mpf_init2(num_nonzero, mpf_get_prec(ret));
	mpf_init2(an, mpf_get_prec(ret));

	mpf_set_ui(num_nonzero, (unsigned long)num_nonzero_mpfpoly(func));
	mpf_set(an, get_mpfpoly_i(func, func->deg));
	mpf_set(ret, num_nonzero);
	for(i = func->deg - 1; i >= 0; i--)
	{
		mpf_div(tmp, get_mpfpoly_i(func, i), an);
		mpf_mul(tmp, tmp, num_nonzero);
		mpf_abs(tmp, tmp);
		
		/* tmp^(deg-i) */
/*		mpf_ln(tmp, tmp);
		mpf_mul_ui(tmp, tmp, (unsigned long)func->deg - i);
		mpf_exp(tmp, tmp);
*/
		dtmp = mpf_get_d(tmp); dtmp = pow(dtmp, 1.0/(double)(func->deg - i)); mpf_set_d(tmp, dtmp);
		if(mpf_cmp(ret, tmp) < 0)
			mpf_set(ret, tmp);
	}
	mpf_clear(tmp);
	mpf_clear(num_nonzero);
	mpf_clear(an);
}

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void mpf_dka_init(CMPFArray x_init, MPFPoly func)
{
	long int i, itmp;
	mpf_t rad, cen, an, tmp, re_cinit, im_cinit;
	MPFCmplx cinit;

	mpf_init2(rad, x_init->prec);
	mpf_init2(cen, x_init->prec);
	mpf_init2(an , x_init->prec);
	mpf_init2(tmp, x_init->prec);
	mpf_init2(re_cinit, x_init->prec);
	mpf_init2(im_cinit, x_init->prec);

	mpf_dka_radius(rad, func);
	mpf_dka_center(cen, func);

//	mpf_out_str(stdout, 10, 0, rad); printf(", "); mpf_out_str(stdout, 10, 0, cen); printf("\n");

	cinit = init_mpfcmplx();
	for(i = 0; i < func->deg; i++)
	{
		set0_mpfcmplx(cinit);
#ifndef USE_MPFR
		mpf_set_d(tmp, (double)(2.0 * M_PI * i / func->deg + 3.0 / (2.0 * func->deg)));
#else
		mpfr_set_d(tmp, (double)(2.0 * M_PI * i / func->deg + 3.0 / (2.0 * func->deg)), bnc_default_rounding_mode);
#endif
		iexp_mpfcmplx(cinit, tmp);
		get_real_mpfcmplx(re_cinit, cinit);
		get_image_mpfcmplx(im_cinit, cinit);

		/* re_cinit = cen + rad * re_cinit */
		mpf_mul(re_cinit, rad, re_cinit);
		mpf_add(re_cinit, cen, re_cinit);

		/* im_cinit = rad * im_cinit */
		mpf_mul(im_cinit, rad, im_cinit);

		set_real_mpfcmplx(cinit, re_cinit);
		set_image_mpfcmplx(cinit, im_cinit);

		abs_mpfcmplx(tmp, cinit);
//		printf("%5d(%f) ", i, mpf2double(tmp)); print_mpfcmplx(cinit);

		set_cmpfarray_i(x_init, i, cinit);
	}

	mpf_clear(rad);
	mpf_clear(cen);
	mpf_clear(an);
	mpf_clear(tmp);
	mpf_clear(re_cinit);
	mpf_clear(im_cinit);
}

/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int mpf_dka(CMPFArray ans, CMPFArray x_init, MPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps)
{
	long int times, i, j, deg, flag;
	mpf_t absmodval, abs_x, abs_newx, mpftmp;
	MPFCmplx modval, up_modval, low_modval, tmp;

//	printf("\n");
//	print_cfarray(x_init);
//	printf("%f, %f\n", abs_eps, rel_eps);

	deg = ans->size;

	mpf_init2(absmodval, ans->prec);
	mpf_init2(abs_x, ans->prec);
	mpf_init2(abs_newx, ans->prec);
	mpf_init2(mpftmp, ans->prec);

	modval = init_mpfcmplx();
	low_modval = init_mpfcmplx();
	up_modval = init_mpfcmplx();
	tmp = init_mpfcmplx();
	for(times = 0; times <= maxtimes; times++)
	{
//		printf("%5d: ", times); fflush(stdout);

		flag = 0;
		for(i = 0; i < deg; i++)
		{
			set_real_mpfcmplx_ui(low_modval, 1UL);
			set_image_mpfcmplx_ui(low_modval, 0UL);
			for(j = 0; j < i; j++)
			{
				set0_mpfcmplx(tmp);
				sub_mpfcmplx(
					tmp,
					get_cmpfarray_i(x_init, i),
					get_cmpfarray_i(x_init, j)
				);
				mul2_mpfcmplx(low_modval, tmp);
			}
			for(j = i + 1; j < deg; j++)
			{
				set0_mpfcmplx(tmp);
				sub_mpfcmplx(
					tmp,
					get_cmpfarray_i(x_init, i),
					get_cmpfarray_i(x_init, j)
				);
				mul2_mpfcmplx(low_modval, tmp);
			}
			mul_mpfcmplx_mpf(low_modval, low_modval, get_mpfpoly_i(func, func->deg));
			ceval_mpfpoly(up_modval, func, get_cmpfarray_i(x_init, i));

			div_mpfcmplx(modval, up_modval, low_modval);
			sub_mpfcmplx(tmp, get_cmpfarray_i(x_init, i), modval);
			set_cmpfarray_i(ans, i, tmp);

//			get_real_mpfcmplx(mpftmp, get_cmpfarray_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			abs_mpfcmplx(absmodval, modval);
			abs_mpfcmplx(abs_x, get_cmpfarray_i(x_init, i));
			abs_mpfcmplx(abs_newx, get_cmpfarray_i(ans, i));

			mpf_add(mpftmp, abs_x, abs_newx);
			mpf_mul(mpftmp, mpftmp, rel_eps);
			mpf_add(mpftmp, mpftmp, abs_eps);
			if( mpf_cmp(absmodval, mpftmp) > 0 )
				flag = 1;

		}

		/* check convergence */
		if(flag == 0)
			break;

		subst_cmpfarray(x_init, ans);

	}

	mpf_clear(absmodval);
	mpf_clear(abs_x);
	mpf_clear(abs_newx);
	mpf_clear(mpftmp);

	printf("ans->prec: %ld\n", ans->prec);
	return times;
}

#endif
