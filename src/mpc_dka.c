/********************************************************************************/
/* mpc_dka.c: Durand-Kerner-Aberth Methods for complex polynomial               */
/* Copyright (C) 2025 Tomonori Kouya                                            */
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

#ifdef USE_GMP
/* mpc_t */

#define DRND (MPC_RNDNN)

/* center of gravity */
/* center = -a_{n-1} / (n * a_n) */
void mpc_dka_center(mpc_t ret, CMPFPoly func)
{
	mpc_set(ret, get_cmpfpoly_i(func, func->deg - 1), DRND);

	mpc_div(ret, ret, get_cmpfpoly_i(func, func->deg), DRND);
	mpc_div_ui(ret, ret, (unsigned long)func->deg, DRND);
	mpc_neg(ret, ret, DRND);
}

/* radius = max (m * |a_i / a_deg|)^(1/(deg-i)) */
void mpc_dka_radius(mpf_t ret, CMPFPoly func)
{
	long int i;
	mpf_t tmp, num_nonzero;
    mpc_t ctmp, an;
	double dtmp;

	mpf_init2(tmp, mpf_get_prec(ret));
	mpf_init2(num_nonzero, mpf_get_prec(ret));

    mpc_init2(ctmp, mpf_get_prec(ret));
    mpc_init2(an, mpf_get_prec(ret));

    mpf_set_ui(num_nonzero, (unsigned long)num_nonzero_cmpfpoly(func));
	mpc_set(an, get_cmpfpoly_i(func, func->deg), DRND);
	mpf_set(ret, num_nonzero);
	for(i = func->deg - 1; i >= 0; i--)
	{
		mpc_div(ctmp, get_cmpfpoly_i(func, i), an, DRND);
        mpc_abs(tmp, ctmp, DRND);
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

    mpc_clear(ctmp);
	mpc_clear(an);
}

// ------------------------------------
// New implementation
// 2025-01-15 (Wed) T.Kouya
// ------------------------------------

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void mpc_dka_init(CMPFArray x_init, CMPFPoly func)
//void mpc_dka_init(mpc_t x_init[], CMPFPoly func)
{
	long int i, itmp;
	mpf_t rad, tmp, re_cinit, im_cinit;
	//MPFCmplx cinit;
	mpf_t cosine, sine;
	mpc_t cinit, cen, an;

	mpf_init2(rad, x_init->prec);
    mpf_init2(tmp, x_init->prec);
	mpf_init2(re_cinit, x_init->prec);
	mpf_init2(im_cinit, x_init->prec);
	mpf_init2(cosine, x_init->prec);
	mpf_init2(sine, x_init->prec);

    //cinit = init_mpfcmplx();
	mpc_init2(cinit, x_init->prec);
	mpc_init2(an, x_init->prec);
	mpc_init2(cen, x_init->prec);

    mpc_dka_radius(rad, func);
	mpc_dka_center(cen, func);

	//mpf_out_str(stdout, 10, 0, rad); printf(", "); mpc_out_str(stdout, 10, 0, cen, get_bnc_default_rounding_mode_c()); printf("\n");

	for(i = 0; i < func->deg; i++)
	{
		//set0_mpfcmplx(cinit);
		mpc_set_ui(cinit, 0UL, get_bnc_default_rounding_mode_c());
		mpf_set_d(tmp, (double)(2.0 * M_PI * i / func->deg + 3.0 / (2.0 * func->deg)));

        //iexp_mpfcmplx(cinit, tmp);
		// ctmp := i * tmp -> exp(i * tmp) := cos(tmp) + i * sin(tmp);
		mpf_cos(cosine, tmp);
		mpf_sin(sine, tmp);
		mpc_set_fr_fr(cinit, cosine, sine, get_bnc_default_rounding_mode_c());

		//get_real_mpfcmplx(re_cinit, cinit);
		//get_image_mpfcmplx(im_cinit, cinit);
		mpf_set(re_cinit, mpc_realref(cinit));
		mpf_set(im_cinit, mpc_imagref(cinit));

		/* re_cinit = cen + rad * re_cinit */
		mpf_mul(re_cinit, rad, re_cinit);
		mpf_add(re_cinit, mpc_realref(cen), re_cinit);

		/* im_cinit = cen + rad * im_cinit */
		mpf_mul(im_cinit, rad, im_cinit);
		mpf_add(im_cinit, mpc_imagref(cen), im_cinit);

		//set_real_mpfcmplx(cinit, re_cinit);
		//set_image_mpfcmplx(cinit, im_cinit);
		mpc_set_fr_fr(cinit, re_cinit, im_cinit, get_bnc_default_rounding_mode_c());

		//abs_mpfcmplx(tmp, cinit);
		mpc_abs(tmp, cinit, get_bnc_default_rounding_mode_c());
//		printf("%5d(%f) ", i, mpf2double(tmp)); print_mpfcmplx(cinit);

		set_cmpfarray_i(x_init, i, cinit);
        //mpc_set(x_init[i], cinit, get_bnc_default_rounding_mode_c());
	}

	mpf_clear(rad);
	mpf_clear(tmp);
	mpf_clear(re_cinit);
	mpf_clear(im_cinit);
	mpf_clear(cosine);
	mpf_clear(sine);

	mpc_clear(cinit);
	mpc_clear(cen);
	mpc_clear(an);
}

/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int mpc_dka(CMPFArray ans, CMPFArray x_init, CMPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps)
//long int mpc_dka(mpc_t ans[], CMPFArray x_init, CMPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps)
{
	long int times, i, j, deg, flag;
	mpf_t absmodval, abs_x, abs_newx, mpftmp;
	//MPFCmplx modval, up_modval, low_modval, tmp;
	mpc_t modval, up_modval, low_modval, tmp;

//	printf("\n");
//	print_cfarray(x_init);
//	printf("%f, %f\n", abs_eps, rel_eps);

	deg = ans->size;
    //deg = func->deg;

	mpf_init2(absmodval, ans->prec);
	mpf_init2(abs_x, ans->prec);
	mpf_init2(abs_newx, ans->prec);
	mpf_init2(mpftmp, ans->prec);

	/*
	modval = init_mpfcmplx();
	low_modval = init_mpfcmplx();
	up_modval = init_mpfcmplx();
	tmp = init_mpfcmplx();
	*/
	mpc_init2(modval, ans->prec); // = init_mpfcmplx();
	mpc_init2(low_modval, ans->prec); // = init_mpfcmplx();
	mpc_init2(up_modval, ans->prec); // = init_mpfcmplx();
	mpc_init2(tmp, ans->prec); // = init_mpfcmplx();

	for(times = 0; times <= maxtimes; times++)
	{
//		printf("%5d: ", times); fflush(stdout);

		flag = 0;
		for(i = 0; i < deg; i++)
		{
			//set_real_mpfcmplx_ui(low_modval, 1UL);
			//set_image_mpfcmplx_ui(low_modval, 0UL);
			mpc_set_ui_ui(low_modval, 1UL, 0UL, get_bnc_default_rounding_mode_c());

			for(j = 0; j < i; j++)
			{
				//set0_mpfcmplx(tmp);
				mpc_set_ui(tmp, 0UL, get_bnc_default_rounding_mode_c());
				//sub_mpfcmplx(
				mpc_sub(
					tmp,
					get_cmpfarray_i(x_init, i),
					get_cmpfarray_i(x_init, j),
					get_bnc_default_rounding_mode_c()
				);
				//mul2_mpfcmplx(low_modval, tmp);
				mpc_mul(low_modval, low_modval, tmp, get_bnc_default_rounding_mode_c());
			}
			for(j = i + 1; j < deg; j++)
			{
				//set0_mpfcmplx(tmp);
				mpc_set_ui(tmp, 0UL, get_bnc_default_rounding_mode_c());
				//sub_mpfcmplx(
				mpc_sub(
					tmp,
					get_cmpfarray_i(x_init, i),
					get_cmpfarray_i(x_init, j),
					get_bnc_default_rounding_mode_c()
				);
				//mul2_mpfcmplx(low_modval, tmp);
				mpc_mul(low_modval, low_modval, tmp, get_bnc_default_rounding_mode_c());
			}
			//mul_mpfcmplx_mpf(low_modval, low_modval, get_cmpfpoly_i(func, func->deg));
			mpc_mul(low_modval, low_modval, get_cmpfpoly_i(func, func->deg), get_bnc_default_rounding_mode_c());
			eval_cmpfpoly(up_modval, func, get_cmpfarray_i(x_init, i));

			//div_mpfcmplx(modval, up_modval, low_modval);
			mpc_div(modval, up_modval, low_modval, get_bnc_default_rounding_mode_c());
			//sub_mpfcmplx(tmp, get_cmpfarray_i(x_init, i), modval);
			mpc_sub(tmp, get_cmpfarray_i(x_init, i), modval, get_bnc_default_rounding_mode_c());
			set_cmpfarray_i(ans, i, tmp);

//			get_real_mpfcmplx(mpftmp, get_cmpfarray_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			//abs_mpfcmplx(absmodval, modval);
			//abs_mpfcmplx(abs_x, get_cmpfarray_i(x_init, i));
			//abs_mpfcmplx(abs_newx, get_cmpfarray_i(ans, i));
			mpc_abs(absmodval, modval, get_bnc_default_rounding_mode_c());
			mpc_abs(abs_x, get_cmpfarray_i(x_init, i), get_bnc_default_rounding_mode_c());
			mpc_abs(abs_newx, get_cmpfarray_i(ans, i), get_bnc_default_rounding_mode_c());

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

    mpc_clear(modval);
	mpc_clear(low_modval);
	mpc_clear(up_modval); 
	mpc_clear(tmp);

	printf("ans->prec: %ld\n", ans->prec);
	return times;
}

#if 0
/* find maximum degree at pol->coef[deg] != 0 */
long int setdegree_cmpfpoly(CMPFPoly pol)
{
	long int i;

	for(i = pol->max_len - 1; i > 0; i--)
	{
		if(mpc_cmp(get_cmpfpoly_i(pol, i), pol->zero) != 0)
		{
			pol->deg = i;
			return i;
		}
	}

	pol->deg = 0;
	return 0;

}
#endif // 0

// deflation from highest degree coef
// ret org(x) / (x - alpha)
/* void deflation_cmpfpoly(CMPFPoly ret, CMPFPoly org, mpc_t alpha)
{
	long int i;
	mpc_t tmp;

	mpc_init2(tmp, ret->prec);

	set_cmpfpoly_i(ret, org->deg - 1, get_cmpfpoly_i(org, org->deg));
	for(i = org->deg - 2; i >= 0; i--)
	{
		mpc_mul(tmp, get_cmpfpoly_i(ret, i + 1), alpha, DRND);
		mpc_add(tmp, tmp, get_cmpfpoly_i(org, i + 1), DRND);
		set_cmpfpoly_i(ret, i, tmp);
	}

	setdegree_cmpfpoly(ret);

	mpc_clear(tmp);
}*/

// deflation from highest degree coef and evaluation
// ret org(x) / (x - alpha)
void eval_deflation_cmpfpoly(mpc_t ret, CMPFPoly org, mpc_t alpha, mpc_t x)
{
	unsigned long prec;
	long int i;
	mpc_t tmp, new_coef_i;

	prec = mpc_get_prec(ret);

	mpc_init2(tmp, prec);
	mpc_init2(new_coef_i, prec);

	//set_cmpfpoly_i(new_coef_i, org->deg - 1, get_cmpfpoly_i(org, org->deg));
	//set_cmpfpoly_i(ret, org->deg - 1, get_cmpfpoly_i(org, org->deg));
	mpc_set(new_coef_i, get_cmpfpoly_i(org, org->deg), DRND);
	mpc_set(ret, get_cmpfpoly_i(org, org->deg), DRND);
	for(i = org->deg - 2; i >= 0; i--)
	{
		mpc_mul(ret, ret, x, DRND);
		//mpc_mul(tmp, get_cmpfpoly_i(ret, i + 1), alpha);
		mpc_mul(tmp, new_coef_i, alpha, DRND);
		mpc_add(new_coef_i, tmp, get_cmpfpoly_i(org, i + 1), DRND);
		//set_cmpfpoly_i(ret, i, new_coef_i);
		mpc_add(ret, ret, new_coef_i, DRND);
	}

	mpc_clear(tmp);
	mpc_clear(new_coef_i);
}

// Ozawa's initial radius : r := (p(center) / a_n)^(1/n)
void mpc_dka_ozawa_radius(mpf_t ret, CMPFPoly func)
{
	unsigned long prec;
	mpc_t barycentric_point, ctmp;
	CMPFPoly in_func[2];

	prec = mpf_get_prec(ret);
	mpc_init2(barycentric_point, prec);
    mpc_init2(ctmp, prec);
	in_func[0] = init2_cmpfpoly(func->max_len, prec);
	in_func[1] = init2_cmpfpoly(func->max_len, prec);

	subst_cmpfpoly(in_func[0], func);

	do{
		mpc_dka_center(barycentric_point, in_func[0]);
		eval_cmpfpoly(ctmp, in_func[0], barycentric_point);

		mpc_div(ctmp, ctmp, get_cmpfpoly_i(in_func[0], in_func[0]->deg), DRND);
		mpc_abs(ret, ctmp, DRND);
		mpfr_rootn_ui(ret, ret, (unsigned long)(in_func[0]->deg), GMP_RNDN);

		// normal return
		if(mpfr_regular_p(ret) != 0)
			break;

		deflation_cmpfpoly(in_func[1], in_func[0], barycentric_point);
		subst_cmpfpoly(in_func[0], in_func[1]);
	} while(1);

	mpc_clear(barycentric_point);
	free_cmpfpoly(in_func[0]);
	free_cmpfpoly(in_func[1]);
}

// ------------------------------------
// New implementation
// 2025-01-16(Thu)
// ------------------------------------

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void mpc_dka_init2(CMPFArray x_init, CMPFPoly func, void (* get_radius)(mpf_t, CMPFPoly), void (* get_center)(mpc_t, CMPFPoly))
{
	long int i, itmp;
	mpf_t rad, an, tmp, re_cinit, im_cinit;
    mpf_t cosine, sine;
	//MPFCmplx cinit;
	mpc_t cen, cinit;

	mpf_init2(rad, x_init->prec);
	mpf_init2(an , x_init->prec);
	mpf_init2(tmp, x_init->prec);
	mpf_init2(re_cinit, x_init->prec);
	mpf_init2(im_cinit, x_init->prec);
    mpf_init2(cosine, x_init->prec);
	mpf_init2(sine, x_init->prec);

    mpc_init2(cen, x_init->prec);

	//mpc_dka_radius(rad, func);
	//mpc_dka_center(cen, func);
	get_radius(rad, func);
	get_center(cen, func);

//	mpc_out_str(stdout, 10, 0, rad); printf(", "); mpc_out_str(stdout, 10, 0, cen); printf("\n");

	//cinit = init_mpfcmplx();
    mpc_init2(cinit, x_init->prec);
	for(i = 0; i < func->deg; i++)
	{
		//set0_mpfcmplx(cinit);
        mpc_set_ui(cinit, 0UL, MPC_RNDNN);
#ifndef USE_MPFR
		mpc_set_d(tmp, (double)(2.0 * M_PI * i / func->deg + 3.0 / (2.0 * func->deg)));
#else
		mpfr_set_d(tmp, (double)(2.0 * M_PI * i / func->deg + 3.0 / (2.0 * func->deg)), GMP_RNDN);
#endif
		//iexp_mpfcmplx(cinit, tmp);
		mpf_cos(cosine, tmp);
		mpf_sin(sine, tmp);
		mpc_set_fr_fr(cinit, cosine, sine, get_bnc_default_rounding_mode_c());
        //get_real_mpfcmplx(re_cinit, cinit);
		//get_image_mpfcmplx(im_cinit, cinit);
		mpf_set(re_cinit, mpc_realref(cinit));
		mpf_set(im_cinit, mpc_imagref(cinit));

		/* re_cinit = cen + rad * re_cinit */
		mpf_mul(re_cinit, rad, re_cinit);
		mpf_add(re_cinit, mpc_realref(cen), re_cinit);

		/* im_cinit = cen + rad * im_cinit */
		mpf_mul(im_cinit, rad, im_cinit);
		mpf_add(im_cinit, mpc_imagref(cen), im_cinit);

        //set_real_mpfcmplx(cinit, re_cinit);
		//set_image_mpfcmplx(cinit, im_cinit);
		mpc_set_fr_fr(cinit,
			re_cinit,
			im_cinit,
			get_bnc_default_rounding_mode_c()
		);

		//abs_mpfcmplx(tmp, cinit);
		mpc_abs(tmp, cinit, get_bnc_default_rounding_mode());
//		printf("%5d(%f) ", i, mpf2double(tmp)); print_mpfcmplx(cinit);

		set_cmpfarray_i(x_init, i, cinit);
	}

	mpf_clear(rad);
	mpf_clear(an);
	mpf_clear(tmp);
	mpf_clear(re_cinit);
	mpf_clear(im_cinit);
    mpf_clear(cosine);
    mpf_clear(sine);

    mpc_clear(cen);
    mpc_clear(cinit);
}


/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int mpc_dka_mod(CMPFArray ans, CMPFArray x_init, CMPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps)
{
	long int times, i, j, deg, flag;
	mpf_t absmodval, abs_x, abs_newx, mpftmp;
	//MPFCmplx modval, up_modval, low_modval, tmp;
	mpc_t modval, up_modval, low_modval, tmp;

//	printf("\n");
//	print_cfarray(x_init);
//	printf("%f, %f\n", abs_eps, rel_eps);

	deg = ans->size;

	mpf_init2(absmodval, ans->prec);
	mpf_init2(abs_x, ans->prec);
	mpf_init2(abs_newx, ans->prec);
	mpf_init2(mpftmp, ans->prec);

	mpc_init2(modval, ans->prec); // = init2_mpfcmplx(ans->prec);
	mpc_init2(low_modval, ans->prec); // = init2_mpfcmplx(ans->prec);
	mpc_init2(up_modval, ans->prec); //  = init2_mpfcmplx(ans->prec);
	mpc_init2(tmp, ans->prec); //  = init2_mpfcmplx(ans->prec);

	for(times = 0; times <= maxtimes; times++)
	{
//		printf("%5d: ", times); fflush(stdout);

		flag = 0;
		for(i = 0; i < deg; i++)
		{
			//set_real_mpfcmplx_ui(low_modval, 1UL);
			//set_image_mpfcmplx_ui(low_modval, 0UL);
            mpc_set_ui_ui(low_modval, 1UL, 0UL, get_bnc_default_rounding_mode_c());

			for(j = 0; j < i; j++)
			{
				//set0_mpfcmplx(tmp);
                mpc_set_ui(tmp, 0UL, get_bnc_default_rounding_mode_c());
				mpc_sub(
					tmp,
					get_cmpfarray_i(x_init, i),
					get_cmpfarray_i(x_init, j), get_bnc_default_rounding_mode_c()
				);
				mpc_mul(low_modval, low_modval, tmp, get_bnc_default_rounding_mode_c());
			}
			for(j = i + 1; j < deg; j++)
			{
				mpc_set_ui(tmp, 0UL, get_bnc_default_rounding_mode_c());
				mpc_sub(
					tmp,
					get_cmpfarray_i(x_init, i),
					get_cmpfarray_i(x_init, j), get_bnc_default_rounding_mode_c()
				);
				mpc_mul(low_modval, low_modval, tmp, get_bnc_default_rounding_mode_c());
			}
			mpc_mul(low_modval, low_modval, get_cmpfpoly_i(func, func->deg), get_bnc_default_rounding_mode_c());
			eval_cmpfpoly(up_modval, func, get_cmpfarray_i(x_init, i));

			mpc_div(modval, up_modval, low_modval, get_bnc_default_rounding_mode_c());
			mpc_sub(tmp, get_cmpfarray_i(x_init, i), modval, get_bnc_default_rounding_mode_c());
			set_cmpfarray_i(ans, i, tmp);

//			get_real_mpfcmplx(mpftmp, get_cmpfarray_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			mpc_abs(absmodval, modval, get_bnc_default_rounding_mode());
			mpc_abs(abs_x, get_cmpfarray_i(x_init, i), get_bnc_default_rounding_mode());
			mpc_abs(abs_newx, get_cmpfarray_i(ans, i), get_bnc_default_rounding_mode());

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

	mpc_clear(modval);
	mpc_clear(low_modval);
	mpc_clear(up_modval);
	mpc_clear(tmp);

	printf("ans->prec: %ld\n", ans->prec);
	return times;
}

/* Petkovic Method : 3rd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int mpc_petckovic(CMPFArray ans, CMPFArray x_init, CMPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps)
{
	long int times, i, j, deg, flag;
	mpf_t absmodval, abs_x, abs_newx, rtmp;
	mpc_t modval, up_modval, low_modval, tmp, mpftmp, mpftmp2;
	CMPFPoly in_func;

//	printf("\n");
//	print_cfarray(x_init);
//	printf("%f, %f\n", abs_eps, rel_eps);

	deg = ans->size;

	mpf_init2(absmodval, ans->prec);
	mpf_init2(abs_x, ans->prec);
	mpf_init2(abs_newx, ans->prec);
	mpf_init2(rtmp, ans->prec);

    mpc_init2(mpftmp, ans->prec);
	mpc_init2(mpftmp2, ans->prec);
	mpc_init2(modval, ans->prec); // = init2_mpfcmplx(ans->prec);
	mpc_init2(low_modval, ans->prec); // = init2_mpfcmplx(ans->prec);
	mpc_init2(up_modval, ans->prec); //  = init2_mpfcmplx(ans->prec);
	mpc_init2(tmp, ans->prec); //  = init2_mpfcmplx(ans->prec);

	in_func = init2_cmpfpoly(func->max_len, func->prec);
	// mpttmp := 1/a_n
	mpc_ui_div(mpftmp, 1UL, get_cmpfpoly_i(func, func->deg), get_bnc_default_rounding_mode_c()); 
	for(i = 0; i <= func->deg; i++)
	{
		mpc_mul(mpftmp2, get_cmpfpoly_i(func, i), mpftmp, get_bnc_default_rounding_mode_c());
		set_cmpfpoly_i(in_func, i, mpftmp2);
	}
	//setdegree_cmpfpoly(in_func);


	for(times = 0; times <= maxtimes; times++)
	{
//		printf("%5d: ", times); fflush(stdout);

		flag = 0;
		for(i = 0; i < deg; i++)
		{
			// get Weierstrass's correction to low_modval
			//set_real_mpfcmplx_ui(low_modval, 1UL);
			//set_image_mpfcmplx_ui(low_modval, 0UL);
            mpc_set_ui_ui(low_modval, 1UL, 0UL, get_bnc_default_rounding_mode_c());
			for(j = 0; j < i; j++)
			{
				mpc_set_ui(tmp, 0UL, get_bnc_default_rounding_mode_c()); //set0_mpfcmplx(tmp);
				mpc_sub(
					tmp,
					get_cmpfarray_i(x_init, i),
					get_cmpfarray_i(x_init, j), get_bnc_default_rounding_mode_c()
				);
				mpc_mul(low_modval, low_modval, tmp, get_bnc_default_rounding_mode_c());
			}
			for(j = i + 1; j < deg; j++)
			{
				mpc_set_ui(tmp, 0UL, get_bnc_default_rounding_mode_c()); //set0_mpfcmplx(tmp);
				mpc_sub(
					tmp,
					get_cmpfarray_i(x_init, i),
					get_cmpfarray_i(x_init, j), get_bnc_default_rounding_mode_c()
				);
				mpc_mul(low_modval, low_modval, tmp, get_bnc_default_rounding_mode_c());
			}
			//mul_mpfcmplx_mpf(low_modval, low_modval, get_cmpfpoly_i(func, func->deg));
			mpc_mul_ui(low_modval, low_modval, 2UL, get_bnc_default_rounding_mode_c()); // 1/2 * modval
			eval_cmpfpoly(up_modval, in_func, get_cmpfarray_i(x_init, i));
			// modval := Weierstrass's correction * 1/2
			mpc_div(modval, up_modval, low_modval, get_bnc_default_rounding_mode_c());
			
			// tmp := z - modval
			mpc_sub(tmp, get_cmpfarray_i(x_init, i), modval, get_bnc_default_rounding_mode_c());
			eval_diff_cmpfpoly(low_modval, in_func, tmp);
			
			// new_z := old_z - f(z) / f'(z - 1/2 * W)
			mpc_div(modval, up_modval, low_modval, get_bnc_default_rounding_mode_c());
			//mul_mpfcmplx_mpf(modval, modval, get_cmpfpoly_i(func, func->deg));
			mpc_sub(tmp, get_cmpfarray_i(x_init, i), modval, get_bnc_default_rounding_mode_c());
			set_cmpfarray_i(ans, i, tmp);

//			get_real_mpfcmplx(mpftmp, get_cmpfarray_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			mpc_abs(absmodval, modval, get_bnc_default_rounding_mode());
			mpc_abs(abs_x, get_cmpfarray_i(x_init, i), get_bnc_default_rounding_mode());
			//abs_mpfcmplx(abs_newx, get_cmpfarray_i(ans, i));

			//mpc_add(mpftmp, abs_x, abs_newx);
			//mpc_mul(mpftmp, mpftmp, rel_eps);
			//mpc_add(mpftmp, mpftmp, abs_eps);
			mpf_mul(rtmp, abs_x, rel_eps);
			mpf_add(rtmp, rtmp, abs_eps);
			// |new_x - old_x| > |old_x| * rel_eps + abs_eps
			if( mpf_cmp(absmodval, rtmp) > 0 )
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
    mpf_clear(rtmp);

    mpc_clear(mpftmp);
	mpc_clear(mpftmp2);
	mpc_clear(modval);
	mpc_clear(low_modval);
	mpc_clear(up_modval);
	mpc_clear(tmp);

	free_cmpfpoly(in_func);

	printf("ans->prec: %ld\n", ans->prec);
	return times;
}

/* Aberth Method : 3rd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int mpc_aberth(CMPFArray ans, CMPFArray x_init, CMPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps)
{
	long int times, i, j, deg, flag;
	mpf_t absmodval, abs_x, abs_newx, mpftmp, mpftmp2;
	mpc_t modval, up_modval, low_modval, tmp, one;

//	printf("\n");
//	print_cfarray(x_init);
//	printf("%f, %f\n", abs_eps, rel_eps);

	deg = ans->size;

	mpf_init2(absmodval, ans->prec);
	mpf_init2(abs_x, ans->prec);
	mpf_init2(abs_newx, ans->prec);
	mpf_init2(mpftmp, ans->prec);
	mpf_init2(mpftmp2, ans->prec);

	mpc_init2(modval, ans->prec); // = init2_mpfcmplx(ans->prec);
	mpc_init2(low_modval, ans->prec); // = init2_mpfcmplx(ans->prec);
	mpc_init2(up_modval, ans->prec); //  = init2_mpfcmplx(ans->prec);
	mpc_init2(tmp, ans->prec); //  = init2_mpfcmplx(ans->prec);

	// complex one := 1
	mpc_init2(one, ans->prec);
	//set_real_mpfcmplx_ui(one, 1UL);
	//set_image_mpfcmplx_ui(one, 0UL);
    mpc_set_ui_ui(one, 1UL, 0UL, get_bnc_default_rounding_mode_c());

	for(times = 0; times <= maxtimes; times++)
	{
//		printf("%5d: ", times); fflush(stdout);

		flag = 0;
		for(i = 0; i < deg; i++)
		{
			// low_modval := sum (z_i - z_k)^(-1)
			//set0_mpfcmplx(low_modval);
            mpc_set_ui(low_modval, 0UL, get_bnc_default_rounding_mode_c());
			for(j = 0; j < i; j++)
			{
				mpc_set_ui(tmp, 0UL, get_bnc_default_rounding_mode_c()); //set0_mpfcmplx(tmp);
				mpc_sub(
					tmp,
					get_cmpfarray_i(x_init, i),
					get_cmpfarray_i(x_init, j), get_bnc_default_rounding_mode_c()
				);
				mpc_div(tmp, one, tmp, get_bnc_default_rounding_mode_c());
				mpc_add(low_modval, low_modval, tmp, get_bnc_default_rounding_mode_c());
			}
			for(j = i + 1; j < deg; j++)
			{
				mpc_set_ui(tmp, 0UL, get_bnc_default_rounding_mode_c()); //set0_mpfcmplx(tmp);
				mpc_sub(
					tmp,
					get_cmpfarray_i(x_init, i),
					get_cmpfarray_i(x_init, j), get_bnc_default_rounding_mode_c()
				);
				mpc_div(tmp, one, tmp, get_bnc_default_rounding_mode_c());
				mpc_add(low_modval, low_modval, tmp, get_bnc_default_rounding_mode_c());
			}
			// up_mpdval := f(z) / f'(z)
			eval_cmpfpoly(up_modval, func, get_cmpfarray_i(x_init, i));
			eval_diff_cmpfpoly(tmp, func, get_cmpfarray_i(x_init, i));
			mpc_div(up_modval, up_modval, tmp, get_bnc_default_rounding_mode_c());
			
			// low_modval := 1 - up_modval * sum (z_i - z_k)^(-1)
			mpc_mul(tmp, up_modval, low_modval, get_bnc_default_rounding_mode_c());
			mpc_sub(low_modval, one, tmp, get_bnc_default_rounding_mode_c());
			
			// new_z := old_z - up_modval / low_modval
			mpc_div(modval, up_modval, low_modval, get_bnc_default_rounding_mode_c());
			mpc_sub(tmp, get_cmpfarray_i(x_init, i), modval, get_bnc_default_rounding_mode_c());
			set_cmpfarray_i(ans, i, tmp);

//			get_real_mpfcmplx(mpftmp, get_cmpfarray_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			mpc_abs(absmodval, modval, get_bnc_default_rounding_mode());
			mpc_abs(abs_x, get_cmpfarray_i(x_init, i), get_bnc_default_rounding_mode());
			//abs_mpfcmplx(abs_newx, get_cmpfarray_i(ans, i));

			//mpc_add(mpftmp, abs_x, abs_newx);
			//mpc_mul(mpftmp, mpftmp, rel_eps);
			//mpc_add(mpftmp, mpftmp, abs_eps);
			mpf_mul(mpftmp, abs_x, rel_eps);
			mpf_add(mpftmp, mpftmp, abs_eps);
			// |new_x - old_x| > |old_x| * rel_eps + abs_eps
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
	mpf_clear(mpftmp2);

	mpc_clear(modval);
	mpc_clear(low_modval);
	mpc_clear(up_modval);
	mpc_clear(tmp);

	printf("ans->prec: %ld\n", ans->prec);
	return times;
}
#endif // USE_GMP

#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus
