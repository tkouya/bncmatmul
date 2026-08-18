/********************************************************************************/
/* td_dka.c: Durand-Kerner-Aberth Methods                                       */
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

#ifdef USE_TDLINEAR
/* DD */
#include "tdlinear.h"
#include "ctdlinear.h"

/* center of gravity */
/* center = -a_{n-1} / (n * a_n) */
void td_dka_center(double ret[TDSIZE], TDPoly func)
{
	rtd_set(ret, get_tdpoly_i(func, func->deg - 1));

	rtd_div(ret, ret, get_tdpoly_i(func, func->deg));
	rtd_div_ui(ret, ret, (unsigned long)func->deg);
	rtd_neg(ret, ret);
}

/* radius = max (m * |a_i / a_deg|)^(1/(deg-i)) */
void td_dka_radius(double ret[TDSIZE], TDPoly func)
{
	long int i;
	double tmp[TDSIZE], num_nonzero[TDSIZE], an[TDSIZE];
	double dtmp;

	rtd_set_ui(num_nonzero, (unsigned long)num_nonzero_tdpoly(func));
	rtd_set(an, get_tdpoly_i(func, func->deg));
	rtd_set(ret, num_nonzero);
	for(i = func->deg - 1; i >= 0; i--)
	{
		rtd_div(tmp, get_tdpoly_i(func, i), an);
		rtd_mul(tmp, tmp, num_nonzero);
		rtd_abs(tmp, tmp);
		
		/* tmp^(deg-i) */
/*		rtd_ln(tmp, tmp);
		rtd_mul_ui(tmp, tmp, (unsigned long)func->deg - i);
		rtd_exp(tmp, tmp);
*/
		dtmp = rtd_get_d(tmp); dtmp = pow(dtmp, 1.0/(double)(func->deg - i)); rtd_set_d(tmp, dtmp);
		if(rtd_cmp(ret, tmp) < 0)
			rtd_set(ret, tmp);
	}
}

// ------------------------------------
// New implementation
// 2025-01-15 (Wed) T.Kouya
// ------------------------------------

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void td_dka_init(CTDVector x_init, TDPoly func)
{
	long int i, itmp;
	double rad[TDSIZE], cen[TDSIZE], an[TDSIZE], tmp[TDSIZE], re_cinit[TDSIZE], im_cinit[TDSIZE];
	//MPFCmplx cinit;
	double cosine[TDSIZE], sine[TDSIZE];
	ctdfloat cinit;

	td_dka_radius(rad, func);
	td_dka_center(cen, func);

//	rtd_out_str(stdout, 10, 0, rad); printf(", "); rtd_out_str(stdout, 10, 0, cen); printf("\n");

	for(i = 0; i < func->deg; i++)
	{
		//set0_mpfcmplx(cinit);
		rctd_set_ui(&cinit, 0UL);
		rtd_set_d(tmp, (double)(2.0 * M_PI * i / func->deg + 3.0 / (2.0 * func->deg)));

		//iexp_mpfcmplx(cinit, tmp);
		// ctmp := i * tmp -> exp(i * tmp) := cos(tmp) + i * sin(tmp);
		//rtd_cos(cosine, tmp);
		//rtd_sin(sine, tmp);
		rtd_func_mpfr(cosine, mpfr_cos, tmp);
		rtd_func_mpfr(sine, mpfr_sin, tmp);
		rctd_set_td_td(&cinit, cosine, sine);

		//get_real_mpfcmplx(re_cinit, cinit);
		//get_image_mpfcmplx(im_cinit, cinit);
		rtd_set(re_cinit, cinit.val_re);
		rtd_set(im_cinit, cinit.val_im);

		/* re_cinit = cen + rad * re_cinit */
		rtd_mul(re_cinit, rad, re_cinit);
		rtd_add(re_cinit, cen, re_cinit);

		/* im_cinit = rad * im_cinit */
		rtd_mul(im_cinit, rad, im_cinit);

		//set_real_mpfcmplx(cinit, re_cinit);
		//set_image_mpfcmplx(cinit, im_cinit);
		rctd_set_td_td(&cinit, re_cinit, im_cinit);

		//abs_mpfcmplx(tmp, cinit);
		rctd_abs_td(tmp, &cinit);
//		printf("%5d(%f) ", i, mpf2double(tmp)); print_mpfcmplx(cinit);

		set_ctdvector_i(x_init, i, &cinit);
	}
}

/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int td_dka(CTDVector ans, CTDVector x_init, TDPoly func, long int maxtimes, double abs_eps[TDSIZE], double rel_eps[TDSIZE])
{
	long int times, i, j, deg, flag;
	double absmodval[TDSIZE], abs_x[TDSIZE], abs_newx[TDSIZE], mpftmp[TDSIZE];
	ctdfloat modval, up_modval, low_modval, tmp, x_init_i;

//	printf("\n");
//	print_cfarray(x_init);
//	printf("%f, %f\n", abs_eps, rel_eps);

	deg = ans->re->dim;

	/*
	modval = init_mpfcmplx();
	low_modval = init_mpfcmplx();
	up_modval = init_mpfcmplx();
	tmp = init_mpfcmplx();
	*/

	for(times = 0; times <= maxtimes; times++)
	{
//		printf("%5d: ", times); fflush(stdout);

		flag = 0;
		for(i = 0; i < deg; i++)
		{
			//set_real_mpfcmplx_ui(low_modval, 1UL);
			//set_image_mpfcmplx_ui(low_modval, 0UL);
			rctd_set_ui_ui(&low_modval, 1UL, 0UL);
			subst_ctdvector_i(&x_init_i, x_init, i);

			for(j = 0; j < i; j++)
			{
				//set0_mpfcmplx(tmp);
				rctd_set_ui(&tmp, 0UL);
				//sub_mpfcmplx(
				rctd_sub(
					&tmp,
					&x_init_i, //get_ctdvector_i(x_init, i),
					get_ctdvector_i(x_init, j)
				);
				//mul2_mpfcmplx(low_modval, tmp);
				rctd_mul(&low_modval, &low_modval, &tmp);
			}
			for(j = i + 1; j < deg; j++)
			{
				//set0_mpfcmplx(tmp);
				rctd_set_ui(&tmp, 0UL);
				//sub_mpfcmplx(
				rctd_sub(
					&tmp,
					&x_init_i, //get_ctdvector_i(x_init, i),
					get_ctdvector_i(x_init, j)
				);
				//mul2_mpfcmplx(low_modval, tmp);
				rctd_mul(&low_modval, &low_modval, &tmp);
			}
			//mul_mpfcmplx_mpf(low_modval, low_modval, get_tdpoly_i(func, func->deg));
			rctd_mul_td(&low_modval, &low_modval, get_tdpoly_i(func, func->deg));
#if (defined(__AVX2__) || defined(__AVX512F__)) || ((defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON)) // ARM NEON
			_bncavx2_ceval_tdpoly_estrin(&up_modval, func, &x_init_i); //get_ctdvector_i(x_init, i));
#else // __AVX2__
			ceval_tdpoly(&up_modval, func, &x_init_i); //get_ctdvector_i(x_init, i));
#endif // __AVX2__

			//div_mpfcmplx(modval, up_modval, low_modval);
			rctd_div(&modval, &up_modval, &low_modval);
			//sub_mpfcmplx(tmp, get_ctdvector_i(x_init, i), modval);
			//rctd_sub(&tmp, get_ctdvector_i(x_init, i), &modval);
			rctd_sub(&tmp, &x_init_i, &modval);
			set_ctdvector_i(ans, i, &tmp);

//			get_real_mpfcmplx(mpftmp, get_ctdvector_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			//abs_mpfcmplx(absmodval, modval);
			//abs_mpfcmplx(abs_x, get_ctdvector_i(x_init, i));
			//abs_mpfcmplx(abs_newx, get_ctdvector_i(ans, i));
			rctd_abs_td(absmodval, &modval);
			rctd_abs_td(abs_x, &x_init_i); // get_ctdvector_i(x_init, i));
			rctd_abs_td(abs_newx, get_ctdvector_i(ans, i));

			rtd_add(mpftmp, abs_x, abs_newx);
			rtd_mul(mpftmp, mpftmp, rel_eps);
			rtd_add(mpftmp, mpftmp, abs_eps);
			if( rtd_cmp(absmodval, mpftmp) > 0 )
				flag = 1;

		}

		/* check convergence */
		if(flag == 0)
			break;

		subst_ctdvector(x_init, ans);

	}
	return times;
}

#if 0
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int _bncavx2_td_dka(CTDVector ans, CTDVector x_init, TDPoly func, long int maxtimes, double abs_eps[TDSIZE], double rel_eps[TDSIZE])
{
	long int times, i, j, deg, flag;
	//double absmodval[TDSIZE], abs_x[TDSIZE], abs_newx[TDSIZE], mpftmp[TDSIZE];
	//ctdfloat modval, up_modval, low_modval, tmp;
	__m256d absmodval[TDSIZE], abs_x[TDSIZE], abs_newx[TDSIZE], mpftmp[TDSIZE];
	__m256d modval_re[TDSIZE], up_modval_re[TDSIZE], low_modval_re[TDSIZE], tmp_re[TDSIZE];
	__m256d modval_im[TDSIZE], up_modval_im[TDSIZE], low_modval_im[TDSIZE], tmp_im[TDSIZE];
	__m256d x_init_re, x_init_im;

//	printf("\n");
//	print_cfarray(x_init);
//	printf("%f, %f\n", abs_eps, rel_eps);

	deg = ans->re->dim;

	/*
	modval = init_mpfcmplx();
	low_modval = init_mpfcmplx();
	up_modval = init_mpfcmplx();
	tmp = init_mpfcmplx();
	*/

	for(times = 0; times <= maxtimes; times++)
	{
//		printf("%5d: ", times); fflush(stdout);

		flag = 0;
		for(i = 0; i < deg; i += _BNC_D_WIDTH)
		{
			//set_real_mpfcmplx_ui(low_modval, 1UL);
			//set_image_mpfcmplx_ui(low_modval, 0UL);
			_bncavx2_rctd_set1_ui_ui(low_modval_re, low_modval_im, 1UL, 0UL);

			for(j = 0; j < i; j++)
			{
				//set0_mpfcmplx(tmp);
				_bncavx2_rctd_set1_ui(tmp, 0UL);
				//sub_mpfcmplx(
				_bncavx2_rctd_sub(
					tmp,
					get_ctdvector_i(x_init, i),
					get_ctdvector_i(x_init, j)
				);
				//mul2_mpfcmplx(low_modval, tmp);
				_bncavx2_rctd_mul(&low_modval, &low_modval, &tmp);
			}
			for(j = i + 1; j < deg; j++)
			{
				//set0_mpfcmplx(tmp);
				_bncavx2_rctd_set_ui(&tmp, 0UL);
				//sub_mpfcmplx(
				_bncavx2_rctd_sub(
					&tmp,
					get_ctdvector_i(x_init, i),
					get_ctdvector_i(x_init, j)
				);
				//mul2_mpfcmplx(low_modval, tmp);
				_bncavx2_rctd_mul(&low_modval, &low_modval, &tmp);
			}
			//mul_mpfcmplx_mpf(low_modval, low_modval, get_tdpoly_i(func, func->deg));
			_bncavx2_rctd_mul_td(low_modval_re, low_modval_im, get_tdpoly_i(func, func->deg));
			_bncavx2_ceval_tdpoly_horner(&up_modval, func, get_ctdvector_i(x_init, i));


			//div_mpfcmplx(modval, up_modval, low_modval);
			_bncavx2_rctd_div(&modval, &up_modval, &low_modval);
			//sub_mpfcmplx(tmp, get_ctdvector_i(x_init, i), modval);
			_bncavx2_rctd_sub(&tmp, get_ctdvector_i(x_init, i), &modval);
			set_ctdvector_i(ans, i, &tmp);

//			get_real_mpfcmplx(mpftmp, get_ctdvector_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			//abs_mpfcmplx(absmodval, modval);
			//abs_mpfcmplx(abs_x, get_ctdvector_i(x_init, i));
			//abs_mpfcmplx(abs_newx, get_ctdvector_i(ans, i));
			_bncavx2_rctd_abs_td(absmodval, &modval);
			_bncavx2_rctd_abs_td(abs_x, get_ctdvector_i(x_init, i));
			_bncavx2_rctd_abs_td(abs_newx, get_ctdvector_i(ans, i));

			_bncavx2_rtd_add(mpftmp, abs_x, abs_newx);
			_bncavx2_rtd_mul(mpftmp, mpftmp, rel_eps);
			_bncavx2_rtd_add(mpftmp, mpftmp, abs_eps);
			if( _bncavx2_rtd_cmp(absmodval, mpftmp) > 0 )
				flag = 1;

		}

		/* check convergence */
		if(flag == 0)
			break;

		subst_ctdvector(x_init, ans);

	}
	return times;
}
#endif // __AVX2__
#endif // 0

// deflation from highest degree coef
// ret org(x) / (x - alpha)
void deflation_tdpoly(TDPoly ret, TDPoly org, double alpha[TDSIZE])
{
	long int i;
	double tmp[TDSIZE];

	set_tdpoly_i(ret, org->deg - 1, get_tdpoly_i(org, org->deg));
	for(i = org->deg - 2; i >= 0; i--)
	{
		rtd_mul(tmp, get_tdpoly_i(ret, i + 1), alpha);
		rtd_add(tmp, tmp, get_tdpoly_i(org, i + 1));
		set_tdpoly_i(ret, i, tmp);
	}

	setdegree_tdpoly(ret);
}

// deflation from highest degree coef and evaluation
// ret org(x) / (x - alpha)
void eval_deflation_tdpoly(double ret[TDSIZE], TDPoly org, double alpha[TDSIZE], double x[TDSIZE])
{
	long int i;
	double tmp[TDSIZE], new_coef_i[TDSIZE];

	//set_tdpoly_i(new_coef_i, org->deg - 1, get_tdpoly_i(org, org->deg));
	//set_tdpoly_i(ret, org->deg - 1, get_tdpoly_i(org, org->deg));
	rtd_set(new_coef_i, get_tdpoly_i(org, org->deg));
	rtd_set(ret, get_tdpoly_i(org, org->deg));
	for(i = org->deg - 2; i >= 0; i--)
	{
		rtd_mul(ret, ret, x);
		//rtd_mul(tmp, get_tdpoly_i(ret, i + 1), alpha);
		rtd_mul(tmp, new_coef_i, alpha);
		rtd_add(new_coef_i, tmp, get_tdpoly_i(org, i + 1));
		//set_tdpoly_i(ret, i, new_coef_i);
		rtd_add(ret, ret, new_coef_i);
	}
}

// Ozawa's initial radius : r := (p(center) / a_n)^(1/n)
void td_dka_ozawa_radius(double ret[TDSIZE], TDPoly func)
{
	double barycentric_point[TDSIZE];
	TDPoly in_func[2];

	in_func[0] = init_tdpoly(func->max_len);
	in_func[1] = init_tdpoly(func->max_len);

	subst_tdpoly(in_func[0], func);

	do{
		td_dka_center(barycentric_point, in_func[0]);
		eval_tdpoly(ret, in_func[0], barycentric_point);

		rtd_div(ret, ret, get_tdpoly_i(in_func[0], in_func[0]->deg));
		rtd_abs(ret, ret);
		//td_rootn_ui(ret, ret, (unsigned long)(in_func[0]->deg), GMP_RNDN);

		// normal return
		//if(mpfr_regular_p(ret) != 0)
			break;

		deflation_tdpoly(in_func[1], in_func[0], barycentric_point);
		subst_tdpoly(in_func[0], in_func[1]);
	} while(1);

	free_tdpoly(in_func[0]);
	free_tdpoly(in_func[1]);
}

// ------------------------------------
// New implementation
// 2025-01-16(Thu)
// ------------------------------------

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void td_dka_init2(CTDVector x_init, TDPoly func, void (* get_radius)(double *, TDPoly), void (* get_center)(double *, TDPoly))
{
	long int i, itmp;
	double rad[TDSIZE], cen[TDSIZE], an[TDSIZE], tmp[TDSIZE], re_cinit[TDSIZE], im_cinit[TDSIZE];
    double cosine[TDSIZE], sine[TDSIZE];
	//MPFCmplx cinit;
	ctdfloat cinit;

	//rtd_dka_radius(rad, func);
	//rtd_dka_center(cen, func);
	get_radius(rad, func);
	get_center(cen, func);

//	rtd_out_str(stdout, 10, 0, rad); printf(", "); rtd_out_str(stdout, 10, 0, cen); printf("\n");

	for(i = 0; i < func->deg; i++)
	{
		//set0_mpfcmplx(cinit);
        rctd_set_ui(&cinit, 0UL);
		rtd_set_d(tmp, (double)(2.0 * M_PI * i / func->deg + 3.0 / (2.0 * func->deg)));

		//iexp_mpfcmplx(cinit, tmp);
		rtd_func_mpfr(cosine, mpfr_cos, tmp);
		rtd_func_mpfr(sine, mpfr_sin, tmp);
		rctd_set_td_td(&cinit, cosine, sine);
        //get_real_mpfcmplx(re_cinit, cinit);
		//get_image_mpfcmplx(im_cinit, cinit);
		rtd_set(re_cinit, cinit.val_re);
		rtd_set(im_cinit, cinit.val_im);

		/* re_cinit = cen + rad * re_cinit */
		rtd_mul(re_cinit, rad, re_cinit);
		rtd_add(re_cinit, cen, re_cinit);

		/* im_cinit = rad * im_cinit */
		rtd_mul(im_cinit, rad, im_cinit);

		//set_real_mpfcmplx(cinit, re_cinit);
		//set_image_mpfcmplx(cinit, im_cinit);
		rctd_set_td_td(&cinit,
			re_cinit,
			im_cinit
		);

		//abs_mpfcmplx(tmp, cinit);
		rctd_abs_td(tmp, &cinit);
//		printf("%5d(%f) ", i, mpf2double(tmp)); print_mpfcmplx(cinit);

		set_ctdvector_i(x_init, i, &cinit);
	}
}


/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int td_dka_mod(CTDVector ans, CTDVector x_init, TDPoly func, long int maxtimes, double abs_eps[TDSIZE], double rel_eps[TDSIZE])
{
	long int times, i, j, deg, flag;
	double absmodval[TDSIZE], abs_x[TDSIZE], abs_newx[TDSIZE], mpftmp[TDSIZE];
	//MPFCmplx modval, up_modval, low_modval, tmp;
	ctdfloat modval, up_modval, low_modval, tmp;

//	printf("\n");
//	print_cfarray(x_init);
//	printf("%f, %f\n", abs_eps, rel_eps);

	deg = ans->re->dim;

	for(times = 0; times <= maxtimes; times++)
	{
//		printf("%5d: ", times); fflush(stdout);

		flag = 0;
		for(i = 0; i < deg; i++)
		{
			//set_real_mpfcmplx_ui(low_modval, 1UL);
			//set_image_mpfcmplx_ui(low_modval, 0UL);
            rctd_set_ui_ui(&low_modval, 1UL, 0UL);

			for(j = 0; j < i; j++)
			{
				//set0_mpfcmplx(tmp);
                rctd_set_ui(&tmp, 0UL);
				rctd_sub(
					&tmp,
					get_ctdvector_i(x_init, i),
					get_ctdvector_i(x_init, j)
				);
				rctd_mul(&low_modval, &low_modval, &tmp);
			}
			for(j = i + 1; j < deg; j++)
			{
				rctd_set_ui(&tmp, 0UL);
				rctd_sub(
					&tmp,
					get_ctdvector_i(x_init, i),
					get_ctdvector_i(x_init, j)
				);
				rctd_mul(&low_modval, &low_modval, &tmp);
			}
			rctd_mul_td(&low_modval, &low_modval, get_tdpoly_i(func, func->deg));
			ceval_tdpoly(&up_modval, func, get_ctdvector_i(x_init, i));

			rctd_div(&modval, &up_modval, &low_modval);
			rctd_sub(&tmp, get_ctdvector_i(x_init, i), &modval);
			set_ctdvector_i(ans, i, &tmp);

//			get_real_mpfcmplx(mpftmp, get_ctdvector_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			rctd_abs_td(absmodval, &modval);
			rctd_abs_td(abs_x, get_ctdvector_i(x_init, i));
			rctd_abs_td(abs_newx, get_ctdvector_i(ans, i));

			rtd_add(mpftmp, abs_x, abs_newx);
			rtd_mul(mpftmp, mpftmp, rel_eps);
			rtd_add(mpftmp, mpftmp, abs_eps);
			if( rtd_cmp(absmodval, mpftmp) > 0 )
				flag = 1;

		}

		/* check convergence */
		if(flag == 0)
			break;

		subst_ctdvector(x_init, ans);

	}

	return times;
}

/* Petkovic Method : 3rd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int td_petckovic(CTDVector ans, CTDVector x_init, TDPoly func, long int maxtimes, double abs_eps[TDSIZE], double rel_eps[TDSIZE])
{
	long int times, i, j, deg, flag;
	double absmodval[TDSIZE], abs_x[TDSIZE], abs_newx[TDSIZE], mpftmp[TDSIZE], mpftmp2[TDSIZE];
	ctdfloat modval, up_modval, low_modval, tmp;
	TDPoly in_func;

//	printf("\n");
//	print_cfarray(x_init);
//	printf("%f, %f\n", abs_eps, rel_eps);

	deg = ans->re->dim;

	in_func = init_tdpoly(func->max_len);
	// mpttmp := 1/a_n
	rtd_ui_div(mpftmp, 1UL, get_tdpoly_i(func, func->deg)); 
	for(i = 0; i <= func->deg; i++)
	{
		rtd_mul(mpftmp2, get_tdpoly_i(func, i), mpftmp);
		set_tdpoly_i(in_func, i, mpftmp2);
	}
	//setdegree_tdpoly(in_func);

	for(times = 0; times <= maxtimes; times++)
	{
//		printf("%5d: ", times); fflush(stdout);

		flag = 0;
		for(i = 0; i < deg; i++)
		{
			// get Weierstrass's correction to low_modval
			//set_real_mpfcmplx_ui(low_modval, 1UL);
			//set_image_mpfcmplx_ui(low_modval, 0UL);
            rctd_set_ui_ui(&low_modval, 1UL, 0UL);
			for(j = 0; j < i; j++)
			{
				rctd_set_ui(&tmp, 0UL); //set0_mpfcmplx(tmp);
				rctd_sub(
					&tmp,
					get_ctdvector_i(x_init, i),
					get_ctdvector_i(x_init, j)
				);
				rctd_mul(&low_modval, &low_modval, &tmp);
			}
			for(j = i + 1; j < deg; j++)
			{
				rctd_set_ui(&tmp, 0UL); //set0_mpfcmplx(tmp);
				rctd_sub(
					&tmp,
					get_ctdvector_i(x_init, i),
					get_ctdvector_i(x_init, j)
				);
				rctd_mul(&low_modval, &low_modval, &tmp);
			}
			//mul_mpfcmplx_mpf(low_modval, low_modval, get_tdpoly_i(func, func->deg));
			rctd_mul_ui(&low_modval, &low_modval, 2UL); // 1/2 * modval
			ceval_tdpoly(&up_modval, in_func, get_ctdvector_i(x_init, i));
			// modval := Weierstrass's correction * 1/2
			rctd_div(&modval, &up_modval, &low_modval);
			
			// tmp := z - modval
			rctd_sub(&tmp, get_ctdvector_i(x_init, i), &modval);
			ceval_diff_tdpoly(&low_modval, in_func, &tmp);
			
			// new_z := old_z - f(z) / f'(z - 1/2 * W)
			rctd_div(&modval, &up_modval, &low_modval);
			//mul_mpfcmplx_mpf(modval, modval, get_tdpoly_i(func, func->deg));
			rctd_sub(&tmp, get_ctdvector_i(x_init, i), &modval);
			set_ctdvector_i(ans, i, &tmp);

//			get_real_mpfcmplx(mpftmp, get_ctdvector_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			rctd_abs_td(absmodval, &modval);
			rctd_abs_td(abs_x, get_ctdvector_i(x_init, i));
			//abs_mpfcmplx(abs_newx, get_ctdvector_i(ans, i));

			//rtd_add(mpftmp, abs_x, abs_newx);
			//rtd_mul(mpftmp, mpftmp, rel_eps);
			//rtd_add(mpftmp, mpftmp, abs_eps);
			rtd_mul(mpftmp, abs_x, rel_eps);
			rtd_add(mpftmp, mpftmp, abs_eps);
			// |new_x - old_x| > |old_x| * rel_eps + abs_eps
			if( rtd_cmp(absmodval, mpftmp) > 0 )
				flag = 1;

		}

		/* check convergence */
		if(flag == 0)
			break;

		subst_ctdvector(x_init, ans);

	}

	free_tdpoly(in_func);

	return times;
}

/* Aberth Method : 3rd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int td_aberth(CTDVector ans, CTDVector x_init, TDPoly func, long int maxtimes, double abs_eps[TDSIZE], double rel_eps[TDSIZE])
{
	long int times, i, j, deg, flag;
	double absmodval[TDSIZE], abs_x[TDSIZE], abs_newx[TDSIZE], mpftmp[TDSIZE], mpftmp2[TDSIZE];
	ctdfloat modval, up_modval, low_modval, tmp, one;

//	printf("\n");
//	print_cfarray(x_init);
//	printf("%f, %f\n", abs_eps, rel_eps);

	deg = ans->re->dim;

	// complex one := 1
    rctd_set_ui_ui(&one, 1UL, 0UL);

	for(times = 0; times <= maxtimes; times++)
	{
//		printf("%5d: ", times); fflush(stdout);

		flag = 0;
		for(i = 0; i < deg; i++)
		{
			// low_modval := sum (z_i - z_k)^(-1)
			//set0_mpfcmplx(low_modval);
            rctd_set_ui(&low_modval, 0UL);
			for(j = 0; j < i; j++)
			{
				rctd_set_ui(&tmp, 0UL); //set0_mpfcmplx(tmp);
				rctd_sub(
					&tmp,
					get_ctdvector_i(x_init, i),
					get_ctdvector_i(x_init, j)
				);
				rctd_div(&tmp, &one, &tmp);
				rctd_add(&low_modval, &low_modval, &tmp);
			}
			for(j = i + 1; j < deg; j++)
			{
				rctd_set_ui(&tmp, 0UL); //set0_mpfcmplx(tmp);
				rctd_sub(
					&tmp,
					get_ctdvector_i(x_init, i),
					get_ctdvector_i(x_init, j)
				);
				rctd_div(&tmp, &one, &tmp);
				rctd_add(&low_modval, &low_modval, &tmp);
			}
			// up_mpdval := f(z) / f'(z)
			ceval_tdpoly(&up_modval, func, get_ctdvector_i(x_init, i));
			ceval_diff_tdpoly(&tmp, func, get_ctdvector_i(x_init, i));
			rctd_div(&up_modval, &up_modval, &tmp);
			
			// low_modval := 1 - up_modval * sum (z_i - z_k)^(-1)
			rctd_mul(&tmp, &up_modval, &low_modval);
			rctd_sub(&low_modval, &one, &tmp);
			
			// new_z := old_z - up_modval / low_modval
			rctd_div(&modval, &up_modval, &low_modval);
			rctd_sub(&tmp, get_ctdvector_i(x_init, i), &modval);
			set_ctdvector_i(ans, i, &tmp);

//			get_real_mpfcmplx(mpftmp, get_ctdvector_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			rctd_abs_td(absmodval, &modval);
			rctd_abs_td(abs_x, get_ctdvector_i(x_init, i));
			//abs_mpfcmplx(abs_newx, get_ctdvector_i(ans, i));

			//rtd_add(mpftmp, abs_x, abs_newx);
			//rtd_mul(mpftmp, mpftmp, rel_eps);
			//rtd_add(mpftmp, mpftmp, abs_eps);
			rtd_mul(mpftmp, abs_x, rel_eps);
			rtd_add(mpftmp, mpftmp, abs_eps);
			// |new_x - old_x| > |old_x| * rel_eps + abs_eps
			if( rtd_cmp(absmodval, mpftmp) > 0 )
				flag = 1;

		}

		/* check convergence */
		if(flag == 0)
			break;

		subst_ctdvector(x_init, ans);

	}

	return times;
}
#endif // USE_TDLINEAR

#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus
