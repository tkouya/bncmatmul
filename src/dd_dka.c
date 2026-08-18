/********************************************************************************/
/* dd_dka.c: Durand-Kerner-Aberth Methods                                       */
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

#ifdef USE_DDLINEAR
/* DD */
#include "ddlinear.h"
#include "cddlinear.h"

/* center of gravity */
/* center = -a_{n-1} / (n * a_n) */
void dd_dka_center(double ret[DDSIZE], DDPoly func)
{
	rdd_set(ret, get_ddpoly_i(func, func->deg - 1));

	rdd_div(ret, ret, get_ddpoly_i(func, func->deg));
	rdd_div_ui(ret, ret, (unsigned long)func->deg);
	rdd_neg(ret, ret);
}

/* radius = max (m * |a_i / a_deg|)^(1/(deg-i)) */
void dd_dka_radius(double ret[DDSIZE], DDPoly func)
{
	long int i;
	double tmp[DDSIZE], num_nonzero[DDSIZE], an[DDSIZE];
	double dtmp;

	rdd_set_ui(num_nonzero, (unsigned long)num_nonzero_ddpoly(func));
	rdd_set(an, get_ddpoly_i(func, func->deg));
	rdd_set(ret, num_nonzero);
	for(i = func->deg - 1; i >= 0; i--)
	{
		rdd_div(tmp, get_ddpoly_i(func, i), an);
		rdd_mul(tmp, tmp, num_nonzero);
		rdd_abs(tmp, tmp);
		
		/* tmp^(deg-i) */
/*		rdd_ln(tmp, tmp);
		rdd_mul_ui(tmp, tmp, (unsigned long)func->deg - i);
		rdd_exp(tmp, tmp);
*/
		dtmp = rdd_get_d(tmp); dtmp = pow(dtmp, 1.0/(double)(func->deg - i)); rdd_set_d(tmp, dtmp);
		if(rdd_cmp(ret, tmp) < 0)
			rdd_set(ret, tmp);
	}
}

// ------------------------------------
// New implementation
// 2025-01-15 (Wed) T.Kouya
// ------------------------------------

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void dd_dka_init(CDDVector x_init, DDPoly func)
{
	long int i, itmp;
	double rad[DDSIZE], cen[DDSIZE], an[DDSIZE], tmp[DDSIZE], re_cinit[DDSIZE], im_cinit[DDSIZE];
	//MPFCmplx cinit;
	double cosine[DDSIZE], sine[DDSIZE];
	cddfloat cinit;

	dd_dka_radius(rad, func);
	dd_dka_center(cen, func);

//	rdd_out_str(stdout, 10, 0, rad); printf(", "); rdd_out_str(stdout, 10, 0, cen); printf("\n");

	for(i = 0; i < func->deg; i++)
	{
		//set0_mpfcmplx(cinit);
		rcdd_set_ui(&cinit, 0UL);
		rdd_set_d(tmp, (double)(2.0 * M_PI * i / func->deg + 3.0 / (2.0 * func->deg)));

		//iexp_mpfcmplx(cinit, tmp);
		// ctmp := i * tmp -> exp(i * tmp) := cos(tmp) + i * sin(tmp);
		//rdd_cos(cosine, tmp);
		//rdd_sin(sine, tmp);
		rdd_func_mpfr(cosine, mpfr_cos, tmp);
		rdd_func_mpfr(sine, mpfr_sin, tmp);
		rcdd_set_dd_dd(&cinit, cosine, sine);

		//get_real_mpfcmplx(re_cinit, cinit);
		//get_image_mpfcmplx(im_cinit, cinit);
		rdd_set(re_cinit, cinit.val_re);
		rdd_set(im_cinit, cinit.val_im);

		/* re_cinit = cen + rad * re_cinit */
		rdd_mul(re_cinit, rad, re_cinit);
		rdd_add(re_cinit, cen, re_cinit);

		/* im_cinit = rad * im_cinit */
		rdd_mul(im_cinit, rad, im_cinit);

		//set_real_mpfcmplx(cinit, re_cinit);
		//set_image_mpfcmplx(cinit, im_cinit);
		rcdd_set_dd_dd(&cinit, re_cinit, im_cinit);

		//abs_mpfcmplx(tmp, cinit);
		rcdd_abs_dd(tmp, &cinit);
//		printf("%5d(%f) ", i, mpf2double(tmp)); print_mpfcmplx(cinit);

		set_cddvector_i(x_init, i, &cinit);
	}
}

/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int dd_dka(CDDVector ans, CDDVector x_init, DDPoly func, long int maxtimes, double abs_eps[DDSIZE], double rel_eps[DDSIZE])
{
	long int times, i, j, deg, flag;
	double absmodval[DDSIZE], abs_x[DDSIZE], abs_newx[DDSIZE], mpftmp[DDSIZE];
	cddfloat modval, up_modval, low_modval, tmp, x_init_i;
	CDDVector low_modval_vec;

//#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
//	low_modval_vec = init_cddvector(deg);
//#endif // __AVX2__

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
			rcdd_set_ui_ui(&low_modval, 1UL, 0UL);
			subst_cddvector_i(&x_init_i, x_init, i);

			for(j = 0; j < i; j++)
			{
				//set0_mpfcmplx(tmp);
				rcdd_set_ui(&tmp, 0UL);
				//sub_mpfcmplx(
				rcdd_sub(
					&tmp,
					&x_init_i, //get_cddvector_i(x_init, i),
					get_cddvector_i(x_init, j)
				);
				//mul2_mpfcmplx(low_modval, tmp);
				rcdd_mul(&low_modval, &low_modval, &tmp);
			}
			for(j = i + 1; j < deg; j++)
			{
				//set0_mpfcmplx(tmp);
				rcdd_set_ui(&tmp, 0UL);
				//sub_mpfcmplx(
				rcdd_sub(
					&tmp,
					&x_init_i, //get_cddvector_i(x_init, i),
					get_cddvector_i(x_init, j)
				);
				//mul2_mpfcmplx(low_modval, tmp);
				rcdd_mul(&low_modval, &low_modval, &tmp);
			}
			//mul_mpfcmplx_mpf(low_modval, low_modval, get_ddpoly_i(func, func->deg));
			rcdd_mul_dd(&low_modval, &low_modval, get_ddpoly_i(func, func->deg));
#if (defined(__AVX2__) || defined(__AVX512F__)) || ((defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON)) // ARM NEON
			_bncavx2_ceval_ddpoly_estrin(&up_modval, func, &x_init_i); //get_cddvector_i(x_init, i));
#else // __AVX2__
			ceval_ddpoly(&up_modval, func, &x_init_i); //get_cddvector_i(x_init, i));
#endif // __AVX2__

			//div_mpfcmplx(modval, up_modval, low_modval);
			rcdd_div(&modval, &up_modval, &low_modval);
			//sub_mpfcmplx(tmp, get_cddvector_i(x_init, i), modval);
			//rcdd_sub(&tmp, get_cddvector_i(x_init, i), &modval);
			rcdd_sub(&tmp, &x_init_i, &modval);
			set_cddvector_i(ans, i, &tmp);

//			get_real_mpfcmplx(mpftmp, get_cddvector_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			//abs_mpfcmplx(absmodval, modval);
			//abs_mpfcmplx(abs_x, get_cddvector_i(x_init, i));
			//abs_mpfcmplx(abs_newx, get_cddvector_i(ans, i));
			rcdd_abs_dd(absmodval, &modval);
			rcdd_abs_dd(abs_x, &x_init_i); // get_cddvector_i(x_init, i));
			rcdd_abs_dd(abs_newx, get_cddvector_i(ans, i));

			rdd_add(mpftmp, abs_x, abs_newx);
			rdd_mul(mpftmp, mpftmp, rel_eps);
			rdd_add(mpftmp, mpftmp, abs_eps);
			if( rdd_cmp(absmodval, mpftmp) > 0 )
				flag = 1;

		}

		/* check convergence */
		if(flag == 0)
			break;

		subst_cddvector(x_init, ans);

	}
	return times;
}

#if 0
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int _bncavx2_dd_dka(CDDVector ans, CDDVector x_init, DDPoly func, long int maxtimes, double abs_eps[DDSIZE], double rel_eps[DDSIZE])
{
	long int times, i, j, deg, flag;
	//double absmodval[DDSIZE], abs_x[DDSIZE], abs_newx[DDSIZE], mpftmp[DDSIZE];
	//cddfloat modval, up_modval, low_modval, tmp;
	__m256d absmodval[DDSIZE], abs_x[DDSIZE], abs_newx[DDSIZE], mpftmp[DDSIZE];
	__m256d modval_re[DDSIZE], up_modval_re[DDSIZE], low_modval_re[DDSIZE], tmp_re[DDSIZE];
	__m256d modval_im[DDSIZE], up_modval_im[DDSIZE], low_modval_im[DDSIZE], tmp_im[DDSIZE];
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
			_bncavx2_rcdd_set1_ui_ui(low_modval_re, low_modval_im, 1UL, 0UL);

			for(j = 0; j < i; j++)
			{
				//set0_mpfcmplx(tmp);
				_bncavx2_rcdd_set1_ui(tmp, 0UL);
				//sub_mpfcmplx(
				_bncavx2_rcdd_sub(
					tmp,
					get_cddvector_i(x_init, i),
					get_cddvector_i(x_init, j)
				);
				//mul2_mpfcmplx(low_modval, tmp);
				_bncavx2_rcdd_mul(&low_modval, &low_modval, &tmp);
			}
			for(j = i + 1; j < deg; j++)
			{
				//set0_mpfcmplx(tmp);
				_bncavx2_rcdd_set_ui(&tmp, 0UL);
				//sub_mpfcmplx(
				_bncavx2_rcdd_sub(
					&tmp,
					get_cddvector_i(x_init, i),
					get_cddvector_i(x_init, j)
				);
				//mul2_mpfcmplx(low_modval, tmp);
				_bncavx2_rcdd_mul(&low_modval, &low_modval, &tmp);
			}
			//mul_mpfcmplx_mpf(low_modval, low_modval, get_ddpoly_i(func, func->deg));
			_bncavx2_rcdd_mul_dd(low_modval_re, low_modval_im, get_ddpoly_i(func, func->deg));
			_bncavx2_ceval_ddpoly_horner(&up_modval, func, get_cddvector_i(x_init, i));


			//div_mpfcmplx(modval, up_modval, low_modval);
			_bncavx2_rcdd_div(&modval, &up_modval, &low_modval);
			//sub_mpfcmplx(tmp, get_cddvector_i(x_init, i), modval);
			_bncavx2_rcdd_sub(&tmp, get_cddvector_i(x_init, i), &modval);
			set_cddvector_i(ans, i, &tmp);

//			get_real_mpfcmplx(mpftmp, get_cddvector_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			//abs_mpfcmplx(absmodval, modval);
			//abs_mpfcmplx(abs_x, get_cddvector_i(x_init, i));
			//abs_mpfcmplx(abs_newx, get_cddvector_i(ans, i));
			_bncavx2_rcdd_abs_dd(absmodval, &modval);
			_bncavx2_rcdd_abs_dd(abs_x, get_cddvector_i(x_init, i));
			_bncavx2_rcdd_abs_dd(abs_newx, get_cddvector_i(ans, i));

			_bncavx2_rdd_add(mpftmp, abs_x, abs_newx);
			_bncavx2_rdd_mul(mpftmp, mpftmp, rel_eps);
			_bncavx2_rdd_add(mpftmp, mpftmp, abs_eps);
			if( _bncavx2_rdd_cmp(absmodval, mpftmp) > 0 )
				flag = 1;

		}

		/* check convergence */
		if(flag == 0)
			break;

		subst_cddvector(x_init, ans);

	}
	return times;
}
#endif // __AVX2__
#endif // 0

// deflation from highest degree coef
// ret org(x) / (x - alpha)
void deflation_ddpoly(DDPoly ret, DDPoly org, double alpha[DDSIZE])
{
	long int i;
	double tmp[DDSIZE];

	set_ddpoly_i(ret, org->deg - 1, get_ddpoly_i(org, org->deg));
	for(i = org->deg - 2; i >= 0; i--)
	{
		rdd_mul(tmp, get_ddpoly_i(ret, i + 1), alpha);
		rdd_add(tmp, tmp, get_ddpoly_i(org, i + 1));
		set_ddpoly_i(ret, i, tmp);
	}

	setdegree_ddpoly(ret);
}

// deflation from highest degree coef and evaluation
// ret org(x) / (x - alpha)
void eval_deflation_ddpoly(double ret[DDSIZE], DDPoly org, double alpha[DDSIZE], double x[DDSIZE])
{
	long int i;
	double tmp[DDSIZE], new_coef_i[DDSIZE];

	//set_ddpoly_i(new_coef_i, org->deg - 1, get_ddpoly_i(org, org->deg));
	//set_ddpoly_i(ret, org->deg - 1, get_ddpoly_i(org, org->deg));
	rdd_set(new_coef_i, get_ddpoly_i(org, org->deg));
	rdd_set(ret, get_ddpoly_i(org, org->deg));
	for(i = org->deg - 2; i >= 0; i--)
	{
		rdd_mul(ret, ret, x);
		//rdd_mul(tmp, get_ddpoly_i(ret, i + 1), alpha);
		rdd_mul(tmp, new_coef_i, alpha);
		rdd_add(new_coef_i, tmp, get_ddpoly_i(org, i + 1));
		//set_ddpoly_i(ret, i, new_coef_i);
		rdd_add(ret, ret, new_coef_i);
	}
}

// Ozawa's initial radius : r := (p(center) / a_n)^(1/n)
void dd_dka_ozawa_radius(double ret[DDSIZE], DDPoly func)
{
	double barycentric_point[DDSIZE];
	DDPoly in_func[2];

	in_func[0] = init_ddpoly(func->max_len);
	in_func[1] = init_ddpoly(func->max_len);

	subst_ddpoly(in_func[0], func);

	do{
		dd_dka_center(barycentric_point, in_func[0]);
		eval_ddpoly(ret, in_func[0], barycentric_point);

		rdd_div(ret, ret, get_ddpoly_i(in_func[0], in_func[0]->deg));
		rdd_abs(ret, ret);
		//dd_rootn_ui(ret, ret, (unsigned long)(in_func[0]->deg), GMP_RNDN);

		// normal return
		//if(mpfr_regular_p(ret) != 0)
			break;

		deflation_ddpoly(in_func[1], in_func[0], barycentric_point);
		subst_ddpoly(in_func[0], in_func[1]);
	} while(1);

	free_ddpoly(in_func[0]);
	free_ddpoly(in_func[1]);
}

// ------------------------------------
// New implementation
// 2025-01-16(Thu)
// ------------------------------------

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void dd_dka_init2(CDDVector x_init, DDPoly func, void (* get_radius)(double *, DDPoly), void (* get_center)(double *, DDPoly))
{
	long int i, itmp;
	double rad[DDSIZE], cen[DDSIZE], an[DDSIZE], tmp[DDSIZE], re_cinit[DDSIZE], im_cinit[DDSIZE];
    double cosine[DDSIZE], sine[DDSIZE];
	//MPFCmplx cinit;
	cddfloat cinit;

	//rdd_dka_radius(rad, func);
	//rdd_dka_center(cen, func);
	get_radius(rad, func);
	get_center(cen, func);

//	rdd_out_str(stdout, 10, 0, rad); printf(", "); rdd_out_str(stdout, 10, 0, cen); printf("\n");

	for(i = 0; i < func->deg; i++)
	{
		//set0_mpfcmplx(cinit);
        rcdd_set_ui(&cinit, 0UL);
		rdd_set_d(tmp, (double)(2.0 * M_PI * i / func->deg + 3.0 / (2.0 * func->deg)));

		//iexp_mpfcmplx(cinit, tmp);
		rdd_cos(cosine, tmp);
		rdd_sin(sine, tmp);
		rcdd_set_dd_dd(&cinit, cosine, sine);
        //get_real_mpfcmplx(re_cinit, cinit);
		//get_image_mpfcmplx(im_cinit, cinit);
		rdd_set(re_cinit, cinit.val_re);
		rdd_set(im_cinit, cinit.val_im);

		/* re_cinit = cen + rad * re_cinit */
		rdd_mul(re_cinit, rad, re_cinit);
		rdd_add(re_cinit, cen, re_cinit);

		/* im_cinit = rad * im_cinit */
		rdd_mul(im_cinit, rad, im_cinit);

		//set_real_mpfcmplx(cinit, re_cinit);
		//set_image_mpfcmplx(cinit, im_cinit);
		rcdd_set_dd_dd(&cinit,
			re_cinit,
			im_cinit
		);

		//abs_mpfcmplx(tmp, cinit);
		rcdd_abs_dd(tmp, &cinit);
//		printf("%5d(%f) ", i, mpf2double(tmp)); print_mpfcmplx(cinit);

		set_cddvector_i(x_init, i, &cinit);
	}
}


/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int dd_dka_mod(CDDVector ans, CDDVector x_init, DDPoly func, long int maxtimes, double abs_eps[DDSIZE], double rel_eps[DDSIZE])
{
	long int times, i, j, deg, flag;
	double absmodval[DDSIZE], abs_x[DDSIZE], abs_newx[DDSIZE], mpftmp[DDSIZE];
	//MPFCmplx modval, up_modval, low_modval, tmp;
	cddfloat modval, up_modval, low_modval, tmp;

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
            rcdd_set_ui_ui(&low_modval, 1UL, 0UL);

			for(j = 0; j < i; j++)
			{
				//set0_mpfcmplx(tmp);
                rcdd_set_ui(&tmp, 0UL);
				rcdd_sub(
					&tmp,
					get_cddvector_i(x_init, i),
					get_cddvector_i(x_init, j)
				);
				rcdd_mul(&low_modval, &low_modval, &tmp);
			}
			for(j = i + 1; j < deg; j++)
			{
				rcdd_set_ui(&tmp, 0UL);
				rcdd_sub(
					&tmp,
					get_cddvector_i(x_init, i),
					get_cddvector_i(x_init, j)
				);
				rcdd_mul(&low_modval, &low_modval, &tmp);
			}
			rcdd_mul_dd(&low_modval, &low_modval, get_ddpoly_i(func, func->deg));
			ceval_ddpoly(&up_modval, func, get_cddvector_i(x_init, i));

			rcdd_div(&modval, &up_modval, &low_modval);
			rcdd_sub(&tmp, get_cddvector_i(x_init, i), &modval);
			set_cddvector_i(ans, i, &tmp);

//			get_real_mpfcmplx(mpftmp, get_cddvector_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			rcdd_abs_dd(absmodval, &modval);
			rcdd_abs_dd(abs_x, get_cddvector_i(x_init, i));
			rcdd_abs_dd(abs_newx, get_cddvector_i(ans, i));

			rdd_add(mpftmp, abs_x, abs_newx);
			rdd_mul(mpftmp, mpftmp, rel_eps);
			rdd_add(mpftmp, mpftmp, abs_eps);
			if( rdd_cmp(absmodval, mpftmp) > 0 )
				flag = 1;

		}

		/* check convergence */
		if(flag == 0)
			break;

		subst_cddvector(x_init, ans);

	}

	return times;
}

/* Petkovic Method : 3rd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int dd_petckovic(CDDVector ans, CDDVector x_init, DDPoly func, long int maxtimes, double abs_eps[DDSIZE], double rel_eps[DDSIZE])
{
	long int times, i, j, deg, flag;
	double absmodval[DDSIZE], abs_x[DDSIZE], abs_newx[DDSIZE], mpftmp[DDSIZE], mpftmp2[DDSIZE];
	cddfloat modval, up_modval, low_modval, tmp;
	DDPoly in_func;

//	printf("\n");
//	print_cfarray(x_init);
//	printf("%f, %f\n", abs_eps, rel_eps);

	deg = ans->re->dim;

	in_func = init_ddpoly(func->max_len);
	// mpttmp := 1/a_n
	rdd_ui_div(mpftmp, 1UL, get_ddpoly_i(func, func->deg)); 
	for(i = 0; i <= func->deg; i++)
	{
		rdd_mul(mpftmp2, get_ddpoly_i(func, i), mpftmp);
		set_ddpoly_i(in_func, i, mpftmp2);
	}
	//setdegree_ddpoly(in_func);

	for(times = 0; times <= maxtimes; times++)
	{
//		printf("%5d: ", times); fflush(stdout);

		flag = 0;
		for(i = 0; i < deg; i++)
		{
			// get Weierstrass's correction to low_modval
			//set_real_mpfcmplx_ui(low_modval, 1UL);
			//set_image_mpfcmplx_ui(low_modval, 0UL);
            rcdd_set_ui_ui(&low_modval, 1UL, 0UL);
			for(j = 0; j < i; j++)
			{
				rcdd_set_ui(&tmp, 0UL); //set0_mpfcmplx(tmp);
				rcdd_sub(
					&tmp,
					get_cddvector_i(x_init, i),
					get_cddvector_i(x_init, j)
				);
				rcdd_mul(&low_modval, &low_modval, &tmp);
			}
			for(j = i + 1; j < deg; j++)
			{
				rcdd_set_ui(&tmp, 0UL); //set0_mpfcmplx(tmp);
				rcdd_sub(
					&tmp,
					get_cddvector_i(x_init, i),
					get_cddvector_i(x_init, j)
				);
				rcdd_mul(&low_modval, &low_modval, &tmp);
			}
			//mul_mpfcmplx_mpf(low_modval, low_modval, get_ddpoly_i(func, func->deg));
			rcdd_mul_ui(&low_modval, &low_modval, 2UL); // 1/2 * modval
			ceval_ddpoly(&up_modval, in_func, get_cddvector_i(x_init, i));
			// modval := Weierstrass's correction * 1/2
			rcdd_div(&modval, &up_modval, &low_modval);
			
			// tmp := z - modval
			rcdd_sub(&tmp, get_cddvector_i(x_init, i), &modval);
			ceval_diff_ddpoly(&low_modval, in_func, &tmp);
			
			// new_z := old_z - f(z) / f'(z - 1/2 * W)
			rcdd_div(&modval, &up_modval, &low_modval);
			//mul_mpfcmplx_mpf(modval, modval, get_ddpoly_i(func, func->deg));
			rcdd_sub(&tmp, get_cddvector_i(x_init, i), &modval);
			set_cddvector_i(ans, i, &tmp);

//			get_real_mpfcmplx(mpftmp, get_cddvector_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			rcdd_abs_dd(absmodval, &modval);
			rcdd_abs_dd(abs_x, get_cddvector_i(x_init, i));
			//abs_mpfcmplx(abs_newx, get_cddvector_i(ans, i));

			//rdd_add(mpftmp, abs_x, abs_newx);
			//rdd_mul(mpftmp, mpftmp, rel_eps);
			//rdd_add(mpftmp, mpftmp, abs_eps);
			rdd_mul(mpftmp, abs_x, rel_eps);
			rdd_add(mpftmp, mpftmp, abs_eps);
			// |new_x - old_x| > |old_x| * rel_eps + abs_eps
			if( rdd_cmp(absmodval, mpftmp) > 0 )
				flag = 1;

		}

		/* check convergence */
		if(flag == 0)
			break;

		subst_cddvector(x_init, ans);

	}

	free_ddpoly(in_func);

	return times;
}

/* Aberth Method : 3rd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int dd_aberth(CDDVector ans, CDDVector x_init, DDPoly func, long int maxtimes, double abs_eps[DDSIZE], double rel_eps[DDSIZE])
{
	long int times, i, j, deg, flag;
	double absmodval[DDSIZE], abs_x[DDSIZE], abs_newx[DDSIZE], mpftmp[DDSIZE], mpftmp2[DDSIZE];
	cddfloat modval, up_modval, low_modval, tmp, one;

//	printf("\n");
//	print_cfarray(x_init);
//	printf("%f, %f\n", abs_eps, rel_eps);

	deg = ans->re->dim;

	// complex one := 1
    rcdd_set_ui_ui(&one, 1UL, 0UL);

	for(times = 0; times <= maxtimes; times++)
	{
//		printf("%5d: ", times); fflush(stdout);

		flag = 0;
		for(i = 0; i < deg; i++)
		{
			// low_modval := sum (z_i - z_k)^(-1)
			//set0_mpfcmplx(low_modval);
            rcdd_set_ui(&low_modval, 0UL);
			for(j = 0; j < i; j++)
			{
				rcdd_set_ui(&tmp, 0UL); //set0_mpfcmplx(tmp);
				rcdd_sub(
					&tmp,
					get_cddvector_i(x_init, i),
					get_cddvector_i(x_init, j)
				);
				rcdd_div(&tmp, &one, &tmp);
				rcdd_add(&low_modval, &low_modval, &tmp);
			}
			for(j = i + 1; j < deg; j++)
			{
				rcdd_set_ui(&tmp, 0UL); //set0_mpfcmplx(tmp);
				rcdd_sub(
					&tmp,
					get_cddvector_i(x_init, i),
					get_cddvector_i(x_init, j)
				);
				rcdd_div(&tmp, &one, &tmp);
				rcdd_add(&low_modval, &low_modval, &tmp);
			}
			// up_mpdval := f(z) / f'(z)
			ceval_ddpoly(&up_modval, func, get_cddvector_i(x_init, i));
			ceval_diff_ddpoly(&tmp, func, get_cddvector_i(x_init, i));
			rcdd_div(&up_modval, &up_modval, &tmp);
			
			// low_modval := 1 - up_modval * sum (z_i - z_k)^(-1)
			rcdd_mul(&tmp, &up_modval, &low_modval);
			rcdd_sub(&low_modval, &one, &tmp);
			
			// new_z := old_z - up_modval / low_modval
			rcdd_div(&modval, &up_modval, &low_modval);
			rcdd_sub(&tmp, get_cddvector_i(x_init, i), &modval);
			set_cddvector_i(ans, i, &tmp);

//			get_real_mpfcmplx(mpftmp, get_cddvector_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			rcdd_abs_dd(absmodval, &modval);
			rcdd_abs_dd(abs_x, get_cddvector_i(x_init, i));
			//abs_mpfcmplx(abs_newx, get_cddvector_i(ans, i));

			//rdd_add(mpftmp, abs_x, abs_newx);
			//rdd_mul(mpftmp, mpftmp, rel_eps);
			//rdd_add(mpftmp, mpftmp, abs_eps);
			rdd_mul(mpftmp, abs_x, rel_eps);
			rdd_add(mpftmp, mpftmp, abs_eps);
			// |new_x - old_x| > |old_x| * rel_eps + abs_eps
			if( rdd_cmp(absmodval, mpftmp) > 0 )
				flag = 1;

		}

		/* check convergence */
		if(flag == 0)
			break;

		subst_cddvector(x_init, ans);

	}

	return times;
}
#endif // USE_DDLINEAR

#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus
