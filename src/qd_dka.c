/********************************************************************************/
/* qd_dka.c: Durand-Kerner-Aberth Methods                                       */
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
void qd_dka_center(double ret[QDSIZE], QDPoly func)
{
	rqd_set(ret, get_qdpoly_i(func, func->deg - 1));

	rqd_div(ret, ret, get_qdpoly_i(func, func->deg));
	rqd_div_ui(ret, ret, (unsigned long)func->deg);
	rqd_neg(ret, ret);
}

/* radius = max (m * |a_i / a_deg|)^(1/(deg-i)) */
void qd_dka_radius(double ret[QDSIZE], QDPoly func)
{
	long int i;
	double tmp[QDSIZE], num_nonzero[QDSIZE], an[QDSIZE];
	double dtmp;

	rqd_set_ui(num_nonzero, (unsigned long)num_nonzero_qdpoly(func));
	rqd_set(an, get_qdpoly_i(func, func->deg));
	rqd_set(ret, num_nonzero);
	for(i = func->deg - 1; i >= 0; i--)
	{
		rqd_div(tmp, get_qdpoly_i(func, i), an);
		rqd_mul(tmp, tmp, num_nonzero);
		rqd_abs(tmp, tmp);
		
		/* tmp^(deg-i) */
/*		rqd_ln(tmp, tmp);
		rqd_mul_ui(tmp, tmp, (unsigned long)func->deg - i);
		rqd_exp(tmp, tmp);
*/
		dtmp = rqd_get_d(tmp); dtmp = pow(dtmp, 1.0/(double)(func->deg - i)); rqd_set_d(tmp, dtmp);
		if(rqd_cmp(ret, tmp) < 0)
			rqd_set(ret, tmp);
	}
}

// ------------------------------------
// New implementation
// 2025-01-15 (Wed) T.Kouya
// ------------------------------------

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void qd_dka_init(CQDVector x_init, QDPoly func)
{
	long int i, itmp;
	double rad[QDSIZE], cen[QDSIZE], an[QDSIZE], tmp[QDSIZE], re_cinit[QDSIZE], im_cinit[QDSIZE];
	//MPFCmplx cinit;
	double cosine[QDSIZE], sine[QDSIZE];
	cqdfloat cinit;

	qd_dka_radius(rad, func);
	qd_dka_center(cen, func);

//	rqd_out_str(stdout, 10, 0, rad); printf(", "); rqd_out_str(stdout, 10, 0, cen); printf("\n");

	for(i = 0; i < func->deg; i++)
	{
		//set0_mpfcmplx(cinit);
		rcqd_set_ui(&cinit, 0UL);
		rqd_set_d(tmp, (double)(2.0 * M_PI * i / func->deg + 3.0 / (2.0 * func->deg)));

		//iexp_mpfcmplx(cinit, tmp);
		// ctmp := i * tmp -> exp(i * tmp) := cos(tmp) + i * sin(tmp);
		//rqd_cos(cosine, tmp);
		//rqd_sin(sine, tmp);
		rqd_func_mpfr(cosine, mpfr_cos, tmp);
		rqd_func_mpfr(sine, mpfr_sin, tmp);
		rcqd_set_qd_qd(&cinit, cosine, sine);

		//get_real_mpfcmplx(re_cinit, cinit);
		//get_image_mpfcmplx(im_cinit, cinit);
		rqd_set(re_cinit, cinit.val_re);
		rqd_set(im_cinit, cinit.val_im);

		/* re_cinit = cen + rad * re_cinit */
		rqd_mul(re_cinit, rad, re_cinit);
		rqd_add(re_cinit, cen, re_cinit);

		/* im_cinit = rad * im_cinit */
		rqd_mul(im_cinit, rad, im_cinit);

		//set_real_mpfcmplx(cinit, re_cinit);
		//set_image_mpfcmplx(cinit, im_cinit);
		rcqd_set_qd_qd(&cinit, re_cinit, im_cinit);

		//abs_mpfcmplx(tmp, cinit);
		rcqd_abs_qd(tmp, &cinit);
//		printf("%5d(%f) ", i, mpf2double(tmp)); print_mpfcmplx(cinit);

		set_cqdvector_i(x_init, i, &cinit);
	}
}

/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int qd_dka(CQDVector ans, CQDVector x_init, QDPoly func, long int maxtimes, double abs_eps[QDSIZE], double rel_eps[QDSIZE])
{
	long int times, i, j, deg, flag;
	double absmodval[QDSIZE], abs_x[QDSIZE], abs_newx[QDSIZE], mpftmp[QDSIZE];
	cqdfloat modval, up_modval, low_modval, tmp, x_init_i;

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
			rcqd_set_ui_ui(&low_modval, 1UL, 0UL);
			subst_cqdvector_i(&x_init_i, x_init, i);

			for(j = 0; j < i; j++)
			{
				//set0_mpfcmplx(tmp);
				rcqd_set_ui(&tmp, 0UL);
				//sub_mpfcmplx(
				rcqd_sub(
					&tmp,
					&x_init_i, //get_cqdvector_i(x_init, i),
					get_cqdvector_i(x_init, j)
				);
				//mul2_mpfcmplx(low_modval, tmp);
				rcqd_mul(&low_modval, &low_modval, &tmp);
			}
			for(j = i + 1; j < deg; j++)
			{
				//set0_mpfcmplx(tmp);
				rcqd_set_ui(&tmp, 0UL);
				//sub_mpfcmplx(
				rcqd_sub(
					&tmp,
					&x_init_i, //get_cqdvector_i(x_init, i),
					get_cqdvector_i(x_init, j)
				);
				//mul2_mpfcmplx(low_modval, tmp);
				rcqd_mul(&low_modval, &low_modval, &tmp);
			}
			//mul_mpfcmplx_mpf(low_modval, low_modval, get_qdpoly_i(func, func->deg));
			rcqd_mul_qd(&low_modval, &low_modval, get_qdpoly_i(func, func->deg));
#if (defined(__AVX2__) || defined(__AVX512F__)) || ((defined(__ARM_NEON) || defined(__aarch64__)) && defined(BNC_ENABLE_NEON)) // ARM NEON
			_bncavx2_ceval_qdpoly_estrin(&up_modval, func, &x_init_i); //get_cqdvector_i(x_init, i));
#else // __AVX2__
			ceval_qdpoly(&up_modval, func, &x_init_i); //get_cqdvector_i(x_init, i));
#endif // __AVX2__

			//div_mpfcmplx(modval, up_modval, low_modval);
			rcqd_div(&modval, &up_modval, &low_modval);
			//sub_mpfcmplx(tmp, get_cqdvector_i(x_init, i), modval);
			//rcqd_sub(&tmp, get_cqdvector_i(x_init, i), &modval);
			rcqd_sub(&tmp, &x_init_i, &modval);
			set_cqdvector_i(ans, i, &tmp);

//			get_real_mpfcmplx(mpftmp, get_cqdvector_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			//abs_mpfcmplx(absmodval, modval);
			//abs_mpfcmplx(abs_x, get_cqdvector_i(x_init, i));
			//abs_mpfcmplx(abs_newx, get_cqdvector_i(ans, i));
			rcqd_abs_qd(absmodval, &modval);
			rcqd_abs_qd(abs_x, &x_init_i); // get_cqdvector_i(x_init, i));
			rcqd_abs_qd(abs_newx, get_cqdvector_i(ans, i));

			rqd_add(mpftmp, abs_x, abs_newx);
			rqd_mul(mpftmp, mpftmp, rel_eps);
			rqd_add(mpftmp, mpftmp, abs_eps);
			if( rqd_cmp(absmodval, mpftmp) > 0 )
				flag = 1;

		}

		/* check convergence */
		if(flag == 0)
			break;

		subst_cqdvector(x_init, ans);

	}
	return times;
}

#if 0
#if defined(__AVX2__) && !defined(__AVX512F__) // __AVX2__
/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int _bncavx2_qd_dka(CQDVector ans, CQDVector x_init, QDPoly func, long int maxtimes, double abs_eps[QDSIZE], double rel_eps[QDSIZE])
{
	long int times, i, j, deg, flag;
	//double absmodval[QDSIZE], abs_x[QDSIZE], abs_newx[QDSIZE], mpftmp[QDSIZE];
	//cqdfloat modval, up_modval, low_modval, tmp;
	__m256d absmodval[QDSIZE], abs_x[QDSIZE], abs_newx[QDSIZE], mpftmp[QDSIZE];
	__m256d modval_re[QDSIZE], up_modval_re[QDSIZE], low_modval_re[QDSIZE], tmp_re[QDSIZE];
	__m256d modval_im[QDSIZE], up_modval_im[QDSIZE], low_modval_im[QDSIZE], tmp_im[QDSIZE];
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
			_bncavx2_rcqd_set1_ui_ui(low_modval_re, low_modval_im, 1UL, 0UL);

			for(j = 0; j < i; j++)
			{
				//set0_mpfcmplx(tmp);
				_bncavx2_rcqd_set1_ui(tmp, 0UL);
				//sub_mpfcmplx(
				_bncavx2_rcqd_sub(
					tmp,
					get_cqdvector_i(x_init, i),
					get_cqdvector_i(x_init, j)
				);
				//mul2_mpfcmplx(low_modval, tmp);
				_bncavx2_rcqd_mul(&low_modval, &low_modval, &tmp);
			}
			for(j = i + 1; j < deg; j++)
			{
				//set0_mpfcmplx(tmp);
				_bncavx2_rcqd_set_ui(&tmp, 0UL);
				//sub_mpfcmplx(
				_bncavx2_rcqd_sub(
					&tmp,
					get_cqdvector_i(x_init, i),
					get_cqdvector_i(x_init, j)
				);
				//mul2_mpfcmplx(low_modval, tmp);
				_bncavx2_rcqd_mul(&low_modval, &low_modval, &tmp);
			}
			//mul_mpfcmplx_mpf(low_modval, low_modval, get_qdpoly_i(func, func->deg));
			_bncavx2_rcqd_mul_qd(low_modval_re, low_modval_im, get_qdpoly_i(func, func->deg));
			_bncavx2_ceval_qdpoly_horner(&up_modval, func, get_cqdvector_i(x_init, i));


			//div_mpfcmplx(modval, up_modval, low_modval);
			_bncavx2_rcqd_div(&modval, &up_modval, &low_modval);
			//sub_mpfcmplx(tmp, get_cqdvector_i(x_init, i), modval);
			_bncavx2_rcqd_sub(&tmp, get_cqdvector_i(x_init, i), &modval);
			set_cqdvector_i(ans, i, &tmp);

//			get_real_mpfcmplx(mpftmp, get_cqdvector_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			//abs_mpfcmplx(absmodval, modval);
			//abs_mpfcmplx(abs_x, get_cqdvector_i(x_init, i));
			//abs_mpfcmplx(abs_newx, get_cqdvector_i(ans, i));
			_bncavx2_rcqd_abs_qd(absmodval, &modval);
			_bncavx2_rcqd_abs_qd(abs_x, get_cqdvector_i(x_init, i));
			_bncavx2_rcqd_abs_qd(abs_newx, get_cqdvector_i(ans, i));

			_bncavx2_rqd_add(mpftmp, abs_x, abs_newx);
			_bncavx2_rqd_mul(mpftmp, mpftmp, rel_eps);
			_bncavx2_rqd_add(mpftmp, mpftmp, abs_eps);
			if( _bncavx2_rqd_cmp(absmodval, mpftmp) > 0 )
				flag = 1;

		}

		/* check convergence */
		if(flag == 0)
			break;

		subst_cqdvector(x_init, ans);

	}
	return times;
}
#endif // __AVX2__
#endif // 0

// deflation from highest degree coef
// ret org(x) / (x - alpha)
void deflation_qdpoly(QDPoly ret, QDPoly org, double alpha[QDSIZE])
{
	long int i;
	double tmp[QDSIZE];

	set_qdpoly_i(ret, org->deg - 1, get_qdpoly_i(org, org->deg));
	for(i = org->deg - 2; i >= 0; i--)
	{
		rqd_mul(tmp, get_qdpoly_i(ret, i + 1), alpha);
		rqd_add(tmp, tmp, get_qdpoly_i(org, i + 1));
		set_qdpoly_i(ret, i, tmp);
	}

	setdegree_qdpoly(ret);
}

// deflation from highest degree coef and evaluation
// ret org(x) / (x - alpha)
void eval_deflation_qdpoly(double ret[QDSIZE], QDPoly org, double alpha[QDSIZE], double x[QDSIZE])
{
	long int i;
	double tmp[QDSIZE], new_coef_i[QDSIZE];

	//set_qdpoly_i(new_coef_i, org->deg - 1, get_qdpoly_i(org, org->deg));
	//set_qdpoly_i(ret, org->deg - 1, get_qdpoly_i(org, org->deg));
	rqd_set(new_coef_i, get_qdpoly_i(org, org->deg));
	rqd_set(ret, get_qdpoly_i(org, org->deg));
	for(i = org->deg - 2; i >= 0; i--)
	{
		rqd_mul(ret, ret, x);
		//rqd_mul(tmp, get_qdpoly_i(ret, i + 1), alpha);
		rqd_mul(tmp, new_coef_i, alpha);
		rqd_add(new_coef_i, tmp, get_qdpoly_i(org, i + 1));
		//set_qdpoly_i(ret, i, new_coef_i);
		rqd_add(ret, ret, new_coef_i);
	}
}

// Ozawa's initial radius : r := (p(center) / a_n)^(1/n)
void qd_dka_ozawa_radius(double ret[QDSIZE], QDPoly func)
{
	double barycentric_point[QDSIZE];
	QDPoly in_func[2];

	in_func[0] = init_qdpoly(func->max_len);
	in_func[1] = init_qdpoly(func->max_len);

	subst_qdpoly(in_func[0], func);

	do{
		qd_dka_center(barycentric_point, in_func[0]);
		eval_qdpoly(ret, in_func[0], barycentric_point);

		rqd_div(ret, ret, get_qdpoly_i(in_func[0], in_func[0]->deg));
		rqd_abs(ret, ret);
		//qd_rootn_ui(ret, ret, (unsigned long)(in_func[0]->deg), GMP_RNDN);

		// normal return
		//if(mpfr_regular_p(ret) != 0)
			break;

		deflation_qdpoly(in_func[1], in_func[0], barycentric_point);
		subst_qdpoly(in_func[0], in_func[1]);
	} while(1);

	free_qdpoly(in_func[0]);
	free_qdpoly(in_func[1]);
}

// ------------------------------------
// New implementation
// 2025-01-16(Thu)
// ------------------------------------

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void qd_dka_init2(CQDVector x_init, QDPoly func, void (* get_radius)(double *, QDPoly), void (* get_center)(double *, QDPoly))
{
	long int i, itmp;
	double rad[QDSIZE], cen[QDSIZE], an[QDSIZE], tmp[QDSIZE], re_cinit[QDSIZE], im_cinit[QDSIZE];
    double cosine[QDSIZE], sine[QDSIZE];
	//MPFCmplx cinit;
	cqdfloat cinit;

	//rqd_dka_radius(rad, func);
	//rqd_dka_center(cen, func);
	get_radius(rad, func);
	get_center(cen, func);

//	rqd_out_str(stdout, 10, 0, rad); printf(", "); rqd_out_str(stdout, 10, 0, cen); printf("\n");

	for(i = 0; i < func->deg; i++)
	{
		//set0_mpfcmplx(cinit);
        rcqd_set_ui(&cinit, 0UL);
		rqd_set_d(tmp, (double)(2.0 * M_PI * i / func->deg + 3.0 / (2.0 * func->deg)));

		//iexp_mpfcmplx(cinit, tmp);
		rqd_func_mpfr(cosine, mpfr_cos, tmp);
		rqd_func_mpfr(sine, mpfr_sin, tmp);
		rcqd_set_qd_qd(&cinit, cosine, sine);
        //get_real_mpfcmplx(re_cinit, cinit);
		//get_image_mpfcmplx(im_cinit, cinit);
		rqd_set(re_cinit, cinit.val_re);
		rqd_set(im_cinit, cinit.val_im);

		/* re_cinit = cen + rad * re_cinit */
		rqd_mul(re_cinit, rad, re_cinit);
		rqd_add(re_cinit, cen, re_cinit);

		/* im_cinit = rad * im_cinit */
		rqd_mul(im_cinit, rad, im_cinit);

		//set_real_mpfcmplx(cinit, re_cinit);
		//set_image_mpfcmplx(cinit, im_cinit);
		rcqd_set_qd_qd(&cinit,
			re_cinit,
			im_cinit
		);

		//abs_mpfcmplx(tmp, cinit);
		rcqd_abs_qd(tmp, &cinit);
//		printf("%5d(%f) ", i, mpf2double(tmp)); print_mpfcmplx(cinit);

		set_cqdvector_i(x_init, i, &cinit);
	}
}


/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int qd_dka_mod(CQDVector ans, CQDVector x_init, QDPoly func, long int maxtimes, double abs_eps[QDSIZE], double rel_eps[QDSIZE])
{
	long int times, i, j, deg, flag;
	double absmodval[QDSIZE], abs_x[QDSIZE], abs_newx[QDSIZE], mpftmp[QDSIZE];
	//MPFCmplx modval, up_modval, low_modval, tmp;
	cqdfloat modval, up_modval, low_modval, tmp;

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
            rcqd_set_ui_ui(&low_modval, 1UL, 0UL);

			for(j = 0; j < i; j++)
			{
				//set0_mpfcmplx(tmp);
                rcqd_set_ui(&tmp, 0UL);
				rcqd_sub(
					&tmp,
					get_cqdvector_i(x_init, i),
					get_cqdvector_i(x_init, j)
				);
				rcqd_mul(&low_modval, &low_modval, &tmp);
			}
			for(j = i + 1; j < deg; j++)
			{
				rcqd_set_ui(&tmp, 0UL);
				rcqd_sub(
					&tmp,
					get_cqdvector_i(x_init, i),
					get_cqdvector_i(x_init, j)
				);
				rcqd_mul(&low_modval, &low_modval, &tmp);
			}
			rcqd_mul_qd(&low_modval, &low_modval, get_qdpoly_i(func, func->deg));
			ceval_qdpoly(&up_modval, func, get_cqdvector_i(x_init, i));

			rcqd_div(&modval, &up_modval, &low_modval);
			rcqd_sub(&tmp, get_cqdvector_i(x_init, i), &modval);
			set_cqdvector_i(ans, i, &tmp);

//			get_real_mpfcmplx(mpftmp, get_cqdvector_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			rcqd_abs_qd(absmodval, &modval);
			rcqd_abs_qd(abs_x, get_cqdvector_i(x_init, i));
			rcqd_abs_qd(abs_newx, get_cqdvector_i(ans, i));

			rqd_add(mpftmp, abs_x, abs_newx);
			rqd_mul(mpftmp, mpftmp, rel_eps);
			rqd_add(mpftmp, mpftmp, abs_eps);
			if( rqd_cmp(absmodval, mpftmp) > 0 )
				flag = 1;

		}

		/* check convergence */
		if(flag == 0)
			break;

		subst_cqdvector(x_init, ans);

	}

	return times;
}

/* Petkovic Method : 3rd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int qd_petckovic(CQDVector ans, CQDVector x_init, QDPoly func, long int maxtimes, double abs_eps[QDSIZE], double rel_eps[QDSIZE])
{
	long int times, i, j, deg, flag;
	double absmodval[QDSIZE], abs_x[QDSIZE], abs_newx[QDSIZE], mpftmp[QDSIZE], mpftmp2[QDSIZE];
	cqdfloat modval, up_modval, low_modval, tmp;
	QDPoly in_func;

//	printf("\n");
//	print_cfarray(x_init);
//	printf("%f, %f\n", abs_eps, rel_eps);

	deg = ans->re->dim;

	in_func = init_qdpoly(func->max_len);
	// mpttmp := 1/a_n
	rqd_ui_div(mpftmp, 1UL, get_qdpoly_i(func, func->deg)); 
	for(i = 0; i <= func->deg; i++)
	{
		rqd_mul(mpftmp2, get_qdpoly_i(func, i), mpftmp);
		set_qdpoly_i(in_func, i, mpftmp2);
	}
	//setdegree_qdpoly(in_func);

	for(times = 0; times <= maxtimes; times++)
	{
//		printf("%5d: ", times); fflush(stdout);

		flag = 0;
		for(i = 0; i < deg; i++)
		{
			// get Weierstrass's correction to low_modval
			//set_real_mpfcmplx_ui(low_modval, 1UL);
			//set_image_mpfcmplx_ui(low_modval, 0UL);
            rcqd_set_ui_ui(&low_modval, 1UL, 0UL);
			for(j = 0; j < i; j++)
			{
				rcqd_set_ui(&tmp, 0UL); //set0_mpfcmplx(tmp);
				rcqd_sub(
					&tmp,
					get_cqdvector_i(x_init, i),
					get_cqdvector_i(x_init, j)
				);
				rcqd_mul(&low_modval, &low_modval, &tmp);
			}
			for(j = i + 1; j < deg; j++)
			{
				rcqd_set_ui(&tmp, 0UL); //set0_mpfcmplx(tmp);
				rcqd_sub(
					&tmp,
					get_cqdvector_i(x_init, i),
					get_cqdvector_i(x_init, j)
				);
				rcqd_mul(&low_modval, &low_modval, &tmp);
			}
			//mul_mpfcmplx_mpf(low_modval, low_modval, get_qdpoly_i(func, func->deg));
			rcqd_mul_ui(&low_modval, &low_modval, 2UL); // 1/2 * modval
			ceval_qdpoly(&up_modval, in_func, get_cqdvector_i(x_init, i));
			// modval := Weierstrass's correction * 1/2
			rcqd_div(&modval, &up_modval, &low_modval);
			
			// tmp := z - modval
			rcqd_sub(&tmp, get_cqdvector_i(x_init, i), &modval);
			ceval_diff_qdpoly(&low_modval, in_func, &tmp);
			
			// new_z := old_z - f(z) / f'(z - 1/2 * W)
			rcqd_div(&modval, &up_modval, &low_modval);
			//mul_mpfcmplx_mpf(modval, modval, get_qdpoly_i(func, func->deg));
			rcqd_sub(&tmp, get_cqdvector_i(x_init, i), &modval);
			set_cqdvector_i(ans, i, &tmp);

//			get_real_mpfcmplx(mpftmp, get_cqdvector_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			rcqd_abs_qd(absmodval, &modval);
			rcqd_abs_qd(abs_x, get_cqdvector_i(x_init, i));
			//abs_mpfcmplx(abs_newx, get_cqdvector_i(ans, i));

			//rqd_add(mpftmp, abs_x, abs_newx);
			//rqd_mul(mpftmp, mpftmp, rel_eps);
			//rqd_add(mpftmp, mpftmp, abs_eps);
			rqd_mul(mpftmp, abs_x, rel_eps);
			rqd_add(mpftmp, mpftmp, abs_eps);
			// |new_x - old_x| > |old_x| * rel_eps + abs_eps
			if( rqd_cmp(absmodval, mpftmp) > 0 )
				flag = 1;

		}

		/* check convergence */
		if(flag == 0)
			break;

		subst_cqdvector(x_init, ans);

	}

	free_qdpoly(in_func);

	return times;
}

/* Aberth Method : 3rd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int qd_aberth(CQDVector ans, CQDVector x_init, QDPoly func, long int maxtimes, double abs_eps[QDSIZE], double rel_eps[QDSIZE])
{
	long int times, i, j, deg, flag;
	double absmodval[QDSIZE], abs_x[QDSIZE], abs_newx[QDSIZE], mpftmp[QDSIZE], mpftmp2[QDSIZE];
	cqdfloat modval, up_modval, low_modval, tmp, one, x_init_i, x_init_j;

//	printf("\n");
//	print_cfarray(x_init);
//	printf("%f, %f\n", abs_eps, rel_eps);

	deg = ans->re->dim;

	// complex one := 1
    rcqd_set_ui_ui(&one, 1UL, 0UL);

	for(times = 0; times <= maxtimes; times++)
	{
//		printf("%5d: ", times); fflush(stdout);

		flag = 0;
		for(i = 0; i < deg; i++)
		{
			subst_cqdvector_i(&x_init_i, x_init, i);

			// low_modval := sum (z_i - z_k)^(-1)
			//set0_mpfcmplx(low_modval);
            rcqd_set_ui(&low_modval, 0UL);
			for(j = 0; j < i; j++)
			{
				rcqd_set_ui(&tmp, 0UL); //set0_mpfcmplx(tmp);
				rcqd_sub(
					&tmp,
					&x_init_i, // get_cqdvector_i(x_init, i),
					get_cqdvector_i(x_init, j)
				);
				rcqd_div(&tmp, &one, &tmp);
				rcqd_add(&low_modval, &low_modval, &tmp);
			}
			for(j = i + 1; j < deg; j++)
			{
				rcqd_set_ui(&tmp, 0UL); //set0_mpfcmplx(tmp);
				rcqd_sub(
					&tmp,
					&x_init_i, // get_cqdvector_i(x_init, i),
					get_cqdvector_i(x_init, j)
				);
				rcqd_div(&tmp, &one, &tmp);
				rcqd_add(&low_modval, &low_modval, &tmp);
			}
			// up_mpdval := f(z) / f'(z)
			ceval_qdpoly(&up_modval, func, &x_init_i); // get_cqdvector_i(x_init, i));
			ceval_diff_qdpoly(&tmp, func, &x_init_i); // get_cqdvector_i(x_init, i));
			rcqd_div(&up_modval, &up_modval, &tmp);
			
			// low_modval := 1 - up_modval * sum (z_i - z_k)^(-1)
			rcqd_mul(&tmp, &up_modval, &low_modval);
			rcqd_sub(&low_modval, &one, &tmp);
			
			// new_z := old_z - up_modval / low_modval
			rcqd_div(&modval, &up_modval, &low_modval);
			rcqd_sub(&tmp, &x_init_i, &modval); // get_cqdvector_i(x_init, i), &modval);
			set_cqdvector_i(ans, i, &tmp);

//			get_real_mpfcmplx(mpftmp, get_cqdvector_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			rcqd_abs_qd(absmodval, &modval);
			rcqd_abs_qd(abs_x, &x_init_i); // get_cqdvector_i(x_init, i));
			//abs_mpfcmplx(abs_newx, get_cqdvector_i(ans, i));

			//rqd_add(mpftmp, abs_x, abs_newx);
			//rqd_mul(mpftmp, mpftmp, rel_eps);
			//rqd_add(mpftmp, mpftmp, abs_eps);
			rqd_mul(mpftmp, abs_x, rel_eps);
			rqd_add(mpftmp, mpftmp, abs_eps);
			// |new_x - old_x| > |old_x| * rel_eps + abs_eps
			if( rqd_cmp(absmodval, mpftmp) > 0 )
				flag = 1;

		}

		/* check convergence */
		if(flag == 0)
			break;

		subst_cqdvector(x_init, ans);

	}

	return times;
}
#endif // USE_TDLINEAR

#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus
