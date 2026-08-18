/********************************************************************************/
/* dka_omp.c: Durand-Kerner-Aberth Methods                                      */
/* Copyright (C) 2022 Tomonori Kouya                                            */
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
//#include "bnc.h"
#include "bncmatmul.h"
#include "bncomp.h"
//#include "omp.h"
//#define BNCOMP_MAX_NUM_THREADS 128

#include <stdio.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#ifdef USE_GMP

/* mpf_t */

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void _bncomp_mpf_dka_init2(CMPFArray x_init, MPFPoly func, void (* get_radius)(mpf_t, MPFPoly), void (* get_center)(mpf_t, MPFPoly))
{
	int thread_index, thread_num;
	long int i, itmp;
	mpf_t rad, cen;
	mpf_t an[BNCOMP_MAX_NUM_THREADS], tmp[BNCOMP_MAX_NUM_THREADS], re_cinit[BNCOMP_MAX_NUM_THREADS], im_cinit[BNCOMP_MAX_NUM_THREADS];
	//MPFCmplx cinit[BNCOMP_MAX_NUM_THREADS];
	mpf_t cosine[BNCOMP_MAX_NUM_THREADS], sine[BNCOMP_MAX_NUM_THREADS];
	mpc_t cinit[BNCOMP_MAX_NUM_THREADS];

	mpf_init2(rad, x_init->prec);
	mpf_init2(cen, x_init->prec);

	thread_num = omp_get_num_threads();
	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(an[thread_index] , x_init->prec);
		mpf_init2(tmp[thread_index], x_init->prec);
		mpf_init2(re_cinit[thread_index], x_init->prec);
		mpf_init2(im_cinit[thread_index], x_init->prec);
		mpf_init2(cosine[thread_index], x_init->prec);
		mpf_init2(sine[thread_index], x_init->prec);

		//cinit[thread_index] = init_mpfcmplx();
		mpc_init2(cinit[thread_index], x_init->prec);

	}

	//mpf_dka_radius(rad, func);
	get_radius(rad, func);
	//mpf_dka_center(cen, func);
	get_center(cen, func);

//	mpf_out_str(stdout, 10, 0, rad); printf(", "); mpf_out_str(stdout, 10, 0, cen); printf("\n");

	//cinit = init_mpfcmplx();

	#pragma omp parallel for private(i, thread_index) shared(rad, cen)
	for(i = 0; i < func->deg; i++)
	{
		thread_index = omp_get_thread_num();

		//set0_mpfcmplx(cinit[thread_index]);		
		mpc_set_ui(cinit[thread_index], 0UL, get_bnc_default_rounding_mode_c());
#ifndef USE_MPFR
		mpf_set_d(tmp[thread_index], (double)(2.0 * M_PI * i / func->deg + 3.0 / (2.0 * func->deg)));
#else // USE_MPFR
		mpfr_set_d(tmp[thread_index], (double)(2.0 * M_PI * i / func->deg + 3.0 / (2.0 * func->deg)), GMP_RNDN); //bnc_default_rounding_mode);
#endif // USE_MPFR
		//iexp_mpfcmplx(cinit[thread_index], tmp[thread_index]);
		mpf_cos(cosine[thread_index], tmp[thread_index]);
		mpf_sin(sine[thread_index], tmp[thread_index]);
		mpc_set_fr_fr(cinit[thread_index], cosine[thread_index], sine[thread_index], get_bnc_default_rounding_mode_c());

		//get_real_mpfcmplx(re_cinit[thread_index], cinit[thread_index]);
		//get_image_mpfcmplx(im_cinit[thread_index], cinit[thread_index]);
		mpf_set(re_cinit[thread_index], mpc_realref(cinit[thread_index]));
		mpf_set(im_cinit[thread_index], mpc_imagref(cinit[thread_index]));

		/* re_cinit = cen + rad * re_cinit */
		mpf_mul(re_cinit[thread_index], rad, re_cinit[thread_index]);
		mpf_add(re_cinit[thread_index], cen, re_cinit[thread_index]);

		/* im_cinit = rad * im_cinit */
		mpf_mul(im_cinit[thread_index], rad, im_cinit[thread_index]);

		//set_real_mpfcmplx(cinit[thread_index], re_cinit[thread_index]);
		//set_image_mpfcmplx(cinit[thread_index], im_cinit[thread_index]);
		mpc_set_fr_fr(cinit[thread_index],
			re_cinit[thread_index],
			im_cinit[thread_index],
			get_bnc_default_rounding_mode_c()
		);

		//abs_mpfcmplx(tmp[thread_index], cinit[thread_index]);
		mpc_abs(tmp[thread_index], cinit[thread_index], get_bnc_default_rounding_mode());
//		printf("%5d(%f) ", i, mpf2double(tmp)); print_mpfcmplx(cinit);

		set_cmpfarray_i(x_init, i, cinit[thread_index]);
	}

	mpf_clear(rad);
	mpf_clear(cen);
	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();

		mpf_clear(an[thread_index]);
		mpf_clear(tmp[thread_index]);
		mpf_clear(re_cinit[thread_index]);
		mpf_clear(im_cinit[thread_index]);
		mpf_clear(cosine[thread_index]);
		mpf_clear(sine[thread_index]);
		//free_mpfcmplx(cinit[thread_index]);
		mpc_clear(cinit[thread_index]);
	}
}

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void _bncomp_mpf_dka_init(CMPFArray x_init, MPFPoly func)
{
	_bncomp_mpf_dka_init2(x_init, func, mpf_dka_radius, mpf_dka_center);
}

/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int _bncomp_mpf_dka(CMPFArray ans, CMPFArray x_init, MPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps)
{
	int thread_index, thread_num;
	long int times, i, j, deg, flag;
	mpf_t absmodval[BNCOMP_MAX_NUM_THREADS], abs_x[BNCOMP_MAX_NUM_THREADS], abs_newx[BNCOMP_MAX_NUM_THREADS], mpftmp[BNCOMP_MAX_NUM_THREADS];
//	MPFCmplx modval[BNCOMP_MAX_NUM_THREADS], up_modval[BNCOMP_MAX_NUM_THREADS], low_modval[BNCOMP_MAX_NUM_THREADS], tmp[BNCOMP_MAX_NUM_THREADS];
	mpc_t modval[BNCOMP_MAX_NUM_THREADS], up_modval[BNCOMP_MAX_NUM_THREADS], low_modval[BNCOMP_MAX_NUM_THREADS], tmp[BNCOMP_MAX_NUM_THREADS];

//	printf("\n");
//	print_cfarray(x_init);
//	printf("%f, %f\n", abs_eps, rel_eps);

	deg = ans->size;

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();

		mpf_init2(absmodval[thread_index], ans->prec);
		mpf_init2(abs_x[thread_index], ans->prec);
		mpf_init2(abs_newx[thread_index], ans->prec);
		mpf_init2(mpftmp[thread_index], ans->prec);

		mpc_init2(modval[thread_index], ans->prec); // = init2_mpfcmplx(ans->prec);
		mpc_init2(low_modval[thread_index], ans->prec); // = init2_mpfcmplx(ans->prec);
		mpc_init2(up_modval[thread_index], ans->prec); // = init2_mpfcmplx(ans->prec);
		mpc_init2(tmp[thread_index], ans->prec); // = init2_mpfcmplx(ans->prec);
	}

	for(times = 0; times <= maxtimes; times++)
	{
		//printf("%5d: ", times); fflush(stdout);

		flag = 0;
		
		#pragma omp parallel for private(i, j, thread_index) shared(modval, up_modval, low_modval, tmp, mpftmp, x_init, func, ans, rel_eps, abs_eps)
		for(i = 0; i < deg; i++)
		{
			thread_index = omp_get_thread_num();

			//set_real_mpfcmplx_ui(low_modval[thread_index], 1UL);
			//set_image_mpfcmplx_ui(low_modval[thread_index], 0UL);
			mpc_set_ui_ui(low_modval[thread_index], 1UL, 0UL, get_bnc_default_rounding_mode_c());

			for(j = 0; j < i; j++)
			{
				//set0_mpfcmplx(tmp[thread_index]);
				mpc_set_ui(tmp[thread_index], 0UL, get_bnc_default_rounding_mode_c());
				mpc_sub(
					tmp[thread_index],
					get_cmpfarray_i(x_init, i),
					get_cmpfarray_i(x_init, j), get_bnc_default_rounding_mode_c()
				);
				mpc_mul(low_modval[thread_index], low_modval[thread_index], tmp[thread_index], get_bnc_default_rounding_mode_c());
			}
			for(j = i + 1; j < deg; j++)
			{
				mpc_set_ui(tmp[thread_index], 0UL, get_bnc_default_rounding_mode_c());
				mpc_sub(
					tmp[thread_index],
					get_cmpfarray_i(x_init, i),
					get_cmpfarray_i(x_init, j), get_bnc_default_rounding_mode_c()
				);
				mpc_mul(low_modval[thread_index], low_modval[thread_index], tmp[thread_index], get_bnc_default_rounding_mode_c());
			}
			mpc_mul_fr(low_modval[thread_index], low_modval[thread_index], get_mpfpoly_i(func, func->deg), get_bnc_default_rounding_mode_c());
			ceval_mpfpoly(up_modval[thread_index], func, get_cmpfarray_i(x_init, i));

			mpc_div(modval[thread_index], up_modval[thread_index], low_modval[thread_index], get_bnc_default_rounding_mode_c());
			mpc_sub(tmp[thread_index], get_cmpfarray_i(x_init, i), modval[thread_index], get_bnc_default_rounding_mode_c());
			set_cmpfarray_i(ans, i, tmp[thread_index]);

//			get_real_mpfcmplx(mpftmp, get_cmpfarray_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			mpc_abs(absmodval[thread_index], modval[thread_index], get_bnc_default_rounding_mode());
			mpc_abs(abs_x[thread_index], get_cmpfarray_i(x_init, i), get_bnc_default_rounding_mode());
			mpc_abs(abs_newx[thread_index], get_cmpfarray_i(ans, i), get_bnc_default_rounding_mode());

			mpf_add(mpftmp[thread_index], abs_x[thread_index], abs_newx[thread_index]);
			mpf_mul(mpftmp[thread_index], mpftmp[thread_index], rel_eps);
			mpf_add(mpftmp[thread_index], mpftmp[thread_index], abs_eps);
			if( mpf_cmp(absmodval[thread_index], mpftmp[thread_index]) > 0 )
				flag += 1;

		}

		/* check convergence */
		if(flag == 0)
			break;

		subst_cmpfarray(x_init, ans);

	}
	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();

		mpf_clear(absmodval[thread_index]);
		mpf_clear(abs_x[thread_index]);
		mpf_clear(abs_newx[thread_index]);
		mpf_clear(mpftmp[thread_index]);

		mpc_clear(modval[thread_index]);
		mpc_clear(low_modval[thread_index]);
		mpc_clear(up_modval[thread_index]);
		mpc_clear(tmp[thread_index]);
	}

	printf("ans->prec: %ld\n", ans->prec);
	return times;
}

/* Petkovic Method : 3rd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int _bncomp_mpf_petckovic(CMPFArray ans, CMPFArray x_init, MPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps)
{
	int thread_index, thread_num;
	long int times, i, j, deg, flag;
	mpf_t absmodval[BNCOMP_MAX_NUM_THREADS], abs_x[BNCOMP_MAX_NUM_THREADS], abs_newx[BNCOMP_MAX_NUM_THREADS], mpftmp[BNCOMP_MAX_NUM_THREADS];
	mpc_t modval[BNCOMP_MAX_NUM_THREADS], up_modval[BNCOMP_MAX_NUM_THREADS], low_modval[BNCOMP_MAX_NUM_THREADS], tmp[BNCOMP_MAX_NUM_THREADS];

//	printf("\n");
//	print_cfarray(x_init);
//	printf("%f, %f\n", abs_eps, rel_eps);

	deg = ans->size;

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();

		mpf_init2(absmodval[thread_index], ans->prec);
		mpf_init2(abs_x[thread_index], ans->prec);
		mpf_init2(abs_newx[thread_index], ans->prec);
		mpf_init2(mpftmp[thread_index], ans->prec);

		mpc_init2(modval[thread_index], ans->prec);
		mpc_init2(low_modval[thread_index], ans->prec);
		mpc_init2(up_modval[thread_index], ans->prec);
		mpc_init2(tmp[thread_index], ans->prec);
	}

	for(times = 0; times <= maxtimes; times++)
	{
		//printf("%5d: ", times); fflush(stdout);

		flag = 0;
		
		#pragma omp parallel for private(i, j, thread_index) shared(x_init, func, ans, rel_eps, abs_eps)
		for(i = 0; i < deg; i++)
		{
			thread_index = omp_get_thread_num();

			// get Weierstrass's correction to low_modval
			//set_real_mpfcmplx_ui(low_modval[thread_index], 1UL);
			//set_image_mpfcmplx_ui(low_modval[thread_index], 0UL);
			mpc_set_ui_ui(low_modval[thread_index], 1UL, 0UL, get_bnc_default_rounding_mode_c());
			for(j = 0; j < i; j++)
			{
				mpc_set_ui(tmp[thread_index], 0UL, get_bnc_default_rounding_mode_c());
				mpc_sub(
					tmp[thread_index],
					get_cmpfarray_i(x_init, i),
					get_cmpfarray_i(x_init, j), get_bnc_default_rounding_mode_c()
				);
				mpc_mul(low_modval[thread_index], low_modval[thread_index], tmp[thread_index], get_bnc_default_rounding_mode_c());
			}
			for(j = i + 1; j < deg; j++)
			{
				mpc_set_ui(tmp[thread_index], 0UL, get_bnc_default_rounding_mode_c());
				mpc_sub(
					tmp[thread_index],
					get_cmpfarray_i(x_init, i),
					get_cmpfarray_i(x_init, j), get_bnc_default_rounding_mode_c()
				);
				mpc_mul(low_modval[thread_index], low_modval[thread_index], tmp[thread_index], get_bnc_default_rounding_mode_c());
			}
			/*
			mul_mpfcmplx_mpf(low_modval, low_modval, get_mpfpoly_i(func, func->deg));
			mul_mpfcmplx_ui(low_modval, low_modval, 2UL); // 1/2 * modval
			ceval_mpfpoly(up_modval, func, get_cmpfarray_i(x_init, i));
			// modval := Weierstrass's correction * 1/2
			div_mpfcmplx(modval, up_modval, low_modval);
			
			// tmp := z - modval
			sub_mpfcmplx(tmp, get_cmpfarray_i(x_init, i), modval);
			ceval_diff_mpfpoly(low_modval, func, tmp);
			
			// new_z := old_z - f(z) / f'(z - 1/2 * W)
			div_mpfcmplx(modval, up_modval, low_modval);
			mul_mpfcmplx_mpf(modval, modval, get_mpfpoly_i(func, func->deg));
			sub_mpfcmplx(tmp, get_cmpfarray_i(x_init, i), modval);
			*/

			mpc_mul_fr(low_modval[thread_index], low_modval[thread_index], get_mpfpoly_i(func, func->deg), get_bnc_default_rounding_mode_c());
			mpc_mul_ui(low_modval[thread_index], low_modval[thread_index], 2UL, get_bnc_default_rounding_mode_c()); // 1/2 * modval
			ceval_mpfpoly(up_modval[thread_index], func, get_cmpfarray_i(x_init, i));
			// modval := Weierstrass's correction * 1/2
			mpc_div(modval[thread_index], up_modval[thread_index], low_modval[thread_index], get_bnc_default_rounding_mode_c());

			// tmp := z - modval
			mpc_sub(tmp[thread_index], get_cmpfarray_i(x_init, i), modval[thread_index], get_bnc_default_rounding_mode_c());
			ceval_diff_mpfpoly(low_modval[thread_index], func, tmp[thread_index]);

			// new_z := old_z - f(z) / f'(z - 1/2 * W)
			mpc_div(modval[thread_index], up_modval[thread_index], low_modval[thread_index], get_bnc_default_rounding_mode_c());
			mpc_mul_fr(modval[thread_index], modval[thread_index], get_mpfpoly_i(func, func->deg), get_bnc_default_rounding_mode_c());
			mpc_sub(tmp[thread_index], get_cmpfarray_i(x_init, i), modval[thread_index], get_bnc_default_rounding_mode_c());

			set_cmpfarray_i(ans, i, tmp[thread_index]);

//			get_real_mpfcmplx(mpftmp, get_cmpfarray_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			mpc_abs(absmodval[thread_index], modval[thread_index], get_bnc_default_rounding_mode());
			mpc_abs(abs_x[thread_index], get_cmpfarray_i(x_init, i), get_bnc_default_rounding_mode());
			//abs_mpfcmplx(abs_newx[thread_index], get_cmpfarray_i(ans, i));

			//mpf_add(mpftmp[thread_index], abs_x[thread_index], abs_newx[thread_index]);
			//mpf_mul(mpftmp[thread_index], mpftmp[thread_index], rel_eps);
			//mpf_add(mpftmp[thread_index], mpftmp[thread_index], abs_eps);
			mpf_mul(mpftmp[thread_index], abs_x[thread_index], rel_eps);
			mpf_add(mpftmp[thread_index], mpftmp[thread_index], abs_eps);
			// |new_x - old_x| > |old_x| * rel_eps + abs_eps
			if( mpf_cmp(absmodval[thread_index], mpftmp[thread_index]) > 0 )
				flag += 1;

		}

		/* check convergence */
		if(flag == 0)
			break;

		subst_cmpfarray(x_init, ans);

	}
	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();

		mpf_clear(absmodval[thread_index]);
		mpf_clear(abs_x[thread_index]);
		mpf_clear(abs_newx[thread_index]);
		mpf_clear(mpftmp[thread_index]);

		mpc_clear(modval[thread_index]);
		mpc_clear(low_modval[thread_index]);
		mpc_clear(up_modval[thread_index]);
		mpc_clear(tmp[thread_index]);
	}

	printf("ans->prec: %ld\n", ans->prec);
	return times;

}

/* Aberth Method : 3rd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int _bncomp_mpf_aberth(CMPFArray ans, CMPFArray x_init, MPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps)
{
	int thread_index, thread_num;
	long int times, i, j, deg, flag;
	mpf_t absmodval[BNCOMP_MAX_NUM_THREADS], abs_x[BNCOMP_MAX_NUM_THREADS], abs_newx[BNCOMP_MAX_NUM_THREADS], mpftmp[BNCOMP_MAX_NUM_THREADS];
	mpc_t modval[BNCOMP_MAX_NUM_THREADS], up_modval[BNCOMP_MAX_NUM_THREADS], low_modval[BNCOMP_MAX_NUM_THREADS], tmp[BNCOMP_MAX_NUM_THREADS], one;

//	printf("\n");
//	print_cfarray(x_init);
//	printf("%f, %f\n", abs_eps, rel_eps);

	deg = ans->size;

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();

		mpf_init2(absmodval[thread_index], ans->prec);
		mpf_init2(abs_x[thread_index], ans->prec);
		mpf_init2(abs_newx[thread_index], ans->prec);
		mpf_init2(mpftmp[thread_index], ans->prec);

		mpc_init2(modval[thread_index], ans->prec);
		mpc_init2(low_modval[thread_index], ans->prec);
		mpc_init2(up_modval[thread_index], ans->prec);
		mpc_init2(tmp[thread_index], ans->prec);
	}

	// complex one := 1
	mpc_init2(one, ans->prec);
	//set_real_mpfcmplx_ui(one, 1UL);
	//set_image_mpfcmplx_ui(one, 0UL);
	mpc_set_ui_ui(one, 1UL, 0UL, get_bnc_default_rounding_mode_c());

	for(times = 0; times <= maxtimes; times++)
	{
		//printf("%5d: ", times); fflush(stdout);

		flag = 0;
		
		#pragma omp parallel for private(i, j, thread_index) shared(x_init, func, ans, rel_eps, abs_eps, one)
		for(i = 0; i < deg; i++)
		{
			thread_index = omp_get_thread_num();

			// low_modval := sum (z_i - z_k)^(-1)
			//set0_mpfcmplx(low_modval[thread_index]);
			mpc_set_ui(low_modval[thread_index], 0UL, get_bnc_default_rounding_mode_c());
			for(j = 0; j < i; j++)
			{
				mpc_set_ui(tmp[thread_index], 0UL, get_bnc_default_rounding_mode_c());
				mpc_sub(
					tmp[thread_index],
					get_cmpfarray_i(x_init, i),
					get_cmpfarray_i(x_init, j), get_bnc_default_rounding_mode_c()
				);
				mpc_div(tmp[thread_index], one, tmp[thread_index], get_bnc_default_rounding_mode_c());
				mpc_add(low_modval[thread_index], low_modval[thread_index], tmp[thread_index], get_bnc_default_rounding_mode_c());
			}
			for(j = i + 1; j < deg; j++)
			{
				mpc_set_ui(tmp[thread_index], 0UL, get_bnc_default_rounding_mode_c());
				mpc_sub(
					tmp[thread_index],
					get_cmpfarray_i(x_init, i),
					get_cmpfarray_i(x_init, j), get_bnc_default_rounding_mode_c()
				);
				mpc_div(tmp[thread_index], one, tmp[thread_index], get_bnc_default_rounding_mode_c());
				mpc_add(low_modval[thread_index], low_modval[thread_index], tmp[thread_index], get_bnc_default_rounding_mode_c());
			}
			// up_mpdval := f(z) / f'(z)
			ceval_mpfpoly(up_modval[thread_index], func, get_cmpfarray_i(x_init, i));
			ceval_diff_mpfpoly(tmp[thread_index], func, get_cmpfarray_i(x_init, i));
			mpc_div(up_modval[thread_index], up_modval[thread_index], tmp[thread_index], get_bnc_default_rounding_mode_c());
			
			// low_modval := 1 - up_modval * sum (z_i - z_k)^(-1)
			mpc_mul(tmp[thread_index], up_modval[thread_index], low_modval[thread_index], get_bnc_default_rounding_mode_c());
			mpc_sub(low_modval[thread_index], one, tmp[thread_index], get_bnc_default_rounding_mode_c());
			
			// new_z := old_z - up_modval / low_modval
			mpc_div(modval[thread_index], up_modval[thread_index], low_modval[thread_index], get_bnc_default_rounding_mode_c());
			mpc_sub(tmp[thread_index], get_cmpfarray_i(x_init, i), modval[thread_index], get_bnc_default_rounding_mode_c());

			set_cmpfarray_i(ans, i, tmp[thread_index]);

//			get_real_mpfcmplx(mpftmp, get_cmpfarray_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			mpc_abs(absmodval[thread_index], modval[thread_index], get_bnc_default_rounding_mode());
			mpc_abs(abs_x[thread_index], get_cmpfarray_i(x_init, i), get_bnc_default_rounding_mode());
			//abs_mpfcmplx(abs_newx[thread_index], get_cmpfarray_i(ans, i));

			//mpf_add(mpftmp[thread_index], abs_x[thread_index], abs_newx[thread_index]);
			//mpf_mul(mpftmp[thread_index], mpftmp[thread_index], rel_eps);
			//mpf_add(mpftmp[thread_index], mpftmp[thread_index], abs_eps);
			mpf_mul(mpftmp[thread_index], abs_x[thread_index], rel_eps);
			mpf_add(mpftmp[thread_index], mpftmp[thread_index], abs_eps);
			// |new_x - old_x| > |old_x| * rel_eps + abs_eps
			if( mpf_cmp(absmodval[thread_index], mpftmp[thread_index]) > 0 )
				flag += 1;

		}

		/* check convergence */
		if(flag == 0)
			break;

		subst_cmpfarray(x_init, ans);

	}

	mpc_clear(one);
	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();

		mpf_clear(absmodval[thread_index]);
		mpf_clear(abs_x[thread_index]);
		mpf_clear(abs_newx[thread_index]);
		mpf_clear(mpftmp[thread_index]);

		mpc_clear(modval[thread_index]);
		mpc_clear(low_modval[thread_index]);
		mpc_clear(up_modval[thread_index]);
		mpc_clear(tmp[thread_index]);
	}

	printf("ans->prec: %ld\n", ans->prec);
	return times;

}
#endif // USE_GMP

#ifdef __cplusplus
} // extern "C" {
#endif // __cplusplus
