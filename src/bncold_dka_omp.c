/********************************************************************************/
/* bncold_dka_omp.c: Durand-Kerner-Aberth Methods                               */
/* Copyright (C) 200-2022 Tomonori Kouya                                        */
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
//#include "bncomp.h"
#include "omp.h"
#define BNCOMP_MAX_NUM_THREADS 128

#include <stdio.h>
#include <math.h>

#ifdef USE_GMP

/* mpf_t */

/* center of gravity */
/* center = -a_{n-1} / (n * a_n) */

#if 0
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

#endif // 0

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void mpf_dka_init2_omp(_bncold_CMPFArray x_init, MPFPoly func, void (* get_radius)(mpf_t, MPFPoly), void (* get_center)(mpf_t, MPFPoly))
{
	int thread_index, thread_num;
	long int i, itmp;
	mpf_t rad, cen;
	mpf_t an[BNCOMP_MAX_NUM_THREADS], tmp[BNCOMP_MAX_NUM_THREADS], re_cinit[BNCOMP_MAX_NUM_THREADS], im_cinit[BNCOMP_MAX_NUM_THREADS];
	MPFCmplx cinit[BNCOMP_MAX_NUM_THREADS];

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

		cinit[thread_index] = init_mpfcmplx();

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

		set0_mpfcmplx(cinit[thread_index]);
#ifndef USE_MPFR
		mpf_set_d(tmp[thread_index], (double)(2.0 * M_PI * i / func->deg + 3.0 / (2.0 * func->deg)));
#else // USE_MPFR
		mpfr_set_d(tmp[thread_index], (double)(2.0 * M_PI * i / func->deg + 3.0 / (2.0 * func->deg)), GMP_RNDN); //bnc_default_rounding_mode);
#endif // USE_MPFR
		iexp_mpfcmplx(cinit[thread_index], tmp[thread_index]);
		get_real_mpfcmplx(re_cinit[thread_index], cinit[thread_index]);
		get_image_mpfcmplx(im_cinit[thread_index], cinit[thread_index]);

		/* re_cinit = cen + rad * re_cinit */
		mpf_mul(re_cinit[thread_index], rad, re_cinit[thread_index]);
		mpf_add(re_cinit[thread_index], cen, re_cinit[thread_index]);

		/* im_cinit = rad * im_cinit */
		mpf_mul(im_cinit[thread_index], rad, im_cinit[thread_index]);

		set_real_mpfcmplx(cinit[thread_index], re_cinit[thread_index]);
		set_image_mpfcmplx(cinit[thread_index], im_cinit[thread_index]);

		abs_mpfcmplx(tmp[thread_index], cinit[thread_index]);
//		printf("%5d(%f) ", i, mpf2double(tmp)); print_mpfcmplx(cinit);

		_bncold_set_cmpfarray_i(x_init, i, cinit[thread_index]);
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
		free_mpfcmplx(cinit[thread_index]);
	}
}

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void mpf_dka_init_omp(_bncold_CMPFArray x_init, MPFPoly func)
{
	int thread_index, thread_num;
	long int i, itmp;
	mpf_t rad, cen;
	mpf_t an[BNCOMP_MAX_NUM_THREADS], tmp[BNCOMP_MAX_NUM_THREADS], re_cinit[BNCOMP_MAX_NUM_THREADS], im_cinit[BNCOMP_MAX_NUM_THREADS];
	MPFCmplx cinit[BNCOMP_MAX_NUM_THREADS];

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

		cinit[thread_index] = init_mpfcmplx();

	}

	mpf_dka_radius(rad, func);
	mpf_dka_center(cen, func);

//	mpf_out_str(stdout, 10, 0, rad); printf(", "); mpf_out_str(stdout, 10, 0, cen); printf("\n");

	//cinit = init_mpfcmplx();

	#pragma omp parallel for private(i, thread_index) shared(rad, cen)
	for(i = 0; i < func->deg; i++)
	{
		thread_index = omp_get_thread_num();

		set0_mpfcmplx(cinit[thread_index]);
#ifndef USE_MPFR
		mpf_set_d(tmp[thread_index], (double)(2.0 * M_PI * i / func->deg + 3.0 / (2.0 * func->deg)));
#else // USE_MPFR
		mpfr_set_d(tmp[thread_index], (double)(2.0 * M_PI * i / func->deg + 3.0 / (2.0 * func->deg)), GMP_RNDN); //bnc_default_rounding_mode);
#endif // USE_MPFR
		iexp_mpfcmplx(cinit[thread_index], tmp[thread_index]);
		get_real_mpfcmplx(re_cinit[thread_index], cinit[thread_index]);
		get_image_mpfcmplx(im_cinit[thread_index], cinit[thread_index]);

		/* re_cinit = cen + rad * re_cinit */
		mpf_mul(re_cinit[thread_index], rad, re_cinit[thread_index]);
		mpf_add(re_cinit[thread_index], cen, re_cinit[thread_index]);

		/* im_cinit = rad * im_cinit */
		mpf_mul(im_cinit[thread_index], rad, im_cinit[thread_index]);

		set_real_mpfcmplx(cinit[thread_index], re_cinit[thread_index]);
		set_image_mpfcmplx(cinit[thread_index], im_cinit[thread_index]);

		abs_mpfcmplx(tmp[thread_index], cinit[thread_index]);
//		printf("%5d(%f) ", i, mpf2double(tmp)); print_mpfcmplx(cinit);

		_bncold_set_cmpfarray_i(x_init, i, cinit[thread_index]);
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
		free_mpfcmplx(cinit[thread_index]);
	}
}
/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int mpf_dka_omp(_bncold_CMPFArray ans, _bncold_CMPFArray x_init, MPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps)
{
	int thread_index, thread_num;
	long int times, i, j, deg, flag;
	mpf_t absmodval[BNCOMP_MAX_NUM_THREADS], abs_x[BNCOMP_MAX_NUM_THREADS], abs_newx[BNCOMP_MAX_NUM_THREADS], mpftmp[BNCOMP_MAX_NUM_THREADS];
	MPFCmplx modval[BNCOMP_MAX_NUM_THREADS], up_modval[BNCOMP_MAX_NUM_THREADS], low_modval[BNCOMP_MAX_NUM_THREADS], tmp[BNCOMP_MAX_NUM_THREADS];

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

		modval[thread_index] = init2_mpfcmplx(ans->prec);
		low_modval[thread_index] = init2_mpfcmplx(ans->prec);
		up_modval[thread_index] = init2_mpfcmplx(ans->prec);
		tmp[thread_index] = init2_mpfcmplx(ans->prec);
	}

	for(times = 0; times <= maxtimes; times++)
	{
		//printf("%5d: ", times); fflush(stdout);

		flag = 0;
		
		#pragma omp parallel for private(i, j, thread_index) shared(modval, up_modval, low_modval, tmp, mpftmp, x_init, func, ans, rel_eps, abs_eps)
		for(i = 0; i < deg; i++)
		{
			thread_index = omp_get_thread_num();

			set_real_mpfcmplx_ui(low_modval[thread_index], 1UL);
			set_image_mpfcmplx_ui(low_modval[thread_index], 0UL);
			for(j = 0; j < i; j++)
			{
				set0_mpfcmplx(tmp[thread_index]);
				sub_mpfcmplx(
					tmp[thread_index],
					_bncold_get_cmpfarray_i(x_init, i),
					_bncold_get_cmpfarray_i(x_init, j)
				);
				mul2_mpfcmplx(low_modval[thread_index], tmp[thread_index]);
			}
			for(j = i + 1; j < deg; j++)
			{
				set0_mpfcmplx(tmp[thread_index]);
				sub_mpfcmplx(
					tmp[thread_index],
					_bncold_get_cmpfarray_i(x_init, i),
					_bncold_get_cmpfarray_i(x_init, j)
				);
				mul2_mpfcmplx(low_modval[thread_index], tmp[thread_index]);
			}
			mul_mpfcmplx_mpf(low_modval[thread_index], low_modval[thread_index], get_mpfpoly_i(func, func->deg));
			_bncold_ceval_mpfpoly(up_modval[thread_index], func, _bncold_get_cmpfarray_i(x_init, i));

			div_mpfcmplx(modval[thread_index], up_modval[thread_index], low_modval[thread_index]);
			sub_mpfcmplx(tmp[thread_index], _bncold_get_cmpfarray_i(x_init, i), modval[thread_index]);
			_bncold_set_cmpfarray_i(ans, i, tmp[thread_index]);

//			get_real_mpfcmplx(mpftmp, get_cmpfarray_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			abs_mpfcmplx(absmodval[thread_index], modval[thread_index]);
			abs_mpfcmplx(abs_x[thread_index], _bncold_get_cmpfarray_i(x_init, i));
			abs_mpfcmplx(abs_newx[thread_index], _bncold_get_cmpfarray_i(ans, i));

			mpf_add(mpftmp[thread_index], abs_x[thread_index], abs_newx[thread_index]);
			mpf_mul(mpftmp[thread_index], mpftmp[thread_index], rel_eps);
			mpf_add(mpftmp[thread_index], mpftmp[thread_index], abs_eps);
			if( mpf_cmp(absmodval[thread_index], mpftmp[thread_index]) > 0 )
				flag += 1;

		}

		/* check convergence */
		if(flag == 0)
			break;

		_bncold_subst_cmpfarray(x_init, ans);

	}
	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();

		mpf_clear(absmodval[thread_index]);
		mpf_clear(abs_x[thread_index]);
		mpf_clear(abs_newx[thread_index]);
		mpf_clear(mpftmp[thread_index]);

		free_mpfcmplx(modval[thread_index]);
		free_mpfcmplx(low_modval[thread_index]);
		free_mpfcmplx(up_modval[thread_index]);
		free_mpfcmplx(tmp[thread_index]);
	}

	printf("ans->prec: %ld\n", ans->prec);
	return times;
}

/* Petkovic Method : 3rd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int mpf_petckovic_omp(_bncold_CMPFArray ans, _bncold_CMPFArray x_init, MPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps)
{
	int thread_index, thread_num;
	long int times, i, j, deg, flag;
	mpf_t absmodval[BNCOMP_MAX_NUM_THREADS], abs_x[BNCOMP_MAX_NUM_THREADS], abs_newx[BNCOMP_MAX_NUM_THREADS], mpftmp[BNCOMP_MAX_NUM_THREADS];
	MPFCmplx modval[BNCOMP_MAX_NUM_THREADS], up_modval[BNCOMP_MAX_NUM_THREADS], low_modval[BNCOMP_MAX_NUM_THREADS], tmp[BNCOMP_MAX_NUM_THREADS];

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

		modval[thread_index] = init2_mpfcmplx(ans->prec);
		low_modval[thread_index] = init2_mpfcmplx(ans->prec);
		up_modval[thread_index] = init2_mpfcmplx(ans->prec);
		tmp[thread_index] = init2_mpfcmplx(ans->prec);
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
			set_real_mpfcmplx_ui(low_modval[thread_index], 1UL);
			set_image_mpfcmplx_ui(low_modval[thread_index], 0UL);
			for(j = 0; j < i; j++)
			{
				set0_mpfcmplx(tmp[thread_index]);
				sub_mpfcmplx(
					tmp[thread_index],
					_bncold_get_cmpfarray_i(x_init, i),
					_bncold_get_cmpfarray_i(x_init, j)
				);
				mul2_mpfcmplx(low_modval[thread_index], tmp[thread_index]);
			}
			for(j = i + 1; j < deg; j++)
			{
				set0_mpfcmplx(tmp[thread_index]);
				sub_mpfcmplx(
					tmp[thread_index],
					_bncold_get_cmpfarray_i(x_init, i),
					_bncold_get_cmpfarray_i(x_init, j)
				);
				mul2_mpfcmplx(low_modval[thread_index], tmp[thread_index]);
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

			mul_mpfcmplx_mpf(low_modval[thread_index], low_modval[thread_index], get_mpfpoly_i(func, func->deg));
			mul_mpfcmplx_ui(low_modval[thread_index], low_modval[thread_index], 2UL); // 1/2 * modval
			_bncold_ceval_mpfpoly(up_modval[thread_index], func, _bncold_get_cmpfarray_i(x_init, i));
			// modval := Weierstrass's correction * 1/2
			div_mpfcmplx(modval[thread_index], up_modval[thread_index], low_modval[thread_index]);

			// tmp := z - modval
			sub_mpfcmplx(tmp[thread_index], _bncold_get_cmpfarray_i(x_init, i), modval[thread_index]);
			_bncold_ceval_diff_mpfpoly(low_modval[thread_index], func, tmp[thread_index]);

			// new_z := old_z - f(z) / f'(z - 1/2 * W)
			div_mpfcmplx(modval[thread_index], up_modval[thread_index], low_modval[thread_index]);
			mul_mpfcmplx_mpf(modval[thread_index], modval[thread_index], get_mpfpoly_i(func, func->deg));
			sub_mpfcmplx(tmp[thread_index], _bncold_get_cmpfarray_i(x_init, i), modval[thread_index]);

			_bncold_set_cmpfarray_i(ans, i, tmp[thread_index]);

//			get_real_mpfcmplx(mpftmp, get_cmpfarray_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			abs_mpfcmplx(absmodval[thread_index], modval[thread_index]);
			abs_mpfcmplx(abs_x[thread_index], _bncold_get_cmpfarray_i(x_init, i));
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

		_bncold_subst_cmpfarray(x_init, ans);

	}
	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();

		mpf_clear(absmodval[thread_index]);
		mpf_clear(abs_x[thread_index]);
		mpf_clear(abs_newx[thread_index]);
		mpf_clear(mpftmp[thread_index]);

		free_mpfcmplx(modval[thread_index]);
		free_mpfcmplx(low_modval[thread_index]);
		free_mpfcmplx(up_modval[thread_index]);
		free_mpfcmplx(tmp[thread_index]);
	}

	printf("ans->prec: %ld\n", ans->prec);
	return times;

}

/* Aberth Method : 3rd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
long int mpf_aberth_omp(_bncold_CMPFArray ans, _bncold_CMPFArray x_init, MPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps)
{
	int thread_index, thread_num;
	long int times, i, j, deg, flag;
	mpf_t absmodval[BNCOMP_MAX_NUM_THREADS], abs_x[BNCOMP_MAX_NUM_THREADS], abs_newx[BNCOMP_MAX_NUM_THREADS], mpftmp[BNCOMP_MAX_NUM_THREADS];
	MPFCmplx modval[BNCOMP_MAX_NUM_THREADS], up_modval[BNCOMP_MAX_NUM_THREADS], low_modval[BNCOMP_MAX_NUM_THREADS], tmp[BNCOMP_MAX_NUM_THREADS], one;

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

		modval[thread_index] = init2_mpfcmplx(ans->prec);
		low_modval[thread_index] = init2_mpfcmplx(ans->prec);
		up_modval[thread_index] = init2_mpfcmplx(ans->prec);
		tmp[thread_index] = init2_mpfcmplx(ans->prec);
	}

	// complex one := 1
	one = init2_mpfcmplx(ans->prec);
	set_real_mpfcmplx_ui(one, 1UL);
	set_image_mpfcmplx_ui(one, 0UL);

	for(times = 0; times <= maxtimes; times++)
	{
		//printf("%5d: ", times); fflush(stdout);

		flag = 0;
		
		#pragma omp parallel for private(i, j, thread_index) shared(x_init, func, ans, rel_eps, abs_eps, one)
		for(i = 0; i < deg; i++)
		{
			thread_index = omp_get_thread_num();

			// low_modval := sum (z_i - z_k)^(-1)
			set0_mpfcmplx(low_modval[thread_index]);
			for(j = 0; j < i; j++)
			{
				set0_mpfcmplx(tmp[thread_index]);
				sub_mpfcmplx(
					tmp[thread_index],
					_bncold_get_cmpfarray_i(x_init, i),
					_bncold_get_cmpfarray_i(x_init, j)
				);
				div_mpfcmplx(tmp[thread_index], one, tmp[thread_index]);
				add2_mpfcmplx(low_modval[thread_index], tmp[thread_index]);
			}
			for(j = i + 1; j < deg; j++)
			{
				set0_mpfcmplx(tmp[thread_index]);
				sub_mpfcmplx(
					tmp[thread_index],
					_bncold_get_cmpfarray_i(x_init, i),
					_bncold_get_cmpfarray_i(x_init, j)
				);
				div_mpfcmplx(tmp[thread_index], one, tmp[thread_index]);
				add2_mpfcmplx(low_modval[thread_index], tmp[thread_index]);
			}
			// up_mpdval := f(z) / f'(z)
			_bncold_ceval_mpfpoly(up_modval[thread_index], func, _bncold_get_cmpfarray_i(x_init, i));
			_bncold_ceval_diff_mpfpoly(tmp[thread_index], func, _bncold_get_cmpfarray_i(x_init, i));
			div_mpfcmplx(up_modval[thread_index], up_modval[thread_index], tmp[thread_index]);
			
			// low_modval := 1 - up_modval * sum (z_i - z_k)^(-1)
			mul_mpfcmplx(tmp[thread_index], up_modval[thread_index], low_modval[thread_index]);
			sub_mpfcmplx(low_modval[thread_index], one, tmp[thread_index]);
			
			// new_z := old_z - up_modval / low_modval
			div_mpfcmplx(modval[thread_index], up_modval[thread_index], low_modval[thread_index]);
			sub_mpfcmplx(tmp[thread_index], _bncold_get_cmpfarray_i(x_init, i), modval[thread_index]);

			_bncold_set_cmpfarray_i(ans, i, tmp[thread_index]);

//			get_real_mpfcmplx(mpftmp, get_cmpfarray_i(ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			abs_mpfcmplx(absmodval[thread_index], modval[thread_index]);
			abs_mpfcmplx(abs_x[thread_index], _bncold_get_cmpfarray_i(x_init, i));
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

		_bncold_subst_cmpfarray(x_init, ans);

	}

	free_mpfcmplx(one);
	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();

		mpf_clear(absmodval[thread_index]);
		mpf_clear(abs_x[thread_index]);
		mpf_clear(abs_newx[thread_index]);
		mpf_clear(mpftmp[thread_index]);

		free_mpfcmplx(modval[thread_index]);
		free_mpfcmplx(low_modval[thread_index]);
		free_mpfcmplx(up_modval[thread_index]);
		free_mpfcmplx(tmp[thread_index]);
	}

	printf("ans->prec: %ld\n", ans->prec);
	return times;

}
#endif // USE_GMP
