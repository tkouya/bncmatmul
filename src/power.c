/********************************************************************************/
/* power.c:                                                                     */
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
/*************************************************/
/* Power and Inverse Power Methods for Eig       */
/*************************************************/
#include "bnc.h"
#include <stdio.h>
#include <math.h>

/*************************************************/
/* Power                                         */
/*************************************************/
#ifdef 0
long fpower_eig(float *eig, FVector eig_vec, FMatrix mat, FVector init_vec, float aeps, float reps, long max_times)
{
	long times;
	FVector tmp_vec_old;
	float eig_old, dtmp;

	if(eig == NULL)
	{
		fprintf(stderr, "Null Pointer *eig (fpower_eig)\n");
		return -1;
	}

	tmp_vec_old = init_fvector(init_vec->dim);

	subst_fvector(tmp_vec_old, init_vec);
	eig_old = 0.0;
	*eig = 1.0;

	/* Main loop */
	for(times = 0; times < max_times; times++)
	{
		mul_fmatrix_fvec(eig_vec, mat, tmp_vec_old);

		*eig = ip_fvector(eig_vec, tmp_vec_old);
		if(fabs(*eig - eig_old) <= aeps + reps * fabs(eig_old) )
			break;

		/* v_new := Av_old / ||Av_old|| */
		dtmp = norm2_fvector(eig_vec);
		if(dtmp == 0.0)
			break;

		cmul_fvector(tmp_vec_old, 1.0/dtmp, eig_vec);
		eig_old = *eig;
	}

	free_fvector(tmp_vec_old);
	return times;
}
#endif // 0
/*************************************************/
/* Power                                         */
/*************************************************/
long dpower_eig(double *eig, DVector eig_vec, DMatrix mat, DVector init_vec, double aeps, double reps, long max_times)
{
	long times;
	DVector tmp_vec_old;
	double eig_old, dtmp;

	if(eig == NULL)
	{
		fprintf(stderr, "Null Pointer *eig (dpower)\n");
		return -1;
	}

	tmp_vec_old = init_dvector(init_vec->dim);

	subst_dvector(tmp_vec_old, init_vec);
	eig_old = 0.0;
	*eig = 1.0;

	/* Main loop */
	for(times = 0; times < max_times; times++)
	{
		mul_dmatrix_dvec(eig_vec, mat, tmp_vec_old);

		*eig = ip_dvector(eig_vec, tmp_vec_old);
		if(fabs(*eig - eig_old) <= aeps + reps * fabs(eig_old) )
			break;

		/* v_new := Av_old / ||Av_old|| */
		dtmp = norm2_dvector(eig_vec);
		if(dtmp == 0.0)
			break;

		cmul_dvector(tmp_vec_old, 1.0/dtmp, eig_vec);
		eig_old = *eig;
	}

	free_dvector(tmp_vec_old);
	return times;
}

#ifdef USE_GMP
/*************************************************/
/* Power                                         */
/*************************************************/
long mpf_power_eig(mpf_t eig, MPFVector eig_vec, MPFMatrix mat, MPFVector init_vec, mpf_t aeps, mpf_t reps, long max_times)
{
	long times;
	MPFVector tmp_vec_old;
	mpf_t eig_old, mpf_tmp0, mpf_tmp1;

	if(eig == NULL)
	{
		fprintf(stderr, "Null Pointer *eig (mpf_power_eig)\n");
		return -1;
	}

	tmp_vec_old = init2_mpfvector(init_vec->dim, prec_mpfvector(eig_vec));

	subst_mpfvector(tmp_vec_old, init_vec);
	mpf_init2(eig_old, mpf_get_prec(eig));
	mpf_set_ui(eig_old, 0UL);
	mpf_init2(mpf_tmp0, mpf_get_prec(eig));
	mpf_init2(mpf_tmp1, mpf_get_prec(eig));
	mpf_set_ui(eig, 1UL);

	/* Main loop */
	for(times = 0; times < max_times; times++)
	{
		mul_mpfmatrix_mpfvec(eig_vec, mat, tmp_vec_old);

		ip_mpfvector(eig, eig_vec, tmp_vec_old);

		/* abs(eig - eig_old) <= aeps + reps * abs(eig_old) */
		mpf_sub(mpf_tmp0, eig, eig_old);
		mpf_abs(mpf_tmp0, mpf_tmp0);
		mpf_abs(mpf_tmp1, eig_old);
		mpf_mul(mpf_tmp1, mpf_tmp1, reps);
		mpf_add(mpf_tmp1, aeps, mpf_tmp1);
		if(mpf_cmp(mpf_tmp0, mpf_tmp1) <= 0)
			break;

		/* v_new := Av_old / ||Av_old|| */
		norm2_mpfvector(mpf_tmp0, eig_vec);
		if(mpf_cmp_ui(mpf_tmp0, 0UL) <= 0)
			break;

		mpf_ui_div(mpf_tmp1, 1UL, mpf_tmp0);
		cmul_mpfvector(tmp_vec_old, mpf_tmp1, eig_vec);
		mpf_set(eig_old, eig);

	}

	free_mpfvector(tmp_vec_old);
	mpf_clear(mpf_tmp0); mpf_clear(mpf_tmp1); mpf_clear(eig_old);
	return times;
}
#endif // USE_GMP
