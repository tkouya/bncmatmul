/********************************************************************************/
/* newton.c: Newton and Simplified Newton Methods                               */
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
#include "bnc.h"
#include <stdio.h>
#include <math.h>

/* Newton Method : 1 dimension */
/* f(x) = 0 -> x_new := x_n - func(x) / dfunc(x) */
long int fnewton_1(float *ans, float x_init, float (* func)(float x), float (* dfunc)(float x), long int maxtimes, float abs_eps, float rel_eps)
{
	float x_old, tmp, tmp1, tmp2;
	long int times;

	/* init */
	x_old = x_init;

	/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* func(x) / dfunc(x) */
		tmp = func(x_old) / dfunc(x_old);

/* check */
		/* |func(x)/dfunc(x)| <= abs_eps + |x_old| * rel_eps */
		tmp1 = abs_eps + (float)fabs(x_old) * rel_eps;
		tmp2 = (float)fabs(tmp);
		if(tmp2 <= tmp1)
			break;

		/* set next step */
		x_old = x_old - tmp;
	}

	/* return */
	if(times >= maxtimes)
		fprintf(stderr, "fnewton_1: Not convergent.\n");

	*ans = x_old;

	return times;
}

/* Simplified Newton Method : 1 dimension */
/* f(x) = 0 -> x_new := x_n - func(x) / dfunc(x_init) */
long int fsnewton_1(float *ans, float x_init, float (* func)(float x), float (* dfunc)(float x), long int maxtimes, float abs_eps, float rel_eps)
{
	float x_old, tmp, tmp1, tmp2, df0;
	long int times;

	/* init */
	x_old = x_init;

	df0 = dfunc(x_init);

	/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* func(x) / dfunc(x) */
		tmp1 = func(x_old);
		tmp = tmp1 / df0;

		/* check */
		/* |func(x)/dfunc(x)| <= abs_eps + |x_old| * rel_eps */
		tmp1 = abs_eps + (float)fabs(x_old) * rel_eps;
		tmp2 = (float)fabs(tmp);
		if(tmp2 <= tmp1)
			break;

		/* set next step */
		x_old = x_old - tmp;
	}

	/* return */
	if(times >= maxtimes)
		fprintf(stderr, "fsnewton_1: Not convergent.\n");
	
	*ans = x_old;

	return times;
}

/* Newton Method */
/* f(x) = 0 -> x_new := x_n - jfunc^(-1)(x) func(x) */
long int fnewton(FVector ans, FVector x_init, void (* func)(FVector vret, FVector x), void (* jfunc)(FMatrix mret, FVector x), long int maxtimes, float abs_eps, float rel_eps)
{
	FVector x_old, x_tmp, x_tmp1;
	FMatrix jacobi;
	float tmp, tmp1;
	long int times;

	/* init */
	x_old = init_fvector(ans->dim);
	x_tmp = init_fvector(ans->dim);
	x_tmp1= init_fvector(ans->dim);
	jacobi= init_fmatrix(ans->dim, ans->dim);

	subst_fvector(x_old, x_init);

	/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* jacobi(x)^(-1) * func(x) */
		func(ans, x_old);
		jfunc(jacobi, x_old);
		FLUdecomp(jacobi);
		SolveFLS(x_tmp, jacobi, ans);

		/* check */
		/* |func(x)/dfunc(x)| <= abs_eps + |x_old| * rel_eps */
		tmp = abs_eps + (float)fabs(normi_fvector(x_old) * rel_eps);
		tmp1 = normi_fvector(x_tmp);
		if(tmp1 <= tmp)
			break;

		/* set next step */
		sub_fvector(x_old, x_old, x_tmp);
	}

	/* return */
	if(times >= maxtimes)
		fprintf(stderr, "fnewton: Not convergent.\n");
	subst_fvector(ans, x_old);

	/* clear */
	free_fvector(x_old);
	free_fvector(x_tmp);
	free_fvector(x_tmp1);
	free_fmatrix(jacobi);

	return times;
}

/* Simplified Newton Method */
/* f(x) = 0 -> x_new := x_n - jfunc^(-1)(x) func(x) */
long int fsnewton(FVector ans, FVector x_init, void (* func)(FVector vret, FVector x), void (* jfunc)(FMatrix mret, FVector x), long int maxtimes, float abs_eps, float rel_eps)
{
	FVector x_old, x_tmp, x_tmp1;
	FMatrix jacobi;
	float tmp, tmp1;
	long int times;

	/* init */
	x_old = init_fvector(ans->dim);
	x_tmp = init_fvector(ans->dim);
	x_tmp1= init_fvector(ans->dim);
	jacobi= init_fmatrix(ans->dim, ans->dim);

	subst_fvector(x_old, x_init);
	jfunc(jacobi, x_init);
	FLUdecomp(jacobi);

	/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* jacobi(x0)^(-1) * func(x) */
		func(ans, x_old);
		SolveFLS(x_tmp, jacobi, ans);

		/* check */
		/* |func(x)/dfunc(x)| <= abs_eps + |x_old| * rel_eps */
		tmp = abs_eps + normi_fvector(x_old) * rel_eps;
		tmp1 = normi_fvector(x_tmp);
		if(tmp1 <= tmp)
			break;

		/* set next step */
		sub_fvector(x_old, x_old, x_tmp);
	}

	/* return */
	if(times >= maxtimes)
		fprintf(stderr, "fsnewton: Not convergent.\n");
	subst_fvector(ans, x_old);

	/* clear */
	free_fvector(x_old);
	free_fvector(x_tmp);
	free_fvector(x_tmp1);
	free_fmatrix(jacobi);

	return times;
}

/* Newton Method : 1 dimension */
/* f(x) = 0 -> x_new := x_n - func(x) / dfunc(x) */
long int dnewton_1(double *ans, double x_init, double (* func)(double x), double (* dfunc)(double x), long int maxtimes, double abs_eps, double rel_eps)
{
	double x_old, tmp, tmp1, tmp2;
	long int times;

	/* init */
	x_old = x_init;

	/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* func(x) / dfunc(x) */
		tmp = func(x_old) / dfunc(x_old);

/* check */
		/* |func(x)/dfunc(x)| <= abs_eps + |x_old| * rel_eps */
		tmp1 = abs_eps + fabs(x_old) * rel_eps;
		tmp2 = fabs(tmp);
		if(tmp2 <= tmp1)
			break;

		/* set next step */
		x_old = x_old - tmp;
	}

	/* return */
	if(times >= maxtimes)
		fprintf(stderr, "dnewton_1: Not convergent.\n");

	*ans = x_old;

	return times;
}

/* Simplified Newton Method : 1 dimension */
/* f(x) = 0 -> x_new := x_n - func(x) / dfunc(x_init) */
long int dsnewton_1(double *ans, double x_init, double (* func)(double x), double (* dfunc)(double x), long int maxtimes, double abs_eps, double rel_eps)
{
	double x_old, tmp, tmp1, tmp2, df0;
	long int times;

	/* init */
	x_old = x_init;

	df0 = dfunc(x_init);

	/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* func(x) / dfunc(x) */
		tmp1 = func(x_old);
		tmp = tmp1 / df0;

		/* check */
		/* |func(x)/dfunc(x)| <= abs_eps + |x_old| * rel_eps */
		tmp1 = abs_eps + fabs(x_old * rel_eps);
		tmp2 = fabs(tmp);
		if(tmp2 <= tmp1)
			break;

		/* set next step */
		x_old = x_old - tmp;
	}

	/* return */
	if(times >= maxtimes)
		fprintf(stderr, "dsnewton_1: Not convergent.\n");
	
	*ans = x_old;

	return times;
}

/* Newton Method */
/* f(x) = 0 -> x_new := x_n - jfunc^(-1)(x) func(x) */
long int dnewton(DVector ans, DVector x_init, void (* func)(DVector vret, DVector x), void (* jfunc)(DMatrix mret, DVector x), long int maxtimes, double abs_eps, double rel_eps)
{
	DVector x_old, x_tmp, x_tmp1;
	DMatrix jacobi;
	double tmp, tmp1;
	long int times;

	/* init */
	x_old = init_dvector(ans->dim);
	x_tmp = init_dvector(ans->dim);
	x_tmp1= init_dvector(ans->dim);
	jacobi= init_dmatrix(ans->dim, ans->dim);

	subst_dvector(x_old, x_init);

	/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* jacobi(x)^(-1) * func(x) */
		func(ans, x_old);
		jfunc(jacobi, x_old);
		DLUdecomp(jacobi);
		SolveDLS(x_tmp, jacobi, ans);

		/* check */
		/* |func(x)/dfunc(x)| <= abs_eps + |x_old| * rel_eps */
		tmp = abs_eps + fabs(normi_dvector(x_old) * rel_eps);
		tmp1 = normi_dvector(x_tmp);
		if(tmp1 <= tmp)
			break;

		/* set next step */
		sub_dvector(x_old, x_old, x_tmp);
	}

	/* return */
	if(times >= maxtimes)
		fprintf(stderr, "dnewton: Not convergent.\n");
	subst_dvector(ans, x_old);

	/* clear */
	free_dvector(x_old);
	free_dvector(x_tmp);
	free_dvector(x_tmp1);
	free_dmatrix(jacobi);

	return times;
}

/* Simplified Newton Method */
/* f(x) = 0 -> x_new := x_n - jfunc^(-1)(x) func(x) */
long int dsnewton(DVector ans, DVector x_init, void (* func)(DVector vret, DVector x), void (* jfunc)(DMatrix mret, DVector x), long int maxtimes, double abs_eps, double rel_eps)
{
	DVector x_old, x_tmp, x_tmp1;
	DMatrix jacobi;
	double tmp, tmp1;
	long int times;

	/* init */
	x_old = init_dvector(ans->dim);
	x_tmp = init_dvector(ans->dim);
	x_tmp1= init_dvector(ans->dim);
	jacobi= init_dmatrix(ans->dim, ans->dim);

	subst_dvector(x_old, x_init);
	jfunc(jacobi, x_init);
	DLUdecomp(jacobi);

	/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* jacobi(x0)^(-1) * func(x) */
		func(ans, x_old);
		SolveDLS(x_tmp, jacobi, ans);

		/* check */
		/* |func(x)/dfunc(x)| <= abs_eps + |x_old| * rel_eps */
		tmp = abs_eps + normi_dvector(x_old) * rel_eps;
		tmp1 = normi_dvector(x_tmp);
		if(tmp1 <= tmp)
			break;

		/* set next step */
		sub_dvector(x_old, x_old, x_tmp);
	}

	/* return */
	if(times >= maxtimes)
		fprintf(stderr, "dsnewton: Not convergent.\n");
	subst_dvector(ans, x_old);

	/* clear */
	free_dvector(x_old);
	free_dvector(x_tmp);
	free_dvector(x_tmp1);
	free_dmatrix(jacobi);

	return times;
}

#ifdef USE_GMP
/* Newton Method : 1 dimension */
/* f(x) = 0 -> x_new := x_n - func(x) / dfunc(x) */
long int mpf_newton_1(mpf_t ans, mpf_t x_init, void (* func)(mpf_t ret, mpf_t x), void (* dfunc)(mpf_t ret, mpf_t x), long int maxtimes, mpf_t abs_eps, mpf_t rel_eps)
{
	mpf_t x_old, tmp, tmp1, tmp2;
	long int times;

	/* init */
	mpf_init2(x_old, mpf_get_prec(ans));
	mpf_init2(tmp  , mpf_get_prec(ans));
	mpf_init2(tmp1 , mpf_get_prec(ans));
	mpf_init2(tmp2 , mpf_get_prec(ans));

	mpf_set(x_old, x_init);

	/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* func(x) / dfunc(x) */
		func(ans, x_old);
		dfunc(tmp, x_old);
		mpf_div(tmp, ans, tmp);

		/* check */
		/* |func(x)/dfunc(x)| <= abs_eps + |x_old| * rel_eps */
		mpf_mul(tmp1, x_old, rel_eps);
		mpf_abs(tmp1, tmp1);
		mpf_add(tmp1, tmp1, abs_eps);
		mpf_abs(tmp2, tmp);
		if(mpf_cmp(tmp2, tmp1) <= 0)
			break;

		/* set next step */
		mpf_sub(x_old, x_old, tmp);
	}

	/* return */
	if(times >= maxtimes)
		fprintf(stderr, "mpf_newton_1: Not convergent.\n");
	mpf_set(ans, x_old);

	/* clear */
	mpf_clear(tmp);
	mpf_clear(tmp1);
	mpf_clear(tmp2);
	mpf_clear(x_old);

	return times;
}

/* Simplified Newton Method : 1 dimension */
/* f(x) = 0 -> x_new := x_n - func(x) / dfunc(x_init) */
long int mpf_snewton_1(mpf_t ans, mpf_t x_init, void (* func)(mpf_t ret, mpf_t x), void (* dfunc)(mpf_t ret, mpf_t x), long int maxtimes, mpf_t abs_eps, mpf_t rel_eps)
{
	mpf_t x_old, tmp, tmp1, tmp2;
	long int times;

	/* init */
	mpf_init2(x_old, mpf_get_prec(ans));
	mpf_init2(tmp  , mpf_get_prec(ans));
	mpf_init2(tmp1 , mpf_get_prec(ans));
	mpf_init2(tmp2 , mpf_get_prec(ans));

	mpf_set(x_old, x_init);
	dfunc(ans, x_init);

	/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* func(x) / dfunc(x) */
		func(tmp1, x_old);
		mpf_div(tmp, tmp1, ans);

		/* check */
		/* |func(x)/dfunc(x)| <= abs_eps + |x_old| * rel_eps */
		mpf_mul(tmp1, x_old, rel_eps);
		mpf_abs(tmp1, tmp1);
		mpf_add(tmp1, tmp1, abs_eps);
		mpf_abs(tmp2, tmp);
		if(mpf_cmp(tmp2, tmp1) <= 0)
			break;

		/* set next step */
		mpf_sub(x_old, x_old, tmp);
	}

	/* return */
	if(times >= maxtimes)
		fprintf(stderr, "mpf_snewton_1: Not convergent.\n");
	mpf_set(ans, x_old);

	/* clear */
	mpf_clear(tmp);
	mpf_clear(tmp1);
	mpf_clear(tmp2);
	mpf_clear(x_old);

	return times;
}

/* Newton Method */
/* f(x) = 0 -> x_new := x_n - jfunc^(-1)(x) func(x) */
long int mpf_newton(MPFVector ans, MPFVector x_init, void (* func)(MPFVector vret, MPFVector x), void (* jfunc)(MPFMatrix mret, MPFVector x), long int maxtimes, mpf_t abs_eps, mpf_t rel_eps)
{
	MPFVector x_old, x_tmp, x_tmp1;
	MPFMatrix jacobi;
	mpf_t tmp, tmp1;
	long int times;

	/* init */
	x_old = init2_mpfvector(ans->dim, ans->prec);
	x_tmp = init2_mpfvector(ans->dim, ans->prec);
	x_tmp1= init2_mpfvector(ans->dim, ans->prec);
	jacobi= init2_mpfmatrix(ans->dim, ans->dim, ans->prec);
	mpf_init2(tmp  , ans->prec);
	mpf_init2(tmp1 , ans->prec);

	subst_mpfvector(x_old, x_init);

	/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* jacobi(x)^(-1) * func(x) */
		func(ans, x_old);
		jfunc(jacobi, x_old);
		MPFLUdecomp(jacobi);
		SolveMPFLS(x_tmp, jacobi, ans);

		/* check */
		/* |func(x)/dfunc(x)| <= abs_eps + |x_old| * rel_eps */
		normi_mpfvector(tmp, x_old);
		mpf_mul(tmp, tmp, rel_eps);
		mpf_add(tmp, tmp, abs_eps);
		normi_mpfvector(tmp1, x_tmp);
		if(mpf_cmp(tmp1, tmp) <= 0)
			break;

		/* set next step */
		sub_mpfvector(x_old, x_old, x_tmp);
	}

	/* return */
	if(times >= maxtimes)
		fprintf(stderr, "mpf_newton: Not convergent.\n");
	subst_mpfvector(ans, x_old);

	/* clear */
	free_mpfvector(x_old);
	free_mpfvector(x_tmp);
	free_mpfvector(x_tmp1);
	free_mpfmatrix(jacobi);
	mpf_clear(tmp);
	mpf_clear(tmp1);

	return times;
}

/* Simplified Newton Method */
/* f(x) = 0 -> x_new := x_n - jfunc^(-1)(x) func(x) */
long int mpf_snewton(MPFVector ans, MPFVector x_init, void (* func)(MPFVector vret, MPFVector x), void (* jfunc)(MPFMatrix mret, MPFVector x), long int maxtimes, mpf_t abs_eps, mpf_t rel_eps)
{
	MPFVector x_old, x_tmp, x_tmp1;
	MPFMatrix jacobi;
	mpf_t tmp, tmp1;
	long int times;

	/* init */
	x_old = init2_mpfvector(ans->dim, ans->prec);
	x_tmp = init2_mpfvector(ans->dim, ans->prec);
	x_tmp1= init2_mpfvector(ans->dim, ans->prec);
	jacobi= init2_mpfmatrix(ans->dim, ans->dim, ans->prec);
	mpf_init2(tmp  , ans->prec);
	mpf_init2(tmp1 , ans->prec);

	subst_mpfvector(x_old, x_init);
	jfunc(jacobi, x_init);
	MPFLUdecomp(jacobi);

	/* Main loop */
	for(times = 0; times < maxtimes; times++)
	{
		/* jacobi(x0)^(-1) * func(x) */
		func(ans, x_old);
		SolveMPFLS(x_tmp, jacobi, ans);

		/* check */
		/* |func(x)/dfunc(x)| <= abs_eps + |x_old| * rel_eps */
		normi_mpfvector(tmp, x_old);
		mpf_mul(tmp, tmp, rel_eps);
		mpf_add(tmp, tmp, abs_eps);
		normi_mpfvector(tmp1, x_tmp);
		if(mpf_cmp(tmp1, tmp) <= 0)
			break;

		/* set next step */
		sub_mpfvector(x_old, x_old, x_tmp);
	}

	/* return */
	if(times >= maxtimes)
		fprintf(stderr, "mpf_snewton: Not convergent.\n");
	subst_mpfvector(ans, x_old);

	/* clear */
	free_mpfvector(x_old);
	free_mpfvector(x_tmp);
	free_mpfvector(x_tmp1);
	free_mpfmatrix(jacobi);
	mpf_clear(tmp);
	mpf_clear(tmp1);

	return times;
}
#endif
