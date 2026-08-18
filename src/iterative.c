/********************************************************************************/
/* iterative.c:                                                                 */
/* Iterative Methods for Linear Systems:                                        */
/* Jacobi Iterative, Gauss-Seidel and SOR Methods                               */
/*                                                                              */
/* Copyright (c) 2002-2011 Tomonori Kouya                                       */
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
#include <stdio.h>
#include <math.h>
//#include "bnc.h"
#include "bncmatmul.h"

/* float */
#if 0
/* get residual */
/* ret := b - Ax */
void get_residual_fvector(FVector ret, FVector b, FMatrix a, FVector x)
{
	set0_fvector(ret);

	/* ret := Ax */
	mul_fmatrix_fvec(ret, a, x);

	/* ret := b - Ax */
	sub_fvector(ret, b, ret);

}

/* Jacobi Iterative Method */
void fjacobi(FVector x, FMatrix a, FVector b, float aeps, float reps, long int max_times)
{
	long int times, i, j, dim;
	float tmp, first_norm2_res;
	FVector new_x, res;

	dim = x->dim;

	/* Initialize */
	new_x = init_fvector(dim);
	res = init_fvector(dim);

	set0_fvector(new_x);
	for(i = 0; i < dim; i++)
		sfvi(x, i, gfvi(b,i) / gfmij(a, i, i));

	first_norm2_res = norm2_fvector(b);

	for(times = 0; times <= max_times; times++)
	{
		for(i = 0; i < dim; i++)
		{
			tmp = 0.0;
			for(j = 0; j < dim; j++)
				tmp += gfmij(a, i, j) * gfvi(x, j);
			sfvi(new_x, i, gfvi(x, i) + (gfvi(b, i) - tmp)/gfmij(a, i, i));
		}

		get_residual_fvector(res, b, a, new_x);
		if(norm2_fvector(res) < aeps + reps * first_norm2_res)
			return;

		printf("%5ld %15.7e\n", times, norm2_fvector(res));

		subst_fvector(x, new_x);

	}

	/* free */
	free_fvector(new_x);
	free_fvector(res);
}

/* Gauss-Seidel Method */
void fgs(FVector x, FMatrix a, FVector b, float aeps, float reps, long int max_times)
{
	long int times, i, j, dim;
	float tmp, first_norm2_res;
	FVector new_x, res;

	dim = x->dim;

	/* Initialize */
	new_x = init_fvector(dim);
	res = init_fvector(dim);

	set0_fvector(new_x);
	for(i = 0; i < dim; i++)
		sfvi(x, i, gfvi(b,i) / gfmij(a, i, i));

	first_norm2_res = norm2_fvector(b);

	for(times = 0; times <= max_times; times++)
	{
		for(i = 0; i < dim; i++)
		{
			tmp = 0.0;
			for(j = 0; j < i; j++)
				tmp += gfmij(a, i, j) * gfvi(new_x, j);
			for(j = i; j < dim; j++)
				tmp += gfmij(a, i, j) * gfvi(x, j);
			sfvi(new_x, i, gfvi(x, i) + (gfvi(b, i) - tmp) / gfmij(a, i, i));
		}

		get_residual_fvector(res, b, a, new_x);
		if(norm2_fvector(res) < aeps + reps * first_norm2_res)
			return;

		printf("%5ld %15.7e\n", times, norm2_fvector(res));

		subst_fvector(x, new_x);
	}

	/* free */
	free_fvector(new_x);
	free_fvector(res);
}

/* Successive Over-Relaxation Method */
void fsor(FVector x, FMatrix a, FVector b, float omega, float aeps, float reps, long int max_times)
{
	long int times, i, j, dim;
	float tmp, first_norm2_res;
	FVector new_x, res;

	dim = x->dim;

	/* Initialize */
	new_x = init_fvector(dim);
	res = init_fvector(dim);

	set0_fvector(new_x);
	for(i = 0; i < dim; i++)
		sfvi(x, i, gfvi(b,i) / gfmij(a, i, i));

	first_norm2_res = norm2_fvector(b);


	for(times = 0; times <= max_times; times++)
	{
		for(i = 0; i < dim; i++)
		{
			tmp = 0.0;
			for(j = 0; j < i; j++)
				tmp += gfmij(a, i, j) * gfvi(new_x, j);
			for(j = i+1; j < dim; j++)
				tmp += gfmij(a, i, j) * gfvi(x, j);
			tmp = (gfvi(b, i) - tmp)/gfmij(a, i, i) - gfvi(x, i);
			sfvi(new_x, i, gfvi(x, i) + omega * tmp);
		}

		get_residual_fvector(res, b, a, new_x);

		if(norm2_fvector(res) < aeps + reps * first_norm2_res)
			return;

		printf("%5ld %15.7e\n", times, norm2_fvector(res));

		subst_fvector(x, new_x);
	}

	/* free */
	free_fvector(new_x);
	free_fvector(res);
}
#endif // 0

/* double */

/* get residual */
void get_residual_dvector(DVector ret, DVector b, DMatrix a, DVector x)
{
	set0_dvector(ret);

	/* ret := Ax */
	mul_dmatrix_dvec(ret, a, x);

	/* ret := b - Ax */
	sub_dvector(ret, b, ret);

}

/* Jacobi Iterative Method */
//void djacobi(DVector x, DMatrix a, DVector b, double aeps, double reps, long int max_times)
void bnc_djacobi(DVector x, DMatrix a, DVector b, double aeps, double reps, long int max_times)
{
	long int times, i, j, dim;
	double tmp, first_norm2_res;
	DVector new_x, res;

	dim = x->dim;

	/* Initialize */
	new_x = init_dvector(dim);
	res = init_dvector(dim);

	set0_dvector(new_x);
	for(i = 0; i < dim; i++)
		sdvi(x, i, gdvi(b,i) / gdmij(a, i, i));

	first_norm2_res = norm2_dvector(b);

	for(times = 0; times <= max_times; times++)
	{
		for(i = 0; i < dim; i++)
		{
			tmp = 0.0;
			for(j = 0; j < dim; j++)
				tmp += gdmij(a, i, j) * gdvi(x, j);
			sdvi(new_x, i, gdvi(x, i) + (gdvi(b, i) - tmp)/gdmij(a, i, i));
		}

		get_residual_dvector(res, b, a, new_x);
		if(norm2_dvector(res) < aeps + reps * first_norm2_res)
			return;

		printf("%5ld %25.17e\n", times, norm2_dvector(res));

		subst_dvector(x, new_x);

	}

	/* free */
	free_dvector(new_x);
	free_dvector(res);
}

/* Gauss-Seidel Method */
void dgs(DVector x, DMatrix a, DVector b, double aeps, double reps, long int max_times)
{
	long int times, i, j, dim;
	double tmp, first_norm2_res;
	DVector new_x, res;

	dim = x->dim;

	/* Initialize */
	new_x = init_dvector(dim);
	res = init_dvector(dim);

	set0_dvector(new_x);
	for(i = 0; i < dim; i++)
		sdvi(x, i, gdvi(b,i) / gdmij(a, i, i));

	first_norm2_res = norm2_dvector(b);

	for(times = 0; times <= max_times; times++)
	{
		for(i = 0; i < dim; i++)
		{
			tmp = 0.0;
			for(j = 0; j < i; j++)
				tmp += gdmij(a, i, j) * gdvi(new_x, j);
			for(j = i; j < dim; j++)
				tmp += gdmij(a, i, j) * gdvi(x, j);
			sdvi(new_x, i, gdvi(x, i) + (gdvi(b, i) - tmp) / gdmij(a, i, i));
		}

		get_residual_dvector(res, b, a, new_x);
		if(norm2_dvector(res) < aeps + reps * first_norm2_res)
			return;

		printf("%5ld %25.17e\n", times, norm2_dvector(res));

		subst_dvector(x, new_x);
	}

	/* free */
	free_dvector(new_x);
	free_dvector(res);
}

/* Successive Over-Relaxation Method */
void dsor(DVector x, DMatrix a, DVector b, double omega, double aeps, double reps, long int max_times)
{
	long int times, i, j, dim;
	double tmp, first_norm2_res;
	DVector new_x, res;

	dim = x->dim;

	/* Initialize */
	new_x = init_dvector(dim);
	res = init_dvector(dim);

	set0_dvector(new_x);
	for(i = 0; i < dim; i++)
		sdvi(x, i, gdvi(b,i) / gdmij(a, i, i));

	first_norm2_res = norm2_dvector(b);

	for(times = 0; times <= max_times; times++)
	{
		for(i = 0; i < dim; i++)
		{
			tmp = 0.0;
			for(j = 0; j < i; j++)
				tmp += gdmij(a, i, j) * gdvi(new_x, j);
			for(j = i+1; j < dim; j++)
				tmp += gdmij(a, i, j) * gdvi(x, j);
			tmp = (gdvi(b, i) - tmp)/gdmij(a, i, i) - gdvi(x, i);
			sdvi(new_x, i, gdvi(x, i) + omega * tmp);
		}

		get_residual_dvector(res, b, a, new_x);

		if(norm2_dvector(res) < aeps + reps * first_norm2_res)
			return;

		printf("%5ld %25.17e\n", times, norm2_dvector(res));

		subst_dvector(x, new_x);
	}

	/* free */
	free_dvector(new_x);
	free_dvector(res);
}

#ifdef USE_GMP
/* get residual */
void get_residual_mpfvector(MPFVector ret, MPFVector b, MPFMatrix a, MPFVector x)
{
	set0_mpfvector(ret);

	/* ret := Ax */
	mul_mpfmatrix_mpfvec(ret, a, x);

	/* ret := b - Ax */
	sub_mpfvector(ret, b, ret);

}

/* Jacobi Iterative Method */
void mpf_jacobi(MPFVector x, MPFMatrix a, MPFVector b, mpf_t aeps, mpf_t reps, long int max_times)
{
	long int times, i, j, dim;
	mpf_t tmp, tmp1, first_norm2_res;
	MPFVector new_x, res;

	dim = x->dim;

	/* Initialize */
	mpf_init2(tmp, x->prec);
	mpf_init2(tmp1, x->prec);
	mpf_init2(first_norm2_res, x->prec);
	new_x = init2_mpfvector(dim, x->prec);
	res = init2_mpfvector(dim, x->prec);

	set0_mpfvector(new_x);
	for(i = 0; i < dim; i++)
	{
		mpf_div(tmp, gmpfvi(b, i), gmpfmij(a, i, i));
		smpfvi(x, i, tmp);
	}

	norm2_mpfvector(first_norm2_res, b);

	for(times = 0; times <= max_times; times++)
	{
		for(i = 0; i < dim; i++)
		{
			mpf_set_ui(tmp, 0UL);
			for(j = 0; j < dim; j++)
			{
				mpf_mul(tmp1, gmpfmij(a, i, j), gmpfvi(x, j));
				mpf_add(tmp, tmp, tmp1);
			}
			mpf_sub(tmp, gmpfvi(b, i), tmp);
			mpf_div(tmp, tmp, gmpfmij(a, i, i));
			mpf_add(tmp, tmp, gmpfvi(x, i));
			smpfvi(new_x, i, tmp);
		}

		get_residual_mpfvector(res, b, a, new_x);
		mpf_mul(tmp, reps, first_norm2_res);
		mpf_add(tmp, tmp, aeps);
		norm2_mpfvector(tmp1, res);
		if(mpf_cmp(tmp1, tmp) < 0)
			return;

		printf("%5ld %25.17e\n", times, mpf2double(tmp1));

		subst_mpfvector(x, new_x);

	}

	/* free */
	mpf_clear(tmp);
	mpf_clear(tmp1);
	mpf_clear(first_norm2_res);
	free_mpfvector(new_x);
	free_mpfvector(res);
}

/* Gauss-Seidel Method */
void mpf_gs(MPFVector x, MPFMatrix a, MPFVector b, mpf_t aeps, mpf_t reps, long int max_times)
{
	long int times, i, j, dim;
	mpf_t tmp, tmp1, first_norm2_res;
	MPFVector new_x, res;

	dim = x->dim;

	/* Initialize */
	mpf_init2(tmp, x->prec);
	mpf_init2(tmp1, x->prec);
	mpf_init2(first_norm2_res, x->prec);
	new_x = init2_mpfvector(dim, x->prec);
	res = init2_mpfvector(dim, x->prec);

	set0_mpfvector(new_x);
	for(i = 0; i < dim; i++)
	{
		mpf_div(tmp, gmpfvi(b, i), gmpfmij(a, i, i));
		smpfvi(x, i, tmp);
	}

	norm2_mpfvector(first_norm2_res, b);

	for(times = 0; times <= max_times; times++)
	{
		for(i = 0; i < dim; i++)
		{
			mpf_set_ui(tmp, 0UL);
			for(j = 0; j < i; j++)
			{
				mpf_mul(tmp1, gmpfmij(a, i, j), gmpfvi(new_x, j));
				mpf_add(tmp, tmp, tmp1);
			}
			for(j = i; j < dim; j++)
			{
				mpf_mul(tmp1, gmpfmij(a, i, j), gmpfvi(new_x, j));
				mpf_add(tmp, tmp, tmp1);
			}
			mpf_sub(tmp, gmpfvi(b, i), tmp);
			mpf_div(tmp, tmp, gmpfmij(a, i, i));
			mpf_add(tmp, gmpfvi(x, i), tmp);
			smpfvi(new_x, i, tmp);
		}

		get_residual_mpfvector(res, b, a, new_x);
		mpf_mul(tmp, reps, first_norm2_res);
		mpf_add(tmp, tmp, aeps);
		norm2_mpfvector(tmp1, res);
		if(mpf_cmp(tmp1, tmp) < 0)
			return;

		printf("%5ld %25.17e\n", times, mpf2double(tmp1));

		subst_mpfvector(x, new_x);
	}

	/* free */
	mpf_clear(tmp);
	mpf_clear(tmp1);
	mpf_clear(first_norm2_res);
	free_mpfvector(new_x);
	free_mpfvector(res);
}

/* Successive Over-Relaxation Method */
void mpf_sor(MPFVector x, MPFMatrix a, MPFVector b, mpf_t omega, mpf_t aeps, mpf_t reps, long int max_times)
{
	long int times, i, j, dim;
	mpf_t tmp, tmp1, first_norm2_res;
	MPFVector new_x, res;

	dim = x->dim;

	/* Initialize */
	mpf_init2(tmp, x->prec);
	mpf_init2(tmp1, x->prec);
	mpf_init2(first_norm2_res, x->prec);
	new_x = init2_mpfvector(dim, x->prec);
	res = init2_mpfvector(dim, x->prec);

	set0_mpfvector(new_x);
	for(i = 0; i < dim; i++)
	{
		mpf_div(tmp, gmpfvi(b, i), gmpfmij(a, i, i));
		smpfvi(x, i, tmp);
	}

	norm2_mpfvector(first_norm2_res, b);

	for(times = 0; times <= max_times; times++)
	{
		for(i = 0; i < dim; i++)
		{
			mpf_set_ui(tmp, 0UL);
			for(j = 0; j < i; j++)
			{
				mpf_mul(tmp1, gmpfmij(a, i, j), gmpfvi(new_x, j));
				mpf_add(tmp, tmp, tmp1);
			}
			for(j = i + 1; j < dim; j++)
			{
				mpf_mul(tmp1, gmpfmij(a, i, j), gmpfvi(x, j));
				mpf_add(tmp, tmp, tmp1);
			}

			/* tmp = (b[i] - tmp)/a[i][i] - x[i]; */
			/* new_x[i] = x[i] + omega * tmp;     */

			mpf_sub(tmp, gmpfvi(b, i), tmp);
			mpf_div(tmp, tmp, gmpfmij(a, i, i));
			mpf_sub(tmp, tmp, gmpfvi(x, i));
			mpf_mul(tmp, omega, tmp);
			mpf_add(tmp, gmpfvi(x, i), tmp);
			smpfvi(new_x, i, tmp);
		}

		get_residual_mpfvector(res, b, a, new_x);
		mpf_mul(tmp, reps, first_norm2_res);
		mpf_add(tmp, tmp, aeps);
		norm2_mpfvector(tmp1, res);
		if(mpf_cmp(tmp1, tmp) < 0)
			return;

		printf("%5ld %25.17e\n", times, mpf2double(tmp1));

		subst_mpfvector(x, new_x);
	}

	/* free */
	mpf_clear(tmp);
	mpf_clear(tmp1);
	mpf_clear(first_norm2_res);
	free_mpfvector(new_x);
	free_mpfvector(res);
}
#endif
