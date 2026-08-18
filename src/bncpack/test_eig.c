/********************************************************************************/
/* test_eig.c:                                                                  */
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
/* Test Program for [f,d,mpf]power_eig           */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bnc.h"

void get_fproblem(FMatrix a, FVector init_vec)
{
	long int i, j, k;
	float tmp;

	/* Frank Matrix */
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			if(i < j)
				set_fmatrix_ij(a, i, j, (float)(a->col_dim - j));
			else
				set_fmatrix_ij(a, i, j, (float)(a->col_dim - i));
		}
	}

	/* set All 1*/
	for(i = 0; i < init_vec->dim; i++)
		set_fvector_i(init_vec, i, 1.0);

}

void get_dproblem(DMatrix a, DVector init_vec)
{
	long int i, j, k;
	double tmp;

	/* Frank Matrix */
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			if(i < j)
				set_dmatrix_ij(a, i, j, (double)(a->col_dim - j));
			else
				set_dmatrix_ij(a, i, j, (double)(a->col_dim - i));
		}
	}

	/* set All 1*/
	for(i = 0; i < init_vec->dim; i++)
		set_dvector_i(init_vec, i, 1.0);
}

#ifdef USE_GMP
void get_mpfproblem(MPFMatrix a, MPFVector init_vec)
{
	long int i, j, k;
	mpf_t tmp;

	mpf_init2(tmp, prec_mpfvector(init_vec));

	/* Frank Matrix */
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			if(i < j)
			{
				mpf_set_si(tmp, a->col_dim - j);
				set_mpfmatrix_ij(a, i, j, tmp);
			}
			else
			{
				mpf_set_si(tmp, a->col_dim - i);
				set_mpfmatrix_ij(a, i, j, tmp);
			}
		}
	}

	/* set All 1*/
	mpf_set_ui(tmp, 1UL);
	for(i = 0; i < init_vec->dim; i++)
		set_mpfvector_i(init_vec, i, tmp);
}
#endif

#define DIM 10

main()
{
	FMatrix fa;
	FVector finit_vec, fmaxeig_vec, fmineig_vec;
	float fmaxeig, fmineig, faeps, freps;
	DMatrix da;
	DVector dinit_vec, dmaxeig_vec, dmineig_vec;
	double dmaxeig, dmineig, daeps, dreps;
#ifdef USE_GMP
	MPFMatrix mpfa;
	MPFVector mpfinit_vec, mpfmaxeig_vec, mpfmineig_vec;
	mpf_t mpfmaxeig, mpfmineig, reps, aeps;
	MPFMatrix mpfa2;
	MPFVector mpfinit_vec2, mpfmaxeig_vec2, mpfmineig_vec2;
	mpf_t mpfmaxeig2, mpfmineig2, reps2, aeps2;
	long int itimes_mpf, itimes_mpf2;
#endif
	long int itimes_f, itimes_d;
	long int i, j;

/* Float */
	/* initialize */
	fa = init_fmatrix(DIM, DIM);
	finit_vec = init_fvector(DIM);
	fmaxeig_vec = init_fvector(DIM);
	fmineig_vec = init_fvector(DIM);

	/* get problem */
	get_fproblem(fa, finit_vec);

	print_fmatrix(fa);

	/* run fpower_eig */
	itimes_f = fpower_eig(&fmaxeig, fmaxeig_vec, fa, finit_vec, 1.0e-20, 1.0e-6, DIM * 10);

	/* print */
	printf("Iterative Times(fpower): %ld\n", itimes_f);

	/* Av = lambda v */
	mul_fmatrix_fvec(finit_vec, fa, fmaxeig_vec);
	for(i = 0; i < DIM; i++)
		printf("%5ld %25.17e %25.17e %25.17e %25.17e\n", i, fmaxeig, gfvi(fmaxeig_vec, i), gfvi(finit_vec, i), gfvi(finit_vec, i) / gfvi(fmaxeig_vec, i));

	/* get problem */
	get_fproblem(fa, finit_vec);

	/* run fivpower_eig */
	itimes_f = fivpower_eig(&fmineig, fmineig_vec, fa, finit_vec, 1.0e-20, 1.0e-6, DIM * 20);

	/* print */
	printf("Iterative Times(fivpower): %ld\n", itimes_f);

	/* Av = lambda v */
	get_fproblem(fa, finit_vec);
	mul_fmatrix_fvec(finit_vec, fa, fmineig_vec);
	for(i = 0; i < DIM; i++)
		printf("%5ld %25.17e %25.17e %25.17e %25.17e\n", i, fmineig, gfvi(fmineig_vec, i), gfvi(finit_vec, i), gfvi(finit_vec, i) / gfvi(fmineig_vec, i));
		
	/* end */
	free_fmatrix(fa);
	free_fvector(finit_vec);
	free_fvector(fmaxeig_vec);
	free_fvector(fmineig_vec);

/* Double */
	/* initialize */
	da = init_dmatrix(DIM, DIM);
	dinit_vec = init_dvector(DIM);
	dmaxeig_vec = init_dvector(DIM);
	dmineig_vec = init_dvector(DIM);

	/* get problem */
	get_dproblem(da, dinit_vec);

	print_dmatrix(da);

	/* run dpower_eig */
	itimes_d = dpower_eig(&dmaxeig, dmaxeig_vec, da, dinit_vec, 1.0e-99, 1.0e-13, DIM * 10);

	/* print */
	printf("Iterative Times(power): %ld\n", itimes_d);
	/* Av = lambda v */
	mul_dmatrix_dvec(dinit_vec, da, dmaxeig_vec);
	for(i = 0; i < DIM; i++)
		printf("%5ld %25.17e %25.17e %25.17e %25.17e\n", i, dmaxeig, gdvi(dmaxeig_vec, i), gdvi(dinit_vec, i), gdvi(dinit_vec, i) / gdvi(dmaxeig_vec, i));
		
	/* get problem */
	get_dproblem(da, dinit_vec);

	/* run dpower_eig */
	itimes_d = divpower_eig(&dmineig, dmineig_vec, da, dinit_vec, 1.0e-99, 1.0e-13, DIM * 20);

	/* print */
	printf("Iterative Times(divpower): %ld\n", itimes_d);
	/* Av = lambda v */
	get_dproblem(da, dinit_vec);
	mul_dmatrix_dvec(dinit_vec, da, dmineig_vec);
	for(i = 0; i < DIM; i++)
		printf("%5ld %25.17e %25.17e %25.17e %25.17e\n", i, dmineig, gdvi(dmineig_vec, i), gdvi(dinit_vec, i), gdvi(dinit_vec, i) / gdvi(dmineig_vec, i));

	/* end */
	free_dmatrix(da);
	free_dvector(dinit_vec);
	free_dvector(dmaxeig_vec);
	free_dvector(dmineig_vec);

#ifdef USE_GMP
/* MPF */
	set_bnc_default_prec(512);

	/* initialize */
	mpfa  = init_mpfmatrix(DIM, DIM);
	mpfa2 = init2_mpfmatrix(DIM, DIM, 1024);
	mpfinit_vec  = init_mpfvector(DIM);
	mpfinit_vec2 = init2_mpfvector(DIM, 1024);
	mpfmaxeig_vec  = init_mpfvector(DIM);
	mpfmaxeig_vec2 = init2_mpfvector(DIM, 1024);
	mpfmineig_vec  = init_mpfvector(DIM);
	mpfmineig_vec2 = init2_mpfvector(DIM, 1024);

	/* get problem */
	get_mpfproblem(mpfa, mpfinit_vec);
	get_mpfproblem(mpfa2, mpfinit_vec2);
	print_mpfmatrix(mpfa);
	print_mpfmatrix(mpfa2);

	/* run power_eig */
	mpf_init_set_d(reps, 1.0e-50); mpf_init_set_d(aeps, 1.0e-99);
	mpf_init2(reps2, 1024); mpf_init2(aeps2, 1024);
	mpf_set_d(reps2, 1.0e-50); mpf_set_d(aeps2, 1.0e-99);
	mpf_init(mpfmaxeig);
	mpf_init2(mpfmaxeig2, 1024);
	itimes_mpf  = mpf_power_eig(mpfmaxeig, mpfmaxeig_vec, mpfa, mpfinit_vec, aeps, reps, DIM * 10);
	itimes_mpf2 = mpf_power_eig(mpfmaxeig2, mpfmaxeig_vec2, mpfa2, mpfinit_vec2, aeps2, reps2, DIM * 10);

	/* print */
	printf("Iterative Times(mpf_power)\n");
	printf("mpf_t(512) : %ld\n", itimes_mpf);
	printf("mpf_t(1024): %ld\n", itimes_mpf2);

	/* Av = lambda v */
	mul_mpfmatrix_mpfvec(mpfinit_vec, mpfa, mpfmaxeig_vec);
	mul_mpfmatrix_mpfvec(mpfinit_vec2, mpfa2, mpfmaxeig_vec2);
	for(i = 0; i < DIM; i++)
		printf("%5ld %25.17e %25.17e %15.7e %15.7e\n",
			i,
			mpf_get_d(mpfmaxeig),
			mpf_get_d(mpfmaxeig2),
//			mpf_get_d(gmpfvi(mpfmaxeig_vec, i)),
//			mpf_get_d(gmpfvi(mpfmaxeig_vec2, i)),
			mpf_get_d(gmpfvi(mpfinit_vec, i)) / mpf_get_d(gmpfvi(mpfmaxeig_vec, i)),
			mpf_get_d(gmpfvi(mpfinit_vec2, i)) / mpf_get_d(gmpfvi(mpfmaxeig_vec2, i))
		);
		
	/* get problem */
	get_mpfproblem(mpfa, mpfinit_vec);
	get_mpfproblem(mpfa2, mpfinit_vec2);

	/* run power_eig */
	mpf_init(mpfmineig);
	mpf_init2(mpfmineig2, 1024);

	itimes_mpf  = mpf_ivpower_eig(mpfmineig, mpfmineig_vec, mpfa, mpfinit_vec, aeps, reps, DIM * 20);
	itimes_mpf2 = mpf_ivpower_eig(mpfmineig2, mpfmineig_vec2, mpfa2, mpfinit_vec2, aeps2, reps2, DIM * 20);

	/* print */
	printf("Iterative Times(mpf_ivpower)\n");
	printf("mpf_t(512) : %ld\n", itimes_mpf);
	printf("mpf_t(1024): %ld\n", itimes_mpf2);

	/* Av = lambda v */
	get_mpfproblem(mpfa, mpfinit_vec);
	get_mpfproblem(mpfa2, mpfinit_vec2);

	mul_mpfmatrix_mpfvec(mpfinit_vec, mpfa, mpfmineig_vec);
	mul_mpfmatrix_mpfvec(mpfinit_vec2, mpfa2, mpfmineig_vec2);

	for(i = 0; i < DIM; i++)
		printf("%5ld %25.17e %25.17e %15.7e %15.7e\n",
			i,
			mpf_get_d(mpfmineig),
			mpf_get_d(mpfmineig2),
//			mpf_get_d(gmpfvi(mpfmineig_vec, i)),
//			mpf_get_d(gmpfvi(mpfmineig_vec2, i)),
			mpf_get_d(gmpfvi(mpfinit_vec, i)) / mpf_get_d(gmpfvi(mpfmineig_vec, i)),
			mpf_get_d(gmpfvi(mpfinit_vec2, i)) / mpf_get_d(gmpfvi(mpfmineig_vec2, i))
		);

	/* end */
	free_mpfmatrix(mpfa);
	free_mpfmatrix(mpfa2);
	free_mpfvector(mpfinit_vec);
	free_mpfvector(mpfinit_vec2);
	free_mpfvector(mpfmaxeig_vec);
	free_mpfvector(mpfmaxeig_vec2);
	free_mpfvector(mpfmineig_vec);
	free_mpfvector(mpfmineig_vec2);
#endif
}

