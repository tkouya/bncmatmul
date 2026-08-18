/********************************************************************************/
/* test_jacobi.c:                                                               */
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
#include <stdio.h>
#include <math.h>

#include "bnc.h"

#define DIM 5

main()
{
	FMatrix fa;
	FVector fb, fx, fans;
	float fomega;
	long i, j;

	DMatrix da;
	DVector db, dx, dans;
	double dcond, dmin_eig, dmax_eig, domega, dtmp;

#ifdef USE_GMP
	MPFMatrix mpfa;
	MPFVector mpfb, mpfx, mpfans;
	mpf_t mpftmp, mpfomega, mpfaeps, mpfreps;
#endif

/* float */
	fa = init_fmatrix(DIM, DIM);
	fb = init_fvector(DIM);
	fx = init_fvector(DIM);
	fans = init_fvector(DIM);

	for(i = 0; i < DIM; i++)
	{
		sfvi(fx, i, (float)i);
		for(j = 0; j < DIM; j++)
		{
			if(j == i - 1 || j == i + 1)
				sfmij(fa, i, j, (float)-1);
			else if(j == i)
				sfmij(fa, i, j, (float)2);
			else
				sfmij(fa, i, j, (float)0);
		}
	}
	mul_fmatrix_fvec(fb, fa, fx);

	printf("Jacobi Iterative Method\n");
	fjacobi(fx, fa, fb, 1e-20, 1e-5, DIM*100);
	print_fvector(fx);

	printf("Gauss-Seidel Method\n");
	fgs(fx, fa, fb, 1e-20, 1e-5, DIM*100);
	print_fvector(fx);

	printf("Successive Over-Relazation Method\n");
		fomega = 0.5;
		printf("Omega: %15.7e\n", fomega);
		fsor(fx, fa, fb, fomega, 1e-20, 1e-5, DIM*100);

		fomega = 1.5;
		printf("Omega: %15.7e\n", fomega);
		fsor(fx, fa, fb, fomega, 1e-20, 1e-5, DIM*100);

		fomega = 1.9;
		printf("Omega: %15.7e\n", fomega);
		fsor(fx, fa, fb, fomega, 1e-20, 1e-5, DIM*100);

	print_fvector(fx);

	free_fmatrix(fa);
	free_fvector(fb);
	free_fvector(fx);
	free_fvector(fans);

/* double */
	da = init_dmatrix(DIM, DIM);
	db = init_dvector(DIM);
	dx = init_dvector(DIM);
	dans = init_dvector(DIM);

	dmin_eig = 1.0e+300;
	dmax_eig = 0.0;

	for(i = 0; i < DIM; i++)
	{
		dcond = fabs(0.5 * (1.0 - cos((2*(i+1)-1)*3.1415926535897932384/(2*DIM + 1))));
		printf("%25.17e\n", dcond);
		if(dcond < dmin_eig)
			dmin_eig = dcond;
		if(dcond > dmax_eig)
			dmax_eig = dcond;
		sdvi(dx, i, (double)i);
	}
	for(i = 0; i < DIM; i++)
	{
		for(j = 0; j < DIM; j++)
		{
			if(j == i - 1 || j == i + 1)
				sdmij(da, i, j, (double)-1);
			else if(j == i)
				sdmij(da, i, j, (double)2);
			else
				sdmij(da, i, j, (double)0);
		}
	}
	mul_dmatrix_dvec(db, da, dx);

	dcond = dmax_eig / dmin_eig;

	printf("Jacobi Iterative Method\n");
	//djacobi(dx, da, db, 1e-50, 1e-10, DIM*100);
	bnc_djacobi(dx, da, db, 1e-50, 1e-10, DIM*100);
	print_dvector(dx);

	printf("Gauss-Seidel Method\n");
	dgs(dx, da, db, 1e-50, 1e-10, DIM*100);
	print_dvector(dx);

	printf("Successive Over-Relazation Method\n");
		domega = 0.5;
		printf("Omega: %25.17e\n", domega);
		dsor(dx, da, db, domega, 1e-50, 1e-10, DIM*100);

		domega = 1.5;
		printf("Omega: %25.17e\n", domega);
		dsor(dx, da, db, domega, 1e-50, 1e-10, DIM*100);

		domega = 1.9;
		printf("Omega: %25.17e\n", domega);
		dsor(dx, da, db, domega, 1e-50, 1e-10, DIM*100);

	print_dvector(dx);

	free_dmatrix(da);
	free_dvector(db);
	free_dvector(dx);
	free_dvector(dans);

#ifdef USE_GMP
/* mpf_t */

	set_bnc_default_prec(128);

	mpfa = init_mpfmatrix(DIM, DIM);
	mpfb = init2_mpfvector(DIM, 256);
	mpfx = init2_mpfvector(DIM, 512);
	mpfans = init_mpfvector(DIM);

	mpf_init_set_d(mpfaeps, 1.0e-300);
	mpf_init_set_d(mpfreps, 1.0e-50);
	mpf_init2(mpftmp, mpfx->prec);
	mpf_init2(mpfomega, mpfx->prec);

	for(i = 0; i < DIM; i++)
	{
		mpf_set_ui(mpftmp, i);
		smpfvi(mpfx, i, mpftmp);
		for(j = 0; j < DIM; j++)
		{
			if(j == i - 1 || j == i + 1)
				smpfmijs(mpfa, i, j, "-1", 10);
			else if(j == i)
				smpfmijs(mpfa, i, j, "2", 10);
			else
				smpfmijs(mpfa, i, j, "0", 10);
		}
	}
	mul_mpfmatrix_mpfvec(mpfb, mpfa, mpfx);

	printf("Jacobi Iterative Method\n");
	mpf_jacobi(mpfx, mpfa, mpfb, mpfaeps, mpfreps, DIM*100);
	print_mpfvector(mpfx);

	printf("Gauss-Seidel Method\n");
	mpf_gs(mpfx, mpfa, mpfb, mpfaeps, mpfreps, DIM*100);
	print_mpfvector(mpfx);

	printf("Successive Over-Relazation Method\n");
		mpf_set_str(mpfomega, "0.5", 10);
		printf("Omega: "); mpf_out_str(stdout, 10, 0, mpfomega); printf("\n");
		mpf_sor(mpfx, mpfa, mpfb, mpfomega, mpfaeps, mpfreps, DIM*100);

		mpf_set_str(mpfomega, "1.5", 10);
		printf("Omega: "); mpf_out_str(stdout, 10, 0, mpfomega); printf("\n");
		mpf_sor(mpfx, mpfa, mpfb, mpfomega, mpfaeps, mpfreps, DIM*100);

		mpf_set_str(mpfomega, "1.9", 10);
		printf("Omega: "); mpf_out_str(stdout, 10, 0, mpfomega); printf("\n");
		mpf_sor(mpfx, mpfa, mpfb, mpfomega, mpfaeps, mpfreps, DIM*100);

	print_mpfvector(mpfx);

	mpf_clear(mpftmp);
	mpf_clear(mpfaeps);
	mpf_clear(mpfreps);
	mpf_clear(mpfomega);
	free_mpfmatrix(mpfa);
	free_mpfvector(mpfb);
	free_mpfvector(mpfx);
	free_mpfvector(mpfans);
#endif
}
