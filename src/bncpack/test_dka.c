/********************************************************************************/
/* test_dka.c:                                                                  */
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
/* test_dka.c                                    */
/*************************************************/
#include <stdio.h>
#include "bnc.h"

#define MAX_LENGTH 1024

main()
{
	long int ftimes, dtimes, mpftimes;
	CFArray cfans, cfinit;
	FPoly ff;
	float fabs_eps, frel_eps;
	CDArray cdans, cdinit;
	DPoly df;
	double dabs_eps, drel_eps;
#ifdef USE_GMP
	CMPFArray cmpfans, cmpfinit;
	MPFPoly mpff;
	mpf_t mpfabs_eps, mpfrel_eps;
#endif

/* float */

	/* init */
	fabs_eps = (float)1.0e-50;
	frel_eps = (float)1.0e-6;
	ff = init_fpoly(MAX_LENGTH);
	cfans = init_cfarray(5);
	cfinit = init_cfarray(5);

/*	// ff = (x-1)(x-2)(x-3) 
	set_fpoly_i(ff, 0, (float)-6);
	set_fpoly_i(ff, 1, (float)11);
	set_fpoly_i(ff, 2, (float)-6);
	set_fpoly_i(ff, 3, (float)1);
*/
	/* ff = (x-1)(x-2)(x-3)(x-4)(x-5) */
	set_fpoly_i(ff, 0, (float)-120);
	set_fpoly_i(ff, 1, (float)274);
	set_fpoly_i(ff, 2, (float)-225);
	set_fpoly_i(ff, 3, (float)85);
	set_fpoly_i(ff, 4, (float)-15);
	set_fpoly_i(ff, 5, (float)1);

	print_fpoly(ff);

	/* set Aberth's initial value */
	fdka_init(cfinit, ff);
	print_cfarray(cfinit);

	/* DKA method */
	ftimes = fdka(cfans, cfinit, ff, 100, fabs_eps, frel_eps);

	/* print answer */
	printf("Iterative times: %d\n", ftimes);
	print_cfarray(cfans);

	/* clear */
	free_fpoly(ff);
	free_cfarray(cfans);
	free_cfarray(cfinit);

/* double */

	/* init */
	dabs_eps = 1.0e-100;
	drel_eps = 1.0e-14;
	df = init_dpoly(MAX_LENGTH);
	cdans = init_cdarray(10);
	cdinit = init_cdarray(10);

/* // ff = (x-1)(x-2)(x-3)(x-4)(x-5) 
	set_dpoly_i(df, 0, (double)-120);
	set_dpoly_i(df, 1, (double)274);
	set_dpoly_i(df, 2, (double)-225);
	set_dpoly_i(df, 3, (double)85);
	set_dpoly_i(df, 4, (double)-15);
	set_dpoly_i(df, 5, (double)1);
*/
	/* ff = (x-1)(x-2)...(x-10) */
	set_dpoly_i(df, 0, (double)3628800);
	set_dpoly_i(df, 1, (double)-10628640);
	set_dpoly_i(df, 2, (double)12753576);
	set_dpoly_i(df, 3, (double)-8409500);
	set_dpoly_i(df, 4, (double)3416930);
	set_dpoly_i(df, 5, (double)-902055);
	set_dpoly_i(df, 6, (double)157773);
	set_dpoly_i(df, 7, (double)-18150);
	set_dpoly_i(df, 8, (double)1320);
	set_dpoly_i(df, 9, (double)-55);
	set_dpoly_i(df,10, (double)1);

	print_dpoly(df);

	/* set Aberth's initial value */
	ddka_init(cdinit, df);
	print_cdarray(cdinit);

	/* DKA method */
	dtimes = ddka(cdans, cdinit, df, 1000, dabs_eps, drel_eps);

	/* print answer */
	printf("Iterative times: %d\n", dtimes);
	print_cdarray(cdans);

	/* clear */
	free_dpoly(df);
	free_cdarray(cdans);
	free_cdarray(cdinit);

#ifdef USE_GMP
/* mpf_t */

	set_bnc_default_prec(512);

	/* init */
	mpf_init_set_d(mpfabs_eps, 1.0e-300);
	mpf_init_set_d(mpfrel_eps, 1.0e-25);
	mpff = init_mpfpoly(MAX_LENGTH);
	cmpfans = init_cmpfarray(20);
	cmpfinit = init_cmpfarray(20);
/*
	//  ff = (x-1)(x-2)...(x-10) 
	set_mpfpoly_i_str(mpff, 0, "3628800", 10);
	set_mpfpoly_i_str(mpff, 1, "-10628640", 10);
	set_mpfpoly_i_str(mpff, 2, "12753576", 10);
	set_mpfpoly_i_str(mpff, 3, "-8409500", 10);
	set_mpfpoly_i_str(mpff, 4, "3416930", 10);
	set_mpfpoly_i_str(mpff, 5, "-902055", 10);
	set_mpfpoly_i_str(mpff, 6, "157773", 10);
	set_mpfpoly_i_str(mpff, 7, "-18150", 10);
	set_mpfpoly_i_str(mpff, 8, "1320", 10);
	set_mpfpoly_i_str(mpff, 9, "-55", 10);
	set_mpfpoly_i_str(mpff,10, "1", 10);
*/
	// ff = (x-1)(x-2)...(x-20) 
	set_mpfpoly_i_str(mpff, 0, "2432902008176640000", 10);
	set_mpfpoly_i_str(mpff, 1, "-8752948036761600000", 10);
	set_mpfpoly_i_str(mpff, 2, "13803759753640704000", 10);
	set_mpfpoly_i_str(mpff, 3, "-12870931245150988800", 10);
	set_mpfpoly_i_str(mpff, 4, "8037811822645051776", 10);
	set_mpfpoly_i_str(mpff, 5, "-3599979517947607200", 10);
	set_mpfpoly_i_str(mpff, 6, "1206647803780373360", 10);
	set_mpfpoly_i_str(mpff, 7, "-311333643161390640", 10);
	set_mpfpoly_i_str(mpff, 8, "63030812099294896", 10);
	set_mpfpoly_i_str(mpff, 9, "-10142299865511450", 10);
	set_mpfpoly_i_str(mpff,10, "1307535010540395", 10);
	set_mpfpoly_i_str(mpff,11, "-135585182899530", 10);
	set_mpfpoly_i_str(mpff,12, "11310276995381", 10);
	set_mpfpoly_i_str(mpff,13, "-756111184500", 10);
	set_mpfpoly_i_str(mpff,14, "40171771630", 10);
	set_mpfpoly_i_str(mpff,15, "-1672280820", 10);
	set_mpfpoly_i_str(mpff,16, "53327946", 10);
	set_mpfpoly_i_str(mpff,17, "-1256850", 10);
	set_mpfpoly_i_str(mpff,18, "20615", 10);
	set_mpfpoly_i_str(mpff,19, "-210", 10);
	set_mpfpoly_i_str(mpff,20, "1", 10);

	print_mpfpoly(mpff);fflush(stdout);

	/* set Aberth's initial value */
	mpf_dka_init(cmpfinit, mpff);

	/* DKA method */
	mpftimes = mpf_dka(cmpfans, cmpfinit, mpff, 1000, mpfabs_eps, mpfrel_eps);

	/* print answer */
	printf("Iterative times: %d\n", mpftimes);
	print_cmpfarray(cmpfans);

	/* clear */
	free_mpfpoly(mpff);
	free_cmpfarray(cmpfans);
	free_cmpfarray(cmpfinit);

#endif
}

