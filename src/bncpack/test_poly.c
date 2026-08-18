/********************************************************************************/
/* test_poly.c:                                                                 */
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
/* Test Program for poly.c */

#include <stdio.h>
#include "bnc.h"

#define MAX_POLY_LEN 4096
#define MAX_DEGREE 1024

main()
{
	long int i;

	FPoly fpa, fpb, fpc;
	DPoly dpa, dpb, dpc;
	FCmplx fca, fcret;
	DCmplx dca, dcret;
#ifdef USE_GMP
	MPFPoly mpf_pa, mpf_pb, mpf_pc;
	mpf_t mpf_x, mpf_ret;
	MPFCmplx mpfca, mpf_cret;
#endif

/* float */
	/* init */
	fpa = init_fpoly(MAX_POLY_LEN);
	fpb = init_fpoly(MAX_POLY_LEN);
	fpc = init_fpoly(MAX_POLY_LEN);

	fca = init_fcmplx();
	fcret = init_fcmplx();
	set_real_fcmplx(fca, 1.0);
	set_image_fcmplx(fca, 1.0);

	for(i = 0; i <= MAX_DEGREE; i++)
	{
		set_fpoly_i(fpa, i, (float)i);
		set_fpoly_i(fpb, i, (float)rand());
		set_fpoly_i(fpc, i, (float)rand());
	}

	printf("fpa: O(x^%d)\n", setdegree_fpoly(fpa));print_fpoly(fpa);
	printf("fpa (1) = %15.7e\n", eval_fpoly(fpa, 1.0));
	printf("fpa'(1) = %15.7e\n", eval_diff_fpoly(fpa, 1.0));
	printf("fpa (1+1i) = "); ceval_fpoly(fcret, fpa, fca); print_fcmplx(fcret);
	printf("fpa'(1+i1) = "); ceval_diff_fpoly(fcret, fpa, fca); print_fcmplx(fcret);
	printf("fpa (2) = %15.7e\n", eval_fpoly(fpa, 2.0));
	printf("fpa'(2) = %15.7e\n", eval_diff_fpoly(fpa, 2.0));
	printf("fpb: \n");print_fpoly(fpb);
	printf("fpc: \n");print_fpoly(fpc);

	/* clear */
	free_fpoly(fpa);
	free_fpoly(fpb);
	free_fpoly(fpc);

/* double */
	/* init */
	dpa = init_dpoly(MAX_POLY_LEN);
	dpb = init_dpoly(MAX_POLY_LEN);
	dpc = init_dpoly(MAX_POLY_LEN);

	dca = init_dcmplx();
	dcret = init_dcmplx();
	set_real_dcmplx(dca, 1.0);
	set_image_dcmplx(dca, 1.0);

	for(i = 0; i <= MAX_DEGREE; i++)
	{
		set_dpoly_i(dpa, i, (double)i);
		set_dpoly_i(dpb, i, (double)rand());
		set_dpoly_i(dpc, i, (double)rand());
	}

	printf("dpa: O(x^%d)\n", setdegree_dpoly(dpa));print_dpoly(dpa);
	printf("dpa (1) = %25.17e\n", eval_dpoly(dpa, 1.0));
	printf("dpa'(1) = %25.17e\n", eval_diff_dpoly(dpa, 1.0));
	printf("dpa (1+1i) = "); ceval_dpoly(dcret, dpa, dca); print_dcmplx(dcret);
	printf("dpa'(1+i1) = "); ceval_diff_dpoly(dcret, dpa, dca); print_dcmplx(dcret);
	printf("dpa (2) = %25.17e\n", eval_dpoly(dpa, 2.0));
	printf("dpa'(2) = %25.17e\n", eval_diff_dpoly(dpa, 2.0));

	printf("dpb: \n");print_dpoly(dpb);
	printf("dpc: \n");print_dpoly(dpc);

	/* clear */
	free_dpoly(dpa);
	free_dpoly(dpb);
	free_dpoly(dpc);

#ifdef USE_GMP
	/* init */
	mpf_pa = init2_mpfpoly(MAX_POLY_LEN, 128);
	mpf_pb = init2_mpfpoly(MAX_POLY_LEN, 256);
	mpf_pc = init2_mpfpoly(MAX_POLY_LEN, 1024);

	mpfca = init2_mpfcmplx(1024);
	mpf_cret = init2_mpfcmplx(1024);
	set_real_mpfcmplx_ui(mpfca, 1UL);
	set_image_mpfcmplx_ui(mpfca, 1UL);

	for(i = 0; i <= MAX_DEGREE; i++)
	{
		set_mpfpoly_i_d(mpf_pa, i, (double)i);
		set_mpfpoly_i_d(mpf_pb, i, (double)rand());
		set_mpfpoly_i_d(mpf_pc, i, (double)rand());
	}

	mpf_init2(mpf_x, 128);
	mpf_init2(mpf_ret, 128);

	printf("mpf_pa: O(x^%d)\n", setdegree_mpfpoly(mpf_pa));print_mpfpoly(mpf_pa);

	mpf_set_ui(mpf_x, 1UL);
	printf("mpf_pa(1) = ");
		eval_mpfpoly(mpf_ret, mpf_pa, mpf_x);
		mpf_out_str(stdout, 0, 10, mpf_ret);
		printf("\n");
	printf("mpf_pa'(1) = ");
		eval_diff_mpfpoly(mpf_ret, mpf_pa, mpf_x);
		mpf_out_str(stdout, 0, 10, mpf_ret);
		printf("\n");
	mpf_set_ui(mpf_x, 1UL);

	printf("mpf_pa(1+1i) = ");
		ceval_mpfpoly(mpf_cret, mpf_pa, mpfca);
		print_mpfcmplx(mpf_cret);
	printf("mpf_pa'(1+1i) = ");
		ceval_diff_mpfpoly(mpf_cret, mpf_pa, mpfca);
		print_mpfcmplx(mpf_cret);

	mpf_set_ui(mpf_x, 2UL);
	printf("mpf_pa(2) = ");
		eval_mpfpoly(mpf_ret, mpf_pa, mpf_x);
		mpf_out_str(stdout, 0, 10, mpf_ret);
		printf("\n");
	printf("mpf_pa'(2) = ");
		eval_diff_mpfpoly(mpf_ret, mpf_pa, mpf_x);
		mpf_out_str(stdout, 0, 10, mpf_ret);
		printf("\n");

	mpf_set_ui(mpf_x, 100000UL);
	printf("mpf_pa(100000) = ");
		eval_mpfpoly(mpf_ret, mpf_pa, mpf_x);
		mpf_out_str(stdout, 0, 10, mpf_ret);
		printf("\n");
	printf("mpf_pa'(100000) = ");
		eval_diff_mpfpoly(mpf_ret, mpf_pa, mpf_x);
		mpf_out_str(stdout, 0, 10, mpf_ret);
		printf("\n");

	printf("mpf_pb: \n");print_mpfpoly(mpf_pb);
	printf("mpf_pc: \n");print_mpfpoly(mpf_pc);

	/* clear */
	free_mpfpoly(mpf_pa);
	free_mpfpoly(mpf_pb);
	free_mpfpoly(mpf_pc);
#endif

}
