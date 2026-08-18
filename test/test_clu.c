/********************************************************************************/
/* test_clu.c:                                                                  */
/* Copyright (C) 2011-2012 Tomonori Kouya                                       */
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
/* Test Program for                              */
/* CDLUdecomp/SolveCDLS, CDLUdecompP/SolveCDLUP, */
/* CDLUdecompC/SolveCDLSC,                       */
/* CMPFLUdecomp/SolveCMPFLS,                     */
/* CMPFLUdecompP/SolveCMPFLUP,                   */
/* CMPFLUdecompC/SolveCMPFLSC                    */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

//#include "bnc.h"
#include "matmul_strassen.h"
#ifdef USE_PTHREAD
  #include "bncpthread.h"
#endif

/* Create Double precision test problems */
void get_cdproblem(CDMatrix a, CDVector b, CDVector ans)
{
	long int i, j, k;
	double _Complex tmp;

	/* Lotkin Matrix */
/*	for(i = 0; i < a->col_dim; i++)
		set_dmatrix_ij(a, 0, i, 1.0);
	for(i = 1; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
			set_dmatrix_ij(a, i, j, 1.0 / (i + j + 1));
	}
*/
	/* random matrix */
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
			set_cdmatrix_ij(a, i, j, (double)rand() + (double)rand() * I);
	}

	/* Answer */
	for(i = 0; i < ans->dim; i++)
	{
		tmp = (double)i + 0 * I;
		set_cdvector_i(ans, i, tmp);
	}

	/* Make constant vector */
	mul_cdmatrix_cdvec(b, a, ans);
}

#ifdef USE_GMP
/* Create Multiple precision test problems */
void get_cmpfproblem(CMPFMatrix a, CMPFVector b, CMPFVector ans)
{
	long int i, j, k;
//	MPFCmplx tmp;
	mpc_t tmp;

//	tmp = init_mpfcmplx();
	mpc_init(tmp);

	/* Lotkin Matrix */
/*	for(i = 0; i < a->col_dim; i++)
		set_mpfmatrix_ij_d(a, 0, i, 1.0);
	for(i = 1; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
		{
			mpf_set_ui(tmp, 1UL);
			mpf_div_ui(tmp, tmp, (unsigned long)(i + j + 1));
			set_mpfmatrix_ij(a, i, j, tmp);
		}
	}
*/
	/* random matrix */
	for(i = 0; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
			set_cmpfmatrix_ij_d(a, i, j, (double)rand() + (double)rand() * I);
	}

	/* Answer */
	for(i = 0; i < ans->dim; i++)
	{
		//set_mpfcmplx_ui_ui(tmp, (unsigned long)i, 0UL);
		mpc_set_ui_ui(tmp, (unsigned long)i, 0UL, MPC_RNDNN);
		set_cmpfvector_i(ans, i, tmp);
	}

	/* Make constant vector */
	mul_cmpfmatrix_cmpfvec(b, a, ans);

	//free_mpfcmplx(tmp);
	mpc_clear(tmp);
}
#endif

//#define DIM 5
//#define DIM 10
#define DIM 128
//#define DIM 1024

int main(void)
{
	CDMatrix da;
	CDVector db, dx, dans;
#ifdef USE_GMP
	CMPFMatrix mpfa;
	CMPFVector mpfb, mpfx, mpfans;
	mpf_t reps, aeps;
#endif
	long int ret_f, ret_d, ret_mpf;
	long int row_ch[DIM], col_ch[DIM];
	long int i, j;

/* Double */
dstart:
	/* initialize */
	da = init_cdmatrix(DIM, DIM);
	db = init_cdvector(DIM);
	dx = init_cdvector(DIM);
	dans = init_cdvector(DIM);

	/* get problem */
	get_cdproblem(da, db, dans);

//	print_dmatrix(da);

	/* run DLUdecomp & SolveDLS */
	// ret_d = CDLUdecomp(da);
	// ret_d = CDLUdecompP(da, row_ch);
	 ret_d = CDLUdecompC(da, row_ch, col_ch);
	// ret_d = SolveCDLS(dx, da, db);
	// ret_d = SolveCDLSP(dx, da, db, row_ch);
	 ret_d = SolveCDLSC(dx, da, db, row_ch, col_ch);

	/* print */
//	printf("  i    row_ch[i]    col_ch[i]\n");
//	for(i = 0; i < DIM; i++)
//		printf("%5ld %25.17e %25.17e\n", i, get_cdvector_i(dx, i), get_cdvector_i(dans, i));
	printf("dx:\n");
	print_cdvector(dx);

	/* end */
	free_cdmatrix(da);
	free_cdvector(db);
	free_cdvector(dx);
	free_cdvector(dans);

//	goto end;

#ifdef USE_GMP
/* MPF */
mpfstart:
	set_bnc_default_prec(256);
	/* initialize */
	mpf_init(reps); mpf_init(aeps);
	mpfa = init_cmpfmatrix(DIM, DIM);
//	mpfa = init2_cmpfmatrix(DIM, DIM, 256);
	mpfb = init_cmpfvector(DIM);
//	mpfb = init2_cmpfvector(DIM, 256);
	mpfx = init_cmpfvector(DIM);
//	mpfx = init2_cmpfvector(DIM, 256);
	mpfans = init_cmpfvector(DIM);

	/* get problem */
	get_cmpfproblem(mpfa, mpfb, mpfans);

//	print_mpfmatrix(mpfa);
#ifdef USE_PTHREAD
  #ifndef PTHREAD_NUM
  	#define PTHREAD_NUM 2
  #endif
	ret_mpf = _pthread_CMPFLUdecompP(mpfa, row_ch, PTHREAD_NUM);
	ret_mpf = _pthread_SolveCMPFLSP(mpfx, mpfa, mpfb, row_ch, PTHREAD_NUM);
#else
	/* run MPFLUdecomp & SolveMPFLS */
	// ret_mpf = CMPFLUdecomp(mpfa);
	 ret_mpf = CMPFLUdecompP(mpfa, row_ch);
	// ret_mpf = CMPFLUdecompC(mpfa, row_ch, col_ch);
	// ret_mpf = SolveCMPFLS(mpfx, mpfa, mpfb);
	 ret_mpf = SolveCMPFLSP(mpfx, mpfa, mpfb, row_ch);
	// ret_mpf = SolveCMPFLSC(mpfx, mpfa, mpfb, row_ch, col_ch);
#endif
	/* print */
/*	for(i = 0; i < DIM; i++)
	{
		printf("%5ld ", i);
		mpf_out_str(stdout, 10, 0, get_cmpfvector_i(mpfx, i));
		printf(" ");
		mpf_out_str(stdout, 10, 0, get_cmpfvector_i(mpfans, i));
		printf("\n");
	}
*/
	printf("mpfx:\n");
	print_cmpfvector(mpfx);

	/* end */
	mpf_clear(reps); mpf_clear(aeps);
	free_cmpfmatrix(mpfa);
	free_cmpfvector(mpfb);
	free_cmpfvector(mpfx);
	free_cmpfvector(mpfans);
#endif

	/* print itimes */
end:
	return 0;

}
