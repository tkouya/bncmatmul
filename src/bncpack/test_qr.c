/********************************************************************************/
/* test_qr.c:                                                                   */
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
/* Test Program for qr.c                         */
/*************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bnc.h"

void get_dproblem(DMatrix a)
{
	long int i, j, k;
	double tmp;

	/* Lotkin Matrix */
	for(i = 0; i < a->col_dim; i++)
		set_dmatrix_ij(a, 0, i, 1.0);
	for(i = 1; i < a->row_dim; i++)
	{
		for(j = 0; j < a->col_dim; j++)
			set_dmatrix_ij(a, i, j, 1.0 / (i + j + 1));
	}
}

#ifdef USE_GMP
void get_mpfproblem(MPFMatrix a)
{
	long int i, j, k;
	mpf_t tmp;

	mpf_init(tmp);

	/* Lotkin Matrix */
	for(i = 0; i < a->col_dim; i++)
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
}
#endif

#define DIM 5

main()
{
	DMatrix da, dq, dr, da2;
#ifdef USE_GMP
	MPFMatrix mpfa, mpfq, mpfr, mpfa2;
#endif
	long int i, j;

/* Double */
dstart:
	/* initialize */
	da = init_dmatrix(DIM, DIM);
	da2 = init_dmatrix(DIM, DIM);
	dq = init_dmatrix(DIM, DIM);
	dr = init_dmatrix(DIM, DIM);

	/* get problem */
	get_dproblem(da);
	subst_dmatrix(da2, da);

	print_dmatrix(da);

	/* Gram-Schmidt */
//	dgram_schmidt(dq, dr, da);
	dmgram_schmidt(dq, dr, da);

	printf("q:\n");print_dmatrix(dq);printf("\n");
	printf("r:\n");print_dmatrix(dr);printf("\n");

	mul_dmatrix(da, dq, dr);
	sub_dmatrix(dr, da, da2);
	printf("q * r - a:\n"); print_dmatrix(dr);printf("\n");

	transpose_dmatrix(dr, dq);
	mul_dmatrix(da, dr, dq);
	printf("q^T * q:\n"); print_dmatrix(da);printf("\n");

	/* QR Method */
	subst_dmatrix(da, da2);
//	int_sym_rand_dmatrix(da, DIM, 1);
	dqr(da, DIM * 10);
	printf("QR:\n"); print_dmatrix(da);printf("\n");
	for(i = 0; i < DIM; i++)
		printf("%5d: %25.17e\n", i, gdmij(da, i, i));

//	goto end;

#ifdef USE_GMP
/* MPF */
mpfstart:

	set_bnc_default_prec(256);

	/* initialize */
	mpfa = init_mpfmatrix(DIM, DIM);
	mpfq = init_mpfmatrix(DIM, DIM);
	mpfr = init_mpfmatrix(DIM, DIM);
	mpfa2 = init_mpfmatrix(DIM, DIM);

	/* get problem */
	get_mpfproblem(mpfa);
	subst_mpfmatrix(mpfa2, mpfa);

	print_mpfmatrix(mpfa);

	/* Gram-Schmidt */
	mpf_gram_schmidt(mpfq, mpfr, mpfa);

	printf("q:\n"); print_mpfmatrix(mpfq); printf("\n");
	printf("r:\n"); print_mpfmatrix(mpfr); printf("\n");

	mul_mpfmatrix(mpfa, mpfq, mpfr);
	sub_mpfmatrix(mpfr, mpfa, mpfa2);
	printf("q * r - a:\n"); print_mpfmatrix(mpfr);printf("\n");

	transpose_mpfmatrix(mpfr, mpfq);
	mul_mpfmatrix(mpfa, mpfr, mpfq);
	printf("q^T * q:\n"); print_mpfmatrix(mpfa);printf("\n");

	/* QR */
	subst_mpfmatrix(mpfa, mpfa2);
	mpf_qr(mpfa, DIM * 10);
	printf("QR:\n"); print_mpfmatrix(mpfa);printf("\n");
	for(i = 0; i < DIM; i++)
		printf("%5d: %25.17e\n", i, mpf_get_d(gmpfmij(mpfa, i, i)));

	/* end */
	free_mpfmatrix(mpfa);
#endif

	/* print itimes */
end:
	return 0;

}

