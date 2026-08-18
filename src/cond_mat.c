/********************************************************************************/
/* cond_mat.c: Estimation of condition numbers for some square matrices         */
/*                                                                              */
/* Copyright (C) 2014 Tomonori Kouya                                            */
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
//#include "bnc.h"
//#include "mpflinear.h"
#include "gtestmat.h"

#define DIM 10
#define MPF_PREC 128

int main()
{
	int select_matrix;
	long int dim = DIM;
	unsigned long prec = MPF_PREC;
#ifdef USE_GMP
	mpf_t mpf_norm, mpf_norm_inv, mpf_cond;
	MPFMatrix mpf_mat, mpf_mat_inv;

	// set precision(binary)
	printf("Set precision in bits = "); scanf("%ld", &prec);

//	set_bnc_default_prec(MPF_PREC);
	set_bnc_default_prec(prec);

	// set dimension
	printf("Set Dimension = "); scanf("%ld", &dim);

	mpf_init(mpf_norm);
	mpf_init(mpf_norm_inv);
	mpf_init(mpf_cond);

	// init matrices
	mpf_mat = init_mpfmatrix(dim, dim);
	mpf_mat_inv = init_mpfmatrix(dim, dim);

	// Matrix selection
	printf("Select Matrix: \n");
	printf("1 ... Hilbert Matrix\n");
	printf("2 ... Lotkin Matrix\n");
	printf("3 ... Frank Matrix\n");

	scanf("No.Matrix ="); scanf("%d", &select_matrix);

	switch(select_matrix)
	{
		case 1:
			// Hilbert matrix
			printf("Hilbert Matrix: dim = %ld, prec = %ld\n", dim, get_bnc_default_prec());
			hilbert_mpfmatrix(mpf_mat, dim);
			break;

		case 2:
			// lotkin matrix
			printf("Lotkin Matrix: dim = %ld, prec = %ld\n", dim, get_bnc_default_prec());
			lotkin_mpfmatrix(mpf_mat, dim);
			break;

		case 3:
			// Frank Matrix
			printf("Frank Matrix: dim = %ld, prec = %ld\n", dim, get_bnc_default_prec());
			frank_mpfmatrix(mpf_mat, dim);
			break;

		default:
			// Random Matrix
			printf("Integer Random Matrix: seed = 1, dim = %ld, prec = %ld\n", dim, get_bnc_default_prec());
			int_sym_rand_mpfmatrix(mpf_mat, dim * dim, 1, dim);
			break;
	}

	// inverse
	subst_mpfmatrix(mpf_mat_inv, mpf_mat);
	inv_mpfmatrix(mpf_mat_inv);

	// normi
	normi_mpfmatrix(mpf_norm, mpf_mat);
	normi_mpfmatrix(mpf_norm_inv, mpf_mat_inv);
	mpf_mul(mpf_cond, mpf_norm, mpf_norm_inv);

	// printf
	printf("mpf_normi     = "); mpf_out_str(stdout, 10, 15, mpf_norm); printf("\n");
	printf("mpf_normi_inv = "); mpf_out_str(stdout, 10, 15, mpf_norm_inv); printf("\n");
	printf("condi         = "); mpf_out_str(stdout, 10, 15, mpf_cond); printf("\n");

	// norm1
	norm1_mpfmatrix(mpf_norm, mpf_mat);
	norm1_mpfmatrix(mpf_norm_inv, mpf_mat_inv);
	mpf_mul(mpf_cond, mpf_norm, mpf_norm_inv);

	// printf
	printf("mpf_norm1     = "); mpf_out_str(stdout, 10, 15, mpf_norm); printf("\n");
	printf("mpf_norm1_inv = "); mpf_out_str(stdout, 10, 15, mpf_norm_inv); printf("\n");
	printf("cond1         = "); mpf_out_str(stdout, 10, 15, mpf_cond); printf("\n");

	// clean
	mpf_clear(mpf_norm);
	mpf_clear(mpf_norm_inv);
	mpf_clear(mpf_cond);

	free_mpfmatrix(mpf_mat);
	free_mpfmatrix(mpf_mat_inv);
#endif

	return 0;
}
