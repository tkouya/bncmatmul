/********************************************************************************/
/* test_cddlinear.c:                                                            */
/* Copyright (C) 2023 Tomonori Kouya                                            */
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
/* For Debug                                     */
/*************************************************/
#include <stdio.h>
#include <math.h>
#include <matmul_strassen.h>

int main(void)
{
	CDDVector dvec;
	CDDMatrix dmat, dmat_a, dmat_b;
#ifdef USE_GMP
	CMPFVector mpfvec;
	CMPFMatrix mpfmat, mpfmat_a, mpfmat_b;
#endif

	long int dim, row_dim, col_dim, i, j;

#ifdef USE_GMP
	set_bnc_default_prec(128);
#endif

	/* For Vectors */
	//for(dim = 2; dim <= 5; dim++)
	for(dim = 2; dim <= 1000; dim *= 10)
	{
		dvec = init_cddvector(dim);
#ifdef USE_GMP
		mpfvec = init_cmpfvector(dim);
		if(mpfvec == NULL)
			exit(0);
#endif
		
		// print vectors
		printf("DD prec. vector: \n");
		print_cddvector(dvec);
#ifdef USE_GMP
		printf("Multiple prec. vector: \n");
		print_cmpfvector(mpfvec);
#endif
		for(i = 0; i < dim; i++)
		{
			set_cddvector_i_cd(dvec, i, (double)(i + 1) + (double)(i + 1) * I);
			printf("test2\n"); fflush(stdout);
#ifdef USE_GMP
			set_cmpfvector_i_d(mpfvec, i, (double)(i + 1) + (double)(i + 1) * I);
#endif
		}

		// print vectors
		printf("DD prec. vector: \n");
		print_cddvector(dvec);
#ifdef USE_GMP
		printf("Multiple prec. vector: \n");
		print_cmpfvector(mpfvec);
#endif

		free_cddvector(dvec);
#ifdef USE_GMP
	printf("test3\n"); fflush(stdout);
		free_cmpfvector(mpfvec);
#endif

	}

	/* For Matrices */
	for(row_dim = 1; row_dim <= 1024; row_dim *= 2)
	//for(row_dim = 2; row_dim <= 3; row_dim++)
	{
		for(col_dim = 1; col_dim <= 1024; col_dim *= 2)
		//for(col_dim = 2; col_dim <= 3; col_dim++)
		{
			dmat   = init_cddmatrix(row_dim, col_dim);
			dmat_a = init_cddmatrix(row_dim, col_dim);
			dmat_b = init_cddmatrix(row_dim, col_dim);

#ifdef USE_GMP
			mpfmat   = init_cmpfmatrix(row_dim, col_dim);
			mpfmat_a = init_cmpfmatrix(row_dim, col_dim);
			mpfmat_b = init_cmpfmatrix(row_dim, col_dim);
#endif

			srand(10);
			for(i = 0; i < row_dim; i++)
			{
				for(j = 0; j < col_dim; j++)
				{
					//set_cdmatrix_ij(dmat, i, j, (double)(i + j + 1) + (double)(col_dim + row_dim - i - j) * I);
					set_cddmatrix_ij_cd(dmat, i, j, (double)rand() + (double)rand() * I);
#ifdef USE_GMP
					//set_cmpfmatrix_ij_d(mpfmat, i, j, (double)(i + j + 1) + (double)(col_dim + row_dim - i - j) * I);
					set_cmpfmatrix_ij_d(mpfmat, i, j, (double)rand() + (double)rand() * I);
#endif
					//printf("%10.3f %10.3f\n", get_fmatrix_ij(fmat, i, j), get_dmatrix_ij(dmat, i, j));
					printf("(%3ld, %3ld) ", i, j);
                    rdd_out_str(get_cddmatrix_ij_cddfloat(dmat, i, j).val_re);
                    printf(" + ");
                    rdd_out_str(get_cddmatrix_ij_cddfloat(dmat, i, j).val_im);
                    printf(" * I\n");
				}
			}
			printf("ddmat:\n");
			print_cddmatrix(dmat);
			subst_cddmatrix(dmat_a, dmat);
			if(row_dim == col_dim)
			{
				printf("dmat_inv:\n");
				inv_cddmatrix(dmat_a);
				print_cddmatrix(dmat_a);
				printf("I = dmat * dmat_inv:\n");
				mul_cddmatrix(dmat_b, dmat, dmat_a);
				print_cddmatrix(dmat_b);
			}
			for(i = 0; i < dmat->re->row_dim; i++)
				for(j = 0; j < dmat->re->col_dim; j++)
                {
					printf("(%3ld, %3ld) ", i, j);
                    rdd_out_str(get_cddmatrix_ij_cddfloat(dmat, i, j).val_re);
                    printf(" + ");
                    rdd_out_str(get_cddmatrix_ij_cddfloat(dmat, i, j).val_im);
                    printf(" * I\n");
                }

#ifdef USE_GMP
			printf("mpfmat:\n");
			print_cmpfmatrix(mpfmat);
			subst_cmpfmatrix(mpfmat_a, mpfmat);
			if(row_dim == col_dim)
			{
				printf("mpfmat_inv:\n");
				inv_cmpfmatrix(mpfmat_a);
				print_cmpfmatrix(mpfmat_a);
				printf("I = mpfmat * mpfmat_inv:\n");
				mul_cmpfmatrix(mpfmat_b, mpfmat, mpfmat_a);
				print_cmpfmatrix(mpfmat_b);
			}
#endif

			free_cddmatrix(dmat);
			free_cddmatrix(dmat_a);
			free_cddmatrix(dmat_b);
#ifdef USE_GMP
			free_cmpfmatrix(mpfmat);
			free_cmpfmatrix(mpfmat_a);
			free_cmpfmatrix(mpfmat_b);
#endif

			printf("\n");
		}
	}
}
