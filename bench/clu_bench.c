/********************************************************************************/
/* clu_bench.c:                                                                 */
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
/* stdio.h */
#ifndef _STDIO_H
#include <stdio.h>
#endif

/* math.h */
#ifndef _MATH_H
#include <math.h>
#endif

//#include "bnc.h"

#ifdef USE_IMKL
	#include "mkl.h"
	#include "mkl_cblas.h" // for Intel Math Kernel Library
#endif

#include "bncomp.h"
#include "matmul_strassen.h"

//#include "lu_bench.h"
#include "get_secv.h"

#ifdef USE_GMP

int main(int argc, char *argv[])
{
	long int dim, *pivot;
	int num_threads;
	double stime, etime[6];
	CDMatrix da;
	CDVector db, dx, dans;
#ifdef USE_GMP
	unsigned long prec;
	long int block_size, min_dim, limit_dim;
	CMPFMatrix mpfa;
	CMPFVector mpfb, mpfx, mpfans;
	mpf_t reps, aeps, relerr[7];
#endif
	long int ret_f, ret_d, ret_mpf;
	long int *row_ch, *col_ch;
#ifdef USE_IMKL
	lapack_int *row_ch_imkl;
#endif
	char fname_A[256], fname_true_x[256], fname_vec_b[256];
	long int i, j;

/* Double */
//	dim = 128;
	if(argc <= 1)
	{
#ifdef USE_GMP
	#ifndef _OPENMP
		fprintf(stderr, "Usage: %s [dim] [prec_b]\n", argv[0]);
	#else // _OPENMP
		fprintf(stderr, "Usage: %s [dim] [prec_b] [#threads]\n", argv[0]);
	#endif // _OPENMP
#else // USE_GMP
	#ifndef _OPENMP
		fprintf(stderr, "Usage: %s [dim]\n", argv[0]);
	#else // _OPENMP
		fprintf(stderr, "Usage: %s [dim] [#threads] \n", argv[0]);
	#endif // _OPENMP
#endif // USE_GMP
		return 0;
	}
	dim = atol(argv[1]);
	if(dim <= 0)
		return 0;

#ifdef USE_GMP
	prec = 128;
	if(argc >= 3)
	{
		prec = (unsigned long)atol(argv[2]);
	//	if(prec < 128)
	//		prec = 128;
	}

#ifdef _OPENMP
	num_threads = 1;
	if(argc >= 4)
	{
		num_threads = atoi(argv[3]);
		if(num_threads < 1)
			num_threads = 1;
	}

//	omp_set_num_threads(num_threads);
//	printf("#Threads: %d\n", omp_get_num_threads());
	set_bncomp_num_threads(num_threads);
#endif // _OPENMP

	goto mpfstart;

#endif // USE_GMP

//	goto end;

#ifdef USE_GMP
/* MPF */
mpfstart: ;

#ifdef USE_OZ
	#ifndef OZ_MAX_NUM_DIV
	#define OZ_MAX_NUM_DIV 10
	#endif // OZ_MAX_NUM_DIV
	int max_num_div = OZ_MAX_NUM_DIV;
//	#else
//	int max_num_div = 10;
//	#endif // OZ_MAX_NUM_DIV
	printf("Ozaki scheme, ");
	printf("max_num_div = %d\n", max_num_div);
#endif // USE_OZ

	set_bnc_default_prec(prec);
	/* initialize */
	mpf_init(reps); mpf_init(aeps);
	mpf_init(relerr[0]);
	mpf_init(relerr[1]);
	mpf_init(relerr[2]);
	mpf_init(relerr[3]);
	mpf_init(relerr[4]);
	mpf_init(relerr[5]);
	mpf_init(relerr[6]);
	mpfa = init_cmpfmatrix(dim, dim);
//	mpfa = init2_mpfmatrix(dim, dim, 256);
	mpfb = init_cmpfvector(dim);
//	mpfb = init2_mpfvector(dim, 256);
	mpfx = init_cmpfvector(dim);
//	mpfx = init2_mpfvector(dim, 256);
	mpfans = init_cmpfvector(dim);
	pivot = (long int *)calloc(dim, sizeof(long int));

	/* get problem */
	//get_mpfproblem(mpfa, mpfb, mpfans);
	sprintf(fname_A, "../python/cmat_a_%ld_%ld_b2048_c.txt", dim, dim);
	sprintf(fname_true_x, "../python/cvec_true_x_%ld_b2048_c.txt", dim);
	sprintf(fname_vec_b, "../python/cvec_b_%ld_b2048_c.txt", dim);	

	read_test_linear_eq_c(mpfa, mpfans, mpfb, (int)dim, fname_A, fname_true_x, fname_vec_b);

//	print_mpfmatrix(mpfa);

	/* run MPFLUdecomp & SolveMPFLS */
	//stime = get_secv();
	stime = get_real_secv();
#ifndef _OPENMP
	//ret_mpf = CMPFLUdecomp(mpfa);
	ret_mpf = CMPFLUdecompPM(mpfa, pivot);
#else // _OPENMP
	//ret_mpf = CMPFLUdecomp_omp(mpfa);
	ret_mpf = CMPFLUdecompPM_omp(mpfa, pivot);
#endif // _OPENMP
	//etime[4] = get_secv() - stime;
	etime[4] = get_real_secv() - stime;
	// ret_mpf = MPFLUdecompP(mpfa, row_ch);
	// ret_mpf = MPFLUdecompC(mpfa, row_ch, col_ch);
	//ret_mpf = SolveCMPFLS(mpfx, mpfa, mpfb);
	ret_mpf = SolveCMPFLSPM(mpfx, mpfa, mpfb, pivot);
	printf("normalLU(%ld, %ld): %g, ", dim, get_bnc_default_prec(), etime[4]);
	// ret_mpf = SolveMPFLSP(mpfx, mpfa, mpfb, row_ch);
	//relerr_element_cmpfvector(relerr[0], relerr[1], relerr[2], mpfx, mpfans, 0);
	relerr3_cmpfvector(relerr[0], relerr[1], relerr[2], relerr[3], relerr[4], relerr[5], relerr[6], mpfx, mpfans, 0);
	//printf("relerr(max, min, ||relerr||): ");
	mpf_out_str(stdout, 10, 3, relerr[0]); printf(", ");
	mpf_out_str(stdout, 10, 3, relerr[1]); printf(", ");
	mpf_out_str(stdout, 10, 3, relerr[2]); printf(", ");
	mpf_out_str(stdout, 10, 3, relerr[3]); printf(", ");
	mpf_out_str(stdout, 10, 3, relerr[4]); printf(", ");
	mpf_out_str(stdout, 10, 3, relerr[5]); printf(", ");
	mpf_out_str(stdout, 10, 3, relerr[6]); printf("\n");
	// ret_mpf = SolveMPFLSC(mpfx, mpfa, mpfb, row_ch, col_ch);

	min_dim = STRASSEN_MIN_DIM;
	etime[5] = 0.0;

	//limit_dim = min_dim * 20;
	limit_dim = dim;
	if(limit_dim > dim) limit_dim = dim;

//	for(block_size = min_dim; block_size <= min_dim * 4; block_size += min_dim)
//	for(block_size = min_dim; block_size <= min_dim * 10; block_size += min_dim)
//	for(block_size = min_dim; block_size <= min_dim * 15; block_size += min_dim)
	for(block_size = min_dim; block_size < limit_dim; block_size += min_dim)
	{
		//get_mpfproblem(mpfa, mpfb, mpfans);
		read_test_linear_eq_c(mpfa, mpfans, mpfb, (int)dim, fname_A, fname_true_x, fname_vec_b);
		//stime = get_secv();
		stime = get_real_secv();
#ifndef _OPENMP
	//	ret_mpf = MPFLUdecomp_strassen(mpfa, block_size);
		#ifdef USE_OZ
		ret_mpf = CMPFLUdecomp_ozPM(mpfa, pivot, block_size, max_num_div);
		printf("oz ");
		#else // USE_OZ
		ret_mpf = CMPFLUdecomp_strassenPM(mpfa, pivot, block_size);
		#endif // USE_OZ
		//ret_mpf = MPFLUdecomp_strassenPM(mpfa, pivot, block_size);
		printf("PM ");
	//	ret_mpf = MPFLUdecomp_strassen(mpfa, 64);
	//	ret_mpf = MPFLUdecomp_strassen(mpfa, 128);
	//	ret_mpf = MPFLUdecomp_strassen(mpfa, 16);
		//etime[5] = get_secv() - stime;
#else // _OPENMP
		//ret_mpf = MPFLUdecomp_strassen_omp(mpfa, block_size);
		ret_mpf = CMPFLUdecomp_strassenPM_omp(mpfa, pivot, block_size);
		printf("PM ");
#endif // _OPENMP
		etime[5] = get_real_secv() - stime;
		//ret_mpf = SolveMPFLS(mpfx, mpfa, mpfb);
		//ret_mpf = SolveMPFLSP(mpfx, mpfa, mpfb, pivot);
		ret_mpf = SolveCMPFLSPM(mpfx, mpfa, mpfb, pivot);
#ifdef USE_BLOCK 
		printf("blockLU(%5ld, %5ld, %5ld): %10.3f, ", dim, get_bnc_default_prec(), block_size, etime[5]);
#elif USE_STRASSEN
		printf("strassenLU(%5ld, %5ld, %5ld): %10.3f, ", dim, get_bnc_default_prec(), block_size, etime[5]);
#elif USE_WINOGRAD
		printf("winogradLU(%5ld, %5ld, %5ld): %10.3f, ", dim, get_bnc_default_prec(), block_size, etime[5]);
#elif USE_OZ
		printf("ozakiLU   (%5ld, %5ld, %5ld): %10.3f, ", dim, get_bnc_default_prec(), block_size, etime[5]);
#endif
		//relerr_element_cmpfvector(relerr[0], relerr[1], relerr[2], mpfx, mpfans, 0);
		relerr3_cmpfvector(relerr[0], relerr[1], relerr[2], relerr[3], relerr[4], relerr[5], relerr[6], mpfx, mpfans, 0);
		//printf("relerr(max, min, ||relerr||): ");
		mpf_out_str(stdout, 10, 3, relerr[0]); printf(", ");
		mpf_out_str(stdout, 10, 3, relerr[1]); printf(", ");
		mpf_out_str(stdout, 10, 3, relerr[2]); printf(", ");
		mpf_out_str(stdout, 10, 3, relerr[3]); printf(", ");
		mpf_out_str(stdout, 10, 3, relerr[4]); printf(", ");
		mpf_out_str(stdout, 10, 3, relerr[5]); printf(", ");
		mpf_out_str(stdout, 10, 3, relerr[6]); printf("\n");

		/*
		if(mpf_cmp_ui(relerr[0], 100UL) > 0)
		{
			print_mpfvector(mpfx);
			for(i = 0; i < mpfx->dim; i++) printf("%ld ", pivot[i]);
			printf("\n");
		}
		*/
	}
	// relerr_element_mpfvector(relerr[0], relerr[1], relerr[2], mpfx, mpfans, 0);
	// printf("relerr(max, min, ||relerr||): ");
	//mpf_out_str(stdout, 10, 3, relerr[0]); printf(", ");
	//mpf_out_str(stdout, 10, 3, relerr[1]); printf(", ");
	//mpf_out_str(stdout, 10, 3, relerr[2]); printf("\n");

	/* print */
	/* for(i = 0; i < dim; i++)
	{
		printf("%5ld ", i);
		mpf_out_str(stdout, 10, 0, get_mpfvector_i(mpfx, i));
		printf(" ");
		mpf_out_str(stdout, 10, 0, get_mpfvector_i(mpfans, i));
		printf("\n");
	}
*/
	/* end */
	mpf_clear(reps); mpf_clear(aeps);
	mpf_clear(relerr[0]); mpf_clear(relerr[1]); mpf_clear(relerr[2]);
	mpf_clear(relerr[3]); mpf_clear(relerr[4]); mpf_clear(relerr[5]);
	mpf_clear(relerr[6]);
	free_cmpfmatrix(mpfa);
	free_cmpfvector(mpfb);
	free_cmpfvector(mpfx);
	free_cmpfvector(mpfans);
	free(pivot);
#endif

	/* print itimes */
//end:
	return 0;

}

#endif // USE_GMP
