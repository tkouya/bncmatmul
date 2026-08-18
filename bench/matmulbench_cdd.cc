/********************************************************************************/
/* matmulbench_cdd.cc:                                                          */
/* Copyright (C) 2023-2026 Tomonori Kouya                                       */
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
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "qd/qd_real.h"

using namespace std;

// mplapack, mpblas
#ifdef USE_MBLAS
#ifdef OLD_MBLAS
    #include <mblas_dd.h>
#else // OLD_MBLAS
    #include <mpblas_dd.h>
#endif // OLD_MBLAS
#endif // USE_MBLAS

#include "matmul_strassen.h"
#include "get_secv.h"

#ifdef USE_OMP
	#include "bncomp.h"
#endif // USE_OMP

// row-major, zero index
#define ROW_ZERO_IJ(i, j, row_dim, col_dim)	((i) * (col_dim) + (j))
// column-major, zero index
#define COL_ZERO_IJ(i, j, row_dim, col_dim)	((i) + (j) * (row_dim))

void usage(const char *commandname)
{
#ifdef USE_OMP
	printf("USAGE-> %s [num_threads] [Maximum seconds]\n", commandname);
#else // USE_OMP
	printf("USAGE-> %s [Maximum seconds]\n", commandname);
#endif // USE_OMP
}

// C := A * B
static void matmul_cddmatrix(CDDMatrix mat_c, CDDMatrix mat_a, CDDMatrix mat_b)
{
#if defined(USE_OMP)
	_bncomp_mul_cddmatrix(mat_c, mat_a, mat_b);
#else // serial
	#ifdef USE_STRASSEN
	mul_cddmatrix_strassen_4m(mat_c, mat_a, mat_b, _BNC_DEFAULT_MIN_DIM_STRASSEN);
	#elif defined(USE_BLOCK)
	mul_cddmatrix_block_4m(mat_c, mat_a, mat_b, _BNC_DEFAULT_MIN_DIM_STRASSEN);
	#else
	mul_cddmatrix(mat_c, mat_a, mat_b);
	#endif
#endif
}

int matmulloop_cdd(long int start_dim, long int end_dim, long int dim_step, long int maxsec, long int opt_dim_flag)
{
	long int dim;
	long int i, j, end_flag = 0;
	long int num_addsub, num_mul, iteration;
	double etime, stime, mflops;
	CDDMatrix mat_a, mat_b, mat_c, mat_c_long;
	dd_real tmp;
	cddfloat ctmp;
	dd_real relerr[7];
#ifdef USE_MBLAS
	dd_complex one = (dd_real)(1.0), zero = (dd_real)(0.0);
	dd_complex *mat_a_m, *mat_b_m, *mat_c_m;
#endif // USE_MBLAS

	for(dim = start_dim; dim <= end_dim; dim += dim_step)
	{
		mat_a = init_cddmatrix(dim, dim);
		mat_b = init_cddmatrix(dim, dim);
		mat_c = init_cddmatrix(dim, dim);
		mat_c_long = init_cddmatrix(dim, dim);

#ifdef USE_MBLAS
		mat_a_m = new dd_complex[dim * dim];
		mat_b_m = new dd_complex[dim * dim];
		mat_c_m = new dd_complex[dim * dim];
#endif // USE_MBLAS

		for(i = 0; i < dim; i++)
		{
			for(j = 0; j < dim; j++)
			{
				tmp = sqrt((dd_real)5.0) * (dd_real)(double)(i + j + 1);
				rdd_set(ctmp.val_re, tmp.x);
				tmp = sqrt((dd_real)2.0) * (dd_real)(double)(i + 1);
				rdd_set(ctmp.val_im, tmp.x);
				set_cddmatrix_ij(mat_a, i, j, &ctmp);
#ifdef USE_MBLAS
				{ dd_complex cv; rdd_set(cv.real().x, ctmp.val_re); rdd_set(cv.imag().x, ctmp.val_im); mat_a_m[COL_ZERO_IJ(i, j, dim, dim)] = cv; }
#endif // USE_MBLAS

				tmp = sqrt((dd_real)3.0) * (dd_real)(double)(2 * dim - (i + j));
				rdd_set(ctmp.val_re, tmp.x);
				tmp = sqrt((dd_real)7.0) * (dd_real)(double)(j + 1);
				rdd_set(ctmp.val_im, tmp.x);
				set_cddmatrix_ij(mat_b, i, j, &ctmp);
#ifdef USE_MBLAS
				{ dd_complex cv; rdd_set(cv.real().x, ctmp.val_re); rdd_set(cv.imag().x, ctmp.val_im); mat_b_m[COL_ZERO_IJ(i, j, dim, dim)] = cv; }
#endif // USE_MBLAS
			}
		}

		iteration = 1;
		do{
			stime = get_real_secv();

			num_addsub = mat_a->re->row_dim * mat_b->re->col_dim * mat_a->re->col_dim;
			num_mul    = mat_a->re->row_dim * mat_b->re->col_dim * mat_a->re->col_dim;
#ifdef USE_MBLAS
			for(i = 0; i < iteration; i++) Cgemm("n", "n", dim, dim, dim, one, mat_a_m, dim, mat_b_m, dim, zero, mat_c_m, dim);
#else // USE_MBLAS
			for(i = 0; i < iteration; i++) matmul_cddmatrix(mat_c, mat_a, mat_b);
#endif // USE_MBLAS

			etime = get_real_secv() - stime;
			iteration *= 2;
		} while(etime < 2);

#ifdef USE_MBLAS
		for(i = 0; i < dim; i++)
			for(j = 0; j < dim; j++)
			{
				dd_complex cv = mat_c_m[COL_ZERO_IJ(i, j, dim, dim)];
				rdd_set(ctmp.val_re, cv.real().x);
				rdd_set(ctmp.val_im, cv.imag().x);
				set_cddmatrix_ij(mat_c, i, j, &ctmp);
			}
#endif // USE_MBLAS

	// get relative errors (reference := native multiplication)
		mul_cddmatrix(mat_c_long, mat_a, mat_b);
		relerr3_cddmatrix(relerr[0].x, relerr[1].x, relerr[2].x, relerr[3].x, relerr[4].x, relerr[5].x, relerr[6].x, mat_c, mat_c_long, 0);

		etime /= iteration / 2;

		mflops = (double)(num_addsub + num_mul) / etime / 1000.0 / 1000.0;
		printf("[%5ld bits] %5ld x %5ld: %9.5f sec [mul]%ld [addsub]%ld %10.3f MFlops", (long int)106, dim, dim, etime, num_mul, num_addsub, mflops);
		for(i = 0; i < 7; i++) { printf(" "); printf("%10.3e", to_double(relerr[i])); }

		if(opt_dim_flag == 1)
			printf(" <-- Optimal dimension %+ld", dim - (end_dim + start_dim) / 2);
		printf("\n");

		free_cddmatrix(mat_a);
		free_cddmatrix(mat_b);
		free_cddmatrix(mat_c);
		free_cddmatrix(mat_c_long);

#ifdef USE_MBLAS
		delete[] mat_a_m;
		delete[] mat_b_m;
		delete[] mat_c_m;
#endif // USE_MBLAS

		// exceed max. comp. time
		if(etime > maxsec)
		{
			end_flag = 1;
			break;
		}
	}

	return end_flag;
}

int main(int argc, char *argv[])
{
	int num_threads = 1;
	long int idim, idim_half, end_flag, min_dim = _BNC_DEFAULT_MIN_DIM_STRASSEN;
	long int maxsec = 1200;

	if(argc < 1)
	{
		usage(argv[0]);
		return 0;
	}
	else if(argc >= 2)
	{
#ifdef USE_OMP
		num_threads = atoi(argv[1]);
		if(num_threads < 1)
			printf("num_threads is 1\n");
		if(argc >= 3)
		{
			maxsec = atol(argv[2]);
			if(maxsec < 0) { printf("Invalid Maximum seconds! (%ld seconds)\n", maxsec); return -1; }
		}
#else // USE_OMP
		maxsec = atol(argv[1]);
		if(maxsec < 0) { printf("Invalid Maximum seconds! (%ld seconds)\n", maxsec); return -1; }
#endif // USE_OMP
	}

	printf("------------------ Benchmark Test using Matrix Multiplication -----------------\n");

	// Initialize QD library
	fpu_fix_start(NULL);
	printf("Maxsec: %ld\n", maxsec);

#ifdef USE_OMP
	set_bncomp_num_threads(num_threads);
	printf("OpenMP #threads = %d\n", omp_get_num_threads());
#endif // USE_OMP

	bnc_print_env_all();

	for(idim = min_dim; idim < 32768; idim *= 2)
	{
		end_flag = 0;

		end_flag = matmulloop_cdd(idim - 1, idim + 1, 1, maxsec, 0);

		idim_half = 3 * idim / 2;
		if(idim_half >= min_dim)
			end_flag = matmulloop_cdd(idim_half - 1, idim_half + 1, 1, maxsec, 0);

		if(end_flag == 1)
			break;
	}
	printf("------------------------------------ END --------------------------------------\n");
}
