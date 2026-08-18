/********************************************************************************/
/* matmulbench_ctd.cc:                                                          */
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

#include "matmul_strassen.h"
#include "get_secv.h"

#ifdef USE_OMP
	#include "bncomp.h"
#endif // USE_OMP

void usage(const char *commandname)
{
#ifdef USE_OMP
	printf("USAGE-> %s [num_threads] [Maximum seconds]\n", commandname);
#else // USE_OMP
	printf("USAGE-> %s [Maximum seconds]\n", commandname);
#endif // USE_OMP
}

// C := A * B
static void matmul_ctdmatrix(CTDMatrix mat_c, CTDMatrix mat_a, CTDMatrix mat_b)
{
#if defined(USE_OMP)
	_bncomp_mul_ctdmatrix(mat_c, mat_a, mat_b);
#else // serial
	#ifdef USE_STRASSEN
	mul_ctdmatrix_strassen_4m(mat_c, mat_a, mat_b, _BNC_DEFAULT_MIN_DIM_STRASSEN);
	#elif defined(USE_BLOCK)
	mul_ctdmatrix_block_4m(mat_c, mat_a, mat_b, _BNC_DEFAULT_MIN_DIM_STRASSEN);
	#else
	mul_ctdmatrix(mat_c, mat_a, mat_b);
	#endif
#endif
}

int matmulloop_ctd(long int start_dim, long int end_dim, long int dim_step, long int maxsec, long int opt_dim_flag)
{
	long int dim;
	long int i, j, end_flag = 0;
	long int num_addsub, num_mul, iteration;
	double etime, stime, mflops;
	CTDMatrix mat_a, mat_b, mat_c, mat_c_long;
	ctdfloat ctmp;
	double tmp[TDSIZE], relerr[7][TDSIZE];

	for(dim = start_dim; dim <= end_dim; dim += dim_step)
	{
		mat_a = init_ctdmatrix(dim, dim);
		mat_b = init_ctdmatrix(dim, dim);
		mat_c = init_ctdmatrix(dim, dim);
		mat_c_long = init_ctdmatrix(dim, dim);

		for(i = 0; i < dim; i++)
		{
			for(j = 0; j < dim; j++)
			{
				rtd_set_ui(tmp, 5UL); rtd_sqrt(tmp, tmp); rtd_mul_ui(tmp, tmp, (unsigned long)(i + j + 1));
				rtd_set(ctmp.val_re, tmp);
				rtd_set_ui(tmp, 2UL); rtd_sqrt(tmp, tmp); rtd_mul_ui(tmp, tmp, (unsigned long)(i + 1));
				rtd_set(ctmp.val_im, tmp);
				set_ctdmatrix_ij(mat_a, i, j, &ctmp);

				rtd_set_ui(tmp, 3UL); rtd_sqrt(tmp, tmp); rtd_mul_ui(tmp, tmp, (unsigned long)(2 * dim - (i + j)));
				rtd_set(ctmp.val_re, tmp);
				rtd_set_ui(tmp, 7UL); rtd_sqrt(tmp, tmp); rtd_mul_ui(tmp, tmp, (unsigned long)(j + 1));
				rtd_set(ctmp.val_im, tmp);
				set_ctdmatrix_ij(mat_b, i, j, &ctmp);
			}
		}

		iteration = 1;
		do{
			stime = get_real_secv();

			num_addsub = mat_a->re->row_dim * mat_b->re->col_dim * mat_a->re->col_dim;
			num_mul    = mat_a->re->row_dim * mat_b->re->col_dim * mat_a->re->col_dim;

			for(i = 0; i < iteration; i++) matmul_ctdmatrix(mat_c, mat_a, mat_b);

			etime = get_real_secv() - stime;
			iteration *= 2;
		} while(etime < 2);

	// get relative errors (reference := native multiplication)
		mul_ctdmatrix(mat_c_long, mat_a, mat_b);
		relerr3_ctdmatrix(relerr[0], relerr[1], relerr[2], relerr[3], relerr[4], relerr[5], relerr[6], mat_c, mat_c_long, 0);

		etime /= iteration / 2;

		mflops = (double)(num_addsub + num_mul) / etime / 1000.0 / 1000.0;
		printf("[%5ld bits] %5ld x %5ld: %9.5f sec [mul]%ld [addsub]%ld %10.3f MFlops", (long int)159, dim, dim, etime, num_mul, num_addsub, mflops);
		for(i = 0; i < 7; i++) { printf(" "); printf("%10.3e", relerr[i][0]); }

		if(opt_dim_flag == 1)
			printf(" <-- Optimal dimension %+ld", dim - (end_dim + start_dim) / 2);
		printf("\n");

		free_ctdmatrix(mat_a);
		free_ctdmatrix(mat_b);
		free_ctdmatrix(mat_c);
		free_ctdmatrix(mat_c_long);

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

		end_flag = matmulloop_ctd(idim - 1, idim + 1, 1, maxsec, 0);

		idim_half = 3 * idim / 2;
		if(idim_half >= min_dim)
			end_flag = matmulloop_ctd(idim_half - 1, idim_half + 1, 1, maxsec, 0);

		if(end_flag == 1)
			break;
	}
	printf("------------------------------------ END --------------------------------------\n");
}
