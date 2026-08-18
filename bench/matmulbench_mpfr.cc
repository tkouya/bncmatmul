/********************************************************************************/
/* matmulbench_mpfr.cc:                                                         */
/* Copyright (C) 2021-2026 Tomonori Kouya                                       */
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

using namespace std;

// mplapack, mpblas
#ifdef USE_MBLAS
#ifdef OLD_MBLAS
    #include <mblas_mpfr.h>
#else // OLD_MBLAS
    #include <mpblas_mpfr.h>
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
	printf("USAGE-> %s [prec(bits)] [num_threads] [Maximum seconds]\n", commandname);
#else // USE_OMP
	printf("USAGE-> %s [prec(bits)] [Maximum seconds]\n", commandname);
#endif // USE_OMP
}

#ifdef USE_GMP

// C := A * B
static void matmul_mpfmatrix(MPFMatrix mat_c, MPFMatrix mat_a, MPFMatrix mat_b)
{
#if defined(USE_OMP)
	#ifdef USE_STRASSEN
	_bncomp_mul_mpfmatrix_strassen(mat_c, mat_a, mat_b, _BNC_DEFAULT_MIN_DIM_STRASSEN);
	#elif defined(USE_BLOCK)
	_bncomp_mul_mpfmatrix_block(mat_c, mat_a, mat_b, _BNC_DEFAULT_MIN_DIM_STRASSEN);
	#else
	_bncomp_mul_mpfmatrix(mat_c, mat_a, mat_b);
	#endif
#else // serial
	#ifdef USE_STRASSEN
	mul_mpfmatrix_strassen(mat_c, mat_a, mat_b, _BNC_DEFAULT_MIN_DIM_STRASSEN);
	#elif defined(USE_BLOCK)
	mul_mpfmatrix_block(mat_c, mat_a, mat_b, _BNC_DEFAULT_MIN_DIM_STRASSEN);
	#else
	mul_mpfmatrix(mat_c, mat_a, mat_b);
	#endif
#endif
}

int matmulloop_mpf(long int start_dim, long int end_dim, long int dim_step, long int maxsec, long int opt_dim_flag, unsigned long prec, long int min_dim)
{
	long int dim;
	long int i, j, end_flag = 0;
	long int num_addsub, num_mul;
	double iteration, etime, stime, mflops;
	MPFMatrix mat_a, mat_b, mat_c, mat_c_long;
	mpf_t tmp, relerr[3];
#ifdef USE_MBLAS
	mpfr_set_default_prec((mp_prec_t)prec);
	mpfr::mpreal::default_prec = (mp_prec_t)prec;
	mpreal *mat_a_m, *mat_b_m, *mat_c_m;
#endif // USE_MBLAS

	mpf_init2(tmp, prec);
	mpf_init2(relerr[0], prec);
	mpf_init2(relerr[1], prec);
	mpf_init2(relerr[2], prec);

	for(dim = start_dim; dim <= end_dim; dim += dim_step)
	{
		mat_a = init2_mpfmatrix(dim, dim, prec);
		mat_b = init2_mpfmatrix(dim, dim, prec);
		mat_c = init2_mpfmatrix(dim, dim, prec);
		mat_c_long = init2_mpfmatrix(dim, dim, (unsigned long)(prec + prec / 2));

#ifdef USE_MBLAS
		mat_a_m = new mpreal[dim * dim];
		mat_b_m = new mpreal[dim * dim];
		mat_c_m = new mpreal[dim * dim];
#endif // USE_MBLAS

		for(i = 0; i < dim; i++)
		{
			for(j = 0; j < dim; j++)
			{
				mpf_sqrt_ui(tmp, 5UL);
				mpf_mul_ui(tmp, tmp, (unsigned long)(i + j + 1));
				set_mpfmatrix_ij(mat_a, i, j, tmp);
#ifdef USE_MBLAS
				mat_a_m[COL_ZERO_IJ(i, j, dim, dim)] = (mpreal)tmp;
#endif // USE_MBLAS

				mpf_sqrt_ui(tmp, 3UL);
				mpf_mul_ui(tmp, tmp, (unsigned long)(2 * dim - (i + j)));
				set_mpfmatrix_ij(mat_b, i, j, tmp);
#ifdef USE_MBLAS
				mat_b_m[COL_ZERO_IJ(i, j, dim, dim)] = (mpreal)tmp;
#endif // USE_MBLAS
			}
		}

		iteration = 1.0;
		do{
			num_addsub = mat_a->row_dim * mat_b->col_dim * mat_a->col_dim;
			num_mul    = mat_a->row_dim * mat_b->col_dim * mat_a->col_dim;

			stime = get_real_secv();

#ifndef USE_MBLAS
			for(i = 0; i < iteration; i++) matmul_mpfmatrix(mat_c, mat_a, mat_b);
#elif defined(USE_MBLAS)
			for(i = 0; i < iteration; i++) Rgemm("n", "n", dim, dim, dim, 1.0, mat_a_m, dim, mat_b_m, dim, 0.0, mat_c_m, dim);
#endif // USE_MBLAS

			etime = get_real_secv() - stime;
			iteration *= 2.0;
		} while(etime < 2);

#ifdef USE_MBLAS
		for(i = 0; i < dim; i++)
			for(j = 0; j < dim; j++)
				set_mpfmatrix_ij(mat_c, i, j, mpfr_ptr(mat_c_m[COL_ZERO_IJ(i, j, dim, dim)]));
#endif // USE_MBLAS

	// get relative errors (reference := native multiplication)
		mul_mpfmatrix(mat_c_long, mat_a, mat_b);
		relerr3_mpfmatrix(relerr[0], relerr[1], relerr[2], mat_c, mat_c_long, 0);

		etime /= iteration / 2;

		mflops = (double)(num_addsub + num_mul) / etime / 1000.0 / 1000.0;
		printf("[%5ld bits] %5ld x %5ld: %9.5f sec [mul]%ld [addsub]%ld %10.3f MFlops", (long int)prec, dim, dim, etime, num_mul, num_addsub, mflops);
		printf(" "); mpf_out_str(stdout, 10, 3, relerr[0]);
		printf(" "); mpf_out_str(stdout, 10, 3, relerr[1]);
		printf(" "); mpf_out_str(stdout, 10, 3, relerr[2]);

		if(opt_dim_flag == 1)
			printf(" <-- Optimal dimension %+ld", dim - (end_dim + start_dim) / 2);
		printf("\n");

		free_mpfmatrix(mat_a);
		free_mpfmatrix(mat_b);
		free_mpfmatrix(mat_c);
		free_mpfmatrix(mat_c_long);

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

	mpf_clear(tmp);
	mpf_clear(relerr[0]);
	mpf_clear(relerr[1]);
	mpf_clear(relerr[2]);

	return end_flag;
}
#endif // USE_GMP

int main(int argc, char *argv[])
{
	int num_threads = 1;
	long int idim, idim_half, end_flag, min_dim = _BNC_DEFAULT_MIN_DIM_STRASSEN;
	long int maxsec = 2000;
	unsigned long prec;

	if(argc <= 1)
	{
		usage(argv[0]);
		return 0;
	}
	else
	{
		prec = atol(argv[1]);
		if(prec <= 0) { printf("Invalid precision in bit! (%ld bits)\n", prec); return -1; }
#ifdef USE_OMP
		if(argc >= 3)
		{
			num_threads = atoi(argv[2]);
			if(num_threads < 1) printf("num_threads is 1\n");
		}
		if(argc >= 4)
		{
			maxsec = atol(argv[3]);
			if(maxsec < 0) { printf("Invalid Maximum seconds! (%ld seconds)\n", maxsec); return -1; }
		}
#else // USE_OMP
		if(argc >= 3)
		{
			maxsec = atol(argv[2]);
			if(maxsec < 0) { printf("Invalid Maximum seconds! (%ld seconds)\n", maxsec); return -1; }
		}
#endif // USE_OMP
	}

	set_bnc_default_prec(prec);
	printf("Maxsec: %ld\n", maxsec);

#ifdef USE_OMP
	set_bncomp_num_threads(num_threads);
	printf("OpenMP #threads = %d\n", omp_get_max_threads());
#endif // USE_OMP

	bnc_print_env_all();
	printf("------------------ Benchmark Test using Matrix Multiplication -----------------\n");

#ifdef USE_GMP
	for(idim = min_dim; idim < 32768; idim *= 2)
	{
		end_flag = 0;

		end_flag = matmulloop_mpf(idim - 1, idim + 1, 1, maxsec, 0, prec, min_dim);

		idim_half = 3 * idim / 2;
		if(idim_half >= min_dim)
			end_flag = matmulloop_mpf(idim_half - 1, idim_half + 1, 1, maxsec, 0, prec, min_dim);

		if(end_flag == 1)
			break;
	}
#endif // USE_GMP
	printf("------------------------------------ END --------------------------------------\n");
}
