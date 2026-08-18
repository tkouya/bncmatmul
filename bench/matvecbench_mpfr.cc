/********************************************************************************/
/* matvecbench_mpf.c:                                                           */
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
#include <cstdio>
#include <cstdlib>
#include <cmath>

// mpblas
#ifdef OLD_MBLAS
    #include <mblas_mpfr.h>
#else
	#include <mpblas_mpfr.h>
#endif // USE_MBLAS

#include "matmul_strassen.h"
#include "get_secv.h"
#ifdef USE_OMP
	#include "bncomp.h"
#endif


// row-major, zero index
#define RAW_ZERO_IJ(i, j, row_dim, col_dim)	((i) * (col_dim) + (j))
// column-major, zero index
#define COL_ZERO_IJ(i, j, row_dim, col_dim)	((i) + (j) * (row_dim))

// 256k bytes
//#define CACHESIZE (256 * 1024)

// 512k bytes
//#define CACHESIZE (512 * 1024)

// 1M bytes
//#define CACHESIZE (1 * 1024 * 1024)

// 2M bytes
//#define CACHESIZE (2 * 1024 * 1024)

void usage(const char *commandname)
{
	#ifdef USE_OMP
		printf("USAGE-> %s precision(in bits) [num_threads] [Maximum seconds]\n", commandname);
	#else // USE_OMP
		printf("USAGE-> %s precision(in bits) [Maximum seconds]\n", commandname);
	#endif // USE_OMP
}

#ifdef USE_GMP

//int matmulloop_mpf(long int start_dim, long int end_dim, long int dim_step, long int maxsec, long int opt_dim_flag, unsigned long prec)
int matvecloop_mpf(long int start_dim, long int end_dim, long int dim_step, long int maxsec, long int opt_dim_flag, unsigned long prec, long int min_dim)
{
	long int dim;
	long int i, j, end_flag = 0;
	long int num_addsub, num_mul;
	double iteration, etime, stime, mflops;
	MPFMatrix da;
    MPFVector vec_x, vec_y; // vec_y := mat_a * vec_x 
	MPFVector vec_y_long;
	mpf_t tmp, relerr[3];
#ifdef USE_MBLAS
	//mpreal::set_default_prec(prec);
	mpfr_set_default_prec((mp_prec_t)prec);
    mpfr::mpreal::default_prec = (mp_prec_t)prec;
	mpreal *da_m, *vec_x_m, *vec_y_m;
#endif // USE_MBLAS

	mpf_init2(tmp, prec);
	mpf_init2(relerr[0], prec);
	mpf_init2(relerr[1], prec);
	mpf_init2(relerr[2], prec);

	for(dim = start_dim; dim <= end_dim; dim += dim_step)
	{
		da = init2_mpfmatrix(dim, dim, prec);
		vec_x = init2_mpfvector(dim, prec);
	    vec_y = init2_mpfvector(dim, prec);
		vec_y_long = init2_mpfvector(dim, (unsigned long)(prec + prec / 2));
#ifdef USE_MBLAS
	// mpack
		da_m = new mpreal[dim * dim];
		vec_x_m = new mpreal[dim];
	    vec_y_m = new mpreal[dim];
#endif // USE_MBLAS
		for(i = 0; i < dim; i++)
		{
			for(j = 0; j < dim; j++)
			{
//				set_mpfmatrix_ij(da, i, j, (i + j + 1) * sqrt(5.0));
				mpf_sqrt_ui(tmp, 5UL);
				mpf_mul_ui(tmp, tmp, (unsigned long)(i + j + 1));
				set_mpfmatrix_ij(da, i, j, tmp);
				// mpack
#ifdef USE_MBLAS
				da_m[COL_ZERO_IJ(i, j, dim, dim)] = (mpreal)tmp;
#endif // USE_MBLAS
            }
			mpf_sqrt_ui(tmp, 3UL);
			mpf_mul_ui(tmp, tmp, (unsigned long)(dim - i));
			set_mpfvector_i(vec_x, i, tmp);
			// mpack
#ifdef USE_MBLAS
			vec_x_m[i] = (mpreal)tmp;
#endif // USE_MBLAS
			mpf_sqrt_ui(tmp, 2UL);
			set_mpfvector_i(vec_y, i, tmp);
			// mpack
#ifdef USE_MBLAS
			vec_y_m[i] = (mpreal)tmp;
#endif // USE_MBLAS
		}

		iteration = 1.0;
		do{
			//stime = get_secv();
			num_addsub = da->row_dim * vec_y->dim;
			num_mul = da->row_dim * vec_y->dim;

			stime = get_real_secv();

#ifndef USE_MBLAS
			for(i = 0; i < iteration; i++)
			{
#ifdef USE_OMP
        		_bncomp_mul_mpfmatrix_mpfvec(vec_y, da, vec_x);
#else // USE_OMP
				mul_mpfmatrix_mpfvec(vec_y, da, vec_x);
#endif // USE_OMP
			}

#elif defined(USE_MBLAS)
			for(i = 0; i < iteration; i++) Rgemv("n", dim, dim, 1.0, da_m, dim, vec_x_m, 1, 0.0, vec_y_m, 1);
#endif // USE_MBLAS
			//etime = get_secv() - stime;
			etime = get_real_secv() - stime;
			iteration *= 2.0;
		} while(etime < 2);
#ifdef USE_MBLAS
		for(i = 0; i < dim; i++)
		{
			set_mpfvector_i(vec_y, i, mpfr_ptr(vec_y_m[i]));
		}
#endif // USE_MBLAS

		// get relative errors
#ifdef USE_OMP
		_bncomp_mul_mpfmatrix_mpfvec(vec_y_long, da, vec_x);
#else // USE_OMP
		mul_mpfmatrix_mpfvec(vec_y_long, da, vec_x);
#endif // USE_OMP
//		mul_mpfmatrix_simple(dc_long, da, db);
		relerr3_mpfvector(relerr[0], relerr[1], relerr[2], vec_y, vec_y_long, 0);

		etime /= iteration / 2;

//		mflops = (double)dim * (double)dim * (double)dim * 2 / etime / 1000.0 / 1000.0;
		mflops = (double)(num_addsub + num_mul) / etime / 1000.0 / 1000.0;
		//printf("[%5d bits] %5d x %5d: %9.5f sec (%8.5f micro-sec/mul) %f MFlops", prec, dim, dim, etime, (etime / dim / dim / dim) * 1000 * 1000, mflops);
		//printf("[%5d bits] %5d x %5d: %9.5f sec (%8.5f micro-sec/mul) %f MFlops", prec, dim, dim, etime, (etime / (double)num_mul) * 1000 * 1000, mflops);
		printf("[%5ld bits] %5ld x %5ld: %9.5f sec [mul]%ld [addsub]%ld", prec, dim, dim, etime, num_mul, num_addsub);
		printf(" "); mpf_out_str(stdout, 10, 3, relerr[0]);
		printf(" "); mpf_out_str(stdout, 10, 3, relerr[1]);
		printf(" "); mpf_out_str(stdout, 10, 3, relerr[2]);// printf("\n");

		if(opt_dim_flag == 1)
			printf(" <-- Optimal dimension %+ld", dim - (end_dim + start_dim) / 2);
		printf("\n");

		free_mpfmatrix(da);
		free_mpfvector(vec_x);
		free_mpfvector(vec_y);
		free_mpfvector(vec_y_long);

		// mpack
#ifdef USE_MBLAS
		delete[] da_m;
		delete[] vec_x_m;
		delete[] vec_y_m;
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
	long int dim, idim, idim_half, opt_dim, end_flag, min_dim = _BNC_DEFAULT_MIN_DIM_STRASSEN;
	long int cachesize = 128, maxsec = 2000;
	unsigned long prec;
	if(argc <= 1)
	{
		usage(argv[0]);
		return 0;
	}
	else if(argc >= 2)
	{
		prec = atol(argv[1]);
		if(prec <= 0)
		{
			printf("Invalid precision in bit! (%ld bits)\n", prec);
			return -1;
		}
#ifdef USE_OMP
		if(argc >= 3)
		{
			num_threads = atoi(argv[2]);
			if(num_threads < 1)
			{
				printf("num_threads is 1\n");
			}
		}
		if(argc >= 4)
		{
			maxsec = atol(argv[3]);
			if(maxsec < 0)
			{
				printf("Invalid Maximum seconds! (%ld seconds)\n", maxsec);
				return -1;
			}
		}
#else // USE_OMP
		if(argc >= 3)
		{
			maxsec = atol(argv[2]);
			if(maxsec < 0)
			{
				printf("Invalid Maximum seconds! (%ld seconds)\n", maxsec);
				return -1;
			}
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

	//min_dim = 2048;

	//for(idim = 32; idim <= 16384; idim *= 2)
	//for(idim = 2; idim <= 2048; idim *= 2)
	for(idim = min_dim; idim < 32768; idim *= 2)
	{
		end_flag = 0;

		end_flag = matvecloop_mpf(idim - 1, idim + 1, 1, maxsec, 0, prec, min_dim);

//		end_flag = matmulloop_mpf(idim - 1, idim + 1, 1, maxsec, 0, prec);
		idim_half = 3 * idim / 2;

		if(idim_half >= min_dim)
			end_flag = matvecloop_mpf(idim_half - 1, idim_half + 1, 1, maxsec, 0, prec, min_dim);

//		if((idim <= opt_dim) && (opt_dim <= idim * 2))
//			end_flag = matmulloop_mpf(opt_dim - 5, opt_dim + 5, 1, maxsec, 1, prec, min_dim);

		if(end_flag == 1)
			break;

	}
	printf("------------------------------------ END --------------------------------------\n");
}

