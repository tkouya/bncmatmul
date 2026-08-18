/********************************************************************************/
/* matvecbench_cmpf.c:                                                          */
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
int matvecloop_cmpf(long int start_dim, long int end_dim, long int dim_step, long int maxsec, long int opt_dim_flag, unsigned long prec, long int min_dim)
{
	long int dim;
	long int i, j, end_flag = 0;
	long int num_addsub, num_mul;
	double iteration, etime, stime, mflops;
	CMPFMatrix mat_a;
    CMPFVector vec_x, vec_y; // vec_y := mat_a * vec_x 
	CMPFVector vec_y_long;
	mpf_t relerr[7], tmp, tmp1;
    mpc_t ctmp;
#ifdef USE_MBLAS
	//mpcomplex::set_default_prec(prec);
	mpfr_set_default_prec((mp_prec_t)prec);
    mpfr::mpreal::default_prec = (mp_prec_t)prec;
    mpfr::mpcomplex::default_real_prec = (mp_prec_t)prec;
    mpfr::mpcomplex::default_imag_prec = (mp_prec_t)prec;
    //mpreal tmp;
	mpcomplex *mat_a_m, *vec_x_m, *vec_y_m;
    mpcomplex cone = mpcomplex(1.0, 0.0), czero = mpcomplex(0.0, 0.0);
#endif // USE_MBLAS

    mpc_init2(ctmp, prec);
	mpf_init2(tmp, prec);
	mpf_init2(tmp1, prec);
	mpf_init2(relerr[0], prec);
	mpf_init2(relerr[1], prec);
	mpf_init2(relerr[2], prec);
	mpf_init2(relerr[3], prec);
	mpf_init2(relerr[4], prec);
	mpf_init2(relerr[5], prec);
	mpf_init2(relerr[6], prec);

    mpf_srand(prec);
	for(dim = start_dim; dim <= end_dim; dim += dim_step)
	{
		mat_a = init2_cmpfmatrix(dim, dim, prec);
		vec_x = init2_cmpfvector(dim, prec);
	    vec_y = init2_cmpfvector(dim, prec);
		vec_y_long = init2_cmpfvector(dim, (unsigned long)(prec + prec / 2));
#ifdef USE_MBLAS
	// mpack
		mat_a_m = new mpcomplex[dim * dim];
		vec_x_m = new mpcomplex[dim];
	    vec_y_m = new mpcomplex[dim];
#endif // USE_MBLAS
		for(i = 0; i < dim; i++)
		{
			for(j = 0; j < dim; j++)
			{
				mpf_urand(tmp); // ru in [0, 1]
				mpf_sub_d(tmp, tmp, 0.5); // (ru - 0.5)
				mpf_nrand(tmp1); mpf_exp(tmp1, tmp1); // exp(rn)
				mpf_mul(tmp, tmp, tmp1);
                mpf_set(mpc_realref(ctmp), tmp);

				mpf_urand(tmp); // ru in [0, 1]
				mpf_sub_d(tmp, tmp, 0.5); // (ru - 0.5)
				mpf_nrand(tmp1); mpf_exp(tmp1, tmp1); // exp(rn)
				mpf_mul(tmp, tmp, tmp1);
                mpf_set(mpc_imagref(ctmp), tmp);
				//mpc_set_fr_fr(ctmp, tmp, tmp, get_bnc_default_rounding_mode());
				set_cmpfmatrix_ij(mat_a, i, j, ctmp); //.get_mpc_t());

				// mpack
#ifdef USE_MBLAS
				mat_a_m[COL_ZERO_IJ(i, j, dim, dim)] = ctmp; // (mpreal)tmp;
#endif // USE_MBLAS
            }
			// tmp := (ru - 0.5) * exp(rn)
			mpf_urand(tmp); // ru in [0, 1]
			mpf_sub_d(tmp, tmp, 0.5); // (ru - 0.5)
			mpf_nrand(tmp1); mpf_exp(tmp1, tmp1); // exp(rn)
			mpf_mul(tmp, tmp, tmp1);
            mpf_set(mpc_realref(ctmp), tmp);

			mpf_urand(tmp); // ru in [0, 1]
			mpf_sub_d(tmp, tmp, 0.5); // (ru - 0.5)
			mpf_nrand(tmp1); mpf_exp(tmp1, tmp1); // exp(rn)
			mpf_mul(tmp, tmp, tmp1);
            mpf_set(mpc_imagref(ctmp), tmp);
			set_cmpfvector_i(vec_x, i, ctmp); //.get_mpc_t());
			// mpack
#ifdef USE_MBLAS
			vec_x_m[i] = (mpcomplex)ctmp; //(mpreal)tmp;
#endif // USE_MBLAS
			//mpf_sqrt_ui(tmp, 2UL);
			//set_mpfvector_i(vec_y, i, tmp);

            //mpc_set_fr(ctmp, mpfr(czero.real()), )
            //set_cmpfvector_i(vec_y, i, czero.mpc);
			// mpack
#ifdef USE_MBLAS
			vec_y_m[i] = czero; //(mpreal)tmp;
#endif // USE_MBLAS
		}
        set0_cmpfvector(vec_y);

		iteration = 1.0;
		do{
			//stime = get_secv();
			num_addsub = mat_a->row_dim * vec_y->dim;
			num_mul = mat_a->row_dim * vec_y->dim;

			stime = get_real_secv();

#ifndef USE_MBLAS
			for(i = 0; i < iteration; i++)
			{
#ifdef USE_OMP
        		_bncomp_mul_cmpfmatrix_cmpfvec(vec_y, mat_a, vec_x);
#else // USE_OMP
            #ifdef USE_4M
				mul_cmpfmatrix_cmpfvec_4m(vec_y, mat_a, vec_x);
            #else // USE_4M
				mul_cmpfmatrix_cmpfvec(vec_y, mat_a, vec_x);
            #endif // USE_4M
#endif // USE_OMP
			}

#elif defined(USE_MBLAS)
			for(i = 0; i < iteration; i++) Cgemv("n", dim, dim, cone, mat_a_m, dim, vec_x_m, 1, czero, vec_y_m, 1);
#endif // USE_MBLAS
			//etime = get_secv() - stime;
			etime = get_real_secv() - stime;
			iteration *= 2.0;
		} while(etime < 2);
#ifdef USE_MBLAS
		for(i = 0; i < dim; i++)
		{
            mpc_set_fr_fr(ctmp, mpfr_ptr(vec_y_m[i].real()), mpfr_ptr(vec_y_m[i].imag()), get_bnc_default_rounding_mode());
			set_cmpfvector_i(vec_y, i, ctmp); //vec_y_m[i].mpc_get_t());
		}
#endif // USE_MBLAS

		// get relative errors
#ifdef USE_OMP
		_bncomp_mul_cmpfmatrix_cmpfvec(vec_y_long, mat_a, vec_x);
#else // USE_OMP
		mul_cmpfmatrix_cmpfvec(vec_y_long, mat_a, vec_x);
#endif // USE_OMP
//		mul_mpfmatrix_simple(dc_long, da, db);
		//relerr3_mpfvector(relerr[0], relerr[1], relerr[2], vec_y, vec_y_long, 0);
		relerr3_cmpfvector(relerr[0], relerr[1], relerr[2], relerr[3], relerr[4], relerr[5], relerr[6], vec_y, vec_y_long, 0);

		etime /= iteration / 2;

//		mflops = (double)dim * (double)dim * (double)dim * 2 / etime / 1000.0 / 1000.0;
		mflops = (double)(num_addsub + num_mul) / etime / 1000.0 / 1000.0;
		//printf("[%5d bits] %5d x %5d: %9.5f sec (%8.5f micro-sec/mul) %f MFlops", prec, dim, dim, etime, (etime / dim / dim / dim) * 1000 * 1000, mflops);
		//printf("[%5d bits] %5d x %5d: %9.5f sec (%8.5f micro-sec/mul) %f MFlops", prec, dim, dim, etime, (etime / (double)num_mul) * 1000 * 1000, mflops);
		//printf("[%5ld bits] %5ld x %5ld: %9.5f sec [mul]%ld [addsub]%ld", prec, dim, dim, etime, num_mul, num_addsub);
		printf("[%5ld bits] %5ld x %5ld: %10.3g sec ", prec, dim, dim, etime); //, num_mul, num_addsub);
		printf(" "); mpf_out_str(stdout, 10, 3, relerr[0]);
		printf(" "); mpf_out_str(stdout, 10, 3, relerr[1]);
		printf(" "); mpf_out_str(stdout, 10, 3, relerr[2]);
		printf(" "); mpf_out_str(stdout, 10, 3, relerr[3]);
		printf(" "); mpf_out_str(stdout, 10, 3, relerr[4]);
		printf(" "); mpf_out_str(stdout, 10, 3, relerr[5]);
		printf(" "); mpf_out_str(stdout, 10, 3, relerr[6]);
        //printf("\n");

		if(opt_dim_flag == 1)
			printf(" <-- Optimal dimension %+ld", dim - (end_dim + start_dim) / 2);
		printf("\n");

		free_cmpfmatrix(mat_a);
		free_cmpfvector(vec_x);
		free_cmpfvector(vec_y);
		free_cmpfvector(vec_y_long);

		// mpack
#ifdef USE_MBLAS
		delete[] mat_a_m;
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

    mpc_clear(ctmp);
	mpf_clear(tmp);
	mpf_clear(tmp1);
	mpf_clear(relerr[0]);
	mpf_clear(relerr[1]);
	mpf_clear(relerr[2]);
	mpf_clear(relerr[3]);
	mpf_clear(relerr[4]);
	mpf_clear(relerr[5]);
	mpf_clear(relerr[6]);

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

		end_flag = matvecloop_cmpf(idim - 1, idim + 1, 1, maxsec, 0, prec, min_dim);

//		end_flag = matmulloop_mpf(idim - 1, idim + 1, 1, maxsec, 0, prec);
		idim_half = 3 * idim / 2;

		if(idim_half >= min_dim)
			end_flag = matvecloop_cmpf(idim_half - 1, idim_half + 1, 1, maxsec, 0, prec, min_dim);

//		if((idim <= opt_dim) && (opt_dim <= idim * 2))
//			end_flag = matmulloop_mpf(opt_dim - 5, opt_dim + 5, 1, maxsec, 1, prec, min_dim);

		if(end_flag == 1)
			break;

	}
	printf("------------------------------------ END --------------------------------------\n");
}

