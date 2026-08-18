/********************************************************************************/
/* matvecbench_ctd.cc:                                                          */
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
#include <iostream>
#include <iomanip>
#include <cstdio>
#include <cstdlib>
#include <cmath>

using namespace std;

//#include "bnc.h"
#include <mpblas_mpfr.h>
#include "matmul_strassen.h"
#include "get_secv.h"

#ifdef USE_OMP
	#include "bncomp.h"
#endif // USE_OMP

// row-major, zero index
#define ROW_ZERO_IJ(i, j, row_dim, col_dim)	((i) * (col_dim) + (j))
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
	printf("USAGE-> %s [num_threads] [Maximum seconds]\n", commandname);
#else // USE_OMP
	printf("USAGE-> %s [Maximum seconds]\n", commandname);
#endif // USE_OMP
}

//#ifdef USE_DDLINEAR

int matvecloop_ctd(long int start_dim, long int end_dim, long int dim_step, long int maxsec, long int opt_dim_flag)
{
	long int dim;
	long int i, j, end_flag = 0;
	long int num_addsub, num_mul, iteration;
	double etime, stime, mflops;
	CTDMatrix mat_a;
    CTDVector vec_x, vec_y; // vec_y := mat_a * vec_x 
	CTDVector vec_y_long;
	tdfloat tmp, tmp1, relerr[7];
    ctdfloat ctmp, one, zero;
	double *ddtmp, *ddtmp2;
#ifdef USE_OZ
	int max_num_div; // For Ozaki scheme
	#ifdef OZ_MAX_NUM_DIV
	max_num_div = OZ_MAX_NUM_DIV;
	#else // OZ_MAX_NUM_DIV
	max_num_div = 10;
	fprintf(stderr, "Warning: no definition of OZ_MAX_NUM_DIV! -> max_num_div = %d\n", max_num_div);
	#endif // OZ_MAX_NUM_DIV
#endif // USE_OZ

    zero.val_re[0] = 0.0; zero.val_re[1] = 0.0; zero.val_re[2] = 0.0;
    zero.val_im[0] = 0.0; zero.val_im[1] = 0.0; zero.val_im[2] = 0.0;

    one.val_re[0] = 1.0; one.val_re[1] = 0.0; one.val_re[2] = 0.0;
    one.val_im[0] = 0.0; one.val_im[1] = 0.0; one.val_im[2] = 0.0;

	//std::cout << "one = " << std::setprecision(32) << one << ", zero = " << zero << std::endl;

	for(dim = start_dim; dim <= end_dim; dim += dim_step)
	{
		mat_a = init_ctdmatrix(dim, dim);
	    vec_x = init_ctdvector(dim);
		vec_y = init_ctdvector(dim);
		vec_y_long = init_ctdvector(dim);

	    rtd_srand((unsigned long)(dim * 159));
		for(i = 0; i < dim; i++)
		{
			for(j = 0; j < dim; j++)
			{
//				set_ddmatrix_ij(mat_a, i, j, (i + j + 1) * sqrt(5.0));
//				rtd_set_ui(tmp.val, 5UL);
//				rtd_sqrt(tmp.val, tmp.val);
				//tmp = sqrt((qd_real)5.0) * (qd_real)(double)(i + j + 1);
				//tmp = sqrt((qd_real)rand()) * (qd_real)(double)(i + j + 1);
				//rtd_sqrt_ui(tmp.val, 5UL);
				//cout << "tmp = " << tmp << endl;
				//rtd_mul_ui(tmp.val, tmp.val, (unsigned long)(i + j + 1));
				//set_ddmatrix_ij(mat_a, i, j, tmp.val);
				// tmp := (ru - 0.5) * exp(rn)
				rtd_urand(tmp.val); // ru in [0, 1]
				rtd_sub_d(tmp.val, tmp.val, 0.5); // (ru - 0.5)
				rtd_nrand(tmp1.val); rtd_exp_mpfr(tmp1.val, tmp1.val); // exp(rn)
				rtd_mul(tmp.val, tmp.val, tmp1.val);
                rtd_set(ctmp.val_re, tmp.val);

				rtd_urand(tmp.val); // ru in [0, 1]
				rtd_sub_d(tmp.val, tmp.val, 0.5); // (ru - 0.5)
				rtd_nrand(tmp1.val);rtd_exp_mpfr(tmp1.val, tmp1.val); // exp(rn)
				rtd_mul(tmp.val, tmp.val, tmp1.val);
                rtd_set(ctmp.val_im, tmp.val);
				//mpc_set_fr_fr(ctmp, tmp, tmp, get_bnc_default_rounding_mode());
				set_ctdmatrix_ij(mat_a, i, j, &ctmp);

            }

           	//set_ddvector_i(vec_x, i, (dim - i) * sqrt(3.0));
			//rtd_sqrt_ui(tmp.val, 3UL);
			//rtd_mul_ui(tmp.val, tmp.val, (unsigned long)(dim - i));
			//tmp = sqrt((qd_real)3.0) * (qd_real)(double)(dim - i);
			// tmp := (ru - 0.5) * exp(rn)
			rtd_urand(tmp.val); // ru in [0, 1]
			rtd_sub_d(tmp.val, tmp.val, 0.5); // (ru - 0.5)
			rtd_nrand(tmp1.val); rtd_exp_mpfr(tmp1.val, tmp1.val); // exp(rn)
			rtd_mul(tmp.val, tmp.val, tmp1.val);
            rtd_set(ctmp.val_re, tmp.val);

			rtd_urand(tmp.val); // ru in [0, 1]
			rtd_sub_d(tmp.val, tmp.val, 0.5); // (ru - 0.5)
			rtd_nrand(tmp1.val);rtd_exp_mpfr(tmp1.val, tmp1.val); // exp(rn)
			rtd_mul(tmp.val, tmp.val, tmp1.val);
            rtd_set(ctmp.val_im, tmp.val);
			set_ctdvector_i(vec_x, i, &ctmp);
		}
		//rtd_sqrt_ui(tmp, 5UL);
		//std::cout << "sqrt5 = " << tmp.to_string() << std::endl;
		//rtd_mul_ui(tmp, tmp, (unsigned long)(2 + 3 + 1));
		//std::cout << "5 / 6 = " << tmp.to_string() << std::endl;

		//normf_ddmatrix(tmp.val, mat_a); std::cout << "||A||_F = " << tmp.val[0] << std::endl;
		//normf_ddmatrix(tmp.val, vec_x); std::cout << "||B||_F = " << tmp.val[0] << std::endl;

		iteration = 1; //1.0;
		do{
			stime = get_real_secv();

			num_addsub = mat_a->re->row_dim * vec_y->re->dim;
			num_mul = mat_a->re->row_dim * vec_y->re->dim;
//#ifndef USE_MBLAS
#ifdef USE_OZ
	for(i = 0; i < iteration; i++) mul_ctdmatrix_ctdvec_oz(vec_y, mat_a, max_num_div, vec_x, max_num_div);

#else // USE_OZ

#ifdef USE_OMP
//			for(i = 0; i < iteration; i++) mul_ddmatrix_block(vec_y, mat_a, vec_x, _BNC_DEFAULT_MIN_DIM_STRASSEN);
			for(i = 0; i < iteration; i++) _bncomp_mul_ctdmatrix_ctdvec(vec_y, mat_a, vec_x);
#else // USE_OMP
			for(i = 0; i < iteration; i++) mul_ctdmatrix_ctdvec(vec_y, mat_a, vec_x);
#endif // USE_OMP

#endif // USE_OZ
			//etime = get_secv() - stime;
			etime = get_real_secv() - stime;
			iteration *= 2; //2.0;
		} while(etime < 2);

		//printf("||C|| = "); normf_ctdmatrix(tmp.val, vec_y); cout << tmp.val[0]; printf("\n");

	// get relative errors
#ifdef USE_OMP
		_bncomp_mul_ctdmatrix_ctdvec(vec_y_long, mat_a, vec_x);
#else // USE_OMP
		mul_ctdmatrix_ctdvec(vec_y_long, mat_a, vec_x);
#endif // USE_OMP
		relerr3_ctdvector(relerr[0].val, relerr[1].val, relerr[2].val, relerr[3].val, relerr[4].val, relerr[5].val, relerr[6].val, vec_y, vec_y_long, 0);
        /* print_tdvector(vec_y->im);
        print_tdvector(vec_y_long->im);
        sub_ctdvector(vec_y_long, vec_y_long, vec_y);
        print_tdvector(vec_y_long->im);
        */
/*
		norm2_ddvector(tmp.val, vec_y);
		std::cout << "||y ||_2     = " << std::setprecision(32) << tmp << std::endl;
		norm2_ddvector(tmp.val, vec_y_long);
		std::cout << "||yl||_2     = " << std::setprecision(32) << tmp << std::endl;
		//sub_ddvector(vec_y, vec_y, vec_y_long);
		
        for(i = 0; i < dim; i++)
		{
			ddtmp = get_ddvector_i(vec_y, i);
			ddtmp2 = get_ddvector_i(vec_y_long, i);
			tmp.val[0]  = ddtmp[0] ; tmp.val[1]  = ddtmp[1];
			tmp2.x[0] = ddtmp2[0]; tmp2.x[1] = ddtmp2[1];
			std::cout << i << " " << std::setprecision(32) << tmp << " " << tmp2 << std::endl;
		}*
		norm2_ddvector(tmp.val, vec_y);
		std::cout << "||y - yl||_2 = " << std::setprecision(32) << tmp << std::endl;
*/
		//printf("etime = %g, Iteration = %ld\n", etime, iteration / 2);
		etime /= iteration / 2;

    	mflops = (double)(num_addsub + num_mul) / etime / 1000.0 / 1000.0;
		//printf("[%5d bits] %5d x %5d: %10.3g sec (%8.5f micro-sec/mul) %f MFlops", prec, dim, dim, etime, (etime / dim / dim / dim) * 1000 * 1000, mflops);
		//printf("[%5d bits] %5d x %5d: %10.3g sec (%8.5f micro-sec/mul) %f MFlops", prec, dim, dim, etime, (etime / (double)num_mul) * 1000 * 1000, mflops);
		//printf("[%5ld bits] %5ld x %5ld: %10.3g sec [mul]%ld [addsub]%ld", (long int)106, dim, dim, etime, num_mul, num_addsub);
		printf("[%5ld bits] %5ld x %5ld: %10.3g sec ", (long int)159, dim, dim, etime);
		printf(" "); printf("%10.3e", relerr[0].val[0]); // std::cout << relerr[0];
		printf(" "); printf("%10.3e", relerr[1].val[0]); // std::cout << relerr[1];
		printf(" "); printf("%10.3e", relerr[2].val[0]); // std::cout << relerr[2]; 
		printf(" "); printf("%10.3e", relerr[3].val[0]); // std::cout << relerr[0];
		printf(" "); printf("%10.3e", relerr[4].val[0]); // std::cout << relerr[1];
		printf(" "); printf("%10.3e", relerr[5].val[0]); // std::cout << relerr[2]; 
		printf(" "); printf("%10.3e", relerr[6].val[0]); // std::cout << relerr[0];
        //printf("\n");

		if(opt_dim_flag == 1)
			printf(" <-- Optimal dimension %+ld", dim - (end_dim + start_dim) / 2);
		printf("\n");

		free_ctdmatrix(mat_a);
		free_ctdvector(vec_x);
		free_ctdvector(vec_y);
		free_ctdvector(vec_y_long);

		// exceed max. comp. time
		if(etime > maxsec)
		{
			end_flag = 1;
			break;
		}
	}

	return end_flag;
}
//#endif // USE_DDLINEAR

int main(int argc, char *argv[])
{
	int num_threads = 1;
	long int dim, idim, idim_half, opt_dim, end_flag, min_dim = _BNC_DEFAULT_MIN_DIM_STRASSEN;
	long int cachesize = 128, maxsec = 1200;

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
		{
			printf("num_threads is 1\n");
		}
		if(argc >= 3)
		{
			maxsec = atol(argv[2]);
			if(maxsec < 0)
			{
				printf("Invalid Maximum seconds! (%ld seconds)\n", maxsec);
				return -1;
			}
		}
#else // USE_OMP
		maxsec = atol(argv[1]);
		if(maxsec < 0)
		{
			printf("Invalid Maximum seconds! (%ld seconds)\n", maxsec);
			return -1;
		}
#endif // USE_OMP
	}

	printf("------------------ Benchmark Test using Matrix Multiplication -----------------\n");

	// Initialize QD library
	//fpu_fix_start(NULL);
	printf("Maxsec: %ld\n", maxsec);

#ifdef USE_OMP
	set_bncomp_num_threads(num_threads);
	printf("OpenMP #threads = %d\n", omp_get_num_threads());
#endif // USE_OMP

#ifdef USE_OZ
	#ifdef OZ_MAX_NUM_DIV
	printf("Ozaki scheme, max_num_div = %d\n", OZ_MAX_NUM_DIV);
	#else // OZ_MAX_NUM_DIV
	printf("Ozaki scheme, max_num_div = %d\n", 10);
	#endif // OZ_MAX_NUM_DIV
#endif // USE_OZ

    bnc_print_env_all();
//	for(idim = 32; idim <= 16384; idim *= 2)
	//for(idim = 2; idim <= 2048; idim *= 2)
	for(idim = min_dim; idim < 32768; idim *= 2)
	//for(idim = 1024; idim < 32768; idim *= 2)
	{
		end_flag = 0;

		end_flag = matvecloop_ctd(idim - 1, idim + 1, 1, maxsec, 0);

//		end_flag = matmulloop_dd(idim - 1, idim + 1, 1, maxsec, 0);
		idim_half = 3 * idim / 2;

		if(idim_half >= min_dim)
			end_flag = matvecloop_ctd(idim_half - 1, idim_half + 1, 1, maxsec, 0);

//		if((idim <= opt_dim) && (opt_dim <= idim * 2))
//			end_flag = matmulloop_ctd(opt_dim - 5, opt_dim + 5, 1, maxsec, 1, min_dim);

		if(end_flag == 1)
			break;

	}
	printf("------------------------------------ END --------------------------------------\n");
}

