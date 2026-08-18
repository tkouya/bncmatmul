/********************************************************************************/
/* matvecbench_td.cc:                                                           */
/* Copyright (C) 2021 Tomonori Kouya                                            */
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
#include "qd/qd_real.h"

using namespace std;

//#include "bnc.h"
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

// Frobenius norm
void normf_tdmatrix_(double ret[TDSIZE], TDMatrix mat)
{
	long int i;
	long int real_total_dim;
	double tmp[TDSIZE];

	real_total_dim = mat->real_row_dim * mat->real_col_dim;

	double mat1[TDSIZE];

	rtd_set0(ret);
	for(i = 0; i < real_total_dim; i++)
	{
		mat1[0] = mat->element[0][i];
		mat1[1] = mat->element[1][i];

		// tmp := mat1[i]^2
		// ret += tmp
		rtd_mul(tmp, mat1, mat1);
		rtd_add(ret, ret, tmp);
	}
	//printf("ret = %25.17e\n", ret[0]);

	rtd_sqrt(tmp, ret);
	//printf("tmp = %25.17e\n", tmp[0]);
	rtd_set(ret, tmp);
	//printf("ret = %25.17e\n", ret[0]);
}

int matvecloop_td(long int start_dim, long int end_dim, long int dim_step, long int maxsec, long int opt_dim_flag)
{
	long int dim;
	long int i, j, end_flag = 0;
	long int num_addsub, num_mul, iteration;
	double etime, stime, mflops;
	TDMatrix mat_a;
    TDVector vec_x, vec_y; // vec_y := mat_a * vec_x 
	//TDVector vec_y_long;
	QDMatrix mat_a_long;
	QDVector vec_x_long, vec_y_long;
    double tmp[TDSIZE], relerr[3][TDSIZE];

	for(dim = start_dim; dim <= end_dim; dim += dim_step)
	{
		mat_a = init_tdmatrix(dim, dim);
	    vec_x = init_tdvector(dim);
		vec_y = init_tdvector(dim);
		//vec_y_long = init_tdvector(dim);
		mat_a_long = init_qdmatrix(dim, dim);
	    vec_x_long = init_qdvector(dim);
		vec_y_long = init_qdvector(dim);

		for(i = 0; i < dim; i++)
		{
			for(j = 0; j < dim; j++)
			{
//				set_tdmatrix_ij(mat_a, i, j, (i + j + 1) * sqrt(5.0));
				rtd_set_ui(tmp, 5UL);
				rtd_sqrt(tmp, tmp);
				rtd_mul_ui(tmp, tmp, (unsigned long)(i + j + 1));
				set_tdmatrix_ij(mat_a, i, j, tmp);
            }

           	//set_tdvector_i(vec_x, i, (dim - i) * sqrt(3.0));
			rtd_sqrt_ui(tmp, 3UL);
			rtd_mul_ui(tmp, tmp, (unsigned long)(dim - i));
			set_tdvector_i(vec_x, i, tmp);

			rtd_sqrt_ui(tmp, 2UL);
			set_tdvector_i(vec_y, i, tmp);
		}
		//rtd_sqrt_ui(tmp, 5UL);
		//std::cout << "sqrt5 = " << tmp.to_string() << std::endl;
		//rtd_mul_ui(tmp, tmp, (unsigned long)(2 + 3 + 1));
		//std::cout << "5 / 6 = " << tmp.to_string() << std::endl;

		//normf_tdmatrix(tmp.x, mat_a); std::cout << "||A||_F = " << tmp.x[0] << std::endl;
		//normf_tdmatrix(tmp.x, vec_x); std::cout << "||B||_F = " << tmp.x[0] << std::endl;

		iteration = 1; //1.0;
		do{
			stime = get_real_secv();

			num_addsub = mat_a->row_dim * vec_y->dim;
			num_mul = mat_a->row_dim * vec_y->dim;
#ifdef USE_OMP
//			for(i = 0; i < iteration; i++) mul_tdmatrix_block(vec_y, mat_a, vec_x, _BNC_DEFAULT_MIN_DIM_STRASSEN);
			for(i = 0; i < iteration; i++) _bncomp_mul_tdmatrix_tdvec(vec_y, mat_a, vec_x);
#else // USE_OMP
			for(i = 0; i < iteration; i++) mul_tdmatrix_tdvec(vec_y, mat_a, vec_x);
#endif // USE_OMP

			//etime = get_secv() - stime;
			etime = get_real_secv() - stime;
			iteration *= 2; //2.0;
		} while(etime < 2);

		//printf("||C|| = "); normf_tdmatrix(tmp.x, vec_y); cout << tmp.x[0]; printf("\n");

		// get relative errors
		subst_qdmatrix_tdmat(mat_a_long, mat_a);
		subst_qdvector_tdvec(vec_x_long, vec_x);
#ifdef USE_OMP
		//_bncomp_mul_tdmatrix_tdvec(vec_y_long, mat_a, vec_x);
		_bncomp_mul_qdmatrix_qdvec(vec_y_long, mat_a_long, vec_x_long);
#else // USE_OMP
		//mul_tdmatrix_tdvec(vec_y_long, mat_a, vec_x);
		mul_qdmatrix_qdvec(vec_y_long, mat_a_long, vec_x_long);
#endif // USE_OMP
		//relerr3_tdvector(relerr[0], relerr[1], relerr[2], vec_y, vec_y_long, 0);
		//printf("qdvec");
		relerr3_tdvector_qdvec(relerr[0], relerr[1], relerr[2], vec_y, vec_y_long, 0);

		//printf("etime = %g, Iteration = %ld\n", etime, iteration / 2);
		etime /= iteration / 2;

    	mflops = (double)(num_addsub + num_mul) / etime / 1000.0 / 1000.0;
		//printf("[%5d bits] %5d x %5d: %9.5f sec (%8.5f micro-sec/mul) %f MFlops", prec, dim, dim, etime, (etime / dim / dim / dim) * 1000 * 1000, mflops);
		//printf("[%5d bits] %5d x %5d: %9.5f sec (%8.5f micro-sec/mul) %f MFlops", prec, dim, dim, etime, (etime / (double)num_mul) * 1000 * 1000, mflops);
		printf("[%5ld bits] %5ld x %5ld: %9.5f sec [mul]%ld [addsub]%ld", (long int)159, dim, dim, etime, num_mul, num_addsub);
		printf(" "); printf("%10.3e", relerr[0][0]); //rtd_out_str(relerr[0]);
		printf(" "); printf("%10.3e", relerr[1][0]); //rtd_out_str(relerr[1]);
		printf(" "); printf("%10.3e", relerr[2][0]); //rtd_out_str(relerr[2]);// printf("\n");

		if(opt_dim_flag == 1)
			printf(" <-- Optimal dimension %+ld", dim - (end_dim + start_dim) / 2);
		printf("\n");

		free_tdmatrix(mat_a);
		free_tdvector(vec_x);
		free_tdvector(vec_y);
		//free_tdvector(vec_y_long);
		free_qdmatrix(mat_a_long);
		free_qdvector(vec_x_long);
		free_qdvector(vec_y_long);

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
    bnc_print_env_all();
	printf("------------------ Benchmark Test using Matrix Multiplication -----------------\n");

	// Initialize QD library
	fpu_fix_start(NULL);
	printf("Maxsec: %ld\n", maxsec);

#ifdef USE_OMP
	set_bncomp_num_threads(num_threads);
	printf("OpenMP #threads = %d\n", omp_get_num_threads());
#endif // USE_OMP

//	for(idim = 32; idim <= 16384; idim *= 2)
	//for(idim = 2; idim <= 2048; idim *= 2)
	for(idim = min_dim; idim < 32768; idim *= 2)
	//for(idim = 1024; idim < 32768; idim *= 2)
	{
		end_flag = 0;

		end_flag = matvecloop_td(idim - 1, idim + 1, 1, maxsec, 0);

//		end_flag = matmulloop_td(idim - 1, idim + 1, 1, maxsec, 0);
		idim_half = 3 * idim / 2;

		if(idim_half >= min_dim)
			end_flag = matvecloop_td(idim_half - 1, idim_half + 1, 1, maxsec, 0);

//		if((idim <= opt_dim) && (opt_dim <= idim * 2))
//			end_flag = matmulloop_td(opt_dim - 5, opt_dim + 5, 1, maxsec, 1, min_dim);

		if(end_flag == 1)
			break;

	}
	printf("------------------------------------ END --------------------------------------\n");
}

