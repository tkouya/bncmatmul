/********************************************************************************/
/* matvecbench_qd.cc:                                                           */
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

// mplapack, mpblas
//#ifdef USE_MBLAS
#ifdef OLD_MBLAS
    #include <mblas_qd.h>
    //#include <mlapack_qd.h>
#else // OLD_MBLAS
    #include <mpblas_qd.h>
    //#include <mplapack_qd.h>
#endif // OLD_MBLAS

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
void normf_qdmatrix_(double ret[QDSIZE], QDMatrix mat)
{
	long int i;
	long int real_total_dim;
	double tmp[QDSIZE];

	real_total_dim = mat->real_row_dim * mat->real_col_dim;

	double mat1[QDSIZE];

	rqd_set0(ret);
	for(i = 0; i < real_total_dim; i++)
	{
		mat1[0] = mat->element[0][i];
		mat1[1] = mat->element[1][i];

		// tmp := mat1[i]^2
		// ret += tmp
		rqd_mul(tmp, mat1, mat1);
		rqd_add(ret, ret, tmp);
	}
	//printf("ret = %25.17e\n", ret[0]);

	rqd_sqrt(tmp, ret);
	//printf("tmp = %25.17e\n", tmp[0]);
	rqd_set(ret, tmp);
	//printf("ret = %25.17e\n", ret[0]);
}

int matvecloop_qd(long int start_dim, long int end_dim, long int dim_step, long int maxsec, long int opt_dim_flag)
{
	long int dim;
	long int i, j, end_flag = 0;
	long int num_addsub, num_mul, iteration;
	double etime, stime, mflops;
	QDMatrix mat_a;
    QDVector vec_x, vec_y; // vec_y := mat_a * vec_x 
	QDVector vec_y_long;
	qd_real tmp, relerr[3], one = (qd_real)1, zero = (qd_real)0;
	qd_real *mat_a_m, *vec_x_m, *vec_y_m;
#ifdef USE_OZ
	int max_num_div; // For Ozaki scheme
	#ifdef OZ_MAX_NUM_DIV
	max_num_div = OZ_MAX_NUM_DIV;
	#else // OZ_MAX_NUM_DIV
	max_num_div = 10;
	fprintf(stderr, "Warning: no definition of OZ_MAX_NUM_DIV! -> max_num_div = %d\n", max_num_div);
	#endif // OZ_MAX_NUM_DIV
#endif // USE_OZ

	for(dim = start_dim; dim <= end_dim; dim += dim_step)
	{
		mat_a = init_qdmatrix(dim, dim);
	    vec_x = init_qdvector(dim);
		vec_y = init_qdvector(dim);
		vec_y_long = init_qdvector(dim);

	// mpack
		mat_a_m = new qd_real[dim * dim];
		vec_x_m = new qd_real[dim];
		vec_y_m = new qd_real[dim];

		for(i = 0; i < dim; i++)
		{
			for(j = 0; j < dim; j++)
			{
//				set_qdmatrix_ij(mat_a, i, j, (i + j + 1) * sqrt(5.0));
//				rqd_set_ui(tmp.x, 5UL);
//				rqd_sqrt(tmp.x, tmp.x);
				tmp = sqrt((qd_real)5.0) * (qd_real)(double)(i + j + 1);
					//tmp = sqrt((qd_real)rand()) * (qd_real)(double)(i + j + 1);
				//rqd_sqrt_ui(tmp.x, 5UL);
				//cout << "tmp = " << tmp << endl;
				//rqd_mul_ui(tmp.x, tmp.x, (unsigned long)(i + j + 1));
				set_qdmatrix_ij(mat_a, i, j, tmp.x);
				// mpack
				mat_a_m[COL_ZERO_IJ(i, j, dim, dim)] = tmp;
				//cout << "tmp = " << tmp << endl;
				//printf("tmp_x = %25.17e\n", tmp.x[0]);
            }

           	//set_qdvector_i(vec_x, i, (dim - i) * sqrt(3.0));
			//rqd_sqrt_ui(tmp.x, 3UL);
			//rqd_mul_ui(tmp.x, tmp.x, (unsigned long)(dim - i));
			tmp = sqrt((qd_real)3.0) * (qd_real)(double)(dim - i);
			set_qdvector_i(vec_x, i, tmp.x);
			// mpack
			vec_x_m[i] = tmp;

			rqd_sqrt_ui(tmp.x, 2UL);
			set_qdvector_i(vec_y, i, tmp.x);
			// mpack
			vec_y_m[i] = tmp;
		}
		//rqd_sqrt_ui(tmp, 5UL);
		//std::cout << "sqrt5 = " << tmp.to_string() << std::endl;
		//rqd_mul_ui(tmp, tmp, (unsigned long)(2 + 3 + 1));
		//std::cout << "5 / 6 = " << tmp.to_string() << std::endl;

		//normf_qdmatrix(tmp.x, mat_a); std::cout << "||A||_F = " << tmp.x[0] << std::endl;
		//normf_qdmatrix(tmp.x, vec_x); std::cout << "||B||_F = " << tmp.x[0] << std::endl;

		iteration = 1; //1.0;
		do{
			stime = get_real_secv();

			num_addsub = mat_a->row_dim * vec_y->dim;
			num_mul = mat_a->row_dim * vec_y->dim;
//#ifndef USE_MBLAS
#ifdef USE_OZ
	for(i = 0; i < iteration; i++) mul_qdmatrix_qdvec_oz(vec_y, mat_a, max_num_div, vec_x, max_num_div);

#else // USE_OZ

#ifndef USE_MBLAS
#ifdef USE_OMP
//			for(i = 0; i < iteration; i++) mul_qdmatrix_block(vec_y, mat_a, vec_x, _BNC_DEFAULT_MIN_DIM_STRASSEN);
			for(i = 0; i < iteration; i++) _bncomp_mul_qdmatrix_qdvec(vec_y, mat_a, vec_x);
#else // USE_OMP
			for(i = 0; i < iteration; i++) mul_qdmatrix_qdvec(vec_y, mat_a, vec_x);
#endif // USE_OMP

#else // USE_MBLAS 
			num_addsub = mat_a->row_dim * vec_y->dim;
			num_mul = mat_a->row_dim * vec_y->dim;
			//cout << "one = " << one << ", zero = " << zero << endl; //printf("Rgemm(%ld, %ld)\n", dim, dim);
			for(i = 0; i < iteration; i++) Rgemv("n", dim, dim, one, mat_a_m, dim, vec_x_m, 1, zero, vec_y_m, 1);

#endif // USE_MBLAS
#endif // USE_OZ

			//etime = get_secv() - stime;
			etime = get_real_secv() - stime;
			iteration *= 2; //2.0;
		} while(etime < 2);

#ifdef USE_MBLAS
		for(i = 0; i < dim; i++)
		{
			set_qdvector_i(vec_y, i, vec_y_m[i].x);
		}
		//printf("||C|| = "); normf_qdmatrix(tmp.x, vec_y); cout << tmp.x[0]; printf("\n");
#else // USE_MBLAS
		//printf("||C|| = "); normf_qdmatrix(tmp.x, vec_y); cout << tmp.x[0]; printf("\n");
#endif // USE_MBLAS

		// get relative errors
#ifdef USE_OMP
		_bncomp_mul_qdmatrix_qdvec(vec_y_long, mat_a, vec_x);
#else // USE_OMP
		mul_qdmatrix_qdvec(vec_y_long, mat_a, vec_x);
#endif // USE_OMP
		relerr3_qdvector(relerr[0].x, relerr[1].x, relerr[2].x, vec_y, vec_y_long, 0);

		//printf("etime = %g, Iteration = %ld\n", etime, iteration / 2);
		etime /= iteration / 2;

    	mflops = (double)(num_addsub + num_mul) / etime / 1000.0 / 1000.0;
		//printf("[%5d bits] %5d x %5d: %9.5f sec (%8.5f micro-sec/mul) %f MFlops", prec, dim, dim, etime, (etime / dim / dim / dim) * 1000 * 1000, mflops);
		//printf("[%5d bits] %5d x %5d: %9.5f sec (%8.5f micro-sec/mul) %f MFlops", prec, dim, dim, etime, (etime / (double)num_mul) * 1000 * 1000, mflops);
		//printf("[%5ld bits] %5ld x %5ld: %9.5f sec [mul]%ld [addsub]%ld", (long int)212, dim, dim, etime, num_mul, num_addsub);
		printf("[%5ld bits] %5ld x %5ld: %9.5f sec ", (long int)212, dim, dim, etime);
		printf(" "); printf("%10.3e", to_double(relerr[0])); //rqd_out_str(relerr[0]);
		printf(" "); printf("%10.3e", to_double(relerr[1])); //rqd_out_str(relerr[1]);
		printf(" "); printf("%10.3e", to_double(relerr[2])); //rqd_out_str(relerr[2]);// printf("\n");

		if(opt_dim_flag == 1)
			printf(" <-- Optimal dimension %+ld", dim - (end_dim + start_dim) / 2);
		printf("\n");

		free_qdmatrix(mat_a);
		free_qdvector(vec_x);
		free_qdvector(vec_y);
		free_qdvector(vec_y_long);

		// mpack
		delete[] mat_a_m;
		delete[] vec_x_m;
		delete[] vec_y_m;

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
	fpu_fix_start(NULL);
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

		end_flag = matvecloop_qd(idim - 1, idim + 1, 1, maxsec, 0);

//		end_flag = matmulloop_qd(idim - 1, idim + 1, 1, maxsec, 0);
		idim_half = 3 * idim / 2;

		if(idim_half >= min_dim)
			end_flag = matvecloop_qd(idim_half - 1, idim_half + 1, 1, maxsec, 0);

//		if((idim <= opt_dim) && (opt_dim <= idim * 2))
//			end_flag = matmulloop_qd(opt_dim - 5, opt_dim + 5, 1, maxsec, 1, min_dim);

		if(end_flag == 1)
			break;

	}
	printf("------------------------------------ END --------------------------------------\n");
}

