/********************************************************************************/
/* short_mmbench_mpf.c:                                                       */
/* Copyright (C) 2016 Tomonori Kouya                                            */
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
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "bnc.h"
#include "matmul_strassen.h"

// Statistics calculation

// average :  mu
double bnc_get_average(double datum[], long int num_datum)
{
	double ret = 0.0;
	long int i;

	if(num_datum <= 0)
	{
		fprintf(stderr, "Warning: num_datum = %ld is invalid!\n", num_datum);
		return ret;
	}

	for(i = 0; i < num_datum; i++)
		ret += datum[i];

	ret /= num_datum;

	return ret;
}

// variance : s^2
double bnc_get_variance(double datum[], long int num_datum)
{
	double ret = 0.0, average;
	long int i;

	if(num_datum <= 0)
	{
		fprintf(stderr, "Warning: num_datum = %ld is invalid!\n", num_datum);
		return ret;
	}

	average = bnc_get_average(datum, num_datum);

	for(i = 0; i < num_datum; i++)
		ret += (datum[i] - average) * (datum[i] - average);

	ret /= num_datum;

	return ret;
}

#ifdef USE_GMP

//
//                mat_c      :=                 mat_a         *      mat_b
//
//                                     min_dim
//                                      <---->
//          <-- col_dim --->            <----- mid_dim ----->   <-- col_dim --->
// -------- +----+----+----+   -------- +----+----+----+----+   +----+----+----+ -------------------
//     ^    |////|////|////|       ^    |////|////|////|////|   |////|////|////|                 ^
//     |    |////|////|////|       |    |////|////|////|////|   |////|////|////| min_dim         |
//     |    +----+----+----+       |    +----+----+----+----+   +----+----+----+ -------         |
//     |    |////|////|////|       |    |////|////|////|////|   |////|////|////|                 |
//     |    |////|////|////|       |    |////|////|////|////|   |////|////|////|                 
//  row_dim +----+----+----+ := row_dim +----+----+----+----+ * +----+----+----+              mid_dim
//          |    |    |    |            |    |    |    |    |   |////|////|////|                 
//     |    |    |    |    |       |    |    |    |    |    |   |////|////|////|                 |
//     |    +----+----+----+       |    +----+----+----+----+   +----+----+----+                 |
//     |    |    |    |    |       |    |    |    |    |    |   |////|////|////|                 |
//     |    |    |    |    |       |    |    |    |    |    |   |////|////|////|                 v
//     |    +----+----+----+       |    +----+----+----+----+   +----+----+----+  --------------------
//     |    |    |    |    |       |    |    |    |    |    |
//     v    |    |    |    |       v    |    |    |    |    |
// -------- +----+----+----+   -------- +----+----+----+----+

int short_mmbench_mpf(long int row_dim, long int col_dim, long int mid_dim, long int min_dim, int trial_times, long int maxsec, unsigned long prec)
{
	long int num_row_div, mat_a_row_dim, mat_a_col_dim, mat_b_row_dim, mat_b_col_dim, mat_c_row_dim, mat_c_col_dim;
	long int i, j, end_flag;
	long int num_addsub, num_mul;
	int times;
	double iteration, *etime = NULL, stime, mflops, etime_average, etime_average_old, etime_max, etime_min;
	MPFMatrix mat_a, mat_b, mat_c, mat_c_long;
	mpf_t tmp, relerr[3];

	mpf_init2(tmp, prec);
	mpf_init2(relerr[0], prec);
	mpf_init2(relerr[1], prec);
	mpf_init2(relerr[2], prec);

	num_row_div = 2;
	mat_a_row_dim = min_dim * num_row_div;
	mat_a_col_dim = mid_dim;
	mat_b_row_dim = mid_dim;
	mat_b_col_dim = col_dim;
	mat_c_row_dim = mat_a_row_dim;
	mat_c_col_dim = col_dim;

	etime = (double *)malloc(sizeof(double) * trial_times);
	etime_average = 0.0;

	printf("[%ld bits] (%ld x %ld) * (%ld x %ld) start!\n", prec, mat_a_row_dim, mat_a_col_dim, row_dim, col_dim);

	for(times = 0; times < trial_times; times++)
	{
		end_flag = 0;

		mat_a = init2_mpfmatrix(mat_a_row_dim, mat_a_col_dim, prec);
		mat_b = init2_mpfmatrix(mat_b_row_dim, mat_b_col_dim, prec);
		mat_c = init2_mpfmatrix(mat_c_row_dim, mat_c_col_dim, prec);
		mat_c_long = init2_mpfmatrix(mat_c_row_dim, mat_c_col_dim, (unsigned long)(prec + prec / 2));

		// set mat_a
		for(i = 0; i < mat_a_row_dim; i++)
		{
			for(j = 0; j < mat_a_col_dim; j++)
			{
				mpf_sqrt_ui(tmp, 5UL);
				mpf_mul_ui(tmp, tmp, (unsigned long)(i + j + 1));
				set_mpfmatrix_ij(mat_a, i, j, tmp);
			}
		}

		// set_mat_b
		for(i = 0; i < mat_b_row_dim; i++)
		{
			for(j = 0; j < mat_b_col_dim; j++)
			{
				mpf_sqrt_ui(tmp, 3UL);
				mpf_mul_ui(tmp, tmp, (unsigned long)(row_dim - i));
				set_mpfmatrix_ij(mat_b, i, j, tmp);
			}
		}

		iteration = 1.0;
		do{
			stime = get_secv();
#ifdef USE_STRASSEN
			for(i = 0; i < iteration; i++)
			{
				reset_num_mul_mpfmatrix_strassen();
				mul_mpfmatrix_strassen(mat_c, mat_a, mat_b, min_dim);
				get_num_mul_mpfmatrix_strassen(&num_addsub, &num_mul);
			}

#elif USE_STRASSEN_PADDING
			for(i = 0; i < iteration; i++)
			{
				reset_num_mul_mpfmatrix_strassen();
				mul_mpfmatrix_strassen_odd_padding(mat_c, mat_a, mat_b, min_dim);
				get_num_mul_mpfmatrix_strassen(&num_addsub, &num_mul);
			}

#elif USE_BLOCK
			num_addsub = mat_a->row_dim * mat_a->col_dim * mat_b->row_dim;
			num_mul = mat_a->row_dim * mat_a->col_dim * mat_b->row_dim;
			for(i = 0; i < iteration; i++) mul_mpfmatrix_block(mat_c, mat_a, mat_b, min_dim);
#else // Simple
			num_addsub = mat_a->row_dim * mat_c->row_dim * mat_c->col_dim;
			num_mul = mat_a->row_dim * mat_c->row_dim * mat_c->col_dim;
			for(i = 0; i < iteration; i++) mul_mpfmatrix_simple(mat_c, mat_a, mat_b);
#endif
			etime[times] = get_secv() - stime;
			iteration *= 2.0;
		} while(etime[times] < 2);

		// get relative errors
		mul_mpfmatrix_strassen(mat_c_long, mat_a, mat_b, _BNC_DEFAULT_MIN_DIM_STRASSEN);
//		mul_mpfmatrix_simple(mat_c_long, mat_a, mat_b);
		relerr3_mpfmatrix(relerr[0], relerr[1], relerr[2], mat_c, mat_c_long, 0);

		etime[times] /= iteration / 2;

		etime_average_old = etime_average;
		etime_average = bnc_get_average(etime, times + 1);
		if(times <= 0)
		{
			etime_max = etime[times];
			etime_min = etime[times];
		}
		else
		{
			if(etime_max < etime[times]) etime_max = etime[times];
			if(etime_min > etime[times]) etime_min = etime[times];

			if(times >= 5)
			{
				if((fabs((etime_average - etime_average_old) / etime_average) <= 0.001) || (etime[times] >= maxsec))
				{
					printf("average seconds: %g\n", etime_average);
					printf("max     seconds: %g\n", etime_max);
					printf("min     seconds: %g\n", etime_min);
					end_flag = 1;
				}
			}
		}

		if((end_flag == 1) || (times >= (trial_times - 1)))
		{
			printf("\n");
			mflops = (double)(num_addsub + num_mul) / etime_average / 1000.0 / 1000.0;
			printf("%5d [%5d bits] (%5d x %5d) * (%5d x %5d): %9.5f sec (%8.5f micro-sec/mul) %f MFlops", times, prec, mat_a->row_dim, mat_a->col_dim, mat_b->row_dim, mat_b->col_dim, etime_average, (etime_average / (double)num_mul) * 1000 * 1000, mflops);
			printf(" "); mpf_out_str(stdout, 10, 3, relerr[0]);
			printf(" "); mpf_out_str(stdout, 10, 3, relerr[1]);
			printf(" "); mpf_out_str(stdout, 10, 3, relerr[2]);
			printf("\n");
			printf("    [%5ld bits] (%5d x %5d) * (%5d x %5d): %9.5f sec (estimated)\n", prec, mat_a->col_dim, mat_a->col_dim, mat_b->row_dim, mat_b->col_dim, ceil((double)(mat_a->col_dim) / (double)(mat_a->row_dim)) * etime_average);
		}
		else
			printf(" %d ...", times);

		free_mpfmatrix(mat_a);
		free_mpfmatrix(mat_b);
		free_mpfmatrix(mat_c);
		free_mpfmatrix(mat_c_long);

		// exceed max. comp. time
		//if(etime > maxsec)
		if(end_flag == 1)
		{
			break;
		}
	}

	mpf_clear(tmp);
	mpf_clear(relerr[0]);
	mpf_clear(relerr[1]);
	mpf_clear(relerr[2]);

	free(etime);

	return end_flag;
}
#endif // USE_GMP

// usage
void usage(const char *commandname)
{
#if defined(USE_STRASSEN) || defined(USE_STRASSEN_PADDING) || defined(USE_BLOCK)
	printf("USAGE-> %s precision(in bits) row_dim col_dim mid_dim [min_dim] [Maximum seconds]\n", commandname);
#else
	printf("USAGE-> %s precision(in bits) row_dim col_dim mid_dim [Maximum seconds]\n", commandname);
#endif
}

int main(int argc, char *argv[])
{
	long int row_dim, col_dim, mid_dim, min_dim = _BNC_DEFAULT_MIN_DIM_STRASSEN;
	long int maxsec = 600;
	unsigned long prec;

	if(argc < 4)
	{
		usage(argv[0]);
		return 0;
	}
	else if(argc >= 5)
	{
		prec = atol(argv[1]);
		if(prec <= 0)
		{
			printf("Invalid precision in bit! (%ld bits)\n", prec);
			return -1;
		}

		row_dim = atol(argv[2]);
		col_dim = atol(argv[3]);
		mid_dim = atol(argv[4]);

#if defined(USE_STRASSEN) || defined(USE_STRASSEN_PADDING) || defined(USE_BLOCK)
		if(row_dim < _BNC_DEFAULT_MIN_DIM_STRASSEN)
		{
			printf("Warning: Too small row_dim (%ld < %ld)\n", row_dim, _BNC_DEFAULT_MIN_DIM_STRASSEN);
		}
		if(col_dim < _BNC_DEFAULT_MIN_DIM_STRASSEN)
		{
			printf("Warning: Too small col_dim (%ld < %ld)\n", col_dim, _BNC_DEFAULT_MIN_DIM_STRASSEN);
		}
		if(mid_dim < _BNC_DEFAULT_MIN_DIM_STRASSEN)
		{
			printf("Warning: Too small mid_dim (%ld < %ld)\n", mid_dim, _BNC_DEFAULT_MIN_DIM_STRASSEN);
		}

		if(argc >= 6)
		{
			min_dim = atol(argv[5]);
			if(min_dim < _BNC_DEFAULT_MIN_DIM_STRASSEN)
			{
				printf("Warning: Too small min_dim (%ld < %ld)\n", min_dim, _BNC_DEFAULT_MIN_DIM_STRASSEN);
				//return -1;
			}
			if(argc >= 7)
				maxsec = atol(argv[6]);
		}
#else
		if(argc >= 6)
			maxsec = atol(argv[5]);
#endif // defined(USE_STRASSEN) || defined(USE_STRASSEN_PADDING) || defined(USE_BLOCK)
	}

	set_bnc_default_prec(prec);

	printf("------------------ Benchmark Test using Matrix Multiplication -----------------\n");

	short_mmbench_mpf(row_dim, col_dim, mid_dim, min_dim, 10, maxsec, prec);

	printf("------------------------------------ END --------------------------------------\n");
}

