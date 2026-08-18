/********************************************************************************/
/* short_mmbench_dd.c:                                                         */
/* Copyright (C) 2017 Tomonori Kouya                                            */
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
#include "bncomp.h"
#include "matmul_strassen.h"

// Statistics calculation

// minimum
double bnc_get_min(long int *index_min, double datum[], long int num_datum)
{
	double ret = 0.0;
	long int i, in_index_min;

	if(num_datum <= 0)
	{
		fprintf(stderr, "Warning: num_datum = %ld is invalid!\n", num_datum);
		return ret;
	}

	ret = datum[0];
	in_index_min = 0;
	for(i = 1; i < num_datum; i++)
	{
		if(ret > datum[i])
		{
			ret = datum[i];
			in_index_min = i;
		}
	}

	if(index_min != NULL)
		*index_min = in_index_min;

	return ret;
}

// maximum
double bnc_get_max(long int *index_max, double datum[], long int num_datum)
{
	double ret = 0.0;
	long int i, in_index_max;

	if(num_datum <= 0)
	{
		fprintf(stderr, "Warning: num_datum = %ld is invalid!\n", num_datum);
		return ret;
	}

	ret = datum[0];
	in_index_max = 0;
	for(i = 1; i < num_datum; i++)
	{
		if(ret < datum[i])
		{
			ret = datum[i];
			in_index_max = i;
		}
	}

	if(index_max != NULL)
		*index_max = in_index_max;

	return ret;
}

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


#define MIN(a, b) (((a) < (b)) ? (a) : (b))

//#ifdef USE_DDLINEAR

// compare three algorithms: simple, block(block_min_dim), strassen(strassen_min_dim)
// mat_c[row_dim, col_dim] := mat_a[row_dim, mid_dim] * mat_b[mid_dim, col_dim]
void get_opt_mmbench_dd(FILE *opt_fp, int flag_opt_print_header, long int row_dim, long int col_dim, long int mid_dim, long int block_min_dim, long int strassen_min_dim, int trial_times, long int maxsec, int num_threads)
{
	long int i, j, end_flag, min_i;
	int times;
	double iteration, stime, *etime_array, etime_average[3], etime_stdev, etime_average_old, etime_max, etime_min;
	DDMatrix mat_a, mat_b, mat_c, mat_c_long;
	double tmp[DDSIZE], relerr[3][DDSIZE];

	etime_array = (double *)calloc(trial_times * DDSIZE, sizeof(double));

	mat_a = init_ddmatrix(row_dim, mid_dim);
	mat_b = init_ddmatrix(mid_dim, col_dim);
	mat_c = init_ddmatrix(row_dim, col_dim);
	mat_c_long = init_ddmatrix(row_dim, col_dim);

	// set mat_a
	for(i = 0; i < row_dim; i++)
	{
		for(j = 0; j < mid_dim; j++)
		{
			rdd_sqrt_ui(tmp, 5UL);
			rdd_mul_ui(tmp, tmp, (unsigned long)(i + j + 1));
			set_ddmatrix_ij(mat_a, i, j, tmp);
		}
	}

	// set_mat_b
	for(i = 0; i < mid_dim; i++)
	{
		for(j = 0; j < col_dim; j++)
		{
			rdd_sqrt_ui(tmp, 3UL);
			rdd_mul_ui(tmp, tmp, (unsigned long)(mat_b->row_dim - i));
			set_ddmatrix_ij(mat_b, i, j, tmp);
		}
	}

#ifdef USE_OMP
	//mul_ddmatrix_strassen(mat_c_long, mat_a, mat_b, _BNC_DEFAULT_MIN_DIM_STRASSEN);
	_bncomp_mul_ddmatrix_strassen(mat_c_long, mat_a, mat_b, _BNC_DEFAULT_MIN_DIM_STRASSEN);
#else // USE_OMP
	mul_ddmatrix_strassen(mat_c_long, mat_a, mat_b, _BNC_DEFAULT_MIN_DIM_STRASSEN);
	//mul_ddmatrix_simple(mat_c_long, mat_a, mat_b);
#endif // USE_OMP

	// Simple
	//printf("simple\n");
	printf("Alg. (min_dim)  #Ths     Average      max.        min.     stdev     rel.err\n");
	//      Alg. (min_dim)  #Ths     Average      max.        min.     stdev
	//      Simple             1      0.1608      0.1611      0.1605   0.0001614
	//      block(  32)        1      0.1608      0.1608      0.1608   9.739e-06
	//      Strassen(  32)     1      0.1124      0.1124      0.1124   1.909e-05

	etime_average[0] = 0.0;
	for(times = 0; times < trial_times; times++)
	{
		end_flag = 0;

		set0_ddmatrix(mat_c);

		iteration = 1.0;
		do{
			stime = get_real_secv();

			//*num_addsub = mat_a->row_dim * mat_a->col_dim * mat_b->row_dim;
			//*num_mul = mat_a->row_dim * mat_a->col_dim * mat_b->row_dim;

#ifdef USE_OMP
			for(i = 0; i < iteration; i++) _bncomp_mul_ddmatrix_simple(mat_c, mat_a, mat_b);
#else // USE_OMP
			for(i = 0; i < iteration; i++) mul_ddmatrix_simple(mat_c, mat_a, mat_b);
#endif // USE_OMP

			etime_array[times] = get_real_secv() - stime;
			iteration *= 2.0;
		} while(etime_array[times] < 2);

		// get relative errors
		relerr3_ddmatrix(relerr[0], relerr[1], relerr[2], mat_c, mat_c_long, 0);

		etime_array[times] /= iteration / 2;

		etime_average_old = etime_average[0];
		etime_average[0] = bnc_get_average(etime_array, times + 1);
		etime_stdev = sqrt(bnc_get_variance(etime_array, times + 1));
		if(times <= 0)
		{
			etime_max = etime_array[times];
			etime_min = etime_array[times];
		}
		else
		{
			if(etime_max < etime_array[times]) etime_max = etime_array[times];
			if(etime_min > etime_array[times]) etime_min = etime_array[times];

			if(times >= 5)
			{
				if((fabs((etime_average[0] - etime_average_old) / etime_average[0]) <= 0.001) || (etime_array[times] >= maxsec))
				{
					//printf("average seconds: %g\n", etime_average);
					//printf("max     seconds: %g\n", etime_max);
					//printf("min     seconds: %g\n", etime_min);
					//printf("stdev          : %g\n", etime_stdev);
					end_flag = 1;
				}
			}
		}
		// exceed max. comp. time
		//if(etime > maxsec)
		if(end_flag == 1)
		{
			break;
		}
	}
	//printf("Simple         %5d %11.4g %11.4g %11.4g %11.4g\n", num_threads, etime_average[0], etime_max, etime_min, etime_stdev);
	printf("Simple         %5d %11.4g %11.4g %11.4g %11.4g ", num_threads, etime_average[0], etime_max, etime_min, etime_stdev);
	rdd_out_str_base(stdout, 10, 3, relerr[0]);
	printf("\n");

	// block(block_min_dim)
	if(block_min_dim <= 0)
		printf("Warning!: block_size = %ld is too small!\n", block_min_dim);

	etime_average[1] = 0.0;
	for(times = 0; times < trial_times; times++)
	{
		end_flag = 0;

		set0_ddmatrix(mat_c);

		iteration = 1.0;
		do{
			stime = get_real_secv();

			//*num_addsub = mat_a->row_dim * mat_a->col_dim * mat_b->row_dim;
			//*num_mul = mat_a->row_dim * mat_a->col_dim * mat_b->row_dim;

	#ifdef USE_OMP
			for(i = 0; i < iteration; i++) _bncomp_mul_ddmatrix_block(mat_c, mat_a, mat_b, block_min_dim);
	#else // USE_OMP
			for(i = 0; i < iteration; i++) mul_ddmatrix_block(mat_c, mat_a, mat_b, block_min_dim);
	#endif // USE_OMP

			etime_array[times] = get_real_secv() - stime;
			iteration *= 2.0;
		} while(etime_array[times] < 2);

		// get relative errors
		relerr3_ddmatrix(relerr[0], relerr[1], relerr[2], mat_c, mat_c_long, 0);

		etime_array[times] /= iteration / 2;

		etime_average_old = etime_average[1];
		etime_average[1] = bnc_get_average(etime_array, times + 1);
		etime_stdev = sqrt(bnc_get_variance(etime_array, times + 1));
		if(times <= 0)
		{
			etime_max = etime_array[times];
			etime_min = etime_array[times];
		}
		else
		{
			if(etime_max < etime_array[times]) etime_max = etime_array[times];
			if(etime_min > etime_array[times]) etime_min = etime_array[times];

			if(times >= 5)
			{
				if((fabs((etime_average[1] - etime_average_old) / etime_average[1]) <= 0.001) || (etime_array[times] >= maxsec))
				{
					//printf("average seconds: %g\n", etime_average);
					//printf("max     seconds: %g\n", etime_max);
					//printf("min     seconds: %g\n", etime_min);
					//printf("stdev          : %g\n", etime_stdev);
					end_flag = 1;
				}
			}
		}
		// exceed max. comp. time
		//if(etime > maxsec)
		if(end_flag == 1)
		{
			break;
		}
	}
	//printf("Block(%4ld)    %5d %11.4g %11.4g %11.4g %11.4g\n", block_min_dim, num_threads, etime_average[1], etime_max, etime_min, etime_stdev);
	printf("Block(%4ld)    %5d %11.4g %11.4g %11.4g %11.4g ", block_min_dim, num_threads, etime_average[1], etime_max, etime_min, etime_stdev);
	rdd_out_str_base(stdout, 10, 3, relerr[0]);
	printf("\n");


	// Strassen(strassen_min_dim)
	etime_average[2] = 0.0;
	for(times = 0; times < trial_times; times++)
	{
		end_flag = 0;

		set0_ddmatrix(mat_c);

		iteration = 1.0;
		do{
			stime = get_real_secv();

			//*num_addsub = mat_a->row_dim * mat_a->col_dim * mat_b->row_dim;
			//*num_mul = mat_a->row_dim * mat_a->col_dim * mat_b->row_dim;

#ifdef USE_OMP
			for(i = 0; i < iteration; i++) _bncomp_mul_ddmatrix_strassen(mat_c, mat_a, mat_b, strassen_min_dim);
#else // USE_OMP
			for(i = 0; i < iteration; i++) mul_ddmatrix_strassen(mat_c, mat_a, mat_b, strassen_min_dim);
#endif // USE_OMP

			etime_array[times] = get_real_secv() - stime;
			iteration *= 2.0;
		} while(etime_array[times] < 2);

		// get relative errors
		relerr3_ddmatrix(relerr[0], relerr[1], relerr[2], mat_c, mat_c_long, 0);

		etime_array[times] /= iteration / 2;

		etime_average_old = etime_average[2];
		etime_average[2] = bnc_get_average(etime_array, times + 1);
		etime_stdev = sqrt(bnc_get_variance(etime_array, times + 1));
		if(times <= 0)
		{
			etime_max = etime_array[times];
			etime_min = etime_array[times];
		}
		else
		{
			if(etime_max < etime_array[times]) etime_max = etime_array[times];
			if(etime_min > etime_array[times]) etime_min = etime_array[times];

			if(times >= 5)
			{
				if((fabs((etime_average[2] - etime_average_old) / etime_average[2]) <= 0.001) || (etime_array[times] >= maxsec))
				{
					//printf("average seconds: %g\n", etime_average);
					//printf("max     seconds: %g\n", etime_max);
					//printf("min     seconds: %g\n", etime_min);
					//printf("stdev          : %g\n", etime_stdev);
					end_flag = 1;
				}
			}
		}
		// exceed max. comp. time
		//if(etime > maxsec)
		if(end_flag == 1)
		{
			break;
		}
	}

//	printf("Strassen(%4ld) %5d %11.4g %11.4g %11.4g %11.4g\n", strassen_min_dim, num_threads, etime_average[2], etime_max, etime_min, etime_stdev);
	printf("Strassen(%4ld) %5d %11.4g %11.4g %11.4g %11.4g ", strassen_min_dim, num_threads, etime_average[2], etime_max, etime_min, etime_stdev);
	rdd_out_str_base(stdout, 10, 3, relerr[0]);
	printf("\n");

	printf("                                        block                                                       \n");
	printf("prec  thread(s) row_dim col_dim mid_dim opt_min_dim block_time(s) min.time(s) min.alg (str._min_dim)\n");
	printf("DD %9d %7ld %7ld %7ld %11ld %11.4g", num_threads, row_dim, col_dim, mid_dim, block_min_dim, etime_average[1]);
	if((opt_fp != NULL) && (opt_fp != stdout))
	{
		if(flag_opt_print_header == 1)
		{
			fprintf(opt_fp, "                                        block                                                       \n");
			fprintf(opt_fp, "prec  thread(s) row_dim col_dim mid_dim opt_min_dim block_time(s) min.time(s) min.alg (str._min_dim)\n");
		}
		fprintf(opt_fp, "DD %9d %7ld %7ld %7ld %11ld %11.4g", num_threads, row_dim, col_dim, mid_dim, block_min_dim, etime_average[1]);
	}

	bnc_get_min(&min_i, etime_average, 3);

	printf("%11.4g ", etime_average[min_i]);
	if((opt_fp != NULL) && (opt_fp != stdout))
		fprintf(opt_fp, "%11.4g ", etime_average[min_i]);

	switch(min_i)
	{
		case 0: // Simple
			printf("simple\n");
			if((opt_fp != NULL) && (opt_fp != stdout))
				fprintf(opt_fp, "simple\n");
			break;

		case 1: // block
			printf("block%7ld\n", block_min_dim);
			if((opt_fp != NULL) && (opt_fp != stdout))
				fprintf(opt_fp, "block%7ld\n", block_min_dim);
			break;

		case 2: // Strassen
			printf("Strassen%7ld\n", strassen_min_dim);
			if((opt_fp != NULL) && (opt_fp != stdout))
				fprintf(opt_fp, "Strassen%7ld\n", strassen_min_dim);
			break;

		default: // ?
			printf("    ?\n");
			if((opt_fp != NULL) && (opt_fp != stdout))
				fprintf(opt_fp, "    ?\n");
			break;
	}

	free_ddmatrix(mat_a);
	free_ddmatrix(mat_b);
	free_ddmatrix(mat_c);
	free_ddmatrix(mat_c_long);

	free(etime_array);

	return;
}

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
int short_mmbench_kernel_dd(double etime_array[], double *relerr[3], long int *num_addsub, long int *num_mul, long int mat_a_row_dim, long int mat_a_col_dim, long int mat_b_row_dim, long int mat_b_col_dim, long int mat_c_row_dim, long int mat_c_col_dim, long int min_dim, int trial_times, long int maxsec)
{
	long int i, j, end_flag;
	int times;
	double iteration, stime, etime_average, etime_stdev, etime_average_old, etime_max, etime_min;
	DDMatrix mat_a, mat_b, mat_c, mat_c_long;
	double tmp[DDSIZE];

	mat_a = init_ddmatrix(mat_a_row_dim, mat_a_col_dim);
	mat_b = init_ddmatrix(mat_b_row_dim, mat_b_col_dim);
	mat_c = init_ddmatrix(mat_c_row_dim, mat_c_col_dim);
	mat_c_long = init_ddmatrix(mat_c_row_dim, mat_c_col_dim);

	// set mat_a
	for(i = 0; i < mat_a_row_dim; i++)
	{
		for(j = 0; j < mat_a_col_dim; j++)
		{
			rdd_sqrt_ui(tmp, 5UL);
			rdd_mul_ui(tmp, tmp, (unsigned long)(i + j + 1));
			set_ddmatrix_ij(mat_a, i, j, tmp);
		}
	}

	// set_mat_b
	for(i = 0; i < mat_b_row_dim; i++)
	{
		for(j = 0; j < mat_b_col_dim; j++)
		{
			rdd_sqrt_ui(tmp, 3UL);
			rdd_mul_ui(tmp, tmp, (unsigned long)(mat_b_row_dim - i));
			set_ddmatrix_ij(mat_b, i, j, tmp);
		}
	}

#ifdef USE_OMP
	//mul_ddmatrix_strassen(mat_c_long, mat_a, mat_b, _BNC_DEFAULT_MIN_DIM_STRASSEN);
	_bncomp_mul_ddmatrix_strassen(mat_c_long, mat_a, mat_b, _BNC_DEFAULT_MIN_DIM_STRASSEN);
#else // USE_OMP
//	mul_ddmatrix_strassen(mat_c_long, mat_a, mat_b, _BNC_DEFAULT_MIN_DIM_STRASSEN);
//	mul_ddmatrix_simple(mat_c_long, mat_a, mat_b);
#endif // USE_OMP

	for(times = 0; times < trial_times; times++)
	{
		end_flag = 0;

		set0_ddmatrix(mat_c);

		iteration = 1.0;
		do{
			stime = get_real_secv();

			*num_addsub = mat_a->row_dim * mat_a->col_dim * mat_b->row_dim;
			*num_mul = mat_a->row_dim * mat_a->col_dim * mat_b->row_dim;

#ifdef USE_OMP
			for(i = 0; i < iteration; i++) _bncomp_mul_ddmatrix_block(mat_c, mat_a, mat_b, min_dim);
#else // USE_OMP
			for(i = 0; i < iteration; i++) mul_ddmatrix_block(mat_c, mat_a, mat_b, min_dim);
#endif // USE_OMP

			etime_array[times] = get_real_secv() - stime;
			iteration *= 2.0;
		} while(etime_array[times] < 2);

		// get relative errors
		//relerr3_ddmatrix(relerr[0], relerr[1], relerr[2], mat_c, mat_c_long, 0);

		etime_array[times] /= iteration / 2;

		etime_average_old = etime_average;
		etime_average = bnc_get_average(etime_array, times + 1);
		etime_stdev = sqrt(bnc_get_variance(etime_array, times + 1));
		if(times <= 0)
		{
			etime_max = etime_array[times];
			etime_min = etime_array[times];
		}
		else
		{
			if(etime_max < etime_array[times]) etime_max = etime_array[times];
			if(etime_min > etime_array[times]) etime_min = etime_array[times];

			if(times >= 5)
			{
				if((fabs((etime_average - etime_average_old) / etime_average) <= 0.001) || (etime_array[times] >= maxsec))
				{
				//	printf("average seconds: %g\n", etime_average);
				//	printf("max     seconds: %g\n", etime_max);
				//	printf("min     seconds: %g\n", etime_min);
				//	printf("stdev          : %g\n", etime_stdev);
					end_flag = 1;
				}
			}
		}

		// exceed max. comp. time
		//if(etime > maxsec)
		if(end_flag == 1)
		{
			break;
		}
	}

	free_ddmatrix(mat_a);
	free_ddmatrix(mat_b);
	free_ddmatrix(mat_c);
	free_ddmatrix(mat_c_long);

	return times;
}


//#define NUM_MIN_DIM 5 // #min_dim
//#define NUM_MIN_DIM 6 // #min_dim
#define NUM_MIN_DIM 7 // #min_dim
long int short_mmbench_dd(long int row_dim, long int col_dim, long int mid_dim, int trial_times, long int maxsec, int num_threads)
{
	//int num_min_dim = 5;
	long int num_row_div, mat_a_row_dim, mat_a_col_dim, mat_b_row_dim, mat_b_col_dim, mat_c_row_dim, mat_c_col_dim;
	long int i, j, end_flag, min_i;
	long int num_addsub, num_mul;
	double iteration, stime, mflops, etime_average, etime_stdev, etime_average_old, etime_max, etime_min, estimated_time[NUM_MIN_DIM];
	double relerr[NUM_MIN_DIM][3][DDSIZE];
	double *etime[NUM_MIN_DIM];
	int times[NUM_MIN_DIM], upper_count = 0;
	long int min_dim[NUM_MIN_DIM];

	for(i = 0; i < NUM_MIN_DIM; i++)
	{
		etime[i] = (double *)malloc(sizeof(double) * trial_times);
	}

	// min_dim = 8, 16, 32, 64, 128
	min_dim[0] = 8;
	min_dim[1] = 16;
	min_dim[2] = 32;
	min_dim[3] = 64;
	min_dim[4] = 128;
	min_dim[5] = 256;
	min_dim[6] = 512;

	num_row_div = 2;

	if((row_dim < min_dim[0]) || (col_dim < min_dim[0]) || (mid_dim < min_dim[0]))
	{
		printf("Warning: row_dim, col_dim, mid_dim = %ld, %ld, %ld is not target to use block algorithm(min_block_size = %ld)!\n", row_dim, col_dim, mid_dim, min_dim[0]);
		return 0;
	}

	printf("[DD bits] (mat_a_dim x %5ld) * (%5ld x %5ld) -> (%5ld x %5ld) * (%5ld x %5ld) \n", mid_dim, mid_dim, col_dim, row_dim, mid_dim, mid_dim, col_dim);
	printf("[DD bits] [min_dim, mat_a_dim] seconds -> seconds (estimated)\n");

	upper_count = 0;
	for(i = 0; i < NUM_MIN_DIM; i++)
	{
		if(min_dim[i] > row_dim) break;

		mat_a_row_dim = (((min_dim[i] * num_row_div) >= row_dim) ? row_dim : (min_dim[i] * num_row_div));
		mat_a_col_dim = mid_dim;
		mat_b_row_dim = mid_dim;
		mat_b_col_dim = col_dim;
		mat_c_row_dim = mat_a_row_dim;
		mat_c_col_dim = col_dim;

		times[i] = short_mmbench_kernel_dd(etime[i], relerr[i], &num_addsub, &num_mul, mat_a_row_dim, mat_a_col_dim, mat_b_row_dim, mat_b_col_dim, mat_c_row_dim, mat_c_col_dim, min_dim[i], trial_times, maxsec);

		etime_average = bnc_get_average(etime[i], (long int)times[i]);
		etime_stdev = sqrt(bnc_get_variance(etime[i], (long int)times[i]));

		mflops = (double)(num_addsub + num_mul) / etime_average / 1000.0 / 1000.0;
		estimated_time[i] = ceil((double)(row_dim) / (double)(min_dim[i])) * etime_average / ((double)mat_a_row_dim / (double)min_dim[i]);

		//printf("%5d [%5d bits] [min_dim = %3ld] (%5d x %5d) * (%5d x %5d): %9.5f sec (%8.5f micro-sec/mul) %f MFlops", times[i], prec, min_dim[i], mat_a_row_dim, mat_a_col_dim, mat_b_row_dim, mat_b_col_dim, etime_average, (etime_average / (double)num_mul) * 1000 * 1000, mflops);
		printf("[DD bits] [%3ld, %5ld] ", min_dim[i], mat_a_row_dim);
		//printf(" "); rdd_out_str_base(stdout, 10, 3, relerr[i][0]);
		//printf(" "); rdd_out_str_base(stdout, 10, 3, relerr[i][1]);
		//printf(" "); rdd_out_str_base(stdout, 10, 3, relerr[i][2]);
		printf("%10.3g -> %10.3g\n", etime_average, estimated_time[i]);
		//printf("%9.5lf -> %9.5lf\n", etime_average, ceil((double)(row_dim) / (double)(mat_a_row_dim)) * etime_average);
//		printf("\n");
//		printf("    [%5ld bits] (%5d x %5d) * (%5d x %5d): %9.5f sec (estimated)\n", prec, mat_a_col_dim, mat_a_col_dim, mat_b_row_dim, mat_b_col_dim, ceil((double)(mat_a_col_dim) / (double)(mat_a_row_dim)) * etime_average);

		// more computational time ?
		if(i > 0)
			if(estimated_time[i] >= estimated_time[i - 1])
				upper_count++;

		if(upper_count >= 2)
			break;

	}

	bnc_get_min(&min_i, estimated_time, i);
	//printf("%d thread(s) (%ld x %ld) = (%ld x %ld) * (%ld * %ld) : %g s, min_dim = %ld\n", num_threads, row_dim, col_dim, row_dim, mid_dim, mid_dim, col_dim, bnc_get_min(&min_i, estimated_time, i), min_dim[min_i]);
	printf("prec  thread(s) row_dim col_dim mid_dim opt_min_dim est.time(s)\n");
	printf("DD   %9d %7ld %7ld %7ld %11ld %11.4g\n", num_threads, row_dim, col_dim, mid_dim, min_dim[min_i], estimated_time[min_i]);

	for(i = 0; i < NUM_MIN_DIM; i++)
	{
		free(etime[i]);
	}

	return min_dim[min_i];
}
//#endif // USE_DDLINEAR

// usage
void usage(const char *commandname)
{
#ifdef USE_OMP
	printf("USAGE-> %s row_dim col_dim mid_dim [num_threads] [Maximum seconds] [opt_file_name]\n", commandname);
#else
	printf("USAGE-> %s row_dim col_dim mid_dim [Maximum seconds] [opt_file_name]\n", commandname);
#endif
}

int main(int argc, char *argv[])
{
	int num_threads = 1;
	long int row_dim, col_dim, mid_dim, block_min_dim, strassen_min_dim;
	long int maxsec = 600;
	FILE *fp = NULL;
	//char fname[256] = "";

	if(argc < 4)
	{
		usage(argv[0]);
		return 0;
	}
	else
	{
		row_dim = atol(argv[1]);
		col_dim = atol(argv[2]);
		mid_dim = atol(argv[3]);

#ifdef USE_OMP 
		if(argc >= 5)
		{
			num_threads = atol(argv[4]);
			if(num_threads <= 0)
			{
				printf("Warning: #num_threads is invalid (%d)\n", num_threads);
				num_threads = 1;
			}
			if(argc >= 6)
			{
				maxsec = atol(argv[5]);
				if(argc >= 7)
				{
					//strcpy(fname, argv[6]);
					fp = fopen(argv[6], "a");
					if(fp == NULL)
					{
						printf("Warning: %s cannot be opened!\n", argv[6]);
					}
					else
						printf("%s opens!\n", argv[6]);
				}
			}
		}
#else // USE_OMP
		if(argc >= 5)
		{
			maxsec = atol(argv[4]);
			if(argc >= 6)
			{
				//strcpy(argv[5], fname);
				fp = fopen(argv[5], "a");
				if(fp == NULL)
				{
					printf("Warning: %s cannot be opened!\n", argv[5]);
				}
				else
					printf("%s opens!\n", argv[5]);
			}
		}
#endif // USE_OMP
	}

	// Initialize QD library
	fpu_fix_start(NULL);


#ifdef USE_OMP
	set_bncomp_num_threads(num_threads);
	printf("OpenMP #threads = %d\n", omp_get_num_threads());
#endif // USE_OMP

	printf("------------------ Benchmark Test using Matrix Multiplication -----------------\n");

	block_min_dim = short_mmbench_dd(row_dim, col_dim, mid_dim, 10, maxsec, num_threads);
#ifdef USE_OMP
	get_opt_mmbench_dd(fp, 0, row_dim, col_dim, mid_dim, block_min_dim, 32, 5, maxsec, num_threads);
#else // USE_OPT
	get_opt_mmbench_dd(fp, 1, row_dim, col_dim, mid_dim, block_min_dim, 32, 5, maxsec, num_threads);
#endif // USE_OPT

	printf("------------------------------------ END --------------------------------------\n");

	if(fp != NULL)
		fclose(fp);

	return block_min_dim;
}

