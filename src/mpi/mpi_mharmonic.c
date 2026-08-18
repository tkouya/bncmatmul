/********************************************************************************/
/* test_mharmonic.c: Test Program for Numerical Integration                     */
/* Copyright (C) 2011 Tomonori Kouya                                            */
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
#include <math.h>

#include "mpi.h"
#include "bnc.h"

#ifdef USE_GMP
#include "mpi_gmp.h"
#endif

#include "mpi_bnc.h"

/* harmonic integral with modified trapezoidal rule*/
double _mpi_dmharmonic_integral_1step_old(double x_start, double x_end, double (* func)(double), long int num_stage, double reps, double aeps, long int *conv_flag, MPI_Comm comm)
{
	long int i, j, num_steps;
	double ret, ad;
	double ex_table[128][128];

	num_steps = 0;
	for(i = 0; i < num_stage; i++)
	{
		num_steps += 2;
		_mpi_dmtrapezoidal_fs_all(&ex_table[i][0], x_start, x_end, func, num_steps, comm);
		//printf("%2d %10.3e", i, ex_table[i][0]);
		for(j = 1; j <= i; j++)
		{
			ex_table[i][j] = ex_table[i][j-1];
			ad = ex_table[i][j-1] - ex_table[i-1][j-1];
			ex_table[i][j] +=  ad / ex_harmonic_den(i, j);
			if(fabs(ad) < fabs(ex_table[i][j]) * reps + aeps)
			{
				ret = ex_table[i][j];
				*conv_flag = 1;
				goto end;
			}
			//printf("%10.3e ", ex_table[i][j]);
		}
		//printf("\n");
	}
	/* not converge */
	*conv_flag = 0;
	ret = ex_table[num_stage - 1][num_stage - 1];

end:
	return ret;
}

/* harmonic integral with modified trapezoidal rule*/
double _mpi_dmharmonic_integral_1step(double x_start, double x_end, double (* func)(double), long int num_stage, double reps, double aeps, long int *conv_flag, MPI_Comm comm)
{
	long int i, j, num_steps;
	double ret, ad;
	double ex_table[128][128];
	int num_procs;

	MPI_Comm_size(comm, &num_procs);

	num_steps = 0;
	for(i = 0; i < num_stage; i++)
	{
//		num_steps += (num_procs + 1);
		num_steps += (num_procs);
		_mpi_dmtrapezoidal_fs_all(&ex_table[i][0], x_start, x_end, func, num_steps, comm);
		//printf("%2d %10.3e", i, ex_table[i][0]);
		for(j = 1; j <= i; j++)
		{
			ex_table[i][j] = ex_table[i][j-1];
			ad = ex_table[i][j-1] - ex_table[i-1][j-1];
			ex_table[i][j] +=  ad / ex_harmonic_den(i, j);
			if(fabs(ad) < fabs(ex_table[i][j]) * reps + aeps)
			{
				ret = ex_table[i][j];
				*conv_flag = 1;
				goto end;
			}
			//printf("%10.3e ", ex_table[i][j]);
		}
		//printf("\n");
	}
	/* not converge */
	*conv_flag = 0;
	ret = ex_table[num_stage - 1][num_stage - 1];

end:
	return ret;
}


/* harmonic integral with modified trapezoidal rule*/
void _mpi_dmharmonic_integral(double *ret, double x_start, double x_end, double (* func)(double), long int max_num_stage, MPI_Comm comm)
{
	long int steps, conv_flag, end_flag, times, in_num_stage;
	double tmp, in_h, old_in_h, old_in_x_end, in_x_start, in_x_end;
	int myrank;

	MPI_Comm_rank(comm, &myrank);

	in_h = (x_end - x_start) / 2;
	old_in_h = in_h;
	in_x_start = x_start;
	in_x_end = in_x_start + in_h;
	in_num_stage = max_num_stage / 2;
//	in_num_stage = max_num_stage;

	/* main loop */
	*ret = 0.0;
	times = 0;
	do
	{
		tmp = _mpi_dmharmonic_integral_1step(in_x_start, in_x_end, func, in_num_stage, 1.0e-12, 0.0, &conv_flag, comm);
		if((tmp + *ret) == *ret)
			conv_flag = 1;
		if(conv_flag != 1)
		{
			//printf("Not Convergent! %e->%e\n", in_h, in_h/2);
			old_in_h = in_h;
			in_h /= 2;
			old_in_x_end = in_x_end;
			in_x_end = in_x_start + in_h;
			in_num_stage++; if(in_num_stage >= max_num_stage) in_num_stage = max_num_stage;
			if(in_x_end <= in_x_start)
			{
				in_h = old_in_h;
				while(in_x_end <= in_x_start)
				{
					in_h *= 2;
					in_x_end = in_x_start + in_h;
				}
				*ret += tmp; // give up!
				if(myrank == 0)
				printf("n %5d %10.3e [%25.17e, %10.3e] %5d %10.3e\n", times++, in_h, in_x_start, in_x_end, in_num_stage, tmp);
				in_x_start = old_in_x_end;
				in_x_end = in_x_start + in_h;
			}
		}
		else
		{
			in_num_stage--; if(in_num_stage < max_num_stage / 2) in_num_stage = max_num_stage / 2;
			*ret += tmp;
			if(myrank == 0)
			printf("y %5d %10.3e [%25.17e, %10.3e] %5d %10.3e\n", times++, in_h, in_x_start, in_x_end, in_num_stage, tmp);
			if(in_x_end >= x_end)
				break;

			in_h *= 2;
			in_x_start = in_x_end;
			in_x_end = in_x_start + in_h;
			if(in_x_end > x_end)
			{
				in_x_end = x_end;
				in_h = in_x_end - in_x_start;
			}
		}
	} while(1);

	return;
}


/* Romberg integral with modified trapezoidal rule */
double _mpi_dmromberg_integral_1step(double x_start, double x_end, double (* func)(double), long int num_stage, double reps, double aeps, long int *conv_flag, MPI_Comm comm)
{
	long int i, j, num_steps;
	double ret, ad;
	double ex_table[128][128];

	num_steps = 1;
	for(i = 0; i < num_stage; i++)
	{
		num_steps *= 2;
		_mpi_dmtrapezoidal_fs_all(&ex_table[i][0], x_start, x_end, func, num_steps, comm);
		//printf("%2d %10.3e ", i, ex_table[i][0]);
		for(j = 1; j <= i; j++)
		{
			ex_table[i][j] = ex_table[i][j-1];
			ad = ex_table[i][j-1] - ex_table[i-1][j-1];
			ex_table[i][j] +=  ad / ex_harmonic_den(i, j);
			if(fabs(ad) < fabs(ex_table[i][j]) * reps + aeps)
			{
				ret = ex_table[i][j];
				*conv_flag = 1;
				goto end;
			}
			//printf("%10.3e ", ex_table[i][j]);
		}
		//printf("\n");
	}
	/* not converge */
	*conv_flag = 0;
	ret = ex_table[num_stage - 1][num_stage - 1];

end:
	return ret;
}

/* Romberg integral with modified trapezoidal rule*/
void _mpi_dmromberg_integral(double *ret, double x_start, double x_end, double (* func)(double), long int max_num_stage, MPI_Comm comm)
{
	long int steps, conv_flag, end_flag, times, in_num_stage;
	double tmp, in_h, old_in_h, old_in_x_end, in_x_start, in_x_end;
	int myrank;

	MPI_Comm_rank(comm, &myrank);

	in_h = (x_end - x_start) / 2;
	old_in_h = in_h;
	in_x_start = x_start;
	in_x_end = in_x_start + in_h;
	in_num_stage = max_num_stage / 2;
//	in_num_stage = max_num_stage;

	/* main loop */
	*ret = 0.0;
	times = 0;
	do
	{
		tmp = _mpi_dmromberg_integral_1step(in_x_start, in_x_end, func, in_num_stage, 1.0e-12, 0.0, &conv_flag, comm);
		if((tmp + *ret) == *ret)
			conv_flag = 1;
		if(conv_flag != 1)
		{
//			printf("Not Convergent! %e->%e\n", in_h, in_h/2);
			old_in_h = in_h;
			in_h /= 2;
			old_in_x_end = in_x_end;
			in_x_end = in_x_start + in_h;
			in_num_stage++; if(in_num_stage >= max_num_stage) in_num_stage = max_num_stage;
			if(in_x_end <= in_x_start)
			{
				in_h = old_in_h;
				while(in_x_end <= in_x_start)
				{
					in_h *= 2;
					in_x_end = in_x_start + in_h;
				}
				*ret += tmp; // give up!
				if(myrank == 0)
				printf("n %5d %10.3e [%25.17e, %10.3e] %5d %10.3e\n", times++, in_h, in_x_start, in_x_end, in_num_stage, tmp);
				in_x_start = old_in_x_end;
				in_x_end = in_x_start + in_h;
			}
		}
		else
		{
			in_num_stage--; if(in_num_stage < max_num_stage / 2) in_num_stage = max_num_stage / 2;
			*ret += tmp;
			if(myrank == 0)
			printf("y %5d %10.3e [%25.17e, %10.3e] %5d %10.3e\n", times++, in_h, in_x_start, in_x_end, in_num_stage, tmp);
			if(in_x_end >= x_end)
				break;

			in_h *= 2;
			in_x_start = in_x_end;
			in_x_end = in_x_start + in_h;
			if(in_x_end > x_end)
			{
				in_x_end = x_end;
				in_h = in_x_end - in_x_start;
			}
		}
	} while(1);

	return;
}

#ifdef USE_GMP
/* (stage/step)^2 - 1 */
/* void mpf_ex_harmonic1_den(mpf_t ret, long int istage, long int jstage)
{
	mpf_set_ui(ret, (unsigned long)(1 * (istage + 1)));
	mpf_div_ui(ret, ret, (unsigned long)(1 * (istage - jstage + 1)));
	mpf_mul(ret, ret, ret);
	mpf_sub_ui(ret, ret, 1UL);
}
*/
/* (stage/step)^2 - 1 */
/*void mpf_ex_harmonic2_den(mpf_t ret, long int istage, long int jstage)
{
	mpf_set_ui(ret, (unsigned long)(4 * (istage + 1)));
	mpf_div_ui(ret, ret, (unsigned long)(4 * (istage - jstage + 1)));
	mpf_mul(ret, ret, ret);
	mpf_sub_ui(ret, ret, 1UL);
}
*/
/* (stage/step)^2 - 1 */
/*void mpf_ex_harmonic3_den(mpf_t ret, long int istage, long int jstage)
{
	mpf_set_ui(ret, (unsigned long)(8 * (istage + 1)));
	mpf_div_ui(ret, ret, (unsigned long)(8 * (istage - jstage + 1)));
	mpf_mul(ret, ret, ret);
	mpf_sub_ui(ret, ret, 1UL);
}
*/

/* harmonic integral with modified trapezoidal rule */
void _mpi_mpf_mharmonic_integral_1step_old(mpf_t ret, mpf_t x_start, mpf_t x_end, void (* func)(mpf_t, mpf_t), long int num_stage, mpf_t reps, mpf_t aeps, long int *conv_flag, MPI_Comm comm)
{
	mpf_t tmp, tmp1, ad;
	long int i, j, num_steps;
	mpf_t ex_table[128][128];

	mpf_init2(tmp, mpf_get_prec(ret));
	mpf_init2(tmp1, mpf_get_prec(ret));
	mpf_init2(ad, mpf_get_prec(ret));
	for(i = 0; i < num_stage; i++)
		for(j = 0; j < num_stage; j++)
			mpf_init2(ex_table[i][j], mpf_get_prec(ret));

	num_steps = 0;
	for(i = 0; i < num_stage; i++)
	{
		num_steps += 2;
		_mpi_mpf_mtrapezoidal_fs_all(ex_table[i][0], x_start, x_end, func, num_steps, comm);
		//printf("%2d %10.3e ", i, mpf_get_d(ex_table[i][0]));
		for(j = 1; j <= i; j++)
		{
			mpf_set(ex_table[i][j], ex_table[i][j-1]);
			mpf_ex_harmonic_den(tmp, i, j);
			mpf_sub(ad, ex_table[i][j-1], ex_table[i-1][j-1]);
			mpf_div(tmp, ad, tmp);
			mpf_add(ex_table[i][j], ex_table[i][j], tmp);
			mpf_abs(ad, ad);
			mpf_abs(tmp, ex_table[i][j]);
			mpf_mul(tmp, tmp, reps);
			mpf_add(tmp, tmp, aeps);
			if(mpf_cmp(ad, tmp) < 0)
			{
				mpf_set(ret, ex_table[i][j]);
				*conv_flag = 1;
				goto end;
			}
			//printf("%10.3e ", mpf_get_d(ex_table[i][j]));
		}
		//printf("\n");
	}

	/* not converge */
	*conv_flag = 0;
	mpf_set(ret, ex_table[num_stage - 1][num_stage - 1]);

end:
	mpf_clear(tmp);
	mpf_clear(tmp1);
	for(i = 0; i < num_stage; i++)
		for(j = 0; j < num_stage; j++)
			mpf_clear(ex_table[i][j]);

	return;
}

/* harmonic integral with modified trapezoidal rule */
void _mpi_mpf_mharmonic_integral_1step(mpf_t ret, mpf_t x_start, mpf_t x_end, void (* func)(mpf_t, mpf_t), long int num_stage, mpf_t reps, mpf_t aeps, long int *conv_flag, MPI_Comm comm)
{
	mpf_t tmp, tmp1, ad;
	long int i, j, num_steps;
	mpf_t ex_table[128][128];
	int num_procs;

	MPI_Comm_size(comm, &num_procs);

	mpf_init2(tmp, mpf_get_prec(ret));
	mpf_init2(tmp1, mpf_get_prec(ret));
	mpf_init2(ad, mpf_get_prec(ret));
	for(i = 0; i < num_stage; i++)
		for(j = 0; j < num_stage; j++)
			mpf_init2(ex_table[i][j], mpf_get_prec(ret));

	num_steps = 0;
	for(i = 0; i < num_stage; i++)
	{
//		num_steps += (num_procs + 1);
		num_steps += (num_procs);
		_mpi_mpf_mtrapezoidal_fs_all(ex_table[i][0], x_start, x_end, func, num_steps, comm);
		//printf("%2d %10.3e ", i, mpf_get_d(ex_table[i][0]));
		for(j = 1; j <= i; j++)
		{
			mpf_set(ex_table[i][j], ex_table[i][j-1]);
			mpf_ex_harmonic_den(tmp, i, j);
			mpf_sub(ad, ex_table[i][j-1], ex_table[i-1][j-1]);
			mpf_div(tmp, ad, tmp);
			mpf_add(ex_table[i][j], ex_table[i][j], tmp);
			mpf_abs(ad, ad);
			mpf_abs(tmp, ex_table[i][j]);
			mpf_mul(tmp, tmp, reps);
			mpf_add(tmp, tmp, aeps);
			if(mpf_cmp(ad, tmp) < 0)
			{
				mpf_set(ret, ex_table[i][j]);
				*conv_flag = 1;
				goto end;
			}
			//printf("%10.3e ", mpf_get_d(ex_table[i][j]));
		}
		//printf("\n");
	}

	/* not converge */
	*conv_flag = 0;
	mpf_set(ret, ex_table[num_stage - 1][num_stage - 1]);

end:
	mpf_clear(tmp);
	mpf_clear(tmp1);
	for(i = 0; i < num_stage; i++)
		for(j = 0; j < num_stage; j++)
			mpf_clear(ex_table[i][j]);

	return;
}


/* harmonic integral with modified trapezoidal rule*/
void _mpi_mpf_mharmonic_integral(mpf_t ret, mpf_t x_start, mpf_t x_end, void (* func)(mpf_t, mpf_t), long int max_num_stage, MPI_Comm comm)
{
	long int steps, conv_flag, end_flag, times, in_num_stage;
	unsigned long prec;
	mpf_t tmp, tmp1, in_h, old_in_h, old_in_x_end, in_x_start, in_x_end, reps, aeps;
	int myrank;

	MPI_Comm_rank(comm, &myrank);

	/* init */
	prec = mpf_get_prec(ret);
	mpf_init2(tmp, prec);
	mpf_init2(tmp1, prec);
	mpf_init2(in_h, prec);
	mpf_init2(old_in_h, prec);
	mpf_init2(old_in_x_end, prec);
	mpf_init2(in_x_start, prec);
	mpf_init2(in_x_end, prec);
	mpf_init2(reps, prec); mpf_set_d(reps, 1.0e-15);
	mpf_init2(aeps, prec); mpf_set_d(aeps, 0.0);

	mpf_sub(in_h, x_end, x_start); mpf_div_ui(in_h, in_h, 2UL);
	mpf_set(old_in_h, in_h);
	mpf_set(in_x_start, x_start);
	mpf_add(in_x_end, in_x_start, in_h);
	in_num_stage = max_num_stage / 2;
//	in_num_stage = max_num_stage;

	/* main loop */
	mpf_set_ui(ret, 0UL);
	times = 0;
	do
	{
		_mpi_mpf_mharmonic_integral_1step(tmp, in_x_start, in_x_end, func, in_num_stage, reps, aeps, &conv_flag, comm);
		mpf_add(tmp1, tmp, ret);
		if(mpf_cmp(tmp1, ret) == 0)
			conv_flag = 1;
		if(conv_flag != 1)
		{
//			printf("Not Convergent! %e->%e\n", in_h, in_h/2);
			mpf_set(old_in_h, in_h);
			mpf_div_ui(in_h, in_h, 2UL);
			mpf_set(old_in_x_end, in_x_end);
			mpf_add(in_x_end, in_x_start, in_h);
			in_num_stage++; if(in_num_stage >= max_num_stage) in_num_stage = max_num_stage;
			if(mpf_cmp(in_x_end, in_x_start) <= 0)
			{
				mpf_set(in_h, old_in_h);
				while(mpf_cmp(in_x_end, in_x_start) <= 0)
				{
					mpf_mul_ui(in_h, in_h, 2UL);
					mpf_add(in_x_end, in_x_start, in_h);
				}
				mpf_add(ret, ret, tmp); // give up!
				if(myrank == 0)
				printf("n %5d %10.3e [%25.17e, %10.3e] %5d %10.3e\n", times++, mpf_get_d(in_h), mpf_get_d(in_x_start), mpf_get_d(in_x_end), in_num_stage, mpf_get_d(tmp));
				mpf_set(in_x_start, old_in_x_end);
				mpf_add(in_x_end, in_x_start, in_h);
			}
		}
		else
		{
			in_num_stage--; if(in_num_stage < max_num_stage / 2) in_num_stage = max_num_stage / 2;
			mpf_add(ret, ret, tmp);
			if(myrank == 0)
			printf("y %5d %10.3e [%25.17e, %10.3e] %5d %10.3e\n", times++, mpf_get_d(in_h), mpf_get_d(in_x_start), mpf_get_d(in_x_end), in_num_stage, mpf_get_d(tmp));
			if(mpf_cmp(in_x_end, x_end) >= 0)
				break;

			mpf_mul_ui(in_h, in_h, 2UL);
			mpf_set(in_x_start, in_x_end);
			mpf_add(in_x_end, in_x_start, in_h);
			if(mpf_cmp(in_x_end, x_end) > 0)
			{
				mpf_set(in_x_end, x_end);
				mpf_sub(in_h, in_x_end, in_x_start);
			}
		}
	} while(1);

	mpf_clear(tmp);
	mpf_clear(tmp1);
	mpf_clear(in_h);
	mpf_clear(old_in_h);
	mpf_clear(old_in_x_end);
	mpf_clear(in_x_start);
	mpf_clear(in_x_end);
	mpf_clear(reps);
	mpf_clear(aeps);

	return;
}

/* Romberg integral with modified trapezoidal rule */
void _mpi_mpf_mromberg_integral_1step(mpf_t ret, mpf_t x_start, mpf_t x_end, void (* func)(mpf_t, mpf_t), long int num_stage, mpf_t reps, mpf_t aeps, long int *conv_flag, MPI_Comm comm)
{
	mpf_t tmp, tmp1, ad;
	long int i, j, num_steps;
	mpf_t ex_table[128][128];

	mpf_init2(tmp, mpf_get_prec(ret));
	mpf_init2(tmp1, mpf_get_prec(ret));
	mpf_init2(ad, mpf_get_prec(ret));
	for(i = 0; i < num_stage; i++)
		for(j = 0; j < num_stage; j++)
			mpf_init2(ex_table[i][j], mpf_get_prec(ret));

	num_steps = 1;
	for(i = 0; i < num_stage; i++)
	{
		num_steps *= 2;
		_mpi_mpf_mtrapezoidal_fs_all(ex_table[i][0], x_start, x_end, func, num_steps, comm);
		//printf("%2d %10.3e ", i, mpf_get_d(ex_table[i][0]));
		for(j = 1; j <= i; j++)
		{
			mpf_set(ex_table[i][j], ex_table[i][j-1]);
			mpf_ex_harmonic_den(tmp, i, j);
			mpf_sub(ad, ex_table[i][j-1], ex_table[i-1][j-1]);
			mpf_div(tmp, ad, tmp);
			mpf_add(ex_table[i][j], ex_table[i][j], tmp);
			mpf_abs(ad, ad);
			mpf_abs(tmp, ex_table[i][j]);
			mpf_mul(tmp, tmp, reps);
			mpf_add(tmp, tmp, aeps);
			if(mpf_cmp(ad, tmp) < 0)
			{
				mpf_set(ret, ex_table[i][j]);
				*conv_flag = 1;
				goto end;
			}
			//printf("%10.3e ", mpf_get_d(ex_table[i][j]));
		}
		//printf("\n");
	}

	/* not converge */
	*conv_flag = 0;
	mpf_set(ret, ex_table[num_stage - 1][num_stage - 1]);

end:
	mpf_clear(tmp);
	mpf_clear(tmp1);
	for(i = 0; i < num_stage; i++)
		for(j = 0; j < num_stage; j++)
			mpf_clear(ex_table[i][j]);

	return;
}

/* Romberg integral with modified trapezoidal rule*/
void _mpi_mpf_mromberg_integral(mpf_t ret, mpf_t x_start, mpf_t x_end, void (* func)(mpf_t, mpf_t), long int max_num_stage, MPI_Comm comm)
{
	long int steps, conv_flag, end_flag, times, in_num_stage;
	unsigned long prec;
	mpf_t tmp, tmp1, in_h, old_in_h, old_in_x_end, in_x_start, in_x_end, reps, aeps;
	int myrank;

	MPI_Comm_rank(comm, &myrank);

	/* init */
	prec = mpf_get_prec(ret);
	mpf_init2(tmp, prec);
	mpf_init2(tmp1, prec);
	mpf_init2(in_h, prec);
	mpf_init2(old_in_h, prec);
	mpf_init2(old_in_x_end, prec);
	mpf_init2(in_x_start, prec);
	mpf_init2(in_x_end, prec);
	mpf_init2(reps, prec); mpf_set_d(reps, 1.0e-15);
	mpf_init2(aeps, prec); mpf_set_d(aeps, 0.0);

	mpf_sub(in_h, x_end, x_start); mpf_div_ui(in_h, in_h, 2UL);
	mpf_set(old_in_h, in_h);
	mpf_set(in_x_start, x_start);
	mpf_add(in_x_end, in_x_start, in_h);
	in_num_stage = max_num_stage / 2;
//	in_num_stage = max_num_stage;

	/* main loop */
	mpf_set_ui(ret, 0UL);
	times = 0;
	do
	{
		_mpi_mpf_mromberg_integral_1step(tmp, in_x_start, in_x_end, func, in_num_stage, reps, aeps, &conv_flag, comm);
		mpf_add(tmp1, tmp, ret);
		if(mpf_cmp(tmp1, ret) == 0)
			conv_flag = 1;
		if(conv_flag != 1)
		{
//			printf("Not Convergent! %e->%e\n", in_h, in_h/2);
			mpf_set(old_in_h, in_h);
			mpf_div_ui(in_h, in_h, 2UL);
			mpf_set(old_in_x_end, in_x_end);
			mpf_add(in_x_end, in_x_start, in_h);
			in_num_stage++; if(in_num_stage >= max_num_stage) in_num_stage = max_num_stage;
			if(mpf_cmp(in_x_end, in_x_start) <= 0)
			{
				mpf_set(in_h, old_in_h);
				while(mpf_cmp(in_x_end, in_x_start) <= 0)
				{
					mpf_mul_ui(in_h, in_h, 2UL);
					mpf_add(in_x_end, in_x_start, in_h);
				}
				mpf_add(ret, ret, tmp); // give up!
				if(myrank == 0)
				printf("n %5d %10.3e [%25.17e, %10.3e] %5d %10.3e\n", times++, mpf_get_d(in_h), mpf_get_d(in_x_start), mpf_get_d(in_x_end), in_num_stage, mpf_get_d(tmp));
				mpf_set(in_x_start, old_in_x_end);
				mpf_add(in_x_end, in_x_start, in_h);
			}
		}
		else
		{
			in_num_stage--; if(in_num_stage < max_num_stage / 2) in_num_stage = max_num_stage / 2;
			mpf_add(ret, ret, tmp);
			if(myrank == 0)
			printf("y %5d %10.3e [%25.17e, %10.3e] %5d %10.3e\n", times++, mpf_get_d(in_h), mpf_get_d(in_x_start), mpf_get_d(in_x_end), in_num_stage, mpf_get_d(tmp));
			if(mpf_cmp(in_x_end, x_end) >= 0)
				break;

			mpf_mul_ui(in_h, in_h, 2UL);
			mpf_set(in_x_start, in_x_end);
			mpf_add(in_x_end, in_x_start, in_h);
			if(mpf_cmp(in_x_end, x_end) > 0)
			{
				mpf_set(in_x_end, x_end);
				mpf_sub(in_h, in_x_end, in_x_start);
			}
		}
	} while(1);

	mpf_clear(tmp);
	mpf_clear(tmp1);
	mpf_clear(in_h);
	mpf_clear(old_in_h);
	mpf_clear(old_in_x_end);
	mpf_clear(in_x_start);
	mpf_clear(in_x_end);
	mpf_clear(reps);
	mpf_clear(aeps);

	return;
}
#endif
