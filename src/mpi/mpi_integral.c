/********************************************************************************/
/* mpi_integral.c:                                                              */
/* Copyright (C) 2003-2011 Tomonori Kouya, All rights reserved.                 */
/*                                                                              */
/* Version 0.0: 2003.08/21                                                      */
/* Version 0.1: 2004.03/06                                                      */
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
#include "bnc.h"
#ifndef __MPI_GMP_H
	#include "mpi_gmp.h"
#endif

/* double: Trapezoidal rule */
void _mpi_dtrapezoidal_fs(double *ptr_ret, double x_start, double x_end, double (*func)(double x), long int num_div, MPI_Comm comm)
{
	static double x, h, local_ret;
	static long int i;

	int myrank, num_procs;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	/* set stepsize */
	h = (x_end - x_start) / num_div;

	/* ret := h * (f(a) + f(b))/2 */
	local_ret = 0.0;
	if(myrank == 0)
	{
		*ptr_ret = h * (func(x_start) + func(x_end)) / 2;
	}

	/* ret += sum^{n-1}_{i = 1} f(x_i) */
	for(i = myrank + 1; i < num_div; i += num_procs)
	{
		x = x_start + h * i;
		local_ret += func(x);
	}

	/* reduce */
	local_ret *= h;
//	printf("h = %e\n", h);
//	printf("local_ret: %e\n", local_ret);
	MPI_Reduce(&local_ret, ptr_ret, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

}

/* double: Trapezoidal rule with allreduce*/
void _mpi_dtrapezoidal_fs_all(double *ptr_ret, double x_start, double x_end, double (*func)(double x), long int num_div, MPI_Comm comm)
{
	static double x, h, local_ret, tmp;
	static long int i;

	int myrank, num_procs;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	/* set stepsize */
	h = (x_end - x_start) / num_div;

	/* ret := h * (f(a) + f(b))/2 */
	local_ret = 0.0;
	*ptr_ret = h * (func(x_start) + func(x_end)) / 2;

	/* ret += sum^{n-1}_{i = 1} f(x_i) */
	for(i = myrank + 1; i < num_div; i += num_procs)
	{
		x = x_start + h * i;
		local_ret += func(x);
	}

	/* reduce */
	local_ret *= h;
//	printf("h = %e\n", h);
//	printf("local_ret: %e\n", local_ret);
	tmp = 0.0;
	MPI_Allreduce(&local_ret, &tmp, 1, MPI_DOUBLE, MPI_SUM, comm);
	*ptr_ret += tmp;

}

/* double: Modified trapezoidal rule with allreduce*/
void _mpi_dmtrapezoidal_fs_all(double *ptr_ret, double x_start, double x_end, double (*func)(double x), long int num_div, MPI_Comm comm)
{
	static double x, old_x_start, old_x_end, h, local_ret, tmp, left, right;
	static long int i;

	int myrank, num_procs;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	/* check start point */
	old_x_start = x_start;
	while(isnan(func(x_start)) || isinf(func(x_start)))
	{
//		printf("%25.17e + %25.17e\n", x_start, pow(2.0, -52.0));
		x_start += pow(2.0, -50.0);
	}
//	printf("new_x_start: %25.17e\n", x_start);
	left = (x_start - old_x_start ) * func(x_start);

	/* check end point */
	old_x_end = x_end;
	while(isnan(func(x_end)) || isinf(func(x_end)))
	{
//		printf("%25.17e - %25.17e\n", x_end, pow(2.0, -52.0));
		x_end -= pow(2.0, -50.0);
	}
//	printf("new_x_start: %25.17e\n", x_start);
	right = (old_x_end - x_end) * func(x_end);

	/* set stepsize */
	h = (x_end - x_start) / num_div;

	/* ret := h * (f(a) + f(b))/2 */
	local_ret = 0.0;
	*ptr_ret = h * (func(x_start) + func(x_end)) / 2;

	/* ret += sum^{n-1}_{i = 1} f(x_i) */
	for(i = myrank + 1; i < num_div; i += num_procs)
	{
		x = x_start + h * i;
		local_ret += func(x);
	}

	/* reduce */
	local_ret *= h;
//	printf("h = %e\n", h);
//	printf("local_ret: %e\n", local_ret);
	MPI_Allreduce(&local_ret, &tmp, 1, MPI_DOUBLE, MPI_SUM, comm);
	*ptr_ret += tmp;
//	*ptr_ret += left;
//	*ptr_ret += right;
}


#ifdef USE_GMP
/* Trapezoidal rule */
void _mpi_mpf_trapezoidal_fs(mpf_t ret, mpf_t x_start, mpf_t x_end, void (*func)(mpf_t, mpf_t), long int num_div, MPI_Comm comm, MPI_Datatype mpi_mpf_type)
{
	mpf_t x, h, tmp, local_ret;
	long int i;

	int myrank, num_procs;
	void *local_ret_buf, *ret_buf;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	mpf_init2(x, mpf_get_prec(x_start));
	mpf_init2(h, mpf_get_prec(x_start));
	mpf_init2(tmp, mpf_get_prec(ret));
	mpf_init2(local_ret, mpf_get_prec(ret));

	/* set Stepsize */
	mpf_sub(h, x_end, x_start); mpf_div_ui(h, h, (unsigned long)num_div);

	/* ret := h * (f(a) + f(b)) / 2 */
	func(ret, x_start);
	func(tmp, x_end);
	mpf_add(ret, ret, tmp); mpf_div_ui(ret, ret, 2UL);

	/* set starting and ending values */
	mpf_add(x, x_start, h);

	/* ret += sum^{n-1}_{i=1} f(x_i) */
	mpf_set_ui(local_ret, 0UL);
	for(i = myrank + 1; i < num_div; i += num_procs)
	{
		/* x = x_start + h * i */
		mpf_mul_ui(tmp, h, (unsigned long)i);
		mpf_add(x, x_start, tmp);
		func(tmp, x);
		mpf_add(local_ret, local_ret, tmp);
	}
//	mpf_mul(local_ret, local_ret, h);

//	printf("local_ret = "); mpf_out_str(stdout, 10, 0, local_ret); printf("\n");

	local_ret_buf = allocbuf_mpf(mpf_get_prec(local_ret), 1);
	ret_buf = allocbuf_mpf(mpf_get_prec(ret), 1);

	pack_mpf(local_ret, 1, local_ret_buf);
	pack_mpf(ret, 1, ret_buf);
	MPI_Reduce(local_ret_buf, ret_buf, 1, mpi_mpf_type, MPI_MPF_SUM, 0, comm);
	unpack_mpf(ret_buf, ret, 1);

	/* ret := h * { (f(a) + f(b)) / 2 + sum^{n-1}_{i=1} f(x_i) } */
	if(myrank == 0)
		mpf_mul(ret, ret, h);

	freebuf_mpf(local_ret_buf);
	freebuf_mpf(ret_buf);

	mpf_clear(x);
	mpf_clear(h);
	mpf_clear(local_ret);
	mpf_clear(tmp);
}

/* Trapezoidal rule with allreduce */
void _mpi_mpf_trapezoidal_fs_all(mpf_t ret, mpf_t x_start, mpf_t x_end, void (*func)(mpf_t, mpf_t), long int num_div, MPI_Comm comm)
{
	mpf_t x, h, tmp, local_ret;
	long int i;

	int myrank, num_procs;
	void *local_ret_buf, *ret_buf;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	mpf_init2(x, mpf_get_prec(x_start));
	mpf_init2(h, mpf_get_prec(x_start));
	mpf_init2(tmp, mpf_get_prec(ret));
	mpf_init2(local_ret, mpf_get_prec(ret));

	/* set Stepsize */
	mpf_sub(h, x_end, x_start); mpf_div_ui(h, h, (unsigned long)num_div);

	/* ret := h * (f(a) + f(b)) / 2 */
	func(ret, x_start);
	func(tmp, x_end);
	mpf_add(ret, ret, tmp); mpf_div_ui(ret, ret, 2UL);

	/* set starting and ending values */
	mpf_add(x, x_start, h);

	/* ret += sum^{n-1}_{i=1} f(x_i) */
	mpf_set_ui(local_ret, 0UL);
	for(i = myrank + 1; i < num_div; i += num_procs)
	{
		/* x = x_start + h * i */
		mpf_mul_ui(tmp, h, (unsigned long)i);
		mpf_add(x, x_start, tmp);
		func(tmp, x);
		mpf_add(local_ret, local_ret, tmp);
	}
//	mpf_mul(local_ret, local_ret, h);

//	printf("local_ret = "); mpf_out_str(stdout, 10, 0, local_ret); printf("\n");

	local_ret_buf = allocbuf_mpf(mpf_get_prec(local_ret), 1);
	ret_buf = allocbuf_mpf(mpf_get_prec(ret), 1);

	pack_mpf(local_ret, 1, local_ret_buf);
	pack_mpf(ret, 1, ret_buf);
	MPI_Allreduce(local_ret_buf, ret_buf, 1, MPI_MPF, MPI_MPF_SUM, comm);
	unpack_mpf(ret_buf, tmp, 1);

	/* ret := h * { (f(a) + f(b)) / 2 + sum^{n-1}_{i=1} f(x_i) } */
	mpf_add(ret, ret, tmp);
	mpf_mul(ret, ret, h);

	freebuf_mpf(local_ret_buf);
	freebuf_mpf(ret_buf);

	mpf_clear(x);
	mpf_clear(h);
	mpf_clear(local_ret);
	mpf_clear(tmp);
}

/* Trapezoidal rule with allreduce and end-point checks */
void _mpi_mpf_mtrapezoidal_fs_all(mpf_t ret, mpf_t x_start, mpf_t x_end, void (*func)(mpf_t, mpf_t), long int num_div, MPI_Comm comm)
{
	mpf_t x, h, tmp, old_x_start, old_x_end, eps, local_ret;
	long int i;
	unsigned long prec;
	int myrank, num_procs;
	void *local_ret_buf, *ret_buf;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	prec = mpf_get_prec(ret);
	mpf_init2(x, mpf_get_prec(x_start));
	mpf_init2(h, mpf_get_prec(x_start));
	mpf_init2(tmp, mpf_get_prec(ret));
	mpf_init2(local_ret, mpf_get_prec(ret));
	mpf_init2(eps, mpf_get_prec(ret));
	mpf_init2(old_x_start, mpf_get_prec(x_start));
	mpf_init2(old_x_end, mpf_get_prec(x_end));

	/* check start point */
	mpf_set(old_x_start, x_start);

	mpf_set_ui(tmp, 2UL);
	mpf_ui_div(tmp, 1UL, tmp);
	mpf_power(eps, tmp, prec-1);
//	mpf_out_str(stdout, 10, 0, eps); printf("\n");
	func(tmp, x_start);
	while(mpfr_nan_p(tmp) || mpfr_inf_p(tmp))
	{
//		mpf_out_str(stdout, 10, 0, x_start); printf("\n");
		mpf_add(x_start, x_start, eps);
		func(tmp, x_start);
	}
//	printf("new_x_start: "); mpf_out_str(stdout, 10, 0, x_start); printf("\n");

	/* check end point */
	mpf_set(old_x_end, x_end);

	mpf_set_ui(tmp, 2UL);
	mpf_ui_div(tmp, 1UL, tmp);
	mpf_power(eps, tmp, prec-1);
//	mpf_out_str(stdout, 10, 0, eps); printf("\n");
	func(tmp, x_end);
	while(mpfr_nan_p(tmp) || mpfr_inf_p(tmp))
	{
//		mpf_out_str(stdout, 10, 0, x_end); printf("\n");
		mpf_sub(x_end, x_end, eps);
		func(tmp, x_end);
	}
//	printf("new_x_end: "); mpf_out_str(stdout, 10, 0, x_end); printf("\n");

	/* set Stepsize */
	mpf_sub(h, x_end, x_start); mpf_div_ui(h, h, (unsigned long)num_div);

	/* ret := h * (f(a) + f(b)) / 2 */
	func(ret, x_start);
	func(tmp, x_end);
	mpf_add(ret, ret, tmp); mpf_div_ui(ret, ret, 2UL);

	/* set starting and ending values */
	mpf_add(x, x_start, h);

	/* ret += sum^{n-1}_{i=1} f(x_i) */
	mpf_set_ui(local_ret, 0UL);
	for(i = myrank + 1; i < num_div; i += num_procs)
	{
		/* x = x_start + h * i */
		mpf_mul_ui(tmp, h, (unsigned long)i);
		mpf_add(x, x_start, tmp);
		func(tmp, x);
		mpf_add(local_ret, local_ret, tmp);
	}
//	mpf_mul(local_ret, local_ret, h);

//	printf("local_ret = "); mpf_out_str(stdout, 10, 0, local_ret); printf("\n");

	local_ret_buf = allocbuf_mpf(mpf_get_prec(local_ret), 1);
	ret_buf = allocbuf_mpf(mpf_get_prec(ret), 1);

	pack_mpf(local_ret, 1, local_ret_buf);
	pack_mpf(ret, 1, ret_buf);
	MPI_Allreduce(local_ret_buf, ret_buf, 1, MPI_MPF, MPI_MPF_SUM, comm);
	unpack_mpf(ret_buf, tmp, 1);

	/* ret := h * { (f(a) + f(b)) / 2 + sum^{n-1}_{i=1} f(x_i) } */
	mpf_add(ret, ret, tmp);
	mpf_mul(ret, ret, h);

	freebuf_mpf(local_ret_buf);
	freebuf_mpf(ret_buf);

	mpf_clear(x);
	mpf_clear(h);
	mpf_clear(local_ret);
	mpf_clear(tmp);
	mpf_clear(eps);
	mpf_clear(old_x_start);
	mpf_clear(old_x_end);
}
#endif
