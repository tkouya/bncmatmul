/********************************************************************************/
/* mpi_dka.c:                                                                   */
/* Copyright (C) 2003-2011 Tomonori Kouya, All rights reserved.                 */
/*                                                                              */
/* Version 0.0: 2003.08/21                                                      */
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
/*************************************************/
/* mpi_dka.c: Durand-Kerner-Aberth Methods       */
/*************************************************/
#include <stdio.h>
#include <math.h>

#include "mpi.h"
#include "bnc.h"
#include "mpi_gmp.h"
#include "mpi_bnc.h"

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void _mpi_ddka_init(CDArray local_x_init, DPoly func, MPI_Comm comm)
{
	int num_procs, myrank;

	long int i, index, itmp, local_dim, d_dim[MPI_GMP_MAXPROCS];
	double rad, cen, an, tmp, re_cinit, im_cinit;
	DCmplx cinit;

	MPI_Comm_rank(comm, &myrank);
	MPI_Comm_size(comm, &num_procs);

	local_dim = _mpi_divide_dim(d_dim, func->deg, num_procs);

	rad = ddka_radius(func);
	cen = ddka_center(func);

//	printf("%f, %f\n", rad, cen);

	cinit = init_dcmplx();
	set0_dcmplx(cinit);
	for(i = 0; i < local_dim; i++)
		set_cdarray_i(local_x_init, i, cinit);

	for(i = 0; i < d_dim[myrank]; i++)
	{
		index = myrank * local_dim + i;

		set0_dcmplx(cinit);
		tmp = (double)(2.0 * M_PI * index / func->deg + 3.0 / (2.0 * func->deg));
		iexp_dcmplx(cinit, tmp);
		re_cinit = get_real_dcmplx(cinit);
		im_cinit = get_image_dcmplx(cinit);

		re_cinit = cen + rad * re_cinit;
		im_cinit = rad * im_cinit;

		set_real_dcmplx(cinit, re_cinit);
		set_image_dcmplx(cinit, im_cinit);

//		printf("%5d(%f) ", i, abs_dcmplx(cinit)); print_dcmplx(cinit);

		set_cdarray_i(local_x_init, i, cinit);
	}
}

/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
void _mpi_ddka(long int *lasttimes, CDArray ans, CDArray local_ans, CDArray x_init, CDArray local_x_init, DPoly func, long int maxtimes, double abs_eps, double rel_eps, MPI_Comm comm)
{
	int num_procs, myrank, cdarray_count;

	long int times, i, j, index, deg, flag, local_flag, local_dim, d_dim[MPI_GMP_MAXPROCS];
	double absmodval, abs_x, abs_newx;
	DCmplx modval, up_modval, low_modval, tmp;
	double tmp_darray[2];
	void *local_buf, *buf;

	MPI_Comm_rank(comm, &myrank);
	MPI_Comm_size(comm, &num_procs);

	deg = ans->size;

	modval = init_dcmplx();
	low_modval = init_dcmplx();
	up_modval = init_dcmplx();
	tmp = init_dcmplx();

	local_dim = _mpi_divide_dim(d_dim, deg, num_procs);
	local_buf = allocbuf_dcmplx(local_dim);
	buf = allocbuf_dcmplx(local_dim * num_procs);

	for(times = 0; times <= maxtimes; times++)
	{

		local_flag = 0; flag = 0;

		cdarray_count = pack_cdarray(local_x_init, local_buf);
		MPI_Allgather(local_buf, cdarray_count, MPI_DOUBLE, buf, cdarray_count, MPI_DOUBLE, comm);
		unpack_cdarray(buf, x_init, deg);

/*		if(myrank == 0)
		{
			printf("\n%5d: local_x_init->size = %d, x_init->size = %d, deg = %d\n", times, local_x_init->size, x_init->size, deg); print_cdarray(x_init);
		}
*/
		for(i = 0; i < d_dim[myrank]; i++)
		{
			index = myrank * local_dim + i;
			set_real_dcmplx(low_modval, 1.0);
			set_image_dcmplx(low_modval, 0.0);
			for(j = 0; j < index; j++)
			{
				set0_dcmplx(tmp);
				sub_dcmplx(
					tmp,
					get_cdarray_i(x_init, index),
					get_cdarray_i(x_init, j)
				);
				mul2_dcmplx(low_modval, tmp);
			}
			for(j = index + 1; j < deg; j++)
			{
				set0_dcmplx(tmp);
				sub_dcmplx(
					tmp,
					get_cdarray_i(x_init, index),
					get_cdarray_i(x_init, j)
				);
				mul2_dcmplx(low_modval, tmp);
			}
			mul_dcmplx_d(low_modval, low_modval, get_dpoly_i(func, func->deg));
			ceval_dpoly(up_modval, func, get_cdarray_i(local_x_init, i));
			div_dcmplx(modval, up_modval, low_modval);
			sub_dcmplx(tmp, get_cdarray_i(local_x_init, i), modval);			set_cdarray_i(local_ans, i, tmp);
/*			if(myrank==0)
			{
				printf("up_modval ->");print_dcmplx(up_modval);
				printf("low_modval->");print_dcmplx(low_modval);
				printf("modval    ->");print_dcmplx(modval);
				printf("tmp       ->");print_dcmplx(tmp);
			}
*/
			/* check convergence */
			absmodval = abs_dcmplx(modval);
			abs_x = abs_dcmplx(get_cdarray_i(local_x_init, i));
			abs_newx = abs_dcmplx(get_cdarray_i(local_ans, i));
			if( absmodval > (abs_x + abs_newx) * rel_eps + abs_eps)
				local_flag += 1;

		}

		/* check convergence */
		MPI_Allreduce(&local_flag, &flag, 1, MPI_LONG, MPI_MAX, comm);
		if(myrank == 0)
//			printf("times = %d, flag = %d\n", times, flag);
		if(flag == 0)
			break;

		subst_cdarray(local_x_init, local_ans);
//		MPI_Barrier(comm);
	}

	*lasttimes = times;
	return;
}

#ifdef USE_GMP

/* Aberth's initial value */
/* (max (m*|a_i / a_deg|)^(1/(deg-i)) */
void _mpi_mpf_dka_init(CMPFArray local_x_init, MPFPoly func, MPI_Comm comm)
{
	int myrank, num_procs;
	long int i, itmp, index, local_dim, d_dim[MPI_GMP_MAXPROCS];
	mpf_t rad, cen, an, tmp, re_cinit, im_cinit;
	MPFCmplx cinit;

	MPI_Comm_rank(comm, &myrank);
	MPI_Comm_size(comm, &num_procs);

	local_dim = _mpi_divide_dim(d_dim, func->deg, num_procs);

	mpf_init2(rad, local_x_init->prec);
	mpf_init2(cen, local_x_init->prec);
	mpf_init2(an , local_x_init->prec);
	mpf_init2(tmp, local_x_init->prec);
	mpf_init2(re_cinit, local_x_init->prec);
	mpf_init2(im_cinit, local_x_init->prec);

	mpf_dka_radius(rad, func);
	mpf_dka_center(cen, func);

//	mpf_out_str(stdout, 10, 0, rad); printf(", "); mpf_out_str(stdout, 10, 0, cen); printf("\n");

	cinit = init_mpfcmplx();
	set0_mpfcmplx(cinit);
	for(i = 0; i < local_dim; i++)
		set_cmpfarray_i(local_x_init, i, cinit);

	for(i = 0; i < d_dim[myrank]; i++)
	{
		index = myrank * local_dim + i;

		set0_mpfcmplx(cinit);
#ifndef USE_MPFR
		mpf_set_d(tmp, (double)(2.0 * M_PI * index / func->deg + 3.0 / (2.0 * func->deg)));
#else
		mpfr_set_d(tmp, (double)(2.0 * M_PI * index / func->deg + 3.0 / (2.0 * func->deg)), bnc_default_rounding_mode);
#endif
		iexp_mpfcmplx(cinit, tmp);
		get_real_mpfcmplx(re_cinit, cinit);
		get_image_mpfcmplx(im_cinit, cinit);

		/* re_cinit = cen + rad * re_cinit */
		mpf_mul(re_cinit, rad, re_cinit);
		mpf_add(re_cinit, cen, re_cinit);

		/* im_cinit = rad * im_cinit */
		mpf_mul(im_cinit, rad, im_cinit);

		set_real_mpfcmplx(cinit, re_cinit);
		set_image_mpfcmplx(cinit, im_cinit);

		abs_mpfcmplx(tmp, cinit);
//		printf("%5d(%f) ", i, mpf2double(tmp)); print_mpfcmplx(cinit);

		set_cmpfarray_i(local_x_init, i, cinit);
	}

	mpf_clear(rad);
	mpf_clear(cen);
	mpf_clear(an);
	mpf_clear(tmp);
	mpf_clear(re_cinit);
	mpf_clear(im_cinit);
}

/* DKA Method : 2nd order */
/* f(x) = 0 -> x_new := x_n - func(x) / func'(x) */
void _mpi_mpf_dka(long int *lasttimes, CMPFArray ans, CMPFArray local_ans, CMPFArray x_init, CMPFArray local_x_init, MPFPoly func, long int maxtimes, mpf_t abs_eps, mpf_t rel_eps, MPI_Comm comm)
{
	int myrank, num_procs, carray_count;

	long int i, j, index, deg, flag, times, local_flag, local_dim, d_dim[MPI_GMP_MAXPROCS];
	mpf_t absmodval, abs_x, abs_newx, mpftmp;
	MPFCmplx modval, up_modval, low_modval, tmp;
	void *local_buf, *buf;

	MPI_Comm_rank(comm, &myrank);
	MPI_Comm_size(comm, &num_procs);

	deg = ans->size;

	mpf_init2(absmodval, ans->prec);
	mpf_init2(abs_x, ans->prec);
	mpf_init2(abs_newx, ans->prec);
	mpf_init2(mpftmp, ans->prec);

	modval = init_mpfcmplx();
	low_modval = init_mpfcmplx();
	up_modval = init_mpfcmplx();
	tmp = init_mpfcmplx();

	local_dim = _mpi_divide_dim(d_dim, deg, num_procs);
	local_buf = allocbuf_mpfcmplx(local_x_init->prec, local_dim);
	buf = allocbuf_mpfcmplx(local_x_init->prec, local_dim * num_procs);

	for(times = 0; times <= maxtimes; times++)
	{

		local_flag = 0; flag = 0;

		carray_count = pack_cmpfarray(local_x_init, local_buf);
		MPI_Allgather(local_buf, carray_count, MPI_BNC_MPFCMPLX, buf, carray_count, MPI_BNC_MPFCMPLX, comm);
		unpack_cmpfarray(buf, x_init, deg);

		for(i = 0; i < d_dim[myrank]; i++)
		{
			index = myrank * local_dim + i;
			set_real_mpfcmplx_ui(low_modval, 1UL);
			set_image_mpfcmplx_ui(low_modval, 0UL);
			for(j = 0; j < index; j++)
			{
				set0_mpfcmplx(tmp);
				sub_mpfcmplx(
					tmp,
					get_cmpfarray_i(x_init, index),
					get_cmpfarray_i(x_init, j)
				);
				mul2_mpfcmplx(low_modval, tmp);
			}
			for(j = index + 1; j < deg; j++)
			{
				set0_mpfcmplx(tmp);
				sub_mpfcmplx(
					tmp,
					get_cmpfarray_i(x_init, index),
					get_cmpfarray_i(x_init, j)
				);
				mul2_mpfcmplx(low_modval, tmp);
			}
			mul_mpfcmplx_mpf(low_modval, low_modval, get_mpfpoly_i(func, func->deg));
			ceval_mpfpoly(up_modval, func, get_cmpfarray_i(local_x_init, i));

			div_mpfcmplx(modval, up_modval, low_modval);
/*			if(myrank==0)
			{
				printf("up_modval ->");print_mpfcmplx(up_modval);
				printf("low_modval->");print_mpfcmplx(low_modval);
				printf("modval    ->");print_mpfcmplx(modval);
				printf("tmp       ->");print_mpfcmplx(tmp);
			}
*/
			sub_mpfcmplx(tmp, get_cmpfarray_i(local_x_init, i), modval);
			set_cmpfarray_i(local_ans, i, tmp);

//			get_real_mpfcmplx(mpftmp, get_cmpfarray_i(local_ans, i)); printf("%25.17e, ", mpf2double(mpftmp));
			/* check convergence */
			abs_mpfcmplx(absmodval, modval);
			abs_mpfcmplx(abs_x, get_cmpfarray_i(local_x_init, i));
			abs_mpfcmplx(abs_newx, get_cmpfarray_i(local_ans, i));

			mpf_add(mpftmp, abs_x, abs_newx);
			mpf_mul(mpftmp, mpftmp, rel_eps);
			mpf_add(mpftmp, mpftmp, abs_eps);
			if( mpf_cmp(absmodval, mpftmp) > 0 )
				local_flag += 1;

		}

		/* check convergence */
		MPI_Allreduce(&local_flag, &flag, 1, MPI_LONG, MPI_MAX, comm);
		if(flag == 0)
			break;

		subst_cmpfarray(local_x_init, local_ans);

	}

	mpf_clear(absmodval);
	mpf_clear(abs_x);
	mpf_clear(abs_newx);
	mpf_clear(mpftmp);

//	printf("ans->prec: %d\n", ans->prec);

	free(buf); free(local_buf);

	*lasttimes = times;
	return;
}

#endif
