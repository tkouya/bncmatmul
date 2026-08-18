/********************************************************************************/
/* mpi_ex_nim.c: Extrapolation NIM Method                                       */
/* Copyright (C) 2011 Tomonori Kouya, All rights reserved.                      */
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

/*                          */
/* NIM Extrapolation Method */
/*                          */
int _mpi_dex_nim_1step_lo(DVector local_y, double x0, DVector local_y0, DVector y0, double h, DMatrix local_mat[], void (* gfunc)(DVector, double, MPI_Comm), double rtol, double atol, MPI_Comm comm)
{
	static long int i, j, k, istep, jstep, local_flag = 0, flag, dim;
	static long istep_l;
	static double ftmp, ftmp1;
	static myrank;

	istep_l = 2;
	dex_nim_in_h = h / (double)(istep_l);
	MPI_Comm_rank(comm, &myrank);

	for(istep = 0; istep < dex_nim_stage; istep++)
	{

/* First, Using Euler Method */
//		func(dex_nim_local_ytmp[0], x0, y0, myrank);
		_mpi_mul_dmatrix_dvec(dex_nim_local_ytmp[0], local_mat, local_y0, y0, comm);
		gfunc(dex_nim_local_ytmp[1], x0, comm);
		add2_dvector(dex_nim_local_ytmp[0], dex_nim_local_ytmp[1]);

		cmul2_dvector(dex_nim_local_ytmp[0], dex_nim_in_h);
		add_dvector(dex_nim_local_y[0][istep], local_y0, dex_nim_local_ytmp[0]);

		subst_dvector(dex_nim_local_ytmp[0], local_y0);
		subst_dvector(dex_nim_local_ytmp[1], dex_nim_local_y[0][istep]);

/* Next, Using Midpoint Rule */
		for(i = 1; i <= istep_l; i++)
		{
			ftmp = x0 + dex_nim_in_h * i;
//			func(dex_nim_local_ytmp[2], ftmp, dex_nim_ytmp[1], myrank);
			_mpi_mul_dmatrix_dvec(dex_nim_local_ytmp[2], local_mat, dex_nim_local_ytmp[1], dex_nim_ytmp[1], comm);
			gfunc(dex_nim_local_ytmp[3], ftmp, comm);
			add2_dvector(dex_nim_local_ytmp[2], dex_nim_local_ytmp[3]);

			cmul2_dvector(dex_nim_local_ytmp[2], dex_nim_in_h);
			for(j = 0; j < local_y->dim; j++)
				sdvi(dex_nim_local_ytmp[2], j, gdvi(dex_nim_local_ytmp[2], j) * 2);
			add_dvector(dex_nim_local_y[istep][0], dex_nim_local_ytmp[0], dex_nim_local_ytmp[2]);
			if(i == istep_l) /* for smoothing */
				subst_dvector(dex_nim_local_ytmp[2], dex_nim_local_ytmp[0]);
			subst_dvector(dex_nim_local_ytmp[0], dex_nim_local_ytmp[1]);
			subst_dvector(dex_nim_local_ytmp[1], dex_nim_local_y[istep][0]);
		}

/* Smoothing */
		add2_dvector(dex_nim_local_y[istep][0], dex_nim_local_ytmp[2]);
		cmul2_dvector(dex_nim_local_ytmp[0], 2.0);
		add2_dvector(dex_nim_local_y[istep][0], dex_nim_local_ytmp[0]);
		cmul2_dvector(dex_nim_local_y[istep][0], 0.25);

/* Extrapolation */
		for(jstep = 1; jstep <= istep; jstep++)
		{
			local_flag = 0;
			flag = 0;
			for(i = 0; i < local_y->dim; i++)
			{
				sdvi(dex_nim_local_ytmp[2], i, gdvi(dex_nim_local_y[istep][jstep-1], i) - gdvi(dex_nim_local_y[istep-1][jstep-1], i));
				sdvi(dex_nim_local_ytmp[2], i, gdvi(dex_nim_local_ytmp[2], i) / ex_nim_den(jstep*2));
				sdvi(dex_nim_local_y[istep][jstep], i, gdvi(dex_nim_local_y[istep][jstep-1], i) + gdvi(dex_nim_local_ytmp[2], i));

//				if(gdvi(dex_nim_y[istep][jstep], i) != gdvi(dex_nim_y[istep][jstep-1], i))
//					flag = 1;
			}

			sub_dvector(dex_nim_local_ytmp[1], dex_nim_local_y[istep][jstep], dex_nim_local_y[istep][jstep-1]);
			ftmp  = normi_dvector(dex_nim_local_ytmp[1]);
			ftmp1 = normi_dvector(dex_nim_local_y[istep][jstep]) * rtol + atol;
			if(ftmp <= ftmp1)
				local_flag = 0;
			else
				local_flag = 1;

			/* convergent check */
			MPI_Allreduce((void *)&local_flag, (void *)&flag, 1, MPI_LONG, MPI_SUM, comm);

			if(flag == 0)
				goto Return;
		}
		istep_l *= 2;
		dex_nim_in_h = h / (double)(istep_l);
	}

Return:
	if(istep == dex_nim_stage)
		istep--;
	if(jstep == dex_nim_stage)
		jstep--;
	subst_dvector(local_y, dex_nim_local_y[istep][jstep]);

	if(flag == 0) /* equal */
		return 0;
//		return istep;
	else		 /* not equal */
		return 1;
}

/*                          */
/* NIM Extrapolation Method */
/*                          */
void _mpi_dex_nim_fs_lo(FILE *fp, double x, DVector y, double x0, DVector y0, long int div_num, DMatrix mat, void (* gfunc)(DVector, double, MPI_Comm), double rtol, double atol, long int stage, MPI_Comm comm)
{
	long int steps, i, num_stage, dim;
	DVector old_y, local_old_y, tmp_y, local_tmp_y, lf_tmp;
	double tmp[3], new_x, old_x, in_h;
	DMatrix local_mat[MPI_GMP_MAXPROCS];
	int myrank, procs;

	/* init */
	MPI_Comm_rank(comm, &myrank);
	MPI_Comm_size(comm, &procs);
	dim = y->dim;

	if(stage + 1 > MPI_MAX_DEX_NIM_STAGE)
	{
		fprintf(stderr, "_mpi_dex_nim_fs: Number of stages exceed.\n");
		return;
	}
//	if(myrank == 0)
//		print_dmatrix(mat);
	_mpi_init_dex_nim(stage + 1, dim, comm);
	_mpi_init_dmatrix(local_mat, dex_nim_ddim, dim, comm);
	_mpi_divide_dmatrix(local_mat, dex_nim_ddim, mat, comm);

	/* set */
	old_x = x0;

	tmp_y = init_dvector(dex_nim_local_dim * procs);
	old_y = init_dvector(dex_nim_local_dim * procs);

	local_old_y = _mpi_init_dvector(dex_nim_ddim, dim, comm);
	local_tmp_y = _mpi_init_dvector(dex_nim_ddim, dim, comm);

	subst_dvector(old_y, y0);
	_mpi_divide_dvector(local_old_y, dex_nim_ddim, old_y, comm);

	/* check interval */
	/* in_h := (x - x0) / div_num */
	if(div_num <= 0)
	{
		fprintf(stderr, "_mpi_dex_nim_fs: Number of division is illegal.\n");
		return;
	}
	in_h = (x - x0) / div_num;

	/* output */
	if(fp != NULL)
	{
		if(myrank == 0)
		{
			printf("           x             ");
			for(i = 0; i < y->dim; i++)
				printf("         y[%5d]          ", i);
			printf("\n");
		}
	}

	/* main loop */
	for(steps = 0; steps < div_num; steps++)
	{

		/* tmp_y := y0 + h * ex_nim(x0, y0) */
		num_stage = _mpi_dex_nim_1step_lo(local_tmp_y, old_x, local_old_y, old_y, in_h, local_mat, gfunc, rtol, atol, comm);

		/* set new x, y */
		old_x += in_h;
		subst_dvector(local_old_y, local_tmp_y);

		/* local_tmp_y -> old_y */
		_mpi_collect_dvector(old_y, dex_nim_ddim, local_tmp_y, comm);
		_mpi_bcast_dvector(old_y, comm);
		_mpi_divide_dvector(local_old_y, dex_nim_ddim, old_y, comm);

		/* x, y1, y2, ..., yn */
		if(fp != NULL)
		{
			if(myrank == 0)
			{
				fprintf(fp, "%25.17e", old_x);
				fprintf(fp, " ");
				for(i = 0; i < old_y->dim; i++)
				{
					fprintf(fp, "%25.17e", gdvi(old_y, i));
					fprintf(fp, " ");
				}
				fprintf(fp, "\n");
			}
		}

	}

	/* finish! */
	if(myrank == 0)
	{
		subst_dvector(y, old_y);
	
		printf("OK! dex_nim_fs has been just finished.\n");
		printf("Stepsize  : ");
			 printf("%25.17e", in_h);
			 printf("\n");
		printf("Number of steps  : %ld\n", steps);
		printf("Integral interval : ");
			 printf("[");
			 printf("%25.17e", x0);
			 printf(", ");
			 printf("%25.17e", old_x);
			 printf("]\n");
		printf("Numerical solution: \n");
			 print_dvector(y);
			 printf("\n");
	}

	/* clear */
	free_dvector(old_y);
	free_dvector(tmp_y);
	_mpi_free_dvector(local_old_y);
	_mpi_free_dvector(local_tmp_y);

	_mpi_free_dmatrix(local_mat, comm);

	_mpi_clear_dex_nim();

}

/*                          */
/* NIM Extrapolation Method */
/*                          */
void _mpi_dex_nim_lo(FILE *fp, double x, DVector y, double x0, DVector y0, double max_h, DMatrix mat, void (* gfunc)(DVector, double, MPI_Comm), double rtol, double atol, long int stage, MPI_Comm comm)
{
	long int steps, i, dim, conv_flag;
	DVector new_y[2], old_y, tmp_y, lf_tmp;
	DVector local_new_y[2], local_old_y, local_tmp_y, local_lf_tmp;
	DMatrix local_mat[MPI_GMP_MAXPROCS];
	double tmp[3], new_x, old_x, in_max_h, in_h, min_h;
	int myrank, procs;

	/* init */
	MPI_Comm_rank(comm, &myrank);
	MPI_Comm_size(comm, &procs);
	dim = y->dim;

	/* init */
	new_y[0] = init_dvector(dim);
	new_y[1] = init_dvector(dim);
	old_y = init_dvector(dim);
	tmp_y = init_dvector(dim);
	lf_tmp = init_dvector(dim);

	if(stage + 1 > MPI_MAX_DEX_NIM_STAGE)
	{
		fprintf(stderr, "dex_nim: Number of stages exceed.\n");
		return;
	}
	_mpi_init_dex_nim(stage + 1, dim, comm);
	_mpi_init_dmatrix(local_mat, dex_nim_ddim, dim, comm);
	_mpi_divide_dmatrix(local_mat, dex_nim_ddim, mat, comm);

	local_old_y = _mpi_init_dvector(dex_nim_ddim, dim, comm);
	local_tmp_y = _mpi_init_dvector(dex_nim_ddim, dim, comm);

	/* set */
	old_x = x0;
	min_h = max_h;
	subst_dvector(old_y, y0);

	_mpi_divide_dvector(local_old_y, dex_nim_ddim, old_y, comm);

	/* check interval */
	/* in_max_h := min((x - x0) / 2, max_h) */
	in_max_h = (x - x0) / 2;
	if(max_h < in_max_h)
		in_max_h = max_h;

	/* output */
	if(fp != NULL)
	{
		if(myrank == 0)
		{
			printf("           x             ");
			for(i = 0; i < y->dim; i++)
				printf("         y[%5d]          ", i);
			printf(" stepsize   ");
			printf("\n");
		}
	}

	/* main loop */
	in_h = max_h;
	for(steps = 0; ; steps++)
	{
		/* set default interval */
//		in_h = max_h;
		in_h *= 2;
		if(in_h > max_h)
			in_h = max_h;

		/* check */
		tmp[0] = old_x + in_h;
		if(tmp[0] > x)
			in_h = x - old_x;

		/* calc euler */
		while(1)
		{
			/* tmp_y := y0 + h * ex(x0, y0) */
			if(_mpi_dex_nim_1step_lo(local_tmp_y, old_x, local_old_y, old_y, in_h, local_mat, gfunc, rtol, atol, comm) == 0)
				break;

			/* h := h / 2 */
			tmp[2] = in_h / 2;

			/* check in_h */
			tmp[0] = old_x + tmp[2];
			if(tmp[0] <= old_x)
			{
				while(tmp[0] <= old_x)
				{
					in_h *= 2;
					tmp[0] = old_x + in_h;
				}
				_mpi_dex_nim_1step_lo(local_tmp_y, old_x, local_old_y, old_y, in_h, local_mat, gfunc, rtol, atol, comm);
			//	fprintf(stderr, "dex_harmonic: inner stepsize is too small.(%15.7e)\n", in_h);
				break;
			}
			else
				in_h = tmp[2];
		}
		if(min_h > in_h)
			min_h = in_h;

		/* set new x, y */
		old_x += in_h;
		subst_dvector(local_old_y, local_tmp_y);

		/* local_tmp_y -> old_y */
		_mpi_collect_dvector(old_y, dex_nim_ddim, local_tmp_y, comm);
		_mpi_bcast_dvector(old_y, comm);
		_mpi_divide_dvector(local_old_y, dex_nim_ddim, old_y, comm);

		/* x, y1, y2, ..., yn */
		if(fp != NULL)
		{
			if(myrank == 0)
			{
//				ansfunc(tmp_y, old_x);
				fprintf(fp, "%25.17e", old_x);
				fprintf(fp, " ");
				for(i = 0; i < old_y->dim; i++)
				{
					fprintf(fp, "%25.17e", gdvi(old_y, i));
//					fprintf(fp, "%10.7e", fabs(gdvi(old_y, i) - gdvi(tmp_y, i))/fabs(gdvi(tmp_y, i)));
					fprintf(fp, " ");
				}
				fprintf(fp, "%15.7e ", in_h);
				fprintf(fp, "\n");
			}
		}

		if(old_x >= x)
			break;

	}

	/* finish! */
	if(myrank == 0)
	{
		subst_dvector(y, old_y);
	
		printf("OK! dex_nim has been just finished.\n");
		printf("Minimum/Maxmum stepsize  : ");
			 printf("%15.7e", min_h);
			 printf(" / ");
			 printf("%15.7e", max_h);
			 printf("\n");
		printf("Number of steps  : %ld\n", steps);
		printf("Integral interval : ");
			 printf("[");
			 printf("%15.7e", x0);
			 printf(", ");
			 printf("%15.7e", old_x);
			 printf("]\n");
		printf("Numerical solution: \n");
			 print_dvector(y);
			 printf("\n");
	//	mpf_set(x, old_x);
	}

	/* clear */
	free_dvector(new_y[0]);
	free_dvector(new_y[1]);
	free_dvector(old_y);
	free_dvector(tmp_y);
	free_dvector(lf_tmp);

	_mpi_free_dvector(local_old_y);
	_mpi_free_dvector(local_tmp_y);

	_mpi_free_dmatrix(local_mat, comm);

	_mpi_clear_dex_nim();

}

#ifdef USE_GMP

/*                          */
/* NIM Extrapolation Method */
/*                          */
int _mpi_mpf_ex_nim_1step_lo(MPFVector local_y, mpf_t x0, MPFVector local_y0, MPFVector y0, mpf_t h, MPFMatrix local_mat[], void (* gfunc)(MPFVector, mpf_t, MPI_Comm), mpf_t rtol, mpf_t atol, MPI_Comm comm)
{
	long int i, j, k, istep, jstep, local_flag = 0, flag, dim;
	long istep_l;
	int myrank;

	istep_l = 2;
	mpf_div_ui(ex_nim_in_h, h, (unsigned long)istep_l);
	MPI_Comm_rank(comm, &myrank);

	for(istep = 0; istep < ex_nim_stage; istep++)
	{

/* First, Using Euler Method */
//		func(ex_nim_local_ytmp[0], x0, y0, myrank);
		_mpi_mul_mpfmatrix_mpfvec(ex_nim_local_ytmp[0], local_mat, local_y0, y0, comm);
		gfunc(ex_nim_local_ytmp[1], x0, comm);
		add2_mpfvector(ex_nim_local_ytmp[0], ex_nim_local_ytmp[1]);

		cmul2_mpfvector(ex_nim_local_ytmp[0], ex_nim_in_h);
		add_mpfvector(ex_nim_local_y[0][istep], local_y0, ex_nim_local_ytmp[0]);

		subst_mpfvector(ex_nim_local_ytmp[0], local_y0);
		subst_mpfvector(ex_nim_local_ytmp[1], ex_nim_local_y[0][istep]);

/* Next, Using Midpoint Rule */
		for(i = 1; i <= istep_l; i++)
		{
			mpf_mul_ui(ex_nim_mpf_tmp, ex_nim_in_h, (unsigned long)i);
			mpf_add(ex_nim_mpf_tmp, x0, ex_nim_mpf_tmp);

//			func(ex_nim_local_ytmp[2], ex_nim_mpf_tmp, ex_nim_ytmp[1], myrank);
			_mpi_mul_mpfmatrix_mpfvec(ex_nim_local_ytmp[2], local_mat, ex_nim_local_ytmp[1], ex_nim_ytmp[1], comm);
			gfunc(ex_nim_local_ytmp[3], ex_nim_mpf_tmp, comm);
			add2_mpfvector(ex_nim_local_ytmp[2], ex_nim_local_ytmp[3]);

			cmul2_mpfvector(ex_nim_local_ytmp[2], ex_nim_in_h);
			for(j = 0; j < local_y->dim; j++)
				mpf_mul_ui(
					get_mpfvector_i(ex_nim_local_ytmp[2], j),
					get_mpfvector_i(ex_nim_local_ytmp[2], j),
					(unsigned long)2
				);
			add_mpfvector(ex_nim_local_y[istep][0], ex_nim_local_ytmp[0], ex_nim_local_ytmp[2]);
			if(i == istep_l)
				subst_mpfvector(ex_nim_local_ytmp[2], ex_nim_local_ytmp[0]);
			subst_mpfvector(ex_nim_local_ytmp[0], ex_nim_local_ytmp[1]);
			subst_mpfvector(ex_nim_local_ytmp[1], ex_nim_local_y[istep][0]);
		}
/* Smoothing */
		add2_mpfvector(ex_nim_local_y[istep][0], ex_nim_local_ytmp[2]);
		mpf_set_ui(ex_nim_mpf_tmp, 2UL);
		cmul2_mpfvector(ex_nim_local_ytmp[0], ex_nim_mpf_tmp);
		add2_mpfvector(ex_nim_local_y[istep][0], ex_nim_local_ytmp[0]);
		mpf_set_ui(ex_nim_mpf_tmp, 1UL);
		mpf_div_ui(ex_nim_mpf_tmp, ex_nim_mpf_tmp, 4UL);
		cmul2_mpfvector(ex_nim_local_y[istep][0], ex_nim_mpf_tmp);

/* Extrapolation */
		for(jstep = 1; jstep <= istep; jstep++)
		{
			local_flag = 0;
			flag = 0;
			for(i = 0; i < local_y->dim; i++)
			{
				mpf_sub(
					get_mpfvector_i(ex_nim_local_ytmp[2], i),
					get_mpfvector_i(ex_nim_local_y[istep][jstep-1], i),
					get_mpfvector_i(ex_nim_local_y[istep-1][jstep-1], i)
				);
				mpf_div_ui(
					get_mpfvector_i(ex_nim_local_ytmp[2], i),
				 	get_mpfvector_i(ex_nim_local_ytmp[2], i),
				 	ex_nim_den(jstep*2)
				);
				mpf_add(
					get_mpfvector_i(ex_nim_local_y[istep][jstep], i),
					get_mpfvector_i(ex_nim_local_y[istep][jstep-1], i),
					get_mpfvector_i(ex_nim_local_ytmp[2], i)
				);
			
//				if(mpf_cmp(get_mpfvector_i(ex_nim_y[istep][jstep], i), get_mpfvector_i(ex_nim_y[istep][jstep-1], i) ) != 0)
//					flag = 1;
			}

			/* (ex_nim_ytmp[0], 0) := || ex_nim_y[istep][jstep] - ex_nim_y[istep][jstep-1] || * rtol + atol */
			sub_mpfvector(ex_nim_local_ytmp[1], ex_nim_local_y[istep][jstep], ex_nim_local_y[istep][jstep-1]);
			normi_mpfvector(get_mpfvector_i(ex_nim_local_ytmp[0], 0), ex_nim_local_ytmp[1]);
//			_mpi_normi_mpfvector(get_mpfvector_i(ex_nim_local_ytmp[0], 0), ex_nim_local_ytmp[1], comm);
	
			/* ex_nim_tmp := || ex_nim_y[istep][jstep] || */
			normi_mpfvector(ex_nim_mpf_tmp, ex_nim_local_y[istep][jstep]);
//			_mpi_normi_mpfvector(ex_nim_mpf_tmp, ex_nim_local_y[istep][jstep], comm);
			mpf_mul(ex_nim_mpf_tmp, ex_nim_mpf_tmp, rtol);
			mpf_add(ex_nim_mpf_tmp, ex_nim_mpf_tmp, atol);

			/* || ex_nim_y[istep][jstep] - ex_nim_y[istep][jstep-1]  || <= || ex_nim_y[istep][jstep] || * rtol + atol */
			if(mpf_cmp(gmpfvi(ex_nim_local_ytmp[0], 0), ex_nim_mpf_tmp) <= 0)
				local_flag = 0;
			else
				local_flag = 1;
		//	print_mpfvector(ex_y[istep][jstep]);

			MPI_Allreduce((void *)&local_flag, (void *)&flag, 1, MPI_LONG, MPI_SUM, comm);
			if(flag == 0)
			{
//				fflush(stdout); printf("return! %ld\n", istep);
				goto Return;
			}
		}
		mpf_div_ui(ex_nim_in_h, ex_nim_in_h, (unsigned long)2);
		istep_l *= 2;
//		printf("%d(%d) ", istep_l, ex_nim_stage);
	}

Return:
	if(istep == ex_nim_stage)
		istep--;
	if(jstep == ex_nim_stage)
		jstep--;
	subst_mpfvector(local_y, ex_nim_local_y[istep][jstep]);

	if(flag == 0) /* equal */
		return 0;
	else		 /* not equal */
		return 1;
}

/*                          */
/* NIM Extrapolation Method */
/* Stepsize fixed           */
void _mpi_mpf_ex_nim_fs_lo(FILE *fp, mpf_t x, MPFVector y, mpf_t x0, MPFVector y0, long int div_num, MPFMatrix mat, void (* gfunc)(MPFVector, mpf_t, MPI_Comm), mpf_t rtol, mpf_t atol, long int stage, MPI_Comm comm)
{
	long int steps, i, dim;
	MPFVector old_y, local_old_y, local_tmp_y, tmp_y, lf_tmp;
	mpf_t tmp[3], new_x, old_x, in_h;
	MPFMatrix local_mat[MPI_GMP_MAXPROCS];
	double start_time, end_time;
	int myrank, procs;

	/* time */
	start_time = get_secv();

	/* init */
	MPI_Comm_rank(comm, &myrank);
	MPI_Comm_size(comm, &procs);
	dim = y->dim;

	lf_tmp = init2_mpfvector(y->dim, y->prec);
	mpf_init2(new_x, mpf_get_prec(x));
	mpf_init2(old_x, mpf_get_prec(x));
	mpf_init2(tmp[0], mpf_get_prec(x));
	mpf_init2(tmp[1], mpf_get_prec(x));
	mpf_init2(tmp[2], mpf_get_prec(x));
	mpf_init2(in_h, mpf_get_prec(x));

	if(stage + 1 > MPI_MAX_EX_NIM_STAGE)
	{
		fprintf(stderr, "mpf_ex_nim_fs: Number of stages exceed.\n");
		return;
	}
	_mpi_init_mpf_ex_nim(stage + 1, y->dim, y->prec, comm);
	_mpi_init_mpfmatrix(local_mat, ex_nim_ddim, dim, comm);
	_mpi_divide_mpfmatrix(local_mat, ex_nim_ddim, mat, comm);

	/* set */
	mpf_set(old_x, x0);

	old_y = init2_mpfvector(ex_nim_local_dim * procs, y->prec);
	tmp_y = init2_mpfvector(ex_nim_local_dim * procs, y->prec);

	local_old_y = _mpi_init_mpfvector(ex_nim_ddim, dim, comm);
	local_tmp_y = _mpi_init_mpfvector(ex_nim_ddim, dim, comm);

	subst_mpfvector(old_y, y0);
	_mpi_divide_mpfvector(local_old_y, ex_nim_ddim, old_y, comm);

	/* check interval */
	/* in_h := (x - x0) / div_num */
	mpf_sub(in_h, x, x0);
	if(div_num <= 0)
	{
		fprintf(stderr, "_mpi_mpf_ex_nim_fs: A number of division is illegal.\n");
		return;
	}
	mpf_div_ui(in_h, in_h, div_num);

	/* output */
	if(fp != NULL)
	{
		if(myrank == 0)
		{
			printf("           x             ");
			for(i = 0; i < y->dim; i++)
				printf("         y[%5d]          ", i);
			printf("\n");
		}
	}

	/* main loop */
	for(steps = 0; steps < div_num; steps++)
	{

		/* calc euler */
		/* tmp_y := y0 + h * euler(x0, y0) */
		_mpi_mpf_ex_nim_1step_lo(local_tmp_y, old_x, local_old_y, old_y, in_h, local_mat, gfunc, rtol, atol, comm);

		/* set new x, y */
		mpf_add(old_x, old_x, in_h);
		subst_mpfvector(local_old_y, local_tmp_y);

		/* local_tmp_y -> old_y */
		_mpi_collect_mpfvector(old_y, ex_nim_ddim, local_tmp_y, comm);
		_mpi_bcast_mpfvector(old_y, comm);
		_mpi_divide_mpfvector(local_old_y, ex_nim_ddim, old_y, comm);

		/* x, y1, y2, ..., yn */
		if(fp != NULL)
		{
			if(myrank == 0)
			{
	//			mpf_out_str(fp, 10, 0, old_x);
				fprintf(fp, "%25.17e", mpf2double(old_x));
				fprintf(fp, " ");
				for(i = 0; i < old_y->dim; i++)
				{
	//				mpf_out_str(fp, 10, 0, gmpfvi(old_y, i));
					fprintf(fp, "%25.17e", mpf_get_d(gmpfvi(old_y, i)));
					fprintf(fp, " ");
				}
				fprintf(fp, "\n");
			}
		}

	}

	/* finish! */
	if(myrank == 0)
	{
		subst_mpfvector(y, old_y);
	
		/* time */
		end_time = get_secv();
	
		printf("OK! mpf_ex_nim_fs has been just finished.\n");
		printf("Stepsize  : ");
			 mpf_out_str(stdout, 10, 0, in_h);
	//		 printf(", Number of division: %ld ", div_num);
			 printf("\n");
		printf("Number of steps  : %ld\n", steps);
		printf("Computational Time(sec): %f\n", end_time - start_time);
		printf("Integral interval : ");
			 printf("[");
			 mpf_out_str(stdout, 10, 0, x0);
			 printf(", ");
			 mpf_out_str(stdout, 10, 0, old_x);
			 printf("]\n");
		printf("Numerical solution: \n");
			 print_mpfvector(y);
			 printf("\n");
	//	mpf_set(x, old_x);
	}

	/* clear */
	free_mpfvector(old_y);
	free_mpfvector(tmp_y);
	free_mpfvector(lf_tmp);
	_mpi_free_mpfvector(local_old_y);
	_mpi_free_mpfvector(local_tmp_y);
	mpf_clear(new_x);
	mpf_clear(old_x);
	mpf_clear(tmp[0]);
	mpf_clear(tmp[1]);
	mpf_clear(tmp[2]);
	mpf_clear(in_h);

	_mpi_free_mpfmatrix(local_mat, comm);

	_mpi_clear_mpf_ex_nim();

}

/*                          */
/* NIM Extrapolation Method */
/*                          */
void _mpi_mpf_ex_nim_lo(FILE *fp, mpf_t x, MPFVector y, mpf_t x0, MPFVector y0, mpf_t max_h, MPFMatrix mat, void (* gfunc)(MPFVector, mpf_t, MPI_Comm), mpf_t rtol, mpf_t atol, long int stage, MPI_Comm comm)
{
	long int steps, i, dim;
	MPFVector new_y[2], old_y, tmp_y, lf_tmp;
	MPFVector local_new_y[2], local_old_y, local_tmp_y, local_lf_tmp;
	mpf_t tmp[3], new_x, old_x, in_max_h, in_h, min_h;
	MPFMatrix local_mat[MPI_GMP_MAXPROCS];
	double start_time, end_time;
	int myrank, procs;

	/* time */
	start_time = get_secv();

	/* init */
	MPI_Comm_rank(comm, &myrank);
	MPI_Comm_size(comm, &procs);
	dim = y->dim;

	/* time */
	start_time = get_secv();
	
	/* init */
	new_y[0] = init2_mpfvector(y->dim, y->prec);
	new_y[1] = init2_mpfvector(y->dim, y->prec);
	old_y = init2_mpfvector(y->dim, y->prec);
	tmp_y = init2_mpfvector(y->dim, y->prec);
	lf_tmp = init2_mpfvector(y->dim, y->prec);
	mpf_init2(new_x, mpf_get_prec(x));
	mpf_init2(old_x, mpf_get_prec(x));
	mpf_init2(tmp[0], mpf_get_prec(x));
	mpf_init2(tmp[1], mpf_get_prec(x));
	mpf_init2(tmp[2], mpf_get_prec(x));
	mpf_init2(in_h, mpf_get_prec(x));
	mpf_init2(in_max_h, mpf_get_prec(x));
	mpf_init2(min_h, mpf_get_prec(x));

	if(stage + 1 > MPI_MAX_EX_NIM_STAGE)
	{
		fprintf(stderr, "_mpi_mpf_ex_nim: Number of stages exceed.\n");
		return;
	}
	_mpi_init_mpf_ex_nim(stage + 1, y->dim, y->prec, comm);
	_mpi_init_mpfmatrix(local_mat, ex_nim_ddim, dim, comm);
	_mpi_divide_mpfmatrix(local_mat, ex_nim_ddim, mat, comm);

	local_old_y = _mpi_init_mpfvector(ex_nim_ddim, dim, comm);
	local_tmp_y = _mpi_init_mpfvector(ex_nim_ddim, dim, comm);

	/* set */
	mpf_set(old_x, x0);
	mpf_set(min_h, max_h);
	subst_mpfvector(old_y, y0);

	_mpi_divide_mpfvector(local_old_y, ex_nim_ddim, old_y, comm);

	/* check interval */
	/* in_max_h := min((x - x0) / 2, max_h) */
	mpf_sub(in_max_h, x, x0);
	mpf_div_ui(in_max_h, in_max_h, 2UL);
	if(mpf_cmp(max_h, in_max_h) < 0)
		mpf_set(in_max_h, max_h);

	/* output */
	if(fp != NULL)
	{
		printf("           x             ");
		for(i = 0; i < y->dim; i++)
			printf("         y[%5d]          ", i);
		printf(" stepsize   ");
		printf("\n");
	}

	/* main loop */
	mpf_set(in_h, max_h);
	for(steps = 0; ; steps++)
	{

		/* set default interval */
		mpf_mul_ui(in_h, in_h, 2UL);
		if(mpf_cmp(in_h, max_h) > 0)
			mpf_set(in_h, max_h);

		/* check */
		mpf_add(tmp[0], old_x, in_h);
		if(mpf_cmp(tmp[0], x) > 0)
			mpf_sub(in_h, x, old_x);

		/* calc euler */
		while(1)
		{
			/* tmp_y := y0 + h * euler(x0, y0) */
			if(_mpi_mpf_ex_nim_1step_lo(local_tmp_y, old_x, local_old_y, old_y, in_h, local_mat, gfunc, rtol, atol, comm) == 0)
				break;

			/* h := h / 2 */
			mpf_div_ui(tmp[2], in_h, 2UL);

			/* check in_h */
			mpf_add(tmp[0], old_x, tmp[2]);
			if(mpf_cmp(tmp[0], old_x) <= 0)
			{
				while(mpf_cmp(tmp[0], old_x) <= 0)
				{
					mpf_mul_ui(in_h, in_h, 2UL);
					mpf_add(tmp[0], old_x, in_h);
				}
				_mpi_mpf_ex_nim_1step_lo(local_tmp_y, old_x, local_old_y, old_y, in_h, local_mat, gfunc, rtol, atol, comm);
//				fprintf(stderr, "mpf_ex_nim: inner stepsize is too small.\n");
				break;
			}
			else
				mpf_set(in_h, tmp[2]);
		}

		if(mpf_cmp(min_h, in_h) > 0)
			mpf_set(min_h, in_h);

		/* set new x, y */
		mpf_add(old_x, old_x, in_h);
		subst_mpfvector(local_old_y, local_tmp_y);

		/* local_tmp_y -> old_y */
		_mpi_collect_mpfvector(old_y, ex_nim_ddim, local_tmp_y, comm);
		_mpi_bcast_mpfvector(old_y, comm);
		_mpi_divide_mpfvector(local_old_y, ex_nim_ddim, old_y, comm);

		/* x, y1, y2, ..., yn */
		if(fp != NULL)
//		if((fp != NULL) && ((steps % itimes_iv) == 0))
		{
			if(myrank == 0)
			{
	//			mpf_out_str(fp, 10, 0, old_x);
				fprintf(fp, "%25.17e", mpf2double(old_x));
//				ansfunc(tmp_y, old_x);
				fprintf(fp, " ");
				for(i = 0; i < old_y->dim; i++)
				{
					mpf_out_str(fp, 10, 0, gmpfvi(old_y, i));
	//				fprintf(fp, "%25.17e", mpf_get_d(gmpfvi(old_y, i) - gmpfvi(tmp_y, i)));
	//				mpf_sub(tmp[0], gmpfvi(old_y, i), gmpfvi(tmp_y, i));
	//				mpf_div(tmp[0], tmp[0], gmpfvi(tmp_y, i));
	//				mpf_abs(tmp[0], tmp[0]);
	//				mpf_out_str(fp, 10, 7, tmp[0]);
					fprintf(fp, " ");
				}
				fprintf(fp, "%15.7e ", mpf2double(in_h));
				fprintf(fp, "\n");
			}
		}

		if(mpf_cmp(old_x, x) >= 0)
			break;

	}

	/* finish! */
	if(myrank == 0)
	{
		subst_mpfvector(y, old_y);
	
		/* time */
		end_time = get_secv();
	
		printf("OK! mpf_ex_nim has been just finished.\n");
		printf("Minimum/Maxmum stepsize  : ");
			 mpf_out_str(stdout, 10, 0, min_h);
			 printf(" / ");
			 mpf_out_str(stdout, 10, 0, max_h);
			 printf("\n");
		printf("Number of steps  : %ld\n", steps);
		printf("Computational Time(sec): %f\n", end_time - start_time);
		printf("Integral interval : ");
			 printf("[");
			 mpf_out_str(stdout, 10, 0, x0);
			 printf(", ");
			 mpf_out_str(stdout, 10, 0, old_x);
			 printf("]\n");
		printf("Numerical solution: \n");
			 print_mpfvector(y);
			 printf("\n");
	//	mpf_set(x, old_x);
	}

	/* clear */
	free_mpfvector(new_y[0]);
	free_mpfvector(new_y[1]);
	free_mpfvector(old_y);
	free_mpfvector(tmp_y);
	free_mpfvector(lf_tmp);
	_mpi_free_mpfvector(local_old_y);
	_mpi_free_mpfvector(local_tmp_y);
	mpf_clear(new_x);
	mpf_clear(old_x);
	mpf_clear(tmp[0]);
	mpf_clear(tmp[1]);
	mpf_clear(tmp[2]);
	mpf_clear(in_h);
	mpf_clear(in_max_h);
	mpf_clear(min_h);

	_mpi_free_mpfmatrix(local_mat, comm);

	_mpi_clear_mpf_ex_nim();

}
#endif
