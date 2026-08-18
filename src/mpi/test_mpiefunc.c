/********************************************************************************/
/* test_mpiefunc.c:                                                             */
/* Copyright (C) 2003-2011 Tomonori Kouya                                       */
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
#include "mpi.h"
#include <stdio.h>
#include <math.h>

//#define USE_MPFR
#include "bnc.h"
#ifdef USE_GMP
#include "mpi_gmp.h"

/* m! */
void mpf_fact(mpf_t ret, long int m)
{
	mpf_set_ui(ret, 1UL);
	if(m <= 0)
		return;

	mpf_set_ui(ret, (unsigned long)m);
	while(--m > 1)
		mpf_mul_ui(ret, ret, m);
}

/* x_m = x^m, m_fac = m! */
/* ret := sum^m_{i=1} x^(myrank * m + i) / (myrank * m + i)! */
int _mpi_partial_exp(mpf_t ret, long int myrank, mpf_t x, mpf_t x_m, long int m, long int width, mpf_t fact_init)
{
	long int i, ret_flag;
	mpf_t x_beki, m_kai, tmp, old_ret;

	mpf_init2(x_beki, mpf_get_prec(ret));
	mpf_init2(m_kai, mpf_get_prec(ret));
	mpf_init2(tmp, mpf_get_prec(ret));
	mpf_init2(old_ret, mpf_get_prec(ret));

	/* x_beki = x^(m + myrank * width) */
	mpf_power(x_beki, x, myrank * width);
	mpf_mul(x_beki, x_beki, x_m);

	/* m_kai = (m + myrank * width)! */
//	mpf_fact(m_kai, m + myrank * width);
	mpf_set(m_kai, fact_init);

	mpf_set_ui(ret, 0UL);
	ret_flag = 1;
	for(i = 1; i <= width; i++)
	{
		mpf_set(old_ret, ret);
		mpf_mul_ui(m_kai, m_kai, m + myrank * width + i);
		mpf_mul(x_beki, x_beki, x);

		/* ret += ret */
		/*  + x^(m + myrank * width + i) / (m + myrank * width + i)! */
		mpf_div(tmp, x_beki, m_kai);
		mpf_add(ret, ret, tmp);

		if(mpf_cmp(old_ret, ret) == 0)
		{
			ret_flag = 0;
			break;
		}
	}

//	printf("rank %d ", myrank);mpf_out_str(stdout, 10, 0, ret); printf("\n");
//	printf("rank %d, m = %d, width = %d\n", myrank, m, width);
	mpf_clear(tmp);
	mpf_clear(old_ret);
	mpf_clear(x_beki);
	mpf_clear(m_kai);

	return ret_flag;
}
#endif

int main(int argc,char *argv[])
{
	int myid, numprocs, i;
	double startwtime = 0.0, endwtime;
	int  namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	int mpf_size, pos;

	void *packed_partial_ans, *packed_ans;

#ifdef USE_GMP
	mpf_t ans, x, partial_ans, fact_init;
	mpf_t old_ans, x_width, x_m, ans_add;
	long m, times, width, end_flag;
	unsigned long int prec;
#endif

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	MPI_Get_processor_name(processor_name,&namelen);

	fprintf(stdout,"Process %d of %d on %s\n",
		myid, numprocs, processor_name);

#ifdef USE_GMP
#define MPF_PREC 16384
//#define MPF_PREC 8192
//#define MPF_PREC 128

	mpf_set_default_prec(MPF_PREC);
	commit_mpf(&(MPI_MPF), MPF_PREC, MPI_COMM_WORLD);
	create_mpf_op(&(MPI_MPF_SUM), _mpi_mpf_add, MPI_COMM_WORLD);

	if(myid == 0)
		startwtime = MPI_Wtime();

	mpf_init(x);
	mpf_init_set_ui(ans, 0UL);
	mpf_init(x_m);
	mpf_init(x_width);
	mpf_init(partial_ans);
	mpf_init(old_ans);
	mpf_init(ans_add);
	mpf_init(fact_init);

	mpf_set_str(x, "1.5", 10);
	m = 0;	width = 8; mpf_set_ui(x_m, 1UL); mpf_power(x_width, x, width);
	mpf_set_ui(ans, 0UL);
	packed_partial_ans = allocbuf_mpf(MPF_PREC, 1);
	packed_ans = allocbuf_mpf(MPF_PREC, 1);
//	while(_mpi_partial_exp(partial_ans, myid, x, x_m, m) == 1)
	end_flag = 1;
	do
	{
		mpf_fact(fact_init, m + myid * width);
		_mpi_partial_exp(partial_ans, myid, x, x_m, m, width, fact_init);
//		if(myid != (numprocs - 1))mpf_fact(fact_init, m + (numprocs - 1)* width);
		pack_mpf(partial_ans, 1, packed_partial_ans);
		MPI_Reduce(packed_partial_ans, packed_ans, 1, MPI_MPF, MPI_MPF_SUM, 0, MPI_COMM_WORLD);
		MPI_Barrier(MPI_COMM_WORLD);
//	printf("rank %d ", myid);mpf_out_str(stdout, 10, 0, partial_ans); printf("\n");
		if(myid == 0)
		{
			unpack_mpf(packed_ans, ans_add, 1);
			mpf_set(old_ans, ans);
			mpf_add(ans, ans, ans_add);
			if(mpf_cmp(old_ans, ans) == 0)
			{
				printf("Convergent!(m = %d)\n", m);
				end_flag = 0;
			}
		}
		MPI_Bcast(&end_flag, 1, MPI_LONG, 0, MPI_COMM_WORLD);
		m += width * numprocs;
		mpf_power(x_m, x_width, numprocs); /* x_m := x^(width * numprocs); */ 
	}while(end_flag == 1);
	mpf_add_ui(ans, ans, 1UL);

//	mpf_exp(ans, x);

	MPI_Barrier(MPI_COMM_WORLD);
	if (myid == 0)
	{
		endwtime = MPI_Wtime();
		printf("wall clock time = %f\n", endwtime-startwtime);		
		printf("ans:");mpf_out_str(stdout, 10, 0, ans);printf("\n");
		fflush( stdout );

		/* use BNCpack */
		printf("BNC: exp(%e) = %e\nmpf_exp = ", 1.0, exp(1.0));
		printf("x: ");mpf_out_str(stdout, 10, 0, x);printf("\n");
		startwtime = get_secv();
		mpf_exp(ans, x);
		endwtime = get_secv();
		printf("   ");mpf_out_str(stdout, 10, 0, ans);printf("\n");
		printf("wall clock time = %f\n", endwtime-startwtime);

	}

	free_mpf(&(MPI_MPF));
	free_mpf_op(&(MPI_MPF_SUM));
#endif

	MPI_Finalize();

}
