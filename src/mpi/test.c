/********************************************************************************/
/* test.c:                                                                      */
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

#ifdef USE_GMP
#include "mpi_gmp.h"
#endif

int main(int argc,char *argv[])
{
	double startwtime = 0.0, endwtime;
	int  namelen, numprocs, myid, times;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Datatype gmp_mpf;
	MPI_Op gmp_mpf_add;
	MPI_Status st;
#ifdef USE_GMP
	mpf_t a, b, c;
	void *a_buf, *b_buf, *c_buf;
	int bufsize, pos, tag;
	int i, j, k, dest, source;
#endif

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	MPI_Get_processor_name(processor_name,&namelen);

	fprintf(stdout,"Process %d of %d on %s\n",
		myid, numprocs, processor_name);


#ifdef USE_GMP

#define MPF_PREC 128
//#define MPF_PREC 256
//#define MPF_PREC 512
#define MAX_TIMES 100000

/* init for mpf_t of GMP */
	mpf_set_default_prec(MPF_PREC);
	if(myid == 0)
		printf("Prec: %d bits(%f decimal digits)\n", MPF_PREC, (double)MPF_PREC * log10(2.0));
	commit_mpf(&gmp_mpf, MPF_PREC, MPI_COMM_WORLD);
	create_mpf_op(&gmp_mpf_add, _mpi_mpf_add, MPI_COMM_WORLD);

	mpf_init_set_ui(a, 2UL);
	mpf_init_set_ui(b, 3UL);
	mpf_init(c);
	mpf_sqrt(a, a); // a = sqrt(2);
	mpf_sqrt(b, b); // b = sqrt(3);

	/* check sizeof int, long int */
	printf("                            : bits\n");
	printf("GMP_LIMB_BITS               : %d\n", GMP_LIMB_BITS);
#ifdef USE_MPFR
#if MPFR_VERSION_MOJOR >= 2
	printf("_mpfr_prec                  : %d\n", a->_mpfr_prec);
#else
	printf("_mpfr_prec                  : %d\n", a->_mpfr_prec);
#endif
#else
	printf("_mp_prec                    : %d\n", a->_mp_prec);
#endif
	printf("                            : Bytes\n");
	printf("sizeof(int)                 : %d\n", (long int)sizeof(int));
	printf("sizeof(long)                : %d\n", (long int)sizeof(long));
#if MPFR_VERSION_MOJOR >= 2
	printf("sizeof(mpfr_sign_t)         : %d\n", (long int)sizeof(mpfr_sign_t));
	printf("sizeof(mpfr_prec_t)         : %d\n", (long int)sizeof(mpfr_prec_t));
#else
	printf("sizeof(mp_size_t)           : %d\n", (long int)sizeof(mp_size_t));
	printf("sizeof(mp_prec_t)           : %d\n", (long int)sizeof(mp_prec_t));
#endif
	printf("sizeof(mp_exp_t)            : %d\n", (long int)sizeof(mp_exp_t));
	printf("sizeof(mp_limb_t)           : %d\n", (long int)sizeof(mp_limb_t));
	printf("sizeof(mp_limb_t *)         : %d\n", (long int)sizeof(mp_limb_t *));
	printf("get_bufsize_mpf(%5d bits) : %d\n", MPF_PREC, (long int)get_bufsize_mpf(a, 1));
	printf("_NUM_LIMB(%5d bits)       : %d\n", MPF_PREC, (long int)_NUM_LIMB(a));

	MPI_Type_size(gmp_mpf, &bufsize);
	a_buf = (void *)malloc(bufsize);
	b_buf = (void *)malloc(bufsize);
	c_buf = (void *)malloc(bufsize);

/* ping test */
	if(myid == 0)
	{
		printf("----- Ping Test Start -----\n");
		startwtime = MPI_Wtime();
	}

	for(times = 0; times < MAX_TIMES; times++)
	{
		tag = 1;
		if(myid == 0)
		{
			pos = 0;
			pack_mpf(a, 1, a_buf);
			MPI_Send(a_buf, 1, gmp_mpf, 1, tag, MPI_COMM_WORLD);
		}
		else if(myid == 1)
		{
			MPI_Recv(c_buf, 1, gmp_mpf, 0, tag, MPI_COMM_WORLD, &st);
			pos = 0;
			unpack_mpf(c_buf, c, 1);
//			mpf_out_str(stdout, 10, 0, c);printf("\n");
		}
	}

	if(myid == 0)
	{
		endwtime = MPI_Wtime();
		printf("Total Time: %f, (%f Sec/SendRecv)\n", 
			endwtime - startwtime, 
			(endwtime - startwtime) / MAX_TIMES);
		printf("----- Ping Test End -----\n");
	}

	free_mpf(&gmp_mpf);
	free_mpf_op(&gmp_mpf_add);
	MPI_Finalize();
	exit(0);
	

/* relay test */
	if(myid == 0)
	{
		printf("----- Relay Test Start -----\n");
		mpf_out_str(stdout, 10, 0, a);printf("\n");
		startwtime = MPI_Wtime();
	}

	for(times = 0; times < 10; times++)
	{
		tag = 1;
		if(myid == 0)
		{
			pos = 0;
			pack_mpf(a, 1, a_buf);
			MPI_Send(a_buf, 1, gmp_mpf, 1, tag, MPI_COMM_WORLD);
		}
		else if((myid > 0) && (myid <= numprocs - 2))
		{
			source = myid - 1;
			dest = myid + 1;
			MPI_Recv(b_buf, 1, gmp_mpf, source, tag, MPI_COMM_WORLD, &st);
			MPI_Send(b_buf, 1, gmp_mpf, dest, tag, MPI_COMM_WORLD);
		}
		else if(myid == numprocs - 1)
		{
			MPI_Recv(b_buf, 1, gmp_mpf, numprocs - 2, tag, MPI_COMM_WORLD, &st);
			MPI_Send(b_buf, 1, gmp_mpf, 0, tag, MPI_COMM_WORLD);
			
		}

		if(myid == 0)
		{
			MPI_Recv(b_buf, 1, gmp_mpf, numprocs - 1, tag, MPI_COMM_WORLD, &st);
			pos = 0;
			unpack_mpf(b_buf, b, 1);
		}
	}
	
	if(myid == 0)
	{
		endwtime = MPI_Wtime();
		mpf_out_str(stdout, 10, 0, b);printf("\n");
		printf("Total Time: %f, (%f Sec/Relay, %f Sec/RecvSend)\n", 
			endwtime - startwtime, 
			(endwtime - startwtime) / times,
			(endwtime - startwtime) / times / numprocs);
		printf("----- Relay Test End -----\n");
	}

/* Collective communication */
	if(myid == 0)
	{
		printf("----- Bcast Test Start -----\n");
		mpf_out_str(stdout, 10, 0, a);printf("\n");
		startwtime = MPI_Wtime();
	}

	for(times = 0; times < 10; times++)
	{
		tag = 3;
		if(myid == 0)
		{
			pos = 0;
			pack_mpf(a, 1, a_buf);
		}
//		MPI_Bcast(a_buf, 1, gmp_mpf, 0, MPI_COMM_WORLD);
		MPI_Scatter(a_buf, 1, gmp_mpf, b_buf, 1, gmp_mpf, 0, MPI_COMM_WORLD);
//		MPI_Gather(a_buf, 1, gmp_mpf, b_buf, 1, gmp_mpf, 0, MPI_COMM_WORLD);
//		MPI_Allgather(a_buf, 1, gmp_mpf, b_buf, 1, gmp_mpf, MPI_COMM_WORLD);
//		MPI_Alltoall(a_buf, 1, gmp_mpf, b_buf, 1, gmp_mpf, MPI_COMM_WORLD);
		if(myid != 0)	
		{
			pos = 0;
			unpack_mpf(a_buf, b, 1);
//			mpf_out_str(stdout, 10, 0, b);printf("\n");
		}
	}	

	if(myid == 0)
	{
		endwtime = MPI_Wtime();
		printf("Total Time: %f, (%f Sec/Bcast)\n", 
			endwtime - startwtime, 
			(endwtime - startwtime) / times);
	} 
/* clear */
	free_mpf(&gmp_mpf);
	free_mpf_op(&gmp_mpf_add);
#endif
	MPI_Finalize();
	return 0;
}
