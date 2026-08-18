/**********************************************/
/* test4.c:                                   */
/* Copyright (C) 2004- Tomonori Kouya         */
/*                                            */
/* Version 0.0: 2004.03/10                    */
/*                                            */
/* This library is free software; you can re- */
/* distribute it and/or modify it under the   */
/* terms of the GNU Lesser General Public     */
/* License as published by the Free Software  */
/* Foundation; either version 2.1 of the      */
/* License, or (at your option) any later     */
/* version.                                   */
/*                                            */
/* This library is distributed in the hope    */
/* that it will be useful, but WITHOUT ANY    */
/* WARRANTY; without even the implied         */
/* warranty of MERCHANTABILITY or FITNESS FOR */
/* A PARTICULAR PURPOSE.  See the GNU Lesser  */
/* General Public License for more details.   */
/**********************************************/
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

	if(numprocs < 4)
	{
		fprintf(stderr, "MORE PROCESSES!(num_procs: %d)\n", numprocs);
		goto end;
	}

	fprintf(stdout,"Process %d of %d on %s\n",
		myid, numprocs, processor_name);

#ifdef USE_GMP

#define MPF_PREC 512
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
		if(myid == 2)
		{
			pos = 0;
			pack_mpf(a, 1, a_buf);
			MPI_Send(a_buf, 1, gmp_mpf, 3, tag, MPI_COMM_WORLD);
		}
		else if(myid == 3)
		{
			MPI_Recv(c_buf, 1, gmp_mpf, 2, tag, MPI_COMM_WORLD, &st);
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
//		MPI_Scatter(a_buf, 1, gmp_mpf, b_buf, 1, gmp_mpf, 0, MPI_COMM_WORLD);
//		MPI_Gather(a_buf, 1, gmp_mpf, b_buf, 1, gmp_mpf, 0, MPI_COMM_WORLD);
//		MPI_Allgather(a_buf, 1, gmp_mpf, b_buf, 1, gmp_mpf, MPI_COMM_WORLD);
		MPI_Alltoall(a_buf, 1, gmp_mpf, b_buf, 1, gmp_mpf, MPI_COMM_WORLD);
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
end:
	MPI_Finalize();
	return 0;
}
