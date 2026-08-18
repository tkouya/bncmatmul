/********************************************************************************/
/* testc.c:                                                                     */
/* Copyright (C) 2003-2011 Tomonori Kouya                                       */
/*                                                                              */
/* Version 0.0: 2003.08/21                                                      */
/* Version 0.0: 2005.05/28 support MPICH2                                       */
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
//#include "mpi_gmp.h"
#include "mpi_gmp.h"
#endif

int main(int argc,char *argv[])
{
	double startwtime = 0.0, endwtime, instartwtime, inendwtime, innertime;
	int  namelen, numprocs, myid, times;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

	MPI_Datatype _local_gmp_mpf;
	MPI_Op _local_gmp_mpf_add;
	MPI_Status st;
#ifdef USE_GMP
	mpf_t a, b, c, mpf_array[128];
	void *a_buf, *b_buf, *c_buf, *mpf_array_buf, *mpf_array_buf_recv;
	int bufsize, array_bufsize, pos, tag;
	int i, j, k, dest, source;
#endif
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	MPI_Get_processor_name(processor_name,&namelen);

	fprintf(stdout,"Process %d of %d on %s\n",
		myid, numprocs, processor_name);

#ifdef USE_GMP

#define MPF_PREC 128
//#define MPF_PREC 256
//#define MPF_PREC 512
//#define MPF_PREC 1024
//#define MPF_PREC 2048
//#define MPF_PREC 4096
//#define MPF_PREC 8192
//#define MPF_PREC 16384
//#define MPF_PREC 32768
//#define MPF_PREC 65536
//#define MPF_PREC 131072
//#define MPF_PREC 262144
//#define MPF_PREC MPF_D100
//#define MAX_TIMES 100000
#define MAX_TIMES 1
//#define MAX_TIMES 10

/* init for mpf_t of GMP */
	mpf_set_default_prec(MPF_PREC);
	if(myid == 0)
		printf("Prec: %d bits(%f decimal digits)\n", MPF_PREC, (double)MPF_PREC * log10(2.0));
	commit_mpf(&_local_gmp_mpf, MPF_PREC, MPI_COMM_WORLD);
	create_mpf_op(&_local_gmp_mpf_add, _mpi_mpf_add, MPI_COMM_WORLD);

	mpf_init_set_ui(a, 2UL);
	mpf_init_set_ui(b, 3UL);
	mpf_init(c);
	mpf_sqrt(a, a); // a = sqrt(2);
	mpf_sqrt(b, b); // b = sqrt(3);
	mpf_set_ui(c, 0UL); // c = 0;

	for(i = 0; i < numprocs; i++)
	{
		mpf_init_set_ui(mpf_array[i], (unsigned long)(i + 1));
		mpf_sqrt(mpf_array[i], mpf_array[i]);
		mpf_mul(mpf_array[i], mpf_array[i], a); // sqrt(i+1)*sqrt(2)
	}

	MPI_Type_size(_local_gmp_mpf, &bufsize);
	printf("MPI_Type_size  : %d\n", bufsize);
	printf("get_bufsize_mpf: %d\n", get_bufsize_mpf(mpf_array[0], 1));
	a_buf = (void *)malloc(bufsize);
	b_buf = (void *)malloc(bufsize);
	c_buf = (void *)malloc(bufsize);
	array_bufsize = bufsize * numprocs;
	mpf_array_buf = (void *)malloc(array_bufsize);
//	mpf_array_buf_recv = (void *)malloc(array_bufsize);
	mpf_array_buf_recv = (void *)malloc(array_bufsize);

/* Collective communication (1) --- Broadcast */
	MPI_Barrier(MPI_COMM_WORLD);
	if(myid == 0)
	{
		printf("----- Bcast Test Start -----\n");
//		mpf_out_str(stdout, 10, 0, a);printf("\n");
		innertime = 0;
		startwtime = MPI_Wtime();
	}

	MPI_Barrier(MPI_COMM_WORLD);
	for(times = 0; times < MAX_TIMES; times++)
	{
		if(myid == 0)
		{
			pack_mpf(a, 1, a_buf);
			instartwtime = MPI_Wtime();
		}

		MPI_Bcast(a_buf, 1, _local_gmp_mpf, 0, MPI_COMM_WORLD);

		if(myid == 0) inendwtime = MPI_Wtime();
		if(myid != 0)	
		{
			//unpack_mpf(a_buf, b, 1);
			//mpf_out_str(stdout, 10, 0, b);printf("\n");
		}
		if(myid == 0) innertime += inendwtime - instartwtime;

	}	

	MPI_Barrier(MPI_COMM_WORLD);
	if(myid == 0)
	{
		endwtime = MPI_Wtime();
		printf("Total Time: %f, (%f Sec/Bcast, %d bits)\n", 
			endwtime - startwtime, 
			(endwtime - startwtime) / times, MPF_PREC);
		printf("In Total Time: (%f Sec/Bcast)\n", 
			innertime / times);
	} 
	MPI_Barrier(MPI_COMM_WORLD);
//	goto gather;

/* Collective communication (2) --- Scatter */
scatter:

	//MPI_Barrier(MPI_COMM_WORLD);
	if(myid == 0)
	{
		printf("----- Scatter Test Start -----\n");
		for(i = 0; i < numprocs; i++){printf("send->");mpf_out_str(stdout, 10, 0, mpf_array[i]);printf("\n");}
		innertime = 0;
		startwtime = MPI_Wtime();
	}

	//MPI_Barrier(MPI_COMM_WORLD);
	for(times = 0; times < MAX_TIMES; times++)
	{
		if(myid == 0)
		{
			pack_mpf(mpf_array[0], numprocs, mpf_array_buf);
			instartwtime = MPI_Wtime();
		}

		MPI_Scatter(mpf_array_buf, 1, _local_gmp_mpf, c_buf, 1, _local_gmp_mpf, 0, MPI_COMM_WORLD);

		if(myid == 0) inendwtime = MPI_Wtime();
		MPI_Barrier(MPI_COMM_WORLD);
		if(myid != 0)	
		{
			unpack_mpf(c_buf, c, 1);
			printf("recv->");mpf_out_str(stdout, 10, 0, c);printf("\n");
		}
		if(myid == 0) innertime += inendwtime - instartwtime;
	}	

	//MPI_Barrier(MPI_COMM_WORLD);
	if(myid == 0)
	{
		endwtime = MPI_Wtime();
		printf("Total Time: %f, (%f Sec/Scatter)\n", 
			endwtime - startwtime, 
			(endwtime - startwtime) / times);
		printf("In Total Time: (%f Sec/Bcast)\n", 
			innertime / times);
	} 

goto end;
/* Collective communication (3) --- Gather */
gather:
	//MPI_Barrier(MPI_COMM_WORLD);
	if(myid == 0)
	{
		printf("----- Gather Test Start -----\n");
//		mpf_out_str(stdout, 10, 0, a);printf("\n");
		innertime = 0;
		startwtime = MPI_Wtime();
	}

	//MPI_Barrier(MPI_COMM_WORLD);
	for(times = 0; times < MAX_TIMES; times++)
	{
		if(myid != 0)
		{
			pack_mpf(c, 1, c_buf);
		}
		if(myid == 0) instartwtime = MPI_Wtime();

		MPI_Gather(c_buf, 1, _local_gmp_mpf, mpf_array_buf, 1, _local_gmp_mpf, 0, MPI_COMM_WORLD);

		//MPI_Barrier(MPI_COMM_WORLD);
		if(myid == 0)	
		{
			inendwtime = MPI_Wtime();
			//unpack_mpf(mpf_array_buf, mpf_array[0], numprocs);
			innertime += inendwtime - instartwtime;
/*			for(i = 0; i < numprocs; i++)
			{
				mpf_out_str(stdout, 10, 0, mpf_array[i]);printf("\n");
			}
			printf("\n");
*/		}
	}	

	//MPI_Barrier(MPI_COMM_WORLD);
	if(myid == 0)
	{
		endwtime = MPI_Wtime();
		printf("Total Time: %f, (%f Sec/Gather)\n", 
			endwtime - startwtime, 
			(endwtime - startwtime) / times);
		printf("In Total Time: (%f Sec/Bcast)\n", 
			innertime / times);
	} 

//	goto end;

/* Collective communication (4) --- AllGather */
allgather:

	MPI_Barrier(MPI_COMM_WORLD);
	if(myid == 0)
	{
		printf("----- AllGather Test Start -----\n");
//		mpf_out_str(stdout, 10, 0, a);printf("\n");
		innertime = 0;
		startwtime = MPI_Wtime();
	}

	MPI_Barrier(MPI_COMM_WORLD);
	for(times = 0; times < MAX_TIMES; times++)
	{
		pack_mpf(c, 1, c_buf);
		if(myid == 0) instartwtime = MPI_Wtime();

		//MPI_Allgather(c_buf, 1, _local_gmp_mpf, mpf_array_buf_recv, 1, _local_gmp_mpf, MPI_COMM_WORLD);
		if(myid == 0) inendwtime = MPI_Wtime();

		MPI_Barrier(MPI_COMM_WORLD);
		if(myid != 0)	
		{
			//unpack_mpf(mpf_array_buf_recv, mpf_array[0], numprocs);
/*			for(i = 0; i < numprocs; i++)
			{
				mpf_out_str(stdout, 10, 0, mpf_array[i]);printf("\n");
			}
			printf("\n");
*/		}
		if(myid == 0) innertime += inendwtime - instartwtime;
	}	

	MPI_Barrier(MPI_COMM_WORLD);
	if(myid == 0)
	{
		endwtime = MPI_Wtime();
		printf("Total Time: %f, (%f Sec/Allgather)\n", 
			endwtime - startwtime, 
			(endwtime - startwtime) / times);
		printf("In Total Time: (%f Sec/Bcast)\n", 
			innertime / times);
	} 

goto end;
/* Collective communication (4) --- AlltoAll */
alltoall:

	MPI_Barrier(MPI_COMM_WORLD);
	if(myid == 0)
	{
		printf("----- AlltoAll Test Start -----\n");
//		mpf_out_str(stdout, 10, 0, a);printf("\n");
		innertime = 0;
		startwtime = MPI_Wtime();
	}

	MPI_Barrier(MPI_COMM_WORLD);
	for(times = 0; times < MAX_TIMES; times++)
	{

		if(myid == 0) instartwtime = MPI_Wtime();
		MPI_Alltoall(mpf_array_buf, 1, _local_gmp_mpf, mpf_array_buf_recv, 1, _local_gmp_mpf, MPI_COMM_WORLD);
		if(myid == 0) inendwtime = MPI_Wtime();

		MPI_Barrier(MPI_COMM_WORLD);
		if(myid != 0)	
		{
			//unpack_mpf(mpf_array_buf_recv, mpf_array[0], 1);
/*			for(i = 0; i < numprocs; i++)
			{
				mpf_out_str(stdout, 10, 0, mpf_array[i]);printf("\n");
			}
			printf("\n");
*/		}
		if(myid == 0) innertime += inendwtime - instartwtime;
	}	

	MPI_Barrier(MPI_COMM_WORLD);
	if(myid == 0)
	{
		endwtime = MPI_Wtime();
		printf("Total Time: %f, (%f Sec/Alltoall)\n", 
			endwtime - startwtime, 
			(endwtime - startwtime) / times);
		printf("In Total Time: (%f Sec/Bcast)\n", 
			innertime / times);
	} 



/* clear */
end:
	free_mpf(&_local_gmp_mpf);
	free_mpf_op(&_local_gmp_mpf_add);
#endif
	MPI_Finalize();
	return 0;
}
