/********************************************************************************/
/* test_mpzq.c:                                                                 */
/* Copyright (C) 2003-2011 Tomonori Kouya                                       */
/*                                                                              */
/* Version 0.0: 2003.05/08                                                      */
/*                                                                              */
/* Usage:                                                                       */
/* % mpicc test_mpzq.c mpi_gmp.c -lgmp                                          */
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
#include "mpi.h"

#ifdef USE_GMP
#include "gmp.h"
#include "mpi_gmp.h"
#endif

#define NFAC 10000

int main(int argc,char *argv[])
{
	long int i;
	int myrank, num_procs, namelen;
	char processor_name[MPI_MAX_PROCESSOR_NAME];

#ifdef USE_GMP
	mpz_t za, zb, zc;
	mpq_t qa, qb, qc;
	void *zbuf, *zbuf_recv[MPI_GMP_MAXPROCS];
	void *qbuf, *qbuf_recv[MPI_GMP_MAXPROCS];
	int zbuf_recv_size[MPI_GMP_MAXPROCS];
	int qbuf_recv_size[MPI_GMP_MAXPROCS];
	MPI_Status st;
#endif

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
	MPI_Get_processor_name(processor_name,&namelen);

	fprintf(stdout,"Process %d of %d on %s\n", myrank, num_procs, processor_name);

#ifdef USE_GMP
/* mpz_t factorial */

	mpz_init_set_ui(za, 1UL); mpz_init(zb); mpz_init(zc);

//	printf("zc->_mp_alloc za->_mp_alloc i!\n"); 
	for(i = myrank; i < NFAC; i += num_procs)
	{
		mpz_mul_ui(zc, za, (unsigned long)(i+1));
//		gmp_printf("%5d %5d %Zd\n", zc->_mp_alloc, za->_mp_alloc, zc);
		mpz_set(za, zc);
	}
	gmp_printf("local fac(%5d) = %Zd\n", i, za);

	if(myrank != 0)
	{
		zbuf_recv_size[myrank] = (int)get_bufsize_mpz(za);
		zbuf = allocbuf_mpz(za);
		pack_mpz(za, zbuf);
		/* Pi -> P0 */
		MPI_Send(&zbuf_recv_size[myrank],
			1,
			MPI_INT,
			0, 0, MPI_COMM_WORLD);
		MPI_Send(zbuf,
			zbuf_recv_size[myrank],
			MPI_BYTE,
			0, 0, MPI_COMM_WORLD);
	}

	if(myrank == 0)
	{
		for(i = 1; i < num_procs; i++)
		{
			MPI_Recv(&zbuf_recv_size[i],
				1,
				MPI_INT,
				i,
				0,
				MPI_COMM_WORLD, &st);
			zbuf_recv[i] = (void *)malloc(sizeof(unsigned char) * zbuf_recv_size[i]);
			MPI_Recv(zbuf_recv[i],
				zbuf_recv_size[i],
				MPI_BYTE,
				i,
				0,
				MPI_COMM_WORLD, &st);
			unpack_mpz(zbuf_recv[i], zb);
			mpz_mul(za, za, zb);
		}
		gmp_printf("fac(%5d) = %Zd\n", i, za);
	}
	mpz_clear(za); mpz_clear(zb); mpz_clear(zc);

//	goto end;

/* mpq_t inverse factorial */

	mpq_init(qa); mpq_init(qb); mpq_init(qc);
	mpq_set_ui(qa, 1UL, 1UL);

//	printf("qc->_mp_num->_mp_alloc qa->_mp_den->_mp_alloc i!\n");
	for(i = myrank; i < NFAC; i += num_procs)
	{
		mpq_set_ui(qb, (unsigned long)(i + 1), 1UL);
		mpq_div(qc, qa, qb);
//		gmp_printf("%5d %5d %Qd\n", qc->_mp_num._mp_alloc, qa->_mp_den._mp_alloc, qc);
		mpq_set(qa, qc);
	}
	gmp_printf("local inv fac(%5d) = %Qd\n", i, qa);

	if(myrank != 0)
	{
		qbuf_recv_size[myrank] = (int)get_bufsize_mpq(qa);
		qbuf = allocbuf_mpq(qa);
		pack_mpq(qa, qbuf);
		/* Pi -> P0 */
		MPI_Send(&qbuf_recv_size[myrank],
			1,
			MPI_INT,
			0, 0, MPI_COMM_WORLD);
		MPI_Send(qbuf,
			qbuf_recv_size[myrank],
			MPI_BYTE,
			0, 0, MPI_COMM_WORLD);
	}

	if(myrank == 0)
	{
		for(i = 1; i < num_procs; i++)
		{
			MPI_Recv(&qbuf_recv_size[i],
				1,
				MPI_INT,
				i,
				0,
				MPI_COMM_WORLD, &st);
			qbuf_recv[i] = (void *)malloc(sizeof(unsigned char) * qbuf_recv_size[i]);
			MPI_Recv(qbuf_recv[i],
				qbuf_recv_size[i],
				MPI_BYTE,
				i,
				0,
				MPI_COMM_WORLD, &st);
			unpack_mpq(qbuf_recv[i], qb);
			mpq_mul(qa, qa, qb);
		}
		gmp_printf("inv fac(%5d) = %Qd\n", i, qa);
	}

	mpq_clear(qa); mpq_clear(qb); mpq_clear(qc);

end:
#endif
	MPI_Finalize();

}
