/********************************************************************************/
/* mpi_matrix_mul.c:                                                            */
/* Copyright (C) 2003-2011 Tomonori Kouya                                       */
/*                                                                              */
/* Version 0.0: 2003.10/29                                                      */
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

/***************************************/
/* Matrix-Matrix Multiply              */
/***************************************/

/* initialize a index of d_mat on PE_i */
long int *init_mm_index(long int mat_dim)
{
	long int i;
	long int *sarray, *tmp;

	if(mat_dim <= 0)
		return;

	sarray = (long int *)calloc(sizeof(long int), (size_t)mat_dim);
	if(sarray == NULL)
	{
		fprintf(stderr, "Cannot allocate index array!(init_mm_index)\n");
		return sarray;
	}

	tmp = sarray;
	for(i = 0; i < mat_dim; i++)
		*sarray++ = i;

	return tmp;
}

/* shift index */
void shift_mm_index(long int *index, long int mat_dim)
{
	long int i, tmp;

	tmp = *(index + 0);
	for(i = 0; i < mat_dim - 1; i++)
		*(index + i) = *(index + i + 1);
	*(index + mat_dim - 1) = tmp;
}

/* check index */
int check_mm_index(long int *index, long int mat_dim)
{
	long int i, tmp;

	for(i = 0; i < mat_dim; i++)
	{
		if(*(index + i) != i)
			return abs(*(index + i) - i);
	}
	return 0;
}

/* Shift index of d_mat on PE_i */
/* <- col_index[0] <- col_index[1] <- ... <- col_index[mat_dim - 1] <- col_index[0] */
void _mpi_send_west(long int *col_index, long int mat_dim)
{
	shift_mm_index(col_index, mat_dim);
}

/* double */

/* send dmatrix to "dest_proc"th PE */
void _mpi_send_dmatrix(DMatrix mat, int dest_proc, MPI_Comm comm)
{
	int myrank;

	MPI_Comm_rank(comm, &myrank);

//	printf("%d to %d\n", myrank, dest_proc);
	MPI_Send(
		mat->element,
		(mat->row_dim) * (mat->col_dim),
		MPI_DOUBLE,
		dest_proc, 0, comm);
}

/* recv dmatrix from "src_proc"th PE */
void _mpi_recv_dmatrix(DMatrix mat, int src_proc, MPI_Comm comm)
{
	int myrank;
	MPI_Status st;

	MPI_Comm_rank(comm, &myrank);

//	printf("%d from %d\n", myrank, src_proc);}
	MPI_Recv(
		mat->element,
		(mat->row_dim) * (mat->col_dim),
		MPI_DOUBLE,
		src_proc, 0, comm, &st);

}


/* Send submatrix to uppper PE */
/* mat on PE_{i-1} <- mat on PE_i */
void _mpi_dmatrix_send_north(DMatrix mat, DMatrix tmp_mat, long int *row_index, long int mat_dim, MPI_Comm comm)
{
	int num_procs, myrank;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	shift_mm_index(row_index, mat_dim);

	if(num_procs <= 1)
		return;

	if(num_procs == 2)
	{
		if(myrank == 0)
		{
			_mpi_recv_dmatrix(tmp_mat, 1, comm);
			_mpi_send_dmatrix(mat, 1, comm);
		}
		else
		{
			_mpi_send_dmatrix(mat, 0, comm);
			_mpi_recv_dmatrix(tmp_mat, 0, comm);
		}
	}
	else
	{
		if(myrank == 0)
		{
			_mpi_recv_dmatrix(tmp_mat, 1, comm);
			_mpi_send_dmatrix(mat, num_procs - 1, comm);
		}
		else if(myrank == (num_procs - 1))
		{
			_mpi_recv_dmatrix(tmp_mat, 0, comm);
			_mpi_send_dmatrix(mat, num_procs - 2, comm);
		}
		else
		{
			_mpi_send_dmatrix(mat, myrank - 1, comm);
			_mpi_recv_dmatrix(tmp_mat, myrank + 1, comm);
		}
	}
	subst_dmatrix(mat, tmp_mat);
}

/* Matrix * Matrix: ret := AB */
void _mpi_mul_dmatrix(DMatrix ret[], DMatrix a[], DMatrix b[], MPI_Comm comm)
{
	long int i, j, mat_dim;
	long int *row_index, *col_index[MPI_GMP_MAXPROCS];
	DMatrix tmp_mat;
	int num_procs, myrank;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	mat_dim = num_procs;

	/* init index */
	row_index = init_mm_index(mat_dim);
	for(i = 0; i < mat_dim; i++)
		col_index[i] = init_mm_index(mat_dim);

	/* init shift */
	for(i = 0; i < myrank; i++)
		_mpi_send_west(row_index, mat_dim);

	tmp_mat = init_dmatrix(b[0]->row_dim, b[0]->col_dim);
	for(i = 0; i < mat_dim; i++)
		set0_dmatrix(ret[i]);

	for(i = 0; i < mat_dim; i++)
	{
		for(j = 0; j < i; j++)
			_mpi_dmatrix_send_north(b[i], tmp_mat, col_index[i], mat_dim, comm);
	}

	/* multiply */
//	printf("Proc %d: \n", myrank);
	for(i = 0; i < mat_dim; i++)
	{
		for(j = 0; j < mat_dim; j++)
		{
			mul_dmatrix(tmp_mat, a[row_index[j]], b[j]);
			add_dmatrix(ret[j], ret[j], tmp_mat);
//			printf("a[%d,%d]*b[%d,%d] ", myrank, row_index[j], (col_index[j])[myrank], j);
		}
//		printf("\n");
		for(j = 0; j < mat_dim; j++)
			_mpi_dmatrix_send_north(b[j], tmp_mat, col_index[j], mat_dim, comm);
		_mpi_send_west(row_index, mat_dim);
	}

	/* back to original b */
	for(i = 0; i < mat_dim; i++)
	{
		while(check_mm_index(col_index[i], mat_dim) != 0)
			_mpi_dmatrix_send_north(b[i], tmp_mat, col_index[i], mat_dim, comm);
	}

	free_dmatrix(tmp_mat);
	free(row_index);
	for(i = 0; i < mat_dim; i++)
		free(col_index[i]);
}

#ifdef USE_GMP

/* mpf_t */

/* send mpfmatrix to "dest_proc"th PE */
void _mpi_send_mpfmatrix(MPFMatrix mat, int dest_proc, MPI_Comm comm)
{
	int myrank;
	void *sbuf;

//	MPI_Comm_rank(comm, &myrank);

	MPI_Send(
		&mat->prec,
		1,
		MPI_UNSIGNED_LONG,
		dest_proc, 0, comm);

	sbuf = allocbuf_mpf(mat->prec, (mat->row_dim) * (mat->col_dim));
	pack_mpf((mpf_ptr)mat->element, (mat->row_dim) * (mat->col_dim), sbuf);
	MPI_Send(
		sbuf,
		(mat->row_dim) * (mat->col_dim),
		MPI_MPF,
		dest_proc, 0, comm);

	free(sbuf);
}

/* recv mpfmatrix from "src_proc"th PE */
void _mpi_recv_mpfmatrix(MPFMatrix mat, int src_proc, MPI_Comm comm)
{
	int myrank;
	void *rbuf;
	MPI_Status st;

//	MPI_Comm_rank(comm, &myrank);

	MPI_Recv(
		&mat->prec,
		1,
		MPI_UNSIGNED_LONG,
		src_proc, 0, comm, &st);

	rbuf = allocbuf_mpf(mat->prec, (mat->row_dim) * (mat->col_dim));
	MPI_Recv(
		rbuf,
		(mat->row_dim) * (mat->col_dim),
		MPI_MPF,
		src_proc, 0, comm, &st);
	unpack_mpf(rbuf, (mpf_ptr)mat->element, (mat->row_dim) * (mat->col_dim));

	free(rbuf);
}


/* Send submatrix to uppper PE */
/* mat on PE_{i-1} <- mat on PE_i */
void _mpi_mpfmatrix_send_north(MPFMatrix mat, MPFMatrix tmp_mat, long int *row_index, long int mat_dim, MPI_Comm comm)
{
	int num_procs, myrank;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	shift_mm_index(row_index, mat_dim);

	if(num_procs <= 1)
		return;

	if(num_procs == 2)
	{
		if(myrank == 0)
		{
			_mpi_recv_mpfmatrix(tmp_mat, 1, comm);
			_mpi_send_mpfmatrix(mat, 1, comm);
		}
		else
		{
			_mpi_send_mpfmatrix(mat, 0, comm);
			_mpi_recv_mpfmatrix(tmp_mat, 0, comm);
		}
	}
	else
	{
		if(myrank == 0)
		{
			_mpi_recv_mpfmatrix(tmp_mat, 1, comm);
			_mpi_send_mpfmatrix(mat, num_procs - 1, comm);
		}
		else if(myrank == (num_procs - 1))
		{
			_mpi_recv_mpfmatrix(tmp_mat, 0, comm);
			_mpi_send_mpfmatrix(mat, num_procs - 2, comm);
		}
		else
		{
			_mpi_send_mpfmatrix(mat, myrank - 1, comm);
			_mpi_recv_mpfmatrix(tmp_mat, myrank + 1, comm);
		}
	}
	subst_mpfmatrix(mat, tmp_mat);
}

/* Matrix * Matrix: ret := AB */
void _mpi_mul_mpfmatrix(MPFMatrix ret[], MPFMatrix a[], MPFMatrix b[], MPI_Comm comm)
{
	long int i, j, mat_dim;
	long int *row_index, *col_index[MPI_GMP_MAXPROCS];
	MPFMatrix tmp_mat;
	int num_procs, myrank;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

	mat_dim = num_procs;

	/* init index */
	row_index = init_mm_index(mat_dim);
	for(i = 0; i < mat_dim; i++)
		col_index[i] = init_mm_index(mat_dim);

	/* init shift */
	for(i = 0; i < myrank; i++)
		_mpi_send_west(row_index, mat_dim);

	tmp_mat = init_mpfmatrix(b[0]->row_dim, b[0]->col_dim);
	for(i = 0; i < mat_dim; i++)
		set0_mpfmatrix(ret[i]);

//	printf("Proc %d: \n", myrank);
	for(i = 0; i < mat_dim; i++)
	{
		for(j = 0; j < i; j++)
			_mpi_mpfmatrix_send_north(b[i], tmp_mat, col_index[i], mat_dim, comm);
	}

	/* multiply */
	for(i = 0; i < mat_dim; i++)
	{
		for(j = 0; j < mat_dim; j++)
		{
			mul_mpfmatrix(tmp_mat, a[row_index[j]], b[j]);
			add_mpfmatrix(ret[j], ret[j], tmp_mat);
		}
		for(j = 0; j < mat_dim; j++)
			_mpi_mpfmatrix_send_north(b[j], tmp_mat, col_index[j], mat_dim, comm);
		_mpi_send_west(row_index, mat_dim);
	}

	/* back to original b */
	for(i = 0; i < mat_dim; i++)
	{
		while(check_mm_index(col_index[i], mat_dim) != 0)
			_mpi_mpfmatrix_send_north(b[i], tmp_mat, col_index[i], mat_dim, comm);
	}

	free_mpfmatrix(tmp_mat);
	free(row_index);
	for(i = 0; i < mat_dim; i++)
		free(col_index[i]);
}

#endif
