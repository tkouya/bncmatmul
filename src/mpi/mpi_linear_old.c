/**********************************************/
/* mpi_linear.c:                              */
/* Copyright (C) 2003- Tomonori Kouya         */
/*                                            */
/* Version 0.0: 2003.08/21                    */
/* Version 0.1: 2004.02/28 Bug fix            */
/*                                            */
/* This library is free software; you can re- */
/* distribute it and/or modify it under the   */
/* terms of the GNU Lesser General Public     */
/* License as published by the Free Software  */
/* Foundation; either version 3 of the      */
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
#include <stdio.h>
#include <math.h>

#include "mpi.h"
#include "bnc.h"

#ifdef USE_GMP
#include "mpi_gmp.h"
#endif

#include "mpi_bnc.h"

/***************************************/
/* Basic Linear Computation            */
/***************************************/

/* double */

/* init for vector */
DVector _mpi_init_dvector(long d_dim[], long int dimension, MPI_Comm comm)
{
	DVector ret = NULL;
	long int i, local_dim; /* local_dim := dimension / num_procs */
	int myrank, num_procs;

	if(dimension <= 0)
	{
		fprintf(stderr, "ERROR: _mpi_init_dvector\n");
		return ret;
	}

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);
	local_dim = _mpi_divide_dim(d_dim, dimension, num_procs);

	ret = (DVector)malloc(sizeof(dvector));
	if(ret == NULL)
		return ret;

	ret->element = (double *)calloc(sizeof(double), local_dim);
	if(ret->element == NULL)
		return ret;

	/* All 0 */
	for(i = 0; i < local_dim; i++)
		*(ret->element + i) = 0.0;

	ret->dim = local_dim;

	return ret;
}

/* free dvector */
void _mpi_free_dvector(DVector vec)
{
	if(vec == NULL)
		return;

	if(vec->element != NULL)
		free(vec->element);

	free(vec);
}

/* Divide original vector on P0 */
void _mpi_divide_dvector(DVector d_vec, long int d_dim[], DVector src_vec, MPI_Comm comm)
{
//	DVector d_vec;

	long int local_dim, index, total_index, i, dim;
	int myrank, procs, num_procs;
	MPI_Status st;

	MPI_Comm_size(comm, &num_procs);
//	MPI_Comm_rank(comm, &myrank);
	MPI_Comm_rank(comm, &procs);

// /* on each process */
//	d_vec = _mpi_init_dvector(d_dim, src_vec->dim, comm);

	if(procs == 0)
		dim = src_vec->dim;

	MPI_Bcast(&dim, 1, MPI_LONG, 0, comm);
//	printf("rank: %d, dim = %d\n", procs, dim);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);

//	for(i = 0; i < procs - 1; i++)
//	printf("rank: %d, dim[%d] = %d, total_index = %d\n", procs, procs, d_dim[procs], total_index);

	/* P0 -> Pi */
	if(procs == 0)
	{
//		print_dvector(d_vec);
		total_index = 0;
		for(i = 0; i < num_procs; i++)
		{
			if(d_dim[i] <= 0)
				break;
			MPI_Send(
				src_vec->element + total_index,
				d_dim[i],
				MPI_DOUBLE,
				i,
				0,
				comm);
			total_index += d_dim[i];
		}
	}
//	else
	{
//		print_dvector(d_vec);
		MPI_Recv(
			d_vec->element,
			d_dim[procs],
			MPI_DOUBLE,
			0, 0, comm, &st);
	}
}

/* Collect vectors on P0 */
void _mpi_collect_dvector(DVector src_vec, long int d_dim[], DVector d_vec, MPI_Comm comm)
{
	long int local_dim, index, total_index, i, dim;
	int myrank, procs, num_procs;
	MPI_Status st;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &procs);

	if(procs == 0)
		dim = src_vec->dim;

	MPI_Bcast(&dim, 1, MPI_LONG, 0, comm);
//	printf("rank: %d, dim = %d\n", procs, dim);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);

	/* Pi -> P0 */

	MPI_Send(
		d_vec->element,
		d_dim[procs],
		MPI_DOUBLE,
		0, 0, comm);

	if(procs == 0)
	{
//		print_dvector(d_vec);
		total_index = 0;
		for(i = 0; i < num_procs; i++)
		{
			MPI_Recv(
				src_vec->element + total_index,
				d_dim[i],
				MPI_DOUBLE,
				i,
				0,
				comm, &st);
			total_index += d_dim[i];
		}
	}

}

/* Matrix */

/* init for matrix */
void _mpi_init_dmatrix(DMatrix ret[], long d_dim[], long int dimension, MPI_Comm comm)
{
        long int local_dim, i, j, k;
	int myrank, num_procs;

        if(dimension <= 0)
        {
                fprintf(stderr, "ERROR: _mpi_init_dmatrix\n");
                return;
        }

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);
	local_dim = _mpi_divide_dim(d_dim, dimension, num_procs);

	for(i = 0; i < num_procs; i++)
	{
		ret[i] = (DMatrix)malloc(sizeof(dmatrix));

		if(ret[i] == NULL)
			return;
	}

	for(i = 0; i < num_procs; i++)
	{
		ret[i]->element = (double *)calloc(sizeof(double), local_dim * local_dim);
		if(ret[i]->element == NULL)
			return;
	}

	/* All 0 */
	for(k = 0; k < num_procs; k++)
	{
		for(i = 0; i < local_dim; i++)
			for(j = 0; j < local_dim; j++)
				*(ret[k]->element + i * local_dim + j) = 0.0;
		ret[k]->row_dim = local_dim;
		ret[k]->col_dim = local_dim;
	}

	return;
}

/* free dmatrix */
void _mpi_free_dmatrix(DMatrix mat[], MPI_Comm comm)
{
	long int k;
	int num_procs;

	MPI_Comm_size(comm, &num_procs);

	for(k = 0; k < num_procs; k++)
	{
		if(mat[k] == NULL)
			continue;

		if(mat[k]->element != NULL)
			free(mat[k]->element);

		free(mat[k]);
	}
}

/* Divide original matrix on P0 */
void _mpi_divide_dmatrix(DMatrix d_mat[], long int d_dim[], DMatrix src_mat, MPI_Comm comm)
{
	long int local_dim, total_row_index, total_col_index, i, j, k, dim;
	int myrank, procs, num_procs;
	MPI_Status st;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &procs);

	if(procs == 0)
		dim = src_mat->row_dim;

	MPI_Bcast(&dim, 1, MPI_LONG, 0, comm);
//	printf("rank: %d, dim = %d\n", procs, dim);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);

//	printf("rank: %d, dim[%d] = %d, total_index = %d\n", procs, procs, d_dim[procs], total_index);

	/* P0 -> Pi */
	if(procs == 0)
	{

		total_row_index = 0;
		for(k = 0; k < num_procs; k++)
		{
			total_col_index = 0;
			for(i = 0; i < num_procs; i++)
			{
				for(j = 0; j < d_dim[k]; j++)
				{
//					printf("(%d, %d, %d, %d) %d\n", k, i, j, d_dim[i], d_dim[k]);
					MPI_Send(
						src_mat->element + (total_row_index + j) * dim + total_col_index,
						d_dim[i],
						MPI_DOUBLE,
						k, 0, comm);
				}
				total_col_index += d_dim[i];
			}
			total_row_index += d_dim[k];
		}
	}

	for(i = 0; i < num_procs; i++)
	{
		for(j = 0; j < d_dim[procs]; j++)
		{
//			printf("rank %d: (%d, %d, %d, %d) %d\n", procs, procs, i, j, d_dim[i], d_dim[procs]);
			MPI_Recv(
				d_mat[i]->element + j * local_dim,
				d_dim[i],
				MPI_DOUBLE,
				0, 0, comm, &st);
		}
	}

}

/* Collect matrices on P0 */
void _mpi_collect_dmatrix(DMatrix src_mat, long int d_dim[], DMatrix d_mat[], MPI_Comm comm)
{
	long int local_dim, total_row_index, total_col_index, i, j, k, dim;
	int myrank, procs, num_procs;
	MPI_Status st;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &procs);

	if(procs == 0)
		dim = src_mat->row_dim;

	MPI_Bcast(&dim, 1, MPI_LONG, 0, comm);
//	printf("rank: %d, dim = %d\n", procs, dim);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);

//	printf("rank: %d, dim[%d] = %d\n", procs, procs, d_dim[procs]);

	/* P0 <- Pi */
	for(i = 0; i < num_procs; i++)
	{
		for(j = 0; j < d_dim[procs]; j++)
		{
			MPI_Send(
				d_mat[i]->element + j * local_dim,
				d_dim[i],
				MPI_DOUBLE,
				0, 0, comm);
		}
	}

	if(procs == 0)
	{
//		print_dmatrix(d_mat);
		total_row_index = 0;
		for(k = 0; k < num_procs; k++)
		{
			total_col_index = 0;
			for(i = 0; i < num_procs; i++)
			{
				for(j = 0; j < d_dim[k]; j++)
				{
					MPI_Recv(
						src_mat->element + (total_row_index + j) * dim + total_col_index,
						d_dim[i],
						MPI_DOUBLE,
						k, 0, comm, &st);
				}
				total_col_index += d_dim[i];
			}
			total_row_index += d_dim[k];
		}
	}

}

/* Inner Product: (a, b) */
/* (1) eval (a, b) locally */
/* (2) Allreduce (a, b) */
double _mpi_ip_dvector(DVector in_a, DVector in_b, MPI_Comm comm)
{
	double ret, tmp;

	tmp = ip_dvector(in_a, in_b);

	MPI_Allreduce(&tmp, &ret, 1, MPI_DOUBLE, MPI_SUM, comm);

	return ret;
}

/* Euclid Norm: ||a||_2 */
/* (1) eval (a, a) locally */
/* (2) Allreduce (a, a) */
/* (3) Sqrt (a, a) */
double _mpi_norm2_dvector(DVector in_a, MPI_Comm comm)
{
	double ret, tmp;

	tmp = ip_dvector(in_a, in_a);

	MPI_Allreduce(&tmp, &ret, 1, MPI_DOUBLE, MPI_SUM, comm);

	return sqrt(ret);
}

/* 1 Norm: ||a||_1 */
double _mpi_norm1_dvector(DVector in_a, MPI_Comm comm)
{
	double ret, tmp;

	tmp = norm1_dvector(in_a);

	MPI_Allreduce(&tmp, &ret, 1, MPI_DOUBLE, MPI_SUM, comm);

	return ret;
}


/* Infinity Norm: ||a||_infty */
double _mpi_normi_dvector(DVector in_a, MPI_Comm comm)
{
	double ret, tmp[MPI_GMP_MAXPROCS];
	int myrank, procs, i;

	MPI_Comm_rank(comm, &myrank);
	MPI_Comm_size(comm, &procs);

	tmp[myrank] = normi_dvector(in_a);

	for(i = 0; i < procs; i++)
		MPI_Bcast((void *)&tmp[i], 1, MPI_DOUBLE, i, comm);

	ret = tmp[0];
	for(i = 0; i < procs; i++)
	{
		if(ret < tmp[i])
			ret = tmp[i];
	}

	return ret;
}



/* Matrix * Vector: Ax = y*/
/* (1) Allgather x */
/* (2) localy = A * x */
void _mpi_mul_dmatrix_dvec(DVector ret, DMatrix a[], DVector x, DVector x_all, MPI_Comm comm)
{
	long int i, j, k, index;
	int num_procs, myrank;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

//	printf("rank %d, x->dim=%d\n", myrank, x->dim);
/*
	index = 0;
	for(i = 0; i < num_procs; i++)
	{
		MPI_Gather(x->element, x->dim, MPI_DOUBLE, x_all->element, x->dim, MPI_DOUBLE, i, comm);
		index += x->dim;
	}
*/
	MPI_Allgather(x->element, x->dim, MPI_DOUBLE, x_all->element, x->dim, MPI_DOUBLE, comm);

	for(i = 0; i < a[0]->row_dim; i++)
	{
		*(ret->element + i) = 0.0;
		index = 0;
		for(k = 0; k < num_procs; k++)
		{
			for(j = 0; j < a[k]->col_dim; j++)
				*(ret->element + i) += *(a[k]->element + i * a[k]->col_dim + j) * *(x_all->element + index++);
		}
	}
}

#ifdef USE_GMP
/* mpf_t */

/* init for vector */
MPFVector _mpi_init_mpfvector(long d_dim[], long int dimension, MPI_Comm comm)
{
	MPFVector ret = NULL;
        long int i, local_dim; /* local_dim := dimension / num_procs */
	int myrank, num_procs;

        if(dimension <= 0)
        {
                fprintf(stderr, "ERROR: _mpi_init_dvector\n");
                return ret;
        }

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);
	local_dim = _mpi_divide_dim(d_dim, dimension, num_procs);

	ret = init_mpfvector(local_dim);

	return ret;
}

/* free dvector */
void _mpi_free_mpfvector(MPFVector vec)
{
	long int i;

	if(vec == NULL)
		return;

	if(vec->element != NULL)
	{
		for(i = 0; i < vec->dim; i++)
			mpf_clear((mpf_ptr)(vec->element + i));
	}

	free(vec);
}

/* Divide original vector on P0 */
void _mpi_divide_mpfvector(MPFVector d_vec, long int d_dim[], MPFVector src_vec, MPI_Comm comm)
{
//	MPFVector d_vec;

	long int local_dim, index, total_index, i, dim;
	int myrank, procs, num_procs;
	MPI_Status st;
	void *sbuf, *rbuf;

	MPI_Comm_size(comm, &num_procs);
//	MPI_Comm_rank(comm, &myrank);
	MPI_Comm_rank(comm, &procs);

// /* on each process */
//	d_vec = _mpi_init_mpfvector(d_dim, src_vec->dim, comm);

	if(procs == 0)
		dim = src_vec->dim;

	MPI_Bcast(&dim, 1, MPI_LONG, 0, comm);
//	printf("rank: %d, dim = %d\n", procs, dim);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);

//	for(i = 0; i < procs - 1; i++)
//	printf("rank: %d, dim[%d] = %d, total_index = %d\n", procs, procs, d_dim[procs], total_index);

	/* P0 -> Pi */
	if(procs == 0)
	{
//		printf("(rank: %d)", procs); print_mpfvector(d_vec);
		total_index = 0;
		sbuf = allocbuf_mpf(src_vec->prec, local_dim);
		for(i = 0; i < num_procs; i++)
		{
			MPI_Send(
				&src_vec->prec,
				1,
				MPI_UNSIGNED_LONG,
				i,
				0,
				comm);
			if(d_dim[i] <= 0)
				break;
			pack_mpf(gmpfvi(src_vec, total_index), d_dim[i], sbuf);
			MPI_Send(
				sbuf,
				d_dim[i],
				MPI_MPF,
				i,
				0,
				comm);
			total_index += d_dim[i];
		}
		free(sbuf);
	}
//	else
	{
		MPI_Recv(
			&d_vec->prec,
			1,
			MPI_UNSIGNED_LONG,
			0, 0, comm, &st);

//		printf("(rank: %d, dim: %d)", procs, d_dim[procs]);print_mpfvector(d_vec);
		if(d_dim[procs] <= 0)
			return;

		rbuf = allocbuf_mpf(d_vec->prec, d_dim[procs]);
//		printf("(rank: %d, prec->%d)\n", procs, d_vec->prec);
		MPI_Recv(
			rbuf,
			d_dim[procs],
			MPI_MPF,
			0, 0, comm, &st);
		unpack_mpf(rbuf, gmpfvi(d_vec, 0), d_dim[procs]);
//		unpack_mpf(rbuf, d_vec)->element, d_dim[procs]);
		free(rbuf);
//		printf("(rank: %d)", procs);print_mpfvector(d_vec);
	}
}

/* Collect vectors on P0 */
void _mpi_collect_mpfvector(MPFVector src_vec, long int d_dim[], MPFVector d_vec, MPI_Comm comm)
{
	long int local_dim, index, total_index, i, dim;
	int myrank, procs, num_procs;
	MPI_Status st;
	void *sbuf, *rbuf;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &procs);

	if(procs == 0)
		dim = src_vec->dim;

	MPI_Bcast(&dim, 1, MPI_LONG, 0, comm);
//	printf("rank: %d, dim = %d\n", procs, dim);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);

	/* Pi -> P0 */

	MPI_Send(
		&d_vec->prec,
		1,
		MPI_UNSIGNED_LONG,
		0, 0, comm);
//	printf("(rank: %d, d_dim[%d]: %d, prec: %d)", procs, procs, d_dim[procs], d_vec->prec);
	sbuf = allocbuf_mpf(d_vec->prec, d_dim[procs]);
//	pack_mpf(d_vec->element, d_dim[procs], sbuf);
	pack_mpf(gmpfvi(d_vec, 0), d_dim[procs], sbuf);
	MPI_Send(
		sbuf,
		d_dim[procs],
		MPI_MPF,
		0, 0, comm);
	free(sbuf);

	if(procs == 0)
	{
//		print_mpfector(d_vec);
		total_index = 0;
		rbuf = allocbuf_mpf(src_vec->prec, local_dim);
		for(i = 0; i < num_procs; i++)
		{
			MPI_Recv(
				&src_vec->prec,
				1,
				MPI_UNSIGNED_LONG,
				i,
				0,
				comm, &st);
			MPI_Recv(
				rbuf,
				d_dim[i],
				MPI_MPF,
				i,
				0,
				comm, &st);
			unpack_mpf(rbuf, gmpfvi(src_vec, total_index), d_dim[i]);
			total_index += d_dim[i];
		}
		free(rbuf);
	}

}

/* Matrix */

/* init for matrix */
void _mpi_init_mpfmatrix(MPFMatrix ret[], long d_dim[], long int dimension, MPI_Comm comm)
{
	long int local_dim, i, j, k;
	int myrank, num_procs;

        if(dimension <= 0)
        {
                fprintf(stderr, "ERROR: _mpi_init_mpfmatrix\n");
                return;
        }

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);
	local_dim = _mpi_divide_dim(d_dim, dimension, num_procs);

	for(i = 0; i < num_procs; i++)
	{
		ret[i] = init_mpfmatrix(local_dim, local_dim);

		if(ret[i] == NULL)
			return;
	}

	return;
}

/* free MPFMatrix */
void _mpi_free_mpfmatrix(MPFMatrix mat[], MPI_Comm comm)
{
	long int k;
	int num_procs;

	MPI_Comm_size(comm, &num_procs);

	for(k = 0; k < num_procs; k++)
	{
		if(mat[k] == NULL)
			continue;

		if(mat[k]->element != NULL)
			free(mat[k]->element);

		free(mat[k]);
	}
}

/* Divide original matrix on P0 */
void _mpi_divide_mpfmatrix(MPFMatrix d_mat[], long int d_dim[], MPFMatrix src_mat, MPI_Comm comm)
{
	long int local_dim, total_row_index, total_col_index, i, j, k, dim;
	int myrank, procs, num_procs;
	MPI_Status st;
	void *sbuf, *rbuf;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &procs);

	if(procs == 0)
		dim = src_mat->row_dim;

	MPI_Bcast(&dim, 1, MPI_LONG, 0, comm);
//	printf("rank: %d, dim = %d\n", procs, dim);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);

//	printf("rank: %d, dim[%d] = %d, total_index = %d\n", procs, procs, d_dim[procs], total_index);

	/* P0 -> Pi */
	if(procs == 0)
	{

		total_row_index = 0;
		sbuf = allocbuf_mpf(src_mat->prec, local_dim);
		for(k = 0; k < num_procs; k++)
		{
			MPI_Send(
				&src_mat->prec,
				1,
				MPI_UNSIGNED_LONG,
				k, 0, comm);
			total_col_index = 0;
			for(i = 0; i < num_procs; i++)
			{
				for(j = 0; j < d_dim[k]; j++)
				{
//					printf("(%d, %d, %d, %d) %d\n", k, i, j, d_dim[i], d_dim[k]);
//					pack_mpf(src_mat->element + (total_row_index + j) * dim + total_col_index, d_dim[i], sbuf);
					pack_mpf((mpf_ptr)(src_mat->element + (total_row_index + j) * dim + total_col_index), d_dim[i], sbuf);
					MPI_Send(
						sbuf,
						d_dim[i],
						MPI_MPF,
						k, 0, comm);
				}
				total_col_index += d_dim[i];
			}
			total_row_index += d_dim[k];
		}
		free(sbuf);
	}

	MPI_Recv(
		&d_mat[procs]->prec,
		1,
		MPI_UNSIGNED_LONG,
		0, 0, comm, &st);
	rbuf = allocbuf_mpf(d_mat[procs]->prec, local_dim);
	for(i = 0; i < num_procs; i++)
	{
		for(j = 0; j < d_dim[procs]; j++)
		{
//			printf("rank %d: (%d, %d, %d, %d) %d\n", procs, procs, i, j, d_dim[i], d_dim[procs]);
			MPI_Recv(
				rbuf,
				d_dim[i],
				MPI_MPF,
				0, 0, comm, &st);
//			unpack_mpf(rbuf, d_mat[i]->element + j * local_dim, d_dim[i]);
			unpack_mpf(rbuf, (mpf_ptr)(d_mat[i]->element + j * local_dim), d_dim[i]);
		}
	}
	free(rbuf);

}

/* Collect matrices on P0 */
void _mpi_collect_mpfmatrix(MPFMatrix src_mat, long int d_dim[], MPFMatrix d_mat[], MPI_Comm comm)
{
	long int local_dim, total_row_index, total_col_index, i, j, k, dim;
	int myrank, procs, num_procs;
	MPI_Status st;
	void *sbuf, *rbuf;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &procs);

	if(procs == 0)
		dim = src_mat->row_dim;

	MPI_Bcast(&dim, 1, MPI_LONG, 0, comm);
//	printf("rank: %d, dim = %d\n", procs, dim);

	local_dim = _mpi_divide_dim(d_dim, dim, num_procs);

//	printf("rank: %d, dim[%d] = %d\n", procs, procs, d_dim[procs]);

	/* P0 <- Pi */
	MPI_Send(
		&d_mat[procs]->prec,
		1,
		MPI_UNSIGNED_LONG,
		0, 0, comm);
	sbuf = allocbuf_mpf(d_mat[procs]->prec, local_dim);
	for(i = 0; i < num_procs; i++)
	{
		for(j = 0; j < d_dim[procs]; j++)
		{
//			pack_mpf(d_mat[i]->element + j * local_dim, d_dim[i], sbuf);
			pack_mpf((mpf_ptr)(d_mat[i]->element + j * local_dim), d_dim[i], sbuf);
			MPI_Send(
				sbuf,
				d_dim[i],
				MPI_MPF,
				0, 0, comm);
		}
	}
	free(sbuf);

	if(procs == 0)
	{
//		print_MPFMatrix(d_mat);
		total_row_index = 0;
		rbuf = allocbuf_mpf(src_mat->prec, local_dim);
		for(k = 0; k < num_procs; k++)
		{
			total_col_index = 0;
			MPI_Recv(
				&src_mat->prec,
				1,
				MPI_UNSIGNED_LONG,
				k, 0, comm, &st);
			for(i = 0; i < num_procs; i++)
			{
				for(j = 0; j < d_dim[k]; j++)
				{
					MPI_Recv(
						rbuf,
						d_dim[i],
						MPI_MPF,
						k, 0, comm, &st);
//					unpack_mpf(rbuf, src_mat->element + (total_row_index + j) * dim + total_col_index, d_dim[i]);
					unpack_mpf(rbuf, (mpf_ptr)(src_mat->element + (total_row_index + j) * dim + total_col_index), d_dim[i]);
				}
				total_col_index += d_dim[i];
			}
			total_row_index += d_dim[k];
		}
		free(rbuf);
	}

}

/* Inner Product: (a, b) */
/* (1) eval (a, b) locally */
/* (2) Allreduce (a, b) */
void _mpi_ip_mpfvector(mpf_t ret, MPFVector in_a, MPFVector in_b, MPI_Comm comm)
{
	mpf_t tmp;
	void *sbuf, *rbuf;

	mpf_init(tmp);

	ip_mpfvector(tmp, in_a, in_b);
	sbuf = allocbuf_mpf(in_a->prec, 1);
	rbuf = allocbuf_mpf(in_a->prec, 1);

	pack_mpf(tmp, 1, sbuf);
	MPI_Allreduce(sbuf, rbuf, 1, MPI_MPF, MPI_MPF_SUM, comm);
	unpack_mpf(rbuf, ret, 1);

	mpf_clear(tmp);
	free(sbuf); // fix
	free(rbuf); // fix

	return;
}

/* Euclid Norm: ||a||_2 */
/* (1) eval (a, a) locally */
/* (2) Allreduce (a, a) */
/* (3) Sqrt (a, a) */
void _mpi_norm2_mpfvector(mpf_t ret, MPFVector in_a, MPI_Comm comm)
{
	mpf_t tmp;
	void *sbuf, *rbuf;

	mpf_init(tmp);

	ip_mpfvector(tmp, in_a, in_a);
	sbuf = allocbuf_mpf(in_a->prec, 1);
	rbuf = allocbuf_mpf(in_a->prec, 1);

	pack_mpf(tmp, 1, sbuf);
	MPI_Allreduce(sbuf, rbuf, 1, MPI_MPF, MPI_MPF_SUM, comm);
	unpack_mpf(rbuf, ret, 1);
	mpf_sqrt(ret, ret);

	mpf_clear(tmp);
	free(sbuf); // fix
	free(rbuf); // fix

	return;
}

/* 1 Norm: ||a||_1 */
void _mpi_norm1_mpfvector(mpf_t ret, MPFVector in_a, MPI_Comm comm)
{
	mpf_t tmp;
	void *sbuf, *rbuf;

	mpf_init(tmp);

	norm1_mpfvector(tmp, in_a);
	sbuf = allocbuf_mpf(in_a->prec, 1);
	rbuf = allocbuf_mpf(in_a->prec, 1);

	pack_mpf(tmp, 1, sbuf);
	MPI_Allreduce(sbuf, rbuf, 1, MPI_MPF, MPI_MPF_SUM, comm);
	unpack_mpf(rbuf, ret, 1);

	mpf_clear(tmp);
	free(sbuf); // fix
	free(rbuf); // fix

	return;
}


/* Infinity Norm: ||a||_infty */
void _mpi_normi_mpfvector(mpf_t ret, MPFVector in_a, MPI_Comm comm)
{
	mpf_t tmp[MPI_GMP_MAXPROCS];
	int myrank, procs, i;
	void *sbuf[MPI_GMP_MAXPROCS];

	MPI_Comm_rank(comm, &myrank);
	MPI_Comm_size(comm, &procs);

	for(i = 0; i < procs; i++)
	{
		mpf_init2(tmp[i], mpf_get_prec(ret));
		sbuf[i] = allocbuf_mpf(mpf_get_prec(ret), 1);
	}

	normi_mpfvector(tmp[myrank], in_a);
	pack_mpf(tmp[myrank], 1, sbuf[myrank]);

	for(i = 0; i < procs; i++)
	{
		MPI_Bcast(sbuf[i], 1, MPI_MPF, i, comm);
		if(i != myrank)
			unpack_mpf(sbuf[i], tmp[i], 1);
	}

	mpf_set(ret, tmp[0]);
	for(i = 0; i < procs; i++)
	{
		if(mpf_cmp(ret, tmp[i]) < 0)
			mpf_set(ret, tmp[i]);
	}

	for(i = 0; i < procs; i++)
	{
		mpf_clear(tmp[i]);
		free(sbuf[i]);
	}

	return;
}



/* Matrix * Vector: Ax = y*/
/* (1) Allgather x */
/* (2) locally = A * x */
void _mpi_mul_mpfmatrix_mpfvec(MPFVector ret, MPFMatrix a[], MPFVector x, MPFVector x_all, MPI_Comm comm)
{
	long int i, j, k, index;
	int num_procs, myrank;
	mpf_t tmp;
	void *sbuf, *rbuf;

	MPI_Comm_size(comm, &num_procs);
	MPI_Comm_rank(comm, &myrank);

//	printf("rank %d, x->dim=%d, prec=%d\n", myrank, x->dim, ret->prec);
/*
	index = 0;
	for(i = 0; i < num_procs; i++)
	{
		MPI_Gather(x->element, x->dim, MPI_MPF, x_all->element, x->dim, MPI_MPF, i, comm);
		index += x->dim;
	}
*/
	sbuf = allocbuf_mpf(x->prec, x->dim);
	rbuf = allocbuf_mpf(x_all->prec, x_all->dim);
//	pack_mpf(x->element, x->dim, sbuf);
	pack_mpf(gmpfvi(x, 0), x->dim, sbuf);
	MPI_Allgather(sbuf, x->dim, MPI_MPF, rbuf, x->dim, MPI_MPF, comm);
//	unpack_mpf(rbuf, x_all->element, x_all->dim);
	unpack_mpf(rbuf, gmpfvi(x_all, 0), x_all->dim);

	mpf_init(tmp);
	for(i = 0; i < a[0]->row_dim; i++)
	{
		mpf_set_ui(*(ret->element + i), 0UL);
		index = 0;
		for(k = 0; k < num_procs; k++)
		{
			for(j = 0; j < a[k]->col_dim; j++)
			{
				mpf_mul(tmp, *(a[k]->element + i * a[k]->col_dim + j), *(x_all->element + index++));
				mpf_add(*(ret->element + i), *(ret->element + i), tmp);
			}
		}
	}
	mpf_clear(tmp);
	free(sbuf); // fix
	free(rbuf); // fix
}
#endif
