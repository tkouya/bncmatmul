/********************************************************************************/
/* matmul_strassen_general_gds.cu:                                              */
/* Copyright (C) 2015 Tomonori Kouya                                            */
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
#include "gdslinear.h" // G[D,Q]DVector, G[D,Q]DMatrix
#include "bncuda.h" // BNCUDA library

//#include "bnc.h"
#include "bncomp.h"
#include "dslinear.h"

#include "matmul_strassen.h"

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
__global__ void _bncu_add_gdsmatrix_partial(\
	gds_real *ret_dev_element, long int ret_row_dim, long int ret_col_dim, \
		long int ret_index_dev0, long int ret_index_dev1, long int ret_index_dev2, long int ret_index_dev3, \
	gds_real *mat_a_dev_element, long int a_row_dim, long int a_col_dim, \
		long int mat_a_index_dev0, long int mat_a_index_dev1, long int mat_a_index_dev2, long int mat_a_index_dev3,\
	gds_real *mat_b_dev_element, long int b_row_dim, long int b_col_dim, \
		long int mat_b_index_dev0, long int mat_b_index_dev1, long int mat_b_index_dev2, long int mat_b_index_dev3 \
)
{
	int index, i, j;
	long int imax, jmax, ret_i, ret_j, a_i, a_j, b_i, b_j;
	gds_real tmp_val;

	imax = ret_index_dev1 - ret_index_dev0;
	jmax = ret_index_dev3 - ret_index_dev2;

	index = threadIdx.x + blockIdx.x * blockDim.x;

	for(i = index; i < imax; i += blockDim.x * gridDim.x)
	{
		ret_i = ret_index_dev0 + i;
		a_i   = mat_a_index_dev0 + i;
		b_i   = mat_b_index_dev0 + i;

		SET0_GDS(tmp_val);

		for(j = 0; j < jmax; j++)
		{
			ret_j = ret_index_dev2 + j;
			a_j = mat_a_index_dev2 + j;
			b_j = mat_b_index_dev2 + j;

//			tmp_val = mat_a_dev_element[a_i * a_col_dim + a_j] + mat_b_dev_element[b_i * b_col_dim + b_j];
//			ret_dev_element[ret_i * ret_col_dim + ret_j] = tmp_val;
			tmp_val = (gds_real)(mat_a_dev_element[a_i * a_col_dim + a_j]) + (gds_real)(mat_b_dev_element[b_i * b_col_dim + b_j]);
			ret_dev_element[ret_i * ret_col_dim + ret_j] = (gds_real)tmp_val;
		}
	}

	__syncthreads();

}

// Suppose that all arguments are allocated on GPU
void _bncuda_add_gdsmatrix_partial(GDSMatrix ret_dev, long int ret_index[4], GDSMatrix mat_a_dev, long int mat_a_index[4], GDSMatrix mat_b_dev, long int mat_b_index[4], int num_blocks_per_grid, int num_threads_per_block)
{
/***********************************************************
	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_dd(tmp_val[thread_index]);
	}

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < imax; i++)
	{
		thread_index = omp_get_thread_num();

		ret_i [thread_index]= ret_index[0] + i;
		a_i[thread_index] = mat_a_index[0] + i;
		b_i[thread_index] = mat_b_index[0] + i;
		//printf("i: %ld %ld %ld\n", ret_i, a_i, b_i);
		for(j = 0; j < jmax; j++)
		{
			//thread_index = omp_get_thread_num();

			ret_j[thread_index] = ret_index[2] + j;
			a_j[thread_index] = mat_a_index[2] + j;
			b_j[thread_index] = mat_b_index[2] + j;
			//printf("j: %ld %ld %ld\n", ret_j, a_j, b_j);

			//tmp_val = get_gdsmatrix_ij(mat_a, a_i, a_j) + get_gdsmatrix_ij(mat_b, b_i, b_j);
			rds_add(tmp_val[thread_index], get_gdsmatrix_ij(mat_a, a_i[thread_index], a_j[thread_index]), get_gdsmatrix_ij(mat_b, b_i[thread_index], b_j[thread_index]));
			set_gdsmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index], tmp_val[thread_index]);
		}
	}
*************************************************************/

/*
	_bncu_add_gdsmatrix_partial<<< num_blocks_per_grid, num_threads_per_block >>>(\
		ret_dev->element, ret_dev->row_dim, ret_dev->col_dim, ret_index_dev, \
		mat_a_dev->element, mat_a_dev->row_dim, mat_a_dev->col_dim, mat_a_index_dev, \
		mat_b_dev->element, mat_b_dev->row_dim, mat_b_dev->col_dim, mat_b_index_dev  \
	);
*/
	_bncu_add_gdsmatrix_partial<<< num_blocks_per_grid, num_threads_per_block >>>(\
		ret_dev->element, ret_dev->row_dim, ret_dev->col_dim, \
			ret_index[0], ret_index[1], ret_index[2], ret_index[3], \
		mat_a_dev->element, mat_a_dev->row_dim, mat_a_dev->col_dim, \
			mat_a_index[0], mat_a_index[1], mat_a_index[2], mat_a_index[3],\
		mat_b_dev->element, mat_b_dev->row_dim, mat_b_dev->col_dim, \
			mat_b_index[0], mat_b_index[1], mat_b_index[2], mat_b_index[3] \
	);

}

// partial add
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
__global__ void _bncu_add_gdsmatrix_partial(\
	gds_real *ret_dev_element, long int ret_row_dim, long int ret_col_dim, long int ret_index_dev[4], \
	gds_real *mat_a_dev_element, long int a_row_dim, long int a_col_dim, long int mat_a_index_dev[4], \
	gds_real *mat_b_dev_element, long int b_row_dim, long int b_col_dim, long int mat_b_index_dev[4]  \
)
{
	int index, i, j;
	long int imax, jmax, ret_i, ret_j, a_i, a_j, b_i, b_j;
	gds_real tmp_val;

	imax = ret_index_dev[1] - ret_index_dev[0];
	jmax = ret_index_dev[3] - ret_index_dev[2];

	index = threadIdx.x + blockIdx.x * blockDim.x;

	for(i = index; i < imax; i += blockDim.x * gridDim.x)
	{
		ret_i = ret_index_dev[0] + i;
		a_i   = mat_a_index_dev[0] + i;
		b_i   = mat_b_index_dev[0] + i;

		SET0_GDS(tmp_val);

		for(j = 0; j < jmax; j++)
		{
			ret_j = ret_index_dev[2] + j;
			a_j = mat_a_index_dev[2] + j;
			b_j = mat_b_index_dev[2] + j;

//			tmp_val = mat_a_dev_element[a_i * a_col_dim + a_j] + mat_b_dev_element[b_i * b_col_dim + b_j];
//			ret_dev_element[ret_i * ret_col_dim + ret_j] = tmp_val;
			tmp_val = (gds_real)(mat_a_dev_element[a_i * a_col_dim + a_j]) + (gds_real)(mat_b_dev_element[b_i * b_col_dim + b_j]);
			ret_dev_element[ret_i * ret_col_dim + ret_j] = (gds_real)tmp_val;
		}
	}

	__syncthreads();

}

// Suppose that all arguments are allocated on GPU
void _bncuda_add_gdsmatrix_partial_dev(GDSMatrix ret_dev, long int ret_index_dev[4], GDSMatrix mat_a_dev, long int mat_a_index_dev[4], GDSMatrix mat_b_dev, long int mat_b_index_dev[4], int num_blocks_per_grid, int num_threads_per_block)
{
/***********************************************************
	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_dd(tmp_val[thread_index]);
	}

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < imax; i++)
	{
		thread_index = omp_get_thread_num();

		ret_i [thread_index]= ret_index[0] + i;
		a_i[thread_index] = mat_a_index[0] + i;
		b_i[thread_index] = mat_b_index[0] + i;
		//printf("i: %ld %ld %ld\n", ret_i, a_i, b_i);
		for(j = 0; j < jmax; j++)
		{
			//thread_index = omp_get_thread_num();

			ret_j[thread_index] = ret_index[2] + j;
			a_j[thread_index] = mat_a_index[2] + j;
			b_j[thread_index] = mat_b_index[2] + j;
			//printf("j: %ld %ld %ld\n", ret_j, a_j, b_j);

			//tmp_val = get_gdsmatrix_ij(mat_a, a_i, a_j) + get_gdsmatrix_ij(mat_b, b_i, b_j);
			rds_add(tmp_val[thread_index], get_gdsmatrix_ij(mat_a, a_i[thread_index], a_j[thread_index]), get_gdsmatrix_ij(mat_b, b_i[thread_index], b_j[thread_index]));
			set_gdsmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index], tmp_val[thread_index]);
		}
	}
*************************************************************/


	_bncu_add_gdsmatrix_partial<<< num_blocks_per_grid, num_threads_per_block >>>(\
		ret_dev->element, ret_dev->row_dim, ret_dev->col_dim, ret_index_dev, \
		mat_a_dev->element, mat_a_dev->row_dim, mat_a_dev->col_dim, mat_a_index_dev, \
		mat_b_dev->element, mat_b_dev->row_dim, mat_b_dev->col_dim, mat_b_index_dev  \
	);
}


// partial sub
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
__global__ void _bncu_sub_gdsmatrix_partial(\
	gds_real *ret_dev_element, long int ret_row_dim, long int ret_col_dim, \
		long int ret_index_dev0, long int ret_index_dev1, long int ret_index_dev2, long int ret_index_dev3,\
	gds_real *mat_a_dev_element, long int a_row_dim, long int a_col_dim, \
		long int mat_a_index_dev0, long int mat_a_index_dev1, long int mat_a_index_dev2, long int mat_a_index_dev3,\
	gds_real *mat_b_dev_element, long int b_row_dim, long int b_col_dim, \
		long int mat_b_index_dev0, long int mat_b_index_dev1, long int mat_b_index_dev2, long int mat_b_index_dev3 \
)
{
	int index, i, j;
	long int imax, jmax, ret_i, ret_j, a_i, a_j, b_i, b_j;
	gds_real tmp_val;

	imax = ret_index_dev1 - ret_index_dev0;
	jmax = ret_index_dev3 - ret_index_dev2;

	index = threadIdx.x + blockIdx.x * blockDim.x;

	for(i = index; i < imax; i += blockDim.x * gridDim.x)
	{
		ret_i = ret_index_dev0 + i;
		a_i = mat_a_index_dev0 + i;
		b_i = mat_b_index_dev0 + i;

		SET0_GDS(tmp_val);

		for(j = 0; j < jmax; j++)
		{
			ret_j = ret_index_dev2 + j;
			a_j = mat_a_index_dev2 + j;
			b_j = mat_b_index_dev2 + j;

			//tmp_val = mat_a_dev_element[a_i * a_col_dim + a_j] - mat_b_dev_element[b_i * b_col_dim + b_j];
			//ret_dev_element[ret_i * ret_col_dim + ret_j] = tmp_val;
			//ret_dev_element[ret_i * ret_col_dim + ret_j] = (gds_real)(mat_a_dev_element[a_i * a_col_dim + a_j]) - (gds_real)(mat_b_dev_element[b_i * b_col_dim + b_j]);
			tmp_val = (gds_real)(mat_a_dev_element[a_i * a_col_dim + a_j]) - (gds_real)(mat_b_dev_element[b_i * b_col_dim + b_j]);
			ret_dev_element[ret_i * ret_col_dim + ret_j] = (gds_real)tmp_val;
		}
	}

	__syncthreads();

}

void _bncuda_sub_gdsmatrix_partial(GDSMatrix ret_dev, long int ret_index[4], GDSMatrix mat_a_dev, long int mat_a_index[4], GDSMatrix mat_b_dev, long int mat_b_index[4], int num_blocks_per_grid, int num_threads_per_block)
{
/***********************************************************
	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_dd(tmp_val[thread_index]);
	}

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < imax; i++)
	{
		thread_index = omp_get_thread_num();

		ret_i[thread_index] = ret_index[0] + i;
		a_i[thread_index] = mat_a_index[0] + i;
		b_i[thread_index] = mat_b_index[0] + i;
		for(j = 0; j < jmax; j++)
		{
			ret_j[thread_index] = ret_index[2] + j;
			a_j[thread_index] = mat_a_index[2] + j;
			b_j[thread_index] = mat_b_index[2] + j;

			//tmp_val = get_gdsmatrix_ij(mat_a, a_i, a_j) - get_gdsmatrix_ij(mat_b, b_i, b_j);
			rds_sub(tmp_val[thread_index], get_gdsmatrix_ij(mat_a, a_i[thread_index], a_j[thread_index]), get_gdsmatrix_ij(mat_b, b_i[thread_index], b_j[thread_index]));
			set_gdsmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index], tmp_val[thread_index]);
		}
	}
*************************************************************/

/*
	_bncu_sub_gdsmatrix_partial<<< num_blocks_per_grid, num_threads_per_block >>>(\
		ret_dev->element, ret_dev->row_dim, ret_dev->col_dim, ret_index_dev, \
		mat_a_dev->element, mat_a_dev->row_dim, mat_a_dev->col_dim, mat_a_index_dev, \
		mat_b_dev->element, mat_b_dev->row_dim, mat_b_dev->col_dim, mat_b_index_dev  \
	);
*/

	_bncu_sub_gdsmatrix_partial<<< num_blocks_per_grid, num_threads_per_block >>>(\
		ret_dev->element, ret_dev->row_dim, ret_dev->col_dim, \
			ret_index[0], ret_index[1], ret_index[2], ret_index[3],\
		mat_a_dev->element, mat_a_dev->row_dim, mat_a_dev->col_dim, \
			mat_a_index[0], mat_a_index[1], mat_a_index[2], mat_a_index[3],\
		mat_b_dev->element, mat_b_dev->row_dim, mat_b_dev->col_dim, \
			mat_b_index[0], mat_b_index[1], mat_b_index[2], mat_b_index[3] \
	);

}

// partial sub
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
__global__ void _bncu_sub_gdsmatrix_partial(\
	gds_real *ret_dev_element, long int ret_row_dim, long int ret_col_dim, long int ret_index_dev[4], \
	gds_real *mat_a_dev_element, long int a_row_dim, long int a_col_dim, long int mat_a_index_dev[4], \
	gds_real *mat_b_dev_element, long int b_row_dim, long int b_col_dim, long int mat_b_index_dev[4]  \
)
{
	int index, i, j;
	long int imax, jmax, ret_i, ret_j, a_i, a_j, b_i, b_j;
	gds_real tmp_val;

	imax = ret_index_dev[1] - ret_index_dev[0];
	jmax = ret_index_dev[3] - ret_index_dev[2];

	index = threadIdx.x + blockIdx.x * blockDim.x;

	for(i = index; i < imax; i += blockDim.x * gridDim.x)
	{
		ret_i = ret_index_dev[0] + i;
		a_i = mat_a_index_dev[0] + i;
		b_i = mat_b_index_dev[0] + i;

		SET0_GDS(tmp_val);

		for(j = 0; j < jmax; j++)
		{
			ret_j = ret_index_dev[2] + j;
			a_j = mat_a_index_dev[2] + j;
			b_j = mat_b_index_dev[2] + j;

			//tmp_val = mat_a_dev_element[a_i * a_col_dim + a_j] - mat_b_dev_element[b_i * b_col_dim + b_j];
			//ret_dev_element[ret_i * ret_col_dim + ret_j] = tmp_val;
			//ret_dev_element[ret_i * ret_col_dim + ret_j] = (gds_real)(mat_a_dev_element[a_i * a_col_dim + a_j]) - (gds_real)(mat_b_dev_element[b_i * b_col_dim + b_j]);
			tmp_val = (gds_real)(mat_a_dev_element[a_i * a_col_dim + a_j]) - (gds_real)(mat_b_dev_element[b_i * b_col_dim + b_j]);
			ret_dev_element[ret_i * ret_col_dim + ret_j] = (gds_real)tmp_val;
		}
	}

	__syncthreads();

}

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
__global__ void _bncu_subst_gdsmatrix_partial(\
	gds_real *ret_dev_element, long int ret_row_dim, long int ret_col_dim, \
		long int ret_index_dev0, long int ret_index_dev1, long int ret_index_dev2, long int ret_index_dev3,\
	gds_real *mat_a_dev_element, long int a_row_dim, long int a_col_dim, \
		long int mat_a_index_dev0, long int mat_a_index_dev1, long int mat_a_index_dev2, long int mat_a_index_dev3 \
)
{
	int index, i, j;
	long int imax, jmax, ret_i, ret_j, a_i, a_j;

	imax = ret_index_dev1 - ret_index_dev0;
	jmax = ret_index_dev3 - ret_index_dev2;

	index = threadIdx.x + blockIdx.x * blockDim.x;

	for(i = index; i < imax; i += blockDim.x * gridDim.x)
	{
		ret_i = ret_index_dev0 + i;
		a_i   = mat_a_index_dev0 + i;

		for(j = 0; j < jmax; j++)
		{
			ret_j = ret_index_dev2 + j;
			a_j = mat_a_index_dev2 + j;

			ret_dev_element[ret_i * ret_col_dim + ret_j] = mat_a_dev_element[a_i * a_col_dim + a_j];
		}
	}

	__syncthreads();

}

void _bncuda_subst_gdsmatrix_partial(GDSMatrix ret_dev, long int ret_index[4], GDSMatrix mat_a_dev, long int mat_a_index[4], int num_blocks_per_grid, int num_threads_per_block)
{
/***********************************************************
	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	thread_num = omp_get_num_threads();

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < imax; i++)
	{
		thread_index = omp_get_thread_num();

		ret_i[thread_index] = ret_index[0] + i;
		a_i[thread_index] = mat_a_index[0] + i;
		for(j = 0; j < jmax; j++)
		{
			ret_j[thread_index] = ret_index[2] + j;
			a_j[thread_index] = mat_a_index[2] + j;

			set_gdsmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index], get_gdsmatrix_ij(mat_a, a_i[thread_index], a_j[thread_index]));
		}
	}
*************************************************************/

	_bncu_subst_gdsmatrix_partial<<< num_blocks_per_grid, num_threads_per_block >>>(\
		ret_dev->element, ret_dev->row_dim, ret_dev->col_dim, \
			ret_index[0], ret_index[1], ret_index[2], ret_index[3], \
		mat_a_dev->element, mat_a_dev->row_dim, mat_a_dev->col_dim, \
			mat_a_index[0], mat_a_index[1], mat_a_index[2], mat_a_index[3] \
	);

}

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
__global__ void _bncu_subst_gdsmatrix_partial(\
	gds_real *ret_dev_element, long int ret_row_dim, long int ret_col_dim, long int ret_index_dev[4], \
	gds_real *mat_a_dev_element, long int a_row_dim, long int a_col_dim, long int mat_a_index_dev[4]  \
)
{
	int index, i, j;
	long int imax, jmax, ret_i, ret_j, a_i, a_j;

	imax = ret_index_dev[1] - ret_index_dev[0];
	jmax = ret_index_dev[3] - ret_index_dev[2];

	index = threadIdx.x + blockIdx.x * blockDim.x;

	for(i = index; i < imax; i += blockDim.x * gridDim.x)
	{
		ret_i = ret_index_dev[0] + i;
		a_i   = mat_a_index_dev[0] + i;

		for(j = 0; j < jmax; j++)
		{
			ret_j = ret_index_dev[2] + j;
			a_j = mat_a_index_dev[2] + j;

			ret_dev_element[ret_i * ret_col_dim + ret_j] = mat_a_dev_element[a_i * a_col_dim + a_j];
		}
	}

	__syncthreads();

}

void _bncuda_subst_gdsmatrix_partial_dev(GDSMatrix ret_dev, long int ret_index_dev[4], GDSMatrix mat_a_dev, long int mat_a_index_dev[4], int num_blocks_per_grid, int num_threads_per_block)
{
/***********************************************************
	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	thread_num = omp_get_num_threads();

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < imax; i++)
	{
		thread_index = omp_get_thread_num();

		ret_i[thread_index] = ret_index[0] + i;
		a_i[thread_index] = mat_a_index[0] + i;
		for(j = 0; j < jmax; j++)
		{
			ret_j[thread_index] = ret_index[2] + j;
			a_j[thread_index] = mat_a_index[2] + j;

			set_gdsmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index], get_gdsmatrix_ij(mat_a, a_i[thread_index], a_j[thread_index]));
		}
	}
*************************************************************/

	_bncu_subst_gdsmatrix_partial<<< num_blocks_per_grid, num_threads_per_block >>>(\
		ret_dev->element, ret_dev->row_dim, ret_dev->col_dim, ret_index_dev, \
		mat_a_dev->element, mat_a_dev->row_dim, mat_a_dev->col_dim, mat_a_index_dev \
	);

}

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
__global__ void _bncu_subst_gdsmatrix_partial_checked( \
	gds_real *ret_dev_element, long int ret_row_dim, long int ret_col_dim, \
		long int ret_index_dev0, long int ret_index_dev1, long int ret_index_dev2, long int ret_index_dev3, \
	gds_real *mat_a_dev_element, long int mat_a_row_dim, long int mat_a_col_dim, \
		long int mat_a_index_dev0, long int mat_a_index_dev1, long int mat_a_index_dev2, long int mat_a_index_dev3 \
)
{
	long int i, j;
	long int ret_i, ret_j, a_i, a_j;
	long int imax, jmax;
	int index;

	imax = ret_index_dev1 - ret_index_dev0;
	jmax = ret_index_dev3 - ret_index_dev2;

	index = threadIdx.x + blockIdx.x * blockDim.x;

	for(i = index; i < imax; i += blockDim.x * gridDim.x)
	{
		ret_i = ret_index_dev0 + i;
		a_i = mat_a_index_dev0 + i;

		if((ret_i >= 0) && (ret_i < ret_row_dim))
		{
			for(j = 0; j < jmax; j++)
			{
				ret_j = ret_index_dev2 + j;
				a_j = mat_a_index_dev2 + j;

				if((ret_j >= 0) && (ret_j < ret_col_dim))
				{
					if((a_i >= 0) && (a_i < mat_a_row_dim) && (a_j >= 0) && (a_j < mat_a_col_dim))
						ret_dev_element[ret_i * ret_col_dim + ret_j] = mat_a_dev_element[a_i * mat_a_col_dim + a_j];
					else
						SET0_GDS(ret_dev_element[ret_i * ret_col_dim + ret_j]); // Padding
				}
			}
		}
	}

	__syncthreads();

}

void _bncuda_subst_gdsmatrix_partial_checked(GDSMatrix ret_dev, long int ret_index[4], GDSMatrix mat_a_dev, long int mat_a_index[4], int num_blocks_per_grid, int num_threads_per_block)
{
/***********************************************************
	int thread_num, thread_index;
	long int i, j;
	long int ret_i[BNCOMP_MAX_NUM_THREADS], ret_j[BNCOMP_MAX_NUM_THREADS], a_i[BNCOMP_MAX_NUM_THREADS], a_j[BNCOMP_MAX_NUM_THREADS];
	long int imax, jmax;
#ifdef __cplusplus
	static gds_real dd_tmp;
#else // __cplusplus
	double *ptr_ddtmp;
#endif // _cplusplus

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < imax; i++)
	{
		thread_index = omp_get_thread_num();

		ret_i[thread_index] = ret_index[0] + i;
		a_i[thread_index] = mat_a_index[0] + i;
		if((ret_i[thread_index] >= 0) && (ret_i[thread_index] < ret->row_dim))
		{
			for(j = 0; j < jmax; j++)
			{
				ret_j[thread_index] = ret_index[2] + j;
				a_j[thread_index] = mat_a_index[2] + j;
				if((ret_j[thread_index] >= 0) && (ret_j[thread_index] < ret->col_dim))
				{
					if((a_i[thread_index] >= 0) && (a_i[thread_index] < mat_a->row_dim) && (a_j[thread_index] >= 0) && (a_j[thread_index] < mat_a->col_dim))
					{
						set_gdsmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index], get_gdsmatrix_ij(mat_a, a_i[thread_index], a_j[thread_index]));
					}
					else
						set0_gdsmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index]); // Padding
					//printf("Warning: ret_index = %d, %d, %d, %d\n", ret_i, ret_j, a_i, a_j);
				}
			}
		}
	}
*************************************************************/

	_bncu_subst_gdsmatrix_partial_checked<<< num_blocks_per_grid, num_threads_per_block >>>(\
		ret_dev->element, ret_dev->row_dim, ret_dev->col_dim, \
			ret_index[0], ret_index[1], ret_index[2], ret_index[3], \
		mat_a_dev->element, mat_a_dev->row_dim, mat_a_dev->col_dim, \
			mat_a_index[0], mat_a_index[1], mat_a_index[2], mat_a_index[3] \
	);

}

// partial set
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
__global__ void _bncu_subst_gdsmatrix_partial_checked(gds_real *ret_dev_element, long int ret_row_dim, long int ret_col_dim, long int ret_index_dev[4], gds_real *mat_a_dev_element, long int mat_a_row_dim, long int mat_a_col_dim, long int mat_a_index_dev[4])
{
	long int i, j;
	long int ret_i, ret_j, a_i, a_j;
	long int imax, jmax;
	int index;

	imax = ret_index_dev[1] - ret_index_dev[0];
	jmax = ret_index_dev[3] - ret_index_dev[2];

	index = threadIdx.x + blockIdx.x * blockDim.x;

	for(i = index; i < imax; i += blockDim.x * gridDim.x)
	{
		ret_i = ret_index_dev[0] + i;
		a_i = mat_a_index_dev[0] + i;

		if((ret_i >= 0) && (ret_i < ret_row_dim))
		{
			for(j = 0; j < jmax; j++)
			{
				ret_j = ret_index_dev[2] + j;
				a_j = mat_a_index_dev[2] + j;

				if((ret_j >= 0) && (ret_j < ret_col_dim))
				{
					if((a_i >= 0) && (a_i < mat_a_row_dim) && (a_j >= 0) && (a_j < mat_a_col_dim))
						ret_dev_element[ret_i * ret_col_dim + ret_j] = mat_a_dev_element[a_i * mat_a_col_dim + a_j];
					else
						SET0_GDS(ret_dev_element[ret_i * ret_col_dim + ret_j]); // Padding
				}
			}
		}
	}

	__syncthreads();

}

void _bncuda_subst_gdsmatrix_partial_checked_dev(GDSMatrix ret_dev, long int ret_index_dev[4], GDSMatrix mat_a_dev, long int mat_a_index_dev[4], int num_blocks_per_grid, int num_threads_per_block)
{
/***********************************************************
	int thread_num, thread_index;
	long int i, j;
	long int ret_i[BNCOMP_MAX_NUM_THREADS], ret_j[BNCOMP_MAX_NUM_THREADS], a_i[BNCOMP_MAX_NUM_THREADS], a_j[BNCOMP_MAX_NUM_THREADS];
	long int imax, jmax;
#ifdef __cplusplus
	static gds_real dd_tmp;
#else // __cplusplus
	double *ptr_ddtmp;
#endif // _cplusplus

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < imax; i++)
	{
		thread_index = omp_get_thread_num();

		ret_i[thread_index] = ret_index[0] + i;
		a_i[thread_index] = mat_a_index[0] + i;
		if((ret_i[thread_index] >= 0) && (ret_i[thread_index] < ret->row_dim))
		{
			for(j = 0; j < jmax; j++)
			{
				ret_j[thread_index] = ret_index[2] + j;
				a_j[thread_index] = mat_a_index[2] + j;
				if((ret_j[thread_index] >= 0) && (ret_j[thread_index] < ret->col_dim))
				{
					if((a_i[thread_index] >= 0) && (a_i[thread_index] < mat_a->row_dim) && (a_j[thread_index] >= 0) && (a_j[thread_index] < mat_a->col_dim))
					{
						set_gdsmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index], get_gdsmatrix_ij(mat_a, a_i[thread_index], a_j[thread_index]));
					}
					else
						set0_gdsmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index]); // Padding
					//printf("Warning: ret_index = %d, %d, %d, %d\n", ret_i, ret_j, a_i, a_j);
				}
			}
		}
	}
*************************************************************/

	_bncu_subst_gdsmatrix_partial_checked<<< num_blocks_per_grid, num_threads_per_block >>>(\
		ret_dev->element, ret_dev->row_dim, ret_dev->col_dim, ret_index_dev, \
		mat_a_dev->element, mat_a_dev->row_dim, mat_a_dev->col_dim, mat_a_index_dev \
	);

}

// reverse sign
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
__global__ void _bncu_neg_gdsmatrix_partial(\
	gds_real *ret_dev_element, long int ret_row_dim, long int ret_col_dim, \
		long int ret_index_dev0, long int ret_index_dev1, long int ret_index_dev2, long int ret_index_dev3,\
	gds_real *mat_a_dev_element, long int a_row_dim, long int a_col_dim, \
		long int mat_a_index_dev0, long int mat_a_index_dev1, long int mat_a_index_dev2, long int mat_a_index_dev3 \
)
{
	int index, i, j;
	long int imax, jmax, ret_i, ret_j, a_i, a_j;

	imax = ret_index_dev1 - ret_index_dev0;
	jmax = ret_index_dev3 - ret_index_dev2;

	index = threadIdx.x + blockIdx.x * blockDim.x;

	for(i = index; i < imax; i += blockDim.x * gridDim.x)
	{
		ret_i = ret_index_dev0 + i;
		a_i   = mat_a_index_dev0 + i;

		for(j = 0; j < jmax; j++)
		{
			ret_j = ret_index_dev2 + j;
			a_j = mat_a_index_dev2 + j;

			NEG_GDS(ret_dev_element[ret_i * ret_col_dim + ret_j], mat_a_dev_element[a_i * a_col_dim + a_j]);
		}
	}

	__syncthreads();

}

void _bncuda_neg_gdsmatrix_partial(GDSMatrix ret_dev, long int ret_index[4], GDSMatrix mat_a_dev, long int mat_a_index[4], int num_blocks_per_grid, int num_threads_per_block)
{
/***********************************************************
	int thread_num, thread_index;
	long int i, j;
	long int ret_i[BNCOMP_MAX_NUM_THREADS], ret_j[BNCOMP_MAX_NUM_THREADS], a_i[BNCOMP_MAX_NUM_THREADS], a_j[BNCOMP_MAX_NUM_THREADS];
	long int imax, jmax;
#ifdef __cplusplus
	gds_real tmp[BNCOMP_MAX_NUM_THREADS];
#else
	double tmp[BNCOMP_MAX_NUM_THREADS][DSSIZE];
#endif // __cplusplus

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_dd(tmp[thread_index]);
	}

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < imax; i++)
	{
		thread_index = omp_get_thread_num();

		ret_i[thread_index] = ret_index[0] + i;
		a_i[thread_index] = mat_a_index[0] + i;
		for(j = 0; j < jmax; j++)
		{
			ret_j[thread_index] = ret_index[2] + j;
			a_j[thread_index] = mat_a_index[2] + j;
			set_gdsmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index], -get_gdsmatrix_ij(mat_a, a_i[thread_index], a_j[thread_index]));
		}
	}
*************************************************************/

	_bncu_neg_gdsmatrix_partial<<< num_blocks_per_grid, num_threads_per_block >>>(\
		ret_dev->element, ret_dev->row_dim, ret_dev->col_dim, \
			ret_index[0], ret_index[1], ret_index[2], ret_index[3], \
		mat_a_dev->element, mat_a_dev->row_dim, mat_a_dev->col_dim, \
			mat_a_index[0], mat_a_index[1], mat_a_index[2], mat_a_index[3] \
	);

}

// reverse sign
// *_index[0] = start_row_index
// *_index[1] = end_row_index
// *_index[2] = start_col_index
// *_index[3] = end_col_index
__global__ void _bncu_neg_gdsmatrix_partial(\
	gds_real *ret_dev_element, long int ret_row_dim, long int ret_col_dim, long int ret_index_dev[4], \
	gds_real *mat_a_dev_element, long int a_row_dim, long int a_col_dim, long int mat_a_index_dev[4]  \
)
{
	int index, i, j;
	long int imax, jmax, ret_i, ret_j, a_i, a_j;

	imax = ret_index_dev[1] - ret_index_dev[0];
	jmax = ret_index_dev[3] - ret_index_dev[2];

	index = threadIdx.x + blockIdx.x * blockDim.x;

	for(i = index; i < imax; i += blockDim.x * gridDim.x)
	{
		ret_i = ret_index_dev[0] + i;
		a_i   = mat_a_index_dev[0] + i;

		for(j = 0; j < jmax; j++)
		{
			ret_j = ret_index_dev[2] + j;
			a_j = mat_a_index_dev[2] + j;

			NEG_GDS(ret_dev_element[ret_i * ret_col_dim + ret_j], mat_a_dev_element[a_i * a_col_dim + a_j]);
		}
	}

	__syncthreads();

}

void _bncuda_neg_gdsmatrix_partial_dev(GDSMatrix ret_dev, long int ret_index_dev[4], GDSMatrix mat_a_dev, long int mat_a_index_dev[4], int num_blocks_per_grid, int num_threads_per_block)
{
/***********************************************************
	int thread_num, thread_index;
	long int i, j;
	long int ret_i[BNCOMP_MAX_NUM_THREADS], ret_j[BNCOMP_MAX_NUM_THREADS], a_i[BNCOMP_MAX_NUM_THREADS], a_j[BNCOMP_MAX_NUM_THREADS];
	long int imax, jmax;
#ifdef __cplusplus
	gds_real tmp[BNCOMP_MAX_NUM_THREADS];
#else
	double tmp[BNCOMP_MAX_NUM_THREADS][DSSIZE];
#endif // __cplusplus

	imax = ret_index[1] - ret_index[0];
	jmax = ret_index[3] - ret_index[2];

	thread_num = omp_get_num_threads();

	#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		set0_dd(tmp[thread_index]);
	}

	#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < imax; i++)
	{
		thread_index = omp_get_thread_num();

		ret_i[thread_index] = ret_index[0] + i;
		a_i[thread_index] = mat_a_index[0] + i;
		for(j = 0; j < jmax; j++)
		{
			ret_j[thread_index] = ret_index[2] + j;
			a_j[thread_index] = mat_a_index[2] + j;
			set_gdsmatrix_ij(ret, ret_i[thread_index], ret_j[thread_index], -get_gdsmatrix_ij(mat_a, a_i[thread_index], a_j[thread_index]));
		}
	}
*************************************************************/

	_bncu_neg_gdsmatrix_partial<<< num_blocks_per_grid, num_threads_per_block >>>(\
		ret_dev->element, ret_dev->row_dim, ret_dev->col_dim, ret_index_dev, \
		mat_a_dev->element, mat_a_dev->row_dim, mat_a_dev->col_dim, mat_a_index_dev \
	);

}
