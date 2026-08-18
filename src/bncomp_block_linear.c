/********************************************************************************/
/* bncomp_block_linear.c: Parallelized Linear computations with block matrix    */
/* Copyright (c) 2016 Tomonori Kouya                                            */
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
/**************************************************/
/* Implicit Runge-Kutta Method based on Jay's way */
/**************************************************/
#include <stdio.h>
#include <math.h>

#include "bnc.h"

#ifdef USE_ATLAS
  #include "clapack.h"
#elif USE_IMKL
  #include "mkl_types.h"
  #include "mkl_cblas.h"
  #include "mkl_lapacke.h"
#endif

#ifdef USE_BIRK
#include "birk.h"
#endif

#ifdef USE_PTHREAD
#include "birk_pthread.h"
#endif

// BNCpack with OpenMP
#include "bncomp.h"

#ifndef _BNCOMP_DEFINE_BNCOMP_GVALS
#define _BNCOMP_DEFINE_BNCOMP_GVALS

double  _bncomp_g_dval[BNCOMP_MAX_NUM_THREADS];
DVector _bncomp_g_dvec[BNCOMP_MAX_NUM_THREADS];
DMatrix _bncomp_g_dmat[BNCOMP_MAX_NUM_THREADS];

// Global working variables for multiple precision arithmetic
#ifdef USE_GMP
mpf_t     _bncomp_g_mpfval[BNCOMP_MAX_NUM_THREADS];
mpf_t     _bncomp_g_mpfval2[BNCOMP_MAX_NUM_THREADS];
MPFVector _bncomp_g_mpfvec[BNCOMP_MAX_NUM_THREADS];
MPFVector _bncomp_g_mpfvec2[BNCOMP_MAX_NUM_THREADS];
MPFVector _bncomp_g_mpfvec3[BNCOMP_MAX_NUM_THREADS];
MPFVector _bncomp_g_mpfvec4[BNCOMP_MAX_NUM_THREADS];
MPFMatrix _bncomp_g_mpfmat[BNCOMP_MAX_NUM_THREADS];
#endif // USE_GMP

#endif // _BNCOMP_DEFINE_BNCOMP_GVALS

/* initialize */
void _bncomp_init_g_d(int vec_dim, int mat_row_dim, int mat_col_dim)
{
	int i, num_threads;

/*	_bncomp_num_tmp = max_num_bncomp_tmp;
	if(max_num_bncomp_tmp >= MAX_NUM_BNCOMP_TMP)
	{
		fprintf(stderr, "Warning: max_num_bncomp_tmp(%d) is exceeded! %d temporary variable is usable.\n", max_num_bncomp_tmp, MAX_NUM_BNCOMP_TMP);
		_bncomp_num_tmp = MAX_NUM_BNCOMP_TMP;
	}
*/
	num_threads = omp_get_max_threads();
	
	#pragma omp parallel for
	for(i = 0; i < num_threads; i++)
	{
		_bncomp_g_dval[i] = 0.0;
		_bncomp_g_dvec[i] = init_dvector(vec_dim);
		_bncomp_g_dmat[i] = init_dmatrix(mat_row_dim, mat_col_dim);
	}
}

/* finalize */
void _bncomp_free_g_d(void)
{
	int i, num_threads;

	num_threads = omp_get_max_threads();

	#pragma omp parallel for
	for(i = 0; i < num_threads; i++)
	{
		free_dvector(_bncomp_g_dvec[i]);
		free_dmatrix(_bncomp_g_dmat[i]);
	}
}	

#ifdef USE_GMP
/* initialize */
void _bncomp_init_g_mpf(int vec_dim, int mat_row_dim, int mat_col_dim)
{
	int i, num_threads;

	num_threads = omp_get_max_threads();
	printf("max #threads = %d\n", num_threads);

/*	_bncomp_num_tmp = max_num_bncomp_tmp;
	if(max_num_bncomp_tmp >= MAX_NUM_BNCOMP_TMP)
	{
		fprintf(stderr, "Warning: max_num_bncomp_tmp(%d) is exceeded! %d temporary variable is usable.\n", max_num_bncomp_tmp, MAX_NUM_BNCOMP_TMP);
		_bncomp_num_tmp = MAX_NUM_BNCOMP_TMP;
	}
*/
	#pragma omp parallel for
	for(i = 0; i < num_threads; i++)
	{
		mpf_init_set_d(_bncomp_g_mpfval[i], 0.0);
		mpf_init_set_d(_bncomp_g_mpfval2[i], 0.0);
		_bncomp_g_mpfvec[i] = init_mpfvector(vec_dim);
		_bncomp_g_mpfvec2[i] = init_mpfvector(vec_dim);
		_bncomp_g_mpfvec3[i] = init_mpfvector(vec_dim);
		_bncomp_g_mpfvec4[i] = init_mpfvector(vec_dim);
		_bncomp_g_mpfmat[i] = init_mpfmatrix(mat_row_dim, mat_col_dim);
	}
}

/* finalize */
void _bncomp_free_g_mpf(void)
{
	int i, num_threads;

	num_threads = omp_get_max_threads();

	#pragma omp parallel for
	for(i = 0; i < num_threads; i++)
	{
		mpf_clear(_bncomp_g_mpfval[i]);
		mpf_clear(_bncomp_g_mpfval2[i]);
		free_mpfvector(_bncomp_g_mpfvec[i]);
		free_mpfvector(_bncomp_g_mpfvec2[i]);
		free_mpfvector(_bncomp_g_mpfvec3[i]);
		free_mpfvector(_bncomp_g_mpfvec4[i]);
		free_mpfmatrix(_bncomp_g_mpfmat[i]);
	}
}	
#endif

/* substitute vector blocks */
void _bncomp_subst_dvector_blocks(DVector ret[], DVector v[], long int num_blocks)
{
	long int i;

	#pragma omp parallel for
	for(i = 0; i < num_blocks; i++)
		subst_dvector(ret[i], v[i]);
}

/* inner product of vector blocks */
double _bncomp_ip_dvector_blocks(DVector va[], DVector vb[], long int num_blocks)
{
	long int i;
	double ret, tmp[BNCOMP_MAX_NUM_THREADS];
	int thread_index;

#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();
	tmp[thread_index] = 0.0;
}

	ret = 0.0;

#pragma omp parallel for private(thread_index)
	for(i = 0; i < num_blocks; i++)
	{
		thread_index = omp_get_thread_num();
		tmp[thread_index] = ip_dvector(va[i], vb[i]);
#pragma omp critical
		ret += tmp[thread_index];
	}

	return ret;
}

/* norm2 of vector blocks */
double _bncomp_norm2_dvector_blocks(DVector v[], long int num_blocks)
{
	long int i, j;
	double ret, tmp[BNCOMP_MAX_NUM_THREADS];
	int thread_index;

#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();
	tmp[thread_index] = 0.0;
}

	ret = 0.0;

#pragma omp parallel for private(j, thread_index)
	for(i = 0; i < num_blocks; i++)
	{
		thread_index = omp_get_thread_num();

		tmp[thread_index] = 0.0;
		for(j = 0; j < v[i]->dim; j++)
			tmp[thread_index] += get_dvector_i(v[i], j) * get_dvector_i(v[i], j);

#pragma omp critical
		ret += tmp[thread_index];
	}

	ret = sqrt(ret);

	return ret;
}

/* ret := va + alpha * vb */
void _bncomp_add_cmul_dvector_blocks(DVector ret[], DVector va[], double alpha, DVector vb[], long int num_blocks)
{
	long int i;

#pragma omp parallel for
	for(i = 0; i < num_blocks; i++)
		add_cmul_dvector(ret[i], va[i], alpha, vb[i]);

	return;
}

/* ret := (A \otimes I) vb[] */
void _bncomp_kmul_dmatrixI_dvector_blocks(DVector ret[], DMatrix mat, DVector vb[], long int num_blocks)
{
	long int i, j;

	if(mat->col_dim > num_blocks)
	{
		fprintf(stderr, "kmul_dmatrixI_dvector_blocks: mat->col_dim(%ld) is larger than num_blocks(%ld)!\n", mat->col_dim, num_blocks);
		return;
	}

#pragma omp parallel for private(j)
	for(i = 0; i < mat->row_dim; i++)
	{
		set0_dvector(ret[i]);
		for(j = 0; j < mat->col_dim; j++)
			add_cmul_dvector(ret[i], ret[i], get_dmatrix_ij(mat, i, j), vb[j]);
	}
}

/* ret := -vb */
void _bncomp_neg_dvector_blocks(DVector vb[], long int num_blocks)
{
	long int i;

#pragma omp parallel for
	for(i = 0; i < num_blocks; i++)
		neg_dvector(vb[i]);
}

/* norm_inf of vector blocks */
double _bncomp_normi_dvector_blocks(DVector v[], long int num_blocks)
{
	long int i, j;
	double ret, tmp[BNCOMP_MAX_NUM_THREADS], tmp2[BNCOMP_MAX_NUM_THREADS];
	int thread_index;

	ret = 0.0;

#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < num_blocks; i++)
	{
		thread_index = omp_get_thread_num();
		tmp2[thread_index] = 0.0;
		for(j = 0; j < v[i]->dim; j++)
		{
			tmp[thread_index] = fabs(get_dvector_i(v[i], j));
			if(tmp2[thread_index] < tmp[thread_index]);
				tmp2[thread_index] = tmp[thread_index];
		}
#pragma omp critical
	{
		if(ret < tmp2[thread_index])
			ret = tmp2[thread_index];
	}
	}

	return ret;
}

/* ret := va - vb */
void _bncomp_sub_dvector_blocks(DVector ret[], DVector va[], DVector vb[], long int num_blocks)
{
	long int i;

#pragma omp parallel for
	for(i = 0; i < num_blocks; i++)
		sub_dvector(ret[i], va[i], vb[i]);
}

/* ret := va + vb */
void _bncomp_add_dvector_blocks(DVector ret[], DVector va[], DVector vb[], long int num_blocks)
{
	long int i;

#pragma omp parallel for
	for(i = 0; i < num_blocks; i++)
		add_dvector(ret[i], va[i], vb[i]);
}

/* ret := sum^n_{i,j=1} aij^2 */
double sumsqr_dmatrix(DMatrix mat)
{
	long int i, j;
	double ret;

	ret = 0.0;

	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
			ret += get_dmatrix_ij(mat, i, j) * get_dmatrix_ij(mat, i, j);
	}

	return ret;
}

/* ret := sum^n_{i,j=1} aij^2 */
double _bncomp_sumsqr_dmatrix(DMatrix mat)
{
	long int i, j;
	double ret, tmp[BNCOMP_MAX_NUM_THREADS];

	ret = 0.0;

#pragma omp parallel for private(j)
	for(i = 0; i < mat->row_dim; i++)
	{
		int thread_index = omp_get_thread_num();
		tmp[thread_index] = 0.0;
		for(j = 0; j < mat->col_dim; j++)
		{
			tmp[thread_index] += get_dmatrix_ij(mat, i, j) * get_dmatrix_ij(mat, i, j);
		}
#pragma omp critical
		ret += tmp[thread_index];
	}

	return ret;
}

/* Frobenius Norm for tridiagonal block matrix */
double _bncomp_normf_dmatrix_tridiag_blocks(DMatrix aim1[], DMatrix aii[], DMatrix aip1[], long int num_blocks)
{
	long int index;
	double ret, tmp[BNCOMP_MAX_NUM_THREADS];

	ret = 0.0;

	if(num_blocks < 1)
		return ret;

	ret += sumsqr_dmatrix(aii [0]);
	if(num_blocks >= 2)
	{
		ret += sumsqr_dmatrix(aip1[0]);
#pragma omp parallel for
		for(index = 1; index <= num_blocks - 2; index++)
		{
			int thread_index = omp_get_thread_num();
			tmp[thread_index] = 0.0;

			tmp[thread_index] += sumsqr_dmatrix(aim1[index - 1]);
			tmp[thread_index] += sumsqr_dmatrix(aii [index    ]);
			tmp[thread_index] += sumsqr_dmatrix(aip1[index    ]);
#pragma omp critical 
			ret += tmp[thread_index];
		}
		ret += sumsqr_dmatrix(aim1[num_blocks - 2]);
		ret += sumsqr_dmatrix(aii [num_blocks - 1]);
	}

	return sqrt(ret);
}

/* [ ret[0]              ]    [ b[0]              ]   [ aii [0] aip1[0]                                                   ]   [ x[0]              ] */
/* [ ret[1]              ]    [ b[1]              ]   [ aim1[0] aii [1] aip1[1]                                           ]   [ x[1]              ] */
/* [ ................... ] := [ ................. ] - [ ................................................................. ] * [ ................. ] */
/* [ ret[num_blocks - 2] ]    [ b[num_blocks - 2] ]   [    aim1[num_blocks - 3] aii [num_blocks - 2] aip1[num_blocsk - 2] ]   [ x[num_blocks - 2] ] */
/* [ ret[num_blocks - 1] ]    [ b[num_blocks - 1] ]   [                         aim1[num_blocks - 2] aii [num_blocks - 1] ]   [ x[num_blocks - 1] ] */
void _bncomp_residual_dmat_dvec_tridiag_blocks(DVector ret[], DVector b[], DMatrix aim1[], DMatrix aii[], DMatrix aip1[], DVector x[], long int num_blocks)
{
	int thread_index;
	long int index;
	DVector tmpvec[4][BNCOMP_MAX_NUM_THREADS];

	// initialize
#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();
	tmpvec[0][thread_index] = init_dvector(ret[0]->dim);
	tmpvec[1][thread_index] = init_dvector(ret[0]->dim);
	tmpvec[2][thread_index] = init_dvector(ret[0]->dim);
	tmpvec[3][thread_index] = init_dvector(ret[0]->dim);
}

//	mul_dmatrix_dvec(r, mat, x);
//	sub_dvector(r, b, r);

	// ret[0] := b[0] - (aii[0] * x[0] + aip1[0] * x[1])
	mul_dmatrix_dvec(tmpvec[0][0], aii [0], x[0]);
	mul_dmatrix_dvec(tmpvec[1][0], aip1[0], x[1]);
	add_dvector(tmpvec[2][0], tmpvec[0][0], tmpvec[1][0]);
	sub_dvector(ret[0], b[0], tmpvec[2][0]);

	// ret[index] ;= b[index] - ( aim1[index - 1] * x[index - 1] + aii[index] * x[index] + aip1[index] * x[index + 1] )
#pragma omp parallel for private(thread_index)
	for(index = 1; index <= num_blocks - 2; index++)
	{
		thread_index = omp_get_thread_num();
	
		//mul_dmatrix_dvec(tmpvec[0], aim1[index - 1], x[index - 1]);
		//mul_dmatrix_dvec(tmpvec[1], aii [index]    , x[index]    );
		//mul_dmatrix_dvec(tmpvec[2], aip1[index]    , x[index + 1]);
		mul_dmatrix_dvec(tmpvec[0][thread_index], aim1[index - 1], x[index - 1]);
		mul_dmatrix_dvec(tmpvec[1][thread_index], aii [index]    , x[index]    );
		mul_dmatrix_dvec(tmpvec[2][thread_index], aip1[index]    , x[index + 1]);
		//add_dvector(tmpvec[3], tmpvec[0], tmpvec[1]);
		//add_dvector(tmpvec[3], tmpvec[3], tmpvec[2]);
		//sub_dvector(ret[index], b[index], tmpvec[3]);
		add_dvector(tmpvec[3][thread_index], tmpvec[0][thread_index], tmpvec[1][thread_index]);
		add_dvector(tmpvec[3][thread_index], tmpvec[3][thread_index], tmpvec[2][thread_index]);
		sub_dvector(ret[index], b[index], tmpvec[3][thread_index]);
	}

	if(num_blocks >= 2)
	{
		// ret[num_blocks - 1] := b[num_blocks - 1] - ( aim1[num_blocks - 2] * x[num_blocks - 2] + aii[num_blocks - 1] * x[num_blocks - 1] )
		mul_dmatrix_dvec(tmpvec[0][0], aim1[num_blocks - 2], x[num_blocks - 2]);
		mul_dmatrix_dvec(tmpvec[1][0], aii [num_blocks - 1], x[num_blocks - 1]);
		add_dvector(tmpvec[2][0], tmpvec[0][0], tmpvec[1][0]);
		sub_dvector(ret[num_blocks - 1], b[num_blocks - 1], tmpvec[2][0]);
	}

	// free
#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();
	free_dvector(tmpvec[0][thread_index]);
	free_dvector(tmpvec[1][thread_index]);
	free_dvector(tmpvec[2][thread_index]);
	free_dvector(tmpvec[3][thread_index]);
}
}

/* [ ret[0]              ]    [ b[0]              ]   [ aii [0] aip1[0]                                                   ]   [ x[0]              ] */
/* [ ret[1]              ]    [ b[1]              ]   [ aim1[0] aii [1] aip1[1]                                           ]   [ x[1]              ] */
/* [ ................... ] := [ ................. ] - [ ................................................................. ] * [ ................. ] */
/* [ ret[num_blocks - 2] ]    [ b[num_blocks - 2] ]   [    aim1[num_blocks - 3] aii [num_blocks - 2] aip1[num_blocsk - 2] ]   [ x[num_blocks - 2] ] */
/* [ ret[num_blocks - 1] ]    [ b[num_blocks - 1] ]   [                         aim1[num_blocks - 2] aii [num_blocks - 1] ]   [ x[num_blocks - 1] ] */
void _bncomp_residual_dbmat_dvec_tridiag_blocks(DVector ret[], DVector b[], DBMatrix aim1[], DBMatrix aii[], DBMatrix aip1[], DVector x[], long int num_blocks)
{
	int thread_index;
	long int index;
	DVector tmpvec[4][BNCOMP_MAX_NUM_THREADS];

	// initialize
#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();
	tmpvec[0][thread_index] = init_dvector(ret[0]->dim);
	tmpvec[1][thread_index] = init_dvector(ret[0]->dim);
	tmpvec[2][thread_index] = init_dvector(ret[0]->dim);
	tmpvec[3][thread_index] = init_dvector(ret[0]->dim);
}

//	mul_dmatrix_dvec(r, mat, x);
//	sub_dvector(r, b, r);

	// ret[0] := b[0] - (aii[0] * x[0] + aip1[0] * x[1])
	mul_dbmatrix_dvec(tmpvec[0][0], aii [0], x[0]);
	mul_dbmatrix_dvec(tmpvec[1][0], aip1[0], x[1]);
	add_dvector(tmpvec[2][0], tmpvec[0][0], tmpvec[1][0]);
	sub_dvector(ret[0], b[0], tmpvec[2][0]);

	// ret[index] ;= b[index] - ( aim1[index - 1] * x[index - 1] + aii[index] * x[index] + aip1[index] * x[index + 1] )
#pragma omp parallel for private(thread_index)
	for(index = 1; index <= num_blocks - 2; index++)
	{
		thread_index = omp_get_thread_num();
	
		//mul_dmatrix_dvec(tmpvec[0], aim1[index - 1], x[index - 1]);
		//mul_dmatrix_dvec(tmpvec[1], aii [index]    , x[index]    );
		//mul_dmatrix_dvec(tmpvec[2], aip1[index]    , x[index + 1]);
		mul_dbmatrix_dvec(tmpvec[0][thread_index], aim1[index - 1], x[index - 1]);
		mul_dbmatrix_dvec(tmpvec[1][thread_index], aii [index]    , x[index]    );
		mul_dbmatrix_dvec(tmpvec[2][thread_index], aip1[index]    , x[index + 1]);
		//add_dvector(tmpvec[3], tmpvec[0], tmpvec[1]);
		//add_dvector(tmpvec[3], tmpvec[3], tmpvec[2]);
		//sub_dvector(ret[index], b[index], tmpvec[3]);
		add_dvector(tmpvec[3][thread_index], tmpvec[0][thread_index], tmpvec[1][thread_index]);
		add_dvector(tmpvec[3][thread_index], tmpvec[3][thread_index], tmpvec[2][thread_index]);
		sub_dvector(ret[index], b[index], tmpvec[3][thread_index]);
	}

	if(num_blocks >= 2)
	{
		// ret[num_blocks - 1] := b[num_blocks - 1] - ( aim1[num_blocks - 2] * x[num_blocks - 2] + aii[num_blocks - 1] * x[num_blocks - 1] )
		mul_dbmatrix_dvec(tmpvec[0][0], aim1[num_blocks - 2], x[num_blocks - 2]);
		mul_dbmatrix_dvec(tmpvec[1][0], aii [num_blocks - 1], x[num_blocks - 1]);
		add_dvector(tmpvec[2][0], tmpvec[0][0], tmpvec[1][0]);
		sub_dvector(ret[num_blocks - 1], b[num_blocks - 1], tmpvec[2][0]);
	}

	// free
#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();
	free_dvector(tmpvec[0][thread_index]);
	free_dvector(tmpvec[1][thread_index]);
	free_dvector(tmpvec[2][thread_index]);
	free_dvector(tmpvec[3][thread_index]);
}
}

/* substitute vector blocks */
void _bncomp_subst_fvector_dvec_blocks(FVector ret[], DVector v[], long int num_blocks)
{
	long int i;

#pragma omp parallel for
	for(i = 0; i < num_blocks; i++)
		subst_fvector_dvec(ret[i], v[i]);
}

/* substitute vector blocks */
void _bncomp_subst_dvector_fvec_blocks(DVector ret[], FVector v[], long int num_blocks)
{
	long int i;

#pragma omp parallel for
	for(i = 0; i < num_blocks; i++)
		subst_dvector_fvec(ret[i], v[i]);
}

/* substitution of tridiagonal block matrix */
void _bncomp_subst_fmatrix_dmat_tridiag_blocks(FMatrix aim1_f[], FMatrix aii_f[], FMatrix aip1_f[], DMatrix aim1[], DMatrix aii[], DMatrix aip1[], long int num_blocks)
{
	long int index;

	if(num_blocks < 1)
		return;

 	subst_fmatrix_dmat(aii_f [0], aii [0]);
	if(num_blocks >= 2)
	{
		subst_fmatrix_dmat(aip1_f[0], aip1[0]);
#pragma omp parallel for
		for(index = 1; index < num_blocks - 1; index++)
		{
			subst_fmatrix_dmat(aim1_f[index - 1], aim1[index - 1]);
			subst_fmatrix_dmat(aii_f [index]    , aii [index]);
			subst_fmatrix_dmat(aip1_f[index]    , aip1[index]);
		}
		subst_fmatrix_dmat(aim1_f[num_blocks - 2], aim1[num_blocks - 2]);
		subst_fmatrix_dmat(aii_f [num_blocks - 1], aii [num_blocks - 1]);
	}
}

/* ret[] := a[] * vb[] */
void _bncomp_mul_dtridiag_dvec_blocks(DVector ret[], DMatrix aim1[], DMatrix aii[], DMatrix aip1[], DVector vb[], DVector tmpv[], long int num_blocks)
{
	long int block_i;

	//mul_dmatrix_dvec(vec[5], a, vec[3]);

	mul_dmatrix_dvec( ret[0], aii [0], vb[0]);
	mul_dmatrix_dvec(tmpv[0], aip1[0], vb[1]);
	add_dvector(ret[0], ret[0], tmpv[0]);

#pragma omp parallel for
	for(block_i = 1; block_i < num_blocks - 1; block_i++)
	{
		int thread_index = omp_get_thread_num();

		mul_dmatrix_dvec( ret[block_i], aim1 [block_i - 1], vb[block_i - 1]);
		mul_dmatrix_dvec(tmpv[block_i], aii  [block_i    ], vb[block_i    ]);
		add_dvector(ret[block_i], ret[block_i], tmpv[block_i]);
		mul_dmatrix_dvec(tmpv[block_i], aip1 [block_i    ], vb[block_i + 1]);
		add_dvector(ret[block_i], ret[block_i], tmpv[block_i]);
	}

	if(num_blocks >= 2)
	{
		mul_dmatrix_dvec( ret[num_blocks - 1], aim1 [num_blocks - 2], vb[num_blocks - 2]);
		mul_dmatrix_dvec(tmpv[num_blocks - 1], aii  [num_blocks - 1], vb[num_blocks - 1]);
		add_dvector(ret[num_blocks - 1], ret[num_blocks - 1], tmpv[num_blocks - 1]);
	}
}

/* ret[] := a_band[] * vb[] */
void _bncomp_mul_dbtridiag_dvec_blocks(DVector ret[], DBMatrix aim1[], DBMatrix aii[], DBMatrix aip1[], DVector vb[], DVector tmpv[], long int num_blocks)
{
	long int block_i;

	//mul_dmatrix_dvec(vec[5], a, vec[3]);

	mul_dbmatrix_dvec( ret[0], aii [0], vb[0]);
	mul_dbmatrix_dvec(tmpv[0], aip1[0], vb[1]);
	add_dvector(ret[0], ret[0], tmpv[0]);

#pragma omp parallel for
	for(block_i = 1; block_i < num_blocks - 1; block_i++)
	{
		int thread_index = omp_get_thread_num();

		mul_dbmatrix_dvec( ret[block_i], aim1 [block_i - 1], vb[block_i - 1]);
		mul_dbmatrix_dvec(tmpv[block_i], aii  [block_i    ], vb[block_i    ]);
		add_dvector(ret[block_i], ret[block_i], tmpv[block_i]);
		mul_dbmatrix_dvec(tmpv[block_i], aip1 [block_i    ], vb[block_i + 1]);
		add_dvector(ret[block_i], ret[block_i], tmpv[block_i]);
	}

	if(num_blocks >= 2)
	{
		mul_dbmatrix_dvec( ret[num_blocks - 1], aim1 [num_blocks - 2], vb[num_blocks - 2]);
		mul_dbmatrix_dvec(tmpv[num_blocks - 1], aii  [num_blocks - 1], vb[num_blocks - 1]);
		add_dvector(ret[num_blocks - 1], ret[num_blocks - 1], tmpv[num_blocks - 1]);
	}
}

/* norm_tol2_dvector */
double _bncomp_norm_tol2_dvector(DVector y_err, DVector y_new, DVector y_old, double rtol, double atol)
{
	long int i;
	double d[BNCOMP_MAX_NUM_THREADS], tmp[BNCOMP_MAX_NUM_THREADS], y_new_i[BNCOMP_MAX_NUM_THREADS], y_old_i[BNCOMP_MAX_NUM_THREADS], ret;

	ret = 0.0;

#pragma omp parallel for
	for(i = 0; i < y_new->dim; i++)
	{
		int thread_index = omp_get_thread_num();

		y_new_i[thread_index] = fabs(get_dvector_i(y_new, i));
		y_old_i[thread_index] = fabs(get_dvector_i(y_old, i));
		if(y_new_i[thread_index] > y_old_i[thread_index])
			d[thread_index] = atol + rtol * y_new_i[thread_index];
		else
			d[thread_index] = atol + rtol * y_old_i[thread_index];

		tmp[thread_index] = get_dvector_i(y_err, i);
		if(d[thread_index] != 0.0)
			tmp[thread_index] /= d[thread_index];

		d[thread_index] = tmp[thread_index] * tmp[thread_index];

#pragma omp critical
		ret += d[thread_index];
	}
	ret /= (double)(y_new->dim);

	ret = sqrt(ret);

	return ret;
}

/* norm_tol2_dvector */
double _bncomp_norm2_diff_dvector(DVector y_new, DVector y_old, double rtol, double atol)
{
	long int i;
	double d[BNCOMP_MAX_NUM_THREADS], tmp[BNCOMP_MAX_NUM_THREADS], y_new_i[BNCOMP_MAX_NUM_THREADS], y_old_i[BNCOMP_MAX_NUM_THREADS], y_diff_i[BNCOMP_MAX_NUM_THREADS], ret;

	ret = 0.0;

#pragma omp parallel for
	for(i = 0; i < y_new->dim; i++)
	{
		int thread_index = omp_get_thread_num();

		y_diff_i[thread_index] = gdvi(y_new, i) - gdvi(y_old, i);
		y_new_i[thread_index] = fabs(get_dvector_i(y_new, i));
		y_old_i[thread_index] = fabs(get_dvector_i(y_old, i));
		if(y_new_i[thread_index] > y_old_i[thread_index])
			d[thread_index] = atol + rtol * y_new_i[thread_index];
		else
			d[thread_index] = atol + rtol * y_old_i[thread_index];
		
		tmp[thread_index] = y_diff_i[thread_index];
		if(d[thread_index] != 0.0)
			tmp[thread_index] /= d[thread_index];

		d[thread_index] = tmp[thread_index] * tmp[thread_index];

#pragma omp critical
		ret += d[thread_index];
	}
	ret /= (double)(y_new->dim);

	ret = sqrt(ret);

	return ret;
}

/* norm1 of vector blocks */
double _bncomp_norm1_dvector_blocks(DVector v[], long int num_blocks)
{
	long int i, j;
	double ret, tmp[BNCOMP_MAX_NUM_THREADS];

	ret = 0.0;

#pragma omp parallel for private(j)
	for(i = 0; i < num_blocks; i++)
	{
		int thread_index = omp_get_thread_num();

		tmp[thread_index] = 0.0;
		for(j = 0; j < v[i]->dim; j++)
			tmp[thread_index] += fabs(get_dvector_i(v[i], j));

#pragma omp critical
		ret += tmp[thread_index];
	}

	return ret;
}

// normalize of vector
// kind of vector norm: 0...Infinity, 1...1norm, 2...euclid norm
// return value : ||v||
//#define BNC_INFINITY_NORM	0
//#define BNC_ONE_NORM		1
//#define BNC_EUCLID_NORM		2

double _bncomp_norm_dvector(DVector vec, int kind_of_norm)
{
	double norm;

	switch(kind_of_norm)
	{
		case BNC_INFINITY_NORM:
			norm = normi_dvector(vec);
			break;
		case BNC_ONE_NORM:
			norm = norm1_dvector(vec);
			break;
		case BNC_EUCLID_NORM:
		default:
			norm = norm2_dvector(vec);
			break;
	}

	return norm;
}

/* ret := val * vec */
void _bncomp_cmul_dvector_blocks(DVector ret[], double val, DVector vec[], long int num_blocks)
{
	long int i;

#pragma omp parallel for
	for(i = 0; i < num_blocks; i++)
		cmul_dvector(ret[i], val, vec[i]);

	return;
}

/* vec := val * vec */
void _bncomp_cmul2_dvector_blocks(DVector vec[], double val, long int num_blocks)
{
	long int i;

#pragma omp parallel for
	for(i = 0; i < num_blocks; i++)
		cmul2_dvector(vec[i], val);

	return;
}

/* ||vec|| */
double _bncomp_norm_dvector_blocks(DVector vec[], long int num_blocks, int kind_of_norm)
{
	double norm;

	switch(kind_of_norm)
	{
		case BNC_INFINITY_NORM:
			norm = _bncomp_normi_dvector_blocks(vec, num_blocks);
			break;
		case BNC_ONE_NORM:
			norm = _bncomp_norm1_dvector_blocks(vec, num_blocks);
			break;
		case BNC_EUCLID_NORM:
		default:
			norm = _bncomp_norm2_dvector_blocks(vec, num_blocks);
			break;
	}

	return norm;
}

/* ret := vec / ||vec|| */
double _bncomp_normalize_dvector_blocks(DVector ret[], DVector vec[], long int num_blocks, int kind_of_norm)
{
	double norm;

	norm = _bncomp_norm_dvector_blocks(vec, num_blocks, kind_of_norm);

	// ret := vec / ||vec||
	_bncomp_cmul_dvector_blocks(ret, 1.0 / norm, vec, num_blocks);

	return norm;
}

/* vec := vec / ||vec|| */
double _bncomp_normalize2_dvector_blocks(DVector vec[], long int num_blocks, int kind_of_norm)
{
	double norm;

	norm = _bncomp_norm_dvector_blocks(vec, num_blocks, kind_of_norm);

	// ret := vec / ||vec||
	_bncomp_cmul2_dvector_blocks(vec, 1.0 / norm, num_blocks);

	return norm;
}

//#include "bncomp_bicgstab.c"

// MPF
#ifdef USE_GMP

/* A^T * diag(d0, ..., dn) */
void _bncomp_mul_mpfmatrixt_mpfdiagmat(MPFMatrix ret, MPFMatrix a, MPFVector d)
{
	long int i, j;
	int thread_index;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS];

#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();
	mpf_init2(tmp[thread_index], ret->prec);
}

	if(a->col_dim != d->dim)
	{
		fprintf(stderr, "Dimension is not equal! : a->row_dim(%ld) != d->dim(%ld)\n", a->col_dim, d->dim);
		return;
	}

	if((a->row_dim != ret->col_dim) || (a->col_dim != ret->row_dim))
	{
		fprintf(stderr, "The size of a and ret is not the same! : a^T(%ld, %ld) != ret(%ld, %ld)\n", a->col_dim, a->row_dim, ret->row_dim, ret->col_dim);
		return;
	}

#pragma omp parallel for private(j, thread_index)
	for(i = 0; i < ret->row_dim; i++)
	{
		thread_index = omp_get_thread_num();
		for(j = 0; j < ret->col_dim; j++)
		{
			mpf_mul(tmp[thread_index], get_mpfmatrix_ij(a, j, i), get_mpfvector_i(d, j));
			set_mpfmatrix_ij(ret, i, j, tmp[thread_index]);
		}
	}

#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();
	mpf_clear(tmp[thread_index]);
}
}

/* substitute vector blocks */
void _bncomp_subst_mpfvector_blocks(MPFVector ret[], MPFVector v[], long int num_blocks)
{
	long int i;

	#pragma omp parallel for
	for(i = 0; i < num_blocks; i++)
		subst_mpfvector(ret[i], v[i]);
}

/* inner product of vector blocks */
void _bncomp_ip_mpfvector_blocks(mpf_t ret, MPFVector va[], MPFVector vb[], long int num_blocks)
{
	long int i;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS];
	int thread_index;

#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();
	mpf_init2(tmp[thread_index], mpf_get_prec(ret));
}

	mpf_set_ui(ret, 0UL); // = 0.0;
#pragma omp parallel for private(thread_index)
	for(i = 0; i < num_blocks; i++)
	{
		thread_index = omp_get_thread_num();
		ip_mpfvector(tmp[thread_index], va[i], vb[i]);
		//ret += ip_mpfvector(va[i], vb[i]);
#pragma omp critical
		mpf_add(ret, ret, tmp[thread_index]);
	}

#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();
	mpf_clear(tmp[thread_index]);
}

	return;
}

/* norm2 of vector blocks */
void _bncomp_norm2_mpfvector_blocks(mpf_t ret, MPFVector v[], long int num_blocks)
{
	long int i, j;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS], tmp2[BNCOMP_MAX_NUM_THREADS];
	int thread_index;

#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();
	mpf_init2(tmp [thread_index], mpf_get_prec(ret));
	mpf_init2(tmp2[thread_index], mpf_get_prec(ret));
}

	mpf_set_ui(ret, 0UL); // = 0.0;

#pragma omp parallel for private(j, thread_index)
	for(i = 0; i < num_blocks; i++)
	{
		thread_index = omp_get_thread_num();
		mpf_set_ui(tmp2[thread_index], 0UL); 
		for(j = 0; j < v[i]->dim; j++)
		{
			// ret += get_mpfvector_i(v[i], j) * get_mpfvector_i(v[i], j);
			mpf_mul(tmp [thread_index], get_mpfvector_i(v[i], j), get_mpfvector_i(v[i], j));
			mpf_add(tmp2[thread_index], tmp2[thread_index], tmp[thread_index]);
		}

#pragma omp critical
		mpf_add(ret, ret, tmp2[thread_index]);

	}

	//ret = sqrt(ret);
	mpf_sqrt(ret, ret);

#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();
	mpf_clear(tmp [thread_index]);
	mpf_clear(tmp2[thread_index]);
}

	return;
}

/* ret := va + alpha * vb */
void _bncomp_add_cmul_mpfvector_blocks(MPFVector ret[], MPFVector va[], mpf_t alpha, MPFVector vb[], long int num_blocks)
{
	long int i;

#pragma omp parallel for
	for(i = 0; i < num_blocks; i++)
		add_cmul_mpfvector(ret[i], va[i], alpha, vb[i]);

	return;
}

/* ret := (A \otimes I) vb[] */
void _bncomp_kmul_mpfmatrixI_mpfvector_blocks(MPFVector ret[], MPFMatrix mat, MPFVector vb[], long int num_blocks)
{
	long int i, j;

	if(mat->col_dim > num_blocks)
	{
		fprintf(stderr, "kmul_mpfmatrixI_mpfvector_blocks: mat->col_dim(%ld) is larger than num_blocks(%ld)!\n", mat->col_dim, num_blocks);
		return;
	}

#pragma omp parallel for private(j)
	for(i = 0; i < mat->row_dim; i++)
	{
		set0_mpfvector(ret[i]);
		for(j = 0; j < mat->col_dim; j++)
			add_cmul_mpfvector(ret[i], ret[i], get_mpfmatrix_ij(mat, i, j), vb[j]);
	}
}

/* ret := -vb */
void _bncomp_neg_mpfvector_blocks(MPFVector vb[], long int num_blocks)
{
	long int i;

#pragma omp parallel for
	for(i = 0; i < num_blocks; i++)
		neg_mpfvector(vb[i]);
}

/* norm_inf of vector blocks */
void _bncomp_normi_mpfvector_blocks(mpf_t ret, MPFVector v[], long int num_blocks)
{
	long int i, j;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS], tmp2[BNCOMP_MAX_NUM_THREADS];
	int thread_index;

#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();
	mpf_init2(tmp [thread_index], mpf_get_prec(ret));
	mpf_init2(tmp2[thread_index], mpf_get_prec(ret));
}

	mpf_set_ui(ret, 0UL); // ret = 0.0;

#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < num_blocks; i++)
	{
		thread_index = omp_get_thread_num();

		mpf_set_ui(tmp2[thread_index], 0UL);
		for(j = 0; j < v[i]->dim; j++)
		{
			//tmp = fabs(get_mpfvector_i(v[i], j));
			mpf_abs(tmp[thread_index], get_mpfvector_i(v[i], j));
			if(mpf_cmp(tmp2[thread_index], tmp[thread_index]) < 0);
				mpf_set(tmp2[thread_index], tmp[thread_index]);
		}
#pragma omp critical
		if(mpf_cmp(ret, tmp2[thread_index]) < 0)
			mpf_set(ret, tmp2[thread_index]);
	}

#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();
	mpf_clear(tmp [thread_index]);
	mpf_clear(tmp2[thread_index]);
}

	return;
}

/* ret := va - vb */
void _bncomp_sub_mpfvector_blocks(MPFVector ret[], MPFVector va[], MPFVector vb[], long int num_blocks)
{
	long int i;

#pragma omp parallel for
	for(i = 0; i < num_blocks; i++)
		sub_mpfvector(ret[i], va[i], vb[i]);
}

/* ret := va + vb */
void _bncomp_add_mpfvector_blocks(MPFVector ret[], MPFVector va[], MPFVector vb[], long int num_blocks)
{
	long int i;

#pragma omp parallel for
	for(i = 0; i < num_blocks; i++)
		add_mpfvector(ret[i], va[i], vb[i]);
}

/* ret := sum^n_{i,j=1} aij^2 */
void sumsqr_mpfmatrix(mpf_t ret, MPFMatrix mat)
{
	long int i, j;
	mpf_t tmp;

	mpf_init2(tmp, mpf_get_prec(ret));
	mpf_set_ui(ret, 0UL);

	for(i = 0; i < mat->row_dim; i++)
	{
		for(j = 0; j < mat->col_dim; j++)
		{
			mpf_mul(tmp, get_mpfmatrix_ij(mat, i, j), get_mpfmatrix_ij(mat, i, j));
			mpf_add(ret, ret, tmp);
		}
	}

	mpf_clear(tmp);
}

/* ret := sum^n_{i,j=1} aij^2 */
void _bncomp_sumsqr_mpfmatrix(mpf_t ret, MPFMatrix mat)
{
	long int i, j;
	mpf_t tmp[BNCOMP_MAX_NUM_THREADS], tmp2[BNCOMP_MAX_NUM_THREADS];
	int thread_index;

#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();
	mpf_init2(tmp [thread_index], mpf_get_prec(ret));
	mpf_init2(tmp2[thread_index], mpf_get_prec(ret));
}

	mpf_set_ui(ret, 0UL);

#pragma omp parallel for private(thread_index, j)
	for(i = 0; i < mat->row_dim; i++)
	{
		thread_index = omp_get_thread_num();
		mpf_set_ui(tmp2[thread_index], 0UL);
		for(j = 0; j < mat->col_dim; j++)
		{
			mpf_mul(tmp[thread_index], get_mpfmatrix_ij(mat, i, j), get_mpfmatrix_ij(mat, i, j));
			mpf_add(tmp2[thread_index], tmp2[thread_index], tmp[thread_index]);
		}
#pragma omp critical
		mpf_add(ret, ret, tmp2[thread_index]);
	}

#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();
	mpf_clear(tmp [thread_index]);
	mpf_clear(tmp2[thread_index]);
}
}

/* Frobenius Norm for tridiagonal block matrix */
void _bncomp_normf_mpfmatrix_tridiag_blocks(mpf_t ret, MPFMatrix aim1[], MPFMatrix aii[], MPFMatrix aip1[], long int num_blocks)
{
	long int index;
	mpf_t tmp[128], tmp2[128];
	int thread_index;

	if(num_blocks < 1)
		return;

#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp [thread_index], mpf_get_prec(ret));
		mpf_init2(tmp2[thread_index], mpf_get_prec(ret));
		mpf_set_ui(tmp [thread_index], 0UL);
		mpf_set_ui(tmp2[thread_index], 0UL);
	}

	mpf_set_ui(ret, 0UL);

	sumsqr_mpfmatrix(tmp[0], aii [0]); mpf_add(ret, ret, tmp[0]);

	if(num_blocks >= 2)
	{
		sumsqr_mpfmatrix(tmp[0], aip1[0]); mpf_add(ret, ret, tmp[0]);
#pragma omp parallel for private(thread_index)
		for(index = 1; index <= num_blocks - 2; index++)
		{
			thread_index = omp_get_thread_num();
			mpf_set_ui(tmp2[thread_index], 0UL);

			sumsqr_mpfmatrix(tmp[thread_index], aim1[index - 1]);
			mpf_add(tmp2[thread_index], tmp2[thread_index], tmp[thread_index]);

			sumsqr_mpfmatrix(tmp[thread_index], aii [index    ]);
			mpf_add(tmp2[thread_index], tmp2[thread_index], tmp[thread_index]);

			sumsqr_mpfmatrix(tmp[thread_index], aip1[index    ]);
			mpf_add(tmp2[thread_index], tmp2[thread_index], tmp[thread_index]);
#pragma omp critical
			mpf_add(ret, ret, tmp2[thread_index]);
		}
		sumsqr_mpfmatrix(tmp[0], aim1[num_blocks - 2]); mpf_add(ret, ret, tmp[0]);
		sumsqr_mpfmatrix(tmp[0], aii [num_blocks - 1]); mpf_add(ret, ret, tmp[0]);
	}

	mpf_sqrt(ret, ret);

#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp[thread_index]);
		mpf_clear(tmp2[thread_index]);
	}
}

/* Frobenius Norm for tridiagonal block band matrix */
void _bncomp_normf_mpfbmatrix_tridiag_blocks(mpf_t ret, MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], long int num_blocks)
{
	long int index;
	mpf_t tmp[128], tmp2[128];
	int thread_index;

	if(num_blocks < 1)
		return;

#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp [thread_index], mpf_get_prec(ret));
		mpf_init2(tmp2[thread_index], mpf_get_prec(ret));
		mpf_set_ui(tmp [thread_index], 0UL);
		mpf_set_ui(tmp2[thread_index], 0UL);
	}

	mpf_set_ui(ret, 0UL);

	sumsqr_mpfbmatrix(tmp[0], aii [0]); mpf_add(ret, ret, tmp[0]);

	if(num_blocks >= 2)
	{
		sumsqr_mpfbmatrix(tmp[0], aip1[0]); mpf_add(ret, ret, tmp[0]);
#pragma omp parallel for schedule(dynamic) private(thread_index,index)
		for(index = 1; index <= num_blocks - 2; index++)
		{
//			printf("%d on %d th thread on _bncomp_normf_mpfbmatrix\n", index, omp_get_thread_num());
			thread_index = omp_get_thread_num();
			mpf_set_ui(tmp2[thread_index], 0UL);

			sumsqr_mpfbmatrix(tmp[thread_index], aim1[index - 1]);
			mpf_add(tmp2[thread_index], tmp2[thread_index], tmp[thread_index]);

			sumsqr_mpfbmatrix(tmp[thread_index], aii [index    ]);
			mpf_add(tmp2[thread_index], tmp2[thread_index], tmp[thread_index]);

			sumsqr_mpfbmatrix(tmp[thread_index], aip1[index    ]);
			mpf_add(tmp2[thread_index], tmp2[thread_index], tmp[thread_index]);
#pragma omp critical
			mpf_add(ret, ret, tmp2[thread_index]);
		}
		sumsqr_mpfbmatrix(tmp[0], aim1[num_blocks - 2]); mpf_add(ret, ret, tmp[0]);
		sumsqr_mpfbmatrix(tmp[0], aii [num_blocks - 1]); mpf_add(ret, ret, tmp[0]);
	}

	mpf_sqrt(ret, ret);

#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp[thread_index]);
		mpf_clear(tmp2[thread_index]);
	}
}

/* [ ret[0]              ]    [ b[0]              ]   [ aii [0] aip1[0]                                                   ]   [ x[0]              ] */
/* [ ret[1]              ]    [ b[1]              ]   [ aim1[0] aii [1] aip1[1]                                           ]   [ x[1]              ] */
/* [ ................... ] := [ ................. ] - [ ................................................................. ] * [ ................. ] */
/* [ ret[num_blocks - 2] ]    [ b[num_blocks - 2] ]   [    aim1[num_blocks - 3] aii [num_blocks - 2] aip1[num_blocsk - 2] ]   [ x[num_blocks - 2] ] */
/* [ ret[num_blocks - 1] ]    [ b[num_blocks - 1] ]   [                         aim1[num_blocks - 2] aii [num_blocks - 1] ]   [ x[num_blocks - 1] ] */
void _bncomp_residual_mpfmat_mpfvec_tridiag_blocks(MPFVector ret[], MPFVector b[], MPFMatrix aim1[], MPFMatrix aii[], MPFMatrix aip1[], MPFVector x[], long int num_blocks)
{
	long int index;
	int num_threads, thread_index, i;
	MPFVector tmpvec[4][BNCOMP_MAX_NUM_THREADS];

	if(num_blocks < 1)
		return;

	num_threads = omp_get_num_threads();

	// initialize
#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
//		printf("thread_index %d ->\n", thread_index);

		tmpvec[0][thread_index] = init2_mpfvector(ret[0]->dim, ret[0]->prec);
		tmpvec[1][thread_index] = init2_mpfvector(ret[0]->dim, ret[0]->prec);
		tmpvec[2][thread_index] = init2_mpfvector(ret[0]->dim, ret[0]->prec);
		tmpvec[3][thread_index] = init2_mpfvector(ret[0]->dim, ret[0]->prec);
	}

//	mul_mpfmatrix_mpfvec(r, mat, x);
//	sub_mpfvector(r, b, r);

	// ret[0] := b[0] - (aii[0] * x[0] + aip1[0] * x[1])
	mul_mpfmatrix_mpfvec(tmpvec[0][0], aii [0], x[0]);
	sub_mpfvector(ret[0], b[0], tmpvec[0][0]);

	if(num_blocks >= 2)
	{
		mul_mpfmatrix_mpfvec(tmpvec[1][0], aip1[0], x[1]);
		add_mpfvector(tmpvec[2][0], tmpvec[0][0], tmpvec[1][0]);
		sub_mpfvector(ret[0], b[0], tmpvec[2][0]);

		// ret[index] ;= b[index] - ( aim1[index - 1] * x[index - 1] + aii[index] * x[index] + aip1[index] * x[index + 1] )
#pragma omp parallel for private(thread_index)
		for(index = 1; index <= num_blocks - 2; index++)
		{
			thread_index = omp_get_thread_num();
			//printf("thread %d -> index %d\n", thread_index, index);

			//mul_mpfmatrix_mpfvec(tmpvec[0], aim1[index - 1], x[index - 1]);
			//mul_mpfmatrix_mpfvec(tmpvec[1], aii [index]    , x[index]    );
			//mul_mpfmatrix_mpfvec(tmpvec[2], aip1[index]    , x[index + 1]);
			mul_mpfmatrix_mpfvec(tmpvec[0][thread_index], aim1[index - 1], x[index - 1]);
			mul_mpfmatrix_mpfvec(tmpvec[1][thread_index], aii [index]    , x[index]    );
			mul_mpfmatrix_mpfvec(tmpvec[2][thread_index], aip1[index]    , x[index + 1]);
			//add_mpfvector(tmpvec[3], tmpvec[0], tmpvec[1]);
			//add_mpfvector(tmpvec[3], tmpvec[3], tmpvec[2]);
			//sub_mpfvector(ret[index], b[index], tmpvec[3]);
			add_mpfvector(tmpvec[3][thread_index], tmpvec[0][thread_index], tmpvec[1][thread_index]);
			add_mpfvector(tmpvec[3][thread_index], tmpvec[3][thread_index], tmpvec[2][thread_index]);
			sub_mpfvector(ret[index], b[index], tmpvec[3][thread_index]);
		}

		// ret[num_blocks - 1] := b[num_blocks - 1] - ( aim1[num_blocks - 2] * x[num_blocks - 2] + aii[num_blocks - 1] * x[num_blocks - 1] )
		mul_mpfmatrix_mpfvec(tmpvec[0][0], aim1[num_blocks - 2], x[num_blocks - 2]);
		mul_mpfmatrix_mpfvec(tmpvec[1][0], aii [num_blocks - 1], x[num_blocks - 1]);
		add_mpfvector(tmpvec[2][0], tmpvec[0][0], tmpvec[1][0]);
		sub_mpfvector(ret[num_blocks - 1], b[num_blocks - 1], tmpvec[2][0]);
	}

	// free
#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();

		free_mpfvector(tmpvec[0][thread_index]);
		free_mpfvector(tmpvec[1][thread_index]);
		free_mpfvector(tmpvec[2][thread_index]);
		free_mpfvector(tmpvec[3][thread_index]);
	}
}

/* [ ret[0]              ]    [ b[0]              ]   [ aii [0] aip1[0]                                                   ]   [ x[0]              ] */
/* [ ret[1]              ]    [ b[1]              ]   [ aim1[0] aii [1] aip1[1]                                           ]   [ x[1]              ] */
/* [ ................... ] := [ ................. ] - [ ................................................................. ] * [ ................. ] */
/* [ ret[num_blocks - 2] ]    [ b[num_blocks - 2] ]   [    aim1[num_blocks - 3] aii [num_blocks - 2] aip1[num_blocsk - 2] ]   [ x[num_blocks - 2] ] */
/* [ ret[num_blocks - 1] ]    [ b[num_blocks - 1] ]   [                         aim1[num_blocks - 2] aii [num_blocks - 1] ]   [ x[num_blocks - 1] ] */
void _bncomp_residual_mpfbmat_mpfvec_tridiag_blocks(MPFVector ret[], MPFVector b[], MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], MPFVector x[], long int num_blocks)
{
	long int index;
	int num_threads, thread_index, i;
	MPFVector tmpvec[4][BNCOMP_MAX_NUM_THREADS];

	if(num_blocks < 1)
		return;

	num_threads = omp_get_num_threads();

	// initialize
#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
//		printf("thread_index %d -> %d\n", thread_index, i);

		tmpvec[0][thread_index] = init2_mpfvector(ret[0]->dim, ret[0]->prec);
		tmpvec[1][thread_index] = init2_mpfvector(ret[0]->dim, ret[0]->prec);
		tmpvec[2][thread_index] = init2_mpfvector(ret[0]->dim, ret[0]->prec);
		tmpvec[3][thread_index] = init2_mpfvector(ret[0]->dim, ret[0]->prec);
	}

//	mul_mpfmatrix_mpfvec(r, mat, x);
//	sub_mpfvector(r, b, r);

	// ret[0] := b[0] - (aii[0] * x[0] + aip1[0] * x[1])
	mul_mpfbmatrix_mpfvec(tmpvec[0][0], aii [0], x[0]);
	sub_mpfvector(ret[0], b[0], tmpvec[0][0]);

	if(num_blocks >= 2)
	{
		mul_mpfbmatrix_mpfvec(tmpvec[1][0], aip1[0], x[1]);
		add_mpfvector(tmpvec[2][0], tmpvec[0][0], tmpvec[1][0]);
		sub_mpfvector(ret[0], b[0], tmpvec[2][0]);

		// ret[index] ;= b[index] - ( aim1[index - 1] * x[index - 1] + aii[index] * x[index] + aip1[index] * x[index + 1] )
#pragma omp parallel for private(thread_index)
		for(index = 1; index <= num_blocks - 2; index++)
		{
			thread_index = omp_get_thread_num();
			//printf("thread %d -> index %d\n", thread_index, index);

			//mul_mpfmatrix_mpfvec(tmpvec[0], aim1[index - 1], x[index - 1]);
			//mul_mpfmatrix_mpfvec(tmpvec[1], aii [index]    , x[index]    );
			//mul_mpfmatrix_mpfvec(tmpvec[2], aip1[index]    , x[index + 1]);
			mul_mpfbmatrix_mpfvec(tmpvec[0][thread_index], aim1[index - 1], x[index - 1]);
			mul_mpfbmatrix_mpfvec(tmpvec[1][thread_index], aii [index]    , x[index]    );
			mul_mpfbmatrix_mpfvec(tmpvec[2][thread_index], aip1[index]    , x[index + 1]);
			//add_mpfvector(tmpvec[3], tmpvec[0], tmpvec[1]);
			//add_mpfvector(tmpvec[3], tmpvec[3], tmpvec[2]);
			//sub_mpfvector(ret[index], b[index], tmpvec[3]);
			add_mpfvector(tmpvec[3][thread_index], tmpvec[0][thread_index], tmpvec[1][thread_index]);
			add_mpfvector(tmpvec[3][thread_index], tmpvec[3][thread_index], tmpvec[2][thread_index]);
			sub_mpfvector(ret[index], b[index], tmpvec[3][thread_index]);
		}

		// ret[num_blocks - 1] := b[num_blocks - 1] - ( aim1[num_blocks - 2] * x[num_blocks - 2] + aii[num_blocks - 1] * x[num_blocks - 1] )
		mul_mpfbmatrix_mpfvec(tmpvec[0][0], aim1[num_blocks - 2], x[num_blocks - 2]);
		mul_mpfbmatrix_mpfvec(tmpvec[1][0], aii [num_blocks - 1], x[num_blocks - 1]);
		add_mpfvector(tmpvec[2][0], tmpvec[0][0], tmpvec[1][0]);
		sub_mpfvector(ret[num_blocks - 1], b[num_blocks - 1], tmpvec[2][0]);
	}

	// free
#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();

		free_mpfvector(tmpvec[0][thread_index]);
		free_mpfvector(tmpvec[1][thread_index]);
		free_mpfvector(tmpvec[2][thread_index]);
		free_mpfvector(tmpvec[3][thread_index]);
	}
}

/* mul vec */
/* ret[] := a[] * x[] */
void _bncomp_mul_mpfmat_mpfvec_tridiag_blocks(MPFVector ret[], MPFMatrix aim1[], MPFMatrix aii[], MPFMatrix aip1[], MPFVector x[], long int num_blocks)
{
	long int index;
	MPFVector tmpvec[3][BNCOMP_MAX_NUM_THREADS];
	int thread_index;

	if(num_blocks < 1)
		return;

	// initialize
#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();

	tmpvec[0][thread_index] = init2_mpfvector(ret[0]->dim, ret[0]->prec);
	tmpvec[1][thread_index] = init2_mpfvector(ret[0]->dim, ret[0]->prec);
	tmpvec[2][thread_index] = init2_mpfvector(ret[0]->dim, ret[0]->prec);
}

//	mul_mpfmatrix_mpfvec(r, mat, x);
//	sub_mpfvector(r, b, r);

	// ret[0] := (aii[0] * x[0] + aip1[0] * x[1])
	mul_mpfmatrix_mpfvec(tmpvec[0][0], aii [0], x[0]);
	subst_mpfvector(ret[0], tmpvec[0][0]);

	if(num_blocks >= 2)
	{
		mul_mpfmatrix_mpfvec(tmpvec[1][0], aip1[0], x[1]);
		add_mpfvector(ret[0], tmpvec[0][0], tmpvec[1][0]);

		// ret[index] ;= ( aim1[index - 1] * x[index - 1] + aii[index] * x[index] + aip1[index] * x[index + 1] )

		#pragma omp parallel for private(thread_index)
		for(index = 1; index <= num_blocks - 2; index++)
		{
			thread_index = omp_get_thread_num();

			mul_mpfmatrix_mpfvec(tmpvec[0][thread_index], aim1[index - 1], x[index - 1]);
			mul_mpfmatrix_mpfvec(tmpvec[1][thread_index], aii [index]    , x[index]    );
			mul_mpfmatrix_mpfvec(tmpvec[2][thread_index], aip1[index]    , x[index + 1]);
			add_mpfvector(ret[index], tmpvec[0][thread_index], tmpvec[1][thread_index]);
			add_mpfvector(ret[index], ret[index], tmpvec[2][thread_index]);
		}
//}
		// ret[num_blocks - 1] := b[num_blocks - 1] - ( aim1[num_blocks - 2] * x[num_blocks - 2] + aii[num_blocks - 1] * x[num_blocks - 1] )
		mul_mpfmatrix_mpfvec(tmpvec[0][0], aim1[num_blocks - 2], x[num_blocks - 2]);
		mul_mpfmatrix_mpfvec(tmpvec[1][0], aii [num_blocks - 1], x[num_blocks - 1]);
		add_mpfvector(ret[num_blocks - 1], tmpvec[0][0], tmpvec[1][0]);
	}

	// free
#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();

	free_mpfvector(tmpvec[0][thread_index]);
	free_mpfvector(tmpvec[1][thread_index]);
	free_mpfvector(tmpvec[2][thread_index]);
}
}

/* mul vec : band */
/* ret[] := a[] * x[] */
void _bncomp_mul_mpfbmat_mpfvec_tridiag_blocks(MPFVector ret[], MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], MPFVector x[], long int num_blocks)
{
	long int index;
	MPFVector tmpvec[3][BNCOMP_MAX_NUM_THREADS];
	int thread_index;

	if(num_blocks < 1)
		return;

	// initialize
#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();

	tmpvec[0][thread_index] = init2_mpfvector(ret[0]->dim, ret[0]->prec);
	tmpvec[1][thread_index] = init2_mpfvector(ret[0]->dim, ret[0]->prec);
	tmpvec[2][thread_index] = init2_mpfvector(ret[0]->dim, ret[0]->prec);
}

//	mul_mpfmatrix_mpfvec(r, mat, x);
//	sub_mpfvector(r, b, r);

	// ret[0] := (aii[0] * x[0] + aip1[0] * x[1])
	mul_mpfbmatrix_mpfvec(tmpvec[0][0], aii [0], x[0]);
	subst_mpfvector(ret[0], tmpvec[0][0]);

	if(num_blocks >= 2)
	{
		mul_mpfbmatrix_mpfvec(tmpvec[1][0], aip1[0], x[1]);
		add_mpfvector(ret[0], tmpvec[0][0], tmpvec[1][0]);

		// ret[index] ;= ( aim1[index - 1] * x[index - 1] + aii[index] * x[index] + aip1[index] * x[index + 1] )

		#pragma omp parallel for private(thread_index)
		for(index = 1; index <= num_blocks - 2; index++)
		{
			thread_index = omp_get_thread_num();

			mul_mpfbmatrix_mpfvec(tmpvec[0][thread_index], aim1[index - 1], x[index - 1]);
			mul_mpfbmatrix_mpfvec(tmpvec[1][thread_index], aii [index]    , x[index]    );
			mul_mpfbmatrix_mpfvec(tmpvec[2][thread_index], aip1[index]    , x[index + 1]);
			add_mpfvector(ret[index], tmpvec[0][thread_index], tmpvec[1][thread_index]);
			add_mpfvector(ret[index], ret[index], tmpvec[2][thread_index]);
		}
//}
		// ret[num_blocks - 1] := b[num_blocks - 1] - ( aim1[num_blocks - 2] * x[num_blocks - 2] + aii[num_blocks - 1] * x[num_blocks - 1] )
		mul_mpfbmatrix_mpfvec(tmpvec[0][0], aim1[num_blocks - 2], x[num_blocks - 2]);
		mul_mpfbmatrix_mpfvec(tmpvec[1][0], aii [num_blocks - 1], x[num_blocks - 1]);
		add_mpfvector(ret[num_blocks - 1], tmpvec[0][0], tmpvec[1][0]);
	}

	// free
#pragma omp parallel private(thread_index)
{
	thread_index = omp_get_thread_num();

	free_mpfvector(tmpvec[0][thread_index]);
	free_mpfvector(tmpvec[1][thread_index]);
	free_mpfvector(tmpvec[2][thread_index]);
}
}

/* ret[] := a_band[] * vb[] with working area */
void _bncomp_mul_mpfbtridiag_mpfvec_blocks(MPFVector ret[], MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], MPFVector vb[], MPFVector tmpv[], long int num_blocks)
{
	long int block_i;

	//mul_mpfmatrix_dvec(vec[5], a, vec[3]);

	mul_mpfbmatrix_mpfvec( ret[0], aii [0], vb[0]);
	mul_mpfbmatrix_mpfvec(tmpv[0], aip1[0], vb[1]);
	add_mpfvector(ret[0], ret[0], tmpv[0]);

#pragma omp parallel for
	for(block_i = 1; block_i < num_blocks - 1; block_i++)
	{
		int thread_index = omp_get_thread_num();

		mul_mpfbmatrix_mpfvec( ret[block_i], aim1 [block_i - 1], vb[block_i - 1]);
		mul_mpfbmatrix_mpfvec(tmpv[block_i], aii  [block_i    ], vb[block_i    ]);
		add_mpfvector(ret[block_i], ret[block_i], tmpv[block_i]);
		mul_mpfbmatrix_mpfvec(tmpv[block_i], aip1 [block_i    ], vb[block_i + 1]);
		add_mpfvector(ret[block_i], ret[block_i], tmpv[block_i]);
	}

	if(num_blocks >= 2)
	{
		mul_mpfbmatrix_mpfvec( ret[num_blocks - 1], aim1 [num_blocks - 2], vb[num_blocks - 2]);
		mul_mpfbmatrix_mpfvec(tmpv[num_blocks - 1], aii  [num_blocks - 1], vb[num_blocks - 1]);
		add_mpfvector(ret[num_blocks - 1], ret[num_blocks - 1], tmpv[num_blocks - 1]);
	}
}

/* ret[] := a[] * vb[] with working area */
void _bncomp_mul_mpftridiag_mpfvec_blocks(MPFVector ret[], MPFMatrix aim1[], MPFMatrix aii[], MPFMatrix aip1[], MPFVector vb[], MPFVector tmpv[], long int num_blocks)
{
	long int block_i;

	//mul_mpfmatrix_dvec(vec[5], a, vec[3]);

	mul_mpfmatrix_mpfvec( ret[0], aii [0], vb[0]);
	mul_mpfmatrix_mpfvec(tmpv[0], aip1[0], vb[1]);
	add_mpfvector(ret[0], ret[0], tmpv[0]);

#pragma omp parallel for
	for(block_i = 1; block_i < num_blocks - 1; block_i++)
	{
		int thread_index = omp_get_thread_num();

		mul_mpfmatrix_mpfvec( ret[block_i], aim1 [block_i - 1], vb[block_i - 1]);
		mul_mpfmatrix_mpfvec(tmpv[block_i], aii  [block_i    ], vb[block_i    ]);
		add_mpfvector(ret[block_i], ret[block_i], tmpv[block_i]);
		mul_mpfmatrix_mpfvec(tmpv[block_i], aip1 [block_i    ], vb[block_i + 1]);
		add_mpfvector(ret[block_i], ret[block_i], tmpv[block_i]);
	}

	if(num_blocks >= 2)
	{
		mul_mpfmatrix_mpfvec( ret[num_blocks - 1], aim1 [num_blocks - 2], vb[num_blocks - 2]);
		mul_mpfmatrix_mpfvec(tmpv[num_blocks - 1], aii  [num_blocks - 1], vb[num_blocks - 1]);
		add_mpfvector(ret[num_blocks - 1], ret[num_blocks - 1], tmpv[num_blocks - 1]);
	}
}

/* substitution of tridiagonal block matrix */
void _bncomp_subst_mpfmatrix_tridiag_blocks(MPFMatrix aim1_f[], MPFMatrix aii_f[], MPFMatrix aip1_f[], MPFMatrix aim1[], MPFMatrix aii[], MPFMatrix aip1[], long int num_blocks)
{
	long int index;

	if(num_blocks < 1)
		return;

 	subst_mpfmatrix(aii_f [0], aii [0]);
	if(num_blocks >= 2)
	{
		subst_mpfmatrix(aip1_f[0], aip1[0]);
#pragma omp parallel for
		for(index = 1; index < num_blocks - 1; index++)
		{
			subst_mpfmatrix(aim1_f[index - 1], aim1[index - 1]);
			subst_mpfmatrix(aii_f [index]    , aii [index]);
			subst_mpfmatrix(aip1_f[index]    , aip1[index]);
		}
		subst_mpfmatrix(aim1_f[num_blocks - 2], aim1[num_blocks - 2]);
		subst_mpfmatrix(aii_f [num_blocks - 1], aii [num_blocks - 1]);
	}
}

/* substitution of tridiagonal block matrix */
void _bncomp_subst_mpfbmatrix_tridiag_blocks(MPFBMatrix aim1_f[], MPFBMatrix aii_f[], MPFBMatrix aip1_f[], MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], long int num_blocks)
{
	long int index;

	if(num_blocks < 1)
		return;

 	subst_mpfbmatrix(aii_f [0], aii [0]);
	if(num_blocks >= 2)
	{
		subst_mpfbmatrix(aip1_f[0], aip1[0]);
#pragma omp parallel for
		for(index = 1; index < num_blocks - 1; index++)
		{
			subst_mpfbmatrix(aim1_f[index - 1], aim1[index - 1]);
			subst_mpfbmatrix(aii_f [index]    , aii [index]);
			subst_mpfbmatrix(aip1_f[index]    , aip1[index]);
		}
		subst_mpfbmatrix(aim1_f[num_blocks - 2], aim1[num_blocks - 2]);
		subst_mpfbmatrix(aii_f [num_blocks - 1], aii [num_blocks - 1]);
	}
}

/* substitute vector blocks */
void _bncomp_subst_dvector_mpfvec_blocks(DVector ret[], MPFVector v[], long int num_blocks)
{
	long int i;

#pragma omp parallel for
	for(i = 0; i < num_blocks; i++)
		subst_dvector_mpfvec(ret[i], v[i]);
}

/* substitute vector blocks */
void _bncomp_subst_mpfvector_dvec_blocks(MPFVector ret[], DVector v[], long int num_blocks)
{
	long int i;

#pragma omp parallel for
	for(i = 0; i < num_blocks; i++)
		subst_mpfvector_dvec(ret[i], v[i]);
}

/* substitution of tridiagonal block matrix */
void _bncomp_subst_dmatrix_mpfmat_tridiag_blocks(DMatrix aim1_f[], DMatrix aii_f[], DMatrix aip1_f[], MPFMatrix aim1[], MPFMatrix aii[], MPFMatrix aip1[], long int num_blocks)
{
	long int index;

	if(num_blocks < 1)
		return;

 	subst_dmatrix_mpfmat(aii_f [0], aii [0]);
	if(num_blocks >= 2)
	{
		subst_dmatrix_mpfmat(aip1_f[0], aip1[0]);

#pragma omp parallel for
		for(index = 1; index < num_blocks - 1; index++)
		{
			subst_dmatrix_mpfmat(aim1_f[index - 1], aim1[index - 1]);
			subst_dmatrix_mpfmat(aii_f [index]    , aii [index]);
			subst_dmatrix_mpfmat(aip1_f[index]    , aip1[index]);
		}
		subst_dmatrix_mpfmat(aim1_f[num_blocks - 2], aim1[num_blocks - 2]);
		subst_dmatrix_mpfmat(aii_f [num_blocks - 1], aii [num_blocks - 1]);
	}
}

/* substitution of tridiagonal block band matrix */
void _bncomp_subst_dbmatrix_mpfbmat_tridiag_blocks(DBMatrix aim1_f[], DBMatrix aii_f[], DBMatrix aip1_f[], MPFBMatrix aim1[], MPFBMatrix aii[], MPFBMatrix aip1[], long int num_blocks)
{
	long int index;

	if(num_blocks < 1)
		return;

 	subst_dbmatrix_mpfbmat(aii_f [0], aii [0]);
	if(num_blocks >= 2)
	{
		subst_dbmatrix_mpfbmat(aip1_f[0], aip1[0]);

#pragma omp parallel for
		for(index = 1; index < num_blocks - 1; index++)
		{
			//printf("%d on %d th thread on _bncomp_subst_dbmatrix_mpfmat\n", index, omp_get_thread_num());
			subst_dbmatrix_mpfbmat(aim1_f[index - 1], aim1[index - 1]);
			subst_dbmatrix_mpfbmat(aii_f [index]    , aii [index]);
			subst_dbmatrix_mpfbmat(aip1_f[index]    , aip1[index]);
		}
		subst_dbmatrix_mpfbmat(aim1_f[num_blocks - 2], aim1[num_blocks - 2]);
		subst_dbmatrix_mpfbmat(aii_f [num_blocks - 1], aii [num_blocks - 1]);
	}
}

/* norm1 of vector blocks */
void _bncomp_norm1_mpfvector_blocks(mpf_t ret, MPFVector v[], long int num_blocks)
{
	long int i, j;
	int thread_index;
	mpf_t tmp[128];

#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_init2(tmp[thread_index], mpf_get_prec(ret));
	}

	mpf_set_ui(ret, 0UL);

#pragma omp parallel for private(j, thread_index)
	for(i = 0; i < num_blocks; i++)
	{
		thread_index = omp_get_thread_num();
		for(j = 0; j < v[i]->dim; j++)
		{
			mpf_abs(tmp[thread_index], get_mpfvector_i(v[i], j));
#pragma omp critical
			mpf_add(ret, ret, tmp[thread_index]);
		}
	}

#pragma omp parallel private(thread_index)
	{
		thread_index = omp_get_thread_num();
		mpf_clear(tmp[thread_index]);
	}

	return;
}

/* ret := vec / ||vec|| */
void _bncomp_normalize_mpfvector(mpf_t norm, MPFVector ret, MPFVector vec, int kind_of_norm)
{
	mpf_t tmp;

	mpf_init2(tmp, ret->prec);

	norm_mpfvector(norm, vec, kind_of_norm);

	// ret := vec / ||vec||
	mpf_ui_div(tmp, 1UL, norm);
	cmul_mpfvector(ret, tmp, vec);

	mpf_clear(tmp);

	return;
}

/* vec := vec / ||vec|| */
void _bncomp_normalize2_mpfvector(mpf_t norm, MPFVector vec, int kind_of_norm)
{
	mpf_t tmp;

	mpf_init2(tmp, vec->prec);

	norm_mpfvector(norm, vec, kind_of_norm);

	// ret := vec / ||vec||
	mpf_ui_div(tmp, 1UL, norm);
	cmul2_mpfvector(vec, tmp);

	mpf_clear(tmp);

	return;
}

/* ret := val * vec */
void _bncomp_cmul_mpfvector_blocks(MPFVector ret[], mpf_t val, MPFVector vec[], long int num_blocks)
{
	long int i;

#pragma omp parallel for
	for(i = 0; i < num_blocks; i++)
		cmul_mpfvector(ret[i], val, vec[i]);

	return;
}

/* vec := val * vec */
void _bncomp_cmul2_mpfvector_blocks(MPFVector vec[], mpf_t val, long int num_blocks)
{
	long int i;

#pragma omp parallel for
	for(i = 0; i < num_blocks; i++)
		cmul2_mpfvector(vec[i], val);

	return;
}

/* vector norm */
void _bncomp_norm_mpfvector_blocks(mpf_t norm, MPFVector vec[], long int num_blocks, int kind_of_norm)
{
	switch(kind_of_norm)
	{
		case BNC_INFINITY_NORM:
			_bncomp_normi_mpfvector_blocks(norm, vec, num_blocks);
			break;
		case BNC_ONE_NORM:
			_bncomp_norm1_mpfvector_blocks(norm, vec, num_blocks);
			break;
		case BNC_EUCLID_NORM:
		default:
			_bncomp_norm2_mpfvector_blocks(norm, vec, num_blocks);
			break;
	}

	return;
}

/* normalization: ret[] := vec[] / ||vec[]|| */
void _bncomp_normalize_mpfvector_blocks(mpf_t norm, MPFVector ret[], MPFVector vec[], long int num_blocks, int kind_of_norm)
{
	mpf_t tmp;

	mpf_init2(tmp, ret[0]->prec);

	_bncomp_norm_mpfvector_blocks(norm, vec, num_blocks, kind_of_norm);

	// ret := vec / ||vec||
	mpf_ui_div(tmp, 1UL, norm);
	_bncomp_cmul_mpfvector_blocks(ret, tmp, vec, num_blocks);

	mpf_clear(tmp);

	return;
}

/* normalization: vec[] /= ||vec[]|| */
void _bncomp_normalize2_mpfvector_blocks(mpf_t norm, MPFVector vec[], long int num_blocks, int kind_of_norm)
{
	mpf_t tmp;

	mpf_init2(tmp, vec[0]->prec);

	_bncomp_norm_mpfvector_blocks(norm, vec, num_blocks, kind_of_norm);
//	printf("norm = "); mpf_out_str(stdout, 10, 17, norm); printf("\n");

	// ret := vec / ||vec||
	mpf_ui_div(tmp, 1UL, norm);
	_bncomp_cmul2_mpfvector_blocks(vec, tmp, num_blocks);

	mpf_clear(tmp);

	return;
}
#endif // USE_GMP


